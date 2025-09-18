import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from src import feedback, store
from src.api import app

client = TestClient(app)
BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_feedback_state() -> None:
    """Ensure feedback parameters are reset between tests."""

    feedback.reset_state()
    yield
    feedback.reset_state()


def create_image_bytes(
    size: tuple[int, int] = (32, 16),
    color: str = "blue",
    add_pill: bool = False,
) -> bytes:
    image = Image.new("RGB", size, color=color)
    if add_pill:
        draw = ImageDraw.Draw(image)
        margin_w, margin_h = int(size[0] * 0.15), int(size[1] * 0.15)
        bbox = [margin_w, margin_h, size[0] - margin_w, size[1] - margin_h]
        draw.ellipse(bbox, fill="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_bag() -> None:
    response = client.post("/api/bags", json={"label": "morning"})
    assert response.status_code == 200
    data = response.json()
    assert data["id"]
    assert data["label"] == "morning"
    assert "created_at" in data


def test_upload_image_and_list() -> None:
    bag_response = client.post("/api/bags", json={"label": "images"})
    bag_response.raise_for_status()
    bag_id = bag_response.json()["id"]

    image_bytes = create_image_bytes()
    files = {"file": ("test.png", image_bytes, "image/png")}

    upload_response = client.post("/api/images", params={"bag_id": bag_id}, files=files)
    assert upload_response.status_code == 200
    image_data = upload_response.json()
    assert image_data["bag_id"] == bag_id
    assert image_data["width"] == 32
    assert image_data["height"] == 16

    stored_path = BASE_DIR / image_data["path"]
    assert stored_path.exists()

    bag_detail = client.get(f"/api/bags/{bag_id}")
    assert bag_detail.status_code == 200
    bag_data = bag_detail.json()
    assert bag_data["id"] == bag_id
    assert bag_data["label"] == "images"
    assert len(bag_data["images"]) == 1
    assert bag_data["images"][0]["id"] == image_data["id"]


def test_get_image_with_detections() -> None:
    bag_response = client.post("/api/bags", json={"label": "detect"})
    bag_response.raise_for_status()
    bag_id = bag_response.json()["id"]

    image_bytes = create_image_bytes(size=(200, 200), color="black", add_pill=True)
    files = {"file": ("pill.png", image_bytes, "image/png")}

    upload_response = client.post("/api/images", params={"bag_id": bag_id}, files=files)
    upload_response.raise_for_status()
    image_id = upload_response.json()["id"]

    response = client.get(f"/api/images/{image_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["image"]["id"] == image_id
    assert data["image"]["bag_id"] == bag_id
    assert len(data["detections"]) >= 1
    assert all(det["image_id"] == image_id for det in data["detections"])

    visualization_path = BASE_DIR / data["visualization_path"]
    assert visualization_path.exists()

    assert "unrecognized_regions" in data
    assert data["message"] is None or isinstance(data["message"], str)

    samples = store.list_samples_for_bag(bag_id)
    assert samples, "embedding samples should be stored"
    first_embed = samples[0]["embed"]
    if isinstance(first_embed, memoryview):
        first_embed = first_embed.tobytes()
    assert isinstance(first_embed, (bytes, bytearray))
    assert len(first_embed) == 4 * 512


def test_compare_endpoint_records_comparison() -> None:
    bag_a = client.post("/api/bags", json={"label": "A"})
    bag_b = client.post("/api/bags", json={"label": "B"})
    bag_a.raise_for_status()
    bag_b.raise_for_status()

    payload = {
        "bag_id_a": bag_a.json()["id"],
        "bag_id_b": bag_b.json()["id"],
        "score_color": 0.7,
        "score_shape": 0.6,
        "score_count": 0.5,
        "score_size": 0.4,
        "decision": "bag_a",
        "preview_path": "data/outputs/sample.png",
    }

    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    data = response.json()

    params = feedback.get_state()
    expected_total = pytest.approx(
        params.compute_total(
            sim_embed=0.0,
            sim_color=payload["score_color"],
            sim_count=payload["score_count"],
            sim_size=payload["score_size"],
            sim_text=0.0,
        ),
        rel=1e-3,
    )
    assert data["bag_id_a"] == payload["bag_id_a"]
    assert data["bag_id_b"] == payload["bag_id_b"]
    assert data["decision"] == "bag_a"
    assert data["preview_path"] == "data/outputs/sample.png"
    assert data["s_total"] == expected_total
    assert data["score_embed"] == pytest.approx(0.0, abs=1e-6)

    stored = store.get_comparison(data["id"])
    assert stored is not None
    assert stored["s_total"] == pytest.approx(data["s_total"], rel=1e-6)


def test_feedback_updates_parameters_and_scores() -> None:
    bag_a = client.post("/api/bags", json={"label": "fa"})
    bag_b = client.post("/api/bags", json={"label": "fb"})
    bag_a.raise_for_status()
    bag_b.raise_for_status()

    payload = {
        "bag_id_a": bag_a.json()["id"],
        "bag_id_b": bag_b.json()["id"],
        "score_color": 0.8,
        "score_shape": 0.7,
        "score_count": 0.6,
        "score_size": 0.5,
        "decision": "bag_a",
    }

    comparison = client.post("/api/compare", json=payload)
    comparison.raise_for_status()
    comp_data = comparison.json()

    params = feedback.get_state()
    initial_weights = params.weights.copy()
    initial_total = comp_data["s_total"]

    feedback_1 = client.post(
        "/api/feedback",
        json={"comparison_id": comp_data["id"], "is_correct": 1, "note": "ok"},
    )
    feedback_1.raise_for_status()
    data_1 = feedback_1.json()

    assert data_1["updated_s_total"] < initial_total
    assert data_1["parameters"]["weights"][0] > pytest.approx(initial_weights[0])
    assert data_1["parameters"]["tau"] < 0.82

    stored_after_first = store.get_comparison(comp_data["id"])
    assert stored_after_first is not None
    assert stored_after_first["s_total"] == pytest.approx(data_1["updated_s_total"], rel=1e-6)

    feedback_2 = client.post(
        "/api/feedback",
        json={"comparison_id": comp_data["id"], "is_correct": 1},
    )
    feedback_2.raise_for_status()
    data_2 = feedback_2.json()

    assert data_2["updated_s_total"] < data_1["updated_s_total"]
    assert data_2["parameters"]["weights"][0] > data_1["parameters"]["weights"][0]

    feedback_3 = client.post(
        "/api/feedback",
        json={"comparison_id": comp_data["id"], "is_correct": 0, "note": "needs review"},
    )
    feedback_3.raise_for_status()
    data_3 = feedback_3.json()

    assert data_3["updated_s_total"] > data_2["updated_s_total"]
    assert data_3["parameters"]["weights"][0] < data_2["parameters"]["weights"][0]

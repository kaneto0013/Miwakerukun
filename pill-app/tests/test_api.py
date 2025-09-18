import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from src.api import app

client = TestClient(app)
BASE_DIR = Path(__file__).resolve().parents[1]


def create_image_bytes(size: tuple[int, int] = (32, 16), color: str = "blue") -> bytes:
    image = Image.new("RGB", size, color=color)
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

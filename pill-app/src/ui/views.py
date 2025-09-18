from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageStat, UnidentifiedImageError

from .. import feedback, signature, store, visualize
from ..image_ops import (
    BagNotFoundError,
    EmptyUploadError,
    ImageNotFoundError,
    InvalidImageError,
    OUTPUT_DIR,
    load_image_with_detections,
    process_upload,
)

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
TEMPLATES = Jinja2Templates(directory=str(ROOT_DIR / "templates"))

router = APIRouter()

_STATUS_MESSAGES = {
    "bag_created": "袋を作成しました。",
    "uploaded": "画像をアップロードしました。",
    "upload_failed": "画像のアップロードに失敗しました。",
    "comparison_created": "比較を実行しました。",
}


def _build_redirect(request: Request, route_name: str, **params: object) -> str:
    base = request.url_for(route_name)
    if params:
        filtered = {key: value for key, value in params.items() if value is not None}
        query = urlencode(filtered, doseq=True)
        if query:
            return f"{base}?{query}"
    return str(base)


def _resolve_file_url(request: Request, relative_path: Optional[str]) -> Optional[str]:
    if not relative_path:
        return None
    candidate = BASE_DIR / relative_path
    if not candidate.exists():
        return None
    return request.url_for("files", path=relative_path)


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return None
    return sum(data) / len(data)


def _average_color(path: Path) -> Optional[tuple[float, float, float]]:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            stats = ImageStat.Stat(image)
            return tuple(stats.mean)
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None


def _average_color_for_bag(bag: Dict[str, Any]) -> Optional[tuple[float, float, float]]:
    for image in bag.get("images", []):
        color = _average_color(BASE_DIR / image["path"])
        if color is not None:
            return color
    return None


def _similarity_ratio(value_a: Optional[float], value_b: Optional[float], default: float = 0.5) -> float:
    if value_a is None or value_b is None:
        return default
    max_value = max(value_a, value_b)
    if max_value <= 0:
        return 1.0
    diff = abs(value_a - value_b) / max_value
    return max(0.0, 1.0 - diff)


def _color_similarity(color_a: Optional[tuple[float, float, float]], color_b: Optional[tuple[float, float, float]]) -> float:
    if color_a is None or color_b is None:
        return 0.5
    diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(color_a, color_b)))
    max_diff = 255.0 * math.sqrt(3)
    return max(0.0, 1.0 - diff / max_diff)


def _bag_statistics(bag: Dict[str, Any]) -> Dict[str, Optional[float]]:
    detection_counts: List[int] = []
    areas: List[float] = []
    aspect_ratios: List[float] = []
    for image in bag.get("images", []):
        detections = store.list_detections(image["id"])
        detection_counts.append(len(detections))
        for det in detections:
            width = max(0, int(det["x2"]) - int(det["x1"]))
            height = max(0, int(det["y2"]) - int(det["y1"]))
            if width <= 0 or height <= 0:
                continue
            area_ratio = (width * height) / float(max(1, image["width"] * image["height"]))
            aspect = max(width, height) / float(max(1, min(width, height)))
            areas.append(area_ratio)
            aspect_ratios.append(aspect)
    average_area = _safe_mean(areas)
    average_aspect = _safe_mean(aspect_ratios)
    total_detections = float(sum(detection_counts)) if detection_counts else 0.0
    return {
        "total_detections": total_detections,
        "average_area": average_area,
        "average_aspect": average_aspect,
        "average_color": _average_color_for_bag(bag),
    }


def _find_preview_path(bag: Dict[str, Any]) -> Optional[str]:
    for image in bag.get("images", []):
        visualization_path = visualize.visualization_path_for(image["id"], OUTPUT_DIR)
        if visualization_path.exists():
            return str(visualization_path.relative_to(BASE_DIR))
    for image in bag.get("images", []):
        candidate = BASE_DIR / image["path"]
        if candidate.exists():
            return image["path"]
    return None


@router.get("/", name="ui_upload")
def upload_page(request: Request, bag_id: Optional[str] = None) -> Any:
    bags = store.list_bags()
    bag_images: Dict[str, List[Dict[str, Any]]] = {}
    bag_cards: List[Dict[str, Any]] = []
    for bag in bags:
        images = store.list_bag_images(bag["id"])
        bag_images[bag["id"]] = images
        preview = images[0]["path"] if images else None
        bag_cards.append(
            {
                "id": bag["id"],
                "label": bag.get("label") or "(ラベル未設定)",
                "created_at": bag["created_at"],
                "image_count": len(images),
                "link": _build_redirect(request, "ui_upload", bag_id=bag["id"]),
                "preview_url": _resolve_file_url(request, preview),
            }
        )

    selected_id = bag_id or (bags[0]["id"] if bags else None)
    selected_bag: Optional[Dict[str, Any]] = None
    if selected_id:
        base_info = next((bag for bag in bags if bag["id"] == selected_id), None)
        if base_info is not None:
            selected_bag = dict(base_info)
            selected_bag["images"] = bag_images.get(selected_id, [])

    image_cards: List[Dict[str, Any]] = []
    if selected_bag:
        for image in selected_bag.get("images", []):
            visualization = visualize.visualization_path_for(image["id"], OUTPUT_DIR)
            image_cards.append(
                {
                    "id": image["id"],
                    "created_at": image["created_at"],
                    "detail_url": request.url_for("ui_image_detail", image_id=image["id"]),
                    "raw_url": _resolve_file_url(request, image["path"]),
                    "visualization_url": _resolve_file_url(
                        request,
                        str(visualization.relative_to(BASE_DIR)) if visualization.exists() else None,
                    ),
                }
            )

    status_key = request.query_params.get("status")
    errors = request.query_params.get("errors")
    toast_message = _STATUS_MESSAGES.get(status_key)
    if errors:
        toast_message = f"一部の画像を処理できませんでした: {errors}"

    context = {
        "request": request,
        "bag_cards": bag_cards,
        "selected_bag": selected_bag,
        "image_cards": image_cards,
        "toast_message": toast_message,
    }
    return TEMPLATES.TemplateResponse("upload.html", context)


@router.post("/bags", name="ui_create_bag")
async def create_bag_action(request: Request, label: str = Form(default="")) -> RedirectResponse:
    record = store.create_bag(label.strip() or None)
    url = _build_redirect(request, "ui_upload", bag_id=record["id"], status="bag_created")
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


@router.post("/bags/{bag_id}/upload", name="ui_upload_images")
async def upload_images_action(
    request: Request,
    bag_id: str,
    files: List[UploadFile] = File(default=[]),
) -> RedirectResponse:
    if not files:
        url = _build_redirect(request, "ui_upload", bag_id=bag_id, status="upload_failed")
        return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)

    failed: List[str] = []
    saved = 0
    for upload in files:
        if not upload.filename:
            continue
        try:
            await process_upload(bag_id, upload)
            saved += 1
        except EmptyUploadError:
            continue
        except InvalidImageError:
            failed.append(upload.filename)
        except BagNotFoundError:
            raise HTTPException(status_code=404, detail="袋が見つかりません")

    status_key = "uploaded" if saved else "upload_failed"
    params: Dict[str, object] = {"bag_id": bag_id, "status": status_key}
    if failed:
        params["errors"] = ",".join(failed)
    url = _build_redirect(request, "ui_upload", **params)
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/images/{image_id}", name="ui_image_detail")
def image_detail(request: Request, image_id: str) -> Any:
    try:
        data = load_image_with_detections(image_id)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail="画像が見つかりません") from exc

    bag = store.get_bag_basic(data.image.bag_id) or {"id": data.image.bag_id, "label": None}
    image_data = data.image.dict()
    context = {
        "request": request,
        "image": image_data,
        "detections": [det.dict() for det in data.detections],
        "unrecognized": [region.dict() for region in data.unrecognized_regions],
        "message": data.message,
        "bag": bag,
        "raw_url": _resolve_file_url(request, image_data["path"]),
        "visualization_url": _resolve_file_url(request, data.visualization_path),
    }
    return TEMPLATES.TemplateResponse("image_detail.html", context)


@router.get("/compare", name="ui_compare")
def compare_page(
    request: Request,
    comparison_id: Optional[str] = None,
    bag_a: Optional[str] = None,
    bag_b: Optional[str] = None,
    status: Optional[str] = None,
) -> Any:
    bags = store.list_bags()
    comparison = store.get_comparison(comparison_id) if comparison_id else None
    selected_a = bag_a or (comparison["bag_id_a"] if comparison else None)
    selected_b = bag_b or (comparison["bag_id_b"] if comparison else None)

    breakdown: Optional[List[Dict[str, Any]]] = None
    preview_url: Optional[str] = None
    feedback_entries: List[Dict[str, Any]] = []
    if comparison:
        breakdown = [
            {"label": "色", "value": float(comparison["score_color"])},
            {"label": "形状", "value": float(comparison["score_shape"])},
            {"label": "個数", "value": float(comparison["score_count"])},
            {"label": "サイズ", "value": float(comparison["score_size"])},
            {"label": "埋め込み", "value": float(comparison.get("score_embed") or 0.0)},
        ]
        preview_url = _resolve_file_url(request, comparison.get("preview_path"))
        feedback_entries = store.list_feedback(comparison_id)

    params = feedback.get_state()
    toast_message = _STATUS_MESSAGES.get(status)
    context = {
        "request": request,
        "bags": bags,
        "selected_a": selected_a,
        "selected_b": selected_b,
        "comparison": comparison,
        "breakdown": breakdown,
        "preview_url": preview_url,
        "feedback_entries": feedback_entries,
        "parameters": {"weights": params.weights, "tau": params.tau},
        "toast_message": toast_message if status else None,
    }
    return TEMPLATES.TemplateResponse("compare.html", context)


@router.post("/compare", name="ui_run_comparison")
def run_comparison(
    request: Request,
    bag_id_a: str = Form(...),
    bag_id_b: str = Form(...),
) -> RedirectResponse:
    bag_a = store.get_bag(bag_id_a)
    bag_b = store.get_bag(bag_id_b)
    if bag_a is None or bag_b is None:
        raise HTTPException(status_code=404, detail="比較対象の袋が見つかりません")

    stats_a = _bag_statistics(bag_a)
    stats_b = _bag_statistics(bag_b)

    score_color = _color_similarity(stats_a["average_color"], stats_b["average_color"])
    score_shape = _similarity_ratio(stats_a["average_aspect"], stats_b["average_aspect"])
    score_count = _similarity_ratio(stats_a["total_detections"], stats_b["total_detections"], default=0.5)
    score_size = _similarity_ratio(stats_a["average_area"], stats_b["average_area"])

    samples_a = store.list_samples_for_bag(bag_id_a)
    samples_b = store.list_samples_for_bag(bag_id_b)
    score_embed = signature.boe_similarity(samples_a, samples_b)

    params = feedback.get_state()
    s_total = params.compute_total(
        sim_embed=score_embed,
        sim_color=score_color,
        sim_count=score_count,
        sim_size=score_size,
        sim_text=0.0,
    )
    decision = "bag_a" if s_total >= params.tau else "bag_b"

    preview_path = _find_preview_path(bag_a) or _find_preview_path(bag_b)

    record = store.create_comparison(
        bag_id_a=bag_id_a,
        bag_id_b=bag_id_b,
        score_color=score_color,
        score_shape=score_shape,
        score_count=score_count,
        score_size=score_size,
        score_embed=score_embed,
        s_total=s_total,
        decision=decision,
        preview_path=preview_path,
    )

    url = _build_redirect(
        request,
        "ui_compare",
        comparison_id=record["id"],
        bag_a=bag_id_a,
        bag_b=bag_id_b,
        status="comparison_created",
    )
    return RedirectResponse(url, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/history", name="ui_history")
def history_page(request: Request) -> Any:
    comparisons = store.list_comparisons()
    bag_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    history_rows: List[Dict[str, Any]] = []

    def _first_image_path(bag_id: str) -> Optional[str]:
        bag = bag_cache.get(bag_id)
        if bag is None:
            bag_cache[bag_id] = store.get_bag(bag_id)
            bag = bag_cache[bag_id]
        if not bag:
            return None
        images = bag.get("images", [])
        for image in images:
            visualization = visualize.visualization_path_for(image["id"], OUTPUT_DIR)
            if visualization.exists():
                return str(visualization.relative_to(BASE_DIR))
        if images:
            first = BASE_DIR / images[0]["path"]
            if first.exists():
                return images[0]["path"]
        return None

    for comp in comparisons:
        preview_url = _resolve_file_url(request, comp.get("preview_path"))
        bag_a_label = comp.get("bag_a_label") or "(A)"
        bag_b_label = comp.get("bag_b_label") or "(B)"
        bag_a_image = _resolve_file_url(request, _first_image_path(comp["bag_id_a"]))
        bag_b_image = _resolve_file_url(request, _first_image_path(comp["bag_id_b"]))
        payload = {
            key: comp[key]
            for key in (
                "id",
                "bag_id_a",
                "bag_id_b",
                "score_color",
                "score_shape",
                "score_count",
                "score_size",
                "score_embed",
                "s_total",
                "decision",
                "preview_path",
                "created_at",
            )
        }
        history_rows.append(
            {
                "id": comp["id"],
                "created_at": comp["created_at"],
                "s_total": float(comp["s_total"]),
                "decision": comp["decision"],
                "preview_url": preview_url,
                "bag_a": {
                    "id": comp["bag_id_a"],
                    "label": bag_a_label,
                    "link": _build_redirect(request, "ui_upload", bag_id=comp["bag_id_a"]),
                    "image_url": bag_a_image,
                },
                "bag_b": {
                    "id": comp["bag_id_b"],
                    "label": bag_b_label,
                    "link": _build_redirect(request, "ui_upload", bag_id=comp["bag_id_b"]),
                    "image_url": bag_b_image,
                },
                "detail_url": _build_redirect(request, "ui_compare", comparison_id=comp["id"]),
                "json_payload": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        )

    context = {
        "request": request,
        "history": history_rows,
    }
    return TEMPLATES.TemplateResponse("history.html", context)


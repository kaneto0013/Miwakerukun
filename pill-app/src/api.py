from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.params import Query
from PIL import Image, UnidentifiedImageError

from . import detect, schemas, store, visualize

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="pill-app")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/bags", response_model=schemas.Bag)
def create_bag(payload: Optional[schemas.BagCreate] = Body(default=None)) -> schemas.Bag:
    label = payload.label if payload else None
    bag = store.create_bag(label)
    return schemas.Bag(**bag)


@app.post("/api/images", response_model=schemas.Image)
async def upload_image(
    bag_id: str = Query(..., description="Identifier of the bag that owns the image"),
    file: UploadFile = File(...),
) -> schemas.Image:
    bag = store.get_bag_basic(bag_id)
    if bag is None:
        raise HTTPException(status_code=404, detail="Bag not found")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        pil_image = Image.open(BytesIO(contents))
        pil_image.load()
        width, height = pil_image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    suffix = Path(file.filename or "").suffix
    if not suffix:
        suffix = ".png"
    filename = f"{uuid.uuid4()}{suffix}"
    file_path = RAW_DIR / filename
    with open(file_path, "wb") as output:
        output.write(contents)

    relative_path = str(file_path.relative_to(BASE_DIR))
    record = store.create_image(bag_id=bag_id, path=relative_path, width=width, height=height)

    try:
        detections, image_size = detect.detect_pills(file_path)
    except Exception:
        detections = []
        image_size = (height, width)

    unrecognized_regions, _ = detect.identify_unrecognized_regions(detections, image_size)
    store.replace_detections(record["id"], detections)

    visualization_path = visualize.visualization_path_for(record["id"], OUTPUT_DIR)
    try:
        visualize.visualize_detections(file_path, detections, unrecognized_regions, visualization_path)
    except FileNotFoundError:
        pass

    return schemas.Image(**record)


@app.get("/api/bags/{bag_id}", response_model=schemas.BagWithImages)
def get_bag(bag_id: str) -> schemas.BagWithImages:
    bag = store.get_bag(bag_id)
    if bag is None:
        raise HTTPException(status_code=404, detail="Bag not found")
    return schemas.BagWithImages(**bag)


@app.get("/api/images/{image_id}", response_model=schemas.ImageDetections)
def get_image_with_detections(image_id: str) -> schemas.ImageDetections:
    image = store.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    detections = store.list_detections(image_id)
    image_path = BASE_DIR / image["path"]
    image_size = (int(image["height"]), int(image["width"]))

    unrecognized_regions, message = detect.identify_unrecognized_regions(detections, image_size)

    visualization_path = visualize.visualization_path_for(image_id, OUTPUT_DIR)
    if not visualization_path.exists() and image_path.exists():
        try:
            visualize.visualize_detections(image_path, detections, unrecognized_regions, visualization_path)
        except FileNotFoundError:
            pass

    relative_visualization_path = str(visualization_path.relative_to(BASE_DIR))

    return schemas.ImageDetections(
        image=schemas.Image(**image),
        detections=[schemas.Detection(**det) for det in detections],
        visualization_path=relative_visualization_path,
        unrecognized_regions=[schemas.RegionFlag(**region) for region in unrecognized_regions],
        message=message,
    )

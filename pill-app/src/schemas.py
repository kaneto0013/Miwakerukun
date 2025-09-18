from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BagCreate(BaseModel):
    label: Optional[str] = None


class Image(BaseModel):
    id: str
    bag_id: str
    path: str
    width: int
    height: int
    created_at: str


class Detection(BaseModel):
    id: Optional[int] = None
    image_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class RegionFlag(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    reason: str


class Bag(BaseModel):
    id: str
    label: Optional[str] = None
    created_at: str


class BagWithImages(Bag):
    images: List[Image] = Field(default_factory=list)


class ImageDetections(BaseModel):
    image: Image
    detections: List[Detection] = Field(default_factory=list)
    visualization_path: str
    unrecognized_regions: List[RegionFlag] = Field(default_factory=list)
    message: Optional[str] = None

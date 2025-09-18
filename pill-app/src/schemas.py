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


class ParameterSnapshot(BaseModel):
    weights: List[float] = Field(default_factory=list)
    tau: float
    w: float


class ComparisonCreate(BaseModel):
    bag_id_a: str
    bag_id_b: str
    score_color: float
    score_shape: float
    score_count: float
    score_size: float
    decision: str
    preview_path: Optional[str] = None


class Comparison(BaseModel):
    id: str
    bag_id_a: str
    bag_id_b: str
    score_color: float
    score_shape: float
    score_count: float
    score_size: float
    score_embed: float = 0.0
    score_ocr: float = 0.0
    s_total: float
    decision: str
    preview_path: Optional[str] = None
    created_at: str


class FeedbackCreate(BaseModel):
    comparison_id: str
    is_correct: int = Field(ge=0, le=1)
    note: Optional[str] = None


class Feedback(BaseModel):
    id: int
    comparison_id: str
    is_correct: int
    note: Optional[str] = None
    created_at: str
    updated_s_total: float
    parameters: ParameterSnapshot

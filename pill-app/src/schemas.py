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


class Bag(BaseModel):
    id: str
    label: Optional[str] = None
    created_at: str


class BagWithImages(Bag):
    images: List[Image] = Field(default_factory=list)

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "app.sqlite"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with sensible defaults."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    """Ensure the database schema exists."""
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bags (
                id TEXT PRIMARY KEY,
                label TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                bag_id TEXT NOT NULL,
                path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (bag_id) REFERENCES bags(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                x1 INTEGER NOT NULL,
                y1 INTEGER NOT NULL,
                x2 INTEGER NOT NULL,
                y2 INTEGER NOT NULL,
                score REAL NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()


def create_bag(label: Optional[str]) -> Dict[str, object]:
    """Create a new bag entry and return its data."""
    bag_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO bags (id, label, created_at) VALUES (?, ?, ?)",
            (bag_id, label, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": bag_id, "label": label, "created_at": now}


def get_bag_basic(bag_id: str) -> Optional[Dict[str, object]]:
    """Fetch a bag without loading its images."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, label, created_at FROM bags WHERE id = ?",
            (bag_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def list_bag_images(bag_id: str) -> List[Dict[str, object]]:
    """List images belonging to a bag."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, bag_id, path, width, height, created_at
            FROM images
            WHERE bag_id = ?
            ORDER BY created_at ASC
            """,
            (bag_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_bag(bag_id: str) -> Optional[Dict[str, object]]:
    """Fetch a bag with its associated images."""
    bag = get_bag_basic(bag_id)
    if bag is None:
        return None
    bag["images"] = list_bag_images(bag_id)
    return bag


def get_image(image_id: str) -> Optional[Dict[str, object]]:
    """Fetch an image record by its identifier."""

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, bag_id, path, width, height, created_at
            FROM images
            WHERE id = ?
            """,
            (image_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def create_image(bag_id: str, path: str, width: int, height: int) -> Dict[str, object]:
    """Create an image record linked to a bag."""
    image_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO images (id, bag_id, path, width, height, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_id, bag_id, path, width, height, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": image_id,
        "bag_id": bag_id,
        "path": path,
        "width": width,
        "height": height,
        "created_at": now,
    }


def replace_detections(image_id: str, detections: Iterable[Dict[str, object]]) -> None:
    """Replace detections for an image with the provided records."""

    conn = get_connection()
    try:
        conn.execute("DELETE FROM detections WHERE image_id = ?", (image_id,))
        conn.executemany(
            """
            INSERT INTO detections (image_id, x1, y1, x2, y2, score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    image_id,
                    int(det["x1"]),
                    int(det["y1"]),
                    int(det["x2"]),
                    int(det["y2"]),
                    float(det["score"]),
                )
                for det in detections
            ],
        )
        conn.commit()
    finally:
        conn.close()


def list_detections(image_id: str) -> List[Dict[str, object]]:
    """List detections stored for an image."""

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, image_id, x1, y1, x2, y2, score
            FROM detections
            WHERE image_id = ?
            ORDER BY score DESC, id ASC
            """,
            (image_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

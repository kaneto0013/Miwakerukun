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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY,
                detection_id INTEGER NOT NULL,
                embed BLOB NOT NULL,
                size_px REAL NOT NULL,
                ocr_text TEXT,
                FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS comparisons (
                id TEXT PRIMARY KEY,
                bag_id_a TEXT NOT NULL,
                bag_id_b TEXT NOT NULL,
                score_color REAL NOT NULL,
                score_shape REAL NOT NULL,
                score_count REAL NOT NULL,
                score_size REAL NOT NULL,
                s_total REAL NOT NULL,
                decision TEXT NOT NULL,
                preview_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (bag_id_a) REFERENCES bags(id) ON DELETE CASCADE,
                FOREIGN KEY (bag_id_b) REFERENCES bags(id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_id TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (comparison_id) REFERENCES comparisons(id) ON DELETE CASCADE
            );
            """
        )
        cursor = conn.execute("PRAGMA table_info(comparisons)")
        column_names = {row["name"] for row in cursor.fetchall()}
        if "score_embed" not in column_names:
            conn.execute("ALTER TABLE comparisons ADD COLUMN score_embed REAL DEFAULT 0.0")
            conn.execute("UPDATE comparisons SET score_embed = 0.0 WHERE score_embed IS NULL")
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


def list_bags(limit: Optional[int] = None) -> List[Dict[str, object]]:
    """Return bags ordered by creation time (newest first)."""

    conn = get_connection()
    try:
        query = "SELECT id, label, created_at FROM bags ORDER BY created_at DESC"
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


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


def replace_detections(image_id: str, detections: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Replace detections for an image with the provided records.

    The newly stored detections are returned, including their database identifiers.
    """

    conn = get_connection()
    stored: List[Dict[str, object]] = []
    try:
        conn.execute("DELETE FROM detections WHERE image_id = ?", (image_id,))
        for det in detections:
            cursor = conn.execute(
                """
                INSERT INTO detections (image_id, x1, y1, x2, y2, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    int(det["x1"]),
                    int(det["y1"]),
                    int(det["x2"]),
                    int(det["y2"]),
                    float(det["score"]),
                ),
            )
            detection_id = int(cursor.lastrowid)
            stored.append(
                {
                    "id": detection_id,
                    "image_id": image_id,
                    "x1": int(det["x1"]),
                    "y1": int(det["y1"]),
                    "x2": int(det["x2"]),
                    "y2": int(det["y2"]),
                    "score": float(det["score"]),
                }
            )
        conn.commit()
    finally:
        conn.close()
    return stored


def insert_samples(samples: Iterable[Dict[str, object]]) -> None:
    """Insert embedding samples for detections."""

    items = list(samples)
    if not items:
        return

    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO samples (id, detection_id, embed, size_px, ocr_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    sample["id"],
                    int(sample["detection_id"]),
                    sqlite3.Binary(sample["embed"]),
                    float(sample.get("size_px", 0.0)),
                    sample.get("ocr_text"),
                )
                for sample in items
            ],
        )
        conn.commit()
    finally:
        conn.close()


def list_samples_for_bag(bag_id: str) -> List[Dict[str, object]]:
    """Return embedding samples associated with all detections in a bag."""

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT
                samples.id,
                samples.detection_id,
                samples.embed,
                samples.size_px,
                samples.ocr_text,
                detections.image_id
            FROM samples
            JOIN detections ON detections.id = samples.detection_id
            JOIN images ON images.id = detections.image_id
            WHERE images.bag_id = ?
            ORDER BY samples.id ASC
            """,
            (bag_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
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


def create_comparison(
    *,
    bag_id_a: str,
    bag_id_b: str,
    score_color: float,
    score_shape: float,
    score_count: float,
    score_size: float,
    score_embed: float,
    s_total: float,
    decision: str,
    preview_path: Optional[str] = None,
    comparison_id: Optional[str] = None,
) -> Dict[str, object]:
    """Store a comparison result and return the created record."""

    record_id = comparison_id or str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO comparisons (
                id,
                bag_id_a,
                bag_id_b,
                score_color,
                score_shape,
                score_count,
                score_size,
                score_embed,
                s_total,
                decision,
                preview_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                bag_id_a,
                bag_id_b,
                float(score_color),
                float(score_shape),
                float(score_count),
                float(score_size),
                float(score_embed),
                float(s_total),
                decision,
                preview_path,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "id": record_id,
        "bag_id_a": bag_id_a,
        "bag_id_b": bag_id_b,
        "score_color": float(score_color),
        "score_shape": float(score_shape),
        "score_count": float(score_count),
        "score_size": float(score_size),
        "score_embed": float(score_embed),
        "s_total": float(s_total),
        "decision": decision,
        "preview_path": preview_path,
        "created_at": now,
    }


def get_comparison(comparison_id: str) -> Optional[Dict[str, object]]:
    """Fetch a stored comparison by its identifier."""

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT
                id,
                bag_id_a,
                bag_id_b,
                score_color,
                score_shape,
                score_count,
                score_size,
                score_embed,
                s_total,
                decision,
                preview_path,
                created_at
            FROM comparisons
            WHERE id = ?
            """,
            (comparison_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def list_comparisons(limit: Optional[int] = None) -> List[Dict[str, object]]:
    """List stored comparisons with associated bag labels."""

    conn = get_connection()
    try:
        query = """
            SELECT
                comparisons.id,
                comparisons.bag_id_a,
                comparisons.bag_id_b,
                comparisons.score_color,
                comparisons.score_shape,
                comparisons.score_count,
                comparisons.score_size,
                comparisons.score_embed,
                comparisons.s_total,
                comparisons.decision,
                comparisons.preview_path,
                comparisons.created_at,
                bag_a.label AS bag_a_label,
                bag_b.label AS bag_b_label
            FROM comparisons
            LEFT JOIN bags AS bag_a ON bag_a.id = comparisons.bag_id_a
            LEFT JOIN bags AS bag_b ON bag_b.id = comparisons.bag_id_b
            ORDER BY comparisons.created_at DESC, comparisons.id DESC
        """
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_comparison_total(comparison_id: str, s_total: float) -> None:
    """Persist an updated total score for a comparison."""

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE comparisons SET s_total = ? WHERE id = ?",
            (float(s_total), comparison_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_feedback(comparison_id: str) -> List[Dict[str, object]]:
    """Return feedback entries for a comparison ordered by creation time."""

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, comparison_id, is_correct, note, created_at
            FROM feedback
            WHERE comparison_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (comparison_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def create_feedback(
    *, comparison_id: str, is_correct: int, note: Optional[str] = None
) -> Dict[str, object]:
    """Insert a feedback entry linked to a comparison."""

    now = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO feedback (comparison_id, is_correct, note, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (comparison_id, int(is_correct), note, now),
        )
        feedback_id = cursor.lastrowid
        conn.commit()
    finally:
        conn.close()

    return {
        "id": int(feedback_id),
        "comparison_id": comparison_id,
        "is_correct": int(is_correct),
        "note": note,
        "created_at": now,
    }

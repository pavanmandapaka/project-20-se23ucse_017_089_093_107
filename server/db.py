import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "evaluation_logs.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            image_path TEXT NOT NULL,
            ground_truth TEXT,
            generated_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inference_id INTEGER NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            details TEXT,
            FOREIGN KEY (inference_id) REFERENCES inferences (id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()
    conn.close()


def log_inference(
    model_version: str,
    image_path: str,
    generated_text: str,
    ground_truth: Optional[str] = None,
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO inferences (model_version, image_path, ground_truth, generated_text, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (model_version, image_path, ground_truth, generated_text, datetime.now()),
    )
    inference_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inference_id


def get_inference(inference_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inferences WHERE id = ?", (inference_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_inferences(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM inferences ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_metric(
    inference_id: int,
    metric_name: str,
    metric_value: float,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    details_str = json.dumps(details) if details else None
    cursor.execute(
        """
        INSERT INTO metrics (inference_id, metric_name, metric_value, details)
        VALUES (?, ?, ?, ?)
        """,
        (inference_id, metric_name, metric_value, details_str),
    )
    conn.commit()
    conn.close()


def update_inference_text(inference_id: int, generated_text: str) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE inferences SET generated_text = ? WHERE id = ?",
        (generated_text, inference_id),
    )
    conn.commit()
    conn.close()


init_db()

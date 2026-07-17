"""
Diagnosis history log — SQLite, zero setup, zero cost. Stores every
diagnosis so the farmer (or the UI) can look back at past results,
and lays the groundwork for Step 6's progress-tracking feature
(comparing repeat photos of the same plant over time).
"""
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

DB_PATH = "storage/history.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@dataclass
class HistoryEntry:
    id: int
    timestamp: str
    image_path: str
    diagnosis_class: Optional[str]
    confidence: Optional[float]
    action: str
    advice_text: Optional[str]
    language: str


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            diagnosis_class TEXT,
            confidence REAL,
            action TEXT NOT NULL,
            advice_text TEXT,
            language TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_entry(
    image_path: str,
    diagnosis_class: Optional[str],
    confidence: Optional[float],
    action: str,
    advice_text: Optional[str],
    language: str,
) -> int:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """INSERT INTO diagnoses
           (timestamp, image_path, diagnosis_class, confidence, action, advice_text, language)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (datetime.utcnow().isoformat(), image_path, diagnosis_class, confidence, action, advice_text, language),
    )
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()
    return entry_id


def get_all_entries(limit: int = 50) -> list[HistoryEntry]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM diagnoses ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [HistoryEntry(**dict(row)) for row in rows]


def get_recent_by_class(diagnosis_class: str, limit: int = 5) -> list[HistoryEntry]:
    """Used later (Step 6) for lightweight 'others nearby also saw this' framing."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM diagnoses WHERE diagnosis_class = ? ORDER BY timestamp DESC LIMIT ?",
        (diagnosis_class, limit),
    ).fetchall()
    conn.close()
    return [HistoryEntry(**dict(row)) for row in rows]

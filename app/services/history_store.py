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


@dataclass
class ProgressInfo:
    previous_entry: HistoryEntry
    days_since: int
    confidence_delta: Optional[float]  # positive = confidence went up since last time


def get_progress_info(diagnosis_class: str, current_confidence: Optional[float]) -> Optional[ProgressInfo]:
    """
    Finds the most recent PRIOR diagnosis of the same class, so the app can
    tell the farmer "you saw this on your plant before" instead of treating
    every visit as a first-time case. Call this BEFORE save_entry() for the
    current diagnosis, so the lookup naturally excludes the current visit.
    Returns None if this is the first time this class has been logged.

    Note: on hosts without persistent disk (e.g. Render's free tier), this
    history resets whenever the container restarts — it tracks progress
    within a running instance's lifetime, not permanently across redeploys.
    """
    previous = next(iter(get_recent_by_class(diagnosis_class, limit=1)), None)
    if previous is None:
        return None

    previous_time = datetime.fromisoformat(previous.timestamp)
    days_since = (datetime.utcnow() - previous_time).days

    confidence_delta = None
    if current_confidence is not None and previous.confidence is not None:
        confidence_delta = current_confidence - previous.confidence

    return ProgressInfo(previous_entry=previous, days_since=days_since, confidence_delta=confidence_delta)


def format_progress_note(progress: ProgressInfo) -> str:
    when = "today" if progress.days_since == 0 else (
        "yesterday" if progress.days_since == 1 else f"{progress.days_since} days ago"
    )
    note = f"This same issue was last diagnosed {when}"
    if progress.confidence_delta is not None:
        if progress.confidence_delta > 0.05:
            note += ", and confidence is higher this time — may be spreading or more visible now."
        elif progress.confidence_delta < -0.05:
            note += ", and confidence is lower this time — may be improving or less visible now."
        else:
            note += ", looking about the same as before."
    else:
        note += "."
    return note

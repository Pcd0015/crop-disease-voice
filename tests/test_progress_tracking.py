"""
Tests progress-tracking logic (comparing today's diagnosis to the last
time the same disease class was logged). Uses a temporary SQLite file so
it doesn't touch your real storage/history.db.

Run with: pytest tests/test_progress_tracking.py -v
"""
import os
import tempfile

import pytest


@pytest.fixture
def history_store(monkeypatch):
    """Reloads history_store pointed at a fresh temp DB for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let init_db create it fresh

    import app.services.history_store as hs
    monkeypatch.setattr(hs, "DB_PATH", path)
    yield hs

    if os.path.exists(path):
        os.remove(path)


def test_no_progress_info_on_first_ever_diagnosis(history_store):
    result = history_store.get_progress_info("Tomato___Late_blight", current_confidence=0.85)
    assert result is None


def test_progress_info_found_after_repeat_diagnosis(history_store):
    history_store.save_entry(
        image_path="storage/uploads/day1.jpg",
        diagnosis_class="Tomato___Late_blight",
        confidence=0.70,
        action="give_advice",
        advice_text="...",
        language="English",
    )

    result = history_store.get_progress_info("Tomato___Late_blight", current_confidence=0.85)

    assert result is not None
    assert result.previous_entry.confidence == 0.70
    assert result.confidence_delta == pytest.approx(0.15)
    assert result.days_since == 0  # saved moments ago in this test


def test_progress_info_ignores_different_disease_class(history_store):
    history_store.save_entry(
        image_path="storage/uploads/day1.jpg",
        diagnosis_class="Tomato___Early_blight",
        confidence=0.70,
        action="give_advice",
        advice_text="...",
        language="English",
    )

    result = history_store.get_progress_info("Tomato___Late_blight", current_confidence=0.85)

    assert result is None


def test_format_progress_note_mentions_improvement_direction(history_store):
    history_store.save_entry(
        image_path="storage/uploads/day1.jpg",
        diagnosis_class="Tomato___Late_blight",
        confidence=0.90,
        action="give_advice",
        advice_text="...",
        language="English",
    )
    progress = history_store.get_progress_info("Tomato___Late_blight", current_confidence=0.60)
    note = history_store.format_progress_note(progress)

    assert "improving" in note or "less visible" in note

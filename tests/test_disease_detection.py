"""
Quick sanity test for the confidence-routing logic. Doesn't need the
actual trained model — tests response_router.py in isolation using a
fake DiagnosisResult, so you can run this immediately without waiting
on training.

Run with: pytest tests/test_disease_detection.py -v
"""
from app.services.disease_detection import DiagnosisResult
from app.services.response_router import route


def _fake_diagnosis(confidence: float, tier: str) -> DiagnosisResult:
    return DiagnosisResult(
        predicted_class="Tomato___Late_blight",
        confidence=confidence,
        tier=tier,
        top_3=[("Tomato___Late_blight", confidence), ("Tomato___Early_blight", 0.1), ("Tomato___healthy", 0.05)],
    )


def test_high_confidence_gives_advice():
    result = route(_fake_diagnosis(0.92, "high"))
    assert result.action == "give_advice"
    assert "Late blight" in result.message


def test_medium_confidence_requests_better_photo():
    result = route(_fake_diagnosis(0.70, "medium"))
    assert result.action == "request_better_photo"
    assert "another photo" in result.message


def test_low_confidence_flags_for_expert():
    result = route(_fake_diagnosis(0.35, "low"))
    assert result.action == "flag_for_expert"
    assert "extension office" in result.message

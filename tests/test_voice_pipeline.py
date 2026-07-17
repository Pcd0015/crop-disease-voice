"""
Tests the voice pipeline orchestration logic WITHOUT needing your trained
model, a live Gemini API key, or internet access — everything's mocked.
This lets you verify the wiring is correct while your model is still
training on Colab.

Run with: pytest tests/test_voice_pipeline.py -v
"""
from unittest.mock import patch, MagicMock

from app.services.disease_detection import DiagnosisResult
from app.services.advice_generation import AdviceResult
from app.services.voice_pipeline import process_photo_and_voice


def _fake_high_confidence_diagnosis():
    return DiagnosisResult(
        predicted_class="Tomato___Late_blight",
        confidence=0.91,
        tier="high",
        top_3=[("Tomato___Late_blight", 0.91)],
    )


def _fake_low_confidence_diagnosis():
    return DiagnosisResult(
        predicted_class="Tomato___Late_blight",
        confidence=0.40,
        tier="low",
        top_3=[("Tomato___Late_blight", 0.40)],
    )


@patch("app.services.voice_pipeline.synthesize_speech")
@patch("app.services.voice_pipeline.generate_advice")
@patch("app.services.voice_pipeline.diagnose_image")
def test_high_confidence_flow_generates_advice_and_audio(mock_diagnose, mock_advice, mock_tts):
    mock_diagnose.return_value = _fake_high_confidence_diagnosis()
    mock_advice.return_value = AdviceResult(text="This is late blight. Here is what to do...", language="English")
    mock_tts.return_value = "storage/audio_responses/fake.mp3"

    result = process_photo_and_voice(image_path="fake_leaf.jpg")

    assert result.action == "give_advice"
    assert result.diagnosis_class == "Tomato___Late_blight"
    assert result.session is not None
    assert result.session.advice_given == "This is late blight. Here is what to do..."
    mock_advice.assert_called_once()
    mock_tts.assert_called_once()


@patch("app.services.voice_pipeline.synthesize_speech")
@patch("app.services.voice_pipeline.generate_advice")
@patch("app.services.voice_pipeline.diagnose_image")
def test_low_confidence_flow_skips_advice_generation(mock_diagnose, mock_advice, mock_tts):
    mock_diagnose.return_value = _fake_low_confidence_diagnosis()
    mock_tts.return_value = "storage/audio_responses/fake.mp3"

    result = process_photo_and_voice(image_path="fake_leaf.jpg")

    assert result.action == "flag_for_expert"
    assert result.diagnosis_class is None
    assert result.session is None
    mock_advice.assert_not_called()  # should never call Gemini for a low-confidence result
    mock_tts.assert_called_once()

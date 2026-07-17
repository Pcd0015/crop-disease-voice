"""
Speech-to-text using faster-whisper — runs Whisper locally, fully free,
no API key or per-call cost. Handles multiple languages automatically
(Whisper auto-detects the spoken language, which matters a lot here
since farmers may speak in Hindi, Marathi, Telugu, etc.).

Model size tradeoff:
  "tiny"  — fastest, least accurate, fine for quick local testing
  "base"  — good default balance for CPU-only free hosting
  "small" — noticeably better accuracy, still runs OK on CPU
  "medium"/"large" — best accuracy but too slow/heavy for free CPU hosting
"""
from dataclasses import dataclass

from faster_whisper import WhisperModel

_model = None
MODEL_SIZE = "base"  # swap to "small" if hosting CPU can handle it


@dataclass
class TranscriptionResult:
    text: str
    detected_language: str
    language_confidence: float


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # compute_type="int8" — quantized, much faster/lighter on CPU-only
        # free hosting (Hugging Face Spaces free tier has no GPU)
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> TranscriptionResult:
    model = _get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)

    full_text = " ".join(segment.text.strip() for segment in segments)

    return TranscriptionResult(
        text=full_text.strip(),
        detected_language=info.language,
        language_confidence=info.language_probability,
    )

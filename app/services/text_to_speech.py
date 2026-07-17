"""
Text-to-speech using gTTS (Google Text-to-Speech) — free, no API key
needed, supports many languages including Hindi, Marathi, Telugu, Tamil,
Bengali, and others relevant here. Requires internet access (it's a free
web service, not a paid API) — for a fully offline alternative, Coqui TTS
is the swap-in option, noted below.
"""
import os
import uuid

from gtts import gTTS

# gTTS language codes for common Indian languages — extend as needed.
# Full list: https://gtts.readthedocs.io/en/latest/module.html#languages-gtts-lang
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Kannada": "kn",
    "Gujarati": "gu",
    "Punjabi": "pa",
}

OUTPUT_DIR = "storage/audio_responses"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def synthesize_speech(text: str, language: str = "English") -> str:
    """Returns the file path of the generated MP3."""
    lang_code = LANGUAGE_CODES.get(language, "en")

    tts = gTTS(text=text, lang=lang_code, slow=False)
    filename = f"{uuid.uuid4()}.mp3"
    output_path = os.path.join(OUTPUT_DIR, filename)
    tts.save(output_path)

    return output_path


# ---------------------------------------------------------------------------
# Offline alternative (no internet dependency) — swap in if you need this
# to work without connectivity. Not wired up by default since Coqui TTS is
# a heavier dependency; uncomment/install `TTS` package if you want this.
# ---------------------------------------------------------------------------
# from TTS.api import TTS as CoquiTTS
# _coqui_model = None
#
# def synthesize_speech_offline(text: str, output_path: str = None) -> str:
#     global _coqui_model
#     if _coqui_model is None:
#         _coqui_model = CoquiTTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
#     output_path = output_path or os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.wav")
#     _coqui_model.tts_to_file(text=text, file_path=output_path)
#     return output_path

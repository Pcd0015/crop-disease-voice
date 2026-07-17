"""
Orchestrates the full voice-in -> diagnosis -> advice -> voice-out flow.
This is what Step 3's Streamlit UI will actually call.

Flow:
  1. Farmer uploads a photo + optionally records a voice question
  2. (If voice given) transcribe it — mostly for logging/context; the
     photo drives the diagnosis regardless of what they said
  3. Run disease detection -> confidence-tier routing
  4. If high/medium confidence -> generate spoken advice via Gemini
     If low confidence -> speak the "please retake photo" message instead
  5. Synthesize the response as audio
  6. Return everything needed for the UI to display + play back
"""
from dataclasses import dataclass
from typing import Optional

from app.services.disease_detection import diagnose_image
from app.services.response_router import route
from app.services.advice_generation import generate_advice, ConversationSession
from app.services.text_to_speech import synthesize_speech
from app.services.speech_to_text import transcribe


@dataclass
class VoiceSessionResult:
    diagnosis_class: Optional[str]
    confidence: Optional[float]
    action: str                    # give_advice | request_better_photo | flag_for_expert
    spoken_text: str
    audio_path: str
    session: Optional[ConversationSession]  # None unless action == give_advice


def process_photo_and_voice(
    image_path: str,
    audio_question_path: Optional[str] = None,
    language: str = "English",
) -> VoiceSessionResult:

    # Optional: transcribe farmer's spoken question (mainly for logging/UI
    # display right now — full grounded use of it comes in Step 4's
    # follow-up conversation flow)
    if audio_question_path:
        transcription = transcribe(audio_question_path)
        # language auto-detection could override the passed-in `language`
        # here if you want the system to infer it rather than requiring
        # the farmer to pre-select — left as an explicit choice for now.

    diagnosis = diagnose_image(image_path)
    routed = route(diagnosis)

    if routed.action == "give_advice":
        advice = generate_advice(
            diagnosis=routed.diagnosis.predicted_class,
            confidence=routed.diagnosis.confidence,
            language=language,
        )
        spoken_text = advice.text
        session = ConversationSession(
            diagnosis=routed.diagnosis.predicted_class,
            confidence=routed.diagnosis.confidence,
            language=language,
            advice_given=advice.text,
        )
    else:
        # request_better_photo / flag_for_expert messages are already
        # farmer-facing plain text from response_router.py — speak them as-is
        spoken_text = routed.message
        session = None

    audio_path = synthesize_speech(spoken_text, language=language)

    return VoiceSessionResult(
        diagnosis_class=routed.diagnosis.predicted_class if routed.action == "give_advice" else None,
        confidence=routed.diagnosis.confidence,
        action=routed.action,
        spoken_text=spoken_text,
        audio_path=audio_path,
        session=session,
    )

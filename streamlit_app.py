"""
Streamlit UI for the crop disease voice assistant. Ties together:
  - Photo upload -> disease detection -> confidence-tier routing
  - Gemini-generated spoken advice (multilingual)
  - Voice question input (optional) via speech-to-text
  - Text-to-speech playback of the response
  - Grounded follow-up conversation
  - Diagnosis history log

DEMO MODE: since the CV model may still be training, there's a toggle to
run the UI against a fake diagnosis instead of the real model — lets you
build/test the full UI experience before training finishes.

Run with: streamlit run streamlit_app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from app.services.response_router import route
from app.services.disease_detection import DiagnosisResult
from app.services.advice_generation import generate_advice, ConversationSession, answer_followup
from app.services.text_to_speech import synthesize_speech, LANGUAGE_CODES
from app.services.speech_to_text import transcribe
from app.services import history_store
from app.services.weather import get_weather

st.set_page_config(page_title="Crop Disease Voice Assistant", page_icon="🌾", layout="centered")

MODEL_FILES_PRESENT = (
    os.path.exists("app/models/crop_disease_model.tflite")
    and os.path.exists("app/models/class_names.json")
)

# ---------------------------------------------------------------------------
# Sidebar: language selector + diagnosis history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Language / भाषा", list(LANGUAGE_CODES.keys()), index=0)

    demo_mode = st.checkbox(
        "Demo mode (skip real model)",
        value=not MODEL_FILES_PRESENT,
        help="Use this while your model is still training on Colab — runs the "
             "full UI flow against a fake diagnosis so you can test everything else.",
    )
    if not MODEL_FILES_PRESENT and not demo_mode:
        st.warning(
            "No trained model found in app/models/. Enable Demo mode above, "
            "or add crop_disease_model.tflite + class_names.json once training finishes."
        )

    st.divider()
    city = st.text_input(
        "Your city (optional)",
        help="Used to check current humidity/temperature for a fungal-disease "
             "risk note alongside your diagnosis. Leave blank to skip.",
    )

    st.divider()
    st.header("Diagnosis History")
    entries = history_store.get_all_entries(limit=10)
    if not entries:
        st.caption("No diagnoses yet.")
    for entry in entries:
        label = entry.diagnosis_class or entry.action.replace("_", " ")
        conf = f" ({entry.confidence:.0%})" if entry.confidence else ""
        st.caption(f"{entry.timestamp[:16].replace('T', ' ')} — {label}{conf}")

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
st.title("🌾 Crop Disease Voice Assistant")
st.caption("Upload a photo of your crop, or ask a question by voice.")

if "session" not in st.session_state:
    st.session_state.session = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

uploaded_image = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

with st.expander("Optional: ask a question by voice"):
    voice_question = st.audio_input("Record your question")

col1, col2 = st.columns([1, 1])
diagnose_clicked = col1.button("🔍 Diagnose", type="primary", disabled=uploaded_image is None)

if diagnose_clicked and uploaded_image is not None:
    os.makedirs("storage/uploads", exist_ok=True)
    image_path = os.path.join("storage/uploads", uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Save + transcribe voice question if provided (mainly displayed for
    # now — grounding follow-up answers in it is a natural next extension)
    transcribed_question = None
    if voice_question is not None:
        os.makedirs("storage/uploads", exist_ok=True)
        audio_path = os.path.join("storage/uploads", "question.wav")
        with open(audio_path, "wb") as f:
            f.write(voice_question.getbuffer())
        with st.spinner("Transcribing your question..."):
            transcription = transcribe(audio_path)
            transcribed_question = transcription.text
            st.info(f"Heard: \"{transcribed_question}\" ({transcription.detected_language})")

    with st.spinner("Analyzing photo..."):
        if demo_mode:
            # Fake diagnosis for UI testing while the real model trains
            diagnosis = DiagnosisResult(
                predicted_class="Tomato___Late_blight",
                confidence=0.91,
                tier="high",
                top_3=[("Tomato___Late_blight", 0.91), ("Tomato___Early_blight", 0.06), ("Tomato___healthy", 0.03)],
            )
        else:
            from app.services.disease_detection import diagnose_image
            diagnosis = diagnose_image(image_path)

        routed = route(diagnosis)

    weather = None
    weather_error = None
    if city.strip():
        with st.spinner("Checking local weather..."):
            try:
                weather = get_weather(city.strip())
            except Exception as e:
                # Weather is a bonus, not required — never block a diagnosis on it
                weather_error = str(e)

    progress = None
    community_note = None
    if routed.action == "give_advice":
        progress = history_store.get_progress_info(
            routed.diagnosis.predicted_class, routed.diagnosis.confidence
        )
        community_note = history_store.get_community_sightings_note(
            routed.diagnosis.predicted_class
        )
    progress_note = history_store.format_progress_note(progress) if progress else None

    if routed.action == "give_advice":
        with st.spinner("Getting advice..."):
            advice = generate_advice(
                diagnosis=routed.diagnosis.predicted_class,
                confidence=routed.diagnosis.confidence,
                language=language,
                weather_risk_note=weather.risk_note if weather else None,
                progress_note=progress_note,
                community_note=community_note,
            )
            spoken_text = advice.text
            st.session_state.session = ConversationSession(
                diagnosis=routed.diagnosis.predicted_class,
                confidence=routed.diagnosis.confidence,
                language=language,
                advice_given=advice.text,
            )
    else:
        spoken_text = routed.message
        st.session_state.session = None

    with st.spinner("Generating voice response..."):
        audio_path = synthesize_speech(spoken_text, language=language)

    history_store.save_entry(
        image_path=image_path,
        diagnosis_class=routed.diagnosis.predicted_class if routed.action == "give_advice" else None,
        confidence=routed.diagnosis.confidence,
        action=routed.action,
        advice_text=spoken_text,
        language=language,
    )

    st.session_state.last_result = {
        "action": routed.action,
        "message": routed.message,
        "spoken_text": spoken_text,
        "audio_path": audio_path,
        "diagnosis": routed.diagnosis,
        "weather": weather,
        "weather_error": weather_error,
        "progress": progress,
        "community_note": community_note,
    }

    # Refresh immediately so the sidebar's Diagnosis History reflects this
    # entry right away, instead of lagging one diagnosis behind until the
    # next interaction (e.g. a follow-up question) triggers a rerun.
    st.rerun()

# ---------------------------------------------------------------------------
# Display result
# ---------------------------------------------------------------------------
if st.session_state.last_result:
    result = st.session_state.last_result
    st.divider()

    if result["action"] == "give_advice":
        st.success(f"**Diagnosis:** {result['diagnosis'].predicted_class.replace('___', ' — ').replace('_', ' ')}")
        st.caption(f"Confidence: {result['diagnosis'].confidence:.0%}")
    elif result["action"] == "request_better_photo":
        st.warning(result["message"])
    else:
        st.error(result["message"])

    if result.get("weather"):
        w = result["weather"]
        icon = "⚠️" if w.risk_note else "🌤️"
        st.info(
            f"{icon} **{w.city}:** {w.temp_c:.0f}°C, {w.humidity}% humidity, {w.description}"
            + (f"\n\n{w.risk_note}" if w.risk_note else "")
        )
    elif result.get("weather_error"):
        st.caption(f"Weather check skipped: {result['weather_error']}")

    if result.get("progress"):
        p = result["progress"]
        prev_conf = f"{p.previous_entry.confidence:.0%}" if p.previous_entry.confidence else "unknown"
        st.info(f"📋 {history_store.format_progress_note(p)} (was {prev_conf} confidence then)")

    if result.get("community_note"):
        st.info(f"👥 {result['community_note']}")

    st.write(result["spoken_text"])
    st.audio(result["audio_path"])

    # -----------------------------------------------------------------
    # Follow-up conversation (only available after a successful diagnosis)
    # -----------------------------------------------------------------
    if st.session_state.session is not None:
        st.divider()
        st.subheader("Ask a follow-up question")

        for q, a in st.session_state.session.history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")

        followup_question = st.text_input("Type your question, or use voice below")
        followup_voice = st.audio_input("Or ask by voice")

        if st.button("Send"):
            question_text = followup_question
            if not question_text and followup_voice is not None:
                audio_path = os.path.join("storage/uploads", "followup.wav")
                with open(audio_path, "wb") as f:
                    f.write(followup_voice.getbuffer())
                question_text = transcribe(audio_path).text

            if question_text:
                with st.spinner("Thinking..."):
                    answer = answer_followup(st.session_state.session, question_text)
                    answer_audio = synthesize_speech(answer, language=language)
                st.markdown(f"**You:** {question_text}")
                st.markdown(f"**Assistant:** {answer}")
                st.audio(answer_audio)
                st.rerun()

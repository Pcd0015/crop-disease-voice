"""
Streamlit UI for the crop disease voice assistant. Ties together:
  - Photo upload OR live camera capture -> disease detection -> confidence-tier routing
  - Gemini-generated spoken advice (multilingual)
  - Voice question input (optional) via speech-to-text
  - Text-to-speech playback of the response
  - Grounded follow-up conversation
  - Diagnosis history log
  - Disease reference library (browse without diagnosing — Plantix-style)
  - Fertilizer calculator (Plantix-style, free/static reference data)
  - Forward-looking disease-risk weather forecast (goes beyond Plantix's
    free tier, which only shows current conditions)

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
from app.services.weather import get_weather, get_risk_forecast
from app.services import fertilizer_calculator
from app.data.disease_library import (
    DISEASE_LIBRARY, get_all_crops, get_entries_for_crop,
    count_diagnosable, count_reference_only,
)
from app.ui_theme import CUSTOM_CSS, banner_html, step_html, field_report_html, tag_pill_html, TIER_COLORS

st.set_page_config(page_title="Crop Disease Voice Assistant", page_icon="🌾", layout="centered")
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

MODEL_FILES_PRESENT = (
    os.path.exists("app/models/crop_disease_model.tflite")
    and os.path.exists("app/models/class_names.json")
)

# ---------------------------------------------------------------------------
# Sidebar: language selector + diagnosis history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="eyebrow">Settings</span>', unsafe_allow_html=True)
    st.markdown("### Preferences")

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

    city = st.text_input(
        "Your city (optional)",
        help="Used to check current + upcoming humidity/temperature for a "
             "fungal-disease risk note alongside your diagnosis. Leave blank to skip.",
    )

    st.divider()
    st.markdown('<span class="eyebrow">Diagnosis History</span>', unsafe_allow_html=True)
    entries = history_store.get_all_entries(limit=10)
    if not entries:
        st.caption("No diagnoses yet — your past results will appear here.")
    for entry in entries:
        label = entry.diagnosis_class or entry.action.replace("_", " ")
        label = label.replace("___", " — ").replace("_", " ")
        conf = f" · {entry.confidence:.0%}" if entry.confidence else ""
        st.markdown(
            f'<div class="history-chip">{entry.timestamp[:16].replace("T", " ")}'
            f'<br><strong>{label}</strong>{conf}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Top-level header + tabs
# ---------------------------------------------------------------------------
st.markdown(
    banner_html(
        "🌾",
        "Crop Disease Voice Assistant",
        "Photo diagnosis, spoken advice, and a free crop reference library — built for the field.",
    ),
    unsafe_allow_html=True,
)

tab_diagnose, tab_library, tab_fertilizer = st.tabs(
    ["🩺  Diagnose", "📚  Disease Library", "🧪  Fertilizer Calculator"]
)

# ===========================================================================
# TAB 1: Diagnose
# ===========================================================================
with tab_diagnose:
    st.caption("Upload a photo of your crop, take one with your camera, or ask a question by voice.")

    if "session" not in st.session_state:
        st.session_state.session = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    st.markdown(step_html(1, "Provide a photo of the affected leaf"), unsafe_allow_html=True)

    input_method = st.radio(
        "How do you want to provide a photo?",
        ["📁 Upload from gallery", "📷 Take a photo now"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_image = None
    if input_method == "📁 Upload from gallery":
        uploaded_image = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    else:
        uploaded_image = st.camera_input("Take a photo of the leaf")

    st.markdown(step_html(2, "Optional — ask a question by voice"), unsafe_allow_html=True)
    with st.expander("Record a question about your crop"):
        voice_question = st.audio_input("Record your question")

    st.markdown(step_html(3, "Get your diagnosis"), unsafe_allow_html=True)
    diagnose_clicked = st.button("🔍  Diagnose", type="primary", disabled=uploaded_image is None, use_container_width=True)

    if diagnose_clicked and uploaded_image is not None:
        os.makedirs("storage/uploads", exist_ok=True)
        # Camera captures don't have a stable filename, so build one that's
        # always valid regardless of input method (upload vs live camera).
        image_name = getattr(uploaded_image, "name", None) or "camera_capture.jpg"
        image_path = os.path.join("storage/uploads", image_name)
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
        risk_forecast = None
        if city.strip():
            with st.spinner("Checking local weather..."):
                try:
                    weather = get_weather(city.strip())
                except Exception as e:
                    # Weather is a bonus, not required — never block a diagnosis on it
                    weather_error = str(e)
            try:
                risk_forecast = get_risk_forecast(city.strip())
            except Exception:
                risk_forecast = None  # forecast is bonus-on-top-of-bonus, fail silently

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
            "risk_forecast": risk_forecast,
            "progress": progress,
            "community_note": community_note,
        }

        # Refresh immediately so the sidebar's Diagnosis History reflects this
        # entry right away, instead of lagging one diagnosis behind until the
        # next interaction (e.g. a follow-up question) triggers a rerun.
        st.rerun()

    # -----------------------------------------------------------------
    # Display result
    # -----------------------------------------------------------------
    if st.session_state.last_result:
        result = st.session_state.last_result
        st.divider()

        if result["action"] == "give_advice":
            diag = result["diagnosis"]
            name = diag.predicted_class.replace("___", " — ").replace("_", " ")
            tier_color = TIER_COLORS.get(diag.tier, TIER_COLORS["medium"])
            st.markdown(
                field_report_html("Field Report", name, diag.confidence, tier_color),
                unsafe_allow_html=True,
            )
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

        if result.get("risk_forecast"):
            with st.expander(f"📅 Upcoming disease-risk windows (next 3 days) — {len(result['risk_forecast'])} found"):
                st.caption(
                    "Forward-looking check: unlike a plain weather snapshot, this scans the "
                    "next 3 days for humidity/temperature conditions that favor fungal disease, "
                    "so you can act before symptoms actually show up."
                )
                for w in result["risk_forecast"][:6]:
                    st.write(f"**{w.when}** — {w.temp_c:.0f}°C, {w.humidity}% humidity, {w.description}")

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

# ===========================================================================
# TAB 2: Disease Library — browse without needing to diagnose (Plantix-style)
# ===========================================================================
with tab_library:
    st.caption(
        f"{len(DISEASE_LIBRARY)} conditions across {len(get_all_crops())} crops — no photo needed. "
        f"{count_diagnosable()} are recognized directly from a photo by this app's model; "
        f"the other {count_reference_only()} are reference-only browsing, clearly marked below. "
        "Symptoms and prevention are general information, not a substitute for an actual diagnosis."
    )

    lib_col1, lib_col2 = st.columns([2, 1])
    with lib_col1:
        crop_filter = st.selectbox("Filter by crop", ["All crops"] + get_all_crops())
    with lib_col2:
        scope_filter = st.selectbox("Show", ["All entries", "Photo-diagnosable only", "Reference only"])

    crops_to_show = get_all_crops() if crop_filter == "All crops" else [crop_filter]
    for crop in crops_to_show:
        crop_entries = get_entries_for_crop(crop)
        if scope_filter == "Photo-diagnosable only":
            crop_entries = [(k, v) for k, v in crop_entries if v.is_diagnosable]
        elif scope_filter == "Reference only":
            crop_entries = [(k, v) for k, v in crop_entries if not v.is_diagnosable]
        if not crop_entries:
            continue

        st.markdown(f"### {crop}")
        for class_name, info in crop_entries:
            icon = "✅" if info.is_healthy else "⚠️"
            with st.expander(f"{icon} {info.condition}"):
                st.markdown(tag_pill_html(info.is_diagnosable), unsafe_allow_html=True)
                st.markdown(f"**Symptoms:** {info.symptoms}")
                st.markdown(f"**Prevention:** {info.prevention}")

# ===========================================================================
# TAB 3: Fertilizer Calculator (Plantix-style, free/static reference data)
# ===========================================================================
with tab_fertilizer:
    st.caption(
        "Get a starting nitrogen/phosphorus/potassium (N-P-K) estimate for your "
        "crop, growth stage, and land area. This is a general reference, not a "
        "substitute for a soil test or local agricultural extension guidance."
    )

    fc_col1, fc_col2, fc_col3 = st.columns(3)
    with fc_col1:
        fc_crop = st.selectbox("Crop", fertilizer_calculator.available_crops())
    with fc_col2:
        fc_stage = st.selectbox("Growth stage", fertilizer_calculator.available_stages())
    with fc_col3:
        fc_area = st.number_input("Land area (acres)", min_value=0.1, value=1.0, step=0.1)

    if st.button("Calculate", type="primary"):
        plan = fertilizer_calculator.calculate(fc_crop, fc_stage, fc_area)
        st.success(f"**Estimated fertilizer needs for {plan.area_acres} acre(s) of {plan.crop} ({plan.stage}):**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Nitrogen (N)", f"{plan.n_kg} kg")
        m2.metric("Phosphorus (P)", f"{plan.p_kg} kg")
        m3.metric("Potassium (K)", f"{plan.k_kg} kg")
        st.caption(plan.note)

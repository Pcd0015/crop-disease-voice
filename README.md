---
title: Crop Disease Voice Assistant
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Crop Disease System with Voice — Step 1: Disease Detection Model

This is Step 1 of the full project. Scope: train and wire up the computer-vision
disease detection model, with confidence-based routing logic. Voice, LLM advice,
UI, and deployment come in later steps.

## What's here

```
crop_disease_voice/
├── training/
│   └── train_disease_model.py   # Run this on Google Colab (free GPU)
├── app/
│   ├── services/
│   │   ├── disease_detection.py  # Loads trained model, runs inference
│   │   └── response_router.py    # Confidence-tier routing (high/medium/low)
│   └── models/                    # Put your trained model files here
│       └── README.md
├── tests/
│   └── test_disease_detection.py # Tests the routing logic (no model needed)
└── requirements.txt
```

## Step 1 setup

### 1. Train the model (Google Colab — free)

1. Go to https://colab.research.google.com/, create a new notebook.
2. Runtime → Change runtime type → **T4 GPU** (free tier).
3. Copy the entire contents of `training/train_disease_model.py` into a cell, run it.
4. Takes ~30-45 min on the free GPU. Trains an EfficientNetB0 on the PlantVillage
   dataset (38 disease classes across 14 crops), auto-downloaded via
   `tensorflow_datasets` — no manual dataset download or Kaggle account needed.
5. At the end, download from the Colab file browser (left sidebar, folder icon):
   - `crop_disease_model.tflite`
   - `class_names.json`
6. Place both files into `app/models/` in this project.

### 2. Install local dependencies (for testing the inference code)

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the logic tests (works even before you've trained the model)

```bash
pytest tests/ -v
```

### 4. Try a real prediction (after model files are in place)

```python
from app.services.disease_detection import diagnose_image
from app.services.response_router import route

diagnosis = diagnose_image("path/to/leaf_photo.jpg")
result = route(diagnosis)
print(result.action)   # give_advice | request_better_photo | flag_for_expert
print(result.message)
```

## What determines quality here

- **Data augmentation during training** (rotation, zoom, brightness/contrast shifts)
  matters a lot — PlantVillage photos are clean lab shots, but real farmer photos
  won't be, so the augmentation simulates that gap.
- **Two-stage transfer learning** (frozen base → fine-tune top layers) — skipping
  stage 2 is the most common reason transfer-learning projects underperform.
- **TFLite export** — smaller/faster for CPU-only free hosting later (Hugging Face
  Spaces / Render free tier have no GPU).
- **Confidence tiering** — this is the actual "advanced" feature here: the system
  is honest about uncertainty instead of always confidently giving an answer.

## Step 2: Voice Pipeline (NEW)

Adds: speech-to-text, Gemini-generated advice (multilingual, organic/chemical
tracks), text-to-speech, and the orchestration layer tying it all together.

### Setup

```bash
pip install -r requirements.txt   # now includes faster-whisper, gTTS, google-generativeai

# Create a .env file (or export directly) with your free Gemini key:
echo "GEMINI_API_KEY=your_key_here" > .env
```

Get a free Gemini API key at https://aistudio.google.com/apikey — no card required.

### Test the wiring WITHOUT your trained model or a live API key

```bash
pytest tests/test_voice_pipeline.py -v
```

This uses mocks for `diagnose_image`, `generate_advice`, and `synthesize_speech`,
so it verifies the pipeline logic (confidence routing, when Gemini gets called
vs skipped, session state) is correct independent of whether your model has
finished training or your API key is set up yet.

### Try it for real (once your Step 1 model files are in `app/models/`)

```python
from dotenv import load_dotenv
load_dotenv()

from app.services.voice_pipeline import process_photo_and_voice

result = process_photo_and_voice(image_path="test_leaf.jpg", language="Hindi")
print(result.action)        # give_advice | request_better_photo | flag_for_expert
print(result.spoken_text)   # the advice text, in Hindi
print(result.audio_path)    # path to the generated MP3 — play this file to hear it
```

### What's in `app/services/` now

- `speech_to_text.py` — faster-whisper, free, runs locally, auto-detects language
- `advice_generation.py` — Gemini: generates advice (organic + chemical tracks,
  multilingual) and supports grounded follow-up conversation (used in Step 4)
- `text_to_speech.py` — gTTS, free, supports Hindi/Marathi/Telugu/Tamil/Bengali/
  Kannada/Gujarati/Punjabi/English (extend `LANGUAGE_CODES` for more)
- `voice_pipeline.py` — orchestrates: photo -> diagnosis -> routing -> advice -> speech

## Step 3: Streamlit UI (NEW)

A real clickable app tying everything together: photo upload, optional voice
question, spoken multilingual advice, follow-up conversation, and a diagnosis
history log — all in one page.

### Run it

```bash
pip install -r requirements.txt   # now includes streamlit
streamlit run streamlit_app.py
```

This opens in your browser automatically (usually http://localhost:8501).

### Demo Mode — use this while your model is still training

The sidebar has a **"Demo mode"** checkbox, automatically enabled if no
trained model files are found in `app/models/`. With it on, the app runs
the entire UI flow (photo upload → fake high-confidence diagnosis → real
Gemini advice → real voice output → real follow-up conversation) without
needing your trained model at all. Once your model files are in place,
untick it (or it'll auto-detect and default to off) to use real predictions.

### What you can test right now, before training finishes

1. Upload any leaf photo (doesn't matter what it actually shows — Demo
   mode ignores the image content and always returns a fake Late Blight
   diagnosis, just to exercise the full flow)
2. Optionally record a voice question using the browser mic
3. Click Diagnose — see the diagnosis, hear the spoken advice
4. Ask a follow-up question (typed or by voice) — grounded in the diagnosis
5. Check the sidebar — your diagnosis just got logged to history

### Once your model is ready

Drop `crop_disease_model.tflite` + `class_names.json` into `app/models/`,
restart the Streamlit app, untick Demo mode — now every diagnosis is a real
model prediction instead of the fake one.

## Step 4: Follow-up conversation — done

Grounded, multilingual follow-up Q&A wired into the Streamlit UI, with
vague/greeting handling and no meta-commentary leakage (fixed a Gemini
thinking-mode token budget bug that was truncating/garbling answers).

## Step 5: Weather risk context — done (partial)

- ✅ OpenWeatherMap integration: current temp/humidity/conditions surfaced
  next to the diagnosis, plus a plain-language fungal-disease risk note when
  humidity/temperature fall in a high-risk band.
- ❌ Agmarknet mandi price lookup — not built.
- ❌ Gov scheme pointer — not built.

## Step 6: Progress tracking + community sightings — done

- ✅ **Progress tracking**: `history_store.get_progress_info()` finds the
  most recent prior diagnosis of the same class and tells the farmer
  whether this looks better, worse, or about the same as last time.
- ✅ **Community sightings** (the "lightweight regional awareness" item):
  `history_store.get_community_sightings_note()` counts how many other
  diagnoses of the same class were logged on this deployed instance in the
  last 14 days, and both Gemini's advice text and the UI mention it when
  present. **Scope/honesty note:** this is not GPS-based — no location is
  collected per entry — so it reflects "how often this app has seen this
  diagnosis recently," not a verified geographic radius. Framed as
  "logged by other users of this app," not "near you."
- ✅ Fixed a bug where the sidebar's Diagnosis History lagged one entry
  behind — the diagnose flow now reruns immediately after saving, so new
  entries show up right away instead of waiting for the next interaction.

## Step 7: Deploy — done

Live on Render (free tier) via Docker. Model served via `.keras`/TFLite,
Gemini key and OpenWeatherMap key read from environment secrets, never
hardcoded.

## Step 8: Plantix-inspired free features + unique differentiators — done

Compared against Plantix's actual free feature set and closed the real gaps
without chasing the parts that need Plantix's scale (10M+ users) to work well:

- ✅ **Live camera capture** — sidebar radio lets a farmer choose "Upload from
  gallery" or "Take a photo now" (`st.camera_input`), so a phone camera can
  feed a photo straight in without a separate gallery step. No new dependency
  — ships with the Streamlit version already pinned.
- ✅ **Disease reference library** (`app/data/disease_library.py`, new
  "📚 Disease Library" tab) — browse symptoms/prevention for all 38 trained
  classes without uploading a photo, filterable by crop. Mirrors Plantix's
  library feature; fully static, zero API cost.
- ✅ **Fertilizer calculator** (`app/services/fertilizer_calculator.py`, new
  "🧪 Fertilizer Calculator" tab) — N-P-K estimate by crop, growth stage, and
  land area. Static reference rates, explicitly labeled as a starting
  estimate, not a soil-test replacement. Mirrors Plantix's calculator;
  zero API cost.
- ✅ **Forward-looking disease-risk forecast** (`weather.get_risk_forecast()`)
  — a genuine differentiator, not a Plantix feature: scans the next 3 days
  of OpenWeatherMap's free 5-day/3-hour forecast for humidity/temperature
  windows that favor fungal disease, so a farmer can act *before* symptoms
  appear instead of only after. Same free API key as the existing weather
  check, no extra cost.
- **Deliberately not built** (need Plantix's 10M+ user scale to work well,
  not free-tier-buildable at real quality): a farmer community forum, and
  true GPS-based regional outbreak mapping (the existing "community
  sightings" note is an honest, smaller-scope stand-in — see Step 6).
- **Already-existing differentiators vs. Plantix** (not new, just reaffirmed
  by this comparison): the voice interface end-to-end, and honest
  confidence-tier routing (Plantix always gives a confident answer; this
  app explicitly asks for a better photo or flags an expert below 80%).

## What's left (optional polish, not required for a working product)

- On-device offline inference (TFLite already exported, just needs a
  client-side integration path — no plan for this yet)
- Diagnosis history persistence across redeploys (currently resets on
  restart on Render's free tier, since there's no attached persistent
  volume — would need Render's paid disk add-on or an external DB like
  Supabase's free tier)
- Blockchain traceability module (mentioned in the original spec as a
  "future-ready" stretch feature — not started)


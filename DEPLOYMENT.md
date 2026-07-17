# Step 4 — Deploy to Hugging Face Spaces (free)

## 0. Before you start
Drop these 4 new files into the **root** of your `crop_disease_voice` project
folder, next to `streamlit_app.py`:
- `Dockerfile`
- `.dockerignore`
- `.gitattributes`
- `README.md` (if you already have a README.md, merge the YAML frontmatter
  block at the top into it — HF Spaces reads that frontmatter to configure
  the Space, and it MUST be the very first thing in the file)

## 1. Make sure your Gemini key is read from an environment variable
Open `app/services/advice_generation.py` (or wherever you call the Gemini
client) and confirm the key is loaded like this — not hardcoded:

```python
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
```

If it's currently hardcoded, fix that now — it's about to go in a public
git history otherwise.

## 2. Create the Space
1. Go to https://huggingface.co/new-space
2. Name it (e.g. `crop-disease-voice`)
3. **SDK: Docker** (not "Streamlit" — we're supplying our own Dockerfile so
   we control the audio/system dependencies)
4. Visibility: Public (required for the free CPU tier) or Private if you
   have a paid plan
5. Click **Create Space**

## 3. Add your secret
In the new Space: **Settings → Variables and secrets → New secret**
- Name: `GEMINI_API_KEY`
- Value: your actual key

This makes it available as an environment variable inside the container at
runtime — never committed to git.

## 4. Push your code
Hugging Face gives you a git remote URL on the Space page. From your
project root:

```bash
cd crop_disease_voice
git init                                   # skip if already a git repo
git lfs install
git add .
git commit -m "Step 4: containerize + deploy"
git remote add space https://huggingface.co/spaces/<your-username>/crop-disease-voice
git push space main
```

(If your default branch is `master` instead of `main`, push that instead:
`git push space master:main`)

Git LFS will automatically pick up `crop_disease_model.keras` and
`crop_disease_model.tflite` because of the `.gitattributes` rules — watch
for "Uploading LFS objects" in the push output.

## 5. Watch the build
The Space page shows a **Building** status with live logs. A cold build
(installing ffmpeg, TensorFlow/Lite, faster-whisper, etc.) typically takes
5–10 minutes on the free tier. Once it flips to **Running**, your app is
live at:

```
https://huggingface.co/spaces/<your-username>/crop-disease-voice
```

## 6. Turn off Demo Mode
Once deployed, untick "Demo mode" in the sidebar so it uses your real
`crop_disease_model.tflite` + `class_names.json` for inference instead of
the fake Late Blight response.

## Free tier limits to know about
- **2 vCPU / 16GB RAM**, no GPU — fine for TFLite inference and
  faster-whisper on short clips, but expect a few seconds of latency per
  request rather than instant.
- **Sleeps after inactivity** and cold-starts on the next visit (10–30s
  wake-up delay) — normal for the free tier, not a bug.
- **Public repo storage** counts against a shared free quota; your model
  files (~32MB combined) are well within it.

## If the build fails
Common first-deploy issues and fixes:
- `ffmpeg: command not found` at runtime → confirm the Dockerfile's
  `apt-get install` line made it into your committed `Dockerfile` (check
  you didn't accidentally push an older version).
- `ModuleNotFoundError` for something you added locally → it's missing
  from `requirements.txt`; add it and re-push.
- App builds but shows a blank page → check the Space's **Logs** tab first;
  it's almost always a missing env var (unset `GEMINI_API_KEY`) or a
  Streamlit port mismatch (must stay `7860` to match `app_port` in the
  README frontmatter).

---
Once this is live and you've confirmed a real (non-demo) diagnosis works
end-to-end on the deployed URL, tell me and we'll move to **Step 5:
multilingual support, weather API context, and the confidence-tier UI
routing** (the >80% / 60–80% / <60% flows from the original plan) — those
still need to be wired into the live app.

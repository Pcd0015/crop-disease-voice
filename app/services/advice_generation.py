"""
Generates farmer-facing treatment advice from a diagnosis, using Gemini.
Handles:
  - Multilingual output (single Gemini call, no separate translation API/cost)
  - Split organic vs chemical treatment tracks
  - Grounded conversational follow-up (keeps diagnosis + history in context,
    so follow-up questions don't need to re-explain the situation)

Uses the current google-genai SDK (the older google-generativeai package is
fully deprecated as of 2026 — no bug fixes or updates, and new API keys can
no longer access older model names like gemini-2.5-flash).

Kept as plain text output here (not JSON-schema constrained like the IDP
project's extraction) since this is farmer-facing prose meant to be read
aloud via TTS — structured JSON would just need re-flattening into text
anyway, so plain generation is simpler and appropriate for this use case.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types

_client = None

# gemini-flash-latest is an alias Google keeps pointed at their current
# recommended fast/cheap model (currently Gemini 3.5 Flash) — using the
# alias rather than pinning a specific version number means this code
# won't break again the next time Google retires an old model name.
DEFAULT_MODEL = "gemini-flash-latest"

ADVICE_PROMPT = """You are an agricultural extension advisor speaking directly to a
smallholder farmer. Your response will be read aloud via text-to-speech, so:
- Use short, simple sentences. No jargon, no markdown, no bullet symbols, no asterisks.
- Output ONLY the spoken response itself — no notes, no meta-commentary about phrasing.
- Speak in {language}.
- Be warm and direct, like a knowledgeable local expert, not a corporate assistant.

The farmer's crop has been diagnosed as: {diagnosis}
Confidence: {confidence:.0%}
{weather_context}{progress_context}
Give advice in exactly this structure, spoken naturally (do not print headers/labels,
just speak the two options as flowing sentences):
1. A one-sentence plain-language explanation of what this disease/issue is.
2. An organic/low-cost treatment option (something affordable and accessible).
3. A faster-acting chemical treatment option (name a general category of treatment,
   not a specific commercial brand).
4. One sentence on how to prevent this from happening again next season.
5. If current weather conditions are noted above as raising disease risk, briefly
   mention that so the farmer knows to act a bit sooner — otherwise skip this point.
6. If this is a repeat case (noted above), briefly acknowledge it and mention whether
   it looks better, worse, or about the same as last time — otherwise skip this point.

Keep the whole response under 150 words — remember, this will be spoken aloud, not read.
"""

FOLLOWUP_PROMPT = """You are continuing a conversation with a farmer about their crop.

Earlier diagnosis: {diagnosis} (confidence: {confidence:.0%})
Advice already given: {previous_advice}

The farmer now says: "{question}"

Respond in {language}, in simple spoken language (short sentences, no jargon, no
markdown, no asterisks, no bullet points, no notes about how you're phrasing things).
Output ONLY the final spoken response itself — nothing else.

If this is just a greeting (like "hi" or "hello") or too vague to be a real question,
respond with a brief friendly greeting and ask what they'd like to know about their
crop. Otherwise, answer the question directly, grounded in the diagnosis and advice
above. If the question is outside what you can responsibly answer (e.g. asks about
human/animal health beyond general safety, or is unrelated to this crop issue), say
so honestly and suggest they consult a local expert or doctor.

Keep the answer under 100 words.
"""


@dataclass
class AdviceResult:
    text: str
    language: str


@dataclass
class ConversationSession:
    """Holds context for a follow-up conversation grounded in one diagnosis."""
    diagnosis: str
    confidence: float
    language: str
    advice_given: str = ""
    history: list = field(default_factory=list)  # [(question, answer), ...]


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


def _generate(prompt: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=600,
            # Gemini 3.5 Flash "thinks" by default, and that reasoning
            # consumes the same token budget as the visible answer —
            # for simple farmer-facing advice text we don't need deep
            # reasoning, so disable it. This fixes both truncated
            # mid-sentence answers and stray reasoning fragments leaking
            # into the visible response.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


def generate_advice(
    diagnosis: str,
    confidence: float,
    language: str = "English",
    weather_risk_note: Optional[str] = None,
    progress_note: Optional[str] = None,
) -> AdviceResult:
    weather_context = f"\nCurrent weather note: {weather_risk_note}\n" if weather_risk_note else ""
    progress_context = f"\nRepeat case note: {progress_note}\n" if progress_note else ""
    prompt = ADVICE_PROMPT.format(
        diagnosis=diagnosis,
        confidence=confidence,
        language=language,
        weather_context=weather_context,
        progress_context=progress_context,
    )
    text = _generate(prompt)
    return AdviceResult(text=text, language=language)


def answer_followup(session: ConversationSession, question: str) -> str:
    prompt = FOLLOWUP_PROMPT.format(
        diagnosis=session.diagnosis,
        confidence=session.confidence,
        previous_advice=session.advice_given,
        question=question,
        language=session.language,
    )
    answer = _generate(prompt)
    session.history.append((question, answer))
    return answer

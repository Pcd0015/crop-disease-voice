"""
Translates a raw DiagnosisResult into the actual farmer-facing flow:
  high   -> proceed straight to Gemini-generated treatment advice
  medium -> ask for a better photo, don't give advice yet (avoids
            confidently-wrong advice on a bad photo)
  low    -> tell the user the image isn't usable, suggest expert contact

This separation matters: it's the difference between a demo that always
"succeeds" and a system that's honest about its own uncertainty — which
is exactly the kind of thing worth highlighting when you present this
project, since it's a real failure mode in careless CV deployments.
"""
from dataclasses import dataclass

from app.services.disease_detection import DiagnosisResult


@dataclass
class RoutedResponse:
    action: str          # "give_advice" | "request_better_photo" | "flag_for_expert"
    message: str
    diagnosis: DiagnosisResult


def route(diagnosis: DiagnosisResult) -> RoutedResponse:
    if diagnosis.tier == "high":
        return RoutedResponse(
            action="give_advice",
            message=(
                f"Diagnosis: {_format_class_name(diagnosis.predicted_class)} "
                f"(confidence: {diagnosis.confidence:.0%})"
            ),
            diagnosis=diagnosis,
        )

    if diagnosis.tier == "medium":
        return RoutedResponse(
            action="request_better_photo",
            message=(
                f"I'm not fully sure — it might be {_format_class_name(diagnosis.predicted_class)} "
                f"({diagnosis.confidence:.0%} confidence), but I'd like to be more certain. "
                "Could you take another photo? Try to get a single leaf in good daylight, "
                "filling most of the frame, without heavy shadows."
            ),
            diagnosis=diagnosis,
        )

    return RoutedResponse(
        action="flag_for_expert",
        message=(
            "I can't get a reliable reading from this image. This could be due to poor "
            "lighting, blur, or the leaf not being clearly visible. Please try again with "
            "a clearer photo, or consider contacting your local agricultural extension "
            "office for an in-person assessment."
        ),
        diagnosis=diagnosis,
    )


def _format_class_name(raw_class_name: str) -> str:
    """PlantVillage class names look like 'Tomato___Late_blight' — clean up for display."""
    parts = raw_class_name.replace("___", " - ").replace("_", " ")
    return parts

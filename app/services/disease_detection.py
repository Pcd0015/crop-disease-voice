"""
Disease detection inference service. Loads the model trained via
training/train_disease_model.py and runs predictions with the
confidence-based routing logic:

  >= 0.80  -> high confidence, give treatment advice directly
  0.60-0.80 -> medium confidence, ask for a clearer/different-angle photo
  < 0.60   -> low confidence, flag for human/expert review

Supports either the full Keras model (.keras) or the TFLite version
(.tflite) — TFLite is smaller and faster for CPU-only deployment, which
matters if you're hosting on a free tier (Hugging Face Spaces / Render
free tier have limited CPU and no GPU).
"""
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_SIZE = 224

HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.60


@dataclass
class DiagnosisResult:
    predicted_class: str
    confidence: float
    tier: str  # "high" | "medium" | "low"
    top_3: list  # [(class_name, confidence), ...]


class DiseaseDetector:
    def __init__(self, use_tflite: bool = True):
        self.use_tflite = use_tflite
        self.class_names = self._load_class_names()
        self.model = self._load_model()

    def _load_class_names(self) -> list:
        path = os.path.join(MODEL_DIR, "class_names.json")
        with open(path) as f:
            return json.load(f)

    def _load_model(self):
        if self.use_tflite:
            import tensorflow as tf
            path = os.path.join(MODEL_DIR, "crop_disease_model.tflite")
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            import tensorflow as tf
            path = os.path.join(MODEL_DIR, "crop_disease_model.keras")
            return tf.keras.models.load_model(path)

    def _preprocess(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _predict_tflite(self, arr: np.ndarray) -> np.ndarray:
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        self.model.set_tensor(input_details[0]["index"], arr)
        self.model.invoke()
        return self.model.get_tensor(output_details[0]["index"])[0]

    def _predict_keras(self, arr: np.ndarray) -> np.ndarray:
        return self.model.predict(arr, verbose=0)[0]

    def diagnose(self, image_path: str) -> DiagnosisResult:
        arr = self._preprocess(image_path)
        probs = self._predict_tflite(arr) if self.use_tflite else self._predict_keras(arr)

        top_indices = np.argsort(probs)[::-1][:3]
        top_3 = [(self.class_names[i], float(probs[i])) for i in top_indices]

        best_class, best_confidence = top_3[0]

        if best_confidence >= HIGH_CONFIDENCE_THRESHOLD:
            tier = "high"
        elif best_confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            tier = "medium"
        else:
            tier = "low"

        return DiagnosisResult(
            predicted_class=best_class,
            confidence=best_confidence,
            tier=tier,
            top_3=top_3,
        )


_detector = None


def get_detector() -> DiseaseDetector:
    global _detector
    if _detector is None:
        _detector = DiseaseDetector(use_tflite=True)
    return _detector


def diagnose_image(image_path: str) -> DiagnosisResult:
    return get_detector().diagnose(image_path)

"""
utils.py
--------
Inference-side helpers for the Flask backend.

CHANGED FOR THE PRETRAINED-MODEL APPROACH:
This module no longer trains or loads a locally-trained Keras model. It
loads a ready-made, publicly available pretrained model from the Hugging
Face Hub and uses it directly for inference.

Model used:
  linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
  https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification

  - Architecture: MobileNetV2, fine-tuned from google/mobilenet_v2_1.0_224
  - Fine-tuned on the "New Plant Diseases Dataset" (Kaggle, augmented
    version of the PlantVillage dataset) — 38 classes across 14 crop
    species, including healthy classes.
  - Self-reported evaluation accuracy on the model card: ~95.4%.
    (This is the model author's own reported number, not something
    measured in this project — see the model card for methodology.)
  - Framework: PyTorch, served through the Hugging Face `transformers`
    library's image-classification pipeline.
  - Weights (~9-10 MB) download automatically the first time this module
    runs and are cached locally by `transformers`/`huggingface_hub`
    (default cache: ~/.cache/huggingface) — no manual download step,
    and no PlantVillage dataset download required.

Requires internet access the FIRST time the app makes a prediction (to
fetch the weights). After that, it works fully offline using the cache.
"""
import json
import os
import re
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import DISEASE_INFO_PATH, CONFIDENCE_LOW_THRESHOLD

MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_error = None

_disease_info = None
_normalized_lookup = None  # normalized_key -> original_key in disease_info.json


def _normalize(label):
    """Lower-cases and collapses any run of non-alphanumeric characters to a
    single underscore, so that minor formatting differences between the
    model's raw label strings and our disease_info.json keys (underscores
    vs commas vs parentheses vs casing) still match. E.g. both
    'Pepper,_bell___Bacterial_spot' and 'Pepper__bell___Bacterial_spot'
    normalize to 'pepper_bell_bacterial_spot'."""
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _load_disease_info():
    global _disease_info, _normalized_lookup
    if _disease_info is None:
        if not os.path.isfile(DISEASE_INFO_PATH):
            raise FileNotFoundError(f"disease_info.json not found at {DISEASE_INFO_PATH}")
        with open(DISEASE_INFO_PATH, "r") as f:
            _disease_info = json.load(f)
        _normalized_lookup = {_normalize(k): k for k in _disease_info}
    return _disease_info


def is_model_available():
    """Checks whether the required libraries (torch + transformers) are
    installed. This does NOT guarantee the weights have downloaded yet —
    that happens lazily on first prediction and needs internet access once."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load_pipeline():
    """Lazily creates (and caches) the Hugging Face image-classification
    pipeline. Downloads the model weights from the Hub on first call if
    they aren't already cached locally. Thread-safe."""
    global _pipeline, _pipeline_error

    if _pipeline is not None:
        return _pipeline
    if _pipeline_error is not None:
        raise _pipeline_error

    with _pipeline_lock:
        if _pipeline is None:
            try:
                from transformers import pipeline
                _pipeline = pipeline("image-classification", model=MODEL_ID)
            except Exception as e:
                _pipeline_error = RuntimeError(
                    f"Could not load the pretrained model '{MODEL_ID}'. "
                    f"If this is the first run, this usually means no internet "
                    f"connection was available to download the weights from "
                    f"the Hugging Face Hub. Original error: {e}"
                )
                raise _pipeline_error
    return _pipeline


def _lookup_info(raw_label):
    info_db = _load_disease_info()
    if raw_label in info_db:
        return info_db[raw_label]
    norm = _normalize(raw_label)
    if norm in _normalized_lookup:
        return info_db[_normalized_lookup[norm]]
    return None


def predict(pil_image):
    """Runs the pretrained model on a single PIL image. Returns a dict with
    the predicted class (the model's raw label string), confidence, top-3
    alternatives, and disease info looked up from data/disease_info.json.
    Confidence scores come directly from the model's softmax output —
    nothing here is fabricated or hardcoded."""
    pipe = load_pipeline()

    results = pipe(pil_image.convert("RGB"), top_k=5)
    # results: list of {"label": str, "score": float}, sorted descending by score
    if not results:
        raise RuntimeError("The model returned no predictions for this image.")

    top = results[0]
    predicted_class_raw = top["label"]
    confidence = round(float(top["score"]) * 100, 2)

    info = _lookup_info(predicted_class_raw)
    if info is None:
        # The model predicted a class we have no info entry for.
        # Do NOT invent information — flag it clearly instead.
        info = {
            "crop": "Unknown",
            "disease": predicted_class_raw,
            "healthy": "healthy" in predicted_class_raw.lower(),
            "cause": "No information available for this exact class label in disease_info.json.",
            "symptoms": "Not available.",
            "prevention": "Not available.",
            "cure": (
                "Not available. The model predicted a label this project's "
                "disease_info.json doesn't have an entry for — add one using "
                "the model's exact label text (or check the normalization "
                "note in backend/utils.py)."
            ),
        }

    top3 = [
        {"class_name": r["label"], "confidence": round(float(r["score"]) * 100, 2)}
        for r in results[:3]
    ]

    return {
        "predicted_class": predicted_class_raw,
        "confidence": confidence,
        "is_healthy": bool(info.get("healthy", False)),
        "low_confidence": confidence < CONFIDENCE_LOW_THRESHOLD,
        "top3": top3,
        "info": info,
    }

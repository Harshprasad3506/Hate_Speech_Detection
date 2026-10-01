"""
======================================================================
HATE SPEECH AND OFFENSIVE LANGUAGE DETECTION — INFERENCE MODULE
======================================================================
Academic ML Prediction & Inference Module for Hate Speech Detection.

Model Architecture:
  Text Input -> Preprocessing (clean_text) -> TF-IDF Vectorizer -> LinearSVC

Functionality:
  1. Loads pre-trained LinearSVC model & TF-IDF vectorizer artifacts.
  2. Reuses clean_text() from src/preprocessing.py for exact pipeline consistency.
  3. Supports both single-sentence CLI prediction and interactive terminal mode.
  4. Returns structured prediction dictionaries for external Python integration.

Training Rule:
  This script ONLY performs inference. It does not fit or retrain models or
  vectorizers. Run `python src/train.py` to train or regenerate artifacts.
======================================================================
"""

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Dynamic path resolution to ensure src.preprocessing imports correctly
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import clean_text

# ====================================================================
# CENTRALIZED CONFIGURATION & MODEL PATHS
# ====================================================================
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "svm_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

DEFAULT_ENCODING = "utf-8"

# Default target class label mapping
DEFAULT_CLASS_LABELS: Dict[int, str] = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

# Logger setup
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure basic logging format for inference execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


# ====================================================================
# ARTIFACT LOADING & VALIDATION
# ====================================================================
def validate_model_files() -> None:
    """
    Verify that required pre-trained model files exist on disk.

    Raises
    ------
    FileNotFoundError
        If svm_model.pkl or tfidf_vectorizer.pkl is missing.
    """
    missing_files = []
    if not MODEL_PATH.exists():
        missing_files.append(str(MODEL_PATH))
    if not VECTORIZER_PATH.exists():
        missing_files.append(str(VECTORIZER_PATH))

    if missing_files:
        error_msg = (
            f"\nRequired model artifact(s) not found:\n"
            + "\n".join(f"  - {f}" for f in missing_files)
            + "\n\nPlease train the model first by running:\n  python src/train.py"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """
    Load serialized LinearSVC model artifact from disk.

    Parameters
    ----------
    model_path : Path
        Path to svm_model.pkl file.

    Returns
    -------
    Any
        Loaded trained LinearSVC model instance.
    """
    logger.info("Loading trained SVM model...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model from '{model_path}': {e}")
        raise e

    if not hasattr(model, "predict"):
        raise ValueError("Loaded model object does not appear to be a valid trained classifier.")

    return model


def load_vectorizer(vectorizer_path: Path = VECTORIZER_PATH) -> Any:
    """
    Load serialized TF-IDF vectorizer artifact from disk.

    Parameters
    ----------
    vectorizer_path : Path
        Path to tfidf_vectorizer.pkl file.

    Returns
    -------
    Any
        Loaded fitted TfidfVectorizer instance.
    """
    logger.info("Loading TF-IDF vectorizer...")
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load vectorizer from '{vectorizer_path}': {e}")
        raise e

    if not hasattr(vectorizer, "transform"):
        raise ValueError("Loaded vectorizer object does not support .transform().")

    return vectorizer


def load_metadata(metadata_path: Path = METADATA_PATH) -> Dict[str, Any]:
    """
    Safely load metadata JSON file if available.

    Parameters
    ----------
    metadata_path : Path
        Path to model_metadata.json file.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary or empty dict if not found/invalid.
    """
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r", encoding=DEFAULT_ENCODING) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load metadata file: {e}")
        return {}


def get_class_labels(metadata: Dict[str, Any]) -> Dict[int, str]:
    """
    Determine class label mapping from metadata or return fallback defaults.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Loaded model metadata dictionary.

    Returns
    -------
    Dict[int, str]
        Class integer ID to human-readable label string mapping.
    """
    if "class_labels" in metadata and isinstance(metadata["class_labels"], dict):
        try:
            return {int(k): str(v) for k, v in metadata["class_labels"].items()}
        except ValueError:
            pass

    return DEFAULT_CLASS_LABELS.copy()


def get_label_name(predicted_class: int, class_labels: Dict[int, str]) -> str:
    """Convert numeric predicted class into human-readable label name."""
    return class_labels.get(predicted_class, f"Unknown Class ({predicted_class})")


# ====================================================================
# PREPROCESSING & INFERENCE CORE
# ====================================================================
def preprocess_input(text: str) -> str:
    """
    Preprocess raw text string using clean_text from preprocessing.py.

    Parameters
    ----------
    text : str
        Raw input sentence string.

    Returns
    -------
    str
        Cleaned text string.

    Raises
    ------
    ValueError
        If text is empty or produces an empty string after cleaning.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a non-empty text string.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("The provided text does not contain usable content after preprocessing.")

    return cleaned


def predict_text(
    text: str,
    model: Any,
    vectorizer: Any,
    class_labels: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Classify input sentence using loaded TF-IDF vectorizer and LinearSVC model.

    Parameters
    ----------
    text : str
        Raw text sentence to classify.
    model : Any
        Pre-trained LinearSVC model instance.
    vectorizer : Any
        Pre-fitted TfidfVectorizer instance.
    class_labels : Optional[Dict[int, str]]
        Class label mapping dictionary.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing input, cleaned text, predicted class, and label.
    """
    if class_labels is None:
        class_labels = DEFAULT_CLASS_LABELS

    cleaned_text = preprocess_input(text)

    # CRITICAL INFERENCE RULE: Never call fit() or fit_transform() during prediction!
    tfidf_features = vectorizer.transform([cleaned_text])
    predicted_class = int(model.predict(tfidf_features)[0])
    predicted_label = get_label_name(predicted_class, class_labels)

    # Compute raw decision scores from LinearSVC (NOT probabilities)
    decision_scores = None
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(tfidf_features)[0]
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            decision_scores = scores
        except Exception:
            decision_scores = None

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "decision_scores": decision_scores
    }


def format_prediction(result: Dict[str, Any]) -> str:
    """Format prediction result dictionary into a clean terminal report string."""
    banner = f"""
============================================================
HATE SPEECH DETECTION - PREDICTION RESULT
============================================================

Input Text:
  {result['original_text']}

Cleaned Text:
  {result['cleaned_text']}

Predicted Class:
  [{result['predicted_class']}] {result['predicted_label']}
"""
    if result.get("decision_scores") is not None:
        banner += f"\nDecision Scores (LinearSVC Raw Margins):\n  {result['decision_scores']}\n"

    banner += "============================================================"
    return banner


# ====================================================================
# EXECUTION MODES (SINGLE PREDICTION & INTERACTIVE TERMINAL)
# ====================================================================
def run_single_prediction(
    text: str,
    model: Any,
    vectorizer: Any,
    class_labels: Dict[int, str]
) -> None:
    """Execute prediction for a single CLI text argument."""
    try:
        result = predict_text(text, model, vectorizer, class_labels)
        print(format_prediction(result))
    except ValueError as val_err:
        print(f"\n[Validation Error] {val_err}")
    except Exception as err:
        logger.error(f"Prediction failed: {err}")


def run_interactive_mode(
    model: Any,
    vectorizer: Any,
    class_labels: Dict[int, str]
) -> None:
    """Start interactive terminal session for classifying multiple user sentences."""
    print("============================================================")
    print("HATE SPEECH AND OFFENSIVE LANGUAGE DETECTION")
    print("============================================================")
    print("Model & TF-IDF Vectorizer loaded successfully.")
    print("Enter text sentence to classify.")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("============================================================\n")

    while True:
        try:
            user_input = input("Text > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting interactive session. Goodbye!")
                break

            if not user_input:
                print("Please enter some text.\n")
                continue

            result = predict_text(user_input, model, vectorizer, class_labels)
            print(f"\n-> Prediction: [{result['predicted_class']}] {result['predicted_label']}")
            print(f"   Cleaned   : {result['cleaned_text']}\n")

        except ValueError as val_err:
            print(f"[Warning] {val_err}\n")
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}\n")


# ====================================================================
# MAIN ENTRYPOINT
# ====================================================================
def main() -> None:
    """Main execution function handling setup, artifact loading, and mode selection."""
    setup_logging()

    try:
        validate_model_files()
        model = load_model(MODEL_PATH)
        vectorizer = load_vectorizer(VECTORIZER_PATH)
        metadata = load_metadata(METADATA_PATH)
        class_labels = get_class_labels(metadata)

        # Check if text was passed as command line arguments
        if len(sys.argv) > 1:
            input_text = " ".join(sys.argv[1:])
            run_single_prediction(input_text, model, vectorizer, class_labels)
        else:
            run_interactive_mode(model, vectorizer, class_labels)

    except FileNotFoundError:
        sys.exit(1)
    except Exception as err:
        logger.error(f"Initialization failure: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

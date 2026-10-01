"""
======================================================================
HATE SPEECH AND OFFENSIVE LANGUAGE DETECTION USING LINEAR SVM
======================================================================
Academic ML Training Pipeline for Hate Speech Detection.
Algorithm: Support Vector Machine (LinearSVC)
Feature Extraction: TF-IDF Vectorization
Dataset: Hate Speech & Offensive Language Dataset
======================================================================
"""

# ====================================================================
# SECTION 1 — PROFESSIONAL IMPORTS
# ====================================================================
import json
import logging
import os
import pickle
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Dynamic path configuration to ensure local modules import correctly
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import clean_text

# ====================================================================
# SECTION 2 — PATH MANAGEMENT
# ====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

DATA_PATH = os.path.join(DATA_DIR, "labeled_data_actual.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

CLASS_DIST_PLOT_PATH = os.path.join(RESULTS_DIR, "class_distribution.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# Ensure required target directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ====================================================================
# SECTION 3 — LOGGING
# ====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ====================================================================
# SECTION 4 & 5 — RANDOM SEED & CONFIGURATION
# ====================================================================
RANDOM_STATE = 42

# Seed Python and NumPy random generators for reproducible ML execution
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Train/Test Split configuration
TEST_SIZE = 0.20

# TF-IDF Feature Extraction configuration parameters
# Controls max vocabulary size, n-gram range (unigram + bigram), and minimum document frequency threshold
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# LinearSVC Classifier Hyperparameters
# C parameter controls soft-margin regularization trade-off
# max_iter defines maximum iterations for solver convergence
SVM_C = 1.0
SVM_MAX_ITER = 2000

# ====================================================================
# SECTION 6 — CLASS LABELS MAPPING
# ====================================================================
CLASS_NAMES = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

# ====================================================================
# SECTION 7 & 8 — DATASET LOADING & VALIDATION
# ====================================================================
def validate_dataset_existence(path: str) -> None:
    """Validate that the target dataset file exists at the specified path."""
    if not os.path.exists(path):
        logger.error(f"Dataset file not found at: {path}")
        raise FileNotFoundError(
            f"Dataset missing at '{path}'. Please ensure labeled_data_actual.csv exists in data/ directory."
        )

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset CSV into pandas DataFrame and log basic summary."""
    logger.info(f"Loading dataset from: {path}")
    validate_dataset_existence(path)
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.exception("Failed to read dataset CSV file.")
        raise e

    if df.empty:
        raise ValueError(f"Loaded dataset from '{path}' is empty.")

    logger.info(f"Dataset loaded successfully | Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def validate_dataset(df: pd.DataFrame) -> None:
    """Verify that required columns 'tweet' and 'class' are present in the dataset."""
    logger.info("Validating dataset columns...")
    required_cols = ["tweet", "class"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Dataset missing required column(s): {missing_cols}")

    if df["class"].isnull().all() or df["tweet"].isnull().all():
        raise ValueError("Critical columns ('tweet' or 'class') contain all null values.")

    logger.info("Dataset validation passed successfully.")

# ====================================================================
# SECTION 10, 11, 12 — DATASET EXPLORATION & CLEANING
# ====================================================================
def display_dataset_overview(df: pd.DataFrame) -> None:
    """Print dataset structural summary and sample records."""
    logger.info("Displaying dataset structural overview:")
    print("-" * 60)
    print(f"Total Records: {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("Data Types:\n", df.dtypes.to_string())
    print("\nSample Head Records:")
    print(df[["class", "tweet"]].head(3).to_string())
    print("-" * 60)

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check and report missing values per column."""
    logger.info("Analyzing missing values in dataset...")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df)) * 100

    missing_df = pd.DataFrame({"Missing Count": null_counts, "Percentage (%)": null_pct})
    print("\nMissing Values Summary:")
    print(missing_df.to_string())

    # Drop rows where essential columns ('tweet' or 'class') are missing
    df_clean = df.dropna(subset=["tweet", "class"]).copy()
    return df_clean

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify and remove duplicate text records."""
    logger.info("Checking for duplicate tweet records...")
    initial_rows = len(df)
    duplicate_count = df.duplicated(subset=["tweet"]).sum()
    logger.info(f"Duplicate records found: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates(subset=["tweet"]).reset_index(drop=True)
        logger.info(f"Removed duplicates | Rows before: {initial_rows}, Rows after: {len(df)}")

    return df

# ====================================================================
# SECTION 13 — CLASS DISTRIBUTION ANALYSIS & PLOTTING
# ====================================================================
def analyze_and_plot_class_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Calculate class frequency distribution and save a visualization plot."""
    logger.info("Analyzing class distribution...")
    class_counts = df["class"].value_counts().sort_index()
    total = len(df)

    print("\nClass Distribution:")
    for cls_val, count in class_counts.items():
        cls_name = CLASS_NAMES.get(cls_val, f"Class {cls_val}")
        pct = (count / total) * 100
        print(f"  [{cls_val}] {cls_name:20s}: {count:6d} ({pct:5.2f}%)")

    # Generate Class Distribution Bar Chart
    plt.figure(figsize=(8, 5))
    palette = sns.color_palette("Blues_r", len(class_counts))
    labels = [CLASS_NAMES.get(i, f"Class {i}") for i in class_counts.index]
    
    bars = plt.bar(labels, class_counts.values, color=palette, edgecolor="black", alpha=0.85)
    plt.title("Class Distribution of Hate Speech Dataset", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Category Class", fontsize=12, labelpad=8)
    plt.ylabel("Number of Samples", fontsize=12, labelpad=8)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bar charts with counts
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (total * 0.01),
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Class distribution chart saved to: {output_path}")

# ====================================================================
# SECTION 15, 16, 17 — TEXT CLEANING & STATISTICAL ANALYSIS
# ====================================================================
def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text column using preprocessing module clean_text function."""
    logger.info("Applying text cleaning (preprocessing)...")
    df = df.copy()
    df["clean_tweet"] = df["tweet"].apply(clean_text)

    # Compare sample original vs cleaned text
    print("\nSample Preprocessing Results (Original -> Cleaned):")
    for i in range(min(3, len(df))):
        print(f" Original [{i+1}]: {df['tweet'].iloc[i]}")
        print(f" Cleaned  [{i+1}]: {df['clean_tweet'].iloc[i]}\n")

    # Filter out empty strings after cleaning
    empty_mask = df["clean_tweet"].str.strip() == ""
    empty_count = empty_mask.sum()
    logger.info(f"Empty text rows after cleaning: {empty_count}")

    if empty_count > 0:
        df = df[~empty_mask].reset_index(drop=True)
        logger.info(f"Removed {empty_count} empty cleaned records. Remaining rows: {len(df)}")

    return df

def analyze_text_statistics(df: pd.DataFrame) -> None:
    """Calculate and display text length and word count statistics."""
    char_lens = df["clean_tweet"].str.len()
    word_counts = df["clean_tweet"].apply(lambda x: len(x.split()))

    print("\nText Preprocessing Statistics:")
    print(f"  Average Character Count : {char_lens.mean():.2f}")
    print(f"  Min / Max Character Count: {char_lens.min()} / {char_lens.max()}")
    print(f"  Average Word Count      : {word_counts.mean():.2f}")
    print(f"  Min / Max Word Count    : {word_counts.min()} / {word_counts.max()}")

# ====================================================================
# SECTION 18 — TRAIN-TEST SPLIT
# ====================================================================
def perform_train_test_split(X: pd.Series, y: pd.Series):
    """Split cleaned dataset into stratified train and test partitions."""
    logger.info(f"Splitting dataset with test_size={TEST_SIZE} and stratify=y...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info(f"Train set size: {len(X_train)} ({100*(1-TEST_SIZE):.0f}%) | Test set size: {len(X_test)} ({100*TEST_SIZE:.0f}%)")
    return X_train, X_test, y_train, y_test

# ====================================================================
# SECTION 19 & 20 — TF-IDF FEATURE EXTRACTION
# ====================================================================
def extract_tfidf_features(X_train: pd.Series, X_test: pd.Series):
    """
    Extract TF-IDF feature representations.
    
    IMPORTANT DATA LEAKAGE PREVENTION:
    TF-IDF must be fitted ONLY on training data (X_train) and then used to transform
    the test data (X_test). Fitting TF-IDF on the entire dataset prior to splitting
    would cause data leakage between train and test distributions.
    """
    logger.info("Initializing TfidfVectorizer...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True
    )

    logger.info("Fitting TF-IDF on training set & transforming train/test partitions...")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    vocab_size = len(tfidf.vocabulary_)
    logger.info(f"TF-IDF Extraction complete | Vocabulary Size: {vocab_size} terms")
    logger.info(f"X_train_tfidf shape: {X_train_tfidf.shape} | X_test_tfidf shape: {X_test_tfidf.shape}")

    # Inspect sample vocabulary terms
    sample_terms = list(tfidf.get_feature_names_out())[:10]
    print(f"Sample TF-IDF Terms (First 10): {sample_terms}")

    return tfidf, X_train_tfidf, X_test_tfidf

# ====================================================================
# SECTION 21 & 22 — LINEAR SVM MODEL TRAINING
# ====================================================================
def train_linear_svc(X_train_tfidf, y_train) -> LinearSVC:
    """
    Initialize and train LinearSVC classifier.
    
    IMPORTANT SVM CONCEPT:
    LinearSVC is a linear Support Vector Machine classifier optimized for text classification.
    It works efficiently with high-dimensional sparse TF-IDF matrices by finding optimal
    hyperplanes separating classes with maximum margin.
    """
    logger.info(f"Initializing LinearSVC classifier (C={SVM_C}, max_iter={SVM_MAX_ITER})...")
    model = LinearSVC(
        C=SVM_C,
        random_state=RANDOM_STATE,
        max_iter=SVM_MAX_ITER
    )

    logger.info("Training LinearSVC model on X_train_tfidf...")
    model.fit(X_train_tfidf, y_train)
    logger.info("SVM Model training completed successfully.")
    return model

# ====================================================================
# SECTION 23 - 28 — MODEL EVALUATION
# ====================================================================
def evaluate_model(model: LinearSVC, X_test_tfidf, y_test) -> dict:
    """Evaluate trained LinearSVC model and calculate metrics from test dataset predictions."""
    logger.info("Generating predictions on X_test_tfidf...")
    y_pred = model.predict(X_test_tfidf)

    if len(y_pred) != X_test_tfidf.shape[0]:
        raise ValueError("Prediction count mismatch with test set sample count.")

    logger.info("Calculating evaluation metrics from actual dataset predictions...")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": clf_report
    }

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS (Actual Calculations)")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("\nClassification Report:\n")
    print(clf_report)
    print("=" * 60)

    return metrics

# ====================================================================
# SECTION 29 — CONFUSION MATRIX PLOT
# ====================================================================
def save_confusion_matrix_plot(cm: np.ndarray, output_path: str) -> None:
    """Generate and save confusion matrix heatmap image."""
    logger.info("Generating confusion matrix plot...")
    plt.figure(figsize=(7, 6))
    labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True
    )
    plt.title("Confusion Matrix - Hate Speech Detection (LinearSVC)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("Actual Class", fontsize=11, labelpad=8)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix image saved to: {output_path}")

# ====================================================================
# SECTION 32 & 33 — MISCLASSIFICATION & SAMPLE PREDICTION ANALYSIS
# ====================================================================
def analyze_misclassifications_and_samples(X_test: pd.Series, y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Display sample test set predictions and inspect misclassified examples."""
    test_df = pd.DataFrame({
        "tweet": X_test.values,
        "actual_class": y_test.values,
        "predicted_class": y_pred
    })

    test_df["actual_label"] = test_df["actual_class"].map(CLASS_NAMES)
    test_df["predicted_label"] = test_df["predicted_class"].map(CLASS_NAMES)
    test_df["correct"] = test_df["actual_class"] == test_df["predicted_class"]

    print("\nSample Test Set Predictions (First 5):")
    print("-" * 60)
    for i, row in test_df.head(5).iterrows():
        status = "CORRECT" if row["correct"] else "MISCLASSIFIED"
        print(f"[{status}] Actual: {row['actual_label']} | Pred: {row['predicted_label']}")
        print(f" Tweet: {row['tweet'][:80]}...\n")

    misclassified = test_df[~test_df["correct"]]
    logger.info(f"Total Misclassifications in test set: {len(misclassified)} / {len(test_df)}")

    print("\nSample Misclassifications (First 5):")
    print("-" * 60)
    for i, row in misclassified.head(5).iterrows():
        print(f"Actual: {row['actual_label']} --> Predicted: {row['predicted_label']}")
        print(f"Tweet : {row['tweet'][:80]}...\n")

# ====================================================================
# SECTION 34 — CUSTOM TEXT PREDICTION FUNCTION
# ====================================================================
def predict_custom_text(text: str, model: LinearSVC, vectorizer: TfidfVectorizer) -> str:
    """Preprocess, vectorise, and classify custom input string."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred_class = model.predict(vec)[0]
    return CLASS_NAMES.get(pred_class, f"Class {pred_class}")

def test_custom_examples(model: LinearSVC, vectorizer: TfidfVectorizer) -> None:
    """Test custom neutral sentences to verify prediction pipeline works on arbitrary inputs."""
    logger.info("Running custom test sentence evaluations...")
    sample_texts = [
        "I really enjoyed reading this book today.",
        "The weather is very pleasant outside.",
        "This service was extremely disappointing and slow."
    ]

    print("\nCustom Inference Test Output:")
    print("-" * 60)
    for text in sample_texts:
        predicted_label = predict_custom_text(text, model, vectorizer)
        print(f" Input Text : \"{text}\"")
        print(f" Prediction : {predicted_label}\n")

# ====================================================================
# SECTION 35, 36, 37 — SAVING & VALIDATING ARTIFACTS
# ====================================================================
def save_model_artifacts(model: LinearSVC, vectorizer: TfidfVectorizer) -> None:
    """Serialize model and TF-IDF vectorizer using pickle."""
    logger.info(f"Saving SVM model to: {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Saving TF-IDF vectorizer to: {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

def save_metrics_and_metadata(metrics: dict) -> None:
    """Save metrics.json and model_metadata.json with calculated pipeline outputs."""
    metrics_data = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"]
    }

    logger.info(f"Saving metrics JSON to: {METRICS_PATH}")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)

    metadata = {
        "algorithm": "LinearSVC",
        "feature_extraction": "TF-IDF",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "max_features": TFIDF_MAX_FEATURES,
        "ngram_range": list(TFIDF_NGRAM_RANGE),
        "svm_c": SVM_C,
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"]
    }

    logger.info(f"Saving metadata JSON to: {METADATA_PATH}")
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

def validate_saved_artifacts() -> None:
    """Verify all generated artifact files exist on disk."""
    artifacts = [
        MODEL_PATH,
        VECTORIZER_PATH,
        METADATA_PATH,
        METRICS_PATH,
        CLASS_DIST_PLOT_PATH,
        CONFUSION_MATRIX_PLOT_PATH
    ]

    missing = [art for art in artifacts if not os.path.exists(art)]
    if missing:
        raise FileNotFoundError(f"Missing required output artifacts: {missing}")

    logger.info("All model artifacts and result plots validated successfully on disk.")

# ====================================================================
# SECTION 38 — TRAINING SUMMARY
# ====================================================================
def print_training_summary(
    df: pd.DataFrame,
    X_train: pd.Series,
    X_test: pd.Series,
    vocab_size: int,
    metrics: dict
) -> None:
    """Display final summary banner of completed ML training execution."""
    summary_banner = f"""
============================================================
HATE SPEECH DETECTION - TRAINING SUMMARY
============================================================

Dataset:
    {DATA_PATH}

Input Feature:
    tweet (text)

Target Variable:
    class (0: Hate Speech, 1: Offensive Language, 2: Neither)

Algorithm:
    Linear Support Vector Machine (LinearSVC)

Feature Extraction:
    TF-IDF (max_features={TFIDF_MAX_FEATURES}, ngrams={TFIDF_NGRAM_RANGE})

Train Samples:
    {len(X_train)}

Test Samples:
    {len(X_test)}

TF-IDF Vocabulary Size:
    {vocab_size}

Performance Metrics (Calculated):
    Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
    Precision : {metrics['precision']:.4f}
    Recall    : {metrics['recall']:.4f}
    F1-Score  : {metrics['f1_score']:.4f}

Saved Model:
    {MODEL_PATH}

Saved Vectorizer:
    {VECTORIZER_PATH}

Saved Metrics:
    {METRICS_PATH}

============================================================
TRAINING COMPLETED SUCCESSFULLY
============================================================
"""
    print(summary_banner)

# ====================================================================
# SECTION 41 — MAIN PIPELINE EXECUTION
# ====================================================================
def main() -> None:
    """Main execution entrypoint for Hate Speech Detection ML Training Pipeline."""
    logger.info("Starting Hate Speech Detection ML Training Pipeline...")

    try:
        # Step 1: Load and Validate Dataset
        df_raw = load_dataset(DATA_PATH)
        validate_dataset(df_raw)
        display_dataset_overview(df_raw)

        # Step 2: Clean and Explore Dataset
        df_clean_na = analyze_missing_values(df_raw)
        df_dedup = remove_duplicates(df_clean_na)
        analyze_and_plot_class_distribution(df_dedup, CLASS_DIST_PLOT_PATH)

        # Step 3: Text Preprocessing & Stats
        df_preprocessed = preprocess_text(df_dedup)
        analyze_text_statistics(df_preprocessed)

        # Step 4: Train / Test Split
        X = df_preprocessed["clean_tweet"]
        y = df_preprocessed["class"]
        X_train, X_test, y_train, y_test = perform_train_test_split(X, y)

        # Step 5: TF-IDF Feature Extraction
        vectorizer, X_train_tfidf, X_test_tfidf = extract_tfidf_features(X_train, X_test)

        # Step 6: Train LinearSVC Model
        model = train_linear_svc(X_train_tfidf, y_train)

        # Step 7: Evaluate Model
        metrics = evaluate_model(model, X_test_tfidf, y_test)
        save_confusion_matrix_plot(metrics["confusion_matrix"], CONFUSION_MATRIX_PLOT_PATH)

        # Step 8: Misclassification & Custom Inference Testing
        analyze_misclassifications_and_samples(X_test, y_test, metrics["y_pred"])
        test_custom_examples(model, vectorizer)

        # Step 9: Save Model Artifacts & Output Validation
        save_model_artifacts(model, vectorizer)
        save_metrics_and_metadata(metrics)
        validate_saved_artifacts()

        # Step 10: Print Final Academic Training Summary
        print_training_summary(df_preprocessed, X_train, X_test, len(vectorizer.vocabulary_), metrics)

    except Exception as error:
        logger.exception(f"Pipeline execution failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
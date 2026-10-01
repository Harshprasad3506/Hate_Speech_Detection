# Hate Speech Detection using Support Vector Machine (SVM)

A college mini-project for classifying text into:

| Label | Class |
|-------|-------|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | Neither |

---

## Objective

Build a machine learning model that reads a tweet and classifies it into one of the three categories above using **TF-IDF** for feature extraction and **LinearSVC** as the classifier.

---

## Technologies

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF, LinearSVC, metrics |
| Matplotlib | Plotting |
| Seaborn | Visualization |
| Jupyter Notebook | Main implementation |

---

## Machine Learning Algorithm

**Support Vector Machine** using `LinearSVC`

LinearSVC is well-suited for text classification because TF-IDF produces high-dimensional sparse matrices, and linear SVM handles these efficiently.

---

## Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)**

- Converts text into numerical vectors.
- Reduces the influence of very common words.
- Configured with unigrams and bigrams, top 10,000 features.

---

## Evaluation Metrics

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
- Confusion Matrix

---

## Project Workflow

```
Dataset (labeled_data.csv)
    → Data Understanding
    → Data Quality Check
    → Class Distribution Analysis
    → Feature / Target Selection  (tweet → class)
    → Text Preprocessing          (clean_text)
    → Train-Test Split            (80 / 20, stratified)
    → TF-IDF Vectorization        (fit on train only)
    → LinearSVC Training
    → Predictions on Test Set
    → Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
    → Custom Sentence Prediction
    → Save Model + TF-IDF Vectorizer
```

---

## Project Structure

```
Hate_Speech_Detection/
│
├── data/
│   └── labeled_data_actual.csv   ← dataset (24,783 tweets)
│
├── notebooks/
│   └── hate_speech_svm.ipynb     ← main implementation (run this)
│
├── models/
│   ├── svm_model.pkl             ← saved LinearSVC
│   └── tfidf_vectorizer.pkl      ← saved TF-IDF vectorizer
│
├── src/
│   ├── preprocessing.py          ← clean_text() utility
│   ├── train.py                  ← standalone training script
│   └── predict.py                ← standalone prediction script
│
├── results/
│   ├── confusion_matrix.png      ← generated after running notebook
│   └── class_distribution.png   ← generated after running notebook
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook (recommended)

Open and run all cells in:

```
notebooks/hate_speech_svm.ipynb
```

### 3. Or run the training script

```bash
python src/train.py
```

### 4. Predict on custom text

```bash
python src/predict.py
```

Or in Python:

```python
from src.predict import predict_text
print(predict_text("This is a nice day!"))
```

---

## Dataset

- **File:** `data/labeled_data_actual.csv`
- **Source:** Davidson et al. (2017) — Hate Speech and Offensive Language dataset
- **Rows:** 24,783 labeled tweets
- **Columns used:** `tweet` (input), `class` (target)
- **No missing values**

---

> **Note:** The model is trained on real social media data. Results reflect actual model performance on the test set — no values are hard-coded.

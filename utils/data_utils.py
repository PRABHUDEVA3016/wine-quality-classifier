"""
Data Utility Functions
Handles loading, preprocessing, and exploration of the Wine Quality Dataset.
"""

import pandas as pd
import numpy as np
import os
import urllib.request

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DATASET_PATH = os.path.join("dataset", "winequality-red.csv")

COLUMN_DESCRIPTIONS = {
    "fixed acidity":        "Most acids involved with wine (tartaric acid). Measured in g/dm³.",
    "volatile acidity":     "Amount of acetic acid (too high → vinegar taste). Measured in g/dm³.",
    "citric acid":          "Adds freshness and flavour to wines. Measured in g/dm³.",
    "residual sugar":       "Amount of sugar remaining after fermentation. Measured in g/dm³.",
    "chlorides":            "Amount of salt in the wine. Measured in g/dm³.",
    "free sulfur dioxide":  "Free SO₂ that prevents microbial growth. Measured in mg/dm³.",
    "total sulfur dioxide": "Total SO₂ (free + bound). Measured in mg/dm³.",
    "density":              "Density of the wine relative to water (g/cm³).",
    "pH":                   "Describes acidity/basicity on a scale from 0–14.",
    "sulphates":            "Wine additive that contributes to SO₂. Measured in g/dm³.",
    "alcohol":              "Percentage of alcohol content (% vol).",
    "quality":              "Output variable scored between 0–10 by wine experts.",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────
def download_dataset() -> str:
    """Download the dataset if it does not exist locally. Returns the path."""
    os.makedirs("dataset", exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download dataset from UCI repository.\nError: {exc}"
            )
    return DATASET_PATH


def load_data() -> pd.DataFrame:
    """Load Wine Quality CSV and return a clean DataFrame."""
    path = download_dataset()
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()  # strip any whitespace in column names
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset.
    - Handle missing values (fill with median).
    - Create binary target: quality >= 7 → Good (1), else Bad (0).
    - Split into features (X) and target (y).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    df_processed : pd.DataFrame  (full dataframe with 'quality_label' column)
    """
    df = df.copy()

    # ── Handle missing values ────────────────────────────────────────────────
    if df.isnull().sum().sum() > 0:
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)

    # ── Binary classification label ──────────────────────────────────────────
    df["quality_label"] = (df["quality"] >= 7).astype(int)  # 1 = Good, 0 = Bad

    # ── Features & Target ────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in ("quality", "quality_label")]
    X = df[feature_cols]
    y = df["quality_label"]

    return X, y, df


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Info Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_dataset_info(df: pd.DataFrame) -> dict:
    """Return a summary dictionary for the dataset."""
    return {
        "rows":          len(df),
        "columns":       len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "duplicates":    df.duplicated().sum(),
        "quality_dist":  df["quality"].value_counts().sort_index().to_dict(),
    }


def get_feature_names(df: pd.DataFrame) -> list:
    """Return only the feature column names (excluding target columns)."""
    return [c for c in df.columns if c not in ("quality", "quality_label")]

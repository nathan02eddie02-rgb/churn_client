"""
preprocessing.py
-----------------
Nettoyage et préparation du dataset Telco Customer Churn (IBM / Kaggle).
- Nettoyage des valeurs manquantes
- Encodage des variables catégorielles
- Normalisation des variables numériques (StandardScaler)
- Split train/test stratifié
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"


def load_raw_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # TotalCharges contient des chaines vides pour les nouveaux clients (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # SeniorCitizen est déjà 0/1, on le laisse tel quel
    # On retire l'identifiant, non prédictif
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def encode_features(df, fit=True, encoders=None):
    """
    Encode les colonnes catégorielles en variables numériques.
    - Colonnes binaires (Yes/No, Male/Female) -> LabelEncoder
    - Colonnes multi-catégories -> one-hot encoding (pd.get_dummies)
    """
    df = df.copy()
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    binary_cols = [
        c for c in text_cols
        if df[c].nunique() == 2 and c != TARGET_COL
    ]
    multi_cols = [
        c for c in text_cols
        if df[c].nunique() > 2 and c != TARGET_COL
    ]

    if encoders is None:
        encoders = {}

    for col in binary_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col])

    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Encodage de la cible
    if TARGET_COL in df.columns and df[TARGET_COL].isin(["Yes", "No"]).any():
        df[TARGET_COL] = df[TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)

    return df, encoders


def scale_features(df, fit=True, scaler=None, numeric_cols=NUMERIC_COLS):
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df, scaler


def build_dataset(path=DATA_PATH, test_size=0.2, random_state=42):
    """
    Pipeline complet : charge, nettoie, encode, normalise, split.
    Retourne X_train, X_test, y_train, y_test, encoders, scaler, feature_names
    """
    raw = load_raw_data(path)
    clean = clean_data(raw)

    encoded, encoders = encode_features(clean, fit=True)

    y = encoded[TARGET_COL]
    X = encoded.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, scaler = scale_features(X_train, fit=True)
    X_test_scaled, _ = scale_features(X_test, fit=False, scaler=scaler)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_names.joblib"))

    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler, list(X.columns)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoders, scaler, feats = build_dataset()
    print("Dimensions X_train:", X_train.shape)
    print("Dimensions X_test :", X_test.shape)
    print("Taux de churn (train):", y_train.mean().round(3))
    print("Taux de churn (test) :", y_test.mean().round(3))
    print("Nombre de features encodées:", len(feats))

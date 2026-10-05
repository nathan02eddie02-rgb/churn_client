"""data.py - chargement mis en cache des données, résultats et modèle."""

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocessing import load_raw_data, load_artifacts, prepare_input  # noqa: E402
from business_simulation import get_business_dataframe, run_simulation  # noqa: E402

OUTPUTS = os.path.join(ROOT, "outputs")
MODELS = os.path.join(ROOT, "models")

# Libellés français pour les variables encodées
PREFIX_FR = {
    "MultipleLines": "Lignes multiples", "InternetService": "Internet", "OnlineSecurity": "Sécurité en ligne",
    "OnlineBackup": "Sauvegarde en ligne", "DeviceProtection": "Protection appareil",
    "TechSupport": "Support technique", "StreamingTV": "TV en streaming", "StreamingMovies": "Films en streaming",
    "Contract": "Contrat", "PaymentMethod": "Paiement",
}
VALUE_FR = {
    "Yes": "oui", "No": "non", "No internet service": "sans internet", "No phone service": "sans téléphone",
    "Fiber optic": "fibre", "DSL": "DSL", "One year": "1 an", "Two year": "2 ans", "Month-to-month": "mensuel",
    "Electronic check": "chèque électronique", "Mailed check": "chèque postal",
    "Bank transfer (automatic)": "virement automatique", "Credit card (automatic)": "carte automatique",
}
SIMPLE_FR = {
    "tenure": "Ancienneté", "MonthlyCharges": "Facture mensuelle", "TotalCharges": "Total facturé",
    "SeniorCitizen": "Client senior", "Partner": "En couple", "Dependents": "Personnes à charge",
    "PhoneService": "Service téléphone", "PaperlessBilling": "Facture dématérialisée", "gender": "Genre",
}


def feature_label(name):
    if name in SIMPLE_FR:
        return SIMPLE_FR[name]
    if "_" in name:
        prefix, value = name.split("_", 1)
        if prefix in PREFIX_FR:
            return f"{PREFIX_FR[prefix]} : {VALUE_FR.get(value, value)}"
    return name


@st.cache_data(show_spinner=False)
def portfolio():
    """Les 7 043 clients au format brut, avec TotalCharges numérique et cible 0/1."""
    df = load_raw_data()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_data(show_spinner=False)
def model_results():
    comp = pd.read_csv(os.path.join(OUTPUTS, "comparaison_modeles.csv"))
    with open(os.path.join(MODELS, "best_model_name.json"), encoding="utf-8") as f:
        best = json.load(f)["best_model"]
    return comp, best


@st.cache_data(show_spinner="Calcul du portefeuille de test…")
def business():
    return get_business_dataframe()


@st.cache_data(show_spinner=False)
def importances():
    imp = pd.read_csv(os.path.join(OUTPUTS, "importance_variables.csv"), index_col=0).iloc[:, 0]
    return imp.sort_values(ascending=False)


@st.cache_data(show_spinner=False)
def smote_counts():
    """Effectifs du jeu d'entraînement avant/après SMOTE (même split que l'entraînement)."""
    y = portfolio()["Churn_bin"]
    y_train, _ = train_test_split(y, test_size=0.2, random_state=42, stratify=y)
    n_stay, n_churn = int((y_train == 0).sum()), int((y_train == 1).sum())
    return {"stay": n_stay, "churn": n_churn, "total": len(y_train)}


@st.cache_resource(show_spinner=False)
def artifacts():
    return load_artifacts()


def score(df_raw):
    """Probabilités de churn pour des clients au format brut."""
    model, encoders, scaler, feature_names = artifacts()
    X = prepare_input(df_raw, encoders, scaler, feature_names)
    return model.predict_proba(X)[:, 1], X


def contributions(X_row):
    """Contribution de chaque variable au score (SHAP via XGBoost), triée par poids absolu."""
    try:
        import xgboost as xgb
        model = artifacts()[0]
        c = model.get_booster().predict(xgb.DMatrix(X_row), pred_contribs=True)[0][:-1]
        s = pd.Series(c, index=X_row.columns)
        return s.reindex(s.abs().sort_values(ascending=False).index)
    except Exception:
        return None

"""
train_models.py
-----------------
Entraîne 4 modèles de classification pour prédire le churn client,
gère le déséquilibre des classes avec SMOTE, compare les performances
et sauvegarde le meilleur modèle.

Modèles comparés :
    1. Régression Logistique
    2. Random Forest
    3. XGBoost
    4. Support Vector Machine (SVM)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

from preprocessing import build_dataset

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_models():
    return {
        "Regression Logistique": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            random_state=42, eval_metric="logloss", n_jobs=-1
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modele": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def main():
    print("=== 1. Preparation des donnees ===")
    X_train, X_test, y_train, y_test, encoders, scaler, feature_names = build_dataset()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Taux de churn avant SMOTE (train): {y_train.mean():.3f}")

    print("\n=== 2. Gestion du desequilibre des classes (SMOTE) ===")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"Distribution avant SMOTE: {y_train.value_counts().to_dict()}")
    print(f"Distribution apres SMOTE : {pd.Series(y_train_bal).value_counts().to_dict()}")

    print("\n=== 3. Entrainement et comparaison des 4 modeles ===")
    models = get_models()
    results = []
    predictions = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"-> Entrainement : {name}")
        model.fit(X_train_bal, y_train_bal)
        metrics, y_pred, y_proba = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        predictions[name] = {"y_pred": y_pred, "y_proba": y_proba}
        fitted_models[name] = model
        print(f"   Accuracy={metrics['Accuracy']:.3f}  F1={metrics['F1-score']:.3f}  ROC-AUC={metrics['ROC-AUC']:.3f}")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "comparaison_modeles.csv"), index=False)
    print("\n=== Tableau comparatif (trie par ROC-AUC) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Modele"]
    best_model = fitted_models[best_name]
    print(f"\n>>> Meilleur modele retenu : {best_name}")

    # --- Sauvegarde du meilleur modele ---
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    with open(os.path.join(MODELS_DIR, "best_model_name.json"), "w") as f:
        json.dump({"best_model": best_name}, f)

    # --- Rapport de classification detaille du meilleur modele ---
    report = classification_report(y_test, predictions[best_name]["y_pred"], target_names=["Reste", "Churn"])
    with open(os.path.join(OUTPUTS_DIR, "rapport_classification_meilleur_modele.txt"), "w") as f:
        f.write(f"Meilleur modele : {best_name}\n\n")
        f.write(report)

    # --- Graphique 1 : comparaison des modeles ---
    plt.figure(figsize=(9, 5))
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    plot_df = results_df.set_index("Modele")[metrics_to_plot]
    plot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Comparaison des 4 modeles de prediction du churn")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "comparaison_modeles.png"), dpi=150)
    plt.close()

    # --- Graphique 2 : courbes ROC ---
    plt.figure(figsize=(7, 6))
    for name in models.keys():
        fpr, tpr, _ = roc_curve(y_test, predictions[name]["y_proba"])
        auc = results_df.loc[results_df["Modele"] == name, "ROC-AUC"].values[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aleatoire")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC - comparaison des modeles")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "courbes_roc.png"), dpi=150)
    plt.close()

    # --- Graphique 3 : matrice de confusion du meilleur modele ---
    cm = confusion_matrix(y_test, predictions[best_name]["y_pred"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Reste", "Churn"], yticklabels=["Reste", "Churn"])
    plt.title(f"Matrice de confusion - {best_name}")
    plt.ylabel("Reel")
    plt.xlabel("Predit")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "matrice_confusion.png"), dpi=150)
    plt.close()

    # --- Graphique 4 : importance des variables (si dispo) ---
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        importances.sort_values().plot(kind="barh", color="#2E86AB")
        plt.title(f"Importance des variables - {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, "importance_variables.png"), dpi=150)
        plt.close()
        importances.to_csv(os.path.join(OUTPUTS_DIR, "importance_variables.csv"))

    # --- Sauvegarde des predictions sur le jeu de test (pour le dashboard business) ---
    test_export = X_test.copy()
    test_export["Churn_reel"] = y_test.values
    test_export["Churn_proba"] = predictions[best_name]["y_proba"]
    test_export["Churn_predit"] = predictions[best_name]["y_pred"]
    test_export.to_csv(os.path.join(OUTPUTS_DIR, "predictions_test_set.csv"), index=False)

    print("\n=== Termine. Fichiers generes dans outputs/ et models/ ===")
    return results_df, best_name


if __name__ == "__main__":
    main()

"""
business_simulation.py
------------------------
Scenario reel : l'entreprise fictive "NexTel" utilise le modele de churn
pour cibler une campagne de retention client, et etablit un bilan
financier (argent depense vs argent recupere / profit genere).

Hypotheses de la simulation (modifiables dans SIMULATION_PARAMS) :
- Cout de contact/retention par client cible par la campagne
- Taux de succes de la campagne (probabilite qu'un client reellement
  a risque, une fois contacte, soit finalement retenu)
- Horizon de valeur (nombre de mois de revenu recupere si le client
  reste client grace a la campagne)
- Seuil de probabilite de churn au-dela duquel un client est cible
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import build_dataset, load_raw_data, clean_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

SIMULATION_PARAMS = {
    "seuil_probabilite": 0.50,      # seuil de probabilite pour cibler un client
    "cout_par_contact": 15,          # cout ($) de l'offre/remise/appel de retention par client cible
    "taux_succes_campagne": 0.35,    # 35% des vrais clients a risque contactes sont effectivement retenus
    "horizon_mois": 12,               # valeur de revenu recuperee sur 12 mois si le client reste
}


def get_business_dataframe():
    """Reconstruit le jeu de test avec les montants originaux (non normalises)
    et les probabilites de churn predites par le meilleur modele."""
    X_train, X_test, y_train, y_test, encoders, scaler, feature_names = build_dataset()

    raw = load_raw_data()
    clean = clean_data(raw)

    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    proba = best_model.predict_proba(X_test)[:, 1]
    pred = best_model.predict(X_test)

    business_df = clean.loc[X_test.index, ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]].copy()
    business_df["Churn_reel"] = y_test.values
    business_df["Churn_proba"] = proba
    business_df["Churn_predit"] = pred

    return business_df


def run_simulation(business_df, params=SIMULATION_PARAMS):
    seuil = params["seuil_probabilite"]
    cout_contact = params["cout_par_contact"]
    taux_succes = params["taux_succes_campagne"]
    horizon = params["horizon_mois"]

    n_total = len(business_df)
    taux_churn_global = business_df["Churn_reel"].mean()

    cibles = business_df[business_df["Churn_proba"] >= seuil].copy()
    n_cibles = len(cibles)

    # Parmi les clients cibles, combien etaient reellement a risque (vrais positifs)
    vrais_positifs = cibles[cibles["Churn_reel"] == 1]
    faux_positifs = cibles[cibles["Churn_reel"] == 0]

    n_vrais_positifs = len(vrais_positifs)
    n_faux_positifs = len(faux_positifs)

    # --- Scenario AVEC modele (ciblage intelligent) ---
    budget_campagne = n_cibles * cout_contact
    clients_sauves = n_vrais_positifs * taux_succes
    # revenu recupere = somme des (MonthlyCharges * horizon) ponderee par le taux de succes,
    # calcule sur les vrais positifs cibles
    revenu_recupere = (vrais_positifs["MonthlyCharges"] * horizon).sum() * taux_succes
    profit_net = revenu_recupere - budget_campagne
    roi_pct = (profit_net / budget_campagne * 100) if budget_campagne > 0 else 0

    # --- Scenario SANS modele (ciblage aleatoire, meme budget / meme nombre de clients contactes) ---
    # Si l'entreprise contacte le meme nombre de clients au hasard, elle atteint
    # les vrais a risque au taux de base (taux de churn global) et non au taux de precision du modele.
    n_vrais_positifs_aleatoire = n_cibles * taux_churn_global
    charge_moyenne = business_df["MonthlyCharges"].mean()
    revenu_recupere_aleatoire = n_vrais_positifs_aleatoire * charge_moyenne * horizon * taux_succes
    budget_aleatoire = n_cibles * cout_contact  # meme budget engage
    profit_net_aleatoire = revenu_recupere_aleatoire - budget_aleatoire
    roi_pct_aleatoire = (profit_net_aleatoire / budget_aleatoire * 100) if budget_aleatoire > 0 else 0

    resultats = {
        "parametres": params,
        "n_clients_total": int(n_total),
        "taux_churn_global": round(float(taux_churn_global), 4),
        "n_clients_cibles": int(n_cibles),
        "n_vrais_positifs": int(n_vrais_positifs),
        "n_faux_positifs": int(n_faux_positifs),
        "precision_ciblage_modele": round(n_vrais_positifs / n_cibles, 4) if n_cibles else 0,
        "avec_modele": {
            "budget_campagne_usd": round(float(budget_campagne), 2),
            "clients_sauves_estimes": round(float(clients_sauves), 1),
            "revenu_recupere_usd": round(float(revenu_recupere), 2),
            "profit_net_usd": round(float(profit_net), 2),
            "roi_pct": round(float(roi_pct), 1),
        },
        "sans_modele_ciblage_aleatoire": {
            "budget_campagne_usd": round(float(budget_aleatoire), 2),
            "clients_sauves_estimes": round(float(n_vrais_positifs_aleatoire * taux_succes), 1),
            "revenu_recupere_usd": round(float(revenu_recupere_aleatoire), 2),
            "profit_net_usd": round(float(profit_net_aleatoire), 2),
            "roi_pct": round(float(roi_pct_aleatoire), 1),
        },
        "gain_efficacite_modele_vs_aleatoire_usd": round(float(profit_net - profit_net_aleatoire), 2),
    }
    return resultats


def main():
    business_df = get_business_dataframe()
    business_df.to_csv(os.path.join(OUTPUTS_DIR, "business_dataframe.csv"), index=False)

    resultats = run_simulation(business_df)

    with open(os.path.join(OUTPUTS_DIR, "bilan_business.json"), "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)

    print("=== BILAN DE LA CAMPAGNE DE RETENTION - NexTel ===\n")
    print(f"Clients total (echantillon test) : {resultats['n_clients_total']}")
    print(f"Taux de churn observe            : {resultats['taux_churn_global']*100:.1f}%")
    print(f"Clients cibles par la campagne    : {resultats['n_clients_cibles']}")
    print(f"Precision du ciblage (vrais/total): {resultats['precision_ciblage_modele']*100:.1f}%\n")

    print(">>> AVEC le modele de churn :")
    for k, v in resultats["avec_modele"].items():
        print(f"   {k}: {v}")

    print("\n>>> SANS modele (ciblage aleatoire, meme budget) :")
    for k, v in resultats["sans_modele_ciblage_aleatoire"].items():
        print(f"   {k}: {v}")

    print(f"\n>>> Gain d'efficacite du modele : {resultats['gain_efficacite_modele_vs_aleatoire_usd']} USD supplementaires de profit net")

    # --- Graphique comparatif ---
    labels = ["Budget campagne", "Revenu recupere", "Profit net"]
    avec = [resultats["avec_modele"]["budget_campagne_usd"],
            resultats["avec_modele"]["revenu_recupere_usd"],
            resultats["avec_modele"]["profit_net_usd"]]
    sans = [resultats["sans_modele_ciblage_aleatoire"]["budget_campagne_usd"],
            resultats["sans_modele_ciblage_aleatoire"]["revenu_recupere_usd"],
            resultats["sans_modele_ciblage_aleatoire"]["profit_net_usd"]]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, avec, width, label="Avec modele ML", color="#2E86AB")
    plt.bar(x + width/2, sans, width, label="Sans modele (aleatoire)", color="#C0C0C0")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x, labels)
    plt.ylabel("Montant (USD)")
    plt.title("Impact financier : ciblage par ML vs ciblage aleatoire")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "bilan_business.png"), dpi=150)
    plt.close()

    return resultats


if __name__ == "__main__":
    main()

"""
app.py - Dashboard Streamlit du projet Churn Client "NexTel"
================================================================
Lancement :  streamlit run app.py   (depuis le dossier dashboard/)

Ce dashboard presente :
  1. Les KPI globaux du portefeuille client
  2. La comparaison des 4 modeles de ML entraines
  3. Le bilan financier interactif de la campagne de retention
     (curseurs pour ajuster le seuil de ciblage, le cout de contact,
     le taux de succes de la campagne et l'horizon de valeur)
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import build_dataset, load_raw_data, clean_data  # noqa: E402
from business_simulation import get_business_dataframe, run_simulation  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(page_title="Dashboard Churn - NexTel", layout="wide", page_icon="📊")


@st.cache_data
def load_all():
    comparaison = pd.read_csv(os.path.join(OUTPUTS_DIR, "comparaison_modeles.csv"))
    business_df = get_business_dataframe()
    with open(os.path.join(MODELS_DIR, "best_model_name.json")) as f:
        best_name = json.load(f)["best_model"]
    return comparaison, business_df, best_name


comparaison, business_df, best_name = load_all()

st.title("📊 Dashboard Churn Client — NexTel")
st.caption("Prediction du churn, comparaison de modeles ML et bilan financier d'une campagne de retention")

# ---------------------------------------------------------------
# SECTION 1 - KPI PORTEFEUILLE
# ---------------------------------------------------------------
st.header("1. Vue d'ensemble du portefeuille client")

n_total = len(business_df)
taux_churn = business_df["Churn_reel"].mean()
charges_moyennes = business_df["MonthlyCharges"].mean()
revenu_mensuel_total = business_df["MonthlyCharges"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clients (echantillon test)", f"{n_total:,}")
c2.metric("Taux de churn observe", f"{taux_churn*100:.1f}%")
c3.metric("Charge mensuelle moyenne", f"${charges_moyennes:.2f}")
c4.metric("Revenu mensuel du portefeuille", f"${revenu_mensuel_total:,.0f}")

st.divider()

# ---------------------------------------------------------------
# SECTION 2 - COMPARAISON DES MODELES
# ---------------------------------------------------------------
st.header("2. Comparaison des 4 modeles de Machine Learning")
st.markdown(f"**Modele retenu (le plus performant) : `{best_name}`**")

st.dataframe(
    comparaison.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"], color="#d4f4dd"),
    use_container_width=True,
)

fig_models = go.Figure()
for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
    fig_models.add_trace(go.Bar(name=metric, x=comparaison["Modele"], y=comparaison[metric]))
fig_models.update_layout(barmode="group", yaxis_range=[0, 1], title="Scores par modele")
st.plotly_chart(fig_models, use_container_width=True)

st.divider()

# ---------------------------------------------------------------
# SECTION 3 - SIMULATION BUSINESS INTERACTIVE
# ---------------------------------------------------------------
st.header("3. Bilan financier de la campagne de retention")
st.markdown(
    "Ajustez les hypotheses de la campagne pour voir l'impact sur le budget, "
    "le revenu recupere et le profit net genere grace au ciblage par le modele."
)

col_a, col_b = st.columns([1, 2])

with col_a:
    seuil = st.slider("Seuil de probabilite de churn pour cibler un client", 0.1, 0.9, 0.5, 0.05)
    cout_contact = st.number_input("Cout de contact/retention par client ($)", min_value=1, max_value=200, value=15)
    taux_succes = st.slider("Taux de succes de la campagne (%)", 5, 80, 35, 5) / 100
    horizon = st.slider("Horizon de valeur recuperee (mois)", 1, 24, 12, 1)

params = {
    "seuil_probabilite": seuil,
    "cout_par_contact": cout_contact,
    "taux_succes_campagne": taux_succes,
    "horizon_mois": horizon,
}
resultats = run_simulation(business_df, params)

with col_b:
    k1, k2, k3 = st.columns(3)
    k1.metric("Clients cibles", resultats["n_clients_cibles"])
    k2.metric("Precision du ciblage", f"{resultats['precision_ciblage_modele']*100:.1f}%")
    k3.metric("Clients sauves (estimation)", resultats["avec_modele"]["clients_sauves_estimes"])

    k4, k5, k6 = st.columns(3)
    k4.metric("Budget campagne", f"${resultats['avec_modele']['budget_campagne_usd']:,.0f}")
    k5.metric("Revenu recupere", f"${resultats['avec_modele']['revenu_recupere_usd']:,.0f}")
    k6.metric(
        "Profit net",
        f"${resultats['avec_modele']['profit_net_usd']:,.0f}",
        delta=f"ROI {resultats['avec_modele']['roi_pct']:.0f}%",
    )

st.subheader("Avec modele ML vs sans modele (ciblage aleatoire, meme budget)")
labels = ["Budget campagne", "Revenu recupere", "Profit net"]
avec = [resultats["avec_modele"][k] for k in ["budget_campagne_usd", "revenu_recupere_usd", "profit_net_usd"]]
sans = [resultats["sans_modele_ciblage_aleatoire"][k] for k in ["budget_campagne_usd", "revenu_recupere_usd", "profit_net_usd"]]

fig_bilan = go.Figure()
fig_bilan.add_trace(go.Bar(name="Avec modele ML", x=labels, y=avec, marker_color="#2E86AB"))
fig_bilan.add_trace(go.Bar(name="Sans modele (aleatoire)", x=labels, y=sans, marker_color="#C0C0C0"))
fig_bilan.update_layout(barmode="group", title="Impact financier compare")
st.plotly_chart(fig_bilan, use_container_width=True)

st.success(
    f"💡 Grace au ciblage par le modele, l'entreprise degage "
    f"**{resultats['gain_efficacite_modele_vs_aleatoire_usd']:,.0f} USD** de profit net supplementaire "
    f"par rapport a une campagne de retention non ciblee, pour un budget identique."
)

st.divider()

# ---------------------------------------------------------------
# SECTION 4 - DISTRIBUTION DU RISQUE
# ---------------------------------------------------------------
st.header("4. Distribution du risque de churn dans le portefeuille")
fig_hist = px.histogram(
    business_df, x="Churn_proba", color=business_df["Churn_reel"].map({0: "Reste", 1: "Churn reel"}),
    nbins=30, barmode="overlay", opacity=0.7,
    labels={"color": "Statut reel"},
    title="Distribution des probabilites de churn predites",
)
fig_hist.add_vline(x=seuil, line_dash="dash", line_color="red", annotation_text="Seuil de ciblage")
st.plotly_chart(fig_hist, use_container_width=True)

st.caption("Projet churn de A a Z — donnees : Telco Customer Churn (IBM/Kaggle) — dashboard genere en Python (Streamlit + Plotly)")

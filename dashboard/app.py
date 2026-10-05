"""
app.py - Dashboard Streamlit : prédiction du churn client (NexTel)
Lancement, depuis le dossier dashboard/ :   streamlit run app.py

Quatre sections : Problématique, Dataset, Visualisations KPI, Prédictions.
"""

import streamlit as st

st.set_page_config(page_title="Churn client - NexTel", page_icon=":material/monitoring:", layout="wide")

import data  # noqa: E402
import theme  # noqa: E402
from views import problematique, dataset, kpi, predictions  # noqa: E402

theme.inject_css()

comp, best = data.model_results()
with st.sidebar:
    st.markdown('<div class="brand">Churn client</div><div class="brand-sub">NexTel, opérateur télécom (cas simulé)</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="fiche"><b>{best}</b><span>modèle retenu parmi 4</span></div>'
                f'<div class="fiche"><b>{comp.iloc[0]["ROC-AUC"]:.3f}</b><span>ROC-AUC sur le jeu de test</span></div>'
                f'<div class="fiche"><b>7 043</b><span>clients, jeu Telco Customer Churn (IBM, Kaggle)</span></div>',
                unsafe_allow_html=True)

pages = [
    st.Page(problematique.render, title="Problématique", icon=":material/help:", url_path="problematique", default=True),
    st.Page(dataset.render, title="Dataset", icon=":material/database:", url_path="dataset"),
    st.Page(kpi.render, title="Visualisations KPI", icon=":material/monitoring:", url_path="kpi"),
    st.Page(predictions.render, title="Prédictions", icon=":material/online_prediction:", url_path="predictions"),
]
st.navigation(pages, position="top").run()

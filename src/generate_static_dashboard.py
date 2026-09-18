"""
generate_static_dashboard.py
------------------------------
Genere un dashboard HTML autonome (un seul fichier, sans serveur) a partir
des resultats du modele et de la simulation business. Utilise Plotly.
Fichier de sortie : outputs/dashboard_nextel.html
"""

import os
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

sys.path.append(os.path.dirname(__file__))
from business_simulation import get_business_dataframe, run_simulation, SIMULATION_PARAMS

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def build_dashboard():
    comparaison = pd.read_csv(os.path.join(OUTPUTS_DIR, "comparaison_modeles.csv"))
    business_df = get_business_dataframe()
    resultats = run_simulation(business_df, SIMULATION_PARAMS)

    with open(os.path.join(BASE_DIR, "models", "best_model_name.json")) as f:
        best_name = json.load(f)["best_model"]

    n_total = len(business_df)
    taux_churn = business_df["Churn_reel"].mean()
    charges_moy = business_df["MonthlyCharges"].mean()
    revenu_total = business_df["MonthlyCharges"].sum()

    # --- Figure 1: comparaison modeles ---
    fig_models = go.Figure()
    for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]:
        fig_models.add_trace(go.Bar(name=metric, x=comparaison["Modele"], y=comparaison[metric]))
    fig_models.update_layout(barmode="group", yaxis_range=[0, 1], title="Comparaison des 4 modeles ML",
                              template="plotly_white", height=420)

    # --- Figure 2: bilan financier ---
    labels = ["Budget campagne", "Revenu recupere", "Profit net"]
    avec = [resultats["avec_modele"][k] for k in ["budget_campagne_usd", "revenu_recupere_usd", "profit_net_usd"]]
    sans = [resultats["sans_modele_ciblage_aleatoire"][k] for k in ["budget_campagne_usd", "revenu_recupere_usd", "profit_net_usd"]]
    fig_bilan = go.Figure()
    fig_bilan.add_trace(go.Bar(name="Avec modele ML", x=labels, y=avec, marker_color="#2E86AB"))
    fig_bilan.add_trace(go.Bar(name="Sans modele (aleatoire)", x=labels, y=sans, marker_color="#C0C0C0"))
    fig_bilan.update_layout(barmode="group", title="Impact financier : ML vs ciblage aleatoire",
                             template="plotly_white", height=420)

    # --- Figure 3: distribution du risque ---
    fig_hist = px.histogram(
        business_df, x="Churn_proba",
        color=business_df["Churn_reel"].map({0: "Reste", 1: "Churn reel"}),
        nbins=30, barmode="overlay", opacity=0.7,
        title="Distribution des probabilites de churn predites",
        template="plotly_white",
    )
    fig_hist.add_vline(x=SIMULATION_PARAMS["seuil_probabilite"], line_dash="dash", line_color="red")
    fig_hist.update_layout(height=420)

    # --- Figure 4: repartition par type de contrat ---
    contract_churn = business_df.groupby("Contract")["Churn_reel"].mean().reset_index()
    fig_contract = px.bar(contract_churn, x="Contract", y="Churn_reel",
                           title="Taux de churn par type de contrat",
                           labels={"Churn_reel": "Taux de churn"}, template="plotly_white")
    fig_contract.update_layout(height=380, yaxis_tickformat=".0%")

    html_parts = []
    html_parts.append(f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif; max-width:1200px; margin:0 auto; padding:24px; color:#1a1a2e;">
    <h1 style="color:#2E86AB;">📊 Dashboard Churn Client — NexTel</h1>
    <p style="color:#555;">Prediction du churn, comparaison de modeles ML et bilan financier d'une campagne de retention.
    Modele retenu : <b>{best_name}</b></p>

    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:24px 0;">
      {_kpi_card("Clients (test)", f"{n_total:,}")}
      {_kpi_card("Taux de churn observe", f"{taux_churn*100:.1f}%")}
      {_kpi_card("Charge mensuelle moyenne", f"${charges_moy:.2f}")}
      {_kpi_card("Revenu mensuel portefeuille", f"${revenu_total:,.0f}")}
    </div>

    <h2>1. Comparaison des modeles</h2>
    {pio.to_html(fig_models, include_plotlyjs="cdn", full_html=False)}

    <h2>2. Bilan de la campagne de retention (scenario par defaut)</h2>
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin:16px 0;">
      {_kpi_card("Clients cibles", f"{resultats['n_clients_cibles']}")}
      {_kpi_card("Precision du ciblage", f"{resultats['precision_ciblage_modele']*100:.1f}%")}
      {_kpi_card("Budget campagne", f"${resultats['avec_modele']['budget_campagne_usd']:,.0f}")}
      {_kpi_card("Profit net", f"${resultats['avec_modele']['profit_net_usd']:,.0f}", highlight=True)}
      {_kpi_card("ROI", f"{resultats['avec_modele']['roi_pct']:.0f}%", highlight=True)}
    </div>
    {pio.to_html(fig_bilan, include_plotlyjs=False, full_html=False)}
    <p style="background:#eaf6ec; padding:12px 16px; border-radius:8px; color:#1a5c2a;">
      💡 Grace au ciblage par le modele, NexTel degage <b>{resultats['gain_efficacite_modele_vs_aleatoire_usd']:,.0f} USD</b>
      de profit net supplementaire par rapport a une campagne non ciblee, pour un budget identique.
    </p>

    <h2>3. Distribution du risque de churn</h2>
    {pio.to_html(fig_hist, include_plotlyjs=False, full_html=False)}

    <h2>4. Churn par type de contrat</h2>
    {pio.to_html(fig_contract, include_plotlyjs=False, full_html=False)}

    <p style="color:#888; font-size:0.85em; margin-top:32px;">
      Donnees : Telco Customer Churn (IBM / Kaggle) — Dashboard genere en Python (Plotly).
      Hypotheses de simulation : {json.dumps(SIMULATION_PARAMS, ensure_ascii=False)}
    </p>
    </div>
    """)

    full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="utf-8"><title>Dashboard Churn - NexTel</title></head>
<body style="margin:0; background:#f7f8fa;">
{''.join(html_parts)}
</body>
</html>"""

    out_path = os.path.join(OUTPUTS_DIR, "dashboard_nextel.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Dashboard statique genere : {out_path}")
    return out_path


def _kpi_card(label, value, highlight=False):
    bg = "#2E86AB" if highlight else "#ffffff"
    color = "#ffffff" if highlight else "#1a1a2e"
    border = "none" if highlight else "1px solid #e0e0e0"
    return f"""
    <div style="flex:1; min-width:180px; background:{bg}; border:{border}; border-radius:10px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:0.8em; color:{'#dceefb' if highlight else '#888'}; text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
      <div style="font-size:1.6em; font-weight:700; color:{color}; margin-top:4px;">{value}</div>
    </div>"""


if __name__ == "__main__":
    build_dashboard()

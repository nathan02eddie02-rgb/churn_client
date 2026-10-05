"""Section 1 - Problématique."""

import streamlit as st

import data
import theme as t


def render():
    df = data.portfolio()
    biz = data.business()
    comp, best = data.model_results()
    res = data.run_simulation(biz)

    rate = df["Churn_bin"].mean()
    gone = df[df["Churn_bin"] == 1]
    lost_monthly = gone["MonthlyCharges"].sum()
    share_rev = lost_monthly / df["MonthlyCharges"].sum()

    left, right = st.columns([1.35, 1], gap="large", vertical_alignment="center")
    with left:
        st.title("Plus d'un client sur quatre résilie son abonnement.")
        st.markdown(
            '<p class="lede">Chaque départ fait disparaître un revenu mensuel récurrent, et il est '
            "impossible de rappeler toute la base : une offre de rétention a un coût. "
            "Ce projet apprend à repérer les clients sur le point de partir, pour concentrer "
            "l'effort là où il rapporte.</p>", unsafe_allow_html=True)
    with right:
        t.pictogram(round(rate * 100))
        st.markdown('<p class="src">Sur 100 clients du jeu de données Telco, taux de churn observé : '
                    f'{t.fr_pct(rate)}.</p>', unsafe_allow_html=True)

    st.markdown("## La question")
    st.markdown(
        '<div class="question">Quels clients risquent de partir, et jusqu\'où peut-on investir pour les '
        "retenir tout en restant rentable ?</div>", unsafe_allow_html=True)

    st.markdown("## Ce que le churn coûte")
    t.ledger([
        (t.fr_int(len(gone)), "clients partis", f"sur {t.fr_int(len(df))} clients", "depart"),
        (t.fr_usd(lost_monthly), "de revenu mensuel perdu", "somme des factures mensuelles des clients partis", "depart"),
        (t.fr_pct(share_rev), "du revenu mensuel total", "part du chiffre d'affaires récurrent concernée"),
    ])

    st.markdown("## La démarche")
    t.steps([
        ("Comprendre les données", "Explorer les 7 043 clients et ce qui distingue ceux qui partent."),
        ("Préparer et rééquilibrer", "Nettoyer, encoder, normaliser, puis corriger le déséquilibre entre clients restés et partis."),
        ("Comparer quatre modèles", "Entraîner quatre algorithmes et retenir le plus performant sur le jeu de test."),
        ("Chiffrer l'impact", "Simuler une campagne de rétention chez NexTel : budget, revenu récupéré, profit."),
    ])

    st.markdown("## Le résultat en bref")
    avec, sans = res["avec_modele"], res["sans_modele_ciblage_aleatoire"]
    t.ledger([
        (f"{comp.iloc[0]['ROC-AUC']:.3f}".replace(".", ","), f"ROC-AUC du modèle retenu ({best})",
         "capacité à classer les clients à risque avant les autres", None),
        (t.fr_usd(avec["profit_net_usd"]), "de profit net avec le modèle",
         f"budget de {t.fr_usd(avec['budget_campagne_usd'])}, ROI de {t.fr_pct(avec['roi_pct'] / 100, 0)}", "gain"),
        (t.fr_usd(sans["profit_net_usd"]), "de profit net sans modèle",
         "même budget, clients contactés au hasard", None),
    ])
    t.note("Le gain vient de la précision du ciblage : "
           f"{t.fr_pct(res['precision_ciblage_modele'])} des clients contactés partaient réellement, "
           f"contre {t.fr_pct(res['taux_churn_global'])} en contactant au hasard. "
           "Les paramètres de la campagne (coût, taux de succès, horizon) sont des hypothèses de simulation, "
           "modifiables dans la section Visualisations KPI.")

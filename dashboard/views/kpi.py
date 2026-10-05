"""Section 3 - Visualisations KPI."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme as t

OUTPUTS = data.OUTPUTS
CONTRACT_FR = {"Month-to-month": "Mensuel", "One year": "1 an", "Two year": "2 ans"}
INTERNET_FR = {"DSL": "DSL", "Fiber optic": "Fibre optique", "No": "Sans internet"}
PAYMENT_FR = {"Electronic check": "Chèque électronique", "Mailed check": "Chèque postal",
              "Bank transfer (automatic)": "Virement automatique", "Credit card (automatic)": "Carte automatique"}


def churn_bar(df, col, labels, title, horizontal=False):
    g = df.groupby(col)["Churn_bin"].mean().rename(index=labels).sort_values(ascending=horizontal)
    colors = [t.CHURN if v == g.max() else "#E9A58F" for v in g.values]
    fig = go.Figure(go.Bar(
        x=g.values if horizontal else g.index, y=g.index if horizontal else g.values,
        orientation="h" if horizontal else "v", marker_color=colors,
        text=[t.fr_pct(v, 1) for v in g.values], textposition="outside", cliponaxis=False))
    t.style_fig(fig, height=300, title_text=title)
    axis = fig.update_xaxes if horizontal else fig.update_yaxes
    axis(range=[0, g.max() * 1.25], tickformat=".0%", showgrid=True, gridcolor="#E3E8ED")
    return fig, g


def render():
    df = data.portfolio()
    st.title("Visualisations KPI")
    st.markdown('<p class="lede">Où se concentrent les départs, quel modèle les anticipe le mieux, '
                "et que rapporte une campagne de rétention ciblée.</p>", unsafe_allow_html=True)

    tab_port, tab_models, tab_biz = st.tabs(["Portefeuille clients", "Performance des modèles", "Bilan financier"])

    # ------------------------------------------------------------ portefeuille
    with tab_port:
        t.ledger([
            (t.fr_int(len(df)), "clients analysés", None),
            (t.fr_pct(df["Churn_bin"].mean()), "taux de churn", "part des clients ayant résilié", "depart"),
            (f"{df['tenure'].median():.0f} mois", "ancienneté médiane", None),
            (f"{df['MonthlyCharges'].mean():.2f}".replace(".", ",") + "\u202f$", "facture mensuelle moyenne", None),
        ])
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig, g = churn_bar(df, "Contract", CONTRACT_FR, "Taux de churn selon le contrat")
            st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")
            st.caption(f"Les contrats mensuels perdent {t.fr_pct(g['Mensuel'])} de leurs clients, "
                       f"contre {t.fr_pct(g['2 ans'])} pour les contrats de 2 ans.")
        with c2:
            fig, g = churn_bar(df, "InternetService", INTERNET_FR, "Taux de churn selon l'accès internet")
            st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")
            st.caption(f"La fibre optique atteint {t.fr_pct(g['Fibre optique'])} de départs, "
                       f"les clients sans internet {t.fr_pct(g['Sans internet'])}.")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            bins = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0 à 12 mois", "13 à 24 mois", "25 à 48 mois", "49 à 72 mois"])
            tmp = df.assign(Tranche=bins)
            g = tmp.groupby("Tranche", observed=True)["Churn_bin"].mean()
            colors = [t.CHURN if v == g.max() else "#E9A58F" for v in g.values]
            fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker_color=colors,
                                   text=[t.fr_pct(v) for v in g.values], textposition="outside", cliponaxis=False))
            t.style_fig(fig, height=300, title_text="Taux de churn selon l'ancienneté")
            fig.update_yaxes(range=[0, g.max() * 1.25], tickformat=".0%")
            st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")
            st.caption(f"Le risque est maximal pendant la première année ({t.fr_pct(g.iloc[0])}) "
                       "puis décroît avec l'ancienneté.")
        with c4:
            fig, g = churn_bar(df, "PaymentMethod", PAYMENT_FR, "Taux de churn selon le moyen de paiement", horizontal=True)
            st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")
            st.caption(f"Le chèque électronique ressort à {t.fr_pct(g['Chèque électronique'])} de départs, "
                       "loin devant les paiements automatiques.")

    # ------------------------------------------------------------ modèles
    with tab_models:
        comp, best = data.model_results()
        metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        fr = {"Accuracy": "Exactitude", "Precision": "Précision", "Recall": "Rappel", "F1-score": "F1", "ROC-AUC": "ROC-AUC"}
        top = comp.iloc[0]
        t.ledger([
            (best, "modèle retenu", "meilleur ROC-AUC sur le jeu de test"),
            (f"{top['ROC-AUC']:.3f}".replace(".", ","), "ROC-AUC", "pouvoir de classement des clients", "gain"),
            (f"{top['Recall']:.0%}".replace("%", "\u202f%"), "rappel", "clients partis détectés"),
            (f"{top['Precision']:.0%}".replace("%", "\u202f%"), "précision", "clients signalés qui partent vraiment"),
        ])
        palette = [t.STAY, "#7FA7C9", t.SIGNAL, "#9AA8B6", t.CHURN]
        fig = go.Figure()
        for m, color in zip(metrics, palette):
            fig.add_bar(name=fr[m], x=comp["Modele"], y=comp[m], marker_color=color)
        t.style_fig(fig, height=380, legend=True, barmode="group", title_text="Les quatre modèles sur cinq métriques")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")

        shown = comp.rename(columns={"Modele": "Modèle", **fr}).copy()
        st.dataframe(shown.style.format({v: "{:.3f}" for v in fr.values()}).highlight_max(subset=list(fr.values()), color="#CFEDE2"),
                     hide_index=True, width="stretch")
        st.caption("ROC-AUC retenu comme critère : il ne dépend pas du seuil de décision et reste fiable "
                   "quand les classes sont déséquilibrées. Les cases vertes marquent le meilleur score de chaque métrique.")

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("### Courbes ROC")
            st.image(f"{OUTPUTS}/courbes_roc.png", width="stretch")
        with col_b:
            st.markdown(f"### Matrice de confusion, {best}")
            st.image(f"{OUTPUTS}/matrice_confusion.png", width="stretch")

        imp = data.importances().head(10).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[data.feature_label(i) for i in imp.index], orientation="h",
                               marker_color=t.SIGNAL))
        t.style_fig(fig, height=380, title_text="Dix variables qui pèsent le plus dans le modèle")
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor="#E3E8ED")
        st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")

    # ------------------------------------------------------------ bilan financier
    with tab_biz:
        biz = data.business()
        st.markdown("### Campagne de rétention chez NexTel")
        st.markdown('<p class="lede">NexTel contacte les clients dont le score de départ dépasse un seuil. '
                    "Chaque contact coûte de l'argent, chaque client sauvé rapporte des mois de facture. "
                    "Ajustez les hypothèses pour voir le bilan évoluer.</p>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4, gap="large")
        seuil = p1.slider("Seuil de ciblage", 0.10, 0.90, 0.50, 0.05, help="Score minimal pour contacter un client")
        cout = p2.number_input("Coût par contact ($)", 1, 200, 15)
        succes = p3.slider("Taux de succès", 5, 80, 35, 5, format="%d %%", help="Part des clients à risque contactés qui restent") / 100
        horizon = p4.slider("Horizon (mois)", 1, 24, 12, 1, help="Mois de facture récupérés pour un client sauvé")

        params = dict(seuil_probabilite=seuil, cout_par_contact=cout, taux_succes_campagne=succes, horizon_mois=horizon)
        r = data.run_simulation(biz, params)
        a, s = r["avec_modele"], r["sans_modele_ciblage_aleatoire"]
        t.ledger([
            (t.fr_int(r["n_clients_cibles"]), "clients contactés", f"précision du ciblage : {t.fr_pct(r['precision_ciblage_modele'])}"),
            (t.fr_usd(a["budget_campagne_usd"]), "dépensés pour la rétention", None),
            (t.fr_usd(a["revenu_recupere_usd"]), "de revenu récupéré", f"{a['clients_sauves_estimes']:.0f} clients sauvés (estimation)"),
            (t.fr_usd(a["profit_net_usd"]), "de profit net", f"ROI de {t.fr_pct(a['roi_pct'] / 100, 0)}", "gain" if a["profit_net_usd"] >= 0 else "depart"),
        ])

        c1, c2 = st.columns(2, gap="large")
        with c1:
            labels = ["Budget", "Revenu récupéré", "Profit net"]
            fig = go.Figure()
            fig.add_bar(name="Avec le modèle", x=labels, y=[a["budget_campagne_usd"], a["revenu_recupere_usd"], a["profit_net_usd"]], marker_color=t.SIGNAL)
            fig.add_bar(name="Ciblage aléatoire", x=labels, y=[s["budget_campagne_usd"], s["revenu_recupere_usd"], s["profit_net_usd"]], marker_color="#B9C4CF")
            t.style_fig(fig, height=360, legend=True, barmode="group", title_text="À budget égal, avec et sans modèle")
            fig.update_yaxes(tickprefix="$")
            st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")
        with c2:
            grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
            sims = [data.run_simulation(biz, {**params, "seuil_probabilite": float(g)}) for g in grid]
            profit = [x["avec_modele"]["profit_net_usd"] for x in sims]
            best_i = int(np.argmax(profit))
            fig = go.Figure()
            fig.add_scatter(x=grid, y=profit, mode="lines", line=dict(color=t.SIGNAL, width=3), name="Profit net")
            fig.add_scatter(x=[grid[best_i]], y=[profit[best_i]], mode="markers+text", marker=dict(size=11, color=t.GAIN),
                            text=[f"  optimum : seuil {grid[best_i]:.2f}".replace(".", ",")], textposition="middle right", name="Optimum")
            fig.add_vline(x=seuil, line_dash="dash", line_color=t.CHURN)
            t.style_fig(fig, height=360, title_text="Profit net selon le seuil de ciblage")
            fig.update_yaxes(tickprefix="$")
            fig.update_xaxes(title_text="Seuil de ciblage")
            st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")

        t.note(f"À budget identique ({t.fr_usd(a['budget_campagne_usd'])}), cibler avec le modèle rapporte "
               f"<b>{t.fr_usd(r['gain_efficacite_modele_vs_aleatoire_usd'])}</b> de profit net de plus qu'un ciblage "
               "aléatoire. La ligne pointillée rouge marque le seuil choisi ; le point vert, le seuil le plus rentable "
               "avec ces hypothèses.")
        t.note("Scénario simulé sur les 1 409 clients du jeu de test. Le coût de contact, le taux de succès et l'horizon "
               "sont des hypothèses : à calibrer avec les résultats de vraies campagnes avant toute décision.", warn=True)

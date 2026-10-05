"""Section 2 - Dataset."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme as t

DICTIONARY = [
    ("Identifiant", "customerID", "texte", "Identifiant unique du client, retiré avant la modélisation"),
    ("Profil", "gender", "catégorie", "Genre du client"),
    ("Profil", "SeniorCitizen", "0 / 1", "Client senior (65 ans et plus)"),
    ("Profil", "Partner", "oui / non", "Vit en couple"),
    ("Profil", "Dependents", "oui / non", "A des personnes à charge"),
    ("Abonnement", "tenure", "nombre", "Ancienneté, en mois"),
    ("Abonnement", "Contract", "catégorie", "Mensuel, 1 an ou 2 ans"),
    ("Abonnement", "PaperlessBilling", "oui / non", "Facture dématérialisée"),
    ("Abonnement", "PaymentMethod", "catégorie", "Chèque électronique, chèque postal, virement ou carte automatiques"),
    ("Abonnement", "MonthlyCharges", "nombre", "Facture mensuelle, en dollars"),
    ("Abonnement", "TotalCharges", "nombre", "Total facturé depuis l'arrivée du client"),
    ("Services", "PhoneService", "oui / non", "Dispose d'une ligne téléphonique"),
    ("Services", "MultipleLines", "catégorie", "Plusieurs lignes téléphoniques"),
    ("Services", "InternetService", "catégorie", "DSL, fibre optique ou aucun"),
    ("Services", "OnlineSecurity", "catégorie", "Option sécurité en ligne"),
    ("Services", "OnlineBackup", "catégorie", "Option sauvegarde en ligne"),
    ("Services", "DeviceProtection", "catégorie", "Option protection de l'appareil"),
    ("Services", "TechSupport", "catégorie", "Option support technique"),
    ("Services", "StreamingTV", "catégorie", "Option télévision en streaming"),
    ("Services", "StreamingMovies", "catégorie", "Option films en streaming"),
    ("Cible", "Churn", "oui / non", "Le client a résilié son abonnement"),
]


def render():
    df = data.portfolio()
    sm = data.smote_counts()
    feature_names = data.artifacts()[3]
    missing = int(df["TotalCharges"].isna().sum())
    rate = df["Churn_bin"].mean()

    st.title("Dataset")
    st.markdown(
        '<p class="lede">Telco Customer Churn : les abonnés d\'un opérateur télécom, décrits par leur profil, '
        "leur abonnement et leurs services, avec une étiquette indiquant s'ils ont résilié. "
        "Jeu de données public publié par IBM et disponible sur Kaggle.</p>", unsafe_allow_html=True)

    t.ledger([
        (t.fr_int(len(df)), "clients", "une ligne par client"),
        (str(df.shape[1]), "colonnes", "19 explicatives, 1 identifiant, 1 cible"),
        (str(missing), "valeurs manquantes", "toutes dans TotalCharges"),
        (t.fr_pct(rate), "de clients partis", "classe minoritaire à rééquilibrer", "depart"),
    ])

    st.markdown("## Aperçu des données")
    col_table, col_donut = st.columns([2.2, 1], gap="large")
    with col_table:
        choice = st.segmented_control("Clients affichés", ["Tous", "Restés", "Partis"], default="Tous",
                                      key="dataset_filter", label_visibility="collapsed")
        view = df.drop(columns=["Churn_bin"])
        if choice == "Restés":
            view = view[df["Churn"] == "No"]
        elif choice == "Partis":
            view = view[df["Churn"] == "Yes"]
        st.dataframe(view.head(200), hide_index=True, height=360, width="stretch")
        st.markdown(f'<p class="src">200 premières lignes sur {t.fr_int(len(view))} clients affichés.</p>',
                    unsafe_allow_html=True)
    with col_donut:
        fig = go.Figure(go.Pie(
            labels=["Restés", "Partis"], values=[(df["Churn_bin"] == 0).sum(), (df["Churn_bin"] == 1).sum()],
            hole=0.66, marker=dict(colors=[t.STAY, t.CHURN], line=dict(color="#fff", width=2)),
            textinfo="percent", sort=False, direction="clockwise"))
        fig.add_annotation(text=f"<b>{t.fr_pct(rate)}</b><br>partis", showarrow=False, font=dict(size=15))
        t.style_fig(fig, height=360, legend=True, title_text="Répartition de la cible")
        st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")

    st.markdown("## Dictionnaire des variables")
    dico = pd.DataFrame(DICTIONARY, columns=["Groupe", "Variable", "Type", "Description"])
    st.dataframe(dico, hide_index=True, height=420, width="stretch")

    st.markdown("## Préparation des données")
    text_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c not in ("customerID", "Churn")]
    n_binary = sum(df[c].nunique() == 2 for c in text_cols)
    n_multi = len(text_cols) - n_binary
    t.steps([
        ("Nettoyage", f"{missing} valeurs vides de TotalCharges (clients à 0 mois d'ancienneté) remplacées par 0. "
                      "Identifiant client retiré."),
        ("Encodage", f"{n_binary} variables binaires converties en 0 / 1, {n_multi} variables à plusieurs modalités "
                     f"en colonnes indicatrices : {len(feature_names)} variables au final."),
        ("Normalisation", "StandardScaler sur l'ancienneté, la facture mensuelle et le total facturé "
                          "(moyenne 0, écart-type 1), ajusté sur le jeu d'entraînement uniquement."),
        ("Rééquilibrage", "SMOTE crée des clients partis synthétiques dans le jeu d'entraînement. "
                          "Le jeu de test garde sa distribution réelle."),
    ])

    col_a, col_b = st.columns([1.3, 1], gap="large", vertical_alignment="center")
    with col_a:
        fig = go.Figure()
        fig.add_bar(name="Restés", x=["Avant SMOTE", "Après SMOTE"], y=[sm["stay"], sm["stay"]], marker_color=t.STAY,
                    text=[t.fr_int(sm["stay"])] * 2, textposition="outside")
        fig.add_bar(name="Partis", x=["Avant SMOTE", "Après SMOTE"], y=[sm["churn"], sm["stay"]], marker_color=t.CHURN,
                    text=[t.fr_int(sm["churn"]), t.fr_int(sm["stay"])], textposition="outside")
        t.style_fig(fig, height=330, legend=True, barmode="group", title_text="Jeu d'entraînement : clients par classe")
        fig.update_yaxes(range=[0, sm["stay"] * 1.18])
        st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")
    with col_b:
        t.note(f"Découpage 80 % / 20 % stratifié : {t.fr_int(sm['total'])} clients pour l'entraînement, "
               f"{t.fr_int(len(df) - sm['total'])} pour le test. La proportion de clients partis est la même "
               "dans les deux jeux.")
        t.note("Le rééquilibrage n'est appliqué qu'à l'entraînement. Évaluer sur des clients synthétiques "
               "gonflerait artificiellement les scores.", warn=True)

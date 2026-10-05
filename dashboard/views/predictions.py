"""Section 4 - Prédictions."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme as t

YESNO = {"No": "Non", "Yes": "Oui"}
CONTRACT_FR = {"Month-to-month": "Mensuel", "One year": "1 an", "Two year": "2 ans"}
INTERNET_FR = {"DSL": "DSL", "Fiber optic": "Fibre optique", "No": "Aucun"}
PAYMENT_FR = {"Electronic check": "Chèque électronique", "Mailed check": "Chèque postal",
              "Bank transfer (automatic)": "Virement automatique", "Credit card (automatic)": "Carte automatique"}
ADDONS = {"OnlineSecurity": "Sécurité en ligne", "OnlineBackup": "Sauvegarde en ligne",
          "DeviceProtection": "Protection appareil", "TechSupport": "Support technique",
          "StreamingTV": "TV en streaming", "StreamingMovies": "Films en streaming"}

PRESET_AT_RISK = dict(gender="Female", SeniorCitizen=0, Partner="No", Dependents="No", tenure=4,
                      Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check",
                      MonthlyCharges=85.0, PhoneService="Yes", MultipleLines="No", InternetService="Fiber optic",
                      OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                      StreamingTV="Yes", StreamingMovies="Yes")
PRESET_LOYAL = dict(gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=60,
                    Contract="Two year", PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)",
                    MonthlyCharges=65.0, PhoneService="Yes", MultipleLines="Yes", InternetService="DSL",
                    OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
                    StreamingTV="No", StreamingMovies="No")


def apply_preset(preset):
    for k, v in preset.items():
        st.session_state[f"p_{k}"] = v


def band(p):
    if p < 0.35:
        return "Faible", "faible", "Peu de signaux de départ : pas d'action de rétention prioritaire."
    if p < 0.60:
        return "Modéré", "modere", "Plusieurs signaux de départ : à surveiller, une offre légère peut suffire."
    return "Élevé", "eleve", "Profil proche de ceux qui ont résilié : à contacter en priorité."


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number=dict(suffix=" %", valueformat=".0f", font=dict(family=t.FONT_HEAD, size=52, color=t.INK)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 35, 60, 100], tickwidth=0, tickfont=dict(size=12, color=t.MUTED)),
                   bar=dict(color=t.INK, thickness=0.28), bgcolor="#fff", borderwidth=0,
                   steps=[dict(range=[0, 35], color="#CFEDE2"), dict(range=[35, 60], color="#F8E3B8"),
                          dict(range=[60, 100], color="#F6C7B7")])))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=t.FONT_BODY))
    return fig


def actions_for(rec, df):
    """Pistes d'action déduites des taux de départ observés dans les données."""
    out = []
    r = lambda mask: df.loc[mask, "Churn_bin"].mean()
    if rec["Contract"] == "Month-to-month":
        out.append("Proposer un contrat d'un an avec remise : les contrats mensuels perdent "
                   f"{t.fr_pct(r(df['Contract'] == 'Month-to-month'))} de leurs clients, "
                   f"contre {t.fr_pct(r(df['Contract'] == 'One year'))} à un an.")
    if rec["PaymentMethod"] == "Electronic check":
        out.append("Proposer le paiement automatique : "
                   f"{t.fr_pct(r(df['PaymentMethod'] == 'Electronic check'))} de départs avec le chèque électronique, "
                   f"{t.fr_pct(r(df['PaymentMethod'] == 'Bank transfer (automatic)'))} avec le virement automatique.")
    if rec["InternetService"] == "Fiber optic" and rec["TechSupport"] == "No":
        fib = df["InternetService"] == "Fiber optic"
        out.append("Offrir le support technique quelques mois : en fibre, "
                   f"{t.fr_pct(r(fib & (df['TechSupport'] == 'No')))} de départs sans support, "
                   f"{t.fr_pct(r(fib & (df['TechSupport'] == 'Yes')))} avec.")
    if rec["tenure"] <= 12:
        out.append("Prévoir un point de suivi pendant la première année, période où le risque de départ est le plus élevé.")
    return out


def form():
    ss = st.session_state
    if "p_tenure" not in ss:
        apply_preset(PRESET_AT_RISK)

    b1, b2, _ = st.columns([1, 1, 2])
    b1.button("Profil à risque", on_click=apply_preset, args=(PRESET_AT_RISK,), width="stretch")
    b2.button("Profil fidèle", on_click=apply_preset, args=(PRESET_LOYAL,), width="stretch")

    st.markdown("##### Profil")
    c = st.columns(4)
    c[0].selectbox("Genre", ["Female", "Male"], key="p_gender", format_func={"Female": "Femme", "Male": "Homme"}.get)
    c[1].selectbox("Senior", [0, 1], key="p_SeniorCitizen", format_func={0: "Non", 1: "Oui"}.get)
    c[2].selectbox("En couple", list(YESNO), key="p_Partner", format_func=YESNO.get)
    c[3].selectbox("Personnes à charge", list(YESNO), key="p_Dependents", format_func=YESNO.get)

    st.markdown("##### Abonnement")
    c = st.columns(2)
    c[0].slider("Ancienneté (mois)", 0, 72, key="p_tenure")
    c[1].slider("Facture mensuelle ($)", 18.0, 120.0, step=0.5, key="p_MonthlyCharges")
    c = st.columns(3)
    c[0].selectbox("Contrat", list(CONTRACT_FR), key="p_Contract", format_func=CONTRACT_FR.get)
    c[1].selectbox("Moyen de paiement", list(PAYMENT_FR), key="p_PaymentMethod", format_func=PAYMENT_FR.get)
    c[2].selectbox("Facture dématérialisée", list(YESNO), key="p_PaperlessBilling", format_func=YESNO.get)

    st.markdown("##### Services")
    c = st.columns(3)
    c[0].selectbox("Téléphone", list(YESNO), key="p_PhoneService", format_func=YESNO.get)
    has_phone = ss["p_PhoneService"] == "Yes"
    if has_phone:
        c[1].selectbox("Plusieurs lignes", list(YESNO), key="p_MultipleLines", format_func=YESNO.get)
    c[2].selectbox("Internet", list(INTERNET_FR), key="p_InternetService", format_func=INTERNET_FR.get)
    has_net = ss["p_InternetService"] != "No"
    if has_net:
        c = st.columns(3)
        for i, (k, label) in enumerate(ADDONS.items()):
            c[i % 3].selectbox(label, list(YESNO), key=f"p_{k}", format_func=YESNO.get)
    else:
        st.caption("Sans accès internet, les options associées ne s'appliquent pas.")

    # Les champs conditionnels ne sont lus que s'ils sont affichés (Streamlit efface l'état des widgets masqués)
    rec = {}
    for k in PRESET_AT_RISK:
        if k == "MultipleLines":
            rec[k] = ss["p_MultipleLines"] if has_phone else "No phone service"
        elif k in ADDONS:
            rec[k] = ss[f"p_{k}"] if has_net else "No internet service"
        else:
            rec[k] = ss[f"p_{k}"]
    rec["TotalCharges"] = float(rec["tenure"]) * float(rec["MonthlyCharges"])
    return rec


def tab_single(df):
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        rec = form()
    p, X = data.score(pd.DataFrame([rec]))
    p = float(p[0])
    label, tone, text = band(p)

    with right:
        with st.container(border=True):
            st.markdown("##### Risque de départ")
            st.plotly_chart(gauge(p), config=t.PLOT_CONFIG, width="stretch")
            st.markdown(f'<div class="risk {tone}">Risque {label.lower()}</div>', unsafe_allow_html=True)
            st.markdown(f"<p>{text}</p>", unsafe_allow_html=True)

            contrib = data.contributions(X)
            if contrib is not None:
                top = contrib.head(6).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=top.values, y=[data.feature_label(i) for i in top.index], orientation="h",
                    marker_color=[t.CHURN if v > 0 else t.STAY for v in top.values]))
                t.style_fig(fig, height=270, title_text="Ce qui pèse dans le score")
                fig.update_xaxes(showticklabels=False, zeroline=True, zerolinecolor=t.MUTED)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, config=t.PLOT_CONFIG, width="stretch")
                st.caption("Orange : pousse vers le départ. Bleu : pousse vers la fidélité.")

            if p >= 0.35:
                acts = actions_for(rec, df)
                if acts:
                    st.markdown("##### Pistes d'action")
                    for a in acts:
                        st.markdown(f"- {a}")
                    st.caption("Associations observées dans les données, pas des relations de cause à effet.")

    t.note("Le score vient d'un modèle entraîné sur des données rééquilibrées (SMOTE) : il surestime la probabilité "
           "absolue de départ. Lisez-le comme un classement du risque entre clients, pas comme une probabilité exacte.",
           warn=True)


def tab_portfolio():
    biz = data.business()
    c1, c2 = st.columns([1, 1.4], gap="large")
    seuil = c1.slider("Score minimal", 0.30, 0.95, 0.60, 0.05)
    contracts = c2.multiselect("Contrat", list(CONTRACT_FR), default=list(CONTRACT_FR), format_func=CONTRACT_FR.get)
    sel = biz[(biz["Churn_proba"] >= seuil) & (biz["Contract"].isin(contracts))].sort_values("Churn_proba", ascending=False)

    if sel.empty:
        st.info("Aucun client ne correspond à ces critères. Baissez le score minimal ou élargissez les contrats.")
        return
    t.ledger([
        (t.fr_int(len(sel)), "clients à traiter", f"sur {t.fr_int(len(biz))} clients du jeu de test"),
        (t.fr_usd(sel["MonthlyCharges"].sum()), "de facturation mensuelle concernée", None, "depart"),
        (t.fr_pct(sel["Churn_reel"].mean()), "sont réellement partis", "vérifié sur le jeu de test"),
    ])
    show = pd.DataFrame({
        "Client": sel["customerID"], "Contrat": sel["Contract"].map(CONTRACT_FR), "Ancienneté (mois)": sel["tenure"],
        "Facture mensuelle": sel["MonthlyCharges"], "Score de départ": sel["Churn_proba"],
        "Parti en réalité": sel["Churn_reel"].map({1: "Oui", 0: "Non"})})
    st.dataframe(show, hide_index=True, height=420, width="stretch", column_config={
        "Score de départ": st.column_config.ProgressColumn("Score de départ", min_value=0.0, max_value=1.0, format="%.2f"),
        "Facture mensuelle": st.column_config.NumberColumn(format="$%.2f")})
    st.download_button("Télécharger la liste (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "clients_a_risque.csv", "text/csv")


def tab_file(df):
    required = [c for c in df.columns if c not in ("customerID", "Churn", "Churn_bin")]
    st.markdown("Importez un fichier CSV de clients pour obtenir leur score de départ. Il doit contenir les "
                f"{len(required)} colonnes du jeu de données d'origine, sans la colonne Churn.")
    sample = df.sample(20, random_state=1).drop(columns=["Churn", "Churn_bin"])
    st.download_button("Télécharger un fichier d'exemple", sample.to_csv(index=False).encode("utf-8"),
                       "exemple_clients.csv", "text/csv")
    up = st.file_uploader("Fichier CSV de clients", type="csv")
    if up is None:
        return
    try:
        new = pd.read_csv(up)
        missing = [c for c in required if c not in new.columns]
        if missing:
            st.error("Colonnes manquantes : " + ", ".join(missing))
            return
        p, _ = data.score(new)
    except Exception as exc:
        st.error(f"Le fichier n'a pas pu être traité. Vérifiez les noms de colonnes et les valeurs. Détail : {exc}")
        return
    out = pd.DataFrame({
        "Client": new["customerID"] if "customerID" in new.columns else new.index,
        "Contrat": new["Contract"].map(CONTRACT_FR), "Ancienneté (mois)": new["tenure"],
        "Facture mensuelle": new["MonthlyCharges"], "Score de départ": p,
        "Niveau de risque": [band(x)[0] for x in p]}).sort_values("Score de départ", ascending=False)
    t.ledger([
        (t.fr_int(len(out)), "clients scorés", None),
        (t.fr_int((out["Niveau de risque"] == "Élevé").sum()), "à risque élevé", "score de 60 % et plus", "depart"),
        (t.fr_int((out["Niveau de risque"] == "Modéré").sum()), "à risque modéré", "score de 35 à 60 %"),
    ])
    st.dataframe(out, hide_index=True, height=420, width="stretch", column_config={
        "Score de départ": st.column_config.ProgressColumn("Score de départ", min_value=0.0, max_value=1.0, format="%.2f"),
        "Facture mensuelle": st.column_config.NumberColumn(format="$%.2f")})
    st.download_button("Télécharger les résultats (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "clients_scores.csv", "text/csv")


def render():
    df = data.portfolio()
    st.title("Prédictions")
    st.markdown('<p class="lede">Estimez le risque de départ d\'un client, repérez les profils à traiter en priorité '
                "et scorez un fichier de clients.</p>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Un client", "Portefeuille à risque", "Fichier de clients"])
    with tab1:
        tab_single(df)
    with tab2:
        tab_portfolio()
    with tab3:
        tab_file(df)

"""
make_summary_image.py
----------------------
Génère le tableau de synthèse du projet (problématique, dataset, méthodologie,
comparaison des modèles, résultats) en image PNG et JPEG.
Les chiffres sont lus dans outputs/ : l'image reste cohérente avec le projet.

Usage :  python make_summary_image.py      ->  outputs/resume_projet_churn.png / .jpg
"""

import os
import json
import textwrap  # noqa: F401  (le découpage des lignes est fait par mesure réelle)

from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
from matplotlib.textpath import TextPath
from sklearn.model_selection import train_test_split

from preprocessing import load_raw_data

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")

# --- palette (identique au dashboard) ---
INK, PAPER, SIGNAL = "#10253B", "#F4F6F8", "#0F8B8D"
STAY, CHURN, GAIN = "#2F6B9A", "#E4572E", "#1F9D74"
HAIR, MUTED, TINT = "#D5DCE4", "#5F6F80", "#E9EEF3"
BEST_BG = "#DDF2E9"

FONT = "Carlito"           # police humaniste lisible, disponible sur la plupart des systèmes
U = 100 / 72               # 1 point = U unités de canevas (100 unités par pouce)
NB = "\u00a0"


def _half_up(x, d=0):
    """Arrondi classique (0,5 vers le haut), contrairement à l'arrondi au pair de Python."""
    return Decimal(repr(round(float(x), 8))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def fr_int(x):
    return f"{_half_up(x):,.0f}".replace(",", NB)


def fr_dec(x, d=3):
    return f"{_half_up(x, d):.{d}f}".replace(".", ",")


def fr_pct(x, d=1):
    return fr_dec(x * 100, d) + NB + "%"


def fp(size, weight="normal"):
    return FontProperties(family=FONT, weight=weight, size=size)


def text_width(s, size, weight="normal"):
    return TextPath((0, 0), s, size=size, prop=fp(size, weight)).get_extents().width * U


def wrap(s, size, width, weight="normal"):
    """Découpe s en lignes de largeur <= width (mesure réelle de la police)."""
    lines, cur = [], ""
    for word in s.split(" "):
        trial = (cur + " " + word).strip()
        if cur and text_width(trial, size, weight) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


class Canvas:
    """Collecte des primitives en coordonnées absolues, dessinées une fois la hauteur connue."""

    def __init__(self, width):
        self.w, self.cmds = width, []

    def rect(self, x, y, w, h, fc, ec="none", lw=0):
        self.cmds.append(("rect", x, y, w, h, fc, ec, lw))

    def pill(self, x, y, w, h, fc):
        self.cmds.append(("pill", x, y, w, h, fc))

    def poly(self, pts, fc):
        self.cmds.append(("poly", pts, fc))

    def line(self, x1, y1, x2, y2, color, lw=1.0):
        self.cmds.append(("line", x1, y1, x2, y2, color, lw))

    def text(self, x, y, s, size, color=INK, weight="normal", ha="left", va="top"):
        self.cmds.append(("text", x, y, s, size, color, weight, ha, va))

    def para(self, x, y, s, size, width, color=INK, weight="normal", spacing=1.38):
        """Paragraphe à retour automatique. Retourne la hauteur occupée."""
        lh = size * spacing * U
        lines = wrap(s, size, width, weight)
        for i, ln in enumerate(lines):
            self.text(x, y + i * lh, ln, size, color, weight)
        return len(lines) * lh

    def render(self, height, path_png, path_jpg):
        fig = plt.figure(figsize=(self.w / 100, height / 100), dpi=200, facecolor=PAPER)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.w)
        ax.set_ylim(height, 0)
        ax.axis("off")
        for c in self.cmds:
            kind = c[0]
            if kind == "rect":
                _, x, y, w, h, fc, ec, lw = c
                ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw))
            elif kind == "pill":
                _, x, y, w, h, fc = c
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                            facecolor=fc, edgecolor="none"))
            elif kind == "poly":
                ax.add_patch(Polygon(c[1], closed=True, facecolor=c[2], edgecolor="none"))
            elif kind == "line":
                _, x1, y1, x2, y2, color, lw = c
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="butt")
            else:
                _, x, y, s, size, color, weight, ha, va = c
                ax.text(x, y, s, fontproperties=fp(size, weight), color=color, ha=ha, va=va)
        fig.savefig(path_png, dpi=200, facecolor=PAPER)
        fig.savefig(path_jpg, dpi=200, facecolor=PAPER, pil_kwargs={"quality": 95})
        plt.close(fig)


def main():
    comp = pd.read_csv(os.path.join(OUT, "comparaison_modeles.csv"))
    with open(os.path.join(OUT, "bilan_business.json"), encoding="utf-8") as f:
        biz = json.load(f)
    params = biz["parametres"]
    avec, sans = biz["avec_modele"], biz["sans_modele_ciblage_aleatoire"]

    raw = load_raw_data()
    y = (raw["Churn"] == "Yes").astype(int)
    n_missing = int(pd.to_numeric(raw["TotalCharges"], errors="coerce").isna().sum())
    ytr, _ = train_test_split(y, test_size=0.2, random_state=42, stratify=y)
    n_stay, n_churn = int((ytr == 0).sum()), int((ytr == 1).sum())
    rate = y.mean()
    best = comp.iloc[0]

    W, X0, X1 = 1200, 40, 1160
    LABEL_W, PAD = 215, 26
    CX = X0 + LABEL_W + 28            # début de la colonne de contenu
    CW = X1 - CX - 24                 # largeur de la colonne de contenu
    cv = Canvas(W)

    # ---------------------------------------------------------------- en-tête
    cv.rect(0, 0, W, 178, INK)
    cv.rect(0, 178, W, 5, SIGNAL)
    cv.text(X0 + 20, 40, "Prédiction du churn client", 38, "#FFFFFF", "bold")
    cv.text(X0 + 20, 106, "De la donnée brute à la décision de rétention : un projet de data science de bout en bout",
            14.5, "#B8C9DA")
    cv.text(X0 + 20, 138, "Comparaison de 4 modèles, gestion du déséquilibre des classes, simulation financière d'une campagne",
            12, "#8FA6BC")

    y = 183 + 30
    rows = []   # (y_top, hauteur, label)

    def start_row(label):
        rows.append([y, None, label])

    def end_row(content_h):
        nonlocal y
        h = content_h + 2 * PAD
        rows[-1][1] = h
        y += h

    # ---------------------------------------------------------------- 1. problématique
    start_row("Problématique")
    q = (f"Plus d'un client sur quatre ({fr_pct(rate)}) résilie son abonnement. Comment repérer à l'avance les "
         "clients à risque pour concentrer une campagne de rétention là où elle est rentable, sans contacter "
         "toute la base ?")
    h = cv.para(CX, y + PAD, q, 15, CW, INK, "normal", 1.42)
    end_row(h)

    # ---------------------------------------------------------------- 2. dataset
    start_row("Dataset")
    items = [
        ("Source", "Telco Customer Churn, publié par IBM, disponible sur Kaggle"),
        ("Taille", f"{fr_int(len(raw))} clients, {raw.shape[1]} colonnes dont 1 identifiant et 1 cible"),
        ("Cible", f"Churn (client parti ou resté) : {fr_pct(rate)} de départs, classes déséquilibrées"),
        ("Variables", "profil du client, abonnement (contrat, ancienneté, facture), services (internet, téléphone, streaming)"),
    ]
    colw = (CW - 30) / 2
    block_h = []
    for i, (k, v) in enumerate(items):
        cx = CX + (i % 2) * (colw + 30)
        cy = y + PAD + (i // 2) * (block_h[0] if i >= 2 else 0)
        cv.text(cx, cy, k, 11.5, MUTED, "bold")
        hh = cv.para(cx, cy + 22, v, 13, colw, INK)
        block_h.append(22 + hh + 16)
    end_row(max(block_h[0], block_h[1]) + max(block_h[2], block_h[3]) - 16)

    # ---------------------------------------------------------------- 3. méthodologie
    start_row("Méthodologie")
    steps = [
        ("Nettoyage", f"{n_missing} valeurs vides de TotalCharges remplacées, identifiant retiré"),
        ("Encodage", "variables binaires et colonnes indicatrices : 30 variables"),
        ("Normalisation", "StandardScaler sur 3 variables numériques, ajusté sur l'entraînement"),
        ("Rééquilibrage", f"SMOTE sur l'entraînement : {fr_int(n_stay)} / {fr_int(n_churn)} devient {fr_int(n_stay)} / {fr_int(n_stay)}"),
        ("4 modèles", "Régression logistique, Random Forest, XGBoost, SVM, test 20 %"),
        ("Simulation", "campagne de rétention NexTel : budget, revenu, profit"),
    ]
    n, notch, gap = len(steps), 16, 6
    cw = (CW + (n - 1) * (notch - gap)) / n
    step = cw - notch + gap
    ch, top = 46, y + PAD
    cap_h = 0
    for i, (title, cap) in enumerate(steps):
        x = CX + i * step
        fill = SIGNAL if i == n - 1 else INK
        pts = [(x, top), (x + cw - notch, top), (x + cw, top + ch / 2), (x + cw - notch, top + ch), (x, top + ch)]
        if i > 0:
            pts.append((x + notch, top + ch / 2))
        cv.poly(pts, fill)
        cx = x + (notch if i > 0 else 0) / 2 + (cw - notch) / 2 + (6 if i > 0 else 0) * 0
        cv.text(cx + 2, top + ch / 2, title, 12.5, "#FFFFFF", "bold", ha="center", va="center")
        hh = cv.para(x + 2, top + ch + 14, cap, 10.5, step - 14, MUTED, "normal", 1.34)
        cap_h = max(cap_h, hh)
    end_row(ch + 14 + cap_h)

    # ---------------------------------------------------------------- 4. comparaison des modèles
    start_row("Comparaison des modèles")
    cols = ["Modèle", "Exactitude", "Précision", "Rappel", "F1", "ROC-AUC"]
    keys = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    names = {"Regression Logistique": "Régression logistique"}
    mw = 250
    nw = (CW - mw) / 5
    rh = 38
    ty = y + PAD
    cv.line(CX, ty + rh, CX + CW, ty + rh, INK, 1.6)
    cv.text(CX + 12, ty + rh / 2, cols[0], 12, MUTED, "bold", va="center")
    for j, c in enumerate(cols[1:]):
        cv.text(CX + mw + nw * j + nw - 14, ty + rh / 2, c, 12, MUTED, "bold", ha="right", va="center")
    maxima = {k: comp[k].max() for k in keys}
    for i, r in comp.iterrows():
        ry = ty + rh * (i + 1)
        is_best = i == 0
        if is_best:
            cv.rect(CX, ry, CW, rh, BEST_BG)
        cv.line(CX, ry + rh, CX + CW, ry + rh, HAIR, 0.8)
        label = names.get(r["Modele"], r["Modele"])
        cv.text(CX + 12, ry + rh / 2, label, 13.5, INK, "bold" if is_best else "normal", va="center")
        if is_best:
            tx = CX + 12 + text_width(label, 13.5, "bold") + 12
            cv.pill(tx, ry + rh / 2 - 11, 62, 22, SIGNAL)
            cv.text(tx + 31, ry + rh / 2, "retenu", 10.5, "#FFFFFF", "bold", ha="center", va="center")
        for j, k in enumerate(keys):
            top_val = abs(r[k] - maxima[k]) < 1e-12
            cv.text(CX + mw + nw * j + nw - 14, ry + rh / 2, fr_dec(r[k]), 13.5, INK,
                    "bold" if top_val else "normal", ha="right", va="center")
    ny = ty + rh * 5 + 12
    nh = cv.para(CX, ny, "Critère de sélection : ROC-AUC, indépendant du seuil de décision et adapté aux classes "
                         "déséquilibrées. Scores calculés sur 1 409 clients de test, sans rééquilibrage. "
                         "En gras : meilleur score de chaque colonne.", 10.5, CW, MUTED, "normal", 1.36)
    end_row(rh * 5 + 12 + nh)

    # ---------------------------------------------------------------- 5. résultats
    start_row("Résultats obtenus")
    stats = [
        (fr_dec(best["ROC-AUC"]), INK, f"ROC-AUC de {best['Modele']}, le modèle retenu"),
        (fr_int(avec["profit_net_usd"]) + NB + "$", GAIN,
         f"de profit net pour la campagne ciblée par le modèle, ROI de {fr_int(avec['roi_pct'])}{NB}%"),
        ("+" + fr_int(biz["gain_efficacite_modele_vs_aleatoire_usd"]) + NB + "$", GAIN,
         f"de profit net face à un ciblage aléatoire, à budget égal de {fr_int(avec['budget_campagne_usd'])}{NB}$"),
    ]
    sw = (CW - 2 * 30) / 3
    sh = 0
    for i, (big, color, cap) in enumerate(stats):
        sx = CX + i * (sw + 30)
        if i > 0:
            cv.line(sx - 15, y + PAD + 4, sx - 15, y + PAD + 100, HAIR, 1)
        cv.text(sx, y + PAD - 4, big, 31, color, "bold")
        hh = cv.para(sx, y + PAD + 52, cap, 11.5, sw - 6, MUTED, "normal", 1.36)
        sh = max(sh, 52 + hh)
    fy = y + PAD + sh + 14
    h1 = cv.para(CX, fy, "Variables les plus influentes dans le modèle : le type de contrat, le paiement par chèque "
                         "électronique et l'accès internet en fibre optique.", 12.5, CW, INK, "normal", 1.4)
    h2 = cv.para(CX, fy + h1 + 6,
                 f"Hypothèses de la simulation : seuil de ciblage {fr_int(params['seuil_probabilite'] * 100)}{NB}%, "
                 f"coût de contact {params['cout_par_contact']}{NB}$, taux de succès {fr_int(params['taux_succes_campagne'] * 100)}{NB}%, "
                 f"horizon de {params['horizon_mois']} mois. À calibrer avec de vraies campagnes.",
                 10.5, CW, MUTED, "normal", 1.36)
    end_row((fy - (y + PAD)) + h1 + 6 + h2)

    # ---------------------------------------------------------------- 6. outils
    start_row("Outils et livrables")
    h1 = cv.para(CX, y + PAD, "Python, pandas, scikit-learn, XGBoost, imbalanced-learn, Plotly, Streamlit, Jupyter.",
                 13, CW, INK)
    h2 = cv.para(CX, y + PAD + h1 + 6, "Livrables : dashboard interactif en 4 sections, notebook d'analyse, "
                                       "code source sur GitHub.", 13, CW, INK)
    end_row(h1 + 6 + h2)

    # ---------------------------------------------------------------- fond du tableau
    table_top, table_bottom = rows[0][0], y
    base = len(cv.cmds)
    bg = []
    bg.append(("rect", X0, table_top, X1 - X0, table_bottom - table_top, "#FFFFFF", HAIR, 1))
    for top_, h_, label in rows:
        bg.append(("rect", X0, top_, LABEL_W, h_, TINT, "none", 0))
        bg.append(("rect", X0, top_, 5, h_, SIGNAL, "none", 0))
    for top_, h_, label in rows[1:]:
        bg.append(("line", X0, top_, X1, top_, HAIR, 1.0))
    # les étiquettes de ligne
    labels = []
    for top_, h_, label in rows:
        lines = wrap(label, 15, LABEL_W - 44, "bold")
        lh = 15 * 1.3 * U
        start = top_ + PAD + 2
        for i, ln in enumerate(lines):
            labels.append(("text", X0 + 24, start + i * lh, ln, 15, INK, "bold", "left", "top"))
    # le fond doit être dessiné avant le contenu : on l'insère après l'en-tête
    header_cmds = cv.cmds[:5]
    content_cmds = cv.cmds[5:base]
    cv.cmds = header_cmds + bg + content_cmds + labels

    # ---------------------------------------------------------------- pied de page
    fy = table_bottom + 22
    cv.text(X0, fy, "Données : Telco Customer Churn (IBM, Kaggle). Scénario de rétention simulé : les résultats "
                    "financiers dépendent des hypothèses retenues.", 10.5, MUTED)
    cv.text(X1, fy, "github.com/nathan02eddie02-rgb/churn_client", 10.5, INK, "bold", ha="right")
    total_h = fy + 40

    png = os.path.join(OUT, "resume_projet_churn.png")
    jpg = os.path.join(OUT, "resume_projet_churn.jpg")
    cv.render(total_h, png, jpg)
    print("Image générée :", png, "et", jpg)


if __name__ == "__main__":
    main()

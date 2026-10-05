"""
theme.py - identité visuelle du dashboard (palette, typographie, composants HTML).

Palette
  encre   #10253B   texte et structure
  papier  #F4F6F8   fond de page
  signal  #0F8B8D   couleur d'action (navigation, boutons)
  reste   #2F6B9A   clients qui restent
  depart  #E4572E   clients qui partent / risque
  gain    #1F9D74   gain, profit
  trait   #D5DCE4   filets
  discret #6B7A8A   texte secondaire
Typographie : Bricolage Grotesque (titres, chiffres) + Instrument Sans (texte).
"""

import re
from decimal import Decimal, ROUND_HALF_UP

import streamlit as st

INK, PAPER, SIGNAL = "#10253B", "#F4F6F8", "#0F8B8D"
STAY, CHURN, GAIN = "#2F6B9A", "#E4572E", "#1F9D74"
HAIRLINE, MUTED, AMBER = "#D5DCE4", "#6B7A8A", "#E3A02B"

FONT_BODY = "'Instrument Sans', 'Segoe UI', system-ui, sans-serif"
FONT_HEAD = "'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"] {{ font-family: {FONT_BODY}; color: {INK}; }}
.block-container {{ max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }}
footer {{ visibility: hidden; }}

h1, h2, h3, h4 {{ font-family: {FONT_HEAD}; color: {INK}; letter-spacing: -0.015em; font-weight: 700; }}
h1 {{ font-size: 2.5rem; line-height: 1.12; }}
h2 {{ font-size: 1.6rem; margin-top: 2.2rem; }}
h3 {{ font-size: 1.15rem; }}
p, li {{ line-height: 1.6; }}

/* barre latérale */
[data-testid="stSidebar"] .brand {{ font-family: {FONT_HEAD}; font-size: 1.5rem; font-weight: 700; margin: .2rem 0 .1rem; }}
[data-testid="stSidebar"] .brand-sub {{ color: #9FB3C6; font-size: .9rem; margin-bottom: 1.4rem; }}
[data-testid="stSidebar"] .fiche {{ border-top: 1px solid #2A4562; padding: .7rem 0; font-size: .92rem; }}
[data-testid="stSidebar"] .fiche b {{ display: block; font-family: {FONT_HEAD}; font-size: 1.25rem; }}
[data-testid="stSidebar"] .fiche span {{ color: #9FB3C6; }}
[data-testid="stSidebar"] a {{ color: #7FD6D8; }}

/* bandeau de chiffres */
.ledger {{ display: grid; border-top: 2px solid {INK}; border-bottom: 1px solid {HAIRLINE}; margin: 1.2rem 0 1.6rem; background: #fff; }}
.ledger .cell {{ padding: 1rem 1.2rem; border-left: 1px solid {HAIRLINE}; }}
.ledger .cell:first-child {{ border-left: none; }}
.ledger .v {{ font-family: {FONT_HEAD}; font-size: 2rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }}
.ledger .l {{ color: {MUTED}; font-size: .9rem; margin-top: .25rem; }}
.ledger .n {{ color: {MUTED}; font-size: .8rem; margin-top: .15rem; }}
.ledger .v.depart {{ color: {CHURN}; }} .ledger .v.gain {{ color: {GAIN}; }}
@media (max-width: 800px) {{ .ledger {{ grid-template-columns: repeat(2, 1fr) !important; }} .ledger .cell:nth-child(odd) {{ border-left: none; }} }}

/* pictogramme : 100 clients */
.picto {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 9px; max-width: 340px; }}
.picto i {{ display: block; aspect-ratio: 1; border-radius: 50%; background: #C3D1DE; }}
.picto i.churn {{ background: {CHURN}; }}
.picto-legend {{ margin-top: .9rem; font-size: .95rem; color: {MUTED}; }}
.picto-legend b {{ color: {INK}; }}

/* étapes de la démarche */
.steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.4rem; margin: 1rem 0 1.6rem; }}
.steps .s {{ border-top: 2px solid {INK}; padding-top: .7rem; }}
.steps .s .k {{ font-family: {FONT_HEAD}; font-weight: 700; font-size: 1.05rem; }}
.steps .s .t {{ color: {MUTED}; font-size: .93rem; margin-top: .25rem; }}
@media (max-width: 800px) {{ .steps {{ grid-template-columns: 1fr 1fr; }} }}

/* question centrale et notes */
.question {{ font-family: {FONT_HEAD}; font-size: 1.55rem; line-height: 1.3; font-weight: 500; border-left: 4px solid {SIGNAL}; padding: .2rem 0 .2rem 1.2rem; margin: 1.2rem 0 1.8rem; max-width: 52rem; }}
.note {{ border-left: 3px solid {SIGNAL}; background: #fff; padding: .7rem 1rem; margin: .8rem 0; font-size: .95rem; }}
.note.warn {{ border-left-color: {AMBER}; }}
.lede {{ font-size: 1.1rem; color: #33485E; max-width: 46rem; }}
.src {{ color: {MUTED}; font-size: .88rem; }}

/* niveau de risque */
.risk {{ font-family: {FONT_HEAD}; font-size: 1.5rem; font-weight: 700; }}
.risk.faible {{ color: {GAIN}; }} .risk.modere {{ color: {AMBER}; }} .risk.eleve {{ color: {CHURN}; }}

/* onglets */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {HAIRLINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {FONT_HEAD}; font-weight: 500; font-size: 1.02rem; padding: .6rem 0; }}
</style>
"""


def inject_css():
    st.markdown(re.sub(r"\n\s*\n", "\n", CSS), unsafe_allow_html=True)


def html(block):
    """Affiche un bloc HTML (sans indentation, pour éviter le rendu en bloc de code Markdown)."""
    st.markdown("".join(line.strip() for line in block.splitlines()), unsafe_allow_html=True)


def ledger(cells):
    """Bandeau de chiffres. cells = [(valeur, libellé, note|None, ton|None), ...]"""
    out = [f'<div class="ledger" style="grid-template-columns:repeat({len(cells)},1fr)">']
    for c in cells:
        value, label = c[0], c[1]
        note = c[2] if len(c) > 2 and c[2] else ""
        tone = c[3] if len(c) > 3 and c[3] else ""
        out.append(f'<div class="cell"><div class="v {tone}">{value}</div><div class="l">{label}</div>')
        if note:
            out.append(f'<div class="n">{note}</div>')
        out.append("</div>")
    out.append("</div>")
    html("".join(out))


def pictogram(n_churn, total=100):
    dots = "".join('<i class="churn"></i>' if k < n_churn else "<i></i>" for k in range(total))
    html(f'<div class="picto">{dots}</div>'
         f'<div class="picto-legend"><b>{n_churn}</b> résilient leur abonnement, <b>{total - n_churn}</b> restent.</div>')


def steps(items):
    cells = "".join(f'<div class="s"><div class="k">{i}. {k}</div><div class="t">{t}</div></div>'
                    for i, (k, t) in enumerate(items, 1))
    html(f'<div class="steps">{cells}</div>')


def note(text, warn=False):
    html(f'<div class="note{" warn" if warn else ""}">{text}</div>')


def style_fig(fig, height=340, legend=False, **layout):
    """Applique l'identité visuelle à une figure Plotly."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, size=13, color=INK),
        title=dict(font=dict(family=FONT_HEAD, size=16, color=INK), x=0, xanchor="left"),
        showlegend=legend, legend=dict(orientation="h", y=-0.18, x=0),
        **layout,
    )
    fig.update_xaxes(showgrid=False, linecolor=HAIRLINE, ticks="outside", tickcolor=HAIRLINE)
    fig.update_yaxes(gridcolor="#E3E8ED", zeroline=False)
    return fig


PLOT_CONFIG = {"displayModeBar": False}


def _half_up(x, d=0):
    """Arrondi classique (0,5 vers le haut), contrairement à l'arrondi au pair de Python."""
    q = Decimal(1).scaleb(-d)
    return Decimal(repr(round(float(x), 8))).quantize(q, rounding=ROUND_HALF_UP)


def fr_int(x):
    return f"{_half_up(x):,.0f}".replace(",", "\u202f")


def fr_pct(x, d=1):
    return f"{_half_up(float(x) * 100, d):,.{d}f}".replace(",", "\u202f").replace(".", ",") + "\u202f%"


def fr_usd(x):
    return fr_int(x) + "\u202f$"

# -*- coding: utf-8 -*-
"""Briques d'écran communes aux assemblages — la charte des modules béton.

``open_bloc_left_right`` / ``close_bloc`` et les couleurs sont ceux de
``poutre.py`` / ``dalle.py`` (copiés, comme dalle.py l'a fait de poutre.py :
même rendu, aucune dépendance entre modules). Les autres briques —
vérification avec barre de taux, bandeau d'état, alerte — sont composées
avec les mêmes fonds, bordures, rayons et icônes.
"""
import streamlit as st

from acier.formats import F, pct

C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌"}
C_TRAIT = {"ok": "#2E7D46", "warn": "#8A6B1F", "nok": "#9C3341"}


def open_bloc_left_right(left, right="", etat="ok", pct_val=None):
    """Header de bloc : texte à gauche + (optionnel) texte à droite + pct +
    icône. HTML sur une seule ligne (voir fix v2.39.1 de poutre.py)."""
    parts = []
    if right:
        parts.append(f"<div style='font-weight:600;opacity:0.9;white-space:nowrap;'>{right}</div>")
    if pct_val is not None:
        try:
            parts.append(f"<div style='font-weight:700;white-space:nowrap;'>{float(pct_val):.0f} %</div>")
        except Exception:
            pass
    parts.append(f"<div style='font-size:20px;line-height:1;'>{C_ICONES.get(etat, '')}</div>")
    right_side = "".join(parts)
    bg = C_COULEURS.get(etat, "#f6f6f6")
    html = (
        f'<div style="background-color:{bg};padding:12px 14px 10px 14px;'
        f'border-radius:10px;border:1px solid #d9d9d9;margin:10px 0 12px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'gap:10px;margin-bottom:6px;">'
        f'<div style="font-weight:700;">{left}</div>'
        f'<div style="display:flex;align-items:center;gap:10px;">{right_side}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def close_bloc():
    st.markdown("</div>", unsafe_allow_html=True)


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def badge_nature(nat):
    """Petit badge de nature : EC · EC + COMP · COMP · INT."""
    return (f"<span style='display:inline-block;border:1px solid #33415C;color:#33415C;"
            f"font-size:11px;padding:0 5px;border-radius:3px;margin-left:6px;font-weight:600;"
            f"vertical-align:middle;white-space:nowrap;'>{_esc(nat)}</span>")


def bloc_verification(c):
    """Une vérification : libellé + nature, Ed / Rd, barre de taux, η, icône —
    même fond, même bordure, même rayon que les blocs béton."""
    ok = bool(c.ok)
    etat = "ok" if ok else "nok"
    nd = 3 if c.unit == "-" else 1
    valeurs = F(c.Ed, nd) + " / " + F(c.Rd, nd) + ("" if c.unit == "-" else " " + c.unit)
    eta = c.eta if c.eta is not None else 0
    larg = 0 if eta != eta else min(eta, 1.0) * 100
    barc = C_TRAIT[etat]
    html = (
        f'<div style="background-color:{C_COULEURS[etat]};padding:8px 14px 8px 14px;border-radius:10px;'
        f'border:1px solid #d9d9d9;margin:6px 0 4px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<div style="font-weight:600;flex:1 1 280px;">{_esc(c.label)}{badge_nature(c.nat)}</div>'
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<div style="white-space:nowrap;color:#374151;font-variant-numeric:tabular-nums;">{valeurs}</div>'
        f'<div style="width:110px;height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;">'
        f'<div style="width:{larg:.1f}%;height:100%;background:{barc};"></div></div>'
        f'<div style="font-weight:700;white-space:nowrap;min-width:58px;text-align:right;">{pct(c.eta, 1)}</div>'
        f'<div style="font-size:18px;line-height:1;">{C_ICONES[etat]}</div>'
        f'</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def bloc_statut(titre, lignes, etat, droite=""):
    """Bandeau d'état permanent : titre fort, lignes secondaires, mention à
    droite (benchmark)."""
    bg = C_COULEURS[etat]
    trait = C_TRAIT[etat]
    corps = "".join(f"<div style='font-size:0.92em;color:#374151;'>{_esc(l)}</div>" for l in lignes if l)
    dr = f"<div style='font-weight:600;white-space:nowrap;color:{trait};'>{_esc(droite)}</div>" if droite else ""
    html = (
        f'<div style="background-color:{bg};padding:12px 14px 10px 14px;border-radius:10px;'
        f'border:1px solid #d9d9d9;border-left:8px solid {trait};margin:4px 0 10px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;">'
        f'<div><div style="font-weight:700;font-size:1.15em;color:{trait};">{_esc(titre)}</div>{corps}</div>{dr}'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def bloc_alerte(texte, bloquante, selectionnee=False):
    """Une alerte : bloquante (rouge) ou informative (ocre), avec le cadre de
    sélection quand elle est localisée."""
    etat = "nok" if bloquante else "warn"
    cadre = f"box-shadow:0 0 0 2px {C_TRAIT[etat]};" if selectionnee else ""
    html = (
        f'<div style="background-color:{C_COULEURS[etat]};padding:8px 12px;border-radius:8px;'
        f'border:1px solid #d9d9d9;border-left:6px solid {C_TRAIT[etat]};margin:4px 0 2px 0;{cadre}">'
        f"<b>{'Bloquant' if bloquante else 'Attention'}</b> – {_esc(texte)}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def tableau_md(entetes, lignes, droite=()):
    """Tableau Markdown ; ``droite`` = indices des colonnes alignées à droite."""
    def cell(s):
        return _esc(s).replace("|", "\\|").replace("\n", " ")
    al = ["---:" if i in droite else "---" for i in range(len(entetes))]
    out = ["| " + " | ".join(cell(e) for e in entetes) + " |", "| " + " | ".join(al) + " |"]
    for l in lignes:
        out.append("| " + " | ".join(cell(x) for x in l) + " |")
    return "\n".join(out)


# --------------------------------------------------------------- refonte
class Html(str):
    """Une cellule déjà en HTML (non échappée) pour ``tableau_html``."""


def taux_html(eta, ok, gras=False):
    """« 74,0 % » coloré : vert vérifié, rouge non vérifié."""
    col = C_TRAIT["ok" if ok else "nok"]
    return Html(f"<span style='color:{col};font-weight:{700 if gras or not ok else 600};"
                f"font-variant-numeric:tabular-nums;white-space:nowrap;'>{_esc(pct(eta, 1))}</span>")


def tableau_html(entetes, lignes, droite=(), largeurs=None, compact=True):
    """Tableau HTML compact (cellules colorées possibles via ``Html``) ;
    ``droite`` = indices des colonnes alignées à droite, ``largeurs`` =
    largeurs CSS optionnelles par colonne."""
    fs = "0.86em" if compact else "0.95em"
    pad = "3px 8px" if compact else "6px 10px"

    def cell(x, i, th=False):
        al = "right" if i in droite else "left"
        w = f"width:{largeurs[i]};" if largeurs and largeurs[i] else ""
        s = x if isinstance(x, Html) else _esc(x)
        tag = "th" if th else "td"
        st_ = (f"padding:{pad};text-align:{al};{w}border-bottom:1px solid #e5e7eb;vertical-align:middle;"
               + ("color:#6E7480;font-weight:600;font-size:0.8em;letter-spacing:.04em;text-transform:uppercase;"
                  "border-bottom:1.5px solid #33415C;white-space:nowrap;" if th else ""))
        return f"<{tag} style='{st_}'>{s}</{tag}>"
    html = [f"<table style='width:100%;border-collapse:collapse;font-size:{fs};margin:2px 0 10px 0;'>",
            "<thead><tr>" + "".join(cell(e, i, True) for i, e in enumerate(entetes)) + "</tr></thead><tbody>"]
    for l in lignes:
        html.append("<tr>" + "".join(cell(x, i) for i, x in enumerate(l)) + "</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def bloc_statut_ligne(etat, titre, complement, elements):
    """Bandeau d'état sur une ligne : icône, statut fort, taux et vérification
    dimensionnante ; en dessous, une ligne de taux par élément."""
    trait = C_TRAIT[etat]
    chips = "".join(
        f"<span style='display:inline-block;margin:0 10px 0 0;color:#374151;white-space:nowrap;'>{_esc(nom)} "
        f"<b style='color:{C_TRAIT['ok' if ok else 'nok']};font-variant-numeric:tabular-nums;'>{_esc(pct(eta, 0))}</b></span>"
        for nom, eta, ok in elements)
    html = (
        f'<div style="background-color:{C_COULEURS[etat]};padding:8px 14px 7px 14px;border-radius:10px;'
        f'border:1px solid #d9d9d9;border-left:8px solid {trait};margin:2px 0 8px 0;">'
        f'<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">'
        f'<span style="font-size:1.15em;line-height:1;">{C_ICONES[etat]}</span>'
        f'<span style="font-weight:700;font-size:1.08em;color:{trait};white-space:nowrap;">{_esc(titre)}</span>'
        f'<span style="color:#374151;">{_esc(complement)}</span></div>'
        f'<div style="font-size:0.88em;margin-top:3px;">{chips}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def bloc_alerte_ligne(titre, detail, bloquante, selectionnee=False):
    """Une alerte sur une ligne : titre court en gras, détail chiffré."""
    etat = "nok" if bloquante else "warn"
    cadre = f"box-shadow:0 0 0 2px {C_TRAIT[etat]};" if selectionnee else ""
    det = f" <span style='color:#374151;'>{_esc(detail)}</span>" if detail else ""
    html = (
        f'<div style="background-color:{C_COULEURS[etat]};padding:5px 12px;border-radius:8px;font-size:0.92em;'
        f'border:1px solid #d9d9d9;border-left:6px solid {C_TRAIT[etat]};margin:2px 0;{cadre}">'
        f"{'⛔' if bloquante else '⚠️'} <b>{_esc(titre)}</b>{det}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def titre_bloc(txt):
    """Titre de bloc de la carte (petites capitales, discret)."""
    st.markdown(f"<div style='font-size:0.72em;font-weight:700;letter-spacing:.07em;color:#33415C;"
                f"line-height:1.15;padding:0 0 9px 0;'>{_esc(txt)}</div>", unsafe_allow_html=True)

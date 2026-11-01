# -*- coding: utf-8 -*-
"""Briques d'écran communes aux assemblages : bandeau d'état, alertes,
tableaux, titres de bloc, légende des natures, replis de l'onglet
Vérifications. Une seule palette (``C_COULEURS`` / ``C_TRAIT`` /
``C_ICONES``) : les modules et ``modules/assemblages.py`` la lisent ici.
"""
import re

import streamlit as st

from acier.formats import pct

C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌"}
C_TRAIT = {"ok": "#2E7D46", "warn": "#8A6B1F", "nok": "#9C3341", "mut": "#6E7480"}
C_ENCRE = "#33415C"          # titres de bloc, badges, filet d'en-tête des tables

# Les trois natures de vérification (badges des tables, légende d'une ligne)
NATURES = {"EC": "formule de l'Eurocode", "COMP": "modèle complémentaire reconnu",
           "INT": "interprétation de l'outil"}


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def badge_nature(nat):
    """Petit badge de nature : EC · EC + COMP · COMP · INT."""
    return (f"<span style='display:inline-block;border:1px solid {C_ENCRE};color:{C_ENCRE};"
            f"font-size:11px;padding:0 5px;border-radius:3px;margin-left:6px;font-weight:600;"
            f"vertical-align:middle;white-space:nowrap;'>{_esc(nat)}</span>")


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


class Html(str):
    """Une cellule déjà en HTML (non échappée) pour ``tableau_html``."""


def taux_html(eta, ok, gras=False):
    """« 74,0 % » coloré : vert vérifié, rouge non vérifié."""
    col = C_TRAIT["ok" if ok else "nok"]
    return Html(f"<span style='color:{col};font-weight:{700 if gras or not ok else 600};"
                f"font-variant-numeric:tabular-nums;white-space:nowrap;'>{_esc(pct(eta, 1))}</span>")


def verdict_html(ok, oui="OK", non="NON OK"):
    """« OK » / « NON OK » en couleur de la palette (Tableau 3.3)."""
    return Html(f"<span style='color:{C_TRAIT['ok' if ok else 'nok']};font-weight:700'>{oui if ok else non}</span>")


# Un mot de l'en-tête (≥ 3 lettres latines, hors parenthèses) passe en capitales ; un symbole (Ed, Rd, η,
# σvm, FT,Rd, Σleff, tp) et une unité entre parenthèses gardent leur casse.
_MOT = re.compile(r"(?<![\w,])([A-Za-zÀ-ÖØ-öø-ÿ]{3,})(?![\w,])")


def capitales(txt):
    """« Vérification » → « VÉRIFICATION », « T (kNm) » inchangé, « η » inchangé.
    Les en-têtes des tables étaient mis en capitales par ``text-transform:
    uppercase`` : « η » devenait « Η » (un H) et « σvm » « ΣVM » (une somme),
    relecture du 01/11/2026 — la casse est donc écrite, jamais transformée."""
    return "".join(x if x.startswith("(") else _MOT.sub(lambda m: m.group(1).upper(), x)
                   for x in re.split(r"(\([^)]*\))", str(txt)))


# Une référence ne se coupe pas dans son sigle : « MSB / P5 §3.2.3 », « Tab. / 3.4 », « P358 / Ch. 8 » se
# lisaient sur deux lignes dans la colonne Réf. (relecture des captures du 01/11/2026). L'espace après
# « MSB », « Tab. », « Ch. », « hyp. » devient insécable ; celle qui précède un « § » reste sécable, sans quoi
# la colonne s'élargirait au téléphone jusqu'à faire défiler la table. Le séparateur « · » reste au bout de la
# ligne qu'il ferme : au téléphone, « Tab. 3.4 » lié laissait le « · » seul sur sa ligne (plat d'âme, 390 px).
_SIGLE = re.compile(r"(?:(?<=\bMSB)|(?<=\bTab\.)|(?<=\bCh\.)|(?<=\bhyp\.)) ")


def lier_sigles(txt):
    """« MSB P5 §3.2 · Tab. 3.4 » → espaces insécables après les sigles et
    devant le séparateur « · »."""
    return _SIGLE.sub("&nbsp;", txt).replace(" · ", "&nbsp;· ")


def tableau_html(entetes, lignes, droite=(), largeurs=None, compact=True, defilant=None, serre=0):
    """Tableau HTML compact (cellules colorées possibles via ``Html``) ;
    ``droite`` = indices des colonnes alignées à droite, ``largeurs`` =
    largeurs CSS optionnelles par colonne.

    ``defilant`` : enveloppe ``overflow-x:auto`` — par défaut dès 5 colonnes
    (point 9 de la consolidation : les tables serrées à 1024 et 390 défilent
    au lieu de déborder). ``serre`` : 1 = cellules resserrées et en-têtes
    repliés sous 1181 px (classe ``tab-serre``), 2 = police réduite en plus
    (``tab-dense``) — règles dans ``carte.css_kit``. Les en-têtes ``str``
    passent par ``capitales`` ; un en-tête ``Html`` est écrit tel quel."""
    fs = "0.86em" if compact else "0.95em"
    pad = "3px 8px" if compact else "6px 10px"

    def cell(x, i, th=False):
        al = "right" if i in droite else "left"
        w = f"width:{largeurs[i]};" if largeurs and largeurs[i] else ""
        if th:
            s = x if isinstance(x, Html) else _esc(capitales(x))
        else:
            s = x if isinstance(x, Html) else lier_sigles(_esc(x))
            if i in droite and not isinstance(x, Html):
                # un nombre ne se coupe jamais de son unité (« 188,2 / kN » sur deux lignes à 390 px, relecture
                # du 01/11/2026) ; une cellule Html choisit ses propres coupures (matrice des cornières)
                s = s.replace(" ", "&nbsp;")
        tag = "th" if th else "td"
        st_ = (f"padding:{pad};text-align:{al};{w}border-bottom:1px solid #e5e7eb;vertical-align:middle;"
               + (f"color:{C_TRAIT['mut']};font-weight:600;font-size:0.8em;letter-spacing:.04em;"
                  f"border-bottom:1.5px solid {C_ENCRE};white-space:nowrap;" if th else ""))
        return f"<{tag} style='{st_}'>{s}</{tag}>"
    # une table sans en-tête (« libellé | valeur » du calcul à la main) n'en dessine pas : la rangée vide et
    # son trait de 1,5 px se lisaient comme une ligne de tableau perdue (relecture des captures du 01/11/2026)
    tete = ("<thead><tr>" + "".join(cell(e, i, True) for i, e in enumerate(entetes)) + "</tr></thead>"
            if any(str(e) for e in entetes) else "")
    html = [f"<table style='width:100%;border-collapse:collapse;font-size:{fs};margin:2px 0 10px 0;'>",
            tete + "<tbody>"]
    for l in lignes:
        html.append("<tr>" + "".join(cell(x, i) for i, x in enumerate(l)) + "</tr>")
    html.append("</tbody></table>")
    out = "".join(html)
    if defilant is None:
        defilant = len(entetes) >= 5
    classe = {0: "", 1: "tab-serre", 2: "tab-serre tab-dense"}[serre]
    if defilant or classe:
        attr = (f" class='{classe}'" if classe else "") + (" style='overflow-x:auto;'" if defilant else "")
        return f"<div{attr}>{out}</div>"
    return out


def bloc_statut_ligne(etat, titre, complement, elements):
    """Bandeau d'état sur une ligne : icône, statut fort, taux et vérification
    dimensionnante ; en dessous, une ligne de taux par élément. Le statut
    n'est PAS en ``nowrap`` : « PLATINE VÉRIFIÉE (calcul manuel) – sous
    réserve des alertes » mesurait 453 px et sortait de l'écran à 390 px."""
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
        f'<span style="font-weight:700;font-size:1.08em;color:{trait};">{_esc(titre)}</span>'
        f'<span style="color:#374151;">{_esc(complement)}</span></div>'
        f'<div style="font-size:0.88em;margin-top:3px;">{chips}</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def bloc_alerte_ligne(titre, detail, bloquante, selectionnee=False):
    """Une alerte sur une ligne : titre court en gras, détail chiffré (étirée
    dans la rangée de ``alertes``, le bouton à sa droite)."""
    etat = "nok" if bloquante else "warn"
    cadre = f"box-shadow:0 0 0 2px {C_TRAIT[etat]};" if selectionnee else ""
    det = f" <span style='color:#374151;'>{_esc(detail)}</span>" if detail else ""
    html = (
        f'<div style="background-color:{C_COULEURS[etat]};padding:5px 12px;border-radius:8px;font-size:0.92em;'
        f'border:1px solid #d9d9d9;border-left:6px solid {C_TRAIT[etat]};margin:2px 0;{cadre}">'
        f"{'⛔' if bloquante else '⚠️'} <b>{_esc(titre)}</b>{det}</div>"
    )
    st.markdown(html, unsafe_allow_html=True, width="stretch")


# ------------------------------------------------------- titres de bloc
# Un titre de bloc tient dans la case de 76 px, jamais coupé (charte §2). À l'espacement du kit (0,07 em), un
# mot de 11 lettres (« RAIDISSEURS », « APPLICATION ») mesurait 77 px : il se coupait à sa césure. Règle du
# kit (consolidation du 01/11/2026, point 20, mesurée par tools/captures_assemblages.js : « titres de bloc
# sans débordement ») : un titre dont un mot compte ≥ 11 lettres prend l'espacement serré.
ESPACEMENT = ".07em"
ESPACEMENT_SERRE = ".05em"
MOT_LONG = 11


def espacement(txt):
    """L'espacement des lettres d'un titre de bloc : serré dès qu'un mot
    compte ``MOT_LONG`` lettres ou plus."""
    return ESPACEMENT_SERRE if any(len(m) >= MOT_LONG for m in re.findall(r"\w+", str(txt))) else ESPACEMENT


def titre_bloc(txt, help=None):
    """Titre de bloc de la carte (petites capitales, discret) ; ``help`` =
    le « ? » du BLOC (conventions communes dites une fois, P3). Un titre
    peut occuper deux lignes dans sa case de 76 px, jamais être coupé.

    ``margin-bottom:16px`` : le conteneur markdown de Streamlit porte une marge
    de −16 px qui compense la marge d'un ``<p>`` final ; un ``<div>`` nu n'en a
    pas, et le titre débordait de 16 px sous sa ligne — dans la carte du calcul
    manuel, il recouvrait le titre de la ligne suivante (mesuré le 31/10/2026)."""
    st.markdown(f"<div style='font-size:0.72em;font-weight:700;letter-spacing:{espacement(txt)};color:{C_ENCRE};"
                f"line-height:1.15;padding:0 0 9px 0;margin-bottom:16px;'>{_esc(txt)}</div>",
                unsafe_allow_html=True, help=help)


def sous_titre_bloc(txt, help=None):
    """Sous-titre gris d'une continuation « ↳ » ou d'une rangée de table,
    avec son « ? » éventuel (« ↳ AILES B (groupe S) ? », « ↳ ETA ? »). Il ne
    se coupe qu'avant « ( » (espaces insécables ailleurs) : une ligne quand
    la case est pleine largeur (≤ 1350 px), deux dans la case de 76 px."""
    tete, _, reste = str(txt).partition(" (")
    html = "&nbsp;".join(_esc(tete).split(" ")) + (" " + "&nbsp;".join(_esc("(" + reste).split(" ")) if reste else "")
    st.markdown(f"<div style='font-size:0.72em;font-weight:600;color:{C_TRAIT['mut']};line-height:1.15;"
                f"padding:0 0 9px 0;margin-bottom:16px;'>{html}</div>", unsafe_allow_html=True, help=help)


# ------------------------------------------- replis de l'onglet Vérifications
def legende_natures(natures=("EC", "COMP", "INT"), references=None, textes=None):
    """Après la dernière table : une caption d'une ligne (badges + sens des
    natures DU MODULE) puis un expander fermé « Légende et références »
    (``references`` = LEGENDE_REFERENCES, source unique avec le pied de la
    note). ``textes`` remplace le sens d'une nature (le pied : « INT modèle
    d'ingénieur »)."""
    textes = dict(NATURES, **(textes or {}))
    corps = " · ".join(badge_nature(n) + " " + _esc(textes.get(n, "")) for n in natures)
    st.markdown(f"<div style='font-size:0.82em;color:{C_TRAIT['mut']};margin:2px 0 4px 0;'>{corps}</div>",
                unsafe_allow_html=True)
    if references:
        with st.expander("Légende et références"):
            st.caption(references)


def expander_perimetre(non_reverifies, sans_objet, titre_non="Non revérifiés ici",
                       titre_sans="Sans objet dans cette configuration"):
    """« Périmètre de la vérification — n non revérifiés · m sans objet » :
    expander fermé, textes intégraux (rien ne se perd, P7). Le titre dit le
    même mot que le contenu : « n non vérifiés » quand ``titre_non`` est
    « Non vérifiés ici » (calcul manuel ; relecture du 01/11/2026 : le titre
    disait « non revérifiés » et le contenu « Non vérifiés ici »)."""
    n, m = len(non_reverifies), len(sans_objet)
    mot = titre_non.lower().replace(" ici", "").rstrip("s")
    titre = f"Périmètre de la vérification — {n} {mot}{'s' if n > 1 else ''} · {m} sans objet"
    with st.expander(titre):
        if non_reverifies:
            st.caption(titre_non + " : " + " ; ".join(non_reverifies) + ".")
        if sans_objet:
            st.caption(titre_sans + " : " + " ; ".join(sans_objet) + ".")


def tableau_33_replie(ok, rendre, titre="Pinces et entraxes (Tab. 3.3)"):
    """Le Tableau 3.3 replié : le verdict dans le titre de l'expander
    (« … : conforme »), ouvert d'office dès une ligne NON OK ; ``rendre()``
    dessine la table, les boutons Localiser et la caption des règles."""
    with st.expander(titre + " : " + ("conforme" if ok else "NON CONFORME"), expanded=not ok):
        rendre()


# ------------------------------------------------------------- les alertes
LARG_LOCALISER = 104        # « Localiser » / « Masquer » : même largeur, boutons alignés à droite


def selectionner(prefixe, ident):
    """Localiser / Masquer : sélectionne l'alerte ``ident`` (ou la désélectionne)."""
    cle = prefixe + "ui_alerte"
    st.session_state[cle] = None if st.session_state.get(cle) == ident else ident


def titre_remarques(remarques, R, ligne):
    """« ⚠️ 2 remarques — Tiges de coin · Soulèvement » : le titre du repli des
    alertes d'information (le titre court de chacune, avant « : », « — » ou
    « ( » ; le titre entier est dans le repli)."""
    noms = [re.split(r" : | — | \(", ligne(a, R)[0])[0] for a in remarques]
    return "⚠️ " + str(len(remarques)) + " remarques — " + " · ".join(noms)


def _alerte(prefixe, a, R, sel, ligne, champs, texte):
    est_sel = sel == a.id
    titre, detail = ligne(a, R)
    with st.container(horizontal=True, wrap=False, gap="small", vertical_alignment="center",
                      key=f"{prefixe}al_{a.id}"):
        bloc_alerte_ligne(titre, detail, a.block, est_sel)
        st.button("Masquer" if est_sel else "Localiser", key=f"{prefixe}btn_al_{a.id}", width=LARG_LOCALISER,
                  on_click=selectionner, args=(prefixe, a.id))
    if est_sel:
        with st.container(border=True):
            st.markdown(texte(a.why if a.why else a.msg))
            if champs:
                champs(a)
            if a.dims or a.elems:
                st.caption("Sur le dessin : cotes et éléments en cause en rouge.")


def alertes(prefixe, R, ligne, champs=None, texte=None):
    """Les alertes des cinq écrans, UNE forme (consolidation du 01/11/2026,
    point 16) : une rangée par alerte — la ligne courte étirée, le bouton
    « Localiser » / « Masquer » de largeur fixe aligné à droite, sur la même
    rangée à toutes les largeurs (en colonnes, il passait sous l'alerte au
    téléphone) ; les alertes d'INFORMATION regroupées dans un repli
    « ⚠️ n remarques — … » dès qu'il y en a deux, ouvert quand l'une d'elles
    est localisée (le premier écran du téléphone garde le bandeau et le
    dessin). Sélectionnée : l'explication (``texte(why)``), puis ``champs(a)``
    — les pastilles des champs en cause (``carte.pastilles``). Rien ne se
    perd : mêmes textes, mêmes boutons (``<p>btn_al_<id>``)."""
    texte = texte or (lambda x: x)
    cle = prefixe + "ui_alerte"
    sel = st.session_state.get(cle)
    if sel and sel not in [a.id for a in R.alerts]:
        st.session_state[cle] = None
        sel = None
        st.toast("Blocage levé : géométrie valide" if (R.geo_ok and R.dist_ok)
                 else "Ce blocage est levé ; il en reste d'autres")
    remarques = [a for a in R.alerts if not a.block]
    for a in R.alerts:
        if a.block or len(remarques) == 1:
            _alerte(prefixe, a, R, sel, ligne, champs, texte)
    if len(remarques) > 1:
        with st.expander(titre_remarques(remarques, R, ligne), expanded=any(a.id == sel for a in remarques)):
            for a in remarques:
                _alerte(prefixe, a, R, sel, ligne, champs, texte)

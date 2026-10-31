# -*- coding: utf-8 -*-
"""Le KIT de la carte de saisie des assemblages — charte ergonomique du
31/10/2026 (docs/assemblages/ERGONOMIE.md). Un module ne garde que ses
DONNÉES (une ``Description``) ; le kit rend la carte, les trois formats
étant servis par UN seul arbre Streamlit et UNE règle CSS sur nos propres
clés (P2).

Grille (mesurée, voir ERGONOMIE.md §2 et scratchpad/ergo/MESURES_grille.md) :

- une LIGNE D'OBJET = ``st.container(horizontal=True, wrap=True)`` à clé
  ``<p>lg_<bloc>_<n>`` : titre du bloc dans une case de 76 px (clé
  ``<p>bt_<bloc>_<n>``) sur la première ligne du bloc, puis 3 ou 4 widgets
  ``width="stretch"`` — jamais 5 (P1) ;
- une CONTINUATION « ↳ » = même conteneur à clé ``<p>ls_<bloc>_<n>``, sous-
  titre gris dans la case des titres, widgets à largeur FIXE de 150 px,
  jamais étirés ;
- une RANGÉE DE TABLE (mode manuel) = clé ``<p>tr_<bloc>_<i>`` à libellés
  visibles, en-tête ``<p>th_<bloc>`` ; l'un des deux est caché par CSS selon
  la largeur (exception nommée de P2) ;
- la règle CSS (``css_kit``) a quatre paliers : ≥ 1351 titre à gauche et
  4 cases ; 1181–1350 titre au-dessus, 4 cases (base 124) ; 641–1180 2 cases
  (base 205) ; ≤ 640 2 cases (base 160).

Clés de session : ``<p><clé>`` — source unique, écrite AVANT le widget par
``valeur_widget`` ; aucune clé sans le préfixe du module (étanchéité).
"""
import re
import unicodedata
from dataclasses import dataclass, field

import streamlit as st

from .ui_commun import titre_bloc, sous_titre_bloc, _esc

LARG_TITRE = 76          # la case des titres de bloc (px)
LARG_SUITE = 150         # une case de continuation « ↳ » (px)
LARG_TETE = 130          # un réglage court de la rangée d'en-tête (px)


def L(cles, cond=None, suite=None):
    """Une ligne de bloc : clés, condition d'affichage, et — pour une
    continuation « ↳ » — son sous-titre (``""`` pour un « ↳ » nu)."""
    return dict(cles=list(cles), cond=cond, suite=suite)


@dataclass
class Description:
    """Ce qu'un module confie au kit : ses champs (``entrees``), ses textes
    (LIBELLE, AIDES, AFFICHAGE — une forme par clé, P5), sa grille (BLOCS,
    AVANCES, CARTOUCHE, EN_TETE) et ses règles (prédim, verrou)."""
    PREFIXE: str
    CHAMPS: dict
    champ_visible: object
    LIBELLE: dict
    AIDES: dict
    AFFICHAGE: dict
    PAS: dict
    BLOCS: list
    AVANCES: list
    PILOTEES_PAR_PREDIM: tuple
    mode_predim: object
    VISSERIE_DEFAUT: str
    GEOMETRIE_DESSIN: frozenset
    EN_TETE: list = field(default_factory=list)      # [(clé, condition)] : la rangée du mode
    SEGMENTS: frozenset = frozenset()                # clés rendues en segmented_control
    CARTOUCHE: list = field(default_factory=list)    # [(titre, lignes)] de l'expander Cartouche
    LIBELLE_MODE: dict = None                        # {mode: {clé: libellé}} — vocabulaire par mode
    VISIBLE_SUPPL: dict = field(default_factory=dict)   # {clé: cond(u)} hors CHAMPS (visserie, projet…)
    ON_CHANGE: dict = field(default_factory=dict)    # {clé: dict(on_change=f, args=(…))}
    mode_courant: object = None                      # u → mode (pour LIBELLE_MODE)
    TITRE_AVANCES: str = "Paramètres avancés"
    TITRE_CARTOUCHE: str = "Cartouche (note et plan)"


# ----------------------------------------------------------------- CSS
def css_kit(prefixe):
    """La règle CSS du kit pour un préfixe de module — lisible par le
    harnais (``st.markdown``). Quatre paliers mesurés (lot 0, 31/10/2026)."""
    p = prefixe
    lg = f'[class*="st-key-{p}lg_"]'
    ls = f'[class*="st-key-{p}ls_"]'
    tr = f'[class*="st-key-{p}tr_"]'
    th = f'[class*="st-key-{p}th_"]'
    bt = f'[class*="st-key-{p}bt_"]'
    cell = '> [data-testid="stElementContainer"]'
    case = f'> [data-testid="stElementContainer"]:not([class*="st-key-{p}btn_"])'   # pas le bouton 🗑️
    titre = f'> [data-testid="stLayoutWrapper"]:has({bt})'
    # Mesuré (lot 0) : gap="small" vaut 16 px dans un conteneur horizontal ; les lignes du kit
    # passent à 8 px : 4 × 124 + 3 × 8 = 520 ≤ 523 px de colonne à 1181, et dans l'expander des
    # avancés (626 px à 1440) 4 × 128 + 76 + 3 × 8 = 612 tient là où 16 px faisait passer la 4e
    # case seule à la ligne ; rangées de table : 5 cases dont 🗑️ = 4 × 125 + 43 à 1440.
    return (
        "<style>\n"
        f"{lg} {titre}, {ls} {titre}, {tr} {titre} {{flex:0 0 {LARG_TITRE}px !important;}}\n"
        f"{lg}, {tr}, [class*=\"st-key-{p}th_\"] {{gap:0.5rem !important;}}\n"
        f"[class*=\"st-key-{p}barre\"] {{gap:4px 0.5rem !important;}}\n"
        f"@media (min-width:1351px){{\n"
        f"  {tr} {case} {{flex-basis:100px !important;}}\n"
        f"  {tr} [data-testid=\"stWidgetLabel\"] {{display:none !important;}}\n"
        f"}}\n"
        f"@media (max-width:1350px){{\n"
        f"  {lg} {titre}, {ls} {titre}, {tr} {titre} {{flex:0 0 100% !important;margin-top:6px;}}\n"
        # 6 px d'air entre le titre posé au-dessus et la première étiquette (relecture du 31/10 : « collé »).
        f"  {bt} [data-testid=\"stMarkdownContainer\"] > div {{padding-bottom:6px !important;margin-bottom:0 !important;}}\n"
        f"  {bt} [data-testid=\"stMarkdown\"] {{width:fit-content !important;}}\n"
        f"  {lg} {{gap:0.5rem !important;}}\n"
        f"  {lg} {cell} {{flex-basis:124px !important;max-width:calc(25% - 6px);}}\n"
        f"  {tr} {cell} {{flex-basis:205px !important;max-width:calc(50% - 4px);}}\n"
        f"  {th} {{display:none !important;}}\n"
        f"}}\n"
        f"@media (max-width:1180px){{ {lg} {cell} {{flex-basis:205px !important;max-width:calc(50% - 4px);}} }}\n"
        f"@media (max-width:640px){{ {lg} {cell}, {tr} {cell} {{flex-basis:160px !important;}} }}\n"
        "</style>"
    )


def injecter_css(prefixe):
    st.markdown(css_kit(prefixe), unsafe_allow_html=True)


# ------------------------------------------------------------ valeurs
def K(D, k):
    return D.PREFIXE + k


def valeur_widget(D, k, v):
    """La valeur normalisée pour le widget de ``k`` (nombre, option valide,
    texte) — l'état est écrit AVANT l'instanciation du widget."""
    f = D.CHAMPS.get(k)
    if f is None:
        return "" if v is None else str(v)
    if f["t"] == "n":
        if v is None or v == "":
            return None
        try:
            return float(str(v).strip().replace(",", "."))
        except (TypeError, ValueError):
            return None
    if f["t"] == "s":
        s = str(v) if v is not None else f["o"][0]
        if s not in f["o"]:
            try:
                s = str(int(float(s)))
            except (TypeError, ValueError):
                pass
        return s if s in f["o"] else f["o"][0]
    return "" if v is None else str(v)


def affiche(D, k, o):
    """L'option ``o`` de ``k`` telle qu'elle s'affiche (valeur brute inchangée)."""
    return D.AFFICHAGE.get(k, {}).get(o, o)


def aide(D, k):
    """Un « ? » seulement s'il apprend quelque chose : une entrée de AIDES (P3)."""
    return D.AIDES.get(k) or None


def libelle(D, k, u, chauds=frozenset()):
    """LIBELLE[k] (ou LIBELLE_MODE du mode courant), « ◆ » si piloté par le
    prédimensionnement, « 🔴 » si en cause dans l'alerte sélectionnée."""
    lab = None
    if D.LIBELLE_MODE and D.mode_courant:
        lab = D.LIBELLE_MODE.get(D.mode_courant(u), {}).get(k)
    if lab is None:
        lab = D.LIBELLE.get(k, k)
    if D.mode_predim(u) and k in D.PILOTEES_PAR_PREDIM:
        lab += " ◆"
    if k in chauds:
        lab = "🔴 " + lab
    return lab


def _verrou():
    # prj_verrou : assemblage verrouillé dans son projet (acier/projet_ui.py)
    return bool(st.session_state.get("prj_verrou", False))


def _desactive(D, k, u):
    return bool(D.mode_predim(u) and k in D.PILOTEES_PAR_PREDIM) or _verrou()


def visible(D, u, k):
    """Le champ est-il affichable ? ``champ_visible`` pour les clés de
    CHAMPS, ``VISIBLE_SUPPL`` pour les autres (visserie, projet, date)."""
    if k in D.VISIBLE_SUPPL and not D.VISIBLE_SUPPL[k](u):
        return False
    if k in D.CHAMPS:
        return D.champ_visible(u, k)
    return True


# ------------------------------------------------------------ widgets
def widget(D, k, u, chauds=frozenset(), width="stretch"):
    """Le widget d'une clé : selectbox (format_func AFFICHAGE) / number_input
    / text_input ; aide depuis AIDES seulement ; désactivé par le verrou du
    projet ou le prédimensionnement."""
    key = K(D, k)
    if k == "visserie":
        st.session_state.setdefault(key, D.VISSERIE_DEFAUT)
        st.text_input(libelle(D, k, u, chauds), key=key, help=aide(D, k), disabled=_verrou(), width=width)
        return
    f = D.CHAMPS[k]
    st.session_state[key] = valeur_widget(D, k, st.session_state.get(key))
    lab = libelle(D, k, u, chauds)
    dis = _desactive(D, k, u)
    if f["t"] == "s":
        aff = D.AFFICHAGE.get(k)
        st.selectbox(lab, f["o"], key=key, help=aide(D, k), disabled=dis, width=width,
                     format_func=(lambda o, a=aff: a.get(o, o)) if aff else str)
    elif f["t"] == "x":
        st.text_input(lab, key=key, help=aide(D, k), disabled=dis, width=width)
    else:
        st.number_input(lab, key=key, step=D.PAS.get(k, 1.0), format="%g", help=aide(D, k), disabled=dis,
                        width=width)


def segment(D, k, u, chauds=frozenset()):
    """Un choix court en ``segmented_control`` (mode, critère, appui,
    périmètre) : ``required=True`` (jamais None), format_func AFFICHAGE,
    mêmes clé et valeurs brutes que l'ancienne selectbox."""
    key = K(D, k)
    f = D.CHAMPS[k]
    st.session_state[key] = valeur_widget(D, k, st.session_state.get(key))
    aff = D.AFFICHAGE.get(k)
    extra = D.ON_CHANGE.get(k, {})
    st.segmented_control(libelle(D, k, u, chauds), f["o"], key=key, required=True, help=aide(D, k),
                         disabled=_desactive(D, k, u), width="content",
                         format_func=(lambda o, a=aff: a.get(o, o)) if aff else str, **extra)


# -------------------------------------------------------------- lignes
def slug(txt):
    """Un identifiant ASCII pour une clé de conteneur (« EFFORTS ELU » → efforts_elu)."""
    s = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "x"


def _cellule_titre(D, bloc, n, titre, help, suite):
    """La case des titres (76 px) : titre du bloc sur la première ligne du
    bloc, sous-titre « ↳ … » gris sur une continuation."""
    with st.container(key=f"{D.PREFIXE}bt_{bloc}_{n}", width=LARG_TITRE):
        if titre:
            titre_bloc(titre, help=help)
        if suite is not None:
            sous_titre_bloc("↳ " + suite if suite else "↳")


def _ligne(D, bloc, n, cles, u, chauds, titre=None, help=None, suite=None):
    genre = "lg" if suite is None else "ls"
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                      key=f"{D.PREFIXE}{genre}_{bloc}_{n}"):
        if titre or suite is not None:
            _cellule_titre(D, bloc, n, titre, help, suite)
        for k in cles:
            widget(D, k, u, chauds, width="stretch" if suite is None else LARG_SUITE)


def ligne_objet(D, bloc, n, cles, u, chauds=frozenset(), titre=None, help=None):
    """Une ligne d'objet : 3 ou 4 cases étirées, titre discret à gauche."""
    _ligne(D, bloc, n, cles, u, chauds, titre=titre, help=help)


def ligne_suite(D, bloc, n, cles, u, chauds=frozenset(), sous_titre="", titre=None, help=None):
    """Une continuation « ↳ » : cases fixes de 150 px, jamais étirées."""
    _ligne(D, bloc, n, cles, u, chauds, titre=titre, help=help, suite=sous_titre)


def lignes_visibles(D, lignes, u):
    """Les lignes affichables de ``lignes`` avec leurs clés visibles."""
    out = []
    for lg in lignes:
        if lg["cond"] and not lg["cond"](u):
            continue
        cles = [k for k in lg["cles"] if visible(D, u, k)]
        if cles:
            out.append((lg, cles))
    return out


def contient_chaud(D, lignes, chauds, u):
    return any(k in chauds for lg in lignes if not lg["cond"] or lg["cond"](u) for k in lg["cles"])


def bloc(D, titre, lignes, u, chauds=frozenset(), ident=None):
    """Un bloc : son titre (avec le « ? » de AIDES[titre] s'il existe) sur la
    première ligne visible, puis ses lignes d'objet et ses continuations."""
    vis = lignes_visibles(D, lignes, u)
    if not vis:
        return
    ident = ident or slug(titre)
    t = (("🔴 " if contient_chaud(D, lignes, chauds, u) else "") + titre) if titre else ""
    h = D.AIDES.get(titre) if titre else None
    for n, (lg, cles) in enumerate(vis):
        _ligne(D, ident, n, cles, u, chauds, titre=t if n == 0 else None, help=h if n == 0 else None,
               suite=lg["suite"])


# ------------------------------------------------------------- en-tête
def en_tete(D, u, chauds=frozenset()):
    """La rangée d'en-tête de la carte : les choix courts en segmented_control
    (+ les réglages du prédimensionnement quand il est actif)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                      key=f"{D.PREFIXE}en_tete"):
        for k, cond in D.EN_TETE:
            if (cond and not cond(u)) or not visible(D, u, k):
                continue
            if k in D.SEGMENTS:
                segment(D, k, u, chauds)
            else:
                widget(D, k, u, chauds, width=LARG_TETE)


# --------------------------------------------------------------- table
def table(prefixe, bloc, entetes, rangees):
    """Une table de rangées homogènes (boulons, éléments, raidisseurs du
    mode manuel) : en-tête ``<p>th_<bloc>`` (libellés de colonnes, aide
    possible) visible ≥ 1351 px, rangées ``<p>tr_<bloc>_<i>`` à libellés
    visibles en dessous — le CSS du kit cache l'un ou l'autre.

    ``entetes`` = [(libellé, aide)] ; ``rangees`` = [(sous_titre, rendre)]
    où ``rendre()`` instancie les widgets de la rangée (libellés visibles)."""
    # L'en-tête est UN markdown en flex : mêmes largeurs que les rangées (76 px de titre, cases
    # « flex:1 1 100px », gap 8 px, 43 px pour le 🗑️) — mesuré au lot 0 ; des markdown séparés
    # dans un conteneur horizontal se tronquaient (marge négative du conteneur markdown).
    def _cellule(lab, h):
        q = (f" <span title='{_esc(h)}' style='display:inline-block;border:1px solid #6E7480;border-radius:50%;"
             f"width:13px;height:13px;line-height:12px;text-align:center;font-size:9px;cursor:help;'>?</span>"
             if h else "")
        return f"<div style='flex:1 1 100px;min-width:0;white-space:nowrap;'>{_esc(lab)}{q}</div>"
    with st.container(key=f"{prefixe}th_{bloc}"):
        st.markdown(f"<div style='display:flex;gap:8px;align-items:flex-end;font-size:0.78em;color:#6E7480;"
                    f"margin-bottom:-6px;'><div style='flex:0 0 {LARG_TITRE}px;'>n°</div>"
                    + "".join(_cellule(lab, h) for lab, h in entetes)
                    + "<div style='flex:0 0 43px;'></div></div>", unsafe_allow_html=True)
    for i, (sous_titre, rendre) in enumerate(rangees):
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                          key=f"{prefixe}tr_{bloc}_{i}"):
            with st.container(key=f"{prefixe}bt_{bloc}_{i}", width=LARG_TITRE):
                sous_titre_bloc(str(sous_titre))
            rendre()


# -------------------------------------------------------------- replis
def avances(D, u, chauds=frozenset()):
    """« Paramètres avancés » : même grille, sous-blocs titrés ; ouvert
    (🔴) quand l'alerte sélectionnée y pointe."""
    chaud = any(contient_chaud(D, lignes, chauds, u) for _, lignes in D.AVANCES)
    with st.expander(("🔴 " if chaud else "") + D.TITRE_AVANCES, expanded=chaud):
        for titre, lignes in D.AVANCES:
            bloc(D, titre, lignes, u, chauds, ident="av_" + slug(titre))


def cartouche(D, u, chauds=frozenset()):
    """« Cartouche (note et plan) » : repère, rédacteur, visserie ; projet et
    date seulement hors projet."""
    with st.expander(D.TITRE_CARTOUCHE):
        for titre, lignes in D.CARTOUCHE:
            bloc(D, titre, lignes, u, chauds, ident="cart_" + slug(titre or "cartouche"))


def carte(D, u, chauds=frozenset()):
    """La carte complète : en-tête, blocs, puis les deux replis."""
    with st.container(gap=None):
        if D.EN_TETE:
            en_tete(D, u, chauds)
        for titre, lignes in D.BLOCS:
            bloc(D, titre, lignes, u, chauds)
    avances(D, u, chauds)
    cartouche(D, u, chauds)


# --------------------------------------------------------- contrôles
def _toutes_lignes(D):
    for _, lignes in list(D.BLOCS) + list(D.AVANCES) + list(D.CARTOUCHE):
        for lg in lignes:
            yield lg


def cles_carte(D):
    """Toutes les clés que la carte peut rendre (en-tête, blocs, avancés,
    cartouche) — pour le contrôle « rien ne se perd »."""
    out = {k for k, _ in D.EN_TETE}
    for lg in _toutes_lignes(D):
        out.update(lg["cles"])
    return out


def rangees(D):
    """``(genre, clés)`` de chaque ligne déclarée : « objet » ou « suite »."""
    return [("objet" if lg["suite"] is None else "suite", list(lg["cles"])) for lg in _toutes_lignes(D)]


def aides_visibles(D, u):
    """Les « ? » visibles dans la CARTE pour l'état ``u`` : en-tête, lignes
    d'objet, continuations visibles, « ? » de bloc — hors avancés, cartouche
    et tête du dessin."""
    out = []
    for k, cond in D.EN_TETE:
        if (not cond or cond(u)) and visible(D, u, k) and aide(D, k):
            out.append(k)
    for titre, lignes in D.BLOCS:
        vis = lignes_visibles(D, lignes, u)
        if not vis:
            continue
        if D.AIDES.get(titre):
            out.append(titre)
        for _, cles in vis:
            out.extend(k for k in cles if aide(D, k))
    return out

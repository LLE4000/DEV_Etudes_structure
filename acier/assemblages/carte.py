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

Consolidation du 01/11/2026 (étape 7) : les briques que les modules avaient
écrites localement pendant le gel du kit reviennent ici — sous-titres « ↳ » à
« ? », bloc à 🔴 de pièce, table à en-tête natif, rangées et continuations
d'actions, CSS unique (colonne figée, tables serrées, alertes), pastilles des
alertes, clic du dessin, panneau Cotes, garde-fou des dimensions vides.

Clés de session : ``<p><clé>`` — source unique, écrite AVANT le widget par
``valeur_widget`` ; aucune clé sans le préfixe du module (étanchéité).
"""
import re
import unicodedata
from dataclasses import dataclass, field

import streamlit as st

from acier.js import N, js_str
from .ui_commun import titre_bloc, sous_titre_bloc

LARG_TITRE = 76          # la case des titres de bloc (px)
LARG_SUITE = 150         # une case de continuation « ↳ » (px)
LARG_TETE = 130          # un réglage court de la rangée d'en-tête (px)
LARG_POUBELLE = 43       # le bouton 🗑️ d'une rangée de table (mesuré au lot 0)
# Point 1 de la consolidation : une ligne d'objet de 2 cases ne s'étire pas (EFFORTS ELU de la platine
# d'about au défaut : 2 cases de 284 px à 1440 contre 138 dans les rangées de 4) — une case ne dépasse pas
# celle d'une rangée de 3 à 1440 : (660 − 76 − 3 × 8) / 3 = 186,7 px.
LARG_CASE_MAX = 187


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
    # Hors de la carte (pastilles d'alerte, toasts du dessin), une clé que seul son sous-titre « ↳ »
    # situe dit son groupe : « Rangées n1 (groupe S) » et non « Rangées n1 » (relecture du 01/11/2026).
    LIBELLE_LONG: dict = field(default_factory=dict)
    # Les dimensions qui doivent être > 0 quand elles sont visibles (profil Personnalisé, tube) et leur
    # valeur PROVISOIRE pour le dessin tant qu'elles sont vides (garde-fou des dimensions vides).
    DIMENSIONS: dict = field(default_factory=dict)


# ----------------------------------------------------------------- CSS
def css_kit(prefixe):
    """La règle CSS du kit pour un préfixe de module — lisible par le
    harnais (``st.markdown``). Quatre paliers mesurés (lot 0, 31/10/2026),
    plus, depuis la consolidation du 01/11/2026, les règles que les modules
    écrivaient localement : colonne du dessin figée, rangées de table (cases
    par deux avec le 🗑️ à leur droite sous 1351 px), titre de table
    ``<p>tt_``, tables serrées (``tab-serre`` / ``tab-dense``), rangées
    d'alerte, panneau Cotes (4 / 3 / 2 colonnes)."""
    p = prefixe
    lg = f'[class*="st-key-{p}lg_"]'
    ls = f'[class*="st-key-{p}ls_"]'
    tr = f'[class*="st-key-{p}tr_"]'
    th = f'[class*="st-key-{p}th_"]'
    tt = f'[class*="st-key-{p}tt_"]'
    bt = f'[class*="st-key-{p}bt_"]'
    cotes = f'[class*="st-key-{p}cotes"]'
    cell = '> [data-testid="stElementContainer"]'
    case = f'> [data-testid="stElementContainer"]:not([class*="st-key-{p}btn_"])'   # pas le bouton 🗑️
    bouton = f'> [data-testid="stElementContainer"][class*="st-key-{p}btn_"]'
    titre = f'> [data-testid="stLayoutWrapper"]:has({bt})'
    nu = f'> [data-testid="stLayoutWrapper"]:has([class*="st-key-{p}bn_"])'      # « ↳ » nu
    md = '[data-testid="stMarkdownContainer"] > div'
    # Mesuré (lot 0) : gap="small" vaut 16 px dans un conteneur horizontal ; les lignes du kit
    # passent à 8 px : 4 × 124 + 3 × 8 = 520 ≤ 523 px de colonne à 1181, et dans l'expander des
    # avancés (626 px à 1440) 4 × 128 + 76 + 3 × 8 = 612 tient là où 16 px faisait passer la 4e
    # case seule à la ligne ; rangées de table : 5 cases dont 🗑️ = 4 × 125 + 43 à 1440.
    return (
        "<style>\n"
        # La colonne du dessin est FIGÉE (sticky ≥ 641 px), reconnue par son contenu ; le padding bas par
        # défaut (10rem) faisait « garer » la colonne sous le haut de l'écran en fin de page.
        f"@media (min-width:641px){{div[data-testid=\"stColumn\"]:has(div.st-key-{p}col_dessin){{position: sticky;"
        f"top:2.875rem;align-self:flex-start;max-height:calc(100vh - 3.5rem);overflow-y:auto;scrollbar-width:thin;}}\n"
        f"  div[data-testid=\"stMainBlockContainer\"]{{padding-bottom:1.5rem;}}}}\n"
        f"{lg} {titre}, {ls} {titre}, {tr} {titre}, {th} {titre}, {ls} {nu} {{flex:0 0 {LARG_TITRE}px !important;}}\n"
        f"{lg}, {ls}, {tr}, {th}, {cotes} {{gap:0.5rem !important;}}\n"
        f"[class*=\"st-key-{p}barre\"], [class*=\"st-key-{p}tete_dessin\"] {{gap:4px 0.5rem !important;}}\n"
        # la rangée d'une alerte : sans la marge −16 px du conteneur markdown (bas du cadre rogné)
        f"[class*=\"st-key-{p}al_\"] [data-testid=\"stMarkdownContainer\"]{{margin-bottom:0 !important;}}\n"
        f"{cotes} {cell} {{flex:1 1 calc(25% - 6px) !important;max-width:calc(25% - 6px);}}\n"
        f"@media (min-width:1351px){{\n"
        f"  {lg} {cell} {{max-width:{LARG_CASE_MAX}px;}}\n"
        # Une continuation de 4 ou 5 cases passe sur deux rangées : la seconde repartait sous la colonne des
        # titres (« Congé », « tf », « r » à 1440, point 12) — retrait de la case des titres (76 + 8 px), la case
        # du sous-titre ramenée dans ce retrait : la 2e rangée s'aligne sous la première case.
        f"  {ls} {{padding-left:{LARG_TITRE + 8}px;}}\n"
        f"  {ls} {titre}, {ls} {nu} {{margin-left:-{LARG_TITRE + 8}px;}}\n"
        f"  {tr} {case} {{flex-basis:100px !important;}}\n"
        # l'en-tête de table s'étire comme les cases de ses rangées (ses libellés tombent sur leurs colonnes)
        f"  {th} {case} {{flex:1 1 100px !important;min-width:0;}}\n"
        # … et son « ? » suit le libellé (au bout de la case, il se lisait comme celui de la colonne suivante)
        f"  {th} [data-testid=\"stMarkdown\"] {{width:fit-content !important;}}\n"
        f"  {tr} [data-testid=\"stWidgetLabel\"], {tt} {{display:none !important;}}\n"
        f"}}\n"
        f"@media (max-width:1350px){{\n"
        # Le titre (ou le sous-titre « ↳ ») posé au-dessus de sa rangée lui appartient : 12 px au-dessus, l'écart
        # de rangée de 8 px au-dessous (point 12 de la consolidation : mesuré sur l'encre, 4 px au-dessus pour 6
        # au-dessous — le sous-titre semblait suivre la rangée précédente). Le texte garde sa marge de 16 px qui
        # compense celle (−16 px) du conteneur markdown : sans elle, il chevauchait de 4 à 5 px la case suivante
        # — invisible sur une étiquette, il rognait « RAIDISSEURS » au-dessus du bouton ➕ (mesuré le 01/11/2026).
        f"  {lg} {titre}, {ls} {titre}, {tr} {titre} {{flex:0 0 100% !important;margin-top:12px;}}\n"
        f"  {ls} {nu} {{flex:0 0 auto !important;width:auto !important;}}\n"
        f"  {bt} {md}, {tt} {md} {{padding-bottom:0 !important;margin-bottom:16px !important;}}\n"
        f"  {bt} [data-testid=\"stMarkdown\"], {tt} [data-testid=\"stMarkdown\"] {{width:fit-content !important;}}\n"
        f"  {tt} {{margin-top:12px;}}\n"
        f"  {lg} {{gap:0.5rem !important;}}\n"
        # {case} et non {cell} : le bouton d'une ligne d'action (« RAIDISSEURS | ➕ Raidisseur », clé btn_)
        # garde sa largeur de contenu — étiré à la base d'une case (124 / 205 / 160 px), il ne ressemblait plus
        # à « ➕ Élément soudé » juste au-dessus (captures du 01/11/2026)
        f"  {lg} {case} {{flex-basis:124px !important;max-width:calc(25% - 6px);}}\n"
        # une rangée de table range ses cases par deux AVEC le 🗑️ à leur droite (x · y · 🗑️ sur une rangée) :
        # une base de 205 / 160 px donnait au 🗑️ une demi-rangée à lui seul (constaté à 1024 et 390)
        f"  {tr} {case} {{flex:1 1 calc(50% - 30px) !important;max-width:calc(50% - 30px) !important;}}\n"
        f"  {tr} {bouton} {{flex:0 0 auto !important;max-width:none !important;}}\n"
        f"  {th} {{display:none !important;}}\n"
        # Deux tables côte à côte (``<p>duo_`` : côté comprimé · côté tendu de la vérification complète) passent
        # l'une sous l'autre : à 1024, deux demi-colonnes de 210 px, et la seconde table sortait de 26 px de la
        # page (captures du 01/11/2026)
        f"  [class*=\"st-key-{p}duo_\"] [data-testid=\"stHorizontalBlock\"] {{flex-wrap:wrap !important;}}\n"
        f"  [class*=\"st-key-{p}duo_\"] [data-testid=\"stColumn\"] {{flex:1 1 100% !important;min-width:100% !important;}}\n"
        f"}}\n"
        # Point 13 : le panneau Cotes en 3 colonnes sous 1181 px (4 × 90 px à 1024 coupait « e1 groupe S
        # (mm) » sur deux lignes), 2 sous 641 — la même règle que la carte.
        f"@media (max-width:1180px){{ {lg} {case} {{flex-basis:205px !important;max-width:calc(50% - 4px);}}\n"
        f"  {cotes} {cell} {{flex:1 1 calc(33.33% - 6px) !important;max-width:calc(33.33% - 6px);}}\n"
        # Une continuation de trois cases (« ↳ AILES A » : Fixation · Rangées n1 · Files n2 ; « ↳ » des efforts :
        # Vx · Vy · TEd) tient sur une rangée : 3 × 150 + 2 × 8 = 466 px ne passaient pas dans les 440 px de la
        # colonne à 1024, et la 3e case partait seule à la ligne (captures du 01/11/2026). Les cases se partagent
        # la rangée entre 120 px (le plancher des cases de nombre) et leurs 150 px.
        f"  {ls} {case} {{flex:1 1 120px !important;min-width:120px !important;max-width:{LARG_SUITE}px !important;"
        f"width:auto !important;}}\n"
        # Tables serrées (ui_commun.tableau_html(serre=)) : tab-serre = cellules resserrées, en-têtes repliés
        # (matrice des cornières : 430 → 309 px de large au plus serré, 699 → 519 px de haut à 1024, mesuré
        # le 31/10/2026) ; tab-dense = police réduite en plus (tables de détail du calcul manuel, 9 et 8
        # colonnes : 470 px avant, débordement au téléphone).
        f"  .tab-serre th, .tab-serre td {{padding:3px 4px !important;}} .tab-serre th {{white-space:normal !important;}}\n"
        f"  .tab-dense table {{font-size:0.76em !important;}} .tab-dense th, .tab-dense td {{padding:2px 4px !important;}}\n"
        f"  .tab-dense th {{letter-spacing:0 !important;}} }}\n"
        f"@media (max-width:640px){{ {lg} {case} {{flex-basis:160px !important;}}\n"
        f"  {cotes} {cell} {{flex:1 1 calc(50% - 4px) !important;max-width:calc(50% - 4px);}}\n"
        f"  .tab-dense table {{font-size:0.7em !important;}} .tab-dense th, .tab-dense td {{padding:2px 3px !important;}} }}\n"
        "</style>"
    )


def injecter_css(prefixe, *extra):
    """Le CSS du kit (et celui d'un module, ``extra``) en UN seul élément
    ``<style>`` : mis bout à bout dans un même markdown, la troisième balise
    était lue comme du texte de la deuxième et ses règles perdues (mesuré le
    31/10/2026) ; et chaque élément vide coûte 16 px d'écart."""
    corps = [b.replace("<style>", "").replace("</style>", "").strip() for b in (css_kit(prefixe),) + extra]
    st.markdown("<style>\n" + "\n".join(corps) + "\n</style>", unsafe_allow_html=True)


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


def cle_suite(suite):
    """La clé d'aide d'un sous-titre « ↳ » : « AILES B » pour « AILES B (groupe S) »."""
    return str(suite).split(" (", 1)[0] if suite else None


def aide_suite(D, suite):
    """Le « ? » d'un sous-titre « ↳ » : AIDES[cle_suite(suite)] (« ↳ AILES B ? »,
    « ↳ ETA ? ») — une entrée de AIDES, comme tout « ? » (P3)."""
    return D.AIDES.get(cle_suite(suite)) if suite else None


def _cle_titre(prefixe, ident, titre, suite):
    """La clé de la case des titres : ``<p>bt_…`` ; ``<p>bn_…`` pour un « ↳ »
    NU (ni titre ni texte de sous-titre) — il reste à gauche de sa rangée à
    toutes les largeurs (sous 1351 px, seul sur une ligne pleine largeur, il
    s'éloignait de ses cases : mesuré le 01/11/2026)."""
    return f"{prefixe}{'bn' if (not titre and suite == '') else 'bt'}_{ident}"


def _cellule_titre(D, bloc, n, titre, help, suite):
    """La case des titres (76 px) : titre du bloc sur la première ligne du
    bloc, sous-titre « ↳ … » gris (avec son « ? ») sur une continuation."""
    with st.container(key=_cle_titre(D.PREFIXE, f"{bloc}_{n}", titre, suite), width=LARG_TITRE):
        if titre:
            titre_bloc(titre, help=help)
        if suite is not None:
            sous_titre_bloc("↳ " + suite if suite else "↳", help=aide_suite(D, suite))


def _ligne(D, bloc, n, cles, u, chauds, titre=None, help=None, suite=None):
    genre = "lg" if suite is None else "ls"
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                      key=f"{D.PREFIXE}{genre}_{bloc}_{n}"):
        if titre or suite is not None:
            _cellule_titre(D, bloc, n, titre, help, suite)
        for k in cles:
            widget(D, k, u, chauds, width="stretch" if suite is None else LARG_SUITE)
    vides = [k for k in dimensions_vides(D, u) if k in cles]
    if vides:
        # garde-fou des dimensions vides : le message court SOUS la ligne qui les porte
        st.caption("⚠️ À saisir : " + ", ".join(D.LIBELLE.get(k, k).split(" (")[0] for k in vides)
                   + " — vérification suspendue.")


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


def bloc(D, titre, lignes, u, chauds=frozenset(), ident=None, chaud=False):
    """Un bloc : son titre (avec le « ? » de AIDES[titre] s'il existe) sur la
    première ligne visible, puis ses lignes d'objet et ses continuations (un
    sous-titre « ↳ » porte le « ? » de AIDES[cle_suite]). ``chaud`` allume le
    🔴 du titre quand l'alerte sélectionnée est portée par une PIÈCE du
    dessin (« anchors », « col ») et pas seulement par un champ."""
    vis = lignes_visibles(D, lignes, u)
    if not vis:
        return
    ident = ident or slug(titre)
    t = (("🔴 " if (chaud or contient_chaud(D, lignes, chauds, u)) else "") + titre) if titre else ""
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


# --------------------------------------------------- tables et actions
def rangee(prefixe, bloc, i, sous_titre, rendre):
    """Une rangée de table ``<p>tr_<bloc>_<i>`` : sous-titre (n° 1, É1, R1)
    dans la case des titres, widgets à libellés visibles (cachés ≥ 1351 px,
    où l'en-tête ``<p>th_<bloc>`` les porte)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                      key=f"{prefixe}tr_{bloc}_{i}"):
        with st.container(key=f"{prefixe}bt_{bloc}_{i}", width=LARG_TITRE):
            sous_titre_bloc(str(sous_titre))
        rendre()


def suite(prefixe, ident, sous_titre, rendre):
    """Une continuation « ↳ » ``<p>ls_<ident>`` dont les cases sont rendues
    par ``rendre()`` (cases de 150 px, boutons ➕). ``ident`` ne doit pas être
    celui d'une table (« elems_0 » donnerait la clé ``<p>bt_elems_0`` de la
    rangée É1 : clé dupliquée, trouvée par le harnais)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                      key=f"{prefixe}ls_{ident}"):
        with st.container(key=_cle_titre(prefixe, ident, None, sous_titre or ""), width=LARG_TITRE):
            sous_titre_bloc("↳ " + sous_titre if sous_titre else "↳")
        rendre()


def ligne_titre(prefixe, ident, titre, aide, rendre, chaud=False):
    """Une ligne d'objet titrée dont la seule case est une action (table vide :
    « ÉLÉMENTS ? | ➕ Élément soudé »)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                      key=f"{prefixe}lg_{ident}_0"):
        with st.container(key=f"{prefixe}bt_{ident}_0", width=LARG_TITRE):
            titre_bloc(("🔴 " if chaud else "") + titre, help=aide)
        rendre()


def _entete_cellule(lab, h):
    """Un libellé de colonne de l'en-tête, avec son « ? » NATIF (``help=`` :
    une infobulle qui s'ouvre aussi au toucher, là où un ``span title`` ne
    s'ouvrait qu'au survol de la souris)."""
    st.markdown(f"<div style='font-size:0.78em;color:#6E7480;white-space:nowrap;line-height:1.15;"
                f"padding:0 0 4px 0;margin-bottom:16px;'>{lab}</div>", unsafe_allow_html=True, help=h, width="stretch")


def table(prefixe, bloc, entetes, rangees, titre=None, aide=None, chaud=False):
    """Une table de rangées homogènes (boulons, éléments, raidisseurs du
    mode manuel) : en-tête ``<p>th_<bloc>`` visible ≥ 1351 px — la case des
    titres porte « n° » ou le TITRE du bloc (avec son « ? »), puis un libellé
    par colonne avec son « ? » natif, puis la place du 🗑️ ; dessous, les
    rangées ``<p>tr_<bloc>_<i>`` à libellés visibles — le CSS du kit cache
    l'un ou l'autre selon la largeur. Sous 1351 px le titre est rendu une
    seconde fois sur sa ligne ``<p>tt_<bloc>``, cachée au-dessus.

    ``entetes`` = [(libellé, aide)] ; ``rangees`` = [(sous_titre, rendre)]
    où ``rendre()`` instancie les widgets de la rangée (libellés visibles),
    ou [(sous_titre, rendre, (ident, sous_titre_suite, rendre_suite))] quand
    la rangée est suivie de sa continuation « ↳ É1 » (``suite``).

    L'en-tête est un conteneur horizontal de la même grille que les rangées
    (76 px de titre, cases à base 100 px, gap 8 px, 43 px pour le 🗑️) ; son
    texte compense la marge −16 px du conteneur markdown : l'ancien en-tête
    (un seul markdown en flex) mesurait 0 px de haut et les rangées le
    recouvraient (mesuré le 31/10/2026 : « x (mm) » invisible)."""
    from .ui_commun import _esc
    t = ("🔴 " if chaud else "") + titre if titre else None
    with st.container(horizontal=True, wrap=False, gap="small", vertical_alignment="bottom",
                      key=f"{prefixe}th_{bloc}"):
        with st.container(key=f"{prefixe}bt_th_{bloc}", width=LARG_TITRE):
            if t:
                titre_bloc(t, help=aide)
            else:
                _entete_cellule("n°", None)
        for lab, h in entetes:
            _entete_cellule(_esc(lab), h)
        with st.container(key=f"{prefixe}btn_th_{bloc}", width=LARG_POUBELLE):
            # la place du 🗑️ — un conteneur VIDE n'en prenait aucune : les libellés glissaient de 25 px
            st.markdown("&nbsp;", unsafe_allow_html=True)
    if t:
        with st.container(key=f"{prefixe}tt_{bloc}"):
            titre_bloc(t, help=aide)
    for i, (sous_titre, rendre, *ensuite) in enumerate(rangees):
        rangee(prefixe, bloc, i, sous_titre, rendre)
        if ensuite and ensuite[0]:
            suite(prefixe, *ensuite[0])


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


def carte(D, u, chauds=frozenset(), chaudes=frozenset()):
    """La carte complète : en-tête, blocs, puis les deux replis. ``chaudes`` :
    les titres de bloc dont le 🔴 est allumé par une pièce du dessin."""
    with st.container(gap=None):
        if D.EN_TETE:
            en_tete(D, u, chauds)
        for titre, lignes in D.BLOCS:
            bloc(D, titre, lignes, u, chauds, chaud=titre in chaudes)
    avances(D, u, chauds)
    cartouche(D, u, chauds)


# ------------------------------------------------ hors carte : même clé
def libelle_long(D, k, u=None, long=True):
    """Le libellé d'une clé HORS de la carte (pastille d'alerte, toast du
    dessin, panneau Cotes) : LIBELLE_LONG s'il existe (« Rangées n1
    (groupe S) »), sinon celui du mode (LIBELLE_MODE), sinon LIBELLE.
    ``long=False`` : une fenêtre d'objet, dont le titre situe déjà la clé
    (« Groupe S »), garde le libellé de la carte."""
    if long and k in D.LIBELLE_LONG:
        return D.LIBELLE_LONG[k]
    if D.LIBELLE_MODE and D.mode_courant and u is not None:
        lab = D.LIBELLE_MODE.get(D.mode_courant(u), {}).get(k)
        if lab:
            return lab
    return D.LIBELLE.get(k, k)


def _ecrire_depuis(D, cle_widget, k):
    """Callback : recopie la valeur d'un widget secondaire (pastille d'alerte,
    panneau des cotes) dans la clé de session du champ (source unique)."""
    st.session_state[K(D, k)] = valeur_widget(D, k, st.session_state.get(cle_widget))


def pastille(D, k, u, disabled=False):
    """La pastille d'une alerte : le même champ, modifiable sur place (libellé
    hors carte, options AFFICHAGE) ; écrit dans la source unique."""
    f = D.CHAMPS[k]
    cle = D.PREFIXE + "fix_" + k
    st.session_state[cle] = valeur_widget(D, k, st.session_state.get(K(D, k)))
    lab = libelle_long(D, k, u)
    if f["t"] == "s":
        aff = D.AFFICHAGE.get(k)
        st.selectbox(lab, f["o"], key=cle, on_change=_ecrire_depuis, args=(D, cle, k), disabled=disabled,
                     format_func=(lambda o, a=aff: a.get(o, o)) if aff else str)
    elif f["t"] == "x":
        st.text_input(lab, key=cle, on_change=_ecrire_depuis, args=(D, cle, k), disabled=disabled)
    else:
        st.number_input(lab, key=cle, step=D.PAS.get(k, 1.0), format="%g",
                        on_change=_ecrire_depuis, args=(D, cle, k), disabled=disabled)


def pastilles(D, a, u, pred=False):
    """Les champs en cause d'une alerte sélectionnée, modifiables sur place
    (le ``champs(a)`` de ``ui_commun.alertes``) ; en prédimensionnement les
    champs pilotés (◆) sont désactivés, et le verrou du projet les fige."""
    if pred and any(k in D.PILOTEES_PAR_PREDIM for k in a.fields):
        st.caption("Mode prédimensionnement : les champs marqués ◆ sont pilotés par la proposition (onglet "
                   "Prédim, ou mode Vérification pour les modifier).")
    vis = [k for k in a.fields if k in D.CHAMPS and D.champ_visible(u, k)]
    if vis:
        st.caption("Champs en cause — modifiables ici :")
        cols = st.columns(min(4, len(vis)))
        for i, k in enumerate(vis):
            with cols[i % len(cols)]:
                pastille(D, k, u, disabled=bool(pred and k in D.PILOTEES_PAR_PREDIM) or _verrou())


def traiter_clic(D, retour, cle_cmp, u, derivee=None):
    """Un message validé du dessin (cote, poignée, fenêtre d'objet) → source
    unique → toast « <libellé> = <valeur> » → recalcul. ``derivee(k, v)``
    écrit une clé dérivée (boulon déplacé du calcul manuel) et rend son
    texte, None si ce n'en est pas une."""
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault(D.PREFIXE + "ui_dernier_clic", {})
    if derniers.get(cle_cmp) == retour.get("t"):
        return
    derniers[cle_cmp] = retour.get("t")
    changes = retour.get("changes")
    if changes is None and retour.get("key"):
        changes = {retour["key"]: retour.get("value")}
    ecrits = []
    for k, v in (changes or {}).items():
        if k in D.CHAMPS:
            st.session_state[K(D, k)] = valeur_widget(D, k, v)
            ecrits.append(libelle_long(D, k, u) + " = " + js_str(valeur_widget(D, k, v)))
        elif derivee is not None:
            txt = derivee(k, v)
            if txt:
                ecrits.append(txt)
    if ecrits:
        st.toast(" ; ".join(sorted(set(ecrits)) if derivee else ecrits)
                 + (" : recalculé" if derivee else " : géométrie et vérifications recalculées"))
        st.rerun()


def champ_fenetre(D, k, u, pred=False, aides=None):
    """Une saisie d'une fenêtre d'objet du dessin — une VUE de la clé : même
    libellé (hors carte), même aide, mêmes options affichées."""
    f = D.CHAMPS[k]
    return dict(key=k, lab=libelle_long(D, k, u, long=False), type=f["t"], options=f.get("o") or [], step=D.PAS.get(k, 1.0),
                dis=bool(pred and k in D.PILOTEES_PAR_PREDIM), help=(aides or D.AIDES).get(k),
                affichage=D.AFFICHAGE.get(k) or None)


def panneau_cotes(D, cotes, u, chauds=frozenset(), aides=None):
    """Le panneau « Cotes » (dessin fixe, tête de la colonne de droite) : une
    saisie par cote, libellé hors carte, aide AIDES — grille du kit
    (``<p>cotes``) : 4 colonnes ≥ 1181 px, 3 jusqu'à 641, 2 au téléphone
    (point 13 : 4 × 90 px à 1024). ``chauds`` : les clés en cause dans
    l'alerte sélectionnée (🔴)."""
    with st.expander("Cotes", expanded=True):
        if not cotes:
            st.caption("Aucune cote modifiable dans cette configuration.")
            return
        with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom",
                          key=D.PREFIXE + "cotes"):
            for k in cotes:
                cle = D.PREFIXE + "cote_" + k
                st.session_state[cle] = valeur_widget(D, k, st.session_state.get(K(D, k)))
                st.number_input(("🔴 " if k in chauds else "") + libelle_long(D, k, u), key=cle,
                                step=D.PAS.get(k, 1.0), format="%g", width="stretch",
                                on_change=_ecrire_depuis, args=(D, cle, k), help=(aides or D.AIDES).get(k))


def cotes_du_panneau(D, u, emises, ordre):
    """Les clés du panneau Cotes : les cotes que le dessin émet (``emises``,
    dans l'ordre de pose — en Vue simple, celles de l'Édition : point 15,
    le panneau était vide), puis la géométrie visible qu'il n'émet pas
    (``ordre`` ∩ champ_visible) : le dessin fixe n'ouvre aucune fenêtre
    d'objet, le panneau est alors le seul lieu de la géométrie (rien ne se
    perd)."""
    out = []
    for k in emises:
        if k in D.GEOMETRIE_DESSIN and k not in out:
            out.append(k)
    out += [k for k in ordre if k in D.GEOMETRIE_DESSIN and k not in out and D.champ_visible(u, k)]
    return out


# ------------------------------------------------------- hors projet
def hors_projet(u):
    """Hors projet (nom du projet vide) : le cartouche de l'écran porte
    Projet ET Date — la condition même de ``projet_ui.infos_cartouche``
    (décision du 01/11/2026, points 3 et 22 : la date de session, toujours
    initialisée par la page projet, cachait Date dans les cinq écrans)."""
    return not st.session_state.get("nom_projet")


HORS_PROJET = {"id_projet": hors_projet, "id_date": hors_projet}


# ---------------------------------------------- garde-fou des dimensions
def dimensions_vides(D, u):
    """Les dimensions visibles (profil Personnalisé, tube, cornière) vides ou
    ≤ 0 — points 8 et 19 de la consolidation : le calcul donnait des taux
    absurdes (> 999 %) ou levait ZeroDivisionError (pied de poteau, tube sans
    dimensions : dessin disparu, cases étirées à 594 px)."""
    return [k for k in D.DIMENSIONS if visible(D, u, k) and not (N(u.get(k)) or 0) > 0]


def dimensions_profil(nom, h, b, tw=None, tf=None, liste="profils", champs=("h", "b", "tw", "tf")):
    """Les valeurs PROVISOIRES des dimensions d'un profil (ou d'une cornière)
    resté vide : celles de ``nom`` dans le catalogue (le profil par défaut
    du module) — clés de saisie ``h``, ``b``, ``tw``, ``tf``."""
    from acier.bibliotheques import DB, find
    x = find(DB[liste], nom)
    return {k: x[c] for k, c in zip((h, b, tw, tf), champs) if k}


def provisoire(D, u):
    """Les saisies avec les dimensions vides remplacées par leur valeur
    PROVISOIRE (DIMENSIONS) : le dessin reste à l'écran pendant la saisie ;
    la vérification, elle, est suspendue (aucun taux n'est montré)."""
    return dict(u, **{k: D.DIMENSIONS[k] for k in dimensions_vides(D, u)})


def message_dimensions(D, u):
    """Le bandeau de l'écran tant que des dimensions manquent."""
    vides = dimensions_vides(D, u)
    return ("Dimensions à saisir dans la carte : " + ", ".join(D.LIBELLE.get(k, k).split(" (")[0] for k in vides)
            + ". Vérification suspendue ; le dessin garde des dimensions provisoires.")


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
    d'objet, continuations visibles et leurs sous-titres « ↳ », « ? » de
    bloc — hors avancés, cartouche et tête du dessin."""
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
        for lg, cles in vis:
            if aide_suite(D, lg["suite"]):
                out.append(cle_suite(lg["suite"]))
            out.extend(k for k in cles if aide(D, k))
    return out

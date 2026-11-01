# -*- coding: utf-8 -*-
"""L'écran du CALCUL MANUEL de la platine : la carte (colonne de droite —
en-tête, matériaux, boulons en table, éléments soudés, raidisseurs, efforts)
et les résultats (bandeau, dessins cliquables, onglets Vérifications ·
Efforts des tiges · Note · Benchmark · Méthode).

Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md §4.5) : les
lignes d'objet sont celles du kit (``ecran_saisie.D_MANUEL``) ; les tables
(boulons, éléments, raidisseurs) ont des rangées ``ppb_tr_<table>_<i>`` à
libellés visibles et un en-tête ``ppb_th_<table>`` — le CSS du kit cache
l'un ou l'autre selon la largeur (exception nommée de P2). hef et dh ont
quitté la carte : ils se règlent sur la coupe A-A, dans les fenêtres Boulons
et Platine du dessin, ou dans le panneau Cotes.

SOURCE UNIQUE : les clés ``ppb_<clé>`` du module ; les listes ``ppb_tiges``,
``ppb_elems``, ``ppb_raid`` sont la source des boulons, éléments et
raidisseurs — les widgets ligne par ligne et les glissements du dessin y
écrivent (clés dérivées « t3_u » : distance au bord gauche, « t3_v » : au
bord haut). La bascule de mode reprend la disposition (conversion.py).
"""
import streamlit as st

from acier.js import N, js_str
from acier.formats import F, pct
from acier.assemblages import carte as kit
from acier.assemblages.ui_commun import (bloc_statut_ligne, badge_nature, tableau_md, tableau_html, taux_html, Html,
                                         titre_bloc, sous_titre_bloc, legende_natures, expander_perimetre, _esc,
                                         C_TRAIT)
from acier.bibliotheques import PERSO
from ..entrees import (CHAMPS, CLES, champ_visible, TYPES_ELEM, TYPE_I, TYPE_RHS, TYPE_CHS, TYPE_PLAT, TYPE_ROND,
                       TYPE_RECT, DIRS_RAID, PROFILS, element_defaut, raidisseur_defaut)
from ..ecran_saisie import (K, valeur_widget, PAS, AIDES_MANUEL as AIDES, AFFICHAGE, D_MANUEL as D,
                            libelle, en_tete, changer_mode, _kit_bloc, aides_visibles as _aides_kit, avances,
                            cartouche)
from .. import ecran_resultats as ER
from ..notation import LEGENDE_REFERENCES
from .. import formules
from . import schemas, synthese, texte
from . import elements as EL
from .methode import METHODE
from .moteur import GROUPES_VERIF
from .conversion import disposition_reguliere

__all__ = ["changer_mode", "carte", "bandeau", "dessins", "onglets"]

ONGLETS = ("Vérifications", "Efforts des tiges", "Note", "Benchmark", "Méthode")
DIMS_PAR_TYPE = {TYPE_I: ("h", "b", "tw", "tf", "r"), TYPE_RHS: ("h", "b", "t"), TYPE_CHS: ("d", "t"),
                 TYPE_PLAT: ("b", "t"), TYPE_ROND: ("d",), TYPE_RECT: ("b", "h")}
LIB_DIM = {"h": "h", "b": "b", "tw": "tw", "tf": "tf", "r": "r", "t": "t", "d": "Ø"}
# Types d'élément courts à l'affichage (charte §3) ; valeurs brutes TYPE_* inchangées.
AFF_TYPES = {TYPE_I: "Profilé I / H", TYPE_RHS: "Tube RHS / SHS", TYPE_CHS: "Tube CHS", TYPE_PLAT: "Plat sur chant",
             TYPE_ROND: "Rond plein", TYPE_RECT: "Rectangle plein"}
# Les quatre critères de l'épaisseur requise (bandeau, bloc d'état, chips)
CRITERES_NOMS = {"T": "tronçon en T du boulon", "S": "section au nu d'une paroi", "c": "bande comprimée",
                 "b": "pression diamétrale"}
# Les clés que la carte manuelle rend hors des lignes du kit : ses tables et le repli « Poser une grille… »
CLES_TABLES = ("tiges", "elems", "raid", "ui_gx", "ui_gy", "ui_gwx", "ui_gwy")
NON_VERIFIES = ("cordons", "raidisseurs eux-mêmes", "poteau",
                "béton d'ancrage (cône, arrachement, bord, pry-out) — logiciel d'ancrage ou vérification complète")
NATURES = ("EC", "EC + INT", "INT")
TEXTES_NATURES = {"EC + INT": "Eurocode et interprétation nommée",
                  "INT": "modèle d'ingénieur (sections au nu, bandes, angles)"}
CAPTION_LEGENDE = ("Vue en plan : aire d'appui (bleu), zone comprimée (rouge), C, axe neutre, T et bras m de chaque "
                   "boulon, bande comprimée (mEd).")
CAPTION_USAGE = CAPTION_LEGENDE + " Glisse un boulon, un élément ou un raidisseur ; touche une pièce ou une cote."

# BRIQUES DE KIT MANQUANTES (kit gelé, amendement 7), signalées pour la consolidation — CSS du module,
# injecté une fois par interface.py (dans le même élément que le CSS du kit) :
# - le titre d'une table (ÉLÉMENTS, RAIDISSEURS) est porté par la première case de son en-tête ≥ 1351 px ;
#   sous 1350 px l'en-tête est caché par le kit, le titre est alors rendu une seconde fois sur sa ligne
#   (ppb_tt_*), cachée ≥ 1351 px ;
# - les tables de détail (9 et 8 colonnes, conteneurs ppb_tab_*) tiennent dans leur colonne à la tablette
#   (440 px à 1024) et au téléphone (358 px) : police et marges réduites, en-têtes sur plusieurs lignes
#   (mesuré : 470 px avant, débordement au téléphone, dernière colonne rognée à 1024) ;
# - une alerte et son bouton « Localiser » sur une rangée (ppb_al_*) : sans la marge −16 px du conteneur
#   markdown, que la rangée horizontale rognait (bas du cadre coupé, constaté sur la capture) ;
# - la tête de la colonne dessin (ppb_tete_dessin) : 4 px entre ses rangées au téléphone (comme la barre) ;
# - sous 1351 px, une rangée de table range ses cases par deux AVEC le 🗑️ à leur droite (x · y · 🗑️ sur une
#   rangée) : la base de 205 / 160 px du kit donnait au 🗑️ une demi-rangée à lui seul, cadre vide de 205 px
#   (constaté sur les captures à 1024 et 390) ; le titre ppb_tt_ prend la forme d'un titre de bloc du kit
#   au-dessus de sa ligne (« ? » collé au texte, 6 px d'air).
_TR = "[class*=\"st-key-ppb_tr_\"] > [data-testid=\"stElementContainer\"]"
CSS_MODULE = ("<style>@media (min-width:1351px){[class*=\"st-key-ppb_tt_\"]{display:none !important;}}\n"
              "@media (max-width:1350px){" + _TR + ":not([class*=\"st-key-ppb_btn_\"]){flex:1 1 calc(50% - 30px) "
              "!important;max-width:calc(50% - 30px) !important;}\n"
              "  " + _TR + "[class*=\"st-key-ppb_btn_\"]{flex:0 0 auto !important;max-width:none !important;}\n"
              "  [class*=\"st-key-ppb_tt_\"]{margin-top:6px;}\n"
              "  [class*=\"st-key-ppb_tt_\"] [data-testid=\"stMarkdown\"]{width:fit-content !important;}\n"
              "  [class*=\"st-key-ppb_tt_\"] [data-testid=\"stMarkdownContainer\"] > div{padding-bottom:6px !important;"
              "margin-bottom:0 !important;}}\n"
              "@media (max-width:1180px){[class*=\"st-key-ppb_tab_\"] table{font-size:0.76em !important;}\n"
              "  [class*=\"st-key-ppb_tab_\"] th, [class*=\"st-key-ppb_tab_\"] td{padding:2px 4px !important;}\n"
              "  [class*=\"st-key-ppb_tab_\"] th{white-space:normal !important;letter-spacing:0 !important;}}\n"
              "@media (max-width:640px){[class*=\"st-key-ppb_tab_\"] table{font-size:0.7em !important;}\n"
              "  [class*=\"st-key-ppb_tab_\"] th, [class*=\"st-key-ppb_tab_\"] td{padding:2px 3px !important;}}\n"
              "[class*=\"st-key-ppb_al_\"] [data-testid=\"stMarkdownContainer\"]{margin-bottom:0 !important;}\n"
              "[class*=\"st-key-ppb_tete_dessin\"]{gap:4px 0.5rem !important;}</style>")
# Un titre de bloc tient dans la case de 76 px sur une ou deux lignes, jamais coupé au milieu d'un mot
# (charte §2) : « RAIDISSEURS » (11 lettres) ne tient pas sur une ligne, il se coupe à sa césure.
TITRE_RAID = "RAIDIS\u00adSEURS"


def _verrou():
    return bool(st.session_state.get("prj_verrou", False))


# --------------------------------------------------------------- les listes
def _liste(cle, etat=None):
    etat = st.session_state if etat is None else etat
    lst = etat.get(K(cle))
    if not isinstance(lst, list):
        lst = []
        etat[K(cle)] = lst
    return lst


def _ecrire_champ_liste(cle, i, champ, wkey, conv=float):
    """on_change d'un widget de ligne : la valeur va dans la liste source."""
    lst = _liste(cle)
    if i < len(lst):
        v = st.session_state.get(wkey)
        try:
            lst[i][champ] = conv(v) if conv is not str else str(v)
        except (TypeError, ValueError):
            pass
        st.session_state[K(cle)] = lst


def _supprimer(cle, i):
    lst = _liste(cle)
    if i < len(lst):
        del lst[i]
        st.session_state[K(cle)] = lst
        st.session_state["_ppb_toast"] = "Ligne supprimée"


def _ajouter(cle, objet):
    lst = _liste(cle)
    lst.append(objet)
    st.session_state[K(cle)] = lst


def _ajouter_boulon(u):
    lp = N(u.get("lp_u")) or 500.0
    tiges = _liste("tiges")
    # au bord bas, à 60 mm, à droite du dernier boulon posé sur cette ligne
    y = -(lp / 2 - 60.0)
    xs = [float(N(t.get("x")) or 0) for t in tiges if abs(float(N(t.get("y")) or 0) - y) < 1e-6]
    x = (max(xs) + 80.0) if xs else 0.0
    _ajouter("tiges", dict(x=x, y=y))


def _poser_grille():
    nx_, ny_ = int(st.session_state.get("ppb_ui_gx") or 2), int(st.session_state.get("ppb_ui_gy") or 2)
    wx, wy = float(st.session_state.get("ppb_ui_gwx") or 400), float(st.session_state.get("ppb_ui_gwy") or 400)
    from .. import geometrie as GE
    st.session_state[K("tiges")] = [dict(x=float(x), y=float(y)) for (x, y, _i, _j) in GE.positions_tiges(nx_, ny_, wx, wy)]
    st.session_state["_ppb_toast"] = f"Grille {nx_} × {ny_} posée"


# ------------------------------------------------------- widgets des tables
def _num(label, wkey, valeur, cle, i, champ, step=1.0, aide=None, width="stretch"):
    st.session_state[wkey] = float(valeur) if valeur is not None else 0.0
    st.number_input(label, key=wkey, step=step, format="%g", on_change=_ecrire_champ_liste, args=(cle, i, champ, wkey),
                    disabled=_verrou(), help=aide, width=width)


def _sel(label, wkey, valeur, options, cle, i, champ, width="stretch", affichage=None):
    st.session_state[wkey] = valeur if valeur in options else options[0]
    st.selectbox(label, options, key=wkey, on_change=_ecrire_champ_liste, args=(cle, i, champ, wkey, str),
                 disabled=_verrou(), width=width,
                 format_func=(lambda o, a=affichage: a.get(o, o)) if affichage else str)


def _poubelle(key, cle, i, aide):
    # « btn_ » dans la clé : le CSS du kit ne compte pas le 🗑️ comme une case
    st.button("🗑️", key=key, on_click=_supprimer, args=(cle, i), disabled=_verrou(), help=aide, width="content")


# --------------------------------------------- les tables (briques locales)
def _entete_table(bloc, entetes, titre=None, aide=None, chaud=False):
    """L'en-tête ``ppb_th_<bloc>`` d'une table (visible ≥ 1351 px) : la case
    des titres porte « n° » ou le TITRE du bloc avec son « ? » ; puis les
    libellés des colonnes (``entetes`` = [(libellé, aide)]) et la place du
    🗑️. Mêmes largeurs que les rangées (kit.table). Sous 1350 px, le titre est
    rendu une seconde fois sur sa propre ligne ``ppb_tt_<bloc>``."""
    def q(h):
        # l'apostrophe échappée : « l'élément » de l'aide RAIDISSEURS fermait l'attribut title
        return (f" <span title='{_esc(h).replace(chr(39), '&#39;')}' style='display:inline-block;border:1px solid #6E7480;border-radius:50%;"
                f"width:13px;height:13px;line-height:12px;text-align:center;font-size:9px;cursor:help;"
                f"font-weight:400;letter-spacing:0;'>?</span>" if h else "")

    def cellule(lab, h):
        return f"<div style='flex:1 1 100px;min-width:0;white-space:nowrap;'>{_esc(lab)}{q(h)}</div>"
    t = ("🔴 " if chaud else "") + titre if titre else None
    premiere = (f"<div style='flex:0 0 {kit.LARG_TITRE}px;font-size:0.92em;font-weight:700;letter-spacing:.07em;"
                f"color:#33415C;line-height:1.15;'>{_esc(t)}{q(aide)}</div>" if t
                else f"<div style='flex:0 0 {kit.LARG_TITRE}px;'>n°</div>")
    with st.container(key=f"ppb_th_{bloc}"):
        # margin-bottom:10px : le conteneur markdown de Streamlit porte une marge de −16 px (cf.
        # ui_commun.titre_bloc) ; l'en-tête du kit.table, sans compensation, mesurait 0 px de haut et les
        # rangées le recouvraient (mesuré le 31/10/2026 : « x (mm) » invisible).
        st.markdown(f"<div style='display:flex;gap:8px;align-items:flex-end;font-size:0.78em;color:#6E7480;"
                    f"padding:0 0 4px 0;margin-bottom:12px;'>" + premiere + "".join(cellule(lab, h) for lab, h in entetes)
                    + "<div style='flex:0 0 43px;'></div></div>", unsafe_allow_html=True)
    if t:
        with st.container(key=f"ppb_tt_{bloc}"):
            titre_bloc(t, help=aide)


def _rangee(bloc, i, sous_titre, rendre):
    """Une rangée de table ``ppb_tr_<bloc>_<i>`` : sous-titre (n° 1, É1, R1)
    dans la case des titres, widgets à libellés visibles (cachés ≥ 1351 px)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=f"ppb_tr_{bloc}_{i}"):
        with st.container(key=f"ppb_bt_{bloc}_{i}", width=kit.LARG_TITRE):
            sous_titre_bloc(sous_titre)
        rendre()


def _suite(ident, sous_titre, rendre):
    """Une continuation « ↳ » ``ppb_ls_<ident>`` : cases de 150 px ou boutons.
    ``ident`` ne doit pas être celui d'une table (« elems_0 » donnerait la clé
    ppb_bt_elems_0 de la rangée É1 : clé dupliquée, trouvée par le harnais)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=f"ppb_ls_{ident}"):
        with st.container(key=f"ppb_bt_{ident}", width=kit.LARG_TITRE):
            sous_titre_bloc("↳ " + sous_titre if sous_titre else "↳")
        rendre()


def _ligne_titre(ident, titre, aide, rendre, chaud=False):
    """Une ligne d'objet titrée dont la seule case est une action (table vide)."""
    with st.container(horizontal=True, wrap=True, gap="small", vertical_alignment="bottom", key=f"ppb_lg_{ident}_0"):
        with st.container(key=f"ppb_bt_{ident}_0", width=kit.LARG_TITRE):
            titre_bloc(("🔴 " if chaud else "") + titre, help=aide)
        rendre()


# ---------------------------------------------------------------- les blocs
def bloc_boulons(u, chauds):
    """BOULONS : Boulon · Classe · Ancrage ? (ligne du kit), puis la table
    n° | x (mm) ? | y (mm) | 🗑️ et les actions ➕ Boulon · Poser une grille…"""
    tiges = _liste("tiges")
    _kit_bloc(D, "BOULONS", D.BLOCS[1][1], u, chauds, chaud="anchors" in chauds)
    if tiges:
        _entete_table("tiges", [("x (mm)", AIDES["tiges"]), ("y (mm)", None)])
        for i, t in enumerate(tiges):
            def rendre(i=i, t=t):
                _num("x (mm)", f"ppb_t{i}_x", N(t.get("x")), "tiges", i, "x", 5.0, aide=AIDES["tiges"])
                _num("y (mm)", f"ppb_t{i}_y", N(t.get("y")), "tiges", i, "y", 5.0)
                _poubelle(f"ppb_btn_tdel_{i}", "tiges", i, "Retirer ce boulon")
            _rangee("tiges", i, f"n° {i + 1}", rendre)

    def actions():
        st.button("➕ Boulon", key="ppb_btn_tadd", on_click=_ajouter_boulon, args=(u,), disabled=_verrou(),
                  width="content", help="Un boulon de plus (puis le déplacer sur le dessin ou taper x, y)")
        with st.popover("Poser une grille…", width="content", disabled=_verrou()):
            for k, v in (("ppb_ui_gx", 2), ("ppb_ui_gy", 2), ("ppb_ui_gwx", 400.0), ("ppb_ui_gwy", 400.0)):
                st.session_state.setdefault(k, v)
            g1, g2 = st.columns(2)
            with g1:
                st.number_input("Files (x)", key="ppb_ui_gx", min_value=1, step=1)
                st.number_input("wx (mm)", key="ppb_ui_gwx", step=10.0, format="%g")
            with g2:
                st.number_input("Rangées (y)", key="ppb_ui_gy", min_value=1, step=1)
                st.number_input("wy (mm)", key="ppb_ui_gwy", step=10.0, format="%g")
            st.button("Poser la grille (remplace les boulons)", key="ppb_btn_grille", on_click=_poser_grille,
                      type="primary", width="stretch")
    _suite("tiges_act", "", actions)


def _suite_element_visible(e, t):
    """« ↳ É{i} » (rotation, dimensions) : profilé Personnalisé, type sans
    catalogue, rotation ≠ 0 ou dessin fixe — sinon la fenêtre de l'élément
    sur le dessin porte rotation et dimensions."""
    return (t != TYPE_I or (e.get("prof") or "HEA 300") == PERSO or int(N(e.get("rot")) or 0) != 0
            or not st.session_state.get("ppb_ui_composant", True))


def bloc_elements(u, chauds):
    """ÉLÉMENTS ? : É{i} | Type | Profilé | x | y | 🗑️ (+ ↳ É{i} si besoin),
    puis ➕ Élément soudé."""
    elems = _liste("elems")
    chaud = "col" in chauds
    ajout = dict(label="➕ Élément soudé", key="ppb_btn_eadd", on_click=_ajouter, args=("elems", element_defaut()),
                 disabled=_verrou(), width="content", help="Profilé en I, tube, plat, rond ou rectangle de plus, à l'axe")
    if not elems:
        _ligne_titre("elements", "ÉLÉMENTS", AIDES["ÉLÉMENTS"], lambda: st.button(**ajout), chaud)
        return
    _entete_table("elems", [("Type", None), ("Profilé", None), ("x (mm)", None), ("y (mm)", None)],
                  "ÉLÉMENTS", AIDES["ÉLÉMENTS"], chaud)
    for i, e in enumerate(elems):
        e = dict(element_defaut(), **{k: v for k, v in e.items() if v is not None})
        t = e.get("type") if e.get("type") in TYPES_ELEM else TYPE_I

        def rendre(i=i, e=e, t=t):
            _sel("Type", f"ppb_e{i}_type", t, TYPES_ELEM, "elems", i, "type", affichage=AFF_TYPES)
            if t == TYPE_I:
                _sel("Profilé", f"ppb_e{i}_prof", e.get("prof") or "HEA 300", PROFILS, "elems", i, "prof")
            else:
                # sans catalogue : le nom de la section (« RHS 200×200×8 »), ses dimensions dans ↳ É{i}
                st.session_state[f"ppb_e{i}_nom"] = EL.nom(EL.normaliser(e))
                st.text_input("Profilé", key=f"ppb_e{i}_nom", disabled=True, width="stretch")
            _num("x (mm)", f"ppb_e{i}_x", N(e.get("x")), "elems", i, "x", 5.0)
            _num("y (mm)", f"ppb_e{i}_y", N(e.get("y")), "elems", i, "y", 5.0)
            _poubelle(f"ppb_btn_edel_{i}", "elems", i, "Retirer cet élément")
        _rangee("elems", i, f"É{i + 1}", rendre)
        if _suite_element_visible(e, t):
            perso = t != TYPE_I or (e.get("prof") or "HEA 300") == PERSO

            def suite(i=i, e=e, t=t, perso=perso):
                _sel("Rotation (°)", f"ppb_e{i}_rot", str(int(N(e.get("rot")) or 0)), ["0", "90"], "elems", i, "rot",
                     width=kit.LARG_SUITE)
                for d in (DIMS_PAR_TYPE[t] if perso else ()):
                    _num(LIB_DIM[d] + " (mm)", f"ppb_e{i}_{d}", N(e.get(d)), "elems", i, d, 1.0, width=kit.LARG_SUITE)
            _suite(f"elemsx_{i}", f"É{i + 1}", suite)
    _suite("elems_act", "", lambda: st.button(**ajout))


def bloc_raidisseurs(u, chauds):
    """RAIDISSEURS ? : R{i} | x | y | Sens | 🗑️ + ↳ R{i} | L | t | h (en-tête
    seulement s'il y en a), puis ➕ Raidisseur."""
    raid = _liste("raid")
    chaud = "raid" in chauds
    ajout = dict(label="➕ Raidisseur", key="ppb_btn_radd", on_click=_ajouter, args=("raid", raidisseur_defaut()),
                 disabled=_verrou(), width="content", help=AIDES["RAIDISSEURS"])
    if not raid:
        _ligne_titre("raidisseurs", TITRE_RAID, None, lambda: st.button(**ajout), chaud)
        return
    _entete_table("raid", [("x (mm)", None), ("y (mm)", None), ("Sens", None)], TITRE_RAID, AIDES["RAIDISSEURS"],
                  chaud)
    for i, r in enumerate(raid):
        r = dict(raidisseur_defaut(), **{k: v for k, v in r.items() if v is not None})

        def rendre(i=i, r=r):
            _num("x (mm)", f"ppb_r{i}_x", N(r.get("x")), "raid", i, "x", 5.0)
            _num("y (mm)", f"ppb_r{i}_y", N(r.get("y")), "raid", i, "y", 5.0)
            _sel("Sens", f"ppb_r{i}_dir", str(r.get("dir") or "+x").replace("-", "−"), DIRS_RAID, "raid", i, "dir")
            _poubelle(f"ppb_btn_rdel_{i}", "raid", i, "Retirer ce raidisseur")
        _rangee("raid", i, f"R{i + 1}", rendre)

        def suite(i=i, r=r):
            _num("L (mm)", f"ppb_r{i}_L", N(r.get("L")), "raid", i, "L", 5.0, width=kit.LARG_SUITE)
            _num("t (mm)", f"ppb_r{i}_t", N(r.get("t")), "raid", i, "t", 1.0, width=kit.LARG_SUITE)
            _num("h (mm)", f"ppb_r{i}_h", N(r.get("h")), "raid", i, "h", 10.0, width=kit.LARG_SUITE)
        _suite(f"raidx_{i}", f"R{i + 1}", suite)
    _suite("raid_act", "", lambda: st.button(**ajout))


def etat_bascule(u):
    """La caption d'état sous l'en-tête : la vérification complète est-elle
    disponible pour cette disposition ? (les deux états restent dits,
    amendement 4)."""
    reg = disposition_reguliere(u)
    return ("Vérification complète : disponible pour cette disposition." if reg["ok"]
            else "Vérification complète : indisponible — " + reg["raison"] + ".")


def aides_visibles(u):
    """Les « ? » visibles dans la carte manuelle au cas ``u`` : en-tête et
    lignes du kit, puis ceux des tables — le x (mm) des boulons, ÉLÉMENTS,
    RAIDISSEURS (s'il y en a)."""
    out = _aides_kit(u, D)
    if u.get("tiges"):
        out.append("tiges")
    out.append("ÉLÉMENTS")
    if u.get("raid"):
        out.append("RAIDISSEURS")
    return out


def carte(u, chauds, lire, ecrire):
    """La carte complète du mode manuel (charte §4.5)."""
    with st.container(gap=None):
        en_tete(u, chauds, lire, ecrire)
        st.caption(etat_bascule(u))
        _kit_bloc(D, "MATÉRIAUX", D.BLOCS[0][1], u, chauds)
        bloc_boulons(u, chauds)
        bloc_elements(u, chauds)
        bloc_raidisseurs(u, chauds)
        _kit_bloc(D, "EFFORTS ELU", D.BLOCS[2][1], u, chauds)
    avances(u, chauds)
    cartouche(u, chauds)


# ------------------------------------------------------------------ bandeau
def complement_bandeau(R):
    """« — épaisseur requise 19,8 ≤ tp 25 mm · 79,3 % · tronçon en T du boulon » :
    l'épaisseur requise et son critère gouvernant, sans répéter le libellé
    de la vérification (charte §9)."""
    P = R.plaque
    tp = N(R.u.tp_u) or 0.0
    inf = P["t_req"] == float("inf")
    req = "∞" if inf else F(P["t_req"], 1)
    tient = (not inf) and P["t_req"] <= tp + 1e-9
    comp = ("— épaisseur requise " + req + (" ≤ " if tient else " > ") + "tp " + F(tp, 0) + " mm · "
            + pct(R.eta_max, 1) + (" · " + CRITERES_NOMS[P["pire"]] if R.ck["tpq"].active else ""))
    if not R.geo_ok:
        comp += " · géométrie non valide"
    if not R.dist_ok:
        comp += " · pinces non conformes"
    return comp


def chips_bandeau(R):
    """Les taux par élément (synthese.ELEMENTS), et à côté de « Platine » les
    quatre critères tT · tS · tc · tb (épaisseur demandée / tp)."""
    base = [(nom, eta, okg) for nom, eta, okg, _ in synthese.taux_par_element(R)]
    P, tp = R.plaque, N(R.u.tp_u) or 0.0
    if not (R.ck["tpq"].active and tp > 0):
        return base
    quatre = [("t" + k, P["t_" + k] / tp, P["t_" + k] <= tp + 1e-9) for k in ("T", "S", "c", "b")]
    out = []
    for c in base:
        out.append(c)
        if c[0] == "Platine":
            out.extend(quatre)
    return out


def bandeau(R):
    ok = R.verified
    titre = R.statut + (" – sous réserve des alertes" if ok and (R.reserve or R.alerts) else "")
    bloc_statut_ligne("ok" if ok else "nok", titre, complement_bandeau(R), chips_bandeau(R))


def information(R):
    """La ligne d'information sous le bandeau : la vérification d'information
    la plus sollicitée, hors verdict."""
    pire = synthese.pire_information(R)
    if pire is not None:
        st.caption("En information, hors verdict : " + synthese.court(pire) + " à " + pct(pire.eta, 1)
                   + (" — non satisfait" if not pire.ok else " — satisfait")
                   + ". Le verdict porte sur la platine ; les tiges et le béton relèvent du logiciel d'ancrage ou de "
                   "la vérification complète.")


# ------------------------------------------------------------------ dessins
def _valeurs(u, R):
    """Les valeurs des clés d'entrée ET des clés dérivées des pièces déplaçables."""
    v = {k: st.session_state.get(K(k)) for k in CLES}
    bp, lp = N(u.get("bp_u")) or 0.0, N(u.get("lp_u")) or 0.0
    for t in R.tiges:
        v[f"t{t['i']}_u"], v[f"t{t['i']}_v"] = round(t["x"] + bp / 2), round(lp / 2 - t["y"])
        v[f"t{t['i']}_x"], v[f"t{t['i']}_y"] = t["x"], t["y"]
    for i, e in enumerate(R.elems):
        v[f"e{i}_u"], v[f"e{i}_v"] = round(e["x"] + bp / 2), round(lp / 2 - e["y"])
        v[f"e{i}_x"], v[f"e{i}_y"], v[f"e{i}_rot"], v[f"e{i}_type"], v[f"e{i}_prof"] = e["x"], e["y"], str(e["rot"]), e["type"], e["prof"]
        for d in ("h", "b", "tw", "tf", "r", "t", "d"):
            v[f"e{i}_{d}"] = e[d]
    for i, r in enumerate(R.raid):
        v[f"r{i}_u"], v[f"r{i}_v"] = round(r["x"] + bp / 2), round(lp / 2 - r["y"])
        v[f"r{i}_x"], v[f"r{i}_y"], v[f"r{i}_dir"], v[f"r{i}_L"], v[f"r{i}_t"], v[f"r{i}_h"] = r["x"], r["y"], r["dir"], r["L"], r["t"], r["h"]
    return v


def _appliquer_derivee(k, v, u, etat=None):
    """Une clé dérivée (« t3_u », « e0_x », « r1_dir »…) → la liste source
    (``etat`` : l'état de session, celui de Streamlit par défaut). True si écrit."""
    import re
    etat = st.session_state if etat is None else etat
    m = re.match(r"^([ter])(\d+)_([A-Za-z]+)$", k)
    if not m:
        return False
    cle = {"t": "tiges", "e": "elems", "r": "raid"}[m.group(1)]
    i, champ = int(m.group(2)), m.group(3)
    lst = _liste(cle, etat)
    if i >= len(lst):
        return False
    bp, lp = N(u.get("bp_u")) or 0.0, N(u.get("lp_u")) or 0.0
    if champ == "u":
        lst[i]["x"] = float(v) - bp / 2
    elif champ == "v":
        lst[i]["y"] = lp / 2 - float(v)
    elif champ in ("type", "prof", "dir"):
        lst[i][champ] = str(v)
    elif champ == "rot":
        lst[i]["rot"] = int(float(v))
    else:
        lst[i][champ] = float(v)
    etat[K(cle)] = lst
    return True


def _traiter_clic(retour, cle_cmp, u):
    if not retour or not isinstance(retour, dict):
        return
    derniers = st.session_state.setdefault("ppb_ui_dernier_clic", {})
    if derniers.get(cle_cmp) == retour.get("t"):
        return
    derniers[cle_cmp] = retour.get("t")
    changes = retour.get("changes") or {}
    ecrits = []
    for k, v in changes.items():
        if k in CHAMPS:
            st.session_state[K(k)] = valeur_widget(k, v)
            ecrits.append(libelle(k, u) + " = " + js_str(valeur_widget(k, v)))
        elif _appliquer_derivee(k, v, u):
            ecrits.append(k.split("_")[0] + " déplacé" if k.endswith(("_u", "_v")) else k + " = " + js_str(v))
    if ecrits:
        st.toast(" ; ".join(sorted(set(ecrits))) + " : recalculé")
        st.rerun()


def _panneaux(R, u):
    """Les fenêtres d'objet du composant — une VUE des mêmes clés (libellés
    LIBELLE / LIBELLE_MODE, aides AIDES, options AFFICHAGE), titres courts,
    aucun doublon (a_w ne reste que dans Soudure, αcc dans les avancés). La
    fenêtre Platine, toujours cliquable, porte aussi hef et dh : ils restent
    atteignables même sans boulon (la coupe n'émet leurs cotes qu'avec un
    boulon) ; la fenêtre Massif porte tg quand il est nul (le mortier n'est
    alors pas dessiné : amendement 2)."""
    def ch(k):
        f = CHAMPS[k]
        return dict(key=k, lab=libelle(k, u), type=f["t"], options=f.get("o") or [], step=PAS.get(k, 1.0), dis=False,
                    help=AIDES.get(k), affichage=AFFICHAGE.get(k) or None)

    def nb(k, lab, step=5.0):
        return dict(key=k, lab=lab, type="n", options=[], step=step, dis=False)

    def se(k, lab, options, affichage=None):
        return dict(key=k, lab=lab, type="s", options=list(options), step=1, dis=False, affichage=affichage)
    ancrage = [ch("hef")] + ([ch("d_h")] if champ_visible(u, "d_h") else [])
    tg_nul = (N(u.get("t_g")) or 0.0) <= 0
    g = {}
    g["plate"] = dict(titre="Platine", champs=[ch("bp_u"), ch("lp_u"), ch("tp_u"), ch("nu_pl")] + ancrage,
                      note="bp, lp, tp, hef et dh : aussi des cotes sur le dessin ; la gorge a : fenêtre Soudure.")
    g["pedestal"] = dict(titre="Massif", champs=[ch("beton"), ch("bc_u"), ch("lc_u"), ch("hc_u")]
                         + ([ch("t_g")] if tg_nul else []))
    g["grout"] = dict(titre="Mortier", champs=[ch("t_g"), ch("f_g")])
    g["weld"] = dict(titre="Soudure", champs=[ch("a_w")],
                     note="La gorge ne sert qu'au bras de levier (0,8·a·√2 déduit) : les cordons ne sont pas vérifiés ici.")
    g["anchors"] = dict(titre="Boulons", champs=[ch("ancr"), ch("classe"), ch("typ_a")] + ancrage)
    g["efforts"] = dict(titre="Efforts",
                        champs=[ch(k) for k in ("N_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed", "T_Ed", "xN", "yN")],
                        note="Mx comprime le côté y > 0 (haut de la vue en plan) ; My comprime x > 0 ; efforts en (xN ; yN).")
    for t in R.tiges:
        g[f"bolt{t['i']}"] = dict(titre=f"Boulon n° {t['i'] + 1}", champs=[nb(f"t{t['i']}_x", "x (mm)"), nb(f"t{t['i']}_y", "y (mm)")],
                                  note="Se déplace aussi à la souris. Pour le retirer : 🗑️ dans la carte.")
    for i, e in enumerate(R.elems):
        champs = [se(f"e{i}_type", "Type", TYPES_ELEM, AFF_TYPES), nb(f"e{i}_x", "x (mm)"), nb(f"e{i}_y", "y (mm)"),
                  se(f"e{i}_rot", "Rotation (°)", ["0", "90"])]
        if e["type"] == TYPE_I:
            champs.insert(1, se(f"e{i}_prof", "Profilé", PROFILS))
        for d in DIMS_PAR_TYPE[e["type"]]:
            if e["type"] == TYPE_I and e["prof"] != PERSO:
                break
            champs.append(nb(f"e{i}_{d}", LIB_DIM[d] + " (mm)", 1.0))
        g[f"elem{i}"] = dict(titre=f"Élément {i + 1} — {EL.nom(e)}", champs=champs, note="Se déplace aussi à la souris.")
    for i, r in enumerate(R.raid):
        g[f"raid{i}"] = dict(titre=f"Raidisseur {i + 1}", champs=[nb(f"r{i}_x", "x (mm)"), nb(f"r{i}_y", "y (mm)"),
                                                                 se(f"r{i}_dir", "Sens", DIRS_RAID), nb(f"r{i}_L", "L (mm)"),
                                                                 nb(f"r{i}_t", "t (mm)", 1.0), nb(f"r{i}_h", "h (mm)", 10.0)])
    return g


def dessins(R, u):
    """La colonne du dessin : en tête le niveau de cotation et la case
    « Dessin interactif », puis coupe A-A et vue en plan. Renvoie ``(e, p, hl)``."""
    niveau = ER.tete_dessin()
    interactif = bool(st.session_state.get("ppb_ui_composant", True))
    hl = ER._alerte_courante(R)
    opt = schemas.options_ecran(R, niveau, hl)
    opt.editables = niveau >= 1
    e, p = schemas.coupe_aa(R, opt), schemas.plan(R, opt)
    valeurs = _valeurs(u, R)
    groupes = _panneaux(R, u)
    for d, cle in ((e, "ppb_cmp_elev"), (p, "ppb_cmp_plan")):
        if interactif:
            from acier.composants.svg_cliquable import svg_cliquable
            _traiter_clic(svg_cliquable(d.svg(identifiant=cle), valeurs, key=cle, groupes=groupes), cle, u)
        else:
            st.markdown(d.svg(identifiant=cle), unsafe_allow_html=True)
    st.caption(CAPTION_USAGE if interactif else CAPTION_LEGENDE)
    return e, p, hl


# ------------------------------------------------- vérifications à la main
def _lig(lab, val, fort=False):
    return [Html(f"<b>{_esc(lab)}</b>" if fort else _esc(lab)), Html(f"<b>{_esc(val)}</b>" if fort else _esc(val))]


def bloc_etat(R):
    """En tête de l'onglet : l'épaisseur requise et ses quatre critères (bloc d'état)."""
    P = R.plaque
    u = R.u
    c = R.ck["tpq"]
    st.markdown("##### Épaisseur de la platine — le calcul à la main")
    if not c.active:
        st.caption("Géométrie non valide ou équilibre introuvable : pas d'épaisseur requise.")
        return False
    req = "∞ (aucune épaisseur ne suffit)" if c.Ed == float("inf") else F(c.Ed, 1) + " mm"
    bloc_statut_ligne("ok" if c.ok else "nok",
                      "Épaisseur requise " + req + (" ≤ " if c.ok else " > ") + "tp = " + F(u.tp_u, 0) + " mm",
                      "— " + pct(c.eta, 1) + " · critère : " + CRITERES_NOMS[P["pire"]] + " · " + P["critere"]
                      + " · fy = " + F(R.fy_p, 0) + " MPa",
                      [(CRITERES_NOMS[k], P["t_" + k] / u.tp_u if u.tp_u else 0.0, P["t_" + k] <= u.tp_u + 1e-9)
                       for k in ("T", "S", "c", "b")])
    return True


def detail_main(R):
    """Le détail, comme à la main : boulon par boulon, face par face, la
    bande comprimée, les trous — les quatre tables de détail."""
    P = R.plaque

    def mm(v):
        return "∞" if v == float("inf") else F(v, 1)
    tendus = [t for t in R.tiges if t["T"] > 1e-9]
    # Les tables l'une SOUS l'autre, sur toute la largeur : « boulon par boulon » (9 colonnes) et « section au
    # nu » (8) débordaient d'une demi-colonne (PC et tablette, constaté le 31/10/2026) ; elles défilent au
    # téléphone (tableau_html(defilant=True)).
    st.markdown("**Côté tendu — boulon par boulon (tronçon en T)**")
    if tendus:
        lignes = []
        for t in tendus:
            lignes.append([js_str(t["i"] + 1), F(t["T"], 1), t["appui"]["face"].replace("élément ", "É"),
                           F(t["m"], 1) + (" (angle)" if t["coin"] else ""), F(t["M"], 2), F(t["leff1"], 0),
                           F(t["FT_Rd"], 0) + " (m" + str(t["mode_ts"]).split(" ")[0] + ")",
                           taux_html(t["eta"], t["eta"] <= 1 + 1e-9), F(t["t_pl"], 1) + " / " + F(t["t_el"], 1)])
        # En-têtes sur deux lignes : la table tient dans la colonne sur tablette (660 px) sans défiler.
        st.markdown(tableau_html(["n°", Html("T<br>(kN)"), "Appui", Html("m<br>(mm)"), Html("T·m<br>(kNm)"),
                                  Html("leff<br>(mm)"), Html("FT,Rd<br>(kN)"), "η", Html("tp plast.<br>/ élast.")],
                                 lignes, droite=(1, 3, 4, 5, 6, 7, 8), defilant=True), unsafe_allow_html=True)
        pt = R.pire_tige
        st.caption(f"Boulon {pt['i'] + 1} : Lb = {F(R.Lb, 0)} / Lb* = {F(min(pt['Lb_star'], 9999), 0)} mm → "
                   + ("avec" if pt["levier"] else "sans") + " effet de levier, mode " + str(pt["mode_ts"])
                   + f" ; Mpl,1 = 0,25·leff·tp²·fy/γM0 = {F(pt['Mpl1'], 2)} kNm ; σ Von Mises de la bande (2·leff) = {F(pt['svm'], 0)} MPa.")
    else:
        st.caption("Aucun boulon tendu : la platine ne fléchit pas en traction.")
    st.markdown("**Côté tendu — section au nu de chaque paroi**")
    if R.sections:
        lignes = []
        for s in R.sections:
            lignes.append([s["nom"].replace("élément ", "É").replace("raidisseur ", "R"),
                           ", ".join(str(i + 1) for i in s["idx"]), ", ".join(F(m, 0) for m in s["m"]),
                           F(s["M"], 2), F(s["b"], 0), F(s["Mpl"], 2), taux_html(s["eta"], s["eta"] <= 1 + 1e-9),
                           F(s["t_pl"], 1) + " / " + F(s["t_el"], 1)])
        st.markdown(tableau_html(["Face", "Boulons", Html("m⊥<br>(mm)"), Html("ΣT·m<br>(kNm)"), Html("b<br>(mm)"),
                                  Html("Mpl<br>(kNm)"), "η", Html("tp plast.<br>/ élast.")],
                                 lignes, droite=(2, 3, 4, 5, 6, 7), defilant=True), unsafe_allow_html=True)
        st.caption("b = min(platine ; L + 2·m) : la face de longueur L et la diffusion à 45° sur le bras.")
    else:
        st.caption("Aucune section : pas de boulon tendu devant une paroi.")
    st.markdown("**Côté comprimé — la bande la plus sollicitée**")
    cs = R.console
    rows = [_lig("Pression sous la platine (répartition rigide)", "p max = " + F(R.pmax, 2) + " MPa ≤ fjd = " + F(R.fjd, 1) + " MPa"),
            _lig("Aire d'appui", ("toute la platine" if R.plein else "bande c = tp·√(fy/(3·fjd·γM0)) = " + F(R.c, 0) + " mm autour des parois")),
            _lig("Bande", cs["nom"].replace("élément ", "É") + " : p au nu = " + F(cs["p0"], 2) + " MPa, console de " + F(cs["L"], 0) + " mm"),
            _lig("Moment pris", "mEd = ∫p·t dt = " + F(cs["m_Ed"], 2) + " kNm/m", True),
            _lig("Résistance élastique", "mRd = tp²·fy/(6·γM0) = " + F(cs["m_Rd"], 2) + " kNm/m → " + (pct(R.ck["pc"].eta, 0) if R.ck["pc"].active else "—")),
            _lig("Épaisseur pour ce moment", "tc = √(6·mEd·γM0/fy) = " + F(cs["t"], 1) + " mm (à tp donné) ; balayage : " + mm(P["t_c"]) + " mm")]
    st.markdown(tableau_html(["", ""], rows, largeurs=["38%", None]), unsafe_allow_html=True)
    st.markdown("**Trous — pression diamétrale (Tab. 3.4)**")
    pb = R.pire_appui
    if pb and pb["Fv"] > 1e-9:
        rows = [_lig("Boulon le plus cisaillé", f"n° {pb['i'] + 1} : Fv = {F(pb['Fv'], 1)} kN (V partagé, torsion, frottement déduit)"),
                _lig("Pinces et entraxes dans le sens de l'effort", f"e1 = {F(pb['e1'], 0)}, p1 = {('∞' if pb['p1'] == float('inf') or pb['p1'] > 1e8 else F(pb['p1'], 0))}, e2 = {F(pb['e2'], 0)}, "
                     f"p2 = {('∞' if pb['p2'] == float('inf') or pb['p2'] > 1e8 else F(pb['p2'], 0))} mm"),
                _lig("Fb,Rd = k1·αb·fu·d·tp/γM2", f"αb = {F(pb['ab'], 2)}, k1 = {F(pb['k1'], 2)} → {F(pb['Fb'], 1)} kN → tb = {F(pb['t_b'], 1)} mm", True)]
    else:
        rows = [_lig("Cisaillement par boulon", "nul (frottement Cf,d·N = " + F(R.Ff_Rd / 1e3, 0) + " kN ≥ V) : pas de pression diamétrale")]
    st.markdown(tableau_html(["", ""], rows, largeurs=["38%", None]), unsafe_allow_html=True)
    st.caption("Chaque épaisseur « balayage » est la plus petite qui satisfait son critère, la répartition étant "
               "recalculée à chaque épaisseur essayée ; « à tp donné » : l'épaisseur que demande le moment calculé "
               "avec l'épaisseur retenue.")


def _parametres_retenus(R):
    u = R.u
    g = [["Mode", "Calcul manuel — critère " + R.plaque["critere"] + " ; appui : " + ("toute la platine" if R.plein else "bande c")],
         ["Platine", js_str(N(u.bp_u)) + " × " + js_str(N(u.lp_u)) + " × " + js_str(N(u.tp_u)) + " mm " + str(u.nu_pl)
          + " – fy = " + F(R.fy_p, 0) + " MPa (épaisseur), fu = " + F(R.fu_p, 0)],
         ["Éléments soudés", " + ".join(f"{EL.nom(e)} en ({F(e['x'], 0)} ; {F(e['y'], 0)})" + (" tourné 90°" if e["rot"] else "") for e in R.elems) or "—"],
         ["Raidisseurs", ", ".join(f"({F(r['x'], 0)} ; {F(r['y'], 0)}) {r['dir']} L {F(r['L'], 0)} t {F(r['t'], 0)}" for r in R.raid) or "—"],
         ["Boulons", js_str(R.n_t) + " × " + str(u.ancr) + " " + str(u.classe) + " – d0 = " + F(R.d0, 0) + " mm, As = " + F(R.As, 0)
          + " mm² – " + ", ".join(f"({F(t['x'], 0)} ; {F(t['y'], 0)})" for t in R.tiges)],
         ["Béton", "fck = " + F(R.fck, 0) + " MPa, fcd = " + F(R.fcd, 1) + " MPa, fjd = βj·fcd·√(Ac1/Ac0) = " + F(R.fjd, 1) + " MPa"],
         ["Bras : cordon déduit", "0,8·a·√2 = " + F(R.cord, 1) + " mm"],
         ["Raideurs de la répartition", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm (" + ("avec" if R.levier else "sans") + " effet de levier)"],
         ["Efforts au point (" + F(R.xN, 0) + " ; " + F(R.yN, 0) + ")", "Mx = " + F(R.Mx_o, 1) + " kNm, My = " + F(R.My_o, 1) + " kNm à l'axe de la platine"],
         ["Longueur d'allongement Lb = 8d + tg + tp + 0,6d", F(R.Lb, 0) + " mm"],
         ["Résistance d'un boulon (acier)", "Ft,Rd = " + F(R.Ft_EC3 / 1e3, 1) + " kN ; NRd,s = " + F(R.N_Rd_s / 1e3, 1) + " kN ; F2,vb,Rd = " + F(R.F2vb / 1e3, 1) + " kN"]]
    with st.expander("Paramètres retenus"):
        st.markdown(tableau_md(["Paramètre", "Valeur"], g))


def _formules(R):
    with st.expander("Formules et substitutions numériques"):
        for grp in GROUPES_VERIF:
            on = [c for c in R.checks if c.grp == grp and c.active]
            if not on:
                continue
            st.markdown(f"**{grp}**")
            for c in on:
                corps = "<br>".join(_esc(l) for l in formules.textes(R, c))
                st.markdown(
                    f"<div style='margin:2px 0 8px 0;padding:4px 10px;border-left:3px solid "
                    f"{C_TRAIT['ok'] if c.ok else C_TRAIT['nok']};font-size:0.88em;'>"
                    f"<b>{_esc(synthese.court(c))}</b>{badge_nature(c.nat)} "
                    f"<span style='color:{C_TRAIT['mut']};'>{_esc(c.ref)}</span><br>"
                    f"<span style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.95em;'>{corps}</span></div>",
                    unsafe_allow_html=True)


def _table_simple(R, cle):
    ls = synthese.lignes(R, cle)
    if not ls:
        return False
    st.markdown(f"##### {synthese.titre_table(cle, R)}")
    lignes = [[Html(f"{l['lab']}{badge_nature(l['nat'])}"), l["Ed"], l["Rd"] + (" " + l["unit"] if l["unit"] else ""),
               taux_html(l["eta"], l["ok"], l["c"] is R.gov), l["ref"]] for l in ls]
    st.markdown(tableau_html(["Vérification", "Ed", "Rd", "η", "Réf."], lignes, droite=(1, 2, 3),
                             largeurs=["34%", "12%", "14%", "9%", None]), unsafe_allow_html=True)
    return True


def onglet_verifications(R):
    """Le bloc d'état et la table des quatre critères en tête ; le détail
    « comme à la main » dans un repli OUVERT ; les tables d'information, la
    légende, le périmètre, les formules, le Tableau 3.3 replié."""
    if bloc_etat(R):
        _table_simple(R, "platine")
        with st.expander("Le détail, comme à la main", expanded=True):
            with st.container(key="ppb_tab_detail"):
                detail_main(R)
    else:
        _table_simple(R, "platine")
    for cle in synthese.ORDRE:
        if cle != "platine":
            _table_simple(R, cle)
    legende_natures(NATURES, LEGENDE_REFERENCES, textes=TEXTES_NATURES)
    expander_perimetre(NON_VERIFIES, [synthese.court(c) for c in R.checks if not c.active],
                       titre_non="Non vérifiés ici")
    _formules(R)
    ER._tableau_33(R)
    _parametres_retenus(R)


def onglet_efforts(R):
    st.caption("Les efforts des tiges viennent de la platine rigide (lit de ressorts sans traction, tiges tendues) — "
               "le modèle des logiciels d'ancrage : ce sont ces valeurs que l'on compare à Spit ou Hilti. "
               + ("Appui sur toute la platine, comme eux." if R.plein else "Appui sur la bande c (EN 1993-1-8) : un peu plus "
                  "de pression et de traction que sur toute la platine."))
    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1, st.container(key="ppb_tab_efforts"):
        st.markdown("##### Boulons : position, effort, appui")
        lignes = []
        for t in R.tiges:
            tendu = t["T"] > 1e-6
            lignes.append([js_str(t["i"] + 1), F(t["x"], 0) + " ; " + F(t["y"], 0), F(t["T"], 1) if tendu else "0",
                           taux_html(t["T"] / (R.Ft_anc / 1e3), t["T"] <= R.Ft_anc / 1e3 + 1e-9) if tendu else "—",
                           F(t["V_ec3"], 1) + " / " + F(t["V_en"], 1),
                           t["appui"]["face"].replace("élément ", "É") if tendu else "—", F(t["m"], 0) if tendu else "—"])
        st.markdown(tableau_html(["n°", Html("x ; y<br>(mm)"), Html("T<br>(kN)"), Html("T /<br>Ft,Rd"),
                                  Html("V EC3 /<br>brut (kN)"), "Appui", Html("m<br>(mm)")], lignes,
                                 droite=(2, 3, 4, 6), defilant=True), unsafe_allow_html=True)
        st.caption("V EC3 : frottement Cf,d·N déduit ; brut : sans frottement (valeur des logiciels d'ancrage).")
    with c2:
        st.markdown("##### Répartition")
        g = [["Résultante comprimée C", F(R.C, 1) + " kN en (" + F(R.xC, 0) + " ; " + F(R.yC, 0) + ") mm"],
             ["Pression maximale", F(R.pmax, 2) + " MPa (fjd = " + F(R.fjd, 1) + ")"],
             ["Zone comprimée", F(R.A_c / 100, 0) + " cm² sur " + F(R.A_eff / 100, 0) + " cm² d'appui"],
             ["Traction totale ΣT", F(R.T_tot, 1) + " kN (T max = " + F(R.T_max, 1) + " kN)"],
             ["Raideurs", "kc = " + F(R.k_c, 1) + " N/mm³ ; kb = " + F(R.k_b / 1e3, 0) + " kN/mm"],
             ["Torsion autour du centre du groupe", F(R.T_G, 2) + " kNm (centre en " + F(R.gx, 0) + " ; " + F(R.gy, 0) + ")"]]
        st.markdown(tableau_md(["Grandeur", "Valeur"], g))


def onglet_note(R, u):
    c1, c2 = st.columns([1.6, 2.6], vertical_alignment="center")
    complet = c1.checkbox("Texte : toutes les vérifications", key="ppb_ui_txt_complet")
    t = texte.construire_texte(R, complet)
    from ..texte import nom_fichier
    with c1:
        st.download_button("⬇️ Exporter le texte (.txt)", data=("﻿" + t).encode("utf-8"),
                           file_name=nom_fichier(u, ".txt"), mime="text/plain;charset=utf-8",
                           width="stretch", key="ppb_btn_txt")
    c2.caption("Note de calcul (2 pages, A4 paysage — le calcul à la main + le plan de principe à l'échelle) : "
               "bouton « 📄 PDF » de la barre d'outils.")
    st.code(t, language=None)


def onglet_benchmark(ecrire):
    from .benchmark import run_bench_manuel, CAS
    B = run_bench_manuel()
    from acier.assemblages.ui_commun import bloc_statut
    bloc_statut("Benchmark du calcul manuel : " + B["statut"],
                ["Chaque cas est recalculé à la main dans tests/test_platine_manuelle.py (formule par formule) et rejoué ici."],
                "ok" if B["ok"] else "nok")
    for i, c in enumerate(B["cas"]):
        st.markdown(f"#### {c['id']} – {c['titre']}")
        k1, k2 = st.columns([4, 1.4], vertical_alignment="center")
        k1.caption(c["data"])
        with k2:
            st.button("Charger ces données dans l'outil", key=f"ppb_btn_bm_m{i}", width="stretch",
                      on_click=_charger, args=(CAS[i]["inp"], ecrire))
        # Un écart qui s'afficherait « ±0,000 % » perd son signe (pas de « -0,000 % »).
        lignes = [[r["lab"], F(r["main"], 3) + ("" if r["unit"] == "-" else " " + r["unit"]), F(r["val"], 3),
                   F((r["ec"] if abs(r["ec"]) >= 5e-6 else 0.0) * 100, 3) + " %", "OK" if r["ok"] else "À CONTRÔLER"]
                  for r in c["rows"]]
        st.markdown(tableau_md(["Grandeur", "Manuel", "Outil", "Écart", ""], lignes, droite=(1, 2, 3)))


def _charger(inp, ecrire):
    from ..entrees import defaults
    d = defaults()
    d.update(inp)
    ecrire(d)
    st.session_state["ppb_ui_alerte"] = None
    st.session_state["ppb_ui_onglet"] = ONGLETS[0]
    st.session_state["_ppb_toast"] = "Données chargées : voir Vérifications"


def onglets(R, u, ecrire):
    st.session_state["_ppb_R"] = R
    st.segmented_control("Onglet", list(ONGLETS), key="ppb_ui_onglet", label_visibility="collapsed", wrap=True)
    onglet = st.session_state.get("ppb_ui_onglet") or ONGLETS[0]
    if onglet not in ONGLETS:
        onglet = ONGLETS[0]
    if onglet == ONGLETS[1]:
        onglet_efforts(R)
    elif onglet == ONGLETS[2]:
        onglet_note(R, u)
    elif onglet == ONGLETS[3]:
        onglet_benchmark(ecrire)
    elif onglet == ONGLETS[4]:
        st.markdown(METHODE)
    else:
        onglet_verifications(R)

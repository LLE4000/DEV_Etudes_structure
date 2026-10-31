# -*- coding: utf-8 -*-
"""La carte de données du pied de poteau : ce qui n'est pas géométrique, en
blocs compacts — la COLONNE DE DROITE de l'écran. La géométrie (platine
bp, lp, tp ; grille des tiges wx, wy ; ancrage hef, dh ; mortier tg ; massif
bc, lc ; gorge a) ne se modifie que sur le dessin, figé dans la colonne de
gauche (ou dans le panneau des cotes en repli).

Clés de session ``ppb_<clé>`` : source unique, écrite AVANT l'instanciation
des widgets (le dessin est rendu avant la carte)."""
import streamlit as st

from acier.assemblages.ui_commun import titre_bloc
from .entrees import CHAMPS, champ_visible, perso_c, acier_perso, VISSERIE_DEFAUT
from .schemas import CLE_PAR_COTE

PREFIXE = "ppb_"

PAS = {"N_Ed": 10.0, "Mx_Ed": 5.0, "My_Ed": 5.0, "Vx_Ed": 5.0, "Vy_Ed": 5.0, "T_Ed": 1.0,
       "a_cc": 0.05, "g_M0": 0.05, "g_M2": 0.05, "g_c": 0.05, "g_Mc": 0.05, "C_f": 0.05, "bw_u": 0.05,
       "fy_u": 5.0, "fu_u": 5.0, "L_c": 100.0, "hc_u": 50.0, "f_g": 5.0,
       "h_u": 5.0, "b_u": 5.0, "tw_u": 0.5, "tf_u": 0.5, "r_u": 0.5, "t_u": 0.5, "n_ax": 1.0, "n_ay": 1.0,
       "bp_u": 10.0, "lp_u": 10.0, "tp_u": 1.0, "w_x": 10.0, "w_y": 10.0, "hef": 10.0, "d_h": 2.0,
       "t_g": 5.0, "bc_u": 50.0, "lc_u": 50.0, "a_w": 0.5}

LIBELLES = {
    "perim": "Périmètre de la vérification",
    "forme": "Forme", "prof_c": "Profilé", "nu_c": "Nuance",
    "h_u": "h (mm)", "b_u": "b (mm)", "tw_u": "tw (mm)", "tf_u": "tf (mm)", "r_u": "r (mm)", "t_u": "t (mm)",
    "nu_pl": "Nuance de la platine",
    "ancr": "Tige", "classe": "Classe", "typ_a": "Ancrage",
    "n_ax": "Files (selon x)", "n_ay": "Rangées (selon y)",
    "beton": "Béton", "fiss": "Fissuré", "hc_u": "Hauteur du massif hc (mm)",
    "N_Ed": "NEd (kN)", "Mx_Ed": "Mx,Ed (kNm)", "My_Ed": "My,Ed (kNm)",
    "Vx_Ed": "Vx,Ed (kN)", "Vy_Ed": "Vy,Ed (kN)", "T_Ed": "TEd (kNm)",
    "a_cc": "αcc", "g_c": "γc", "g_Mc": "γMc", "g_M0": "γM0", "g_M2": "γM2", "C_f": "Cf,d",
    "f_g": "Mortier fg (MPa)", "L_c": "Hauteur du poteau Lc (mm)", "contrev": "Contreventé",
    "fy_u": "fy (MPa)", "fu_u": "fu (MPa)", "bw_u": "βw",
    "id_projet": "Projet", "id_rep": "Repère", "id_red": "Rédacteur", "id_date": "Date",
}

# Ce qui se modifie sur le dessin — jamais dans la carte
GEOMETRIE_DESSIN = frozenset(CLE_PAR_COTE.values())

BLOCS = [
    ("PÉRIMÈTRE", [(["perim"], None)]),
    ("POTEAU", [(["forme", "prof_c", "nu_c"], None),
                (["h_u", "b_u", "tw_u", "tf_u", "r_u", "t_u"], perso_c)]),
    ("PLATINE", [(["nu_pl"], None)]),
    ("TIGES", [(["ancr", "classe", "typ_a"], None), (["n_ax", "n_ay"], None)]),
    ("BÉTON", [(["beton", "fiss", "hc_u"], None)]),
    ("EFFORTS ELU", [(["N_Ed", "Mx_Ed", "My_Ed"], None), (["Vx_Ed", "Vy_Ed", "T_Ed"], None)]),
]
AVANCES = [(["a_cc", "g_c", "g_Mc"], None), (["g_M0", "g_M2", "C_f"], None),
           (["f_g", "L_c", "contrev"], None), (["fy_u", "fu_u", "bw_u"], acier_perso)]
IDENTIFICATION = [(["id_projet", "id_rep"], None), (["id_red", "id_date"], None)]


def K(k):
    return PREFIXE + k


def valeur_widget(k, v):
    f = CHAMPS.get(k)
    if f is None:
        return v
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


def libelle(f, u, chauds):
    lab = LIBELLES.get(f["k"], f["l"])
    if f["k"] in chauds:
        lab = "🔴 " + lab
    return lab


def _aide(f, u):
    h = f.get("h") or ""
    if f["l"] != LIBELLES.get(f["k"], f["l"]):
        h = f["l"] + (". " + h if h else "")
    return h or None


def widget(f, u, chauds):
    k = f["k"]; key = K(k)
    st.session_state[key] = valeur_widget(k, st.session_state.get(key))
    lab = libelle(f, u, chauds)
    # prj_verrou : assemblage verrouillé dans son projet (acier/projet_ui.py)
    dis = bool(st.session_state.get("prj_verrou", False))
    if f["t"] == "s":
        st.selectbox(lab, f["o"], key=key, help=_aide(f, u), disabled=dis)
    elif f["t"] == "x":
        st.text_input(lab, key=key, help=_aide(f, u), disabled=dis)
    else:
        st.number_input(lab, key=key, step=PAS.get(k, 1.0), format="%g", help=_aide(f, u), disabled=dis)


def _lignes(lignes, u, chauds, titre=None):
    premiere = True
    for cles, cond in lignes:
        if cond and not cond(u):
            continue
        cles = [k for k in cles if champ_visible(u, k)]
        if not cles:
            continue
        if titre is not None:
            cols = st.columns([0.62] + [1] * len(cles), gap="small", vertical_alignment="bottom")
            with cols[0]:
                if premiere:
                    titre_bloc(titre)
                else:
                    st.empty()
            cols = cols[1:]
        else:
            cols = st.columns(len(cles), gap="small")
        for col, k in zip(cols, cles):
            with col:
                widget(CHAMPS[k], u, chauds)
        premiere = False


def _contient_chaud(lignes, chauds, u):
    return any(k in chauds for cles, cond in lignes if not cond or cond(u) for k in cles)


def ligne_visserie(u):
    """Annotation de FABRICATION hors moteur : rondelles et écrous par tige —
    cartouche du plan de principe, enregistrée avec le calcul."""
    cols = st.columns([0.62, 3], gap="small", vertical_alignment="bottom")
    with cols[0]:
        titre_bloc("VISSERIE")
    with cols[1]:
        st.session_state.setdefault("ppb_visserie", VISSERIE_DEFAUT)
        st.text_input("Par tige", key="ppb_visserie",
                      help="Rondelles, écrou et contre-écrou par tige — porté au cartouche du plan de principe.",
                      disabled=bool(st.session_state.get("prj_verrou", False)))


def avances(u, chauds=frozenset()):
    chaud_av = _contient_chaud(AVANCES, chauds, u)
    with st.expander(("🔴 " if chaud_av else "") + "Paramètres avancés", expanded=chaud_av):
        st.caption("Coefficients partiels (γM0 = 1,00 ANB belge ; γM2 = 1,25 ; γc = γMc = 1,5 ; αcc = 0,85 — "
                   "hypothèse à confirmer sur le texte NBN), mortier, hauteur du poteau pour la classification "
                   "de rigidité.")
        with st.container(gap=None):
            _lignes(AVANCES, u, chauds)
        st.checkbox("Dessin interactif (cotes et pièces cliquables sur le schéma)", key="ppb_ui_composant",
                    help="Désactivé : les cotes se modifient dans le panneau « Cotes » en tête de cette colonne.")


def identification(u, chauds=frozenset()):
    with st.expander("Identification (note et export)"):
        with st.container(gap=None):
            _lignes(IDENTIFICATION, u, chauds)


def carte(u, chauds=frozenset()):
    """La carte complète : blocs par objet, visserie, puis les deux replis."""
    with st.container(gap=None):
        for titre, lignes in BLOCS:
            _lignes(lignes, u, chauds, ("🔴 " if _contient_chaud(lignes, chauds, u) else "") + titre)
            if titre == "TIGES":
                ligne_visserie(u)
    avances(u, chauds)
    identification(u, chauds)

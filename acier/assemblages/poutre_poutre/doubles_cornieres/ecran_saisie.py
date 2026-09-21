# -*- coding: utf-8 -*-
"""La carte de données : ce qui n'est pas géométrique, en blocs compacts.

Un objet = un bloc de deux lignes : MODE ET FIXATIONS, PROFILÉS, CORNIÈRES,
BOULONS, EFFORTS ; puis, repliés, PARAMÈTRES AVANCÉS (ce qu'on ne modifie
pas dans 90 % des cas) et IDENTIFICATION. La géométrie — entraxes, pinces,
longueur et position des cornières, grugeage, jeu, gorges et retours des
cordons — ne se modifie que sur le dessin (une seule entrée par paramètre).

Chaque champ est un widget Streamlit dont la clé est ``asm_<clé>`` : la
valeur vit dans ``st.session_state`` (source unique, partagée avec le dessin,
les exports et la note). Aucun ``value=`` n'est passé aux widgets : l'état
est écrit AVANT leur instanciation.
"""
import streamlit as st

from acier.assemblages.ui_commun import titre_bloc
from acier.bibliotheques import ORI_P, ORI_S
from .entrees import (mode_predim, PILOTEES_PAR_PREDIM, CHAMPS, champ_visible,
                      perso_P, perso_S, perso_C, _cat_non_A, _acier_perso)
from .schemas import CLE_PAR_COTE

# Affichage abrégé de certaines options (la valeur enregistrée ne change pas)
AFFICHAGE = {"orient": {ORI_P: "Grande aile → principale", ORI_S: "Grande aile → secondaire"}}

PREFIXE = "asm_"

# Pas des champs numériques (convenance du compteur ; la saisie reste libre)
PAS = {"V_Ed": 5.0, "N_Ed": 5.0, "H_Ed": 5.0, "M_Ed": 1.0,
       "mu_s": 0.05, "k_s": 0.05, "k_ser": 0.05, "g_M0": 0.05, "g_M2n": 0.05, "g_M2": 0.05,
       "g_M3": 0.05, "g_M3s": 0.05, "k_rot": 0.05, "eta_c": 0.05, "k_e1": 0.1, "k_p1": 0.1,
       "k_e2": 0.1, "bw_u": 0.05, "fy_u": 5.0, "fu_u": 5.0,
       "LC_u": 5.0, "z_C": 5.0, "p1S_u": 5.0, "p2_S": 5.0, "e1S_u": 5.0, "e2b_u": 5.0,
       "p1P_u": 5.0, "p2_P": 5.0, "e1P_u": 5.0, "gA_u": 5.0, "d_top": 5.0, "d_nt": 5.0,
       "d_nb": 5.0, "l_n": 5.0, "k1_u": 5.0, "k2_u": 5.0, "hP_u": 5.0, "bP_u": 5.0,
       "hS_u": 5.0, "bS_u": 5.0, "lh_P": 5.0, "lh_S": 5.0,
       "twP_u": 0.5, "tfP_u": 0.5, "rP_u": 0.5, "twS_u": 0.5, "tfS_u": 0.5, "rS_u": 0.5,
       "kt_u": 0.5, "kr_u": 0.5, "a_P": 0.5, "a_S": 0.5}

# Libellés courts de la carte (l'aide longue reste dans l'infobulle)
LIBELLES = {
    "mode_calc": "Mode", "fix_P": "Ailes A → principale", "fix_S": "Ailes B → secondaire",
    "prof_P": "Principale (porteuse)", "nu_P": "Nuance", "prof_S": "Secondaire (portée)", "nu_S": "Nuance",
    "hP_u": "h (mm)", "bP_u": "b (mm)", "twP_u": "tw (mm)", "tfP_u": "tf (mm)", "rP_u": "r (mm)",
    "hS_u": "h (mm)", "bS_u": "b (mm)", "twS_u": "tw (mm)", "tfS_u": "tf (mm)", "rS_u": "r (mm)",
    "corn_u": "Cornière", "nu_C": "Nuance", "orient": "Orientation", "k1_u": "Grande aile (mm)",
    "k2_u": "Petite aile (mm)", "kt_u": "Épaisseur (mm)", "kr_u": "Congé (mm)",
    "boulon_u": "Boulon", "classe": "Classe", "trou": "Trou", "cat": "Catégorie",
    "n1S_u": "S : rangées n1", "n2S_u": "S : files n2", "n1P_u": "P : rangées n1", "n2P_u": "P : files n2",
    "mu_s": "μ", "k_s": "ks", "k_ser": "ELS / ELU",
    "V_Ed": "VEd (kN)", "N_Ed": "NEd (kN)", "H_Ed": "HEd (kN)", "M_Ed": "MEd (kNm)",
    "r_n": "Rayon de grugeage (mm)", "lt_ok": "Secondaire maintenue au déversement",
    "d0_u": "d0 imposé (mm)", "filet": "Filetage dans le plan cisaillé",
    "g_M0": "γM0", "g_M2n": "γM2 sections nettes", "g_M2": "γM2 boulons, soudures", "g_M3": "γM3",
    "g_M3s": "γM3,ser", "opt_exc": "Excentricité gA reprise par P", "k_rot": "Facteur sur Fv,Rd (P)",
    "opt_blf": "βLf assemblages longs", "expo": "Exposé aux intempéries",
    "fy_u": "fy (MPa)", "fu_u": "fu (MPa)", "bw_u": "βw",
    "eta_c": "Taux cible", "k_e1": "e1 = k·d0", "k_p1": "p1 = k·d0", "k_e2": "e2 = k·d0",
    "pd_dmin": "Ø minimal", "pd_dmax": "Ø maximal",
    "id_projet": "Projet", "id_rep": "Repère", "id_red": "Rédacteur", "id_date": "Date",
}

# Ce qui se modifie sur le dessin — jamais dans la carte
GEOMETRIE_DESSIN = frozenset(CLE_PAR_COTE.values()) | {"lh_P"}

# Blocs essentiels : (titre, lignes de clés, condition d'affichage de la ligne)
BLOCS = [
    ("MODE ET FIXATIONS", [(["mode_calc", "fix_P", "fix_S"], None)]),
    ("PROFILÉS", [(["prof_P", "nu_P"], None), (["hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"], perso_P),
                  (["prof_S", "nu_S"], None), (["hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"], perso_S)]),
    ("CORNIÈRES", [(["corn_u", "nu_C", "orient"], None), (["k1_u", "k2_u", "kt_u", "kr_u"], perso_C)]),
    ("BOULONS", [(["boulon_u", "classe", "trou", "cat"], None),
                 (["n1S_u", "n2S_u", "n1P_u", "n2P_u"], None),
                 (["mu_s", "k_s", "k_ser"], _cat_non_A)]),
    ("EFFORTS ELU", [(["V_Ed", "N_Ed", "H_Ed", "M_Ed"], None)]),
]
AVANCES = [(["r_n", "lt_ok"], None), (["d0_u", "filet"], None),
           (["g_M0", "g_M2", "g_M2n"], None), (["g_M3", "g_M3s"], _cat_non_A),
           (["opt_exc", "k_rot"], None), (["opt_blf", "expo"], None),
           (["fy_u", "fu_u", "bw_u"], _acier_perso),
           (["eta_c", "k_e1", "k_p1", "k_e2"], mode_predim), (["pd_dmin", "pd_dmax"], mode_predim)]
IDENTIFICATION = [(["id_projet", "id_rep"], None), (["id_red", "id_date"], None)]


def K(k):
    return PREFIXE + k


def valeur_widget(k, v):
    """Normalise une valeur pour son widget : nombre → float (ou None si
    vide), choix → chaîne, texte → chaîne."""
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
        # les sélecteurs 1 / 2 acceptent un nombre (1.0 → « 1 »)
        if s not in f["o"]:
            try:
                s = str(int(float(s)))
            except (TypeError, ValueError):
                pass
        return s if s in f["o"] else f["o"][0]
    return "" if v is None else str(v)


def libelle(f, u, chauds):
    """Libellé court du champ : marque ◆ en prédimensionnement, 🔴 si le
    champ est mis en cause par l'alerte sélectionnée."""
    lab = LIBELLES.get(f["k"], f["l"])
    if mode_predim(u) and f["k"] in PILOTEES_PAR_PREDIM:
        lab += " ◆"
    if f["k"] in chauds:
        lab = "🔴 " + lab
    return lab


def _aide(f, u):
    h = f.get("h") or ""
    if f["l"] != LIBELLES.get(f["k"], f["l"]):
        h = f["l"] + (". " + h if h else "")
    if mode_predim(u) and f["k"] in PILOTEES_PAR_PREDIM:
        h = (h + " " if h else "") + "◆ Piloté par la proposition du prédimensionnement."
    return h or None


def widget(f, u, chauds):
    """Le widget d'un champ, sur sa clé de session."""
    k = f["k"]; key = K(k)
    st.session_state[key] = valeur_widget(k, st.session_state.get(key))
    lab = libelle(f, u, chauds)
    dis = bool(mode_predim(u) and k in PILOTEES_PAR_PREDIM)
    if f["t"] == "s":
        aff = AFFICHAGE.get(k)
        st.selectbox(lab, f["o"], key=key, help=_aide(f, u), disabled=dis,
                     format_func=(lambda o, a=aff: a.get(o, o)) if aff else str)
    elif f["t"] == "x":
        st.text_input(lab, key=key, help=_aide(f, u))
    else:
        st.number_input(lab, key=key, step=PAS.get(k, 1.0), format="%g", help=_aide(f, u), disabled=dis)


def _lignes(lignes, u, chauds, titre=None):
    """Les lignes d'un bloc : une ligne = des widgets côte à côte ; le titre
    du bloc, s'il y en a un, occupe une colonne étroite à gauche de la
    première ligne, aligné sur les champs."""
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


def carte(u, chauds=frozenset()):
    """La carte de données pour l'état ``u`` ; ``chauds`` = clés mises en
    cause par l'alerte sélectionnée (champs marqués, replis dépliés)."""
    with st.container(gap=None):
        for titre, lignes in BLOCS:
            _lignes(lignes, u, chauds, ("🔴 " if _contient_chaud(lignes, chauds, u) else "") + titre)
    chaud_av = _contient_chaud(AVANCES, chauds, u)
    with st.expander(("🔴 " if chaud_av else "") + "Paramètres avancés", expanded=chaud_av):
        st.caption("Coefficients partiels et options du modèle : valeurs recommandées de l'EN 1993, à confirmer "
                   "par rapport à l'ANB applicable.")
        with st.container(gap=None):
            _lignes(AVANCES, u, chauds)
        st.checkbox("Dessin interactif (cotes et groupes cliquables)", key="asm_ui_composant",
                    help="Désactivé : le dessin reste affiché et les cotes se modifient dans le panneau « Cotes ».")
    with st.expander("Identification (note et export)"):
        with st.container(gap=None):
            _lignes(IDENTIFICATION, u, chauds)


# rétro-compatibilité (tests, anciens appels)
render = carte

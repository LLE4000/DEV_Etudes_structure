# -*- coding: utf-8 -*-
"""Colonne de saisie : les 84 entrées, par groupes, avec les conditions
d'affichage du formulaire de référence.

Chaque champ est un widget Streamlit dont la clé est ``asm_<clé>`` : la
valeur vit dans ``st.session_state`` (source unique, partagée avec le dessin,
le panneau des cotes, les exports et le rapport). Aucun ``value=`` n'est passé
aux widgets : l'état est écrit AVANT leur instanciation.
"""
import streamlit as st

from .entrees import (groupes_visibles, mode_predim, PILOTEES_PAR_PREDIM, CHAMPS)

PREFIXE = "asm_"

# Pas des champs numériques (convenance du compteur ; la saisie reste libre)
PAS = {"V_Ed": 5.0, "N_Ed": 5.0, "H_Ed": 5.0, "M_Ed": 1.0,
       "mu_s": 0.05, "k_s": 0.05, "k_ser": 0.05, "g_M0": 0.05, "g_M2n": 0.05, "g_M2": 0.05,
       "g_M3": 0.05, "g_M3s": 0.05, "k_rot": 0.05, "eta_c": 0.05, "k_e1": 0.1, "k_p1": 0.1,
       "k_e2": 0.1, "bw_u": 0.05, "fy_u": 5.0, "fu_u": 5.0,
       "LC_u": 5.0, "z_C": 5.0, "p1S_u": 5.0, "p2_S": 5.0, "e1S_u": 5.0, "e2b_u": 5.0,
       "p1P_u": 5.0, "p2_P": 5.0, "e1P_u": 5.0, "gA_u": 5.0, "d_top": 5.0, "d_nt": 5.0,
       "d_nb": 5.0, "l_n": 5.0, "k1_u": 5.0, "k2_u": 5.0, "hP_u": 5.0, "bP_u": 5.0,
       "hS_u": 5.0, "bS_u": 5.0,
       "twP_u": 0.5, "tfP_u": 0.5, "rP_u": 0.5, "twS_u": 0.5, "tfS_u": 0.5, "rS_u": 0.5,
       "kt_u": 0.5, "kr_u": 0.5, "a_P": 0.5, "a_S": 0.5}

# Groupes dépliés par défaut (les autres restent repliés, sauf s'ils portent
# un champ mis en cause par l'alerte sélectionnée)
OUVERTS = {"Mode de calcul et fixations", "Poutre principale (porteuse)",
           "Poutre secondaire (portée) et grugeage", "Cornières (2 cornières identiques)",
           "Boulons", "Groupe S – boulons dans l'âme de la poutre secondaire (2 plans)",
           "Groupe P – boulons dans l'âme de la poutre principale (1 plan, par cornière)",
           "Soudures (cordons d'angle)", "Efforts de calcul ELU", "Règles du prédimensionnement"}


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
    """Libellé du champ : unité, marque ◆ en prédimensionnement, 🔴 si le
    champ est mis en cause par l'alerte sélectionnée."""
    lab = f["l"]
    if f.get("un") and f["un"] != "-":
        lab += f" ({f['un']})"
    if mode_predim(u) and f["k"] in PILOTEES_PAR_PREDIM:
        lab += " ◆"
    if f["k"] in chauds:
        lab = "🔴 " + lab
    return lab


def _aide(f, u):
    h = f.get("h") or ""
    if mode_predim(u) and f["k"] in PILOTEES_PAR_PREDIM:
        h = (h + " " if h else "") + "◆ Remplacé par la proposition du prédimensionnement."
    return h or None


def widget(f, u, chauds):
    """Le widget d'un champ, sur sa clé de session."""
    k = f["k"]; key = K(k)
    st.session_state[key] = valeur_widget(k, st.session_state.get(key))
    lab = libelle(f, u, chauds)
    if f["t"] == "s":
        st.selectbox(lab, f["o"], key=key, help=_aide(f, u))
    elif f["t"] == "x":
        st.text_input(lab, key=key, help=_aide(f, u))
    else:
        st.number_input(lab, key=key, step=PAS.get(k, 1.0), format="%g", help=_aide(f, u))


def render(u, chauds=frozenset()):
    """La colonne de saisie pour l'état ``u`` ; ``chauds`` = clés mises en
    cause par l'alerte sélectionnée (groupes dépliés, champs marqués)."""
    for g, champs in groupes_visibles(u):
        chaud = any(f["k"] in chauds for f in champs)
        titre = ("🔴 " if chaud else "") + g["t"]
        with st.expander(titre, expanded=chaud or g["t"] in OUVERTS):
            if g.get("note"):
                st.caption(g["note"])
            for f in champs:
                widget(f, u, chauds)

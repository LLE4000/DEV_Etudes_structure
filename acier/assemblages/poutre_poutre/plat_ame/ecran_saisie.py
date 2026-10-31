# -*- coding: utf-8 -*-
"""La carte de données du plat d'âme — les DONNÉES seulement : le kit
(``acier/assemblages/carte.py``) rend la carte sur les trois formats.

Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md §4.2) :

    EN-TÊTE     [ Vérification | Prédimensionnement ] ?   (+ Taux cible · Ø min · Ø max en prédim)
    PROFILÉS    | Principale ▾ | Nuance ▾ | Secondaire ▾ | Nuance ▾ |   ↳ perso (si Personnalisé)
      ↳ PLAT     | Nuance du plat ▾ |
    BOULONS     | Boulon ▾ | Classe ▾ | Trou ▾ ? | Catégorie ▾ ? |
      ↳ groupe   | Rangées n1 | Files n2 ▾ |     ↳ glissement (cat ≠ A) | μ | ks | ELS / ELU ? |
    EFFORTS ELU ? | VEd (kN) | NEd (kN) | MEd (kNm) |
    ▸ Paramètres avancés   ▸ Cartouche (note et plan)

La géométrie (hp, tp, bp, zp, pinces, entraxes, grugeage, jeu, gorge) ne se
modifie que sur le dessin (P6). Une forme par clé : ``LIBELLE`` est lu par
la carte, les fenêtres d'objet du dessin, le panneau Cotes et les chips
d'alerte (P5) ; un « ? » n'existe que pour une entrée de ``AIDES`` (P3).

Clés de session ``fpl_<clé>`` : source unique, écrite AVANT l'instanciation
des widgets (le dessin est rendu avant la carte)."""
import streamlit as st

from acier.assemblages import carte as kit
from acier.assemblages.carte import L
from .entrees import (mode_predim, PILOTEES_PAR_PREDIM, CHAMPS, UNITE, champ_visible, perso_P, perso_S,
                      _cat_non_A, _acier_perso, VISSERIE_DEFAUT, MODE_VERIF, MODE_PREDIM)
from .schemas import CLE_PAR_COTE

PREFIXE = "fpl_"

PAS = {"V_Ed": 5.0, "N_Ed": 5.0, "M_Ed": 1.0,
       "mu_s": 0.05, "k_s": 0.05, "k_ser": 0.05, "g_M0": 0.05, "g_M1": 0.05,
       "g_M2n": 0.05, "g_M2": 0.05, "g_M3": 0.05, "g_M3s": 0.05,
       "eta_c": 0.05, "k_e1": 0.1, "k_p1": 0.1, "k_e2": 0.1, "bw_u": 0.05,
       "fy_u": 5.0, "fu_u": 5.0,
       "hp_u": 5.0, "bp_u": 5.0, "z_C": 5.0, "p1_u": 5.0, "p2_u": 5.0,
       "e1_u": 5.0, "e2b_u": 5.0, "d_top": 5.0, "d_nt": 5.0, "d_nb": 5.0,
       "l_n": 5.0, "hP_u": 5.0, "bP_u": 5.0, "hS_u": 5.0, "bS_u": 5.0,
       "twP_u": 0.5, "tfP_u": 0.5, "rP_u": 0.5, "twS_u": 0.5, "tfS_u": 0.5,
       "rS_u": 0.5, "tp_u": 0.5, "a_w": 0.5}

# Libellés sans unité (≤ 16 dans une rangée de 4, ≤ 20 ailleurs) ; l'unité vient de UNITE.
_LIBELLE = {
    "mode_calc": "Mode",
    "prof_P": "Principale", "nu_P": "Nuance", "prof_S": "Secondaire", "nu_S": "Nuance",
    "hP_u": "h", "bP_u": "b", "twP_u": "tw", "tfP_u": "tf", "rP_u": "r",
    "hS_u": "h", "bS_u": "b", "twS_u": "tw", "tfS_u": "tf", "rS_u": "r",
    "nu_pl": "Nuance du plat",
    "boulon_u": "Boulon", "classe": "Classe", "trou": "Trou", "cat": "Catégorie",
    "n1_u": "Rangées n1", "n2_u": "Files n2",
    "mu_s": "μ", "k_s": "ks (Tab. 3.6)", "k_ser": "ELS / ELU",
    "V_Ed": "VEd", "N_Ed": "NEd", "M_Ed": "MEd",
    "r_n": "Rayon grugeage", "lt_ok": "Déversement empêché",
    "d0_u": "d0 imposé", "filet": "Filetage cisaillé",
    "g_M0": "γM0", "g_M1": "γM1 (ANB)", "g_M2": "γM2 boulons", "g_M2n": "γM2 nettes",
    "g_M3": "γM3 (ELU)", "g_M3s": "γM3,ser (ELS)",
    "opt_blf": "βLf joints longs", "expo": "Intempéries",
    "fy_u": "fy", "fu_u": "fu", "bw_u": "βw",
    "eta_c": "Taux cible", "k_e1": "e1 = k·d0", "k_p1": "p1 = k·d0", "k_e2": "e2 = k·d0",
    "pd_dmin": "Ø minimal", "pd_dmax": "Ø maximal",
    "id_projet": "Projet", "id_rep": "Repère", "id_red": "Rédacteur", "id_date": "Date",
    "visserie": "Visserie par boulon",
    # géométrie : fenêtres d'objet, panneau Cotes, chips d'alerte
    "hp_u": "hp", "tp_u": "tp", "bp_u": "bp", "z_C": "zp", "a_w": "a",
    "e1_u": "e1", "p1_u": "p1", "e2b_u": "e2,b", "p2_u": "p2",
    "g_h": "gh", "d_top": "Δz", "d_nt": "dc,sup", "d_nb": "dc,inf", "l_n": "c",
    # tête de la colonne dessin
    "ui_niveau": "Niveau de cotation", "ui_composant": "Dessin interactif",
}
LIBELLE = {k: v + (" (" + UNITE[k].strip() + ")" if UNITE.get(k) else "") for k, v in _LIBELLE.items()}

# Les aides (« ? ») — textes de la charte §3 ; une clé absente n'a pas de « ? ».
AIDES = {
    "mode_calc": "Prédimensionnement : boulons, rangées et pièce d'attache proposés (onglet Prédim) puis "
                 "vérifiés ; les champs ◆ sont alors pilotés par la proposition.",
    "trou": "Surdimensionné : Fb,Rd × 0,8 (EN 1993-1-8 Tab. 3.4) et d0 à saisir dans Paramètres avancés › TROUS.",
    "cat": "A : pression diamétrale. B : sans glissement à l'ELS. C : sans glissement à l'ELU (boulons "
           "précontraints 8.8 ou 10.9).",
    "k_ser": "Rapport des efforts ELS / ELU ; 1,0 = hypothèse sécuritaire.",
    "r_n": "Information d'exécution : non utilisé dans les résistances ; dessiné sur l'élévation.",
    "lt_ok": "Condition de la règle de stabilité locale du grugeage (MSB P5 §4.2.5) ; un plat long sur poutre "
             "non maintenue demande un examen à part.",
    "d0_u": "Vide = d0 normal du boulon (Tab. 3.3) ; à saisir pour un trou surdimensionné.",
    "g_M1": "Instabilités (déversement du plat long). NBN EN 1993-1-1 ANB : 1,10 ; valeur recommandée de "
            "l'EN : 1,00.",
    "opt_blf": "βLf des assemblages longs, EN 1993-1-8 §3.8 (Lj > 15d). Les guides ne l'appliquent pas aux "
               "attaches d'âme ; Oui = variante conservatrice.",
    "expo": "Assemblage exposé aux intempéries : active les pinces maximales 4t + 40 mm (Tab. 3.3).",
    "visserie": "Rondelles et écrous par boulon, portés au cartouche du plan. Ex. : « 1 rondelle + 2 écrous "
                "(contre-écrou) », « 2 rondelles + 1 écrou ».",
    "ui_composant": "Décoché : dessins fixes ; les cotes se modifient dans le panneau « Cotes » en tête de la "
                    "colonne de droite.",
    # aides portées par les fenêtres d'objet et le panneau Cotes (géométrie)
    "hp_u": "Recommandation des guides : hp ≥ 0,6·h de la secondaire (maintien en torsion).",
    "tp_u": "Ductilité (rotule par ovalisation) : tp ≤ 0,5·d du boulon (SCI P358 / MSB P5).",
    "a_w": "Deux cordons, un par face du plat ; pleine résistance a ≥ tp·fy·βw·γM2/(2·fu·γM0) (§4.5.3.3).",
    "g_h": "Jeu entre l'âme porteuse et l'about de la secondaire ; aussi par glissement horizontal de la poutre.",
    "d_top": "0 = dessus des semelles au même niveau ; aussi par glissement vertical de la poutre secondaire.",
    "d_nt": "0 = semelle supérieure non grugée.",
    "d_nb": "0 = semelle inférieure non grugée.",
    # aides de BLOC (titre_bloc(help=)) : les conventions communes, dites une fois
    "EFFORTS ELU": "Efforts dans la secondaire : NEd > 0 traction ; MEd à la face de l'âme porteuse (articulé : "
                   "normalement 0, pris en valeur absolue).",
    # « COEFFICIENTS » (12 lettres) ne tient pas dans la case des titres de 76 px (charte §2 : ≤ 11
    # caractères, jamais coupé) : le sous-bloc s'appelle « COEFF. γ », son « ? » dit l'ANB une fois.
    "COEFF. γ": "ANB belge : γM0 1,00 et γM1 1,10 (NBN EN 1993-1-1 ANB). γM2 boulons : boulons, pression "
                "diamétrale, soudures ; γM2 nettes : sections nettes, bloc.",
    "PRÉDIM": "Pinces et entraxes proposés en multiples de d0 ; minima normatifs e ≥ 1,2·d0, p1 ≥ 2,2·d0 "
              "(Tab. 3.3). Règles de détail modifiables.",
}

# Options courtes à l'AFFICHAGE ; les valeurs brutes (fichiers, tests) ne changent pas (P4).
AFFICHAGE = {"mode_calc": {MODE_VERIF: "Vérification", MODE_PREDIM: "Prédimensionnement"}}

# Ce qui se modifie sur le dessin — jamais dans la carte
GEOMETRIE_DESSIN = frozenset(CLE_PAR_COTE.values())

EN_TETE = [("mode_calc", None), ("eta_c", mode_predim), ("pd_dmin", mode_predim), ("pd_dmax", mode_predim)]
SEGMENTS = frozenset({"mode_calc"})

BLOCS = [
    ("PROFILÉS", [L(["prof_P", "nu_P", "prof_S", "nu_S"]),
                  L(["hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"], perso_P, "principale"),
                  L(["hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"], perso_S, "secondaire"),
                  L(["nu_pl"], None, "PLAT")]),
    ("BOULONS", [L(["boulon_u", "classe", "trou", "cat"]),
                 L(["n1_u", "n2_u"], None, "groupe"),
                 L(["mu_s", "k_s", "k_ser"], _cat_non_A, "glissement")]),
    ("EFFORTS ELU", [L(["V_Ed", "N_Ed", "M_Ed"])]),
]
AVANCES = [
    ("GRUGEAGE", [L(["r_n", "lt_ok"])]),
    ("TROUS", [L(["d0_u", "filet"])]),
    ("COEFF. γ", [L(["g_M0", "g_M1", "g_M2", "g_M2n"]), L(["g_M3", "g_M3s"], _cat_non_A, "glissement")]),
    ("MODÈLE", [L(["opt_blf", "expo"])]),
    ("ACIER PERSO", [L(["fy_u", "fu_u", "bw_u"], _acier_perso)]),
    ("PRÉDIM", [L(["k_e1", "k_p1", "k_e2"], mode_predim)]),
]


def _hors_projet_nom(u):
    # la condition même de interface._infos : dans un projet la case est écrasée par le projet
    return not st.session_state.get("nom_projet")


def _hors_projet_date(u):
    return not st.session_state.get("date")


CARTOUCHE = [("", [L(["id_rep", "id_red", "visserie"]), L(["id_projet", "id_date"], None, "hors projet")])]

D = kit.Description(
    PREFIXE=PREFIXE, CHAMPS=CHAMPS, champ_visible=champ_visible, LIBELLE=LIBELLE, AIDES=AIDES,
    AFFICHAGE=AFFICHAGE, PAS=PAS, BLOCS=BLOCS, AVANCES=AVANCES, PILOTEES_PAR_PREDIM=PILOTEES_PAR_PREDIM,
    mode_predim=mode_predim, VISSERIE_DEFAUT=VISSERIE_DEFAUT, GEOMETRIE_DESSIN=GEOMETRIE_DESSIN,
    EN_TETE=EN_TETE, SEGMENTS=SEGMENTS, CARTOUCHE=CARTOUCHE,
    VISIBLE_SUPPL={"id_projet": _hors_projet_nom, "id_date": _hors_projet_date})


def K(k):
    return PREFIXE + k


def valeur_widget(k, v):
    """Lue aussi par interface.py et acier/projet_ui.py (hydratation)."""
    return kit.valeur_widget(D, k, v)


def carte(u, chauds=frozenset()):
    """La carte complète : en-tête, blocs par objet, puis les deux replis."""
    kit.carte(D, u, chauds)

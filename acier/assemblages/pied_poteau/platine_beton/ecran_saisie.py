# -*- coding: utf-8 -*-
"""La carte de données du pied de poteau — les DONNÉES seulement : le kit
(``acier/assemblages/carte.py``) rend la carte sur les trois formats.

Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md §4.4, §4.5) :

  Vérification complète (``D``)
    EN-TÊTE     [ Calcul manuel | Vérif. complète ] ?   [ EC3 seul | EC3 + EC2-4 ] ?
    POTEAU      | Forme ▾ | Profilé ▾ | Nuance ▾ |      ↳ perso | h | b | tw | tf | r |  (RHS : h | b | t paroi)
    MATÉRIAUX   | Nuance platine ▾ | Béton ▾ | Fissuré ▾ | Massif hc (mm) |
    TIGES       | Tige ▾ | Classe ▾ | Ancrage ▾ ? |      ↳ grille | Files (x) | Rangées (y) |
      ↳ ETA ? (scellée) | τRk (MPa) | k1 cône | NRk,s (kN) | VRk,s (kN) |
    EFFORTS ELU ? | NEd (kN) | Mx,Ed (kNm) | My,Ed (kNm) |      ↳ | Vx,Ed (kN) | Vy,Ed (kN) | TEd (kNm) |

  Calcul manuel (``D_MANUEL`` ; les tables sont rendues par manuel/ecran.py)
    EN-TÊTE     [ Calcul manuel | Vérif. complète ] ?  [ Élastique | Plastique ] ?  [ Bande c | Toute la platine ] ?
    MATÉRIAUX   | Nuance platine ▾ | Béton ▾ | Massif hc (mm) |
    BOULONS     | Boulon ▾ | Classe ▾ | Ancrage ▾ ? |   puis la table des boulons, ÉLÉMENTS, RAIDISSEURS
    EFFORTS ELU ? | NEd | Mx,Ed | My,Ed |   ↳ | Vx,Ed | Vy,Ed | TEd |

  ▸ Paramètres avancés : COEFF. γ ? (↳ αcc ?) · APPUI · RIGIDITÉ (complet) · APPLICATION (manuel) · ACIER PERSO
  ▸ Cartouche (note et plan) : Repère · Rédacteur · Visserie par tige / par boulon ? (+ ↳ hors projet)

La géométrie (platine bp, lp, tp ; grille wx, wy ; ancrage hef, dh ;
mortier tg ; massif bc, lc ; gorge a) ne se modifie que sur le dessin (P6),
dans les deux modes. Une forme par clé : ``LIBELLE`` (et ``LIBELLE_MODE`` :
« Tige » en vérification complète, « Boulon » en calcul manuel) est lu par la
carte, les fenêtres d'objet du dessin, le panneau Cotes et les chips
d'alerte (P5) ; un « ? » n'existe que pour une entrée de ``AIDES`` (P3).

Clés de session ``ppb_<clé>`` : source unique, écrite AVANT l'instanciation
des widgets (le dessin est rendu avant la carte)."""
import dataclasses

import streamlit as st

from acier.assemblages import carte as kit
from acier.assemblages.carte import L
from .entrees import (CHAMPS, UNITE, champ_visible, perso_c, acier_perso, eta, est_complet, VISSERIE_DEFAUT, defaults,
                      MODE_MANUEL, MODE_COMPLET, PERIM_EC3, PERIM_TOUT, CRIT_ELAS, CRIT_EC3, APPUI_EC3, APPUI_PLEIN,
                      FORME_I, FORME_RHS, TYPE_TETE, TYPE_CROSSE, TYPE_ETA)
from .schemas import CLE_PAR_COTE
from .manuel.schemas import CLE_PAR_COTE as CLE_PAR_COTE_MANUEL

PREFIXE = "ppb_"

PAS = {"N_Ed": 10.0, "Mx_Ed": 5.0, "My_Ed": 5.0, "Vx_Ed": 5.0, "Vy_Ed": 5.0, "T_Ed": 1.0,
       "a_cc": 0.05, "g_M0": 0.05, "g_M2": 0.05, "g_c": 0.05, "g_Mc": 0.05, "C_f": 0.05, "bw_u": 0.05,
       "fy_u": 5.0, "fu_u": 5.0, "L_c": 100.0, "hc_u": 50.0, "f_g": 5.0, "xN": 5.0, "yN": 5.0,
       "h_u": 5.0, "b_u": 5.0, "tw_u": 0.5, "tf_u": 0.5, "r_u": 0.5, "t_u": 0.5, "n_ax": 1.0, "n_ay": 1.0,
       "bp_u": 10.0, "lp_u": 10.0, "tp_u": 1.0, "w_x": 10.0, "w_y": 10.0, "hef": 10.0, "d_h": 2.0,
       "t_g": 5.0, "bc_u": 50.0, "lc_u": 50.0, "a_w": 0.5}

# Libellés sans unité (≤ 16 dans une rangée de 4, ≤ 20 ailleurs) ; l'unité vient de UNITE.
_LIBELLE = {
    "mode": "Mode", "perim": "Périmètre", "crit": "Critère", "appui": "Appui",
    "forme": "Forme", "prof_c": "Profilé", "nu_c": "Nuance",
    "h_u": "h", "b_u": "b", "tw_u": "tw", "tf_u": "tf", "r_u": "r", "t_u": "t paroi",
    "nu_pl": "Nuance platine", "beton": "Béton", "fiss": "Fissuré", "hc_u": "Massif hc",
    "ancr": "Tige", "classe": "Classe", "typ_a": "Ancrage",
    "n_ax": "Files (x)", "n_ay": "Rangées (y)",
    "eta_tau": "τRk", "eta_k1": "k1 cône", "eta_nrks": "NRk,s", "eta_vrks": "VRk,s",
    "N_Ed": "NEd", "Mx_Ed": "Mx,Ed", "My_Ed": "My,Ed", "Vx_Ed": "Vx,Ed", "Vy_Ed": "Vy,Ed", "T_Ed": "TEd",
    "g_M0": "γM0", "g_M2": "γM2", "g_c": "γc béton", "g_Mc": "γMc (EN 1992-4)", "a_cc": "αcc",
    "C_f": "Cf,d frottement", "f_g": "Mortier fg", "L_c": "Hauteur Lc", "contrev": "Contreventé",
    "xN": "xN", "yN": "yN",
    "fy_u": "fy", "fu_u": "fu", "bw_u": "βw",
    "id_projet": "Projet", "id_rep": "Repère", "id_red": "Rédacteur", "id_date": "Date",
    "visserie": "Visserie par tige",
    # géométrie : fenêtres d'objet, panneau Cotes, chips d'alerte
    "bp_u": "bp", "lp_u": "lp", "tp_u": "tp", "a_w": "a", "t_g": "tg", "bc_u": "bc", "lc_u": "lc",
    "w_x": "wx", "w_y": "wy", "hef": "hef", "d_h": "dh tête",
    # tête de la colonne dessin
    "ui_niveau": "Niveau de cotation", "ui_composant": "Dessin interactif",
}
LIBELLE = {k: v + (" (" + UNITE[k].strip() + ")" if UNITE.get(k) else "") for k, v in _LIBELLE.items()}

# Vocabulaire PAR MODE (exception nommée P5) : la même clé ppb_ancr est une « Tige » d'ancrage en
# vérification complète et un « Boulon » en calcul manuel (le moteur manuel dit « boulon 1 »).
LIBELLE_MODE = {MODE_COMPLET: {"ancr": "Tige", "visserie": "Visserie par tige"},
                MODE_MANUEL: {"ancr": "Boulon", "visserie": "Visserie par boulon"}}


def mode_courant(u):
    """Le mode affiché (pour LIBELLE_MODE) : vérification complète ou calcul manuel."""
    return MODE_COMPLET if est_complet(u) else MODE_MANUEL


def libelle(k, u):
    """LIBELLE de la clé dans le mode de ``u`` (fenêtres, chips, panneau Cotes)."""
    return LIBELLE_MODE[mode_courant(u)].get(k) or LIBELLE.get(k, k)


# Les aides (« ? ») — textes de la charte §3 ; une clé absente n'a pas de « ? ».
AIDES = {
    "mode": "Calcul manuel : boulons, éléments et raidisseurs posés librement, épaisseur requise comme à la "
            "main. Complète : grille régulière, Tab. 6.7, EN 1992-4.",
    "perim": "EC3 seul : platine, tiges, poteau, cordons et appui fjd décident ; béton d'ancrage (EN 1992-4) en "
             "information. EC3 + EC2-4 : il entre dans le verdict.",
    "typ_a": "Tête / plaque : hef et dh se règlent sur la coupe A-A. Crosse : adhérence. Scellée : valeurs de "
             "l'ETA (τRk, k1, NRk,s, VRk,s).",
    "a_cc": "Coefficient de l'ANB belge pour fcd = αcc·fck/γc ; 0,85 par défaut, à confirmer.",
    "C_f": "Frottement platine / mortier, déduit du cisaillement des tiges (V EC3).",
    "L_c": "Hauteur du poteau pour la classification de rigidité du pied (EN 1993-1-8 §5.2.2.5), en information.",
    "visserie": "Rondelles, écrou et contre-écrou par tige, portés au cartouche du plan de principe.",
    "ui_composant": "Décoché : dessins fixes ; les cotes se modifient dans le panneau « Cotes » en tête de la "
                    "colonne de droite.",
    # aides de BLOC (titre_bloc(help=) ou sous-titre ↳) : les conventions communes, dites une fois
    "EFFORTS ELU": "N > 0 compression ; Mx > 0 comprime le côté y > 0, My > 0 comprime x > 0 ; T = torsion. "
                   "Appliqués en (xN ; yN), (0 ; 0) = centre de la platine.",
    "ETA": "Valeurs de l'ETA de la tige scellée (état du béton, classe) ; k1 = kcr ou kucr (7,7 / 11,0 à défaut) ; "
           "NRk,s et VRk,s : 0 = calculé par l'outil.",
    # « COEFFICIENTS » (12 lettres) ne tient pas dans la case des titres de 76 px (charte §2 : ≤ 11
    # caractères) : le sous-bloc s'appelle « COEFF. γ », son « ? » dit l'ANB une fois (les deux captions
    # ANB de 163 et 190 caractères de l'ancienne carte disparaissent).
    "COEFF. γ": "γM0 1,00 et γM2 1,25 (ANB belge) ; γc = γMc 1,50 ; αcc 0,85 (fcd = αcc·fck/γc) — hypothèse à "
                "confirmer sur le texte NBN.",
}
# Le calcul manuel : mêmes aides, plus celles de ses choix, de ses tables, et la visserie « par boulon ».
AIDES_MANUEL = dict(AIDES, **{
    "crit": "Élastique : Von Mises ≤ fy/γM0 sans effet de levier — ≈ 1,22 × l'épaisseur plastique (√1,5), "
            "jusqu'à 1,73 (√3) avec levier. Plastique : tronçon en T Mpl.",
    "appui": "Bande c autour des parois (EN 1993-1-8 §6.2.5, sûr) ou toute la platine (rigide : efforts des tiges "
             "comparables aux logiciels d'ancrage).",
    "xN": "Point d'application de N, Mx, My, Vx, Vy ; (0 ; 0) = centre de la platine. Un élément excentré crée le "
          "moment N·e tout seul.",
    "visserie": "Rondelles et écrous par boulon, portés au cartouche du plan. Ex. : « 1 rondelle + 2 écrous "
                "(contre-écrou) », « 2 rondelles + 1 écrou ».",
    # le « ? » de x (mm) de la table des boulons : l'ancienne légende « Repère : origine… »
    "tiges": "Origine au centre de la platine, x vers la droite, y vers le haut (vue en plan). Chaque boulon se "
             "déplace aussi sur le dessin.",
    "ÉLÉMENTS": "Élément soudé = profilé I/H, tube, plat, rond ou rectangle posé en (x ; y). Rotation et dimensions : "
                "dans sa fenêtre sur le dessin (ici si le dessin est fixe).",
    "RAIDISSEURS": "Plat vertical soudé sur la platine et l'élément, parti de (x ; y) dans le sens donné ; h sert "
                   "aux coupes.",
})

# Options courtes à l'AFFICHAGE ; les valeurs brutes (fichiers, tests) ne changent pas (P4).
AFFICHAGE = {"mode": {MODE_MANUEL: "Calcul manuel", MODE_COMPLET: "Vérif. complète"},
             "perim": {PERIM_EC3: "EC3 seul", PERIM_TOUT: "EC3 + EC2-4"},
             "crit": {CRIT_ELAS: "Élastique", CRIT_EC3: "Plastique"},
             "appui": {APPUI_EC3: "Bande c", APPUI_PLEIN: "Toute la platine"},
             "forme": {FORME_I: "I / H", FORME_RHS: "Tube RHS / SHS"},
             "typ_a": {TYPE_TETE: "Tête / plaque", TYPE_CROSSE: "Crosse", TYPE_ETA: "Scellée (ETA)"}}

# Ce qui se modifie sur le dessin — jamais dans la carte (les cotes des DEUX modes ; dh compris : la
# coupe manuelle l'émet, manuel/schemas.CLE_PAR_COTE le porte depuis le 31/10/2026).
GEOMETRIE_DESSIN = frozenset(CLE_PAR_COTE.values()) | frozenset(CLE_PAR_COTE_MANUEL.values())
# L'ordre du panneau Cotes (dessin fixe) : il liste GEOMETRIE_DESSIN ∩ champ_visible, pas seulement les
# cotes que le dessin émet — tg nul, hef sans boulon y restent atteignables (amendement 2, §4.5).
COTES_ORDRE = ("bp_u", "lp_u", "tp_u", "a_w", "t_g", "bc_u", "lc_u", "w_x", "w_y", "hef", "d_h")
assert set(COTES_ORDRE) == GEOMETRIE_DESSIN

# La rangée d'en-tête : la visibilité de chaque choix suit champ_visible (périmètre : complet ;
# critère et appui : manuel).
EN_TETE = [("mode", None), ("perim", None), ("crit", None), ("appui", None)]
SEGMENTS = frozenset({"mode", "perim", "crit", "appui"})

BLOCS = [
    ("POTEAU", [L(["forme", "prof_c", "nu_c"]),
                L(["h_u", "b_u", "tw_u", "tf_u", "r_u", "t_u"], perso_c, "perso")]),
    ("MATÉRIAUX", [L(["nu_pl", "beton", "fiss", "hc_u"])]),
    ("TIGES", [L(["ancr", "classe", "typ_a"]),
               L(["n_ax", "n_ay"], None, "grille"),
               L(["eta_tau", "eta_k1", "eta_nrks", "eta_vrks"], eta, "ETA")]),
    ("EFFORTS ELU", [L(["N_Ed", "Mx_Ed", "My_Ed"]), L(["Vx_Ed", "Vy_Ed", "T_Ed"], None, "")]),
]
# Calcul manuel : les lignes du kit ; la table des boulons, ÉLÉMENTS et RAIDISSEURS sont rendus entre
# BOULONS et EFFORTS ELU par manuel/ecran.py (fiss n'y est pas visible : MATÉRIAUX a trois cases).
BLOCS_MANUEL = [
    ("MATÉRIAUX", [L(["nu_pl", "beton", "fiss", "hc_u"])]),
    ("BOULONS", [L(["ancr", "classe", "typ_a"])]),
    ("EFFORTS ELU", [L(["N_Ed", "Mx_Ed", "My_Ed"]), L(["Vx_Ed", "Vy_Ed", "T_Ed"], None, "")]),
]
# Sans césure : l'espacement serré du kit (ui_commun.espacement, mot ≥ 11 lettres) le tient sur une ligne
# de la case de 76 px.
TITRE_APPLICATION = "APPLICATION"
# Paramètres avancés des deux modes : chaque ligne suit champ_visible (γMc, RIGIDITÉ : complet ;
# APPLICATION : manuel).
AVANCES = [
    ("COEFF. γ", [L(["g_M0", "g_M2", "g_c", "g_Mc"]), L(["a_cc"], None, "")]),
    ("APPUI", [L(["C_f", "f_g"])]),
    ("RIGIDITÉ", [L(["L_c", "contrev"])]),
    # « APPLICATION » (11 lettres) mesurait 76,5 px pour une case de 76 px (constaté à 1440 : « APPLICATIO / N »,
    # puis « APPLI-CATION » à la césure — un titre ne se coupe jamais, charte §2) : espacement serré de
    # 0,05 em, règle du kit depuis la consolidation (ui_commun.espacement : tout mot ≥ 11 lettres).
    (TITRE_APPLICATION, [L(["xN", "yN"])]),
    ("ACIER PERSO", [L(["fy_u", "fu_u", "bw_u"], acier_perso)]),
]


CARTOUCHE = [("", [L(["id_rep", "id_red", "visserie"]), L(["id_projet", "id_date"], None, "hors projet")])]
# Garde-fou des dimensions vides (consolidation, point 19) : un profil Personnalisé ou un tube sans
# dimensions levait ZeroDivisionError (dessin disparu, cases étirées) ; la vérification est suspendue et le
# dessin garde, provisoirement, les dimensions du profil par défaut (épaisseur du tube : celle de son aile).
DIMENSIONS = kit.dimensions_profil(defaults()["prof_c"], "h_u", "b_u", "tw_u", "tf_u")
DIMENSIONS["t_u"] = DIMENSIONS["tf_u"]

D = kit.Description(
    PREFIXE=PREFIXE, CHAMPS=CHAMPS, champ_visible=champ_visible, LIBELLE=LIBELLE, AIDES=AIDES,
    AFFICHAGE=AFFICHAGE, PAS=PAS, BLOCS=BLOCS, AVANCES=AVANCES, PILOTEES_PAR_PREDIM=(),
    mode_predim=lambda u: False, VISSERIE_DEFAUT=VISSERIE_DEFAUT, GEOMETRIE_DESSIN=GEOMETRIE_DESSIN,
    EN_TETE=EN_TETE, SEGMENTS=SEGMENTS, CARTOUCHE=CARTOUCHE, LIBELLE_MODE=LIBELLE_MODE,
    VISIBLE_SUPPL=dict(kit.HORS_PROJET), mode_courant=mode_courant, DIMENSIONS=DIMENSIONS)
D_MANUEL = dataclasses.replace(D, AIDES=AIDES_MANUEL, BLOCS=BLOCS_MANUEL)


def description(u):
    """La Description du mode de ``u``."""
    return D if est_complet(u) else D_MANUEL


def K(k):
    return PREFIXE + k


def valeur_widget(k, v):
    """Lue aussi par interface.py et acier/projet_ui.py (hydratation). Les
    LISTES du calcul manuel (tiges, elems, raid) passent telles quelles."""
    if k not in CHAMPS:
        return v
    return kit.valeur_widget(D, k, v)


# ------------------------------------------------------------ bascule de mode
def changer_mode(lire, ecrire):
    """on_change du choix de mode : la disposition est reprise d'un mode à
    l'autre ; vers la vérification complète seulement si elle est régulière
    (sinon le mode reste manuel et la raison est dite)."""
    from .manuel.conversion import vers_manuel, vers_complet, disposition_reguliere
    u = lire()
    prec = st.session_state.get("ppb_ui_mode_prec")
    nouveau = st.session_state.get(K("mode"))
    if nouveau == prec:
        return
    if nouveau == MODE_COMPLET:
        v = vers_complet(dict(u, mode=MODE_MANUEL))
        if v is None:
            raison = disposition_reguliere(u)["raison"]
            st.session_state[K("mode")] = MODE_MANUEL
            st.session_state["_ppb_toast"] = "Vérification complète impossible : " + raison + "."
            return
        ecrire(v)
        st.session_state["_ppb_toast"] = "Disposition reprise dans la vérification complète"
    else:
        ecrire(vers_manuel(dict(u, mode=MODE_COMPLET)))
        st.session_state["_ppb_toast"] = "Grille et profilé repris dans le calcul manuel"
    st.session_state["ppb_ui_mode_prec"] = nouveau
    st.session_state["ppb_ui_alerte"] = None


def en_tete(u, chauds, lire, ecrire):
    """La rangée d'en-tête des deux cartes : le mode (on_change = la bascule,
    qui reprend la disposition), puis le périmètre (complet) ou le critère et
    l'appui (manuel) — segmented_control du kit, valeurs brutes inchangées."""
    st.session_state["ppb_ui_mode_prec"] = st.session_state.get(K("mode"))
    Dm = dataclasses.replace(description(u), ON_CHANGE={"mode": dict(on_change=changer_mode, args=(lire, ecrire))})
    kit.en_tete(Dm, u, chauds)


def aides_visibles(u, Dx=None):
    """Les « ? » visibles dans la CARTE (``kit.aides_visibles`` : en-tête, lignes, sous-titres « ↳ ETA »,
    titres de bloc ; la carte manuelle ajoute ceux de ses tables : manuel/ecran.aides_visibles)."""
    return kit.aides_visibles(Dx or description(u), u)


def avances(u, chauds=frozenset()):
    """« Paramètres avancés » du mode de ``u`` : sous-blocs titrés sur la grille du kit."""
    kit.avances(description(u), u, chauds)


def cartouche(u, chauds=frozenset()):
    """« Cartouche (note et plan) » : Repère · Rédacteur · Visserie (+ ↳ hors projet)."""
    kit.cartouche(description(u), u, chauds)


def carte(u, chauds=frozenset(), lire=None, ecrire=None):
    """La carte de la VÉRIFICATION COMPLÈTE : en-tête, blocs par objet, puis
    les deux replis (la carte du calcul manuel : manuel/ecran.carte)."""
    with st.container(gap=None):
        if lire is not None:
            en_tete(u, chauds, lire, ecrire)
        for titre, lignes in D.BLOCS:
            kit.bloc(D, titre, lignes, u, chauds, chaud=(titre == "TIGES" and "anchors" in chauds)
                     or (titre == "POTEAU" and "col" in chauds))
    avances(u, chauds)
    cartouche(u, chauds)

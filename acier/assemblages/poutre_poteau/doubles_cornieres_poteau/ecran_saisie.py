# -*- coding: utf-8 -*-
"""La carte de données des doubles cornières sur poteau — les DONNÉES
seulement : le kit (``acier/assemblages/carte.py``) rend la carte sur les
trois formats.

Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md §4.1,
étendue à la famille poutre–poteau) :

    EN-TÊTE     [ Vérification | Prédimensionnement ] ?  [ Aile | Âme ] ?
    PROFILÉS    | Poteau ▾ | Nuance ▾ | Poutre portée ▾ | Nuance ▾ |   ↳ perso
    CORNIÈRES   | Cornière ▾ | Nuance ▾ | Orientation ▾ |   ↳ perso
    BOULONS     | Boulon ▾ | Classe ▾ | Trou ▾ ? | Catégorie ▾ ? |
      ↳ AILES A  | Fixation ailes A ▾ | Rangées n1 | Files n2 ▾ |
      ↳ AILES B ?| Fixation ailes B ▾ | Rangées n1 | Files n2 ▾ |
      ↳ glissement (cat ≠ A) | μ | ks (Tab. 3.6) | ELS / ELU ? |
    EFFORTS ELU ? | VEd | NEd | HEd | MEd |   ↳ nature | Nature de NEd ▾ |
    ▸ Paramètres avancés   ▸ Cartouche (note et plan)

Le CAS D'APPUI est en en-tête, en ``segmented_control`` à côté du mode : c'est
la clé qui commande toute la mécanique (POUTRE_POTEAU.md §10.6).

La géométrie (hauteur et position des cornières, pinces, entraxes,
trusquinage gA, grugeage, jeux, RECOUPE, gorges, retours) ne se modifie que
sur le dessin (P6) — y compris ``rec_S`` et ``j_lat``, qui sont des cotes de
la vue en plan. Une forme par clé : ``LIBELLE`` est lu par la carte, les
fenêtres d'objet du dessin, le panneau Cotes et les chips d'alerte (P5) ; un
« ? » n'existe que pour une entrée de ``AIDES`` (P3).

Clés de session ``dcc_<clé>`` : source unique, écrite AVANT l'instanciation
des widgets (le dessin est rendu avant la carte)."""
from acier.assemblages import carte as kit
from acier.assemblages.carte import L
from acier.bibliotheques import ORI_P, ORI_S
from .entrees import (mode_predim, PILOTEES_PAR_PREDIM, CHAMPS, UNITE, champ_visible, perso_P, perso_S,
                      perso_C, _cat_non_A, _acier_perso, _tying_N, traction, boulonne_P, boulonne_S,
                      VISSERIE_DEFAUT, defaults, MODE_VERIF, MODE_PREDIM, APPUI_AILE, APPUI_AME,
                      NAT_ELU, NAT_TYING)
from .schemas import CLE_PAR_COTE

PREFIXE = "dcc_"

PAS = {"V_Ed": 5.0, "N_Ed": 5.0, "H_Ed": 5.0, "M_Ed": 1.0,
       "mu_s": 0.05, "k_s": 0.05, "k_ser": 0.05, "g_M0": 0.05, "g_M2n": 0.05,
       "g_M2": 0.05, "g_M3": 0.05, "g_M3s": 0.05, "g_Mu": 0.05, "k_rot": 0.05,
       "eta_c": 0.05, "k_e1": 0.1, "k_p1": 0.1, "k_e2": 0.1, "bw_u": 0.05,
       "fy_u": 5.0, "fu_u": 5.0,
       "LC_u": 5.0, "z_C": 5.0, "p1S_u": 5.0, "p2_S": 5.0, "e1S_u": 5.0, "e2b_u": 5.0,
       "p1P_u": 5.0, "p2_P": 5.0, "e1P_u": 5.0, "gA_u": 5.0, "d_nt": 5.0, "d_nb": 5.0,
       "l_n": 5.0, "rec_S": 5.0, "j_lat": 1.0, "lh_P": 5.0, "lh_S": 5.0,
       "k1_u": 5.0, "k2_u": 5.0, "hP_u": 5.0, "bP_u": 5.0, "hS_u": 5.0, "bS_u": 5.0,
       "twP_u": 0.5, "tfP_u": 0.5, "rP_u": 0.5, "twS_u": 0.5, "tfS_u": 0.5, "rS_u": 0.5,
       "kt_u": 0.5, "kr_u": 0.5, "a_P": 0.5, "a_S": 0.5}

# Libellés sans unité (≤ 16 dans une rangée de 4, ≤ 20 ailleurs) ; l'unité vient de UNITE.
_LIBELLE = {
    "mode_calc": "Mode", "appui": "Appui",
    "prof_P": "Poteau", "nu_P": "Nuance", "prof_S": "Poutre portée", "nu_S": "Nuance",
    "hP_u": "h", "bP_u": "b", "twP_u": "tw", "tfP_u": "tf", "rP_u": "r",
    "hS_u": "h", "bS_u": "b", "twS_u": "tw", "tfS_u": "tf", "rS_u": "r",
    "corn_u": "Cornière", "nu_C": "Nuance", "orient": "Grande aile côté",
    "k1_u": "Grande aile", "k2_u": "Petite aile", "kt_u": "Épaisseur", "kr_u": "Congé",
    "boulon_u": "Boulon", "classe": "Classe", "trou": "Trou", "cat": "Catégorie",
    "fix_P": "Fixation ailes A", "n1P_u": "Rangées n1", "n2P_u": "Files n2",
    "fix_S": "Fixation ailes B", "n1S_u": "Rangées n1", "n2S_u": "Files n2",
    "mu_s": "μ", "k_s": "ks (Tab. 3.6)", "k_ser": "ELS / ELU",
    "V_Ed": "VEd", "N_Ed": "NEd", "H_Ed": "HEd", "M_Ed": "MEd", "nat_N": "Nature de NEd",
    "r_n": "Rayon grugeage", "lt_ok": "Déversement empêché",
    "d0_u": "d0 imposé", "filet": "Filetage cisaillé",
    "g_M0": "γM0", "g_M2": "γM2 boulons", "g_M2n": "γM2 nettes",
    "g_M3": "γM3 (ELU)", "g_M3s": "γM3,ser (ELS)", "g_Mu": "γMu (tying)",
    "opt_exc": "Excentricité gA", "k_rot": "k sur Fv,Rd (P)",
    "opt_blf": "βLf joints longs", "expo": "Intempéries",
    "fy_u": "fy", "fu_u": "fu", "bw_u": "βw",
    "eta_c": "Taux cible", "k_e1": "e1 = k·d0", "k_p1": "p1 = k·d0", "k_e2": "e2 = k·d0",
    "pd_dmin": "Ø minimal", "pd_dmax": "Ø maximal",
    "id_projet": "Projet", "id_rep": "Repère", "id_red": "Rédacteur", "id_date": "Date",
    "visserie": "Visserie par boulon",
    # géométrie : fenêtres d'objet, panneau Cotes, chips d'alerte
    "LC_u": "hc", "z_C": "zc", "g_h": "gh", "l_n": "c", "d_nt": "dc,sup", "d_nb": "dc,inf",
    "e1S_u": "e1 groupe S", "p1S_u": "p1 groupe S", "e2b_u": "e2,b", "p2_S": "p2 groupe S",
    "e1P_u": "e1 groupe P", "p1P_u": "p1 groupe P", "gA_u": "gA", "p2_P": "p2 groupe P",
    "a_P": "a ailes A", "lh_P": "ℓh ailes A", "a_S": "a ailes B", "lh_S": "ℓh ailes B",
    "rec_S": "Recoupe rc", "j_lat": "Jeu latéral j",
    # tête de la colonne dessin
    "ui_niveau": "Niveau de cotation", "ui_composant": "Dessin interactif",
}
LIBELLE = {k: v + (" (" + UNITE[k].strip() + ")" if UNITE.get(k) else "") for k, v in _LIBELLE.items()}

# Hors de la carte (pastilles d'alerte, toasts du dessin, panneau Cotes), le groupe est dit :
# « Rangées n1 » seul ne situait pas la clé.
LIBELLE_LONG = {"n1P_u": "Rangées n1 (groupe P)", "n2P_u": "Files n2 (groupe P)",
                "n1S_u": "Rangées n1 (groupe S)", "n2S_u": "Files n2 (groupe S)"}

# Les aides (« ? ») — textes de la charte §3 ; une clé absente n'a pas de « ? ».
AIDES = {
    "mode_calc": "Prédimensionnement : boulons, rangées et pièce d'attache proposés (onglet Prédim) puis "
                 "vérifiés ; les champs ◆ sont alors pilotés par la proposition.",
    "appui": "Aile : les ailes A portent sur la face extérieure d'une aile, de part et d'autre de l'âme "
             "du poteau. Âme : elles portent sur l'âme, entre les ailes.",
    "trou": "Surdimensionné : Fb,Rd × 0,8 (EN 1993-1-8 Tab. 3.4) et d0 à saisir dans Paramètres avancés › TROUS.",
    "cat": "A : pression diamétrale. B : sans glissement à l'ELS. C : sans glissement à l'ELU (boulons "
           "précontraints 8.8 ou 10.9).",
    "k_ser": "Rapport des efforts ELS / ELU ; 1,0 = hypothèse sécuritaire.",
    "nat_N": "ELU : fy/γM0 et γM2, combiné avec VEd. Tying : force de chaînage seule, fu/γMu (MSB P5 §2.3).",
    "r_n": "Information d'exécution : non utilisé dans les résistances ; dessiné sur l'élévation.",
    "lt_ok": "Condition de la règle de stabilité locale du grugeage (MSB P5 §4.2.5) ; facultative sur un "
             "poteau, qui n'oblige pas à gruger.",
    "d0_u": "Vide = d0 normal du boulon (Tab. 3.3) ; à saisir pour un trou surdimensionné.",
    "g_Mu": "1,1 : valeur des guides (MSB P5 §2.3, SCI P358) ; l'EN 1993-1-8 ne la fixe pas — à confirmer "
            "pour la Belgique.",
    "opt_exc": "Non : pratique ECCS n°126 / MSB P5. Oui : variante conservatrice, M = (VEd/2)·eP par cornière.",
    "k_rot": "0,8 : MSB P5 §4.2.1.2 / SCI P358 ; 1,0 pour désactiver.",
    "opt_blf": "βLf des assemblages longs, EN 1993-1-8 §3.8 (Lj > 15d). Les guides ne l'appliquent pas aux "
               "attaches d'âme ; Oui = variante conservatrice.",
    "expo": "Assemblage exposé aux intempéries : active les pinces maximales 4t + 40 mm (Tab. 3.3).",
    "visserie": "Rondelles et écrous par boulon, portés au cartouche du plan. Ex. : « 1 rondelle + 2 écrous "
                "(contre-écrou) », « 2 rondelles + 1 écrou ».",
    "ui_composant": "Décoché : dessins fixes ; les cotes se modifient dans le panneau « Cotes » en tête de la "
                    "colonne de droite.",
    # aides portées par les fenêtres d'objet et le panneau Cotes (géométrie)
    "LC_u": "Recommandation des guides : hc ≥ 0,6·h de la poutre portée (maintien en torsion).",
    "d_nt": "0 = semelle supérieure non grugée ; un poteau n'oblige pas à gruger.",
    "d_nb": "0 = semelle inférieure non grugée.",
    "g_h": "Jeu entre la face d'appui du poteau et l'about de la poutre ; aussi par glissement de la poutre.",
    "gA_u": "Trusquinage : talon de la cornière → 1re file. Il donne le gauge p3 = tw,S + 2·gA, qui doit "
            "dégager le congé du poteau : p3 ≥ twP + 2·rP + dw.",
    "a_P": "Cordon vertical en bout d'aile A, sur toute la hauteur hc, contre le poteau.",
    "a_S": "Cordon vertical en bout d'aile B, sur toute la hauteur hc, contre l'âme portée.",
    "rec_S": "Cas (b) : longueur recoupée des semelles depuis l'about, pour entrer entre les ailes du "
             "poteau. Minimum (bP − twP)/2 + j − gh. 0 = aucune.",
    "j_lat": "Cas (b) : jeu de montage entre la semelle de la poutre et l'aile du poteau ; il commande le "
             "passage hP − 2·tfP − 2·j.",
    # aides de BLOC (titre_bloc(help=)) et de sous-titre « ↳ » : dites une fois
    "EFFORTS ELU": "Efforts dans la poutre portée : NEd > 0 traction (sa nature, ELU ou tying, apparaît "
                   "alors) ; HEd perpendiculaire à l'âme ; MEd à la face d'appui (articulé : 0).",
    "AILES B": "Ailes B sur l'âme de la poutre portée (groupe S, deux plans). Soudées des deux côtés : "
               "assemblage peu déformable, à réserver à des cas particuliers.",
    # « COEFFICIENTS » (12 lettres) ne tient pas dans la case des titres de 76 px (charte §2) :
    # le sous-bloc s'appelle « COEFF. γ », son « ? » dit l'ANB une fois.
    "COEFF. γ": "ANB belge : γM0 1,00 (NBN EN 1993-1-1 ANB). γM2 boulons : boulons, pression diamétrale, "
                "soudures ; γM2 nettes : sections nettes, bloc ; γMu : tying seulement.",
    "PRÉDIM": "Pinces et entraxes proposés en multiples de d0 ; minima normatifs e ≥ 1,2·d0, p1 ≥ 2,2·d0 "
              "(Tab. 3.3). Règles de détail modifiables.",
}

# Options courtes à l'AFFICHAGE ; les valeurs brutes (fichiers, tests) ne changent pas (P4).
AFFICHAGE = {"mode_calc": {MODE_VERIF: "Vérification", MODE_PREDIM: "Prédimensionnement"},
             "appui": {APPUI_AILE: "Aile", APPUI_AME: "Âme"},
             "orient": {ORI_P: "Poteau", ORI_S: "Poutre portée"},
             "nat_N": {NAT_ELU: "ELU", NAT_TYING: "Tying"}}

# Ce qui se modifie sur le dessin — jamais dans la carte (CLE_PAR_COTE : 20 cotes)
GEOMETRIE_DESSIN = frozenset(CLE_PAR_COTE.values())

EN_TETE = [("mode_calc", None), ("appui", None), ("eta_c", mode_predim),
           ("pd_dmin", mode_predim), ("pd_dmax", mode_predim)]
SEGMENTS = frozenset({"mode_calc", "appui"})

# Sous-titres des continuations du bloc BOULONS : une fixation gouverne un groupe de boulons
SUITE_A, SUITE_B = "AILES A (groupe P)", "AILES B (groupe S)"

BLOCS = [
    ("PROFILÉS", [L(["prof_P", "nu_P", "prof_S", "nu_S"]),
                  L(["hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"], perso_P, "poteau"),
                  L(["hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"], perso_S, "portée")]),
    # orient n'est rendu que pour une cornière inégale ou Personnalisé (entrees.corniere_inegale)
    ("CORNIÈRES", [L(["corn_u", "nu_C", "orient"]),
                   L(["k1_u", "k2_u", "kt_u", "kr_u"], perso_C, "perso")]),
    # n1 / n2 d'un groupe ne sont rendus que si sa fixation est boulonnée (entrees.GROUPES)
    ("BOULONS", [L(["boulon_u", "classe", "trou", "cat"]),
                 L(["fix_P", "n1P_u", "n2P_u"], None, SUITE_A),
                 L(["fix_S", "n1S_u", "n2S_u"], None, SUITE_B),
                 L(["mu_s", "k_s", "k_ser"], _cat_non_A, "glissement")]),
    ("EFFORTS ELU", [L(["V_Ed", "N_Ed", "H_Ed", "M_Ed"]),
                     L(["nat_N"], traction, "nature")]),
]
AVANCES = [
    ("GRUGEAGE", [L(["r_n", "lt_ok"])]),
    ("TROUS", [L(["d0_u", "filet"])]),
    ("COEFF. γ", [L(["g_M0", "g_M2", "g_M2n"]),
                  L(["g_M3", "g_M3s"], _cat_non_A, "glissement"),
                  L(["g_Mu"], _tying_N, "tying")]),
    ("MODÈLE", [L(["opt_exc", "k_rot", "opt_blf", "expo"])]),
    ("ACIER PERSO", [L(["fy_u", "fu_u", "bw_u"], _acier_perso)]),
    ("PRÉDIM", [L(["k_e1", "k_p1", "k_e2"], mode_predim)]),
]


def _avec_boulons(u):
    # la visserie n'a de sens que s'il y a des boulons
    return boulonne_P(u) or boulonne_S(u)


CARTOUCHE = [("", [L(["id_rep", "id_red", "visserie"]), L(["id_projet", "id_date"], None, "hors projet")])]

# Garde-fou des dimensions vides : un profil ou une cornière Personnalisé sans
# dimensions suspend la vérification ; le dessin garde celles du défaut.
DIMENSIONS = dict(kit.dimensions_profil(defaults()["prof_P"], "hP_u", "bP_u", "twP_u", "tfP_u"),
                  **kit.dimensions_profil(defaults()["prof_S"], "hS_u", "bS_u", "twS_u", "tfS_u"))
DIMENSIONS.update(kit.dimensions_profil(defaults()["corn_u"], "k1_u", "k2_u", "kt_u", liste="cornieres",
                                        champs=("a1", "a2", "t")))


D = kit.Description(
    PREFIXE=PREFIXE, CHAMPS=CHAMPS, champ_visible=champ_visible, LIBELLE=LIBELLE, AIDES=AIDES,
    AFFICHAGE=AFFICHAGE, PAS=PAS, BLOCS=BLOCS, AVANCES=AVANCES, PILOTEES_PAR_PREDIM=PILOTEES_PAR_PREDIM,
    mode_predim=mode_predim, VISSERIE_DEFAUT=VISSERIE_DEFAUT, GEOMETRIE_DESSIN=GEOMETRIE_DESSIN,
    EN_TETE=EN_TETE, SEGMENTS=SEGMENTS, CARTOUCHE=CARTOUCHE,
    VISIBLE_SUPPL=dict(kit.HORS_PROJET, visserie=_avec_boulons), DIMENSIONS=DIMENSIONS,
    LIBELLE_LONG=LIBELLE_LONG)


def K(k):
    return PREFIXE + k


def valeur_widget(k, v):
    """Lue aussi par interface.py et acier/projet_ui.py (hydratation)."""
    return kit.valeur_widget(D, k, v)


def aides_visibles(u):
    """Les « ? » visibles dans la CARTE (le kit compte ceux des sous-titres « ↳ »)."""
    return kit.aides_visibles(D, u)


def carte(u, chauds=frozenset()):
    """La carte complète : en-tête, blocs par objet, puis les deux replis."""
    kit.carte(D, u, chauds)

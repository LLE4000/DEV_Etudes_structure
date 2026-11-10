# -*- coding: utf-8 -*-
"""La RÉSISTANCE AU FEU du béton armé par les VALEURS TABULÉES — EN 1992-1-2:2004, section 5 (feu normalisé ISO 834,
30 à 240 minutes, béton de granulats siliceux de densité normale, jusqu'à C50/60).

Pour un élément, une durée et ses paramètres, ``exigence`` rend ce que le tableau de la norme demande : les
combinaisons (dimension minimale / distance à l'axe ``a``) de la durée, celle que la dimension réelle autorise, la
dimension minimale, les notes du tableau, ses conditions d'emploi, sa source et l'état de sa vérification. ``tableau``
rend le tableau entier pour l'écran. C'est le SEUL endroit du dépôt où ces valeurs sont écrites : la rubrique
« Enrobages » de la bibliothèque (``beton/bibliotheque/enrobage.py``) et l'ancien module « Enrobage »
(``beton/enrobage/donnees.py``) les lisent ici.

Tableaux portés (chacun dans UNE table nommée, ci-dessous) :

    poteaux                      tableau 5.2a   §5.3.2   méthode A : μ_fi = 0,2 / 0,5 / 0,7, exposé sur plusieurs côtés ou un seul
                                 éq. (5.7)      §5.3.2(4) la formule de la méthode A, AU CHOIX (jamais par défaut)
    voiles non porteurs          tableau 5.3    §5.4.1   épaisseur seule (critères E et I)
    voiles porteurs              tableau 5.4    §5.4.2   μ_fi = 0,35 / 0,7, une ou deux faces exposées
    poutres sur appuis simples   tableau 5.5    §5.6.2   quatre combinaisons b_min / a ; a_sd = a + 10 mm
    poutres continues            tableau 5.6    §5.6.3
    dalles pleines               tableau 5.8    §5.7.2   une direction, deux directions ; continues : §5.7.3
    planchers-dalles             tableau 5.9    §5.7.4

Règles de lecture (dites dans le résultat, jamais appliquées en silence) :

- RIEN N'EST INTERPOLÉ entre deux lignes ou deux colonnes (la norme le permet, §5.2 ; ne pas interpoler est du côté
  sûr) : la dimension réelle choisit la combinaison la plus favorable qu'elle autorise — la plus grande dimension
  minimale qui ne la dépasse pas ; sans dimension, la première combinaison (la plus petite section, le plus grand a) ;
- une dimension réelle sous le minimum du tableau, ou une case vide du tableau (poteau R 240 à μ_fi = 0,7) : la durée
  est « NON ATTEINTE par la méthode tabulée » — aucune valeur n'est inventée ;
- ``a`` est une valeur NOMINALE (§5.2) : aucune tolérance d'exécution ne s'y ajoute ;
- poutres à un seul lit : a_sd = a + 10 mm pour les barres d'angle, mesuré depuis la face latérale, jusqu'à la largeur
  de la colonne 4 du tableau 5.5 (la troisième combinaison) et de la colonne 3 du tableau 5.6 (la deuxième), bornes
  comprises ; au-delà, aucune majoration (§5.6.1(8) ; ``RANG_SANS_A_SD``) ;
- un paramètre hors du domaine d'un tableau ou de la formule est REFUSÉ (``ValueError``, le champ nommé).

La formule (5.7) des poteaux (méthode A) ::

    R = 120·[(R_ηfi + R_a + R_l + R_b + R_n)/120]^1,8
    R_ηfi = 83·[1,00 − μ_fi·(1 + ω)/((0,85/α_cc) + ω)]      R_a = 1,60·(a − 30)      R_l = 9,60·(5 − l_0,fi)
    R_b = 0,09·b′     b′ = 2·A_c/(b + h) (le diamètre d'un poteau rond)     R_n = 0 (4 barres) ou 12 (plus de 4)

Avec α_cc = 0,85 (``ALPHA_CC`` : l'annexe belge de l'EN 1992-1-1), (0,85/α_cc) vaut 1 et le terme en ω se simplifie :
R_ηfi = 83·(1 − μ_fi), quel que soit ω. Le tableau 5.2a de l'EN, lui, est établi avec α_cc = 1,0 : il est plus sévère
(R 90, μ_fi = 0,7, 350 mm : 53 mm contre 47). L'annexe belge (NBN EN 1992-1-2 ANB) remplace le tableau 5.2a par deux
tableaux 5.2.a-ANB que cette formule reproduit — le texte de l'ANB n'a pas pu être lu : le tableau de l'EN reste la
lecture PAR DÉFAUT, et la formule dit d'où elle vient (``MENTION_FORMULE``).

Bibliothèque standard seule.
"""
import math

NORME = "EN 1992-1-2:2004"
SECTION = "section 5 (valeurs tabulées)"
R_DUREES = ("R30", "R60", "R90", "R120", "R180", "R240")
MINUTES = {r: int(r[1:]) for r in R_DUREES}
ELEMENTS = ("poutre", "poteau", "dalle", "voile")

# ---------------------------------------------------------------------------------------------- les tableaux (mm)
# Une combinaison : (dimension minimale, distance à l'axe a, repère). Repères de la norme :
#   « ** » au moins 8 barres (tableau 5.2a) ; « * » l'enrobage exigé par l'EN 1992-1-1 gouverne normalement.

# POTEAUX — EN 1992-1-2:2004, tableau 5.2a (§5.3.2, méthode A) : b_min / a, par durée et par colonne
# (μ_fi = 0,2 / 0,5 / 0,7 : poteau exposé sur plus d'un côté ; « un_cote » : exposé sur un seul côté, μ_fi = 0,7).
POTEAU_5_2A = {
    "R30": {"mu02": ((200, 25, ""),), "mu05": ((200, 25, ""),), "mu07": ((200, 32, ""), (300, 27, "")), "un_cote": ((155, 25, ""),)},
    "R60": {"mu02": ((200, 25, ""),), "mu05": ((200, 36, ""), (300, 31, "")), "mu07": ((250, 46, ""), (350, 40, "")), "un_cote": ((155, 25, ""),)},
    "R90": {"mu02": ((200, 31, ""), (300, 25, "")), "mu05": ((300, 45, ""), (400, 38, "")), "mu07": ((350, 53, ""), (450, 40, "**")),
            "un_cote": ((155, 25, ""),)},
    "R120": {"mu02": ((250, 40, ""), (350, 35, "")), "mu05": ((350, 45, "**"), (450, 40, "**")), "mu07": ((350, 57, "**"), (450, 51, "**")),
             "un_cote": ((175, 35, ""),)},
    "R180": {"mu02": ((350, 45, "**"),), "mu05": ((350, 63, "**"),), "mu07": ((450, 70, "**"),), "un_cote": ((230, 55, ""),)},
    "R240": {"mu02": ((350, 61, "**"),), "mu05": ((450, 75, "**"),), "mu07": (), "un_cote": ((295, 70, ""),)},
}
# VOILES NON PORTEURS (cloisons) — EN 1992-1-2:2004, tableau 5.3 (§5.4.1) : épaisseur minimale, sans distance à l'axe
VOILE_5_3 = {"R30": 60, "R60": 80, "R90": 100, "R120": 120, "R180": 150, "R240": 175}
# VOILES PORTEURS — EN 1992-1-2:2004, tableau 5.4 (§5.4.2) : épaisseur / a, par durée et par colonne
# (μ_fi = 0,35 ou 0,7 ; « _1 » : une face exposée, « _2 » : deux faces exposées).
VOILE_5_4 = {
    "R30": {"mu035_1": (100, 10, "*"), "mu035_2": (120, 10, "*"), "mu07_1": (120, 10, "*"), "mu07_2": (120, 10, "*")},
    "R60": {"mu035_1": (110, 10, "*"), "mu035_2": (120, 10, "*"), "mu07_1": (130, 10, "*"), "mu07_2": (140, 10, "*")},
    "R90": {"mu035_1": (120, 20, "*"), "mu035_2": (140, 10, "*"), "mu07_1": (140, 25, ""), "mu07_2": (170, 25, "")},
    "R120": {"mu035_1": (150, 25, ""), "mu035_2": (160, 25, ""), "mu07_1": (160, 35, ""), "mu07_2": (220, 35, "")},
    "R180": {"mu035_1": (180, 40, ""), "mu035_2": (200, 45, ""), "mu07_1": (210, 50, ""), "mu07_2": (270, 55, "")},
    "R240": {"mu035_1": (230, 55, ""), "mu035_2": (250, 55, ""), "mu07_1": (270, 60, ""), "mu07_2": (350, 60, "")},
}
# POUTRES SUR APPUIS SIMPLES — EN 1992-1-2:2004, tableau 5.5 (§5.6.2) : les combinaisons possibles b_min / a
POUTRE_5_5 = {
    "R30": ((80, 25, ""), (120, 20, ""), (160, 15, "*"), (200, 15, "*")),
    "R60": ((120, 40, ""), (160, 35, ""), (200, 30, ""), (300, 25, "")),
    "R90": ((150, 55, ""), (200, 45, ""), (300, 40, ""), (400, 35, "")),
    "R120": ((200, 65, ""), (240, 60, ""), (300, 55, ""), (500, 50, "")),
    "R180": ((240, 80, ""), (300, 70, ""), (400, 65, ""), (600, 60, "")),
    "R240": ((280, 90, ""), (350, 80, ""), (500, 75, ""), (700, 70, "")),
}
# POUTRES CONTINUES — EN 1992-1-2:2004, tableau 5.6 (§5.6.3) : les combinaisons possibles b_min / a
POUTRE_5_6 = {
    "R30": ((80, 15, "*"), (160, 12, "*")),
    "R60": ((120, 25, ""), (200, 12, "*")),
    "R90": ((150, 35, ""), (250, 25, "")),
    "R120": ((200, 45, ""), (300, 35, ""), (450, 35, ""), (500, 30, "")),
    "R180": ((240, 60, ""), (400, 50, ""), (550, 50, ""), (600, 40, "")),
    "R240": ((280, 75, ""), (500, 60, ""), (650, 60, ""), (700, 50, "")),
}
# DALLES PLEINES SUR APPUIS SIMPLES — EN 1992-1-2:2004, tableau 5.8 (§5.7.2) : épaisseur h_s, puis a selon la portée
# (une direction ; deux directions avec l_y/l_x ≤ 1,5 ; deux directions avec 1,5 < l_y/l_x ≤ 2) — (a, repère)
DALLE_5_8 = {
    "R30": {"hs": 60, "une": (10, "*"), "deux_15": (10, "*"), "deux_20": (10, "*")},
    "R60": {"hs": 80, "une": (20, ""), "deux_15": (10, "*"), "deux_20": (15, "*")},
    "R90": {"hs": 100, "une": (30, ""), "deux_15": (15, "*"), "deux_20": (20, "")},
    "R120": {"hs": 120, "une": (40, ""), "deux_15": (20, ""), "deux_20": (25, "")},
    "R180": {"hs": 150, "une": (55, ""), "deux_15": (30, ""), "deux_20": (40, "")},
    "R240": {"hs": 175, "une": (65, ""), "deux_15": (40, ""), "deux_20": (50, "")},
}
# PLANCHERS-DALLES — EN 1992-1-2:2004, tableau 5.9 (§5.7.4) : épaisseur h_s / a
DALLE_5_9 = {"R30": (150, 10, "*"), "R60": (180, 15, "*"), "R90": (200, 25, ""), "R120": (200, 35, ""), "R180": (200, 45, ""),
             "R240": (200, 50, "")}

# ---------------------------------------------------------------------------------------------- les règles nommées
A_SD_ANGLE = 10          # mm : a_sd = a + 10 pour les barres d'angle d'une poutre à un seul lit (§5.6.1(8), note des tableaux 5.5 et 5.6)
# JUSQU'À quelle largeur a_sd est majoré : celle de la combinaison de ce RANG dans la ligne du tableau, borne COMPRISE ;
# pour une largeur plus grande, aucune majoration. EN 1992-1-2 §5.6.1(8) et la note sous les tableaux : « jusqu'à la
# largeur de la colonne 4 du tableau 5.5 et de la colonne 3 du tableau 5.6 » — les colonnes sont numérotées avec celle
# des durées : c'est la TROISIÈME combinaison du tableau 5.5 (rang 2) et la DEUXIÈME du tableau 5.6 (rang 1).
# (Décision de l'intégrateur du 10/11/2026 : la lecture de la norme ; la conception écrivait « la dernière colonne ».)
RANG_SANS_A_SD = {"5.5": 2, "5.6": 1}
COLONNE_A_SD = {"5.5": "colonne 4 du tableau 5.5", "5.6": "colonne 3 du tableau 5.6"}     # le nom que la norme donne à cette colonne
A_PEAU = 70              # mm : à partir de cette distance à l'axe, une armature de peau est à prévoir (§4.5.2(2))
ALPHA_CC = 0.85          # α_cc de l'annexe belge de l'EN 1992-1-1 (NBN EN 1992-1-1 ANB) : la formule (5.7) le lit ici, et ici seulement
# le domaine de la formule (5.7), §5.3.2(4)
A_FORMULE = (25.0, 80.0)         # mm
B_PRIME = (200.0, 450.0)         # mm
H_SUR_B_MAX = 1.5
L0_FI = (2.0, 6.0)               # m ; sous 2 m, la valeur à 2 m (côté sûr, dit par la norme)
MU_FORMULE = (0.2, 0.7)          # sous 0,2, la valeur à 0,2 (côté sûr)
L0_FI_DEFAUT = 3.0               # m : la borne du tableau 5.2a
MU_POTEAU = {0.2: "mu02", 0.5: "mu05", 0.7: "mu07"}
MU_VOILE = (0.35, 0.7)
APPUIS = (("simples", "sur appuis simples", "5.5"), ("continue", "continue", "5.6"))
PORTEES = (("une", "une direction"), ("deux_15", "deux directions, l_y/l_x ≤ 1,5"), ("deux_20", "deux directions, 1,5 < l_y/l_x ≤ 2"),
           ("continue", "dalle continue"), ("plancher_dalle", "plancher-dalle"))
# §5.7.3(1) : pour une dalle continue (une ou deux directions), les valeurs des COLONNES 2 ET 4 du tableau 5.8 —
# l'épaisseur h_s et la distance a de la colonne « deux directions, l_y/l_x ≤ 1,5 »
COLONNE_DALLE_CONTINUE = "deux_15"

NOTE_8_BARRES = "** au moins 8 barres longitudinales"
NOTE_ETOILE = "* l'enrobage exigé par l'EN 1992-1-1 gouverne normalement"
NOTE_NOMINAL = "a est une valeur nominale (§5.2) : aucune tolérance d'exécution ne s'y ajoute"
NOTE_INTERPOLATION = "rien n'est interpolé entre deux combinaisons (la norme le permet ; ne pas interpoler est du côté sûr)"
NOTE_A_SD = ("a_sd = a + 10 mm : distance à l'axe des barres d'angle depuis la face latérale, poutre à un seul lit, jusqu'à la largeur de "
             "la colonne 4 du tableau 5.5 (colonne 3 du tableau 5.6) ; au-delà, aucune majoration (§5.6.1(8))")
MENTION_FORMULE = ("Formule (5.7) avec α_cc = 0,85 : ce que donnent les tableaux 5.2.a-ANB de l'annexe belge — à contrôler sur le "
                   "texte de l'ANB:2021 ; le tableau belge exclut certaines colonnes de 200 mm.")
MENTION_PAS_DEFAUT = "Le tableau 5.2a de l'EN (α_cc = 1,0) reste la lecture par défaut : il est le plus sévère."
NON_COUVERT = (
    "méthode B des poteaux (tableau 5.2b) et annexe C",
    "poteaux hors du domaine de la méthode A : l_0,fi > 3 m (tableau) ou > 6 m (formule), excentricité au-delà de e_max = 0,15·h — la valeur belge n'a pas pu être relue",
    "sections en I : largeur d'âme b_w (la Belgique retient la classe WC)",
    "précontrainte : a + 10 mm pour les barres, + 15 mm pour les fils et torons",
    "plusieurs lits : distance moyenne a_m (éq. 5.5), à la charge de l'ingénieur",
    "bétons à haute résistance (au-delà de C50/60)",
    "granulats calcaires : la réduction de 10 % des dimensions n'est pas appliquée (côté sûr)",
    "murs coupe-feu (critère M) ; méthodes de calcul simplifiées et avancées",
)

# L'ÉTAT DE VÉRIFICATION de chaque table, tel que les deux rapports de sources du lot l'établissent (10/11/2026 ; le
# second, terminé, ne contredit AUCUNE case : les 96 cases des tableaux 5.4, 5.5, 5.6, 5.8 et 5.9 sont retrouvées dans
# au moins une source publiée ; dix d'entre elles — R 180 des 5.5 et 5.6, REI 30 et REI 180 du 5.9 — seulement dans du
# code publié : elles se relisent sur la norme papier) ;
# l'écran le montre sous le tableau. Il commence toujours par l'une des quatre formules : « recoupé sur plusieurs
# sources », « recoupé sur une source », « partiellement recoupé : … », « transcrit, non recoupé ».
VERIFICATIONS = {
    "5.2a": "recoupé sur plusieurs sources",
    "5.3": "recoupé sur une source (une seconde, de code publié, concorde)",
    "5.4": "recoupé sur plusieurs sources",
    "5.5": "partiellement recoupé : R 30 à R 120 et R 240 sur plusieurs sources ; R 180 sur du code publié seulement (trois sources concordantes)",
    "5.6": "partiellement recoupé : R 30 à R 120 et R 240 sur plusieurs sources ; R 180 sur du code publié seulement (deux sources concordantes)",
    "5.8": "recoupé sur plusieurs sources",
    "5.9": "partiellement recoupé : REI 60, REI 90, REI 120 et REI 240 sur plusieurs sources ; REI 30 et REI 180 sur une seule source de code publié",
    "5.7": "partiellement recoupé : les coefficients redonnent 99 cases sur 102 d'un extrait du tableau 5.2.a.2-ANB",
}

# les conditions d'emploi du tableau 5.8 selon la portée de la dalle (le tableau entier les porte toutes)
CONDITIONS_DALLE = {
    "une": (),
    "deux": ("deux directions : dalle appuyée sur ses quatre bords — sinon la traiter comme portant dans une direction",),
    "continue": ("dalle continue : redistribution des moments à froid ≤ 15 % — au-delà, chaque travée se traite sur appuis simples",
                 "dalle continue : armatures supérieures A_s ≥ 0,005·A_c sur les appuis intermédiaires dans les cas du §5.7.3(3)"),
}

# la description de chaque tableau : son titre, sa clause, le critère de la durée (R, EI, REI), la dimension qu'il
# borne, ses notes et ses conditions d'emploi
TABLEAUX = {
    "5.2a": {"titre": "Poteaux rectangulaires ou circulaires, méthode A", "clause": "§5.3.2", "critere": "R",
             "dimension": ("b", "plus petit côté ou diamètre"),
             "notes": (NOTE_8_BARRES, "tableau établi avec α_cc = 1,0 (valeur recommandée de l'EN 1992-1-1)", NOTE_NOMINAL, NOTE_INTERPOLATION),
             "conditions": ("ossature contreventée, poteau principalement comprimé",
                            "longueur efficace au feu l_0,fi ≤ 3 m",
                            "excentricité du premier ordre au feu e = M_0Ed,fi / N_0Ed,fi ≤ e_max = 0,15·h (ou b) — valeur recommandée ; "
                            "celle de l'annexe belge n'a pas pu être relue",
                            "armatures A_s < 0,04·A_c",
                            "μ_fi = N_Ed,fi / N_Rd (éq. 5.6), N_Rd calculé à froid")},
    "5.3": {"titre": "Voiles non porteurs (cloisons)", "clause": "§5.4.1", "critere": "EI", "dimension": ("e", "épaisseur"),
            "notes": ("critères E et I : l'épaisseur seule, aucune distance à l'axe n'est exigée", NOTE_INTERPOLATION),
            "conditions": ("hauteur libre du voile / épaisseur ≤ 40",)},
    "5.4": {"titre": "Voiles porteurs en béton armé", "clause": "§5.4.2", "critere": "REI", "dimension": ("e", "épaisseur"),
            "notes": (NOTE_ETOILE, NOTE_NOMINAL, NOTE_INTERPOLATION),
            "conditions": ("hauteur libre du voile / épaisseur ≤ 40",
                           "μ_fi = N_Ed,fi / N_Rd (défini comme pour les poteaux, §5.3.2(3))")},
    "5.5": {"titre": "Poutres sur appuis simples", "clause": "§5.6.2", "critere": "R", "dimension": ("b", "largeur"),
            "notes": (NOTE_A_SD, NOTE_ETOILE, NOTE_NOMINAL, NOTE_INTERPOLATION),
            "conditions": ("poutre exposée au feu sur trois côtés (la face supérieure protégée par la dalle)",
                           "armatures en un seul lit (plusieurs lits : distance moyenne a_m, non couverte)")},
    "5.6": {"titre": "Poutres continues", "clause": "§5.6.3", "critere": "R", "dimension": ("b", "largeur"),
            "notes": (NOTE_A_SD, NOTE_ETOILE, NOTE_NOMINAL, NOTE_INTERPOLATION),
            "conditions": ("redistribution des moments à froid ≤ 15 % — au-delà, chaque travée se traite sur appuis simples (tableau 5.5)",
                           "poutre exposée au feu sur trois côtés ; armatures en un seul lit",
                           "R 90 et plus : armatures supérieures prolongées de 0,3·l_eff de part et d'autre de chaque appui intermédiaire (§5.6.3)")},
    "5.8": {"titre": "Dalles pleines sur appuis simples, portant dans une ou deux directions", "clause": "§5.7.2", "critere": "REI",
            "dimension": ("h_s", "épaisseur"),
            "notes": (NOTE_ETOILE, "h_s : épaisseur de la dalle, revêtement de sol incombustible compris (critères E et I)",
                      "dalle continue (§5.7.3) : h_s et la distance a de la colonne « deux directions, l_y/l_x ≤ 1,5 »",
                      NOTE_NOMINAL, NOTE_INTERPOLATION),
            "conditions": CONDITIONS_DALLE["deux"] + CONDITIONS_DALLE["continue"]},
    "5.9": {"titre": "Planchers-dalles", "clause": "§5.7.4", "critere": "REI", "dimension": ("h_s", "épaisseur"),
            "notes": (NOTE_ETOILE, "a : distance à l'axe des armatures du lit inférieur", NOTE_NOMINAL, NOTE_INTERPOLATION),
            "conditions": ("redistribution des moments à froid ≤ 15 % — au-delà, a se lit dans la colonne « une direction » du tableau 5.8",
                           "REI 90 et plus : au moins 20 % des armatures supérieures sur appuis filantes sur toute la travée, dans les bandes sur appuis",
                           "l'épaisseur minimale ne se réduit pas en comptant le revêtement de sol")},
    "5.7": {"titre": "Poteaux, formule de la méthode A", "clause": "§5.3.2(4), éq. (5.7)", "critere": "R",
            "dimension": ("b′", "largeur équivalente 2·A_c/(b + h)"),
            "notes": ("α_cc = 0,85 (annexe belge de l'EN 1992-1-1) : (0,85/α_cc) = 1, le terme en ω se simplifie et R_ηfi = 83·(1 − μ_fi)",
                      "a exigé arrondi au millimètre supérieur, jamais sous 25 mm", NOTE_NOMINAL),
            "conditions": ("ossature contreventée, poteau exposé sur plus d'un côté",
                           "domaine : 25 ≤ a ≤ 80 mm ; 200 ≤ b′ ≤ 450 mm ; h ≤ 1,5·b ; 2 ≤ l_0,fi ≤ 6 m ; μ_fi ≤ 0,7",
                           "excentricité du premier ordre au feu e ≤ e_max = 0,15·h (ou b) — valeur recommandée ; celle de l'annexe belge n'a pas pu être relue",
                           "armatures A_s < 0,04·A_c")},
}


def source(numero: str) -> str:
    """La source d'un tableau, telle que l'écran l'écrit : la norme, l'édition, le tableau, la clause."""
    t = TABLEAUX[numero]
    quoi = "formule (5.7)" if numero == "5.7" else f"tableau {numero}"
    return f"{NORME}, {quoi} ({t['clause'].split(',')[0]}) — {t['titre'][0].lower()}{t['titre'][1:]}"


# ---------------------------------------------------------------------------------------------- la lecture stricte
def _nombre(x, nom: str):
    """Un nombre fini, écrit avec un point ou une virgule ; None reste None ; tout le reste est refusé."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    if isinstance(x, bool):
        raise ValueError(f"{nom} : un nombre est attendu")
    try:
        v = float(str(x).replace(",", ".").replace(" ", "").replace(" ", ""))
    except ValueError:
        raise ValueError(f"{nom} : nombre illisible « {str(x)[:40]} »") from None
    if not math.isfinite(v):
        raise ValueError(f"{nom} : nombre illisible « {str(x)[:40]} »")
    return v


def _dimension(x, nom: str, mini: float = 20.0, maxi: float = 5000.0):
    v = _nombre(x, nom)
    if v is None:
        return None
    if not mini <= v <= maxi:
        raise ValueError(f"{nom} : entre {_fr(mini)} et {_fr(maxi)} mm (reçu {_fr(v)})")
    return v


def _fr(x) -> str:
    """Un nombre dans un message : sans zéro inutile, virgule décimale."""
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def _parmi(x, permis, nom: str):
    if isinstance(x, bool) or x not in permis:
        raise ValueError(f"{nom} : « {str(x)[:40]} » n'est pas proposé ({', '.join(str(p) for p in permis)})")
    return x


def _mu(x, permis, nom: str) -> float:
    v = _nombre(x, nom)
    for p in permis:
        if v is not None and abs(v - p) < 1e-9:
            return p
    raise ValueError(f"{nom} : {' ou '.join(_fr(p) for p in permis)} — les colonnes du tableau, rien n'est interpolé (reçu « {str(x)[:40]} »)")


def _booleen(x, nom: str) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str) and x.strip().lower() in ("1", "true", "oui", "on", "vrai"):
        return True
    if isinstance(x, str) and x.strip().lower() in ("0", "false", "non", "off", "faux"):
        return False
    raise ValueError(f"{nom} : oui ou non attendu (reçu « {str(x)[:40]} »)")


def _duree(r) -> str:
    if r not in R_DUREES:
        raise ValueError(f"durée de résistance au feu : « {str(r)[:40]} » ({', '.join(R_DUREES)})")
    return r


def _connus(element: str, parametres: dict, permis) -> None:
    inconnus = sorted(k for k in parametres if k not in permis)
    if inconnus:
        raise ValueError(f"paramètre(s) inconnu(s) pour un élément « {element} » : {', '.join(inconnus)} (admis : {', '.join(permis)})")


def libelle_duree(r: str, critere: str = "R") -> str:
    """« R 90 », « REI 90 », « EI 90 » : la durée avec son critère (R : résistance ; E : étanchéité ; I : isolation)."""
    return f"{critere} {MINUTES[r]}"


# ---------------------------------------------------------------------------------------------- les combinaisons
def _combinaison(dimension, a, repere: str = "") -> dict:
    return {"dimension": dimension, "a": a, "repere": repere, "huit_barres": repere == "**", "etoile": repere == "*"}


def _retenir(combinaisons: list, dimension):
    """(indice, atteinte) de la combinaison que la dimension réelle autorise : la plus favorable — la plus grande
    dimension minimale qui ne dépasse pas la dimension réelle. Sans dimension : la première, « à vérifier » (None).
    Dimension sous le minimum : la première, NON atteinte. Tableau sans valeur : (None, False)."""
    if not combinaisons:
        return None, False
    if dimension is None:
        return 0, None
    permises = [i for i, c in enumerate(combinaisons) if c["dimension"] <= dimension]
    if not permises:
        return 0, False
    return max(permises, key=lambda i: combinaisons[i]["dimension"]), True


def _texte_combinaison(c: dict) -> str:
    """« 350/53 », « 450/40** », « 60 » (épaisseur seule)."""
    if c["a"] is None:
        return f"{c['dimension']}"
    return f"{c['dimension']}/{c['a']}{c['repere']}"


def _resultat(element: str, r: str, numero: str, colonne: str, colonne_titre: str, combinaisons: list, dimension, parametres: dict,
              vide: str = "") -> dict:
    """Ce que ``exigence`` rend pour une lecture de tableau : la combinaison retenue et ce qu'elle demande."""
    t = TABLEAUX[numero]
    symbole, nom_dim = t["dimension"]
    i, atteinte = _retenir(combinaisons, dimension)
    ret = combinaisons[i] if i is not None else None
    duree = libelle_duree(r, t["critere"])
    raison = ""
    if ret is None:
        raison = vide or f"le tableau {numero} ne donne aucune valeur pour {duree}"
    elif atteinte is False:
        raison = (f"{nom_dim} {symbole} = {_fr(dimension)} mm sous le minimum du tableau {numero} pour {duree} "
                  f"({ret['dimension']} mm)")
    return {"element": element, "r": r, "minutes": MINUTES[r], "duree": duree, "critere": t["critere"], "methode": "tableau",
            "tableau": numero, "titre": t["titre"], "clause": t["clause"], "colonne": colonne, "colonne_titre": colonne_titre,
            "parametres": parametres, "combinaisons": combinaisons, "indice": i, "retenue": ret,
            "combinaison": _texte_combinaison(ret) if ret else None,
            "a": ret["a"] if ret else None, "a_sd": None, "angle": False,
            "symbole_dimension": symbole, "nom_dimension": nom_dim, "dimension": dimension,
            "dimension_min": ret["dimension"] if ret else None,
            "huit_barres": bool(ret and ret["huit_barres"]), "etoile": bool(ret and ret["etoile"]),
            "atteinte": atteinte, "raison": raison, "mentions": [], "formule": None, "comparaison": None,
            "notes": list(t["notes"]), "conditions": list(t["conditions"]), "non_couvert": list(NON_COUVERT),
            "source": source(numero), "verification": VERIFICATIONS[numero]}


# ---------------------------------------------------------------------------------------------- poutres
def _combinaisons_poutre(appuis: str, r: str) -> list:
    table = POUTRE_5_5 if appuis == "simples" else POUTRE_5_6
    return [_combinaison(*c) for c in table[r]]


def largeur_a_sd(numero: str, combinaisons: list) -> int:
    """La largeur JUSQU'À laquelle (comprise) a_sd est majoré de 10 mm : celle de la combinaison de rang
    ``RANG_SANS_A_SD`` dans la ligne du tableau (colonne 4 du tableau 5.5, colonne 3 du tableau 5.6)."""
    return combinaisons[RANG_SANS_A_SD[numero]]["dimension"]


def _poutre(r: str, appuis="simples", b=None) -> dict:
    appuis = _parmi(appuis, tuple(k for k, _, _ in APPUIS), "appuis de la poutre")
    b = _dimension(b, "largeur b de la poutre")
    numero = dict((k, n) for k, _, n in APPUIS)[appuis]
    libelle = dict((k, t) for k, t, _ in APPUIS)[appuis]
    out = _resultat("poutre", r, numero, appuis, f"poutre {libelle}", _combinaisons_poutre(appuis, r), b,
                    {"appuis": appuis, "b": b})
    # barres d'angle d'une poutre à un seul lit : a_sd = a + 10 mm jusqu'à la largeur de la colonne nommée (comprise) ; au-delà, a_sd = a
    seuil = largeur_a_sd(numero, out["combinaisons"])
    out["angle"] = b is None or b <= seuil
    out["a_sd"] = out["a"] + (A_SD_ANGLE if out["angle"] else 0)
    out["largeur_a_sd"] = seuil
    return out


# ---------------------------------------------------------------------------------------------- dalles
def _dalle(r: str, portee="une", h=None) -> dict:
    portee = _parmi(portee, tuple(k for k, _ in PORTEES), "portée de la dalle")
    h = _dimension(h, "épaisseur h de la dalle")
    libelle = dict(PORTEES)[portee]
    if portee == "plancher_dalle":
        hs, a, repere = DALLE_5_9[r]
        return _resultat("dalle", r, "5.9", "a", libelle, [_combinaison(hs, a, repere)], h, {"portee": portee, "h": h})
    ligne = DALLE_5_8[r]
    colonne = COLONNE_DALLE_CONTINUE if portee == "continue" else portee
    a, repere = ligne[colonne]
    out = _resultat("dalle", r, "5.8", colonne, libelle, [_combinaison(ligne["hs"], a, repere)], h, {"portee": portee, "h": h})
    # les conditions d'emploi de CETTE portée (le tableau entier les porte toutes)
    out["conditions"] = list(CONDITIONS_DALLE["continue" if portee == "continue" else ("une" if portee == "une" else "deux")])
    if portee == "continue":
        out["mentions"].append("Dalle continue (§5.7.3) : l'épaisseur h_s et la distance a de la colonne « deux directions, "
                               "l_y/l_x ≤ 1,5 » du tableau 5.8, sous les conditions de redistribution et d'armatures sur appuis.")
    return out


# ---------------------------------------------------------------------------------------------- voiles
def _voile(r: str, porteur=True, mu_fi=0.7, faces=1, e=None) -> dict:
    porteur = _booleen(porteur, "voile porteur")
    e = _dimension(e, "épaisseur e du voile")
    if not porteur:
        return _resultat("voile", r, "5.3", "ei", "voile non porteur", [_combinaison(VOILE_5_3[r], None)], e,
                         {"porteur": False, "e": e})
    mu = _mu(mu_fi, MU_VOILE, "taux de chargement μ_fi du voile")
    n = _nombre(faces, "faces exposées du voile")
    if n not in (1, 2):
        raise ValueError(f"faces exposées du voile : 1 ou 2 (reçu « {str(faces)[:40]} »)")
    n = int(n)
    colonne = f"mu{'035' if mu == 0.35 else '07'}_{n}"
    ep, a, repere = VOILE_5_4[r][colonne]
    return _resultat("voile", r, "5.4", colonne, f"voile porteur, μ_fi = {_fr(mu)}, {'une face exposée' if n == 1 else 'deux faces exposées'}",
                     [_combinaison(ep, a, repere)], e, {"porteur": True, "mu_fi": mu, "faces": n, "e": e})


# ---------------------------------------------------------------------------------------------- poteaux : la formule (5.7)
def _barres(x) -> str:
    """« 4 » (quatre barres, aux angles) ou « plus » (plus de quatre) ; un nombre de barres est lu de même."""
    if isinstance(x, str) and x.strip().lower() in ("plus", "plus de 4", ">4", "> 4"):
        return "plus"
    n = _nombre(x, "nombre de barres du poteau")
    if n is None or n != int(n) or n < 4:
        raise ValueError(f"nombre de barres du poteau : 4 ou « plus de 4 » (reçu « {str(x)[:40]} »)")
    return "4" if n == 4 else "plus"


def r_eta(mu_fi: float, omega: float = 0.0) -> float:
    """R_ηfi = 83·[1,00 − μ_fi·(1 + ω)/((0,85/α_cc) + ω)]. Avec α_cc = 0,85 le rapport vaut 1 quel que soit ω : le
    taux mécanique d'armatures n'est donc pas demandé (ω = 0 ici ; avec un autre α_cc, ω = 0 serait le cas le plus
    sévère)."""
    return 83.0 * (1.0 - mu_fi * (1.0 + omega) / ((0.85 / ALPHA_CC) + omega))


def somme_exigee(minutes: float) -> float:
    """La somme R_ηfi + R_a + R_l + R_b + R_n qui donne la durée R : l'éq. (5.7) inversée, 120·(R/120)^(1/1,8)."""
    return 120.0 * (minutes / 120.0) ** (1.0 / 1.8)


def _entree_formule(b, h, l0_fi, mu_fi, barres) -> dict:
    """Les paramètres de la formule (5.7) relus et contrôlés : hors domaine, REFUS (le champ nommé) ; l_0,fi sous 2 m
    et μ_fi sous 0,2 sont ramenés à la borne — côté sûr, permis par la norme — et c'est DIT (``mentions``)."""
    b = _dimension(b, "plus petit côté b du poteau")
    if b is None:
        raise ValueError("plus petit côté b du poteau : nécessaire à la formule (5.7)")
    h = _dimension(h, "grand côté h du poteau")
    h = b if h is None else h
    if h < b:
        raise ValueError(f"grand côté h du poteau : au moins égal au plus petit côté b (h = {_fr(h)} mm, b = {_fr(b)} mm)")
    if h > H_SUR_B_MAX * b + 1e-9:
        raise ValueError(f"grand côté h du poteau : h ≤ 1,5·b dans le domaine de la formule (5.7) (h = {_fr(h)} mm, 1,5·b = {_fr(1.5 * b)} mm)")
    b_prime = 2.0 * b * h / (b + h)
    if not B_PRIME[0] - 1e-9 <= b_prime <= B_PRIME[1] + 1e-9:
        raise ValueError(f"plus petit côté b du poteau : b′ = 2·b·h/(b + h) = {_fr(round(b_prime, 1))} mm hors du domaine de la formule (5.7) "
                         f"({_fr(B_PRIME[0])} à {_fr(B_PRIME[1])} mm)")
    l0 = _nombre(l0_fi, "longueur efficace au feu l_0,fi")
    l0 = L0_FI_DEFAUT if l0 is None else l0
    if l0 <= 0:
        raise ValueError(f"longueur efficace au feu l_0,fi : un nombre de mètres positif (reçu {_fr(l0)})")
    if l0 > L0_FI[1] + 1e-9:
        raise ValueError(f"longueur efficace au feu l_0,fi : au plus {_fr(L0_FI[1])} m dans le domaine de la formule (5.7) (reçu {_fr(l0)} m)")
    mu = _nombre(mu_fi, "taux de chargement μ_fi du poteau")
    if mu is None:
        raise ValueError("taux de chargement μ_fi du poteau : nécessaire à la formule (5.7)")
    if mu <= 0:
        raise ValueError(f"taux de chargement μ_fi du poteau : un nombre positif (reçu {_fr(mu)})")
    if mu > MU_FORMULE[1] + 1e-9:
        raise ValueError(f"taux de chargement μ_fi du poteau : au plus {_fr(MU_FORMULE[1])} dans le domaine de la formule (5.7) (reçu {_fr(mu)})")
    mentions = []
    l0_calcul, mu_calcul = l0, mu
    if l0 < L0_FI[0]:
        l0_calcul = L0_FI[0]
        mentions.append(f"l_0,fi = {_fr(l0)} m est sous 2 m : la formule est appliquée avec l_0,fi = 2 m (côté sûr, §5.3.2(4)).")
    if mu < MU_FORMULE[0]:
        mu_calcul = MU_FORMULE[0]
        mentions.append(f"μ_fi = {_fr(mu)} est sous 0,2 : la formule est appliquée avec μ_fi = 0,2 (côté sûr).")
    n = _barres(barres)
    return {"b": b, "h": h, "b_prime": b_prime, "l0_fi": l0, "l0_calcul": l0_calcul, "mu_fi": mu, "mu_calcul": mu_calcul, "barres": n,
            "alpha_cc": ALPHA_CC, "R_eta": r_eta(mu_calcul), "R_l": 9.60 * (5.0 - l0_calcul), "R_b": 0.09 * b_prime,
            "R_n": 0.0 if n == "4" else 12.0, "mentions": mentions}


def poteau_formule(r, b, h=None, l0_fi=L0_FI_DEFAUT, mu_fi=0.7, barres=4) -> dict:
    """La distance à l'axe EXIGÉE par la formule (5.7) pour la durée ``r`` : a = 30 + [120·(R/120)^(1/1,8) − R_ηfi −
    R_l − R_b − R_n]/1,60, arrondie au millimètre SUPÉRIEUR, jamais sous 25 mm ; au-delà de 80 mm la durée est « non
    atteinte » (hors du domaine de la formule). Rend les cinq termes, b′, la valeur brute et la valeur retenue."""
    r = _duree(r)
    e = _entree_formule(b, h, l0_fi, mu_fi, barres)
    exige = somme_exigee(MINUTES[r])
    r_a = exige - e["R_eta"] - e["R_l"] - e["R_b"] - e["R_n"]
    a_brut = 30.0 + r_a / 1.60
    a = max(int(A_FORMULE[0]), int(math.ceil(a_brut - 1e-9)))
    atteinte = a_brut <= A_FORMULE[1] + 1e-9
    raison = "" if atteinte else (f"la formule (5.7) demande a = {_fr(round(a_brut, 1))} mm, au-delà de {_fr(A_FORMULE[1])} mm : "
                                  "hors de son domaine")
    return {**e, "r": r, "minutes": MINUTES[r], "somme_exigee": exige, "R_a": r_a, "a_brut": a_brut, "a": a,
            "plancher": a_brut < A_FORMULE[0], "atteinte": atteinte, "raison": raison}


def poteau_duree(a, b, h=None, l0_fi=L0_FI_DEFAUT, mu_fi=0.7, barres=4) -> dict:
    """La durée R OBTENUE (minutes) pour une distance à l'axe ``a`` donnée, par la formule (5.7). ``a`` sous 25 mm :
    refusé (hors domaine) ; au-delà de 80 mm : la formule est appliquée avec 80 mm (côté sûr), et c'est dit."""
    a = _nombre(a, "distance à l'axe a")
    if a is None or a < A_FORMULE[0] - 1e-9:
        raise ValueError(f"distance à l'axe a : au moins {_fr(A_FORMULE[0])} mm dans le domaine de la formule (5.7) "
                         f"(reçu {'rien' if a is None else _fr(a) + ' mm'})")
    e = _entree_formule(b, h, l0_fi, mu_fi, barres)
    a_calcul = min(a, A_FORMULE[1])
    if a > A_FORMULE[1]:
        e["mentions"].append(f"a = {_fr(a)} mm dépasse 80 mm : la durée est calculée avec a = 80 mm (côté sûr).")
    r_a = 1.60 * (a_calcul - 30.0)
    somme = e["R_eta"] + r_a + e["R_l"] + e["R_b"] + e["R_n"]
    minutes = 120.0 * (somme / 120.0) ** 1.8 if somme > 0 else 0.0
    classe = next((d for d in reversed(R_DUREES) if MINUTES[d] <= minutes + 1e-9), None)
    return {**e, "a": a, "a_calcul": a_calcul, "R_a": r_a, "somme": somme, "R": minutes, "classe": classe}


# ---------------------------------------------------------------------------------------------- poteaux
EXPOSITIONS_POTEAU = (("plusieurs", "exposé sur plus d'un côté"), ("un", "exposé sur un seul côté"))
METHODES_POTEAU = (("tableau", "tableau 5.2a"), ("formule", "formule (5.7)"))
METHODE_DEFAUT = "tableau"       # la formule n'est JAMAIS la méthode par défaut (un test le fige)


def _colonne_poteau(exposition: str, mu: float) -> tuple:
    if exposition == "un":
        return "un_cote", "poteau exposé sur un seul côté, μ_fi = 0,7"
    return MU_POTEAU[mu], f"poteau exposé sur plus d'un côté, μ_fi = {_fr(mu)}"


def _poteau_tableau(r: str, exposition: str, mu_fi, b) -> dict:
    b = _dimension(b, "plus petit côté b du poteau")
    mentions = []
    if exposition == "un":
        mu_lu = _nombre(mu_fi, "taux de chargement μ_fi du poteau")
        if mu_lu is not None and mu_lu > 0.7 + 1e-9:
            raise ValueError(f"taux de chargement μ_fi du poteau : au plus 0,7 (reçu {_fr(mu_lu)})")
        if mu_lu is not None and abs(mu_lu - 0.7) > 1e-9:
            mentions.append(f"Poteau exposé sur un seul côté : le tableau 5.2a ne donne que la colonne μ_fi = 0,7 ; elle est retenue pour "
                            f"μ_fi = {_fr(mu_lu)} (côté sûr).")
        mu = 0.7
    else:
        mu = _mu(mu_fi, tuple(MU_POTEAU), "taux de chargement μ_fi du poteau")
    colonne, titre = _colonne_poteau(exposition, mu)
    combinaisons = [_combinaison(*c) for c in POTEAU_5_2A[r][colonne]]
    out = _resultat("poteau", r, "5.2a", colonne, titre, combinaisons, b,
                    {"methode": "tableau", "exposition": exposition, "mu_fi": mu, "b": b},
                    vide=f"le tableau 5.2a n'a pas de valeur pour {libelle_duree(r)} à μ_fi = {_fr(mu)} — méthode B (tableau 5.2b), annexe C ou calcul")
    out["mentions"] += mentions
    return out


def _poteau(r: str, methode=METHODE_DEFAUT, exposition="plusieurs", b=None, mu_fi=0.7, h=None, l0_fi=None, barres=None) -> dict:
    methode = _parmi(methode, tuple(k for k, _ in METHODES_POTEAU), "méthode du poteau")
    exposition = _parmi(exposition, tuple(k for k, _ in EXPOSITIONS_POTEAU), "exposition du poteau au feu")
    if methode == "tableau":
        for nom, v in (("grand côté h", h), ("longueur efficace l_0,fi", l0_fi), ("nombre de barres", barres)):
            if v is not None:
                raise ValueError(f"{nom} du poteau : sans objet avec le tableau 5.2a (il sert à la formule (5.7))")
        return _poteau_tableau(r, exposition, mu_fi, b)
    if exposition == "un":
        raise ValueError("exposition du poteau au feu : la formule (5.7) vaut pour un poteau exposé sur plus d'un côté — "
                         "pour un seul côté, lire le tableau 5.2a")
    f = poteau_formule(r, b, h, L0_FI_DEFAUT if l0_fi is None else l0_fi, mu_fi, 4 if barres is None else barres)
    t = TABLEAUX["5.7"]
    # la comparaison : ce que le tableau 5.2a de l'EN donne pour la même largeur, dans la première colonne tabulée qui
    # couvre μ_fi (0,2, 0,5 ou 0,7 : la colonne du dessus, jamais celle du dessous)
    mu_tab = next(m for m in sorted(MU_POTEAU) if m >= f["mu_calcul"] - 1e-9)
    tab = _poteau_tableau(r, "plusieurs", mu_tab, f["b"])
    if tab["retenue"] is None:
        texte = "aucune valeur"
    elif tab["atteinte"] is False:
        texte = f"{tab['combinaison']} (b = {_fr(f['b'])} mm sous le minimum)"
    else:
        texte = tab["combinaison"]
    comparaison = {"tableau": "5.2a", "mu_fi": mu_tab, "texte": texte, "a": tab["a"] if tab["atteinte"] else None,
                   "dimension_min": tab["dimension_min"], "atteinte": tab["atteinte"], "huit_barres": tab["huit_barres"],
                   "phrase": f"tableau 5.2a de l'EN (μ_fi = {_fr(mu_tab)}) : {texte}"}
    combinaison = _combinaison(None, f["a"], "")
    return {"element": "poteau", "r": r, "minutes": MINUTES[r], "duree": libelle_duree(r), "critere": "R", "methode": "formule",
            "tableau": "5.7", "titre": t["titre"], "clause": t["clause"], "colonne": None,
            "colonne_titre": f"poteau exposé sur plus d'un côté, μ_fi = {_fr(f['mu_fi'])}, l_0,fi = {_fr(f['l0_fi'])} m, "
                             f"{'4 barres' if f['barres'] == '4' else 'plus de 4 barres'}",
            "parametres": {"methode": "formule", "exposition": "plusieurs", "mu_fi": f["mu_fi"], "b": f["b"], "h": f["h"], "l0_fi": f["l0_fi"],
                           "barres": f["barres"]},
            "combinaisons": [combinaison] if f["atteinte"] else [], "indice": 0 if f["atteinte"] else None,
            "retenue": combinaison if f["atteinte"] else None, "combinaison": f"a ≥ {f['a']}" if f["atteinte"] else None,
            "a": f["a"] if f["atteinte"] else None, "a_sd": None, "angle": False,
            "symbole_dimension": "b′", "nom_dimension": t["dimension"][1], "dimension": f["b_prime"], "dimension_min": B_PRIME[0],
            "huit_barres": False, "etoile": False, "atteinte": f["atteinte"], "raison": f["raison"],
            "mentions": f["mentions"] + [MENTION_FORMULE, MENTION_PAS_DEFAUT], "formule": f, "comparaison": comparaison,
            "notes": list(t["notes"]), "conditions": list(t["conditions"]), "non_couvert": list(NON_COUVERT),
            "source": source("5.7"), "verification": VERIFICATIONS["5.7"]}


# ---------------------------------------------------------------------------------------------- la porte
PARAMETRES = {"poutre": ("appuis", "b"), "dalle": ("portee", "h"),
              "poteau": ("methode", "exposition", "b", "mu_fi", "h", "l0_fi", "barres"),
              "voile": ("porteur", "mu_fi", "faces", "e")}
_LECTEURS = {"poutre": _poutre, "dalle": _dalle, "poteau": _poteau, "voile": _voile}


def exigence(element: str, r: str, **parametres) -> dict:
    """Ce que la méthode tabulée demande à un ÉLÉMENT pour la durée ``r`` (« R30 » … « R240 ») :

    - poutre : ``appuis`` (« simples » → tableau 5.5 | « continue » → 5.6), ``b`` (largeur, mm) ;
    - dalle : ``portee`` (« une » | « deux_15 » | « deux_20 » | « continue » | « plancher_dalle » → 5.9), ``h`` (épaisseur, mm) ;
    - poteau : ``methode`` (« tableau », le défaut | « formule »), ``exposition`` (« plusieurs » | « un »), ``b`` (plus
      petit côté ou diamètre, mm), ``mu_fi`` (0,2 | 0,5 | 0,7 pour le tableau ; quelconque jusqu'à 0,7 pour la
      formule) et, pour la formule seulement, ``h`` (grand côté, mm ; b par défaut), ``l0_fi`` (m ; 3 par défaut),
      ``barres`` (4 | « plus ») ;
    - voile : ``porteur`` (oui par défaut), ``mu_fi`` (0,35 | 0,7), ``faces`` (1 | 2), ``e`` (épaisseur, mm).

    Rend : ``tableau`` (son numéro), ``combinaisons`` (celles de la durée, dans la colonne des paramètres),
    ``retenue`` (celle que la dimension réelle autorise ; sans dimension : la première), ``a``, ``a_sd`` (poutre à un
    lit), ``dimension_min``, ``dimension`` (réelle), ``huit_barres``, ``atteinte`` (True ; False : « non atteinte par
    la méthode tabulée », ``raison`` dit pourquoi ; None : dimension non donnée, à vérifier), ``mentions`` (ce qui a
    été ramené ou choisi, dit), ``notes``, ``conditions``, ``non_couvert``, ``source``, ``verification`` ; pour la
    formule : ``formule`` (ses termes) et ``comparaison`` (le tableau 5.2a de l'EN pour la même largeur)."""
    if element not in _LECTEURS:
        raise ValueError(f"élément : « {str(element)[:40]} » ({', '.join(ELEMENTS)})")
    r = _duree(r)
    _connus(element, parametres, PARAMETRES[element])
    return _LECTEURS[element](r, **parametres)


# ---------------------------------------------------------------------------------------------- le tableau entier
def _numero(element: str, parametres: dict) -> str:
    if element == "poutre":
        return dict((k, n) for k, _, n in APPUIS)[_parmi(parametres.get("appuis", "simples"), tuple(k for k, _, _ in APPUIS), "appuis de la poutre")]
    if element == "dalle":
        return "5.9" if parametres.get("portee") == "plancher_dalle" else "5.8"
    if element == "voile":
        return "5.4" if _booleen(parametres.get("porteur", True), "voile porteur") else "5.3"
    return "5.2a"


def tableau(element: str, **parametres) -> dict:
    """Le tableau ENTIER de l'élément et de sa variante, pour l'écran : ``numero``, ``titre``, ``clause``, ``critere``,
    ``durees``, ``colonnes`` ([{cle, titre, sous, sorte, parametres}] — ``sorte`` : « combinaison » (dimension / a),
    « dimension » (une épaisseur seule) ou « a » (une distance seule) ; ``parametres`` : ceux qui choisissent la
    colonne), ``lignes`` ({durée: {colonne: [combinaison, …]}}), ``notes``, ``conditions``, ``verification``,
    ``source``. Les paramètres sont ceux d'``exigence`` (seuls ceux qui choisissent le tableau sont lus)."""
    if element not in _LECTEURS:
        raise ValueError(f"élément : « {str(element)[:40]} » ({', '.join(ELEMENTS)})")
    _connus(element, parametres, PARAMETRES[element])
    numero = _numero(element, parametres)
    t = TABLEAUX[numero]
    if numero == "5.2a":
        colonnes = [{"cle": k, "titre": f"μ_fi = {_fr(m)}", "sous": "exposé sur plus d'un côté", "sorte": "combinaison",
                     "parametres": {"exposition": "plusieurs", "mu_fi": m}} for m, k in MU_POTEAU.items()]
        colonnes.append({"cle": "un_cote", "titre": "μ_fi = 0,7", "sous": "exposé sur un seul côté", "sorte": "combinaison",
                         "parametres": {"exposition": "un", "mu_fi": 0.7}})
        lignes = {r: {c["cle"]: [_combinaison(*x) for x in POTEAU_5_2A[r][c["cle"]]] for c in colonnes} for r in R_DUREES}
    elif numero == "5.3":
        colonnes = [{"cle": "ei", "titre": "e", "sous": "épaisseur minimale", "sorte": "dimension", "parametres": {"porteur": False}}]
        lignes = {r: {"ei": [_combinaison(VOILE_5_3[r], None)]} for r in R_DUREES}
    elif numero == "5.4":
        colonnes = [{"cle": f"mu{'035' if m == 0.35 else '07'}_{n}", "titre": f"μ_fi = {_fr(m)}",
                     "sous": "une face exposée" if n == 1 else "deux faces exposées", "sorte": "combinaison",
                     "parametres": {"porteur": True, "mu_fi": m, "faces": n}} for m in MU_VOILE for n in (1, 2)]
        lignes = {r: {c["cle"]: [_combinaison(*VOILE_5_4[r][c["cle"]])] for c in colonnes} for r in R_DUREES}
    elif numero in ("5.5", "5.6"):
        appuis = "simples" if numero == "5.5" else "continue"
        n = max(len(_combinaisons_poutre(appuis, r)) for r in R_DUREES)
        colonnes = [{"cle": f"c{i + 1}", "titre": str(i + 1), "sous": "b_min / a", "sorte": "combinaison", "parametres": {"appuis": appuis}}
                    for i in range(n)]
        lignes = {}
        for r in R_DUREES:
            cs = _combinaisons_poutre(appuis, r)
            lignes[r] = {f"c{i + 1}": ([cs[i]] if i < len(cs) else []) for i in range(n)}
    elif numero == "5.8":
        colonnes = [{"cle": "hs", "titre": "h_s", "sous": "épaisseur minimale", "sorte": "dimension", "parametres": {}}]
        colonnes += [{"cle": k, "titre": "a", "sous": dict(PORTEES)[k], "sorte": "a", "parametres": {"portee": k}}
                     for k in ("une", "deux_15", "deux_20")]
        lignes = {r: {"hs": [_combinaison(DALLE_5_8[r]["hs"], None)],
                      **{k: [_combinaison(None, *DALLE_5_8[r][k])] for k in ("une", "deux_15", "deux_20")}} for r in R_DUREES}
    else:
        colonnes = [{"cle": "hs", "titre": "h_s", "sous": "épaisseur minimale", "sorte": "dimension", "parametres": {"portee": "plancher_dalle"}},
                    {"cle": "a", "titre": "a", "sous": "lit inférieur", "sorte": "a", "parametres": {"portee": "plancher_dalle"}}]
        lignes = {r: {"hs": [_combinaison(DALLE_5_9[r][0], None)], "a": [_combinaison(None, DALLE_5_9[r][1], DALLE_5_9[r][2])]}
                  for r in R_DUREES}
    return {"element": element, "numero": numero, "titre": t["titre"], "clause": t["clause"], "critere": t["critere"],
            "symbole_dimension": t["dimension"][0], "nom_dimension": t["dimension"][1],
            "durees": list(R_DUREES), "libelles": {r: libelle_duree(r, t["critere"]) for r in R_DUREES},
            "colonnes": colonnes, "lignes": lignes, "notes": list(t["notes"]), "conditions": list(t["conditions"]),
            "verification": VERIFICATIONS[numero], "source": source(numero)}

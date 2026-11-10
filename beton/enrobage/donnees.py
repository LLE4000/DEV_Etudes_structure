# -*- coding: utf-8 -*-
"""Enrobage du béton — les DONNÉES de l'écran Streamlit ``modules/enrobage.py`` (VERSION 2.0), la référence de parité,
recopiées telles quelles : classes d'exposition et leurs textes, classe structurale par défaut, tableau c_min,dur, tableaux
feu (distance à l'axe « a ») des DALLES et des POUTRES, listes des widgets, textes d'aide, captions et guide des classes.

``tests/test_enrobage_moteur.py`` relit ces valeurs DANS le fichier du module (arbre syntaxique, sans l'importer) et dans
ce que son écran affiche (rejoué avec un faux ``st`` et avec l'AppTest de Streamlit) : une copie qui diverge rougit.
Les écarts de ces tableaux avec l'EN 1992-1-1 / l'EN 1992-1-2 relevés à l'extraction sont REPRODUITS (parité) et écrits
dans ``docs/beton/ENROBAGE_MOTEUR.md`` § 6 — on ne les corrige pas ici.

TROIS ÉCARTS VOULUS avec l'écran Streamlit depuis le 10/11/2026 (lot Q, docs/site/LOT_Q_QUATRE_MODULES.md § 4) : le site
ne montre plus une valeur fausse.
- D1 — les POTEAUX : ``FEU_POTEAU`` n'est plus la table de l'écran (elle suivait la colonne μ_fi = 0,2 sous un libellé
  « μ_fi = 0,7 ») ; elle est LUE dans ``beton/bibliotheque/feu.py`` (tableau 5.2a de l'EN 1992-1-2:2004, colonne
  μ_fi = 0,7, poteau exposé sur plus d'un côté — ce que le libellé annonce). R240 n'y a pas de valeur : aucune
  variante, « non couvert par le tableau ».
- D2 — les VOILES : ``FEU_VOILE`` est LUE de même (tableau 5.4, μ_fi = 0,7, une face exposée) : R180 = 50 et R240 = 60
  (l'écran : 45 et 55).
- D6 — le Ø d'étrier n'est plus retiré de a pour une dalle ou un voile (``moteur.calculer``).
"""
from beton.bibliotheque import feu as FEU

# ============================================================ durabilité (EN 1992-1-1 §4.4.1)
# Classes d'exposition : (titre court, description, exemples, quand la choisir) — EXPOSITION_INFO
EXPOSITION_INFO = {
    "X0": (
        "Aucun risque de corrosion ni d'attaque",
        "Béton non armé, ou béton armé en ambiance très sèche.",
        "Intérieur de bâtiments à très faible humidité (locaux chauffés secs).",
        "Élément jamais mouillé, armatures sans risque de corrosion. Rare pour du béton armé.",
    ),
    "XC1": (
        "Sec, ou humide en permanence",
        "Corrosion induite par carbonatation, ambiance sèche ou toujours humide.",
        "Intérieur de bâtiments (humidité faible) ; béton en permanence immergé dans l'eau.",
        "Élément intérieur au sec, OU totalement noyé sous l'eau sans alternance.",
    ),
    "XC2/XC3": (
        "Humidité modérée / rarement sec",
        "Carbonatation, contact durable avec l'eau ou humidité de l'air élevée.",
        "Fondations enterrées (XC2) ; béton extérieur abrité de la pluie (XC3).",
        "Fondations, vides sanitaires humides, éléments extérieurs sous auvent.",
    ),
    "XC4": (
        "Alternativement humide et sec",
        "Carbonatation, cycles d'humidification / séchage.",
        "Surfaces extérieures exposées à la pluie mais non aux sels.",
        "Façades, éléments extérieurs directement exposés à la pluie.",
    ),
    "XD1/XS1": (
        "Humidité modérée / air marin",
        "Corrosion par chlorures (sels de déverglaçage XD1, air marin XS1).",
        "Surfaces exposées aux embruns salins ou à des projections de sels.",
        "Ouvrages à moins de ~1 km de la côte, ou exposés aux sels de route sans contact direct.",
    ),
    "XD2/XS2": (
        "Humide, rarement sec / immersion marine",
        "Chlorures en milieu humide (XD2) ou immersion en eau de mer (XS2).",
        "Piscines, réservoirs d'eaux chlorées ; parties d'ouvrages marins immergées.",
        "Élément en contact durable avec une eau chlorée ou l'eau de mer.",
    ),
    "XD3/XS3": (
        "Alternativement humide et sec / zones de marnage et embruns",
        "Chlorures avec cycles humide/sec (XD3) ou zones de marnage/embruns (XS3).",
        "Tabliers de ponts, dalles de parking exposées aux sels ; ouvrages en zone de marée.",
        "Cas le plus sévère : sels + alternance humide/sec, embruns marins directs.",
    ),
}
EXPO_CLASSES = list(EXPOSITION_INFO.keys())

# Classe structurale proposée par défaut selon l'exposition — CLASSE_STRUCTURALE_DEFAULT (« S4 = référence pour durée
# de vie 50 ans ; ajustée selon exposition — indicatif »)
CLASSE_STRUCTURALE_DEFAULT = {
    "X0": "S1", "XC1": "S2", "XC2/XC3": "S4", "XC4": "S4",
    "XD1/XS1": "S5", "XD2/XS2": "S5", "XD3/XS3": "S6",
}

# c_min,dur (mm) : classe structurale (lignes) × classe d'exposition (colonnes, dans l'ordre de EXPO_CLASSES) — C_MIN_DUR
C_MIN_DUR = {
    "S1": [10, 10, 10, 15, 20, 25, 30],
    "S2": [10, 10, 15, 20, 25, 30, 35],
    "S3": [10, 15, 20, 25, 30, 35, 40],
    "S4": [15, 20, 25, 30, 35, 40, 45],
    "S5": [20, 25, 30, 35, 40, 45, 50],
    "S6": [25, 30, 35, 40, 45, 50, 55],
}
CLASSES_STRUCTURALES = list(C_MIN_DUR.keys())

# ============================================================ feu (EN 1992-1-2, méthode tabulée §5)
# Distance à l'axe « a » (mm) des armatures principales ; R = durée de résistance au feu (min) — R_LEVELS
R_LEVELS = ["R30", "R60", "R90", "R120", "R180", "R240"]
# dalles pleines portant sur une direction (tab. 5.8), armatures inférieures — FEU_DALLE
FEU_DALLE = {
    "R30": 10, "R60": 20, "R90": 30, "R120": 40, "R180": 55, "R240": 65,
}
# voiles porteurs exposés sur une face, μ_fi = 0,7 (tab. 5.4) — ÉCART VOULU D2 : lus dans beton/bibliotheque/feu.py (la
# source unique des tableaux du feu), plus dans l'écran Streamlit (qui écrivait 45 et 55 mm pour R180 et R240)
FEU_VOILE = {r: FEU.VOILE_5_4[r]["mu07_1"][1] for r in R_LEVELS}
# poteaux exposés sur plus d'une face, μ_fi = 0,7 (tab. 5.2a) : R → [(b_min mm, a mm), …] — ÉCART VOULU D1 : lus dans
# beton/bibliotheque/feu.py, colonne μ_fi = 0,7 (l'écran Streamlit portait la colonne μ_fi = 0,2). R240 : le tableau n'a
# pas de valeur à μ_fi = 0,7 — la liste est VIDE, le moteur dit « non couvert par le tableau » (aucune valeur inventée)
FEU_POTEAU = {r: [(b, a) for b, a, _ in FEU.POTEAU_5_2A[r]["mu07"]] for r in R_LEVELS}
# les combinaisons de ce tableau qui demandent AU MOINS 8 BARRES (repère « ** » de la norme) : R → [b_min mm, …]
FEU_POTEAU_8_BARRES = {r: [b for b, _, repere in FEU.POTEAU_5_2A[r]["mu07"] if repere == "**"] for r in R_LEVELS}
# les tableaux et les clauses de l'EN 1992-1-2:2004 dont viennent ces deux tables, et la colonne lue
SOURCES_FEU = {"Poteau": FEU.source("5.2a") + " : colonne μ_fi = 0,7, exposé sur plus d'un côté",
               "Voile": FEU.source("5.4") + " : colonne μ_fi = 0,7, une face exposée"}
# les éléments dont les armatures principales sont entourées d'un étrier : il se retire de la distance à l'axe au feu.
# ÉCART VOULU D6 : pour une dalle ou un voile, le Ø « étrier » de l'écran (8 mm par défaut) n'est plus retiré de a.
TYPES_A_ETRIER = ("Poutre", "Poteau")
# poutres sur appuis simples (tab. 5.5) : R → [(b_min mm, a mm), …] — FEU_POUTRE
FEU_POUTRE = {
    "R30":  [(80, 25), (120, 20), (160, 15), (200, 15)],
    "R60":  [(120, 40), (160, 35), (200, 30), (300, 25)],
    "R90":  [(150, 55), (200, 45), (300, 40), (400, 35)],
    "R120": [(200, 65), (240, 60), (300, 55), (500, 50)],
    "R180": [(240, 80), (300, 70), (400, 65), (600, 60)],
    "R240": [(280, 90), (350, 80), (500, 75), (700, 70)],
}

# ============================================================ les widgets de l'écran (listes, défauts, bornes)
TYPES = ["Poutre", "Poteau", "Dalle", "Voile"]          # « Type d'élément » (selectbox, défaut le premier)
POSITIONS = ["Intérieur", "Extérieur"]                  # « Position dans l'ouvrage » (selectbox) — n'entre dans AUCUN calcul
DELTAS_DEV = [10, 5, 0]                                 # « Tolérance d'exécution Δc_dev [mm] » (radio, index 0)
DIAM_BARRE_DEFAUT, DIAM_BARRE_MIN = 20, 6               # number_input(value=20, step=2, min_value=6) : un entier
DIAM_ETRIER_DEFAUT, DIAM_ETRIER_MIN = 8, 0              # number_input(value=8, step=2, min_value=0) : un entier
# GARDE-FOU DU SITE (écart voulu, docs/beton/ENROBAGE_MOTEUR.md § 5) : l'écran n'a pas de maximum (un Ø de 10⁹ mm s'y
# calcule, et 10⁴⁰⁰ fait déborder les flottants) ; au-delà de ces bornes une saisie est absurde et refusée avec un
# message. 60 mm couvre la plus grosse barre (Ø40, Ø50 exceptionnel) et le Ø équivalent d'un paquet (Ø_n ≤ 55 mm,
# EN 1992-1-1 §8.9.1) ; 40 mm couvre tout étrier et toute barre extérieure aux armatures principales.
DIAM_BARRE_MAX, DIAM_ETRIER_MAX = 60, 40
R_DEFAUT = "R60"                                        # select_slider(value="R60")
TYPES_VARIANTES = ("Poutre", "Poteau")                  # le menu « Largeur minimale » n'existe que pour eux

# ============================================================ les textes de l'écran (libellés, aides, captions)
TITRE = "Calcul de l'enrobage du béton"
CAPTION_TITRE = ("Enrobage nominal retenu = max( durabilité EN 1992-1-1 ; feu EN 1992-1-2 méthode tabulée ). "
                 "Valeurs Annexe nationale belge — à valider par l'ingénieur.")
AIDE_ETRIER = "Décompté dans la vérification au feu (a mesuré à l'axe de la barre principale)."
AIDE_DELTA_DEV = "10 mm par défaut ; peut être réduit avec contrôle qualité renforcé (ANB)."
AIDE_CLASSE_STRUCT = ("S4 = référence (durée de vie 50 ans). Ajustée selon durée de vie, "
                      "classe de résistance, compacité (EN 1992-1-1 tab. 4.3N).")
AIDE_LARGEUR = "À largeur plus grande, la distance à l'axe requise diminue."
CAPTION_FEU = ("μ_fi ≈ 0,7 (taux d'exploitation courant). Pour poteaux/voiles très chargés, "
               "vérifier les tableaux complets.")
LIBELLE_FEU = "Vérifier la résistance au feu (méthode tabulée EN 1992-1-2)"
INVITE_FEU = "- Cochez « Vérifier la résistance au feu » pour l'inclure."

# le guide « Comment choisir la classe d'exposition ? » (markdown de l'écran, mot pour mot)
GUIDE_INTRO = ("La classe d'exposition décrit **l'agression environnementale** subie par le béton. "
               "On identifie d'abord le **mécanisme dominant**, puis le **degré d'humidité / d'exposition**.")
GUIDE_CLASSES = (
    "**1. Pas de risque** — X0 : élément jamais mouillé, béton non armé ou ambiance très sèche.\n\n"
    "**2. Carbonatation (XC)** — corrosion par CO₂, cas le plus fréquent en bâtiment :\n"
    "- XC1 : intérieur sec, ou béton en permanence immergé\n"
    "- XC2 : fondations, contact durable avec l'eau/sol humide\n"
    "- XC3 : extérieur abrité de la pluie, ou humidité de l'air élevée\n"
    "- XC4 : extérieur exposé à la pluie (cycles humide/sec)\n\n"
    "**3. Chlorures hors mer (XD)** — sels de déverglaçage, eaux industrielles :\n"
    "- XD1 : projections/embruns de sels · XD2 : immersion en eau chlorée · XD3 : cycles humide/sec avec sels (ponts, parkings)\n\n"
    "**4. Chlorures marins (XS)** — eau de mer :\n"
    "- XS1 : air marin (côte) · XS2 : immersion permanente · XS3 : zone de marnage et embruns\n\n"
    "**5. Autres agressions** (gel/dégel XF, chimique XA) : non couvertes ici, à traiter séparément."
)
GUIDE_REGLE = ("Règle pratique", "prendre la classe la plus sévère applicable")
GUIDE_PLUSIEURS = ("Un même élément peut relever de **plusieurs classes** (ex. une dalle de parking extérieure : "
                   "XC4 + XD3 + XF). On retient alors, pour chaque mécanisme, la classe la plus contraignante, "
                   "et l'enrobage résulte de la combinaison la plus sévère.")

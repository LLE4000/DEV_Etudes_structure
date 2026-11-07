# -*- coding: utf-8 -*-
"""Poutre bois — les DONNÉES de l'écran Streamlit ``modules/poutre_bois.py`` (la référence de parité), recopiées
telles quelles : la base des classes de résistance, les sections standard, γ_M, k_mod, k_def, les listes de ses
menus, les défauts et les bornes de ses champs.

Ce sont des COPIES nommées : ``tests/test_poutre_bois_moteur.py`` les relit dans le module (ses dictionnaires) et
dans son écran (un faux ``st`` qui enregistre chaque widget : libellé, options, défaut, bornes) et rougit au moindre
écart. Les valeurs de la base ne sont pas celles d'une édition unique d'EN 338 : les écarts relevés sont dans
``docs/bois/POUTRE_BOIS_MOTEUR.md`` (§ 6) ; elles sont reproduites, pas corrigées.
"""

REFERENCE = "modules/poutre_bois.py"

# ---- TIMBER_BDD : la base intégrée (« valeurs usuelles EN 338 – à ajuster si besoin ») ; MPa et kg/m³
CLASSES = {
    "C14": {"fm_k": 14, "ft0_k": 8, "fc0_k": 16, "fc90_k": 2.0, "fv_k": 2.0, "E0_mean": 7000, "G_mean": 440, "rho_mean": 350, "rho_k": 290},
    "C16": {"fm_k": 16, "ft0_k": 10, "fc0_k": 17, "fc90_k": 2.2, "fv_k": 2.5, "E0_mean": 8000, "G_mean": 500, "rho_mean": 370, "rho_k": 310},
    "C18": {"fm_k": 18, "ft0_k": 11, "fc0_k": 18, "fc90_k": 2.2, "fv_k": 2.5, "E0_mean": 9000, "G_mean": 560, "rho_mean": 380, "rho_k": 320},
    "C20": {"fm_k": 20, "ft0_k": 12, "fc0_k": 19, "fc90_k": 2.3, "fv_k": 2.5, "E0_mean": 9500, "G_mean": 600, "rho_mean": 390, "rho_k": 330},
    "C22": {"fm_k": 22, "ft0_k": 13, "fc0_k": 20, "fc90_k": 2.5, "fv_k": 3.0, "E0_mean": 10000, "G_mean": 625, "rho_mean": 410, "rho_k": 340},
    "C24": {"fm_k": 24, "ft0_k": 14, "fc0_k": 21, "fc90_k": 2.7, "fv_k": 4.0, "E0_mean": 11000, "G_mean": 690, "rho_mean": 420, "rho_k": 350},
    "C27": {"fm_k": 27, "ft0_k": 16, "fc0_k": 23, "fc90_k": 2.9, "fv_k": 4.0, "E0_mean": 11500, "G_mean": 720, "rho_mean": 450, "rho_k": 380},
    "C30": {"fm_k": 30, "ft0_k": 18, "fc0_k": 24, "fc90_k": 3.0, "fv_k": 4.0, "E0_mean": 12000, "G_mean": 750, "rho_mean": 460, "rho_k": 380},
    "C35": {"fm_k": 35, "ft0_k": 21, "fc0_k": 27, "fc90_k": 3.4, "fv_k": 4.5, "E0_mean": 13000, "G_mean": 810, "rho_mean": 500, "rho_k": 410},
    "C40": {"fm_k": 40, "ft0_k": 24, "fc0_k": 30, "fc90_k": 3.8, "fv_k": 5.0, "E0_mean": 14000, "G_mean": 875, "rho_mean": 520, "rho_k": 450},
}

# ---- SECTIONS_STD : les sections standard (mm), dans l'ordre du menu de l'écran
SECTIONS_STD = [
    {"tag": "38x150", "b": 38, "h": 150}, {"tag": "45x70", "b": 45, "h": 70}, {"tag": "45x90", "b": 45, "h": 90},
    {"tag": "45x120", "b": 45, "h": 120}, {"tag": "45x145", "b": 45, "h": 145}, {"tag": "45x220", "b": 45, "h": 220},
    {"tag": "63x150", "b": 63, "h": 150}, {"tag": "63x175", "b": 63, "h": 175}, {"tag": "63x200", "b": 63, "h": 200},
    {"tag": "75x225", "b": 75, "h": 225}, {"tag": "100x250", "b": 100, "h": 250},
]

# ---- les paramètres EC5 du module
GAMMA_M = 1.30                                   # bois massif
KMOD = {                                         # par classe de service et durée dominante
    1: {"permanent": 0.60, "long": 0.70, "moyen": 0.80, "court": 0.90, "instant": 1.10},
    2: {"permanent": 0.60, "long": 0.70, "moyen": 0.80, "court": 0.90, "instant": 1.10},
    3: {"permanent": 0.50, "long": 0.55, "moyen": 0.65, "court": 0.70, "instant": 0.90},
}
KDEF = {1: 0.60, 2: 0.80, 3: 2.00}
KH_BORNES = (0.6, 1.3)                           # kh_depth : (150/h)^0,2 borné entre ces deux valeurs
H_REFERENCE = 150.0                              # mm, la hauteur de référence de k_h

# ---- les listes des menus de l'écran (dans SON ordre ; la première est le défaut sauf indication)
CLASSES_SERVICE = (1, 2, 3)                      # st.radio, défaut 1
DUREES = ("permanent", "long", "moyen", "court", "instant")   # st.selectbox, index=1 : « long »
LIMITES_INST = ("L/300", "L/350", "L/400", "L/200")           # « Limite instantanée », défaut L/300
LIMITES_FIN = ("L/200", "L/250", "L/300")                     # « Limite finale », défaut L/200

# ---- les champs numériques de l'écran : (défaut, min, max) de ses st.number_input
CHAMPS = {
    "b": (80.0, 30.0, 300.0),          # « Largeur b (mm) » (section sur mesure)
    "h": (240.0, 80.0, 500.0),         # « Hauteur h (mm) »
    "L": (4.0, 0.5, 12.0),             # « Portée L (m) »
    "G": (2.0, 0.0, 50.0),             # « Charge permanente G (kN/m) »
    "Q": (3.0, 0.0, 50.0),             # « Charge d'exploitation Q (kN/m) »
    "gamma_G": (1.35, 1.0, 2.0),       # « γ_G »
    "gamma_Q": (1.50, 1.0, 2.0),       # « γ_Q »
    "psi2": (0.30, 0.1, 1.0),          # « ψ₂ (quasi-permanent) »
    "a": (60.0, 30.0, 200.0),          # « Largeur d'appui a (mm) » (grisé tant que l'option est éteinte)
}

# ---- la poutre neuve : l'écran au premier affichage
DEFAUTS = {
    "nom": "Poutre 1",
    "classe": "C24",                   # index=classes.index("C24")
    "service": CLASSES_SERVICE[0],
    "duree": DUREES[1],
    "standard": True,                  # st.toggle « Section standard », value=True
    "section": SECTIONS_STD[0]["tag"],   # le menu sans index : la PREMIÈRE section, 38x150
    "lim_inst": LIMITES_INST[0],
    "lim_fin": LIMITES_FIN[0],
    "appui": False,                    # st.toggle « Compression ⟂ fil à l'appui », value=False
}
DEFAUTS.update({k: v[0] for k, v in CHAMPS.items()})

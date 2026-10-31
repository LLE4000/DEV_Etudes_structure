# -*- coding: utf-8 -*-
"""Entrées du PIED DE POTEAU sur béton : platine rectangulaire centrée sur le
poteau, tiges d'ancrage en grille (deux files en x, deux à quatre rangées
en y), massif de béton centré, mortier de scellement.

Conventions (axes de la platine, origine à l'axe du poteau) :
- x : parallèle aux ailes (largeur b du profilé), y : hauteur h du profilé ;
- N_Ed > 0 en COMPRESSION (N_Ed < 0 : soulèvement) ;
- Mx_Ed : moment qui comprime le côté y > 0 (axe fort d'un profilé en I) ;
  My_Ed : moment qui comprime le côté x > 0 (axe faible) ;
- Vx_Ed, Vy_Ed : efforts tranchants selon x et y ; T_Ed : torsion (Mz).

Unités de saisie : mm, kN, kNm, MPa. Voir docs/assemblages/PIED_POTEAU.md.
"""
from acier.bibliotheques import DB, PERSO, noms

FORME_I = "Profilé en I ou H"
FORME_RHS = "Profil creux rectangulaire ou carré (RHS / SHS)"
TYPE_TETE = "Tige à tête (ou plaque d'ancrage)"
TYPE_CROSSE = "Tige à crosse (adhérence)"
TYPE_ETA = "Tige post-installée (valeurs de l'ETA)"
OUI, NON = "Oui", "Non"
BETONS = ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60"]

# PÉRIMÈTRE de la vérification. Par défaut, l'EN 1993-1-8 seule décide :
# platine (épaisseur), tiges (acier), poteau, cordons, béton d'APPUI (fjd) ;
# les modes de ruine du béton d'ANCRAGE (EN 1992-4 : cône, arrachement, bord,
# pry-out, interaction) sont calculés et affichés en INFORMATION. Le périmètre
# complet les fait entrer dans le verdict.
PERIM_EC3 = "Platine et tiges — EN 1993-1-8 (béton d'ancrage en information)"
PERIM_TOUT = "Complet — EN 1993-1-8 + EN 1992-4 (béton d'ancrage dans le verdict)"

# MODE de l'écran. Par défaut le CALCUL MANUEL de la platine (lot D) : boulons
# posés librement, un ou plusieurs éléments soudés, raidisseurs ; l'épaisseur
# requise se lit comme sur une note à la main (moment pris, bras de levier,
# largeur de plat). La VÉRIFICATION COMPLÈTE (lots A à C : Table 6.7,
# tronçons en T par rangée, EN 1992-4) reste disponible pour une disposition
# régulière (grille centrée, un profilé en I ou RHS).
MODE_MANUEL = "Calcul manuel de la platine (boulons et éléments libres)"
MODE_COMPLET = "Vérification complète EN 1993-1-8 + EN 1992-4 (grille régulière)"
MODES = [MODE_MANUEL, MODE_COMPLET]

# Critère de résistance du calcul manuel : l'EN 1993-1-8 (tronçon en T
# plastique côté tendu, console élastique côté comprimé) ou tout élastique
# (contrainte de Von Mises ≤ fy/γM0 dans chaque bande).
CRIT_EC3 = "EN 1993-1-8 : tronçon en T plastique (Mpl), console élastique"
CRIT_ELAS = "Élastique : contrainte de Von Mises ≤ fy/γM0 partout"
CRITERES = [CRIT_EC3, CRIT_ELAS]

# Aire d'appui de la répartition (calcul manuel) : la bande c de l'EN 1993-1-8
# autour des parois, ou toute la platine (platine rigide, comme les logiciels
# d'ancrage qui donnent les efforts des tiges).
APPUI_EC3 = "Aire efficace EN 1993-1-8 (bande c autour des parois)"
APPUI_PLEIN = "Toute la platine (platine rigide, comme les logiciels d'ancrage)"
APPUIS = [APPUI_EC3, APPUI_PLEIN]

# Annotation de FABRICATION, hors moteur et hors CLES : composition de la
# boulonnerie par tige (cartouche du plan de principe), même mécanisme que
# les assemblages poutre–poutre.
VISSERIE_DEFAUT = "1 rondelle + 1 écrou + 1 contre-écrou"


# Éléments soudés du calcul manuel : un profilé en I ou H (catalogue ou
# dimensions), un profil creux rectangulaire ou circulaire, un plat sur chant,
# un rond plein, un rectangle plein (massif, âme seule…).
TYPE_I = "Profilé en I ou H"
TYPE_RHS = "Profil creux rectangulaire (RHS / SHS)"
TYPE_CHS = "Profil creux circulaire (CHS)"
TYPE_PLAT = "Plat sur chant"
TYPE_ROND = "Rond plein"
TYPE_RECT = "Rectangle plein"
TYPES_ELEM = [TYPE_I, TYPE_RHS, TYPE_CHS, TYPE_PLAT, TYPE_ROND, TYPE_RECT]
DIRS_RAID = ["+x", "−x", "+y", "−y"]


def element_defaut(type_=TYPE_I):
    """Un élément soudé neuf : à l'axe de la platine, dimensions courantes.
    Un profilé du catalogue lit ses dimensions dans la bibliothèque ; les
    champs h, b, tw, tf, r, t, d ne servent qu'aux formes personnalisées."""
    return dict(type=type_, prof="HEA 300", x=0, y=0, rot=0, h=200, b=200, tw=8, tf=12, r=0, t=8, d=200)


def raidisseur_defaut():
    """Un raidisseur neuf : plat vertical soudé sur la platine et sur
    l'élément, parti de (x ; y) dans la direction donnée."""
    return dict(x=150, y=0, dir="+x", L=80, t=10, h=150)


def disposition_defaut():
    """La disposition libre par défaut du calcul manuel — la même que la
    grille du cas du bureau : HEA 300 à l'axe, quatre boulons à (±200 ; ±200),
    pas de raidisseur."""
    return dict(tiges=[dict(x=-200, y=-200), dict(x=200, y=-200), dict(x=-200, y=200), dict(x=200, y=200)],
                elems=[element_defaut(TYPE_I)], raid=[])


def defaults():
    """Les entrées et leur valeur par défaut — l'exemple du bureau : HEA 300,
    platine 500 × 500 × 25, quatre tiges M24 8.8, N = 300 kN, Mx = 50 kNm,
    My = 30 kNm."""
    d = disposition_defaut()
    return {
        "id_projet": "", "id_rep": "", "id_red": "", "id_date": "",
        "mode": MODE_MANUEL,
        "tiges": d["tiges"], "elems": d["elems"], "raid": d["raid"],
        "crit": CRIT_EC3, "appui": APPUI_EC3, "xN": 0, "yN": 0,
        "perim": PERIM_EC3,
        "forme": FORME_I,
        "prof_c": "HEA 300", "h_u": None, "b_u": None, "tw_u": None, "tf_u": None, "r_u": None,
        "t_u": None, "nu_c": "S355",
        "bp_u": 500, "lp_u": 500, "tp_u": 25, "nu_pl": "S355", "a_w": 8,
        "n_ax": 2, "n_ay": 2, "w_x": 400, "w_y": 400,
        "ancr": "M24", "classe": "8.8", "typ_a": TYPE_TETE, "hef": 300, "d_h": 40,
        "eta_tau": 10, "eta_k1": 7.7, "eta_nrks": 0, "eta_vrks": 0,
        "t_g": 30, "f_g": 30,
        "beton": "C30/37", "fiss": OUI, "a_cc": 0.85,
        "bc_u": 900, "lc_u": 900, "hc_u": 600,
        "N_Ed": 300, "Mx_Ed": 50, "My_Ed": 30, "Vx_Ed": 0, "Vy_Ed": 40, "T_Ed": 0,
        "L_c": 4000, "contrev": NON,
        "g_M0": 1.0, "g_M2": 1.25, "g_c": 1.5, "g_Mc": 1.5, "C_f": 0.2,
        "fy_u": None, "fu_u": None, "bw_u": None,
    }


CLES = tuple(defaults().keys())
# Les clés du calcul manuel qui sont des LISTES (boulons, éléments soudés,
# raidisseurs) : hors widgets, copiées telles quelles.
LISTES = ("tiges", "elems", "raid")

# Clés passées par N() en tête de compute()
NUM = ["xN", "yN", "h_u", "b_u", "tw_u", "tf_u", "r_u", "t_u", "bp_u", "lp_u", "tp_u", "a_w",
       "n_ax", "n_ay", "w_x", "w_y", "hef", "d_h", "eta_tau", "eta_k1", "eta_nrks", "eta_vrks", "t_g", "f_g", "a_cc",
       "bc_u", "lc_u", "hc_u", "N_Ed", "Mx_Ed", "My_Ed", "Vx_Ed", "Vy_Ed", "T_Ed",
       "L_c", "g_M0", "g_M2", "g_c", "g_Mc", "C_f", "fy_u", "fu_u", "bw_u"]

PROFILS = [PERSO] + noms(DB["profils"])
ACIERS = noms(DB["aciers"]) + [PERSO]
BOULONS = noms(DB["boulons"])
CLASSES = noms(DB["classes"])
OUI_NON = [OUI, NON]


# --------------------------------------------------------------- conditions
def forme_I(u):
    return u.get("forme") != FORME_RHS


def forme_RHS(u):
    return u.get("forme") == FORME_RHS


def perso_c(u):
    return forme_RHS(u) or u.get("prof_c") == PERSO


def tete(u):
    return u.get("typ_a") not in (TYPE_CROSSE, TYPE_ETA)


def eta(u):
    return u.get("typ_a") == TYPE_ETA


def acier_perso(u):
    return PERSO in (u.get("nu_c"), u.get("nu_pl"))


def est_manuel(u):
    return u.get("mode") != MODE_COMPLET


def est_complet(u):
    return u.get("mode") == MODE_COMPLET


def ancrage_txt(u, court=False):
    """« tête Ø40 », « crosse » ou « scellée (ETA) » — pour les textes."""
    t = u.get("typ_a")
    if t == TYPE_CROSSE:
        return "crosse"
    if t == TYPE_ETA:
        return "scellée (ETA)" if court else "scellée, valeurs ETA"
    try:
        return ("tête Ø" if court else "tête dh = ") + str(int(float(u.get("d_h") or 0)))
    except (TypeError, ValueError):
        return "tête"


def fck_de(nom):
    """« C30/37 » → 30 (résistance caractéristique sur cylindre, MPa)."""
    try:
        return float(str(nom).strip().upper().lstrip("C").split("/")[0])
    except (TypeError, ValueError):
        return 0.0


def fy_epaisseur(fy, t):
    """Limite d'élasticité selon l'épaisseur (EN 10025-2) : −10 MPa entre 16
    et 40 mm, −20 MPa entre 40 et 63 mm (S355 : 355 / 345 / 335)."""
    if t is None or t <= 16:
        return fy
    if t <= 40:
        return fy - 10
    return fy - 20


def _c(k, l, t="n", o=None, un=None, h=None, si=None):
    """Un champ : clé, libellé, type (n nombre / s choix / x texte), options,
    unité, aide, condition d'affichage."""
    return dict(k=k, l=l, t=t, o=o, un=un, h=h, si=si)


# ------------------------------------------------------------------ groupes
GROUPES = [
    dict(t="Identification",
         note="Reprise dans la note et l'export texte. Un champ vide n'est pas imprimé.",
         f=[_c("id_projet", "Projet", "x"), _c("id_rep", "Repère de l'assemblage", "x"),
            _c("id_red", "Rédacteur", "x"), _c("id_date", "Date", "x")]),
    dict(t="Mode",
         f=[_c("mode", "Mode de calcul", "s", MODES,
               h="Calcul manuel : boulons et éléments soudés posés librement, raidisseurs, épaisseur requise "
                 "lue comme sur une note à la main. Vérification complète : grille régulière centrée sur un "
                 "profilé en I ou RHS — Table 6.7, tronçons en T par rangée, ancrages EN 1992-4."),
            _c("crit", "Critère de résistance de la platine", "s", CRITERES, si=est_manuel,
               h="EN 1993-1-8 : moment plastique du tronçon en T côté tendu (Tableau 6.2), console élastique "
                 "côté comprimé (§6.2.5). Élastique : σ = 6·M/(b·t²) composée avec τ, Von Mises ≤ fy/γM0 — "
                 "environ 1,22 fois l'épaisseur plastique."),
            _c("appui", "Aire d'appui de la répartition", "s", APPUIS, si=est_manuel,
               h="Bande c autour de chaque paroi (EN 1993-1-8 §6.2.5, sûr) ou toute la platine (platine "
                 "rigide : les efforts des tiges se comparent alors à ceux des logiciels d'ancrage)."),
            _c("xN", "xN — point d'application des efforts selon x", "n", un="mm", si=est_manuel,
               h="Les efforts N, Mx, My, Vx, Vy agissent en (xN ; yN) — l'axe du poteau. Un élément excentré "
                 "par rapport aux boulons crée le moment N·e tout seul."),
            _c("yN", "yN — point d'application selon y", "n", un="mm", si=est_manuel)]),
    dict(t="Périmètre", si=est_complet,
         f=[_c("perim", "Périmètre de la vérification", "s", [PERIM_EC3, PERIM_TOUT],
               h="EN 1993-1-8 seule : platine, tiges (acier), poteau, cordons et béton d'appui fjd décident ; "
                 "les ruines du béton d'ancrage (EN 1992-4) sont affichées en information. Complet : elles "
                 "entrent dans le verdict.")]),
    dict(t="Poteau", si=est_complet,
         f=[_c("forme", "Forme du profilé", "s", [FORME_I, FORME_RHS]),
            _c("prof_c", "Profilé", "s", PROFILS, si=forme_I),
            _c("h_u", "h (mm)", "n", un="mm", si=perso_c), _c("b_u", "b (mm)", "n", un="mm", si=perso_c),
            _c("tw_u", "tw (mm)", "n", un="mm", si=lambda u: forme_I(u) and perso_c(u)),
            _c("tf_u", "tf (mm)", "n", un="mm", si=lambda u: forme_I(u) and perso_c(u)),
            _c("r_u", "r (mm)", "n", un="mm", si=lambda u: forme_I(u) and perso_c(u)),
            _c("t_u", "t (mm) — épaisseur de paroi", "n", un="mm", si=forme_RHS),
            _c("nu_c", "Nuance", "s", ACIERS),
            _c("L_c", "Hauteur du poteau (classification de rigidité)", "n", un="mm"),
            _c("contrev", "Ossature contreventée", "s", OUI_NON)]),
    dict(t="Platine",
         f=[_c("bp_u", "bp — largeur selon x", "n", un="mm"), _c("lp_u", "lp — longueur selon y", "n", un="mm"),
            _c("tp_u", "tp — épaisseur", "n", un="mm"), _c("nu_pl", "Nuance", "s", ACIERS),
            _c("a_w", "a — gorge des cordons poteau / platine", "n", un="mm")]),
    dict(t="Tiges d'ancrage",
         note="Grille centrée : n files selon x espacées de wx au total, n rangées selon y espacées de wy.",
         f=[_c("n_ax", "Files (selon x)", "n", si=est_complet), _c("n_ay", "Rangées (selon y)", "n", si=est_complet),
            _c("w_x", "wx — entraxe total des files", "n", un="mm", si=est_complet),
            _c("w_y", "wy — entraxe total des rangées", "n", un="mm", si=est_complet),
            _c("ancr", "Tige", "s", BOULONS), _c("classe", "Classe", "s", CLASSES),
            _c("typ_a", "Type d'ancrage", "s", [TYPE_TETE, TYPE_CROSSE, TYPE_ETA]),
            _c("hef", "hef — profondeur d'ancrage", "n", un="mm"),
            _c("d_h", "dh — diamètre de la tête / plaque", "n", un="mm", si=tete),
            _c("eta_tau", "τRk — adhérence caractéristique de l'ETA", "n", un="MPa", si=lambda u: eta(u) and est_complet(u),
               h="Valeur de l'ETA de la tige scellée, pour l'état du béton (fissuré ou non) et la classe retenue."),
            _c("eta_k1", "k1 — cône de béton (ETA)", "n", si=lambda u: eta(u) and est_complet(u),
               h="Facteur k1 (kcr fissuré, kucr non fissuré) de l'ETA ; 7,7 / 11,0 à défaut (EN 1992-4)."),
            _c("eta_nrks", "NRk,s — acier en traction (ETA, 0 = calcul)", "n", un="kN",
               si=lambda u: eta(u) and est_complet(u)),
            _c("eta_vrks", "VRk,s — acier en cisaillement (ETA, 0 = calcul)", "n", un="kN",
               si=lambda u: eta(u) and est_complet(u))]),
    dict(t="Béton et scellement",
         f=[_c("beton", "Classe de béton", "s", BETONS), _c("fiss", "Béton fissuré", "s", OUI_NON, si=est_complet),
            _c("a_cc", "αcc", "n", h="Coefficient de l'ANB belge pour fcd = αcc·fck/γc ; 0,85 par défaut, à confirmer."),
            _c("bc_u", "Massif — dimension selon x", "n", un="mm"), _c("lc_u", "Massif — dimension selon y", "n", un="mm"),
            _c("hc_u", "Massif — hauteur", "n", un="mm"),
            _c("t_g", "Mortier — épaisseur", "n", un="mm"), _c("f_g", "Mortier — résistance", "n", un="MPa")]),
    dict(t="Efforts ELU",
         f=[_c("N_Ed", "NEd (compression > 0)", "n", un="kN"),
            _c("Mx_Ed", "Mx,Ed (comprime y > 0)", "n", un="kNm"), _c("My_Ed", "My,Ed (comprime x > 0)", "n", un="kNm"),
            _c("Vx_Ed", "Vx,Ed", "n", un="kN"), _c("Vy_Ed", "Vy,Ed", "n", un="kN"),
            _c("T_Ed", "TEd — torsion", "n", un="kNm")]),
    dict(t="Coefficients",
         f=[_c("g_M0", "γM0", "n"), _c("g_M2", "γM2", "n"), _c("g_c", "γc (béton)", "n"),
            _c("g_Mc", "γMc (EN 1992-4)", "n", si=est_complet), _c("C_f", "Cf,d — frottement platine / mortier", "n")]),
    dict(t="Acier personnalisé", si=acier_perso,
         f=[_c("fy_u", "fy", "n", un="MPa"), _c("fu_u", "fu", "n", un="MPa"), _c("bw_u", "βw", "n")]),
]

COURT = {"bp_u": "bp", "lp_u": "lp", "tp_u": "tp", "a_w": "a", "w_x": "wx", "w_y": "wy", "xN": "xN", "yN": "yN",
         "mode": "mode", "crit": "critère", "appui": "appui",
         "hef": "hef", "d_h": "dh", "eta_tau": "τRk", "eta_k1": "k1", "eta_nrks": "NRk,s", "eta_vrks": "VRk,s",
         "t_g": "tg", "f_g": "fg", "N_Ed": "NEd", "Mx_Ed": "Mx", "My_Ed": "My",
         "Vx_Ed": "Vx", "Vy_Ed": "Vy", "T_Ed": "TEd", "prof_c": "profilé", "ancr": "tige",
         "bc_u": "bc", "lc_u": "lc", "hc_u": "hc", "n_ax": "files", "n_ay": "rangées",
         "h_u": "h", "b_u": "b", "tw_u": "tw", "tf_u": "tf", "r_u": "r", "t_u": "t", "L_c": "Lc"}
UNITE = {}
CHAMPS = {}
for _g in GROUPES:
    for _f in _g["f"]:
        UNITE[_f["k"]] = " " + _f["un"] if _f.get("un") and _f["un"] != "-" else ""
        COURT.setdefault(_f["k"], _f["l"])
        CHAMPS[_f["k"]] = _f


def groupes_visibles(u):
    out = []
    for g in GROUPES:
        if g.get("si") and not g["si"](u):
            continue
        champs = [f for f in g["f"] if not f.get("si") or f["si"](u)]
        out.append((g, champs))
    return out


def champ_visible(u, k):
    return any(f["k"] == k for _, fs in groupes_visibles(u) for f in fs)


def charger_json(objet):
    """Fusion d'un fichier enregistré sur les défauts, clé par clé (les clés
    inconnues sont ignorées). Un fichier d'avant le calcul manuel (sans clé
    « mode ») s'ouvre en VÉRIFICATION COMPLÈTE : c'est ce qu'il décrivait."""
    d = defaults()
    if not isinstance(objet, dict):
        raise ValueError("Structure JSON inattendue")
    for k in d:
        if k in objet:
            d[k] = objet[k]
    if "mode" not in objet:
        d["mode"] = MODE_COMPLET
    for k in LISTES:
        if not isinstance(d[k], list):
            d[k] = disposition_defaut()[k]
    return d

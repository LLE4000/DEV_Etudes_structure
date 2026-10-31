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
OUI, NON = "Oui", "Non"
BETONS = ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60"]

# PÉRIMÈTRE de la vérification. Par défaut, l'EN 1993-1-8 seule décide :
# platine (épaisseur), tiges (acier), poteau, cordons, béton d'APPUI (fjd) ;
# les modes de ruine du béton d'ANCRAGE (EN 1992-4 : cône, arrachement, bord,
# pry-out, interaction) sont calculés et affichés en INFORMATION. Le périmètre
# complet les fait entrer dans le verdict.
PERIM_EC3 = "Platine et tiges — EN 1993-1-8 (béton d'ancrage en information)"
PERIM_TOUT = "Complet — EN 1993-1-8 + EN 1992-4 (béton d'ancrage dans le verdict)"

# Annotation de FABRICATION, hors moteur et hors CLES : composition de la
# boulonnerie par tige (cartouche du plan de principe), même mécanisme que
# les assemblages poutre–poutre.
VISSERIE_DEFAUT = "1 rondelle + 1 écrou + 1 contre-écrou"


def defaults():
    """Les entrées et leur valeur par défaut — l'exemple du bureau : HEA 300,
    platine 500 × 500 × 25, quatre tiges M24 8.8, N = 300 kN, Mx = 50 kNm,
    My = 30 kNm."""
    return {
        "id_projet": "", "id_rep": "", "id_red": "", "id_date": "",
        "perim": PERIM_EC3,
        "forme": FORME_I,
        "prof_c": "HEA 300", "h_u": None, "b_u": None, "tw_u": None, "tf_u": None, "r_u": None,
        "t_u": None, "nu_c": "S355",
        "bp_u": 500, "lp_u": 500, "tp_u": 25, "nu_pl": "S355", "a_w": 8,
        "n_ax": 2, "n_ay": 2, "w_x": 400, "w_y": 400,
        "ancr": "M24", "classe": "8.8", "typ_a": TYPE_TETE, "hef": 300, "d_h": 40,
        "t_g": 30, "f_g": 30,
        "beton": "C30/37", "fiss": OUI, "a_cc": 0.85,
        "bc_u": 900, "lc_u": 900, "hc_u": 600,
        "N_Ed": 300, "Mx_Ed": 50, "My_Ed": 30, "Vx_Ed": 0, "Vy_Ed": 40, "T_Ed": 0,
        "L_c": 4000, "contrev": NON,
        "g_M0": 1.0, "g_M2": 1.25, "g_c": 1.5, "g_Mc": 1.5, "C_f": 0.2,
        "fy_u": None, "fu_u": None, "bw_u": None,
    }


CLES = tuple(defaults().keys())

# Clés passées par N() en tête de compute()
NUM = ["h_u", "b_u", "tw_u", "tf_u", "r_u", "t_u", "bp_u", "lp_u", "tp_u", "a_w",
       "n_ax", "n_ay", "w_x", "w_y", "hef", "d_h", "t_g", "f_g", "a_cc",
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
    return u.get("typ_a") != TYPE_CROSSE


def acier_perso(u):
    return PERSO in (u.get("nu_c"), u.get("nu_pl"))


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
    dict(t="Périmètre",
         f=[_c("perim", "Périmètre de la vérification", "s", [PERIM_EC3, PERIM_TOUT],
               h="EN 1993-1-8 seule : platine, tiges (acier), poteau, cordons et béton d'appui fjd décident ; "
                 "les ruines du béton d'ancrage (EN 1992-4) sont affichées en information. Complet : elles "
                 "entrent dans le verdict.")]),
    dict(t="Poteau",
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
         f=[_c("n_ax", "Files (selon x)", "n"), _c("n_ay", "Rangées (selon y)", "n"),
            _c("w_x", "wx — entraxe total des files", "n", un="mm"), _c("w_y", "wy — entraxe total des rangées", "n", un="mm"),
            _c("ancr", "Tige", "s", BOULONS), _c("classe", "Classe", "s", CLASSES),
            _c("typ_a", "Type d'ancrage", "s", [TYPE_TETE, TYPE_CROSSE]),
            _c("hef", "hef — profondeur d'ancrage", "n", un="mm"),
            _c("d_h", "dh — diamètre de la tête / plaque", "n", un="mm", si=tete)]),
    dict(t="Béton et scellement",
         f=[_c("beton", "Classe de béton", "s", BETONS), _c("fiss", "Béton fissuré", "s", OUI_NON),
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
            _c("g_Mc", "γMc (EN 1992-4)", "n"), _c("C_f", "Cf,d — frottement platine / mortier", "n")]),
    dict(t="Acier personnalisé", si=acier_perso,
         f=[_c("fy_u", "fy", "n", un="MPa"), _c("fu_u", "fu", "n", un="MPa"), _c("bw_u", "βw", "n")]),
]

COURT = {"bp_u": "bp", "lp_u": "lp", "tp_u": "tp", "a_w": "a", "w_x": "wx", "w_y": "wy",
         "hef": "hef", "d_h": "dh", "t_g": "tg", "f_g": "fg", "N_Ed": "NEd", "Mx_Ed": "Mx", "My_Ed": "My",
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
    inconnues sont ignorées)."""
    d = defaults()
    if not isinstance(objet, dict):
        raise ValueError("Structure JSON inattendue")
    for k in d:
        if k in objet:
            d[k] = objet[k]
    return d

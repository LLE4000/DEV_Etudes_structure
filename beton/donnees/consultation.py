# -*- coding: utf-8 -*-
"""Les rubriques du module « Données béton » : ce que chaque tableau MONTRE, lu à sa source — jamais recopié.

    classes()   les classes de béton du bureau (``beton_classes.json``, lu par ``beton/poutre/entree.classes_beton``),
                avec, pour chacune, les grandeurs du tableau 3.1 de l'EN 1992-1-1 (``ec2.py``) et celles de la méthode du
                bureau (α_b, μ du fichier ; τ_adm des trois bandes, calculés par ``beton/poutre/moteur.materiaux``)
    aciers()    les nuances d'acier des modules béton (``beton/poutre/entree.NUANCES``), f_yd de la méthode du bureau
                (γ_s = 1,5, le défaut de ``beton/poutre/entree``) et de l'EN 1992-1-1 (γ_s = 1,15), f_yd calculé par
                le MÊME moteur (``materiaux``) ; E_s et ε_yd (``ec2.py``)

Le tableau des armatures (``beton/armatures/tableau.py``), les tables d'enrobage (``beton/enrobage/donnees.py``) et les
ancrages (``beton/donnees/ancrages.py``) sont lus directement par l'écran : ils sont déjà des données nommées.

Chaque colonne est décrite par ``COLONNES_*`` : (clé, symbole, nom, unité, décimales de la source, groupe). Les
décimales sont celles du tableau publié (f_ctm à 0,1 MPa, E_cm au GPa, comme le tableau 3.1) ou de la donnée du
bureau (α_b à 0,01, μ à 0,0001, comme ``beton_classes.json``).
"""
import copy
import functools

from beton.armatures import tableau as T
from beton.poutre import entree as PE
from beton.poutre import moteur as PM

from . import ec2 as EC

FICHIER_CLASSES = "beton_classes.json"
SOURCE_BUREAU = "méthode du bureau (modules Poutre, Dalle, Prédalle)"

# --------------------------------------------------------------------------------------------- classes de béton
GROUPES_CLASSES = (("ec2", "EN 1992-1-1 · tableau 3.1"), ("bureau", "Méthode du bureau"))
COLONNES_CLASSES = (
    ("fck", "f_ck", "résistance caractéristique sur cylindre", "MPa", 0, "classe"),
    ("fck_cube", "f_ck,cube", "résistance caractéristique sur cube", "MPa", 0, "classe"),
    ("fcm", "f_cm", "résistance moyenne en compression", "MPa", 0, "ec2"),
    ("fctm", "f_ctm", "résistance moyenne en traction", "MPa", 1, "ec2"),
    ("fctk_005", "f_ctk,0,05", "traction caractéristique, fractile 5 %", "MPa", 1, "ec2"),
    ("fctk_095", "f_ctk,0,95", "traction caractéristique, fractile 95 %", "MPa", 1, "ec2"),
    ("Ecm", "E_cm", "module d'élasticité sécant", "GPa", 0, "ec2"),
    ("alpha_b", "α_b", "coefficient de la hauteur utile minimale", "", 2, "bureau"),
    ("mu_a500", "μ_B500", "moment réduit limite, acier B500", "", 4, "bureau"),
    ("mu_a400", "μ_B400", "moment réduit limite, acier B400", "", 4, "bureau"),
    ("tau_1", "τ_adm,I", "tranchant admissible sans étriers", "N/mm²", 2, "bureau"),
    ("tau_2", "τ_adm,II", "tranchant admissible avec étriers", "N/mm²", 2, "bureau"),
    ("tau_4", "τ_adm,IV", "tranchant admissible, barres inclinées et étriers", "N/mm²", 2, "bureau"),
)


def _poutre(classe: str, fyk: int, gamma_s: float) -> dict:
    """Une poutre neuve du moteur des poutres, avec la classe, la nuance et le γ_s demandés (normalisée par lui)."""
    return PE.normaliser({**PE.nouvelle_poutre(), "beton": classe, "fyk": fyk, "gamma_s": gamma_s})


def materiaux(classe: str, fyk: int, gamma_s: float) -> dict:
    """Les matériaux que le moteur des poutres lit et calcule pour cette classe, cette nuance et ce γ_s
    (``beton/poutre/moteur.materiaux`` : f_ck, f_ck,cube, f_ctm, α_b, μ, f_yd, τ_adm,I/II/IV)."""
    return PM.materiaux(_poutre(classe, fyk, gamma_s))


@functools.lru_cache(maxsize=1)
def _classes() -> tuple:
    out = []
    gamma_bureau = PE.DEFAUTS_POUTRE["gamma_s"]
    for nom, d in PE.classes_beton().items():
        fck = float(d["fck"])
        m = materiaux(nom, PE.NUANCES[-1], gamma_bureau)
        out.append({"nom": nom, "fck": fck, "fck_cube": float(d["fck_cube"]),
                    "fcm": EC.fcm(fck), "fctm": EC.fctm(fck), "fctk_005": EC.fctk_005(fck), "fctk_095": EC.fctk_095(fck),
                    "Ecm": EC.Ecm(fck),
                    "alpha_b": d["alpha_b"], "mu_a500": d["mu_a500"], "mu_a400": d["mu_a400"],
                    "tau_1": m["tau_1"], "tau_2": m["tau_2"], "tau_4": m["tau_4"],
                    "fctm_bureau": m["fctm"],
                    "tau_lim_fichier": d.get("tau_lim")})
    return tuple(out)


def classes() -> list:
    """Les classes de béton du bureau, dans l'ordre du fichier : une ligne par classe, une valeur par colonne de
    ``COLONNES_CLASSES`` (non arrondie), plus ``fctm_bureau`` (le f_ctm que le moteur des poutres calcule : le même que
    celui du tableau 3.1 jusqu'à C50/60) et ``tau_lim_fichier`` (la valeur « tau_lim » du fichier, qu'AUCUN moteur ne
    lit : les τ_adm sont recalculés depuis f_ck,cube). Une copie : la modifier ne touche à rien."""
    return copy.deepcopy(list(_classes()))


def noms_classes() -> list:
    return [c["nom"] for c in _classes()]


# --------------------------------------------------------------------------------------------- aciers d'armature
GAMMA_S_BUREAU = PE.DEFAUTS_POUTRE["gamma_s"]       # « Coefficient acier ELS » des modules béton (défaut 1,5)
COLONNES_ACIERS = (
    ("fyk", "f_yk", "limite d'élasticité caractéristique", "MPa", 0, "nuance"),
    ("fyd_bureau", "f_yd", "méthode du bureau, γ_s = {g_bureau}", "MPa", 0, "bureau"),
    ("fyd_ec2", "f_yd", "EN 1992-1-1 (ELU), γ_s = {g_ec2}", "MPa", 0, "ec2"),
    ("Es", "E_s", "module d'élasticité", "GPa", 0, "ec2"),
    ("eps_yd", "ε_yd", "allongement à f_yd (EN 1992-1-1)", "‰", 2, "ec2"),
)


@functools.lru_cache(maxsize=1)
def _aciers() -> tuple:
    out = []
    classe = PE.BETON_DEFAUT                         # la classe n'entre pas dans f_yd (materiaux la demande)
    for n in PE.NUANCES:
        fb = materiaux(classe, n, GAMMA_S_BUREAU)["fyd"]
        fe = materiaux(classe, n, EC.GAMMA_S)["fyd"]
        out.append({"nom": f"B{n}", "nuance": n, "fyk": float(n), "fyd_bureau": fb, "fyd_ec2": fe, "Es": EC.E_S,
                    "eps_yd": EC.eps_yd(fe), "gamma_bureau": GAMMA_S_BUREAU, "gamma_ec2": EC.GAMMA_S,
                    "masse_volumique": T.MASSE_VOLUMIQUE})
    return tuple(out)


def aciers() -> list:
    """Les nuances d'acier des modules béton (B400, B500) : f_yk, f_yd du bureau (γ_s = 1,5) et de l'EN 1992-1-1
    (γ_s = 1,15) — tous deux par ``beton/poutre/moteur.materiaux`` —, E_s, ε_yd et la masse volumique du tableau des
    armatures du bureau. Une copie."""
    return copy.deepcopy(list(_aciers()))

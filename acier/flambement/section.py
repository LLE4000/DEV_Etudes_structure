# -*- coding: utf-8 -*-
"""Un profilé du catalogue vu par le flambement : son aire, ses deux axes de flambement (inertie, rayon de giration),
ses parois comprimées (c, t, type, nombre) et sa fabrication (laminé, creux fini à chaud ou formé à froid, U, cornière).

Toutes les grandeurs viennent du catalogue (acier/donnees/profiles.json, source unique) : c/t des parois y est déjà
écrit (cf_tf, cw_tw, définitions du Tableau 5.2) ; une seule exception, la paroi des tubes RECTANGULAIRES, prise à
b − 3t comme l'écrit l'EN 1993-1-5 §4.4(2) (le catalogue y met, pour les formés à froid, la largeur plate nominale,
plus favorable). Unités du catalogue : mm, cm², cm⁴, cm.

Cornières : elles flambent autour de leurs axes PRINCIPAUX ; l'axe « y » du module y est l'axe fort u-u, l'axe « z »
l'axe faible v-v (I_u, I_v, i_u, i_v du catalogue).
"""
from typing import Dict, List, Optional

from acier.profiles import catalogue as CAT

# les familles du catalogue que le module calcule, et ce qu'elles sont pour le Tableau 6.2
FABRICATION = {
    "IPE": "lamine_I", "IPE A": "lamine_I", "IPE O": "lamine_I", "IPN": "lamine_I", "HEA": "lamine_I", "HEB": "lamine_I",
    "HEM": "lamine_I", "HEAA": "lamine_I", "HL": "lamine_I", "HD": "lamine_I", "HP": "lamine_I",
    "UPE": "U", "UPN": "U", "L": "L", "LI": "L",
    "CHS": "creux_chaud", "SHS": "creux_chaud", "RHS": "creux_chaud",
    "CHS-F": "creux_froid", "SHS-F": "creux_froid", "RHS-F": "creux_froid",
}
FAMILLES = tuple(FABRICATION)
B_MOINS_3T = 3.0               # largeur d'une paroi de tube rectangulaire : b − 3·t (EN 1993-1-5 §4.4(2))
NOMS_AXES = {"lamine_I": ("y-y", "z-z"), "U": ("y-y", "z-z"), "creux_chaud": ("y-y", "z-z"), "creux_froid": ("y-y", "z-z"),
             "L": ("u-u", "v-v")}


def profil(nom: str) -> Dict:
    """L'enregistrement du catalogue ; ValueError si le nom est inconnu ou hors des familles du module."""
    p = CAT.par_nom().get(nom)
    if p is None:
        raise ValueError(f"profilé inconnu du catalogue : « {nom} »")
    if p["famille"] not in FABRICATION:
        raise ValueError(f"famille non prise en charge : {p['famille']}")
    return p


def noms(familles: Optional[List[str]] = None) -> List[str]:
    """Les profilés des familles du module, dans l'ordre du catalogue."""
    fams = FAMILLES if familles is None else familles
    return [p["nom"] for p in CAT.profils() if p["famille"] in fams]


def epaisseur(p: Dict) -> float:
    """L'épaisseur nominale qui fixe f_y (Tableau 3.1) : la plus forte paroi (t_f, t_w ou t), mm."""
    return max(float(p.get(k) or 0.0) for k in ("tf", "tw", "t"))


def parois(p: Dict) -> List[Dict]:
    """Les parois comprimées : {nom, type (interne | console | tube | corniere), c (mm), t (mm), c_t, n (nombre)}.
    c/t est celui du catalogue (cf_tf : semelle, grande aile, paroi b ; cw_tw : âme, petite aile, paroi h), sauf pour
    les tubes rectangulaires : b − 3t (voir plus bas)."""
    fab = FABRICATION[p["famille"]]
    if fab == "lamine_I":
        return [{"nom": "semelle", "type": "console", "t": p["tf"], "c_t": p["cf_tf"], "c": p["cf_tf"] * p["tf"], "n": 4},
                {"nom": "âme", "type": "interne", "t": p["tw"], "c_t": p["cw_tw"], "c": p["cw_tw"] * p["tw"], "n": 1}]
    if fab == "U":
        return [{"nom": "semelle", "type": "console", "t": p["tf"], "c_t": p["cf_tf"], "c": p["cf_tf"] * p["tf"], "n": 2},
                {"nom": "âme", "type": "interne", "t": p["tw"], "c_t": p["cw_tw"], "c": p["cw_tw"] * p["tw"], "n": 1}]
    if p["famille"].startswith("CHS"):
        return [{"nom": "paroi", "type": "tube", "t": p["t"], "c_t": p["cf_tf"], "c": p["d"], "n": 1}]
    if fab in ("creux_chaud", "creux_froid"):
        # tube rectangulaire : la largeur de paroi est b − 3t (EN 1993-1-5 §4.4(2) « b − 3t for flanges of RHS »), finis
        # à chaud COMME formés à froid. Le catalogue écrit pour les formés à froid la largeur plate nominale b − 2·r_o
        # (r_o = 2t à 3t, EN 10219-2) : jusqu'à 9 % d'aire efficace en trop, et 63 tubes une classe trop bas. Pour les
        # finis à chaud (r_o = 1,5t), les deux sont égaux.
        t = p["t"]
        cb, ch = p["b"] - B_MOINS_3T * t, p["h"] - B_MOINS_3T * t
        return [{"nom": "paroi b", "type": "interne", "t": t, "c_t": cb / t, "c": cb, "n": 2, "cote": ("b", p["b"])},
                {"nom": "paroi h", "type": "interne", "t": t, "c_t": ch / t, "c": ch, "n": 2, "cote": ("h", p["h"])}]
    # cornière : les deux ailes, en console ; la largeur de référence de l'EN 1993-1-5 §4.4(2) est h (la grande
    # aile) pour les cornières à ailes égales ET inégales : c/t = h/t pour les deux ailes ; c (l'aire retirée) est la
    # longueur de chaque aile
    return [{"nom": "aile h", "type": "console", "t": p["t"], "c_t": p["cf_tf"], "c": p["h"], "n": 1},
            {"nom": "aile b", "type": "console", "t": p["t"], "c_t": p["cf_tf"], "c": p["b"], "n": 1}]


def axes(p: Dict) -> Dict[str, Dict]:
    """Les deux axes de flambement : {"y": {nom, I (cm⁴), i (cm)}, "z": {…}} — u-u et v-v pour une cornière."""
    fab = FABRICATION[p["famille"]]
    ny, nz = NOMS_AXES[fab]
    if fab == "L":
        return {"y": {"nom": ny, "I": p["Iu"], "i": p["iu"]}, "z": {"nom": nz, "I": p["Iv"], "i": p["iv"]}}
    return {"y": {"nom": ny, "I": p["Iy"], "i": p["iy"]}, "z": {"nom": nz, "I": p["Iz"], "i": p["iz"]}}


def symetrique(p: Dict) -> bool:
    """Bi-symétrique (I, H, creux) : centre de gravité et centre de cisaillement confondus."""
    return FABRICATION[p["famille"]] in ("lamine_I", "creux_chaud", "creux_froid")

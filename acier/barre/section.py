# -*- coding: utf-8 -*-
"""Un profilé vu par la vérification d'une barre : ses dimensions et ses propriétés en N et mm, sa FORME et sa
FABRICATION — lus dans le catalogue (acier/donnees/profiles.json, source unique) ou saisis.

    {"nom", "famille", "forme", "fabrication", "axes": ("y-y", "z-z"),
     "h", "b", "tw", "tf", "r", "t", "d"            (mm ; None quand la forme n'en a pas)
     "A" (mm²), "Iy", "Iz" (mm⁴), "iy", "iz" (mm), "Wel_y", "Wel_z", "Wpl_y", "Wpl_z" (mm³),
     "It" (mm⁴), "Iw" (mm⁶), "Avz", "Avy" (mm²), "masse" (kg/m),
     "cf_tf", "cw_tw"                                (c/t des parois, définitions du tableau 5.2 de l'EN 1993-1-1)}

FORME : « I » (I et H à deux axes de symétrie), « tube_rond », « tube_carre », « tube_rect », « U », « L », « autre ».
FABRICATION : celle qui choisit la courbe de flambement — « lamine », « creux_chaud », « creux_froid » pour le
catalogue ; une section saisie peut en dire d'autres (``FABRICATIONS``).

L'axe « y » est l'axe FORT, « z » l'axe faible. Une cornière flambe autour de ses axes PRINCIPAUX : son « y » est u-u,
son « z » v-v (I_u, I_v, i_u, i_v du catalogue), comme dans acier/flambement/section.py.
"""
from typing import Dict, Optional

from acier.profiles import catalogue as CAT

from .resultat import fr

# famille du catalogue → (forme, fabrication)
FAMILLES: Dict[str, tuple] = {
    "IPE": ("I", "lamine"), "IPE A": ("I", "lamine"), "IPE O": ("I", "lamine"), "IPN": ("I", "lamine"),
    "HEA": ("I", "lamine"), "HEB": ("I", "lamine"), "HEM": ("I", "lamine"), "HEAA": ("I", "lamine"),
    "HL": ("I", "lamine"), "HD": ("I", "lamine"), "HP": ("I", "lamine"),
    "UPE": ("U", "lamine"), "UPN": ("U", "lamine"), "L": ("L", "lamine"), "LI": ("L", "lamine"),
    "CHS": ("tube_rond", "creux_chaud"), "SHS": ("tube_carre", "creux_chaud"), "RHS": ("tube_rect", "creux_chaud"),
    "CHS-F": ("tube_rond", "creux_froid"), "SHS-F": ("tube_carre", "creux_froid"), "RHS-F": ("tube_rect", "creux_froid"),
}
FORMES = ("I", "tube_rond", "tube_carre", "tube_rect", "U", "L", "autre")
# les fabrications que la norme historique distingue pour le choix de la courbe (docs/acier/BARRES_ACIER.md § 4.2)
FABRICATIONS = ("lamine", "soude_oxycoupe", "soude_larges_plats", "lamine_semelles_additionnelles", "recuit",
                "creux_chaud", "creux_froid", "caisson", "caisson_recuit")
TUBES = ("tube_rond", "tube_carre", "tube_rect")
DIMENSIONS = ("h", "b", "tw", "tf", "r", "t", "d")
# (clé de la section, clé du catalogue, facteur vers N et mm) ; les axes d'une cornière sont pris à part
_PROPRIETES = (("A", "A", 1e2), ("Wpl_y", "Wpl_y", 1e3), ("Wpl_z", "Wpl_z", 1e3), ("It", "It", 1e4), ("Iw", "Iw", 1e6),
               ("Avz", "Avz", 1e2), ("Avy", "Avy", 1e2), ("masse", "masse", 1.0))
_AXES = {False: (("Iy", "Iy", 1e4), ("Iz", "Iz", 1e4), ("iy", "iy", 10.0), ("iz", "iz", 10.0), ("Wel_y", "Wel_y", 1e3),
                 ("Wel_z", "Wel_z", 1e3)),
         True: (("Iy", "Iu", 1e4), ("Iz", "Iv", 1e4), ("iy", "iu", 10.0), ("iz", "iv", 10.0), ("Wel_y", "Wel_u", 1e3),
                ("Wel_z", "Wel_v", 1e3))}
OBLIGATOIRES = ("A", "Iy", "Iz", "iy", "iz")
AXES_INVERSES = "I ou H avec I_y < I_z ({Iy} < {Iz} cm⁴) : axes inversés ? Verdict bloqué jusqu'à la correction."


def _valeur(p: Dict, cle: str, facteur: float) -> Optional[float]:
    v = p.get(cle)
    return None if v is None else float(v) * facteur


def du_catalogue(nom: str) -> Dict:
    """La section d'un profilé du catalogue ; ValueError si le nom est inconnu ou la famille hors du module."""
    p = CAT.par_nom().get(nom)
    if p is None:
        raise ValueError(f"profilé inconnu du catalogue : « {nom} »")
    if p["famille"] not in FAMILLES:
        raise ValueError(f"famille non prise en charge : {p['famille']}")
    forme, fabrication = FAMILLES[p["famille"]]
    s = {"nom": p["nom"], "famille": p["famille"], "forme": forme, "fabrication": fabrication,
         "axes": ("u-u", "v-v") if forme == "L" else ("y-y", "z-z")}
    for k in DIMENSIONS:
        s[k] = _valeur(p, k, 1.0)
    for cle, src, f in _PROPRIETES + _AXES[forme == "L"]:
        s[cle] = _valeur(p, src, f)
    # c/t des parois : ceux du catalogue (semelle ou paroi b ; âme ou paroi h) — une seule source avec acier/flambement
    s["cf_tf"], s["cw_tw"] = _valeur(p, "cf_tf", 1.0), _valeur(p, "cw_tw", 1.0)
    return s


def saisie(forme: str, fabrication: str, nom: str = "section saisie", **proprietes) -> Dict:
    """Une section dont on donne les propriétés (N et mm) : le mode « efforts imposés » des classeurs, une section
    hors catalogue. ``A``, ``Iy``, ``Iz``, ``iy``, ``iz`` sont obligatoires ; le reste vaut None s'il manque (la
    vérification qui en a besoin le dit)."""
    if forme not in FORMES:
        raise ValueError(f"forme inconnue : {forme!r}")
    if fabrication not in FABRICATIONS:
        raise ValueError(f"fabrication inconnue : {fabrication!r}")
    connues = set(DIMENSIONS) | {c for c, _, _ in _PROPRIETES + _AXES[False]} | {"A_ame", "A_ailes", "cf_tf", "cw_tw"}
    inconnues = sorted(set(proprietes) - connues)
    if inconnues:
        raise ValueError(f"propriété(s) inconnue(s) : {', '.join(inconnues)}")
    s = {"nom": nom, "famille": None, "forme": forme, "fabrication": fabrication,
         "axes": ("u-u", "v-v") if forme == "L" else ("y-y", "z-z")}
    for k in sorted(connues):
        v = proprietes.get(k)
        s[k] = None if v is None else float(v)
    manquent = [k for k in OBLIGATOIRES if not s[k] or s[k] <= 0]
    if manquent:
        raise ValueError(f"propriété(s) manquante(s) ou nulle(s) : {', '.join(manquent)}")
    # un I ou un H dont l'axe fort est le moins raide : axes inversés à l'import (brief § 10, P41, test 15 du § 11.5) — le
    # verdict est bloqué jusqu'à la correction
    if forme == "I" and s["Iy"] < s["Iz"]:
        raise ValueError(AXES_INVERSES.replace("{Iy}", fr(s["Iy"] / 1e4, 1)).replace("{Iz}", fr(s["Iz"] / 1e4, 1)))
    # c/t des parois d'un I dont on donne les dimensions : semelle en console (b − t_w − 2r)/2, âme h − 2·t_f − 2r
    # (r = 0 : une section soudée, parois prises entre nus — du côté sûr)
    if forme == "I" and all(s.get(k) for k in ("h", "b", "tw", "tf")):
        r = s.get("r") or 0.0
        if s["cf_tf"] is None:
            s["cf_tf"] = (s["b"] - s["tw"] - 2.0 * r) / 2.0 / s["tf"]
        if s["cw_tw"] is None:
            s["cw_tw"] = (s["h"] - 2.0 * s["tf"] - 2.0 * r) / s["tw"]
    return s


def hauteur_ame(s: Dict) -> Optional[float]:
    """h_w = h − 2·t_f (mm), la hauteur de l'âme entre semelles d'un I — celle d'un PRS est saisie (acier/barre/prs.py) ;
    None pour une autre forme."""
    if s.get("prs"):
        return s["prs"]["hw"]
    if s["forme"] != "I" or not s.get("h") or not s.get("tf"):
        return None
    return s["h"] - 2.0 * s["tf"]


def soude(s: Dict) -> bool:
    """Une section en I SOUDÉE (EN 1993-1-1 tableaux 6.2, 6.4, 6.5 ; § 6.2.6(3) d) et e)) : un PRS, que ses bords soient
    saisis ou non, ou un I saisi « soudé »."""
    return bool(s.get("prs")) or s.get("fabrication") in ("soude_oxycoupe", "soude_larges_plats")


def cote(s: Dict, ou: str) -> float:
    """La cote (mm) AU-DESSUS du centre de cisaillement où se repère une charge ou un maintien (brief § 7.7) : « haut »,
    « bas » (les faces extérieures des semelles : z_g = ±h/2 d'un I bisymétrique) ; « semelle_haut », « semelle_bas »
    (leurs plans moyens : ±(h − t_f)/2) ; « centre » : 0. Un PRS monosymétrique : depuis SON centre de cisaillement."""
    if ou == "centre":
        return 0.0
    if s.get("prs"):
        from . import prs as PR
        return PR.cotes(s)[{"haut": "face_haut", "bas": "face_bas"}.get(ou, ou)]
    h = s.get("h") or 0.0
    if ou in ("haut", "bas"):
        return (1.0 if ou == "haut" else -1.0) * h / 2.0
    demi = (h - (s.get("tf") or s.get("t") or 0.0)) / 2.0
    return demi if ou == "semelle_haut" else -demi

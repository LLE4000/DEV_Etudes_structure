# -*- coding: utf-8 -*-
"""La bibliothèque des sections d'un garde-corps : les profilés creux du catalogue
(``acier/donnees/profiles.json`` : CHS, SHS, RHS finis à chaud et formés à froid), les
plats et les ronds pleins des séries de stock, et les sections NON STANDARD définies par
leurs dimensions (tube rond, caisson, plat, rond) calculées par ``acier.profiles.familles``
— la même fonction que le catalogue livré, donc les mêmes nombres (source unique).

Une section est un dictionnaire aux unités du catalogue (mm ; cm², cm⁴, cm³ ; kg/m) :
``nom``, ``famille``, ``forme``, ``dims``, ``standard`` (du catalogue ou d'une série de
stock) et les propriétés A, Iy, Wel_y, Wpl_y, Iz, Wel_z, Avz, Avy, masse, h/b/t/d.
AXE FORT = le sens de h : un effort dans le sens de h fléchit la section autour de y-y
(Wel_y, Iy, aire de cisaillement Avz). Pour une section pleine, l'aire de cisaillement des
« vieilles formules » est l'aire entière (τ = V/A).

Une spécification de section (ce que l'état enregistre) est ``{"nom": "CHS 48.3x3.2"}``
pour une section standard, ou ``{"famille": "PLAT", "dims": {"h": 50, "b": 10}}`` pour une
section non standard.
"""
import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from acier.profiles import catalogue as CAT
from acier.profiles import familles as F

CREUX_CHAUD = ("CHS", "SHS", "RHS")
CREUX_FROID = ("CHS-F", "SHS-F", "RHS-F")
FAMILLES_CREUX = CREUX_CHAUD + CREUX_FROID
# Les séries de stock des plats (EN 10058) et des ronds (EN 10060) usuelles en garde-corps :
# largeur h (dans le sens de la flexion) × épaisseur b.
PLATS_STOCK = [(h, b) for h in (30, 40, 50, 60, 70, 80, 100, 120) for b in (5, 6, 8, 10, 12, 15, 20) if b < h]
RONDS_STOCK = [10, 12, 14, 16, 18, 20, 25, 30]
# Familles non standard (dimensions libres) : clé → (libellé, dimensions)
NON_STANDARD = {"TUBE": ("tube rond", ("d", "t")), "CAISSON": ("tube rectangulaire", ("h", "b", "t")),
                "PLAT": ("plat", ("h", "b")), "ROND": ("rond plein", ("d",))}
GROUPES = {"creux": "profilés creux (CHS, SHS, RHS)", "froid": "profilés creux formés à froid", "plats": "plats", "ronds": "ronds pleins"}
LIBELLE_FAMILLE = {"CHS": "tubes ronds", "SHS": "tubes carrés", "RHS": "tubes rectangulaires", "CHS-F": "tubes ronds (froid)",
                   "SHS-F": "tubes carrés (froid)", "RHS-F": "tubes rectangulaires (froid)", "PLAT": "plats", "ROND": "ronds",
                   "TUBE": "tube rond", "CAISSON": "caisson"}


def _dims_de(rec: Dict) -> Dict[str, float]:
    return {k: float(rec[k]) for k in F.FAMILLES[rec["famille"]]["dims"] if rec.get(k) is not None}


def _normaliser(rec: Dict, nom: str, famille: str, dims: Dict[str, float], standard: bool) -> Dict:
    """Le dictionnaire de section : propriétés utiles, aire de cisaillement entière pour une
    section pleine, dimensions au premier niveau."""
    forme = F.FAMILLES[famille]["forme"]
    A = float(rec["A"])
    plein = forme in ("plein_rect", "plein_rond")
    sec = {"nom": nom, "famille": famille, "forme": forme, "dims": dict(dims), "standard": bool(standard),
           "groupe": "plein" if plein else "creux",
           "A": A, "Iy": float(rec["Iy"]), "Wel_y": float(rec["Wel_y"]), "Wpl_y": float(rec.get("Wpl_y") or 0.0),
           "Iz": float(rec["Iz"]), "Wel_z": float(rec["Wel_z"]), "Wpl_z": float(rec.get("Wpl_z") or 0.0),
           "Avz": float(rec["Avz"]) if rec.get("Avz") else A, "Avy": float(rec["Avy"]) if rec.get("Avy") else A,
           "masse": float(rec["masse"]), "It": float(rec.get("It") or 0.0)}
    for k in ("h", "b", "t", "d"):
        sec[k] = float(dims[k]) if k in dims else None
    if forme == "creux_rond" or forme == "plein_rond":
        sec["h"] = sec["b"] = sec["d"]
    return sec


def _nom_stock(famille: str, dims: Dict[str, float]) -> str:
    if famille == "PLAT":
        return f"Plat {dims['h']:g}x{dims['b']:g}"
    if famille == "ROND":
        return f"Rond Ø{dims['d']:g}"
    if famille == "TUBE":
        return f"Tube Ø{dims['d']:g}x{dims['t']:g}"
    return f"Caisson {dims['h']:g}x{dims['b']:g}x{dims['t']:g}"


@lru_cache(maxsize=4096)
def _calcul(famille: str, cle: tuple) -> Dict:
    dims = dict(cle)
    rec = F.calculer(famille, dims, torsion_exacte=False)
    return _normaliser(rec, _nom_stock(famille, dims), famille, dims, False)


def section_libre(famille: str, dims: Dict[str, float]) -> Dict:
    """Une section calculée de ses dimensions (non standard par défaut ; les séries de stock
    la marquent standard)."""
    if famille not in NON_STANDARD:
        raise ValueError(f"famille non standard inconnue : {famille!r}")
    d = {}
    for k in NON_STANDARD[famille][1]:
        v = float(dims.get(k, 0) or 0)
        if v <= 0:
            raise ValueError(f"dimension {k} manquante pour la section {NON_STANDARD[famille][0]}")
        d[k] = v
    if "t" in d and ("d" in d and 2 * d["t"] >= d["d"] or "h" in d and 2 * d["t"] >= min(d["h"], d["b"])):
        raise ValueError("épaisseur de paroi t trop grande pour la section")
    return dict(_calcul(famille, tuple(sorted(d.items()))))


@lru_cache(maxsize=1)
def _catalogue_creux() -> Dict[str, Dict]:
    out = {}
    for rec in CAT.profils():
        if rec["famille"] in FAMILLES_CREUX:
            out[rec["nom"]] = _normaliser(rec, rec["nom"], rec["famille"], _dims_de(rec), True)
    return out


@lru_cache(maxsize=1)
def _stock_pleins() -> Dict[str, Dict]:
    out = {}
    for h, b in PLATS_STOCK:
        s = section_libre("PLAT", {"h": h, "b": b})
        s["standard"] = True
        out[s["nom"]] = s
    for d in RONDS_STOCK:
        s = section_libre("ROND", {"d": d})
        s["standard"] = True
        out[s["nom"]] = s
    return out


def standards() -> Dict[str, Dict]:
    """Toutes les sections standard par nom : creux du catalogue, plats et ronds de stock."""
    out = dict(_catalogue_creux())
    out.update(_stock_pleins())
    return out


def section(spec: Dict) -> Dict:
    """La section d'une spécification : ``{"nom": …}`` (standard) ou ``{"famille", "dims"}`` (libre)."""
    if not isinstance(spec, dict):
        raise ValueError("spécification de section invalide")
    nom = spec.get("nom")
    if nom:
        std = standards()
        if nom in std:
            return dict(std[nom])
        raise ValueError(f"section inconnue de la bibliothèque : {nom}")
    return section_libre(spec.get("famille", "PLAT"), spec.get("dims") or {})


def spec_de(sec: Dict) -> Dict:
    """La spécification à enregistrer pour une section."""
    if sec.get("standard"):
        return {"nom": sec["nom"]}
    return {"famille": sec["famille"], "dims": dict(sec["dims"])}


def groupe_de(sec: Dict) -> str:
    f = sec["famille"]
    if f in CREUX_CHAUD or f in ("TUBE", "CAISSON"):
        return "creux"
    if f in CREUX_FROID:
        return "froid"
    if f in ("PLAT",):
        return "plats"
    return "ronds"


def catalogue(groupes: Optional[Sequence[str]] = None, masse_max: Optional[float] = None,
              dim_max: Optional[float] = None) -> List[Dict]:
    """Les sections standard des groupes demandés (``GROUPES`` ; None = tous), bornées en masse
    (kg/m) et en encombrement (mm), triées par masse croissante."""
    out = []
    for s in standards().values():
        if groupes is not None and groupe_de(s) not in groupes:
            continue
        if masse_max is not None and s["masse"] > masse_max:
            continue
        if dim_max is not None and max(s["h"] or 0, s["b"] or 0) > dim_max:
            continue
        out.append(dict(s))
    out.sort(key=lambda s: (s["masse"], s["nom"]))
    return out


def libelle(sec: Dict) -> str:
    """« CHS 48.3x3.2 · 3,56 kg/m » ; une section libre dit « non standard »."""
    txt = f"{sec['nom']} · {sec['masse']:.2f} kg/m".replace(".", ",")
    if not sec.get("standard"):
        txt += " · non standard"
    return txt


def dimensions_texte(sec: Dict) -> str:
    """Les dimensions en clair : « Ø 48,3 × 3,2 », « 60 × 40 × 3 », « 50 × 10 »."""
    d = sec["dims"]
    if sec["forme"] in ("creux_rond", "plein_rond"):
        return (f"Ø {d['d']:g}" + (f" × {d['t']:g}" if "t" in d else "")).replace(".", ",")
    if sec["forme"] == "creux_rect":
        return f"{d['h']:g} × {d['b']:g} × {d['t']:g}".replace(".", ",")
    return f"{d['h']:g} × {d['b']:g}".replace(".", ",")


def contours(sec: Dict):
    """Les contours (mm) de la section, pour les dessins — ceux du calcul."""
    return F.contours(sec["famille"], sec["dims"])


def nb_standards() -> int:
    return len(standards())


def masse_lineaire(sec: Dict, longueur_mm: float) -> float:
    """Masse (kg) d'une longueur de section."""
    return sec["masse"] * longueur_mm / 1000.0


def encombrement(sec: Dict) -> float:
    """Le plus grand côté (mm) : diamètre, hauteur ou largeur."""
    return max(float(sec.get("h") or 0), float(sec.get("b") or 0), float(sec.get("d") or 0))


def est_creux(sec: Dict) -> bool:
    return sec["forme"] in ("creux_rond", "creux_rect")


def pi_d(sec: Dict) -> float:
    """Le diamètre équivalent de l'encombrement (pour les dessins de section)."""
    return encombrement(sec) if sec["forme"] in ("creux_rond", "plein_rond") else math.hypot(sec["h"] or 0, sec["b"] or 0)

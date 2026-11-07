# -*- coding: utf-8 -*-
"""Taux d'armature — la GÉOMÉTRIE de la section : l'âme, les talons, l'aire, le polygone du béton et la position des
barres de chaque lit. Chaque fonction reprend l'expression de ``modules/taux_armature.py`` (BeamGeometry,
build_geometry) DANS LE MÊME ORDRE D'OPÉRATIONS : les flottants sont identiques au bit près.

Unités du module : cm ; origine au milieu de l'âme, fibre inférieure y = 0 ; les talons sont en bas, vers l'extérieur.
Une géométrie est un dict : ``b``, ``h``, ``c`` (enrobage), ``bg``, ``hg`` (talon gauche), ``bd``, ``hd`` (talon droit).
"""
from typing import List, Optional, Tuple

from . import entree as EN

GROUPES_BAS = ("ame_inf", "talon_g_inf", "talon_d_inf")
GROUPES_HAUT = ("ame_sup", "talon_g_sup", "talon_d_sup")


def talons(poutre: dict) -> Tuple[float, float, float, float]:
    """Les talons que le module CALCULE (build_geometry) : il lit le libellé de la forme — « gauche » dans le libellé
    → talon gauche, « droit » → talon droit — avec « largeur or 10.0 », « hauteur or 15.0 ».

    DÉFAUT REPRODUIT (D2) : « Rectangulaire + deux talons » ne contient ni « gauche » ni « droit » : la section à deux
    talons est calculée SANS talon (volume, dessin), alors que l'écran montre leurs quatre champs."""
    lib = EN.FORMES[poutre["forme"]]
    tg, td = poutre["talon_gauche"], poutre["talon_droit"]
    if "gauche" in lib:
        bg = tg["b"] or 10.0
        hg = tg["h"] or 15.0
    else:
        bg = 0.0
        hg = 0.0
    if "droit" in lib:
        bd = td["b"] or 10.0
        hd = td["h"] or 15.0
    else:
        bd = 0.0
        hd = 0.0
    return bg, hg, bd, hd


def geometrie(poutre: dict) -> dict:
    bg, hg, bd, hd = talons(poutre)
    return {"b": poutre["b"], "h": poutre["h"], "c": poutre["enrobage"], "bg": bg, "hg": hg, "bd": bd, "hd": hd}


def polygone(g: dict) -> List[Tuple[float, float]]:
    """Le contour du béton (concrete_polygon) : âme et talons inférieurs, en cm."""
    b = g["b"]
    h = g["h"]
    xL = -b / 2.0
    xR = b / 2.0
    has_left = g["bg"] > 0 and g["hg"] > 0
    has_right = g["bd"] > 0 and g["hd"] > 0
    XL_ext = xL - g["bg"] if has_left else xL
    XR_ext = xR + g["bd"] if has_right else xR
    pts = [(XL_ext, 0.0)]
    if has_left:
        pts.append((XL_ext, g["hg"]))
        pts.append((xL, g["hg"]))
    pts.append((xL, h))
    pts.append((xR, h))
    if has_right:
        pts.append((xR, g["hd"]))
        pts.append((XR_ext, g["hd"]))
    pts.append((XR_ext, 0.0))
    if XR_ext != XL_ext:
        pts.append((XL_ext, 0.0))
    return pts


def aire_cm2(g: dict) -> float:
    """L'aire de la section (area_cm2) : âme, plus chaque talon défini, en cm²."""
    area = g["b"] * g["h"]
    if g["bg"] > 0 and g["hg"] > 0:
        area += g["bg"] * g["hg"]
    if g["bd"] > 0 and g["hd"] > 0:
        area += g["bd"] * g["hd"]
    return area


def bande(g: dict, pos: str) -> Tuple[Optional[float], float, float]:
    """(y de la face béton, x min, x max) d'une position (_horizontal_band) ; y None : barres latérales verticales.
    Une position de talon sur une section sans ce talon lève ValueError, comme le module (« Talon gauche non défini. »)."""
    c = g["c"]
    b = g["b"]
    xL = -b / 2.0
    xR = b / 2.0
    if pos == "ame_inf":
        y_face = 0.0
        x_min = xL + c
        x_max = xR - c
    elif pos == "ame_sup":
        y_face = g["h"]
        x_min = xL + c
        x_max = xR - c
    elif pos == "talon_g_inf":
        if g["bg"] <= 0 or g["hg"] <= 0:
            raise ValueError("Talon gauche non défini.")
        y_face = 0.0
        x_min = xL - g["bg"] + c
        x_max = xL - c
    elif pos == "talon_g_sup":
        if g["bg"] <= 0 or g["hg"] <= 0:
            raise ValueError("Talon gauche non défini.")
        y_face = g["hg"]
        x_min = xL - g["bg"] + c
        x_max = xL - c
    elif pos == "talon_d_inf":
        if g["bd"] <= 0 or g["hd"] <= 0:
            raise ValueError("Talon droit non défini.")
        y_face = 0.0
        x_min = xR + c
        x_max = xR + g["bd"] - c
    elif pos == "talon_d_sup":
        if g["bd"] <= 0 or g["hd"] <= 0:
            raise ValueError("Talon droit non défini.")
        y_face = g["hd"]
        x_min = xR + c
        x_max = xR + g["bd"] - c
    elif pos in ("lat_g", "lat_d", "lat_2"):
        y_face = None
        if pos == "lat_g":
            x_min = xL + c
            x_max = x_min
        elif pos == "lat_d":
            x_max = xR - c
            x_min = x_max
        else:
            x_min = xL + c
            x_max = xR - c
    else:
        raise ValueError(f"Position inconnue : {pos}")
    return y_face, x_min, x_max


def centres(g: dict, pos: str, n_bars: int, phi_mm: float, lit: int = 0) -> List[Tuple[float, float]]:
    """Les centres des barres d'un lit (bar_centers), en cm. ``lit`` : 0 pour le lit le plus proche de la face.

    Lits horizontaux : axe à c + Ø/2 de la face, lits espacés de Ø + 2 cm, barres réparties de x min à x max (une
    seule : au milieu). Latérales : n + 2 barres réparties de c à h − c, les deux extrêmes retirées (une seule : à
    mi-hauteur), à x = ±(b/2 − c)."""
    if n_bars <= 0:
        return []
    y_face, x_min, x_max = bande(g, pos)
    c = g["c"]
    if y_face is not None:
        r_cm = phi_mm / 20.0
        base_cover = c + r_cm
        pitch = 2.0 * r_cm + 2.0
        if pos in GROUPES_BAS:
            y = y_face + base_cover + lit * pitch
        else:
            y = y_face - (base_cover + lit * pitch)
        if n_bars == 1:
            xs = [(x_min + x_max) / 2.0]
        else:
            xs = [x_min + i * (x_max - x_min) / (n_bars - 1) for i in range(n_bars)]
        return [(x, y) for x in xs]
    y_min = c
    y_max = g["h"] - c
    if n_bars == 1:
        ys = [(y_min + y_max) / 2.0]
    else:
        n_eff = n_bars + 2
        idxs = range(1, n_eff - 1)
        ys = [y_min + i * (y_max - y_min) / (n_eff - 1) for i in idxs]
    xL = -g["b"] / 2.0 + c
    xR = g["b"] / 2.0 - c
    pts: List[Tuple[float, float]] = []
    if pos in ("lat_g", "lat_2"):
        pts.extend((xL, y) for y in ys)
    if pos in ("lat_d", "lat_2"):
        pts.extend((xR, y) for y in ys)
    return pts


def groupe(pos: str) -> str:
    """La classe des lits pour les couleurs et les codes (_position_group) : « bas », « haut », « cote »."""
    if pos in GROUPES_BAS:
        return "bas"
    if pos in GROUPES_HAUT:
        return "haut"
    return "cote"

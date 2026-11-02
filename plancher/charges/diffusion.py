# -*- coding: utf-8 -*-
"""Diffusion d'une charge concentrée dans le plancher, par formules fermées
(§ 4 de l'étude) : largeurs efficaces d'une dalle unidirectionnelle
(DAfStb Heft 240), règle du levier et parts de l'annexe C de l'EN 1168
pour des hourdis, superposition « en créneaux », lignes d'influence d'une
poutre sur deux appuis.

Ce qui vient d'un texte NON relu est un paramètre, jamais une valeur :
- ``b_m`` (largeur efficace en flexion) : la forme t_y + k·x·(1 − x/l) est
  connue, le coefficient ``k`` reste à transcrire ; tant qu'il vaut ``None``,
  la largeur en flexion est t_y (P-4, conservatif) ;
- les parts de la figure C.1 de l'EN 1168 sont FOURNIES par le bureau
  (``parts``) ; sans elles, l'élément chargé porte tout.
"""
from typing import Dict, List, Sequence, Tuple
import math

from .modele import Element


# ---------------------------------------------------------------- Heft 240
def t_y(b0: float, h1: float, h: float, l: float) -> float:
    """Largeur de contact diffusée : t_y = b0 + 2·h1 + h (diffusion à 45°
    dans le revêtement h1, puis jusqu'au plan moyen de la dalle), bornée
    par 0,8·l."""
    return min(b0 + 2.0 * h1 + h, 0.8 * l)


def b_v(ty: float, a: float, b_max: float, porte_a_faux: bool = False) -> float:
    """Largeur efficace en tranchant : t_y + 0,5·a (0,3·a en porte-à-faux),
    ``a`` = distance de la charge à l'appui considéré, bornée par la
    largeur disponible."""
    return min(ty + (0.3 if porte_a_faux else 0.5) * a, b_max)


def b_m(ty: float, x: float, l: float, k=None, b_max=None) -> float:
    """Largeur efficace en flexion : t_y tant que ``k`` n'est pas transcrit
    (P-4) ; sinon t_y + k·x·(1 − x/l)."""
    w = ty if k is None else ty + k * x * (1.0 - x / l)
    return min(w, b_max) if b_max is not None else w


# ---------------------------------------------------------------- lignes d'influence (poutre sur deux appuis)
def moment_ponctuel(P: float, xp: float, x: float, L: float) -> float:
    """Moment en ``x`` sous une charge ``P`` en ``xp`` (kN·m, positif en travée)."""
    if x <= xp:
        return P * x * (L - xp) / L
    return P * xp * (L - x) / L


def reactions_ponctuelle(P: float, xp: float, L: float) -> Tuple[float, float]:
    """Réactions (A, B) sous une charge ponctuelle."""
    return P * (L - xp) / L, P * xp / L


def moment_uniforme(w: float, x: float, L: float) -> float:
    """Moment en ``x`` sous une charge uniforme ``w`` (kN/m)."""
    return w * x * (L - x) / 2.0


def tranchant_uniforme(w: float, L: float) -> float:
    return w * L / 2.0


# ---------------------------------------------------------------- règle du levier
def parts_levier(y_c: float, b: float, elements: Sequence[Element]) -> Dict[int, float]:
    """Part de chaque élément sous une empreinte de largeur ``b`` centrée en
    ``y_c`` : proportionnelle au recouvrement — ce qui est la règle du
    levier pour une pression uniforme sur deux appuis indépendants. Une
    empreinte hors du plancher est tronquée (sa part perdue revient aux
    éléments touchés, par renormalisation)."""
    y0, y1 = y_c - b / 2.0, y_c + b / 2.0
    parts = {}
    for e in elements:
        rec = min(y1, e.y1) - max(y0, e.y0)
        if rec > 1e-12:
            parts[e.index] = rec
    if not parts:                          # largeur nulle : l'élément qui contient y_c
        for e in elements:
            if e.y0 <= y_c <= e.y1:
                return {e.index: 1.0}
        return {}
    tot = sum(parts.values())
    return {i: v / tot for i, v in parts.items()}


# ---------------------------------------------------------------- annexe C (parts fournies)
def parts_voisins(parts: Dict[int, float], i: int, n: int) -> Dict[int, float]:
    """Répartit la charge de l'élément ``i`` vers ses voisins selon les parts
    fournies ``{0: élément chargé, 1: chaque premier voisin, 2: …}``. La part
    d'un voisin absent (rive) revient à l'élément chargé — conservatif. Les
    parts sont renormalisées pour sommer à 1."""
    out: Dict[int, float] = {}
    for k, p in parts.items():
        if k == 0:
            out[i] = out.get(i, 0.0) + p
            continue
        for j in (i - k, i + k):
            if 0 <= j < n:
                out[j] = out.get(j, 0.0) + p
            else:
                out[i] = out.get(i, 0.0) + p
    tot = sum(out.values())
    return {j: v / tot for j, v in out.items()}


# ---------------------------------------------------------------- créneaux
def max_creneaux(boites: List[Tuple[float, float, float]], y_min: float, y_max: float) -> Tuple[float, float]:
    """Maximum sur la largeur de m(y) = Σ_j val_j / w_j sur chaque créneau
    [y_j − w_j/2, y_j + w_j/2], tronqué au plancher (rive : la valeur se
    concentre sur la largeur restante). Renvoie (max, y du max)."""
    if not boites:
        return 0.0, y_min
    inter = []
    for yc, w, val in boites:
        lo, hi = max(y_min, yc - w / 2.0), min(y_max, yc + w / 2.0)
        if hi - lo <= 1e-12:
            lo, hi = max(y_min, min(yc, y_max)), max(y_min, min(yc, y_max)) + 1e-9
        inter.append((lo, hi, val / (hi - lo)))
    cand = sorted({lo for lo, _, _ in inter} | {hi for _, hi, _ in inter})
    best, y_best = -math.inf, y_min
    for y in cand:
        for eps in (1e-9, -1e-9):          # juste à droite et juste à gauche de chaque arête
            yy = y + eps
            if yy < y_min or yy > y_max:
                continue
            m = sum(v for lo, hi, v in inter if lo <= yy <= hi)
            # Un plateau n'est jamais plat au bruit près (les valeurs viennent de numpy,
            # et deux versions de numpy ne font pas le même bruit) : à égalité dans la
            # tolérance relative 1e-9, c'est la PREMIÈRE arête (y le plus bas) qui nomme le
            # maximum. Sans cela, le « y du max » changeait d'interpréteur à interpréteur.
            if best == -math.inf or m > best + 1e-9 * max(1.0, abs(best)):
                best, y_best = m, y
    return best, y_best

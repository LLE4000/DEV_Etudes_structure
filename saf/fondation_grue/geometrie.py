# -*- coding: utf-8 -*-
"""Fondation de grue — la GÉOMÉTRIE en plan : polygones, pieds du châssis, plaques d'appui, plots, bandes d'intégration,
contour décalé. Bibliothèque standard seule ; les points sont des couples (x, y) en mètres.

Les règles sont celles du calcul de référence (tests/fondation_grue_ref.py), reprises à l'identique :
- un point est DANS un polygone par le lancer de rayon (un point sur le bord peut tomber d'un côté ou de l'autre) ;
- un polygone est « dans la dalle » si, rétréci de 1 mm vers son centre, tous ses sommets sont dans le contour et aucun
  de ses côtés ne coupe un côté du contour (bord sur bord admis) ;
- les pieds sont numérotés P1 (haut-gauche), P2 (haut-droite), P3 (bas-droite), P4 (bas-gauche), sens horaire vu du
  dessus, châssis non tourné ; les coins d'une plaque : haut-gauche puis sens horaire (avant rotation), comme le fichier
  SAF du bureau.
"""
import math
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]
TOLERANCE = 1e-3                 # 1 mm : bord sur bord admis, deux nœuds confondus


def dans_polygone(x: float, y: float, poly: Sequence[Point]) -> bool:
    """Le point (x, y) est-il dans le polygone ? (lancer de rayon, la règle du calcul de référence)"""
    dedans = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            dedans = not dedans
    return dedans


def tourner(cx: float, cy: float, pts: Sequence[Point], beta_deg: float) -> List[Point]:
    """Les points ``pts`` (repère local) tournés de β autour de l'origine, puis posés en (cx, cy)."""
    t = math.radians(beta_deg)
    c, s = math.cos(t), math.sin(t)
    return [(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts]


def pieds(cx: float, cy: float, a: float, b: float, beta_deg: float = 0.0) -> List[Point]:
    """Les centres des pieds P1 à P4 d'un châssis d'entraxes a (selon x) et b (selon y), centré en (cx, cy)."""
    return tourner(cx, cy, [(-a / 2, b / 2), (a / 2, b / 2), (a / 2, -b / 2), (-a / 2, -b / 2)], beta_deg)


def rectangle(x: float, y: float, ap: float, bp: float, beta_deg: float = 0.0) -> List[Point]:
    """Les coins d'un rectangle ap × bp centré en (x, y) : haut-gauche puis sens horaire (avant rotation)."""
    return tourner(x, y, [(-ap / 2, bp / 2), (ap / 2, bp / 2), (ap / 2, -bp / 2), (-ap / 2, -bp / 2)], beta_deg)


def aire_signee(poly: Sequence[Point]) -> float:
    """L'aire signée (formule du lacet) : positive dans le sens trigonométrique."""
    n = len(poly)
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))


def aire(poly: Sequence[Point]) -> float:
    return abs(aire_signee(poly))


def centre(poly: Sequence[Point]) -> Point:
    """Le centre d'une surface d'appui : la moyenne de ses sommets (la règle du calcul de référence)."""
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def sens_trigo(poly: Sequence[Point]) -> List[Point]:
    """Le polygone dans le sens trigonométrique, en gardant son premier sommet."""
    p = [tuple(q) for q in poly]
    return p if aire_signee(p) >= 0 else [p[0]] + p[:0:-1]


def sens_horaire(poly: Sequence[Point]) -> List[Point]:
    """Le polygone dans le sens horaire, en gardant son premier sommet."""
    p = [tuple(q) for q in poly]
    return p if aire_signee(p) <= 0 else [p[0]] + p[:0:-1]


def _croise(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Les segments p1-p2 et q1-q2 se coupent-ils franchement ?"""
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return o(q1, q2, p1) * o(q1, q2, p2) < 0 and o(p1, p2, q1) * o(p1, p2, q2) < 0


def se_recoupe(poly: Sequence[Point]) -> bool:
    """Un côté du polygone en coupe-t-il un autre ? (un contour mal dessiné, un décalage trop grand)"""
    n = len(poly)
    return any(_croise(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n])
               for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1))


def dans_dalle(contour: Sequence[Point], poly: Sequence[Point], tol: float = TOLERANCE) -> bool:
    """Le polygone (plaque ou plot) est-il entièrement dans la dalle ? Bord sur bord admis : il est d'abord rétréci de
    ``tol`` vers son centre ; puis tous ses sommets sont dans le contour et aucun de ses côtés n'en coupe un côté."""
    cx, cy = centre(poly)
    Q = []
    for x, y in poly:
        d = math.hypot(x - cx, y - cy)
        f = 1 - tol / max(d, tol)
        Q.append((cx + (x - cx) * f, cy + (y - cy) * f))
    if not all(dans_polygone(x, y, contour) for x, y in Q):
        return False
    n, m = len(contour), len(Q)
    return not any(_croise(Q[i], Q[(i + 1) % m], contour[j], contour[(j + 1) % n]) for i in range(m) for j in range(n))


def distance_point_segment(p: Point, a: Point, b: Point) -> float:
    ex, ey = b[0] - a[0], b[1] - a[1]
    L2 = ex * ex + ey * ey
    t = 0.0 if L2 == 0 else min(1.0, max(0.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / L2))
    return math.hypot(p[0] - (a[0] + t * ex), p[1] - (a[1] + t * ey))


def distance_au_bord(contour: Sequence[Point], poly: Sequence[Point]) -> float:
    """La plus petite distance entre un polygone et le contour : ses sommets aux bords du contour, et les sommets du
    contour à ses côtés."""
    n, m = len(contour), len(poly)
    d = min(distance_point_segment(q, contour[i], contour[(i + 1) % n]) for q in poly for i in range(n))
    return min(d, min(distance_point_segment(v, poly[i], poly[(i + 1) % m]) for v in contour for i in range(m)))


def distance_plaques_bord(contour: Sequence[Point], plaques: Sequence[Sequence[Point]]) -> Tuple[float, bool]:
    """(la distance mini entre les plaques et le contour, les plaques sont toutes dans la dalle)."""
    return min(distance_au_bord(contour, pl) for pl in plaques), all(dans_dalle(contour, pl) for pl in plaques)


def se_chevauchent(p: Sequence[Point], q: Sequence[Point], tol: float = TOLERANCE) -> bool:
    """Deux polygones CONVEXES (deux plots) se chevauchent-ils ? (axes séparateurs ; bord sur bord admis à ``tol``)"""
    for poly in (p, q):
        n = len(poly)
        for i in range(n):
            ex, ey = poly[(i + 1) % n][0] - poly[i][0], poly[(i + 1) % n][1] - poly[i][1]
            L = math.hypot(ex, ey)
            if L == 0:
                continue
            nx, ny = -ey / L, ex / L
            pa = [x * nx + y * ny for x, y in p]
            qa = [x * nx + y * ny for x, y in q]
            if max(pa) <= min(qa) + tol or max(qa) <= min(pa) + tol:
                return False
    return True


def couper_droite(contour: Sequence[Point], p0: Point, d: Point, t_plaques: Sequence[float]) -> Optional[Tuple[Point, Point]]:
    """Le segment de la droite p0 + t·d qui est dans le contour et qui contient les plaques (t moyen de ``t_plaques``)."""
    ts = []
    n = len(contour)
    for i in range(n):
        a, b = contour[i], contour[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        den = d[0] * (-ey) - d[1] * (-ex)
        if abs(den) < 1e-12:
            continue
        rx, ry = a[0] - p0[0], a[1] - p0[1]
        t = (rx * (-ey) - ry * (-ex)) / den
        s = (d[0] * ry - d[1] * rx) / den
        if -1e-9 <= s <= 1 + 1e-9:
            ts.append(t)
    ts = sorted(set(round(t, 9) for t in ts))
    tm = sum(t_plaques) / len(t_plaques)
    for t1, t2 in zip(ts[:-1], ts[1:]):
        mx, my = p0[0] + d[0] * (t1 + t2) / 2, p0[1] + d[1] * (t1 + t2) / 2
        if t1 <= tm <= t2 and dans_polygone(mx, my, contour):
            return (p0[0] + d[0] * t1, p0[1] + d[1] * t1), (p0[0] + d[0] * t2, p0[1] + d[1] * t2)
    return None


NOMS_BANDES = (("BX1", (0, 1), "x"), ("BX2", (3, 2), "x"), ("BY1", (0, 3), "y"), ("BY2", (1, 2), "y"))


def bandes_integration(contour: Sequence[Point], centres: Sequence[Point], beta_deg: float = 0.0, largeur: float = 1.0) -> List[dict]:
    """Les 4 bandes d'intégration qui passent par les centres des surfaces d'appui (P1 à P4), le long des axes du
    châssis, limitées à la dalle : BX1 (P1-P2), BX2 (P4-P3), BY1 (P1-P4), BY2 (P2-P3). Largeur totale symétrique.
    Chaque bande : {nom, debut, fin, largeur, pieds (les deux pieds traversés, indices 0 à 3)}."""
    t = math.radians(beta_deg)
    axes = {"x": (math.cos(t), math.sin(t)), "y": (-math.sin(t), math.cos(t))}
    out = []
    for nom, (i, j), axe in NOMS_BANDES:
        u = axes[axe]
        p0 = centres[i]
        tj = (centres[j][0] - p0[0]) * u[0] + (centres[j][1] - p0[1]) * u[1]
        seg = couper_droite(contour, p0, u, (0.0, tj))
        if seg:
            out.append({"nom": nom, "debut": seg[0], "fin": seg[1], "largeur": largeur, "pieds": (i, j)})
    return out


def decaler(contour: Sequence[Point], e: float) -> List[Point]:
    """Le contour décalé de ``e`` vers l'extérieur (coins en onglet : chaque sommet va à l'intersection de ses deux
    côtés décalés) — l'anneau D2 du SAF (2 cm), l'agrandissement de « Proposer ». Le contour rendu a le sens et le
    premier sommet du contour donné."""
    signe = 1.0 if aire_signee(contour) >= 0 else -1.0
    n = len(contour)
    out = []
    for i in range(n):
        p, q, r = contour[i - 1], contour[i], contour[(i + 1) % n]
        # les normales extérieures des deux côtés qui se rejoignent en q
        normales = []
        for a, b in ((p, q), (q, r)):
            ex, ey = b[0] - a[0], b[1] - a[1]
            L = math.hypot(ex, ey)
            normales.append((signe * ey / L, -signe * ex / L))
        (n1x, n1y), (n2x, n2y) = normales
        den = 1 + n1x * n2x + n1y * n2y
        if den < 1e-9:                                           # demi-tour : pas d'onglet
            out.append((q[0] + e * n1x, q[1] + e * n1y))
        else:
            out.append((q[0] + e * (n1x + n2x) / den, q[1] + e * (n1y + n2y) / den))
    return out


def boite(poly: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

# -*- coding: utf-8 -*-
"""Le tracé des cordons : une CHAÎNE (polyligne, arc, pourtour) devient des
cordons droits pour le calcul, et une soudure DISCONTINUE devient des
tronçons régulièrement répartis sur son chemin.

- chaîne OUVERTE (cordon courbe, tronçon) : la longueur efficace l − 2a
  (EN 1993-1-8 §4.5.1) s'obtient en RACCOURCISSANT la polyligne de a à chaque
  bout ; ses segments sont ``continu`` (pas de longueur minimale propre) et la
  chaîne entière doit avoir l − 2a ≥ max(30 mm ; 6a), sinon tous ses segments
  sont exclus (``exclu``) ;
- chaîne FERMÉE (pourtour) : pas de bout, longueur entière ;
- tronçons : n tronçons de longueur l_t sur un chemin — aux deux extrémités et
  régulièrement entre sur un chemin ouvert, régulièrement répartis à partir du
  point haut sur un chemin fermé. L'espacement est rendu pour le message
  §4.3.2.2 (≤ min(200 mm ; 12 t en compression ; 16 t en traction)).

Le côté de la pièce rapportée est celui de ``cordon_angle`` : +1 à gauche du
sens de parcours (l'intérieur d'un pourtour parcouru dans le sens
trigonométrique), −1 à droite. Unités : mm, degrés pour les angles.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cordon import A_MIN, L_MIN_ABS, Cordon, cordon_angle

Point = Tuple[float, float]


# ------------------------------------------------------------------ chemins
def _pts(points: Sequence) -> List[np.ndarray]:
    out = []
    for p in points:
        q = np.asarray(p, dtype=float)
        out.append(np.array([q[0], q[1]]))
    return out


def longueur_chemin(points: Sequence, ferme: bool = False) -> float:
    pts = _pts(points)
    L = sum(float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1))
    if ferme and len(pts) > 1:
        L += float(np.linalg.norm(pts[0] - pts[-1]))
    return L


def _segments(points: Sequence, ferme: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
    pts = _pts(points)
    seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    if ferme and len(pts) > 1:
        seg.append((pts[-1], pts[0]))
    return seg


def point_chemin(points: Sequence, ferme: bool, s: float) -> np.ndarray:
    """Point d'abscisse curviligne s (mm) sur le chemin ; s est ramené dans
    [0, L] par périodicité sur un chemin fermé, borné sur un chemin ouvert."""
    L = longueur_chemin(points, ferme)
    if L <= 0:
        return _pts(points)[0]
    s = s % L if ferme else min(max(0.0, s), L)
    acc = 0.0
    for p, q in _segments(points, ferme):
        d = float(np.linalg.norm(q - p))
        if s <= acc + d + 1e-9:
            t = 0.0 if d <= 0 else (s - acc) / d
            return p + (q - p) * min(max(t, 0.0), 1.0)
        acc += d
    return _pts(points)[0] if ferme else _pts(points)[-1]


def sous_chemin(points: Sequence, ferme: bool, s0: float, s1: float) -> List[np.ndarray]:
    """Polyligne du chemin entre les abscisses s0 et s1 (s1 > s0 ; sur un chemin
    fermé l'intervalle peut franchir l'origine) : les deux bouts interpolés et
    les sommets intermédiaires."""
    L = longueur_chemin(points, ferme)
    if L <= 0 or s1 <= s0:
        return [point_chemin(points, ferme, s0)]
    if ferme:
        base = s0 % L
        s1 = base + (s1 - s0)
        s0 = base
    else:
        s0, s1 = max(0.0, s0), min(L, s1)
    out = [point_chemin(points, ferme, s0)]
    acc = 0.0
    segs = _segments(points, ferme)
    # on parcourt le chemin (deux tours au plus sur un fermé) et on garde les sommets dans ]s0, s1[
    tours = 2 if ferme else 1
    for _ in range(tours):
        for p, q in segs:
            d = float(np.linalg.norm(q - p))
            s_fin = acc + d
            if s0 < s_fin < s1 - 1e-9:
                out.append(q.copy())
            acc = s_fin
    out.append(point_chemin(points, ferme, s1))
    # retire les doublons consécutifs
    propre = [out[0]]
    for p in out[1:]:
        if float(np.linalg.norm(p - propre[-1])) > 1e-9:
            propre.append(p)
    return propre


def abscisse_pres(points: Sequence, ferme: bool, P) -> float:
    """Abscisse curviligne du point du chemin le plus proche de P."""
    P = np.asarray(P, dtype=float)[:2]
    best, s_best, acc = None, 0.0, 0.0
    for p, q in _segments(points, ferme):
        d = float(np.linalg.norm(q - p))
        if d <= 0:
            continue
        t = float(np.dot(P - p, q - p)) / d ** 2
        t = min(max(t, 0.0), 1.0)
        dist = float(np.linalg.norm(p + (q - p) * t - P))
        if best is None or dist < best:
            best, s_best = dist, acc + t * d
        acc += d
    return s_best


# ------------------------------------------------------------------ formes
def arc(centre: Point, r: float, ang1: float, ang2: float, pas_deg: float = 15.0) -> List[Point]:
    """Polyligne d'un arc de cercle de ang1 à ang2 (degrés, sens trigonométrique),
    cordes de pas_deg au plus ; un tour complet ou plus donne un cercle fermé
    (dernier point non répété)."""
    tot = float(ang2) - float(ang1)
    if tot <= 0:
        raise ValueError("arc : l'angle de fin doit dépasser l'angle de début")
    ferme = tot >= 360.0 - 1e-9
    tot = min(tot, 360.0)
    n = max(2, int(math.ceil(tot / pas_deg)))
    pts = []
    for k in range(n if ferme else n + 1):
        a = math.radians(float(ang1) + tot * k / n)
        pts.append((centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)))
    return pts


def est_ferme_arc(ang1: float, ang2: float) -> bool:
    return float(ang2) - float(ang1) >= 360.0 - 1e-9


# ------------------------------------------------------------------ chaînes et tronçons
def chaine(nom: str, points: Sequence, a: float, cote: int = +1, ferme: bool = False, constructeur=cordon_angle,
           **kw) -> List[Cordon]:
    """Les cordons droits d'une chaîne : ouverte → raccourcie de a à chaque bout
    (l_eff = l − 2a), exclue si l − 2a < max(30 mm ; 6a) ; fermée → entière.
    ``constructeur`` : cordon_angle (défaut) ou une soudure bout à bout en T ;
    une chaîne à pleine pénétration (``pleine=True``) n'est ni raccourcie ni exclue."""
    L = longueur_chemin(points, ferme)
    exclu = False
    pleine = bool(kw.get("pleine"))
    if ferme or pleine:
        poly = _pts(points) + ([_pts(points)[0]] if ferme else [])
    else:
        l_eff = L - 2.0 * a
        exclu = l_eff < max(L_MIN_ABS, 6.0 * a) - 1e-9 or a < A_MIN - 1e-9
        poly = sous_chemin(points, False, a, L - a) if l_eff > 1e-9 else [point_chemin(points, False, L / 2)] * 2
    cs = []
    nseg = len(poly) - 1
    for i in range(nseg):
        p, q = poly[i], poly[i + 1]
        if float(np.linalg.norm(q - p)) <= 1e-9:
            continue
        nom_i = nom if nseg == 1 else f"{nom} ({i + 1})"
        cs.append(constructeur(nom_i, tuple(p), tuple(q), a, cote=cote, continu=True, chaine=nom, ferme=ferme, l_chaine=L,
                               exclu=exclu, **kw))
    if not cs:  # chaîne dégénérée : un cordon nul, exclu
        p = _pts(points)[0]
        cs.append(constructeur(nom, tuple(p), tuple(p + np.array([1e-3, 0.0])), a, cote=cote, continu=True, chaine=nom,
                               ferme=ferme, l_chaine=L, exclu=True, **kw))
    return cs


def abscisses_troncons(L: float, ferme: bool, n: int, l_t: float, s_origine: float = 0.0) -> List[float]:
    """Abscisses de début des n tronçons de longueur l_t : ouvert → aux deux
    bouts et régulièrement entre ; fermé → régulièrement, le premier centré sur
    s_origine. Refus si les tronçons ne tiennent pas sur le chemin."""
    n = int(n)
    if n < 1 or l_t <= 0:
        raise ValueError("tronçons : n ≥ 1 et longueur > 0 exigés")
    if n * l_t > L + 1e-9:
        raise ValueError(f"tronçons : {n} × {l_t:g} mm = {n * l_t:g} mm ne tiennent pas sur {L:.0f} mm")
    if ferme:
        return [(s_origine - l_t / 2.0 + k * L / n) % L for k in range(n)]
    if n == 1:
        return [(L - l_t) / 2.0]
    return [k * (L - l_t) / (n - 1) for k in range(n)]


def espacement_troncons(L: float, ferme: bool, n: int, l_t: float) -> Optional[float]:
    """Espace libre entre deux tronçons consécutifs (None pour un tronçon seul sur un chemin ouvert)."""
    n = int(n)
    if ferme:
        return L / n - l_t
    return None if n < 2 else (L - n * l_t) / (n - 1)


def tronconner(nom: str, points: Sequence, a: float, cote: int, n: int, l_t: float, ferme: bool = False,
               s_origine: float = 0.0, constructeur=cordon_angle, **kw) -> List[Cordon]:
    """Les cordons des n tronçons de longueur l_t répartis sur le chemin ;
    chaque tronçon est une chaîne ouverte « nom · tronçon k »."""
    L = longueur_chemin(points, ferme)
    cs: List[Cordon] = []
    for k, s0 in enumerate(abscisses_troncons(L, ferme, n, l_t, s_origine), 1):
        poly = sous_chemin(points, ferme, s0, s0 + l_t)
        cs += chaine(f"{nom} · tronçon {k}", poly, a, cote, ferme=False, constructeur=constructeur, **kw)
    return cs


def cordons_droits_ou_troncons(nom: str, p1, p2, a: float, cote: int, retours: bool, troncons: Optional[Dict] = None,
                               constructeur=cordon_angle, **kw) -> List[Cordon]:
    """Un cordon droit continu (retours d'angle possibles), ou ses tronçons
    si ``troncons`` = {"n": ≥ 1, "l": mm}."""
    tr = troncons or {}
    n = int(tr.get("n", 0) or 0)
    if n >= 1:
        return tronconner(nom, [p1, p2], a, cote, n, float(tr["l"]), ferme=False, constructeur=constructeur, **kw)
    return [constructeur(nom, p1, p2, a, cote=cote, retours=retours, **kw)]


def chaine_ou_troncons(nom: str, points: Sequence, a: float, cote: int, ferme: bool, troncons: Optional[Dict] = None,
                       s_origine: float = 0.0, constructeur=cordon_angle, **kw) -> List[Cordon]:
    """Une chaîne entière, ou ses tronçons si ``troncons`` = {"n": ≥ 1, "l": mm}."""
    tr = troncons or {}
    n = int(tr.get("n", 0) or 0)
    if n >= 1:
        return tronconner(nom, points, a, cote, n, float(tr["l"]), ferme=ferme, s_origine=s_origine, constructeur=constructeur, **kw)
    return chaine(nom, points, a, cote, ferme=ferme, constructeur=constructeur, **kw)

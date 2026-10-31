# -*- coding: utf-8 -*-
"""La platine COMPRIMÉE, bande par bande — le calcul manuel du côté comprimé.

Une BANDE de largeur unité part du nu d'une face d'appui (paroi d'un
élément, raidisseur, cercle d'un rond) perpendiculairement à elle, en
CONSOLE, et porte la pression de la répartition élastique
``p = kc·(a + bx·x + by·y)`` sur ce qu'elle traverse de la zone de contact
(polygones convexes du moteur) : moment au nu ``m = ∫ p·t dt`` (t : distance
au nu). Autour d'un rond la bande est RADIALE et s'élargit avec le rayon :
``m = (1/R)·∫ p·t·(R + t) dt``.

RÈGLE DE PARTAGE (cellule de Voronoï) : chaque paroi reprend la pression qui
lui est PLUS PROCHE qu'à toute autre paroi. La bande s'arrête donc au bord
de la platine ou là où un autre appui devient plus proche — entre deux
ailes, chacune porte sa moitié en console ; près d'un raidisseur, c'est lui
qui porte. C'est la règle du calcul à la main, du côté de la sécurité (une
bande tenue aux deux bouts porte moins : p·L²/12 contre p·L²/8), et elle
est cohérente avec la bande c d'EN 1993-1-8 §6.2.5 : en mode bande c, le
moment d'une bande ne dépasse jamais p·c²/2.

On échantillonne la face (41 bandes) ou le cercle (72 rayons) et on garde la
bande la PLUS sollicitée : la bande d'un mètre « au point le plus comprimé »
du calcul à la main, la pression linéaire en plus. Intégration de
Gauss–Legendre à 5 points par segment (exacte : pression linéaire, poids
polynomial).

Unités : N, mm → N·mm par mm.
"""
import math

from .. import geometrie as GE

N_BANDES = 41
N_RAYONS = 72
_GAUSS = ((-0.9061798459386640, 0.2369268850561891), (-0.5384693101056831, 0.4786286704993665),
          (0.0, 0.5688888888888889), (0.5384693101056831, 0.4786286704993665),
          (0.9061798459386640, 0.2369268850561891))


def _segment_rayon(ox, oy, nx, ny, poly):
    """``(t0, t1)`` : la portion du rayon O + t·n (t ≥ 0) à l'intérieur d'un
    polygone convexe, None s'il ne le traverse pas."""
    if GE.aire(poly) < 0:
        poly = list(reversed(poly))
    t0, t1 = 0.0, float("inf")
    n = len(poly)
    for i in range(n):
        px, py = poly[i]
        qx, qy = poly[(i + 1) % n]
        ex, ey = qx - px, qy - py
        # intérieur à gauche de p → q (polygone direct) : cross(e, X − p) ≥ 0
        c0 = ex * (oy - py) - ey * (ox - px)
        c1 = ex * ny - ey * nx
        if abs(c1) < 1e-15:
            if c0 < -1e-9:
                return None
            continue
        tc = -c0 / c1
        if c1 > 0:
            t0 = max(t0, tc)
        else:
            t1 = min(t1, tc)
        if t0 > t1:
            return None
    return (t0, t1) if t1 > t0 + 1e-9 else None


def _sortie_rect(ox, oy, nx, ny, rect):
    """La distance t ≥ 0 à laquelle le rayon SORT du rectangle qui contient
    son origine (la platine), ∞ si jamais."""
    x0, y0, x1, y1 = rect
    t = float("inf")
    if nx > 1e-15:
        t = min(t, (x1 - ox) / nx)
    elif nx < -1e-15:
        t = min(t, (x0 - ox) / nx)
    if ny > 1e-15:
        t = min(t, (y1 - oy) / ny)
    elif ny < -1e-15:
        t = min(t, (y0 - oy) / ny)
    return max(0.0, t)


def distance_appui(x, y, appui):
    """Distance d'un point à un appui : rectangle ``(x0, y0, x1, y1)`` (0 à
    l'intérieur) ou cercle ``("c", cx, cy, R)`` (|ρ − R|)."""
    if appui[0] == "c":
        return abs(math.hypot(x - appui[1], y - appui[2]) - appui[3])
    x0, y0, x1, y1 = appui
    return math.hypot(x - min(max(x, x0), x1), y - min(max(y, y0), y1))


def arret(ox, oy, nx, ny, plaque, autres):
    """La longueur de la bande : jusqu'au bord de la platine, ou jusqu'au
    point où un autre appui (``autres`` : rectangles ou cercles, sans la
    paroi d'origine) devient plus proche que la face d'origine (distance t).
    ``g(t) = min d_autre(t) − t`` décroît (1-Lipschitz) : dichotomie."""
    L = _sortie_rect(ox, oy, nx, ny, plaque)
    if not autres or L <= 0:
        return L

    def g(t):
        x, y = ox + t * nx, oy + t * ny
        return min(distance_appui(x, y, a) for a in autres) - t
    if g(L) >= 0:
        return L
    lo, hi = 0.0, L
    if g(lo) < 0:
        return 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if g(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


def moment_bande(ox, oy, nx, ny, L, polys, press, R=0.0):
    """Le moment par unité de largeur au nu, ``∫ p·t dt`` (× (R + t)/R pour
    une bande radiale) sur les portions de bande en contact."""
    m = 0.0
    for poly in polys:
        seg = _segment_rayon(ox, oy, nx, ny, poly)
        if seg is None:
            continue
        a, b = max(0.0, seg[0]), min(L, seg[1])
        if b <= a:
            continue
        h, c = 0.5 * (b - a), 0.5 * (a + b)
        for xi, wi in _GAUSS:
            t = c + h * xi
            w = t * ((R + t) / R if R else 1.0)
            m += wi * h * press(ox + t * nx, oy + t * ny) * w
    return m


def bandes_face(face, plaque, autres, polys, press, n=N_BANDES):
    """La bande la plus sollicitée le long d'une face ``dict(p0, p1, n, L)`` :
    ``dict(m, s, ox, oy, nx, ny, L, p0)`` (``p0`` : pression au nu)."""
    (x0, y0), (x1, y1) = face["p0"], face["p1"]
    nx, ny = face["n"]
    Lf = face["L"]
    pire = None
    for k in range(n):
        f = (k + 0.5) / n if n > 1 else 0.5
        ox, oy = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
        L = arret(ox, oy, nx, ny, plaque, autres)
        if L <= 1e-9:
            continue
        m = moment_bande(ox, oy, nx, ny, L, polys, press)
        if pire is None or m > pire["m"]:
            pire = dict(m=m, s=f * Lf, ox=ox, oy=oy, nx=nx, ny=ny, L=L, p0=press(ox, oy))
    return pire


def bandes_cercle(cx, cy, R, sens, plaque, autres, polys, press, n=N_RAYONS):
    """La bande radiale la plus sollicitée autour d'un cercle (``sens`` = +1
    vers l'extérieur, −1 vers l'intérieur d'un tube : bande droite)."""
    pire = None
    for k in range(n):
        th = 2 * math.pi * k / n
        nx, ny = sens * math.cos(th), sens * math.sin(th)
        ox, oy = cx + R * math.cos(th), cy + R * math.sin(th)
        L = arret(ox, oy, nx, ny, plaque, autres)
        if sens < 0:
            L = min(L, R)            # au-delà du centre du vide, l'autre côté du tube est plus proche
        if L <= 1e-9:
            continue
        m = moment_bande(ox, oy, nx, ny, L, polys, press, R=(R if sens > 0 else 0.0))
        if pire is None or m > pire["m"]:
            pire = dict(m=m, s=math.degrees(th), ox=ox, oy=oy, nx=nx, ny=ny, L=L, p0=press(ox, oy))
    return pire

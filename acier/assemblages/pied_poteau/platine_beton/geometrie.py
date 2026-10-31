# -*- coding: utf-8 -*-
"""Géométrie du pied de poteau — SANS MAILLAGE.

Tout ce qui sert à calculer une répartition de pression sous la platine se
ramène à des polygones : rectangles (parois du profilé élargies de la bande
``c`` d'EN 1993-1-8 §6.2.5), union exacte de rectangles, découpage d'un
polygone par un demi-plan (Sutherland–Hodgman, un seul bord) et intégrales
exactes de 1, x, y, x², xy, y² sur un polygone (formules de Green). Aucune
grille : les résultats sont exacts à l'arithmétique flottante près.

Repère : origine à l'axe du poteau, x parallèle aux ailes, y selon la hauteur.
Rectangles : tuples ``(x0, y0, x1, y1)`` avec x0 < x1, y0 < y1 ; polygones :
listes de sommets ``(x, y)`` dans l'ordre trigonométrique.
"""
import math

EPS = 1e-9


# ------------------------------------------------------------- polygones
def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def clip(poly, nx, ny, d):
    """Partie du polygone où nx·x + ny·y ≥ d (découpage par un demi-plan)."""
    out = []
    n = len(poly)
    if n == 0:
        return out
    for i in range(n):
        px, py = poly[i]
        qx, qy = poly[(i + 1) % n]
        fp = nx * px + ny * py - d
        fq = nx * qx + ny * qy - d
        if fp >= 0:
            out.append((px, py))
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((px + t * (qx - px), py + t * (qy - py)))
    return out


def moments(poly):
    """``(∫1, ∫x, ∫y, ∫x², ∫xy, ∫y²) dA`` exacts sur le polygone."""
    A = Sx = Sy = Ixx = Ixy = Iyy = 0.0
    n = len(poly)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        cr = x0 * y1 - x1 * y0
        A += cr
        Sx += (x0 + x1) * cr
        Sy += (y0 + y1) * cr
        Ixx += (x0 * x0 + x0 * x1 + x1 * x1) * cr
        Iyy += (y0 * y0 + y0 * y1 + y1 * y1) * cr
        Ixy += (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1) * cr
    return A / 2, Sx / 6, Sy / 6, Ixx / 12, Ixy / 24, Iyy / 12


def aire(poly):
    return moments(poly)[0]


# ------------------------------------------------------------ rectangles
def intersection(r, bornes):
    """Rectangle ∩ rectangle (None si vide)."""
    x0, y0, x1, y1 = r
    X0, Y0, X1, Y1 = bornes
    a, b, c, d = max(x0, X0), max(y0, Y0), min(x1, X1), min(y1, Y1)
    return (a, b, c, d) if c - a > EPS and d - b > EPS else None


def union_rectangles(rects):
    """Union EXACTE de rectangles alignés sur les axes, rendue comme une liste
    de rectangles DISJOINTS : découpage en bandes verticales aux abscisses des
    bords, fusion des intervalles en y dans chaque bande, puis fusion des
    bandes voisines qui portent les mêmes intervalles."""
    rects = [r for r in rects if r is not None and r[2] - r[0] > EPS and r[3] - r[1] > EPS]
    if not rects:
        return []
    xs = sorted({x for (x0, _y0, x1, _y1) in rects for x in (x0, x1)})
    bandes = []
    for xa, xb in zip(xs, xs[1:]):
        if xb - xa <= EPS:
            continue
        xm = 0.5 * (xa + xb)
        iv = sorted((y0, y1) for (x0, y0, x1, y1) in rects if x0 <= xm <= x1)
        fus = []
        for y0, y1 in iv:
            if fus and y0 <= fus[-1][1] + EPS:
                fus[-1][1] = max(fus[-1][1], y1)
            else:
                fus.append([y0, y1])
        bandes.append([xa, xb, tuple((round(a, 9), round(b, 9)) for a, b in fus)])
    out = []
    for xa, xb, iv in bandes:
        if out and out[-1][2] == iv and abs(out[-1][1] - xa) <= EPS:
            out[-1][1] = xb
        else:
            out.append([xa, xb, iv])
    return [(xa, y0, xb, y1) for xa, xb, iv in out for (y0, y1) in iv]


def aire_rects(rects):
    return sum((x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in rects)


def centroide_rects(rects):
    A = aire_rects(rects)
    if A <= 0:
        return 0.0, 0.0
    sx = sum((x1 - x0) * (y1 - y0) * 0.5 * (x0 + x1) for (x0, y0, x1, y1) in rects)
    sy = sum((x1 - x0) * (y1 - y0) * 0.5 * (y0 + y1) for (x0, y0, x1, y1) in rects)
    return sx / A, sy / A


# ------------------------------------------------------------ le profilé
def parois_I(h, b, tw, tf):
    """Les trois parois d'un profilé en I : ``(nom, x0, y0, x1, y1)``."""
    return [("aile+", -b / 2, h / 2 - tf, b / 2, h / 2),
            ("aile-", -b / 2, -h / 2, b / 2, -h / 2 + tf),
            ("ame", -tw / 2, -h / 2 + tf, tw / 2, h / 2 - tf)]


def parois_RHS(h, b, t):
    """Les quatre parois d'un profil creux rectangulaire (angles vifs)."""
    return [("paroi+y", -b / 2, h / 2 - t, b / 2, h / 2),
            ("paroi-y", -b / 2, -h / 2, b / 2, -h / 2 + t),
            ("paroi+x", b / 2 - t, -h / 2 + t, b / 2, h / 2 - t),
            ("paroi-x", -b / 2, -h / 2 + t, -b / 2 + t, h / 2 - t)]


def bande(paroi, c, plaque):
    """Une paroi élargie de ``c`` de chaque côté, limitée à la platine
    (``plaque`` = (x0, y0, x1, y1))."""
    _n, x0, y0, x1, y1 = paroi
    return intersection((x0 - c, y0 - c, x1 + c, y1 + c), plaque)


def aire_efficace(parois, c, plaque):
    """L'aire efficace d'EN 1993-1-8 §6.2.5 : union des parois élargies de
    ``c``, découpée par la platine — liste de rectangles disjoints."""
    return union_rectangles([bande(p, c, plaque) for p in parois])


def tronçon_comprime(parois, c, plaque, sens, forme):
    """Le tronçon en T comprimé pour un moment qui comprime le côté ``sens``
    (``(0, 1)``, ``(0, −1)``, ``(1, 0)`` ou ``(−1, 0)``) : la bande de la
    paroi de ce côté (profilé en I : l'aile pour ±y, les DEMI-bandes des deux
    ailes pour ±x — l'âme n'y entre pas). Retourne ``(rects, aire, bras)``,
    ``bras`` = distance de son centre de gravité à l'axe du poteau."""
    sx, sy = sens
    out = []
    for p in parois:
        nom = p[0]
        r = bande(p, c, plaque)
        if r is None:
            continue
        if forme == "I":
            if sy and nom == ("aile+" if sy > 0 else "aile-"):
                out.append(r)
            elif sx and nom.startswith("aile"):
                moitie = intersection(r, (0, -1e9, 1e9, 1e9) if sx > 0 else (-1e9, -1e9, 0, 1e9))
                if moitie:
                    out.append(moitie)
        else:
            if (sy and nom == ("paroi+y" if sy > 0 else "paroi-y")) or (sx and nom == ("paroi+x" if sx > 0 else "paroi-x")):
                out.append(r)
    A = aire_rects(out)
    gx, gy = centroide_rects(out)
    return out, A, abs(gx * sx + gy * sy)


# ------------------------------------------------------------- les tiges
def positions_tiges(n_ax, n_ay, w_x, w_y):
    """Grille centrée : ``n_ax`` files en x réparties sur ``w_x``, ``n_ay``
    rangées en y sur ``w_y``. Retourne ``[(x, y, i_rangée, j_file), …]``."""
    n_ax, n_ay = max(1, int(n_ax)), max(1, int(n_ay))
    xs = [(-w_x / 2 + j * w_x / (n_ax - 1)) if n_ax > 1 else 0.0 for j in range(n_ax)]
    ys = [(-w_y / 2 + i * w_y / (n_ay - 1)) if n_ay > 1 else 0.0 for i in range(n_ay)]
    return [(x, y, i, j) for i, y in enumerate(ys) for j, x in enumerate(xs)]


def classer_tige(x, y, forme, dims, a_w):
    """Où est la tige par rapport au profilé, et contre quelle ligne d'appui
    la platine fléchit (tronçon en T) :

    - ``ext``  : au-delà d'une paroi, dans sa largeur — tronçon en T codifié
                 (rangée extérieure à l'aile, EN 1993-1-8 Tableau 6.6) ;
    - ``coin`` : au-delà de l'aile ET de son about — pas de longueur efficace
                 codifiée : tronçon en T contre la face d'aile, longueur
                 plafonnée (modèle d'ingénieur) ;
    - ``lat``  : au-delà de l'about d'aile, entre les niveaux d'aile — appui
                 réduit à l'épaisseur de l'aile (modèle d'ingénieur) ;
    - ``int``  : entre les ailes, contre l'âme — tronçon en T non raidi
                 (Tableau 6.4, conservatif) ;
    - ``dedans`` : dans le vide d'un profil creux — géométrie invalide.

    Retourne ``dict(classe, ligne, m)`` : ``ligne`` = ("y", ±1) pour une
    paroi normale à y, ("x", ±1) pour une paroi normale à x, ``m`` = distance
    au bord de l'appui, cordon déduit (0,8·a·√2) ou congé (0,8·r)."""
    h, b = dims["h"], dims["b"]
    sy, sx = (1 if y >= 0 else -1), (1 if x >= 0 else -1)
    cord = 0.8 * a_w * math.sqrt(2)
    if forme == "I":
        tw, tf, r = dims["tw"], dims["tf"], dims.get("r", 0) or 0
        if abs(y) > h / 2:
            m = abs(y) - h / 2 - cord
            return dict(classe="ext" if abs(x) <= b / 2 else "coin", ligne=("y", sy), m=m)
        if abs(x) > b / 2:
            return dict(classe="lat", ligne=("x", sx), m=abs(x) - b / 2 - cord)
        return dict(classe="int", ligne=("x", sx), m=abs(x) - tw / 2 - 0.8 * r,
                    m2=h / 2 - tf - abs(y))
    t = dims["t"]
    dy, dx = abs(y) - h / 2, abs(x) - b / 2
    if dy > 0 and dx <= 0:
        return dict(classe="ext", ligne=("y", sy), m=dy - cord)
    if dx > 0 and dy <= 0:
        return dict(classe="ext", ligne=("x", sx), m=dx - cord)
    if dx > 0 and dy > 0:
        return dict(classe="coin", ligne=("y", sy) if dy <= dx else ("x", sx), m=min(dx, dy) - cord)
    if abs(x) < b / 2 - t and abs(y) < h / 2 - t:
        return dict(classe="dedans", ligne=("y", sy), m=0.0)
    return dict(classe="int", ligne=("y", sy) if abs(y) >= abs(x) else ("x", sx), m=0.0)

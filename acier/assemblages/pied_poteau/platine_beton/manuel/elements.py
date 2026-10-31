# -*- coding: utf-8 -*-
"""Éléments soudés sur la platine et raidisseurs — la GÉOMÉTRIE du calcul
manuel, dans le repère de la platine (origine au centre, x et y du moteur).

Un élément (profilé en I ou H, profil creux rectangulaire ou circulaire,
plat sur chant, rond plein, rectangle plein) et un raidisseur donnent :

- leurs PAROIS : rectangles alignés sur les axes (comme ``geometrie.parois_I``)
  — un rond ou un tube n'a pas de paroi rectangulaire, il est ROND ;
- leurs FACES : les lignes d'appui de la platine (bords des parois), et
  l'ENVELOPPE (rectangle extérieur, cercle pour les ronds) ;
- leurs BANDES d'appui (parois élargies de ``c`` — EN 1993-1-8 §6.2.5) : un
  rond est découpé en LAMELLES horizontales d'aire exacte (chaque lamelle a
  l'aire de la tranche de disque qu'elle remplace) ;
- la distance d'un boulon à l'appui le plus proche, et la face en cause.

Unités : mm. Rien n'est vérifié ici : la géométrie seulement.
"""
import math

from acier.bibliotheques import DB, PERSO, find
from acier.js import N
from ..entrees import (TYPE_I, TYPE_RHS, TYPE_CHS, TYPE_PLAT, TYPE_ROND, TYPE_RECT, TYPES_ELEM, DIRS_RAID,
                       element_defaut, raidisseur_defaut)
from .. import geometrie as GE

N_LAMELLES = 48          # lamelles sur le diamètre d'un rond (aire exacte, contour approché)


# ----------------------------------------------------------------- lecture
def _n(v, defaut=0.0):
    x = N(v)
    return defaut if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def normaliser(e):
    """Un élément lu (dict libre) → dict complet, nombres flottants ; les
    dimensions d'un profilé du catalogue viennent de la bibliothèque."""
    base = element_defaut()
    e = dict(e or {})
    t = e.get("type") if e.get("type") in TYPES_ELEM else TYPE_I
    out = dict(type=t, prof=str(e.get("prof") or base["prof"]), x=_n(e.get("x")), y=_n(e.get("y")),
               rot=90 if int(_n(e.get("rot"))) % 180 == 90 else 0)
    for k in ("h", "b", "tw", "tf", "r", "t", "d"):
        out[k] = _n(e.get(k, base[k]), base[k])
    if t == TYPE_I and out["prof"] != PERSO:
        p = find(DB["profils"], out["prof"])
        if p is None:
            out["prof"] = PERSO
        else:
            out.update(h=float(p["h"]), b=float(p["b"]), tw=float(p["tw"]), tf=float(p["tf"]), r=float(p["r"]))
    return out


def normaliser_raid(r):
    base = raidisseur_defaut()
    r = dict(r or {})
    d = str(r.get("dir") or "+x").replace("-", "−")
    return dict(x=_n(r.get("x"), base["x"]), y=_n(r.get("y"), base["y"]), dir=d if d in DIRS_RAID else "+x",
                L=max(0.0, _n(r.get("L"), base["L"])), t=max(0.0, _n(r.get("t"), base["t"])),
                h=max(0.0, _n(r.get("h"), base["h"])))


def nom(e):
    """« HEA 300 », « RHS 200×150×8 », « CHS 219,1×8 », « plat 200×10 », « rond Ø20 »…"""
    t = e["type"]

    def f(v):
        return ("%g" % v).replace(".", ",")
    if t == TYPE_I:
        return e["prof"] if e["prof"] != PERSO else f"I {f(e['h'])}×{f(e['b'])}×{f(e['tw'])}/{f(e['tf'])}"
    if t == TYPE_RHS:
        return f"RHS {f(e['h'])}×{f(e['b'])}×{f(e['t'])}"
    if t == TYPE_CHS:
        return f"CHS {f(e['d'])}×{f(e['t'])}"
    if t == TYPE_PLAT:
        return f"plat {f(e['b'])}×{f(e['t'])}"
    if t == TYPE_ROND:
        return f"rond Ø{f(e['d'])}"
    return f"rectangle {f(e['b'])}×{f(e['h'])}"


def est_rond(e):
    return e["type"] in (TYPE_CHS, TYPE_ROND)


def geometrie_valide(e):
    t = e["type"]
    if t == TYPE_I:
        return e["h"] > 0 and e["b"] > 0 and e["tw"] > 0 and e["tf"] > 0 and e["h"] > 2 * e["tf"] and e["b"] > e["tw"]
    if t == TYPE_RHS:
        return e["h"] > 0 and e["b"] > 0 and e["t"] > 0 and 2 * e["t"] < min(e["h"], e["b"])
    if t == TYPE_CHS:
        return e["d"] > 0 and e["t"] > 0 and 2 * e["t"] < e["d"]
    if t == TYPE_PLAT:
        return e["b"] > 0 and e["t"] > 0
    if t == TYPE_ROND:
        return e["d"] > 0
    return e["b"] > 0 and e["h"] > 0


# ------------------------------------------------------------------ parois
def _tourne(rects, rot):
    """Rotation de 90° d'une liste de rectangles (x, y) → (−y, x) : la hauteur
    h passe le long de x."""
    if rot != 90:
        return rects
    return [(n_, -y1, x0, -y0, x1) for (n_, x0, y0, x1, y1) in rects]


def _decale(rects, dx, dy):
    return [(n_, x0 + dx, y0 + dy, x1 + dx, y1 + dy) for (n_, x0, y0, x1, y1) in rects]


def parois(e):
    """Les parois de l'élément : ``(nom, x0, y0, x1, y1)`` dans le repère de la
    platine (vide pour un rond ou un tube circulaire)."""
    t = e["type"]
    if t == TYPE_I:
        r = GE.parois_I(e["h"], e["b"], e["tw"], e["tf"])
    elif t == TYPE_RHS:
        r = GE.parois_RHS(e["h"], e["b"], e["t"])
    elif t == TYPE_PLAT:
        r = [("plat", -e["b"] / 2, -e["t"] / 2, e["b"] / 2, e["t"] / 2)]
    elif t == TYPE_RECT:
        r = [("rect", -e["b"] / 2, -e["h"] / 2, e["b"] / 2, e["h"] / 2)]
    else:
        return []
    return _decale(_tourne(r, e["rot"]), e["x"], e["y"])


def rayons(e):
    """``(R_ext, R_int)`` d'un rond (R_int = 0) ou d'un tube circulaire."""
    if e["type"] == TYPE_CHS:
        return e["d"] / 2, e["d"] / 2 - e["t"]
    if e["type"] == TYPE_ROND:
        return e["d"] / 2, 0.0
    return 0.0, 0.0


def enveloppe(e):
    """Le rectangle extérieur de l'élément (le carré circonscrit d'un rond)."""
    if est_rond(e):
        R = rayons(e)[0]
        return (e["x"] - R, e["y"] - R, e["x"] + R, e["y"] + R)
    ps = parois(e)
    return (min(p[1] for p in ps), min(p[2] for p in ps), max(p[3] for p in ps), max(p[4] for p in ps))


def rect_raid(r):
    """Le rectangle d'un raidisseur : parti de (x ; y), longueur L dans sa
    direction, épaisseur t centrée sur la ligne."""
    x, y, L, t = r["x"], r["y"], r["L"], r["t"]
    if r["dir"] == "+x":
        return (x, y - t / 2, x + L, y + t / 2)
    if r["dir"] == "−x":
        return (x - L, y - t / 2, x, y + t / 2)
    if r["dir"] == "+y":
        return (x - t / 2, y, x + t / 2, y + L)
    return (x - t / 2, y - L, x + t / 2, y)


# ------------------------------------------------------------------ bandes
def _tranche(R, a, b):
    """Aire exacte de la tranche de disque de rayon R entre y = a et y = b
    (demi-largeur intégrée) : ∫√(R² − y²) dy."""
    def Fy(y):
        y = max(-R, min(R, y))
        return 0.5 * (y * math.sqrt(max(0.0, R * R - y * y)) + R * R * math.asin(y / R)) if R > 0 else 0.0
    return Fy(b) - Fy(a)


def lamelles(cx, cy, R_ext, R_int, plaque, n=N_LAMELLES):
    """Un disque (R_int = 0) ou un anneau découpé en ``n`` lamelles
    horizontales, chacune de l'aire EXACTE de la tranche qu'elle remplace ;
    découpées par la platine. Rectangles ``(x0, y0, x1, y1)``."""
    if R_ext <= 0:
        return []
    h = 2 * R_ext / n
    out = []
    for k in range(n):
        ya, yb = -R_ext + k * h, -R_ext + (k + 1) * h
        wo = _tranche(R_ext, ya, yb) / h
        wi = _tranche(R_int, ya, yb) / h if R_int > 0 else 0.0
        rects = [(cx - wo, cy + ya, cx - wi, cy + yb), (cx + wi, cy + ya, cx + wo, cy + yb)] if wi > 1e-9 \
            else [(cx - wo, cy + ya, cx + wo, cy + yb)]
        for rc in rects:
            if rc[2] - rc[0] > 1e-9:
                i = GE.intersection(rc, plaque)
                if i:
                    out.append(i)
    return out


def bandes(e, c, plaque):
    """Les bandes d'appui de l'élément (parois élargies de ``c``, EN 1993-1-8
    §6.2.5), découpées par la platine — rectangles disjoints après union."""
    if est_rond(e):
        R_ext, R_int = rayons(e)
        return lamelles(e["x"], e["y"], R_ext + c, max(0.0, R_int - c), plaque)
    out = []
    for p in parois(e):
        b = GE.bande(p, c, plaque)
        if b:
            out.append(b)
    return out


def bande_raid(r, c, plaque):
    x0, y0, x1, y1 = rect_raid(r)
    return GE.intersection((x0 - c, y0 - c, x1 + c, y1 + c), plaque)


def aire_section(e):
    """L'aire de la section de l'élément (mm²) — pour le point d'application
    par défaut et le dessin."""
    if est_rond(e):
        R_ext, R_int = rayons(e)
        return math.pi * (R_ext ** 2 - R_int ** 2)
    return sum((x1 - x0) * (y1 - y0) for (_n, x0, y0, x1, y1) in parois(e))


# ------------------------------------------------------------------- faces
def faces_rect(nom_, rect):
    """Les quatre faces d'un rectangle : ``dict(nom, p0, p1, n, L)``, normale
    sortante unitaire, ``p0 → p1`` dans le sens direct autour du rectangle."""
    x0, y0, x1, y1 = rect
    return [dict(nom=nom_ + " +y", p0=(x0, y1), p1=(x1, y1), n=(0.0, 1.0), L=x1 - x0),
            dict(nom=nom_ + " −y", p0=(x0, y0), p1=(x1, y0), n=(0.0, -1.0), L=x1 - x0),
            dict(nom=nom_ + " +x", p0=(x1, y0), p1=(x1, y1), n=(1.0, 0.0), L=y1 - y0),
            dict(nom=nom_ + " −x", p0=(x0, y0), p1=(x0, y1), n=(-1.0, 0.0), L=y1 - y0)]


def dedans_rect(rect, x, y, marge=0.0):
    x0, y0, x1, y1 = rect
    return (x0 - marge) < x < (x1 + marge) and (y0 - marge) < y < (y1 + marge)


def dedans(e, x, y, marge=0.0):
    """Le point (x ; y) est-il dans l'emprise de l'élément (vide d'un tube
    compris : un boulon n'y a pas sa place) ?"""
    if est_rond(e):
        return math.hypot(x - e["x"], y - e["y"]) < rayons(e)[0] + marge
    return dedans_rect(enveloppe(e), x, y, marge)


def appui_proche_rect(nom_, rect, x, y):
    """Le point d'appui le plus proche de (x ; y) sur le bord d'un rectangle :
    ``dict(d, px, py, face, L, s, coin, nx, ny, faces)`` — ``coin`` : la
    projection tombe hors de la face (le bras est la diagonale jusqu'à
    l'angle) ; ``(nx, ny)`` : direction unitaire de l'appui vers le boulon ;
    ``faces`` : la face en cause, ou les DEUX faces de l'angle (lignes
    d'appui des sections au nu)."""
    x0, y0, x1, y1 = rect
    px, py = min(max(x, x0), x1), min(max(y, y0), y1)
    d = math.hypot(x - px, y - py)
    in_x, in_y = x0 < x < x1, y0 < y < y1
    fs = {f["nom"].rsplit(" ", 1)[1]: f for f in faces_rect(nom_, rect)}
    if d <= 1e-9:
        return dict(d=0.0, px=px, py=py, face=nom_, L=0.0, s=0.0, coin=False, nx=0.0, ny=0.0, dedans=True, faces=[])
    nx, ny = (x - px) / d, (y - py) / d
    if in_x and not in_y:
        k = "+y" if y > y1 else "−y"
        return dict(d=d, px=px, py=py, face=nom_ + " " + k, L=x1 - x0, s=x - x0, coin=False, nx=nx, ny=ny,
                    dedans=False, faces=[fs[k]])
    if in_y and not in_x:
        k = "+x" if x > x1 else "−x"
        return dict(d=d, px=px, py=py, face=nom_ + " " + k, L=y1 - y0, s=y - y0, coin=False, nx=nx, ny=ny,
                    dedans=False, faces=[fs[k]])
    kx = "+x" if x > x1 else "−x"
    ky = "+y" if y > y1 else "−y"
    return dict(d=d, px=px, py=py, face=nom_ + " angle " + kx + " " + ky, L=0.0, s=0.0, coin=True, nx=nx, ny=ny,
                dedans=False, faces=[fs[ky], fs[kx]])


def appui_proche(e, i, x, y):
    """L'appui le plus proche de (x ; y) sur l'élément ``i`` (toutes ses
    parois, ou son cercle extérieur)."""
    etq = f"élément {i + 1} ({nom(e)})"
    if est_rond(e):
        R = rayons(e)[0]
        rho = math.hypot(x - e["x"], y - e["y"])
        if rho < 1e-9:
            return dict(d=0.0, px=e["x"], py=e["y"], face=etq, L=2 * R, s=0.0, coin=False, nx=0.0, ny=0.0, dedans=True)
        nx, ny = (x - e["x"]) / rho, (y - e["y"]) / rho
        return dict(d=rho - R, px=e["x"] + R * nx, py=e["y"] + R * ny, face=etq + " cercle", L=2 * R, s=R,
                    coin=False, nx=nx, ny=ny, dedans=rho < R, cercle=True, faces=[])
    meilleur = None
    for (n_, x0, y0, x1, y1) in parois(e):
        a = appui_proche_rect(etq + " " + n_, (x0, y0, x1, y1), x, y)
        if meilleur is None or a["d"] < meilleur["d"]:
            meilleur = a
    return meilleur


def appui_proche_raid(r, i, x, y):
    return appui_proche_rect(f"raidisseur {i + 1}", rect_raid(r), x, y)


# ----------------------------------------------------------------- contour
def contour(e):
    """Ce que le dessin trace : ``dict(t="I", pts)`` (section en I avec ses
    congés), ``dict(t="rect", rects)`` (parois pleines), ``dict(t="anneau",
    cx, cy, R, Ri)``, ``dict(t="disque", cx, cy, R)`` — repère du moteur."""
    t = e["type"]
    if t == TYPE_I:
        from ....poutre_poutre.doubles_cornieres.schemas import _section_I
        pts = [(x, -(y - e["h"] / 2)) for x, y in _section_I(e["b"], e["h"], e["tw"], e["tf"], e["r"])]
        if e["rot"] == 90:
            pts = [(-y, x) for x, y in pts]
        return dict(t="I", pts=[(x + e["x"], y + e["y"]) for x, y in pts])
    if t == TYPE_CHS:
        return dict(t="anneau", cx=e["x"], cy=e["y"], R=e["d"] / 2, Ri=e["d"] / 2 - e["t"])
    if t == TYPE_ROND:
        return dict(t="disque", cx=e["x"], cy=e["y"], R=e["d"] / 2)
    return dict(t="rect", rects=[(x0, y0, x1, y1) for (_n, x0, y0, x1, y1) in parois(e)], env=enveloppe(e))

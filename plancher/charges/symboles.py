# -*- coding: utf-8 -*-
"""Symboles d'engins en vue de dessus, au trait, À L'ÉCHELLE de leur fiche.

Chaque symbole est dessiné dans le repère de l'engin (x vers l'avant, y
vers la droite du dessin), en MILLIMÈTRES, à partir des appuis du gabarit
(positions, empreintes) et de quelques dimensions de silhouette
(``longueur``, ``largeur``, ``fleche``, ``tourelle``) qui, absentes, se
déduisent de l'emprise des appuis. Le plan les pose par une transformation
``translate · rotate``. Les semelles, plateaux et aires efficaces ne sont
PAS dans le symbole : le plan les dessine à part (objets cliquables).

Aucune dépendance : des chaînes SVG. Les épaisseurs de trait sont en
unités du dessin (mm) ; ``vector-effect="non-scaling-stroke"`` les garde
constantes à l'écran.
"""
from typing import Dict, List, Sequence

NOIR, GRIS, ROUE, CAPOT = "#111827", "#6b7280", "#d1d5db", "#f3f4f6"
NS = 'vector-effect="non-scaling-stroke"'


def _rect(x, y, w, h, fill="white", rx=0.0, sw=1.1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{rx:.0f}" fill="{fill}" stroke="{NOIR}" stroke-width="{sw}" {NS}{d}/>'


def _poly(pts, fill="white", sw=1.1):
    return f'<polygon points="{" ".join(f"{x:.0f},{y:.0f}" for x, y in pts)}" fill="{fill}" stroke="{NOIR}" stroke-width="{sw}" {NS}/>'


def _circle(cx, cy, r, fill="white", sw=1.1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="{fill}" stroke="{NOIR}" stroke-width="{sw}" {NS}{d}/>'


def _bras(x1, y1, x2, y2, w):
    """Bras rectangulaire de largeur w entre deux points."""
    import math
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    ux, uy = -dy / n * w / 2, dx / n * w / 2
    return _poly([(x1 + ux, y1 + uy), (x2 + ux, y2 + uy), (x2 - ux, y2 - uy), (x1 - ux, y1 - uy)])


def _emprise(appuis: Sequence[Dict]):
    xs = [a["dx"] * 1e3 for a in appuis] or [0.0]
    ys = [a["dy"] * 1e3 for a in appuis] or [0.0]
    return min(xs), max(xs), min(ys), max(ys)


def symbole(gabarit: Dict) -> str:
    """Le symbole d'un gabarit (dict de la bibliothèque, dimensions en m)."""
    t = gabarit.get("type_appuis", "pieds")
    appuis = gabarit.get("appuis", [])
    sil = gabarit.get("silhouette", {})
    x0, x1, y0, y1 = _emprise(appuis)
    if t == "pieds":
        return _pieds(appuis, sil, x0, x1, y0, y1)
    if t == "roues":
        return _roues(appuis, sil, x0, x1, y0, y1)
    if t == "chenilles":
        return _chenilles(appuis, sil, x0, x1, y0, y1)
    if t == "surfacique":
        return ""
    return _ponctuel()


def _pieds(appuis, sil, x0, x1, y0, y1) -> str:
    """Engin sur stabilisateurs : châssis et roues, bras en ciseaux vers les
    appuis, tourelle (cabine à gauche, moteur à l'arrière), flèche à droite."""
    L = sil.get("longueur", 0.0) * 1e3 or 0.95 * (x1 - x0) or 4000.0
    W = sil.get("largeur", 0.0) * 1e3 or 0.5 * (y1 - y0) or 2200.0
    fl = sil.get("fleche", 0.0) * 1e3 or 0.55 * L
    s = []
    for sx in (-1, 1):                                       # pneus
        for sy in (-1, 1):
            s.append(_rect(sx * 0.285 * L - 0.12 * L, sy * (W / 2 - 0.18 * W) - 0.09 * W, 0.24 * L, 0.18 * W, fill=ROUE, rx=60))
    s.append(_rect(-L / 2, -W / 2, L, W, rx=150))            # châssis
    for a in appuis:                                          # bras vers chaque appui
        ax, ay = a["dx"] * 1e3, a["dy"] * 1e3
        bx = max(-L / 2 * 0.9, min(L / 2 * 0.9, ax * 0.75))
        by = max(-W / 2 * 0.9, min(W / 2 * 0.9, ay * 0.45))
        s.append(_bras(bx, by, ax, ay, 0.1 * W))
    if sil.get("tourelle", True):
        s.append(_rect(-0.42 * L, -0.46 * W, 0.66 * L, 0.92 * W, rx=0.3 * W))        # tourelle
        s.append(_circle(-0.1 * L, 0, 0.38 * W, fill="none", sw=0.8, dash="40 25"))   # couronne
        s.append(_rect(-0.4 * L, -0.38 * W, 0.17 * L, 0.76 * W, fill=CAPOT, rx=60))   # moteur
        s.append(_rect(-0.06 * L, -0.45 * W, 0.2 * L, 0.38 * W, rx=70))              # cabine
        s.append(_poly([(-0.36 * L, 0.08 * W), (fl, 0.12 * W), (fl, 0.34 * W), (-0.36 * L, 0.36 * W)]))   # flèche
        s.append(f'<line x1="{-0.36 * L:.0f}" y1="{0.22 * W:.0f}" x2="{fl:.0f}" y2="{0.22 * W:.0f}" stroke="{NOIR}" stroke-width="0.6" {NS}/>')
        s.append(_rect(fl, -0.02 * W, 0.03 * L, 0.46 * W))                           # tablier
        s.append(_circle(-0.1 * L, 0, 60, fill=NOIR))
    return "".join(s)


def _roues(appuis, sil, x0, x1, y0, y1) -> str:
    """Engin à roues : châssis, pneus aux appuis, cabine à gauche, capot à
    droite, flèche au centre-droit, tablier et fourches à l'avant."""
    L = sil.get("longueur", 0.0) * 1e3 or 1.5 * (x1 - x0) or 4500.0
    W = sil.get("largeur", 0.0) * 1e3 or 1.2 * (y1 - y0) or 2300.0
    s = []
    for a in appuis:
        ax, ay, ea, eb = a["dx"] * 1e3, a["dy"] * 1e3, a["a"] * 1e3, a["b"] * 1e3
        s.append(_rect(ax - 0.14 * L, ay - 0.1 * W, 0.28 * L, 0.2 * W, fill=ROUE, rx=60))
    s.append(_rect(-L / 2, -W / 2, L, W, rx=150))
    s.append(_rect(-0.3 * L, -0.48 * W, 0.27 * L, 0.33 * W, fill=CAPOT, rx=50))
    s.append(_rect(-0.31 * L, 0.12 * W, 0.26 * L, 0.37 * W, rx=70))
    s.append(_poly([(-0.47 * L, -0.22 * W), (0.53 * L, -0.19 * W), (0.53 * L, 0.01 * W), (-0.47 * L, 0.03 * W)]))
    s.append(_rect(0.53 * L, -0.31 * W, 0.03 * L, 0.54 * W))
    s.append(_rect(0.56 * L, -0.24 * W, 0.26 * L, 0.05 * W, fill=NOIR))
    s.append(_rect(0.56 * L, 0.19 * W, 0.26 * L, 0.05 * W, fill=NOIR))
    return "".join(s)


def _chenilles(appuis, sil, x0, x1, y0, y1) -> str:
    """Mini-grue sur chenilles et stabilisateurs : chenilles, caisson, bras
    diagonaux vers les appuis, tourelle et flèche."""
    L = sil.get("longueur", 0.0) * 1e3 or 0.8 * (x1 - x0) or 2900.0
    W = sil.get("largeur", 0.0) * 1e3 or 0.35 * (y1 - y0) or 1200.0
    fl = sil.get("fleche", 0.0) * 1e3 or 0.9 * L
    s = []
    for sy in (-1, 1):
        s.append(_rect(-L / 2, sy * 0.42 * W - 0.14 * W, L, 0.28 * W, fill=ROUE, rx=50))
    s.append(_rect(-0.38 * L, -0.35 * W, 0.76 * L, 0.7 * W, rx=60))
    for a in appuis:
        ax, ay = a["dx"] * 1e3, a["dy"] * 1e3
        s.append(_bras(0.0 if abs(ax) < 1 else (ax / abs(ax)) * 0.3 * L, (ay / abs(ay) if abs(ay) > 1 else 0.0) * 0.3 * W, ax, ay, 0.13 * W))
    s.append(_circle(-0.07 * L, 0, 0.32 * W))
    s.append(_poly([(-0.07 * L, -0.13 * W), (fl, -0.08 * W), (fl, 0.08 * W), (-0.07 * L, 0.13 * W)]))
    s.append(_circle(-0.07 * L, 0, 50, fill=NOIR))
    return "".join(s)


def _ponctuel() -> str:
    """Étançon ou charge ponctuelle : cercle barré."""
    r = 180
    return (_circle(0, 0, r, sw=1.4)
            + f'<line x1="{-r * 0.65:.0f}" y1="{-r * 0.65:.0f}" x2="{r * 0.65:.0f}" y2="{r * 0.65:.0f}" stroke="{NOIR}" stroke-width="1.2" {NS}/>'
            + f'<line x1="{-r * 0.65:.0f}" y1="{r * 0.65:.0f}" x2="{r * 0.65:.0f}" y2="{-r * 0.65:.0f}" stroke="{NOIR}" stroke-width="1.2" {NS}/>')


def pictogramme(gabarit: Dict, w: int = 112, h: int = 72) -> str:
    """Le symbole en vignette (bibliothèque), avec ses semelles."""
    body = symbole(gabarit)
    appuis = gabarit.get("appuis", [])
    x0, x1, y0, y1 = _emprise(appuis)
    sil = gabarit.get("silhouette", {})
    lw = max(x1 - x0 + 800, sil.get("longueur", 0.0) * 1e3 + 400, 1500.0)
    lh = max(y1 - y0 + 800, sil.get("largeur", 0.0) * 1e3 + 400, 1500.0)
    if gabarit.get("type_appuis") == "pieds":
        lw = max(lw, sil.get("fleche", 0.0) * 1e3 * 2 + 400)
    sc = min((w - 10) / lw, (h - 8) / lh)
    pads = "".join(f'<rect x="{a["dx"] * 1e3 - 300:.0f}" y="{a["dy"] * 1e3 - 300:.0f}" width="600" height="600" fill="white" stroke="{NOIR}" stroke-width="1" {NS}/>'
                   f'<rect x="{a["dx"] * 1e3 - a["a"] * 500:.0f}" y="{a["dy"] * 1e3 - a["b"] * 500:.0f}" width="{a["a"] * 1e3:.0f}" height="{a["b"] * 1e3:.0f}" fill="{NOIR}"/>'
                   for a in appuis) if gabarit.get("type_appuis") in ("pieds", "chenilles") else ""
    if gabarit.get("type_appuis") == "surfacique":
        body = f'<rect x="-900" y="-600" width="1800" height="1200" fill="none" stroke="#7c3aed" stroke-width="1.2" {NS}/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
            f'<g transform="translate({w / 2} {h / 2}) scale({sc:.5f})">{body}{pads}</g></svg>')

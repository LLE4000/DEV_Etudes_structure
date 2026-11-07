# -*- coding: utf-8 -*-
"""Les dessins du tableau des armatures, à l'échelle (unités : mm), dans le style des dessins du bureau (traits noirs,
cotes fines à traits obliques, acier plein) — celui de api_web/dessin_poutre.py et acier/garde_corps/schemas.py :

- ``nappe`` : un mètre de nappe en coupe, les barres à leur espacement et à leur diamètre réels ; chaque barre au
  milieu de la bande de largeur e qu'elle arme — c'est la définition de la section par mètre (a·100/e) ;
- ``diametres`` : les douze diamètres du tableau, à la même échelle, avec leur section et leur masse linéique ; le
  Ø retenu et celui de la disposition la plus économique marqués.

Bibliothèque standard seule ; les textes venus de l'état (rien ici) seraient échappés.
"""
import math
from html import escape
from typing import Optional

from . import tableau as T

ENCRE, ACIER, COTE, BANDE, BANDE2, ACCENT, OR = "#15181F", "#33415C", "#7F8C9A", "#E9EEF6", "#F4F6FA", "#2B4BCF", "#C98F00"
POLICE = "system-ui,Arial,sans-serif"


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _fr(x: float, nd: int = 0) -> str:
    """Un nombre à la française dans un dessin (espace fine des milliers)."""
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _txt(x, y, t, taille=16.0, ancre="middle", couleur=ENCRE, gras=True, extra="") -> str:
    poids = ' font-weight="600"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{t}</text>')


def _ligne(x1, y1, x2, y2, couleur=COTE, ep=1.0, extra="") -> str:
    return f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"{extra}/>'


def _cote_h(x1, x2, y, texte, taille, t=8.0, ep=1.6) -> str:
    """Une cote horizontale : la ligne, deux traits obliques, le nombre au-dessus."""
    return (_ligne(x1, y, x2, y, COTE, ep * 0.8) + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, ep)
            + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, ep) + _txt((x1 + x2) / 2, y - taille * 0.45, texte, taille))


def _etoile(cx, cy, r, couleur=OR) -> str:
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append(f"{_f(cx + rr * math.cos(a))},{_f(cy + rr * math.sin(a))}")
    return f'<polygon points="{" ".join(pts)}" fill="{couleur}"/>'


def _rupture(x1, x2, y, amp=14.0) -> str:
    """Une ligne de rupture horizontale (trait fin, un « Z » au milieu) : la nappe continue au-delà."""
    xm = (x1 + x2) / 2
    pts = [(x1, y), (xm - 18, y), (xm - 6, y - amp), (xm + 6, y + amp), (xm + 18, y), (x2, y)]
    return (f'<polyline points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in pts)}" fill="none" stroke="{ENCRE}" '
            f'stroke-width="1.4" stroke-linejoin="round"/>')


def nappe(ident: str, d: int, e_cm: float, titre: str, pied: str) -> str:
    """Un mètre de nappe : barres Ø ``d`` (mm) espacées de ``e_cm`` (cm), chacune au milieu de sa bande de largeur e
    (teintées une sur deux, la dernière coupée au mètre), en VUE EN PLAN (les barres à leur diamètre réel, la nappe
    coupée par deux lignes de rupture) et en COUPE A-A (les barres en section, les cotes). ``titre`` et ``pied`` sont
    du TEXTE (échappés ici)."""
    e = e_cm * 10.0                                  # mm
    L = 1000.0
    r = d / 2
    xs = []
    x = e / 2
    while x < L and len(xs) < 400:
        xs.append(x)
        x += e
    mixte = ' stroke-dasharray="26 7 4 7"'

    def bandes(y0, h):
        out, k, x0 = [], 0, 0.0
        while x0 < L - 1e-9 and k < 400:
            x1 = min(x0 + e, L)
            out.append(f'<rect x="{_f(x0)}" y="{_f(y0)}" width="{_f(x1 - x0)}" height="{_f(h)}" '
                       f'fill="{BANDE if k % 2 == 0 else BANDE2}" stroke="none"/>')
            k, x0 = k + 1, x0 + e
        return "".join(out)

    g = []
    # ---- la vue en plan : les barres, filantes, entre deux lignes de rupture
    yp0, yp1 = 150.0, 500.0
    g.append(_txt(-60, 116, "VUE EN PLAN", 26, ancre="start", couleur=COTE))
    g.append(bandes(yp0, yp1 - yp0))
    for x in xs:
        g.append(f'<rect x="{_f(x - r)}" y="{_f(yp0)}" width="{_f(d)}" height="{_f(yp1 - yp0)}" fill="{ENCRE}"/>')
    g.append(_rupture(-30, L + 30, yp0) + _rupture(-30, L + 30, yp1))
    g.append(_ligne(0, yp0 - 26, 0, yp1 + 26, ENCRE, 1.6, mixte) + _ligne(L, yp0 - 26, L, yp1 + 26, ENCRE, 1.6, mixte))
    # le trait de coupe A-A (traits forts aux bouts, flèches vers le bas : le sens de la vue)
    ya = yp1 - 70
    g.append(_ligne(-50, ya, L + 50, ya, ENCRE, 1.0, ' stroke-dasharray="20 6 4 6"'))
    for xa, ancre in ((-50, "end"), (L + 50, "start")):
        g.append(_ligne(xa, ya, xa + (22 if ancre == "end" else -22), ya, ENCRE, 4.0)
                 + f'<polygon points="{_f(xa - 6)},{_f(ya + 6)} {_f(xa + 6)},{_f(ya + 6)} {_f(xa)},{_f(ya + 24)}" fill="{ENCRE}"/>'
                 + _txt(xa + (-12 if ancre == "end" else 12), ya - 10, "A", 26, ancre=ancre))
    # ---- la coupe A-A : les barres en section, les cotes
    yc = 670.0
    h_b = 100.0
    # le titre de la coupe AU-DESSUS de l'axe de la première barre (x = 0, qui monte jusqu'à yc − h_b/2 − 24) : à
    # yc − 70, il était barré par ce trait mixte
    g.append(_txt(-60, yc - h_b / 2 - 36, "COUPE A-A", 26, ancre="start", couleur=COTE))
    g.append(bandes(yc - h_b / 2, h_b))
    g.append(_ligne(0, yc - h_b / 2 - 24, 0, yc + h_b / 2 + 24, ENCRE, 1.6, mixte)
             + _ligne(L, yc - h_b / 2 - 24, L, yc + h_b / 2 + 24, ENCRE, 1.6, mixte)
             + _ligne(-30, yc, L + 30, yc, COTE, 1.0, mixte))
    for x in xs:
        g.append(f'<circle cx="{_f(x)}" cy="{_f(yc)}" r="{_f(r)}" fill="{ENCRE}" stroke="{ENCRE}" stroke-width="1.5"/>')
    tc = 34.0
    y1 = yc + h_b / 2 + 66
    e_txt = _fr(e, 0 if e == int(e) else 1)
    if xs:
        g.append(_ligne(xs[0], yc + r + 6, xs[0], y1 + 12, COTE, 1.0, ' stroke-dasharray="6 5"'))
        g.append(_ligne(0, yc + h_b / 2 + 24, 0, y1 + 12, COTE, 1.0))
        if len(xs) > 1:
            g.append(_ligne(xs[1], yc + r + 6, xs[1], y1 + 12, COTE, 1.0, ' stroke-dasharray="6 5"'))
            g.append(_cote_h(xs[0], xs[1], y1, f"e = {e_txt}", tc))
        if e / 2 >= 60:
            g.append(_cote_h(0, xs[0], y1, _fr(e / 2, 0 if e / 2 == int(e / 2) else 1), tc * 0.85))
    y2 = y1 + 76
    g.append(_ligne(L, yc + h_b / 2 + 24, L, y2 + 12, COTE, 1.0) + _ligne(0, y1 + 12, 0, y2 + 12, COTE, 1.0))
    g.append(_cote_h(0, L, y2, "1 000 mm", tc))
    # le titre et le pied (le contenu de la bande : barres par mètre, section, masse)
    haut = _txt(L / 2, 44, escape(titre), 38)
    note = _txt(L / 2, 84, f"chaque barre arme une bande de largeur e = {e_txt} mm", 26, couleur=COTE, gras=False)
    bas = _txt(L / 2, y2 + 78, escape(pied), 32, gras=False)
    vb = f"-90 0 {_f(L + 180)} {_f(y2 + 108)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Un mètre de nappe, en plan et en coupe">{haut}{note}{"".join(g)}{bas}</svg>')


def diametres(ident: str, retenu: Optional[int] = None, optimal: Optional[int] = None) -> str:
    """Les douze diamètres du tableau à la même échelle (2:1 — deux unités du dessin par mm), trois rangées de quatre,
    chacun avec sa section (mm²) et sa masse linéique (kg/m) — les valeurs du tableau. Le Ø retenu est encadré en
    bleu, celui de la plus économique porte une étoile."""
    k = 2.0                                          # l'échelle du dessin : 2:1
    col, haut_c, larg_c = 4, 170.0, 130.0
    g = []
    for i, d in enumerate(T.DIAMETRES):
        cx = larg_c * (i % col) + larg_c / 2
        y0 = haut_c * (i // col)
        cy = y0 + 54
        rr = k * d / 2
        if d == retenu:
            g.append(f'<rect x="{_f(cx - larg_c / 2 + 5)}" y="{_f(y0 + 5)}" width="{_f(larg_c - 10)}" height="{_f(haut_c - 10)}" rx="9" '
                     f'fill="#EEF1FE" stroke="{ACCENT}" stroke-width="1.6"/>')
        g.append(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(rr)}" fill="{ENCRE}"/>')
        # les axes de la barre, en trait mixte, dépassent le cercle
        g.append(_ligne(cx - rr - 8, cy, cx + rr + 8, cy, COTE, 0.7, ' stroke-dasharray="8 2.5 1.5 2.5"')
                 + _ligne(cx, cy - rr - 8, cx, cy + rr + 8, COTE, 0.7, ' stroke-dasharray="8 2.5 1.5 2.5"'))
        couleur = ACCENT if d == retenu else ENCRE
        g.append(_txt(cx, y0 + 122, f"Ø{d}", 17, couleur=couleur))
        if d == optimal:
            g.append(_etoile(cx + 27, y0 + 116, 7))
        g.append(_txt(cx, y0 + 140, f"{_fr(T.arrondi(T.aire(d)))} mm²", 12, gras=False))
        g.append(_txt(cx, y0 + 155, f"{T.masse_txt(T.masse_lineique(d))} kg/m", 12, couleur=COTE, gras=False))
    h = haut_c * math.ceil(len(T.DIAMETRES) / col)
    # l'échelle : un trait de 20 mm (40 unités)
    xe = larg_c * col - 20
    ech = (_ligne(xe - k * 20, h + 16, xe, h + 16, ENCRE, 1.6) + _ligne(xe - k * 20, h + 11, xe - k * 20, h + 21, ENCRE, 1.4)
           + _ligne(xe, h + 11, xe, h + 21, ENCRE, 1.4) + _txt(xe - k * 20 - 8, h + 20, "20 mm · tous à la même échelle", 12, ancre="end", gras=False))
    note = _txt(12, h + 20, f"ρ = {_fr(T.MASSE_VOLUMIQUE)} kg/m³ (le tableau du bureau)", 12, ancre="start", couleur=COTE, gras=False)
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(larg_c * col)} {_f(h + 34)}" width="100%" '
            f'role="img" aria-label="Les diamètres du tableau, à l\'échelle">{"".join(g)}{ech}{note}</svg>')

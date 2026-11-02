# -*- coding: utf-8 -*-
"""Peintre du SVG des dessins (plan, coupe, balayage) sur un canevas
reportlab : la note PDF montre LE MÊME dessin que l'écran, sans seconde
implémentation (R2 — source unique).

Sous-ensemble SVG couvert, celui qu'émettent ``dessin.py`` et
``symboles.py`` : ``rect``, ``circle``, ``line``, ``polygon``,
``polyline``, ``text``, ``g`` avec ``transform`` (translate, rotate,
matrix, scale, composés), ``fill`` / ``stroke`` / ``stroke-width`` /
``stroke-dasharray`` / ``opacity``, ``text-anchor`` et ``font-weight``.
Ignoré : ``style``, ``defs`` (motifs, marqueurs), ``title``, coins
arrondis, ``vector-effect`` (les traits sont pris en points, bornés).
Un motif de hachures (``fill="url(#…)"``) devient un aplat clair.

Repère : le SVG a l'ordonnée vers le bas, le PDF vers le haut ; la matrice
de base retourne l'axe y, et les textes sont posés à part, à l'endroit,
avec l'angle de la matrice courante (un ``rotate(-90)`` du SVG donne un
texte qui monte).
"""
import math
import re
import xml.etree.ElementTree as ET_
from typing import Optional, Tuple

from reportlab.lib.colors import HexColor

from ndc_pdf.fonts import draw_text, register_all, string_width

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TRANS = re.compile(r"(translate|rotate|matrix|scale)\s*\(([^)]*)\)")
MOTIFS = {"url(#chp_hc)": "#D7DBE1", "url(#chp_hs)": "#E8DDF8"}     # hachures du plan : aplats clairs
POLICE, POLICE_GRAS = "Carlito", "Carlito-Bold"


class Mat:
    """Matrice affine SVG (a b c d e f) : x' = a·x + c·y + e, y' = b·x + d·y + f."""
    __slots__ = ("a", "b", "c", "d", "e", "f")

    def __init__(self, a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f

    def mul(self, o: "Mat") -> "Mat":
        """self · o : ``o`` s'applique d'abord au point, puis ``self``."""
        return Mat(self.a * o.a + self.c * o.b, self.b * o.a + self.d * o.b,
                   self.a * o.c + self.c * o.d, self.b * o.c + self.d * o.d,
                   self.a * o.e + self.c * o.f + self.e, self.b * o.e + self.d * o.f + self.f)

    def pt(self, x: float, y: float) -> Tuple[float, float]:
        return self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f

    def echelle(self) -> float:
        return math.sqrt(abs(self.a * self.d - self.b * self.c))

    def angle(self) -> float:
        """Angle (°) de l'axe x local dans le repère du canevas."""
        return math.degrees(math.atan2(self.b, self.a))


def transformation(s: Optional[str]) -> Mat:
    """La matrice d'un attribut ``transform`` (composition de gauche à droite)."""
    m = Mat()
    for nom, args in _TRANS.findall(s or ""):
        v = [float(x) for x in _NUM.findall(args)]
        if nom == "translate":
            t = Mat(e=v[0], f=v[1] if len(v) > 1 else 0.0)
        elif nom == "scale":
            t = Mat(a=v[0], d=v[1] if len(v) > 1 else v[0])
        elif nom == "matrix" and len(v) == 6:
            t = Mat(*v)
        elif nom == "rotate":
            th = math.radians(v[0])
            r = Mat(math.cos(th), math.sin(th), -math.sin(th), math.cos(th))
            if len(v) == 3:
                r = Mat(e=v[1], f=v[2]).mul(r).mul(Mat(e=-v[1], f=-v[2]))
            t = r
        else:
            continue
        m = m.mul(t)
    return m


def _tag(el) -> str:
    return el.tag.split("}")[-1]


def _f(el, nom, defaut=0.0) -> float:
    v = el.get(nom)
    if v is None:
        return defaut
    m = _NUM.search(v)
    return float(m.group(0)) if m else defaut


def _couleur(v: Optional[str], defaut=None):
    if v is None:
        return defaut
    v = v.strip()
    if v == "none":
        return None
    if v.startswith("url("):
        return HexColor(MOTIFS.get(v, "#E3E5EA"))
    if v.startswith("#"):
        return HexColor(v if len(v) == 7 else "#" + "".join(ch * 2 for ch in v[1:4]))
    return HexColor({"white": "#FFFFFF", "black": "#000000"}.get(v, "#000000"))


def _bbox_svg(svg: str) -> Tuple[float, float, float, float]:
    root = ET_.fromstring(svg)
    vb = [float(x) for x in _NUM.findall(root.get("viewBox", "0 0 100 100"))]
    return (vb[0], vb[1], vb[2], vb[3]) if len(vb) == 4 else (0.0, 0.0, 100.0, 100.0)


def peindre_svg(c, svg: str, x: float, y: float, w: float, h: float, marge: float = 2.0,
                police: str = POLICE, police_gras: str = POLICE_GRAS) -> float:
    """Peint ``svg`` dans le rectangle ``(x, y, w, h)`` du canevas (origine en
    bas à gauche), à l'échelle, centré. Rend l'échelle (points par unité SVG)."""
    register_all()
    root = ET_.fromstring(svg)
    vx, vy, vw, vh = _bbox_svg(svg)
    s = min((w - 2 * marge) / vw, (h - 2 * marge) / vh)
    ox = x + (w - vw * s) / 2
    oy = y + h - (h - vh * s) / 2
    base = Mat(s, 0.0, 0.0, -s, ox - vx * s, oy + vy * s)
    c.saveState()
    _peindre_noeud(c, root, base, s, police, police_gras)
    c.restoreState()
    return s


def _peindre_noeud(c, el, ctm: Mat, s: float, police: str, police_gras: str) -> None:
    for ch in list(el):
        tag = _tag(ch)
        if tag in ("style", "defs", "title", "pattern", "marker", "desc", "metadata"):
            continue
        if ch.get("opacity") is not None and _f(ch, "opacity", 1.0) <= 0.0:
            continue
        m = ctm.mul(transformation(ch.get("transform")))
        if tag == "g" or tag == "svg":
            _peindre_noeud(c, ch, m, s, police, police_gras)
            continue
        fill = _couleur(ch.get("fill"), HexColor("#000000"))
        stroke = _couleur(ch.get("stroke"), None)
        if tag == "text":
            _texte(c, ch, m, fill, police, police_gras)
            continue
        lw = min(1.4, max(0.3, _f(ch, "stroke-width", 1.0) * 0.45)) if stroke is not None else 0.0
        dash = [max(0.4, float(v) * s) for v in _NUM.findall(ch.get("stroke-dasharray") or "")] or None
        if tag == "rect":
            x0, y0, wd, ht = _f(ch, "x"), _f(ch, "y"), _f(ch, "width"), _f(ch, "height")
            pts = [m.pt(x0, y0), m.pt(x0 + wd, y0), m.pt(x0 + wd, y0 + ht), m.pt(x0, y0 + ht)]
            _poly(c, pts, fill, stroke, lw, dash, True)
        elif tag == "circle":
            cx, cy = m.pt(_f(ch, "cx"), _f(ch, "cy"))
            _style(c, fill, stroke, lw, dash)
            c.circle(cx, cy, _f(ch, "r") * m.echelle(), stroke=1 if stroke is not None else 0, fill=1 if fill is not None else 0)
            _fin_dash(c, dash)
        elif tag == "line":
            if stroke is None:
                continue
            (x1, y1), (x2, y2) = m.pt(_f(ch, "x1"), _f(ch, "y1")), m.pt(_f(ch, "x2"), _f(ch, "y2"))
            _style(c, None, stroke, lw, dash)
            c.line(x1, y1, x2, y2)
            _fin_dash(c, dash)
        elif tag in ("polygon", "polyline"):
            v = [float(t) for t in _NUM.findall(ch.get("points") or "")]
            pts = [m.pt(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
            if len(pts) >= 2:
                _poly(c, pts, fill if tag == "polygon" else None, stroke, lw, dash, tag == "polygon")


def _style(c, fill, stroke, lw, dash):
    if fill is not None:
        c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke)
        c.setLineWidth(lw)
    if dash:
        c.setDash(dash)


def _fin_dash(c, dash):
    if dash:
        c.setDash()


def _poly(c, pts, fill, stroke, lw, dash, fermer):
    _style(c, fill, stroke, lw, dash)
    p = c.beginPath()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    if fermer:
        p.close()
    c.drawPath(p, stroke=1 if stroke is not None else 0, fill=1 if (fill is not None and fermer) else 0)
    _fin_dash(c, dash)


def _texte(c, el, m: Mat, fill, police, police_gras):
    txt = "".join(el.itertext()).strip()
    if not txt:
        return
    taille = _f(el, "font-size", 12.0) * m.echelle()
    if taille < 2.2:
        return
    gras = _f(el, "font-weight", 400.0) >= 600
    f = police_gras if gras else police
    x, y = m.pt(_f(el, "x"), _f(el, "y"))
    ancre = el.get("text-anchor", "start")
    align = {"middle": "center", "end": "right"}.get(ancre, "left")
    ang = m.angle()
    c.saveState()
    c.translate(x, y)
    if abs(ang) > 0.5:
        c.rotate(ang)
    draw_text(c, 0.0, 0.0, txt, f, taille, fill if fill is not None else HexColor("#000000"), align)
    c.restoreState()


def largeur_texte(txt: str, taille: float, gras: bool = False) -> float:
    return string_width(txt, POLICE_GRAS if gras else POLICE, taille)

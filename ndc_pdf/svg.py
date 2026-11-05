# -*- coding: utf-8 -*-
"""Peintre SVG → canevas reportlab, partagé par les notes : la note PDF
montre LE MÊME dessin que l'écran, sans seconde implémentation (source
unique).

Sous-ensemble SVG couvert, celui qu'émettent les modules de dessin de
l'application : ``rect``, ``circle``, ``line``, ``polygon``, ``polyline``,
``path`` (M L H V Z, arcs A échantillonnés, courbes C Q échantillonnées,
``fill-rule`` evenodd), ``text`` (``text-anchor``, ``font-weight``, ``dy``),
``g`` avec ``transform`` (translate, rotate, matrix, scale, composés),
``fill`` / ``stroke`` / ``stroke-width`` / ``stroke-dasharray`` / ``opacity``.
Un ``fill="url(#motif)"`` est peint en HACHURES (lignes du motif, découpées
par la forme) quand le ``<pattern>`` est défini dans le SVG, en aplat clair
sinon (ou si ``motifs`` impose une couleur) ; un ``stroke="url(#dégradé)"``
d'un ``line`` est peint par tronçons aux couleurs du ``<linearGradient>``.
Ignoré : ``style``, ``marker``, ``title``, coins arrondis, ``vector-effect``
(les traits sont pris en points, bornés).

Repère : le SVG a l'ordonnée vers le bas, le PDF vers le haut ; la matrice
de base retourne l'axe y, et les textes sont posés à part, à l'endroit,
avec l'angle de la matrice courante (un ``rotate(-90)`` du SVG donne un
texte qui monte).
"""
import math
import re
import xml.etree.ElementTree as ET_
from typing import Dict, List, Optional, Tuple

from reportlab.lib.colors import Color, HexColor

from .fonts import draw_text, register_all, string_width

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TRANS = re.compile(r"(translate|rotate|matrix|scale)\s*\(([^)]*)\)")
_PATH = re.compile(r"[MmLlHhVvZzAaCcQqSsTt]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
POLICE, POLICE_GRAS = "Carlito", "Carlito-Bold"
APLAT = "#E3E5EA"


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

    def vec(self, x: float, y: float) -> Tuple[float, float]:
        """Un vecteur (sans translation)."""
        return self.a * x + self.c * y, self.b * x + self.d * y

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


def _hex(v: str):
    v = v.strip()
    if v.startswith("#"):
        return HexColor(v if len(v) == 7 else "#" + "".join(ch * 2 for ch in v[1:4]))
    return HexColor({"white": "#FFFFFF", "black": "#000000", "none": "#000000"}.get(v, "#000000"))


def _bbox_svg(svg: str) -> Tuple[float, float, float, float]:
    root = ET_.fromstring(svg)
    vb = [float(x) for x in _NUM.findall(root.get("viewBox", "0 0 100 100"))]
    return (vb[0], vb[1], vb[2], vb[3]) if len(vb) == 4 else (0.0, 0.0, 100.0, 100.0)


class _Ctx:
    def __init__(self, c, s, police, police_gras, motifs):
        self.c, self.s, self.police, self.police_gras = c, s, police, police_gras
        self.motifs = motifs or {}
        self.degrades: Dict[str, Dict] = {}
        self.patterns: Dict[str, Dict] = {}


def _lire_defs(ctx: _Ctx, root) -> None:
    """Dégradés (arrêts) et motifs (hachures) de tout le document."""
    for el in root.iter():
        tag = _tag(el)
        if tag == "linearGradient" and el.get("id"):
            stops = []
            for st in el:
                if _tag(st) == "stop":
                    o = st.get("offset", "0")
                    off = float(o[:-1]) / 100.0 if o.endswith("%") else float(o or 0)
                    stops.append((off, _hex(st.get("stop-color", "#000000"))))
            ctx.degrades[el.get("id")] = {"x1": _f(el, "x1"), "y1": _f(el, "y1"), "x2": _f(el, "x2"), "y2": _f(el, "y2"),
                                          "stops": stops or [(0.0, HexColor("#000000"))]}
        elif tag == "pattern" and el.get("id"):
            ligne = next((ch for ch in el if _tag(ch) == "line"), None)
            if ligne is None:
                continue
            rot = 0.0
            mm = re.search(r"rotate\(\s*([-+]?[\d.]+)", el.get("patternTransform", "") or "")
            if mm:
                rot = float(mm.group(1))
            dx, dy = _f(ligne, "x2") - _f(ligne, "x1"), _f(ligne, "y2") - _f(ligne, "y1")
            ctx.patterns[el.get("id")] = {"pas": max(0.5, _f(el, "width", 6.0)), "rot": rot, "dir": (dx, dy),
                                          "couleur": _hex(ligne.get("stroke", "#888888")), "lw": _f(ligne, "stroke-width", 0.8)}


def _ref(v: Optional[str]) -> Optional[str]:
    if v and v.strip().startswith("url("):
        return v.strip()[4:].rstrip(")").strip().lstrip("#")
    return None


def _couleur(v: Optional[str], defaut=None, motifs: Optional[Dict] = None):
    if v is None:
        return defaut
    v = v.strip()
    if v == "none":
        return None
    if v.startswith("url("):
        return HexColor((motifs or {}).get(v, APLAT))
    return _hex(v)


def peindre_svg(c, svg: str, x: float, y: float, w: float, h: float, marge: float = 2.0,
                police: str = POLICE, police_gras: str = POLICE_GRAS, motifs: Optional[Dict] = None) -> float:
    """Peint ``svg`` dans le rectangle ``(x, y, w, h)`` du canevas (origine en
    bas à gauche), à l'échelle, centré. Rend l'échelle (points par unité SVG).
    ``motifs`` : {"url(#id)": "#couleur"} impose un aplat à un motif."""
    register_all()
    root = ET_.fromstring(svg)
    vx, vy, vw, vh = _bbox_svg(svg)
    s = min((w - 2 * marge) / vw, (h - 2 * marge) / vh)
    ox = x + (w - vw * s) / 2
    oy = y + h - (h - vh * s) / 2
    base = Mat(s, 0.0, 0.0, -s, ox - vx * s, oy + vy * s)
    ctx = _Ctx(c, s, police, police_gras, motifs)
    _lire_defs(ctx, root)
    c.saveState()
    _peindre_noeud(ctx, root, base)
    c.restoreState()
    return s


def _peindre_noeud(ctx: _Ctx, el, ctm: Mat) -> None:
    c = ctx.c
    for ch in list(el):
        tag = _tag(ch)
        if tag in ("style", "defs", "title", "pattern", "marker", "desc", "metadata", "linearGradient", "radialGradient"):
            continue
        if ch.get("opacity") is not None and _f(ch, "opacity", 1.0) <= 0.0:
            continue
        m = ctm.mul(transformation(ch.get("transform")))
        if tag == "g" or tag == "svg":
            _peindre_noeud(ctx, ch, m)
            continue
        if tag == "text":
            _texte(ctx, ch, m, _couleur(ch.get("fill"), HexColor("#000000"), ctx.motifs))
            continue
        fill_v, stroke_v = ch.get("fill"), ch.get("stroke")
        motif = None
        rid = _ref(fill_v)
        if rid and f"url(#{rid})" not in ctx.motifs and rid in ctx.patterns:
            motif = ctx.patterns[rid]
            fill = None
        else:
            fill = _couleur(fill_v, HexColor("#000000"), ctx.motifs)
        degrade = ctx.degrades.get(_ref(stroke_v) or "")
        stroke = _couleur(stroke_v, None, ctx.motifs) if degrade is None else None
        lw = min(1.4, max(0.3, _f(ch, "stroke-width", 1.0) * 0.45)) if (stroke is not None or degrade) else 0.0
        dash = [max(0.4, float(v) * ctx.s) for v in _NUM.findall(ch.get("stroke-dasharray") or "")] or None
        evenodd = (ch.get("fill-rule") or "").strip() == "evenodd"
        if tag == "rect":
            x0, y0, wd, ht = _f(ch, "x"), _f(ch, "y"), _f(ch, "width"), _f(ch, "height")
            pts = [m.pt(x0, y0), m.pt(x0 + wd, y0), m.pt(x0 + wd, y0 + ht), m.pt(x0, y0 + ht)]
            _formes(ctx, [(pts, True)], fill, stroke, lw, dash, motif, m, False)
        elif tag == "circle":
            cx, cy = m.pt(_f(ch, "cx"), _f(ch, "cy"))
            r = _f(ch, "r") * m.echelle()
            _style(c, fill, stroke, lw, dash)
            c.circle(cx, cy, r, stroke=1 if stroke is not None else 0, fill=1 if fill is not None else 0)
            _fin_dash(c, dash)
        elif tag == "line":
            (x1, y1), (x2, y2) = m.pt(_f(ch, "x1"), _f(ch, "y1")), m.pt(_f(ch, "x2"), _f(ch, "y2"))
            if degrade is not None:
                _ligne_degradee(ctx, ch, m, degrade, lw, dash)
            elif stroke is not None:
                _style(c, None, stroke, lw, dash)
                c.line(x1, y1, x2, y2)
                _fin_dash(c, dash)
        elif tag in ("polygon", "polyline"):
            v = [float(t) for t in _NUM.findall(ch.get("points") or "")]
            pts = [m.pt(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
            if len(pts) >= 2:
                _formes(ctx, [(pts, tag == "polygon")], fill if tag == "polygon" else None, stroke, lw, dash, motif, m, False)
        elif tag == "path":
            sous = _chemin(ch.get("d") or "")
            sous = [([m.pt(px, py) for px, py in pts], ferme) for pts, ferme in sous if len(pts) >= 2]
            if sous:
                _formes(ctx, sous, fill, stroke, lw, dash, motif, m, evenodd)


# ---------------------------------------------------------------- formes, hachures, dégradés
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


def _path(c, sous):
    p = c.beginPath()
    for pts, ferme in sous:
        p.moveTo(*pts[0])
        for q in pts[1:]:
            p.lineTo(*q)
        if ferme:
            p.close()
    return p


def _formes(ctx: _Ctx, sous: List, fill, stroke, lw, dash, motif, m: Mat, evenodd: bool):
    """Un ou plusieurs sous-chemins : aplat et/ou hachures (découpées), puis le contour."""
    c = ctx.c
    remplir = fill is not None and any(f for _, f in sous)
    if remplir:
        _style(c, fill, None, 0, None)
        c.drawPath(_path(c, sous), stroke=0, fill=1, fillMode=0 if evenodd else 1)
    if motif is not None and any(f for _, f in sous):
        _hachurer(ctx, sous, motif, m, evenodd)
    if stroke is not None:
        _style(c, None, stroke, lw, dash)
        c.drawPath(_path(c, sous), stroke=1, fill=0)
        _fin_dash(c, dash)


def _hachurer(ctx: _Ctx, sous: List, motif: Dict, m: Mat, evenodd: bool):
    """Les lignes du motif, dans la direction du motif tourné, découpées par la forme."""
    c = ctx.c
    pts = [p for ps, _ in sous for p in ps]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not pts or max(xs) - min(xs) < 0.5 or max(ys) - min(ys) < 0.5:
        return
    rot = math.radians(motif["rot"])
    dxs, dys = motif["dir"]
    # direction des lignes dans le repère du motif tourné, puis dans le canevas (partie linéaire de m)
    dx = dxs * math.cos(rot) - dys * math.sin(rot)
    dy = dxs * math.sin(rot) + dys * math.cos(rot)
    ux, uy = m.vec(dx, dy)
    n = math.hypot(ux, uy) or 1.0
    ux, uy = ux / n, uy / n
    nx, ny = -uy, ux
    pas = motif["pas"] * m.echelle()
    if pas < 0.6:
        return
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 2
    offs = [((px - cx) * nx + (py - cy) * ny) for px, py in pts]
    c.saveState()
    c.clipPath(_path(c, sous), stroke=0, fill=0, fillMode=0 if evenodd else 1)
    c.setStrokeColor(motif["couleur"])
    c.setLineWidth(min(0.8, max(0.25, motif["lw"] * 0.45)))
    k0 = math.floor(min(offs) / pas) - 1
    k1 = math.ceil(max(offs) / pas) + 1
    for k in range(k0, k1 + 1):
        o = k * pas
        bx, by = cx + nx * o, cy + ny * o
        c.line(bx - ux * diag, by - uy * diag, bx + ux * diag, by + uy * diag)
    c.restoreState()


def _ligne_degradee(ctx: _Ctx, el, m: Mat, deg: Dict, lw: float, dash):
    """Une ligne au trait dégradé : huit tronçons aux couleurs interpolées entre les arrêts."""
    c = ctx.c
    x1, y1, x2, y2 = _f(el, "x1"), _f(el, "y1"), _f(el, "x2"), _f(el, "y2")
    gx, gy = deg["x2"] - deg["x1"], deg["y2"] - deg["y1"]
    g2 = gx * gx + gy * gy
    stops = deg["stops"]

    def couleur(t):
        t = min(1.0, max(0.0, t))
        for (o1, c1), (o2, c2) in zip(stops, stops[1:]):
            if t <= o2:
                u = 0.0 if o2 <= o1 else (t - o1) / (o2 - o1)
                return Color(c1.red + (c2.red - c1.red) * u, c1.green + (c2.green - c1.green) * u, c1.blue + (c2.blue - c1.blue) * u)
        return stops[-1][1]
    n = 8
    c.setLineWidth(lw)
    if dash:
        c.setDash(dash)
    for k in range(n):
        ta, tb = k / n, (k + 1) / n
        pa = (x1 + (x2 - x1) * ta, y1 + (y2 - y1) * ta)
        pb = (x1 + (x2 - x1) * tb, y1 + (y2 - y1) * tb)
        mid = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        t = ((mid[0] - deg["x1"]) * gx + (mid[1] - deg["y1"]) * gy) / g2 if g2 > 0 else 0.0
        c.setStrokeColor(couleur(t))
        (ax, ay), (bx, by) = m.pt(*pa), m.pt(*pb)
        c.line(ax, ay, bx, by)
    if dash:
        c.setDash()


# ---------------------------------------------------------------- chemins
def _arc(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2) -> List[Tuple[float, float]]:
    """Points d'un arc elliptique SVG (paramétrage par les extrémités, F.6.5), cordes de 15° au plus."""
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return [(x2, y2)]
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1.0:
        rx *= math.sqrt(lam)
        ry *= math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den)) if den > 0 else 0.0
    if fa == fs:
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx, cy = cp * cxp - sp * cyp + (x1 + x2) / 2.0, sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    th1 = ang(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not fs and dth > 0:
        dth -= 2 * math.pi
    elif fs and dth < 0:
        dth += 2 * math.pi
    n = max(4, int(math.ceil(abs(dth) / (math.pi / 12))))
    out = []
    for k in range(1, n + 1):
        t = th1 + dth * k / n
        ct, st_ = math.cos(t), math.sin(t)
        out.append((cx + rx * ct * cp - ry * st_ * sp, cy + rx * ct * sp + ry * st_ * cp))
    out[-1] = (x2, y2)
    return out


def _bezier(p0, p1, p2, p3=None, n=10) -> List[Tuple[float, float]]:
    out = []
    for k in range(1, n + 1):
        t = k / n
        if p3 is None:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        else:
            x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t * t * p2[0] + t ** 3 * p3[0]
            y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t * t * p2[1] + t ** 3 * p3[1]
        out.append((x, y))
    return out


def _chemin(d: str) -> List[Tuple[List[Tuple[float, float]], bool]]:
    """Les sous-chemins (points, fermé) d'un attribut ``d``."""
    toks = _PATH.findall(d)
    sous: List = []
    pts: List = []
    ferme = False
    cmd = None
    i = 0
    cur = (0.0, 0.0)
    debut = (0.0, 0.0)

    def nb(k):
        return float(toks[k])

    def nouveau():
        nonlocal pts, ferme
        if pts:
            sous.append((pts, ferme))
        pts, ferme = [], False
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                ferme = True
                cur = debut
                nouveau()
                continue
        if cmd is None:
            i += 1
            continue
        rel = cmd.islower()
        C = cmd.upper()
        try:
            if C == "M":
                x, y = nb(i), nb(i + 1)
                i += 2
                cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
                nouveau()
                debut = cur
                pts = [cur]
                cmd = "l" if rel else "L"
            elif C == "L":
                x, y = nb(i), nb(i + 1)
                i += 2
                cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
                pts.append(cur)
            elif C == "H":
                x = nb(i)
                i += 1
                cur = (cur[0] + x if rel else x, cur[1])
                pts.append(cur)
            elif C == "V":
                y = nb(i)
                i += 1
                cur = (cur[0], cur[1] + y if rel else y)
                pts.append(cur)
            elif C == "A":
                rx, ry, phi, fa, fs, x, y = (nb(i + k) for k in range(7))
                i += 7
                fin = (cur[0] + x, cur[1] + y) if rel else (x, y)
                pts.extend(_arc(cur[0], cur[1], rx, ry, phi, int(fa), int(fs), fin[0], fin[1]))
                cur = fin
            elif C == "C":
                v = [nb(i + k) for k in range(6)]
                i += 6
                if rel:
                    v = [v[0] + cur[0], v[1] + cur[1], v[2] + cur[0], v[3] + cur[1], v[4] + cur[0], v[5] + cur[1]]
                pts.extend(_bezier(cur, (v[0], v[1]), (v[2], v[3]), (v[4], v[5])))
                cur = (v[4], v[5])
            elif C == "Q":
                v = [nb(i + k) for k in range(4)]
                i += 4
                if rel:
                    v = [v[0] + cur[0], v[1] + cur[1], v[2] + cur[0], v[3] + cur[1]]
                pts.extend(_bezier(cur, (v[0], v[1]), (v[2], v[3])))
                cur = (v[2], v[3])
            else:                      # S, T : approchés par un segment
                n_args = 4 if C == "S" else 2
                v = [nb(i + k) for k in range(n_args)]
                i += n_args
                fin = (cur[0] + v[-2], cur[1] + v[-1]) if rel else (v[-2], v[-1])
                pts.append(fin)
                cur = fin
        except (IndexError, ValueError):
            break
    nouveau()
    return sous


# ---------------------------------------------------------------- textes
def _texte(ctx: _Ctx, el, m: Mat, fill):
    c = ctx.c
    txt = "".join(el.itertext()).strip()
    if not txt:
        return
    taille = _f(el, "font-size", 12.0) * m.echelle()
    if taille < 2.2:
        return
    gras = _f(el, "font-weight", 400.0) >= 600
    f = ctx.police_gras if gras else ctx.police
    x, y = m.pt(_f(el, "x") + _f(el, "dx", 0.0), _f(el, "y") + _f(el, "dy", 0.0))
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

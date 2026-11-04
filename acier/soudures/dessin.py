# -*- coding: utf-8 -*-
"""Les schémas des soudures en SVG : la configuration vue du bon côté, les
pièces en gris, les cordons en orange colorés par leur taux, les efforts en
rouge avec leur sens, le point gouvernant, et l'encart « la gorge » qui
montre σ⊥, τ⊥, τ∥ au point le plus chargé (EN 1993-1-8 fig. 4.5).

Tout est dessiné depuis les objets du calcul (cordons, pièces des
constructeurs, résultats) : le dessin est une vue, jamais une seconde
géométrie. Unités du dessin : mm du plan de joint, mis à l'échelle dans un
canevas en pixels. Palette : pièces #CBD5E1 / #E2E8F0, trait #1F2937,
cordons #D98A00, efforts #9C3341, taux vert #2E7D46 / ocre #D98A00 / rouge #9C3341.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ENCRE, PIECE_A, PIECE_B, CORDON, EFFORT, MUT, FOND = "#1F2937", "#CBD5E1", "#E2E8F0", "#D98A00", "#9C3341", "#6E7480", "#F8FAFC"
OK, ATT, NOK = "#2E7D46", "#D98A00", "#9C3341"
POLICE = 'font-family="Inter, Segoe UI, Arial, sans-serif"'


def _fmt(v: float, n: int = 1) -> str:
    s = f"{v:,.{n}f}".replace(",", " ").replace(".", ",")
    return s


def couleur_taux(t: float) -> str:
    """Vert jusqu'à 70 %, dégradé vers l'ocre à 100 %, rouge au-delà."""
    if t is None:
        return MUT
    if t >= 1.0:
        return NOK
    if t <= 0.7:
        return OK
    u = (t - 0.7) / 0.3
    def mix(a, b):
        return int(round(a + (b - a) * u))
    c1, c2 = (0x2E, 0x7D, 0x46), (0xD9, 0x8A, 0x00)
    return "#%02X%02X%02X" % tuple(mix(a, b) for a, b in zip(c1, c2))


class Canevas:
    """Un repère mm → px : origine et échelle communes à tout le dessin."""

    def __init__(self, largeur_px: int, hauteur_px: int, xmin: float, xmax: float, ymin: float, ymax: float, marge: float = 36.0):
        self.W, self.H = largeur_px, hauteur_px
        self.ech = min((largeur_px - 2 * marge) / max(1e-9, xmax - xmin), (hauteur_px - 2 * marge) / max(1e-9, ymax - ymin))
        self.x0 = marge + ((largeur_px - 2 * marge) - (xmax - xmin) * self.ech) / 2 - xmin * self.ech
        self.y0 = marge + ((hauteur_px - 2 * marge) - (ymax - ymin) * self.ech) / 2 + ymax * self.ech
        self.out: List[str] = []

    def X(self, x: float) -> float:
        return self.x0 + x * self.ech

    def Y(self, y: float) -> float:
        return self.y0 - y * self.ech

    # ---- primitives
    def polygone(self, pts, fill=PIECE_B, stroke=ENCRE, sw=1.0, dash=None, opacity=1.0):
        d = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<polygon points="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da} fill-opacity="{opacity}"/>')

    def ligne(self, p, q, stroke=ENCRE, sw=1.0, dash=None, cap="butt"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<line x1="{self.X(p[0]):.1f}" y1="{self.Y(p[1]):.1f}" x2="{self.X(q[0]):.1f}" y2="{self.Y(q[1]):.1f}" '
                        f'stroke="{stroke}" stroke-width="{sw}"{da} stroke-linecap="{cap}"/>')

    def texte(self, x, y, t, taille=11, fill=ENCRE, anc="middle", gras=False, dx=0.0, dy=0.0, rot=None):
        g = ' font-weight="600"' if gras else ""
        tr = f' transform="rotate({rot} {self.X(x) + dx:.1f} {self.Y(y) + dy:.1f})"' if rot is not None else ""
        self.out.append(f'<text x="{self.X(x) + dx:.1f}" y="{self.Y(y) + dy:.1f}" font-size="{taille}" fill="{fill}" '
                        f'text-anchor="{anc}"{g}{tr}>{t}</text>')

    def fleche(self, p, q, label="", stroke=EFFORT, sw=2.2, taille=11.5, pos_label="fin"):
        """Flèche droite de p vers q (mm), pointe en q, libellé près de la queue."""
        x1, y1, x2, y2 = self.X(p[0]), self.Y(p[1]), self.X(q[0]), self.Y(q[1])
        self._fleche_px(x1, y1, x2, y2, label, stroke, sw, taille, pos_label)

    def _fleche_px(self, x1, y1, x2, y2, label="", stroke=EFFORT, sw=2.2, taille=11.5, pos_label="fin"):
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        ux, uy = dx / L, dy / L
        h = 9.0
        bx, by = x2 - ux * h, y2 - uy * h
        self.out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.out.append(f'<polygon points="{x2:.1f},{y2:.1f} {bx - uy * 4.5:.1f},{by + ux * 4.5:.1f} {bx + uy * 4.5:.1f},{by - ux * 4.5:.1f}" fill="{stroke}"/>')
        if label:
            if pos_label == "fin":
                lx, ly = x2 + ux * 6 - uy * 10, y2 + uy * 6 + ux * 10
            else:
                lx, ly = x1 - ux * 6 - uy * 10, y1 - uy * 6 + ux * 10
            anc = "start" if ux >= 0 and abs(ux) > 0.3 else ("end" if ux < -0.3 else "middle")
            self.out.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" font-size="{taille}" fill="{stroke}" text-anchor="{anc}" font-weight="600">{label}</text>')

    def moment(self, c, r_px, label="", sens=+1, stroke=EFFORT, taille=11.5):
        """Flèche courbe autour du point c (mm), rayon en px ; sens +1 trigonométrique."""
        cx, cy = self.X(c[0]), self.Y(c[1])
        a0, a1 = (-150.0, 60.0) if sens > 0 else (150.0, -60.0)
        p0 = (cx + r_px * math.cos(math.radians(a0)), cy - r_px * math.sin(math.radians(a0)))
        p1 = (cx + r_px * math.cos(math.radians(a1)), cy - r_px * math.sin(math.radians(a1)))
        sweep = 0 if sens > 0 else 1
        self.out.append(f'<path d="M{p0[0]:.1f},{p0[1]:.1f} A{r_px},{r_px} 0 1 {sweep} {p1[0]:.1f},{p1[1]:.1f}" fill="none" stroke="{stroke}" stroke-width="2"/>')
        # pointe tangente en p1
        t = math.radians(a1) + (math.pi / 2 if sens > 0 else -math.pi / 2)
        tx, ty = math.cos(t), -math.sin(t)
        self.out.append(f'<polygon points="{p1[0] + tx * 7:.1f},{p1[1] + ty * 7:.1f} {p1[0] - ty * 4.5:.1f},{p1[1] + tx * 4.5:.1f} '
                        f'{p1[0] + ty * 4.5:.1f},{p1[1] - tx * 4.5:.1f}" fill="{stroke}"/>')
        if label:
            self.out.append(f'<text x="{cx:.1f}" y="{cy + r_px + 16:.1f}" font-size="{taille}" fill="{stroke}" text-anchor="middle" font-weight="600">{label}</text>')

    def symbole_normal(self, c, label="", sortant=True, stroke=EFFORT, taille=11.5):
        """⊙ (vers le lecteur) ou ⊗ (loin du lecteur) : effort normal au dessin."""
        cx, cy = self.X(c[0]), self.Y(c[1])
        self.out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#fff" stroke="{stroke}" stroke-width="2"/>')
        if sortant:
            self.out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.2" fill="{stroke}"/>')
        else:
            self.out.append(f'<path d="M{cx - 4:.1f},{cy - 4:.1f}L{cx + 4:.1f},{cy + 4:.1f}M{cx - 4:.1f},{cy + 4:.1f}L{cx + 4:.1f},{cy - 4:.1f}" stroke="{stroke}" stroke-width="2"/>')
        if label:
            self.out.append(f'<text x="{cx + 11:.1f}" y="{cy + 4:.1f}" font-size="{taille}" fill="{stroke}" font-weight="600">{label}</text>')

    def cote(self, p, q, label, decal=14.0, taille=10):
        """Cote entre p et q (mm), décalée de decal px perpendiculairement."""
        x1, y1, x2, y2 = self.X(p[0]), self.Y(p[1]), self.X(q[0]), self.Y(q[1])
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        nx, ny = -dy / L * decal, dx / L * decal
        self.out.append(f'<line x1="{x1 + nx:.1f}" y1="{y1 + ny:.1f}" x2="{x2 + nx:.1f}" y2="{y2 + ny:.1f}" stroke="{EFFORT}" stroke-width="0.8"/>')
        for (x, y) in ((x1, y1), (x2, y2)):
            self.out.append(f'<line x1="{x + nx * 0.6:.1f}" y1="{y + ny * 0.6:.1f}" x2="{x + nx * 1.4:.1f}" y2="{y + ny * 1.4:.1f}" stroke="{EFFORT}" stroke-width="0.8"/>')
        rot = math.degrees(math.atan2(dy, dx))
        if rot > 90 or rot < -90:
            rot += 180
        mx, my = (x1 + x2) / 2 + nx * 1.6, (y1 + y2) / 2 + ny * 1.6
        self.out.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="{taille}" fill="{EFFORT}" text-anchor="middle" transform="rotate({rot:.1f} {mx:.1f} {my:.1f})">{label}</text>')

    def cordon_taux(self, c, taux3: Sequence[float], epaisseur_px: float = 7.0, gouvernant: Optional[float] = None, nom_grad: str = "g"):
        """Le cordon de p1 à p2 coloré par ses taux (début, milieu, fin) ; le
        point gouvernant (s) marqué ●."""
        x1, y1, x2, y2 = self.X(c["p1"][0]), self.Y(c["p1"][1]), self.X(c["p2"][0]), self.Y(c["p2"][1])
        gid = f"{nom_grad}_{abs(hash((round(x1), round(y1), round(x2), round(y2)))) % 100000}"
        stops = "".join(f'<stop offset="{o}" stop-color="{couleur_taux(t)}"/>' for o, t in zip((0, 0.5, 1), taux3))
        self.out.append(f'<defs><linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">{stops}</linearGradient></defs>')
        self.out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="url(#{gid})" stroke-width="{epaisseur_px}" stroke-linecap="round"/>')
        if gouvernant is not None:
            gx, gy = x1 + (x2 - x1) * gouvernant, y1 + (y2 - y1) * gouvernant
            self.out.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5" fill="#fff" stroke="{NOK}" stroke-width="2"/>')

    def svg(self, fond: Optional[str] = None) -> str:
        bg = f'<rect width="{self.W}" height="{self.H}" fill="{fond}" rx="8"/>' if fond else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" viewBox="0 0 {self.W} {self.H}" {POLICE} font-size="11">'
                + bg + "".join(self.out) + "</svg>")


# ====================================================================== vues
def _taux3(res: Dict, nom: str) -> List[float]:
    pts = sorted([p for p in res["points"] if p["cordon"] == nom], key=lambda p: p["s"])
    return [p["taux"] for p in pts] if len(pts) == 3 else [0.0, 0.0, 0.0]


def _gouv_s(res: Dict, nom: str) -> Optional[float]:
    g = res.get("gouvernant")
    return g["s"] if g and g.get("cordon") == nom else None


def _cordons_taux(cv: Canevas, res: Dict, ep: float = 7.0, annoter: bool = True):
    for c in res.get("cordons", []):
        cv.cordon_taux(c, _taux3(res, c["nom"]), ep, _gouv_s(res, c["nom"]))
    g = res.get("gouvernant")
    if g and annoter and g.get("cordon") != "pièce":
        px, py = g["point"][0], g["point"][1]
        cv.texte(px, py, f"● {g['taux'] * 100:.0f} %", 10.5, NOK, "start", True, dx=9, dy=-6)


def vue_plan_recouvrement(res: Dict, W: int = 560, H: int = 340) -> str:
    """C1 : vue en plan, plat A dessous (gris foncé), plat B dessus, cordons sur
    les bords de B, N le long, V en travers."""
    pc = res["pieces"]
    xs = [x for x, _ in pc["A"]] + [x for x, _ in pc["B"]]
    ys = [y for _, y in pc["A"]] + [y for _, y in pc["B"]]
    cv = Canevas(W, H, min(xs), max(xs) + 40, min(ys) - 20, max(ys) + 20)
    cv.polygone(pc["A"], PIECE_A)
    cv.polygone(pc["B"], PIECE_B, opacity=0.92)
    L = max(x for x, _ in pc["B"]) / 1.6
    b = max(y for _, y in pc["B"]) * 2
    cv.texte(L * 1.15, b * 0.62, "plat B (dessus)", 10, MUT, "middle")
    cv.texte(-L * 0.3, -b * 0.62, "plat A (dessous)", 10, MUT, "middle")
    _cordons_taux(cv, res)
    e = res["efforts"]
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((L * 1.25, 0.0), (L * 1.25 + s * L * 0.3, 0.0), f"N = {_fmt(abs(e.Vx) / 1e3)} kN")
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((L * 1.1, -s * b * 0.1), (L * 1.1, s * b * 0.45), f"V = {_fmt(abs(e.Vy) / 1e3)} kN")
    cv.cote((0, -b / 2), (L, -b / 2), f"L = {_fmt(L, 0)}", -16)
    cv.cote((L * 1.6, -b / 2), (L * 1.6, b / 2), f"b = {_fmt(b, 0)}", 12)
    return cv.svg(FOND)


def vue_elevation_T(res: Dict, W: int = 560, H: int = 340) -> str:
    """C2 : à gauche la section (A horizontal, B debout, deux triangles de
    cordon, N, V⊥, M), à droite la vue en plan (longueur l, V∥, taux)."""
    pc = res["pieces"]
    tA, tB, l, hB = pc["tA"], pc["tB"], pc["l"], pc["hB"]
    cv = Canevas(W, H, -hB, l + 60, -tA - 20, hB + 30)
    # section à gauche : largeur hB, A horizontal sous z = 0, B debout
    cv.polygone([(-hB, -tA), (-hB * 0.05, -tA), (-hB * 0.05, 0), (-hB, 0)], PIECE_A)
    xc = -hB * 0.525
    cv.polygone([(xc - tB / 2, 0), (xc + tB / 2, 0), (xc + tB / 2, hB), (xc - tB / 2, hB)], PIECE_B)
    a = res["cordons"][0]["a"] if res.get("cordons") else 5.0
    z = a * 1.414
    cv.polygone([(xc + tB / 2, 0), (xc + tB / 2 + z, 0), (xc + tB / 2, z)], CORDON, ENCRE, 0.8)
    if len(res.get("cordons", [])) > 1:
        cv.polygone([(xc - tB / 2, 0), (xc - tB / 2 - z, 0), (xc - tB / 2, z)], CORDON, ENCRE, 0.8)
    e = res["efforts"]
    if abs(e.N) > 0:
        s = 1 if e.N > 0 else -1
        cv.fleche((xc, hB * 0.55), (xc, hB * 0.55 + s * hB * 0.35), f"N = {_fmt(abs(e.N) / 1e3)} kN")
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((xc - s * hB * 0.1, hB * 0.8), (xc + s * hB * 0.3, hB * 0.8), f"V⊥ = {_fmt(abs(e.Vy) / 1e3)} kN")
    if abs(e.Mx) > 0:
        cv.moment((xc, hB * 0.45), 22, f"M = {_fmt(abs(e.Mx) / 1e6)} kN·m", +1 if e.Mx > 0 else -1)
    cv.texte(xc, -tA - 12, "section", 10, MUT)
    # plan à droite : la jonction de longueur l, cordons colorés, V∥
    y0 = hB * 0.35
    cv.polygone([(0, y0 - hB * 0.3), (l, y0 - hB * 0.3), (l, y0 + hB * 0.3), (0, y0 + hB * 0.3)], PIECE_A)
    cv.polygone([(0, y0 - tB / 2), (l, y0 - tB / 2), (l, y0 + tB / 2), (0, y0 + tB / 2)], PIECE_B)
    for c in res.get("cordons", []):
        cc = dict(c, p1=[c["p1"][0], c["p1"][1] + y0], p2=[c["p2"][0], c["p2"][1] + y0])
        cv.cordon_taux(cc, _taux3(res, c["nom"]), 7.0, _gouv_s(res, c["nom"]))
    g = res.get("gouvernant")
    if g:
        cv.texte(g["point"][0], g["point"][1] + y0, f"● {g['taux'] * 100:.0f} %", 10.5, NOK, "start", True, dx=9, dy=-8)
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((l / 2 - s * l * 0.15, y0 + hB * 0.42), (l / 2 + s * l * 0.15, y0 + hB * 0.42), f"V∥ = {_fmt(abs(e.Vx) / 1e3)} kN")
    cv.cote((0, y0 - hB * 0.3), (l, y0 - hB * 0.3), f"l = {_fmt(l, 0)}", -14)
    cv.texte(l / 2, y0 - hB * 0.3 - 26, "vue en plan", 10, MUT)
    return cv.svg(FOND)


def vue_elevation_console(res: Dict, W: int = 560, H: int = 340) -> str:
    """C3 : poteau à gauche (face = plan de joint), console vers la droite, V à
    l'excentricité e, M = V·e, cordons verticaux colorés sur la face."""
    pc = res["pieces"]
    h, t, ex, Lc = pc["h"], pc["t"], pc["e"], pc["longueur"]
    cv = Canevas(W, H, -90, Lc + 40, -h * 0.8, h * 0.9)
    cv.polygone([(-80, -h * 0.75), (0, -h * 0.75), (0, h * 0.85), (-80, h * 0.85)], PIECE_A)
    cv.texte(-40, h * 0.78, "poteau", 10, MUT)
    cv.polygone([(0, h / 2), (Lc, h / 2), (Lc, h / 2 - h * 0.45), (0, -h / 2)], PIECE_B)
    cv.texte(Lc * 0.5, h * 0.22, f"plat {_fmt(t, 0)} mm", 10, MUT)
    # cordons : sur la face (x = 0 du dessin), de −h/2 à h/2 ; les deux cordons sont confondus en élévation
    for c in res.get("cordons", []):
        cc = dict(c, p1=[0.0, c["p1"][1]], p2=[0.0, c["p2"][1]])
        cv.cordon_taux(cc, _taux3(res, c["nom"]), 8.0, _gouv_s(res, c["nom"]))
    g = res.get("gouvernant")
    if g:
        cv.texte(0, g["point"][1], f"● {g['taux'] * 100:.0f} %", 10.5, NOK, "end", True, dx=-10, dy=4)
    e = res["efforts"]
    V = -e.Vy
    if abs(V) > 0:
        s = -1 if V > 0 else 1
        cv.fleche((ex, h / 2 - s * h * 0.5), (ex, h / 2 - s * h * 0.05 + (0 if s < 0 else 0)), f"V = {_fmt(abs(V) / 1e3)} kN", pos_label="debut")
        cv.moment((Lc * 0.22, 0.0), 24, f"M = V·e = {_fmt(abs(V) * ex / 1e6)} kN·m", -1 if V > 0 else 1)
    if abs(e.N) > 0:
        s = 1 if e.N > 0 else -1
        cv.fleche((Lc * 0.75, 0.0), (Lc * 0.75 + s * Lc * 0.2, 0.0), f"N = {_fmt(abs(e.N) / 1e3)} kN")
    cv.cote((0, h / 2 + 10), (ex, h / 2 + 10), f"e = {_fmt(ex, 0)}", 16)
    cv.cote((Lc + 14, -h / 2), (Lc + 14, h / 2), f"h = {_fmt(h, 0)}", -16) if False else cv.cote((-6, -h / 2), (-6, h / 2), f"h = {_fmt(h, 0)}", 18)
    return cv.svg(FOND)


def vue_plan_gousset(res: Dict, W: int = 560, H: int = 340) -> str:
    """C4 : le plat vu en plan, cordons sur les côtés choisis, V, H, N au point d'application."""
    pc = res["pieces"]
    B = pc["B"]
    xs, ys = [x for x, _ in B], [y for _, y in B]
    cv = Canevas(W, H, min(xs) - 30, max(xs) + 60, min(ys) - 40, max(ys) + 40)
    cv.polygone([(min(xs) - 25, min(ys) - 35), (max(xs) + 55, min(ys) - 35), (max(xs) + 55, max(ys) + 35), (min(xs) - 25, max(ys) + 35)], PIECE_A, opacity=0.5)
    cv.polygone(B, PIECE_B)
    cv.texte((min(xs) + max(xs)) / 2, max(ys) - 18, f"plat {_fmt(pc['t'], 0)} mm, posé à plat", 10, MUT)
    _cordons_taux(cv, res)
    e = res["efforts"]
    xV, yV = pc["xV"], pc["yV"]
    cv.out.append(f'<circle cx="{cv.X(xV):.1f}" cy="{cv.Y(yV):.1f}" r="3" fill="{EFFORT}"/>')
    V = -e.Vy
    if abs(V) > 0:
        s = -1 if V > 0 else 1
        cv.fleche((xV, yV - s * 5), (xV, yV + s * 60), f"V = {_fmt(abs(V) / 1e3)} kN")
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((xV + s * 5, yV), (xV + s * 60, yV), f"H = {_fmt(abs(e.Vx) / 1e3)} kN")
    if abs(e.N) > 0:
        cv.symbole_normal((xV, yV), f"N = {_fmt(abs(e.N) / 1e3)} kN", e.N > 0)
    pr = res.get("proprietes") or {}
    if "xG" in pr:
        cv.out.append(f'<circle cx="{cv.X(pr["xG"]):.1f}" cy="{cv.Y(pr["yG"]):.1f}" r="3" fill="none" stroke="{ENCRE}" stroke-width="1"/>')
        cv.texte(pr["xG"], pr["yG"], "G", 9.5, ENCRE, "start", dx=5, dy=-4)
        cv.cote((pr["xG"], min(ys) - 12), (xV, min(ys) - 12), f"e = {_fmt(xV - pr['xG'], 0)}", -14)
    return cv.svg(FOND)


def vue_face_profil(res: Dict, W: int = 560, H: int = 340) -> str:
    """C5 / C6 : la platine vue de face, le profilé (ou le tube) en section,
    les cordons autour, V_z, V_y, N (⊙/⊗), M_y, M_z (double flèche), T."""
    pc = res["pieces"]
    pl = pc["platine"]
    xs, ys = [x for x, _ in pl], [y for _, y in pl]
    cv = Canevas(W, H, min(xs) - 70, max(xs) + 70, min(ys) - 30, max(ys) + 30)
    cv.polygone(pl, PIECE_A)
    rec = pc["profil"]
    if "contour" in pc:
        cv.polygone(pc["contour"], PIECE_B)
        tube_t = rec.get("t", 0.0)
        if tube_t:
            from acier.profiles import familles as F
            try:
                rings = F.contours(rec["famille"], rec["dims"])
                for ring in rings[1:]:
                    cv.polygone([(float(x), float(y)) for x, y in ring], PIECE_A, ENCRE, 0.8)
            except Exception:  # noqa: BLE001 — le trou du tube n'est qu'un ornement
                pass
    else:
        from acier.profiles import familles as F
        try:
            ring = F.contours(rec["famille"], rec["dims"])[0]
            cv.polygone([(float(x), float(y)) for x, y in ring], PIECE_B)
        except Exception:  # noqa: BLE001
            h, b = rec["h"], rec["b"]
            cv.polygone([(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)], PIECE_B)
    cv.texte(0, max(ys) + 14, rec["nom"], 10.5, ENCRE, gras=True)
    for c in res.get("cordons", []):
        cv.cordon_taux(c, _taux3(res, c["nom"]), 5.5, _gouv_s(res, c["nom"]))
    g = res.get("gouvernant")
    if g:
        cv.texte(g["point"][0], g["point"][1], f"● {g['taux'] * 100:.0f} %", 10.5, NOK, "start", True, dx=9, dy=-6)
    e = res["efforts"]
    R = max(max(xs), max(ys))
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((R + 35, -s * 40), (R + 35, s * 40), f"V_z = {_fmt(abs(e.Vy) / 1e3)} kN")
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((-s * 40, -R - 18), (s * 40, -R - 18), f"V_y = {_fmt(abs(e.Vx) / 1e3)} kN")
    if abs(e.N) > 0:
        cv.symbole_normal((0, 0), f"N = {_fmt(abs(e.N) / 1e3)} kN", e.N > 0)
    if abs(e.Mx) > 0:
        s = 1 if e.Mx > 0 else -1
        cv.fleche((-R - 40, 0), (-R - 40 + s * 0.001, 0), "")
        cv.out.append(f'<line x1="{cv.X(-R - 46):.1f}" y1="{cv.Y(0):.1f}" x2="{cv.X(-R - 14):.1f}" y2="{cv.Y(0):.1f}" stroke="{EFFORT}" stroke-width="2.2"/>')
        tip = cv.X(-R - 14) if s > 0 else cv.X(-R - 46)
        d = 1 if s > 0 else -1
        cv.out.append(f'<polygon points="{tip:.1f},{cv.Y(0):.1f} {tip - d * 9:.1f},{cv.Y(0) - 4.5:.1f} {tip - d * 9:.1f},{cv.Y(0) + 4.5:.1f}" fill="{EFFORT}"/>')
        cv.out.append(f'<polygon points="{tip - d * 7:.1f},{cv.Y(0):.1f} {tip - d * 16:.1f},{cv.Y(0) - 4.5:.1f} {tip - d * 16:.1f},{cv.Y(0) + 4.5:.1f}" fill="{EFFORT}"/>')
        cv.texte(-R - 30, 0, f"M_y = {_fmt(abs(e.Mx) / 1e6)} kN·m", 11, EFFORT, "middle", True, dy=-10)
    if abs(e.My) > 0:
        cv.texte(0, R + 30, f"M_z = {_fmt(abs(e.My) / 1e6)} kN·m (double flèche verticale)", 10.5, EFFORT, "middle", True)
    if abs(e.Mz) > 0:
        cv.moment((0, 0), min(60, R * 0.5 * cv.ech), f"T = {_fmt(abs(e.Mz) / 1e6)} kN·m", 1 if e.Mz > 0 else -1)
    return cv.svg(FOND)


def vue_plan_bout_a_bout(res: Dict, W: int = 560, H: int = 340) -> str:
    """C7 : les deux plats dans leur plan, la soudure entre eux, N de part et d'autre, V le long, M."""
    pc = res["pieces"]
    b = pc["b"]
    cv = Canevas(W, H, -b * 1.1, b * 1.1, -b * 0.75, b * 0.75)
    cv.polygone(pc["A"], PIECE_A)
    cv.polygone(pc["B"], PIECE_B)
    cv.texte(-b / 2, -b * 0.6, "plat A", 10, MUT)
    cv.texte(b / 2, -b * 0.6, "plat B", 10, MUT)
    if pc["pleine"]:
        cv.out.append(f'<line x1="{cv.X(0):.1f}" y1="{cv.Y(-b / 2):.1f}" x2="{cv.X(0):.1f}" y2="{cv.Y(b / 2):.1f}" stroke="{CORDON}" stroke-width="7"/>')
        cv.texte(0, b / 2 + 12, "pleine pénétration : la pièce gouverne", 10.5, ENCRE)
    else:
        for c in res.get("cordons", []):
            cv.cordon_taux(c, _taux3(res, c["nom"]), 7.0, _gouv_s(res, c["nom"]))
        g = res.get("gouvernant")
        if g:
            cv.texte(0, g["point"][1], f"● {g['taux'] * 100:.0f} %", 10.5, NOK, "start", True, dx=9, dy=-6)
    e = res["efforts"]
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((b * 0.55, 0), (b * 0.55 + s * b * 0.3, 0), f"N = {_fmt(abs(e.Vx) / 1e3)} kN")
        cv.fleche((-b * 0.55, 0), (-b * 0.55 - s * b * 0.3, 0), "")
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((b * 0.35, -s * b * 0.1), (b * 0.35, s * b * 0.3), f"V = {_fmt(abs(e.Vy) / 1e3)} kN")
    if abs(e.Mz) > 0:
        cv.moment((b * 0.5, b * 0.3), 20, f"M = {_fmt(abs(e.Mz) / 1e6)} kN·m", 1 if e.Mz > 0 else -1)
    cv.cote((b * 0.95, -b / 2), (b * 0.95, b / 2), f"b = {_fmt(b, 0)}", -14)
    return cv.svg(FOND)


def vue_plan_libre(res: Dict, W: int = 560, H: int = 340) -> str:
    """C8 : les cordons tels que saisis, le centre G, le point de référence et
    les composantes d'effort."""
    cs = res.get("cordons", [])
    e = res["efforts"]
    xs = [c["p1"][0] for c in cs] + [c["p2"][0] for c in cs] + [e.reference[0]]
    ys = [c["p1"][1] for c in cs] + [c["p2"][1] for c in cs] + [e.reference[1]]
    if not xs:
        xs, ys = [0.0], [0.0]
    m = max(30.0, 0.25 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    cv = Canevas(W, H, min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m)
    for c in cs:
        cv.cordon_taux(c, _taux3(res, c["nom"]), 7.0, _gouv_s(res, c["nom"]))
        cv.texte(c["p1"][0], c["p1"][1], c["nom"], 9.5, MUT, "start", dx=6, dy=12)
    g = res.get("gouvernant")
    if g:
        cv.texte(g["point"][0], g["point"][1], f"● {g['taux'] * 100:.0f} %", 10.5, NOK, "start", True, dx=9, dy=-6)
    pr = res.get("proprietes") or {}
    if "xG" in pr:
        cv.out.append(f'<circle cx="{cv.X(pr["xG"]):.1f}" cy="{cv.Y(pr["yG"]):.1f}" r="3" fill="none" stroke="{ENCRE}" stroke-width="1"/>')
        cv.texte(pr["xG"], pr["yG"], "G", 9.5, ENCRE, "start", dx=5, dy=-4)
    rx, ry = e.reference[0], e.reference[1]
    cv.out.append(f'<circle cx="{cv.X(rx):.1f}" cy="{cv.Y(ry):.1f}" r="3" fill="{EFFORT}"/>')
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((rx + s * 4, ry), (rx + s * m * 0.9, ry), f"V_x = {_fmt(abs(e.Vx) / 1e3)} kN")
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((rx, ry + s * 4), (rx, ry + s * m * 0.9), f"V_y = {_fmt(abs(e.Vy) / 1e3)} kN")
    if abs(e.N) > 0:
        cv.symbole_normal((rx, ry), f"N = {_fmt(abs(e.N) / 1e3)} kN", e.N > 0)
    libs = []
    if abs(e.Mx) > 0:
        libs.append(f"M_x = {_fmt(abs(e.Mx) / 1e6)} kN·m")
    if abs(e.My) > 0:
        libs.append(f"M_y = {_fmt(abs(e.My) / 1e6)} kN·m")
    if abs(e.Mz) > 0:
        cv.moment((rx, ry), 26, f"M_z = {_fmt(abs(e.Mz) / 1e6)} kN·m", 1 if e.Mz > 0 else -1)
    if libs:
        cv.texte(min(xs) - m * 0.9, max(ys) + m * 0.8, " · ".join(libs), 10.5, EFFORT, "start", True)
    return cv.svg(FOND)


VUES = {"C1": vue_plan_recouvrement, "C2": vue_elevation_T, "C3": vue_elevation_console, "C4": vue_plan_gousset,
        "C5": vue_face_profil, "C6": vue_face_profil, "C7": vue_plan_bout_a_bout, "C8": vue_plan_libre}


def vue_principale(res: Dict, W: int = 560, H: int = 340) -> str:
    if res.get("statut") == "refus" or "efforts" not in res:
        cv = Canevas(W, H, 0, 100, 0, 60)
        cv.texte(50, 30, "Calcul impossible : voir les messages", 12, NOK)
        return cv.svg(FOND)
    return VUES[res["config"]](res, W, H)


# ====================================================================== encart « la gorge »
def encart_gorge(res: Dict, W: int = 300, H: int = 250) -> str:
    """La section du cordon gouvernant : les deux faces, le triangle de
    gorge a, le plan de gorge (pointillé), σ⊥ normal à ce plan, τ⊥ dans ce
    plan, τ∥ vers le lecteur (⊙) ou l'inverse (⊗), avec les valeurs."""
    g = res.get("gouvernant")
    cv = Canevas(W, H, 0, 100, 0, 100, marge=10)
    cv.texte(50, 94, "LA GORGE · point gouvernant", 9.5, MUT)
    if not g or g.get("cordon") == "pièce":
        cv.texte(50, 50, "pas de cordon gouvernant" if not g else "pleine pénétration : la pièce gouverne", 10.5, ENCRE)
        return cv.svg(FOND)
    a = g["a"]
    bout = any(c.get("bout_a_bout") for c in []) or False
    # géométrie du dessin (unités arbitraires) : pièce A horizontale en bas, pièce B debout à gauche
    cv.polygone([(10, 14), (88, 14), (88, 24), (10, 24)], PIECE_A)        # A
    cv.polygone([(10, 24), (24, 24), (24, 84), (10, 84)], PIECE_B)        # B
    z = 24.0                                                               # côté du triangle (dessin)
    cv.polygone([(24, 24), (24 + z, 24), (24, 24 + z)], CORDON, ENCRE, 0.8, opacity=0.9)
    # plan de gorge : de la racine (24,24) vers le milieu de l'hypoténuse
    mx, my = 24 + z / 2, 24 + z / 2
    cv.ligne((24, 24), (mx, my), ENCRE, 1.0, "3 2")
    cv.texte(mx, my, f"a = {_fmt(a, 1)}", 9.5, ENCRE, "start", dx=6, dy=10)
    cv.texte(60, 19, "pièce A (support)", 8.5, MUT)
    cv.texte(17, 60, "B", 9.5, MUT)
    # composantes : σ⊥ normal au plan de gorge (direction (1,−1)/√2 depuis le milieu), τ⊥ le long du plan ((1,1)/√2)
    sp, tp, tpar = g["sigma_perp"], g["tau_perp"], g["tau_par"]
    L = 26.0
    s1 = 1 if sp >= 0 else -1
    cv.fleche((mx, my), (mx + s1 * L * 0.707, my - s1 * L * 0.707), f"σ⊥ = {_fmt(sp)}", EFFORT, 2.0, 10.5)
    s2 = 1 if tp >= 0 else -1
    cv.fleche((mx, my), (mx + s2 * L * 0.707, my + s2 * L * 0.707), f"τ⊥ = {_fmt(tp)}", EFFORT, 2.0, 10.5)
    cv.symbole_normal((mx, my), "", tpar >= 0)
    cv.texte(50, 6, f"τ∥ = {_fmt(tpar)} N/mm² (le long du cordon, {'⊙ vers soi' if tpar >= 0 else '⊗'})", 9.5, EFFORT)
    cv.texte(50, 0, f"σ_c = √(σ⊥² + 3(τ⊥² + τ∥²)) = {_fmt(g['sigma_c'])} N/mm²", 9.5, ENCRE, dy=-2)
    return cv.svg(FOND)


def legende() -> str:
    return ("■ pièces · ■ cordons colorés par leur taux (vert ≤ 70 %, ocre, rouge ≥ 100 %) · ● point gouvernant · "
            "flèches : efforts dans le sens saisi (⊙ vers soi, ⊗ vers le fond) · encart : composantes sur la gorge, EN 1993-1-8 fig. 4.5")

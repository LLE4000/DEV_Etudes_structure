# -*- coding: utf-8 -*-
"""Petit kit SVG de dessinateur : repère en mm, traits fins, hachures, cotes, flèches d'effort, textes à halo.

Repère « monde » : u vers la droite, y vers le HAUT (mm) ; le kit convertit en pixels (y vers le bas).
Aucune ressource externe : polices système (Inter, DejaVu Sans), hachures en motifs SVG internes.
"""
import math
from xml.sax.saxutils import escape

POLICE = "Inter, 'DejaVu Sans', Arial, sans-serif"
ENCRE = "#1d1d1b"
ENCRE2 = "#55534e"
MUET = "#8a8780"
COTE = "#1f4e8c"          # bleu de cote, comme le schéma de la note
SECTION = "#e8590c"       # orange des sections, comme le schéma de la note
CHARGE = "#d0201a"        # rouge de la charge
COMP = "#1a7f4b"          # vert de la résultante de compression (pastilles vertes de la note)
ACIER = "#eef0f3"
BETON = "#f3f1ec"


def fr(x, nd=1):
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s


class Dessin:
    def __init__(self, largeur, hauteur, fond="#ffffff"):
        self.W, self.H = largeur, hauteur
        self.el = []
        self.defs = []
        self.fond = fond
        self._motifs()

    # ---------------------------------------------------------------- repère
    def repere(self, ox, oy, ech):
        """Origine du monde (0, 0) au pixel (ox, oy), ech px par mm, y vers le haut."""
        self.ox, self.oy, self.ech = ox, oy, ech

    def P(self, u, y):
        return (self.ox + u * self.ech, self.oy - y * self.ech)

    # ---------------------------------------------------------------- motifs
    def _motifs(self):
        self.defs.append(
            '<pattern id="hachAcier" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
            '<line x1="0" y1="0" x2="0" y2="6" stroke="#6b6b6b" stroke-width="0.7"/></pattern>')
        self.defs.append(
            '<pattern id="hachBeton" patternUnits="userSpaceOnUse" width="34" height="34">'
            '<rect width="34" height="34" fill="#f4f2ed"/>'
            '<path d="M4 9 l4 -6 l4 6 z" fill="none" stroke="#b9b4a8" stroke-width="0.8"/>'
            '<path d="M21 27 l3 -5 l3 5 z" fill="none" stroke="#b9b4a8" stroke-width="0.8"/>'
            '<circle cx="25" cy="8" r="1.1" fill="#b9b4a8"/><circle cx="9" cy="24" r="0.9" fill="#b9b4a8"/>'
            '<circle cx="16" cy="16" r="0.7" fill="#c8c3b8"/></pattern>')
        self.defs.append(
            '<pattern id="hachMoment" patternUnits="userSpaceOnUse" width="5" height="5">'
            '<line x1="0" y1="2.5" x2="5" y2="2.5" stroke="#e8590c" stroke-width="0.6" stroke-opacity="0.75"/></pattern>')
        self.defs.append(
            '<pattern id="hachMomentV" patternUnits="userSpaceOnUse" width="5" height="5">'
            '<line x1="2.5" y1="0" x2="2.5" y2="5" stroke="#e8590c" stroke-width="0.6" stroke-opacity="0.75"/></pattern>')
        self.defs.append(
            '<pattern id="hachVert" patternUnits="userSpaceOnUse" width="5" height="5">'
            '<line x1="0" y1="2.5" x2="5" y2="2.5" stroke="#1a7f4b" stroke-width="0.6" stroke-opacity="0.7"/></pattern>')
        self.defs.append(
            '<pattern id="hachVertV" patternUnits="userSpaceOnUse" width="5" height="5">'
            '<line x1="2.5" y1="0" x2="2.5" y2="5" stroke="#1a7f4b" stroke-width="0.6" stroke-opacity="0.7"/></pattern>')
        self.defs.append(
            '<pattern id="hachGris" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(-45)">'
            '<line x1="0" y1="0" x2="0" y2="7" stroke="#a9a59c" stroke-width="0.8"/></pattern>')
        for nom, coul in (("fl", ENCRE), ("flC", COTE), ("flR", CHARGE), ("flV", COMP), ("flO", SECTION), ("flG", ENCRE2)):
            self.defs.append(
                f'<marker id="{nom}" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="9" markerHeight="9" '
                f'orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0 1.2 L10 5 L0 8.8 z" fill="{coul}"/></marker>')

    # ---------------------------------------------------------------- primitives (pixels)
    def add(self, s):
        self.el.append(s)

    def ligne_px(self, x1, y1, x2, y2, coul=ENCRE, ep=1.0, tirets=None, fleche=None, debut=None, op=1.0):
        d = f' stroke-dasharray="{tirets}"' if tirets else ""
        m = (f' marker-end="url(#{fleche})"' if fleche else "") + (f' marker-start="url(#{debut})"' if debut else "")
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{coul}" stroke-width="{ep}"'
                 f' stroke-opacity="{op}"{d}{m}/>')

    def texte_px(self, x, y, txt, taille=14, coul=ENCRE, ancre="start", gras=False, halo=True, rot=0, ital=False,
                 base="middle"):
        w = ' font-weight="600"' if gras else ""
        it = ' font-style="italic"' if ital else ""
        tr = f' transform="rotate({rot} {x:.2f} {y:.2f})"' if rot else ""
        h = (' paint-order="stroke" stroke="#ffffff" stroke-width="4" stroke-linejoin="round"') if halo else ""
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{POLICE}" font-size="{taille}" fill="{coul}" '
                 f'text-anchor="{ancre}" dominant-baseline="{base}"{w}{it}{tr}{h}>{txt}</text>')

    def rect_px(self, x, y, w, h, fill="none", stroke=ENCRE, ep=1.0, rx=0, op=1.0):
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" '
                 f'fill-opacity="{op}" stroke="{stroke}" stroke-width="{ep}"/>')

    # ---------------------------------------------------------------- primitives (monde)
    def ligne(self, a, b, **kw):
        (x1, y1), (x2, y2) = self.P(*a), self.P(*b)
        self.ligne_px(x1, y1, x2, y2, **kw)

    def poly(self, pts, fill="none", stroke=ENCRE, ep=1.0, ferme=True, op=1.0, tirets=None):
        p = " ".join(f"{x:.2f},{y:.2f}" for x, y in (self.P(*q) for q in pts))
        tag = "polygon" if ferme else "polyline"
        d = f' stroke-dasharray="{tirets}"' if tirets else ""
        self.add(f'<{tag} points="{p}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" stroke-width="{ep}" '
                 f'stroke-linejoin="round"{d}/>')

    def texte(self, u, y, txt, **kw):
        x, yy = self.P(u, y)
        self.texte_px(x, yy, txt, **kw)

    def rond(self, u, y, r_px, fill="#fff", stroke=ENCRE, ep=1.0):
        x, yy = self.P(u, y)
        self.add(f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="{r_px}" fill="{fill}" stroke="{stroke}" stroke-width="{ep}"/>')

    def pastille(self, u, y, num, coul=SECTION, r_px=11, taille=14):
        x, yy = self.P(u, y)
        self.add(f'<circle cx="{x:.2f}" cy="{yy:.2f}" r="{r_px}" fill="{coul}" stroke="#fff" stroke-width="1.5"/>')
        self.texte_px(x, yy + 0.5, str(num), taille=taille, coul="#fff", ancre="middle", gras=True, halo=False)

    def fleche(self, a, b, coul=CHARGE, ep=2.2, marq="flR", tirets=None):
        (x1, y1), (x2, y2) = self.P(*a), self.P(*b)
        self.ligne_px(x1, y1, x2, y2, coul=coul, ep=ep, fleche=marq, tirets=tirets)

    # ---------------------------------------------------------------- cotes
    def cote(self, a, b, dec, txt, coul=COTE, taille=14, cote_txt=None, ext=True, pos=0.5, fl=True, ecart_txt=10,
             ep=0.9):
        """Cote entre les points monde a et b, décalée de `dec` pixels perpendiculairement (à gauche de a→b)."""
        (x1, y1), (x2, y2) = self.P(*a), self.P(*b)
        dx, dy = x2 - x1, y2 - y1
        Lp = math.hypot(dx, dy) or 1.0
        nx, ny = dy / Lp, -dx / Lp           # normale (vers la gauche du sens a→b, à l'écran)
        X1, Y1 = x1 + nx * dec, y1 + ny * dec
        X2, Y2 = x2 + nx * dec, y2 + ny * dec
        if ext:
            s = 1 if dec >= 0 else -1
            self.ligne_px(x1 + nx * 3 * s, y1 + ny * 3 * s, X1 + nx * 5 * s, Y1 + ny * 5 * s, coul=coul, ep=0.6, op=0.8)
            self.ligne_px(x2 + nx * 3 * s, y2 + ny * 3 * s, X2 + nx * 5 * s, Y2 + ny * 5 * s, coul=coul, ep=0.6, op=0.8)
        if fl:
            self.ligne_px(X1, Y1, X2, Y2, coul=coul, ep=ep, fleche="flC", debut="flC")
        else:
            self.ligne_px(X1, Y1, X2, Y2, coul=coul, ep=ep)
        xm, ym = X1 + (X2 - X1) * pos, Y1 + (Y2 - Y1) * pos
        ang = math.degrees(math.atan2(Y2 - Y1, X2 - X1))
        if ang >= 89.9:
            ang -= 180
        if ang < -90.1:
            ang += 180
        s = 1 if dec >= 0 else -1
        tx, ty = xm + nx * ecart_txt * s, ym + ny * ecart_txt * s
        self.texte_px(tx, ty, txt, taille=taille, coul=coul, ancre="middle", rot=ang)

    # ---------------------------------------------------------------- sortie
    def svg(self, titre=None):
        t = f"<title>{escape(titre)}</title>" if titre else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" viewBox="0 0 {self.W} {self.H}">'
                f'{t}<defs>{"".join(self.defs)}</defs><rect width="100%" height="100%" fill="{self.fond}"/>'
                + "\n".join(self.el) + "</svg>")

    def ecrire(self, chemin, titre=None):
        with open(chemin, "w", encoding="utf-8") as fh:
            fh.write(self.svg(titre))


# ---------------------------------------------------------------- géométrie d'une cornière laminée
def contour_L(larg, haut, e, r, r2, n=24):
    """Contour (monde, mm) d'une cornière : dos de l'aile verticale sur u = 0, dessous sur y = 0."""
    pts = [(0.0, 0.0), (larg, 0.0), (larg, e - r2)]
    for k in range(1, n):
        a = k / n * math.pi / 2
        pts.append((larg - r2 + r2 * math.cos(a), e - r2 + r2 * math.sin(a)))
    pts.append((larg - r2, e))
    pts.append((e + r, e))
    for k in range(1, n):
        a = -math.pi / 2 - k / n * math.pi / 2
        pts.append((e + r + r * math.cos(a), e + r + r * math.sin(a)))
    pts.append((e, e + r))
    pts.append((e, haut - r2))
    for k in range(1, n):
        a = k / n * math.pi / 2
        pts.append((e - r2 + r2 * math.cos(a), haut - r2 + r2 * math.sin(a)))
    pts.append((e - r2, haut))
    pts.append((0.0, haut))
    return pts

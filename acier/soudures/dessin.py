# -*- coding: utf-8 -*-
"""L'encart « LA GORGE » (SVG) et la carte des taux (HTML) : l'encart montre
la section du cordon gouvernant — les deux pièces coupées, le cordon d'angle,
la pénétration partielle ou la pleine pénétration — et les composantes σ⊥,
τ⊥, τ∥ au point le plus chargé (EN 1993-1-8 fig. 4.5), avec le signe de σ⊥
(sortante : traction, entrante : compression).

Les vues des configurations (plan de joint, coupes, élévations) vivent dans
``schemas.py``, sur le kit de dessin des assemblages. Ici reste un petit
canevas SVG en pixels (hachures, flèches, textes à halo) pour l'encart.
Palette : pièces #F1F5F9 / #E2E8F0, trait #111827, hachures #6B7280, cordon
#D98A00, efforts #9C3341, taux vert / ocre / rouge.
"""
import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ENCRE, TRAIT, PIECE_A, PIECE_B, CORDON, EFFORT = "#111827", "#111827", "#E2E8F0", "#F1F5F9", "#D98A00", "#9C3341"
HACH, COTE, MUT, FOND = "#6B7280", "#374151", "#6B7280", "#FFFFFF"
OK, ATT, NOK = "#2E7D46", "#D98A00", "#9C3341"
POLICE = 'font-family="Inter, Segoe UI, Arial, sans-serif"'
_compteur = itertools.count(1)


def _fmt(v: float, n: int = 1) -> str:
    return f"{v:,.{n}f}".replace(",", " ").replace(".", ",")


def _larg(t: str, taille: float) -> float:
    """Largeur estimée d'un texte (px) : 0,56 em par caractère."""
    return 0.56 * taille * len(t) + 2


def couleur_taux(t: Optional[float]) -> str:
    """Vert jusqu'à 70 %, dégradé vers l'ocre à 100 %, rouge au-delà."""
    if t is None:
        return MUT
    if t >= 1.0:
        return NOK
    if t <= 0.7:
        return OK
    u = (t - 0.7) / 0.3
    c1, c2 = (0x2E, 0x7D, 0x46), (0xD9, 0x8A, 0x00)
    return "#%02X%02X%02X" % tuple(int(round(a + (b - a) * u)) for a, b in zip(c1, c2))


class Canevas:
    """Un repère mm → px (y vers le haut), origine et échelle communes à tout le dessin."""

    def __init__(self, largeur_px: int, hauteur_px: int, xmin: float, xmax: float, ymin: float, ymax: float, marge: float = 40.0):
        self.W, self.H = largeur_px, hauteur_px
        self.ech = min((largeur_px - 2 * marge) / max(1e-9, xmax - xmin), (hauteur_px - 2 * marge) / max(1e-9, ymax - ymin))
        self.x0 = marge + ((largeur_px - 2 * marge) - (xmax - xmin) * self.ech) / 2 - xmin * self.ech
        self.y0 = marge + ((hauteur_px - 2 * marge) - (ymax - ymin) * self.ech) / 2 + ymax * self.ech
        self.p = f"s{next(_compteur)}_"
        self.out: List[str] = []

    def X(self, x: float) -> float:
        return self.x0 + x * self.ech

    def Y(self, y: float) -> float:
        return self.y0 - y * self.ech

    # ---------------------------------------------------------------- primitives
    def _defs(self) -> str:
        p = self.p
        return (f'<defs><pattern id="{p}hA" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">'
                f'<line x1="0" y1="0" x2="0" y2="7" stroke="{HACH}" stroke-width="0.7"/></pattern>'
                f'<pattern id="{p}hB" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(-45)">'
                f'<line x1="0" y1="0" x2="0" y2="7" stroke="{HACH}" stroke-width="0.7"/></pattern>'
                f'<pattern id="{p}hW" patternUnits="userSpaceOnUse" width="3.5" height="3.5" patternTransform="rotate(45)">'
                f'<line x1="0" y1="0" x2="0" y2="3.5" stroke="#8A5A00" stroke-width="0.8"/></pattern></defs>')

    def polygone(self, pts, fill=PIECE_B, stroke=TRAIT, sw=1.3, dash=None, opacity=1.0, hachure: Optional[str] = None):
        """Polygone en mm ; hachure "A" / "B" (pièces coupées) ou "W" (cordon coupé)."""
        d = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<polygon points="{d}" fill="{fill}" stroke="none" fill-opacity="{opacity}"/>')
        if hachure:
            self.out.append(f'<polygon points="{d}" fill="url(#{self.p}h{hachure})" stroke="none"/>')
        self.out.append(f'<polygon points="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{da} stroke-linejoin="round"/>')

    def anneau(self, ext, trous: Sequence, fill=PIECE_B, stroke=TRAIT, sw=1.3, hachure: Optional[str] = None):
        """Polygone à trous (section d'un tube) : règle evenodd."""
        def chem(ring):
            return "M" + " L".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in ring) + " Z"
        d = chem(ext) + " " + " ".join(chem(t) for t in trous)
        self.out.append(f'<path d="{d}" fill="{fill}" fill-rule="evenodd" stroke="none"/>')
        if hachure:
            self.out.append(f'<path d="{d}" fill="url(#{self.p}h{hachure})" fill-rule="evenodd" stroke="none"/>')
        self.out.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')

    def ligne(self, p, q, stroke=TRAIT, sw=1.0, dash=None, cap="butt"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<line x1="{self.X(p[0]):.1f}" y1="{self.Y(p[1]):.1f}" x2="{self.X(q[0]):.1f}" y2="{self.Y(q[1]):.1f}" '
                        f'stroke="{stroke}" stroke-width="{sw}"{da} stroke-linecap="{cap}"/>')

    def ligne_px(self, x1, y1, x2, y2, stroke=TRAIT, sw=1.0, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{da}/>')

    def axe(self, p, q):
        """Trait d'axe (mixte fin)."""
        self.ligne(p, q, COTE, 0.6, "9 3 1.5 3")

    def cache(self, p, q, stroke=TRAIT, sw=0.9):
        """Arête cachée (interrompu)."""
        self.ligne(p, q, stroke, sw, "5 3")

    def texte(self, x, y, t, taille=11, fill=ENCRE, anc="middle", gras=False, dx=0.0, dy=0.0, rot=None, boite=False):
        self.texte_px(self.X(x) + dx, self.Y(y) + dy, t, taille, fill, anc, gras, rot, boite)

    def texte_px(self, px, py, t, taille=11, fill=ENCRE, anc="middle", gras=False, rot=None, boite=False):
        g = ' font-weight="600"' if gras else ""
        tr = f' transform="rotate({rot} {px:.1f} {py:.1f})"' if rot is not None else ""
        if boite:
            w = _larg(t, taille)
            bx = px - w / 2 if anc == "middle" else (px - 2 if anc == "start" else px - w + 2)
            self.out.append(f'<rect x="{bx:.1f}" y="{py - taille * 0.85:.1f}" width="{w:.1f}" height="{taille * 1.15:.1f}" rx="2" '
                            f'fill="{FOND}" fill-opacity="0.92"{tr}/>')
        self.out.append(f'<text x="{px:.1f}" y="{py:.1f}" font-size="{taille}" fill="{fill}" text-anchor="{anc}"{g}{tr}>{t}</text>')

    def fleche(self, p, q, label="", stroke=EFFORT, sw=2.2, taille=11.5, cote=+1):
        """Flèche droite de p vers q (mm), pointe en q ; libellé sur le côté ``cote``
        (+1 : à droite d'une flèche verticale, au-dessus d'une horizontale)."""
        self.fleche_px(self.X(p[0]), self.Y(p[1]), self.X(q[0]), self.Y(q[1]), label, stroke, sw, taille, cote)

    def fleche_px(self, x1, y1, x2, y2, label="", stroke=EFFORT, sw=2.2, taille=11.5, cote=+1):
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        ux, uy = dx / L, dy / L
        h = 9.0
        bx, by = x2 - ux * h, y2 - uy * h
        self.out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.out.append(f'<polygon points="{x2:.1f},{y2:.1f} {bx - uy * 4.2:.1f},{by + ux * 4.2:.1f} {bx + uy * 4.2:.1f},{by - ux * 4.2:.1f}" fill="{stroke}"/>')
        if not label:
            return
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if abs(uy) >= abs(ux):                      # flèche verticale : libellé à côté du fût
            self.texte_px(mx + cote * 9, my + 4, label, taille, stroke, "start" if cote > 0 else "end", True, boite=True)
        else:                                       # horizontale : au-dessus (cote +1) ou au-dessous
            self.texte_px(mx, my - 8 if cote > 0 else my + taille + 4, label, taille, stroke, "middle", True, boite=True)

    def moment(self, c, r_px, label="", sens=+1, stroke=EFFORT, taille=11.5, pos="bas"):
        """Flèche courbe (3/4 de tour) autour du point c (mm), rayon en px ; sens +1 trigonométrique."""
        cx, cy = self.X(c[0]), self.Y(c[1])
        a0, a1 = (-135.0, 135.0) if sens > 0 else (-45.0, -315.0)
        p0 = (cx + r_px * math.cos(math.radians(a0)), cy - r_px * math.sin(math.radians(a0)))
        p1 = (cx + r_px * math.cos(math.radians(a1)), cy - r_px * math.sin(math.radians(a1)))
        sweep = 0 if sens > 0 else 1
        self.out.append(f'<path d="M{p0[0]:.1f},{p0[1]:.1f} A{r_px},{r_px} 0 1 {sweep} {p1[0]:.1f},{p1[1]:.1f}" fill="none" stroke="{stroke}" stroke-width="2"/>')
        t = math.radians(a1) + (math.pi / 2 if sens > 0 else -math.pi / 2)
        tx, ty = math.cos(t), -math.sin(t)
        self.out.append(f'<polygon points="{p1[0] + tx * 8:.1f},{p1[1] + ty * 8:.1f} {p1[0] - ty * 4.2:.1f},{p1[1] + tx * 4.2:.1f} '
                        f'{p1[0] + ty * 4.2:.1f},{p1[1] - tx * 4.2:.1f}" fill="{stroke}"/>')
        self.out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.8" fill="{stroke}"/>')
        if label:
            if pos == "bas":
                self.texte_px(cx, cy + r_px + taille + 4, label, taille, stroke, "middle", True, boite=True)
            elif pos == "haut":
                self.texte_px(cx, cy - r_px - 6, label, taille, stroke, "middle", True, boite=True)
            elif pos == "droite":
                self.texte_px(cx + r_px + 8, cy + 4, label, taille, stroke, "start", True, boite=True)
            else:
                self.texte_px(cx - r_px - 8, cy + 4, label, taille, stroke, "end", True, boite=True)

    def vecteur_moment(self, c, direction, label="", longueur_px=44.0, stroke=EFFORT, taille=11.5, cote=+1):
        """Vecteur moment (double pointe) centré en c (mm), le long de ``direction`` (sens positif)."""
        cx, cy = self.X(c[0]), self.Y(c[1])
        ux, uy = direction[0], -direction[1]
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        x1, y1 = cx - ux * longueur_px / 2, cy - uy * longueur_px / 2
        x2, y2 = cx + ux * longueur_px / 2, cy + uy * longueur_px / 2
        h = 8.0
        self.out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2 - ux * h:.1f}" y2="{y2 - uy * h:.1f}" stroke="{stroke}" stroke-width="2.2"/>')
        for k in (0.0, 1.0):
            tx, ty = x2 - ux * h * k, y2 - uy * h * k
            self.out.append(f'<polygon points="{tx:.1f},{ty:.1f} {tx - ux * h - uy * 4.2:.1f},{ty - uy * h + ux * 4.2:.1f} '
                            f'{tx - ux * h + uy * 4.2:.1f},{ty - uy * h - ux * 4.2:.1f}" fill="{stroke}"/>')
        if label:
            if abs(uy) >= abs(ux):
                self.texte_px(cx + cote * 9, cy + 4, label, taille, stroke, "start" if cote > 0 else "end", True, boite=True)
            else:
                self.texte_px(cx, cy - 9 if cote > 0 else cy + taille + 5, label, taille, stroke, "middle", True, boite=True)

    def symbole_normal(self, c, label="", sortant=True, stroke=EFFORT, taille=11.5, cote=+1):
        """⊙ (vers le lecteur) ou ⊗ (loin du lecteur) : effort normal au dessin."""
        cx, cy = self.X(c[0]), self.Y(c[1])
        self.out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#fff" stroke="{stroke}" stroke-width="2"/>')
        if sortant:
            self.out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.2" fill="{stroke}"/>')
        else:
            self.out.append(f'<path d="M{cx - 4:.1f},{cy - 4:.1f}L{cx + 4:.1f},{cy + 4:.1f}M{cx - 4:.1f},{cy + 4:.1f}L{cx + 4:.1f},{cy - 4:.1f}" stroke="{stroke}" stroke-width="2"/>')
        if label:
            self.texte_px(cx + cote * 12, cy + 4, label, taille, stroke, "start" if cote > 0 else "end", True, boite=True)

    def cote(self, p, q, label, decal=16.0, pos=0.5, taille=10, attaches=True):
        """Cote entre p et q (mm) : lignes d'attache, ligne de cote décalée de
        ``decal`` px (à droite du sens p→q si > 0), pointes, texte à ``pos``."""
        x1, y1, x2, y2 = self.X(p[0]), self.Y(p[1]), self.X(q[0]), self.Y(q[1])
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        ux, uy = dx / L, dy / L
        nx, ny = -uy * decal, ux * decal
        if attaches:
            s = 1.0 if decal >= 0 else -1.0
            for (x, y) in ((x1, y1), (x2, y2)):
                self.out.append(f'<line x1="{x + nx * 0.12:.1f}" y1="{y + ny * 0.12:.1f}" x2="{x + nx + (-uy) * s * 2.5:.1f}" '
                                f'y2="{y + ny + ux * s * 2.5:.1f}" stroke="{COTE}" stroke-width="0.6"/>')
        ax, ay, bx, by = x1 + nx, y1 + ny, x2 + nx, y2 + ny
        self.out.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{COTE}" stroke-width="0.7"/>')
        h = 6.0
        for (x, y, sx, sy) in ((ax, ay, ux, uy), (bx, by, -ux, -uy)):
            self.out.append(f'<polygon points="{x:.1f},{y:.1f} {x + sx * h - sy * 1.7:.1f},{y + sy * h + sx * 1.7:.1f} '
                            f'{x + sx * h + sy * 1.7:.1f},{y + sy * h - sx * 1.7:.1f}" fill="{COTE}"/>')
        rot = math.degrees(math.atan2(dy, dx))
        if rot > 90 or rot < -90:
            rot += 180
        # texte du côté opposé aux pièces (au-dessus de la ligne dans son sens de lecture)
        sgn = 1.0 if decal >= 0 else -1.0
        mx, my = ax + (bx - ax) * pos + (-uy) * sgn * 4.5, ay + (by - ay) * pos + ux * sgn * 4.5
        anc = "middle"
        self.out.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="{taille}" fill="{COTE}" text-anchor="{anc}" '
                        f'transform="rotate({rot:.1f} {mx:.1f} {my:.1f})" dy="{-2.5 if sgn > 0 else taille * 0.85:.1f}">{label}</text>')

    def cordon_taux(self, c, taux3: Sequence[float], epaisseur_px: float = 7.0, gouvernant: Optional[float] = None, cache: bool = False):
        """Le cordon de p1 à p2 coloré par ses taux (début, milieu, fin) ; le point gouvernant marqué ◉."""
        x1, y1, x2, y2 = self.X(c["p1"][0]), self.Y(c["p1"][1]), self.X(c["p2"][0]), self.Y(c["p2"][1])
        gid = f"{self.p}g{len(self.out)}"
        stops = "".join(f'<stop offset="{o}" stop-color="{couleur_taux(t)}"/>' for o, t in zip((0, 0.5, 1), taux3))
        self.out.append(f'<defs><linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">{stops}</linearGradient></defs>')
        da = ' stroke-dasharray="6 4"' if cache else ""
        self.out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="url(#{gid})" stroke-width="{epaisseur_px}" '
                        f'stroke-linecap="{"butt" if cache else "round"}"{da}/>')
        if gouvernant is not None:
            gx, gy = x1 + (x2 - x1) * gouvernant, y1 + (y2 - y1) * gouvernant
            self.out.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="5.5" fill="#fff" stroke="{NOK}" stroke-width="2"/>'
                            f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="2" fill="{NOK}"/>')

    def cordon_exclu(self, c, epaisseur_px: float = 5.0):
        self.ligne(c["p1"], c["p2"], MUT, epaisseur_px, "3 4")

    def marque_G(self, x, y, label="G"):
        cx, cy = self.X(x), self.Y(y)
        self.out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="none" stroke="{ENCRE}" stroke-width="1"/>'
                        f'<line x1="{cx - 6:.1f}" y1="{cy:.1f}" x2="{cx + 6:.1f}" y2="{cy:.1f}" stroke="{ENCRE}" stroke-width="0.8"/>'
                        f'<line x1="{cx:.1f}" y1="{cy - 6:.1f}" x2="{cx:.1f}" y2="{cy + 6:.1f}" stroke="{ENCRE}" stroke-width="0.8"/>')
        self.texte_px(cx + 7, cy - 5, label, 9.5, ENCRE, "start")

    def point_R(self, x, y, label="R"):
        cx, cy = self.X(x), self.Y(y)
        self.out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{EFFORT}"/>')
        self.texte_px(cx - 7, cy + 12, label, 9.5, EFFORT, "end")

    def repere_xy(self, px=28.0, py=None, long_px=26.0, noms=("x", "y")):
        """Petit repère x/y dans le coin bas-gauche (px)."""
        py = self.H - 26.0 if py is None else py
        self.fleche_px(px, py, px + long_px, py, "", COTE, 1.0, 9)
        self.fleche_px(px, py, px, py - long_px, "", COTE, 1.0, 9)
        self.texte_px(px + long_px + 4, py + 3, noms[0], 9.5, COTE, "start")
        self.texte_px(px + 4, py - long_px - 2, noms[1], 9.5, COTE, "start")

    def titre(self, t: str, sous: str = ""):
        self.texte_px(10, 15, t, 10.5, ENCRE, "start", True)
        if sous:
            self.texte_px(self.W - 10, 15, sous, 9.5, MUT, "end")

    def cartouche(self, t: str):
        self.texte_px(self.W - 10, self.H - 8, t, 9, MUT, "end")

    def svg(self, fond: Optional[str] = FOND, cadre: bool = True) -> str:
        bg = f'<rect width="{self.W}" height="{self.H}" fill="{fond}"/>' if fond else ""
        fr = f'<rect x="0.5" y="0.5" width="{self.W - 1}" height="{self.H - 1}" fill="none" stroke="#CBD5E1" stroke-width="1"/>' if cadre else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" width="100%" style="max-width:{self.W}px;display:block" '
                f'{POLICE} font-size="11">' + bg + self._defs() + "".join(self.out) + fr + "</svg>")


# ====================================================================== encart « la gorge »
def encart_gorge(res: Dict, W: int = 320, H: int = 330) -> str:
    """La section du cordon gouvernant : les deux pièces coupées (hachurées),
    le triangle du cordon, le plan de gorge a (interrompu), σ⊥ normale à ce
    plan, τ⊥ dans ce plan, τ∥ le long du cordon (⊙ vers soi / ⊗), puis les
    valeurs et le critère de la méthode choisie."""
    g = res.get("gouvernant")
    cv = Canevas(W, H, 0, 100, 0, 100, marge=8)
    cv.titre("LA GORGE AU POINT GOUVERNANT")
    if not g:
        cv.texte(50, 55, "pas de cordon gouvernant", 10.5, ENCRE)
        return cv.svg()
    a = g["a"]
    cord = next((c for c in res.get("cordons", []) if c["nom"] == g["cordon"]), {})
    bout = bool(cord.get("bout_a_bout"))
    pleine = bool(cord.get("pleine"))
    t_piece = float(cord.get("t_piece") or a)
    coplanaire = pleine and res.get("config") == "C7"
    nom_ch = cord.get("chaine") or g["cordon"]
    sp, tp, tpar = g["sigma_perp"], g["tau_perp"], g["tau_par"]
    if coplanaire:
        # bout à bout coplanaire à pleine pénétration : le V sur toute l'épaisseur, la « gorge » est la section du plat
        cv.polygone([(4, 50), (48, 50), (42, 78), (4, 78)], PIECE_A, hachure="A")
        cv.polygone([(52, 50), (96, 50), (96, 78), (58, 78)], PIECE_B, hachure="B")
        cv.polygone([(48, 50), (52, 50), (58, 78), (42, 78)], CORDON, TRAIT, 0.8, hachure="W")
        cv.ligne((50, 50), (50, 78), ENCRE, 1.1, "3 2")
        cv.texte(50, 52, f"t = {_fmt(t_piece, 0)}", 9.5, ENCRE, "start", dx=10, dy=4, boite=True)
        mx, my = 50.0, 66.0
        cv.fleche((mx, my), (mx + (20 if sp >= 0 else -20), my), "σ⊥", EFFORT, 2.0, 10.5, cote=+1)
        cv.fleche((mx, my), (mx, my + (14 if tp >= 0 else -14)), "τ⊥", EFFORT, 2.0, 10.5, cote=+1)
        cv.symbole_normal((mx, my), "", tpar >= 0)
        cv.texte(50, 44, "pleine pénétration : la soudure vaut le plat, gorge = t", 9, MUT)
    elif pleine:
        # pièce debout soudée à pleine pénétration : toute l'épaisseur t est soudée, la « gorge » est la section de la pièce à sa racine
        cv.polygone([(4, 30), (70, 30), (70, 42), (4, 42)], PIECE_A, hachure="A")
        cv.polygone([(16, 54), (30, 54), (30, 96), (16, 96)], PIECE_B, hachure="B")
        cv.polygone([(8, 42), (38, 42), (30, 50), (30, 54), (16, 54), (16, 50)], CORDON, TRAIT, 0.9, hachure="W")
        cv.ligne((16, 42), (30, 42), ENCRE, 1.2, "3 2")
        cv.texte(30, 42, f"gorge = t = {_fmt(t_piece, 0)}", 9.5, ENCRE, "start", dx=10, dy=-4, boite=True)
        cv.texte(10, 35, "A", 9.5, MUT)
        cv.texte(23, 80, "B", 9.5, MUT)
        mx, my = 23.0, 42.0
        cv.symbole_normal((mx, my), "", tpar >= 0)
        q = (mx, my + (17 if sp >= 0 else -14))
        if sp >= 0:
            cv.fleche((mx, my + 3), q, "", EFFORT, 2.0)
        else:
            cv.fleche((mx, my - 3), q, "", EFFORT, 2.0)
        cv.texte(q[0], q[1], "σ⊥", 10.5, EFFORT, "start", True, dx=5, dy=(4 if sp >= 0 else -2), boite=True)
        q2 = (mx + (19 if tp >= 0 else -19), my)
        cv.fleche((mx + (3 if tp >= 0 else -3), my), q2, "", EFFORT, 2.0)
        cv.texte(q2[0], q2[1], "τ⊥", 10.5, EFFORT, "start" if tp >= 0 else "end", True, dx=(4 if tp >= 0 else -4), dy=-5, boite=True)
        cv.texte(39, 62, "τ∥", 10.5, EFFORT, "middle", True, boite=True)
        cv.texte(98, 90, nom_ch, 9.5, ENCRE, "end", True)
        cv.texte(98, 84, g["position"], 9.5, MUT, "end")
        cv.texte(98, 76, f"η = {g['taux'] * 100:.0f} %", 11, couleur_taux(g["taux"]), "end", True)
        cv.texte(98, 68, "pleine pénétration", 9, MUT, "end")
        cv.texte(98, 62, "la soudure vaut la pièce", 9, MUT, "end")
    elif bout:
        # pénétration partielle : deux plats bout à bout, la gorge = la pénétration, normale à l'effort
        cv.polygone([(4, 50), (47, 50), (47, 78), (4, 78)], PIECE_A, hachure="A")
        cv.polygone([(53, 50), (96, 50), (96, 78), (53, 78)], PIECE_B, hachure="B")
        cv.polygone([(42, 78), (58, 78), (50, 62)], CORDON, TRAIT, 0.8, hachure="W")
        cv.ligne((50, 62), (50, 78), ENCRE, 1.1, "3 2")
        cv.texte(50, 64, f"p = a = {_fmt(a, 1)}", 9.5, ENCRE, "start", dx=8, dy=4, boite=True)
        mx, my = 50.0, 70.0
        cv.fleche((mx, my), (mx + (20 if sp >= 0 else -20), my), "σ⊥", EFFORT, 2.0, 10.5, cote=+1)
        cv.fleche((mx, my), (mx, my + (16 if tp >= 0 else -16)), "τ⊥", EFFORT, 2.0, 10.5, cote=+1)
        cv.symbole_normal((mx, my), "", tpar >= 0)
        cv.texte(50, 44, "bout à bout, pénétration partielle : gorge ⊥ à N", 9, MUT)
    else:
        # pièce A horizontale en bas (support), pièce B debout, cordon dans le dièdre
        cv.polygone([(4, 30), (70, 30), (70, 42), (4, 42)], PIECE_A, hachure="A")
        cv.polygone([(16, 42), (30, 42), (30, 96), (16, 96)], PIECE_B, hachure="B")
        z = 34.0
        cv.polygone([(30, 42), (30 + z, 42), (30, 42 + z)], CORDON, TRAIT, 0.9, hachure="W")
        mx, my = 30 + z / 2, 42 + z / 2                       # milieu de la face du cordon
        cv.ligne((30, 42), (mx, my), ENCRE, 1.2, "3 2")       # le plan de gorge
        cv.texte(30 + z / 4, 42 + z / 4, f"a = {_fmt(a, 1)}", 9.5, ENCRE, "start", dx=6, dy=-3, boite=True)
        cv.texte(10, 35, "A", 9.5, MUT)
        cv.texte(23, 72, "B", 9.5, MUT)
        L = 17.0
        s1 = 1 if sp >= 0 else -1
        s2 = 1 if tp >= 0 else -1
        # σ⊥ normale au plan de gorge, depuis la face du cordon : sortante (traction) ou entrante (compression)
        q = (mx + L * 0.707, my - L * 0.707)
        if s1 > 0:
            cv.fleche((mx, my), q, "", EFFORT, 2.0)
        else:
            cv.fleche(q, (mx, my), "", EFFORT, 2.0)
        cv.texte(q[0], q[1], "σ⊥", 10.5, EFFORT, "start", True, dx=5, dy=4, boite=True)
        q2 = (mx + L * 0.707, my + L * 0.707)
        if s2 > 0:
            cv.fleche((mx, my), q2, "", EFFORT, 2.0)
        else:
            cv.fleche(q2, (mx, my), "", EFFORT, 2.0)
        cv.texte(q2[0], q2[1], "τ⊥", 10.5, EFFORT, "start", True, dx=5, dy=4, boite=True)
        cv.symbole_normal((mx, my), "", tpar >= 0)
        cv.texte(39, 74, "τ∥", 10.5, EFFORT, "middle", True, boite=True)
        cv.texte(98, 90, nom_ch, 9.5, ENCRE, "end", True)
        cv.texte(98, 84, g["position"], 9.5, MUT, "end")
        cv.texte(98, 76, f"η = {g['taux'] * 100:.0f} %", 11, couleur_taux(g["taux"]), "end", True)
    # les valeurs
    lignes = [(f"σ⊥ = {_fmt(sp)}   τ⊥ = {_fmt(tp)}   τ∥ = {_fmt(tpar)} N/mm²", 9.5, ENCRE),
              (f"{'traction' if sp >= 0 else 'compression'} sur la gorge · τ∥ {'⊙ vers soi' if tpar >= 0 else '⊗ vers le fond'} · |f| = {_fmt(g['f'], 0)} N/mm", 8.8, MUT),
              (f"σ_c = √(σ⊥² + 3(τ⊥² + τ∥²)) = {_fmt(g['sigma_c'])} N/mm²", 9.5, ENCRE)]
    crit = g.get("critere", "")
    if crit:
        lignes.append((crit[:72], 9.0, ENCRE))
    a_droite = coplanaire or (bout and not pleine)                    # pas d'étiquettes à droite du dessin : une ligne de plus
    if a_droite:
        lignes.append((f"{nom_ch} · {g['position']} · η = {g['taux'] * 100:.0f} %", 10, couleur_taux(g["taux"])))
    for i, (t, taille, coul) in enumerate(lignes):
        cv.texte(50, 24 - i * 6.3, t, taille, coul, "middle", gras=(i == len(lignes) - 1 and a_droite))
    return cv.svg()


# ====================================================================== carte des taux (HTML)
def carte_taux(res: Dict) -> str:
    """Les cordons tels que nommés avec leur taux en barre colorée (HTML sans Streamlit)."""
    ch = res.get("chaines") or []
    if not ch:
        return '<div style="font-size:12px;color:#6b7280">aucun cordon utile</div>'
    out = ['<table style="width:100%;border-collapse:collapse;font-size:12px">']
    for c in sorted(ch, key=lambda k: -k["taux"]):
        t = c["taux"]
        col = couleur_taux(t)
        out.append(f'<tr><td style="padding:3px 4px;white-space:nowrap"><b>{c["nom"]}</b> <span style="color:#6b7280">a{_fmt(c["a"], 0)} · '
                   f'{_fmt(c["l_eff"], 0)} mm eff.</span></td>'
                   f'<td style="width:45%;padding:3px 4px"><div style="height:7px;background:#f1f3f5;border-radius:3px;overflow:hidden">'
                   f'<div style="height:100%;width:{min(t, 1) * 100:.0f}%;background:{col}"></div></div></td>'
                   f'<td style="padding:3px 4px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;color:{col};font-weight:600">{t * 100:.0f} %</td></tr>')
    out.append("</table>")
    return "".join(out)


def legende() -> str:
    return ("Pièces au trait fort, sections hachurées, lignes de rupture où la pièce continue · cordons en bande ocre à leur taille a·√2, "
            "symbole de soudure EN 22553 (a : gorge, s : pénétration), interrompu gris : cordon qui ne compte pas · ◉ point gouvernant et son taux · "
            "G centre des gorges, R point de référence · flèches rouges : efforts dans le sens saisi (⊙ vers soi, ⊗ vers le fond, ⇒ vecteur moment, "
            "flèche courbe : moment dans le plan) · cotes en mm · encart : les composantes sur la gorge, EN 1993-1-8 fig. 4.5")

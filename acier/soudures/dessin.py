# -*- coding: utf-8 -*-
"""Les schémas des soudures en SVG, en style DESSIN TECHNIQUE : pièces au
trait fort, sections hachurées, cotes fines à lignes d'attache, axes en
trait mixte, efforts en rouge avec leur sens, cordons colorés par leur taux,
point gouvernant marqué, et l'encart « la gorge » qui montre σ⊥, τ⊥, τ∥ au
point le plus chargé (EN 1993-1-8 fig. 4.5).

Tout est dessiné depuis les objets du calcul (cordons, pièces des
constructeurs, résultats) : le dessin est une vue, jamais une seconde
géométrie. Unités du dessin : mm du plan de joint, mis à l'échelle dans un
canevas en pixels ; le SVG porte un viewBox et s'étire à la largeur de sa
colonne. Palette : pièces #F1F5F9 / #E2E8F0, trait #111827, hachures #6B7280,
cordons #D98A00, efforts #9C3341, cotes #374151, taux vert / ocre / rouge.
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


# ====================================================================== outils communs aux vues
def _taux3(res: Dict, nom: str) -> List[float]:
    pts = sorted([p for p in res["points"] if p["cordon"] == nom], key=lambda p: p["s"])
    return [p["taux"] for p in pts] if len(pts) == 3 else [0.0, 0.0, 0.0]


def _gouv_s(res: Dict, nom: str) -> Optional[float]:
    g = res.get("gouvernant")
    return g["s"] if g and g.get("cordon") == nom else None


def _cordons(cv: Canevas, res: Dict, ep: float = 7.0, transformer=None, caches: Sequence[str] = ()):
    """Tous les cordons utiles colorés, les exclus en pointillé, le point gouvernant."""
    tr = transformer or (lambda c: c)
    for c in res.get("cordons_exclus", []):
        cv.cordon_exclu(tr(c), ep * 0.7)
    for c in res.get("cordons", []):
        cv.cordon_taux(tr(c), _taux3(res, c["nom"]), ep, _gouv_s(res, c["nom"]), cache=c["nom"] in caches)


def _etiquette_gouvernant(cv: Canevas, res: Dict, transformer=None, u=None):
    """« ◉ 61 % » à côté du point gouvernant, décalé du côté de la pièce
    rapportée (−u_A) — les noms et gorges vont de l'autre côté."""
    g = res.get("gouvernant")
    if not g or g.get("cordon") == "pièce":
        return
    c = next((k for k in res.get("cordons", []) if k["nom"] == g["cordon"]), None)
    pt = (transformer or (lambda p: p))(g["point"])
    ua = u if u is not None else ((-c["u_A"][0], -c["u_A"][1]) if c else (1.0, 0.0))
    n = math.hypot(ua[0], ua[1]) or 1.0
    dx, dy = ua[0] / n * 17, -ua[1] / n * 17
    anc = "start" if dx > 3 else ("end" if dx < -3 else "middle")
    cv.texte(pt[0], pt[1], f"◉ {g['taux'] * 100:.0f} %", 10.5, NOK, anc, True, dx=dx, dy=dy + 4, boite=True)


def _milieu_chaine(res: Dict, nom_chaine: str, transformer=None) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Milieu du segment médian d'une chaîne et son u_A (pour poser un nom hors de la pièce)."""
    segs = [k for k in res.get("cordons", []) + res.get("cordons_exclus", []) if (k.get("chaine") or k["nom"]) == nom_chaine]
    c = segs[len(segs) // 2]
    c2 = (transformer or (lambda k: k))(c)
    mx, my = (c2["p1"][0] + c2["p2"][0]) / 2, (c2["p1"][1] + c2["p2"][1]) / 2
    ua = c.get("u_A", [1.0, 0.0])
    return (mx, my), (ua[0], ua[1])


def _etiquette_gorge(cv: Canevas, res: Dict, nom: str, texte: str, transformer=None, decal_px: float = 14.0):
    """« a5 » près du milieu d'un cordon (ou d'une chaîne), hors de la pièce rapportée."""
    c = next((k for k in res.get("cordons", []) if k["nom"] == nom or k.get("chaine") == nom), None)
    if c is None:
        return
    (mx, my), ua = _milieu_chaine(res, c.get("chaine") or c["nom"], transformer)
    n = math.hypot(ua[0], ua[1]) or 1.0
    dx, dy = ua[0] / n * decal_px, -ua[1] / n * decal_px
    anc = "start" if dx > 3 else ("end" if dx < -3 else "middle")
    cv.texte(mx, my, texte, 9.5, "#8A5A00", anc, False, dx=dx, dy=dy + 3.5, boite=True)


def _gorges(res: Dict) -> str:
    gs = sorted({round(c["a"], 1) for c in res.get("cordons", [])})
    return " / ".join(f"a{_fmt(a, 0) if float(a).is_integer() else _fmt(a, 1)}" for a in gs)


def _triangle_cordon(cv: Canevas, racine, u_face_A, u_face_B, a: float, trait=True):
    """Triangle d'un cordon d'angle coupé : racine, ailes a·√2 le long des deux faces."""
    z = a * 1.4142
    p = np.asarray(racine, float)
    A = p + np.asarray(u_face_A, float) * z
    B = p + np.asarray(u_face_B, float) * z
    cv.polygone([tuple(p), tuple(A), tuple(B)], CORDON, TRAIT if trait else CORDON, 0.8, hachure="W")


def _val_kN(v):
    return f"{_fmt(abs(v) / 1e3)} kN"


def _val_kNm(v):
    return f"{_fmt(abs(v) / 1e6)} kN·m"


# ====================================================================== C1 recouvrement (plan)
def vue_plan_recouvrement(res: Dict, W: int = 640, H: int = 400) -> str:
    pc = res["pieces"]
    L, b = pc["L"], pc["b"]
    xs = [x for x, _ in pc["A"]] + [x for x, _ in pc["B"]]
    cv = Canevas(W, H, min(xs) - 10, L * 1.6 + 115, -b * 0.75 - 50, b * 0.75 + 45, marge=34)
    cv.titre("C1 · RECOUVREMENT — vue de dessus", f"cordons {_gorges(res)} · cotes en mm")
    cv.polygone(pc["A"], PIECE_A)
    cv.polygone(pc["B"], PIECE_B)
    cv.cache((L, -b / 2), (L, b / 2))                       # bout du plat A sous B
    cv.texte(L * 0.2, -b * 0.63, f"plat A (dessous) · t {_fmt(pc['tA'], 0)}", 10, MUT, "middle")
    cv.texte(L * 0.5, 0.0, f"plat B (dessus) · t {_fmt(pc['tB'], 0)}", 10, MUT, "middle", dy=4)
    _cordons(cv, res, 7.0, caches=("frontal (bout de A)",))
    _etiquette_gouvernant(cv, res)
    e = res["efforts"]
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.axe((L * 1.6, 0), (L * 1.6 + 30, 0))
        cv.fleche((L * 1.6 + 30, 0), (L * 1.6 + 30 + s * 60, 0), f"N = {_val_kN(e.Vx)}")
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((L * 1.35, s * (b / 2 + 3)), (L * 1.35, s * (b / 2 + 48)), f"V = {_val_kN(e.Vy)}")
    cv.cote((0, -b * 0.75), (L, -b * 0.75), f"L = {_fmt(L, 0)}", 18)
    cv.cote((L * 1.6, -b / 2), (L * 1.6, b / 2), f"b = {_fmt(b, 0)}", -16, pos=0.25)
    cv.repere_xy()
    cv.cartouche("cordons colorés par leur taux · ◉ point gouvernant · pointillé : cordon caché (face inférieure)")
    return cv.svg()


# ====================================================================== C2 jonction en T (coupe + plan)
def vue_elevation_T(res: Dict, W: int = 640, H: int = 400) -> str:
    pc = res["pieces"]
    tA, tB, l, hB = pc["tA"], pc["tB"], pc["l"], pc["hB"]
    a = res["cordons"][0]["a"] if res.get("cordons") else 5.0
    larg_coupe = max(hB, 110.0)
    cv = Canevas(W, H, -larg_coupe - 30, l + 70, -tA - 60, hB + 40, marge=34)
    cv.titre("C2 · JONCTION EN T — coupe A-A et vue en plan", f"cordons {_gorges(res)} · cotes en mm")
    # --- coupe à gauche : A horizontal (coupé), B debout (coupée), triangles des cordons
    x_c = -larg_coupe / 2 - 15
    cv.polygone([(x_c - larg_coupe / 2, -tA), (x_c + larg_coupe / 2, -tA), (x_c + larg_coupe / 2, 0), (x_c - larg_coupe / 2, 0)], PIECE_A, hachure="A")
    cv.polygone([(x_c - tB / 2, 0), (x_c + tB / 2, 0), (x_c + tB / 2, hB), (x_c - tB / 2, hB)], PIECE_B, hachure="B")
    _triangle_cordon(cv, (x_c + tB / 2, 0), (1, 0), (0, 1), a)
    if len(res.get("cordons", [])) > 1 or any(c["nom"].startswith("cordon 2") for c in res.get("cordons", []) + res.get("cordons_exclus", [])):
        _triangle_cordon(cv, (x_c - tB / 2, 0), (-1, 0), (0, 1), a)
    cv.axe((x_c, -tA - 12), (x_c, hB + 12))
    cv.texte(x_c, -tA - 24, "coupe A-A", 10, MUT, "middle")
    cv.texte(x_c + tB / 2, hB * 0.62, f"B · t {_fmt(tB, 0)}", 9.5, MUT, "start", dx=8, dy=3.5, boite=True)
    cv.texte(x_c + larg_coupe / 2, -tA / 2, f"A · t {_fmt(tA, 0)}", 9.5, MUT, "end", dx=-6, dy=3.5, boite=True)
    e = res["efforts"]
    if abs(e.N) > 0:
        s = 1 if e.N > 0 else -1
        cv.fleche((x_c, hB + 3 if s > 0 else hB + 48), (x_c, hB + 48 if s > 0 else hB + 3), f"N = {_val_kN(e.N)}")
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((x_c + s * (tB / 2 + 2), hB * 0.8), (x_c + s * (tB / 2 + 50), hB * 0.8), f"V⊥ = {_val_kN(e.Vy)}")
    if abs(e.Mx) > 0:
        cv.moment((x_c, hB * 0.42), 20, f"M = {_val_kNm(e.Mx)}", +1 if e.Mx > 0 else -1, pos="gauche")
    # --- plan à droite : A vu de dessus, l'empreinte de B, les cordons des deux côtés
    y0 = hB * 0.45
    hA = min(hB * 0.6, 90.0)
    cv.polygone([(0, y0 - hA / 2), (l, y0 - hA / 2), (l, y0 + hA / 2), (0, y0 + hA / 2)], PIECE_A)
    cv.polygone([(0, y0 - tB / 2), (l, y0 - tB / 2), (l, y0 + tB / 2), (0, y0 + tB / 2)], PIECE_B)
    deplace = lambda c: dict(c, p1=[c["p1"][0], c["p1"][1] + y0], p2=[c["p2"][0], c["p2"][1] + y0])  # noqa: E731
    _cordons(cv, res, 6.0, deplace)
    _etiquette_gouvernant(cv, res, lambda p: (p[0], p[1] + y0))
    # trait de coupe A-A sur le plan
    xa = l * 0.28
    cv.axe((xa, y0 - hA / 2 - 26), (xa, y0 + hA / 2 + 22))
    for yy, dy in ((y0 - hA / 2 - 26, -1), (y0 + hA / 2 + 22, 1)):
        cv.fleche((xa, yy), (xa - 14, yy), "", COTE, 1.2)
        cv.texte(xa - 20, yy, "A", 9.5, COTE, "end", True, dy=3.5)
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.axe((l, y0), (l + 12, y0))
        cv.fleche((l + 12 if s > 0 else l + 62, y0), (l + 62 if s > 0 else l + 12, y0), f"V∥ = {_val_kN(e.Vx)}")
    cv.cote((0, y0 - hA / 2), (l, y0 - hA / 2), f"l = {_fmt(l, 0)}", 18)
    cv.texte(l / 2, y0 + hA / 2 + 30, "vue en plan", 10, MUT, "middle")
    cv.cartouche("sections hachurées · cordons colorés par leur taux · ◉ point gouvernant")
    return cv.svg()


# ====================================================================== C3 console (élévation)
def vue_elevation_console(res: Dict, W: int = 640, H: int = 400) -> str:
    pc = res["pieces"]
    h, t, ex, Lc = pc["h"], pc["t"], pc["e"], pc["longueur"]
    a = res["cordons"][0]["a"] if res.get("cordons") else 5.0
    bp = 90.0                                            # largeur dessinée du poteau
    cv = Canevas(W, H, -bp - 16, Lc + 110, -h * 0.72 - 50, h / 2 + 95, marge=34)
    cv.titre("C3 · CONSOLE FLÉCHIE — élévation", f"cordons {_gorges(res)} · cotes en mm")
    cv.polygone([(-bp, -h * 0.72 - 30), (0, -h * 0.72 - 30), (0, h / 2 + 70), (-bp, h / 2 + 70)], PIECE_A)
    cv.axe((-bp / 2, -h * 0.72 - 36), (-bp / 2, h / 2 + 76))
    cv.texte(-bp / 2, -h * 0.72 - 22, "poteau (face)", 9.5, MUT, "middle")
    cv.polygone([(0, h / 2), (Lc, h / 2), (Lc, -h / 2), (0, -h / 2)], PIECE_B)
    cv.texte(Lc * 0.62, -h * 0.36, f"plat t = {_fmt(t, 0)}", 10, MUT, "middle")
    # les cordons sont vus par la tranche sur la face : tous en x = 0 (le dessin les superpose)
    sur_face = lambda c: dict(c, p1=[0.0, c["p1"][1]], p2=[0.0, c["p2"][1]])  # noqa: E731
    _cordons(cv, res, 8.0, sur_face)
    _etiquette_gouvernant(cv, res, lambda p: (0.0, p[1]), u=(-1.0, 0.0))
    n_c = len({c["chaine"] for c in res.get("cordons", [])})
    cv.texte(0, 0, f"{_gorges(res)} · {n_c} cordon{'s' if n_c > 1 else ''}", 9.5, "#8A5A00", "end", dx=-10, dy=3.5, boite=True)
    e = res["efforts"]
    V = -e.Vy
    if abs(V) > 0:
        if V > 0:
            cv.fleche((ex, h / 2 + 60), (ex, h / 2 + 3), f"V = {_val_kN(V)}")
        else:
            cv.fleche((ex, h / 2 + 3), (ex, h / 2 + 60), f"V = {_val_kN(V)}")
        cv.axe((ex, h / 2 - 14), (ex, h / 2 + 64))
        cv.moment((Lc * 0.28, 0.0), 22, f"M = V·e = {_val_kNm(abs(V) * ex)}", -1 if V > 0 else 1, pos="bas")
    if abs(e.N) > 0:
        s = 1 if e.N > 0 else -1
        cv.axe((Lc, 0), (Lc + 30, 0))
        cv.fleche((Lc + 30, 0), (Lc + 30 + s * 56, 0), f"N = {_val_kN(e.N)}")
    cv.cote((0, h / 2 + 30), (ex, h / 2 + 30), f"e = {_fmt(ex, 0)}", 16)
    cv.cote((Lc, -h / 2), (Lc, h / 2), f"h = {_fmt(h, 0)}", -18, pos=0.3)
    # mini vue de dessus (×2,5) : le plat perpendiculaire à la face, les deux cordons dans leur dièdre
    k = 2.5
    y_v = -h * 0.72 - 8
    x_v = Lc * 0.32
    tf = 10.0 * k
    cv.polygone([(x_v - 26 * k, y_v - tf), (x_v + 26 * k, y_v - tf), (x_v + 26 * k, y_v), (x_v - 26 * k, y_v)], PIECE_A, hachure="A")
    cv.polygone([(x_v - t * k / 2, y_v), (x_v + t * k / 2, y_v), (x_v + t * k / 2, y_v + 16 * k), (x_v - t * k / 2, y_v + 16 * k)], PIECE_B, hachure="B")
    _triangle_cordon(cv, (x_v + t * k / 2, y_v), (1, 0), (0, 1), a * k)
    if n_c > 1 or any("gauche" in c["nom"] for c in res.get("cordons", []) + res.get("cordons_exclus", [])):
        _triangle_cordon(cv, (x_v - t * k / 2, y_v), (-1, 0), (0, 1), a * k)
    cv.texte(x_v + 30 * k, y_v + 8 * k, "vue de dessus ×2,5 : le plat et ses cordons sur la face", 9.5, MUT, "start")
    cv.cartouche("cordons colorés par leur taux · ◉ point gouvernant · trait mixte : ligne d'action")
    return cv.svg()


# ====================================================================== C4 gousset (plan)
def vue_plan_gousset(res: Dict, W: int = 640, H: int = 400) -> str:
    pc = res["pieces"]
    A, B = pc["A"], pc["B"]
    xs = [x for x, _ in A] + [x for x, _ in B]
    ys = [y for _, y in A] + [y for _, y in B]
    cv = Canevas(W, H, min(xs) - 40, max(xs) + 80, min(ys) - 70, max(ys) + 30, marge=34)
    cv.titre("C4 · GOUSSET EN TORSION — vue de face (plat posé sur l'appui)", f"cordons {_gorges(res)} · cotes en mm")
    cv.polygone(A, PIECE_A)
    cv.texte(min(x for x, _ in A) + 6, max(y for _, y in A) - 12, "appui", 9.5, MUT, "start")
    cv.polygone(B, PIECE_B)
    cv.texte(pc["bx"] / 2, pc["by"] - 16, f"plat t = {_fmt(pc['t'], 0)}, posé à plat", 10, MUT, "middle")
    _cordons(cv, res, 7.0)
    _etiquette_gouvernant(cv, res)
    e = res["efforts"]
    xV, yV = pc["xV"], pc["yV"]
    cv.point_R(xV, yV, "")
    V = -e.Vy
    if abs(V) > 0:
        s = -1 if V > 0 else 1
        cv.fleche((xV, yV + s * 4), (xV, yV + s * 62), f"V = {_val_kN(V)}")
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((xV + s * 4, yV), (xV + s * 62, yV), f"H = {_val_kN(e.Vx)}")
    if abs(e.N) > 0:
        cv.symbole_normal((xV, yV), f"N = {_val_kN(e.N)}", e.N > 0, cote=+1 if e.Vx <= 0 else -1)
    pr = res.get("proprietes") or {}
    yb = min(y for _, y in A)
    cv.cote((0, yb), (pc["bx"], yb), f"b_x = {_fmt(pc['bx'], 0)}", 16)
    cv.cote((0, 0), (0, pc["by"]), f"b_y = {_fmt(pc['by'], 0)}", -(abs(min(x for x, _ in A)) * cv.ech + 16))
    if "xG" in pr:
        cv.marque_G(pr["xG"], pr["yG"])
        cv.cote((pr["xG"], yb), (xV, yb), f"e = {_fmt(xV - pr['xG'], 0)}", 40)
    cv.repere_xy()
    cv.cartouche("G centre des gorges · R point d'application · cordons colorés par leur taux · ◉ point gouvernant")
    return cv.svg()


# ====================================================================== C5 / C6 platine (face + élévation)
def _section_profil(cv: Canevas, rec: Dict, pc: Dict):
    """La section du profilé (coupée : hachurée), ou l'anneau du tube."""
    from acier.profiles import familles as F
    try:
        rings = F.contours(rec["famille"], rec["dims"])
        rings = [[(float(x), float(y)) for x, y in r] for r in rings]
    except Exception:  # noqa: BLE001 — famille sans contour : rectangle d'encombrement
        h, b = rec.get("h", rec.get("d", 100.0)), rec.get("b", rec.get("d", 100.0))
        rings = [[(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]]
    if len(rings) > 1:
        cv.anneau(rings[0], rings[1:], PIECE_B, hachure="B")
    else:
        cv.polygone(rings[0], PIECE_B, hachure="B")


def vue_face_profil(res: Dict, W: int = 640, H: int = 400) -> str:
    pc = res["pieces"]
    pl = pc["platine"]
    rec = pc["profil"]
    xs, ys = [x for x, _ in pl], [y for _, y in pl]
    hp = max(ys) - min(ys)
    larg_face = max(xs) - min(xs)
    # repère commun : la face à gauche, l'élévation à droite (décalée de x_el)
    x_el = max(xs) + 190
    l_el = max(0.75 * hp, 110.0)
    cv = Canevas(W, H, min(xs) - 90, x_el + l_el + 80, min(ys) - 88, max(ys) + 38, marge=34)
    tube = "contour" in pc
    cv.titre(("C6 · TUBE SUR PLATINE" if tube else "C5 · PLATINE D'ABOUT DE PROFILÉ") + " — vue de face et élévation",
             f"{rec['nom']} · cordons {_gorges(res)} · cotes en mm")
    cv.polygone(pl, PIECE_A)
    cv.texte(min(xs) + 6, max(ys) - 12, "platine", 9.5, MUT, "start")
    _section_profil(cv, rec, pc)
    cv.axe((min(xs) - 10, 0), (max(xs) + 10, 0))
    cv.axe((0, min(ys) - 10), (0, max(ys) + 10))
    _cordons(cv, res, 5.5)
    _etiquette_gouvernant(cv, res)
    if tube:
        _etiquette_gorge(cv, res, res["cordons"][0]["nom"] if res.get("cordons") else "", f"a{_fmt(res['cordons'][0]['a'], 0)}" if res.get("cordons") else "")
    else:
        sup = next((c["nom"] for c in res.get("cordons", []) if c["nom"].startswith("semelle sup. ext")), None)
        ame = next((c["nom"] for c in res.get("cordons", []) if c["nom"].startswith("âme droite")), None)
        if sup:
            _etiquette_gorge(cv, res, sup, f"a_f = {_fmt(next(c['a'] for c in res['cordons'] if c['nom'] == sup), 0)}", decal_px=12)
        if ame:
            _etiquette_gorge(cv, res, ame, f"a_w = {_fmt(next(c['a'] for c in res['cordons'] if c['nom'] == ame), 0)}", decal_px=12)
    e = res["efforts"]
    h_pr = rec.get("h", rec.get("d", hp * 0.6))
    # efforts dans la face : V_z (vertical), V_y (horizontal), N (⊙/⊗), T (courbe), M_y et M_z (vecteurs)
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((max(xs) + 30, -s * 28), (max(xs) + 30, s * 28), f"V_z = {_val_kN(e.Vy)}")
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((-s * 30, min(ys) - 26), (s * 30, min(ys) - 26), f"V_y = {_val_kN(e.Vx)}", cote=-1)
    if abs(e.N) > 0:
        cv.symbole_normal((0, 0), f"N = {_val_kN(e.N)}", e.N > 0)
    if abs(e.Mz) > 0:
        cv.moment((0, 0), max(22.0, min(0.3 * h_pr * cv.ech, 42)), f"T = {_val_kNm(e.Mz)}", 1 if e.Mz > 0 else -1, pos="bas")
    if abs(e.Mx) > 0:
        s = 1 if e.Mx > 0 else -1
        cv.vecteur_moment((min(xs) - 42, 0), (s, 0), f"M_y = {_val_kNm(e.Mx)}", 44, cote=+1)
        cv.texte(min(xs) - 42, 0, "tend " + ("le haut" if s > 0 else "le bas"), 9.5, EFFORT, "middle", dy=16, boite=True)
    if abs(e.My) > 0:
        s = 1 if e.My > 0 else -1
        cv.vecteur_moment((0, max(ys) + 22), (0, s), f"M_z = {_val_kNm(e.My)}", 40, cote=+1)
        cv.texte(0, max(ys) + 22, "tend " + ("la gauche" if s > 0 else "la droite"), 9.5, EFFORT, "start", dx=14 + _larg(f"M_z = {_val_kNm(e.My)}", 11.5), dy=4, boite=True)
    cv.texte((min(xs) + max(xs)) / 2, min(ys) - 78, "vue de face (le profilé est coupé)", 10, MUT, "middle")
    # --- élévation à droite : la platine coupée (debout), le profilé qui en sort, les cordons haut et bas, N, V_z, M_y
    tp = 16.0
    cv.polygone([(x_el - tp, min(ys) + 8), (x_el, min(ys) + 8), (x_el, max(ys) - 8), (x_el - tp, max(ys) - 8)], PIECE_A, hachure="A")
    cv.polygone([(x_el, -h_pr / 2), (x_el + l_el, -h_pr / 2), (x_el + l_el, h_pr / 2), (x_el, h_pr / 2)], PIECE_B)
    if not tube and "tf" in rec:
        cv.ligne((x_el, h_pr / 2 - rec["tf"]), (x_el + l_el, h_pr / 2 - rec["tf"]), TRAIT, 0.8)
        cv.ligne((x_el, -h_pr / 2 + rec["tf"]), (x_el + l_el, -h_pr / 2 + rec["tf"]), TRAIT, 0.8)
    a_h = res["cordons"][0]["a"] if res.get("cordons") else 5.0
    _triangle_cordon(cv, (x_el, h_pr / 2), (0, 1), (1, 0), a_h)
    _triangle_cordon(cv, (x_el, -h_pr / 2), (0, -1), (1, 0), a_h)
    cv.axe((x_el - tp - 8, 0), (x_el + l_el + 8, 0))
    xe = x_el + l_el
    if abs(e.N) > 0:
        s = 1 if e.N > 0 else -1
        cv.fleche((xe + (4 if s > 0 else 52), 0), (xe + (52 if s > 0 else 4), 0), f"N = {_val_kN(e.N)}", cote=-1)
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((xe - 14, s * (h_pr / 2 + 4)), (xe - 14, s * (h_pr / 2 + 46)), f"V_z = {_val_kN(e.Vy)}", cote=-1)
    if abs(e.Mx) > 0:
        cv.moment((x_el + l_el * 0.5, 0), 20, f"M_y = {_val_kNm(e.Mx)}", 1 if e.Mx > 0 else -1, pos="haut")
    cv.texte(x_el + l_el / 2, min(ys) - 78, "élévation", 10, MUT, "middle")
    cv.cartouche("sections hachurées · cordons colorés par leur taux · ◉ point gouvernant · ⊙ vers soi · ⊗ vers le fond")
    return cv.svg()


# ====================================================================== C7 bout à bout (plan + coupe)
def vue_plan_bout_a_bout(res: Dict, W: int = 640, H: int = 400) -> str:
    pc = res["pieces"]
    b, t = pc["b"], pc["t"]
    cv = Canevas(W, H, -b - 100, b + 100, -b / 2 - 150, b / 2 + 40, marge=34)
    cv.titre("C7 · BOUT À BOUT — vue en plan et coupe", f"plats {_fmt(b, 0)} × {_fmt(t, 0)} · cotes en mm")
    cv.polygone(pc["A"], PIECE_A)
    cv.polygone(pc["B"], PIECE_B)
    cv.texte(-b / 2, -b * 0.35, "plat A", 10, MUT)
    cv.texte(b / 2, -b * 0.35, "plat B", 10, MUT)
    cv.axe((-b - 36, 0), (b + 36, 0))
    if pc["pleine"]:
        w = max(t, 4.0)
        cv.polygone([(-w / 2, -b / 2), (w / 2, -b / 2), (w / 2, b / 2), (-w / 2, b / 2)], CORDON, TRAIT, 0.8, hachure="W")
        cv.texte(0, b / 2, "pleine pénétration : la pièce gouverne", 10, ENCRE, "middle", dy=-10, boite=True)
    else:
        _cordons(cv, res, 7.0)
        _etiquette_gouvernant(cv, res, u=(1.0, 0.0))
    e = res["efforts"]
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((b + 30, 0), (b + 30 + s * 60, 0), f"N = {_val_kN(e.Vx)}")
        cv.fleche((-b - 30, 0), (-b - 30 - s * 60, 0), f"N = {_val_kN(e.Vx)}")
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((b * 0.5, s * (b / 2 + 3)), (b * 0.5, s * (b / 2 + 48)), f"V = {_val_kN(e.Vy)}")
    if abs(e.Mz) > 0:
        cv.moment((b * 0.5, 0), 20, f"M = {_val_kNm(e.Mz)}", 1 if e.Mz > 0 else -1, pos="bas")
    cv.cote((-b, -b / 2), (-b, b / 2), f"b = {_fmt(b, 0)}", 18, pos=0.25)
    # coupe sous le plan : l'épaisseur t, la soudure (V pleine ou deux pénétrations partielles p)
    yc = -b / 2 - 95
    ech_c = 3.0
    tt = t * ech_c
    cv.polygone([(-b * 0.7, yc), (-tt * 0.6, yc), (-tt * 0.6 - tt * 0.5, yc + tt), (-b * 0.7, yc + tt)], PIECE_A, hachure="A") if pc["pleine"] else \
        cv.polygone([(-b * 0.7, yc), (-1.5, yc), (-1.5, yc + tt), (-b * 0.7, yc + tt)], PIECE_A, hachure="A")
    cv.polygone([(tt * 0.6, yc), (b * 0.7, yc), (b * 0.7, yc + tt), (tt * 0.6 + tt * 0.5, yc + tt)], PIECE_B, hachure="B") if pc["pleine"] else \
        cv.polygone([(1.5, yc), (b * 0.7, yc), (b * 0.7, yc + tt), (1.5, yc + tt)], PIECE_B, hachure="B")
    if pc["pleine"]:
        cv.polygone([(-tt * 0.6, yc), (tt * 0.6, yc), (tt * 0.6 + tt * 0.5, yc + tt), (-tt * 0.6 - tt * 0.5, yc + tt)], CORDON, TRAIT, 0.8, hachure="W")
        cv.texte(0, yc + tt, "soudure en V, pleine pénétration", 9.5, MUT, "middle", dy=-6 - tt * 0, boite=True) if False else None
    else:
        p = res["cordons"][0]["a"] if res.get("cordons") else float(t) / 2
        pp = p * ech_c
        deux = len(res.get("cordons", [])) > 1
        cv.polygone([(-pp * 0.9, yc + tt), (pp * 0.9, yc + tt), (0, yc + tt - pp)], CORDON, TRAIT, 0.8, hachure="W")
        if deux:
            cv.polygone([(-pp * 0.9, yc), (pp * 0.9, yc), (0, yc + pp)], CORDON, TRAIT, 0.8, hachure="W")
        cv.cote((0, yc + tt - pp), (0, yc + tt), f"p = {_fmt(p, 0)}", b * 0.7 * cv.ech + 46, taille=9)
    cv.cote((b * 0.7, yc), (b * 0.7, yc + tt), f"t = {_fmt(t, 0)}", 16, taille=9)
    cv.texte(0, yc - 16, "coupe (épaisseur ×3)", 9.5, MUT, "middle")
    cv.cartouche("sections hachurées · ◉ point gouvernant · trait mixte : ligne d'action de N")
    return cv.svg()


# ====================================================================== C8 cordons libres (plan)
def vue_plan_libre(res: Dict, W: int = 640, H: int = 400) -> str:
    cs = res.get("cordons", []) + res.get("cordons_exclus", [])
    e = res["efforts"]
    xs = [c["p1"][0] for c in cs] + [c["p2"][0] for c in cs] + [e.reference[0]]
    ys = [c["p1"][1] for c in cs] + [c["p2"][1] for c in cs] + [e.reference[1]]
    if not xs:
        xs, ys = [0.0], [0.0]
    m = max(60.0, 0.3 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    cv = Canevas(W, H, min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m * 0.8, marge=34)
    cv.titre("C8 · CORDONS LIBRES — vue en plan (plan de joint)", f"cordons {_gorges(res)} · coordonnées en mm")
    _cordons(cv, res, 7.0)
    # le nom de chaque chaîne au milieu de son segment médian, hors de la pièce (u_A)
    for nom in list(dict.fromkeys((c.get("chaine") or c["nom"]) for c in cs)):
        (mx, my), ua = _milieu_chaine(res, nom)
        a_ch = next(c["a"] for c in cs if (c.get("chaine") or c["nom"]) == nom)
        n = math.hypot(ua[0], ua[1]) or 1.0
        dx, dy = ua[0] / n * 15, -ua[1] / n * 15
        cv.texte(mx, my, f"{nom} · a{_fmt(a_ch, 0)}", 9.5, "#8A5A00", "start" if dx > 3 else ("end" if dx < -3 else "middle"), dx=dx, dy=dy + 3.5, boite=True)
    _etiquette_gouvernant(cv, res)
    pr = res.get("proprietes") or {}
    if "xG" in pr:
        cv.marque_G(pr["xG"], pr["yG"])
    rx, ry = e.reference[0], e.reference[1]
    cv.point_R(rx, ry)
    px = 1.0 / cv.ech                                   # 1 px en mm
    if abs(e.Vx) > 0:
        s = 1 if e.Vx > 0 else -1
        cv.fleche((rx + s * 5 * px, ry), (rx + s * 60 * px, ry), f"V_x = {_val_kN(e.Vx)}", cote=+1)
    if abs(e.Vy) > 0:
        s = 1 if e.Vy > 0 else -1
        cv.fleche((rx, ry + s * 5 * px), (rx, ry + s * 60 * px), f"V_y = {_val_kN(e.Vy)}")
    if abs(e.N) > 0:
        cv.symbole_normal((rx, ry), "", e.N > 0)
        cv.texte(rx, ry, f"N = {_val_kN(e.N)} {'⊙' if e.N > 0 else '⊗'}", 11.5, EFFORT, "end", True, dx=-12, dy=-8, boite=True)
    if abs(e.Mx) > 0:
        s = 1 if e.Mx > 0 else -1
        cv.vecteur_moment((rx, ry + 44 * px), (s, 0), f"M_x = {_val_kNm(e.Mx)}", 40, cote=+1)
    if abs(e.My) > 0:
        s = 1 if e.My > 0 else -1
        cv.vecteur_moment((rx - 44 * px, ry), (0, s), f"M_y = {_val_kNm(e.My)}", 40, cote=-1)
    if abs(e.Mz) > 0:
        cv.moment((rx, ry), 24, "", 1 if e.Mz > 0 else -1)
        cv.texte(rx, ry, f"M_z = {_val_kNm(e.Mz)} ↻" if e.Mz < 0 else f"M_z = {_val_kNm(e.Mz)} ↺", 11.5, EFFORT, "end", True, dx=-22, dy=34, boite=True)
    cv.repere_xy()
    cv.cartouche("G centre des gorges · R point de référence · ⇒ vecteur moment (règle de la main droite) · ◉ point gouvernant")
    return cv.svg()


VUES = {"C1": vue_plan_recouvrement, "C2": vue_elevation_T, "C3": vue_elevation_console, "C4": vue_plan_gousset,
        "C5": vue_face_profil, "C6": vue_face_profil, "C7": vue_plan_bout_a_bout, "C8": vue_plan_libre}


def vue_principale(res: Dict, W: int = 640, H: int = 400) -> str:
    if res.get("statut") == "refus" or "efforts" not in res:
        cv = Canevas(W, H, 0, 100, 0, 60)
        cv.titre("CALCUL IMPOSSIBLE")
        cv.texte(50, 30, "Voir les messages : " + (res.get("messages") or ["—"])[0][:90], 11, NOK)
        return cv.svg()
    return VUES[res["config"]](res, W, H)


# ====================================================================== encart « la gorge »
def encart_gorge(res: Dict, W: int = 320, H: int = 330) -> str:
    """La section du cordon gouvernant : les deux pièces coupées (hachurées),
    le triangle du cordon, le plan de gorge a (interrompu), σ⊥ normale à ce
    plan, τ⊥ dans ce plan, τ∥ le long du cordon (⊙ vers soi / ⊗), puis les
    valeurs et le critère de la méthode choisie."""
    g = res.get("gouvernant")
    cv = Canevas(W, H, 0, 100, 0, 100, marge=8)
    cv.titre("LA GORGE AU POINT GOUVERNANT")
    if not g or g.get("cordon") == "pièce":
        cv.texte(50, 55, "pas de cordon gouvernant" if not g else "pleine pénétration : la pièce gouverne", 10.5, ENCRE)
        if g:
            cv.texte(50, 42, f"σ = {_fmt(g['sigma_perp'])} · τ = {_fmt(g['tau_par'])} N/mm² · η = {g['taux'] * 100:.0f} %", 10, MUT)
        return cv.svg()
    a = g["a"]
    cord = next((c for c in res.get("cordons", []) if c["nom"] == g["cordon"]), {})
    bout = bool(cord.get("bout_a_bout"))
    nom_ch = cord.get("chaine") or g["cordon"]
    sp, tp, tpar = g["sigma_perp"], g["tau_perp"], g["tau_par"]
    if bout:
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
    if bout:
        lignes.append((f"{nom_ch} · {g['position']} · η = {g['taux'] * 100:.0f} %", 10, couleur_taux(g["taux"])))
    for i, (t, taille, coul) in enumerate(lignes):
        cv.texte(50, 24 - i * 6.3, t, taille, coul, "middle", gras=(i == len(lignes) - 1 and bout))
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
    return ("Pièces au trait fort, sections hachurées · cordons colorés par leur taux (vert ≤ 70 %, ocre jusqu'à 100 %, rouge au-delà), "
            "pointillé gris : cordon qui ne compte pas · ◉ point gouvernant · flèches rouges : efforts dans le sens saisi "
            "(⊙ vers soi, ⊗ vers le fond, ⇒ vecteur moment, flèche courbe : moment dans le plan) · cotes en mm · "
            "encart : les composantes sur la gorge, EN 1993-1-8 fig. 4.5")

# -*- coding: utf-8 -*-
"""L'ENROBAGE dessiné à l'échelle (mm) à partir du résultat du moteur ``beton/enrobage`` : la barre principale, l'étrier
(ou la barre extérieure), l'enrobage nominal retenu c_nom, l'enrobage minimal c_min et la bande de tolérance Δc_dev,
la distance à l'axe a de la barre, comparée à celle que demande le tableau feu.

- poutre, poteau : un DÉTAIL D'ANGLE de la coupe (deux parements, l'étrier qui contourne la barre d'angle) ;
- dalle, voile : une COUPE de la face exposée (une dalle vue en coupe, un voile en coupe horizontale), les barres
  principales en rangée.

c_nom se pose au nu de l'armature la plus extérieure (l'étrier), comme la vérification au feu de l'écran Streamlit
(c_feu,min = a − Ø/2 − Ø_étrier) ; la barre d'angle est logée dans le coude de l'étrier, d'où a = c_nom + Ø_étrier +
Ø/2. L'espacement des barres dessinées est une convention de dessin (pas une donnée du calcul) et n'est pas coté.

Le style est celui de ``api_web/dessin_poutre.py`` (traits noirs, béton gris clair, acier bleu-gris, cotes fines,
exigence au feu en rouge).
"""
from html import escape
from typing import List

from .dessin_poutre import ACIER, BETON, COTE, EFFORT, ENCRE, POLICE

TOLERANCE = "#2F6FDB"          # la bande Δc_dev (bleu de la charte, transparent)


def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _mm(x) -> str:
    """Une cote en mm à la française : 30, 18,5."""
    return _f(float(x)).replace(".", ",")


class Toile:
    """Les traits d'un dessin, dans une unité de texte ``u`` (les tailles de police et de trait en dépendent)."""

    def __init__(self, u: float):
        self.u, self.g = u, []

    def txt(self, x, y, t, k=1.0, ancre="middle", couleur=ENCRE, gras=False, rot=None, italique=False):
        poids = ' font-weight="600"' if gras else ""
        style = ' font-style="italic"' if italique else ""
        r = f' transform="rotate({_f(rot)} {_f(x)} {_f(y)})"' if rot is not None else ""
        self.g.append(f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(self.u * k)}" fill="{couleur}" '
                      f'font-family="{POLICE}"{poids}{style}{r}>{t}</text>')

    def ligne(self, x1, y1, x2, y2, couleur=COTE, k=1.0, tirets=""):
        d = f' stroke-dasharray="{tirets}"' if tirets else ""
        self.g.append(f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" '
                      f'stroke-width="{_f(self.u * 0.085 * k)}"{d}/>')

    def largeur(self, texte: str, k=1.0) -> float:
        """La largeur estimée d'un texte (police proportionnelle, chiffres tabulaires)."""
        import re
        return len(re.sub(r"<[^>]+>", "", texte)) * 0.56 * self.u * k

    def tient(self, x1, x2, texte, k=0.92) -> bool:
        """Le texte d'une cote tient-il entre ses deux traits ?"""
        return self.largeur(texte, k) + self.u < abs(x2 - x1)

    def cote_h(self, x1, x2, y, texte, k=0.92, couleur=ENCRE, dehors="droite"):
        """Une cote horizontale ; le texte au-dessus s'il tient entre les traits, sinon prolongé hors des traits : à
        droite du second, ou à gauche du premier (``dehors="gauche"`` : quand une ligne d'attache passe à droite)."""
        t = self.u * 0.42
        self.ligne(x1, y, x2, y)
        for x in (x1, x2):
            self.ligne(x - t, y + t, x + t, y - t, ENCRE, 1.4)
        if self.tient(x1, x2, texte, k):
            self.txt((x1 + x2) / 2, y - self.u * 0.45, texte, k, couleur=couleur)
        elif dehors == "gauche":
            self.ligne(x1, y, x1 - self.u * 0.9, y)
            self.txt(x1 - self.u * 1.1, y + self.u * 0.33, texte, k, "end", couleur=couleur)
        else:
            self.ligne(x2, y, x2 + self.u * 0.9, y)
            self.txt(x2 + self.u * 1.1, y + self.u * 0.33, texte, k, "start", couleur=couleur)

    def cote_v(self, x, y1, y2, texte, k=0.92, couleur=ENCRE):
        """Une cote verticale (y1 < y2) ; le texte tourné le long de la ligne, ou au-dessus de y1 s'il ne tient pas."""
        t = self.u * 0.42
        self.ligne(x, y1, x, y2)
        for y in (y1, y2):
            self.ligne(x - t, y + t, x + t, y - t, ENCRE, 1.4)
        if texte is None:
            return
        w = self.largeur(texte, k)
        if w + self.u < abs(y2 - y1):
            self.txt(x - self.u * 0.45, (y1 + y2) / 2, texte, k, couleur=couleur, rot=-90)
        else:
            self.ligne(x, y1, x, y1 - self.u * 0.9)
            self.txt(x + self.u * 0.33, y1 - self.u * 1.1, texte, k, "start", couleur=couleur, rot=-90)

    def rupture(self, x1, y1, x2, y2):
        """Une ligne de rupture (le béton coupé) : droite, avec un « Z » au milieu."""
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lx, ly = x2 - x1, y2 - y1
        n = (lx * lx + ly * ly) ** 0.5
        ux, uy = lx / n, ly / n
        px, py = -uy, ux
        z = self.u * 0.7
        pts = [(x1, y1), (mx - ux * z, my - uy * z), (mx - ux * z * 0.35 + px * z, my - uy * z * 0.35 + py * z),
               (mx + ux * z * 0.35 - px * z, my + uy * z * 0.35 - py * z), (mx + ux * z, my + uy * z), (x2, y2)]
        self.g.append(f'<polyline points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in pts)}" fill="none" stroke="{ENCRE}" '
                      f'stroke-width="{_f(self.u * 0.08)}"/>')

    def renvoi(self, x0, y0, x1, y1, texte, k=0.95, couleur=ENCRE):
        """Un renvoi : un point sur l'objet, un trait, le texte en marge."""
        self.g.append(f'<circle cx="{_f(x0)}" cy="{_f(y0)}" r="{_f(self.u * 0.16)}" fill="{couleur}"/>')
        self.ligne(x0, y0, x1, y1, couleur, 1.0)
        self.ligne(x1, y1, x1 + self.u * 0.6, y1, couleur, 1.0)
        self.txt(x1 + self.u * 0.85, y1 + self.u * 0.33, texte, k, "start", couleur=couleur)


def _indice(base: str, ind: str, suite: str, u: float) -> str:
    """Un symbole à indice dans un texte SVG : c_nom = 30 → c<tspan…>nom</tspan><tspan…> = 30</tspan>."""
    return (f"{base}<tspan font-size='70%' dy='{_f(u * 0.25)}'>{ind}</tspan>"
            f"<tspan dy='{_f(-u * 0.25)}'>{suite}</tspan>")


def _renvois(T: Toile, renvois: List[tuple], xm: float, bas: float) -> float:
    """Les textes en marge droite, espacés d'au moins 1,55 u dans l'ordre vertical des points visés, le dernier pas
    plus bas que ``bas`` ; rend la plus haute ordonnée écrite (la boîte en tient compte)."""
    u = T.u
    renvois.sort(key=lambda r: r[0][1])
    ys = []
    for (px, py), _, _ in renvois:
        ys.append(py if not ys else max(py, ys[-1] + 1.55 * u))
    deb = min(0.0, bas - ys[-1])
    ys = [y + deb for y in ys]
    for ((px, py), t, coul), y in zip(renvois, ys):
        T.renvoi(px, py, xm, y, t, couleur=coul)
    return ys[0]


def _boite(ident: str, T: Toile, x0: float, y0: float, x1: float, y1: float, titre: str, c: float) -> str:
    vb = f"{_f(x0)} {_f(y0)} {_f(x1 - x0)} {_f(y1 - y0)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="{escape(titre)} : enrobage nominal {_mm(c)} mm">{"".join(T.g)}</svg>')


def _angle(ident: str, res: dict) -> str:
    """Poutre, poteau : le détail d'angle de la coupe (deux parements, l'étrier qui contourne la barre d'angle)."""
    el, f = res["element"], res["feu"]
    d, e, c, dd = float(el["diam_barre"]), float(el["diam_etrier"]), float(res["c_nom"]), float(el["delta_dev"])
    c_min = c - dd
    pas = d + max(d, 20.0)                                       # convention de dessin : l'entraxe des barres dessinées
    xc0 = c + e + d / 2                                          # l'axe de la barre d'angle
    W = max(120.0, xc0 + pas + d / 2 + max(40.0, d))
    H = W
    u = 1.2 * H / 23
    T = Toile(u)
    g = T.g
    sw = u * 0.085
    tir = f"{_f(u * 0.5)} {_f(u * 0.3)}"
    fin = f"{_f(u * 0.3)} {_f(u * 0.25)}"
    # ---- le béton, la bande de tolérance Δc_dev (entre c_min et c_nom), la ligne de c_min en tirets, les parements
    g.append(f'<rect x="0" y="0" width="{_f(W)}" height="{_f(H)}" fill="{BETON}"/>')
    if dd > 0:
        g.append(f'<path d="M{_f(c_min)} 0 V{_f(H - c_min)} H{_f(W)} V{_f(H - c)} H{_f(c)} V0 Z" fill="{TOLERANCE}" fill-opacity=".14"/>')
    g.append(f'<path d="M{_f(c_min)} 0 V{_f(H - c_min)} H{_f(W)}" fill="none" stroke="{TOLERANCE}" stroke-width="{_f(sw * 1.1)}" '
             f'stroke-dasharray="{tir}"/>')
    g.append(f'<path d="M0 0 V{_f(H)} H{_f(W)}" fill="none" stroke="{ENCRE}" stroke-width="{_f(sw * 2.6)}" stroke-linejoin="miter"/>')
    T.rupture(0, 0, W, 0)
    T.rupture(W, 0, W, H)
    # ---- l'étrier (il contourne la barre d'angle), les barres
    yc = H - xc0
    if e > 0:
        r = d / 2 + e / 2
        g.append(f'<path d="M{_f(c + e / 2)} 0 V{_f(yc)} A{_f(r)} {_f(r)} 0 0 0 {_f(xc0)} {_f(yc + r)} H{_f(W)}" fill="none" '
                 f'stroke="{ACIER}" stroke-width="{_f(e)}" stroke-linecap="butt"/>')
    xs = [xc0, xc0 + pas]
    for x in xs:
        g.append(f'<circle cx="{_f(x)}" cy="{_f(yc)}" r="{_f(d / 2)}" fill="{ENCRE}"/>')
    # ---- les cotes sous le parement : c_nom, puis a (à l'axe de la barre) ; la distance demandée au feu en rouge
    t_c = _indice("c", "nom", f" = {_mm(c)}", u)
    t_a = f"a = {_mm(c)} + {_mm(e)} + {_mm(d / 2)} = {_mm(xc0)}" if e > 0 else f"a = {_mm(c)} + {_mm(d / 2)} = {_mm(xc0)}"
    y1, y2 = H + 2.0 * u, H + 4.3 * u
    T.ligne(0, H + u * 0.4, 0, y2 + u * 0.5)
    T.ligne(c, H - c, c, y1 + u * 0.5, COTE, 0.8, fin)
    T.ligne(xc0, yc, xc0, y2 + u * 0.5, COTE, 0.8, fin)
    # c_nom : au-dessus de sa cote s'il y tient ; sinon À GAUCHE (à droite, la ligne d'attache de l'axe le barrerait)
    c_dedans = T.tient(0, c, t_c)
    T.cote_h(0, c, y1, t_c, dehors="gauche")
    T.cote_h(0, xc0, y2, t_a)
    # la même cote c_nom sur le parement vertical
    xv = -2.0 * u
    T.ligne(-u * 0.4, H - c, xv - u * 0.5, H - c, COTE, 0.8, fin)
    T.ligne(-u * 0.4, H, xv - u * 0.5, H)
    T.cote_v(xv, H - c, H, t_c)
    if f is not None and f.get("a") is not None:      # (poteau R240 : le tableau n'a pas de valeur, rien à repérer)
        ar = float(f["a"])
        t_feu = _indice("a", "feu", f" ≥ {_mm(ar)} · {escape(f['r'])}", u)
        g.append(f'<path d="M{_f(ar)} {_f(y2)} l{_f(-u * 0.38)} {_f(u * 0.75)} h{_f(u * 0.76)} Z" fill="{EFFORT}"/>')
        T.txt(max(ar, T.largeur(t_feu, 0.86) / 2 - 3.2 * u), y2 + u * 1.85, t_feu, 0.86, couleur=EFFORT, gras=True)
    # ---- les renvois, en marge droite
    renvois = [((xs[-1] + d / 2 * 0.7, yc - d / 2 * 0.7), f"barre Ø{_mm(d)}", ENCRE)]
    if e > 0:
        renvois.append(((W - 0.5 * (W - xs[-1] - d / 2), H - c - e / 2), f"étrier Ø{_mm(e)}", ACIER))
    if dd > 0:
        renvois.append(((W - u * 1.2, H - c + dd / 2), _indice("Δc", "dev", f" = {_mm(dd)}", u), TOLERANCE))
    renvois.append(((W - u * 2.4, H - c_min), _indice("c", "min", f" = {_mm(c_min)}", u), TOLERANCE))
    haut = min(-1.2 * u, _renvois(T, renvois, W + 1.8 * u, H + 0.6 * u) - 1.2 * u)
    # ---- la boîte
    gauche = 3.6 * u if c_dedans else max(3.6 * u, T.largeur(t_c, 0.92) + 1.8 * u)
    droite = max(T.largeur(t, 0.95) for _, t, _ in renvois) + 3.4 * u
    bas = 7.6 * u if f is not None else 5.6 * u
    return _boite(ident, T, -gauche, haut, W + droite, H + bas, "Détail d'angle", c)


def _face(ident: str, res: dict) -> str:
    """Dalle, voile : la coupe de la face exposée (une dalle vue en coupe, un voile en coupe horizontale), les barres
    principales en rangée derrière la barre extérieure (Ø « étrier » de l'écran). Le texte est plus grand, rapporté au
    dessin, que dans le détail d'angle : la coupe est large et basse, la toile la réduit à sa largeur."""
    el, f = res["element"], res["feu"]
    d, e, c, dd = float(el["diam_barre"]), float(el["diam_etrier"]), float(res["c_nom"]), float(el["delta_dev"])
    c_min = c - dd
    pas = d + max(d, 20.0)                                       # convention de dessin : l'entraxe des barres dessinées
    xc0 = c + e + d / 2                                          # la distance a (parement → axe des barres)
    x0 = max(30.0, 1.2 * pas)                                    # la première barre dessinée
    W = x0 + 2 * pas + d / 2 + max(30.0, d)
    H = max(0.65 * W, c + e + d + max(50.0, 1.5 * d))
    u = max(W, 1.3 * H) / 13.5
    T = Toile(u)
    g = T.g
    tir = f"{_f(u * 0.5)} {_f(u * 0.3)}"
    fin = f"{_f(u * 0.3)} {_f(u * 0.25)}"
    # ---- le béton : rempli, le parement exposé en trait fort en bas, coupé à gauche et à droite, continu en haut
    g.append(f'<rect x="0" y="0" width="{_f(W)}" height="{_f(H)}" fill="{BETON}"/>')
    if dd > 0:
        g.append(f'<rect x="0" y="{_f(H - c)}" width="{_f(W)}" height="{_f(dd)}" fill="{TOLERANCE}" fill-opacity=".14"/>')
    T.ligne(0, H - c_min, W, H - c_min, TOLERANCE, 1.3, tir)
    g.append(f'<line x1="0" y1="{_f(H)}" x2="{_f(W)}" y2="{_f(H)}" stroke="{ENCRE}" stroke-width="{_f(u * 0.085 * 2.6)}"/>')
    T.rupture(0, H, 0, 0)
    T.rupture(W, 0, W, H)
    T.ligne(0, 0, W, 0, ENCRE, 0.6, f"{_f(u * 0.9)} {_f(u * 0.5)}")
    # ---- la barre extérieure (ou l'étrier), puis les barres principales
    yc = H - xc0
    if e > 0:
        g.append(f'<rect x="0" y="{_f(H - c - e)}" width="{_f(W)}" height="{_f(e)}" fill="{ACIER}"/>')
    xs = [x0, x0 + pas, x0 + 2 * pas]
    for x in xs:
        g.append(f'<circle cx="{_f(x)}" cy="{_f(yc)}" r="{_f(d / 2)}" fill="{ENCRE}"/>')
    # ---- les cotes verticales à gauche (c_nom, a), la distance demandée au feu en rouge ; leurs textes à
    # l'horizontale, en marge, reliés par un trait (une cote de 20 mm ne porte pas un texte lisible)
    x1, x2 = -1.3 * u, -2.6 * u
    T.ligne(-u * 0.4, H, x2 - u * 0.5, H)
    T.ligne(0, H - c, x1 - u * 0.5, H - c, COTE, 0.8, fin)
    T.ligne(xs[0], yc, x2 - u * 0.5, yc, COTE, 0.8, fin)
    T.cote_v(x1, H - c, H, None)
    T.cote_v(x2, yc, H, None)
    gauches = [(x1, H - c / 2, _indice("c", "nom", f" = {_mm(c)}", u), ENCRE, False),
               (x2, H - xc0 / 2, f"a = {_mm(xc0)}", ENCRE, False)]
    if f is not None and f.get("a") is not None:      # (poteau R240 : le tableau n'a pas de valeur, rien à repérer)
        ar = float(f["a"])
        g.append(f'<path d="M{_f(x2)} {_f(H - ar)} l{_f(-u * 0.75)} {_f(-u * 0.38)} v{_f(u * 0.76)} Z" fill="{EFFORT}"/>')
        gauches.append((x2 - u * 0.75, H - ar, _indice("a", "feu", f" ≥ {_mm(ar)} · {escape(f['r'])}", u), EFFORT, True))
    gauches.sort(key=lambda r: r[1])
    xt = x2 - 1.5 * u
    ys = []
    for _, y, _, _, _ in gauches:
        ys.append(y if not ys else max(y, ys[-1] + 1.45 * u))
    deb = min(0.0, H + 0.3 * u - ys[-1])
    ys = [y + deb for y in ys]
    for (xa, ya, t, coul, gras), y in zip(gauches, ys):
        T.ligne(xa, ya, xt + u * 0.3, y, coul, 0.9)
        T.txt(xt, y + u * 0.33, t, 0.9, "end", couleur=coul, gras=gras)
    # ---- sous le parement : son nom, puis la distance à l'axe décomposée (comme la cote a du détail d'angle)
    T.txt(W / 2, H + 1.25 * u, "parement exposé", 0.8, couleur=COTE, italique=True)
    t_a = (f"a = {_mm(c)} + {_mm(e)} + {_mm(d / 2)} = {_mm(xc0)}" if e > 0 else f"a = {_mm(c)} + {_mm(d / 2)} = {_mm(xc0)}")
    t_a_long = _indice("a = c", "nom", " + Ø", u) + (_indice("", "e", " + Ø/2", u) if e > 0 else "/2") + f" = {t_a[4:]}"
    T.txt(W / 2, H + 2.75 * u, t_a_long, 0.9)
    # ---- les renvois, en marge droite
    renvois = [((xs[-1] + d / 2 * 0.7, yc - d / 2 * 0.7), f"barre Ø{_mm(d)}", ENCRE)]
    if e > 0:
        renvois.append(((W - 0.3 * (W - xs[-1]), H - c - e / 2), f"barre extérieure Ø{_mm(e)}", ACIER))
    if dd > 0:
        renvois.append(((W - u * 1.0, H - c + dd / 2), _indice("Δc", "dev", f" = {_mm(dd)}", u), TOLERANCE))
    renvois.append(((W - u * 2.0, H - c_min), _indice("c", "min", f" = {_mm(c_min)}", u), TOLERANCE))
    haut = min(-0.9 * u, _renvois(T, renvois, W + 1.6 * u, H + 0.6 * u) - 1.1 * u, ys[0] - 1.1 * u)
    # ---- la boîte
    gauche = max(T.largeur(t, 0.9) for _, _, t, _, _ in gauches) - xt + 0.8 * u
    droite = max(T.largeur(t, 0.95) for _, t, _ in renvois) + 3.0 * u
    larg_a = T.largeur(t_a_long, 0.9) / 2 - W / 2
    titre = "Coupe de la dalle" if el["type"] == "Dalle" else "Coupe horizontale du voile"
    return _boite(ident, T, -max(gauche, larg_a + u), haut, W + max(droite, larg_a + u), H + 3.5 * u, titre, c)


def coupe(ident: str, res: dict) -> str:
    """Le dessin de l'enrobage d'un élément (le résultat de ``moteur.calculer``) : le détail d'angle d'une poutre ou d'un
    poteau, la coupe de la face exposée d'une dalle ou d'un voile."""
    return _angle(ident, res) if res["element"]["type"] in ("Poutre", "Poteau") else _face(ident, res)

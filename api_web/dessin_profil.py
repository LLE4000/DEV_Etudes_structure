# -*- coding: utf-8 -*-
"""Les dessins du module « Choix de profilé » : la SECTION du profilé retenu à l'échelle (contour exact du catalogue,
hachures, axes y-y et z-z, cotes h et b, repères t_w, t_f, r, la fiche des dimensions et des propriétés) ; et la
palette, les textes et les traits que le dessin de la barre (api_web/dessin_barre.py) lui emprunte.

Rien n'est calculé ici : le contour vient d'``acier/profiles`` (source unique, celle du catalogue et de son calcul).
Le style suit les dessins du garde-corps et de la poutre : traits noirs, acier gris hachuré à 45°, cotes fines à traits
obliques, efforts en rouge. Toutes les longueurs de la SECTION sont en mm de la scène, les épaisseurs de trait et les
textes proportionnels à la plus grande dimension : le dessin a la même allure de l'IPE 80 au HL 1100.
"""
import math
from html import escape
from typing import Dict, List, Sequence, Tuple

from acier.profiles import familles as FA

ENCRE, ACIER, HACH, COTE, AXE, EFFORT = "#15181F", "#DCE2E9", "#8794A3", "#5B6573", "#7A8290", "#9C3341"
GRIS, GRILLE, MUET, ACCENT, ETOILE = "#A7B0BD", "#E7EAEF", "#6B7280", "#2B4BCF", "#E0A100"
NIVEAU = {"p": ("#86EFAC", "#166534"), "f": ("#15803D", "#0F5A2C"), "r": ("#DC2626", "#991B1B")}
POLICE = "system-ui,-apple-system,'Segoe UI',Arial,sans-serif"


def _f(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _nb(x: float, nd: int) -> str:
    """Un nombre à la française (virgule, espace fine des milliers)."""
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def _brut(x: float) -> str:
    """Une cote telle que le catalogue la donne : 190, 6,5, 18 (sans zéro inutile)."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _sym(s: str, fs: float) -> str:
    """Un symbole « W_el,y = 389 » écrit avec son indice (tspan décalé), le texte échappé."""
    out, i = "", 0
    s = str(s)
    while i < len(s):
        j = s.find("_", i)
        if j < 0:
            out += escape(s[i:])
            break
        out += escape(s[i:j])
        k = j + 1
        while k < len(s) and (s[k].isalnum() or s[k] == ","):
            k += 1
        dy = 0.32 * fs
        out += f'<tspan font-size="{_f(0.72 * fs)}" dy="{_f(dy)}">{escape(s[j + 1:k])}</tspan><tspan dy="{_f(-dy)}">​</tspan>'
        i = k
    return out.replace("λ\u0304", '<tspan text-decoration="overline">λ</tspan>')


def _txt(x, y, t, fs, ancre="middle", couleur=ENCRE, gras=False, extra="", brut=False) -> str:
    poids = ' font-weight="650"' if gras else ""
    corps = t if brut else _sym(t, fs)
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(fs)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{corps}</text>')


def _ligne(x1, y1, x2, y2, couleur, lw, extra="") -> str:
    return f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(lw)}"{extra}/>'


# ------------------------------------------------------------------ la section
def _hachures(anneaux: Sequence[Sequence[Tuple[float, float]]], pas: float) -> List[Tuple[float, float, float, float]]:
    """Hachures à 45° de la matière (règle pair-impair sur TOUS les contours : un trou reste blanc) : segments
    (x1, y1, x2, y2) sur les droites x + y = c, espacées de ``pas``."""
    aretes = [(r[i], r[(i + 1) % len(r)]) for r in anneaux for i in range(len(r))]
    cs = [x + y for r in anneaux for x, y in r]
    c0, c1 = min(cs), max(cs)
    d = pas * math.sqrt(2)
    out = []
    c = c0 + d / 2
    while c < c1:
        inter = []
        for (x1, y1), (x2, y2) in aretes:
            f1, f2 = x1 + y1 - c, x2 + y2 - c
            if (f1 < 0) != (f2 < 0):
                t = f1 / (f1 - f2)
                inter.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        inter.sort(key=lambda p: p[0])
        for k in range(0, len(inter) - 1, 2):
            (xa, ya), (xb, yb) = inter[k], inter[k + 1]
            if abs(xa - xb) > 1e-9:
                out.append((xa, ya, xb, yb))
        c += d
    return out


def _cote(x1, y1, x2, y2, texte, u, cote_texte=-1.0) -> str:
    """Une cote : la ligne, deux traits obliques, le texte au milieu (au-dessus, ou à gauche si verticale)."""
    t = 1.3 * u
    lw = 0.22 * u
    out = _ligne(x1, y1, x2, y2, COTE, lw)
    for x, y in ((x1, y1), (x2, y2)):
        out += _ligne(x - t, y + t, x + t, y - t, ENCRE, 0.3 * u)
    fs = 4.4 * u
    if abs(y1 - y2) < 1e-9:
        out += _txt((x1 + x2) / 2, y1 + (cote_texte * 1.6 * u if cote_texte < 0 else 5.4 * u), texte, fs)
    else:
        xt, yt = x1 - 1.6 * u, (y1 + y2) / 2
        out += _txt(xt, yt, texte, fs, extra=f' transform="rotate(-90 {_f(xt)} {_f(yt)})"')
    return out


def _attache(x1, y1, x2, y2, u) -> str:
    return _ligne(x1, y1, x2, y2, COTE, 0.16 * u, f' stroke-dasharray="{_f(1.1 * u)} {_f(0.8 * u)}"')


def _repere(xp, yp, xt, yt, texte, u) -> str:
    """Un repère : un point sur la pièce, un trait de rappel, le texte (t_w = 6,5)."""
    fs = 4.2 * u
    return (f'<circle cx="{_f(xp)}" cy="{_f(yp)}" r="{_f(0.55 * u)}" fill="{ENCRE}"/>'
            + f'<polyline points="{_f(xp)},{_f(yp)} {_f(xt - 0.8 * u)},{_f(yt)} {_f(xt)},{_f(yt)}" fill="none" stroke="{COTE}" stroke-width="{_f(0.18 * u)}"/>'
            + _txt(xt + 0.8 * u, yt + 0.36 * fs, texte, fs, ancre="start"))


def _tableau(x0, y0, titre, lignes, u) -> Tuple[str, float]:
    """Un petit tableau de la fiche : le titre en capitales, puis « symbole · valeur unité » ; rend (svg, hauteur)."""
    fs = 4.1 * u
    pas = 6.4 * u
    largeur = 50 * u
    out = _txt(x0, y0, titre.upper(), 3.3 * u, ancre="start", couleur=MUET, gras=True, extra=' letter-spacing="0.6"')
    out += _ligne(x0, y0 + 2.0 * u, x0 + largeur, y0 + 2.0 * u, GRILLE, 0.25 * u)
    y = y0 + 2.0 * u + pas
    for sym, val, unite in lignes:
        out += _txt(x0, y - 1.6 * u, sym, fs, ancre="start", couleur=MUET)
        out += _txt(x0 + 37 * u, y - 1.6 * u, val, fs, ancre="end", gras=True, brut=True)
        out += _txt(x0 + 38.6 * u, y - 1.6 * u, unite, 3.6 * u, ancre="start", couleur=MUET, brut=True)
        out += _ligne(x0, y + 0.4 * u, x0 + largeur, y + 0.4 * u, GRILLE, 0.16 * u)
        y += pas
    return out, y - y0


def section(ident: str, nom: str, sous_titre: str, famille: str, dims: Dict[str, float], dimensions: List[Tuple[str, str, str]],
            proprietes: List[Tuple[str, str, str]], efforts: str, fiche: bool = True) -> str:
    """La section du profilé à l'échelle (mm), sa fiche dessous (la feuille est plus haute que large). ``dimensions`` et ``proprietes`` :
    [(symbole, valeur écrite, unité)] ; ``efforts`` : la ligne des efforts (texte), sous le dessin. ``fiche`` faux : le
    dessin seul (la note PDF écrit les dimensions et les propriétés à côté, avec ses propres tableaux)."""
    anneaux = [[(float(p[0]), -float(p[1])) for p in r] for r in FA.contours(famille, dims)]
    xs = [x for r in anneaux for x, _ in r]
    ys = [y for r in anneaux for _, y in r]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    b, h = x1 - x0, y1 - y0
    u = max(b, h) / 100.0
    g = []
    # la matière : remplie, hachurée, cernée
    d = " ".join("M" + " L".join(f"{_f(x)},{_f(y)}" for x, y in r) + " Z" for r in anneaux)
    g.append(f'<path d="{d}" fill="{ACIER}" fill-rule="evenodd" stroke="none"/>')
    g.append("".join(_ligne(a, b_, c, d_, HACH, 0.17 * u) for a, b_, c, d_ in _hachures(anneaux, 2.3 * u)))
    g.append(f'<path d="{d}" fill="none" stroke="{ENCRE}" stroke-width="{_f(0.55 * u)}" stroke-linejoin="round"/>')
    # les axes par le centre de gravité (symétrie des I et H : l'origine du contour)
    tiret = f' stroke-dasharray="{_f(5 * u)} {_f(1.4 * u)} {_f(0.7 * u)} {_f(1.4 * u)}"'
    g.append(_ligne(x0 - 5 * u, 0, x1 + 5 * u, 0, AXE, 0.22 * u, tiret))
    g.append(_ligne(0, y0 - 5 * u, 0, y1 + 5 * u, AXE, 0.22 * u, tiret))
    g.append(_txt(x1 + 6.2 * u, 1.5 * u, "y", 4.4 * u, ancre="start", couleur=AXE, extra=' font-style="italic"'))
    g.append(_txt(1.6 * u, y0 - 5.6 * u, "z", 4.4 * u, ancre="start", couleur=AXE, extra=' font-style="italic"'))
    # les cotes : la hauteur à gauche (le diamètre d'un tube rond), la largeur dessous (un tube rond n'en a pas d'autre)
    xh = x0 - 11 * u
    g.append(_attache(x0 - 1.2 * u, y0, xh - 1.5 * u, y0, u) + _attache(x0 - 1.2 * u, y1, xh - 1.5 * u, y1, u))
    g.append(_cote(xh, y0, xh, y1, f"h = {_brut(dims['h'])}" if "h" in dims else f"d = {_brut(dims['d'])}", u))
    yb = y1 + 10 * u
    if "b" in dims:
        g.append(_attache(x0, y1 + 1.2 * u, x0, yb + 1.5 * u, u) + _attache(x1, y1 + 1.2 * u, x1, yb + 1.5 * u, u))
        g.append(_cote(x0, yb, x1, yb, f"b = {_brut(dims['b'])}", u, cote_texte=+1.0))
    # les repères des épaisseurs et du congé, alignés à droite de la section
    xr = x1 + 10 * u
    tw, tf, t = dims.get("tw"), dims.get("tf"), dims.get("t")
    reperes = []
    if tf:
        reperes.append((x1 - min(0.18 * b, 9 * u), y0 + tf / 2, f"t_f = {_brut(tf)}"))
    r = dims.get("r", dims.get("r1"))
    if r and tw and tf:
        a = r * (1 - math.sqrt(0.5))
        reperes.append((tw / 2 + a, y0 + tf + a, f"r = {_brut(r)}" if "r" in dims else f"r_1 = {_brut(r)}"))
    if tw:
        reperes.append((tw / 2, y0 + 0.36 * h, f"t_w = {_brut(tw)}"))
    if t and "d" in dims:                          # un tube rond : le repère sur la paroi, en haut à droite
        rm = (dims["d"] - t) / 2.0
        reperes.append((rm * math.cos(math.radians(40)), -rm * math.sin(math.radians(40)), f"t = {_brut(t)}"))
    elif t:                                        # un tube carré ou rectangulaire : sur la paroi de droite
        reperes.append((x1 - t / 2, y0 + 0.36 * h, f"t = {_brut(t)}"))
    yt = y0 - 1.0 * u
    for xp, yp, t in reperes:
        yt = max(yt + 7.2 * u, yp)
        g.append(_repere(xp, yp, xr, yt, t, u))
    # le nom au-dessus (au-dessus de l'axe z), les efforts dessous
    xc = (x0 + x1) / 2
    g.append(_txt(xc, y0 - 19 * u, nom, 6.4 * u, gras=True))
    g.append(_txt(xc, y0 - 12.6 * u, sous_titre, 3.9 * u, couleur=MUET))
    if efforts:
        g.append(_txt(xc, yb + 15.5 * u, efforts, 4.4 * u, couleur=EFFORT, gras=True))
    # la fiche, SOUS la section : dimensions puis propriétés, côte à côte (la feuille est plus haute que large)
    yf = yb + (25 * u if efforts else 14 * u)
    xf = xc - 56 * u
    haut = y0 - 27 * u
    if fiche:
        t1, h1 = _tableau(xf, yf, "Dimensions", dimensions, u)
        t2, h2 = _tableau(xf + 62 * u, yf, "Propriétés", proprietes, u)
        g += [t1, t2]
        gauche = min(xh - 8 * u, xf - 4 * u)
        droite = max(xr + 30 * u, xf + 116 * u)
        bas = yf + max(h1, h2) + 2 * u
    else:
        gauche, droite, bas = xh - 8 * u, xr + 30 * u, yf - 6 * u
    vb = f"{_f(gauche)} {_f(haut)} {_f(droite - gauche)} {_f(bas - haut)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Section {escape(nom)} à l\'échelle">{"".join(g)}</svg>')

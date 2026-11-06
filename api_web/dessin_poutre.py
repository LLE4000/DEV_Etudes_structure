# -*- coding: utf-8 -*-
"""La COUPE d'une poutre en béton armé, dessinée à l'échelle (mm) à partir de la
géométrie que le module poutre a calculée — b, h, enrobage, étrier, et la distance de chaque lit (parement → axe) telle
que modules/poutre.py la retient. Les barres d'un lit sont réparties également entre les branches de l'étrier : c'est une
convention de dessin, pas une donnée de calcul.

Le style suit les dessins du garde-corps : traits noirs, béton gris clair, cotes fines, efforts en rouge. Le site
(api_web/poutre.py) et les maquettes (site/maquettes/extraire_donnees.py) dessinent avec ce MÊME fichier.
"""
from html import escape
from typing import List, Tuple

ENCRE, BETON, ACIER, COTE, EFFORT = "#15181F", "#E6EAEF", "#33415C", "#7F8C9A", "#9C3341"
POLICE = "system-ui,Arial,sans-serif"


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _txt(x, y, t, taille=16.0, ancre="middle", couleur=ENCRE, gras=True, extra="") -> str:
    poids = ' font-weight="600"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{t}</text>')


def _cote_h(x1, x2, y, texte, taille=15.0) -> str:
    """Une cote horizontale : ligne, deux traits obliques, le nombre au-dessus."""
    t = 6.0
    return (f'<line x1="{_f(x1)}" y1="{_f(y)}" x2="{_f(x2)}" y2="{_f(y)}" stroke="{COTE}" stroke-width="1"/>'
            f'<line x1="{_f(x1 - t)}" y1="{_f(y + t)}" x2="{_f(x1 + t)}" y2="{_f(y - t)}" stroke="{ENCRE}" stroke-width="1.2"/>'
            f'<line x1="{_f(x2 - t)}" y1="{_f(y + t)}" x2="{_f(x2 + t)}" y2="{_f(y - t)}" stroke="{ENCRE}" stroke-width="1.2"/>'
            + _txt((x1 + x2) / 2, y - 7, texte, taille))


def _cote_v(x, y1, y2, texte, taille=15.0, cote_droit=False) -> str:
    """Une cote verticale : le nombre tourné le long de la ligne."""
    t = 6.0
    xt = x + (16 if cote_droit else -8)
    return (f'<line x1="{_f(x)}" y1="{_f(y1)}" x2="{_f(x)}" y2="{_f(y2)}" stroke="{COTE}" stroke-width="1"/>'
            f'<line x1="{_f(x - t)}" y1="{_f(y1 + t)}" x2="{_f(x + t)}" y2="{_f(y1 - t)}" stroke="{ENCRE}" stroke-width="1.2"/>'
            f'<line x1="{_f(x - t)}" y1="{_f(y2 + t)}" x2="{_f(x + t)}" y2="{_f(y2 - t)}" stroke="{ENCRE}" stroke-width="1.2"/>'
            + _txt(xt, (y1 + y2) / 2, texte, taille, extra=f' transform="rotate(-90 {_f(xt)} {_f((y1 + y2) / 2)})"'))


def _attache(x, y1, y2) -> str:
    return f'<line x1="{_f(x)}" y1="{_f(y1)}" x2="{_f(x)}" y2="{_f(y2)}" stroke="{COTE}" stroke-width=".8" stroke-dasharray="4 3"/>'


def section(b_cm: float, h_cm: float, c_cm: float, d_et_mm: float, lits_inf: List[Tuple[int, int, float]],
            lits_sup: List[Tuple[int, int, float]], etrier: str, efforts: str, dx: float = 0.0,
            tendu: str = "inf") -> Tuple[str, float, float]:
    """Le groupe SVG d'une coupe, décalé de ``dx`` : (svg, largeur occupée, hauteur occupée) en mm.

    ``lits_*`` : [(n, Ø mm, distance parement → axe en cm), …] — valeurs du module poutre. ``tendu`` : la face tendue
    (« inf » en travée, « sup » sur appui) ; la hauteur utile d se cote depuis la face comprimée jusqu'aux armatures tendues."""
    b, h, c, de = b_cm * 10, h_cm * 10, c_cm * 10, d_et_mm
    g = []
    # le béton, puis l'étrier (épaisseur réelle, angles arrondis comme un façonnage)
    g.append(f'<rect x="0" y="0" width="{_f(b)}" height="{_f(h)}" fill="{BETON}" stroke="{ENCRE}" stroke-width="2.2"/>')
    g.append(f'<rect x="{_f(c + de / 2)}" y="{_f(c + de / 2)}" width="{_f(b - 2 * c - de)}" height="{_f(h - 2 * c - de)}" '
             f'rx="{_f(2.5 * de)}" fill="none" stroke="{ACIER}" stroke-width="{_f(de)}" stroke-linejoin="round"/>')

    def barres(lits, en_bas):
        out, y_axes = [], []
        for n, d, dist_cm in lits:
            y = h - dist_cm * 10 if en_bas else dist_cm * 10
            y_axes.append(y)
            x0, x1 = c + de + d / 2, b - c - de - d / 2
            for k in range(n):
                x = x0 if n == 1 else x0 + (x1 - x0) * k / (n - 1)
                out.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(d / 2)}" fill="{ENCRE}"/>')
        return "".join(out), y_axes

    s_inf, y_inf = barres(lits_inf, True)
    s_sup, y_sup = barres(lits_sup, False)
    g += [s_inf, s_sup]
    # les cotes : b dessous, h à gauche, d utile à droite (de la face comprimée à l'axe des armatures tendues), l'enrobage
    # en haut à gauche
    g.append(_attache(0, h + 8, h + 58) + _attache(b, h + 8, h + 58) + _cote_h(0, b, h + 50, f"b = {_f(b)}"))
    g.append(_attache(-8, 0, 0) + _attache(-8, h, h) + _cote_v(-48, 0, h, f"h = {_f(h)}"))
    y_tendu = y_sup if tendu == "sup" else y_inf
    if y_tendu:
        yd = sum(y_tendu) / len(y_tendu)
        y0 = h if tendu == "sup" else 0.0  # la face comprimée
        g.append(_attache(b + 8, y0, y0) + f'<line x1="{_f(b)}" y1="{_f(yd)}" x2="{_f(b + 64)}" y2="{_f(yd)}" stroke="{COTE}" stroke-width=".8" stroke-dasharray="4 3"/>'
                 + _cote_v(b + 48, min(y0, yd), max(y0, yd), f"d = {_f(abs(yd - y0))}", cote_droit=True))
    g.append(_txt(c / 2, -12, f"c = {_f(c)}", 13, couleur=COTE, gras=False))
    # les étiquettes des armatures, sous et au-dessus de la coupe ; l'étrier sur le côté
    lab_inf = " + ".join(f"{n}Ø{d}" for n, d, _ in lits_inf)
    lab_sup = " + ".join(f"{n}Ø{d}" for n, d, _ in lits_sup)
    g.append(_txt(b / 2, h + 92, lab_inf, 17))
    g.append(_txt(b / 2, -44, lab_sup, 17))
    g.append(_txt(b / 2, h + 118, etrier, 14, couleur=ACIER, gras=False))
    g.append(_txt(b / 2, -76, efforts, 14.5, couleur=EFFORT))
    return f'<g transform="translate({_f(dx)} 0)">{"".join(g)}</g>', b, h


def svg(ident: str, coupes: List[dict], titre: str = "") -> str:
    """Une ou plusieurs coupes côte à côte, à la même échelle ; chaque coupe porte son nom au-dessus. Le nom et le titre
    sont du TEXTE (un nom de section peut venir d'un fichier .stab) : ils sont échappés ici."""
    parts, x, hmax = [], 0.0, 0.0
    for k, cp in enumerate(coupes):
        g, w, hh = section(cp["b"], cp["h"], cp["c"], cp["d_et"], cp["inf"], cp["sup"], cp["etrier"], cp["efforts"], dx=x,
                           tendu=cp.get("tendu", "inf"))
        parts.append(g)
        if len(coupes) > 1:
            parts.append(_txt(x + w / 2, -108, escape(cp["nom"]), 18))
        x += w + 300
        hmax = max(hmax, hh)
    larg = x - 300
    haut_titre = 140 if len(coupes) > 1 else 110
    vb = f"{_f(-130)} {_f(-haut_titre)} {_f(larg + 300)} {_f(hmax + haut_titre + 150)}"
    tt = _txt(larg / 2, -haut_titre + 26, escape(titre), 18) if titre else ""
    return (f'<svg id="{ident}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Coupe de poutre en béton armé">{tt}{"".join(parts)}</svg>')

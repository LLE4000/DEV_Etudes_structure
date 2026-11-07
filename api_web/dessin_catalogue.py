# -*- coding: utf-8 -*-
"""Les dessins du catalogue des profilés : la SECTION d'un profilé, à l'échelle (mm), et la comparaison de plusieurs
sections côte à côte, à la même échelle.

Le contour dessiné est CELUI DU CALCUL (``acier.profiles.familles.contours`` : congés, semelles coniques, coins des
creux), les axes passent par le centre de gravité du moteur (y_c, z_c), le centre de cisaillement des U et les axes
principaux des cornières sont ceux du moteur. Rien n'est recalculé ici : on place.

Le style suit les dessins du site (``api_web/dessin_poutre.py``, garde-corps) : acier hachuré, trait de contour fort,
cotes fines à traits obliques, axes en trait mixte. Les tailles (texte, traits obliques, décalages des cotes) sont
proportionnelles à la section, de sorte qu'une cornière de 20 mm et une HE 1000 se lisent pareil à l'écran ; les
traits gardent leur épaisseur à l'écran (``vector-effect``).
"""
import math
from html import escape
from typing import Dict, List, Sequence, Tuple

from acier.profiles import familles as F

ENCRE, ACIER, HACHURE, COTE, AXE, REPERE = "#15181F", "#E4E9F0", "#7C8AA0", "#7F8C9A", "#5B6B82", "#9C3341"
POLICE = "system-ui,Arial,sans-serif"
NS = ' vector-effect="non-scaling-stroke"'


def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _n(x: float) -> str:
    """Une dimension écrite comme sur un plan : 300, 7,1, 168,3."""
    return f"{float(x):.2f}".rstrip("0").rstrip(".").replace(".", ",")


def _sym(s: str) -> str:
    """« t_w » → t avec w en indice (tspan) ; le reste échappé."""
    if "_" not in s:
        return escape(s)
    a, b = s.split("_", 1)
    return f'{escape(a)}<tspan font-size="72%" dy="0.28em">{escape(b)}</tspan><tspan dy="-0.28em"></tspan>'


class _Plume:
    """Ce qui est commun à un dessin : l'unité de taille u (mm du dessin pour un « pixel » d'écran à peu près)."""

    def __init__(self, u: float):
        self.u = u

    def txt(self, x, y, t, taille=13.0, ancre="middle", couleur=ENCRE, gras=True, italique=False, extra="") -> str:
        poids = ' font-weight="600"' if gras else ""
        it = ' font-style="italic"' if italique else ""
        return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille * self.u)}" fill="{couleur}" '
                f'font-family="{POLICE}"{poids}{it}{extra}>{t}</text>')

    def ligne(self, x1, y1, x2, y2, couleur=COTE, ep=1.0, tirets="") -> str:
        d = f' stroke-dasharray="{tirets}"' if tirets else ""
        return (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"'
                f'{d}{NS}/>')

    def tic(self, x, y) -> str:
        t = 4.5 * self.u
        return self.ligne(x - t, y + t, x + t, y - t, ENCRE, 1.3)

    def axe(self, x1, y1, x2, y2) -> str:
        return self.ligne(x1, y1, x2, y2, AXE, 0.9, "10 3 2 3")

    def cote_h(self, x1, x2, y, texte, dessous=False) -> str:
        """Une cote horizontale : ligne, traits obliques, le texte au-dessus (ou dessous)."""
        yt = y + 15 * self.u if dessous else y - 6 * self.u
        return self.ligne(x1, y, x2, y) + self.tic(x1, y) + self.tic(x2, y) + self.txt((x1 + x2) / 2, yt, texte)

    def cote_v(self, x, y1, y2, texte, droite=False) -> str:
        """Une cote verticale, le texte tourné le long de la ligne."""
        xt = x + 15 * self.u if droite else x - 6 * self.u
        ym = (y1 + y2) / 2
        return (self.ligne(x, y1, x, y2) + self.tic(x, y1) + self.tic(x, y2)
                + self.txt(xt, ym, texte, extra=f' transform="rotate(-90 {_f(xt)} {_f(ym)})"'))

    def cote_ext_h(self, x1, x2, y, texte, cote_droit=True) -> str:
        """Une cote d'épaisseur (horizontale) : flèches extérieures, le texte à côté (une âme mince n'a pas la place
        d'un texte entre ses deux faces)."""
        L = 16 * self.u
        f = 5 * self.u
        out = self.ligne(x1 - L, y, x2 + L, y)
        out += (f'<path d="M{_f(x1)},{_f(y)} l{_f(-f * 1.6)},{_f(-f * .55)} v{_f(f * 1.1)} Z M{_f(x2)},{_f(y)} l{_f(f * 1.6)},{_f(-f * .55)} '
                f'v{_f(f * 1.1)} Z" fill="{ENCRE}"/>')
        xt = x2 + L + 3 * self.u if cote_droit else x1 - L - 3 * self.u
        return out + self.txt(xt, y + 4.5 * self.u, texte, ancre="start" if cote_droit else "end")

    def cote_ext_v(self, x, y1, y2, texte, dessus=True) -> str:
        """Une cote d'épaisseur verticale (semelle, aile) : flèches extérieures, le texte au bout."""
        L = 16 * self.u
        f = 5 * self.u
        ya, yb = min(y1, y2), max(y1, y2)
        out = self.ligne(x, ya - L, x, yb + L)
        out += (f'<path d="M{_f(x)},{_f(ya)} l{_f(-f * .55)},{_f(-f * 1.6)} h{_f(f * 1.1)} Z M{_f(x)},{_f(yb)} l{_f(-f * .55)},{_f(f * 1.6)} '
                f'h{_f(f * 1.1)} Z" fill="{ENCRE}"/>')
        return out + (self.txt(x, ya - L - 4 * self.u, texte) if dessus else self.txt(x, yb + L + 14 * self.u, texte))

    def repere(self, x, y, lettre, couleur=REPERE, dx=8.0, dy=-8.0) -> str:
        """Un point repéré (G, S) : un petit cercle barré et sa lettre."""
        r = 3.6 * self.u
        return (f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="#fff" stroke="{couleur}" stroke-width="1.2"{NS}/>'
                + self.ligne(x - r, y, x + r, y, couleur, 1.0) + self.ligne(x, y - r, x, y + r, couleur, 1.0)
                + self.txt(x + dx * self.u, y + dy * self.u, lettre, 12, "start", couleur, italique=True))

    def attache_v(self, x, y1, y2) -> str:
        return self.ligne(x, y1, x, y2, COTE, 0.7, "4 3")

    def attache_h(self, y, x1, x2) -> str:
        return self.ligne(x1, y, x2, y, COTE, 0.7, "4 3")


def _chemin(rings, X, Y) -> str:
    out = []
    for ring in rings:
        out.append(" ".join(f"{'M' if i == 0 else 'L'}{_f(X(pt[0]))},{_f(Y(pt[1]))}" for i, pt in enumerate(ring)) + " Z")
    return " ".join(out)


def _hachure(ident: str, pas: float) -> str:
    return (f'<defs><pattern id="{ident}_h" patternUnits="userSpaceOnUse" width="{_f(pas)}" height="{_f(pas)}" '
            f'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="{_f(pas)}" stroke="{HACHURE}" stroke-width="0.7"{NS}/>'
            f'</pattern></defs>')


def _corps(ident: str, rings, X, Y) -> str:
    d = _chemin(rings, X, Y)
    return (f'<path d="{d}" fill="{ACIER}" fill-rule="evenodd"/><path d="{d}" fill="url(#{ident}_h)" fill-rule="evenodd"/>'
            f'<path d="{d}" fill="none" stroke="{ENCRE}" stroke-width="1.8" stroke-linejoin="round"{NS}/>')


def _boite(rings) -> Tuple[float, float, float, float]:
    ys = [float(pt[0]) for r in rings for pt in r]
    zs = [float(pt[1]) for r in rings for pt in r]
    return min(ys), max(ys), min(zs), max(zs)


def _dims(p: Dict) -> Dict[str, float]:
    return {k: float(v) for k, v in (p.get("dims") or {}).items()}


def legende(p: Dict) -> str:
    """Les dimensions du profilé en une ligne (TEXTE) : « h 300 · b 300 · tw 11 · tf 19 · r 27 mm »."""
    d = _dims(p)
    return " · ".join(f"{k} {_n(d[k])}" for k in F.FAMILLES[p["famille"]]["dims"] if k in d) + " mm"


def section(ident: str, p: Dict) -> str:
    """La section d'un profilé, à l'échelle, cotée : h et b (ou Ø), épaisseurs, congé, axes y-y et z-z par le centre
    de gravité ; le centre de cisaillement d'un U ; les axes principaux u et v d'une cornière."""
    fam = p["famille"]
    forme = F.FAMILLES[fam]["forme"]
    d = _dims(p)
    rings = F.contours(fam, d)
    y0, y1, z0, z1 = _boite(rings)
    L, H = y1 - y0, z1 - z0
    S = max(L, H)
    u = S / 300.0                                   # l'unité de taille : 1/300 de la plus grande dimension
    X = lambda y: y                                  # noqa: E731 — le repère du contour : y vers la droite,
    Y = lambda z: -z                                 # noqa: E731 — z vers le haut (l'écran descend)
    pl = _Plume(u)
    gy = float(p["yc"]) if p.get("yc") is not None else (y0 + y1) / 2
    gz = float(p["zc"]) if p.get("zc") is not None else (z0 + z1) / 2
    g = [_hachure(ident, 7 * u), _corps(ident, rings, X, Y)]
    # les axes par le centre de gravité, débordant de la section ; leurs lettres aux deux bouts (à gauche, au-delà du
    # centre de cisaillement d'un U, qui tombe hors de la section)
    ex = 34 * u
    ys = gy + float(p["y_s"]) if forme in ("U", "U_conique") and p.get("y_s") is not None else y0
    exg = ex + max(0.0, y0 - ys)
    g.append(pl.axe(X(y0) - exg, Y(gz), X(y1) + ex, Y(gz)) + pl.axe(X(gy), Y(z1) - ex, X(gy), Y(z0) + ex))
    g.append(pl.txt(X(y1) + ex + 3 * u, Y(gz) + 4.5 * u, "y", 12, "start", AXE, italique=True)
             + pl.txt(X(y0) - exg - 3 * u, Y(gz) + 4.5 * u, "y", 12, "end", AXE, italique=True)
             + pl.txt(X(gy) + 5 * u, Y(z1) - ex + 4 * u, "z", 12, "start", AXE, italique=True)
             + pl.txt(X(gy) + 5 * u, Y(z0) + ex + 4 * u, "z", 12, "start", AXE, italique=True))
    # les cotes d'encombrement : la largeur dessous, la hauteur à gauche (Ø pour un rond)
    yb = Y(z0) + 62 * u
    xg = X(y0) - 70 * u - max(0.0, y0 - ys)          # la cote h passe au-delà du centre de cisaillement d'un U
    if forme in ("creux_rond", "plein_rond"):
        g.append(pl.attache_v(X(y0), Y(gz), yb + 6 * u) + pl.attache_v(X(y1), Y(gz), yb + 6 * u)
                 + pl.cote_h(X(y0), X(y1), yb, f"Ø {_n(d['d'])}", dessous=True))
    else:
        g.append(pl.attache_v(X(y0), Y(z0) + 8 * u, yb + 6 * u) + pl.attache_v(X(y1), Y(z0) + 8 * u, yb + 6 * u)
                 + pl.cote_h(X(y0), X(y1), yb, f"b = {_n(d['b'])}", dessous=True))
        g.append(pl.attache_h(Y(z1), X(y0) - 8 * u, xg - 6 * u) + pl.attache_h(Y(z0), X(y0) - 8 * u, xg - 6 * u)
                 + pl.cote_v(xg, Y(z1), Y(z0), f"h = {_n(d['h'])}"))
    # les épaisseurs et le congé, selon la forme
    if forme in ("I", "I_conique"):
        h, b, tw, tf = d["h"], d["b"], d["tw"], d["tf"]
        zc = gz + 0.22 * h
        g.append(pl.cote_ext_h(X(-tw / 2), X(tw / 2), Y(zc), _sym("t_w") + f" = {_n(tw)}"))
        xr = X(b / 2) + 30 * u
        g.append(pl.attache_h(Y(h / 2), X(b / 2) + 6 * u, xr + 6 * u) + pl.attache_h(Y(h / 2 - tf), X(b / 2) + 6 * u, xr + 6 * u)
                 + pl.cote_ext_v(xr, Y(h / 2), Y(h / 2 - tf), _sym("t_f") + f" = {_n(tf)}"))
        r = d.get("r", d.get("r1", 0.0))
        if r > 0:
            cy, cz = -tw / 2 - r, -h / 2 + tf + r                     # le congé en bas à gauche
            px, pz = cy + r * math.cos(math.radians(-45)), cz + r * math.sin(math.radians(-45))
            tx, tz = px - 46 * u, pz + 30 * u
            g.append(pl.ligne(X(px), Y(pz), X(tx), Y(tz), ENCRE, 0.9) + pl.ligne(X(tx), Y(tz), X(tx) - 34 * u, Y(tz), ENCRE, 0.9)
                     + pl.txt(X(tx) - 34 * u, Y(tz) - 4 * u, f"r = {_n(r)}", 12, "start", ENCRE, False))
    elif forme in ("U", "U_conique"):
        h, b, tw, tf = d["h"], d["b"], d["tw"], d["tf"]
        zc = gz + 0.22 * h
        g.append(pl.cote_ext_h(X(0), X(tw), Y(zc), _sym("t_w") + f" = {_n(tw)}"))
        xr = X(b) + 30 * u
        g.append(pl.attache_h(Y(h / 2), X(b) + 6 * u, xr + 6 * u) + pl.attache_h(Y(h / 2 - tf), X(b) + 6 * u, xr + 6 * u)
                 + pl.cote_ext_v(xr, Y(h / 2), Y(h / 2 - tf), _sym("t_f") + f" = {_n(tf)}"))
        g.append(pl.repere(X(gy), Y(gz), "G", AXE, 7, -7))
        if p.get("y_s") is not None:
            g.append(pl.repere(X(gy + float(p["y_s"])), Y(gz + float(p.get("z_s") or 0.0)), "S", REPERE, -4, 17))
    elif forme == "L":
        h, b, t = d["h"], d["b"], d["t"]
        g.append(pl.cote_ext_h(X(0), X(t), Y(h - 0.18 * h), "t = " + _n(t)))
        g.append(pl.repere(X(gy), Y(gz), "G", AXE, 7, -7))
        a = math.radians(float(p.get("alpha_deg") or 0.0))
        if abs(a) > 1e-9:                                           # les axes principaux u (fort) et v
            R = 0.62 * S
            cu, su = math.cos(a), math.sin(a)
            g.append(pl.ligne(X(gy - R * cu), Y(gz - R * su), X(gy + R * cu), Y(gz + R * su), REPERE, 0.8, "10 3 2 3")
                     + pl.ligne(X(gy + R * su), Y(gz - R * cu), X(gy - R * su), Y(gz + R * cu), REPERE, 0.8, "10 3 2 3")
                     + pl.txt(X(gy + R * cu) + 4 * u, Y(gz + R * su), "u", 12, "start", REPERE, italique=True)
                     + pl.txt(X(gy - R * su) + 4 * u, Y(gz + R * cu), "v", 12, "start", REPERE, italique=True))
    elif forme == "creux_rect":
        h, b, t = d["h"], d["b"], d["t"]
        g.append(pl.cote_ext_h(X(b / 2 - t), X(b / 2), Y(0.22 * h), "t = " + _n(t)))
    elif forme == "creux_rond":
        D, t = d["d"], d["t"]
        g.append(pl.cote_ext_h(X(D / 2 - t), X(D / 2), Y(0.36 * D), "t = " + _n(t)))
    elif forme == "plein_rect":
        g.append(pl.attache_h(Y(z1), X(y0) - 8 * u, xg - 6 * u))
    vx0, vx1 = X(y0) - 112 * u - max(0.0, y0 - ys), X(y1) + 112 * u
    vy0, vy1 = Y(z1) - 62 * u, Y(z0) + 96 * u
    vb = f"{_f(vx0)} {_f(vy0)} {_f(vx1 - vx0)} {_f(vy1 - vy0)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Section {escape(p["nom"])}">{"".join(g)}</svg>')


def vignette(ident: str, p: Dict, taille: int = 58) -> str:
    """La section en petit (le cadre Résultat) : le contour seul, plein."""
    rings = F.contours(p["famille"], _dims(p))
    y0, y1, z0, z1 = _boite(rings)
    S = max(y1 - y0, z1 - z0)
    m = 0.08 * S
    d = _chemin(rings, lambda y: y, lambda z: -z)
    vb = f"{_f(y0 - m)} {_f(-z1 - m)} {_f(y1 - y0 + 2 * m)} {_f(z1 - z0 + 2 * m)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{taille}" height="{taille}" '
            f'aria-hidden="true"><path d="{d}" fill="{ACIER}" fill-rule="evenodd" stroke="{ENCRE}" stroke-width="1.6"{NS}/></svg>')


def comparaison(ident: str, profils: Sequence[Dict]) -> str:
    """Les sections côte à côte, à la MÊME échelle, posées sur une même ligne, le nom et la masse dessous."""
    morceaux: List[Tuple[Dict, list, Tuple[float, float, float, float]]] = []
    for p in profils:
        rings = F.contours(p["famille"], _dims(p))
        morceaux.append((p, rings, _boite(rings)))
    S = max(max(b[1] - b[0], b[3] - b[2]) for _, _, b in morceaux)
    hmax = max(b[3] - b[2] for _, _, b in morceaux)
    # l'unité de taille suit la plus grande section ET la largeur de la rangée : quatre ou cinq sections côte à côte
    # sont posées sur ~700 px d'écran, et les noms dessous doivent rester lisibles (≈ 15 px), pas tomber à 9 px
    largeur = sum(b[1] - b[0] for _, _, b in morceaux) + 0.45 * S * (len(morceaux) - 1)
    u = max(S / 230.0, largeur / 700.0)
    pl = _Plume(u)
    ecart = max(0.45 * S, 70 * u)
    g = [_hachure(ident, 6 * u)]
    x = 0.0
    for k, (p, rings, (y0, y1, z0, z1)) in enumerate(morceaux):
        larg = max(y1 - y0, 60 * u)
        dx = x + (larg - (y1 - y0)) / 2 - y0
        g.append(_corps(f"{ident}", rings, lambda y, dx=dx: y + dx, lambda z, z0=z0: -(z - z0)))
        xm = x + larg / 2
        g.append(pl.txt(xm, 30 * u, escape(p["nom"]), 17))
        g.append(pl.txt(xm, 52 * u, f"{p['masse']:.1f} kg/m".replace(".", ","), 14, couleur=COTE, gras=False))
        x += larg + ecart
    larg_tot = x - ecart
    g.append(pl.ligne(-20 * u, 0, larg_tot + 20 * u, 0, COTE, 0.8, "6 3"))
    vb = f"{_f(-30 * u)} {_f(-hmax - 20 * u)} {_f(larg_tot + 60 * u)} {_f(hmax + 84 * u)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Comparaison des sections">{"".join(g)}</svg>')

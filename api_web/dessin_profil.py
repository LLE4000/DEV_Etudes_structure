# -*- coding: utf-8 -*-
"""Les dessins du module « Choix de profilé » : la SECTION du profilé retenu à l'échelle (contour exact du catalogue,
hachures, axes y-y et z-z, cotes h et b, repères t_w, t_f, r, la fiche des dimensions et des propriétés), la courbe
TAUX / MASSE de tous les profilés de la sélection, les courbes de FLAMBEMENT χ(λ̄) avec les points du profilé retenu.

Rien n'est calculé ici : le contour vient d'``acier/profiles`` (source unique, celle du catalogue et de son calcul), les
taux du moteur ``acier/choix_profil`` ; les courbes χ(λ̄) passent par ``calcul.chi_courbe`` (l'expression du module).
Le style suit les dessins du garde-corps et de la poutre : traits noirs, acier gris hachuré à 45°, cotes fines à traits
obliques, efforts en rouge. Toutes les longueurs de la SECTION sont en mm de la scène, les épaisseurs de trait et les
textes proportionnels à la plus grande dimension : le dessin a la même allure de l'IPE 80 au HL 1100.
"""
import math
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

from acier.choix_profil import calcul as CA
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
            proprietes: List[Tuple[str, str, str]], efforts: str) -> str:
    """La section du profilé à l'échelle (mm), sa fiche dessous (la feuille est plus haute que large). ``dimensions`` et ``proprietes`` :
    [(symbole, valeur écrite, unité)] ; ``efforts`` : la ligne des efforts (texte), sous le dessin."""
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
    # les cotes : h à gauche, b dessous
    xh = x0 - 11 * u
    g.append(_attache(x0 - 1.2 * u, y0, xh - 1.5 * u, y0, u) + _attache(x0 - 1.2 * u, y1, xh - 1.5 * u, y1, u))
    g.append(_cote(xh, y0, xh, y1, f"h = {_brut(dims['h'])}", u))
    yb = y1 + 10 * u
    g.append(_attache(x0, y1 + 1.2 * u, x0, yb + 1.5 * u, u) + _attache(x1, y1 + 1.2 * u, x1, yb + 1.5 * u, u))
    g.append(_cote(x0, yb, x1, yb, f"b = {_brut(dims['b'])}", u, cote_texte=+1.0))
    # les repères des épaisseurs et du congé, alignés à droite de la section
    xr = x1 + 10 * u
    tw, tf = dims.get("tw"), dims.get("tf")
    reperes = []
    if tf:
        reperes.append((x1 - min(0.18 * b, 9 * u), y0 + tf / 2, f"t_f = {_brut(tf)}"))
    r = dims.get("r", dims.get("r1"))
    if r and tw and tf:
        a = r * (1 - math.sqrt(0.5))
        reperes.append((tw / 2 + a, y0 + tf + a, f"r = {_brut(r)}" if "r" in dims else f"r_1 = {_brut(r)}"))
    if tw:
        reperes.append((tw / 2, y0 + 0.36 * h, f"t_w = {_brut(tw)}"))
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
    t1, h1 = _tableau(xf, yf, "Dimensions", dimensions, u)
    t2, h2 = _tableau(xf + 62 * u, yf, "Propriétés", proprietes, u)
    g += [t1, t2]
    gauche = min(xh - 8 * u, xf - 4 * u)
    droite = max(xr + 30 * u, xf + 116 * u)
    haut = y0 - 27 * u
    bas = yf + max(h1, h2) + 2 * u
    vb = f"{_f(gauche)} {_f(haut)} {_f(droite - gauche)} {_f(bas - haut)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Section {escape(nom)} à l\'échelle">{"".join(g)}</svg>')


# ------------------------------------------------------------------ les graphiques
def _graduations(vmax: float, n: int = 6) -> Tuple[float, List[float]]:
    """Un pas « rond » (1, 2, 2,5 ou 5 × 10ⁿ) et les graduations de 0 à au moins vmax."""
    brut = vmax / n
    p = 10 ** math.floor(math.log10(brut))
    pas = next(k * p for k in (1, 2, 2.5, 5, 10) if k * p >= brut)
    haut = math.ceil(vmax / pas - 1e-9) * pas
    return haut, [i * pas for i in range(int(round(haut / pas)) + 1)]


def _etoile(cx, cy, r) -> str:
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else 0.45 * r
        pts.append(f"{_f(cx + rr * math.cos(a))},{_f(cy + rr * math.sin(a))}")
    return f'<polygon points="{" ".join(pts)}" fill="{ETOILE}" stroke="#fff" stroke-width="1.2"/>'


def _etiquette(x, y, texte, ancre="start", couleur=ENCRE, fs=11.5, gras=True) -> str:
    """Un texte cerné de blanc (lisible sur la grille et les courbes)."""
    poids = ' font-weight="650"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(fs)}" fill="{couleur}" font-family="{POLICE}"{poids} '
            f'stroke="#fff" stroke-width="3.2" paint-order="stroke" stroke-linejoin="round">{_sym(texte, fs)}</text>')


def courbe(ident: str, points: List[Dict], retenu: Optional[str], leger: Optional[str]) -> str:
    """Le taux de chaque profilé de la sélection en fonction de sa masse : une ligne grise par famille, un point par
    profilé dans la couleur de son taux, la bande 95–100 %, la limite 100 %, le retenu (✓) et le plus léger qui tient
    (★). ``points`` : nom, famille, masse, pct (le taux en %), niveau, aide (le texte de la bulle)."""
    W, H = 760.0, 600.0
    L, R, T, B = 60.0, 118.0, 46.0, 50.0
    pw, ph = W - L - R, H - T - B
    pts = [p for p in points if p["masse"] is not None]
    xmax, xt = _graduations(max(p["masse"] for p in pts) * 1.04 if pts else 10.0)
    ymax = min(200.0, max(110.0, max((p["pct"] for p in pts), default=0.0) * 1.05))
    ymax, yt = _graduations(ymax, 6)

    def X(m):
        return L + pw * m / xmax

    def Y(v):
        return T + ph * (1 - min(v, ymax) / ymax)
    g = [f'<rect x="{_f(L)}" y="{_f(T)}" width="{_f(pw)}" height="{_f(ph)}" fill="#fff"/>']
    # la grille, les graduations
    for v in yt:
        g.append(_ligne(L, Y(v), L + pw, Y(v), GRILLE, 1))
        g.append(_txt(L - 8, Y(v) + 4, f"{_nb(v, 0)}", 11, ancre="end", couleur=MUET, brut=True))
    for m in xt:
        g.append(_ligne(X(m), T + ph, X(m), T + ph + 4, GRIS, 1))
        g.append(_txt(X(m), T + ph + 18, _nb(m, 0), 11, couleur=MUET, brut=True))
    g.append(_ligne(L, T + ph, L + pw, T + ph, GRIS, 1))
    # la bande d'attention et la limite
    g.append(f'<rect x="{_f(L)}" y="{_f(Y(100))}" width="{_f(pw)}" height="{_f(Y(95) - Y(100))}" fill="#DCFCE7"/>')
    g.append(_ligne(L, Y(100), L + pw, Y(100), NIVEAU["r"][0], 1.4, ' stroke-dasharray="6 4"'))
    g.append(_txt(L + pw + 6, Y(100) + 4, "100 %", 11, ancre="start", couleur=NIVEAU["r"][1], gras=True, brut=True))
    g.append(_txt(L + pw + 6, Y(95) + 13, "95 %", 10.5, ancre="start", couleur=NIVEAU["f"][1], brut=True))
    # une ligne par famille, puis les points
    familles: Dict[str, List[Dict]] = {}
    for p in sorted(pts, key=lambda p: p["masse"]):
        familles.setdefault(p["famille"], []).append(p)
    fins = []
    for fam, ps in familles.items():
        if len(ps) > 1:
            g.append(f'<polyline points="{" ".join(f"{_f(X(p["masse"]))},{_f(Y(p["pct"]))}" for p in ps)}" fill="none" stroke="{GRIS}" '
                     f'stroke-width="1.4" stroke-linejoin="round"/>')
        der = ps[-1]
        fins.append([Y(der["pct"]), X(der["masse"]), fam])
    for p in pts:
        x, y = X(p["masse"]), Y(p["pct"])
        aide = f"<title>{escape(p['aide'])}</title>"
        if p["pct"] > ymax:
            g.append(f'<polygon points="{_f(x)},{_f(y - 1)} {_f(x - 4.5)},{_f(y + 7)} {_f(x + 4.5)},{_f(y + 7)}" fill="{GRIS}">{aide}</polygon>')
        else:
            fond, bord = NIVEAU.get(p["niveau"], NIVEAU["p"])
            g.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="4.2" fill="{fond}" stroke="{bord}" stroke-width="1">{aide}</circle>')
    # les noms des familles dans la marge de droite, sans chevauchement
    fins.sort()
    dernier = -1e9
    for f in fins:
        f.append(max(f[0], dernier + 14))
        dernier = f[3]
    for y, x, fam, yl in fins:
        g.append(_ligne(x + 5, y, L + pw + 40, yl - 4, GRILLE, 1))
        g.append(_txt(L + pw + 44, yl, fam, 11, ancre="start", couleur=MUET, gras=True, brut=True))
    # le retenu et le plus léger qui tient
    par_nom = {p["nom"]: p for p in pts}
    marques = []
    # le plus léger qui tient sous le retenu (le cas courant : l'optimal de l'écran est le plus chargé) : son étiquette va
    # SOUS l'étoile, celle du retenu au-dessus de son cercle — chacune de son côté, hors de la bande 95–100 %
    sous = bool(retenu and leger and retenu != leger and retenu in par_nom and leger in par_nom
                and par_nom[leger]["pct"] < par_nom[retenu]["pct"])
    if leger and leger in par_nom:
        p = par_nom[leger]
        g.append(_etoile(X(p["masse"]), Y(p["pct"]), 9))
        marques.append((p, f"★ {p['nom']} · {_nb(p['pct'], 0)} %", ETOILE, sous))
    if retenu and retenu in par_nom:
        p = par_nom[retenu]
        g.append(f'<circle cx="{_f(X(p["masse"]))}" cy="{_f(Y(p["pct"]))}" r="8.5" fill="none" stroke="{ACCENT}" stroke-width="2.2"/>')
        marques.append((p, f"✓ {p['nom']} · {_nb(p['pct'], 0)} %", ACCENT, False))
    occupe = []
    for p, t, c, dessous in marques:
        x, y = X(p["masse"]), Y(p["pct"])
        larg = len(t) * 6.6
        ancre, xl = ("start", x + 12) if x + 12 + larg < L + pw else ("end", x - 12)
        yl = y + 22 if (dessous and y + 22 < T + ph - 4) or y - 12 <= T + 12 else y - 12
        for yo in occupe:
            if abs(yl - yo) < 15:
                yl = yo + 17 if yl >= yo else yo - 17
        occupe.append(yl)
        g.append(_etiquette(xl, yl, t, ancre, ENCRE if c == ACCENT else "#7A5600"))
    # les titres des axes et la légende
    g.append(_txt(L + pw / 2, H - 10, "Masse (kg/m)", 11.5, couleur=MUET, brut=True))
    g.append(_txt(16, T + ph / 2, "Taux η (%)", 11.5, couleur=MUET, brut=True, extra=f' transform="rotate(-90 16 {_f(T + ph / 2)})"'))
    lg, x = [], L
    for niv, t in (("p", "< 95 %"), ("f", "95 – 100 %"), ("r", "> 100 %")):
        fond, bord = NIVEAU[niv]
        lg.append(f'<circle cx="{_f(x + 5)}" cy="18" r="4.6" fill="{fond}" stroke="{bord}" stroke-width="1"/>'
                  + _txt(x + 14, 22, t, 11, ancre="start", couleur=MUET, brut=True))
        x += 26 + len(t) * 6.2
    lg.append(f'<circle cx="{_f(x + 6)}" cy="18" r="7" fill="none" stroke="{ACCENT}" stroke-width="2"/>' + _txt(x + 18, 22, "retenu", 11, ancre="start", couleur=MUET, brut=True))
    x += 72
    lg.append(_etoile(x + 6, 18, 7) + _txt(x + 18, 22, "le plus léger qui tient", 11, ancre="start", couleur=MUET, brut=True))
    x += 158
    if any(p["pct"] > ymax for p in pts):
        lg.append(f'<polygon points="{_f(x + 6)},11 {_f(x + 1.5)},{_f(19 + 4)} {_f(x + 10.5)},{_f(19 + 4)}" fill="{GRIS}"/>'
                  + _txt(x + 18, 22, f"au-delà de {_nb(ymax, 0)} %", 11, ancre="start", couleur=MUET, brut=True))
    g += lg
    return (f'<svg id="{escape(ident)}" class="pf-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(W)} {_f(H)}" width="100%" role="img" '
            f'aria-label="Taux de chaque profilé en fonction de sa masse">{"".join(g)}</svg>')


def flambement(ident: str, courbe_cle: str, alphas: Sequence[Tuple[str, float]], axes: List[Dict]) -> str:
    """Les cinq courbes de flambement χ(λ̄) de l'écran (celle retenue en couleur) et les points du profilé retenu.
    ``axes`` : axe, lambdabar, chi, texte (la bulle et l'étiquette)."""
    W, H = 760.0, 600.0
    L, R, T, B = 60.0, 40.0, 30.0, 50.0
    pw, ph = W - L - R, H - T - B
    lmax = max([2.5] + [a["lambdabar"] * 1.15 for a in axes])
    lmax, xt = _graduations(lmax, 6)

    def X(v):
        return L + pw * v / lmax

    def Y(c):
        return T + ph * (1 - c / 1.1)
    g = [f'<rect x="{_f(L)}" y="{_f(T)}" width="{_f(pw)}" height="{_f(ph)}" fill="#fff"/>']
    for c in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        g.append(_ligne(L, Y(c), L + pw, Y(c), GRILLE, 1))
        g.append(_txt(L - 8, Y(c) + 4, _nb(c, 1), 11, ancre="end", couleur=MUET, brut=True))
    for v in xt:
        g.append(_ligne(X(v), T + ph, X(v), T + ph + 4, GRIS, 1))
        g.append(_txt(X(v), T + ph + 18, _nb(v, 1), 11, couleur=MUET, brut=True))
    g.append(_ligne(L, T + ph, L + pw, T + ph, GRIS, 1))
    n = 120
    for cle, alpha in alphas:
        pts = " ".join(f"{_f(X(lmax * k / n))},{_f(Y(CA.chi_courbe(lmax * k / n, alpha)[1]))}" for k in range(n + 1))
        choisie = cle == courbe_cle
        g.append(f'<polyline points="{pts}" fill="none" stroke="{ACCENT if choisie else GRIS}" stroke-width="{2.4 if choisie else 1.1}"/>')
        xe = min(1.35, lmax * 0.6)
        ye = Y(CA.chi_courbe(xe, alpha)[1])
        g.append(_etiquette(X(xe) + 10, ye + 4, cle, "start", ACCENT if choisie else MUET, 12 if choisie else 11, choisie))
    # la légende des courbes, en haut à droite (là où les courbes ne passent pas) : α de chacune, la retenue en couleur
    xl, yl = L + pw - 196, T + 22
    g.append(f'<rect x="{_f(xl - 12)}" y="{_f(yl - 18)}" width="200" height="{_f(26 + 19 * len(alphas))}" rx="6" fill="#fff" stroke="{GRILLE}"/>')
    g.append(_txt(xl, yl, "Courbes (EN 1993-1-1 §6.3.1.2)", 10.5, ancre="start", couleur=MUET, gras=True))
    for i, (cle, alpha) in enumerate(alphas):
        y = yl + 19 * (i + 1)
        choisie = cle == courbe_cle
        g.append(_ligne(xl, y - 4, xl + 24, y - 4, ACCENT if choisie else GRIS, 2.4 if choisie else 1.1))
        g.append(_txt(xl + 32, y, f"{cle} · α = {_nb(alpha, 2)}" + (" · retenue" if choisie else ""), 11, ancre="start",
                      couleur=ACCENT if choisie else ENCRE, gras=choisie))
    for a in axes:
        x, y = X(a["lambdabar"]), Y(a["chi"])
        g.append(_ligne(x, y, x, T + ph, ACCENT, 1, ' stroke-dasharray="4 3"'))
        g.append(_ligne(L, y, x, y, ACCENT, 1, ' stroke-dasharray="4 3"'))
        g.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="5.5" fill="{ACCENT}" stroke="#fff" stroke-width="1.6"><title>{escape(a["texte"])}</title></circle>')
        g.append(_etiquette(x + 10, y - 10, a["texte"], "start", ENCRE, 11.5))
    g.append(_txt(L + pw / 2, H - 10, "Élancement réduit λ̄", 11.5, couleur=MUET))
    g.append(_txt(16, T + ph / 2, "Coefficient de réduction χ", 11.5, couleur=MUET, brut=True,
                  extra=f' transform="rotate(-90 16 {_f(T + ph / 2)})"'))
    return (f'<svg id="{escape(ident)}" class="pf-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(W)} {_f(H)}" width="100%" role="img" '
            f'aria-label="Courbes de flambement et points du profilé retenu">{"".join(g)}</svg>')

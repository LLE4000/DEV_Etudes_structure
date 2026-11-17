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
# (« v » : à vérifier — une méthode hors de son domaine (lot A) : l'ocre de la pastille « v » de l'écran, jamais la couleur
# « conforme » ; copie NOMMÉE de --vrf et --vrf-encre de site/web/css/site.css, liée par tests/test_api_web_profil.py)
NIVEAU = {"p": ("#86EFAC", "#166534"), "f": ("#15803D", "#0F5A2C"), "r": ("#DC2626", "#991B1B"), "v": ("#B86E00", "#8A5A00")}
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
    if not texte:                                  # (le croquis d'un PRS place lui-même le texte de ses cotes)
        return out
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


# ------------------------------------------------------------------ le croquis coté d'un PRS (brief v12 § 15.5, § 18.7)
LARGEUR_CAR = 0.56              # la largeur moyenne d'un caractère, en tailles de police (pour placer les textes sans recouvrement)


def _visible(t: str) -> str:
    """Le texte qu'on LIT d'un symbole (« t_w = 15 » → « tw = 15 ») : la place qu'il prend."""
    return t.replace("_", "")


def _marque_g(x: float, y: float, u: float) -> str:
    """Le centre de gravité : un cercle blanc barré d'une croix (un symbole, pas une couleur)."""
    r = 1.7 * u
    return (f'<g data-repere="G" data-cx="{_f(x)}" data-cy="{_f(y)}"><circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="#fff" '
            f'stroke="{ENCRE}" stroke-width="{_f(0.32 * u)}"/>'
            + _ligne(x - r, y, x + r, y, ENCRE, 0.26 * u) + _ligne(x, y - r, x, y + r, ENCRE, 0.26 * u) + "</g>")


def _marque_c(x: float, y: float, u: float) -> str:
    """Le centre de cisaillement : un losange plein (un autre symbole que celui de G), posé PAR-DESSUS : sur un PRS
    bisymétrique, C est en G — le losange se lit dans le cercle."""
    d = 1.25 * u
    return (f'<path data-repere="C" data-cx="{_f(x)}" data-cy="{_f(y)}" d="M{_f(x)},{_f(y - d)} L{_f(x + d)},{_f(y)} L{_f(x)},{_f(y + d)} '
            f'L{_f(x - d)},{_f(y)} Z" fill="{ACCENT}" stroke="{ENCRE}" stroke-width="{_f(0.2 * u)}"/>')


def section_prs(s: Dict, largeur=None, ident: str = "pf_section", tableau=None) -> str:
    """Le CROQUIS COTÉ d'un PRS, à l'échelle (mm) : ses trois plats hachurés, les quatre cordons (côté s = a·√2), les axes
    y-y et z-z PAR LE CENTRE DE GRAVITÉ (un PRS monosymétrique ne l'a pas à mi-hauteur), le centre de gravité G et le
    centre de cisaillement C — chacun une lettre ET un symbole ; les cotes h, b₁, b₂ et les repères t₁, h_w × t_w, a,
    t₂ ; « mm » et la méthode sous le croquis. ``s`` : la section du moteur (acier/barre/prs.section_prs) ; ``largeur`` :
    la largeur du dessin (None : celle de la vue) ; ``tableau`` : [(symbole, valeur, unité, origine)] écrit dessous (la
    vue « Section » de l'écran : api_web/profil_prs.proprietes) — sans lui, le croquis seul (la note écrit son tableau).
    Repère du dessin : x vers la droite depuis le plan de l'âme, y = −z (z depuis la face inférieure, vers le haut)."""
    p = s["prs"]
    b1, t1, hw, tw, b2, t2 = (float(p[k]) for k in ("b1", "t1", "hw", "tw", "b2", "t2"))
    h, zg, zc, sv = s["h"], s["z_g"], s["z_g"] + s["z_s"], float(s.get("s") or 0.0)
    bm = max(b1, b2)
    u = max(bm, h) / 100.0
    boites: List[Tuple[float, float, float, float]] = []      # (x1, y1, x2, y2) des textes et des marques : le cadre du dessin
    g: List[str] = []

    def texte(x, y, t, fs, ancre="middle", couleur=ENCRE, gras=False, extra="", rot=False, brut=False):
        w = len(_visible(t)) * LARGEUR_CAR * fs * (1.12 if gras else 1.0)
        x1 = x - w / 2 if ancre == "middle" else (x - w if ancre == "end" else x)
        boites.append((x - 0.8 * fs, y - w / 2, x + 0.2 * fs, y + w / 2) if rot else (x1, y - 0.8 * fs, x1 + w, y + 0.2 * fs))
        g.append(_txt(x, y, t, fs, ancre=ancre, couleur=couleur, gras=gras, extra=extra, brut=brut))

    # la matière : le contour du I, rempli, hachuré, cerné ; les joints des trois plats ; les cordons
    yh, yb = -(t2 + hw), -t2                                  # le dessous de la semelle haute, le dessus de la semelle basse
    contour = [(-b1 / 2, -h), (b1 / 2, -h), (b1 / 2, yh), (tw / 2, yh), (tw / 2, yb), (b2 / 2, yb), (b2 / 2, 0.0), (-b2 / 2, 0.0),
               (-b2 / 2, yb), (-tw / 2, yb), (-tw / 2, yh), (-b1 / 2, yh)]
    d = "M" + " L".join(f"{_f(x)},{_f(y)}" for x, y in contour) + " Z"
    g.append(f'<path d="{d}" fill="{ACIER}" stroke="none"/>')
    g.append("".join(_ligne(a, b_, c, d_, HACH, 0.17 * u) for a, b_, c, d_ in _hachures([contour], 2.3 * u)))
    g.append(f'<path d="{d}" fill="none" stroke="{ENCRE}" stroke-width="{_f(0.55 * u)}" stroke-linejoin="round"/>')
    g.append(_ligne(-tw / 2, yh, tw / 2, yh, ENCRE, 0.3 * u) + _ligne(-tw / 2, yb, tw / 2, yb, ENCRE, 0.3 * u))
    if sv > 0:
        for yj, sens in ((yh, 1.0), (yb, -1.0)):
            for cote in (1.0, -1.0):
                x0 = cote * tw / 2
                g.append(f'<path d="M{_f(x0)},{_f(yj)} L{_f(x0 + cote * sv)},{_f(yj)} L{_f(x0)},{_f(yj + sens * sv)} Z" fill="{ENCRE}"/>')
    # les axes PAR LE CENTRE DE GRAVITÉ ; G et C sur l'axe de symétrie (le plan de l'âme)
    tiret = f' stroke-dasharray="{_f(5 * u)} {_f(1.4 * u)} {_f(0.7 * u)} {_f(1.4 * u)}"'
    g.append(_ligne(-bm / 2 - 5 * u, -zg, bm / 2 + 5 * u, -zg, AXE, 0.22 * u, tiret + ' data-axe="y"'))
    g.append(_ligne(0, -h - 5 * u, 0, 5 * u, AXE, 0.22 * u, tiret + ' data-axe="z"'))
    texte(bm / 2 + 6.2 * u, -zg + 1.5 * u, "y", 4.4 * u, ancre="start", couleur=AXE, extra=' font-style="italic"')
    texte(1.6 * u, -h - 5.6 * u, "z", 4.4 * u, ancre="start", couleur=AXE, extra=' font-style="italic"')
    g.append(_marque_g(0.0, -zg, u))
    g.append(_marque_c(0.0, -zc, u))
    texte(-2.6 * u, -zg - 2.0 * u, "G", 4.4 * u, ancre="end", gras=True)
    texte(2.6 * u, -zc - 2.0 * u, "C", 4.4 * u, ancre="start", gras=True, couleur=ACCENT)
    # les cotes : h à gauche ; b₁ au-dessus, b₂ dessous
    xh = -bm / 2 - 11 * u
    g.append(_attache(-b1 / 2 - 1.2 * u, -h, xh - 1.5 * u, -h, u) + _attache(-b2 / 2 - 1.2 * u, 0.0, xh - 1.5 * u, 0.0, u))
    g.append(_cote(xh, -h, xh, 0.0, "", u))
    texte(xh - 1.6 * u, -h / 2, f"h = {_brut(h)}", 4.4 * u, extra=f' transform="rotate(-90 {_f(xh - 1.6 * u)} {_f(-h / 2)})"', rot=True)
    yc1, yc2 = -h - 12 * u, 10 * u
    g.append(_attache(-b1 / 2, -h - 1.2 * u, -b1 / 2, yc1 - 1.5 * u, u) + _attache(b1 / 2, -h - 1.2 * u, b1 / 2, yc1 - 1.5 * u, u))
    g.append(_cote(-b1 / 2, yc1, b1 / 2, yc1, "", u))
    texte(0.0, yc1 - 1.6 * u, f"b_1 = {_brut(b1)}", 4.4 * u)
    g.append(_attache(-b2 / 2, 1.2 * u, -b2 / 2, yc2 + 1.5 * u, u) + _attache(b2 / 2, 1.2 * u, b2 / 2, yc2 + 1.5 * u, u))
    g.append(_cote(-b2 / 2, yc2, b2 / 2, yc2, "", u))
    texte(0.0, yc2 + 5.4 * u, f"b_2 = {_brut(b2)}", 4.4 * u)
    # les repères, alignés à droite, de haut en bas : t₁, la gorge, l'âme, t₂
    gorge = f"a = {_brut(p['a'])}" if p.get("a") is not None else "a non saisie (s = 0)"
    # (le repère de l'âme, loin de l'axe y : son texte et la lettre « y » ne se touchent pas)
    places = [yb - f * hw for f in (0.38, 0.22, 0.6)]
    y_ame = next((y_ for y_ in places if abs(y_ + zg) >= 9 * u), max(places, key=lambda y_: abs(y_ + zg)))
    reperes = [(b1 / 2 - min(0.18 * b1, 9 * u), -h + t1 / 2, f"t_1 = {_brut(t1)}"),
               (tw / 2 + sv / 3, yh + sv / 3, gorge),
               (tw / 2, y_ame, f"h_w × t_w = {_brut(hw)} × {_brut(tw)}"),
               (b2 / 2 - min(0.18 * b2, 9 * u), -t2 / 2, f"t_2 = {_brut(t2)}")]
    xr, yt = bm / 2 + 11 * u, -h - 8.0 * u
    for xp, yp, t in reperes:
        yt = max(yt + 7.2 * u, yp)
        fs = 4.2 * u
        g.append(f'<circle cx="{_f(xp)}" cy="{_f(yp)}" r="{_f(0.55 * u)}" fill="{ENCRE}"/>'
                 f'<polyline points="{_f(xp)},{_f(yp)} {_f(xr - 0.8 * u)},{_f(yt)} {_f(xr)},{_f(yt)}" fill="none" stroke="{COTE}" stroke-width="{_f(0.18 * u)}"/>')
        texte(xr + 0.8 * u, yt + 0.36 * fs, t, fs, ancre="start")
    # le nom au-dessus ; sous le croquis, la légende des symboles, puis « mm » et la méthode
    texte(0.0, yc1 - 14 * u, s["nom"], 6.4 * u, gras=True)
    texte(0.0, yc1 - 8 * u, "PRS monosymétrique" if s["monosymetrique"] else "PRS bisymétrique", 3.9 * u, couleur=MUET)
    yl = max(yc2 + 12 * u, yt + 6 * u)
    x0 = -bm / 2 - 8 * u
    g.append(_marque_g(x0 + 1.7 * u, yl - 1.3 * u, u))
    texte(x0 + 4.6 * u, yl, "G : centre de gravité", 3.8 * u, ancre="start", couleur=MUET)
    x1 = x0 + 4.6 * u + len("G : centre de gravité") * LARGEUR_CAR * 3.8 * u + 4 * u
    g.append(_marque_c(x1 + 1.7 * u, yl - 1.3 * u, u))
    texte(x1 + 4.6 * u, yl, "C : centre de cisaillement", 3.8 * u, ancre="start", couleur=MUET)
    texte(x0, yl + 6.2 * u, "Cotes en mm · trois plats, cordons non comptés (côté sûr)", 3.8 * u, ancre="start", couleur=MUET)
    y = yl + 6.2 * u
    if tableau:
        y += 9 * u
        texte(x0, y, "PROPRIÉTÉS", 3.3 * u, ancre="start", couleur=MUET, gras=True, extra=' letter-spacing="0.6"', brut=True)
        fs, pas = 4.1 * u, 6.4 * u
        y += 2.0 * u
        largeur_t = 134 * u
        g.append(_ligne(x0, y, x0 + largeur_t, y, GRILLE, 0.25 * u))
        y += pas
        for sym, val, unite, origine in tableau:
            texte(x0, y - 1.6 * u, sym, fs, ancre="start", couleur=MUET)
            texte(x0 + 45 * u, y - 1.6 * u, val, fs, ancre="end", gras=True, brut=True)
            if unite:
                texte(x0 + 46.6 * u, y - 1.6 * u, unite, 3.6 * u, ancre="start", couleur=MUET, brut=True)
            texte(x0 + 60 * u, y - 1.6 * u, origine, 3.6 * u, ancre="start", couleur=MUET)
            g.append(_ligne(x0, y + 0.4 * u, x0 + largeur_t, y + 0.4 * u, GRILLE, 0.16 * u))
            y += pas
        boites.append((x0, y, x0 + largeur_t, y))
    gauche = min([xh - 8 * u] + [b[0] for b in boites]) - 2 * u
    droite = max([bm / 2 + 8 * u] + [b[2] for b in boites]) + 2 * u
    haut = min(b[1] for b in boites) - 2 * u
    bas = max(b[3] for b in boites) + 2 * u
    vb = f"{_f(gauche)} {_f(haut)} {_f(droite - gauche)} {_f(bas - haut)}"
    larg = "100%" if largeur is None else _f(float(largeur))
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{larg}" role="img" '
            f'aria-label="Section {escape(s["nom"])} à l\'échelle, G et C repérés">{"".join(g)}</svg>')

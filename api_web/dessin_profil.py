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
from typing import Dict, List, Optional, Sequence, Tuple

from acier.profiles import familles as FA

from . import profil_mots as M

ENCRE, ACIER, HACH, COTE, AXE, EFFORT = "#15181F", "#DCE2E9", "#8794A3", "#5B6573", "#7A8290", "#9C3341"
GRIS, GRILLE, MUET, ACCENT, ETOILE = "#A7B0BD", "#E7EAEF", "#6B7280", "#2B4BCF", "#E0A100"
CISAILLEMENT = "#0F766E"         # l'effort tranchant (le diagramme de V de la barre) et le cisaillement (le dessin des contraintes)
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


def largeur_gras(texte: str, fs: float) -> float:
    """La largeur d'un texte gras, estimée (un dessin n'a pas la métrique de la police) : 0,58 du corps par caractère — celle du
    dessin de la barre et du dessin des contraintes."""
    return 0.58 * fs * len(texte)


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


# ------------------------------------------------------------------ les CONTRAINTES dans la section (mise à jour 1, étape 2)
# Un à trois VOLETS côte à côte. Un volet = LA COUPE du profilé, à l'échelle, peinte par bandes (la teinte monte avec la
# contrainte rapportée à sa limite), et LE DIAGRAMME de la contrainte le long de la hauteur : sa plus grande valeur en gras,
# sa limite en trait interrompu. Rien n'est recalculé ici : chaque volet lit les grandeurs de SA vérification
# (``v["section"]`` du moteur). L'unité est le point de la note ; à l'écran, le composer à 450 de large (hs = 110).
COMPRESSION, TRACTION, EQUIVALENTE = EFFORT, ACCENT, "#33415C"     # les couleurs des diagrammes du module (CISAILLEMENT : en tête)
N_BANDES = 40                    # les bandes d'une coupe peinte : le peintre du kit des notes ne peint pas de dégradé
K_LIMITE = 60.0                  # entre la base d'un diagramme et le trait de sa limite, au plus
ECART_VOLETS = 12.0              # entre deux volets
PART_COUPE = 0.37                # la largeur que la coupe peut prendre dans son volet, au plus
H_TETE, Y_CARTE = 15.0, 13.0     # la tête d'un volet (titre, lieu, filet) ; la ligne des valeurs au-dessus de la fibre haute
LIGNE_LIMITE = 9.5               # la ligne de plus du nom d'une limite qui ne tient pas à côté de la valeur de l'autre fibre
                                 # (7 : les boîtes des lettres se touchaient encore à l'écran — le harnais)
MARGE_NOM = 6.0                  # l'écart estimé sous lequel il n'y tient pas (PRS-C4 : 10 à l'écran, 17 dans la note)
FORMES_CONTRAINTES = ("I", "tube_carre", "tube_rect")    # les formes qui ont un dessin (un tube rond : aucun)


def _teinte(couleur: str, t: float) -> str:
    """Du blanc (t = 0) à la couleur (t = 1) : une bande de la coupe."""
    c = [int(couleur[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(255 + (v - 255) * (0.08 + 0.92 * max(0.0, min(1.0, t)))):02X}" for v in c)


class Coupe:
    """La section vue par le dessin : des PLATS (x1, x2, z1, z2) en mm — x depuis l'axe de l'âme, z depuis la face
    inférieure —, à l'échelle, dans une hauteur ``hs`` et une largeur ``large`` au plus, depuis le bord gauche ``x0``.
    I et H laminés (les congés ne sont pas dessinés), PRS à semelles égales ou non, tubes carrés et rectangulaires."""

    def __init__(self, s: Dict, x0: float, y0: float, hs: float, large: float = 38.0):
        p = s.get("prs") if isinstance(s.get("prs"), dict) else None
        h = self.h = float(s["h"])
        if s["forme"] == "I":
            b1, t1, tw, b2, t2 = (float(p[k]) for k in ("b1", "t1", "tw", "b2", "t2")) if p else (float(s[k]) for k in ("b", "tf", "tw", "b", "tf"))
            self.plats = [(-b2 / 2, b2 / 2, 0.0, t2), (-tw / 2, tw / 2, t2, h - t1), (-b1 / 2, b1 / 2, h - t1, h)]
            self.t_bas, self.t_haut = t2, t1
            self.contours = [[(-b1 / 2, h), (b1 / 2, h), (b1 / 2, h - t1), (tw / 2, h - t1), (tw / 2, t2), (b2 / 2, t2), (b2 / 2, 0), (-b2 / 2, 0),
                              (-b2 / 2, t2), (-tw / 2, t2), (-tw / 2, h - t1), (-b1 / 2, h - t1)]]
        elif s["forme"] in ("tube_carre", "tube_rect"):
            b, t = float(s["b"]), float(s["t"])
            self.plats = [(-b / 2, b / 2, 0.0, t), (-b / 2, -b / 2 + t, t, h - t), (b / 2 - t, b / 2, t, h - t), (-b / 2, b / 2, h - t, h)]
            self.t_bas = self.t_haut = t
            self.contours = [[(-b / 2, 0), (b / 2, 0), (b / 2, h), (-b / 2, h)], [(-b / 2 + t, t), (b / 2 - t, t), (b / 2 - t, h - t), (-b / 2 + t, h - t)]]
        else:
            raise ValueError(f"pas de coupe peinte pour la forme « {s['forme']} »")
        self.zg, self.Iy, self.A = float(s.get("z_g") or h / 2.0), float(s["Iy"]), float(s["A"])
        b_max = max(x2 for _x1, x2, _z1, _z2 in self.plats) - min(x1 for x1, _x2, _z1, _z2 in self.plats)
        self.e = min(hs / h, large / b_max)                       # pt par mm
        self.hs = h * self.e
        self.y0 = y0 + (hs - self.hs) / 2.0
        self.xc, self.x1 = x0 + b_max * self.e / 2.0, x0 + b_max * self.e
        zs = sorted({round(h * i / N_BANDES, 6) for i in range(N_BANDES + 1)} | {self.t_bas, h - self.t_haut, round(self.zg, 6)})
        self.bandes = list(zip(zs[:-1], zs[1:]))                  # (une bande est dans UN étage de plats, d'UN côté de l'axe)

    def Y(self, z: float) -> float:
        """L'ordonnée du dessin d'une cote z (mm depuis la face inférieure)."""
        return self.y0 + (self.h - z) * self.e

    def ame(self, z: float) -> bool:
        return self.t_bas < z < self.h - self.t_haut

    def peinte(self, couleur_de) -> List[str]:
        """La coupe peinte bande par bande (``couleur_de(z)`` au milieu de la bande), puis son contour."""
        g = []
        for za, zb in self.bandes:
            zm = (za + zb) / 2.0
            for x1, x2, z1, z2 in self.plats:
                if z1 <= zm <= z2:
                    larg = max((x2 - x1) * self.e, 2.0)           # (un plat d'au moins 2 pt : une âme mince se voit)
                    g.append(f'<rect x="{self.xc + (x1 + x2) / 2 * self.e - larg / 2:.2f}" y="{self.Y(zb):.2f}" width="{larg:.2f}" '
                             f'height="{(zb - za) * self.e + 0.15:.2f}" fill="{couleur_de(zm)}"/>')
        for c in self.contours:
            g.append('<path d="M' + " L".join(f"{self.xc + x * self.e:.2f},{self.Y(z):.2f}" for x, z in c)
                     + f' Z" fill="none" stroke="{ENCRE}" stroke-width="0.45" stroke-linejoin="round"/>')
        return g

    def profil(self, xb: float, K: float, valeur, couleur_de) -> List[str]:
        """UN DIAGRAMME le long de la hauteur : depuis la base ``xb``, ``valeur(z)`` portée à droite (positive) ou à gauche
        (négative), ``K`` points par N/mm² ; chaque bande peinte par ``couleur_de(z)`` ; puis son contour et sa base."""
        g, bord, eps = [], [(xb, self.Y(0.0))], 1e-6 * self.h
        for za, zb in self.bandes:
            va, vb = valeur(za + eps), valeur(zb - eps)
            if abs(va) + abs(vb) > 0:
                g.append(f'<polygon points="{xb:.2f},{self.Y(za) + 0.1:.2f} {xb + va * K:.2f},{self.Y(za) + 0.1:.2f} {xb + vb * K:.2f},{self.Y(zb):.2f} '
                         f'{xb:.2f},{self.Y(zb):.2f}" fill="{couleur_de((za + zb) / 2)}"/>')
            bord += [(xb + va * K, self.Y(za)), (xb + vb * K, self.Y(zb))]
        bord.append((xb, self.Y(self.h)))
        g.append('<path d="M' + " L".join(f"{x:.2f},{y:.2f}" for x, y in bord) + f'" fill="none" stroke="{ENCRE}" stroke-width="0.6" stroke-linejoin="round"/>')
        g.append(_ligne(xb, self.y0 - 2.5, xb, self.y0 + self.hs + 2.5, ENCRE, 0.8))
        return g


def _habillage(K_: Coupe, xb: float, kd: float, limite, x_fin: float, valeur: float, en_haut: bool, couleur: str,
               occupe: Optional[Tuple[float, float]] = None) -> Tuple[List[str], Tuple[float, float]]:
    """Ce que les volets ont en commun : les rappels des deux fibres, l'axe du centre de gravité, LA LIMITE (``limite`` =
    (symbole, valeur)) en trait interrompu, et la plus grande valeur, en gras, près de sa fibre ; le nom de la limite est
    à l'autre bout de son trait : ils ne se touchent jamais. ``occupe`` : l'étendue (x) d'un texte déjà posé sur la ligne du
    nom (la valeur de l'autre fibre, à gauche de la base) : quand l'écart ESTIMÉ entre les deux est sous ``MARGE_NOM``, le nom
    passe à la ligne suivante, loin de la coupe (la marge couvre la police de l'écran, plus large que l'estimation). Rend les
    éléments et la hauteur que la ligne de plus demande (sous le volet, au-dessus) : (0 ou ``LIGNE_LIMITE``, 0 ou ``LIGNE_LIMITE``).
    Au-dessus, la rangée la fait en descendant (``_rangee``)."""
    y0, y1, yn = K_.y0, K_.y0 + K_.hs, K_.Y(K_.zg)
    xl = xb + limite[1] * kd
    g = [_ligne(K_.x1 + 2, y, xb - 1.5, y, GRIS, 0.3, ' stroke-dasharray="1.2 1.4"') for y in (y0, y1)]
    g.append(_ligne(K_.xc - (K_.x1 - K_.xc) - 2.0, yn, min(max(xl, xb + valeur * kd) + 5.0, x_fin), yn, AXE, 0.4, ' stroke-dasharray="5 1.6 1 1.6"'))
    g.append(_ligne(xl, y0 - 2.5, xl, y1 + 2.5, ENCRE, 0.7, ' stroke-dasharray="2.2 1.6"'))
    y_val, y_lim = (y0 - 3.6, y1 + 8.8) if en_haut else (y1 + 8.8, y0 - 3.6)
    g.append(_txt(xb + 2.0, y_val, M.r1(valeur), 8.0, "start", couleur, True))
    nom = f"{limite[0]} = {M.r1(limite[1])}"
    demi = largeur_gras(nom, 6.2) / 2.0
    # (le nom de la limite reste à droite de la base : à gauche, à la même hauteur, s'écrit la valeur de l'autre fibre — trouvé par
    #  le harnais, une poutre symétrique : « σ_adm,d = 95,3 » sur « 84,2 » ; et il ne sort pas du volet)
    x_nom = min(max(xl, xb + 1.0 + demi), x_fin - 1.0 - demi)
    if occupe is not None and x_nom - demi < occupe[1] + MARGE_NOM:
        # (le harnais, deux fois : une poutre au-delà de sa limite — le diagramme prend la place, le nom ne tient plus à droite
        #  de la base, « σ_adm,d = 95,3 » sur « 111,1 » ; puis, la police réelle plus large que l'estimée, « σ_adm,d = 60,5 » sur
        #  « 107,7 » à 2 d'écart estimé. Pas toujours : la ligne de plus grandit le dessin, et la note de PRS-C4 (écart 17) doit
        #  garder sa mise en page)
        y_lim, extra = (y_lim + LIGNE_LIMITE, (LIGNE_LIMITE, 0.0)) if en_haut else (y_lim - LIGNE_LIMITE, (0.0, LIGNE_LIMITE))
    else:
        extra = (0.0, 0.0)
    g.append(_txt(x_nom, y_lim, nom, 6.2, "middle", ENCRE, True))
    return g, extra


def _volet(s: Dict, vo: Dict, xo: float, w: float, hs: float, decale: float = 0.0) -> Tuple[List[str], Tuple[float, float]]:
    """UN VOLET, depuis l'abscisse ``xo``, sur la largeur ``w`` : sa tête (son titre, son lieu à droite, un filet), la
    coupe peinte, le diagramme ; et la hauteur ajoutée sous le volet (``_habillage``). ``vo`` = {"genre", "titre", "lieu",
    "limite": (symbole, valeur), et les grandeurs du genre} :
        « sigma » : "N" (N, > 0 : compression), "My" (N·mm, > 0 : fibre haute comprimée) ; et "sigma" (N/mm²) quand la
                    formule de la vérification amplifie la contrainte (φ, PE) : le profil est mis à l'échelle de cette valeur ;
                    "N_reel" (N) : l'effort normal de la section quand le dessin ne le porte pas (σ_M, la flexion de l'Eurocode 3)
                    — les mots « comprimée » et « tendue » disent l'état réel ;
        « tau »   : "tau" (N/mm²) ;
        « eq »    : "sigma" (N/mm², la contrainte normale DE LA FORMULE, à la fibre extrême), "tau" (N/mm²), "N" et "My" (la
                    forme de σ(z), à l'échelle de la formule)."""
    lim, x_fin = vo["limite"], xo + w
    corps = min(7.4, 7.4 * (w - largeur_gras(vo.get("lieu") or "", 5.8) * 0.8 - 6.0) / largeur_gras(vo["titre"], 7.4))
    g = [_txt(xo, 8.0, vo["titre"], corps, "start", ENCRE, True)]
    if vo.get("lieu"):
        g.append(_txt(x_fin, 8.0, vo["lieu"], 5.8, "end", MUET))
    g.append(_ligne(xo, 11.2, x_fin, 11.2, GRIS, 0.45))
    K_ = Coupe(s, xo + 3.0, H_TETE + Y_CARTE + decale, hs, PART_COUPE * w)       # (``decale`` : la place d'un nom passé au-dessus)
    y0, y1, place = K_.y0, K_.y0 + K_.hs, x_fin - 4.0 - K_.x1 - 7.0
    if vo["genre"] == "sigma":
        def phys(z: float) -> float:                              # > 0 : compression
            return vo.get("N", 0.0) / K_.A + vo["My"] * (z - K_.zg) / K_.Iy
        k = vo["sigma"] / max(abs(phys(K_.h)), abs(phys(0.0))) if vo.get("sigma") else 1.0              # (une formule qui amplifie)

        def sig(z: float) -> float:
            return k * phys(z)
        haut, bas = sig(K_.h), sig(0.0)
        # la fibre qui gouverne : la plus sollicitée ; à valeurs égales (une section symétrique, à l'arrondi près), la COMPRIMÉE
        ecart = abs(haut) - abs(bas)
        en_haut = ecart > 0 if abs(ecart) > 1e-9 * max(abs(haut), abs(bas), 1.0) else haut >= bas
        gouv, autre = (haut, bas) if en_haut else (bas, haut)
        sens = 1.0 if gouv >= 0 else -1.0                         # la fibre qui gouverne se dessine à droite de la base
        gauche = abs(autre) if autre * gouv < 0 else 0.0
        kd = min(K_LIMITE / lim[1], place / (max(abs(gouv), lim[1]) + gauche))
        xb = K_.x1 + 7.0 + gauche * kd

        def col(z: float) -> str:
            return _teinte(COMPRESSION if sig(z) > 0 else TRACTION, abs(sig(z)) / lim[1])
        g += K_.peinte(col) + K_.profil(xb, kd, lambda z: sens * sig(z), col)
        # l'autre fibre : sa valeur, de son côté de la base — quand le profil est celui des efforts (k = 1). Remis à l'échelle de la
        # formule (φ, PE ; ou une formule qui somme les valeurs absolues), elle ne serait la valeur d'aucune formule : elle n'est
        # pas écrite (relecture : « 43,1 » écrit sur une fibre à 32,8)
        a_ecrire, occupe = None, None
        if abs(autre) > 1e-9 and abs(k - 1.0) <= 5e-4:
            texte, x_a, ancre = M.r1(abs(autre)), xb - 2.0 if gauche else xb + 2.0 + abs(autre) * kd, "end" if gauche else "start"
            if not gauche and x_a + largeur_gras(texte, 7.0) > x_fin - 1.0:     # (du même signe, presque aussi grande : elle
                x_a, ancre = x_fin - 1.0, "end"                                 #  sortirait du volet — relecture, N = −1 000 kN)
            a_ecrire = _txt(x_a, (y1 + 8.8 if en_haut else y0 - 3.6) if gauche else (y1 - 2.0 if en_haut else y0 + 7.0),
                            texte, 7.0, ancre, COMPRESSION if autre > 0 else TRACTION, True)
            occupe = (x_a - largeur_gras(texte, 7.0), x_a) if gauche else None       # (à gauche : sur la ligne du nom de la limite)
        h_, extra = _habillage(K_, xb, kd, lim, x_fin, abs(gouv), en_haut, COMPRESSION if gouv > 0 else TRACTION, occupe)
        g += h_ + ([a_ecrire] if a_ecrire else [])
        # les deux mots (lisible en noir et blanc) disent l'état RÉEL de la fibre : σ_M (H-DEV) et σ de l'Eurocode 3 dessinent la
        # flexion seule — sous N, une fibre « tendue » par la flexion peut être comprimée ; elle n'a alors pas de mot (relecture)
        n_reel = vo.get("N_reel", vo.get("N", 0.0))

        def reel(z: float) -> float:
            return n_reel / K_.A + vo["My"] * (z - K_.zg) / K_.Iy
        for v, z, y in ((haut, K_.h, y0 - 3.6), (bas, 0.0, y1 + 7.8)):
            if abs(v) > 1e-9 and v * reel(z) > 0:
                mot = "comprimée" if v > 0 else "tendue"          # (une coupe élancée : le mot ne sort pas du volet, à gauche)
                g.append(_txt(max(K_.xc, xo + 1.0 + largeur_gras(mot, 5.4) / 2.0), y, mot, 5.4, "middle", MUET))
    elif vo["genre"] == "tau":
        t_ = abs(vo["tau"])

        def tau(z: float) -> float:
            return t_ if K_.ame(z) else 0.0
        xb = K_.x1 + 7.0
        kd = min(K_LIMITE / lim[1], place / max(t_, lim[1]))

        def col(z: float) -> str:
            return _teinte(CISAILLEMENT, tau(z) / lim[1]) if K_.ame(z) else "#FFFFFF"
        h_, extra = _habillage(K_, xb, kd, lim, x_fin, t_, True, CISAILLEMENT)
        g += K_.peinte(col) + K_.profil(xb, kd, tau, col) + h_
    else:
        # σ(z) = N/A + M_y·(z − z_g)/I_y, à l'échelle de la formule (φ, PE) ; τ sur toute la hauteur : le maximum du dessin est σ_eq
        # de la formule. (La relecture : σ·(z − z_g)/c perdait N/A — sous N seul, σ_eq tombait à √3·τ au centre de gravité.)
        t_ = abs(vo["tau"])

        def phys(z: float) -> float:
            return vo.get("N", 0.0) / K_.A + vo.get("My", 0.0) * (z - K_.zg) / K_.Iy
        p_max = max(abs(phys(K_.h)), abs(phys(0.0)))
        k = vo["sigma"] / p_max if p_max > 1e-12 else 0.0

        def eq(z: float) -> float:
            return math.hypot(k * phys(z), math.sqrt(3.0) * t_)
        haut, bas = eq(K_.h), eq(0.0)
        # la valeur en gras à la fibre la plus sollicitée ; à valeurs égales, à la plus COMPRIMÉE — celle du volet « Flexion »
        ecart = haut - bas
        en_haut = ecart > 0 if abs(ecart) > 1e-9 * max(haut, bas, 1.0) else phys(K_.h) >= phys(0.0)
        xb = K_.x1 + 7.0
        kd = min(K_LIMITE / lim[1], place / max(haut, bas, lim[1]))

        def col(z: float) -> str:
            return _teinte(EQUIVALENTE, eq(z) / lim[1])
        h_, extra = _habillage(K_, xb, kd, lim, x_fin, max(haut, bas), en_haut, EQUIVALENTE)
        g += K_.peinte(col) + K_.profil(xb, kd, eq, col) + h_
    return g, extra


def _rangee(s: Dict, volets: Sequence[Dict], largeur: float, hs: float) -> Tuple[List[str], float]:
    """Les éléments d'UNE rangée de volets côte à côte sur ``largeur`` (un filet entre deux), et sa hauteur. Un seul volet garde la
    largeur d'un volet de trois : il ne s'étire pas."""
    n = len(volets)
    w = min((largeur - ECART_VOLETS * (n - 1)) / n, 150.0)
    hs = Coupe(s, 0.0, 0.0, hs, PART_COUPE * w).hs                # (un profilé large : sa coupe est moins haute, le dessin aussi)
    # (un nom de limite passé à la ligne : sous un volet, la rangée grandit ; au-dessus, TOUTE la rangée descend d'une ligne — ses
    #  coupes restent alignées. La décision ne dépend que des abscisses : une première passe la lit)
    decale = max(_volet(s, vo, i * (w + ECART_VOLETS), w, hs)[1][1] for i, vo in enumerate(volets))
    g: List[str] = []
    bas = 0.0
    for i, vo in enumerate(volets):
        xo = i * (w + ECART_VOLETS)
        g_, (b_, _h) = _volet(s, vo, xo, w, hs, decale)
        g, bas = g + g_, max(bas, b_)
    H = H_TETE + Y_CARTE + decale + hs + 12.0 + bas
    for i in range(1, len(volets)):
        xo = i * (w + ECART_VOLETS)
        g.append(_ligne(xo - ECART_VOLETS / 2, 2.0, xo - ECART_VOLETS / 2, H - 2.0, GRIS, 0.35))
    return g, H


def _svg_contraintes(ident: str, largeur: float, H: float, corps: str) -> str:
    return (f'<svg id="{ident}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {largeur:.2f} {H:.2f}" width="100%" role="img" '
            f'aria-label="Les contraintes dans la section : la coupe peinte et le diagramme le long de la hauteur">{corps}</svg>')


def contraintes(ident: str, s: Dict, volets: Sequence[Dict], largeur: float, hs: float = 96.0) -> str:
    """LE DESSIN DES CONTRAINTES (mise à jour 1, étape 2) : ``volets`` (un à trois) côte à côte sur ``largeur`` ; ``hs`` : la
    hauteur de la coupe. L'unité est le point de la note : dans une colonne de note, la largeur de la colonne (hs = 84) ; à
    l'écran, 450 (hs = 110)."""
    g, H = _rangee(s, volets, largeur, hs)
    return _svg_contraintes(ident, largeur, H, "".join(g))


def contraintes_rangees(ident: str, s: Dict, rangees: Sequence[Tuple[str, Sequence[Dict]]], largeur: float, hs: float = 110.0) -> str:
    """Les volets de PLUSIEURS méthodes, une rangée par méthode (« Les deux », à l'écran) : ``rangees`` = [(nom de la méthode,
    volets)] ; le nom n'est écrit, au-dessus de sa rangée, que s'il y a plus d'une rangée."""
    if len(rangees) == 1:
        return contraintes(ident, s, rangees[0][1], largeur, hs)
    g, y = [], 0.0
    for nom, volets in rangees:
        g.append(_txt(0.0, y + 9.0, nom, 8.0, "start", MUET, True))
        y += 13.0
        r, H = _rangee(s, volets, largeur, hs)
        g.append(f'<g transform="translate(0 {y:.2f})">{"".join(r)}</g>')
        y += H + 6.0
    return _svg_contraintes(ident, largeur, y - 6.0, "".join(g))


# ------------------------------------------------------------------ les volets d'une méthode, lus sur ses vérifications
TITRES_VOLETS = {"sigma": "Flexion", "tau": "Cisaillement", "eq": "Contrainte équivalente"}
# la vérification qui porte le volet « Flexion » de chaque méthode, dans l'ordre de préférence (le déversement des classeurs, sinon
# — ou sous un effort normal, que σ_M ne porte pas — la contrainte normale ; l'Eurocode 3 : la flexion, quand elle est vérifiée en
# élastique — le moteur ne rend sa section qu'alors)
FLEXION_DE = {"historique": ("H-DEV", "H-SIGMA"), "ec3": ("E-My",)}
LIMITES_FLEXION = {"H-DEV": "σ_adm,d", "H-SIGMA": "σ_adm", "E-My": "f_y/γ_M0"}


def volets(k: str, x: Dict, s: Dict) -> Dict[str, Dict]:
    """Les volets de la méthode ``k`` (son résultat ``x``) : {"sigma" | "tau" | "eq" : volet}, chacun avec l'``id`` de la
    vérification qui le porte. RIEN N'EST RECALCULÉ : chaque volet lit les grandeurs que sa vérification rend de SA section
    (``v["section"]`` du moteur) et sa limite. Aucun volet pour une forme sans coupe peinte (un tube rond), ni pour une section
    en flexion d'axe faible (M_z ≠ 0) ; ni cisaillement ni équivalente pour une section cisaillée selon y (V_y ≠ 0)."""
    if (s or {}).get("forme") not in FORMES_CONTRAINTES:
        return {}
    par_id = {v["id"]: v for v in x.get("verifications") or ()}

    def droite(v: Optional[Dict], cisaille: bool = False) -> Optional[Dict]:
        # (M_z ≠ 0 : pas de dessin ; V_y ≠ 0 n'écarte que le cisaillement et l'équivalente — τ = τ_z + τ_y ne se dessine pas
        #  comme V_z/A_âme — la flexion σ(z) n'en dépend pas : relecture)
        sec = (v or {}).get("section")
        return sec if sec and not sec.get("Mz") and not (cisaille and sec.get("Vy")) else None

    def lieu(v: Dict) -> Optional[str]:
        return f"x = {M.m(v['x'])} m" if v.get("x") is not None else None
    out: Dict[str, Dict] = {}
    for id_ in FLEXION_DE.get(k, ()):
        v = par_id.get(id_)
        sec = droite(v)
        if sec is None or not sec.get("sigma") or (not sec.get("My") and not sec.get("N")):
            continue
        if id_ == "H-DEV" and sec.get("N"):
            # σ_M (H-DEV) ne porte pas N : sous un effort normal, son dessin ne serait pas σ(z) = N/A + M_y·(z − z_g)/I_y — la
            # flexion est alors celle de la contrainte normale, qui le porte (relecture : une fibre comprimée dessinée tendue)
            continue
        lim = v["limite"] if id_ != "E-My" else sec.get("limite")
        if not lim:
            continue
        # σ_M (H-DEV, sans effort normal) et σ de l'Eurocode 3 (M_Ed·z/I_y, § 2.3) : la flexion seule — "N_reel" garde N pour les
        # mots ; H-SIGMA : N/A + M_y·(z − z_g)/I_y, à l'échelle de la formule (φ, PE l'amplifient)
        out["sigma"] = {"genre": "sigma", "titre": TITRES_VOLETS["sigma"], "lieu": lieu(v), "id": id_, "limite": (LIMITES_FLEXION[id_], lim),
                        "N": sec["N"] if id_ == "H-SIGMA" else 0.0, "N_reel": sec["N"] or 0.0, "My": sec["My"], "sigma": sec["sigma"]}
        break
    v = par_id.get("H-EQ") if k == "historique" else None
    sec = droite(v, cisaille=True)
    if sec is not None and sec.get("sigma") is not None and sec.get("tau") is not None and sec.get("repere_tau") and v["limite"]:
        out["tau"] = {"genre": "tau", "titre": TITRES_VOLETS["tau"], "lieu": lieu(v), "id": "H-EQ", "tau": sec["tau"],
                      "limite": ("σ_adm/√3", sec["repere_tau"])}
        out["eq"] = {"genre": "eq", "titre": TITRES_VOLETS["eq"], "lieu": lieu(v), "id": "H-EQ", "sigma": sec["sigma"], "tau": sec["tau"],
                     "N": sec.get("N") or 0.0, "My": sec.get("My") or 0.0, "limite": ("σ_adm", v["limite"])}
    return out


def volets_du_bloc(k: str, v: Dict, vs: Dict[str, Dict]) -> List[Dict]:
    """Les volets qu'un bloc de la NOTE pose sous ses formules : « Flexion » sous la vérification qui la porte ; « Cisaillement »
    et « Équivalente » côte à côte sous la contrainte équivalente. Dans la note, un volet n'écrit pas son abscisse : la première
    ligne du bloc la dit."""
    if v["id"] == "H-EQ" and "eq" in vs:
        return [dict(vs["tau"], lieu=None), dict(vs["eq"], lieu=None, titre="Équivalente")]
    if "sigma" in vs and vs["sigma"]["id"] == v["id"]:
        return [dict(vs["sigma"], lieu=None)]
    return []

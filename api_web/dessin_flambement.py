# -*- coding: utf-8 -*-
"""Les dessins du module « Flambement » : la BARRE vue dans chacun de ses deux plans de flambement (appuis, maintiens,
effort, déformée du mode, longueur de flambement), la SECTION du profilé (contour exact du catalogue, hachures, axes
de flambement et leur courbe, cotes, fiche), les COURBES de flambement χ(λ̄) avec le point de chaque axe.

Rien n'est calculé ici : les grandeurs viennent du moteur (acier/flambement) et le contour d'acier/profiles (source
unique, celle du catalogue). Seules les formes des modes propres (sinus, cosinus, mode encastré–articulé tan βL = βL)
sont des tracés : leur amplitude est amplifiée, c'est dit sur la feuille. La barre et la section sont à l'échelle (mm) ;
les épaisseurs de trait et les textes sont proportionnels à la taille du dessin, pour que la feuille garde la même
allure d'une barre de 1 m à une barre de 15 m.

Style des dessins du garde-corps et de la poutre : traits noirs, acier gris hachuré à 45°, cotes fines à traits
obliques, efforts en rouge. Tout texte venu de l'état (le nom d'une barre) est échappé ici.
"""
import math
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

from acier.flambement import entree as EN
from acier.flambement import normes as NO
from acier.profiles import familles as FA

from . import commun as C

ENCRE, ACIER, HACH, COTE, AXE, EFFORT = "#15181F", "#DCE2E9", "#8794A3", "#5B6573", "#7A8290", "#9C3341"
GRIS, GRILLE, MUET, ACCENT, SOL = "#A7B0BD", "#E7EAEF", "#6B7280", "#2B4BCF", "#C9CFD8"
COUL_AXE = {"y": "#2B4BCF", "z": "#C4532E"}            # y-y en bleu, z-z en brique : les mêmes sur les trois dessins
NIVEAU = {"p": "#166534", "f": "#15803D", "r": "#DC2626", "c": "#6B7280", "x": "#DC2626"}
POLICE = "system-ui,-apple-system,'Segoe UI',Arial,sans-serif"
BETA_EA = 4.493409457909064                        # mode encastré–articulé : tan(βL) = βL


# ------------------------------------------------------------------ petites briques
def _f(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _nb(x: float, nd: int) -> str:
    """Un nombre à la française (virgule, espace fine des milliers)."""
    return C.fr(x, nd)


def _k(k: float) -> str:
    """Un coefficient de longueur : 1,0 · 0,7 · 0,85 · 0,333."""
    s = f"{k:.3f}".rstrip("0")
    return (s + "0" if s.endswith(".") else s).replace(".", ",")


def _sym(s: str, fs: float) -> str:
    """Un texte « N_Ed = 500 kN » écrit avec ses indices (tspan décalé) et λ̄ surligné ; le texte est échappé."""
    out, i, s = "", 0, str(s)
    while i < len(s):
        j = s.find("_", i)
        if j < 0:
            out += escape(s[i:])
            break
        out += escape(s[i:j])
        k = j + 1
        while k < len(s) and (s[k].isalnum() or s[k] == ","):
            k += 1
        if k > j + 1 and s[k - 1] == ",":
            k -= 1
        dy = 0.3 * fs
        out += (f'<tspan font-size="{_f(0.72 * fs)}" dy="{_f(dy)}">{escape(s[j + 1:k])}</tspan>'
                f'<tspan dy="{_f(-dy)}">​</tspan>')
        i = k
    return out.replace("λ̄", '<tspan text-decoration="overline">λ</tspan>')


def _txt(x, y, t, fs, ancre="middle", couleur=ENCRE, gras=False, extra="", brut=False) -> str:
    poids = ' font-weight="650"' if gras else ""
    corps = t if brut else _sym(t, fs)
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(fs)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{corps}</text>')


def _ligne(x1, y1, x2, y2, couleur, lw, extra="") -> str:
    return f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(lw)}"{extra}/>'


def _cote_v(x, y1, y2, texte, u, droite=False, couleur=COTE, fs: Optional[float] = None) -> str:
    """Une cote verticale : la ligne, deux traits obliques, le nombre tourné le long de la ligne."""
    fs = fs or 3.4 * u
    t = 1.1 * u
    out = _ligne(x, y1, x, y2, couleur, 0.2 * u)
    for y in (y1, y2):
        out += _ligne(x - t, y + t, x + t, y - t, ENCRE, 0.28 * u)
    xt = x + (0.25 * fs + 0.8 * fs if droite else -0.4 * fs)
    yt = (y1 + y2) / 2
    return out + _txt(xt, yt, texte, fs, extra=f' transform="rotate(-90 {_f(xt)} {_f(yt)})"')


def _cote_h(x1, x2, y, texte, u, dessous=False, fs: Optional[float] = None) -> str:
    fs = fs or 3.4 * u
    t = 1.1 * u
    out = _ligne(x1, y, x2, y, COTE, 0.2 * u)
    for x in (x1, x2):
        out += _ligne(x - t, y + t, x + t, y - t, ENCRE, 0.28 * u)
    return out + _txt((x1 + x2) / 2, y + (0.4 * fs + 0.75 * fs if dessous else -0.4 * fs), texte, fs)


def _attache(x1, y1, x2, y2, u) -> str:
    return _ligne(x1, y1, x2, y2, COTE, 0.14 * u, f' stroke-dasharray="{_f(1.0 * u)} {_f(0.7 * u)}"')


def _hachures_rect(x0, y0, x1, y1, pas, lw, couleur=HACH) -> str:
    """Des hachures à 45° dans un rectangle (le sol, un encastrement)."""
    out = []
    c = x0 + y0 + pas / 2
    while c < x1 + y1:
        xa, ya = max(x0, c - y1), min(y1, c - x0)
        xb, yb = min(x1, c - y0), max(y0, c - x1)
        if xb > xa:
            out.append(_ligne(xa, ya, xb, yb, couleur, lw))
        c += pas
    return "".join(out)


def _fleche(x1, y1, x2, y2, couleur, lw, tete) -> str:
    """Une flèche de (x1, y1) vers (x2, y2), pointe pleine."""
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    xb, yb = x2 - ux * tete, y2 - uy * tete
    return (_ligne(x1, y1, xb, yb, couleur, lw)
            + f'<path d="M{_f(x2)},{_f(y2)} L{_f(xb + px * tete * 0.42)},{_f(yb + py * tete * 0.42)} '
              f'L{_f(xb - px * tete * 0.42)},{_f(yb - py * tete * 0.42)} Z" fill="{couleur}"/>')


# ------------------------------------------------------------------ les modes de flambement
def mode(appuis: str, maintiens: int, xi: float) -> float:
    """La forme du mode (non normée) au point ξ = x/L compté depuis le PIED de la barre (l'appui nommé en premier)."""
    if appuis == "AA":
        return math.sin((maintiens + 1) * math.pi * xi)
    if appuis == "EA":
        b = BETA_EA
        return math.cos(b * xi) - math.sin(b * xi) / b + xi - 1.0
    if appuis == "EE":
        return 1.0 - math.cos(2.0 * math.pi * xi)
    if appuis == "EG":
        return 0.5 * (1.0 - math.cos(math.pi * xi))
    if appuis == "EL":
        return 1.0 - math.cos(0.5 * math.pi * xi)
    return math.sin(math.pi * xi)                       # k ou L_cr saisis : une demi-onde, à titre d'image


def mode_norme(appuis: str, maintiens: int, n: int = 72) -> List[Tuple[float, float]]:
    """[(ξ, w)] avec max |w| = 1, w ≥ 0 au plus grand écart."""
    pts = [(i / n, mode(appuis, maintiens, i / n)) for i in range(n + 1)]
    m = max(pts, key=lambda p: abs(p[1]))[1] or 1.0
    return [(x, w / m) for x, w in pts]


def troncon_lcr(appuis: str, maintiens: int) -> Optional[Tuple[float, float]]:
    """La part de la barre (ξ₀, ξ₁) entre deux points d'inflexion du mode, longue de L_cr, quand elle tient dans la
    barre : la cote de L_cr s'y pose."""
    if appuis == "AA":
        return (0.0, 1.0 / (maintiens + 1))
    if appuis == "EE":
        return (0.25, 0.75)
    if appuis == "EA":
        return (math.atan(BETA_EA) / BETA_EA, 1.0)
    return None


# ------------------------------------------------------------------ la barre
def _appui_pied(x, y, larg, u, encastre: bool) -> str:
    """Le pied de la barre (y : le bas de la barre, en coordonnées du dessin) : articulation ou encastrement."""
    out = []
    if encastre:
        w = max(larg * 2.2, 14 * u)
        out.append(f'<rect x="{_f(x - w / 2)}" y="{_f(y)}" width="{_f(w)}" height="{_f(3.6 * u)}" fill="{SOL}" stroke="none"/>')
        out.append(_hachures_rect(x - w / 2, y, x + w / 2, y + 3.6 * u, 1.4 * u, 0.16 * u))
        out.append(_ligne(x - w / 2, y, x + w / 2, y, ENCRE, 0.5 * u))
    else:
        h = 5.0 * u
        r = 0.9 * u
        out.append(f'<path d="M{_f(x)},{_f(y + r)} L{_f(x - 3.2 * u)},{_f(y + h)} L{_f(x + 3.2 * u)},{_f(y + h)} Z" fill="#FFFFFF" '
                   f'stroke="{ENCRE}" stroke-width="{_f(0.36 * u)}" stroke-linejoin="round"/>')
        out.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="{_f(0.36 * u)}"/>')
        w = 10 * u
        out.append(f'<rect x="{_f(x - w / 2)}" y="{_f(y + h)}" width="{_f(w)}" height="{_f(2.4 * u)}" fill="{SOL}" stroke="none"/>')
        out.append(_hachures_rect(x - w / 2, y + h, x + w / 2, y + h + 2.4 * u, 1.3 * u, 0.15 * u))
        out.append(_ligne(x - w / 2, y + h, x + w / 2, y + h, ENCRE, 0.4 * u))
    return "".join(out)


def _guide(x, y, larg, u, haut: float, rotation_bloquee: bool) -> str:
    """Un maintien latéral de part et d'autre de la barre au niveau y (tête, ou maintien intermédiaire) : deux taquets
    hachurés ; plus hauts quand la rotation est bloquée (encastrement)."""
    out = []
    h = haut
    for s in (-1, 1):
        x0 = x + s * (larg / 2 + 0.5 * u)
        x1 = x0 + s * 3.2 * u
        xa, xb = min(x0, x1), max(x0, x1)
        out.append(f'<rect x="{_f(xa)}" y="{_f(y - h / 2)}" width="{_f(xb - xa)}" height="{_f(h)}" fill="{SOL}" stroke="none"/>')
        out.append(_hachures_rect(xa, y - h / 2, xb, y + h / 2, 1.2 * u, 0.14 * u))
        out.append(_ligne(x0, y - h / 2, x0, y + h / 2, ENCRE, 0.4 * u))
    if not rotation_bloquee:
        out.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(0.8 * u)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="{_f(0.3 * u)}"/>')
    return "".join(out)


def _tete(x, y, larg, u, appuis: str) -> str:
    """La tête de la barre selon les appuis : guidée articulée, encastrée, encastrée glissante, libre."""
    if appuis in ("AA", "EA", "k", "L"):
        return _guide(x, y + 2.0 * u, larg, u, 4.0 * u, False)
    if appuis == "EE":
        return _guide(x, y + 4.0 * u, larg, u, 8.0 * u, True)
    if appuis == "EG":
        # encastrement glissant : la rotation est bloquée, le déplacement latéral libre (le chariot se dessine décalé)
        w = max(larg * 2.4, 13 * u)
        out = (f'<rect x="{_f(x - w / 2)}" y="{_f(y - 1.6 * u)}" width="{_f(w)}" height="{_f(1.6 * u)}" fill="{ENCRE}"/>')
        for s in (-0.3, 0.3):
            out += f'<circle cx="{_f(x + s * w)}" cy="{_f(y - 2.5 * u)}" r="{_f(0.9 * u)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="{_f(0.3 * u)}"/>'
        out += f'<rect x="{_f(x - w / 2 - 1.5 * u)}" y="{_f(y - 6.3 * u)}" width="{_f(w + 3 * u)}" height="{_f(2.9 * u)}" fill="{SOL}"/>'
        out += _hachures_rect(x - w / 2 - 1.5 * u, y - 6.3 * u, x + w / 2 + 1.5 * u, y - 3.4 * u, 1.2 * u, 0.14 * u)
        out += _ligne(x - w / 2 - 1.5 * u, y - 3.4 * u, x + w / 2 + 1.5 * u, y - 3.4 * u, ENCRE, 0.36 * u)
        return out
    return ""                                                       # libre


def _panneau(cx: float, ax: Dict, L_mm: float, larg: float, u: float, N_txt: str, couleur: str, niv: str) -> Tuple[str, float]:
    """La barre dans un plan de flambement, centrée en x = cx, tête en y = 0 et pied en y = L ; rend (svg, demi-largeur
    occupée à droite pour les cotes)."""
    g = []
    ap, n = ax["appuis"], ax.get("maintiens", 0)
    # la barre (largeur réelle dans ce plan), l'axe non déformé
    g.append(f'<rect x="{_f(cx - larg / 2)}" y="0" width="{_f(larg)}" height="{_f(L_mm)}" fill="{ACIER}" stroke="{ENCRE}" '
             f'stroke-width="{_f(0.45 * u)}"/>')
    g.append(_ligne(cx, -3 * u, cx, L_mm + 3 * u, AXE, 0.16 * u, f' stroke-dasharray="{_f(4 * u)} {_f(1.2 * u)} {_f(0.6 * u)} {_f(1.2 * u)}"'))
    # les appuis
    g.append(_appui_pied(cx, L_mm, larg, u, encastre=ap in ("EA", "EE", "EG", "EL")))
    g.append(_tete(cx, 0.0, larg, u, ap))
    if ap == "AA":
        for i in range(1, n + 1):
            g.append(_guide(cx, L_mm * (1 - i / (n + 1)), larg, u, 3.0 * u, False))
    # la déformée (amplitude amplifiée)
    amp = 0.085 * L_mm
    pts = mode_norme(ap, n)
    d = " ".join(f"{'M' if i == 0 else 'L'}{_f(cx + amp * w)},{_f(L_mm * (1 - xi))}" for i, (xi, w) in enumerate(pts))
    g.append(f'<path d="{d}" fill="none" stroke="{couleur}" stroke-width="{_f(0.55 * u)}" stroke-dasharray="{_f(2.2 * u)} {_f(1.0 * u)}" '
             f'stroke-linecap="round"/>')
    # l'effort en tête (au-dessus des appuis)
    y_tete = -7.0 * u if ap == "EG" else -1.0 * u
    g.append(_fleche(cx, y_tete - 13 * u, cx, y_tete, EFFORT, 0.6 * u, 2.6 * u))
    g.append(_txt(cx + 1.6 * u, y_tete - 9.5 * u, N_txt, 3.5 * u, ancre="start", couleur=EFFORT, gras=True))
    # la cote de L_cr, à droite, sur la part de la barre entre deux points d'inflexion
    dr = max(larg / 2, amp) + 7 * u
    tr = troncon_lcr(ap, n)
    if tr:
        y0, y1 = L_mm * (1 - tr[1]), L_mm * (1 - tr[0])
        xc = cx + dr
        g.append(_attache(cx + larg / 2 + 0.8 * u, y0, xc + 1.2 * u, y0, u) + _attache(cx + larg / 2 + 0.8 * u, y1, xc + 1.2 * u, y1, u))
        g.append(_cote_v(xc, y0, y1, f"L_cr = {_nb(ax['Lcr'] * 1000, 0)}", u, droite=True, couleur=couleur))
    return "".join(g), dr + 6 * u


def barre(ident: str, r: Dict, extents: Dict[str, float], nom: str) -> str:
    """Les deux plans de flambement côte à côte, à la même échelle : la barre (largeur réelle dans le plan), ses
    appuis, ses maintiens, N_Ed en tête, la déformée du mode (amplifiée) et la cote de L_cr ; sous chaque barre, le
    résultat de l'axe. ``extents`` : la largeur de la section dans chaque plan (mm)."""
    L_mm = r["L"] * 1000.0
    # l'unité des traits et des textes : le centième de la barre, ou de six largeurs de section pour une barre courte
    # et trapue (sans quoi une barre de 0,5 m en HE 300 porterait des textes illisibles à côté de sa section)
    u = max(L_mm, 6.0 * max(extents.values())) / 100.0
    N_txt = f"N_Ed = {_nb(r['N'], 0 if r['N'] >= 100 else 1)} kN"
    larg = {"y": max(extents["y"], 0.5 * u), "z": max(extents["z"], 0.5 * u)}
    pas = max(0.62 * L_mm, max(larg.values()) * 2.6 + 34 * u)
    parts = []
    xs = {"y": 0.0, "z": pas}
    for cle in ("y", "z"):
        ax = r["axes"][cle]
        niv = C.niveau(ax["taux"])                # le code couleur du bureau (seuils du moteur) : une seule source
        g, _ = _panneau(xs[cle], ax, L_mm, larg[cle], u, N_txt, COUL_AXE[cle], niv)
        parts.append(g)
        cx = xs[cle]
        parts.append(_txt(cx, -31 * u, f"Flambement autour de {ax['nom']}", 4.2 * u, gras=True))
        lib = EN.LIB_APPUIS[ax["appuis"]]
        if ax["appuis"] == "AA" and ax.get("maintiens"):
            lib += f" · {ax['maintiens']} maintien{'s' if ax['maintiens'] > 1 else ''}"
        parts.append(_txt(cx, -26.2 * u, lib, 3.3 * u, couleur=MUET))
        yb = L_mm + 14 * u
        k = ax["k"]
        parts.append(_txt(cx, yb, f"L_cr = {_k(k)} × {_nb(r['L'], 2)} = {_nb(ax['Lcr'], 2)} m", 3.5 * u, gras=True, couleur=COUL_AXE[cle]))
        parts.append(_txt(cx, yb + 5.2 * u, f"courbe {ax['courbe']} · λ̄ = {_nb(ax['lambdabar'], 3)} · χ = {_nb(ax['chi'], 3)}", 3.3 * u))
        t = ax["taux"]
        parts.append(_txt(cx, yb + 10.6 * u, f"N_b,Rd = {_nb(ax['NbRd'], 1)} kN · η = {C.pc(C.pct(t))} %", 3.4 * u, gras=True,
                          couleur=NIVEAU.get(niv, ENCRE)))
    # la longueur de la barre, cotée à gauche du premier plan
    xg = -max(larg["y"] / 2, 0.085 * L_mm) - 12 * u
    parts.append(_attache(-larg["y"] / 2 - 0.8 * u, 0, xg - 1.2 * u, 0, u) + _attache(-larg["y"] / 2 - 0.8 * u, L_mm, xg - 1.2 * u, L_mm, u))
    parts.append(_cote_v(xg, 0, L_mm, f"L = {_nb(L_mm, 0)}", u))
    parts.append(_txt(pas / 2, L_mm + 33 * u, "déformées du mode de flambement amplifiées · barre à l'échelle (mm)", 2.9 * u, couleur=MUET))
    x0 = xg - 8 * u
    x1 = pas + max(larg["z"] / 2, 0.085 * L_mm) + 26 * u
    y0, y1 = -36 * u, L_mm + 36 * u
    vb = f"{_f(x0)} {_f(y0)} {_f(x1 - x0)} {_f(y1 - y0)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Barre {escape(nom)} : appuis et déformée de flambement par axe">{"".join(parts)}</svg>')


# ------------------------------------------------------------------ la section
def _hachures(anneaux: Sequence[Sequence[Tuple[float, float]]], pas: float) -> List[Tuple[float, float, float, float]]:
    """Hachures à 45° de la matière (pair-impair sur TOUS les contours : un vide reste blanc)."""
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


def _brut(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def section(ident: str, p: Dict, r: Dict, fiche: List[Tuple[str, str, str]]) -> str:
    """La section à l'échelle (mm), centrée sur son centre de gravité : la matière hachurée, les deux axes de
    flambement avec leur courbe (u-u et v-v pour une cornière, y-y et z-z en léger), les cotes, les épaisseurs, la
    fiche des grandeurs du calcul (``fiche`` : [(symbole, valeur, unité)])."""
    dims = p["dims"]
    yc, zc = float(p.get("yc") or 0.0), float(p.get("zc") or 0.0)
    anneaux = [[(float(q[0]) - yc, -(float(q[1]) - zc)) for q in ring] for ring in FA.contours(p["famille"], dims)]
    xs = [x for a in anneaux for x, _ in a]
    ys = [y for a in anneaux for _, y in a]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    larg, haut = x1 - x0, y1 - y0
    u = max(larg, haut) / 100.0
    g = []
    dpath = " ".join("M" + " L".join(f"{_f(x)},{_f(y)}" for x, y in a) + " Z" for a in anneaux)
    g.append(f'<path d="{dpath}" fill="{ACIER}" fill-rule="evenodd" stroke="none"/>')
    g.append("".join(_ligne(a, b_, c, d_, HACH, 0.17 * u) for a, b_, c, d_ in _hachures(anneaux, 2.4 * u)))
    g.append(f'<path d="{dpath}" fill="none" stroke="{ENCRE}" stroke-width="{_f(0.55 * u)}" stroke-linejoin="round"/>')
    # les axes : ceux du flambement en couleur (le trait mixte des axes), avec leur courbe au bout
    fs = 4.6 * u
    tiret = f' stroke-dasharray="{_f(5 * u)} {_f(1.4 * u)} {_f(0.7 * u)} {_f(1.4 * u)}"'
    axes = r.get("axes") or {}
    courbes = r.get("courbes") or {}
    angle = math.radians(float(p.get("alpha_deg") or 0.0)) if p["famille"] in ("L", "LI") else 0.0
    larg_lab = 0.56 * fs * 13                                   # « v · courbe a0 » : la largeur d'une étiquette d'axe
    bornes = [x0, x1, y0, y1]                                   # ce que le dessin occupe (gauche, droite, haut, bas)

    def occupe(x, y, ancre="middle", w=0.0):
        xa = x - (w if ancre == "end" else (w / 2 if ancre == "middle" else 0.0))
        bornes[0], bornes[1] = min(bornes[0], xa), max(bornes[1], xa + w)
        bornes[2], bornes[3] = min(bornes[2], y - fs), max(bornes[3], y + 0.3 * fs)

    def etiquette(cle):
        nom = (axes.get(cle) or {}).get("nom") or ("y-y" if cle == "y" else "z-z")
        cb = courbes.get(cle)
        return f"{nom[0]} · courbe {cb}" if cb else nom[0]
    if not angle:
        ey, ez = 7 * u, 7 * u
        g.append(_ligne(x0 - ey, 0, x1 + ey, 0, COUL_AXE["y"], 0.3 * u, tiret))
        g.append(_ligne(0, y0 - ez, 0, y1 + ez, COUL_AXE["z"], 0.3 * u, tiret))
        g.append(_txt(x1 + ey + 1.5 * u, 0.36 * fs, etiquette("y"), fs, ancre="start", couleur=COUL_AXE["y"], gras=True))
        occupe(x1 + ey + 1.5 * u, 0.36 * fs, "start", larg_lab)
        g.append(_txt(1.5 * u, y0 - ez - 1.2 * u, etiquette("z"), fs, ancre="start", couleur=COUL_AXE["z"], gras=True))
        occupe(1.5 * u, y0 - ez - 1.2 * u, "start", larg_lab)
    else:
        # cornière : y-y et z-z en léger, u-u et v-v (axes principaux) en couleur
        g.append(_ligne(x0 - 5 * u, 0, x1 + 5 * u, 0, GRIS, 0.16 * u, tiret))
        g.append(_ligne(0, y0 - 5 * u, 0, y1 + 5 * u, GRIS, 0.16 * u, tiret))
        g.append(_txt(x1 + 6 * u, 0.36 * fs, "y", 0.9 * fs, ancre="start", couleur=GRIS, extra=' font-style="italic"'))
        g.append(_txt(1.4 * u, y0 - 6 * u, "z", 0.9 * fs, ancre="start", couleur=GRIS, extra=' font-style="italic"'))
        R = max(math.hypot(x, y) for a in anneaux for x, y in a) + 6 * u
        for cle, phi in (("y", angle), ("z", angle + math.pi / 2)):
            dx, dy = math.cos(phi), -math.sin(phi)
            c = COUL_AXE[cle]
            g.append(_ligne(-R * dx, -R * dy, R * dx, R * dy, c, 0.3 * u, tiret))
            xt, yt = (R + 1.5 * u) * dx, (R + 1.5 * u) * dy + 0.36 * fs
            anc = "start" if dx > 0.3 else ("end" if dx < -0.3 else "middle")
            g.append(_txt(xt, yt, etiquette(cle), fs, ancre=anc, couleur=c, gras=True))
            occupe(xt, yt, anc, larg_lab)
            occupe(-R * dx, -R * dy)
    # le centre de gravité
    g.append(f'<circle cx="0" cy="0" r="{_f(0.9 * u)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="{_f(0.3 * u)}"/>')
    # les cotes : hauteur à gauche, largeur dessous
    xh = x0 - 12 * u
    g.append(_attache(x0 - 1.2 * u, y0, xh - 1.5 * u, y0, u) + _attache(x0 - 1.2 * u, y1, xh - 1.5 * u, y1, u))
    hv = dims.get("h", dims.get("d"))
    g.append(_cote_v(xh, y0, y1, f"{'h' if 'h' in dims else 'd'} = {_brut(hv)}", u, fs=fs))
    occupe(xh - 1.4 * fs, y1)
    yb = max(y1 + 12 * u, bornes[3] + 3 * u) if not angle else y1 + 12 * u
    g.append(_attache(x0, y1 + 1.2 * u, x0, yb + 1.5 * u, u) + _attache(x1, y1 + 1.2 * u, x1, yb + 1.5 * u, u))
    bv = dims.get("b", dims.get("d"))
    g.append(_cote_h(x0, x1, yb, f"{'b' if 'b' in dims else 'd'} = {_brut(bv)}", u, dessous=True, fs=fs))
    # les épaisseurs, en légende sous la cote
    eps = " · ".join(f"{k} = {_brut(dims[k])}" for k in ("tw", "tf", "t", "r", "r1") if dims.get(k) is not None)
    eps = eps.replace("tw", "t_w").replace("tf", "t_f").replace("r1", "r_1")
    y_eps = yb + 1.15 * fs + 1.5 * fs
    g.append(_txt((x0 + x1) / 2, y_eps, eps, 0.92 * fs, couleur=MUET))
    occupe((x0 + x1) / 2, y_eps)
    # le titre, au-dessus de tout
    y_titre = bornes[2] - 4 * u
    g.append(_txt(0, y_titre, p["nom"].replace("x", "×"), 1.35 * fs, gras=True))
    # la fiche, à droite
    xf = bornes[1] + 12 * u
    yf = y_titre - 0.4 * fs
    pas = 1.6 * fs
    tab = [_txt(xf, yf, "LE CALCUL", 0.72 * fs, ancre="start", couleur=MUET, gras=True, extra=' letter-spacing="0.6"'),
           _ligne(xf, yf + 0.4 * fs, xf + 13.5 * fs, yf + 0.4 * fs, GRILLE, 0.25 * u)]
    y = yf + 0.4 * fs + pas
    for sym, val, unite in fiche:
        tab.append(_txt(xf, y - 0.36 * fs, sym, fs, ancre="start", couleur=MUET))
        tab.append(_txt(xf + 9.4 * fs, y - 0.36 * fs, escape(val), fs, ancre="end", gras=True, brut=True))
        tab.append(_txt(xf + 9.8 * fs, y - 0.36 * fs, unite, 0.82 * fs, ancre="start", couleur=MUET))
        tab.append(_ligne(xf, y + 0.1 * fs, xf + 13.5 * fs, y + 0.1 * fs, GRILLE, 0.16 * u))
        y += pas
    g += tab
    gx0 = bornes[0] - 6 * u
    gx1 = xf + 13.5 * fs + 4 * u
    gy0 = y_titre - 1.4 * fs - 4 * u
    gy1 = max(y_eps + 6 * u, y + 2 * u, bornes[3] + 4 * u)
    vb = f"{_f(gx0)} {_f(gy0)} {_f(gx1 - gx0)} {_f(gy1 - gy0)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Section {escape(p["nom"])}">{"".join(g)}</svg>')


# ------------------------------------------------------------------ les courbes χ(λ̄)
def courbes(ident: str, r: Dict) -> str:
    """Les cinq courbes de flambement (Tableau 6.1), celles des deux axes en couleur, l'hyperbole d'Euler 1/λ̄², et le
    point (λ̄, χ) de chaque axe. Un graphique, pas un dessin à l'échelle."""
    axes = r["axes"]
    lmax = max(3.0, math.ceil((max(a["lambdabar"] for a in axes.values()) + 0.25) * 2) / 2)
    W, H = 1000.0, 560.0
    mg, md, mh, mb = 92.0, 250.0, 34.0, 70.0
    pw, ph = W - mg - md, H - mh - mb

    def X(lb):
        return mg + pw * lb / lmax

    def Y(chi):
        return mh + ph * (1 - chi / 1.1)

    g = [f'<rect x="{_f(mg)}" y="{_f(mh)}" width="{_f(pw)}" height="{_f(ph)}" fill="#FFFFFF" stroke="{GRILLE}"/>']
    pas_l = 0.5 if lmax <= 4 else 1.0
    k = 0.0
    while k <= lmax + 1e-9:
        g.append(_ligne(X(k), mh, X(k), mh + ph, GRILLE, 1))
        g.append(_txt(X(k), mh + ph + 24, _nb(k, 1), 17, couleur=MUET))
        k += pas_l
    for c in range(0, 12, 2):
        v = c / 10
        g.append(_ligne(mg, Y(v), mg + pw, Y(v), GRILLE, 1))
        g.append(_txt(mg - 10, Y(v) + 6, _nb(v, 1), 17, ancre="end", couleur=MUET))
    g.append(_txt(mg + pw / 2, H - 12, "élancement réduit λ̄", 19, couleur=ENCRE))
    g.append(_txt(26, mh + ph / 2, "coefficient de réduction χ", 19, extra=f' transform="rotate(-90 26 {_f(mh + ph / 2)})"'))
    # Euler
    pts = [(lb, min(1.0 / lb ** 2, 1.1)) for lb in [0.05 * i for i in range(19, int(lmax / 0.05) + 1)]]
    g.append(f'<path d="{" ".join(("M" if i == 0 else "L") + f"{_f(X(a))},{_f(Y(b))}" for i, (a, b) in enumerate(pts))}" fill="none" '
             f'stroke="{GRIS}" stroke-width="1.4" stroke-dasharray="6 5"/>')
    g.append(_txt(X(min(1.25, lmax)) + 8, Y(1 / 1.25 ** 2) - 8, "Euler 1/λ̄²", 16, ancre="start", couleur=MUET))
    utilisees = {a["courbe"]: cle for cle, a in axes.items()}
    lab = []
    for nom, alpha in NO.ALPHA.items():
        pts = []
        for i in range(0, int(lmax / 0.02) + 1):
            lb = 0.02 * i
            pts.append((lb, NO.phi_chi(lb, alpha)[1]))
        cle = utilisees.get(nom)
        coul = COUL_AXE[cle] if cle else GRIS
        lw = 3.0 if cle else 1.3
        g.append(f'<path d="{" ".join(("M" if i == 0 else "L") + f"{_f(X(a))},{_f(Y(b))}" for i, (a, b) in enumerate(pts))}" fill="none" '
                 f'stroke="{coul}" stroke-width="{lw}" stroke-linejoin="round"/>')
        lab.append((nom, alpha, coul, cle))
    # la légende des courbes, à droite
    xl = mg + pw + 26
    g.append(_txt(xl, mh + 12, "COURBES (TABLEAU 6.1)", 14, ancre="start", couleur=MUET, gras=True, extra=' letter-spacing="0.6"'))
    for i, (nom, alpha, coul, cle) in enumerate(lab):
        y = mh + 44 + 30 * i
        g.append(_ligne(xl, y - 5, xl + 30, y - 5, coul, 3.0 if cle else 1.5))
        g.append(_txt(xl + 40, y, f"{nom} · α = {_nb(alpha, 2)}", 17, ancre="start", couleur=ENCRE if cle else MUET, gras=bool(cle)))
    # les points des deux axes
    yl = mh + 44 + 30 * len(lab) + 26
    g.append(_txt(xl, yl, "LA BARRE", 14, ancre="start", couleur=MUET, gras=True, extra=' letter-spacing="0.6"'))
    deja = []
    for j, (cle, a) in enumerate(axes.items()):
        x, y = X(a["lambdabar"]), Y(a["chi"])
        coul = COUL_AXE[cle]
        g.append(_ligne(x, y, x, mh + ph, coul, 1.2, ' stroke-dasharray="3 4"'))
        g.append(_ligne(mg, y, x, y, coul, 1.2, ' stroke-dasharray="3 4"'))
        g.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="7" fill="{coul}" stroke="#FFFFFF" stroke-width="2.4"/>')
        dy = -14 if not any(abs(y - q) < 26 and abs(x - p) < 160 for p, q in deja) else 26
        g.append(_txt(x + 10, y + dy, a["nom"], 17, ancre="start", couleur=coul, gras=True))
        deja.append((x, y))
        yy = yl + 28 + 46 * j
        g.append(f'<circle cx="{_f(xl + 7)}" cy="{_f(yy - 5)}" r="6" fill="{coul}"/>')
        g.append(_txt(xl + 20, yy, f"{a['nom']} · courbe {a['courbe']}", 17, ancre="start", couleur=coul, gras=True))
        g.append(_txt(xl + 20, yy + 21, f"λ̄ = {_nb(a['lambdabar'], 3)} · χ = {_nb(a['chi'], 3)}", 16, ancre="start", couleur=ENCRE))
    return (f'<svg id="{escape(ident)}" class="fl-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(W)} {_f(H)}" width="100%" '
            f'role="img" aria-label="Courbes de flambement χ en fonction de λ̄">{"".join(g)}</svg>')

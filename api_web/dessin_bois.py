# -*- coding: utf-8 -*-
"""Les dessins de la poutre bois, en SVG, À L'ÉCHELLE : le schéma statique (charges G et Q, appuis, portée) et, dessous,
les diagrammes d'effort tranchant et de moment ; la déformée et la section (cernes du bois, contraintes σ et τ).

Tout vient du calcul (``bois.poutre.moteur.verifier``) et des courbes qui portent ses maxima
(``bois.poutre.diagrammes``) : le dessin ne calcule aucune valeur qu'il affiche. Les vues en élévation ont la portée
à l'échelle (la hauteur de la poutre et la largeur d'appui aussi) ; les ordonnées des diagrammes (V, M, w, σ, τ) ont
leur propre échelle, écrite sur la vue. La section est à l'échelle dans ses deux directions.

Le style suit les dessins du garde-corps et de la poutre béton armé : traits noirs, cotes fines grises avec traits
obliques, charges en rouge (exploitation) et en ardoise (permanentes), bois en teinte claire veinée.
"""
from typing import Dict, List, Sequence, Tuple

from bois.poutre import diagrammes as DG

from . import commun as C

ENCRE, COTE, ARDOISE, ROUGE = "#15181F", "#7F8C9A", "#33415C", "#9C3341"
BOIS, VEINE, BOIS_CLAIR = "#EBDDBF", "#B08D57", "#F3E9D3"
TEINTE_G, TEINTE_Q, TEINTE_V, TEINTE_M = "#E7EBF1", "#F6E2E5", "#E2EAF5", "#F6E2E5"
POLICE = "system-ui,Arial,sans-serif"
X0, LARGEUR = 104.0, 520.0            # la portée occupe 520 unités à partir de x = 104, quelle que soit L
LARGEUR_SVG = 760.0                  # la feuille : à peu près la proportion de la toile, texte lisible


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


HALO = ' stroke="#FFFFFF" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
HB_MAX = 72.0          # la hauteur dessinée de la poutre au plus (une poutre haute sur une portée courte écraserait la feuille)
APPUI_MAX = 0.18       # la largeur d'appui dessinée au plus, en part de la portée


def _txt(x, y, t, taille=15.0, ancre="middle", couleur=ENCRE, gras=False, extra="", halo=False) -> str:
    """Un texte ; ``halo`` : détouré de blanc, lisible quand il croise un trait (axe, limite, diagramme)."""
    poids = ' font-weight="600"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{HALO if halo else ""}{extra}>{t}</text>')


def _sub(base: str, indice: str, reste: str = "") -> str:
    """« M<sub>Ed</sub> = … » en SVG : l'indice plus petit et plus bas, puis la ligne reprend sa hauteur."""
    return f"{base}<tspan font-size='72%' dy='4'>{indice}</tspan><tspan dy='-4'>{reste}</tspan>"


def _ligne(x1, y1, x2, y2, couleur=ENCRE, ep=1.0, tirets="") -> str:
    d = f' stroke-dasharray="{tirets}"' if tirets else ""
    return f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"{d}/>'


def _poly(points: Sequence[Tuple[float, float]], fond="none", trait=ENCRE, ep=1.4, ferme=True, opacite=None) -> str:
    pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
    balise = "polygon" if ferme else "polyline"
    op = f' fill-opacity="{opacite}"' if opacite is not None else ""
    return f'<{balise} points="{pts}" fill="{fond}"{op} stroke="{trait}" stroke-width="{_f(ep)}" stroke-linejoin="round"/>'


def _fleche(x, y_haut, y_bas, couleur, ep=1.3, tete=7.0) -> str:
    """Une flèche verticale vers le bas, pointe en (x, y_bas)."""
    return (_ligne(x, y_haut, x, y_bas - tete + 1, couleur, ep)
            + _poly([(x, y_bas), (x - tete * 0.45, y_bas - tete), (x + tete * 0.45, y_bas - tete)], couleur, couleur, 0.6))


def _cote_h(x1, x2, y, texte, taille=15.0) -> str:
    """Une cote horizontale : ligne, deux traits obliques, le texte au-dessus."""
    t = 5.0
    return (_ligne(x1, y, x2, y, COTE, 1.0) + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, 1.2)
            + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, 1.2) + _txt((x1 + x2) / 2, y - 7, texte, taille))


def _cote_v(x, y1, y2, texte, taille=15.0) -> str:
    """Une cote verticale : le texte tourné le long de la ligne, à sa gauche."""
    t = 5.0
    xt = x - 8
    return (_ligne(x, y1, x, y2, COTE, 1.0) + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, 1.2)
            + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, 1.2)
            + _txt(xt, (y1 + y2) / 2, texte, taille, extra=f' transform="rotate(-90 {_f(xt)} {_f((y1 + y2) / 2)})"'))


def _sol(x, y, largeur=44.0) -> str:
    """La ligne de sol d'un appui et ses hachures."""
    out = _ligne(x - largeur / 2, y, x + largeur / 2, y, ENCRE, 1.6)
    k = x - largeur / 2 + 4
    while k <= x + largeur / 2:
        out += _ligne(k, y, k - 7, y + 8, ENCRE, 0.9)
        k += 7
    return out


def _appui(x, y, mobile: bool, taille=1.0) -> str:
    """Un appui sous le point (x, y) : articulation fixe (triangle) ou appui à rouleaux (triangle, deux rouleaux)."""
    hh, dl = 22.0 * taille, 15.0 * taille
    out = _poly([(x, y), (x - dl, y + hh), (x + dl, y + hh)], "#FFFFFF", ENCRE, 1.5)
    if mobile:
        r = 3.6 * taille
        out += (f'<circle cx="{_f(x - 7 * taille)}" cy="{_f(y + hh + r)}" r="{_f(r)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="1.2"/>'
                f'<circle cx="{_f(x + 7 * taille)}" cy="{_f(y + hh + r)}" r="{_f(r)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="1.2"/>')
        return out + _sol(x, y + hh + 2 * r, 44 * taille)
    return out + _sol(x, y + hh, 44 * taille)


def _veines(x1, x2, y1, y2) -> str:
    """Les veines du bois le long de la poutre : deux ou trois traits légèrement ondulés."""
    h = y2 - y1
    if h < 9:
        return ""
    out = ""
    n = 3 if h > 30 else 2
    for i in range(n):
        y = y1 + h * (i + 1) / (n + 1)
        amp = min(2.5, h * 0.06)
        d = f"M{_f(x1 + 6)} {_f(y)}"
        pas = (x2 - x1 - 12) / 8
        for j in range(8):
            xa = x1 + 6 + pas * (j + 0.5)
            xb = x1 + 6 + pas * (j + 1)
            d += f" Q{_f(xa)} {_f(y + (amp if (i + j) % 2 else -amp))} {_f(xb)} {_f(y)}"
        out += f'<path d="{d}" fill="none" stroke="{VEINE}" stroke-width="0.8" opacity=".85"/>'
    return out


def _svg(ident: str, largeur: float, hauteur: float, corps: str, etiquette: str) -> str:
    return (f'<svg id="{C.esc(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(largeur)} {_f(hauteur)}" width="100%" '
            f'role="img" aria-label="{C.esc(etiquette)}">{corps}</svg>')


def _v(x: float, nd: int = 2) -> str:
    """Une valeur du calcul écrite sur le dessin : ``nd`` décimales, davantage pour un petit nombre non nul (deux
    chiffres significatifs, quatre décimales au plus) — un effort de 0,0024 kN·m ne s'écrit pas « 0,00 »."""
    while nd and nd < 4 and x != 0 and abs(x) < 10 ** (1 - nd):
        nd += 1
    return C.fr(x, nd)


def _signe(x: float, signe: str) -> str:
    """« +14,40 » / « −14,40 » ; une valeur nulle s'écrit « 0 », jamais « −0,00 »."""
    return f"{signe}{_v(x)}" if x else "0"


def _echelle(r: Dict) -> float:
    """Unités de dessin par millimètre : la portée occupe toujours LARGEUR unités."""
    return LARGEUR / r["geometrie"]["L_mm"]


def _bandes_charges(ch: Dict, y_haut: float, x1: float, x2: float) -> Tuple[str, float]:
    """Les charges réparties : la bande Q au-dessus de la bande G, hauteurs proportionnelles à leur valeur (au moins
    lisibles) ; rend (svg, y du bas des bandes)."""
    G, Q = ch["G"], ch["Q"]
    total = G + Q
    if total <= 0:
        return _txt((x1 + x2) / 2, y_haut + 30, "aucune charge (G = Q = 0)", 15, couleur=COTE), y_haut + 44
    HT = 76.0
    hq = max(14.0, HT * Q / total) if Q > 0 else 0.0
    hg = max(14.0, HT * G / total) if G > 0 else 0.0
    out, y = "", y_haut
    n = 17
    for valeur, hb, teinte, couleur, nom in ((Q, hq, TEINTE_Q, ROUGE, "Q"), (G, hg, TEINTE_G, ARDOISE, "G")):
        if hb <= 0:
            continue
        out += f'<rect x="{_f(x1)}" y="{_f(y)}" width="{_f(x2 - x1)}" height="{_f(hb)}" fill="{teinte}"/>'
        out += _ligne(x1, y, x2, y, couleur, 1.6)
        for i in range(n):
            xa = x1 + (x2 - x1) * i / (n - 1)
            out += _fleche(xa, y, y + hb, couleur, 1.1, min(7.0, hb * 0.55))
        out += _txt(x2 + 16, y + hb / 2 + 5, f"{nom} = {C.nb(valeur)} kN/m", 15, "start", couleur, True)
        y += hb
    return out, y


# ------------------------------------------------------------------ 1. la poutre : schéma statique et efforts
def _hachures(xs: List[float], y0: float, ys: List[float], couleur: str) -> str:
    return "".join(_ligne(x, y0, x, y, couleur, 0.6) for x, y in zip(xs, ys) if abs(y - y0) > 1.5)


def _section_txt(geo: Dict) -> str:
    return f"{C.nb(geo['b'])} × {C.nb(geo['h'])}"


def poutre(ident: str, r: Dict) -> str:
    """La feuille de la poutre : en haut le schéma statique à l'échelle (charges G et Q réparties, poutre de hauteur
    h, appuis — articulation en A, rouleaux en B —, largeur d'appui quand la compression à l'appui est vérifiée,
    réactions) ; dessous, alignés sur la même portée, les diagrammes de l'effort tranchant V(x) (positif au-dessus de
    l'axe) et du moment M(x) (positif sous l'axe, côté des fibres tendues) à l'ELU ; la portée cotée."""
    k = _echelle(r)
    geo, ch, g, p = r["geometrie"], r["charges"], r["grandeurs"], r["poutre"]
    x1, x2 = X0, X0 + LARGEUR
    xm = (x1 + x2) / 2
    # les charges, la poutre (hauteur à l'échelle, débord d'une demi-largeur d'appui)
    # à l'échelle de la portée, sauf une poutre très haute ou un appui très large sur une portée courte : réduits
    # (au plus HB_MAX unités, APPUI_MAX de la portée) pour que la feuille reste lisible — et c'est écrit dessous
    a_vrai, hb_vrai = p["a"] * k, geo["h"] * k
    a = min(a_vrai, APPUI_MAX * LARGEUR)
    e = max(a / 2, 4.0)
    hb = min(max(hb_vrai, 3.0), HB_MAX)
    reduits = [t for t, oui in (("hauteur de la poutre", hb_vrai > HB_MAX), ("largeur d'appui", p["appui"] and a_vrai > a)) if oui]
    out, y_b = _bandes_charges(ch, 26.0, x1, x2)
    y_p = y_b + 2
    out += (f'<rect x="{_f(x1 - e)}" y="{_f(y_p)}" width="{_f(x2 - x1 + 2 * e)}" height="{_f(hb)}" fill="{BOIS}" stroke="{ENCRE}" stroke-width="2"/>'
            + _veines(x1 - e, x2 + e, y_p, y_p + hb))
    y_a = y_p + hb
    if p["appui"]:
        for x in (x1, x2):
            out += f'<rect x="{_f(x - a / 2)}" y="{_f(y_a)}" width="{_f(a)}" height="6" fill="{ARDOISE}"/>'
        y_a += 6
        out += _txt(x1 + max(a / 2, 15) + 10, y_a + 14, f"a = {C.nb(p['a'])} mm", 13.5, "start", ARDOISE, True, halo=True)
    out += _appui(x1, y_a, False) + _appui(x2, y_a, True)
    out += _txt(x1 - 24, y_a + 20, "A", 18, "end", gras=True) + _txt(x2 + 24, y_a + 20, "B", 18, "start", gras=True)
    y_r = y_a + 56
    reac = _sub("R", "Ed", f" = {_v(g['V_Ed'])} kN")
    out += _txt(x1, y_r, reac, 14.5, couleur=ROUGE, gras=True) + _txt(x2, y_r, reac, 14.5, couleur=ROUGE, gras=True)
    if p["appui"] and g["R_appui"] is not None:
        rs = _sub("R", "G+Q", f" = {_v(g['R_appui'])} kN")
        out += _txt(x1, y_r + 19, rs, 13.5, couleur=ARDOISE) + _txt(x2, y_r + 19, rs, 13.5, couleur=ARDOISE)
        y_r += 19
    # V(x)
    cb = DG.courbes(r, 75)
    xs = [x1 + LARGEUR * x / geo["L"] for x in cb["x"]]
    HV, HM = 64.0, 96.0
    yv = y_r + 34 + HV
    vmax = max(abs(g["V_Ed"]), 1e-12)
    yv_s = [yv - HV * v / vmax for v in cb["V"]]
    out += _hachures(xs, yv, yv_s, ARDOISE)
    out += _poly([(x1, yv)] + list(zip(xs, yv_s)) + [(x2, yv)], TEINTE_V, ARDOISE, 1.6, opacite=".8")
    out += _ligne(x1 - 8, yv, x2 + 8, yv, ENCRE, 1.2)
    out += _txt(x1 - 20, yv - 4, "V", 18, "end", ARDOISE, True) + _txt(x1 - 20, yv + 14, "kN", 13, "end", COTE)
    out += _txt(x1 + 8, yv - HV - 8, _signe(g["V_Ed"], "+"), 15, "start", ARDOISE, True)
    out += _txt(x2 - 8, yv + HV + 20, _signe(g["V_Ed"], "−"), 15, "end", ARDOISE, True)
    # M(x)
    ym = yv + HV + 46
    mmax = max(abs(g["M_Ed"]), 1e-12)
    ym_s = [ym + HM * m / mmax for m in cb["M"]]
    out += _hachures(xs, ym, ym_s, ROUGE)
    out += _poly([(x1, ym)] + list(zip(xs, ym_s)) + [(x2, ym)], TEINTE_M, ROUGE, 1.6, opacite=".8")
    out += _ligne(x1 - 8, ym, x2 + 8, ym, ENCRE, 1.2)
    out += _txt(x1 - 20, ym + 12, "M", 18, "end", ROUGE, True) + _txt(x1 - 20, ym + 30, "kN·m", 13, "end", COTE)
    out += _txt(xm, ym + HM + 24, _sub("M", "Ed", f" = {_v(g['M_Ed'])} kN·m"), 15.5, couleur=ROUGE, gras=True)
    # la mi-portée, la portée
    out += _ligne(xm, yv - HV - 14, xm, ym + HM + 6, COTE, 0.8, "6 4") + _txt(xm + 6, yv - HV - 2, "L/2", 13, "start", COTE, halo=True)
    yc = ym + HM + 66
    out += (_ligne(x1, y_a + 34, x1, yc + 6, COTE, 0.6, "4 3") + _ligne(x2, y_a + 34, x2, yc + 6, COTE, 0.6, "4 3")
            + _cote_h(x1, x2, yc, f"L = {C.fr(geo['L_mm'], 0)}", 16))
    q = _sub("q", "ELU", f" = {C.nb(ch['gamma_G'])} × {C.nb(ch['G'])} + {C.nb(ch['gamma_Q'])} × {C.nb(ch['Q'])} = {_v(ch['q_ELU'])} kN/m")
    out += _txt(xm, yc + 38, q, 15.5, couleur=ROUGE, gras=True)
    out += _txt(xm, yc + 60, "deux appuis, charge uniforme · V et M à l'ELU, ordonnées à leur propre échelle", 13, couleur=COTE)
    if reduits:
        dims = f"h = {C.nb(geo['h'])} mm" + (f", a = {C.nb(p['a'])} mm" if "largeur d'appui" in reduits else "")
        out += _txt(xm, yc + 80, f"dessin réduit pour la lisibilité : {dims} (seule la portée est à l'échelle)", 13, couleur=ROUGE)
        yc += 20
    return _svg(ident, LARGEUR_SVG, yc + 74, out, "Schéma statique et diagrammes des efforts de la poutre bois")


# ------------------------------------------------------------------ 2. la déformée
def deformee(ident: str, r: Dict) -> str:
    """Les déformées instantanée et finale le long de la portée, les limites L/x reportées sur le tiers central ;
    ordonnées amplifiées (ou réduites), le rapport écrit sur la vue. Les flèches sont celles de l'écran Streamlit."""
    g, geo, p = r["grandeurs"], r["geometrie"], r["poutre"]
    cb = DG.courbes(r, 49)
    k = _echelle(r)
    x1, x2 = X0, X0 + LARGEUR
    xs = [x1 + LARGEUR * x / geo["L"] for x in cb["x"]]
    y0 = 50.0
    HW = 230.0
    ref = max(g["w_inst"], g["w_fin"], g["lim_inst_mm"], g["lim_fin_mm"])
    f = HW / ref if ref > 0 else 0.0
    out = f'<rect x="{_f(x1 - 4)}" y="{_f(y0 - 5)}" width="{_f(LARGEUR + 8)}" height="10" fill="{BOIS}" opacity=".45"/>'
    out += _ligne(x1 - 16, y0, x2 + 16, y0, COTE, 1.0, "10 4 2 4")
    out += _appui(x1, y0 + 5, False, 0.8) + _appui(x2, y0 + 5, True, 0.8)
    out += _txt(x1 - 20, y0 + 22, "A", 16, "end", gras=True) + _txt(x2 + 20, y0 + 22, "B", 16, "start", gras=True)
    # les limites, sur le tiers central ; leurs textes ne se chevauchent pas
    xa, xb = x1 + LARGEUR / 3, x1 + 2 * LARGEUR / 3
    yl = sorted([(y0 + g["lim_inst_mm"] * f, p["lim_inst"], ARDOISE), (y0 + g["lim_fin_mm"] * f, p["lim_fin"], ROUGE)], key=lambda t: t[0])
    y_txt = -1e9
    for y, t, couleur in yl:
        out += _ligne(xa, y, xb, y, couleur, 1.2, "6 4")
        y_txt = max(y + 5, y_txt + 16)
        out += _txt(xa - 8, y_txt, t, 13, "end", couleur, True, halo=True)
    # les déformées
    for cle, couleur in (("w_inst", ARDOISE), ("w_fin", ROUGE)):
        out += _poly(list(zip(xs, [y0 + w * f for w in cb[cle]])), "none", couleur, 2.4, ferme=False)
    xm = (x1 + x2) / 2
    out += _ligne(xm, y0 + 8, xm, y0 + HW + 12, COTE, 0.8, "6 4")
    # les valeurs, en légende
    yg = y0 + HW + 58
    for i, (cle, lim, u, couleur, nom) in enumerate((("w_inst", "lim_inst_mm", "u_wi", ARDOISE, "inst"), ("w_fin", "lim_fin_mm", "u_wf", ROUGE, "fin"))):
        yy = yg + 28 * i
        out += _ligne(x1 - 30, yy - 5, x1, yy - 5, couleur, 2.6)
        val = _sub("w", nom, f" = {_v(g[cle], 1)} mm  ·  limite {p['lim_' + nom]} = {_v(g[lim], 1)} mm  ·  taux {C.fr(g[u] * 100, 0)} %")
        out += _txt(x1 + 10, yy, val, 15, "start", couleur, True)
    rapport = f / k if k > 0 else 0.0
    if rapport >= 1:
        ech = f"déformée amplifiée × {C.fr(rapport, 0 if rapport >= 10 else 1)}"
    elif rapport > 0:
        ech = f"déformée réduite au 1/{C.fr(1 / rapport, 0)}"
    else:
        ech = "déformée nulle"
    yc = yg + 66
    out += _txt(xm, yc, f"{ech} par rapport à la portée · flèches à mi-portée, celles de l'écran Streamlit", 13, couleur=COTE)
    return _svg(ident, LARGEUR_SVG, yc + 16, out, "Déformée de la poutre bois")


# ------------------------------------------------------------------ 3. la section
def section(ident: str, r: Dict) -> str:
    """La section b × h à l'échelle (cernes du bois en bout), cotée, et les contraintes de calcul à côté : σ_m
    (linéaire, compression en haut) face à f_m,d, τ (parabolique, maximale sur l'axe neutre) face à f_v,d."""
    geo, g, mat = r["geometrie"], r["grandeurs"], r["materiau"]
    b, h = float(geo["b"]), float(geo["h"])
    s = min(400.0 / h, 220.0 / b)
    bs, hs = b * s, h * s
    sx, sy = 112.0, 62.0
    clip = f"{ident}_bord_{_f(b)}x{_f(h)}".replace(".", "_")      # unique par section : deux dessins d'une page ne se mêlent pas
    out = f'<defs><clipPath id="{C.esc(clip)}"><rect x="{_f(sx)}" y="{_f(sy)}" width="{_f(bs)}" height="{_f(hs)}"/></clipPath></defs>'
    out += f'<rect x="{_f(sx)}" y="{_f(sy)}" width="{_f(bs)}" height="{_f(hs)}" fill="{BOIS_CLAIR}"/>'
    # les cernes : des cercles autour d'un cœur placé sous la section, coupés au bord
    cx, cy = sx + bs * 0.62, sy + hs * 1.18
    rmax = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
    pas = max(hs, bs) * 0.07
    cernes, rr, n = "", pas, 0
    while rr < rmax:
        cernes += f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(rr)}" fill="none" stroke="{VEINE}" stroke-width="{".7" if n % 3 else "1.1"}" opacity=".75"/>'
        n += 1
        rr += pas * (0.75 + 0.5 * ((n * 7) % 5) / 5)
    out += f'<g clip-path="url(#{C.esc(clip)})">{cernes}</g>'
    out += f'<rect x="{_f(sx)}" y="{_f(sy)}" width="{_f(bs)}" height="{_f(hs)}" fill="none" stroke="{ENCRE}" stroke-width="2.2"/>'
    # cotes
    out += (_ligne(sx, sy + hs + 6, sx, sy + hs + 44, COTE, 0.8, "4 3") + _ligne(sx + bs, sy + hs + 6, sx + bs, sy + hs + 44, COTE, 0.8, "4 3")
            + _cote_h(sx, sx + bs, sy + hs + 36, f"b = {C.nb(b)}", 16))
    out += (_ligne(sx - 6, sy, sx - 50, sy, COTE, 0.8, "4 3") + _ligne(sx - 6, sy + hs, sx - 50, sy + hs, COTE, 0.8, "4 3")
            + _cote_v(sx - 40, sy, sy + hs, f"h = {C.nb(h)}", 16))
    yn = sy + hs / 2
    # σ_m : compression en haut (à gauche de l'axe), traction en bas (à droite)
    SW = 78.0
    xs_ = sx + bs + 66 + SW
    smax = max(g["sigma_m_Ed"], g["fm_d"], 1e-12)
    ws, wf = SW * g["sigma_m_Ed"] / smax, SW * g["fm_d"] / smax
    TW = 66.0
    xt = xs_ + SW + 92
    out += _ligne(sx + bs + 6, yn, xt + TW + 96, yn, COTE, 0.9, "12 4 2 4")
    out += _poly([(xs_, sy), (xs_ - ws, sy), (xs_, yn)], TEINTE_V, ARDOISE, 1.4)
    out += _poly([(xs_, yn), (xs_ + ws, sy + hs), (xs_, sy + hs)], TEINTE_M, ROUGE, 1.4)
    out += _ligne(xs_, sy - 8, xs_, sy + hs + 8, ENCRE, 1.2)
    out += _ligne(xs_ - wf, sy, xs_ - wf, yn, ENCRE, 0.9, "4 3") + _ligne(xs_ + wf, yn, xs_ + wf, sy + hs, ENCRE, 0.9, "4 3")
    out += _txt(xs_, sy - 24, _sub("σ", "m", " (MPa)"), 15, gras=True)
    out += (_txt(xs_ - ws / 2, sy - 7, "compression", 11.5, couleur=ARDOISE, halo=True)
            + _txt(xs_ + ws / 2, sy + hs + 16, "traction", 11.5, couleur=ROUGE, halo=True))
    # τ : parabole, maximale sur l'axe neutre
    tmax = max(g["tau_Ed"], g["fv_d"], 1e-12)
    wt, wv = TW * g["tau_Ed"] / tmax, TW * g["fv_d"] / tmax
    pts = [(xt, sy)] + [(xt + wt * (1 - 4 * (i / 24 - 0.5) ** 2), yn + (i / 24 - 0.5) * hs) for i in range(25)] + [(xt, sy + hs)]
    out += _poly(pts, TEINTE_V, ARDOISE, 1.4)
    out += _ligne(xt, sy - 8, xt, sy + hs + 8, ENCRE, 1.2)
    out += _ligne(xt + wv, sy, xt + wv, sy + hs, ENCRE, 0.9, "4 3")
    out += _txt(xt + 20, sy - 24, "τ (MPa)", 15, gras=True)
    out += _txt(xt + TW + 96, yn - 7, "axe neutre", 12.5, "end", COTE)
    # les valeurs sous chaque diagramme : la contrainte (trait plein) et la résistance (tirets)
    for x, sym, ind, val, lim_ind, lim in ((xs_, "σ", "m,Ed", g["sigma_m_Ed"], "m,d", g["fm_d"]), (xt + TW / 2, "τ", "Ed", g["tau_Ed"], "v,d", g["fv_d"])):
        out += _txt(x, sy + hs + 40, _sub(sym, ind, f" = {_v(val)}"), 14, couleur=ARDOISE, gras=True)
        out += _txt(x, sy + hs + 60, _sub("f", lim_ind, f" = {_v(lim)}"), 13.5, couleur=COTE)
    # la légende
    yl = sy + hs + 102
    xc = (sx + xt + TW) / 2
    out += _txt(xc, yl, f"{C.esc(mat['classe'])} · {_section_txt(geo)} mm", 17, "middle", ENCRE, True)
    out += _txt(xc, yl + 24, f"A = {C.fr(geo['A'] / 100, 1)} cm² · I = {C.fr(geo['I'] / 1e4, 1)} cm⁴ · W = {C.fr(geo['W'] / 1e3, 1)} cm³ · "
                f"contraintes de calcul à l'ELU", 13.5, "middle", COTE)
    largeur = max(xt + TW + 110, xc * 2)
    return _svg(ident, largeur, yl + 40, out, "Section de la poutre bois et contraintes")

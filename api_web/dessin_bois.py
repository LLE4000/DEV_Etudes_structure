# -*- coding: utf-8 -*-
"""Les dessins de la poutre bois, en SVG : le SCHÉMA et les EFFORTS (charges g et q, appuis dessinés à leur longueur,
effort tranchant et moment de la combinaison ELU qui gouverne, nommée), la DÉFORMÉE (w_inst et w_net,fin, leurs limites
en tirets, les valeurs et leur « L/n »), la SECTION à l'échelle (cotes b et h, fil du bois discret — cernes d'un bois
massif, lamelles horizontales d'un lamellé-collé —, le maintien latéral, les contraintes de calcul).

Tout vient du calcul (``bois.poutre.moteur.verifier``) et des courbes qui portent ses maxima
(``bois.poutre.diagrammes``) : le dessin ne calcule aucune valeur qu'il affiche. Les vues en élévation ont la portée à
l'échelle (la hauteur de la poutre, la longueur d'appui et le débord aussi, sauf réduction dite) ; les ordonnées des
diagrammes (V, M, w, σ, τ) ont leur propre échelle. La section est à l'échelle dans ses deux directions.

Le style suit les dessins du garde-corps et de la poutre béton armé : traits noirs, cotes fines grises à traits
obliques, charges variables en rouge et permanentes en ardoise, bois en teinte claire.
"""
from typing import Dict, List, Sequence, Tuple

from bois.poutre import diagrammes as DG
from bois.poutre import donnees as D

from . import commun as C

ENCRE, COTE, ARDOISE, ROUGE = "#15181F", "#7F8C9A", "#33415C", "#9C3341"
BOIS, VEINE, BOIS_CLAIR, PLATELAGE = "#EBDDBF", "#B08D57", "#F3E9D3", "#DCC9A0"
TEINTE_G, TEINTE_Q, TEINTE_V, TEINTE_M = "#E7EBF1", "#F6E2E5", "#E2EAF5", "#F6E2E5"
POLICE = "system-ui,Arial,sans-serif"
X0, LARGEUR = 104.0, 520.0            # la portée occupe 520 unités à partir de x = 104, quelle que soit L
LARGEUR_SVG = 760.0                  # la feuille : à peu près la proportion de la toile, texte lisible
HALO = ' stroke="#FFFFFF" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
HB_MAX = 64.0          # la hauteur dessinée de la poutre au plus (une poutre haute sur une portée courte écraserait la feuille)
APPUI_MAX = 0.16       # la longueur d'appui dessinée au plus, en part de la portée
DEBORD_MAX = 0.06      # le débord dessiné au plus, en part de la portée
HD_MIN = 120.0         # la hauteur des diagrammes de contraintes au moins (une section à plat les écraserait)


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _txt(x, y, t, taille=15.0, ancre="middle", couleur=ENCRE, gras=False, extra="", halo=False) -> str:
    """Un texte ; ``halo`` : détouré de blanc, lisible quand il croise un trait (axe, limite, diagramme)."""
    poids = ' font-weight="600"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{HALO if halo else ""}{extra}>{t}</text>')


def _sub(base: str, indice: str, reste: str = "") -> str:
    """« M<sub>d</sub> = … » en SVG : l'indice plus petit et plus bas, puis la ligne reprend sa hauteur."""
    return f"{base}<tspan font-size='72%' dy='4'>{indice}</tspan><tspan dy='-4'>{reste}</tspan>"


def _symbole(symbole: str, reste: str = "") -> str:
    """Un symbole du moteur (« σ_m,d », « k_crit·f_m,d ») écrit en SVG, chaque indice plus petit et plus bas ; ``reste`` :
    ce qui suit (« = 14,77 »)."""
    out, morceaux = "", symbole.split("·")
    for i, m in enumerate(morceaux):
        base, _, indice = m.partition("_")
        suite = "·" if i < len(morceaux) - 1 else reste
        out += _sub(C.esc(base), C.esc(indice), suite) if indice else f"{C.esc(base)}{suite}"
    return out


def _ligne(x1, y1, x2, y2, couleur=ENCRE, ep=1.0, tirets="") -> str:
    d = f' stroke-dasharray="{tirets}"' if tirets else ""
    return f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"{d}/>'


def _poly(points: Sequence[Tuple[float, float]], fond="none", trait=ENCRE, ep=1.4, ferme=True, opacite=None) -> str:
    pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in points)
    balise = "polygon" if ferme else "polyline"
    op = f' fill-opacity="{opacite}"' if opacite is not None else ""
    return f'<{balise} points="{pts}" fill="{fond}"{op} stroke="{trait}" stroke-width="{_f(ep)}" stroke-linejoin="round"/>'


def _rect(x, y, w, h, fond="none", trait=ENCRE, ep=1.4, extra="") -> str:
    return (f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" fill="{fond}" stroke="{trait}" '
            f'stroke-width="{_f(ep)}"{extra}/>')


def _fleche(x, y_haut, y_bas, couleur, ep=1.3, tete=7.0) -> str:
    """Une flèche verticale vers le bas, pointe en (x, y_bas)."""
    return (_ligne(x, y_haut, x, y_bas - tete + 1, couleur, ep)
            + _poly([(x, y_bas), (x - tete * 0.45, y_bas - tete), (x + tete * 0.45, y_bas - tete)], couleur, couleur, 0.6))


def _large(texte: str, taille: float) -> float:
    """La largeur d'un texte, ESTIMÉE (un demi-cadratin par caractère : mesuré de 0,40 à 0,56 dans le navigateur)."""
    return 0.5 * taille * len(texte)


def _cote_h(x1, x2, y, texte, taille=15.0) -> str:
    """Une cote horizontale : ligne, deux traits obliques, le texte au-dessus — ou à droite de la cote quand elle est
    trop courte pour le porter (une section étroite)."""
    t = 5.0
    out = (_ligne(x1, y, x2, y, COTE, 1.0) + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, 1.2)
           + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, 1.2))
    if x2 - x1 < _large(texte, taille) + 14:
        return out + _txt(x2 + 12, y + taille * 0.35, texte, taille, "start")
    return out + _txt((x1 + x2) / 2, y - 7, texte, taille)


def _cote_v(x, y1, y2, texte, taille=15.0) -> str:
    """Une cote verticale : le texte tourné le long de la ligne, à sa gauche — ou écrit à l'horizontale, à gauche de
    la cote, quand elle est trop courte pour le porter (une section à plat)."""
    t = 5.0
    xt = x - 8
    out = (_ligne(x, y1, x, y2, COTE, 1.0) + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, 1.2)
           + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, 1.2))
    if y2 - y1 < _large(texte, taille) + 14:
        return out + _txt(x - 10, (y1 + y2) / 2 + taille * 0.35, texte, taille, "end")
    return out + _txt(xt, (y1 + y2) / 2, texte, taille, extra=f' transform="rotate(-90 {_f(xt)} {_f((y1 + y2) / 2)})"')


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
    hh, dl = 20.0 * taille, 14.0 * taille
    out = _poly([(x, y), (x - dl, y + hh), (x + dl, y + hh)], "#FFFFFF", ENCRE, 1.5)
    if mobile:
        r = 3.4 * taille
        out += (f'<circle cx="{_f(x - 7 * taille)}" cy="{_f(y + hh + r)}" r="{_f(r)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="1.2"/>'
                f'<circle cx="{_f(x + 7 * taille)}" cy="{_f(y + hh + r)}" r="{_f(r)}" fill="#FFFFFF" stroke="{ENCRE}" stroke-width="1.2"/>')
        return out + _sol(x, y + hh + 2 * r, 44 * taille)
    return out + _sol(x, y + hh, 44 * taille)


def _veines(x1, x2, y1, y2) -> str:
    """Le fil du bois le long de la poutre : deux ou trois traits légèrement ondulés, discrets."""
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
        out += f'<path d="{d}" fill="none" stroke="{VEINE}" stroke-width="0.8" opacity=".6"/>'
    return out


def _lamelles_long(x1, x2, y1, y2, n: int) -> str:
    """Les plans de collage d'un lamellé-collé vus en élévation : ``n`` lamelles horizontales ; quand elles seraient
    trop serrées pour se lire (moins de 3 unités), un plan sur deux, sur trois… est tracé — le dessin dit « lamellé »
    sans noircir la poutre."""
    if n < 2:
        return ""
    pas = 1
    while (y2 - y1) * pas / n < 3.0 and pas < n:
        pas += 1
    return "".join(_ligne(x1, y1 + (y2 - y1) * i / n, x2, y1 + (y2 - y1) * i / n, VEINE, 0.7) for i in range(pas, n, pas))


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


def _lamelles(r: Dict) -> int:
    """Le nombre de lamelles dessinées d'un lamellé-collé (lamelles de 40 mm environ) ; 0 pour un bois massif."""
    if r["materiau"]["type"] != D.LAMELLE:
        return 0
    return max(2, round(r["geometrie"]["h"] / D.EPAISSEUR_LAMELLE))


def _bandes_charges(ch: Dict, y_haut: float, x1: float, x2: float) -> Tuple[str, float]:
    """Les charges réparties : la bande q (variable) au-dessus de la bande g (permanente, poids propre compris),
    hauteurs proportionnelles à leur valeur (au moins lisibles) ; rend (svg, y du bas des bandes)."""
    g, q = ch["g"], ch["q"]
    total = g + q
    if total <= 0:
        return _txt((x1 + x2) / 2, y_haut + 30, "aucune charge", 15, couleur=COTE), y_haut + 44
    HT = 72.0
    hq = max(14.0, HT * q / total) if q > 0 else 0.0
    hg = max(14.0, HT * g / total) if g > 0 else 0.0
    out, y = "", y_haut
    n = 17
    for valeur, hb, teinte, couleur, nom in ((q, hq, TEINTE_Q, ROUGE, "q"), (g, hg, TEINTE_G, ARDOISE, "g")):
        if hb <= 0:
            continue
        out += f'<rect x="{_f(x1)}" y="{_f(y)}" width="{_f(x2 - x1)}" height="{_f(hb)}" fill="{teinte}"/>'
        out += _ligne(x1, y, x2, y, couleur, 1.6)
        for i in range(n):
            xa = x1 + (x2 - x1) * i / (n - 1)
            out += _fleche(xa, y, y + hb, couleur, 1.1, min(7.0, hb * 0.55))
        out += _txt(x2 + 14, y + hb / 2 + 5, f"{nom} = {_v(valeur, 2)} kN/m", 14.5, "start", couleur, True)
        y += hb
    return out, y


# ------------------------------------------------------------------ 1. le schéma et les efforts
def _hachures(xs: List[float], y0: float, ys: List[float], couleur: str) -> str:
    return "".join(_ligne(x, y0, x, y, couleur, 0.6) for x, y in zip(xs, ys) if abs(y - y0) > 1.5)


def poutre(ident: str, r: Dict) -> str:
    """La feuille de la poutre : en haut le schéma à l'échelle des longueurs (charges g et q réparties, poutre de
    hauteur h, appuis dessinés à leur longueur a — articulation en A, rouleaux en B —, débord, réactions) ; dessous,
    alignés sur la même portée, l'effort tranchant V(x) (positif au-dessus de l'axe) et le moment M(x) (positif sous
    l'axe, côté des fibres tendues) de la combinaison ELU qui gouverne, nommée ; la portée cotée."""
    k = _echelle(r)
    geo, ch, ef, ap, co = r["geometrie"], r["charges"], r["efforts"], r["appui"], r["coefficients"]
    comb = next(c for c in r["combinaisons"] if c["cle"] == ef["combinaison"])
    x1, x2 = X0, X0 + LARGEUR
    xm = (x1 + x2) / 2
    # à l'échelle de la portée, sauf une poutre très haute, un appui très long ou un grand débord sur une portée courte :
    # réduits pour que la feuille reste lisible — et c'est écrit dessous
    a_vrai, hb_vrai, d_vrai = ap["a"] * k, geo["h"] * k, ap["debord"] * k
    a = max(min(a_vrai, APPUI_MAX * LARGEUR), 5.0)
    hb = min(max(hb_vrai, 4.0), HB_MAX)
    deb = min(d_vrai, DEBORD_MAX * LARGEUR)
    reduits = [t for t, oui in (("la hauteur de la poutre", hb_vrai > HB_MAX), ("la longueur d'appui", a_vrai > APPUI_MAX * LARGEUR),
                                ("le débord", d_vrai > DEBORD_MAX * LARGEUR)) if oui]
    out, y_b = _bandes_charges(ch, 24.0, x1, x2)
    y_p = y_b + 2
    xa, xb = x1 - a / 2 - deb, x2 + a / 2 + deb
    out += _rect(xa, y_p, xb - xa, hb, BOIS, ENCRE, 2.0)
    out += _lamelles_long(xa, xb, y_p, y_p + hb, _lamelles(r)) if _lamelles(r) else _veines(xa, xb, y_p, y_p + hb)
    # les appuis, à leur longueur a : une cale sous la poutre, puis le symbole (articulation, rouleaux)
    y_a = y_p + hb
    for x in (x1, x2):
        out += _rect(x - a / 2, y_a, a, 7.0, ARDOISE, ARDOISE, 0.6)
    y_a += 7
    out += _appui(x1, y_a, False) + _appui(x2, y_a, True)
    out += _txt(x1 - 22, y_a + 18, "A", 17, "end", gras=True) + _txt(x2 + 22, y_a + 18, "B", 17, "start", gras=True)
    out += _txt(x1 + max(a / 2, 16) + 8, y_a + 13, f"a = {C.nb(ap['a'])} mm" + (f" · débord {C.nb(ap['debord'])}" if ap["debord"] else ""),
                13, "start", ARDOISE, True, halo=True)
    y_r = y_a + 52
    reac = _sub("R", "d", f" = {_v(ef['R'])} kN")
    out += _txt(x1, y_r, reac, 14.5, couleur=ROUGE, gras=True) + _txt(x2, y_r, reac, 14.5, couleur=ROUGE, gras=True)
    # V(x)
    cb = DG.courbes(r, 75)
    xs = [x1 + LARGEUR * x / geo["L"] for x in cb["x"]]
    HV, HM = 58.0, 90.0
    yv = y_r + 30 + HV
    vmax = max(abs(ef["V_d"]), 1e-12)
    yv_s = [yv - HV * v / vmax for v in cb["V"]]
    out += _hachures(xs, yv, yv_s, ARDOISE)
    out += _poly([(x1, yv)] + list(zip(xs, yv_s)) + [(x2, yv)], TEINTE_V, ARDOISE, 1.6, opacite=".8")
    out += _ligne(x1 - 8, yv, x2 + 8, yv, ENCRE, 1.2)
    out += _txt(x1 - 20, yv - 4, "V", 18, "end", ARDOISE, True) + _txt(x1 - 20, yv + 14, "kN", 13, "end", COTE)
    out += _txt(x1 + 8, yv - HV - 7, _signe(ef["V_d"], "+"), 15, "start", ARDOISE, True)
    out += _txt(x2 - 8, yv + HV + 19, _signe(ef["V_d"], "−"), 15, "end", ARDOISE, True)
    # M(x)
    ym = yv + HV + 42
    mmax = max(abs(ef["M_d"]), 1e-12)
    ym_s = [ym + HM * m / mmax for m in cb["M"]]
    out += _hachures(xs, ym, ym_s, ROUGE)
    out += _poly([(x1, ym)] + list(zip(xs, ym_s)) + [(x2, ym)], TEINTE_M, ROUGE, 1.6, opacite=".8")
    out += _ligne(x1 - 8, ym, x2 + 8, ym, ENCRE, 1.2)
    out += _txt(x1 - 20, ym + 12, "M", 18, "end", ROUGE, True) + _txt(x1 - 20, ym + 30, "kN·m", 13, "end", COTE)
    out += _txt(xm, ym + HM + 23, _sub("M", "d", f" = {_v(ef['M_d'])} kN·m"), 15.5, couleur=ROUGE, gras=True)
    # la mi-portée, la portée
    out += _ligne(xm, yv - HV - 12, xm, ym + HM + 6, COTE, 0.8, "6 4") + _txt(xm + 6, yv - HV - 1, "L/2", 13, "start", COTE, halo=True)
    yc = ym + HM + 62
    out += (_ligne(x1, y_a + 30, x1, yc + 6, COTE, 0.6, "4 3") + _ligne(x2, y_a + 30, x2, yc + 6, COTE, 0.6, "4 3")
            + _cote_h(x1, x2, yc, f"L = {C.fr(geo['L_mm'], 0)}", 16))
    # la combinaison qui gouverne, écrite avec ses termes (ceux du moteur)
    if comb["cle"] == "G":
        pd = f" = {C.nb(co['gamma_G'])} × {_v(ch['g'], 3)} = {_v(comb['p_d'], 3)} kN/m"
    else:
        pd = f" = {C.nb(co['gamma_G'])} × {_v(ch['g'], 3)} + {C.nb(co['gamma_Q'])} × {_v(ch['q'], 3)} = {_v(comb['p_d'], 3)} kN/m"
    out += _txt(xm, yc + 36, _sub("p", "d", pd), 15.5, couleur=ROUGE, gras=True)
    out += _txt(xm, yc + 57, f"ELU · combinaison {C.esc(comb['nom'])} ({C.esc(comb['libelle_duree'])}) · V et M à leur propre échelle", 13, couleur=COTE)
    if reduits:
        out += _txt(xm, yc + 76, f"dessin réduit pour la lisibilité : {', '.join(reduits)} (la portée reste à l'échelle)", 13, couleur=ROUGE)
        yc += 19
    return _svg(ident, LARGEUR_SVG, yc + 70, out, "Schéma de la poutre bois et diagrammes des efforts")


# ------------------------------------------------------------------ 2. la déformée
def _rapport(x) -> str:
    """« L/610 » : la flèche rapportée à la portée (le moteur donne L/w) ; rien sans flèche vers le bas."""
    return f"L/{C.fr(x, 0)}" if x else ""


def deformee(ident: str, r: Dict) -> str:
    """Les déformées instantanée et finale nette le long de la portée, leurs limites L/n en tirets sur le tiers
    central, les valeurs à mi-portée et leur « L/n » ; ordonnées amplifiées, le rapport écrit sur la vue."""
    fl, geo = r["fleches"], r["geometrie"]
    cb = DG.courbes(r, 49)
    k = _echelle(r)
    x1, x2 = X0, X0 + LARGEUR
    xs = [x1 + LARGEUR * x / geo["L"] for x in cb["x"]]
    HW = 330.0
    bas = max(fl["w_inst"], fl["w_net_fin"], fl["lim_inst"], fl["lim_fin"], max(cb["w_net_fin"]))
    haut = max(0.0, -min(cb["w_net_fin"]))                        # une poutre qui reste cambrée monte au-dessus de l'axe
    f = HW / (bas + haut) if bas + haut > 0 else 0.0
    y0 = 52.0 + haut * f
    out = _rect(x1 - 4, y0 - 5, LARGEUR + 8, 10, BOIS, "none", 0, ' opacity=".5"')
    out += _ligne(x1 - 16, y0, x2 + 16, y0, COTE, 1.0, "10 4 2 4")
    out += _appui(x1, y0 + 5, False, 0.8) + _appui(x2, y0 + 5, True, 0.8)
    out += _txt(x1 - 20, y0 + 22, "A", 16, "end", gras=True) + _txt(x2 + 20, y0 + 22, "B", 16, "start", gras=True)
    # les limites, sur le tiers central ; leurs textes ne se chevauchent pas
    xa, xb = x1 + LARGEUR / 3, x1 + 2 * LARGEUR / 3
    limites = sorted([(y0 + fl["lim_inst"] * f, f"L/{fl['n_inst']} = {_v(fl['lim_inst'], 1)} mm", ARDOISE),
                      (y0 + fl["lim_fin"] * f, f"L/{fl['n_fin']} = {_v(fl['lim_fin'], 1)} mm", ROUGE)], key=lambda t: t[0])
    y_txt = -1e9
    for y, t, couleur in limites:
        out += _ligne(xa, y, xb, y, couleur, 1.3, "7 4")
        y_txt = max(y + 5, y_txt + 17)
        out += _txt(xb + 8, y_txt, t, 13, "start", couleur, True, halo=True)
    # les déformées
    for cle, couleur in (("w_inst", ARDOISE), ("w_net_fin", ROUGE)):
        out += _poly(list(zip(xs, [y0 + w * f for w in cb[cle]])), "none", couleur, 2.4, ferme=False)
    xm = (x1 + x2) / 2
    out += _ligne(xm, y0 + 8, xm, y0 + bas * f + 12, COTE, 0.8, "6 4")
    for w, couleur in ((fl["w_inst"], ARDOISE), (fl["w_net_fin"], ROUGE)):
        out += f'<circle cx="{_f(xm)}" cy="{_f(y0 + w * f)}" r="3.4" fill="#FFFFFF" stroke="{couleur}" stroke-width="1.8"/>'
    # les valeurs, en légende : la flèche, son « L/n », sa limite, son taux
    vs = {v["cle"]: v for v in r["verifications"]}
    yg = y0 + bas * f + 54
    lignes = (("w_inst", fl["w_inst"], fl["rapport_inst"], fl["n_inst"], vs["fleche_inst"], ARDOISE, "inst"),
              ("w_net_fin", fl["w_net_fin"], fl["rapport_fin"], fl["n_fin"], vs["fleche_fin"], ROUGE, "net,fin"))
    for i, (_cle, w, rapport, n, v, couleur, nom) in enumerate(lignes):
        yy = yg + 27 * i
        out += _ligne(x1 - 30, yy - 5, x1, yy - 5, couleur, 2.6)
        rp = f" = {_rapport(rapport)}" if rapport else ""
        signe = "≤" if v["taux"] <= 1 else "&gt;"
        val = _sub("w", nom, f" = {_v(w, 2)} mm{rp}  {signe}  L/{n} = {_v(v['limite'], 1)} mm  ·  {C.pc(C.pct(v['taux']))} %")
        out += _txt(x1 + 10, yy, val, 15, "start", couleur, True)
    rapport = f / k if k > 0 else 0.0
    if rapport >= 1:
        ech = f"déformée amplifiée × {C.fr(rapport, 0 if rapport >= 10 else 1)}"
    elif rapport > 0:
        ech = f"déformée réduite au 1/{C.fr(1 / rapport, 0)}"
    else:
        ech = "déformée nulle"
    yc = yg + 62
    out += _txt(xm, yc, f"{ech} par rapport à la portée · flexion et effort tranchant", 13, couleur=COTE)
    if fl["w_c"]:
        yc += 18
        out += _txt(xm, yc, f"contre-flèche de {_v(fl['w_c'], 1)} mm déduite de la flèche finale (prise parabolique le long de la portée)", 13, couleur=COTE)
    return _svg(ident, LARGEUR_SVG, yc + 16, out, "Déformée de la poutre bois")


# ------------------------------------------------------------------ 3. la section
def _maintien(r: Dict, sx: float, sy: float, bs: float, hs: float) -> Tuple[str, float]:
    """Le maintien latéral, dessiné sur la section : un platelage fixé sur la fibre comprimée quand il est continu ;
    sinon rien n'est posé et la légende le dit. Rend (svg, débord du platelage de chaque côté de la section)."""
    dv = r["deversement"]
    if dv["type"] != "continu":
        return "", 0.0
    ep, dep = 13.0, min(40.0, max(22.0, bs * 0.4))
    x1, x2, y = sx - dep, sx + bs + dep, sy - ep
    out = _rect(x1, y, x2 - x1, ep, PLATELAGE, ENCRE, 1.3)
    out += _ligne(x1 + 4, y + ep / 2, x2 - 4, y + ep / 2, VEINE, 0.7)
    # deux fixations (une seule sur une section étroite) : la tête sur le platelage, la tige dans la poutre
    for x in ((sx + bs * 0.28, sx + bs * 0.72) if bs >= 34 else (sx + bs / 2,)):
        out += _ligne(x, y - 1.5, x, y + ep + 12, ENCRE, 1.5) + _ligne(x - 4, y - 1.5, x + 4, y - 1.5, ENCRE, 2.0)
    return out, dep


def section(ident: str, r: Dict) -> str:
    """La section b × h à l'échelle, cotée (cernes d'un bois massif, lamelles horizontales d'un lamellé-collé, le
    platelage fixé d'un maintien continu), et à côté les contraintes de calcul de la combinaison qui gouverne chacune :
    σ_m (linéaire, compression en haut) face à sa limite k_crit·f_m,d, τ (parabolique, maximale sur l'axe neutre) face
    à f_v,d. Les diagrammes sont tracés sur la hauteur de la section ; quand elle est à plat (un madrier de
    plateforme), ils gardent une hauteur lisible, centrée sur l'axe neutre — et la légende le dit."""
    geo, mat, dv = r["geometrie"], r["materiau"], r["deversement"]
    vs = {v["cle"]: v for v in r["verifications"]}
    b, h = float(geo["b"]), float(geo["h"])
    s = min(380.0 / h, 230.0 / b)
    a_plat = h * s < HD_MIN
    if a_plat:
        s = min(s, 160.0 / b)                                     # moins large : la feuille garde sa largeur (le téléphone la lit)
    bs, hs = b * s, h * s
    hd = max(hs, HD_MIN)                                          # la hauteur des diagrammes
    sx = 150.0 if a_plat else 138.0
    haut = 74.0                                                   # le haut des diagrammes (et de la section, sauf à plat)
    yn = haut + hd / 2                                            # l'axe neutre
    sy = yn - hs / 2                                              # le haut de la section
    bas = haut + hd                                               # le bas des diagrammes (≥ celui de la section)
    clip = f"{ident}_bord_{_f(b)}x{_f(h)}".replace(".", "_")      # unique par section : deux dessins d'une page ne se mêlent pas
    out = f'<defs><clipPath id="{C.esc(clip)}"><rect x="{_f(sx)}" y="{_f(sy)}" width="{_f(bs)}" height="{_f(hs)}"/></clipPath></defs>'
    out += f'<rect x="{_f(sx)}" y="{_f(sy)}" width="{_f(bs)}" height="{_f(hs)}" fill="{BOIS_CLAIR}"/>'
    n_lam = _lamelles(r)
    if n_lam:
        # les lamelles horizontales, et dans chacune un cerne à peine marqué
        fil = "".join(_ligne(sx, sy + hs * i / n_lam, sx + bs, sy + hs * i / n_lam, VEINE, 0.9) for i in range(1, n_lam)) if hs / n_lam >= 3 else ""
        if hs / n_lam >= 9:
            for i in range(n_lam):
                yy = sy + hs * (i + 0.5) / n_lam
                sens = 1 if i % 2 else -1
                fil += (f'<path d="M{_f(sx)} {_f(yy)} Q{_f(sx + bs / 2)} {_f(yy + sens * hs / n_lam * 0.32)} {_f(sx + bs)} {_f(yy)}" fill="none" '
                        f'stroke="{VEINE}" stroke-width="0.6" opacity=".45"/>')
    else:
        # les cernes : des cercles autour d'un cœur placé sous la section, coupés au bord, discrets
        cx, cy = sx + bs * 0.62, sy + hs * 1.18
        rmax = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        pas = max(hs, bs) * 0.07
        fil, rr, n = "", pas, 0
        while rr < rmax:
            fil += f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(rr)}" fill="none" stroke="{VEINE}" stroke-width="{".6" if n % 3 else "1"}" opacity=".5"/>'
            n += 1
            rr += pas * (0.75 + 0.5 * ((n * 7) % 5) / 5)
    out += f'<g clip-path="url(#{C.esc(clip)})">{fil}</g>'
    out += f'<rect x="{_f(sx)}" y="{_f(sy)}" width="{_f(bs)}" height="{_f(hs)}" fill="none" stroke="{ENCRE}" stroke-width="2.2"/>'
    m_svg, gauche = _maintien(r, sx, sy, bs, hs)
    out += m_svg
    # le maintien latéral, en mots au-dessus de la section
    if dv["type"] == "continu":
        mt = "maintien continu de la fibre comprimée"
    else:
        l_ef = _symbole("l_ef", f" = {C.fr(dv['l_ef'], 0)} mm")
        mt = (f"maintien latéral aux appuis seulement · {l_ef}" if dv["type"] == "appuis"
              else f"entretoises tous les {C.fr(dv['entraxe'], 0)} mm · {l_ef}")
    out += _txt(max(sx + bs / 2, 190.0), sy - 34, mt, 13, "middle", ARDOISE, True)
    # cotes
    out += (_ligne(sx, sy + hs + 6, sx, sy + hs + 44, COTE, 0.8, "4 3") + _ligne(sx + bs, sy + hs + 6, sx + bs, sy + hs + 44, COTE, 0.8, "4 3")
            + _cote_h(sx, sx + bs, sy + hs + 36, f"b = {C.nb(b)}", 16))
    # (la ligne de rappel du haut part du bord du platelage, quand il y en a un : elle ne le traverse pas)
    xh = sx - gauche - 16
    out += (_ligne(sx - gauche - 5, sy, xh - 10, sy, COTE, 0.8, "4 3") + _ligne(sx - 6, sy + hs, xh - 10, sy + hs, COTE, 0.8, "4 3")
            + _cote_v(xh, sy, sy + hs, f"h = {C.nb(h)}", 16))
    # σ_m : compression en haut (à gauche de l'axe), traction en bas (à droite)
    vm, vt = vs["flexion"], vs["cisaillement"]
    SW = 78.0
    xs_ = max(sx + bs + (40 if a_plat else 66) + SW, 330.0 + SW)
    smax = max(vm["valeur"], vm["limite"], 1e-12)
    ws, wf = SW * vm["valeur"] / smax, SW * vm["limite"] / smax
    TW = 66.0
    xt = xs_ + SW + 96
    out += _ligne(sx + bs + 6, yn, xt + TW + 96, yn, COTE, 0.9, "12 4 2 4")
    out += _poly([(xs_, haut), (xs_ - ws, haut), (xs_, yn)], TEINTE_V, ARDOISE, 1.4)
    out += _poly([(xs_, yn), (xs_ + ws, bas), (xs_, bas)], TEINTE_M, ROUGE, 1.4)
    out += _ligne(xs_, haut - 8, xs_, bas + 8, ENCRE, 1.2)
    out += _ligne(xs_ - wf, haut, xs_ - wf, yn, ENCRE, 0.9, "4 3") + _ligne(xs_ + wf, yn, xs_ + wf, bas, ENCRE, 0.9, "4 3")
    out += _txt(xs_, haut - 14, _sub("σ", "m", " (N/mm²)"), 14.5, gras=True)
    # (chaque mot du côté LIBRE de l'axe : il ne croise ni le triangle ni sa limite)
    out += (_txt(xs_ + 7, haut + 16, "compression", 13, "start", ARDOISE)
            + _txt(xs_ - 7, bas - 8, "traction", 13, "end", ROUGE))
    # τ : parabole, maximale sur l'axe neutre
    tmax = max(vt["valeur"], vt["limite"], 1e-12)
    wt, wv = TW * vt["valeur"] / tmax, TW * vt["limite"] / tmax
    pts = [(xt, haut)] + [(xt + wt * (1 - 4 * (i / 24 - 0.5) ** 2), yn + (i / 24 - 0.5) * hd) for i in range(25)] + [(xt, bas)]
    out += _poly(pts, TEINTE_V, ARDOISE, 1.4)
    out += _ligne(xt, haut - 8, xt, bas + 8, ENCRE, 1.2)
    out += _ligne(xt + wv, haut, xt + wv, bas, ENCRE, 0.9, "4 3")
    out += _txt(xt + TW / 2, haut - 14, "τ (N/mm²)", 14.5, gras=True)
    out += _txt(xt + TW + 96, yn - 7, "axe neutre", 13, "end", COTE)
    # les valeurs sous chaque diagramme : la contrainte (trait plein) et sa limite (tirets), avec leurs symboles du moteur
    noms = {c["cle"]: c["nom"] for c in r["combinaisons"]}
    for x, v in ((xs_, vm), (xt + TW / 2, vt)):
        out += _txt(x, bas + 34, _symbole(v["symbole"], f" = {_v(v['valeur'])}"), 14, couleur=ARDOISE, gras=True)
        out += _txt(x, bas + 54, _symbole(v["symbole_limite"], f" = {_v(v['limite'])}"), 13.5, couleur=COTE)
        out += _txt(x, bas + 73, C.esc(noms[v["combinaison"]]), 13, couleur=COTE)
    # la légende
    yl = bas + 112
    xc = max(xt + TW + 104, LARGEUR_SVG) / 2                      # la légende au milieu de la feuille
    lam = " · lamelles horizontales" if n_lam else ""
    out += _txt(xc, yl, f"{C.esc(mat['classe'])} · {C.esc(geo['designation'])} mm", 17, "middle", ENCRE, True)
    out += _txt(xc, yl + 23, f"{C.esc(mat['libelle_type'])}{lam} · A = {C.fr(geo['A'] / 100, 1)} cm² · W = {C.fr(geo['W'] / 1e3, 1)} cm³ · "
                f"I = {C.fr(geo['I'] / 1e4, 0)} cm⁴", 13.5, "middle", COTE)
    out += _txt(xc, yl + 43, "contraintes de calcul à l'ELU, chacune dans la combinaison qui la gouverne ; limites en tirets", 13, "middle", COTE)
    if a_plat:
        yl += 18
        out += _txt(xc, yl + 43, "section à plat : diagrammes tracés sur une hauteur agrandie (la section est à l'échelle)", 13, "middle", COTE)
    largeur = max(xt + TW + 104, LARGEUR_SVG)
    return _svg(ident, largeur, yl + 58, out, "Section de la poutre bois et contraintes de calcul")

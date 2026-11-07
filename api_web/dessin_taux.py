# -*- coding: utf-8 -*-
"""Les dessins du module « Taux d'armature », à l'échelle (mm), à partir de ce que le moteur ``beton/taux`` a calculé :

- ``coupe`` : la section (âme et talons), le béton en gris, chaque barre à la position que l'écran Streamlit lui donne
  (``GE.centres`` : axes à c + Ø/2 de la face, de x = −b/2 + c à b/2 − c), dans la couleur de sa famille (le dégradé
  de l'écran), le contour d'étrier compté par le module (rectangle de l'âme à l'enrobage), les cotes et la légende ;
- ``elevation`` : la poutre en élévation, les barres de chaque famille de « De » à « À », les étriers de chaque zone
  à leur espacement (le nombre que le module compte), la chaîne des zones, puis les barres développées famille par
  famille. Les longueurs sont à l'échelle ; les hauteurs sont amplifiées (écrit sur le dessin) quand la poutre est
  très élancée ;
- ``tableau`` : le tableau récapitulatif des armatures de l'écran (un bloc HTML, pas un dessin).

Le style suit les dessins du site (api_web/dessin_poutre.py, acier/garde_corps/schemas.py) : traits noirs, béton gris
clair, cotes fines à traits obliques. Rien n'est calculé ici : les positions, longueurs et masses viennent du moteur.
Tout texte venu de la note (nom de zone, de type d'étrier) est échappé.
"""
from html import escape
from typing import Dict, List

ENCRE, BETON, COTE, ETRIER, FANTOME = "#15181F", "#E6EAEF", "#7F8C9A", "#C08A00", "#9AA4B4"
POLICE = "system-ui,Arial,sans-serif"
# le bord sombre des barres, par groupe de lits : il garde lisibles les couleurs pâles du dégradé de l'écran
BORDS = {"bas": "#7A1F1F", "haut": "#1D2F6B", "cote": "#333B48"}
GROUPES = {"bas": "lits du bas", "haut": "lits du haut", "cote": "latérales"}


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _mm(x: float) -> str:
    """Une cote en mm, sans décimale inutile (« 600 », « 52,5 »)."""
    s = f"{x:.1f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _txt(x, y, t, taille, ancre="middle", couleur=ENCRE, gras=True, extra="") -> str:
    poids = ' font-weight="600"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{t}</text>')


def _ligne(x1, y1, x2, y2, couleur=COTE, ep=1.0, extra="") -> str:
    return f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"{extra}/>'


def _cote_h(x1, x2, y, texte, fs, ep) -> str:
    """Une cote horizontale : ligne, deux traits obliques, le nombre au-dessus."""
    t = 0.32 * fs
    return (_ligne(x1, y, x2, y, COTE, ep) + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, 1.2 * ep)
            + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, 1.2 * ep) + _txt((x1 + x2) / 2, y - 0.38 * fs, texte, fs))


def _cote_v(x, y1, y2, texte, fs, ep, a_droite=False) -> str:
    """Une cote verticale : le nombre tourné le long de la ligne."""
    t = 0.32 * fs
    xt = x + (0.95 * fs if a_droite else -0.42 * fs)
    yt = (y1 + y2) / 2
    return (_ligne(x, y1, x, y2, COTE, ep) + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, 1.2 * ep)
            + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, 1.2 * ep)
            + _txt(xt, yt, texte, fs, extra=f' transform="rotate(-90 {_f(xt)} {_f(yt)})"'))


def _attache(x1, y1, x2, y2, ep) -> str:
    return _ligne(x1, y1, x2, y2, COTE, 0.8 * ep, ' stroke-dasharray="4 3"')


def _svg(ident: str, vb: List[float], corps: str, titre: str) -> str:
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{" ".join(_f(v) for v in vb)}" width="100%" '
            f'role="img" aria-label="{escape(titre)}">{corps}</svg>')


# ============================================================ la coupe
def coupe(ident: str, res: Dict) -> str:
    """La section à l'échelle (mm) : béton, contour d'étrier compté, barres, cotes et légende des familles."""
    g = res["geometrie"]
    h = g["h"] * 10
    poly = [(x * 10, h - y * 10) for x, y in g["polygone"]]
    xs = [p[0] for p in poly]
    x_min, x_max = min(xs), max(xs)
    larg = x_max - x_min
    E = max(larg, h)
    fs = max(0.04 * E, 9.0)
    ep = max(E / 400, 0.6)
    out = []
    # les talons que le module NE COMPTE PAS (deux talons : défaut D2) : en fantôme, avec leur raison
    p = res["poutre"]
    if g.get("talons_ignores"):
        b2 = g["b"] * 10 / 2
        for cote, sgn in (("talon_gauche", -1), ("talon_droit", 1)):
            tb, th = p[cote]["b"] * 10, p[cote]["h"] * 10
            x0 = -b2 - tb if sgn < 0 else b2
            out.append(f'<rect x="{_f(x0)}" y="{_f(h - th)}" width="{_f(tb)}" height="{_f(th)}" fill="none" stroke="{FANTOME}" '
                       f'stroke-width="{_f(ep * 1.4)}" stroke-dasharray="{_f(3 * ep)} {_f(2.5 * ep)}"/>')
            x_min, x_max = min(x_min, x0), max(x_max, x0 + tb)
    pts = " ".join(f"{_f(x)},{_f(y)}" for x, y in poly)
    out.append(f'<polygon points="{pts}" fill="{BETON}" stroke="{ENCRE}" stroke-width="{_f(2.2 * ep)}" stroke-linejoin="miter"/>')
    # le contour d'étrier que le module compte : le rectangle de l'âme à l'enrobage (stirrup_length_m), au Ø réel
    c = g["c"] * 10
    b2 = g["b"] * 10 / 2
    ds = sorted({e["d"] for e in res["etriers"]})
    if ds:
        de = ds[-1]
        out.append(f'<rect x="{_f(-b2 + c)}" y="{_f(c)}" width="{_f(2 * b2 - 2 * c)}" height="{_f(h - 2 * c)}" rx="{_f(1.5 * de)}" fill="none" '
                   f'stroke="{ETRIER}" stroke-width="{_f(de)}" stroke-opacity=".85"/>')
    # les barres : la position et la couleur de l'écran, un bord sombre par groupe
    for f in res["familles"]:
        r = f["d"] / 2
        for x, y in f["barres"]:
            out.append(f'<circle cx="{_f(x * 10)}" cy="{_f(h - y * 10)}" r="{_f(r)}" fill="{f["couleur"]}" stroke="{BORDS[f["groupe"]]}" '
                       f'stroke-width="{_f(max(r * 0.16, 0.6 * ep))}"/>')
    # les cotes : en bas la chaîne (talons, âme) puis l'ensemble ; à gauche h (et le talon gauche) ; à droite le talon droit
    yb = h + 1.6 * fs
    xl, xr = -b2, b2
    chaine = []
    if g["bg"] > 0:
        chaine.append((xl - g["bg"] * 10, xl))
    chaine.append((xl, xr))
    if g["bd"] > 0:
        chaine.append((xr, xr + g["bd"] * 10))
    for a, b in chaine:
        out.append(_attache(a, h + 0.3 * fs, a, yb + 0.4 * fs, ep) + _attache(b, h + 0.3 * fs, b, yb + 0.4 * fs, ep)
                   + _cote_h(a, b, yb, _mm(b - a), 0.85 * fs, ep))
    if len(chaine) > 1:
        a, b = chaine[0][0], chaine[-1][1]
        yb2 = yb + 1.9 * fs
        out.append(_attache(a, yb + 0.4 * fs, a, yb2 + 0.4 * fs, ep) + _attache(b, yb + 0.4 * fs, b, yb2 + 0.4 * fs, ep)
                   + _cote_h(a, b, yb2, _mm(b - a), 0.85 * fs, ep))
        yb = yb2
    # les talons fantômes (D2) : leur raison SOUS la chaîne de cotes (posée au-dessus, elle recouvrait la cote de l'âme)
    if g.get("talons_ignores"):
        out.append(_txt((x_min + x_max) / 2, yb + 1.5 * fs, "talons non comptés par le module (défaut D2)", 0.8 * fs, couleur=FANTOME, gras=False))
        yb += 1.5 * fs
    x_gauche = min(xl - g["bg"] * 10, x_min)
    xh = x_gauche - 1.7 * fs
    if g["bg"] > 0:
        hg = g["hg"] * 10
        out.append(_attache(x_gauche - 0.3 * fs, h - hg, xh - 0.3 * fs, h - hg, ep)
                   + _cote_v(xh, h - hg, h, _mm(hg), 0.85 * fs, ep))
        xh -= 1.9 * fs
    out.append(_attache(xl - 0.3 * fs, 0, xh - 0.3 * fs, 0, ep) + _attache(x_gauche - 0.3 * fs, h, xh - 0.3 * fs, h, ep)
               + _cote_v(xh, 0, h, _mm(h), 0.85 * fs, ep))
    x_droite = max(xr + g["bd"] * 10, x_max)
    if g["bd"] > 0:
        hd = g["hd"] * 10
        xd = x_droite + 1.4 * fs
        out.append(_attache(x_droite + 0.3 * fs, h - hd, xd + 0.3 * fs, h - hd, ep)
                   + _cote_v(xd, h - hd, h, _mm(hd), 0.85 * fs, ep, a_droite=True))
        x_droite = xd + 1.2 * fs
    # l'enrobage, au coin haut gauche de l'âme
    out.append(_txt(xl, -0.55 * fs, f"c = {_mm(c)}", 0.78 * fs, ancre="start", couleur=COTE, gras=False))
    # la légende SOUS la coupe : une case par famille (pastille, code, barres, position), puis les étriers ; en
    # colonnes, pour que le dessin garde sa taille au PC comme au téléphone
    x_g0 = xh - 1.6 * fs
    lignes = [(f["couleur"], BORDS[f["groupe"]], f["code"], f["libelle"], f["libelle_position"]) for f in res["familles"]]
    if ds:
        lignes.append((None, None, "Étriers", ", ".join(f"Ø{_mm(d)}" for d in ds), "contour compté : l'âme à l'enrobage"))
    larg_dessin = x_droite + 1.2 * fs - x_g0
    col = 11.6 * fs
    ncol = max(1, min(len(lignes), int(larg_dessin // col), 3))
    if len(lignes) >= 3:
        ncol = max(ncol, 2)
    y0 = yb + 2.6 * fs
    leg = []
    for k, (couleur, bord, code, lib, pos) in enumerate(lignes):
        xg = x_g0 + 0.4 * fs + (k % ncol) * col
        yg = y0 + (k // ncol) * 2.3 * fs
        if couleur is None:
            leg.append(_ligne(xg, yg - 0.33 * fs, xg + 0.84 * fs, yg - 0.33 * fs, ETRIER, 0.3 * fs))
        else:
            leg.append(f'<circle cx="{_f(xg + 0.42 * fs)}" cy="{_f(yg - 0.33 * fs)}" r="{_f(0.42 * fs)}" fill="{couleur}" stroke="{bord}" '
                       f'stroke-width="{_f(0.09 * fs)}"/>')
        leg.append(_txt(xg + 1.3 * fs, yg, escape(code), 0.92 * fs, ancre="start"))
        leg.append(_txt(xg + (5.2 if couleur is None else 4.3) * fs, yg, escape(lib), 0.92 * fs, ancre="start", gras=False))
        leg.append(_txt(xg + 1.3 * fs, yg + 0.95 * fs, escape(pos), 0.7 * fs, ancre="start", couleur=COTE, gras=False))
    nlig = -(-len(lignes) // ncol) if lignes else 0
    if not lignes:
        leg.append(_txt(x_g0 + 0.4 * fs, y0, "Aucune armature : la section du béton seule", 0.85 * fs, ancre="start", couleur=COTE, gras=False))
        nlig = 1
    out += leg
    larg = max(larg_dessin, ncol * col + 0.4 * fs)
    vb = [x_g0, -1.8 * fs, larg, y0 + nlig * 2.3 * fs + 1.8 * fs - 0.6 * fs]
    return _svg(ident, vb, "".join(out), "Coupe de la poutre : béton, barres et étriers")


# ============================================================ l'élévation
_AMPLIS = (1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)
# 6 (et non 10) : une poutre courante de 6 m × 60 cm dessinée à l'échelle n'était qu'une bande de 65 px au PC, ses
# étriers à 11 px les uns des autres ; dessinée deux fois plus haute, barres et étriers se lisent (relecture du 07/11/2026)
ELANCEMENT = 6


def amplification(L_m: float, h_cm: float) -> float:
    """Le facteur des hauteurs de l'élévation : 1 tant que la poutre n'est pas plus de 6 fois plus longue que haute ;
    au-delà, la plus petite valeur ronde qui ramène l'élancement dessiné à 6 (écrit sur le dessin et dans sa légende)."""
    r = L_m * 1000 / (h_cm * 10)
    if r <= ELANCEMENT:
        return 1.0
    return next((k for k in _AMPLIS if r / k <= ELANCEMENT), _AMPLIS[-1])


def _zones_rangs(etriers: List[Dict]) -> List[int]:
    """Le rang de chaque zone dans la chaîne de cotes : deux zones qui se recouvrent vont sur deux rangs."""
    rangs, fins = [], []
    for e in etriers:
        r = 0
        while r < len(fins) and fins[r] > e["x1"] + 1e-9:
            r += 1
        if r == len(fins):
            fins.append(0.0)
        fins[r] = e["x2"]
        rangs.append(r)
    return rangs


def elevation(ident: str, res: Dict) -> str:
    """La poutre en élévation, puis les barres développées : longueurs à l'échelle (mm)."""
    g = res["geometrie"]
    Lmm = g["L"] * 1000
    k = amplification(g["L"], g["h"])
    H = g["h"] * 10 * k
    fs = max(Lmm / 38, 6.0)
    ep = fs / 16
    c = g["c"] * 10 * k
    out = [f'<rect x="0" y="0" width="{_f(Lmm)}" height="{_f(H)}" fill="{BETON}" stroke="{ENCRE}" stroke-width="{_f(1.8 * ep)}"/>']
    # les étriers de chaque zone, au nombre que le module compte, à partir du début de la zone
    for e in res["etriers"]:
        x0, s = g["L"] * e["x1"] / 100 * 1000, e["s"] * 10
        for i in range(e["n"]):
            x = x0 + i * s
            out.append(_ligne(x, c, x, H - c, ETRIER, max(e["d"] * 0.9, 1.1 * ep)))
    # les barres longitudinales : une ligne par hauteur de barres, de « De » à « À »
    for f in res["familles"]:
        if f["L_active"] <= 0:
            continue
        x1, x2 = Lmm * f["de"] / 100, Lmm * f["a"] / 100
        for y in sorted({round(b[1], 6) for b in f["barres"]}):
            Y = H - y * 10 * k
            out.append(f'<line x1="{_f(x1)}" y1="{_f(Y)}" x2="{_f(x2)}" y2="{_f(Y)}" stroke="{BORDS[f["groupe"]]}" '
                       f'stroke-width="{_f(max(f["d"] * k, 2.2 * ep) + 1.2 * ep)}" stroke-linecap="butt"/>'
                       f'<line x1="{_f(x1)}" y1="{_f(Y)}" x2="{_f(x2)}" y2="{_f(Y)}" stroke="{f["couleur"]}" '
                       f'stroke-width="{_f(max(f["d"] * k, 2.2 * ep))}"/>')
    if k > 1:
        out.append(_txt(Lmm, -0.6 * fs, f"hauteurs × {_mm(k)} · longueurs à l'échelle", 0.8 * fs, ancre="end", couleur=COTE, gras=False))
    out.append(_cote_v(-1.2 * fs, 0, H, _mm(g["h"] * 10), 0.85 * fs, ep))
    # la chaîne des zones d'étriers (sous la poutre), puis la portée
    y = H + 2.1 * fs
    rangs = _zones_rangs(res["etriers"])
    for e, r in zip(res["etriers"], rangs):
        x1, x2 = g["L"] * e["x1"] / 100 * 1000, g["L"] * e["x2"] / 100 * 1000
        yy = y + r * 2.0 * fs
        texte = f"{escape(e['zone'])} · {e['n']} × Ø{_mm(e['d'])} · e = {_mm(e['s'] * 10)}"
        if len(texte) * 0.5 * 0.8 * fs > (x2 - x1):
            texte = escape(e["zone"])
        out.append(_attache(x1, H, x1, yy + 0.3 * fs, ep) + _attache(x2, H, x2, yy + 0.3 * fs, ep) + _cote_h(x1, x2, yy, texte, 0.8 * fs, ep))
    if rangs:
        y += (max(rangs) + 1) * 2.0 * fs
    out.append(_attache(0, H, 0, y + 0.3 * fs, ep) + _attache(Lmm, H, Lmm, y + 0.3 * fs, ep) + _cote_h(0, Lmm, y, f"L = {_mm(Lmm)}", 0.9 * fs, ep))
    # les barres développées, famille par famille : longueur droite et retours
    y += 2.6 * fs
    if res["familles"]:
        out.append(_txt(-9.6 * fs, y, "BARRES DÉVELOPPÉES", 0.72 * fs, ancre="start", couleur=COTE))
        y += 1.5 * fs
    for f in res["familles"]:
        x1, x2 = Lmm * f["de"] / 100, Lmm * f["a"] / 100
        out.append(f'<circle cx="{_f(-9.2 * fs)}" cy="{_f(y - 0.3 * fs)}" r="{_f(0.36 * fs)}" fill="{f["couleur"]}" stroke="{BORDS[f["groupe"]]}" '
                   f'stroke-width="{_f(0.08 * fs)}"/>')
        out.append(_txt(-8.5 * fs, y, escape(f["code"]), 0.85 * fs, ancre="start") + _txt(-5.6 * fs, y, escape(f["libelle"]), 0.85 * fs, ancre="start", gras=False))
        if f["L_active"] > 0:
            yl = y - 0.3 * fs
            out.append(f'<line x1="{_f(x1)}" y1="{_f(yl)}" x2="{_f(x2)}" y2="{_f(yl)}" stroke="{BORDS[f["groupe"]]}" stroke-width="{_f(0.34 * fs)}"/>'
                       f'<line x1="{_f(x1)}" y1="{_f(yl)}" x2="{_f(x2)}" y2="{_f(yl)}" stroke="{f["couleur"]}" stroke-width="{_f(0.22 * fs)}"/>')
            # les retours : un crochet par retour, dessiné à chaque bout (le module ne dit pas où ils sont)
            haut = f["groupe"] == "haut"
            hr = min(f["h_retour"] * 10 * k, 1.1 * fs) if f["retours"] else 0
            for i in range(f["retours"]):
                xr = x1 if i == 0 else x2
                out.append(_ligne(xr, yl, xr, yl + (hr if haut else -hr), BORDS[f["groupe"]], 0.3 * fs, ' stroke-linecap="round"'))
            lib = f"{f['n_eff']} × {_mm(f['L_active'] * 1000)}"
            if f["retours"]:
                lib += f" + {f['retours'] * f['n_eff']} retour{'s' if f['retours'] * f['n_eff'] > 1 else ''} de {_mm(f['h_retour'] * 10)}"
            xm = (x1 + x2) / 2
            out.append(_txt(xm, yl - 0.5 * fs, lib, 0.72 * fs, couleur=ENCRE, gras=False))
        else:
            out.append(_txt(0, y, "longueur nulle : « À » n'est pas au-delà de « De »", 0.75 * fs, ancre="start", couleur=COTE, gras=False))
        y += 2.0 * fs
    vb = [-10.2 * fs, -1.9 * fs, Lmm + 11.2 * fs, y + 0.8 * fs]
    return _svg(ident, vb, "".join(out), "Élévation de la poutre : barres et zones d'étriers")


# ============================================================ le tableau récapitulatif
def tableau(lignes: List[Dict], total: Dict) -> str:
    """Le tableau récapitulatif des armatures (celui de l'écran Streamlit), en HTML : une ligne par famille, puis une
    par zone d'étriers, puis le total. ``lignes`` : couleur, code, position, n, d, L_active, L_tot, m_lin, masse —
    déjà écrits (texte) sauf la couleur ; les textes venus de la note sont échappés ici."""
    if not lignes:
        return ('<div class="tx-vue"><div class="tx-vide"><b>Aucune armature</b>Ajouter un lit, un renfort, des barres latérales '
                'ou une zone d\'étriers : le tableau récapitulatif apparaît ici.</div></div>')
    rows = "".join(
        f'<tr><td class="co"><i style="background:{li["couleur"]};box-shadow:inset 0 0 0 1.5px {li["bord"]}"></i></td>'
        f'<th>{escape(li["code"])}</th><td class="po">{escape(li["position"])}</td><td>{li["n"]}</td><td>{li["d"]}</td>'
        f'<td>{li["L_active"]}</td><td>{li["L_tot"]}</td><td>{li["m_lin"]}</td><td class="kg">{li["masse"]}</td></tr>' for li in lignes)
    pied = (f'<tr><td></td><th>Total</th><td class="po">{escape(total["libelle"])}</td><td></td><td></td><td></td>'
            f'<td>{total["L_tot"]}</td><td></td><td class="kg">{total["masse"]}</td></tr>')
    return ('<div class="tx-vue"><div class="tx-defile"><table class="tx-tab"><thead><tr><th></th><th>Code</th><th class="po">Position</th>'
            '<th>n barres</th><th>Ø<small>mm</small></th><th>L active<small>m</small></th><th>L tot<small>m</small></th>'
            f'<th>kg/m</th><th>kg tot</th></tr></thead><tbody>{rows}</tbody><tfoot>{pied}</tfoot></table></div>'
            f'<div class="tx-pied">{total["pied"]}</div></div>')

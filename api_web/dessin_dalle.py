# -*- coding: utf-8 -*-
"""Les dessins d'une section de DALLE en béton armé, à l'échelle (mm), à partir de ce que le moteur ``beton/dalle`` a
calculé : l'épaisseur h, la largeur de la bande b, et pour chaque face (inf. / sup.) et chaque direction (X / Y) les
couches d'armatures — treillis, barres Ø/e, « n barres » — avec l'axe de chaque couche tel que le moteur le retient
(automatique ou saisi). Rien n'est recalculé ici.

Deux vues par section :
- la COUPE : deux coupes de la bande à la même échelle, perpendiculaires à la direction principale puis à la
  secondaire ; les barres coupées sont des points (Ø réel), les barres parallèles au plan de coupe des traits (épaisseur
  = Ø) ; cotes h, largeur vue, axes des couches de base et hauteur utile d de la face TENDUE de la direction en points
  (la supérieure quand son moment l'emporte, comme la coupe de la poutre) ; la légende donne l'axe de chaque couche ;
- le PLAN des nappes : la nappe inférieure et la nappe supérieure vues de dessus, barres X horizontales et barres Y
  verticales à leur espacement réel, cotes de la vue et des espacements, repère des directions (la principale en gras).

Conventions de dessin (pas des données de calcul) : la bande est une tranche d'une dalle continue, bordée de lignes de
rupture ; au-delà de 1,20 m, seule sa partie centrale est dessinée (la cote le dit). Les barres d'une couche de base sont
centrées sur la bande, celles d'un renfort intercalées d'une demi-maille (la convention de la note PDF du bureau) ; les
« n barres » sont réparties également dans la bande. Couleurs : la base en acier foncé, chaque renfort dans sa couleur.

Les NIVEAUX (docs/site/DALLES_FUSION.md, § 4) : chaque couche — base ou renfort, direction coupée ou parallèle — est
dessinée à l'axe que le moteur a UTILISÉ, ``couches[j]["e"]`` du résultat (imposé, automatique du moteur, ou automatique
du site écrit dans la copie envoyée au moteur : api_web/dalle_positions.py). Le dessin ne recalcule aucun niveau : deux
directions au même niveau sur la coupe, c'est le calcul qui les y a posées (un axe saisi) ; à des niveaux distincts —
la direction secondaire, base et renforts, posée sur la principale, toujours depuis le 09/11/2026 —, c'est qu'il les y a
comptées. La cote de d est h − y_G du calcul, pour la face tendue de la direction coupée. Une couche que le site a posée
sur la nappe principale (``empilee``, posé par ``dalle_positions.annoter`` : base ou renfort secondaire) le dit dans la
légende, avec ce qui la porte (``sur`` : la base principale, ou un renfort principal plus haut qu'elle — S5).

Le style est celui des dessins du site (api_web/dessin_poutre.py) : traits noirs, béton gris clair, cotes fines,
sollicitations en rouge. Les textes venus de l'état (noms) sont échappés ici.

Les PIÈCES (l'établi compact, js/compact.js) : avec le rang ``i`` de la section, chaque couche est un groupe
``<g class="pc couche" data-sel="s<i>.<face>-<j>">`` (ses points dans une coupe, ses traits dans l'autre et dans le
plan), le béton ``s<i>.beton`` et les sollicitations ``s<i>.efforts`` — les mêmes clés que les lignes des données
(``data-cible``, api_web/plaque_compact.py) : un clic sur l'une montre l'autre. Sans ``i``, le dessin est le même qu'avant,
à l'octet près.
"""
from html import escape
from typing import Dict, List, Optional, Tuple

ENCRE, BETON, COTE, EFFORT = "#15181F", "#E6EAEF", "#7F8C9A", "#9C3341"
PAPIER_PLAN = "#F1F3F6"
COULEURS = ("#1F2A3D", "#1F5FA8", "#A9670F", "#23704A")     # la base, puis les renforts 1, 2, 3 (barres coupées, plan)
TEINTES = ("#A3AEC0", "#93B9E6", "#E2BE84", "#8FC7A6")      # les mêmes, en clair : barres parallèles au plan de coupe
POLICE = "system-ui,Arial,sans-serif"
LARGEUR_VUE_MAX = 1200.0                                     # mm : au-delà, la bande est dessinée en partie
TYPES_COURTS = {"treillis": "treillis", "barres": "barres", "n_barres": "n barres"}


def _f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _mm(x: float) -> str:
    """Une cote en mm, à la française : « 1 000 », « 35 », « 32,5 »."""
    s = f"{x:,.1f}".replace(",", " ").replace(".", ",")
    return s[:-2] if s.endswith(",0") else s


def _txt(x, y, t, taille, ancre="middle", couleur=ENCRE, gras=True, extra="") -> str:
    poids = ' font-weight="600"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{t}</text>')


def _ligne(x1, y1, x2, y2, couleur=COTE, ep=1.0, extra="") -> str:
    return (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" '
            f'stroke-width="{_f(ep)}"{extra}/>')


def _cote_h(x1, x2, y, texte, u) -> str:
    """Une cote horizontale : ligne, deux traits obliques, le nombre au-dessus."""
    t = 0.28 * u
    return (_ligne(x1, y, x2, y, COTE, 0.05 * u) + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, 0.06 * u)
            + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, 0.06 * u) + _txt((x1 + x2) / 2, y - 0.35 * u, texte, 0.8 * u))


def _cote_v(x, y1, y2, texte, u, droite=False) -> str:
    """Une cote verticale : le nombre tourné le long de la ligne."""
    t = 0.28 * u
    xt = x + (0.95 * u if droite else -0.4 * u)
    ym = (y1 + y2) / 2
    return (_ligne(x, y1, x, y2, COTE, 0.05 * u) + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, 0.06 * u)
            + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, 0.06 * u)
            + _txt(xt, ym, texte, 0.8 * u, extra=f' transform="rotate(-90 {_f(xt)} {_f(ym)})"'))


def _attache(x1, y1, x2, y2, u) -> str:
    return _ligne(x1, y1, x2, y2, COTE, 0.04 * u, f' stroke-dasharray="{_f(0.2 * u)} {_f(0.15 * u)}"')


def _rupture_v(x, y1, y2, u) -> str:
    """Une ligne de rupture verticale (la bande continue au-delà) : un trait fin brisé en son milieu."""
    m, z = (y1 + y2) / 2, min(0.7 * u, 0.3 * (y2 - y1))
    pts = [(x, y1 - 0.4 * u), (x, m - z), (x + 0.45 * z, m - 0.35 * z), (x - 0.45 * z, m + 0.35 * z), (x, m + z), (x, y2 + 0.4 * u)]
    return (f'<polyline points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in pts)}" fill="none" stroke="{ENCRE}" '
            f'stroke-width="{_f(0.05 * u)}"/>')


def _rupture_h(y, x1, x2, u) -> str:
    m, z = (x1 + x2) / 2, min(0.7 * u, 0.3 * (x2 - x1))
    pts = [(x1 - 0.4 * u, y), (m - z, y), (m - 0.35 * z, y - 0.45 * z), (m + 0.35 * z, y + 0.45 * z), (m + z, y), (x2 + 0.4 * u, y)]
    return (f'<polyline points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in pts)}" fill="none" stroke="{ENCRE}" '
            f'stroke-width="{_f(0.05 * u)}"/>')


def _fleche(x1, y1, x2, y2, u, couleur=ENCRE, ep=None) -> str:
    """Une flèche du repère des directions."""
    import math
    ep = ep or 0.07 * u
    a = math.atan2(y2 - y1, x2 - x1)
    L, w = 0.45 * u, 0.2 * u
    bx, by = x2 - L * math.cos(a), y2 - L * math.sin(a)
    p1 = (bx + w * math.sin(a), by - w * math.cos(a))
    p2 = (bx - w * math.sin(a), by + w * math.cos(a))
    return (_ligne(x1, y1, bx, by, couleur, ep)
            + f'<polygon points="{_f(x2)},{_f(y2)} {_f(p1[0])},{_f(p1[1])} {_f(p2[0])},{_f(p2[1])}" fill="{couleur}"/>')


# ------------------------------------------------------------------ la position des barres (convention de dessin)
def positions(c: Dict, b_mm: float, j: int) -> List[float]:
    """Les abscisses (mm, depuis le bord de la bande) des barres d'une couche : « n barres » réparties également dans la
    bande (une demi-maille aux bords) ; treillis et barres à leur espacement, autant de barres que la bande en porte
    (b / e arrondi), centrées sur la bande (base) ou intercalées d'une demi-maille (renfort)."""
    if c["type"] == "n_barres":
        pas = b_mm / c["n"]
        return [pas * (k + 0.5) for k in range(c["n"])]
    pas = float(c["esp"])
    n = max(1, round(b_mm / pas))
    debut = (b_mm - (n - 1) * pas) / 2
    xs = [debut + k * pas for k in range(n)]
    if j > 0:
        xs = [x + pas / 2 for x in xs if x + pas / 2 < b_mm]
    return xs


def fenetre(b_mm: float) -> Tuple[float, float]:
    """La partie dessinée de la bande : (début, largeur), centrée, d'au plus LARGEUR_VUE_MAX."""
    W = min(b_mm, LARGEUR_VUE_MAX)
    return (b_mm - W) / 2, W


def _couleur(j: int) -> str:
    return COULEURS[j % len(COULEURS)]


def piece(i: Optional[int], cle: str, contenu: str, classe: str = "couche") -> str:
    """Une pièce cliquable du dessin (l'établi compact) : ``contenu`` dans un groupe ``data-sel="s<i>.<cle>"`` ; sans
    rang de section, le contenu seul."""
    if i is None or not contenu:
        return contenu
    return f'<g class="pc {classe}" data-sel="s{i}.{escape(cle)}">{contenu}</g>'


def _teinte(j: int) -> str:
    return TEINTES[j % len(TEINTES)]


def _nom_couche(j: int) -> str:
    return "base" if j == 0 else f"renfort {j}"


def _texte_vue(b_mm: float, W: float) -> str:
    return f"b = {_mm(b_mm)}" if W >= b_mm else f"{_mm(W)} vus sur b = {_mm(b_mm)}"


# ------------------------------------------------------------------ la coupe
def _coupe(s: Dict, dk: str, principale: str, b_mm: float, h_mm: float, u: float, y0: float,
           i: Optional[int] = None) -> Tuple[str, float]:
    """Une coupe de la bande perpendiculaire à la direction ``dk`` (ses barres en points, celles de l'autre direction en
    traits), posée à l'ordonnée ``y0`` : (svg, hauteur occupée)."""
    autre = "x" if dk == "y" else "y"
    x0, W = fenetre(b_mm)
    g = []
    # le titre de la coupe
    titre = f"Coupe ⊥ {dk.upper()} · barres {dk.upper()} en coupe" + (" (principale)" if dk == principale else "")
    g.append(_txt(0, y0 + 0.9 * u, escape(titre), 0.95 * u, ancre="start"))
    yc = y0 + 2.4 * u                                            # le dessus du béton
    # les niveaux : axe de chaque couche, depuis le dessus du béton
    couches = []
    for cote in ("inf", "sup"):
        for d_ in (dk, autre):
            for j, c in enumerate(s["faces"][f"{cote}_{d_}"]["couches"]):
                y = (h_mm - c["e"] * 10.0) if cote == "inf" else c["e"] * 10.0
                couches.append((cote, d_, j, c, y))
    y_bas = max([h_mm] + [y + c["d"] / 2 for *_, c, y in couches])
    y_haut = min([0.0] + [y - c["d"] / 2 for *_, c, y in couches])
    # le béton : un rectangle gris, dessus et dessous en trait fort, bordé de lignes de rupture (une tranche de dalle)
    g.append(piece(i, "beton", f'<rect x="0" y="{_f(yc)}" width="{_f(W)}" height="{_f(h_mm)}" fill="{BETON}"/>'
                   + _ligne(0, yc, W, yc, ENCRE, 0.09 * u) + _ligne(0, yc + h_mm, W, yc + h_mm, ENCRE, 0.09 * u), "beton"))
    g.append(_rupture_v(0, yc, yc + h_mm, u) + _rupture_v(W, yc, yc + h_mm, u))
    # les barres de l'autre direction : des traits (épaisseur = Ø), puis celles de la direction coupée : des points
    for cote, d_, j, c, y in couches:
        if d_ == autre:
            g.append(piece(i, f"{cote}_{d_}-{j}", _ligne(0, yc + y, W, yc + y, _teinte(j), max(c["d"], 0.04 * u))))
    for cote, d_, j, c, y in couches:
        if d_ == dk:
            r = c["d"] / 2
            points = ""
            for x in positions(c, b_mm, j):
                xv = x - x0
                if r <= xv <= W - r:
                    points += (f'<circle cx="{_f(xv)}" cy="{_f(yc + y)}" r="{_f(r)}" fill="{_couleur(j)}" stroke="#FFFFFF" '
                               f'stroke-width="{_f(max(0.3 * r, 0.03 * u))}" paint-order="stroke"/>')
            g.append(piece(i, f"{cote}_{d_}-{j}", points))
    # les cotes : h à gauche, la largeur vue dessous, à droite l'axe des bases et la hauteur utile de la direction coupée
    g.append(_attache(-0.3 * u, yc, -1.6 * u, yc, u) + _attache(-0.3 * u, yc + h_mm, -1.6 * u, yc + h_mm, u)
             + _cote_v(-1.3 * u, yc, yc + h_mm, f"h = {_mm(h_mm)}", u))
    y_cote = yc + y_bas + 1.4 * u
    g.append(_cote_h(0, W, y_cote, _texte_vue(b_mm, W), u))
    xd = W + 1.3 * u
    for cote in ("inf", "sup"):
        c = s["faces"][f"{cote}_{dk}"]["couches"][0]
        y = (h_mm - c["e"] * 10.0) if cote == "inf" else c["e"] * 10.0
        y_face = h_mm if cote == "inf" else 0.0
        g.append(_attache(W + 0.2 * u, yc + y, xd + 0.4 * u, yc + y, u)
                 + _cote_v(xd, yc + min(y, y_face), yc + max(y, y_face), _mm(c["e"] * 10.0), u, droite=True))
    # la hauteur utile de la face TENDUE de la direction coupée (sup. si son moment l'emporte, comme la poutre) : du
    # parement comprimé au centre de gravité des couches tendues
    tendue = "sup" if s[f"M{dk}_sup"] > s[f"M{dk}_inf"] else "inf"
    yG = s["faces"][f"{tendue}_{dk}"]["yG"] * 10.0
    xd2 = W + 3.3 * u
    if tendue == "inf":
        y1, y2 = yc, yc + h_mm - yG          # du dessus du béton au centre de gravité des couches inférieures
    else:
        y1, y2 = yc + yG, yc + h_mm          # du centre de gravité des couches supérieures au dessous du béton
    g.append(_attache(W + 0.2 * u, y1, xd2 + 0.4 * u, y1, u) + _attache(W + 0.2 * u, y2, xd2 + 0.4 * u, y2, u)
             + _cote_v(xd2, y1, y2, f"d = {_mm(h_mm - yG)}", u, droite=True))
    # la légende : les couches, inférieures à gauche, supérieures à droite (points : la direction coupée)
    y_leg = y_cote + 1.6 * u
    n_max = 0
    deux_colonnes = W + 4.6 * u >= 32 * u          # sinon (bande étroite) les supérieures sous les inférieures
    for k, cote in enumerate(("inf", "sup")):
        lignes = [(d_, j, c) for d_ in (dk, autre) for j, c in enumerate(s["faces"][f"{cote}_{d_}"]["couches"])]
        xl = k * (W + 4.6 * u) / 2 if deux_colonnes else 0.0
        if not deux_colonnes and k == 1:
            y_leg += (n_max + 1.4) * 1.3 * u
            n_max = 0
        g.append(_txt(xl, y_leg, "Inférieures" if cote == "inf" else "Supérieures", 0.78 * u, ancre="start", couleur=COTE))
        for n, (d_, j, c) in enumerate(lignes):
            yy = y_leg + (n + 1) * 1.3 * u
            if d_ == dk:
                g.append(f'<circle cx="{_f(xl + 0.35 * u)}" cy="{_f(yy - 0.3 * u)}" r="{_f(0.26 * u)}" fill="{_couleur(j)}"/>')
            else:
                g.append(_ligne(xl, yy - 0.3 * u, xl + 0.7 * u, yy - 0.3 * u, _teinte(j), 0.24 * u))
            # l'axe réel de la couche (celui du calcul) ; posée par le site sur la nappe principale : dit sur QUOI elle
            # repose (``sur_court`` de dalle_positions.annoter : « la base Y », ou le renfort plus haut qu'elle, S5 —
            # court : la ligne doit tenir dans sa colonne ; « posée sur la base Y » débordait sur la colonne des
            # supérieures, vu sur la capture du 09/11/2026, et « sur le renfort Y » aussi, capture S5)
            g.append(_txt(xl + 1.0 * u, yy, f"{d_.upper()} · {escape(c['libelle'])} · axe à {_mm(c['e'] * 10.0)}"
                          + (f" · sur {c.get('sur_court') or 'la base ' + principale.upper()}" if c.get("empilee") else ""),
                          0.78 * u, ancre="start", gras=False))
        n_max = max(n_max, len(lignes))
    hauteur = (y_leg + (n_max + 0.6) * 1.3 * u) - y0
    return "".join(g), hauteur


def _efforts(s: Dict) -> str:
    parts = []
    for dk in ("x", "y"):
        for cote in ("inf", "sup"):
            m = s[f"M{dk}_{cote}"]
            if m:
                parts.append(f"M<tspan font-size='70%' dy='4'>{dk},{cote}</tspan><tspan dy='-4'> = {_mm(m)}</tspan>")
    txt = " · ".join(parts)
    txt = (txt + " kN·m") if parts else "aucun moment"
    if s["V"]:
        txt += f" · V = {_mm(s['V'])} kN"
    return txt


def svg_coupe(ident: str, s: Dict, dalle: Dict, i: Optional[int] = None) -> str:
    """Les deux coupes d'une section (la direction principale d'abord), à la même échelle ; ``i`` : le rang de la
    section, pour ses pièces cliquables."""
    b_mm, h_mm = dalle["b"] * 10.0, dalle["h"] * 10.0
    _, W = fenetre(b_mm)
    u = max(W, 420.0) / 34.0
    p = s["principale"]
    g, y = [], 0.0
    if i is None:
        g.append(_txt(max(W, 27.8 * u) / 2, 0.2 * u, _efforts(s), 0.9 * u, couleur=EFFORT))
    # (l'établi compact, rang i : les efforts sont écrits UNE fois, dans la bande de l'en-tête de la vue, sans rouge)
    y = 0.9 * u if i is None else 0.0
    for dk in (p, "x" if p == "y" else "y"):
        morceau, hh = _coupe(s, dk, p, b_mm, h_mm, u, y, i)
        g.append(morceau)
        y += hh + 0.8 * u
    largeur = max(W + 8.4 * u, 31.0 * u)            # une bande étroite garde la place des titres et de la légende
    haut = -1.2 * u if i is None else -0.3 * u
    vb = f"{_f(-3.2 * u)} {_f(haut)} {_f(largeur)} {_f(y + 0.6 * u - haut - 1.2 * u)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Coupes de la dalle, section {escape(s["nom"])}">{"".join(g)}</svg>')


# ------------------------------------------------------------------ le plan des nappes
def _plan(s: Dict, cote: str, principale: str, b_mm: float, u: float, dx: float, i: Optional[int] = None) -> Tuple[str, float]:
    """La nappe ``cote`` vue de dessus, dans un carré de la largeur vue : barres X horizontales, Y verticales."""
    x0, W = fenetre(b_mm)
    g = [f'<g transform="translate({_f(dx)} 0)">']
    g.append(_txt(W / 2, -2.6 * u, "Nappe inférieure" if cote == "inf" else "Nappe supérieure", 1.0 * u))
    # la limite de la vue (la nappe continue au-delà) : un trait mixte fin
    g.append(f'<rect x="0" y="0" width="{_f(W)}" height="{_f(W)}" fill="{PAPIER_PLAN}" stroke="{COTE}" stroke-width="{_f(0.05 * u)}" '
             f'stroke-dasharray="{_f(0.6 * u)} {_f(0.18 * u)} {_f(0.1 * u)} {_f(0.18 * u)}"/>')
    # l'ordre de pose : vue de dessus, la couche la plus haute recouvre les autres
    traits = []
    for dk in ("x", "y"):
        for j, c in enumerate(s["faces"][f"{cote}_{dk}"]["couches"]):
            haut = c["e"] if cote == "inf" else -c["e"]
            traits.append((haut, dk, j, c))
    traits.sort(key=lambda t: t[0])
    for _, dk, j, c in traits:
        ep = max(c["d"], 0.04 * u)
        marge = c["d"] / 2 + 0.1 * u          # une barre sur la limite de la vue ne se dessine pas
        barres = ""
        for p in positions(c, b_mm, j):
            v = p - x0
            if not marge < v < W - marge:
                continue
            if dk == "x":     # une barre X court suivant X : un trait horizontal, à l'ordonnée de sa position en Y
                barres += _ligne(0.1 * u, W - v, W - 0.1 * u, W - v, _couleur(j), ep)
            else:
                barres += _ligne(v, 0.1 * u, v, W - 0.1 * u, _couleur(j), ep)
        g.append(piece(i, f"{cote}_{dk}-{j}", barres))
    # les cotes : la vue en haut, l'espacement de la base X à droite, celui de la base Y dessous
    g.append(_cote_h(0, W, -1.0 * u, _texte_vue(b_mm, W), u))
    for dk in ("x", "y"):
        c = s["faces"][f"{cote}_{dk}"]["couches"][0]
        ps = [p - x0 for p in positions(c, b_mm, 0) if c["d"] / 2 + 0.1 * u < p - x0 < W - c["d"] / 2 - 0.1 * u]
        if len(ps) < 2:
            continue
        k = min(range(len(ps) - 1), key=lambda i: abs((ps[i] + ps[i + 1]) / 2 - W / 2))
        a, b = ps[k], ps[k + 1]
        if dk == "x":
            g.append(_attache(W + 0.15 * u, W - a, W + 1.1 * u, W - a, u) + _attache(W + 0.15 * u, W - b, W + 1.1 * u, W - b, u)
                     + _cote_v(W + 0.8 * u, W - b, W - a, _mm(b - a), u, droite=True))
        else:
            g.append(_attache(a, W + 0.15 * u, a, W + 1.6 * u, u) + _attache(b, W + 0.15 * u, b, W + 1.6 * u, u)
                     + _cote_h(a, b, W + 1.3 * u, _mm(b - a), u))
    # la légende : la direction principale d'abord
    y_leg = W + 3.0 * u
    n = 0
    for dk in (principale, "x" if principale == "y" else "y"):
        for j, c in enumerate(s["faces"][f"{cote}_{dk}"]["couches"]):
            yy = y_leg + n * 1.35 * u
            g.append(_ligne(0, yy - 0.3 * u, 0.8 * u, yy - 0.3 * u, _couleur(j), 0.22 * u))
            sens = "horizontales" if dk == "x" else "verticales"
            g.append(_txt(1.1 * u, yy, f"{dk.upper()} {_nom_couche(j)} · {escape(c['libelle'])} · {sens}", 0.8 * u,
                          ancre="start", gras=dk == principale and j == 0))
            n += 1
    g.append("</g>")
    return "".join(g), y_leg + n * 1.35 * u


def _repere(x, y, u, principale: str) -> str:
    """Le repère des directions : X vers la droite, Y vers le haut ; la principale en trait fort, nommée dessous."""
    L = 2.2 * u
    out = _fleche(x, y, x + L, y, u, ep=(0.12 if principale == "x" else 0.06) * u)
    out += _fleche(x, y, x, y - L, u, ep=(0.12 if principale == "y" else 0.06) * u)
    out += _txt(x + L + 0.25 * u, y + 0.3 * u, "X", 0.9 * u, ancre="start", gras=principale == "x")
    out += _txt(x, y - L - 0.3 * u, "Y", 0.9 * u, gras=principale == "y")
    out += _txt(x + L + 1.5 * u, y + 0.3 * u, f"direction principale : {principale.upper()}", 0.78 * u, ancre="start",
                couleur=COTE, gras=False)
    return out


def svg_plan(ident: str, s: Dict, dalle: Dict, i: Optional[int] = None) -> str:
    """Les nappes inférieure et supérieure d'une section, côte à côte, à la même échelle, avec le repère des directions ;
    ``i`` : le rang de la section, pour ses pièces cliquables."""
    b_mm = dalle["b"] * 10.0
    _, W = fenetre(b_mm)
    u = max(W, 420.0) / 16.0
    p = s["principale"]
    gap = 4.2 * u
    g1, h1 = _plan(s, "inf", p, b_mm, u, 0.0, i)
    g2, h2 = _plan(s, "sup", p, b_mm, u, W + gap, i)
    y_rep = h1 + 2.8 * u                            # le repère, sous la légende de la nappe inférieure
    rep = _repere(0.3 * u, y_rep, u, p)
    haut = max(y_rep + 0.8 * u, h2)
    vb = f"{_f(-0.8 * u)} {_f(-3.8 * u)} {_f(2 * W + gap + 3.4 * u)} {_f(haut + 4.0 * u)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Plan des nappes, section {escape(s["nom"])}">{g1}{g2}{rep}</svg>')


# ------------------------------------------------------------------ toutes les sections (l'établi compact, vue « Toutes »)
def svg_toutes(ident: str, sections: List[Dict], dalle: Dict) -> str:
    """Toutes les sections l'une sous l'autre, à la même échelle : pour chacune, son nom et ses sollicitations, puis sa
    coupe perpendiculaire à sa direction principale (les barres principales en points), avec ses pièces cliquables."""
    b_mm, h_mm = dalle["b"] * 10.0, dalle["h"] * 10.0
    _, W = fenetre(b_mm)
    u = max(W, 420.0) / 34.0
    large = max(W, 27.8 * u)
    g, y = [], 0.0
    for i, s in enumerate(sections):
        g.append(_txt(0, y + 0.3 * u, escape(s["titre"]), 1.05 * u, ancre="start"))
        g.append(piece(i, "efforts", _txt(large, y + 0.3 * u, _efforts(s), 0.85 * u, ancre="end", couleur=COTE, gras=False), "efforts"))
        y += 1.0 * u
        morceau, hh = _coupe(s, s["principale"], s["principale"], b_mm, h_mm, u, y, i)
        g.append(morceau)
        y += hh + 1.8 * u
    largeur = max(W + 8.4 * u, 31.0 * u)
    vb = f"{_f(-3.2 * u)} {_f(-1.4 * u)} {_f(largeur)} {_f(y)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Toutes les sections de la dalle">{"".join(g)}</svg>')

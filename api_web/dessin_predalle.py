# -*- coding: utf-8 -*-
"""Les dessins d'une section de PRÉDALLE, à l'échelle (mm), à partir de ce que le moteur ``beton/predalle`` a calculé :
l'épaisseur totale h, la peau préfabriquée h_p, la largeur de la bande b, et pour chaque face (inf. / sup.) et chaque
direction (X / Y) les couches d'armatures — treillis, barres Ø/e, « n barres » — avec l'axe de chaque couche tel que le
moteur le retient (automatique selon la règle de la prédalle, ou saisi). Rien n'est recalculé ici.

Deux vues par section :
- la COUPE : deux coupes de la bande à la même échelle, perpendiculaires à la direction principale puis à la
  secondaire. La prédalle (préfabriquée) est hachurée, le béton coulé en place est uni ; le plan de reprise entre les
  deux est un trait fort. Les barres coupées sont des points (Ø réel), les barres parallèles au plan de coupe des traits
  (épaisseur = Ø). Cotes : à gauche h et sa décomposition (prédalle + coulé en place) ; à droite l'axe des bases de la
  direction coupée et sa hauteur utile d ;
- le PLAN des nappes : la nappe inférieure et la nappe supérieure vues de dessus, barres X horizontales et barres Y
  verticales à leur espacement réel ; la légende dit où chaque couche est posée (dans ou sur la prédalle).

Conventions de dessin (pas des données de calcul) : la bande est une tranche d'une dalle continue, bordée de lignes de
rupture ; au-delà de 1,20 m, seule sa partie centrale est dessinée (la cote le dit). Les barres d'une couche de base sont
centrées sur la bande, celles d'un renfort intercalées d'une demi-maille ; les « n barres » sont réparties également.
Les raidisseurs (treillis de levage et de rigidité des prédalles) ne sont pas dessinés : l'écran Streamlit ne les
connaît pas. Style des dessins du site (api_web/dessin_poutre.py, api_web/dessin_dalle.py) : traits noirs, béton gris
clair, cotes fines, sollicitations en rouge. Les textes venus de l'état (noms) sont échappés ici.

Les PIÈCES (l'établi compact) : les mêmes que celles des dalles (api_web/dessin_dalle.py, ``piece``) — chaque couche
``s<i>.<face>-<j>``, le béton ``s<i>.beton``, les sollicitations ``s<i>.efforts`` ; sans rang de section, le dessin est
le même qu'avant, à l'octet près.

Les NIVEAUX (docs/site/DALLES_FUSION.md, § 4) : chaque couche est dessinée à l'axe que le moteur a UTILISÉ,
``couches[j]["e"]`` du résultat — la règle de pose de la prédalle pour les couches inférieures, l'axe imposé, ou
l'automatique du site (nappes supérieures empilées — la direction secondaire, base et renforts, posée sur la
principale —, écrit dans la copie envoyée au moteur : api_web/dalle_positions.py).
Rien n'est recalculé ici. La cote de d est h − y_G du calcul, pour la face TENDUE de la direction coupée (la supérieure
quand son moment l'emporte, comme la coupe des dalles). La légende dit la pose du moteur (dans / sur la prédalle), « axe
saisi » pour un axe imposé dans l'état, « sur la base … » (ou « sur renfort … », S5 : ce qui la porte) pour une
couche que le site a empilée — ces deux derniers d'après ``auto``, ``empilee`` et ``sur`` tels que
``dalle_positions.annoter`` les pose ; sans annotation, le résultat du moteur dit « saisi » d'un axe que le site lui a
imposé.
"""
import math
from html import escape
from typing import Dict, List, Optional, Tuple

from . import dalle_nappes as NAP
from .dessin_dalle import entrees_plan, legende_nappes, piece, rappel_jeu

ENCRE, BETON, COTE, EFFORT = "#15181F", "#E6EAEF", "#7F8C9A", "#9C3341"
PEAU, HACHURE = "#D3DAE3", "#8C97A6"                          # la prédalle préfabriquée : un gris plus soutenu, hachuré
PAPIER_PLAN = "#F1F3F6"
COULEURS = ("#1F2A3D", "#1F5FA8", "#A9670F", "#23704A")     # la base, puis les renforts 1, 2, 3 (barres coupées, plan)
TEINTES = ("#A3AEC0", "#93B9E6", "#E2BE84", "#8FC7A6")      # les mêmes, en clair : barres parallèles au plan de coupe
POLICE = "system-ui,Arial,sans-serif"
LARGEUR_VUE_MAX = 1200.0                                     # mm : au-delà, la bande est dessinée en partie
POSES = {"predalle": "dans la prédalle", "sur_predalle": "sur la prédalle", "superieure": ""}
# le JOINT (lot 5) : les barres de couture, d'une couleur à elles ; le jour dessiné entre deux prédalles et la marge vue au-delà
# des barres (conventions de dessin, pas des données de calcul)
COUTURE = "#B5470B"
JOUR_JOINT = 20.0                                            # mm
MARGE_JOINT = 200.0                                          # mm, de chaque côté, au-delà du bout des barres


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


def _cote_v(x, y1, y2, texte, u, droite=False, taille=0.8) -> str:
    """Une cote verticale : le nombre tourné le long de la ligne."""
    t = 0.28 * u
    xt = x + (0.95 * u if droite else -0.4 * u)
    ym = (y1 + y2) / 2
    return (_ligne(x, y1, x, y2, COTE, 0.05 * u) + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, 0.06 * u)
            + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, 0.06 * u)
            + _txt(xt, ym, texte, taille * u, extra=f' transform="rotate(-90 {_f(xt)} {_f(ym)})"'))


def _attache(x1, y1, x2, y2, u) -> str:
    return _ligne(x1, y1, x2, y2, COTE, 0.04 * u, f' stroke-dasharray="{_f(0.2 * u)} {_f(0.15 * u)}"')


def _rupture_v(x, y1, y2, u) -> str:
    """Une ligne de rupture verticale (la bande continue au-delà) : un trait fin brisé en son milieu."""
    m, z = (y1 + y2) / 2, min(0.7 * u, 0.3 * (y2 - y1))
    pts = [(x, y1 - 0.4 * u), (x, m - z), (x + 0.45 * z, m - 0.35 * z), (x - 0.45 * z, m + 0.35 * z), (x, m + z), (x, y2 + 0.4 * u)]
    return (f'<polyline points="{" ".join(f"{_f(a)},{_f(b)}" for a, b in pts)}" fill="none" stroke="{ENCRE}" '
            f'stroke-width="{_f(0.05 * u)}"/>')


def _fleche(x1, y1, x2, y2, u, couleur=ENCRE, ep=None) -> str:
    """Une flèche du repère des directions."""
    ep = ep or 0.07 * u
    a = math.atan2(y2 - y1, x2 - x1)
    L, w = 0.45 * u, 0.2 * u
    bx, by = x2 - L * math.cos(a), y2 - L * math.sin(a)
    p1 = (bx + w * math.sin(a), by - w * math.cos(a))
    p2 = (bx - w * math.sin(a), by + w * math.cos(a))
    return (_ligne(x1, y1, bx, by, couleur, ep)
            + f'<polygon points="{_f(x2)},{_f(y2)} {_f(p1[0])},{_f(p1[1])} {_f(p2[0])},{_f(p2[1])}" fill="{couleur}"/>')


def _hachures(x, y, w, h, pas, ep) -> str:
    """Des hachures à 45° dans le rectangle (x, y, w, h), découpées à la main (pas de motif SVG : rien qui dépende de
    l'identifiant d'un autre dessin de la page)."""
    out = []
    k = 0.0
    while k < w + h:
        # la droite x' − x + (y' − y) = k, coupée par le rectangle
        x1, y1 = x + max(0.0, k - h), y + min(h, k)
        x2, y2 = x + min(w, k), y + max(0.0, k - w)
        if x2 - x1 > 1e-6:
            out.append(f"M{_f(x1)} {_f(y1)}L{_f(x2)} {_f(y2)}")
        k += pas
    return f'<path d="{"".join(out)}" stroke="{HACHURE}" stroke-width="{_f(ep)}" fill="none"/>'


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


def _teinte(j: int) -> str:
    return TEINTES[j % len(TEINTES)]



def _texte_vue(b_mm: float, W: float) -> str:
    return f"b = {_mm(b_mm)}" if W >= b_mm else f"{_mm(W)} vus sur b = {_mm(b_mm)}"


def _pose_txt(c: Dict, principale: str = "", fil: bool = False) -> str:
    """Où la couche est posée, pour la légende : posée par le site sur la nappe principale (``empilee``) — sur ce qui la
    porte, ``sur_court`` de dalle_positions.annoter (la base, ou un renfort principal plus haut qu'elle, S5) —, « axe
    saisi » (imposé dans l'état), ou la règle de la prédalle. ``fil`` : un fil d'un treillis unique (la légende commune,
    dessin_dalle.entrees_nappe) — « sur la base Y » n'en dit rien, ses deux directions sont soudées ; sa règle de pose,
    si."""
    if c.get("empilee") and not fil:
        if c.get("sur_court"):
            return f"sur {c['sur_court']}"                 # la forme courte : la ligne tient dans sa colonne
        return f"sur la base {principale.upper()}" if principale else "sur la base principale"
    if not c.get("auto", True):
        return "axe saisi"
    return POSES.get(c.get("pose"), "")       # (un fil soudé de la peau : « dans la prédalle », posé par annoter, DAL5b)


# ------------------------------------------------------------------ la coupe
def _coupe(s: Dict, dk: str, principale: str, b_mm: float, h_mm: float, hp_mm: float, u: float, y0: float,
           i: Optional[int] = None) -> Tuple[str, float]:
    """Une coupe de la bande perpendiculaire à la direction ``dk`` (ses barres en points, celles de l'autre direction en
    traits), posée à l'ordonnée ``y0`` : (svg, hauteur occupée) — un groupe ``data-coupe="<dk>"``. La clé des deux bétons
    et la légende des nappes sont COMMUNES aux deux coupes, sous elles (``_cle_betons``, dessin_dalle.legende_nappes)."""
    autre = "x" if dk == "y" else "y"
    x0, W = fenetre(b_mm)
    g = []
    titre = f"Direction {dk.upper()}" + (" · principale" if dk == principale else "")
    g.append(_txt(0, y0 + 0.9 * u, escape(titre), 0.95 * u, ancre="start"))
    yc = y0 + 2.4 * u                                            # le dessus du béton
    yp = yc + h_mm - hp_mm                                       # le plan de reprise : dessus de la prédalle
    couches = []
    for cote in ("inf", "sup"):
        for d_ in (dk, autre):
            for j, c in enumerate(s["faces"][f"{cote}_{d_}"]["couches"]):
                y = (h_mm - c["e_reel"] * 10.0) if cote == "inf" else c["e_reel"] * 10.0
                couches.append((cote, d_, j, c, y))
    y_bas = max([h_mm] + [y + c["d"] / 2 for *_, c, y in couches])
    # le béton : coulé en place (uni) au-dessus, prédalle (hachurée, plus soutenue) dessous ; bordés de lignes de rupture
    g.append(piece(i, "beton", f'<rect x="0" y="{_f(yc)}" width="{_f(W)}" height="{_f(h_mm - hp_mm)}" fill="{BETON}"/>'
                   + f'<rect x="0" y="{_f(yp)}" width="{_f(W)}" height="{_f(hp_mm)}" fill="{PEAU}"/>'
                   + _hachures(0, yp, W, hp_mm, 0.55 * u, 0.035 * u)
                   + _ligne(0, yc, W, yc, ENCRE, 0.09 * u) + _ligne(0, yc + h_mm, W, yc + h_mm, ENCRE, 0.09 * u)
                   + _ligne(0, yp, W, yp, ENCRE, 0.06 * u), "beton"))  # (le dernier trait : le plan de reprise)
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
    # les noms des deux bétons, à droite dans chaque zone, là où aucune barre ne passe (sinon la clé sous la coupe suffit)
    occupe = sorted((y - c["d"] / 2, y + c["d"] / 2) for *_, c, y in couches)
    for texte, z0, z1 in (("PRÉDALLE", h_mm - hp_mm, h_mm), ("COULÉ EN PLACE", 0.0, h_mm - hp_mm)):
        libres, a = [], z0
        for b0, b1 in occupe:
            if b1 <= a or b0 >= z1:
                continue
            if b0 > a:
                libres.append((a, b0))
            a = max(a, b1)
        if a < z1:
            libres.append((a, z1))
        a, b = max(libres, key=lambda x: x[1] - x[0], default=(0.0, 0.0))
        if b - a >= 0.95 * u:
            yy = yc + (a + b) / 2
            lt = 0.36 * u * len(texte) + 0.6 * u
            g.append(f'<rect x="{_f(W - lt - 0.5 * u)}" y="{_f(yy - 0.38 * u)}" width="{_f(lt)}" height="{_f(0.76 * u)}" '
                     f'rx="{_f(0.15 * u)}" fill="#FFFFFF" fill-opacity="0.88"/>')
            g.append(_txt(W - 0.5 * u - lt / 2, yy + 0.2 * u, texte, 0.54 * u, couleur="#4A5563"))
    # les cotes : à gauche h, et sa décomposition prédalle + coulé en place ; la largeur vue dessous
    xg1, xg2 = -1.3 * u, -2.9 * u
    g.append(_attache(-0.3 * u, yc, xg2 - 0.3 * u, yc, u) + _attache(-0.3 * u, yc + h_mm, xg2 - 0.3 * u, yc + h_mm, u)
             + _attache(-0.3 * u, yp, xg1 - 0.3 * u, yp, u))
    g.append(_cote_v(xg1, yp, yc + h_mm, _mm(hp_mm), u, taille=0.72))
    g.append(_cote_v(xg1, yc, yp, _mm(h_mm - hp_mm), u, taille=0.72))
    g.append(_cote_v(xg2, yc, yc + h_mm, f"h = {_mm(h_mm)}", u))
    y_cote = yc + y_bas + 1.4 * u
    g.append(_cote_h(0, W, y_cote, _texte_vue(b_mm, W), u))
    # à droite : l'axe des bases de la direction coupée, puis sa hauteur utile
    xd = W + 1.3 * u
    for cote in ("inf", "sup"):
        c = s["faces"][f"{cote}_{dk}"]["couches"][0]
        y = (h_mm - c["e_reel"] * 10.0) if cote == "inf" else c["e_reel"] * 10.0
        y_face = h_mm if cote == "inf" else 0.0
        g.append(_attache(W + 0.2 * u, yc + y, xd + 0.4 * u, yc + y, u)
                 + _cote_v(xd, yc + min(y, y_face), yc + max(y, y_face), _mm(c["e_reel"] * 10.0), u, droite=True))
    # la hauteur utile de la face TENDUE de la direction coupée (sup. si son moment l'emporte, comme la coupe des dalles) :
    # du parement comprimé au centre de gravité des couches tendues — y_G du calcul
    tendue = "sup" if s[f"M{dk}_sup"] > s[f"M{dk}_inf"] else "inf"
    yG = s["faces"][f"{tendue}_{dk}"]["yG"] * 10.0
    xd2 = W + 3.3 * u
    if tendue == "inf":
        y1, y2 = yc, yc + h_mm - yG          # du dessus du béton au centre de gravité des couches inférieures
    else:
        y1, y2 = yc + yG, yc + h_mm          # du centre de gravité des couches supérieures au dessous du béton
    g.append(_attache(W + 0.2 * u, y1, xd2 + 0.4 * u, y1, u) + _attache(W + 0.2 * u, y2, xd2 + 0.4 * u, y2, u)
             + _cote_v(xd2, y1, y2, f"d = {_mm(h_mm - yG)}", u, droite=True))
    return f'<g data-coupe="{dk}">{"".join(g)}</g>', y_cote + 0.6 * u - y0


def _coupe_joint(s: Dict, principale: str, h_mm: float, hp_mm: float, W: float, u: float, y0: float,
                 i: Optional[int] = None) -> Tuple[str, float]:
    """Le JOINT de deux prédalles (lot 5), coupé selon la direction SECONDAIRE, posé à l'ordonnée ``y0`` : les deux
    peaux (hachurées), le jour du joint rempli de béton coulé en place ; la base secondaire, dans la peau, qui s'arrête à
    l'enrobage de chaque bord ; les barres principales de la peau en points ; les renforts et la nappe supérieure ; les
    BARRES DE COUTURE à leur position réelle (``s["joint"]``), centrées sur le joint, leurs deux longueurs cotées. Tout est
    dessiné à une même échelle ``k`` (la vue tient dans la largeur ``W`` des coupes). (svg, hauteur occupée) — un groupe
    ``data-joint="<direction>"``, la couture une pièce ``s<i>.joint-0``."""
    j = s["joint"]
    sec, c0, lg = j["direction"], j["couture"], j["longueur"]
    cote_mm = lg["cote"] * 10.0
    L = 2.0 * cote_mm + 2.0 * MARGE_JOINT
    k = min(1.0, W / L)
    xj = L / 2.0
    X = lambda x: x * k                                     # noqa: E731 — l'échelle de la vue
    g = [_txt(0, y0 + 0.9 * u, escape(f"Joint des prédalles · direction {sec.upper()}"), 0.95 * u, ancre="start")]
    yc = y0 + 2.4 * u
    H, HP = h_mm * k, hp_mm * k
    yp = yc + H - HP
    jour = JOUR_JOINT / 2.0
    enr = s["grandeurs"]["enrobage_calcul"] * 10.0
    beton = (f'<rect x="0" y="{_f(yc)}" width="{_f(X(L))}" height="{_f(H)}" fill="{BETON}"/>'
             + "".join(f'<rect x="{_f(X(a))}" y="{_f(yp)}" width="{_f(X(b_ - a))}" height="{_f(HP)}" fill="{PEAU}"/>'
                       + _hachures(X(a), yp, X(b_ - a), HP, 0.55 * u, 0.035 * u) for a, b_ in ((0.0, xj - jour), (xj + jour, L)))
             + _ligne(0, yc, X(L), yc, ENCRE, 0.09 * u) + _ligne(0, yc + H, X(L), yc + H, ENCRE, 0.09 * u)
             + _ligne(0, yp, X(xj - jour), yp, ENCRE, 0.06 * u) + _ligne(X(xj + jour), yp, X(L), yp, ENCRE, 0.06 * u)
             + _ligne(X(xj - jour), yp, X(xj - jour), yc + H, ENCRE, 0.06 * u) + _ligne(X(xj + jour), yp, X(xj + jour), yc + H, ENCRE, 0.06 * u))
    g.append(piece(i, "beton", beton, "beton"))
    g.append(_rupture_v(0, yc, yc + H, u) + _rupture_v(X(L), yc, yc + H, u))
    lignes, points = [], []
    for cote in ("inf", "sup"):
        for d_ in (sec, principale):
            for jj, c in enumerate(s["faces"][f"{cote}_{d_}"]["couches"]):
                y = yc + ((h_mm - c["e_reel"] * 10.0) if cote == "inf" else c["e_reel"] * 10.0) * k
                ep = max(c["d"] * k, 0.04 * u)
                dans_peau = cote == "inf" and jj == 0               # (une base inférieure : dans la peau, coupée au joint)
                if d_ == sec:
                    morceaux = ((enr, xj - jour - enr), (xj + jour + enr, L - enr)) if dans_peau else ((0.0, L),)
                    lignes.append(piece(i, f"{cote}_{d_}-{jj}", "".join(_ligne(X(a), y, X(b_), y, _teinte(jj), ep) for a, b_ in morceaux)))
                else:
                    r = c["d"] * k / 2.0
                    pas = float(c["esp"]) if c["type"] != "n_barres" else 200.0
                    xs, x = [], pas / 2.0
                    while x < L:
                        if not (dans_peau and abs(x - xj) < jour + enr):
                            xs.append(x)
                        x += pas
                    points.append(piece(i, f"{cote}_{d_}-{jj}", "".join(
                        f'<circle cx="{_f(X(xx))}" cy="{_f(y)}" r="{_f(max(r, 0.05 * u))}" fill="{_couleur(jj)}" stroke="#FFFFFF" '
                        f'stroke-width="{_f(max(0.3 * r, 0.03 * u))}" paint-order="stroke"/>' for xx in xs)))
    g += lignes + points
    yb = yc + (h_mm - c0["e_reel"] * 10.0) * k
    g.append(piece(i, "joint-0", _ligne(X(xj - cote_mm), yb, X(xj + cote_mm), yb, COUTURE, max(c0["d"] * k, 0.06 * u)), "couche"))
    # le joint : son axe, en tirets, et son nom ; les deux longueurs de la couture, cotées sous la vue
    g.append(_ligne(X(xj), yc - 0.4 * u, X(xj), yc + H + 0.4 * u, COTE, 0.04 * u, ' stroke-dasharray="4 3"'))
    g.append(_txt(X(xj), yc - 0.6 * u, "joint", 0.7 * u, couleur="#4A5563"))
    yq = yc + H + 1.4 * u
    g.append(_cote_h(X(xj - cote_mm), X(xj), yq, _mm(cote_mm), u) + _cote_h(X(xj), X(xj + cote_mm), yq, _mm(cote_mm), u))
    # à droite : la position réelle de la couture, depuis le dessous
    xd = X(L) + 1.3 * u
    g.append(_attache(X(L) + 0.2 * u, yb, xd + 0.4 * u, yb, u) + _cote_v(xd, yb, yc + H, _mm(c0["e_reel"] * 10.0), u, droite=True))
    lib = f"barres de couture {c0['libelle']} · 2 × {_mm(cote_mm)} mm (l₀ = {_mm(round(lg['l0']))} mm)"
    g.append(f'<rect x="0" y="{_f(yq + 1.0 * u)}" width="{_f(1.3 * u)}" height="{_f(0.22 * u)}" fill="{COUTURE}"/>')
    g.append(_txt(1.7 * u, yq + 1.35 * u, escape(lib), 0.74 * u, ancre="start", gras=False))
    return f'<g data-joint="{sec}">{"".join(g)}</g>', yq + 1.9 * u - y0


def _cle_betons(W: float, u: float, y: float, h_mm: float, hp_mm: float) -> Tuple[str, float]:
    """La clé des deux bétons, une fois sous les coupes (DAL5 : elle se répétait sous chacune) : la prédalle préfabriquée
    (hachurée) et le béton coulé en place, avec leurs épaisseurs. (svg, hauteur occupée)."""
    yk = y + 0.6 * u
    g = [f'<rect x="0" y="{_f(yk - 0.62 * u)}" width="{_f(1.3 * u)}" height="{_f(0.75 * u)}" fill="{PEAU}" stroke="{ENCRE}" '
         f'stroke-width="{_f(0.03 * u)}"/>' + _hachures(0, yk - 0.62 * u, 1.3 * u, 0.75 * u, 0.3 * u, 0.03 * u),
         _txt(1.7 * u, yk, f"prédalle préfabriquée · {_mm(hp_mm)} mm", 0.74 * u, ancre="start", gras=False)]
    xk = max(14.5 * u, (W + 4.6 * u) / 2)
    g.append(f'<rect x="{_f(xk)}" y="{_f(yk - 0.62 * u)}" width="{_f(1.3 * u)}" height="{_f(0.75 * u)}" fill="{BETON}" stroke="{ENCRE}" '
             f'stroke-width="{_f(0.03 * u)}"/>')
    g.append(_txt(xk + 1.7 * u, yk, f"béton coulé en place · {_mm(h_mm - hp_mm)} mm", 0.74 * u, ancre="start", gras=False))
    return "".join(g), 1.6 * u


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


def svg_coupe(ident: str, s: Dict, predalle: Dict, i: Optional[int] = None) -> str:
    """Les deux coupes d'une section (la direction principale d'abord), à la même échelle, puis, une fois, la clé des
    deux bétons et la légende COMMUNE des nappes (dessin_dalle.legende_nappes, la pose de la prédalle dite) ; ``i`` : le
    rang de la section, pour ses pièces cliquables."""
    b_mm, h_mm, hp_mm = predalle["b"] * 10.0, predalle["h"] * 10.0, predalle["h_pre"] * 10.0
    _, W = fenetre(b_mm)
    u = max(W, 420.0) / 34.0
    p = s["principale"]
    g, y = [], 0.0
    if i is None:
        g.append(_txt(max(W, 27.8 * u) / 2, 0.2 * u, _efforts(s), 0.9 * u, couleur=EFFORT))
    # (l'établi compact, rang i : les efforts sont écrits UNE fois, dans la bande de l'en-tête de la vue, sans rouge)
    y = 0.9 * u if i is None else 0.0
    for dk in (p, "x" if p == "y" else "y"):
        morceau, hh = _coupe(s, dk, p, b_mm, h_mm, hp_mm, u, y, i)
        g.append(morceau)
        y += hh + 0.8 * u
    if s.get("joint"):                              # (le joint des prédalles et ses barres de couture, lot 5)
        morceau, hh = _coupe_joint(s, p, h_mm, hp_mm, W, u, y, i)
        g.append(morceau)
        y += hh + 0.8 * u
    cle, hk = _cle_betons(W, u, y, h_mm, hp_mm)
    y0 = y + hk + 0.4 * u
    legende, hl = legende_nappes(s, p, W, u, y0, lambda c, fil=False: _pose_txt(c, p, fil), taille=0.74)
    g.append(cle + legende)
    note = rappel_jeu(predalle)
    if note:                                        # (le jeu et l'arrondi : dans le calcul seulement, dit discrètement)
        g.append(_txt(0.0, y0 + hl + 0.5 * u, escape(note), 0.66 * u, ancre="start", couleur=COTE, gras=False))
        hl += 1.0 * u
    y = y0 + hl + 0.6 * u                           # (le bas du cadre : 0,6 u sous la dernière ligne, ses jambages compris)
    largeur = max(W + 10.2 * u, 32.8 * u)
    haut = -1.2 * u if i is None else -0.3 * u
    vb = f"{_f(-5.0 * u)} {_f(haut)} {_f(largeur)} {_f(y + 0.6 * u - haut - 1.2 * u)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Coupes de la prédalle, section {escape(s["nom"])}">{"".join(g)}</svg>')


# ------------------------------------------------------------------ le plan des nappes
def _plan(s: Dict, cote: str, principale: str, b_mm: float, u: float, dx: float, i: Optional[int] = None) -> Tuple[str, float]:
    """La nappe ``cote`` vue de dessus, dans un carré de la largeur vue : barres X horizontales, Y verticales."""
    x0, W = fenetre(b_mm)
    g = [f'<g transform="translate({_f(dx)} 0)">']
    g.append(_txt(W / 2, -2.6 * u, NAP.TITRES[cote], 1.0 * u))   # « Armature inférieure » (plus « Nappe », 13/11/2026)
    fond = PEAU if cote == "inf" else PAPIER_PLAN
    g.append(f'<rect x="0" y="0" width="{_f(W)}" height="{_f(W)}" fill="{fond}" stroke="{COTE}" stroke-width="{_f(0.05 * u)}" '
             f'stroke-dasharray="{_f(0.6 * u)} {_f(0.18 * u)} {_f(0.1 * u)} {_f(0.18 * u)}"/>')
    traits = []
    for dk in ("x", "y"):
        for j, c in enumerate(s["faces"][f"{cote}_{dk}"]["couches"]):
            haut = c["e_reel"] if cote == "inf" else -c["e_reel"]
            traits.append((haut, dk, j, c))
    traits.sort(key=lambda t: t[0])
    for _, dk, j, c in traits:
        ep = max(c["d"], 0.04 * u)
        marge = c["d"] / 2 + 0.1 * u
        barres = ""
        for p in positions(c, b_mm, j):
            v = p - x0
            if not marge < v < W - marge:
                continue
            if dk == "x":
                barres += _ligne(0.1 * u, W - v, W - 0.1 * u, W - v, _couleur(j), ep)
            else:
                barres += _ligne(v, 0.1 * u, v, W - 0.1 * u, _couleur(j), ep)
        g.append(piece(i, f"{cote}_{dk}-{j}", barres))
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
    # la légende : chaque armature une fois (un treillis unique en une entrée), avec sa pose (dans / sur la prédalle)
    y_leg = W + 3.0 * u
    n = 0
    for j, texte, gras in entrees_plan(s, cote, principale, lambda c: _pose_txt(c, principale)):
        yy = y_leg + n * 1.35 * u
        g.append(_ligne(0, yy - 0.3 * u, 0.8 * u, yy - 0.3 * u, _couleur(j), 0.22 * u))
        g.append(_txt(1.1 * u, yy, escape(texte), 0.7 * u, ancre="start", gras=gras))
        n += 1
    g.append("</g>")
    return "".join(g), y_leg + n * 1.35 * u


def _repere(x, y, u, principale: str) -> str:
    """Le repère des directions : X vers la droite, Y vers le haut ; la principale en trait fort, nommée à côté."""
    L = 2.2 * u
    out = _fleche(x, y, x + L, y, u, ep=(0.12 if principale == "x" else 0.06) * u)
    out += _fleche(x, y, x, y - L, u, ep=(0.12 if principale == "y" else 0.06) * u)
    out += _txt(x + L + 0.25 * u, y + 0.3 * u, "X", 0.9 * u, ancre="start", gras=principale == "x")
    out += _txt(x, y - L - 0.3 * u, "Y", 0.9 * u, gras=principale == "y")
    out += _txt(x + L + 1.5 * u, y + 0.3 * u, f"barres X horizontales, Y verticales · direction principale : {principale.upper()} · fond grisé : la prédalle", 0.74 * u,
                ancre="start", couleur=COTE, gras=False)
    return out


def svg_plan(ident: str, s: Dict, predalle: Dict, i: Optional[int] = None) -> str:
    """Les nappes inférieure et supérieure d'une section, côte à côte, à la même échelle, avec le repère des directions ;
    ``i`` : le rang de la section, pour ses pièces cliquables."""
    b_mm = predalle["b"] * 10.0
    _, W = fenetre(b_mm)
    u = max(W, 420.0) / 16.0
    p = s["principale"]
    gap = 5.2 * u
    g1, h1 = _plan(s, "inf", p, b_mm, u, 0.0, i)
    g2, h2 = _plan(s, "sup", p, b_mm, u, W + gap, i)
    y_rep = max(h1, h2) + 2.8 * u
    rep = _repere(0.3 * u, y_rep, u, p)
    haut = y_rep + 0.8 * u
    vb = f"{_f(-0.8 * u)} {_f(-3.8 * u)} {_f(2 * W + gap + 3.4 * u)} {_f(haut + 4.0 * u)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Plan des armatures, section {escape(s["nom"])}">{g1}{g2}{rep}</svg>')


# ------------------------------------------------------------------ toutes les sections (l'établi compact, vue « Toutes »)
def svg_toutes(ident: str, sections: List[Dict], predalle: Dict) -> str:
    """Toutes les sections l'une sous l'autre, à la même échelle : pour chacune, son nom et ses sollicitations, puis sa
    coupe perpendiculaire à sa direction principale (la prédalle hachurée), avec ses pièces cliquables, et la légende de
    ses nappes ; la clé des deux bétons une fois, à la fin (la même pour toutes)."""
    b_mm, h_mm, hp_mm = predalle["b"] * 10.0, predalle["h"] * 10.0, predalle["h_pre"] * 10.0
    _, W = fenetre(b_mm)
    u = max(W, 420.0) / 34.0
    large = max(W, 27.8 * u)
    g, y = [], 0.0
    for i, s in enumerate(sections):
        p = s["principale"]
        g.append(_txt(0, y + 0.3 * u, escape(s["titre"]), 1.05 * u, ancre="start"))
        g.append(piece(i, "efforts", _txt(large, y + 0.3 * u, _efforts(s), 0.85 * u, ancre="end", couleur=COTE, gras=False), "efforts"))
        y += 1.0 * u
        morceau, hh = _coupe(s, p, p, b_mm, h_mm, hp_mm, u, y, i)
        g.append(morceau)
        y += hh + 0.4 * u
        legende, hl = legende_nappes(s, p, W, u, y + 0.4 * u, lambda c, fil=False, p=p: _pose_txt(c, p, fil), taille=0.74)
        g.append(legende)
        y += hl + 1.8 * u
    cle, hk = _cle_betons(W, u, y - 1.4 * u, h_mm, hp_mm)
    g.append(cle)
    y += hk
    largeur = max(W + 10.2 * u, 32.8 * u)
    vb = f"{_f(-5.0 * u)} {_f(-1.4 * u)} {_f(largeur)} {_f(y)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="Toutes les sections de la prédalle">{"".join(g)}</svg>')

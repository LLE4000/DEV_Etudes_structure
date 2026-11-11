# -*- coding: utf-8 -*-
"""La COUPE COTÉE d'un hourdis alvéolé de la Bibliothèque Béton armé (rubrique « Hourdis »), en mm.

Le TRACÉ est le relevé vectoriel de la coupe du fabricant (``dessin`` du catalogue : contour et alvéoles, relevés dans
le PDF par tools/transcrire_hourdis_ergon.py) ; les COTES sont celles que le fabricant imprime (``cotes`` : chaîne
horizontale entre axes d'alvéoles, chaîne verticale, dimensions des alvéoles) — rien n'est mesuré sur le tracé pour
être coté. Les axes des alvéoles sont posés aux cotes du fabricant ; la hauteur des alvéoles profilées (SPN) est dans la
légende (``legende_alveoles``).

La Toile et le style sont ceux des dessins de la bibliothèque (``api_web/dessin_enrobage.py`` : traits noirs, béton
gris clair, cotes fines) ; l'unité de texte ``U`` (mm) tient le texte à 12 px environ quand la coupe est montrée à
560 px de large (la largeur minimale que la page lui donne, avec un défilement au téléphone).
"""
from html import escape
from typing import Dict, List

from .dessin_enrobage import Toile, _f, _mm
from .dessin_poutre import BETON, COTE, ENCRE

U = 30.0                         # mm : l'unité de texte du dessin
AXE = "#6A7787"                  # les axes des alvéoles (traits mixtes)


def _chemin(polys: List[List[List[float]]], H: float) -> str:
    """Un chemin SVG fermé par polygone, y du catalogue (vers le haut) retourné vers le bas de l'écran."""
    return " ".join("M" + " L".join(f"{_f(x)},{_f(H - y)}" for x, y in pts) + " Z" for pts in polys)


def _etiquettes_alveoles(p: Dict) -> List[tuple]:
    """(abscisse du centre, texte) des alvéoles à nommer : la première de chaque dimension — « Ø 98 » pour une alvéole
    circulaire, sa largeur pour une alvéole oblongue ou profilée (la hauteur est dans la chaîne verticale, ou cotée à
    part pour le SPN)."""
    a = p["cotes"]["alveoles"]
    centres, x = [], 0.0
    for e in p["cotes"]["chaine_horizontale"][:-1]:
        x += e
        centres.append(x)
    if a["forme"] == "circulaire":
        return [(centres[0], f"Ø{chr(160)}{_mm(a['diametre'])}")]
    if a["forme"] == "oblongue":
        return [(centres[0], _mm(a["largeur"]))]
    vus, out = set(), []
    for c, l in zip(centres, a["largeurs"]):
        if l not in vus:
            vus.add(l)
            out.append((c, _mm(l)))
    return out


def _cote_v_droite(T: Toile, x: float, y1: float, y2: float, texte: str, k: float = 0.85) -> None:
    """Une cote verticale dont le texte s'écrit à l'HORIZONTALE, à droite du trait, au milieu du segment : il tient
    même sur les membrures de 34 mm, où un texte couché le long du trait déborderait."""
    t = T.u * 0.42
    T.ligne(x, y1, x, y2)
    for y in (y1, y2):
        T.ligne(x - t, y + t, x + t, y - t, ENCRE, 1.4)
    T.txt(x + T.u * 0.4, (y1 + y2) / 2.0 + T.u * 0.3, texte, k, "start")


def legende_alveoles(p: Dict) -> str:
    """Les alvéoles en mots, pour la légende de la coupe : « 8 alvéoles Ø 98 », « 5 alvéoles 169 × 197 »,
    « 9 alvéoles de 93, 102 et 44 × 140 »."""
    a = p["cotes"]["alveoles"]
    if a["forme"] == "circulaire":
        return f"{a['nombre']} alvéoles Ø{chr(160)}{_mm(a['diametre'])}"
    if a["forme"] == "oblongue":
        return f"{a['nombre']} alvéoles {_mm(a['largeur'])}{chr(160)}×{chr(160)}{_mm(a['hauteur'])}"
    largeurs = list(dict.fromkeys(a["largeurs"]))
    return (f"{a['nombre']} alvéoles de {', '.join(_mm(x) for x in largeurs[:-1])} et {_mm(largeurs[-1])}"
            f"{chr(160)}×{chr(160)}{_mm(a['hauteur'])}")


def coupe(ident: str, p: Dict) -> str:
    """La coupe cotée du profil ``p`` (une entrée de ``profils`` d'un catalogue) en SVG."""
    T = Toile(U)
    H, B = float(p["h"]), float(p["b"])
    c = p["cotes"]
    d = p["dessin"]
    # le béton et ses alvéoles (un seul chemin, règle « evenodd » : les alvéoles sont des trous)
    T.g.append(f'<path d="{_chemin([d["contour"]] + d["alveoles"], H)}" fill="{BETON}" fill-rule="evenodd" stroke="{ENCRE}" '
               f'stroke-width="{_f(U * 0.07)}" stroke-linejoin="round"/>')
    # les axes des alvéoles, aux cotes du fabricant, prolongés jusqu'à la chaîne horizontale
    y_ch = H + 1.5 * U
    xs, x = [0.0], 0.0
    for e in c["chaine_horizontale"]:
        x += e
        xs.append(x)
    for xa in xs[1:-1]:
        T.ligne(xa, -0.5 * U, xa, y_ch + 0.35 * U, AXE, 0.8, f"{_f(U * 0.6)} {_f(U * 0.18)} {_f(U * 0.1)} {_f(U * 0.18)}")
    for xb in (0.0, B):                                          # lignes d'attache des bords
        T.ligne(xb, H + 0.25 * U, xb, H + 3.35 * U, COTE, 0.8)
    # la chaîne horizontale (entre axes, bords compris) et la largeur
    for x1, x2, e in zip(xs[:-1], xs[1:], c["chaine_horizontale"]):
        T.cote_h(x1, x2, y_ch, _mm(e), k=0.9)
    T.cote_h(0.0, B, H + 3.1 * U, _mm(B), k=0.95)
    # la chaîne verticale (du haut vers le bas, textes à l'horizontale) et la hauteur, à droite
    x_v, x_h = B + 1.3 * U, B + 4.9 * U
    T.ligne(B + 0.2 * U, 0.0, x_h + 0.35 * U, 0.0, COTE, 0.8)
    T.ligne(B + 0.2 * U, H, x_h + 0.35 * U, H, COTE, 0.8)
    ys, y = [0.0], 0.0
    for e in c["chaine_verticale"]:
        y += e
        ys.append(y)
    for y1, y2, e in zip(ys[:-1], ys[1:], c["chaine_verticale"]):
        _cote_v_droite(T, x_v, y1, y2, _mm(e))
        if 0 < y2 < H:
            T.ligne(B + 0.2 * U, y2, x_v + 0.35 * U, y2, COTE, 0.8)
    T.cote_v(x_h, 0.0, H, _mm(H), k=0.95)
    # les alvéoles : leur dimension, dans la première de chaque sorte
    yc = H - (min(y for v in d["alveoles"] for _, y in v) + max(y for v in d["alveoles"] for _, y in v)) / 2.0
    for xc, t in _etiquettes_alveoles(p):
        T.txt(xc, yc + U * 0.33, t, 0.85, couleur=ENCRE, gras=True)
    x_min, x_max = -1.0 * U, x_h + 0.9 * U
    y_min, y_max = -1.2 * U, H + 3.6 * U
    vb = f"{_f(x_min)} {_f(y_min)} {_f(x_max - x_min)} {_f(y_max - y_min)}"
    return (f'<svg id="{escape(ident)}" xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="100%" role="img" '
            f'aria-label="{escape(p["nom"])} : coupe cotée, h {_mm(H)} mm, b {_mm(B)} mm">{"".join(T.g)}</svg>')

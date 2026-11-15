# -*- coding: utf-8 -*-
"""Le PLAN d'une fondation de grue (module « Générateur SAF », api_web/saf.py) : la dalle, le châssis posé, les surfaces
d'appui numérotées P1 à P4, les plots, les bandes d'intégration, les cotes — et, par-dessus la dalle, la carte de la
contrainte de contact ou du tassement d'un cas.

Ce fichier ne calcule rien : il DESSINE le modèle (saf/fondation_grue/projet.modele) et un champ du calcul
(saf/fondation_grue/verification.calculer). Les couleurs sont celles de la bibliothèque des châssis de grue
(grues/chassis/src/chassis_dessin.js) : les surfaces d'appui en orange, les cotes en bleu, les axes d'appui en rouge,
trait mixte.

LE PLAN SE MANIPULE (js/plan.js, qui ne sait rien de ce qui est dessiné — le moteur écrit dans le dessin ce que chaque
geste règle) :
- ``data-plan-glisser`` : un objet qui se déplace (le châssis en bloc, un sommet du contour, un bord du rectangle) —
  {"chemin", "x", "y" (les valeurs de départ ; une seule : le geste n'a qu'un sens), "ech" (unités du dessin par mètre),
  "pas" (la grille), "ox", "oy" (l'origine de la grille), "ax", "ay" (les accroches), "noms", "unite"} ;
- ``data-plan-tourner`` : la poignée de rotation — {"chemin", "cx", "cy" (le centre, unités du dessin), "angle",
  "accroches", "cible" (le groupe qui tourne)} ;
- ``data-plan-cote`` : une cote qui se tape — {"chemin", "valeur", "unite"} ;
- ``data-plan-sec`` : un objet qui ouvre sa rubrique ; ``data-plan-clic`` : un objet qu'un appui bref RÈGLE — {"chemin",
  "valeur"} (un bord de la dalle devient le repère des cotes du châssis) ; ``data-plan-bord`` : un bord du contour (deux
  appuis : un sommet de plus) ; ``data-plan-fond`` : la feuille (un clic hors de la dalle : un sommet de plus, en contour).

    plan(el, m, champ=None, interactif=True, carte="image") -> le SVG
    champ : None | {"mode": "sigma" | "w" | "m" | "taux", "dalle" (la dalle maillée), "valeurs" (aux nœuds), "decolle"
            (nœuds décollés, booléens), "maxi" (la valeur pointée), "x", "y" (où), "limite" (la valeur du trait noir — kPa
            pour σ, 1 pour un taux de travail — ou None), "unite" ((symbole, diviseur, décimales) de l'affichage : σ en
            kPa ÷ diviseur ; lot 4, § 2.4), "symbole" (σ, w, m…), "titre"}
"""
import base64
import json
import math
import struct
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from saf.fondation_grue import geometrie as G
from saf.fondation_grue import projet as PR

from . import commun as C

ECH = 100.0                                   # unités du dessin par mètre
ENCRE, SCHEMA, BETON, BETON_BORD = "#1d2b3a", "#8793a1", "#eceff2", "#64748b"
APPUI, APPUI_FOND = "#d9480f", "#ffd9c2"
COTE, AXE, ROUGE, ROUGE_FOND = "#2c6cb0", "#b4384e", "#dc2626", "#fee2e2"
BANDE = "#2c6cb0"
PLOT, PLOT_FOND = "#475569", "#cbd5e1"
DECOLLE = (148, 163, 184)                     # la zone décollée, sur la carte
POLICE = "Inter,system-ui,sans-serif"
# l'échelle des couleurs (lot 4, § 2.3) : UN dégradé à plusieurs couleurs, type « turbo » — bleu foncé, cyan, vert, jaune,
# orange, rouge —, de 0 au maximum affiché (pour la contrainte : au plus grand de σ max et de la limite). Au-delà de la
# limite, on ne passe plus en rouge uni : la limite est un TRAIT NOIR sur la légende et une LIGNE NOIRE sur le plan. Le même
# dégradé pour la contrainte, le tassement, les moments et le taux de travail du treillis.
TURBO = ((0.0, (48, 18, 59)), (0.13, (65, 88, 206)), (0.27, (40, 160, 240)), (0.40, (27, 214, 200)), (0.53, (70, 245, 120)),
         (0.66, (170, 250, 55)), (0.77, (240, 215, 50)), (0.87, (252, 150, 40)), (0.95, (225, 70, 15)), (1.0, (170, 20, 5)))
LIMITE = "#000000"                            # le trait de la limite (légende) et la ligne σ = limite (plan)
IMAGE_PIXELS = 220                            # la carte en image : son plus grand côté, après lissage
NIVEAUX_VECTEUR = 14                          # la carte en vecteurs (la note PDF) : des paliers de couleur


def _n(x: float, nd: int = 2) -> str:
    """Un nombre du plan : « 8,00 », « 0,75 »."""
    return C.fr(x, nd)


def _attr(d: Dict) -> str:
    return C.esc(json.dumps(d, ensure_ascii=False, separators=(",", ":")))


class _Vue:
    """Le repère du dessin : x vers la droite, y vers le HAUT (le modèle) → y vers le bas (le SVG)."""

    def __init__(self, contour: Sequence, autres: Sequence[Sequence]):
        self.x0, self.y0, self.x1, self.y1 = G.boite(contour)
        pts = [p for poly in autres for p in poly]
        self.gx0, self.gy0 = min([self.x0] + [p[0] for p in pts]), min([self.y0] + [p[1] for p in pts])
        self.gx1, self.gy1 = max([self.x1] + [p[0] for p in pts]), max([self.y1] + [p[1] for p in pts])
        self.t = max((self.gx1 - self.gx0) * ECH, (self.gy1 - self.gy0) * ECH, 300.0) * 0.024      # la taille du texte

    def X(self, x: float) -> float:
        return (x - self.x0) * ECH

    def Y(self, y: float) -> float:
        return (self.y1 - y) * ECH

    def pts(self, poly: Sequence) -> str:
        return " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in poly)


def _texte(x: float, y: float, txt: str, t: float, couleur: str = ENCRE, ancre: str = "middle", gras: bool = False, angle: float = 0.0,
           attrs: str = "", halo: bool = True) -> str:
    """Un texte, sur un liseré blanc qui le détache du dessin (comme les cotes de la bibliothèque des châssis)."""
    rot = f' transform="rotate({angle:.1f} {x:.1f} {y:.1f})"' if abs(angle) > 0.05 else ""
    g = ' font-weight="600"' if gras else ""
    commun = f'x="{x:.1f}" y="{y:.1f}" font-size="{t:.1f}" font-family="{POLICE}" text-anchor="{ancre}"{g}{rot}'
    fond = (f'<text {commun} fill="none" stroke="#ffffff" stroke-width="{t * 0.32:.1f}" stroke-linejoin="round">{C.esc(txt)}</text>'
            if halo else "")
    return f'{fond}<text {commun} fill="{couleur}"{attrs}>{C.esc(txt)}</text>'


def _lisible(angle: float) -> float:
    """L'angle d'un texte le long d'une ligne, ramené dans [−90°, 90°[ : il se lit de gauche à droite, et de bas en haut
    le long d'une verticale."""
    while angle >= 90.0 - 1e-9:
        angle -= 180.0
    while angle < -90.0 - 1e-9:
        angle += 180.0
    return angle


def _cote(v: _Vue, a, b, decal: float, txt: str, couleur: str = COTE, attrs: str = "", rappel: bool = True) -> str:
    """Une cote entre deux points du modèle. ``decal`` (unités du dessin) : la ligne de cote est décalée de ce côté de
    a→b — positif : à gauche du sens de marche (au-dessus d'une cote qui va de gauche à droite). Le texte est centré
    au-dessus de la ligne ; une cote trop courte pour lui l'écrit au bout."""
    ax, ay, bx, by = v.X(a[0]), v.Y(a[1]), v.X(b[0]), v.Y(b[1])
    L = math.hypot(bx - ax, by - ay)
    if L < 0.5:
        return ""
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = uy, -ux                                             # la normale, à gauche du sens de marche (y vers le bas)
    t = v.t
    p1, p2 = (ax + nx * decal, ay + ny * decal), (bx + nx * decal, by + ny * decal)
    out = f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" stroke="{couleur}" stroke-width="{t * 0.07:.2f}"/>'
    s = 1.0 if decal >= 0 else -1.0
    for (px, py), (qx, qy) in (((ax, ay), p1), ((bx, by), p2)):
        if rappel and abs(decal) > t * 0.3:                      # la ligne de rappel, du point à la ligne de cote
            out += (f'<line x1="{px + nx * s * t * 0.25:.1f}" y1="{py + ny * s * t * 0.25:.1f}" x2="{qx + nx * s * t * 0.35:.1f}" '
                    f'y2="{qy + ny * s * t * 0.35:.1f}" stroke="{couleur}" stroke-width="{t * 0.045:.2f}"/>')
        out += (f'<line x1="{qx - (ux + nx) * t * 0.28:.1f}" y1="{qy - (uy + ny) * t * 0.28:.1f}" x2="{qx + (ux + nx) * t * 0.28:.1f}" '
                f'y2="{qy + (uy + ny) * t * 0.28:.1f}" stroke="{couleur}" stroke-width="{t * 0.1:.2f}"/>')
    ang = _lisible(math.degrees(math.atan2(uy, ux)))
    hx, hy = math.sin(math.radians(ang)), -math.cos(math.radians(ang))          # « au-dessus » du texte
    large = len(txt) * t * 0.56
    if L >= large + t * 0.6:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        out += _texte(mx + hx * t * 0.32, my + hy * t * 0.32, txt, t, couleur, "middle", angle=ang, attrs=attrs)
    else:                                                        # au bout de la cote, du côté de b
        dx, dy = (ux, uy) if (ux * math.cos(math.radians(ang)) + uy * math.sin(math.radians(ang))) >= 0 else (-ux, -uy)
        bout = p2 if (dx, dy) == (ux, uy) else p1
        out += _texte(bout[0] + dx * t * 0.5 + hx * t * 0.32, bout[1] + dy * t * 0.5 + hy * t * 0.32, txt, t, couleur, "start", angle=ang,
                      attrs=attrs)
    return out


# ------------------------------------------------------------------ la carte (contrainte de contact, tassement)
def _couleurs(valeurs: np.ndarray, rampe: Sequence) -> np.ndarray:
    """Les couleurs (n × 3, 0 à 255) de valeurs déjà ramenées à l'échelle de la rampe."""
    xs = [x for x, _ in rampe]
    return np.stack([np.interp(valeurs, xs, [c[i] for _, c in rampe]) for i in range(3)], axis=-1)


def _echelle(champ: Dict) -> Tuple[Sequence, float]:
    """(la rampe, ce qui vaut 1 sur la rampe) : le dégradé « turbo », de 0 au maximum affiché — le plus grand du maximum
    du champ et de la limite, quand il y en a une (§ 2.3)."""
    if champ.get("par_element"):
        v_ = np.asarray(champ["valeurs"], float)
        maxi = max(float(np.nanmax(v_)) if np.isfinite(v_).any() else 0.0, 1e-9)
    else:
        maxi = max(float(np.nanmax(np.where(champ["dalle"].on0, champ["valeurs"], np.nan))), 1e-9)
    if champ.get("limite"):
        maxi = max(maxi, float(champ["limite"]))
    return TURBO, maxi


def iso_ligne(d, valeurs: np.ndarray, niveau: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """La ligne où le champ aux nœuds vaut ``niveau`` (σ = limite, taux = 100 %) : des segments [(x, y), (x, y)] en m,
    par les carrés marchants, dans les éléments de la dalle seulement."""
    nnx = d.nx + 1
    g = np.asarray(valeurs, float)
    n0 = d.J * nnx + d.I
    coins = np.stack([n0, n0 + 1, n0 + 1 + nnx, n0 + nnx], 1)
    v = g[coins] - niveau
    coupe = (v.min(axis=1) < 0) & (v.max(axis=1) >= 0)
    segs = []
    pos = ((0, 0), (1, 0), (1, 1), (0, 1))
    for e in np.flatnonzero(coupe):
        x0, y0 = d.x0 + d.I[e] * d.a, d.y0 + d.J[e] * d.b
        pts = []
        for k in range(4):
            a, b = v[e, k], v[e, (k + 1) % 4]
            if (a < 0) != (b < 0):
                t = a / (a - b)
                (ua, va), (ub, vb) = pos[k], pos[(k + 1) % 4]
                pts.append((x0 + (ua + t * (ub - ua)) * d.a, y0 + (va + t * (vb - va)) * d.b))
        for k in range(0, len(pts) - 1, 2):
            segs.append((pts[k], pts[k + 1]))
    return segs


def _iso(v: "_Vue", champ: Dict, ident: str) -> str:
    """La LIGNE NOIRE de la limite sur le plan (σ = limite ; taux = 100 %), découpée au contour, avec son étiquette."""
    if not champ.get("limite"):
        return ""
    segs = iso_ligne(champ["dalle"], champ["valeurs"], float(champ["limite"]))
    if not segs:
        return ""
    t = v.t
    d_ = "".join(f"M{v.X(a[0]):.1f} {v.Y(a[1]):.1f}L{v.X(b[0]):.1f} {v.Y(b[1]):.1f}" for a, b in segs)
    # l'étiquette au segment le plus haut puis le plus à droite (loin des pieds et de leurs repères, en général)
    a, b = max(segs, key=lambda sg: (round(sg[0][1] + sg[1][1], 3), sg[0][0] + sg[1][0]))
    xm, ym = v.X((a[0] + b[0]) / 2), v.Y((a[1] + b[1]) / 2)
    return (f'<g class="sf-iso" pointer-events="none"><path d="{d_}" fill="none" stroke="{LIMITE}" stroke-width="{t * 0.09:.2f}" '
            f'stroke-linecap="round" clip-path="url(#{ident}_dalle)"/>'
            + _texte(xm, ym - t * 0.35, _valeur(champ, float(champ["limite"])), t * 0.8, LIMITE, "middle", gras=True) + "</g>")


def _valeur(champ: Dict, x: float, unite: bool = True) -> str:
    """Une valeur du champ dans l'unité de l'affichage (σ : kg/cm², kPa ou MPa — § 2.4 ; un taux : en %)."""
    sym, div, nd = champ.get("unite") or ("kPa", 1.0, 1)
    if champ["mode"] == "taux":
        return f"{_n(x * 100.0, 0)} %"
    x = x * champ.get("signe", 1.0)            # (les moments en haut : la couleur suit |m|, la valeur garde son signe)
    return f"{_n(x / div, nd)} {sym}" if unite else _n(x / div, nd)


def _png(largeur: int, hauteur: int, rgba: bytes) -> bytes:
    """Une image PNG (RGBA, 8 bits) : la bibliothèque standard suffit."""
    def bloc(nom: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + nom + d + struct.pack(">I", zlib.crc32(nom + d) & 0xFFFFFFFF)
    lignes = b"".join(b"\x00" + rgba[y * 4 * largeur:(y + 1) * 4 * largeur] for y in range(hauteur))
    return (b"\x89PNG\r\n\x1a\n" + bloc(b"IHDR", struct.pack(">IIBBBBB", largeur, hauteur, 8, 6, 0, 0, 0))
            + bloc(b"IDAT", zlib.compress(lignes, 6)) + bloc(b"IEND", b""))


def _poids(n: int, f: int) -> np.ndarray:
    """Les poids d'une interpolation linéaire de n points en (n − 1)·f + 1 : une matrice (sorties × entrées)."""
    m = (n - 1) * f + 1
    pos = np.arange(m) / f
    i0 = np.minimum(np.floor(pos).astype(int), max(n - 2, 0))
    t = pos - i0
    W = np.zeros((m, n))
    W[np.arange(m), i0] = 1.0 - t
    if n > 1:
        W[np.arange(m), i0 + 1] += t
    return W


def _image(v: _Vue, d, rgba: np.ndarray, f: int, ident: str) -> str:
    """Une image (ny·f × nx·f, RGBA, la ligne du HAUT d'abord) posée sur la grille des éléments, découpée au contour."""
    hauteur, largeur = rgba.shape[:2]
    donnees = base64.b64encode(_png(largeur, hauteur, rgba.tobytes())).decode("ascii")
    x, y = v.X(d.x0), v.Y(d.y0 + d.ny * d.b)
    return (f'<image x="{x:.1f}" y="{y:.1f}" width="{d.nx * d.a * ECH:.1f}" height="{d.ny * d.b * ECH:.1f}" preserveAspectRatio="none" '
            f'clip-path="url(#{ident}_dalle)" href="data:image/png;base64,{donnees}"/>')


def _carte_elements(v: _Vue, champ: Dict, ident: str) -> str:
    """La carte d'un champ AU CENTRE DES ÉLÉMENTS (les moments de calcul) : une case par élément, à sa couleur, sans
    lissage — une moyenne à travers le bord d'un plot coulé inventerait des valeurs (§ 4.4)."""
    d = champ["dalle"]
    rampe, unite = _echelle(champ)
    f = max(1, min(8, IMAGE_PIXELS // max(d.nx, d.ny, 1)))
    rgba = np.zeros((d.ny, d.nx, 4))
    val = np.asarray(champ["valeurs"], float)
    fini = np.isfinite(val)                                    # (un élément sans valeur reste sans couleur)
    rgba[d.J, d.I, :3] = _couleurs(np.where(fini, val, 0.0) / unite, rampe)
    rgba[d.J, d.I, 3] = np.where(fini, 255, 0)
    img = np.rint(rgba).astype(np.uint8).repeat(f, axis=0).repeat(f, axis=1)[::-1]
    return _image(v, d, np.ascontiguousarray(img), f, ident)


def _carte_image(v: _Vue, champ: Dict, ident: str) -> str:
    """La carte en IMAGE (l'écran) : les valeurs aux nœuds, interpolées entre eux AVANT d'être mises en couleur (le
    passage de la limite reste une ligne nette), découpées au contour de la dalle. Un point hors de la dalle n'entre pas
    dans l'interpolation : le bord garde ses valeurs."""
    if champ.get("par_element"):
        return _carte_elements(v, champ, ident)
    d = champ["dalle"]
    rampe, unite = _echelle(champ)
    dedans = d.on0.reshape(d.ny + 1, d.nnx).astype(float)
    g = np.where(d.on0, champ["valeurs"], 0.0).reshape(d.ny + 1, d.nnx)
    dec = np.where(d.on0, champ["decolle"], False).reshape(d.ny + 1, d.nnx).astype(float)
    f = max(1, min(8, IMAGE_PIXELS // max(d.nx, d.ny, 1)))
    Wy, Wx = _poids(d.ny + 1, f), _poids(d.nnx, f)
    poids = Wy @ dedans @ Wx.T
    sur = np.maximum(poids, 1e-12)
    rgb = _couleurs((Wy @ g @ Wx.T) / sur / unite, rampe)
    rgb = np.where(((Wy @ dec @ Wx.T) / sur > 0.5)[..., None], np.array(DECOLLE, dtype=float), rgb)
    alpha = np.where(poids > 1e-9, 255, 0)
    img = np.concatenate([np.rint(rgb), alpha[..., None]], axis=-1).astype(np.uint8)[::-1]      # la ligne du haut d'abord
    hauteur, largeur = img.shape[:2]
    donnees = base64.b64encode(_png(largeur, hauteur, img.tobytes())).decode("ascii")
    pa, pb = d.a / f, d.b / f                                    # un point de l'image est centré sur sa valeur
    x, y = v.X(d.x0 - pa / 2), v.Y(d.y0 + d.ny * d.b + pb / 2)
    return (f'<image x="{x:.1f}" y="{y:.1f}" width="{largeur * pa * ECH:.1f}" height="{hauteur * pb * ECH:.1f}" preserveAspectRatio="none" '
            f'clip-path="url(#{ident}_dalle)" href="data:image/png;base64,{donnees}"/>')


def _carte_vecteur(v: _Vue, champ: Dict) -> str:
    """La carte en RECTANGLES (la note PDF, qui ne peint pas d'image) : un élément du maillage par case, à la moyenne de
    ses quatre nœuds, en paliers ; les cases voisines de même palier sont réunies."""
    d = champ["dalle"]
    rampe, unite = _echelle(champ)
    if champ.get("par_element"):
        val, dec = np.nan_to_num(np.asarray(champ["valeurs"], float), nan=0.0) / unite, np.zeros(len(d.I), bool)
    else:
        val = champ["valeurs"][d.noeuds].mean(axis=1) / unite
        dec = champ["decolle"][d.noeuds].mean(axis=1) > 0.5
    haut = max(x for x, _ in rampe)
    palier = np.minimum(np.floor(np.clip(val, 0.0, haut) / haut * NIVEAUX_VECTEUR), NIVEAUX_VECTEUR - 1).astype(int)   # (haut = 1)
    palier = np.where(dec, -1, palier)
    couleurs = _couleurs((np.arange(NIVEAUX_VECTEUR) + 0.5) / NIVEAUX_VECTEUR * haut, rampe)
    out, a, b = [], d.a * ECH, d.b * ECH
    ordre = np.lexsort((d.I, d.J))
    I, J, P = d.I[ordre], d.J[ordre], palier[ordre]
    k = 0
    while k < len(I):
        k2 = k
        while k2 + 1 < len(I) and J[k2 + 1] == J[k] and I[k2 + 1] == I[k2] + 1 and P[k2 + 1] == P[k]:
            k2 += 1
        c = DECOLLE if P[k] < 0 else tuple(int(round(x)) for x in couleurs[P[k]])
        x, y = v.X(d.x0 + I[k] * d.a), v.Y(d.y0 + (J[k] + 1) * d.b)
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{(k2 - k + 1) * a + 0.3:.1f}" height="{b + 0.3:.1f}" fill="#{c[0]:02x}{c[1]:02x}{c[2]:02x}"/>')
        k = k2 + 1
    return "".join(out)


def _legende(v: _Vue, champ: Dict, x: float, y: float, hauteur: float) -> str:
    """L'échelle des couleurs, à droite du plan : la barre, ses bornes, la limite (contrainte), la zone décollée."""
    rampe, unite = _echelle(champ)
    t = v.t
    haut = max(xr for xr, _ in rampe)
    n = 32
    pas = hauteur / n
    cs = _couleurs((np.arange(n) + 0.5) / n * haut, rampe)
    out = "".join(f'<rect x="{x:.1f}" y="{y + hauteur - (i + 1) * pas:.1f}" width="{t * 0.9:.1f}" height="{pas + 0.4:.1f}" '
                  f'fill="#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"/>' for i, c in enumerate(cs))
    out += f'<rect x="{x:.1f}" y="{y:.1f}" width="{t * 0.9:.1f}" height="{hauteur:.1f}" fill="none" stroke="{SCHEMA}" stroke-width="{t * 0.05:.2f}"/>'
    sym = (champ.get("unite") or ("kPa", 1.0, 1))[0]
    reperes = [(0.0, "0", False), (1.0, _valeur(champ, unite, False) if champ["mode"] != "taux" else _valeur(champ, unite), False)]
    lim = champ.get("limite")
    if lim:
        r_l = float(lim) / unite
        # (la valeur du haut s'efface quand la limite la touche : deux textes l'un sur l'autre ne se lisent pas)
        if abs(1.0 - r_l) * hauteur < t * 0.9:
            reperes = reperes[:1]
        reperes.append((r_l, (_valeur(champ, float(lim), False) if champ["mode"] != "taux" else _valeur(champ, float(lim))) + " limite", True))
    for r, txt, noir in reperes:
        yy = y + hauteur - r * hauteur
        if noir:
            # la limite : un petit trait NOIR, épais, en travers de la barre (§ 2.3)
            out += (f'<line class="sf-lim" x1="{x - t * 0.25:.1f}" y1="{yy:.1f}" x2="{x + t * 1.15:.1f}" y2="{yy:.1f}" stroke="{LIMITE}" '
                    f'stroke-width="{t * 0.16:.2f}"/>' + _texte(x + t * 1.35, yy + t * 0.33, txt, t * 0.82, LIMITE, "start", gras=True))
        else:
            out += (f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x + t * 1.15:.1f}" y2="{yy:.1f}" stroke="{ENCRE}" stroke-width="{t * 0.06:.2f}"/>'
                    + _texte(x + t * 1.35, yy + t * 0.33, txt, t * 0.82, ENCRE, "start"))
    # (le titre au-dessus de la valeur du haut de l'échelle, sans la toucher : à 0,55 t, les parenthèses de « (kPa) »
    # mordaient de 1 px sur « 61,3 » — vu au harnais, tablette, 14/11/2026)
    titre = champ.get("symbole") or ("σ" if champ["mode"] == "sigma" else "w")
    out += _texte(x, y - t * 0.95, titre + ("" if champ["mode"] == "taux" else f" ({sym})"), t * 0.82, ENCRE, "start", gras=True)
    if champ.get("decolle") is not None and bool(np.any(np.where(champ["dalle"].on0, champ["decolle"], False))):
        yy = y + hauteur + t * 0.7
        out += (f'<rect x="{x:.1f}" y="{yy:.1f}" width="{t * 0.9:.1f}" height="{t * 0.9:.1f}" fill="#{DECOLLE[0]:02x}{DECOLLE[1]:02x}{DECOLLE[2]:02x}" '
                f'stroke="{SCHEMA}" stroke-width="{t * 0.05:.2f}"/>' + _texte(x + t * 1.35, yy + t * 0.72, "décollé", t * 0.82, ENCRE, "start"))
    return out


# ------------------------------------------------------------------ les accroches du châssis, d'un sommet
def _accroches_chassis(el: Dict, m: Dict) -> Tuple[List[float], List[float]]:
    """Où le centre du châssis s'accroche, en cotes (e_x, e_y) par rapport au centre de la dalle : le centre et les axes
    de la dalle (0), les sommets et les milieux des bords, et — dalle rectangulaire — « une plaque à la distance mini »
    de chaque bord."""
    c = m["contour"]
    x0, y0, x1, y1 = G.boite(c)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = {0.0}, {0.0}
    for i, p in enumerate(c):
        q = c[(i + 1) % len(c)]
        for x, y in (p, ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)):
            ax.add(round(x - cx, 4))
            ay.add(round(y - cy, 4))
    if el["dalle"]["forme"] == "rectangle":
        g = el["grue"]
        px0, py0, px1, py1 = G.boite([p for pl in m["plaques"] for p in pl])
        mx, my = m["centre"]
        ax.update((round(x0 + g["dmin"] + (mx - px0) - cx, 4), round(x1 - g["dmin"] - (px1 - mx) - cx, 4)))
        ay.update((round(y0 + g["dmin"] + (my - py0) - cy, 4), round(y1 - g["dmin"] - (py1 - my) - cy, 4)))
    return sorted(ax), sorted(ay)


def _angles_bords(contour: Sequence) -> List[float]:
    """Les angles (0 à 90° exclu) auxquels le châssis est parallèle à un bord de la dalle."""
    out = set()
    for i, p in enumerate(contour):
        q = contour[(i + 1) % len(contour)]
        a = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 90.0
        out.add(round(a, 2) % 90.0)
    return sorted(out)


def _plus_proche_bord(contour: Sequence, poly: Sequence) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """(le sommet de la plaque le plus près d'un bord, le point de ce bord en face, la distance)."""
    meilleur = None
    for p in poly:
        for i, a in enumerate(contour):
            b = contour[(i + 1) % len(contour)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
            q = (a[0] + t * dx, a[1] + t * dy)
            dist = math.hypot(p[0] - q[0], p[1] - q[1])
            if meilleur is None or dist < meilleur[2] - 1e-9:
                meilleur = (p, q, dist)
    return meilleur


# ------------------------------------------------------------------ le plan
def plan(el: Dict, m: Dict, champ: Optional[Dict] = None, interactif: bool = True, carte: str = "image", ident: str = "saf",
         legende: bool = False, armatures: Optional[Dict] = None, texte: float = 1.0) -> str:
    """Le plan de la fondation (voir l'en-tête du fichier). ``interactif`` faux : le plan de la note PDF, sans poignée ;
    ``carte`` : « image » (l'écran) ou « vecteur » (la note) ; ``legende`` : la place de la légende est GARDÉE même sans
    carte à peindre (l'écran, quand le calcul est impossible : la dalle ne saute pas de côté à l'instant où une surface
    d'appui sort de la dalle — on veut la ramener, pas la chercher) ; ``armatures`` : la carte Armatures (lot 5, § 5.1) —
    {"ef", "fer", "face"} : les zones rouges et les renforts de la face, et leur légende (dessin_armatures) ; ``texte`` : le
    facteur de la taille du texte (la note le grossit pour que ses dessins se lisent imprimés : 7 pt au moins, lot 5 § 10.3)."""
    c, g, ctl = m["contour"], el["grue"], m["controles"]
    plots = m["plots"]
    v = _Vue(c, list(m["plaques"]) + (plots["polys"] if plots else []))
    v.t0 = v.t                     # (l'unité des traits : elle ne grossit pas avec le texte)
    v.t *= texte
    t = v.t
    rect = el["dalle"]["forme"] == "rectangle"
    cx, cy = m["centre"]
    bx, by = (v.x0 + v.x1) / 2, (v.y0 + v.y1) / 2                # le centre de la boîte de la dalle
    out: List[str] = []
    # ---- la dalle, la carte
    sec = ' data-plan-sec="dalle"' if interactif else ""
    out.append(f'<defs><clipPath id="{ident}_dalle"><polygon points="{v.pts(c)}"/></clipPath></defs>')
    out.append(f'<polygon class="sf-dalle sf-forme" points="{v.pts(c)}" fill="{BETON}" stroke="none"{sec}/>')
    if champ is not None:
        out.append(f'<g class="sf-carte" pointer-events="none">{_carte_image(v, champ, ident) if carte == "image" else _carte_vecteur(v, champ)}</g>')
        out.append(_iso(v, champ, ident))
    if armatures is not None:
        from . import dessin_armatures as DA
        out.append(DA.couche(v, el, armatures["ef"], armatures["fer"], armatures["face"], ident, _texte, armatures.get("court", False)))
    out.append(f'<polygon class="sf-forme" points="{v.pts(c)}" fill="none" stroke="{BETON_BORD}" stroke-width="{t * 0.11:.2f}" stroke-linejoin="round" pointer-events="none"/>')
    if interactif and rect:
        # un rectangle : son bord droit et son bord haut se tirent (L_x, L_y) — SOUS le châssis : une plaque posée au bord
        # reste saisissable ; un appui bref sur un bord le prend pour repère des cotes du châssis
        tirer = {1: ("dalle/Lx", v.x1 - v.x0, "L_x", [1, 2], "h"), 2: ("dalle/Ly", v.y1 - v.y0, "L_y", [2, 3], "v")}
        for i, p in enumerate(c):
            q = c[(i + 1) % 4]
            clic = f' data-plan-clic="{_attr({"chemin": "grue/repere", "valeur": i})}"'
            trait = (f'x1="{v.X(p[0]):.1f}" y1="{v.Y(p[1]):.1f}" x2="{v.X(q[0]):.1f}" y2="{v.Y(q[1]):.1f}" stroke="{ENCRE}" stroke-opacity="0" '
                     f'stroke-width="{t * 0.8:.1f}"')
            if i in tirer:
                chemin, val, nom, suit, sens = tirer[i]
                d = {"chemin": chemin, "ech": ECH, "pas": 0.05, "ox": 0.0, "oy": 0.0, "noms": [nom], "unite": "m", "min": 1.0, "max": 60.0,
                     "suit": {"sel": ".sf-forme", "i": suit}}
                d["x" if sens == "h" else "y"] = round(val, 4)
                out.append(f'<line class="sf-tirer {sens}" {trait} data-plan-glisser="{_attr(d)}"{clic}><title>Tirer ce bord : {nom.replace("_", "")} — '
                           f'le toucher : repère des cotes du châssis</title></line>')
            else:
                out.append(f'<line class="sf-bord sf-repere" {trait}{clic}><title>Toucher ce bord : repère des cotes du châssis</title></line>')
    cb = PR.cotes_au_bord(el, m)
    if cb is not None:
        out.append(f'<line class="sf-bord-choisi" x1="{v.X(cb["a"][0]):.1f}" y1="{v.Y(cb["a"][1]):.1f}" x2="{v.X(cb["b"][0]):.1f}" y2="{v.Y(cb["b"][1]):.1f}" '
                   f'stroke="{COTE}" stroke-width="{t * 0.26:.2f}" stroke-linecap="round" pointer-events="none"/>')
    # ---- le repère : l'origine au coin bas-gauche de la boîte, les axes de la dalle
    ox, oy = v.X(v.x0) - t * 1.9, v.Y(v.y0) + t * 1.9
    out.append(f'<g pointer-events="none"><circle cx="{v.X(v.x0):.1f}" cy="{v.Y(v.y0):.1f}" r="{t * 0.2:.1f}" fill="{ENCRE}"/>'
               f'<path d="M{ox:.1f} {oy:.1f}h{t * 1.3:.1f}M{ox:.1f} {oy:.1f}v{-t * 1.3:.1f}" stroke="{ENCRE}" stroke-width="{t * 0.08:.2f}" fill="none"/>'
               f'<path d="M{ox + t * 1.3:.1f} {oy:.1f}l{-t * 0.42:.1f} {-t * 0.18:.1f}v{t * 0.36:.1f}zM{ox:.1f} {oy - t * 1.3:.1f}l{-t * 0.18:.1f} {t * 0.42:.1f}h{t * 0.36:.1f}z" fill="{ENCRE}"/>'
               + _texte(ox + t * 1.45, oy + t * 0.28, "x", t * 0.78, ENCRE, "start", halo=False) + _texte(ox, oy - t * 1.45, "y", t * 0.78, ENCRE, "middle", halo=False)
               + f'<line x1="{v.X(bx):.1f}" y1="{v.Y(v.y1) - t:.1f}" x2="{v.X(bx):.1f}" y2="{v.Y(v.y0) + t:.1f}" stroke="{SCHEMA}" stroke-width="{t * 0.045:.2f}" stroke-dasharray="{t * 0.9:.1f} {t * 0.3:.1f} {t * 0.12:.1f} {t * 0.3:.1f}"/>'
               f'<line x1="{v.X(v.x0) - t:.1f}" y1="{v.Y(by):.1f}" x2="{v.X(v.x1) + t:.1f}" y2="{v.Y(by):.1f}" stroke="{SCHEMA}" stroke-width="{t * 0.045:.2f}" stroke-dasharray="{t * 0.9:.1f} {t * 0.3:.1f} {t * 0.12:.1f} {t * 0.3:.1f}"/></g>')
    # ---- les bandes d'intégration : la largeur (voile), l'axe, le nom
    sec = ' data-plan-sec="bandes"' if interactif else ""
    bandes = []
    for b in m["bandes"]:
        (x1, y1), (x2, y2) = b["debut"], b["fin"]
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-9:
            continue
        nx, ny = -(y2 - y1) / L * b["largeur"] / 2, (x2 - x1) / L * b["largeur"] / 2
        quad = [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]
        bandes.append(f'<polygon points="{v.pts(quad)}" fill="{BANDE}" fill-opacity="0.07" stroke="{BANDE}" stroke-width="{t * 0.05:.2f}" '
                      f'stroke-dasharray="{t * 0.35:.1f} {t * 0.25:.1f}" clip-path="url(#{ident}_dalle)"{sec}/>'
                      f'<line x1="{v.X(x1):.1f}" y1="{v.Y(y1):.1f}" x2="{v.X(x2):.1f}" y2="{v.Y(y2):.1f}" stroke="{BANDE}" stroke-width="{t * 0.07:.2f}" '
                      f'stroke-dasharray="{t * 0.8:.1f} {t * 0.22:.1f} {t * 0.12:.1f} {t * 0.22:.1f}" pointer-events="none"/>')
        # le nom : au bout GAUCHE de la bande (en bas pour une bande verticale), juste hors de la dalle, le long de son axe —
        # jamais du côté de la légende des couleurs, à droite du plan (un châssis tourné de 90° inversait le sens des
        # bandes : « BY2 » passait sous la barre de la légende, vu sur une capture du harnais) ; une bande en diagonale
        # dont le bout gauche est sur le bord du haut y garde son nom (loin de la légende)
        verticale = abs(x2 - x1) <= 0.1 * abs(y2 - y1)
        if (not verticale and x2 < x1) or (verticale and y2 < y1):
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        ang = _lisible(-math.degrees(math.atan2(uy, ux)))
        px, py = v.X(x1) - ux * t * 0.45, v.Y(y1) + uy * t * 0.45
        vers_droite = (ux * math.cos(math.radians(ang)) - uy * math.sin(math.radians(ang))) > 0
        bandes.append(_texte(px + math.sin(math.radians(ang)) * t * -0.3, py + math.cos(math.radians(ang)) * t * 0.3, b["nom"], t * 0.78, BANDE,
                             "end" if vers_droite else "start", gras=True, angle=ang))
    out.append(f'<g class="sf-bandes">{"".join(bandes)}</g>')
    # ---- le châssis : tout ce qui le suit quand on le déplace (les plots, les axes d'appui, les surfaces, les repères)
    ch: List[str] = []
    if plots:
        mauvais = [not (ctl["plots_dedans"][i] and ctl["plots_couvrent"][i]) or not ctl["plots_separes"] for i in range(4)]
        for i, q in enumerate(plots["polys"]):
            coul, fond = (ROUGE, ROUGE_FOND) if mauvais[i] else (PLOT, PLOT_FOND)
            ch.append(f'<polygon class="sf-plot" points="{v.pts(q)}" fill="{fond}" fill-opacity="0.55" stroke="{coul}" stroke-width="{t * 0.08:.2f}" stroke-linejoin="round"/>')
    pieds = m["pieds"]
    for i in range(4):                                           # les axes d'appui : trait mixte rouge, prolongé
        (x1, y1), (x2, y2) = pieds[i], pieds[(i + 1) % 4]
        L = math.hypot(x2 - x1, y2 - y1) or 1.0
        ex_, ey_ = (x2 - x1) / L * 0.45, (y2 - y1) / L * 0.45
        ch.append(f'<line x1="{v.X(x1 - ex_):.1f}" y1="{v.Y(y1 - ey_):.1f}" x2="{v.X(x2 + ex_):.1f}" y2="{v.Y(y2 + ey_):.1f}" stroke="{AXE}" '
                  f'stroke-width="{t * 0.06:.2f}" stroke-dasharray="{t * 0.9:.1f} {t * 0.22:.1f} {t * 0.13:.1f} {t * 0.22:.1f}"/>')
    for i, pl in enumerate(m["plaques"]):
        bon = ctl["plaques_dedans"][i] and ctl["distances"][i] >= g["dmin"] - G.TOLERANCE
        coul, fond = (APPUI, APPUI_FOND) if bon else (ROUGE, ROUGE_FOND)
        ch.append(f'<polygon class="sf-appui" points="{v.pts(pl)}" fill="{fond}" stroke="{coul}" stroke-width="{t * 0.11:.2f}" stroke-linejoin="round"/>')
    # les numéros P1 à P4 : vers l'intérieur du châssis (les distances au bord s'écrivent à l'extérieur)
    for i, ctr in enumerate(m["centres"]):
        dx, dy = cx - ctr[0], cy - ctr[1]
        L = math.hypot(dx, dy) or 1.0
        r = max(math.hypot(p[0] - ctr[0], p[1] - ctr[1]) for p in m["plaques"][i]) + 1.0 * t / ECH
        ch.append(_texte(v.X(ctr[0] + dx / L * r), v.Y(ctr[1] + dy / L * r) + t * 0.33, f"P{i + 1}", t * 0.95, APPUI, "middle", gras=True))
    # le centre du châssis (une croix) et la poignée de rotation, sur son axe x
    ch.append(f'<path d="M{v.X(cx) - t * 0.5:.1f} {v.Y(cy):.1f}h{t:.1f}M{v.X(cx):.1f} {v.Y(cy) - t * 0.5:.1f}v{t:.1f}" stroke="{AXE}" stroke-width="{t * 0.09:.2f}" fill="none"/>')
    rb = math.radians(g["beta"])
    rp = min(m["chassis"]["a"], m["chassis"]["b"]) * 0.28
    hx_, hy_ = cx + rp * math.cos(rb), cy + rp * math.sin(rb)
    if interactif:
        ch.append(f'<line x1="{v.X(cx):.1f}" y1="{v.Y(cy):.1f}" x2="{v.X(hx_):.1f}" y2="{v.Y(hy_):.1f}" stroke="{AXE}" stroke-width="{t * 0.05:.2f}"/>')
        tourner = {"chemin": "grue/beta", "cx": round(v.X(cx), 2), "cy": round(v.Y(cy), 2), "angle": g["beta"], "cible": ".sf-chassis",
                   "accroches": sorted({0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0}
                                       | {a + q for a in _angles_bords(c) for q in (0.0, 90.0, 180.0, 270.0)})}
        ch.append(f'<circle class="sf-poignee" cx="{v.X(hx_):.1f}" cy="{v.Y(hy_):.1f}" r="{t * 0.5:.1f}" fill="#ffffff" stroke="{AXE}" '
                  f'stroke-width="{t * 0.1:.2f}" data-plan-tourner="{_attr(tourner)}"><title>Tourner le châssis</title></circle>')
    ax_, ay_ = _accroches_chassis(el, m)
    glisser = {"chemin": "grue/position", "x": round(g["ex"], 4), "y": round(g["ey"], 4), "ech": ECH, "pas": 0.05, "ox": round(bx - v.x0, 4),
               "oy": round(by - v.y0, 4), "ax": ax_, "ay": ay_, "noms": ["e_x", "e_y"], "unite": "m"}
    attrs = f' data-plan-glisser="{_attr(glisser)}" data-plan-sec="grue"' if interactif else ""
    out.append(f'<g class="sf-chassis"{attrs}>{"".join(ch)}</g>')
    # ---- les cotes
    cotes: List[str] = []
    clic = (lambda chemin, valeur, unite="m": f' class="sf-clic" data-plan-cote="{_attr({"chemin": chemin, "valeur": valeur, "unite": unite})}"') \
        if interactif else (lambda *a, **k: "")
    bas, gauche = v.Y(min(v.gy0, v.y0)), v.X(min(v.gx0, v.x0))
    dec_bas = -(bas - v.Y(v.y0) + t * 2.6)
    dec_gauche = v.X(v.x0) - gauche + t * 2.6
    Lx, Ly = v.x1 - v.x0, v.y1 - v.y0
    cotes.append(_cote(v, (v.x0, v.y0), (v.x1, v.y0), dec_bas, _n(Lx), attrs=clic("dalle/Lx", Lx) if rect else ""))
    cotes.append(_cote(v, (v.x0, v.y0), (v.x0, v.y1), dec_gauche, _n(Ly), attrs=clic("dalle/Ly", Ly) if rect else ""))
    # la position du châssis : deux cotes entre son centre et le repère — les axes de la dalle, ou le bord choisi (la
    # distance au bord, et la position le long du bord depuis son début)
    if cb is not None:
        cotes.append(_cote(v, cb["pied"], (cx, cy), 0.0, _n(abs(cb["perp"])), attrs=clic("grue/cote/perp", cb["perp"])))
        cotes.append(_cote(v, cb["a"], cb["pied"], -t * 1.3, _n(abs(cb["long"])), attrs=clic("grue/cote/long", cb["long"])))
    else:
        if abs(g["ex"]) > 5e-4:
            cotes.append(_cote(v, (bx, cy), (cx, cy), 0.0, _n(abs(g["ex"])), attrs=clic("grue/ex", g["ex"])))
        if abs(g["ey"]) > 5e-4:
            cotes.append(_cote(v, (cx, by), (cx, cy), 0.0, _n(abs(g["ey"])), attrs=clic("grue/ey", g["ey"])))
    if abs(g["beta"] % 360.0) > 0.005:
        cotes.append(_texte(v.X(hx_) + t * 0.9, v.Y(hy_) - t * 0.5, f"β = {C.nb(round(g['beta'], 2), 2)}°", t * 0.85, AXE, "start",
                            attrs=clic("grue/beta", g["beta"], "°")))
    # les distances plaque–bord (rouge sous le minimum)
    for i, pl in enumerate(m["plaques"]):
        if not ctl["plaques_dedans"][i]:
            continue
        p, q, dist = _plus_proche_bord(c, pl)
        if dist < 1e-3:
            continue
        cotes.append(_cote(v, q, p, 0.0, _n(dist), ROUGE if dist < g["dmin"] - G.TOLERANCE else COTE, rappel=False))
    # les plots : leurs dimensions, écrites sous le plot de P4
    if plots:
        q = plots["polys"][3]
        bas_plot = min(q, key=lambda p: (p[1], p[0]))
        cotes.append(_texte(v.X(sum(p[0] for p in q) / 4), v.Y(bas_plot[1]) + t * 1.15,
                            f"plot {_n(plots['a'])} × {_n(plots['b'])} × {_n(plots['hauteur'])}", t * 0.8, PLOT, "middle"))
    out.append(f'<g class="sf-cotes">{"".join(cotes)}</g>')
    # ---- le contour : ses bords (un double-clic y ajoute un sommet) et ses sommets (ils se déplacent)
    if interactif and not rect:
        n = len(c)
        poignees = []
        for i, p in enumerate(c):
            q = c[(i + 1) % n]
            poignees.append(f'<line class="sf-bord" x1="{v.X(p[0]):.1f}" y1="{v.Y(p[1]):.1f}" x2="{v.X(q[0]):.1f}" y2="{v.Y(q[1]):.1f}" stroke="{ENCRE}" '
                            f'stroke-opacity="0" stroke-width="{t * 0.9:.1f}" data-plan-bord="{i}" data-plan-clic="{_attr({"chemin": "grue/repere", "valeur": i})}">'
                            f'<title>Toucher ce bord : repère des cotes du châssis — deux fois : un sommet de plus</title></line>')
        for i, p in enumerate(c):
            avant, apres = c[(i - 1) % n], c[(i + 1) % n]
            d = {"chemin": f"dalle/sommet/{i}", "x": round(p[0], 4), "y": round(p[1], 4), "ech": ECH, "pas": 0.05, "ox": 0.0, "oy": 0.0,
                 "ax": sorted({round(avant[0], 4), round(apres[0], 4)}), "ay": sorted({round(avant[1], 4), round(apres[1], 4)}),
                 "noms": ["x", "y"], "unite": "m", "double": f"dalle/sommets/{i}/-", "suit": {"sel": ".sf-forme", "i": [i]}}
            poignees.append(f'<circle class="sf-sommet" cx="{v.X(p[0]):.1f}" cy="{v.Y(p[1]):.1f}" r="{t * 0.42:.1f}" fill="#ffffff" stroke="{ENCRE}" '
                            f'stroke-width="{t * 0.1:.2f}" data-plan-glisser="{_attr(d)}"><title>Sommet {i + 1} : le glisser pour le déplacer</title></circle>')
        out.append(f'<g class="sf-contour">{"".join(poignees)}</g>')
    if armatures is not None:
        from . import dessin_armatures as DA
        out.append(DA.reperes(v, armatures["fer"], armatures["face"], _texte, armatures.get("court", False)))
    # ---- la valeur maximale pointée, l'échelle des couleurs
    droite = v.X(max(v.gx1, v.x1))
    if champ is not None:
        px, py = v.X(champ["x"]), v.Y(champ["y"])
        sx, sy = (-1.0 if px > v.X(bx) else 1.0), (1.0 if py < v.Y(by) else -1.0)
        qx, qy = px + sx * t * 1.5, py + sy * t * 1.5
        out.append(f'<g class="sf-max" pointer-events="none"><line x1="{px:.1f}" y1="{py:.1f}" x2="{qx:.1f}" y2="{qy:.1f}" stroke="{ENCRE}" stroke-width="{t * 0.08:.2f}"/>'
                   f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{t * 0.3:.1f}" fill="#ffffff" stroke="{ENCRE}" stroke-width="{t * 0.11:.2f}"/>'
                   + _texte(qx + sx * t * 0.25, qy + (t * 0.85 if sy > 0 else -t * 0.2),
                            f"{champ.get('symbole') or ('σ' if champ['mode'] == 'sigma' else 'w')} {champ.get('mot', 'max')} {_valeur(champ, champ['maxi'])}",
                            t * 0.9, ENCRE,
                            "end" if sx < 0 else "start", gras=True) + "</g>")
        out.append(f'<g pointer-events="none">{_legende(v, champ, droite + t * 1.6, v.Y(v.y1) + t * 1.2, min(v.Y(v.y0) - v.Y(v.y1) - t * 4.0, t * 16))}</g>')
    elif armatures is not None:
        from . import dessin_armatures as DA
        out.append(DA.legende(v, el, armatures["fer"], armatures["face"], droite + t * 1.6, v.Y(v.y1) + t * 1.2, _texte))
    # ---- la feuille
    x_min = min(gauche, v.X(v.x0)) - t * 5.4
    y_min = v.Y(max(v.gy1, v.y1)) - t * 2.6
    x_max = droite + (t * 9.5 if champ is not None or legende or armatures is not None else t * 3.0)
    y_max = max(bas, v.Y(v.y0)) + t * 4.4
    fond = (f'<rect class="sf-fond" x="{x_min:.1f}" y="{y_min:.1f}" width="{x_max - x_min:.1f}" height="{y_max - y_min:.1f}" fill="#ffffff" fill-opacity="0" '
            f'data-plan-fond="{_attr({"chemin": "dalle/sommet/+", "ech": ECH, "pas": 0.05})}"/>') if interactif and not rect else ""
    origine = {"x": round(v.X(v.x0), 2), "y": round(v.Y(v.y0), 2)}
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x_min:.1f} {y_min:.1f} {x_max - x_min:.1f} {y_max - y_min:.1f}" role="img" '
            f'aria-label="Plan de la fondation" class="sf-plan" data-plan="{_attr({"ech": ECH, "origine": origine})}">'
            f'{fond}{"".join(out)}</svg>')


def numerotation() -> str:
    """Le petit schéma de numérotation des pieds (rubrique « Réactions ») : P1 haut-gauche, P2 haut-droite, P3
    bas-droite, P4 bas-gauche — sens horaire vu du dessus, châssis non tourné."""
    pts = ((14, 12, "P1", "end", -4, -3), (66, 12, "P2", "start", 4, -3), (66, 48, "P3", "start", 4, 9), (14, 48, "P4", "end", -4, 9))
    carres = "".join(f'<rect x="{x - 5}" y="{y - 5}" width="10" height="10" fill="{APPUI_FOND}" stroke="{APPUI}" stroke-width="1.2"/>'
                     f'<text x="{x + dx}" y="{y + dy}" font-size="9" font-weight="600" font-family="{POLICE}" fill="{APPUI}" text-anchor="{a}">{n}</text>'
                     for x, y, n, a, dx, dy in pts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-14 0 108 60" class="sf-num" role="img" aria-label="Numérotation des pieds : P1 en haut à gauche, '
            f'puis dans le sens horaire"><rect x="14" y="12" width="52" height="36" fill="none" stroke="{AXE}" stroke-width="0.8" stroke-dasharray="7 2 1.5 2"/>'
            f'<path d="M40 30m-9 0a9 9 0 1 1 9 9" fill="none" stroke="{SCHEMA}" stroke-width="1"/><path d="M40 39l3.2-2.6-0.6 4.4z" fill="{SCHEMA}"/>{carres}</svg>')

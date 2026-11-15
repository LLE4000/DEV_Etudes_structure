# -*- coding: utf-8 -*-
"""La carte ARMATURES du plan (lot 5, cahier des charges du 15/11/2026, § 5.1) — et les deux plans du principe de la note
(§ 10.5) : sur le plan de la fondation (``dessin_saf.plan``), pour la face choisie (Bas, Haut), les zones rouges (là où le
treillis ne suffit pas, hors des bandes), les renforts des bandes (leur contour, leurs barres, leur repère « R1 · 4 Ø12 »)
et les renforts locaux (contour en tirets, leurs barres, leur repère), et une petite légende. Rien d'autre.

Ce fichier ne calcule rien : il dessine le principe d'armatures (``saf.fondation_grue.armatures.calculer``)."""
import math
from typing import Dict, List

import numpy as np

from . import commun as C
from .dessin_saf import ECH

ZONE = "#ef4444"                 # la zone rouge (treillis insuffisant), en voile
BANDE_R = "#0f766e"              # un renfort de bande
LOCAL_R = "#9a3412"              # un renfort local
BARRE = "#1d2b3a"


def _lignes_barres(v, coins, n: int, couleur: str, t: float) -> str:
    """Les n barres d'un renfort, réparties sur sa largeur, le long de sa longueur (coins : début gauche, fin gauche,
    fin droite, début droite)."""
    if n <= 0:
        return ""
    a, b, c, d = (np.asarray(p, float) for p in coins)
    out = []
    for k in range(n):
        f = (k + 0.5) / n
        p, q = a + (d - a) * f, b + (c - b) * f
        out.append(f'<line x1="{v.X(p[0]):.1f}" y1="{v.Y(p[1]):.1f}" x2="{v.X(q[0]):.1f}" y2="{v.Y(q[1]):.1f}" stroke="{couleur}" '
                   f'stroke-width="{t * 0.07:.2f}" stroke-linecap="round"/>')
    return "".join(out)


def _coins(r: Dict):
    """Les coins d'un renfort, début gauche → fin gauche → fin droite → début droite, le long de ses barres."""
    if r.get("coins"):
        return r["coins"]
    x0, x1, y0, y1 = r["x0"], r["x1"], r["y0"], r["y1"]
    if r["sens"] == "x":
        return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
    return [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]


def zones(v, ef: Dict, fer: Dict, face: str) -> str:
    """Les zones rouges de la face : une case par nœud rouge (x ou y), une demi-maille autour du nœud ; les cases voisines
    d'une ligne réunies."""
    g = ef["grille"]
    rouge = np.zeros((g["ny"], g["nx"]), bool)
    for s in ("x", "y"):
        m = fer["rouge"].get(f"{face}_{s}")
        if m is not None:
            rouge |= m
    out = []
    a, b = g["a"] * ECH, g["b"] * ECH
    for j in range(g["ny"]):
        i = 0
        while i < g["nx"]:
            if rouge[j, i]:
                k = i
                while k + 1 < g["nx"] and rouge[j, k + 1]:
                    k += 1
                x = v.X(g["x0"] + (i - 0.5) * g["a"])
                y = v.Y(g["y0"] + (j + 0.5) * g["b"])
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{(k - i + 1) * a:.1f}" height="{b:.1f}" fill="{ZONE}" fill-opacity="0.38"/>')
                i = k + 1
            else:
                i += 1
    return f'<g class="sf-rouge">{"".join(out)}</g>' if out else ""


def couche(v, el: Dict, ef: Dict, fer: Dict, face: str, ident: str, texte, court: bool = False) -> str:
    """La couche « Armatures » d'une face, à poser sur le plan (``v`` : le repère du plan ; ``texte`` : son écriture ;
    ``court`` : le repère seul, « R1 » — la note, dont le tableau des renforts dit les barres)."""
    t = v.t
    e = getattr(v, "t0", t)        # (les traits : l'unité du dessin, pas celle du texte grossi de la note)
    out = [f'<g clip-path="url(#{ident}_dalle)" pointer-events="none">{zones(v, ef, fer, face)}</g>']
    for r in fer["renforts"]:
        if r["face"] != face:
            continue
        cs = _coins(r)
        bande = r["type"] == "bande"
        coul = BANDE_R if bande else LOCAL_R
        tirets = "" if bande else f' stroke-dasharray="{e * 0.45:.1f} {e * 0.25:.1f}"'
        out.append(f'<g class="sf-renfort" pointer-events="none"><polygon points="{v.pts(cs)}" fill="{coul}" fill-opacity="0.10" stroke="{coul}" '
                   f'stroke-width="{e * 0.09:.2f}"{tirets}/>{"" if court and not bande else _lignes_barres(v, cs, r["n"], coul, e)}</g>')
    return f'<g class="sf-armatures">{"".join(out)}</g>'


def reperes(v, fer: Dict, face: str, texte, court: bool = False) -> str:
    """Les repères des renforts de la face (« R1 · 4 Ø12 » ; la note : « R1 », ceux des bandes), posés au-dessus du plan."""
    t = v.t
    out = []
    for r in fer["renforts"]:
        if r["face"] != face or (court and r["type"] != "bande"):
            continue
        cs = _coins(r)
        cx = sum(p[0] for p in cs) / 4
        cy = sum(p[1] for p in cs) / 4
        coul = BANDE_R if r["type"] == "bande" else LOCAL_R
        out.append(texte(v.X(cx), v.Y(cy) + t * 0.25, f"{r['rep']} · {r['n']} Ø{r['d']}" if r["n"] and not court else r["rep"], t * 0.62, coul, "middle", gras=True))
    return f'<g class="sf-reperes" pointer-events="none">{"".join(out)}</g>'


def legende(v, el: Dict, fer: Dict, face: str, x: float, y: float, texte) -> str:
    """La petite légende de la carte Armatures, à droite du plan."""
    t = v.t
    from . import saf_armatures as SA
    lignes = [(BANDE_R, "plein", "renfort de bande"), (LOCAL_R, "tirets", "renfort local"), (ZONE, "zone", "zone rouge")]
    out = [texte(x, y, f"{'Bas' if face == 'inf' else 'Haut'} : {SA.treillis_face(el, face).replace(C.NBSP, ' ')}", t * 0.78, BARRE, "start", gras=True)]
    for k, (coul, sorte, mot) in enumerate(lignes):
        yy = y + t * (1.35 + 1.25 * k)
        if sorte == "zone":
            out.append(f'<rect x="{x:.1f}" y="{yy - t * 0.62:.1f}" width="{t * 1.2:.1f}" height="{t * 0.75:.1f}" fill="{coul}" fill-opacity="0.38"/>')
        else:
            tirets = "" if sorte == "plein" else f' stroke-dasharray="{t * 0.35:.1f} {t * 0.2:.1f}"'
            out.append(f'<rect x="{x:.1f}" y="{yy - t * 0.62:.1f}" width="{t * 1.2:.1f}" height="{t * 0.75:.1f}" fill="{coul}" fill-opacity="0.10" '
                       f'stroke="{coul}" stroke-width="{t * 0.08:.2f}"{tirets}/>')
        out.append(texte(x + t * 1.5, yy, mot, t * 0.72, BARRE, "start"))
    return f'<g class="sf-arm-leg" pointer-events="none">{"".join(out)}</g>'

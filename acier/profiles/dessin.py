# -*- coding: utf-8 -*-
"""Dessin SVG d'une section : contour exact (celui du calcul, source
unique), axes y-y et z-z par le centre de gravité, cotes principales.
Sans Streamlit."""
from typing import Dict, Optional

import numpy as np

from . import familles as F

ENCRE, ACIER, COTE, AXE = "#1F2937", "#CBD5E1", "#9C3341", "#6E7480"


def _fmt(v: float) -> str:
    return f"{v:g}".replace(".", ",")


def svg_section(famille: str, dims: Dict[str, float], taille: int = 240, props: Optional[Dict] = None,
                titre: str = "") -> str:
    """SVG carré de ``taille`` px : la section à l'échelle, remplie, avec ses
    axes et les cotes h, b (ou d) et les épaisseurs en légende."""
    rings = F.contours(famille, dims)
    pts = np.vstack(rings)
    ymin, ymax = float(pts[:, 0].min()), float(pts[:, 0].max())
    zmin, zmax = float(pts[:, 1].min()), float(pts[:, 1].max())
    larg, haut = ymax - ymin, zmax - zmin
    marge = 34.0
    ech = (taille - 2 * marge) / max(larg, haut)
    cy = marge + (taille - 2 * marge - larg * ech) / 2
    cz = marge + (taille - 2 * marge - haut * ech) / 2

    def X(y):
        return cy + (y - ymin) * ech

    def Z(z):
        return taille - (cz + (z - zmin) * ech)

    chemins = []
    for ring in rings:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(p[0]):.2f},{Z(p[1]):.2f}" for i, p in enumerate(ring)) + " Z"
        chemins.append(d)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{taille}" height="{taille}" viewBox="0 0 {taille} {taille}" '
           f'font-family="Inter, Segoe UI, Arial, sans-serif" font-size="10">',
           f'<path d="{" ".join(chemins)}" fill="{ACIER}" stroke="{ENCRE}" stroke-width="1.2" fill-rule="evenodd"/>']
    # axes par le centre de gravité
    if props and props.get("yc") is not None:
        gy, gz = props["yc"], props["zc"]
    else:
        gy, gz = (ymin + ymax) / 2, (zmin + zmax) / 2
    out.append(f'<line x1="{marge - 14}" y1="{Z(gz):.1f}" x2="{taille - marge + 14}" y2="{Z(gz):.1f}" stroke="{AXE}" stroke-width="0.8" stroke-dasharray="6 3 1.5 3"/>')
    out.append(f'<line x1="{X(gy):.1f}" y1="{marge - 14}" x2="{X(gy):.1f}" y2="{taille - marge + 14}" stroke="{AXE}" stroke-width="0.8" stroke-dasharray="6 3 1.5 3"/>')
    out.append(f'<text x="{taille - marge + 16}" y="{Z(gz) + 3.5:.1f}" fill="{AXE}" font-size="9">y</text>')
    out.append(f'<text x="{X(gy) + 4:.1f}" y="{marge - 16}" fill="{AXE}" font-size="9">z</text>')
    # cotes : largeur en haut, hauteur à gauche
    yt = marge - 20
    out.append(f'<line x1="{X(ymin):.1f}" y1="{yt}" x2="{X(ymax):.1f}" y2="{yt}" stroke="{COTE}" stroke-width="0.8"/>')
    out.append(f'<line x1="{X(ymin):.1f}" y1="{yt - 4}" x2="{X(ymin):.1f}" y2="{yt + 4}" stroke="{COTE}" stroke-width="0.8"/>')
    out.append(f'<line x1="{X(ymax):.1f}" y1="{yt - 4}" x2="{X(ymax):.1f}" y2="{yt + 4}" stroke="{COTE}" stroke-width="0.8"/>')
    out.append(f'<text x="{(X(ymin) + X(ymax)) / 2:.1f}" y="{yt - 4}" text-anchor="middle" fill="{COTE}">{_fmt(larg)}</text>')
    xl = marge - 20
    out.append(f'<line x1="{xl}" y1="{Z(zmax):.1f}" x2="{xl}" y2="{Z(zmin):.1f}" stroke="{COTE}" stroke-width="0.8"/>')
    out.append(f'<line x1="{xl - 4}" y1="{Z(zmax):.1f}" x2="{xl + 4}" y2="{Z(zmax):.1f}" stroke="{COTE}" stroke-width="0.8"/>')
    out.append(f'<line x1="{xl - 4}" y1="{Z(zmin):.1f}" x2="{xl + 4}" y2="{Z(zmin):.1f}" stroke="{COTE}" stroke-width="0.8"/>')
    out.append(f'<text x="{xl - 5}" y="{(Z(zmin) + Z(zmax)) / 2:.1f}" text-anchor="middle" fill="{COTE}" '
               f'transform="rotate(-90 {xl - 5} {(Z(zmin) + Z(zmax)) / 2:.1f})">{_fmt(haut)}</text>')
    # légende des épaisseurs
    f = F.FAMILLES[famille]
    leg = " · ".join(f"{k} {_fmt(float(dims[k]))}" for k in f["dims"] if k not in ("h", "b", "d") and dims.get(k) is not None)
    if leg:
        out.append(f'<text x="{taille / 2:.1f}" y="{taille - 8}" text-anchor="middle" fill="{ENCRE}" font-size="9.5">{leg}</text>')
    if titre:
        out.append(f'<text x="{marge - 24}" y="{taille - 8}" fill="{ENCRE}" font-size="10" font-weight="600">{titre}</text>')
    out.append("</svg>")
    return "".join(out)

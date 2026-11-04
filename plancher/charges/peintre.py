# -*- coding: utf-8 -*-
"""Peintre du SVG des dessins (plan, coupe, balayage) sur un canevas
reportlab : la note PDF montre LE MÊME dessin que l'écran, sans seconde
implémentation (R2 — source unique).

Le peintre lui-même est partagé par toutes les notes (``ndc_pdf/svg.py``) ;
ce module ne garde que ce qui est propre au plancher : ses motifs de
hachures (``fill="url(#chp_hc)"`` et ``url(#chp_hs)``) peints en aplats
clairs, comme dans les notes publiées, et les noms que l'écran et les tests
importent ici (``Mat``, ``transformation``, ``peindre_svg``,
``largeur_texte``, ``POLICE``).
"""
from typing import Optional

from ndc_pdf.svg import Mat, transformation, largeur_texte, POLICE, POLICE_GRAS  # noqa: F401
from ndc_pdf.svg import peindre_svg as _peindre_svg

MOTIFS = {"url(#chp_hc)": "#D7DBE1", "url(#chp_hs)": "#E8DDF8"}     # hachures du plan : aplats clairs


def peindre_svg(c, svg: str, x: float, y: float, w: float, h: float, marge: float = 2.0,
                police: str = POLICE, police_gras: str = POLICE_GRAS, motifs: Optional[dict] = None) -> float:
    """Peint ``svg`` dans le rectangle ``(x, y, w, h)`` du canevas, à l'échelle,
    centré ; rend l'échelle (points par unité SVG). Les motifs du plancher
    sont des aplats clairs."""
    return _peindre_svg(c, svg, x, y, w, h, marge, police, police_gras, {**MOTIFS, **(motifs or {})})

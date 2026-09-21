# -*- coding: utf-8 -*-
"""Formatage des nombres, identique au HTML de référence.

- ``F(x, n)``  : ``toFixed(n)`` avec la virgule décimale ; « — » si non fini ;
- ``f0(x)``    : 0 ou 1 décimale (1 si la partie décimale dépasse 0,05), signe
                 typographique « − » ;
- ``pct(e, n)`` : pourcentage, « > 999 % » au-delà de 9,99, « — » si absent.

Ces trois fonctions écrivent les chaînes ``vals`` des vérifications, les
explications chiffrées des alertes, l'export texte et les cotes des dessins :
elles font partie de la parité.
"""
import math

from .js import to_fixed, js_round


def _fini(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def F(x, n=1):
    """``F(x, n)`` du moteur JS : nombre → chaîne à n décimales, virgule."""
    if x is None or not _fini(x):
        return "—"
    return to_fixed(x, n).replace(".", ",")


def f0(x):
    """``f0(x)`` : une décimale seulement si nécessaire, signe « − »."""
    if x is None or not _fini(x):
        return "—"
    n = 1 if abs(x - js_round(x)) > 0.05 else 0
    return F(x, n).replace("-", "−", 1)


def pct(e, n=0):
    """``pct(e, n)`` de l'interface : taux → « 35,4 % »."""
    if e is None or (isinstance(e, float) and math.isnan(e)):
        return "—"
    if e > 9.99:
        return "> 999 %"
    return F(e * 100, n or 0) + " %"

# -*- coding: utf-8 -*-
"""Formules et substitutions numériques des vérifications du plat d'âme.

Chaque vérification imprime : sa formule générale, ses valeurs introduites
(``vals`` composées par le moteur — jamais recalculées ici), puis la ligne
« η = Ed/Rd = … ». Un test recompose la ligne du taux et vérifie qu'elle
correspond à ``eta`` du moteur."""
from acier.formats import F, pct


def textes(R, c):
    """Les lignes imprimées pour la vérification ``c``."""
    nd = 3 if c.unit == "-" else 2
    un = "" if c.unit == "-" else " " + c.unit
    out = [c.formula]
    if c.vals:
        out.append(c.vals)
    out.append("η = " + F(c.Ed, nd) + un + " / " + F(c.Rd, nd) + un + " = " + pct(c.eta, 1)
               + (" ≤ 100 %" if c.ok else " > 100 %"))
    return out

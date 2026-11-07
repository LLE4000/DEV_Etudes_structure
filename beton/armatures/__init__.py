# -*- coding: utf-8 -*-
"""Tableau des armatures (sections d'acier par nombre de barres et par espacement, masses linéiques), sans Streamlit :
les valeurs de l'image « Tableau armature.png » que l'écran ``modules/tableau_armatures.py`` affiche, recalculées et
prouvées identiques au chiffre près (``tests/test_armatures_moteur.py`` relit l'image dans ses pixels).

    from beton.armatures import moteur
    res = moteur.calculer(moteur.nouveau_besoin())

- ``tableau`` : les lignes et colonnes de l'image, les formules (π·Ø²/4, ×n, ×100/e, ρ = 7 860 kg/m³), l'arrondi ;
- ``moteur``  : un besoin (section requise), la disposition retenue, la plus économique, le tableau marqué — et le
  FORMAT complet (entrée, sortie) dans sa docstring.
"""
from .moteur import (  # noqa: F401 — l'API
    ESPACEMENTS,
    calculer,
    candidats,
    disposition,
    normaliser,
    nouveau_besoin,
    optimale,
)

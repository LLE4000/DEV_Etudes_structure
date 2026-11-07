# -*- coding: utf-8 -*-
"""Enrobage du béton (EN 1992-1-1 §4.4.1 et EN 1992-1-2 méthode tabulée), sans Streamlit : le calcul extrait de l'écran
``modules/enrobage.py`` (VERSION 2.0), qui reste la référence de parité (``tests/test_enrobage_moteur.py``).

    from beton.enrobage import moteur
    res = moteur.calculer(moteur.nouvel_element())

- ``donnees`` : les tableaux, les listes et les textes de l'écran, recopiés tels quels ;
- ``entree``  : l'élément, ses défauts, sa normalisation stricte, la saisie d'un champ (effets de bord de l'écran) ;
- ``moteur``  : les fonctions de l'écran, l'enchaînement de son calcul, les termes nommés avec leur règle, les
  avertissements, le sélecteur de la largeur au feu — et le FORMAT de sortie dans sa docstring.
"""
from .moteur import (  # noqa: F401 — l'API
    a_feu,
    appliquer,
    c_durabilite,
    c_feu,
    calculer,
    candidats_feu,
    normaliser,
    nouvel_element,
    variantes,
)

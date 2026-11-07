# -*- coding: utf-8 -*-
"""Maturité du béton (résistance au jeune âge, EN 1992-1-1 §3.1.2 et annexe B), sans Streamlit : le calcul extrait de
l'écran ``modules/age_beton.py`` v3.2, qui reste la référence de parité (``tests/test_maturite_moteur.py``).

    from beton.maturite import moteur
    res = moteur.calculer(moteur.nouvel_element())

- ``ec2``     : les formules du module, recopiées (β_cc, f_ck(t), âge équivalent, seuil, classe minimale, essai) ;
- ``releve``  : le relevé de températures journalières (lecture du tableau saisi, synthèse comme le mode Météo) ;
- ``entree``  : les valeurs par défaut de l'écran, ses bornes, la normalisation stricte ;
- ``moteur``  : le calcul de tout ce que l'écran affiche — et le FORMAT complet (entrée, sortie) dans sa docstring.

Bibliothèques : la bibliothèque standard et numpy (présent dans le navigateur, Pyodide) ; ni streamlit, ni
matplotlib, ni requests, ni pandas.
"""
from .moteur import calculer, candidats_classes, normaliser, nouvel_element  # noqa: F401 — l'API

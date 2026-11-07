# -*- coding: utf-8 -*-
"""Dalle en béton armé (méthode du bureau), sans Streamlit : le calcul extrait de l'écran ``modules/dalle.py`` v2.2
(dalle bidirectionnelle), qui reste la référence de parité (``tests/test_dalle_moteur.py``).

    from beton.dalle import moteur
    res = moteur.verifier(moteur.nouvelle_dalle())

- ``entree``  : listes, bornes et valeurs par défaut de l'écran, classes de béton, normalisation ;
- ``moteur``  : couches (treillis, barres, n barres), centre de gravité, calcul des sections, vérifications, sélecteur
  de la nappe de base — et le FORMAT complet (entrée, sortie) dans sa docstring ;
- ``fichier`` : la lecture d'un fichier de l'écran Streamlit (« dalle_ba.json », migration v1 comprise).
La base des treillis soudés est ``beton/treillis.py`` (partagée avec les prédalles).
"""
from .moteur import (  # noqa: F401 — l'API
    candidats_base,
    classes_beton,
    nom_section_suivant,
    normaliser,
    nouvelle_dalle,
    nouvelle_section,
    verifier,
)

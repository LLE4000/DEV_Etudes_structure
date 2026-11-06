# -*- coding: utf-8 -*-
"""Poutre en béton armé (méthode du bureau), sans Streamlit : le calcul extrait de l'écran
``modules/poutre.py`` v2.40, qui reste la référence de parité (``tests/test_poutre_moteur.py``).

    from beton.poutre import moteur
    res = moteur.verifier(moteur.nouvelle_poutre())

- ``entree``    : listes et valeurs par défaut du module, classes de béton, normalisation ;
- ``geometrie`` : position des lits, centre de gravité, aires, étriers ;
- ``moteur``    : calcul des sections, vérifications, sélecteur d'armatures — et le FORMAT complet
  (entrée, sortie) dans sa docstring.
"""
from .moteur import (  # noqa: F401 — l'API
    candidats_armatures,
    classes_beton,
    nom_section_suivant,
    normaliser,
    nouvelle_poutre,
    nouvelle_section,
    verifier,
)

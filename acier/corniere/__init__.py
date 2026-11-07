# -*- coding: utf-8 -*-
"""Cornière ancrée (méthode de l'écran ``modules/corniere.py``), sans Streamlit : le calcul extrait de l'écran,
qui reste la référence de parité (``tests/test_corniere_moteur.py``).

    from acier.corniere import moteur
    res = moteur.verifier(moteur.nouvelle_corniere())

- ``entree`` : défauts et bornes des champs de l'écran, catalogue des cornières (acier/profiles), normalisation ;
- ``moteur`` : le calcul de l'écran mot pour mot (modèle lame, transversal et longitudinal), les quatre
  vérifications, les formules du détail, le sélecteur trié par masse — et le FORMAT complet dans sa docstring.
"""
from .moteur import (  # noqa: F401 — l'API
    calculer,
    candidats,
    cornieres,
    normaliser,
    nouvelle_corniere,
    optimal,
    section_catalogue,
    verifier,
)

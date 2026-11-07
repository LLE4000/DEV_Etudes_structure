# -*- coding: utf-8 -*-
"""Taux d'armature d'une poutre en béton armé (quantitatif), sans Streamlit : le calcul extrait de l'écran
``modules/taux_armature.py``, qui reste la référence de parité (``tests/test_taux_moteur.py``).

    from beton.taux import moteur
    res = moteur.calculer(moteur.nouvelle_poutre())

- ``entree``    : les listes, les libellés, les bornes et les valeurs par défaut de l'écran ; la normalisation ;
- ``geometrie`` : la section (âme et talons), son aire, la position des barres de chaque lit ;
- ``moteur``    : longueurs, étriers, quantités, taux d'armature, couleurs et codes des familles — et le FORMAT
  complet (entrée, sortie) dans sa docstring.
"""
from .moteur import (  # noqa: F401 — l'API
    calculer,
    candidats_diametre,
    normaliser,
    nouveau_type_etrier,
    nouvelle_famille,
    nouvelle_poutre,
    nouvelle_zone,
    options_positions,
    synthese_note,
)

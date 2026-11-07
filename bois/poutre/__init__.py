# -*- coding: utf-8 -*-
"""Poutre en bois massif sur deux appuis (vérifications EC5 simplifiées de l'écran), sans Streamlit : le calcul
extrait de l'écran ``modules/poutre_bois.py``, qui reste la référence de parité (``tests/test_poutre_bois_moteur.py``).

    from bois.poutre import moteur
    res = moteur.verifier(moteur.nouvelle_poutre())

- ``donnees``     : la base des classes, les sections standard, γ_M, k_mod, k_def, les menus, défauts et bornes ;
- ``entree``      : la poutre neuve et la normalisation stricte (ValueError hors du domaine de l'écran) ;
- ``moteur``      : le calcul de l'écran mot pour mot, les vérifications, les formules du détail, le sélecteur de
  sections — et le FORMAT complet dans sa docstring ;
- ``diagrammes``  : les courbes V(x), M(x) et w(x) des dessins (mêmes maxima que le calcul).
"""
from .moteur import (  # noqa: F401 — l'API
    ECARTS,
    candidats_sections,
    normaliser,
    nouvelle_poutre,
    optimal,
    section_standard,
    verifier,
)

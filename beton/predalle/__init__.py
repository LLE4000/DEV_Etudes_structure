# -*- coding: utf-8 -*-
"""Prédalle en béton armé (méthode du bureau), sans Streamlit : le calcul extrait de l'écran ``modules/predalle.py``
v1.1, qui reste la référence de parité (``tests/test_predalle_moteur.py``).

    from beton.predalle import moteur
    res = moteur.verifier(moteur.nouvelle_predalle())

- ``entree``  : la peau préfabriquée (h_pre, bornes de l'écran), les défauts de l'écran prédalle, la normalisation
  (le reste est contrôlé par le moteur des dalles) ;
- ``moteur``  : la règle prédalle des axes automatiques, le calcul (celui du moteur des dalles, ``beton/dalle``, sur les
  axes figés — comme la note PDF de l'écran), les avertissements propres à la prédalle, le sélecteur de la nappe de base ;
- ``fichier`` : la lecture d'un fichier de l'écran Streamlit (« predalle_ba.json »).
La base des treillis soudés est ``beton/treillis.py`` (partagée avec les dalles).
"""
from .moteur import (  # noqa: F401 — l'API
    candidats_base,
    classes_beton,
    nom_section_suivant,
    normaliser,
    nouvelle_predalle,
    nouvelle_section,
    verifier,
)

# -*- coding: utf-8 -*-
"""Maturité du béton : à quel âge, et quel jour, un béton coulé par ce temps atteint une résistance — EN 1992-1-1:2004
§3.1.2, NBN B 15-400 (complément belge à la NBN EN 13670), EN 13670:2009 §8.5. Le module a été REFAIT sur la norme le
10/11/2026 (``docs/site/LOT_Q_QUATRE_MODULES.md`` § 2) : il ne reproduit plus l'écran Streamlit ``modules/age_beton.py``.

    from beton.maturite import moteur
    res = moteur.calculer(moteur.nouvel_element())

- ``lois``    : les tables des normes (classes, durcissement, lois de température) et les formules, en ``math`` pur ;
- ``releve``  : le relevé de températures journalières (lecture du tableau collé ou du fichier) ;
- ``meteo``   : la météo du lieu (Open-Meteo), sans réseau : le plan des appels et la lecture des réponses ;
- ``entree``  : le coulage neuf, les bornes, la normalisation stricte ;
- ``moteur``  : le calcul de tout ce que l'écran montre — et le FORMAT complet (entrée, sortie) dans sa docstring.

Bibliothèque standard seule : ni numpy, ni streamlit, ni matplotlib, ni requests, ni pandas.
"""
from .moteur import calculer, candidats_classes, normaliser, nouvel_element  # noqa: F401 — l'API

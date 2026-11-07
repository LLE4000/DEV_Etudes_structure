# -*- coding: utf-8 -*-
"""Rigidité du sol (coefficient de réaction de Winkler), sans Streamlit : le calcul de l'écran
``modules/rigidite_sol.py`` v5.0, qui reste la référence de parité (``tests/test_sol_moteur.py``).

- ``base``     : la base des sols (copie conforme de modules/sol_base.py) ;
- ``theorie``  : diffusion de Boussinesq/Newmark, tassement par tranches, Robertson (copie conforme de
  modules/sol_theorie.py) ;
- ``imports``  : lecture des essais GEF et CSV, exports CSV (extrait de modules/sol_import.py, sans les PDF) ;
- ``rigidite`` : ce que l'ÉCRAN fait de ces briques — sondages et couches, préremplissage par type de sol, import d'un
  essai, k encadré et zoné, enveloppe, panneau SCIA, données des graphiques, et les cinq autres modes de l'écran
  (vérification rapide, théories, abaque, plat en béton) — avec son format complet dans sa docstring.

Bibliothèque standard seule : le paquet se charge et calcule dans le navigateur (Pyodide).
"""

# -*- coding: utf-8 -*-
"""Moteur « Choix de profilé » (EC3, section élastique + flambement) SANS Streamlit ni pandas.

Extrait de l'écran Streamlit ``modules/choix_profile.py``, qui reste la RÉFÉRENCE ; la preuve de parité est
``tests/test_choix_profil_moteur.py`` et la méthode est écrite dans ``docs/acier/CHOIX_PROFIL_MOTEUR.md``.

- ``calcul``  : les fonctions du module recopiées dans le même ordre d'opérations (chargement des profilés,
  résistances de section, taux, flambement) — les flottants sont identiques au bit près ;
- ``entree``  : les listes, les valeurs par défaut de l'écran, la normalisation STRICTE d'un élément ;
- ``moteur``  : le calcul d'un élément (le tableau trié de l'écran, l'optimal du module, le profilé retenu, ses
  résistances, ses taux, ses formules, ses avertissements) et les candidats du sélecteur.

Bibliothèque standard seule, plus numpy chargé par le catalogue des profilés (``acier/profiles``) : le chemin du calcul
tourne dans le navigateur (Pyodide).
"""

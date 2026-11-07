# -*- coding: utf-8 -*-
"""Flambement par flexion d'une barre comprimée — EN 1993-1-1 §6.3.1 (ANB belge : γ_M0 = γ_M1 = 1,0).

NOUVEAU CALCUL : l'écran Streamlit (modules/flambement.py) n'en avait aucun (« Page en construction »). Ce paquet est
donc écrit ici, et non extrait ; il est à valider par le bureau (docs/acier/FLAMBEMENT_MOTEUR.md).

- ``normes``  : ce que la norme fixe — f_y (Tableau 3.1), ε, les limites de classe (Tableau 5.2), la largeur efficace
  (EN 1993-1-5 §4.4), les facteurs d'imperfection (Tableau 6.1), le choix de la courbe (Tableau 6.2), χ (§6.3.1.2) ;
- ``section`` : un profilé du catalogue (acier/donnees/profiles.json, source unique) vu par le flambement : aires,
  inerties par axe, parois comprimées, fabrication ;
- ``entree``  : la barre (profilé, nuance, effort, longueur, appuis par axe), ses défauts et sa relecture ;
- ``moteur``  : la vérification d'une barre et la liste des profilés candidats (le sélecteur).

Sans Streamlit ni bibliothèque graphique ; le calcul n'emploie que la bibliothèque standard (le catalogue, lui,
charge numpy pour ses contours : il est présent dans le navigateur).
"""

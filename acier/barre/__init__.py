# -*- coding: utf-8 -*-
"""Une BARRE en acier — poutre, poteau, poutre-poteau — vérifiée par DEUX méthodes calculées ensemble :

- la méthode HISTORIQUE du bureau (ancienne norme belge NBN B 51-001 : contraintes admissibles, charges de service,
  sécurité s = 1,5), celle des classeurs « Flambement » et de la documentation théorique du bureau ;
- l'EUROCODE 3 (NBN EN 1993-1-1 et NBN EN 1993-1-5, annexes belges).

La spécification vérifiée, ses sources, ses valeurs de référence et l'état des lots : docs/acier/BARRES_ACIER.md.

- ``section``    : un profilé du catalogue (acier/donnees/profiles.json, source unique) — ou des propriétés saisies —
                   vu par les deux méthodes, en N et mm ;
- ``historique`` : la méthode historique (flambement, second ordre, flexion composée plane et déviée, déversement de
                   la documentation, cisaillement, contrainte équivalente), ses garde-fous ;
- ``formules``   : les gabarits des formules que le moteur nomme (la note et les fenêtres de calcul les écrivent).

Unités internes : N, mm, N/mm². Sans Streamlit ni bibliothèque graphique ; le calcul n'emploie que la bibliothèque
standard (le catalogue, lui, charge numpy pour ses contours : il est présent dans le navigateur).
"""

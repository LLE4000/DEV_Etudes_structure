# -*- coding: utf-8 -*-
"""Assemblage poutre–poutre par PLAT D'ÂME SOUDÉ (« fin plate »).

Un plat vertical soudé (double cordon) sur l'âme de la poutre porteuse,
boulonné à l'âme de la poutre portée. Attache simple la plus courante avec
les doubles cornières — EN 1993-1-8 et modèles du guide « Steel Buildings in
Europe – Part 5: Joint Design » §3 (fin plate), SCI P358, ECCS n°126.

- ``entrees``    : les 67 données d'entrée, défauts, groupes, conditions ;
- ``moteur``     : calcul complet (vérifications, alertes, Tableau 3.3,
                   prédimensionnement) — spécifique à cet assemblage ;
- ``schemas``    : élévation, vue en plan, vue de droite (kit de dessin du
                   module doubles cornières, réutilisé) ;
- ``interface``  : écran Streamlit (deux colonnes, dessin figé) ;
- ``note``       : note de calcul 2 pages (synthèse + plan de principe) ;
- ``benchmark``  : exemple publié (MSB P5 §3.4) et calculs manuels.
"""

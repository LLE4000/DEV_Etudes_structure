# -*- coding: utf-8 -*-
"""Assemblage poutre–poutre par PLATINE D'ABOUT de hauteur partielle
(« partial depth end plate », « header plate »).

Une platine soudée en about de la poutre portée (double cordon sur l'âme,
sur une hauteur partielle hp), boulonnée sur l'âme de la poutre porteuse
par deux files verticales de boulons. Troisième attache simple de la
famille, avec les doubles cornières et le plat d'âme — EN 1993-1-8 et
modèles du guide « Steel Buildings in Europe – Part 5: Joint Design »,
SCI P358, ECCS n°126.

- ``entrees``    : les données d'entrée, défauts, groupes, conditions ;
- ``moteur``     : calcul complet (vérifications, alertes, Tableau 3.3,
                   prédimensionnement) — spécifique à cet assemblage ;
- ``schemas``    : élévation, vue en plan, vue de droite (kit de dessin du
                   module doubles cornières, réutilisé) ;
- ``interface``  : écran Streamlit (deux colonnes, dessin figé) ;
- ``note``       : note de calcul 2 pages (synthèse + plan de principe) ;
- ``benchmark``  : exemple publié et calculs manuels.
"""

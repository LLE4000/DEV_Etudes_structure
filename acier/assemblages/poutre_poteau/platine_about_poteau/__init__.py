# -*- coding: utf-8 -*-
"""Assemblage poutre–poteau par PLATINE D'ABOUT de hauteur partielle.

Platine mince soudée en about de la poutre portée et boulonnée par deux
files symétriques sur l'aile du poteau (cas a) ou dans son âme (cas b) —
EN 1993-1-8, MSB Part 5 §2, SCI P358, ECCS n°126, ANB belge, tying.

Les vérifications, les clés, les dessins et les cas de benchmark sont ceux
de l'étude `docs/assemblages/POUTRE_POTEAU.md` (§8.3, §10.6, §10.7, §11.3) ;
la fiche du module est `docs/assemblages/PLATINE_ABOUT_POTEAU.md`. Le
découpage en treize fichiers est celui du GABARIT de la famille
(`plat_ame_poteau`).

Deux composants NORMATIFS que les trois modules poutre–poutre n'ont pas,
parce qu'ici les boulons traversent le POTEAU : le TRONÇON EN T de l'aile
non raidie (Tableau 6.4) et l'ÂME DU POTEAU EN TRACTION TRANSVERSALE
(§6.2.6.3). Les briques de calcul communes à la famille viennent de
`acier/assemblages/mecanique.py` ; la page 1 de la note et la page 2 (plan
de principe) sont des SOUS-CLASSES de celles des modules poutre–poutre.
"""

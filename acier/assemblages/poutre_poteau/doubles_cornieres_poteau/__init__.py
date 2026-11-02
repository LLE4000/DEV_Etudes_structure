# -*- coding: utf-8 -*-
"""Assemblage poutre–poteau par DOUBLES CORNIÈRES D'ÂME.

Deux cornières boulonnées ou soudées sur l'aile du poteau (cas a) ou dans
son âme (cas b), et sur l'âme de la poutre portée — EN 1993-1-8,
MSB Part 5 §4, SCI P358, ECCS n°126, ANB belge, tying.

Les vérifications, les clés, les dessins et les cas de benchmark sont ceux
de l'étude `docs/assemblages/POUTRE_POTEAU.md` (§8.2, §10.6, §10.7, §11.4) ;
la fiche du module est `docs/assemblages/DOUBLES_CORNIERES_POTEAU.md`. Le
GABARIT de la famille est `plat_ame_poteau` (lot 2).

Les briques de calcul communes à la famille viennent de
`acier/assemblages/mecanique.py` ; la page 1 de la note et la page 2 (plan de
principe) sont des SOUS-CLASSES de celles des modules poutre–poutre.
"""

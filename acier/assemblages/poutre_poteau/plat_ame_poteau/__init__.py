# -*- coding: utf-8 -*-
"""Assemblage poutre–poteau par PLAT D'ÂME SOUDÉ (« fin plate »).

Plat vertical soudé sur l'aile du poteau, au droit de son âme (cas a), ou
dans l'âme du poteau (cas b), et boulonné à l'âme de la poutre portée —
EN 1993-1-8, MSB Part 5 §3, SCI P358, ECCS n°126, ANB belge, tying.

C'est le GABARIT de la famille poutre–poteau : les vérifications, les clés,
les dessins et les cas de benchmark sont ceux de l'étude
`docs/assemblages/POUTRE_POTEAU.md` (§8.1, §10.6, §10.7, §11.2) ; la fiche du
module est `docs/assemblages/PLAT_AME_POTEAU.md`.

Les briques de calcul communes à la famille viennent de
`acier/assemblages/mecanique.py` ; la page 1 de la note et la page 2 (plan de
principe) sont des SOUS-CLASSES de celles des modules poutre–poutre.
"""

# -*- coding: utf-8 -*-
"""Famille Poutre – Poteau.

Trois attaches ARTICULÉES d'une poutre portée sur un poteau en I ou H, dans
les DEUX cas d'appui : sur l'AILE (cas a) et dans l'ÂME (cas b) —
`docs/assemblages/POUTRE_POTEAU.md` :

- ``plat_ame_poteau``          : plat d'âme soudé (fin plate) ;
- ``doubles_cornieres_poteau`` : doubles cornières d'âme ;
- ``platine_about_poteau``     : platine d'about de hauteur partielle.

**Les trois sont disponibles** au registre (lots 2, 3 et 4 ;
``plat_ame_poteau`` est le GABARIT des deux autres). Ce qu'elles ont en
commun vit à trois endroits, et nulle part ailleurs :

- le CALCUL, dans `acier/assemblages/mecanique.py` — dont
  ``geometrie_poteau``, les largeurs utiles du poteau dans les deux cas
  d'appui ;
- les MOTS, dans ``textes.py`` — les huit limites communes du §8.4 et les
  deux alertes communes (``aile_poutre``, ``ame_hp``) ;
- les TRAITS, dans ``dessin.py`` — la marge du tronçon de poteau, les deux
  noms de pièce et la section du poteau tournée selon l'appui.

Ce qui diffère d'une attache à l'autre reste chez elle, et c'est voulu : la
neuvième limite (flexion hors plan de l'âme) et l'alerte ``b1_long`` du plat
d'âme. Une rédaction différente n'est pas une copie.
"""

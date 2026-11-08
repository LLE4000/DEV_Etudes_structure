# -*- coding: utf-8 -*-
"""Le moteur de la BIBLIOTHÈQUE BÉTON ARMÉ du site (docs/site/BIBLIOTHEQUES.md) : des grandeurs de référence CALCULÉES
depuis les textes normatifs, nommées, jamais recopiées — les tableaux de l'écran en sont des vues.

- ``ancrages``  : longueurs d'ancrage et de recouvrement des barres (EN 1992-1-1 §8.4, §8.7) ;
- ``enrobage``  : enrobage nominal aux valeurs de la norme (EN 1992-1-1 §4.4.1, EN 1992-1-2 §5), avec la classe
                  structurale calculée (tableau 4.3N) et les libellés belges de la NIT 260 ;
- ``aciers``    : les aciers d'armature (nuances, classes de ductilité, EN 1992-1-1 §3.2 et annexe C).

Les bétons (tableau 3.1) restent dans ``beton/donnees/ec2.py``, complété des déformations et de f_cd. Les sections
d'armatures restent dans ``beton/armatures``. Bibliothèque standard seule : le paquet tourne dans le navigateur.
"""

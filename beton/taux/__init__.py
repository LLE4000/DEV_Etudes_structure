# -*- coding: utf-8 -*-
"""Taux d'armature d'une poutre en béton armé : un QUANTITATIF — la masse d'acier à commander, son bordereau par repère
et par diamètre, le taux en kg/m³. Bibliothèque standard seule ; refait sur la conception du 10/11/2026
(docs/site/LOT_Q_QUATRE_MODULES.md § 3 ; docs/beton/TAUX_ARMATURE_MOTEUR.md).

    from beton.taux import moteur
    res = moteur.calculer(moteur.nouvelle_poutre())

- ``entree``        : la poutre type d'une note neuve, les listes, les bornes des champs ; la normalisation stricte ;
- ``geometrie``     : la section (âme et talons), son aire, son polygone ; la position des barres, des cadres et des
  épingles pour la coupe ;
- ``moteur``        : les longueurs, le compte des étriers, le bordereau, les synthèses, le taux, les avertissements, le
  CSV — et la MÉTHODE et le FORMAT complets dans sa docstring ;
- ``saisie``        : les gestes de l'écran (ajouter une ligne, une zone ; ce qui suit une limite ou la longueur) ;
- ``depuis_poutre`` : recevoir une poutre dimensionnée dans « Poutres béton armé ».
"""
from .depuis_poutre import convertir  # noqa: F401 — l'API
from .moteur import (  # noqa: F401
    bordereau_csv,
    calculer,
    compter_etriers,
    normaliser,
    nouvelle_poutre,
    synthese_note,
)

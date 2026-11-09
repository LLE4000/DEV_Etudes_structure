# -*- coding: utf-8 -*-
"""Cornière d'appui, refondue le 10/11/2026 sur la méthode du classeur de l'ingénieur (feuille P8) — deux
configurations (A : aile horizontale en bas, B : en haut), deux modes (historique, équilibre), les jeux par
l'enveloppe de neuf combinaisons. Décisions : ``docs/acier/CORNIERE_DECISIONS.md`` ; livraison et preuve :
``docs/acier/CORNIERE_MOTEUR.md`` ; test : ``tests/test_corniere_moteur.py``.

    from acier.corniere import moteur
    res = moteur.verifier(moteur.nouvelle_corniere())

- ``entree``   : les champs du § 2 (défauts, domaine, unités mm/kN/N/mm²), le catalogue des cornières (acier/profiles),
                 les nuances du site, la normalisation stricte, la relecture d'une note v1 (modèle lame) ;
- ``statique`` : la statique d'une combinaison (M, N, V le long des deux ailes, pour chaque mode et configuration),
                 l'épaisseur du congé t(s), la recherche du maximum ;
- ``moteur``   : l'enveloppe des combinaisons, les vérifications, les ancrages, les diagrammes, les formules du détail,
                 les avertissements, le sélecteur — et le CONTRAT complet dans sa docstring.
"""
from .moteur import (  # noqa: F401 — l'API
    calculer,
    candidats,
    combinaisons,
    cornieres,
    normaliser,
    nouvelle_corniere,
    optimal,
    section,
    section_catalogue,
    verifier,
)

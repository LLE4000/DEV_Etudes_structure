# -*- coding: utf-8 -*-
"""Générateur SAF — FONDATION DE GRUE : une dalle en béton armé sur sol élastique sous les quatre pieds d'une grue à tour.

    geometrie.py     polygones, pieds, plaques, plots, bandes d'intégration, contour décalé          (bibliothèque standard)
    positions.py     les positions des charges « qui tournent » et les cas de charge                 (bibliothèque standard)
    plaque.py        la plaque mince sur sol élastique en compression seule (élément ACM, numpy)
    verification.py  la vérification rapide : tous les cas, plusieurs k, « Proposer »
    chassis.py       le châssis d'un projet : celui de la bibliothèque (grues/chassis) ou un châssis saisi
    projet.py        le projet (ses valeurs par défaut, sa relecture) et le modèle qu'on en tire
    proposition.py   « Proposer » : pour chaque épaisseur, la plus petite dalle qui tient
    ecriture.py      les feuilles du SAF d'un projet ; saf/classeur.py les écrit en .xlsx
    ndc_type.json    le contenu du fichier SAF du bureau (NDC_TYPE.xlsx, annexe B du cahier des charges, textes du projet
                     neutralisés) : feuilles, en-têtes, tableaux, lignes fixes, Id — lu par ecriture.reference()

Unités : kN, m, kPa, MN/m³ ; w positif vers le bas ; contrainte de contact σ = k·w.
Le calcul de référence, indépendant (numpy et scipy), est tests/fondation_grue_ref.py : tests/test_saf_grue_moteur.py
compare ce moteur à lui, cas par cas.
"""

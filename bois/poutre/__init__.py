# -*- coding: utf-8 -*-
"""Poutre en bois massif ou lamellé-collé sur deux appuis, vérifiée selon l'Eurocode 5 (EN 1995-1-1 et son annexe
belge) — refaite le 10/11/2026 sur la norme (docs/site/LOT_Q_QUATRE_MODULES.md § 1 ; docs/bois/POUTRE_BOIS_MOTEUR.md).
Bibliothèque standard seule : le moteur tourne aussi dans le navigateur (Pyodide).

    from bois.poutre import moteur
    res = moteur.verifier(moteur.nouvelle_poutre())

- ``donnees``     : les classes (EN 338:2016, EN 14080:2013), γ_M, k_mod, k_def, k_cr, k_sys, k_h, les natures de la
  charge variable (durée, ψ₂), les sections standard, les bornes des champs, la poutre neuve ;
- ``entree``      : la poutre neuve et la relecture stricte (ValueError en français, qui nomme le champ) ;
- ``moteur``      : le calcul (deux combinaisons ELU, flexion et déversement, cisaillement, appui, flèches), les
  formules « symbole = expression = valeurs = résultat », le sélecteur de sections — et le FORMAT dans sa docstring ;
- ``diagrammes``  : les courbes V(x), M(x) et w(x) des dessins (mêmes maxima que le calcul).
"""
from .moteur import (  # noqa: F401 — l'API
    candidats_sections,
    normaliser,
    nouvelle_poutre,
    optimal,
    verifier,
)

# -*- coding: utf-8 -*-
"""Composant Streamlit bidirectionnel : un SVG dont les cotes et les groupes
se modifient sur place.

Générique et sans étape de compilation (HTML + JS simples, servis par
Streamlit) : il affiche un SVG fourni par Python et rend cliquables

- ``<g data-key="…">``    : une cote — saisie numérique à l'endroit de la
                            cote (Entrée valide, Échap annule) ;
- ``<g data-action="clé:±1">`` : une poignée — la valeur de ``clé`` change
                            de ±1 (jamais sous 1) ;
- ``<g data-group="G">``  : un objet (étiquette de groupe, pièce du dessin,
                            étiquette d'efforts) — fenêtre dont les champs
                            sont décrits par ``groupes[G]`` (``titre``,
                            ``note``, ``champs`` = liste de ``dict(key,
                            lab, type 'n'|'s', options, step, dis)``) ;
                            la pièce s'illumine tant que la fenêtre est
                            ouverte (classe ``sel``) ;
- ``<g data-drag="clé">`` : une pièce qui se DÉPLACE horizontalement — la
                            valeur suit le glissement (mm entiers, ≥ 0,
                            bulle « gh = 25 mm » pendant le geste), un
                            relâchement sans mouvement vaut clic.

Chaque validation renvoie à Python ``{"changes": {clé: valeur, …}, "sym",
"seq", "t"}`` ; Python met à jour l'état unique et relance le calcul. Le
composant ne connaît ni le moteur ni les clés : il est réutilisable tel
quel par les prochains assemblages.

La valeur retournée persiste d'une relance à l'autre : l'appelant compare
``t`` (horodatage du clic) à celui déjà traité pour ce composant.
"""
import os

import streamlit.components.v1 as components

_DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_composant = components.declare_component("svg_cliquable", path=_DOSSIER)


def svg_cliquable(svg, valeurs, key, groupes=None, palette=None):
    """Affiche ``svg`` ; ``valeurs`` = valeur courante de chaque clé
    modifiable (préremplissage des saisies) ; ``groupes`` = définition des
    fenêtres de groupe. Retourne le dernier message validé (dict) ou None."""
    return _composant(svg=svg, valeurs=valeurs, groupes=groupes or {}, palette=palette or {},
                      key=key, default=None)

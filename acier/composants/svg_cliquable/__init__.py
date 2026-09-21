# -*- coding: utf-8 -*-
"""Composant Streamlit bidirectionnel : un SVG dont les cotes se modifient
sur place.

Générique et sans étape de compilation (HTML + JS simples, servis par
Streamlit) : il affiche un SVG fourni par Python, rend cliquable tout élément
``<g data-key="…">``, ouvre une saisie numérique à l'endroit de la cote et
renvoie ``{"key", "value", "sym", "seq", "t"}`` à Python, qui met à jour
l'état unique et relance le calcul. Réutilisable tel quel par les prochains
assemblages : il ne connaît ni le moteur ni les clés.

Usage ::

    retour = svg_cliquable(svg, valeurs={"LC_u": 190, …}, key="asm_cmp_elev")
    if retour is nouveau : st.session_state[…] = retour["value"] ; st.rerun()

La valeur retournée persiste d'une relance à l'autre : l'appelant compare
``t`` (horodatage du clic) à celui déjà traité pour ce composant.
"""
import os

import streamlit.components.v1 as components

_DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_composant = components.declare_component("svg_cliquable", path=_DOSSIER)


def svg_cliquable(svg, valeurs, key, palette=None):
    """Affiche ``svg`` ; ``valeurs`` = valeur courante de chaque clé
    modifiable (préremplissage de la saisie). Retourne le dernier clic
    validé (dict) ou None."""
    return _composant(svg=svg, valeurs=valeurs, palette=palette or {}, key=key, default=None)

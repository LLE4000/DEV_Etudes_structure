# -*- coding: utf-8 -*-
"""Bibliothèques de la famille Acier : profilés, aciers, boulons, classes,
cornières, ailes et épaisseurs standard.

Source unique : ``acier/donnees/bibliotheques.json``, extrait tel quel du
corrigé de référence (``acier/reference/reference_double_corniere.json``,
clé ``bibliotheques``). Un test vérifie l'égalité des deux ; rien n'est
recopié à la main.

SAUF les f_y et f_u des aciers : ils dépendent de l'épaisseur (option a de l'ANB
belge, 3.2.1(1)) et se lisent dans la table unique du site, ``acier/materiaux.py``,
par ``acier(nom, t, produit)``. La bibliothèque ne garde de chaque acier que son
nom et son β_w (EN 1993-1-8, tableau 4.1) ; le corrigé de référence, lui, garde
ses valeurs du tableau 3.1 de l'EN (l'écart est nommé et testé).

Les enregistrements gardent les noms de champs du moteur de référence :
profilé ``n h b tw tf r`` ; acier ``n bw`` (+ ``fy fu`` par ``acier``) ; boulon ``n d d0 As A dm dw`` ;
classe ``n fyb fub av`` ; cornière ``n a1 a2 t r`` ; ailes ``[aile, rayon]`` ;
épaisseurs ``[…]``. Unités : mm, mm², MPa.
"""
import json
import os

from acier import materiaux as MA

PERSO = "Personnalisé"
ORI_P = "Grande aile sur poutre principale"
ORI_S = "Grande aile sur poutre secondaire"

_CHEMIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "donnees", "bibliotheques.json")


def _charger():
    with open(_CHEMIN, "r", encoding="utf-8") as fh:
        return json.load(fh)


DB = _charger()
"""``DB.profils`` (90), ``DB.aciers`` (6), ``DB.boulons`` (8), ``DB.classes`` (4),
``DB.cornieres`` (18), ``DB.ailes`` (8), ``DB.epais`` (5) — accès par clé."""


def find(liste, n):
    """Premier enregistrement de nom ``n`` ; ``ValueError`` s'il n'existe pas
    (le moteur JS lèverait une TypeError : l'interface affiche alors
    « Données incomplètes »)."""
    for x in liste:
        if x["n"] == n:
            return x
    raise ValueError(f"Enregistrement inconnu : {n!r}")


def acier(nom, t, produit="lamine"):
    """L'acier ``nom`` pour une pièce d'épaisseur nominale ``t`` (mm) : {"n", "fy", "fu", "bw"} — f_y et f_u de la
    table unique (acier/materiaux.py, tableaux 3.3 ANB à 3.12 ANB, selon l'épaisseur et le produit : « lamine »,
    « creux_chaud », « creux_froid »), β_w de la bibliothèque. ValueError (MA.HorsTable) hors des tableaux."""
    a = find(DB["aciers"], nom)
    return {"n": nom, "fy": MA.fy(nom, t, produit), "fu": MA.fu(nom, t, produit), "bw": a["bw"]}


def idx(liste, n):
    """Indice du premier enregistrement de nom ``n`` ; 0 s'il n'existe pas."""
    for i, x in enumerate(liste):
        if x["n"] == n:
            return i
    return 0


def noms(liste):
    return [x["n"] for x in liste]


def aire_profil(p):
    """Aire recalculée : A = 2·b·tf + (h − 2tf)·tw + (4 − π)·r² (mm²)."""
    import math
    return 2 * p["b"] * p["tf"] + (p["h"] - 2 * p["tf"]) * p["tw"] \
        + (4 - math.pi) * p["r"] ** 2

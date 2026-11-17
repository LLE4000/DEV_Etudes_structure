# -*- coding: utf-8 -*-
"""Les NUANCES du module Barres, et de lui seul (brief v12 § 4.1 et § 7.1 ; lot C, C1b) : S235, S275, S355, S420,
S460 — et la NORME PRODUIT de S420 et S460 : N/NL (EN 10025-3, à défaut) ou M/ML (EN 10025-4).

Les VALEURS restent celles de la table unique acier/materiaux.py (option a de l'ANB:2018, 3.2.1(1) ; tableaux 3.3 ANB
à 3.12 ANB) : ce fichier choisit seulement la LIGNE (nuance et norme produit → clé de la table) et garde la borne du
domaine du module (80 mm, ``acier/flambement/normes.T_MAX_TABLEAU``). Les autres modules (Flambement, cornière,
Catalogue) gardent leur liste, ``materiaux.NUANCES``, où S420 n'est pas.

    NUANCES                       ("S235", "S275", "S355", "S420", "S460")
    NORMES_PRODUIT                (None, "M") — None : EN 10025-2 pour S235 à S355, N/NL (EN 10025-3) pour S420 et
                                  S460 ; « M » : M/ML (EN 10025-4), pour S420 et S460 seulement
    norme_effective(nuance, norme)   la norme produit qui SERT (None ou « M ») : « M » est sans effet sur S235 à S355
    cle(nuance, norme)            la clé de la table (« S420 », « S420M », « S460M »…)
    fy(nuance, t, produit, norme), fu(…)    N/mm² ; Refus hors du domaine (nuance, plus de 80 mm, hors du tableau)
    source(nuance, t, produit, norme, grandeur)   « tableau 3.5 ANB (EN 10025-4:2004, S420M), 63 < t ≤ 80 mm »
    tableau(nuance, produit, norme)               « tableau 3.5 ANB »

Le défaut N/NL est le côté sûr pour f_y : à toute épaisseur des deux tableaux, la f_y de N/NL est au plus celle de
M/ML (S420 de 70 mm : 370 contre 380 N/mm²) — À VÉRIFIER à la relecture (tests/test_barre_nuances.py le mesure). Ce
n'est pas vrai de f_u (S420 de 50 mm : 520 en N/NL, 500 en M/ML), qu'aucune vérification des barres n'emploie en v1.
« M » donné avec S235, S275 ou S355 est sans effet (EN 10025-2, une f_y au plus égale) : l'écran ne le propose qu'avec
S420 et S460 (À VÉRIFIER).
"""
from typing import Optional

from acier import materiaux as MA
from acier.flambement import normes as NO
from acier.formats import f0, f_ecart, mm_fr

from .resultat import Refus

NUANCES = ("S235", "S275", "S355", "S420", "S460")       # par limite élastique croissante (la liste de l'écran, C3)
NORMES_PRODUIT = (None, "M")
AVEC_NORME = ("S420", "S460")                              # les nuances dont la norme produit change la ligne de la table
LIBELLES_NORME = {None: "N/NL", "M": "M/ML"}               # (S420 et S460 ; S235 à S355 : EN 10025-2)


def _controler(nuance: str, norme_produit: Optional[str]) -> None:
    if nuance not in NUANCES:
        raise Refus(f"nuance inconnue : {nuance!r} ({', '.join(NUANCES[:-1])} ou {NUANCES[-1]})")
    if norme_produit not in NORMES_PRODUIT:
        raise Refus(f"norme produit : vide (N/NL) ou « M » (M/ML) — reçu {norme_produit!r}")


def norme_effective(nuance: str, norme_produit: Optional[str] = None) -> Optional[str]:
    """La norme produit qui sert : « M » seulement pour S420 et S460 ; None sinon."""
    _controler(nuance, norme_produit)
    return norme_produit if nuance in AVEC_NORME else None


def cle(nuance: str, norme_produit: Optional[str] = None) -> str:
    """La clé de la table acier/materiaux.py : « S420 » (N/NL), « S420M », « S460 » (N/NL), « S460M »…"""
    return nuance + (norme_effective(nuance, norme_produit) or "")


def _lire(lecteur, nuance: str, t: float, produit: str, norme_produit: Optional[str]) -> float:
    """La valeur de la table, avec la borne du DOMAINE du module (80 mm) — les messages de
    acier/flambement/normes.limite_elastique, mot pour mot (copie nommée, liée par tests/test_barre_nuances.py)."""
    k = cle(nuance, norme_produit)
    if t > NO.T_MAX_TABLEAU:
        raise Refus(f"épaisseur {f_ecart(t, NO.T_MAX_TABLEAU)} mm > {f0(NO.T_MAX_TABLEAU)} mm : hors du domaine du module")
    try:
        return lecteur(k, t, produit)
    except MA.HorsTable as e:
        raise Refus(mm_fr(e)) from None


def fy(nuance: str, t: float, produit: str = "lamine", norme_produit: Optional[str] = None) -> float:
    """f_y (N/mm²) pour l'épaisseur nominale ``t`` (mm) et le produit (« lamine », « creux_chaud », « creux_froid »)."""
    return _lire(MA.fy, nuance, t, produit, norme_produit)


def fu(nuance: str, t: float, produit: str = "lamine", norme_produit: Optional[str] = None) -> float:
    """f_u (N/mm²) pour l'épaisseur nominale ``t`` (mm) et le produit."""
    return _lire(MA.fu, nuance, t, produit, norme_produit)


def source(nuance: str, t: float, produit: str = "lamine", norme_produit: Optional[str] = None, grandeur: str = "fy") -> str:
    """D'où vient la valeur : « tableau 3.4 ANB (EN 10025-3:2004, S420N), 16 < t ≤ 40 mm » — la norme produit y est dite."""
    return MA.source(cle(nuance, norme_produit), t, produit, grandeur)


def tableau(nuance: str, produit: str = "lamine", norme_produit: Optional[str] = None) -> str:
    """Le tableau de l'ANB, en court : « tableau 3.5 ANB »."""
    return f"tableau {MA.ligne(cle(nuance, norme_produit), produit)['tableau']}"

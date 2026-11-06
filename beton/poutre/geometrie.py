# -*- coding: utf-8 -*-
"""Poutre en béton armé — la GÉOMÉTRIE des aciers : position de chaque lit (distance parement → axe),
centre de gravité y_G d'une face, aires des barres, section des armatures d'effort tranchant.

Chaque fonction reprend l'expression du module ``modules/poutre.py`` (la fonction d'origine est
nommée), DANS LE MÊME ORDRE d'opérations : les flottants sont identiques au bit près, ce que
``tests/test_poutre_moteur.py`` vérifie. Unités : Ø en mm, distances en cm, aires en mm².
"""
import math

from .entree import BRINS

ENROBAGE_SI_NUL = 3.0    # cm — le module lit l'enrobage « float(enrobage or 3.0) » (_auto_dist_lit)
D_ETRIER_SI_AUCUN = 8.0  # mm — _stirrup_diam_mm sans aucune ligne (inatteignable : une ligne au moins)


def aire_barre(d_mm: float) -> float:
    """Aire d'une barre (mm²) : π·(Ø/2)² — _bar_area_mm2."""
    return math.pi * (d_mm / 2.0) ** 2


def arrondi_demi_cm(x_cm: float) -> float:
    """Arrondi au 0,5 cm SUPÉRIEUR — _round_up_to_half_cm."""
    return math.ceil(float(x_cm) * 2.0) / 2.0


def enrobage_de_calcul(enrobage: float) -> float:
    """L'enrobage (cm) qui place le lit 1 : la saisie, sauf un enrobage NUL, que le module lit 3,0 cm
    (« or 3.0 » dans _auto_dist_lit, alors que son écran accepte 0). Défaut du module reproduit,
    signalé par un avertissement de ``moteur.verifier``."""
    return float(enrobage or ENROBAGE_SI_NUL)


def diametre_etrier(etriers: list) -> float:
    """Ø (mm) des armatures d'effort tranchant qui place le lit 1 : le plus grand des lignes
    (cas défavorable) — _stirrup_diam_mm."""
    diams = [float(e["d"]) for e in etriers]
    return max(diams) if diams else D_ETRIER_SI_AUCUN


def distances_lits(lits: list, enrobage_calc: float, d_etrier: float, jeu_premier_lit: float,
                   jeu_entre_lits: float) -> list:
    """Distance parement → axe de chaque lit (cm), AUTOMATIQUE comme à l'écran — _auto_dist_lit et
    _get_dist_lit (l'ancienne saisie manuelle n'est plus offerte par le module) :

        lit 1 : enrobage + arr½(Ø_étrier/10 + Ø₁/20) + jeu premier lit
        lit i : dist(i−1) + arr½(Ø(i−1)/20) + jeu entre lits + arr½(Øᵢ/20)

    arr½ = arrondi au 0,5 cm supérieur ; Ø_étrier = le plus grand des étriers et épingles."""
    out = []
    for i, lit in enumerate(lits):
        if i == 0:
            dist = enrobage_calc + arrondi_demi_cm(d_etrier / 10.0 + lit["d"] / 20.0) + jeu_premier_lit
        else:
            dist = (out[-1]
                    + arrondi_demi_cm(lits[i - 1]["d"] / 20.0)
                    + jeu_entre_lits
                    + arrondi_demi_cm(lit["d"] / 20.0))
        out.append(dist)
    return out


def face(lits: list, distances: list, yG_impose=None):
    """Une face : (As total en mm², y_G calculé en cm, y_G retenu en cm) — _layers_geometry.

    y_G calculé = Σ(Asᵢ·eᵢ) / ΣAsᵢ, eᵢ = distance du lit i ; y_G retenu = l'imposé s'il est donné
    (champ « CDG (cm) » de l'écran, _ycdg_manual), sinon le calculé."""
    As_tot = 0.0
    somme_As_e = 0.0
    for lit, e in zip(lits, distances):
        As_i = lit["n"] * aire_barre(lit["d"])
        As_tot += As_i
        somme_As_e += As_i * e
    yG_calcule = (somme_As_e / As_tot) if As_tot > 0 else distances[0]
    return As_tot, yG_calcule, (yG_calcule if yG_impose is None else yG_impose)


def aire_etriers(etriers: list) -> float:
    """A_st (mm²) = Σ brins × π·(Ø/2)², une ligne = un étrier (2 brins) ou une épingle (1 brin) —
    _shear_lines_total_Ast_mm2."""
    Ast = 0.0
    for e in etriers:
        Ast += BRINS[e["type"]] * aire_barre(float(e["d"]))
    return Ast


def libelle_lits(lits: list) -> str:
    """« 2Ø16 + 2Ø20 » : la désignation que l'écran affiche (_layers_geometry)."""
    return " + ".join(f"{lit['n']}Ø{lit['d']}" for lit in lits)


def resume_etriers(etriers: list) -> str:
    """« Étrier Ø8 + 2 épingles Ø8 » : la désignation du bandeau des étriers — _shear_lines_summary."""
    ordre, compte = [], {}
    for e in etriers:
        cle = ("Épingle" if BRINS[e["type"]] == 1 else "Étrier", int(float(e["d"])))
        if cle not in compte:
            compte[cle] = 0
            ordre.append(cle)
        compte[cle] += 1
    return " + ".join(f"{nom} Ø{d}" if compte[(nom, d)] == 1 else f"{compte[(nom, d)]} {nom.lower()}s Ø{d}"
                      for nom, d in ordre)

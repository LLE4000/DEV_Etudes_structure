# -*- coding: utf-8 -*-
"""Fondations de grues — l'ÉCRITURE d'une formule : la même dans la fenêtre de calcul de l'écran et dans la note PDF.

``formules.py`` dit de QUELS termes et par QUEL gabarit une valeur du classeur se retrouve. Ce fichier en fait ce qui
s'imprime — « symbole = expression = valeurs = résultat » — avec le kit des substitutions des notes
(``acier/assemblages/poutre_poutre/doubles_cornieres/formules.py`` ; il se charge sans numpy). Une seule écriture pour
l'écran et pour la note (R2) :

    sub(fx) -> Sub                                la substitution du kit pour une formule du moteur
    morceaux(fx) -> ([nom, littéral, numérique], résultat)      sans doublon, comme le kit
    litteral(fx) -> str                           l'expression en symboles (la colonne « Formule » d'un tableau)
    numerique(fx) -> str                          l'expression en valeurs imprimées
    relire(texte) -> float                        la VALEUR d'une expression imprimée

``relire`` est la preuve de l'écriture : ``tests/test_fondation_grue_moteur.py`` relit la chaîne IMPRIMÉE de chaque
formule et exige la valeur du gabarit — un « × 10³ » imprimé « × 100 » (trouvé le 11/11/2026) ne passe plus.

Une retouche à ce que le kit écrit : « /(V_max) » devient « /V_max » — les parenthèses qui restent autour d'un symbole
seul quand le facteur d'unité (« × 10³ ») est ôté de la forme littérale.
"""
import math
import re

from acier.assemblages.poutre_poutre.doubles_cornieres import formules as KIT

# « /(symbole) » : un seul symbole entre parenthèses derrière une barre de fraction (ni opérateur, ni espace dedans)
_PARENTHESES_INUTILES = re.compile(r"/\(([^()\s·×+−/*-]+)\)")
_FONCTIONS = {"sqrt": math.sqrt, "max": max, "min": min, "abs": abs, "pi": math.pi}


def decimales(v) -> int:
    """Les décimales d'un terme imprimé : CINQ chiffres significatifs (le kit en met quatre). Un calcul refait à la main
    avec les nombres imprimés retombe ainsi sur le résultat imprimé (« 1032,4 × 1,15 − … = 468,3 » et non « 1032 × … »),
    et un terme se lit comme le résultat qui l'a donné deux lignes plus haut (« 1101,4 kN »)."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v == 0 or abs(v - round(v)) < 1e-9:
        return 0
    return max(0, min(5, 4 - int(math.floor(math.log10(abs(v))))))


def sub(fx: dict) -> KIT.Sub:
    return KIT.Sub(fx["nom"], fx["gabarit"], {t[0]: (t[1], t[2], decimales(t[2])) for t in fx["termes"]}, fx["resultat"], fx["unite"], fx["nd"])


def _retoucher(expression: str) -> str:
    return _PARENTHESES_INUTILES.sub(r"/\1", expression)


def litteral(fx: dict) -> str:
    """L'expression en symboles, sans les facteurs d'unités."""
    return _retoucher(KIT.litteral(sub(fx)))


def numerique(fx: dict) -> str:
    """L'expression en valeurs, telles qu'elles sont imprimées (cinq chiffres significatifs : ``decimales``)."""
    return KIT.numerique(sub(fx))


def morceaux(fx: dict):
    """``([nom, littéral, numérique], résultat avec son unité)`` : les membres à joindre par « = » ; un membre qui
    répéterait le précédent ne s'écrit pas (une formule sans gabarit : le nom et le résultat)."""
    parts, res = KIT.morceaux(sub(fx))
    return [parts[0]] + [_retoucher(p) for p in parts[1:]], res


def relire(texte: str) -> float:
    """La valeur d'une expression numérique IMPRIMÉE (« 989,6 × 10³ × 100/100 », « √(425,9 × 10⁶/(…)) », « |−71,23| »).
    La virgule décimale est entre deux chiffres ; celle qui sépare deux arguments est suivie d'un espace."""
    t = texte.replace("\u202f", "").replace("\u00a0", "")
    t = t.replace("10³", "1000").replace("10⁶", "1e6")
    t = t.replace("^1,5", "**1.5").replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)
    t = t.replace("×", "*").replace("·", "*").replace("−", "-").replace("π", "pi")
    t = re.sub(r"√(\d+(?:\.\d+)?)", r"sqrt(\1)", t).replace("√(", "sqrt(")
    t = re.sub(r"\|([^|]*)\|", r"abs(\1)", t)
    return float(eval(t, {"__builtins__": {}}, dict(_FONCTIONS)))  # noqa: S307 — une expression que ce paquet a écrite

# -*- coding: utf-8 -*-
"""Poutre bois — les COURBES des dessins : effort tranchant V(x), moment M(x) et déformées w(x) le long de la poutre
sur deux appuis sous charge uniforme.

Rien de neuf : chaque courbe est la forme exacte de la poutre isostatique, PORTÉE PAR LE MAXIMUM QUE LE CALCUL A
DONNÉ (``moteur.verifier``) — V(0) = V_Ed, M(L/2) = M_Ed, w(L/2) = w_inst ou w_fin, tels que l'écran Streamlit les
calcule (flèches comprises, avec leur écart d'unité : voir ``moteur.ECARTS``). Le dessin ne recalcule donc aucune
valeur affichée ; il les répartit le long de la portée.

    ξ = x / L ;  V(x) = V_Ed·(1 − 2ξ) ;  M(x) = 4·M_Ed·ξ·(1 − ξ) ;  w(x) = w_max·(16/5)·ξ·(1 − 2ξ² + ξ³)
"""
from typing import Dict, List


def forme_fleche(xi: float) -> float:
    """La déformée de la poutre isostatique sous charge uniforme, rapportée à sa flèche à mi-portée (1 en ξ = ½)."""
    return 16.0 / 5.0 * xi * (1.0 - 2.0 * xi * xi + xi ** 3)


def courbes(res: Dict, n: int = 49) -> Dict[str, List]:
    """Les courbes d'un résultat de ``moteur.verifier`` sur ``n`` points (n impair : mi-portée comprise) :
    ``{"x": [m], "V": [kN], "M": [kN·m], "w_inst": [mm], "w_fin": [mm]}``."""
    if n < 3 or n % 2 == 0:
        raise ValueError("n : entier impair ≥ 3 attendu")
    L = res["geometrie"]["L"]
    g = res["grandeurs"]
    out = {"x": [], "V": [], "M": [], "w_inst": [], "w_fin": []}
    for i in range(n):
        xi = i / (n - 1)
        out["x"].append(L * xi)                       # x(L) = L exactement
        out["V"].append(g["V_Ed"] * (1.0 - 2.0 * xi))
        out["M"].append(4.0 * g["M_Ed"] * xi * (1.0 - xi))
        f = forme_fleche(xi)
        out["w_inst"].append(g["w_inst"] * f)
        out["w_fin"].append(g["w_fin"] * f)
    return out

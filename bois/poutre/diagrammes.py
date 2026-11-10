# -*- coding: utf-8 -*-
"""Poutre bois — les COURBES des dessins : l'effort tranchant V(x) et le moment M(x) de la combinaison ELU qui
gouverne, les déformées w_inst(x), w_fin(x) et w_net,fin(x), le long de la poutre sur deux appuis sous charge
uniformément répartie.

Chaque courbe est la forme exacte de la poutre isostatique, calculée à partir des grandeurs que ``moteur.verifier``
a rendues (charges, modules, section, k_def, ψ₂) — avec les mêmes expressions que lui à mi-portée : les maxima des
courbes SONT ceux du calcul (le test le prouve à 1e-9 près) :

    V(x) = p_d·(L/2 − x)                    V(0) = V_d
    M(x) = p_d·x·(L − x)/2                  M(L/2) = M_d
    w(p, x) = p·x·(L³ − 2·L·x² + x³)/(24·E·I) + 1,2·p·x·(L − x)/(2·G·A)        flexion + effort tranchant
                                            w(p, L/2) = 5·p·L⁴/(384·E·I) + 1,2·p·L²/(8·G·A)
    w_inst(x) = w(g, x) + w(q, x)
    w_fin(x) = w(g, x)·(1 + k_def) + w(q, x)·(1 + ψ₂·k_def)
    w_net,fin(x) = w_fin(x) − w_c·4·ξ·(1 − ξ)      la contre-flèche prise parabolique (ξ = x/L), w_c à mi-portée

Unités : x en m, V en kN, M en kN·m, w en mm.
"""
from typing import Dict, List

from . import donnees as D


def fleche(p: float, x_mm: float, L_mm: float, E_mod: float, inertie: float, G_mod: float, aire: float) -> float:
    """La flèche (mm) à l'abscisse x (mm) sous la charge répartie p (N/mm) : flexion + effort tranchant."""
    flexion = p * x_mm * (L_mm ** 3 - 2.0 * L_mm * x_mm ** 2 + x_mm ** 3) / (24.0 * E_mod * inertie)
    tranchant = D.FORME_TRANCHANT * p * x_mm * (L_mm - x_mm) / (2.0 * G_mod * aire)
    return flexion + tranchant


def courbes(res: Dict, n: int = 49) -> Dict[str, List]:
    """Les courbes d'un résultat de ``moteur.verifier`` sur ``n`` points (n impair : la mi-portée en fait partie) :
    ``{"combinaison", "nom", "x", "V", "M", "w_inst", "w_fin", "w_net_fin"}``."""
    if n < 3 or n % 2 == 0:
        raise ValueError("n : un entier impair d'au moins 3 est attendu")
    geo, mat, ch, co, ef, fl = res["geometrie"], res["materiau"], res["charges"], res["coefficients"], res["efforts"], res["fleches"]
    L, L_mm = geo["L"], geo["L_mm"]
    out = {"combinaison": ef["combinaison"], "nom": ef["nom"], "x": [], "V": [], "M": [], "w_inst": [], "w_fin": [], "w_net_fin": []}
    for i in range(n):
        xi = i / (n - 1)
        x = L * xi
        out["x"].append(x)
        out["V"].append(ef["p_d"] * (L / 2.0 - x))
        out["M"].append(ef["p_d"] * x * (L - x) / 2.0)
        w_g = fleche(ch["g"], L_mm * xi, L_mm, mat["E0_mean"], geo["I"], mat["G_mean"], geo["A"])
        w_q = fleche(ch["q"], L_mm * xi, L_mm, mat["E0_mean"], geo["I"], mat["G_mean"], geo["A"])
        w_fin = w_g * (1.0 + co["kdef"]) + w_q * (1.0 + co["psi2"] * co["kdef"])
        out["w_inst"].append(w_g + w_q)
        out["w_fin"].append(w_fin)
        out["w_net_fin"].append(w_fin - fl["w_c"] * 4.0 * xi * (1.0 - xi))
    return out

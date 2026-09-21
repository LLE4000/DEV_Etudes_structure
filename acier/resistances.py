# -*- coding: utf-8 -*-
"""Résistances élémentaires des boulons et des cordons — EN 1993-1-8.

Communes à la famille Acier ; le moteur de chaque assemblage les appelle avec
ses propres valeurs. Unités : MPa, mm, mm² ; résultats en kN (boulons) ou en
MPa (cordons). Les formules sont celles du moteur de référence, à
l'identique.
"""
import math

from .js import dv, mx, mn

S3 = math.sqrt(3)


def Fv_Rd(a_v, f_ub, A, g_M2):
    """Cisaillement d'un boulon par plan : αv·fub·A/γM2 (Tableau 3.4), kN."""
    return dv(a_v * f_ub * A, g_M2) / 1000


def Ft_Rd(f_ub, As, g_M2):
    """Traction d'un boulon : 0,9·fub·As/γM2 (Tableau 3.4), kN."""
    return dv(0.9 * f_ub * As, g_M2) / 1000


def Bp_Rd(d_m, t_fu_min, g_M2):
    """Poinçonnement : 0,6·π·dm·tp·fu/γM2 (Tableau 3.4), kN ; ``t_fu_min`` =
    min(t·fu) des pièces sous la tête ou l'écrou."""
    return dv(0.6 * math.pi * d_m * t_fu_min, g_M2) / 1000


def Fp_C(f_ub, As):
    """Précontrainte : 0,7·fub·As (§3.9.1), kN."""
    return 0.7 * f_ub * As / 1000


def beta_Lf(Lj, d, actif):
    """Coefficient des assemblages longs (§3.8) : 1 − (Lj − 15d)/(200d),
    borné à 0,75 ; 1 si l'option est inactive ou Lj ≤ 15d."""
    if actif and Lj > 15 * d:
        return mx(0.75, 1 - dv(Lj - 15 * d, 200 * d))
    return 1


def fvw_d(fu, bw, g_M2):
    """Résistance de calcul du cordon (méthode simplifiée, §4.5.3.3) :
    fu/(√3·βw·γM2), MPa."""
    return dv(fu, S3 * bw * g_M2)


def fu_bw_cordon(fu_C, bw_C, fu_autre, bw_autre):
    """Acier de référence d'un cordon entre deux pièces : la plus faible
    résistance ultime et le βw qui va avec."""
    return mn(fu_C, fu_autre), (bw_C if fu_C <= fu_autre else bw_autre)

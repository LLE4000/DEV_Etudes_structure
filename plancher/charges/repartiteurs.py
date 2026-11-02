# -*- coding: utf-8 -*-
"""Répartiteurs (§ 5 de l'étude) : deux régimes, un moteur.

- Plateau posé directement sur le plancher : régime de RÉSISTANCE. Le
  plateau ne porte en console que sur une bande ``c = t·√(f/(3·p))`` de
  chaque côté de l'empreinte (EN 1993-1-8 §6.2.5, la même règle que les
  platines de pied de poteau de l'application) ; comme la pression ``p``
  dépend de l'aire efficace, on itère.
- Filière ou plateau à cheval sur plusieurs éléments : régime de
  RIGIDITÉ. Poutre d'Euler sur appuis élastiques, chaque élément étant un
  ressort ; méthode des déplacements, deux degrés de liberté par nœud,
  ``numpy.linalg.solve``. Les efforts internes se déduisent des réactions
  par l'équilibre (pas de signe à deviner dans les matrices élémentaires).

Unités : bande efficace en mm, N, MPa ; poutre sur ressorts en m, kN, kN/m,
kN·m².
"""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------- bande efficace c
def bande_c(t_mm: float, f_MPa: float, p_MPa: float) -> float:
    """c = t·√(f / (3·p)) — EN 1993-1-8 §6.2.5 (le γM0 est dans ``f``)."""
    if p_MPa <= 0:
        return math.inf
    return t_mm * math.sqrt(f_MPa / (3.0 * p_MPa))


def aire_efficace(P_N: float, a_mm: float, b_mm: float, A_mm: float, B_mm: float,
                  t_mm: float, f_MPa: float, iterations: int = 50, tol: float = 1e-6) -> Dict:
    """Aire efficace (a + 2c) × (b + 2c) d'un plateau A × B d'épaisseur t
    sous une empreinte a × b chargée de P : itération sur c et p, c plafonné
    par le bord du plateau. Renvoie a_eff, b_eff, c, p (MPa), la part
    efficace du plateau et ``complet`` (le plateau travaille en entier)."""
    if a_mm > A_mm + 1e-9 or b_mm > B_mm + 1e-9:
        raise ValueError("Empreinte plus grande que le plateau.")
    c_max = min((A_mm - a_mm) / 2.0, (B_mm - b_mm) / 2.0)
    c = 0.0
    for _ in range(iterations):
        p = P_N / ((a_mm + 2 * c) * (b_mm + 2 * c))
        c_new = min(bande_c(t_mm, f_MPa, p), c_max)
        if abs(c_new - c) < tol:
            c = c_new
            break
        c = c_new
    p = P_N / ((a_mm + 2 * c) * (b_mm + 2 * c))
    a_eff, b_eff = a_mm + 2 * c, b_mm + 2 * c
    return {"a_eff": a_eff, "b_eff": b_eff, "c": c, "p": p, "c_max": c_max,
            "part_efficace": (a_eff * b_eff) / (A_mm * B_mm), "complet": c >= c_max - 1e-9}


# ---------------------------------------------------------------- propriétés d'une section en I (repli)
def section_I(h: float, b: float, tw: float, tf: float, r: float = 0.0) -> Dict:
    """Inertie et module élastique d'une section en I (mm) : ailes, âme et
    quatre congés de rayon r (aire (1 − π/4)·r², centre à 0,2234·r du coin ;
    inertie propre des congés négligée). Repli quand le catalogue ne donne
    pas Iy ; un test compare au catalogue."""
    I = b * h ** 3 / 12.0 - (b - tw) * (h - 2 * tf) ** 3 / 12.0
    if r > 0:
        a_c = (1.0 - math.pi / 4.0) * r * r
        d_c = (h / 2.0 - tf) - r * (10.0 - 3.0 * math.pi) / (12.0 - 3.0 * math.pi)
        I += 4.0 * a_c * d_c * d_c
    return {"Iy_mm4": I, "Wel_y_mm3": I / (h / 2.0)}


# ---------------------------------------------------------------- poutre sur appuis élastiques
def _k_elem(EI: float, l: float) -> np.ndarray:
    """Matrice de raideur d'un élément de poutre d'Euler (w1, θ1, w2, θ2)."""
    c = EI / l ** 3
    return c * np.array([[12, 6 * l, -12, 6 * l],
                         [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                         [-12, -6 * l, 12, -6 * l],
                         [6 * l, 2 * l * l, -6 * l, 4 * l * l]])


def poutre_sur_ressorts(EI: float, longueur: float, ressorts: Sequence[Tuple[float, float]],
                        charges: Sequence[Tuple[float, float]]) -> Dict:
    """Poutre de raideur ``EI`` (kN·m²) et de longueur ``longueur`` (m), libre
    à ses extrémités, tenue par des ressorts (abscisse s, raideur k en kN/m)
    et chargée par des charges ponctuelles (abscisse s, P en kN, positives
    vers le bas). Au moins deux ressorts d'abscisses distinctes.

    Renvoie les flèches aux nœuds, les réactions R_i = k_i·w_i (kN, vers le
    haut), le moment et le tranchant aux nœuds par l'équilibre (M positif
    en travée, V à droite de chaque nœud), et leurs maxima."""
    ressorts = [(float(s), float(k)) for s, k in ressorts]
    charges = [(float(s), float(P)) for s, P in charges]
    if len({round(s, 9) for s, _ in ressorts}) < 2:
        raise ValueError("Poutre sur ressorts : au moins deux ressorts d'abscisses distinctes.")
    for s, _ in ressorts + charges:
        if s < -1e-9 or s > longueur + 1e-9:
            raise ValueError(f"Abscisse {s} hors de la filière [0, {longueur}].")
    xs = sorted({0.0, longueur} | {round(s, 9) for s, _ in ressorts} | {round(s, 9) for s, _ in charges})
    n = len(xs)
    idx = {x: i for i, x in enumerate(xs)}
    K = np.zeros((2 * n, 2 * n))
    for i in range(n - 1):
        l = xs[i + 1] - xs[i]
        ke = _k_elem(EI, l)
        dofs = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
        for a in range(4):
            for b in range(4):
                K[dofs[a], dofs[b]] += ke[a, b]
    for s, k in ressorts:
        K[2 * idx[round(s, 9)], 2 * idx[round(s, 9)]] += k
    F = np.zeros(2 * n)
    for s, P in charges:
        F[2 * idx[round(s, 9)]] += P
    u = np.linalg.solve(K, F)
    w = [float(v) for v in u[0::2]]
    reactions = [(s, float(k * w[idx[round(s, 9)]])) for s, k in ressorts]
    # efforts par l'équilibre, de gauche à droite
    M, V = [], []
    for x in xs:
        v = sum(R for s, R in reactions if s < x - 1e-9) - sum(P for s, P in charges if s < x - 1e-9)
        m = sum(R * (x - s) for s, R in reactions if s < x - 1e-9) - sum(P * (x - s) for s, P in charges if s < x - 1e-9)
        v_droite = v + sum(R for s, R in reactions if abs(s - x) <= 1e-9) - sum(P for s, P in charges if abs(s - x) <= 1e-9)
        M.append(m)
        V.append((v, v_droite))
    somme_R = sum(R for _, R in reactions)
    somme_P = sum(P for _, P in charges)
    return {"x": xs, "w": w, "reactions": reactions, "M": M, "V": V,
            "M_max": max(abs(m) for m in M), "V_max": max(max(abs(a), abs(b)) for a, b in V),
            "somme_R": somme_R, "somme_P": somme_P, "ecart_equilibre": abs(somme_R - somme_P)}


def verifier_filiere(res: Dict, W_el_cm3: float, fy: float, gamma_M0: float, gamma_f: float) -> Dict:
    """Flexion de la filière : γF·M_max ≤ W_el·fy/γM0 (taux en service)."""
    M_Rd = W_el_cm3 * 1e3 * fy / gamma_M0 / 1e6      # kN·m
    return {"M_Ed_service": res["M_max"], "M_Rd": M_Rd, "taux": gamma_f * res["M_max"] / M_Rd if M_Rd > 0 else math.inf}

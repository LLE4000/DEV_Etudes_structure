# -*- coding: utf-8 -*-
"""Ce que la norme fixe pour le flambement par flexion, en une seule source.

EN 1993-1-1:2005 (+AC:2009) : Tableau 3.1 (f_y), Tableau 5.2 (classes en compression pure), §6.2.4 (N_c,Rd),
§6.3.1.2 (χ, φ, λ̄), Tableau 6.1 (α), Tableau 6.2 (choix de la courbe). EN 1993-1-5:2006 §4.4 (largeur efficace d'une
paroi comprimée, ψ = 1). ANB belge : γ_M0 = 1,0 ; γ_M1 = 1,0.

L'expression de χ est écrite comme celle de l'écran Streamlit existant (modules/choix_profile.py, ec3_buckling_chi),
opération par opération : les deux donnent le même flottant (tests/test_flambement_moteur.py).
"""
import math
from typing import Dict, Optional, Tuple

E_ACIER = 210000.0                       # N/mm², EN 1993-1-1 §3.2.6
GAMMA_M0 = 1.0                           # ANB belge
GAMMA_M1 = 1.0                           # ANB belge
GAMMA_MIN, GAMMA_MAX = 1.0, 1.5          # bornes admises à la saisie (un γ sous 1,0 n'est pas admis)

# ------------------------------------------------------------------ Tableau 3.1 : f_y selon l'épaisseur nominale
# (nuance) → (f_y pour t ≤ 40 mm, f_y pour 40 < t ≤ 80 mm), N/mm² ; laminés EN 10025-2/-3/-4 et creux EN 10210-1.
# Les creux formés à froid (EN 10219-1) n'ont que la colonne t ≤ 40 mm.
NUANCES: Dict[str, Tuple[float, float]] = {"S235": (235.0, 215.0), "S275": (275.0, 255.0), "S355": (355.0, 335.0),
                                           "S460": (460.0, 430.0)}
T_MAX_TABLEAU = 80.0                     # mm : au-delà, le Tableau 3.1 ne donne rien


class HorsDomaine(ValueError):
    """Une donnée que la norme (ou ce module) ne couvre pas : le calcul est refusé, avec ce message."""


def limite_elastique(nuance: str, t: float, froid: bool = False) -> float:
    """f_y du Tableau 3.1 pour l'épaisseur nominale ``t`` (mm) ; HorsDomaine au-delà du tableau."""
    if nuance not in NUANCES:
        raise HorsDomaine(f"nuance inconnue : {nuance}")
    if t <= 40.0:
        return NUANCES[nuance][0]
    if froid:
        raise HorsDomaine(f"creux formé à froid de {t:g} mm : le Tableau 3.1 (EN 10219) s'arrête à 40 mm")
    if t <= T_MAX_TABLEAU:
        return NUANCES[nuance][1]
    raise HorsDomaine(f"épaisseur {t:g} mm > 80 mm : hors du Tableau 3.1")


def epsilon(fy: float) -> float:
    """ε = √(235 / f_y) (Tableau 5.2)."""
    return math.sqrt(235.0 / fy)


def lambda_1(fy: float, E: float = E_ACIER) -> float:
    """λ₁ = π·√(E / f_y) = 93,9·ε (§6.3.1.3)."""
    return math.pi * math.sqrt(E / fy)


# ------------------------------------------------------------------ Tableau 5.2 : classes en compression pure
# (type de paroi) → limites de c/t des classes 1, 2, 3, en multiples de ε (ε² pour les tubes ronds)
LIMITES_CLASSE = {"interne": (33.0, 38.0, 42.0), "console": (9.0, 10.0, 14.0), "tube": (50.0, 70.0, 90.0)}
# cornières (Tableau 5.2, feuille 3) : classe 3 si h/t ≤ 15·ε et (b + h)/(2·t) ≤ 11,5·ε
CORNIERE_H_T, CORNIERE_BH_2T = 15.0, 11.5


def classe_paroi(type_paroi: str, c_t: float, eps: float) -> int:
    """La classe d'une paroi comprimée (1 à 4) ; ``c_t`` est c/t (d/t pour un tube rond)."""
    k = eps ** 2 if type_paroi == "tube" else eps
    for n, lim in enumerate(LIMITES_CLASSE[type_paroi], start=1):
        if c_t <= lim * k:
            return n
    return 4


def classe_corniere(h_t: float, bh_2t: float, eps: float) -> int:
    """Une cornière comprimée : 3 si elle tient les deux limites de la feuille 3, 4 sinon (le Tableau 5.2 ne donne
    pour elle qu'une limite de classe 3 ; les classes 1 à 3 ont la même résistance en compression)."""
    return 3 if (h_t <= CORNIERE_H_T * eps and bh_2t <= CORNIERE_BH_2T * eps) else 4


# ------------------------------------------------------------------ EN 1993-1-5 §4.4 : largeur efficace (ψ = 1)
K_SIGMA = {"interne": 4.0, "console": 0.43}          # Tableaux 4.1 et 4.2, compression uniforme (ψ = 1)
LAMBDA_P_LIM = {"interne": 0.673, "console": 0.748}  # 0,5 + √(0,085 − 0,055·ψ) ; 0,748


def elancement_paroi(type_paroi: str, c_t: float, eps: float) -> float:
    """λ̄_p = (b̄/t) / (28,4·ε·√k_σ)."""
    return c_t / (28.4 * eps * math.sqrt(K_SIGMA[type_paroi]))


def rho(type_paroi: str, lp: float) -> float:
    """ρ ≤ 1 : paroi interne ρ = (λ̄_p − 0,055·(3 + ψ))/λ̄_p² ; en console ρ = (λ̄_p − 0,188)/λ̄_p² (ψ = 1)."""
    if lp <= LAMBDA_P_LIM[type_paroi]:
        return 1.0
    if type_paroi == "interne":
        r = (lp - 0.055 * (3.0 + 1.0)) / lp ** 2
    else:
        r = (lp - 0.188) / lp ** 2
    return min(r, 1.0)


# ------------------------------------------------------------------ Tableaux 6.1 et 6.2 : courbes de flambement
ALPHA: Dict[str, float] = {"a0": 0.13, "a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}
COURBES = tuple(ALPHA)


def courbes(fabrication: str, nuance: str, h: Optional[float] = None, b: Optional[float] = None,
            tf: Optional[float] = None) -> Tuple[str, str, str]:
    """(courbe autour de y-y, courbe autour de z-z, la ligne du Tableau 6.2 retenue).

    ``fabrication`` : « lamine_I » (I et H laminés), « creux_chaud » (EN 10210), « creux_froid » (EN 10219),
    « U » (U, T, pleins), « L » (cornières)."""
    s460 = nuance == "S460"
    if fabrication == "lamine_I":
        if h / b > 1.2:
            if tf <= 40.0:
                return ("a0", "a0", "laminé en I, h/b > 1,2, t_f ≤ 40 mm, S460") if s460 else \
                       ("a", "b", "laminé en I, h/b > 1,2, t_f ≤ 40 mm")
            if tf <= 100.0:
                return ("a", "a", "laminé en I, h/b > 1,2, 40 < t_f ≤ 100 mm, S460") if s460 else \
                       ("b", "c", "laminé en I, h/b > 1,2, 40 < t_f ≤ 100 mm")
            # h/b > 1,2 et t_f > 100 mm : le Tableau 6.2 ne prévoit pas ce cas ; aucun laminé du catalogue ne l'atteint
            raise HorsDomaine("laminé en I, h/b > 1,2 et t_f > 100 mm : cas absent du Tableau 6.2")
        if tf <= 100.0:
            return ("a", "a", "laminé en I, h/b ≤ 1,2, t_f ≤ 100 mm, S460") if s460 else \
                   ("b", "c", "laminé en I, h/b ≤ 1,2, t_f ≤ 100 mm")
        return ("c", "c", "laminé en I, h/b ≤ 1,2, t_f > 100 mm, S460") if s460 else \
               ("d", "d", "laminé en I, h/b ≤ 1,2, t_f > 100 mm")
    if fabrication == "creux_chaud":
        return ("a0", "a0", "creux fini à chaud, S460") if s460 else ("a", "a", "creux fini à chaud")
    if fabrication == "creux_froid":
        return ("c", "c", "creux formé à froid")
    if fabrication == "U":
        return ("c", "c", "section en U")
    if fabrication == "L":
        return ("b", "b", "cornière")
    raise HorsDomaine(f"fabrication inconnue : {fabrication}")


# ------------------------------------------------------------------ §6.3.1.2 : χ
def phi_chi(lambdabar: float, alpha: float) -> Tuple[float, float]:
    """φ = 0,5·[1 + α·(λ̄ − 0,2) + λ̄²] ; χ = 1 / (φ + √(φ² − λ̄²)) ≤ 1 — les deux lignes d'ec3_buckling_chi
    (modules/choix_profile.py), mot pour mot."""
    phi = 0.5 * (1.0 + alpha * (lambdabar - 0.2) + lambdabar**2)
    chi = 1.0 / (phi + math.sqrt(max(phi**2 - lambdabar**2, 0.0)))
    chi = min(chi, 1.0)
    return phi, chi


def euler(A_mm2: float, I_mm4: float, L_mm: float, fy: float, E: float = E_ACIER) -> Tuple[float, float]:
    """N_cr = π²·E·I / L_cr² (N) et λ̄ = √(A·f_y / N_cr), écrits comme ec3_buckling_chi."""
    Ncr = (math.pi**2 * E * I_mm4) / (L_mm**2)
    lambdabar = math.sqrt((A_mm2 * fy) / Ncr)
    return Ncr, lambdabar

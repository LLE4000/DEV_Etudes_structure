# -*- coding: utf-8 -*-
"""Les critères : EN 1993-1-8 §4.5.3.2 (méthode directionnelle), §4.5.3.3
(méthode simplifiée), méthode ELS du bureau (contrainte admissible sur la
résultante), pièce d'une soudure bout à bout à pleine pénétration (§4.7.1).

Acier de référence d'un cordon : fu et βw de la pièce la plus faible
(``acier.resistances.fu_bw_cordon``) ; βw du Tableau 4.1 porté par la
bibliothèque des nuances. γM2 = 1,25 (ANB belge), modifiable.
Unités : N/mm², N/mm, mm.
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional

from acier.resistances import fvw_d

G_KG = 9.81            # N par kg (P-2 : 10 en option — la pratique qui donne « 138 »)
GAMMA_M2 = 1.25        # ANB belge
GAMMA_F_EQUIV = 1.5    # rapport ELU / ELS pour l'équivalent Eurocode de la règle du bureau (P-3)
K_BUREAU = 100.0       # kg par mm de gorge et par cm de cordon (P-1)
EUROCODE, ELS = "eurocode", "els"
DIRECTIONNELLE, SIMPLIFIEE = "directionnelle", "simplifiee"


@dataclass
class Methode:
    mode: str = EUROCODE             # « eurocode » (efforts ELU) | « els » (efforts de service)
    ec: str = DIRECTIONNELLE         # méthode Eurocode : directionnelle | simplifiee
    gamma_M2: float = GAMMA_M2
    k: float = K_BUREAU              # kg/(mm·cm)
    g: float = G_KG                  # N/kg
    gamma_F: float = GAMMA_F_EQUIV


def tau_adm_bureau(k: float, g: float = G_KG) -> float:
    """k kg/(mm·cm) → contrainte admissible sur la gorge (N/mm²) : k·g/10
    (1 cm de cordon = 10 mm). 100 kg/(mm·cm) → 98,1 N/mm² (g = 9,81)."""
    return k * g / 10.0


def k_equivalent_eurocode(fu: float, bw: float, gamma_M2: float = GAMMA_M2, gamma_F: float = GAMMA_F_EQUIV,
                         g: float = G_KG) -> Dict[str, float]:
    """Ce que l'Eurocode donne au niveau de service : f_vw,d/γ_F en N/mm² et en
    kg/(mm·cm) (S235 : 138,6 N/mm², 141 kg/(mm·cm) à 9,81 N/kg)."""
    tau = fvw_d(fu, bw, gamma_M2) / gamma_F
    return {"tau_N_mm2": tau, "k_kg_mm_cm": tau * 10.0 / g, "f_vw_d": fvw_d(fu, bw, gamma_M2)}


# ------------------------------------------------------------------ Eurocode
def directionnelle(sigma_perp: float, tau_perp: float, tau_par: float, fu: float, bw: float,
                   gamma_M2: float = GAMMA_M2) -> Dict[str, float]:
    """§4.5.3.2 : √(σ⊥² + 3·(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) et σ⊥ ≤ 0,9·fu/γM2."""
    sc = math.sqrt(sigma_perp ** 2 + 3.0 * (tau_perp ** 2 + tau_par ** 2))
    lim1 = fu / (bw * gamma_M2)
    lim2 = 0.9 * fu / gamma_M2
    t1 = sc / lim1
    t2 = abs(sigma_perp) / lim2
    return {"sigma_c": sc, "lim_c": lim1, "taux_c": t1, "lim_perp": lim2, "taux_perp": t2, "taux": max(t1, t2),
            "critere": "σ_c ≤ fu/(βw·γM2)" if t1 >= t2 else "σ⊥ ≤ 0,9·fu/γM2"}


def simplifiee(f: float, a: float, fu: float, bw: float, gamma_M2: float = GAMMA_M2) -> Dict[str, float]:
    """§4.5.3.3 : |f| ≤ F_w,Rd = f_vw,d·a, f_vw,d = fu/(√3·βw·γM2)."""
    fv = fvw_d(fu, bw, gamma_M2)
    Fw = fv * a
    return {"f_vw_d": fv, "F_w_Rd": Fw, "taux": f / Fw if Fw > 0 else math.inf, "critere": "|f| ≤ f_vw,d·a"}


# ------------------------------------------------------------------ bureau
def els_bureau(f: float, a: float, k: float = K_BUREAU, g: float = G_KG) -> Dict[str, float]:
    """Méthode ELS du bureau : résultante de service |f| ≤ (k·g/10)·a."""
    tau = tau_adm_bureau(k, g)
    Fadm = tau * a
    return {"tau_adm": tau, "F_adm": Fadm, "taux": f / Fadm if Fadm > 0 else math.inf, "critere": "|f| ≤ k·a (bureau)"}


# ------------------------------------------------------------------ bout à bout, pleine pénétration
def piece_pleine_penetration(N: float, V: float, M: float, b: float, t: float, fy: float,
                             gamma_M0: float = 1.0) -> Dict[str, float]:
    """§4.7.1 : la soudure vaut la pièce la plus faible ; vérification élastique
    de la pièce (b × t) : σ = N/(b·t) + 6·M/(b·t²), τ = V/(b·t),
    √(σ² + 3τ²) ≤ fy/γM0."""
    A = b * t
    sig = N / A + 6.0 * M / (b * t * t)
    tau = V / A
    sc = math.sqrt(sig ** 2 + 3.0 * tau ** 2)
    lim = fy / gamma_M0
    return {"sigma": sig, "tau": tau, "sigma_c": sc, "lim": lim, "taux": sc / lim, "critere": "pièce : √(σ² + 3τ²) ≤ fy/γM0"}


def evaluer(point: Dict, methode: Methode, fu: float, bw: float) -> Dict[str, float]:
    """Le taux d'un point de cordon selon la méthode choisie, plus les deux
    autres lectures pour comparaison (directionnelle, simplifiée, bureau)."""
    a = point["cordon"].a
    d = directionnelle(point["sigma_perp"], point["tau_perp"], point["tau_par"], fu, bw, methode.gamma_M2)
    s = simplifiee(point["f"], a, fu, bw, methode.gamma_M2)
    e = els_bureau(point["f"], a, methode.k, methode.g)
    if methode.mode == ELS:
        actif = e
    elif methode.ec == SIMPLIFIEE:
        actif = s
    else:
        actif = d
    return {"taux": actif["taux"], "critere": actif["critere"], "directionnelle": d, "simplifiee": s, "bureau": e}

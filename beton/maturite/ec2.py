# -*- coding: utf-8 -*-
"""Les formules du module « Âge béton », RECOPIÉES de l'écran Streamlit ``modules/age_beton.py`` (v3.2), lignes 61 à
166, 319 à 324 et 434 à 442 : même code, même ordre d'opérations, mêmes appels numpy — les flottants sont identiques au
bit près (``tests/test_maturite_moteur.py``). Seuls les NOMS sont passés au français ; la correspondance est écrite à
côté de chaque fonction et dans ``docs/beton/MATURITE_MOTEUR.md``.

Base normative (en-tête du module) : EN 1992-1-1 §3.1.2 (éq. 3.1, 3.2), annexe B (éq. B.10) ; s = 0,20 (R) / 0,25 (N)
/ 0,38 (S) ; maturité t_e = Σ Δt_i · exp(−E_a/R · (1/T − 1/293,15)).

numpy est permis (il est dans Pyodide) et NÉCESSAIRE à la parité : np.exp et math.exp peuvent différer au dernier bit.
"""
import math

import numpy as np

# ------------------------------------------------------------------ les données du module (lignes 61 à 76, 185 à 187)
CLASSES = ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60"]

CIMENTS = {
    "Classe R — durcissement rapide": 0.20,
    "Classe N — durcissement normal": 0.25,
    "Classe S — durcissement lent":   0.38,
}
CIMENT_DETAIL = {
    "Classe R — durcissement rapide": "CEM 42,5R · CEM 52,5N · CEM 52,5R",
    "Classe N — durcissement normal": "CEM 32,5R · CEM 42,5N",
    "Classe S — durcissement lent":   "CEM 32,5N",
}

R_GAZ = 8.314
EA_EC2 = 4000.0 * R_GAZ
T_REF_K = 293.15

HORIZON_PREVISION_J = 16        # horizon de l'API de prévision (Open-Meteo), cité dans les messages
MARGE_PREVISION_C = 3.0         # variante défavorable : °C retirés aux jours prévus


# ------------------------------------------------------------------ les calculs (fonctions pures du module)
def fck_classe(label: str) -> int:
    """« C30/37 » → 30. Module : ``parse_fck``."""
    return int(label.split("/")[0].replace("C", ""))


def beta_cc(t_days_equiv, s: float):
    """β_cc(t_e) = exp(s·[1 − √(28/t_e)]), t_e ramené à 1e-6 j au moins. Module : ``beta_cc``."""
    t = np.maximum(np.asarray(t_days_equiv, dtype=float), 1e-6)
    return np.exp(s * (1.0 - np.sqrt(28.0 / t)))


def fck_age_equivalent(fck28: float, s: float, t_days_equiv):
    """f_ck(t) = β_cc(t_e)·f_cm − 8 pour t_e < 28 j, f_ck au-delà (f_cm = f_ck + 8). Module : ``fck_of_age_equiv``."""
    fcm = fck28 + 8.0
    t_e = np.asarray(t_days_equiv, dtype=float)
    val = beta_cc(t_e, s) * fcm - 8.0
    return np.where(t_e < 28.0, val, float(fck28))


def age_equivalent(t_days_real, T_celsius: float, Ea: float = EA_EC2):
    """t_e = t·exp(−E_a/R·[1/(T + 273,15) − 1/293,15]) à température constante. Module : ``age_equiv``."""
    T_abs = float(T_celsius) + 273.15
    factor = math.exp((-Ea / R_GAZ) * (1.0 / T_abs - 1.0 / T_REF_K))
    return np.asarray(t_days_real, dtype=float) * factor


def facteur_maturite(T_celsius: float, Ea: float = EA_EC2) -> float:
    """exp(−E_a/R·[1/(T + 273,15) − 1/293,15]) : les jours équivalents à 20 °C d'un jour à T. Module : ``maturity_factor``."""
    T_abs = float(T_celsius) + 273.15
    return math.exp((-Ea / R_GAZ) * (1.0 / T_abs - 1.0 / T_REF_K))


def age_equivalent_serie(temps_celsius, Ea: float = EA_EC2, dt_days: float = 1.0):
    """t_e = Σ Δt·facteur(T_i) sur un relevé journalier. Module : ``age_equiv_from_series``."""
    arr = np.asarray(temps_celsius, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sum([facteur_maturite(T, Ea) for T in arr]) * dt_days)


def fck_cible_pourcentage(fck28: float, pct: float) -> float:
    """f_cible = f_ck·p/100. Module : ``fck_target_from_pct``."""
    return fck28 * float(pct) / 100.0


def temps_reel_pour_cible(fck28: float, s: float, target_MPa: float,
                          T_celsius: float, Ea: float = EA_EC2,
                          tmax: float = 90.0, tol: float = 1e-3):
    """Le temps réel (j) où f_ck(t) atteint la cible, à température CONSTANTE : dichotomie sur [0,05 ; 90] j, 80 pas,
    arrêt dès |f − cible| < 0,001 MPa ; None si la cible est ≤ 0, dépasse f_ck ou n'est pas atteinte en 90 j.
    Module : ``t_real_for_target``."""
    if target_MPa <= 0 or target_MPa > fck28 + 1e-9:
        return None
    lo, hi = 0.05, float(tmax)
    f_hi = float(fck_age_equivalent(fck28, s, age_equivalent(hi, T_celsius, Ea)))
    if f_hi + 1e-6 < target_MPa:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = float(fck_age_equivalent(fck28, s, age_equivalent(mid, T_celsius, Ea)))
        if abs(f_mid - target_MPa) < tol:
            return mid
        if f_mid < target_MPa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def classe_min_pour_delai(target_MPa: float, s: float, T_celsius: float,
                          delai_jours: float, Ea: float = EA_EC2):
    """La première classe (dans l'ordre de CLASSES) dont f_ck(délai) + 1e-6 ≥ cible : (classe, f_ck, f_ck atteint) ;
    None si aucune. Module : ``classe_min_pour_delai``."""
    t_e = float(age_equivalent(delai_jours, T_celsius, Ea))
    for label in CLASSES:
        fck28 = fck_classe(label)
        fck_atteint = float(fck_age_equivalent(fck28, s, t_e))
        if fck_atteint + 1e-6 >= target_MPa:
            return label, fck28, fck_atteint
    return None


def classe_estimee_depuis_mesure(res_mesuree: float, age_jours: float, s: float,
                                 T_celsius: float, Ea: float = EA_EC2):
    """La classe estimée d'après un essai : f_ck,28 estimé = (f_mes + 8)/β_cc(t_e) − 8 (f_mes si t_e ≥ 28 j), puis la
    classe LA PLUS PROCHE (argmin |f_ck − estimé|). Module : ``classe_estimee_depuis_mesure``."""
    t_e = float(age_equivalent(age_jours, T_celsius, Ea))
    if t_e >= 28.0:
        fck28_est = res_mesuree
    else:
        b = float(beta_cc(t_e, s))
        fck28_est = (res_mesuree + 8.0) / b - 8.0
    fcks = [fck_classe(c) for c in CLASSES]
    idx = int(np.argmin([abs(f - fck28_est) for f in fcks]))
    classe = CLASSES[idx]
    fck_classe_ = fcks[idx]
    ecart = (fck28_est - fck_classe_) / fck_classe_ * 100.0 if fck_classe_ else 0.0
    return {"t_e": t_e, "fck28_est": fck28_est, "classe": classe,
            "fck_classe": fck_classe_, "ecart_pct": ecart}


def profil_defavorable(temps, prevision, marge=MARGE_PREVISION_C):
    """La variante DÉFAVORABLE d'un relevé : jours mesurés intacts, jours prévus (et prolongés) abaissés de ``marge``
    °C. Module : ``profil_defavorable``."""
    return [t - marge if (p and not math.isnan(t)) else t
            for t, p in zip(temps, prevision)]


def age_equivalent_point(t_sel, T, profil_temps, Ea):
    """t_e à l'âge t_sel : la somme jour par jour sur le relevé s'il existe (au plus ses n premiers jours, le reste
    d'un jour fractionnaire au facteur du jour suivant), sinon à T constante. Module : ``_age_equiv_point``."""
    if profil_temps:
        n = min(int(round(t_sel)), len(profil_temps))
        t_e = age_equivalent_serie(profil_temps[:n], Ea)
        reste = t_sel - n
        if reste > 0 and n < len(profil_temps):
            t_e += facteur_maturite(profil_temps[n], Ea) * reste
        return t_e
    return float(age_equivalent(t_sel, T, Ea))

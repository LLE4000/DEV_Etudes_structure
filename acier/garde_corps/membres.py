# -*- coding: utf-8 -*-
"""La mécanique des éléments d'un garde-corps, aux « vieilles formules » du bureau :

- une CONSOLE (montant encastré en pied, console de fixation au mur) chargée en tête par
  deux efforts transversaux perpendiculaires P₁ (sens de h, axe fort) et P₂ (sens de b) et
  un effort axial N, de longueur L : M₁ = P₁·L, M₂ = P₂·L, V = P, flèche P·L³/(3EI) ;
- une POUTRE SUR DEUX APPUIS (lisse haute, lisse intermédiaire, barreau) de portée L sous
  une charge répartie q et une charge ponctuelle Q à mi-portée : M = qL²/8 + QL/4 (ou le
  maximum des deux cas, s'ils ne se combinent pas), V = qL/2 + Q/2, flèche 5qL⁴/(384EI) +
  QL³/(48EI) ; deux plans de flexion (horizontal et vertical) ;
- la VÉRIFICATION d'une section : σ = M₁/W_el,y + M₂/W_el,z + N/A ≤ σ_lim, τ = V/A_v ≤ τ_lim,
  σ_eq = √(σ² + 3τ²) ≤ σ_lim (contrainte équivalente, composée au même point : du côté sûr),
  flèche ≤ L/x (ou une valeur en mm), chaque critère donnant un taux ; le plus grand gouverne.

Limites : vérification à l'ELS — contraintes admissibles sur efforts de service, DÉRIVÉES de la
nuance : σ_adm = f_y/γ_adm (γ_adm = 1,5 : 156,7 N/mm² pour le S235), τ_adm = σ_adm/√3 (90,5),
σ_eq ≤ σ_adm (règle du bureau, 05/11/2026 — les anciennes constantes 160 et 90 étaient
fausses) ; Eurocode — efforts de calcul (× γ_Q), σ ≤ f_y/γ_M0 (élastique, W_el : du côté sûr
pour les classes 1 et 2), τ ≤ f_y/(√3·γ_M0), σ_eq ≤ f_y/γ_M0 (EN 1993-1-1 §6.2.1(5)), flèche
sous efforts de service. Unités internes : N, mm, N·mm ; les sections viennent de
``bibliotheque`` (cm², cm⁴, cm³).
"""
import math
from typing import Dict, Optional

E_ACIER = 210000.0        # N/mm²
GAMMA_ADM = 1.5           # σ_adm = f_y/γ_adm, τ_adm = σ_adm/√3 (vérification à l'ELS)
GAMMA_Q = 1.5
GAMMA_M0 = 1.0
SQRT3 = math.sqrt(3.0)
# Flèche admissible par défaut : L/100 — l'ancien module proposait L/200, que le montant courant
# (CHS 48,3 × 3,2 à 1,20 m, H = 1,10 m, 0,5 kN/m) ne tient pas ; H/100 est la pratique des
# garde-corps. C'est un choix du bureau, modifiable (L/x ou une valeur en mm).
FLECHE_X_DEFAUT = 100.0
# Les seuils du statut — et du CODE COULEUR des taux, tranché par le bureau le 06/11/2026 : rouge au-dessus de 100 %, vert foncé de
# 95 à 100 % (« conforme, attention » : 0,95 inclus), vert pâle en dessous. L'écran et la note n'ajoutent que les couleurs (source unique).
SEUIL_ATTENTION, SEUIL_NOK = 0.95, 1.0
OK, ATTENTION, NOK, REFUS = "ok", "attention", "nok", "refus"
ELS, EUROCODE = "els", "eurocode"


def statut(taux: Optional[float]) -> str:
    if taux is None:
        return REFUS
    return NOK if taux > SEUIL_NOK else ATTENTION if taux >= SEUIL_ATTENTION else OK


def methode_defaut() -> Dict:
    return {"mode": ELS, "gamma_adm": GAMMA_ADM, "fleche_mode": "L/x", "fleche_val": FLECHE_X_DEFAUT,
            "gamma_Q": GAMMA_Q, "gamma_M0": GAMMA_M0, "gamma_F": 1.5, "nuance": "S235", "E": E_ACIER,
            "k": 100.0, "g": 9.81, "gamma_M2": 1.25, "ec": "directionnelle", "comb": "somme"}


def gamma_adm(meth: Dict) -> float:
    return float(meth.get("gamma_adm") or GAMMA_ADM)


def limites(meth: Dict, fy: float) -> Dict:
    """σ_lim, τ_lim (N/mm²) et le coefficient de charge γ pour les résistances, selon la méthode —
    les deux dérivées de la nuance : f_y/γ_adm et f_y/(√3·γ_adm) à l'ELS, f_y/γ_M0 et f_y/(√3·γ_M0)
    en Eurocode ; la contrainte équivalente σ_eq a la limite de σ."""
    if meth.get("mode") == EUROCODE:
        gM0 = float(meth.get("gamma_M0") or GAMMA_M0)
        return {"sigma": fy / gM0, "tau": fy / (SQRT3 * gM0), "gamma": float(meth.get("gamma_Q") or GAMMA_Q), "coef": gM0,
                "texte_sigma": f"f_y/γ_M0 = {fy:g}/{gM0:g}", "texte_tau": "f_y/(√3·γ_M0)", "nature": "efforts de calcul (× γ_Q)"}
    ga = gamma_adm(meth)
    return {"sigma": fy / ga, "tau": fy / (SQRT3 * ga), "gamma": 1.0, "coef": ga,
            "texte_sigma": f"σ_adm = f_y/{ga:g}", "texte_tau": "τ_adm = σ_adm/√3", "nature": "efforts de service"}


def fleche_limite(meth: Dict, L: float) -> float:
    """La flèche admissible (mm) d'une longueur L (mm) : L/x ou une valeur en mm."""
    val = float(meth.get("fleche_val") or FLECHE_X_DEFAUT)
    if meth.get("fleche_mode") == "mm":
        return max(val, 0.1)
    return L / max(val, 1.0)


# ------------------------------------------------------------------ mécanique
def console(P1: float, P2: float, N: float, L: float, E: float, Iy: float, Iz: float) -> Dict:
    """Console de longueur L (mm) chargée en tête : P₁, P₂ (N) transversaux, N axial (N, > 0
    compression) ; Iy, Iz en mm⁴. Moments en N·mm, flèches en mm (résultante des deux plans)."""
    M1, M2 = P1 * L, P2 * L
    f1 = P1 * L ** 3 / (3.0 * E * Iy) if Iy > 0 else float("inf")
    f2 = P2 * L ** 3 / (3.0 * E * Iz) if Iz > 0 else 0.0
    return {"M1": M1, "M2": M2, "V1": P1, "V2": P2, "N": N, "f1": f1, "f2": f2, "f": math.hypot(f1, f2),
            "formule_M": "M = P·H", "formule_V": "V = P", "formule_f": "f = P·H³/(3·E·I)"}


# Poutre CONTINUE à travées égales sous charge répartie (tables classiques des poutres continues,
# EI constant) : (moment maximal / qL², effort tranchant maximal / qL, flèche maximale × EI / qL⁴)
# — deux travées : appui central ; trois travées : premier appui intermédiaire et travée de rive ;
# quatre travées et plus : les valeurs de quatre travées, du côté sûr.
COEF_CONTINUE = {2: (0.125, 0.625, 0.00541), 3: (0.100, 0.600, 0.0069), 4: (0.107, 0.607, 0.0065)}
APPUIS = {"deux": "sur deux appuis (une travée)", "continue": "continue sur plusieurs appuis"}


def coef_continue(n: int):
    return COEF_CONTINUE[2 if n <= 2 else (3 if n == 3 else 4)]


def poutre_continue(q: float, Q: float, L: float, n: int, E: float, I: float) -> Dict:
    """Poutre continue sur n travées égales de portée L (mm) sous q (N/mm) : M, V et f par les
    coefficients des tables ; une charge ponctuelle Q à mi-portée garde ses valeurs sur deux
    appuis (QL/4, QL³/48EI), du côté sûr."""
    if n <= 1:
        return poutre(q, Q, L, E, I, "somme")
    kM, kV, kf = coef_continue(n)
    Mq, MQ = kM * q * L ** 2, Q * L / 4.0
    Vq, VQ = kV * q * L, Q / 2.0
    fq = kf * q * L ** 4 / (E * I) if I > 0 else float("inf")
    fQ = Q * L ** 3 / (48.0 * E * I) if I > 0 else (float("inf") if Q else 0.0)
    return {"M": Mq + MQ, "V": Vq + VQ, "f": fq + fQ, "Mq": Mq, "MQ": MQ, "fq": fq, "fQ": fQ, "comb": "somme", "n": n,
            "kM": kM, "kV": kV, "kf": kf, "continue": True}


def poutre(q: float, Q: float, L: float, E: float, I: float, comb: str = "somme") -> Dict:
    """Poutre sur deux appuis (L en mm) sous q (N/mm) répartie et Q (N) à mi-portée ; ``comb``
    « somme » (les deux agissent ensemble) ou « max » (deux cas séparés)."""
    Mq, MQ = q * L ** 2 / 8.0, Q * L / 4.0
    Vq, VQ = q * L / 2.0, Q / 2.0
    fq = 5.0 * q * L ** 4 / (384.0 * E * I) if I > 0 else float("inf")
    fQ = Q * L ** 3 / (48.0 * E * I) if I > 0 else (float("inf") if Q else 0.0)
    if comb == "max":
        return {"M": max(Mq, MQ), "V": max(Vq, VQ), "f": max(fq, fQ), "Mq": Mq, "MQ": MQ, "fq": fq, "fQ": fQ, "comb": "max"}
    return {"M": Mq + MQ, "V": Vq + VQ, "f": fq + fQ, "Mq": Mq, "MQ": MQ, "fq": fq, "fQ": fQ, "comb": "somme"}


# ------------------------------------------------------------------ vérification
def verifier_section(sec: Dict, M1: float, M2: float, N: float, V1: float, V2: float, f: float, L: float, meth: Dict,
                     fy: float, gamma: Optional[float] = None) -> Dict:
    """Contraintes et taux d'une section sous M₁ (axe fort, N·mm), M₂ (axe faible), N (N,
    compression > 0), V₁ (cisaillement dans le sens de h, N), V₂ (sens de b), flèche f (mm,
    efforts de service) sur la longueur L. ``gamma`` multiplie M, N, V pour les résistances
    (γ_Q en Eurocode, 1 en ELS) — la flèche reste de service."""
    lim = limites(meth, fy)
    g = lim["gamma"] if gamma is None else gamma
    Wy, Wz, A = sec["Wel_y"] * 1e3, sec["Wel_z"] * 1e3, sec["A"] * 1e2
    Avz, Avy = sec["Avz"] * 1e2, sec["Avy"] * 1e2
    s_M1 = g * abs(M1) / Wy if Wy > 0 else float("inf")
    s_M2 = g * abs(M2) / Wz if Wz > 0 else 0.0
    s_N = g * abs(N) / A if A > 0 else 0.0
    sigma = s_M1 + s_M2 + s_N
    t1 = g * abs(V1) / Avz if Avz > 0 else 0.0
    t2 = g * abs(V2) / Avy if Avy > 0 else 0.0
    tau = max(t1, t2)
    sigma_eq = math.sqrt(sigma ** 2 + 3.0 * tau ** 2) if math.isfinite(sigma) else float("inf")
    f_lim = fleche_limite(meth, L)
    taux = {"sigma": sigma / lim["sigma"], "tau": tau / lim["tau"], "sigma_eq": sigma_eq / lim["sigma"],
            "fleche": f / f_lim if f_lim > 0 else float("inf")}
    # le critère qui gouverne : σ_eq ≥ σ toujours — quand les contraintes gouvernent, c'est la
    # contrainte équivalente qui est nommée (σ seule n'est jamais la pire)
    pire = max(taux, key=taux.get)
    return {"M1": M1, "M2": M2, "N": N, "V1": V1, "V2": V2, "gamma": g,
            "sigma_M1": s_M1, "sigma_M2": s_M2, "sigma_N": s_N, "sigma": sigma, "tau": tau, "sigma_eq": sigma_eq, "fleche": f, "fleche_lim": f_lim,
            "lim_sigma": lim["sigma"], "lim_tau": lim["tau"], "coef_lim": lim["coef"], "taux": taux, "taux_max": taux[pire], "critere": pire,
            "statut": statut(taux[pire]), "Wy": Wy, "Wz": Wz, "A": A, "Av": Avz if t1 >= t2 else Avy, "L": L}


def critere_texte(cle: str) -> str:
    return {"sigma": "contrainte σ", "tau": "cisaillement τ", "sigma_eq": "σ équivalente", "fleche": "flèche"}.get(cle, cle)

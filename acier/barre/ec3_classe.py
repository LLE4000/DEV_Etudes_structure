# -*- coding: utf-8 -*-
"""EN 1993-1-1 § 5.5 : la CLASSE d'une section sous N, M_y et M_z (tableau 5.2) ; EN 1993-1-5 § 4.4 : sa section
EFFICACE quand elle est de classe 4. Sections à deux axes de symétrie : I et H, tubes carrés et rectangulaires, tubes
ronds. docs/acier/BARRES_ACIER.md § 5.2 et 5.3.

LA CLASSE dépend de l'effort : elle se refait pour chaque combinaison. Chaque paroi reçoit son ÉTAT :

  « compression »  comprimée uniformément — interne 33ε / 38ε / 42ε ; en console 9ε / 10ε / 14ε ; tube rond
                   50ε² / 70ε² / 90ε² (compression ou flexion) ;
  « flexion »      une âme fléchie sans effort normal — 72ε / 83ε / 124ε ;
  « flexion composée »  une âme fléchie ET comprimée (ou tendue) :
        α = 0,5 + N_Ed / (2·c·t_w·f_y)  borné à [0 ; 1]   (l'axe neutre plastique ; deux âmes pour un tube : 2·t)
        ψ = 2·N_Ed / (A·f_y) − 1        borné à 1         (le rapport des contraintes élastiques, fibre comprimée à f_y)
        classes 1 et 2 : 396ε/(13α − 1) et 456ε/(13α − 1) si α > 0,5 ; 36ε/α et 41,5ε/α sinon
        classe 3       : 42ε/(0,67 + 0,33ψ) si ψ > −1 ; 62ε·(1 − ψ)·√(−ψ) sinon ;
  « tendue »       aucune limite.

DU CÔTÉ SÛR : une semelle — ou la paroi d'un tube qui sert de semelle — est classée en compression UNIFORME dès qu'un
moment existe, même sous un gradient (flexion d'axe faible, flexion déviée) ; l'âme d'un I sous M_z et N l'est aussi.

LA SECTION EFFICACE, selon l'EN 1993-1-5 § 4.3(3) et (4) : A_eff sous la compression SEULE, W_eff sous la flexion
SEULE. Deux axes de symétrie : A_eff n'a pas d'excentricité (e_N = 0). Pour W_eff : la membrure comprimée d'abord, puis
l'âme avec le ψ que donne l'axe neutre déplacé par la membrure réduite (§ 4.4(3)). Les parties inefficaces sont
RETIRÉES de la section brute du catalogue (aire, inertie — congés compris) ; W_eff se rapporte à la fibre comprimée
extrême de la section BRUTE (du côté sûr quand le bout d'une console est retiré).

    λ̄_p = (c/t) / (28,4·ε·√k_σ)
    paroi interne : ρ = (λ̄_p − 0,055·(3 + ψ)) / λ̄_p² ≤ 1   au-delà de 0,5 + √(0,085 − 0,055·ψ)        (tableau 4.1)
    console       : ρ = (λ̄_p − 0,188) / λ̄_p² ≤ 1           au-delà de 0,748                          (tableau 4.2)

Tube rond de classe 4 : EN 1993-1-6, non couvert.
"""
import math
from typing import Dict, List, Optional, Tuple

from acier.flambement import normes as NO

from . import section as SE
from .resultat import NonCouvert, Refus

# ------------------------------------------------------------------ tableau 5.2
LIMITES_COMPRESSION = NO.LIMITES_CLASSE          # interne 33/38/42 ; console 9/10/14 ; tube 50/70/90 (une seule source)
LIMITES_FLEXION = (72.0, 83.0, 124.0)            # paroi interne fléchie
FC_HAUT = (396.0, 456.0)                         # flexion composée, α > 0,5 : k·ε/(13α − 1)
FC_BAS = (36.0, 41.5)                            # flexion composée, α ≤ 0,5 : k·ε/α
B_MOINS_3T = 3.0                                 # paroi d'un tube carré ou rectangulaire : b − 3·t (EN 1993-1-5 § 4.4(2))
COMPRESSION, FLEXION, FLEXION_COMPOSEE, TENDUE = "compression", "flexion", "flexion_composee", "tendue"
# ------------------------------------------------------------------ EN 1993-1-5, tableaux 4.1 et 4.2
K_SIGMA_CONSOLE = 0.43                           # console comprimée uniformément (ψ = 1)
LAMBDA_P_CONSOLE = 0.748
BE1_PSI_NEGATIF = 0.4                            # ψ < 0 : b_e1 = 0,4·b_eff (côté comprimé), b_e2 = 0,6·b_eff (côté axe neutre)


def parois(s: Dict) -> List[Dict]:
    """Les parois de la section : [{"role", "nom", "type" (interne | console | tube), "c", "t" (mm), "c_t", "n"}].
    I et H : c/t du catalogue (semelle (b − t_w − 2r)/2, âme h − 2·t_f − 2r) ; tubes carrés et rectangulaires : b − 3·t
    et h − 3·t ; tube rond : d/t."""
    forme = s["forme"]
    if forme == "I":
        if not all(s.get(k) for k in ("cf_tf", "cw_tw", "tf", "tw")):
            raise Refus("h, b, t_w, t_f manquent : la classe d'un I en dépend")
        return [{"role": "semelle", "nom": "semelle", "type": "console", "t": s["tf"], "c_t": s["cf_tf"], "c": s["cf_tf"] * s["tf"], "n": 4},
                {"role": "ame", "nom": "âme", "type": "interne", "t": s["tw"], "c_t": s["cw_tw"], "c": s["cw_tw"] * s["tw"], "n": 1}]
    if forme in ("tube_carre", "tube_rect"):
        if not all(s.get(k) for k in ("h", "b", "t")):
            raise Refus("h, b, t manquent : la classe d'un tube en dépend")
        t = s["t"]
        cb, ch = s["b"] - B_MOINS_3T * t, s["h"] - B_MOINS_3T * t
        return [{"role": "b", "nom": "paroi b", "type": "interne", "t": t, "c_t": cb / t, "c": cb, "n": 2},
                {"role": "h", "nom": "paroi h", "type": "interne", "t": t, "c_t": ch / t, "c": ch, "n": 2}]
    if forme == "tube_rond":
        if not all(s.get(k) for k in ("d", "t")):
            raise Refus("d, t manquent : la classe d'un tube rond en dépend")
        return [{"role": "tube", "nom": "paroi", "type": "tube", "t": s["t"], "c_t": s["d"] / s["t"], "c": s["d"], "n": 1}]
    raise NonCouvert("Cornière, U ou té : hors du module (compression simple : module Flambement).")


def alpha_psi(N: float, c: float, t_ames: float, A: float, fy: float) -> Tuple[float, float]:
    """(α, ψ) d'une âme fléchie sous l'effort normal ``N`` (N, compression > 0) ; ``t_ames`` : l'épaisseur cumulée des
    âmes (t_w ; 2·t pour un tube)."""
    alpha = min(max(0.5 + N / (2.0 * c * t_ames * fy), 0.0), 1.0)
    psi = min(2.0 * N / (A * fy) - 1.0, 1.0)
    return alpha, psi


def limites_flexion_composee(alpha: float, psi: float, eps: float) -> Tuple[List[Optional[float]], List[Tuple[str, Optional[float]]]]:
    """Les limites c/t des classes 1, 2, 3 d'une paroi interne fléchie et comprimée, et la règle de chacune (gabarit de
    acier/barre/formules.py, coefficient). None : pas de limite (l'âme est tendue sur toute sa hauteur)."""
    if alpha <= 0.0:
        l12: List[Optional[float]] = [None, None]
        r12: List[Tuple[str, Optional[float]]] = [("", None), ("", None)]
    elif alpha > 0.5:
        l12 = [k * eps / (13.0 * alpha - 1.0) for k in FC_HAUT]
        r12 = [("e_lim12_haut", k) for k in FC_HAUT]
    else:
        l12 = [k * eps / alpha for k in FC_BAS]
        r12 = [("e_lim12_bas", k) for k in FC_BAS]
    if psi > -1.0:
        l3, r3 = 42.0 * eps / (0.67 + 0.33 * psi), ("e_lim3_haut", None)
    else:
        l3, r3 = 62.0 * eps * (1.0 - psi) * math.sqrt(-psi), ("e_lim3_bas", None)
    return l12 + [l3], r12 + [r3]


def _classe(c_t: float, limites: List[Optional[float]]) -> int:
    for k, lim in enumerate(limites, start=1):
        if lim is None or c_t <= lim:
            return k
    return 4


def classer(s: Dict, fy: float, N: float = 0.0, My: float = 0.0, Mz: float = 0.0) -> Dict:
    """La classe de la section sous ``N`` (N, compression > 0), ``My`` et ``Mz`` (seule leur présence compte) :
    {"classe", "epsilon", "parois": [{…, "etat", "alpha", "psi", "limites" [3], "regles" [3], "classe"}]}."""
    eps = NO.epsilon(fy)
    A = s["A"]
    fy_, fz = My != 0.0, Mz != 0.0
    out = []
    for p in parois(s):
        p = dict(p)
        role = p["role"]
        alpha = psi = None
        if role in ("semelle", "tube"):
            etat = COMPRESSION if (N > 0 or fy_ or fz) else TENDUE
        else:
            # la paroi sert de SEMELLE au moment qui la comprime uniformément, d'ÂME à l'autre
            semelle_de, ame_de = {"ame": (False, fy_), "b": (fy_, fz), "h": (fz, fy_)}[role]
            if semelle_de:
                etat = COMPRESSION
            elif ame_de:
                etat = FLEXION if N == 0 else FLEXION_COMPOSEE
            else:
                etat = COMPRESSION if N > 0 else TENDUE
        if etat == TENDUE:
            limites, regles = [None, None, None], [("", None)] * 3
        elif etat == COMPRESSION:
            k = eps ** 2 if p["type"] == "tube" else eps
            limites = [lim * k for lim in LIMITES_COMPRESSION[p["type"]]]
            regles = [("e_lim_tube" if p["type"] == "tube" else "e_lim", lim) for lim in LIMITES_COMPRESSION[p["type"]]]
        elif etat == FLEXION:
            limites = [lim * eps for lim in LIMITES_FLEXION]
            regles = [("e_lim", lim) for lim in LIMITES_FLEXION]
        else:
            alpha, psi = alpha_psi(N, p["c"], p["n"] * p["t"], A, fy)
            limites, regles = limites_flexion_composee(alpha, psi, eps)
        p.update({"etat": etat, "alpha": alpha, "psi": psi, "limites": limites, "regles": [list(r) for r in regles],
                  "classe": _classe(p["c_t"], limites)})
        out.append(p)
    return {"classe": max(p["classe"] for p in out), "epsilon": eps, "parois": out}


# ------------------------------------------------------------------ EN 1993-1-5 § 4.4
def k_sigma_interne(psi: float) -> float:
    """k_σ d'une paroi interne (tableau 4.1), ψ = σ₂/σ₁ ≥ −3."""
    if psi >= 1.0:
        return 4.0
    if psi > 0.0:
        return 8.2 / (1.05 + psi)
    if psi == 0.0:
        return 7.81
    if psi > -1.0:
        return 7.81 - 6.29 * psi + 9.78 * psi * psi
    if psi == -1.0:
        return 23.9
    return 5.98 * (1.0 - psi) ** 2


def k_sigma_console(psi: float) -> float:
    """k_σ d'une paroi en console dont le BORD LIBRE est le plus comprimé (tableau 4.2), 1 ≥ ψ ≥ −3 : 0,43 pour ψ = 1."""
    return 0.57 - 0.21 * psi + 0.07 * psi * psi


def lambda_p(c_t: float, eps: float, k_sigma: float) -> float:
    """λ̄_p = (c/t) / (28,4·ε·√k_σ)."""
    return c_t / (28.4 * eps * math.sqrt(k_sigma))


def rho_interne(lp: float, psi: float) -> float:
    if lp <= 0.5 + math.sqrt(0.085 - 0.055 * psi):
        return 1.0
    return min((lp - 0.055 * (3.0 + psi)) / lp ** 2, 1.0)


def rho_console(lp: float) -> float:
    if lp <= LAMBDA_P_CONSOLE:
        return 1.0
    return min((lp - 0.188) / lp ** 2, 1.0)


def aire_efficace(s: Dict, eps: float) -> Dict:
    """A_eff sous compression uniforme (ψ = 1) : {"A_eff", "reduite", "parois": [{…, "k_sigma", "lambda_p", "rho",
    "retrait" (mm²)}]}."""
    out = []
    for p in parois(s):
        if p["type"] == "tube":
            raise NonCouvert("Tube rond de classe 4 : EN 1993-1-6, hors du module.")
        p = dict(p)
        interne = p["type"] == "interne"
        ks = 4.0 if interne else K_SIGMA_CONSOLE
        lp = lambda_p(p["c_t"], eps, ks)
        rho = rho_interne(lp, 1.0) if interne else rho_console(lp)
        p.update({"psi": 1.0, "k_sigma": ks, "lambda_p": lp, "rho": rho, "retrait": p["n"] * (1.0 - rho) * p["c"] * p["t"]})
        out.append(p)
    A_eff = s["A"] - sum(p["retrait"] for p in out)
    return {"A_eff": A_eff, "reduite": A_eff < s["A"], "parois": out}


def module_efficace(s: Dict, eps: float, axe: str) -> Dict:
    """W_eff autour de ``axe`` (« y » fort, « z » faible) sous flexion seule : {"W_eff" (mm³), "I_eff" (mm⁴), "e"
    (déplacement de l'axe neutre vers la fibre tendue, mm), "v" (fibre comprimée extrême de la section brute, mm),
    "reduit", "membrure": {…, "psi", "k_sigma", "lambda_p", "rho"}, "ame": {…} | None}."""
    forme, A = s["forme"], s["A"]
    w = {p["role"]: p for p in parois(s)}
    if forme == "I":
        cf, tf = w["semelle"]["c"], s["tf"]
        if axe == "y":
            inertie, v, Wel = s["Iy"], s["h"] / 2.0, s["Wel_y"]
            psi_m, ks = 1.0, K_SIGMA_CONSOLE
            lp = lambda_p(cf / tf, eps, ks)
            rho = rho_console(lp)
            long = (1.0 - rho) * cf                      # le bout des deux consoles de la semelle comprimée
            trou = (2.0 * long * tf, v - tf / 2.0, 2.0 * long * tf ** 3 / 12.0)
            ame = (w["ame"]["c"], s["tw"], 1)
        else:
            inertie, v, Wel = s["Iz"], s["b"] / 2.0, s["Wel_z"]
            psi_m = (v - cf) / v                         # contrainte à la naissance de la console / au bord libre (section brute)
            ks = k_sigma_console(psi_m)
            lp = lambda_p(cf / tf, eps, ks)
            rho = rho_console(lp)
            long = (1.0 - rho) * cf                      # le bout des deux consoles comprimées (une par semelle)
            trou = (2.0 * long * tf, v - long / 2.0, 2.0 * tf * long ** 3 / 12.0)
            ame = None
        membrure = {"nom": "semelle", "type": "console", "c": cf, "t": tf, "c_t": cf / tf}
    elif forme in ("tube_carre", "tube_rect"):
        t = s["t"]
        if axe == "y":
            inertie, v, Wel, m, a = s["Iy"], s["h"] / 2.0, s["Wel_y"], w["b"], w["h"]
        else:
            inertie, v, Wel, m, a = s["Iz"], s["b"] / 2.0, s["Wel_z"], w["h"], w["b"]
        psi_m, ks = 1.0, 4.0
        lp = lambda_p(m["c_t"], eps, ks)
        rho = rho_interne(lp, 1.0)
        long = (1.0 - rho) * m["c"]                      # le milieu de la paroi comprimée
        trou = (long * t, v - t / 2.0, long * t ** 3 / 12.0)
        ame = (a["c"], t, 2)
        membrure = {"nom": m["nom"], "type": "interne", "c": m["c"], "t": t, "c_t": m["c_t"]}
    else:
        raise NonCouvert("Section efficace : I, H et tubes carrés ou rectangulaires seulement.")
    if not Wel or not inertie:
        raise Refus("I et W_el manquent : le module efficace en dépend")
    membrure.update({"psi": psi_m, "k_sigma": ks, "lambda_p": lp, "rho": rho})
    trous = [trou] if trou[0] > 0 else []
    ame_out = None
    if ame is not None:
        c, t, n = ame
        zG1 = -sum(a_ * z for a_, z, _ in trous) / (A - sum(a_ for a_, _, _ in trous))
        haut, bas = c / 2.0 - zG1, c / 2.0 + zG1         # de l'axe neutre aux deux bords de la partie droite de l'âme
        psi_w = -bas / haut
        ks_w = k_sigma_interne(psi_w)
        lp_w = lambda_p(c / t, eps, ks_w)
        rho_w = rho_interne(lp_w, psi_w)
        if psi_w < 0:
            bc = haut
            beff = rho_w * bc
            be1 = BE1_PSI_NEGATIF * beff
        else:
            bc = c
            beff = rho_w * bc
            be1 = 2.0 / (5.0 - psi_w) * beff
        long_w = bc - beff
        if long_w > 0:
            z_b = c / 2.0 - be1
            trous.append((n * long_w * t, z_b - long_w / 2.0, n * t * long_w ** 3 / 12.0))
        ame_out = {"nom": "âme" if forme == "I" else ("paroi h" if axe == "y" else "paroi b"), "type": "interne", "c": c, "t": t,
                   "c_t": c / t, "psi": psi_w, "k_sigma": ks_w, "lambda_p": lp_w, "rho": rho_w, "b_c": bc, "b_eff": beff}
    if not trous:
        return {"W_eff": Wel, "I_eff": inertie, "e": 0.0, "v": v, "reduit": False, "membrure": membrure, "ame": ame_out}
    A_net = A - sum(a_ for a_, _, _ in trous)
    zG = -sum(a_ * z for a_, z, _ in trous) / A_net
    I_eff = inertie - sum(a_ * z * z + i0 for a_, z, i0 in trous) - A_net * zG * zG
    W_eff = min(I_eff / (v - zG), Wel)
    return {"W_eff": W_eff, "I_eff": I_eff, "e": -zG, "v": v, "reduit": True, "membrure": membrure, "ame": ame_out}


def hauteur_cisaillee(s: Dict, direction: str) -> Optional[Dict]:
    """L'âme qui porte l'effort tranchant de ``direction`` (« z » : parallèle à l'âme ; « y » : parallèle aux semelles)
    pour le critère de voilement h_w/t_w ≤ 72·ε/η (EN 1993-1-5 § 5.1(2)) : {"hw", "tw"} — None si la question ne se
    pose pas (tube rond ; semelles en console d'un I sous V_y)."""
    if s["forme"] == "I":
        return {"hw": SE.hauteur_ame(s), "tw": s["tw"]} if direction == "z" else None
    if s["forme"] in ("tube_carre", "tube_rect"):
        cote = s["h"] if direction == "z" else s["b"]
        return {"hw": cote - B_MOINS_3T * s["t"], "tw": s["t"]}
    return None

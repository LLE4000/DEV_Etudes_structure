# -*- coding: utf-8 -*-
"""EUROCODE 3, le FLAMBEMENT d'une barre comprimée (NBN EN 1993-1-1 § 6.3.1) : par flexion autour de chaque axe, et
par torsion pour un I ou un H. docs/acier/BARRES_ACIER.md § 5.5.

    N_cr = π²·E·I / L_cr²                 λ̄ = √(A·f_y / N_cr)                     (A_eff en classe 4)
    Φ = 0,5·[1 + α·(λ̄ − 0,2) + λ̄²]        χ = 1 / (Φ + √(Φ² − λ̄²)) ≤ 1            N_b,Rd = χ·A·f_y / γ_M1
    torsion (§ 6.3.1.4, deux axes de symétrie) : N_cr,T = (G·I_t + π²·E·I_w / L_T²) / i₀²,  i₀² = i_y² + i_z² ; courbe de
    l'axe z. Elle n'est rendue que si N_cr,T < N_cr,z (sinon elle ne peut pas gouverner).

Les courbes (tableau 6.2) et χ sont ceux du module Flambement en service (acier/flambement/normes.py, une seule
source) ; s'y ajoutent ici le I soudé (b et c jusqu'à 40 mm de semelle, c et d au-delà) et le caisson soudé (b).
Le flambement est dit négligeable quand λ̄ ≤ 0,2 ou N_Ed/N_cr ≤ 0,04 (§ 6.3.1.2(4)) — la formule donne alors χ = 1 ou
presque, elle n'est pas court-circuitée.

Les grandeurs servent aussi à la flexion composée (annexe A) : elles sont toujours calculées ; les vérifications ne
sont rendues que sous un effort de compression.
"""
import math
from typing import Dict, Optional

from acier.flambement import normes as NO

from . import section as SE
from .resultat import Journal, Refus, fr as _fr, terme as _t

REF = "EN 1993-1-1 § 6.3.1"
LAMBDA_NEGLIGE, RAPPORT_NEGLIGE = 0.2, 0.04      # § 6.3.1.2(4)
SEMELLE_EPAISSE = 40.0                           # mm : tableau 6.2, sections en I soudées
NOMS_AXE = {"y": "fort", "z": "faible"}


def courbes(s: Dict, nuance: Optional[str], fy: float) -> Dict:
    """Les courbes de flambement du tableau 6.2 : {"y", "z", "regle"}. Refus si le tableau ne dit rien de la section
    (la barre peut alors donner ses deux courbes)."""
    forme, fab = s["forme"], s["fabrication"]
    nu = nuance or ("S460" if fy >= 460.0 else "S235")           # seule la ligne « S460 » du tableau dépend de la nuance
    try:
        if forme == "I" and fab == "lamine":
            if not s.get("h") or not s.get("b") or not s.get("tf"):
                raise Refus("h, b, t_f manquent : la courbe d'un I laminé en dépend")
            cy, cz, regle = NO.courbes("lamine_I", nu, s["h"], s["b"], s["tf"])
        elif forme == "I" and fab in ("soude_oxycoupe", "soude_larges_plats"):
            if not s.get("tf"):
                raise Refus("t_f manque : la courbe d'un I soudé en dépend")
            mince = s["tf"] <= SEMELLE_EPAISSE
            cy, cz = ("b", "c") if mince else ("c", "d")
            regle = "I soudé, t_f ≤ 40 mm" if mince else "I soudé, t_f > 40 mm"
        elif forme in SE.TUBES and fab in ("creux_chaud", "creux_froid"):
            cy, cz, regle = NO.courbes(fab, nu)
        elif forme in ("tube_carre", "tube_rect") and fab in ("caisson", "caisson_recuit"):
            cy, cz, regle = "b", "b", "caisson soudé"
        else:
            raise Refus(f"le tableau 6.2 ne donne pas de courbe pour : {forme}, {fab}")
    except NO.HorsDomaine as e:
        raise Refus(str(e)) from None
    return {"y": cy, "z": cz, "regle": regle}


def ajouter(J: Journal, d: Dict) -> None:
    """Les grandeurs du flambement (``G["flambement"]``) et, sous compression, les vérifications E-Nb-y, E-Nb-z, E-Nb-T."""
    G, s = J.G, d["section"]
    N = d["NEd"]
    if not G.get("couvert"):
        return
    P = d["parametres"]
    E, Gc, gM1 = P["E"], P["G"], P["gamma_M1"]
    A_N, fy = G["A_N"], G["fy"]
    comprime = N > 0
    forces = d["axes"]["y"]["courbe"] is not None and d["axes"]["z"]["courbe"] is not None
    try:
        cb = courbes(s, d["nuance"], fy)
    except Refus:
        if not forces:
            raise
        cb = {"y": None, "z": None, "regle": "courbes données"}
    F: Dict = {"regle": cb["regle"], "gamma_M1": gM1}
    G["flambement"] = F
    nomA = "A_eff" if A_N != s["A"] else "A"
    for a in ("y", "z"):
        ax = d["axes"][a]
        nom = s["axes"][0 if a == "y" else 1]
        inertie, Lcr = s[f"I{a}"], ax["Lcr"]
        courbe = ax["courbe"] or cb[a]
        alpha = NO.ALPHA[courbe]
        g = {"nom": nom, "L": ax["L"], "u": ax["u"], "Lcr": Lcr, "courbe": courbe, "courbe_norme": cb[a], "alpha": alpha,
             "courbe_forcee": ax["courbe"] is not None and cb[a] is not None and ax["courbe"] != cb[a]}
        F[a] = g
        if not Lcr > 0:                                          # axe tenu sur toute la longueur : pas de flambement
            g.update({"N_cr": None, "lambda_bar": 0.0, "Phi": None, "chi": 1.0, "N_b_Rd": A_N * fy / gM1, "neglige": True})
            continue
        Ncr = math.pi ** 2 * E * inertie / Lcr ** 2
        lb = math.sqrt(A_N * fy / Ncr)
        Phi, chi = NO.phi_chi(lb, alpha)
        NbRd = chi * A_N * fy / gM1
        g.update({"N_cr": Ncr, "lambda_bar": lb, "Phi": Phi, "chi": chi, "N_b_Rd": NbRd,
                  "neglige": lb <= LAMBDA_NEGLIGE or (comprime and N / Ncr <= RAPPORT_NEGLIGE)})
        cles = [J.f(f"N_cr_{a}", f"N_cr,{a}", [_t("E", E, "N/mm²"), _t(f"I_{a}", inertie / 1e4, "cm⁴"), _t(f"L_cr,{a}", Lcr / 1e3, "m")],
                    Ncr / 1e3, "kN", "e_N_cr"),
                J.f(f"lambda_bar_{a}", f"λ̄_{a}", [_t(nomA, A_N, "mm²"), _t("f_y", fy, "N/mm²"), _t(f"N_cr,{a}", Ncr / 1e3, "kN")], lb, "",
                    "e_lambda_bar"),
                J.f(f"Phi_{a}", f"Φ_{a}", [_t("α", alpha), _t(f"λ̄_{a}", lb)], Phi, "", "e_Phi")]
        if chi < 1.0:
            cles.append(J.f(f"chi_{a}", f"χ_{a}", [_t(f"Φ_{a}", Phi), _t(f"λ̄_{a}", lb)], chi, "", "e_chi"))
        cles.append(J.f(f"N_b_Rd_{a}", f"N_b,{a},Rd", [_t(f"χ_{a}", chi), _t(nomA, A_N, "mm²"), _t("f_y", fy, "N/mm²"), _t("γ_M1", gM1)],
                        NbRd / 1e3, "kN", "e_N_b_Rd"))
        g["formules"] = cles
        if g["courbe_forcee"]:
            J.alerte(f"courbe_{a}", "attention", f"Axe {nom} : courbe {courbe} imposée (le tableau 6.2 donne {cb[a]} : {cb['regle']}).")
        if comprime:
            J.verif(f"E-Nb-{a}", f"Flambement, axe {NOMS_AXE[a]} {nom}", REF, f"N_Ed ≤ N_b,{a},Rd", N / 1e3, NbRd / 1e3, "kN", cles,
                    messages=["Flambement négligeable (λ̄ ≤ 0,2 ou N_Ed/N_cr ≤ 0,04)."] if g["neglige"] else ())
    # ---- flambement par torsion d'un I ou d'un H (centre de cisaillement confondu avec le centre de gravité)
    LT = d["torsion"]["Lcr"]
    if s["forme"] == "I" and LT > 0 and s.get("It") and s.get("Iw") is not None:
        i0 = math.sqrt(s["iy"] ** 2 + s["iz"] ** 2)
        NcrT = (Gc * s["It"] + math.pi ** 2 * E * s["Iw"] / LT ** 2) / i0 ** 2
        courbe = F["z"]["courbe"]
        lb = math.sqrt(A_N * fy / NcrT)
        Phi, chi = NO.phi_chi(lb, NO.ALPHA[courbe])
        NbRd = chi * A_N * fy / gM1
        gouverne = F["z"]["N_cr"] is None or NcrT < F["z"]["N_cr"]
        F["T"] = {"Lcr": LT, "i0": i0, "N_cr": NcrT, "lambda_bar": lb, "Phi": Phi, "chi": chi, "N_b_Rd": NbRd, "courbe": courbe,
                  "gouverne": gouverne}
        if gouverne:
            cles = [J.f("i0", "i₀", [_t("i_y", s["iy"], "mm"), _t("i_z", s["iz"], "mm")], i0, "mm", "e_i0"),
                    J.f("N_cr_T", "N_cr,T", [_t("G", Gc, "N/mm²"), _t("I_t", s["It"] / 1e4, "cm⁴"), _t("E", E, "N/mm²"),
                                             _t("I_w", s["Iw"] / 1e6, "cm⁶"), _t("L_T", LT / 1e3, "m"), _t("i₀", i0, "mm")], NcrT / 1e3, "kN",
                        "e_N_cr_T"),
                    J.f("lambda_bar_T", "λ̄_T", [_t(nomA, A_N, "mm²"), _t("f_y", fy, "N/mm²"), _t("N_cr,T", NcrT / 1e3, "kN")], lb, "",
                        "e_lambda_bar"),
                    J.f("Phi_T", "Φ_T", [_t("α", NO.ALPHA[courbe]), _t("λ̄_T", lb)], Phi, "", "e_Phi")]
            if chi < 1.0:
                cles.append(J.f("chi_T", "χ_T", [_t("Φ_T", Phi), _t("λ̄_T", lb)], chi, "", "e_chi"))
            cles.append(J.f("N_b_Rd_T", "N_b,T,Rd", [_t("χ_T", chi), _t(nomA, A_N, "mm²"), _t("f_y", fy, "N/mm²"), _t("γ_M1", gM1)],
                            NbRd / 1e3, "kN", "e_N_b_Rd"))
            F["T"]["formules"] = cles
            if comprime:
                J.verif("E-Nb-T", "Flambement par torsion", f"{REF}.4", "N_Ed ≤ N_b,T,Rd", N / 1e3, NbRd / 1e3, "kN", cles)
    if comprime:
        J.hyp.append(f"Courbes de flambement : {F['y']['courbe']} (axe fort), {F['z']['courbe']} (axe faible) — {cb['regle']} ; "
                     f"γ_M1 = {_fr(gM1, 2)}.")

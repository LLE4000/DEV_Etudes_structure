# -*- coding: utf-8 -*-
"""EUROCODE 3, la FLEXION COMPOSÉE AVEC INSTABILITÉ (NBN EN 1993-1-1 § 6.3.3) : les expressions (6.61) et (6.62), avec
les facteurs d'interaction de la MÉTHODE 1 (annexe A). docs/acier/BARRES_ACIER.md § 5.7.

    (6.61)  N_Ed/(χ_y·N_Rk/γ_M1) + k_yy·M_y,Ed/(χ_LT·M_y,Rk/γ_M1) + k_yz·M_z,Ed/(M_z,Rk/γ_M1) ≤ 1
    (6.62)  N_Ed/(χ_z·N_Rk/γ_M1) + k_zy·M_y,Ed/(χ_LT·M_y,Rk/γ_M1) + k_zz·M_z,Ed/(M_z,Rk/γ_M1) ≤ 1

N_Rk = A·f_y, M_Rk = W·f_y selon la classe (tableau 6.7 : A_eff et W_eff en classe 4 ; e_N = 0, deux axes de symétrie).

Annexe A, tableau A.1 —
    μ_i = (1 − N_Ed/N_cr,i)/(1 − χ_i·N_Ed/N_cr,i)      w_i = W_pl,i/W_el,i ≤ 1,5      n_pl = N_Ed/(N_Rk/γ_M1)
    a_LT = 1 − I_t/I_y ≥ 0      λ̄_max = max(λ̄_y ; λ̄_z)      λ̄_0 : l'élancement de déversement sous moment CONSTANT
    si λ̄_0 ≤ 0,2·√C₁·[(1 − N_Ed/N_cr,z)·(1 − N_Ed/N_cr,T)]^¼ : C_my = C_my,0 ; C_mz = C_mz,0 ; C_mLT = 1
    sinon : C_my = C_my,0 + (1 − C_my,0)·√ε_y·a_LT/(1 + √ε_y·a_LT) ; C_mz = C_mz,0 ;
            C_mLT = C_my²·a_LT/√[(1 − N_Ed/N_cr,z)·(1 − N_Ed/N_cr,T)] ≥ 1 ; ε_y = (M_y,Ed/N_Ed)·(A/W_el,y)
    classes 1 et 2 :  k_yy = C_my·C_mLT·μ_y/(1 − N_Ed/N_cr,y)/C_yy          k_yz = C_mz·μ_y/(1 − N_Ed/N_cr,z)/C_yz·0,6·√(w_z/w_y)
                      k_zy = C_my·C_mLT·μ_z/(1 − N_Ed/N_cr,y)/C_zy·0,6·√(w_y/w_z)     k_zz = C_mz·μ_z/(1 − N_Ed/N_cr,z)/C_zz
        C_yy = 1 + (w_y − 1)·[(2 − 1,6/w_y·C_my²·λ̄_max − 1,6/w_y·C_my²·λ̄_max²)·n_pl − b_LT] ≥ W_el,y/W_pl,y
        C_yz = 1 + (w_z − 1)·[(2 − 14·C_mz²·λ̄_max²/w_z⁵)·n_pl − c_LT] ≥ 0,6·√(w_z/w_y)·W_el,z/W_pl,z
        C_zy = 1 + (w_y − 1)·[(2 − 14·C_my²·λ̄_max²/w_y⁵)·n_pl − d_LT] ≥ 0,6·√(w_y/w_z)·W_el,y/W_pl,y
        C_zz = 1 + (w_z − 1)·[(2 − 1,6/w_z·C_mz²·λ̄_max − 1,6/w_z·C_mz²·λ̄_max²)·n_pl − e_LT] ≥ W_el,z/W_pl,z
        b_LT = 0,5·a_LT·λ̄_0²·M_y,Ed/(χ_LT·M_pl,y,Rd)·M_z,Ed/M_pl,z,Rd        c_LT = 10·a_LT·λ̄_0²/(5 + λ̄_z⁴)·M_y,Ed/(C_my·χ_LT·M_pl,y,Rd)
        d_LT = 2·a_LT·λ̄_0/(0,1 + λ̄_z⁴)·M_y,Ed/(C_my·χ_LT·M_pl,y,Rd)·M_z,Ed/(C_mz·M_pl,z,Rd)
        e_LT = 1,7·a_LT·λ̄_0/(0,1 + λ̄_z⁴)·M_y,Ed/(C_my·χ_LT·M_pl,y,Rd)
    classes 3 et 4 :  les mêmes k SANS les C_ij ni le facteur 0,6·√(w/w).

Tableau A.2 — C_mi,0 selon le diagramme du moment : linéaire 0,79 + 0,21·ψ + 0,36·(ψ − 0,33)·N_Ed/N_cr,i ; charge
répartie 1 + 0,03·N_Ed/N_cr,i ; charge concentrée à mi-portée 1 − 0,18·N_Ed/N_cr,i ; cas général
1 + (π²·E·I_i·|δ|/(L²·|M_i,Ed|) − 1)·N_Ed/N_cr,i (δ : la flèche maximale ; à défaut, le moment est pris constant).
(Le brief avait échangé les deux cas chargés : écart E2 du deuxième avis.)

PARAMÈTRES À CONFIRMER sur l'annexe belge (parametres.py) : la méthode 1 ; la borne inférieure de C_mi,0, prise à
1 − N_Ed/N_cr,i.

Une barre sans déversement possible (tube, semelle comprimée tenue) : χ_LT = 1, λ̄_0 = 0 — donc C_my = C_my,0,
C_mLT = 1 et b_LT = c_LT = d_LT = e_LT = 0. Sans effort normal (flexion déviée seule) : les mêmes expressions, N_Ed = 0.
Une barre TENDUE et fléchie autour de ses deux axes est vérifiée comme si elle n'était pas tendue (du côté sûr).
"""
import math
from typing import Dict, List, Optional

from . import ec3_deversement as DV
from .resultat import NOK, Journal, Refus, terme as _t

REF = "EN 1993-1-1 § 6.3.3, annexe A"
W_MAX = 1.5                      # w_y, w_z ≤ 1,5


def cm0(J: Journal, axe: str, fl: Dict, N: float, Ncr: Optional[float], E: float, inertie: float, M: float, borne: bool) -> float:
    """C_mi,0 (tableau A.2) de l'axe ``axe`` pour le diagramme ``fl`` ; nommé dans le journal."""
    r = N / Ncr if Ncr else 0.0
    diag = fl["diagramme"]
    termes_N = [_t("N_Ed", N / 1e3, "kN"), _t(f"N_cr,{axe}", (Ncr or 0.0) / 1e3, "kN")]
    if not Ncr or N == 0:
        c, gabarit, termes = 1.0, None, []
        if diag == DV.LINEAIRE:
            c = 0.79 + 0.21 * fl["psi"]
            gabarit, termes = "e_Cm0_lineaire_sans_N", [_t(f"ψ_{axe}", fl["psi"])]
    elif diag == DV.LINEAIRE:
        c = 0.79 + 0.21 * fl["psi"] + 0.36 * (fl["psi"] - 0.33) * r
        gabarit, termes = "e_Cm0_lineaire", [_t(f"ψ_{axe}", fl["psi"])] + termes_N
    elif diag == DV.REPARTI:
        c = 1.0 + 0.03 * r
        gabarit, termes = "e_Cm0_reparti", termes_N
    elif diag == DV.PONCTUEL:
        c = 1.0 - 0.18 * r
        gabarit, termes = "e_Cm0_ponctuel", termes_N
    elif diag == DV.GENERAL and fl.get("delta") and fl.get("L") and M > 0:
        c = 1.0 + (math.pi ** 2 * E * inertie * abs(fl["delta"]) / (fl["L"] ** 2 * M) - 1.0) * r
        gabarit = "e_Cm0_general"
        termes = [_t("E", E, "N/mm²"), _t(f"I_{axe}", inertie / 1e4, "cm⁴"), _t("δ", abs(fl["delta"]), "mm"), _t("L", fl["L"] / 1e3, "m"),
                  _t(f"M_{axe},Ed", M / 1e6, "kN·m")] + termes_N
    else:                                         # moment constant (ψ = 1) — et tout diagramme non décrit, du côté sûr
        c = 0.79 + 0.21 + 0.36 * (1.0 - 0.33) * r
        gabarit, termes = "e_Cm0_lineaire", [_t(f"ψ_{axe}", 1.0)] + termes_N
    if gabarit:
        J.f(f"Cm0_{axe}", f"C_m{axe},0", termes, c, "", gabarit)
    if borne and c < 1.0 - r:                     # borne inférieure de l'annexe belge (supposée : 1 − N_Ed/N_cr,i)
        c = 1.0 - r
        J.f(f"Cm0_{axe}_borne", f"C_m{axe},0", termes_N, c, "", "e_Cm0_borne")
    return c


def ajouter(J: Journal, d: Dict) -> None:
    """Les vérifications E-661 et E-662, quand la barre est comprimée ET fléchie, ou fléchie autour de ses deux axes."""
    G, s = J.G, d["section"]
    if not G.get("couvert"):
        return
    N = max(d["NEd"], 0.0)                       # une traction n'est pas comptée : la barre est prise sans effort normal (du côté sûr)
    My, Mz = G["My_calcul"], G["Mz_calcul"]
    if not ((N > 0 and (My > 0 or Mz > 0)) or (My > 0 and Mz > 0)):
        return
    P = d["parametres"]
    E, gM0, gM1 = P["E"], P["gamma_M0"], P["gamma_M1"]
    fy, classe, A_N = G["fy"], G["classe"], G["A_N"]
    F, D = G["flambement"], G["deversement"]
    plastique = classe <= 2
    Wy, Wz = G["W_y"], G["W_z"]
    manque = [n_ for n_, v, m in (("W_y", Wy, My), ("W_z", Wz, Mz)) if m > 0 and not v]
    if plastique:
        manque += [n_ for n_ in ("Wel_y", "Wel_z", "Wpl_y", "Wpl_z") if not s.get(n_)]
    if manque:
        raise Refus(f"propriété(s) manquante(s) pour la flexion composée : {', '.join(manque)}")
    Ncry, Ncrz = F["y"]["N_cr"], F["z"]["N_cr"]
    NcrT = F["T"]["N_cr"] if "T" in F else None
    ry, rz, rT = (N / x if x else 0.0 for x in (Ncry, Ncrz, NcrT))
    chiy, chiz, chiLT = F["y"]["chi"], F["z"]["chi"], D["chi_LT"]
    lby, lbz, lb0 = F["y"]["lambda_bar"], F["z"]["lambda_bar"], D["lambda_0"]
    lbmax = max(lby, lbz)
    titre = "Flexion composée et flambement" if N > 0 else "Flexion déviée et déversement"
    if ry >= 1.0 or rz >= 1.0 or rT >= 1.0:
        J.verif("E-661", titre + " (6.61)", REF, "N_Ed < N_cr", None, None, "", [], statut=NOK,
                messages=["N_Ed atteint une charge critique : la barre flambe."])
        return
    A = G.setdefault("annexe_A", {})
    cles: List[str] = []
    NRk = A_N * fy
    MyRk = Wy * fy if My > 0 else None
    MzRk = Wz * fy if Mz > 0 else None
    # ---- les facteurs communs
    muy = (1.0 - ry) / (1.0 - chiy * ry)
    muz = (1.0 - rz) / (1.0 - chiz * rz)
    npl = N / (NRk / gM1)
    aLT = max(1.0 - s["It"] / s["Iy"], 0.0) if s.get("It") else 0.0
    if N > 0:
        for a_, Ncr_, r__, chi_, mu_ in (("y", Ncry, ry, chiy, muy), ("z", Ncrz, rz, chiz, muz), ("T", NcrT, rT, None, None)):
            if not Ncr_:
                continue
            cles.append(J.f(f"rapport_{a_}", f"N_Ed/N_cr,{a_}", [_t("N_Ed", N / 1e3, "kN"), _t(f"N_cr,{a_}", Ncr_ / 1e3, "kN")], r__, "",
                            "e_rapport"))
            if mu_ is not None:
                cles.append(J.f(f"mu_{a_}", f"μ_{a_}", [_t(f"N_Ed/N_cr,{a_}", r__), _t(f"χ_{a_}", chi_)], mu_, "", "e_mu"))
    borne = isinstance(P["borne_Cmi0"], str) and P["borne_Cmi0"].startswith("1 -")
    Cmy0 = cm0(J, "y", d["flexion"]["y"], N, Ncry, E, s["Iy"], My, borne) if My > 0 else 1.0
    Cmz0 = cm0(J, "z", d["flexion"]["z"], N, Ncrz, E, s["Iz"], Mz, borne) if Mz > 0 else 1.0
    cles += [k for k in ("Cm0_y", "Cm0_y_borne", "Cm0_z", "Cm0_z_borne") if any(f["cle"] == k for f in J.calcul)]
    # ---- C_my, C_mz, C_mLT
    C1 = D["C1"]
    lim = 0.2 * math.sqrt(C1) * ((1.0 - rz) * (1.0 - rT)) ** 0.25
    Cmy, Cmz, CmLT = Cmy0, Cmz0, 1.0
    susceptible = lb0 > lim and My > 0
    if lb0 > 0:
        cles.append(J.f("lambda_0", "λ̄_0", [_t("W_y", Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("M_cr,0", D["M_cr0"] / 1e6, "kN·m")], lb0, "",
                        "e_lambda_LT"))
        cles.append(J.f("lambda_0_lim", "λ̄_0,lim", [_t("C₁", C1), _t("N_Ed/N_cr,z", rz), _t("N_Ed/N_cr,T", rT)], lim, "", "e_lambda_0_lim"))
    if susceptible:
        if N > 0:
            Wel = (G["efficace"]["Weff_y"] if classe == 4 else s["Wel_y"])
            epsy = My / N * A_N / Wel
            racine = math.sqrt(epsy) * aLT
            Cmy = Cmy0 + (1.0 - Cmy0) * racine / (1.0 + racine)
            cles.append(J.f("eps_y", "ε_y", [_t("M_y,Ed", My / 1e6, "kN·m"), _t("N_Ed", N / 1e3, "kN"), _t("A", A_N, "mm²"),
                                             _t("W_y", Wel / 1e3, "cm³")], epsy, "", "e_eps_y"))
            cles.append(J.f("Cmy", "C_my", [_t("C_my,0", Cmy0), _t("ε_y", epsy), _t("a_LT", aLT)], Cmy, "", "e_Cmy"))
            A["eps_y"] = epsy
        else:
            Cmy = 1.0 if aLT > 0 else Cmy0        # N_Ed = 0 : ε_y infini
        brut = Cmy ** 2 * aLT / math.sqrt((1.0 - rz) * (1.0 - rT))
        CmLT = max(brut, 1.0)
        if brut >= 1.0:
            cles.append(J.f("CmLT", "C_mLT", [_t("C_my", Cmy), _t("a_LT", aLT), _t("N_Ed/N_cr,z", rz), _t("N_Ed/N_cr,T", rT)], CmLT, "", "e_CmLT"))
    A.update({"mu_y": muy, "mu_z": muz, "n_pl": npl, "a_LT": aLT, "C_my0": Cmy0, "C_mz0": Cmz0, "C_my": Cmy, "C_mz": Cmz, "C_mLT": CmLT,
              "lambda_0": lb0, "lambda_0_lim": lim, "C1": C1, "lambda_max": lbmax, "plastique": plastique})

    def r_(x: float) -> float:
        return 1.0 - x

    # ---- les facteurs k
    if plastique:
        wy_brut, wz_brut = s["Wpl_y"] / s["Wel_y"], s["Wpl_z"] / s["Wel_z"]
        wy, wz = min(wy_brut, W_MAX), min(wz_brut, W_MAX)
        Mply, Mplz = s["Wpl_y"] * fy / gM0, s["Wpl_z"] * fy / gM0
        ty = My / (chiLT * Mply)                               # M_y,Ed/(χ_LT·M_pl,y,Rd)
        tz = Mz / Mplz
        bLT = 0.5 * aLT * lb0 ** 2 * ty * tz
        cLT = 10.0 * aLT * lb0 ** 2 / (5.0 + lbz ** 4) * ty / Cmy
        dLT = 2.0 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy * tz / Cmz
        eLT = 1.7 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy
        Cyy = 1.0 + (wy - 1.0) * ((2.0 - 1.6 / wy * Cmy ** 2 * lbmax - 1.6 / wy * Cmy ** 2 * lbmax ** 2) * npl - bLT)
        Cyz = 1.0 + (wz - 1.0) * ((2.0 - 14.0 * Cmz ** 2 * lbmax ** 2 / wz ** 5) * npl - cLT)
        Czy = 1.0 + (wy - 1.0) * ((2.0 - 14.0 * Cmy ** 2 * lbmax ** 2 / wy ** 5) * npl - dLT)
        Czz = 1.0 + (wz - 1.0) * ((2.0 - 1.6 / wz * Cmz ** 2 * lbmax - 1.6 / wz * Cmz ** 2 * lbmax ** 2) * npl - eLT)
        planchers = {"C_yy": s["Wel_y"] / s["Wpl_y"], "C_yz": 0.6 * math.sqrt(wz / wy) * s["Wel_z"] / s["Wpl_z"],
                     "C_zy": 0.6 * math.sqrt(wy / wz) * s["Wel_y"] / s["Wpl_y"], "C_zz": s["Wel_z"] / s["Wpl_z"]}
        bruts = {"C_yy": Cyy, "C_yz": Cyz, "C_zy": Czy, "C_zz": Czz}
        Cyy, Cyz, Czy, Czz = (max(bruts[k], planchers[k]) for k in ("C_yy", "C_yz", "C_zy", "C_zz"))
        A.update({"w_y": wy, "w_z": wz, "b_LT": bLT, "c_LT": cLT, "d_LT": dLT, "e_LT": eLT, "C_yy": Cyy, "C_yz": Cyz, "C_zy": Czy,
                  "C_zz": Czz, "planchers": [k for k in bruts if bruts[k] < planchers[k]]})
        t_wy, t_wz, t_npl, t_lm = _t("w_y", wy), _t("w_z", wz), _t("n_pl", npl), _t("λ̄_max", lbmax)
        if wy_brut <= W_MAX:
            cles.append(J.f("w_y", "w_y", [_t("W_pl,y", s["Wpl_y"] / 1e3, "cm³"), _t("W_el,y", s["Wel_y"] / 1e3, "cm³")], wy, "", "e_w"))
        if wz_brut <= W_MAX:
            cles.append(J.f("w_z", "w_z", [_t("W_pl,z", s["Wpl_z"] / 1e3, "cm³"), _t("W_el,z", s["Wel_z"] / 1e3, "cm³")], wz, "", "e_w"))
        if N > 0:
            cles.append(J.f("n_pl", "n_pl", [_t("N_Ed", N / 1e3, "kN"), _t("N_Rk", NRk / 1e3, "kN"), _t("γ_M1", gM1)], npl, "", "e_npl"))
        t_My = [_t("M_y,Ed", My / 1e6, "kN·m"), _t("χ_LT", chiLT), _t("M_pl,y,Rd", Mply / 1e6, "kN·m")]
        t_Mz = [_t("M_z,Ed", Mz / 1e6, "kN·m"), _t("M_pl,z,Rd", Mplz / 1e6, "kN·m")]
        avec_LT = aLT > 0 and lb0 > 0 and My > 0
        if avec_LT:
            cles.append(J.f("a_LT", "a_LT", [_t("I_t", s["It"] / 1e4, "cm⁴"), _t("I_y", s["Iy"] / 1e4, "cm⁴")], aLT, "", "e_aLT"))
        if My > 0:
            if avec_LT and Mz > 0:
                cles.append(J.f("b_LT", "b_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0)] + t_My + t_Mz, bLT, "", "e_bLT"))
            if bruts["C_yy"] >= planchers["C_yy"]:
                cles.append(J.f("C_yy", "C_yy", [t_wy, _t("C_my", Cmy), t_lm, t_npl, _t("b_LT", bLT)], Cyy, "", "e_Cyy"))
            if avec_LT and Mz > 0:
                cles.append(J.f("d_LT", "d_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0), _t("λ̄_z", lbz), t_My[0], _t("C_my", Cmy), t_My[1], t_My[2],
                                                 t_Mz[0], _t("C_mz", Cmz), t_Mz[1]], dLT, "", "e_dLT"))
            if bruts["C_zy"] >= planchers["C_zy"]:
                cles.append(J.f("C_zy", "C_zy", [t_wy, _t("C_my", Cmy), t_lm, t_npl, _t("d_LT", dLT)], Czy, "", "e_Czy"))
        if Mz > 0:
            if avec_LT:
                cles.append(J.f("c_LT", "c_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0), _t("λ̄_z", lbz), t_My[0], _t("C_my", Cmy), t_My[1], t_My[2]],
                                cLT, "", "e_cLT"))
                cles.append(J.f("e_LT", "e_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0), _t("λ̄_z", lbz), t_My[0], _t("C_my", Cmy), t_My[1], t_My[2]],
                                eLT, "", "e_eLT"))
            if bruts["C_yz"] >= planchers["C_yz"]:
                cles.append(J.f("C_yz", "C_yz", [t_wz, _t("C_mz", Cmz), t_lm, t_npl, _t("c_LT", cLT)], Cyz, "", "e_Cyz"))
            if bruts["C_zz"] >= planchers["C_zz"]:
                cles.append(J.f("C_zz", "C_zz", [t_wz, _t("C_mz", Cmz), t_lm, t_npl, _t("e_LT", eLT)], Czz, "", "e_Czz"))
        kyy = Cmy * CmLT * muy / r_(ry) / Cyy
        kyz = Cmz * muy / r_(rz) / Cyz * 0.6 * math.sqrt(wz / wy)
        kzy = Cmy * CmLT * muz / r_(ry) / Czy * 0.6 * math.sqrt(wy / wz)
        kzz = Cmz * muz / r_(rz) / Czz
        if My > 0:
            cles.append(J.f("k_yy", "k_yy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_y", muy), _t("N_Ed/N_cr,y", ry), _t("C_yy", Cyy)], kyy, "",
                            "e_kyy"))
            cles.append(J.f("k_zy", "k_zy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_z", muz), _t("N_Ed/N_cr,y", ry), _t("C_zy", Czy), t_wy, t_wz],
                            kzy, "", "e_kzy"))
        if Mz > 0:
            cles.append(J.f("k_yz", "k_yz", [_t("C_mz", Cmz), _t("μ_y", muy), _t("N_Ed/N_cr,z", rz), _t("C_yz", Cyz), t_wz, t_wy], kyz, "", "e_kyz"))
            cles.append(J.f("k_zz", "k_zz", [_t("C_mz", Cmz), _t("μ_z", muz), _t("N_Ed/N_cr,z", rz), _t("C_zz", Czz)], kzz, "", "e_kzz"))
    else:
        kyy = Cmy * CmLT * muy / r_(ry)
        kyz = Cmz * muy / r_(rz)
        kzy = Cmy * CmLT * muz / r_(ry)
        kzz = Cmz * muz / r_(rz)
        if My > 0:
            cles.append(J.f("k_yy", "k_yy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_y", muy), _t("N_Ed/N_cr,y", ry)], kyy, "", "e_k_el_y"))
            cles.append(J.f("k_zy", "k_zy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_z", muz), _t("N_Ed/N_cr,y", ry)], kzy, "", "e_k_el_y"))
        if Mz > 0:
            cles.append(J.f("k_yz", "k_yz", [_t("C_mz", Cmz), _t("μ_y", muy), _t("N_Ed/N_cr,z", rz)], kyz, "", "e_k_el_z"))
            cles.append(J.f("k_zz", "k_zz", [_t("C_mz", Cmz), _t("μ_z", muz), _t("N_Ed/N_cr,z", rz)], kzz, "", "e_k_el_z"))
    A.update({"k_yy": kyy, "k_yz": kyz, "k_zy": kzy, "k_zz": kzz, "N_Rk": NRk, "M_y_Rk": MyRk, "M_z_Rk": MzRk, "chi_LT": chiLT})
    G["a_confirmer_interaction"] = ["methode_interaction", "borne_Cmi0"]

    # ---- (6.61) et (6.62)
    for id_, eq, chi, a, ky, kz in (("E-661", "6.61", chiy, "y", kyy, kyz), ("E-662", "6.62", chiz, "z", kzy, kzz)):
        termes, taux, parts = [], 0.0, []
        if N > 0:
            taux += N / (chi * NRk / gM1)
            termes += [_t("N_Ed", N / 1e3, "kN"), _t(f"χ_{a}", chi), _t("N_Rk", NRk / 1e3, "kN")]
            parts.append("N")
        if My > 0:
            taux += ky * My / (chiLT * MyRk / gM1)
            termes += [_t(f"k_{a}y", ky), _t("M_y,Ed", My / 1e6, "kN·m"), _t("χ_LT", chiLT), _t("M_y,Rk", MyRk / 1e6, "kN·m")]
            parts.append("My")
        if Mz > 0:
            taux += kz * Mz / (MzRk / gM1)
            termes += [_t(f"k_{a}z", kz), _t("M_z,Ed", Mz / 1e6, "kN·m"), _t("M_z,Rk", MzRk / 1e6, "kN·m")]
            parts.append("Mz")
        termes.append(_t("γ_M1", gM1))
        cle = J.f(f"eq_{eq.replace('.', '')}", "η", termes, taux, "", "e_66_" + "_".join(parts))
        critere = " + ".join(x for x, p in ((f"N_Ed/(χ_{a}·N_Rk/γ_M1)", "N"), (f"k_{a}y·M_y,Ed/(χ_LT·M_y,Rk/γ_M1)", "My"),
                                            (f"k_{a}z·M_z,Ed/(M_z,Rk/γ_M1)", "Mz")) if p in parts) + " ≤ 1"
        J.verif(id_, f"{titre} ({eq})", REF, critere, taux, 1.0, "", cles + [cle], taux=taux)
    J.hyp.append("Flexion composée avec instabilité : méthode 1 (annexe A) — annexe belge, à confirmer.")

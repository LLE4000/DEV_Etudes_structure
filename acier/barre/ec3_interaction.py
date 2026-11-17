# -*- coding: utf-8 -*-
"""EUROCODE 3, la FLEXION COMPOSÉE AVEC INSTABILITÉ (NBN EN 1993-1-1 § 6.3.3) : les expressions (6.61) et (6.62) de la
MÉTHODE 1, normative en Belgique (ANB:2018, 6.3.3(5)), sous l'écriture de l'ANNEXE D ANB, prise comme un tout (sans
morceau de l'annexe A ni de l'annexe B). docs/acier/BARRES_ACIER.md § 5.7.

    (6.61)  N_Ed/(χ_y·N_Rk/γ_M1) + k_yy·M_y,Ed/(χ_LT·M_y,Rk/γ_M1) + k_yz·M_z,Ed/(M_z,Rk/γ_M1) ≤ 1
    (6.62)  N_Ed/(χ_z·N_Rk/γ_M1) + k_zy·M_y,Ed/(χ_LT·M_y,Rk/γ_M1) + k_zz·M_z,Ed/(M_z,Rk/γ_M1) ≤ 1

N_Rk = A·f_y, M_Rk = W·f_y selon la classe (tableau 6.7 : A_eff et W_eff en classe 4 ; e_N = 0, deux axes de symétrie,
donc ΔM = 0). La résistance des sections d'extrémité est vérifiée en plus (ec3_section.py ; note 1 de l'annexe D ANB).

Annexe D ANB —
  ÉTAPE 1 : la barre est-elle « susceptible de se déformer par torsion » ?
    I_t ≥ I_y : non ;  sinon non si λ̄_0 ≤ 0,2·√C₁·[(1 − N_Ed/N_cr,z)·(1 − N_Ed/N_cr,TF)]^¼, oui au-delà
    λ̄_0 : l'élancement de déversement sous moment UNIFORME ; C₁ selon l'annexe E ANB (ec3_deversement : le rapport
    M_cr/M_cr,0 du solveur, charges au centre de cisaillement) ; N_cr,TF : (F.4) de l'annexe F ANB, qui redonne (F.3),
    N_cr,T, pour une section à deux axes de symétrie (y₀ = z₀ = 0) — le module ne traite que celles-là ici
  ÉTAPE 2.a (non susceptible) : pas de χ_LT ni de C_mLT ; C_my = C_my,0 ; C_ij sans b_LT, c_LT, d_LT, e_LT
  ÉTAPE 2.b (susceptible) : C_my = C_my,0 + (1 − C_my,0)·√ε_y·a_LT/(1 + √ε_y·a_LT), ε_y = (M_y,Ed/N_Ed)·(A/W_el,y)
    (A_eff/W_eff,y en classe 4) ; C_mz = C_mz,0 ; C_mLT = C_my²·a_LT/√[(1 − N_Ed/N_cr,z)·(1 − N_Ed/N_cr,T)] ≥ 1
  commun : μ_i = (1 − N_Ed/N_cr,i)/(1 − χ_i·N_Ed/N_cr,i) ; a_LT = 1 − I_t/I_y ≥ 0 ; λ̄_max = max(λ̄_y ; λ̄_z) ;
    n_pl = N_Ed/(N_Rk/γ_M1) (l'annexe A corrigée par l'AC:2009 écrit γ_M0 : même résultat tant que γ_M0 = γ_M1 ; sinon
    la variante γ_M0 est calculée et affichée, l'équivalence n'étant pas établie — À CONFIRMER, brief § 7.8)
  classes 1 et 2 : w_y = W_pl,y/W_el,y ≤ 1,5 ; w_z = W_pl,z/W_el,z ≤ 1,5
        k_yy = C_my·C_mLT·μ_y/(1 − N_Ed/N_cr,y)/C_yy          k_yz = C_mz·μ_y/(1 − N_Ed/N_cr,z)/C_yz·0,6·√(w_z/w_y)
        k_zy = C_my·C_mLT·μ_z/(1 − N_Ed/N_cr,y)/C_zy·0,6·√(w_y/w_z)     k_zz = C_mz·μ_z/(1 − N_Ed/N_cr,z)/C_zz
        C_yy = 1 + (w_y − 1)·[(2 − 1,6/w_y·C_my²·λ̄_max − 1,6/w_y·C_my²·λ̄_max²)·n_pl − b_LT] ≥ W_el,y/W_pl,y
        C_yz = 1 + (w_z − 1)·[(2 − 14·C_mz²·λ̄_max²/w_z⁵)·n_pl − c_LT] ≥ 0,6·√(w_z/w_y)·W_el,z/W_pl,z
        C_zy = 1 + (w_y − 1)·[(2 − 14·C_my²·λ̄_max²/w_y⁵)·n_pl − d_LT] ≥ 0,6·√(w_y/w_z)·W_el,y/W_pl,y
        C_zz = 1 + (w_z − 1)·(2 − 1,6/w_z·C_mz²·λ̄_max − 1,6/w_z·C_mz²·λ̄_max² − e_LT)·n_pl ≥ W_el,z/W_pl,z
            (e_LT DANS le facteur n_pl : annexe D ANB, p. 28, et AC:2009, modification 35 b ; le texte de 2005 le mettait dehors)
        b_LT = 0,5·a_LT·λ̄_0²·M_y,Ed/(χ_LT·M_pl,y,Rd)·M_z,Ed/M_pl,z,Rd        c_LT = 10·a_LT·λ̄_0²/(5 + λ̄_z⁴)·M_y,Ed/(C_my·χ_LT·M_pl,y,Rd)
        d_LT = 2·a_LT·λ̄_0/(0,1 + λ̄_z⁴)·M_y,Ed/(C_my·χ_LT·M_pl,y,Rd)·M_z,Ed/(C_mz·M_pl,z,Rd)
        e_LT = 1,7·a_LT·λ̄_0/(0,1 + λ̄_z⁴)·M_y,Ed/(C_my·χ_LT·M_pl,y,Rd)
  classes 3 et 4 : w_y = w_z = 1, C_yy = C_zz = 1, C_yz = C_zy = 0,6 — les k s'écrivent alors sans C_ij ni 0,6·√(w/w).

Tableau D.1 ANB — C_mi,0 selon le diagramme du moment, SANS borne inférieure (ψ = −1 et N_Ed/N_cr = 0,1 donnent 0,532) :
moments d'extrémité 0,79 + 0,21·ψ + 0,36·(ψ − 0,33)·N_Ed/N_cr,i ; deux appuis simples : charge répartie
1 + 0,03·N_Ed/N_cr,i, charge concentrée à mi-portée 1 − 0,18·N_Ed/N_cr,i ; cas général
1 + (π²·E·I_i·|δ|/(L²·|M_i,Ed|) − 1)·N_Ed/N_cr,i (δ : la flèche maximale ; à défaut, le moment est pris constant).
(Le brief avait échangé les deux cas chargés : écart E2 du deuxième avis.)

Une barre sans déversement possible (tube, semelle comprimée tenue) : λ̄_0 = 0, donc étape 2.a. Sans effort normal
(flexion déviée seule) : les mêmes expressions, N_Ed = 0. Une barre TENDUE et fléchie autour de ses deux axes est
vérifiée comme si elle n'était pas tendue (du côté sûr).
"""
import math
from typing import Dict, List, Optional

from . import ec3_deversement as DV
from . import prs as PR
from .resultat import NOK, NON_COUVERT, Journal, Refus, fr as _fr, terme as _t

REF = "EN 1993-1-1 § 6.3.3, annexe D ANB"
W_MAX = 1.5                      # w_y, w_z ≤ 1,5


def cm0(J: Journal, axe: str, fl: Dict, N: float, Ncr: Optional[float], E: float, inertie: float, M: float) -> float:
    """C_mi,0 (tableau D.1 ANB) de l'axe ``axe`` pour le diagramme ``fl`` ; nommé dans le journal. Pas de borne inférieure."""
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
    if PR.monosymetrique(s):
        # PRS monosymétrique (brief § 7.8 et 18.8) : 6.3.3(1) vise les barres à section BISYMÉTRIQUE — non couvert ; la
        # résistance de la section reste vérifiée (§ 6.2)
        J.verif("E-661", "Flexion composée avec instabilité (6.61), (6.62)", REF, "", None, None, "", [], statut=NON_COUVERT,
                messages=[J.alerte("prs_NM", "attention", "PRS monosymétrique : flexion composée avec instabilité (§ 6.3.3) non couverte — "
                                                          "elle vise les sections bisymétriques ; la section reste vérifiée.")])
        return
    if d.get("console") and N > 0 and d["parametres"]["console_NM"] == "non_couvert":
        # les diagrammes du tableau A.2 (et D.1 ANB) sont ceux d'une barre entre deux appuis : aucun ne donne C_m pour une
        # console — « non couvert » (brief § 7.8, DÉCISION BUREAU n° 9, défaut v1 ; test 30 du § 11.5), LU DANS LA TABLE
        # (parametres « console_NM ») : la table n'admet encore que lui — aucune méthode fixée par le bureau (À VÉRIFIER)
        J.verif("E-661", "Flexion composée et flambement (6.61), (6.62)", REF, "", None, None, "", [], statut=NON_COUVERT,
                messages=[J.alerte("console_NM", "attention", "Console sous effort normal et flexion : la flexion composée avec instabilité "
                                                              "(§ 6.3.3, annexe D ANB) n'est pas couverte — aucun C_m ne vaut pour une console ; "
                                                              "N seul et M seul sont vérifiés.")])
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
    NcrT = F["T"]["N_cr"] if "T" in F else None                      # = N_cr,TF : (F.4) redonne (F.3), deux axes de symétrie
    ry, rz, rT = (N / x if x else 0.0 for x in (Ncry, Ncrz, NcrT))
    chiy, chiz = F["y"]["chi"], F["z"]["chi"]
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
    g_npl, g_autre = (gM1, gM0) if P["npl_gamma"] != "gamma_M0" else (gM0, gM1)      # γ_M1 : l'annexe D ANB (défaut de la table)
    npl = N / (NRk / g_npl)
    aLT = max(1.0 - s["It"] / s["Iy"], 0.0) if s.get("It") else 0.0
    if N > 0:
        for a_, Ncr_, r__, chi_, mu_ in (("y", Ncry, ry, chiy, muy), ("z", Ncrz, rz, chiz, muz), ("T", NcrT, rT, None, None)):
            if not Ncr_:
                continue
            cles.append(J.f(f"rapport_{a_}", f"N_Ed/N_cr,{a_}", [_t("N_Ed", N / 1e3, "kN"), _t(f"N_cr,{a_}", Ncr_ / 1e3, "kN")], r__, "",
                            "e_rapport"))
            if mu_ is not None:
                cles.append(J.f(f"mu_{a_}", f"μ_{a_}", [_t(f"N_Ed/N_cr,{a_}", r__), _t(f"χ_{a_}", chi_)], mu_, "", "e_mu"))
    Cmy0 = cm0(J, "y", d["flexion"]["y"], N, Ncry, E, s["Iy"], My) if My > 0 else 1.0
    Cmz0 = cm0(J, "z", d["flexion"]["z"], N, Ncrz, E, s["Iz"], Mz) if Mz > 0 else 1.0
    cles += [k for k in ("Cm0_y", "Cm0_z") if any(f["cle"] == k for f in J.calcul)]
    # ---- ÉTAPE 1 de l'annexe D ANB : la barre est-elle susceptible de se déformer par torsion ?
    C1 = D["C1"]
    lim = 0.2 * math.sqrt(C1) * ((1.0 - rz) * (1.0 - rT)) ** 0.25
    rigide = bool(s.get("It")) and s["It"] >= s["Iy"]                 # I_t ≥ I_y : non susceptible, sans autre critère
    Cmy, Cmz, CmLT = Cmy0, Cmz0, 1.0
    susceptible = lb0 > lim and My > 0 and not rigide
    etape = "2.b" if susceptible else "2.a"
    if lb0 > 0 and not rigide:
        if D.get("M_cr_centre"):
            cles.append(J.f("C1", "C₁", [_t("M_cr,c", D["M_cr_centre"] / 1e6, "kN·m"), _t("M_cr,0", D["M_cr0"] / 1e6, "kN·m")], C1, "", "e_C1"))
        cles.append(J.f("lambda_0", "λ̄_0", [_t("W_y", Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("M_cr,0", D["M_cr0"] / 1e6, "kN·m")], lb0, "",
                        "e_lambda_LT"))
        cles.append(J.f("lambda_0_lim", "λ̄_0,lim", [_t("C₁", C1), _t("N_Ed/N_cr,z", rz), _t("N_Ed/N_cr,TF", rT)], lim, "", "e_lambda_0_lim"))
    # étape 2.a : ni χ_LT ni C_mLT dans (6.61), (6.62) — le déversement seul reste vérifié par E-Mb
    chiLT = D["chi_LT"] if susceptible else 1.0
    nom_chi = D.get("symbole_chi", "χ_LT")               # « χ_LT,mod » quand le déversement l'a modifié : le nom de SA ligne
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
              "lambda_0": lb0, "lambda_0_lim": lim, "C1": C1, "lambda_max": lbmax, "plastique": plastique, "etape": etape,
              "susceptible": susceptible, "It_sup_Iy": rigide})

    def r_(x: float) -> float:
        return 1.0 - x

    # ---- les termes b_LT à e_LT (étape 2.b seulement) et les planchers des C_ij (classes 1 et 2)
    if plastique:
        wy_brut, wz_brut = s["Wpl_y"] / s["Wel_y"], s["Wpl_z"] / s["Wel_z"]
        wy, wz = min(wy_brut, W_MAX), min(wz_brut, W_MAX)
        Mply, Mplz = s["Wpl_y"] * fy / gM0, s["Wpl_z"] * fy / gM0
        ty = My / (chiLT * Mply)                               # M_y,Ed/(χ_LT·M_pl,y,Rd)
        tz = Mz / Mplz
        bLT = cLT = dLT = eLT = 0.0
        if susceptible:
            bLT = 0.5 * aLT * lb0 ** 2 * ty * tz
            cLT = 10.0 * aLT * lb0 ** 2 / (5.0 + lbz ** 4) * ty / Cmy
            dLT = 2.0 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy * tz / Cmz
            eLT = 1.7 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy
        planchers = {"C_yy": s["Wel_y"] / s["Wpl_y"], "C_yz": 0.6 * math.sqrt(wz / wy) * s["Wel_z"] / s["Wpl_z"],
                     "C_zy": 0.6 * math.sqrt(wy / wz) * s["Wel_y"] / s["Wpl_y"], "C_zz": s["Wel_z"] / s["Wpl_z"]}

    def facteurs(npl: float) -> Dict:
        """Les C_ij (classes 1 et 2 ; bruts et retenus) et les k_ij pour ce n_pl."""
        if plastique:
            Cyy = 1.0 + (wy - 1.0) * ((2.0 - 1.6 / wy * Cmy ** 2 * lbmax - 1.6 / wy * Cmy ** 2 * lbmax ** 2) * npl - bLT)
            Cyz = 1.0 + (wz - 1.0) * ((2.0 - 14.0 * Cmz ** 2 * lbmax ** 2 / wz ** 5) * npl - cLT)
            Czy = 1.0 + (wy - 1.0) * ((2.0 - 14.0 * Cmy ** 2 * lbmax ** 2 / wy ** 5) * npl - dLT)
            Czz = 1.0 + (wz - 1.0) * (2.0 - 1.6 / wz * Cmz ** 2 * lbmax - 1.6 / wz * Cmz ** 2 * lbmax ** 2 - eLT) * npl
            bruts = {"C_yy": Cyy, "C_yz": Cyz, "C_zy": Czy, "C_zz": Czz}
            Cyy, Cyz, Czy, Czz = (max(bruts[k], planchers[k]) for k in ("C_yy", "C_yz", "C_zy", "C_zz"))
            kyy = Cmy * CmLT * muy / r_(ry) / Cyy
            kyz = Cmz * muy / r_(rz) / Cyz * 0.6 * math.sqrt(wz / wy)
            kzy = Cmy * CmLT * muz / r_(ry) / Czy * 0.6 * math.sqrt(wy / wz)
            kzz = Cmz * muz / r_(rz) / Czz
            return {"bruts": bruts, "C_yy": Cyy, "C_yz": Cyz, "C_zy": Czy, "C_zz": Czz, "k_yy": kyy, "k_yz": kyz, "k_zy": kzy, "k_zz": kzz}
        # classes 3 et 4 : w = 1, C_yy = C_zz = 1, C_yz = C_zy = 0,6 — les 0,6 se simplifient
        kyy = Cmy * CmLT * muy / r_(ry)
        kyz = Cmz * muy / r_(rz)
        kzy = Cmy * CmLT * muz / r_(ry)
        kzz = Cmz * muz / r_(rz)
        return {"k_yy": kyy, "k_yz": kyz, "k_zy": kzy, "k_zz": kzz}

    K = facteurs(npl)
    kyy, kyz, kzy, kzz = K["k_yy"], K["k_yz"], K["k_zy"], K["k_zz"]
    if plastique:
        bruts = K["bruts"]
        Cyy, Cyz, Czy, Czz = K["C_yy"], K["C_yz"], K["C_zy"], K["C_zz"]
        A.update({"w_y": wy, "w_z": wz, "b_LT": bLT, "c_LT": cLT, "d_LT": dLT, "e_LT": eLT, "C_yy": Cyy, "C_yz": Cyz, "C_zy": Czy,
                  "C_zz": Czz, "planchers": [k for k in bruts if bruts[k] < planchers[k]]})
        t_wy, t_wz, t_npl, t_lm = _t("w_y", wy), _t("w_z", wz), _t("n_pl", npl), _t("λ̄_max", lbmax)
        if wy_brut <= W_MAX:
            cles.append(J.f("w_y", "w_y", [_t("W_pl,y", s["Wpl_y"] / 1e3, "cm³"), _t("W_el,y", s["Wel_y"] / 1e3, "cm³")], wy, "", "e_w"))
        if wz_brut <= W_MAX:
            cles.append(J.f("w_z", "w_z", [_t("W_pl,z", s["Wpl_z"] / 1e3, "cm³"), _t("W_el,z", s["Wel_z"] / 1e3, "cm³")], wz, "", "e_w"))
        if N > 0:
            cles.append(J.f("n_pl", "n_pl", [_t("N_Ed", N / 1e3, "kN"), _t("N_Rk", NRk / 1e3, "kN"), _t("γ_M1" if g_npl == gM1 else "γ_M0", g_npl)],
                            npl, "", "e_npl"))
        t_My = [_t("M_y,Ed", My / 1e6, "kN·m"), _t(nom_chi, chiLT), _t("M_pl,y,Rd", Mply / 1e6, "kN·m")]
        t_Mz = [_t("M_z,Ed", Mz / 1e6, "kN·m"), _t("M_pl,z,Rd", Mplz / 1e6, "kN·m")]
        avec_LT = susceptible and aLT > 0
        if avec_LT:
            cles.append(J.f("a_LT", "a_LT", [_t("I_t", s["It"] / 1e4, "cm⁴"), _t("I_y", s["Iy"] / 1e4, "cm⁴")], aLT, "", "e_aLT"))

        def ligne_C(cle: str, symbole: str, t_w, t_Cm, nom_LT: str, LT: float, gab: str, gab_a: str) -> None:
            """La ligne d'un C_ij : avec son terme b_LT…e_LT à l'étape 2.b, sans lui à l'étape 2.a (annexe D ANB)."""
            if bruts[cle] < planchers[cle]:
                return
            termes = [t_w, t_Cm, t_lm, t_npl] + ([_t(nom_LT, LT)] if avec_LT else [])
            cles.append(J.f(cle, symbole, termes, K[cle], "", gab if avec_LT else gab_a))

        if My > 0:
            if avec_LT and Mz > 0:
                cles.append(J.f("b_LT", "b_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0)] + t_My + t_Mz, bLT, "", "e_bLT"))
            ligne_C("C_yy", "C_yy", t_wy, _t("C_my", Cmy), "b_LT", bLT, "e_Cyy", "e_Cii_a")
            if avec_LT and Mz > 0:
                cles.append(J.f("d_LT", "d_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0), _t("λ̄_z", lbz), t_My[0], _t("C_my", Cmy), t_My[1], t_My[2],
                                                 t_Mz[0], _t("C_mz", Cmz), t_Mz[1]], dLT, "", "e_dLT"))
            ligne_C("C_zy", "C_zy", t_wy, _t("C_my", Cmy), "d_LT", dLT, "e_Czy", "e_Cij_a")
        if Mz > 0:
            if avec_LT:
                cles.append(J.f("c_LT", "c_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0), _t("λ̄_z", lbz), t_My[0], _t("C_my", Cmy), t_My[1], t_My[2]],
                                cLT, "", "e_cLT"))
                cles.append(J.f("e_LT", "e_LT", [_t("a_LT", aLT), _t("λ̄_0", lb0), _t("λ̄_z", lbz), t_My[0], _t("C_my", Cmy), t_My[1], t_My[2]],
                                eLT, "", "e_eLT"))
            ligne_C("C_yz", "C_yz", t_wz, _t("C_mz", Cmz), "c_LT", cLT, "e_Cyz", "e_Cij_a")
            ligne_C("C_zz", "C_zz", t_wz, _t("C_mz", Cmz), "e_LT", eLT, "e_Czz", "e_Cii_a")
        if My > 0:
            cles.append(J.f("k_yy", "k_yy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_y", muy), _t("N_Ed/N_cr,y", ry), _t("C_yy", Cyy)], kyy, "",
                            "e_kyy"))
            cles.append(J.f("k_zy", "k_zy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_z", muz), _t("N_Ed/N_cr,y", ry), _t("C_zy", Czy), t_wy, t_wz],
                            kzy, "", "e_kzy"))
        if Mz > 0:
            cles.append(J.f("k_yz", "k_yz", [_t("C_mz", Cmz), _t("μ_y", muy), _t("N_Ed/N_cr,z", rz), _t("C_yz", Cyz), t_wz, t_wy], kyz, "", "e_kyz"))
            cles.append(J.f("k_zz", "k_zz", [_t("C_mz", Cmz), _t("μ_z", muz), _t("N_Ed/N_cr,z", rz), _t("C_zz", Czz)], kzz, "", "e_kzz"))
    else:
        if My > 0:
            cles.append(J.f("k_yy", "k_yy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_y", muy), _t("N_Ed/N_cr,y", ry)], kyy, "", "e_k_el_y"))
            cles.append(J.f("k_zy", "k_zy", [_t("C_my", Cmy), _t("C_mLT", CmLT), _t("μ_z", muz), _t("N_Ed/N_cr,y", ry)], kzy, "", "e_k_el_y"))
        if Mz > 0:
            cles.append(J.f("k_yz", "k_yz", [_t("C_mz", Cmz), _t("μ_y", muy), _t("N_Ed/N_cr,z", rz)], kyz, "", "e_k_el_z"))
            cles.append(J.f("k_zz", "k_zz", [_t("C_mz", Cmz), _t("μ_z", muz), _t("N_Ed/N_cr,z", rz)], kzz, "", "e_k_el_z"))
    A.update({"k_yy": kyy, "k_yz": kyz, "k_zy": kzy, "k_zz": kzz, "N_Rk": NRk, "M_y_Rk": MyRk, "M_z_Rk": MzRk, "chi_LT": chiLT})
    G["a_confirmer_interaction"] = ["npl_gamma"] if (N > 0 and plastique and gM0 != gM1) else []

    # ---- (6.61) et (6.62)
    def taux_de(chi: float, ky: float, kz: float) -> float:
        return ((N / (chi * NRk / gM1) if N > 0 else 0.0) + (ky * My / (chiLT * MyRk / gM1) if My > 0 else 0.0)
                + (kz * Mz / (MzRk / gM1) if Mz > 0 else 0.0))

    variante = None
    if N > 0 and plastique and gM0 != gM1:
        # γ_M0 ≠ γ_M1 : l'annexe D ANB (n_pl avec γ_M1) décide ; l'annexe A corrigée (γ_M0) est calculée et affichée,
        # l'équivalence des deux écritures n'étant pas établie (brief § 7.8, test 26 du § 11.5 — À CONFIRMER)
        K0 = facteurs(N / (NRk / g_autre))
        variante = {"n_pl": N / (NRk / g_autre), "t61": taux_de(chiy, K0["k_yy"], K0["k_yz"]), "t62": taux_de(chiz, K0["k_zy"], K0["k_zz"])}
        A["variante_gamma_M0"] = variante
        nom_v = "γ_M0" if g_npl == gM1 else "γ_M1"
        J.alerte("npl_gamma_M0", "attention", f"n_pl avec {nom_v} : (6.61) {_fr(variante['t61'], 3)} ; (6.62) {_fr(variante['t62'], 3)} — à confirmer.")
    gab_LT = "e_66_" if susceptible else "e_66a_"
    for id_, eq, chi, a, ky, kz in (("E-661", "6.61", chiy, "y", kyy, kyz), ("E-662", "6.62", chiz, "z", kzy, kzz)):
        termes, taux, parts = [], 0.0, []
        if N > 0:
            taux += N / (chi * NRk / gM1)
            termes += [_t("N_Ed", N / 1e3, "kN"), _t(f"χ_{a}", chi), _t("N_Rk", NRk / 1e3, "kN")]
            parts.append("N")
        if My > 0:
            taux += ky * My / (chiLT * MyRk / gM1)
            termes += [_t(f"k_{a}y", ky), _t("M_y,Ed", My / 1e6, "kN·m")] + ([_t(nom_chi, chiLT)] if susceptible else []) \
                + [_t("M_y,Rk", MyRk / 1e6, "kN·m")]
            parts.append("My")
        if Mz > 0:
            taux += kz * Mz / (MzRk / gM1)
            termes += [_t(f"k_{a}z", kz), _t("M_z,Ed", Mz / 1e6, "kN·m"), _t("M_z,Rk", MzRk / 1e6, "kN·m")]
            parts.append("Mz")
        termes.append(_t("γ_M1", gM1))
        gab = (gab_LT if "My" in parts else "e_66_") + "_".join(parts)
        cle = J.f(f"eq_{eq.replace('.', '')}", "η", termes, taux, "", gab)
        nom_My = f"{nom_chi}·M_y,Rk" if susceptible else "M_y,Rk"
        critere = " + ".join(x for x, p in ((f"N_Ed/(χ_{a}·N_Rk/γ_M1)", "N"), (f"k_{a}y·M_y,Ed/({nom_My}/γ_M1)", "My"),
                                            (f"k_{a}z·M_z,Ed/(M_z,Rk/γ_M1)", "Mz")) if p in parts) + " ≤ 1"
        J.verif(id_, f"{titre} ({eq})", REF, critere, taux, 1.0, "", cles + [cle], taux=taux)
    pourquoi = ("I_t ≥ I_y" if rigide else (f"λ̄_0 = {_fr(lb0, 3)} > {_fr(lim, 3)}" if susceptible else
                (f"λ̄_0 = {_fr(lb0, 3)} ≤ {_fr(lim, 3)}" if My > 0 else "pas de M_y")))
    J.hyp.append(f"Flexion composée avec instabilité : méthode 1, annexe D ANB (ANB:2018, 6.3.3(5)) ; étape {etape} "
                 f"({'susceptible' if susceptible else 'non susceptible'} de se déformer par torsion : {pourquoi}).")

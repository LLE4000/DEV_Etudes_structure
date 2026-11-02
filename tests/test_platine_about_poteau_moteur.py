# -*- coding: utf-8 -*-
"""Moteur de l'assemblage POUTRE–POTEAU par platine d'about de hauteur
partielle.

Chaque garantie rougit si on la retire :
  1. HÉRITAGE : l'exemple publié (MSB Part 5 §2.4, Worked Example: Partial
     depth end plate) rejoué avec appui sur l'AILE d'un poteau — les NEUF
     valeurs publiées sortent inchangées, parce qu'aucune ne dépend du
     support ; le tronçon en T de l'exemple est l'ancre de la formule que
     l'aile du poteau réutilise.
  2. CAS (a) AILE, le défaut du module : les nombres du §11.3 BM2 de
     l'étude, recalculés à la main dans la suite — dont la pression
     diamétrale de l'aile SANS pince d'extrémité verticale.
  3. LE GAUGE : p3 ≥ twP + 2·rP + dw REFUSE le p3 = 90 usuel sur un HEB 300,
     l'alerte le dit et propose le minimum (105 mm).
  4. CAS (b) ÂME : la platine 290 × 220 ne tient pas entre les congés ;
     p3 = 110 / bp = 190 passent ; cisaillement local brut et net ACTIFS en
     cas (b) et INACTIFS en cas (a) ; tronçon en T de l'aile et âme en
     traction transversale, l'inverse.
  5. TYING : les DEUX tronçons en T du §7.3 (platine 261,330 kN, aile du
     poteau 955,485 kN, rapport 3,656), l'âme du poteau en traction
     transversale (1 428,0 kN), le poinçonnement des deux pièces.
  6. ALERTES : les neuves (p3_congeP, p3_libreP, bp_libreP, bp_aile,
     aile_poutre, ame_hp) et celles qui ont DISPARU (pl_web, dnt_min,
     dnb_min, ln_min, gap_neg, d_top).
  7. BRIQUES PARTAGÉES : le moteur ne recopie pas la mécanique de la
     famille — il appelle acier/assemblages/mecanique.py, et les grandeurs
     recalculées par ce fichier lui sont identiques.
  8. PRÉDIM : proposition complète, p3 = 105 en cas (a) (le congé du poteau),
     apply_solution recopie tout.
  9. ROBUSTESSE : 300 tirages aléatoires sans exception, dans les deux cas.
 10. RÉGRESSION FIGÉE : tests/ref_pac.json.

Lancement : python3 tests/test_platine_about_poteau_moteur.py
"""
import json
import math
import os
import random
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.bibliotheques import PERSO  # noqa: E402
from acier.assemblages import mecanique as MEC  # noqa: E402
from acier.assemblages.poutre_poteau.platine_about_poteau import moteur, entrees as E  # noqa: E402
from acier.assemblages.poutre_poteau.platine_about_poteau import benchmark as BM  # noqa: E402

OK, KO = [], []
S3 = math.sqrt(3)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * abs(b)


# ================================================================
print("=== 1. HÉRITAGE : l'exemple publié MSB Part 5 §2.4 sur un POTEAU ===")
R = moteur.compute(BM.WE)
chk("âme de la poutre – cisaillement sur hp = 614 kN (Av = 430 × 9 = 3 870 mm²)",
    pres(R.ck["vwS"].Rd, 614, 0.01), str(R.ck["vwS"].Rd))
chk("un boulon en cisaillement Fv,Rd = 94 kN (arrondi de la source)", pres(R.Fv_Rd, 94, 0.01), str(R.Fv_Rd))
chk("pression diamétrale de la platine : 107 kN aux extrémités (αb arrondi 0,61) et 142 kN à l'intérieur",
    pres(R.Fbb_L, 107, 0.01) and pres(R.Fbi_L, 142, 0.01), f"{R.Fbb_L} {R.Fbi_L}")
chk("groupe de boulons FRd = 0,8·ns·Fv,Rd = 902 kN", pres(R.ck["bg"].Rd, 902, 0.01), str(R.ck["bg"].Rd))
chk("platine : min(brut ; net ; bloc) = 1 182 kN",
    pres(min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd), 1182, 0.01),
    str(min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd)))
chk("tying : platine en flexion (tronçon en T, mode 1) = 493 kN", pres(R.ck["ptT"].Rd, 493, 0.01),
    str(R.ck["ptT"].Rd))
chk("tying : âme de la poutre en traction tw·hp·fu/γMu = 1 513 kN", pres(R.ck["tnS"].Rd, 1513, 0.01),
    str(R.ck["tnS"].Rd))
chk("cordon de pleine résistance S275 : a ≥ 0,48·tw", pres(R.k_plein, 0.48, 0.01), str(R.k_plein))
chk("ANCRE du tronçon en T : m = 59,164 ; n = 30 ; ew = 9,25 ; Σleff = 430 (= hp) ; "
    "modes 491,86 / 783,20 / 1 924,36 kN",
    pres(R.m_T, (140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2), 1e-12) and R.nT == 30 and R.ew_T == 9.25
    and R.leff_T == 430 and pres(R.FT1, 491.8580, 1e-5) and pres(R.FT2, 783.1995, 1e-5)
    and pres(R.FT3, 1924.3636, 1e-5), f"{R.m_T} {R.leff_T} {R.FT1} {R.FT2} {R.FT3}")
chk("l'exemple est VÉRIFIÉ, dimensionnante = l'âme de la poutre sur hp (η = 0,895)",
    R.verified and R.gov.key == "vwS" and pres(R.eta_max, 550 / 614.445, 1e-6), R.gov.key)
chk("benchmark du module : statut VALIDÉ, pire écart 0,95 % (αb arrondi par la source)",
    BM.run_bench()["statut"] == "VALIDÉ" and BM.run_bench()["worst"] <= 0.01,
    str(BM.run_bench()["worst"]))

# ================================================================
print("\n=== 2. CAS (a) AILE : le cas par défaut (§11.3 BM2) ===")
R0 = moteur.compute({})
fy, fu, bw = 355.0, 510.0, 0.9
d, d0, As, dm, dw = 20.0, 22.0, 245.0, 31.8, 37.0
Fv = 0.6 * 800 * As / 1.25 / 1000
hp, tp, bp, n1, p1, e1, p3 = 290.0, 10.0, 220.0, 4, 70.0, 40.0, 140.0
ns = 2 * n1
chk("defaults() : 66 clés, appui sur l'aile, d_top n'existe plus, grugeage à 0",
    len(E.defaults()) == 66 and E.defaults()["appui"] == E.APPUI_AILE and "d_top" not in E.defaults()
    and E.defaults()["d_nt"] == 0 and E.defaults()["d_nb"] == 0 and E.defaults()["l_n"] == 0,
    str(len(E.defaults())))
chk("groupe CENTRÉ : ns = 8 boulons, Fv,Ed = VEd/ns = 25 kN, aucun moment d'excentricité",
    R0.n_b == ns and R0.Fv_Ed == 25.0 and "centré" in R0.mod)
chk("pince latérale e2 = (bp − p3)/2 = 40 mm ; pince au bout d'aile e = (bP − p3)/2 = 80 mm ; e1' = 40",
    R0.e_2 == 40 and R0.e_P == 80 and R0.e1bot == 40, f"{R0.e_2} {R0.e_P} {R0.e1bot}")
_k1 = min(2.8 * 40 / d0 - 1.7, 1.4 * p3 / d0 - 1.7, 2.5)
_abb = min(e1 / (3 * d0), 800 / fu, 1)
_abi = min(p1 / (3 * d0) - 0.25, 800 / fu, 1)
chk("platine : k1 = 2,5 ; αb extrémité 0,606061 et intérieur 0,810606 → Fb 123,636 / 165,364 kN",
    R0.k1_L == 2.5 and pres(R0.abb_L, _abb, 1e-12) and pres(R0.abi_L, _abi, 1e-12)
    and pres(R0.Fbb_L, _k1 * _abb * fu * d * tp / 1.25 / 1000, 1e-12)
    and pres(R0.Fbi_L, _k1 * _abi * fu * d * tp / 1.25 / 1000, 1e-12), f"{R0.Fbb_L} {R0.Fbi_L}")
chk("groupe : Fv,Rd ≤ (Fb,Rd)min → FRd = 0,8·ns·Fv,Rd = 602,112 kN (η = 0,332)",
    pres(R0.ck["bg"].Rd, 0.8 * ns * Fv, 1e-12) and "0,8·ns·Fv,Rd" in R0.branche_g
    and pres(R0.ck["bg"].eta, 200 / (0.8 * ns * Fv), 1e-12), str(R0.ck["bg"].Rd))
chk("platine : VRd,g = 936,035 (2 plans, 1/1,27) ; VRd,n = 951,658 ; VRd,b = 945,799 kN (k = 1)",
    pres(R0.ck["pcg"].Rd, 2 * (hp * tp) * fy / (1.27 * S3) / 1000, 1e-12)
    and pres(R0.ck["pcn"].Rd, 2 * tp * (hp - n1 * d0) * fu / (S3 * 1.25) / 1000, 1e-12)
    and pres(R0.ck["pcb"].Rd, 2 * (510 * 290 / 1.25 + 355 * 1730 / S3) / 1000, 1e-12)
    and R0.k_bloc == 1.0 and R0.hp_136,
    f"{R0.ck['pcg'].Rd} {R0.ck['pcn'].Rd} {R0.ck['pcb'].Rd}")
chk("platine : Ant = 290 et Anv = 1 730 mm²", R0.Ant_p == 290 and R0.Anv_p == 1730)
chk("âme portée : Av = hp·tw = 2 494 mm² ; VRd = 511,169 kN → DIMENSIONNANTE, η = 0,391",
    R0.Av_w == 2494 and pres(R0.ck["vwS"].Rd, 2494 * fy / S3 / 1000, 1e-12)
    and R0.gov.key == "vwS" and pres(R0.eta_max, 0.3913, 0.001), f"{R0.gov.key} {R0.eta_max}")
chk("AILE DU POTEAU : αb = 0,810606 du seul p1 (aucune pince d'extrémité), k1 = 2,5 du gauge et du "
    "bout d'aile, t = tfP = 19 → Fb,Rd = 314,191 kN/boulon et VRd = 2 513,53 kN (η = 0,0796)",
    pres(R0.ab_P, _abi, 1e-12) and R0.k1_P == 2.5 and R0.t_appui == 19
    and pres(R0.Fb_P, 2.5 * _abi * fu * d * 19 / 1.25 / 1000, 1e-12)
    and pres(R0.ck["pdP"].Rd, ns * R0.Fb_P, 1e-12) and pres(R0.ck["pdP"].eta, 0.0796, 0.001),
    f"{R0.Fb_P} {R0.ck['pdP'].Rd}")
chk("cordons : qz = VEd/(2hp) = 344,83 N/mm ; fvw,d = 261,732 MPa ; Fw,Rd = 1 308,66 N/mm",
    pres(R0.qz_w, 200000 / (2 * hp), 1e-12) and pres(R0.fvw, fu / (S3 * bw * 1.25), 1e-12)
    and pres(R0.ck["w"].Rd, fu / (S3 * bw * 1.25) * 5, 1e-12), f"{R0.qz_w} {R0.fvw}")
chk("gorge de pleine résistance recommandée 0,553727·tw = 4,762 mm (a = 5 : pas d'alerte)",
    pres(R0.a_plein, bw * 1.25 * fy / (math.sqrt(2) * fu) * 8.6, 1e-12)
    and not any(a.id == "a_rec" for a in R0.alerts), str(R0.a_plein))
chk("ductilité : tp,max = 0,36·d·√(fub/fy) = 10,81 mm (tp = 10 : pas d'alerte)",
    pres(R0.tp_max, 0.36 * 20 * math.sqrt(800 / 355), 1e-12) and R0.rec_tp
    and not any(a.id in ("tp_duct", "tp_min") for a in R0.alerts), str(R0.tp_max))
chk("moment apporté au POTEAU : e_col = hP/2 + tp = 160 mm ; VEd·e_col = 32,0 kNm (information)",
    R0.e_col == 160 and R0.M_col == 32.0, f"{R0.e_col} {R0.M_col}")
chk("cas par défaut VÉRIFIÉ, aucune alerte", R0.verified and not R0.alerts, str([a.id for a in R0.alerts]))
chk("Tableau 3.3 : 5 lignes actives en cas (a) — e1p, e2p, eP (bout d'aile), p1, p3",
    [x.id for x in R0.dist] == ["e1p", "e2p", "eP", "p1", "p3"] and all(x.ok for x in R0.dist),
    str([x.id for x in R0.dist]))
chk("γM1 n'est pas demandé (aucune instabilité de plat) ; γMu par défaut = 1,10 (guides)",
    "g_M1" not in E.defaults() and R0.u.g_Mu == 1.1)

# ================================================================
print("\n=== 3. LE GAUGE : la règle qui refuse le p3 = 90 usuel ===")
chk("gauge minimal sur l'aile d'un HEB 300 avec M20 : p3 ≥ twP + 2·rP + dw = 102 mm",
    R0.p3_congeP == 11 + 2 * 27 + 37, str(R0.p3_congeP))
R90 = moteur.compute(dict(p3_u=90, bp_u=170))
a90 = [a for a in R90.alerts if a.id == "p3_congeP"]
chk("p3 = 90 mm : alerte BLOQUANTE p3_congeP, l'assemblage n'est pas vérifié",
    bool(a90) and a90[0].block and not R90.geo_ok and not R90.verified,
    str([a.id for a in R90.alerts]))
chk("… et l'alerte DIT la règle, le chiffre et le minimum à prendre (105 mm), avec les gauges usuels",
    bool(a90) and "twP + 2·rP + dw" in a90[0].why and "102,0 mm" in a90[0].why
    and "Prendre p3 = 105 mm" in a90[0].why and "90 / 100 / 120 / 140" in a90[0].why,
    a90[0].why if a90 else "")
chk("p3 = 105 mm (le minimum proposé) passe", moteur.compute(dict(p3_u=105, bp_u=185)).verified)
chk("p3 = 160 mm : le MAXIMUM du Tableau 3.3 (min(14t ; 200) = 140) rougit",
    not moteur.compute(dict(p3_u=160, bp_u=240)).dist_ok
    and [x.ok for x in moteur.compute(dict(p3_u=160, bp_u=240)).dist if x.id == "p3"] == [False])
chk("pince au bout d'aile : p3 = 260 sur un HEB 300 → e = 20 < 1,2·d0 = 26,4 mm, la ligne eP rougit",
    [x.ok for x in moteur.compute(dict(p3_u=260, bp_u=320)).dist if x.id == "eP"] == [False])
chk("bp > bP : alerte d'information bp_aile (la platine dépasse le bout d'aile, ce n'est pas impossible)",
    any(a.id == "bp_aile" and not a.block for a in moteur.compute(dict(bp_u=320)).alerts))

# ================================================================
print("\n=== 4. CAS (b) ÂME : la platine entre les congés (§11.3 BM4) ===")
Rb0 = moteur.compute(dict(appui=E.APPUI_AME))
chk("la platine 290 × 220 du cas (a) NE TIENT PAS en cas (b) : alerte bloquante bp_libreP (≤ 198 mm)",
    any(a.id == "bp_libreP" and a.block for a in Rb0.alerts) and not Rb0.geo_ok
    and pres(Rb0.bp_maxP, 198, 1e-12), str([a.id for a in Rb0.alerts]))
R4 = moteur.compute(BM.BM4)
chk("p3 = 110 et bp = 190 : la platine tient, l'écrou dégage le congé (73,5 ≤ 104, marge 30,5 mm)",
    R4.verified and pres(R4.demi_droit - (R4.p_3 / 2 + R4.d_w / 2), 30.5, 1e-12), str(R4.alerts))
chk("géométrie du poteau : largeur droite 208 mm, demi-largeur 104, passage de la semelle 252 mm, "
    "p3 maximal 171 mm",
    R4.libre_droit == 208 and R4.demi_droit == 104 and R4.libre_ailes == 252 and R4.p3_maxP == 171,
    f"{R4.libre_droit} {R4.demi_droit} {R4.libre_ailes} {R4.p3_maxP}")
chk("ÂME DU POTEAU : k1 = 2,5 du seul gauge (aucun bord libre), t = twP = 11 → Fb,Rd = 181,900 kN "
    "et VRd = 1 455,20 kN",
    R4.t_appui == 11 and R4.k1_P == 2.5 and pres(R4.Fb_P, 181.9, 1e-4)
    and pres(R4.ck["pdP"].Rd, 8 * 181.9, 1e-4), f"{R4.Fb_P} {R4.ck['pdP'].Rd}")
chk("cisaillement local de l'âme ACTIF : brut 653,820 kN (η = 0,153) et net 523,412 kN (η = 0,191), "
    "Ed = VEd/2 = 100 kN",
    R4.ck["vlP"].active and R4.ck["vnP"].active and R4.ck["vlP"].Ed == 100
    and pres(R4.ck["vlP"].Rd, 11 * 290 * 355 / S3 / 1000, 1e-12)
    and pres(R4.ck["vnP"].Rd, 11 * (290 - 4 * 22) * 510 / (S3 * 1.25) / 1000, 1e-12)
    and pres(R4.ck["vlP"].eta, 0.15294722129620533, 1e-9) and pres(R4.ck["vnP"].eta, 0.1911, 0.001),
    f"{R4.ck['vlP'].Rd} {R4.ck['vnP'].Rd}")
chk("cas (a) : le cisaillement local de l'âme est SANS OBJET (l'attache est sur l'aile)",
    not R0.ck["vlP"].active and not R0.ck["vnP"].active)
chk("l'alerte d'information ame_hp (flexion hors plan non vérifiée) n'existe qu'en cas (b)",
    any(a.id == "ame_hp" and not a.block for a in R4.alerts)
    and not any(a.id == "ame_hp" for a in R0.alerts))
chk("BM4 : l'IPE 400 (bS = 180) passe entre les ailes du HEB 300 — aucune recoupe nécessaire",
    R4.passe_S and R4.rec_ok and not any(a.id == "aile_poutre" for a in R4.alerts))
chk("recoupe minimale (bP − twP)/2 + j − tp = 139,5 mm (l'about est à tp de la face de l'âme)",
    pres(R4.rec_min, (300 - 11) / 2 + 5 - 10, 1e-12), str(R4.rec_min))
Ri = moteur.compute(dict(BM.BM4, prof_S="HEA 300", z_C=45, hp_u=190, n1_u=3, p1_u=60, e1_u=35, V_Ed=125))
chk("HEA 300 portée (bS = 300 > 252) sans recoupe : alerte BLOQUANTE aile_poutre",
    any(a.id == "aile_poutre" and a.block for a in Ri.alerts) and not Ri.geo_ok and not Ri.verified)
Rr = moteur.compute(dict(BM.BM4, prof_S="HEA 300", rec_S=145, z_C=45, hp_u=190, n1_u=3, p1_u=60,
                         e1_u=35, V_Ed=125))
chk("… recoupe de 145 mm ≥ 139,5 : l'alerte disparaît et l'assemblage est vérifié",
    not any(a.id == "aile_poutre" for a in Rr.alerts) and Rr.verified)
Rp3 = moteur.compute(dict(appui=E.APPUI_AME, p3_u=180, bp_u=190))
chk("p3 = 180 en cas (b) : l'écrou sort de la partie droite de l'âme → alerte bloquante p3_libreP",
    any(a.id == "p3_libreP" and a.block for a in Rp3.alerts), str([a.id for a in Rp3.alerts]))
chk("cas (b) : e_col = tp = 10 mm (la face d'appui est l'âme) ; VEd·e_col = 2,0 kNm",
    R4.e_col == 10 and R4.M_col == 2.0, f"{R4.e_col} {R4.M_col}")
chk("cas (b) : le Tableau 3.3 n'a PAS de ligne « bout d'aile » (il n'y a pas de bord libre)",
    [x.id for x in R4.dist] == ["e1p", "e2p", "p1", "p3"], str([x.id for x in R4.dist]))

# ================================================================
print("\n=== 5. TYING : les DEUX tronçons en T (§7.3) ===")
Rt = moteur.compute(BM.BM3)
chk("tying : NEd est sorti de la combinaison ELU (N_elu = 0) et vérifié seul (N_ty = 250) avec γMu",
    Rt.tying and Rt.N_elu == 0 and Rt.N_ty == 250 and Rt.g_T == 1.1)
chk("Ft,Rd,u = 0,9·fub·As/γMu = 160,364 kN ; ΣFt = 8 × = 1 282,91 kN ; ew = dw/4 = 9,25 mm",
    pres(Rt.Ft_Rd_u, 0.9 * 800 * 245 / 1.1 / 1000, 1e-12) and pres(Rt.SFt, 1282.909, 1e-5)
    and Rt.ew_T == 9.25, f"{Rt.Ft_Rd_u} {Rt.SFt}")
chk("PLATINE : m = 60,043 ; n = 40 ; Σleff = 290 (= hp) ; Mpl,1 = 3,3614 kNm ; "
    "FT,1 = 261,330 ; FT,2 = 580,141 ; FT,3 = 1 282,91 → FT,Rd = 261,330 kN (η = 0,9566)",
    pres(Rt.m_T, (140 - 8.6) / 2 - 0.8 * 5 * math.sqrt(2), 1e-12) and Rt.nT == 40 and Rt.leff_T == 290
    and pres(Rt.Mpl1_T / 1e6, 3.3614, 1e-4) and pres(Rt.FT1, 261.3299, 1e-5)
    and pres(Rt.FT2, 580.1406, 1e-5) and pres(Rt.FT3, 1282.909, 1e-5)
    and pres(Rt.ck["ptT"].Rd, 261.3299, 1e-5) and pres(Rt.ck["ptT"].eta, 0.9566, 0.001),
    f"{Rt.m_T} {Rt.leff_T} {Rt.FT1} {Rt.FT2}")
chk("AILE DU POTEAU (composant NEUF) : m = (p3 − twP)/2 − 0,8·rP = 42,900 ; n = 53,625 ; "
    "Σleff = 280 (= n1·p1, Tab. 6.4, aucune rangée d'extrémité) ; Mpl,1 = 11,7161 kNm",
    pres(Rt.m_fP, (140 - 11) / 2 - 0.8 * 27, 1e-12) and pres(Rt.n_fP, 53.625, 1e-12)
    and Rt.e_fP == 80 and Rt.leff_fP == 280 and pres(Rt.Mpl1_fP / 1e6, 11.7161, 1e-4),
    f"{Rt.m_fP} {Rt.n_fP} {Rt.leff_fP} {Rt.Mpl1_fP / 1e6}")
chk("aile du poteau : FT,1 = 1 296,99 ; FT,2 = 955,485 (déterminant) ; FT,3 = 1 282,91 → "
    "FT,Rd = 955,485 kN (η = 0,262)",
    pres(Rt.FT1_P, 1296.9893, 1e-5) and pres(Rt.FT2_P, 955.4849, 1e-5)
    and pres(Rt.FT3_P, 1282.909, 1e-5) and pres(Rt.ck["tfP"].Rd, 955.4849, 1e-5)
    and pres(Rt.ck["tfP"].eta, 0.2616, 0.001), f"{Rt.FT1_P} {Rt.FT2_P} {Rt.ck['tfP'].Rd}")
chk("l'aile du poteau est 3,656 fois plus forte que la platine : le composant neuf ne gouverne pas ici, "
    "et la dimensionnante en tying est la PLATINE",
    pres(Rt.FT_Rd_P / Rt.FT_Rd, 3.6562, 1e-4) and Rt.gov.key == "ptT",
    f"{Rt.FT_Rd_P / Rt.FT_Rd} {Rt.gov.key}")
chk("âme du poteau en traction transversale : ω = 1, beff = Σleff de l'aile = 280 → 1 428,0 kN",
    Rt.ck["twtP"].active and Rt.omega_P == 1.0 and Rt.beff_P == 280
    and pres(Rt.ck["twtP"].Rd, 280 * 11 * 510 / 1.1 / 1000, 1e-12)
    and Rt.ck["twtP"].nat == "EC + INT", str(Rt.ck["twtP"].Rd))
chk("poinçonnement : Ft,Ed = NEd/ns = 31,25 kN ; platine 277,911 kN et aile 528,031 kN → le min",
    pres(Rt.ck["pp"].Ed, 31.25, 1e-12) and pres(Rt.Bp_pl, 277.911, 1e-4)
    and pres(Rt.Bp_P, 528.031, 1e-4) and pres(Rt.ck["pp"].Rd, 277.911, 1e-4),
    f"{Rt.Bp_pl} {Rt.Bp_P} {Rt.ck['pp'].Rd}")
chk("tying : âme portée en traction sur hp = 1 156,31 kN ; cordons qx = 431,03 pour 1 487,11 N/mm",
    pres(Rt.ck["tnS"].Rd, 2494 * 510 / 1.1 / 1000, 1e-12)
    and pres(Rt.ck["wT"].Ed, 250000 / 580, 1e-12)
    and pres(Rt.ck["wT"].Rd, 510 / (S3 * 0.9 * 1.1) * 5, 1e-12), str(Rt.ck["wT"].Rd))
Rtb = moteur.compute(BM.BM4T)
chk("cas (b) en tying : le tronçon en T de l'aile et l'âme en traction transversale sont SANS OBJET "
    "(la flexion hors plan de l'âme n'est pas couverte), le poinçonnement prend t = twP = 11",
    not Rtb.ck["tfP"].active and not Rtb.ck["twtP"].active
    and pres(Rtb.Bp_P, 0.6 * math.pi * dm * 11 * 510 / 1.1 / 1000, 1e-12)
    and pres(Rtb.Bp_P, 305.702, 1e-4), str(Rtb.Bp_P))
Re = moteur.compute(dict(N_Ed=80))
chk("ELU fondamental : NEd reste combiné à VEd (γM2n pour les ultimes), bvt actif, wT inactif, "
    "les deux tronçons en T avec fy/γM0",
    not Re.tying and Re.N_elu == 80 and Re.g_T == 1.25 and Re.ck["bvt"].active
    and not Re.ck["wT"].active and Re.ck["ptT"].active and Re.ck["tfP"].active
    and pres(Re.Mpl1_T, 0.25 * 290 * 100 * 355 / 1.0, 1e-9), str(Re.Mpl1_T))
chk("une compression (NEd < 0) n'a pas de nature : nat_N invisible, aucune vérification de traction",
    not E.traction(dict(N_Ed=-60)) and not moteur.compute(dict(N_Ed=-60)).ck["ptT"].active
    and not moteur.compute(dict(N_Ed=-60)).ck["tfP"].active)

# ================================================================
print("\n=== 6. Alertes : les neuves, et celles qui ont disparu ===")
chk("alertes RETIRÉES (fausses pour un poteau) : pl_web, dnt_min, dnb_min, ln_min, gap_neg",
    not ({"pl_web", "dnt_min", "dnb_min", "ln_min", "gap_neg"}
         & {a.id for cas in ({}, dict(appui=E.APPUI_AME, bp_u=190, p3_u=110), dict(hp_u=330, z_C=35),
                             dict(prof_S="HEB 500"), dict(d_nt=50, l_n=10, z_C=60))
            for a in moteur.compute(cas).alerts}))
chk("une platine sur TOUTE la hauteur du poteau est permise (la contrainte « partie droite de l'âme "
    "porteuse » n'existe plus)", moteur.compute(dict(hp_u=330, z_C=35, n1_u=4)).geo_ok)
chk("hp = 140 : pince basse < 1,2·d0 → alerte bloquante e1bot",
    any(a.id == "e1bot" and a.block for a in moteur.compute(dict(hp_u=140)).alerts))
chk("bp = 170 avec p3 = 140 : pince latérale e2 = 15 < 26,4 → alerte bloquante e2p",
    any(a.id == "e2p" and a.block for a in moteur.compute(dict(bp_u=170)).alerts))
chk("tp = 14 > 0,36·d·√(fub/fy) : alerte de ductilité (non bloquante)",
    any(a.id == "tp_duct" and not a.block for a in moteur.compute(dict(tp_u=14)).alerts))
chk("tp = 6 : alerte « moins de 8 mm déconseillé » (MSB P5 §2.1)",
    any(a.id == "tp_min" and not a.block for a in moteur.compute(dict(tp_u=6)).alerts))
chk("hp = 180 < 0,6·h : recommandation signalée",
    any(a.id == "hp_rec" for a in moteur.compute(dict(hp_u=180, n1_u=2)).alerts))
chk("a = 3 mm : gorge sous la pleine résistance recommandée (alerte a_rec)",
    any(a.id == "a_rec" for a in moteur.compute(dict(a_w=3)).alerts))
chk("p3 = 60 avec a = 8 : la rondelle chevauche le cordon → alerte bloquante p3_cordon",
    any(a.id == "p3_cordon" and a.block for a in moteur.compute(dict(p3_u=60, bp_u=140, a_w=8)).alerts))
chk("catégorie B avec des boulons 4.6 : alerte bloquante cat_cl",
    any(a.id == "cat_cl" and a.block for a in moteur.compute(dict(cat="B", classe="4.6")).alerts))
chk("trou surdimensionné sans d0 : alerte bloquante d0_manq",
    any(a.id == "d0_manq" and a.block for a in moteur.compute(dict(trou="Surdimensionné")).alerts))
chk("grugeage à 0 par défaut (un poteau n'oblige pas à gruger) et bloc du grugeage inactif",
    R0.cas_g == 0 and not R0.ck["mN"].active and not R0.ck["stN"].active and not R0.ck["vgS"].active)
Rg = moteur.compute(dict(d_nt=50, d_nb=40, l_n=120, z_C=60, hp_u=240, n1_u=3))
chk("… mais tout le bloc de la poutre grugée s'active à l'identique quand on grugue, bras tp + c",
    Rg.cas_g == 2 and Rg.ck["mN"].active and Rg.ck["stN"].active and Rg.ck["stD"].active
    and Rg.ck["vgS"].active and Rg.e_N == 10 + 120 and pres(Rg.ck["mN"].Ed, 200 * 0.130, 1e-12))

# ================================================================
print("\n=== 7. Les briques de mecanique.py sont APPELÉES, pas recopiées ===")
pl = MEC.portance_bout_interieur(R0.t_p, R0.fu_L, R0.f_ub, R0.d_b, R0.d_0, R0.e1_min, R0.e_2,
                                 R0.p_1, R0.p_3, 1.25, 1.0)
chk("pression diamétrale de la platine : k1, αb et Fb identiques à mecanique.portance_bout_interieur",
    pl["k1"] == R0.k1_L and pl["ab_bout"] == R0.abb_L and pl["ab_int"] == R0.abi_L
    and pl["Fb_bout"] == R0.Fbb_L and pl["Fb_int"] == R0.Fbi_L)
be = MEC.bearing(R0.t_appui, R0.fu_P, R0.f_ub, R0.d_b, R0.d_0, None, R0.p_1, R0.n_1,
                 R0.e_P, R0.p_3, 2, 1.25, 1.0)
chk("pression diamétrale du poteau : αb, k1 et Fb identiques à mecanique.bearing(e1 = None)",
    be["abv"] == R0.ab_P and be["k1v"] == R0.k1_P and be["Fbv"] == R0.Fb_P)
gb = MEC.groupe_boulons(R0.Fv_Rd * R0.bLf, R0.Fbb_L, R0.Fbi_L, 2, R0.n_b - 2)
chk("groupe de fixations : FRd et la branche identiques à mecanique.groupe_boulons",
    gb[0] == R0.FRd_g and gb[1] == R0.branche_g)
sg = MEC.section_grugee(dict(h=Rg.h_S, b=Rg.b_S, tw=Rg.tw_S, tf=Rg.tf_S, r=Rg.r_S),
                        Rg.fy_S, 50, 40, 1.0, 200.0, True)
chk("poutre grugée : cas_g, Av, W_N, Mv,N,Rd, ln,max identiques à mecanique.section_grugee",
    sg["cas_g"] == Rg.cas_g and sg["Av"] == Rg.Av_S and sg["W_N"] == Rg.W_N
    and sg["Mv_N"] == Rg.ck["mN"].Rd and sg["ln_max"] == Rg.ln_max)
l1, l2 = MEC.leff_aile_poteau(Rt.n_1, Rt.p_1, Rt.m_fP, Rt.e_fP)
chk("longueurs efficaces de l'AILE DU POTEAU : identiques à mecanique.leff_aile_poteau "
    "(Σcp isolées 1 078,19 ; Σnc isolées 1 086,40 ; Σcp groupe 560 ; Σnc groupe 280)",
    l1 == Rt.leff_fP and l2 == Rt.leff2_fP and l1 == 280
    and pres(4 * 2 * math.pi * Rt.m_fP, 1078.19, 1e-4)
    and pres(4 * (4 * Rt.m_fP + 1.25 * 80), 1086.40, 1e-4), f"{l1} {l2}")
lp1, lp2 = MEC.leff_platine_bords_libres(Rt.n_1, Rt.p_1, Rt.e_1, Rt.e1bot, Rt.m_T, Rt.e_2)
chk("longueurs efficaces de la PLATINE : identiques à mecanique.leff_platine_bords_libres",
    lp1 == Rt.leff_T and lp2 == Rt.leff2_T)
tt = MEC.troncon_T(Rt.leff_fP, Rt.leff2_fP, Rt.tf_P, Rt.m_fP, Rt.n_fP, Rt.ew_fP,
                   Rt.fu_P, 1.1, Rt.SFt)
chk("tronçon en T : les trois modes identiques à mecanique.troncon_T",
    tt["FT1"] == Rt.FT1_P and tt["FT2"] == Rt.FT2_P and tt["FT3"] == Rt.FT3_P
    and tt["FT_Rd"] == Rt.FT_Rd_P)
chk("ductilité et gorge de pleine résistance : identiques à mecanique.tp_ductile / "
    "coef_pleine_resistance",
    MEC.tp_ductile(R0.d_b, R0.f_ub, R0.fy_L) == R0.tp_max
    and MEC.coef_pleine_resistance(R0.fy_S, R0.fuw, R0.bww, 1.25, 1.0) == R0.k_plein)

# ================================================================
print("\n=== 8. Prédimensionnement ===")
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT"))
chk("proposition complète au cas par défaut", Rp.pd.found and Rp.pd.msg == "Proposition complète", Rp.pd.msg)
chk("cas (a) : p3 proposé = ceil5(max(90 ; 2,4·d0 ; twP + 2·rP + dw)) = 105 mm (le congé du poteau "
    "commande)", Rp.pd.p3 == 105, str(Rp.pd.p3))
Rpa = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT", appui=E.APPUI_AME))
chk("cas (b) : le congé de l'aile ne s'applique plus → p3 proposé = 90 mm (valeur du guide)",
    Rpa.pd.p3 == 90, str(Rpa.pd.p3))
chk("épaisseur proposée entre 8 mm et la limite de ductilité",
    8 <= Rp.pd.tp <= 0.36 * Rp.d_b * math.sqrt(800 / Rp.fy_L) + 1e-9, str(Rp.pd.tp))
chk("bp proposé = p3 + 2·e2 et, en cas (b), la platine tient entre les congés",
    Rp.pd.bp == Rp.pd.p3 + 2 * Rp.pd.e2 and Rpa.pd.bp <= Rpa.libre_droit - 2 * 5,
    f"{Rp.pd.bp} {Rpa.pd.bp}")
u2 = moteur.apply_solution(dict(), Rp.pd.pick)
chk("apply_solution : mode VÉRIFICATION, boulon, rangées, p3, hp, tp, bp, a recopiés",
    u2["mode_calc"] == "VÉRIFICATION" and u2["boulon_u"] == Rp.pd.boulon
    and u2["n1_u"] == Rp.pd.n and u2["p3_u"] == Rp.pd.p3 and u2["hp_u"] == Rp.pd.hp
    and u2["tp_u"] == Rp.pd.tp and u2["bp_u"] == Rp.pd.bp and u2["a_w"] == Rp.pd.a)
Rv = moteur.compute(u2)
chk("la solution recopiée est vérifiée en détail", Rv.verified, Rv.gov.key if Rv.gov else "")
Rv2 = moteur.compute(moteur.apply_solution(dict(appui=E.APPUI_AME), Rpa.pd.pick))
chk("… en cas (b) aussi", Rv2.verified, str([a.id for a in Rv2.alerts]))
chk("un poteau trop étroit pour le gauge : le prédimensionnement le DIT",
    "AUCUN entraxe des files ne tient" in moteur.compute(
        dict(mode_calc="PRÉDIMENSIONNEMENT", prof_P="IPE 200")).pd.msg,
    moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT", prof_P="IPE 200")).pd.msg[:60])

# ================================================================
print("\n=== 9. Robustesse (300 tirages aléatoires, les deux cas d'appui) ===")
rng = random.Random(11)
nerr = 0
for i in range(300):
    try:
        moteur.compute(dict(
            appui=rng.choice([E.APPUI_AILE, E.APPUI_AME]),
            hp_u=rng.choice([0, 60, 140, 190, 290, 400]), tp_u=rng.choice([0, 6, 8, 10, 15, 20]),
            bp_u=rng.choice([0, 100, 150, 190, 220, 320]), z_C=rng.choice([0, 20, 40, 90]),
            n1_u=rng.randint(1, 8), p1_u=rng.choice([0, 40, 60, 90]),
            e1_u=rng.choice([0, 25, 40, 60]), p3_u=rng.choice([0, 60, 90, 110, 140, 260]),
            rec_S=rng.choice([0, 100, 150]), j_lat=rng.choice([0, 5, 10]),
            d_nt=rng.choice([0, 30, 50]), d_nb=rng.choice([0, 40]), l_n=rng.choice([0, 100, 150]),
            V_Ed=rng.choice([0, 50, 200, 400, 900]), N_Ed=rng.choice([-60, 0, 80, 250]),
            nat_N=rng.choice([E.NAT_ELU, E.NAT_TYING]),
            boulon_u=rng.choice(["M12", "M16", "M20", "M24"]),
            classe=rng.choice(["4.6", "8.8", "10.9"]), cat=rng.choice(["A", "B", "C"]),
            trou=rng.choice(["Normal", "Surdimensionné"]),
            prof_S=rng.choice(["IPE 200", "IPE 400", "HEA 300", "HEB 500"]),
            prof_P=rng.choice(["HEB 300", "HEA 400", "HEB 600", "IPE 300"])))
    except Exception as e:  # noqa: BLE001
        nerr += 1
        if nerr == 1:
            print("      première exception :", type(e).__name__, e, "au tirage", i)
chk("aucune exception sur 300 tirages", nerr == 0, str(nerr))
chk("compute({}) et compute(None) passent", moteur.compute({}).verified and moteur.compute(None).verified)
chk("un profil PERSO sans dimensions ne lève pas",
    moteur.compute(dict(prof_P=PERSO)).statut.startswith("ASSEMBLAGE"))
chk("charger_json ignore les clés inconnues et complète les absentes",
    E.charger_json({"V_Ed": 300, "inconnue": 1})["V_Ed"] == 300
    and len(E.charger_json({})) == 66 and E.charger_json({})["appui"] == E.APPUI_AILE
    and E.charger_json({})["nat_N"] == E.NAT_ELU)

# ================================================================
print("\n=== 10. Régression figée (tests/ref_pac.json) ===")
with open("tests/ref_pac.json", encoding="utf-8") as fh:
    REF = json.load(fh)
ndiff = 0
for nom, cas in REF.items():
    Rr = moteur.compute(cas["inp"])
    for k, (ed, rdv, eta) in cas["checks"].items():
        c = Rr.ck.get(k)
        if c is None or not c.active or abs(c.Ed - ed) > 1e-9 * max(1, abs(ed)) \
           or abs(c.Rd - rdv) > 1e-9 * max(1, abs(rdv)) or abs(c.eta - eta) > 1e-12 * max(1, abs(eta)):
            ndiff += 1
            print("   ", nom, k, "attendu", (ed, rdv, eta), "obtenu",
                  (c.Ed, c.Rd, c.eta) if c and c.active else "inactif")
    for k, v in cas["scal"].items():
        if abs(Rr[k] - v) > 1e-9 * max(1, abs(v)):
            ndiff += 1
            print("   ", nom, "scalaire", k, "attendu", v, "obtenu", Rr[k])
    if (Rr.gov.key if Rr.gov else None) != cas["gov"] or Rr.verified != cas["verified"] \
       or Rr.geo_ok != cas["geo_ok"] or Rr.dist_ok != cas["dist_ok"] \
       or sorted(a.id for a in Rr.alerts) != cas["alerts"]:
        ndiff += 1
        print("   ", nom, "synthèse divergente")
chk(f"RÉGRESSION CALCUL : 0 différence sur les {len(REF)} cas figés "
    f"({sum(len(c['checks']) for c in REF.values())} vérifications)", ndiff == 0, str(ndiff))
chk("les cas figés couvrent les deux appuis, le gauge refusé, la platine qui ne tient pas, la recoupe "
    "suffisante et insuffisante, le grugeage, NEd ELU, le tying (aile et âme), les catégories B et C, "
    "le trou surdimensionné, un profil PERSO, l'exemple publié, hp < 1,36·p3, un cas NON VÉRIFIÉ et "
    "le prédimensionnement",
    len(REF) >= 13
    and {"defaut_aile", "ame_b0", "ame_recoupe", "ame_recoupe_insuffisante", "ame_bp_refuse",
         "aile_p3_90_refuse", "aile_p3_large", "grugeage_double", "N_elu", "N_tying", "N_tying_ame",
         "cat_B", "cat_C", "trou_surdim", "WE_endplate", "perso", "hp_court", "non_verifie",
         "predim"} <= set(REF), str(sorted(set(REF))))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

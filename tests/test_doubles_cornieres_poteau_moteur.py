# -*- coding: utf-8 -*-
"""Moteur de l'assemblage POUTRE–POTEAU par DOUBLES CORNIÈRES d'âme.

Chaque garantie rougit si on la retire :
  1. HÉRITAGE : l'exemple publié (MSB Part 5 §4.4, Worked Example: Angle Web
     Cleats) rejoué avec appui sur l'AILE d'un poteau — les SEIZE valeurs
     publiées sortent inchangées, et l'ANCRE du tronçon en T (modes 1/2/3)
     aussi, parce qu'aucune ne dépend du support.
  2. CAS (a) AILE, le défaut du module : les nombres du §11.4 BM2 de l'étude,
     recalculés à la main dans la suite — dont la PRESSION DIAMÉTRALE de
     l'aile (αb sans pince d'extrémité, k1 du gauge et du bout d'aile).
  3. CAS (a) TYING (§11.4 BM3) : le tronçon en T des ailes A ET celui de
     l'AILE DU POTEAU (Tab. 6.4, aile non raidie), l'âme du poteau en
     traction transversale (ω = 1), le poinçonnement — les cornières
     gouvernent, rapport 1,915.
  4. CAS (b) ÂME (§11.4 BM4) : une L100 NE PASSE PAS, une L90 passe ;
     pression diamétrale de l'âme, cisaillement local brut et net (et = 5·d),
     les trois conditions géométriques du cas (b).
  5. ALERTES : les neuves (p3_congeP, bA_congeP, p3_libreP, aile_poutre,
     ame_hp) et celles qui ont DISPARU (P_web, dnt_min, dnb_min, ln_min,
     gap_neg, d_top).
  6. BRIQUES PARTAGÉES : le moteur ne recopie pas la mécanique de la
     famille — il appelle acier/assemblages/mecanique.py, et les grandeurs
     recalculées par ce fichier lui sont identiques.
  7. PRÉDIM : le gauge p3 encadré (congé du poteau ↔ Tableau 3.3),
     apply_solution recopie tout.
  8. ROBUSTESSE : 300 tirages aléatoires sans exception, dans les deux cas.
  9. RÉGRESSION FIGÉE : tests/ref_dcc.json.

Lancement : python3 tests/test_doubles_cornieres_poteau_moteur.py
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
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau import moteur, entrees as E  # noqa: E402
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau import benchmark as BM  # noqa: E402

OK, KO = [], []
S3 = math.sqrt(3)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * abs(b)


# ================================================================
print("=== 1. HÉRITAGE : l'exemple publié MSB Part 5 §4.4 sur un POTEAU ===")
R = moteur.compute(BM.WE)


def vr(k):
    return R.u.V_Ed / R.ck[k].eta


chk("géométrie : z = 50 mm, p3 = tw,S + 2·gA = 109 mm, pince au bout d'aile 95,5 mm",
    R.z_S == 50 and R.p_3 == 109 and R.e_P_fl == 95.5, f"{R.z_S} {R.p_3} {R.e_P_fl}")
chk("un boulon en cisaillement Fv,Rd = 94 kN (p. 5-70, arrondi de la source)",
    pres(R.Fv_Rd, 94, 0.01), str(R.Fv_Rd))
chk("groupe S excentré – cisaillement des boulons VRd = 962 kN (p. 5-70)",
    pres(vr("bv_S"), 962, 0.01), str(vr("bv_S")))
chk("pression diamétrale cornière Fb,ver,Rd = Fb,hor,Rd = 105 kN (p. 5-71, ≤ 1 %)",
    pres(R.Fbv_B, 105, 0.01) and pres(R.Fbh_B, 105, 0.01), f"{R.Fbv_B} {R.Fbh_B}")
chk("groupe S – pression diamétrale sur les cornières VRd = 1 075 kN (p. 5-71)",
    pres(vr("pdB"), 1075, 0.01), str(vr("pdB")))
chk("pression diamétrale âme Fb,ver = 125 et Fb,hor = 94 kN (p. 5-72/73)",
    pres(R.Fbv_S, 125, 0.01) and pres(R.Fbh_S, 94, 0.01), f"{R.Fbv_S} {R.Fbh_S}")
chk("groupe S – pression diamétrale sur l'âme VRd = 583 kN (p. 5-73)",
    pres(vr("pdS"), 583, 0.01), str(vr("pdS")))
chk("côté support – FRd = 0,8·ns·Fv,Rd = 902 kN (p. 5-74)",
    pres(vr("bv_P"), 902, 0.01), str(vr("bv_P")))
chk("cornières – cisaillement brut 1 076, net 1 184, bloc 954 kN (p. 5-74/75/76)",
    pres(R.ck["cgB"].Rd, 1076, 0.01) and pres(R.ck["cnB"].Rd, 1184, 0.01)
    and pres(R.ck["cbB"].Rd, 954, 0.01) and pres(R.ck["cbA"].Rd, 954, 0.01),
    f"{R.ck['cgB'].Rd} {R.ck['cnB'].Rd} {R.ck['cbB'].Rd}")
chk("âme – brut 953, net 956, bloc 501 kN (p. 5-77 ; aire calculée +0,25 %)",
    pres(R.ck["vgS"].Rd, 953, 0.01) and pres(R.ck["vnS"].Rd, 956, 0.01)
    and pres(R.ck["vbS"].Rd, 501, 0.01),
    f"{R.ck['vgS'].Rd} {R.ck['vnS'].Rd} {R.ck['vbS'].Rd}")
chk("VÉRIFIÉ ; dimensionnante = rupture de bloc de l'âme portée",
    R.verified and R.gov.key == "vbS", R.gov.key)
Rt = moteur.compute(BM.WE_T)
chk("ANCRE du tronçon en T (p. 5-78/79) : Ft,Rd,u 160, Σleff 430 mm, Mpl 4,2 kNm",
    pres(Rt.Ft_Rd, 160, 0.01) and Rt.leff_T == 430 and pres(Rt.Mpl_T, 4.2, 0.01),
    f"{Rt.Ft_Rd} {Rt.leff_T} {Rt.Mpl_T}")
chk("ANCRE des modes 1 / 2 / 3 : 696 / 1 190 / 1 920 kN (arrondis de la source, ≤ 1 %)",
    pres(Rt.FT_1, 696, 0.01) and pres(Rt.FT_2, 1190, 0.01) and pres(Rt.FT_3, 1920, 0.01),
    f"{Rt.FT_1} {Rt.FT_2} {Rt.FT_3}")
chk("cas (a) : AUCUN cisaillement local du poteau (vlP et vnP inactifs, étude §4.1)",
    not R.ck["vlP"].active and not R.ck["vnP"].active)
chk("cas (a) : le groupe « Poteau porteur » porte la pression diamétrale de l'AILE",
    R.ck["pdP"].active and pres(R.t_appui, 15, 1e-12))

# ================================================================
print("\n=== 2. CAS (a) AILE : le cas par défaut (§11.4 BM2) ===")
R0 = moteur.compute({})
fy, fu, bw = 355.0, 510.0, 0.9
d, d0, As, dm, dw = 20.0, 22.0, 245.0, 31.8, 37.0
Fv = 0.6 * 800 * As / 1.25 / 1000
hP, bP, twP, tfP, rP = 300.0, 300.0, 11.0, 19.0, 27.0
twS = 8.6
bA = bB = 100.0
tc, rc = 10.0, 12.0
hc, n1, p1, e1, e2b, gh, zc, gA = 290.0, 4, 70.0, 40.0, 40.0, 10.0, 40.0, 50.0
p3 = twS + 2 * gA
z = gh + e2b
MS = 200 * z / 1000
Ip = n1 * p1 ** 2 * (n1 ** 2 - 1) / 12
Fz = 200 / n1
Fx = MS * 1000 * ((n1 - 1) * p1 / 2) / Ip
FS = math.hypot(Fz, Fx)
FP = 100 / n1
chk("gauge p3 = tw,S + 2·gA = 108,6 mm ; gauge minimal twP + 2·rP + dw = 102 mm (DE JUSTESSE)",
    R0.p_3 == p3 and R0.p3_congeP == twP + 2 * rP + dw and R0.p_3 > R0.p3_congeP,
    f"{R0.p_3} {R0.p3_congeP}")
chk("pince au bout d'aile e = (bP − p3)/2 = 95,7 mm ; pinces e2 des ailes A et B = 50 / 50",
    pres(R0.e_P_fl, (bP - p3) / 2, 1e-12) and R0.e2a_P == 50 and R0.e2a_S == 50,
    f"{R0.e_P_fl} {R0.e2a_P} {R0.e2a_S}")
chk("trou hors du congé de la cornière : gA − d0/2 = 39 ≥ tc + rc = 22 (gA mini 31 mm)",
    R0.gA_min == tc + rc + d0 / 2 and R0.g_A - d0 / 2 >= tc + rc, str(R0.gA_min))
chk("z = gh + e2,b = 50 mm ; MS = 10,0 kNm ; Ip = 24 500 mm²",
    R0.z_S == z and R0.M_S == MS and R0.Ip_S == Ip, f"{R0.z_S} {R0.M_S} {R0.Ip_S}")
chk("F_S = 65,854 kN ; taux du groupe S (2 plans) = 0,350",
    pres(R0.F_S, FS, 1e-12) and pres(R0.ck["bv_S"].eta, FS / (2 * Fv), 1e-12), str(R0.F_S))
chk("F_P = (VEd/2)/n1P = 25 kN ; taux du groupe P (k_rot·Fv,Rd = 75,264) = 0,332",
    R0.F_P == FP and pres(R0.ck["bv_P"].eta, FP / (0.8 * Fv), 1e-12), str(R0.ck["bv_P"].eta))
chk("cornières : VRd,g = 936,035 (1/1,27) ; VRd,n = 951,658 kN",
    pres(R0.ck["cgB"].Rd, 2 * hc * tc * fy / (1.27 * S3) / 1000, 1e-12)
    and pres(R0.ck["cnB"].Rd, 2 * tc * (hc - n1 * d0) * fu / (S3 * 1.25) / 1000, 1e-12),
    f"{R0.ck['cgB'].Rd} {R0.ck['cnB'].Rd}")
chk("ailes B : Ant = 390 et Anv = 1730 mm² ; VRd,b = 868,279 kN (2 cornières)",
    R0.Ant_B == 390 and R0.Anv_B == 1730
    and pres(R0.ck["cbB"].Rd, 2 * (0.5 * fu * 390 / 1.25 + fy * 1730 / S3) / 1000, 1e-12),
    f"{R0.Ant_B} {R0.Anv_B} {R0.ck['cbB'].Rd}")
chk("ailes B : flexion MEd = 5,0 / MRd = 49,759 kNm → η = 0,100",
    R0.ck["flB"].Ed == 5.0 and pres(R0.ck["flB"].Rd, tc * hc ** 2 / 6 * fy / 1e6, 1e-12),
    f"{R0.ck['flB'].Ed} {R0.ck['flB'].Rd}")
_abv_C = min(min(e1, 40.0) / (3 * d0), p1 / (3 * d0) - 0.25, 800 / fu, 1)
_Fbv_C = min(2.8 * 50.0 / d0 - 1.7, 2.5) * _abv_C * fu * d * tc / 1.25 / 1000
_abh_C = min(50.0 / (3 * d0), 800 / fu, 1)
_Fbh_C = min(2.8 * min(e1, 40.0) / d0 - 1.7, 1.4 * p1 / d0 - 1.7, 2.5) * _abh_C * fu * d * tc / 1.25 / 1000
_ipd_B = math.hypot(Fz / 2 / _Fbv_C, Fx / 2 / _Fbh_C)
chk("pression diamétrale des ailes A et B : Fb,ver,Rd = 123,636 kN ; η ailes B = 0,2452 (effort F/2)",
    pres(R0.Fbv_B, _Fbv_C, 1e-12) and pres(R0.Fbv_A, _Fbv_C, 1e-12)
    and pres(R0.Fbh_B, _Fbh_C, 1e-12) and pres(R0.ck["pdB"].eta, _ipd_B, 1e-12)
    and pres(_ipd_B, 0.2452, 0.001), f"{R0.Fbv_B} {R0.ck['pdB'].eta}")
chk("âme portée : e1,b = 80 et he = 110 mm ; taux de pression diamétrale = 0,5349",
    R0.e1b_S == 80 and R0.h_e == 110 and pres(R0.ck["pdS"].eta, 0.5348613587240804, 1e-12),
    str(R0.ck["pdS"].eta))
# --- la PRESSION DIAMÉTRALE de l'AILE du poteau : αb sans pince d'extrémité,
# --- k1 du gauge p3 ET du bout d'aile (étude §4.2)
_abv_P = min(p1 / (3 * d0) - 0.25, 800 / fu, 1)
_k1_P = min(2.8 * ((bP - p3) / 2) / d0 - 1.7, 1.4 * p3 / d0 - 1.7, 2.5)
_Fb_Pa = _k1_P * _abv_P * fu * d * tfP / 1.25 / 1000
chk("AILE DU POTEAU : αb = 0,810606 (aucune pince d'extrémité), k1 = 2,5, Fb,Rd = 314,191 kN",
    pres(R0.abv_P, _abv_P, 1e-12) and R0.k1v_P == 2.5 and pres(R0.Fbv_P, _Fb_Pa, 1e-12)
    and pres(R0.Fbv_P, 314.1909, 1e-6), str(R0.Fbv_P))
chk("AILE DU POTEAU : taux = 25 / 314,191 = 0,0796", pres(R0.ck["pdP"].eta, FP / _Fb_Pa, 1e-12),
    str(R0.ck["pdP"].eta))
chk("cas par défaut VÉRIFIÉ, dimensionnante = pression diamétrale de l'âme portée (η = 0,535)",
    R0.verified and R0.gov.key == "pdS" and pres(R0.eta_max, 0.5349, 0.001))
chk("moment apporté au POTEAU : e_col = hP/2 + z = 200 mm ; VEd·e_col = 40,0 kNm (information)",
    R0.e_col == 200 and R0.M_col == 40.0, f"{R0.e_col} {R0.M_col}")
chk("Tableau 3.3 : 9 lignes actives par défaut, dont le GAUGE p3 et la pince au bout d'aile",
    [x.id for x in R0.dist] == ["S_e1c", "S_e2c", "S_e2b", "S_p1", "P_e1c", "P_e2c", "P_p1",
                                "P_p3", "P_eP"] and all(x.ok for x in R0.dist),
    str([x.id for x in R0.dist]))
chk("defaults() : 88 clés, appui sur l'aile, d_top n'existe plus",
    len(E.defaults()) == 88 and E.defaults()["appui"] == E.APPUI_AILE and "d_top" not in E.defaults(),
    str(len(E.defaults())))
chk("γM0 par défaut = 1,00 (ANB belge) ; γMu par défaut = 1,10 (guides)",
    R0.u.g_M0 == 1 and R0.u.g_Mu == 1.1)

# ================================================================
print("\n=== 3. CAS (a) TYING : les deux tronçons en T (§11.4 BM3) ===")
R3 = moteur.compute(BM.BM3)
Ftu = 0.9 * 800 * As / 1.1 / 1000
SFt = 2 * n1 * Ftu
ew = dw / 4
chk("tying : NEd est sorti de la combinaison ELU (N_elu = 0) et vérifié seul (N_ty = 300)",
    R3.tying and R3.N_elu == 0 and R3.N_ty == 300 and R3.g_T == 1.1 and R3.g_B == 1.1)
chk("tying : Ft,Rd,u = 160,364 kN ; ΣFt,Rd,u = 2·n1P·Ft,Rd,u = 1 282,91 kN",
    pres(R3.Ft_Rd_g, Ftu, 1e-12) and pres(R3.SFt_T, SFt, 1e-12), f"{R3.Ft_Rd_g} {R3.SFt_T}")
m_A = (p3 - twS - 2 * tc - 1.6 * rc) / 2
n_A = min(50.0, 1.25 * m_A)
chk("ailes A : m = 30,400 ; n = 38,000 ; Σleff = 290 mm ; Mpl,1 = 3,3614 kNm",
    pres(R3.m_T, m_A, 1e-12) and pres(R3.n_T, n_A, 1e-12) and R3.leff_T == 290
    and pres(R3.Mpl_T, 0.25 * 290 * tc ** 2 * fu / 1.1 / 1e6, 1e-12),
    f"{R3.m_T} {R3.n_T} {R3.leff_T} {R3.Mpl_T}")
_Mpl_A = 0.25 * 290 * tc ** 2 * fu / 1.1
_FT1_A = (8 * n_A - 2 * ew) * _Mpl_A / (2 * m_A * n_A - ew * (m_A + n_A)) / 1000
_FT2_A = (2 * _Mpl_A + n_A * SFt * 1000) / (m_A + n_A) / 1000
chk("ailes A : modes 572,015 / 811,013 / 1 282,91 kN → FT,Rd = 572,015 ; η = 0,5245",
    pres(R3.FT_1, _FT1_A, 1e-12) and pres(R3.FT_2, _FT2_A, 1e-12) and pres(R3.FT_3, SFt, 1e-12)
    and pres(R3.ck["tsA"].Rd, _FT1_A, 1e-12) and pres(R3.ck["tsA"].eta, 300 / _FT1_A, 1e-12),
    f"{R3.FT_1} {R3.FT_2} {R3.FT_Rd}")
m_P = (p3 - twP) / 2 - 0.8 * rP
e_P = (bP - p3) / 2
n_Pf = min(e_P, 1.25 * m_P)
_leff_P = min(min(n1 * 2 * math.pi * m_P, n1 * (4 * m_P + 1.25 * e_P)), min(n1 * 2 * p1, n1 * p1))
_Mpl_P = 0.25 * _leff_P * tfP ** 2 * fu / 1.1
_FT1_P = (8 * n_Pf - 2 * ew) * _Mpl_P / (2 * m_P * n_Pf - ew * (m_P + n_Pf)) / 1000
_FT2_P = (2 * _Mpl_P + n_Pf * SFt * 1000) / (m_P + n_Pf) / 1000
chk("AILE DU POTEAU : m = 27,200 ; n = 34,000 ; Σleff = 280 mm (Tab. 6.4, rangées en groupe n1·p1)",
    pres(R3.m_P, m_P, 1e-12) and pres(R3.n_Pf, n_Pf, 1e-12)
    and R3.leff1_P == 280 and R3.leff2_P == 280, f"{R3.m_P} {R3.n_Pf} {R3.leff1_P}")
chk("AILE DU POTEAU : Mpl,1 = 11,7161 kNm ; modes 2 314,03 / 1 095,61 / 1 282,91 kN",
    pres(R3.Mpl_P, _Mpl_P / 1e6, 1e-12) and pres(R3.FTP_1, _FT1_P, 1e-12)
    and pres(R3.FTP_2, _FT2_P, 1e-12) and pres(R3.FTP_3, SFt, 1e-12),
    f"{R3.Mpl_P} {R3.FTP_1} {R3.FTP_2}")
chk("AILE DU POTEAU : FT,Rd = mode 2 = 1 095,61 kN → η = 0,274 ; LES CORNIÈRES GOUVERNENT (1,915)",
    pres(R3.ck["tfP"].Rd, _FT2_P, 1e-12) and pres(R3.ck["tfP"].eta, 300 / _FT2_P, 1e-12)
    and pres(R3.FTP_Rd / R3.FT_Rd, 1.915345, 1e-5), str(R3.FTP_Rd / R3.FT_Rd))
chk("âme du poteau en traction transversale (ω = 1, beff = Σleff) = 1 428,0 kN",
    pres(R3.ck["twtP"].Rd, 280 * twP * fu / 1.1 / 1000, 1e-12), str(R3.ck["twtP"].Rd))
chk("poinçonnement : aile A de la cornière 277,911 kN ; AILE DU POTEAU 528,031 kN",
    pres(R3.Bp_Rd, 0.6 * math.pi * dm * tc * fu / 1.1 / 1000, 1e-12)
    and pres(R3.ck["ppP"].Rd, 0.6 * math.pi * dm * tfP * fu / 1.1 / 1000, 1e-12),
    f"{R3.Bp_Rd} {R3.ck['ppP'].Rd}")
chk("tying : la force de chaînage CISAILLE le groupe S (2 plans, γMu) — ligne bvT_S",
    R3.ck["bvT_S"].active and pres(R3.ck["bvT_S"].Ed, 300 / n1, 1e-12)
    and pres(R3.ck["bvT_S"].Rd, 2 * 0.6 * 800 * As / 1.1 / 1000, 1e-12), str(R3.ck["bvT_S"].Rd))
chk("tying : ailes B en traction nette 1 685,78 kN ; âme portée (hauteur hc) 724,886 kN",
    pres(R3.ck["tnB"].Rd, 2 * 0.9 * tc * (hc - n1 * d0) * fu / 1.1 / 1000, 1e-12)
    and pres(R3.ck["tnS"].Rd, 0.9 * twS * (hc - n1 * d0) * fu / 1.1 / 1000, 1e-12),
    f"{R3.ck['tnB'].Rd} {R3.ck['tnS'].Rd}")
chk("tying : bt_P prend NEd/(2·nP) et min(Ft,Rd,u ; Bp,Rd cornière)",
    pres(R3.ck["bt_P"].Ed, 300 / (2 * n1), 1e-12)
    and pres(R3.ck["bt_P"].Rd, min(Ftu, R3.Bp_Rd), 1e-12))
chk("tying : l'interaction V + T de l'ELU est INACTIVE (la force de chaînage est seule)",
    not R3.ck["bi_P"].active)
chk("cas (a) : le groupe « Poteau porteur » porte pdP, tfP, twtP et ppP (pas vlP ni vnP)",
    [c.key for c in R3.checks if c.grp == "Poteau porteur" and c.active]
    == ["pdP", "tfP", "twtP", "ppP"],
    str([c.key for c in R3.checks if c.grp == "Poteau porteur" and c.active]))
Re = moteur.compute(dict(N_Ed=80))
chk("ELU fondamental : NEd reste combiné à VEd (γM2n pour les nettes), bvT_S inactif, tsA actif",
    not Re.tying and Re.N_elu == 80 and Re.g_T == 1.25 and not Re.ck["bvT_S"].active
    and Re.ck["tsA"].active and Re.ck["tfP"].active and Re.ck["bi_P"].active)
chk("ELU : NEd entre dans Fx du boulon le plus sollicité du groupe S", Re.Fx_S > R0.Fx_S)
chk("une compression (NEd < 0) n'a pas de nature : nat_N invisible, pas de vérification de traction",
    not E.traction(dict(N_Ed=-60)) and not moteur.compute(dict(N_Ed=-60)).ck["tsA"].active)

# ================================================================
print("\n=== 4. CAS (b) ÂME : une L100 ne passe pas, une L90 passe (§11.4 BM4) ===")
R4 = moteur.compute(BM.BM4)
chk("cas (b) : largeur droite entre congés 208 mm ; demi-largeur 104 ; aile A maximale 99,7 mm",
    R4.libre_droit == 208 and R4.demi_droit == 104 and pres(R4.bA_max, 99.7, 1e-12),
    f"{R4.libre_droit} {R4.demi_droit} {R4.bA_max}")
chk("L90×90×9 : le bout de l'aile A dégage le congé (4,3 + 90 = 94,3 ≤ 104) — aucune alerte",
    not any(a.id == "bA_congeP" for a in R4.alerts) and pres(R4.tw_S / 2 + R4.b_A, 94.3, 1e-12))
chk("L100×100×10 en cas (b) : 4,3 + 100 = 104,3 > 104 → alerte BLOQUANTE bA_congeP",
    any(a.id == "bA_congeP" and a.block for a in moteur.compute(dict(appui=E.APPUI_AME)).alerts)
    and not moteur.compute(dict(appui=E.APPUI_AME)).geo_ok)
chk("l'écrou dégage le congé : p3/2 + dw/2 = 72,8 ≤ 104 (marge 31,2 mm) ; p3 maximal 171 mm",
    pres(R4.p_3 / 2 + R4.d_w / 2, 72.8, 1e-12) and pres(R4.p3_max_ame, 171.0, 1e-12),
    f"{R4.p_3 / 2 + R4.d_w / 2} {R4.p3_max_ame}")
chk("… un gauge trop grand en cas (b) : alerte BLOQUANTE p3_libreP",
    any(a.id == "p3_libreP" and a.block
        for a in moteur.compute(dict(appui=E.APPUI_AME, corn_u="L150x150x10", gA_u=90)).alerts))
chk("cas (b) : le gauge minimal du congé de l'AILE ne s'applique PAS (p3_congeP inactive)",
    not any(a.id == "p3_congeP" for a in moteur.compute(dict(appui=E.APPUI_AME, corn_u="L90x90x9",
                                                             gA_u=40)).alerts))
_k1_Pb = min(1.4 * p3 / d0 - 1.7, 2.5)
_Fb_Pb = _k1_Pb * _abv_P * fu * d * twP / 1.25 / 1000
chk("ÂME DU POTEAU : aucun bord libre — k1 = 2,5 du gauge seul, Fb,Rd = 181,900 kN, η = 0,1374",
    pres(R4.Fbv_P, _Fb_Pb, 1e-12) and pres(R4.Fbv_P, 181.8996, 1e-5)
    and pres(R4.ck["pdP"].eta, 25 / _Fb_Pb, 1e-12), str(R4.Fbv_P))
_Av_P = twP * (5 * d + (n1 - 1) * p1 + min(p3 / 2, 5 * d))
chk("ÂME DU POTEAU : et = 5·d = 100, eb = min(p3/2 ; 5d) = 54,3 → Av = 4 007,3 mm²",
    R4.et_P == 100 and pres(R4.eb_P, 54.3, 1e-12) and pres(R4.Av_P, _Av_P, 1e-12),
    f"{R4.et_P} {R4.eb_P} {R4.Av_P}")
chk("ÂME DU POTEAU : cisaillement local BRUT 821,334 kN (η = 0,122) et NET 715,934 kN (η = 0,140)",
    R4.ck["vlP"].active and pres(R4.ck["vlP"].Rd, _Av_P * fy / S3 / 1000, 1e-12)
    and R4.ck["vlP"].Ed == 100
    and pres(R4.ck["vnP"].Rd, (_Av_P - n1 * d0 * twP) * fu / (S3 * 1.25) / 1000, 1e-12),
    f"{R4.ck['vlP'].Rd} {R4.ck['vnP'].Rd}")
chk("cas (b) : ni tronçon en T de l'aile ni âme en traction transversale (ils sont du cas a)",
    not R4.ck["tfP"].active and not R4.ck["twtP"].active
    and not moteur.compute(dict(BM.BM4, N_Ed=300, nat_N=E.NAT_TYING)).ck["tfP"].active)
chk("cas (b) : poinçonnement de l'ÂME (t = twP) = 305,702 kN en tying",
    pres(moteur.compute(dict(BM.BM4, N_Ed=300, nat_N=E.NAT_TYING)).ck["ppP"].Rd,
         0.6 * math.pi * dm * twP * fu / 1.1 / 1000, 1e-12))
chk("cas (b) : l'IPE 400 (bS = 180) passe entre les ailes du HEB 300 (252 mm) — pas de recoupe",
    R4.passe_S and R4.rec_ok and R4.libre_ailes == 252
    and not any(a.id == "aile_poutre" for a in R4.alerts))
chk("cas (b) : recoupe minimale (bP − twP)/2 + j − gh = 139,5 mm",
    pres(R4.rec_min, (bP - twP) / 2 + 5 - gh, 1e-12), str(R4.rec_min))
Ri = moteur.compute(dict(appui=E.APPUI_AME, corn_u="L90x90x9", prof_S="HEA 300", z_C=45,
                         LC_u=190, n1S_u=3, n1P_u=3, p1S_u=60, p1P_u=60, e1S_u=35, e1P_u=35,
                         V_Ed=125))
chk("cas (b), HEA 300 portée (bS = 300 > 252) sans recoupe : alerte BLOQUANTE aile_poutre",
    any(a.id == "aile_poutre" and a.block for a in Ri.alerts) and not Ri.geo_ok)
Rr = moteur.compute(dict(appui=E.APPUI_AME, corn_u="L90x90x9", prof_S="HEA 300", rec_S=145,
                         z_C=45, LC_u=190, n1S_u=3, n1P_u=3, p1S_u=60, p1P_u=60, e1S_u=35,
                         e1P_u=35, V_Ed=125))
chk("… recoupe de 145 mm ≥ 139,5 : l'alerte disparaît et l'assemblage est vérifié",
    not any(a.id == "aile_poutre" for a in Rr.alerts) and Rr.verified)
chk("cas (b) : l'alerte d'information ame_hp (flexion hors plan non vérifiée) est toujours là",
    any(a.id == "ame_hp" and not a.block for a in Rr.alerts)
    and not any(a.id == "ame_hp" for a in R0.alerts))
chk("cas (b) : le groupe « Poteau porteur » porte pdP, vlP et vnP",
    [c.key for c in R4.checks if c.grp == "Poteau porteur" and c.active] == ["pdP", "vlP", "vnP"],
    str([c.key for c in R4.checks if c.grp == "Poteau porteur" and c.active]))
chk("benchmark du module : statut VALIDÉ, pire écart 0,76 % (arrondis de la source)",
    BM.run_bench()["statut"] == "VALIDÉ" and BM.run_bench()["worst"] <= 0.01,
    str(BM.run_bench()["worst"]))

# ================================================================
print("\n=== 5. Alertes : les neuves, et celles qui ont disparu ===")
chk("gauge insuffisant sur l'AILE (gA = 40 → p3 = 88,6 < 102) : alerte BLOQUANTE p3_congeP",
    any(a.id == "p3_congeP" and a.block for a in moteur.compute(dict(gA_u=40)).alerts)
    and not moteur.compute(dict(gA_u=40)).geo_ok)
chk("alertes RETIRÉES (fausses pour un poteau) : P_web, dnt_min, dnb_min, ln_min, gap_neg",
    not ({"P_web", "dnt_min", "dnb_min", "ln_min", "gap_neg"}
         & {a.id for cas in ({}, dict(appui=E.APPUI_AME, corn_u="L90x90x9"),
                             dict(LC_u=340, z_C=0), dict(prof_S="HEB 500"),
                             dict(d_nt=50, l_n=10))
            for a in moteur.compute(cas).alerts}))
chk("des cornières sur TOUTE la hauteur du poteau sont permises (plus de « partie droite de "
    "l'âme porteuse »)", moteur.compute(dict(LC_u=330, z_C=35, n1S_u=4, n1P_u=4)).geo_ok)
chk("hc = 140 : pinces basses < 1,2·d0 → alertes bloquantes S_e1bot et P_e1bot",
    {"S_e1bot", "P_e1bot"} <= {a.id for a in moteur.compute(dict(LC_u=140)).alerts})
chk("trou de boulon dans le congé de la cornière : alerte bloquante fillet",
    any(a.id == "fillet" and a.block for a in moteur.compute(dict(gA_u=30, corn_u="L150x150x15")).alerts))
chk("deux files par cornière et boulons tendus : alerte d'information n2P_T (tronçon en T sur p3)",
    any(a.id == "n2P_T" and not a.block
        for a in moteur.compute(dict(n2P_u="2", p2_P=60, corn_u="L150x150x12", gA_u=55,
                                     N_Ed=200, nat_N=E.NAT_TYING)).alerts))
chk("HEd > 10 % de VEd : alerte HEd ; cornières soudées des deux côtés : alerte soude2",
    any(a.id == "HEd" for a in moteur.compute(dict(H_Ed=40)).alerts)
    and any(a.id == "soude2" for a in moteur.compute(dict(fix_P=E.SOUDEE, fix_S=E.SOUDEE)).alerts))
chk("grugeage à 0 par défaut (un poteau n'oblige pas à gruger) et bloc du grugeage inactif",
    E.defaults()["d_nt"] == 0 and E.defaults()["d_nb"] == 0 and E.defaults()["l_n"] == 0
    and R0.cas_g == 0 and not R0.ck["mN"].active and not R0.ck["stN"].active)
Rg = moteur.compute(dict(d_nt=50, d_nb=40, l_n=120, z_C=60, LC_u=240, n1S_u=3, n1P_u=3))
chk("… mais tout le bloc de la poutre grugée s'active à l'identique quand on grugue",
    Rg.cas_g == 2 and Rg.ck["mN"].active and Rg.ck["stN"].active and Rg.ck["stD"].active)

# ================================================================
print("\n=== 6. Les briques de mecanique.py sont APPELÉES, pas recopiées ===")
sg = MEC.section_grugee(dict(h=Rg.h_S, b=Rg.b_S, tw=Rg.tw_S, tf=Rg.tf_S, r=Rg.r_S),
                        Rg.fy_S, 50, 40, 1.0, Rg.u.V_Ed, True, " (SCI P358 Check 7)")
chk("poutre grugée : cas_g, Av, W_N, Mv,N,Rd, ln,max identiques à mecanique.section_grugee",
    sg["cas_g"] == Rg.cas_g and sg["Av"] == Rg.Av_S and sg["W_N"] == Rg.W_N
    and sg["Mv_N"] == Rg.ck["mN"].Rd and sg["ln_max"] == Rg.ln_max and sg["val_N"] == Rg.val_N)
ge = MEC.groupe_excentre(R0.n1_S, R0.n2_S, R0.p1_S, R0.u.p2_S, R0.u.V_Ed, 0, 0)
chk("groupe excentré : Ip et les bras extrêmes identiques à mecanique.groupe_excentre",
    ge["Ip"] == R0.Ip_S and ge["xm"] == R0.xm_S and ge["ym"] == R0.ym_S)
be = MEC.bearing(R0.t_C, R0.fu_C, R0.f_ub, R0.d_b, R0.d_0, min(R0.e1_S, R0.e1bot_S), R0.p1_S,
                 R0.n1_S, R0.e2a_S, R0.u.p2_S, R0.n2_S, 1.25, 1.0,
                 Fz=R0.Fz_S / 2, Fx=R0.Fx_S / 2)
chk("pression diamétrale : Fbv, Fbh et l'interaction identiques à mecanique.bearing",
    be["Fbv"] == R0.Fbv_B and be["Fbh"] == R0.Fbh_B and be["ipd"] == R0.ipd_B)
bp = MEC.bearing(R0.t_appui, R0.fu_P, R0.f_ub, R0.d_b, R0.d_0, None, R0.p1_P, R0.n1_P,
                 R0.e_P_fl, R0.u.p2_P, R0.n2_P, 1.25, 1.0,
                 exV=1.4 * R0.p_3 / R0.d_0 - 1.7, exH=R0.p_3 / (3 * R0.d_0) - 0.25,
                 Fz=R0.Fz_P, Fx=R0.Fx_P)
chk("pression diamétrale de l'AILE du poteau : même appel de mecanique.bearing (e1 = None)",
    bp["Fbv"] == R0.Fbv_P and bp["abv"] == R0.abv_P and bp["k1v"] == R0.k1v_P)
lp = MEC.leff_aile_poteau(R3.n1_P, R3.p1_P, R3.m_P, R3.e_P_fl)
chk("longueurs efficaces de l'aile non raidie : identiques à mecanique.leff_aile_poteau",
    lp == (R3.leff1_P, R3.leff2_P) and moteur.leff_aile_poteau(4, 70, 42.9, 80) == (280.0, 280.0),
    str(lp))
tA = MEC.troncon_T(R3.leff_T, R3.leff_T, R3.t_C, R3.m_T, R3.n_T, R3.ew_T, R3.fu_C, 1.1, R3.SFt_T)
chk("tronçon en T : modes 1 / 2 / 3 identiques à mecanique.troncon_T",
    tA["FT1"] == R3.FT_1 and tA["FT2"] == R3.FT_2 and tA["FT3"] == R3.FT_3)
bt = MEC.bolt_table(R0.n1_S, R0.n2_S, R0.p1_S, R0.u.p2_S, R0.u.V_Ed, 0, R0.M_S, R0.Ip_S)
chk("efforts par boulon : identiques à mecanique.bolt_table",
    [(b.i, b.fx, b.fz) for b in bt] == [(b.i, b.fx, b.fz) for b in R0.tabS])
chk("Σleff,1 = Σleff,2 = n1·p1 quand les rangées sont en groupe (la branche Σcp est inerte)",
    MEC.leff_aile_poteau(4, 70, 27.2, 95.7) == (280.0, 280.0))
chk("une seule rangée : il n'y a pas de groupe, seules les formules isolées s'appliquent",
    MEC.leff_aile_poteau(1, 70, 27.2, 95.7)[0] == min(2 * math.pi * 27.2, 4 * 27.2 + 1.25 * 95.7))

# ================================================================
print("\n=== 7. Prédimensionnement ===")
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT"))
chk("proposition complète au cas par défaut", Rp.pd.found and Rp.pd.msg == "Proposition complète",
    Rp.pd.msg)
chk("le gauge proposé est ENCADRÉ : ≥ twP + 2·rP + dw (aile) et ≤ min(14·tc ; 200) du Tab. 3.3",
    Rp.pd.p3min - 1e-9 <= Rp.pd.p3 <= Rp.pd.p3max + 1e-9
    and Rp.pd.p3min == twP + 2 * rP + Rp.d_w,
    f"{Rp.pd.p3min} ≤ {Rp.pd.p3} ≤ {Rp.pd.p3max}")
chk("… et l'épaisseur de la cornière est relevée s'il le faut pour que la fenêtre existe",
    Rp.pd.tC >= Rp.pd.treq - 1e-9 and 14 * Rp.pd.tC >= Rp.pd.p3 - 1e-9,
    f"tC {Rp.pd.tC} treq {Rp.pd.treq}")
u0 = dict(E.defaults())
u2 = moteur.apply_solution(u0, Rp.pd.pick)
chk("apply_solution : mode VÉRIFICATION, boulon, rangées, hc, gA et la cornière recopiés",
    u2["mode_calc"] == "VÉRIFICATION" and u2["boulon_u"] == Rp.pd.boulon
    and u2["n1S_u"] == Rp.pd.n and u2["n1P_u"] == Rp.pd.n and u2["LC_u"] == Rp.pd.LC
    and u2["gA_u"] == Rp.pd.gA)
Rv = moteur.compute(u2)
chk("la solution recopiée est vérifiée en détail, Tableau 3.3 compris",
    Rv.verified and Rv.dist_ok, (Rv.gov.key if Rv.gov else "") + " " + str(round(Rv.eta_max, 3)))
Rpa = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT", appui=E.APPUI_AME))
chk("le prédimensionnement du cas (b) plafonne l'aile A par la partie droite de l'âme",
    Rpa.pd.leg + Rpa.tw_S / 2 <= Rpa.demi_droit + 1e-9, f"{Rpa.pd.leg} {Rpa.demi_droit}")

# ================================================================
print("\n=== 8. Robustesse (300 tirages aléatoires, les deux cas d'appui) ===")
rng = random.Random(7)
nerr = 0
for i in range(300):
    try:
        moteur.compute(dict(
            appui=rng.choice([E.APPUI_AILE, E.APPUI_AME]),
            fix_P=rng.choice([E.BOULONNEE, E.SOUDEE]), fix_S=rng.choice([E.BOULONNEE, E.SOUDEE]),
            corn_u=rng.choice(["L60x60x6", "L90x90x9", "L100x100x10", "L150x150x15", "L120x80x10"]),
            LC_u=rng.choice([0, 60, 140, 190, 290, 400]), z_C=rng.choice([0, 20, 40, 90]),
            n1S_u=rng.randint(1, 8), n2S_u=rng.choice([1, 2]), p1S_u=rng.choice([0, 40, 60, 90]),
            p2_S=rng.choice([0, 50, 70]), e1S_u=rng.choice([0, 25, 40, 60]),
            e2b_u=rng.choice([0, 30, 40, 80]),
            n1P_u=rng.randint(1, 8), n2P_u=rng.choice([1, 2]), p1P_u=rng.choice([0, 40, 60, 90]),
            p2_P=rng.choice([0, 50, 70]), e1P_u=rng.choice([0, 25, 40, 60]),
            gA_u=rng.choice([0, 30, 50, 55, 90]),
            a_P=rng.choice([0, 4, 5, 8]), lh_P=rng.choice([0, 40]),
            a_S=rng.choice([0, 4, 5, 8]), lh_S=rng.choice([0, 40]),
            g_h=rng.choice([0, 10, 25, 155]), rec_S=rng.choice([0, 100, 150]),
            j_lat=rng.choice([0, 5, 10]),
            d_nt=rng.choice([0, 30, 50]), d_nb=rng.choice([0, 40]), l_n=rng.choice([0, 100, 150]),
            V_Ed=rng.choice([0, 50, 200, 400, 900]), N_Ed=rng.choice([-60, 0, 80, 300]),
            H_Ed=rng.choice([0, 30]), nat_N=rng.choice([E.NAT_ELU, E.NAT_TYING]),
            M_Ed=rng.choice([0, 5]), boulon_u=rng.choice(["M12", "M16", "M20", "M24"]),
            classe=rng.choice(["4.6", "8.8", "10.9"]), cat=rng.choice(["A", "B", "C"]),
            trou=rng.choice(["Normal", "Surdimensionné"]),
            prof_S=rng.choice(["IPE 200", "IPE 400", "HEA 300", "HEB 500"]),
            prof_P=rng.choice(["HEB 300", "HEA 400", "HEB 600", "IPE 300"])))
    except Exception as e:  # noqa: BLE001
        nerr += 1
        if nerr == 1:
            print("      première exception :", type(e).__name__, e, "au tirage", i)
chk("aucune exception sur 300 tirages", nerr == 0, str(nerr))
chk("compute({}) et compute(None) passent",
    moteur.compute({}).verified and moteur.compute(None).verified)
chk("un profil PERSO sans dimensions ne lève pas",
    moteur.compute(dict(prof_P=PERSO)).statut.startswith("ASSEMBLAGE")
    and moteur.compute(dict(corn_u=PERSO)).statut.startswith("ASSEMBLAGE"))
chk("charger_json ignore les clés inconnues et complète les absentes",
    E.charger_json({"V_Ed": 300, "inconnue": 1})["V_Ed"] == 300
    and len(E.charger_json({})) == 88 and E.charger_json({})["appui"] == E.APPUI_AILE)

# ================================================================
print("\n=== 9. Régression figée (tests/ref_dcc.json) ===")
with open("tests/ref_dcc.json", encoding="utf-8") as fh:
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
    f"({sum(len(c['checks']) for c in REF.values())} vérifications, "
    f"{sum(len(c['scal']) for c in REF.values())} scalaires)", ndiff == 0, str(ndiff))
chk("les cas figés couvrent les DEUX appuis × les DEUX fixations, le grugeage, le tying, "
    "les catégories B et C, le trou surdimensionné, deux files et un profil PERSO",
    len(REF) >= 14
    and {"defaut_aile", "ame_b0", "ame_b2_recoupe", "ame_recoupe_insuffisante",
         "ame_L100_bloquant", "grugeage_double", "N_elu", "N_tying", "N_tying_ame", "cat_B",
         "cat_C", "trou_surdim", "soudee_P", "soudee_S", "soudee_deux", "soudee_P_ame",
         "deux_files", "gauge_insuffisant", "perso", "predim",
         "WE_cornieres", "WE_tying"} <= set(REF))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

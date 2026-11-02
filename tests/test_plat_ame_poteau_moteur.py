# -*- coding: utf-8 -*-
"""Moteur de l'assemblage POUTRE–POTEAU par plat d'âme soudé (fin plate).

Chaque garantie rougit si on la retire :
  1. HÉRITAGE : l'exemple publié (MSB Part 5 §3.4, Worked Example: Fin Plate)
     rejoué avec appui sur l'AILE d'un poteau — les DIX valeurs publiées
     sortent inchangées, parce qu'aucune ne dépend du support.
  2. CAS (a) AILE, le défaut du module : les vingt nombres du §11.2 BM2 de
     l'étude, recalculés à la main dans la suite.
  3. CAS (b) ÂME : BM3 (variante b1, qui NE PASSE PAS), BM3' (redimensionné),
     BM4 (about au contact) ; cisaillement local de l'âme du poteau ACTIF en
     cas (b) et INACTIF en cas (a).
  4. TYING : la force de chaînage est vérifiée SEULE, avec γMu, et elle
     CISAILLE les boulons (le plat la reprend dans son plan).
  5. ALERTES : les neuves (aile_poutre, b1_long, ame_hp) et celles qui ont
     DISPARU (pl_web, dnt_min, dnb_min, ln_min, gap_neg, d_top).
  6. BRIQUES PARTAGÉES : le moteur ne recopie pas la mécanique de la
     famille — il appelle acier/assemblages/mecanique.py, et les grandeurs
     recalculées par ce fichier lui sont identiques.
  7. PRÉDIM : proposition complète, apply_solution recopie tout.
  8. ROBUSTESSE : 300 tirages aléatoires sans exception, dans les deux cas.
  9. RÉGRESSION FIGÉE : tests/ref_fpc.json.

Lancement : python3 tests/test_plat_ame_poteau_moteur.py
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
from acier.assemblages.poutre_poteau.plat_ame_poteau import moteur, entrees as E  # noqa: E402
from acier.assemblages.poutre_poteau.plat_ame_poteau import benchmark as BM  # noqa: E402

OK, KO = [], []
S3 = math.sqrt(3)


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * abs(b)


# ================================================================
print("=== 1. HÉRITAGE : l'exemple publié MSB Part 5 §3.4 sur un POTEAU ===")
R = moteur.compute(BM.WE)


def vr(k):
    return R.u.V_Ed / R.ck[k].eta


chk("géométrie : z = 80 mm (gh + e2,b + p2/2), e2 plat = 50, e1,b âme = 90",
    R.z_p == 80 and R.e2_p == 50 and R.e1b_S == 90)
chk("inertie polaire du groupe Ip = 107 000 mm² (p. 5-40)", R.Ip == 107000, str(R.Ip))
chk("groupe excentré – cisaillement des boulons VRd = 584 kN (p. 5-40, arrondis de la source)",
    pres(vr("bv"), 584, 0.01), str(vr("bv")))
chk("pression diamétrale du plat VRd = 605 kN (p. 5-41, arrondis de la source ≤ 1 %)",
    pres(vr("pdL"), 605, 0.011), str(vr("pdL")))
chk("pression diamétrale de l'âme VRd = 624 kN (p. 5-42)", pres(vr("pdS"), 624, 0.01), str(vr("pdS")))
chk("plat – cisaillement section brute VRd = 450 kN (p. 5-43)", pres(R.ck["pcg"].Rd, 450, 0.01), str(R.ck["pcg"].Rd))
chk("plat – cisaillement section nette VRd = 497 kN (p. 5-43)", pres(R.ck["pcn"].Rd, 497, 0.01), str(R.ck["pcn"].Rd))
chk("plat – rupture de bloc VRd = 483 kN (p. 5-43)", pres(R.ck["pcb"].Rd, 483, 0.01), str(R.ck["pcb"].Rd))
chk("âme – cisaillement brut VRd = 953 kN (p. 5-45 ; aire calculée +0,25 %)",
    pres(R.ck["vgS"].Rd, 953, 0.01), str(R.ck["vgS"].Rd))
chk("âme – cisaillement net VRd = 995 kN (p. 5-45 ; même écart d'aire)",
    pres(R.ck["vnS"].Rd, 995, 0.01), str(R.ck["vnS"].Rd))
chk("âme – rupture de bloc VRd = 507 kN (p. 5-45, poutre NON grugée : (n1 − 1)·d0)",
    pres(R.ck["vbS"].Rd, 507, 0.01), str(R.ck["vbS"].Rd))
chk("plat LONG (z = 80 > tp/0,15 = 66,7) : λLT = 38,80 ; pb = 264,65 MPa ; Mb,Rd = 57,16 kNm (γM1 = 1,0)",
    R.long_p and pres(R.lam_LT, 38.797938089542846, 1e-9) and pres(R.pb_LT, 264.647581710414, 1e-9)
    and pres(R.ck["plt"].Rd, 57.163877649449425, 1e-9))
chk("VÉRIFIÉ ; dimensionnante = cisaillement brut du plat (la plus petite résistance du guide)",
    R.verified and R.gov.key == "pcg", R.gov.key)
chk("cas (a) : AUCUNE vérification du poteau porteur (groupe VIDE, étude §4.4)",
    not R.ck["vlP"].active and not [c for c in R.checks if c.grp == "Poteau porteur" and c.active])

# ================================================================
print("\n=== 2. CAS (a) AILE : le cas par défaut (§11.2 BM2) ===")
R0 = moteur.compute({})
fy, fu, bw = 355.0, 510.0, 0.9
d, d0, As, dw = 20.0, 22.0, 245.0, 37.0
Fv = 0.6 * 800 * As / 1.25 / 1000
hp, tp, bp, n1, p1, e1, e2b, gh, zc = 290.0, 10.0, 100.0, 4, 70.0, 40.0, 40.0, 10.0, 40.0
z = gh + e2b
MS = 200 * z / 1000
Ip = n1 * p1 ** 2 * (n1 ** 2 - 1) / 12
Fz = 200 / n1
Fx = MS * 1000 * ((n1 - 1) * p1 / 2) / Ip
Fb = math.hypot(Fz, Fx)
chk("z = gh + e2,b = 50 mm ; MS = 10,0 kNm ; Ip = 24 500 mm²",
    R0.z_p == z and R0.M_S == MS and R0.Ip == Ip, f"{R0.z_p} {R0.M_S} {R0.Ip}")
chk("Fz = 50,000 ; Fx = 42,857 ; F,Ed = 65,854 kN",
    pres(R0.Fz_b, Fz, 1e-12) and pres(R0.Fx_b, Fx, 1e-12) and pres(R0.F_b, Fb, 1e-12), str(R0.F_b))
chk("taux des boulons = 0,700 (Fv,Rd = 94,08 kN)", pres(R0.ck["bv"].eta, Fb / Fv, 1e-12), str(R0.ck["bv"].eta))
chk("pince e2 du plat = 50 mm ; pince basse e1' = 40 mm", R0.e2_p == 50 and R0.e1bot == 40)
_abv = min(min(e1, 40.0) / (3 * d0), p1 / (3 * d0) - 0.25, 800 / fu, 1)
_Fbv = min(2.8 * 50.0 / d0 - 1.7, 2.5) * _abv * fu * d * tp / 1.25 / 1000
_Fbh = min(2.8 * 40.0 / d0 - 1.7, 1.4 * p1 / d0 - 1.7, 2.5) * min(50.0 / (3 * d0), 800 / fu, 1) * fu * d * tp / 1.25 / 1000
chk("plat : αb,ver = 0,606061 ; k1,ver = 2,5 ; Fb,ver = 123,636 et Fb,hor = 154,545 kN",
    pres(R0.abv_L, _abv, 1e-12) and R0.k1v_L == 2.5
    and pres(R0.Fbv_L, _Fbv, 1e-12) and pres(R0.Fbh_L, _Fbh, 1e-12), f"{R0.Fbv_L} {R0.Fbh_L}")
chk("plat : taux de pression diamétrale (interaction) = 0,4904",
    pres(R0.ck["pdL"].eta, math.hypot(Fz / _Fbv, Fx / _Fbh), 1e-12), str(R0.ck["pdL"].eta))
chk("plat : VRd,g = 468,017 (1/1,27) ; VRd,n = 475,829 ; Veff,2,Rd = 434,140 kN",
    pres(R0.ck["pcg"].Rd, hp * tp * fy / (1.27 * S3) / 1000, 1e-12)
    and pres(R0.ck["pcn"].Rd, (hp - n1 * d0) * tp * fu / (S3 * 1.25) / 1000, 1e-12)
    and pres(R0.ck["pcb"].Rd, 0.5 * fu * 390 / 1.25 / 1000 + fy * 1730 / S3 / 1000, 1e-12),
    f"{R0.ck['pcg'].Rd} {R0.ck['pcn'].Rd} {R0.ck['pcb'].Rd}")
chk("plat : Ant = 390 et Anv = 1730 mm²", R0.Ant_p == 390 and R0.Anv_p == 1730)
chk("plat COURT (hp ≥ 2,73·z) et pas LONG (z ≤ tp/0,15 = 66,7) : ni flexion ni déversement",
    R0.court and not R0.long_p and not R0.ck["pfl"].active and not R0.ck["plt"].active)
chk("âme portée : e1,b = 80 et he = 110 mm ; taux de pression diamétrale = 0,5349",
    R0.e1b_S == 80 and R0.h_e == 110 and pres(R0.ck["pdS"].eta, 0.5348613587240804, 1e-12),
    str(R0.ck["pdS"].eta))
_A = 2 * 180 * 13.5 + (400 - 27) * 8.6 + (4 - math.pi) * 441
_Av = max(_A - 2 * 180 * 13.5 + (8.6 + 42) * 13.5, (400 - 27) * 8.6)
chk("âme portée : Av = 4 269,46 mm² ; VRd,g = 875,065 kN",
    pres(R0.Av_S, _Av, 1e-12) and pres(R0.ck["vgS"].Rd, _Av * fy / S3 / 1000, 1e-12), str(R0.ck["vgS"].Rd))
chk("cordons : fvw,d = 261,732 MPa ; qz = 344,83 ; qx = 356,72 ; Fw,Ed = 496,14 N/mm ; η = 0,379",
    pres(R0.fvw, fu / (S3 * bw * 1.25), 1e-12) and pres(R0.qz_w, 200000 / (2 * hp), 1e-12)
    and pres(R0.qx_w, 3 * MS * 1e6 / hp ** 2, 1e-12) and pres(R0.qw, 496.1390260410242, 1e-9),
    f"{R0.qz_w} {R0.qx_w} {R0.qw}")
chk("gorge de pleine résistance recommandée 3,915 mm (a = 5 : pas d'alerte)",
    pres(R0.a_plein, tp * fy * bw * 1.25 / (2 * fu), 1e-12) and not any(a.id == "a_rec" for a in R0.alerts),
    str(R0.a_plein))
chk("moment apporté au POTEAU : e_col = hP/2 + z = 200 mm ; VEd·e_col = 40,0 kNm (information)",
    R0.e_col == 200 and R0.M_col == 40.0, f"{R0.e_col} {R0.M_col}")
chk("cas par défaut VÉRIFIÉ, dimensionnante = les boulons (η = 0,700)",
    R0.verified and R0.gov.key == "bv" and pres(R0.eta_max, 0.7, 0.001))
chk("Tableau 3.3 : 4 lignes actives par défaut (e1p, e2p, e2b, p1 ; p2 inactif)",
    [x.id for x in R0.dist] == ["e1p", "e2p", "e2b", "p1"] and all(x.ok for x in R0.dist),
    str([x.id for x in R0.dist]))
chk("γM1 par défaut = 1,10 (ANB belge) ; γMu par défaut = 1,10 (guides)",
    R0.u.g_M1 == 1.1 and R0.u.g_Mu == 1.1)
chk("defaults() : 71 clés, appui sur l'aile, d_top n'existe plus",
    len(E.defaults()) == 71 and E.defaults()["appui"] == E.APPUI_AILE and "d_top" not in E.defaults(),
    str(len(E.defaults())))

# ================================================================
print("\n=== 3. CAS (b) ÂME : BM3 (b1), BM3' (redimensionné), BM4 ===")
R3 = moteur.compute(BM.BM3)
chk("gh minimal de la variante b1 = (bP − twP)/2 + j = 149,5 mm (HEB 300, j = 5)",
    pres(R3.gh_b1, (300 - 11) / 2 + 5, 1e-12), str(R3.gh_b1))
chk("b1 : z = 195 mm (× 3,9) ; MS = 39,0 kNm ; F,Ed = 174,461 kN ; taux des boulons 1,854 > 1",
    R3.z_p == 195 and R3.M_S == 39 and pres(R3.F_b, 174.46126989643733, 1e-9)
    and pres(R3.ck["bv"].eta, 1.8543927497495465, 1e-9) and not R3.ck["bv"].ok, str(R3.F_b))
chk("b1 : plat LONG — λLT = 54,366 ; pb = 276,516 MPa ; Wel = 140 166,67 mm³ ; Mb,Rd = 35,235 kNm ; η = 1,107",
    R3.long_p and pres(R3.lam_LT, 2.8 * math.sqrt(195 * 290 / 150), 1e-12)
    and pres(R3.pb_LT, 276.5156657700093, 1e-9) and pres(R3.Wel_p, 10 * 290 ** 2 / 6, 1e-12)
    and pres(R3.ck["plt"].Rd, 35.23479922917845, 1e-9) and not R3.ck["plt"].ok,
    f"{R3.lam_LT} {R3.pb_LT} {R3.ck['plt'].Rd}")
chk("b1 : âme du poteau ACTIVE — VRd = twP·hp·fy/(√3·γM0) = 653,820 kN, Ed = VEd/2 = 100, η = 0,153",
    R3.ck["vlP"].active and pres(R3.ck["vlP"].Rd, 11 * 290 * 355 / S3 / 1000, 1e-12)
    and R3.ck["vlP"].Ed == 100 and pres(R3.ck["vlP"].eta, 0.15294722129620533, 1e-9),
    str(R3.ck["vlP"].Rd))
chk("b1 : NON VÉRIFIÉ, et l'alerte b1_long le dit", not R3.verified
    and any(a.id == "b1_long" for a in R3.alerts))
R3p = moteur.compute(BM.BM3P)
chk("b1 redimensionné : Ip = 81 000 mm² ; z = 225 ; MS = 45 kNm ; F,Ed = 76,085 kN ; taux boulons 0,561",
    R3p.Ip == 81000 and R3p.z_p == 225 and R3p.M_S == 45
    and pres(R3p.F_b, 76.08474807008886, 1e-9) and pres(R3p.ck["bv"].eta, 0.5612956508947774, 1e-9),
    f"{R3p.Ip} {R3p.F_b}")
chk("b1 redimensionné : λLT = 51,121 ; pb = 287,879 MPa ; Wel = 204 800 ; Mb,Rd = 53,598 kNm ; η = 0,840",
    pres(R3p.lam_LT, 2.8 * math.sqrt(225 * 320 / (1.5 * 144)), 1e-12)
    and pres(R3p.pb_LT, 287.8785864351627, 1e-9) and R3p.Wel_p == 204800
    and pres(R3p.ck["plt"].Rd, 53.597758638110285, 1e-9)
    and pres(R3p.ck["plt"].eta, 0.8395873473709605, 1e-9), str(R3p.ck["plt"].Rd))
chk("b1 redimensionné : ductilité tp = 12 ≤ 0,5·d = 12 (pas d'alerte tp_duct)",
    R3p.rec_tp and not any(a.id == "tp_duct" for a in R3p.alerts))
# L'étude annonce ce cas « ✔ » sur ces deux taux seulement : l'outil trouve
# DEUX autres défauts, et il faut qu'il les trouve (rouge « a trouvé »).
chk("b1 redimensionné : l'outil TROUVE la pression diamétrale de l'âme portée à 1,148 > 1",
    pres(R3p.ck["pdS"].eta, 1.1476555530588206, 1e-9) and not R3p.ck["pdS"].ok,
    str(R3p.ck["pdS"].eta))
chk("b1 redimensionné : l'outil TROUVE p2 = 60 mm sous le minimum 2,4·d0 = 62,4 mm (M24, Tab. 3.3)",
    not R3p.dist_ok and [x.ok for x in R3p.dist if x.id == "p2"] == [False]
    and pres([x.min for x in R3p.dist if x.id == "p2"][0], 2.4 * 26, 1e-12))
R4 = moteur.compute(BM.BM4)
chk("BM4 : about au contact de l'âme — z retombe à 50 mm et tous les taux de BM2 reviennent",
    R4.z_p == 50 and pres(R4.ck["bv"].eta, R0.ck["bv"].eta, 1e-12)
    and pres(R4.ck["pdS"].eta, R0.ck["pdS"].eta, 1e-12))
chk("BM4 : largeur droite entre congés 208 mm ; demi-largeur 104 ; passage de la semelle 252 mm",
    R4.libre_droit == 208 and R4.demi_droit == 104 and R4.libre_ailes == 252,
    f"{R4.libre_droit} {R4.demi_droit} {R4.libre_ailes}")
chk("BM4 : l'IPE 400 (bS = 180) passe entre les ailes du HEB 300 — aucune recoupe nécessaire",
    R4.passe_S and R4.rec_ok and not any(a.id == "aile_poutre" for a in R4.alerts))
chk("BM4 : recoupe minimale (bP − twP)/2 + j − gh = 139,5 mm si la poutre ne passait pas",
    pres(R4.rec_min, (300 - 11) / 2 + 5 - 10, 1e-12), str(R4.rec_min))
chk("BM4 : VÉRIFIÉ, et l'âme du poteau est vérifiée (η = 0,153)",
    R4.verified and R4.ck["vlP"].active and pres(R4.ck["vlP"].eta, 0.15294722129620533, 1e-9))
chk("cas (b) : le groupe « Poteau porteur » porte exactement UNE vérification",
    len([c for c in R4.checks if c.grp == "Poteau porteur" and c.active]) == 1)
chk("benchmark du module : statut VALIDÉ, pire écart 0,76 % (arrondis de la source)",
    BM.run_bench()["statut"] == "VALIDÉ" and BM.run_bench()["worst"] <= 0.01,
    str(BM.run_bench()["worst"]))

# ================================================================
print("\n=== 4. TYING : la force de chaînage, seule, avec γMu ===")
Rt = moteur.compute(dict(N_Ed=250, nat_N=E.NAT_TYING))
chk("tying : NEd est sorti de la combinaison ELU (N_elu = 0) et vérifié seul (N_ty = 250)",
    Rt.tying and Rt.N_elu == 0 and Rt.N_ty == 250 and Rt.g_T == 1.1)
chk("tying : boulons en CISAILLEMENT — Fv,Rd,u = 106,909 kN/boulon, n·Fv,Rd,u = 427,636 kN",
    Rt.ck["bvT"].active and pres(Rt.Fv_Rd_u, 0.6 * 800 * 245 / 1.1 / 1000, 1e-12)
    and pres(Rt.n_b * Rt.Fv_Rd_u, 427.63636363636357, 1e-9), str(Rt.Fv_Rd_u))
chk("tying : plat, traction section nette 0,9·Anet·fu/γMu = 842,891 kN",
    pres(Rt.ck["ptn"].Rd, 0.9 * (290 - 4 * 22) * 10 * 510 / 1.1 / 1000, 1e-12), str(Rt.ck["ptn"].Rd))
chk("tying : âme portée, traction section nette (hauteur participante hp) = 724,886 kN",
    pres(Rt.ck["tnS"].Rd, 0.9 * 8.6 * (290 - 4 * 22) * 510 / 1.1 / 1000, 1e-12), str(Rt.ck["tnS"].Rd))
chk("tying : cordons — qx = NEd/(2hp) = 431,03 N/mm ; Fw,Rd,u = 1 487,11 N/mm",
    pres(Rt.ck["wT"].Ed, 250000 / 580, 1e-12)
    and pres(Rt.ck["wT"].Rd, 510 / (S3 * 0.9 * 1.1) * 5, 1e-12), str(Rt.ck["wT"].Rd))
chk("tying : rupture de bloc EN TRACTION du plat et de l'âme actives",
    Rt.ck["pbt"].active and Rt.ck["tbS"].active)
chk("tying : côté POTEAU, rien de plus (le plat est soudé, la force est dans son plan)",
    not [c for c in Rt.checks if c.grp == "Poteau porteur" and c.active])
Re = moteur.compute(dict(N_Ed=80))
chk("ELU fondamental : NEd reste combiné à VEd (γM2 pour les sections nettes), bvT et wT inactifs",
    not Re.tying and Re.N_elu == 80 and Re.g_T == 1.25
    and not Re.ck["bvT"].active and not Re.ck["wT"].active and Re.ck["ptn"].active)
chk("ELU : NEd entre dans Fx du boulon le plus sollicité", Re.Fx_b > R0.Fx_b)
chk("une compression (NEd < 0) n'a pas de nature : nat_N invisible, pas de vérification de traction",
    not E.traction(dict(N_Ed=-60)) and not moteur.compute(dict(N_Ed=-60)).ck["ptn"].active)

# ================================================================
print("\n=== 5. Alertes : les neuves, et celles qui ont disparu ===")
Ri = moteur.compute(dict(appui=E.APPUI_AME, prof_S="HEA 300", z_C=45, hp_u=190, n1_u=3,
                         p1_u=60, e1_u=35, V_Ed=125))
chk("cas (b), HEA 300 portée (bS = 300 > 252) sans recoupe : alerte BLOQUANTE aile_poutre",
    any(a.id == "aile_poutre" and a.block for a in Ri.alerts) and not Ri.geo_ok and not Ri.verified)
Rr = moteur.compute(dict(appui=E.APPUI_AME, prof_S="HEA 300", rec_S=145, z_C=45, hp_u=190,
                         n1_u=3, p1_u=60, e1_u=35, V_Ed=125))
chk("… recoupe de 145 mm ≥ 139,5 : l'alerte disparaît et l'assemblage est vérifié",
    not any(a.id == "aile_poutre" for a in Rr.alerts) and Rr.verified)
chk("cas (b) : l'alerte d'information ame_hp (flexion hors plan non vérifiée) est toujours là",
    any(a.id == "ame_hp" and not a.block for a in Rr.alerts)
    and not any(a.id == "ame_hp" for a in R0.alerts))
chk("cas (b) variante b1 : l'alerte d'information b1_long", any(a.id == "b1_long" for a in R3.alerts))
chk("alertes RETIRÉES (fausses pour un poteau) : pl_web, dnt_min, dnb_min, ln_min, gap_neg",
    not ({"pl_web", "dnt_min", "dnb_min", "ln_min", "gap_neg"}
         & {a.id for cas in ({}, dict(appui=E.APPUI_AME), dict(hp_u=400, z_C=0),
                             dict(prof_S="HEB 500"), dict(d_nt=50, l_n=10))
            for a in moteur.compute(cas).alerts}))
chk("un plat sur TOUTE la hauteur du poteau est permis (la contrainte « partie droite de l'âme "
    "porteuse » n'existe plus)", moteur.compute(dict(hp_u=300, z_C=45, n1_u=4)).geo_ok)
chk("hp = 140 : pince basse < 1,2·d0 → alerte bloquante e1bot",
    any(a.id == "e1bot" and a.block for a in moteur.compute(dict(hp_u=140)).alerts))
chk("tp = 15 > 0,5·d : alerte de ductilité (non bloquante)",
    any(a.id == "tp_duct" and not a.block for a in moteur.compute(dict(tp_u=15)).alerts))
chk("hp < 0,6·h : recommandation signalée", any(a.id == "hp_rec" for a in moteur.compute(dict(hp_u=170)).alerts))
chk("plat long et poutre non maintenue : alerte long_nr",
    any(a.id == "long_nr" for a in moteur.compute(dict(e2b_u=80, bp_u=160, lt_ok="Non")).alerts))
chk("grugeage à 0 par défaut (un poteau n'oblige pas à gruger) et bloc du grugeage inactif",
    E.defaults()["d_nt"] == 0 and E.defaults()["d_nb"] == 0 and E.defaults()["l_n"] == 0
    and R0.cas_g == 0 and not R0.ck["mN"].active and not R0.ck["stN"].active)
Rg = moteur.compute(dict(d_nt=50, d_nb=40, l_n=120, z_C=60, hp_u=240, n1_u=3, V_Ed=125))
chk("… mais tout le bloc de la poutre grugée s'active à l'identique quand on grugue",
    Rg.cas_g == 2 and Rg.ck["mN"].active and Rg.ck["stN"].active and Rg.ck["stD"].active)

# ================================================================
print("\n=== 6. Les briques de mecanique.py sont APPELÉES, pas recopiées ===")
sg = MEC.section_grugee(dict(h=Rg.h_S, b=Rg.b_S, tw=Rg.tw_S, tf=Rg.tf_S, r=Rg.r_S),
                        Rg.fy_S, 50, 40, 1.0, 125.0, True)
chk("poutre grugée : cas_g, Av, W_N, Mv,N,Rd, ln,max identiques à mecanique.section_grugee",
    sg["cas_g"] == Rg.cas_g and sg["Av"] == Rg.Av_S and sg["W_N"] == Rg.W_N
    and sg["Mv_N"] == Rg.ck["mN"].Rd and sg["ln_max"] == Rg.ln_max)
ge = MEC.groupe_excentre(R0.n_1, R0.n_2, R0.p_1, R0.u.p2_u, R0.u.V_Ed, 0, R0.M_S)
chk("groupe excentré : Ip, Fz, Fx, F identiques à mecanique.groupe_excentre",
    ge["Ip"] == R0.Ip and ge["Fz"] == R0.Fz_b and ge["Fx"] == R0.Fx_b and ge["F"] == R0.F_b)
be = MEC.bearing(R0.t_p, R0.fu_L, R0.f_ub, R0.d_b, R0.d_0, min(R0.e_1, R0.e1bot), R0.p_1, R0.n_1,
                 R0.e2_p, R0.u.p2_u, R0.n_2, 1.25, 1.0, Fz=R0.Fz_b, Fx=R0.Fx_b)
chk("pression diamétrale : Fbv, Fbh et l'interaction identiques à mecanique.bearing",
    be["Fbv"] == R0.Fbv_L and be["Fbh"] == R0.Fbh_L and be["ipd"] == R0.ipd_L)
chk("déversement : λLT et pb identiques à mecanique.lam_LT / pb_bs5950",
    MEC.lam_LT(R3.z_p, R3.h_p, R3.t_p) == R3.lam_LT
    and MEC.pb_bs5950(R3.lam_LT, R3.fy_L) == R3.pb_LT)
chk("gorge de pleine résistance : identique à mecanique.a_pleine_resistance_plat",
    MEC.a_pleine_resistance_plat(R0.t_p, R0.fy_L, R0.bww, R0.fuw, 1.25, 1.0) == R0.a_plein)
chk("le plateau de la courbe : pb = py pour λ ≤ λL0", moteur.pb_bs5950(20, 355) == 355)

# ================================================================
print("\n=== 7. Prédimensionnement ===")
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT"))
chk("proposition complète au cas par défaut", Rp.pd.found and Rp.pd.msg == "Proposition complète", Rp.pd.msg)
chk("épaisseur proposée ≤ 0,5·d (ductilité)", Rp.pd.tp <= 0.5 * Rp.d_b + 1e-9, str(Rp.pd.tp))
u2 = moteur.apply_solution(dict(), Rp.pd.pick)
chk("apply_solution : mode VÉRIFICATION, boulon, rangées, hp, tp, bp, a recopiés",
    u2["mode_calc"] == "VÉRIFICATION" and u2["boulon_u"] == Rp.pd.boulon
    and u2["n1_u"] == Rp.pd.n and u2["hp_u"] == Rp.pd.hp and u2["tp_u"] == Rp.pd.tp
    and u2["a_w"] == Rp.pd.a)
Rv = moteur.compute(u2)
chk("la solution recopiée est vérifiée en détail", Rv.verified, Rv.gov.key if Rv.gov else "")
Rpa = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT", appui=E.APPUI_AME))
chk("le prédimensionnement ne dépend pas du cas d'appui (le poteau n'y entre que par le cordon)",
    Rpa.pd.hp == Rp.pd.hp and Rpa.pd.tp == Rp.pd.tp and Rpa.pd.n == Rp.pd.n)

# ================================================================
print("\n=== 8. Robustesse (300 tirages aléatoires, les deux cas d'appui) ===")
rng = random.Random(7)
nerr = 0
for i in range(300):
    try:
        moteur.compute(dict(
            appui=rng.choice([E.APPUI_AILE, E.APPUI_AME]),
            hp_u=rng.choice([0, 60, 140, 190, 290, 400]), tp_u=rng.choice([0, 6, 8, 10, 15, 20]),
            bp_u=rng.choice([0, 60, 100, 160, 245]), z_C=rng.choice([0, 20, 40, 90]),
            n1_u=rng.randint(1, 8), n2_u=rng.choice([1, 2]), p1_u=rng.choice([0, 40, 60, 90]),
            p2_u=rng.choice([0, 50, 70]), e1_u=rng.choice([0, 25, 40, 60]),
            e2b_u=rng.choice([0, 30, 40, 80]), g_h=rng.choice([0, 10, 25, 155]),
            rec_S=rng.choice([0, 100, 150]), j_lat=rng.choice([0, 5, 10]),
            d_nt=rng.choice([0, 30, 50]), d_nb=rng.choice([0, 40]), l_n=rng.choice([0, 100, 150]),
            V_Ed=rng.choice([0, 50, 200, 400, 900]), N_Ed=rng.choice([-60, 0, 80, 250]),
            nat_N=rng.choice([E.NAT_ELU, E.NAT_TYING]),
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
chk("compute({}) et compute(None) passent", moteur.compute({}).verified and moteur.compute(None).verified)
chk("un profil PERSO sans dimensions ne lève pas",
    moteur.compute(dict(prof_P=PERSO)).statut.startswith("ASSEMBLAGE"))
chk("charger_json ignore les clés inconnues et complète les absentes",
    E.charger_json({"V_Ed": 300, "inconnue": 1})["V_Ed"] == 300
    and len(E.charger_json({})) == 71 and E.charger_json({})["appui"] == E.APPUI_AILE)

# ================================================================
print("\n=== 9. Régression figée (tests/ref_fpc.json) ===")
with open("tests/ref_fpc.json", encoding="utf-8") as fh:
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
chk("les cas figés couvrent les deux appuis, les trois variantes du cas (b), le grugeage, "
    "le tying, les catégories B et C, le trou surdimensionné et un profil PERSO",
    len(REF) >= 12
    and {"defaut_aile", "ame_b0", "ame_b1", "ame_b1_redim", "ame_b2_recoupe", "grugeage_double",
         "N_elu", "N_tying", "cat_B", "cat_C", "trou_surdim", "WE_finplate"} <= set(REF))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

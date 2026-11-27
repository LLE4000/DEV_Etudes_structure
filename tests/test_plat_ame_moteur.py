# -*- coding: utf-8 -*-
"""Moteur de l'assemblage par plat d'âme soudé (fin plate).

Chaque garantie rougit si on la retire :
  1. EXEMPLE PUBLIÉ (MSB Part 5 §3.4, Worked Example: Fin Plate — IPE A 550
     S275, plat 360×160×10 S275, 2 files de 5 M20 8.8, z = 80 mm) : les
     valeurs du guide, EN DIRECT (1 plan de cisaillement, sans la
     transposition ×2 du benchmark du module doubles cornières).
  2. CAS PAR DÉFAUT : les grandeurs partagées avec le module doubles
     cornières (même z, même groupe, même âme) tombent sur les MÊMES
     nombres — dont le taux de pression diamétrale de l'âme 0,739914…
     ancré par le calcul manuel VAL-A du module 1.
  3. DÉVERSEMENT DU PLAT LONG : courbe BS 5950-1 Annexe B.2.1 (Table 17),
     E = 205 000 MPa — valeurs manuelles indépendantes.
  4. ALERTES ET DOMAINE : pince basse, plat hors âme droite, ductilité
     (SCI P358 §5.3 (3) : tp ou tw ≤ k·d, section 9), plat long non
     maintenu, hp ≥ 0,6·h.
  5. PRÉDIM : proposition complète, apply_solution recopie tout.
  6. ROBUSTESSE : 200 tirages aléatoires sans exception.

Lancement : python3 tests/test_plat_ame_moteur.py
"""
import math
import os
import random
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.bibliotheques import PERSO  # noqa: E402
from acier.assemblages.poutre_poutre.plat_ame import moteur  # noqa: E402

# Lot A (16/11/2026) : les exemples publiés et les calculs à la main ci-dessous gardent LEUR matériau (EN 1993-1-1
# tableau 3.1, sans épaisseur : S355 355/510, S275 275/430 — la table du corrigé de référence) ; ce sont des
# validations de FORMULES. Les notes prennent la table belge (tableau 3.3 ANB), contrôlée à la section « Lot A ».
sys.path.insert(0, os.path.join(RACINE, "tests"))
import lot_a_materiau as LA  # noqa: E402


def calcul_main(u=None):
    """Le moteur sur le matériau des calculs à la main de cette suite (validation de formules)."""
    return moteur.compute(u, LA.TABLEAU_3_1)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * abs(b)


# ================================================================
print("=== 1. Exemple publié : MSB Part 5 §3.4 (fin plate) ===")
WE = dict(prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275",
          prof_P=PERSO, hP_u=800, bP_u=300, twP_u=15, tfP_u=30, rP_u=27, nu_P="S275",
          nu_pl="S275", d_top=60, d_nt=0, d_nb=0, l_n=0, g_h=10, z_C=50,
          hp_u=360, tp_u=10, bp_u=160, boulon_u="M20", classe="8.8",
          n1_u=5, n2_u=2, p1_u=70, p2_u=60, e1_u=40, e2b_u=40, V_Ed=350, g_M1=1.0)
R = calcul_main(WE)


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
chk("flexion du plat sans objet (hp = 360 ≥ 2,73·z = 218,4)", not R.ck["pfl"].active and R.court)
# LE HARNAIS SUIT (lot de clôture du 27/11/2026) : il figeait « plat long » jugé sur z = 80 (le centre du groupe,
# règle du MSB) ; le bureau a retenu le SCI, qui le juge sur zp = gh + e2,b = 50 (la PREMIÈRE file) : ce plat est COURT
chk("plat COURT pour le SCI (zp = 50 ≤ tp/0,15 = 66,7), long pour le MSB (z = 80) : ni déversement ni section ABCD "
    "dans le verdict",
    not R.long_p and R.long_msb and R.g_B == 50 and not R.ck["plt"].active and not R.ck["abS"].active)
chk("… et la section ABCD du MSB reste calculée pour INFORMATION : 38,5 kNm demandés, 65,87 résistants (le guide "
    "imprime 38 et 66, p. 5-46) — 58 %, aucun motif « à vérifier »",
    pres(R.abcd_msb.MEd, 38.5, 1e-12) and pres(R.abcd_msb.MRd, 66, 0.01) and pres(R.eta_abcd_msb, 38.5 / 65.873, 1e-4)
    and R.a_verifier == "", f"{R.abcd_msb.MEd} {R.abcd_msb.MRd} {R.a_verifier}")
chk("VÉRIFIÉ à VEd = 350 kN ; dimensionnante = cisaillement brut du plat (450 kN, "
    "la plus petite résistance du guide)", R.verified and R.gov.key == "pcg", R.gov.key)

print("\n=== 2. Géométrie partagée avec le module 1 (e1 = 35, p1 = 60 : l'ancien cas par défaut) ===")
# LE HARNAIS SUIT (27/11/2026) : le cas par défaut tient désormais les pinces à 2·d du détail éprouvé (e1 = 40,
# p1 = 55 — SCI P358 §5.3 (5)) ; les ancres partagées avec les doubles cornières gardent LEUR géométrie, saisie ici
Rdef = calcul_main(dict())
chk("cas par défaut : e1 = 40 = 2·d, p1 = 55, hp = 190 (40 + 2 × 55 + 40) — toutes les pinces ≥ 2d, VÉRIFIÉ sans motif "
    "« à vérifier » ; l'ancienne géométrie (e1 = 35) sort « À VÉRIFIER (articulation à justifier : e1 du plat …) »",
    Rdef.e_1 == 40 and Rdef.p_1 == 55 and Rdef.e1bot == 40 and Rdef.rec_2d and Rdef.verified and Rdef.a_verifier == ""
    and Rdef.statut == "ASSEMBLAGE VÉRIFIÉ" and not any(a.id == "pince_2d" for a in Rdef.alerts),
    f"{Rdef.e_1} {Rdef.p_1} {Rdef.statut}")
R0 = calcul_main(dict(e1_u=35, p1_u=60))
chk("e1 = 35 < 2d = 40 : VÉRIFIÉ par le calcul, mais statut « À VÉRIFIER (articulation à justifier : e1 du plat = "
    "35 mm < 2d = 40 mm) » et avis non bloquant",
    R0.verified and not R0.rec_2d and R0.pince_2d == ("e1 du plat", 35.0)
    and R0.statut == "ASSEMBLAGE À VÉRIFIER (articulation à justifier : e1 du plat = 35 mm < 2d = 40 mm)"
    and any(a.id == "pince_2d" and not a.block for a in R0.alerts), R0.statut)
chk("e1 = 35, p1 = 60 : z = 50, Ip = 7200, effort de boulon 66,699… kN (mêmes nombres que doubles cornières)",
    R0.z_p == 50 and R0.Ip == 7200 and pres(R0.F_b, 66.6992108065922, 1e-9), str(R0.F_b))
chk("taux de pression diamétrale de l'âme = 0,739914371796275 (ancre VAL-A du module 1)",
    abs(R0.ck["pdS"].eta - 0.739914371796275) < 1e-12, str(R0.ck["pdS"].eta))
chk("grugeage : A_Tee 6121, Av 2358,5, Wel,N 139377,43, Mv,N,Rd 49,479 kNm, ln,max 290 (VAL-A)",
    R0.A_Tee == 6121 and R0.Av_S == 2358.5 and pres(R0.W_N, 139377.425915224, 1e-9)
    and pres(R0.ck["mN"].Rd, 49.4789861999044, 1e-9) and R0.ln_max == 290)
chk("plat court par défaut (z = 50 ≤ tp/0,15 = 66,7) : ni flexion ni déversement",
    not R0.ck["pfl"].active and not R0.ck["plt"].active)
# LE HARNAIS SUIT (27/11/2026) : la longueur EFFICACE d'un cordon est hp − 2a = 178 mm (EN 1993-1-8 §4.5.1(1)),
# plus hp = 190 — qz 328,9 → 351,1 N/mm (+6,7 %), qx 519,4 → 591,8 N/mm (+13,9 % : le bras de flexion est au carré)
chk("soudure : ℓ = hp − 2a = 178 mm ; qz = VEd/(2ℓ) = 351,1 N/mm ; qx = 3·MS/ℓ² = 591,8 N/mm ; Fw,Rd = fvw·a",
    R0.Lw1 == 178 and pres(R0.qz_w, 125000 / 356, 1e-9) and pres(R0.qx_w, 3 * 6.25e6 / 178 ** 2, 1e-9)
    and pres(R0.ck["w"].Rd, R0.fvw * 6, 1e-9))      # a = 6 mm par défaut depuis le 25/11/2026 (pleine résistance)
# --- corrections du 25/11/2026 (audit des formules)
# gorge de PLEINE RÉSISTANCE : le plat tendu plastifie avant le cordon — tp·fy·βw·γM2/(√2·fu·γM0), la règle des guides
# (0,55·tp en S355) ; le diviseur était 2 (plat cisaillé : 3,915 mm pour tp = 10, une gorge de 5 passait sans avis)
chk("gorge de pleine résistance : 10·355·0,9·1,25/(√2·510) = 5,537 mm ; a = 6 par défaut : pas d'avis ; a = 5 : avis "
    "non bloquant",
    pres(R0.a_plein, 10 * 355 * 0.9 * 1.25 / (math.sqrt(2) * 510), 1e-12) and pres(R0.a_plein, 5.5373, 1e-4)
    and R0.a_w == 6 and not any(a.id == "a_rec" for a in R0.alerts)
    and any(a.id == "a_rec" and not a.block for a in calcul_main(dict(a_w=5)).alerts), str(R0.a_plein))
from acier.assemblages import mecanique as _MEC  # noqa: E402
from acier.assemblages.poutre_poutre.plat_ame.entrees import MODE_PREDIM as _MP  # noqa: E402
_Rp = calcul_main(dict(mode_calc=_MP))
chk("prédimensionnement : chaque plat proposé reçoit sa gorge de pleine résistance (0,554·tp arrondi au mm supérieur)",
    all(r.prop.a == max(3, math.ceil(r.prop.tp * 355 * 0.9 * 1.25 / (math.sqrt(2) * 510) - _MEC.TOL_GORGE))
        for r in _Rp.pd.rows if r.geom) and _Rp.pd.a == max(3, math.ceil(_Rp.pd.tp * 0.55373 - 0.05)),
    f"{_Rp.pd.tp} {_Rp.pd.a}")
# flexion du plat : « sans objet si hp ≥ 2,73·z » ne couvre que M = VEd·z — avec un MEd, la vérification reste due
_Rm = calcul_main(dict(M_Ed=4))
chk("plat court (hp = 190 ≥ 2,73·z = 136,5) : flexion sans objet si MEd = 0 ; ACTIVE avec MEd = 4 kNm — "
    "Ed = 125·0,05 + 4 = 10,25 kNm, Rd = Wel·fy = 21,36 kNm",
    R0.court and not R0.ck["pfl"].active and _Rm.court and _Rm.ck["pfl"].active
    and pres(_Rm.ck["pfl"].Ed, 10.25, 1e-12) and pres(_Rm.ck["pfl"].Rd, 10 * 190 ** 2 / 6 * 355 / 1e6, 1e-12),
    f"{_Rm.ck['pfl'].active} {_Rm.ck['pfl'].Ed} {_Rm.ck['pfl'].Rd}")
# interaction à la 2e file : MEd s'ajoute, comme à l'extrémité du grugeage (mN)
_R2 = calcul_main(dict(n2_u=2, bp_u=160, d_nb=40, M_Ed=4))
_R2s = calcul_main(dict(n2_u=2, bp_u=160, d_nb=40))
chk("2e file de boulons (n2 = 2), MEd = 4 kNm : Ed = 125·(10 + 40 + 60)/1000 + 4 = 17,75 kNm (13,75 sans MEd)",
    _R2.ck["m2"].active and pres(_R2.ck["m2"].Ed, 17.75, 1e-12) and pres(_R2s.ck["m2"].Ed, 13.75, 1e-12)
    and pres(_R2.ck["m2"].Rd, _R2s.ck["m2"].Rd, 1e-12), f"{_R2.ck['m2'].Ed} {_R2s.ck['m2'].Ed}")
chk("cisaillement local âme porteuse : Av = tw·hp = 11 × 190 = 2090 mm², Ed = VEd/2",
    R0.Av_P == 2090 and R0.ck["vlP"].Ed == 62.5)
chk("cas par défaut VÉRIFIÉ, dimensionnante = pression diamétrale de l'âme",
    R0.verified and R0.gov.key == "pdS")
chk("γM1 par défaut = 1,10 (ANB belge)", R0.u.g_M1 == 1.1)

print("\n=== 3. Plat LONG selon le SCI P358 (Check 3 et Check 4) — règle retenue le 27/11/2026 ===")
# LE HARNAIS SUIT : il figeait la courbe BS 5950-1 des profilés laminés (pb, E = 205 000), sans le 0,6, avec le bras z.
# L'ancienne fonction reste (test ancien / nouveau, tests/test_assemblages_cloture.py) ; le calcul ne l'appelle plus.
from acier.assemblages import sci as SCI  # noqa: E402
S3 = math.sqrt(3)
chk("ancienne courbe gardée comme fonction : pb = py pour λ ≤ λL0 (plateau)", moteur.pb_bs5950(20, 275) == 275)
chk("χLT de la Table 5.2 du guide (expression 6.56, αLT = 0,76) : 0,25 → 0,96 ; 0,50 → 0,78 ; 0,75 → 0,61 ; "
    "1,00 → 0,47 ; 1,50 → 0,28 ; 2,00 → 0,18 ; 2,70 → 0,11, à l'arrondi de la table",
    all(abs(SCI.chi_LT(l) - x) <= 0.005 + 1e-12 for l, x in ((0.25, 0.96), (0.50, 0.78), (0.75, 0.61), (1.00, 0.47),
                                                               (1.50, 0.28), (2.00, 0.18), (2.70, 0.11))),
    str([round(SCI.chi_LT(l), 4) for l in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.7)]))
# un plat réellement long : l'exemple avec gh = 40 → zp = 80 > 66,7 ; z = 110 ; hp = 360 < 2,73·z = 300,3 ? non : court
RL = calcul_main(dict(WE, g_h=40, bp_u=190))
lam = 2.8 * math.sqrt(80 * 360 / (1.5 * 100))
lamb = lam / (math.pi * math.sqrt(210000 / 275))
phi = 0.5 * (1 + 0.76 * (lamb - 0.2) + lamb * lamb)
chi = 1 / (phi + math.sqrt(phi * phi - lamb * lamb))
chk("plat long (gh = 40 : zp = 80 > tp/0,15 = 66,7) : λ = 2,8·√(zp·hp/(1,5·tp²)) = 38,80 — avec zp, la première "
    "file, et non z = 110 ; λ̄LT = λ/(π·√(E/fy)) = 0,447 ; χLT = 0,817",
    RL.long_p and RL.g_B == 80 and RL.z_p == 110 and pres(RL.lam_LT, lam, 1e-12) and pres(RL.lamb_LT, lamb, 1e-12)
    and pres(RL.chi_LT, chi, 1e-12) and pres(chi, 0.8168, 1e-4), f"{RL.lam_LT} {RL.lamb_LT} {RL.chi_LT}")
chk("résistance : min(Wel·χLT·fy/(0,6·γM1) = 80,87 ; Wel·fy/γM0 = 59,40) = 59,40 kNm — le 0,6 du diagramme "
    "triangulaire ; demande VEd·z = 350 × 0,110 = 38,5 kNm",
    RL.ck["plt"].active and pres(RL.Mb_LT, 216000 * chi * 275 / 0.6 / 1e6, 1e-12)
    and pres(RL.ck["plt"].Rd, 59.4, 1e-12) and pres(RL.ck["plt"].Ed, 38.5, 1e-12), f"{RL.Mb_LT} {RL.ck['plt'].Rd}")
# le déversement GOUVERNE sur un plat mince et très long : tp = 8, gh = 100 (zp = 140), γM1 = 1,10
RLg = calcul_main(dict(WE, tp_u=8, g_h=100, bp_u=250, g_M1=1.1))
_lb = 2.8 * math.sqrt(140 * 360 / (1.5 * 64)) / (math.pi * math.sqrt(210000 / 275))
_ph = 0.5 * (1 + 0.76 * (_lb - 0.2) + _lb * _lb)
_ch = 1 / (_ph + math.sqrt(_ph * _ph - _lb * _lb))
chk("plat de 8 mm, zp = 140, γM1 = 1,10 : le déversement gouverne — Wel·χLT·fy/(0,6·γM1) = 172 800 × 0,618 × 275/0,66 "
    "= 44,5 kNm < Wel·fy = 47,52 kNm ; avec γM1 = 1,0 le plafond élastique reprend la main",
    pres(RLg.chi_LT, _ch, 1e-12) and pres(RLg.ck["plt"].Rd, 172800 * _ch * 275 / (0.6 * 1.1) / 1e6, 1e-12)
    and RLg.ck["plt"].Rd < 172800 * 275 / 1e6
    and pres(calcul_main(dict(WE, tp_u=8, g_h=100, bp_u=250, g_M1=1.0)).ck["plt"].Rd, 172800 * 275 / 1e6, 1e-12),
    f"{RLg.chi_LT} {RLg.ck['plt'].Rd}")
chk("MEd imposé : le diagramme n'est plus triangulaire, le 0,6 n'est PAS appliqué (coefficient 1)",
    calcul_main(dict(WE, g_h=40, bp_u=190, M_Ed=5)).k_tri == 1.0 and RL.k_tri == 0.6)
# SECTION ABCD de l'âme portée (Check 4) : VEd·(zp + p2) ≤ Mc,BC,Rd + Vpl,AB,Rd·(n1 − 1)·p1
_VAB = 9 * 100 * 275 / S3 / 1000
_Mc = 275 * 9 * 280 ** 2 / 6 / 1e6
chk("section ABCD (plat long) : 350 × (80 + 60) = 49,0 kNm ≤ Mc,BC,Rd 32,34 + Vpl,AB,Rd 142,89 × 0,280 = 72,35 kNm ; "
    "VBC,Ed = 350 × 280/547 = 179,2 ≤ 0,5·Vpl,BC,Rd = 200,1 : pas de réduction",
    RL.ck["abS"].active and pres(RL.ck["abS"].Ed, 49.0, 1e-12) and pres(RL.abcd.V_AB, _VAB, 1e-12)
    and pres(RL.abcd.Mc_BC, _Mc, 1e-12) and pres(RL.ck["abS"].Rd, _Mc + _VAB * 0.28, 1e-12) and not RL.abcd.fort
    and pres(RL.abcd.VBC, 350 * 280 / 547, 1e-12), f"{RL.ck['abS'].Ed} {RL.ck['abS'].Rd}")
RLf = calcul_main(dict(WE, g_h=40, bp_u=190, V_Ed=420))
chk("… cisaillement FORT de BC (VEd = 420 : VBC,Ed = 215,0 > 200,1) : Mc,BC,Rd réduit par 1 − (2·VBC/Vpl,BC − 1)²",
    RLf.abcd.fort and pres(RLf.abcd.Mc_BC, _Mc * (1 - (2 * (420 * 280 / 547) / (9 * 280 * 275 / S3 / 1000) - 1) ** 2), 1e-12),
    str(RLf.abcd.Mc_BC))
chk("plat long sur une poutre NON maintenue : statut « à vérifier » (le guide ne l'admet pas sans essais)",
    "plat long et poutre non maintenue" in calcul_main(dict(WE, g_h=40, bp_u=190, lt_ok="Non")).a_verifier)

print("\n=== 4. Alertes et domaine ===")
Ra = moteur.compute(dict(hp_u=300))
chk("hp = 300 avec 3 rangées : pince basse OK mais plat hors âme droite → alerte pl_web",
    any(a.id == "pl_web" for a in Ra.alerts))
Rb = moteur.compute(dict(hp_u=140))
chk("hp = 140 : pince basse < 1,2·d0 → alerte bloquante e1bot",
    any(a.id == "e1bot" and a.block for a in Rb.alerts) and not Rb.verified)
Rc = moteur.compute(dict(tp_u=15))
chk("tp = 15 > 0,42·d = 8,4 et tw = 8,5 > 8,4 (S355) : alerte ductilité (non bloquante)",
    any(a.id == "tp_duct" and not a.block for a in Rc.alerts))
Rd_ = moteur.compute(dict(e2b_u=70, lt_ok="Non"))
chk("zp = gh + e2,b = 80 > tp/0,15 et poutre non maintenue : alerte plat long",
    Rd_.long_p and any(a.id == "long_nr" for a in Rd_.alerts))
chk("hp < 0,6·h : recommandation signalée", any(a.id == "hp_rec" for a in moteur.compute(dict(hp_u=170)).alerts))
chk("Tableau 3.3 : 6 lignes actives par défaut (e1p, e2p, e1b, e2b, p1 ; p2 inactif)",
    len(R0.dist) == 5 and all(x.ok for x in R0.dist), str([x.id for x in R0.dist]))

print("\n=== 5. Prédimensionnement ===")
# LE HARNAIS SUIT (27/11/2026) : les propositions tiennent les pinces à 2·d du détail éprouvé (SCI P358 §5.3 (5)).
# Sur les 199 mm disponibles de la HEA 300 grugée, 3 M20 à e1 = 40 et p1 = 55 sont à 92,9 % sous 125 kN — au-dessus
# du taux cible de 90 % : plus de proposition « complète » à cette charge (elle existait à 80,3 % avec e1 = 35) ;
# à 110 kN (81,8 %) la proposition complète revient.
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT", V_Ed=110))
chk("proposition complète à 110 kN, pinces ≥ 2·d ; à 125 kN (cas par défaut) aucune combinaison sous le taux cible",
    Rp.pd.found and Rp.pd.msg == "Proposition complète" and Rp.pd.e1 >= 2 * Rp.pd.pick.b["d"]
    and Rp.pd.e2 >= 2 * Rp.pd.pick.b["d"] and not moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT")).pd.found,
    Rp.pd.msg)
chk("prédimensionnement : AUCUNE ligne proposée sous 2·d (e1 et e2, tous diamètres)",
    all(r.e1 >= 2 * r.b["d"] and r.e2 >= 2 * r.b["d"] for r in Rp.pd.rows))
chk("épaisseur proposée ≤ plafond de ductilité (0,42·d en S355, SCI P358 §5.3 (3))",
    Rp.pd.tp <= Rp.pd.pick.prop.tmax + 1e-9, str(Rp.pd.tp))
u2 = moteur.apply_solution(dict(V_Ed=110), Rp.pd.pick)
chk("apply_solution : mode VÉRIFICATION, boulon, rangées, hp, tp, bp, a recopiés",
    u2["mode_calc"] == "VÉRIFICATION" and u2["boulon_u"] == Rp.pd.boulon
    and u2["n1_u"] == Rp.pd.n and u2["hp_u"] == Rp.pd.hp and u2["tp_u"] == Rp.pd.tp
    and u2["a_w"] == Rp.pd.a)
Rv = moteur.compute(u2)
chk("la solution recopiée est vérifiée en détail", Rv.verified, Rv.gov.key if Rv.gov else "")

print("\n=== 6. Robustesse (200 tirages aléatoires) ===")
rng = random.Random(7)
nerr = 0
for i in range(200):
    try:
        moteur.compute(dict(
            hp_u=rng.choice([0, 60, 140, 190, 260, 400]), tp_u=rng.choice([0, 6, 8, 10, 15, 20]),
            bp_u=rng.choice([0, 60, 100, 160, 220]), z_C=rng.choice([0, 20, 50, 90]),
            n1_u=rng.randint(1, 8), n2_u=rng.choice([1, 2]), p1_u=rng.choice([0, 40, 60, 90]),
            p2_u=rng.choice([0, 50, 70]), e1_u=rng.choice([0, 25, 35, 60]),
            e2b_u=rng.choice([0, 30, 40, 80]), g_h=rng.choice([0, 10, 25]),
            d_nt=rng.choice([0, 30, 50]), d_nb=rng.choice([0, 40]), l_n=rng.choice([0, 100, 150]),
            V_Ed=rng.choice([0, 50, 125, 400, 900]), N_Ed=rng.choice([-60, 0, 80]),
            M_Ed=rng.choice([0, 5]), boulon_u=rng.choice(["M12", "M16", "M20", "M24"]),
            classe=rng.choice(["4.6", "8.8", "10.9"]), cat=rng.choice(["A", "B", "C"]),
            trou=rng.choice(["Normal", "Surdimensionné"]),
            prof_S=rng.choice(["IPE 200", "HEA 300", "HEB 500"]),
            prof_P=rng.choice(["HEA 400", "HEB 600"])))
    except Exception as e:  # noqa: BLE001
        nerr += 1
        if nerr == 1:
            print("      première exception :", type(e).__name__, e, "au tirage", i)
chk("aucune exception sur 200 tirages", nerr == 0, str(nerr))

print("\n=== 7. Régression figée (tests/ref_plat_ame.json, 13 cas, 215 vérifications) ===")
import json  # noqa: E402
with open("tests/ref_plat_ame.json", encoding="utf-8") as fh:
    REF = json.load(fh)
ndiff = 0
for nom, cas in REF.items():
    Rr = moteur.compute(cas["inp"])
    if {c.key for c in Rr.checks if c.active} != set(cas["checks"]) or Rr.a_verifier != cas["a_verifier"]:
        ndiff += 1      # (une vérification ACTIVE de plus ou de moins, ou un motif « à vérifier » différent)
        print("   ", nom, "vérifications actives ou motif divergents", Rr.a_verifier)
    for k, (ed, rd, eta) in cas["checks"].items():
        c = Rr.ck.get(k)
        if c is None or not c.active or abs(c.Ed - ed) > 1e-9 * max(1, abs(ed)) \
           or abs(c.Rd - rd) > 1e-9 * max(1, abs(rd)) or abs(c.eta - eta) > 1e-12 * max(1, abs(eta)):
            ndiff += 1
            print("   ", nom, k, "attendu", (ed, rd, eta), "obtenu",
                  (c.Ed, c.Rd, c.eta) if c and c.active else "inactif")
    for k, v in cas["scal"].items():
        if abs(Rr[k] - v) > 1e-9 * max(1, abs(v)):
            ndiff += 1
            print("   ", nom, "scalaire", k, "attendu", v, "obtenu", Rr[k])
    if (Rr.gov.key if Rr.gov else None) != cas["gov"] or Rr.verified != cas["verified"] \
       or sorted(a.id for a in Rr.alerts) != cas["alerts"]:
        ndiff += 1
        print("   ", nom, "synthèse divergente")
chk("RÉGRESSION CALCUL : 0 différence sur les 13 cas figés", ndiff == 0, str(ndiff))

# ================================================================
print("\n=== 8. Lot A : f_y et f_u de chaque pièce selon SON épaisseur (tableau 3.3 ANB) ===")
# NBN EN 1993-1-1 ANB:2018 3.2.1(1) (option a) ; profilé : max(tf, tw) ; plat : son épaisseur. La note neuve :
# poutre principale HEA 400 (tf = 19 mm) → f_y = 345 ; f_u = 470 partout (3 à 100 mm).
LA.jouer(chk, moteur.compute, attendus=[
    ("note neuve", {}, {"P": (19, LA.FY_S355_40, LA.FU_S355), "L": (10, LA.FY_S355, LA.FU_S355)}),
    ("IPE 600 portée par un HEB 600 (tf = 19 et 30 mm)", dict(prof_S="IPE 600", prof_P="HEB 600", V_Ed=250),
     {"S": (19, LA.FY_S355_40, LA.FU_S355), "P": (30, LA.FY_S355_40, LA.FU_S355)}),
    ("plat de 20 mm", dict(tp_u=20), {"L": (20, LA.FY_S355_40, LA.FU_S355)}),
    ("plat de 70 mm S355", dict(tp_u=70), {"L": (70, LA.FY_S355_80, LA.FU_S355)}),
    ("S460 N/NL (tableau 3.4 ANB)", dict(nu_S="S460 N/NL", prof_S="IPE 600"), {"S": (19, LA.FY_S460_40, LA.FU_S460)}),
], ligne_fu=("note neuve", {}, "pdS"),
   grugeage=(dict(prof_S="IPE 600", prof_P="HEB 600", nu_S="S275", d_nt=40, l_n=120, V_Ed=250), "S"),
   perso=(dict(nu_pl=PERSO, fy_u=300, fu_u=420, bw_u=0.85, tp_u=20), "L", 300, 420, "plat de 20 mm, f_y et f_u saisis tels quels"),
   hors_table=(dict(tp_u=300), "plat"))

# ================================================================
print("\n=== 8 bis. Lot B : un refus de la table s'écrit à la virgule (brief v12 § 19.2) ===")
from acier import materiaux as MA_  # noqa: E402
from acier.assemblages import materiau as MT_  # noqa: E402


def refus(u):
    try:
        moteur.compute(u)
    except MA_.HorsTable as e:
        return str(e)
    return None


def ctl_refus_virgule():
    return refus(dict(tp_u=260.5)) == "plat : S355 laminé de 260,5 mm, au-delà du tableau 3.3 ANB (250 mm)"


chk("plat de 260,5 mm : « plat : S355 laminé de 260,5 mm, au-delà du tableau 3.3 ANB (250 mm) » (et non 260.5)",
    ctl_refus_virgule(), refus(dict(tp_u=260.5)))
_avant = MT_.mm_fr
MT_.mm_fr = str
try:
    chk("mutation « refus écrit avec le point (260.5 mm) » : le contrôle rougit", not ctl_refus_virgule())
finally:
    MT_.mm_fr = _avant

# ================================================================
print("\n=== 9. Lot B : ductilité de l'attache, SCI P358 §5.3 (3) (brief v12 § 19.1 point 3) ===")
# tp ≤ k·d (nuance du plat) OU tw ≤ k·d (nuance de la poutre portée) ; k = 0,50 en S235 et S275, 0,42 en S355 ; au-delà :
# non couvert. Recommandation, jamais un verdict ; le prédimensionnement suit la même règle. IPE 300 : t_w = 7,1 mm.
import lot_b_ductilite as LB  # noqa: E402
from acier.assemblages.poutre_poutre.plat_ame import note as NOTE_LB  # noqa: E402  (la note PDF : l'avis long y reste entier)
LB.jouer(chk, moteur.compute, "IPE 300", NOTE_LB.generer_note)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

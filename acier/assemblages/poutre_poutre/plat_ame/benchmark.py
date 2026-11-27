# -*- coding: utf-8 -*-
"""Benchmark du module plat d'âme : les exemples publiés des deux guides
rejoués EN DIRECT — MSB Part 5 §3.4 (Worked Example: Fin Plate, un plan de
cisaillement, sans la transposition ×2 du benchmark du module doubles
cornières) et, depuis le lot de clôture du 27/11/2026, l'exemple 1 du SCI
P358 (deux poutres grugées, une et deux files : la procédure RETENUE par le
bureau, grugeage court compris), la table du déversement du plat long
(Table 5.2 du SCI) — et des calculs manuels indépendants pour ce que les
exemples ne couvrent pas (plat réellement long, soudure).

Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Un écart
qui porte une explication (``exp``) ne compte pas comme inattendu. Aucune
formule n'a été ajustée pour coller à la source.
"""
import math

from acier.bibliotheques import PERSO
from acier.js import dv, mx
from .moteur import compute
from acier.assemblages import materiau as MT    # le matériau des exemples (validation de formules)
from acier.assemblages import sci as SCI

S3 = math.sqrt(3)

E_ARR = "Arrondis intermédiaires de la source (αb = 0,61 pour 0,606 ; Fv,Rd = 94 pour 94,08)."
E_AIRE = ("La source utilise l'aire catalogue A = 11700 mm² ; l'outil calcule A = 2b·tf + (h − 2tf)·tw "
          "+ (4 − π)·r² = 11729 mm² (+0,25 %), d'où Av = 6030 au lieu de 6001 mm².")

WE = dict(prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275",
          prof_P=PERSO, hP_u=800, bP_u=300, twP_u=15, tfP_u=30, rP_u=27, nu_P="S275",
          nu_pl="S275", d_top=60, d_nt=0, d_nb=0, l_n=0, g_h=10, z_C=50,
          hp_u=360, tp_u=10, bp_u=160, boulon_u="M20", classe="8.8",
          n1_u=5, n2_u=2, p1_u=70, p2_u=60, e1_u=40, e2b_u=40, V_Ed=350, g_M1=1.0)


# Exemple 1 du SCI P358 (plats d'âme, p. 137-153) : deux poutres S275 grugées (50 × 110) sur l'âme d'une
# 610×229×140 UKB, plats de 10 mm S275, M20 8.8, p1 = 70, e1 = 40, e2,b = 40, gh = 10, soudures de 8 mm (a = 5,6).
# Le matériau du guide EST la table du site ; γM2 des sections nettes (1,1) et γM1 (1,0) sont ceux de l'annexe
# britannique, repris pour comparer les valeurs imprimées.
_SCI_P = dict(prof_P=PERSO, hP_u=617.2, bP_u=230.2, twP_u=13.1, tfP_u=22.1, rP_u=12.7, nu_P="S275")
_SCI_C = dict(nu_pl="S275", d_top=0, d_nt=50, d_nb=0, l_n=110, g_h=10, z_C=50, tp_u=10, boulon_u="M20", classe="8.8",
              p1_u=70, p2_u=60, e1_u=40, e2b_u=40, g_M1=1.0, g_M2n=1.1, a_w=5.6)
SCI_1A = dict(_SCI_P, prof_S=PERSO, hS_u=412.8, bS_u=179.5, twS_u=9.5, tfS_u=16.0, rS_u=10.2, nu_S="S275",
              n1_u=4, n2_u=1, hp_u=290, bp_u=100, V_Ed=200, **_SCI_C)
SCI_1B = dict(_SCI_P, prof_S=PERSO, hS_u=533.1, bS_u=209.3, twS_u=10.1, tfS_u=15.6, rS_u=12.7, nu_S="S275",
              n1_u=5, n2_u=2, hp_u=360, bp_u=160, V_Ed=350, **_SCI_C)
# un plat réellement LONG : l'exemple du MSB avec gh = 40 (zp = 80 > tp/0,15 = 66,7 ; z = 110)
LONG = dict(WE, g_h=40, bp_u=190)
E_SCI = "Arrondis intermédiaires de la source (αb, k1, résultante par boulon)."
E_385 = "Arrondi de la source : 350 × 0,110 = 38,5 kNm, imprimé 38."


def vr(k):
    """Résistance du groupe exprimée en VEd : VEd / η de la vérification k."""
    return lambda R: dv(R.u.V_Ed, R.ck[k].eta)


def rd(k):
    return lambda R: R.ck[k].Rd


def va(k):
    return lambda R: R[k]


SRC = dict(
    titre="Steel Buildings in Europe – Multi-Storey Steel Buildings – Part 5: Joint Design "
          "— §3.4 Worked Example: Fin Plate",
    org="ArcelorMittal / Peiner Träger / Corus – contenu technique CTICM et SCI (projet RFCS SECHALO, RFS2-CT-2008-0030)",
    annee="feuilles de calcul datées 06/2009",
    norme="EN 1993-1-8:2005 et EN 1993-1-1:2005, valeurs recommandées (le module applique par défaut "
          "l'ANB belge : γM1 = 1,10 — l'exemple est rejoué avec γM1 = 1,0 comme la source)",
    url="https://steelconstruction.info/images/5/53/SBE_MS5.pdf")

SRC_SCI = dict(
    titre="SCI P358 — Joints in Steel Construction: Simple Joints to Eurocode 3 — Fin plates, Worked example 1 "
          "(beam to beam) et Table 5.2",
    org="The Steel Construction Institute / The British Constructional Steelwork Association (édition de 2014)",
    annee="pages 137 à 153 : tableau de synthèse des Checks 2 à 6 (p. 139) ; Table 5.2 p. 120",
    norme="BS EN 1993-1-8 et BS EN 1993-1-1 avec l'annexe britannique (γM2 = 1,25 boulons, 1,1 sections nettes ; "
          "γM1 = 1,0)",
    url="https://www.steelconstruction.info/images/3/3d/SCI_P358.pdf")

BENCH = [
    dict(id="BM1", titre="§3.4 Worked Example: Fin Plate — VEd = 350 kN, 2 files de boulons, z = 80 mm, "
                         "p. 5-38 à 5-45",     # le MÊME exemple sert, transposé, de BM4 au module doubles cornières
         data="IPE A 550 S275 (h 547, b 210, tw 9, tf 15,7, r 24), plat 360×160×10 S275, M20 8.8, "
              "n1 = 5, n2 = 2, p1 = 70, p2 = 60, e1 = 40, e2 = 50, e2,b = 40, e1,b = 90, poutre non grugée",
         inp=WE, rows=[
             ["p. 5-40", "Inertie polaire du groupe de boulons I", 107000, va("Ip"), "mm²", ""],
             ["p. 5-40", "Groupe excentré – cisaillement des boulons VRd", 584, vr("bv"), "kN", E_ARR],
             ["p. 5-41", "Pression diamétrale sur le plat VRd", 605, vr("pdL"), "kN", E_ARR],
             ["p. 5-42", "Pression diamétrale sur l'âme VRd", 624, vr("pdS"), "kN", ""],
             ["p. 5-43", "Plat – cisaillement section brute VRd,g", 450, rd("pcg"), "kN", ""],
             ["p. 5-43", "Plat – cisaillement section nette VRd,n", 497, rd("pcn"), "kN", ""],
             ["p. 5-43", "Plat – rupture de bloc VRd,b", 483, rd("pcb"), "kN", ""],
             ["p. 5-45", "Âme – cisaillement section brute VRd,g", 953, rd("vgS"), "kN", E_AIRE],
             ["p. 5-45", "Âme – cisaillement section nette VRd,n", 995, rd("vnS"), "kN", E_AIRE],
             ["p. 5-45", "Âme – rupture de bloc VRd,b (non grugée : (n1 − 1)·d0)", 507, rd("vbS"), "kN", ""],
             ["p. 5-46", "Section ABCD de l'âme (plat long pour le MSB, z = 80 > 66,7) – moment demandé VEd·(zp + p2)",
              38, lambda R: R.abcd_msb.MEd, "kNm", E_385],
             ["p. 5-46", "Section ABCD de l'âme – moment résistant Mc,BC,Rd + Fpl,AB,Rd·(n1 − 1)·p1 (information : le "
                         "plat est COURT pour le SCI, zp = 50 ≤ 66,7)", 66, lambda R: R.abcd_msb.MRd, "kNm", ""]]),
    dict(id="BM2", titre="SCI P358, plats d'âme, exemple 1 — poutre 406×178×74 UKB grugée, UNE file, "
                         "VEd = 200 kN, plat 290 × 100 × 10",
         src=SRC_SCI, mat=None, source=MT.SOURCE_TABLE,
         data="406×178×74 UKB S275 (tw 9,5) grugée 50 × 110 sur l'âme d'une 610×229×140 UKB (tw 13,1), plat "
              "290×100×10 S275, 4 M20 8.8 sur une file, p1 = 70, e1 = 40, e2,b = 40, gh = 10 (zp = 50 : plat court), "
              "soudures de 8 mm (a = 5,6), VEd = 200 kN ; γM2 = 1,1 pour les sections nettes et γM1 = 1,0 (annexe "
              "britannique, reprise pour comparer)",
         inp=SCI_1A, rows=[
             ["p. 139", "Check 2 — groupe de boulons en cisaillement", 288, vr("bv"), "kN", ""],
             ["p. 139", "Check 2 — pression diamétrale sur le plat", 332, vr("pdL"), "kN", E_SCI],
             ["p. 139", "Check 2 — pression diamétrale sur l'âme portée", 291, vr("pdS"), "kN", E_SCI],
             ["p. 139", "Check 3 — plat en cisaillement, min(brut ; net ; bloc)", 347,
              lambda R: min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd), "kN", ""],
             ["p. 139", "Check 3 — plat COURT : Wel·fy/(z·γM0)", 771, lambda R: R.Wel_p * R.fy_L / R.z_p / 1000, "kN", ""],
             ["p. 139", "Check 4 — âme portée en cisaillement, min(brut ; net ; bloc)", 312,
              lambda R: min(R.ck["vgS"].Rd, R.ck["vnS"].Rd, R.ck["vbS"].Rd), "kN", ""],
             ["p. 139", "Check 5 — résistance en flexion au grugeage Mv,N,Rd", 89, rd("mN"), "kNm", ""],
             ["p. 139", "Check 5 — moment au grugeage VEd·(gh + ln)", 24, lambda R: R.ck["mN"].Ed, "kNm", ""],
             ["p. 139", "Check 6 — longueur limite du grugeage", 413, rd("stN"), "mm", ""]]),
    dict(id="BM3", titre="SCI P358, plats d'âme, exemple 1 — poutre 533×210×92 UKB grugée, DEUX files, "
                         "VEd = 350 kN, plat 360 × 160 × 10 : grugeage court (xN = 10 mm < 2d)",
         src=SRC_SCI, mat=None, source=MT.SOURCE_TABLE,
         data="533×210×92 UKB S275 (tw 10,1) grugée 50 × 110 sur la même poutre porteuse, plat 360×160×10 S275, "
              "5 × 2 M20 8.8, p2 = 60 (zp = 50 : plat court pour le SCI ; z = 80), VEd = 350 kN. L'extrémité du "
              "grugeage est à xN = 110 − 100 = 10 mm de la 2e file : le guide prend le plus grand des deux bras",
         inp=SCI_1B, rows=[
             ["p. 139", "Check 2 — groupe de boulons en cisaillement", 577, vr("bv"), "kN", E_SCI],
             ["p. 139", "Check 2 — pression diamétrale sur le plat", 574, vr("pdL"), "kN", ""],
             ["p. 139", "Check 2 — pression diamétrale sur l'âme portée", 565, vr("pdS"), "kN", ""],
             ["p. 139", "Check 3 — plat en cisaillement, min(brut ; net ; bloc)", 450,
              lambda R: min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd), "kN", ""],
             ["p. 139", "Check 3 — plat COURT : Wel·fy/(z·γM0)", 743, lambda R: R.Wel_p * R.fy_L / R.z_p / 1000, "kN", ""],
             ["p. 139", "Check 4 — âme portée en cisaillement, min(brut ; net ; bloc)", 481,
              lambda R: min(R.ck["vgS"].Rd, R.ck["vnS"].Rd, R.ck["vbS"].Rd), "kN", ""],
             ["p. 139", "Check 5 — résistance en flexion au grugeage (xN < 2d : Mc,Rd du Check 4)", 164, rd("mN"), "kNm", ""],
             ["p. 139", "Check 5 — moment au grugeage, max(VEd·(gh + ln) ; VEd·(gh + e2,b + p2))", 42,
              lambda R: R.ck["mN"].Ed, "kNm", ""],
             ["p. 139", "Check 6 — longueur limite du grugeage", 533, rd("stN"), "mm", ""]]),
    dict(id="BM4", titre="SCI P358, Table 5.2 — coefficient de déversement χLT du plat long (expression 6.56 de "
                         "l'EN 1993-1-1, αLT = 0,76)",
         src=SRC_SCI, mat=None, source=MT.SOURCE_TABLE,
         data="La table que le guide imprime pour le plat long (p. 120, arrondie à deux décimales) : cinq de ses "
              "lignes, recalculées par la fonction du moteur", inp={}, rows=[
             ["p. 120", "χLT pour λ̄LT = 0,25", 0.96, lambda R: SCI.chi_LT(0.25), "-", ""],
             ["p. 120", "χLT pour λ̄LT = 0,50", 0.78, lambda R: SCI.chi_LT(0.50), "-", ""],
             ["p. 120", "χLT pour λ̄LT = 0,75", 0.61, lambda R: SCI.chi_LT(0.75), "-", ""],
             ["p. 120", "χLT pour λ̄LT = 1,00", 0.47, lambda R: SCI.chi_LT(1.00), "-", ""],
             ["p. 120", "χLT pour λ̄LT = 1,25", 0.36, lambda R: SCI.chi_LT(1.25), "-", ""]]),
]

NOCOVER = [
    ["Déversement du plat long (zp > tp/0,15) et section ABCD de l'âme selon le SCI", "Les plats des exemples "
     "publiés sont COURTS pour le SCI (zp = 50 ≤ tp/0,15 = 66,7) : la table de χLT du guide est rejouée (BM4), "
     "la formule complète et la section ABCD sont recalculées à la main sur un plat réellement long (VAL-A). "
     "Formules : SCI P358 Check 3 et Check 4."],
    ["Poinçonnement de l'âme porteuse", "Les exemples du guide n'appliquent que la forme prudente "
     "(tp ≤ t2·fu,2/fy,p, sans γM2) ; la forme rigoureuse du Check 10 est recalculée à la main (VAL-B)."],
    ["Flexion du plat (hp < 2,73·z)", "Sans objet sur l'exemple publié (hp ≥ 2,73·z) : calcul manuel "
     "indépendant (VAL-B). Formule : MSB Part 5 §3.2.3, Wel = tp·hp²/6."],
    ["Soudure (double cordon sous V et VEd·z)", "Aucun exemple publié consulté : calcul manuel indépendant "
     "(VAL-B). Résistance : EN 1993-1-8 §4.5.3.3."],
    ["Poutre grugée (flexion, stabilité) et cisaillement local de l'âme porteuse",
     "Mêmes règles et mêmes nombres que le module doubles cornières (MSB Part 5 §4.2.4 / §4.2.5 ; "
     "SCI P358 un seul côté chargé), déjà ancrés par son calcul manuel VAL-A."],
]

# --- VAL-A : plat LONG (zp = 80, z = 110, hp = 360, tp = 10, S275, γM1 = 1,0), SCI P358 Check 3 et Check 4
_lam = 2.8 * math.sqrt(80 * 360 / (1.5 * 10 * 10))
_lamb = _lam / (math.pi * math.sqrt(210000 / 275))
_phi = 0.5 * (1 + 0.76 * (_lamb - 0.2) + _lamb ** 2)
_chi = 1 / (_phi + math.sqrt(_phi ** 2 - _lamb ** 2))
_MbLT = 216000 * _chi * 275 / (0.6 * 1.0) / 1e6          # Wel·χLT·fy/(0,6·γM1)
_Mel = 216000 * 275 / 1.0 / 1e6                          # Wel·fy/γM0
# section ABCD (IPE A 550, tw 9, fy 275, n1 = 5, p1 = 70, deux files, e2,b = 40, p2 = 60, hb = 547)
_VAB = 9 * (40 + 60) * 275 / S3 / 1000
_VplBC = 9 * 280 * 275 / S3 / 1000
_VBC = 350 * 280 / 547
_McBC = 275 * 9 * 280 ** 2 / 6 / 1e6                     # VBC,Ed = 179 ≤ 0,5·Vpl,BC,Rd = 200 : pas de réduction
_MRd = _McBC + _VAB * 280 / 1000

VALID = [
    dict(id="VAL-A", titre="Plat réellement LONG : l'exemple du MSB avec gh = 40 (plat 360×190×10, zp = 80 > "
                           "tp/0,15 = 66,7, z = 110) — déversement et section ABCD selon le SCI",
         inp=LONG, rows=[
        ["Élancement 2,8·√(zp·hp/(1,5·tp²))", _lam, 38.79690712, va("lam_LT"), "-"],
        ["Élancement réduit λ̄LT = λ/(π·√(E/fy))", _lamb, 0.44689, va("lamb_LT"), "-"],
        ["χLT (expression 6.56, αLT = 0,76)", _chi, 0.81685, va("chi_LT"), "-"],
        ["Wel·χLT·fy/(0,6·γM1)", _MbLT, 80.868, va("Mb_LT"), "kNm"],
        ["Résistance retenue : min(déversement ; Wel·fy/γM0)", min(_MbLT, _Mel), 59.4, lambda R: R.ck["plt"].Rd, "kNm"],
        ["ABCD – moment demandé VEd·(zp + p2)", 350 * 0.140, 49.0, lambda R: R.ck["abS"].Ed, "kNm"],
        ["ABCD – Vpl,AB,Rd = tw·(e2,b + p2)·fy/(√3·γM0)", _VAB, 142.894, lambda R: R.abcd.V_AB, "kN"],
        ["ABCD – VBC,Ed = VEd·(n1 − 1)·p1/hb", _VBC, 179.159, lambda R: R.abcd.VBC, "kN"],
        ["ABCD – Mc,BC,Rd = fy·tw·((n1 − 1)·p1)²/(6·γM0) (VBC,Ed ≤ 0,5·Vpl,BC,Rd = 200,1)", _McBC, 32.34,
         lambda R: R.abcd.Mc_BC, "kNm"],
        ["ABCD – moment résistant Mc,BC,Rd + Vpl,AB,Rd·(n1 − 1)·p1", _MRd, 72.350, lambda R: R.ck["abS"].Rd, "kNm"]]),
    dict(id="VAL-B", titre="Géométrie partagée avec le module doubles cornières (HEA 300 grugée sur HEA 400, plat "
                           "190×100×10 S355, 3 M20 8.8, e1 = 35, p1 = 60, VEd = 125 kN) — l'ancien cas par défaut, "
                           "gardé pour ses nombres communs ; le défaut du module tient désormais 2d (e1 = 40, p1 = 55)",
         inp=dict(e1_u=35, p1_u=60), rows=[
        ["Moment d'excentricité MS = VEd·z", 6.25, 6.25, va("M_S"), "kNm"],
        ["Effort maximal sur un boulon", 66.69921080659218, 66.6992108065922, va("F_b"), "kN"],
        # ancre VAL-A du module doubles cornières (même modèle d'interaction)
        ["Taux – pression diamétrale de l'âme (interaction)",
         0.739914371796275, 0.739914371796275, lambda R: R.ck["pdS"].eta, "-"],
        ["Plat – cisaillement brut VRd,g = hp·tp/1,27·fy/√3", 306.7176754884, 306.717675488,
         rd("pcg"), "kN"],
        ["Soudure – qz = VEd/(2·(hp − 2a))", 125000 / (2 * 178), 351.123595506, va("qz_w"), "N/mm"],
        ["Soudure – qx = 3·MS/(hp − 2a)²", 3 * 6.25e6 / 178 ** 2, 591.781340866, va("qx_w"), "N/mm"],
        ["Soudure – résistance Fw,Rd = fvw,d·a (a = 6)", 6 * 510 / (S3 * 0.9 * 1.25), 1570.39273219,
         lambda R: R.ck["w"].Rd, "N/mm"],
        # ancre VAL-A du module doubles cornières (même règle de grugeage)
        ["Grugeage – moment résistant Mv,N,Rd", 49.47898619990443,
         49.4789861999044, rd("mN"), "kNm"],
        ["Cisaillement local âme porteuse – Av = tw·hp", 2090, 2090, va("Av_P"), "mm²"],
        ["Poinçonnement de l'âme porteuse – tw·fu·hp²/(6·γM2) (tw = 11, fu = 510, γM2 = 1,25)",
         11 * 510 * 190 ** 2 / (6 * 1.25) / 1e6, 27.0028, rd("poi"), "kNm"],
        ["Poinçonnement – forme prudente : tp ≤ tw·fu/(fy,p·γM2)", 11 * 510 / (355 * 1.25), 12.642, va("tp_poi"), "mm"]]),
]


def run_bench():
    """Rejoue l'exemple publié et les calculs manuels ; retourne ``bench``,
    ``valid``, ``worst``, ``ok`` et ``statut`` (« VALIDÉ » / « À CONTRÔLER »)."""
    out = []; worst = 0; n_unexp = 0
    for b in BENCH:
        R = compute(b["inp"], b.get("mat", MT.EXEMPLES)); rows = []
        for r in b["rows"]:
            v = r[3](R); ec = (v - r[2]) / r[2]; a = abs(ec)
            if a <= 0.01:
                st = "OK"
            elif a <= 0.03:
                st = "Écart expliqué" if r[5] else "À expliquer"
            else:
                st = "Écart > 3 % documenté" if r[5] else "À vérifier"
            if a > 0.01 and not r[5]:
                n_unexp += 1
            worst = mx(worst, a)
            rows.append(dict(page=r[0], lab=r[1], ref=r[2], val=v, unit=r[4], ec=ec, st=st, exp=r[5]))
        out.append(dict(id=b["id"], titre=b["titre"], data=MT.avec_exemple(b["data"], R, b.get("source", MT.SOURCE_EXEMPLES)),
                        rows=rows, src=b.get("src", SRC)))
    nv = 0; val = []
    for c in VALID:
        R = compute(c["inp"], MT.EXEMPLES); rows = []
        for r in c["rows"]:
            v = r[3](R); ec = (v - r[1]) / r[1] if r[1] != 0 else v
            if abs(ec) > 0.001:
                nv += 1
            rows.append(dict(lab=r[0], hand=r[1], xls=r[2], val=v, unit=r[4], ec=ec,
                             st="OK" if abs(ec) <= 0.001 else "À vérifier"))
        val.append(dict(id=c["id"], titre=MT.avec_exemple(c["titre"], R), rows=rows))
    ok = n_unexp == 0 and nv == 0
    return dict(bench=out, valid=val, worst=worst, ok=ok, statut="VALIDÉ" if ok else "À CONTRÔLER")

# -*- coding: utf-8 -*-
"""Benchmark du module platine d'about : les exemples publiés des deux guides
rejoués EN DIRECT — MSB Part 5 §2.4 (effort tranchant ET tying) et, depuis le
lot de clôture du 27/11/2026, l'exemple 1 du SCI P358 (deux poutres grugées
sur une poutre porteuse : la procédure RETENUE par le bureau, élément porteur
et grugeage compris) — et des calculs manuels indépendants pour ce que les
exemples ne couvrent pas (soudure en contrainte, cas par défaut).

Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Un écart
qui porte une explication (``exp``) ne compte pas comme inattendu. Aucune
formule n'a été ajustée pour coller à la source.
"""
import math

from acier.bibliotheques import PERSO
from acier.js import mx
from .entrees import NAT_TYING
from .moteur import compute
from acier.assemblages import materiau as MT    # le matériau des exemples (validation de formules)
from acier.assemblages import sci as SCI

S3 = math.sqrt(3)

E_AB = "Arrondi de la source : αb = 0,61 pour e1/3d0 = 40/66 = 0,606."
E_FV = "Arrondi de la source : Fv,Rd = 94 kN pour 94,08 kN."
E_M = "Arrondi de la source : m = 59 mm pour 59,16 mm (a = 5,6 mm)."

# Exemple publié : IPE A 550 S275 portée ; l'exemple ne définit PAS
# l'élément porteur — âme de 15 mm S275 retenue (celle du benchmark du plat
# d'âme), sans effet sur les valeurs comparées. e1 = 40 et p1 = 70 sont
# reconstitués de hp = 430 et de 6 rangées (40 + 5 × 70 + 40).
WE = dict(prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275",
          prof_P=PERSO, hP_u=800, bP_u=300, twP_u=15, tfP_u=30, rP_u=27, nu_P="S275",
          nu_pl="S275", d_top=60, d_nt=0, d_nb=0, l_n=0, z_C=50,
          hp_u=430, tp_u=12, bp_u=200, boulon_u="M20", classe="8.8",
          n1_u=6, p1_u=70, e1_u=40, p3_u=140, a_w=5.6,
          V_Ed=550, N_Ed=275, nat_N=NAT_TYING, g_Mu=1.1)


# Exemple 1 du SCI P358 (platines d'about, p. 32-43) : deux poutres S275 grugées (50 × 110) sur l'âme d'une
# 610×229×140 UKB, platines 200 mm de large, M20 8.8, p3 = 140, e1 = 40, p1 = 70, soudures de 6 mm (a = 4,24).
# Le matériau du guide (S275 : fy 275 / 265 au-delà de 16 mm, fu 410, βw 0,85) EST la table du site ; seul
# γM2 des sections nettes est celui de l'annexe britannique (1,1), repris ici pour comparer les valeurs imprimées.
_SCI_P = dict(prof_P=PERSO, hP_u=617.2, bP_u=230.2, twP_u=13.1, tfP_u=22.1, rP_u=12.7, nu_P="S275")
_SCI_C = dict(nu_pl="S275", boulon_u="M20", classe="8.8", p1_u=70, e1_u=40, p3_u=140, bp_u=200, a_w=4.24,
              d_top=0, d_nt=50, d_nb=0, l_n=110, z_C=50, g_M2n=1.1)
SCI_1A = dict(_SCI_P, prof_S=PERSO, hS_u=412.8, bS_u=179.5, twS_u=9.5, tfS_u=16.0, rS_u=10.2, nu_S="S275",
              n1_u=4, hp_u=290, tp_u=10, V_Ed=340, **_SCI_C)
SCI_1B = dict(_SCI_P, prof_S=PERSO, hS_u=533.1, bS_u=209.3, twS_u=10.1, tfS_u=15.6, rS_u=12.7, nu_S="S275",
              n1_u=6, hp_u=430, tp_u=12, V_Ed=550, **_SCI_C)
E_FB = "Arrondi de la source : αb = e1/3d0 = 0,61 pour 0,606."
E_09 = ("Le guide MSB P5 calcule l'âme sans le 0,9 ; la ligne compare SA valeur. Le moteur applique la règle RETENUE "
        "(SCI P358 Check 4, Av = 0,9·hp·tw) et rend 0,9 × cette valeur : 553 kN.")


def rd(k):
    return lambda R: R.ck[k].Rd


def va(k):
    return lambda R: R[k]


SRC = dict(
    titre="Steel Buildings in Europe – Multi-Storey Steel Buildings – Part 5: Joint Design "
          "— §2.4 Worked Example: Partial depth end plate",
    org="ArcelorMittal / Peiner Träger / Corus – contenu technique CTICM et SCI (projet RFCS SECHALO, RFS2-CT-2008-0030)",
    annee="feuilles de calcul 1 à 7 du §2.4 (même document que le benchmark du plat d'âme)",
    norme="EN 1993-1-8:2005 et EN 1993-1-1:2005, valeurs recommandées ; tying avec fu/γMu, γMu = 1,1 "
          "(MSB P5 §2.3)",
    url="https://steelconstruction.info/images/5/53/SBE_MS5.pdf")

SRC_SCI = dict(
    titre="SCI P358 — Joints in Steel Construction: Simple Joints to Eurocode 3 — Partial depth end plates, "
          "Worked example 1 (beam to beam)",
    org="The Steel Construction Institute / The British Constructional Steelwork Association (édition de 2014)",
    annee="pages 32 à 43 : Checks 4, 5, 6, 8, 9 et 10 de la procédure",
    norme="BS EN 1993-1-8 et BS EN 1993-1-1 avec l'annexe britannique (γM2 = 1,25 boulons, 1,1 sections nettes)",
    url="https://www.steelconstruction.info/images/3/3d/SCI_P358.pdf")

BENCH = [
    dict(id="BM1", titre="§2.4 Worked Example: Partial depth end plate — VEd = 550 kN et force de "
                         "chaînage (tying) 275 kN, platine 430 × 200 × 12 S275, 12 boulons M20 8.8",
         data="IPE A 550 S275 (tw 9), platine 430×200×12 S275, M20 8.8, 6 rangées × 2 files, p3 = 140, "
              "e2 = 30, cordon 8 mm (a = 5,6), VEd = 550 kN, tying 275 kN (fu/γMu). e1 = 40 et p1 = 70 "
              "reconstitués (40 + 5 × 70 + 40 = 430) ; élément porteur non défini par l'exemple (âme de "
              "15 mm S275 retenue, sans effet sur les valeurs comparées)",
         inp=WE, rows=[
             ["§2.4", "Âme de la poutre – cisaillement sur hp, SANS le 0,9 du SCI (Av = 430 × 9 = 3870 mm²)", 614,
              lambda R: R.ck["vwS"].Rd / SCI.K_AV, "kN", E_09],
             ["§2.4", "Un boulon en cisaillement Fv,Rd", 94, va("Fv_Rd"), "kN", E_FV],
             ["§2.4", "Pression diamétrale sur la platine – boulons d'extrémité", 107, va("Fbb_L"), "kN", E_AB],
             ["§2.4", "Pression diamétrale sur la platine – boulons intérieurs", 142, va("Fbi_L"), "kN", ""],
             ["§2.4", "Groupe de boulons FRd = 0,8·ns·Fv,Rd (les deux guides rendent la même valeur)", 902, rd("bg"), "kN", E_FV],
             ["§2.4", "Platine – cisaillement, min(brut ; net ; bloc) = rupture de bloc", 1182,
              lambda R: min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd), "kN", ""],
             ["§2.4", "Tying – platine en flexion (tronçon en T, mode 1)", 493, rd("ptT"), "kN", E_M],
             ["§2.4", "Tying – âme de la poutre en traction tw·hp·fu/γMu", 1513, rd("tnS"), "kN", ""],
             ["§2.2", "Cordon à pleine résistance en S275 : a ≥ 0,48·tw", 0.48, va("k_plein"), "-", ""]]),
    dict(id="BM2", titre="SCI P358, platines d'about, exemple 1 — poutre 406×178×74 UKB grugée, VEd = 340 kN, "
                         "platine 290 × 200 × 10, 8 boulons M20 8.8",
         src=SRC_SCI, mat=None, source=MT.SOURCE_TABLE,
         data="406×178×74 UKB S275 (tw 9,5) grugée 50 × 110 sur l'âme d'une 610×229×140 UKB (tw 13,1), platine "
              "290×200×10 S275, M20 8.8, 4 rangées × 2 files, p3 = 140, e1 = 40, p1 = 70, soudure de 6 mm (a = 4,24), "
              "VEd = 340 kN ; γM2 = 1,1 pour les sections nettes (annexe britannique, reprise pour comparer)",
         inp=SCI_1A, rows=[
             ["p. 35", "Check 4 — âme portée au droit de la platine, Av = 0,9·hp·tw", 394, rd("vwS"), "kN", ""],
             ["p. 36", "Check 5 — section grugée, Vpl,N,Rd", 561, rd("vgS"), "kN", ""],
             ["p. 37", "Check 5 — module élastique du Té, Wel,N (cm³)", 324, lambda R: R.W_N / 1000, "cm³", ""],
             ["p. 37", "Check 5 — Mv,N,Rd (cisaillement fort)", 85.1, rd("mN"), "kNm", ""],
             ["p. 38", "Check 6 — longueur limite du grugeage (hb/tw ≤ 54,3 : ln ≤ hb)", 412.8, rd("stN"), "mm", ""],
             ["p. 39", "Check 8 — pression diamétrale sur la platine, boulon d'extrémité", 85, va("Fbb_L"), "kN", E_FB],
             ["p. 40", "Check 8 — groupe : Fb,Rd ≤ 0,8·Fv,Rd → n·Fb,Rd", 602, rd("bg"), "kN", ""],
             ["p. 40", "Check 9 — platine, cisaillement brut", 725, rd("pcg"), "kN", ""],
             ["p. 41", "Check 9 — platine, cisaillement net", 869, rd("pcn"), "kN", ""],
             ["p. 41", "Check 9 — platine, rupture de bloc", 691, rd("pcb"), "kN", ""],
             ["p. 42", "Check 10 — âme porteuse, cisaillement local brut (et = 90, eb = 70)", 742, rd("vlP"), "kN", ""],
             ["p. 42", "Check 10 — âme porteuse, cisaillement local net", 795, rd("vnP"), "kN", ""],
             ["p. 43", "Check 10 — âme porteuse, pression diamétrale d'un boulon", 174, va("Fbi_P"), "kN", ""]]),
    dict(id="BM3", titre="SCI P358, platines d'about, exemple 1 — poutre 533×210×92 UKB grugée, VEd = 550 kN, "
                         "platine 430 × 200 × 12, 12 boulons M20 8.8",
         src=SRC_SCI, mat=None, source=MT.SOURCE_TABLE,
         data="533×210×92 UKB S275 (tw 10,1) grugée 50 × 110 sur la même poutre porteuse, platine 430×200×12 S275, "
              "6 rangées × 2 files, VEd = 550 kN ; mêmes détails",
         inp=SCI_1B, rows=[
             ["p. 35", "Check 4 — âme portée au droit de la platine, Av = 0,9·hp·tw", 621, rd("vwS"), "kN", ""],
             ["p. 36", "Check 5 — section grugée, Vpl,N,Rd", 794, rd("vgS"), "kN", ""],
             ["p. 37", "Check 5 — module élastique du Té, Wel,N (cm³)", 598, lambda R: R.W_N / 1000, "cm³", ""],
             ["p. 37", "Check 5 — Mv,N,Rd (cisaillement fort)", 140, rd("mN"), "kNm", ""],
             ["p. 39", "Check 8 — pression diamétrale sur la platine, boulon d'extrémité", 102, va("Fbb_L"), "kN", E_FB],
             ["p. 40", "Check 8 — groupe : Fb,Rd > 0,8·Fv,Rd → 0,8·n·Fv,Rd", 902, rd("bg"), "kN", E_FV],
             ["p. 40", "Check 9 — platine, cisaillement brut", 1290, rd("pcg"), "kN", ""],
             ["p. 41", "Check 9 — platine, cisaillement net", 1539, rd("pcn"), "kN", ""],
             ["p. 41", "Check 9 — platine, rupture de bloc", 1195, rd("pcb"), "kN", ""]]),
]

NOCOVER = [
    ["Platine courte en flexion dans son plan (hp < 1,36·p3)",
     "Aucun des exemples publiés n'a de platine courte : formule du SCI P358 Check 9 (p. 23), recalculée à la "
     "main dans les tests du lot de clôture."],
    ["Traction ELU combinée avec VEd (interaction, tronçon en T avec fy/γM0, poinçonnement)",
     "Aucun exemple publié consulté : formules directes de l'EN 1993-1-8 (Tableau 3.4, §6.2.4) ; le "
     "tronçon en T est le calcul du tying, ancré par l'exemple publié et VAL-A."],
    ["Soudure en contrainte (§4.5.3.3)",
     "Le guide justifie le cordon par la règle de pleine résistance (comparée ci-dessus) ; la "
     "vérification en contrainte est ancrée par VAL-B."],
    ["Glissement (catégories B et C)", "Aucun exemple publié consulté : formule directe EN 1993-1-8 §3.9."],
]

# --- VAL-A : tronçon en T de l'exemple publié, recalculé à la main
_m = (140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2)
_n = min(30, 1.25 * _m)
_ew = 37 / 4
_leff = 2 * min(2 * _m + 0.625 * 30 + 35, 40 + 35) + 4 * 70          # = hp = 430
_Mpl = 0.25 * _leff * 12 ** 2 * 430 / 1.1                              # N·mm
_Ft = 0.9 * 800 * 245 / 1.1                                            # N, un boulon (γMu)
_FT1 = (8 * _n - 2 * _ew) * _Mpl / (2 * _m * _n - _ew * (_m + _n)) / 1000
_FT2 = (2 * _Mpl + _n * 12 * _Ft) / (_m + _n) / 1000
_FT3 = 12 * _Ft / 1000

# --- VAL-B : cas par défaut, recalculé à la main
_k1 = 2.8 * 30 / 22 - 1.7
_Fbb = _k1 * (35 / 66) * 510 * 20 * 10 / 1.25 / 1000
_kpl = 0.9 * 1.25 * 355 / (math.sqrt(2) * 510)

VALID = [
    dict(id="VAL-A", titre="Tronçon en T de l'exemple publié (tying, fu/γMu) — m, Σleff et les trois modes",
         inp=WE, rows=[
        ["m = (p3 − tw)/2 − 0,8·a·√2", _m, 59.1643, va("m_T"), "mm"],
        ["Σleff (rangées en groupe, bords libres) = hp", _leff, 430, va("leff_T"), "mm"],
        ["Mode 1, méthode 2 : (8n − 2ew)·Mpl/(2mn − ew(m + n))", _FT1, 491.86, va("FT1"), "kN"],
        ["Mode 2 : (2·Mpl + n·ΣFt)/(m + n)", _FT2, 783.20, va("FT2"), "kN"],
        ["Mode 3 : ΣFt,Rd = 12 × 0,9·fub·As/γMu", _FT3, 1924.36, va("FT3"), "kN"]]),
    dict(id="VAL-B", titre="Cas par défaut (HEA 300 grugée sur HEA 400, platine 190×150×10 S355, "
                           "3 × 2 M20 8.8, VEd = 125 kN)",
         inp={}, rows=[
        ["Pression diamétrale, boulon d'extrémité (k1 = 2,118 ; αb = 35/66)", _Fbb, 91.66, va("Fbb_L"), "kN"],
        ["Groupe (SCI Check 8) : Fb,Rd,min = 91,66 kN > 0,8·Fv,Rd = 75,26 kN → 0,8·n·Fv,Rd",
         min(6 * _Fbb, 0.8 * 6 * 0.6 * 800 * 245 / 1.25 / 1000), 451.58, rd("bg"), "kN"],
        ["Âme portée – 0,9·hp·tw·fy/√3", 0.9 * 190 * 8.5 * 355 / S3 / 1000, 297.91, rd("vwS"), "kN"],
        ["Rupture de bloc, k = 1 (hp ≥ 1,36·p3)",
         2 * (510 * 10 * (30 - 11) / 1.25 + 355 * 10 * (190 - 35 - 2.5 * 22) / S3) / 1000, 564.96, rd("pcb"), "kN"],
        ["Grugeage – MEd = VEd·(tp + c)", 125 * 0.160, 20.0, lambda R: R.ck["mN"].Ed, "kNm"],
        # même ancre que les deux autres modules de la famille (même règle de grugeage)
        ["Grugeage – Mv,N,Rd", 49.47898619990443, 49.4790, rd("mN"), "kNm"],
        ["Âme porteuse – cisaillement local tw·(et + 2·p1 + eb)·fy/√3 (et = 85, eb = min(105 ; p3/2 ; 5d) = 45)",
         11 * (85 + 120 + 45) * 355 / S3 / 1000, 563.64, rd("vlP"), "kN"],
        ["Soudure – Fw,Rd = fu/(√3·βw·γM2)·a (a = 5)", 5 * 510 / (S3 * 0.9 * 1.25), 1308.66, rd("w"), "N/mm"],
        ["Pleine résistance S355 : a/tw = βw·γM2·fy/(√2·fu)", _kpl, 0.5537, va("k_plein"), "-"],
        ["Ductilité : tp,max = 0,36·d·√(fub/fy)", 0.36 * 20 * math.sqrt(800 / 355), 10.81, va("tp_max"), "mm"]]),
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

# -*- coding: utf-8 -*-
"""Benchmark du module platine d'about : l'exemple publié du guide (MSB Part
5 §2.4, Worked Example: Partial depth end plate) rejoué EN DIRECT — effort
tranchant ET tying — et des calculs manuels indépendants pour ce que
l'exemple ne couvre pas (grugeage, élément porteur, soudure en contrainte).

Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Un écart
qui porte une explication (``exp``) ne compte pas comme inattendu. Aucune
formule n'a été ajustée pour coller à la source.
"""
import math

from acier.bibliotheques import PERSO
from acier.js import mx
from .entrees import NAT_TYING
from .moteur import compute

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

BENCH = [
    dict(id="BM1", titre="§2.4 Worked Example: Partial depth end plate — VEd = 550 kN et force de "
                         "chaînage (tying) 275 kN, platine 430 × 200 × 12 S275, 12 boulons M20 8.8",
         data="IPE A 550 S275 (tw 9), platine 430×200×12 S275, M20 8.8, 6 rangées × 2 files, p3 = 140, "
              "e2 = 30, cordon 8 mm (a = 5,6), VEd = 550 kN, tying 275 kN (fu/γMu). e1 = 40 et p1 = 70 "
              "reconstitués (40 + 5 × 70 + 40 = 430) ; élément porteur non défini par l'exemple (âme de "
              "15 mm S275 retenue, sans effet sur les valeurs comparées)",
         inp=WE, rows=[
             ["§2.4", "Âme de la poutre – cisaillement sur hp (Av = 430 × 9 = 3870 mm²)", 614, rd("vwS"), "kN", ""],
             ["§2.4", "Un boulon en cisaillement Fv,Rd", 94, va("Fv_Rd"), "kN", E_FV],
             ["§2.4", "Pression diamétrale sur la platine – boulons d'extrémité", 107, va("Fbb_L"), "kN", E_AB],
             ["§2.4", "Pression diamétrale sur la platine – boulons intérieurs", 142, va("Fbi_L"), "kN", ""],
             ["§2.4", "Groupe de boulons FRd = 0,8·ns·Fv,Rd (Fv,Rd ≤ (Fb,Rd)min)", 902, rd("bg"), "kN", E_FV],
             ["§2.4", "Platine – cisaillement, min(brut ; net ; bloc) = rupture de bloc", 1182,
              lambda R: min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd), "kN", ""],
             ["§2.4", "Tying – platine en flexion (tronçon en T, mode 1)", 493, rd("ptT"), "kN", E_M],
             ["§2.4", "Tying – âme de la poutre en traction tw·hp·fu/γMu", 1513, rd("tnS"), "kN", ""],
             ["§2.2", "Cordon à pleine résistance en S275 : a ≥ 0,48·tw", 0.48, va("k_plein"), "-", ""]]),
]

NOCOVER = [
    ["Élément porteur (pression diamétrale et cisaillement local de l'âme porteuse)",
     "Le guide MSB P5 §2.2 ne vérifie pas l'élément porteur (l'exemple n'en définit pas) : modèle SCI "
     "P358, même formulation que les deux autres assemblages ; calcul manuel VAL-B."],
    ["Poutre grugée (cisaillement, flexion, stabilité)",
     "L'exemple publié n'est pas grugé : mêmes règles que les deux autres assemblages (MSB P5 §4.2.4 / "
     "§4.2.5), bras de levier tp + c ; calcul manuel VAL-B."],
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
        ["Groupe : ns·(Fb,Rd)min (Fb,ext < Fv,Rd < Fb,int)", 6 * _Fbb, 549.96, rd("bg"), "kN"],
        ["Âme portée – hp·tw·fy/√3", 190 * 8.5 * 355 / S3 / 1000, 331.01, rd("vwS"), "kN"],
        ["Rupture de bloc, k = 1 (hp ≥ 1,36·p3)",
         2 * (510 * 10 * (30 - 11) / 1.25 + 355 * 10 * (190 - 35 - 2.5 * 22) / S3) / 1000, 564.96, rd("pcb"), "kN"],
        ["Grugeage – MEd = VEd·(tp + c)", 125 * 0.160, 20.0, lambda R: R.ck["mN"].Ed, "kNm"],
        ["Grugeage – Mv,N,Rd (même ancre que les deux autres modules)", 49.47898619990443, 49.4790, rd("mN"), "kNm"],
        ["Âme porteuse – cisaillement local tw·hp·fy/√3", 11 * 190 * 355 / S3 / 1000, 428.37, rd("vlP"), "kN"],
        ["Soudure – Fw,Rd = fu/(√3·βw·γM2)·a (a = 5)", 5 * 510 / (S3 * 0.9 * 1.25), 1308.66, rd("w"), "N/mm"],
        ["Pleine résistance S355 : a/tw = βw·γM2·fy/(√2·fu)", _kpl, 0.5537, va("k_plein"), "-"],
        ["Ductilité : tp,max = 0,36·d·√(fub/fy)", 0.36 * 20 * math.sqrt(800 / 355), 10.81, va("tp_max"), "mm"]]),
]


def run_bench():
    """Rejoue l'exemple publié et les calculs manuels ; retourne ``bench``,
    ``valid``, ``worst``, ``ok`` et ``statut`` (« VALIDÉ » / « À CONTRÔLER »)."""
    out = []; worst = 0; n_unexp = 0
    for b in BENCH:
        R = compute(b["inp"]); rows = []
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
        out.append(dict(id=b["id"], titre=b["titre"], data=b["data"], rows=rows))
    nv = 0; val = []
    for c in VALID:
        R = compute(c["inp"]); rows = []
        for r in c["rows"]:
            v = r[3](R); ec = (v - r[1]) / r[1] if r[1] != 0 else v
            if abs(ec) > 0.001:
                nv += 1
            rows.append(dict(lab=r[0], hand=r[1], xls=r[2], val=v, unit=r[4], ec=ec,
                             st="OK" if abs(ec) <= 0.001 else "À vérifier"))
        val.append(dict(id=c["id"], titre=c["titre"], rows=rows))
    ok = n_unexp == 0 and nv == 0
    return dict(bench=out, valid=val, worst=worst, ok=ok, statut="VALIDÉ" if ok else "À CONTRÔLER")

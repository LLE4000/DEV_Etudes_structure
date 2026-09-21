# -*- coding: utf-8 -*-
"""Benchmark du module plat d'âme : l'exemple publié du guide (MSB Part 5
§3.4, Worked Example: Fin Plate) rejoué EN DIRECT — un plan de cisaillement,
sans la transposition ×2 du benchmark du module doubles cornières — et des
calculs manuels indépendants pour ce que l'exemple ne couvre pas
(déversement du plat long, soudure, grugeage).

Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Un écart
qui porte une explication (``exp``) ne compte pas comme inattendu. Aucune
formule n'a été ajustée pour coller à la source.
"""
import math

from acier.bibliotheques import PERSO
from acier.js import dv, mx
from .moteur import compute, pb_bs5950

S3 = math.sqrt(3)

E_ARR = "Arrondis intermédiaires de la source (αb = 0,61 pour 0,606 ; Fv,Rd = 94 pour 94,08)."
E_AIRE = ("La source utilise l'aire catalogue A = 11700 mm² ; l'outil calcule A = 2b·tf + (h − 2tf)·tw "
          "+ (4 − π)·r² = 11729 mm² (+0,25 %), d'où Av = 6030 au lieu de 6001 mm².")

WE = dict(prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275",
          prof_P=PERSO, hP_u=800, bP_u=300, twP_u=15, tfP_u=30, rP_u=27, nu_P="S275",
          nu_pl="S275", d_top=60, d_nt=0, d_nb=0, l_n=0, g_h=10, z_C=50,
          hp_u=360, tp_u=10, bp_u=160, boulon_u="M20", classe="8.8",
          n1_u=5, n2_u=2, p1_u=70, p2_u=60, e1_u=40, e2b_u=40, V_Ed=350, g_M1=1.0)


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

BENCH = [
    dict(id="BM1", titre="§3.4 Worked Example: Fin Plate — VEd = 350 kN, 2 files de boulons, z = 80 mm, "
                         "p. 5-38 à 5-45 (le MÊME exemple sert, transposé, de BM4 au module doubles cornières)",
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
             ["p. 5-45", "Âme – rupture de bloc VRd,b (non grugée : (n1 − 1)·d0)", 507, rd("vbS"), "kN", ""]]),
]

NOCOVER = [
    ["Déversement du plat long (z > tp/0,15)", "L'exemple du guide est un plat long, mais la valeur imprimée "
     "n'a pas pu être extraite de la source ici : calcul manuel indépendant (VAL-A). Formules : SCI P358, "
     "courbe BS 5950-1 Annexe B.2.1 (Table 17), E = 205 000 MPa."],
    ["Flexion du plat (hp < 2,73·z)", "Sans objet sur l'exemple publié (hp ≥ 2,73·z) : calcul manuel "
     "indépendant (VAL-B). Formule : MSB Part 5 §3.2.3, Wel = tp·hp²/6."],
    ["Soudure (double cordon sous V et VEd·z)", "Aucun exemple publié consulté : calcul manuel indépendant "
     "(VAL-B). Résistance : EN 1993-1-8 §4.5.3.3."],
    ["Poutre grugée (flexion, stabilité) et cisaillement local de l'âme porteuse",
     "Mêmes règles et mêmes nombres que le module doubles cornières (MSB Part 5 §4.2.4 / §4.2.5 ; "
     "SCI P358 un seul côté chargé), déjà ancrés par son calcul manuel VAL-A."],
]

_lam = 2.8 * math.sqrt(80 * 360 / (1.5 * 10 * 10))
_pE = math.pi ** 2 * 205000 / _lam ** 2
_lam0 = 0.4 * math.sqrt(math.pi ** 2 * 205000 / 275)
_eta = 0.007 * (_lam - _lam0)
_phi = (275 + (_eta + 1) * _pE) / 2
_pb = _pE * 275 / (_phi + math.sqrt(_phi * _phi - _pE * 275))

VALID = [
    dict(id="VAL-A", titre="Déversement du plat long sur l'exemple publié (plat 360×10, z = 80 > tp/0,15 = 66,7)",
         inp=WE, rows=[
        ["Élancement λLT = 2,8·√(z·hp/(1,5·tp²))", _lam, 38.79690712, va("lam_LT"), "-"],
        ["Contrainte de déversement pb (Perry-Robertson, E = 205 000)", _pb, 264.6626333, va("pb_LT"), "MPa"],
        ["Moment résistant Mb,Rd = Wel·pb/γM1 (γM1 = 1,0)", 216000 * _pb / 1e6, 57.16712879,
         lambda R: R.ck["plt"].Rd, "kNm"],
        ["Plateau : pb = py pour λ ≤ λL0", 275, 275, lambda R: pb_bs5950(20, 275), "MPa"]]),
    dict(id="VAL-B", titre="Cas par défaut (HEA 300 grugée sur HEA 400, plat 190×100×10 S355, 3 M20 8.8, "
                           "VEd = 125 kN) — mêmes nombres partagés que le module doubles cornières",
         inp={}, rows=[
        ["Moment d'excentricité MS = VEd·z", 6.25, 6.25, va("M_S"), "kNm"],
        ["Effort maximal sur un boulon", 66.69921080659218, 66.6992108065922, va("F_b"), "kN"],
        ["Taux – pression diamétrale de l'âme (interaction) — ancre VAL-A du module 1",
         0.739914371796275, 0.739914371796275, lambda R: R.ck["pdS"].eta, "-"],
        ["Plat – cisaillement brut VRd,g = hp·tp/1,27·fy/√3", 306.7176754884, 306.717675488,
         rd("pcg"), "kN"],
        ["Soudure – qz = VEd/(2hp)", 125000 / 380, 328.947368421, va("qz_w"), "N/mm"],
        ["Soudure – qx = 3·MS/hp²", 3 * 6.25e6 / 36100, 519.390581717, va("qx_w"), "N/mm"],
        ["Soudure – résistance Fw,Rd = fvw,d·a (a = 5)", 5 * 510 / (S3 * 0.9 * 1.25), 1308.66061016,
         lambda R: R.ck["w"].Rd, "N/mm"],
        ["Grugeage – moment résistant Mv,N,Rd (ancre VAL-A du module 1)", 49.47898619990443,
         49.4789861999044, rd("mN"), "kNm"],
        ["Cisaillement local âme porteuse – Av = tw·hp", 2090, 2090, va("Av_P"), "mm²"]]),
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

# -*- coding: utf-8 -*-
"""Benchmark du module : exemples publiés et calculs manuels indépendants.

Transcription des tables ``BENCH`` (4 exemples, 37 valeurs publiées),
``NOCOVER`` (vérifications sans exemple publié), ``VALID`` (2 cas de calcul
manuel, 13 + 9 valeurs) et de ``runBench`` du moteur de référence. Les
valeurs de référence, pages et explications sont celles de la source citée
(``SRC``) ; aucune formule n'a été ajustée pour coller à la source.

Seuils : écart ≤ 1 % OK ; 1 à 3 % à expliquer ; > 3 % à vérifier. Un écart
qui porte une explication (``exp``) ne compte pas comme inattendu.
"""
import math

from acier.bibliotheques import PERSO, ORI_S
from acier.js import dv, mx
from .moteur import compute

S3 = math.sqrt(3)

E_ARR = "Arrondis intermédiaires de la source (αb = 0,61 pour 0,606 ; Fv,Rd = 94 pour 94,08 ; m = 31 pour 31,2)."
E_AIRE = "La source utilise l'aire catalogue A = 11700 mm² ; l'outil calcule A = 2b·tf + (h − 2tf)·tw + (4 − π)·r² = 11729 mm² (+0,25 %), d'où Av = 6030 au lieu de 6001 mm²."
E_BLOC = "Hypothèse différente sur Anv : le chapitre « fin plate » de la source (§3.2.5.1) retire (n1 − 1)·d0 pour une poutre non grugée ; l'outil applique la règle du chapitre « double angle web cleats » (§4.2.3.1) : (n1 − 0,5)·d0, côté sécurité. Formule non modifiée ; la ligne suivante reproduit l'hypothèse de la source."

IPEA = dict(prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275", nu_C="S275",
            prof_P=PERSO, hP_u=800, bP_u=300, twP_u=15, tfP_u=30, rP_u=27, nu_P="S275", d_top=60,
            d_nt=0, d_nb=0, l_n=0, g_h=10, boulon_u="M20", classe="8.8", z_C=50, e2b_u=40)


def mix(*ds):
    o = {}
    for d in ds:
        o.update(d)
    return o


BM1 = mix(IPEA, dict(corn_u=PERSO, k1_u=90, k2_u=90, kt_u=10, kr_u=11, V_Ed=450, LC_u=430, n1S_u=6, n2S_u=1,
                     p1S_u=70, e1S_u=40, n1P_u=6, n2P_u=1, p1P_u=70, e1P_u=40, gA_u=50))
BM4 = mix(IPEA, dict(corn_u=PERSO, k1_u=160, k2_u=90, kt_u=10, kr_u=11, orient=ORI_S, V_Ed=350, LC_u=360,
                     n1S_u=5, n2S_u=2, p1S_u=70, p2_S=60, e1S_u=40, n1P_u=5, n2P_u=1, p1P_u=70, e1P_u=40, gA_u=50))


def vr(k):
    """Résistance du groupe exprimée en VEd : VEd / η de la vérification k."""
    return lambda R: dv(R.u.V_Ed, R.ck[k].eta)


def rd(k):
    return lambda R: R.ck[k].Rd


def va(k):
    return lambda R: R[k]


SRC = dict(
    titre="Steel Buildings in Europe – Multi-Storey Steel Buildings – Part 5: Joint Design",
    org="ArcelorMittal / Peiner Träger / Corus – contenu technique CTICM et SCI (projet RFCS SECHALO, RFS2-CT-2008-0030)",
    annee="feuilles de calcul datées 06/2009",
    norme="EN 1993-1-8:2005 et EN 1993-1-1:2005, valeurs recommandées (γM0 = 1,0 ; γM2 = 1,25 ; γMu = 1,1)",
    url="https://steelconstruction.info/images/5/53/SBE_MS5.pdf")


def _bm4_anv_source(R):
    u = R.u
    anv = R.tw_S * (R.e1b_S + (R.n1_S - 1) * R.p1_S - (R.n1_S - 1) * R.d_0)
    return (dv(0.5 * R.fu_S * R.Ant_S, u.g_M2n) + dv(R.fy_S * anv / S3, u.g_M0)) / 1000


BENCH = [
    dict(id="BM1", titre="§4.4 Worked Example: Angle Web Cleats – effort tranchant VEd = 450 kN, p. 5-68 à 5-77",
         data="IPE A 550 S275 (h 547, b 210, tw 9, tf 15,7, r 24), 2 L90x90x10 S275, M20 8.8, n1 = 6, p1 = 70, e1 = 40, e2 = 40, Lc = 430, gh = 10, e2,b = 40 (z = 50), p3 = 109, poutre non grugée",
         inp=BM1, rows=[
             ["p. 5-70", "Cisaillement d'un boulon Fv,Rd (1 plan)", 94, va("Fv_Rd"), "kN", E_ARR],
             ["p. 5-70", "Groupe S excentré – cisaillement des boulons VRd", 962, vr("bv_S"), "kN", E_ARR],
             ["p. 5-71", "Pression diamétrale cornière Fb,ver,Rd", 105, va("Fbv_B"), "kN", E_ARR],
             ["p. 5-71", "Pression diamétrale cornière Fb,hor,Rd", 105, va("Fbh_B"), "kN", E_ARR],
             ["p. 5-71", "Groupe S – pression diamétrale sur les cornières VRd", 1075, vr("pdB"), "kN", E_ARR],
             ["p. 5-72", "Pression diamétrale âme Fb,ver,Rd", 125, va("Fbv_S"), "kN", ""],
             ["p. 5-73", "Pression diamétrale âme Fb,hor,Rd", 94, va("Fbh_S"), "kN", ""],
             ["p. 5-73", "Groupe S – pression diamétrale sur l'âme VRd", 583, vr("pdS"), "kN", ""],
             ["p. 5-74", "Côté poutre porteuse – FRd = 0,8·ns·Fv,Rd", 902, vr("bv_P"), "kN", E_ARR],
             ["p. 5-74", "Cornières – cisaillement section brute VRd,g", 1076, rd("cgB"), "kN", ""],
             ["p. 5-75", "Cornières – cisaillement section nette VRd,n", 1184, rd("cnB"), "kN", ""],
             ["p. 5-75", "Cornières – rupture de bloc VRd,b (côté poutre portée)", 954, rd("cbB"), "kN", ""],
             ["p. 5-76", "Cornières – rupture de bloc VRd,b (côté poutre porteuse)", 954, rd("cbA"), "kN", ""],
             ["p. 5-77", "Âme – cisaillement section brute VRd,g", 953, rd("vgS"), "kN", E_AIRE],
             ["p. 5-77", "Âme – cisaillement section nette VRd,n", 956, rd("vnS"), "kN", E_AIRE],
             ["p. 5-77", "Âme – rupture de bloc VRd,b", 501, rd("vbS"), "kN", ""]]),
    dict(id="BM2", titre="§4.4 même exemple – résistances en traction (tying FEd = 370 kN), p. 5-80 à 5-82. Le guide utilise γMu = 1,1 sur fu : reproduit avec γM2 (sections nettes) = 1,1",
         data="Données BM1 + NEd = 370 kN", inp=mix(BM1, dict(N_Ed=370, g_M2n=1.1)), rows=[
             ["p. 5-80", "Cornières – rupture de bloc en traction, cas 1", 2060, va("Vt1_B"), "kN", ""],
             ["p. 5-81", "Cornières – rupture de bloc en traction, cas 2", 2195, va("Vt2_B"), "kN", ""],
             ["p. 5-82", "Âme – traction section nette", 944, rd("tnS"), "kN", ""],
             ["p. 5-82", "Âme – rupture de bloc en traction, cas 1", 927, rd("tbS"), "kN", ""]]),
    dict(id="BM3", titre="§4.4 même exemple – tronçon en T équivalent des cornières (tying), p. 5-78 à 5-79. Le guide utilise fu/γMu : reproduit avec un acier de cornière personnalisé fy = fu = 430 MPa et γM0 = γM2 = 1,1",
         data="Données BM1 + NEd = 370 kN", inp=mix(BM1, dict(N_Ed=370, nu_C=PERSO, fy_u=430, fu_u=430, bw_u=0.85, g_M0=1.1, g_M2=1.1)), rows=[
             ["p. 5-79", "Traction d'un boulon Ft,Rd,u", 160, va("Ft_Rd"), "kN", ""],
             ["p. 5-78", "Longueur efficace Σleff", 430, va("leff_T"), "mm", ""],
             ["p. 5-78", "Moment plastique Mpl,1,Rd,u", 4.2, va("Mpl_T"), "kNm", ""],
             ["p. 5-79", "Tronçon en T – mode 1", 696, va("FT_1"), "kN", E_ARR],
             ["p. 5-79", "Tronçon en T – mode 2", 1190, va("FT_2"), "kN", E_ARR],
             ["p. 5-79", "Tronçon en T – mode 3", 1920, va("FT_3"), "kN", E_ARR]]),
    dict(id="BM4", titre="§3.4 Worked Example: Fin Plate (VEd = 350 kN, 2 files de boulons, z = 80 mm), p. 5-38 à 5-45 – transposé : le plat de 10 mm (1 plan de cisaillement) est remplacé par 2 cornières de 10 mm (2 plans) ; les valeurs de la source relatives au plat et aux boulons sont multipliées par 2, celles de l'âme sont inchangées",
         data="IPE A 550 S275, plat 360x160x10 S275, M20 8.8, n1 = 5, n2 = 2, p1 = 70, p2 = 60, e1 = 40, e2 = 50, e2,b = 40, z = 80, e1,b = 90",
         inp=BM4, rows=[
             ["p. 5-40", "Inertie polaire du groupe de boulons I", 107000, va("Ip_S"), "mm²", ""],
             ["p. 5-40", "Groupe excentré à 2 files – cisaillement des boulons VRd (584 × 2)", 1168, vr("bv_S"), "kN", ""],
             ["p. 5-41", "Pression diamétrale sur le plat VRd (605 × 2)", 1210, vr("pdB"), "kN", E_ARR],
             ["p. 5-42", "Pression diamétrale sur l'âme VRd", 624, vr("pdS"), "kN", ""],
             ["p. 5-43", "Plat – cisaillement section brute (450 × 2)", 900, rd("cgB"), "kN", ""],
             ["p. 5-43", "Plat – cisaillement section nette (497 × 2)", 994, rd("cnB"), "kN", ""],
             ["p. 5-43", "Plat – rupture de bloc à 2 files (483 × 2)", 966, rd("cbB"), "kN", ""],
             ["p. 5-45", "Âme – cisaillement section brute", 953, rd("vgS"), "kN", E_AIRE],
             ["p. 5-45", "Âme – cisaillement section nette", 995, rd("vnS"), "kN", E_AIRE],
             ["p. 5-45", "Âme – rupture de bloc à 2 files, formule de l'outil", 507, rd("vbS"), "kN", E_BLOC],
             ["p. 5-45", "Âme – rupture de bloc à 2 files, avec l'hypothèse de la source sur Anv", 507, _bm4_anv_source, "kN", ""]]),
]

NOCOVER = [
    ["Interaction traction / cisaillement des boulons, glissement (cat. B et C)", "Aucun exemple publié consulté : calcul manuel indépendant (cas VAL-B). Formules : EN 1993-1-8 Tableau 3.4, §3.9.1 et §3.9.2."],
    ["Poutre secondaire grugée (flexion au droit du grugeage, stabilité locale)", "Les exemples du guide sont non grugés : calcul manuel indépendant (cas VAL-A). Formules : MSB Part 5 §4.2.4 et §4.2.5 ; seconde source : SCI P358 Checks 5 et 6."],
    ["Soudures", "Aucun exemple publié consulté pour un groupe de cordons excentré : calcul manuel indépendant (cas VAL-B). Résistance : EN 1993-1-8 §4.5.3.3."],
    ["Vérification locale de l'âme de la poutre principale", "Aucun exemple publié consulté : calcul manuel indépendant (cas VAL-A). Modèle : SCI P358 Check 10."],
]

VALID = [
    dict(id="VAL-A", titre="Cas par défaut (HEA 300 grugée sur HEA 400, S355, L100x100x10, 3 + 2×3 M20 8.8, VEd = 125 kN)", inp={}, rows=[
        ["Moment d'excentricité MS = VEd·z", 6.25, 6.25, va("M_S"), "kNm"],
        ["Effort maximal sur un boulon du groupe S", 66.69921080659218, 66.6992108065922, va("F_S"), "kN"],
        ["Résistance d'un boulon du groupe S (2 plans)", 188.16, 188.16, rd("bv_S"), "kN"],
        ["Taux – pression diamétrale de l'âme (interaction)", 0.739914371796275, 0.739914371796275, lambda R: R.ck["pdS"].eta, "-"],
        ["Aire du Té grugé ATee", 6121, 6121, va("A_Tee"), "mm²"],
        ["Aire de cisaillement de la section grugée Av,N", 2358.5, 2358.5, va("Av_S"), "mm²"],
        ["Module élastique du Té Wel,N", 139377.42591522375, 139377.425915224, va("W_N"), "mm³"],
        ["Moment résistant au grugeage Mv,N,Rd", 49.47898619990443, 49.4789861999044, rd("mN"), "kNm"],
        ["Sollicitation au grugeage VEd·(gh + ln)", 20, 20, lambda R: R.ck["mN"].Ed, "kNm"],
        ["Longueur de grugeage maximale (stabilité locale)", 290, 290, rd("stN"), "mm"],
        ["Âme porteuse – aire de cisaillement locale Av", 2906.75, 2906.75, va("Av_P"), "mm²"],
        ["Âme porteuse – résistance au cisaillement local", 595.7655777132654, 595.765577713265, rd("vlP"), "kN"],
        ["Ailes B – moment résistant élastique dans leur plan", 21.359166666666663, 21.3591666666667, rd("flB"), "kNm"]]),
    dict(id="VAL-B", titre="Cas par défaut modifié : ailes B soudées (a = 5 mm, retours 40 mm), NEd = 50 kN, HEd = 20 kN, boulons 10.9 catégorie C",
         inp=dict(fix_S="Soudée", N_Ed=50, H_Ed=20, cat="C", classe="10.9", a_S=5, lh_S=40), rows=[
        ["Cordons S – longueur totale par cornière", 270, 270, va("Lw_S"), "mm"],
        ["Cordons S – position du centre de gravité", 5.925925925925926, 5.92592592592593, va("xg_S"), "mm"],
        ["Cordons S – inertie polaire (par unité de gorge), intégration numérique", 1326768.5162685185, 1326768.51851852, va("Iw_S"), "mm³"],
        ["Cordons S – effort par unité de longueur au point critique", 640.364181825083, 640.364181099529, lambda R: R.ck["wS"].Ed, "N/mm"],
        ["Cordons S – résistance Fw,Rd = fvw,d·a", 1308.6606101631517, 1308.66061016315, rd("wS"), "N/mm"],
        ["Boulons P – effort de cisaillement par boulon", 21.098314835286935, 21.0983148352869, va("F_P"), "kN"],
        ["Boulons P – traction par boulon (NEd + couple HEd·z/p3)", 13.625826952127937, 13.6258269521279, va("Ft_P"), "kN"],
        ["Boulons P – interaction cisaillement + traction", 0.2704631532922377, 0.270463153292238, lambda R: R.ck["bi_P"].Ed, "-"],
        ["Boulons P – résistance au glissement (cat. C)", 38.543841225191436, 38.5438412251914, rd("gl_P"), "kN"]]),
]


def run_bench():
    """Rejoue les 4 exemples publiés et les 2 calculs manuels ; retourne
    ``bench`` (lignes ``page lab ref val unit ec st exp``), ``valid`` (lignes
    ``lab hand xls val unit ec st``), ``worst``, ``ok`` et ``statut``
    (« VALIDÉ » / « À CONTRÔLER »)."""
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

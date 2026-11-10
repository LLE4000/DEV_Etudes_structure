# -*- coding: utf-8 -*-
"""Le moteur de la BIBLIOTHÈQUE BÉTON ARMÉ (beton/bibliotheque, et les déformations ajoutées à beton/donnees/ec2.py) :
chaque grandeur est CALCULÉE depuis l'Eurocode 2 et confrontée à une source PUBLIÉE transcrite ci-dessous.

1. Tableau 3.1 (EN 1992-1-1) : les déformations ε_c1, ε_cu1, ε_c2, ε_cu2, n, ε_c3, ε_cu3 des relations analytiques
   redonnent les valeurs imprimées de C12/15 à C90/105 ; f_cd = α_cc f_ck / γ_c.
2. Ancrages et recouvrements (§8.4, §8.7.3) : la chaîne f_ctk,0,05 → f_ctd → f_bd → l_b,rqd → l_bd, l_0 vérifiée à la
   main sur un cas, puis les 77 cases du tableau 24 (l_bd) et les 77 du tableau 25 (l_0) de la NIT 260 (CSTC, 2017),
   transcrites des extraits transmis le 08/11/2026 : écart borné à max(20 mm ; 2 %) jusqu'à Ø 32 et 3 % au-delà
   (la NIT arrondit au plus près et se dit approximative ; le moteur arrondit au centimètre supérieur). Les règles
   de bord : minimums, compression, α2 borné, produit ≥ 0,7, η2, plafond C60/75, α6 du tableau 8.3, refus.
3. Enrobage, durabilité (§4.4.1) : le tableau 4.4N (= NIT 260 tableau 20, ligne S4 transcrite), les modifications du
   tableau 4.3N (= NIT 260 tableau 21) dont l'exemple de la NIT (p. 49 : dalle C30/37, XC1, Ø 12 → S2, c_nom 22 mm), les
   bornes S1/S6, la classe imposée, le coulage contre le sol, refus.
3 bis. FEU (beton/bibliotheque/feu.py, EN 1992-1-2:2004 section 5 — lot Q du 10/11/2026) : CHAQUE CASE des sept tableaux
   (5.2a, 5.3, 5.4, 5.5, 5.6, 5.8, 5.9) relue contre une transcription INDÉPENDANTE écrite ici d'après le § 4.5 de la
   conception (deux copies nommées, liées par ce test — et liées au document lui-même) ; les repères « * » et « ** » ;
   la monotonie ; les cas d'hier (D1 : poteau R 90 μ_fi 0,7 → 350/53 ; D2 : voile REI 180 → 210/50 ; D6 : aucun étrier
   retiré d'une dalle) ; a_sd (§5.6.1(8) : colonne 4 du tableau 5.5, colonne 3 du tableau 5.6) ; « non atteinte » ; la
   formule (5.7) recalculée à la main avec ``math`` seul (quatre contrôles, un exemple publié, l'aller-retour a → durée
   → a, les monotonies, chaque borne du domaine refusée, « jamais par défaut »), et son RECOUPEMENT avec le tableau
   5.2a (α_cc = 1,0) ; l'enrobage que le feu impose (a nominale : sans Δc_dev) ; la feuille de calcul recalculable.
4. Aciers : les exigences de l'annexe C (classes A, B, C), les six nuances des modules.
5. Mutations : abîmer un coefficient, une formule ou une case fait rougir le contrôle qui le garde — dont les défauts
   d'hier rejoués (une case changée, la colonne μ_fi décalée, a_sd oublié ou mal borné, l'étrier retiré d'une dalle…).

    python3 tests/test_bibliotheque_moteur.py
"""
import inspect
import json
import math
import os
import re
import sys
import textwrap

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton.bibliotheque import aciers as S  # noqa: E402
from beton.bibliotheque import ancrages as A  # noqa: E402
from beton.bibliotheque import enrobage as E  # noqa: E402
from beton.bibliotheque import feu as F  # noqa: E402
from beton.donnees import ec2 as EC  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def refus(f, mot):
    """f() doit lever ValueError dont le message (français) contient ``mot``."""
    try:
        f()
    except ValueError as e:
        return mot in str(e), str(e)
    return False, "aucun refus"


def lignes(texte: str) -> dict:
    return {li.split(";")[0]: [float(x.replace(",", ".")) for x in li.split(";")[1:]] for li in texte.strip().splitlines()}


# ============================================================================================ 1. tableau 3.1
print("=== 1 · Tableau 3.1 : déformations des relations analytiques contre les valeurs imprimées")
# EN 1992-1-1:2004, tableau 3.1, lignes des déformations (‰) et de n, telles qu'imprimées, de C12/15 à C90/105
TABLEAU_3_1_DEF = lignes("""
fck;12;16;20;25;30;35;40;45;50;55;60;70;80;90
eps_c1;1,8;1,9;2,0;2,1;2,2;2,25;2,3;2,4;2,45;2,5;2,6;2,7;2,8;2,8
eps_cu1;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,2;3,0;2,8;2,8;2,8
eps_c2;2,0;2,0;2,0;2,0;2,0;2,0;2,0;2,0;2,0;2,2;2,3;2,4;2,5;2,6
eps_cu2;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,1;2,9;2,7;2,6;2,6
n;2,0;2,0;2,0;2,0;2,0;2,0;2,0;2,0;2,0;1,75;1,6;1,45;1,4;1,4
eps_c3;1,75;1,75;1,75;1,75;1,75;1,75;1,75;1,75;1,75;1,8;1,9;2,0;2,2;2,3
eps_cu3;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,5;3,1;2,9;2,7;2,6;2,6""")
FONCTIONS_3_1 = {"eps_c1": "eps_c1", "eps_cu1": "eps_cu1", "eps_c2": "eps_c2", "eps_cu2": "eps_cu2", "n": "n_parabole",
                 "eps_c3": "eps_c3", "eps_cu3": "eps_cu3"}
TOLERANCE_3_1 = 0.051   # ‰ : les valeurs imprimées ont une ou deux décimales (2,25 ; 1,75) ; le pire écart est 0,043 (ε_cu1 C70)


def ecarts_3_1(ec) -> list:
    out = []
    for ligne, fn in FONCTIONS_3_1.items():
        for fck, attendu in zip(TABLEAU_3_1_DEF["fck"], TABLEAU_3_1_DEF[ligne]):
            v = getattr(ec, fn)(fck)
            if abs(v - attendu) > TOLERANCE_3_1:
                out.append((ligne, int(fck), round(v, 3), attendu))
    return out


ec_3_1 = ecarts_3_1(EC)
chk("les sept lignes de déformations redonnent le tableau 3.1 (98 valeurs, |Δ| ≤ 0,051 ‰)", not ec_3_1, ec_3_1)
pire = max(abs(getattr(EC, fn)(fck) - att) for li, fn in FONCTIONS_3_1.items() for fck, att in zip(TABLEAU_3_1_DEF["fck"], TABLEAU_3_1_DEF[li]))
print(f"     pire écart aux valeurs imprimées : {pire:.3f} ‰")
chk("ε_cu3 = ε_cu2 (même relation, tableau 3.1)", all(EC.eps_cu3(f) == EC.eps_cu2(f) for f in (30, 55, 70, 90)))
chk("f_cd = α_cc f_ck / γ_c : C30/37 → 20,0 MPa (α_cc 1,0, γ_c 1,5) et 17,0 MPa avec α_cc = 0,85",
    abs(EC.fcd(30) - 20.0) < 1e-9 and abs(EC.fcd(30, alpha_cc=0.85) - 17.0) < 1e-9)
chk("γ_c = 1,5 et α_cc recommandé = 1,0 sont nommés dans ec2", EC.GAMMA_C == 1.5 and EC.ALPHA_CC_RECOMMANDE == 1.0)

# ============================================================================================ 2. ancrages
print("\n=== 2 · Ancrages et recouvrements : la chaîne à la main, puis la NIT 260 (tableaux 24 et 25)")
r = A.calculer({"phi": 16, "classe": "C30/37"})
ad = r["adherence"]
fctm = 0.30 * 30 ** (2 / 3)
chk("C30/37 : f_ctk,0,05 = 0,7 · 0,30 · 30^(2/3) = 2,028 MPa", abs(ad["fctk_005"] - 0.7 * fctm) < 1e-9 and abs(ad["fctk_005"] - 2.028) < 0.001)
chk("f_ctd = 1,0 · f_ctk,0,05 / 1,5 = 1,352 MPa", abs(ad["fctd"] - 0.7 * fctm / 1.5) < 1e-9)
chk("f_bd = 2,25 · 1,0 · 1,0 · f_ctd = 3,042 MPa", abs(ad["fbd"] - 2.25 * 0.7 * fctm / 1.5) < 1e-9 and abs(ad["fbd"] - 3.042) < 0.001)
chk("f_yd = 500 / 1,15 = 434,8 MPa, σ_sd = f_yd (barre pleinement sollicitée)", abs(r["fyd"] - 500 / 1.15) < 1e-9 and r["sigma_sd"] == r["fyd"])
lb = 16 / 4 * (500 / 1.15) / (2.25 * 0.7 * fctm / 1.5)
chk("l_b,rqd = (Ø/4) σ_sd / f_bd = 571,8 mm", abs(r["lb_rqd"] - lb) < 1e-9 and abs(lb - 571.8) < 0.1)
chk("l_bd = l_b,rqd (α = 1, le minimum ne gouverne pas) et l_0 = 1,5 · l_b,rqd = 857,8 mm (ρ1 = 100 % → α6 = 1,5)",
    abs(r["lbd"]["valeur"] - lb) < 1e-9 and not r["lbd"]["gouverne_min"] and abs(r["l0"]["valeur"] - 1.5 * lb) < 1e-9
    and r["l0"]["alpha6"] == 1.5)
chk("l_b,min = max(0,3 l_b,rqd ; 10 Ø ; 100) = 171,5 mm ; l_0,min = max(0,3 α6 l_b,rqd ; 15 Ø ; 200) = 257,3 mm",
    abs(r["lbd"]["min"] - 0.3 * lb) < 1e-9 and abs(r["l0"]["min"] - 0.3 * 1.5 * lb) < 1e-9)
cles = [t["cle"] for t in r["termes"]]
chk("les termes de la justification sont dans l'ordre du calcul (f_ctk → f_bd → σ_sd → l_b,rqd → α → l_bd → α6 → l_0)",
    [cles.index(c) for c in ("fctk", "fbd", "sigma", "lb_rqd", "alpha1", "lbd", "alpha6", "l0")] == sorted(cles.index(c) for c in ("fctk", "fbd", "sigma", "lb_rqd", "alpha1", "lbd", "alpha6", "l0")))
chk("chaque terme porte symbole, nom, valeur et une règle qui cite son paragraphe, son tableau ou son équation",
    all(t["symbole"] and t["nom"] and t["valeur"] is not None and ("§" in t["regle"] or "tableau" in t["regle"] or "éq." in t["regle"]) for t in r["termes"]))
chk("aucun avertissement pour une barre droite tendue de Ø 16 en bonnes conditions", r["avertissements"] == [])

# NIT 260 (CSTC, mars 2017), tableau 24 : longueur d'ancrage l_bd (mm) ; tableau 25 : longueur de recouvrement l_0 (mm)
# — B500, bonnes conditions d'adhérence, barre droite, σ_sd = f_yd, plus de 50 % de barres recouvertes (α6 = 1,5) ;
# lignes Ø 6, 8, 10, 12, 14, 16, 20, 25, 32, 40, 50 ; transcrits des extraits transmis le 08/11/2026.
NIT_24 = lignes("""
C20/25;280;380;470;570;660;760;950;1180;1510;2060;2880
C25/30;240;320;400;480;560;640;810;1010;1290;1750;2460
C30/37;220;290;360;440;510;580;730;910;1160;1580;2210
C35/45;190;260;320;380;450;510;640;800;1020;1390;1950
C40/50;180;240;290;350;410;470;590;730;940;1280;1790
C45/55;160;220;270;330;380;440;540;680;870;1180;1660
C50/60;150;200;250;300;350;400;510;630;810;1100;1540""")
NIT_25 = lignes("""
C20/25;430;570;710;850;990;1130;1420;1770;2270;3080;4320
C25/30;360;480;600;730;850;970;1210;1510;1930;2630;3680
C30/37;330;440;540;650;760;870;1090;1360;1740;2360;3320
C35/45;290;380;480;580;670;770;960;1200;1540;2090;2930
C40/50;260;350;440;530;620;710;880;1100;1410;1920;2690
C45/55;240;330;410;490;570;650;820;1020;1310;1770;2490
C50/60;230;300;380;460;530;610;760;950;1210;1650;2310""")
assert set(NIT_24) == set(A.CLASSES) == set(NIT_25) and all(len(v) == len(A.DIAMETRES) for v in list(NIT_24.values()) + list(NIT_25.values()))


def tolerance(phi: float, nit: float) -> float:
    """L'écart admis entre la case du moteur (cm supérieur) et la NIT : max(20 mm ; 2 %) jusqu'à Ø 32, 3 % au-delà."""
    return max(20.0, 0.02 * nit) if phi <= 32 else 0.03 * nit


ECART_LONGUEUR = (0.015, 5.0)   # la longueur calculée s'écarte de la NIT d'au plus 1,5 % + 5 mm (l'arrondi au plus près de la NIT)


def ecarts_nit(module, grandeur: str, nit: dict) -> tuple:
    """([(classe, Ø, moteur, NIT, écart)] des cases arrondies hors tolérance ; pire écart de la case arrondie (mm) ;
    [(classe, Ø, mm, NIT)] des longueurs NON arrondies hors de ECART_LONGUEUR ; pire écart systématique en %, une fois
    l'arrondi de la NIT (5 mm) retiré."""
    t = module.tableau(grandeur)
    hors, pire_mm, hors_brut, pire_pc = [], 0.0, [], 0.0
    for cl, valeurs in nit.items():
        for phi, v in zip(A.DIAMETRES, valeurs):
            case = t["cellules"][phi][cl]
            d, brut = case["cm"] - v, case["mm"] - v
            pire_mm = max(pire_mm, abs(d))
            pire_pc = max(pire_pc, (abs(brut) - ECART_LONGUEUR[1]) / v * 100)
            if abs(d) > tolerance(phi, v):
                hors.append((cl, phi, case["cm"], v, d))
            if abs(brut) > ECART_LONGUEUR[0] * v + ECART_LONGUEUR[1]:
                hors_brut.append((cl, phi, round(case["mm"], 1), v))
    return hors, pire_mm, hors_brut, pire_pc


for grandeur, nit, nom in (("lbd", NIT_24, "tableau 24 (l_bd)"), ("l0", NIT_25, "tableau 25 (l_0, α6 = 1,5)")):
    hors, pire_mm, hors_brut, pire_pc = ecarts_nit(A, grandeur, nit)
    chk(f"NIT 260 {nom} : les 77 cases du moteur sont dans la tolérance (max(2 cm ; 2 %) jusqu'à Ø 32, 3 % au-delà)", not hors, hors[:6])
    print(f"     pire écart au {nom} : {pire_mm:.0f} mm sur la case arrondie ; {pire_pc:.2f} % sur la longueur calculée, arrondi de la NIT retiré")
    chk(f"NIT 260 {nom} : l'écart mesuré le 08/11/2026 tient — case arrondie ≤ 50 mm, longueur calculée à 1,5 % + 5 mm de la NIT sur les 77 cases",
        pire_mm <= 50 and not hors_brut, (pire_mm, hors_brut[:4]))
t24 = A.tableau("lbd")
chk("le tableau porte Ø × classe, les coefficients l/Ø par classe (C30/37 : 35,7 Ø ≈ les « 36 Ø » du bureau), la source et le contrôle",
    t24["diametres"] == list(A.DIAMETRES) and t24["classes"] == list(A.CLASSES) and abs(t24["coefficients"]["C30/37"] - 35.74) < 0.01
    and "EN 1992-1-1" in t24["source"] and "NIT 260" in t24["controle"])
chk("les cases croissent avec Ø et décroissent avec la classe de béton",
    all(t24["cellules"][a][c]["mm"] < t24["cellules"][b][c]["mm"] for a, b in zip(A.DIAMETRES, A.DIAMETRES[1:]) for c in A.CLASSES)
    and all(t24["cellules"][p][a]["mm"] > t24["cellules"][p][b]["mm"] for a, b in zip(A.CLASSES, A.CLASSES[1:]) for p in A.DIAMETRES))
chk("conditions médiocres : l_bd × 1/0,7 (η1)", abs(A.tableau("lbd", conditions="mediocres")["cellules"][16]["C30/37"]["mm"] / t24["cellules"][16]["C30/37"]["mm"] - 1 / 0.7) < 1e-9)
chk("B400 : l_bd × 400/500", abs(A.tableau("lbd", fyk=400)["cellules"][16]["C30/37"]["mm"] / t24["cellules"][16]["C30/37"]["mm"] - 0.8) < 1e-9)
chk("arrondi au centimètre supérieur : 434,8 → 440 ; 480,0 → 480 ; 571,8 → 580", (A.arrondi_cm(434.8), A.arrondi_cm(480.0), A.arrondi_cm(571.8)) == (440, 480, 580))

# règles de bord
r40 = A.calculer({"phi": 40, "classe": "C30/37"})
chk("Ø 40 : η2 = (132 − 40)/100 = 0,92, et l'avertissement « gros diamètre » (§8.8)",
    abs(r40["adherence"]["eta2"] - 0.92) < 1e-9 and any(a["code"] == "gros_diametre" for a in r40["avertissements"]))
chk("Ø 32 : η2 = 1,0 et aucun avertissement de gros diamètre", A.calculer({"phi": 32, "classe": "C30/37"})["adherence"]["eta2"] == 1.0
    and not any(a["code"] == "gros_diametre" for a in A.calculer({"phi": 32, "classe": "C30/37"})["avertissements"]))
chk("C70/85 : f_ctk,0,05 plafonné à C60/75 pour f_bd (§8.4.2(2)) — même f_bd que C60/75, et le terme le dit",
    abs(A.calculer({"phi": 16, "classe": "C70/85"})["adherence"]["fbd"] - A.calculer({"phi": 16, "classe": "C60/75"})["adherence"]["fbd"]) < 1e-12
    and "limité à C60/75" in next(t["regle"] for t in A.calculer({"phi": 16, "classe": "C70/85"})["termes"] if t["cle"] == "fctk"))
rc = A.calculer({"phi": 16, "classe": "C30/37", "sollicitation": "compression", "forme": "courbe", "c_d": 100, "K": 0.1, "lam": 1, "p": 5})
chk("compression : α1 = α2 = α3 = α5 = 1,0 quelles que soient la forme, c_d, K, p ; l_b,min = 0,6 l_b,rqd",
    all(rc["alphas"][k] == 1.0 for k in ("alpha1", "alpha2", "alpha3", "alpha5")) and abs(rc["lbd"]["min"] - 0.6 * rc["lb_rqd"]) < 1e-9)
rt = A.calculer({"phi": 16, "classe": "C30/37", "taux": 0.2})
chk("barre peu sollicitée (σ_sd = 0,2 f_yd) : l_b,min = 10 Ø = 160 mm gouverne, avertissement « lbd_min »",
    rt["lbd"]["gouverne_min"] and rt["lbd"]["valeur"] == 160.0 and any(a["code"] == "lbd_min" for a in rt["avertissements"]))
rt6 = A.calculer({"phi": 6, "classe": "C50/60", "taux": 0.1})
chk("Ø 6 peu sollicité : le plancher de 100 mm gouverne l_bd, 200 mm gouverne l_0", rt6["lbd"]["valeur"] == 100.0 and rt6["l0"]["valeur"] == 200.0)
rh = A.calculer({"phi": 16, "classe": "C30/37", "forme": "courbe", "c_d": 60})
chk("crochet tendu avec c_d = 60 > 3 Ø : α1 = 0,7 ; α2 = 1 − 0,15 (c_d − 3 Ø)/Ø = 0,8875", rh["alphas"]["alpha1"] == 0.7 and abs(rh["alphas"]["alpha2"] - 0.8875) < 1e-9)
rd = A.calculer({"phi": 16, "classe": "C30/37", "forme": "droit", "c_d": 60})
chk("barre droite avec c_d = 60 : α1 = 1,0 ; α2 = 1 − 0,15 (c_d − Ø)/Ø = 0,5875 borné à 0,7", rd["alphas"]["alpha1"] == 1.0 and rd["alphas"]["alpha2"] == 0.7)
rd2 = A.calculer({"phi": 16, "classe": "C30/37", "forme": "droit", "c_d": 10})
chk("c_d < Ø : α2 = 1 − 0,15 (c_d − Ø)/Ø > 1 borné à 1,0", rd2["alphas"]["alpha2"] == 1.0)
rp = A.calculer({"phi": 16, "classe": "C30/37", "c_d": 40, "K": 0.1, "lam": 3, "p": 7.5})
al = rp["alphas"]
chk("α2 = 0,775, α3 = 1 − 0,1 · 3 = 0,7, α5 = 1 − 0,04 · 7,5 = 0,7 : produit 0,38 borné à 0,7 (éq. 8.4) et le terme « produit borné » apparaît",
    abs(al["alpha2"] - 0.775) < 1e-9 and al["alpha3"] == 0.7 and al["alpha5"] == 0.7 and al["produit_235"] == 0.7
    and any(t["cle"] == "prod" for t in rp["termes"]) and abs(rp["lbd"]["brut"] - 0.7 * rp["lb_rqd"]) < 1e-9)
chk("α4 = 0,7 avec une barre transversale soudée, et il ne compte pas dans l_0 (éq. 8.10 : α1 α2 α3 α5 α6)",
    A.calculer({"phi": 16, "classe": "C30/37", "soude": True})["alphas"]["alpha4"] == 0.7
    and abs(A.calculer({"phi": 16, "classe": "C30/37", "soude": True})["l0"]["valeur"] - r["l0"]["valeur"]) < 1e-9
    and abs(A.calculer({"phi": 16, "classe": "C30/37", "soude": True})["lbd"]["valeur"] - 0.7 * lb) < 1e-9)
chk("α6 du tableau 8.3 : < 25 % → 1,0 ; 33 % → 1,15 ; 50 % → 1,41 ; > 50 % → 1,5 (borné)",
    A.alpha6(20) == 1.0 and abs(A.alpha6(33) - 1.1489) < 0.001 and abs(A.alpha6(50) - 1.4142) < 0.001 and A.alpha6(60) == 1.5 and A.alpha6(100) == 1.5)
chk("crochet sans c_d : aucune réduction, et l'avertissement « cd_inconnu »", A.calculer({"phi": 16, "classe": "C30/37", "forme": "courbe"})["alphas"]["alpha1"] == 1.0
    and any(a["code"] == "cd_inconnu" for a in A.calculer({"phi": 16, "classe": "C30/37", "forme": "courbe"})["avertissements"]))
chk("conditions médiocres : l'avertissement rappelle la figure 8.2", any("figure 8.2" in a["message"] for a in A.calculer({"phi": 16, "classe": "C30/37", "conditions": "mediocres"})["avertissements"]))
chk("les paramètres se relisent en chaînes avec virgule (« 0,5 »)", abs(A.calculer({"phi": "16", "classe": "C30/37", "taux": "0,5"})["sigma_sd"] - 0.5 * 500 / 1.15) < 1e-9)
for nom, params, mot in (("classe illisible", {"classe": "C30"}, "illisible"), ("classe hors tableau", {"classe": "C95/110"}, "hors du tableau"),
                         ("Ø 60", {"phi": 60}, "au plus 50"), ("Ø 4", {"phi": 4}, "au moins 6"), ("f_yk 450", {"fyk": 450}, "n'est pas permis"),
                         ("conditions « moyennes »", {"conditions": "moyennes"}, "n'est pas permis"), ("taux 1,2", {"taux": 1.2}, "au plus 1"),
                         ("K 0,2", {"K": 0.2}, "au plus 0,1"), ("Ø booléen", {"phi": True}, "nombre est attendu"), ("ρ1 négatif", {"rho1": -1}, "au moins 0")):
    ok, msg = refus(lambda p=params: A.calculer(p), mot)
    chk(f"refus « {nom} » en français", ok, msg)
ok, msg = refus(lambda: A.tableau("lb_rqd"), "lbd")
chk("refus d'une grandeur de tableau inconnue", ok, msg)

# ============================================================================================ 3. enrobage
print("\n=== 3 · Enrobage, durabilité : tableau 4.4N (NIT 260 T20), modifications 4.3N (T21), exemple de la NIT")
# NIT 260, tableau 20 (= EN 1992-1-1 tableau 4.4N, ligne S4), transcrit de l'extrait : X0 10, XC1 15, XC2/XC3 25, XC4 30,
# XD1/XS1 35, XD2/XS2 40, XD3/XS3 45 (mm), avec la correspondance belge EI / EE1-EE2 / EE3 / ES2 / ES1-ES3 / EE4-ES4
NIT_20_S4 = (10, 15, 25, 30, 35, 40, 45)
NIT_20_BELGES = ("—", "EI", "EE1, EE2", "EE3", "ES2", "ES1, ES3", "EE4, ES4")
chk("tableau 4.4N, ligne S4 = NIT 260 tableau 20", E.C_MIN_DUR["S4"] == NIT_20_S4)
chk("les libellés belges (NBN B 15-001) du tableau 20 sont en regard des colonnes", tuple(E.LIBELLES_BELGES[c] for c in E.COLONNES_4_4N) == NIT_20_BELGES)
chk("tableau 4.4N : six classes structurales, sept colonnes, chaque ligne croît avec l'exposition, S3 → S6 par pas de 5 mm "
    "(hors X0, qui reste à 10 jusqu'à S4), chaque colonne croît avec la classe structurale",
    len(E.C_MIN_DUR) == 6 and all(len(v) == 7 for v in E.C_MIN_DUR.values())
    and all(v[i] <= v[i + 1] for v in E.C_MIN_DUR.values() for i in range(6))
    and all(E.C_MIN_DUR[f"S{n + 1}"][i] - E.C_MIN_DUR[f"S{n}"][i] == 5 for n in (3, 4, 5) for i in range(1, 7))
    and tuple(E.C_MIN_DUR[s][0] for s in E.CLASSES_STRUCTURALES) == (10, 10, 10, 10, 15, 20)
    and all(E.C_MIN_DUR[a][i] <= E.C_MIN_DUR[b][i] for a, b in zip(E.CLASSES_STRUCTURALES, E.CLASSES_STRUCTURALES[1:]) for i in range(7)))
chk("chaque classe d'exposition portée a sa colonne, son libellé belge et son seuil du tableau 4.3N",
    all(e[1] in E.COLONNES_4_4N and e[2] == E.LIBELLES_BELGES[e[1]] and e[3].startswith("C") for e in E.EXPOSITIONS))

# l'exemple de la NIT 260 (p. 49) : dalle, C30/37, XC1, treillis Ø 12 → S4 − 1 (dalle) − 1 (C30/37 ≥ C30/37 pour XC1) = S2 ;
# c_min,dur = 10 ; c_min = max(12 ; 10 ; 10) = 12 ; c_nom = 12 + 10 = 22 mm
ex = E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12, "phi_etrier": 0})
s = ex["durabilite"]["classe_structurale"]
chk("exemple NIT 260 p. 49 : S2 (dalle −1, classe de résistance −1), c_min,dur 10, c_min 12 (c_min,b), c_nom 22 mm",
    s["retenue"] == "S2" and sorted(m["code"] for m in s["modifications"]) == ["dalle", "resistance"] and ex["durabilite"]["c_min_dur"] == 10
    and ex["durabilite"]["c_min"] == 12 and ex["durabilite"]["terme_c_min"] == "c_min,b" and ex["c_nom"] == 22 and ex["gouvernant"] == "durabilite")
chk("… et le terme S cite le tableau 4.3N et la NIT 260 (tableau 21)", "4.3N" in next(t["regle"] for t in ex["termes"] if t["cle"] == "S") and "NIT 260" in next(t["regle"] for t in ex["termes"] if t["cle"] == "S"))
# tableau 21 (4.3N) : la classe de résistance seuil dépend de l'exposition
p4 = E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": 20, "phi_etrier": 8})
chk("poutre XC4 C30/37 (< C40/50) : S4, c_min,dur 30, c_min 30, c_nom 40 mm", p4["durabilite"]["classe_structurale"]["retenue"] == "S4"
    and p4["durabilite"]["c_min_dur"] == 30 and p4["c_nom"] == 40)
p4b = E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C40/50", "phi": 20, "phi_etrier": 8})
chk("poutre XC4 C40/50 (seuil atteint) : S3, c_min,dur 25, c_nom 35 mm", p4b["durabilite"]["classe_structurale"]["retenue"] == "S3" and p4b["c_nom"] == 35)
chk("XC1 : seuil C30/37 ; XC2/XC3 : C35/45 ; XC4 et XD1/XS1 : C40/50 ; XD3/XS3 : C45/55 (tableau 4.3N)",
    tuple(dict((e[0], e[3]) for e in E.EXPOSITIONS)[k] for k in ("XC1", "XC2", "XC3", "XC4", "XD1", "XS1", "XD3", "XS3")) == ("C30/37", "C35/45", "C35/45", "C40/50", "C40/50", "C40/50", "C45/55", "C45/55"))
p100 = E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "duree": 100, "phi": 20})
chk("durée d'utilisation 100 ans : +2 → S6, c_min,dur 40", p100["durabilite"]["classe_structurale"]["retenue"] == "S6" and p100["durabilite"]["c_min_dur"] == 40)
p100b = E.calculer({"element": "poutre", "exposition": "XD3", "classe": "C30/37", "duree": 100, "phi": 20, "qualite": False})
chk("100 ans en XD3 : S6 (l'extrême haut du tableau 4.3N), c_min,dur 55", p100b["durabilite"]["classe_structurale"]["retenue"] == "S6" and p100b["durabilite"]["c_min_dur"] == 55)
pq = E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C40/50", "qualite": True, "phi": 10})
chk("dalle, C40/50 en XC1, maîtrise de la qualité : S4 − 3 = S1 (l'extrême bas du tableau 4.3N), c_min,dur 10",
    pq["durabilite"]["classe_structurale"]["retenue"] == "S1" and pq["durabilite"]["c_min_dur"] == 10)
pb = E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C40/50", "qualite": True, "duree": 100, "phi": 10})
chk("… avec 100 ans : S3 (4 + 2 − 3)", pb["durabilite"]["classe_structurale"]["retenue"] == "S3")
chk("les modifications du tableau 4.3N vont de −3 à +2 : la classe calculée est toujours S1 à S6 (sur les 2 × 2 × 2 × 11 × 14 combinaisons)",
    all(E.classe_structurale({"classe": f"C{f}/{f + 7}", "exposition": x, "duree": d, "dalle": da, "qualite": q, "classe_structurale": None})["calculee"] in E.CLASSES_STRUCTURALES
        for f in (12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90) for x in E.EXPO_CLES for d in (50, 100) for da in (False, True) for q in (False, True)))
pi = E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "classe_structurale": "S2", "phi": 20})
chk("classe structurale imposée S2 : retenue S2 (calculée S4 citée), c_min,dur 20", pi["durabilite"]["classe_structurale"]["retenue"] == "S2"
    and pi["durabilite"]["classe_structurale"]["imposee"] and pi["durabilite"]["classe_structurale"]["calculee"] == "S4" and pi["durabilite"]["c_min_dur"] == 20
    and "imposée" in next(t["regle"] for t in pi["termes"] if t["cle"] == "S"))
pg = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 32, "dg": 40})
chk("Ø 32 avec granulat 40 mm : c_min,b = 32 + 5 = 37 gouverne, c_nom 47", pg["durabilite"]["c_min_b"] == 37 and pg["durabilite"]["majoration_granulat"] and pg["c_nom"] == 47)
pd5 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20, "delta_dev": 5})
chk("Δc_dev = 5 mm (assurance qualité) : c_nom = 20 + 5 = 25", pd5["c_nom"] == 25 and pd5["durabilite"]["delta_dev"] == 5)
ps = E.calculer({"element": "poutre", "exposition": "XC2", "classe": "C30/37", "phi": 20, "contre_sol": "direct"})
chk("coulé directement contre le sol : c_nom ≥ 75 mm (k2) gouverne", ps["c_nom"] == 75 and ps["gouvernant"] == "sol" and any(t["cle"] == "sol" for t in ps["termes"]))
ps2 = E.calculer({"element": "poutre", "exposition": "XC2", "classe": "C30/37", "phi": 20, "contre_sol": "prepare"})
chk("sur béton de propreté : c_nom ≥ 40 mm (k1) ; XC2 C30/37 Ø 20 donne 35 → 40 gouverne", ps2["c_nom_dur"] == 40 and ps2["gouvernant"] == "sol")
chk("X0 : avertissement « béton non armé ou ambiance très sèche »", any(a["code"] == "x0" for a in E.calculer({"element": "poutre", "exposition": "X0", "classe": "C30/37"})["avertissements"]))

for nom, params, mot in (("exposition XF1", {"exposition": "XF1"}, "n'est pas une classe"), ("élément « semelle »", {"element": "semelle"}, "élément"),
                         ("durée 75 ans", {"duree": 75}, "50 ou 100"), ("Δc_dev 7", {"delta_dev": 7}, "10, 5 ou 0"), ("classe structurale S7", {"classe_structurale": "S7"}, "S1 à S6"),
                         ("feu R45", {"feu": {"r": "R45"}}, "résistance au feu"), ("Ø 3", {"phi": 3}, "entre 6 et 60"), ("contre sol « oui »", {"contre_sol": "oui"}, "coulé contre le sol"),
                         ("classe C30", {"classe": "C30"}, "illisible")):
    ok, msg = refus(lambda p=params: E.calculer(p), mot)
    chk(f"refus « {nom} » en français", ok, msg)

# ============================================================================================ 3 bis. feu
print("\n=== 3 bis · Feu : les tableaux de l'EN 1992-1-2:2004 section 5, case par case, contre une transcription indépendante")
# La SECONDE copie des tableaux du feu (R2) : recopiée ligne à ligne du § 4.5 de docs/site/LOT_Q_QUATRE_MODULES.md (la
# conception du lot Q, d'après l'EN 1992-1-2:2004 section 5), dans SON écriture — « dimension minimale/a », « ** » :
# au moins 8 barres, « — » : pas de valeur. La première copie est beton/bibliotheque/feu.py ; ce test les lie, case par
# case ; plus bas, il lie aussi cette copie au document lui-même.
CONCEPTION = {
    "5.2a": """
        R 30  | 200/25          | 200/25              | 200/32 · 300/27     | 155/25
        R 60  | 200/25          | 200/36 · 300/31     | 250/46 · 350/40     | 155/25
        R 90  | 200/31 · 300/25 | 300/45 · 400/38     | 350/53 · 450/40**   | 155/25
        R 120 | 250/40 · 350/35 | 350/45** · 450/40** | 350/57** · 450/51** | 175/35
        R 180 | 350/45**        | 350/63**            | 450/70**            | 230/55
        R 240 | 350/61**        | 450/75**            | —                   | 295/70""",
    "5.4": """
        REI 30  | 100/10 | 120/10 | 120/10 | 120/10
        REI 60  | 110/10 | 120/10 | 130/10 | 140/10
        REI 90  | 120/20 | 140/10 | 140/25 | 170/25
        REI 120 | 150/25 | 160/25 | 160/35 | 220/35
        REI 180 | 180/40 | 200/45 | 210/50 | 270/55
        REI 240 | 230/55 | 250/55 | 270/60 | 350/60""",
    "5.3": "EI 30 : 60 · EI 60 : 80 · EI 90 : 100 · EI 120 : 120 · EI 180 : 150 · EI 240 : 175",
    "5.5": """
        R 30  | 80/25  | 120/20 | 160/15 | 200/15
        R 60  | 120/40 | 160/35 | 200/30 | 300/25
        R 90  | 150/55 | 200/45 | 300/40 | 400/35
        R 120 | 200/65 | 240/60 | 300/55 | 500/50
        R 180 | 240/80 | 300/70 | 400/65 | 600/60
        R 240 | 280/90 | 350/80 | 500/75 | 700/70""",
    "5.6": """
        R 30  | 80/15  | 160/12 |        |
        R 60  | 120/25 | 200/12 |        |
        R 90  | 150/35 | 250/25 |        |
        R 120 | 200/45 | 300/35 | 450/35 | 500/30
        R 180 | 240/60 | 400/50 | 550/50 | 600/40
        R 240 | 280/75 | 500/60 | 650/60 | 700/50""",
    "5.8": """
        REI 30  | 60  | 10 | 10 | 10
        REI 60  | 80  | 20 | 10 | 15
        REI 90  | 100 | 30 | 15 | 20
        REI 120 | 120 | 40 | 20 | 25
        REI 180 | 150 | 55 | 30 | 40
        REI 240 | 175 | 65 | 40 | 50""",
    "5.9": "REI 30 : 150/10 · REI 60 : 180/15 · REI 90 : 200/25 · REI 120 : 200/35 · REI 180 : 200/45 · REI 240 : 200/50",
}
# le tableau que le moteur rend pour chaque numéro : (élément, paramètres de ``feu.tableau``)
VARIANTES_FEU = {"5.2a": ("poteau", {}), "5.3": ("voile", {"porteur": False}), "5.4": ("voile", {}), "5.5": ("poutre", {}),
                 "5.6": ("poutre", {"appuis": "continue"}), "5.8": ("dalle", {}), "5.9": ("dalle", {"portee": "plancher_dalle"})}
COLONNES_FEU = {"5.2a": ("mu02", "mu05", "mu07", "un_cote"), "5.3": ("ei",), "5.4": ("mu035_1", "mu035_2", "mu07_1", "mu07_2"),
                "5.5": ("c1", "c2", "c3", "c4"), "5.6": ("c1", "c2", "c3", "c4"), "5.8": ("hs", "une", "deux_15", "deux_20"), "5.9": ("hs", "a")}


def _duree_cle(libelle: str) -> str:
    """« R 90 », « REI 90 », « EI 90 » → « R90 » (la clé des durées du moteur)."""
    return "R" + libelle.split()[-1]


def _combinaisons_lues(case: str, sorte: str) -> list:
    """Une case de la conception → [(dimension, a, huit barres)] : « 350/53 · 450/40** » ; « — » ou vide : [] ;
    ``sorte`` « dimension » : un nombre seul est une dimension ; « a » : une distance à l'axe."""
    out = []
    for morceau in case.split("·"):
        m = morceau.strip()
        if not m or m == "—":
            continue
        huit = m.endswith("**")
        m = m.rstrip("*")
        if "/" in m:
            d, a = m.split("/")
            out.append((int(d), int(a), huit))
        elif sorte == "a":
            out.append((None, int(m), huit))
        else:
            out.append((int(m), None, huit))
    return out


def cases_conception(numero: str, textes: dict = None) -> dict:
    """La transcription de la conception, case par case : {durée: {colonne: [(dimension, a, huit barres)]}}."""
    texte = (textes or CONCEPTION)[numero]
    cols = COLONNES_FEU[numero]
    out = {}
    if numero in ("5.3", "5.9"):
        for morceau in texte.split("·"):
            libelle, valeur = (x.strip() for x in morceau.split(":"))
            if numero == "5.3":
                out[_duree_cle(libelle)] = {"ei": [(int(valeur), None, False)]}
            else:
                hs, a = valeur.split("/")
                out[_duree_cle(libelle)] = {"hs": [(int(hs), None, False)], "a": [(None, int(a), False)]}
        return out
    for ligne in texte.strip().splitlines():
        cases = [c.strip() for c in ligne.strip().strip("|").split("|")]
        cases += [""] * (len(cols) + 1 - len(cases))
        sortes = ("dimension", "a", "a", "a") if numero == "5.8" else ("combinaison",) * len(cols)
        out[_duree_cle(cases[0])] = {c: _combinaisons_lues(cases[1 + i], sortes[i]) for i, c in enumerate(cols)}
    return out


def cases_moteur(module, numero: str) -> dict:
    """Le même tableau tel que le MOTEUR le rend (``feu.tableau``), dans la même forme."""
    element, params = VARIANTES_FEU[numero]
    t = module.tableau(element, **params)
    assert t["numero"] == numero and tuple(c["cle"] for c in t["colonnes"]) == COLONNES_FEU[numero], (numero, t["numero"], [c["cle"] for c in t["colonnes"]])
    return {r: {c: [(x["dimension"], x["a"], x["huit_barres"]) for x in t["lignes"][r][c]] for c in COLONNES_FEU[numero]} for r in t["durees"]}


def ecarts_feu(module, textes: dict = None) -> tuple:
    """([(tableau, durée, colonne, moteur, conception)] des cases qui diffèrent, nombre de cases comparées)."""
    ecarts, n = [], 0
    for numero in CONCEPTION:
        co, mo = cases_conception(numero, textes), cases_moteur(module, numero)
        if set(co) != set(mo):
            ecarts.append((numero, "durées", sorted(co), sorted(mo)))
            continue
        for r in mo:
            for c in COLONNES_FEU[numero]:
                n += 1
                if mo[r][c] != co[r][c]:
                    ecarts.append((numero, r, c, mo[r][c], co[r][c]))
    return ecarts, n


ec_feu, n_feu = ecarts_feu(F)
chk(f"les sept tableaux du feu (5.2a, 5.3, 5.4, 5.5, 5.6, 5.8, 5.9) du moteur = la transcription de la conception, sur {n_feu} cases "
    "(dimension minimale, distance à l'axe, repère « au moins 8 barres »)", not ec_feu and n_feu == 6 * (4 + 1 + 4 + 4 + 4 + 4 + 2), ec_feu[:5])
print(f"     {n_feu} cases comparées, {sum(len(v) for num in CONCEPTION for li in cases_moteur(F, num).values() for v in li.values())} combinaisons")

# … et la transcription de ce test est bien celle du DOCUMENT de conception (troisième copie, le texte que l'ingénieur relit)
DOC_CONCEPTION = os.path.join(RACINE, "docs", "site", "LOT_Q_QUATRE_MODULES.md")


def conception_du_document() -> dict:
    """Les tableaux du § 4.5 relus dans le document : les lignes « | R 30 | … » dans l'ordre (5.2a, 5.4, 5.5, 5.6, 5.8),
    et les deux tableaux écrits en ligne (5.3, 5.9)."""
    with open(DOC_CONCEPTION, encoding="utf-8") as f:
        doc = f.read()
    section = doc[doc.index("### 4.5 Tableaux retenus"):doc.index("### 4.6 Les preuves")]
    lignes = [li.strip().strip("|") for li in section.splitlines() if re.match(r"\|\s*(R|REI) \d+\s*\|", li.strip())]
    assert len(lignes) == 30, len(lignes)
    plat = re.sub(r"\s+", " ", section)
    out = {num: "\n".join(lignes[6 * i:6 * i + 6]) for i, num in enumerate(("5.2a", "5.4", "5.5", "5.6", "5.8"))}
    out["5.3"] = re.search(r"EI 30 : 60 · EI 60.*?EI 240 : \d+", plat).group(0)
    out["5.9"] = re.search(r"REI 30 : \d+/\d+ · REI 60.*?REI 240 : \d+/\d+", plat).group(0)
    return out


if os.path.exists(DOC_CONCEPTION):
    du_doc = conception_du_document()
    diff = [num for num in CONCEPTION if cases_conception(num, du_doc) != cases_conception(num)]
    chk("la transcription de ce test = les tableaux du § 4.5 du document de conception (docs/site/LOT_Q_QUATRE_MODULES.md), case par case",
        not diff, diff)
else:
    print("     (document de conception absent : la troisième copie n'est pas relue)")

# les repères « * » (« l'enrobage de l'EN 1992-1-1 gouverne normalement ») : une note d'information, sans effet sur
# le calcul. Tableau 5.4 : ceux du premier rapport de sources (texte NF EN) ; tableau 5.8 : ceux du second (texte de
# l'EN, édition bilingue) ; tableaux 5.5, 5.6 et 5.9 : repères non recoupés — ils portent sur toutes les distances a ≤ 15 mm.
ETOILES_SOURCES = {"5.4": {("R30", "mu035_1"), ("R30", "mu035_2"), ("R30", "mu07_1"), ("R30", "mu07_2"), ("R60", "mu035_1"), ("R60", "mu035_2"),
                           ("R60", "mu07_1"), ("R60", "mu07_2"), ("R90", "mu035_1"), ("R90", "mu035_2")},
                   "5.8": {("R30", "une"), ("R30", "deux_15"), ("R30", "deux_20"), ("R60", "deux_15"), ("R60", "deux_20"), ("R90", "deux_15")}}


def etoiles(numero: str) -> set:
    element, params = VARIANTES_FEU[numero]
    t = F.tableau(element, **params)
    return {(r, c) for r in t["durees"] for c in COLONNES_FEU[numero] for x in t["lignes"][r][c] if x["etoile"]}


chk("repères « * » des tableaux 5.4 et 5.8 : ceux que les rapports de sources ont relevés", all(etoiles(n) == ETOILES_SOURCES[n] for n in ETOILES_SOURCES),
    {n: sorted(etoiles(n) ^ ETOILES_SOURCES[n]) for n in ETOILES_SOURCES})
chk("repères « * » des tableaux 5.5, 5.6 et 5.9 : exactement les distances a ≤ 15 mm ; aucun repère dans les tableaux 5.2a et 5.3",
    all(x["etoile"] == (x["a"] is not None and x["a"] <= 15) for n in ("5.5", "5.6", "5.9") for li in F.tableau(*VARIANTES_FEU[n][:1], **VARIANTES_FEU[n][1])["lignes"].values()
        for v in li.values() for x in v) and not etoiles("5.2a") and not etoiles("5.3"))

# la monotonie : dans une case ou une ligne, la dimension croît et a ne croît jamais ; dans une colonne, a et la dimension
# ne décroissent jamais quand la durée augmente (à rang égal de combinaison)
def non_monotones() -> list:
    out = []
    for numero in CONCEPTION:
        mo = cases_moteur(F, numero)
        cols = COLONNES_FEU[numero]
        for r in F.R_DUREES:
            ligne = [x for c in cols for x in mo[r][c]] if numero in ("5.5", "5.6") else None
            for groupe in ([ligne] if ligne is not None else [mo[r][c] for c in cols]):
                for x, y in zip(groupe, groupe[1:]):
                    if not (x[0] < y[0] and x[1] >= y[1]):
                        out.append((numero, r, x, y))
        for c in cols:
            for r1, r2 in zip(F.R_DUREES, F.R_DUREES[1:]):
                for x, y in zip(mo[r1][c], mo[r2][c]):
                    if (x[0] is not None and y[0] < x[0]) or (x[1] is not None and y[1] < x[1]):
                        out.append((numero, c, r1, r2, x, y))
    return out


chk("monotonie des sept tableaux : à durée donnée la dimension croît et a ne croît jamais ; dans chaque colonne, ni a ni la dimension "
    "ne décroissent quand la durée augmente", not non_monotones(), non_monotones()[:4])


def a_exige(element, r, **k):
    return F.exigence(element, r, **k)["a"]


chk("monotonie de ``exigence`` : à dimension donnée a ne décroît pas quand R augmente ; à R donné a ne croît pas quand la dimension augmente "
    "(poutres 5.5 et 5.6, poteaux aux quatre colonnes, de 50 à 800 mm par pas de 10 mm)",
    all(a_exige("poutre", r, appuis=ap, b=b1) >= a_exige("poutre", r, appuis=ap, b=b2)
        for ap in ("simples", "continue") for r in F.R_DUREES for b1, b2 in zip(range(700, 801, 10), range(710, 811, 10)))
    and all((F.exigence("poutre", r, appuis=ap, b=b1)["a"] or 0) >= (F.exigence("poutre", r, appuis=ap, b=b2)["a"] or 0)
            for ap in ("simples", "continue") for r in F.R_DUREES for b1, b2 in zip(range(50, 800, 10), range(60, 810, 10))
            if F.exigence("poutre", r, appuis=ap, b=b1)["atteinte"])
    and all((F.exigence("poteau", r, exposition=ex, mu_fi=mu, b=b1)["a"] or 0) >= (F.exigence("poteau", r, exposition=ex, mu_fi=mu, b=b2)["a"] or 0)
            for ex, mu in (("plusieurs", 0.2), ("plusieurs", 0.5), ("plusieurs", 0.7), ("un", 0.7)) for r in F.R_DUREES
            for b1, b2 in zip(range(50, 800, 10), range(60, 810, 10)) if F.exigence("poteau", r, exposition=ex, mu_fi=mu, b=b1)["atteinte"])
    and all(a_exige("poutre", r1, appuis=ap, b=800) <= a_exige("poutre", r2, appuis=ap, b=800)
            for ap in ("simples", "continue") for r1, r2 in zip(F.R_DUREES, F.R_DUREES[1:]))
    and all(a_exige("poteau", r1, mu_fi=mu, b=800) <= a_exige("poteau", r2, mu_fi=mu, b=800)
            for mu in (0.2, 0.5) for r1, r2 in zip(F.R_DUREES, F.R_DUREES[1:])))

# ---- les cas d'hier (les défauts D1, D2, D6 de l'ancien module) et les lectures du tableau
print("     — les cas d'hier et les lectures")
po = F.exigence("poteau", "R90", mu_fi=0.7, b=350)
chk("D1 d'hier — poteau R 90, μ_fi = 0,7, b = 350 mm : 350/53 (l'ancien module donnait 200/31, la colonne μ_fi = 0,2)",
    po["combinaison"] == "350/53" and po["a"] == 53 and po["dimension_min"] == 350 and po["atteinte"] is True and po["tableau"] == "5.2a"
    and not po["huit_barres"] and po["colonne"] == "mu07", po)
po = F.exigence("poteau", "R90", mu_fi=0.7, b=450)
chk("… et avec b = 450 mm : 450/40**, la combinaison la plus favorable que la dimension autorise, « au moins 8 barres »",
    po["combinaison"] == "450/40**" and po["a"] == 40 and po["huit_barres"] and po["indice"] == 1 and [c["dimension"] for c in po["combinaisons"]] == [350, 450])
po = F.exigence("poteau", "R90", mu_fi=0.7, b=449)
chk("… b = 449 mm : encore 350/53 (rien n'est interpolé : la plus grande dimension minimale qui ne dépasse pas la dimension réelle)", po["combinaison"] == "350/53")
po = F.exigence("poteau", "R90", mu_fi=0.7)
chk("… sans dimension : la première combinaison (350/53), « à vérifier » (atteinte None)", po["combinaison"] == "350/53" and po["atteinte"] is None and po["dimension"] is None)
po = F.exigence("poteau", "R90", mu_fi=0.7, b=300)
chk("… b = 300 mm < 350 : durée NON ATTEINTE par la méthode tabulée, la raison nomme le tableau, la dimension et le minimum",
    po["atteinte"] is False and po["combinaison"] == "350/53" and "350 mm" in po["raison"] and "5.2a" in po["raison"] and "300 mm" in po["raison"], po["raison"])
po = F.exigence("poteau", "R240", mu_fi=0.7, b=600)
chk("poteau R 240 à μ_fi = 0,7 : la case est VIDE — aucune valeur inventée, durée non atteinte, renvoi à la méthode B",
    po["retenue"] is None and po["a"] is None and po["atteinte"] is False and po["combinaisons"] == [] and "pas de valeur" in po["raison"] and "méthode B" in po["raison"])
chk("poteau : les trois colonnes « plusieurs côtés » à R 120 (250/40 · 350/35 ; 350/45** · 450/40** ; 350/57** · 450/51**) et la colonne « un côté » (175/35)",
    [F.exigence("poteau", "R120", mu_fi=m, b=b)["combinaison"] for m in (0.2, 0.5, 0.7) for b in (350, 450)] == ["350/35", "350/35", "350/45**", "450/40**", "350/57**", "450/51**"]
    and F.exigence("poteau", "R120", mu_fi=0.2, b=300)["combinaison"] == "250/40" and F.exigence("poteau", "R120", exposition="un", b=200)["combinaison"] == "175/35")
pu = F.exigence("poteau", "R90", exposition="un", mu_fi=0.5, b=200)
chk("poteau exposé sur UN côté : le tableau n'a que μ_fi = 0,7 — la colonne est retenue pour μ_fi = 0,5 et c'est DIT (côté sûr)",
    pu["combinaison"] == "155/25" and pu["colonne"] == "un_cote" and any("un seul côté" in m and "0,5" in m for m in pu["mentions"])
    and F.exigence("poteau", "R90", exposition="un", mu_fi=0.7, b=200)["mentions"] == [])
vo = F.exigence("voile", "R180", mu_fi=0.7, faces=1)
chk("D2 d'hier — voile porteur REI 180, μ_fi = 0,7, une face : 210/50 (l'ancien module donnait 45) ; REI 240 : 270/60 (55)",
    vo["combinaison"] == "210/50" and vo["a"] == 50 and vo["tableau"] == "5.4" and vo["duree"] == "REI 180"
    and F.exigence("voile", "R240", mu_fi=0.7, faces=1)["combinaison"] == "270/60")
chk("voile : deux faces (270/55), μ_fi = 0,35 (180/40 et 200/45), et le voile NON porteur (tableau 5.3 : EI 180, 150 mm, aucune distance à l'axe)",
    F.exigence("voile", "R180", mu_fi=0.7, faces=2)["combinaison"] == "270/55" and F.exigence("voile", "R180", mu_fi=0.35, faces=1)["combinaison"] == "180/40"
    and F.exigence("voile", "R180", mu_fi=0.35, faces=2)["combinaison"] == "200/45"
    and (lambda v: v["tableau"] == "5.3" and v["duree"] == "EI 180" and v["dimension_min"] == 150 and v["a"] is None and v["combinaison"] == "150")(F.exigence("voile", "R180", porteur=False)))
chk("voile de 200 mm pour REI 180 (210 exigés) : non atteinte ; de 210 mm : atteinte",
    F.exigence("voile", "R180", e=200)["atteinte"] is False and F.exigence("voile", "R180", e=210)["atteinte"] is True)
pt = F.exigence("poutre", "R60", b=200)
chk("poutre R 60 de 200 mm sur appuis simples (tableau 5.5) : 200/30, a = 30, a_sd = 40", pt["tableau"] == "5.5" and pt["combinaison"] == "200/30" and pt["a"] == 30 and pt["a_sd"] == 40 and pt["angle"])
pc = F.exigence("poutre", "R60", appuis="continue", b=200)
chk("… continue (tableau 5.6) : 200/12*, l'enrobage de l'EN 1992-1-1 gouverne normalement ; a_sd = 22 (200 mm est la largeur de la colonne 3)",
    pc["tableau"] == "5.6" and pc["combinaison"] == "200/12*" and pc["a"] == 12 and pc["etoile"] and pc["a_sd"] == 22)
chk("poutre : sans largeur, la première combinaison (120/40, a_sd = 50) ; b = 100 mm < 120 : non atteinte ; b = 2 000 mm : la dernière (300/25)",
    (lambda x: x["combinaison"] == "120/40" and x["a_sd"] == 50 and x["atteinte"] is None)(F.exigence("poutre", "R60"))
    and F.exigence("poutre", "R60", b=100)["atteinte"] is False and F.exigence("poutre", "R60", b=2000)["combinaison"] == "300/25")
# a_sd (§5.6.1(8) et la note sous les tableaux — décision de l'intégrateur du 10/11/2026, la lecture de la NORME) : a + 10 mm
# pour les barres d'angle d'une poutre à UN lit, jusqu'à la largeur de la COLONNE 4 du tableau 5.5 (colonnes numérotées
# avec celle des durées : la TROISIÈME combinaison) et de la COLONNE 3 du tableau 5.6 (la DEUXIÈME), bornes comprises ;
# pour une largeur plus grande, aucune majoration. Les exemples ci-dessous sont écrits en clair.
A_SD_EXEMPLES = [  # (appuis, durée, b, combinaison, a, a_sd)
    ("simples", "R60", 200, "200/30", 30, 40), ("simples", "R60", 250, "200/30", 30, 30), ("simples", "R60", 300, "300/25", 25, 25),
    ("continue", "R90", 250, "250/25", 25, 35), ("continue", "R90", 300, "250/25", 25, 25),
    ("continue", "R120", 300, "300/35", 35, 45), ("continue", "R120", 450, "450/35", 35, 35)]


def a_sd_faux(module) -> list:
    return [(ap, r, b, x["combinaison"], x["a"], x["a_sd"]) for ap, r, b, comb, a, asd in A_SD_EXEMPLES
            for x in (module.exigence("poutre", r, appuis=ap, b=b),) if (x["combinaison"].rstrip("*"), x["a"], x["a_sd"]) != (comb, a, asd)]


chk("a_sd = a + 10 mm jusqu'à la largeur de la colonne 4 du tableau 5.5 (3ᵉ combinaison) et de la colonne 3 du tableau 5.6 (2ᵉ), bornes comprises ; "
    "au-delà, aucune majoration — 5.5 R 60 : b 200 → 40, b 250 → 30, b 300 → 25 ; 5.6 R 90 : b 250 → 35, b 300 → 25 ; 5.6 R 120 : b 300 → 45, b 450 → 35",
    not a_sd_faux(F) and F.A_SD_ANGLE == 10 and F.RANG_SANS_A_SD == {"5.5": 2, "5.6": 1}, a_sd_faux(F))
chk("… la largeur limite de chaque durée : tableau 5.5 → 160, 200, 300, 300, 400, 500 mm ; tableau 5.6 → 160, 200, 250, 300, 400, 500 mm ; "
    "à la largeur limite a_sd est majoré, 1 mm au-delà il ne l'est plus",
    [F.exigence("poutre", r, b=1000)["largeur_a_sd"] for r in F.R_DUREES] == [160, 200, 300, 300, 400, 500]
    and [F.exigence("poutre", r, appuis="continue", b=1000)["largeur_a_sd"] for r in F.R_DUREES] == [160, 200, 250, 300, 400, 500]
    and all((lambda x, y: x["a_sd"] == x["a"] + 10 and x["angle"] and y["a_sd"] == y["a"] and not y["angle"])(
        F.exigence("poutre", r, appuis=ap, b=lim), F.exigence("poutre", r, appuis=ap, b=lim + 1))
        for ap in ("simples", "continue") for r in F.R_DUREES for lim in (F.exigence("poutre", r, appuis=ap, b=1000)["largeur_a_sd"],)))
chk("… sans largeur donnée : la première combinaison, a_sd majoré (le cas le plus sévère) ; la note du tableau cite la colonne 4, la colonne 3 et §5.6.1(8)",
    F.exigence("poutre", "R60")["a_sd"] == 50 and F.exigence("poutre", "R90", appuis="continue")["a_sd"] == 45
    and all(mot in F.NOTE_A_SD for mot in ("colonne 4 du tableau 5.5", "colonne 3 du tableau 5.6", "§5.6.1(8)", "aucune majoration")) and "rovisoire" not in F.NOTE_A_SD)
da = F.exigence("dalle", "R90", h=200)
chk("dalle REI 90 (tableau 5.8) : une direction 100/30 ; deux directions 15 et 20 ; continue : la colonne « l_y/l_x ≤ 1,5 » (§5.7.3 : colonnes 2 et 4), dit",
    da["combinaison"] == "100/30" and da["duree"] == "REI 90" and da["symbole_dimension"] == "h_s"
    and [F.exigence("dalle", "R90", portee=p)["a"] for p in ("une", "deux_15", "deux_20", "continue")] == [30, 15, 20, 15]
    and F.exigence("dalle", "R90", portee="continue")["colonne"] == "deux_15" and any("§5.7.3" in m for m in F.exigence("dalle", "R90", portee="continue")["mentions"]))
chk("dalle : les conditions d'emploi sont celles de SA portée (une direction : aucune ; continue : redistribution ≤ 15 % et armatures sur appuis)",
    F.exigence("dalle", "R90")["conditions"] == [] and any("15 %" in c for c in F.exigence("dalle", "R90", portee="continue")["conditions"])
    and any("quatre bords" in c for c in F.exigence("dalle", "R90", portee="deux_15")["conditions"]))
chk("plancher-dalle REI 90 (tableau 5.9) : 200/25 ; une dalle de 180 mm : non atteinte", F.exigence("dalle", "R90", portee="plancher_dalle", h=200)["combinaison"] == "200/25"
    and F.exigence("dalle", "R90", portee="plancher_dalle", h=200)["tableau"] == "5.9" and F.exigence("dalle", "R90", portee="plancher_dalle", h=180)["atteinte"] is False)
chk("chaque exigence porte son tableau, sa source (norme, édition, tableau, clause), ses notes, son état de vérification — l'une des quatre formules",
    all(e["source"].startswith("EN 1992-1-2:2004, tableau ") and "§5." in e["source"] and e["notes"] and e["non_couvert"]
        and e["verification"].startswith(("recoupé sur plusieurs sources", "recoupé sur une source", "partiellement recoupé : ", "transcrit, non recoupé"))
        for e in (F.exigence("poutre", "R60"), F.exigence("poutre", "R60", appuis="continue"), F.exigence("dalle", "R60"), F.exigence("dalle", "R60", portee="plancher_dalle"),
                  F.exigence("poteau", "R60"), F.exigence("voile", "R60"), F.exigence("voile", "R60", porteur=False)))
    and set(F.VERIFICATIONS) == set(F.TABLEAUX) and all(F.tableau(*v[:1], **v[1])["verification"] == F.VERIFICATIONS[n] for n, v in VARIANTES_FEU.items()))
chk("les conditions d'emploi : méthode A (l_0,fi ≤ 3 m, e ≤ 0,15·h, A_s < 0,04·A_c), voiles (élancement ≤ 40), poutres et dalles continues (redistribution ≤ 15 %)",
    all(any(mot in c for c in F.TABLEAUX[n]["conditions"]) for n, mots in (("5.2a", ("l_0,fi ≤ 3 m", "0,15·h", "A_s < 0,04·A_c")), ("5.3", ("≤ 40",)), ("5.4", ("≤ 40",)),
                                                                           ("5.6", ("≤ 15 %",)), ("5.8", ("≤ 15 %",)), ("5.9", ("≤ 15 %",))) for mot in mots))
chk("non couvert, dit : méthode B, sections en I (classe WC), précontrainte, plusieurs lits, haute résistance, granulats calcaires",
    all(any(mot in x for x in F.NON_COUVERT) for mot in ("méthode B", "classe WC", "précontrainte", "plusieurs lits", "haute résistance", "calcaires")))

# ---- la formule (5.7) : recalculée À LA MAIN (math seul), sur des valeurs écrites en clair
print("     — la formule (5.7) des poteaux, à la main")


def a_main(minutes, b_prime, l0, mu, n, alpha_cc=0.85, omega=0.0):
    """a = 30 + [120·(R/120)^(1/1,8) − R_ηfi − R_l − R_b − R_n]/1,60, écrit ici sans rien lire du moteur."""
    r_eta = 83.0 * (1.00 - mu * (1.0 + omega) / ((0.85 / alpha_cc) + omega))
    r_l = 9.60 * (5.0 - l0)
    r_b = 0.09 * b_prime
    r_n = 0.0 if n == 4 else 12.0
    return 30.0 + (120.0 * (minutes / 120.0) ** (1.0 / 1.8) - r_eta - r_l - r_b - r_n) / 1.60


MAIN_57 = [  # (R, b, μ_fi, barres, a calculé à 0,1 près, a retenu) — l_0,fi = 3 m ; les quatre contrôles du § 4.5 de la conception
    ("R90", 350, 0.7, 4, 46.7, 47), ("R90", 300, 0.5, 4, 39.1, 40), ("R90", 300, 0.5, "plus", 31.6, 32), ("R60", 200, 0.7, 4, 42.2, 43)]
for r_, b_, mu_, n_, brut_, a_ in MAIN_57:
    fo = F.poteau_formule(r_, b_, b_, 3, mu_, n_)
    main = a_main(int(r_[1:]), b_, 3.0, mu_, 4 if n_ == 4 else 8)
    chk(f"formule (5.7) à la main — {r_}, {b_} × {b_}, l_0,fi = 3 m, μ_fi = {mu_}, {'4 barres' if n_ == 4 else 'plus de 4 barres'} : a = {brut_} → {a_} mm",
        abs(main - brut_) < 0.05 and abs(fo["a_brut"] - main) < 1e-9 and fo["a"] == a_ == math.ceil(main) and fo["atteinte"], (main, fo["a_brut"], fo["a"]))
fo = F.poteau_formule("R90", 350, 350, 3, 0.7, 4)
chk("… ses cinq termes : R_ηfi = 83·(1 − 0,7) = 24,9 ; R_l = 9,6·(5 − 3) = 19,2 ; R_b = 0,09·350 = 31,5 ; R_n = 0 ; R_a = 26,68 ; b′ = 350 ; α_cc = 0,85 nommé",
    abs(fo["R_eta"] - 24.9) < 1e-9 and abs(fo["R_l"] - 19.2) < 1e-9 and abs(fo["R_b"] - 31.5) < 1e-9 and fo["R_n"] == 0 and abs(fo["R_a"] - 1.6 * (fo["a_brut"] - 30)) < 1e-9
    and fo["b_prime"] == 350 and fo["alpha_cc"] == F.ALPHA_CC == 0.85 and abs(fo["somme_exigee"] - 120 * 0.75 ** (1 / 1.8)) < 1e-9)
chk("avec α_cc = 0,85 le terme en ω se simplifie : R_ηfi = 83·(1 − μ_fi) quel que soit ω (0, 0,1, 0,5, 1)",
    all(abs(F.r_eta(mu, om) - 83 * (1 - mu)) < 1e-9 for mu in (0.2, 0.35, 0.5, 0.7) for om in (0.0, 0.1, 0.5, 1.0)))
chk("b′ = 2·b·h/(b + h) : 300 × 450 → 360 mm ; un poteau rond ou carré : son côté", abs(F.poteau_formule("R90", 300, 450, 3, 0.5, 4)["b_prime"] - 360.0) < 1e-9
    and F.poteau_formule("R90", 300, None, 3, 0.5, 4)["b_prime"] == 300)
chk("arrondi au millimètre SUPÉRIEUR, jamais sous 25 mm : R 30, 400 mm, μ_fi = 0,2 → 25 (calculé bien en dessous) ; une valeur entière reste elle-même",
    F.poteau_formule("R30", 400, 400, 3, 0.2, 4)["a"] == 25 and F.poteau_formule("R30", 400, 400, 3, 0.2, 4)["plancher"]
    and all(F.poteau_formule(r, b, b, l0, mu, n)["a"] in (25, math.ceil(F.poteau_formule(r, b, b, l0, mu, n)["a_brut"] - 1e-9))
            for r in F.R_DUREES for b in (200, 300, 450) for l0 in (2, 3, 6) for mu in (0.2, 0.5, 0.7) for n in (4, "plus")))
fo240 = F.poteau_formule("R240", 450, 450, 3, 0.7, 4)
chk("R 240, 450 mm, μ_fi = 0,7 : a calculé 87,4 mm > 80 mm — durée NON ATTEINTE par la formule (hors de son domaine), dit",
    not fo240["atteinte"] and abs(fo240["a_brut"] - a_main(240, 450, 3, 0.7, 4)) < 1e-9 and 87 < fo240["a_brut"] < 88 and "80 mm" in fo240["raison"])
# l'exemple chiffré PUBLIÉ que le second rapport de sources a relevé (support de cours) :
# R = 120·((35,4 + 27,2 + 28,8 + 32,4 + 12)/120)^1,8 = 150 min — soit a = 47 mm, l_0,fi = 2 m, b′ = 360 mm, plus de 4 barres,
# et R_ηfi = 35,4, ce que 83·(1 − μ_fi) donne pour μ_fi = 0,5735
ex = F.poteau_duree(47, 360, 360, 2, 1 - 35.4 / 83, "plus")
chk("exemple publié de la formule (5.7) : R = 120·((35,4 + 27,2 + 28,8 + 32,4 + 12)/120)^1,8 = 150 min",
    abs(ex["R_eta"] - 35.4) < 1e-9 and abs(ex["R_a"] - 27.2) < 1e-9 and abs(ex["R_l"] - 28.8) < 1e-9 and abs(ex["R_b"] - 32.4) < 1e-9 and ex["R_n"] == 12
    and abs(ex["R"] - 120 * (135.8 / 120) ** 1.8) < 1e-9 and abs(ex["R"] - 150) < 0.5 and ex["classe"] == "R120", ex["R"])
# l'aller-retour a → durée → a, et les monotonies
aller = []
for r_ in F.R_DUREES:
    for b_ in (200, 250, 300, 350, 400, 450):
        for l0 in (2, 3, 4.5, 6):
            for mu_ in (0.2, 0.35, 0.5, 0.7):
                for n_ in (4, "plus"):
                    fo = F.poteau_formule(r_, b_, b_, l0, mu_, n_)
                    if not fo["atteinte"]:
                        continue
                    du = F.poteau_duree(fo["a"], b_, b_, l0, mu_, n_)
                    juste = F.poteau_duree(max(25.0, fo["a_brut"]), b_, b_, l0, mu_, n_)
                    # la durée obtenue avec le a retenu (arrondi au-dessus) couvre la durée demandée ; avec le a exact, elle la redonne
                    if du["R"] < int(r_[1:]) - 1e-6 or (fo["a_brut"] >= 25 and abs(juste["R"] - int(r_[1:])) > 1e-6):
                        aller.append((r_, b_, l0, mu_, n_, fo["a"], du["R"]))
chk("aller-retour a → durée → a sur 1 152 poteaux : la durée obtenue avec le a retenu couvre la durée demandée, et le a exact la redonne à 10⁻⁶ min",
    not aller, aller[:3])
chk("monotonies de la formule : la durée croît avec a, avec b′ et avec le nombre de barres ; elle décroît quand μ_fi ou l_0,fi augmentent",
    all(F.poteau_duree(a, 300, 300, 3, 0.5, 4)["R"] < F.poteau_duree(a + 5, 300, 300, 3, 0.5, 4)["R"] for a in range(25, 76, 5))
    and all(F.poteau_duree(40, b, b, 3, 0.5, 4)["R"] < F.poteau_duree(40, b + 50, b + 50, 3, 0.5, 4)["R"] for b in range(200, 401, 50))
    and F.poteau_duree(40, 300, 300, 3, 0.5, 4)["R"] < F.poteau_duree(40, 300, 300, 3, 0.5, "plus")["R"]
    and all(F.poteau_duree(40, 300, 300, 3, m, 4)["R"] > F.poteau_duree(40, 300, 300, 3, m + 0.1, 4)["R"] for m in (0.2, 0.3, 0.4, 0.5, 0.6))
    and all(F.poteau_duree(40, 300, 300, l, 0.5, 4)["R"] > F.poteau_duree(40, 300, 300, l + 1, 0.5, 4)["R"] for l in (2, 3, 4, 5)))
for nom, args, mot in (("b′ < 200 mm", ("R90", 190, 190, 3, 0.5, 4), "plus petit côté b"), ("b′ > 450 mm", ("R90", 460, 460, 3, 0.5, 4), "plus petit côté b"),
                       ("h > 1,5·b", ("R90", 300, 451, 3, 0.5, 4), "grand côté h"), ("h < b", ("R90", 300, 250, 3, 0.5, 4), "grand côté h"),
                       ("l_0,fi > 6 m", ("R90", 300, 300, 6.1, 0.5, 4), "l_0,fi"), ("l_0,fi nul", ("R90", 300, 300, 0, 0.5, 4), "l_0,fi"),
                       ("μ_fi > 0,7", ("R90", 300, 300, 3, 0.71, 4), "μ_fi"), ("μ_fi nul", ("R90", 300, 300, 3, 0, 4), "μ_fi"),
                       ("3 barres", ("R90", 300, 300, 3, 0.5, 3), "nombre de barres"), ("durée R 45", ("R45", 300, 300, 3, 0.5, 4), "durée"),
                       ("b illisible", ("R90", "abc", None, 3, 0.5, 4), "plus petit côté b"), ("sans b", ("R90", None, None, 3, 0.5, 4), "plus petit côté b")):
    ok, msg = refus(lambda a=args: F.poteau_formule(*a), mot)
    chk(f"formule (5.7), hors domaine refusé avec le champ nommé : {nom}", ok, msg)
chk("les bornes du domaine elles-mêmes sont admises : b′ = 200 et 450 mm, h = 1,5·b, l_0,fi = 6 m, μ_fi = 0,7",
    all(F.poteau_formule("R60", *a)["a"] >= 25 for a in ((200, 200, 3, 0.5, 4), (450, 450, 3, 0.5, 4), (300, 450, 3, 0.5, 4), (300, 300, 6, 0.5, 4), (300, 300, 3, 0.7, 4))))
bas = F.poteau_formule("R90", 300, 300, 1.5, 0.1, 4)
chk("l_0,fi = 1,5 m et μ_fi = 0,1 : ramenés à 2 m et à 0,2 (côté sûr, permis par la norme) — et c'est DIT dans le résultat, jamais en silence",
    bas["l0_calcul"] == 2.0 and bas["mu_calcul"] == 0.2 and bas["l0_fi"] == 1.5 and bas["mu_fi"] == 0.1 and len(bas["mentions"]) == 2
    and "2 m" in bas["mentions"][0] and "0,2" in bas["mentions"][1] and abs(bas["a_brut"] - a_main(90, 300, 2.0, 0.2, 4)) < 1e-9
    and F.poteau_formule("R90", 300, 300, 2, 0.2, 4)["mentions"] == [])
ok, msg = refus(lambda: F.poteau_duree(24, 300, 300, 3, 0.5, 4), "distance à l'axe a")
chk("durée obtenue : a < 25 mm refusé (hors domaine) ; a > 80 mm : calculée avec 80 mm, et c'est dit", ok
    and F.poteau_duree(95, 300, 300, 3, 0.5, 4)["a_calcul"] == 80 and F.poteau_duree(95, 300, 300, 3, 0.5, 4)["R"] == F.poteau_duree(80, 300, 300, 3, 0.5, 4)["R"]
    and any("80 mm" in m for m in F.poteau_duree(95, 300, 300, 3, 0.5, 4)["mentions"]), msg)
# la formule et le tableau se recoupent : avec α_cc = 1,0 (la valeur du tableau 5.2a), ω = 0 et l_0,fi = 3 m, la formule
# (5.7) recalculée ICI redonne chaque case des trois colonnes « plusieurs côtés » du tableau 5.2a à 1,5 mm près — avec
# R_n = 12 exactement pour les cases marquées « ** » —, et la case vide (R 240, μ_fi = 0,7) est celle où a dépasse 80 mm
def recoupement(module) -> list:
    """Les combinaisons « plusieurs côtés » du tableau 5.2a que la formule recalculée ici (α_cc = 1,0, ω = 0, l_0,fi = 3 m)
    ne redonne pas à 1,5 mm près, ou que le mauvais nombre de barres redonnerait aussi bien (jugé seulement au-dessus
    du plancher de 25 mm : à 25 mm, quatre barres ou huit donnent la même case)."""
    out = []
    for r_ in module.R_DUREES:
        for mu_, col in ((0.2, "mu02"), (0.5, "mu05"), (0.7, "mu07")):
            for b_, a_, repere in module.POTEAU_5_2A[r_][col]:
                calc = a_main(int(r_[1:]), b_, 3.0, mu_, 8 if repere == "**" else 4, alpha_cc=1.0)
                autre = a_main(int(r_[1:]), b_, 3.0, mu_, 4 if repere == "**" else 8, alpha_cc=1.0)
                if abs(max(25.0, calc) - a_) > 1.5 or (a_ > 26.5 and abs(max(25.0, autre) - a_) <= 1.5):
                    out.append((r_, col, b_, a_, round(calc, 1), round(autre, 1)))
    return out


chk("recoupement tableau ↔ formule : la formule (5.7) à α_cc = 1,0, ω = 0, l_0,fi = 3 m redonne les 26 combinaisons « plusieurs côtés » du tableau 5.2a "
    "à 1,5 mm près, les repères « ** » (R_n = 12) compris ; R 240 à μ_fi = 0,7 : a > 80 mm, la case vide du tableau",
    not recoupement(F) and sum(len(F.POTEAU_5_2A[r][c]) for r in F.R_DUREES for c in ("mu02", "mu05", "mu07")) == 26
    and a_main(240, 450, 3.0, 0.7, 8, alpha_cc=1.0) > 80, recoupement(F)[:4])
# ``exigence`` avec la formule : jamais par défaut ; la mention et la comparaison avec le tableau de l'EN
chk("la formule n'est JAMAIS la méthode par défaut : ``exigence`` d'un poteau lit le tableau 5.2a ; ``METHODE_DEFAUT`` vaut « tableau »",
    F.METHODE_DEFAUT == "tableau" and F.exigence("poteau", "R90", b=350)["methode"] == "tableau" and F.exigence("poteau", "R90", b=350)["tableau"] == "5.2a"
    and F.exigence("poteau", "R90", b=350)["formule"] is None and F.exigence("poteau", "R90", b=350)["a"] == 53)
ef = F.exigence("poteau", "R90", methode="formule", mu_fi=0.7, b=350)
chk("formule choisie : a = 47 mm, la mention « tableaux 5.2.a-ANB… à contrôler sur l'ANB:2021 ; le tableau belge exclut certaines colonnes de 200 mm », "
    "et la comparaison « tableau 5.2a de l'EN (μ_fi = 0,7) : 350/53 »",
    ef["methode"] == "formule" and ef["a"] == 47 and ef["atteinte"] and ef["formule"]["a"] == 47
    and any("5.2.a-ANB" in m and "ANB:2021" in m and "colonnes de 200 mm" in m for m in ef["mentions"])
    and ef["comparaison"]["phrase"] == "tableau 5.2a de l'EN (μ_fi = 0,7) : 350/53" and ef["comparaison"]["a"] == 53
    and "formule (5.7)" in ef["source"] and "0,85" in " ".join(ef["notes"]), (ef["mentions"], ef["comparaison"]))
chk("… la comparaison prend la première colonne tabulée qui couvre μ_fi (0,35 → 0,5 ; 0,6 → 0,7), jamais celle du dessous",
    F.exigence("poteau", "R90", methode="formule", mu_fi=0.35, b=300)["comparaison"]["phrase"] == "tableau 5.2a de l'EN (μ_fi = 0,5) : 300/45"
    and F.exigence("poteau", "R90", methode="formule", mu_fi=0.6, b=450)["comparaison"]["phrase"] == "tableau 5.2a de l'EN (μ_fi = 0,7) : 450/40**"
    and "aucune valeur" in F.exigence("poteau", "R240", methode="formule", mu_fi=0.7, b=450)["comparaison"]["phrase"])
chk("le tableau de l'EN est plus sévère que la formule belge sur chaque combinaison « plusieurs côtés » du tableau 5.2a (b′ ≥ 200 mm)",
    all(F.poteau_formule(r, b, b, 3, mu, "plus" if rep == "**" else 4)["a"] <= a for r in F.R_DUREES for mu, col in ((0.2, "mu02"), (0.5, "mu05"), (0.7, "mu07"))
        for b, a, rep in F.POTEAU_5_2A[r][col]))
for nom, f_, mot in (("formule pour un poteau exposé sur un seul côté", lambda: F.exigence("poteau", "R90", methode="formule", exposition="un", b=300), "exposition"),
                     ("grand côté h donné au tableau", lambda: F.exigence("poteau", "R90", b=300, h=400), "grand côté h"),
                     ("μ_fi = 0,6 au tableau (rien n'est interpolé)", lambda: F.exigence("poteau", "R90", mu_fi=0.6, b=300), "μ_fi"),
                     ("μ_fi = 0,5 pour un voile", lambda: F.exigence("voile", "R90", mu_fi=0.5), "μ_fi"),
                     ("trois faces pour un voile", lambda: F.exigence("voile", "R90", faces=3), "faces"),
                     ("appuis « encastrée »", lambda: F.exigence("poutre", "R90", appuis="encastrée"), "appuis"),
                     ("portée « trois »", lambda: F.exigence("dalle", "R90", portee="trois"), "portée"),
                     ("largeur négative", lambda: F.exigence("poutre", "R90", b=-5), "largeur b"),
                     ("épaisseur illisible", lambda: F.exigence("dalle", "R90", h="x"), "épaisseur h"),
                     ("paramètre d'un autre élément (μ_fi pour une poutre)", lambda: F.exigence("poutre", "R90", mu_fi=0.7), "mu_fi"),
                     ("élément « semelle »", lambda: F.exigence("semelle", "R90"), "élément"),
                     ("durée R 45", lambda: F.exigence("poutre", "R45"), "durée"),
                     ("méthode « B »", lambda: F.exigence("poteau", "R90", methode="B"), "méthode")):
    ok, msg = refus(f_, mot)
    chk(f"feu, refus en français, le champ nommé : {nom}", ok, msg)

# ---- l'enrobage que le feu impose (beton/bibliotheque/enrobage.py lit feu.py : une source)
print("     — l'enrobage que le feu impose")
pf = E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60", "b": 250}})
f = pf["feu"]
chk("poutre R 60, b = 250 : combinaison 200/30 du tableau 5.5 ; b dépasse 200 mm (colonne 4) : a_sd = a = 30, c_nom,feu = 30 − 10 − 8 = 12 ; "
    "la durabilité (40) gouverne",
    f["combinaison"] == "200/30" and f["a"] == 30 and f["a_sd"] == 30 and not f["angle"] and f["c_nom"] == 12 and f["c_bas"] == 12 and f["c_cote"] == 12
    and pf["c_nom"] == 40 and pf["gouvernant"] == "durabilite" and f["atteinte"] is True)
pf200 = E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60", "b": 200}})
chk("poutre R 60, b = 200 : a = 30, a_sd = 40 (barres d'angle), c_nom,feu = 40 − 10 − 8 = 22 (la face latérale fixe l'enrobage ; sous-face : 12)",
    pf200["feu"]["a_sd"] == 40 and pf200["feu"]["angle"] and pf200["feu"]["c_nom"] == 22 and pf200["feu"]["c_bas"] == 12 and pf200["feu"]["c_cote"] == 22)
pf2 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 16, "phi_etrier": 6, "feu": {"r": "R120", "b": 240}})
f2 = pf2["feu"]
chk("poutre R 120, b = 240, Ø 16 / étrier 6 : a = 60, a_sd = 70, c_nom,feu = 70 − 8 − 6 = 56 > c_nom,dur 26 : le feu gouverne",
    f2["combinaison"] == "240/60" and f2["a_sd"] == 70 and f2["c_nom"] == 56 and pf2["c_nom"] == 56 and pf2["gouvernant"] == "feu")
chk("a est une valeur NOMINALE : c_nom,feu ne dépend pas de Δc_dev (10, 5, 0 mm) — la tolérance ne s'ajoute qu'à la durabilité",
    len({E.calculer({"element": "poutre", "phi": 16, "phi_etrier": 6, "delta_dev": dd, "feu": {"r": "R120", "b": 240}})["c_nom_feu"] for dd in (10, 5, 0)}) == 1
    and [E.calculer({"element": "poutre", "phi": 16, "phi_etrier": 6, "delta_dev": dd, "feu": {"r": "R120", "b": 240}})["c_nom_dur"] for dd in (10, 5, 0)] == [26, 21, 16])
pf3 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R90", "b": 400}})
chk("poutre R 90, b = 400 (au-delà des 300 mm de la colonne 4) : a = 35, a_sd non majoré, c_nom,feu = 35 − 10 − 8 = 17",
    pf3["feu"]["a"] == 35 and not pf3["feu"]["angle"] and pf3["feu"]["a_sd"] == 35 and pf3["feu"]["c_nom"] == 17)
pf4 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60", "b": 100}})
chk("poutre R 60, b = 100 < 120 : durée non atteinte par la méthode tabulée — avertissement GRAVE, la première combinaison sert de repère",
    pf4["feu"]["atteinte"] is False and any(a["code"] == "feu_non_atteint" and a["grave"] and "non atteinte par la méthode tabulée" in a["message"] for a in pf4["avertissements"])
    and pf4["feu"]["combinaison"] == "120/40")
pfd = E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12, "feu": {"r": "R60", "h": 200}})
chk("dalle REI 60 (tableau 5.8) : h_s ≥ 80, a = 20, c_nom,feu = 20 − 6 = 14 ; la durabilité (22) gouverne", pfd["feu"]["dimension_min"] == 80 and pfd["feu"]["a"] == 20
    and pfd["feu"]["c_nom"] == 14 and pfd["c_nom"] == 22 and pfd["feu"]["atteinte"] is True)
chk("D6 d'hier — DALLE ou VOILE : aucun étrier n'est retiré de a, quel que soit « Ø de l'étrier » (c_nom,feu = a − Ø/2 : 14 mm avec un étrier de 0, 8 ou 16)",
    [E.calculer({"element": el, "phi": 12, "phi_etrier": pe, "feu": {"r": "R60"}})["c_nom_feu"] for el in ("dalle",) for pe in (0, 8, 16)] == [14, 14, 14]
    and [E.calculer({"element": "voile", "phi": 12, "phi_etrier": pe, "feu": {"r": "R120"}})["c_nom_feu"] for pe in (0, 8, 16)] == [29, 29, 29])
chk("… seule la nappe extérieure, quand il y en a une, se retire (Ø_ext, 0 par défaut) : dalle REI 180, Ø 12, nappe Ø 10 → 55 − 6 − 10 = 39 ; sans nappe : 49",
    E.calculer({"element": "dalle", "phi": 12, "phi_ext": 10, "feu": {"r": "R180"}})["c_nom_feu"] == 39 and E.DEFAUTS["phi_ext"] == 0
    and E.calculer({"element": "dalle", "phi": 12, "feu": {"r": "R180"}})["c_nom_feu"] == 49 and E.calculer({"element": "dalle", "phi": 12, "feu": {"r": "R180"}})["gouvernant"] == "feu")
chk("… pour une POUTRE ou un POTEAU, c'est l'étrier qui se retire (8 mm par défaut), jamais la nappe extérieure",
    E.DEFAUTS["phi_etrier"] == 8 and E.calculer({"element": "poteau", "phi": 20, "phi_ext": 12, "feu": {"r": "R90", "b": 350}})["c_nom_feu"] == 53 - 10 - 8
    and E.phi_exterieur(E.normaliser({"element": "poutre", "phi_etrier": 10, "phi_ext": 5})) == 10 and E.phi_exterieur(E.normaliser({"element": "voile", "phi_etrier": 10, "phi_ext": 5})) == 5)
pp = E.calculer({"element": "poteau", "exposition": "XC1", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R90", "mu_fi": 0.7, "b": 350}})
chk("poteau R 90, μ_fi = 0,7, 350 mm : a = 53, c_nom,feu = 53 − 10 − 8 = 35 > 30 : le feu gouverne ; la distance obtenue a_réel = 35 + 8 + 10 = 53 ≥ 53",
    pp["feu"]["a"] == 53 and pp["c_nom_feu"] == 35 and pp["c_nom"] == 35 and pp["gouvernant"] == "feu" and pp["a_reel"] == 53 and pp["feu"]["marge"] == 0)
pp8 = E.calculer({"element": "poteau", "phi": 20, "feu": {"r": "R90", "mu_fi": 0.7, "b": 450}})
chk("poteau 450 mm : 450/40** → avertissement « au moins 8 barres »", any(a["code"] == "huit_barres" and "au moins 8 barres" in a["message"] for a in pp8["avertissements"]))
pp240 = E.calculer({"element": "poteau", "phi": 20, "feu": {"r": "R240", "mu_fi": 0.7, "b": 600}})
chk("poteau R 240 à μ_fi = 0,7 : aucun c_nom,feu (la case est vide), c_nom = durabilité, avertissement grave « durée non atteinte par la méthode tabulée »",
    pp240["c_nom_feu"] is None and pp240["c_nom"] == pp240["c_nom_dur"] and pp240["feu"]["a"] is None
    and any(a["code"] == "feu_non_atteint" and a["grave"] for a in pp240["avertissements"]))
pv = E.calculer({"element": "voile", "phi": 12, "feu": {"r": "R180", "mu_fi": 0.7, "faces": 1, "e": 250}})
chk("voile REI 180, μ_fi = 0,7, une face, Ø 12 : a = 50, c_nom,feu = 50 − 6 = 44 : le feu gouverne", pv["feu"]["a"] == 50 and pv["c_nom"] == 44 and pv["gouvernant"] == "feu")
pvn = E.calculer({"element": "voile", "phi": 12, "feu": {"r": "R180", "porteur": False, "e": 150}})
chk("voile NON porteur : l'épaisseur seule (150 mm), aucun enrobage imposé par le feu — et c'est dit", pvn["c_nom_feu"] is None and pvn["feu"]["atteinte"] is True
    and any(a["code"] == "feu_sans_a" for a in pvn["avertissements"]))
pfo = E.calculer({"element": "poteau", "phi": 20, "phi_etrier": 8, "feu": {"r": "R90", "methode": "formule", "mu_fi": 0.7, "b": 350}})
chk("poteau par la formule (5.7) : a = 47, c_nom,feu = 29 < 30 : la durabilité gouverne ; a_réel = 48 mm → durée obtenue 93 min (classe R 90) ; "
    "la ligne de comparaison « tableau 5.2a de l'EN : 350/53 » est un terme",
    pfo["c_nom_feu"] == 29 and pfo["c_nom"] == 30 and pfo["a_reel"] == 48 and abs(pfo["feu"]["duree_obtenue"]["R"] - 120 * ((24.9 + 1.6 * 18 + 19.2 + 31.5) / 120) ** 1.8) < 1e-9
    and pfo["feu"]["duree_obtenue"]["classe"] == "R90" and next(t for t in pfo["termes"] if t["cle"] == "comparaison")["valeur"] == "350/53"
    and any(a["code"] == "feu_mention" and "ANB:2021" in a["message"] for a in pfo["avertissements"]))
chk("par défaut (sans « methode »), l'enrobage d'un poteau lit le TABLEAU 5.2a, pas la formule", pp["feu"]["methode"] == "tableau" and pp["feu"]["formule"] is None)
chk("armature de peau : rappel dès que la distance à l'axe obtenue atteint 70 mm (poutre R 240 de 280 mm : a_sd = 100) ; pas en dessous",
    any(a["code"] == "armature_peau" and "§4.5.2" in a["message"] for a in E.calculer({"element": "poutre", "feu": {"r": "R240", "b": 280}})["avertissements"])
    and not any(a["code"] == "armature_peau" for a in pf["avertissements"]) and F.A_PEAU == 70)
chk("les conditions d'emploi du tableau sont un avertissement (non grave) dès que le feu est demandé ; sans feu, aucun avertissement de feu",
    any(a["code"] == "feu_conditions" and not a["grave"] and "l_0,fi ≤ 3 m" in a["message"] for a in pp["avertissements"])
    and not any(a["code"].startswith("feu") for a in E.calculer({"element": "poteau"})["avertissements"]))
# les termes : la feuille de calcul « symbole = expression = valeurs = résultat », recalculable à partir de ses seules valeurs
def recalcule(valeurs: str) -> float:
    """Une expression écrite avec ses valeurs (« max(30 ; 32) », « 50 − 10 − 8 », « 120·[(24,9 + …)/120]^1,8 ») → sa valeur."""
    e = re.sub(r"(\d),(\d)", r"\1.\2", valeurs).replace("−", "-").replace(";", ",").replace("·", "*").replace("^", "**").replace("[", "(").replace("]", ")")
    assert re.fullmatch(r"[\d\s.,+\-*/()max]+", e), valeurs
    return eval(e, {"max": max})  # noqa: S307 — une expression faite de nombres, d'opérateurs et de max


def termes_faux(res) -> list:
    out = []
    for t in res["termes"]:
        if not t["valeurs"] or not isinstance(t["valeur"], (int, float)) or isinstance(t["valeur"], bool):
            continue
        attendu = t.get("brut", t["valeur"])
        # les termes de la formule (5.7) sont écrits au dixième : l'écart admis est celui de cette écriture
        tol = 0.2 if t["cle"] in ("R_exige", "a", "R_obtenue", "R_eta", "R_l", "R_b") and res["feu"] and res["feu"]["methode"] == "formule" else 1e-9
        if t["cle"] == "R_obtenue":
            tol = 0.6
        if abs(recalcule(t["valeurs"]) - attendu) > tol:
            out.append((t["cle"], t["valeurs"], recalcule(t["valeurs"]), attendu))
    return out


CAS_TERMES = [pf, pf200, pf2, pf3, pf4, pfd, pp, pp8, pp240, pv, pvn, pfo, ex_nit := E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12}),
              E.calculer({"element": "poutre", "exposition": "XC2", "phi": 32, "dg": 40, "contre_sol": "direct", "feu": {"r": "R30", "b": 500}}),
              E.calculer({"element": "dalle", "phi": 20, "phi_ext": 10, "feu": {"r": "R30"}}),
              E.calculer({"element": "poteau", "phi": 25, "phi_etrier": 10, "feu": {"r": "R120", "methode": "formule", "mu_fi": 0.45, "b": 300, "h": 400, "l0_fi": 4.2, "barres": "plus"}})]
faux = [(i, x) for i, r_ in enumerate(CAS_TERMES) for x in termes_faux(r_)]
chk(f"la feuille de calcul : chaque terme « symbole = expression = valeurs = résultat » se RECALCULE à partir de ses seules valeurs écrites ({len(CAS_TERMES)} cas, "
    f"{sum(1 for r_ in CAS_TERMES for t in r_['termes'] if t['valeurs'])} expressions)", not faux, faux[:4])
chk("les termes : durabilité, puis feu, puis l'enrobage retenu et la distance à l'axe OBTENUE (a_réel = c_nom + Ø_étrier + Ø/2, comparée à la distance exigée)",
    [t["groupe"] for t in pf["termes"]] == sorted([t["groupe"] for t in pf["termes"]], key=("durabilite", "feu", "retenu").index)
    and [t["cle"] for t in pf200["termes"]][-2:] == ["cnom", "a_reel"] and pf200["termes"][-1]["valeur"] == 40 + 8 + 10
    and pf200["termes"][-1]["comparaison"] == {"symbole": "a_sd", "valeur": 40, "sens": "≥", "ok": True}
    and pf["termes"][-1]["comparaison"] == {"symbole": "a", "valeur": 30, "sens": "≥", "ok": True}
    and [t["cle"] for t in pf["termes"] if t["groupe"] == "feu"] == ["combinaison", "dimension", "a", "asd", "cfeu"])
chk("le terme de la dimension minimale se compare à la dimension réelle : « atteinte » (b = 250 ≥ 200), « non atteinte » (b = 100 < 120), « à vérifier » (sans dimension)",
    next(t for t in pf["termes"] if t["cle"] == "dimension")["etat"] == "atteinte" and next(t for t in pf4["termes"] if t["cle"] == "dimension")["etat"] == "non atteinte"
    and next(t for t in pf4["termes"] if t["cle"] == "dimension")["comparaison"]["ok"] is False
    and next(t for t in E.calculer({"element": "dalle", "feu": {"r": "R60"}})["termes"] if t["cle"] == "dimension")["etat"] == "à vérifier")
chk("chaque terme porte symbole, nom, valeur et une règle ; les termes du feu nomment le tableau, la combinaison et la clause de l'EN 1992-1-2",
    all(t["symbole"] and t["nom"] and t["valeur"] is not None and t["regle"] for r_ in CAS_TERMES for t in r_["termes"])
    and "tableau 5.5" in next(t for t in pf["termes"] if t["cle"] == "combinaison")["regle"] and "§5.6.2" in next(t for t in pf["termes"] if t["cle"] == "combinaison")["regle"]
    and next(t for t in pf["termes"] if t["cle"] == "combinaison")["valeur"] == "200/30" and "§5.2" in next(t for t in pf["termes"] if t["cle"] == "cfeu")["regle"])
chk("les termes normatifs de la durabilité citent leur paragraphe ou leur tableau ; le terme retenu dit ce qui gouverne",
    all("§" in t["regle"] or "tableau" in t["regle"] for t in pf["termes"] if t["cle"] in ("cmin_b", "S", "cmin_dur", "cmin", "ddev", "cnom_dur"))
    and "la durabilité gouverne" in next(t for t in pf["termes"] if t["cle"] == "cnom")["regle"] and "le feu gouverne" in next(t for t in pp["termes"] if t["cle"] == "cnom")["regle"])
chk("le résultat se relit en JSON strict (aucun objet Python, aucun infini)", all(json.loads(json.dumps(r_, allow_nan=False)) == json.loads(json.dumps(r_)) for r_ in CAS_TERMES))
chk("les tableaux de l'écran : 4.4N (6 × 7 + libellés belges) ; le feu : celui de l'élément et de sa variante (5.5 / 5.6, 5.8 / 5.9, 5.2a, 5.4 / 5.3)",
    E.tableau_c_min_dur()["belges"] == list(NIT_20_BELGES) and [E.tableau_feu(*v[:1], **v[1])["numero"] for v in VARIANTES_FEU.values()] == list(VARIANTES_FEU)
    and E.R_DUREES is F.R_DUREES)
chk("une seule source : l'enrobage ne porte plus aucun tableau du feu (ni FEU_POUTRE, ni FEU_DALLE, ni FEU_A_TRANSCRIRE)",
    not any(hasattr(E, n) for n in ("FEU_POUTRE", "FEU_DALLE", "FEU_A_TRANSCRIRE", "A_SD_ANGLE")))

# ============================================================================================ 4. aciers
print("\n=== 4 · Aciers d'armature : annexe C")
chk("six nuances (B400/B500 × A/B/C), E_s 200 GPa, 7 850 kg/m³", len(S.nuances()) == 6 and S.E_S == 200.0 and S.MASSE_VOLUMIQUE == 7850.0)
fb = S.fiche(500, "B")
chk("B500B : k ≥ 1,08, ε_uk ≥ 5,0 % ; B500A : k ≥ 1,05, ε_uk ≥ 2,5 % ; B500C : 1,15 ≤ k < 1,35, ε_uk ≥ 7,5 % (tableau C.1)",
    (fb["k_min"], fb["k_max"], fb["eps_uk_min"]) == (1.08, None, 5.0) and (S.fiche(500, "A")["k_min"], S.fiche(500, "A")["eps_uk_min"]) == (1.05, 2.5)
    and (S.fiche(400, "C")["k_min"], S.fiche(400, "C")["k_max"], S.fiche(400, "C")["eps_uk_min"]) == (1.15, 1.35, 7.5))
chk("la fiche nomme sa source", "annexe C" in fb["source"])
ok, msg = refus(lambda: S.fiche(450, "B"), "nuance")
chk("refus d'une nuance hors des modules", ok, msg)
ok, msg = refus(lambda: S.fiche(500, "D"), "ductilité")
chk("refus d'une classe de ductilité inconnue", ok, msg)

# ============================================================================================ 5. mutations
print("\n=== 5 · Mutations : chaque contrôle rougit quand on abîme ce qu'il garde")


class Abime:
    """Un module avec des attributs remplacés, le temps d'un bloc ``with``."""

    def __init__(self, module, **attrs):
        self.m, self.attrs, self.garde = module, attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.garde[k] = getattr(self.m, k)
            setattr(self.m, k, v)
        return self.m

    def __exit__(self, *a):
        for k, v in self.garde.items():
            setattr(self.m, k, v)


with Abime(EC, eps_c2=lambda f: 2.0):
    chk("mutation « ε_c2 sans sa branche haute résistance » : le tableau 3.1 rougit", ecarts_3_1(EC))
with Abime(EC, eps_c1=lambda f: min(2.8, 0.7 * EC.fcm(f) ** 0.30)):
    chk("mutation « ε_c1 avec l'exposant 0,30 » : le tableau 3.1 rougit", ecarts_3_1(EC))
with Abime(EC, n_parabole=lambda f: 2.0):
    chk("mutation « n sans sa branche haute résistance » : le tableau 3.1 rougit", ecarts_3_1(EC))
with Abime(EC, eps_cu1=lambda f: 3.5 if f <= 50 else 2.8 + 27.0 * ((98.0 - EC.fcm(f)) / 100.0) ** 3):
    chk("mutation « ε_cu1 à la puissance 3 » : le tableau 3.1 rougit", ecarts_3_1(EC))
fbd_vrai = A.adherence
with Abime(A, adherence=lambda fck, phi, c: {**fbd_vrai(fck, phi, c), "fbd": fbd_vrai(fck, phi, c)["fbd"] * 2.5 / 2.25}):
    chk("mutation « f_bd avec 2,5 au lieu de 2,25 » : la NIT 260 (tableau 24) rougit", ecarts_nit(A, "lbd", NIT_24)[0])
with Abime(A, PHI_ETA2=100.0):
    chk("mutation « η2 = 1 au-delà de Ø 32 » : la NIT 260 (tableau 24, Ø 40 et 50) rougit", ecarts_nit(A, "lbd", NIT_24)[0])
with Abime(A, ALPHA6_MAX=1.4):
    chk("mutation « α6 borné à 1,4 » : la NIT 260 (tableau 25) rougit", ecarts_nit(A, "l0", NIT_25)[0])
with Abime(A, GAMMA_C=1.4):
    chk("mutation « γ_c = 1,4 » : la NIT 260 (tableau 24) rougit", ecarts_nit(A, "lbd", NIT_24)[0])
with Abime(A, GAMMA_S=1.0):
    chk("mutation « γ_s = 1,0 » : la NIT 260 (tableau 24) rougit", ecarts_nit(A, "lbd", NIT_24)[0])
with Abime(A, FCK_LIMITE_FBD=90.0):
    chk("mutation « plafond C60/75 retiré » : C70/85 n'a plus le f_bd de C60/75",
        abs(A.calculer({"phi": 16, "classe": "C70/85"})["adherence"]["fbd"] - A.calculer({"phi": 16, "classe": "C60/75"})["adherence"]["fbd"]) > 0.01)
with Abime(E, C_MIN_DUR={**E.C_MIN_DUR, "S4": (10, 20, 25, 30, 35, 40, 45)}):
    chk("mutation « S4/XC1 = 20 (l'écart D4 de l'écran Streamlit) » : le tableau 20 de la NIT rougit", E.C_MIN_DUR["S4"] != NIT_20_S4)
    chk("… l'exemple de la NIT (S2) ne bouge pas, une poutre XC1 C25/30 Ø 12 (S4) passe de 25 à 30 mm",
        E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12})["c_nom"] == 22
        and E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C25/30", "phi": 12})["c_nom"] == 30)
chk("… et hors mutation cette poutre XC1 C25/30 Ø 12 vaut S4, c_min,dur 15, c_nom 25 mm", E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C25/30", "phi": 12})["c_nom"] == 25)
with Abime(E, EXPOSITIONS=tuple((e[0], e[1], e[2], "C90/105", e[4]) if e[0] == "XC1" else e for e in E.EXPOSITIONS)):
    chk("mutation « seuil de résistance de XC1 inatteignable » : l'exemple de la NIT donne S3 (le contrôle « S2 » rougit) ; "
        "c_nom reste 22 parce que S2 et S3 valent 10 mm en XC1 — c'est la classe qui est gardée, pas seulement le résultat",
        E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12})["durabilite"]["classe_structurale"]["retenue"] == "S3"
        and E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12})["c_nom"] == 22
        and E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20})["durabilite"]["c_min_dur"] == 15)
chk("… et hors mutation la poutre XC1 C30/37 vaut S3, c_min,dur 10", E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20})["durabilite"]["c_min_dur"] == 10)
with Abime(S, DUCTILITE={**S.DUCTILITE, "B": {**S.DUCTILITE["B"], "eps_uk_min": 2.5}}):
    chk("mutation « ε_uk de la classe B = 2,5 » : la fiche B500B ne redonne plus le tableau C.1", S.fiche(500, "B")["eps_uk_min"] != 5.0)
# ---- le feu : chaque défaut d'hier, et chaque règle de lecture, rejoué comme une mutation du moteur
def mutant(module, fonction, avant: str, apres: str):
    """Une copie de ``fonction`` dont le source a ``avant`` remplacé par ``apres`` (le texte doit y être), compilée dans
    l'espace du module."""
    src = textwrap.dedent(inspect.getsource(fonction))
    assert avant in src, (fonction.__name__, avant)
    espace = dict(vars(module))
    exec(compile(src.replace(avant, apres, 1), f"<mutant {fonction.__name__}>", "exec"), espace)
    return espace[fonction.__name__]


def hier() -> list:
    """Les contrôles des cas d'hier et des règles de lecture, rejoués : ceux qui ne tiennent plus (un contrôle qui
    lève une erreur ne tient plus non plus)."""
    controles = (
        ("D1 : poteau R 90 μ 0,7 b 350 → 350/53", lambda: F.exigence("poteau", "R90", mu_fi=0.7, b=350)["combinaison"] == "350/53"),
        ("poteau b 450 → 450/40**", lambda: F.exigence("poteau", "R90", mu_fi=0.7, b=450)["combinaison"] == "450/40**"),
        ("poteau b 300 < 350 → non atteinte", lambda: F.exigence("poteau", "R90", mu_fi=0.7, b=300)["atteinte"] is False),
        ("poteau R 240 μ 0,7 → case vide", lambda: F.exigence("poteau", "R240", methode="tableau", mu_fi=0.7, b=600)["retenue"] is None),
        ("D2 : voile REI 180 → 210/50", lambda: F.exigence("voile", "R180", mu_fi=0.7, faces=1)["combinaison"] == "210/50"),
        ("D6 : dalle, aucun étrier retiré", lambda: E.calculer({"element": "dalle", "phi": 12, "phi_etrier": 8, "feu": {"r": "R60"}})["c_nom_feu"] == 14),
        ("D6 : voile, aucun étrier retiré", lambda: E.calculer({"element": "voile", "phi": 12, "phi_etrier": 8, "feu": {"r": "R120"}})["c_nom_feu"] == 29),
        ("a nominale : Δc_dev ne s'ajoute pas", lambda: len({E.calculer({"element": "poutre", "phi": 16, "phi_etrier": 6, "delta_dev": dd,
                                                                         "feu": {"r": "R120", "b": 240}})["c_nom_feu"] for dd in (10, 5, 0)}) == 1),
        ("dalle continue : colonne l_y/l_x ≤ 1,5", lambda: [F.exigence("dalle", "R90", portee=p)["a"] for p in ("une", "deux_15", "deux_20", "continue")] == [30, 15, 20, 15]),
        ("la formule n'est pas le défaut", lambda: F.exigence("poteau", "R90", b=350)["methode"] == "tableau"
         and E.calculer({"element": "poteau", "feu": {"r": "R90", "b": 350}})["feu"]["methode"] == "tableau"))
    out = []
    for nom, controle in controles:
        try:
            ok = controle()
        except Exception:  # noqa: BLE001 — un moteur abîmé qui lève : le contrôle ne tient plus
            ok = False
        if not ok:
            out.append(nom)
    return out


def formule_fausse() -> list:
    out = [(r_, b_, mu_, n_) for r_, b_, mu_, n_, brut_, a_ in MAIN_57
           for fo in (F.poteau_formule(r_, b_, b_, 3, mu_, n_),) if abs(fo["a_brut"] - brut_) > 0.05 or fo["a"] != a_]
    if F.poteau_formule("R30", 400, 400, 3, 0.2, 4)["a"] != 25:
        out.append("plancher de 25 mm")
    return out


DOMAINE = (("b′ < 200", ("R90", 190, 190, 3, 0.5, 4)), ("b′ > 450", ("R90", 460, 460, 3, 0.5, 4)), ("h > 1,5·b", ("R90", 300, 451, 3, 0.5, 4)),
           ("l_0,fi > 6 m", ("R90", 300, 300, 6.1, 0.5, 4)), ("μ_fi > 0,7", ("R90", 300, 300, 3, 0.71, 4)))


def domaine_ouvert() -> list:
    """Les bornes du domaine de la formule (5.7) qui ne sont plus refusées."""
    out = []
    for nom, args in DOMAINE:
        try:
            F.poteau_formule(*args)
            out.append(nom)
        except ValueError:
            pass
    return out


chk("hors mutation, les contrôles rejoués ci-dessous sont tous verts (transcription, recoupement, a_sd, cas d'hier, formule, domaine)",
    not ecarts_feu(F)[0] and not recoupement(F) and not a_sd_faux(F) and not hier() and not formule_fausse() and not domaine_ouvert(),
    (ecarts_feu(F)[0][:2], recoupement(F)[:2], a_sd_faux(F), hier(), formule_fausse(), domaine_ouvert()))
with Abime(F, POTEAU_5_2A={**F.POTEAU_5_2A, "R90": {**F.POTEAU_5_2A["R90"], "mu07": ((350, 43, ""), (450, 40, "**"))}}):
    chk("mutation « une case changée : tableau 5.2a, R 90, μ_fi = 0,7, 350/43 » : la transcription rougit sur cette case, le recoupement par la formule aussi, et D1",
        [e[:3] for e in ecarts_feu(F)[0]] == [("5.2a", "R90", "mu07")] and recoupement(F) and "D1 : poteau R 90 μ 0,7 b 350 → 350/53" in hier())
with Abime(F, POUTRE_5_6={**F.POUTRE_5_6, "R120": ((200, 45, ""), (300, 35, ""), (450, 30, ""), (500, 30, ""))}):
    chk("mutation « une case changée : tableau 5.6, R 120, 450/30 » : la transcription rougit sur cette case", [e[:3] for e in ecarts_feu(F)[0]] == [("5.6", "R120", "c3")])
with Abime(F, DALLE_5_8={**F.DALLE_5_8, "R180": {**F.DALLE_5_8["R180"], "une": (50, "")}}):
    chk("mutation « une case changée : tableau 5.8, REI 180, une direction, a = 50 » : la transcription rougit sur cette case", [e[:3] for e in ecarts_feu(F)[0]] == [("5.8", "R180", "une")])
with Abime(F, POTEAU_5_2A={**F.POTEAU_5_2A, "R90": {**F.POTEAU_5_2A["R90"], "mu07": ((350, 53, ""), (450, 40, ""))}}):
    chk("mutation « le repère ** de 450/40 retiré » : la transcription rougit, et le recoupement par la formule (R_n = 12) aussi", ecarts_feu(F)[0] and recoupement(F))
with Abime(F, _colonne_poteau=lambda exposition, mu: ("un_cote", "un côté") if exposition == "un" else ("mu02", "colonne décalée")):
    chk("mutation « colonne μ_fi décalée : le poteau à μ_fi = 0,7 lit la colonne μ_fi = 0,2 » — exactement le défaut D1 d'hier (R 90 : 200/31 au lieu de 350/53) : rougit",
        "D1 : poteau R 90 μ 0,7 b 350 → 350/53" in hier() and F.exigence("poteau", "R90", mu_fi=0.7, b=200)["combinaison"] == "200/31")
with Abime(F, POTEAU_5_2A={**F.POTEAU_5_2A, "R240": {**F.POTEAU_5_2A["R240"], "mu07": ((450, 40, ""),)}}):
    chk("mutation « R 240 à μ_fi = 0,7 : la variante 450/40 inventée par l'ancien écran » : la case vide rougit, la transcription et le recoupement aussi",
        "poteau R 240 μ 0,7 → case vide" in hier() and ecarts_feu(F)[0] and recoupement(F))
with Abime(F, VOILE_5_4={**F.VOILE_5_4, "R180": {**F.VOILE_5_4["R180"], "mu07_1": (210, 45, "")}}):
    chk("mutation « voile REI 180 : a = 45 » — le défaut D2 d'hier : rougit (cas d'hier et transcription)", "D2 : voile REI 180 → 210/50" in hier() and ecarts_feu(F)[0])
with Abime(E, ELEMENTS_A_ETRIER=("poutre", "poteau", "dalle", "voile")):
    chk("mutation « l'étrier retiré de a pour une dalle ou un voile » — le défaut D6 d'hier : rougit (14 → 6 mm, 29 → 21 mm)",
        hier() == ["D6 : dalle, aucun étrier retiré", "D6 : voile, aucun étrier retiré"])
vrai_feu = E.feu
with Abime(E, feu=lambda p: (lambda f: f and {**f, "c_nom": None if f["c_nom"] is None else f["c_nom"] + p["delta_dev"]})(vrai_feu(p))):
    chk("mutation « Δc_dev ajouté à l'enrobage du feu » (a n'est plus nominale) : rougit", "a nominale : Δc_dev ne s'ajoute pas" in hier())
with Abime(F, A_SD_ANGLE=0):
    chk("mutation « a_sd oublié » (a_sd = a) : les exemples de a_sd rougissent", len(a_sd_faux(F)) == 3, a_sd_faux(F))
with Abime(F, RANG_SANS_A_SD={"5.5": -1, "5.6": -1}):
    chk("mutation « a_sd majoré jusqu'à la DERNIÈRE colonne » (rang −1 : la lecture de la conception d'avant) : rougit sur b = 250 et b = 300 du 5.5 R 60 et b = 450 du 5.6 R 120",
        [x[:3] for x in a_sd_faux(F)] == [("simples", "R60", 250), ("simples", "R60", 300), ("continue", "R120", 450)], a_sd_faux(F))
with Abime(F, RANG_SANS_A_SD={"5.5": 1, "5.6": 0}):
    chk("mutation « a_sd majoré jusqu'à la colonne d'avant » (rangs 1 et 0) : rougit", a_sd_faux(F))
with Abime(F, _LECTEURS={**F._LECTEURS, "poutre": mutant(F, F._poutre, "b <= seuil", "b < seuil")}):
    chk("mutation « la borne de a_sd exclue » (b < largeur de la colonne) : rougit sur b = 200 (5.5 R 60), b = 250 (5.6 R 90) et b = 300 (5.6 R 120)",
        [x[:3] for x in a_sd_faux(F)] == [("simples", "R60", 200), ("continue", "R90", 250), ("continue", "R120", 300)], a_sd_faux(F))
with Abime(F, _retenir=lambda combinaisons, dimension: (0, None if dimension is None else True) if combinaisons else (None, False)):
    chk("mutation « la dimension réelle ignorée : toujours la première combinaison, jamais “non atteinte” » : rougit",
        "poteau b 450 → 450/40**" in hier() and "poteau b 300 < 350 → non atteinte" in hier())
with Abime(F, _retenir=lambda combinaisons, dimension: (len(combinaisons) - 1, None if dimension is None else True) if combinaisons else (None, False)):
    chk("mutation « la combinaison la moins sévère sans regarder la dimension » (non conservative) : rougit", "D1 : poteau R 90 μ 0,7 b 350 → 350/53" in hier())
with Abime(F, COLONNE_DALLE_CONTINUE="deux_20"):
    chk("mutation « dalle continue lue dans une autre colonne » : rougit", hier() == ["dalle continue : colonne l_y/l_x ≤ 1,5"])
with Abime(F, _LECTEURS={**F._LECTEURS, "poteau": lambda r, methode="formule", **k: F._poteau(r, methode=methode, **k)}):
    chk("mutation « la formule (5.7) devenue la méthode par défaut » : rougit", "la formule n'est pas le défaut" in hier())
with Abime(F, ALPHA_CC=1.0):
    chk("mutation « α_cc = 1,0 dans la formule » (la valeur de l'EN, pas la belge) : les quatre contrôles à la main rougissent (46,7 → 53,1 mm)",
        len(formule_fausse()) >= 4 and abs(F.poteau_formule("R90", 350, 350, 3, 0.7, 4)["a_brut"] - a_main(90, 350, 3.0, 0.7, 4, alpha_cc=1.0)) < 1e-9)
with Abime(F, poteau_formule=mutant(F, F.poteau_formule, "int(math.ceil(a_brut - 1e-9))", "int(math.floor(a_brut))")):
    chk("mutation « a arrondi au millimètre INFÉRIEUR » : les contrôles à la main rougissent (46 au lieu de 47)", len(formule_fausse()) >= 4)
with Abime(F, poteau_formule=mutant(F, F.poteau_formule, "a = max(int(A_FORMULE[0]), ", "a = max(0, ")):
    chk("mutation « le plancher de 25 mm retiré » : rougit", formule_fausse() == ["plancher de 25 mm"])
with Abime(F, poteau_formule=mutant(F, F.poteau_formule, "a_brut = 30.0 + r_a / 1.60", "a_brut = 30.0 + r_a / 1.50")):
    chk("mutation « 1,60 devenu 1,50 dans R_a » : les contrôles à la main rougissent", len(formule_fausse()) >= 4)
with Abime(F, _entree_formule=mutant(F, F._entree_formule, '"R_l": 9.60 * (5.0 - l0_calcul)', '"R_l": 9.60 * (4.0 - l0_calcul)')):
    chk("mutation « R_l = 9,60·(4 − l_0,fi) » : les contrôles à la main rougissent", len(formule_fausse()) >= 4)
with Abime(F, _entree_formule=mutant(F, F._entree_formule, '"R_b": 0.09 * b_prime', '"R_b": 0.09 * b')):
    chk("mutation « R_b calculé sur b au lieu de b′ » : un poteau 300 × 450 ne donne plus la valeur de b′ = 360 mm",
        abs(F.poteau_formule("R90", 300, 450, 3, 0.5, 4)["a_brut"] - a_main(90, 360.0, 3.0, 0.5, 4)) > 1.0)
for nom, attrs in (("b′ jusqu'à 10 m", {"B_PRIME": (0.0, 10000.0)}), ("h jusqu'à 15·b", {"H_SUR_B_MAX": 15.0}), ("l_0,fi jusqu'à 60 m", {"L0_FI": (2.0, 60.0)}),
                   ("μ_fi jusqu'à 7", {"MU_FORMULE": (0.2, 7.0)})):
    with Abime(F, **attrs):
        chk(f"mutation « le domaine de la formule ouvert : {nom} » : le refus correspondant ne tombe plus (le contrôle rougit)", domaine_ouvert())
chk("après les mutations du feu, tout est rétabli", not ecarts_feu(F)[0] and not recoupement(F) and not a_sd_faux(F) and not hier() and not formule_fausse() and not domaine_ouvert())
chk("après les mutations, tout est rétabli (NIT 24/25, tableau 3.1, exemple NIT, B500B)",
    not ecarts_nit(A, "lbd", NIT_24)[0] and not ecarts_nit(A, "l0", NIT_25)[0] and not ecarts_3_1(EC)
    and E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12})["c_nom"] == 22 and S.fiche(500, "B")["eps_uk_min"] == 5.0)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

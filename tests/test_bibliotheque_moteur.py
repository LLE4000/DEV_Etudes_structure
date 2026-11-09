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
3. Enrobage (§4.4.1, EN 1992-1-2 §5) : le tableau 4.4N (= NIT 260 tableau 20, ligne S4 transcrite), les modifications
   du tableau 4.3N (= NIT 260 tableau 21) dont l'exemple de la NIT (p. 49 : dalle C30/37, XC1, Ø 12 → S2, c_nom 22 mm),
   les bornes S1/S6, la classe imposée, le coulage contre le sol, le feu des poutres (tableau 5.5, a_sd) et des dalles
   (tableau 5.8), les poteaux et voiles non transcrits (dits, jamais inventés), refus.
4. Aciers : les exigences de l'annexe C (classes A, B, C), les six nuances des modules.
5. Mutations : abîmer un coefficient, une formule ou une case fait rougir le contrôle qui le garde.

    python3 tests/test_bibliotheque_moteur.py
"""
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton.bibliotheque import aciers as S  # noqa: E402
from beton.bibliotheque import ancrages as A  # noqa: E402
from beton.bibliotheque import enrobage as E  # noqa: E402
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
print("\n=== 3 · Enrobage : tableau 4.4N (NIT 260 T20), modifications 4.3N (T21), exemple de la NIT, feu")
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

# feu — tableau 5.5 (poutres sur appuis simples) et 5.8 (dalles)
pf = E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60", "b_min": 250}})
f = pf["feu"]
chk("poutre R60, b = 250 : ligne (200, 30) du tableau 5.5, a_sd = 40 (b < 300), c_feu = 40 − 10 − 8 = 22 ; la durabilité (40) gouverne",
    f["ligne"] == {"b_min": 200, "a": 30} and f["a_sd"] == 40 and f["angle"] and f["c_nom"] == 22 and pf["c_nom"] == 40 and pf["gouvernant"] == "durabilite")
pf2 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 16, "phi_etrier": 6, "feu": {"r": "R120", "b_min": 240}})
f2 = pf2["feu"]
chk("poutre R120, b = 240, Ø 16 / étrier 6 : a = 60, a_sd = 70, c_feu = 70 − 8 − 6 = 56 > c_nom,dur 26 : le feu gouverne",
    f2["ligne"] == {"b_min": 240, "a": 60} and f2["a_sd"] == 70 and f2["c_nom"] == 56 and pf2["c_nom"] == 56 and pf2["gouvernant"] == "feu")
pf3 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R90", "b_min": 400}})
chk("poutre R90, b = 400 (dernière colonne) : a = 35, pas de majoration a_sd", pf3["feu"]["a"] == 35 and not pf3["feu"]["angle"] and pf3["feu"]["a_sd"] == 35)
pf4 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60", "b_min": 100}})
chk("poutre R60, b = 100 < 120 : la méthode tabulée ne couvre pas, avertissement « poutre_etroite »", pf4["feu"]["trop_etroite"] and any(a["code"] == "poutre_etroite" for a in pf4["avertissements"]))
pf5 = E.calculer({"element": "poutre", "exposition": "XC1", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60"}})
chk("poutre R60 sans largeur : la ligne la plus étroite (120, 40) et a_sd = 50", pf5["feu"]["ligne"] == {"b_min": 120, "a": 40} and pf5["feu"]["a_sd"] == 50)
pfd = E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12, "phi_etrier": 0, "feu": {"r": "R60"}})
chk("dalle R60 (tableau 5.8) : h_s ≥ 80, a = 20, c_feu = 20 − 6 = 14 ; durabilité 22 gouverne", pfd["feu"]["h_min"] == 80 and pfd["feu"]["a"] == 20 and pfd["feu"]["c_nom"] == 14 and pfd["c_nom"] == 22)
pfd2 = E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12, "phi_etrier": 0, "feu": {"r": "R180"}})
chk("dalle R180 : a = 55, c_feu = 49 > 22 : le feu gouverne", pfd2["feu"]["a"] == 55 and pfd2["c_nom"] == 49 and pfd2["gouvernant"] == "feu")
pfp = E.calculer({"element": "poteau", "exposition": "XC1", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60"}})
chk("poteau : le tableau 5.2a n'est pas transcrit — c_feu absent, avertissement, c_nom = durabilité (aucune valeur inventée)",
    pfp["feu"]["transcrit"] is False and pfp["c_nom_feu"] is None and any(a["code"] == "feu_a_transcrire" for a in pfp["avertissements"]) and pfp["c_nom"] == pfp["c_nom_dur"])
chk("tableau 5.5 : six durées R, largeurs croissantes et a décroissant sur chaque ligne ; a croît avec R à largeur égale",
    tuple(E.FEU_POUTRE) == E.R_DUREES and all(all(x[0] < y[0] and x[1] >= y[1] for x, y in zip(l, l[1:])) for l in E.FEU_POUTRE.values())
    and all(E.FEU_POUTRE[a][0][1] <= E.FEU_POUTRE[b][0][1] for a, b in zip(E.R_DUREES, E.R_DUREES[1:])))
chk("tableau 5.8 : h_s et a croissent avec R (60/10 … 175/65)", tuple(E.FEU_DALLE[r] for r in E.R_DUREES) == ((60, 10), (80, 20), (100, 30), (120, 40), (150, 55), (175, 65)))
chk("les tableaux de l'écran : 4.4N (6 × 7 + libellés belges), feu poutre transcrit, feu poteau non transcrit",
    E.tableau_c_min_dur()["belges"] == list(NIT_20_BELGES) and E.tableau_feu("poutre")["transcrit"] and not E.tableau_feu("poteau")["transcrit"])
chk("les termes de l'enrobage finissent par c_nom, chacun avec une règle ; les termes normatifs citent leur paragraphe ou leur tableau",
    pf["termes"][-1]["cle"] == "cnom" and all(t["regle"] for t in pf["termes"])
    and all("§" in t["regle"] or "tableau" in t["regle"] for t in pf["termes"] if t["cle"] in ("cmin_b", "S", "cmin_dur", "cmin", "ddev", "cnom_dur", "a", "asd")))
for nom, params, mot in (("exposition XF1", {"exposition": "XF1"}, "n'est pas une classe"), ("élément « semelle »", {"element": "semelle"}, "élément"),
                         ("durée 75 ans", {"duree": 75}, "50 ou 100"), ("Δc_dev 7", {"delta_dev": 7}, "10, 5 ou 0"), ("classe structurale S7", {"classe_structurale": "S7"}, "S1 à S6"),
                         ("feu R45", {"feu": {"r": "R45"}}, "résistance au feu"), ("Ø 3", {"phi": 3}, "entre 6 et 60"), ("contre sol « oui »", {"contre_sol": "oui"}, "coulé contre le sol"),
                         ("classe C30", {"classe": "C30"}, "illisible")):
    ok, msg = refus(lambda p=params: E.calculer(p), mot)
    chk(f"refus « {nom} » en français", ok, msg)

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
with Abime(E, FEU_POUTRE={**E.FEU_POUTRE, "R60": ((120, 40), (160, 35), (200, 35), (300, 25))}):
    chk("mutation « tableau 5.5 R60 (200, 35) » : le c_feu de la poutre R60 b 250 passe de 22 à 27",
        E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60", "b_min": 250}})["feu"]["c_nom"] == 27)
with Abime(E, A_SD_ANGLE=0):
    chk("mutation « a_sd = a » : la barre d'angle perd ses 10 mm (c_feu 12 au lieu de 22)",
        E.calculer({"element": "poutre", "exposition": "XC4", "classe": "C30/37", "phi": 20, "phi_etrier": 8, "feu": {"r": "R60", "b_min": 250}})["feu"]["c_nom"] == 12)
with Abime(S, DUCTILITE={**S.DUCTILITE, "B": {**S.DUCTILITE["B"], "eps_uk_min": 2.5}}):
    chk("mutation « ε_uk de la classe B = 2,5 » : la fiche B500B ne redonne plus le tableau C.1", S.fiche(500, "B")["eps_uk_min"] != 5.0)
chk("après les mutations, tout est rétabli (NIT 24/25, tableau 3.1, exemple NIT, B500B)",
    not ecarts_nit(A, "lbd", NIT_24)[0] and not ecarts_nit(A, "l0", NIT_25)[0] and not ecarts_3_1(EC)
    and E.calculer({"element": "dalle", "exposition": "XC1", "classe": "C30/37", "phi": 12})["c_nom"] == 22 and S.fiche(500, "B")["eps_uk_min"] == 5.0)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

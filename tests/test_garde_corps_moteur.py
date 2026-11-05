# -*- coding: utf-8 -*-
"""Tests du moteur « Garde-corps » (acier/garde_corps, sans Streamlit). Chaque garantie
rougit si on la retire :

  1. GÉOMÉTRIE : répartition régulière, entraxes saisis (un poteau central à deux distances
     différentes), longueurs tributaires, montant d'angle d'un L et d'un U, extrémité ancrée
     dans le mur (le mur reprend la demi-travée), phrase.
  2. CHARGES : catégories du Tableau 6.12, valeur libre, mode P direct, remplissage.
  3. MÉCANIQUE : console (M = P·H, V = P, f = PH³/3EI), poutre (qL²/8, QL/4, 5qL⁴/384EI,
     somme / max), vérification de section (σ = M/W + N/A, τ = V/A_v, flèche), limites
     ELS et Eurocode (γ_Q sur les contraintes, pas sur la flèche), statuts.
  4. BIBLIOTHÈQUE : creux du catalogue, plats et ronds de stock, sections libres (TUBE,
     CAISSON, PLAT, ROND) par la même fonction que le catalogue, axe fort = sens de h,
     aire de cisaillement entière pour un plein, spécifications, refus.
  5. VÉRIFICATION COMPLÈTE : montant gouvernant, angle vérifié dans les deux orientations,
     lisse biaxiale, barreau (deux cas), lisses intermédiaires, panneau, masse, messages.
  6. FIXATION : cas de soudure (C6 tube, C3 plat, C8 libre) et ses efforts, platine
     boulonnée (bras de levier, Z, F_t, F_v, boulon EN 1993-1-8, cheville ETA, interaction,
     ELS /γ_F, flexion de la platine, épaisseur requise, refus), système mur (console,
     traction des chevilles : enveloppe des deux modèles), scellé.
  7. CANDIDATS : tri par masse, le plus léger qui tient, filtre de groupe, application.
  8. ÉTAT ET JSON : nouveau cas, normalisation d'un fichier incomplet, aller-retour.

Lancement : python3 tests/test_garde_corps_moteur.py (depuis la racine).
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.garde_corps import bibliotheque as BI  # noqa: E402
from acier.garde_corps import charges as CH  # noqa: E402
from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import fixation as FX  # noqa: E402
from acier.garde_corps import geometrie as GE  # noqa: E402
from acier.garde_corps import membres as ME  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cas(**kw):
    c = ET.nouveau_cas()
    for k, v in kw.items():
        if isinstance(v, dict):
            c[k] = {**c[k], **v}
        else:
            c[k] = v
    return c


# ================================================================
print("\n=== 1. Géométrie ===")
chk("répartition régulière : 3600 à l'entraxe maxi 1000 → 4 travées de 900", GE.entraxes_reguliers(3600, 1000) == [900.0] * 4)
chk("3600 à 1200 → 3 travées exactes ; 3601 → 4 travées", len(GE.entraxes_reguliers(3600, 1200)) == 3 and len(GE.entraxes_reguliers(3601, 1200)) == 4)
g = GE.geometrie_defaut()
g["troncons"][0] = {"L": 3600, "s_max": 1000, "entraxes": []}
a = GE.analyser(g)
chk("droit, 4 travées : 5 montants, tributaires 450 / 900 / 900 / 900 / 450, s_max 900, L 3600",
    a["n_montants"] == 5 and [m["trib1"] for m in a["montants"]] == [450, 900, 900, 900, 450] and a["s_max"] == 900 and a["L_total"] == 3600)
g["troncons"][0]["entraxes"] = [1200, 900]
a = GE.analyser(g)
chk("entraxes saisis 1200 + 900 : trois montants à x = 0, 1200, 2100, tributaires 600 / 1050 / 450 (le central n'est pas symétrique)",
    [m["x"] for m in a["montants"]] == [0, 1200, 2100] and [m["trib1"] for m in a["montants"]] == [600, 1050, 450] and a["s_max"] == 1200)
chk("un montant courant n'est pas d'angle et n'a pas de seconde direction", all(not m["coin"] and m["trib2"] == 0 for m in a["montants"]))
g["fin"] = "mur"
a = GE.analyser(g)
chk("fin ancrée dans le mur : plus de dernier montant, le mur reprend la demi-travée 450",
    a["n_montants"] == 2 and a["reprises_mur"] == [{"ou": "fin", "x": 2100, "trib": 450.0}] and a["montants"][-1]["trib1"] == 1050)
g["fin"] = "montant"
g["forme"] = "L"
g["troncons"][1] = {"L": 2000, "s_max": 1000, "entraxes": []}
a = GE.analyser(g)
coin = [m for m in a["montants"] if m["coin"]]
chk("en L : un montant d'angle à la jonction, une demi-travée de chaque tronçon dans deux directions (450 et 500)",
    len(coin) == 1 and coin[0]["x"] == 2100 and coin[0]["trib1"] == 450 and coin[0]["trib2"] == 500 and a["n_montants"] == 5)
g["forme"] = "U"
g["troncons"][2] = {"L": 1500, "s_max": 1000, "entraxes": []}
a = GE.analyser(g)
chk("en U : deux montants d'angle, longueur développée 5600", a["n_coins"] == 2 and a["L_total"] == 5600)
chk("la phrase nomme la forme, les travées et les angles", "en U" in GE.phrase(g, a) and "d'angle" in GE.phrase(g, a))
g2 = GE.geometrie_defaut()
g2["troncons"][0] = {"L": 1000, "s_max": 1000, "entraxes": []}
g2["debut"] = g2["fin"] = "mur"
chk("une seule travée ancrée aux deux bouts : refus (aucun montant)", any("Refus" in m for m in GE.analyser(g2)["messages"]))

# ================================================================
print("\n=== 2. Charges ===")
chk("catégorie A : 0,5 kN/m ; C2-C4 : 1,0 ; C5 : 3,0 ; E : 2,0 (valeurs recommandées du Tableau 6.12)",
    CH.qh_de_categorie("A") == 0.5 and CH.qh_de_categorie("C2-C4") == 1.0 and CH.qh_de_categorie("C5") == 3.0 and CH.qh_de_categorie("E") == 2.0)
c = CH.normaliser({"categorie": "D", "qh": 7.0})
chk("une catégorie impose sa valeur (D : 1,0 malgré 7 saisi) ; la valeur libre la garde", c["qh"] == 1.0 and CH.normaliser({"categorie": "perso", "qh": 0.8})["qh"] == 0.8)
chk("défauts : q_v 0,5 kN/m, q_r 0,5 kN/m², Q_r 0,5 kN, mode q", c["qv"] == 0.5 and c["qr"] == 0.5 and c["Qr"] == 0.5 and c["mode"] == "q")
chk("l'origine de q_h dit la catégorie et le tableau ; le mode P dit la saisie directe",
    "Tab. 6.12" in CH.origine_qh(c) and "saisie directement" in CH.origine_qh(CH.normaliser({"mode": "P"})))

# ================================================================
print("\n=== 3. Mécanique ===")
E = 210000.0
c = ME.console(450.0, 0.0, 450.0, 1100.0, E, 115857.0, 115857.0)
chk("console : M = P·H = 495 000 N·mm, V = P, f = PH³/(3EI) = 8,206 mm", pres(c["M1"], 495000.0) and c["V1"] == 450.0 and pres(c["f"], 450 * 1100 ** 3 / (3 * E * 115857)))
p = ME.poutre(0.5, 300.0, 900.0, E, 76200.0)
chk("poutre : qL²/8 + QL/4 = 50 625 + 67 500, V = qL/2 + Q/2, f = 5qL⁴/384EI + QL³/48EI",
    pres(p["M"], 0.5 * 900 ** 2 / 8 + 300 * 900 / 4) and pres(p["V"], 225 + 150) and pres(p["f"], 5 * 0.5 * 900 ** 4 / (384 * E * 76200) + 300 * 900 ** 3 / (48 * E * 76200)))
pm = ME.poutre(0.5, 300.0, 900.0, E, 76200.0, "max")
chk("combinaison « max » : le plus grand des deux cas", pres(pm["M"], 67500.0) and pres(pm["V"], 225.0))
sec = BI.section({"nom": "CHS 48.3x3.2"})
meth = ME.methode_defaut()
v = ME.verifier_section(sec, 495000.0, 0.0, 450.0, 450.0, 0.0, 8.206, 1100.0, meth, 235.0)
chk("vérification ELS : σ = M/W + N/A = 104,2, τ = V/A_v = 1,56, flèche/(L/100) ; σ_adm 160, τ_adm 90 ; η = max",
    pres(v["sigma"], 495000 / (sec["Wel_y"] * 1e3) + 450 / (sec["A"] * 1e2)) and pres(v["tau"], 450 / (sec["Avz"] * 1e2))
    and pres(v["fleche_lim"], 11.0) and v["lim_sigma"] == 160 and v["lim_tau"] == 90 and pres(v["taux_max"], 8.206 / 11.0) and v["critere"] == "fleche")
me = {**meth, "mode": "eurocode"}
ve = ME.verifier_section(sec, 495000.0, 0.0, 450.0, 450.0, 0.0, 8.206, 1100.0, me, 235.0)
chk("Eurocode : γ_Q = 1,5 sur σ et τ, limites fy/γM0 = 235 et fy/√3, la flèche reste de service",
    pres(ve["sigma"], 1.5 * v["sigma"]) and pres(ve["tau"], 1.5 * v["tau"]) and pres(ve["lim_sigma"], 235.0) and pres(ve["lim_tau"], 235 / math.sqrt(3))
    and pres(ve["fleche"], v["fleche"]) and pres(ve["fleche_lim"], v["fleche_lim"]))
chk("flèche en mm : la limite est la valeur saisie", pres(ME.fleche_limite({**meth, "fleche_mode": "mm", "fleche_val": 25.0}, 1100.0), 25.0))
chk("statuts : ≤ 0,85 ok, ≤ 1 attention, > 1 nok, None refus", ME.statut(0.85) == "ok" and ME.statut(0.9) == "attention" and ME.statut(1.01) == "nok" and ME.statut(None) == "refus")

# ================================================================
print("\n=== 4. Bibliothèque ===")
std = BI.standards()
n_creux = sum(1 for r in CAT.profils() if r["famille"] in BI.FAMILLES_CREUX)
chk(f"tous les creux du catalogue ({n_creux}) + plats et ronds de stock = {len(std)} sections standard",
    sum(1 for s in std.values() if s["groupe"] == "creux") == n_creux and len(std) == n_creux + len(BI.PLATS_STOCK) + len(BI.RONDS_STOCK))
rec = CAT.par_nom()["RHS 60x40x3"]
s = BI.section({"nom": "RHS 60x40x3"})
chk("un creux du catalogue garde ses nombres (Wel_y, Avz, masse) et h = 60 est l'axe fort",
    pres(s["Wel_y"], rec["Wel_y"]) and pres(s["Avz"], rec["Avz"]) and pres(s["masse"], rec["masse"]) and s["h"] == 60 and s["Wel_y"] > s["Wel_z"])
pl = BI.section({"nom": "Plat 50x10"})
chk("Plat 50×10 : A = 5 cm², W_el,y = b·h²/6 = 4,167 cm³, I_y = 10,42 cm⁴, aire de cisaillement entière, standard",
    pres(pl["A"], 5.0) and pres(pl["Wel_y"], 10 * 50 ** 2 / 6 / 1e3) and pres(pl["Iy"], 10 * 50 ** 3 / 12 / 1e4) and pres(pl["Avz"], 5.0) and pl["standard"])
ro = BI.section({"nom": "Rond Ø12"})
chk("Rond Ø12 : A = π·6², W = π·12³/32", pres(ro["A"], math.pi * 36 / 100, 1e-4) and pres(ro["Wel_y"], math.pi * 12 ** 3 / 32 / 1e3, 1e-3))
tl = BI.section({"famille": "TUBE", "dims": {"d": 48.3, "t": 3.2}})
chk("section libre TUBE 48,3 × 3,2 : les mêmes nombres que le CHS 48.3x3.2 du catalogue, non standard",
    pres(tl["A"], sec["A"], 1e-4) and pres(tl["Wel_y"], sec["Wel_y"], 1e-4) and not tl["standard"] and tl["nom"] == "Tube Ø48.3x3.2")
cs = BI.section({"famille": "CAISSON", "dims": {"h": 60, "b": 40, "t": 3}})
chk("CAISSON 60×40×3 à coins vifs : A = 60·40 − 54·34 = 5,64 cm²", pres(cs["A"], (60 * 40 - 54 * 34) / 100))
chk("spécification : standard → {nom}, libre → {famille, dims}", BI.spec_de(s) == {"nom": "RHS 60x40x3"} and BI.spec_de(tl) == {"famille": "TUBE", "dims": {"d": 48.3, "t": 3.2}})
try:
    BI.section({"nom": "IPE 200"})
    chk("une section hors bibliothèque est refusée", False)
except ValueError:
    chk("une section hors bibliothèque est refusée", True)
try:
    BI.section({"famille": "TUBE", "dims": {"d": 20, "t": 12}})
    chk("un tube de paroi trop épaisse est refusé", False)
except ValueError:
    chk("un tube de paroi trop épaisse est refusé", True)
cat = BI.catalogue(["creux"], masse_max=10.0)
chk("catalogue filtré : creux seulement, ≤ 10 kg/m, par masse croissante",
    all(BI.groupe_de(x) == "creux" and x["masse"] <= 10 for x in cat) and [x["masse"] for x in cat] == sorted(x["masse"] for x in cat) and len(cat) > 50)

# ================================================================
print("\n=== 5. Vérification complète ===")
c0 = ET.nouveau_cas()
r = ET.verifier(c0)
chk("cas par défaut : conforme, 5 montants, le gouvernant est un montant courant (M2) par la flèche",
    r["statut"] == "ok" and len(r["montants"]) == 5 and r["gouvernant"]["cle"] == "montant" and r["montant"]["nom"] == "M2" and r["montant"]["verif"]["critere"] == "fleche")
m2 = r["montant"]["verif"]
chk("montant M2 : P = q_h·900 = 0,45 kN, N = 0,45 kN, M = 0,495 kN·m, σ = 104,2 N/mm², f = 8,21 mm ≤ 11",
    pres(m2["P1"], 450.0) and pres(m2["N"], 450.0) and pres(m2["M1"], 495000.0) and pres(m2["sigma"], 104.2, 1e-3) and pres(m2["fleche"], 8.206, 1e-3) and pres(m2["fleche_lim"], 11.0))
li = r["lisse"]["verif"]
chk("lisse haute : portée 900, M_h = M_v = qL²/8 = 0,0506 kN·m, σ = M_v/W_y + M_h/W_z, flèche résultante √(f_h² + f_v²)",
    pres(li["L"], 900.0) and pres(li["Mh"], 0.5 * 900 ** 2 / 8) and pres(li["Mv"], li["Mh"]) and pres(li["fleche"], math.hypot(li["fh"], li["fv"])))
b = r["remplissage"]["verif"]
chk("barreaux Plat 30×8 sur 900 à 110 : le cas Q_r = 0,5 kN gouverne (M = QL/4 = 0,1125 kN·m, σ = 93,75) devant la pression q_r·e",
    b["cas"] == "Q" and pres(b["M1"], 500 * 900 / 4) and pres(b["sigma"], 93.75, 1e-3) and b["verif_q"]["taux_max"] < b["verif_Q"]["taux_max"])
chk("les éléments : montant, lisse, barreau, soudure, ancrage, platine ; une masse indicative",
    [e["cle"] for e in r["elements"]] == ["montant", "lisse", "barreau", "soudure", "ancrage", "platine"] and 50 < r["masse"] < 150)
cL = cas(geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}})
rL = ET.verifier(cL)
g = rL["montant"]["verif"]
chk("en L avec un RHS 60×40 : le montant d'angle gouverne, vérifié dans l'orientation la pire (P selon b)",
    g["coin"] and g.get("orientation_inversee") and g["P2"] > 0 and g["taux_max"] > max(v["taux_max"] for v in rL["montants"] if not v["coin"]))
cP = cas(charges={"mode": "P", "P": 1.0})
rP = ET.verifier(cP)
chk("mode P direct : chaque montant reçoit 1 kN, la lisse l'équivalent P/s = 1,11 kN/m",
    all(pres(v["P1"], 1000.0) for v in rP["montants"]) and pres(rP["lisse"]["verif"]["qh"], 1000 / 900))
cI = cas(remplissage={"type": "lisses", "n_lisses": 2, "entraxe_lisses": 330.0, "section_lisses": {"nom": "CHS 26.9x3.2"}})
rI = ET.verifier(cI)
chk("lisses intermédiaires : portée s_max, charge q_r·e = 0,165 kN/m ou Q_r", rI["remplissage"]["verif"]["L"] == 900 and pres(rI["remplissage"]["verif"]["q"], 0.5 * 330 / 1e3)
    and any(e["cle"] == "lisse_inter" for e in rI["elements"]))
rPan = ET.verifier(cas(remplissage={"type": "panneau"}))
chk("panneau : pas d'élément vérifié, un message pour le fournisseur", not any(e["cle"] in ("barreau", "lisse_inter") for e in rPan["elements"]) and any("fournisseur" in m for m in rPan["messages"]))
cH = cas(geometrie={"H": 1300.0})
chk("H > 1,20 m : message (bras de levier pris tel quel, côté sûr)", any("1,20 m" in m or "1.20 m" in m for m in ET.verifier(cH)["messages"]))
cM = cas(geometrie={"fin": "mur"})
chk("fin ancrée dans le mur : le message donne l'effort repris par le mur", any("mur" in m and "kN" in m for m in ET.verifier(cM)["messages"]))

# ================================================================
print("\n=== 6. Fixation ===")
fx = r["fixation"]
sres = fx["soudure"]["res"]
chk("CHS du catalogue : cas C6 tube sur platine, efforts N = −0,45 (compression), V_z = 0,45, M_y = −0,495 kN·m",
    sres["config"] == "C6" and pres(fx["soudure"]["cas"]["efforts"]["N"], -0.45) and pres(fx["soudure"]["cas"]["efforts"]["Vz"], 0.45)
    and pres(fx["soudure"]["cas"]["efforts"]["My"], -0.495))
chk("la soudure est jugée par la méthode ELS du bureau (k = 100) quand le garde-corps l'est", sres["methode"].mode == "els" and sres["methode"].k == 100.0 and fx["soudure"]["taux"] > 0.5)
chk("le groupe de cordons fait le tour du tube : A_w ≈ π·d·a", pres(sres["proprietes"]["A_w"], math.pi * 48.3 * 4.0, 0.02))
rPl = ET.verifier(cas(montant={"section": {"nom": "Plat 50x10"}}))
cs3 = rPl["fixation"]["soudure"]["cas"]
chk("plat : cas C3 console fléchie, h = 50, t = 10, e = H = 1100, V = P, N = −N", sres is not None and cs3["config"] == "C3" and cs3["params"]["h"] == 50 and cs3["params"]["t"] == 10
    and cs3["params"]["e"] == 1100 and pres(cs3["efforts"]["V"], 0.45) and pres(cs3["efforts"]["N"], -0.45))
rT = ET.verifier(cas(montant={"section": {"famille": "TUBE", "dims": {"d": 48.3, "t": 3.2}}}))
cs8 = rT["fixation"]["soudure"]["cas"]
chk("tube libre : cas C8, un arc de 360° de rayon d/2, V_y = P, M_x = −P·H ; même taux que le C6 du CHS (±5 %)",
    cs8["config"] == "C8" and cs8["params"]["elements"][0]["type"] == "arc" and pres(cs8["params"]["elements"][0]["r"], 24.15)
    and pres(cs8["efforts"]["Vy"], 0.45) and pres(cs8["efforts"]["Mx"], -0.495) and pres(rT["fixation"]["soudure"]["taux"], fx["soudure"]["taux"], 0.05))
rC = ET.verifier(cas(montant={"section": {"famille": "CAISSON", "dims": {"h": 60, "b": 40, "t": 3}}}))
chk("caisson libre : cas C8, quatre segments sur le contour 40 × 60, A_w = 200·a", rC["fixation"]["soudure"]["cas"]["config"] == "C8"
    and len(rC["fixation"]["soudure"]["cas"]["params"]["elements"]) == 4 and pres(rC["fixation"]["soudure"]["res"]["proprietes"]["A_w"], 200 * 4.0, 0.05))
pl = fx["platine"]
chk("platine : z = e_x/2 + h_p/2 = 74,15, Z = (M − N·h_p/2)/z = 6,53 kN, F_t = Z/2, F_v = V/4",
    pres(pl["z"], 74.15) and pres(pl["Z"], (495000 - 450 * 48.3 / 2) / 74.15) and pres(pl["Ft"], pl["Z"] / 2) and pres(pl["Fv"], 450 / 4))
chk("cheville ETA en ELS : N_Rd/γ_F = 9/1,5 = 6 kN, interaction (β_N)^1,5 + (β_V)^1,5",
    pres(pl["FtRd"], 6000.0) and pres(pl["inter"], (pl["Ft"] / 6000) ** 1.5 + (pl["Fv"] / 8000) ** 1.5))
chk("flexion de la platine : m = e_x/2 − h_p/2 − 0,8·a·√2 = 21,3, b_eff = min(l_p ; 2·min(2πm ; 4m + 1,25e)) = 150, σ_p = 6·Z·m/(b·t²), t_req",
    pres(pl["m"], 50 - 24.15 - 0.8 * 4 * math.sqrt(2)) and pres(pl["b_eff"], 150.0) and pres(pl["sigma_p"], 6 * pl["Z"] * pl["m"] / (150 * 100))
    and pres(pl["tp_req"], math.sqrt(6 * pl["Z"] * pl["m"] / (150 * 160))))
rB = ET.verifier(cas(fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon", "boulon": "M12", "classe": "8.8"}}))
pb = rB["fixation"]["platine"]
chk("boulon M12 8.8 (Tab. 3.4) : F_t,Rd = 0,9·800·84,3/1,25 = 48,5 kN, /1,5 en ELS = 32,4 ; interaction F_v/F_v,Rd + F_t/(1,4·F_t,Rd)",
    pres(pb["FtRd"], 0.9 * 800 * 84.3 / 1.25 / 1.5) and pres(pb["FvRd"], 0.6 * 800 * 84.3 / 1.25 / 1.5) and pres(pb["inter"], pb["Fv"] / pb["FvRd"] + pb["Ft"] / (1.4 * pb["FtRd"])))
rE = ET.verifier(cas(methode={"mode": "eurocode"}, fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}}))
pe = rE["fixation"]["platine"]
chk("Eurocode : efforts × 1,5 et résistances entières (F_t,Rd = 48,5 kN) ; la soudure en directionnelle",
    pres(pe["M"], 1.5 * 495000) and pres(pe["FtRd"], 0.9 * 800 * 84.3 / 1.25) and rE["fixation"]["soudure"]["res"]["methode"].mode == "eurocode")
rX = ET.verifier(cas(fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "ex": 40.0}}))
chk("rangées trop proches (e_x ≤ h_p) : refus de la platine, message", rX["fixation"]["platine"]["taux_max"] is None and any("Refus" in m for m in rX["messages"]))
rS = ET.verifier(cas(fixation={"type": "soudee"}))
chk("montant soudé sur un plat existant : soudure vérifiée, pas de platine", rS["fixation"]["soudure"] is not None and rS["fixation"]["platine"] is None)
rZ = ET.verifier(cas(fixation={"type": "scellee"}))
chk("montant scellé : aucune fixation vérifiée, message", rZ["fixation"]["soudure"] is None and any("scellé" in m for m in rZ["messages"]))
cW = cas(geometrie={"systeme": "mur"})
rW = ET.verifier(cW)
cw = rW["console"]["verif"]
chk("système mur : la lisse est principale, les consoles portent R_v = q_v·s, R_h = q_h·s, M = R_v·e, σ = M/W + R_h/A",
    rW["lisse"]["role"] == "principal" and rW["montant"] is None and pres(cw["Rv"], 450.0) and pres(cw["Rh"], 450.0) and pres(cw["M1"], 450 * 100)
    and pres(cw["sigma"], 45000 / (rW["console"]["section"]["Wel_y"] * 1e3) + 450 / (rW["console"]["section"]["A"] * 1e2)))
pw = rW["fixation"]["platine"]
chk("console au mur : la platine est tirée (N_t = R_h) ; Z = enveloppe (M + N_t·h_p/2)/z et M/e_x + N_t/2",
    pw["N"] < 0 and pres(pw["Z"], max((45000 + 450 * 25) / 75.0, 45000 / 100 + 225)))
chk("console : cas de soudure C3 avec N = +R_h (tire la console hors du mur) et V = R_v", rW["fixation"]["soudure"]["cas"]["config"] == "C3"
    and pres(rW["fixation"]["soudure"]["cas"]["efforts"]["N"], 0.45) and pres(rW["fixation"]["soudure"]["cas"]["efforts"]["V"], 0.45))

# ================================================================
print("\n=== 7. Candidats ===")
cands = ET.candidats(c0, ["creux"])
chk("les candidats creux, par masse croissante, chacun avec son taux", len(cands) > 100 and [x["masse"] for x in cands] == sorted(x["masse"] for x in cands)
    and all(0 <= x["taux"] for x in cands))
best = ET.plus_leger(cands)
chk("le plus léger qui tient est le premier dont η ≤ 1 : CHS 48.3x2.5 (η 91 %)", best["section"]["nom"] == "CHS 48.3x2.5" and best["taux"] <= 1.0
    and all(x["taux"] > 1.0 for x in cands[:cands.index(best)]))
chk("le CHS 48.3x3.2 du cas est un candidat au même taux que la vérification", any(x["section"]["nom"] == "CHS 48.3x3.2" and pres(x["taux"], r["taux_max"]) for x in cands))
c1 = ET.nouveau_cas()
ET.appliquer_section(c1, best["section"])
chk("appliquer le candidat écrit sa spécification dans le montant", c1["montant"]["section"] == {"nom": "CHS 48.3x2.5"})
cands_mur = ET.candidats(cW, ["creux"])
chk("système mur : les candidats sont jugés comme lisse (portée s_max)", ET.element_principal(cW) == "lisse" and len(cands_mur) > 100 and cands_mur[0]["taux"] > 0)

# ================================================================
print("\n=== 8. État et JSON ===")
e0 = ET.nouvel_etat()
txt = ET.vers_json(e0)
e1 = ET.depuis_json(txt)
chk("aller-retour JSON : un cas identique", e1["cas"][0] == e0["cas"][0] and e1["actif"] == 0)
vieux = json.dumps({"cas": [{"nom": "ancien", "geometrie": {"forme": "L", "troncons": [{"L": 3000}]}, "charges": {"categorie": "D"}}]})
e2 = ET.depuis_json(vieux)
c2 = e2["cas"][0]
chk("un fichier incomplet est complété : trois tronçons, q_h de la catégorie D, sections et fixation par défaut, méthode ELS",
    len(c2["geometrie"]["troncons"]) == 3 and c2["geometrie"]["troncons"][0]["L"] == 3000 and c2["charges"]["qh"] == 1.0
    and c2["montant"]["section"] == {"nom": "CHS 48.3x3.2"} and c2["fixation"]["type"] == "platine" and c2["methode"]["mode"] == "els")
chk("un cas complété se vérifie", ET.verifier(c2)["statut"] in ("ok", "attention", "nok"))
try:
    ET.depuis_json("{}")
    chk("un fichier sans cas est refusé", False)
except ValueError:
    chk("un fichier sans cas est refusé", True)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

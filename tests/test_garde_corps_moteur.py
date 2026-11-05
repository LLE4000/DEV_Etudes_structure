# -*- coding: utf-8 -*-
"""Tests du moteur « Garde-corps » v2 (acier/garde_corps, sans Streamlit). Chaque garantie
rougit si on la retire :

  1. GÉOMÉTRIE : répartition régulière, entraxes saisis (un poteau central à deux distances
     différentes), longueurs tributaires, montant d'angle d'un L et d'un U, extrémité ancrée
     dans le mur (le mur reprend la demi-travée), phrase.
  2. TYPES ET CHARGES : les cinq types du menu fixent système et remplissage ; les quatre
     charges nommées (q_h, q_v, Q_r, q_r) avec leur origine ; catégories du Tableau 6.12 ;
     valeur libre ; mode P ; cas distincts par défaut, cumul en option.
  3. MÉCANIQUE : console (M = P·H, V = P, f = PH³/3EI), poutre sur deux appuis (qL²/8, QL/4,
     5qL⁴/384EI, somme / max), poutre continue (coefficients des tables : 0,125 / 0,100 /
     0,107 qL²), vérification de section, limites ELS et Eurocode, statuts.
  4. BIBLIOTHÈQUE : creux du catalogue, plats et ronds de stock, sections libres, axe fort =
     sens de h, aire de cisaillement entière pour un plein, spécifications, refus.
  5. VÉRIFICATION COMPLÈTE : le montant sous q_h seule (cas 1) gouverne, le cas 2 (q_v seule)
     est calculé ; main courante : cas 1 et 2 séparés, le pire ; cumul : efforts additionnés ;
     continuité ; angle vérifié dans les deux orientations ; barreaux à portée DÉDUITE de H,
     h_bas et des sections, cas 3 (Q_r) et 4 (q_r) ; lisses intermédiaires ; panneau ; masse.
  6. FIXATION : cas de soudure (C6 tube, C3 plat, C8 libre) et ses efforts sous le cas de charge
     qui gouverne (N = 0 sous q_h seule), platine (bras de levier, Z, F_t, F_v, boulon
     EN 1993-1-8, cheville ETA, interaction, ELS /γ_F, flexion, épaisseur requise, refus),
     système mur (console sous q_v, platine tirée sous q_h : enveloppe des deux modèles), scellé.
  7. CANDIDATS PAR RÔLE : montant, main courante, barreaux, lisses intermédiaires, console ;
     tri par masse ; le plus léger qui tient ; application de la section au rôle.
  8. ÉTAT ET JSON : nouveau cas, fichier v1 relu (type déduit, portée des barreaux déduite),
     aller-retour.

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
from acier.garde_corps import types as TY  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cas(type_=TY.TYPE_DEFAUT, **kw):
    c = ET.nouveau_cas(None, type_)
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
chk("entraxes saisis 1200 + 900 : trois montants à x = 0, 1200, 2100, tributaires 600 / 1050 / 450",
    [m["x"] for m in a["montants"]] == [0, 1200, 2100] and [m["trib1"] for m in a["montants"]] == [600, 1050, 450] and a["s_max"] == 1200)
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
print("\n=== 2. Types et charges ===")
chk("cinq types : barreaudage (défaut), lisses, panneau, main courante seule, murale", list(TY.TYPES) == ["barreaudage", "lisses", "panneau", "main_courante", "murale"]
    and TY.TYPE_DEFAUT == "barreaudage")
for ty, (sysm, rp) in {"barreaudage": ("montants", "barreaux"), "lisses": ("montants", "lisses"), "panneau": ("montants", "panneau"),
                       "main_courante": ("montants", "aucun"), "murale": ("mur", "aucun")}.items():
    c = ET.nouveau_cas("t", ty)
    chk(f"type « {ty} » : système {sysm}, remplissage {rp}, retrouvé par type_de", c["geometrie"]["systeme"] == sysm and c["remplissage"]["type"] == rp and TY.type_de(c) == ty)
c = ET.nouveau_cas()
c["remplissage"]["type"] = "lisses"
chk("un remplissage changé à la main redéfinit le type déduit (lisses)", TY.type_de(c) == "lisses")
chk("sections par défaut : montant et main courante RHS 50x30x3, barreaux Plat 30x8, lisse basse RHS 50x25x2.5 (froid)",
    c["montant"]["section"] == {"nom": "RHS 50x30x3"} and c["lisse"]["section"] == {"nom": "RHS 50x30x3"} and c["remplissage"]["section"] == {"nom": "Plat 30x8"}
    and c["remplissage"]["section_basse"] == {"nom": "RHS 50x25x2.5 (froid)"} and c["lisse"]["appuis"] == "deux")
chk("catégorie A : 0,5 kN/m ; C2-C4 : 1,0 ; C5 : 3,0 ; E : 2,0 (valeurs recommandées du Tableau 6.12)",
    CH.qh_de_categorie("A") == 0.5 and CH.qh_de_categorie("C2-C4") == 1.0 and CH.qh_de_categorie("C5") == 3.0 and CH.qh_de_categorie("E") == 2.0)
ch = CH.normaliser({"categorie": "D", "qh": 7.0})
chk("une catégorie impose sa valeur (D : 1,0 malgré 7 saisi) ; la valeur libre la garde", ch["qh"] == 1.0 and CH.normaliser({"categorie": "perso", "qh": 0.8})["qh"] == 0.8)
chk("défauts : q_v 0,5 kN/m, q_r 0,5 kN/m², Q_r 0,5 kN, mode q, cas distincts (cumul False)", ch["qv"] == 0.5 and ch["qr"] == 0.5 and ch["Qr"] == 0.5 and ch["mode"] == "q" and ch["cumul"] is False)
tab = CH.tableau(ch)
chk("les quatre charges nommées, numérotées 1 à 4 : q_h, q_v, Q_r, q_r avec valeur, unité et norme",
    [t["n"] for t in tab] == [1, 2, 3, 4] and [t["symbole"] for t in tab] == ["q_h", "q_v", "Q_r", "q_r"] and tab[0]["valeur"] == 1.0 and tab[0]["unite"] == "kN/m"
    and "Tab. 6.12" in tab[0]["norme"] and "Q_kh,2" in tab[2]["norme"] and tab[3]["unite"] == "kN/m²")
chk("mode P : la charge 1 devient P par montant (saisie directe)", CH.tableau(CH.normaliser({"mode": "P", "P": 1.2}))[0]["symbole"] == "P" and CH.tableau(CH.normaliser({"mode": "P", "P": 1.2}))[0]["valeur"] == 1.2)
chk("la phrase des cas : distincts par défaut, cumulés en option", "distincts" in CH.phrase_cumul(ch) and "cumulées" in CH.phrase_cumul({**ch, "cumul": True}))

# ================================================================
print("\n=== 3. Mécanique ===")
E = 210000.0
c0 = ME.console(450.0, 0.0, 450.0, 1100.0, E, 115857.0, 115857.0)
chk("console : M = P·H = 495 000 N·mm, V = P, f = PH³/(3EI)", pres(c0["M1"], 495000.0) and c0["V1"] == 450.0 and pres(c0["f"], 450 * 1100 ** 3 / (3 * E * 115857)))
p = ME.poutre(0.5, 300.0, 900.0, E, 76200.0)
chk("poutre sur deux appuis : qL²/8 + QL/4, V = qL/2 + Q/2, f = 5qL⁴/384EI + QL³/48EI",
    pres(p["M"], 0.5 * 900 ** 2 / 8 + 300 * 900 / 4) and pres(p["V"], 225 + 150) and pres(p["f"], 5 * 0.5 * 900 ** 4 / (384 * E * 76200) + 300 * 900 ** 3 / (48 * E * 76200)))
pm = ME.poutre(0.5, 300.0, 900.0, E, 76200.0, "max")
chk("combinaison « max » : le plus grand des deux cas", pres(pm["M"], 67500.0) and pres(pm["V"], 225.0))
pc2, pc3, pc5 = (ME.poutre_continue(0.5, 0.0, 900.0, n, E, 76200.0) for n in (2, 3, 5))
chk("poutre continue : 2 travées 0,125 qL², 0,625 qL, 0,00541 qL⁴/EI ; 3 travées 0,100 ; ≥ 4 travées 0,107 (tables)",
    pres(pc2["M"], 0.125 * 0.5 * 900 ** 2) and pres(pc2["V"], 0.625 * 0.5 * 900) and pres(pc2["f"], 0.00541 * 0.5 * 900 ** 4 / (E * 76200))
    and pres(pc3["M"], 0.100 * 0.5 * 900 ** 2) and pres(pc5["M"], 0.107 * 0.5 * 900 ** 2))
pcQ = ME.poutre_continue(0.0, 300.0, 900.0, 3, E, 76200.0)
chk("continue : une charge ponctuelle garde QL/4 et QL³/48EI (côté sûr) ; une travée = deux appuis", pres(pcQ["M"], 300 * 900 / 4) and pres(pcQ["f"], 300 * 900 ** 3 / (48 * E * 76200))
    and pres(ME.poutre_continue(0.5, 0.0, 900.0, 1, E, 76200.0)["M"], 0.5 * 900 ** 2 / 8))
sec = BI.section({"nom": "CHS 48.3x3.2"})
meth = ME.methode_defaut()
v = ME.verifier_section(sec, 495000.0, 0.0, 450.0, 450.0, 0.0, 8.206, 1100.0, meth, 235.0)
chk("vérification ELS : σ = M/W + N/A, τ = V/A_v, flèche/(L/100) ; σ_adm 160, τ_adm 90 ; η = max",
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
sec_m = r["section_montant"]
chk("cas par défaut (barreaudage, RHS 50x30x3) : conforme, 5 montants, le montant M2 gouverne par la flèche sous le cas 1 (q_h seule)",
    r["statut"] == "ok" and r["type"] == "barreaudage" and len(r["montants"]) == 5 and r["gouvernant"]["cle"] == "montant" and r["montant"]["nom"] == "M2"
    and r["montant"]["verif"]["critere"] == "fleche" and r["montant"]["verif"]["cas_charge"] == "1")
m2 = r["montant"]["verif"]
chk("montant M2 sous q_h seule : P = 0,45 kN, N = 0, M = 0,495 kN·m, σ = M/W_el,y = 91,2 N/mm², f = 7,01 mm ≤ 11 ; le cas 2 (q_v seule) est calculé et faible",
    pres(m2["P1"], 450.0) and m2["N"] == 0.0 and pres(m2["M1"], 495000.0) and pres(m2["sigma"], 495000 / (sec_m["Wel_y"] * 1e3)) and pres(m2["fleche"], 7.01, 2e-3)
    and pres(m2["fleche_lim"], 11.0) and "2" in m2["cas_taux"] and m2["cas_taux"]["2"] < 0.02 and pres(m2["N_v"], 450.0))
li = r["lisse"]["verif"]
chk("main courante : portée 900 sur deux appuis, cas 1 (q_h) et cas 2 (q_v) séparés, le pire gouverne (σ = M_h/W_el,z, pas de somme)",
    pres(li["L"], 900.0) and li["appuis"] == "deux" and pres(li["Mh"], 0.5 * 900 ** 2 / 8) and pres(li["Mv"], li["Mh"]) and li["cas_charge"] == "1"
    and pres(li["sigma"], li["Mh"] / (r["lisse"]["section"]["Wel_z"] * 1e3)) and set(li["cas_taux"]) == {"1", "2"} and pres(li["taux_max"], max(li["cas_taux"].values())))
b = r["remplissage"]["verif"]
chk("barreaux Plat 30×8 à 110 : portée DÉDUITE 950 = 1100 − h_bas 100 − 25 − 25 ; le cas 3 (Q_r = 0,5 kN) gouverne (M = QL/4) devant le cas 4 (q_r·e)",
    pres(r["remplissage"]["L"], 950.0) and b["cas_charge"] == "3" and pres(b["M1"], 500 * 950 / 4) and b["cas_taux"]["4"] < b["cas_taux"]["3"])
chk("la lisse basse est une section du cas (non calculée), à h_bas = 100 mm", r["remplissage"]["section_basse"]["nom"] == "RHS 50x25x2.5 (froid)" and r["remplissage"]["h_bas"] == 100.0)
chk("les éléments : montant, main courante, barreaux, soudure, chevilles, platine ; une masse indicative",
    [e["cle"] for e in r["elements"]] == ["montant", "lisse", "barreau", "soudure", "ancrage", "platine"] and 50 < r["masse"] < 200)
rC = ET.verifier(cas(lisse={"appuis": "continue"}))
lc = rC["lisse"]["verif"]
chk("main courante continue sur 4 travées : M_h = 0,107 qL², η plus faible que sur deux appuis", lc["appuis"] == "continue" and lc["n_travees"] == 4 and pres(lc["kM"], 0.107)
    and pres(lc["Mh"], 0.107 * 0.5 * 900 ** 2) and lc["taux_max"] < li["taux_max"])
rK = ET.verifier(cas(charges={"cumul": True}))
mk, lk = rK["montant"]["verif"], rK["lisse"]["verif"]
chk("cumul : montant sous q_h et q_v (cas 1+2 : σ = M/W + N/A), main courante biaxiale (σ = M_v/W_y + M_h/W_z)",
    mk["cas_charge"] == "1+2" and pres(mk["N"], 450.0) and pres(mk["sigma"], m2["sigma"] + 450 / (sec_m["A"] * 1e2)) and lk["cas_charge"] == "1+2"
    and pres(lk["sigma"], lk["Mv"] / (r["lisse"]["section"]["Wel_y"] * 1e3) + lk["Mh"] / (r["lisse"]["section"]["Wel_z"] * 1e3)))
cL = cas(geometrie={"forme": "L"}, montant={"section": {"nom": "RHS 60x40x3"}})
rL = ET.verifier(cL)
gL = rL["montant"]["verif"]
chk("en L avec un RHS 60×40 : le montant d'angle gouverne, vérifié dans l'orientation la pire (P selon b)",
    gL["coin"] and gL.get("orientation_inversee") and gL["P2"] > 0 and gL["taux_max"] > max(v["taux_max"] for v in rL["montants"] if not v["coin"]))
rP = ET.verifier(cas(charges={"mode": "P", "P": 1.0}))
chk("mode P direct : chaque montant reçoit 1 kN, la main courante l'équivalent P/s = 1,11 kN/m",
    all(pres(v["P1"], 1000.0) for v in rP["montants"]) and pres(rP["lisse"]["verif"]["qh"], 1000 / 900))
rI = ET.verifier(ET.nouveau_cas("l", "lisses"))
chk("type lisses : lisses intermédiaires sur la portée s_max, charge q_r·e = 0,165 kN/m ou Q_r (cas 3 / 4)", rI["remplissage"]["verif"]["L"] == 900
    and pres(rI["remplissage"]["verif"]["q"], 0.5 * 330 / 1e3) and any(e["cle"] == "lisse_inter" for e in rI["elements"]) and rI["remplissage"]["verif"]["cas_charge"] in ("3", "4"))
rPan = ET.verifier(ET.nouveau_cas("p", "panneau"))
chk("type panneau : pas d'élément de remplissage vérifié, une lisse basse, un message pour le fournisseur",
    not any(e["cle"] in ("barreau", "lisse_inter") for e in rPan["elements"]) and rPan["remplissage"].get("section_basse") and any("fournisseur" in m for m in rPan["messages"]))
chk("H > 1,20 m : message (bras de levier pris tel quel, côté sûr)", any("1,20 m" in m or "1.20 m" in m for m in ET.verifier(cas(geometrie={"H": 1300.0}))["messages"]))
chk("fin ancrée dans le mur : le message donne l'effort repris par le mur", any("mur" in m and "kN" in m for m in ET.verifier(cas(geometrie={"fin": "mur"}))["messages"]))

# ================================================================
print("\n=== 6. Fixation ===")
fx = r["fixation"]
sres = fx["soudure"]["res"]
chk("RHS du catalogue : cas C6 tube sur platine sous le cas 1 : N = 0, V_z = 0,45, M_y = −0,495 kN·m",
    sres["config"] == "C6" and fx["soudure"]["cas_charge"] == "1" and pres(fx["soudure"]["cas"]["efforts"]["N"], 0.0) and pres(fx["soudure"]["cas"]["efforts"]["Vz"], 0.45)
    and pres(fx["soudure"]["cas"]["efforts"]["My"], -0.495))
chk("la soudure est jugée par la méthode ELS du bureau (k = 100) quand le garde-corps l'est", sres["methode"].mode == "els" and sres["methode"].k == 100.0 and fx["soudure"]["taux"] > 0.4)
chk("le groupe de cordons fait le tour du RHS 50 × 30 : A_w ≈ périmètre × a", pres(sres["proprietes"]["A_w"], sres["pieces"]["perimetre"] * 4.0, 0.02))
rPl = ET.verifier(cas(montant={"section": {"nom": "Plat 50x10"}}))
cs3 = rPl["fixation"]["soudure"]["cas"]
chk("plat : cas C3 console fléchie, h = 50, t = 10, e = H = 1100, V = P, N = 0 sous q_h seule", cs3["config"] == "C3" and cs3["params"]["h"] == 50 and cs3["params"]["t"] == 10
    and cs3["params"]["e"] == 1100 and pres(cs3["efforts"]["V"], 0.45) and pres(cs3["efforts"]["N"], 0.0))
rT = ET.verifier(cas(montant={"section": {"famille": "TUBE", "dims": {"d": 48.3, "t": 3.2}}}))
cs8 = rT["fixation"]["soudure"]["cas"]
rChs = ET.verifier(cas(montant={"section": {"nom": "CHS 48.3x3.2"}}))
chk("tube libre : cas C8, un arc de 360° de rayon d/2, V_y = P, M_x = −P·H ; même taux que le C6 du CHS (±5 %)",
    cs8["config"] == "C8" and cs8["params"]["elements"][0]["type"] == "arc" and pres(cs8["params"]["elements"][0]["r"], 24.15)
    and pres(cs8["efforts"]["Vy"], 0.45) and pres(cs8["efforts"]["Mx"], -0.495) and pres(rT["fixation"]["soudure"]["taux"], rChs["fixation"]["soudure"]["taux"], 0.05))
rCa = ET.verifier(cas(montant={"section": {"famille": "CAISSON", "dims": {"h": 60, "b": 40, "t": 3}}}))
chk("caisson libre : cas C8, quatre segments sur le contour 40 × 60, A_w = 200·a", rCa["fixation"]["soudure"]["cas"]["config"] == "C8"
    and len(rCa["fixation"]["soudure"]["cas"]["params"]["elements"]) == 4 and pres(rCa["fixation"]["soudure"]["res"]["proprietes"]["A_w"], 200 * 4.0, 0.05))
pl = fx["platine"]
chk("platine sous le cas 1 (N = 0) : z = e_x/2 + h_p/2 = 75, Z = M/z = 6,6 kN, F_t = Z/2, F_v = V/4",
    pl["cas_charge"] == "1" and pres(pl["z"], 75.0) and pres(pl["Z"], 495000 / 75.0) and pres(pl["Ft"], pl["Z"] / 2) and pres(pl["Fv"], 450 / 4))
chk("cheville ETA en ELS : N_Rd/γ_F = 9/1,5 = 6 kN, interaction (β_N)^1,5 + (β_V)^1,5",
    pres(pl["FtRd"], 6000.0) and pres(pl["inter"], (pl["Ft"] / 6000) ** 1.5 + (pl["Fv"] / 8000) ** 1.5))
chk("flexion de la platine : m = e_x/2 − h_p/2 − 0,8·a·√2 = 20,5, b_eff = min(l_p ; 2·min(2πm ; 4m + 1,25e)) = 150, σ_p = 6·Z·m/(b·t²), t_req",
    pres(pl["m"], 50 - 25 - 0.8 * 4 * math.sqrt(2)) and pres(pl["b_eff"], 150.0) and pres(pl["sigma_p"], 6 * pl["Z"] * pl["m"] / (150 * 100))
    and pres(pl["tp_req"], math.sqrt(6 * pl["Z"] * pl["m"] / (150 * 160))))
plK = rK["fixation"]["platine"]
chk("cumul : la platine reçoit aussi N = 0,45 kN de compression (Z = (M − N·h_p/2)/z), cas 1+2", plK["cas_charge"] == "1+2" and pres(plK["Z"], (495000 - 450 * 25) / 75.0))
rB = ET.verifier(cas(fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon", "boulon": "M12", "classe": "8.8"}}))
pb = rB["fixation"]["platine"]
chk("boulon M12 8.8 (Tab. 3.4) : F_t,Rd = 0,9·800·84,3/1,25 = 48,5 kN, /1,5 en ELS ; interaction F_v/F_v,Rd + F_t/(1,4·F_t,Rd)",
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
cW = ET.nouveau_cas("m", "murale")
rW = ET.verifier(cW)
cw = rW["console"]["verif"]
chk("murale : la main courante est principale, la console est fléchie par q_v seule (cas 2 : M = R_v·e, N = 0) et tirée par q_h seule (cas 1 : σ = R_h/A)",
    rW["lisse"]["role"] == "principal" and rW["montant"] is None and cw["cas_charge"] == "2" and pres(cw["Rv"], 450.0) and pres(cw["M1"], 450 * 100) and cw["N"] == 0.0
    and pres(cw["Rh"], 450.0) and "1" in cw["cas_taux"])
pw = rW["fixation"]["platine"]
chk("console au mur : la platine est jugée sous le pire des cas — ici q_v (Z = M/z = 0,6 kN) devant q_h (couple de boulons N_t/2 = 0,225 kN)",
    pw["cas_charge"] == "2" and pres(pw["Z"], 45000 / 75.0) and pw["N"] == 0.0)
chk("console : cas de soudure C3 avec V = R_v sous le cas 2", rW["fixation"]["soudure"]["cas"]["config"] == "C3" and pres(rW["fixation"]["soudure"]["cas"]["efforts"]["V"], 0.45))
rWK = ET.verifier(cas("murale", charges={"cumul": True}))
chk("murale cumulée : la platine est tirée (N_t = R_h) et fléchie en même temps, enveloppe des deux modèles",
    rWK["fixation"]["platine"]["N"] < 0 and pres(rWK["fixation"]["platine"]["Z"], max((45000 + 450 * 25) / 75.0, 45000 / 100 + 225)))

# ================================================================
print("\n=== 7. Candidats par rôle ===")
chk("rôles du cas : barreaudage → montant, main courante, barreaux ; lisses → … lisses intermédiaires ; murale → main courante, console",
    ET.roles_du_cas(c0) == ["montant", "lisse", "barreau"] and ET.roles_du_cas(ET.nouveau_cas("l", "lisses")) == ["montant", "lisse", "lisse_inter"]
    and ET.roles_du_cas(cW) == ["lisse", "console"])
cands = ET.candidats(c0, "montant", ["creux"])
chk("les candidats montants creux, par masse croissante, chacun avec son taux", len(cands) > 100 and [x["masse"] for x in cands] == sorted(x["masse"] for x in cands)
    and all(0 <= x["taux"] for x in cands))
best = ET.plus_leger(cands)
chk("le plus léger tube qui tient comme montant : CHS 48.3x2.5 (η 91 %)", best["section"]["nom"] == "CHS 48.3x2.5" and best["taux"] <= 1.0
    and all(x["taux"] > 1.0 for x in cands[:cands.index(best)]))
chk("le RHS 50x30x3 du cas est un candidat au même taux que la vérification", any(x["section"]["nom"] == "RHS 50x30x3" and pres(x["taux"], r["montant"]["verif"]["taux_max"]) for x in cands))
cb = ET.candidats(c0, "barreau", ["plats", "ronds"], 15.0)
chk("candidats barreaux (plats et ronds) : jugés sur la portée déduite 950 sous Q_r ; le plus léger qui tient est un plat", cb and ET.plus_leger(cb)["section"]["famille"] == "PLAT"
    and any(x["section"]["nom"] == "Plat 30x8" and pres(x["taux"], r["remplissage"]["verif"]["taux_max"]) for x in cb))
cl = ET.candidats(c0, "lisse", ["creux"])
chk("candidats main courante : jugés comme lisse sur deux appuis (cas 1 ou 2)", cl and any(x["section"]["nom"] == "RHS 50x30x3" and pres(x["taux"], li["taux_max"]) for x in cl))
cc = ET.candidats(cW, "console", ["plats"])
chk("candidats console (murale) : jugés comme console", cc and any(x["section"]["nom"] == "Plat 50x10" and pres(x["taux"], cw["taux_max"]) for x in cc))
c1 = ET.nouveau_cas()
ET.appliquer_section(c1, best["section"], "montant")
ET.appliquer_section(c1, BI.section({"nom": "Rond Ø14"}), "barreau")
ET.appliquer_section(c1, BI.section({"nom": "CHS 42.4x3.2"}), "lisse")
chk("appliquer une section par rôle écrit sa spécification au bon endroit", c1["montant"]["section"] == {"nom": "CHS 48.3x2.5"} and c1["remplissage"]["section"] == {"nom": "Rond Ø14"}
    and c1["lisse"]["section"] == {"nom": "CHS 42.4x3.2"})

# ================================================================
print("\n=== 8. État et JSON ===")
e0 = ET.nouvel_etat()
txt = ET.vers_json(e0)
e1 = ET.depuis_json(txt)
chk("aller-retour JSON : un cas identique, version 2", e1["cas"][0] == e0["cas"][0] and e1["actif"] == 0 and e1["version"] == 2)
vieux = json.dumps({"cas": [{"nom": "ancien", "geometrie": {"forme": "L", "troncons": [{"L": 3000}]}, "charges": {"categorie": "D"},
                             "remplissage": {"type": "lisses", "L": 900}}]})
e2 = ET.depuis_json(vieux)
c2 = e2["cas"][0]
chk("un fichier v1 est relu : trois tronçons, q_h de la catégorie D, type déduit « lisses », sections et fixation par défaut, méthode ELS, cas distincts",
    len(c2["geometrie"]["troncons"]) == 3 and c2["geometrie"]["troncons"][0]["L"] == 3000 and c2["charges"]["qh"] == 1.0 and c2["type"] == "lisses"
    and c2["montant"]["section"] == {"nom": "RHS 50x30x3"} and c2["fixation"]["type"] == "platine" and c2["methode"]["mode"] == "els" and c2["charges"]["cumul"] is False)
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

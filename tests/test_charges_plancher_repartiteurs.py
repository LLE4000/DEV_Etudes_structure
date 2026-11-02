# -*- coding: utf-8 -*-
"""Charges sur plancher — répartiteurs (plancher/charges/repartiteurs.py).

Garanties, chacune rougit si on retire ce qu'elle verrouille :
  1. Bande efficace c = t·√(f/(3·p)) ; aire efficace itérée à la main dans
     le test (boucle indépendante) ; c plafonné par le bord du plateau ;
     plateau épais → complet ; empreinte plus grande que le plateau → erreur.
  2. Propriétés d'une section en I : le repli ``section_I`` retrouve, à 2 %
     près, Iy et Wel des profilés de ``profiles_test.json`` (valeurs du
     catalogue de l'application, pas de mémoire).
  3. Poutre sur ressorts : statique (deux ressorts : réactions 2/3 · 1/3
     quelle que soit EI, M = P·L/4 sur deux ressorts d'extrémité) ; rigide
     sur trois ressorts égaux → trois réactions égales ; très souple → tout
     sur le ressort sous la charge ; Σ R = Σ P ; symétrie ; moment nul aux
     extrémités libres ; erreurs sur un seul ressort ou une abscisse hors
     poutre.
  4. Flexion de la filière (ELU) : taux = γQ·M/(W·fy/γM0), γQ sur les charges de chantier (P-2).

Lancement : python3 tests/test_charges_plancher_repartiteurs.py (depuis la racine).
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from plancher.charges import repartiteurs as RP  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# 1. bande efficace
chk("c = t·√(f/(3p))", pres(RP.bande_c(20.0, 235.0, 0.5), 20.0 * math.sqrt(235.0 / 1.5)))
chk("c infini si p ≤ 0", RP.bande_c(20.0, 235.0, 0.0) == math.inf)
P, a, b, A, B, t, f = 95e3, 300.0, 300.0, 600.0, 600.0, 20.0, 235.0
c = 0.0
for _ in range(200):                       # itération indépendante
    p_ = P / ((a + 2 * c) * (b + 2 * c))
    c = min(t * math.sqrt(f / (3 * p_)), (A - a) / 2)
ae = RP.aire_efficace(P, a, b, A, B, t, f)
chk("aire efficace : c itéré à la main", pres(ae["c"], c, 1e-6), f"{ae['c']} vs {c}")
chk("aire efficace : a_eff = a + 2c, pression cohérente", pres(ae["a_eff"], a + 2 * ae["c"]) and
    pres(ae["p"], P / (ae["a_eff"] * ae["b_eff"])))
chk("aire efficace : c ≤ c_max", ae["c"] <= ae["c_max"] + 1e-9)
chk("tôle de 20 sous 95 kN : c ≥ c_max, plateau entier efficace", ae["complet"] and pres(ae["a_eff"], A) and pres(ae["part_efficace"], 1.0))
ae3 = RP.aire_efficace(P, a, b, A, B, 6.0, f)
chk("tôle de 6 : partiellement efficace, c < c_max", ae3["part_efficace"] < 1.0 and not ae3["complet"] and ae3["c"] < ae3["c_max"])
ae2 = RP.aire_efficace(P, a, b, A, B, 100.0, f)
chk("plateau épais : complet, aire = plateau", ae2["complet"] and pres(ae2["a_eff"], A) and pres(ae2["part_efficace"], 1.0))
try:
    RP.aire_efficace(P, 700.0, 300.0, A, B, t, f)
    chk("empreinte > plateau : erreur", False)
except ValueError:
    chk("empreinte > plateau : erreur", True)

# 2. section en I contre le catalogue de l'application
with open("profiles_test.json", "r", encoding="utf-8") as fh:
    prof = json.load(fh)
ecarts = []
for nom, p in prof.items():
    if not all(p.get(k) for k in ("h", "b", "tw", "tf", "Iy", "Wel")):
        continue
    s = RP.section_I(p["h"], p["b"], p["tw"], p["tf"], p.get("r", 0.0))
    ecarts.append((nom, abs(s["Iy_mm4"] / 1e4 - p["Iy"]) / p["Iy"], abs(s["Wel_y_mm3"] / 1e3 - p["Wel"]) / p["Wel"]))
pire = max(ecarts, key=lambda t: max(t[1], t[2]))
chk("section_I : Iy et Wel à 2 % du catalogue sur tous les profilés", len(ecarts) >= 40 and max(pire[1], pire[2]) < 0.02, str(pire))

# 3. poutre sur ressorts
r = RP.poutre_sur_ressorts(1.0, 3.0, [(0.0, 50.0), (3.0, 50.0)], [(1.0, 30.0)])
chk("deux ressorts : réactions 2/3 · 1/3 (statique)", pres(r["reactions"][0][1], 20.0, 1e-9) and pres(r["reactions"][1][1], 10.0, 1e-9))
r2 = RP.poutre_sur_ressorts(1e6, 3.0, [(0.0, 7.0), (3.0, 900.0)], [(1.0, 30.0)])
chk("deux ressorts : indépendant de EI et des raideurs", pres(r2["reactions"][0][1], 20.0, 1e-9) and pres(r2["reactions"][1][1], 10.0, 1e-9))
r3 = RP.poutre_sur_ressorts(1.0, 4.0, [(0.0, 10.0), (4.0, 10.0)], [(2.0, 8.0)])
chk("M = P·L/4 sur deux ressorts d'extrémité", pres(r3["M_max"], 8.0, 1e-9))
rr = RP.poutre_sur_ressorts(1e9, 2.4, [(0.0, 100.0), (1.2, 100.0), (2.4, 100.0)], [(1.2, 30.0)])
chk("rigide sur trois ressorts égaux : réactions égales", all(pres(R, 10.0, 1e-6) for _, R in rr["reactions"]), str(rr["reactions"]))
rs = RP.poutre_sur_ressorts(1e-6, 2.4, [(0.0, 100.0), (1.2, 100.0), (2.4, 100.0)], [(1.2, 30.0)])
chk("très souple : tout sur le ressort sous la charge", rs["reactions"][1][1] > 29.9, str(rs["reactions"]))
chk("Σ R = Σ P", rr["ecart_equilibre"] < 1e-6 * 30.0 and rs["ecart_equilibre"] < 1e-9 and r["ecart_equilibre"] < 1e-9,
    str((rr["ecart_equilibre"], rs["ecart_equilibre"], r["ecart_equilibre"])))
rm = RP.poutre_sur_ressorts(500.0, 3.6, [(0.6, 80.0), (1.8, 80.0), (3.0, 80.0)], [(1.8, 60.0)])
chk("symétrie", pres(rm["reactions"][0][1], rm["reactions"][2][1], 1e-9))
chk("moment nul aux extrémités libres", abs(rm["M"][0]) < 1e-9 and abs(rm["M"][-1]) < 1e-9)
chk("intermédiaire entre rigide et souple", 20.0 < rm["reactions"][1][1] < 60.0, str(rm["reactions"]))
for args, nom in (((1.0, 3.0, [(1.0, 10.0)], [(1.0, 5.0)]), "un seul ressort"),
                  ((1.0, 3.0, [(0.0, 10.0), (3.0, 10.0)], [(4.0, 5.0)]), "charge hors poutre")):
    try:
        RP.poutre_sur_ressorts(*args)
        chk(f"erreur : {nom}", False)
    except ValueError:
        chk(f"erreur : {nom}", True)

# 4. flexion de la filière
v = RP.verifier_filiere({"M_max": 34.0}, 215.6, 235.0, 1.0, 1.5)
chk("filière : M_Rd = W·fy/γM0", pres(v["M_Rd"], 215.6e3 * 235.0 / 1e6))
chk("filière : taux = γQ·M/M_Rd (ELU sur les charges de chantier)", pres(v["taux"], 1.5 * 34.0 / v["M_Rd"]))

print(f"RÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("  ÉCHEC :", nom, info)
sys.exit(1 if KO else 0)

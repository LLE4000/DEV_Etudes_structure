# -*- coding: utf-8 -*-
"""Moteur des soudures (acier/soudures) : décomposition sur la gorge,
groupe de cordons (répartition élastique), critères EN 1993-1-8 §4.5.3 et
méthode ELS du bureau, configurations C1 à C8 contre les formules fermées
classiques, état et JSON, pureté.

Lancement : python3 tests/test_soudures_moteur.py
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import numpy as np  # noqa: E402

from acier.soudures import cordon as C  # noqa: E402
from acier.soudures import groupe as G  # noqa: E402
from acier.soudures import criteres as K  # noqa: E402
from acier.soudures import configurations as CF  # noqa: E402
from acier.soudures import etat as ET  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


FU, BW, GM2 = 360.0, 0.8, 1.25            # S235 (bibliothèque) ; γM2 ANB
S2, S3 = math.sqrt(2), math.sqrt(3)

print("=== 1. Le cordon : décomposition sur la gorge (fig. 4.5) ===")
c = C.cordon_angle("c", (0, 0), (100, 0), 5.0, cote=+1)
r1 = c.composantes((0, 100, 0))        # en travers, dans le plan de joint
r2 = c.composantes((0, 0, 100))        # arrachement
r3 = c.composantes((100, 0, 0))        # le long
chk("force en travers dans le plan : |σ⊥| = |τ⊥| = f/(a·√2), τ∥ = 0",
    pres(abs(r1["sigma_perp"]), 100 / (5 * S2)) and pres(abs(r1["tau_perp"]), 100 / (5 * S2)) and pres(r1["tau_par"], 0.0))
chk("arrachement : mêmes modules, signe relatif opposé (la géométrie décide)",
    pres(abs(r2["sigma_perp"]), 100 / (5 * S2)) and pres(abs(r2["tau_perp"]), 100 / (5 * S2))
    and (r1["sigma_perp"] * r1["tau_perp"]) * (r2["sigma_perp"] * r2["tau_perp"]) < 0)
chk("SIGNE : l'arrachement (+z) met la gorge en TRACTION, σ⊥ > 0 ; en travers, la pièce qui VIENT sur le cordon (force selon u_A = −y) le comprime, "
    "σ⊥ < 0, celle qui s'en éloigne (+y) le tend",
    r2["sigma_perp"] > 0 and c.composantes((0, -100, 0))["sigma_perp"] < 0 and c.composantes((0, 100, 0))["sigma_perp"] > 0)
chk("le long : τ∥ = f/a seul", pres(r3["tau_par"], 20.0) and pres(r3["sigma_perp"], 0.0) and pres(r3["tau_perp"], 0.0))
chk("ailes perpendiculaires à l'axe et entre elles ; u_A hors de la pièce (pièce à gauche de +x → u_A = −y)",
    pres(float(np.dot(c.e_a, c.u_A)), 0.0) and pres(float(np.dot(c.u_A, c.u_B)), 0.0) and tuple(np.round(c.u_A, 9)) == (0.0, -1.0, 0.0))
cb = C.cordon_bout_a_bout("b", (0, -50), (0, 50), 6.0, cote=+1)
rb = cb.composantes((120, 0, 0))
chk("bout à bout partiel : gorge perpendiculaire à n = x → σ⊥ = f/a, τ⊥ = τ∥ = 0",
    pres(abs(rb["sigma_perp"]), 20.0) and pres(rb["tau_perp"], 0.0, 1e-9) and pres(rb["tau_par"], 0.0, 1e-9))
cT = C.cordon_bout_a_bout_T("T", (0, 0), (100, 0), 6.0, cote=+1)
rT = cT.composantes((0, 0, 120))
chk("bout à bout en T : gorge parallèle à la face d'appui → l'arrachement donne σ⊥ = f/a > 0, τ = 0 ; en travers τ⊥ = f/a",
    pres(rT["sigma_perp"], 20.0) and pres(rT["tau_perp"], 0.0, 1e-9) and pres(abs(cT.composantes((0, 60, 0))["tau_perp"]), 10.0))
chk("longueur efficace l − 2a, retours → l ; cordon utile si l_eff ≥ max(30 ; 6a) et a ≥ 3",
    pres(C.cordon_angle("x", (0, 0), (100, 0), 5.0).l_eff, 90.0) and pres(C.cordon_angle("x", (0, 0), (100, 0), 5.0, retours=True).l_eff, 100.0)
    and not C.cordon_angle("x", (0, 0), (36, 0), 5.0).utile and C.cordon_angle("x", (0, 0), (40, 0), 5.0).utile
    and not C.cordon_angle("x", (0, 0), (100, 0), 2.5).utile)
chk("βLw : 1 jusqu'à 150a, 0,8 à 300a, ≥ 0", pres(C.beta_Lw(750, 5), 1.0) and pres(C.beta_Lw(1500, 5), 0.8) and C.beta_Lw(1e6, 5) == 0.0)

print("\n=== 2. Le groupe : répartition élastique ===")
cs = [C.cordon_angle("D", (5, -100), (5, 100), 5.0, cote=+1, retours=True), C.cordon_angle("G", (-5, -100), (-5, 100), 5.0, cote=-1, retours=True)]
g = G.Groupe(cs)
pr = g.proprietes()
chk("deux cordons verticaux de 200 à ±5 : A_w = 2·a·l, I_x = 2·a·l³/12, I_y = 2·a·l·5², I_xy = 0",
    pres(pr["A_w"], 2000.0) and pres(pr["Ix"], 2 * 5 * 200 ** 3 / 12) and pres(pr["Iy"], 2 * 5 * 200 * 25) and pres(pr["Ixy"], 0.0))
pts = g.forces(G.Efforts(Vy=-60e3, reference=(0, 0, 150)))
haut = next(p for p in pts if p["cordon"].nom == "D" and p["s"] == 1.0)
chk("console V = 60 kN à e = 150 : f_n = 3M/l² = 675 N/mm en tête, f_a = V/(2l) = 150 N/mm",
    pres(abs(haut["f_vec"][2]), 675.0, 1e-9) and pres(abs(haut["f_vec"][1]), 150.0))
chk("console : σ⊥ = τ⊥ = 95,5, τ∥ = 30,0 N/mm², |f| = 691,5 N/mm (exemple de l'étude)",
    pres(abs(haut["sigma_perp"]), 675 / (5 * S2)) and pres(abs(haut["tau_perp"]), 675 / (5 * S2)) and pres(abs(haut["tau_par"]), 30.0)
    and pres(haut["f"], math.hypot(675, 150)))
milieu = next(p for p in pts if p["cordon"].nom == "D" and p["s"] == 0.5)
chk("au milieu : flexion nulle, cisaillement seul", pres(milieu["f_vec"][2], 0.0, 1e-9) and pres(milieu["f"], 150.0))
# torsion d'un cordon seul : J = a·l³/12, force tangentielle au bout = 6T/l²
g1 = G.Groupe([C.cordon_angle("s", (0, -100), (0, 100), 5.0, cote=-1, retours=True)])
p1 = g1.forces(G.Efforts(Mz=1.2e6))
bout = next(p for p in p1 if p["s"] == 1.0)
chk("torsion d'un cordon seul de 200 mm : |f| au bout = 6·T/l² = 180 N/mm, perpendiculaire à l'axe",
    pres(bout["f"], 6 * 1.2e6 / 200 ** 2) and pres(abs(bout["f_vec"][0]), 180.0) and pres(bout["f_vec"][1], 0.0, 1e-9))
try:
    g1.forces(G.Efforts(My=1e6))
    chk("flexion autour de l'axe du cordon seul : refus", False)
except ValueError:
    chk("flexion autour de l'axe du cordon seul : refus (pas d'inertie)", True)
# gorges différentes : à contrainte égale, force proportionnelle à a
g2 = G.Groupe([C.cordon_angle("a4", (0, 0), (100, 0), 4.0, cote=1, retours=True), C.cordon_angle("a8", (0, 20), (100, 20), 8.0, cote=-1, retours=True)])
pg = g2.proprietes()
p2 = g2.forces(G.Efforts(Vx=120e3, reference=(pg["xG"], pg["yG"], 0.0)))      # effort au centre des gorges : pas de torsion
f4 = next(p["f"] for p in p2 if p["cordon"].nom == "a4")
f8 = next(p["f"] for p in p2 if p["cordon"].nom == "a8")
chk("gorges 4 et 8 : forces 1 : 2, même τ∥ = 120 000/(100·12) = 100 N/mm²", pres(f8, 2 * f4) and pres(f4 / 4, 100.0))
# transport : le point de référence excentré crée le moment
pA = G.Groupe(cs).forces(G.Efforts(Vy=-60e3, reference=(0, 0, 150)))
pB = G.Groupe(cs).forces(G.Efforts(Vy=-60e3, Mx=60e3 * 150))
chk("transport des efforts : V à (0,0,e) ≡ V + M_x = V·e au centre", all(pres(a["f"], b["f"], 1e-9) for a, b in zip(pA, pB)))

print("\n=== 3. Les critères ===")
d = K.directionnelle(95.5, 95.5, 30.0, FU, BW, GM2)
chk("directionnelle : σ_c = √(σ⊥² + 3(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) = 360 ; σ⊥ ≤ 0,9·fu/γM2 = 259,2",
    pres(d["sigma_c"], math.sqrt(95.5 ** 2 + 3 * (95.5 ** 2 + 30 ** 2))) and pres(d["lim_c"], 360.0) and pres(d["lim_perp"], 259.2))
s = K.simplifiee(691.5, 5.0, FU, BW, GM2)
chk("simplifiée : f_vw,d = fu/(√3·βw·γM2) = 207,8 ; |f| ≤ f_vw,d·a", pres(s["f_vw_d"], FU / (S3 * BW * GM2)) and pres(s["taux"], 691.5 / (s["f_vw_d"] * 5)))
chk("cisaillement pur le long : directionnelle = simplifiée (√3·τ∥ ≤ fu/(βw·γM2) ⟺ f ≤ f_vw,d·a)",
    pres(K.directionnelle(0, 0, 100.0, FU, BW, GM2)["taux"], K.simplifiee(500.0, 5.0, FU, BW, GM2)["taux"]))
chk("force en travers pure : directionnelle / simplifiée = √2/√3",
    pres(K.directionnelle(100 / S2, 100 / S2, 0, FU, BW, GM2)["taux"] / K.simplifiee(100.0, 1.0, FU, BW, GM2)["taux"], S2 / S3))
chk("bureau : 100 kg/(mm·cm) = 98,1 N/mm² (g = 9,81), 100 N/mm² (g = 10)",
    pres(K.tau_adm_bureau(100.0), 98.1) and pres(K.tau_adm_bureau(100.0, 10.0), 100.0))
kq = K.k_equivalent_eurocode(FU, BW, GM2)
chk("équivalent Eurocode S235 : f_vw,d/1,5 = 138,6 N/mm² = 141 kg/(mm·cm) ; S355 : 174,5",
    pres(kq["tau_N_mm2"], FU / (S3 * BW * GM2) / 1.5) and pres(kq["k_kg_mm_cm"], kq["tau_N_mm2"] * 10 / 9.81)
    and pres(K.k_equivalent_eurocode(510.0, 0.9, GM2)["tau_N_mm2"], 510 / (S3 * 0.9 * GM2) / 1.5))
e = K.els_bureau(461.0, 5.0, 100.0)
chk("ELS bureau : |f| ≤ k·a → 461/(98,1·5) = 0,94", pres(e["taux"], 461.0 / (98.1 * 5)))
pp = K.piece_pleine_penetration(200e3, 0.0, 0.0, 150.0, 12.0, 235.0)
chk("pleine pénétration : pièce σ = N/(b·t) = 111,1 ≤ fy/γM0", pres(pp["sigma"], 200e3 / 1800) and pres(pp["taux"], 200e3 / 1800 / 235))

print("\n=== 4. Les configurations contre les formules fermées ===")
def verif(config, params=None, efforts=None, methode=None):
    cas = ET.nouveau_cas(config)
    cas["params"].update(params or {})
    cas["efforts"].update(efforts or {})
    cas["methode"].update(methode or {})
    return ET.verifier(cas)
# C1 latéraux seuls : τ∥ = N/(2·a·L)
r = verif("C1", {"lateraux": True, "frontaux": 0, "retours": True, "b": 100, "L_rec": 120, "a": 5}, {"N": 80, "V": 0})
chk("C1 recouvrement, cordons latéraux : τ∥ = N/(2·a·L) = 66,7 N/mm², σ⊥ = τ⊥ = 0 ; η directionnelle = η simplifiée",
    r["statut"] != ET.REFUS and pres(abs(r["gouvernant"]["tau_par"]), 80e3 / (2 * 5 * 120)) and pres(r["gouvernant"]["sigma_perp"], 0.0, 1e-9)
    and pres(r["comparaison"]["directionnelle"], r["comparaison"]["simplifiee"]), str(r["messages"]))
r = verif("C1", {"lateraux": False, "frontaux": 1, "retours": True, "b": 100, "a": 5}, {"N": 50, "V": 0})
chk("C1 cordon frontal seul : f = N/b = 500 N/mm, σ⊥ = τ⊥ = f/(a√2) = 70,7, σ_c = √2·f/a = 141,4",
    pres(r["gouvernant"]["f"], 500.0) and pres(abs(r["gouvernant"]["sigma_perp"]), 500 / (5 * S2)) and pres(r["gouvernant"]["sigma_c"], S2 * 100))
r = verif("C1", {"lateraux": True, "frontaux": 0, "retours": True, "b": 100, "L_rec": 1000, "a": 5}, {"N": 80})
chk("C1 recouvrement long (L = 1000 > 150a = 750) : βLw = 1,2 − 0,2·1000/750 = 0,933 appliqué, message",
    pres(r["beta_Lw"], 1.2 - 0.2 * 1000 / 750) and any("βLw" in m for m in r["messages"]))
# C2
r = verif("C2", {"l": 200, "tB": 10, "deux_cordons": True, "retours": True, "a": 5}, {"N": 40, "Va": 0, "Vp": 0, "M": 0})
chk("C2 T en arrachement, deux cordons : f = N/(2l) = 100 N/mm chacun, |σ⊥| = |τ⊥| = 14,1 N/mm²",
    pres(r["gouvernant"]["f"], 100.0) and pres(abs(r["gouvernant"]["sigma_perp"]), 100 / (5 * S2)))
r = verif("C2", {"l": 200, "tB": 10, "deux_cordons": True, "retours": True, "a": 5}, {"N": 0, "Va": 0, "Vp": 0, "M": 2.0})
chk("C2 T en flexion M = 2 kN·m : couple des deux cordons f = M/(l·t_B) = 1 000 N/mm", pres(r["gouvernant"]["f"], 2e6 / (200 * 10)))
r = verif("C2", {"l": 200, "tB": 10, "deux_cordons": False, "retours": True, "a": 5}, {"N": 0, "Va": 0, "Vp": 0, "M": 2.0})
chk("C2 un seul cordon avec M : refus (flexion autour de son propre axe)", r["statut"] == ET.REFUS and any("inertie" in m for m in r["messages"]))
# C3 : l'exemple de l'étude
r = verif("C3", {"h": 200, "t": 10, "e": 150, "a": 5, "retours": True}, {"V": 60, "N": 0})
chk("C3 console : η = 0,550 (directionnelle), 0,665 (simplifiée) ; ELS avec les mêmes efforts 1,41 ; équivalent 1,00",
    pres(r["comparaison"]["directionnelle"], 0.5496, 1e-3) and pres(r["comparaison"]["simplifiee"], 0.6654, 1e-3)
    and pres(r["comparaison"]["bureau"], 691.5 / (98.1 * 5), 1e-3) and pres(r["comparaison"]["bureau_equiv"], 691.5 / (138.6 * 5), 2e-3),
    str({k: round(v, 3) for k, v in r["comparaison"].items()}))
chk("C3 : gouverne un cordon en tête ou en pied (flexion), statut ok (< 85 %)",
    r["gouvernant"]["position"] in ("haut", "bas") and r["statut"] == ET.OK)
r_els = verif("C3", {"h": 200, "t": 10, "e": 150, "a": 5, "retours": True}, {"V": 40, "N": 0}, {"mode": "els"})
chk("C3 en méthode ELS (V_ser = 40 kN, k = 100) : η = 0,94, statut attention, message avec l'équivalent 138,6",
    pres(r_els["taux_max"], 461.0 / 490.5, 2e-3) and r_els["statut"] == ET.ATTENTION and any("138.6" in m or "138,6" in m for m in r_els["messages"]))
r_els2 = verif("C3", {"h": 200, "t": 10, "e": 150, "a": 5, "retours": True}, {"V": 40, "N": 0}, {"mode": "els", "k": 141.2})
chk("C3 en ELS avec k = valeur Eurocode (141,2) : η = 0,665 comme la simplifiée à l'ELU", pres(r_els2["taux_max"], 0.665, 3e-3))
# C4 gousset : un cordon vertical, V excentré → torsion 6T/l² + cisaillement V/l
r = verif("C4", {"bx": 200, "by": 200, "cotes": ["gauche"], "a": 5, "retours": True, "xV": 100, "yV": 100}, {"V": 30, "H": 0, "N": 0})
f_t, f_v = 6 * 30e3 * 100 / 200 ** 2, 30e3 / 200
chk("C4 gousset, cordon gauche seul, V = 30 kN à e = 100 : |f| au bout = √((6T/l²)² + (V/l)²)",
    pres(r["gouvernant"]["f"], math.hypot(f_t, f_v), 1e-9), str(r["gouvernant"]["f"]))
r = verif("C4", {"bx": 200, "by": 150, "cotes": ["gauche", "haut", "bas"], "a": 5, "retours": True, "xV": 250, "yV": 75}, {"V": 40})
chk("C4 trois côtés : trois cordons, point gouvernant au bout d'un cordon horizontal (le plus loin de G)",
    len(r["cordons"]) == 3 and r["gouvernant"]["cordon"] in ("haut", "bas") and r["gouvernant"]["position"] in ("gauche", "droite"))
# C5 platine : M_y seul, semelles extérieures seules → f = M/(h·b)
r = verif("C5", {"profil": "IPE 200", "semelles": "extérieur", "ame": False, "a_f": 5, "retours": True}, {"N": 0, "Vz": 0, "Vy": 0, "My": 30, "Mz": 0, "T": 0})
chk("C5 platine IPE 200, M_y seul, semelles extérieures : f = M/(h·b) = 30e6/(200·100) = 1 500 N/mm",
    pres(r["gouvernant"]["f"], 30e6 / (200 * 100), 1e-9) and len(r["cordons"]) == 2, str(r["gouvernant"]["f"]))
r = verif("C5", {"profil": "IPE 200", "semelles": "deux faces", "ame": True, "a_f": 5, "a_w": 4, "retours": True}, {"N": 0, "Vz": 60, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
ame = [c for c in r["cordons"] if c["nom"].startswith("âme")]
chk("C5 complet, V_z seul : huit cordons, les cordons d'âme travaillent en τ∥, part de l'âme = a_w·l_w/A_w",
    len(r["cordons"]) == 8 and len(ame) == 2 and all(pres(abs(p["tau_perp"]), 0.0, 1e-9) for p in r["points"] if p["cordon"].startswith("âme")))
# C6 tube : N seul → f = N/périmètre, |σ⊥| = |τ⊥| = f/(a√2)
r = verif("C6", {"profil": "CHS 114.3x5", "a": 4}, {"N": 100, "Vz": 0, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
peri = sum(c["l_eff"] for c in r["cordons"])
chk("C6 tube rond, N seul : 24 tronçons d'un cordon continu (sans longueur minimale propre), f = N/périmètre, |σ⊥| = |τ⊥| = f/(a√2)",
    len(r["cordons"]) == 24 and not r["messages"] and pres(r["gouvernant"]["f"], 100e3 / peri, 1e-6)
    and pres(abs(r["gouvernant"]["sigma_perp"]), 100e3 / peri / (4 * S2), 1e-6), str(r["messages"][:2]))
rs = verif("C6", {"profil": "SHS 100x100x5", "a": 4}, {"N": 0, "Vz": 30, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
chk("C6 tube carré : 20 tronçons (4 côtés + 4 × 4 par coin arrondi), tous utiles, cisaillement réparti sur le périmètre",
    len(rs["cordons"]) == 20 and not rs["messages"] and pres(rs["gouvernant"]["f"], 30e3 / sum(c["l_eff"] for c in rs["cordons"]), 1e-6))
r = verif("C6", {"profil": "CHS 114.3x5", "a": 4}, {"N": 0, "Vz": 0, "Vy": 0, "My": 0, "Mz": 0, "T": 2.0})
chk("C6 tube rond en torsion : τ∥ = T/(2π·R²·a) uniforme (Bredt sur la ligne de gorge)",
    all(pres(abs(p["tau_par"]), 2e6 / (2 * math.pi * (114.3 / 2) ** 2 * 4), 2e-2) for p in r["points"]), str(r["gouvernant"]["tau_par"]))
# C7
r = verif("C7", {"b": 150, "t": 12, "penetration": "pleine"}, {"N": 200, "V": 0, "M": 0})
chk("C7 pleine pénétration : la pièce gouverne, σ = N/(b·t) = 111,1, η = 111,1/235",
    r["pleine"] and pres(r["taux_max"], 200e3 / (150 * 12) / 235) and any("pièce" in m for m in r["messages"]))
r = verif("C7", {"b": 150, "t": 12, "penetration": "partielle", "p": 4, "deux_faces": True}, {"N": 200, "V": 0, "M": 0})
chk("C7 pénétration partielle deux faces p = 4 : σ⊥ = N/(2·b·p) = 166,7 N/mm² (> 0 : traction), τ⊥ = τ∥ = 0, deux soudures",
    pres(r["gouvernant"]["sigma_perp"], 200e3 / (2 * 150 * 4)) and pres(r["gouvernant"]["tau_perp"], 0.0, 1e-9) and len(r["chaines"]) == 2 and not r["pleine"])
r = verif("C7", {"b": 150, "t": 12, "penetration": "partielle", "p": 4, "deux_faces": True}, {"N": 0, "V": 0, "M": 3.0})
chk("C7 partielle en flexion dans le plan : σ⊥ aux bords = 6M/(2·p·b²) = 100 N/mm²",
    pres(abs(r["gouvernant"]["sigma_perp"]), 6 * 3e6 / (2 * 4 * 150 ** 2), 1e-6), str(r["gouvernant"]["sigma_perp"]))
r = verif("C7", {"b": 150, "t": 12, "penetration": "partielle", "p": 6, "deux_faces": True}, {"N": 200, "V": 0, "M": 0})
chk("C7 partielle p = 6 des deux faces sur t = 12 : pénétration complète → pleine pénétration, la pièce gouverne (§4.7.1), note",
    r["pleine"] and any("complète" in m and "§4.7.1" in m for m in r["messages"]) and pres(r["taux_max"], 200e3 / 1800 / 235))
# C8
r = verif("C8")
r3 = verif("C3", {"h": 200, "t": 100, "e": 150, "a": 5, "retours": False}, {"V": 60, "N": 0})
chk("C8 cordons libres (défaut : deux cordons de 200 à ±50, V_y = −60 kN, M_x = 9 kN·m) ≡ C3 h 200, t 100, e 150 : même η",
    r3["statut"] != ET.REFUS and pres(r["taux_max"], r3["taux_max"], 1e-9), f"{r['taux_max']} vs {r3['taux_max']}")

print("\n=== 5. État, JSON, refus ===")
et = ET.nouvel_etat()
chk("état neuf : un cas C3, méthode Eurocode directionnelle, γM2 1,25, k 100, g 9,81, γ_F 1,5",
    et["cas"][0]["config"] == "C3" and et["cas"][0]["methode"] == {"mode": "eurocode", "ec": "directionnelle", "gamma_M2": 1.25, "k": 100.0, "g": 9.81, "gamma_F": 1.5})
c = ET.nouveau_cas("C1"); c["params"]["nuance_B"] = "S355"; ET.changer_config(c, "C5")
chk("changer de configuration : paramètres et efforts de la nouvelle, nuance conservée, nom suivi",
    c["config"] == "C5" and c["params"]["profil"] == "IPE 200" and c["params"]["nuance_B"] == "S355" and "C5" in c["nom"])
et["cas"].append(c)
d2 = ET.depuis_json(ET.vers_json(et))
chk("JSON aller-retour : deux cas, actif borné, champs complétés", len(d2["cas"]) == 2 and d2["actif"] == 0 and "gamma_F" in d2["cas"][1]["methode"])
try:
    ET.depuis_json(json.dumps({"cas": [{"config": "ZZ"}]}))
    chk("JSON : configuration inconnue refusée", False)
except ValueError:
    chk("JSON : configuration inconnue refusée", True)
r = verif("C3", {"a": 2.0})
chk("gorge de 2 mm : aucun cordon utile → refus avec message", r["statut"] == ET.REFUS and any("Refus" in m for m in r["messages"]))
r = verif("C3", {"h": 200, "t": 10, "e": 150, "a": 5}, {"V": 130})
chk("console sous 130 kN : non conforme (η > 1)", r["statut"] == ET.NOK and r["taux_max"] > 1.0)
chk("acier de référence : S355 (B) sur S235 (A) → fu 360, βw 0,8 du S235",
    ET.acier_cordon({"nuance_A": "S235", "nuance_B": "S355"})["fu"] == 360.0 and ET.acier_cordon({"nuance_A": "S235", "nuance_B": "S355"})["bw"] == 0.8)
chk("le moteur n'importe pas Streamlit",
    not any("import streamlit" in open(os.path.join("acier", "soudures", f), encoding="utf-8").read()
            for f in os.listdir(os.path.join("acier", "soudures")) if f.endswith(".py")))

print("\n=== 6. Tracé : chaînes, tronçons, arcs (soudures discontinues, cordons sur n'importe quoi) ===")
from acier.soudures import trace as TR  # noqa: E402
carre = [(0, 0), (100, 0), (100, 100), (0, 100)]
chk("chemin fermé : longueur du carré 400, point à s = 150 → (100, 50)",
    pres(TR.longueur_chemin(carre, True), 400.0) and pres(TR.point_chemin(carre, True, 150)[0], 100.0) and pres(TR.point_chemin(carre, True, 150)[1], 50.0))
sc = TR.sous_chemin(carre, True, 350, 450)
chk("sous-chemin qui franchit l'origine : (0,50) → (0,0) → (50,0), longueur 100",
    len(sc) == 3 and pres(sc[1][0], 0.0, 1e-9) and pres(sc[1][1], 0.0, 1e-9) and pres(TR.longueur_chemin(sc), 100.0))
ch = TR.chaine("c", [(0, 0), (200, 0)], 5.0, +1)
chk("chaîne ouverte de 200, a = 5 : un segment raccourci de a à chaque bout (5 → 195), l_eff = 190, longueur réelle gardée",
    len(ch) == 1 and pres(ch[0].P1[0], 5.0) and pres(ch[0].P2[0], 195.0) and pres(ch[0].l_eff, 190.0) and pres(ch[0].l_chaine, 200.0) and ch[0].utile)
ch = TR.chaine("c", [(0, 0), (35, 0)], 5.0, +1)
chk("chaîne de 35 avec a = 5 : l − 2a = 25 < 30 → exclue (ne compte pas)", ch and not ch[0].utile and ch[0].exclu)
chk("chaîne fermée : pas de bout, longueur entière, 4 segments", len(TR.chaine("p", carre, 5.0, +1, ferme=True)) == 4
    and pres(sum(c.l_eff for c in TR.chaine("p", carre, 5.0, +1, ferme=True)), 400.0))
tr = TR.tronconner("t", [(0, 0), (200, 0)], 5.0, +1, 3, 40.0)
chk("3 tronçons de 40 sur 200 : aux deux bouts et au milieu (0–40, 80–120, 160–200), chacun raccourci de a, espacement 40",
    len(tr) == 3 and pres(tr[0].P1[0], 5.0) and pres(tr[1].P1[0], 85.0) and pres(tr[2].P2[0], 195.0) and all(pres(c.l_eff, 30.0) for c in tr)
    and pres(TR.espacement_troncons(200, False, 3, 40), 40.0))
try:
    TR.tronconner("t", [(0, 0), (200, 0)], 5.0, +1, 4, 60.0)
    chk("tronçons qui ne tiennent pas (4 × 60 > 200) : ValueError", False)
except ValueError:
    chk("tronçons qui ne tiennent pas (4 × 60 > 200) : ValueError", True)
s_haut = TR.abscisse_pres(carre, True, (50.0, 100.0))
tr = TR.tronconner("p", carre, 5.0, +1, 4, 50.0, ferme=True, s_origine=s_haut)
centres = [tuple(np.round((c.P1[:2] + c.P2[:2]) / 2, 6)) for c in tr]
chk("4 tronçons de 50 sur le carré, le premier centré en haut : centres au milieu des quatre côtés",
    pres(s_haut, 250.0) and len(tr) == 4 and (50.0, 100.0) in centres and (0.0, 50.0) in centres and (50.0, 0.0) in centres and (100.0, 50.0) in centres, str(centres))
a4 = TR.arc((0, 0), 100.0, 0.0, 90.0)
chk("arc de 90° r = 100 : 7 points (cordes de 15°), longueur 156,6 (polygone) ; cercle entier : 24 points, fermé",
    len(a4) == 7 and pres(TR.longueur_chemin(a4), 12 * 100 * math.sin(math.radians(7.5)), 1e-6) and len(TR.arc((0, 0), 100.0, 0.0, 360.0)) == 24
    and TR.est_ferme_arc(0.0, 360.0) and not TR.est_ferme_arc(0.0, 180.0))

print("\n=== 7. Les configurations par tronçons et les cordons libres généralisés ===")
r = verif("C6", {"profil": "CHS 114.3x5", "a": 4, "troncons": {"n": 4, "l": 60}}, {"N": 100, "Vz": 0, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
chk("C6 CHS par 4 tronçons de 60, a = 4 : 4 chaînes de l_eff = 52, N seul → f = N/(4·52) = 480,8 N/mm, message §4.3.2.2 avec l'espacement",
    len(r["chaines"]) == 4 and all(pres(c["l_eff"], 52.0) and pres(c["l"], 60.0) for c in r["chaines"]) and pres(r["gouvernant"]["f"], 100e3 / 208.0, 1e-6)
    and any("§4.3.2.2" in m and "4 tronçons" in m for m in r["messages"]), str(r["messages"][:2]))
peri = r["pieces"]["perimetre"]
esp = [m for m in r["messages"] if "espacement" in m][0]
chk("C6 : espacement = périmètre/4 − 60 écrit dans le message", f"espacement {peri / 4 - 60:.0f} mm" in esp, esp)
rs = verif("C6", {"profil": "SHS 100x100x5", "a": 4, "troncons": {"n": 4, "l": 50}}, {"N": 0, "Vz": 30, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
cent = [np.round((np.array(c["p1"][:2]) + np.array(c["p2"][:2])) / 2, 3) for c in rs["cordons"]]
chk("C6 SHS par 4 tronçons de 50 : un tronçon droit centré sur chaque côté (centre du premier en (0 ; 50))",
    len(rs["chaines"]) == 4 and any(abs(c[0]) < 1e-6 and abs(c[1] - 50.0) < 1e-6 for c in cent), str(cent))
r = verif("C3", {"h": 200, "t": 10, "e": 150, "a": 5, "troncons": {"n": 2, "l": 60}}, {"V": 60, "N": 0})
Ix = 4 * 5 * (50 ** 3 / 12 + 50 * 70 ** 2)
f_att = math.hypot(5 * 9e6 * 95 / Ix, 5 * 60e3 / 1000)
chk("C3 par 2 tronçons de 60 par côté (h = 200) : 4 chaînes de 50 efficaces centrées à ±70, |f| en tête = √((a·M·95/I_x)² + (a·V/A_w)²)",
    len(r["chaines"]) == 4 and pres(r["gouvernant"]["f"], f_att, 1e-6) and r["gouvernant"]["position"] in ("haut", "bas"), f"{r['gouvernant']['f']} vs {f_att}")
chk("C3 tronçons : message §4.3.2.2 avec les limites pour t = 10 (12 t = 120, 16 t = 160), espacement 80 non dépassé",
    any("espacement 80 mm" in m and "12 t = 120" in m and "DÉPASSÉ" not in m for m in r["messages"]), str(r["messages"]))
r = verif("C3", {"h": 400, "t": 10, "e": 150, "a": 5, "troncons": {"n": 2, "l": 50}}, {"V": 60, "N": 0})
chk("C3 h = 400, 2 tronçons de 50 : espacement 300 > min(200 ; 16 t) → « DÉPASSÉ » dans le message",
    any("espacement 300 mm" in m and "DÉPASSÉ" in m for m in r["messages"]), str(r["messages"]))
r = verif("C3", {"h": 200, "t": 10, "e": 150, "a": 5, "troncons": {"n": 4, "l": 60}}, {"V": 60, "N": 0})
chk("C3 : 4 tronçons de 60 ne tiennent pas sur 200 → refus", r["statut"] == ET.REFUS and any("ne tiennent pas" in m for m in r["messages"]))
r = verif("C1")
chk("C1 par défaut : le cordon frontal est au bout du plat B (x = 0), nommé ainsi ; deux frontaux → le bout de A en x = L",
    any(c["nom"] == "frontal (bout de B)" and pres(c["p1"][0], 0.0, 1e-9) for c in r["cordons"])
    and any(c["nom"] == "frontal (bout de A)" and pres(c["p1"][0], 120.0, 1e-9) for c in verif("C1", {"frontaux": 2})["cordons"]))
r = verif("C1", {"troncons": {"n": 2, "l": 40}, "frontaux": 0, "retours": False, "L_rec": 120, "b": 100, "a": 5}, {"N": 50, "V": 0})
chk("C1 latéraux par 2 tronçons de 40 (bouts), frontaux 0 : 4 chaînes de 30 efficaces, τ∥ = N/(4·a·30)",
    len(r["chaines"]) == 4 and pres(abs(r["gouvernant"]["tau_par"]), 50e3 / (4 * 5 * 30), 1e-6))
r = verif("C4", {"bx": 200, "by": 150, "cotes": ["gauche", "haut", "bas"], "xV": 250, "yV": 75})
chk("C4 : l'appui couvre la zone soudée (0 … b_x) et le plat déborde jusqu'au point d'application (porte-à-faux)",
    max(x for x, _ in r["pieces"]["A"]) == 225 and max(x for x, _ in r["pieces"]["B"]) == 270)
r = verif("C8", {"elements": [{"type": "arc", "xc": 0, "yc": 0, "r": 50, "ang1": 0, "ang2": 360, "a": 5, "cote": 1}]},
          {"N": 100, "Vx": 0, "Vy": 0, "Mx": 0, "My": 0, "Mz": 0})
peri = sum(c["l_eff"] for c in r["cordons"])
chk("C8 arc fermé r = 50 (cercle, 24 cordes), N seul : une chaîne « arc 1 » de 24 segments, f = N/périmètre, |σ⊥| = |τ⊥| = f/(a√2)",
    len(r["chaines"]) == 1 and r["chaines"][0]["segments"] == 24 and r["chaines"][0]["nom"] == "arc 1"
    and pres(r["gouvernant"]["f"], 100e3 / peri, 1e-6) and pres(abs(r["gouvernant"]["sigma_perp"]), 100e3 / peri / (5 * S2), 1e-6), str(r["messages"][:2]))
r = verif("C8", {"elements": [{"type": "arc", "xc": 0, "yc": 0, "r": 50, "ang1": 0, "ang2": 180, "a": 5, "cote": 1},
                              {"type": "segment", "x1": -50, "y1": 0, "x2": 50, "y2": 0, "a": 5, "cote": 1}]},
          {"N": 100, "Vx": 0, "Vy": 0, "Mx": 0, "My": 0, "Mz": 0})
demi = [c for c in r["chaines"] if c["nom"] == "arc 1"][0]
chk("C8 demi-cercle ouvert + corde : l'arc est une chaîne raccourcie de a à chaque bout (l_eff = l − 2a), la corde un cordon droit (100 − 2a = 90)",
    pres(demi["l_eff"], demi["l"] - 10.0, 1e-6) and pres([c for c in r["chaines"] if c["nom"] == "cordon 2"][0]["l_eff"], 90.0) and r["statut"] != ET.REFUS, str(r["messages"][:2]))
r = verif("C8", {"elements": [{"type": "arc", "xc": 0, "yc": 0, "r": 60, "ang1": 0, "ang2": 360, "a": 4, "cote": 1, "troncons": {"n": 3, "l": 50}}]},
          {"N": 100, "Vx": 0, "Vy": 0, "Mx": 0, "My": 0, "Mz": 0})
chk("C8 cercle par 3 tronçons de 50, a = 4 : 3 chaînes de 42 efficaces, le premier tronçon centré en haut",
    len(r["chaines"]) == 3 and all(pres(c["l_eff"], 42.0, 1e-6) for c in r["chaines"])
    and any(abs((c["p1"][0] + c["p2"][0]) / 2) < 8 and (c["p1"][1] + c["p2"][1]) / 2 > 55 for c in r["cordons"]), str(r["chaines"]))
vieux = json.dumps({"cas": [{"nom": "x", "config": "C8", "params": {"cordons": [{"x1": 0, "y1": 0, "x2": 0, "y2": 200, "a": 5, "cote": 1}]},
                             "efforts": {"Vx": 10}}]})
d = ET.depuis_json(vieux)
chk("fichier ancien (C8 par « cordons ») : converti en « elements » de type segment, nom donné",
    "cordons" not in d["cas"][0]["params"] and d["cas"][0]["params"]["elements"][0]["type"] == "segment"
    and d["cas"][0]["params"]["elements"][0]["nom"] == "cordon 1" and len(d["cas"][0]["params"]["elements"]) == 1)
r = verif("C5", {"profil": "IPE 200", "semelles": "extérieur", "ame": False, "a_f": 5, "troncons_semelles": {"n": 2, "l": 30}},
          {"N": 0, "Vz": 0, "Vy": 0, "My": 30, "Mz": 0, "T": 0})
chk("C5 semelles par 2 tronçons de 30 : l − 2a = 20 < 30 → toutes les chaînes exclues, refus avec le message de la chaîne",
    r["statut"] == ET.REFUS and r["chaines"] == [] and any("ne compte pas" in m and "l = 30 mm" in m for m in r["messages"]), str(r["messages"][:2]))
r = verif("C5", {"profil": "IPE 200", "semelles": "extérieur", "ame": False, "a_f": 5, "troncons_semelles": {"n": 2, "l": 50}},
          {"N": 0, "Vz": 0, "Vy": 0, "My": 30, "Mz": 0, "T": 0})
chk("C5 semelles extérieures par 2 tronçons de 50 (bouts de semelle) : 4 chaînes de 40 efficaces, f = M/(h·Σl_eff) = 1 875 N/mm",
    len(r["chaines"]) == 4 and pres(r["gouvernant"]["f"], 30e6 / (200 * 80), 1e-6), str(r["gouvernant"]["f"]))
chk("sortie : chaque cordon porte u_A (hors de la pièce rapportée) pour le dessin ; C3 cordon droit → u_A = +x",
    all(len(c["u_A"]) == 3 for c in r["cordons"]) and pres(verif("C3")["cordons"][0]["u_A"][0], 1.0, 1e-9))

print("\n=== 8. Types de jonction : pénétration partielle, pleine pénétration = la pièce (EN 1993-1-8 §4.7) ===")
r = verif("C2", {"l": 200, "tB": 30, "tA": 30, "jonction": "pleine"}, {"N": 400, "Va": 0, "Vp": 0, "M": 0})
chk("C2 pleine pénétration (plats de 30), N = 400 kN : σ = N/(l·t) = 66,7 N/mm² sur la section, η = σ/fy = 0,284 ; UNE soudure de gorge t = 30 et l = 200, message §4.7.1",
    r["pleine"] and pres(abs(r["gouvernant"]["sigma_perp"]), 400e3 / 6000) and pres(r["taux_max"], 400e3 / 6000 / 235)
    and len(r["chaines"]) == 1 and pres(r["chaines"][0]["a"], 30.0) and pres(r["chaines"][0]["l_eff"], 200.0) and r["chaines"][0]["segments"] == 1
    and any("§4.7.1" in m and "pièce" in m for m in r["messages"]), str(r["chaines"]))
r = verif("C2", {"l": 200, "tB": 30, "tA": 30, "jonction": "pleine"}, {"N": 0, "Va": 0, "Vp": 0, "M": 10})
chk("C2 pleine en flexion M = 10 kN·m : les trois lignes de gorge (Simpson) rendent I = l·t³/12 exactement → σ au bord = 6M/(l·t²) = 333,3 N/mm²",
    pres(abs(r["gouvernant"]["sigma_perp"]), 6 * 10e6 / (200 * 30 ** 2), 1e-9) and "bord" in r["gouvernant"]["cordon"], str(r["gouvernant"]["sigma_perp"]))
r = verif("C2", {"l": 200, "tB": 30, "tA": 30, "jonction": "pleine"}, {"N": 0, "Va": 50, "Vp": 0, "M": 0})
chk("C2 pleine en cisaillement V∥ = 50 kN : τ∥ = V/(l·t) = 8,33 N/mm², σ⊥ = 0 → critère de la pièce √(3)·τ ≤ fy/γM0",
    pres(abs(r["gouvernant"]["tau_par"]), 50e3 / 6000) and pres(r["gouvernant"]["sigma_perp"], 0.0, 1e-9) and pres(r["taux_max"], S3 * 50e3 / 6000 / 235, 1e-9))
r = verif("C2", {"l": 200, "tB": 30, "tA": 30, "jonction": "pleine"}, {"N": 400, "Va": 0, "Vp": 0, "M": 0}, {"mode": "els"})
chk("C2 pleine en méthode ELS (efforts de service) : la pièce est comparée à fy/γ_F = 156,7 → η = 0,426, message",
    pres(r["taux_max"], 400e3 / 6000 / (235 / 1.5), 1e-9) and any("fy/γ_F" in m for m in r["messages"]))
r = verif("C2", {"l": 200, "tB": 10, "jonction": "partielle", "p": 4, "renfort": 0, "retours": True}, {"N": 40, "Va": 0, "Vp": 0, "M": 0})
chk("C2 pénétration partielle p = 4 des deux faces, sans renfort : gorge p parallèle à la face, σ⊥ = N/(2·p·l) = 25,0 N/mm², τ = 0, deux soudures « bout à bout »",
    len(r["chaines"]) == 2 and all(c["bout_a_bout"] and pres(c["a"], 4.0) for c in r["chaines"]) and pres(abs(r["gouvernant"]["sigma_perp"]), 40e3 / (2 * 4 * 200))
    and pres(r["gouvernant"]["tau_perp"], 0.0, 1e-9) and not r["pleine"], str(r["chaines"]))
r = verif("C2", {"l": 200, "tB": 10, "jonction": "partielle", "p": 3, "renfort": 3}, {"N": 40, "Va": 0, "Vp": 0, "M": 0})
chk("C2 partielle p = 3 avec renfort a_r = 3 : cordon d'angle à pénétration profonde de gorge p + a_r = 6 (§4.5.2(3)), composantes à 45° (|σ⊥| = |τ⊥|)",
    len(r["chaines"]) == 2 and all(not c["bout_a_bout"] and pres(c["a"], 6.0) for c in r["chaines"])
    and pres(abs(r["gouvernant"]["sigma_perp"]), abs(r["gouvernant"]["tau_perp"])), str(r["chaines"]))
r = verif("C2", {"l": 200, "tB": 10, "jonction": "partielle", "p": 5, "renfort": 0}, {"N": 40, "Va": 0, "Vp": 0, "M": 0})
chk("C2 partielle p = 5 des deux faces sur t = 10 : 2p ≥ t et c = 0 ≤ min(t/5 ; 3) → traitée comme pleine (§4.7.3), note dans les messages",
    r["pleine"] and any("§4.7.3" in m for m in r["messages"]), str(r["messages"][:1]))
r = verif("C2", {"l": 200, "tB": 10, "jonction": "partielle", "p": 5, "renfort": 0, "deux_cordons": False}, {"N": 0, "Va": 40, "Vp": 0, "M": 0})
chk("C2 partielle p = 5 d'une seule face (cisaillée le long) : pas de règle du T, une soudure de gorge 5",
    r["statut"] != ET.REFUS and not r["pleine"] and len(r["chaines"]) == 1 and pres(r["chaines"][0]["a"], 5.0), str(r["messages"][:1]))
r = verif("C3", {"h": 200, "t": 30, "e": 150, "jonction": "pleine"}, {"V": 60, "N": 0})
chk("C3 console en pleine pénétration (plat de 30 sur l'aile d'un HEB 200) : σ en tête = 6·V·e/(t·h²) = 45,0 N/mm², η = 0,191 ; poteau du catalogue dans les pièces, t_min = min(t ; t_f) = 15",
    r["pleine"] and pres(abs(r["gouvernant"]["sigma_perp"]), 6 * 60e3 * 150 / (30 * 200 ** 2), 1e-9) and r["pieces"]["poteau"]["nom"] == "HEB 200"
    and pres(r["pieces"]["t_min"], 15.0), str(r["gouvernant"]))
chk("C3 sur un plat (poteau = « plat ») : pas de profilé support, t_min = t", verif("C3", {"poteau": "plat"})["pieces"]["poteau"] is None and pres(verif("C3", {"poteau": "plat"})["pieces"]["t_min"], 10.0))
chk("C3 par défaut (cordons d'angle) : face « droit » à x = +t/2, face « gauche » à −t/2",
    all(pres(c["p1"][0], 5.0 if c["nom"] == "cordon droit" else -5.0) for c in verif("C3")["cordons"]) and len(verif("C3")["cordons"]) == 2)
r = verif("C5", {"profil": "IPE 200", "jonction_semelles": "pleine", "ame": False, "retours": True}, {"N": 0, "Vz": 0, "Vy": 0, "My": 30, "Mz": 0, "T": 0})
I_f = 2 * (100 * 8.5 ** 3 / 12 + 100 * 8.5 * ((200 - 8.5) / 2) ** 2)
chk("C5 semelles à pleine pénétration, M_y = 30 kN·m : I des deux semelles exact (b·tf³/12 + b·tf·d²), σ = M·(h/2)/I = 192,4 N/mm², deux soudures de gorge t_f = 8,5",
    r["pleine"] and pres(abs(r["gouvernant"]["sigma_perp"]), 30e6 * 100 / I_f, 1e-9) and len(r["chaines"]) == 2 and all(pres(c["a"], 8.5) for c in r["chaines"]),
    str(r["gouvernant"]["sigma_perp"]))
r = verif("C5", {"profil": "IPE 200", "jonction_semelles": "pleine", "jonction_ame": "angle", "ame": True, "a_w": 4, "retours": True}, {"N": 0, "Vz": 60, "Vy": 0, "My": 30, "Mz": 0, "T": 0})
chk("C5 semelles pleines + âme en cordons d'angle : groupe mixte, 6 lignes pleines, pas « tout plein », chaque point selon SON critère (pièce ou cordon)",
    not r["pleine"] and r["n_pleine"] == 6 and any(p["pleine"] for p in r["points"]) and any(not p["pleine"] for p in r["points"])
    and all(("pièce" in p["critere"]) == p["pleine"] for p in r["points"]))
r = verif("C5", {"profil": "IPE 200", "jonction_semelles": "angle", "jonction_ame": "pleine", "ame": True, "a_f": 5, "retours": True}, {"N": 0, "Vz": 60, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
chk("C5 âme à pleine pénétration : faces « droite » à x = +t_w/2 et « gauche » à −t_w/2 (lignes de bord), gorge t_w = 5,6",
    any(c["nom"] == "âme (bord 1)" and pres(c["p1"][0], 2.8, 1e-9) for c in r["cordons"]) and any(c["nom"] == "âme (bord 2)" and pres(c["p1"][0], -2.8, 1e-9) for c in r["cordons"])
    and pres([c for c in r["chaines"] if c["nom"] == "âme"][0]["a"], 5.6), str([(c["nom"], c["p1"]) for c in r["cordons"] if c["nom"].startswith("âme")]))
r = verif("C6", {"profil": "CHS 114.3x5", "jonction": "pleine"}, {"N": 100, "Vz": 0, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
peri = sum(c["l_eff"] for c in r["cordons"])
chk("C6 tube rond à pleine pénétration, N = 100 kN : σ = N/(t·périmètre moyen) sur toute la paroi (ligne moyenne, 24 cordes), une chaîne « pourtour » de gorge t = 5",
    r["pleine"] and pres(abs(r["gouvernant"]["sigma_perp"]), 100e3 / (5 * peri), 1e-9) and pres(r["chaines"][0]["a"], 5.0)
    and pres(peri, 24 * 109.3 * math.sin(math.radians(7.5)), 1e-6), str(r["chaines"]))
r = verif("C6", {"profil": "CHS 114.3x5", "jonction": "partielle", "p": 3, "renfort": 0}, {"N": 100, "Vz": 0, "Vy": 0, "My": 0, "Mz": 0, "T": 0})
chk("C6 tube en pénétration partielle p = 3 sans renfort : gorge 3 parallèle à la platine, σ⊥ = N/(p·périmètre), τ = 0",
    pres(abs(r["gouvernant"]["sigma_perp"]), 100e3 / (3 * sum(c["l_eff"] for c in r["cordons"])), 1e-9) and pres(r["gouvernant"]["tau_perp"], 0.0, 1e-9) and not r["pleine"])
r = verif("C7", {"b": 150, "t": 12, "penetration": "pleine"}, {"N": 0, "V": 0, "M": 3.0})
chk("C7 pleine en flexion dans le plan : la soudure vaut le plat, σ = 6M/(t·b²) = 66,7 N/mm² (une seule soudure de gorge t dans le groupe)",
    pres(abs(r["gouvernant"]["sigma_perp"]), 6 * 3e6 / (12 * 150 ** 2), 1e-9) and len(r["cordons"]) == 1 and r["cordons"][0]["pleine"])
r = verif("C7", {"b": 150, "t": 12, "penetration": "pleine"}, {"N": 200, "V": 0, "M": 0}, {"mode": "els"})
chk("C7 pleine en ELS : σ ≤ fy/γ_F = 156,7 → η = 0,709", pres(r["taux_max"], 200e3 / 1800 / (235 / 1.5), 1e-9))
gp = K.gorge_pleine_resistance(30.0, 235.0, FU, BW, GM2)
chk("gorge de pleine résistance S235 : deux cordons d'angle équivalents au plat tendu → a ≥ 0,46·t (directionnelle), 0,57·t (simplifiée) ; t = 30 → 13,8 et 17,0 mm",
    pres(gp["ratio_dir"], S2 * BW * GM2 * 235 / (2 * FU)) and pres(gp["ratio_simp"], 235 / (2 * FU / (S3 * BW * GM2)))
    and pres(gp["directionnelle"], 13.85, 1e-3) and pres(gp["simplifiee"], 16.96, 1e-3))
chk("a_pleine rendu par la vérification dès que la pièce rapportée a une épaisseur (C2 t_B = 30)", pres(verif("C2", {"tB": 30})["a_pleine"]["directionnelle"], gp["directionnelle"]))
mb = K.metal_base(100.0, 0.0, 50.0, 235.0)
chk("métal de base : √(σ² + 3τ²) = 132,3 ≤ fy/γM0 = 235, critère nommé « pièce » ; en service, fy/γ_F",
    pres(mb["sigma_c"], math.hypot(100, 50 * S3)) and pres(mb["lim_c"], 235.0) and "pièce" in mb["critere"]
    and pres(K.metal_base(100.0, 0.0, 0.0, 235.0, 1.5, service=True)["lim_c"], 235 / 1.5))
r = verif("C2", {"l": 200, "tB": 30, "tA": 30, "jonction": "pleine", "troncons": {"n": 3, "l": 40}}, {"N": 400, "Va": 0, "Vp": 0, "M": 0})
chk("C2 pleine : les tronçons ne s'appliquent pas (la soudure vaut la pièce sur toute la longueur), pas de message de soudure discontinue",
    r["pleine"] and pres(r["chaines"][0]["l_eff"], 200.0) and not any("discontinue" in m for m in r["messages"]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

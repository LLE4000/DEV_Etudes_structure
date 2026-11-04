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
chk("le long : τ∥ = f/a seul", pres(r3["tau_par"], 20.0) and pres(r3["sigma_perp"], 0.0) and pres(r3["tau_perp"], 0.0))
chk("ailes perpendiculaires à l'axe et entre elles ; u_A hors de la pièce (pièce à gauche de +x → u_A = −y)",
    pres(float(np.dot(c.e_a, c.u_A)), 0.0) and pres(float(np.dot(c.u_A, c.u_B)), 0.0) and tuple(np.round(c.u_A, 9)) == (0.0, -1.0, 0.0))
cb = C.cordon_bout_a_bout("b", (0, -50), (0, 50), 6.0, cote=+1)
rb = cb.composantes((120, 0, 0))
chk("bout à bout partiel : gorge perpendiculaire à n = x → σ⊥ = f/a, τ⊥ = τ∥ = 0",
    pres(abs(rb["sigma_perp"]), 20.0) and pres(rb["tau_perp"], 0.0, 1e-9) and pres(rb["tau_par"], 0.0, 1e-9))
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
r = verif("C7", {"b": 150, "t": 12, "penetration": "partielle", "p": 6, "deux_faces": True}, {"N": 200, "V": 0, "M": 0})
chk("C7 pénétration partielle deux faces p = 6 : σ⊥ = N/(2·b·p) = 111,1 N/mm², τ⊥ = τ∥ = 0",
    pres(abs(r["gouvernant"]["sigma_perp"]), 200e3 / (2 * 150 * 6)) and pres(r["gouvernant"]["tau_perp"], 0.0, 1e-9))
r = verif("C7", {"b": 150, "t": 12, "penetration": "partielle", "p": 6, "deux_faces": True}, {"N": 0, "V": 0, "M": 3.0})
chk("C7 partielle en flexion dans le plan : σ⊥ aux bords = 6M/(2·p·b²) = 66,7 N/mm²",
    pres(abs(r["gouvernant"]["sigma_perp"]), 6 * 3e6 / (2 * 6 * 150 ** 2), 1e-6), str(r["gouvernant"]["sigma_perp"]))
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

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

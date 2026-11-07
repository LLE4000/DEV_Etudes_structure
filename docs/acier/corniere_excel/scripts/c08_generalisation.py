# -*- coding: utf-8 -*-
"""c08 — Ce qui empêche de généraliser les formules du classeur.

1. cornière RETOURNÉE (configuration B : aile horizontale en haut, aile verticale vers le bas) : équilibre
   refait, comparé à ce que rendrait le classeur si on le remplissait tel quel ;
2. ailes inégales : RECHERCHEV sur des noms en double (orientation inaccessible) ; rayon r2 au lieu de r1 ;
3. ce que larg. et haut. changent réellement dans la feuille (presque rien) ;
4. épaisseur et nuance : f_y saisi à la main.
"""
import copy
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import commun as K  # noqa: E402

RACINE = os.path.dirname(ICI)
p = K.CAS
e, r, X, Y, R, haut = p["e"], p["r"], p["X"], p["Y"], p["R"], p["haut"]
b = p["b_cm"] * 10
W, A = b * e ** 2 / 6, b * e
xl = K.classeur(**p)
adm = xl["adm"]
res = {}

# ---------------------------------------------------------------- 1. configuration B
# y' = distance depuis la face supérieure (talon, en haut) vers le bas. L'aile verticale descend jusqu'à y' = haut
# (bout libre). Sous la charge, la cornière tourne autour de son BOUT INFÉRIEUR (compression contre le mur près du
# bout), la cheville est tendue. Moment autour du point de compression (bout, décalé de δ_c) : T·(haut − δ_c − y'_a) = R·X.
# Aile verticale : de la cheville au coin, M = T·(haut − δ_c − y') − T·(y'_a − y') ... = T·(haut − δ_c − y'_a) = R·X
# (constant : le couple cheville / compression est entièrement sous la section) ; de la cheville au bout, M décroît.
def config_B(ya, dc):
    z = haut - dc - ya
    T = R * X / z
    M_coin = R * X / 1000                     # kN·m, constant entre la cheville et le coin (si ya > e + r)
    sig = M_coin * 1e6 / W
    tau = R * 1e3 / A                          # même convention que le classeur
    eq = math.sqrt(sig ** 2 + 3 * tau ** 2)
    return dict(ya=ya, dc=dc, z=z, T=T, N_par_cheville=T / p["n_anc"], M3=M_coin, sigma3=sig, taux3=eq / adm)


print("Configuration B (aile horizontale en haut) — même cornière, même charge :")
B1 = config_B(85.0, 10.0)                     # on garde « Y = 85 » compté depuis l'aile horizontale
B2 = config_B(35.0, 10.0)                     # cheville à 85 mm du BOUT (même bras qu'en A)
for nom, Bx in (("cheville à 85 mm de l'aile horizontale", B1), ("cheville à 85 mm du bout libre", B2)):
    print(f"  {nom:42s} : bras {Bx['z']:.0f} mm → T = {Bx['T']:.2f} kN ({Bx['N_par_cheville']:.2f} par cheville ; "
          f"classeur {xl['N_ELS']:.2f}) ; M à la section 3 = {Bx['M3']:.3f} kN·m (classeur {xl['S3']['M']:.3f}) → "
          f"taux S3 = {Bx['taux3'] * 100:.1f} % (classeur {xl['S3']['taux'] * 100:.2f} %)")
res["config_B"] = dict(cheville_85_depuis_aile=B1, cheville_85_depuis_bout=B2,
                       rapport_M3=B1["M3"] / xl["S3"]["M"], rapport_T_cas1=B1["T"] / (xl["N_ELS"] * 3))
print(f"  → le classeur rempli tel quel sous-estime : M3 ×{B1['M3'] / xl['S3']['M']:.3f} ; "
      f"T ×{B1['T'] / (xl['N_ELS'] * 3):.2f} (cas 1) ou ×{B2['T'] / (xl['N_ELS'] * 3):.2f} (cas 2)")
print("  aile verticale entre le coin et la cheville : effort normal R en COMPRESSION (cheville sous la charge),"
      " au lieu d'une traction en configuration A")

# ---------------------------------------------------------------- 2. ailes inégales
print("\nAiles inégales : L150x100x12 (lignes 6 et 7 du catalogue du classeur)")
d = copy.deepcopy(p)
d.update(larg=150.0, haut=100.0, e=12.0, r=6.0, X=75.0, Y=70.0)          # ce que RECHERCHEV rend (ligne 6)
c_vl = K.classeur(**d)
d2 = copy.deepcopy(d)
d2.update(larg=100.0, haut=150.0)                                         # l'orientation voulue (ligne 7)
c_ok = K.classeur(**d2)
d3 = copy.deepcopy(d2)
d3.update(r=12.0)                                                          # rayon de congé r1 (EN 10056-1, site)
c_r1 = K.classeur(**d3)
for nom, c in (("RECHERCHEV (larg 150, haut 100, r = 6)", c_vl), ("ligne 7 voulue (larg 100, haut 150, r = 6)", c_ok),
               ("ligne 7 avec r1 = 12", c_r1)):
    print(f"  {nom:44s} : taux S1 {c['S1']['taux'] * 100:6.2f} %, S2 {c['S2']['taux'] * 100:6.2f} %, "
          f"S3 {c['S3']['taux'] * 100:6.2f} %, S4 {c['S4']['taux'] * 100:6.2f} % ; max {c['D3_taux_max'] * 100:6.2f} % ; "
          f"larg − X = {c['A20_larg_moins_X']:+.0f} ; haut − Y = {c['A21_haut_moins_Y']:+.0f} ; C90 = {c['C90_arrachement']:.1f}")
res["inegales"] = dict(recherchev=dict(taux=c_vl["D3_taux_max"], C90=c_vl["C90_arrachement"]),
                       voulue=dict(taux=c_ok["D3_taux_max"], C90=c_ok["C90_arrachement"]),
                       r1=dict(taux=c_r1["D3_taux_max"]))

# ---------------------------------------------------------------- 3. ce que larg. et haut. changent
base = K.classeur(**p)
var = K.classeur(**{**p, "larg": 60.0, "haut": 400.0})
diff = [k for k in ("S1", "S2", "S3", "S4") if base[k]["taux"] != var[k]["taux"]]
print(f"\nlarg. 120 → 60 et haut. 120 → 400 : sections dont le taux change : {diff or 'aucune'} ; "
      f"N_ELS identique : {base['N_ELS'] == var['N_ELS']} ; C90 {base['C90_arrachement']:.2f} → {var['C90_arrachement']:.2f}")
res["larg_haut_sans_effet"] = dict(sections_modifiees=diff)
json.dump(res, open(os.path.join(RACINE, "sorties", "c08_generalisation.json"), "w"), ensure_ascii=False, indent=1,
          default=float)

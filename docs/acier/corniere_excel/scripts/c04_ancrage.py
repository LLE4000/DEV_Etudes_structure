# -*- coding: utf-8 -*-
"""c04 — Modèle d'ancrage, pivots et jeux : ce que le classeur suppose, ce que l'équilibre impose.

1. les DEUX pivots du classeur (sections : e/2 ; ancrages : C76) et l'effort de traction qui en résulte ;
2. l'équilibre complet du corps libre (réaction verticale de la cheville à x_V, compression à y_c) ;
3. la valeur HILTI notée en G85 face à C85 ;
4. un jeu de 20 mm (la valeur par défaut demandée) mis dans chacune des trois cases Xj, Yj, C76 — et le piège
   d'unité de C76 (cm) ;
5. sensibilité à Y, charge au-delà de l'aile, ancrage hors de l'aile : ce que le classeur rend sans prévenir.
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
SORTIE = os.path.join(RACINE, "sorties", "c04_ancrage.json")
p = K.CAS
e, r, X, Y, R = p["e"], p["r"], p["X"], p["Y"], p["R"]
xl = K.classeur(**p)
res = {}

# ---------------------------------------------------------------- 1. deux pivots dans la même feuille
T_sec = R * X / xl["C25_YH"]                # ce que les sections 3-4 supposent : T·YH = R·X (pivot à e/2)
T_anc = xl["N_ELS"] * p["n_anc"]           # ce que C82 calcule : R·X/(Y − 10 mm)
print(f"Pivot des sections 3-4 : y = e/2 = {e / 2} mm → T = R·X/YH = {T_sec:.3f} kN")
print(f"Pivot des ancrages (C76 = 1 cm) : y = 10 mm → T = R·X/(Y − 10) = {T_anc:.3f} kN (×{T_anc / T_sec:.4f})")
res["deux_pivots"] = dict(T_sections=T_sec, T_ancrages=T_anc, rapport=T_anc / T_sec)

# ---------------------------------------------------------------- 2. équilibre complet
# moment autour du point de compression (0, y_c) : T·(Y − y_c) = R·(X + Xj − x_V), x_V = abscisse de la réaction
# verticale de la cheville (0 : au nu du mur ; e/2 : milieu de l'appui de la tige dans le trou).
tab = []
for x_V in (0.0, e / 2):
    for y_c in (0.0, e / 2, 10.0, 15.0):
        T = R * (X - x_V) / (Y - y_c)
        tab.append(dict(x_V=x_V, y_c=y_c, T=T, rapport_C82=T / T_anc))
        print(f"  x_V = {x_V:4.1f}, y_c = {y_c:4.1f} → T = {T:6.2f} kN ({(T / T_anc - 1) * 100:+5.1f} % / classeur C82)")
res["equilibre"] = tab
# lecture des EF (c03) si disponibles
try:
    ef = json.load(open(os.path.join(RACINE, "sorties", "c03_ef_h0.5.json")))
    res["EF"] = dict(base=ef["base"], cheville=ef["variantes_cheville"], mur=ef["variantes_mur"])
    print("EF (c03) — T selon la raideur de la cheville :",
          ", ".join(f"δ={v['cheville_delta_mm']}: {v['T_kN']:.1f} kN (rondelle y={v['y_rondelle']:.0f})"
                    for v in ef["variantes_cheville"]))
except FileNotFoundError:
    print("(c03 pas encore lancé)")

# ---------------------------------------------------------------- 3. HILTI (G85, G86)
g85, g86 = 24.68, 22.97
z_h = p["R_elu"] * X / (p["n_anc"] * g85)
print(f"G85 = {g85} kN contre C85 = {xl['N_ELU']:.3f} kN → ×{g85 / xl['N_ELU']:.4f} ; bras implicite = {z_h:.2f} mm "
      f"(pivot implicite à {Y - z_h:.2f} mm) ; G86 = {g86} = C86 = {xl['V_ELU']:.3f}")
print(f"Rapport ELU/ELS du classeur C71/C70 = {p['R_elu'] / R:.4f} (au-delà de γ_Q = 1,50)")
res["hilti"] = dict(G85=g85, C85=xl["N_ELU"], rapport=g85 / xl["N_ELU"], bras=z_h, pivot=Y - z_h,
                    rapport_ELU_ELS=p["R_elu"] / R)


# ---------------------------------------------------------------- 4. un jeu de 20 mm dans chaque case
def cas(**kw):
    d = copy.deepcopy(p)
    d.update(kw)
    return K.classeur(**d)


def ligne(nom, c):
    return dict(nom=nom, M1=c["S1"]["M"], M3=c["S3"]["M"], MMax=c["MMax"], taux1=c["S1"]["taux"],
                taux3=c["S3"]["taux"], taux_max=c["D3_taux_max"], N_ELS=c["N_ELS"], z=c["z_ancrage_mm"])


jeux = [ligne("cas de contrôle", xl),
        ligne("Xj = 20 mm (C14)", cas(Xj=20.0)),
        ligne("Yj = 20 mm (C17)", cas(Yj=20.0)),
        ligne("Yj = −20 mm (C17)", cas(Yj=-20.0)),
        ligne("C76 = 2 cm (= 20 mm)", cas(jeu_anc_cm=2.0)),
        ligne("C76 = 20 (« 20 mm » tapé dans une case en cm)", cas(jeu_anc_cm=20.0)),
        ligne("Xj = −20 mm (C14)", cas(Xj=-20.0)),
        ligne("C8 = 300 (« 300 mm » tapé dans une case en cm)", cas(b_cm=300.0))]
print("\nUn jeu de 20 mm, case par case :")
for j in jeux:
    print(f"  {j['nom']:48s} M1 = {j['M1']:.3f}  M3 = {j['M3']:.3f}  taux S1 = {j['taux1'] * 100:6.2f} %  "
          f"taux S3 = {j['taux3'] * 100:6.2f} %  N_ELS/cheville = {j['N_ELS']:8.3f} kN (bras {j['z']:.0f} mm)")
res["jeux"] = jeux

# ---------------------------------------------------------------- 5. sensibilité à Y et domaines non contrôlés
print("\nSensibilité à Y (autres données du cas) :")
sens = []
for Yv in (25.0, 28.0, 33.0, 40.0, 60.0, 85.0, 100.0, 110.0, 120.0, 130.0):
    try:
        c = cas(Y=Yv)
        d = dict(Y=Yv, Ye=c["C23_Ye"], M3=c["S3"]["M"], taux3=c["S3"]["taux"], taux4=c["S4"]["taux"],
                 taux_max=c["D3_taux_max"], N=c["N_ELS"], C90=c["C90_arrachement"], A21=c["A21_haut_moins_Y"])
    except ZeroDivisionError:
        d = dict(Y=Yv, erreur="division par zéro")
    sens.append(d)
    if "erreur" in d:
        print(f"  Y = {Yv:5.0f} : division par zéro")
    else:
        print(f"  Y = {Yv:5.0f} : Ye = {d['Ye']:6.1f}  M3 = {d['M3']:+.3f}  taux S3 = {d['taux3'] * 100:+7.2f} %  "
              f"S4 = {d['taux4'] * 100:6.2f} %  max = {d['taux_max'] * 100:6.2f} %  N/cheville = {d['N']:7.2f} kN  "
              f"Haut − Y = {d['A21']:+.0f}  C90 = {d['C90']:+.1f}")
res["sensibilite_Y"] = sens
print("\nCharge au-delà de l'aile, ou dans le congé :")
dom = []
for Xv in (20.0, 28.0, 33.0, 75.0, 120.0, 140.0, 200.0):
    c = cas(X=Xv)
    d = dict(X=Xv, Xe=c["C19_Xe"], larg_moins_X=c["A20_larg_moins_X"], M1=c["S1"]["M"], taux1=c["S1"]["taux"],
             taux2=c["S2"]["taux"], taux_max=c["D3_taux_max"], N=c["N_ELS"])
    dom.append(d)
    print(f"  X = {Xv:5.0f} : Xe = {d['Xe']:+5.0f}  larg − X = {d['larg_moins_X']:+5.0f}  M1 = {d['M1']:+.3f}  "
          f"taux S1 = {d['taux1'] * 100:+7.2f} %  S2 = {d['taux2'] * 100:6.2f} %  max = {d['taux_max'] * 100:6.2f} %  "
          f"N = {d['N']:.2f} kN")
res["domaine_X"] = dom

# rapport ELU/ELS et lecture de la combinaison
with open(SORTIE, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=float)
print("→", SORTIE)

# ---------------------------------------------------------------- 6. C88 et C90 : calculés, jamais comparés
d0 = p["d_anc"] + 2.0                       # HYPOTHÈSE : trou Ø d + 2 mm (jeu normal) — le classeur ne le donne pas
c90_trou = xl["V_ELS"] * 1000 / (2 * (p["haut"] - Y - d0 / 2) * e)
print(f"\nC88 (pression diamétrale) = {xl['C88_pression']:.2f} MPa et C90 (arrachement) = {xl['C90_arrachement']:.2f} MPa :"
      f" aucune cellule ne les compare à une limite. C90 avec le trou déduit (Ø{d0:.0f}) : {c90_trou:.2f} MPa "
      f"(×{c90_trou / xl['C90_arrachement']:.3f})")
res["C88_C90"] = dict(C88=xl["C88_pression"], C90=xl["C90_arrachement"], C90_trou=c90_trou, d0=d0)
with open(SORTIE, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1, default=float)

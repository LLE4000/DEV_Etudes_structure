# -*- coding: utf-8 -*-
"""S06 — ELS, ELU, feuille E10A BD, efforts à transmettre à Hilti PROFIS.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s06_elu.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openpyxl  # noqa: E402
import s_commun as K  # noqa: E402
from s_longi import modele  # noqa: E402

S = K.Sortie("s06_elu")
C = K.CAS
e, R, X, Y = C["e"], C["R"], C["X"], C["Y"]
h = K.historique(C)
rho = C["R_ELU"] / R

S("# S06 — ELS, ELU et ancrages")
S("")
S("## 1. Le passage ELS → ELU du classeur (C85 = (C71/C70)·C82, C86 = (C71/C70)·C83)")
S(f"C71/C70 = {C['R_ELU'] / 1e3:.1f}/{R / 1e3:.2f} = {rho:.5f}")
for g in (1.0, 0.75, 0.5, 0.25, 0.0):
    S(f"   part permanente G/(G+Q) = {g:.2f} → (1,35·G + 1,5·Q)/(G + Q) = {1.35 * g + 1.5 * (1 - g):.4f}")
S("Un rapport supérieur à 1,50 ne peut pas venir de 1,35·G + 1,5·Q appliqués aux mêmes charges que l'ELS caractéristique, "
  "AVEC K_FI = 1,0 (classe de conséquences CC2).")
_G12, _Q12 = 21.5353, 9.1301        # volée 12 (section 2 ci-dessous)
S(f"avec K_FI = 1,1 (CC3, EN 1990 annexe B, tableau B3, si l'annexe nationale l'applique) : rapport maximal {1.1 * 1.5:.3f} ; "
  f"volée 12 : {1.1 * (1.35 * _G12 + 1.5 * _Q12) / (_G12 + _Q12):.3f} (≠ {rho:.3f}) — la conclusion tient pour cet escalier")
for nom, psi in (("fréquente (ψ1 = 0,5)", 0.5), ("quasi permanente (ψ2 = 0,3)", 0.3)):
    # (1,35 G + 1,5 Q)/(G + ψ Q) = ρ  →  G/Q = (1,5 − ρ ψ)/(ρ − 1,35)
    gq = (1.5 - rho * psi) / (rho - 1.35)
    S(f"   si 42,76 kN était une combinaison {nom} : il faudrait G/Q = {gq:.3f}")

S("")
S("## 2. Feuille « E10A BD » (lue, elle n'alimente aucune formule de P8)")
wb = openpyxl.load_workbook(K.XLSX, data_only=True)
ws = wb["E10A BD"]
S(f"D3 = {ws['D3'].value} ; G3 = {ws['G3'].value} (diviseur) ; F2 = {ws['F2'].value} (multiplie la surface)")
S("| volée | poids C (kN) | charge F (kN) | G = D (kN/m) | Q = G (kN/m) | (C + F)/2 (kN) | D + G | 1,35·D + 1,5·G | rapport |")
vol = {}
for row in range(4, 12):
    v = ws[f"A{row}"].value
    Cc, F, D, G = (ws[f"{c}{row}"].value for c in "CFDG")
    vol[v] = dict(C=Cc, F=F, D=D, G=G)
    S(f"| {v} | {Cc:.3f} | {F:.2f} | {D:.3f} | {G:.3f} | {(Cc + F) / 2:.3f} | {D + G:.3f} | {1.35 * D + 1.5 * G:.3f} | "
      f"{(1.35 * D + 1.5 * G) / (D + G):.4f} |")
cands = []
for v, q in vol.items():
    for nom, val in (("(C + F)/2", (q["C"] + q["F"]) / 2), ("D + G", q["D"] + q["G"]), ("1,35·D + 1,5·G", 1.35 * q["D"] + 1.5 * q["G"]),
                     ("(1,35·C + 1,5·F)/2", (1.35 * q["C"] + 1.5 * q["F"]) / 2)):
        for cible, nc in ((42.76, "C12 = 42,76"), (68.9, "C71 = 68,9")):
            cands.append((abs(val / cible - 1), v, nom, val, nc))
cands.sort()
for d, v, nom, val, nc in cands[:4]:
    S(f"   proche de {nc} : volée {v}, {nom} = {val:.4f} ({100 * (val / float(nc.split('= ')[1].replace(',', '.')) - 1):+.3f} %)")
s1012 = (vol[10]["C"] + vol[10]["F"] + vol[12]["C"] + vol[12]["F"]) / 2
S(f"   (C + F)/2 des volées 10 et 12 : {s1012:.3f} kN ({100 * (s1012 / 68.9 - 1):+.2f} % sur 68,9) — somme non pondérée")
S(f"   42,76 kN sur 0,30 m = {42.76 / 0.30:.1f} kN/m, contre {max(q['D'] + q['G'] for q in vol.values()):.2f} kN/m au plus dans la feuille")
S.v("volees", vol)
q12 = vol[12]
S.v("v12", dict(ELU=1.35 * q12["D"] + 1.5 * q12["G"], ratio=(1.35 * q12["D"] + 1.5 * q12["G"]) / (q12["D"] + q12["G"])))

S("")
S("## 3. Efforts par ancrage pour Hilti — cas de contrôle (300 mm, 3 ancrages)")
fb0 = K.corps_libre_A(C, 0.0, e / 2)
g12 = q12["D"] / (q12["D"] + q12["G"])
G_, Q_ = R * g12, R * (1 - g12)
R_elu12 = 1.35 * G_ + 1.5 * Q_
S("| méthode | N_ELS | V_ELS | N_ELU | V_ELU (kN par ancrage) | ELU |")
S(f"| classeur (bras Y − 1 cm) | {h['N_els'] / 1e3:.3f} | {h['V_els'] / 1e3:.3f} | {h['N_elu'] / 1e3:.3f} | {h['V_elu'] / 1e3:.3f} | ×{rho:.4f} |")
S(f"| équilibre P0 (pivot à l'arête, V sur l'axe) | {fb0['H'] / 3e3:.3f} | {R / 3e3:.3f} | {rho * fb0['H'] / 3e3:.3f} | {rho * R / 3e3:.3f} | ×{rho:.4f} (même R_ELU) |")
S(f"| P0, ELU = 1,35·G + 1,5·Q avec le partage G/Q de la volée 12 (HYPOTHÈSE : G = {G_ / 1e3:.2f}, Q = {Q_ / 1e3:.2f} kN) | "
  f"{fb0['H'] / 3e3:.3f} | {R / 3e3:.3f} | {fb0['H'] * R_elu12 / R / 3e3:.3f} | {R_elu12 / 3e3:.3f} | R_ELU = {R_elu12 / 1e3:.2f} kN (×{R_elu12 / R:.4f}) |")
S(f"actions sur la platine (aile verticale de 300 mm) si PROFIS répartit lui-même : V = {C['R_ELU'] / 1e3:.1f} kN vers le bas, "
  f"M = V·X = {C['R_ELU'] * X / 1e6:.3f} kN·m (bras compté depuis la face du support), N = 0")
S.v("hilti", dict(P0_N=fb0["H"] / 3, R_elu12=R_elu12))

S("")
S("## 4. Cornière longue : l'ELU par ancrage se calcule sur la combinaison chargée")
S("HYPOTHÈSE d'illustration : G = 30 kN répartis sur 2,00 m ; Q = 30 kN sur 300 mm centrés à 0,40 m ; ancrages raides.")
kw = dict(kN=100e3, kV=30e3, k_pl=None)
from s_longi import k_pl as KPL  # noqa: E402
kw["k_pl"] = KPL
chG = ((1000.0, 2000.0, 30e3),)
chQ = ((400.0, 300.0, 30e3),)
mG = modele(charges=chG, **kw)
mQ = modele(charges=chQ, **kw)
mS = modele(charges=chG + chQ, **kw)
mU = modele(charges=((1000.0, 2000.0, 1.35 * 30e3), (400.0, 300.0, 1.5 * 30e3)), **kw)
coef = (1.35 * 30 + 1.5 * 30) / 60
S("contrôle de superposition N(G+Q) − N(G) − N(Q) = " + ", ".join(f"{(a - b - c) / 1e3:+.3f}" for a, b, c in zip(mS["N"], mG["N"], mQ["N"])) + " kN")
S("| ancrage | N_ELS (G+Q) | N_ELU (1,35G + 1,5Q) | coef. global 1,425 × N_ELS | rapport du classeur × N_ELS | V_ELS | V_ELU |")
for i, z in enumerate((200, 1000, 1800)):
    S(f"| {i + 1} (z = {z}) | {mS['N'][i] / 1e3:.2f} | {mU['N'][i] / 1e3:.2f} | {coef * mS['N'][i] / 1e3:.2f} ({100 * (coef * mS['N'][i] / mU['N'][i] - 1):+.1f} %) | "
      f"{rho * mS['N'][i] / 1e3:.2f} | {mS['V'][i] / 1e3:.2f} | {mU['V'][i] / 1e3:.2f} |")
H_elu = (1.35 * 30e3 + 1.5 * 30e3) * (X - e / 2) / Y
S(f"platine rigide unique de 2 m (tractions égales) : {H_elu / 3e3:.2f} kN par ancrage, contre {mU['N'].max() / 1e3:.2f} kN "
  f"pour le plus chargé ({100 * (mU['N'].max() / (H_elu / 3) - 1):+.0f} % ; la platine rigide le sous-estime de {100 * (1 - H_elu / 3 / mU['N'].max()):.0f} %)")
S.v("long_elu", dict(N=mU["N"].tolist(), V=mU["V"].tolist(), rigide=H_elu / 3))
S.ecrire()

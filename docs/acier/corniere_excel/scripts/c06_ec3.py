# -*- coding: utf-8 -*-
"""c06 — La méthode aux contraintes admissibles face à l'EN 1993-1-1 (ELU, γ_M0 = 1,0).

σ_adm = f_y / 1,5 sur des sollicitations ELS non majorées  ⇔  1,5 · σ_ELS ≤ f_y : c'est une vérification
élastique avec UN coefficient global 1,5 sur toutes les actions. L'EC3 (élastique) demande γ_F · σ_ELS ≤ f_y / γ_M0,
avec γ_F = γ_G = 1,35 ou γ_Q = 1,50 selon la nature de chaque action (EN 1990, 6.10), et autorise le PLASTIQUE pour
une section rectangulaire pleine (classe 1) : M_pl = b·t²·f_y / 4 = 1,5 · M_el.

Calculé pour le cas de contrôle, sections 1 et 3, avec plusieurs rapports ELU/ELS γ_eq :
1,35 (tout permanent), 1,50 (tout variable), ceux des volées de la feuille E10A BD (G et Q lus dans le classeur),
et celui que porte le classeur lui-même pour les ancrages (C71/C70 = 68,9/42,76).
"""
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import commun as K  # noqa: E402
import openpyxl  # noqa: E402

RACINE = os.path.dirname(ICI)
p = K.CAS
fy, e, r, R = p["fy"], p["e"], p["r"], p["R"]
b = p["b_cm"] * 10
xl = K.classeur(**p)
GM0 = 1.0

# rapports G/Q des volées (feuille E10A BD : D = poids propre kN/m, G = exploitation kN/m) — LUS dans le classeur
wb = openpyxl.load_workbook(os.path.join(RACINE, "source", "realex_cornieres.xlsx"), data_only=True)
F = wb["E10A BD"]
volees = []
for row in range(4, 12):
    g, qq, nom = F[f"D{row}"].value, F[f"G{row}"].value, F[f"A{row}"].value
    if isinstance(g, (int, float)) and isinstance(qq, (int, float)):
        volees.append((nom, g, qq, (1.35 * g + 1.5 * qq) / (g + qq)))
gam_vol = [v[3] for v in volees]
print("Volées E10A BD : γ_eq = (1,35 G + 1,5 Q)/(G + Q) de", f"{min(gam_vol):.4f} à {max(gam_vol):.4f}",
      "(part variable de", f"{min(v[2] / (v[1] + v[2]) for v in volees) * 100:.1f} à "
                           f"{max(v[2] / (v[1] + v[2]) for v in volees) * 100:.1f} %)")

gammas = [("tout permanent, 6.10", 1.35), ("volées E10A BD (min)", min(gam_vol)), ("volées E10A BD (max)", max(gam_vol)),
          ("tout variable, 6.10", 1.50), ("classeur, C71/C70 (ancrages)", p["R_elu"] / R)]

# section 3 : deux lectures de la statique — celle du classeur (M3, τ = R/A) et la statique cohérente
# (x_V = e/2, pivot e/2 : N = R traction, τ = T/A) de c05.
A = b * e
Wel = b * e ** 2 / 6
Wpl = b * e ** 2 / 4
Npl = A * fy / GM0
Vpl = A * fy / math.sqrt(3) / GM0
T_coh = R * (p["X"] - e / 2) / (p["Y"] - e / 2)
M3_coh = T_coh * (p["Y"] - e - r) / 1000          # kN·m
sections = {
    "S1": dict(M=xl["S1"]["M"], N=0.0, V=R),
    "S3 (statique du classeur)": dict(M=xl["S3"]["M"], N=0.0, V=R),
    "S3 (statique cohérente)": dict(M=M3_coh, N=R, V=T_coh),
}
res = {"volees": volees, "lignes": []}
print(f"\nMéthode du classeur : taux = σ_eq/(f_y/1,5) ; S1 {xl['S1']['taux'] * 100:.2f} %, S3 {xl['S3']['taux'] * 100:.2f} %")
print(f"{'section':28s} {'γ_eq':>7s} {'adm. ELS':>9s} {'EC3 élast.':>11s} {'EC3 plast.':>11s}  (γ_M0 = 1,0)")
for sn, s in sections.items():
    sig = s["M"] * 1e6 / Wel + s["N"] * 1e3 / A
    tau = s["V"] * 1e3 / A
    eq_els = math.sqrt(sig ** 2 + 3 * tau ** 2)
    adm = eq_els / (fy / p["gM"])
    for nom, g in gammas:
        el = g * eq_els / (fy / GM0)                                     # EN 1993-1-1 6.2.1(5)
        MEd, NEd, VEd = g * s["M"] * 1e6, g * s["N"] * 1e3, g * s["V"] * 1e3
        rho = 0.0 if VEd <= 0.5 * Vpl else (2 * VEd / Vpl - 1) ** 2     # 6.2.8 (sans effet ici)
        fyr = (1 - rho) * fy
        MN = Wpl * fyr / GM0 * (1 - (NEd / (A * fyr / GM0)) ** 2)       # rectangle : M_N,Rd = M_pl,Rd (1 − n²)
        pl = MEd / MN
        res["lignes"].append(dict(section=sn, gamma_nom=nom, gamma=g, adm=adm, ec3_el=el, ec3_pl=pl,
                                  V_sur_Vpl=VEd / Vpl, n=NEd / Npl))
        print(f"{sn:28s} {g:7.4f} {adm * 100:8.2f} % {el * 100:10.2f} % {pl * 100:10.2f} %   [{nom}]")
# équivalence exacte
print(f"\nIdentité : taux admissible = taux EC3 élastique pour γ_eq = {p['gM']:.2f} "
      f"(S3 : {xl['S3']['taux'] * 100:.4f} % = {1.5 * xl['S3']['sigma_eq'] / fy * 100:.4f} %)")
ratio_pl = p["gM"] * Wpl / Wel
print(f"Plastique : le classeur est plus sévère que l'EC3 plastique d'un facteur 1,5 × 1,5 / γ_eq = "
      f"{ratio_pl / 1.5:.3f} (γ = 1,5) à {ratio_pl / 1.35:.3f} (γ = 1,35)")
res["facteurs_plastique"] = [ratio_pl / 1.5, ratio_pl / 1.35]
json.dump(res, open(os.path.join(RACINE, "sorties", "c06_ec3.json"), "w"), ensure_ascii=False, indent=1, default=str)

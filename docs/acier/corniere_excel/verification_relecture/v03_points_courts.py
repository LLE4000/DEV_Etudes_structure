# -*- coding: utf-8 -*-
"""V03 — Points 4, 5, 8, 9, 10, 11, 12, 14 de la relecture (calculs courts, lecture du classeur et du catalogue).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/v03_points_courts.py
"""
import json
import math
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
RAC = os.path.dirname(ICI)
XLSX = os.path.join(RAC, "source", "realex_cornieres.xlsx")
R, X, Y, e, r, L, fy = 42760.0, 75.0, 85.0, 15.0, 13.0, 300.0, 355.0
SADM = fy / 1.5
TADM = SADM / math.sqrt(3)

# ------------------------------------------------------------------ point 4 : moteur du site, valeurs par défaut
print("== point 4 : moteur actuel du site (lecture seule), cas de contrôle")
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/user/integ")
from acier.corniere import entree as EN, moteur as M  # noqa: E402

print("   défauts de l'écran (CHAMPS) : " + ", ".join(f"{k} = {EN.CHAMPS[k][0]:g}" for k in ("gamma_ELU", "sig_lim_ELS", "tau_lim_ELS", "sig_lim_ELU", "s")))
base = dict(M.nouvelle_corniere(), standard=True, profil="L 120x120x15", charge="ponctuelle", V=R / 1e3, L=L / 1000, e=X / 10)
g = M.verifier(base)["grandeurs"]
print(f"   tout par défaut (γ_ELU 1,35 ; σ_lim 155 ; τ_lim 0,8 ; σ_eq,lim,ELU 186) : σ_éq,ELS = {g['sigma_eq_ELS']:.3f} ; "
      f"U_σ = {100 * g['U_sigma']:.2f} % ; U_τ = {100 * g['U_tau']:.1f} % ; U_ELU = {100 * g['U_ELU']:.1f} % ; U_long,ELU = {100 * g['U_long_ELU']:.1f} %")
g2 = M.verifier(dict(base, sig_lim_ELS=SADM, tau_lim_ELS=TADM, sig_lim_ELU=SADM))["grandeurs"]
print(f"   σ_lim,ELS = f_y/1,5 = {SADM:.2f} saisi : U_σ = {100 * g2['U_sigma']:.2f} %")
g3 = M.verifier(dict(base, gamma_ELU=1.0))["grandeurs"]
print(f"   (γ_ELU = 1,0 et limites par défaut : U_ELU = {100 * g3['U_ELU']:.1f} %)")

# ------------------------------------------------------------------ point 5 : hauteur de l'ancrage, configuration A
print("\n== point 5 : hauteur de l'ancrage Y (configuration A, X = 75)")
W, A = L * e * e / 6, L * e
for Yv in (65.0, 85.0, 105.0):
    M3c = R * X * (Yv - e - r) / (Yv - e / 2)
    t3c = math.sqrt((M3c / W) ** 2 + 3 * (R / A) ** 2) / SADM
    H0 = R * (X - e / 2) / Yv
    s0 = H0 * (Yv - e - r) / W + R / A
    t30 = math.sqrt(s0 ** 2 + 3 * (H0 / A) ** 2) / SADM
    print(f"   Y = {Yv:.0f} mm : taux 3 classeur {100 * t3c:.2f} % ; taux 3 P0 {100 * t30:.2f} % ; traction P0 H = {H0 / 1e3:.2f} kN ; "
          f"traction du classeur (3 × C82) = {R * X / (Yv - 10) / 1e3:.2f} kN")

# ------------------------------------------------------------------ point 8 : conséquence locale au droit de l'ancrage
print("\n== point 8 : aile verticale au droit de l'ancrage le plus chargé (analogie du tronçon en T)")
E = 210000.0
m = (Y - e) - 0.8 * r
leff = min(2 * math.pi * m, 4 * m + 1.25 * (120.0 - Y))
print(f"   m = (Y − e) − 0,8·r = {m:.1f} mm ; l_eff = min(2π·m ; 4·m + 1,25·(h − Y)) = min({2 * math.pi * m:.1f} ; {4 * m + 1.25 * (120 - Y):.1f}) = {leff:.1f} mm")
print(f"   k_pl = 3·E·(l_eff·e³/12)/Y³ = {3 * E * (leff * e ** 3 / 12) / Y ** 3 / 1e3:.1f} kN/mm (console de longueur Y encastrée au talon)")
print(f"   variante : console de longueur Y − (e + r) = {Y - e - r:.0f} mm encastrée en section 3 : {3 * E * (leff * e ** 3 / 12) / (Y - e - r) ** 3 / 1e3:.1f} kN/mm")
s05 = json.load(open(os.path.join(RAC, "synthese", "sorties", "s05_longue.json"), encoding="utf-8"))["cas"]
for nom in ("ancrages rigides", "ancrages raides (100 / 30 kN/mm) + aile", "ancrages souples (10 / 5 kN/mm) + aile"):
    N1, V1 = max(s05[nom]["N"]), max(s05[nom]["V"])
    sY = 6 * N1 * (Y - e - r) / (leff * e * e) + V1 / (leff * e)
    sm = 6 * N1 * m / (leff * e * e) + V1 / (leff * e)
    print(f"   {nom} : N1 = {N1 / 1e3:.2f} kN, V1 = {V1 / 1e3:.2f} kN → bras Ye = 57 : {sY:.1f} N/mm² ({100 * sY / SADM:.1f} %) ; "
          f"bras m = {m:.1f} : {sm:.1f} N/mm² ({100 * sm / SADM:.1f} %)")

# ------------------------------------------------------------------ point 9 : seuils des sections « e + r »
print("\n== point 9 : seuils sous lesquels la section « e + r » dépasse la section « e »")


def bis(f, a, b):
    fa = f(a)
    for _ in range(200):
        c = (a + b) / 2
        if (f(c) > 0) == (fa > 0):
            a, fa = c, f(c)
        else:
            b = c
    return (a + b) / 2


a = e + r
W1, W2, A1, A2 = L * e * e / 6, L * a * a / 6, L * e, L * a
fl = lambda Xp: R * (Xp - e) / W2 - R * (Xp - e - r) / W1
eq = lambda Xp: math.hypot(R * (Xp - e) / W2, math.sqrt(3) * R / A2) - math.hypot(R * (Xp - e - r) / W1, math.sqrt(3) * R / A1)
print(f"   aile horizontale : flexion seule X' < {bis(fl, a, 200):.3f} mm ; σ_éq du classeur X' < {bis(eq, a, 200):.3f} mm")
for Xv in (75.0, 40.0):
    Mm = R * Xv
    fv = lambda Yp: Mm * (Yp - e) / (Yp - e / 2) / W2 - Mm * (Yp - e - r) / (Yp - e / 2) / W1
    ev = lambda Yp: (math.hypot(Mm * (Yp - e) / (Yp - e / 2) / W2, math.sqrt(3) * R / A2)
                     - math.hypot(Mm * (Yp - e - r) / (Yp - e / 2) / W1, math.sqrt(3) * R / A1))
    print(f"   aile verticale (X = {Xv:g}) : flexion seule Y' < {bis(fv, a, 200):.3f} mm ; σ_éq du classeur Y' < {bis(ev, a, 200):.3f} mm")

# ------------------------------------------------------------------ point 10 : K_FI
print("\n== point 10 : rapport ELU/ELS avec K_FI (EN 1990, annexe B, tableau B3)")
G12, Q12 = 21.5353, 9.1301
for kfi in (1.0, 1.1):
    rv = kfi * (1.35 * G12 + 1.5 * Q12) / (G12 + Q12)
    print(f"   K_FI = {kfi:.1f} : rapport maximal (tout variable) {kfi * 1.5:.3f} ; volée 12 : {rv:.3f} ; C71/C70 = {68.9 / 42.76:.3f}")

# ------------------------------------------------------------------ points 11 et 14 : qui lit quoi, A12/C12
print("\n== points 11 et 14 : lecture du classeur (openpyxl)")
import openpyxl  # noqa: E402

wb = openpyxl.load_workbook(XLSX, data_only=False)
ws = wb["P8 "]
form = {c.coordinate: c.value for row in ws.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("=")}
for cible in ("C4", "C5", "A20", "A21", "C16"):
    pat = re.compile(r"(?<![A-Z$])\$?" + cible[0] + r"\$?" + cible[1:] + r"(?!\d)")
    print(f"   {cible} lue par : {sorted(k for k, v in form.items() if pat.search(v))}")
print(f"   C90 = {form['C90']}")
print(f"   A12 = {ws['A12'].value!r} ; C12 = {ws['C12'].value!r} ; B12 = {ws['B12'].value!r}")

# ------------------------------------------------------------------ point 12 : catalogue du site
print("\n== point 12 : catalogue du site")
d = json.load(open("/home/user/integ/acier/donnees/profiles.json", encoding="utf-8"))
cor = [p for p in d["profils"] if p.get("famille") in ("L", "LI")]
print(f"   cornières : {len(cor)} (L {sum(p['famille'] == 'L' for p in cor)}, LI {sum(p['famille'] == 'LI' for p in cor)}) ; "
      f"courant = true : {sum(bool(p.get('courant')) for p in cor)} ; false : {sum(p.get('courant') is False for p in cor)}")
p300 = [p for p in cor if p["nom"] == "L 300x300x35"][0]
print(f"   L 300x300x35 : courant = {p300.get('courant')} ; r1 = {p300.get('r1')}")
for cle in ("source", "provenance", "norme"):
    if cle in p300:
        print(f"   L 300x300x35 {cle} = {p300[cle]!r}")
txt = open("/home/user/integ/docs/acier/PROFILES.md", encoding="utf-8").read()
for lg in txt.splitlines():
    if "compl" in lg.lower() and ("cornière" in lg.lower() or "10056" in lg):
        print("   PROFILES.md : " + lg.strip()[:260])

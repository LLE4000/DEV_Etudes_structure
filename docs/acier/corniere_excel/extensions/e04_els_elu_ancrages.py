# -*- coding: utf-8 -*-
"""E04 — ELS / ELU : efforts dans la cornière et efforts par ancrage pour Hilti PROFIS.

1. Le passage ELS → ELU du classeur (C85 = C71/C70 · C82) face aux combinaisons de l'EN 1990.
2. La feuille E10A BD : G et Q par appui et par mètre, combinaisons.
3. Efforts par ancrage, cas de contrôle (une coupe) : méthode du classeur, corps libre, ELU combiné.
4. Cornière longue : G réparti sur 2 m et Q localisé — efforts par ancrage par cas, puis combinés,
   face à un coefficient global appliqué à l'ELS.

Lancer : /usr/bin/python3.12 -I e04_els_elu_ancrages.py   (charge e03 en silence, ~20 s)
"""
import contextlib
import io
import math
import os
import re
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import ext_commun as K  # noqa: E402

out = []
P = out.append
c = K.CAS
e, r, X, Y, R, L = c["e"], c["r"], c["X"], c["Y"], c["R"], c["L"]
cache = K.releve_excel()

P("# E04 — ELS, ELU et efforts par ancrage")
P("")
P("## 1. Le passage ELS → ELU du classeur")
rho = cache["C71"] / cache["C70"]
P(f"C71/C70 = {cache['C71']} / {cache['C70']} = {rho:.5f} ; N_ELU = ρ·N_ELS = {cache['C85']:.4f} kN (C85), "
  f"V_ELU = {cache['C86']:.4f} kN (C86) ; valeurs tapées à côté : G85 = 24,68 kN, G86 = 22,97 kN")
P("Avec γG = 1,35 et γQ = 1,50 appliqués aux MÊMES charges caractéristiques, le rapport ELU/ELS caractéristique est "
  "compris entre 1,35 (tout permanent) et 1,50 (tout variable) :")
for g_frac in (1.0, 0.75, 0.5, 0.25, 0.0):
    rr = 1.35 * g_frac + 1.5 * (1 - g_frac)
    P(f"   G/(G+Q) = {g_frac:.2f} → ELU/ELS = {rr:.4f}")
P(f"Le rapport du classeur, {rho:.4f}, dépasse 1,50 : 68,9 kN ne peut pas venir de 42,76 kN par les coefficients de "
  f"l'EN 1990 (6.10) sur les mêmes charges. Les deux réactions viennent de calculs ou de combinaisons différents.")
for nom, psi in (("fréquente (ψ1 = 0,5)", 0.5), ("quasi permanente (ψ2 = 0,3)", 0.3)):
    # rapport (1,35 g + 1,5 q)/(g + ψ q) = ρ  →  g/q
    gq = (1.5 - rho * psi) / (rho - 1.35)
    P(f"   si 42,76 kN était une combinaison {nom} : il faudrait G/Q = {gq:.3f} pour retrouver 68,9 kN")

# ---------------------------------------------------------------- E10A BD
P("")
P("## 2. Feuille « E10A BD » (lue dans le relevé) : charges par appui et par mètre")
cel = {}
feuille = None
for l_ in open(K.RELEVE, encoding="utf-8"):
    m = re.match(r"===== FEUILLE (.+?) \(", l_)
    if m:
        feuille = m.group(1).strip()
        continue
    if feuille != "E10A BD":
        continue
    m = re.match(r"([A-I])(\d+)\t(?:FORMULE .*\t=> |VALEUR )(.+)$", l_.rstrip("\n"))
    if m:
        try:
            cel[m.group(1) + m.group(2)] = float(m.group(3))
        except ValueError:
            pass
P("| volée | G (D, kN/m) | Q (G, kN/m) | G+Q | G+0,5Q | 1,35G+1,5Q | ELU/ELS car. |")
P("|---|---|---|---|---|---|---|")
proches = []
for row in range(4, 12):
    if f"D{row}" not in cel:
        continue
    G_, Q_ = cel[f"D{row}"], cel[f"G{row}"]
    elu = 1.35 * G_ + 1.5 * Q_
    P(f"| {cel[f'A{row}']:.0f} | {G_:.3f} | {Q_:.3f} | {G_ + Q_:.3f} | {G_ + 0.5 * Q_:.3f} | {elu:.3f} | "
      f"{elu / (G_ + Q_):.4f} |")
    for nom, v in (("G+Q", G_ + Q_), ("1,35G+1,5Q", elu), ("G+0,5Q", G_ + 0.5 * Q_)):
        for cible in (42.76, 68.9):
            if abs(v / cible - 1) < 0.005:
                proches.append(f"volée {cel[f'A{row}']:.0f} : {nom} = {v:.4f} kN/m ≈ {cible} ({(v / cible - 1) * 100:+.3f} %)")
P("Coïncidences à 0,5 % près avec 42,76 ou 68,9 : " + ("; ".join(proches) if proches else "aucune"))
P(f"D3 = G3 = {cel.get('D3')} (diviseur commun : largeur probable de la volée, m) ; F2 = {cel.get('F2')} kN/m² "
  f"(charge d'exploitation probable).")

# ---------------------------------------------------------------- une coupe
P("")
P("## 3. Efforts par ancrage — cas de contrôle (coupe de 300 mm, 3 ancrages)")
n = 3
Hxl = R * X / (Y - 10.0)
H0 = R * (X - e / 2) / Y
P("| méthode | N_ELS | V_ELS | N_ELU | V_ELU (kN par ancrage) | hypothèse ELU |")
P("|---|---|---|---|---|---|")
P(f"| classeur (bras Y − 1 cm, R·X) | {Hxl / n / 1e3:.3f} | {R / n / 1e3:.3f} | {Hxl / n / 1e3 * rho:.3f} | "
  f"{R / n / 1e3 * rho:.3f} | × 68,9/42,76 |")
P(f"| corps libre, pivot à l'arête (maillage), V sur l'axe | {H0 / n / 1e3:.3f} | {R / n / 1e3:.3f} | "
  f"{H0 / n / 1e3 * rho:.3f} | {R / n / 1e3 * rho:.3f} | × 68,9/42,76 (même réaction ELU) |")
# ELU par combinaison avec une répartition G/Q supposée (celle de la volée 12)
G12, Q12 = cel["D11"], cel["G11"]
Gs = R * G12 / (G12 + Q12)
Qs = R * Q12 / (G12 + Q12)
Relu = 1.35 * Gs + 1.5 * Qs
P(f"| idem, ELU = 1,35·G + 1,5·Q avec G/Q de la volée 12 (G = {Gs / 1e3:.2f}, Q = {Qs / 1e3:.2f} kN, HYPOTHÈSE) | "
  f"{H0 / n / 1e3:.3f} | {R / n / 1e3:.3f} | {R and H0 / R * Relu / n / 1e3:.3f} | {Relu / n / 1e3:.3f} | "
  f"R_ELU = {Relu / 1e3:.2f} kN (×{Relu / R:.4f}) |")
z_hilti = cache["C71"] * X / (3 * 24.68)
P(f"Bras de levier qui redonne G85 = 24,68 kN avec R_ELU = 68,9 kN, X = 75 mm et 3 ancrages : z = {z_hilti:.2f} mm, "
  f"soit une compression à {Y - z_hilti:.2f} mm au-dessus du dessous de la cornière (≈ e = 15 mm : bord inférieur d'une "
  f"platine qui serait l'aile verticale AU-DESSUS de l'aile horizontale — interprétation).")
P(f"Actions à donner à PROFIS pour la platine de 300 mm (aile verticale) à l'ELU, si on le laisse répartir : "
  f"V = {cache['C71']:.1f} kN vers le bas, M = V·X = {cache['C71'] * X / 1e3:.3f} kN·m autour de l'axe horizontal du mur "
  f"(bras compté depuis la face du béton), N = 0.")

# ---------------------------------------------------------------- cornière longue
P("")
P("## 4. Cornière longue : un coefficient global sur l'ELS ne donne pas l'ELU par ancrage")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    E3 = runpy.run_path(os.path.join(K.ICI, "e03_longitudinal.py"))
modele = E3["modele"]
kpl = E3["k_pl_cons"]
opts = dict(kN=100e3, kV=30e3, k_pl=kpl)
G_tot = 30.0e3            # HYPOTHÈSE : permanentes réparties sur les 2 m (15 kN/m)
Q_tot = 30.0e3            # HYPOTHÈSE : exploitation localisée, 300 mm centrés à 0,40 m
mG = modele(charges=((1000.0, 2000.0, G_tot),), **opts)
mQ = modele(charges=((400.0, 300.0, Q_tot),), **opts)
P(f"Hypothèse d'illustration : G = {G_tot / 1e3:.0f} kN répartis sur 2,00 m ; Q = {Q_tot / 1e3:.0f} kN sur 300 mm centrés à "
  f"0,40 m ; ancrages raides (100 et 30 kN/mm).")
NG, VG, NQ, VQ = mG["N"], mG["V"], mQ["N"], mQ["V"]
mGQ = modele(charges=((1000.0, 2000.0, G_tot), (400.0, 300.0, Q_tot)), **opts)
P(f"contrôle de superposition : N(G+Q) − N(G) − N(Q) = "
  + ", ".join(f"{(a - b - c_) / 1e3:+.4f}" for a, b, c_ in zip(mGQ["N"], NG, NQ)) + " kN (le contact unilatéral "
  "n'est pas linéaire : la combinaison se calcule chargée, pas par superposition)")
mELU = modele(charges=((1000.0, 2000.0, 1.35 * G_tot), (400.0, 300.0, 1.5 * Q_tot)), **opts)
P("ELU calculé directement sur la combinaison 1,35·G + 1,5·Q : N = " + ", ".join(f"{x / 1e3:.2f}" for x in mELU["N"])
  + " kN ; V = " + ", ".join(f"{x / 1e3:.2f}" for x in mELU["V"]) + " kN")
mELS = modele(charges=((1000.0, 2000.0, G_tot), (400.0, 300.0, Q_tot)), **opts)
els = mELS["N"]
elu = mELU["N"]
vels = mELS["V"]
velu = mELU["V"]
glob = (1.35 * G_tot + 1.5 * Q_tot) / (G_tot + Q_tot)
P(f"coefficient global équivalent sur la réaction totale : (1,35·G + 1,5·Q)/(G + Q) = {glob:.4f}")
P("| ancrage | N_G seul | N_Q seul | N_ELS (G+Q) | N_ELU (1,35G+1,5Q) | coef. global × N_ELS | écart | V_ELS | V_ELU | V global | écart |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for i in range(3):
    ng = glob * els[i]
    vg = glob * vels[i]
    P(f"| {i + 1} (z = {mG['ancrages'][i]:.0f}) | {NG[i] / 1e3:.2f} | {NQ[i] / 1e3:.2f} | {els[i] / 1e3:.2f} | "
      f"{elu[i] / 1e3:.2f} | {ng / 1e3:.2f} | {(ng / elu[i] - 1) * 100 if elu[i] else 0:+.1f} % | {vels[i] / 1e3:.2f} | "
      f"{velu[i] / 1e3:.2f} | {vg / 1e3:.2f} | {(vg / velu[i] - 1) * 100 if velu[i] else 0:+.1f} % |")
rho_g = rho
P(f"avec le rapport du classeur (×{rho_g:.4f}) : N_ELU = " + ", ".join(f"{rho_g * x / 1e3:.2f}" for x in els) + " kN")
P(f"platine RIGIDE de 2 m (répartition « R/n » des tractions, ce que donnerait une platine unique dans un logiciel qui la "
  f"suppose rigide) : N_ELU = {elu.sum() / 3 / 1e3:.2f} kN par ancrage, contre "
  f"{elu.max() / 1e3:.2f} kN pour l'ancrage le plus chargé du modèle longitudinal "
  f"({(elu.max() / (elu.sum() / 3) - 1) * 100:+.0f} %).")
K.ecrire(os.path.join(K.ICI, "e04_sortie.txt"), out)
print("\n".join(out))

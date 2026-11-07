# -*- coding: utf-8 -*-
"""V01 — Point 1 de la relecture : support souple. D'où vient le « +3,8 % » ?

A. Modèle de c03c (scripts/ef_coupe.py : rondelle Ø30 RIGIDE, cheville δ = 0,1 mm sous la traction du classeur,
   cisaillement en e/2) : on LIT directement le centre des forces de contact rondelle–aile (pas de statique déduite),
   sur mur rigide et à 30 N/mm³.
B. Modèle de s07 (extensions/ef_l.py, maillage fin du document : ancrage sous l'écrou ± 3 mm, 300 N/mm par mm,
   cisaillement au nœud de mi-épaisseur) : balayage du module de réaction du mur 1e5 → 30 N/mm³, puis « rondelle
   élastique » ± 15 mm à 30 N/mm³. Section 3 nominale = |M|/W + N/A (résultantes du maillage), face au classeur.
C. Poutre : pivot au point trouvé, cisaillement sur l'axe (u_V = e/2) → taux 3, face au 88,86 % du classeur.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/v01_support_souple.py
"""
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
RAC = os.path.dirname(ICI)
sys.path.insert(0, os.path.join(RAC, "scripts"))
sys.path.insert(1, os.path.join(RAC, "extensions"))
import numpy as np  # noqa: E402

R, X, Y, e, r, L = 42760.0, 75.0, 85.0, 15.0, 13.0, 300.0
a = e + r
q = R / L
SADM = 355.0 / 1.5
S3_CLASSEUR = R * X * (Y - e - r) / (Y - e / 2) / (L * e * e / 6)       # 209,662 N/mm²
print(f"σ3 du classeur (flexion seule) = {S3_CLASSEUR:.3f} N/mm²")

# ---------------------------------------------------------------- A. modèle c03c
import commun as KC  # noqa: E402
import ef_coupe as EF  # noqa: E402

p = KC.CAS
xl = KC.classeur(**p)
K0 = p["n_anc"] * (xl["N_ELS"] * 1000 / 0.1) / (p["b_cm"] * 10)
m = EF.Coupe(p["larg"], p["haut"], e, r, h=0.5)
print("\n# A. Modèle de c03c (rondelle Ø30 rigide, cheville δ = 0,1 mm)")
for kw in (None, 30.0):
    res = m.resoudre(q, X, Y, k_mur=kw, k_tige=K0)
    fr = res["rond_f"]
    yT = (fr * res["rond_y"]).sum() / fr.sum()
    T = res["T"] * L / 1e3
    stat = res["yc"] + R * (X - e / 2) / (res["T"] * L)
    actifs = res["rond_y"][np.abs(fr) > 1e-9 * np.abs(fr).max()]
    print(f"mur {'rigide' if kw is None else f'{kw:g} N/mm³':>10s} : T = {T:.2f} kN ; pivot y_c = {res['yc']:.2f} mm ; "
          f"centre des forces de contact rondelle–aile y_T = {yT:.2f} mm (lu) ; par la statique {stat:.2f} mm ; "
          f"nœuds de rondelle en contact de y = {actifs.min():.1f} à {actifs.max():.1f} mm (rondelle 70 à 100)")

# ---------------------------------------------------------------- B. modèle s07 (maillage fin)
import ef_l  # noqa: E402

GEO = dict(B=120.0, H=120.0, e=e, r=r, Y=Y)
MAIL = dict(n_t=10, n_c=16, n_leg=40, grad=4.0)


def cas(k_mur, b_rond=3.0):
    pb = ef_l.Probleme(GEO, "A", maillage=MAIL, solveur="bande")
    pb.resoudre(q, X, k_anc=300.0, k_mur=k_mur, b_rond=b_rond)
    rr = pb.reactions
    bas = rr["mur_y"] < Y
    yc = (rr["mur_f"][bas] * rr["mur_y"][bas]).sum() / rr["mur_f"][bas].sum()
    cp = pb.coupe((0.0, a), (e, a), n=60, normale="y")
    snom = abs(cp["M"]) / (e * e / 6) + cp["N"] / e
    return dict(H=rr["H"] * L, yH=rr["y_H"], yc=yc, M3=abs(cp["M"]) * L, N3=cp["N"] * L, snom=snom,
                tete=rr["mur_f"][~bas].sum() * L)


print("\n# B. Modèle de s07 (écrou ± 3 mm, 30 kN/mm par cheville, cisaillement à mi-épaisseur), maillage fin")
B = {}
for km in (1e5, 1e3, 300.0, 100.0, 30.0):
    o = cas(km)
    B[km] = o
    print(f"mur {km:>8g} N/mm³ : H = {o['H'] / 1e3:.3f} kN à y = {o['yH']:.2f} ; pivot y_c = {o['yc']:.2f} mm ; "
          f"M3 = {o['M3'] / 1e6:.4f} kN·m ; N3 = {o['N3'] / 1e3:.2f} kN ; σ3 nominale = {o['snom']:.2f} N/mm² "
          f"({100 * (o['snom'] / S3_CLASSEUR - 1):+.1f} % / classeur) ; contact en tête {o['tete'] / 1e3:.2f} kN")
o15 = cas(30.0, b_rond=15.0)
print(f"rondelle élastique ± 15 mm, mur 30 N/mm³ : H = {o15['H'] / 1e3:.3f} kN à y = {o15['yH']:.2f} ; pivot {o15['yc']:.2f} mm ; "
      f"σ3 nominale = {o15['snom']:.2f} ({100 * (o15['snom'] / S3_CLASSEUR - 1):+.1f} %)")
d = B[30.0]["snom"] / B[1e5]["snom"] - 1
print(f"montée de σ3 nominale entre mur raide et 30 N/mm³ (modèle s07) : {100 * d:+.1f} % "
      f"({B[1e5]['snom']:.1f} → {B[30.0]['snom']:.1f} N/mm², soit {100 * (B[30.0]['snom'] - B[1e5]['snom']) / SADM:.1f} points de σ_adm)")

# ---------------------------------------------------------------- C. poutre, pivot trouvé, V sur l'axe
print("\n# C. Poutre (V sur l'axe, σ = M/W + N/A, τ = H/A), pivot au point trouvé")
for yP in (0.0, B[30.0]["yc"], 11.76):
    H = R * (X - e / 2) / (Y - yP)
    M = H * (Y - e - r)
    s = M / (L * e * e / 6) + R / (L * e)
    t = H / (L * e)
    print(f"pivot {yP:5.2f} mm : H = {H / 1e3:.2f} kN ; taux 3 = {100 * math.sqrt(s * s + 3 * t * t) / SADM:.2f} % (classeur 88,86 %)")
# Quelle position de la résultante d'ancrage ferait dépasser le classeur sur mur à 30 N/mm³ (y_c = 11,76) ?
yc = 11.76
for yT in (85.0, 92.5, 99.0):
    H = R * (X - e / 2) / (yT - yc)
    M = H * (yT - a)
    s = M / (L * e * e / 6) + R / (L * e)
    print(f"  résultante d'ancrage à y_T = {yT:5.1f} mm (pivot 11,76) : σ3 nominale {s:.1f} N/mm² ({100 * (s / S3_CLASSEUR - 1):+.1f} %)")

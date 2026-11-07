# -*- coding: utf-8 -*-
"""V02 — Points 2, 3 et 6 de la relecture : modèle longitudinal de la cornière de 2 m (synthese/s_longi.py).

§2  « Effet de levier permis » avec section indéformable : le résultat dépend-il de la pénalité du mur kw ?
    Où le contact en tête apparaît-il, et le contact bas y est-il actif en même temps ?
§3  Cisaillement : règle du levier, poutre continue sur trois appuis rigides (poutre de Bernoulli écrite ICI,
    indépendante de s_longi), balayage de la raideur verticale des ancrages, 48EI/s³.
§6  Torsion ×1000 seule (G·It) et cornière ×1000 (E et G) selon la raideur des ancrages ; k_θ·s/(G·It).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/v02_longitudinal.py
"""
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "synthese"))
import numpy as np  # noqa: E402
import s_longi as SL  # noqa: E402

R, X, Y, e = 42760.0, 75.0, 85.0, 15.0
H0 = R * (X - e / 2) / Y
k_pl = SL.k_pl
rig = dict(kN=1e9, kV=1e9, k_pl=None)
raid = dict(kN=100e3, kV=30e3, k_pl=k_pl)
soup = dict(kN=10e3, kV=5e3, k_pl=k_pl)
f = lambda v: " / ".join(f"{x / 1e3:.2f}" for x in v)

print("§2 — section indéformable, ancrages rigides, contact en tête permis : dépendance à la pénalité du mur kw")
for kw in (1e4, 1e5, 1e6, 1e7):
    m = SL.modele(**rig, kw=kw, levier=True)
    zt = m["z"][m["C_haut"] > 0]
    zb = m["z"][m["C_bas"] > 0]
    both = np.intersect1d(np.round(zt, 3), np.round(zb, 3))
    print(f"  kw = {kw:.0e} N/mm par mm : N = {f(m['N'])} kN ; N1/(H/3) = {m['N'][0] / (H0 / 3):.2f} ; contact en tête "
          f"{m['C_haut'].sum() / 1e3:.2f} kN sur z = {zt.min() if len(zt) else float('nan'):.0f}–{zt.max() if len(zt) else float('nan'):.0f} mm ; "
          f"nœuds où haut ET bas sont en contact : {len(both)}")
print("  sans contact en tête (même balayage) :")
for kw in (1e4, 1e5, 1e6, 1e7):
    m = SL.modele(**rig, kw=kw, levier=False)
    print(f"  kw = {kw:.0e} : N = {f(m['N'])} kN ; N1/(H/3) = {m['N'][0] / (H0 / 3):.2f}")

print("\n§3 — cisaillement des ancrages")
zz = np.array([200.0, 1000.0, 1800.0])
zg = zz.mean()
Vlev = R / 3 + R * (400.0 - zg) * (zz - zg) / ((zz - zg) ** 2).sum()
print(f"  règle du levier (poutre rigide sur appuis égaux) : V = {f(Vlev)} kN")

# poutre de Bernoulli, flexion verticale seule, appuis rigides (écrite ici)
E, I = 210000.0, 444.881e4
h = 5.0
zs = np.arange(0.0, 2000.0 + h / 2, h)
n = len(zs)
K = np.zeros((2 * n, 2 * n))
ke = E * I / h ** 3 * np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                                [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
F = np.zeros(2 * n)
q = R / 300.0
for i in range(n - 1):
    d = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
    K[np.ix_(d, d)] += ke
    a, b = max(zs[i], 250.0), min(zs[i + 1], 550.0)
    if b > a:                               # charge répartie sur [250 ; 550], forces nodales cohérentes exactes
        xg, wg = np.polynomial.legendre.leggauss(6)
        for x_, w_ in zip(xg, wg):
            s = (a + b) / 2 + (b - a) / 2 * x_
            xi = (s - zs[i]) / h
            Nh = np.array([1 - 3 * xi ** 2 + 2 * xi ** 3, h * (xi - 2 * xi ** 2 + xi ** 3), 3 * xi ** 2 - 2 * xi ** 3, h * (-xi ** 2 + xi ** 3)])
            F[d] += -q * Nh * w_ * (b - a) / 2
appuis = [int(round(z / h)) for z in zz]
libres = [k for k in range(2 * n) if k not in [2 * a_ for a_ in appuis]]
U = np.zeros(2 * n)
U[libres] = np.linalg.solve(K[np.ix_(libres, libres)], F[libres])
Rv = (K @ U - F)[[2 * a_ for a_ in appuis]]
print(f"  poutre continue, flexion verticale seule, appuis rigides : V = {f(Rv)} kN (ΣV = {Rv.sum() / 1e3:.3f}) ; "
      f"V3 / règle du levier = {Rv[2] / Vlev[2]:.2f}")
print(f"  48EI/s³ (s = 800 mm, EI = {E * I / 1e9:.1f} kN·m²) = {48 * E * I / 800 ** 3 / 1e3:.1f} kN/mm")
print("  modèle longitudinal (flexion biaxiale + torsion), raideur verticale des ancrages k_V :")
for kV in (1e9, 1e5, 3e4, 1e4, 5e3, 1e3, 1e2):
    m = SL.modele(kN=100e3, kV=kV, k_pl=k_pl, levier=False)
    print(f"    k_V = {kV / 1e3:>9g} kN/mm (k_N = 100 + aile) : V = {f(m['V'])} kN")
for nom, kw in (("rigides", rig), ("raides 100/30 + aile", raid), ("souples 10/5 + aile", soup)):
    m = SL.modele(**kw, levier=False)
    print(f"    tableau du §5.3, {nom:22s}: V = {f(m['V'])} kN ; V3 / règle du levier = {m['V'][2] / Vlev[2]:.2f}")

print("\n§6 — torsion ×1000 selon la raideur des ancrages")
It0 = SL.It
for nom, kw in (("rigides", rig), ("raides 100/30 + aile", raid), ("souples 10/5 + aile", soup)):
    m1 = SL.modele(**kw, levier=False)
    SL.It = It0 * 1000.0
    m2 = SL.modele(**kw, levier=False)
    SL.It = It0
    m3 = SL.modele(**kw, levier=False, rigide=1e3)
    print(f"  ancrages {nom:22s}: N = {f(m1['N'])} ; G·It ×1000 : {f(m2['N'])} ; E et G ×1000 : {f(m3['N'])} kN "
          f"(R/n : {H0 / 3e3:.2f})")
G = 81000.0
for nom, kN in (("rigides", 1e9), ("raides 100 + aile", 1 / (1 / 100e3 + 1 / k_pl)), ("souples 10 + aile", 1 / (1 / 10e3 + 1 / k_pl))):
    kth = kN * Y ** 2
    print(f"  k_θ·s/(G·It), s = 800 mm, ancrages {nom} : {kth * 800 / (G * It0):.3g}")

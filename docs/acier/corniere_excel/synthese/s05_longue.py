# -*- coding: utf-8 -*-
"""S05 — Cornière longue : 2,00 m, trois ancrages, charge localisée.

Modèle longitudinal repris de extensions/e03_longitudinal.py (même équations, relancé ici) : poutre à section
indéformable sur appuis élastiques ; inconnues U, V (translations du centre de cisaillement S = (e/2 ; e/2)), φ
(rotation) ; flexion biaxiale avec produit d'inertie, torsion de Saint-Venant (gauchissement négligé) ; propriétés
lues dans le catalogue du site ; ancrage = tirant horizontal à la hauteur Y (raideur k_N en série avec la flexion
locale de l'aile k_pl) + appui vertical k_V ; mur = contact unilatéral continu à l'arête inférieure (pivot P0).
Le contact en tête d'aile (« effet de levier ») N'EST PAS retenu : la section étant indéformable, il ne représente pas la
flexion de l'aile au-dessus de l'ancrage et son résultat suit la pénalité du mur (§2d) ; l'effet de levier se juge sur
le maillage de la coupe (s07).

Exemple (HYPOTHÈSES d'illustration) : L120x120x15, longueur 2 000 mm, ancrages à z = 200, 1 000, 1 800 mm, Y = 85,
X = 75, R = 42,76 kN répartis sur 300 mm centrés à z = 400 mm.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s05_longue.py
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import s_commun as K  # noqa: E402

S = K.Sortie("s05_longue")
C = K.CAS
e, r, R, Xc, Yc = C["e"], C["r"], C["R"], C["X"], C["Y"]
import s_longi as SL  # noqa: E402
from s_longi import (prof, modele, E_, G_, Iuu, Iyy, Iuy, It, Wel)  # noqa: E402


Y, yP, Xp = Yc, 0.0, Xc
m_tee = (Y - e) - 0.8 * r
leff = min(2 * math.pi * m_tee, 4 * m_tee + 1.25 * (120.0 - Y))
k_pl = 3 * E_ * (leff * e ** 3 / 12) / (Y - yP) ** 3
H_tot = R * (Xp - e / 2) / (Y - yP)
S("# S05 — cornière longue 2,00 m, 3 ancrages")
S("")
S("## 1. Section et raideurs")
S(f"L120x120x15 (catalogue du site) : Iy = Iz = {prof['Iy']} cm⁴, Iyz = {prof['Iyz']} cm⁴, It = {prof['It']} cm⁴, Wel = {prof['Wel_y']} cm³")
S(f"G·It = {G_ * It / 1e9:.2f} kN·m² ; E·I = {E_ * Iuu / 1e9:.1f} kN·m² (rapport {E_ * Iuu / (G_ * It):.0f})")
S(f"tronçon en T (analogie EN 1993-1-8, à confirmer) : m = (Y − e) − 0,8·r = {m_tee:.1f} mm ; "
  f"l_eff = min(2π·m ; 4·m + 1,25·(h − Y)) = min({2 * math.pi * m_tee:.1f} ; {4 * m_tee + 1.25 * (120.0 - Y):.1f}) = {leff:.1f} mm")
S(f"flexion locale de l'aile : k_pl = 3·E·(l_eff·e³/12)/Y³ = {k_pl / 1e3:.1f} kN/mm — console de longueur Y ENCASTRÉE au talon "
  f"(bloc d'angle supposé ne pas tourner par rapport à la section indéformable) ; console de longueur Ye = {Y - e - r:.0f} mm "
  f"encastrée en section 3 : {3 * E_ * (leff * e ** 3 / 12) / (Y - e - r) ** 3 / 1e3:.1f} kN/mm")
for kN_ in (10e3, 30e3, 100e3):
    kNe = 1 / (1 / kN_ + 1 / k_pl)
    S(f"ancrage {kN_ / 1e3:.0f} kN/mm + aile → {kNe / 1e3:.1f} kN/mm ; k_θ·s/(G·It) (s = 800 mm) = {kNe * (Y - yP) ** 2 * 800 / (G_ * It):.2f}")

S("")
S("## 2. Répartition entre les ancrages")
S(f"référence « R/n » : V = {R / 3e3:.2f} kN et N = H/3 = {H_tot / 3e3:.2f} kN par ancrage (H = R·(X − e/2)/Y = {H_tot / 1e3:.2f} kN) ; "
  f"le classeur donnerait N = V = {R * Xp / (Y - 10) / 3e3:.2f} kN")
zz = np.array([200.0, 1000.0, 1800.0])
zg = zz.mean()
V_lev = R / 3 + R * (400.0 - zg) * (zz - zg) / ((zz - zg) ** 2).sum()
S("règle du levier d'une poutre rigide : V_i = R/n + R·e_z·(z_i − z_G)/Σ(z_i − z_G)² = " + " / ".join(f"{v / 1e3:.2f}" for v in V_lev) + " kN")
cas = [
    ("ancrages rigides", dict(kN=1e9, kV=1e9, k_pl=None)),
    ("ancrages raides (100 / 30 kN/mm) + aile", dict(kN=100e3, kV=30e3, k_pl=k_pl)),
    ("ancrages souples (10 / 5 kN/mm) + aile", dict(kN=10e3, kV=5e3, k_pl=k_pl)),
]
S("| hypothèse | N1 | N2 | N3 (kN) | V1 | V2 | V3 (kN) | N1/(H/3) | V1/(R/3) | T max (kN·m) | τ torsion (N/mm²) | M long max (kN·m) | σz max (N/mm²) |")
res = {}
for nom, kw in cas:
    m = modele(**kw, levier=False)
    tau = np.abs(m["T"]).max() * e / It
    Mlong = np.abs(m["Mv"][5:-5]).max()
    res[nom] = dict(N=m["N"].tolist(), V=m["V"].tolist(), T=float(np.abs(m["T"]).max()), tau=float(tau),
                    Ml=float(Mlong), sz=float(m["sz"][5:-5].max()), levier=float(m["C_haut"].sum()),
                    z=m["z"][::4].tolist(), Tz=m["T"][::4].tolist(), Mvz=m["Mv"][::4].tolist())
    S(f"| {nom} | " + " | ".join(f"{x / 1e3:.2f}" for x in m["N"]) + " | " + " | ".join(f"{x / 1e3:.2f}" for x in m["V"])
      + f" | {m['N'][0] / (H_tot / 3):.2f} | {m['V'][0] / (R / 3):.2f} | {np.abs(m['T']).max() / 1e6:.3f} | {tau:.0f} | "
      f"{Mlong / 1e6:.2f} | {m['sz'][5:-5].max():.0f} |")
    S(f"   contrôle : ΣV = {m['V'].sum() / 1e3:.3f} kN ; ΣN − ΣC = {(m['N'].sum() - m['C_bas'].sum() - m['C_haut'].sum()) / 1e3:.4f} kN ; "
      f"contact en tête {m['C_haut'].sum() / 1e3:.2f} kN ; itérations {m['it']}")
sadm = 355 / 1.5
S(f"τ_adm = {sadm / math.sqrt(3):.1f} N/mm² ; torsion : " + " ; ".join(f"{n} {100 * v['tau'] / (sadm / math.sqrt(3)):.0f} %" for n, v in res.items()))
S.v("cas", {k: {kk: vv for kk, vv in v.items() if kk not in ("z", "Tz", "Mvz")} for k, v in res.items()})
S.v("H_tot", H_tot)
S.v("V_levier", V_lev.tolist())

S("")
S("## 2b. « R/n » en traction : il faut une cornière raide en torsion PAR RAPPORT AUX ancrages (k_θ·s/(G·It) → 0)")
for nom, kw in cas:
    m1 = modele(**kw, levier=False)
    SL.It = It * 1000.0
    m2 = modele(**kw, levier=False)
    SL.It = It
    m3 = modele(**kw, levier=False, rigide=1e3)
    kNe = kw["kN"] if kw["k_pl"] is None else 1 / (1 / kw["kN"] + 1 / kw["k_pl"])
    S(f"{nom} : N = " + " / ".join(f"{x / 1e3:.2f}" for x in m1["N"]) + " ; G·It ×1000 : " + " / ".join(f"{x / 1e3:.2f}" for x in m2["N"])
      + " ; E et G ×1000 : " + " / ".join(f"{x / 1e3:.2f}" for x in m3["N"]) + f" kN ; k_θ·s/(G·It) (s = 800) = {kNe * Y ** 2 * 800 / (G_ * It):.3g}")

S("")
S("## 2c. Cisaillement : règle du levier = limite d'ancrages SOUPLES devant la cornière ; ancrages raides = poutre continue")
for kV in (1e9, 1e5, 3e4, 1e4, 5e3, 1e3, 1e2):
    m = modele(kN=100e3, kV=kV, k_pl=k_pl, levier=False)
    S(f"k_V = {kV / 1e3:g} kN/mm (k_N = 100 kN/mm + aile) : V = " + " / ".join(f"{x / 1e3:.2f}" for x in m["V"]) + " kN")
S(f"V3 / règle du levier : rigides {res[cas[0][0]]['V'][2] / V_lev[2]:.2f} ; raides {res[cas[1][0]]['V'][2] / V_lev[2]:.2f} ; "
  f"souples {res[cas[2][0]]['V'][2] / V_lev[2]:.2f}")
# poutre de Bernoulli, flexion verticale seule, trois appuis rigides (écrite ici, sans s_longi)
hh = 5.0
zb = np.arange(0.0, 2000.0 + hh / 2, hh)
nb = len(zb)
Kb = np.zeros((2 * nb, 2 * nb))
Fb = np.zeros(2 * nb)
keb = E_ * Iuu / hh ** 3 * np.array([[12, 6 * hh, -12, 6 * hh], [6 * hh, 4 * hh * hh, -6 * hh, 2 * hh * hh],
                                      [-12, -6 * hh, 12, -6 * hh], [6 * hh, 2 * hh * hh, -6 * hh, 4 * hh * hh]])
xg, wg = np.polynomial.legendre.leggauss(6)
for i in range(nb - 1):
    dd = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
    Kb[np.ix_(dd, dd)] += keb
    a_, b_ = max(zb[i], 250.0), min(zb[i + 1], 550.0)
    if b_ > a_:
        for x_, w_ in zip(xg, wg):
            xi = ((a_ + b_) / 2 + (b_ - a_) / 2 * x_ - zb[i]) / hh
            Nh = np.array([1 - 3 * xi ** 2 + 2 * xi ** 3, hh * (xi - 2 * xi ** 2 + xi ** 3), 3 * xi ** 2 - 2 * xi ** 3, hh * (-xi ** 2 + xi ** 3)])
            Fb[dd] += -(R / 300.0) * Nh * w_ * (b_ - a_) / 2
ia = [2 * int(round(z / hh)) for z in zz]
lib = [k for k in range(2 * nb) if k not in ia]
Ub = np.zeros(2 * nb)
Ub[lib] = np.linalg.solve(Kb[np.ix_(lib, lib)], Fb[lib])
Vb = (Kb @ Ub - Fb)[ia]
S("poutre continue, flexion verticale seule, appuis rigides : V = " + " / ".join(f"{x / 1e3:.2f}" for x in Vb) + " kN")
S(f"raideur de flexion d'une travée 48·E·I/s³ (s = 800 mm) = {48 * E_ * Iuu / 800 ** 3 / 1e3:.1f} kN/mm")

S("")
S("## 2d. Pourquoi l'« effet de levier » n'a pas de sens dans ce modèle (section indéformable)")
S("ancrages rigides, contact en tête permis, pénalité du mur kw variable : le résultat ne converge pas, il suit kw")
for kw_ in (1e4, 1e5, 1e6, 1e7):
    m = modele(kN=1e9, kV=1e9, k_pl=None, levier=True, kw=kw_)
    S(f"   kw = {kw_:.0e} N/mm par mm : N = " + " / ".join(f"{x / 1e3:.2f}" for x in m["N"]) + f" kN ; N1/(H/3) = {m['N'][0] / (H_tot / 3):.2f} ; "
      f"contact en tête {m['C_haut'].sum() / 1e3:.2f} kN ; nœuds où le haut ET le bas touchent : "
      f"{int(np.sum((m['C_haut'] > 0) & (m['C_bas'] > 0)))}")
S("   (sans contact en tête, le même balayage donne N1/(H/3) = " + " ; ".join(
    f"{modele(kN=1e9, kV=1e9, k_pl=None, levier=False, kw=k_)['N'][0] / (H_tot / 3):.2f}" for k_ in (1e4, 1e5, 1e6, 1e7)) + ")")

S("")
S("## 3. Conséquence locale : aile verticale au droit de l'ancrage le plus chargé, largeur l_eff")
S("σ = 6·N1·bras/(l_eff·e²) + V1/(l_eff·e) — bras m (tronçon en T, cohérent avec l_eff) ; pour mémoire bras Ye = Y − e − r (section 3)")
for nom in ("ancrages rigides", "ancrages raides (100 / 30 kN/mm) + aile", "ancrages souples (10 / 5 kN/mm) + aile"):
    N1, V1 = max(res[nom]["N"]), max(res[nom]["V"])
    s_m = 6 * N1 * m_tee / (leff * e ** 2) + V1 / (leff * e)
    s_y = 6 * N1 * (Y - e - r) / (leff * e ** 2) + V1 / (leff * e)
    S(f"{nom} : N1 = {N1 / 1e3:.2f} kN, V1 = {V1 / 1e3:.2f} kN → bras m = {m_tee:.1f} : σ = {s_m:.1f} N/mm² ({100 * s_m / sadm:.1f} %) ; "
      f"bras Ye = {Y - e - r:.0f} : {s_y:.1f} N/mm² ({100 * s_y / sadm:.1f} %)")

S("")
S("## 4. Variante : ancrages à 400, 1 000, 1 600 mm (un ancrage au droit de la charge)")
for nom, kw in cas[:3]:
    m = modele(ancrages=(400.0, 1000.0, 1600.0), levier=False, **kw)
    S(f"{nom} : N = " + ", ".join(f"{x / 1e3:.2f}" for x in m["N"]) + " kN ; V = " + ", ".join(f"{x / 1e3:.2f}" for x in m["V"])
      + f" kN ; τ torsion max = {np.abs(m['T']).max() * e / It:.0f} N/mm²")

with open(os.path.join(K.SORTIES, "s05_courbes.json"), "w", encoding="utf-8") as fh:
    json.dump(dict(res=res, H_tot=H_tot, R=R, ancrages=zz.tolist(), V_levier=V_lev.tolist()), fh)
S.ecrire()

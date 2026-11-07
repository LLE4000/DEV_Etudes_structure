# -*- coding: utf-8 -*-
"""c07 — La largeur b (« longueur reprise », P8!C8) et la répartition le long de la cornière.

1. diffusion à 45° d'une charge d'appui de largeur c jusqu'aux sections 1 et 2 : largeur efficace, facteur sur σ ;
2. l'aile verticale autour d'une cheville : condition pour que le moment soit uniforme sur b à la section 3 ;
   longueurs efficaces du tronçon en T équivalent (analogie EN 1993-1-8, rangée isolée) ;
3. répartition entre ancrages : poutre continue (EF de poutre 1D) sur appuis RIGIDES ou ÉLASTIQUES (chevilles),
   - le cas du classeur (300 mm, 3 chevilles, charge répartie sur toute la longueur),
   - la cornière de 2,00 m de la demande (3 chevilles, charge de 42,76 kN sur 300 mm entre deux chevilles).
"""
import json
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import numpy as np  # noqa: E402
import commun as K  # noqa: E402

RACINE = os.path.dirname(ICI)
p = K.CAS
e, r, X, Y, R = p["e"], p["r"], p["X"], p["Y"], p["R"]
b = p["b_cm"] * 10
xl = K.classeur(**p)
res = {}

# ---------------------------------------------------------------- 1. diffusion dans l'aile horizontale
print("Diffusion à 45° depuis une charge d'appui de largeur c (centrée, loin des extrémités) :")
d1, d2 = xl["C19_Xe"], xl["C20_Xer"]
tab = []
for c in (50.0, 100.0, 150.0, 200.0, 250.0, 300.0):
    be1, be2 = min(b, c + 2 * d1), min(b, c + 2 * d2)
    tab.append(dict(c=c, b_eff_S1=be1, facteur_S1=b / be1, b_eff_S2=be2, facteur_S2=b / be2,
                    taux_S1=xl["S1"]["taux"] * b / be1))
    print(f"  c = {c:5.0f} mm : b_eff(S1) = {be1:5.0f} mm (σ ×{b / be1:.2f} → taux S1 {xl['S1']['taux'] * b / be1 * 100:6.1f} %),"
          f" b_eff(S2) = {be2:5.0f} mm (×{b / be2:.2f})")
print(f"  largeur d'appui minimale pour que b = {b:.0f} mm soit atteint : S1 c ≥ {b - 2 * d1:.0f} mm, "
      f"S2 c ≥ {b - 2 * d2:.0f} mm ; charge à une extrémité (diffusion d'un seul côté) : S1 c ≥ {b - d1:.0f} mm")
res["diffusion"] = tab

# ---------------------------------------------------------------- 2. autour d'une cheville
n = p["n_anc"]
s_anc = b / n
d_w = 30.0
Ye = xl["C23_Ye"]
b_cheville = d_w + 2 * Ye
print(f"\nChevilles : {n} sur {b:.0f} mm → entraxe {s_anc:.0f} mm si régulières (le classeur ne donne pas leur position).")
print(f"  diffusion à 45° depuis une rondelle Ø{d_w:.0f} jusqu'à la section 3 ({Ye:.0f} mm plus bas) : {b_cheville:.0f} mm "
      f"≥ entraxe {s_anc:.0f} → moment uniforme sur b à la section 3 si l'entraxe ≤ {b_cheville:.0f} mm")
m = (Y - e) - 0.8 * r                      # distance de l'axe de la cheville à la face de l'autre aile − 0,8 r
ex = p["haut"] - Y
lcp, lnc = 2 * math.pi * m, 4 * m + 1.25 * ex
print(f"  tronçon en T équivalent (analogie EN 1993-1-8, rangée isolée) : m = (Y − e) − 0,8 r = {m:.1f} mm, "
      f"e = haut − Y = {ex:.0f} mm → l_eff,cp = 2πm = {lcp:.0f} mm, l_eff,nc = 4m + 1,25e = {lnc:.0f} mm")
res["cheville"] = dict(entraxe=s_anc, diffusion_S3=b_cheville, m=m, ex=ex, l_eff_cp=lcp, l_eff_nc=lnc)


# ---------------------------------------------------------------- 3. poutre continue (EF de poutre d'Euler-Bernoulli)
def poutre(L, appuis, k_appui, charges, EI, n_el=800):
    """L (mm), appuis (positions mm), k_appui (N/mm, None = rigide), charges [(x0, x1, P en N)] réparties
    uniformément sur [x0, x1]. Rend les réactions (N) et le moment fléchissant max (N·mm)."""
    xs = np.linspace(0, L, n_el + 1)
    for a in appuis:                                   # nœuds exactement sur les appuis
        xs = np.union1d(xs, [a])
    for (x0, x1, _) in charges:
        xs = np.union1d(xs, [x0, x1])
    nn = len(xs)
    Kg = np.zeros((2 * nn, 2 * nn))
    F = np.zeros(2 * nn)
    for i in range(nn - 1):
        h = xs[i + 1] - xs[i]
        ke = EI / h ** 3 * np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                                     [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]])
        idx = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
        Kg[np.ix_(idx, idx)] += ke
        xm = 0.5 * (xs[i] + xs[i + 1])
        for (x0, x1, P) in charges:
            if x0 <= xm <= x1:
                w = P / (x1 - x0)
                F[idx] += w * np.array([h / 2, h * h / 12, h / 2, -h * h / 12])
    ia = [int(np.argmin(np.abs(xs - a))) for a in appuis]
    if k_appui is None:
        libres = [j for j in range(2 * nn) if j not in [2 * i for i in ia]]
        u = np.zeros(2 * nn)
        u[libres] = np.linalg.solve(Kg[np.ix_(libres, libres)], F[libres])
        reac = -(Kg @ u - F)[[2 * i for i in ia]]      # réaction vers le haut (opposée à la charge), positive
    else:
        for i in ia:
            Kg[2 * i, 2 * i] += k_appui
        u = np.linalg.solve(Kg, F)
        reac = np.array([k_appui * u[2 * i] for i in ia])
    # moments aux nœuds par la statique (somme des moments à gauche)
    Mx = []
    for xx in xs:
        Mv = 0.0
        for a, Ra in zip(appuis, reac):
            if a < xx:
                Mv += -Ra * (xx - a)       # Ra : réaction opposée à la charge, positive (rigide comme ressort)
        for (x0, x1, P) in charges:
            if xx > x0:
                xe = min(xx, x1)
                w = P / (x1 - x0)
                Mv += w * (xe - x0) * (xx - (x0 + xe) / 2)
        Mx.append(Mv)
    Mx = np.array(Mx)
    return np.abs(reac), float(np.abs(Mx).max()), float(xs[int(np.argmax(np.abs(Mx)))])


with open("/home/user/integ/acier/donnees/profiles.json", encoding="utf-8") as f:
    prof = {pp["nom"]: pp for pp in json.load(f)["profils"]}
L120 = prof["L 120x120x15"]
Iy = L120["Iy"] * 1e4                      # mm⁴ (axe géométrique parallèle à une aile)
Wy = L120["Wel_y"] * 1e3                   # mm³ (minimal)
It = L120["It"] * 1e4
EI = 210000.0 * Iy
print(f"\nL120×120×15 (catalogue du site) : I_y = {L120['Iy']:.1f} cm⁴, W_el,y = {L120['Wel_y']:.2f} cm³, "
      f"I_t = {L120['It']:.2f} cm⁴")
k1_01 = xl["N_ELS"] * 1000 / 0.1          # cheville : 0,1 mm sous 14,25 kN
k1_05 = xl["N_ELS"] * 1000 / 0.5
# (a) le cas du classeur
for nom, k in (("appuis rigides", None), ("chevilles δ = 0,1 mm", k1_01), ("chevilles δ = 0,5 mm", k1_05)):
    reac, Mmax, xm = poutre(300.0, [50.0, 150.0, 250.0], k, [(0.0, 300.0, R * 1000)], EI)
    print(f"  classeur (300 mm, chevilles à 50/150/250 supposées) — {nom:22s} : réactions "
          f"{', '.join(f'{v / 1000:.2f}' for v in reac)} kN (R/n = {R / n:.2f}) ; M_long max = {Mmax / 1e6:.4f} kN·m")
    res.setdefault("classeur_long", []).append(dict(nom=nom, reactions=list(reac / 1000), Mmax=Mmax / 1e6))
# (b) la cornière de 2,00 m de la demande
L2 = 2000.0
app = [100.0, 1000.0, 1900.0]
ch = [(400.0, 700.0, R * 1000)]
print(f"  cornière 2,00 m, chevilles à {app} mm, charge {R} kN répartie sur 400–700 mm :")
for nom, k in (("appuis rigides", None), ("chevilles δ = 0,1 mm", k1_01), ("chevilles δ = 0,5 mm", k1_05)):
    reac, Mmax, xm = poutre(L2, app, k, ch, EI)
    sl = Mmax / Wy
    print(f"    {nom:22s} : réactions {', '.join(f'{v / 1000:6.2f}' for v in reac)} kN (R/n = {R / 3:.2f}) ; "
          f"M_long max = {Mmax / 1e6:.3f} kN·m à x = {xm:.0f} mm → σ_long = {sl:.1f} MPa (taux seul "
          f"{sl / xl['adm'] * 100:.1f} %)")
    res.setdefault("cornière_2m", []).append(dict(nom=nom, reactions=list(reac / 1000), Mmax=Mmax / 1e6, x=xm,
                                                  sigma_long=sl))
# moment de torsion à transmettre de la charge vers les chevilles (le moment transversal R·X)
GIt = 81000.0 * It
print(f"  moment transversal à transmettre le long de la cornière jusqu'aux chevilles : R·X = {R * X / 1000:.3f} kN·m ; "
      f"raideur de torsion G·I_t = {GIt / 1e9:.2f} kN·m²/rad (I_t = {L120['It']:.2f} cm⁴)")
# rotation de torsion si tout le moment passait en torsion sur 450 mm (de la charge à la cheville voisine)
th = (R * X * 1e3 / 2) * 450.0 / GIt
print(f"    si la moitié passait par torsion libre sur 450 mm : rotation {math.degrees(th):.2f}° "
      f"(→ elle passe en réalité par la flexion de l'aile verticale contre le mur, plaque appuyée aux chevilles)")
res["torsion"] = dict(RX=R * X / 1000, GIt=GIt, rotation_deg=math.degrees(th))
json.dump(res, open(os.path.join(RACINE, "sorties", "c07_largeur.json"), "w"), ensure_ascii=False, indent=1,
          default=float)

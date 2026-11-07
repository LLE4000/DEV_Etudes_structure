# -*- coding: utf-8 -*-
"""V01b — Point 1, suite : le classeur n'est-il dépassé QUE si la résultante d'ancrage monte au-dessus de l'axe ?

1. Poutre (V et résultante d'ancrage sur l'axe) : pivot y_P* pour lequel la section 3 de l'équilibre atteint le
   classeur (σ nominale = 209,66 N/mm² ; taux 88,86 %).
2. Maillage de s07 (écrou ± 3 mm, maillage fin) : modules de réaction 20, 10, 5 N/mm³ → pivot et σ3 nominale ;
   seuil par interpolation logarithmique.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/v01b_seuil_mur.py
"""
import math
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "extensions"))
import ef_l  # noqa: E402

R, X, Y, e, r, L = 42760.0, 75.0, 85.0, 15.0, 13.0, 300.0
a = e + r
SADM = 355.0 / 1.5
W, A = L * e * e / 6, L * e
S3C = R * X * (Y - a) / (Y - e / 2) / W
T3C = math.sqrt(S3C ** 2 + 3 * (R / A) ** 2) / SADM


def poutre(yP):
    H = R * (X - e / 2) / (Y - yP)
    s = H * (Y - a) / W + R / A
    return s, math.sqrt(s * s + 3 * (H / A) ** 2) / SADM


def bis(f, lo, hi):
    for _ in range(200):
        c = (lo + hi) / 2
        if f(c) > 0:
            hi = c
        else:
            lo = c
    return (lo + hi) / 2


print(f"classeur : σ3 = {S3C:.3f} N/mm², taux 3 = {100 * T3C:.2f} %")
yS = bis(lambda y: poutre(y)[0] - S3C, 0.0, 40.0)
yT = bis(lambda y: poutre(y)[1] - T3C, 0.0, 40.0)
print(f"poutre, ancrage sur son axe : σ3 nominale = classeur pour un pivot à {yS:.2f} mm ; taux 3 = classeur pour {yT:.2f} mm")

GEO = dict(B=120.0, H=120.0, e=e, r=r, Y=Y)
MAIL = dict(n_t=10, n_c=16, n_leg=40, grad=4.0)
pts = []
for km in (30.0, 20.0, 10.0, 5.0):
    pb = ef_l.Probleme(GEO, "A", maillage=MAIL, solveur="bande")
    pb.resoudre(q := R / L, X, k_anc=300.0, k_mur=km, b_rond=3.0)
    rr = pb.reactions
    bas = rr["mur_y"] < Y
    yc = (rr["mur_f"][bas] * rr["mur_y"][bas]).sum() / rr["mur_f"][bas].sum()
    cp = pb.coupe((0.0, a), (e, a), n=60, normale="y")
    sn = abs(cp["M"]) / (e * e / 6) + cp["N"] / e
    pts.append((km, sn))
    print(f"maillage s07, mur {km:g} N/mm³ : H = {rr['H'] * L / 1e3:.2f} kN à y = {rr['y_H']:.2f} ; pivot {yc:.2f} mm ; "
          f"σ3 nominale {sn:.1f} N/mm² ({100 * (sn / S3C - 1):+.1f} % / classeur)", flush=True)
for (k1, s1), (k2, s2) in zip(pts[:-1], pts[1:]):
    if (s1 - S3C) * (s2 - S3C) <= 0:
        f = (S3C - s1) / (s2 - s1)
        ks = math.exp(math.log(k1) + f * (math.log(k2) - math.log(k1)))
        print(f"seuil (interpolation logarithmique entre {k1:g} et {k2:g} N/mm³) : σ3 nominale = classeur vers {ks:.0f} N/mm³")

# -*- coding: utf-8 -*-
"""Relecture adverse — 5b. Sensibilités de mon maillage : finesse (pic au congé), largeur d'appui de l'écrou
(± 3 mm contre rondelle ± 15 mm) sur mur raide et souple. Lancer : python3.12 -I -B r05b_sensibilites.py <h>"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import griddata
import r05_ef as F

h = float(sys.argv[1])
N, El = F.maillage(h)
K = F.assembler(N, El)
print(f"h = {h} : {len(N)} nœuds")
a, e, r, R, LZ = F.a, F.e, F.r, F.R, F.LZ
us = np.linspace(0.0, e, 301)
for washer in (3.0, 15.0):
    for ks in (None, 1e3, 30.0):
        res = F.resoudre(N, El, K, "A", k_mur_s=ks, washer=washer, pas=h)
        Ht, C, yc, Ch = F.resume(f"rondelle ±{washer:g}, mur {'raide' if ks is None else f'{ks:g} N/mm³'}", N, res, F.Y)
        S = F.contraintes_nodales(N, El, res["U"])
        syv = griddata(N, S[:, 1], np.array([(u, a) for u in us]), method="linear")
        Nn = np.trapezoid(syv, us) * LZ
        Mm = np.trapezoid(syv * (us - e / 2), us) * LZ
        snom = abs(Mm) / (LZ * e * e / 6) + Nn / (LZ * e)
        print(f"     section 3 : M = {abs(Mm)/1e6:.4f} kN·m, σ nominal = {snom:.2f} ({snom/209.662-1:+.2%} / classeur 209,66)")
        if washer == 3.0 and ks is None:
            sx, sy, txy = S[:, 0], S[:, 1], S[:, 2]
            s1 = (sx + sy) / 2 + np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
            arc = np.where(np.abs(np.hypot(N[:, 0] - a, N[:, 1] - a) - r) < 1e-6)[0]
            k = arc[np.argmax(s1[arc])]
            uu = N[k, 0]
            tl = a - math.sqrt(max(r * r - (a - uu) ** 2, 0))
            sp_ = R * (F.X - uu) / (LZ * tl * tl / 6)
            print(f"     pic au congé σ1 = {s1[k]:.1f} N/mm² à {a-uu:.2f} mm du pied (u = {uu:.2f}) ; poutre {sp_:.1f} ; "
                  f"rapport {s1[k]/sp_:.3f} ; / σadm {s1[k]/(355/1.5):.3%}")
            for u_, yv in ((a, e), (e, a)):
                comp = sx if u_ == a else sy
                v = griddata(N, comp, np.array([(u_, yv)]), method="linear")[0]
                print(f"     pied du congé ({u_:g}, {yv:g}) : {v:.1f}")
            v2b = griddata(N, sx, np.array([(e, 0.0)]), method="linear")[0]
            v2h = griddata(N, sx, np.array([(e, a)]), method="linear")[0]
            v4m = griddata(N, sy, np.array([(0.0, e)]), method="linear")[0]
            print(f"     section 2 : bas {v2b:.2f}, haut {v2h:.2f} (×{abs(v2b)/65.449:.3f}) ; section 4 au mur {v4m:.2f} (×{abs(v4m)/73.894:.3f})")

# -*- coding: utf-8 -*-
"""Relecture adverse — 5c. Convergence du pic au congé (mur raide, ancrage ±3 mm, 30 kN/mm)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import r05_ef as F
for h in [float(x) for x in sys.argv[1:]]:
    N, El = F.maillage(h)
    K = F.assembler(N, El)
    res = F.resoudre(N, El, K, "A", pas=h)
    Ht, C, yc, _ = F.resume(f"h = {h}", N, res, F.Y)
    S = F.contraintes_nodales(N, El, res["U"])
    sx, sy, txy = S[:, 0], S[:, 1], S[:, 2]
    s1 = (sx + sy) / 2 + np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    a, r = F.a, F.r
    arc = np.where(np.abs(np.hypot(N[:, 0] - a, N[:, 1] - a) - r) < 1e-6)[0]
    for cote, m in (("horizontale", N[arc, 0] > N[arc, 1]), ("verticale", N[arc, 0] < N[arc, 1])):
        k = arc[m][np.argmax(s1[arc[m]])]
        d = a - N[k, 0] if cote == "horizontale" else a - N[k, 1]
        print(f"   pic côté aile {cote} : σ1 = {s1[k]:.1f} N/mm² à {d:.2f} mm du pied ; / σadm {s1[k]/(355/1.5):.2%}")

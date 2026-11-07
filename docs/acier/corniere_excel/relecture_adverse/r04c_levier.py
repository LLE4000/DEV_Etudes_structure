# -*- coding: utf-8 -*-
"""Relecture adverse — 4c. Cornière longue, « effet de levier permis » avec section indéformable :
dépendance du résultat à la raideur de pénalité du mur (indétermination)."""
import os, sys, runpy, io, contextlib
sys.argv = [sys.argv[0], sys.argv[1]]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    M = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r04_longi.py"))
import numpy as np
K0, F = M["rigidite"](), M["charge"]()
H0 = M["H0"]
for kmur in (1e4, 1e5, 1e6, 1e7):
    o = M["resoudre"](K0, F, kN=1e9, kV=1e9, kmur=kmur, haut=True)
    print(f"mur {kmur:.0e} N/mm par mm : N = {np.round(o['N']/1e3,2)} ; contact en tête {o['Ch']/1e3:.2f} kN ; N1/(H/3) = {o['N'][0]/(H0/3):.2f}")

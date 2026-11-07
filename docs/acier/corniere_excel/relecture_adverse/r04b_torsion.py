# -*- coding: utf-8 -*-
"""Relecture adverse — 4b. Torsion ×1000 selon la raideur des ancrages ; rapport k_θ·s/(G·It)."""
import os, sys, runpy
sys.argv = [sys.argv[0], sys.argv[1]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    M = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r04_longi.py"))
import numpy as np
K1000 = M["rigidite"](1000.0)
F = M["charge"]()
kloc = 81.4e3
for nom, kw in [("rigides", dict()), ("raides 100/30 + aile", dict(kN=1/(1/100e3+1/kloc), kV=30e3)),
                ("souples 10/5 + aile", dict(kN=1/(1/10e3+1/kloc), kV=5e3))]:
    o = M["resoudre"](K1000, F, **kw)
    print(f"GIt ×1000, {nom:22s}: N = {np.round(o['N']/1e3, 2)}")
G, It = 81000.0, M["It"]
for nom, kN in [("raides", 1/(1/100e3+1/kloc)), ("souples", 1/(1/10e3+1/kloc))]:
    kth = kN * 85.0 ** 2
    for s in (800.0, 400.0):
        print(f"k_θ·s/(G·It) {nom}, s = {s:g} : {kth*s/(G*It):.2f}")

# -*- coding: utf-8 -*-
"""Relecture adverse — 4d. Cornière longue à l'ELS/ELU (§5.5) : G = 30 kN sur 2 m, Q = 30 kN sur 300 mm à 0,40 m."""
import os, sys, runpy, io, contextlib
sys.argv = [sys.argv[0], sys.argv[1]]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    M = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r04_longi.py"))
import numpy as np
K0, charge, resoudre = M["rigidite"](), M["charge"], M["resoudre"]
kw = dict(kN=1 / (1 / 100e3 + 1 / 81.4e3), kV=30e3)
FG = charge(0.0, 2000.0, 30e3)
FQ = charge(250.0, 550.0, 30e3)
els = resoudre(K0, FG + FQ, **kw)
elu = resoudre(K0, 1.35 * FG + 1.5 * FQ, **kw)
g = resoudre(K0, FG, **kw); q = resoudre(K0, FQ, **kw)
sup = 1.35 * g["N"] + 1.5 * q["N"]
print("N_ELS =", np.round(els["N"] / 1e3, 2))
print("N_ELU =", np.round(elu["N"] / 1e3, 2), " V_ELU =", np.round(elu["V"] / 1e3, 2))
print("1,425 × N_ELS =", np.round(1.425 * els["N"] / 1e3, 2), " écarts", np.round((1.425 * els["N"] / elu["N"] - 1) * 100, 1), "%")
print("1,611 × N_ELS =", np.round(68.9 / 42.76 * els["N"] / 1e3, 2))
print("superposition 1,35 N_G + 1,5 N_Q =", np.round(sup / 1e3, 3), " écart max", np.round(np.abs(sup - elu["N"]).max() / 1e3, 3), "kN")
Ht = (1.35 * 30e3 + 1.5 * 30e3) * 67.5 / 85.0
print(f"platine rigide : {Ht/3e3:.2f} kN par ancrage ; / N1 calculé {Ht/3/elu['N'].max()-1:.1%}")

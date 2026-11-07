# -*- coding: utf-8 -*-
"""Diagnostic : résultantes de coupe face à la statique exacte (N = 0 sur les coupes verticales de l'aile
horizontale ; N = q sur les coupes horizontales de l'aile verticale), selon le pas du maillage."""
import os, sys
ICI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ICI)
import commun as K, ef_coupe as EF
p = K.CAS; e, r, X, Y = p["e"], p["r"], p["X"], p["Y"]; q = p["R"] * 1000 / (p["b_cm"] * 10)
for h in (1.0, 0.5):
    m = EF.Coupe(p["larg"], p["haut"], e, r, h=h)
    m.resoudre(q, X, Y, k_tige=p["n_anc"] * (14253.3 / 0.1) / 300)
    print(f"pas {h} mm")
    for xx in (15.0, 15.5, 17, 20, 24, 28, 35):
        t = K.epaisseur_verticale(xx, e, r) if xx > e else e + r
        c = m.coupe((xx, 0), (xx, t), n=400)
        print(f"   coupe verticale x = {xx:5.1f} (t = {t:6.2f}) : N = {c['N']:+8.2f} (0)   V = {c['V']:+8.2f} (±{q:.1f})   "
              f"M = {c['M']:+9.1f} (±{q * (X - xx):.1f})")
    for yy in (15.0, 15.5, 17, 20, 24, 28, 40):
        t = K.epaisseur_verticale(yy, e, r) if yy > e else e + r
        c = m.coupe((0, yy), (t, yy), n=400)
        print(f"   coupe horizontale y = {yy:5.1f} (t = {t:6.2f}) : N = {c['N']:+8.2f} (q = {q:.1f})")

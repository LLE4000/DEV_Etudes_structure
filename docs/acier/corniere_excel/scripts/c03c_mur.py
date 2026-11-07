# -*- coding: utf-8 -*-
"""c03c — Support souple : section 3 comparée À BASE ÉGALE (contrainte linéarisée EF = nominale, face au classeur),
et résidu d'équilibre global du cas de base. Pas de 0,5 mm, cheville δ = 0,1 mm."""
import json, os, sys
ICI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ICI)
import numpy as np
import commun as K, ef_coupe as EF
p = K.CAS; e, r, X, Y = p["e"], p["r"], p["X"], p["Y"]; b = p["b_cm"] * 10; q = p["R"] * 1000 / b
xl = K.classeur(**p)
K0 = p["n_anc"] * (xl["N_ELS"] * 1000 / 0.1) / b
m = EF.Coupe(p["larg"], p["haut"], e, r, h=0.5)
out = []
for kw in (None, 1000.0, 300.0, 100.0, 30.0):
    R0 = m.resoudre(q, X, Y, k_mur=kw, k_tige=K0)
    if kw is None:
        print(f"Résidu d'équilibre : |ΣV − q| = {abs(R0['V'] - q):.2e} N/mm ; |T − C| = {abs(R0['T'] - R0['C']):.2e} N/mm")
    c3 = m.coupe((0, e + r), (e, e + r), n=300)
    hs = [m.coupe((0, yy), (K.epaisseur_verticale(yy, e, r), yy), n=150) for yy in np.linspace(24, 27.9, 14)]
    lin_max = max(h["lin_max"] for h in hs + [c3])
    d = dict(k_mur=kw, yc=R0["yc"], T_kN=R0["T"] * b / 1000, S3_lin=c3["lin_max"], S3_max=c3["sn_max"],
             conge_lin_max=lin_max, N=c3["N"], M=abs(c3["M"]))
    out.append(d)
    print(f"mur {'rigide' if kw is None else f'k = {kw:g}':>10s} : y_c = {d['yc']:5.2f} mm ; S3 linéarisée = {d['S3_lin']:6.1f} MPa "
          f"({(d['S3_lin'] / xl['S3']['sigma'] - 1) * 100:+5.1f} % / classeur {xl['S3']['sigma']:.1f}) ; S3 max = {d['S3_max']:6.1f} ; "
          f"max linéarisé dans le congé = {lin_max:6.1f} ({(lin_max / xl['S3']['sigma'] - 1) * 100:+5.1f} %)")
json.dump(out, open(os.path.join(os.path.dirname(ICI), "sorties", "c03c_mur.json"), "w"), indent=1, default=float)

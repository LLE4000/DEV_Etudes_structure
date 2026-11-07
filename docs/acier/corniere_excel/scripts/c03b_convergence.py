# -*- coding: utf-8 -*-
"""c03b — Convergence du maillage sur le cas de base (cheville δ = 0,1 mm, mur rigide) : pas 1 ; 0,5 ; 0,25 mm."""
import json, os, sys, time
ICI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ICI)
import numpy as np
import commun as K, ef_coupe as EF
p = K.CAS; e, r, X, Y = p["e"], p["r"], p["X"], p["Y"]; b = p["b_cm"] * 10; q = p["R"] * 1000 / b
xl = K.classeur(**p)
K0 = p["n_anc"] * (xl["N_ELS"] * 1000 / 0.1) / b
out = []
for h in (1.0, 0.5, 0.25):
    t0 = time.time()
    m = EF.Coupe(p["larg"], p["haut"], e, r, h=h)
    R0 = m.resoudre(q, X, Y, k_tige=K0)
    x, y = m.nodes[:, 0], m.nodes[:, 1]
    arc = np.where(np.abs(np.hypot(x - e - r, y - e - r) - r) < 1e-6)[0]
    i1 = arc[np.argmax(m.s1[arc])]
    S = {s: m.coupe(A, B, n=300)["sn_max"] for s, (A, B) in
         {"S1": ((e + r, 0), (e + r, e)), "S2": ((e, 0), (e, e + r)), "S3": ((0, e + r), (e, e + r)),
          "S4": ((0, e), (e + r, e))}.items()}
    bv = max(m.coupe((xx, 0), (xx, K.epaisseur_verticale(xx, e, r)), n=150)["sn_max"] for xx in np.linspace(24, 27.5, 15))
    bh = max(m.coupe((0, yy), (K.epaisseur_verticale(yy, e, r), yy), n=150)["sn_max"] for yy in np.linspace(24, 27.5, 15))
    d = dict(h=h, noeuds=len(m.nodes), T_kN=R0["T"] * b / 1000, s1_conge=float(m.s1[i1]),
             angle=float(np.degrees(np.arctan2(y[i1] - e - r, x[i1] - e - r))), vm_conge=float(m.vm[arc].max()),
             max_coupes_horizontale_aile=bv, max_coupes_aile_verticale=bh, **S)
    out.append(d)
    print(f"pas {h:4} mm ({len(m.nodes)} nœuds, {time.time() - t0:.0f} s) : T = {d['T_kN']:.2f} kN ; σ1 congé = "
          f"{d['s1_conge']:.1f} MPa ({d['angle']:.1f}°) ; vM congé = {d['vm_conge']:.1f} ; max coupes aile horiz. = {bv:.1f}"
          f" ; aile vert. = {bh:.1f} ; S1 {S['S1']:.1f} S2 {S['S2']:.1f} S3 {S['S3']:.1f} S4 {S['S4']:.1f}")
json.dump(out, open(os.path.join(os.path.dirname(ICI), "sorties", "c03b_convergence.json"), "w"), indent=1)

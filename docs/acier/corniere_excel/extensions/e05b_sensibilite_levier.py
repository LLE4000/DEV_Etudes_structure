import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ef_l
geo = dict(B=120.0, H=120.0, e=15.0, r=13.0, Y=85.0)
q = 42760.0/300.0
pb = ef_l.Probleme(geo, "A", maillage=dict(n_t=6, n_c=8, n_leg=20), solveur="scipy")
# Sensibilité de l effet de levier : demi-largeur d appui de l ecrou (3 mm) ou rondelle Ø30 (15 mm), mur raide ou souple
for br in (3.0, 15.0):
  for km in (1e5, 1e3):
    for k in (300.0, 1e3, 3e3, 1e4, 1e5, 1e7):
        pb.resoudre(q, 75.0, k_anc=k, k_mur=km, b_rond=br)
        r = pb.reactions
        y, f = r["mur_y"], r["mur_f"]
        bas = y < 40
        print(f"b_rond={br:4.0f} k_mur={km:g} k_anc={k:8g} N/mm/mm (={k*100/1e3:8.1f} kN/mm par ancrage)  H={r['H']*300/1e3:6.2f} kN  bas={f[bas].sum()*300/1e3:6.2f} kN à y={(f[bas]*y[bas]).sum()/max(f[bas].sum(),1e-9):5.2f}  haut={f[~bas].sum()*300/1e3:6.2f} kN  it={pb.iterations}")

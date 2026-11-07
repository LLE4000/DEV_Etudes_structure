import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ef_l
geo = dict(B=120.0, H=120.0, e=15.0, r=13.0, Y=85.0)
q = 42760.0/300.0
for m in (dict(n_t=6, n_c=8, n_leg=20), dict(n_t=10, n_c=16, n_leg=40)):
  for dep in ("tout", "bas"):
    t0 = time.perf_counter()
    pb = ef_l.Probleme(geo, "A", maillage=m, solveur="bande")
    pb.resoudre(q, 75.0, k_anc=300.0, k_mur=1e5, b_rond=3.0, depart=dep)
    print(pb.n, dep, "iter", pb.iterations, "résolution %.2f s"%pb.t_resolution, "total %.2f s"%(time.perf_counter()-t0), "H %.3f"%(pb.reactions["H"]*300/1e3))

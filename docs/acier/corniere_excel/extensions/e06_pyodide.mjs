// E06 — le prototype de maillage dans Pyodide 0.29.5 + numpy 2.2.5 (ceux du site), sous Node.
// Lancer : node e06_pyodide.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const ici = dirname(fileURLToPath(import.meta.url));
const { loadPyodide } = await import(join(ici, "pyodide", "pyodide.mjs"));
const t0 = performance.now();
const py = await loadPyodide({ indexURL: join(ici, "pyodide") + "/" });
await py.loadPackage("numpy", { messageCallback: () => {} });
const tCharge = (performance.now() - t0) / 1000;
py.FS.writeFile("/home/pyodide/ef_l.py", readFileSync(join(ici, "ef_l.py"), "utf8"));
const code = `
import sys, time, json
sys.path.insert(0, "/home/pyodide")
import numpy as np, ef_l
geo = dict(B=120.0, H=120.0, e=15.0, r=13.0, Y=85.0)
q = 42760.0/300.0
out = {"python": sys.version.split()[0], "numpy": np.__version__}
rows = []
for nom, m in (("gros", dict(n_t=6, n_c=8, n_leg=20)), ("fin", dict(n_t=10, n_c=16, n_leg=40))):
    t0 = time.perf_counter()
    pb = ef_l.Probleme(geo, "A", maillage=m, solveur="bande")
    t1 = time.perf_counter()
    pb.resoudre(q, 75.0, k_anc=300.0, k_mur=1e5, b_rond=3.0)
    t2 = time.perf_counter()
    s, vm = pb.contraintes_noeuds()
    t3 = time.perf_counter()
    rows.append(dict(maillage=nom, noeuds=int(pb.n), inconnues=int(2*pb.n), demi_bande=int(pb.b),
                     t_maillage_assemblage=t1-t0, t_resolution=t2-t1, iterations=int(pb.iterations),
                     t_contraintes=t3-t2, t_total=t3-t0, H_kN=float(pb.reactions["H"]*300/1e3), vm_max=float(vm.max())))
out["cas"] = rows
json.dumps(out)
`;
const res = JSON.parse(py.runPython(code));
res.t_chargement_pyodide_numpy = tCharge;
console.log(JSON.stringify(res, null, 1));
writeFileSync(join(ici, "e06_sortie.json"), JSON.stringify(res, null, 1));

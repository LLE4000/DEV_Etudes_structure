#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lance toutes les suites de tests du dépôt, une par une, comme des
scripts (elles ne sont pas des modules pytest : chacune imprime son
compte-rendu et rend un code de sortie), puis résume.

    python3 lancer_tests.py            # toutes les suites
    python3 lancer_tests.py assemblages  # celles dont le nom contient « assemblages »

Code de sortie : 0 si toutes les suites sont vertes, 1 sinon.
"""
import glob
import os
import re
import subprocess
import sys
import time

RACINE = os.path.dirname(os.path.abspath(__file__))
os.chdir(RACINE)
filtre = sys.argv[1] if len(sys.argv) > 1 else ""
suites = sorted(f for f in glob.glob("tests/test_*.py") if filtre in f)
resume = []
for f in suites:
    t0 = time.time()
    p = subprocess.run([sys.executable, f], capture_output=True, text=True)
    duree = time.time() - t0
    m = re.findall(r"RÉSULTAT\s*(?:FINAL)?\s*:\s*(\d+) OK, (\d+) échec", p.stdout)
    ok, ko = (int(m[-1][0]), int(m[-1][1])) if m else (0, 1)
    ignore = "ignoré" in p.stdout and not m
    etat = "IGNORÉ" if ignore else ("VERT" if p.returncode == 0 and ko == 0 else "ROUGE")
    resume.append((f, ok, ko, etat, duree))
    print(f"{etat:7s} {f:45s} {ok:4d} OK {ko:3d} échec(s)  {duree:6.1f} s")
    if etat == "ROUGE":
        print("        " + (p.stdout.strip().splitlines() or [""])[-1][:160])
        for l in p.stderr.strip().splitlines()[-3:]:
            print("        " + l[:160])
tot_ok = sum(r[1] for r in resume)
tot_ko = sum(r[2] for r in resume)
rouges = [r[0] for r in resume if r[3] == "ROUGE"]
print(f"\nTOTAL : {tot_ok} OK, {tot_ko} échec(s) — {len(suites) - len(rouges)}/{len(suites)} suites vertes")
sys.exit(1 if rouges else 0)

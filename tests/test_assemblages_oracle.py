# -*- coding: utf-8 -*-
"""Parité aléatoire Python ↔ JavaScript via l'oracle Node.

Tire au hasard AU MOINS 300 jeux d'entrées dans des plages réalistes (tous
les profilés, nuances, boulons, classes, cornières, fixations, options,
efforts de signes variés, les deux modes de calcul, valeurs personnalisées,
files en chaîne ou en nombre), fait calculer le moteur du HTML de
référence sous Node (outils/oracle_double_corniere.js) et compare grandeur
par grandeur avec le moteur Python : écart relatif ≤ 1e-9, égalité stricte
sur les booléens, identifiants, chaînes.

Si ``node`` n'est pas disponible, le test le dit et s'arrête sans échec
(le corrigé des 31 cas reste couvert par test_assemblages_parite.py).

Lancement : python3 tests/test_assemblages_oracle.py [nombre_de_cas]
"""
import json
import os
import random
import shutil
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tests"))

from acier.bibliotheques import DB, PERSO, ORI_P, ORI_S  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import moteur  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.parite import (  # noqa: E402
    instantane, comparer)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


N_CAS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SCRATCH = os.environ.get("TMPDIR", "/tmp")
NODE = shutil.which("node")
HTML = os.path.join(RACINE, "acier", "reference", "assemblage_double_corniere_EC3.html")
ORACLE = os.path.join(RACINE, "outils", "oracle_double_corniere.js")

if not NODE:
    print("  IGNORÉ  node introuvable : test d'oracle non exécuté "
          "(la parité reste prouvée sur les 31 cas du corrigé)")
    print("\nRÉSULTAT : 0 OK, 0 échec(s) — ignoré")
    sys.exit(0)


from alea_double_corniere import tirage  # noqa: E402


rng = random.Random(20260921)
cas = [dict(id=f"alea_{i:04d}", inputs=tirage(rng)) for i in range(N_CAS)]
fichier = os.path.join(SCRATCH, "cas_oracle_double_corniere.json")
with open(fichier, "w", encoding="utf-8") as fh:
    json.dump(cas, fh, ensure_ascii=False)

print(f"=== Oracle Node sur {N_CAS} cas aléatoires ===")
proc = subprocess.run([NODE, ORACLE, HTML, fichier], capture_output=True, text=True,
                      timeout=600)
chk("l'oracle Node s'exécute", proc.returncode == 0, proc.stderr[:300])
attendus = json.loads(proc.stdout) if proc.returncode == 0 else []

total = 0
divergents = []
for c in attendus:
    try:
        R = moteur.compute(c["inputs"])
        n, ecarts = comparer(c["attendu"], instantane(R), tol=1e-9, chemin=c["id"])
    except Exception as e:  # noqa: BLE001
        n, ecarts = 0, [f"{c['id']} : exception {type(e).__name__}: {e}"]
    total += n
    if ecarts:
        divergents.append((c["id"], ecarts[:3]))
chk(f"{len(attendus)} cas comparés, {total} grandeurs, {len(divergents)} cas divergent(s)",
    len(attendus) >= 300 and not divergents,
    " || ".join(f"{i}: {' | '.join(e)}" for i, e in divergents[:3]))
modes = sum(1 for c in cas if c["inputs"]["mode_calc"] != "VÉRIFICATION")
soud = sum(1 for c in cas if "Soudée" in (c["inputs"]["fix_P"], c["inputs"]["fix_S"]))
alertes = sum(1 for c in attendus if c["attendu"]["alertes"])
chk(f"couverture : {modes} cas en prédimensionnement, {soud} avec soudure, "
    f"{alertes} avec au moins une alerte", modes > 30 and soud > 60 and alertes > 100)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:600])
sys.exit(1 if KO else 0)

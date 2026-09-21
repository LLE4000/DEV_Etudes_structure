# -*- coding: utf-8 -*-
"""NON-RÉGRESSION DU CALCUL après la refonte de l'écran et de la note.

Le moteur ne devait pas changer : pour chacun des 31 cas de référence
(``acier/reference/reference_double_corniere.json``, figé avant la refonte,
commit 52b5e4e), on compare au corrigé chaque grandeur nommée par le moteur —
scalaires, Ed / Rd / η / statut de chaque vérification, vérification
dimensionnante, alertes (message, explication, champs, cotes), Tableau 3.3,
efforts par boulon, prédimensionnement — puis les cinq exports texte.

Le test imprime « RÉGRESSION CALCUL : 0 différence » ou liste chaque
différence. Il rougit si on retire une grandeur ou si une valeur bouge.

Lancement : python3 tests/test_assemblages_regression.py
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, parite, texte  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)

print("=== Résultats du moteur : 31 cas de référence ===")
n_tot = 0
diffs = []
for cas in REF["cas"]:
    R = moteur.compute(cas["inputs"])
    n, ecarts = parite.comparer(cas["attendu"], parite.instantane(R), chemin=cas["id"])
    n_tot += n
    diffs.extend(ecarts)
    # le corrigé garde les mêmes vérifications, dans le même ordre
    if [v["key"] for v in cas["attendu"]["verifications"]] != [c.key for c in R.checks]:
        diffs.append(cas["id"] + " : liste des vérifications différente")
chk(f"{len(REF['cas'])} cas, {n_tot} grandeurs comparées", n_tot > 20000 and not diffs, " || ".join(diffs[:5]))

print("\n=== Exports texte ===")
for ident, versions in REF["exports_texte"].items():
    inputs = {c["id"]: c["inputs"] for c in REF["cas"]}[ident] if ident != "defaut" else {}
    R = moteur.compute(inputs)
    for nom, complet in (("essentiel", False), ("complet", True)):
        if nom in versions:
            t = texte.construire_texte(R, complet)
            if t.strip() != versions[nom].strip():
                diffs.append(f"export {ident} {nom} : texte différent")
chk("les exports texte du corrigé sont reproduits", not any(d.startswith("export") for d in diffs))

print()
if diffs:
    print(f"RÉGRESSION CALCUL : {len(diffs)} différence(s)")
    for d in diffs[:40]:
        print("   -", d)
else:
    print("RÉGRESSION CALCUL : 0 différence")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:600])
sys.exit(1 if KO else 0)

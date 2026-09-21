# -*- coding: utf-8 -*-
"""Parité du moteur Python avec le moteur HTML de référence.

Corrigé : acier/reference/reference_double_corniere.json (produit par le
moteur du HTML via l'oracle Node). Chaque garantie rougit si on la retire :
  1. LES 31 CAS : chaque scalaire, chaque vérification (active, Ed, Rd, η,
     ok, vals), chaque alerte (identifiant, bloquante, message, explication,
     champs, cotes, éléments), chaque ligne du Tableau 3.3, les efforts par
     boulon, le prédimensionnement (solution retenue, et toutes les lignes
     pour « defaut » et « predim_* »), le statut, le taux maximal et la
     vérification gouvernante — écart relatif ≤ 1e-9, égalité stricte sur
     les booléens, identifiants, chaînes.
  2. LE BENCHMARK : les 37 valeurs publiées et les 22 valeurs de calcul
     manuel, avec des statuts identiques ; statut global VALIDÉ.
  3. L'INSTANTANÉ a exactement la forme du corrigé (mêmes clés, mêmes
     comptes : 190 scalaires, 38 vérifications, 8 lignes du Tableau 3.3 sur
     le cas par défaut).

Lancement : python3 tests/test_assemblages_parite.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, benchmark  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.parite import (  # noqa: E402
    instantane, comparer)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open(os.path.join(RACINE, "acier", "reference", "reference_double_corniere.json"),
          encoding="utf-8") as fh:
    REF = json.load(fh)

# ================================================================
print("=== 1. Les 31 cas du corrigé ===")
total = 0
for cas in REF["cas"]:
    attendu = cas["attendu"]
    try:
        R = moteur.compute(cas["inputs"])
        obtenu = instantane(R, lignes_predim="lignes" in attendu["predim"])
        n, ecarts = comparer(attendu, obtenu, tol=1e-9, chemin=cas["id"])
    except Exception as e:  # noqa: BLE001 — un plantage est un échec de parité
        n, ecarts = 0, [f"{cas['id']} : exception {type(e).__name__}: {e}"]
    total += n
    chk(f"{cas['id']} — {n} grandeurs, {len(ecarts)} écart(s)", not ecarts,
        " | ".join(ecarts[:4]))
chk(f"au moins 23 000 grandeurs comparées sur les 31 cas ({total})", total >= 23000)

# ================================================================
print("\n=== 2. Le benchmark du module ===")
B = benchmark.run_bench()
BR = REF["benchmark"]
chk("statut global VALIDÉ, identique au corrigé",
    B["statut"] == BR["statut"] == "VALIDÉ" and B["ok"] is True)
chk("écart maximal identique", abs(B["worst"] - BR["worst"]) <= 1e-9 * max(1, BR["worst"]),
    f"{B['worst']} / {BR['worst']}")
n_pub = 0
ec = []
for b, br in zip(B["bench"], BR["bench"]):
    for r, rr in zip(b["rows"], br["rows"]):
        n_pub += 1
        if abs(r["val"] - rr["val"]) > 1e-9 * max(1, abs(rr["val"])) or r["st"] != rr["st"] \
                or abs(r["ec"] - rr["ec"]) > 1e-9:
            ec.append(f"{b['id']} {rr['lab'][:40]} : {r['val']} / {rr['val']} ; {r['st']} / {rr['st']}")
chk(f"37 valeurs publiées : valeurs et statuts identiques ({n_pub})", n_pub == 37 and not ec,
    " | ".join(ec[:3]))
n_man = 0
ec = []
for c, cr in zip(B["valid"], BR["valid"]):
    for r, rr in zip(c["rows"], cr["rows"]):
        n_man += 1
        if abs(r["val"] - rr["val"]) > 1e-9 * max(1, abs(rr["val"])) or r["st"] != rr["st"]:
            ec.append(f"{c['id']} {rr['lab'][:40]} : {r['val']} / {rr['val']}")
chk(f"22 valeurs de calcul manuel : valeurs et statuts identiques ({n_man})",
    n_man == 22 and not ec, " | ".join(ec[:3]))
chk("toutes les lignes manuelles sont OK",
    all(r["st"] == "OK" for c in B["valid"] for r in c["rows"]))

# ================================================================
print("\n=== 3. Forme de l'instantané ===")
R0 = moteur.compute({})
S0 = instantane(R0)
A0 = REF["cas"][0]["attendu"]
chk("mêmes clés de premier niveau que le corrigé", set(S0) == set(A0))
chk("190 scalaires sur le cas par défaut", len(S0["scalaires"]) == 190, str(len(S0["scalaires"])))
chk("38 vérifications, dans l'ordre du corrigé",
    [c["key"] for c in S0["verifications"]] == [c["key"] for c in A0["verifications"]])
chk("8 lignes du Tableau 3.3 sur le cas par défaut", len(S0["tableau_3_3"]) == 8)
chk("56 lignes de prédimensionnement (8 boulons × 7 nombres de rangées)",
    len(S0["predim"]["lignes"]) == 56)
chk("le catalogue des vérifications (libellé, unité, référence, nature, formule, "
    "essentielle) est celui du corrigé",
    [(c.key, c.grp, c.label, c.unit, c.ref, c.nat, c.formula, c.ess) for c in R0.checks]
    == [(c["key"], c["groupe"], c["libelle"], c["unite"], c["reference"], c["nature"],
         c["formule"], c["essentielle"]) for c in REF["catalogue_verifications"]])
chk("valeurs par défaut : les 84 clés du corrigé",
    moteur.defaults() == REF["valeurs_par_defaut"])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Bibliothèques de la famille Acier — source unique et vraisemblance.

  1. SOURCE UNIQUE : acier/donnees/bibliotheques.json est identique à la
     clé « bibliotheques » du corrigé de référence (aucune recopie).
  2. COMPTES : 90 profilés, 6 aciers, 8 boulons, 4 classes, 18 cornières,
     8 ailes, 5 épaisseurs.
  3. CROISEMENT avec le catalogue du dépôt (profiles_test.json) : mêmes
     h, b, tw, tf, r sur les 31 profilés communs ; aire recalculée
     A = 2·b·tf + (h − 2tf)·tw + (4 − π)·r² à moins de 1 % de l'aire
     catalogue.
  4. VRAISEMBLANCE : classes 4.6/5.6/8.8/10.9 (Tableau 3.1), αv (Tableau
     3.4), d0 = d + 1 / + 2 / + 3 (EN 1090-2), aciers du Tableau 3.1 de
     l'EN 1993-1-1 (t ≤ 40 mm).

Lancement : python3 tests/test_assemblages_bibliotheques.py
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.bibliotheques import DB, find, aire_profil, PERSO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)

print("=== 1. Source unique ===")
chk("bibliotheques.json ≡ corrigé de référence", DB == REF["bibliotheques"])

print("\n=== 2. Comptes ===")
for k, n in (("profils", 90), ("aciers", 6), ("boulons", 8), ("classes", 4),
             ("cornieres", 18), ("ailes", 8), ("epais", 5)):
    chk(f"{k} : {n}", len(DB[k]) == n, str(len(DB[k])))
chk("« Personnalisé » n'est pas une entrée de bibliothèque",
    all(x["n"] != PERSO for k in ("profils", "aciers", "cornieres") for x in DB[k]))

print("\n=== 3. Croisement avec profiles_test.json ===")
with open("profiles_test.json", encoding="utf-8") as fh:
    CAT = json.load(fh)
communs = [p for p in DB["profils"] if p["n"] in CAT]
chk("31 profilés communs", len(communs) == 31, str(len(communs)))
div = [p["n"] for p in communs
       if any(abs(float(p[k]) - float(CAT[p["n"]][k])) > 1e-9 for k in ("h", "b", "tw", "tf", "r"))]
chk("dimensions h, b, tw, tf, r identiques au catalogue", not div, str(div))
pires = sorted(((abs(aire_profil(p) / 100 - CAT[p["n"]]["A"]) / CAT[p["n"]]["A"], p["n"])
                for p in communs), reverse=True)
chk("aire recalculée à moins de 1 % de l'aire catalogue (tous les communs)",
    pires[0][0] < 0.01, f"pire : {pires[0][1]} {pires[0][0]*100:.2f} %")

print("\n=== 4. Vraisemblance ===")
cl = {c["n"]: c for c in DB["classes"]}
chk("classes : fyb / fub du Tableau 3.1",
    (cl["4.6"]["fyb"], cl["4.6"]["fub"]) == (240, 400) and (cl["5.6"]["fyb"], cl["5.6"]["fub"]) == (300, 500)
    and (cl["8.8"]["fyb"], cl["8.8"]["fub"]) == (640, 800) and (cl["10.9"]["fyb"], cl["10.9"]["fub"]) == (900, 1000))
chk("αv = 0,6 (4.6, 5.6, 8.8) et 0,5 (10.9)",
    cl["4.6"]["av"] == cl["5.6"]["av"] == cl["8.8"]["av"] == 0.6 and cl["10.9"]["av"] == 0.5)
chk("trous normaux : d0 = d + 1 (M12), d + 2 (M16–M24), d + 3 (M27+)",
    all(b["d0"] - b["d"] == (1 if b["d"] < 16 else 2 if b["d"] <= 24 else 3) for b in DB["boulons"]))
chk("As des boulons ISO 898 (M12 84,3 … M36 817)",
    [b["As"] for b in DB["boulons"]] == [84.3, 157, 245, 303, 353, 459, 561, 817])
chk("aires nominales A = π·d²/4 à 0,1 mm² près",
    all(abs(b["A"] - math.pi * b["d"] ** 2 / 4) < 0.1 for b in DB["boulons"]))
chk("aciers : S235 360, S275 430, S355 510 (Tab. 3.1) ; ligne « fu 490 » présente",
    find(DB["aciers"], "S235")["fu"] == 360 and find(DB["aciers"], "S275")["fu"] == 430
    and find(DB["aciers"], "S355")["fu"] == 510 and find(DB["aciers"], "S355 (fu 490)")["fu"] == 490)
chk("cornières : t < a2 ≤ a1 et rayon > 0",
    all(0 < c["t"] < c["a2"] <= c["a1"] and c["r"] > 0 for c in DB["cornieres"]))
chk("ailes standard croissantes, épaisseurs croissantes",
    [a[0] for a in DB["ailes"]] == sorted(a[0] for a in DB["ailes"]) and DB["epais"] == sorted(DB["epais"]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Bibliothèques de la famille Acier — source unique et vraisemblance.

  1. SOURCE UNIQUE : acier/donnees/bibliotheques.json est identique à la
     clé « bibliotheques » du corrigé de référence (aucune recopie), À UN ÉCART
     NOMMÉ PRÈS : ses aciers n'ont plus de f_y ni de f_u. Ils dépendent de
     l'épaisseur (option a de l'ANB belge, 3.2.1(1)) et se lisent dans la table
     unique du site (acier/materiaux.py) ; le corrigé garde ceux du tableau 3.1
     de l'EN, que la Belgique n'admet pas.
  2. COMPTES : 90 profilés, 6 aciers, 8 boulons, 4 classes, 18 cornières,
     8 ailes, 5 épaisseurs.
  3. CROISEMENT avec le catalogue du dépôt (acier/donnees/profiles.json) : mêmes
     h, b, tw, tf, r sur les 31 profilés communs ; aire recalculée
     A = 2·b·tf + (h − 2tf)·tw + (4 − π)·r² à moins de 1 % de l'aire
     catalogue.
  4. VRAISEMBLANCE : classes 4.6/5.6/8.8/10.9 (Tableau 3.1), αv (Tableau
     3.4), d0 = d + 1 / + 2 / + 3 (EN 1090-2), aciers des tableaux 3.3 ANB et
     3.4 ANB (f_u entre 3 et 100 mm : S235 360, S275 410, S355 470).

Lancement : python3 tests/test_assemblages_bibliotheques.py
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.bibliotheques import DB, find, aire_profil, PERSO, acier  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)

print("=== 1. Source unique ===")
REF_SANS_FY_FU = dict(REF["bibliotheques"], aciers=[{k: v for k, v in a.items() if k not in ("fy", "fu")}
                                                    for a in REF["bibliotheques"]["aciers"]])
chk("bibliotheques.json ≡ corrigé de référence, f_y et f_u des aciers retirés (seul écart)", DB == REF_SANS_FY_FU)
chk("l'écart est bien celui-là : le corrigé porte f_y et f_u, la bibliothèque n'en porte plus",
    all("fy" in a and "fu" in a for a in REF["bibliotheques"]["aciers"])
    and all("fy" not in a and "fu" not in a for a in DB["aciers"]))

print("\n=== 2. Comptes ===")
for k, n in (("profils", 90), ("aciers", 6), ("boulons", 8), ("classes", 4),
             ("cornieres", 18), ("ailes", 8), ("epais", 5)):
    chk(f"{k} : {n}", len(DB[k]) == n, str(len(DB[k])))
chk("« Personnalisé » n'est pas une entrée de bibliothèque",
    all(x["n"] != PERSO for k in ("profils", "aciers", "cornieres") for x in DB[k]))

print("\n=== 3. Croisement avec le catalogue des profilés (acier/donnees/profiles.json) ===")
from acier.profiles import catalogue as CATALOGUE  # noqa: E402
CAT = CATALOGUE.legacy()
communs = [p for p in DB["profils"] if p["n"] in CAT]
chk("au moins 80 des 90 profilés communs au catalogue (31 avant la transcription du 04/11/2026)", len(communs) >= 80, str(len(communs)))
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
chk("aciers à 10 mm (tableau 3.3 ANB) : f_u S235 360, S275 410, S355 470 ; la ligne « fu 490 » suit le S355 belge (470)",
    [acier(n, 10)["fu"] for n in ("S235", "S275", "S355", "S355 (fu 490)")] == [360, 410, 470, 470])
chk("aciers à 20 mm : f_y S235 225, S275 265, S355 345, S460 N/NL 440 (palier à 16 mm)",
    [acier(n, 20)["fy"] for n in ("S235", "S275", "S355", "S460 N/NL")] == [225, 265, 345, 440])
chk("β_w de la bibliothèque (EN 1993-1-8, tableau 4.1) : 0,8 / 0,85 / 0,9 / 1,0",
    [acier(n, 10)["bw"] for n in ("S235", "S275", "S355", "S460 N/NL")] == [0.8, 0.85, 0.9, 1])
chk("cornières : t < a2 ≤ a1 et rayon > 0",
    all(0 < c["t"] < c["a2"] <= c["a1"] and c["r"] > 0 for c in DB["cornieres"]))
chk("ailes standard croissantes, épaisseurs croissantes",
    [a[0] for a in DB["ailes"]] == sorted(a[0] for a in DB["ailes"]) and DB["epais"] == sorted(DB["epais"]))

print("\n=== 5. Textes d'aide des nuances (lot A) : la règle belge, lue dans la table ===")
import importlib  # noqa: E402
from acier.assemblages import materiau as MT  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import entrees as EDC  # noqa: E402
_t = MT.texte_nuances()
chk("le texte dit la règle : f_y et f_u selon l'épaisseur, tableau 3.3 ANB, S355 : fu = 470 de 3 à 100 mm, la ligne "
    "« fu 490 » suit le S355 belge", "tableau 3.3 ANB" in _t and "fu = 470 MPa de 3 à 100 mm" in _t
    and "« S355 (fu 490) » suit le S355 belge (fu = 470 MPa)" in _t and "510" not in _t, _t)
_meth = [importlib.import_module(f"acier.assemblages.{m}.methode") for m in (
    "poutre_poutre.doubles_cornieres", "poutre_poutre.plat_ame", "poutre_poutre.platine_about",
    "poutre_poteau.plat_ame_poteau", "poutre_poteau.doubles_cornieres_poteau", "poutre_poteau.platine_about_poteau")]
chk("les six rubriques Méthode et l'aide de la nuance (doubles cornières) portent ce texte, plus « fu = 510 »",
    all(_t in m.METHODE and "fu = 510" not in m.METHODE and "{NUANCES}" not in m.METHODE for m in _meth)
    and EDC.CHAMPS["nu_P"].get("h") == _t, str([m.__name__ for m in _meth if _t not in m.METHODE]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

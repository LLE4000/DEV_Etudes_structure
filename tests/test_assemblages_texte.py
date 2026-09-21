# -*- coding: utf-8 -*-
"""Export texte : identique aux cinq exports de référence, en version
essentielle et en version complète (mêmes lignes, mêmes nombres).

Lancement : python3 tests/test_assemblages_texte.py
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, texte  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)
cas = {c["id"]: c for c in REF["cas"]}

print("=== Exports texte de référence ===")
n_lignes = 0
for ident, ex in REF["exports_texte"].items():
    R = moteur.compute(cas[ident]["inputs"])
    for version, complet in (("essentiel", False), ("complet", True)):
        attendu = ex[version].split("\n")
        obtenu = texte.construire_texte(R, complet).split("\n")
        n_lignes += len(attendu)
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(attendu, obtenu)) if a != b]
        chk(f"{ident} — {version} : {len(attendu)} lignes identiques",
            len(attendu) == len(obtenu) and not diff,
            f"{len(attendu)}/{len(obtenu)} lignes ; première différence : {diff[:1]}")
chk(f"au moins 500 lignes comparées ({n_lignes})", n_lignes >= 500)
chk("nom de fichier : caractères sûrs", texte.nom_fichier({"id_rep": "P3/S7 é"}, ".txt") == "assemblage_P3_S7_.txt"
    and texte.nom_fichier({}, ".json") == "assemblage_double_corniere.json")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)

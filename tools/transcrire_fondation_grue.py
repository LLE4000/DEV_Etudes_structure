# -*- coding: utf-8 -*-
"""Fondations de grues — relevé des quatre classeurs du bureau vers ``tests/ref_fondation_grue.json``.

Les classeurs (validés par le bureau) ne sont pas versés au dépôt : ce script en relève, cellule par cellule, les
SAISIES et les VALEURS EN CACHE des formules, que ``tests/test_fondation_grue_classeur.py`` oppose ensuite au moteur
``beton/fondation_grue``. Le relevé est une MESURE : il se date et ne se retouche pas à la main.

    /usr/bin/python3.12 tools/transcrire_fondation_grue.py --base BASE.xlsx --gww GWW.xlsx --var4 VAR4.xlsx --var8 VAR8.xlsx

Ce qui est relevé :
- ``Fiche de calcul`` (les quatre classeurs) : toutes les cellules — constantes (saisies, libellés) et formules ;
- BASE et GWW, les quatre grands tableaux ``R+V+M support chassis_CAS A.a / A.b / B.a / B.b`` : les données (B5:B9),
  les résultats (B12:D19), les lignes 22 à 35 des 48 colonnes, et les colonnes B et AW en entier (M(x), 211 lignes) ;
- BASE et GWW, ``R+V+M support chassis_CAS A`` et ``_CAS B`` (les maximums) et ``M semelle sur pieux_CAS A`` ;
- ``Base de données`` (identique dans les quatre).

openpyxl n'est PAS une dépendance du moteur ni des tests : seul ce script le charge.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import openpyxl
from openpyxl.utils import get_column_letter

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORTIE = os.path.join(RACINE, "tests", "ref_fondation_grue.json")
FICHE = "Fiche de calcul"
BASE_DE_DONNEES = "Base de données"
TABLEAUX = ("R+V+M support chassis_CAS A.a", "R+V+M support chassis_CAS A.b",
            "R+V+M support chassis_CAS B.a", "R+V+M support chassis_CAS B.b")
MAXIMUMS = ("R+V+M support chassis_CAS A", "R+V+M support chassis_CAS B")
SEMELLE = "M semelle sur pieux_CAS A"


def _valeur(v):
    """Une valeur de cellule JSON-sérialisable (les erreurs Excel restent des textes « #N/A »)."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        return v
    return str(v)


def _est_formule(v) -> bool:
    return (isinstance(v, str) and v.startswith("=")) or v.__class__.__name__ == "ArrayFormula"


def _releve(sf, sv, plages=None) -> dict:
    """Saisies (constantes) et formules (valeur en cache) d'une feuille ; ``plages`` = filtre (ligne, colonne) → bool."""
    saisies, formules = {}, {}
    for ligne in sf.iter_rows():
        for c in ligne:
            if c.value is None or (plages and not plages(c.row, c.column)):
                continue
            if _est_formule(c.value):
                formules[c.coordinate] = _valeur(sv[c.coordinate].value)
            else:
                saisies[c.coordinate] = _valeur(c.value)
    return {"saisies": saisies, "formules": formules}


def _dans_tableau(ligne: int, colonne: int) -> bool:
    """Ce qu'on garde d'un grand tableau : A:D des lignes 1 à 19, les lignes 22 à 35, les colonnes A, B et AW ensuite."""
    if ligne <= 19:
        return colonne <= 4
    if ligne <= 35:
        return True
    return colonne in (1, 2, 49)


def relever(chemin: str) -> dict:
    wf = openpyxl.load_workbook(chemin, data_only=False)
    wv = openpyxl.load_workbook(chemin, data_only=True)
    with open(chemin, "rb") as f:
        empreinte = hashlib.sha256(f.read()).hexdigest()
    feuilles = {FICHE: _releve(wf[FICHE], wv[FICHE]),
                BASE_DE_DONNEES: _releve(wf[BASE_DE_DONNEES], wv[BASE_DE_DONNEES])}
    for nom in TABLEAUX:
        if nom in wf.sheetnames:
            feuilles[nom] = _releve(wf[nom], wv[nom], _dans_tableau)
    for nom in MAXIMUMS + (SEMELLE,):
        if nom in wf.sheetnames:
            feuilles[nom] = _releve(wf[nom], wv[nom])
    return {"sha256": empreinte, "feuilles_du_classeur": wf.sheetnames, "feuilles": feuilles}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for cle in ("base", "gww", "var4", "var8"):
        p.add_argument(f"--{cle}", required=True, help=f"le classeur {cle.upper()} (.xlsx)")
    p.add_argument("--sortie", default=SORTIE)
    a = p.parse_args()
    classeurs = {cle.upper(): relever(getattr(a, cle)) for cle in ("base", "gww", "var4", "var8")}
    doc = {"_doc": "Fondations de grues : saisies et valeurs en cache des quatre classeurs du bureau (BASE, GWW = "
                   "Greenwood, VAR4, VAR8), relevées par tools/transcrire_fondation_grue.py. Une MESURE : ne pas "
                   "retoucher à la main. Voir docs/beton/FONDATIONS_GRUE_ETUDE.md.",
           "date": datetime.date.today().isoformat(),
           "classeurs": classeurs}
    with open(a.sortie, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=0, sort_keys=True)
        f.write("\n")
    for nom, c in classeurs.items():
        n = sum(len(x["formules"]) for x in c["feuilles"].values())
        s = sum(len(x["saisies"]) for x in c["feuilles"].values())
        print(f"{nom}: {len(c['feuilles'])} feuilles, {s} constantes, {n} formules — sha256 {c['sha256'][:12]}…")
    print("écrit :", os.path.relpath(a.sortie, RACINE), f"({os.path.getsize(a.sortie) // 1024} Ko)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Assemble ANALYSE_CORNIERE_EXCEL.md : source rédigée + annexe A tirée de la sortie de s01 (aucune recopie à la main).

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/assembler.py
"""
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ICI, "ANALYSE_source.md"), encoding="utf-8").read()
annexe = json.load(open(os.path.join(ICI, "sorties", "s01_classeur.json"), encoding="utf-8"))["annexe"]
lignes = ["| Cellule | Formule | Symbole | Valeur |", "|---|---|---|---|"]
for ref, fx, sym, val in annexe:
    lignes.append(f"| `{ref}` | `{fx}` | {sym} | {val} |")
doc = src.replace("<!-- ANNEXE_A -->", "\n".join(lignes))
dest = os.path.join(os.path.dirname(ICI), "ANALYSE_CORNIERE_EXCEL.md")
with open(dest, "w", encoding="utf-8") as fh:
    fh.write(doc)
print(dest, len(doc.splitlines()), "lignes")

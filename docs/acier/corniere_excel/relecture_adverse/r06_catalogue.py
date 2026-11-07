# -*- coding: utf-8 -*-
"""Relecture adverse — 6. Feuille CORNIERES face au catalogue du site. Arguments : xlsx, profiles.json."""
import json
import sys
import openpyxl

wb = openpyxl.load_workbook(sys.argv[1], data_only=True)
ws = wb["CORNIERES"]
P = {p["nom"]: p for p in json.load(open(sys.argv[2]))["profils"] if p["famille"] in ("L", "LI")}
print("cornières du site :", len(P), " égales :", sum(p["famille"] == "L" for p in P.values()))
lignes = []
for i, row in enumerate(ws.iter_rows(min_row=2, max_row=59, values_only=True), start=2):
    lignes.append((i,) + tuple(row[:7]))
print("lignes de données :", sum(1 for l in lignes if l[1] != "NON STANDARD"), " ; dernière :", lignes[-1][1])
vus, doubles, jamais = {}, [], []
for l in lignes:
    if l[1] == "NON STANDARD":
        continue
    if l[1] in vus:
        doubles.append(l[1]); jamais.append(l[0])
    else:
        vus[l[1]] = l
print("désignations :", len(vus), " doubles :", len(doubles), " lignes jamais lues :", jamais)
present, absent, r1, r2, autre, masses = [], [], [], [], [], []
for l in lignes:
    i, nom, b, hh, t, rr, m, g = l
    if nom == "NON STANDARD":
        continue
    a1, a2, tt = nom[1:].split("x")
    cands = [f"L {a1}x{a2}x{tt}", f"L {a2}x{a1}x{tt}"]
    p = next((P[c] for c in cands if c in P), None)
    if p is None:
        absent.append((i, nom)); continue
    present.append((i, nom))
    d = p["dims"]
    same = {d["h"], d["b"]} == {float(b), float(hh)} and d["t"] == t
    if not same:
        print("  dimensions différentes :", nom, d)
    if rr == d["r1"]: r1.append(nom)
    elif rr == d["r2"]: r2.append((nom, rr, d["r1"]))
    else: autre.append((nom, rr, d["r1"], d["r2"]))
    masses.append((m / p["masse"] - 1, nom, m, p["masse"]))
print("présentes :", len(present), "lignes,", len({n for _, n in present}), "désignations")
print("absentes :", len(absent), "lignes,", sorted({n for _, n in absent}), len({n for _, n in absent}))
print("rayon = r1 :", len(r1), " = r2 :", r2, " autre :", autre)
masses.sort()
print(f"masse : de {masses[0][0]:+.2%} ({masses[0][1]} {masses[0][2]} / {masses[0][3]:.3f}) à {masses[-1][0]:+.2%} ({masses[-1][1]})")
print("colonne G :", [l[7] for l in lignes if l[7] is not None])

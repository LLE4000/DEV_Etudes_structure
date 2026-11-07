# -*- coding: utf-8 -*-
"""Comparaison chiffrée du catalogue CORNIERES du classeur avec le catalogue partagé du site
(/home/user/integ/acier/donnees/profiles.json, familles L et LI, EN 10056-1) : dimensions, rayon, masse.

Pour chaque ligne du classeur : la cornière du site de même désignation (ailes dans un ordre ou dans l'autre),
l'écart sur chaque aile, l'épaisseur, le rayon (comparé au congé intérieur r1 ET au rayon d'extrémité r2 du site),
la masse ; et l'effet de l'écart de rayon sur la section 3 du cas de contrôle (pour situer).

Lancement : cd /tmp && /usr/bin/python3.12 -I <ce dossier>/catalogue.py   → <ce dossier>/sorties/catalogue.txt
Lecture seule des deux sources.
"""
import json
import os
from collections import Counter

import openpyxl

ICI = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(ICI, "source", "realex_cornieres.xlsx")
PROFILES = "/home/user/integ/acier/donnees/profiles.json"
SORTIE = os.path.join(ICI, "sorties", "catalogue.txt")


def g(x):
    return f"{x:g}"


def main():
    ws = openpyxl.load_workbook(XLSX, data_only=True)["CORNIERES"]
    entete = [ws.cell(1, j).value for j in range(1, 8)]
    lignes = []
    for i in range(2, 60):
        nom = ws.cell(i, 1).value
        if nom is None:
            continue
        lignes.append({"ligne": i, "nom": nom, "larg": ws.cell(i, 2).value, "haut": ws.cell(i, 3).value,
                       "e": ws.cell(i, 4).value, "r": ws.cell(i, 5).value, "kg": ws.cell(i, 6).value,
                       "acier": ws.cell(i, 7).value})
    site = {p["nom"]: p for p in json.load(open(PROFILES, encoding="utf-8"))["profils"] if p.get("famille") in ("L", "LI")}

    L = []
    p = L.append
    p("CATALOGUE CORNIERES du classeur face à acier/donnees/profiles.json — produit par catalogue.py\n")
    p(f"en-têtes : {entete}")
    p(f"lignes de données : {len(lignes)} (dont 'NON STANDARD' sans dimension) ; cornières du site (L + LI) : {len(site)}")
    p("")
    p("ligne | nom classeur     | larg×haut×e (classeur) | r cl. | kg/m cl. || site (nom)        | h×b×t        | r1   | r2   | kg/m site | Δr vs r1 | Δr vs r2 | Δmasse")
    stats = Counter()
    ecarts_r1, ecarts_m = [], []
    for x in lignes:
        if x["nom"] == "NON STANDARD":
            p(f"{x['ligne']:5d} | {x['nom']:16s} | (pas de dimension : renvoie vers P8!K4:K7)")
            continue
        a, b, t = x["larg"], x["haut"], x["e"]
        cand = [f"L {g(max(a, b))}x{g(min(a, b))}x{g(t)}", f"L {g(a)}x{g(b)}x{g(t)}", f"L {g(b)}x{g(a)}x{g(t)}"]
        s = next((site[c] for c in cand if c in site), None)
        if s is None:
            stats["absente"] += 1
            p(f"{x['ligne']:5d} | {x['nom']:16s} | {g(a):>4}×{g(b):<4}×{g(t):<4}         | {g(x['r']):>5} | {g(x['kg']):>7} || ABSENTE du site")
            continue
        d = s["dims"]
        stats["presente"] += 1
        dims_ok = sorted([a, b]) == sorted([d["h"], d["b"]]) and t == d["t"]
        dr1 = x["r"] - d["r1"]
        dr2 = x["r"] - d["r2"]
        dm = x["kg"] / s["masse"] - 1
        ecarts_r1.append((dr1, x["nom"], x["ligne"]))
        ecarts_m.append((dm, x["nom"], x["ligne"]))
        if dr1 == 0:
            stats["r=r1"] += 1
        elif dr2 == 0:
            stats["r=r2"] += 1
        else:
            stats["r autre"] += 1
        if not dims_ok:
            stats["dims différentes"] += 1
        p(f"{x['ligne']:5d} | {x['nom']:16s} | {g(a):>4}×{g(b):<4}×{g(t):<4}         | {g(x['r']):>5} | {g(x['kg']):>7} || "
          f"{s['nom']:17s} | {g(d['h'])}×{g(d['b'])}×{g(d['t']):<5} | {g(d['r1']):>4} | {g(d['r2']):>4} | {s['masse']:9.3f} | "
          f"{dr1:+6.1f}   | {dr2:+6.1f}   | {dm:+7.2%}")
    p("")
    p(f"bilan : présentes au site {stats['presente']}, absentes {stats['absente']} ; rayon = r1 du site : {stats['r=r1']}, "
      f"= r2 (rayon d'extrémité) : {stats['r=r2']}, autre : {stats['r autre']} ; dimensions différentes : {stats['dims différentes']}")
    if ecarts_m:
        mn = min(ecarts_m)
        mx = max(ecarts_m)
        p(f"écart de masse : de {mn[0]:+.2%} ({mn[1]}, ligne {mn[2]}) à {mx[0]:+.2%} ({mx[1]}, ligne {mx[2]})")
    p("lignes où le rayon du classeur n'est pas le congé r1 du site :")
    for dr1, nom, li in sorted(ecarts_r1):
        if dr1 != 0:
            s = next(v for k, v in site.items() if k.replace(" ", "") == nom or k.replace(" ", "") == nom)
            p(f"   ligne {li} {nom} : r classeur = {g(dr1 + s['dims']['r1'])}, r1 site = {g(s['dims']['r1'])}, r2 site = {g(s['dims']['r2'])} (Δ = {dr1:+g} mm)")

    # doublons de nom : VLOOKUP(…, FALSE) rend toujours la PREMIÈRE ligne
    p("")
    p("doublons de nom (VLOOKUP exact rend la première ligne ; la seconde est INATTEIGNABLE) :")
    noms = [x["nom"] for x in lignes]
    for nom, n in Counter(noms).items():
        if n > 1:
            r = [x for x in lignes if x["nom"] == nom]
            p(f"   {nom:12s} lignes {[x['ligne'] for x in r]} : larg×haut = "
              + " puis ".join(f"{g(x['larg'])}×{g(x['haut'])}" for x in r) + f" → le classeur prend toujours {g(r[0]['larg'])}×{g(r[0]['haut'])}")
    # colonne G
    p("")
    p("colonne G 'ACIER' : " + ", ".join(f"ligne {x['ligne']} = {x['acier']}" for x in lignes if x["acier"] is not None)
      + " — c'est la liste de validation de P8!C3 (nuance), sans lien avec la cornière de la ligne")

    distincts = {x["nom"] for x in lignes if x["nom"] != "NON STANDARD"}
    abs_noms = set()
    for x in lignes:
        if x["nom"] == "NON STANDARD":
            continue
        a, b, t = x["larg"], x["haut"], x["e"]
        if not any(cn in site for cn in (f"L {g(max(a, b))}x{g(min(a, b))}x{g(t)}", f"L {g(a)}x{g(b)}x{g(t)}")):
            abs_noms.add(x["nom"])
    p("")
    p(f"noms distincts (hors NON STANDARD) : {len(distincts)} ; présents au site : {len(distincts - abs_noms)} ; "
      f"absents : {len(abs_noms)} → {', '.join(sorted(abs_noms))}")

    # L150x100x12 : le rayon 6 du classeur face au r1 = 12 du site, sous la charge du cas de contrôle
    def taux_max(a, b, t, r, R=42.76e3, X=75.0, Y=85.0, Lr=300.0, fy=355.0, gm=1.5):
        Ye, Yer, YH = Y - t - r, Y - t, Y - t / 2
        Hh = R * X / YH
        We, Wer = Lr * t ** 2 / 6, Lr * (t + r) ** 2 / 6
        te, ter = R / (Lr * t), R / (Lr * (t + r))
        q = [((R * (X - t - r) / We) ** 2 + 3 * te ** 2) ** 0.5, ((R * (X - t) / Wer) ** 2 + 3 * ter ** 2) ** 0.5,
             ((Hh * Ye / We) ** 2 + 3 * te ** 2) ** 0.5, ((Hh * Yer / Wer) ** 2 + 3 * ter ** 2) ** 0.5]
        return max(q) / (fy / gm)
    p(f"L150x100x12 (150 horizontale × 100 verticale), charge du cas de contrôle : taux max avec r = 6 (classeur) = "
      f"{taux_max(150, 100, 12, 6):.4%} ; avec r = 12 (site, r1) = {taux_max(150, 100, 12, 12):.4%}")

    # effet du rayon sur la section 3 du cas de contrôle (pour situer un écart de rayon)
    R, X, Y, e, Lr, fy, gm = 42.76e3, 75.0, 85.0, 15.0, 300.0, 355.0, 1.5
    p("")
    p("sensibilité du cas de contrôle (L120×120×15) au rayon r — taux de la section 3 :")
    for r in (13.0, 6.5, 0.0):
        Ye, YH = Y - e - r, Y - e / 2
        M3 = R * X * Ye / YH
        s3 = M3 / (Lr * e ** 2 / 6)
        t3 = R / (Lr * e)
        taux = (s3 ** 2 + 3 * t3 ** 2) ** 0.5 / (fy / gm)
        M1 = R * (X - e - r)
        s1 = M1 / (Lr * e ** 2 / 6)
        taux1 = (s1 ** 2 + 3 * t3 ** 2) ** 0.5 / (fy / gm)
        p(f"   r = {g(r):>4} mm : taux section 3 = {taux:.4%} ; taux section 1 = {taux1:.4%}")

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(SORTIE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

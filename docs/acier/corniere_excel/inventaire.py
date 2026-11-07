# -*- coding: utf-8 -*-
"""Inventaire du classeur REALEX « Dimensionnement cornières » : chaque cellule, sa formule, sa valeur en cache,
les cellules qu'elle lit (antécédents) et celles qui la lisent (dépendants : formules, zones de texte du dessin,
listes de validation, mises en forme conditionnelles). En déduit : les cellules de SAISIE (constantes lues par une
formule), les constantes et formules ISOLÉES (lues par rien), les renvois d'une feuille à l'autre.

Lancement (depuis un autre dossier, interpréteur isolé) :
    cd /tmp && /usr/bin/python3.12 -I <ce dossier>/inventaire.py
Écrit <ce dossier>/sorties/inventaire.txt.
"""
import os
import re
import sys
import zipfile
from collections import defaultdict

import openpyxl
from openpyxl.utils import column_index_from_string, get_column_letter

ICI = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(ICI, "source", "realex_cornieres.xlsx")
SORTIE = os.path.join(ICI, "sorties", "inventaire.txt")

# une référence : feuille facultative ('P8 '! ou CORNIERES!), cellule, plage facultative
REF = re.compile(r"(?:'([^']+)'!|([A-Za-z0-9_]+)!)?\$?([A-Z]{1,3})\$?(\d+)(?::\$?([A-Z]{1,3})\$?(\d+))?")


def refs(formule, feuille):
    """Les cellules lues par une formule (une plage est dépliée cellule par cellule)."""
    out = []
    f = re.sub(r'"[^"]*"', '""', formule)          # le contenu des chaînes n'est pas une référence
    for m in REF.finditer(f):
        debut = m.start()
        # écarte les noms de fonction suivis d'un chiffre (aucun ici) et les morceaux d'identifiants
        if debut > 0 and (f[debut - 1].isalnum() or f[debut - 1] == "_"):
            continue
        fe = m.group(1) or m.group(2) or feuille
        c1, l1 = m.group(3), int(m.group(4))
        if m.group(5):
            c2, l2 = m.group(5), int(m.group(6))
            for ci in range(column_index_from_string(c1), column_index_from_string(c2) + 1):
                for li in range(l1, l2 + 1):
                    out.append((fe, f"{get_column_letter(ci)}{li}"))
        else:
            out.append((fe, f"{c1}{l1}"))
    return out


def main():
    wb_f = openpyxl.load_workbook(XLSX, data_only=False)
    wb_v = openpyxl.load_workbook(XLSX, data_only=True)
    cellules = {}                                    # (feuille, adresse) -> (formule ou None, valeur)
    for ws in wb_f.worksheets:
        wv = wb_v[ws.title]
        for row in ws.iter_rows():
            for c in row:
                if c.value is None:
                    continue
                f = c.value if isinstance(c.value, str) and c.value.startswith("=") else None
                cellules[(ws.title, c.coordinate)] = (f, wv[c.coordinate].value)

    dependants = defaultdict(list)
    antecedents = {}
    for (fe, ad), (f, v) in cellules.items():
        if f:
            r = refs(f, fe)
            antecedents[(fe, ad)] = r
            for x in r:
                dependants[x].append(f"formule {fe.strip()}!{ad}")

    # zones de texte du dessin liées à une cellule (P8, drawing1.xml : attribut textlink)
    z = zipfile.ZipFile(XLSX)
    dessin = z.read("xl/drawings/drawing1.xml").decode("utf-8")
    liens = []
    for m in re.finditer(r"<xdr:sp .*?</xdr:sp>", dessin, re.S):
        s = m.group(0)
        nom = re.search(r'name="([^"]+)"', s).group(1)
        tl = re.search(r'textlink="([^"]*)"', s)
        texte = "".join(re.findall(r"<a:t>([^<]*)</a:t>", s))
        if tl and tl.group(1):
            ad = tl.group(1).replace("$", "")
            liens.append((nom, ad, texte))
            dependants[("P8 ", ad)].append(f"zone de texte « {nom} » du dessin")
    # listes de validation (extension x14 de la feuille P8)
    f1 = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    validations = re.findall(r"<x14:dataValidation .*?<xm:f>([^<]+)</xm:f>.*?<xm:sqref>([^<]+)</xm:sqref>", f1, re.S)
    for plage, cible in validations:
        for x in refs(plage, "P8 "):
            dependants[x].append(f"liste de validation de P8!{cible}")
    # mises en forme conditionnelles (références cassées #REF!)
    mfc = re.findall(r'<conditionalFormatting sqref="([^"]+)">(.*?)</conditionalFormatting>', f1, re.S)
    noms = re.findall(r'<definedName name="([^"]+)"[^>]*>([^<]+)</definedName>', z.read("xl/workbook.xml").decode())

    L = []
    p = L.append
    p("INVENTAIRE DU CLASSEUR — produit par inventaire.py (lecture des formules et des valeurs en cache)\n")
    for ws in wb_f.worksheets:
        fe = ws.title
        nf = sum(1 for (a, _), (f, _) in cellules.items() if a == fe and f)
        nc = sum(1 for (a, _), (f, _) in cellules.items() if a == fe and not f)
        p(f"Feuille « {fe} » : {nf} formules, {nc} constantes (textes compris)")
    p("")
    p("== 1. Formules, antécédents, dépendants ==")
    for (fe, ad), (f, v) in cellules.items():
        if not f:
            continue
        ant = antecedents[(fe, ad)]
        ant_txt = ", ".join(sorted({(x[0].strip() + "!" if x[0] != fe else "") + x[1] for x in ant}))
        if len(ant) > 12:
            ant_txt = f"{len(ant)} cellules (plage)"
        dep = dependants.get((fe, ad), [])
        p(f"{fe.strip()}!{ad:5s} {f:70s} => {v!r}")
        p(f"        lit : {ant_txt or '—'}")
        p(f"        lue par : {'; '.join(dep) if dep else 'RIEN (cellule terminale)'}")

    p("")
    p("== 2. Constantes lues par une formule (SAISIES et données) ==")
    for (fe, ad), (f, v) in cellules.items():
        if f or fe == "CORNIERES":
            continue
        dep = dependants.get((fe, ad), [])
        if dep and not isinstance(v, str) or (dep and fe == "P8 " and ad == "B3"):
            p(f"{fe.strip()}!{ad:5s} = {v!r:10}  lue par : {'; '.join(dep)}")
    p("")
    p("== 3. Constantes NUMÉRIQUES lues par rien (hors feuille CORNIERES) ==")
    for (fe, ad), (f, v) in cellules.items():
        if f or fe == "CORNIERES" or isinstance(v, str):
            continue
        if not dependants.get((fe, ad)):
            p(f"{fe.strip()}!{ad:5s} = {v!r}")
    p("")
    p("== 4. Formules lues par rien (ni formule, ni dessin) ==")
    for (fe, ad), (f, v) in cellules.items():
        if f and not dependants.get((fe, ad)):
            p(f"{fe.strip()}!{ad:5s} {f}  => {v!r}")
    p("")
    p("== 5. Renvois d'une feuille à l'autre ==")
    n = 0
    for (fe, ad), ant in antecedents.items():
        autres = sorted({x[0] for x in ant if x[0] != fe})
        if autres:
            n += 1
            p(f"{fe.strip()}!{ad} lit la feuille {', '.join(a.strip() for a in autres)}")
    lit_e10a = [k for k, ant in antecedents.items() if any(x[0] == "E10A BD" for x in ant) and k[0] != "E10A BD"]
    lit_par_e10a = [k for k, ant in antecedents.items() if k[0] == "E10A BD" and any(x[0] != "E10A BD" for x in ant)]
    p(f"Formules hors « E10A BD » qui la lisent : {len(lit_e10a)} ; formules de « E10A BD » qui lisent ailleurs : {len(lit_par_e10a)}")
    p("")
    p("== 6. Zones de texte du dessin (P8) liées à une cellule ==")
    for nom, ad, texte in liens:
        f, v = cellules.get(("P8 ", ad), (None, None))
        p(f"{nom:14s} -> P8!{ad:4s} affiche {texte!r:14}  (cellule : {'VIDE' if v is None else repr(v)})")
    p("")
    p("== 7. Listes de validation, noms définis, mises en forme conditionnelles ==")
    for plage, cible in validations:
        p(f"validation de P8!{cible} : liste {plage}")
    for nom, cible in noms:
        p(f"nom défini {nom} = {cible}")
    for sq, corps in mfc:
        formules = re.findall(r"<formula>([^<]+)</formula>", corps)
        p(f"MFC {sq:18s} : {formules}")
    # usage du nom ListeCorniere
    usage = [k for k, (f, v) in cellules.items() if f and "ListeCorniere" in f]
    p(f"formules qui utilisent ListeCorniere : {len(usage)}")
    noms_cat = [cellules[("CORNIERES", f"A{i}")][1] for i in range(2, 60)]
    p(f"noms de la liste CORNIERES!A2:A59 : {len(noms_cat)}, distincts : {len(set(noms_cat))}")
    dbl = sorted({x for x in noms_cat if noms_cat.count(x) > 1})
    p(f"noms en double ({len(dbl)}) : {', '.join(dbl)}")
    hors_nom = [x for x in noms_cat[55:]]
    p(f"lignes de la liste hors du nom ListeCorniere (A57:A59) : {hors_nom}")

    # E10A BD : les colonnes recalculées
    p("")
    p("== 8. Feuille E10A BD : recalcul des colonnes ==")
    e = {ad: v for (fe, ad), (f, v) in cellules.items() if fe == "E10A BD"}
    larg, q = e["D3"], e["F2"]
    p(f"D3 = G3 = {larg} (diviseur commun) ; F2 = {q} (multiplie la surface)")
    p("volée | masse B (kg) | poids C=B/100 (kN) | D=C/2/D3 (kN/m) | surface E (m²) | F=E·F2 (kN) | G=F/2/G3 (kN/m) | D+G (kN/m) | (C+F)/2 (kN)")
    for i in range(4, 12):
        B, C, D, E, F, G = (e[f"{c}{i}"] for c in "BCDEFG")
        assert abs(C - B / 100) < 1e-9 and abs(D - C / 2 / larg) < 1e-9 and abs(F - E * q) < 1e-9 and abs(G - F / 2 / larg) < 1e-9
        p(f"{e[f'A{i}']:>5} | {B:12.2f} | {C:18.4f} | {D:15.4f} | {E:14.2f} | {F:11.2f} | {G:15.4f} | {D + G:10.4f} | {(C + F) / 2:12.4f}")
    p(f"cellules isolées : I10 = {e['I10']}, I13 = {e['I13']}, H15 = {e['H15']}, I15 = {e['I15']}, H16 = {e['H16']}")
    p(f"  I13/I10 - 1 = {e['I13'] / e['I10'] - 1:+.4%} ; I15/H16 - 1 = {e['I15'] / e['H16'] - 1:+.4%}")
    p(f"  H15 = C10/2 + 8,7 : C10/2 = {e['C10'] / 2:.4f} ; F10/2 = {e['F10'] / 2:.4f}")

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(SORTIE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())

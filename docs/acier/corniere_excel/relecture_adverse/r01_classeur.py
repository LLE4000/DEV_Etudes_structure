# -*- coding: utf-8 -*-
"""Relecture adverse — 1. Le classeur lui-même : formules, cache, formats, mises en forme, validations, dessin.
Script indépendant (n'importe rien des scripts de l'analyse relue). Lancer depuis /tmp :
    /usr/bin/python3.12 -I -B <ce fichier> <chemin du xlsx>
"""
import math
import re
import sys
import zipfile
import openpyxl

XLSX = sys.argv[1]
wb_f = openpyxl.load_workbook(XLSX, data_only=False)
wb_v = openpyxl.load_workbook(XLSX, data_only=True)
print("Feuilles :", [repr(n) for n in wb_f.sheetnames])

# ---------------------------------------------------------------- comptes
for nom in wb_f.sheetnames:
    ws, wv = wb_f[nom], wb_v[nom]
    nf = nt = nc = 0
    for row in ws.iter_rows():
        for c in row:
            if c.value is None:
                continue
            if isinstance(c.value, str) and c.value.startswith("="):
                v = wv[c.coordinate].value
                if isinstance(v, str):
                    nt += 1
                else:
                    nf += 1
            else:
                nc += 1
    print(f"{nom!r}: formules numériques {nf}, formules texte {nt}, total formules {nf+nt}, constantes {nc}, dims {ws.dimensions}")

ws = wb_f[wb_f.sheetnames[0]]
wv = wb_v[wb_v.sheetnames[0]]
print("Zone d'impression :", ws.print_area, " sauts de ligne :", [b.id for b in ws.row_breaks.brk])
print("Noms définis :", [(n, wb_f.defined_names[n].attr_text) for n in wb_f.defined_names])

# formats de nombre utiles
for a in ["C3", "C8", "C12", "C13", "C14", "C16", "C17", "C25", "C76", "G85", "G86", "C82", "C83", "C88", "C90", "D3"]:
    print(f"  format {a}: {ws[a].number_format!r}  valeur cache {wv[a].value!r}")

# mises en forme conditionnelles
print("Mises en forme conditionnelles :")
for rng, regles in ws.conditional_formatting._cf_rules.items():
    for r in regles:
        print("  ", rng.sqref, r.type, r.operator, r.formula)

# validations
print("Validations :")
for dv in ws.data_validations.dataValidation:
    print("  ", dv.sqref, dv.type, dv.formula1)

# ---------------------------------------------------------------- qui lit quoi
formules = {}
for nom in wb_f.sheetnames:
    for row in wb_f[nom].iter_rows():
        for c in row:
            if isinstance(c.value, str) and c.value.startswith("="):
                formules[(nom, c.coordinate)] = c.value
lecteurs = {}
for (nom, coord), f in formules.items():
    for ref in re.findall(r"\$?([A-Z]{1,2})\$?(\d+)", f.split("!")[-1] if "CORNIERES" in f else f):
        lecteurs.setdefault(ref[0] + ref[1], set()).add(coord)
for cible in ["A20", "A21", "C53", "C54", "C33", "C43", "C25", "C17", "C76", "C14", "C5", "C4", "L22", "C71", "G85", "G86"]:
    print(f"  {cible} lu par : {sorted(lecteurs.get(cible, []))}")
# feuille E10A BD lue depuis une autre feuille ?
print("Formules qui citent 'E10A' :", [k for k, f in formules.items() if "E10A" in f])
print("Formules qui citent CORNIERES :", [k for k, f in formules.items() if "CORNIERES" in f])

# ---------------------------------------------------------------- mon modèle en symboles, face au cache
def P8(fy=355, L_cm=30, gM=1.5, R=42.76, X=75, Xj=0, Y=85, Yj=0, e=15, r=13, larg=120, haut=120,
       Rulu=68.9, n=3, d=16, jeu_cm=1):
    L = L_cm * 10.0
    out = {}
    out["C19"] = X - r - e
    out["C20"] = X - e
    out["A20"] = larg - X
    out["A21"] = haut - Y
    out["C23"] = Y - e - r
    out["C24"] = Y - e
    out["C25"] = Y - e / 2
    sadm = fy / gM
    tadm = fy / gM / math.sqrt(3)
    W1 = L * e ** 2 / 6
    W2 = L * (e + r) ** 2 / 6
    Mmax = (X + Xj) / 1000 * R
    M = {1: R * (X - r - e + Xj) / 1000, 2: R * (X - e + Xj) / 1000,
         3: (Y - e - r + Yj) / (Y - e / 2 + Yj) * Mmax, 4: (Y - e + Yj) / (Y - e / 2 + Yj) * Mmax}
    W = {1: W1, 2: W2, 3: W1, 4: W2}
    t = {1: e, 2: e + r, 3: e, 4: e + r}
    cel = {1: ("C35", "G31", "G34", "G37", "I37"), 2: ("C45", "G41", "G44", "G47", "I47"),
           3: ("C57", "G51", "G54", "G57", "I57"), 4: ("C65", "G61", "G64", "G67", "I67")}
    for k in (1, 2, 3, 4):
        s = M[k] * 1e6 / W[k]
        ta = R * 1000 / (L * t[k])
        seq = math.sqrt(s ** 2 + 3 * ta ** 2)
        cM, cs, ct, ce, ci = cel[k]
        out[cM], out[cs], out[ct], out[ce], out[ci] = M[k], s, ta, seq, seq / sadm
    out["C55"] = out["C63"] = out["C78"] = Mmax
    out["C56"] = (Y - e - r + Yj) / (Y - e / 2 + Yj)
    out["C64"] = (Y - e + Yj) / (Y - e / 2 + Yj)
    out["H31"] = sadm
    out["H34"] = tadm
    out["C32"] = W1
    out["C42"] = W2
    out["C82"] = Mmax / ((Y / 10) * 0.01 - jeu_cm * 0.01) / n
    out["C83"] = R / n
    out["C85"] = Rulu / R * out["C82"]
    out["C86"] = Rulu / R * out["C83"]
    out["C88"] = out["C83"] * 1e3 / (d * e)
    out["C90"] = out["C83"] * 1000 / (2 * (haut - Y) * e)
    out["D3"] = max(out["I37"], out["I47"], out["I57"], out["I67"])
    out["I31"] = out["G31"] / sadm
    out["I51"] = out["G51"] / sadm
    out["I41"] = out["G41"] / sadm
    out["I61"] = out["G61"] / sadm
    out["I34"] = out["G34"] / tadm
    out["I44"] = out["G44"] / tadm
    out["I54"] = out["G54"] / tadm
    out["I64"] = out["G64"] / tadm
    return out

mod = P8()
ecart = 0.0
for k, v in mod.items():
    cache = wv[k].value
    if isinstance(cache, (int, float)) and cache != 0:
        ecart = max(ecart, abs(v - cache) / abs(cache))
    elif cache != v:
        print("  écart", k, v, cache)
print(f"Mon modèle en symboles : {len(mod)} cellules comparées au cache, écart relatif max {ecart:.2e}")
for k in ["C35", "C45", "C57", "C65", "D3", "I37", "I47", "I57", "I67", "G37", "G47", "G57", "G67", "C82", "C85"]:
    print(f"  {k} = {mod[k]:.6f}   (cache {wv[k].value})")

# ---------------------------------------------------------------- le dessin : zones de texte liées
z = zipfile.ZipFile(XLSX)
dr = z.read("xl/drawings/drawing1.xml").decode("utf8")
liens = re.findall(r'<xdr:sp macro="[^"]*" textlink="([^"]*)"', dr)
print("Zones de texte liées :", len(liens), liens)
nb_sp = len(re.findall(r"<xdr:sp ", dr))
print("Formes <xdr:sp> :", nb_sp)
textes_fixes = re.findall(r"<a:t>([^<]*)</a:t>", dr)
print("Textes <a:t> du dessin :", textes_fixes)

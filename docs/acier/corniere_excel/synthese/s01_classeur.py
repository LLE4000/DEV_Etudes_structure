# -*- coding: utf-8 -*-
"""S01 — Le classeur relu : inventaire, reproduction du cas de contrôle cellule par cellule, parité aléatoire.

1. Compte des formules et constantes par feuille ; liens du dessin ; validations ; nom défini ; mises en forme
   conditionnelles ; feuille E10A BD lue ou non.
2. Formules de P8 recalculées par l'interpréteur (s_commun.Classeur) : écart au cache.
3. Méthode en symboles (s_commun.historique) face au cache, cellule par cellule.
4. 600 saisies tirées au hasard (graine fixe) : symboles face aux formules réévaluées.
5. Tableau de l'annexe A : cellule, formule, symbole, valeur.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/s01_classeur.py
"""
import math
import os
import random
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s_commun as K  # noqa: E402

S = K.Sortie("s01_classeur")
cl = K.Classeur()
P8 = K.FEUILLE

# ------------------------------------------------------------------ 1. structure
S("# S01 — le classeur relu")
S("")
S("## 1. Structure")
for nom in cl.noms:
    nf = [k for k in cl.formules if k[0] == nom]
    nn = [k for k in nf if K.Classeur.traduire(cl.formules[k]) is not None]
    nc = [k for k in cl.constantes if k[0] == nom]
    S(f"feuille « {nom} » : {len(nf)} formules ({len(nn)} numériques, {len(nf) - len(nn)} de texte), {len(nc)} constantes")
    S.v(f"formules_{nom.strip()}", len(nf))
    S.v(f"formules_num_{nom.strip()}", len(nn))
    S.v(f"constantes_{nom.strip()}", len(nc))
lecteurs_e10 = [k for k, f in cl.formules.items() if "E10A" in f and k[0] != "E10A BD"]
S(f"formules hors de « E10A BD » qui la lisent : {len(lecteurs_e10)}")
S.v("lecteurs_E10A", len(lecteurs_e10))

z = zipfile.ZipFile(K.XLSX)
dess = z.read("xl/drawings/drawing1.xml").decode()
liens = re.findall(r'textlink="([^"]*)"', dess)
liens_ok = [x.replace("$", "") for x in liens if x]
S(f"zones de texte du dessin liées à une cellule : {len(liens_ok)} → {', '.join(liens_ok)} ; sans lien : {liens.count('')}")
S.v("liens_dessin", liens_ok)

import openpyxl  # noqa: E402
wb = openpyxl.load_workbook(K.XLSX)
ws = wb[P8]
feuille1 = z.read("xl/worksheets/sheet1.xml").decode()
for bloc in re.findall(r"<x14:dataValidation .*?</x14:dataValidation>", feuille1, re.S):
    f1 = re.search(r"<xm:f>(.*?)</xm:f>", bloc).group(1)
    sq = re.search(r"<xm:sqref>(.*?)</xm:sqref>", bloc).group(1)
    S(f"liste de validation de {sq} : {f1}")
for nom_def in (wb.defined_names.keys() if hasattr(wb.defined_names, "keys") else []):
    S(f"nom défini {nom_def} : {wb.defined_names[nom_def].attr_text}")
S(f"zone d'impression : {ws.print_area} ; sauts de page après les lignes {[br.id for br in ws.row_breaks.brk]}")
for ref in ("C8", "C25", "C76", "C12", "C13", "C14", "C16", "C17", "C3"):
    S(f"format de {ref} : {ws[ref].number_format!r}")
S(f"libellé B43 : {ws['B43'].value!r} (à côté de C43 = $C$20, soit Xe+r)")
S(f"A12 = {ws['A12'].value!r} ; B12 = {ws['B12'].value!r} ; C12 = {ws['C12'].value!r} (la mention de page est dans le libellé A12)")
_fx = {c.coordinate: c.value for row in ws.iter_rows() for c in row if isinstance(c.value, str) and c.value.startswith("=")}
for _cib in ("C4", "C5", "A20", "A21"):
    _pat = re.compile(r"(?<![A-Z$])\$?" + _cib[0] + r"\$?" + _cib[1:] + r"(?!\d)")
    S(f"formules de P8 qui lisent {_cib} : {sorted(k for k, v in _fx.items() if _pat.search(v)) or 'aucune'}")
S(f"C90 = {_fx['C90']} (recalcule haut. − Y, ne lit pas A21)")
ref_casses = []
for plage in ws.conditional_formatting:
    for regle in plage.rules:
        f = " ".join(regle.formula or [])
        S(f"mise en forme conditionnelle {plage.sqref} : {regle.type} {regle.operator or ''} {f}")
        if "#REF!" in f:
            ref_casses.append(str(plage.sqref))
S.v("mfc_ref_cassees", ref_casses)

# ------------------------------------------------------------------ 2. interpréteur face au cache
S("")
S("## 2. Formules de P8 recalculées telles qu'elles sont écrites")
ev = cl.evaluer()
ecarts = []
for ref, val in ev.items():
    cache = cl.cache[(P8, ref)]
    if isinstance(cache, (int, float)):
        ecarts.append((abs(val - cache) / max(abs(cache), 1e-30), ref))
ecarts.sort(reverse=True)
S(f"{len(ecarts)} formules numériques de P8 recalculées ; plus grand écart relatif au cache {ecarts[0][0]:.1e} ({ecarts[0][1]})")
S.v("p8_formules_num", len(ecarts))
S.v("p8_ecart_interpreteur", ecarts[0][0])

# ------------------------------------------------------------------ 3. symboles face au cache
S("")
S("## 3. Méthode réécrite en symboles face aux valeurs enregistrées (cas de contrôle)")
mod = K.cellules_modele(K.CAS)
compte, pire = 0, (0.0, "")
for ref, val in mod.items():
    cache = cl.cache.get((P8, ref))
    if not isinstance(cache, (int, float)):
        continue
    d = abs(val - cache) / max(abs(cache), 1e-30)
    compte += 1
    if d > pire[0]:
        pire = (d, ref)
non_couvertes = sorted(set(ev) - set(mod))
S(f"{compte} cellules comparées ; plus grand écart relatif {pire[0]:.1e} ({pire[1] or '—'}) ; formules numériques hors modèle : {', '.join(non_couvertes)}")
S.v("cellules_comparees", compte)
S.v("ecart_max_modele", pire[0])
h = K.historique(K.CAS)
for i, nom in enumerate(("M1", "M2", "M3", "M4")):
    S(f"{nom} = {h['M'][i] / 1e6:.6f} kN·m (attendu par l'ingénieur {K.ATTENDU[nom]}) ; taux {100 * h['taux'][i]:.4f} %")
S(f"taux max {100 * h['taux_max']:.4f} % en section {h['section_max']} (D3 en cache : {100 * cl.cache[(P8, 'D3')]:.4f} %)")

# ------------------------------------------------------------------ 4. parité aléatoire
S("")
S("## 4. Parité sur des saisies tirées au hasard")
rnd = random.Random(20261107)
noms_cat = []
for row in cl.cat:
    if row[0] and row[0] not in noms_cat and row[0] != "NON STANDARD":
        noms_cat.append(row[0])
premiere = {}
for row in cl.cat:
    if row[0] not in premiere:
        premiere[row[0]] = row
n_cas, n_cmp, pire = 0, 0, (0.0, None)
for k in range(600):
    if rnd.random() < 0.7:
        nom = rnd.choice(noms_cat)
        _, larg, haut, e, r = premiere[nom]
    else:
        nom = None
        e = rnd.choice([5, 6, 8, 10, 12, 15, 20])
        r = rnd.choice([4, 6, 8, 10, 13, 16])
        larg, haut = rnd.choice([60, 80, 100, 120, 150, 200]), rnd.choice([60, 80, 100, 120, 150, 200])
    p = dict(K.CAS, larg=float(larg), haut=float(haut), e=float(e), r=float(r))
    p["fy"] = rnd.choice([235.0, 355.0])
    p["gM"] = rnd.choice([1.0, 1.1, 1.5])
    p["L"] = 10.0 * rnd.choice([10, 20, 25, 30, 40, 100])
    p["R"] = rnd.uniform(1, 120) * 1e3
    p["X"] = rnd.uniform(e + r + 1, larg)
    p["Xj"] = rnd.choice([0.0, 5.0, 10.0, 20.0, rnd.uniform(0, 30)])
    p["Y"] = rnd.uniform(e + r + 5, haut)
    p["Yj"] = rnd.choice([0.0, 5.0, 10.0, 20.0, rnd.uniform(0, 30)])
    p["n"] = rnd.choice([1, 2, 3, 4])
    p["d"] = rnd.choice([10.0, 12.0, 16.0, 20.0])
    p["jeu"] = rnd.choice([0.0, 5.0, 10.0, 15.0])
    p["R_ELU"] = p["R"] * rnd.uniform(1.3, 1.7)
    if p["Y"] - p["jeu"] < 1:
        continue
    ev = cl.evaluer(K.saisies_classeur(p, nom))
    mod = K.cellules_modele(p)
    n_cas += 1
    for ref, val in mod.items():
        if ref not in ev:
            continue
        d = abs(val - ev[ref]) / max(abs(ev[ref]), 1e-12)
        n_cmp += 1
        if d > pire[0]:
            pire = (d, ref)
S(f"{n_cas} saisies (cornière du catalogue lue sur sa première ligne, comme RECHERCHEV, ou « NON STANDARD » ; f_y, γ, L, "
  f"R, X, Xj, Y, Yj, n, d, jeu, R_ELU variés) ; {n_cmp} comparaisons ; plus grand écart relatif {pire[0]:.1e} ({pire[1]})")
S.v("parite_cas", n_cas)
S.v("parite_comparaisons", n_cmp)
S.v("parite_ecart", pire[0])

# ------------------------------------------------------------------ 5. annexe A
SYMB = {
    "D3": "taux max = MAX(I37 ; I47 ; I57 ; I67)",
    "C4": "larg. (aile horizontale) — catalogue col. 2 ou K4", "C5": "haut. (aile verticale) — col. 3 ou K5",
    "C6": "e — col. 4 ou K6", "C7": "r (congé) — col. 5 ou K7",
    "C19": "Xe = X − r − e", "A20": "larg. − X (dessin)", "C20": "Xe+r = X − e", "A21": "haut. − Y (dessin)",
    "L22": "3,7 × 0,19 × 1 230 (isolée, sens inconnu)",
    "C23": "Ye = Y − e − r", "C24": "Ye+r = Y − e", "C25": "YH = Y − e/2",
    "C31": "t₁ = e", "C32": "W₁ = L·e²/6", "C33": "Xe", "C34": "Xj", "C35": "M1 = R·(Xe + Xj)",
    "F31": "copie de M1", "F32": "copie de W₁", "F34": "copie de R",
    "G31": "σ₁ = M1/W₁", "H31": "σ_adm = f_y/γ", "I31": "σ₁/σ_adm",
    "G34": "τ₁ = R/(L·e)", "H34": "τ_adm = f_y/(γ·√3)", "I34": "τ₁/τ_adm",
    "G37": "σ_éq,1 = √(σ₁² + 3τ₁²)", "H37": "σ_adm", "I37": "taux 1",
    "C41": "t₂ = e + r", "C42": "W₂ = L·(e+r)²/6", "C43": "Xe+r (libellé B43 « Xe = »)", "C44": "Xj",
    "C45": "M2 = R·(Xe+r + Xj)", "F41": "copie de M2", "F42": "copie de W₂", "F44": "copie de R",
    "G41": "σ₂ = M2/W₂", "H41": "σ_adm", "I41": "σ₂/σ_adm", "G44": "τ₂ = R/(L·(e+r))", "H44": "τ_adm",
    "I44": "τ₂/τ_adm", "G47": "σ_éq,2", "H47": "σ_adm", "I47": "taux 2",
    "C51": "t₃ = e", "C52": "W₃ = L·e²/6", "C53": "X (affiché, non relu)", "C54": "Xj (affiché, non relu)",
    "C55": "Mmax = R·(X + Xj)", "C56": "ρ₃ = (Ye + Yj)/(YH + Yj)", "C57": "M3 = ρ₃·Mmax",
    "F51": "copie de M3", "F52": "copie de W₃", "F54": "copie de R",
    "G51": "σ₃ = M3/W₃", "H51": "σ_adm", "I51": "σ₃/σ_adm", "G54": "τ₃ = R/(L·e)", "H54": "τ_adm",
    "I54": "τ₃/τ_adm", "G57": "σ_éq,3", "H57": "σ_adm", "I57": "taux 3",
    "C61": "t₄ = e + r", "C62": "W₄ = L·(e+r)²/6", "C63": "Mmax (2ᵉ copie)", "C64": "ρ₄ = (Ye+r + Yj)/(YH + Yj)",
    "C65": "M4 = ρ₄·Mmax", "F61": "copie de M4", "F62": "copie de W₄", "F64": "copie de R",
    "G61": "σ₄ = M4/W₄", "H61": "σ_adm", "I61": "σ₄/σ_adm", "G64": "τ₄ = R/(L·(e+r))", "H64": "τ_adm",
    "I64": "τ₄/τ_adm", "G67": "σ_éq,4", "H67": "σ_adm", "I67": "taux 4",
    "C70": "R_ELS", "D73": "L en cm (affichage)", "C75": "Y en cm", "C78": "Mmax (3ᵉ copie)",
    "C82": "N_ELS = Mmax/((Y − jeu)·n)", "C83": "V_ELS = R/n", "C85": "N_ELU = N_ELS·R_ELU/R_ELS",
    "C86": "V_ELU = V_ELS·R_ELU/R_ELS", "C88": "pression diamétrale V_ELS/(d·e)",
    "C90": "arrachement V_ELS/(2·(haut. − Y)·e)",
}
UNITE = {"F31": "kN·m", "F41": "kN·m", "F51": "kN·m", "F61": "kN·m", "F34": "kN", "F44": "kN", "F54": "kN", "F64": "kN",
         "D3": "%", "C35": "kN·m", "C45": "kN·m", "C55": "kN·m", "C57": "kN·m", "C63": "kN·m", "C65": "kN·m",
         "C78": "kN·m", "C82": "kN", "C83": "kN", "C85": "kN", "C86": "kN", "C70": "kN", "C88": "N/mm²", "C90": "N/mm²",
         "C75": "cm", "D73": "cm"}


def ordre(ref):
    m = re.match(r"([A-Z]+)(\d+)", ref)
    return (int(m.group(2)), m.group(1))


S("")
S("## 5. Annexe A — formules numériques de P8 (cellule | formule | symbole | valeur en cache)")
lignes_annexe = []
for (nom, ref) in sorted([k for k in cl.formules if k[0] == P8], key=lambda k: ordre(k[1])):
    fx = cl.formules[(nom, ref)]
    if K.Classeur.traduire(fx) is None:
        continue
    val = cl.cache[(nom, ref)]
    if ref.startswith("I") or ref == "D3":
        txt = K.pc(val, 2)
    elif ref in ("C56", "C64"):
        txt = K.fr(val, 5)
    elif abs(val) >= 1000:
        txt = K.fr(val, 0)
    else:
        txt = K.fr(val, 5 if UNITE.get(ref) == "kN·m" else 3).rstrip("0").rstrip(",")
    u = UNITE.get(ref, "")
    lignes_annexe.append((ref, fx.replace("|", "¦"), SYMB.get(ref, "?"), txt + ("" if u in ("", "%") else " " + u)))
    S(f"| {ref} | `{fx}` | {SYMB.get(ref, '?')} | {lignes_annexe[-1][3]} |")
S.v("annexe", lignes_annexe)
S.ecrire()

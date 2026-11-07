# -*- coding: utf-8 -*-
"""Contrôle : chaque nombre cité dans ANALYSE_CORNIERE_EXCEL.md doit se retrouver dans une sortie de script.

Un nombre du document est « couvert » si une valeur des sorties, arrondie à la précision écrite dans le document,
lui est égale — telle quelle, ou ×100 (fraction écrite en pourcentage). Seules les sorties IMPRIMÉES (.txt) servent de
référence, pas les fichiers de courbes. Sont ignorés : les entiers inférieurs à 10, les références de cellules, de normes,
de fichiers et les dates. Les nombres non couverts sont listés : chacun doit être justifié ou corrigé.

Lancer : cd /tmp && /usr/bin/python3.12 -I -B <ce dossier>/verifier_chiffres.py
"""
import glob
import os
import re

ICI = os.path.dirname(os.path.abspath(__file__))
RAC = os.path.dirname(ICI)
doc = open(os.path.join(RAC, "ANALYSE_CORNIERE_EXCEL.md"), encoding="utf-8").read()
sources = glob.glob(os.path.join(ICI, "sorties", "*.txt"))       # sorties imprimées seulement (pas les courbes)
sources += [os.path.join(RAC, "sorties", "c03c_mur.json"), os.path.join(RAC, "sorties", "c03b_convergence.txt"),
            os.path.join(RAC, "extensions", "e06_sortie.json"), os.path.join(RAC, "extensions", "e07_sortie.txt")]
corpus = ""
for f in sources:
    if f.endswith("s07_figure.json"):
        continue
    corpus += open(f, encoding="utf-8").read() + "\n"
vals = set()
fr_corpus = re.sub(r"(?<=\d)[\u202f ](?=\d{3}\b)", "", corpus)          # milliers à la française
fr_corpus = re.sub(r"(?<=\d),(?=\d)", ".", fr_corpus)                    # virgule décimale
for txt in (corpus.replace(",", " "), fr_corpus):
    for m in re.finditer(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?", txt):
        try:
            vals.add(abs(float(m.group(0))))
        except ValueError:
            pass
vals = sorted(vals)

texte = doc
texte = re.sub(r"`[^`]*`", " ", texte)                      # cellules, formules, fichiers
texte = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", texte)        # images
texte = re.sub(r"\[s0\d[^\]]*\]", " ", texte)               # renvois aux scripts
texte = re.sub(r"EN \d+(?:-\d+)*(?: \d+\.\d+(?:\.\d+)*)?", " ", texte)
texte = re.sub(r"\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{2}", " ", texte)
texte = re.sub(r"L\d+(?:[x×]\d+(?:[.,]\d+)?)+", " ", texte)
texte = re.sub(r"§\d+(?:\.\d+)*", " ", texte)
texte = re.sub(r"\b[A-Z]{1,2}\d+\b", " ", texte)              # références de cellules hors backticks
texte = re.sub(r"\d+ᵉ|10⁻¹⁶|10⁻¹⁵|10³|10⁻³|10⁻⁶", " ", texte)
texte = re.sub(r"[²³⁴⁻¹⁰⁵⁶]", " ", texte)
texte = re.sub(r"\b(?:s|f)0\d\b|\bP0\b|\bP1\b|\bQ1\b|\bP10\b|\bP15\b|A[123]\b", " ", texte)
nombres = re.findall(r"(?<![\w,.])\d{1,3}(?:[  ]\d{3})+(?:,\d+)?(?![\d,])|(?<![\w,.])\d+(?:,\d+)?(?![\d])", texte)
FACTEURS = (1.0, 100.0)
manquants = []
for n in nombres:
    s = n.replace(" ", "").replace(" ", "")
    nd = len(s.split(",")[1]) if "," in s else 0
    v = float(s.replace(",", "."))
    if nd == 0 and v < 10:
        continue
    ok = False
    for fac in FACTEURS:
        for c in vals:
            if round(c * fac, nd) == v or abs(c * fac - v) <= 0.5 * 10 ** (-nd) * 1.0000001:
                ok = True
                break
        if ok:
            break
    if not ok:
        manquants.append(n)
print(f"{len(nombres)} nombres lus dans le document ; {len(manquants)} non retrouvés dans les sorties :")
for m in sorted(set(manquants), key=lambda x: float(x.replace(' ', '').replace(' ', '').replace(',', '.'))):
    print("  ", m)

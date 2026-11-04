# -*- coding: utf-8 -*-
"""Tests des schémas du garde-corps (acier/garde_corps/schemas.py) : quatre vues sur le kit
des assemblages, rien n'est recalculé, et chaque garantie rougit si on la retire :

  1. VUES : élévation, plan, détail, section pour le cas par défaut ; pas de détail quand le
     montant est scellé ; aucune vue sur un refus de géométrie ; SVG autonome avec viewBox.
  2. ÉLÉVATION : un rectangle de montant par montant à son abscisse, la lisse sur toute la
     longueur, les barreaux à leur entraxe, les platines, la chaîne des entraxes sur UNE
     rangée (une seule ligne de cote continue), H, les noms M1 … et le gouvernant ◉ avec son
     taux, les murs d'extrémité, les angles d'un L, ⊙ pour q_h, flèches pour q_v.
  3. PLAN : droit = un tronçon, L = deux, U = trois ; un montant par position ; charges
     perpendiculaires ; le mur du système mur du côté extérieur.
  4. DÉTAIL : platine, deux ancrages à ±e_x/2, cordons en ocre, cotes b_p, e_x, t_p, h_p, a ;
     sans platine quand le montant est soudé sur un plat ; console au mur.
  5. SECTION : contour du catalogue, hachures de la paroi seulement (tube), cotes Ø ou h × b, t.
  6. PEINTRE : chaque vue se peint en PDF par le peintre du kit sans erreur, et ses textes
     s'y retrouvent.
  7. PURETÉ : aucun import de Streamlit ni du moteur dans schemas.py.

Lancement : python3 tests/test_garde_corps_schemas.py (depuis la racine).
"""
import io
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402

from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import fixation as FX  # noqa: E402
from acier.garde_corps import schemas as SC  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.rapport import peindre  # noqa: E402
from ndc_pdf.kit import Doc  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def cas(**kw):
    c = ET.nouveau_cas()
    for k, v in kw.items():
        c[k] = {**c[k], **v} if isinstance(v, dict) else v
    return c


def prims(d, t=None, cls=None):
    """Les primitives du corps (à plat), filtrées par type et par classe."""
    out = []

    def rec(lst):
        for p in lst:
            if p["t"] == "g":
                rec(p["enfants"])
            else:
                out.append(p)
    rec(d.corps)
    return [p for p in out if (t is None or p["t"] == t) and (cls is None or cls in p["cls"])]


def textes(d):
    out = []

    def rec(lst):
        for p in lst:
            if p["t"] == "text":
                out.append(str(p["txt"]))
            elif p["t"] == "g":
                rec(p["enfants"])
    rec(d.corps)
    rec(d.cotes)
    return out


def vue(res, cle):
    return {k: d for k, _, d in SC.vues(res)}[cle]


# ================================================================
print("\n=== 1. Vues ===")
res = ET.verifier(cas())
vs = SC.vues(res)
chk("quatre vues : élévation, plan, détail, section", [k for k, _, _ in vs] == ["elevation", "plan", "detail", "section"])
svg = SC.svg(vs[0][2], "t0")
chk("SVG autonome : viewBox, style embarqué, identifiant", svg.startswith("<svg") and 'viewBox="' in svg and "<style>" in svg and 'id="t0"' in svg)
chk("montant scellé : pas de vue de détail", [k for k, _, _ in SC.vues(ET.verifier(cas(fixation={"type": "scellee"})))] == ["elevation", "plan", "section"])
g2 = cas()
g2["geometrie"]["troncons"][0] = {"L": 1000, "s_max": 1000, "entraxes": []}
g2["geometrie"]["debut"] = g2["geometrie"]["fin"] = "mur"
chk("refus de géométrie : aucune vue", SC.vues(ET.verifier(g2)) == [])

# ================================================================
print("\n=== 2. Élévation ===")
e = vue(res, "elevation")
rects = prims(e, "rect", "pp")
xs_m = sorted(round((r["x"] + r["w"] / 2), 1) for r in rects if r["h"] > 500)
chk("cinq montants à x = 0, 900, 1800, 2700, 3600 (rectangles de la hauteur H)", xs_m == [0.0, 900.0, 1800.0, 2700.0, 3600.0], str(xs_m))
lisse = [r for r in prims(e, "rect", "ps") if abs(r["w"] - 3600) < 1e-6]
chk("la lisse haute court sur toute la longueur, à y = −H", len(lisse) >= 1 and abs(lisse[0]["y"] + lisse[0]["h"] / 2 + 1100) < 1e-6)
barreaux = [r for r in prims(e, "rect", "ps") if r["w"] < 80 and r["h"] > 500]
chk("les barreaux à l'entraxe de 110 mm (≈ 32 sur 3,60 m), dessinés à une largeur lisible (≥ 0,35·fs)", 28 <= len(barreaux) <= 34
    and abs((barreaux[1]["x"] - barreaux[0]["x"]) - 110) < 1e-6 and barreaux[0]["w"] >= 0.35 * e.fs - 1e-6, str(len(barreaux)))
platines = [r for r in rects if abs(r["w"] - 150) < 1e-6 and abs(r["h"] - 10) < 1e-6]
chk("une platine 150 × 10 sous chaque montant", len(platines) == 5)
chaines = [p for p in e.cotes if p["t"] == "g" and any(q["t"] == "path" and "dln" in q["cls"] and len(q["lignes"]) >= 6 for q in p["enfants"])]
chk("la chaîne des entraxes : UNE ligne de cote continue avec ses repères (une rangée)", len(chaines) == 1)
txt = textes(e)
chk("les noms M1 … M5, le gouvernant ◉ M2 avec son taux, H = 1100, L = 3600", all(any(t == n for t in txt) for n in ("M1", "M3", "M4", "M5"))
    and any("◉ M2" in t and "%" in t for t in txt) and any("H = 1100" in t for t in txt) and any("L = 3600" in t for t in txt))
chk("le bloc des charges nomme q_h (⊙ vers soi) et q_v", any("q_h = 0,50 kN/m" in t for t in txt) and any("q_v = 0,50 kN/m" in t for t in txt))
chk("⊙ (q_h vers soi) et flèches ↓ (q_v) sur la lisse", len(prims(e, "circle", "gv")) >= 6 and len(prims(e, "poly", "fzh")) >= 6)
rL = ET.verifier(cas(geometrie={"forme": "L", "fin": "mur"}))
eL = vue(rL, "elevation")
chk("en L avec fin au mur : un trait d'axe à l'angle, un mur hachuré à droite, « angle » écrit", len(prims(eL, "path", "ax")) >= 1
    and any(p["t"] == "poly" and max(x for x, _ in p["pts"]) > 6000 for p in prims(eL, "poly", "pp")) and any(t == "angle" for t in textes(eL)))
rM = ET.verifier(cas(geometrie={"systeme": "mur"}, remplissage={"type": "aucun"}))
eM = vue(rM, "elevation")
chk("système mur : consoles C1 … C5 nommées, pas de barreau", any(t == "C1" for t in textes(eM)) and not [r for r in prims(eM, "rect", "ps") if r["w"] < 20 and r["h"] > 500])

# ================================================================
print("\n=== 3. Plan ===")
p = vue(res, "plan")
chk("droit : un tronçon, cinq montants, L₁ = 3600", len(prims(p, "path", "fl2")) == 1 and len(prims(p, "circle", "pp")) == 5 and any("L" in t and "3600" in t for t in textes(p)))
pL = vue(rL, "plan")
chk("en L : deux tronçons, L₂ coté à droite", len(prims(pL, "path", "fl2")) == 2 and any("L" in t and "2400" in t for t in textes(pL)))
rU = ET.verifier(cas(geometrie={"forme": "U"}))
pU = vue(rU, "plan")
chk("en U : trois tronçons, des flèches de charge sur chacun", len(prims(pU, "path", "fl2")) == 3 and len(prims(pU, "poly", "fzh")) >= 9)
pM = vue(rM, "plan")
murs = [q for q in prims(pM, "poly", "pp") if len(q["pts"]) == 4 and max(y for _, y in q["pts"]) > 50]
chk("système mur : le mur hachuré du côté extérieur (y > 0 sur le dessin)", len(murs) >= 1 and sum(len(p["lignes"]) for p in prims(pM, "path", "ht")) >= 2)
chk("système mur : pas de barreau en élévation", not [r for r in prims(eM, "rect", "ps") if r["w"] < 80 and r["h"] > 500])

# ================================================================
print("\n=== 4. Détail ===")
d = vue(res, "detail")
tiges = prims(d, "rect", "tg")
chk("deux ancrages à ±e_x/2 = ±50 mm", len(tiges) == 2 and sorted(round(r["x"] + r["w"] / 2) for r in tiges) == [-50, 50])
chk("la platine 150 × 10 et les deux cordons en ocre", any(abs(r["w"] - 150) < 1e-6 and abs(r["h"] - 10) < 1e-6 for r in prims(d, "rect", "pp")) and len(prims(d, "poly", "wb")) == 2)
txt = textes(d)
chk("cotes b_p 150, e_x 100, t_p 10, h_p 48,3, a 4 ; efforts V, N, M dans le bloc", all(any(s in t for t in txt) for s in ("b_p", "e_x", "t_p", "h_p", "a 4"))
    and any("V = 0,45 kN" in t and "M = 0,495" in t for t in txt))
dS = vue(ET.verifier(cas(fixation={"type": "soudee"})), "detail")
chk("soudé sur un plat : pas d'ancrage, pas de platine, cordons présents", not prims(dS, "rect", "tg") and len(prims(dS, "poly", "wb")) == 2
    and not any(abs(r["h"] - 10) < 1e-6 and abs(r["w"] - 150) < 1e-6 for r in prims(dS, "rect", "pp")))
dM = vue(rM, "detail")
chk("console au mur : titre « Console au mur », plat hachuré, N vers le mur", any("Console au mur" in t for t in textes(dM)) and len(prims(dM, "path", "ht")) >= 2
    and any("traction" in t for t in textes(dM)))
dB = vue(ET.verifier(cas(fixation={"ancrage": {**FX.fixation_defaut()["ancrage"], "type": "boulon"}})), "detail")
chk("boulons : support acier (pas de teinte béton)", not prims(dB, "poly", "be") and prims(dB, "poly", "pp"))
chk("chevilles : support béton", prims(d, "poly", "be"))

# ================================================================
print("\n=== 5. Section ===")
s = vue(res, "section")
chk("CHS : contour extérieur, trou, hachures de la paroi seulement, Ø 48,3 et t 3,2", len(prims(s, "poly", "pp")) == 1 and len(prims(s, "poly", "bo")) == 1
    and any("48,3" in t for t in textes(s)) and any("3,2" in t for t in textes(s)))
hach = [l for p in prims(s, "path", "ht") for l in p["lignes"]]
r_int = 48.3 / 2 - 3.2
chk("aucune hachure ne traverse le trou : le milieu de chaque segment est dans la paroi (à ≥ r_int du centre), et il y en a de chaque côté",
    hach and all(((l[0][0] + l[1][0]) / 2) ** 2 + ((l[0][1] + l[1][1]) / 2) ** 2 >= (r_int - 0.05) ** 2 for l in hach) and len(hach) >= 12)
sP = vue(ET.verifier(cas(montant={"section": {"nom": "Plat 50x10"}})), "section")
chk("plat 50 × 10 : plein hachuré, cotes h 50 et b 10", any("h" in t and "50" in t for t in textes(sP)) and any("b" in t and "10" in t for t in textes(sP)) and prims(sP, "path", "ht"))

# ================================================================
print("\n=== 6. Peintre ===")
for nom, r in (("défaut", res), ("L mur", rL), ("système mur", rM), ("U", rU)):
    chemin = os.path.join("/tmp", f"gc_schemas_{abs(hash(nom)) % 1000}.pdf")
    doc = Doc(chemin, landscape(A4))
    doc.new_page(landscape(A4))
    try:
        y = 420
        for k, titre, dv in SC.vues(r):
            peindre(doc, dv, 30, y - 180, 380, 180)
            y -= 190
            if y < 100:
                doc.new_page(landscape(A4))
                y = 420
        doc.save()
        texte = " ".join(pg.get_text() for pg in pymupdf.open(chemin))
        chk(f"{nom} : toutes les vues se peignent, et « M1 » ou « C1 » et « L₁ » sont dans le PDF", ("M1" in texte or "C1" in texte) and "L" in texte)
    except Exception as ex:  # noqa: BLE001
        chk(f"{nom} : toutes les vues se peignent", False, f"{type(ex).__name__}: {ex}"[:200])

# ================================================================
print("\n=== 7. Pureté ===")
src = open("acier/garde_corps/schemas.py", encoding="utf-8").read()
chk("schemas.py n'importe ni Streamlit ni etat ni membres", not re.search(r"^\s*(import|from)\s+streamlit", src, re.M)
    and not re.search(r"from \. import (etat|membres|fixation)", src) and "verifier(" not in src)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Schémas techniques des soudures (acier/soudures/schemas.py), sur le kit
de dessin des assemblages :

 - chaque configuration a ses vues (2 ou 3, C8 : 1), toutes des SVG valides
   dont le corps et les cotes tiennent DANS la boîte de vue (rien de coupé :
   étiquettes d'effort, symboles de soudure, noms des cordons) ;
 - ce qui doit s'y lire s'y lit : profilé du catalogue, cotes, symbole de
   soudure EN 22553 selon la jonction (a, s + a, pleine pénétration), point
   gouvernant et son taux, efforts dans le sens saisi, tronçons, cordons
   exclus en interrompu ;
 - les mêmes scènes se peignent dans un PDF par le peintre du kit ;
 - pureté : aucun Streamlit.

Lancement : python3 tests/test_soudures_schemas.py
"""
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as XT

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.soudures import etat as ET  # noqa: E402
from acier.soudures import schemas as SC  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def verif(config, params=None, efforts=None):
    cas = ET.nouveau_cas(config)
    cas["params"].update(params or {})
    cas["efforts"].update(efforts or {})
    return ET.verifier(cas)


def textes(svg: str):
    return re.findall(r">([^<>]+)</text>", svg)


def dans_la_boite(d, tol=1.0):
    """Le corps et les cotes tiennent dans la boîte de vue (rien n'est coupé)."""
    x1, y1, w, h = d.viewbox
    ex = SC._extent(d.corps + d.cotes, d.fs)
    return ex is not None and ex[0] >= x1 - tol and ex[1] >= y1 - tol and ex[2] <= x1 + w + tol and ex[3] <= y1 + h + tol, \
        f"viewbox {tuple(round(v) for v in d.viewbox)} étendue {tuple(round(v) for v in (ex or ()))}"


print("=== 1. Les vues de chaque configuration : nombre, SVG valides, rien de coupé ===")
ATTENDU = {"C1": 2, "C2": 2, "C3": 3, "C4": 2, "C5": 2, "C6": 2, "C7": 2, "C8": 1}
SVGS = {}
for cfg, n in ATTENDU.items():
    r = verif(cfg)
    vues = SC.vues(r)
    svgs = [SC.svg(d, f"t_{cfg}_{i}") for i, (_, _, d) in enumerate(vues)]
    SVGS[cfg] = (vues, svgs)
    valides = all(XT.fromstring(s) is not None for s in svgs)
    boites = [dans_la_boite(d) for _, _, d in vues]
    chk(f"{cfg} : {n} vue(s) ({', '.join(t for _, t, _ in vues)}), SVG valides, tout dans la boîte",
        len(vues) == n and valides and all(ok for ok, _ in boites), str([i for ok, i in boites if not ok]))
chk("la première vue est la vue principale : C3 élévation, C5 vue de face, C7 vue en plan",
    SVGS["C3"][0][0][0] == "elevation" and SVGS["C5"][0][0][0] == "joint" and SVGS["C7"][0][0][0] == "joint")

print("\n=== 2. Ce qui doit se lire ===")
t1 = " ".join(textes(SVGS["C1"][1][0])) + " | " + " ".join(textes(SVGS["C1"][1][1]))
chk("C1 : L_rec = 120, b = 100, symbole « a 5 », plats nommés avec leur épaisseur, coupe A-A cotée t_A / t_B",
    "L_rec = 120" in t1 and "b = 100" in t1 and "a 5" in t1 and "Plat A · t 10" in t1 and "Plat B · t 10" in t1 and "t_A 10" in t1 and "t_B 10" in t1, t1)
t2 = " ".join(textes(SVGS["C2"][1][0]))
chk("C2 coupe : t_B 10, t_A 12, N = 40,0 kN et M = 1,00 kN·m dans le sens saisi, symbole a 5",
    "t_B 10" in t2 and "t_A 12" in t2 and "N = 40,0 kN" in t2 and "M = 1,00 kN·m" in t2 and "a 5" in t2, t2)
t3 = " ".join(textes(SVGS["C3"][1][0]))
chk("C3 élévation : HEB 200 du catalogue nommé et coté (h_p = 200), plat 200 × 10, e = 150, h = 200, V = 60,0 kN, ◉ 61 %",
    "HEB 200" in t3 and "h_p = 200" in t3 and "Console · plat 200 × 10" in t3 and "e = 150" in t3 and "h = 200" in t3
    and "V = 60,0 kN" in t3 and "◉ 61 %" in t3, t3)
t3b = " ".join(textes(SVGS["C3"][1][1]))
chk("C3 coupe horizontale : t 10, b_p = 200, symbole a 5 ; section en I réelle (congés : polygone de plus de 20 sommets)",
    "t 10" in t3b and "b_p = 200" in t3b and "a 5" in t3b
    and max(len(p.get("points").split()) for p in XT.fromstring(SVGS["C3"][1][1]).iter("{http://www.w3.org/2000/svg}polygon")) > 20, t3b)
t4 = " ".join(textes(SVGS["C4"][1][0]))
chk("C4 : b_x = 200, b_y = 150, x_V = 250, y_V 75, bloc des efforts « V_y = 40,0 kN », trait de coupe A-A",
    "b_x = 200" in t4 and "b_y = 150" in t4 and "x_V = 250" in t4 and "y_V 75" in t4 and "V_y = 40,0 kN" in t4 and t4.count("A") >= 2, t4)
t5 = " ".join(textes(SVGS["C5"][1][0])) + " | " + " ".join(textes(SVGS["C5"][1][1]))
chk("C5 : IPE 200 sur platine t 15, h = 200, b = 100, symboles a 5 (semelles) et a 4 (âme), M_y = 30,00 kN·m et V_z = 60,0 kN en élévation",
    "IPE 200 sur platine t 15" in t5 and "h = 200" in t5 and "b = 100" in t5 and "a 5" in t5 and "a 4" in t5
    and "M_y = 30,00 kN·m" in t5 and "V_z = 60,0 kN" in t5 and "t_p 15" in t5, t5)
t6 = " ".join(textes(SVGS["C6"][1][0]))
chk("C6 : SHS 100x100x5 nommé, b = 100 et h = 100, bloc « N = 50,0 kN ⊙ (arrachement) », symbole a 4",
    "SHS 100x100x5" in t6 and "b = 100" in t6 and "h = 100" in t6 and "N = 50,0 kN ⊙ (arrachement)" in t6 and "a 4" in t6, t6)
t7 = " ".join(textes(SVGS["C7"][1][0])) + " | " + " ".join(textes(SVGS["C7"][1][1]))
chk("C7 pleine : b = 150, N = 200,0 kN, symbole « pleine pénétration », coupe « chanfrein en X » cotée t 12",
    "b = 150" in t7 and "N = 200,0 kN" in t7 and "pleine pénétration" in t7 and "chanfrein en X" in t7 and "t 12" in t7, t7)
t8 = " ".join(textes(SVGS["C8"][1][0]))
chk("C8 : les chaînes nommées avec leur gorge (cordon 1 · a 5), R et G, bloc V_y = 60,0 kN · M_x = 9,00 kN·m",
    "cordon 1 · a 5" in t8 and "cordon 2 · a 5" in t8 and "R" in t8 and "G" in t8 and "V_y = 60,0 kN" in t8 and "M_x = 9,00 kN·m" in t8, t8)

print("\n=== 3. Jonctions, tronçons, exclus, variantes ===")
r = verif("C2", {"jonction": "pleine", "tB": 30, "tA": 30}, {"N": 400})
v = SC.vues(r)
s = SC.svg(v[0][2])
chk("C2 pleine pénétration : symbole « pleine pénétration » (V), pas de « a », chanfrein hachuré ocre (classe htw)",
    "pleine pénétration" in " ".join(textes(s)) and "a 5" not in " ".join(textes(s)) and 'class="ht htw"' in s and all(dans_la_boite(d)[0] for _, _, d in v))
r = verif("C2", {"jonction": "partielle", "p": 6, "renfort": 3, "tB": 20})
s = SC.svg(SC.vues(r)[0][2])
chk("C2 pénétration partielle avec renfort : symbole « s 6 + a 3 »", "s 6 + a 3" in " ".join(textes(s)))
r = verif("C3", {"troncons": {"n": 2, "l": 60}})
v = SC.vues(r)
s_el = SC.svg(v[0][2])
chk("C3 par 2 tronçons : deux bandes de cordon en élévation (pas une), vue de face avec quatre bandes",
    s_el.count('class="wb"') == 2 and SC.svg(v[2][2]).count('class="wb"') == 4, f"{s_el.count('class=\"wb\"')}")
r = verif("C3", {"a": 2.0})
v = SC.vues(r)
chk("C3 gorge 2 mm (refus après construction) : les vues existent, cordons exclus en interrompu (classe wbx), pas de point gouvernant",
    len(v) == 3 and 'class="wbx"' in SC.svg(v[2][2]) and "◉" not in " ".join(textes(SC.svg(v[0][2]))))
r = verif("C3", {"poteau": "plat"})
s = SC.svg(SC.vues(r)[0][2])
chk("C3 sur un plat : « Appui » à la place du profilé, pas de cote h_p", "Appui" in " ".join(textes(s)) and "h_p" not in " ".join(textes(s)))
r = verif("C5", {"jonction_semelles": "pleine", "jonction_ame": "partielle"})
s = SC.svg(SC.vues(r)[0][2])
chk("C5 semelles pleines + âme partielle : symboles « pleine pénétration » et « s 4 »", "pleine pénétration" in " ".join(textes(s)) and "s 4" in " ".join(textes(s)))
r = verif("C6", {"profil": "CHS 114.3x5", "troncons": {"n": 4, "l": 60}})
v = SC.vues(r)
s = SC.svg(v[0][2])
chk("C6 CHS par 4 tronçons : d = 114,3, quatre bandes sur le pourtour, ◉ 112 % (non conforme)",
    "d = 114,3" in " ".join(textes(s)) and s.count('class="wb"') >= 4 and "◉ 112 %" in " ".join(textes(s)), " ".join(textes(s)))
r = verif("C6", {"profil": "CHS 114.3x5", "jonction": "pleine"})
s = SC.svg(SC.vues(r)[0][2])
chk("C6 tube à pleine pénétration : un anneau de soudure (une bande fermée), symbole « pleine pénétration »",
    s.count('class="wb"') == 1 and "pleine pénétration" in " ".join(textes(s)))
r = verif("C7", {"penetration": "partielle"}, {"M": 2.0, "V": 20.0})
v = SC.vues(r)
t7p = " ".join(textes(SC.svg(v[0][2]))) + " | " + " ".join(textes(SC.svg(v[1][2])))
chk("C7 partielle p = 5 : symbole « s 5 », coupe « pénétration partielle p des deux faces » cotée p 5, V et M dessinés",
    "s 5" in t7p and "pénétration partielle p des deux faces" in t7p and "p 5" in t7p and "V = 20,0 kN" in t7p and "M = 2,00 kN·m" in t7p, t7p)
r = verif("C8", {"elements": [{"type": "arc", "xc": 0, "yc": 0, "r": 60, "ang1": 0, "ang2": 360, "a": 4, "cote": 1, "troncons": {"n": 3, "l": 50}},
                              {"type": "segment", "x1": -100, "y1": -90, "x2": 100, "y2": -90, "a": 5, "cote": 1}]}, {"N": 50, "Mz": 3})
v = SC.vues(r)
s = SC.svg(v[0][2])
chk("C8 cercle par tronçons + segment : trois arcs nommés « arc 1 · tronçon k · a 4 », M_z en arc, N en ⊙, tout dans la boîte",
    all(f"arc 1 · tronçon {k} · a 4" in " ".join(textes(s)) for k in (1, 2, 3)) and "M_z = 3,00 kN·m" in " ".join(textes(s))
    and "⊙" in " ".join(textes(s)) and dans_la_boite(v[0][2])[0], dans_la_boite(v[0][2])[1])
r = verif("C1", {}, {"N": -80})
s = SC.svg(SC.vues(r)[0][2])
chk("C1 en compression (N < 0) : la flèche vient sur le bout du plat (pointe au bout), étiquette N = 80,0 kN",
    "N = 80,0 kN" in " ".join(textes(s)) and 'class="fzh"' in s)
chk("refus avant construction (profilé inconnu) : aucune vue", SC.vues(verif("C5", {"profil": "IPE 999"})) == [])

print("\n=== 4. Le peintre du kit (PDF) et la pureté ===")
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from ndc_pdf.kit import Doc  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.rapport import peindre  # noqa: E402
import pymupdf  # noqa: E402

fd, chemin = tempfile.mkstemp(suffix=".pdf")
os.close(fd)
d = Doc(chemin, landscape(A4), title="essai")
d.new_page(landscape(A4))
x = 20
for dv in (SVGS["C3"][0][0][2], SVGS["C5"][0][1][2], SVGS["C7"][0][0][2], SVGS["C7"][0][1][2]):
    peindre(d, dv, x, 300, 195, 200)
    x += 200
d.save()
texte = " ".join(p.get_text() for p in pymupdf.open(chemin))
os.remove(chemin)
chk("quatre scènes peintes dans un PDF (C3 élévation, C5 élévation, C7 plan et coupe) : HEB 200, e = 150, IPE 200, t_p 15, b = 150, chanfrein en X s'y lisent",
    all(m in texte for m in ("HEB 200", "e = 150", "IPE 200", "t_p 15", "b = 150", "chanfrein en X")), texte[:300])
src_schemas = open("acier/soudures/schemas.py", encoding="utf-8").read()
chk("schemas.py n'importe pas Streamlit et ne recalcule rien (aucun import du moteur : criteres, groupe, etat)",
    "streamlit" not in src_schemas and re.search(r"^(from|import) .*\b(criteres|groupe|etat)\b", src_schemas, re.M) is None)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)

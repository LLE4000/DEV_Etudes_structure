# -*- coding: utf-8 -*-
"""Note de calcul « Charges sur plancher » (lot 3) : ``plancher/charges/note.py``,
son peintre SVG → reportlab et les textes de méthode partagés.

Chaque garantie rougit si on la retire :
  1. FORME : DEUX pages A4 paysage, aucun débordement, sur quatre cas —
     hourdis complet (engin, plateau, étançon, filière), dalle + étançon +
     stock, prédalle + file d'étais + grue araignée + filière (appuis en haut
     et en bas), hourdis sans résistances (refus) ; variante normale sur le
     cas complet ; la note se génère en moins de 3 s.
  2. CONTENU p.1 : statut (NON CONFORME / DONNÉES MANQUANTES), η max, famille
     et objet gouvernants, taux par famille, deux lignes de données (plancher,
     charges et répartiteurs), tableau des ÉLÉMENTS (H1…H8, M_Ed, M_Rd, η),
     tableau des CHARGES (E1, ses appuis, T1), VÉRIFICATIONS groupées (un
     stock de 32 empreintes = UNE ligne « la pire », jamais 96), CONCLUSION,
     légende P-2 / P-3.
  3. PLAN PEINT depuis le MÊME SVG que l'écran : ses textes sont dans le PDF
     (L = 7,20 m, A, B, H7, P2 · 95 kN, étiquette de l'engin), la coupe aussi
     (aire efficace), et rien de l'interface n'est imprimé (ni « clic », ni
     les poignées ±, ni l'invite du bord libre) ; échelle du plan au
     cartouche.
  4. PAGE 2 : MÉTHODE ET DÉTAIL, les sections de ``methode.py`` (source
     unique : un fragment de chaque section est retrouvé mot pour mot),
     références, bibliothèque avec la source du gabarit, balayage s'il y a
     un engin, « à transcrire », cartouche (cas, plancher, date).
  5. PEINTRE : composition translate · rotate · matrix sur un point, rect
     sous rotate(90), texte vertical et motif de hachures rendus, balise
     inconnue ignorée, opacité nulle non peinte.
  6. GROUPEMENT : ``_grouper`` ramène S1.1 / S1.2 / S1.3 à une ligne (la
     pire) et laisse E1.P1 / E1.P2 distincts.

Lancement : python3 tests/test_charges_plancher_note.py (depuis la racine).
"""
import io
import os
import re
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402

from plancher.charges import etat as ET  # noqa: E402
from plancher.charges import methode as ME  # noqa: E402
from plancher.charges import note as NT  # noqa: E402
from plancher.charges import peintre as PE  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def textes(pdf):
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    return doc, [pg.get_text() for pg in doc]


def sans_espaces(t):
    return t.replace(" ", "").replace("\n", "")


def cas_base():
    etat = ET.nouvel_etat()
    return etat, ET.cas_actif(etat)


# ---------------------------------------------------------------- les quatre cas
e0, c0 = cas_base()
h = c0["plancher"]["hourdis"]
h["M_Rd"], h["V_Rd"], h["source"] = 60.0, 80.0, "valeurs d'essai du harnais"
ET.ajouter_engin(e0, c0, next(n for n in ET.gabarits(e0) if "MRT 2150" in n))
c0["charges"][0]["P"] = {"P1": 95.0, "P2": 95.0, "P3": 60.0, "P4": 60.0}
ET.regler_repartiteur(e0, c0, "E1.P2", "type", "plateau")
ET.ajouter_charge(e0, c0, "ponctuel")
ET.ajouter_filiere(e0, c0, ["E1.P1", "E1.P3"], longueur=5.6)
c0["selection"] = "rep:E1.P2"
e1, c1 = cas_base()
c1["plancher"]["type"] = "dalle"
ET.ajouter_charge(e1, c1, "ponctuel")
ET.ajouter_charge(e1, c1, "surfacique")
e2, c2 = cas_base()
c2["plancher"]["type"] = "predalle"
c2["plancher"]["sens"] = 2
ET.ajouter_charge(e2, c2, "lineaire")
ET.ajouter_engin(e2, c2, next(n for n in ET.gabarits(e2) if "Grue araignée" in n))
c2["charges"][-1]["P"] = {"P1": 40.0, "P2": 40.0, "P3": 40.0, "P4": 40.0}
ET.ajouter_filiere(e2, c2, ["E1.P1", "E1.P2"], longueur=4.8)
e3, c3 = cas_base()
INFOS = {"nom_projet": "Halle d'essai", "partie": "Plancher R+1", "date": "02/11/2026", "indice": "0"}

print("=== 1. Forme ===")
notes = {}
for nom, (e, c) in (("complet", (e0, c0)), ("dalle", (e1, c1)), ("predalle", (e2, c2)), ("refus", (e3, c3))):
    t0 = time.time()
    pdf = NT.generer_note(e, c, INFOS)
    dt = time.time() - t0
    doc, tx = textes(pdf)
    notes[nom] = (pdf, doc, tx)
    chk(f"{nom} : deux pages A4 paysage", len(doc) == 2 and all(abs(pg.rect.width - 842) < 1 and abs(pg.rect.height - 595) < 1 for pg in doc),
        f"{len(doc)} pages")
    chk(f"{nom} : aucun débordement", not NT.derniers_avertissements, str(NT.derniers_avertissements))
    chk(f"{nom} : générée en moins de 3 s", dt < 3.0, f"{dt:.2f} s")
chk("cas complet : variante normale (pas de réduction de taille)", NT.generer_note(e0, c0, INFOS) and NT.derniere_variante == "normale",
    str(NT.derniere_variante))

print("\n=== 2. Contenu de la page 1 ===")
pdf, doc, tx = notes["complet"]
p1, p1n = tx[0], sans_espaces(tx[0])
calc = NT.Calcul(e0, c0)
g = calc.res["gouverne"]
chk("bandeau : bureau, titre, projet · partie · cas, date · indice",
    "Bureaud'ÉtudesValens" in p1n and "Charges de chantier sur plancher" in p1 and "Halle d'essai" in p1 and "Plancher R+1" in p1
    and "cas C1" in p1 and "02/11/2026" in p1 and "indice 0" in p1)
chk("statut NON CONFORME, η max, gouverne", "NONCONFORME" in p1n and "ηmax" in p1n and NT._pct(calc.res["taux_max"]).replace(" ", "") in p1n
    and f"gouverne:{g['famille']}·{g['objet']}" in p1n)
chk("taux par famille (Flexion, Tranchant, Filières)", all(x in p1 for x in ("Flexion", "Tranchant", "Filières")))
chk("lignes de données : plancher (M_Rd, présentes) et charges (E1, T1, répartiteurs)",
    "M_Rd 60,0 kN·m" in p1 and "présentes 1,00 kN/m²" in p1 and "E1 Télescopique rotatif" in p1 and "T1 étançon" in p1 and "E1.P2 : Tôle acier" in p1
    and "F1 = HEB 140" in p1)
chk("tableau des ÉLÉMENTS : H1 … H8, M_Ed, M_Rd, η", "ÉLÉMENTS" in p1n and all(f"\nH{i}\n" in p1 for i in range(1, 9)) and "M_Ed" in p1 and "M_Rd" in p1)
h7 = next(x for x in calc.res["elements"] if x["nom"] == "H7")
chk("H7 : M_Ed et η de la note = ceux du moteur", NT._fmt(h7["M_max"], 1) in p1 and NT._pct(h7["taux_M"]) in p1)
chk("tableau des CHARGES : E1 et ses quatre appuis, T1, répartiteurs", "CHARGES" in p1n and all(f"\nP{i}\n" in p1 for i in range(1, 5))
    and "\nT1\n" in p1 and "filière F1" in p1 and "Tôle acier S235" in p1)
chk("VÉRIFICATIONS : plateau, filière, local ; membrure sous la filière groupée (éléments → une ligne « la pire », famille « local ×n ») ; "
    "les deux lignes de la filière (flexion, ressort qui tire) restent distinctes",
    "VÉRIFICATIONS" in p1n and "plateau" in p1 and re.search(r"local ×\d", p1) and "(la pire)" in p1
    and p1[p1.find("FAMILLE"):].count("membrure sous F1") == 1 and p1[p1.find("FAMILLE"):].count("F1 · HEB 140") >= 3
    and "filière ×" not in p1)
chk("bandeau : les lignes non vérifiées d'une même origine sont comptées ensemble (« membrure sous F1 · HEB 140 (×n) »)",
    re.search(r"membrure sous F1 · HEB 140 \(×\d\)", p1) is not None)
chk("CONCLUSION et légende P-2 / P-3", "CONCLUSION" in p1n and "Taux maximal" in p1 and "P-2" in p1 and "P-3" in p1)
chk("lignes non vérifiées annoncées", "non vérifiée(s)" in p1)
pd1 = notes["dalle"][2][0]
chk("dalle : bandes B1…, flexion et tranchant par mètre, stock de 32 empreintes = UNE ligne de pression localisée",
    "\nB1\n" in pd1 and "kN·m/m" in pd1 and pd1.count("pression localisée sous S1") == 1 and "local ×32" in pd1)
pr1 = notes["refus"][2][0]
chk("refus : DONNÉES MANQUANTES et le message du moteur", "DONNÉESMANQUANTES" in sans_espaces(pr1) and "M_Rd et V_Rd" in pr1)
bd = NT.Calcul(e1, c1).res["elements"]
chk("dalle : les taux par bande restent à l'échelle du taux global (plus de 10⁹ % sur une bande non chargée — défaut du 02/11/2026)",
    max(b["taux"] for b in bd) <= NT.Calcul(e1, c1).res["taux_max"] * 1.0001 and all(b["taux"] < 100 for b in bd), str([round(b["taux"], 2) for b in bd]))

print("\n=== 3. Plan peint depuis le SVG de l'écran ===")
chk("textes du plan dans le PDF : L = 7,20 m, A, B, H7, P2 · 95 kN, étiquette de l'engin",
    "L = 7,20 m" in p1 and "\nA\n" in p1 and "\nB\n" in p1 and "P2 · 95 kN" in p1 and "E1 · Télescopique rotatif" in p1)
chk("coupe peinte : aire efficace et plateau", "aire efficace" in p1 and "plateau 600" in p1)
chk("rien de l'interface : ni « clic », ni poignées ±, ni invite du bord libre",
    "clic" not in p1 and "bord libre" not in p1 and "\n−\n" not in p1 and "glisser" not in p1)
chk("échelle du plan au cartouche (1:n)", any(l.startswith("1:") and l[2:].isdigit() for l in tx[1].split("\n")))
pp1 = notes["predalle"][2][0]
chk("appuis en haut et en bas : le plan transposé est peint (B en bas, largeur en cote)", "largeur 4,80 m" in pp1 or "largeur 9,60 m" in pp1)

print("\n=== 4. Page 2 : méthode et détail ===")
p2, p2n = tx[1], sans_espaces(tx[1])
chk("titre et références", "MÉTHODEETDÉTAIL" in p2n and "EN 1168" in p2 and "Heft 240" in p2 and "EN 1993-1-8" in p2 and "EN 1992-1-1" in p2)
for sec in ME.SECTIONS:
    frag = sec["paragraphes"][0][:38]
    chk(f"section « {sec['titre']} » : titre et premier fragment (source unique methode.py)",
        sans_espaces(sec["titre"].upper()) in p2n and sans_espaces(frag) in p2n, frag)
chk("bibliothèque : gabarit de l'engin et sa source, plateau, filière", "Engin Télescopique rotatif" in p2 and "Source :" in p2
    and "Plateau Tôle acier S235" in p2 and "Filière F1 : HEB 140" in p2)
chk("balayage de position peint, avec sa phrase", "BALAYAGE" in p2n and ("Positions admissibles" in p2 or "Aucune position admissible" in p2)
    and "position actuelle" in p2)
chk("à transcrire : k du Heft 240, parts C.1, joint, ANB, catalogue", "ÀTRANSCRIRE" in p2n and "Heft 240" in p2 and "figure C.1" in p2 and "ANB" in p2)
chk("cartouche : cas, plancher, charges, répartiteurs, date · indice, échelle", all(sans_espaces(x) in p2n for x in ("CAS", "PLANCHER", "CHARGES", "RÉPARTITEURS", "DATE · INDICE", "ÉCHELLE DU PLAN")))
chk("refus : pas de balayage, méthode quand même", "BALAYAGE" not in sans_espaces(notes["refus"][2][1]) and "Heft 240" in notes["refus"][2][1])

print("\n=== 5. Peintre SVG ===")
m = PE.transformation("translate(10 20) rotate(90)")
chk("translate · rotate : (1, 0) → (10, 21)", all(pres(a, b) for a, b in zip(m.pt(1.0, 0.0), (10.0, 21.0))), str(m.pt(1.0, 0.0)))
m2 = PE.transformation("matrix(0 1 1 0 0 0)")
chk("matrix(0 1 1 0 0 0) transpose : (2, 3) → (3, 2)", all(pres(a, b) for a, b in zip(m2.pt(2.0, 3.0), (3.0, 2.0))))
m3 = PE.transformation("translate(5 5) matrix(0 1 1 0 0 0) rotate(90)")
chk("composition à trois : (1, 0) → rotate (0, 1) → matrix (1, 0) → translate (6, 5)", all(pres(a, b) for a, b in zip(m3.pt(1.0, 0.0), (6.0, 5.0))), str(m3.pt(1.0, 0.0)))
chk("échelle et angle de la matrice", pres(PE.transformation("scale(2)").echelle(), 2.0) and pres(PE.transformation("rotate(90)").angle(), 90.0))
svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500"><style>.x{}</style><defs><pattern id="p"/></defs>'
       '<rect x="100" y="100" width="300" height="100" fill="url(#chp_hc)" stroke="#111827" stroke-width="1"/>'
       '<g transform="translate(600 300) rotate(90)"><rect x="-50" y="-20" width="100" height="40" fill="#fee2e2"/>'
       '<text x="0" y="0" font-size="40" text-anchor="middle">Horizontal</text></g>'
       '<text transform="translate(80 400) rotate(-90)" font-size="40" fill="#374151" text-anchor="middle">Vertical</text>'
       '<circle cx="800" cy="100" r="30" fill="white" stroke="#111827"/><line x1="0" y1="480" x2="1000" y2="480" stroke="#000" stroke-dasharray="20 10"/>'
       '<foo x="1"/><rect x="0" y="0" width="10" height="10" fill="white" opacity="0"/><text x="900" y="450" font-size="30" font-weight="700">Gras</text></svg>')
buf = io.BytesIO()
c = rl_canvas.Canvas(buf, pagesize=(300, 200))
s = PE.peindre_svg(c, svg, 10, 10, 280, 180)
c.showPage()
c.save()
doc_s = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
ts = doc_s[0].get_text()
chk("peinture sans erreur, échelle = min(280/1000, 180/500) hors marge", 0.25 < s < 0.29, str(s))
chk("textes rendus : horizontal, vertical (rotate −90) et gras", "Horizontal" in ts and "Vertical" in ts and "Gras" in ts)
dess = doc_s[0].get_drawings()
chk("rect, cercle et ligne pointillée dessinés ; motif → aplat ; balise inconnue et opacité nulle ignorées",
    len(dess) >= 4 and any(d.get("dashes") not in (None, "[] 0") for d in dess), str(len(dess)))

print("\n=== 6. Groupement des lignes ===")
v = lambda obj, taux, st="ok": {"famille": "local", "objet": obj, "E": 1.0, "R": 2.0, "taux": taux, "statut": st, "methode": "", "unite": "kN"}
grp = NT._grouper([v("pression localisée sous S1.1", 0.2), v("pression localisée sous S1.2", 0.9, "attention"), v("pression localisée sous S1.3", 0.5),
                   v("pression localisée sous E1.P1", 0.3), v("pression localisée sous E1.P2", 0.4)])
chk("S1.1 / S1.2 / S1.3 → une ligne, la pire (0,9, attention), nommée « sous S1 » ; E1.P1 et E1.P2 distincts",
    len(grp) == 3 and grp[0][1] == 3 and grp[0][0]["taux"] == 0.9 and grp[0][0]["objet"] == "pression localisée sous S1"
    and grp[1][1] == 1 and grp[2][1] == 1, str([(g[0]["objet"], g[1]) for g in grp]))
grp2 = NT._grouper([v("membrure sous F1 · HEB 140→H2", None, "non_verifie"), v("membrure sous F1 · HEB 140→H3", None, "non_verifie")])
chk("filière → éléments : une ligne « sous F1 · HEB 140 »", len(grp2) == 1 and grp2[0][1] == 2 and grp2[0][0]["objet"] == "membrure sous F1 · HEB 140")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("  ÉCHEC :", nom, info)
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Note de calcul (note.py) — la note du bouton « Générer PDF » : page 1 de
calcul, page 2 « PLAN DE PRINCIPE » (trois vues cotées à la même échelle
normalisée, la plus grande qui tient).

Chaque garantie rougit si on la retire :
  1. FORME : DEUX pages A4 paysage, palette 01_encre, sans débordement
     (aucun avertissement) sur les 31 cas de référence ; le corps des
     formules n'est pas réduit sous 6,0 pt ; les tableaux ne s'ajustent
     jamais (la variante ne touche que les formules).
  2. CONTENU p.1 : statut, taux maximal, dimensionnante, taux par élément ;
     en-tête COMPACT (2 lignes d'objets + 1 ligne d'efforts, pas de
     géométrie détaillée) ; hypothèses d'une ligne, références abrégées
     (MSB P5) ; Ed / Rd de CHAQUE vérification active tels que le moteur
     les donne ; Tableau 3.3 ; identification ; pied de page en 2 lignes ;
     les formules obligatoires sont imprimées ; les phrases supprimées le
     21/09/2026 ne reviennent pas.
  3. DESSIN p.1 : élévation et plan peints, en notation Eurocode (hc, zc).
  3 bis. PLAN DE PRINCIPE (p.2) : titre, « Échelle 1:5 » (cas par défaut,
     échelle et disposition exposées), trois vues titrées, renvois de
     perçage (3×Ø22, 6×Ø22), rayon du grugeage, cote zt, cartouche avec
     visserie — et JAMAIS deux fois la même cote sur la planche.
  4. APPLICATION : « 📄 Générer PDF » produit la note de deux pages ; le
     rapport détaillé (3 pages) reste disponible dans l'onglet Note.

Lancement : python3 tests/test_assemblages_note.py
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402

from acier.formats import F, pct  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, note, synthese  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)
SCRATCH = os.environ.get("TMPDIR", "/tmp")

print("=== 1. Forme ===")
R = moteur.compute(dict(id_projet="Halle A", id_rep="P3/S7", id_red="X", id_date="21/09/2026"))
pdf = note.generer_note(R, {"nom_projet": "Halle A", "partie": "P3/S7", "date": "21/09/2026", "indice": "0",
                            "visserie": "1 rondelle + 1 écrou"})
with open(os.path.join(SCRATCH, "note_une_page_test.pdf"), "wb") as fh:
    fh.write(pdf)
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("deux pages : la note, puis le plan de principe", doc.page_count == 2, str(doc.page_count))
chk("A4 paysage", doc[0].rect.width > doc[0].rect.height and abs(doc[0].rect.width - 841.9) < 1)
chk("aucun débordement, variante « toutes » (toutes les lignes de formules)",
    not note.derniers_avertissements and note.derniere_variante == "toutes", str(note.derniers_avertissements))
spans = [s for b in doc[0].get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
t_form = {round(s["size"], 1) for s in spans if s["text"].startswith(("Fb,ver,Rd", "η = ", "Fv,Rd"))}
t_leg = {round(s["size"], 1) for s in spans if s["text"].startswith(("Références", "Notations"))}
t_tab = {round(s["size"], 1) for s in spans if s["text"].strip() in ("Cisaillement – groupe S", "Pression diamétrale – âme")}
chk("corps : formules 6,0 pt, tableaux 6,5 pt (titres de formules 6,3), légendes 5,6 pt — jamais moins "
    "(seules les cotes des dessins sont à l'échelle)",
    t_form == {6.0} and 6.5 in t_tab and min(t_tab) >= 6.3 and t_leg == {5.6}, f"{t_form} {t_tab} {t_leg}")
pix = doc[0].get_pixmap(dpi=60)
chk("page rendue (pixels non blancs)", sum(1 for i in range(0, len(pix.samples), 3 * 97) if pix.samples[i] < 250) > 300)
nb = 0
for cas in REF["cas"]:
    Rc = moteur.compute(cas["inputs"])
    p = note.generer_note(Rc, {})
    dc = pymupdf.open(stream=p, filetype="pdf")
    if dc.page_count != 2 or note.derniers_avertissements or note.derniere_echelle is None:
        nb += 1
        print("      ", cas["id"], dc.page_count, note.derniere_echelle, note.derniers_avertissements)
chk("31 cas de référence : deux pages, échelle normalisée trouvée, sans débordement", nb == 0, str(nb))

print("\n=== 2. Contenu lu dans le moteur ===")
t = doc[0].get_text()
s = t.replace(" ", "").replace("\n", "")
chk("bandeau : VÉRIFIÉ, η max, dimensionnante, taux par élément",
    "VÉRIFIÉ" in s and ("ηmax" + pct(R.eta_max, 1).replace(" ", "")) in s and "dimensionnant" in t
    and all(x in t for x in ("Boulons", "Cornières", "Portée", "Porteuse")))
manq = []
for c in R.checks:
    if not c.active:
        continue
    ed, rd, un = synthese.ed_rd(c)
    if ed not in t or rd not in t or pct(c.eta, 1) not in t:
        manq.append(c.key)
chk("Ed, Rd et η de chaque vérification active, tels que le moteur les donne", not manq, str(manq))
chk("Tableau 3.3 : chaque ligne avec sa valeur et son minimum",
    all(F(x.val, 1) in t and F(x.min, 1) in t for x in R.dist) and "TAB.3.3" in s.upper())
chk("hypothèses : z et MS", F(R.zeff, 1) in t and F(R.M_S, 2) in t)
chk("identification et cartouche", "Halle A" in t and "P3/S7" in t and "Bureaud'ÉtudesValens" in s and "21/09/2026" in t)
chk("pied de page en 2 lignes : références (MSB P5) et notations abrégées",
    "EC3 = EN 1993-1-8 sauf indication" in t and "MSB P5" in t
    and "hc, zc : hauteur et position des cornières" in t)
chk("conclusion", "ASSEMBLAGE VÉRIFIÉ à l'ELU" in t)
# --- en-tête compact (finalisation du 21/09/2026)
chk("en-tête ligne 1 : Principale | Secondaire · grugée | Cornières",
    "Principale HEA 400 S355" in t and "Secondaire HEA 300 S355 · grugée" in t
    and "Cornières 2 × L100x100x10 S355" in t)
chk("en-tête ligne 2 : Boulons | Groupe S | Groupe P",
    "Boulons M20 8.8 cat. A · trous Ø22" in t and "Groupe S 3 × 1" in t and "Groupe P 2 × (3 × 1)" in t)
chk("en-tête ligne 3 : les efforts seuls",
    "VEd 125,0 kN" in t and "NEd 0,0 kN" in t and "HEd 0,0 kN" in t and "MEd 0,00 kNm" in t)
chk("hypothèses d'une ligne, références abrégées",
    "Articulé — rotule à la face de l'âme porteuse (MSB P5 §4.2.1.1)." in t
    and "Groupe P : cisaillement centré, 0,80·Fv,Rd (MSB P5 §4.2.1.2)." in t
    and "Grugeage : flexion de la section réduite (MSB P5 §4.2.4)." in t)
# --- phrases supprimées le 21/09/2026 : elles ne reviennent pas
chk("plus de géométrie détaillée en tête (elle est cotée page 2)",
    "e1/p1" not in t and "hc 190 – zc 50" not in t and "gA 55 – p3" not in t)
chk("plus de « Autres vérifications : formules dans le rapport détaillé »", "Autres vérifications" not in t)
chk("plus d'hypothèses longues (répartition élastique, interaction quadratique)",
    "répartition élastique" not in t and "interaction quadratique" not in t)
chk("MSB toujours abrégé « MSB P5 » (jamais « MSB Part 5 »)", "MSB Part 5" not in t and "MSB P5" in t)
# --- aucune information en double sur la page
chk("pas de cartouche sous les dessins (il répétait la ligne de données et les hypothèses)",
    "Cornières : 2 ×" not in t and "Excentricité de calcul" not in t and "Grugeage : sup." not in t
    and "Boulons M20 – classe" not in t)
tt = " ".join(t.split())
chk("la conclusion ne répète ni le taux maximal ni la dimensionnante (déjà au bandeau)",
    "Taux maximal" not in t and tt.count("74,0 %") == 3)     # bandeau, ligne pdS, formule de pdS
chk("MS et z écrits une seule fois en clair (hypothèses)",
    tt.count("MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm") == 1
    and tt.count("z = " + F(R.zeff, 1) + " mm") == 1)
oblig = {R.gov.key} | {c.key for c in R.checks if c.active and not c.ok} | {p.key for _, _, _, p in synthese.taux_par_element(R)}
d = note.Note(R, {"bureau": "x", "date": "", "indice": "0"}, "toutes")
from ndc_pdf.kit import Doc  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
dd = Doc(os.path.join(SCRATCH, "note_essai.pdf"), landscape(A4)); dd.new_page(landscape(A4)); d.construire(dd); dd.save()
chk("formules obligatoires imprimées (dimensionnante, une par élément) + d'autres essentielles",
    oblig <= set(d.formules_imprimees) and len(d.formules_imprimees) > len(oblig), str(d.formules_imprimees))
chk("la substitution de la dimensionnante est sur la page (91,95 kN, 74,0 %)",
    "91,95 kN" in t and "= 74,0 %" in t)
Rn = moteur.compute(dict(V_Ed=400))
tn = pymupdf.open(stream=note.generer_note(Rn, {}), filetype="pdf")[0].get_text()
chk("cas non vérifié : NON VÉRIFIÉ et NON OK sur la page",
    "NON VÉRIFIÉ" in tn and "ASSEMBLAGE NON VÉRIFIÉ" in tn)
Rb = moteur.compute(dict(LC_u=260))
tb = pymupdf.open(stream=note.generer_note(Rb, {}), filetype="pdf")[0].get_text()
chk("alerte bloquante en tête de page, en notation Eurocode",
    "hauteur disponible insuffisante" in tb and "zc + hc = 50 + 260" in tb)

print("\n=== 3. Dessins ===")
chk("cotes de l'élévation et du plan en notation Eurocode (hc, zc, gA, bA)",
    all(x in t for x in ("hc = 190", "zc 50", "gA 55", "bA = 100")) and "Lc = 190" not in t)
chk("noms des profilés", "HEA 400" in t and "HEA 300" in t)

print("\n=== 3 bis. Page 2 : PLAN DE PRINCIPE ===")
p2 = doc[1].get_text()
q2 = p2.replace(" ", "").replace("\n", "")
chk("titre, échelle normalisée affichée — 1:5 sur le cas par défaut, trois vues côte à côte",
    "PLANDEPRINCIPE" in q2 and "Échelle1:5" in q2
    and note.derniere_echelle == 5 and note.derniere_disposition == "trois vues côte à côte",
    f"{note.derniere_echelle} {note.derniere_disposition}")
chk("trois vues titrées : élévation, vue en plan, vue de droite",
    all(x in q2 for x in ("ÉLÉVATION", "VUEENPLAN", "VUEDEDROITE")))
chk("renvois de perçage : 3×Ø22 (groupe S) et 6×Ø22 (groupe P, deux cornières)",
    "3×Ø22" in p2 and "6×Ø22" in p2)
chk("rayon du grugeage renvoyé (r 10) et position du perçage P depuis le dessus de la porteuse (zt = 85)",
    "r 10" in p2 and "zt = 85" in p2)
chk("cotes de fabrication réparties SANS doublon : hc, gA, p3, bS, c une seule fois chacune",
    p2.count("hc = 190") == 1 and p2.count("gA 55") == 1 and p2.count("p3 = 118,5") == 1
    and p2.count("bS = 300") == 1 and p2.count("c = 150") == 1)
chk("cartouche : assemblage, poutres, cornières (hc · zc), fixations avec Ø des trous, date, indice, échelle",
    all(x in q2 for x in ("ASSEMBLAGE", "POUTRES", "CORNIÈRES", "FIXATIONS", "ÉCHELLE"))
    and "trousØ22" in q2 and "hc190·zc50mm" in q2 and "21/09/2026" in p2)
chk("cartouche : la visserie saisie est écrite (rondelles, écrous)",
    "par boulon : 1 rondelle + 1 écrou" in p2)
chk("la page 2 ne porte pas les grandeurs de calcul (z, MS : page 1 seulement)",
    "MS =" not in p2 and "z = 50,0 mm" not in p2)
Rv = moteur.compute(dict())
pv = pymupdf.open(stream=note.generer_note(Rv, {}), filetype="pdf")[1].get_text()
chk("sans visserie fournie, le cartouche porte la composition par défaut",
    "par boulon : 1 rondelle + 1 écrou" in pv)

print("\n=== 4. Depuis l'application ===")
from streamlit.testing.v1 import AppTest  # noqa: E402
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at.session_state["page"] = "Assemblages métalliques"
at.session_state["asm_courant"] = "doubles_cornieres"
at.run()
at.button(key="asm_btn_pdf").click(); at.run()
chk("« Générer PDF » produit la note", not at.exception and bool(at.session_state.get("asm_pdf_bytes")), str(at.exception))
if at.session_state.get("asm_pdf_bytes"):
    da = pymupdf.open(stream=at.session_state["asm_pdf_bytes"], filetype="pdf")
    chk("note de l'application : 2 pages, statut du moteur, plan de principe avec la visserie de l'écran",
        da.page_count == 2 and "VÉRIFIÉ" in da[0].get_text()
        and "par boulon : 1 rondelle + 1 écrou" in da[1].get_text())
at.session_state["asm_ui_onglet"] = "Note"; at.run()
at.button(key="asm_btn_pdf_detail").click(); at.run()
chk("onglet Note : le rapport détaillé (3 pages) reste disponible",
    not at.exception and bool(at.session_state.get("asm_pdf_detail_bytes"))
    and pymupdf.open(stream=at.session_state["asm_pdf_detail_bytes"], filetype="pdf").page_count == 3, str(at.exception))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

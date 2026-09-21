# -*- coding: utf-8 -*-
"""Note de calcul d'UNE page (note.py) — la note du bouton « Générer PDF ».

Chaque garantie rougit si on la retire :
  1. FORME : une seule page A4 paysage, palette 01_encre, sans débordement
     (aucun avertissement) sur les 31 cas de référence ; le corps des
     formules n'est pas réduit sous 6,0 pt ; les tableaux ne s'ajustent
     jamais (la variante ne touche que les formules).
  2. CONTENU : statut, taux maximal, dimensionnante, taux par élément ;
     Ed / Rd de CHAQUE vérification active tels que le moteur les donne ;
     Tableau 3.3 ; hypothèses (z, MS) ; identification ; légende des
     références et des notations ; les formules obligatoires (dimensionnante,
     non vérifiées, une par élément) sont imprimées.
  3. DESSIN : élévation et plan peints, en notation Eurocode (hc, zc, gA).
  4. APPLICATION : « 📄 Générer PDF » produit la note d'une page ; le
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
pdf = note.generer_note(R, {"nom_projet": "Halle A", "partie": "P3/S7", "date": "21/09/2026", "indice": "0"})
with open(os.path.join(SCRATCH, "note_une_page_test.pdf"), "wb") as fh:
    fh.write(pdf)
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("une seule page", doc.page_count == 1, str(doc.page_count))
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
    if dc.page_count != 1 or note.derniers_avertissements:
        nb += 1
        print("      ", cas["id"], dc.page_count, note.derniers_avertissements)
chk("31 cas de référence : une page, sans débordement", nb == 0, str(nb))

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
chk("légendes des références et des notations", "EC3 = EN 1993-1-8 sauf indication" in t and "hc : hauteur des cornières" in t)
chk("conclusion", "ASSEMBLAGE VÉRIFIÉ à l'ELU" in t)
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

print("\n=== 4. Depuis l'application ===")
from streamlit.testing.v1 import AppTest  # noqa: E402
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at.session_state["page"] = "Assemblages métalliques"
at.session_state["asm_courant"] = "doubles_cornieres"
at.run()
at.button(key="asm_btn_pdf").click(); at.run()
chk("« Générer PDF » produit la note d'une page", not at.exception and bool(at.session_state.get("asm_pdf_bytes")), str(at.exception))
if at.session_state.get("asm_pdf_bytes"):
    da = pymupdf.open(stream=at.session_state["asm_pdf_bytes"], filetype="pdf")
    chk("note de l'application : 1 page, statut du moteur", da.page_count == 1 and "VÉRIFIÉ" in da[0].get_text())
at.session_state["asm_ui_onglet"] = "Note"; at.run()
at.button(key="asm_btn_pdf_detail").click(); at.run()
chk("onglet Note : le rapport détaillé (3 pages) reste disponible",
    not at.exception and bool(at.session_state.get("asm_pdf_detail_bytes"))
    and pymupdf.open(stream=at.session_state["asm_pdf_detail_bytes"], filetype="pdf").page_count == 3, str(at.exception))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

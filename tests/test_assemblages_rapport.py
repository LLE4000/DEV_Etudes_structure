# -*- coding: utf-8 -*-
"""Note de calcul PDF de l'assemblage.

Chaque garantie rougit si on la retire :
  1. FORME : garde portrait + 2 planches paysage (A4), palette 01_encre,
     cartouche du bureau, sans débordement (doc.warnings vide) sur le cas
     par défaut et sur des configurations variées (soudée, 2 files, NEd,
     bloquée) — dans les 2 pages de calcul, jamais sous le corps des notes
     béton.
  2. CONTENU : le rapport lit le moteur — pour chaque vérification active,
     la planche de synthèse porte « F(Ed) / F(Rd) » tels que le moteur les
     donne ; le statut, le taux maximal, la dimensionnante y sont ; la
     planche de développement porte la formule des vérifications
     essentielles.
  3. DESSIN : les deux vues sont peintes (textes des cotes présents).
  4. APPLICATION : le bouton « Générer PDF » de l'écran produit la note.
  5. NON-RÉGRESSION : l'extension de ndc_pdf ne change pas l'étalon béton
     (tests/test_export_pdf_ndc.py, lancé séparément).

Lancement : python3 tests/test_assemblages_rapport.py
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402

from acier.formats import F, pct  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import moteur, rapport  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)
cas = {c["id"]: c["inputs"] for c in REF["cas"]}
SCRATCH = os.environ.get("TMPDIR", "/tmp")

print("=== 1. Forme ===")
R = moteur.compute(dict(id_projet="Halle A", id_rep="P3/S7", id_red="X", id_date="21/09/2026"))
pdf = rapport.generer_pdf(R, {"nom_projet": "Halle A", "partie": "P3/S7", "date": "21/09/2026", "indice": "0"})
with open(os.path.join(SCRATCH, "note_assemblage_test.pdf"), "wb") as fh:
    fh.write(pdf)
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("garde + 2 planches", doc.page_count == 3, str(doc.page_count))
chk("garde en portrait, planches en paysage",
    doc[0].rect.width < doc[0].rect.height and doc[1].rect.width > doc[1].rect.height
    and abs(doc[1].rect.width - 841.9) < 1 and abs(doc[1].rect.height - 595.3) < 1)
chk("aucun débordement (variante " + str(rapport.derniere_variante) + ")", not rapport.derniers_avertissements,
    str(rapport.derniers_avertissements))
t0, t1, t2 = doc[0].get_text(), doc[1].get_text(), doc[2].get_text()
# les titres à lettres espacées (interlettrage) s'extraient avec des espaces
s0, s2 = t0.replace(" ", ""), t2.replace(" ", "")
chk("cartouche du bureau et sommaire sur la garde",
    "Bureaud'ÉtudesValens" in s0 and "PROJET" in s0 and "Halle A" in t0 and "SOMMAIRE" in s0 and "synthèse" in t0)
chk("synthèse en 2 colonnes sur le cas par défaut (lisibilité)", rapport.derniere_variante[0] == 2,
    str(rapport.derniere_variante))
chk("titres des planches", "Doubles cornières d'âme – synthèse" in t1 and "développement" in t2)

print("\n=== 2. Contenu lu dans le moteur ===")
manq = []
for c in R.checks:
    if not c.active:
        continue
    nd = 3 if c.unit == "-" else 1
    if (F(c.Ed, nd) + " / " + F(c.Rd, nd)) not in t1:
        manq.append(c.key)
chk("synthèse : Ed / Rd de chaque vérification active, tels que le moteur les donne", not manq, str(manq))
chk("synthèse : statut, taux maximal et dimensionnante",
    "ASSEMBLAGE VÉRIFIÉ" in t1 and pct(R.eta_max, 1) in t1 and rapport.COURT[R.gov.key] in t1)
chk("synthèse : hypothèses (z, MS) et coefficients partiels",
    F(R.zeff, 1) in t1 and F(R.M_S, 2) in t1 and "Coefficients partiels" in t1)
chk("développement : formule de la vérification dimensionnante et sa référence",
    R.gov.formula[:40] in t2.replace("\n", " ") and R.gov.ref[:25] in t2.replace("\n", " "))
chk("développement : paramètres retenus et pinces (Tableau 3.3)",
    "PARAMÈTRESRETENUS" in s2 and "TABLEAU3.3" in s2 and F(R.p_3, 1) in t2)
chk("identification reprise", "P3/S7" in t1 and "Halle A" in t0)

print("\n=== 3. Dessins peints ===")
chk("cotes de l'élévation et du plan présentes (Lc, zc, gA, bA)",
    all(x in t1 for x in ("Lc = 190", "zc 50", "gA 55", "bA = 100")))
chk("noms des profilés sur le dessin", "HEA 400" in t1 and "HEA 300" in t1)
pix = doc[1].get_pixmap(dpi=60)
chk("planche rendue (pixels non blancs)", sum(1 for i in range(0, len(pix.samples), 3 * 97)
                                              if pix.samples[i] < 250) > 200)

print("\n=== 4. Configurations variées : 2 pages de calcul, sans débordement ===")
for ident in ("VAL_B", "deux_files_double_grugeage", "N_traction", "soude_2_cotes", "bloc_Lc260", "options_conservatrices"):
    Rc = moteur.compute(cas[ident])
    pdf = rapport.generer_pdf(Rc, {})
    dc = pymupdf.open(stream=pdf, filetype="pdf")
    chk(f"{ident} : 3 pages, sans débordement, variante {rapport.derniere_variante}",
        dc.page_count == 3 and not rapport.derniers_avertissements, str(rapport.derniers_avertissements)[:200])
    tt = dc[1].get_text()
    chk(f"{ident} : statut du moteur sur la planche",
        ("ASSEMBLAGE VÉRIFIÉ" in tt) == bool(Rc.verified) and ("NON VÉRIFIÉ" in tt) == (not Rc.verified))

print("\n=== 5. Depuis l'application ===")
from streamlit.testing.v1 import AppTest  # noqa: E402
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at.session_state["page"] = "Assemblages métalliques"
at.session_state["asm_courant"] = "doubles_cornieres"
at.run()
at.button(key="asm_btn_pdf").click(); at.run()
chk("le bouton « Générer PDF » produit la note", not at.exception and bool(at.session_state.get("asm_pdf_bytes")),
    str(at.exception))
if at.session_state.get("asm_pdf_bytes"):
    da = pymupdf.open(stream=at.session_state["asm_pdf_bytes"], filetype="pdf")
    chk("note de l'application : 3 pages, planche de synthèse", da.page_count == 3 and "synthèse" in da[1].get_text())

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

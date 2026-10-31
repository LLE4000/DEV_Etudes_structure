# -*- coding: utf-8 -*-
"""Tests du RAPPORT de projet d'assemblages (lots 2 et 3) : page(s) de garde
+ récapitulatif + notes assemblées + signets.

Chaque garantie rougit si on la retire :
  1. RAPPORT : trois assemblages (un par type) → 1 garde portrait, 1 page de
     récapitulatif, puis la note de chacun (pages comptées sur les PDF
     produits) ; sommaire avec repères, pastilles et numéros de page exacts ;
     récapitulatif (repère, type, profilés, taux, dimensionnante, état) ; un
     signet par partie ; le cartouche de chaque note porte le repère.
  2. DÉBORDEMENT : quatorze assemblages → le sommaire continue sur une
     deuxième page de garde, les numéros de page en tiennent compte.
  3. ERREUR : un assemblage dont le calcul échoue donne une page d'erreur et
     une pastille « calcul impossible », le rapport sort quand même.
  4. PAGE PROJET : 📄 génère le rapport (pages, nom de fichier) ; le bouton de
     téléchargement disparaît dès que le projet change (pas de rapport périmé).
  La garde Poutre/Dalle reste identique : tests/test_export_pdf_ndc.py.

Lancement : python3 tests/test_rapport_projet.py (depuis la racine).
"""
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from acier import projet as modele  # noqa: E402
from acier import rapport_projet as rp  # noqa: E402
from acier import registre  # noqa: E402

OK, KO = [], []


def compact(s):
    """Le texte des titres est dessiné avec un espacement de lettres : pymupdf le
    relit « S O M M A I R E » ; on compare sans blancs."""
    return re.sub(r"\s+", "", s)


def dls(at):
    return {d.key for d in at.get("download_button")}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def defauts(type_id, **surcharge):
    m = registre.charger(registre.trouver(type_id)[1])
    u = {k: m.ecran_saisie.valeur_widget(k, v) for k, v in m.defaults().items()}
    u.update(surcharge)
    return u


INFOS = {"nom_projet": "Reamenagement", "partie": "Hall", "date": "30/10/2026", "indice": "B"}

# ================================================================
print("=== 1. Rapport : garde + récapitulatif + notes + signets ===")
p = modele.nouveau()
modele.ajouter(p, "platine_about", defauts("platine_about", V_Ed=180))
modele.ajouter(p, "plat_ame", defauts("plat_ame"))
modele.ajouter(p, "doubles_cornieres", defauts("doubles_cornieres", V_Ed=400))
elems = rp.notes(p, INFOS)
pages = [n for (_e, _info, _pdf, n) in elems]
infos_ = [info for (_e, info, _pdf, _n) in elems]
chk("trois notes produites, deux pages chacune aujourd'hui, états ok / ok / ko, taux et dimensionnante",
    pages == [2, 2, 2] and infos_[2]["etat"] == "ko" and infos_[0]["etat"] in ("ok", "att")
    and infos_[1]["etat"] in ("ok", "att") and infos_[2]["eta"] > 1 and infos_[2]["gov"], f"{pages} {infos_}")
pdf = rp.generer_rapport(p, INFOS)
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("pages = 1 garde + 1 récapitulatif + Σ notes", doc.page_count == 2 + sum(pages), str(doc.page_count))
chk("pages 1 et 2 en portrait, notes en paysage", doc[0].rect.height > doc[0].rect.width
    and doc[1].rect.height > doc[1].rect.width
    and all(doc[i].rect.width > doc[i].rect.height for i in range(2, doc.page_count)))
garde = doc[0].get_text()
chk("garde : bureau, titre, cartouche du projet", "Bureaud'ÉtudesValens" in compact(garde)
    and "Assemblagesmétalliques" in compact(garde) and "Reamenagement" in garde and "Hall" in garde and "30/10/2026" in garde)
chk("sommaire : récapitulatif en tête (sans numéro), repères, types, descriptions", all(s in garde for s in (
    "Récapitulatif des assemblages", "A1 — Platine d'about (hauteur partielle)", "A2 — Plat d'âme soudé (fin plate)",
    "A3 — Doubles cornières d'âme", "V = 180 kN", "V = 400 kN")) and "04\n" not in garde)
chk("sommaire : numéros de page exacts (2, 3, 5, 7)", all(f"page {n}" in garde for n in (2, 3, 5, 7))
    and "page 4" not in garde and "page 6" not in garde)
chk("sommaire : pastilles (VÉRIFIÉ / NON VÉRIFIÉ) et légende", "NON VÉRIFIÉ" in garde and "VÉRIFIÉ" in garde
    and "Rouge : non vérifié" in garde)
recap = doc[1].get_text()
chk("récapitulatif : titre, une ligne par assemblage avec taux, dimensionnante et état",
    "Récapitulatifdesassemblages" in compact(recap) and all(s in recap for s in ("A1", "A2", "A3", "Doubles cornières d'âme"))
    and f"{infos_[2]['eta'] * 100:.0f} %" in recap and "NON VÉRIFIÉ" in recap
    and (infos_[2]["gov"][:20] in recap or "…" in recap), recap[:300])
toc = doc.get_toc()
chk("un signet par partie, sur la première page de chacune",
    toc == [[1, "Récapitulatif", 2], [1, "A1 — Platine d'about (hauteur partielle)", 3],
            [1, "A2 — Plat d'âme soudé (fin plate)", 5], [1, "A3 — Doubles cornières d'âme", 7]], str(toc))
chk("chaque note porte son repère dans son cartouche (partie · repère)",
    "Hall · A1" in doc[2].get_text() and "Hall · A2" in doc[4].get_text() and "Hall · A3" in doc[6].get_text(),
    doc[2].get_text()[:200])
chk("métadonnées du PDF", doc.metadata.get("title") == "Note de calcul — Assemblages métalliques")
chk("infos de la note sans partie projet : le repère seul", rp.infos_note({"nom_projet": "X"}, modele.entrees(p)[0])["partie"] == "A1")

# ================================================================
print("\n=== 2. Débordement du sommaire ===")
p2 = modele.nouveau()
for i in range(14):
    modele.ajouter(p2, "plat_ame", defauts("plat_ame"))
pdf2 = rp.generer_rapport(p2, INFOS)
doc2 = pymupdf.open(stream=pdf2, filetype="pdf")
g1, g2 = doc2[0].get_text(), doc2[1].get_text()
chk("deux pages de garde portrait, « SOMMAIRE (suite) » sur la seconde",
    doc2[0].rect.height > doc2[0].rect.width and doc2[1].rect.height > doc2[1].rect.width
    and "SOMMAIRE(suite)" in compact(g2) and "SOMMAIRE(suite)" not in compact(g1))
chk("récapitulatif + 10 entrées sur la première page, 4 sur la suite", "10\n" in g1 and "11\n" in g2
    and "11\n" not in g1 and "A11" in g2 and "A14" in g2)
chk("récapitulatif page 3, première note page 4, numéros et signets cohérents", "page 3" in g1 and "page 4" in g1
    and "page 30" in g2 and doc2.page_count == 3 + 14 * 2 and doc2.get_toc()[0] == [1, "Récapitulatif", 3]
    and doc2.get_toc()[1][2] == 4 and doc2.get_toc()[-1][2] == 30)
chk("le récapitulatif est bien page 3", "Récapitulatifdesassemblages" in compact(doc2[2].get_text()))

# ================================================================
print("\n=== 3. Un assemblage dont le calcul échoue ===")
p3 = modele.nouveau()
modele.ajouter(p3, "plat_ame", defauts("plat_ame"))
modele.ajouter(p3, "platine_about", defauts("platine_about"))
m_pab = registre.charger(registre.trouver("platine_about")[1])
_compute = m_pab.compute


def _casse(u):
    raise ZeroDivisionError("division par zéro simulée")


m_pab.compute = _casse
try:
    pdf3 = rp.generer_rapport(p3, INFOS)
finally:
    m_pab.compute = _compute
doc3 = pymupdf.open(stream=pdf3, filetype="pdf")
chk("le rapport sort : garde + récapitulatif + note A1 (2 p.) + page d'erreur A2 (1 p.)", doc3.page_count == 5
    and "CALCULIMPOSSIBLE" in compact(doc3[4].get_text()) and "ZeroDivisionError" in doc3[4].get_text())
chk("sommaire et récapitulatif : pastille « CALCUL IMPOSSIBLE », page 5 pour A2",
    "CALCUL IMPOSSIBLE" in doc3[0].get_text() and "page 5" in doc3[0].get_text() and doc3.get_toc()[2][2] == 5
    and "CALCUL IMPOSSIBLE" in doc3[1].get_text() and "—" in doc3[1].get_text())

# ================================================================
print("\n=== 4. Page projet : 📄 Générer PDF ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=300)
at.session_state["page"] = "Assemblages métalliques"
at.run()
chk("projet vide : 📄 désactivé", at.button(key="prj_btn_pdf").disabled)
at.button(key="asm_btn_open_platine_about").click(); at.run()
at.button(key="pab_btn_retour").click(); at.run()
at.button(key="prj_btn_nouveau").click(); at.run()
at.button(key="asm_btn_open_plat_ame").click(); at.run()
at.button(key="fpl_btn_retour").click(); at.run()
at.session_state["nom_projet"] = "Reamenagement"; at.session_state["partie"] = "Hall"
at.session_state["indice"] = "B"; at.session_state["date"] = "30/10/2026"
at.run()
at.button(key="prj_btn_pdf").click(); at.run()
octets = at.session_state.get("prj_pdf_bytes")
chk("📄 : rapport généré, 6 pages, sans exception", not at.exception and bool(octets)
    and pymupdf.open(stream=octets, filetype="pdf").page_count == 6, str(at.exception)[:200])
chk("bouton de téléchargement présent", "prj_btn_pdf_dl" in dls(at))
at.button(key="prj_btn_dup_1").click(); at.run()
chk("le projet change → le rapport périmé n'est plus proposé", "prj_btn_pdf_dl" not in dls(at)
    and at.session_state.get("prj_pdf_bytes"))
chk("nom de fichier du rapport", modele.nom_fichier({"nom_projet": "Reamenagement", "partie": "Hall", "date": "30/10/2026",
                                                     "indice": "B"}, ".pdf") == "REA_Assemblages Hall#B_30-10-2026.pdf")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

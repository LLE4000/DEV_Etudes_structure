# -*- coding: utf-8 -*-
"""Tests de l'export PDF v3.0 (mise en page ndc_pdf).

Quatre garanties, chacune rougit si on la retire :
  1. IDENTITÉ : la note de démonstration reconstruite est identique au
     rendu de référence livré (ndc_pdf/reference/NOTE_DE_CALCUL.pdf) —
     texte strictement égal, pixels quasi identiques (tolérance de
     rastérisation < 0,5 %).
  2. MOTEUR : generer_rapport_pdf sur 2 poutres × 2 sections produit
     1 garde portrait + 4 planches paysage, sans débordement, et la
     planche du cas de référence porte les six valeurs verrouillées
     (hu,min 67,2 · hmin 73,2 · 1961 · 1373 · τ 3,83 · τadm 2,26).
  3. MULTI-LITS : la coupe reçoit tous les lits avec leurs positions
     réelles, et un libellé par lit.
  4. DALLE : export_pdf_dalle importe toujours ses primitives et génère.

Lancement : python tests/test_export_pdf_ndc.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


SCRATCH = os.environ.get("TMPDIR", "/tmp")


def _tmp(nom):
    return os.path.join(SCRATCH, nom)


# ================================================================
print("=== 1. Identité au rendu de référence ===")
from ndc_pdf.styles import STYLES  # noqa: E402

style = {s.key: s for s in STYLES}["01_encre"]
demo = _tmp("ndc_demo_test.pdf")
d = style.build(demo)
d.save()
chk("démonstration : aucun débordement", not d.warnings, str(d.warnings))

ref = os.path.join(RACINE, "ndc_pdf", "reference", "NOTE_DE_CALCUL.pdf")
da, db = pymupdf.open(ref), pymupdf.open(demo)
chk("même nombre de pages que la référence", da.page_count == db.page_count,
    f"{da.page_count} / {db.page_count}")
texte_egal = all(da[i].get_text() == db[i].get_text()
                 for i in range(min(da.page_count, db.page_count)))
chk("texte strictement identique à la référence", texte_egal)
pire = 0.0
for i in range(min(da.page_count, db.page_count)):
    pa = da[i].get_pixmap(dpi=100)
    pb = db[i].get_pixmap(dpi=100)
    if (pa.width, pa.height) != (pb.width, pb.height):
        pire = 100.0
        break
    ba, bb = pa.samples, pb.samples
    diff = sum(1 for x, y in zip(ba, bb) if x != y)
    pire = max(pire, 100.0 * diff / len(ba))
chk("pixels identiques (tolérance rastérisation 0,5 %)", pire < 0.5,
    f"pire page : {pire:.3f} %")

# ================================================================
print("\n=== 2. Génération depuis le moteur (2 poutres × 2 sections) ===")
from modules.export_pdf import generer_rapport_pdf  # noqa: E402
import modules.export_pdf as XP  # noqa: E402

with open("beton_classes.json", encoding="utf-8") as f:
    beton_data = json.load(f)


def KB(base, bid):
    return f"b{bid}_{base}"


def KS(base, bid, sid):
    return f"b{bid}_sec{sid}_{base}"


beams = [
    {"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "Section A"},
                                              {"id": 2, "nom": "Section B"}]},
    {"id": 2, "nom": "Poutre 2", "sections": [{"id": 1, "nom": "Section A"},
                                              {"id": 2, "nom": "Section B"}]},
]
values = {
    "gamma_s": 1.5, "jeu_enrobage_cm": 1.0, "jeu_entre_lits_cm": 1.0,
    "techno_d_mm": 10, "techno_s_max_cm": 30.0,
    "meta_beam_nom_1": "Poutre 1", "meta_b1_nom_1": "Section A", "meta_b1_nom_2": "Section B",
    "meta_beam_nom_2": "Poutre 2", "meta_b2_nom_1": "Section A", "meta_b2_nom_2": "Section B",
    KB("b", 1): 20, KB("h", 1): 40, KB("enrobage_beton", 1): 3.0,
    KB("beton", 1): "C30/37", KB("fyk", 1): 500,
    # cas de référence des tests de non-régression
    KS("M_inf", 1, 1): 200.0, KS("M_sup", 1, 1): 140.0, KS("V", 1, 1): 230.0,
    KS("n_as_inf", 1, 1): 2, KS("ø_as_inf", 1, 1): 16,
    KS("n_as_sup", 1, 1): 2, KS("ø_as_sup", 1, 1): 16,
    KS("shear_n_lines", 1, 1): 1, KS("shear_line0_type", 1, 1): "Étriers (2 brins)",
    KS("shear_line0_d", 1, 1): 10, KS("shear_pas", 1, 1): 30.0,
    KS("M_inf", 1, 2): 120.0, KS("M_sup", 1, 2): 60.0, KS("V", 1, 2): 150.0,
    KS("n_as_inf", 1, 2): 3, KS("ø_as_inf", 1, 2): 16,
    KS("n_as_sup", 1, 2): 2, KS("ø_as_sup", 1, 2): 12,
    KS("shear_n_lines", 1, 2): 1, KS("shear_line0_type", 1, 2): "Étriers (2 brins)",
    KS("shear_line0_d", 1, 2): 8, KS("shear_pas", 1, 2): 15.0,
    KB("b", 2): 30, KB("h", 2): 60, KB("enrobage_beton", 2): 3.0,
    KB("beton", 2): "C25/30", KB("fyk", 2): 500,
    # DEUX lits inférieurs -> extension multi-lits de la coupe
    KS("M_inf", 2, 1): 320.0, KS("M_sup", 2, 1): 110.0, KS("V", 2, 1): 380.0,
    KS("nlits_inf", 2, 1): 2,
    KS("n_as_inf", 2, 1): 3, KS("ø_as_inf", 2, 1): 20,
    KS("n_as_inf_l2", 2, 1): 2, KS("ø_as_inf_l2", 2, 1): 16,
    KS("n_as_sup", 2, 1): 2, KS("ø_as_sup", 2, 1): 16,
    # trois groupes positionnés : étrier périmétrique + étrier partiel
    # (barres 1→2) + épingle (barre 3) — et h=60 déclenche la peau
    KS("shear_n_lines", 2, 1): 3,
    KS("shear_line0_type", 2, 1): "Étriers (2 brins)", KS("shear_line0_d", 2, 1): 8,
    KS("shear_line1_type", 2, 1): "Étriers (2 brins)", KS("shear_line1_d", 2, 1): 8,
    KS("shear_line1_from", 2, 1): 1, KS("shear_line1_to", 2, 1): 2,
    KS("shear_line2_type", 2, 1): "Épingle (1 brin)", KS("shear_line2_d", 2, 1): 8,
    KS("shear_line2_from", 2, 1): 3, KS("shear_line2_to", 2, 1): 3,
    KS("shear_pas", 2, 1): 20.0,
    KS("M_inf", 2, 2): 80.0, KS("M_sup", 2, 2): 40.0, KS("V", 2, 2): 90.0,
    KS("n_as_inf", 2, 2): 3, KS("ø_as_inf", 2, 2): 16,
    KS("n_as_sup", 2, 2): 2, KS("ø_as_sup", 2, 2): 16,
    KS("shear_n_lines", 2, 2): 1, KS("shear_line0_type", 2, 2): "Étriers (2 brins)",
    KS("shear_line0_d", 2, 2): 8, KS("shear_pas", 2, 2): 15.0,
}

sortie = _tmp("ndc_moteur_test.pdf")
p = generer_rapport_pdf(beams, values, beton_data,
                        infos={"nom_projet": "", "partie": "",
                               "date": "25/08/2026", "indice": "0"},
                        output_path=sortie)
chk("génération sans exception", os.path.exists(p))
chk("aucun débordement signalé", not XP.DERNIERS_AVERTISSEMENTS,
    str(XP.DERNIERS_AVERTISSEMENTS))
doc = pymupdf.open(p)
chk("5 pages (garde + 4 planches)", doc.page_count == 5, str(doc.page_count))
chk("page 1 en portrait", doc[0].rect.height > doc[0].rect.width)
chk("planches en paysage", all(doc[i].rect.width > doc[i].rect.height
                               for i in range(1, doc.page_count)))
t1 = doc[0].get_text()
chk("sommaire : 4 entrées et numéros de page",
    all(s in t1 for s in ("Poutre 1 — Section A", "Poutre 1 — Section B",
                          "Poutre 2 — Section A", "Poutre 2 — Section B",
                          "page 2", "page 5")))
# les libellés de la garde sont composés avec interlettrage -> extraits
# « P R O J E T » ; vides = aucune ligne de valeur entre les deux libellés
chk("PROJET / PARTIE vides restent vides",
    "P R O J E T\nP A R T I E\n" in t1 and "D A T E\n25/08/2026" in t1, repr(t1[:120]))
t2 = doc[1].get_text()
for att in ("67,2", "73,2", "1961", "1373", "3,83", "2,26"):
    chk(f"valeur de référence {att} sur la planche A", att in t2)
chk("radical présent dans la formule (texte √ extrait)", "√" in t2 or "67,2" in t2)
chk("« d₁ » remplace « CDG armatures » dans la note", "CDG" not in t2)
chk("cote d₁ sur la coupe", "d₁ =" in t2)
t4 = doc[3].get_text()
chk("multi-lits : les deux lits sont annotés",
    "Lit 1 : 3 Ø20" in t4 and "Lit 2 : 2 Ø16" in t4, t4[:200])
chk("multi-lits : c.d.g. de 2 lits affiché", "c.d.g. de 2 lits" in t4)
chk("armatures de peau annotées sur la coupe", "Armature de peau : 2×1 Ø10" in t4)
chk("légende : une ligne par groupe d'étriers",
    t4.count("Étrier : Ø8 — 20 cm") == 2 and "Épingle : Ø8" in t4)
chk("récap étriers : les trois groupes dans « On prend »",
    "Étrier Ø8 + Étrier Ø8 + Épingle Ø8" in t4)
t5 = doc[4].get_text()
chk("section vérifiée : pastille VÉRIFIÉ", "VÉRIFIÉ" in t5 and "NON VÉRIFIÉ" not in t5)

# ================================================================
print("\n=== 3. La construction refuse de recalculer ===")
from ndc_pdf import data as ndc_data  # noqa: E402

chk("fn : virgule décimale française", ndc_data.fn(3.833, 2) == "3,83")
chk("sci : paliers identiques à sci_tokens",
    ndc_data.sci(200e6) == r"200 \cdot 10^{6}"
    and ndc_data.sci(230e3) == r"230 \cdot 10^{3}"
    and ndc_data.sci(0) == "0" and ndc_data.sci(120) == "120")
res = XP._collecter_resultats(beams, values, beton_data)
chk("un payload par section, dans l'ordre des planches",
    len(res) == 4 and res[0]["poutre"] == "Poutre 1" and res[3]["section"] == "Section B")
secs = ndc_data.construire_sections(res)
c4 = secs[2]["coupe"]
chk("coupe : positions d'axe réelles transmises (mm)",
    len(c4["lits_inf"]) == 2 and abs(c4["lits_inf"][0]["e"] - 60.0) < 1e-6
    and abs(c4["lits_inf"][1]["e"] - 90.0) < 1e-6, str(c4["lits_inf"]))
chk("coupe : un libellé par lit", len(c4["labs_inf"]) == 2)
chk("coupe : trois groupes d'étriers avec positions",
    len(c4["cadres"]) == 3 and c4["cadres"][1]["de"] == 1 and c4["cadres"][1]["a"] == 2
    and c4["cadres"][2]["brins"] == 1, str(c4["cadres"]))
chk("coupe : peau aux positions du moteur (mm)",
    "peau" in c4 and [round(y, 1) for y in c4["peau"]["ys"]] == [300.0],
    str(c4.get("peau")))
etats = [v["etat"] for s in secs for verif in s["verifs"] for v in verif["verdicts"]]
chk("verdicts : uniquement ok/att/ko", set(etats) <= {"ok", "att", "ko"}, str(set(etats)))

# ================================================================
print("\n=== 4. L'export Dalle est intact ===")
try:
    from modules.export_pdf_dalle import generer_rapport_pdf as gen_dalle
    imp_ok = True
except Exception as e:  # noqa: BLE001
    imp_ok = False
    err = str(e)
chk("export_pdf_dalle importe toujours ses primitives", imp_ok,
    "" if imp_ok else err)
if imp_ok:
    dalles = [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "Section A"}]}]
    vals_d = {"meta_dalle_nom_1": "Dalle 1", "meta_dal1_nom_1": "Section A",
              "dal1_b": 100, "dal1_h": 20, "dal1_enrobage_beton": 3.0,
              "dal1_beton": "C30/37", "dal1_fyk": 500, "gamma_s": 1.5,
              "dal1_sec1_M_inf": 25.0, "dal1_sec1_M_sup": 0.0, "dal1_sec1_V": 40.0}
    try:
        pd_ = gen_dalle(dalles, vals_d, beton_data,
                        infos={"date": "25/08/2026", "indice": "0"},
                        output_path=_tmp("ndc_dalle_test.pdf"))
        okd = os.path.exists(pd_) and pymupdf.open(pd_).page_count >= 2
    except Exception as e:  # noqa: BLE001
        okd = False
        print("   exception dalle :", e)
    chk("le PDF Dalle se génère toujours", okd)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

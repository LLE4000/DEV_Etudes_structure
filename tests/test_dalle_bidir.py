# -*- coding: utf-8 -*-
"""Tests de la dalle BIDIRECTIONNELLE (dalle.py 2.1 + note ndc_pdf).

Garanties, chacune rougit si on la retire :
  1. CAS COMPLET (Mx inf/sup, My inf/sup, Vmax tous ≠ 0) : les QUATRE
     familles d'armatures sont calculées indépendamment, avec les
     valeurs de référence recalculées à la main :
       My,inf = 35 kN·m, d = 15,5 cm -> Aₛ,req = 753 mm²
       Mx,inf = 28 kN·m              -> 602 mm²
       Mx,sup = 12 kN·m              -> 258 mm²
       My,sup =  9 kN·m              -> 194 mm²
     La direction PRINCIPALE est un CHOIX utilisateur (défaut Y) et
     s'affiche d'abord ; la bascule vers X réordonne tout.
  2. MIGRATION v1 -> v2 : un ancien état (M_inf/M_sup + couches inf/sup)
     devient la direction X, renforts compris, anciennes clés purgées.
  3. NOTE DE CALCUL : le bouton PDF de l'application produit une garde
     + UNE planche par section, dans l'ordre imposé (hauteur, inf P,
     inf S, sup P, sup S, tranchant τ seul — sans étriers), avec les
     schémas par direction et « bande = 1,00 m ».
  4. REPLI v1 de l'export : un dict de valeurs non migré reste lisible.

Lancement : python tests/test_dalle_bidir.py (depuis la racine).
"""
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def app():
    from modules import dalle
    dalle.show()


def md(at):
    return "\n".join(str(m.value) for m in at.markdown)


# ================================================================
print("=== 1. Cas complet : 4 familles indépendantes ===")
at = AppTest.from_function(app, default_timeout=180)
at.run()
at.text_input(key="dal1_sec1_Mx_inf_raw").set_value("28,00")
at.text_input(key="dal1_sec1_Mx_sup_raw").set_value("12,00")
at.text_input(key="dal1_sec1_My_inf_raw").set_value("35,00")
at.text_input(key="dal1_sec1_My_sup_raw").set_value("9,00")
at.text_input(key="dal1_sec1_V_raw").set_value("60,00")
at.run(); at.run()
chk("aucune exception", not at.exception, str(at.exception))
t = md(at)
for attendu in ("Armatures inférieures — dir. Y (principale)",
                "Armatures supérieures — dir. Y (principale)",
                "Armatures inférieures — dir. X (secondaire)",
                "Armatures supérieures — dir. X (secondaire)"):
    chk(f"carte « {attendu[22:]} »", attendu in t)
chk("principale (Y) affichée avant la secondaire",
    0 <= t.find("dir. Y (principale)") < t.find("dir. X (secondaire)"))
reqs = re.findall(r"Aₛ,req = (\d+) mm²", t)
chk("Aₛ,req des 4 familles, indépendants (753/194/602/258)",
    reqs[:4] == ["753", "194", "602", "258"], str(reqs[:4]))
chk("hauteur en 2 lignes : « hᵤ,min + d₁ … ≤ h » sur une seule ligne",
    re.search(r"hᵤ,min \+ d₁ = [\d,]+ \+ [\d,]+ = \*\*[\d,]+ cm\*\* [≤>] h", t) is not None)
chk("hauteur : M max des 4 moments (35 -> hᵤ,min 12,6 cm)", "**12,6 cm**" in t)

print("\n=== 2. Migration v1 -> v2 ===")
a2 = AppTest.from_function(app, default_timeout=180)
a2.session_state["dalles"] = [{"id": 1, "nom": "Dalle 1",
                               "sections": [{"id": 1, "nom": "A"}]}]
for k, v in {
    "dal1_sec1_M_inf": 28.0, "dal1_sec1_M_sup": 12.0, "dal1_sec1_V": 55.0,
    "dal1_sec1_ncouches_inf": 2,
    "dal1_sec1_arm_type_inf_c1": "Treillis",
    "dal1_sec1_treillis_inf_c1": "10/10/100/100",
    "dal1_sec1_arm_type_inf_c2": "Barres",
    "dal1_sec1_ø_barres_inf_c2": 12, "dal1_sec1_esp_barres_inf_c2": 150,
    "dal1_sec1_ncouches_sup": 1,
    "dal1_sec1_treillis_sup_c1": "8/8/150/150",
}.items():
    a2.session_state[k] = v
a2.run()
chk("migration : aucune exception", not a2.exception, str(a2.exception))
ss = a2.session_state
chk("M_inf -> Mx_inf, M_sup -> Mx_sup",
    ss["dal1_sec1_Mx_inf"] == 28.0 and ss["dal1_sec1_Mx_sup"] == 12.0)
chk("couches v1 -> direction X (renfort compris)",
    ss["dal1_sec1_ncouches_inf_x"] == 2
    and ss["dal1_sec1_treillis_inf_x_c1"] == "10/10/100/100"
    and ss["dal1_sec1_ø_barres_inf_x_c2"] == 12
    and ss["dal1_sec1_treillis_sup_x_c1"] == "8/8/150/150")
chk("anciennes clés purgées",
    all(k not in ss.filtered_state for k in
        ("dal1_sec1_M_inf", "dal1_sec1_ncouches_inf", "dal1_sec1_treillis_inf_c1")))
chk("direction Y sur ses défauts", ss["dal1_sec1_ncouches_inf_y"] == 1)

print("\n=== 3. Note de calcul par le bouton de l'application ===")
a3 = AppTest.from_function(app, default_timeout=240)
a3.run()
a3.text_input(key="dal1_sec1_Mx_inf_raw").set_value("28,00")
a3.text_input(key="dal1_sec1_My_inf_raw").set_value("35,00")
a3.text_input(key="dal1_sec1_My_sup_raw").set_value("9,00")
a3.text_input(key="dal1_sec1_Mx_sup_raw").set_value("12,00")
a3.text_input(key="dal1_sec1_V_raw").set_value("60,00")
a3.run(); a3.run()
a3.button(key="dalle_btn_pdf").click()
a3.run()
chk("génération PDF sans exception", not a3.exception, str(a3.exception))
pdf = a3.session_state["dalle_pdf_bytes"]
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("garde + UNE planche par section (v2.1)", doc.page_count == 2, str(doc.page_count))
chk("garde en portrait, planche en paysage",
    doc[0].rect.height > doc[0].rect.width
    and doc[1].rect.width > doc[1].rect.height)
t2 = doc[1].get_text()
chk("ordre : hauteur, inf P, inf S, sup P, sup S, tranchant",
    0 <= t2.find("Vérification de la hauteur")
    < t2.find("Armatures inférieures — direction Y (principale)")
    < t2.find("Armatures inférieures — direction X (secondaire)")
    < t2.find("Armatures supérieures — direction Y (principale)")
    < t2.find("Armatures supérieures — direction X (secondaire)")
    < t2.find("Vérification de l'effort tranchant"))
chk("les 4 moments et V max sur la planche",
    all(v in t2 for v in ("28,0", "12,0", "35,0", "9,0", "60,0")))
chk("coupe : bande de 1,00 m", "bande = 1,00 m" in t2)
# armatures par défaut identiques en X et Y -> UN SEUL schéma (point 8)
chk("directions identiques : un seul schéma combiné",
    "Directions Y et X (identiques)" in t2 and "Direction secondaire" not in t2)

# on différencie la direction X -> DEUX schémas titrés
a3.selectbox(key="dal1_sec1_treillis_sup_x_c1").set_value("8/8/150/150")
a3.run()
a3.button(key="dalle_btn_pdf").click()
a3.run()
t2b = pymupdf.open(stream=a3.session_state["dalle_pdf_bytes"], filetype="pdf")[1].get_text()
chk("directions différentes : deux schémas titrés par direction",
    "Direction principale : Y" in t2b and "Direction secondaire : X" in t2b
    and "en points : dir. X" in t2b)
chk("tranchant SANS étriers : τ contre τ_adm,I seulement",
    "adm,I" in t2 and "Étrier" not in t2 and "Pas théorique" not in t2)
chk("Aₛ,req indépendants dans la note (753 principale / 602 secondaire)",
    "753" in t2 and "602" in t2)

print("\n=== 3b. Bascule de la direction principale vers X ===")
a3.selectbox(key="dal1_dir_principale").set_value("X")
a3.run()
t = "\n".join(str(m.value) for m in a3.markdown)
chk("X devient principale à l'écran (malgré My > Mx)",
    "dir. X (principale)" in t and "dir. Y (secondaire)" in t
    and t.find("dir. X (principale)") < t.find("dir. Y (secondaire)"))
a3.button(key="dalle_btn_pdf").click()
a3.run()
tX = pymupdf.open(stream=a3.session_state["dalle_pdf_bytes"], filetype="pdf")[1].get_text()
chk("X principale aussi dans la note",
    "direction X (principale)" in tX and "Direction principale : X" in tX
    and "Direction secondaire : Y" in tX)

print("\n=== 4. Export : repli sur un dict de valeurs v1 ===")
from modules.export_pdf_dalle import generer_rapport_pdf  # noqa: E402
with open("beton_classes.json", encoding="utf-8") as f:
    bd = json.load(f)
dalles = [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]}]
v1 = {"meta_dalle_nom_1": "Dalle 1", "meta_dal1_nom_1": "A",
      "dal1_b": 100, "dal1_h": 20, "dal1_enrobage_beton": 3.0,
      "dal1_beton": "C30/37", "dal1_fyk": 500, "gamma_s": 1.5,
      "dal1_sec1_M_inf": 25.0, "dal1_sec1_M_sup": 0.0, "dal1_sec1_V": 40.0,
      "dal1_sec1_ncouches_inf": 1, "dal1_sec1_treillis_inf_c1": "10/10/100/100"}
sortie = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ndc_dalle_v1_test.pdf")
p = generer_rapport_pdf(dalles, v1, bd, infos={"date": "31/08/2026"}, output_path=sortie)
docv1 = pymupdf.open(p)
tv1 = docv1[1].get_text()
chk("un dict v1 se génère sans migration préalable",
    docv1.page_count == 2 and "25,0" in tv1, str(docv1.page_count))
chk("v1 : l'ancien M_inf devient la direction X (Y principale par défaut)",
    "direction X (secondaire)" in tv1)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

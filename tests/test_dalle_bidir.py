# -*- coding: utf-8 -*-
"""Tests de la dalle BIDIRECTIONNELLE (dalle.py 2.1 + note ndc_pdf).

Garanties, chacune rougit si on la retire :
  1. CAS COMPLET (Mx inf/sup, My inf/sup, Vmax tous ≠ 0) : les QUATRE
     familles d'armatures sont calculées indépendamment, avec les
     valeurs de référence recalculées à la main :
       (jeu premier lit : 0 par défaut -> dist. d'axe Ø10 = 3,5 cm)
       My,inf = 35 kN·m, d = 16,5 cm -> Aₛ,req = 707 mm²
       Mx,inf = 28 kN·m              -> 566 mm²
       Mx,sup = 12 kN·m              -> 242 mm²
       My,sup =  9 kN·m              -> 182 mm²
     La direction PRINCIPALE est un CHOIX utilisateur (défaut Y) et
     s'affiche d'abord ; la bascule vers X réordonne tout.
  2. MIGRATION v1 -> v2 : un ancien état (M_inf/M_sup + couches inf/sup)
     devient la direction X, renforts compris, anciennes clés purgées.
  3. NOTE DE CALCUL : le bouton PDF de l'application produit une garde
     + UNE planche par section, dans l'ordre imposé (hauteur, inf P,
     inf S, sup P, sup S, tranchant τ seul — sans étriers), avec les
     schémas par direction (COUPE PERPENDICULAIRE : la direction montrée
     en points, l'autre filante) et « bande = 1,00 m ». Les titres disent
     « direction principale » SANS la lettre (le libellé de moment la
     porte juste dessous — retour bureau du 31/08).
  4. REPLI v1 de l'export : un dict de valeurs non migré reste lisible.
  5. NIVEAUX : une même couche d'armatures est au MÊME niveau dans les
     deux schémas (filante dans l'un, en points dans l'autre), et deux
     axes ne se confondent jamais (écart >= (Øa+Øb)/2).
  6. RENFORTS « n barres » (3 Ø12 : As = n·aire/(b/100)) et DISTANCE
     D'AXE SAISIE par couche : le CDG pondéré, Aₛ,req, les schémas et
     la note suivent la saisie ; champ vidé = retour à l'automatique.

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
chk("Aₛ,req des 4 familles, indépendants (707/182/566/242 — jeu 0 par défaut)",
    reqs[:4] == ["707", "182", "566", "242"], str(reqs[:4]))
chk("hauteur en 2 lignes : « hᵤ,min + d₁ … ≤ h » sur une seule ligne",
    re.search(r"hᵤ,min \+ d₁ = [\d,]+ \+ [\d,]+ = \*\*[\d,]+ cm\*\* [≤>] h", t) is not None)
chk("hauteur : M max des 4 moments (35 -> hᵤ,min 12,6 cm)", "**12,6 cm**" in t)
# ergonomie du tableau des couches (retour bureau du 31/08) : pas de
# colonne As (mm²/m) — redondante avec « Aₛ fourni » — et l'ajout de
# renfort est un « ＋ » seul (le libellé long débordait sur deux lignes)
chk("tableau des couches : la colonne As (mm²/m) a disparu",
    not any("as_disp" in (ti.key or "") for ti in at.text_input))
chk("bouton d'ajout de renfort : un « ＋ » seul",
    at.button(key="dal1_sec1_btn_add_couche_inf_y").label == "＋")

print("\n=== 1b. Renfort « n barres » et niveau saisi par couche ===")
# 3 Ø12 posées dans la bande : As = 3·113,1 = 339 mm²/m, qui s'ajoute
# au treillis (785) -> 1125 mm²/m ; puis la couche est posée à 8,0 cm
# d'axe -> le CDG pondéré passe à 4,9 cm et Aₛ,req suit (d = 15,1 cm).
at.session_state["dal1_sec1_ncouches_inf_y"] = 2
at.session_state["dal1_sec1_arm_type_inf_y_c2"] = "n barres"
at.session_state["dal1_sec1_n_barres_inf_y_c2"] = 3
at.session_state["dal1_sec1_ø_barres_inf_y_c2"] = 12
at.run()
chk("« n barres » : aucune exception", not at.exception, str(at.exception))
t = md(at)
chk("détail : « Treillis 10/10/100/100 + 3 Ø12 »",
    "Treillis 10/10/100/100 + 3 Ø12" in t)
chk("Aₛ fourni = 785 + 339 = 1125 mm²/m", "Aₛ fourni = 1125 mm²/m" in t)
at.text_input(key="dal1_sec1_dist_axe_inf_y_c2").set_value("8,0")
at.run()
t = md(at)
chk("niveau saisi : CDG pondéré recalculé (3,5/8,0 -> 4,9 cm)",
    at.text_input(key="dal1_sec1_ycdg_inf_y").value == "4,9")
chk("Aₛ,req inf. principale suit le d réduit (770 mm²)",
    re.search(r"Aₛ,req = 770 mm²", t) is not None,
    str(re.findall(r"Aₛ,req = (\d+) mm²", t)[:2]))
# champ vidé -> retour à l'automatique (Ø12, jeu 0 : 3,0 + 1,0 = 4,0 cm)
at.text_input(key="dal1_sec1_dist_axe_inf_y_c2").set_value("")
at.run()
chk("champ vidé : retour à la distance automatique",
    at.text_input(key="dal1_sec1_dist_axe_inf_y_c2").value == "4,0")
chk("jeu premier lit : 0 par défaut (retour bureau du 01/09)",
    float(at.session_state["jeu_enrobage_cm"]) == 0.0)

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
    < t2.find("Armatures inférieures — direction principale")
    < t2.find("Armatures inférieures — direction secondaire")
    < t2.find("Armatures supérieures — direction principale")
    < t2.find("Armatures supérieures — direction secondaire")
    < t2.find("Vérification de l'effort tranchant"))
chk("titres sans lettre — elle vit dans les libellés de moments (Y avant X)",
    "direction Y (principale)" not in t2
    and 0 <= t2.find("Moment inférieur Y") < t2.find("Moment inférieur X")
    < t2.find("Moment supérieur Y") < t2.find("Moment supérieur X"))
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
    and "filants : dir. X" in t2b)
chk("tranchant SANS étriers : τ contre τ_adm,I seulement",
    "adm,I" in t2 and "Étrier" not in t2 and "Pas théorique" not in t2)
chk("Aₛ,req indépendants dans la note (707 principale / 566 secondaire)",
    "707" in t2 and "566" in t2)
chk("échelle NORMALISÉE, juste sur l'A4 (1:25, plus de 1:21)",
    "éch. horiz. 1:25" in t2)

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
chk("X principale aussi dans la note (moments X avant Y, schémas retitrés)",
    0 <= tX.find("Moment inférieur X") < tX.find("Moment inférieur Y")
    and "Direction principale : X" in tX and "Direction secondaire : Y" in tX)

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
    "direction secondaire" in tv1
    and 0 <= tv1.find("Moment inférieur Y") < tv1.find("Moment inférieur X"))

print("\n=== 5. Niveaux d'armatures identiques entre les deux schémas ===")
# Le cas du retour bureau : un RENFORT Ø12 en Y inf, X différencié —
# la barre doit être au même niveau vue filante (schéma principal) et
# vue en points (schéma secondaire).
from modules.export_pdf_dalle import _collecter_resultats  # noqa: E402
from ndc_pdf.data import _coupe_dalle_depuis_R  # noqa: E402
v2 = {"meta_dalle_nom_1": "Dalle 1", "meta_dal1_nom_1": "A",
      "dal1_b": 100, "dal1_h": 20, "dal1_enrobage_beton": 3.0,
      "dal1_beton": "C30/37", "dal1_fyk": 500, "gamma_s": 1.5,
      "dal1_dir_principale": "Y",
      "dal1_sec1_Mx_inf": 28.0, "dal1_sec1_Mx_sup": 12.0,
      "dal1_sec1_My_inf": 35.0, "dal1_sec1_My_sup": 9.0, "dal1_sec1_V": 60.0,
      "dal1_sec1_ncouches_inf_y": 2,
      "dal1_sec1_arm_type_inf_y_c1": "Treillis",
      "dal1_sec1_treillis_inf_y_c1": "10/10/100/100",
      # renfort « n barres » posé à un niveau SAISI (8,0 cm d'axe)
      "dal1_sec1_arm_type_inf_y_c2": "n barres",
      "dal1_sec1_ø_barres_inf_y_c2": 12, "dal1_sec1_n_barres_inf_y_c2": 3,
      "dal1_sec1_dist_auto_inf_y_c2": False,
      "dal1_sec1_dist_axe_inf_y_c2": "8,0",
      "dal1_sec1_treillis_sup_y_c1": "8/8/150/150",
      "dal1_sec1_arm_type_sup_x_c1": "Barres",
      "dal1_sec1_ø_barres_sup_x_c1": 10, "dal1_sec1_esp_barres_sup_x_c1": 200}
R5 = _collecter_resultats(dalles, v2, bd)[0]["R"]
geo5 = R5["dirs"]["y"]["geo_inf"]
chk("export : couche « n barres » = 3 Ø12 -> 339 mm²/m",
    geo5["couches"][1]["valeur"] == "3 Ø12"
    and abs(geo5["couches"][1]["As_pm"] - 339.29) < 0.5)
chk("export : niveau saisi respecté (e = 8,0 cm) et CDG pondéré",
    abs(geo5["couches"][1]["e"] - 8.0) < 1e-9
    and abs(geo5["e_cdg"] - 4.858) < 0.01)
coupe = _coupe_dalle_depuis_R(R5)
chk("cas différencié : deux schémas", len(coupe["schemas"]) == 2)
sch_p, sch_s = coupe["schemas"]
# coupe PERPENDICULAIRE : dans son schéma la direction montrée est en
# POINTS ; dans l'autre schéma elle file. Mêmes niveaux dans les deux.
for face in ("inf", "sup"):
    pp = [c["e"] for c in sch_p[f"points_{face}"]]
    fs = [c["e"] for c in sch_s[f"filants_{face}"]]
    chk(f"{face} : la dir. principale au même niveau en points (P) et filante (S)",
        pp == fs, f"{pp} vs {fs}")
    fp = [c["e"] for c in sch_p[f"filants_{face}"]]
    ps = [c["e"] for c in sch_s[f"points_{face}"]]
    chk(f"{face} : la dir. secondaire au même niveau filante (P) et en points (S)",
        fp == ps, f"{fp} vs {ps}")
# jamais poussé vers l'enrobage : niveau dessiné >= distance d'axe réelle,
# couche par couche (le renfort du retour bureau partait sous le treillis)
chk("aucune couche poussée vers l'enrobage (e dessiné >= e réel)",
    all(c["e"] >= float(g["e"]) * 10.0 - 1e-9
        for face, dk, cle in (("inf", "y", "points_inf"), ("sup", "y", "points_sup"),
                              ("inf", "x", "filants_inf"), ("sup", "x", "filants_sup"))
        for c, g in zip(sch_p[cle],
                        R5["dirs"][dk]["geo_inf" if face == "inf" else "geo_sup"]["couches"])))
# deux axes ne se confondent jamais : écart >= (Øa+Øb)/2 sur chaque face
for face in ("inf", "sup"):
    axes = sorted((c["e"], c["dia"]) for c in
                  sch_p[f"filants_{face}"] + sch_p[f"points_{face}"])
    chk(f"{face} : écart physique (Øa+Øb)/2 entre axes voisins",
        all(b[0] - a[0] >= (a[1] + b[1]) / 2.0 - 1e-9
            for a, b in zip(axes, axes[1:])),
        str(axes))
# le renfort au niveau SAISI dans les DEUX schémas (80 mm), et « n »
# transmis pour que le dessin répartisse 3 barres dans la bande
chk("schémas : renfort à 80 mm (niveau saisi) dans les deux vues",
    abs(sch_p["points_inf"][1]["e"] - 80.0) < 1e-6
    and abs(sch_s["filants_inf"][1]["e"] - 80.0) < 1e-6)
chk("schémas : n = 3 transmis au dessin (coupé dans SON schéma)",
    sch_p["points_inf"][1]["n"] == 3)
# et la note complète porte le libellé « 3 Ø12 »
p5 = generer_rapport_pdf(dalles, v2, bd, infos={"date": "31/08/2026"},
                         output_path=os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                                  "ndc_dalle_nbarres_test.pdf"))
t5n = pymupdf.open(p5)[1].get_text()
chk("note : « 3 Ø12 » dans le détail retenu et la légende",
    t5n.count("3 Ø12") >= 2 and "1125" in t5n.replace(" ", "").replace(" ", ""))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

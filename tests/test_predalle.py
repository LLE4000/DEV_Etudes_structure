# -*- coding: utf-8 -*-
"""Tests du module PRÉDALLE (predalle.py 1.0 — copie adaptée de dalle.py 2.1).

La différence prédalle est GÉOMÉTRIQUE, pas formulaire — chaque garantie
rougit si on la retire :
  1. POSITIONS PAR DÉFAUT (h = 22, h_pre = 6, Y principale) : la couche 1
     inférieure PRINCIPALE vit dans la peau préfabriquée, posée en FOND
     DE MOULE (enrobage + demi-Ø, SANS jeu premier lit : 3,5 cm pour
     Ø10) ; la couche 1 inférieure SECONDAIRE se pose AU-DESSUS de la
     prédalle (h_pre + demi-Ø = 6,5 cm). Chaque direction garde SA
     hauteur utile : avec My,inf = Mx,inf = 30, la direction secondaire
     demande PLUS d'acier (645 contre 541 mm²).
  2. RENFORT INFÉRIEUR PRINCIPAL (3 Ø12) : posé sur chantier, donc
     AU-DESSUS de la prédalle (7,0 cm) — le CDG pondéré (4,6), d (17,4)
     et Aₛ,req (573) suivent la position réelle.
  3. BASCULE X principale : les positions par défaut suivent (X passe
     dans la peau, Y au-dessus).
  4. NOTE PDF : mise en page Dalle réutilisée (traduction pre -> dal),
     avec « dont prédalle préf. / dont coulé en place », le trait de
     clivage, les d propres à chaque direction et le renfort.
  5. Le module Dalle reste intact : sa suite tourne inchangée (lancée
     séparément), et ses clés « dal » ne reçoivent RIEN de la prédalle.

Lancement : python tests/test_predalle.py (depuis la racine).
"""
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
    from modules import predalle
    predalle.show()


def md(at):
    return "\n".join(str(m.value) for m in at.markdown)


# ================================================================
print("=== 1. Positions par défaut : d propre à chaque direction ===")
at = AppTest.from_function(app, default_timeout=240)
at.run()
chk("le module s'ouvre sans exception", not at.exception, str(at.exception))
chk("défauts prédalle : h = 22 cm, h_pre = 6 cm",
    at.session_state["pre1_h"] == 22
    and float(at.session_state["pre1_h_pre"]) == 6.0)
at.text_input(key="pre1_sec1_Mx_inf_raw").set_value("30,00")
at.text_input(key="pre1_sec1_My_inf_raw").set_value("30,00")
at.text_input(key="pre1_sec1_Mx_sup_raw").set_value("12,00")
at.text_input(key="pre1_sec1_My_sup_raw").set_value("9,00")
at.text_input(key="pre1_sec1_V_raw").set_value("60,00")
at.run(); at.run()
chk("aucune exception", not at.exception, str(at.exception))
chk("inf. principale (Y, Ø10) en FOND DE MOULE, sans jeu : 3,5 cm",
    at.text_input(key="pre1_sec1_dist_axe_inf_y_c1").value == "3,5")
chk("inf. secondaire (X, Ø10) AU-DESSUS de la prédalle : 6,5 cm",
    at.text_input(key="pre1_sec1_dist_axe_inf_x_c1").value == "6,5")
t = md(at)
reqs = re.findall(r"Aₛ,req = (\d+) mm²", t)
chk("My,inf = Mx,inf = 30 : chaque direction garde SON d "
    "(541 en principale, 645 en secondaire)",
    reqs[:4] == ["541", "171", "645", "229"], str(reqs[:4]))
chk("la direction secondaire demande PLUS d'acier (d plus faible)",
    int(reqs[2]) > int(reqs[0]))
chk("hauteur : hᵤ,min + d₁ = 18,1 cm ≤ h = 22", "**18,1 cm**" in t and "h = **22 cm**" in t)

print("\n=== 2. Renfort inférieur principal posé AU-DESSUS de la prédalle ===")
at.session_state["pre1_sec1_ncouches_inf_y"] = 2
at.session_state["pre1_sec1_arm_type_inf_y_c2"] = "n barres"
at.session_state["pre1_sec1_n_barres_inf_y_c2"] = 3
at.session_state["pre1_sec1_ø_barres_inf_y_c2"] = 12
at.run()
chk("renfort : aucune exception", not at.exception, str(at.exception))
chk("renfort Ø12 posé au-dessus de la peau : 7,0 cm (6,0 + demi-Ø arrondi)",
    at.text_input(key="pre1_sec1_dist_axe_inf_y_c2").value == "7,0")
chk("CDG pondéré recalculé : (785·3,5 + 339·7,0)/1124 = 4,6 cm",
    at.text_input(key="pre1_sec1_ycdg_inf_y").value == "4,6")
t = md(at)
reqs = re.findall(r"Aₛ,req = (\d+) mm²", t)
chk("Aₛ,req principal suit le d réduit (573 mm²)",
    reqs[0] == "573", str(reqs[:1]))
chk("détail : « Treillis 10/10/100/100 + 3 Ø12 »",
    "Treillis 10/10/100/100 + 3 Ø12" in t)

print("\n=== 3. Bascule X principale : les positions suivent ===")
at.selectbox(key="pre1_dir_principale").set_value("X")
at.run()
chk("X passe dans la peau (3,5), Y au-dessus (6,5)",
    at.text_input(key="pre1_sec1_dist_axe_inf_x_c1").value == "3,5"
    and at.text_input(key="pre1_sec1_dist_axe_inf_y_c1").value == "6,5")
chk("le renfort (couche 2) reste au-dessus : 7,0 cm",
    at.text_input(key="pre1_sec1_dist_axe_inf_y_c2").value == "7,0")
at.selectbox(key="pre1_dir_principale").set_value("Y")
at.run()

print("\n=== 4. Note PDF : mise en page Dalle, géométrie prédalle ===")
at.button(key="predalle_btn_pdf").click()
at.run()
chk("génération PDF sans exception", not at.exception, str(at.exception))
pdf = at.session_state["predalle_pdf_bytes"]
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("garde + UNE planche par section", doc.page_count == 2, str(doc.page_count))
t2 = doc[1].get_text()
chk("titre : Prédalle 1 — Section A", "Prédalle 1" in t2)
chk("DIMENSIONS : dont prédalle préf. 6 cm / dont coulé en place 16 cm",
    "dont prédalle préf." in t2 and "dont coulé en place" in t2 and "16" in t2)
chk("coupe : « prédalle : 6 cm » écrit DANS la zone teintée des deux schémas, "
    "et le pied détaille préfabriqué / coulé",
    t2.count("prédalle : 6 cm") >= 2 and "prédalle préf. : 6 cm" in t2)
chk("ordre de la note inchangé (hauteur, inf P, inf S, sup P, sup S, tranchant)",
    0 <= t2.find("Vérification de la hauteur")
    < t2.find("Armatures inférieures — direction principale")
    < t2.find("Armatures inférieures — direction secondaire")
    < t2.find("Armatures supérieures — direction principale")
    < t2.find("Armatures supérieures — direction secondaire")
    < t2.find("Vérification de l'effort tranchant"))
chk("chaque direction garde SON d dans la note (17,4 principale / 15,5 secondaire)",
    "17,4" in t2 and "15,5" in t2)
chk("les deux schémas titrés (directions différentes par construction)",
    "Direction principale : Y" in t2 and "Direction secondaire : X" in t2
    and "identiques" not in t2)
chk("renfort « 3 Ø12 » dans la note", "3 Ø12" in t2)
chk("tranchant sans étriers (τ_adm,I)", "adm,I" in t2 and "Étrier" not in t2)

print("\n=== 5. Étanchéité : la prédalle n'écrit RIEN dans les clés Dalle ===")
fuites = [k for k in at.session_state.filtered_state
          if re.match(r"^dal\d+_", str(k)) or str(k).startswith("meta_dalle_nom_")]
chk("aucune clé « dal » créée par le module Prédalle", not fuites, str(fuites[:6]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

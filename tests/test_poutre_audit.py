# -*- coding: utf-8 -*-
"""Corrections de l'audit du 15/09/2026 — côté POUTRE.

Chaque contrôle rougit si la correction est retirée :

  C-1  La hauteur minimale est celle de la famille la PLUS EXIGEANTE
       (hᵤ,min(M) + son d₁), pas seulement celle du moment maximal.
       Cas : M_inf = 150 kNm sur trois lits (d₁ = 8,0) contre
       M_sup = 160 kNm sur un lit (d₁ = 5,0) — c'est le moment le
       PLUS FAIBLE qui gouverne, parce que son bras de levier est
       plus court : 47,5 + 8,0 = 55,5 cm (et non 49,1 + 5,0 = 54,1).

  I-1  La case « Distance axe lit » suit le recalcul : elle se figeait
       (st.text_input avec value= ET une clé déjà en session).

  I-2  Dans la note, la formule du pas théorique se recalcule telle
       qu'elle est imprimée (d en cm, résultat en cm).

  I-4  La note Poutre est compacte comme la note Dalle : le critère
       tient dans la deuxième formule, aucune ligne clé-valeur ne
       répète un résultat déjà en gras.

Lancement : python tests/test_poutre_audit.py (depuis la racine).
"""
import math
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
    from modules import poutre
    poutre.show()


def md(at):
    return "\n".join(str(m.value) for m in at.markdown)


# ================================================================
print("=== C-1. La famille la plus exigeante gouverne la hauteur ===")
at = AppTest.from_function(app, default_timeout=300)
at.run()
at.number_input(key="b1_b").set_value(30)
at.number_input(key="b1_h").set_value(60)
# face inférieure : trois lits 2Ø16 -> d₁ = (5,0+8,0+11,0)/3 = 8,0 cm
at.session_state["b1_sec1_nlits_inf"] = 3
for i in (2, 3):
    at.session_state[f"b1_sec1_n_as_inf_l{i}"] = 2
    at.session_state[f"b1_sec1_ø_as_inf_l{i}"] = 16
# face supérieure : un seul lit -> d₁ = 5,0 cm
at.text_input(key="b1_sec1_M_inf_raw").set_value("150,00")
at.text_input(key="b1_sec1_M_sup_raw").set_value("160,00")
at.text_input(key="b1_sec1_V_raw").set_value("120,00")
at.run(); at.run()
chk("aucune exception", not at.exception, str(at.exception))

alpha_b, mu, b_mm = 12.96, 0.1709, 300.0
hu = {m: math.sqrt(m * 1e6 / (alpha_b * b_mm * mu)) / 10 for m in (150.0, 160.0)}
d1_inf = float(at.text_input(key="b1_sec1_ycdg_inf").value.replace(",", "."))
d1_sup = float(at.text_input(key="b1_sec1_ycdg_sup").value.replace(",", "."))
chk("d₁ : 8,0 cm en inférieur (3 lits), 5,0 cm en supérieur (1 lit)",
    abs(d1_inf - 8.0) < 0.05 and abs(d1_sup - 5.0) < 0.05,
    f"{d1_inf} / {d1_sup}")
exigeant = hu[150.0] + d1_inf     # 47,5 + 8,0 = 55,5
autre = hu[160.0] + d1_sup        # 49,1 + 5,0 = 54,1
chk("manuel : c'est M_inf (le moment le PLUS FAIBLE) qui exige le plus",
    exigeant > autre, f"{exigeant:.2f} vs {autre:.2f}")

t = md(at)
ligne_hu = [l for l in t.splitlines() if "hᵤ,min = √" in l]
chk("écran : hᵤ,min calculée sur 150 kNm (la famille gouvernante), pas 160",
    bool(ligne_hu) and "150" in ligne_hu[0] and "160" not in ligne_hu[0],
    str(ligne_hu[:1]))
chk("écran : hᵤ,min + d₁ = 47,5 + 8,0 = 55,5 cm",
    re.search(r"hᵤ,min \+ d₁ = 47,5 \+ 8,0 = \*\*55,5 cm\*\*", t) is not None,
    str(re.findall(r"hᵤ,min \+ d₁ = [^\n]+", t)[:1]))
chk("écran : l'ancienne combinaison (49,1 + 5,0 = 54,1) a disparu",
    "54,1 cm" not in t)

print("\n=== I-1. La case « Distance axe lit » suit le recalcul ===")
a2 = AppTest.from_function(app, default_timeout=300)
a2.run()
a2.number_input(key="b1_b").set_value(30)
a2.number_input(key="b1_h").set_value(60)
a2.text_input(key="b1_sec1_M_inf_raw").set_value("180,00")
a2.text_input(key="b1_sec1_V_raw").set_value("230,00")
a2.run(); a2.run()
chk("Ø16 : case = 5,0 cm (3,0 + arr(0,8 + 0,8) + 0,0)",
    a2.text_input(key="b1_sec1_dist_disp_inf_1").value == "5,0")
a2.selectbox(key="b1_sec1_ø_as_inf").set_value(25)
a2.run()
chk("Ø25 : la case passe à 5,5 cm (3,0 + arr(0,8 + 1,25 = 2,05 -> 2,5))",
    a2.text_input(key="b1_sec1_dist_disp_inf_1").value == "5,5",
    a2.text_input(key="b1_sec1_dist_disp_inf_1").value)
chk("Ø25 : la case et le CDG disent la même chose",
    a2.text_input(key="b1_sec1_dist_disp_inf_1").value
    == a2.text_input(key="b1_sec1_ycdg_inf").value)

print("\n=== I-2 / I-4. La note Poutre : formule juste et mise en page compacte ===")
a2.button(key="btn_pdf").click()
a2.run()
chk("génération PDF sans exception", not a2.exception, str(a2.exception))
doc = pymupdf.open(stream=a2.session_state["pdf_bytes"], filetype="pdf")
tp = doc[1].get_text()

m = re.search(r"sth\s*=\s*([\d,]+)\s*·\s*([\d,]+)\s*·\s*([\d,]+)\s*/\s*([\d,]+)"
              r"\s*·\s*10\s*3\s*=\s*([\d,]+)\s*cm", tp)
chk("ligne « Pas théorique » lisible dans la note", m is not None,
    repr(tp[tp.find("Pas th"):tp.find("Pas th") + 80]))
if m:
    f = lambda s: float(s.replace(",", "."))          # noqa: E731
    calc = f(m.group(1)) * f(m.group(2)) * f(m.group(3)) / (f(m.group(4)) * 1e3)
    imprime = f(m.group(5))
    chk("I-2 : la formule imprimée reproduit le résultat imprimé "
        f"({calc:.2f} ≈ {imprime})", abs(calc - imprime) <= 0.1,
        f"{calc:.2f} vs {imprime}")

i_h = tp.find("Vérification de la hauteur")
bloc_h = tp[i_h:i_h + 260]
chk("I-4 : le critère « ≤/> h = … » tient dans la deuxième formule",
    re.search(r"hu,min \+ d1 = [\d,]+ \+ [\d,]+ = [\d,]+ cm\s*[≤>]\s*h\s*=\s*\d+ cm", bloc_h)
    is not None, repr(bloc_h[:200]))
chk("I-4 : plus de ligne clé-valeur « Hauteur minimale de la poutre » isolée",
    bloc_h.count("Hauteur minimale de la poutre") == 1, bloc_h.count("Hauteur minimale de la poutre"))
chk("I-4 : plus de lignes clé-valeur « Acier minimal » dans les blocs d'armatures",
    "Acier minimal" not in tp)
chk("I-4 : « Acier requis » ne subsiste que comme libellé de formule (2 blocs)",
    tp.count("Acier requis") == 2, tp.count("Acier requis"))
chk("la note ne déborde pas", not __import__("modules.export_pdf", fromlist=["x"]).DERNIERS_AVERTISSEMENTS,
    str(__import__("modules.export_pdf", fromlist=["x"]).DERNIERS_AVERTISSEMENTS))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

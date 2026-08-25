# -*- coding: utf-8 -*-
"""
Tests d'interface (streamlit.testing) du module Rigidité du sol,
plus la non-régression du cas de référence du module Poutre.

Lancer :  python3 tests/test_sol_interface.py
"""
import os
import sys

_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_RACINE)
sys.path.insert(0, _RACINE)
from streamlit.testing.v1 import AppTest  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown)


def app_sol():
    from modules import rigidite_sol
    rigidite_sol.show()


print("=== 1. Rendu et calcul par défaut ===")
at = AppTest.from_function(app_sol, default_timeout=90)
at.run()
chk("aucun crash au premier rendu", not at.exception, str(at.exception))
chk("titre présent", "Raideur élastique des sols" in md(at))
chk("mention sans IA", "sans IA" in md(at) or any("sans IA" in str(m.value) for m in at.markdown))

ss = at.session_state
chk("un sondage par défaut", len(ss["soundings"]) == 1)
chk("géométrie de fondation initialisée", ss["rs_B"] == 2.0 and ss["rs_q"] == 150.0)

txt = md(at)
chk("panneau SCIA affiché", "encoder dans SCIA" in txt)
chk("graphique de diffusion annoncé", "Comment la contrainte se diffuse" in txt)
chk("zonage affiché", "Zonage de la dalle" in txt)

print("\n=== 2. Le calcul réagit à la géométrie (k dépend de B) ===")
at.number_input(key="rs_B").set_value(2.0)
at.number_input(key="rs_L").set_value(2.0)
at.run()
k_small = None
for d in at.dataframe:
    try:
        k_small = float(d.value["k bas [MN/m³]"].iloc[0])
        break
    except Exception:
        continue
at.number_input(key="rs_B").set_value(8.0)
at.number_input(key="rs_L").set_value(8.0)
at.run()
k_big = None
for d in at.dataframe:
    try:
        k_big = float(d.value["k bas [MN/m³]"].iloc[0])
        break
    except Exception:
        continue
chk("aucun crash après changement de géométrie", not at.exception, str(at.exception))
chk("k diminue quand la fondation grandit",
    k_small is not None and k_big is not None and k_big < k_small,
    f"B=2 : {k_small} / B=8 : {k_big}")

print("\n=== 3. Niveau d'assise pris en compte (défaut F2) ===")
at.number_input(key="rs_B").set_value(2.0)
at.number_input(key="rs_L").set_value(2.0)
at.number_input(key="rs_D").set_value(0.0)
at.run()
sid = int(at.session_state["soundings"][0]["id"])
# couche molle en surface + couche raide dessous
from modules import rigidite_sol as RS  # noqa: E402
at.session_state[RS._order_key(sid)] = [1, 2]
at.session_state[RS._layer_key(sid, 1, "h")] = 2.0
at.session_state[RS._layer_key(sid, 1, "M")] = 1.5
at.session_state[RS._layer_key(sid, 1, "gamma")] = 17.0
at.session_state[RS._layer_key(sid, 1, "type")] = "Tourbe"
at.session_state[RS._layer_key(sid, 1, "type_prev")] = "Tourbe"
at.session_state[RS._layer_key(sid, 2, "h")] = 20.0
at.session_state[RS._layer_key(sid, 2, "M")] = 60.0
at.session_state[RS._layer_key(sid, 2, "gamma")] = 20.0
at.session_state[RS._layer_key(sid, 2, "type")] = "Sable dense"
at.session_state[RS._layer_key(sid, 2, "type_prev")] = "Sable dense"
at.run()
k_D0 = None
for d in at.dataframe:
    try:
        k_D0 = float(d.value["k bas [MN/m³]"].iloc[0]); break
    except Exception:
        continue
at.number_input(key="rs_D").set_value(2.0)
at.run()
k_D2 = None
for d in at.dataframe:
    try:
        k_D2 = float(d.value["k bas [MN/m³]"].iloc[0]); break
    except Exception:
        continue
chk("fonder sous la tourbe augmente nettement k",
    k_D0 is not None and k_D2 is not None and k_D2 > 2 * k_D0,
    f"D=0 : {k_D0} / D=2 : {k_D2}")

print("\n=== 4. Zonage centre / bord / angle ===")
zon = None
for d in at.dataframe:
    try:
        if "Zone" in d.value.columns:
            zon = d.value
            break
    except Exception:
        continue
chk("tableau de zonage présent", zon is not None)
if zon is not None:
    ks = list(zon["k bas [MN/m³]"])
    chk("k croît du centre vers l'angle", ks[0] <= ks[1] and ks[1] <= ks[3],
        str(ks))

print("\n=== 5. Réinitialisation scopée (défaut F3) ===")
def app_mixte():
    import streamlit as st
    from modules import poutre, dalle, rigidite_sol
    p = st.session_state.get("page_test", "Sol")
    if p == "Poutre":
        poutre.show()
    elif p == "Dalle":
        dalle.show()
    else:
        rigidite_sol.show()

am = AppTest.from_function(app_mixte, default_timeout=90)
am.session_state["page_test"] = "Poutre"
am.run()
chk("module Poutre rendu", not am.exception, str(am.exception))
am.session_state["page_test"] = "Dalle"
am.run()
chk("module Dalle rendu", not am.exception, str(am.exception))
n_beams = len(am.session_state["beams"])
n_dalles = len(am.session_state["dalles"])
am.session_state["page_test"] = "Sol"
am.run()
chk("module Sol rendu après les autres", not am.exception, str(am.exception))
am.button(key="rs_reset").click()
am.run()
chk("aucun crash après réinitialisation", not am.exception, str(am.exception))
_apres = len(am.session_state["beams"]) if "beams" in am.session_state else -1
chk("les POUTRES survivent à la réinitialisation du module sol",
    _apres == n_beams, f"avant {n_beams}, après {_apres}")
chk("les DALLES survivent à la réinitialisation du module sol",
    "dalles" in am.session_state and len(am.session_state["dalles"]) == n_dalles)
chk("les sondages sont bien réinitialisés", len(am.session_state["soundings"]) == 1)

print("\n=== 6. Modes secondaires ===")
for mode, cle in (("3. Vérification rapide k = q / w", "k = q / w"),
                  ("4. Comparer les théories", "Comparer"),
                  ("5. Abaque des sols", "Abaque"),
                  ("6. Raideur d'un plat en béton", "plat en béton")):
    a2 = AppTest.from_function(app_sol, default_timeout=90)
    a2.run()
    a2.selectbox(key="rs_mode").set_value(mode)
    a2.run()
    chk(f"mode « {mode[:28]}… » sans crash", not a2.exception, str(a2.exception))

print("\n=== 7. Abaque : correction de Terzaghi appliquée (défaut F7) ===")
a3 = AppTest.from_function(app_sol, default_timeout=90)
a3.run()
a3.selectbox(key="rs_mode").set_value("5. Abaque des sols")
a3.run()
ab = None
for d in a3.dataframe:
    try:
        if "k plaque 0,30 m" in d.value.columns:
            ab = d.value
            break
    except Exception:
        continue
chk("colonne « k plaque » nommée explicitement", ab is not None)
if ab is not None:
    cols = [c for c in ab.columns if c.startswith("k pour B")]
    chk("colonne k corrigée pour B présente", len(cols) == 1, str(list(ab.columns)))

print("\n=== 8. Non-régression : cas de référence Poutre ===")
def app_poutre():
    from modules import poutre
    poutre.show()

ap = AppTest.from_function(app_poutre, default_timeout=90)
ap.run()
ap.number_input(key="b1_b").set_value(20)
ap.number_input(key="b1_h").set_value(40)
ap.text_input(key="b1_sec1_M_inf_raw").set_value("200,00")
ap.text_input(key="b1_sec1_M_sup_raw").set_value("140,00")
ap.text_input(key="b1_sec1_V_raw").set_value("230,00")
ap.run(); ap.run()
t = md(ap)
chk("Poutre : hu,min = 67,2 cm", "**67,2 cm**" in t)
chk("Poutre : hmin = 73,2 cm", "**73,2 cm**" in t)
chk("Poutre : As,req inf = 1961 mm²", "Aₛ,req = 1961 mm²" in t)
chk("Poutre : As,req sup = 1373 mm²", "Aₛ,req = 1373 mm²" in t)
chk("Poutre : τ = 3,83", "3.83 N/mm²" in t or "3,83" in t)
chk("Poutre : τadm = 2,26", "2.26 N/mm²" in t or "2,26" in t)


for nom, info in KO:
    print("   -", nom, "|", info[:200])


print("\n=== 9. Écritures différées vers des clés de widgets ===")
a9 = AppTest.from_function(app_sol, default_timeout=90)
a9.run()
sid9 = int(a9.session_state["soundings"][0]["id"])
# simule ce que fait le bouton « Remplir le tableau » : écrire des clés
# de widgets DÉJÀ instanciés dans le run (nom du sondage, nappe)
a9.session_state["rs_pending"] = {f"snd{sid9}_nom": "CPT-IMPORTE",
                                  "rs_nappe_active": True, "rs_nappe": 4.5}
a9.run()
chk("aucune exception après écriture différée", not a9.exception, str(a9.exception))
chk("nom du sondage appliqué", a9.session_state[f"snd{sid9}_nom"] == "CPT-IMPORTE",
    a9.session_state[f"snd{sid9}_nom"])
chk("nappe appliquée", a9.session_state["rs_nappe"] == 4.5)
chk("file d'attente vidée", "rs_pending" not in a9.session_state
    or not a9.session_state["rs_pending"])

print("\n=== 10. Clés de couche complètes (pas de fantômes) ===")
from modules import rigidite_sol as RS10  # noqa: E402
a10 = AppTest.from_function(app_sol, default_timeout=90)
a10.run()
s10 = int(a10.session_state["soundings"][0]["id"])
ids10 = list(a10.session_state[RS10._order_key(s10)])
lid10 = ids10[1]
# poser des clés issues d'un import CPT sur la couche 2
for champ, val in (("M_haut", 95.0), ("Ic", 3.45), ("sbt", "Argile silteuse")):
    a10.session_state[RS10._layer_key(s10, lid10, champ)] = val
a10.run()
avant = [k for k in a10.session_state.filtered_state
         if k.startswith(f"snd{s10}_layer_{lid10}_")]
chk("clés d'import bien posées avant suppression",
    any(k.endswith("_M_haut") for k in avant), str(avant))
# suppression par le VRAI bouton de l'interface (callback dans le run)
a10.button(key=f"rs_del_l_{s10}_{lid10}").click()
a10.run()
restant = [k for k in a10.session_state.filtered_state
           if k.startswith(f"snd{s10}_layer_{lid10}_")]
chk("aucune clé fantôme après suppression d'une couche", not restant, str(restant))
chk("LAYER_FIELDS couvre M_haut, Ic et sbt",
    all(c in RS10.LAYER_FIELDS for c in ("M_haut", "Ic", "sbt")),
    str(RS10.LAYER_FIELDS))
chk("aucune exception après suppression", not a10.exception, str(a10.exception))

print(f"\nRÉSULTAT FINAL : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", info[:200])

sys.exit(0 if not KO else 1)

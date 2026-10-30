# -*- coding: utf-8 -*-
"""Tests du PROJET d'assemblages métalliques (lot 1) : un fichier = plusieurs
assemblages, comme Poutre et Dalle.

Chaque garantie rougit si on la retire :
  1. MODÈLE (acier/projet.py, sans Streamlit) : repères A1, A2…, dupliquer,
     renommer, déplacer, supprimer, fichier projet, relecture des fichiers
     d'assemblage seul, nom de fichier.
  2. PAGE PROJET : projet vide → catalogue ; « Créer » ajoute A1 et ouvre
     l'écran de son type (repère = champ « Repère de l'assemblage ») ; barre
     ◀ Projet + sélecteur, sans 📂 ; une saisie est dans l'entrée après
     ◀ Projet ; ➕ → catalogue → A2 d'un autre type ; le sélecteur rouvre A1
     avec sa saisie.
  3. ALLER-RETOUR PAR L'ACCUEIL : l'entrée garde la saisie (recopie au début
     du run), et l'écran se rouvre dessus (ré-hydratation) — pas de défauts.
  4. 🔄 D'UN ÉCRAN : valeurs par défaut dans l'écran ET dans l'entrée, repère
     conservé. PIÈGE 1 : le 🔄 des doubles cornières (efface `asm_*`) laisse
     le projet intact.
  5. PIÈGE 2 : deux entrées du même type — la note PDF générée pour A1 n'est
     plus téléchargeable sur A4 ; le dernier clic du dessin est conservé.
  6. MENU ⋯ : dupliquer (copie après l'original, mêmes valeurs), renommer,
     ordre, supprimer avec confirmation (Annuler puis Confirmer).
  7. FICHIERS : chargement d'un fichier projet (remplace, infos comprises),
     d'un fichier d'assemblage seul (s'ajoute), détection d'un fichier brut.
  8. ÉCRAN OUVERT HORS PROJET (asm_courant posé directement, comme les
     tests existants) : 📂 et 💾 de l'assemblage seul, ◀ Projet mène au projet.

Lancement : python3 tests/test_projet_assemblages.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

from acier import projet as modele  # noqa: E402
from acier import projet_ui  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


def app(page="Assemblages métalliques"):
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
    at.session_state["page"] = page
    return at


N_RUN = [0]


def run(at):
    at.run()
    N_RUN[0] += 1
    if at.exception:
        chk(f"aucune exception (relance {N_RUN[0]})", False, str(at.exception[0].message)[:200])
    return at


def boutons(at):
    return {b.key for b in at.button if b.key}


def prj(at):
    return at.session_state["prj_projet"]


def entree(at, ident):
    return modele.trouver(prj(at), ident)


# ================================================================
print("=== 1. Modèle (sans Streamlit) ===")
p = modele.nouveau()
e1 = modele.ajouter(p, "plat_ame", {"V_Ed": 125}, "vis")
chk("premier repère A1, id 1", modele.repere(e1) == "A1" and e1["id"] == 1 and e1["visserie"] == "vis")
e2 = modele.ajouter(p, "platine_about", {"id_rep": ""})
chk("deuxième repère A2, id 2", modele.repere(e2) == "A2" and e2["id"] == 2)
e3 = modele.ajouter(p, "doubles_cornieres", {"id_rep": "P7"})
chk("un repère saisi est gardé", modele.repere(e3) == "P7")
c = modele.dupliquer(p, 1)
chk("dupliquer : juste après l'original, « (copie) », mêmes valeurs, id neuf",
    [x["id"] for x in modele.entrees(p)] == [1, c["id"], 2, 3] and modele.repere(c) == "A1 (copie)"
    and c["values"]["V_Ed"] == 125 and c["values"] is not e1["values"])
chk("renommer, et refus du vide", modele.renommer(p, c["id"], "B7") and modele.repere(c) == "B7"
    and not modele.renommer(p, c["id"], "  ") and modele.repere(c) == "B7")
chk("déplacer : monter A2 d'un rang ; monter le premier est refusé",
    modele.deplacer(p, 2, -1) and [x["id"] for x in modele.entrees(p)] == [1, 2, c["id"], 3]
    and not modele.deplacer(p, 1, -1))
chk("supprimer, et inconnu refusé", modele.supprimer(p, c["id"]) and len(modele.entrees(p)) == 3
    and not modele.supprimer(p, 99))
modele.supprimer(p, 1)
chk("le repère libre réutilise le trou", modele.repere_libre(p) == "A1")
infos = {"nom_projet": "Reamenagement", "partie": "Hall", "date": "30/10/2026", "indice": "B"}
pl = modele.payload(p, infos)
chk("payload : version, projet, entrées avec repère et type",
    pl["version"] == modele.VERSION and pl["projet"]["partie"] == "Hall"
    and [(x["type"], x["repere"]) for x in pl["assemblages"]] == [("platine_about", "A2"), ("doubles_cornieres", "P7")])
chk("nom de fichier AAA_Assemblages Partie#Indice_Date",
    modele.nom_fichier(infos, ".json") == "REA_Assemblages Hall#B_30-10-2026.json")


def norm(t, src):
    return dict(src.get("values") if isinstance(src.get("values"), dict) else src)


infos2, lot = modele.lire_fichier(json.loads(json.dumps(pl)), norm)
chk("relecture du fichier projet", infos2["partie"] == "Hall" and [t for t, _, _, _ in lot] == ["platine_about", "doubles_cornieres"]
    and lot[1][1]["id_rep"] == "P7")
infos3, lot3 = modele.lire_fichier({"version": "assemblage-plat-ame-1.0", "assemblage": "plat_ame",
                                    "values": {"V_Ed": 200}, "visserie": "x"}, norm)
chk("fichier d'assemblage seul (déverrouillé)", infos3 is None and lot3 == [("plat_ame", {"V_Ed": 200}, "x", False)])
_, lot4 = modele.lire_fichier({"V_Ed": 1}, norm, detecter=lambda s: "doubles_cornieres")
chk("fichier brut : type détecté par l'appelant", lot4[0][0] == "doubles_cornieres")
try:
    modele.lire_fichier({"V_Ed": 1}, norm)
    chk("fichier brut sans détection → ValueError", False)
except ValueError:
    chk("fichier brut sans détection → ValueError", True)
try:
    modele.lire_fichier([1, 2], norm)
    chk("structure inattendue → ValueError", False)
except ValueError:
    chk("structure inattendue → ValueError", True)

# ================================================================
print("\n=== 2. Page projet : création, ouverture, synchronisation ===")
at = app(); run(at)
chk("projet vide → catalogue, bouton « Créer » de chaque type, pas de liste",
    {"asm_btn_open_platine_about", "asm_btn_open_plat_ame", "asm_btn_open_doubles_cornieres"} <= boutons(at)
    and "Premier assemblage" in md(at) and "Assemblages (0)" in md(at))
at.button(key="asm_btn_open_platine_about").click(); run(at)
chk("« Créer » ajoute A1 et ouvre l'écran platine d'about",
    at.session_state["asm_courant"] == "platine_about" and at.session_state["prj_ouvert"] == 1
    and modele.repere(entree(at, 1)) == "A1" and entree(at, 1)["type"] == "platine_about")
chk("le repère est le champ « Repère de l'assemblage » de l'écran (source unique)",
    at.session_state["pab_id_rep"] == "A1")
chk("barre : ◀ Projet, sélecteur [A1 ▾], pas de 📂",
    "pab_btn_retour" in boutons(at) and any(s.key == "pab_sel_projet" for s in at.selectbox)
    and "pab_btn_open_toggle" not in boutons(at))
at.number_input(key="pab_V_Ed").set_value(132); run(at)
at.button(key="pab_btn_retour").click(); run(at)
chk("◀ Projet : liste avec A1, pastille et taux", "asm_courant" not in at.session_state.to_dict()
    and "**A1**" in md(at) and "η =" in md(at) and "Assemblages (1)" in md(at))
chk("la saisie V = 132 est dans l'entrée", entree(at, 1)["values"]["V_Ed"] == 132)
at.button(key="prj_btn_nouveau").click(); run(at)
chk("➕ Nouvel assemblage → catalogue avec « Annuler »",
    "prj_btn_catalogue_fermer" in boutons(at) and "asm_btn_open_plat_ame" in boutons(at))
at.button(key="asm_btn_open_plat_ame").click(); run(at)
chk("A2 plat d'âme créé et ouvert", at.session_state["asm_courant"] == "plat_ame"
    and at.session_state["prj_ouvert"] == 2 and at.session_state["fpl_id_rep"] == "A2")
at.selectbox(key="fpl_sel_projet").set_value(projet_ui.libelle_entree(entree(at, 1))); run(at)
chk("sélecteur → A1 : écran platine, V = 132 restauré", at.session_state["asm_courant"] == "platine_about"
    and at.session_state["prj_ouvert"] == 1 and at.session_state["pab_V_Ed"] == 132)

# ================================================================
print("\n=== 3. Aller-retour par l'accueil ===")
at.number_input(key="pab_V_Ed").set_value(140); run(at)
at.session_state["page"] = "Accueil"; run(at)
chk("sur l'accueil, l'entrée garde V = 140 (recopie au début du run)", entree(at, 1)["values"]["V_Ed"] == 140)
chk("…et l'état de l'écran a été effacé par Streamlit (c'est le cas à couvrir)",
    "pab_V_Ed" not in at.session_state.to_dict())
at.session_state["page"] = "Assemblages métalliques"; run(at)
chk("retour : l'écran A1 se rouvre avec V = 140, pas la valeur par défaut",
    at.session_state["asm_courant"] == "platine_about" and at.session_state.get("pab_V_Ed") == 140)
run(at)
chk("un run de plus : l'entrée n'a pas été écrasée", entree(at, 1)["values"]["V_Ed"] == 140)

# ================================================================
print("\n=== 4. 🔄 d'un écran, piège 1 ===")
at.button(key="pab_btn_reset").click(); run(at)
chk("🔄 : écran et entrée aux valeurs par défaut, repère conservé",
    at.session_state["pab_V_Ed"] == 125 and entree(at, 1)["values"]["V_Ed"] == 125
    and modele.repere(entree(at, 1)) == "A1" and at.session_state["pab_id_rep"] == "A1")
at.button(key="pab_btn_retour").click(); run(at)
at.button(key="prj_btn_nouveau").click(); run(at)
at.button(key="asm_btn_open_doubles_cornieres").click(); run(at)
chk("A3 doubles cornières ouvert", at.session_state["asm_courant"] == "doubles_cornieres" and at.session_state["prj_ouvert"] == 3)
at.button(key="asm_btn_reset").click(); run(at)
chk("PIÈGE 1 : le 🔄 des doubles cornières (efface asm_*) laisse le projet intact",
    len(modele.entrees(prj(at))) == 3 and at.session_state["prj_ouvert"] == 3
    and at.session_state["asm_courant"] == "doubles_cornieres" and at.session_state["asm_id_rep"] == "A3")

# ================================================================
print("\n=== 5. Piège 2 : deux entrées du même type ===")
at.selectbox(key="asm_sel_projet").set_value(projet_ui.libelle_entree(entree(at, 1))); run(at)
at.button(key="pab_btn_pdf").click(); run(at)
chk("note PDF générée pour A1, téléchargeable", bool(at.session_state.get("pab_pdf_bytes"))
    and "pab_btn_pdf_dl" in {d.key for d in at.get("download_button")})
at.session_state["pab_ui_dernier_clic"]["pab_cmp_elev"] = 4242
at.button(key="pab_btn_retour").click(); run(at)
at.button(key="prj_btn_nouveau").click(); run(at)
at.button(key="asm_btn_open_platine_about").click(); run(at)
chk("A4 platine ouvert : la note de A1 n'est plus téléchargeable",
    at.session_state["prj_ouvert"] == 4 and not at.session_state.get("pab_pdf_bytes")
    and "pab_btn_pdf_dl" not in {d.key for d in at.get("download_button")})
chk("…et le dernier clic du dessin est conservé (pas de rejeu sur A4)",
    at.session_state["pab_ui_dernier_clic"].get("pab_cmp_elev") == 4242)
chk("A4 a les valeurs par défaut, A1 garde les siennes",
    at.session_state["pab_V_Ed"] == 125 and at.session_state["pab_id_rep"] == "A4")

# ================================================================
print("\n=== 6. Menu ⋯ : dupliquer, renommer, ordre, supprimer ===")
at.number_input(key="pab_V_Ed").set_value(150); run(at)
at.button(key="pab_btn_retour").click(); run(at)
at.button(key="prj_btn_dup_4").click(); run(at)
ids = [x["id"] for x in modele.entrees(prj(at))]
chk("📋 : copie juste après A4, « A4 (copie) », V = 150", ids == [1, 2, 3, 4, 5]
    and modele.repere(entree(at, 5)) == "A4 (copie)" and entree(at, 5)["values"]["V_Ed"] == 150)
at.text_input(key="prj_rep_5").set_value("P12"); run(at)
chk("renommer dans le menu ⋯", modele.repere(entree(at, 5)) == "P12" and "**P12**" in md(at))
at.button(key="prj_btn_up_5").click(); run(at)
chk("↑ Monter", [x["id"] for x in modele.entrees(prj(at))] == [1, 2, 3, 5, 4])
chk("↑ du premier et ↓ du dernier désactivés",
    at.button(key="prj_btn_up_1").disabled and at.button(key="prj_btn_down_4").disabled)
at.button(key="prj_btn_suppr_5").click(); run(at)
chk("🗑️ demande confirmation", {"prj_btn_suppr_ok_5", "prj_btn_suppr_non_5"} <= boutons(at)
    and len(modele.entrees(prj(at))) == 5)
at.button(key="prj_btn_suppr_non_5").click(); run(at)
chk("Annuler : rien supprimé", "prj_btn_suppr_ok_5" not in boutons(at) and len(modele.entrees(prj(at))) == 5)
at.button(key="prj_btn_suppr_5").click(); run(at)
at.button(key="prj_btn_suppr_ok_5").click(); run(at)
chk("Confirmer : P12 supprimé", [x["id"] for x in modele.entrees(prj(at))] == [1, 2, 3, 4] and "P12" not in md(at))
chk("en-tête : compte et non vérifiés", "Assemblages (4)" in md(at))

# ================================================================
print("\n=== 7. Fichiers ===")
pl = modele.payload(prj(at), {"nom_projet": "Réaménagement", "partie": "Hall", "date": "30/10/2026", "indice": "B"})
SCRIPT = """
import os, sys
sys.path.insert(0, %r); os.chdir(%r)
import streamlit as st
from acier import projet_ui
st.session_state["page"] = "Assemblages métalliques"
projet_ui.projet()
st.session_state["_msg"] = projet_ui.charger_fichier(st.session_state["_data"])
""" % (RACINE, RACINE)
at2 = AppTest.from_string(SCRIPT, default_timeout=240)
at2.session_state["_data"] = json.loads(json.dumps(pl))
at2.run()
chk("fichier projet : remplace le projet et ses informations",
    not at2.exception and at2.session_state["_msg"].startswith("Projet chargé : 4")
    and [x["type"] for x in modele.entrees(at2.session_state["prj_projet"])]
    == ["platine_about", "plat_ame", "doubles_cornieres", "platine_about"]
    and at2.session_state["partie"] == "Hall" and at2.session_state["chk_infos_projet"] is True
    and modele.trouver(at2.session_state["prj_projet"], 4)["values"]["V_Ed"] == 150)
at2.session_state["_data"] = {"version": "assemblage-plat-ame-1.0", "assemblage": "plat_ame",
                              "values": {"V_Ed": 210}, "visserie": "visserie X"}
at2.run()
chk("fichier d'assemblage seul : s'ajoute au projet avec le repère libre",
    not at2.exception and at2.session_state["_msg"] == "Assemblage ajouté au projet"
    and len(modele.entrees(at2.session_state["prj_projet"])) == 5
    and modele.entrees(at2.session_state["prj_projet"])[-1]["values"]["V_Ed"] == 210
    and modele.repere(modele.entrees(at2.session_state["prj_projet"])[-1]) == "A5"
    and modele.entrees(at2.session_state["prj_projet"])[-1]["visserie"] == "visserie X")
from acier.assemblages.poutre_poutre.doubles_cornieres import interface as dc  # noqa: E402
brut = dict(dc.defaults()); brut["V_Ed"] = 333
chk("fichier brut (clés à la racine) : type détecté", projet_ui._detecter_type(brut) == "doubles_cornieres"
    and projet_ui._detecter_type({"x": 1}) is None)
at2.session_state["_data"] = brut
at2.run()
chk("…et chargé comme doubles cornières", not at2.exception
    and modele.entrees(at2.session_state["prj_projet"])[-1]["type"] == "doubles_cornieres"
    and modele.entrees(at2.session_state["prj_projet"])[-1]["values"]["V_Ed"] == 333)

# ================================================================
print("\n=== 8. Écran ouvert hors projet (comportement d'avant) ===")
at = app(); at.session_state["asm_courant"] = "platine_about"; run(at)
chk("sans entrée ouverte : 📂 de l'assemblage seul, pas de sélecteur, 💾 présent",
    "pab_btn_open_toggle" in boutons(at) and not any(s.key == "pab_sel_projet" for s in at.selectbox)
    and "pab_btn_retour" in boutons(at))
at.button(key="pab_btn_retour").click(); run(at)
chk("◀ Projet mène à la page projet (vide → catalogue)", "asm_courant" not in at.session_state.to_dict()
    and "asm_btn_open_platine_about" in boutons(at))

# ================================================================
print("\n=== 9. Verrou 🔒 et variante (lot 3) ===")


def infos_at(at):
    return "\n".join(str(i.value) for i in at.info)


at = app(); run(at)
at.button(key="asm_btn_open_platine_about").click(); run(at)
at.number_input(key="pab_V_Ed").set_value(160); run(at)
at.button(key="pab_btn_retour").click(); run(at)
at.button(key="prj_btn_lock_1").click(); run(at)
chk("🔒 : entrée verrouillée, ligne marquée, repère et suppression désactivés",
    modele.est_verrouille(entree(at, 1)) and "🔒" in md(at) and at.text_input(key="prj_rep_1").disabled
    and at.button(key="prj_btn_suppr_1").disabled and "Déverrouiller" in at.button(key="prj_btn_lock_1").label)
chk("le modèle refuse de supprimer ou renommer une entrée verrouillée",
    not modele.supprimer(prj(at), 1) and not modele.renommer(prj(at), 1, "Z")
    and len(modele.entrees(prj(at))) == 1 and modele.repere(entree(at, 1)) == "A1")
at.button(key="prj_btn_ouvrir_1").click(); run(at)
chk("écran verrouillé : bandeau, saisies et visserie désactivées, 🔄 désactivé, bouton 🔓",
    "verrouillé" in infos_at(at) and at.number_input(key="pab_V_Ed").disabled
    and at.text_input(key="pab_visserie").disabled and at.button(key="pab_btn_reset").disabled
    and "pab_btn_unlock" in boutons(at))
at.session_state["pab_V_Ed"] = 999; run(at)
chk("une saisie forcée rebondit sur la valeur enregistrée", at.session_state["pab_V_Ed"] == 160
    and entree(at, 1)["values"]["V_Ed"] == 160)
at.button(key="pab_btn_unlock").click(); run(at)
chk("🔓 depuis l'écran : modifiable à nouveau, plus de bandeau", not modele.est_verrouille(entree(at, 1))
    and not at.number_input(key="pab_V_Ed").disabled and "verrouillé" not in infos_at(at))
at.number_input(key="pab_V_Ed").set_value(170); run(at)
chk("…et la saisie est reprise", entree(at, 1)["values"]["V_Ed"] == 170)
at.button(key="pab_btn_retour").click(); run(at)
at.button(key="prj_btn_lock_1").click(); run(at)
pl = modele.payload(prj(at), {})
chk("le fichier projet porte le verrou", pl["assemblages"][0]["verrou"] is True)
_, lot = modele.lire_fichier(json.loads(json.dumps(pl)), norm)
chk("…et le relit", lot[0][3] is True)
at.button(key="prj_btn_var_1").click(); run(at)
chk("🧬 Variante : copie « (variante) » ouverte aussitôt, déverrouillée, mêmes valeurs, l'original reste verrouillé",
    at.session_state["prj_ouvert"] == 2 and at.session_state["asm_courant"] == "platine_about"
    and modele.repere(entree(at, 2)) == "A1 (variante)" and not modele.est_verrouille(entree(at, 2))
    and at.session_state["pab_V_Ed"] == 170 and not at.number_input(key="pab_V_Ed").disabled
    and modele.est_verrouille(entree(at, 1)))
at.number_input(key="pab_V_Ed").set_value(190); run(at)
at.selectbox(key="pab_sel_projet").set_value(projet_ui.libelle_entree(entree(at, 1))); run(at)
chk("retour sur l'original verrouillé : V = 170 intact, la variante garde 190",
    at.session_state["pab_V_Ed"] == 170 and at.number_input(key="pab_V_Ed").disabled
    and entree(at, 2)["values"]["V_Ed"] == 190)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

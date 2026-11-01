# -*- coding: utf-8 -*-
"""Tests de l'écran « Assemblages métalliques » (AppTest, application réelle),
après la finalisation du 21/09/2026 : DEUX COLONNES — à gauche le dessin
(élévation puis vue en plan, empilées, colonne FIGÉE par CSS sticky), à
droite la carte compacte par objet, puis les onglets. Tout se règle aussi
sur le dessin (cotes, panneaux de pièce, poignées ±, glissement).

Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md §4.1) : la
carte est rendue par le kit (acier/assemblages/carte.py), le mode est un
segmented_control, la visserie est au cartouche, la case « Dessin
interactif » en tête de la colonne dessin.

Chaque garantie rougit si on la retire :
  1. NAVIGATION : sélection, carte « Doubles cornières d'âme », accueil.
  2. ÉTAT ET ÉCRAN : 84 clés ; titre court + barre d'outils partagée, plus de
     ❔ ni de « Version » ; statut une ligne (dimensionnante nommée comme
     dans la note) + taux par élément ; deux dessins EMPILÉS dans la colonne
     figée (CSS sticky injecté, conteneur asm_col_dessin) ; la carte par
     objet à droite (en-tête du mode, profilés, cornières, boulons avec les
     fixations dans leur groupe, efforts, avancés, cartouche) — JAMAIS la
     géométrie (elle ne se règle que sur le dessin) ; « Grande aile côté »
     seulement pour une cornière inégale ; onglet Vérifications par défaut,
     légende, périmètre et Tableau 3.3 dans des replis.
  3. PANNEAUX D'OBJET : chaque pièce a sa fenêtre complète, titre court,
     libellés LIBELLE, aides AIDES, options AFFICHAGE (poutres — la
     secondaire reçoit gh, Δz, dc,sup, dc,inf, c —, cornières avec
     fixations, boulons, groupes S et P, efforts) ; les champs pilotés ◆ y
     sont désactivés en prédimensionnement ; la fenêtre des cordons apparaît
     quand un côté est soudé.
  4. RECALCUL : VEd = 400 (panneau efforts) → NON VÉRIFIÉ ; hc = 260 →
     alerte courte, « Localiser », chip hc modifiable.
  5. SOURCE UNIQUE : cote, poignée +, fenêtre de groupe, glissement (gh),
     panneau de pièce (profilé) écrivent tous dans asm_<clé>.
  6. PRÉDIM : mode dans l'en-tête de la carte (segmented_control) avec Taux
     cible · Ø minimal · Ø maximal ; « Appliquer » recopie la solution.
  7. REPLI : dessin interactif désactivé → dessins statiques à gauche, le
     panneau des cotes s'ajoute en tête de la colonne de droite, mêmes clés.
  8. EXPORTS : texte du corrigé, enregistrer / charger (visserie comprise),
     réinitialiser, benchmark, méthode (« Ce que l'outil vérifie »,
     « Données de bibliothèque »). 9. ÉTANCHÉITÉ.
 10. CHARTE : rien ne se perd (87 clés), aides ≤ 160 et seulement depuis
     AIDES, libellés ≤ 22 / 16 / 20, options ≤ 28, aucune ligne d'objet > 4,
     5 « ? » au défaut, orientation pour 5 des 18 cornières.

Lancement : python3 tests/test_assemblages_interface.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

import re  # noqa: E402

from acier.assemblages import carte  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres import (interface, ecran_resultats, ecran_saisie,  # noqa: E402
                                                               moteur, entrees, synthese)
from acier.assemblages.poutre_poutre.doubles_cornieres.ecran_saisie import valeur_widget  # noqa: E402
from acier.bibliotheques import DB, ORI_P, ORI_S  # noqa: E402

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
    """Relance l'application et vérifie qu'aucune exception n'est levée."""
    at.run()
    N_RUN[0] += 1
    if at.exception:
        chk(f"aucune exception (relance {N_RUN[0]})", False, str(at.exception[0].message)[:200])
    return at


def cles_widgets(at):
    # le mode est un segmented_control (charte du 31/10/2026) : compté avec les autres widgets
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox)
            + list(at.segmented_control) if w.key}


def expanders(at):
    return [str(x.label) for x in at.expander]


def message(at, changes, t):
    """Simule un message validé du dessin (cote, poignée, fenêtre de groupe,
    panneau de pièce, glissement) : ce que fait _traiter_clic."""
    derniers = at.session_state["asm_ui_dernier_clic"]
    if derniers.get("asm_cmp_elev") != t:
        derniers["asm_cmp_elev"] = t
        for k, v in changes.items():
            at.session_state["asm_" + k] = valeur_widget(k, v)


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)

# ================================================================
print("=== 1. Navigation ===")
at = app("Accueil"); run(at)
chk("la carte « Assemblages métalliques » est sur l'accueil (ligne Acier)",
    "page=Assemblages métalliques" in md(at))
at = app(); run(at)
chk("la page de sélection se charge sans exception", not at.exception, str(at.exception))
chk("famille « Poutre – Poutre » et carte « Doubles cornières d'âme »",
    "Poutre – Poutre" in md(at) and "Doubles cornières d'âme" in md(at))
at.button(key="asm_btn_open_doubles_cornieres").click(); run(at)
chk("« Ouvrir » mène à l'écran du module", not at.exception and at.session_state["asm_courant"] == "doubles_cornieres",
    str(at.exception))
chk("en-tête du module", "doubles cornières d'âme" in md(at))

# ================================================================
print("\n=== 2. État par défaut et écran ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("asm_") and k[4:] in interface.CLES]
chk("les 84 clés d'entrée existent en session", len(cles) == 84, str(len(cles)))
t = md(at)
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ", "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t)
R0 = moteur.compute({k: at.session_state["asm_" + k] for k in interface.CLES})
chk("statut sur une ligne : 74,0 % et la dimensionnante nommée comme dans la note (synthese.court) ; "
    "pas de mention du benchmark",
    "— 74,0 % · Pression diamétrale – âme</span>" in t and synthese.court(R0.gov) == "Pression diamétrale – âme"
    and "Benchmark du module" not in t)
chk("taux par élément : Boulons, Cornières, Portée, Porteuse", all(x in t for x in ("Boulons", "Cornières", "Portée", "Porteuse")))
chk("les dessins sont rendus (deux composants, empilés dans la colonne de gauche)",
    len(at.get("component_instance")) == 2, str(len(at.get("component_instance"))))
t = md(at)
chk("colonne du dessin FIGÉE : CSS sticky injecté et ciblé sur le conteneur asm_col_dessin",
    "position: sticky" in t and "st-key-asm_col_dessin" in t)
w = cles_widgets(at)
# k[4:] in CLES écarte déjà les chips d'alerte (asm_fix_<clé> → « fix_<clé> »
# n'est pas une entrée), sans écarter les vraies clés asm_fix_P / asm_fix_S
donnees = {k for k in w if k.startswith("asm_") and k[4:] in interface.CLES}
# Projet · Date du cartouche : hors projet (nom_projet vide), les deux — la condition même de
# projet_ui.infos_cartouche (décision du 01/11/2026 : la date de session cachait id_date dans les cinq écrans)
etat = at.session_state.to_dict()
hors_projet = ["id_projet", "id_date"] if not etat.get("nom_projet") else []
CARTE_ATTENDUE = {"asm_" + k for k in [
    "mode_calc",                                                  # EN-TÊTE (segmented_control)
    "prof_P", "nu_P", "prof_S", "nu_S",                           # PROFILÉS
    "corn_u", "nu_C",                                             # CORNIÈRES (orient : cornière inégale seulement)
    "boulon_u", "classe", "trou", "cat",                          # BOULONS
    "fix_P", "n1P_u", "n2P_u", "fix_S", "n1S_u", "n2S_u",         # ↳ AILES A (groupe P) · ↳ AILES B (groupe S)
    "V_Ed", "N_Ed", "H_Ed", "M_Ed",                               # EFFORTS ELU
    "r_n", "lt_ok", "d0_u", "filet", "g_M0", "g_M2", "g_M2n",     # avancés (repliés)
    "opt_exc", "k_rot", "opt_blf", "expo",
    "id_rep", "id_red"] + hors_projet}                            # cartouche
chk("colonne de droite : la carte par objet, complète — et rien d'autre",
    donnees == CARTE_ATTENDUE, str(sorted(donnees ^ CARTE_ATTENDUE)))
chk("le mode est un segmented_control (clé asm_mode_calc, valeurs brutes inchangées)",
    any(x.key == "asm_mode_calc" for x in at.segmented_control)
    and at.session_state["asm_mode_calc"] == "VÉRIFICATION")
chk("plus de bloc MODE ET FIXATIONS ni VISSERIE ; les fixations ouvrent leur groupe de boulons "
    "(↳ AILES A (groupe P), ↳ AILES B (groupe S) avec son « ? »)",
    "MODE ET FIXATIONS" not in t and ">VISSERIE<" not in t and "AILES&nbsp;A" in t and "AILES&nbsp;B" in t
    and any(getattr(m, "help", None) == ecran_saisie.AIDES["AILES B"] for m in at.markdown))
chk("cornière à ailes égales (défaut L100x100x10) : pas de « Grande aile côté » ; valeur conservée en session",
    "asm_orient" not in w and at.session_state["asm_orient"] == ORI_P)
chk("titre court, caption « titre long · version 1.3 », barre d'outils partagée (clés inchangées)",
    "## Doubles cornières d'âme" in t and "Assemblage poutre–poutre – doubles cornières d'âme · version 1.3" in t
    and {"asm_btn_retour", "asm_btn_home", "asm_btn_save_dl", "asm_btn_pdf", "asm_btn_reset"}
    <= {b.key for b in list(at.button) + list(at.get("download_button"))})
chk("plus de bouton ❔ ni de cellule « Version »",
    not any(b.key == "asm_btn_version_dc" for b in at.button) and "Version 1.3" not in t)
av = [x for x in at.expander if str(x.label) == "Paramètres avancés"]
chk("« Dessin interactif » en tête de la colonne dessin (plus dans les avancés), avec son « ? »",
    "asm_ui_composant" in w and av and not [c for c in av[0].checkbox if c.key == "asm_ui_composant"]
    and any(c.key == "asm_ui_composant" and c.help == ecran_saisie.AIDES["ui_composant"] for c in at.checkbox))
chk("la géométrie ne se règle QUE sur le dessin (aucune cote dans la carte)",
    not donnees & {"asm_" + k for k in ecran_saisie.GEOMETRIE_DESSIN})
cart = [x for x in at.expander if str(x.label) == "Cartouche (note et plan)"]
chk("visserie (rondelles, écrous) : saisie libre au cartouche, défaut « 1 rondelle + 1 écrou »",
    "asm_visserie" in w and at.session_state["asm_visserie"] == "1 rondelle + 1 écrou"
    and cart and any(c.key == "asm_visserie" for c in cart[0].text_input))
chk("onglet par défaut : Vérifications, tableaux par élément ; Tableau 3.3 replié, verdict dans le titre",
    at.session_state["asm_ui_onglet"] == "Vérifications" and "Cisaillement – groupe S" in t and "AILE A : Ed / Rd" in t
    and "Poutre secondaire (portée)" in t and "Cornières — aile A / aile B" in t
    and "Pinces et entraxes (Tab. 3.3) : conforme" in expanders(at))
chk("légende des natures après les tables, références dans un repli ; périmètre replié (textes intégraux)",
    "formule de l'Eurocode" in t and "Légende et références" in expanders(at)
    and "Références : EC3 = EN 1993-1-8" in t and "torsion induite par une attache d'un seul côté" in t
    and "Périmètre de la vérification — 4 non revérifiés · 16 sans objet" in expanders(at)
    and "Paramètres retenus" in expanders(at))
chk("matrice des cornières serrée sous 1181 px et défilante au téléphone (tableau_html(serre=1, defilant=True)) ; "
    "colonne Réf. sans commentaire interne",
    "<div class='tab-serre' style='overflow-x:auto;'><table" in t and "AILE A : Ed / Rd · η</th>" in t
    and "même modèle que" not in t)
chk("l'écran invite à toucher les pièces et à glisser la poutre",
    "touche une pièce" in t and "glisse la poutre portée" in t)

# ================================================================
print("\n=== 3. Panneaux d'objet ===")
u = {k: at.session_state["asm_" + k] for k in interface.CLES}
R = moteur.compute(u)
pan = ecran_resultats._panneaux(R, u)
chk("panneaux : groupes S et P, boulons, poutres, cornières, efforts",
    {"S", "P", "bolts", "beamS", "beamP", "cleat", "efforts"} <= set(pan))
cles_de = {g: [c["key"] for c in d["champs"]] for g, d in pan.items()}
chk("poutre secondaire : profilé, nuance, maintien, rayon du grugeage — et gh, Δz, dc,sup, dc,inf, c",
    cles_de["beamS"] == ["prof_S", "nu_S", "lt_ok", "r_n", "g_h", "d_top", "d_nt", "d_nb", "l_n"])
chk("cornières : cornière, nuance, orientation ET les deux fixations",
    cles_de["cleat"] == ["corn_u", "nu_C", "orient", "fix_P", "fix_S"])
chk("boulons : diamètre, classe, trou, catégorie, filetage (catégorie A : sans μ ni ks)",
    cles_de["bolts"] == ["boulon_u", "classe", "trou", "cat", "filet"])
chk("efforts : VEd, NEd, HEd, MEd", cles_de["efforts"] == ["V_Ed", "N_Ed", "H_Ed", "M_Ed"])
chk("fenêtres d'objet : titres courts, libellés LIBELLE (une forme par clé), aides AIDES",
    [pan[k]["titre"] for k in ("S", "P", "bolts", "beamS", "beamP", "cleat", "efforts")]
    == ["Groupe S", "Groupe P", "Boulons", "Poutre secondaire", "Poutre principale", "Cornières", "Efforts"]
    and all(c["lab"] == ecran_saisie.LIBELLE[c["key"]] for g in pan.values() for c in g["champs"])
    and all(c["help"] == ecran_saisie.AIDES.get(c["key"]) for g in pan.values() for c in g["champs"]))
orient_f = next(c for c in pan["cleat"]["champs"] if c["key"] == "orient")
chk("fenêtre Cornières : orientation toujours là, options affichées « Principale / Secondaire » (valeurs brutes "
    "inchangées) ; les deux fixations ne se confondent pas",
    orient_f["affichage"] == {ORI_P: "Principale", ORI_S: "Secondaire"} and orient_f["options"] == [ORI_P, ORI_S]
    and [c["lab"] for c in pan["cleat"]["champs"] if c["key"] in ("fix_P", "fix_S")]
    == ["Fixation ailes A", "Fixation ailes B"])
with open("acier/composants/svg_cliquable/static/index.html", encoding="utf-8") as fh:
    js = fh.read()
chk("composant : une fenêtre affiche l'option courte (affichage) et renvoie la valeur brute",
    "c.affichage[o]" in js and "op.value = o" in js)
pan_b = ecran_resultats._panneaux(moteur.compute(dict(cat="B")), dict(u, cat="B"))
chk("catégorie B : μ, ks et ELS/ELU s'ajoutent au panneau des boulons",
    [c["key"] for c in pan_b["bolts"]["champs"]][-3:] == ["mu_s", "k_s", "k_ser"])
up = dict(u, prof_S="Personnalisé")
pan_p = ecran_resultats._panneaux(moteur.compute(up), up)
chk("profilé personnalisé : h, b, tw, tf, r apparaissent dans le panneau",
    [c["key"] for c in pan_p["beamS"]["champs"]][2:7] == ["hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"])
uw = dict(u, fix_S="Soudée")
pan_w = ecran_resultats._panneaux(moteur.compute(uw), uw)
chk("côté soudé : panneau des cordons (gorge, retours, fixation), plus de groupe S",
    "S" not in pan_w and [c["key"] for c in pan_w["weldS"]["champs"]] == ["a_S", "lh_S", "fix_S"])
uwp = dict(u, fix_P="Soudée")
pan_wp = ecran_resultats._panneaux(moteur.compute(uwp), uwp)
chk("fenêtres Cordons A / B à titres courts ; a_P et a_S seulement dans leur fenêtre ; ℓh ailes A (mm)",
    pan_w["weldS"]["titre"] == "Cordons B" and pan_wp["weldP"]["titre"] == "Cordons A"
    and [c["lab"] for c in pan_wp["weldP"]["champs"]][:2] == ["a ailes A (mm)", "ℓh ailes A (mm)"]
    and sum(1 for g in list(pan_wp.values()) + list(pan_w.values()) for c in g["champs"] if c["key"] in ("a_P", "a_S")) == 2)
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT"))
pan_pred = ecran_resultats._panneaux(Rp, dict(u, mode_calc="PRÉDIMENSIONNEMENT"))
chk("prédimensionnement : boulon et rangées désactivés (◆), profilé et efforts libres",
    pan_pred["bolts"]["champs"][0]["dis"] and pan_pred["S"]["champs"][0]["dis"]
    and not pan_pred["beamS"]["champs"][0]["dis"] and not pan_pred["efforts"]["champs"][0]["dis"])

# ================================================================
print("\n=== 4. Recalcul et alertes ===")
message(at, {"V_Ed": 400}, 1001); run(at)
chk("panneau efforts : VEd = 400 kN → ASSEMBLAGE NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
message(at, {"V_Ed": 125, "LC_u": 260}, 1002); run(at)
t = md(at)
chk("hc = 260 → alerte courte « Impossible — hauteur disponible insuffisante »",
    "Impossible — hauteur disponible insuffisante" in t and "zc + hc = 50 + 260 = 310 mm" in t)
at.button(key="asm_btn_al_h_dispo").click(); run(at)
t = md(at)
chk("« Localiser » : explication chiffrée en notation Eurocode (h − (tf + r) = 249 mm)",
    "zc + hc = 50 + 260 = 310 mm" in t and "= 249 mm" in t)
chk("chip d'alerte : hc modifiable sur place (asm_fix_LC_u, libellé hc)",
    any(w.key == "asm_fix_LC_u" and w.label.startswith("hc") for w in at.number_input))
at.number_input(key="asm_fix_LC_u").set_value(190.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["asm_LC_u"]) == 190.0 and "hauteur disponible insuffisante" not in md(at))

# ================================================================
print("\n=== 5. Source unique : tous les gestes du dessin ===")
message(at, {"z_C": 60}, 2001); run(at)
chk("cote zc → asm_z_C = 60", float(at.session_state["asm_z_C"]) == 60.0 and not at.exception)
message(at, {"n1S_u": 4}, 2002); run(at)
chk("poignée + du groupe S → 4 rangées", float(at.session_state["asm_n1S_u"]) == 4.0)
message(at, {"n1S_u": 3, "p1S_u": 70, "e1S_u": 40}, 2003); run(at)
chk("fenêtre de groupe → n1, p1, e1 écrits d'un coup",
    float(at.session_state["asm_p1S_u"]) == 70.0 and float(at.session_state["asm_e1S_u"]) == 40.0)
message(at, {"p1S_u": 60, "e1S_u": 35, "z_C": 50, "g_h": 25}, 2004); run(at)
chk("glissement de la poutre portée → gh = 25, excentricité recalculée (η max 88,5 %)",
    float(at.session_state["asm_g_h"]) == 25.0 and "88,5 %" in md(at), md(at)[:120])
message(at, {"prof_S": "HEA 320", "g_h": 10}, 2005); run(at)
chk("panneau de pièce : profilé secondaire HEA 320 appliqué (statut recalculé)",
    at.session_state["asm_prof_S"] == "HEA 320" and "ASSEMBLAGE VÉRIFIÉ" in md(at))
message(at, {"prof_S": "HEA 300"}, 2006); run(at)

# ================================================================
print("\n=== 6. Prédimensionnement ===")
at.segmented_control(key="asm_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
chk("le mode se règle dans l'en-tête de la carte (segmented_control) ; bandeau prédimensionnement",
    "Mode prédimensionnement" in md(at))
chk("en prédim : Taux cible · Ø minimal · Ø maximal dans la rangée d'en-tête ; un seul signal (bande ocre + ◆), "
    "plus de caption « ◆ pilotés » ni de caption d'usage sous le dessin",
    {"asm_eta_c", "asm_pd_dmin", "asm_pd_dmax"} <= cles_widgets(at) and "◆ pilotés par la proposition" not in md(at)
    and "touche une pièce" not in md(at)
    and any(x.label.endswith("◆") for x in at.selectbox if x.key == "asm_corn_u"))
pd = REF["cas"][0]["attendu"]["predim"]
at.session_state["asm_ui_onglet"] = "Prédim"; run(at)
chk("onglet Prédim : solutions proposées ; une seule caption d'usage (« Seule la vérification détaillée fait foi »)",
    "Solutions proposées" in md(at) and md(at).count("Seule la vérification détaillée fait foi") == 1
    and "Mode PRÉDIMENSIONNEMENT actif" not in md(at))
boutons = [b for b in at.button if b.key and b.key.startswith("asm_btn_sol_")]
chk("au moins une solution cliquable", len(boutons) >= 1)
retenue = [b for b in boutons if b.type == "primary"]
(retenue[0] if retenue else boutons[0]).click(); run(at)
chk("« Appliquer » : VÉRIFICATION, boulon, rangées, hc, cornière de la solution",
    at.session_state["asm_mode_calc"] == "VÉRIFICATION" and at.session_state["asm_boulon_u"] == pd["boulon"]
    and float(at.session_state["asm_n1S_u"]) == pd["n"] and float(at.session_state["asm_LC_u"]) == pd["LC"],
    f"{at.session_state['asm_boulon_u']} {at.session_state['asm_n1S_u']} {at.session_state['asm_LC_u']}")

# ================================================================
print("\n=== 7. Repli : dessin interactif désactivé ===")
at.button(key="asm_btn_reset").click(); run(at)
at.checkbox(key="asm_ui_composant").uncheck(); run(at)
w = cles_widgets(at)
chk("dessins statiques à gauche ; le panneau des cotes s'ajoute à la carte (colonne de droite)",
    {"asm_prof_S", "asm_corn_u", "asm_boulon_u", "asm_V_Ed", "asm_n1S_u"} <= w
    and "asm_cote_LC_u" in at.session_state.to_dict() and len(at.get("component_instance")) == 0)
at.number_input(key="asm_cote_LC_u").set_value(200.0); run(at)
chk("panneau des cotes → asm_LC_u = 200 (même source)", float(at.session_state["asm_LC_u"]) == 200.0)
at.number_input(key="asm_V_Ed").set_value(400.0); run(at)
chk("carte → VEd = 400 : NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
at.number_input(key="asm_V_Ed").set_value(125.0)
at.number_input(key="asm_cote_LC_u").set_value(190.0); run(at)
at.checkbox(key="asm_ui_composant").check(); run(at)
chk("dessin interactif rétabli : deux composants, le panneau des cotes disparaît, la carte reste",
    len(at.get("component_instance")) == 2 and "asm_cote_LC_u" not in cles_widgets(at)
    and "asm_prof_S" in cles_widgets(at))

# ================================================================
print("\n=== 8. Exports, enregistrer / charger, réinitialiser ===")
at.button(key="asm_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts", float(at.session_state["asm_LC_u"]) == 190.0
    and at.session_state["asm_prof_S"] == "HEA 300")
at.text_input(key="asm_visserie").set_value("1 rondelle + 2 écrous (contre-écrou)"); run(at)
chk("la visserie saisie persiste (source unique), et l'enregistrement l'emporte",
    at.session_state["asm_visserie"] == "1 rondelle + 2 écrous (contre-écrou)"
    and "visserie" in interface._payload({}))
at.button(key="asm_btn_reset").click(); run(at)
chk("réinitialiser rétablit la visserie par défaut", at.session_state["asm_visserie"] == "1 rondelle + 1 écrou")
# Dans un PROJET l'assemblage porte un repère (A1) que 🔄 conserve ; le corrigé
# est celui du cas par défaut SANS repère : on le vide pour la comparaison.
at.session_state["asm_id_rep"] = ""
at.session_state["asm_ui_onglet"] = "Note"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Note : le texte essentiel du cas par défaut est celui du corrigé",
    code.strip() == REF["exports_texte"]["defaut"]["essentiel"].strip())
at.checkbox(key="asm_ui_txt_complet").check(); run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("version complète identique au corrigé", code.strip() == REF["exports_texte"]["defaut"]["complet"].strip())
payload = {"version": "x", "values": dict(REF["cas"][6]["inputs"], **{})}
charge = interface.charger_payload(payload)
brut = interface.charger_payload(REF["cas"][6]["inputs"])
chk("charger : format du module et fichier brut de l'outil HTML donnent le même état",
    charge == brut and charge["fix_S"] == "Soudée" and charge["N_Ed"] == 50)
at.session_state["asm_ui_onglet"] = "Benchmark"; run(at)
chk("onglet Benchmark : 4 exemples, 2 calculs manuels, statut du benchmark",
    all(x in md(at) for x in ("BM1", "BM4", "VAL-A", "VAL-B", "Benchmark du module : VALIDÉ")))
at.button(key="asm_btn_bm_0").click(); run(at)
chk("« Charger ces données » : BM1 dans l'outil (VEd = 450, hc = 430)",
    float(at.session_state["asm_V_Ed"]) == 450.0 and float(at.session_state["asm_LC_u"]) == 430.0)
at.session_state["asm_ui_onglet"] = "Méthode"; run(at)
chk("onglet Méthode : modèle mécanique et sources", "Modèle mécanique" in md(at) and "Sources" in md(at))
chk("le texte de l'ancien bouton ❔ vit dans Méthode « Ce que l'outil vérifie » ; « Données de bibliothèque » "
    "reçoit les aides de la nuance et de la cornière",
    "## Ce que l'outil vérifie" in md(at) and "Parité prouvée avec l'outil de référence" in md(at)
    and "### Données de bibliothèque" in md(at) and "« S355 (fu 490) »" in md(at)
    and "à contrôler avec le catalogue du fournisseur" in md(at))
at.session_state["asm_ui_onglet"] = "Vérifications"; run(at)
t = md(at)
chk("onglet Vérifications : tableaux, formules et substitutions, sans objet",
    all(x in t for x in ("Cisaillement – groupe S", "Poutre principale (porteuse)", "Fb,ver,Rd = k1,ver·αb,ver·fu·d·tw/γM2",
                         "Sans objet dans cette configuration")))

# ================================================================
print("\n=== 9. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict() if k.startswith(("b1_", "dal", "pre", "meta_"))]
chk("aucune clé béton créée par le module", not fuites, str(fuites[:5]))
at.button(key="asm_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

# ================================================================
print("\n=== 10. Charte (les contrôles du kit sur les doubles cornières) ===")
D = ecran_saisie.D
u0 = entrees.defaults()
R0 = moteur.compute(dict(u0))
cles_carte = carte.cles_carte(D)
cles_fenetres = {c["key"] for g in ecran_resultats._panneaux(R0, u0).values() for c in g["champs"]}
attendues = set(entrees.CHAMPS) | set(ecran_saisie.GEOMETRIE_DESSIN) | {"ui_niveau", "ui_composant"}
atteignables = cles_carte | set(ecran_saisie.GEOMETRIE_DESSIN) | cles_fenetres | set(ecran_resultats.CLES_TETE_DESSIN)
manquantes = sorted(attendues - atteignables)
chk("rien ne se perd : les 87 clés de l'inventaire (CHAMPS ∪ GEOMETRIE_DESSIN ∪ {ui_niveau, ui_composant} + "
    "visserie au cartouche) ont une destination",
    len(attendues) == 86 and "visserie" in cles_carte and not manquantes, f"{len(attendues)} clés, manquantes {manquantes}")
chk("GEOMETRIE_DESSIN = les 19 cotes de CLE_PAR_COTE (lh_P compris) ; ∩ carte = ∅ ; ONGLETS inchangés",
    len(ecran_saisie.GEOMETRIE_DESSIN) == 19 and "lh_P" in ecran_saisie.GEOMETRIE_DESSIN
    and not (cles_carte & set(ecran_saisie.GEOMETRIE_DESSIN))
    and ecran_resultats.ONGLETS == ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode"))
sans_aide = [k for k in entrees.CHAMPS if k not in D.AIDES and carte.aide(D, k) is not None]
longues = [k for k, v in D.AIDES.items() if len(v) > 160]
chk("aide(k) is None hors AIDES ; toute aide ≤ 160", not sans_aide and not longues, str(sans_aide + longues))
chk("aucun « ? » sur Nuance, Profilé, Cornière, Boulon, Classe, Fixation, dimensions, efforts, identification",
    not ({"nu_P", "nu_S", "nu_C", "prof_P", "prof_S", "corn_u", "boulon_u", "classe", "fix_P", "fix_S", "n1S_u",
          "hP_u", "hS_u", "k1_u", "V_Ed", "N_Ed", "H_Ed", "M_Ed", "id_rep", "id_red"} & set(D.AIDES)))
trop = [(k, v) for k, v in D.LIBELLE.items() if len(v) > 22]
r4 = [(k, D.LIBELLE[k]) for genre, cles in carte.rangees(D) if genre == "objet" and len(cles) == 4
      for k in cles if len(D.LIBELLE[k]) > 16]
r3 = [(k, D.LIBELLE[k]) for genre, cles in carte.rangees(D) if (genre == "suite" or len(cles) == 3)
      for k in cles if len(D.LIBELLE[k]) > 20]
chk("LIBELLE ≤ 22 partout, ≤ 16 dans une rangée de 4, ≤ 20 dans une rangée de 3 ou une continuation",
    not trop and not r4 and not r3, str(trop + r4 + r3))


def _mot(v):
    t_ = re.match(r"[A-Za-zÀ-ÿ]+", v)
    return bool(t_) and len(t_.group(0)) >= 3


chk("capitale initiale des mots ; unités tirées de UNITE ; ℓh ailes A (plus de libellé de 46 caractères)",
    not [v for v in D.LIBELLE.values() if _mot(v) and v[:1] != v[:1].upper()]
    and D.LIBELLE["V_Ed"] == "VEd (kN)" and D.LIBELLE["M_Ed"] == "MEd (kNm)" and D.LIBELLE["LC_u"] == "hc (mm)"
    and D.LIBELLE["lh_P"] == "ℓh ailes A (mm)" and D.LIBELLE["r_n"] == "Rayon grugeage (mm)")
longues = [(k, o) for k, f in entrees.CHAMPS.items() for o in (f.get("o") or []) if len(carte.affiche(D, k, o)) > 28]
chk("options affichées ≤ 28 ; AFFICHAGE ⊂ options brutes ; Grande aile côté → Principale / Secondaire",
    not longues and all(set(v) <= set(entrees.CHAMPS[k]["o"]) for k, v in D.AFFICHAGE.items())
    and carte.affiche(D, "orient", ORI_P) == "Principale" and carte.affiche(D, "orient", ORI_S) == "Secondaire"
    and carte.affiche(D, "mode_calc", "PRÉDIMENSIONNEMENT") == "Prédimensionnement", str(longues))
chk("aucune ligne d'objet > 4 widgets ; les fixations sont la 1re case de la continuation de leur groupe",
    not [c for g_, c in carte.rangees(D) if g_ == "objet" and len(c) > 4]
    and ("suite", ["fix_P", "n1P_u", "n2P_u"]) in carte.rangees(D)
    and ("suite", ["fix_S", "n1S_u", "n2S_u"]) in carte.rangees(D))
vis = ecran_saisie.aides_visibles(u0)
chk("« ? » visibles dans la carte au défaut : 5 (mode, trou, catégorie, ↳ AILES B, EFFORTS ELU) au lieu de 22",
    len(vis) == 5 and set(vis) == {"mode_calc", "trou", "cat", "AILES B", "EFFORTS ELU"}, str(vis))
inegales = sorted(c["n"] for c in DB["cornieres"] if entrees.champ_visible(dict(u0, corn_u=c["n"]), "orient"))
chk("« Grande aile côté » pour les 5 cornières inégales de DB.cornieres et Personnalisé seulement — jamais en "
    "prédim (cornière proposée à ailes égales : le moteur ignore l'orientation)",
    inegales == sorted(["L100x75x8", "L120x80x8", "L120x80x10", "L150x90x10", "L150x100x10"])
    and len(DB["cornieres"]) == 18 and entrees.champ_visible(dict(u0, corn_u="Personnalisé"), "orient")
    and not entrees.champ_visible(dict(u0, corn_u="L150x90x10", mode_calc="PRÉDIMENSIONNEMENT"), "orient"),
    str(inegales))
chk("grugeage nommé c / dc,sup / dc,inf dans GROUPES (vue) et LIBELLE ; GROUPES sans note morte",
    "dc,sup" in entrees.CHAMPS["d_nt"]["l"] and "dc,inf" in entrees.CHAMPS["d_nb"]["l"]
    and " c " in entrees.CHAMPS["l_n"]["l"] and D.LIBELLE["l_n"] == "c (mm)" and D.LIBELLE["d_nt"] == "dc,sup (mm)"
    and not any(g_.get("note") for g_ in entrees.GROUPES))
at3 = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at3.session_state["page"] = "Assemblages métalliques"
at3.session_state["asm_courant"] = "doubles_cornieres"
at3.session_state["asm_corn_u"] = "L150x90x10"
run(at3)
orient_w = [x for x in at3.selectbox if x.key == "asm_orient"]
chk("cornière inégale (L150x90x10) : « Grande aile côté » dans la carte, options « Principale / Secondaire »",
    bool(orient_w) and orient_w[0].label == "Grande aile côté"
    and orient_w[0].options == ["Principale", "Secondaire"] and orient_w[0].value == ORI_P)


# ================================================================
# Consolidation du 01/11/2026, point 6 : le Tableau 3.3 NON CONFORME — un contrôle par module. L'expander
# s'ouvre d'office (le verdict est dans son titre) et « Localiser » sélectionne l'alerte de la pince.
print("\n=== 11. Tableau 3.3 NON CONFORME ===")
at33 = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at33.session_state["page"] = "Assemblages métalliques"
run(at33)
at33.button(key="asm_btn_open_doubles_cornieres").click(); run(at33)
at33.session_state["asm_e1S_u"] = 20.0
run(at33)
e33 = [x for x in at33.expander if str(x.label).startswith("Pinces et entraxes (Tab. 3.3)")]
chk("e1 groupe S = 20 mm → « Pinces et entraxes (Tab. 3.3) : NON CONFORME », expander ouvert d'office",
    bool(e33) and str(e33[0].label) == "Pinces et entraxes (Tab. 3.3) : NON CONFORME" and e33[0].proto.expanded,
    str([str(x.label) for x in e33]))
at33.button(key="asm_btn_dist_S_e1c").click(); run(at33)
chk("« Localiser » de la ligne NON OK sélectionne son alerte (dist_S_e1c) ; sa rangée dit « Masquer »",
    at33.session_state["asm_ui_alerte"] == "dist_S_e1c" and at33.button(key="asm_btn_al_dist_S_e1c").label == "Masquer",
    str(at33.session_state["asm_ui_alerte"]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

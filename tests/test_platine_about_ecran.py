# -*- coding: utf-8 -*-
"""Écran, dessins et note du module platine d'about (AppTest, application
réelle).

Chaque garantie rougit si on la retire :
  1. NAVIGATION : la carte « Platine d'about (hauteur partielle) » est
     DISPONIBLE sur la page de sélection, « Ouvrir » mène au module.
  2. ÉCRAN : titre court + barre d'outils partagée, deux colonnes, colonne
     du dessin FIGÉE (CSS sticky, conteneur pab_col_dessin), deux composants,
     carte par objet à droite rendue par le kit (mode en segmented_control,
     « Dessin interactif » en tête du dessin, visserie au cartouche) SANS la
     géométrie (elle vit sur le dessin) ; nature de NEd seulement si NEd > 0 ;
     onglet Vérifications : légende, périmètre et Tableau 3.3 dans des replis.
  3. SOURCE UNIQUE : un message du dessin (cote, glissement vertical,
     panneau) écrit dans pab_<clé> et recalcule ; VEd = 450 → NON VÉRIFIÉ ;
     NEd > 0 fait apparaître la nature, Tying fait apparaître γMu.
  4. PRÉDIM : solutions cliquables, « Appliquer » recopie et repasse en
     VÉRIFICATION.
  5. DESSINS : SVG valides (3 vues × écran/fabrication) ; la portée ne
     glisse QUE verticalement (pas de jeu gh) ; renvois de perçage, zt ;
     répartition des cotes SANS doublon sur le plan de principe ;
     composant : glissement vertical seul pris en charge.
  6. NOTE : 2 pages A4 paysage sans débordement (défaut + cas figés),
     échelle normalisée, cartouche PLATINE + visserie + rédacteur, γM0 ANB ;
     HYPOTHÈSES ≤ 3 lignes ≤ 110 (textes exacts de la charte).
  7. BENCHMARK dans l'application : VALIDÉ ; exemple chargé ; exports (le
     rédacteur saisi figure dans l'export texte).
  8. ÉTANCHÉITÉ : aucune clé asm_ / fpl_ / béton créée.
  9. CHARTE : rien ne se perd (66 clés), aides ≤ 160 et seulement depuis
     AIDES, libellés ≤ 22 / 16 / 20, options ≤ 28, aucune ligne d'objet > 4,
     4 « ? » au défaut, HYPOTHÈSES sur les 13 cas figés sans segment perdu.

Lancement : python3 tests/test_platine_about_ecran.py
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.assemblages import carte  # noqa: E402
from acier.assemblages.poutre_poutre.platine_about import (interface, ecran_saisie, ecran_resultats,  # noqa: E402
                                                           entrees, moteur, schemas, note, texte)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


def run(at):
    at.run()
    if at.exception:
        chk("aucune exception", False, str(at.exception[0].message)[:200])
    return at


def cles_widgets(at):
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox)
            + list(at.segmented_control) if w.key}


def expanders(at):
    return [str(x.label) for x in at.expander]


def message(at, changes, t):
    derniers = at.session_state["pab_ui_dernier_clic"]
    if derniers.get("pab_cmp_elev") != t:
        derniers["pab_cmp_elev"] = t
        for k, v in changes.items():
            at.session_state["pab_" + k] = ecran_saisie.valeur_widget(k, v)


# ================================================================
print("=== 1. Navigation ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at.session_state["page"] = "Assemblages métalliques"
run(at)
chk("la carte « Platine d'about (hauteur partielle) » est sur la page de sélection, ouvrable",
    "Platine d'about (hauteur partielle)" in md(at)
    and any(b.key == "asm_btn_open_platine_about" for b in at.button))
at.button(key="asm_btn_open_platine_about").click(); run(at)
chk("« Ouvrir » mène au module", not at.exception and at.session_state["asm_courant"] == "platine_about")
chk("en-tête du module", "platine d'about de hauteur partielle" in md(at))

# ================================================================
print("\n=== 2. Écran ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("pab_") and k[4:] in interface.CLES]
chk(f"les {len(interface.CLES)} clés d'entrée existent en session", len(cles) == len(interface.CLES), str(len(cles)))
t = md(at)
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ, 51,7 % (stabilité du grugeage), dimensionnante nommée comme dans la note",
    "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t and "51,7 %" in t
    and "Grugeage – stabilité (c)" in t)
chk("taux par élément : Boulons, Platine, Portée, Porteuse, Soudure",
    all(x in t for x in ("Boulons", "Platine", "Portée", "Porteuse", "Soudure")))
chk("deux composants empilés dans la colonne figée (CSS sticky sur pab_col_dessin)",
    len(at.get("component_instance")) == 2 and "position: sticky" in t and "st-key-pab_col_dessin" in t)
w = cles_widgets(at)
donnees = {k for k in w if k.startswith("pab_") and k[4:] in interface.CLES}
chk("la géométrie ne se règle QUE sur le dessin (aucune cote dans la carte)",
    not donnees & {"pab_" + k for k in ecran_saisie.GEOMETRIE_DESSIN}, str(sorted(donnees)))
chk("carte par objet : mode, profilés, nuance de la platine, boulons, rangées, efforts, visserie",
    {"pab_mode_calc", "pab_prof_P", "pab_prof_S", "pab_nu_pl", "pab_boulon_u", "pab_n1_u",
     "pab_V_Ed", "pab_N_Ed", "pab_visserie", "pab_g_M0"} <= w)
chk("la nature de NEd n'apparaît qu'avec NEd > 0 (au défaut NEd = 0 : une compression n'a pas de nature)",
    "pab_nat_N" not in w and at.session_state["pab_nat_N"] == entrees.NAT_ELU)
chk("γMu n'apparaît qu'en tying", "pab_g_Mu" not in w)
chk("le mode est un segmented_control (clé pab_mode_calc, valeurs brutes inchangées)",
    any(x.key == "pab_mode_calc" for x in at.segmented_control)
    and at.session_state["pab_mode_calc"] == "VÉRIFICATION")
chk("« Dessin interactif » en tête de la colonne dessin (clé pab_ui_composant), titre court et barre partagée",
    "pab_ui_composant" in w and "Platine d'about partielle" in t and "version 1.0" in t
    and {"pab_btn_retour", "pab_btn_home", "pab_btn_save_dl", "pab_btn_pdf", "pab_btn_reset"}
    <= {b.key for b in list(at.button) + list(at.get("download_button"))})
chk("plus de bouton ❔ ni de cellule Version", not any(b.key == "pab_btn_version" for b in at.button)
    and "Version 1.0" not in t)
chk("onglet par défaut : Vérifications (groupe de boulons) ; Tableau 3.3 replié, verdict dans le titre",
    at.session_state["pab_ui_onglet"] == "Vérifications"
    and "Groupe : cisaillement et pression diam." in t and "Pinces et entraxes (Tab. 3.3) : conforme" in expanders(at))
chk("la limite « flexion TRANSVERSALE de l'âme porteuse » est écrite en clair (expander Périmètre)",
    "flexion TRANSVERSALE de l'âme porteuse" in t
    and any(x.startswith("Périmètre de la vérification — 5 non revérifiés · 6 sans objet") for x in expanders(at)))
chk("légende des natures après les tables, références dans un repli",
    "formule de l'Eurocode" in t and "Légende et références" in expanders(at)
    and "Références : EC3 = EN 1993-1-8" in t)
chk("colonne Réf. sans commentaire interne", "même modèle que" not in t)
chk("le texte de l'ancien bouton ❔ vit dans Méthode « Ce que l'outil vérifie » ; « Données de bibliothèque » ajouté",
    "#### Ce que l'outil vérifie" in ecran_resultats.METHODE and "tying §2.3" in ecran_resultats.METHODE
    and "#### Données de bibliothèque" in ecran_resultats.METHODE)

# ================================================================
print("\n=== 3. Source unique ===")
message(at, {"V_Ed": 450}, 1001); run(at)
chk("message du dessin : VEd = 450 → NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
message(at, {"V_Ed": 125, "hp_u": 140}, 1002); run(at)
chk("hp = 140 → alerte bloquante pince basse, « Localiser » disponible",
    "pince basse" in md(at) and any(b.key == "pab_btn_al_e1bot" for b in at.button))
at.button(key="pab_btn_al_e1bot").click(); run(at)
chk("chip d'alerte : hp modifiable sur place", any(x.key == "pab_fix_hp_u" for x in at.number_input))
at.number_input(key="pab_fix_hp_u").set_value(190.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["pab_hp_u"]) == 190.0 and "pince basse" not in md(at))
message(at, {"d_top": 10}, 1003); run(at)
chk("glissement vertical Δz → clé écrite, recalcul", float(at.session_state["pab_d_top"]) == 10.0
    and not at.exception)
message(at, {"d_top": 0}, 1004); run(at)
at.number_input(key="pab_N_Ed").set_value(60.0); run(at)
chk("NEd = 60 > 0 : la nature de NEd apparaît dans la carte (ELU / Tying), γMu toujours absent",
    "pab_nat_N" in cles_widgets(at) and "pab_g_Mu" not in cles_widgets(at))
at.selectbox(key="pab_nat_N").set_value("Tying (accidentel)"); run(at)
chk("nature Tying : γMu apparaît dans les paramètres avancés (4e case de COEFF. γ)",
    "pab_g_Mu" in cles_widgets(at) and at.session_state["pab_nat_N"] == "Tying (accidentel)")
at.number_input(key="pab_N_Ed").set_value(0.0); run(at)
chk("NEd = 0 : nature et γMu disparaissent, la valeur de la nature reste en session (fichier conservé)",
    "pab_nat_N" not in cles_widgets(at) and "pab_g_Mu" not in cles_widgets(at)
    and at.session_state["pab_nat_N"] == "Tying (accidentel)")
at.session_state["pab_nat_N"] = entrees.NAT_ELU; run(at)

# ================================================================
print("\n=== 4. Prédim ===")
at.segmented_control(key="pab_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
chk("bandeau prédimensionnement", "Mode prédimensionnement" in md(at))
chk("en prédim : Taux cible · Ø minimal · Ø maximal dans la rangée d'en-tête, plus de caption « ◆ pilotés »",
    {"pab_eta_c", "pab_pd_dmin", "pab_pd_dmax"} <= cles_widgets(at) and "◆ pilotés par la proposition" not in md(at))
at.session_state["pab_ui_onglet"] = "Prédim"; run(at)
boutons = [b for b in at.button if b.key and b.key.startswith("pab_btn_sol_")]
chk("solutions proposées cliquables", len(boutons) >= 1)
retenue = [b for b in boutons if b.type == "primary"]
(retenue[0] if retenue else boutons[0]).click(); run(at)
lu = {k: at.session_state.get("pab_" + k) for k in interface.CLES}
Rl = moteur.compute(lu)
chk("« Appliquer » : VÉRIFICATION + platine et boulons recopiés, vérifiés",
    at.session_state["pab_mode_calc"] == "VÉRIFICATION" and Rl.verified
    and 8 <= float(at.session_state["pab_tp_u"]) <= Rl.tp_max + 1e-9)
at.button(key="pab_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts", float(at.session_state["pab_hp_u"]) == 190.0
    and float(at.session_state["pab_p3_u"]) == 90.0)

# ================================================================
print("\n=== 5. Dessins ===")
R0 = moteur.compute(dict())
for nom, fab in (("écran", False), ("fabrication", True)):
    for f, excl in (("elevation", schemas.EXCLURE_ELEVATION), ("plan", schemas.EXCLURE_PLAN),
                    ("vue_droite", schemas.EXCLURE_DROITE)):
        o = schemas.options_fabrication(excl, 12) if fab else schemas.options_ecran(R0)
        svg = getattr(schemas, f)(R0, o).svg(identifiant="t")
        ET.fromstring(svg)
poutre = next(p for p in schemas.elevation(R0, schemas.options_ecran(R0)).corps
              if p.get("t") == "g" and p.get("grp") == "beamS")
chk("SVG valides (3 vues × écran/fabrication) ; la portée glisse VERTICALEMENT seulement (pas de jeu gh)",
    poutre.get("drag2") == "d_top" and not poutre.get("drag"))
svg_ec = schemas.elevation(R0, schemas.options_ecran(R0)).svg()
chk("élévation écran : glissement vertical seul exposé (data-drag2 sans data-drag), poignées n1",
    'data-drag2="d_top"' in svg_ec and "data-drag=" not in svg_ec
    and 'data-action="n1_u:+1"' in svg_ec and 'data-action="n1_u:-1"' in svg_ec)
with open("acier/composants/svg_cliquable/static/index.html", encoding="utf-8") as fh:
    js = fh.read()
chk("composant : les pièces à glissement vertical seul sont prises en charge",
    "g.ed[data-drag], g.ed[data-drag2]" in js and "kx ? Math.max" in js)
sd = schemas.vue_droite(R0, schemas.options_fabrication(schemas.EXCLURE_DROITE, 12))
face = next(p for p in sd.corps if p["t"] == "poly" and "pp" in p["cls"])
xs = sorted(pt[0] for pt in face["pts"])
chk("vue de droite : âme porteuse rompue des deux côtés, platine et coupe de la portée",
    xs[0] < xs[1] - 1.5 and xs[-1] > xs[-2] + 1.5
    and sum(1 for p in sd.corps if p["t"] == "rect" and "co" in p["cls"]) == 1
    and sum(1 for p in sd.corps if p["t"] == "circle" and "bo" in p["cls"]) == 6)
se = schemas.elevation(R0, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)).svg()
sp = schemas.plan(R0, schemas.options_fabrication(schemas.EXCLURE_PLAN, 12)).svg()
sdv = sd.svg()
chk("fabrication : renvois 6×Ø22 (élévation, vue de droite) et r 10 ; zt sur la vue de droite",
    ">6×Ø22<" in se and ">6×Ø22<" in sdv and ">r 10<" in se and ">zt 50<" in sdv)
chk("fabrication : pas d'étiquette de cote nulle (dc,inf 0, Δz 0)", ">dc,inf 0<" not in se and ">Δz 0<" not in se)
cotes = ("hp = 190", "zp 50", "p3 = 90", "bp = 150", "c = 150", "tp 10", "e1 35")
doubles = [c for c in cotes if (se + sp + sdv).count(">" + c + "<") > 1]
absentes = [c for c in cotes if (se + sp + sdv).count(">" + c + "<") == 0]
chk("chaque cote de fabrication posée UNE fois sur la planche (hp, zp, p3, bp, c, tp, e1)",
    not doubles and not absentes, f"doubles {doubles} absentes {absentes}")


def aretes_cachees_ok(R):
    """Arêtes cachées de la vue en plan : horizontales tracées de gauche à
    droite et dans le champ ; verticales (bord de semelle porteuse) à b/2
    exactement — jamais rabattues dans le champ."""
    d = schemas.plan(R, schemas.options_ecran(R))
    xE = d.viewbox[0] + d.viewbox[2]
    for p in d.corps:
        if p.get("t") != "path" or "hd" not in p.get("cls", []):
            continue
        for (xa, ya), (xb, yb) in (l for l in p["lignes"]):
            if abs(ya - yb) < 1e-9 and not (xa < xb <= xE):
                return False
            if abs(xa - xb) < 1e-9 and abs(xa - R.b_P / 2) > 1e-9:
                return False
    return True


chk("vue en plan : arêtes cachées dans le champ, dans le bon sens, bord de semelle à b/2 (défaut, "
    "grugé en bas, porteuse étroite)",
    aretes_cachees_ok(R0) and aretes_cachees_ok(moteur.compute(dict(d_nb=40, l_n=60)))
    and aretes_cachees_ok(moteur.compute(dict(prof_P="IPE 300", d_top=50, d_nt=0, l_n=0, z_C=55))))
g = ecran_resultats._panneaux(R0, entrees.defaults())
chk("fenêtres d'objet : titres courts, libellés LIBELLE (une forme par clé), aides AIDES",
    [g[k]["titre"] for k in ("G", "bolts", "beamS", "beamP", "plate", "weld", "efforts")]
    == ["Groupe", "Boulons", "Poutre secondaire", "Poutre principale", "Platine", "Soudure", "Efforts"]
    and all(c["lab"] == ecran_saisie.LIBELLE[c["key"]] for grp in g.values() for c in grp["champs"])
    and all(c["help"] == ecran_saisie.AIDES.get(c["key"]) for grp in g.values() for c in grp["champs"]))
chk("fenêtre Platine sans a_w (doublon retiré) ; Soudure le garde ; Poutre secondaire reçoit Δz, dc,sup, dc,inf, c",
    "a_w" not in {c["key"] for c in g["plate"]["champs"]} and [c["key"] for c in g["weld"]["champs"]] == ["a_w"]
    and {"d_top", "d_nt", "d_nb", "l_n", "lt_ok", "r_n"} <= {c["key"] for c in g["beamS"]["champs"]})

# ================================================================
print("\n=== 6. Note (2 pages) ===")
R = moteur.compute(dict(id_projet="Halle A", id_rep="P3/S9", id_red="J. Dupont"))
pdf = note.generer_note(R, {"nom_projet": "Halle A", "partie": "P3/S9", "date": "30/10/2026",
                            "indice": "0", "visserie": "1 rondelle + 1 écrou"})
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("deux pages A4 paysage, sans débordement, échelle et disposition exposées",
    doc.page_count == 2 and not note.derniers_avertissements
    and note.derniere_echelle is not None and note.derniere_disposition is not None,
    str(note.derniers_avertissements))
t1 = doc[0].get_text(); s1 = t1.replace(" ", "").replace("\n", "")
chk("page 1 : bandeau, en-tête compact, hypothèses, tableaux, conclusion",
    "VÉRIFIÉ" in s1 and "Platine 190×150×10 S355" in t1 and "Groupe 3 × 2 (p3 90, 1 plan)" in t1
    and "VEd 125,0 kN" in t1 and "rotule à la face de l'âme porteuse" in t1
    and "PLATINE D'ABOUT" in t1.upper() and "ASSEMBLAGE VÉRIFIÉ à l'ELU" in t1)
chk("page 1 : conclusion avec γM0 ANB (pas de γM1 : aucune instabilité ici)",
    "(γM0 : ANB belge)" in t1 and "γM1" not in t1)
lignes = note.lignes_hypotheses(R)
chk("HYPOTHÈSES : ≤ 3 lignes de ≤ 110 caractères (note_kit.empaqueter)",
    1 <= len(lignes) <= 3 and all(len(l) <= 110 for l in lignes), str([len(l) for l in lignes]))
chk("HYPOTHÈSES : textes exacts de la charte (défaut : NEd = 0, grugeage, cat A)",
    lignes == ["Articulé : rotule à la face de l'âme porteuse (MSB P5 §2.2) ; groupe centré, VEd/n par boulon.",
               "Platine : hp ≥ 1,36·p3 → rupture de bloc, k = 1,0 ; grugeage : section réduite, bras tp + c "
               "(MSB P5 §4.2.4).",
               "Cordons sur hp (§4.5.3.3), pleine résistance 0,55·tw = 4,7 mm ; γM0 1,00 (ANB) · γM2 1,25."],
    str(lignes))
l_elu = note.lignes_hypotheses(moteur.compute(dict(N_Ed=60, cat="B")))
l_ty = note.lignes_hypotheses(moteur.compute(dict(N_Ed=60, nat_N=entrees.NAT_TYING, p3_u=140, bp_u=200, hp_u=180,
                                                   n1_u=2, p1_u=80, e1_u=50)))
l_ty2 = note.lignes_hypotheses(moteur.compute(dict(N_Ed=60, nat_N=entrees.NAT_TYING, d_nt=0, l_n=0, cat="B", g_M2n=1.2)))
chk("HYPOTHÈSES : variantes de la charte (NEd ELU cat B ; tying k = 0,5 ; tying sans grugeage, γM2 ≠ γM2,net)",
    l_elu[1:] == ["Platine hp ≥ 1,36·p3 → k = 1,0 ; grugeage : bras tp + c (§4.2.4) ; NEd ELU avec VEd (§6.2.4).",
                  "Cordons hp, pleine résistance 0,55·tw = 4,7 mm ; γM0 1,00 (ANB) · γM2 1,25 · γM3 1,25."]
    and l_ty[1:] == ["Platine hp < 1,36·p3 → k = 0,5 ; grugeage : bras tp + c (§4.2.4) ; tying : chaînage seul, "
                     "fu/γMu (§2.3).",
                     "Cordons hp, pleine résistance 0,55·tw = 4,7 mm ; γM0 1,00 (ANB) · γM2 1,25 · γMu 1,10."]
    and l_ty2[1:] == ["Platine : hp ≥ 1,36·p3 → rupture de bloc, k = 1,0 ; tying : force de chaînage seule, "
                      "fu/γMu (MSB P5 §2.3).",
                      "Cordons hp, pleine résistance 0,55·tw = 4,7 mm ; γM0 1,00 · γM2 1,25 · γM2,net 1,20 · "
                      "γM3 1,25 · γMu 1,10."],
    str(l_elu + l_ty + l_ty2))
t2 = doc[1].get_text(); s2 = t2.replace(" ", "").replace("\n", "")
chk("page 2 : PLAN DE PRINCIPE, trois vues titrées, échelle affichée",
    "PLANDEPRINCIPE" in s2 and all(x in s2 for x in ("ÉLÉVATION", "VUEENPLAN", "VUEDEDROITE"))
    and "Échelle1:" in s2)
chk("page 2 : cartouche PLATINE + visserie + Ø des trous + p3",
    "190×150×10 S355" in t2 and "par boulon : 1 rondelle + 1 écrou" in t2 and "trous Ø22" in t2
    and "3 × 2 (p3 90)" in t2)
chk("le rédacteur saisi figure au cartouche de la note (p. 1 et p. 2) — amendement 3, un champ saisi et jamais lu",
    "J. Dupont" in t1 and "J. Dupont" in t2)
with open("tests/ref_platine_about.json", encoding="utf-8") as fh:
    REF = json.load(fh)
nb = 0
for nom, cas in REF.items():
    p = note.generer_note(moteur.compute(cas["inp"]), {})
    dc = pymupdf.open(stream=p, filetype="pdf")
    if dc.page_count != 2 or note.derniers_avertissements or note.derniere_echelle is None:
        nb += 1
        print("      ", nom, dc.page_count, note.derniere_echelle, note.derniers_avertissements)
chk(f"les {len(REF)} cas figés : 2 pages, échelle trouvée, sans débordement", nb == 0, str(nb))

# ================================================================
print("\n=== 7. Benchmark et exports dans l'application ===")
at.session_state["pab_ui_onglet"] = "Benchmark"; run(at)
chk("onglet Benchmark : exemple publié + calculs manuels, statut VALIDÉ",
    all(x in md(at) for x in ("BM1", "VAL-A", "VAL-B", "Benchmark du module : VALIDÉ")))
at.button(key="pab_btn_bm_0").click(); run(at)
chk("« Charger ces données » : l'exemple publié dans l'outil (VEd = 550, tying 275, hp = 430)",
    float(at.session_state["pab_V_Ed"]) == 550.0 and float(at.session_state["pab_hp_u"]) == 430.0
    and at.session_state["pab_nat_N"] == "Tying (accidentel)" and "ASSEMBLAGE VÉRIFIÉ" in md(at))
at.button(key="pab_btn_reset").click(); run(at)
at.button(key="pab_btn_pdf").click(); run(at)
chk("« Générer PDF » depuis l'application : 2 pages",
    bool(at.session_state.get("pab_pdf_bytes"))
    and pymupdf.open(stream=at.session_state["pab_pdf_bytes"], filetype="pdf").page_count == 2)
at.session_state["pab_ui_onglet"] = "Note"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Note : export texte avec statut, groupe et références",
    "PLATINE D'ABOUT" in code and "VÉRIFIÉ" in code and "Tableau 3.3 : conforme" in code
    and "MSB P5 §2.2.4" in code)
chk("le rédacteur saisi figure dans l'export texte (amendement 3 : un champ saisi et jamais lu)",
    "Rédacteur : J. Dupont" in texte.construire_texte(moteur.compute(dict(id_red="J. Dupont")))
    and "Rédacteur" not in texte.construire_texte(moteur.compute(dict())))
at.session_state["pab_ui_onglet"] = "Méthode"; run(at)
chk("onglet Méthode : ce que l'outil vérifie, modèle, tying, ANB belge, bibliothèque, limites",
    all(x in md(at) for x in ("Ce que l'outil vérifie", "Modèle mécanique", "Tying", "Annexe nationale belge",
                              "Données de bibliothèque", "Limites du modèle")))

# ================================================================
print("\n=== 8. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict()
          if k.startswith(("b1_", "dal", "pre", "asm_", "fpl_")) and k != "asm_courant"]
chk("aucune clé asm_ / fpl_ / béton créée par le module", not fuites, str(fuites[:5]))
at.button(key="pab_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

# ================================================================
print("\n=== 9. Charte (les contrôles du kit sur la platine d'about) ===")
D = ecran_saisie.D
u0 = entrees.defaults()
R0 = moteur.compute(dict(u0))
cles_carte = carte.cles_carte(D)
cles_fenetres = {c["key"] for grp in ecran_resultats._panneaux(R0, u0).values() for c in grp["champs"]}
attendues = set(entrees.CHAMPS) | set(ecran_saisie.GEOMETRIE_DESSIN) | {"ui_niveau", "ui_composant"}
atteignables = cles_carte | set(ecran_saisie.GEOMETRIE_DESSIN) | cles_fenetres | set(ecran_resultats.CLES_TETE_DESSIN)
manquantes = sorted(attendues - atteignables)
chk("rien ne se perd : les 66 clés de CHAMPS ∪ GEOMETRIE_DESSIN ∪ {ui_niveau, ui_composant} ont une destination",
    len(attendues) == 66 and not manquantes, f"{len(attendues)} clés, manquantes {manquantes}")
chk("la visserie reste atteignable (cartouche) ; ONGLETS inchangés",
    "visserie" in cles_carte and ecran_resultats.ONGLETS == ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode"))
chk("GEOMETRIE_DESSIN ∩ carte = ∅", not (cles_carte & set(ecran_saisie.GEOMETRIE_DESSIN)))
sans_aide = [k for k in entrees.CHAMPS if k not in D.AIDES and carte.aide(D, k) is not None]
longues = [k for k, v in D.AIDES.items() if len(v) > 160]
chk("aide(k) is None hors AIDES ; toute aide ≤ 160", not sans_aide and not longues, str(sans_aide + longues))
chk("aucun « ? » sur Nuance, Profilé, Boulon, Classe, dimensions, efforts, identification",
    not ({"nu_P", "nu_S", "nu_pl", "prof_P", "prof_S", "boulon_u", "classe", "hP_u", "hS_u",
          "V_Ed", "N_Ed", "n1_u", "id_rep", "id_red"} & set(D.AIDES)))
trop = [(k, v) for k, v in D.LIBELLE.items() if len(v) > 22]
r4 = [(k, D.LIBELLE[k]) for genre, cles in carte.rangees(D) if genre == "objet" and len(cles) == 4
      for k in cles if len(D.LIBELLE[k]) > 16]
r3 = [(k, D.LIBELLE[k]) for genre, cles in carte.rangees(D) if (genre == "suite" or len(cles) == 3)
      for k in cles if len(D.LIBELLE[k]) > 20]
chk("LIBELLE ≤ 22 partout, ≤ 16 dans une rangée de 4, ≤ 20 dans une rangée de 3 ou une continuation",
    not trop and not r4 and not r3, str(trop + r4 + r3))


def _mot(v):
    t = re.match(r"[A-Za-zÀ-ÿ]+", v)
    return bool(t) and len(t.group(0)) >= 3


chk("capitale initiale des mots ; unités tirées de UNITE (VEd (kN), hp (mm), Rayon grugeage (mm))",
    not [v for v in D.LIBELLE.values() if _mot(v) and v[:1] != v[:1].upper()]
    and D.LIBELLE["V_Ed"] == "VEd (kN)" and D.LIBELLE["hp_u"] == "hp (mm)" and D.LIBELLE["r_n"] == "Rayon grugeage (mm)"
    and D.LIBELLE["n1_u"] == "Rangées n1 (2 files)")
longues = [(k, o) for k, f in entrees.CHAMPS.items() for o in (f.get("o") or []) if len(carte.affiche(D, k, o)) > 28]
chk("options affichées ≤ 28 ; AFFICHAGE ⊂ options brutes ; Nature de NEd → ELU / Tying, valeurs brutes inchangées",
    not longues and all(set(v) <= set(entrees.CHAMPS[k]["o"]) for k, v in D.AFFICHAGE.items())
    and carte.affiche(D, "nat_N", entrees.NAT_ELU) == "ELU" and carte.affiche(D, "nat_N", entrees.NAT_TYING) == "Tying"
    and entrees.NAT_TYING == "Tying (accidentel)" and entrees.MODE_VERIF == "VÉRIFICATION", str(longues))
chk("aucune ligne d'objet > 4 widgets ; ↳ PLATINE et ↳ groupe à une case",
    not [c for g_, c in carte.rangees(D) if g_ == "objet" and len(c) > 4]
    and any(g_ == "suite" and c == ["nu_pl"] for g_, c in carte.rangees(D))
    and any(g_ == "suite" and c == ["n1_u"] for g_, c in carte.rangees(D)))
vis = carte.aides_visibles(D, u0)
chk("« ? » visibles dans la carte au défaut ≤ 4 (mode, trou, catégorie, EFFORTS ELU)",
    len(vis) <= 4 and set(vis) == {"mode_calc", "trou", "cat", "EFFORTS ELU"}, str(vis))
chk("grugeage nommé c / dc,sup / dc,inf dans GROUPES (vue) et LIBELLE",
    "dc,sup" in entrees.CHAMPS["d_nt"]["l"] and "dc,inf" in entrees.CHAMPS["d_nb"]["l"]
    and D.LIBELLE["l_n"] == "c (mm)" and D.LIBELLE["d_nt"] == "dc,sup (mm)")
chk("GROUPES sans note morte", not any(g_.get("note") for g_ in entrees.GROUPES))
pertes, trop = [], []
for nom, cas in REF.items():
    Rc = moteur.compute(cas["inp"])
    segs = note.segments_hypotheses(Rc)
    ls = note.lignes_hypotheses(Rc)
    txt = " ".join(ls)
    if len(ls) > 3 or any(len(l) > 110 for l in ls):
        trop.append((nom, [len(l) for l in ls]))
    for s in segs:
        formes = [s] if isinstance(s, str) else list(s)
        if not any(f[1:] in txt for f in formes):
            pertes.append((nom, formes[-1][:30]))
chk(f"HYPOTHÈSES des {len(REF)} cas figés : ≤ 3 lignes ≤ 110, aucun segment perdu", not trop and not pertes,
    str(trop + pertes))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

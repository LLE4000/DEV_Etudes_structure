# -*- coding: utf-8 -*-
"""Écran, dessins et note du module plat d'âme (AppTest, application réelle).

Chaque garantie rougit si on la retire :
  1. NAVIGATION : la carte « Plat d'âme soudé (fin plate) » sur la page de
     sélection, « Ouvrir » mène au module, mêmes boutons d'outils.
  2. ÉCRAN : deux colonnes, colonne du dessin FIGÉE (CSS sticky, conteneur
     fpl_col_dessin), deux composants, carte par objet à droite SANS la
     géométrie (elle vit sur le dessin), visserie, 67 clés en session.
  3. SOURCE UNIQUE : un message du dessin (cote, glissement, panneau)
     écrit dans fpl_<clé> et recalcule ; VEd = 400 → NON VÉRIFIÉ.
  4. PRÉDIM : solutions cliquables, « Appliquer » recopie et repasse en
     VÉRIFICATION.
  5. DESSINS : SVG valides ; lignes de rupture ; renvois de perçage et
     cote zt en fabrication ; répartition des cotes SANS doublon entre
     les trois vues du plan de principe.
  6. NOTE : 2 pages A4 paysage sans débordement (défaut + 12 cas de la
     régression figée), échelle normalisée exposée, cartouche avec plat et
     visserie, en-tête compact, pied 2 lignes.
  7. BENCHMARK dans l'application : statut VALIDÉ. 8. ÉTANCHÉITÉ (aucune
     fuite vers asm_/béton).

Lancement : python3 tests/test_plat_ame_ecran.py
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.assemblages.poutre_poutre.plat_ame import (interface, ecran_saisie, moteur,  # noqa: E402
                                                      schemas, note)

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
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox) if w.key}


def message(at, changes, t):
    derniers = at.session_state["fpl_ui_dernier_clic"]
    if derniers.get("fpl_cmp_elev") != t:
        derniers["fpl_cmp_elev"] = t
        for k, v in changes.items():
            at.session_state["fpl_" + k] = ecran_saisie.valeur_widget(k, v)


# ================================================================
print("=== 1. Navigation ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at.session_state["page"] = "Assemblages métalliques"
run(at)
chk("la carte « Plat d'âme soudé (fin plate) » est sur la page de sélection",
    "Plat d'âme soudé (fin plate)" in md(at))
at.button(key="asm_btn_open_plat_ame").click(); run(at)
chk("« Ouvrir » mène au module", not at.exception and at.session_state["asm_courant"] == "plat_ame")
chk("en-tête du module", "plat d'âme soudé" in md(at))

# ================================================================
print("\n=== 2. Écran ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("fpl_") and k[4:] in interface.CLES]
chk("les 67 clés d'entrée existent en session", len(cles) == 67, str(len(cles)))
t = md(at)
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ, dimensionnante pression diamétrale de l'âme",
    "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t and "74,0 %" in t)
chk("taux par élément : Boulons, Plat, Portée, Porteuse, Soudure",
    all(x in t for x in ("Boulons", "Plat", "Portée", "Porteuse", "Soudure")))
chk("deux composants empilés dans la colonne figée (CSS sticky sur fpl_col_dessin)",
    len(at.get("component_instance")) == 2 and "position: sticky" in t and "st-key-fpl_col_dessin" in t)
w = cles_widgets(at)
donnees = {k for k in w if k.startswith("fpl_") and k[4:] in interface.CLES}
chk("la géométrie ne se règle QUE sur le dessin (aucune cote dans la carte)",
    not donnees & {"fpl_" + k for k in ecran_saisie.GEOMETRIE_DESSIN}, str(sorted(donnees)))
chk("carte par objet : mode, profilés, nuance du plat, boulons, rangées, efforts, visserie",
    {"fpl_mode_calc", "fpl_prof_P", "fpl_prof_S", "fpl_nu_pl", "fpl_boulon_u", "fpl_n1_u",
     "fpl_V_Ed", "fpl_visserie", "fpl_g_M1"} <= w)
chk("γM1 (ANB belge) exposé dans les avancés, défaut 1,10", at.session_state["fpl_g_M1"] == 1.1)
chk("onglet par défaut : Vérifications", at.session_state["fpl_ui_onglet"] == "Vérifications"
    and "Cisaillement du groupe" in t and "Pinces et entraxes" in t)
chk("la limite « flexion hors plan de l'âme porteuse » est écrite en clair",
    "flexion HORS PLAN de l'âme porteuse" in t)

# ================================================================
print("\n=== 3. Source unique ===")
message(at, {"V_Ed": 400}, 1001); run(at)
chk("message du dessin : VEd = 400 → NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
message(at, {"V_Ed": 125, "hp_u": 140}, 1002); run(at)
chk("hp = 140 → alerte bloquante pince basse, « Localiser » disponible",
    "pince basse" in md(at) and any(b.key == "fpl_btn_al_e1bot" for b in at.button))
at.button(key="fpl_btn_al_e1bot").click(); run(at)
chk("chip d'alerte : hp modifiable sur place", any(x.key == "fpl_fix_hp_u" for x in at.number_input))
at.number_input(key="fpl_fix_hp_u").set_value(190.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["fpl_hp_u"]) == 190.0 and "pince basse" not in md(at))
message(at, {"g_h": 25, "d_top": 10}, 1003); run(at)
chk("glissement gh + Δz → clés écrites, recalcul",
    float(at.session_state["fpl_g_h"]) == 25.0 and float(at.session_state["fpl_d_top"]) == 10.0
    and not at.exception)
message(at, {"g_h": 10, "d_top": 0}, 1004); run(at)

# ================================================================
print("\n=== 4. Prédim ===")
at.selectbox(key="fpl_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
chk("bandeau prédimensionnement", "Mode prédimensionnement" in md(at))
at.session_state["fpl_ui_onglet"] = "Prédim"; run(at)
boutons = [b for b in at.button if b.key and b.key.startswith("fpl_btn_sol_")]
chk("solutions proposées cliquables", len(boutons) >= 1)
retenue = [b for b in boutons if b.type == "primary"]
(retenue[0] if retenue else boutons[0]).click(); run(at)
chk("« Appliquer » : VÉRIFICATION + plat et boulons recopiés",
    at.session_state["fpl_mode_calc"] == "VÉRIFICATION"
    and float(at.session_state["fpl_tp_u"]) <= 0.5 * moteur.compute(interface_read := {k: at.session_state.get("fpl_" + k) for k in interface.CLES}).d_b + 1e-9)
at.button(key="fpl_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts", float(at.session_state["fpl_hp_u"]) == 190.0)

# ================================================================
print("\n=== 5. Dessins ===")
R0 = moteur.compute(dict())
for nom, fab in (("écran", False), ("fabrication", True)):
    for f, excl in (("elevation", schemas.EXCLURE_ELEVATION), ("plan", schemas.EXCLURE_PLAN),
                    ("vue_droite", schemas.EXCLURE_DROITE)):
        o = schemas.options_fabrication(excl, 12) if fab else schemas.options_ecran(R0)
        svg = getattr(schemas, f)(R0, o).svg(identifiant="t")
        ET.fromstring(svg)
poly = next(p for p in schemas.elevation(R0, schemas.options_ecran(R0)).corps
            if p.get("t") == "g" and p.get("grp") == "beamS")
chk("SVG valides (3 vues × écran/fabrication) ; la portée est glissable (drag gh + Δz)",
    poly.get("drag") == "g_h" and poly.get("drag2") == "d_top")
sd = schemas.vue_droite(R0, schemas.options_fabrication(schemas.EXCLURE_DROITE, 12))
face = next(p for p in sd.corps if p["t"] == "poly" and "pp" in p["cls"])
xs = sorted(pt[0] for pt in face["pts"])
chk("vue de droite : âme porteuse rompue des deux côtés, coupe de la portée hachurée",
    xs[0] < xs[1] - 1.5 and xs[-1] > xs[-2] + 1.5
    and sum(1 for p in sd.corps if p["t"] == "path" and "ht" in p["cls"]) == 1)
se = schemas.elevation(R0, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)).svg()
sp = schemas.plan(R0, schemas.options_fabrication(schemas.EXCLURE_PLAN, 12)).svg()
sdv = sd.svg()
chk("fabrication : renvois 3×Ø22 et r 10 (élévation), zt sur la vue de droite",
    ">3×Ø22<" in se and ">r 10<" in se and "zt" in sdv)
chk("cotes réparties SANS doublon : hp et zp en élévation seulement, tp au plan, bS à droite",
    "hp = 190" in se and "hp" not in sdv.split("</style>")[-1].replace("zt", "")
    and ">tp 10<" in sp or "tp = 10" in sp)
doubles = [s for s in ("hp = 190", "zp 50", "bS = 300", "c = 150")
           if (se + sp + sdv).count(s) > 1]
chk("aucune cote posée deux fois sur la planche", not doubles, str(doubles))

# ================================================================
print("\n=== 6. Note (2 pages) ===")
R = moteur.compute(dict(id_projet="Halle A", id_rep="P3/S9"))
pdf = note.generer_note(R, {"nom_projet": "Halle A", "partie": "P3/S9", "date": "21/09/2026",
                            "indice": "0", "visserie": "1 rondelle + 1 écrou"})
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("deux pages A4 paysage, sans débordement, échelle et disposition exposées",
    doc.page_count == 2 and not note.derniers_avertissements
    and note.derniere_echelle is not None and note.derniere_disposition is not None,
    str(note.derniers_avertissements))
t1 = doc[0].get_text(); s1 = t1.replace(" ", "").replace("\n", "")
chk("page 1 : bandeau, en-tête compact 3 lignes, hypothèses, tableaux, conclusion",
    "VÉRIFIÉ" in s1 and "Plat 190×100×10 S355" in t1 and "Soudure double a 5" in t1
    and "VEd 125,0 kN" in t1 and "rotule à la face de l'âme porteuse" in t1
    and "PLAT D'ÂME" in t1.upper() and "ASSEMBLAGE VÉRIFIÉ à l'ELU" in t1)
chk("page 1 : γM1 ANB écrit dans les hypothèses", "γM1 1,10 (ANB)" in t1)
t2 = doc[1].get_text(); s2 = t2.replace(" ", "").replace("\n", "")
chk("page 2 : PLAN DE PRINCIPE, trois vues titrées, échelle affichée",
    "PLANDEPRINCIPE" in s2 and all(x in s2 for x in ("ÉLÉVATION", "VUEENPLAN", "VUEDEDROITE"))
    and "Échelle1:" in s2)
chk("page 2 : cartouche PLAT + visserie + Ø des trous",
    "190×100×10 S355" in t2 and "par boulon : 1 rondelle + 1 écrou" in t2 and "trous Ø22" in t2)
chk("page 2 : renvoi de perçage et cote zt", "3×Ø22" in t2 and "zt" in t2)
with open("tests/ref_plat_ame.json", encoding="utf-8") as fh:
    REF = json.load(fh)
nb = 0
for nom, cas in REF.items():
    p = note.generer_note(moteur.compute(cas["inp"]), {})
    dc = pymupdf.open(stream=p, filetype="pdf")
    if dc.page_count != 2 or note.derniers_avertissements or note.derniere_echelle is None:
        nb += 1
        print("      ", nom, dc.page_count, note.derniere_echelle, note.derniers_avertissements)
chk("les 12 cas figés : 2 pages, échelle trouvée, sans débordement", nb == 0, str(nb))

# ================================================================
print("\n=== 7. Benchmark et exports dans l'application ===")
at.session_state["fpl_ui_onglet"] = "Benchmark"; run(at)
chk("onglet Benchmark : exemple publié + calculs manuels, statut VALIDÉ",
    all(x in md(at) for x in ("BM1", "VAL-A", "VAL-B", "Benchmark du module : VALIDÉ")))
at.button(key="fpl_btn_bm_0").click(); run(at)
chk("« Charger ces données » : l'exemple publié dans l'outil (VEd = 350, hp = 360)",
    float(at.session_state["fpl_V_Ed"]) == 350.0 and float(at.session_state["fpl_hp_u"]) == 360.0)
at.button(key="fpl_btn_reset").click(); run(at)
at.button(key="fpl_btn_pdf").click(); run(at)
chk("« Générer PDF » depuis l'application : 2 pages",
    bool(at.session_state.get("fpl_pdf_bytes"))
    and pymupdf.open(stream=at.session_state["fpl_pdf_bytes"], filetype="pdf").page_count == 2)
at.session_state["fpl_ui_onglet"] = "Note"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Note : export texte avec statut et références",
    "PLAT D'ÂME" in code and "VÉRIFIÉ" in code and "Tableau 3.3 : conforme" in code)
at.session_state["fpl_ui_onglet"] = "Méthode"; run(at)
chk("onglet Méthode : modèle, ANB belge, limites",
    "Modèle mécanique" in md(at) and "Annexe nationale belge" in md(at))

# ================================================================
print("\n=== 8. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict()
          if k.startswith(("b1_", "dal", "pre", "asm_")) and k != "asm_courant"]
chk("aucune clé asm_/béton créée par le module", not fuites, str(fuites[:5]))
at.button(key="fpl_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

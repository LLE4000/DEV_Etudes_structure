# -*- coding: utf-8 -*-
"""Écran, dessins et note du module platine d'about (AppTest, application
réelle).

Chaque garantie rougit si on la retire :
  1. NAVIGATION : la carte « Platine d'about (hauteur partielle) » est
     DISPONIBLE sur la page de sélection, « Ouvrir » mène au module.
  2. ÉCRAN : deux colonnes, colonne du dessin FIGÉE (CSS sticky, conteneur
     pab_col_dessin), deux composants, carte par objet à droite SANS la
     géométrie (elle vit sur le dessin), visserie, nature de NEd.
  3. SOURCE UNIQUE : un message du dessin (cote, glissement vertical,
     panneau) écrit dans pab_<clé> et recalcule ; VEd = 450 → NON VÉRIFIÉ.
  4. PRÉDIM : solutions cliquables, « Appliquer » recopie et repasse en
     VÉRIFICATION.
  5. DESSINS : SVG valides (3 vues × écran/fabrication) ; la portée ne
     glisse QUE verticalement (pas de jeu gh) ; renvois de perçage, zt ;
     répartition des cotes SANS doublon sur le plan de principe ;
     composant : glissement vertical seul pris en charge.
  6. NOTE : 2 pages A4 paysage sans débordement (défaut + cas figés),
     échelle normalisée, cartouche PLATINE + visserie, γM0 ANB.
  7. BENCHMARK dans l'application : VALIDÉ ; exemple chargé ; exports.
  8. ÉTANCHÉITÉ : aucune clé asm_ / fpl_ / béton créée.

Lancement : python3 tests/test_platine_about_ecran.py
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

from acier.assemblages.poutre_poutre.platine_about import (interface, ecran_saisie, moteur,  # noqa: E402
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
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ, 51,7 % (stabilité du grugeage)",
    "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t and "51,7 %" in t)
chk("taux par élément : Boulons, Platine, Portée, Porteuse, Soudure",
    all(x in t for x in ("Boulons", "Platine", "Portée", "Porteuse", "Soudure")))
chk("deux composants empilés dans la colonne figée (CSS sticky sur pab_col_dessin)",
    len(at.get("component_instance")) == 2 and "position: sticky" in t and "st-key-pab_col_dessin" in t)
w = cles_widgets(at)
donnees = {k for k in w if k.startswith("pab_") and k[4:] in interface.CLES}
chk("la géométrie ne se règle QUE sur le dessin (aucune cote dans la carte)",
    not donnees & {"pab_" + k for k in ecran_saisie.GEOMETRIE_DESSIN}, str(sorted(donnees)))
chk("carte par objet : mode, profilés, nuance de la platine, boulons, rangées, efforts, nature de NEd, visserie",
    {"pab_mode_calc", "pab_prof_P", "pab_prof_S", "pab_nu_pl", "pab_boulon_u", "pab_n1_u",
     "pab_V_Ed", "pab_N_Ed", "pab_nat_N", "pab_visserie", "pab_g_M0"} <= w)
chk("γMu n'apparaît qu'en tying", "pab_g_Mu" not in w)
chk("onglet par défaut : Vérifications (groupe de boulons, Tableau 3.3)",
    at.session_state["pab_ui_onglet"] == "Vérifications"
    and "Groupe : cisaillement et pression diam." in t and "Pinces et entraxes" in t)
chk("la limite « flexion TRANSVERSALE de l'âme porteuse » est écrite en clair",
    "flexion TRANSVERSALE de l'âme porteuse" in t)

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
at.selectbox(key="pab_nat_N").set_value("Tying (accidentel)"); run(at)
chk("nature Tying : γMu apparaît dans les paramètres avancés", "pab_g_Mu" in cles_widgets(at))
at.selectbox(key="pab_nat_N").set_value("ELU fondamental"); run(at)

# ================================================================
print("\n=== 4. Prédim ===")
at.selectbox(key="pab_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
chk("bandeau prédimensionnement", "Mode prédimensionnement" in md(at))
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

# ================================================================
print("\n=== 6. Note (2 pages) ===")
R = moteur.compute(dict(id_projet="Halle A", id_rep="P3/S9"))
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
t2 = doc[1].get_text(); s2 = t2.replace(" ", "").replace("\n", "")
chk("page 2 : PLAN DE PRINCIPE, trois vues titrées, échelle affichée",
    "PLANDEPRINCIPE" in s2 and all(x in s2 for x in ("ÉLÉVATION", "VUEENPLAN", "VUEDEDROITE"))
    and "Échelle1:" in s2)
chk("page 2 : cartouche PLATINE + visserie + Ø des trous + p3",
    "190×150×10 S355" in t2 and "par boulon : 1 rondelle + 1 écrou" in t2 and "trous Ø22" in t2
    and "3 × 2 (p3 90)" in t2)
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
at.session_state["pab_ui_onglet"] = "Méthode"; run(at)
chk("onglet Méthode : modèle, tying, ANB belge, limites",
    all(x in md(at) for x in ("Modèle mécanique", "Tying", "Annexe nationale belge", "Limites du modèle")))

# ================================================================
print("\n=== 8. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict()
          if k.startswith(("b1_", "dal", "pre", "asm_", "fpl_")) and k != "asm_courant"]
chk("aucune clé asm_ / fpl_ / béton créée par le module", not fuites, str(fuites[:5]))
at.button(key="pab_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

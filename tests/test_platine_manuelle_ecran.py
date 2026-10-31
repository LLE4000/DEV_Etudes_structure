# -*- coding: utf-8 -*-
"""Écran du CALCUL MANUEL de la platine (AppTest, application réelle) — lot D.

Chaque garantie rougit si on la retire :
  1. OUVERTURE : « Créer » ouvre l'écran en calcul manuel ; bandeau
     « PLATINE VÉRIFIÉE (calcul manuel) » avec l'épaisseur requise ; le
     calcul à la main en tête des vérifications (boulon par boulon, section
     au nu, bande comprimée, trous) ; deux dessins ; la carte (mode, boulons
     en table, éléments, raidisseurs, efforts).
  2. SOURCE UNIQUE : la table écrit dans ppb_tiges et recalcule ; ➕ / 🗑️
     ajoutent et retirent ; un message du dessin (clé dérivée « t0_u »)
     déplace le boulon ; un élément se tourne et se déplace ; un raidisseur
     raccourcit le bras du boulon.
  3. BASCULE DE MODE : vers la vérification complète quand la disposition est
     régulière (grille et profilé repris), refusée sinon (raison dite, mode
     conservé) ; retour au manuel avec la grille.
  4. ONGLETS : Efforts des tiges, Note (export texte), Benchmark (calculs
     manuels VALIDÉS, « charger »), Méthode.
  5. NOTE PDF : 2 pages sans débordement, plan de principe à l'échelle.
  6. PROJET : description de la liste, rapport du projet avec un calcul manuel.
  7. ÉTANCHÉITÉ : aucune clé asm_ / fpl_ / pab_ créée.

Lancement : python3 tests/test_platine_manuelle_ecran.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.assemblages.pied_poteau.platine_beton import interface, note  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.calcul import compute  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.entrees import (MODE_COMPLET, MODE_MANUEL, TYPE_RHS,  # noqa: E402
                                                                  CRIT_ELAS)
from acier import projet as modele, rapport_projet  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


def run(at):
    at.run()
    if at.exception:
        chk("aucune exception", False, str(at.exception[0].message)[:200])
    return at


def lu(at):
    return {k: at.session_state.get("ppb_" + k) for k in interface.CLES}


def message(at, changes, t):
    """Un message du composant (glissement ou fenêtre d'objet), comme le
    navigateur l'enverrait : rejoué au prochain run."""
    derniers = at.session_state["ppb_ui_dernier_clic"]
    if derniers.get("ppb_cmp_plan") != t:
        derniers["ppb_cmp_plan"] = t
        from acier.assemblages.pied_poteau.platine_beton.manuel import ecran as EM
        for k, v in changes.items():
            if k in interface.CLES:
                at.session_state["ppb_" + k] = v
            else:
                EM._appliquer_derivee(k, v, lu(at), at.session_state)


# ================================================================
print("=== 1. Ouverture ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
at.session_state["page"] = "Assemblages métalliques"
run(at)
at.button(key="asm_btn_open_platine_beton").click(); run(at)
t = md(at)
chk("« Créer » ouvre l'écran en CALCUL MANUEL (mode par défaut)", at.session_state["ppb_mode"] == MODE_MANUEL
    and at.session_state["asm_courant"] == "platine_beton")
chk("bandeau : PLATINE VÉRIFIÉE (calcul manuel), épaisseur requise 15,6 mm ≤ tp = 25 mm",
    "PLATINE VÉRIFIÉE (calcul manuel)" in t and "épaisseur requise 15,6 mm ≤ tp = 25 mm" in t)
chk("le calcul à la main en tête : boulon par boulon, section au nu, bande comprimée, trous",
    all(x in t for x in ("Épaisseur de la platine — le calcul à la main", "boulon par boulon", "section au nu de chaque paroi",
                         "la bande la plus sollicitée", "pression diamétrale", "mEd = ∫p·t dt")))
chk("deux dessins (coupe A-A, vue en plan) dans la colonne figée", len(at.get("component_instance")) == 2
    and "st-key-ppb_col_dessin" in t)
w = {x.key for x in list(at.number_input) + list(at.selectbox) if x.key}
chk("carte : mode, critère, appui, xN/yN, boulons (table x, y), élément (type, profilé, x, y), efforts",
    {"ppb_mode", "ppb_crit", "ppb_appui", "ppb_xN", "ppb_yN", "ppb_t0_x", "ppb_t3_y", "ppb_e0_type", "ppb_e0_prof",
     "ppb_e0_x", "ppb_N_Ed", "ppb_Mx_Ed", "ppb_Vy_Ed", "ppb_ancr", "ppb_classe"} <= w, str(sorted(w)[:40]))
chk("la vérification complète est annoncée disponible (grille régulière)",
    "disponible pour cette disposition" in t)
R0 = compute(lu(at))
chk("le moteur appelé par l'écran est le calcul manuel (R.mode) et retrouve l'épaisseur du bandeau",
    R0.mode == "manuel" and abs(R0.plaque["t_req"] - 15.56) < 0.05)

# ================================================================
print("\n=== 2. Source unique ===")
at.number_input(key="ppb_t0_x").set_value(-100.0); run(at)
chk("la table écrit dans ppb_tiges (boulon 1 : x = −100) et l'écran dit la grille irrégulière",
    at.session_state["ppb_tiges"][0]["x"] == -100.0 and "indisponible" in md(at))
at.button(key="ppb_btn_tadd").click(); run(at)
chk("➕ Boulon : cinq boulons, le nouveau au bord bas", len(at.session_state["ppb_tiges"]) == 5
    and at.session_state["ppb_tiges"][4]["y"] == -190.0)
at.button(key="ppb_btn_tdel_4").click(); run(at)
chk("🗑️ retire le cinquième", len(at.session_state["ppb_tiges"]) == 4)
message(at, {"t0_u": 50}, 2001); run(at)
chk("message du dessin « t0_u = 50 » (distance au bord gauche) → boulon 1 en x = 50 − 250 = −200",
    at.session_state["ppb_tiges"][0]["x"] == -200.0 and "disponible pour cette disposition" in md(at))
at.selectbox(key="ppb_e0_type").set_value(TYPE_RHS); run(at)
# Le RHS par défaut (200 × 200 × 8) tient DANS l'emprise du HEA 300 : le boulon 1 en (−200 ; −200) s'appuie sur
# l'angle (−100 ; −100), d = 100·√2 = 141,4 mm, m = 141,4 − 0,8·8·√2 = 132,4 mm — plus LONG qu'avec le HEA 300
# (angle d'aile (−150 ; −145) : d = 74,3, m = 65,3 mm).
Rq = compute(lu(at))
chk("élément 1 en RHS : les dimensions h, b, t apparaissent ; le boulon 1 s'appuie sur l'angle du RHS, m = 132,4 mm",
    {"ppb_e0_h", "ppb_e0_b", "ppb_e0_t"} <= {x.key for x in at.number_input if x.key}
    and "angle" in Rq.tiges[0]["appui"]["face"] and abs(Rq.tiges[0]["m"] - (141.42 - 0.8 * 8 * 2 ** 0.5)) < 0.1
    and Rq.tiges[0]["m"] > R0.tiges[0]["m"], f"m = {Rq.tiges[0]['m']:.1f}")
at.selectbox(key="ppb_e0_type").set_value("Profilé en I ou H"); run(at)
at.number_input(key="ppb_e0_x").set_value(40.0); run(at)
Rx = compute(lu(at))
chk("élément déplacé de 40 mm : résultante déplacée, grille irrégulière pour la vérification complète",
    abs(Rx.elems[0]["x"] - 40.0) < 1e-9 and "indisponible" in md(at) and "profilé doit être à l'axe" in md(at))
at.number_input(key="ppb_e0_x").set_value(0.0); run(at)
at.button(key="ppb_btn_radd").click(); run(at)
at.number_input(key="ppb_r0_x").set_value(-150.0); run(at)
at.selectbox(key="ppb_r0_dir").set_value("−y"); run(at)
at.number_input(key="ppb_r0_y").set_value(-145.0); run(at)
at.number_input(key="ppb_r0_L").set_value(60.0); run(at)
Rr = compute(lu(at))
chk("un raidisseur −y sous l'angle de l'aile (−150 ; −145), L = 60 : le boulon 1 s'appuie dessus, bras plus court",
    len(at.session_state["ppb_raid"]) == 1 and "raidisseur" in Rr.tiges[0]["appui"]["face"] and Rr.tiges[0]["m"] < R0.tiges[0]["m"],
    f"{Rr.tiges[0]['appui']['face']} m = {Rr.tiges[0]['m']:.1f}")
at.button(key="ppb_btn_rdel_0").click(); run(at)
chk("🗑️ retire le raidisseur", len(at.session_state["ppb_raid"]) == 0)
at.selectbox(key="ppb_crit").set_value(CRIT_ELAS); run(at)
chk("critère élastique : le bandeau et les tableaux passent en σ de Von Mises", "Von Mises" in md(at)
    and compute(lu(at)).elastique)
at.selectbox(key="ppb_crit").set_value(interface.defaults()["crit"]); run(at)

# ================================================================
print("\n=== 3. Bascule de mode ===")
at.selectbox(key="ppb_mode").set_value(MODE_COMPLET); run(at)
chk("vers la vérification complète : grille 2 × 2 de 400 et HEA 300 repris, bandeau PIED DE POTEAU",
    at.session_state["ppb_mode"] == MODE_COMPLET and float(at.session_state["ppb_w_x"]) == 400.0
    and at.session_state["ppb_prof_c"] == "HEA 300" and "PIED DE POTEAU VÉRIFIÉ" in md(at))
chk("le bloc MODE est aussi dans l'écran complet", any(x.key == "ppb_mode" for x in at.selectbox))
# Dans l'écran complet, wy est une COTE du dessin (jamais un champ de la carte) : le message du composant,
# comme dans tests/test_pied_poteau_ecran.py.
message(at, {"w_y": 360}, 3001); run(at)
chk("écran complet : la cote wy = 360 du dessin écrit dans la source unique", float(at.session_state["ppb_w_y"]) == 360.0)
at.selectbox(key="ppb_mode").set_value(MODE_MANUEL); run(at)
chk("retour au manuel : les quatre boulons de la grille 400 × 360",
    at.session_state["ppb_mode"] == MODE_MANUEL and sorted(t["y"] for t in at.session_state["ppb_tiges"]) == [-180.0, -180.0, 180.0, 180.0])
at.number_input(key="ppb_t1_x").set_value(150.0); run(at)
at.selectbox(key="ppb_mode").set_value(MODE_COMPLET); run(at)
chk("disposition irrégulière : la bascule est REFUSÉE, le mode reste manuel, la raison est dite",
    at.session_state["ppb_mode"] == MODE_MANUEL and "grille" in str(at.session_state.get("_ppb_toast", "")) + md(at))
at.number_input(key="ppb_t1_x").set_value(200.0); run(at)

# ================================================================
print("\n=== 4. Onglets ===")
at.session_state["ppb_ui_onglet"] = "Efforts des tiges"; run(at)
t = md(at)
chk("Efforts des tiges : table des boulons (T, appui, m), répartition (C, pmax, raideurs), comparaison aux logiciels d'ancrage",
    all(x in t for x in ("Boulons : position, effort, appui", "Résultante comprimée C", "Spit ou Hilti")))
at.session_state["ppb_ui_onglet"] = "Note"; run(at)
chk("Note : export texte du calcul à la main (bras, moment, section, bande, épaisseur requise)",
    any(b.key == "ppb_btn_txt" for b in at.get("download_button"))
    and all(x in md(at) + "\n".join(str(c.value) for c in at.code) for x in ("CALCUL MANUEL", "ÉPAISSEUR REQUISE", "bras m =")))
at.session_state["ppb_ui_onglet"] = "Benchmark"; run(at)
t = md(at)
chk("Benchmark : calculs manuels VALIDÉS (VAL-M1 à VAL-M4) rejoués", "VALIDÉ (calculs manuels)" in t
    and all(f"VAL-M{i}" in t for i in (1, 2, 3, 4)))
at.button(key="ppb_btn_bm_m1").click(); run(at)
chk("« Charger ces données » : VAL-M2 (boulons devant l'aile, suspente) dans l'outil, VÉRIFIÉE",
    float(at.session_state["ppb_N_Ed"]) == -100.0 and at.session_state["ppb_tiges"][0]["y"] == -215
    and "PLATINE VÉRIFIÉE (calcul manuel)" in md(at))
at.session_state["ppb_ui_onglet"] = "Méthode"; run(at)
chk("Méthode : le modèle dans l'ordre d'une note à la main", "Le calcul manuel de la platine" in md(at))
at.session_state["ppb_ui_onglet"] = "Vérifications"; run(at)

# ================================================================
print("\n=== 5. Note PDF ===")
at.button(key="ppb_btn_reset").click(); run(at)
R = compute(lu(at))
pdf = note.generer_note(R, {"nom_projet": "Essai", "partie": "A1", "date": "31/10/2026", "indice": "0", "visserie": ""})
doc = pymupdf.open(stream=pdf, filetype="pdf")
p1 = doc[0].get_text()
p2 = doc[1].get_text() if doc.page_count > 1 else ""
chk("deux pages A4 paysage, sans débordement", doc.page_count == 2 and not note.derniers_avertissements,
    str(note.derniers_avertissements)[:200])
# Le peintre PDF du kit pose les glyphes un à un : pymupdf les relit séparés d'espaces — on compare sans
# espaces ni retours, comme tests/test_pied_poteau_ecran.py.
s1 = p1.replace(" ", "").replace("\n", "")
chk("page 1 : titre « calcul manuel de la platine », VÉRIFIÉ ηmax, hypothèses, les quatre critères, le calcul à la main "
    "(boulon 1 : T, appui, m), section au nu, bande au nu",
    all(x in s1 for x in ("calculmanueldelaplatine", "VÉRIFIÉηmax", "HYPOTHÈSES", "ÉPAISSEURREQUISEETSESQUATRECRITÈRES",
                          "LECALCULÀLAMAIN", "Boulon1T=", "sectionaunu", "bandeaunu")), s1[:300])
chk("page 2 : PLAN DE PRINCIPE à l'échelle, vue en plan, coupes, cartouche avec les perçages",
    all(x in p2.replace(" ", "").upper() for x in ("PLANDEPRINCIPE", "COUPEA-A", "VUEENPLAN", "ÉCHELLE"))
    and note.derniere_echelle is not None, p2[:300])
at.button(key="ppb_btn_pdf").click(); run(at)
chk("« Générer PDF » depuis l'application : 2 pages",
    pymupdf.open(stream=at.session_state["ppb_pdf_bytes"], filetype="pdf").page_count == 2)

# ================================================================
print("\n=== 6. Projet ===")
e = at.session_state["prj_projet"]["assemblages"][0]
desc = modele.description(e["values"])
chk("la liste décrit un calcul manuel : élément, platine, boulons", "calcul manuel" in desc and "HEA 300" in desc
    and "4 boulons" in desc, desc)
rp = rapport_projet.generer_rapport(at.session_state["prj_projet"], {"nom_projet": "Essai", "partie": "", "date": "31/10/2026", "indice": "0"})
drp = pymupdf.open(stream=rp, filetype="pdf")
chk("rapport du projet avec un calcul manuel : garde + récapitulatif + note de 2 pages", drp.page_count == 4
    and "Platine de pied de poteau sur béton" in drp[0].get_text())

# ================================================================
print("\n=== 7. Étanchéité ===")
chk("aucune clé asm_ / fpl_ / pab_ créée par le module",
    not [k for k in at.session_state.to_dict() if k.startswith(("asm_", "fpl_", "pab_")) and k != "asm_courant"])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

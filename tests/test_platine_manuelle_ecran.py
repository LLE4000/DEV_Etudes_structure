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
  8. CHARTE ERGONOMIQUE du 31/10/2026 (§4.5, amendements 1 à 4) : en-tête
     Mode · Critère · Appui en segmented_control (défaut ÉLASTIQUE, §12) et
     l'état de la bascule dit dans les deux cas ; hef et dh hors de la carte
     (GEOMETRIE_DESSIN ∩ carte = ∅), atteignables sans boulon (fenêtre
     Platine) et dans le panneau Cotes ; tg nul dans la fenêtre Massif et le
     panneau Cotes ; xN / yN dans les avancés ; « Boulon », « Visserie par
     boulon » ; 7 « ? » ; tables du kit (ppb_th_ / ppb_tr_) ; ↳ É1 seulement
     si besoin ; replis de l'onglet Vérifications ; titre court, barre
     partagée ; rédacteur au cartouche et dans l'export ; HYPOTHÈSES ≤ 3 × 110
     (défaut, variantes, VAL-M1 à VAL-M4) ; conclusion qui dit le critère ;
     rien ne se perd.

Lancement : python3 tests/test_platine_manuelle_ecran.py
"""
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import pymupdf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.assemblages import carte as kit  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton import interface, note, ecran_saisie, ecran_resultats  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton import texte as texte_ppb  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.calcul import compute  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.manuel import ecran as EM, note as NM, benchmark as BM  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.entrees import (MODE_COMPLET, MODE_MANUEL, TYPE_RHS,  # noqa: E402
                                                                  CRIT_ELAS, CRIT_EC3, APPUI_PLEIN, TYPE_CROSSE,
                                                                  CHAMPS, champ_visible, defaults)
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


def cles_widgets(at):
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox)
            + list(at.segmented_control) if w.key}


def expanders(at):
    return [str(x.label) for x in at.expander]


def expander(at, titre):
    return next((x for x in at.expander if str(x.label).startswith(titre)), None)


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
# Défaut ÉLASTIQUE depuis le 31/10/2026 (charte §12) : 19,8 mm au lieu de 15,6, et le bandeau dit le critère
# gouvernant sans répéter « Épaisseur requise de la platine » (charte §9).
chk("bandeau : PLATINE VÉRIFIÉE (calcul manuel) — épaisseur requise 19,8 ≤ tp 25 mm · 79,3 % · tronçon en T du boulon",
    "PLATINE VÉRIFIÉE (calcul manuel)" in t and "— épaisseur requise 19,8 ≤ tp 25 mm · 79,3 % · tronçon en T du boulon" in t
    and "Épaisseur requise de la platine" not in t)
chk("bandeau : à côté de « Platine 79 % », les quatre critères tT · tS · tc · tb",
    all(x in t for x in (">tT <b", ">tS <b", ">tc <b", ">tb <b")) and t.index("Platine <b") < t.index(">tT <b"))
chk("le calcul à la main en tête : boulon par boulon, section au nu, bande comprimée, trous",
    all(x in t for x in ("Épaisseur de la platine — le calcul à la main", "boulon par boulon", "section au nu de chaque paroi",
                         "la bande la plus sollicitée", "pression diamétrale", "mEd = ∫p·t dt")))
chk("deux dessins (coupe A-A, vue en plan) dans la colonne figée", len(at.get("component_instance")) == 2
    and "st-key-ppb_col_dessin" in t)
w = {x.key for x in list(at.number_input) + list(at.selectbox) + list(at.segmented_control) if x.key}
chk("carte : mode, critère, appui (segmented), xN/yN, boulons (table x, y), élément (type, profilé, x, y), efforts",
    {"ppb_mode", "ppb_crit", "ppb_appui", "ppb_xN", "ppb_yN", "ppb_t0_x", "ppb_t3_y", "ppb_e0_type", "ppb_e0_prof",
     "ppb_e0_x", "ppb_N_Ed", "ppb_Mx_Ed", "ppb_Vy_Ed", "ppb_ancr", "ppb_classe"} <= w, str(sorted(w)[:40]))
chk("la vérification complète est annoncée disponible (grille régulière)",
    "disponible pour cette disposition" in t)
R0 = compute(lu(at))
chk("le moteur appelé par l'écran est le calcul manuel (R.mode) et retrouve l'épaisseur du bandeau (élastique, "
    "19,81 mm mesurés le 31/10/2026)",
    R0.mode == "manuel" and R0.elastique and abs(R0.plaque["t_req"] - 19.81) < 0.05)

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
# Le défaut est élastique : on pose « Plastique » (CRIT_EC3), puis on revient au défaut.
at.segmented_control(key="ppb_crit").set_value(CRIT_EC3); run(at)
t = md(at)
chk("critère « Plastique » : le calcul à la main passe en EN 1993-1-8 (tronçon en T Mpl), épaisseur plus faible",
    "EN 1993-1-8 (plastique côté tendu)" in t and "élastique (Von Mises)" not in t and not compute(lu(at)).elastique
    and compute(lu(at)).plaque["t_req"] < R0.plaque["t_req"])
at.segmented_control(key="ppb_crit").set_value(interface.defaults()["crit"]); run(at)
chk("retour au défaut « Élastique » : le bandeau et les tableaux passent en σ de Von Mises",
    "élastique (Von Mises)" in md(at) and compute(lu(at)).elastique and at.session_state["ppb_crit"] == CRIT_ELAS)

# ================================================================
print("\n=== 3. Bascule de mode ===")
at.segmented_control(key="ppb_mode").set_value(MODE_COMPLET); run(at)
chk("vers la vérification complète : grille 2 × 2 de 400 et HEA 300 repris, bandeau PIED DE POTEAU",
    at.session_state["ppb_mode"] == MODE_COMPLET and float(at.session_state["ppb_w_x"]) == 400.0
    and at.session_state["ppb_prof_c"] == "HEA 300" and "PIED DE POTEAU VÉRIFIÉ" in md(at))
chk("le choix du mode est aussi dans l'en-tête de l'écran complet (segmented_control)",
    any(x.key == "ppb_mode" for x in at.segmented_control) and not any(x.key == "ppb_crit" for x in at.segmented_control))
# Dans l'écran complet, wy est une COTE du dessin (jamais un champ de la carte) : le message du composant,
# comme dans tests/test_pied_poteau_ecran.py.
message(at, {"w_y": 360}, 3001); run(at)
chk("écran complet : la cote wy = 360 du dessin écrit dans la source unique", float(at.session_state["ppb_w_y"]) == 360.0)
at.segmented_control(key="ppb_mode").set_value(MODE_MANUEL); run(at)
chk("retour au manuel : les quatre boulons de la grille 400 × 360",
    at.session_state["ppb_mode"] == MODE_MANUEL and sorted(t["y"] for t in at.session_state["ppb_tiges"]) == [-180.0, -180.0, 180.0, 180.0])
at.number_input(key="ppb_t1_x").set_value(150.0); run(at)
at.segmented_control(key="ppb_mode").set_value(MODE_COMPLET); run(at)
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

# ================================================================
print("\n=== 8. Charte ergonomique (§4.5, amendements 1 à 4) ===")
from acier.assemblages.pied_poteau.platine_beton.manuel import schemas as SM  # noqa: E402
D = ecran_saisie.D_MANUEL
at.button(key="ppb_btn_reset").click(); run(at)
t = md(at)
u0 = lu(at)
R0 = compute(u0)
w = cles_widgets(at)
chk("titre court « Pied de poteau sur béton » + caption « Platine sur béton – pied de poteau · version x.y » ; plus de "
    "bouton ❔ ni de cellule Version",
    "## Pied de poteau sur béton" in t and "Platine sur béton – pied de poteau · version " + interface.MODULE_VERSION in t
    and not any(b.key == "ppb_btn_version" for b in at.button) and "Version 1.1</div>" not in t)
chk("barre d'outils partagée (projet_ui.barre_outils) : ◀ Projet · 🏠 · 💾 · 📄 PDF · 🔄, clés inchangées",
    {"ppb_btn_retour", "ppb_btn_home", "ppb_btn_pdf", "ppb_btn_reset"} <= {b.key for b in at.button}
    and at.button(key="ppb_btn_pdf").label == "📄 PDF" and any(b.key == "ppb_btn_save_dl" for b in at.get("download_button")))
seg = {x.key: x for x in at.segmented_control}
chk("en-tête : Mode · Critère · Appui en segmented_control ; défaut Calcul manuel · Élastique · Bande c (valeurs brutes)",
    {"ppb_mode", "ppb_crit", "ppb_appui"} <= set(seg) and "ppb_perim" not in seg
    and at.session_state["ppb_crit"] == CRIT_ELAS and at.session_state["ppb_mode"] == MODE_MANUEL)
chk("options courtes à l'affichage (valeurs brutes inchangées) : Calcul manuel / Vérif. complète · Élastique / "
    "Plastique · Bande c / Toute la platine · Tête / plaque / Crosse / Scellée (ETA)",
    [kit.affiche(D, "mode", o) for o in CHAMPS["mode"]["o"]] == ["Calcul manuel", "Vérif. complète"]
    and [kit.affiche(D, "crit", o) for o in CHAMPS["crit"]["o"]] == ["Élastique", "Plastique"]
    and [kit.affiche(D, "appui", o) for o in CHAMPS["appui"]["o"]] == ["Bande c", "Toute la platine"]
    and [kit.affiche(D, "typ_a", o) for o in CHAMPS["typ_a"]["o"]] == ["Tête / plaque", "Crosse", "Scellée (ETA)"]
    and all(len(kit.affiche(D, k, o)) <= 28 for k in CHAMPS for o in (CHAMPS[k].get("o") or [])))
chk("l'état de la bascule est dit dans les deux cas (amendement 4) : « Vérification complète : disponible pour cette "
    "disposition. »", "Vérification complète : disponible pour cette disposition." in t
    and EM.etat_bascule(dict(u0, raid=[dict(x=150, y=0, dir="+x", L=80, t=10)])).startswith(
        "Vérification complète : indisponible — "))
carte_w = {k for k in w if k.startswith("ppb_") and k[4:] in CHAMPS}
chk("(13) hef et dh ont quitté la carte : GEOMETRIE_DESSIN ∩ widgets de la carte = ∅ ; manuel/schemas.CLE_PAR_COTE "
    "reçoit dh",
    not carte_w & {"ppb_" + k for k in ecran_saisie.GEOMETRIE_DESSIN} and not kit.cles_carte(D) & ecran_saisie.GEOMETRIE_DESSIN
    and {"hef", "d_h"} <= ecran_saisie.GEOMETRIE_DESSIN and SM.CLE_PAR_COTE.get("dh") == "d_h", str(sorted(carte_w)))
av = expander(at, "Paramètres avancés")
chk("xN / yN dans Paramètres avancés › APPLICATION ; « Dessin interactif » quitte les avancés pour la tête du dessin",
    av is not None and {"ppb_xN", "ppb_yN"} <= {x.key for x in av.number_input}
    and "ppb_ui_composant" not in {x.key for x in av.checkbox} and "ppb_ui_composant" in w
    and ">APPLICATION</div>" in md(at))
chk("vocabulaire du mode (LIBELLE_MODE) : « Boulon », « Visserie par boulon » ; MATÉRIAUX à trois cases (pas Fissuré)",
    at.selectbox(key="ppb_ancr").label == "Boulon" and at.text_input(key="ppb_visserie").label == "Visserie par boulon"
    and "ppb_fiss" not in w and {"ppb_nu_pl", "ppb_beton", "ppb_hc_u"} <= w)
vis = EM.aides_visibles(u0)
chk("7 « ? » dans la carte au défaut : Mode, Critère, Appui, Ancrage, EFFORTS ELU, ÉLÉMENTS, x (mm) de la table",
    len(vis) == 7 and set(vis) == {"mode", "crit", "appui", "typ_a", "EFFORTS ELU", "ÉLÉMENTS", "tiges"}, str(vis))
chk("table des boulons du kit : en-tête ppb_th_tiges (n° · x (mm) ? · y (mm)) ; l'ancienne légende « Repère : "
    "origine… » est devenue le « ? » de x (mm)",
    "n°</div>" in t and "x (mm)" in t and ecran_saisie.AIDES_MANUEL["tiges"] in t
    and at.number_input(key="ppb_t0_x").help == ecran_saisie.AIDES_MANUEL["tiges"] and "Repère : origine" not in t)
chk("ÉLÉMENTS : HEA 300 du catalogue à l'axe sur UNE rangée (pas de ↳ É1 : rotation et dimensions dans sa fenêtre)",
    {"ppb_e0_type", "ppb_e0_prof", "ppb_e0_x", "ppb_e0_y"} <= w and "ppb_e0_rot" not in w and "ppb_e0_h" not in w)
# Relecture du 01/11/2026 : la zone de texte de la case Type mesure 63 px à 1351 et 75 px à 1440 (case de 113 /
# 125 px moins la flèche et 16 px de marge) ; « Tube RHS / SHS » (91 px), « Rectangle plein » (90), « Plat sur
# chant » (81) et « Profilé I / H » (64, à 1351) y étaient rognés. Libellés courts, mesurés ≤ 58,8 px.
sel_type = at.selectbox(key="ppb_e0_type")
chk("type d'élément : libellés courts (Profilé I/H · Tube RHS · Tube CHS · Plat · Rond · Rectangle), valeurs brutes "
    "inchangées, mêmes libellés dans la fenêtre de l'élément",
    list(sel_type.options) == ["Profilé I/H", "Tube RHS", "Tube CHS", "Plat", "Rond", "Rectangle"]
    and sel_type.value == EM.TYPES_ELEM[0] and list(EM.AFF_TYPES) == list(EM.TYPES_ELEM)
    and EM._panneaux(R0, u0)["elem0"]["champs"][0]["affichage"] is EM.AFF_TYPES, str(list(sel_type.options)))
chk("RAIDISSEURS : sans raidisseur, ni en-tête ni rangée — ➕ Raidisseur seul",
    any(b.key == "ppb_btn_radd" for b in at.button) and "ppb_r0_x" not in w and ">RAIDISSEURS</div>" in t)
# Relecture du 01/11/2026 : « RAIDISSEURS » et « APPLICATION » mesuraient 76,5 px pour une case de 76 px et se
# coupaient à leur césure (« RAIDIS-SEURS », « APPLI-CATION ») ; un titre ne se coupe jamais (charte §2). Plus de
# césure, un espacement serré de 0,05 em (CSS du module) : 74,0 px mesurés dans le navigateur, une ligne.
chk("RAIDISSEURS et APPLICATION sans césure, espacement serré de 0,05 em dans leurs quatre lieux (ligne sans "
    "raidisseur, en-tête de table, titre au-dessus sous 1351 px, avancés)",
    "\u00ad" not in EM.TITRE_RAID + ecran_saisie.TITRE_APPLICATION and "\u00ad" not in t + md(at)
    and EM.ESPACEMENT_SERRE == ".05em" and "{letter-spacing:.05em !important;}" in EM.CSS_MODULE
    and all(f'st-key-ppb_{k}"' in EM.CSS_MODULE for k in ("bt_raidisseurs_", "bt_av_application_", "tt_raid", "th_raid"))
    and kit.slug(ecran_saisie.TITRE_APPLICATION) == "application")
at.checkbox(key="ppb_ui_composant").uncheck(); run(at)
w2 = cles_widgets(at)
cotes = expander(at, "Cotes")
kc = {x.key for x in cotes.number_input} if cotes is not None else set()
chk("dessin fixe : panneau Cotes = GEOMETRIE_DESSIN ∩ champ_visible (hef, dh, tg compris) ; ↳ É1 (rotation) dans la carte",
    {"ppb_cote_hef", "ppb_cote_d_h", "ppb_cote_t_g", "ppb_cote_bp_u", "ppb_cote_a_w"} <= kc
    and "ppb_cote_w_x" not in kc and "ppb_e0_rot" in w2, str(sorted(kc)))
at.checkbox(key="ppb_ui_composant").check(); run(at)
u_sans = dict(u0, tiges=[])
R_sans = compute(u_sans)
g = EM._panneaux(R_sans, u_sans)
coupe = SM.coupe_aa(R_sans, SM.options_ecran(R_sans))
chk("(11) zéro boulon, dessin interactif : la coupe n'émet plus hef, mais la fenêtre Platine (toujours cliquable) "
    "porte hef et dh",
    "hef" not in [k for k, _s, _i in SM.cotes_modifiables(coupe)] and {"hef", "d_h"} <= {c["key"] for c in g["plate"]["champs"]}
    and 'data-group="plate"' in SM.plan(R_sans, SM.options_ecran(R_sans)).svg())
u_dh0 = dict(u0, d_h=0)
R_dh0 = compute(u_dh0)
chk("dh = 0 : étiquette « dh 0 » éditable sur la coupe A-A (comme « Δz 0 »)",
    "d_h" in [k for k, _s, _i in SM.cotes_modifiables(SM.coupe_aa(R_dh0, SM.options_ecran(R_dh0)))])
u_tg = dict(u0, t_g=0)
R_tg = compute(u_tg)
chk("tg = 0 (amendement 2) : tg dans la fenêtre Massif ET dans le panneau Cotes ; tg > 0 : fenêtre Mortier seulement",
    "t_g" in {c["key"] for c in EM._panneaux(R_tg, u_tg)["pedestal"]["champs"]} and "t_g" in ecran_resultats.cotes_panneau(u_tg)
    and "t_g" not in {c["key"] for c in EM._panneaux(R0, u0)["pedestal"]["champs"]})
g0 = EM._panneaux(R0, u0)
chk("fenêtres d'objet : titres courts (Platine, Massif, Mortier, Soudure, Boulons, Efforts) ; a_w seulement dans "
    "Soudure ; αcc hors du Massif ; libellés LIBELLE du mode (« Boulon ») ; xN / yN restent dans Efforts",
    [g0[k]["titre"] for k in ("plate", "pedestal", "grout", "weld", "anchors", "efforts")]
    == ["Platine", "Massif", "Mortier", "Soudure", "Boulons", "Efforts"]
    and [k for k, grp in g0.items() if "a_w" in {c["key"] for c in grp["champs"]}] == ["weld"]
    and "a_cc" not in {c["key"] for c in g0["pedestal"]["champs"]}
    and next(c for c in g0["anchors"]["champs"] if c["key"] == "ancr")["lab"] == "Boulon"
    and {"xN", "yN"} <= {c["key"] for c in g0["efforts"]["champs"]}
    and next(c for c in g0["anchors"]["champs"] if c["key"] == "typ_a")["affichage"]["Tige à tête (ou plaque d'ancrage)"]
    == "Tête / plaque")
chk("alertes d'information regroupées : « ⚠️ 2 remarques — Tiges de coin · Soulèvement », textes intégraux dedans",
    "⚠️ 2 remarques — Tiges de coin · Soulèvement" in expanders(at) and "Tiges de coin : longueur efficace plafonnée" in t)
chk("onglet Vérifications : bloc d'état et table des quatre critères en tête ; les tables de détail dans le repli "
    "« Le détail, comme à la main » ; légende (INT modèle d'ingénieur), « Non vérifiés ici », Tab. 3.3 repliés",
    "Le détail, comme à la main" in expanders(at) and "Légende et références" in expanders(at)
    and "Pinces et entraxes (Tab. 3.3) : conforme" in expanders(at)
    and any(x.startswith("Périmètre de la vérification — 4 non revérifiés") for x in expanders(at))
    and "Non vérifiés ici : cordons ; raidisseurs eux-mêmes ; poteau ; béton d'ancrage" in t
    and "modèle d'ingénieur (sections au nu, bandes, angles)" in t
    and t.index("Platine — épaisseur requise et ses quatre critères") < t.index("Côté tendu — boulon par boulon"))


# --- le calcul à la main dit le taux du critère RETENU (relecture du 01/11/2026)
# Sous le défaut élastique, la table « boulon par boulon » montrait « FT,Rd 149 » suivi de « η 54,4 % » (σvm/fy,
# pas T/FT,Rd = 26 %) et la table des sections « Mpl 5,17 » suivi de « η 46,9 % » (M/Mpl = 31 %). La colonne qui
# précède η est désormais la grandeur du critère retenu : σvm en élastique, FT,Rd / Mpl en plastique.
def table_detail(at, marque):
    """(en-têtes, rangées) de la table de détail dont l'en-tête contient ``marque`` ; texte des cellules."""
    import re
    for m in at.markdown:
        v = str(m.value)
        if "<table" in v and marque in v:
            nu = lambda c: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip()  # noqa: E731
            return ([nu(c) for c in re.findall(r"<th[^>]*>(.*?)</th>", v)],
                    [[nu(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", r)] for r in re.findall(r"<tr>(.*?)</tr>", v)
                     if "<td" in r])
    return [], []


from acier.formats import F as _F, pct as _pct  # noqa: E402
pt0, ps0 = R0.pire_tige, R0.pire_section
eb, rb = table_detail(at, "T·m<br>(kNm)")
es, rs = table_detail(at, "ΣT·m<br>(kNm)")
lb = next((r for r in rb if r and r[0] == str(pt0["i"] + 1)), [])
ls_ = next((r for r in rs if r and r[0] == ps0["nom"].replace("élément ", "É")), [])
chk("écran, élastique (défaut) : boulon 1 — σvm 188 MPa puis η 54,4 % (= σvm / 345), plus de colonne FT,Rd ; section "
    "au nu — σvm 162 puis η 46,9 % ; FT,Rd et Mpl en information plastique dans les légendes",
    R0.elastique and eb[6:8] == ["σvm (MPa)", "η"] and not any("FT,Rd" in x for x in eb)
    and lb[6:8] == [_F(pt0["svm"], 0), _pct(pt0["eta"], 1)] and _pct(pt0["eta"], 1) == "54,4 %"
    and es[5:7] == ["σvm (MPa)", "η"] and not any("Mpl" in x for x in es)
    and ls_[5:7] == [_F(ps0["svm"], 0), _pct(ps0["eta"], 1)] and _pct(ps0["eta"], 1) == "46,9 %"
    and f"FT,Rd = {_F(pt0['FT_Rd'], 0)} kN, mode {pt0['mode_ts']} (plastique, information)" in md(at)
    and f"Mpl = 0,25·b·tp²·fy/γM0 = {_F(ps0['Mpl'], 2)} kNm pour" in md(at)
    and "Critère élastique retenu : η = σvm / (fy/γM0), fy/γM0 = 345 MPa" in md(at)
    # le σ garde sa casse sous les en-têtes en capitales du kit (« ΣVM » se lisait comme une somme)
    and md(at).count("<span style='text-transform:none'>σvm</span>") == 2, str([eb, lb, es, ls_]))
at.segmented_control(key="ppb_crit").set_value(CRIT_EC3); run(at)
Rp0 = compute(lu(at))
ptp, psp = Rp0.pire_tige, Rp0.pire_section
ebp, rbp = table_detail(at, "T·m<br>(kNm)")
esp, rsp = table_detail(at, "ΣT·m<br>(kNm)")
lbp = next((r for r in rbp if r and r[0] == str(ptp["i"] + 1)), [])
lsp = next((r for r in rsp if r and r[0] == psp["nom"].replace("élément ", "É")), [])
chk("écran, plastique : boulon 1 — FT,Rd 149 (mode 2) puis η 26,1 % = T/FT,Rd ; section au nu — Mpl 5,17 puis η 30,8 %",
    not Rp0.elastique and ebp[6:8] == ["FT,Rd (kN)", "η"] and lbp[6:8] == [_F(ptp["FT_Rd"], 0) + " (m" + str(ptp["mode_ts"]).split(" ")[0] + ")", _pct(ptp["eta"], 1)]
    and _pct(ptp["T"] / ptp["FT_Rd"], 1) == _pct(ptp["eta"], 1) == "26,1 %"
    and esp[5:7] == ["Mpl (kNm)", "η"] and lsp[5:7] == [_F(psp["Mpl"], 2), _pct(psp["eta"], 1)] and _pct(psp["eta"], 1) == "30,8 %"
    and "Critère élastique retenu" not in md(at), str([ebp, lbp, esp, lsp]))
at.segmented_control(key="ppb_crit").set_value(CRIT_ELAS); run(at)
chk("onglet Méthode : « Ce que l'outil vérifie » (le texte de l'ancien ❔) en tête", EM.METHODE.lstrip().startswith(
    "#### Ce que l'outil vérifie"))

# --- note et export : rédacteur, hypothèses, conclusion
Rr = compute(dict(u0, id_red="J. Dupont"))
pr = pymupdf.open(stream=note.generer_note(Rr, {"nom_projet": "Halle A", "partie": "A1", "date": "31/10/2026",
                                                "indice": "0"}), filetype="pdf")
chk("le rédacteur saisi est imprimé au cartouche du plan (p. 2) et dans l'export texte (amendement 3)",
    "J.Dupont" in pr[1].get_text().replace(" ", "") and "Rédacteur : J. Dupont" in texte_ppb.construire_texte(Rr)
    and "Rédacteur" not in texte_ppb.construire_texte(R0))
hyp = NM.lignes_hypotheses(R0)
chk("HYPOTHÈSES du défaut : trois lignes exactes (charte §11, CHARTE_decisions), ≤ 110 caractères",
    hyp == ["Calcul manuel — critère élastique (Von Mises ≤ fy/γM0) ; appui : bande c = 59 mm (EC3 §6.2.5).",
            "Platine rigide, lit sans traction (kc 109 N/mm³, kb 454 kN/mm) ; bras m à l'appui, cordon 9,1 mm déduit.",
            "Leff ≤ L/2 + m ; tiges, appui : information ; γM0 1,00 (ANB) · γM2 1,25 · γc 1,50 (αcc 0,85)."], str(hyp))
v_pl = NM.lignes_hypotheses(compute(dict(u0, crit=CRIT_EC3)))[0]
v_plein = NM.lignes_hypotheses(compute(dict(u0, appui=APPUI_PLEIN)))[0]
R43 = compute(dict(u0, tp_u=43, crit=CRIT_EC3))
v_43 = NM.lignes_hypotheses(R43)[0]
chk("variantes : plastique, toute la platine, bande c à trois chiffres (forme courte : sans « (EC3 §6.2.5) »)",
    v_pl == "Calcul manuel — critère plastique EN 1993-1-8 (Mpl, console élastique) ; appui : bande c = 59 mm (EC3 §6.2.5)."
    and v_plein.endswith("; appui : toute la platine (rigide).")
    and R43.c >= 99.5 and v_43.endswith("; appui : bande c = " + str(round(R43.c)) + " mm.") and len(v_43) <= 110,
    str([v_pl, v_plein, v_43]))
pertes, trop, deb = [], [], []
for c in BM.CAS:
    Rc = compute(dict(c["inp"]))
    segs = NM.segments_hypotheses(Rc)
    ls = NM.lignes_hypotheses(Rc)
    if len(ls) > 3 or any(len(x) > 110 for x in ls):
        trop.append(c["id"])
    txt = " ".join(ls)
    for sg in segs:
        formes = [sg] if isinstance(sg, str) else list(sg)
        if not any(f[1:] in txt for f in formes):
            pertes.append((c["id"], formes[-1][:30]))
    dc = pymupdf.open(stream=note.generer_note(Rc, {}), filetype="pdf")
    if dc.page_count != 2 or note.derniers_avertissements:
        deb.append(c["id"])
chk("les 4 cas figés (VAL-M1 à VAL-M4) : ≤ 3 lignes ≤ 110, aucun segment perdu, note de 2 pages sans débordement",
    not trop and not pertes and not deb, str([trop, pertes, deb]))
s_el = pymupdf.open(stream=note.generer_note(R0, {}), filetype="pdf")[0].get_text().replace(" ", "").replace("\n", "")
s_pl = pymupdf.open(stream=note.generer_note(compute(dict(u0, crit=CRIT_EC3)), {}), filetype="pdf")[0].get_text() \
    .replace(" ", "").replace("\n", "")
chk("CONCLUSION de la note : elle dit le critère retenu (« selon Von Mises ≤ fy/γM0 (critère élastique) » au défaut, "
    "« selon EN 1993-1-8 §6.2.4 (tronçons en T) » en plastique)",
    "selonVonMises≤fy/γM0(critèreélastique)" in s_el and "selonEN1993-1-8§6.2.4(tronçonsenT)" not in s_el
    and "selonEN1993-1-8§6.2.4(tronçonsenT)" in s_pl)
# Le calcul à la main de la NOTE dit le taux du critère retenu (relecture du 01/11/2026) : sous le défaut
# élastique, « Boulon 1 … → FT,Rd = 149 kN (mode 2) → 26 % » et « Section … Mpl = 5,17 kNm → 47 % » contredisaient
# la table des critères (54 %). Le texte de la page est relu sans espaces (ni insécables) : « σvm=188MPa≤… ».
s_el, s_pl = s_el.replace("\xa0", ""), s_pl.replace("\xa0", "")


def segment(s, debut, fins=("Section", "Bandecomprimée", "Trous", "Épaisseurrequise")):
    """Le texte d'une ligne du calcul à la main : de ``debut`` à la ligne suivante."""
    i = s.find(debut)
    if i < 0:
        return ""
    j = min([k for k in (s.find(f, i + len(debut)) for f in fins) if k > 0] or [len(s)])
    return s[i:j]


def sans_esp(x):
    return x.replace(" ", "")


pt0, ps0 = R0.pire_tige, R0.pire_section
nb_el = segment(s_el, f"Boulon{pt0['i'] + 1}T=")
ns_el = segment(s_el, "Section" + sans_esp(ps0["nom"].replace("élément ", "É")) + ":")
chk("note, élastique : la ligne Boulon 1 dit « σvm = 188 MPa ≤ fy/γM0 = 345 MPa → 54 % » (plus « → 26 % ») et la "
    "ligne Section « σvm = 162 MPa ≤ 345 MPa → 47 % » (plus « Mpl = 5,17 kNm → »)",
    sans_esp(f"σvm = {_F(pt0['svm'], 0)} MPa ≤ fy/γM0 = {_F(R0.ck['ptl'].Rd, 0)} MPa → {_pct(pt0['eta'], 0)}") in nb_el
    and sans_esp(f"→ {_pct(pt0['T'] / pt0['FT_Rd'], 0)}") not in nb_el and "(plastique,information)" in nb_el
    and sans_esp(f"σvm = {_F(ps0['svm'], 0)} MPa ≤ {_F(R0.ck['pts'].Rd, 0)} MPa → {_pct(ps0['eta'], 0)}") in ns_el
    and sans_esp(f"Mpl = {_F(ps0['Mpl'], 2)} kNm →") not in ns_el
    and _pct(pt0["eta"], 0) == "54 %" and _pct(ps0["eta"], 0) == "47 %", nb_el + " | " + ns_el)
Rpl0 = compute(dict(u0, crit=CRIT_EC3))
ptl0 = Rpl0.pire_tige
nb_pl = segment(s_pl, f"Boulon{ptl0['i'] + 1}T=")
chk("note, plastique : la ligne Boulon 1 porte T/FT,Rd — « FT,Rd = 149 kN (mode 2) → 26 % »",
    sans_esp(f"FT,Rd = {_F(ptl0['FT_Rd'], 0)} kN (mode {str(ptl0['mode_ts']).split(' ')[0]}) → {_pct(ptl0['eta'], 0)}") in nb_pl
    and _pct(ptl0["T"] / ptl0["FT_Rd"], 0) == _pct(ptl0["eta"], 0) == "26 %" and "σvm" not in nb_pl, nb_pl)

# --- rien ne se perd (mode manuel)
variantes = [u0, dict(u0, typ_a=TYPE_CROSSE), dict(u0, nu_pl="Personnalisé", fy_u=355, fu_u=510, bw_u=0.9),
             dict(u0, raid=[dict(x=150, y=0, dir="+x", L=80, t=10)]), dict(u0, t_g=0)]
manquantes = set()
for v in variantes:
    Rv = compute(v)
    atteignables = (kit.cles_carte(D) | {c["key"] for grp in EM._panneaux(Rv, v).values() for c in grp["champs"]}
                    | set(ecran_resultats.cotes_panneau(v)) | set(ecran_resultats.CLES_TETE_DESSIN) | set(EM.CLES_TABLES))
    manquantes |= {k for k in CHAMPS if champ_visible(v, k)} - atteignables
chk("rien ne se perd (manuel) : chaque clé visible en calcul manuel a une destination (carte, avancés, cartouche, "
    "tables, fenêtres, panneau Cotes, tête du dessin)", not manquantes and len(CHAMPS) == 61, str(sorted(manquantes)))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

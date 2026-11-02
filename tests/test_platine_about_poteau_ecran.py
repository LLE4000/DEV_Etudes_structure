# -*- coding: utf-8 -*-
"""Écran, dessins et note du module platine d'about sur poteau (AppTest,
application réelle).

Chaque garantie rougit si on la retire :
  1. NAVIGATION : la carte « Platine d'about (hauteur partielle) sur poteau »
     sur la page de sélection, « Créer » mène au module, mêmes boutons
     d'outils.
  2. ÉCRAN : titre court + barre d'outils partagée, deux colonnes, colonne du
     dessin FIGÉE (CSS sticky, conteneur pac_col_dessin), deux composants,
     carte par objet rendue par le kit (mode ET APPUI en segmented_control,
     « Dessin interactif » en tête du dessin, visserie au cartouche) SANS la
     géométrie (elle vit sur le dessin), 66 clés en session ; onglet
     Vérifications : légende, périmètre et Tableau 3.3 dans des replis.
  3. LA CHARTE : rien ne se perd (CHAMPS ∪ GEOMETRIE_DESSIN atteignables),
     5 « ? » visibles au défaut, libellés ≤ 16 / 20 / 22, aides ≤ 160,
     options affichées ≤ 28, aucune ligne d'objet > 4 widgets, préfixe étanche.
  4. LA BASCULE « APPUI » : le groupe « Poteau porteur » existe TOUJOURS (la
     platine est boulonnée) mais son contenu change — cisaillement local de
     l'âme en cas (b), tronçon en T de l'aile et âme en traction transversale
     en cas (a) sous traction ; le GAUGE refuse le p3 = 90 ; la limite de la
     flexion hors plan est écrite en cas (b).
  5. SOURCE UNIQUE : un message du dessin (cote, panneau) écrit dans
     pac_<clé> et recalcule ; VEd = 900 → NON VÉRIFIÉ.
  6. DESSINS : les trois vues en SVG valide dans les DEUX cas d'appui, avec
     une géométrie de canevas non vide ; cotes réparties SANS doublon sur la
     planche du plan de principe.
  7. NOTE : 2 pages A4 paysage sans débordement (les deux appuis + les 19 cas
     figés), HYPOTHÈSES ≤ 3 × 110, cartouche POTEAU · POUTRE avec l'appui.
  8. BENCHMARK dans l'application : statut VALIDÉ, un contrôle par cas.
  9. EXPORTS et MÉTHODE. 9 bis. Le contrat du PROJET. 10. ÉTANCHÉITÉ.
 11. Tableau 3.3 NON CONFORME.

Lancement : python3 tests/test_platine_about_poteau_ecran.py
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

from acier.assemblages import carte as kit  # noqa: E402
from acier.assemblages.poutre_poteau.platine_about_poteau import (interface, ecran_saisie,  # noqa: E402
                                                                  ecran_resultats, entrees, moteur,
                                                                  schemas, note, benchmark, methode)

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
    derniers = at.session_state["pac_ui_dernier_clic"]
    if derniers.get("pac_cmp_elev") != t:
        derniers["pac_cmp_elev"] = t
        for k, v in changes.items():
            at.session_state["pac_" + k] = ecran_saisie.valeur_widget(k, v)


# ================================================================
print("=== 1. Navigation ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=300)
at.session_state["page"] = "Assemblages métalliques"
run(at)
chk("la carte « Platine d'about (hauteur partielle) sur poteau » est sur la page de sélection",
    "Platine d'about (hauteur partielle) sur poteau" in md(at))
chk("la famille « Poutre – Poteau » est titrée", "#### Poutre – Poteau" in md(at))
at.button(key="asm_btn_open_platine_about_poteau").click(); run(at)
chk("« Créer » mène au module", not at.exception and at.session_state["asm_courant"] == "platine_about_poteau")
chk("en-tête du module", "platine d'about" in md(at) and "poutre–poteau" in md(at))

# ================================================================
print("\n=== 2. Écran ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("pac_") and k[4:] in interface.CLES]
chk("les 66 clés d'entrée existent en session", len(cles) == 66, str(len(cles)))
t = md(at)
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ, dimensionnante l'âme de la poutre portée (39,1 %)",
    "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t and "39,1 %" in t)
chk("taux par élément : Boulons, Platine, Portée, Poteau, Soudure (le poteau est TOUJOURS là)",
    all(x in t for x in ("Boulons", "Platine", "Portée", "Poteau", "Soudure")))
chk("deux composants empilés dans la colonne figée (CSS sticky sur pac_col_dessin)",
    len(at.get("component_instance")) == 2 and "position: sticky" in t and "st-key-pac_col_dessin" in t)
w = cles_widgets(at)
donnees = {k for k in w if k.startswith("pac_") and k[4:] in interface.CLES}
chk("la géométrie ne se règle QUE sur le dessin (aucune cote dans la carte)",
    not donnees & {"pac_" + k for k in ecran_saisie.GEOMETRIE_DESSIN}, str(sorted(donnees)))
chk("carte par objet : mode, appui, profilés, nuance de la platine, boulons, rangées, efforts, visserie",
    {"pac_mode_calc", "pac_appui", "pac_prof_P", "pac_prof_S", "pac_nu_pl", "pac_boulon_u",
     "pac_n1_u", "pac_V_Ed", "pac_N_Ed", "pac_visserie"} <= w, str(sorted(w))[:200])
chk("le mode ET l'appui sont des segmented_control (valeurs brutes inchangées)",
    {"pac_mode_calc", "pac_appui"} <= {x.key for x in at.segmented_control}
    and at.session_state["pac_mode_calc"] == "VÉRIFICATION"
    and at.session_state["pac_appui"] == "Aile du poteau")
chk("« Dessin interactif » en tête de la colonne dessin, titre court et barre partagée",
    "pac_ui_composant" in w and "Platine d'about sur poteau" in t and "version 1.0" in t
    and {"pac_btn_retour", "pac_btn_home", "pac_btn_save_dl", "pac_btn_pdf", "pac_btn_reset"}
    <= {b.key for b in list(at.button) + list(at.get("download_button"))})
chk("γM1 n'est pas demandé (aucune instabilité de plat) ; γMu caché tant qu'il n'y a pas de tying",
    "pac_g_M1" not in w and "pac_g_Mu" not in w and "g_M1" not in entrees.defaults())
chk("onglet par défaut : Vérifications ; Tableau 3.3 replié, verdict dans le titre",
    at.session_state["pac_ui_onglet"] == "Vérifications"
    and "Groupe : cisaillement et pression diam." in t
    and "Pinces et entraxes (Tab. 3.3) : conforme" in expanders(at))
chk("légende des natures après les tables, références dans un repli",
    "formule de l'Eurocode" in t and "Légende et références" in expanders(at)
    and "Références : EC3 = EN 1993-1-8" in t)
chk("colonne Réf. sans commentaire interne", "même modèle que le module" not in t)
chk("« Paramètres retenus » dit le cas d'appui, le gauge minimal et le moment apporté au poteau",
    "Cas d'appui" in t and "Gauge minimal du poteau" in t and "Moment apporté au POTEAU" in t
    and "32,00 kNm" in t)

# ================================================================
print("\n=== 3. La charte (données de la carte) ===")
D = ecran_saisie.D
u0 = entrees.defaults()
cc = kit.cles_carte(D)
R0 = moteur.compute({})
fen = {c["key"] for g in ecran_resultats._panneaux(R0, u0).values() for c in g["champs"]}
attendues = set(entrees.CHAMPS) | set(ecran_saisie.GEOMETRIE_DESSIN) | {"ui_niveau", "ui_composant"}
manquantes = sorted(attendues - set(ecran_saisie.GEOMETRIE_DESSIN) - (cc | fen | set(ecran_resultats.CLES_TETE_DESSIN)))
chk("rien ne se perd : chaque clé hors géométrie a une destination (carte, fenêtre, tête du dessin)",
    not manquantes, str(manquantes))
chk("la géométrie ne se règle jamais dans la carte (GEOMETRIE_DESSIN ∩ carte = ∅)",
    not (cc & set(ecran_saisie.GEOMETRIE_DESSIN)))
Rame = moteur.compute(dict(appui=entrees.APPUI_AME, bp_u=190, p3_u=110))
uame = dict(u0, appui=entrees.APPUI_AME, bp_u=190, p3_u=110)


def manque_geometrie(u, R):
    opt = schemas.options_ecran(R, 1, None)
    opt.editables = True
    cotes = {k for v in ("elevation", "plan", "coupe_aa")
             for k, _s, _i in schemas.cotes_modifiables(getattr(schemas, v)(R, opt))}
    f = {c["key"] for g in ecran_resultats._panneaux(R, u).values() for c in g["champs"]}
    return sorted(k for k in ecran_saisie.GEOMETRIE_DESSIN
                  if entrees.champ_visible(u, k) and k not in cotes | f)


chk("rien ne se perd : chaque cote VISIBLE est émise par un dessin ou portée par une fenêtre "
    "(les deux appuis, la recoupe, la poutre grugée)",
    not any(manque_geometrie(u, moteur.compute(u)) for u in (
        u0, uame, dict(uame, rec_S=150, prof_S="HEA 300"),
        dict(u0, d_nt=50, d_nb=40, l_n=120, z_C=60, hp_u=240, n1_u=3))),
    str([manque_geometrie(u, moteur.compute(u)) for u in (u0, uame)]))
chk("la visserie reste atteignable (cartouche)", "visserie" in cc)
chk("ONGLETS inchangés", ecran_resultats.ONGLETS == ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode"))
chk("« ? » visibles dans la carte au défaut : 5 (Mode, Appui, Trou, Catégorie, EFFORTS ELU)",
    kit.aides_visibles(D, u0) == ["mode_calc", "appui", "trou", "cat", "EFFORTS ELU"],
    str(kit.aides_visibles(D, u0)))
chk("aide(k) is None hors AIDES ; toute aide ≤ 160 caractères",
    not [k for k in entrees.CHAMPS if k not in D.AIDES and kit.aide(D, k) is not None]
    and not [k for k, v in D.AIDES.items() if len(v) > 160])
chk("aucun « ? » sur Nuance, Profilé, Boulon, Classe, dimensions, efforts, identification",
    not ({"nu_P", "nu_S", "nu_pl", "prof_P", "prof_S", "boulon_u", "classe", "hP_u", "hS_u",
          "V_Ed", "N_Ed", "id_rep", "id_red"} & set(D.AIDES)))
chk("libellés ≤ 22 partout, ≤ 16 dans une rangée de 4, ≤ 20 dans une rangée de 3 ou une continuation",
    not [v for v in D.LIBELLE.values() if len(v) > 22]
    and not [k for g, cs in kit.rangees(D) if g == "objet" and len(cs) == 4 for k in cs
             if len(D.LIBELLE.get(k, "")) > 16]
    and not [k for g, cs in kit.rangees(D) if g == "suite" or len(cs) == 3 for k in cs
             if len(D.LIBELLE.get(k, "")) > 20])
chk("options affichées ≤ 28 et AFFICHAGE ⊂ options brutes",
    not [o for m in D.AFFICHAGE.values() for o in m.values() if len(o) > 28]
    and all(set(m) <= set(entrees.CHAMPS[k]["o"] or []) for k, m in D.AFFICHAGE.items()))
chk("aucune ligne d'objet > 4 widgets", not [cs for g, cs in kit.rangees(D) if g == "objet" and len(cs) > 4])
chk("étanchéité du préfixe : pac_ est disjoint des autres écrans",
    ecran_saisie.PREFIXE == "pac_" and interface.PREFIXE == "pac_"
    and not any(p in ecran_saisie.PREFIXE for p in ("pab_", "fpc_", "fpl_", "asm_", "ppb_")))
chk("les fenêtres d'objet lisent LIBELLE (une forme par clé)",
    all(c["lab"] == D.LIBELLE[c["key"]] for g in ecran_resultats._panneaux(Rame, uame).values()
        for c in g["champs"]),
    str([(c["key"], c["lab"]) for g in ecran_resultats._panneaux(Rame, uame).values()
         for c in g["champs"] if c["lab"] != D.LIBELLE.get(c["key"])][:4]))
chk("fenêtre Poteau : le cas d'appui et le profilé ; fenêtre Groupe : p3 ; fenêtre Poutre portée : "
    "dc, c et — en cas (b) — rc et j ; la gorge a ne reste que dans Soudure",
    {"appui", "prof_P"} <= {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["col"]["champs"]}
    and "p3_u" in {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["G"]["champs"]}
    and {"d_nt", "d_nb", "l_n", "rec_S", "j_lat"}
    <= {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["beamS"]["champs"]}
    and "a_w" not in {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["plate"]["champs"]}
    and "a_w" in {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["weld"]["champs"]})

# ================================================================
print("\n=== 4. La bascule « appui » ===")
chk("cas (a) : la table « Poteau porteur » porte la pression diamétrale de l'aile",
    "##### Poteau porteur" in t and "Pression diamétrale" in t)
chk("cas (a) : le repli « Périmètre » dit les 8 limites, et le cisaillement local de l'âme est "
    "SANS OBJET",
    any(x.startswith("Périmètre de la vérification — 8 non revérifiés") for x in expanders(at))
    and "Cisaillement local de l'âme – brut" in t,
    str([x for x in expanders(at) if x.startswith("Périmètre")]))
at.segmented_control(key="pac_appui").set_value("Âme du poteau"); run(at)
t2 = md(at)
chk("cas (b) : la platine 290 × 220 ne tient pas entre les congés — alerte bloquante, « Localiser », "
    "et la limite chiffrée (198 mm) est dite",
    "ne tient pas entre les congés de l'âme" in t2 and "198 mm ; actuel : bp = 220 mm" in t2
    and any(b.key == "pac_btn_al_bp_libreP" for b in at.button) and "ASSEMBLAGE NON VÉRIFIÉ" in t2)
message(at, {"bp_u": 190, "p3_u": 110}, 2001); run(at)
t2 = md(at)
chk("cas (b) : p3 = 110 / bp = 190 → VÉRIFIÉ, et le cisaillement local de l'âme paraît (brut et net)",
    "ASSEMBLAGE VÉRIFIÉ" in t2 and "Cisaillement local de l'âme – brut" in t2
    and "Cisaillement local de l'âme – net" in t2)
chk("cas (b) : la limite de la flexion HORS PLAN de l'âme est écrite (9 limites)",
    "flexion HORS PLAN de l'âme du poteau" in t2
    and any(x.startswith("Périmètre de la vérification — 9 non revérifiés") for x in expanders(at)),
    str([x for x in expanders(at) if x.startswith("Périmètre")]))
chk("cas (b) : l'alerte d'information ame_hp est affichée",
    any(b.key == "pac_btn_al_ame_hp" for b in at.button))
chk("cas (b) : « Paramètres retenus » donne la largeur droite, la largeur maximale de la platine et "
    "le passage de la semelle",
    "Largeur droite de l'âme entre congés" in t2 and "Largeur maximale de la platine" in t2
    and "Passage de la semelle entre ailes" in t2)
at.segmented_control(key="pac_appui").set_value("Aile du poteau"); run(at)
message(at, {"bp_u": 220, "p3_u": 140}, 2002); run(at)
chk("retour en cas (a) : le cisaillement local redevient sans objet et l'assemblage est vérifié",
    "ASSEMBLAGE VÉRIFIÉ" in md(at) and "Cisaillement local de l'âme – brut" in md(at))
# --- LE GAUGE : la règle qui refuse le p3 usuel des âmes
message(at, {"p3_u": 90, "bp_u": 170}, 2003); run(at)
t3 = md(at)
chk("p3 = 90 mm sur un HEB 300 : alerte BLOQUANTE, le minimum 102 mm et les gauges usuels sont dits",
    "l'écrou chevauche le congé" in t3 and any(b.key == "pac_btn_al_p3_congeP" for b in at.button)
    and "ASSEMBLAGE NON VÉRIFIÉ" in t3)
at.button(key="pac_btn_al_p3_congeP").click(); run(at)
chk("chip d'alerte : p3 modifiable sur place, et la raison chiffrée est à l'écran",
    any(x.key == "pac_fix_p3_u" for x in at.number_input)
    and "Prendre p3 = 105 mm" in md(at))
at.number_input(key="pac_fix_p3_u").set_value(140.0); run(at)
message(at, {"bp_u": 220}, 2004); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["pac_p3_u"]) == 140.0 and "l'écrou chevauche le congé" not in md(at))
# --- la bascule « nature de NEd » : le TYING ouvre γMu et les deux tronçons en T
at.number_input(key="pac_N_Ed").set_value(250.0); run(at)
chk("NEd > 0 : « Nature de NEd » paraît sur la ligne EFFORTS ELU (ELU par défaut)",
    "pac_nat_N" in cles_widgets(at) and at.session_state["pac_nat_N"] == "ELU fondamental"
    and "pac_g_Mu" not in cles_widgets(at))
tel = md(at)
chk("NEd à l'ELU : les DEUX tronçons en T sont à l'écran (platine et AILE DU POTEAU) avec l'âme en "
    "traction transversale",
    all(x in tel for x in ("Flexion en traction (tronçon en T)", "Aile en flexion (tronçon en T)",
                           "Âme en traction transversale", "Poinçonnement")))
at.selectbox(key="pac_nat_N").set_value("Tying (accidentel)"); run(at)
tty = md(at)
chk("TYING : γMu paraît dans les avancés (défaut 1,10) et l'alerte le dit",
    "pac_g_Mu" in cles_widgets(at) and at.session_state["pac_g_Mu"] == 1.1
    and "NEd en tying : force de chaînage vérifiée seule (γMu)" in tty and "γMu = 1,10" in tty)
chk("TYING : « Paramètres retenus » donne les deux tronçons en T et leur rapport",
    "Tronçon en T de la PLATINE" in tty and "Tronçon en T de l'AILE DU POTEAU" in tty
    and "Rapport aile du poteau / platine" in tty and "3,66" in tty)
at.number_input(key="pac_N_Ed").set_value(0.0); run(at)
chk("retour à NEd = 0 : « Nature de NEd » et γMu redisparaissent",
    "pac_nat_N" not in cles_widgets(at) and "pac_g_Mu" not in cles_widgets(at))

# ================================================================
print("\n=== 5. Source unique ===")
message(at, {"V_Ed": 900}, 2005); run(at)
chk("message du dessin : VEd = 900 → NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
message(at, {"V_Ed": 200, "hp_u": 140}, 2006); run(at)
chk("hp = 140 → alerte bloquante pince basse, « Localiser » disponible",
    "pince basse" in md(at) and any(b.key == "pac_btn_al_e1bot" for b in at.button))
at.button(key="pac_btn_al_e1bot").click(); run(at)
chk("chip d'alerte : hp modifiable sur place", any(x.key == "pac_fix_hp_u" for x in at.number_input))
at.number_input(key="pac_fix_hp_u").set_value(290.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["pac_hp_u"]) == 290.0 and "pince basse" not in md(at))
at.session_state["pac_ui_composant"] = False; run(at)
chk("dessin fixe : le panneau « Cotes » paraît et porte hp, bp, zp, e1, p1, p3, tp, a",
    "Cotes" in expanders(at)
    and {"pac_cote_hp_u", "pac_cote_bp_u", "pac_cote_z_C", "pac_cote_e1_u", "pac_cote_p1_u",
         "pac_cote_p3_u", "pac_cote_tp_u", "pac_cote_a_w"} <= cles_widgets(at),
    str(sorted(k for k in cles_widgets(at) if k.startswith("pac_cote_"))))
at.session_state["pac_ui_composant"] = True; run(at)

# ================================================================
print("\n=== 6. Dessins ===")
for nom, cas in (("aile", {}), ("âme", dict(appui=entrees.APPUI_AME, bp_u=190, p3_u=110)),
                 ("âme recoupée", dict(appui=entrees.APPUI_AME, bp_u=190, p3_u=110, prof_S="HEA 300",
                                       rec_S=145, z_C=45, hp_u=190, n1_u=3, p1_u=60, e1_u=35)),
                 ("grugée", dict(d_nt=50, d_nb=40, l_n=120, z_C=60, hp_u=240, n1_u=3)),
                 ("1 rangée", dict(n1_u=1, hp_u=180, e1_u=40, z_C=60))):
    Rd_ = moteur.compute(cas)
    for f, excl in (("elevation", schemas.EXCLURE_ELEVATION), ("plan", schemas.EXCLURE_PLAN),
                    ("coupe_aa", schemas.EXCLURE_COUPE)):
        for mode, o in (("écran", schemas.options_ecran(Rd_)),
                        ("fabrication", schemas.options_fabrication(excl, 12))):
            d = getattr(schemas, f)(Rd_, o)
            svg = d.svg(identifiant="t")
            try:
                ET.fromstring(svg)
            except Exception as e:  # noqa: BLE001
                chk(f"SVG valide – {nom} / {f} / {mode}", False, str(e)[:120])
            # contrôle de GÉOMÉTRIE du canevas : une vue peut être complète
            # dans le DOM et vide à l'écran
            vb = d.viewbox
            if not (vb[2] > 50 and vb[3] > 50 and len(d.corps) >= 3 and len(d.cotes) >= 2):
                chk(f"géométrie du canevas – {nom} / {f} / {mode}", False,
                    f"viewbox {vb}, {len(d.corps)} pièces, {len(d.cotes)} cotes")
chk("les trois vues rendent du SVG valide et un canevas non vide, dans les deux cas d'appui "
    "(5 configurations × 3 vues × écran/fabrication)", True)
R0 = moteur.compute({})
Ra = moteur.compute(dict(appui=entrees.APPUI_AME, bp_u=190, p3_u=110))
chk("le poteau est une PIÈCE nommée sur l'élévation et sur la vue en plan",
    any(p.get("grp") == "col" for p in schemas.elevation(R0, schemas.options_ecran(R0)).corps)
    and any(p.get("grp") == "col" for p in schemas.plan(R0, schemas.options_ecran(R0)).corps))
pl = schemas.plan(Ra, schemas.options_ecran(Ra))
chk("vue en plan, cas (b) : le poteau est en coupe HACHURÉE (la poutre arrive entre les ailes)",
    any(p.get("t") == "g" and p.get("grp") == "col"
        and any(e.get("t") == "path" and "ht" in e["cls"] for e in p["enfants"]) for p in pl.corps))
cp = schemas.coupe_aa(Ra, schemas.options_fabrication(schemas.EXCLURE_COUPE, 12)).svg()
chk("coupe A-A, cas (b) : la LARGEUR DROITE entre congés est cotée (ℓ = 208 mm)",
    "208" in cp and "ℓ" in cp)
se = schemas.elevation(R0, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)).svg()
sp = schemas.plan(R0, schemas.options_fabrication(schemas.EXCLURE_PLAN, 12)).svg()
sc = schemas.coupe_aa(R0, schemas.options_fabrication(schemas.EXCLURE_COUPE, 12)).svg()
chk("fabrication : renvoi 8×Ø22 sur l'élévation et sur la coupe A-A",
    ">8×Ø22<" in se and ">8×Ø22<" in sc)
doubles = [s for s in ("hp = 290", "zp 40", "bS = 180", "bp = 220", "tp 10", "p3 = 140",
                       "h = 300", "b = 300", "e = 80")
           if (se + sp + sc).count(s) > 1]
chk("aucune cote posée deux fois sur la planche du plan de principe", not doubles, str(doubles))
sea = schemas.elevation(Ra, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)).svg()
spa = schemas.plan(Ra, schemas.options_fabrication(schemas.EXCLURE_PLAN, 12)).svg()
sca = schemas.coupe_aa(Ra, schemas.options_fabrication(schemas.EXCLURE_COUPE, 12)).svg()
doubles_b = [s for s in ("hp = 290", "h = 300", "b = 300", "ℓ = 208", "p3 = 110", "j ", "rc ")
             if (sea + spa + sca).count(s) > 1]
chk("cas (b) : la largeur hP est cotée sur la vue en plan, bP sur l'élévation, chacune UNE fois",
    not doubles_b and "b = 300" in sea and "h = 300" in spa, str(doubles_b))

# ================================================================
print("\n=== 7. Note (2 pages) ===")
Rn = moteur.compute(dict(id_projet="Halle A", id_rep="P3/S9", id_red="J. Dupont"))
pdf = note.generer_note(Rn, {"nom_projet": "Halle A", "partie": "P3/S9", "date": "02/11/2026",
                             "indice": "0", "visserie": "1 rondelle + 1 écrou"})
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("deux pages A4 paysage, sans débordement, échelle et disposition exposées",
    doc.page_count == 2 and not note.derniers_avertissements
    and note.derniere_echelle is not None and note.derniere_disposition is not None,
    str(note.derniers_avertissements))
t1 = doc[0].get_text(); s1 = t1.replace(" ", "").replace("\n", "")
n1 = " ".join(t1.split())          # le PDF coupe les lignes : on cherche dans le texte normalisé
chk("page 1 : bandeau, en-tête compact 3 lignes, hypothèses, tableaux, conclusion",
    "VÉRIFIÉ" in s1 and "Platine 290×220×10 S355" in t1 and "Soudure double a 5" in t1
    and "VEd 200,0 kN" in t1 and "PLATINE D'ABOUT" in t1.upper()
    and "ASSEMBLAGE VÉRIFIÉ à l'ELU" in t1)
chk("page 1 : le cas d'appui est dans l'en-tête ET dans les hypothèses",
    "appui : aile du poteau" in n1 and "rotule à la face de l'aile du poteau" in n1)
chk("page 1 : le groupe CENTRÉ et le moment apporté au POTEAU sont imprimés",
    "groupe centré" in n1 and "Au poteau VEd·e = 32,0 kNm" in n1)
chk("page 1 : γM0 ANB écrit dans les hypothèses", "γM0 1,00 (ANB)" in n1)
chk("page 1 : la table « Poteau porteur » EST imprimée (la platine est boulonnée)",
    "POTEAUPORTEUR" in t1.replace(" ", "").upper())
chk("page 1 : le gauge p3 est dans l'en-tête des fixations", "p3 140" in n1)
# la ligne « Notations » du pied : elle doit TENIR sur la largeur de la note
from acier.assemblages.poutre_poteau.platine_about_poteau.notation import NOTATIONS  # noqa: E402
_lu = " ".join(t1.split())
chk("pied de la note : les lignes « Références » et « Notations » tiennent sans être coupées",
    "…" not in _lu.split("Notations :")[-1]
    and "…" not in _lu.split("Références :")[-1].split("Notations")[0]
    and NOTATIONS.split(" : ", 1)[1][:40] in _lu, _lu[-220:])
chk("HYPOTHÈSES : ≤ 3 lignes de ≤ 110 caractères (note_kit.empaqueter)",
    1 <= len(note.lignes_hypotheses(Rn)) <= 3 and all(len(l) <= 110 for l in note.lignes_hypotheses(Rn)),
    str([len(l) for l in note.lignes_hypotheses(Rn)]))
t2 = doc[1].get_text(); s2 = t2.replace(" ", "").replace("\n", "")
n2 = " ".join(t2.split())
chk("page 2 : PLAN DE PRINCIPE, trois vues titrées, échelle affichée",
    "PLANDEPRINCIPE" in s2 and all(x in s2 for x in ("ÉLÉVATION", "VUEENPLAN", "COUPEA-A"))
    and "Échelle1:" in s2)
# les libellés du cartouche sont tracés avec un interlettrage (track) : le PDF les rend
# « P O T E A U » — on les cherche donc sans espaces
chk("page 2 : cartouche POTEAU · POUTRE, appui, PLATINE, visserie, Ø des trous et RÉDACTEUR",
    "POTEAU·POUTRE" in s2 and "appui : aile du poteau" in n2 and "290×220×10 S355" in n2
    and "par boulon : 1 rondelle + 1 écrou" in n2 and "trous Ø22" in n2 and "J. Dupont" in n2,
    f"{'POTEAU·POUTRE' in s2} {'appui : aile du poteau' in n2} {'J. Dupont' in n2}")
Rb = moteur.compute(dict(appui=entrees.APPUI_AME, bp_u=190, p3_u=110))
pdfb = note.generer_note(Rb, {"date": "02/11/2026", "indice": "0"})
docb = pymupdf.open(stream=pdfb, filetype="pdf")
tb1 = " ".join(docb[0].get_text().split()); tb2 = " ".join(docb[1].get_text().split())
chk("cas (b) : 2 pages, le cartouche dit « âme du poteau », le cisaillement local est dans les "
    "hypothèses et la flexion hors plan dans l'alerte du bandeau",
    docb.page_count == 2 and not note.derniers_avertissements
    and "appui : âme du poteau" in tb2
    and "Appui dans l'âme : cisaillement local un seul côté" in tb1
    and "flexion hors plan de l'âme" in tb1.lower(),
    f"{docb.page_count} {note.derniers_avertissements}")
with open("tests/ref_pac.json", encoding="utf-8") as fh:
    REF = json.load(fh)
nb = 0
for nom, cas in REF.items():
    p = note.generer_note(moteur.compute(cas["inp"]), {})
    dc = pymupdf.open(stream=p, filetype="pdf")
    if dc.page_count != 2 or note.derniers_avertissements or note.derniere_echelle is None:
        nb += 1
        print("      ", nom, dc.page_count, note.derniere_echelle, note.derniers_avertissements)
chk(f"les {len(REF)} cas figés : 2 pages, échelle trouvée, sans débordement", nb == 0, str(nb))
nh = [nom for nom, cas in REF.items()
      if not (1 <= len(note.lignes_hypotheses(moteur.compute(cas["inp"]))) <= 3
              and all(len(l) <= 110 for l in note.lignes_hypotheses(moteur.compute(cas["inp"]))))]
chk("HYPOTHÈSES de tous les cas figés : ≤ 3 lignes de ≤ 110", not nh, str(nh))

# ================================================================
print("\n=== 8. Benchmark dans l'application ===")
at.session_state["pac_ui_onglet"] = "Benchmark"; run(at)
tb = md(at)
chk("onglet Benchmark : les quatre cas, les calculs manuels, statut VALIDÉ",
    all(x in tb for x in ("BM1", "BM2", "BM3", "BM4", "VAL-A", "VAL-B", "VAL-C",
                          "Benchmark du module : VALIDÉ")))
B = benchmark.run_bench()
chk("un contrôle par cas de benchmark : chaque ligne de chaque cas est à son écart annoncé",
    all(r["st"] in ("OK", "Écart expliqué") for b in B["bench"] for r in b["rows"])
    and all(r["st"] == "OK" for c in B["valid"] for r in c["rows"]),
    str([(b["id"], r["lab"][:40], r["st"]) for b in B["bench"] for r in b["rows"]
         if r["st"] not in ("OK", "Écart expliqué")][:3]))
chk("BM1 (héritage) : les neuf valeurs publiées, pire écart 0,95 %",
    len(B["bench"][0]["rows"]) == 9 and B["worst"] <= 0.01, str(B["worst"]))
chk("sept « vérifications sans exemple publié consulté » sont nommées", len(benchmark.NOCOVER) == 7)
at.button(key="pac_btn_bm_0").click(); run(at)
chk("« Charger ces données » : l'exemple publié dans l'outil (VEd = 550, hp = 430)",
    float(at.session_state["pac_V_Ed"]) == 550.0 and float(at.session_state["pac_hp_u"]) == 430.0)
at.button(key="pac_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts", float(at.session_state["pac_hp_u"]) == 290.0)

# ================================================================
print("\n=== 9. Prédim, exports, méthode ===")
at.segmented_control(key="pac_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
chk("bandeau prédimensionnement", "Mode prédimensionnement" in md(at))
at.session_state["pac_ui_onglet"] = "Prédim"; run(at)
boutons = [b for b in at.button if b.key and b.key.startswith("pac_btn_sol_")]
chk("solutions proposées cliquables", len(boutons) >= 1)
chk("le détail dit que le gauge proposé dégage le congé du poteau",
    "écrou hors congé du poteau" in md(at))
retenue = [b for b in boutons if b.type == "primary"]
(retenue[0] if retenue else boutons[0]).click(); run(at)
chk("« Appliquer » : VÉRIFICATION + platine, p3 et boulons recopiés",
    at.session_state["pac_mode_calc"] == "VÉRIFICATION"
    and float(at.session_state["pac_p3_u"]) >= 102.0)
at.button(key="pac_btn_reset").click(); run(at)
at.button(key="pac_btn_pdf").click(); run(at)
chk("« 📄 PDF » depuis l'application : 2 pages",
    bool(at.session_state.get("pac_pdf_bytes"))
    and pymupdf.open(stream=at.session_state["pac_pdf_bytes"], filetype="pdf").page_count == 2)
at.session_state["pac_ui_onglet"] = "Note"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Note : export texte avec statut, appui, gauge et références",
    "PLATINE D'ABOUT" in code and "VÉRIFIÉ" in code and "appui : aile du poteau" in code
    and "gauge minimal" in code and "Tableau 3.3 : conforme" in code)
from acier.assemblages.poutre_poteau.platine_about_poteau import texte as texte_pac  # noqa: E402
chk("le rédacteur saisi figure dans l'export texte (amendement 3)",
    "Rédacteur : J. Dupont" in texte_pac.construire_texte(moteur.compute(dict(id_red="J. Dupont")))
    and "Rédacteur" not in texte_pac.construire_texte(moteur.compute(dict())))
chk("export texte, cas (b) : la largeur droite entre congés et la flexion hors plan non vérifiée",
    "largeur droite entre congés" in texte_pac.construire_texte(Rb)
    and "flexion HORS PLAN" in texte_pac.construire_texte(Rb))
at.session_state["pac_ui_onglet"] = "Méthode"; run(at)
tm = md(at)
chk("onglet Méthode : ce que l'outil vérifie, les deux cas d'appui, le GAUGE, l'ANB, les limites, "
    "les notations",
    all(x in tm for x in ("Ce que l'outil vérifie", "Les deux cas d'appui", "Le GAUGE",
                          "Annexe nationale belge", "Données de bibliothèque", "Limites du modèle",
                          "Notations")))
chk("onglet Méthode : les NEUF limites du §8.4 et les points tranchés P-1, P-3, P-4, P-5, P-6, P-7, P-9",
    all(l.split(" ")[0] in tm for l in methode.limites(True))
    and all(x in tm for x in ("**P-1**", "**P-3**", "**P-4**", "**P-5**", "**P-6**", "**P-7**", "**P-9**"))
    and len(methode.limites(True)) == 9 and len(methode.limites(False)) == 8)

# ================================================================
print("\n=== 9 bis. Le contrat du PROJET (registre, description, rapport mixte) ===")
from acier import registre, projet as modele, rapport_projet as rp  # noqa: E402

_t = registre.trouver("platine_about_poteau")
chk("le registre déclare le module DISPONIBLE, avec son court (≤ 18) et sa vignette",
    _t is not None and _t[0].id == "poutre_poteau" and _t[1].disponible
    and _t[1].court == "Platine/poteau" and len(_t[1].court) <= 18
    and 'role="img"' in _t[1].vignette)
_m = registre.charger(_t[1])
chk("le module expose le contrat du projet (CLES, defaults, compute, charger_payload, "
    "MODULE_VERSION, ecran_saisie.valeur_widget, note.generer_note)",
    len(_m.CLES) == 66 and callable(_m.compute) and callable(_m.charger_payload)
    and _m.MODULE_VERSION == "1.0" and callable(_m.ecran_saisie.valeur_widget)
    and callable(__import__(_m.__name__.rsplit(".", 1)[0] + ".note", fromlist=["x"]).generer_note))
chk("acier/projet.py:description lit prof_S et prof_P sans changement",
    modele.description(entrees.defaults()).startswith("IPE 400 sur HEB 300"),
    modele.description(entrees.defaults()))
_p = modele.nouveau()


def _defauts(type_id, **surcharge):
    mm = registre.charger(registre.trouver(type_id)[1])
    uu = {k: mm.ecran_saisie.valeur_widget(k, v) for k, v in mm.defaults().items()}
    uu.update(surcharge)
    return uu


modele.ajouter(_p, "platine_about", _defauts("platine_about"))
modele.ajouter(_p, "platine_about_poteau", _defauts("platine_about_poteau", id_rep="P5/S2"))
modele.ajouter(_p, "platine_about_poteau", _defauts("platine_about_poteau", appui=entrees.APPUI_AME,
                                                    bp_u=190, p3_u=110, id_rep="P6/S2"))
_el = rp.notes(_p, {"nom_projet": "Halle A", "partie": "Hall", "date": "02/11/2026", "indice": "0"})
chk("rapport de projet MIXTE (un pab_ + deux pac_) : trois notes de 2 pages, états lisibles",
    [n for (_e, _i, _pdf, n) in _el] == [2, 2, 2]
    and all(i["etat"] in ("ok", "att") for (_e, i, _pdf, _n) in _el),
    str([(i["etat"], round(i["eta"], 3)) for (_e, i, _pdf, _n) in _el]))
_doc = pymupdf.open(stream=rp.generer_rapport(_p, {"nom_projet": "Halle A", "partie": "Hall",
                                                   "date": "02/11/2026", "indice": "0"}),
                    filetype="pdf")
_txt = " ".join(" ".join(_doc[i].get_text().split()) for i in range(2))
chk("le récapitulatif nomme le type et les deux repères du module neuf",
    _doc.page_count == 2 + 6 and "P5/S2" in _txt and "P6/S2" in _txt
    and "Platine d'about (hauteur partielle) sur poteau" in _txt, f"{_doc.page_count} pages")

# ================================================================
print("\n=== 10. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict()
          if k.startswith(("b1_", "dal", "pre", "asm_", "fpl_", "fpc_", "pab_", "ppb_"))
          and k != "asm_courant"]
chk("aucune clé d'un autre écran créée par le module", not fuites, str(fuites[:5]))
at.button(key="pac_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

# ================================================================
print("\n=== 11. Tableau 3.3 NON CONFORME ===")
at33 = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=300)
at33.session_state["page"] = "Assemblages métalliques"
run(at33)
at33.button(key="asm_btn_open_platine_about_poteau").click(); run(at33)
at33.session_state["pac_e1_u"] = 20.0
run(at33)
e33 = [x for x in at33.expander if str(x.label).startswith("Pinces et entraxes (Tab. 3.3)")]
chk("e1 = 20 mm → « Pinces et entraxes (Tab. 3.3) : NON CONFORME », expander ouvert d'office",
    bool(e33) and str(e33[0].label) == "Pinces et entraxes (Tab. 3.3) : NON CONFORME" and e33[0].proto.expanded,
    str([str(x.label) for x in e33]))
at33.button(key="pac_btn_dist_e1p").click(); run(at33)
chk("« Localiser » de la ligne NON OK sélectionne son alerte (dist_e1p) ; sa rangée dit « Masquer »",
    at33.session_state["pac_ui_alerte"] == "dist_e1p"
    and at33.button(key="pac_btn_al_dist_e1p").label == "Masquer",
    str(at33.session_state["pac_ui_alerte"]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

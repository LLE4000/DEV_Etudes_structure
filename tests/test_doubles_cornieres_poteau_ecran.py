# -*- coding: utf-8 -*-
"""Écran, dessins et note du module doubles cornières sur poteau (AppTest,
application réelle).

Chaque garantie rougit si on la retire :
  1. NAVIGATION : la carte « Doubles cornières d'âme sur poteau » sur la page
     de sélection, « Créer » mène au module, mêmes boutons d'outils.
  2. ÉCRAN : titre court + barre d'outils partagée, deux colonnes, colonne du
     dessin FIGÉE (CSS sticky, conteneur dcc_col_dessin), deux composants,
     carte par objet rendue par le kit (mode ET APPUI en segmented_control,
     « Dessin interactif » en tête du dessin, visserie au cartouche) SANS la
     géométrie (elle vit sur le dessin), 88 clés en session ; onglet
     Vérifications : matrice des cornières, légende, périmètre et Tableau 3.3
     dans des replis.
  3. LA CHARTE : rien ne se perd (CHAMPS ∪ GEOMETRIE_DESSIN atteignables),
     6 « ? » visibles au défaut, libellés ≤ 16 / 20 / 22, aides ≤ 160,
     options affichées ≤ 28, aucune ligne d'objet > 4 widgets, préfixe étanche.
  4. LA BASCULE « APPUI » : en cas (b) le groupe « Poteau porteur » porte le
     cisaillement local de l'âme et la limite de la flexion hors plan est
     écrite ; en cas (a) il porte la pression diamétrale de l'aile, et le
     repli « Périmètre » dit qu'aucun cisaillement local de l'aile n'est
     demandé. La bascule « nature de NEd » ouvre γMu et les deux tronçons en T.
  5. SOURCE UNIQUE : un message du dessin (cote, glissement, panneau) écrit
     dans dcc_<clé> et recalcule ; VEd = 600 → NON VÉRIFIÉ.
  6. DESSINS : les trois vues en SVG valide dans les DEUX cas d'appui, avec
     une géométrie de canevas non vide ; cotes réparties SANS doublon sur la
     planche du plan de principe.
  7. NOTE : 2 pages A4 paysage sans débordement (les deux appuis + les cas
     figés), HYPOTHÈSES ≤ 3 × 110, cartouche POTEAU · POUTRE avec l'appui.
  8. BENCHMARK dans l'application : statut VALIDÉ, un contrôle par cas.
  9. EXPORTS et MÉTHODE. 10. ÉTANCHÉITÉ. 11. Tableau 3.3 NON CONFORME.

Lancement : python3 tests/test_doubles_cornieres_poteau_ecran.py
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
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau import (  # noqa: E402
    interface, ecran_saisie, ecran_resultats, entrees, moteur, schemas, note, benchmark, methode)

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
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input)
            + list(at.checkbox) + list(at.segmented_control) if w.key}


def expanders(at):
    return [str(x.label) for x in at.expander]


def table_apres(at, titre):
    """Le tableau HTML qui SUIT le titre « ##### <titre> » — mesurer ce que la
    table porte, pas ce que la page garde (les libellés des vérifications
    inactives paraissent aussi dans le repli « Périmètre »)."""
    vals = [str(m.value) for m in at.markdown]
    for i, v in enumerate(vals):
        if v.strip() == "##### " + titre:
            return vals[i + 1] if i + 1 < len(vals) else ""
    return ""


def message(at, changes, t):
    derniers = at.session_state["dcc_ui_dernier_clic"]
    if derniers.get("dcc_cmp_elev") != t:
        derniers["dcc_cmp_elev"] = t
        for k, v in changes.items():
            at.session_state["dcc_" + k] = ecran_saisie.valeur_widget(k, v)


# ================================================================
print("=== 1. Navigation ===")
at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=300)
at.session_state["page"] = "Assemblages métalliques"
run(at)
chk("la carte « Doubles cornières d'âme sur poteau » est sur la page de sélection",
    "Doubles cornières d'âme sur poteau" in md(at))
chk("la famille « Poutre – Poteau » est titrée", "#### Poutre – Poteau" in md(at))
at.button(key="asm_btn_open_doubles_cornieres_poteau").click(); run(at)
chk("« Créer » mène au module",
    not at.exception and at.session_state["asm_courant"] == "doubles_cornieres_poteau")
chk("en-tête du module", "doubles cornières" in md(at) and "poutre–poteau" in md(at))

# ================================================================
print("\n=== 2. Écran ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("dcc_") and k[4:] in interface.CLES]
chk("les 88 clés d'entrée existent en session", len(cles) == 88, str(len(cles)))
t = md(at)
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ, dimensionnante la pression diamétrale de l'âme portée",
    "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t and "53,5 %" in t)
chk("taux par élément : Boulons, Cornières, Portée, Poteau",
    all(x in t for x in ("Boulons", "Cornières", "Portée", "Poteau")))
chk("deux composants empilés dans la colonne figée (CSS sticky sur dcc_col_dessin)",
    len(at.get("component_instance")) == 2 and "position: sticky" in t
    and "st-key-dcc_col_dessin" in t)
w = cles_widgets(at)
donnees = {k for k in w if k.startswith("dcc_") and k[4:] in interface.CLES}
chk("la géométrie ne se règle QUE sur le dessin (aucune cote dans la carte)",
    not donnees & {"dcc_" + k for k in ecran_saisie.GEOMETRIE_DESSIN}, str(sorted(donnees)))
chk("carte par objet : mode, appui, profilés, cornière, boulons, fixations, rangées, efforts, visserie",
    {"dcc_mode_calc", "dcc_appui", "dcc_prof_P", "dcc_prof_S", "dcc_corn_u", "dcc_nu_C",
     "dcc_boulon_u", "dcc_fix_P", "dcc_fix_S", "dcc_n1S_u", "dcc_n1P_u", "dcc_V_Ed",
     "dcc_H_Ed", "dcc_visserie"} <= w, str(sorted(w))[:200])
chk("le mode ET l'appui sont des segmented_control (valeurs brutes inchangées)",
    {"dcc_mode_calc", "dcc_appui"} <= {x.key for x in at.segmented_control}
    and at.session_state["dcc_mode_calc"] == "VÉRIFICATION"
    and at.session_state["dcc_appui"] == "Aile du poteau")
chk("« Dessin interactif » en tête de la colonne dessin, titre court et barre partagée",
    "dcc_ui_composant" in w and "Cornières sur poteau" in t and "version 1.0" in t
    and {"dcc_btn_retour", "dcc_btn_home", "dcc_btn_save_dl", "dcc_btn_pdf", "dcc_btn_reset"}
    <= {b.key for b in list(at.button) + list(at.get("download_button"))})
chk("γMu caché tant qu'il n'y a pas de tying ; « Orientation » cachée pour une cornière à ailes égales",
    "dcc_g_Mu" not in w and "dcc_orient" not in w)
chk("onglet par défaut : Vérifications ; matrice des cornières ; Tableau 3.3 replié",
    at.session_state["dcc_ui_onglet"] == "Vérifications"
    and "Cornières — aile A / aile B" in t
    and "Pinces et entraxes (Tab. 3.3) : conforme" in expanders(at))
chk("légende des natures après les tables, références dans un repli",
    "formule de l'Eurocode" in t and "Légende et références" in expanders(at)
    and "Références : EC3 = EN 1993-1-8" in t)
chk("« Paramètres retenus » dit le cas d'appui, le gauge p3 et le moment apporté au poteau",
    "Cas d'appui" in t and "Gauge p3 = tw,S + 2·gA" in t and "Moment apporté au POTEAU" in t
    and "40,00 kNm" in t)

# ================================================================
print("\n=== 3. La charte (données de la carte) ===")
D = ecran_saisie.D
u0 = entrees.defaults()
cc = kit.cles_carte(D)
R0 = moteur.compute({})
fen = {c["key"] for g in ecran_resultats._panneaux(R0, u0).values() for c in g["champs"]}
attendues = set(entrees.CHAMPS) | set(ecran_saisie.GEOMETRIE_DESSIN) | {"ui_niveau", "ui_composant"}
manquantes = sorted(attendues - set(ecran_saisie.GEOMETRIE_DESSIN)
                    - (cc | fen | set(ecran_resultats.CLES_TETE_DESSIN)))
chk("rien ne se perd : chaque clé hors géométrie a une destination (carte, fenêtre, tête du dessin)",
    not manquantes, str(manquantes))
chk("la géométrie ne se règle jamais dans la carte (GEOMETRIE_DESSIN ∩ carte = ∅)",
    not (cc & set(ecran_saisie.GEOMETRIE_DESSIN)))
uame = dict(u0, appui=entrees.APPUI_AME, corn_u="L90x90x9")
Rame = moteur.compute(uame)


def manque_geometrie(u, R):
    opt = schemas.options_ecran(R, 1, None)
    opt.editables = True
    cotes = {k for v in ("elevation", "plan", "coupe_aa")
             for k, _s, _i in schemas.cotes_modifiables(getattr(schemas, v)(R, opt))}
    f = {c["key"] for g in ecran_resultats._panneaux(R, u).values() for c in g["champs"]}
    return sorted(k for k in ecran_saisie.GEOMETRIE_DESSIN
                  if entrees.champ_visible(u, k) and k not in cotes | f)


_cas_geo = (u0, uame, dict(uame, rec_S=150),
            dict(u0, n2S_u="2", p2_S=60, n2P_u="2", p2_P=60, corn_u="L150x150x12", gA_u=55),
            dict(u0, d_nt=50, d_nb=40, l_n=120, z_C=60, LC_u=240, n1S_u=3, n1P_u=3),
            dict(u0, fix_P=entrees.SOUDEE), dict(u0, fix_S=entrees.SOUDEE),
            dict(u0, fix_P=entrees.SOUDEE, fix_S=entrees.SOUDEE))
chk("rien ne se perd : chaque cote VISIBLE est émise par un dessin ou portée par une fenêtre "
    "(les deux appuis, les deux fixations, deux files, poutre grugée, recoupe)",
    not any(manque_geometrie(u, moteur.compute(u)) for u in _cas_geo),
    str([manque_geometrie(u, moteur.compute(u)) for u in _cas_geo]))
chk("la visserie reste atteignable (cartouche)", "visserie" in cc)
chk("ONGLETS inchangés",
    ecran_resultats.ONGLETS == ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode"))
chk("« ? » visibles dans la carte au défaut : 6 (Mode, Appui, Trou, Catégorie, ↳ AILES B, EFFORTS ELU)",
    ecran_saisie.aides_visibles(u0) == ["mode_calc", "appui", "trou", "cat", "AILES B",
                                        "EFFORTS ELU"],
    str(ecran_saisie.aides_visibles(u0)))
chk("aide(k) is None hors AIDES ; toute aide ≤ 160 caractères",
    not [k for k in entrees.CHAMPS if k not in D.AIDES and kit.aide(D, k) is not None]
    and not [k for k, v in D.AIDES.items() if len(v) > 160])
chk("aucun « ? » sur Nuance, Profilé, Boulon, Classe, dimensions, efforts, identification",
    not ({"nu_P", "nu_S", "nu_C", "prof_P", "prof_S", "boulon_u", "classe", "hP_u", "hS_u",
          "V_Ed", "N_Ed", "H_Ed", "M_Ed", "id_rep", "id_red"} & set(D.AIDES)))
chk("libellés ≤ 22 partout, ≤ 16 dans une rangée de 4, ≤ 20 dans une rangée de 3 ou une continuation",
    not [v for v in D.LIBELLE.values() if len(v) > 22]
    and not [k for g, cs in kit.rangees(D) if g == "objet" and len(cs) == 4 for k in cs
             if len(D.LIBELLE.get(k, "")) > 16]
    and not [k for g, cs in kit.rangees(D) if g == "suite" or len(cs) == 3 for k in cs
             if len(D.LIBELLE.get(k, "")) > 20])
chk("options affichées ≤ 28 et AFFICHAGE ⊂ options brutes",
    not [o for m in D.AFFICHAGE.values() for o in m.values() if len(o) > 28]
    and all(set(m) <= set(entrees.CHAMPS[k]["o"] or []) for k, m in D.AFFICHAGE.items()))
chk("aucune ligne d'objet > 4 widgets",
    not [cs for g, cs in kit.rangees(D) if g == "objet" and len(cs) > 4])
chk("étanchéité du préfixe : dcc_ est disjoint des autres écrans",
    ecran_saisie.PREFIXE == "dcc_" and interface.PREFIXE == "dcc_"
    and not any(p in ecran_saisie.PREFIXE for p in ("asm_", "fpl_", "fpc_", "pab_", "ppb_")))
chk("les fenêtres d'objet lisent LIBELLE (une forme par clé)",
    all(c["lab"] == D.LIBELLE[c["key"]] for g in ecran_resultats._panneaux(Rame, uame).values()
        for c in g["champs"]),
    str([(c["key"], c["lab"]) for g in ecran_resultats._panneaux(Rame, uame).values()
         for c in g["champs"] if c["lab"] != D.LIBELLE.get(c["key"])][:4]))
chk("fenêtre Poteau : le cas d'appui et le profilé ; fenêtre Poutre portée : gh, dc, c et — en "
    "cas (b) — rc et j ; fenêtre Groupe P : gA ; les gorges a ne restent que dans les Cordons",
    {"appui", "prof_P"} <= {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["col"]["champs"]}
    and {"g_h", "d_nt", "d_nb", "l_n", "rec_S", "j_lat"}
    <= {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["beamS"]["champs"]}
    and "gA_u" in {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["P"]["champs"]}
    and "a_P" not in {c["key"] for c in ecran_resultats._panneaux(Rame, uame)["cleat"]["champs"]}
    and "a_P" in {c["key"] for c in
                  ecran_resultats._panneaux(moteur.compute(dict(u0, fix_P=entrees.SOUDEE)),
                                            dict(u0, fix_P=entrees.SOUDEE))["weldP"]["champs"]})

# ================================================================
print("\n=== 4. La bascule « appui » ===")
chk("cas (a) : le repli « Périmètre » dit les 8 limites et l'absence de cisaillement local de l'aile",
    any(x.startswith("Périmètre de la vérification — 8 non revérifiés") for x in expanders(at))
    and "l'attache est centrée sur l'âme du poteau" in t,
    str([x for x in expanders(at) if x.startswith("Périmètre")]))
_tab_a = table_apres(at, "Poteau porteur")
chk("cas (a) : la table « Poteau porteur » porte la pression diamétrale de l'AILE, et rien de plus",
    "##### Poteau porteur" in t and "Pression diamétrale" in _tab_a
    and "Cisaillement local" not in _tab_a, _tab_a[:200])
at.segmented_control(key="dcc_appui").set_value("Âme du poteau"); run(at)
t2 = md(at)
chk("cas (b) : alerte BLOQUANTE bA_congeP — une L100 ne passe pas entre les congés du HEB 300",
    "ASSEMBLAGE NON VÉRIFIÉ" in t2 and "l'aile A dépasse la partie droite" in t2
    and any(b.key == "dcc_btn_al_bA_congeP" for b in at.button),
    str([b.key for b in at.button if b.key and b.key.startswith("dcc_btn_al_")]))
# une L90×90×9 passe : c'est le résultat utile du cas (b)
at.selectbox(key="dcc_corn_u").set_value("L90x90x9"); run(at)
t2 = md(at)
chk("cas (b) avec une L90×90×9 : VÉRIFIÉ, et la table « Poteau porteur » porte le cisaillement local",
    "ASSEMBLAGE VÉRIFIÉ" in t2 and "Cisaillement local brut" in table_apres(at, "Poteau porteur")
    and "Cisaillement local net" in table_apres(at, "Poteau porteur"))
chk("cas (b) : la limite de la flexion HORS PLAN de l'âme est écrite (9 limites)",
    "flexion HORS PLAN de l'âme du poteau" in t2
    and any(x.startswith("Périmètre de la vérification — 9 non revérifiés") for x in expanders(at)),
    str([x for x in expanders(at) if x.startswith("Périmètre")]))
chk("cas (b) : l'alerte d'information ame_hp est affichée",
    any(b.key == "dcc_btn_al_ame_hp" for b in at.button))
chk("cas (b) : « Paramètres retenus » donne la largeur droite, l'aile A maximale et le passage",
    "Largeur droite de l'âme entre congés" in t2 and "Longueur d'aile A maximale" in t2
    and "Passage de la semelle entre ailes" in t2)
at.segmented_control(key="dcc_appui").set_value("Aile du poteau"); run(at)
at.selectbox(key="dcc_corn_u").set_value("L100x100x10"); run(at)
chk("retour en cas (a) : le cisaillement local quitte la table du poteau",
    "Cisaillement local" not in table_apres(at, "Poteau porteur")
    and "Pression diamétrale" in table_apres(at, "Poteau porteur"))

# --- la bascule « nature de NEd » : le TYING ouvre γMu et les deux tronçons en T
at.number_input(key="dcc_N_Ed").set_value(300.0); run(at)
chk("NEd > 0 : « Nature de NEd » paraît (ELU par défaut), γMu toujours caché",
    "dcc_nat_N" in cles_widgets(at) and at.session_state["dcc_nat_N"] == "ELU fondamental"
    and "dcc_g_Mu" not in cles_widgets(at))
at.selectbox(key="dcc_nat_N").set_value("Tying (accidentel)"); run(at)
tty = md(at)
chk("TYING : γMu paraît dans les avancés (défaut 1,10) et l'alerte le dit",
    "dcc_g_Mu" in cles_widgets(at) and at.session_state["dcc_g_Mu"] == 1.1
    and "NEd en tying : force de chaînage vérifiée seule (γMu)" in tty and "γMu = 1,10" in tty)
chk("TYING : les composantes de la force de chaînage sont à l'écran, AILE DU POTEAU comprise",
    all(x in tty for x in ("Tying – cisaillement du groupe S", "Aile A – tronçon en T",
                           "Aile en flexion (tronçon en T)", "Âme en traction transversale",
                           "Poinçonnement", "Traction nette", "Bloc en traction")))
at.number_input(key="dcc_N_Ed").set_value(0.0); run(at)
chk("retour à NEd = 0 : « Nature de NEd » et γMu redisparaissent",
    "dcc_nat_N" not in cles_widgets(at) and "dcc_g_Mu" not in cles_widgets(at))

# ================================================================
print("\n=== 5. Source unique ===")
message(at, {"V_Ed": 600}, 1001); run(at)
chk("message du dessin : VEd = 600 → NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
message(at, {"V_Ed": 200, "LC_u": 140}, 1002); run(at)
chk("hc = 140 : alerte bloquante pince basse, « Localiser » disponible",
    "pince basse" in md(at) and any(b.key == "dcc_btn_al_S_e1bot" for b in at.button))
at.button(key="dcc_btn_al_S_e1bot").click(); run(at)
chk("chip d'alerte : hc modifiable sur place", any(x.key == "dcc_fix_LC_u" for x in at.number_input))
at.number_input(key="dcc_fix_LC_u").set_value(290.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["dcc_LC_u"]) == 290.0 and "pince basse" not in md(at))
message(at, {"g_h": 25}, 1003); run(at)
chk("glissement gh → clé écrite, recalcul",
    float(at.session_state["dcc_g_h"]) == 25.0 and not at.exception)
message(at, {"g_h": 10}, 1004); run(at)
at.session_state["dcc_ui_composant"] = False; run(at)
chk("dessin fixe : le panneau « Cotes » paraît et porte hc, zc, e1, p1, gA, gh",
    "Cotes" in expanders(at)
    and {"dcc_cote_LC_u", "dcc_cote_z_C", "dcc_cote_e1S_u", "dcc_cote_p1S_u",
         "dcc_cote_e1P_u", "dcc_cote_p1P_u", "dcc_cote_gA_u", "dcc_cote_g_h"} <= cles_widgets(at),
    str(sorted(k for k in cles_widgets(at) if k.startswith("dcc_cote_"))))
at.session_state["dcc_ui_composant"] = True; run(at)

# ================================================================
print("\n=== 6. Dessins ===")
for nom, cas in (("aile", {}), ("âme b0", dict(appui=entrees.APPUI_AME, corn_u="L90x90x9")),
                 ("âme recoupée", dict(appui=entrees.APPUI_AME, corn_u="L90x90x9",
                                       prof_S="HEA 300", rec_S=145, z_C=45, LC_u=190,
                                       n1S_u=3, n1P_u=3, p1S_u=60, p1P_u=60, e1S_u=35, e1P_u=35)),
                 ("grugée", dict(d_nt=50, d_nb=40, l_n=120, z_C=60, LC_u=240, n1S_u=3, n1P_u=3)),
                 ("soudée des deux côtés", dict(fix_P=entrees.SOUDEE, fix_S=entrees.SOUDEE)),
                 ("deux files", dict(n2S_u="2", p2_S=60, n2P_u="2", p2_P=60,
                                     corn_u="L150x150x12", gA_u=55))):
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
    "(6 configurations × 3 vues × écran/fabrication)", True)
R0 = moteur.compute({})
Ra = moteur.compute(dict(appui=entrees.APPUI_AME, corn_u="L90x90x9"))
poly = next(p for p in schemas.elevation(R0, schemas.options_ecran(R0)).corps
            if p.get("t") == "g" and p.get("grp") == "beamS")
chk("la poutre portée est glissable (drag gh ; plus de Δz : un poteau n'a pas de dessus)",
    poly.get("drag") == "g_h" and poly.get("drag2") is None)
chk("le poteau est une PIÈCE nommée sur l'élévation et la vue en plan",
    any(p.get("grp") == "col" for p in schemas.elevation(R0, schemas.options_ecran(R0)).corps)
    and any(p.get("grp") == "col" for p in schemas.plan(R0, schemas.options_ecran(R0)).corps))
pl = schemas.plan(Ra, schemas.options_ecran(Ra))
chk("vue en plan : le poteau est en coupe HACHURÉE (cas b : la poutre arrive entre les ailes)",
    any(p.get("t") == "g" and p.get("grp") == "col"
        and any(e.get("t") == "path" and "ht" in e["cls"] for e in p["enfants"]) for p in pl.corps))
chk("vue en plan : les DEUX cornières sont des pièces nommées (une par côté de l'âme)",
    len([p for p in pl.corps if p.get("t") == "g" and p.get("grp") == "cleat"]) == 2)
spa = schemas.plan(Ra, schemas.options_fabrication(schemas.EXCLURE_PLAN, 12)).svg()
chk("vue en plan, cas (b) : la LARGEUR DROITE entre congés est cotée (ℓ = 208 mm)",
    "208" in spa and "ℓ" in spa)
se = schemas.elevation(R0, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)).svg()
sp = schemas.plan(R0, schemas.options_fabrication(schemas.EXCLURE_PLAN, 12)).svg()
sc = schemas.coupe_aa(R0, schemas.options_fabrication(schemas.EXCLURE_COUPE, 12)).svg()
chk("fabrication : renvoi 4×Ø22 sur l'élévation (groupe S) et 8×Ø22 sur la coupe A-A (groupe P)",
    ">4×Ø22<" in se and ">8×Ø22<" in sc)
doubles = [s for s in ("hc = 290", "zc 40", "bS = 180", "bA = 100", "bB = 100", "p3 = 109",
                       "h = 300", "b = 300", "gA 50")
           if (se + sp + sc).count(s) > 1]
chk("aucune cote posée deux fois sur la planche du plan de principe", not doubles, str(doubles))
sea = schemas.elevation(Ra, schemas.options_fabrication(schemas.EXCLURE_ELEVATION, 12)).svg()
sca = schemas.coupe_aa(Ra, schemas.options_fabrication(schemas.EXCLURE_COUPE, 12)).svg()
doubles_b = [s for s in ("hc = 290", "h = 300", "b = 300", "ℓ = 208", "j ", "rc ")
             if (sea + spa + sca).count(s) > 1]
chk("cas (b) : la largeur hP est cotée sur la vue en plan, bP sur l'élévation, chacune UNE fois",
    not doubles_b and "b = 300" in sea and "h = 300" in spa, str(doubles_b))

# ================================================================
print("\n=== 7. Note (2 pages) ===")
Rn = moteur.compute(dict(id_projet="Halle A", id_rep="P3/S9"))
pdf = note.generer_note(Rn, {"nom_projet": "Halle A", "partie": "P3/S9", "date": "02/11/2026",
                             "indice": "0", "visserie": "1 rondelle + 1 écrou"})
doc = pymupdf.open(stream=pdf, filetype="pdf")
chk("deux pages A4 paysage, sans débordement, échelle et disposition exposées",
    doc.page_count == 2 and not note.derniers_avertissements
    and note.derniere_echelle is not None and note.derniere_disposition is not None,
    str(note.derniers_avertissements))
t1 = doc[0].get_text(); s1 = t1.replace(" ", "").replace("\n", "")
n1 = " ".join(t1.split())
chk("page 1 : bandeau, en-tête compact 3 lignes, hypothèses, tableaux, conclusion",
    "VÉRIFIÉ" in s1 and "Cornières 2 × L100x100x10 S355" in t1 and "VEd 200,0 kN" in t1
    and "DOUBLES CORNIÈRES" in t1.upper() and "ASSEMBLAGE VÉRIFIÉ à l'ELU" in t1)
chk("page 1 : le cas d'appui est dans l'en-tête ET dans les hypothèses",
    "appui : aile du poteau" in n1
    and "appui sur l'aile, attache dans le plan de l'âme du poteau" in n1.lower(), n1[:400])
chk("page 1 : le gauge p3 et le moment apporté au POTEAU sont imprimés (ils ne vivent nulle "
    "part ailleurs)", "gauge p3 108,6" in n1.lower() and "au poteau ved·e" in n1.lower(),
    n1[:600])
chk("page 1 : γM0 ANB écrit dans les hypothèses", "γM0 1,00 (ANB)" in n1)
chk("page 1 : la table des CORNIÈRES est imprimée (une ligne par aile)",
    "CORNIÈRES" in t1.upper() and "Aile A – pression diamétrale" in n1
    and "Aile B – pression diamétrale" in n1)
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau.notation import NOTATIONS  # noqa: E402
_lu = " ".join(t1.split())
chk("pied de la note : les lignes « Références » et « Notations » tiennent sans être coupées",
    "…" not in _lu.split("Notations :")[-1]
    and "…" not in _lu.split("Références :")[-1].split("Notations")[0]
    and NOTATIONS.split(" : ", 1)[1][:40] in _lu, _lu[-220:])
chk("HYPOTHÈSES : ≤ 3 lignes de ≤ 110 caractères (note_kit.empaqueter)",
    1 <= len(note.lignes_hypotheses(Rn)) <= 3
    and all(len(l) <= 110 for l in note.lignes_hypotheses(Rn)),
    str([len(l) for l in note.lignes_hypotheses(Rn)]))
t2p = doc[1].get_text(); s2 = t2p.replace(" ", "").replace("\n", "")
n2 = " ".join(t2p.split())
chk("page 2 : PLAN DE PRINCIPE, trois vues titrées, échelle affichée",
    "PLANDEPRINCIPE" in s2 and all(x in s2 for x in ("ÉLÉVATION", "VUEENPLAN", "COUPEA-A"))
    and "Échelle1:" in s2)
# les libellés du cartouche sont tracés avec un interlettrage (track) : le PDF les rend
# « P O T E A U » — on les cherche donc sans espaces
chk("page 2 : cartouche POTEAU · POUTRE, appui, CORNIÈRES, visserie et Ø des trous",
    "POTEAU·POUTRE" in s2 and "appui : aile du poteau" in n2 and "2 × L100x100x10 S355" in n2
    and "par boulon : 1 rondelle + 1 écrou" in n2 and "trous Ø22" in n2,
    f"{'POTEAU·POUTRE' in s2} {'appui : aile du poteau' in n2}")
Rb = moteur.compute(dict(appui=entrees.APPUI_AME, corn_u="L90x90x9"))
pdfb = note.generer_note(Rb, {"date": "02/11/2026", "indice": "0"})
docb = pymupdf.open(stream=pdfb, filetype="pdf")
tb1 = " ".join(docb[0].get_text().split()); tb2 = " ".join(docb[1].get_text().split())
chk("cas (b) : 2 pages, le cisaillement local est imprimé, le cartouche dit « âme du poteau »",
    docb.page_count == 2 and not note.derniers_avertissements
    and "Cisaillement local brut" in tb1 and "appui : âme du poteau" in tb2,
    f"{docb.page_count} {note.derniers_avertissements}")
chk("cas (b) : l'appui dans l'âme, le cisaillement local ET la flexion hors plan non vérifiée "
    "sont dits dans les hypothèses",
    "appui dans l'âme : cisaillement local un seul côté" in tb1.lower()
    and "flexion hors plan" in tb1.lower(), tb1[:500])
with open("tests/ref_dcc.json", encoding="utf-8") as fh:
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
at.session_state["dcc_ui_onglet"] = "Benchmark"; run(at)
tb = md(at)
chk("onglet Benchmark : les cinq cas, les calculs manuels, statut VALIDÉ",
    all(x in tb for x in ("BM1", "BM1'", "BM2", "BM3", "BM4", "VAL-A", "VAL-B", "VAL-C",
                          "Benchmark du module : VALIDÉ")))
B = benchmark.run_bench()
chk("un contrôle par cas de benchmark : chaque ligne de chaque cas est à son écart annoncé",
    all(r["st"] in ("OK", "Écart expliqué") for b in B["bench"] for r in b["rows"])
    and all(r["st"] == "OK" for c in B["valid"] for r in c["rows"]),
    str([(b["id"], r["lab"][:40], r["st"]) for b in B["bench"] for r in b["rows"]
         if r["st"] not in ("OK", "Écart expliqué")][:3]))
chk("BM1 (héritage) : les seize valeurs publiées, pire écart 0,76 %",
    len(B["bench"][0]["rows"]) == 16 and B["worst"] <= 0.01, str(B["worst"]))
chk("BM1' : l'ancre publiée du tronçon en T (6 valeurs, modes 1/2/3)",
    len(B["bench"][1]["rows"]) == 6)
chk("sept « vérifications sans exemple publié consulté » sont nommées", len(benchmark.NOCOVER) == 7)
at.button(key="dcc_btn_bm_0").click(); run(at)
chk("« Charger ces données » : l'exemple publié dans l'outil (VEd = 450, hc = 430)",
    float(at.session_state["dcc_V_Ed"]) == 450.0 and float(at.session_state["dcc_LC_u"]) == 430.0)
at.button(key="dcc_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts", float(at.session_state["dcc_LC_u"]) == 290.0)

# ================================================================
print("\n=== 9. Prédim, exports, méthode ===")
at.segmented_control(key="dcc_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
chk("bandeau prédimensionnement", "Mode prédimensionnement" in md(at))
at.session_state["dcc_ui_onglet"] = "Prédim"; run(at)
chk("le détail de la solution nomme le trusquinage et le gauge",
    "Trusquinage gA / gauge p3" in md(at))
boutons = [b for b in at.button if b.key and b.key.startswith("dcc_btn_sol_")]
chk("solutions proposées cliquables", len(boutons) >= 1)
retenue = [b for b in boutons if b.type == "primary"]
(retenue[0] if retenue else boutons[0]).click(); run(at)
chk("« Appliquer » : VÉRIFICATION + cornière, boulons et gA recopiés",
    at.session_state["dcc_mode_calc"] == "VÉRIFICATION"
    and float(at.session_state["dcc_gA_u"]) > 0)
at.button(key="dcc_btn_reset").click(); run(at)
at.button(key="dcc_btn_pdf").click(); run(at)
chk("« 📄 PDF » depuis l'application : 2 pages",
    bool(at.session_state.get("dcc_pdf_bytes"))
    and pymupdf.open(stream=at.session_state["dcc_pdf_bytes"], filetype="pdf").page_count == 2)
at.session_state["dcc_ui_onglet"] = "Note"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Note : export texte avec statut, appui, gauge et références",
    "DOUBLES CORNIÈRES" in code and "VÉRIFIÉ" in code and "appui : aile du poteau" in code
    and "gauge p3 = tw,S + 2·gA" in code and "Tableau 3.3 : conforme" in code)
from acier.assemblages.poutre_poteau.doubles_cornieres_poteau import texte as texte_dcc  # noqa: E402
chk("le rédacteur saisi figure dans l'export texte (amendement 3)",
    "Rédacteur : J. Dupont" in texte_dcc.construire_texte(moteur.compute(dict(id_red="J. Dupont")))
    and "Rédacteur" not in texte_dcc.construire_texte(moteur.compute(dict())))
chk("export texte, cas (b) : la largeur droite entre congés et l'aile A maximale",
    "largeur droite entre congés" in texte_dcc.construire_texte(Rb)
    and "aile A maximale" in texte_dcc.construire_texte(Rb))
at.session_state["dcc_ui_onglet"] = "Méthode"; run(at)
tm = md(at)
chk("onglet Méthode : ce que l'outil vérifie, les deux cas d'appui, les quatre conditions "
    "géométriques, l'ANB, les limites, les notations",
    all(x in tm for x in ("Ce que l'outil vérifie", "Les deux cas d'appui",
                          "Les quatre conditions GÉOMÉTRIQUES du poteau",
                          "Annexe nationale belge", "Données de bibliothèque",
                          "Limites du modèle", "Notations")))
chk("onglet Méthode : les NEUF limites du §8.4 et les points tranchés P-1, P-3 à P-9",
    all(l.split(" ")[0] in tm for l in methode.limites(True))
    and all(x in tm for x in ("**P-1**", "**P-3**", "**P-4**", "**P-5**", "**P-6**", "**P-7**",
                              "**P-8**", "**P-9**"))
    and len(methode.limites(True)) == 9 and len(methode.limites(False)) == 8)
chk("onglet Méthode : l'écart assumé (la ligne bvT_S ajoutée au §8.2) est écrit",
    "bvT_S" in tm and "non conservatif" in tm)

# ================================================================
print("\n=== 9 bis. Le contrat du PROJET (registre, description, rapport mixte) ===")
from acier import registre, projet as modele, rapport_projet as rp  # noqa: E402

_t = registre.trouver("doubles_cornieres_poteau")
chk("le registre déclare le module DISPONIBLE, avec son court (≤ 18) et sa vignette",
    _t is not None and _t[0].id == "poutre_poteau" and _t[1].disponible
    and _t[1].court == "Cornières/poteau" and len(_t[1].court) <= 18
    and 'role="img"' in _t[1].vignette)
_m = registre.charger(_t[1])
chk("le module expose le contrat du projet (CLES, defaults, compute, charger_payload, "
    "MODULE_VERSION, ecran_saisie.valeur_widget, note.generer_note)",
    len(_m.CLES) == 88 and callable(_m.compute) and callable(_m.charger_payload)
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


modele.ajouter(_p, "doubles_cornieres", _defauts("doubles_cornieres"))
modele.ajouter(_p, "doubles_cornieres_poteau", _defauts("doubles_cornieres_poteau", id_rep="P3/S9"))
modele.ajouter(_p, "doubles_cornieres_poteau",
               _defauts("doubles_cornieres_poteau", appui=entrees.APPUI_AME,
                        corn_u="L90x90x9", id_rep="P4/S9"))
_el = rp.notes(_p, {"nom_projet": "Halle A", "partie": "Hall", "date": "02/11/2026", "indice": "0"})
chk("rapport de projet MIXTE (un asm_ + deux dcc_) : trois notes de 2 pages, états lisibles",
    [n for (_e, _i, _pdf, n) in _el] == [2, 2, 2]
    and all(i["etat"] in ("ok", "att") for (_e, i, _pdf, _n) in _el),
    str([(i["etat"], round(i["eta"], 3)) for (_e, i, _pdf, _n) in _el]))
_doc = pymupdf.open(stream=rp.generer_rapport(_p, {"nom_projet": "Halle A", "partie": "Hall",
                                                   "date": "02/11/2026", "indice": "0"}),
                    filetype="pdf")
_txt = " ".join(" ".join(_doc[i].get_text().split()) for i in range(2))
chk("le récapitulatif nomme le type et les deux repères du module neuf",
    _doc.page_count == 2 + 6 and "P3/S9" in _txt and "P4/S9" in _txt
    and "Doubles cornières d'âme sur poteau" in _txt, f"{_doc.page_count} pages")

# ================================================================
print("\n=== 10. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict()
          if k.startswith(("b1_", "dal", "pre", "asm_", "fpl_", "fpc_", "pab_", "ppb_"))
          and k != "asm_courant"]
chk("aucune clé d'un autre écran créée par le module", not fuites, str(fuites[:5]))
at.button(key="dcc_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

# ================================================================
print("\n=== 11. Tableau 3.3 NON CONFORME ===")
at33 = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=300)
at33.session_state["page"] = "Assemblages métalliques"
run(at33)
at33.button(key="asm_btn_open_doubles_cornieres_poteau").click(); run(at33)
at33.session_state["dcc_e1S_u"] = 20.0
run(at33)
e33 = [x for x in at33.expander if str(x.label).startswith("Pinces et entraxes (Tab. 3.3)")]
chk("e1 = 20 mm → « Pinces et entraxes (Tab. 3.3) : NON CONFORME », expander ouvert d'office",
    bool(e33) and str(e33[0].label) == "Pinces et entraxes (Tab. 3.3) : NON CONFORME"
    and e33[0].proto.expanded, str([str(x.label) for x in e33]))
at33.button(key="dcc_btn_dist_S_e1c").click(); run(at33)
chk("« Localiser » de la ligne NON OK sélectionne son alerte (dist_S_e1c) ; sa rangée dit « Masquer »",
    at33.session_state["dcc_ui_alerte"] == "dist_S_e1c"
    and at33.button(key="dcc_btn_al_dist_S_e1c").label == "Masquer",
    str(at33.session_state["dcc_ui_alerte"]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

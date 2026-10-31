# -*- coding: utf-8 -*-
"""Le KIT partagé des écrans d'assemblage (charte ergonomique du 31/10/2026,
docs/assemblages/ERGONOMIE.md §13) et son premier gabarit, le plat d'âme.

Chaque garantie rougit si on la retire :
  1. RIEN NE SE PERD (plat d'âme) : CHAMPS ∪ GEOMETRIE_DESSIN ∪ {ui_niveau,
     ui_composant} ⊂ carte ∪ avancés ∪ cartouche ∪ cotes ∪ fenêtres ∪ tête du
     dessin ; ONGLETS et clés d'export inchangés.
  2. AIDES : aide(k) is None hors AIDES ; toute aide ≤ 160 caractères.
  3. LIBELLÉS : ≤ 22 partout, ≤ 16 dans une rangée de 4, ≤ 20 dans une
     rangée de 3 ou une continuation (LIBELLE et LIBELLE_MODE).
  4. OPTIONS : affichées ≤ 28 ; AFFICHAGE ⊂ options brutes.
  5. CSS : css_kit porte les quatre paliers (1351 / 1350 / 1180 / 640).
  6. GRILLE : aucune ligne d'objet > 4 widgets (les continuations ↳ et les
     rangées de table ne sont pas des lignes d'objet).
  7. ÉTANCHÉITÉ : la carte du plat d'âme ne crée que des clés fpl_.
  8. HYPOTHÈSES : note_kit.empaqueter → ≤ 3 lignes ≤ 110, aucun segment
     perdu, formes courtes du dernier au premier, ValueError sinon ; sur le
     défaut ET les 12 cas figés du plat d'âme.
  9. « ? » visibles dans la carte au cas par défaut : plat d'âme ≤ 4.
 10. BARRE ET TITRE : registre.court (≤ 18) ; libelle_entree le compose ;
     bloc_statut_ligne sans nowrap sur le statut ; tableau_html(defilant=).

Lancement : python3 tests/test_assemblages_kit.py
"""
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


# ================================================================
print("=== 0. Le kit s'importe ===")
try:
    from acier.assemblages import carte, note_kit, ui_commun  # noqa: E402
    from acier import projet_ui, registre  # noqa: E402
    from acier.assemblages.poutre_poutre.plat_ame import (ecran_saisie, ecran_resultats, interface,  # noqa: E402
                                                          note, moteur, entrees)
    IMPORT_OK = True
except Exception as e:  # noqa: BLE001
    IMPORT_OK = False
    chk("import du kit et du plat d'âme", False, f"{type(e).__name__}: {e}")

if IMPORT_OK:
    D = ecran_saisie.D
    chk("Description du plat d'âme : préfixe fpl_", D.PREFIXE == "fpl_")

    # ================================================================
    print("\n=== 1. Rien ne se perd (plat d'âme) ===")
    u0 = entrees.defaults()
    R0 = moteur.compute(dict(u0))
    cles_carte = carte.cles_carte(D)
    cles_fenetres = {c["key"] for g in ecran_resultats._panneaux(R0, u0).values() for c in g["champs"]}
    cles_dessin = set(ecran_resultats.CLES_TETE_DESSIN)
    attendues = set(entrees.CHAMPS) | set(ecran_saisie.GEOMETRIE_DESSIN) | {"ui_niveau", "ui_composant"}
    atteignables = cles_carte | set(ecran_saisie.GEOMETRIE_DESSIN) | cles_fenetres | cles_dessin
    manquantes = sorted(attendues - atteignables)
    chk("chaque clé de CHAMPS ∪ GEOMETRIE_DESSIN ∪ {ui_niveau, ui_composant} a une destination",
        not manquantes, str(manquantes))
    chk("la visserie reste atteignable (cartouche)", "visserie" in cles_carte)
    chk("ONGLETS inchangés", ecran_resultats.ONGLETS == ("Vérifications", "Prédim", "Note", "Benchmark", "Méthode"))
    chk("la géométrie ne se règle jamais dans la carte (GEOMETRIE_DESSIN ∩ carte = ∅)",
        not (cles_carte & set(ecran_saisie.GEOMETRIE_DESSIN)))
    # fenêtres : a_w ne reste que dans Soudure ; la fenêtre Poutre secondaire reçoit gh, Δz, dc, c
    g = ecran_resultats._panneaux(R0, u0)
    chk("fenêtre Plat sans a_w (doublon retiré) ; Soudure le garde",
        "a_w" not in {c["key"] for c in g["plate"]["champs"]} and "a_w" in {c["key"] for c in g["weld"]["champs"]})
    chk("fenêtre Poutre secondaire : gh, Δz, dc,sup, dc,inf, c",
        {"g_h", "d_top", "d_nt", "d_nb", "l_n"} <= {c["key"] for c in g["beamS"]["champs"]})
    chk("les fenêtres lisent LIBELLE (une forme par clé)",
        all(c["lab"] == D.LIBELLE[c["key"]] for grp in g.values() for c in grp["champs"]),
        str([(c["key"], c["lab"]) for grp in g.values() for c in grp["champs"] if c["lab"] != D.LIBELLE.get(c["key"])][:4]))

    # ================================================================
    print("\n=== 2. Aides ===")
    sans_aide = [k for k in entrees.CHAMPS if k not in D.AIDES and carte.aide(D, k) is not None]
    chk("aide(k) is None hors AIDES", not sans_aide, str(sans_aide))
    longues = [k for k, v in D.AIDES.items() if len(v) > 160]
    chk("toute aide ≤ 160 caractères", not longues, str(longues))
    chk("aucun « ? » sur Nuance, Profilé, Boulon, Classe, dimensions, efforts, identification",
        not ({"nu_P", "nu_S", "nu_pl", "prof_P", "prof_S", "boulon_u", "classe", "hP_u", "hS_u",
              "V_Ed", "N_Ed", "M_Ed", "id_rep", "id_red"} & set(D.AIDES)))

    # ================================================================
    print("\n=== 3. Libellés ===")
    trop = [(k, v) for k, v in D.LIBELLE.items() if len(v) > 22]
    chk("LIBELLE ≤ 22 partout", not trop, str(trop))
    r4 = [(k, D.LIBELLE[k]) for genre, cles in carte.rangees(D) if genre == "objet" and len(cles) == 4
          for k in cles if len(D.LIBELLE[k]) > 16]
    chk("≤ 16 dans une rangée de 4", not r4, str(r4))
    r3 = [(k, D.LIBELLE[k]) for genre, cles in carte.rangees(D) if (genre == "suite" or len(cles) == 3)
          for k in cles if len(D.LIBELLE[k]) > 20]
    chk("≤ 20 dans une rangée de 3 ou une continuation", not r3, str(r3))
    def _mot(v):
        # un mot (≥ 3 lettres latines en tête) prend la capitale ; un symbole (h, tw, ks, fy, e1, μ, γM0) non
        t = re.match(r"[A-Za-zÀ-ÿ]+", v)
        return bool(t) and len(t.group(0)) >= 3
    minus = [v for v in D.LIBELLE.values() if _mot(v) and v[:1] != v[:1].upper()]
    chk("capitale initiale des mots (les symboles h, tw, ks, γM0 gardent leur casse)", not minus, str(minus))
    chk("unités tirées de UNITE (VEd (kN), hp (mm), MEd (kNm))",
        D.LIBELLE["V_Ed"] == "VEd (kN)" and D.LIBELLE["hp_u"] == "hp (mm)" and D.LIBELLE["M_Ed"] == "MEd (kNm)")

    # ================================================================
    print("\n=== 4. Options ===")
    longues = []
    for k, f in entrees.CHAMPS.items():
        for o in (f.get("o") or []):
            if len(carte.affiche(D, k, o)) > 28:
                longues.append((k, o))
    chk("options affichées ≤ 28", not longues, str(longues))
    chk("AFFICHAGE ⊂ options brutes",
        all(set(v) <= set(entrees.CHAMPS[k]["o"]) for k, v in D.AFFICHAGE.items()))
    chk("mode affiché « Vérification / Prédimensionnement », valeurs brutes inchangées",
        carte.affiche(D, "mode_calc", entrees.MODE_VERIF) == "Vérification"
        and carte.affiche(D, "mode_calc", entrees.MODE_PREDIM) == "Prédimensionnement"
        and entrees.MODE_VERIF == "VÉRIFICATION")

    # ================================================================
    print("\n=== 5. CSS du kit ===")
    css = carte.css_kit("fpl_")
    chk("les quatre paliers", all(x in css for x in ("(max-width:1350px)", "(max-width:1180px)",
                                                    "(max-width:640px)", "(min-width:1351px)")))
    chk("sur nos propres clés (lg_, bt_, tr_, th_)", all("st-key-fpl_" + x in css for x in ("lg_", "bt_", "tr_", "th_")))

    # ================================================================
    print("\n=== 6. Grille ===")
    trop = [cles for genre, cles in carte.rangees(D) if genre == "objet" and len(cles) > 4]
    chk("aucune ligne d'objet > 4 widgets", not trop, str(trop))
    chk("une continuation peut n'avoir qu'une case (↳ PLAT)",
        any(genre == "suite" and cles == ["nu_pl"] for genre, cles in carte.rangees(D)))

    # ================================================================
    print("\n=== 7. Étanchéité des préfixes ===")
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from acier.assemblages.poutre_poutre.plat_ame import ecran_saisie, interface\n"
        "interface._init_etat()\n"
        "u = interface.lire_entrees()\n"
        "ecran_saisie.carte(u, set())\n", default_timeout=120)
    at.run()
    chk("la carte seule s'exécute", not at.exception, str(at.exception[0].message)[:200] if at.exception else "")
    cles_w = {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input)
              + list(at.checkbox) + list(at.segmented_control) if w.key}
    chk("tous les widgets de la carte sont préfixés fpl_", cles_w and all(k.startswith("fpl_") for k in cles_w),
        str(sorted(k for k in cles_w if not k.startswith("fpl_"))))
    chk("aucune clé de session hors fpl_", all(k.startswith("fpl_") for k in at.session_state.to_dict()),
        str([k for k in at.session_state.to_dict() if not k.startswith("fpl_")][:5]))
    chk("le mode est un segmented_control sur la clé fpl_mode_calc",
        any(w.key == "fpl_mode_calc" for w in at.segmented_control))

    # ================================================================
    print("\n=== 8. HYPOTHÈSES : note_kit.empaqueter ===")
    emp = note_kit.empaqueter
    chk("un segment par ligne quand ils ne tiennent pas ensemble",
        emp(["a" * 60, "b" * 60], n_max=3, larg=110) == ["A" + "a" * 59 + ".", "B" + "b" * 59 + "."])
    chk("deux segments sur une ligne quand ils tiennent (« ; » et point final)",
        emp(["un", "deux"], n_max=3, larg=110) == ["Un ; deux."])
    chk("forme courte prise du dernier au premier",
        emp([("x" * 50, "x"), ("y" * 70, "y")], n_max=1, larg=110) == ["X" + "x" * 49 + " ; y."])
    def _raises(*args, **kw):
        try:
            emp(*args, **kw)
        except ValueError:
            return True
        return False
    chk("segment jamais coupé : ValueError si rien ne tient", _raises(["z" * 120], n_max=3, larg=110))
    chk("ValueError aussi quand les formes courtes ne suffisent pas",
        _raises([("a" * 100, "a" * 60), "b" * 60, "c" * 60, "d" * 60], n_max=3, larg=110))
    chk("le grec n'est pas mis en capitale", emp(["γM0 1,00"], n_max=1, larg=110) == ["γM0 1,00."])
    lignes = note.lignes_hypotheses(R0)
    print("      défaut :", lignes)
    chk("plat d'âme, défaut : ≤ 3 lignes ≤ 110", 1 <= len(lignes) <= 3 and all(len(l) <= 110 for l in lignes),
        str([len(l) for l in lignes]))
    chk("textes exacts de la charte (défaut)",
        lignes == ["Articulé : rotule à la face de l'âme porteuse (MSB P5 §3.2) ; z = 50,0 mm ; le groupe reprend VEd·z + |MEd|.",
                   "Plat court (z ≤ tp/0,15 : pas de déversement) ; grugeage : flexion de la section réduite (§4.2.4).",
                   "Cordons pleine hauteur, un par face (§4.5.3.3) ; γM0 1,00 · γM1 1,10 (ANB) · γM2 1,25."], str(lignes))
    with open("tests/ref_plat_ame.json", encoding="utf-8") as fh:
        REF = json.load(fh)
    pertes, trop = [], []
    for nom, cas in REF.items():
        R = moteur.compute(cas["inp"])
        segs = note.segments_hypotheses(R)
        ls = note.lignes_hypotheses(R)
        txt = " ".join(ls)
        if len(ls) > 3 or any(len(l) > 110 for l in ls):
            trop.append((nom, [len(l) for l in ls]))
        for s in segs:
            formes = [s] if isinstance(s, str) else list(s)
            if not any(f[1:] in txt for f in formes):
                pertes.append((nom, formes[-1][:30]))
    chk("12 cas figés : ≤ 3 lignes ≤ 110", not trop, str(trop))
    chk("12 cas figés : aucun segment perdu", not pertes, str(pertes))
    R_long = moteur.compute(dict(u0, hp_u=360, bp_u=160, tp_u=10, V_Ed=350, n2_u="2", cat="B"))
    ls = note.lignes_hypotheses(R_long)
    chk("variante plat long + cat B : « Plat long » et « γM3 » écrits",
        any(l.startswith("Plat long (z > tp/0,15 : déversement, courbe BS 5950-1)") for l in ls)
        and any("γM3 1,25" in l for l in ls), str(ls))

    # ================================================================
    print("\n=== 9. « ? » visibles dans la carte au défaut ===")
    vis = carte.aides_visibles(D, u0)
    chk("plat d'âme : ≤ 4 « ? » (mode, trou, catégorie, EFFORTS ELU)",
        len(vis) <= 4 and set(vis) == {"mode_calc", "trou", "cat", "EFFORTS ELU"}, str(vis))

    # ================================================================
    print("\n=== 10. Barre, titre, briques ===")
    courts = {a.id: a.court for f in registre.FAMILLES for a in f.assemblages}
    chk("registre.court : les quatre, ≤ 18 caractères",
        courts == {"doubles_cornieres": "Doubles cornières", "plat_ame": "Plat d'âme",
                   "platine_about": "Platine d'about", "platine_beton": "Pied de poteau"}
        and all(len(v) <= 18 for v in courts.values()), str(courts))
    chk("libelle_entree compose « A1 · Plat d'âme » avec court ; titre inchangé pour le sommaire",
        projet_ui.libelle_entree({"type": "plat_ame", "values": {"id_rep": "A1"}}) == "A1 · Plat d'âme"
        and projet_ui.titre_de("plat_ame") == "Plat d'âme soudé (fin plate)")
    at2 = AppTest.from_string(
        "from acier.assemblages import ui_commun\n"
        "ui_commun.bloc_statut_ligne('ok', 'PLATINE VÉRIFIÉE (calcul manuel) – sous réserve des alertes', '— 79,3 %', [('T', 0.5, True)])\n"
        "ui_commun.titre_bloc('EFFORTS ELU', help='aide du bloc')\n"
        "import streamlit as st\n"
        "st.markdown(ui_commun.tableau_html(['a', 'b'], [[1, 2]], defilant=True), unsafe_allow_html=True)\n",
        default_timeout=60)
    at2.run()
    t2 = md(at2)
    chk("bloc_statut_ligne : le statut n'est plus en nowrap", not at2.exception and "white-space:nowrap;'>PLATINE" not in t2
        and "PLATINE VÉRIFIÉE" in t2)
    chk("titre_bloc(help=) : un « ? » au niveau du titre", any(m.help == "aide du bloc" for m in at2.markdown if hasattr(m, "help")))
    chk("tableau_html(defilant=True) : enveloppe overflow-x:auto", "overflow-x:auto" in t2)
    chk("un seul _esc (le plat d'âme importe celui de ui_commun)", not hasattr(ecran_resultats, "_esc")
        or ecran_resultats._esc is ui_commun._esc)
    chk("code mort retiré de ui_commun (open_bloc_left_right, close_bloc, bloc_verification)",
        not any(hasattr(ui_commun, f) for f in ("open_bloc_left_right", "close_bloc", "bloc_verification")))
    chk("ui_commun.legende_natures, expander_perimetre, tableau_33_replie existent",
        all(callable(getattr(ui_commun, f, None)) for f in ("legende_natures", "expander_perimetre", "tableau_33_replie")))
    chk("projet_ui.en_tete et barre_outils existent",
        callable(getattr(projet_ui, "en_tete", None)) and callable(getattr(projet_ui, "barre_outils", None)))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

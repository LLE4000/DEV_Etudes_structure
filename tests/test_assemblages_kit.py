# -*- coding: utf-8 -*-
"""Le KIT partagé des écrans d'assemblage (charte ergonomique du 31/10/2026,
docs/assemblages/ERGONOMIE.md §13) et son premier gabarit, le plat d'âme.

Chaque garantie rougit si on la retire :
  1. RIEN NE SE PERD (plat d'âme) : CHAMPS ∪ {ui_niveau, ui_composant} ⊂
     carte ∪ avancés ∪ cartouche ∪ fenêtres ∪ tête du dessin ; ONGLETS et clés
     d'export inchangés. 1 bis (cinq écrans) : chaque clé de GEOMETRIE_DESSIN
     visible est une cote du vrai dessin ou dans une fenêtre, et dans le
     panneau Cotes rendu quand le dessin est fixe.
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
    # la géométrie n'est plus « atteignable » d'elle-même (contrôle tautologique jusqu'au 01/11/2026) : ses
    # destinations réelles sont vérifiées en 1 bis (cote du dessin, fenêtre, panneau Cotes)
    atteignables = cles_carte | cles_fenetres | cles_dessin | set(ecran_saisie.GEOMETRIE_DESSIN) & cles_fenetres
    manquantes = sorted(attendues - set(ecran_saisie.GEOMETRIE_DESSIN) - atteignables)
    chk("chaque clé de CHAMPS ∪ {ui_niveau, ui_composant} hors géométrie a une destination (carte, fenêtre, tête du "
        "dessin)", not manquantes, str(manquantes))
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
    # Consolidation du 01/11/2026, point 5 : l'ancien contrôle était TAUTOLOGIQUE pour la géométrie
    # (« atteignables » contenait GEOMETRIE_DESSIN lui-même). Chaque clé de GEOMETRIE_DESSIN visible doit être
    # émise comme cote par le VRAI dessin (Feuille.dim key=…) ou portée par une fenêtre d'objet — dessin
    # interactif, défaut et variantes — et figurer dans le panneau Cotes RENDU quand le dessin est fixe (Vue
    # simple et Édition : la fenêtre ne s'ouvre plus), sur les cinq écrans.
    print("\n=== 1 bis. Rien ne se perd : la géométrie (cinq écrans, dessin interactif et dessin fixe) ===")
    from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import cotes_modifiables
    from acier.assemblages.poutre_poutre.doubles_cornieres import (entrees as E_dc, moteur as M_dc, schemas as S_dc,
                                                                   ecran_saisie as ES_dc, ecran_resultats as ER_dc)
    from acier.assemblages.poutre_poutre.platine_about import (entrees as E_pb, moteur as M_pb, schemas as S_pb,
                                                               ecran_saisie as ES_pb, ecran_resultats as ER_pb)
    from acier.assemblages.pied_poteau.platine_beton import (entrees as E_pp, schemas as S_pp, ecran_saisie as ES_pp,
                                                             ecran_resultats as ER_pp)
    from acier.assemblages.pied_poteau.platine_beton.manuel import schemas as S_pm, ecran as EM_pp
    from acier.assemblages.pied_poteau.platine_beton.calcul import compute as C_pp
    from acier.assemblages.poutre_poutre.plat_ame import schemas as S_fpl

    def _manque_interactif(u, Rc, G, visible, sch, vues, panneaux):
        opt = sch.options_ecran(Rc, 1, None)
        opt.editables = True
        cotes = {k for v in vues for k, _s, _i in cotes_modifiables(getattr(sch, v)(Rc, opt))}
        fen = {c["key"] for g_ in panneaux(Rc, u).values() for c in g_["champs"]}
        return sorted(k for k in G if visible(u, k) and k not in cotes | fen)

    ECRANS = [  # nom, préfixe, entrées, calcul, schémas, vues, fenêtres, géométrie, module de l'écran, variantes
        ("plat d'âme", "fpl_", entrees, moteur.compute, S_fpl, ("elevation", "plan"), ecran_resultats._panneaux,
         ecran_saisie.GEOMETRIE_DESSIN, "acier.assemblages.poutre_poutre.plat_ame",
         [{}, dict(n2_u="2"), dict(d_nt=0, d_nb=0, l_n=0)]),
        ("platine d'about", "pab_", E_pb, M_pb.compute, S_pb, ("elevation", "plan"), ER_pb._panneaux, ES_pb.GEOMETRIE_DESSIN,
         "acier.assemblages.poutre_poutre.platine_about", [{}, dict(N_Ed=60), dict(d_nt=0, d_nb=0, l_n=0)]),
        ("doubles cornières", "asm_", E_dc, M_dc.compute, S_dc, ("elevation", "plan"), ER_dc._panneaux, ES_dc.GEOMETRIE_DESSIN,
         "acier.assemblages.poutre_poutre.doubles_cornieres",
         [{}, dict(fix_P="Soudée", fix_S="Soudée"), dict(n2S_u="2", n2P_u="2"), dict(fix_P="Soudée", n2S_u="2")]),
        ("pied complet", "ppb_", E_pp, C_pp, S_pp, ("coupe_aa", "plan"), ER_pp._panneaux, ES_pp.GEOMETRIE_DESSIN,
         "acier.assemblages.pied_poteau.platine_beton",
         [dict(mode=E_pp.MODE_COMPLET), dict(mode=E_pp.MODE_COMPLET, t_g=0), dict(mode=E_pp.MODE_COMPLET, typ_a=E_pp.TYPE_CROSSE)]),
        ("calcul manuel", "ppb_", E_pp, C_pp, S_pm, ("coupe_aa", "plan"), EM_pp._panneaux, ES_pp.GEOMETRIE_DESSIN,
         "acier.assemblages.pied_poteau.platine_beton", [{}, dict(tiges=[]), dict(t_g=0)]),
    ]
    for nom, p, E, calc, sch, vues, panneaux, G, mod, variantes in ECRANS:
        manques = []
        for var in variantes:
            u = dict(E.defaults(), **var)
            m = _manque_interactif(u, calc(dict(u)), G, E.champ_visible, sch, vues, panneaux)
            if m:
                manques.append((var, m))
        chk(f"{nom} : chaque clé de GEOMETRIE_DESSIN visible est une cote du dessin ou dans une fenêtre "
            f"(défaut + {len(variantes) - 1} variante(s))", not manques, str(manques))
        fixes = []
        for niveau in (0, 1):
            etat = dict(variantes[0], ui_composant=False, ui_niveau=niveau)
            src = ("import streamlit as st\nfrom " + mod + " import interface\n"
                   + "".join(f"st.session_state.setdefault({p + k!r}, {v!r})\n" for k, v in etat.items())
                   + "interface.show()\n")
            atg = AppTest.from_string(src, default_timeout=180)
            atg.run()
            panneau = {w.key[len(p) + 5:] for w in atg.number_input if w.key and w.key.startswith(p + "cote_")}
            u = dict(E.defaults(), **variantes[0])
            manque = sorted(k for k in G if E.champ_visible(u, k) and k not in panneau)
            if atg.exception or manque:
                fixes.append((niveau, manque, [str(x.message)[:80] for x in atg.exception]))
        chk(f"{nom} : dessin fixe — le panneau Cotes rendu porte toute la géométrie visible, en Vue simple "
            "(point 15 : il était vide) comme en Édition", not fixes, str(fixes))

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
    # recherche lexicographique (01/11/2026) : un segment raccourci rend aux suivants leur forme longue
    chk("un segment raccourci rend aux suivants leur forme longue (lexicographique, pas un compteur)",
        emp([("a" * 70, "a" * 40), ("b" * 60, "b" * 50)], n_max=1, larg=110) == ["A" + "a" * 39 + " ; " + "b" * 60 + "."])
    chk("une_fois : un jeu sans doublon par ligne passe avant",
        emp([("x MSB P5 §1" + "x" * 40, "y" * 20), "z MSB P5 §2"], n_max=1, larg=110, une_fois=note_kit.MSB)
        == ["Y" + "y" * 19 + " ; z MSB P5 §2."])
    chk("une_fois : à défaut, le renvoi reste (jamais retiré)",
        emp(["a MSB P5 §1", "b MSB P5 §2"], n_max=1, larg=110, une_fois=note_kit.MSB) == ["A MSB P5 §1 ; b MSB P5 §2."])
    chk("coefficients : trois formes quand γM2 = γM2,net, deux sinon (une source pour les trois modules)",
        note_kit.coefficients("γM0 1,00 · γM2 1,25", 1.25, 1.25)
        == ("γM0 1,00 · γM2 1,25 (boulons, soudures, sections nettes)", "γM0 1,00 · γM2 1,25 (boulons, soudures, nettes)",
            "γM0 1,00 · γM2 1,25")
        and note_kit.coefficients("γM0 1,00 (ANB) · γM2 1,25", 1.25, 1.2, tete_net="γM0 1,00 · γM2 1,25")[1]
        == "γM0 1,00 · γM2 1,25 · γM2,net 1,20")
    lignes = note.lignes_hypotheses(R0)
    print("      défaut :", lignes)
    chk("plat d'âme, défaut : ≤ 3 lignes ≤ 110", 1 <= len(lignes) <= 3 and all(len(l) <= 110 for l in lignes),
        str([len(l) for l in lignes]))
    chk("textes exacts de la charte (défaut)",
        lignes == ["Articulé : rotule à la face de l'âme porteuse (MSB P5 §3.2) ; z = 50,0 mm ; le groupe reprend VEd·z + |MEd|.",
                   "Plat court (z ≤ tp/0,15 : pas de déversement) ; grugeage : flexion de la section réduite "
                   "(MSB P5 §4.2.4).",
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
    # point 10 (01/11/2026) : un § du guide porte toujours « MSB P5 » (un § nu est de l'EN 1993-1-8), une fois par ligne
    msb_nu = re.compile(r"(?<!MSB P5 )§(4\.2|3\.2|2\.)")
    toutes = [l for cas in REF.values() for l in note.lignes_hypotheses(moteur.compute(cas["inp"]))]
    chk("12 cas figés : aucun renvoi au guide sans « MSB P5 », au plus une fois par ligne",
        not [l for l in toutes if msb_nu.search(l) or l.count("MSB P5") > 1],
        str([l for l in toutes if msb_nu.search(l) or l.count("MSB P5") > 1][:2]))
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
    chk("registre.court : les sept (quatre disponibles + les trois attaches poutre–poteau "
        "« à venir »), ≤ 18 caractères",
        courts == {"doubles_cornieres": "Doubles cornières", "plat_ame": "Plat d'âme",
                   "platine_about": "Platine d'about", "platine_beton": "Pied de poteau",
                   "plat_ame_poteau": "Plat d'âme/poteau", "doubles_cornieres_poteau": "Cornières/poteau",
                   "platine_about_poteau": "Platine/poteau"}
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

if IMPORT_OK:
    # ================================================================
    print("\n=== 11. Consolidation du 01/11/2026 : une source par brique ===")
    css = carte.css_kit("asm_")
    chk("CSS du kit : colonne du dessin figée, tables serrées (tab-serre : 3px 4px, en-têtes repliés ; tab-dense : "
        "police réduite), panneau Cotes 4 / 3 / 2, case ≤ 187 px, continuation alignée, rangées d'alerte",
        "position: sticky" in css and "st-key-asm_col_dessin" in css
        and ".tab-serre th, .tab-serre td {padding:3px 4px !important;}" in css
        and ".tab-serre th {white-space:normal !important;}" in css and ".tab-dense table {font-size:0.76em" in css
        and "calc(25% - 6px)" in css and "calc(33.33% - 6px)" in css and 'st-key-asm_cotes' in css
        and "max-width:187px" in css and "padding-left:84px" in css and "margin-left:-84px" in css
        and 'st-key-asm_al_' in css and carte.LARG_CASE_MAX == 187)
    at3 = AppTest.from_string(
        "import streamlit as st\nfrom acier.assemblages import carte\ncarte.injecter_css('fpl_', '<style>.x{a:b}</style>')\n",
        default_timeout=60)
    at3.run()
    chk("injecter_css : le CSS du kit et celui d'un module en UN seul <style>",
        len(at3.markdown) == 1 and str(at3.markdown[0].value).count("<style>") == 1 and ".x{a:b}" in str(at3.markdown[0].value))
    th = ui_commun.tableau_html(["Vérification", "Ed", "η", "T (kNm)", "Réf."], [[1, 2, 3, 4, 5]])
    chk("en-têtes : capitales écrites, symboles et unités intacts (« η » n'est plus « Η »), plus de text-transform ; "
        "≥ 5 colonnes : défilant d'office",
        ">VÉRIFICATION</th>" in th and ">Ed</th>" in th and ">η</th>" in th and ">T (kNm)</th>" in th
        and ">RÉF.</th>" in th and "text-transform" not in th and th.startswith("<div style='overflow-x:auto;'>")
        and not ui_commun.tableau_html(["a", "b"], [[1, 2]]).startswith("<div"))
    chk("tableau_html(serre=1 / 2) : classes tab-serre / tab-serre tab-dense",
        ui_commun.tableau_html(["a"], [[1]], serre=1).startswith("<div class='tab-serre'>")
        and ui_commun.tableau_html(["a"], [[1]], serre=2).startswith("<div class='tab-serre tab-dense'>"))
    # relecture des captures finales (01/11/2026) : « MSB / P5 §3.2.3 » et « Tab. / 3.4 » coupés dans la colonne
    # Réf. ; une rangée d'en-tête vide au-dessus des tables « libellé | valeur » du calcul à la main
    chk("table : un sigle de référence ne se coupe pas (« MSB P5 », « Tab. 3.4 », « Ch. 8 », « hyp. outil ») et le "
        "« · » ferme sa ligne (seul sur une ligne au téléphone sinon) — la coupure avant « § » reste permise ; une "
        "table sans en-tête n'en dessine pas",
        "Tab.&nbsp;3.4&nbsp;· MSB&nbsp;P5 §4.2.1.1" in ui_commun.tableau_html(["Réf."], [["Tab. 3.4 · MSB P5 §4.2.1.1"]])
        and "hyp.&nbsp;outil&nbsp;· P358 Ch.&nbsp;8" in ui_commun.tableau_html(["Réf."], [["hyp. outil · P358 Ch. 8"]])
        and "<thead>" not in ui_commun.tableau_html(["", ""], [["a", "b"]])
        and "<thead>" in ui_commun.tableau_html(["", "Valeur"], [["a", "b"]]))
    _case_ls = ('[class*="st-key-asm_ls_"] > [data-testid="stElementContainer"]:not([class*="st-key-asm_btn_"]) '
                '{flex:1 1 120px !important;min-width:120px !important;max-width:150px !important;')
    _case_lg = ('[class*="st-key-asm_lg_"] > [data-testid="stElementContainer"]:not([class*="st-key-asm_btn_"]) '
                '{flex-basis:205px')
    chk("CSS du kit : sous 1181 px une continuation de trois cases tient sur une rangée (120 à 150 px : 3 × 150 + 16 "
        "ne passaient pas dans 440 px à 1024) ; deux tables duo_ l'une sous l'autre sous 1351 px ; le bouton d'une "
        "ligne d'action garde sa largeur",
        _case_ls in css and css.index("max-width:1180px") < css.index(_case_ls) < css.index("max-width:640px")
        and _case_lg in css and 'st-key-asm_duo_"] [data-testid="stColumn"] {flex:1 1 100% !important;' in css
        and css.index("max-width:1350px") < css.index("st-key-asm_duo_") < css.index("max-width:1180px"))
    from acier.assemblages.pied_poteau.platine_beton import notation as NP_
    import inspect as _insp
    chk("pied : la clé de rangée du moteur dite en clair (« y-:-200 » → « y− (y = -200 mm) »), une source pour "
        "l'écran (Vérifications, Répartition) et la note",
        NP_.rangee_lisible("y-:-200") == "y− (y = -200 mm)" and NP_.rangee_lisible("x+:150") == "x+ (x = 150 mm)"
        and NP_.rangee_lisible("—") == "—" and _insp.getsource(ER_pp).count("rangee_lisible(") == 3
        and 'T["cle"] +' not in _insp.getsource(ER_pp))
    at4 = AppTest.from_string(
        "from acier.assemblages import ui_commun\n"
        "ui_commun.expander_perimetre(['a', 'b'], ['c'])\n"
        "ui_commun.expander_perimetre(['a', 'b', 'c', 'd'], [], titre_non='Non vérifiés ici')\n"
        "ui_commun.expander_perimetre(['a'], [], titre_non='Non vérifiés ici')\n", default_timeout=60)
    at4.run()
    chk("périmètre : le titre dit le mot du contenu (« non revérifiés » / « non vérifiés », singulier compris)",
        [str(x.label) for x in at4.expander] == ["Périmètre de la vérification — 2 non revérifiés · 1 sans objet",
                                                 "Périmètre de la vérification — 4 non vérifiés · 0 sans objet",
                                                 "Périmètre de la vérification — 1 non vérifié · 0 sans objet"],
        str([str(x.label) for x in at4.expander]))
    chk("titre de bloc : espacement serré pour un mot ≥ 11 lettres (RAIDISSEURS, APPLICATION), sinon 0,07 em",
        ui_commun.espacement("RAIDISSEURS") == ".05em" and ui_commun.espacement("🔴 APPLICATION") == ".05em"
        and ui_commun.espacement("EFFORTS ELU") == ".07em" and ui_commun.espacement("COEFF. γ") == ".07em")
    # hors projet : Projet ET Date ; le cartouche lit id_date hors projet (la date de session le masquait)
    at5 = AppTest.from_string(
        "import streamlit as st\nfrom acier import projet_ui\nfrom acier.assemblages import carte\n"
        "st.session_state['date'] = '01/11/2026'\n"
        "st.session_state['r1'] = projet_ui.infos_cartouche('fpl_', {'id_date': '05/11/2026', 'id_projet': 'P'}, 'v')\n"
        "st.session_state['h1'] = carte.HORS_PROJET['id_date']({})\n"
        "st.session_state['nom_projet'] = 'Halle'\n"
        "st.session_state['r2'] = projet_ui.infos_cartouche('fpl_', {'id_date': '05/11/2026'}, 'v')\n"
        "st.session_state['h2'] = carte.HORS_PROJET['id_date']({})\n", default_timeout=60)
    at5.run()
    chk("id_date tranché (points 3 et 22) : hors projet, Projet · Date visibles et la note imprime id_date ; "
        "dans un projet, la date du projet",
        at5.session_state["h1"] is True and at5.session_state["r1"]["date"] == "05/11/2026"
        and at5.session_state["r1"]["nom_projet"] == "P" and at5.session_state["h2"] is False
        and at5.session_state["r2"]["date"] == "01/11/2026" and at5.session_state["r2"]["nom_projet"] == "Halle")
    from acier.assemblages.poutre_poutre.doubles_cornieres import ecran_saisie as ES_dc2
    chk("hors de la carte, le groupe est dit : « Rangées n1 (groupe S) » (toasts, pastilles) ; la fenêtre « Groupe S » "
        "garde le libellé de la carte",
        carte.libelle_long(ES_dc2.D, "n1S_u") == "Rangées n1 (groupe S)" and carte.libelle_long(ES_dc2.D, "n2P_u")
        == "Files n2 (groupe P)" and carte.champ_fenetre(ES_dc2.D, "n1S_u", {})["lab"] == "Rangées n1"
        and carte.libelle_long(D, "V_Ed") == "VEd (kN)")
    chk("une seule brique par fonction : plus de _kit_bloc, _kit_sous_titre, _entete_table, CSS_MODULE, _un_style, "
        "_CSS_MATRICE, _mini_champ, _traiter_clic dans les modules",
        not any(hasattr(m, f) for m in (ES_dc2, ecran_resultats) for f in ("_kit_bloc", "_kit_sous_titre", "_aide_suite",
                                                                            "_mini_champ", "_traiter_clic"))
        and not any(hasattr(EM_pp, f) for f in ("_entete_table", "_rangee", "_suite", "_ligne_titre", "CSS_MODULE"))
        and not hasattr(ES_pp, "_kit_bloc") and not hasattr(ER_dc, "_CSS_MATRICE"))
    # garde-fou des dimensions vides (points 8 et 19) : les cinq écrans
    GARDE = [("fpl_", "acier.assemblages.poutre_poutre.plat_ame", {"prof_P": "Personnalisé"}, "h, b, tw, tf"),
             ("pab_", "acier.assemblages.poutre_poutre.platine_about", {"prof_S": "Personnalisé"}, "h, b, tw, tf"),
             ("asm_", "acier.assemblages.poutre_poutre.doubles_cornieres", {"corn_u": "Personnalisé"},
              "Grande aile, Petite aile, Épaisseur"),
             ("ppb_", "acier.assemblages.pied_poteau.platine_beton", {"mode": E_pp.MODE_COMPLET, "forme": E_pp.FORME_RHS},
              "h, b, t paroi"),
             ("ppb_", "acier.assemblages.pied_poteau.platine_beton", {"mode": E_pp.MODE_COMPLET, "prof_c": "Personnalisé"},
              "h, b, tw, tf")]
    rates = []
    for p, mod, etat, dims in GARDE:
        src = ("import streamlit as st\nfrom " + mod + " import interface\n"
               + "".join(f"st.session_state.setdefault({p + k!r}, {v!r})\n" for k, v in etat.items())
               + "interface.show()\n")
        atd = AppTest.from_string(src, default_timeout=180)
        atd.run()
        t = md(atd)
        ok = (not atd.exception and not atd.error and len(atd.get("component_instance")) == 2
              and "Données incomplètes" in t and "Dimensions à saisir dans la carte : " + dims in t
              and "⚠️ À saisir : " + dims + " — vérification suspendue." in t and "VÉRIFIÉ" not in t
              and "dès que les dimensions sont saisies" in t)
        if not ok:
            rates.append((mod.split(".")[-1], list(etat.values())[-1], [str(x.message)[:60] for x in atd.exception],
                          [str(x.value)[:60] for x in atd.error], len(atd.get("component_instance"))))
    chk("dimensions vides (Personnalisé, tube RHS) : message court sous la ligne, dessin conservé, vérification "
        "suspendue — plus d'exception ni de taux absurde (cinq cas)", not rates, str(rates))
    # alertes : une forme pour les cinq écrans (rangée al_, bouton de largeur fixe, repli des remarques)
    import inspect
    sources = {n: inspect.getsource(m) for n, m in (("plat", ecran_resultats), ("pab", ER_pb), ("dc", ER_dc),
                                                     ("ppb", ER_pp))}
    chk("alertes : les cinq écrans appellent ui_commun.alertes (rangée alignée, repli « n remarques »)",
        all("ui_commun.alertes(PREFIXE" in x for x in sources.values()) and "st.columns([7, 1]" not in "".join(sources.values()))

if IMPORT_OK:
    # Notations (point 4) : une source pour la ligne du pied de la note ET le paragraphe de Méthode ; la ligne
    # tient sans « … » (celle du calcul manuel mesurait 801 pt pour 798 : coupée — défaut trouvé le 01/11/2026)
    import importlib
    from ndc_pdf import fonts as _fonts
    from ndc_pdf.styles import Encre
    from reportlab.lib.pagesizes import A4, landscape
    _fonts.register_all()
    largeur = landscape(A4)[0] - 2 * note.M
    mesures, ecarts = [], []
    for mod in ("poutre_poutre.doubles_cornieres", "poutre_poutre.plat_ame", "poutre_poutre.platine_about",
                "pied_poteau.platine_beton", "pied_poteau.platine_beton.manuel"):
        n_ = importlib.import_module("acier.assemblages." + mod + ".note")
        m_ = importlib.import_module("acier.assemblages." + mod + ".methode")
        w_ = _fonts.string_width(n_.NOTATIONS_COURTES, Encre().f_b, 5.6)
        mesures.append(round(w_))
        if w_ > largeur or n_.NOTATIONS_COURTES.split(" : ", 1)[1] not in m_.METHODE or "#### Notations" not in m_.METHODE:
            ecarts.append((mod, round(w_)))
    chk(f"Notations : la ligne de la note tient (≤ {largeur:.0f} pt à 5,6 pt : {mesures}) et l'onglet Méthode la "
        "reprend mot pour mot (cinq écrans)", not ecarts, str(ecarts))
    chk("les cotes réduites à leur symbole y sont définies (zp, c, dc, gh, Δz, p3, e2,b, ℓh, hef, dh, tg, a)",
        all(x in importlib.import_module("acier.assemblages.poutre_poutre.doubles_cornieres.notation").NOTATIONS
            for x in ("hc, zc", "c, dc", "gh, Δz", "gA, p3", "e2,b", "ℓh"))
        and all(x in importlib.import_module("acier.assemblages.pied_poteau.platine_beton.notation").NOTATIONS_MANUEL
                for x in ("hef, dh", "tg", "bc, lc, hc")))
    # « répartition élastique » : phrase interdite des notes depuis le 21/09/2026 — la référence du calcul manuel
    # l'imprimait (point 23) ; la VUE l'abrège, le moteur garde sa référence
    import pymupdf
    from acier.assemblages.pied_poteau.platine_beton.manuel import note as NM_
    pdf_m = NM_.generer_note(C_pp(E_pp.defaults()), {"nom_projet": "", "partie": "", "date": "01/11/2026",
                                                     "indice": "0", "visserie": "x"})
    tm = " ".join(" ".join(pg.get_text() for pg in pymupdf.open(stream=pdf_m, filetype="pdf")).split())
    from acier.assemblages.pied_poteau.platine_beton.manuel import moteur as MM_
    chk("note du calcul manuel sans « répartition élastique » (« … EC2 §6.7 (répartition) ») ; le moteur garde sa "
        "référence", "répartition élastique" not in tm.lower() and "§6.7 (répartition)" in tm
        and "répartition élastique" in MM_.compute(E_pp.defaults()).ck["bet"].ref
        if "bet" in MM_.compute(E_pp.defaults()).ck else "répartition élastique" not in tm.lower(),
        tm[tm.find("§6.7") - 40: tm.find("§6.7") + 40])
    # relecture des notes finales (01/11/2026) : la 1re ligne BOULONS du cartouche du calcul manuel mesurait 273 pt
    # pour 168 — coupée d'un « … », elle perdait l'ancrage et hef ; la note complète imprimait « rangée y-:-200 »
    pg2 = " ".join(pymupdf.open(stream=pdf_m, filetype="pdf")[1].get_text().split())
    chk("cartouche du calcul manuel : ancrage et hef en 1re ligne, visserie puis « perçages cotés depuis les bords » "
        "en 2e, rien de coupé", "trous Ø26 · tête Ø40 · hef 300" in pg2
        and "par boulon : x · perçages cotés depuis les bords" in pg2 and "…" not in pg2, pg2[pg2.find("BOULONS"):][:200])
    from acier.assemblages.pied_poteau.platine_beton import note as NC_
    pdf_c = NC_.generer_note(C_pp(dict(E_pp.defaults(), mode=E_pp.MODE_COMPLET)),
                             {"nom_projet": "", "partie": "", "date": "01/11/2026", "indice": "0", "visserie": "x"})
    brut_c = "\n".join(pg.get_text() for pg in pymupdf.open(stream=pdf_c, filetype="pdf"))
    tc = " ".join(brut_c.split())
    orphelines = [l for l in brut_c.splitlines() if l.strip() in ("mm", "kN", "kNm", "kNm/m", "MPa")]
    chk("note de la vérification complète : aucune unité seule en tête de ligne (« tT = 8,6 / mm » avant la "
        "relecture du 01/11/2026)", not orphelines, str(orphelines))
    chk("note de la vérification complète : la rangée tendue dite en clair (« rangée y− (y = -200 mm) »), plus la "
        "clé du moteur", "rangée y− (y = -200 mm) : M = T·m" in tc and "y-:-200" not in tc,
        tc[tc.find("Tronçon en T"):][:120])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

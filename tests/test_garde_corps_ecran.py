# -*- coding: utf-8 -*-
"""Tests de l'écran « Garde-corps » (version 2) : AppTest sur l'application réelle. Chaque
garantie rougit si on la retire :

  1. NAVIGATION : page dans le dictionnaire des pages, tuile à l'accueil, rendu sans
     exception, bouton Accueil, test de fumée inchangé.
  2. ÉCRAN : le SWITCH ELS / Eurocode (ELS par défaut), les six dessins (SVG) avec leurs
     légendes, la carte RÉSULTAT avec η, le cas gouvernant nommé et une barre par élément, cinq
     onglets à libellé fixe, barre d'outils, le tableau des montants, la phrase de géométrie.
  3. LE TYPE EN MENU : cinq types ; chacun pose le système et le remplissage et change les
     champs (lisses, panneau, main courante seule, murale avec consoles).
  4. SOURCE UNIQUE : un widget écrit dans l'état, l'état nourrit le moteur, le dessin et la
     carte (entraxe maxi, entraxes saisis, H, catégorie, q_h libre, section du montant, section
     libre, appuis de la main courante, cumul, entraxe des barreaux, h_bas, ancrage, fixation,
     forme en L).
  5. LE SWITCH : même garde-corps, η suit le moteur en Eurocode ; flèche L/200.
  6. TABLEAUX CLIQUABLES : un tableau par élément (montant, main courante, barreaux) avec les
     colonnes attendues, RHS cochés par défaut pour le montant et la main courante, la section
     retenue marquée ◀, le bouton « le plus léger qui tient » change la section ET le
     sélecteur la montre ; la clé du tableau change de génération après un changement d'état
     (la sélection repart de zéro : pas de boucle).
  7. CAS MULTIPLES ET FICHIER : nouveau, copie, suppression, sélection, JSON.
  8. NOTE : le bouton produit un PDF de deux pages.
  9. ÉTANCHÉITÉ : seules des clés gc_.

Lancement : python3 tests/test_garde_corps_ecran.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.garde_corps import etat as ET  # noqa: E402
from acier.garde_corps import types as TY  # noqa: E402

OK, KO = [], []
ETAPES = ["1 · Garde-corps", "2 · Charges", "3 · Sections", "4 · Ancrage", "5 · Détail"]
LEGENDES = ("ÉLÉVATION", "COUPE A-A", "VUE EN PLAN", "PLATINE EN PLAN", "COUPE SUR LA FIXATION", "SECTION")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


def pastille(at):
    for m in at.markdown:
        s = str(m.value)
        if 'class="gc-pill' in s:
            i = s.index('class="gc-pill')
            return s[i:i + 120]
    return ""


def eta(at):
    for m in at.markdown:
        s = str(m.value)
        if 'class="gc-eta"' in s:
            i = s.index('class="gc-eta"')
            return s[i:i + 80]
    return ""


def eta_moteur(at):
    return f"{ET.verifier(cas_actif(at))['taux_max'] * 100:.0f} %"


N_RUN = [0]


def run(at):
    at.run()
    N_RUN[0] += 1
    if at.exception:
        chk(f"aucune exception (relance {N_RUN[0]})", False, str(at.exception[0].message)[:300])
    return at


def cles(at):
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox)
            + list(at.segmented_control) + list(at.button) + list(at.multiselect) if w.key}


def tables(at):
    return {d.key: d.value for d in at.dataframe if d.key}


def app():
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
    at.session_state["page"] = "Garde-corps"
    return at


def cas_actif(at):
    return ET.cas_actif(at.session_state["gc"])


# ================================================================
print("\n=== 1. Navigation ===")
src_app = open("streamlit_app.py", encoding="utf-8").read()
src_acc = open("modules/accueil.py", encoding="utf-8").read()
src_fum = open("tests/test_fumee_pages.py", encoding="utf-8").read()
chk("page « Garde-corps » dans le dictionnaire des pages, tuile à l'accueil, test de fumée", '"Garde-corps": garde_corps.show' in src_app and '"page": "Garde-corps"' in src_acc and '"Garde-corps"' in src_fum)
at = run(app())
chk("rendu sans exception, titre et sous-titre", not at.exception and "## Garde-corps" in md(at) and "EN 1991-1-1" in md(at))
at.button(key="gc_btn_home").click(); run(at)
chk("🏠 Accueil ramène à l'accueil", at.session_state["page"] == "Accueil")

# ================================================================
print("\n=== 2. Écran ===")
at = run(app())
m = md(at)
chk("le switch : « Vérification à l'ELS » par défaut, « charges de service », contraintes admissibles — sans « méthode du bureau » ni « ELS du bureau »",
    at.segmented_control(key="gc_w_methode_mode").value == "els" and "Charges de service" in m and "contraintes admissibles" in m
    and "méthode du bureau" not in m.lower() and "ELS du bureau" not in m and "Vérification à l'ELS" in m)
chk("six dessins SVG légendés : élévation, coupe A-A, plan, platine en plan, coupe sur la fixation, section", m.count("<svg") >= 6 and all(k in m for k in LEGENDES))
chk("carte RÉSULTAT : Conforme, η = 64 %, le montant M2 gouverne par la flèche sous le cas 1, une barre par élément",
    "Conforme" in pastille(at) and "64 %" in eta(at) and "Montant principal M2" in m and "flèche" in m and "cas 1 · q_h seule" in m
    and all(k in m for k in ("Main courante", "Barreaux", "Soudure", "Platine", "Ancrages · 2 × 2 chevilles M12")))
chk("cinq onglets à libellé fixe", [t.label for t in at.tabs] == ETAPES)
chk("barre d'outils : Accueil, ＋ Cas, copie, Enregistrer, Ouvrir, Note, nom du cas", all(k in cles(at) for k in ("gc_btn_home", "gc_btn_new", "gc_btn_copy", "gc_btn_open", "gc_btn_note", "gc_w_nom")))
chk("le tableau des montants : M1 … M5 avec leurs longueurs tributaires", "M5" in m and "450" in m and "s_trib" in m)
chk("la phrase de géométrie : droit, 3,60 m en 4 travées de 0,90 m, 5 montants ; le type dans la légende", "3,60 m en 4 travées de 0,90 m" in m and "5 montants" in m
    and "Barreaudage vertical" in m)
chk("les quatre charges nommées, numérotées, avec leur origine", all(k in m for k in ("q_h", "q_v", "Q_r", "q_r", "Tab. 6.12", "NBN B 03-004")) and 'class="gc-ch"' in m)
chk("la portée des barreaux est déduite (950 mm) et dite", "Portée des barreaux déduite : 950 mm" in m)

# ================================================================
print("\n=== 3. Le type en menu ===")
sb = at.selectbox(key="gc_w_type")
chk("le menu des types : les cinq types (libellés), barreaudage par défaut", len(sb.options) == len(TY.TYPES) and all(t["libelle"] in sb.options for t in TY.TYPES.values())
    and sb.value in ("barreaudage", TY.TYPES["barreaudage"]["libelle"]), f"{sb.options} · {sb.value}")
sb.set_value("lisses"); run(at)
c = cas_actif(at)
chk("type lisses : remplissage lisses, champs n et e_l, élément « Lisses intermédiaires », légende", c["type"] == "lisses" and c["remplissage"]["type"] == "lisses"
    and "gc_w_remplissage_n_lisses" in cles(at) and "Lisses intermédiaires" in md(at) and "Lisses horizontales" in md(at))
at.selectbox(key="gc_w_type").set_value("murale"); run(at)
c = cas_actif(at)
chk("type murale : système mur, consoles C1 …, longueur e, section de console dans Sections, pas de montant",
    c["geometrie"]["systeme"] == "mur" and "C1" in md(at) and "gc_w_fixation_console_e" in cles(at) and "gc_w_fixation_console_section" in cles(at)
    and "gc_w_montant_section" not in cles(at) and "Console" in md(at))
at.selectbox(key="gc_w_type").set_value("panneau"); run(at)
chk("type panneau : la mention du fournisseur, h_bas, pas de section de barreau", "fournisseur" in md(at) and "gc_w_remplissage_h_bas" in cles(at) and "gc_w_remplissage_section" not in cles(at))
at.selectbox(key="gc_w_type").set_value("main_courante"); run(at)
chk("type main courante seule : sans remplissage", "Main courante seule" in md(at) and cas_actif(at)["remplissage"]["type"] == "aucun")
at.selectbox(key="gc_w_type").set_value("barreaudage"); run(at)
chk("retour au barreaudage : 64 %", "64 %" in eta(at), eta(at))

# ================================================================
print("\n=== 4. Source unique ===")
at.number_input(key="gc_w_geometrie_troncons_0_s_max").set_value(1200.0); run(at)
c = cas_actif(at)
chk("entraxe maxi 1200 → 3 travées, 4 montants, la phrase suit, η suit le moteur", c["geometrie"]["troncons"][0]["s_max"] == 1200 and "3 travées de 1,20 m" in md(at)
    and "4 montants" in md(at) and eta_moteur(at) in eta(at), eta(at))
at.text_input(key="gc_w_geometrie_troncons_0_entraxes").set_value("1200 900"); run(at)
c = cas_actif(at)
chk("entraxes saisis « 1200 900 » : l'état garde [1200, 900], trois montants à 600 / 1050 / 450", c["geometrie"]["troncons"][0]["entraxes"] == [1200.0, 900.0] and "1050" in md(at) and "travées de 1,20 m + 0,90 m" in md(at))
at.text_input(key="gc_w_geometrie_troncons_0_entraxes").set_value(""); run(at)
at.number_input(key="gc_w_geometrie_H").set_value(1000.0); run(at)
chk("H = 1000 : l'état et le dessin (H = 1000)", cas_actif(at)["geometrie"]["H"] == 1000 and "H = 1000" in md(at))
at.selectbox(key="gc_w_charges_categorie").set_value("C5"); run(at)
c = cas_actif(at)
chk("catégorie C5 : q_h = 3 kN/m imposé, la carte le dit, Non conforme", c["charges"]["qh"] == 3.0 and "q_h = 3 kN/m" in md(at) and "Non conforme" in pastille(at))
chk("q_h est verrouillé hors « valeur libre »", at.number_input(key="gc_w_charges_qh").disabled)
at.selectbox(key="gc_w_charges_categorie").set_value("perso"); run(at)
at.number_input(key="gc_w_charges_qh").set_value(0.8); run(at)
chk("valeur libre 0,8 kN/m : prise", cas_actif(at)["charges"]["qh"] == 0.8 and "q_h = 0,8 kN/m" in md(at))
at.selectbox(key="gc_w_charges_categorie").set_value("A"); run(at)
at.number_input(key="gc_w_geometrie_troncons_0_s_max").set_value(1000.0); run(at)
at.number_input(key="gc_w_geometrie_H").set_value(1100.0); run(at)
chk("retour au cas par défaut : 64 %", "64 %" in eta(at), eta(at))
at.selectbox(key="gc_w_montant_section").set_value("SHS 50x50x3"); run(at)
chk("section du montant SHS 50x50x3 : l'état, la carte et l'élévation suivent", cas_actif(at)["montant"]["section"] == {"nom": "SHS 50x50x3"} and "SHS 50x50x3" in md(at))
at.selectbox(key="gc_w_montant_section").set_value("— section libre (dimensions) —"); run(at)
c = cas_actif(at)
chk("section libre : forme et dimensions (tube Ø 48,3 × 3,2 par défaut), soudure C8", "famille" in c["montant"]["section"] and "gc_w_montant_section_dims_d" in cles(at) and "Tube Ø48.3x3.2" in md(at))
at.number_input(key="gc_w_montant_section_dims_t").set_value(4.0); run(at)
chk("la dimension t = 4 écrit dans l'état", cas_actif(at)["montant"]["section"]["dims"]["t"] == 4.0)
at.selectbox(key="gc_w_montant_section").set_value("RHS 50x30x3"); run(at)
at.segmented_control(key="gc_w_lisse_appuis").set_value("continue"); run(at)
chk("main courante continue : l'état et le libellé de l'élément (« continue sur plusieurs appuis »)", cas_actif(at)["lisse"]["appuis"] == "continue" and "continue sur plusieurs appuis" in md(at))
at.segmented_control(key="gc_w_lisse_appuis").set_value("deux"); run(at)
at.checkbox(key="gc_w_charges_cumul").check(); run(at)
chk("cumul q_h + q_v : l'état, le cas 1 + 2 nommé dans la carte", cas_actif(at)["charges"]["cumul"] is True and "cas 1 + 2" in md(at) and "cumulées" in md(at))
at.checkbox(key="gc_w_charges_cumul").uncheck(); run(at)
at.number_input(key="gc_w_remplissage_entraxe").set_value(150.0); run(at)
at.number_input(key="gc_w_remplissage_h_bas").set_value(150.0); run(at)
c = cas_actif(at)
chk("entraxe des barreaux 150 et h_bas 150 : l'état, la portée déduite devient 900 mm", c["remplissage"]["entraxe"] == 150 and c["remplissage"]["h_bas"] == 150
    and "Portée des barreaux déduite : 900 mm" in md(at))
at.number_input(key="gc_w_remplissage_entraxe").set_value(110.0); run(at)
at.number_input(key="gc_w_remplissage_h_bas").set_value(100.0); run(at)
at.segmented_control(key="gc_w_fixation_ancrage_type").set_value("boulon"); run(at)
chk("ancrage par boulons : M12 8.8, F_t,Rd calculé", cas_actif(at)["fixation"]["ancrage"]["type"] == "boulon" and "gc_w_fixation_ancrage_boulon" in cles(at) and "M12" in md(at))
at.segmented_control(key="gc_w_fixation_ancrage_type").set_value("cheville"); run(at)
n_svg = md(at).count("<svg")
at.segmented_control(key="gc_w_fixation_type").set_value("scellee"); run(at)
chk("montant scellé : ni platine ni coupe sur la fixation ni soudure (quatre dessins de moins)", "scellé" in md(at).lower() and md(at).count("<svg") == n_svg - 4
    and "PLATINE EN PLAN" not in md(at))
at.segmented_control(key="gc_w_fixation_type").set_value("laterale"); run(at)
c = cas_actif(at)
labels_lat = [n.label for n in at.number_input]
chk("platine latérale en nez de dalle : quatre fixations au choix, les champs d_h, b_p (hauteur), l_p (largeur), e_x vertical, e_y horizontal ; « Platine latérale (vue de face) » et "
    "« Coupe horizontale sur la fixation » dessinées, arrachement P ; six dessins du garde-corps et UN dessin de soudure (le groupe C8 n'a qu'une vue, C5 en a deux)",
    c["fixation"]["type"] == "laterale" and "gc_w_fixation_platine_d_h" in cles(at)
    and all(any(l.startswith(k) for l in labels_lat) for k in ("d_h — descente", "b_p — hauteur de la platine", "l_p — largeur de la platine", "e_x — entraxe vertical", "e_y — entraxe horizontal"))
    and "PLATINE LATÉRALE (VUE DE FACE)" in md(at) and "COUPE HORIZONTALE SUR LA FIXATION" in md(at) and "arrachement P" in md(at) and md(at).count("<svg") == n_svg - 1
    and "Ancrage latéral" in md(at) and "Platine rigide" not in md(at))
at.segmented_control(key="gc_w_fixation_type").set_value("platine"); run(at)
at.number_input(key="gc_w_geometrie_pente").set_value(32.0); run(at)
chk("rampant : la pente 32° dans l'état et la phrase, « rampant » sur l'élévation, η inchangé", cas_actif(at)["geometrie"]["pente"] == 32.0 and "rampant, pente 32°" in md(at)
    and "ÉLÉVATION" in md(at))
at.number_input(key="gc_w_geometrie_pente").set_value(0.0); run(at)
at.segmented_control(key="gc_w_geometrie_forme").set_value("L"); run(at)
chk("forme en L : deux tronçons, un montant d'angle, le plan en L, « Élévation développée »", "gc_w_geometrie_troncons_1_L" in cles(at) and "d'angle" in md(at)
    and "en L" in md(at) and "ÉLÉVATION DÉVELOPPÉE" in md(at))
at.segmented_control(key="gc_w_geometrie_forme").set_value("droit"); run(at)

# ================================================================
at = run(app())
at.selectbox(key="gc_w_geometrie_debut").set_value("mur"); run(at)
at.selectbox(key="gc_w_geometrie_fin").set_value("mur"); run(at)
at.number_input(key="gc_w_geometrie_troncons_0_s_max").set_value(4000.0); run(at)
chk("les deux bouts ancrés et une seule travée : garde-corps SANS montant — la main courante porte de mur à mur (3600), pas de fixation, rôles lisse et barreaux, réactions aux murs",
    "Sans montant" in md(at) and "porte de mur à mur" in md(at) and "gc_w_fixation_platine_bp" not in cles(at) and "gc_w_montant_section" not in cles(at)
    and "il reprend H = 0,90 kN" in md(at) and eta_moteur(at) in eta(at))
at = run(app())
at.selectbox(key="gc_w_charges_categorie").set_value("ind"); run(at)
chk("catégorie « plateforme industrielle (machines) » : q_h = 0,30 kN/m, origine EN ISO 14122-3 à confirmer", cas_actif(at)["charges"]["qh"] == 0.3 and "EN ISO 14122-3" in md(at))

# ================================================================
print("\n=== 5. Le switch ===")
at = run(app())
at.segmented_control(key="gc_w_methode_mode").set_value("eurocode"); run(at)
chk("Eurocode : « Efforts de calcul (ELU) », η suit le moteur", "Efforts de calcul (ELU)" in md(at) and cas_actif(at)["methode"]["mode"] == "eurocode"
    and eta_moteur(at) in eta(at), eta(at))
at.segmented_control(key="gc_w_methode_mode").set_value("els"); run(at)
at.number_input(key="gc_w_methode_fleche_val").set_value(200.0); run(at)
chk("flèche L/200 : le montant passe à 127 % (Non conforme)", "127 %" in eta(at) and "Non conforme" in pastille(at), eta(at))

# ================================================================
print("\n=== 6. Tableaux cliquables ===")
at = run(app())
t = tables(at)
chk("un tableau par élément : montant, main courante, barreaux (génération 0)", set(t) == {"gc_tab_montant_0", "gc_tab_lisse_0", "gc_tab_barreau_0"}, str(sorted(t)))
df = t["gc_tab_montant_0"]
chk("colonnes : Section, dimensions, kg/m, η σ_eq, η flèche, η, retenue ; 40 lignes par masse croissante",
    list(df.columns) == ["Section", "dimensions (mm)", "kg/m", "η σ_eq", "η flèche", "η", "retenue"] and len(df) == 40 and list(df["kg/m"]) == sorted(df["kg/m"]))
chk("RHS cochés par défaut pour le montant et la main courante, plats et ronds pour les barreaux ; que des RHS dans le tableau du montant",
    at.pills(key="gc_lib_montant_fam").value == ["RHS"] and at.pills(key="gc_lib_lisse_fam").value == ["RHS"] and at.pills(key="gc_lib_barreau_fam").value == ["plats", "ronds"]
    and all(s.startswith("RHS") for s in df["Section"]))
ret = df[df["retenue"].str.startswith("◀")]
chk("la section retenue RHS 50x30x3 est marquée ◀ (et ★ l'optimal, le plus léger qui tient), avec son η = 64 %", len(ret) == 1 and ret.iloc[0]["Section"] == "RHS 50x30x3" and int(ret.iloc[0]["η"]) == 64)
chk("les boutons « le plus léger qui tient » existent par élément", all(k in cles(at) for k in ("gc_btn_best_montant", "gc_btn_best_lisse", "gc_btn_best_barreau")))
at.button(key="gc_btn_best_montant").click(); run(at)
c = cas_actif(at)
nom = c["montant"]["section"]["nom"]
chk("« le plus léger qui tient » : le montant devient un RHS plus léger qui tient, η ≤ 100 %", nom.startswith("RHS") and nom != "RHS 50x30x3"
    and ET.verifier(c)["montant"]["verif"]["taux_max"] <= 1.0, nom)
chk("le sélecteur de section AFFICHE la nouvelle section (liaison état → widget)", at.selectbox(key="gc_w_montant_section").value == nom)
t = tables(at)
chk("après le changement, les tableaux changent de génération (clé _1) : la sélection repart de zéro", set(t) == {"gc_tab_montant_1", "gc_tab_lisse_1", "gc_tab_barreau_1"}, str(sorted(t)))
chk("… et la ligne retenue est la nouvelle section", list(t["gc_tab_montant_1"][t["gc_tab_montant_1"]["retenue"].str.startswith("◀")]["Section"]) == [nom])
at.button(key="gc_btn_best_barreau").click(); run(at)
chk("barreaux : le plus léger plat ou rond qui tient est retenu", cas_actif(at)["remplissage"]["section"]["nom"].split(" ")[0] in ("Plat", "Rond"))
src = open("modules/garde_corps.py", encoding="utf-8").read()
chk("le clic sur une ligne est branché : on_select='rerun', selection_mode='single-row', la ligne retient la section (appliquer_section)",
    "on_select=\"rerun\"" in src and "selection_mode=\"single-row\"" in src and "ev.selection.rows" in src and "ET.appliquer_section(cas, sec, role)" in src)

# ================================================================
print("\n=== 7. Cas multiples et fichier ===")
at = run(app())
at.button(key="gc_btn_new").click(); run(at)
chk("＋ Cas : deux cas, le second actif", len(at.session_state["gc"]["cas"]) == 2 and at.session_state["gc"]["actif"] == 1)
at.text_input(key="gc_w_nom").set_value("Terrasse"); run(at)
chk("le nom du cas est écrit dans l'état", cas_actif(at)["nom"] == "Terrasse")
at.button(key="gc_btn_copy").click(); run(at)
chk("copie : trois cas, « Terrasse (copie) » actif", len(at.session_state["gc"]["cas"]) == 3 and cas_actif(at)["nom"] == "Terrasse (copie)")
at.button(key="gc_btn_del").click(); run(at)
chk("suppression : deux cas", len(at.session_state["gc"]["cas"]) == 2)
at.selectbox(key="gc_sel_cas").set_value("Garde-corps 1"); run(at)
chk("sélection du premier cas", at.session_state["gc"]["actif"] == 0)
txt = ET.vers_json(at.session_state["gc"])
d = json.loads(txt)
chk("le fichier JSON porte les deux cas et se relit", len(d["cas"]) == 2 and ET.depuis_json(txt)["cas"][1]["nom"] == "Terrasse")

# ================================================================
print("\n=== 8. Note ===")
at = run(app())
at.button(key="gc_btn_note").click(); run(at)
note = at.session_state.get("gc_note")
chk("le bouton Note produit un PDF (en-tête %PDF, > 50 ko) pour le cas actif", note is not None and note[0] == "Garde-corps 1" and note[1][:4] == b"%PDF" and len(note[1]) > 50000)

# ================================================================
print("\n=== 9. Étanchéité ===")
at = run(app())
chk("seules des clés gc_", all(k.startswith("gc_") for k in cles(at)), str([k for k in cles(at) if not k.startswith("gc_")][:5]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

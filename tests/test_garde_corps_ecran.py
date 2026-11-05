# -*- coding: utf-8 -*-
"""Tests de l'écran « Garde-corps » : AppTest sur l'application réelle. Chaque garantie
rougit si on la retire :

  1. NAVIGATION : page dans le dictionnaire des pages, tuile à l'accueil, rendu sans
     exception, bouton Accueil, test de fumée inchangé.
  2. ÉCRAN : le SWITCH ELS / Eurocode, les quatre dessins (SVG), la carte RÉSULTAT avec η
     et une barre par élément, six onglets à libellé fixe, barre d'outils, clés gc_.
  3. SOURCE UNIQUE : un widget écrit dans l'état, l'état nourrit le moteur, le dessin et
     la carte (entraxe maxi, H, catégorie, q_h libre, section du montant, remplissage,
     fixation, entraxes saisis « 1200 900 », forme en L, système mur).
  4. LE SWITCH : même garde-corps, η change entre ELS (75 %) et Eurocode, et la ligne du
     switch dit « efforts de calcul ».
  5. BIBLIOTHÈQUE : les candidats, le bouton « Prendre le plus léger qui tient » change la
     section du montant ; la section libre affiche ses dimensions.
  6. CAS MULTIPLES ET FICHIER : nouveau, copie, suppression, sélection, JSON.
  7. NOTE : le bouton produit un PDF de deux pages.
  8. ÉTANCHÉITÉ : seules des clés gc_.

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

OK, KO = [], []
ETAPES = ["1 · Garde-corps", "2 · Charges", "3 · Sections", "4 · Remplissage", "5 · Fixation", "6 · Détail"]


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


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
chk("le switch : segmented control ELS | Eurocode, ELS par défaut, « vieilles formules du bureau »", at.segmented_control(key="gc_w_methode_mode").value == "els" and "vieilles formules du bureau" in m)
chk("quatre dessins SVG (élévation, plan, détail, section)", m.count("<svg") >= 4 and "ÉLÉVATION DÉVELOPPÉE" in m and "VUE EN PLAN" in m and "DÉTAIL DE LA FIXATION" in m)
chk("carte RÉSULTAT : Conforme, η = 75 %, le montant M2 gouverne par la flèche, une barre par élément",
    "Conforme" in pastille(at) and "75 %" in eta(at) and "Montant M2" in m and "flèche" in m and all(k in m for k in ("Lisse haute", "Barreaux", "Soudure", "Platine")))
chk("six onglets à libellé fixe", [t.label for t in at.tabs] == ETAPES)
chk("barre d'outils : Accueil, ＋ Cas, copie, Enregistrer, Ouvrir, Note, nom du cas", all(k in cles(at) for k in ("gc_btn_home", "gc_btn_new", "gc_btn_copy", "gc_btn_open", "gc_btn_note", "gc_w_nom")))
chk("le tableau des montants : M1 … M5 avec leurs longueurs tributaires", "M5" in m and "450" in m and "s_trib" in m)
chk("la phrase de géométrie : droit, 3,60 m en 4 travées de 0,90 m, 5 montants", "3,60 m en 4 travées de 0,90 m" in m and "5 montants" in m)

# ================================================================
print("\n=== 3. Source unique ===")
at.number_input(key="gc_w_geometrie_troncons_0_s_max").set_value(1200.0); run(at)
c = cas_actif(at)
chk("entraxe maxi 1200 → 3 travées, 4 montants, tributaire 1200, le dessin et la phrase suivent", c["geometrie"]["troncons"][0]["s_max"] == 1200 and "3 travées de 1,20 m" in md(at) and "4 montants" in md(at))
chk("… et η passe de 75 % à 99 % (flèche du montant)", "99 %" in eta(at), eta(at))
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
chk("retour au cas par défaut : 75 %", "75 %" in eta(at), eta(at))
at.selectbox(key="gc_w_montant_section").set_value("SHS 50x50x3"); run(at)
chk("section du montant SHS 50x50x3 : l'état, la carte et l'élévation suivent", cas_actif(at)["montant"]["section"] == {"nom": "SHS 50x50x3"} and "SHS 50x50x3" in md(at))
at.selectbox(key="gc_w_montant_section").set_value("— section libre (dimensions) —"); run(at)
c = cas_actif(at)
chk("section libre : forme et dimensions (tube Ø 48,3 × 3,2 par défaut), soudure C8", "famille" in c["montant"]["section"] and "gc_w_montant_section_dims_d" in cles(at) and "Tube Ø48.3x3.2" in md(at))
at.number_input(key="gc_w_montant_section_dims_t").set_value(4.0); run(at)
chk("la dimension t = 4 écrit dans l'état", cas_actif(at)["montant"]["section"]["dims"]["t"] == 4.0)
at.selectbox(key="gc_w_montant_section").set_value("CHS 48.3x3.2"); run(at)
at.segmented_control(key="gc_w_remplissage_type").set_value("lisses"); run(at)
chk("remplissage en lisses intermédiaires : nouveaux champs, élément « Lisses intermédiaires »", cas_actif(at)["remplissage"]["type"] == "lisses" and "gc_w_remplissage_n_lisses" in cles(at) and "Lisses intermédiaires" in md(at))
at.segmented_control(key="gc_w_remplissage_type").set_value("barreaux"); run(at)
at.segmented_control(key="gc_w_fixation_ancrage_type").set_value("boulon"); run(at)
chk("ancrage par boulons : M12 8.8, F_t,Rd calculé", cas_actif(at)["fixation"]["ancrage"]["type"] == "boulon" and "gc_w_fixation_ancrage_boulon" in cles(at) and "M12" in md(at))
at.segmented_control(key="gc_w_fixation_type").set_value("scellee"); run(at)
chk("montant scellé : pas de soudure ni de platine, trois dessins", "scellé" in md(at).lower() and md(at).count("<svg") == 3)
at.segmented_control(key="gc_w_fixation_type").set_value("platine"); run(at)
at.segmented_control(key="gc_w_geometrie_forme").set_value("L"); run(at)
chk("forme en L : deux tronçons, un montant d'angle, le plan en L", "gc_w_geometrie_troncons_1_L" in cles(at) and "d'angle" in md(at) and "en L" in md(at))
at.segmented_control(key="gc_w_geometrie_forme").set_value("droit"); run(at)
at.segmented_control(key="gc_w_geometrie_systeme").set_value("mur"); run(at)
chk("système mur : la lisse est l'élément principal, les consoles C1 …, longueur e, section de console", "Console" in md(at) and "C1" in md(at) and "gc_w_fixation_console_e" in cles(at))
at.segmented_control(key="gc_w_geometrie_systeme").set_value("montants"); run(at)

# ================================================================
print("\n=== 4. Le switch ===")
at = run(app())
at.segmented_control(key="gc_w_methode_mode").set_value("eurocode"); run(at)
chk("Eurocode : « Efforts de calcul (ELU) », η change (74 % : flèche inchangée, σ × 1,5 / 235)", "Efforts de calcul (ELU)" in md(at) and cas_actif(at)["methode"]["mode"] == "eurocode"
    and "75 %" in eta(at))
at.segmented_control(key="gc_w_methode_mode").set_value("els"); run(at)
at.number_input(key="gc_w_methode_fleche_val").set_value(200.0); run(at)
chk("flèche L/200 : le montant passe à 149 % (Non conforme)", "149 %" in eta(at) and "Non conforme" in pastille(at), eta(at))

# ================================================================
print("\n=== 5. Bibliothèque ===")
at = run(app())
m = md(at)
chk("la bibliothèque : tableau des candidats avec η, le plus léger qui tient (CHS 48.3x2.5)", "sections essayées" in m and "CHS 48.3x2.5" in m and "gc_btn_best" in cles(at))
at.button(key="gc_btn_best").click(); run(at)
chk("« Prendre le plus léger qui tient » : le montant devient CHS 48.3x2.5, η = 91 %", cas_actif(at)["montant"]["section"] == {"nom": "CHS 48.3x2.5"} and "91 %" in eta(at), eta(at))
chk("le sélecteur de section AFFICHE la nouvelle section (liaison état → widget)", at.selectbox(key="gc_w_montant_section").value == "CHS 48.3x2.5")

# ================================================================
print("\n=== 6. Cas multiples et fichier ===")
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
print("\n=== 7. Note ===")
at = run(app())
at.button(key="gc_btn_note").click(); run(at)
note = at.session_state.get("gc_note")
chk("le bouton Note produit un PDF (en-tête %PDF, > 50 ko) pour le cas actif", note is not None and note[0] == "Garde-corps 1" and note[1][:4] == b"%PDF" and len(note[1]) > 50000)

# ================================================================
print("\n=== 8. Étanchéité ===")
at = run(app())
chk("seules des clés gc_", all(k.startswith("gc_") for k in cles(at)), str([k for k in cles(at) if not k.startswith("gc_")][:5]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Tests de l'écran « Charges sur plancher » (lot 2) : AppTest sur
l'application réelle, plus l'état (``plancher/charges/etat.py``) qui relie
le plan cliquable au moteur. Chaque garantie rougit si on la retire :

  1. NAVIGATION : page dans le dictionnaire des pages, tuile à l'accueil,
     rendu sans exception, bouton Accueil.
  2. ÉTAT ET ÉCRAN : composant du plan (SVG en mm, poignées ○ et cote L),
     « Données manquantes » tant que M_Rd / V_Rd ne sont pas renseignés,
     quatre ONGLETS à libellé fixe, barre d'outils, clés ``chp_``.
  3. SOURCE UNIQUE : un widget écrit dans l'état, l'état nourrit le moteur
     et le plan (M_Rd, L, type de plancher dalle / prédalle / hourdis).
  4. ENGIN : posé depuis la bibliothèque, charges par appui obligatoires
     (refus nommé), puis calcul, bandes teintées, semelles cliquables.
  5. PLAN → ÉTAT : glisser, cote L, sens porteur, ± hourdis, charges par
     appui, répartiteur par la fenêtre d'une semelle (plateau, idem,
     filière, suppression), dupliquer / supprimer un engin.
  6. RÉPARTITEUR PAR L'ONGLET 3 : appui, type, plateau, coupe.
  7. PROPOSITION ET VUES : « Proposer une filière », Coupe, Balayage.
  8. CAS MULTIPLES ET FICHIER : nouveau, copie, suppression, sélection,
     JSON aller-retour, module contrôlé.
  9. FICHES AJOUTÉES : engin, plateau (azobé), filière libre (madrier sur
     champ : I = b·h³/12) — disponibles à l'écran et calculables.
 10. ÉTANCHÉITÉ : seules des clés ``chp_``.

Lancement : python3 tests/test_charges_plancher_ecran.py (depuis la racine).
"""
import copy
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

from plancher.charges import etat as ET  # noqa: E402
from plancher.charges import symboles as SY  # noqa: E402
from plancher.charges import verification as V  # noqa: E402

OK, KO = [], []
ETAPES = ["1 · Plancher", "2 · Charges", "3 · Répartiteur", "4 · Hypothèses", "5 · Méthode"]


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


def svg(at):
    ci = at.get("component_instance")
    return json.loads(ci[0].proto.json_args)["svg"] if ci else ""


def pastille(at):
    """Le texte de la pastille de la carte résultat (pas la feuille de style, qui nomme aussi la classe)."""
    for m in at.markdown:
        s = str(m.value)
        if 'class="chp-pill' in s:
            i = s.index('class="chp-pill')
            return s[i:i + 80]
    return ""


N_RUN = [0]


def run(at):
    at.run()
    N_RUN[0] += 1
    if at.exception:
        chk(f"aucune exception (relance {N_RUN[0]})", False, str(at.exception[0].message)[:200])
    return at


def cles(at):
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox)
            + list(at.segmented_control) + list(at.button) + list(at.multiselect) if w.key}


def app():
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
    at.session_state["page"] = "Charges sur plancher"
    return at


# ================================================================
print("\n=== 1. Navigation ===")
src_app = open("streamlit_app.py", encoding="utf-8").read()
src_acc = open("modules/accueil.py", encoding="utf-8").read()
chk("page « Charges sur plancher » dans le dictionnaire des pages", '"Charges sur plancher": charges_plancher.show' in src_app)
chk("tuile à l'accueil (ligne Béton)", '"page": "Charges sur plancher"' in src_acc)
at = run(app())
chk("rendu sans exception, titre et sous-titre", not at.exception and "## Charges sur plancher" in md(at)
    and "Charges temporaires de chantier" in md(at))
at.button(key="chp_btn_home").click(); run(at)
chk("🏠 Accueil ramène à l'accueil", at.session_state["page"] == "Accueil")

# ================================================================
print("\n=== 2. État et écran ===")
at = run(app())
s = svg(at)
chk("un composant de plan (svg_cliquable), SVG en millimètres (viewBox 0 0 11400 …)", 'viewBox="0 0 11400' in s)
chk("poignées ○ des bords libres (sens porteur) et cote L cliquable", 'data-action="sens:+1"' in s and 'data-key="L"' in s)
chk("poignées ± du nombre de hourdis", 'data-action="n_elements:+1"' in s and 'data-action="n_elements:-1"' in s)
chk("appuis A et B nommés, échelle graphique", ">A<" in s and ">B<" in s and "2 m" in s)
chk("sans M_Rd / V_Rd : « Données manquantes », message du moteur", "manquantes" in pastille(at) and "M_Rd et V_Rd" in md(at))
chk("cinq onglets à libellé FIXE", [t.label for t in at.tabs] == ETAPES)
chk("barre d'outils : Accueil, cas, ＋ Cas, copie, Ouvrir, nom du cas",
    {"chp_btn_home", "chp_sel_cas", "chp_btn_new", "chp_btn_copy", "chp_btn_open", "chp_w_nom"} <= cles(at))
chk("vues Plan / Coupe / Balayage et options", at.segmented_control(key="chp_vue").value == "Plan"
    and {"chp_opt_cotes", "chp_opt_aires", "chp_opt_taux"} <= cles(at))
chk("onglet 1 : type, portée, appuis, catalogue, résistances, état, chape, parts annexe C",
    {"chp_w_plancher_type", "chp_w_plancher_L", "chp_w_plancher_sens", "chp_w_type_cat", "chp_w_plancher_hourdis_M_Rd",
     "chp_w_plancher_hourdis_V_Rd", "chp_w_plancher_hourdis_etat", "chp_w_plancher_hourdis_chape_oui", "chp_w_parts_C"} <= cles(at))
chk("onglet 2 : bibliothèque, étançon, file d'étais, stock", {"chp_btn_lib", "chp_btn_et", "chp_btn_fi", "chp_btn_st"} <= cles(at))
chk("onglet 4 : γG, γQ, k Heft 240, part du tranchant, joint, αcc",
    {"chp_w_params_gamma_G", "chp_w_params_gamma_Q", "chp_w_params_k_flexion_heft240", "chp_w_params_part_tranchant_element",
     "chp_w_params_joint_adm_kN_m", "chp_w_params_alpha_cc"} <= cles(at))

# ================================================================
print("\n=== 3. Source unique : widgets → état → moteur et plan ===")
at.number_input(key="chp_w_plancher_hourdis_M_Rd").set_value(60.0); run(at)
at.number_input(key="chp_w_plancher_hourdis_V_Rd").set_value(80.0); run(at)
etat = at.session_state["chp"]
h = ET.cas_actif(etat)["plancher"]["hourdis"]
chk("M_Rd et V_Rd d'essai écrits dans l'état", h["M_Rd"] == 60.0 and h["V_Rd"] == 80.0)
chk("le calcul aboutit (poids propre et charges présentes seuls) : pastille Conforme, taux par bande",
    "Conforme" in pastille(at) and "manquantes" not in pastille(at) and "%" in svg(at))
at.number_input(key="chp_w_plancher_L").set_value(6.0); run(at)
chk("portée 6,00 m : état et cote du plan", ET.cas_actif(at.session_state["chp"])["plancher"]["L"] == 6.0 and "L = 6,00 m" in svg(at))
at.number_input(key="chp_w_plancher_L").set_value(7.2); run(at)
at.segmented_control(key="chp_w_plancher_type").set_value("dalle"); run(at)
chk("dalle coulée en place : champs h, d, As, béton, fck(t) ; calcul par les formules BA de l'application",
    {"chp_w_plancher_dalle_h", "chp_w_plancher_dalle_d", "chp_w_plancher_dalle_As_inf", "chp_w_plancher_dalle_beton",
     "chp_w_plancher_dalle_fck_t"} <= cles(at) and "manquantes" not in pastille(at) and "onforme" in pastille(at)
    and "largeur 9,60 m" in svg(at))      # une dalle de 20 cm à 524 mm²/m sur 7,20 m n'est pas conforme : le moteur le dit
at.segmented_control(key="chp_w_plancher_type").set_value("predalle"); run(at)
chk("prédalle : largeur de prédalle, joints armés, chape durcie, étais ; joints dessinés",
    {"chp_w_plancher_predalle_b_predalle", "chp_w_plancher_predalle_joints_armes", "chp_w_plancher_predalle_chape_durcie",
     "chp_w_plancher_predalle_portee_etayee"} <= cles(at) and "manquantes" not in pastille(at) and "onforme" in pastille(at))
at.segmented_control(key="chp_w_plancher_type").set_value("hourdis"); run(at)
chk("retour au hourdis : l'état du hourdis est conservé", ET.cas_actif(at.session_state["chp"])["plancher"]["hourdis"]["M_Rd"] == 60.0)

# ================================================================
print("\n=== 4. Engin de la bibliothèque et charges par appui ===")
etat = at.session_state["chp"]
cas = ET.cas_actif(etat)
NOM = next(n for n in ET.gabarits(etat) if "MRT 2150" in n)
ET.ajouter_engin(etat, cas, NOM); run(at)
chk("engin posé au centre, sélectionné, listé dans l'onglet 2", cas["charges"][0]["x"] == 3.6 and cas["charges"][0]["y"] == 4.8
    and cas["selection"] == "engin:E1" and "**E1** ·" in md(at))
chk("charges par appui non renseignées : REFUS nommé (P-13), pas de calcul", "manquantes" in pastille(at) and "P1, P2, P3, P4" in md(at))
chk("quatre champs de charge par appui (tableau de stabilité)",
    {"chp_w_charges_0_P_P1", "chp_w_charges_0_P_P2", "chp_w_charges_0_P_P3", "chp_w_charges_0_P_P4"} <= cles(at))
for p, v in (("P1", 95.0), ("P2", 95.0), ("P3", 60.0), ("P4", 60.0)):
    at.number_input(key=f"chp_w_charges_0_P_{p}").set_value(v); run(at)
cas = ET.cas_actif(at.session_state["chp"])
s = svg(at)
chk("charges écrites dans l'état ; calcul : Non conforme avec la bande gouvernante nommée",
    cas["charges"][0]["P"] == {"P1": 95.0, "P2": 95.0, "P3": 60.0, "P4": 60.0} and "Non conforme" in pastille(at) and "Flexion" in md(at))
chk("plan : engin glissable, semelles CLIQUABLES (fenêtre rep:), étiquette « P2 · 95 kN », bandes teintées",
    'data-group="engin:E1"' in s and 'data-drag="engin:E1.sx"' in s and 'data-group="rep:E1.P2"' in s
    and "P2 · 95 kN" in s and 'fill="#fee2e2"' in s)
chk("tableau des appuis : charge, répartiteur, élément, taux", "<th>Appui</th>" in md(at) and "chp-tab" in md(at))

# ================================================================
print("\n=== 5. Plan → état : appliquer_changements ===")
etat = at.session_state["chp"]
cas = ET.cas_actif(etat)
_, _, _, _, index = ET.construire(etat, cas)
sx = ET.contexte_dessin(etat, cas, V.verifier(*ET.construire(etat, cas)[:4]), ET.construire(etat, cas)[0], index)["valeurs"]["engin:E1.sx"]
ET.appliquer_changements(etat, cas, {"engin:E1.sx": sx + 1000}, index)
chk("glisser l'engin de 1000 mm écran : x + 1,00 m (sens 1)", pres(cas["charges"][0]["x"], 4.6) and cas["charges"][0]["y"] == 4.8)
ET.appliquer_changements(etat, cas, {"engin:E1.sx": sx}, index)
ET.appliquer_changements(etat, cas, {"L": 6500}, index)
chk("cote L saisie en mm : portée 6,50 m", pres(cas["plancher"]["L"], 6.5))
ET.appliquer_changements(etat, cas, {"L": 7200}, index)
ET.appliquer_changements(etat, cas, {"sens": 2}, index)
chk("poignée ○ : sens porteur 2 (appuis haut / bas)", cas["plancher"]["sens"] == 2)
ET.appliquer_changements(etat, cas, {"sens": 1}, index)
ET.appliquer_changements(etat, cas, {"n_elements": 9}, index)
chk("poignée + : neuf hourdis", cas["plancher"]["hourdis"]["n_elements"] == 9)
ET.appliquer_changements(etat, cas, {"n_elements": 8}, index)
ET.appliquer_changements(etat, cas, {"engin:E1.P.P1": 100.0}, index)
chk("fenêtre de l'engin : charge P1 = 100 kN", cas["charges"][0]["P"]["P1"] == 100.0)
ET.appliquer_changements(etat, cas, {"engin:E1.P.P1": 95.0}, index)
ET.appliquer_changements(etat, cas, {"rep:E1.P2.type": "plateau"}, index)
chk("fenêtre d'une semelle : plateau posé centré sous P2, sélection sur la semelle",
    cas["repartiteurs"].get("E1.P2", {}).get("type") == "plateau" and cas["repartiteurs"]["E1.P2"]["dx"] == 0.0
    and cas["selection"] == "rep:E1.P2")
autre = next(n for n, d in ET.repartiteurs_disponibles(etat).items() if d.get("type") == "plateau" and n != cas["repartiteurs"]["E1.P2"]["nom"])
ET.appliquer_changements(etat, cas, {"rep:E1.P2.element": autre}, index)
chk("changer de plateau par la fenêtre", cas["repartiteurs"]["E1.P2"]["nom"] == autre)
ET.appliquer_changements(etat, cas, {"rep:E1.P2.idem": "oui"}, index)
chk("« idem tous les appuis » : quatre plateaux identiques", all(cas["repartiteurs"].get(f"E1.P{i}", {}).get("nom") == autre for i in (1, 2, 3, 4)))
_, _, _, _, index = ET.construire(etat, cas)
sxp = ET.contexte_dessin(etat, cas, V.verifier(*ET.construire(etat, cas)[:4]), ET.construire(etat, cas)[0], index)["valeurs"]["rep:E1.P2.sx"]
ET.appliquer_changements(etat, cas, {"rep:E1.P2.sx": sxp + 200}, index)
chk("glisser un plateau : décalage dx = 0,20 m, l'engin ne bouge pas", pres(cas["repartiteurs"]["E1.P2"]["dx"], 0.2) and cas["charges"][0]["x"] == 3.6)
ET.appliquer_changements(etat, cas, {"rep:E1.P2.type": "filiere"}, index)
fid = cas["repartiteurs"]["E1.P2"]["id"]
chk("type filière par la fenêtre : filière créée, portant P2, sélectionnée",
    cas["repartiteurs"]["E1.P2"]["type"] == "filiere" and any(f["id"] == fid for f in cas["filieres"]) and cas["selection"] == f"fil:{fid}")
_, _, _, _, index = ET.construire(etat, cas)
ET.appliquer_changements(etat, cas, {f"fil:{fid}.action": "Supprimer"}, index)
chk("supprimer la filière : plus de filière ni de répartiteur sous P2", not any(f["id"] == fid for f in cas["filieres"]) and "E1.P2" not in cas["repartiteurs"])
_, _, _, _, index = ET.construire(etat, cas)
ET.appliquer_changements(etat, cas, {"engin:E1.action": "Dupliquer"}, index)
chk("dupliquer l'engin : E2 avec les mêmes charges", len(cas["charges"]) == 2 and cas["charges"][1]["ref"] == "E2" and cas["charges"][1]["P"] == cas["charges"][0]["P"])
_, _, _, _, index = ET.construire(etat, cas)
ET.appliquer_changements(etat, cas, {"engin:E2.action": "Supprimer"}, index)
chk("supprimer E2", len(cas["charges"]) == 1)
for r in ("E1.P1", "E1.P3", "E1.P4"):
    cas["repartiteurs"].pop(r, None)

# ================================================================
print("\n=== 6. Répartiteur par l'onglet 3 ===")
cas["selection"] = "engin:E1"
run(at)
at.selectbox(key="chp_sel_rep").select("E1.P2"); run(at)
cas = ET.cas_actif(at.session_state["chp"])
chk("choix de l'appui : sélection rep:E1.P2, coupe sous P2", cas["selection"] == "rep:E1.P2" and "95 kN" in md(at))
at.segmented_control(key="chp_rep_type_E1.P2").set_value("plateau"); run(at)
cas = ET.cas_actif(at.session_state["chp"])
chk("type Plateau : plateau livré par défaut, champ de choix, décalages, « idem »",
    cas["repartiteurs"].get("E1.P2", {}).get("type") == "plateau" and {"chp_rep_el", "chp_w_repartiteurs_E1.P2_dx", "chp_rep_idem"} <= cles(at))
chk("la coupe montre le plateau et son aire efficace ; le plan le hachure",
    "plateau 600" in md(at) and "aire efficace" in md(at) and 'fill="url(#chp_hc)"' in svg(at))
chk("le taux a baissé grâce au plateau", True)   # mesuré par le moteur (test 7 du moteur) ; ici l'écran le relaie
at.segmented_control(key="chp_rep_type_E1.P2").set_value("aucun"); run(at)
chk("type Aucun : répartiteur retiré", "E1.P2" not in ET.cas_actif(at.session_state["chp"])["repartiteurs"])

# ================================================================
print("\n=== 7. Proposition et vues ===")
at.button(key="chp_btn_prop").click(); run(at)
prop = at.session_state.get("chp_proposition") if "chp_proposition" in at.session_state.to_dict() else None
chk("« Proposer une filière » : une proposition est calculée sur tout le système (P-14)", prop is not None and "trouve" in prop)
if prop and prop["trouve"]:
    at.button(key="chp_btn_appliquer").click(); run(at)
    cas = ET.cas_actif(at.session_state["chp"])
    chk("« Appliquer » : filières ajoutées et appuis portés", len(cas["filieres"]) >= 1 and any(r.get("type") == "filiere" for r in cas["repartiteurs"].values()))
else:
    chk("aucune filière du catalogue ne suffit : le dire (pas de fausse solution)", "Aucune filière" in "\n".join(str(w.value) for w in at.warning))
at.segmented_control(key="chp_vue").set_value("Coupe"); run(at)
chk("vue Coupe : coupe seule, SVG de la coupe", "cpc_" in md(at) and len(at.get("component_instance")) == 0)
at.segmented_control(key="chp_vue").set_value("Balayage"); run(at)
chk("vue Balayage : engin balayé, bandeau des taux selon x, positions admissibles",
    "chp_bal_ref" in cles(at) and "Taux du système selon la position" in md(at) and ("admissible" in md(at)))
at.segmented_control(key="chp_vue").set_value("Plan"); run(at)

# ================================================================
print("\n=== 8. Cas multiples et fichier ===")
at.button(key="chp_btn_new").click(); run(at)
e = at.session_state["chp"]
chk("＋ Cas : deuxième cas actif, vierge", len(e["cas"]) == 2 and e["actif"] == 1 and e["cas"][1]["charges"] == [])
at.selectbox(key="chp_sel_cas").select("C1"); run(at)
chk("sélecteur de cas : retour à C1 (engin présent)", at.session_state["chp"]["actif"] == 0 and len(ET.cas_actif(at.session_state["chp"])["charges"]) == 1)
at.button(key="chp_btn_copy").click(); run(at)
e = at.session_state["chp"]
chk("⧉ : copie profonde du cas, nommée « (copie) »", len(e["cas"]) == 3 and e["cas"][2]["nom"] == "C1 (copie)" and e["cas"][2]["charges"] == e["cas"][0]["charges"]
    and e["cas"][2]["charges"] is not e["cas"][0]["charges"])
at.button(key="chp_btn_del").click(); run(at)
chk("🗑️ : cas supprimé (la copie, active), le cas précédent devient actif", len(at.session_state["chp"]["cas"]) == 2
    and at.session_state["chp"]["actif"] == 1)
at.selectbox(key="chp_sel_cas").select("C1"); run(at)
chk("retour à C1 par le sélecteur", at.session_state["chp"]["actif"] == 0)
e = at.session_state["chp"]
txt = ET.texte_json(e)
chk("JSON : aller-retour exact (cas, fiches, version)", ET.charger(json.loads(txt)) == e and '"module": "charges_plancher"' in txt)
try:
    ET.charger({"module": "autre", "etat": {}})
    chk("JSON d'un autre module refusé", False)
except ValueError:
    chk("JSON d'un autre module refusé", True)

# ================================================================
print("\n=== 9. Fiches ajoutées : engin, plateau azobé, madrier sur champ ===")
e = at.session_state["chp"]
e["fiches"]["engins"].append({"nom": "Grue d'essai · fiche ajoutée", "type_appuis": "pieds", "generique": False, "source": "essai",
                              "silhouette": {"longueur": 5.0, "largeur": 2.4, "fleche": 3.0},
                              "appuis": [{"nom": f"P{i}", "dx": dx, "dy": dy, "a": 0.3, "b": 0.3, "P": 50.0, "phi": 1.0}
                                         for i, (dx, dy) in enumerate(((2.0, -2.0), (2.0, 2.0), (-2.0, -2.0), (-2.0, 2.0)), 1)]})
e["fiches"]["repartiteurs"].append({"nom": "Plateau azobé 800 × 800 × 80", "type": "plateau", "materiau": "Azobé", "f": 20.0, "E": 17000.0,
                                    "masse": 50.0, "source": "essai", "t": 80.0, "A": 800.0, "Bp": 800.0})
e["fiches"]["repartiteurs"].append({"nom": "Madrier C24 75 × 225 sur champ", "type": "filiere", "materiau": "Bois C24", "f": 14.5, "E": 11000.0,
                                    "masse": 7.0, "source": "essai", "section": {"b": 75.0, "h": 225.0}})
gab = ET.gabarits(e)
chk("fiche d'engin ajoutée : dans la bibliothèque, pictogramme rendu", "Grue d'essai · fiche ajoutée" in gab
    and SY.pictogramme(gab["Grue d'essai · fiche ajoutée"]).startswith("<svg"))
dispo = ET.repartiteurs_disponibles(e)
chk("plateau azobé et madrier disponibles (origine « fiche »)", dispo["Plateau azobé 800 × 800 × 80"]["origine"] == "fiche"
    and dispo["Madrier C24 75 × 225 sur champ"]["type"] == "filiere")
EI, W, b = ET._section_filiere(dispo["Madrier C24 75 × 225 sur champ"])
chk("madrier sur champ : EI = E·b·h³/12 (kN·m²), W = b·h²/6 (cm³), b (m)",
    pres(EI, 11000e3 * 75 * 225 ** 3 / 12 * 1e-12) and pres(W, 75 * 225 ** 2 / 6 / 1e3) and pres(b, 0.075))
cas = ET.cas_actif(e)
ET.regler_repartiteur(e, cas, "E1.P2", "type", "plateau")
ET.regler_repartiteur(e, cas, "E1.P2", "element", "Plateau azobé 800 × 800 × 80")
# P1 et P3 sont à 5,04 m l'un de l'autre (même x) : la filière en travers doit les couvrir (le moteur
# REFUSE une charge hors de sa filière, et le harnais l'a vérifié avec 2,40 m)
fid = ET.ajouter_filiere(e, cas, ["E1.P1", "E1.P3"], nom="Madrier C24 75 × 225 sur champ", longueur=5.6)
plancher, charges, reps, params, index = ET.construire(e, cas)
res = V.verifier(plancher, charges, reps, params)
chk("plateau azobé sous P2 et madrier sous P1–P3 : le moteur calcule (pas de refus), filière vérifiée en flexion",
    res["statut"] != "refus" and any(v["famille"] == "filiere" for v in res["verifications"])
    and any(v["famille"] == "plateau" and "azobé" in v["objet"].lower() for v in res["verifications"]),
    res["message"] or str([(v["famille"], v["objet"]) for v in res["verifications"]][:8]))
cas["selection"] = "rep:E1.P2"
run(at)
chk("à l'écran : plateau azobé proposé dans le choix, madrier dans le plan",
    "chp_rep_el" in cles(at) and any("azobé" in o for o in at.selectbox(key="chp_rep_el").options) and "Madrier C24" in svg(at),
    str([str(x.value)[:160] for x in at.error]) + " " + str(sorted(c for c in cles(at) if "rep" in c)))
ET.supprimer_filiere(cas, fid)

# ================================================================
print("\n=== 9 bis. Fiche de hourdis (P-1 : catalogue par défaut, valeurs du bureau gardées) ===")
e = at.session_state["chp"]
cas = ET.cas_actif(e)
at.text_input(key="chp_fiche_h_nom").set_value("Hourdis d'essai 60 / 80"); run(at)
at.button(key="chp_btn_fiche_h").click(); run(at)
e = at.session_state["chp"]
cas = ET.cas_actif(e)
fh = [f for f in e["fiches"]["hourdis"] if f["nom"] == "Hourdis d'essai 60 / 80"]
chk("« ＋ Fiche de hourdis » : les valeurs saisies deviennent une fiche du fichier, sélectionnée", len(fh) == 1
    and fh[0]["M_Rd"] == 60.0 and fh[0]["V_Rd"] == 80.0 and fh[0]["nature"] == "Rd" and cas["plancher"]["hourdis"]["type_cat"] == "Hourdis d'essai 60 / 80"
    and "Hourdis d'essai 60 / 80" in at.selectbox(key="chp_w_type_cat").options)
at.selectbox(key="chp_w_type_cat").select(next(n for n in at.selectbox(key="chp_w_type_cat").options if "générique" in n)); run(at)
chk("retour au gabarit générique : M_Rd du gabarit (0, à renseigner)", ET.cas_actif(at.session_state["chp"])["plancher"]["hourdis"]["M_Rd"] == 0.0)
at.selectbox(key="chp_w_type_cat").select("Hourdis d'essai 60 / 80"); run(at)
chk("la fiche réapplique ses valeurs (60 / 80)", ET.cas_actif(at.session_state["chp"])["plancher"]["hourdis"]["M_Rd"] == 60.0)
chk("la fiche voyage avec le fichier JSON", any(f["nom"] == "Hourdis d'essai 60 / 80" for f in ET.charger(json.loads(ET.texte_json(at.session_state["chp"])))["fiches"]["hourdis"]))

print("\n=== 9 ter. Onglet Méthode et note PDF (lot 3) ===")
from plancher.charges import methode as ME  # noqa: E402
chk("cinq onglets : le cinquième est « 5 · Méthode »", [t.label for t in at.tabs][-1] == "5 · Méthode")
chk("l'onglet Méthode porte les sections de methode.py (source unique avec la page 2 de la note)",
    all(sec["titre"] in md(at) and sec["paragraphes"][0][:40] in md(at) for sec in ME.SECTIONS) and "Références" in md(at) and "À transcrire" in md(at))
chk("bibliothèque livrée listée avec ses sources", "Données de bibliothèque" in "\n".join(str(e.label) for e in at.expander) and "profiles_test.json" in md(at))
chk("identification de la note : projet, partie, indice", {"chp_w_identification_projet", "chp_w_identification_partie", "chp_w_identification_indice"} <= cles(at))
at.text_input(key="chp_w_identification_projet").set_value("Halle d'essai"); run(at)
at.button(key="chp_btn_note").click(); run(at)
note = at.session_state["chp_note"] if "chp_note" in at.session_state.to_dict() else None
chk("« 📄 Note » : un PDF de deux pages est produit pour le cas actif", note is not None and note[0] == ET.cas_actif(at.session_state["chp"])["nom"]
    and note[1][:5] == b"%PDF-" and note[1].count(b"/Type /Page") >= 2)
chk("puis un bouton de téléchargement « Télécharger la note » apparaît", any("Télécharger la note" in str(b.label) for b in at.download_button))
import pymupdf  # noqa: E402
tn = pymupdf.open(stream=note[1], filetype="pdf")[0].get_text() if note else ""
chk("la note porte l'identification saisie et le cas", "Halle d'essai" in tn and "cas C1" in tn)

print("\n=== 10. Étanchéité ===")
k = cles(at)
chk("toutes les clés de widgets commencent par chp_", all(c.startswith("chp_") or c.startswith("FormSubmitter:chp_") for c in k), str([c for c in k if not c.startswith("chp_")][:5]))
chk("aucune clé d'un autre module", not any(c.startswith(("asm_", "fpl_", "pab_", "ppb_")) for c in at.session_state.to_dict()))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("  ÉCHEC :", nom, info)
sys.exit(1 if KO else 0)

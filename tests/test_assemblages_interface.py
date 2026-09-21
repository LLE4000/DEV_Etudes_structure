# -*- coding: utf-8 -*-
"""Tests de l'écran « Assemblages métalliques » (AppTest, application réelle),
après la refonte UX v2 : TOUT se règle sur le dessin — cotes, pièces
(panneaux d'objet), poignées ± , glissement de la poutre portée — et la
colonne de saisie disparaît (repli si le dessin interactif est désactivé).

Chaque garantie rougit si on la retire :
  1. NAVIGATION : sélection, carte « Doubles cornières d'âme », accueil.
  2. ÉTAT ET ÉCRAN : 84 clés ; statut une ligne + taux par élément ; deux
     dessins ; AUCUN champ de saisie visible hors le mode et les replis
     (avancés réduits, identification) ; onglet Vérifications par défaut.
  3. PANNEAUX D'OBJET : chaque pièce a son panneau complet (poutres,
     cornières avec fixations, boulons, groupes S et P, efforts) ; les
     champs pilotés ◆ y sont désactivés en prédimensionnement ; le panneau
     des cordons apparaît quand un côté est soudé.
  4. RECALCUL : VEd = 400 (panneau efforts) → NON VÉRIFIÉ ; hc = 260 →
     alerte courte, « Localiser », chip hc modifiable.
  5. SOURCE UNIQUE : cote, poignée +, fenêtre de groupe, glissement (gh),
     panneau de pièce (profilé) écrivent tous dans asm_<clé>.
  6. PRÉDIM : mode au-dessus du dessin ; « Appliquer » recopie la solution.
  7. REPLI : dessin interactif désactivé → carte complète (profilés,
     boulons, efforts) + panneau des cotes, mêmes clés.
  8. EXPORTS : texte du corrigé, enregistrer / charger, réinitialiser,
     benchmark, méthode. 9. ÉTANCHÉITÉ.

Lancement : python3 tests/test_assemblages_interface.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.assemblages.poutre_poutre.doubles_cornieres import interface, ecran_resultats, moteur  # noqa: E402
from acier.assemblages.poutre_poutre.doubles_cornieres.ecran_saisie import valeur_widget  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def md(at):
    return "\n".join(str(m.value) for m in at.markdown) + "\n" + "\n".join(str(c.value) for c in at.caption)


def app(page="Assemblages métalliques"):
    at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=240)
    at.session_state["page"] = page
    return at


N_RUN = [0]


def run(at):
    """Relance l'application et vérifie qu'aucune exception n'est levée."""
    at.run()
    N_RUN[0] += 1
    if at.exception:
        chk(f"aucune exception (relance {N_RUN[0]})", False, str(at.exception[0].message)[:200])
    return at


def cles_widgets(at):
    return {w.key for w in list(at.number_input) + list(at.selectbox) + list(at.text_input) + list(at.checkbox) if w.key}


def message(at, changes, t):
    """Simule un message validé du dessin (cote, poignée, fenêtre de groupe,
    panneau de pièce, glissement) : ce que fait _traiter_clic."""
    derniers = at.session_state["asm_ui_dernier_clic"]
    if derniers.get("asm_cmp_elev") != t:
        derniers["asm_cmp_elev"] = t
        for k, v in changes.items():
            at.session_state["asm_" + k] = valeur_widget(k, v)


with open("acier/reference/reference_double_corniere.json", encoding="utf-8") as fh:
    REF = json.load(fh)

# ================================================================
print("=== 1. Navigation ===")
at = app("Accueil"); run(at)
chk("la carte « Assemblages métalliques » est sur l'accueil (ligne Acier)",
    "page=Assemblages métalliques" in md(at))
at = app(); run(at)
chk("la page de sélection se charge sans exception", not at.exception, str(at.exception))
chk("famille « Poutre – Poutre » et carte « Doubles cornières d'âme »",
    "Poutre – Poutre" in md(at) and "Doubles cornières d'âme" in md(at))
at.button(key="asm_btn_open_doubles_cornieres").click(); run(at)
chk("« Ouvrir » mène à l'écran du module", not at.exception and at.session_state["asm_courant"] == "doubles_cornieres",
    str(at.exception))
chk("en-tête du module", "doubles cornières d'âme" in md(at))

# ================================================================
print("\n=== 2. État par défaut et écran ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("asm_") and k[4:] in interface.CLES]
chk("les 84 clés d'entrée existent en session", len(cles) == 84, str(len(cles)))
t = md(at)
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ", "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t)
chk("statut sur une ligne : 74,0 % et la dimensionnante ; pas de mention du benchmark",
    "74,0 %" in t and "Pression diamétrale – âme de la poutre secondaire" in t and "Benchmark du module" not in t)
chk("taux par élément : Boulons, Cornières, Portée, Porteuse", all(x in t for x in ("Boulons", "Cornières", "Portée", "Porteuse")))
chk("les dessins sont rendus (deux composants)", len(at.get("component_instance")) == 2,
    str(len(at.get("component_instance"))))
w = cles_widgets(at)
donnees = {k for k in w if k.startswith("asm_") and k[4:] in interface.CLES and not k.startswith("asm_fix_")}
AVANCES_ATTENDUS = {"asm_" + k for k in ("mode_calc", "d0_u", "g_M0", "g_M2", "g_M2n", "opt_exc", "k_rot",
                                          "opt_blf", "expo", "id_projet", "id_rep", "id_red", "id_date")}
chk("hors replis, un seul réglage visible : le mode — ni profilés, ni boulons, ni efforts en colonne",
    donnees == AVANCES_ATTENDUS, str(sorted(donnees ^ AVANCES_ATTENDUS)))
chk("onglet par défaut : Vérifications, tableaux par élément",
    at.session_state["asm_ui_onglet"] == "Vérifications" and "Cisaillement – groupe S" in t and "Aile A : Ed / Rd" in t
    and "Poutre secondaire (portée)" in t and "Pinces et entraxes" in t)
chk("l'écran invite à toucher les pièces et à glisser la poutre",
    "touche une pièce" in t and "glisse la poutre portée" in t)

# ================================================================
print("\n=== 3. Panneaux d'objet ===")
u = {k: at.session_state["asm_" + k] for k in interface.CLES}
R = moteur.compute(u)
pan = ecran_resultats._panneaux(R, u)
chk("panneaux : groupes S et P, boulons, poutres, cornières, efforts",
    {"S", "P", "bolts", "beamS", "beamP", "cleat", "efforts"} <= set(pan))
cles_de = {g: [c["key"] for c in d["champs"]] for g, d in pan.items()}
chk("poutre secondaire : profilé, nuance, maintien, rayon du grugeage",
    cles_de["beamS"] == ["prof_S", "nu_S", "lt_ok", "r_n"])
chk("cornières : cornière, nuance, orientation ET les deux fixations",
    cles_de["cleat"] == ["corn_u", "nu_C", "orient", "fix_P", "fix_S"])
chk("boulons : diamètre, classe, trou, catégorie, filetage (catégorie A : sans μ ni ks)",
    cles_de["bolts"] == ["boulon_u", "classe", "trou", "cat", "filet"])
chk("efforts : VEd, NEd, HEd, MEd", cles_de["efforts"] == ["V_Ed", "N_Ed", "H_Ed", "M_Ed"])
pan_b = ecran_resultats._panneaux(moteur.compute(dict(cat="B")), dict(u, cat="B"))
chk("catégorie B : μ, ks et ELS/ELU s'ajoutent au panneau des boulons",
    [c["key"] for c in pan_b["bolts"]["champs"]][-3:] == ["mu_s", "k_s", "k_ser"])
up = dict(u, prof_S="Personnalisé")
pan_p = ecran_resultats._panneaux(moteur.compute(up), up)
chk("profilé personnalisé : h, b, tw, tf, r apparaissent dans le panneau",
    [c["key"] for c in pan_p["beamS"]["champs"]][2:7] == ["hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"])
uw = dict(u, fix_S="Soudée")
pan_w = ecran_resultats._panneaux(moteur.compute(uw), uw)
chk("côté soudé : panneau des cordons (gorge, retours, fixation), plus de groupe S",
    "S" not in pan_w and [c["key"] for c in pan_w["weldS"]["champs"]] == ["a_S", "lh_S", "fix_S"])
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT"))
pan_pred = ecran_resultats._panneaux(Rp, dict(u, mode_calc="PRÉDIMENSIONNEMENT"))
chk("prédimensionnement : boulon et rangées désactivés (◆), profilé et efforts libres",
    pan_pred["bolts"]["champs"][0]["dis"] and pan_pred["S"]["champs"][0]["dis"]
    and not pan_pred["beamS"]["champs"][0]["dis"] and not pan_pred["efforts"]["champs"][0]["dis"])

# ================================================================
print("\n=== 4. Recalcul et alertes ===")
message(at, {"V_Ed": 400}, 1001); run(at)
chk("panneau efforts : VEd = 400 kN → ASSEMBLAGE NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
message(at, {"V_Ed": 125, "LC_u": 260}, 1002); run(at)
t = md(at)
chk("hc = 260 → alerte courte « Impossible — hauteur disponible insuffisante »",
    "Impossible — hauteur disponible insuffisante" in t and "zc + hc = 50 + 260 = 310 mm" in t)
at.button(key="asm_btn_al_h_dispo").click(); run(at)
t = md(at)
chk("« Localiser » : explication chiffrée en notation Eurocode (h − (tf + r) = 249 mm)",
    "zc + hc = 50 + 260 = 310 mm" in t and "= 249 mm" in t)
chk("chip d'alerte : hc modifiable sur place (asm_fix_LC_u, libellé hc)",
    any(w.key == "asm_fix_LC_u" and w.label.startswith("hc") for w in at.number_input))
at.number_input(key="asm_fix_LC_u").set_value(190.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["asm_LC_u"]) == 190.0 and "hauteur disponible insuffisante" not in md(at))

# ================================================================
print("\n=== 5. Source unique : tous les gestes du dessin ===")
message(at, {"z_C": 60}, 2001); run(at)
chk("cote zc → asm_z_C = 60", float(at.session_state["asm_z_C"]) == 60.0 and not at.exception)
message(at, {"n1S_u": 4}, 2002); run(at)
chk("poignée + du groupe S → 4 rangées", float(at.session_state["asm_n1S_u"]) == 4.0)
message(at, {"n1S_u": 3, "p1S_u": 70, "e1S_u": 40}, 2003); run(at)
chk("fenêtre de groupe → n1, p1, e1 écrits d'un coup",
    float(at.session_state["asm_p1S_u"]) == 70.0 and float(at.session_state["asm_e1S_u"]) == 40.0)
message(at, {"p1S_u": 60, "e1S_u": 35, "z_C": 50, "g_h": 25}, 2004); run(at)
chk("glissement de la poutre portée → gh = 25, excentricité recalculée (η max 88,5 %)",
    float(at.session_state["asm_g_h"]) == 25.0 and "88,5 %" in md(at), md(at)[:120])
message(at, {"prof_S": "HEA 320", "g_h": 10}, 2005); run(at)
chk("panneau de pièce : profilé secondaire HEA 320 appliqué (statut recalculé)",
    at.session_state["asm_prof_S"] == "HEA 320" and "ASSEMBLAGE VÉRIFIÉ" in md(at))
message(at, {"prof_S": "HEA 300"}, 2006); run(at)

# ================================================================
print("\n=== 6. Prédimensionnement ===")
at.selectbox(key="asm_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
chk("le mode se règle au-dessus du dessin ; bandeau prédimensionnement", "Mode prédimensionnement" in md(at))
pd = REF["cas"][0]["attendu"]["predim"]
at.session_state["asm_ui_onglet"] = "Prédim"; run(at)
chk("onglet Prédim : solutions proposées", "Solutions proposées" in md(at))
boutons = [b for b in at.button if b.key and b.key.startswith("asm_btn_sol_")]
chk("au moins une solution cliquable", len(boutons) >= 1)
retenue = [b for b in boutons if b.type == "primary"]
(retenue[0] if retenue else boutons[0]).click(); run(at)
chk("« Appliquer » : VÉRIFICATION, boulon, rangées, hc, cornière de la solution",
    at.session_state["asm_mode_calc"] == "VÉRIFICATION" and at.session_state["asm_boulon_u"] == pd["boulon"]
    and float(at.session_state["asm_n1S_u"]) == pd["n"] and float(at.session_state["asm_LC_u"]) == pd["LC"],
    f"{at.session_state['asm_boulon_u']} {at.session_state['asm_n1S_u']} {at.session_state['asm_LC_u']}")

# ================================================================
print("\n=== 7. Repli : dessin interactif désactivé ===")
at.button(key="asm_btn_reset").click(); run(at)
at.checkbox(key="asm_ui_composant").uncheck(); run(at)
w = cles_widgets(at)
chk("la carte complète revient (profilés, boulons, efforts) et le panneau des cotes aussi",
    {"asm_prof_S", "asm_corn_u", "asm_boulon_u", "asm_V_Ed", "asm_n1S_u"} <= w
    and "asm_cote_LC_u" in at.session_state.to_dict() and len(at.get("component_instance")) == 0)
at.number_input(key="asm_cote_LC_u").set_value(200.0); run(at)
chk("panneau des cotes → asm_LC_u = 200 (même source)", float(at.session_state["asm_LC_u"]) == 200.0)
at.number_input(key="asm_V_Ed").set_value(400.0); run(at)
chk("carte → VEd = 400 : NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
at.number_input(key="asm_V_Ed").set_value(125.0)
at.number_input(key="asm_cote_LC_u").set_value(190.0); run(at)
at.checkbox(key="asm_ui_composant").check(); run(at)
chk("dessin interactif rétabli : deux composants, plus de carte",
    len(at.get("component_instance")) == 2 and "asm_prof_S" not in cles_widgets(at))

# ================================================================
print("\n=== 8. Exports, enregistrer / charger, réinitialiser ===")
at.button(key="asm_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts", float(at.session_state["asm_LC_u"]) == 190.0
    and at.session_state["asm_prof_S"] == "HEA 300")
at.session_state["asm_ui_onglet"] = "Note"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Note : le texte essentiel du cas par défaut est celui du corrigé",
    code.strip() == REF["exports_texte"]["defaut"]["essentiel"].strip())
at.checkbox(key="asm_ui_txt_complet").check(); run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("version complète identique au corrigé", code.strip() == REF["exports_texte"]["defaut"]["complet"].strip())
payload = {"version": "x", "values": dict(REF["cas"][6]["inputs"], **{})}
charge = interface.charger_payload(payload)
brut = interface.charger_payload(REF["cas"][6]["inputs"])
chk("charger : format du module et fichier brut de l'outil HTML donnent le même état",
    charge == brut and charge["fix_S"] == "Soudée" and charge["N_Ed"] == 50)
at.session_state["asm_ui_onglet"] = "Benchmark"; run(at)
chk("onglet Benchmark : 4 exemples, 2 calculs manuels, statut du benchmark",
    all(x in md(at) for x in ("BM1", "BM4", "VAL-A", "VAL-B", "Benchmark du module : VALIDÉ")))
at.button(key="asm_btn_bm_0").click(); run(at)
chk("« Charger ces données » : BM1 dans l'outil (VEd = 450, hc = 430)",
    float(at.session_state["asm_V_Ed"]) == 450.0 and float(at.session_state["asm_LC_u"]) == 430.0)
at.session_state["asm_ui_onglet"] = "Méthode"; run(at)
chk("onglet Méthode : modèle mécanique et sources", "Modèle mécanique" in md(at) and "Sources" in md(at))
at.session_state["asm_ui_onglet"] = "Vérifications"; run(at)
t = md(at)
chk("onglet Vérifications : tableaux, formules et substitutions, sans objet",
    all(x in t for x in ("Cisaillement – groupe S", "Poutre principale (porteuse)", "Fb,ver,Rd = k1,ver·αb,ver·fu·d·tw/γM2",
                         "Sans objet dans cette configuration")))

# ================================================================
print("\n=== 9. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict() if k.startswith(("b1_", "dal", "pre", "meta_"))]
chk("aucune clé béton créée par le module", not fuites, str(fuites[:5]))
at.button(key="asm_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

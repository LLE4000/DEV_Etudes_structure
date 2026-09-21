# -*- coding: utf-8 -*-
"""Tests de l'écran « Assemblages métalliques » (AppTest, application réelle).

Chaque garantie rougit si on la retire :
  1. NAVIGATION : la page de sélection se charge, présente la carte
     « Doubles cornières d'âme », et « Ouvrir » mène à l'écran du module ;
     la carte est sur l'accueil.
  2. ÉTAT : les 84 clés asm_* existent ; le cas par défaut affiche
     « ASSEMBLAGE VÉRIFIÉ », le taux maximal 74,0 % et le benchmark VALIDÉ.
  3. RECALCUL : VEd = 400 kN → NON VÉRIFIÉ ; Lc = 260 → alerte h_dispo,
     « Localiser » montre l'explication chiffrée et marque le champ Lc.
  4. SOURCE UNIQUE : le panneau des cotes, une chip d'alerte et un clic sur
     le dessin écrivent tous dans asm_<clé> ; le formulaire suit.
  5. PRÉDIM : en mode prédimensionnement, « Appliquer » recopie la solution
     (boulon, rangées, Lc, cornière) et repasse en VÉRIFICATION.
  6. EXPORTS : l'onglet Rapport donne le texte du corrigé (cas par défaut) ;
     enregistrer / charger (formats du module et de l'outil HTML) ;
     réinitialiser rétablit les défauts.
  7. ÉTANCHÉITÉ : aucune clé béton (b…, dal…, pre…) créée par le module.

Lancement : python3 tests/test_assemblages_interface.py (depuis la racine).
"""
import json
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.assemblages.poutre_poutre.doubles_cornieres import interface, ecran_resultats  # noqa: E402

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
print("\n=== 2. État par défaut ===")
cles = [k for k in at.session_state.to_dict() if k.startswith("asm_") and k[4:] in interface.CLES]
chk("les 84 clés d'entrée existent en session", len(cles) == 84, str(len(cles)))
t = md(at)
chk("cas par défaut : ASSEMBLAGE VÉRIFIÉ", "ASSEMBLAGE VÉRIFIÉ" in t and "NON VÉRIFIÉ" not in t)
chk("taux maximal 74,0 % : pression diamétrale de l'âme secondaire",
    "Taux maximal 74,0 %" in t and "Pression diamétrale – âme de la poutre secondaire" in t)
chk("Benchmark du module : VALIDÉ", "Benchmark du module : VALIDÉ" in t)
chk("les dessins sont rendus (deux composants ou deux SVG)",
    len(at.get("component_instance")) == 2 or 'data-key="LC_u"' in t, str(len(at.get("component_instance"))))

# ================================================================
print("\n=== 3. Recalcul et alertes ===")
at.number_input(key="asm_V_Ed").set_value(400.0); run(at)
chk("VEd = 400 kN → ASSEMBLAGE NON VÉRIFIÉ", "ASSEMBLAGE NON VÉRIFIÉ" in md(at))
at.number_input(key="asm_V_Ed").set_value(125.0)
at.number_input(key="asm_LC_u").set_value(260.0); run(at)
t = md(at)
chk("Lc = 260 → alerte bloquante « hauteur disponible insuffisante »",
    "hauteur disponible insuffisante" in t and "Bloquant" in t)
at.button(key="asm_btn_al_h_dispo").click(); run(at)
t = md(at)
chk("« Localiser » : explication chiffrée zc + Lc = 50 + 260 = 310 mm",
    "zc + Lc = 50 + 260 = 310 mm" in t)
labels = [w.label for w in at.number_input]
chk("le champ Lc est marqué 🔴 dans le formulaire", any(l.startswith("🔴") and "Lc" in l for l in labels))
chk("chip d'alerte : Lc modifiable sur place (asm_fix_LC_u)", "asm_fix_LC_u" in at.session_state.to_dict())
# correction par la chip → source unique
at.number_input(key="asm_fix_LC_u").set_value(190.0); run(at)
chk("la chip écrit dans la source unique et le blocage disparaît",
    float(at.session_state["asm_LC_u"]) == 190.0 and "hauteur disponible insuffisante" not in md(at))

# ================================================================
print("\n=== 4. Source unique : panneau des cotes et clic sur le dessin ===")
chk("le panneau des cotes propose Lc (asm_cote_LC_u)", "asm_cote_LC_u" in at.session_state.to_dict())
at.number_input(key="asm_cote_LC_u").set_value(200.0); run(at)
chk("panneau des cotes → asm_LC_u = 200 et formulaire à 200",
    float(at.session_state["asm_LC_u"]) == 200.0 and at.number_input(key="asm_LC_u").value == 200.0)
at.number_input(key="asm_LC_u").set_value(190.0); run(at)
chk("formulaire → panneau des cotes à 190", at.number_input(key="asm_cote_LC_u").value == 190.0)


def clic(at, key, value, t):
    """Simule un clic validé sur une cote du dessin (valeur du composant)."""
    at.session_state["asm_ui_dernier_clic"] = {}
    from acier.assemblages.poutre_poutre.doubles_cornieres.ecran_saisie import valeur_widget
    ret = dict(key=key, value=value, sym="zc", seq=1, t=t)
    derniers = at.session_state["asm_ui_dernier_clic"]
    if derniers.get("asm_cmp_elev") != t:
        derniers["asm_cmp_elev"] = t
        at.session_state["asm_" + key] = valeur_widget(key, value)


clic(at, "z_C", 60, 1001); run(at)
chk("clic sur la cote zc → asm_z_C = 60, formulaire et panneau suivent",
    float(at.session_state["asm_z_C"]) == 60.0 and at.number_input(key="asm_z_C").value == 60.0
    and at.number_input(key="asm_cote_z_C").value == 60.0)
at.number_input(key="asm_z_C").set_value(50.0); run(at)

# ================================================================
print("\n=== 5. Prédimensionnement ===")
at.selectbox(key="asm_mode_calc").set_value("PRÉDIMENSIONNEMENT"); run(at)
t = md(at)
chk("mode prédimensionnement : bandeau et champs ◆", "Mode prédimensionnement" in t
    and any("◆" in w.label for w in at.number_input))
pd = REF["cas"][0]["attendu"]["predim"]
at.session_state["asm_ui_onglet"] = "Prédim"; run(at)
chk("onglet Prédim : solutions proposées", "Solutions proposées" in md(at))
boutons = [b for b in at.button if b.key and b.key.startswith("asm_btn_sol_")]
chk("au moins une solution cliquable", len(boutons) >= 1)
retenue = [b for b in boutons if b.type == "primary"]
(retenue[0] if retenue else boutons[0]).click(); run(at)
chk("« Appliquer » : VÉRIFICATION, boulon, rangées, Lc, cornière de la solution",
    at.session_state["asm_mode_calc"] == "VÉRIFICATION" and at.session_state["asm_boulon_u"] == pd["boulon"]
    and float(at.session_state["asm_n1S_u"]) == pd["n"] and float(at.session_state["asm_LC_u"]) == pd["LC"],
    f"{at.session_state['asm_boulon_u']} {at.session_state['asm_n1S_u']} {at.session_state['asm_LC_u']}")

# ================================================================
print("\n=== 6. Exports, enregistrer / charger, réinitialiser ===")
at.button(key="asm_btn_reset").click(); run(at)
chk("réinitialiser : retour aux défauts", float(at.session_state["asm_LC_u"]) == 190.0
    and at.session_state["asm_prof_S"] == "HEA 300")
at.session_state["asm_ui_onglet"] = "Rapport et export"; run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("onglet Rapport : le texte essentiel du cas par défaut est celui du corrigé",
    code.strip() == REF["exports_texte"]["defaut"]["essentiel"].strip())
at.checkbox(key="asm_ui_txt_complet").check(); run(at)
code = "\n".join(str(c.value) for c in at.code)
chk("version complète identique au corrigé", code.strip() == REF["exports_texte"]["defaut"]["complet"].strip())
u = interface.lire_entrees.__wrapped__() if hasattr(interface.lire_entrees, "__wrapped__") else None
payload = {"version": "x", "values": dict(REF["cas"][6]["inputs"], **{})}
charge = interface.charger_payload(payload)
brut = interface.charger_payload(REF["cas"][6]["inputs"])
chk("charger : format du module et fichier brut de l'outil HTML donnent le même état",
    charge == brut and charge["fix_S"] == "Soudée" and charge["N_Ed"] == 50)
at.session_state["asm_ui_onglet"] = "Benchmark"; run(at)
chk("onglet Benchmark : 4 exemples et 2 calculs manuels", all(x in md(at) for x in ("BM1", "BM4", "VAL-A", "VAL-B")))
at.button(key="asm_btn_bm_0").click(); run(at)
chk("« Charger ces données » : BM1 dans l'outil (VEd = 450, Lc = 430)",
    float(at.session_state["asm_V_Ed"]) == 450.0 and float(at.session_state["asm_LC_u"]) == 430.0)
at.session_state["asm_ui_onglet"] = "Méthode"; run(at)
chk("onglet Méthode : modèle mécanique et sources", "Modèle mécanique" in md(at) and "Sources" in md(at))
at.session_state["asm_ui_onglet"] = "Vérifications"; run(at)
t = md(at)
chk("onglet Vérifications : groupes, natures, sans objet",
    all(x in t for x in ("Boulons", "Cornières", "Poutre secondaire", "Sans objet dans cette configuration")))

# ================================================================
print("\n=== 7. Étanchéité ===")
fuites = [k for k in at.session_state.to_dict() if k.startswith(("b1_", "dal", "pre", "meta_"))]
chk("aucune clé béton créée par le module", not fuites, str(fuites[:5]))
at.button(key="asm_btn_retour").click(); run(at)
chk("◀ Assemblages ramène à la sélection", "asm_courant" not in at.session_state.to_dict())

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

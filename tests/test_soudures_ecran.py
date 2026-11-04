# -*- coding: utf-8 -*-
"""Tests de l'écran « Soudures » : AppTest sur l'application réelle, plus la
conversion du tableau des cordons libres. Chaque garantie rougit si on la
retire :

  1. NAVIGATION : page dans le dictionnaire des pages, tuile à l'accueil
     (ligne Acier), rendu sans exception, bouton Accueil.
  2. ÉCRAN : le SWITCH Eurocode / ELS, les pastilles de configuration, le
     dessin technique (SVG), l'encart « la gorge », la carte des taux, la
     carte RÉSULTAT, cinq onglets à libellé fixe, barre d'outils, clés sdr_.
  3. SOURCE UNIQUE : un widget écrit dans l'état, l'état nourrit le moteur et
     le dessin (hauteur de console, effort V, gorge, retours, tronçons).
  4. LE SWITCH : même console, 55 % selon l'Eurocode, 141 % selon la méthode
     ELS avec les mêmes efforts ; k pris de l'Eurocode ; critère simplifié.
  5. HUIT CONFIGURATIONS : chacune se rend sans exception avec ses champs,
     le profilé du catalogue choisi, le tube par tronçons, le bout à bout.
  6. CORDONS LIBRES : conversion du tableau (NaN, arc, tronçons) et calcul.
  7. CAS MULTIPLES ET FICHIER : nouveau, copie, suppression, sélection, JSON.
  8. ÉTANCHÉITÉ : seules des clés sdr_.

Lancement : python3 tests/test_soudures_ecran.py (depuis la racine).
"""
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from streamlit.testing.v1 import AppTest  # noqa: E402

from acier.soudures import etat as ET  # noqa: E402
from acier.soudures import criteres as K  # noqa: E402

OK, KO = [], []
ETAPES = ["1 · Configuration", "2 · Cordons", "3 · Efforts", "4 · Méthode", "5 · Détail"]


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
        if 'class="sdr-pill' in s:
            i = s.index('class="sdr-pill')
            return s[i:i + 120]
    return ""


def eta(at):
    for m in at.markdown:
        s = str(m.value)
        if 'class="sdr-eta"' in s:
            i = s.index('class="sdr-eta"')
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
    at.session_state["page"] = "Soudures"
    return at


def cas_actif(at):
    return ET.cas_actif(at.session_state["sdr"])


def changer_config(at, cfg):
    """Le choix de configuration (radio) : son callback change la config et fait relire les widgets."""
    at.radio(key="sdr_w_config").set_value(cfg)
    return run(at)


# ================================================================
print("\n=== 1. Navigation ===")
src_app = open("streamlit_app.py", encoding="utf-8").read()
src_acc = open("modules/accueil.py", encoding="utf-8").read()
chk("page « Soudures » dans le dictionnaire des pages", '"Soudures": soudures.show' in src_app)
chk("tuile à l'accueil (ligne Acier)", '"page": "Soudures"' in src_acc and src_acc.index('"page": "Soudures"') > src_acc.index("Acier"))
at = run(app())
chk("rendu sans exception, titre et sous-titre", not at.exception and "## Soudures" in md(at) and "EN 1993-1-8" in md(at))
at.button(key="sdr_btn_home").click(); run(at)
chk("🏠 Accueil ramène à l'accueil", at.session_state["page"] == "Accueil")

# ================================================================
print("\n=== 2. Écran ===")
at = run(app())
m = md(at)
chk("le switch : segmented control Eurocode (ELU) | Méthode ELS (bureau), Eurocode par défaut",
    at.segmented_control(key="sdr_w_methode_mode").value == K.EUROCODE and "Efforts de calcul (ELU)" in m)
chk("choix de configuration C1 … C8 (radio sur une ligne, qui passe à la ligne sur tablette), C3 (console) par défaut",
    at.radio(key="sdr_w_config").value == "C3" and len(at.radio(key="sdr_w_config").options) == 8)
chk("schémas techniques du kit des assemblages : trois vues de la console (élévation, coupe horizontale, vue de face), HEB 200 du catalogue, "
    "hachures, cotes e = 150, bande de cordon, point gouvernant",
    "ÉLÉVATION" in m and "COUPE HORIZONTALE" in m and "VUE DE FACE" in m and 'id="sdr_v0"' in m and 'id="sdr_v2"' in m and "HEB 200" in m
    and 'class="ht"' in m and "e = 150" in m and 'class="wb"' in m and "◉ " in m)
chk("encart « la gorge » : σ⊥, τ⊥, τ∥, σ_c, critère", "LA GORGE AU POINT GOUVERNANT" in m and "σ_c = √(σ⊥² + 3(τ⊥² + τ∥²))" in m)
chk("carte des taux : cordon droit et cordon gauche avec leur η", "CARTE DES TAUX" in m and "cordon droit" in m and "cordon gauche" in m)
chk("carte RÉSULTAT : pastille Conforme, η = 61 % (console par défaut sans retours : l_eff = 190), quatre méthodes comparées",
    "Conforme" in pastille(at) and "η = 61 %" in eta(at) and "EC3 directionnelle ◀" in m and "ELS bureau k = 100" in m)
chk("cinq onglets à libellé FIXE", [t.label for t in at.tabs] == ETAPES)
chk("barre d'outils : Accueil, cas, ＋ Cas, copie, Enregistrer, Ouvrir, nom du cas",
    {"sdr_btn_home", "sdr_sel_cas", "sdr_btn_new", "sdr_btn_copy", "sdr_btn_open", "sdr_w_nom"} <= cles(at))
chk("onglet 1 (console) : h, t, e, a, deux cordons, retours, tronçons, nuances",
    {"sdr_w_params_h", "sdr_w_params_t", "sdr_w_params_e", "sdr_w_params_a", "sdr_w_params_deux_cordons", "sdr_w_params_retours",
     "sdr_w_troncons_on", "sdr_w_params_nuance_A", "sdr_w_params_nuance_B"} <= cles(at))
chk("onglet 3 : V et N ; onglet 4 : critère, γM2, k, g, γ_F, bouton « Prendre la valeur Eurocode »",
    {"sdr_w_efforts_V", "sdr_w_efforts_N", "sdr_w_methode_ec", "sdr_w_methode_gamma_M2", "sdr_w_methode_k", "sdr_w_methode_g",
     "sdr_w_methode_gamma_F", "sdr_btn_kq"} <= cles(at))
chk("onglet 5 : tableau du détail par point avec les trois taux", "η dir." in m and "η simpl." in m and "η ELS" in m and "class='g'" in m)

# ================================================================
print("\n=== 3. Source unique : widgets → état → moteur et dessin ===")
at.number_input(key="sdr_w_params_h").set_value(300.0); run(at)
chk("hauteur 300 : état, cote du dessin, η baisse sous 40 %", cas_actif(at)["params"]["h"] == 300.0 and "h = 300" in md(at)
    and ("η = 2" in eta(at) or "η = 3" in eta(at)), eta(at))
at.number_input(key="sdr_w_params_h").set_value(200.0); run(at)
at.number_input(key="sdr_w_efforts_V").set_value(130.0); run(at)
chk("V = 130 kN : Non conforme, η > 100 %, flèche « V = 130,0 kN » sur le dessin",
    "Non conforme" in pastille(at) and cas_actif(at)["efforts"]["V"] == 130.0 and "V = 130,0 kN" in md(at))
at.number_input(key="sdr_w_efforts_V").set_value(60.0); run(at)
at.checkbox(key="sdr_w_params_retours").check(); run(at)
chk("retours d'angle : longueur efficace entière (200 mm), η 61 → 55 % (l'exemple de l'étude)",
    cas_actif(at)["params"]["retours"] is True and "200 mm eff." in md(at) and "η = 55 %" in eta(at), eta(at))
at.checkbox(key="sdr_w_params_retours").uncheck(); run(at)
at.checkbox(key="sdr_w_troncons_on").check(); run(at)
chk("case « par tronçons » : n = 1 par défaut, champs n et l apparus", cas_actif(at)["params"]["troncons"]["n"] == 1
    and {"sdr_w_params_troncons_n", "sdr_w_params_troncons_l"} <= cles(at))
at.number_input(key="sdr_w_params_troncons_n").set_value(2); run(at)
at.number_input(key="sdr_w_params_troncons_l").set_value(60.0); run(at)
chk("2 tronçons de 60 : quatre chaînes dans la carte des taux, message §4.3.2.2 avec l'espacement 80 mm",
    cas_actif(at)["params"]["troncons"] == {"n": 2, "l": 60.0} and "cordon droit · tronçon 2" in md(at) and "espacement 80 mm" in md(at))
at.checkbox(key="sdr_w_troncons_on").uncheck(); run(at)
chk("case décochée : cordon continu de nouveau", cas_actif(at)["params"]["troncons"]["n"] == 0 and "tronçon" not in md(at).split("CARTE DES TAUX")[1][:300])
at.number_input(key="sdr_w_params_a").set_value(2.0); run(at)
chk("gorge 2 mm : « Calcul impossible », message « ne compte pas »", "impossible" in pastille(at) and "ne compte pas" in md(at))
at.number_input(key="sdr_w_params_a").set_value(5.0); run(at)

# ================================================================
print("\n=== 4. Le switch ===")
at.segmented_control(key="sdr_w_methode_mode").set_value(K.ELS); run(at)
m = md(at)
chk("méthode ELS : efforts de service, k = 100 → 98,1 N/mm², l'Eurocode donnerait 138,6 (141 kg)",
    cas_actif(at)["methode"]["mode"] == K.ELS and "Efforts de service" in m and "98.1 N/mm²" in m.replace(",", ".") and "138.6" in m.replace(",", "."))
chk("même console, mêmes efforts : η = 156 % Non conforme selon le bureau (|f| = 764 N/mm contre 98,1 × 5)",
    "Non conforme" in pastille(at) and "η = 156 %" in eta(at) and "ELS bureau k = 100 ◀" in m, eta(at))
at.button(key="sdr_btn_kq").click(); run(at)
chk("« Prendre la valeur Eurocode » : k = 141, η = 110 %", pres(cas_actif(at)["methode"]["k"], 141.2, 1e-3) and "η = 110 %" in eta(at), eta(at))
at.button(key="sdr_btn_k100").click(); run(at)
chk("« Revenir à 100 »", cas_actif(at)["methode"]["k"] == 100.0)
at.segmented_control(key="sdr_w_methode_mode").set_value(K.EUROCODE); run(at)
at.segmented_control(key="sdr_w_methode_ec").set_value(K.SIMPLIFIEE); run(at)
chk("critère simplifié : η = 74 % (764/(207,8 × 5)), « EC3 simplifiée ◀ »", "η = 74 %" in eta(at) and "EC3 simplifiée ◀" in md(at), eta(at))
at.segmented_control(key="sdr_w_methode_ec").set_value(K.DIRECTIONNELLE); run(at)
at.selectbox(key="sdr_w_params_nuance_B").select("S355"); run(at)
chk("nuance B S355 sur A S235 : acier de référence S235 (le plus faible), même η",
    cas_actif(at)["params"]["nuance_B"] == "S355" and "acier de référence S235" in md(at) and "η = 61 %" in eta(at), eta(at))
at.selectbox(key="sdr_w_params_nuance_B").select("S235"); run(at)

# ================================================================
print("\n=== 5. Les huit configurations ===")
attendus = {"C1": ({"sdr_w_params_b", "sdr_w_params_L_rec", "sdr_w_params_frontaux", "sdr_w_params_lateraux"}, "VUE EN PLAN", "Recouvrement"),
            "C2": ({"sdr_w_params_l", "sdr_w_params_tB", "sdr_w_params_deux_cordons", "sdr_w_params_jonction"}, "COUPE A-A", "Jonction en T"),
            "C4": ({"sdr_w_params_bx", "sdr_w_params_by", "sdr_w_params_xV", "sdr_w_params_yV", "sdr_w_params_poteau"}, "VUE EN PLAN", "Gousset"),
            "C5": ({"sdr_w_params_profil", "sdr_w_params_a_f", "sdr_w_params_a_w", "sdr_w_params_semelles", "sdr_w_troncons_semelles_on",
                    "sdr_w_params_jonction_semelles", "sdr_w_params_jonction_ame", "sdr_w_params_tp"}, "VUE DE FACE", "Platine d'about"),
            "C6": ({"sdr_w_params_profil", "sdr_w_params_a", "sdr_w_troncons_on", "sdr_w_params_jonction"}, "VUE DE FACE", "Tube sur platine"),
            "C7": ({"sdr_w_params_b", "sdr_w_params_t", "sdr_w_params_penetration", "sdr_w_params_p"}, "VUE EN PLAN", "Bout à bout"),
            "C8": ({"sdr_w_params_xR", "sdr_w_params_yR", "sdr_w_params_zR"}, "VUE DU PLAN DE JOINT", "Cordons libres")}
for cfg, (attendu, titre, lib) in attendus.items():
    at = changer_config(at, cfg)
    chk(f"{cfg} : champs propres (dont le type de jonction, le support), vue principale titrée « {titre} », calcul conforme avec les défauts",
        attendu <= cles(at) and titre in md(at) and lib in md(at) and 'id="sdr_v0"' in md(at) and "onforme" in pastille(at),
        f"{sorted(attendu - cles(at))} {pastille(at)[:40]}")
at = changer_config(at, "C5")
at.selectbox(key="sdr_w_params_profil").select("HEA 200"); run(at)
chk("C5 : HEA 200 choisi au catalogue, nommé sur le dessin, huit cordons",
    cas_actif(at)["params"]["profil"] == "HEA 200" and "HEA 200 sur platine t 15" in md(at) and "semelle sup. int. droite" in md(at))
at.segmented_control(key="sdr_w_params_jonction_semelles").set_value("pleine"); run(at)
chk("C5 semelles à pleine pénétration : les champs de gorge des semelles disparaissent, l'équivalent en cordons d'angle est dit (a ≥ 0,46·t), "
    "symbole « pleine pénétration » sur le dessin, deux soudures « semelle sup. / inf. » de gorge t_f = 10",
    "sdr_w_params_a_f" not in cles(at) and "sdr_w_params_a_w" in cles(at) and "a ≥ 0,46·t" in md(at) and "pleine pénétration" in md(at)
    and "semelle sup." in md(at) and cas_actif(at)["params"]["jonction_semelles"] == "pleine")
at.segmented_control(key="sdr_w_params_jonction_semelles").set_value("angle"); run(at)
at = changer_config(at, "C6")
at.selectbox(key="sdr_w_params_profil").select("CHS 114.3x5"); run(at)
chk("C3 → C6 : la gorge du widget SUIT l'état (4 mm, défaut du tube), pas l'ancienne valeur 5 de la console",
    at.number_input(key="sdr_w_params_a").value == 4.0 and cas_actif(at)["params"]["a"] == 4.0 and at.session_state["sdr_w_params_a"] == 4.0)
at.checkbox(key="sdr_w_troncons_on").check(); run(at)
at.number_input(key="sdr_w_params_troncons_n").set_value(4); run(at)
at.number_input(key="sdr_w_params_troncons_l").set_value(50.0); run(at)
m = md(at)
chk("C6 : CHS 114,3 × 5 par 4 tronçons de 50 — plus de cordon partout : quatre chaînes, message §4.3.2.2",
    "pourtour · tronçon 4" in m and "§4.3.2.2" in m and "4 tronçons de 50 mm" in m)
at = changer_config(at, "C7")
at.segmented_control(key="sdr_w_params_penetration").set_value("partielle"); run(at)
chk("C7 partielle (p = 5 par défaut sur t = 12) : deux cordons de pénétration, encart « gorge ⊥ à N », coupe cotée p 5, symbole « s 5 »",
    "pénétration face 2" in md(at) and "gorge ⊥ à N" in md(at) and "p 5" in md(at) and "s 5" in md(at))

# ================================================================
print("\n=== 6. Cordons libres ===")
from modules.soudures import elements_depuis_lignes  # noqa: E402
nan = float("nan")
els = elements_depuis_lignes([{"type": "segment", "nom": "", "x1": 0, "y1": 0, "x2": 200, "y2": 0, "xc": nan, "yc": nan, "r": nan, "ang1": nan, "ang2": nan, "a": 5, "cote": 1, "n": nan, "l": nan},
                              {"type": "arc", "nom": "cercle", "x1": nan, "y1": nan, "x2": nan, "y2": nan, "xc": 0, "yc": 100, "r": 40, "ang1": 0, "ang2": 360, "a": 4, "cote": -1.0, "n": 3, "l": 50}])
chk("tableau → éléments : cellules vides remplacées par les défauts, types et côtés entiers, tronçons",
    els[0]["type"] == "segment" and els[0]["x2"] == 200.0 and els[0]["troncons"] == {"n": 0, "l": 60.0} and math.isfinite(els[0]["xc"])
    and els[1]["type"] == "arc" and els[1]["cote"] == -1 and els[1]["troncons"] == {"n": 3, "l": 50.0} and els[1]["nom"] == "cercle")
at = changer_config(at, "C8")
cas_actif(at)["params"]["elements"] = els       # ce que le tableau écrit dans l'état
run(at)
m = md(at)
chk("C8 : segment + cercle par 3 tronçons calculés ensemble, chaînes nommées, arc dessiné",
    "cercle · tronçon 3" in m and "cordon 1" in m and "onforme" in pastille(at), pastille(at)[:40])
at.number_input(key="sdr_w_params_xR").set_value(100.0); run(at)
chk("point de référence déplacé : état et dessin (R)", cas_actif(at)["params"]["xR"] == 100.0 and "R point de référence" in md(at))

# ================================================================
print("\n=== 7. Cas multiples et fichier ===")
at = run(app())
at.number_input(key="sdr_w_params_h").set_value(333.0); run(at)
at.button(key="sdr_btn_new").click(); run(at)
etat = at.session_state["sdr"]
chk("＋ Cas : deuxième cas actif, sélecteur à jour", len(etat["cas"]) == 2 and etat["actif"] == 1 and at.selectbox(key="sdr_sel_cas").value == etat["cas"][1]["nom"])
chk("＋ Cas : le widget de hauteur est réassigné à la valeur du nouveau cas (200), pas 333 (synchronisation au rendu)",
    at.number_input(key="sdr_w_params_h").value == 200.0 and at.session_state["sdr_w_params_h"] == 200.0)
at.button(key="sdr_btn_copy").click(); run(at)
chk("⧉ : copie du cas", len(at.session_state["sdr"]["cas"]) == 3 and "(copie)" in at.session_state["sdr"]["cas"][2]["nom"])
at.button(key="sdr_btn_del").click(); run(at)
chk("🗑️ : suppression, cas précédent actif", len(at.session_state["sdr"]["cas"]) == 2 and at.session_state["sdr"]["actif"] == 1)
at.selectbox(key="sdr_sel_cas").select(at.session_state["sdr"]["cas"][0]["nom"]); run(at)
chk("sélection du premier cas", at.session_state["sdr"]["actif"] == 0)
at.text_input(key="sdr_w_nom").set_value("Console du portique"); run(at)
chk("nom du cas écrit dans l'état et dans le sélecteur", at.session_state["sdr"]["cas"][0]["nom"] == "Console du portique"
    and at.selectbox(key="sdr_sel_cas").value == "Console du portique")
at.button(key="sdr_btn_note").click(); run(at)
chk("📄 Note : la note PDF du cas est produite (deux pages) et son bouton de téléchargement apparaît",
    at.session_state["sdr_note"][0] == "Console du portique" and at.session_state["sdr_note"][1][:4] == b"%PDF"
    and any(b.key == "sdr_dl_note" for b in at.get("download_button")))
texte = ET.vers_json(at.session_state["sdr"])
d2 = ET.depuis_json(texte)
chk("JSON aller-retour : deux cas, même état", len(d2["cas"]) == 2 and d2["cas"][0]["nom"] == "Console du portique" and d2["cas"][0] == at.session_state["sdr"]["cas"][0])

# ================================================================
print("\n=== 8. Étanchéité ===")
at = run(app())
autres = {k for k in cles(at) if not k.startswith("sdr_")}
chk("seules des clés sdr_ (hors boutons partagés)", not autres, str(sorted(autres))[:200])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

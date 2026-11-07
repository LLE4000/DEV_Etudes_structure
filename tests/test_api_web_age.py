# -*- coding: utf-8 -*-
"""L'API du site pour la maturité du béton (api_web/age.py), vérifiée en Python ordinaire.

  1. Inventaire : chaque champ, option, résultat, avertissement et dessin de l'écran Streamlit modules/age_beton.py a
     sa place sur le site (la liste est écrite ici, et chaque entrée est cherchée dans les fragments rendus).
  2. Ce que l'écran affiche = ce que le moteur calcule : le verdict, chaque ligne du résultat, les fenêtres de calcul,
     le sélecteur de classe, le tableau, la courbe (ses 500 points, ses repères), sur des coulages variés.
  3. Les saisies : nombres à la française, refus avec message (l'état reste intact), effets de bord (borne de l'âge,
     classes comparées dans l'ordre du multiselect, relevé collé ou lu, relevé refusé gardé à l'écran).
  4. Le sélecteur de classe : trié par résistance, la classe retenue (✓) et la plus petite qui tient (★) du moteur ;
     un clic retient la classe.
  5. Les éléments de la note (plusieurs coulages), l'échappement de tout texte venu de l'état, du JSON strict.
  6. Pureté : le module du site et ses dessins se chargent et répondent sans streamlit, matplotlib, pandas, requests.

    python3.12 tests/test_api_web_age.py
"""
import copy
import html
import json
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import age as A  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_age as DA  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.maturite import entree as EN  # noqa: E402
from beton.maturite import moteur as M  # noqa: E402
from beton.maturite.ec2 import CLASSES  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    texte = R.traiter(json.dumps(k))
    return json.loads(texte)


def note(*coulages, actif=0):
    return {"version": 1, "coulages": [EN.normaliser(c) for c in coulages], "actif": actif}


def coulage(nom="Coulage 1", **k):
    e = EN.nouvel_element(nom)
    for cle, v in k.items():
        if "__" in cle:
            a, b = cle.split("__")
            e[a][b] = v
        else:
            e[cle] = v
    return EN.normaliser(e)


def saisir(etat, chemin, valeur):
    out = req(module="age", action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/") if isinstance(chemin, str) else chemin,
              valeur=valeur)
    assert out["ok"], out
    return out


def texte(h):
    """Le texte d'un fragment HTML, sans balises ni entités."""
    return html.unescape(re.sub(r"<[^>]+>", "", h))


RELEVE = ("Date;T moy (°C);Source\n2026-11-02;9,5;mesuré\n2026-11-03;8,0;mesuré\n2026-11-04;—;mesuré\n2026-11-05;6,5;mesuré\n"
          "2026-11-06;5,0;mesuré\n2026-11-07;4,0;mesuré\n2026-11-08;6,0;prévision\n2026-11-09;7,5;prévision\n2026-11-10;8,5;prévision\n"
          "2026-11-11;7,0;prolongé\n2026-11-12;7,0;prolongé\n2026-11-13;7,0;prolongé")
TOUT = dict(seuil__afficher=True, optim__afficher=True, comparateur__afficher=True, essai__mpa=18.0, essai__afficher=True, age=7)

# ------------------------------------------------------------------ 1. inventaire de l'écran Streamlit
print("=== 1. Inventaire : chaque élément de l'écran Streamlit a sa place sur le site ===")
r_const = req(module="age", action="calculer", etat=note(coulage(**TOUT, date_coulage="2026-11-02")))
e_rel = note(coulage(temperature__mode="releve", temperature__releve=RELEVE, temperature__lieu="Namur", **TOUT))
r_rel = req(module="age", action="calculer", etat=e_rel)
e_av = note(coulage(avance=True, Ea_kJ=40.0, seuil__mode="mpa", seuil__mpa=12.5))
r_av = req(module="age", action="calculer", etat=e_av)
chk("les trois notes de l'inventaire se calculent", r_const["ok"] and r_rel["ok"] and r_av["ok"], [r_const.get("erreur"), r_rel.get("erreur"), r_av.get("erreur")])


def tout(r):
    return " ".join([r["donnees"], r["resultat"], r["volets"], r["phrase"]] + [v["svg"] + v["titre"] + v.get("legende", "") for v in r["vues"]])


def champ(r, chemin):
    return f'data-chemin="{chemin}"' in r["donnees"]


# (élément de l'écran Streamlit, où il est sur le site, preuve)
INVENTAIRE = [
    ("titre « Évolution de la résistance du béton — EC2 » et légende EN 1992-1-1 §3.1.2 / annexe B", "pied du résultat, fenêtre ①",
     "EC2 §3.1.2 et annexe B" in r_const["resultat"] and "EN 1992-1-1 §3.1.2" in r_const["volets"]),
    ("⚙️ paramètres avancés (toggle)", "section 8 Méthode, case", champ(r_const, "avance")),
    ("Ea [kJ/mol] (20 à 60)", "section 8, champ grisé sans ⚙️", champ(r_const, "Ea_kJ") and 'data-min="20.0" data-max="60.0"' in r_const["donnees"]),
    ("Classe béton (7 classes, C30/37)", "section 1, menu", all(f'value="{c}"' in r_const["donnees"] for c in CLASSES)),
    ("Classe ciment (R, N, S) et sa légende CEM… — s =", "section 1, segments + aide",
     champ(r_const, "ciment") and "CEM 32,5R · CEM 42,5N — s = 0,25" in r_const["donnees"]),
    ("Date de coulage", "section 1, champ JJ/MM/AAAA", champ(r_const, "date_coulage") and 'value="02/11/2026"' in r_const["donnees"]),
    ("Température : Manuel / Météo", "section 2, segments Constante / Relevé", champ(r_const, "temperature/mode")),
    ("T moyenne (°C) (−5 à 60)", "section 2, champ", champ(r_const, "temperature/T") and 'data-min="-5.0" data-max="60.0"' in r_const["donnees"]),
    ("Lieu du chantier", "section 2 (relevé), champ", champ(r_rel, "temperature/lieu")),
    ("Coulage / Fin (décoffrage) / Récupérer la météo (Open-Meteo)", "section 2 : le relevé SE SAISIT (zone de texte, fichier)",
     '<textarea data-chemin="temperature/releve"' in r_rel["donnees"] and 'type="file" data-chemin="temperature/fichier"' in r_rel["donnees"]),
    ("badge « lieu — n j mesurés · n j de prévision · n j prolongés »", "section 2, aide",
     "Namur — 6 j mesurés · 3 j de prévision · 3 j prolongés" in texte(r_rel["donnees"])),
    ("mesures T min / T moy / T max", "section 2, rangée de mesures", all(x in texte(r_rel["donnees"]) for x in ("T min", "T moy", "T max"))),
    ("avertissement de prolongation (au-delà de l'horizon)", "résultat, message", "prolongé(s) au-delà des prévisions" in r_rel["resultat"]),
    ("légende « n jour(s) sans donnée, remplacé(s) par la moyenne »", "résultat, message", "sans donnée, remplacé(s) par la moyenne" in r_rel["resultat"]),
    ("tableau « Données météo » (Date, T moy, Source)", "vue Tableau (Date, T, Source…)",
     any(v["cle"] == "tableau" and "Source" in v["svg"] and "prévision" in v["svg"] for v in r_rel["vues"])),
    ("mesures Jours · t_e · t_e défavorable", "section 2, rangée de mesures", "t<sub>e</sub> défav." in r_rel["donnees"]),
    ("① Âge (jours réels), curseur 1 à 28 (1 à n jours du relevé)", "section 3, champ entier + âges usuels",
     'data-chemin="age" data-sorte="entier"' in r_const["donnees"] and 'data-max="28"' in r_const["donnees"] and 'data-max="12"' in r_rel["donnees"]),
    ("① bloc « fck(t j à T °C) : x MPa · y % de fck(28) »", "verdict + ligne « Résistance à t j »",
     "Résistance à 7 j" in r_const["resultat"] and "72 % de f<sub>ck</sub>" in r_const["resultat"]),
    ("① légende t_e = … (source) — ≤ 3 j : hors domaine", "verdict (t_e), message grave hors domaine", "t<sub>e</sub> = 7,0 j" in r_const["resultat"]),
    ("① légende défavorable (jours prévus − 3 °C)", "ligne « Défavorable à t j » + message", "Défavorable à 7 j" in r_rel["resultat"]),
    ("② Cible : Pourcentage / Résistance (MPa)", "section 4, segments", champ(r_const, "seuil/mode")),
    ("② Pourcentage 30 à 100 par 5", "section 4, menu", all(f'value="{p}"' in r_const["donnees"] for p in range(30, 101, 5))),
    ("② Résistance (MPa) 1 à fck", "section 4, champ borné", champ(r_av, "seuil/mpa") and 'data-max="30"' in r_av["donnees"]),
    ("② bloc « Seuil 75 % = 22,5 MPa — après t j — le JJ/MM/AAAA » / « non atteignable »", "ligne « Seuil à t j » (cible, temps, date) + fenêtre",
     "22,5 MPa · après 7,9 j · 09/11/2026" in texte(r_const["resultat"]) and "le 09/11/2026" in r_const["volets"]
     and "seuil 75 % = 22,5 MPa" in r_const["volets"]),
    ("② Afficher sur le graphique", "section 4, case", champ(r_const, "seuil/afficher")),
    ("③ Résistance (MPa), Délai (jours)", "section 5, champs", champ(r_const, "optim/mpa") and champ(r_const, "optim/delai")),
    ("③ bloc « Classe minimale : X (actuelle) » — « f MPa à d j » / « Aucune classe… »", "ligne « Classe minimale » ★, sélecteur, fenêtre",
     'data-pk="classe"' in r_const["resultat"] and "<b>Classe minimale</b>" in r_const["resultat"] and "C35/45 · 30,8 MPa" in texte(r_const["resultat"])
     and "→ C35/45" in r_const["volets"]),
    ("③ Afficher sur le graphique", "section 5, case", champ(r_const, "optim/afficher")),
    ("④ Classes à comparer (multiselect)", "section 6, sept cases", all(champ(r_const, f"comparateur/classes/{i}") for i in range(7))),
    ("④ Ciment (comparateur)", "section 6, segments", champ(r_const, "comparateur/ciment")),
    ("④ Afficher sur le graphique", "section 6, case", champ(r_const, "comparateur/afficher")),
    ("⑤ Mesure (MPa), Âge (j), T cure (°C)", "section 7, champs", all(champ(r_const, f"essai/{k}") for k in ("mpa", "age", "T"))),
    ("⑤ blocs « Classe estimée ≈ X (fck28 ≈ f MPa) » et « Cohérent / Au-dessus / Sous — écart » ", "ligne « Essai à t j », message, fenêtre",
     "<b>Essai à 7 j</b>" in r_const["resultat"] and "≈ C25/30 · fck,28 ≈ 25,4 MPa" in texte(r_const["resultat"]) and "sous C30/37 (écart −15 %)" in texte(r_const["resultat"])
     and "sous C30/37" in r_const["volets"]),
    ("⑤ Afficher sur le graphique", "section 7, case", champ(r_const, "essai/afficher")),
    ("figure : courbe de référence, défavorable, comparateur, optimisation, point, seuil, essai, t_e = 28 j, légende",
     "vue Courbe (SVG)", all(x in r_rel["vues"][0]["svg"] for x in ("comparateur", "classe minimale", "seuil ②", "essai ⑤", "défavorable"))
     and "= 28 j" in r_const["vues"][0]["svg"]),
    ("expander « Formules EC2 » (β_cc, f_cm, t_e, s, E_a/R, domaine 3 < t_e < 28)", "fenêtre ①, « Les formules de l'EC2 »",
     "Domaine de validité" in r_const["volets"] and "4 000 K" in r_const["volets"]),
    ("export PDF (données prêtes, pas de PDF dans Streamlit)", "bouton Note PDF grisé (hors périmètre)", True),
]
manque = [(n, ou) for n, ou, ok in INVENTAIRE if not ok]
chk(f"inventaire : les {len(INVENTAIRE)} éléments de l'écran Streamlit sont sur le site", not manque, manque)

# ------------------------------------------------------------------ 2. ce que l'écran affiche = ce que le moteur calcule
print("\n=== 2. L'écran affiche ce que le moteur calcule ===")
LIGNE = re.compile(r'<div class="el( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)">([^<]+)</span>', re.S)
CAS = [coulage(), coulage(**TOUT), coulage(classe="C20/25", ciment="S", age=2, temperature__T=5.0),
       coulage(classe="C50/60", ciment="R", age=1, temperature__T=60.0), coulage(temperature__T=-5.0, seuil__pct=100, optim__mpa=60.0, optim__delai=2.0),
       coulage(essai__mpa=33.0, essai__age=35.0), coulage(avance=True, Ea_kJ=45.0, temperature__T=8.0, age=10),
       coulage(temperature__mode="releve", temperature__releve=RELEVE, age=12, **{k: v for k, v in TOUT.items() if k != "age"}),
       coulage(temperature__mode="releve", temperature__releve="1\n0,5\n-1\n2\n3", age=5),
       coulage(temperature__T=-5.0, optim__delai=0.5, age=1)]
fautes = []
for k, c in enumerate(CAS):
    r = req(module="age", action="calculer", etat=note(c))
    if not r["ok"]:
        fautes.append((k, r.get("erreur")))
        continue
    m = M.calculer(c)
    a, s, o, es = m["age"], m["seuil"], m["optim"], m["essai"]
    lignes = {nom: (bool(gov), niv, txt) for gov, nom, niv, txt in LIGNE.findall(r["resultat"])}
    # le verdict : f_ck(t) au dixième, le pourcentage
    if f'{C.fr(a["fck"], 1).replace("-", "−")}<small>' not in r["resultat"]:
        fautes.append((k, "verdict", a["fck"]))
    # ① : le pourcentage de f_ck, la ligne gouvernante
    l1 = lignes.get(f"Résistance à {a['t']} j")
    if not l1 or not l1[0] or l1[2] != f"{C.pc(C.pct(max(a['pct'], 0) / 100))} %":
        fautes.append((k, "①", l1, a["pct"]))
    if f"{C.fr(a['fck'], 2).replace('-', '−')} MPa" not in r["resultat"]:
        fautes.append((k, "① MPa"))
    # ② : la cible à l'âge choisi, le temps d'obtention
    l2 = lignes.get(f"Seuil à {a['t']} j")
    t2 = s["taux_age"]
    if not l2 or l2[2] != f"{C.pc(C.pct(min(max(t2, 0), 9.99)) if t2 is not None else C.pct(9.99))} %":
        fautes.append((k, "②", l2, t2))
    if (s["t"] is None) != ("non atteignable" in r["resultat"]):
        fautes.append((k, "② atteignable"))
    # ③ : la classe minimale et sa résistance au délai (le bloc de l'écran), ★ ; « aucune », en rouge, sinon
    l3 = lignes.get("Classe minimale")
    t3 = texte(r["resultat"])
    if o["classe"]:
        opt = next(x for x in o["classes"] if x["classe"] == o["classe"])
        if not l3 or l3[1] == "r" or f"{o['classe']} · {C.fr(o['fck'], 1).replace('-', '−')} MPa" not in t3 \
                or l3[2] != f"{C.pc(A._taux_classe(opt))}{C.NBSP}%" or 'class="leger"' not in r["resultat"]:
            fautes.append((k, "③", l3, o["classe"]))
    elif not l3 or l3[1] != "r" or "aucune ne suffit" not in t3:
        fautes.append((k, "③ aucune", l3))
    # ⑤ : la ligne n'existe qu'avec une mesure
    if es["actif"] != (f"Essai à {C.nb(es['age'])} j" in r["resultat"]) or (es["actif"] and f"≈ {es['classe']}" not in r["resultat"]):
        fautes.append((k, "⑤"))
    # les avertissements du moteur, tous écrits
    for av in m["avertissements"]:
        if C.symb_math(av["message"]) not in r["resultat"]:
            fautes.append((k, "avertissement", av["code"]))
    # les vues : la courbe, le relevé seulement avec un relevé exploitable, le tableau ligne par ligne
    cles = [v["cle"] for v in r["vues"]]
    if cles != ["courbe"] + (["releve"] if m["temperature"]["source"] == "releve" else []) + ["tableau"]:
        fautes.append((k, "vues", cles))
    tab = next(v["svg"] for v in r["vues"] if v["cle"] == "tableau")
    if tab.count("<tr") - 1 != len(m["tableau"]["lignes"]) or tab.count('<tr class="on') != 1:
        fautes.append((k, "tableau", tab.count("<tr")))
    # le résumé : une consultation, nommée avec sa résistance
    if r["resume"]["niveau"] != "i" or r["resume"]["taux"] is not None or C.fr(a["fck"], 1).replace("-", "−") not in r["resume"]["gouvernant"]:
        fautes.append((k, "résumé", r["resume"]))
    # le JSON du routeur est strict (pas de NaN ni d'infini, même quand f_ck ≤ 0)
    try:
        json.loads(R.traiter(json.dumps({"module": "age", "action": "calculer", "etat": note(c)})), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except ValueError as x:
        fautes.append((k, "JSON", x))
chk(f"{len(CAS)} coulages : verdict, lignes ①②③⑤, avertissements, vues, tableau, résumé = le moteur ; JSON strict", not fautes, fautes[:4])

# la courbe : les 500 points de la référence (abscisses 1 à 40 j), les repères demandés et eux seuls
c = coulage(**TOUT)
m = M.calculer(c)
svg = DA.courbe("t", m)
chemin = re.search(r'<path d="(M[^"]+)" fill="none" stroke="#2e6fb0" stroke-width="2.6"', svg)
pts = chemin.group(1)[1:].split(" L") if chemin else []
chk("courbe : la référence est tracée avec ses 500 points (np.linspace(1, 40, 500) du module)", len(pts) == 500, len(pts))
svg0 = DA.courbe("t", M.calculer(coulage()))
chk("courbe : sans case cochée, ni seuil, ni classe minimale, ni comparateur, ni essai (comme la figure du module)",
    not any(x in svg0 for x in ("seuil ②", "classe minimale", "comparateur", "essai ⑤")) and all(x in svg for x in ("seuil ②", "classe minimale", "comparateur", "essai ⑤")))
chk("courbe : le point de l'âge choisi porte « f MPa à t j » (l'annotation du module)", f"{DA.fr(m['age']['fck'], 1)} MPa à 7 j" in svg)
chk("courbe : chaque classe comparée nommée au bout de sa courbe (la classe du coulage n'est pas retracée)",
    all(f">{cl}<" in svg or f">{cl} ③<" in svg for cl in CLASSES) and svg.count(">C30/37<") == 1)
svg_loin = DA.courbe("t", M.calculer(coulage(optim__delai=200.0, optim__mpa=30.0, optim__afficher=True)))
chk("courbe : un délai au-delà de 120 j est signalé au bord (flèche) au lieu d'écraser l'axe", "200,0 j →" in svg_loin)
chk("courbe : le domaine t_e ≤ 3 j est grisé et légendé", "hors du domaine de l'EC2" in svg and 'fill="#F1F2F5"' in svg)

# ------------------------------------------------------------------ 3. les saisies
print("\n=== 3. Les saisies : lecture, refus, effets de bord ===")
e0 = A.nouvel_etat()
o = saisir(e0, "temperature/T", "12,5")
chk("« 12,5 » °C est lu 12.5 et recalcule", o["etat"]["coulages"][0]["temperature"]["T"] == 12.5 and o["message"] is None)
o = saisir(e0, "optim/delai", "1 000")
chk("« 1 000 » (espace des milliers) est lu 1000", o["etat"]["coulages"][0]["optim"]["delai"] == 1000.0 and o["message"] is None)
REFUS = [("temperature/T", "-6", "Température de cure"), ("temperature/T", "61", "Température de cure"), ("age", "0", "Âge"),
         ("age", "29", "Âge"), ("age", "7,5", "entier"), ("Ea_kJ", "19", "Énergie d'activation"), ("seuil/mpa", "31", "dépasse f_ck"),
         ("seuil/mpa", "0,5", "Résistance du seuil"), ("optim/mpa", "0,9", "Résistance cible"), ("optim/delai", "0,4", "Délai"),
         ("essai/mpa", "-1", "Mesure"), ("essai/age", "0,4", "Âge de l'essai"), ("essai/T", "70", "T de cure"), ("date_coulage", "demain", "date"),
         ("temperature/T", "abc", "illisible"), ("classe", "C16/20", "classe"), ("ciment", "X", "ciment"), ("seuil/pct", "72", "pas de 5"),
         ("temperature/mode", "météo", "cure"), ("inconnu", "1", "Champ inconnu"),
         # saisies absurdes : l'écran n'a pas de borne haute, le site en met une (10 ans, 200 MPa) — au-delà, un délai de 1e308 j
         # déborde en infini à 60 °C et le JSON du navigateur ne le porte pas
         ("optim/delai", "1e308", "Délai"), ("optim/delai", "3651", "Délai"), ("essai/age", "1e30", "Âge de l'essai"),
         ("optim/mpa", "201", "Résistance cible"), ("essai/mpa", "1e308", "Mesure"), ("temperature/T", "1e400", "illisible"),
         ("Ea_kJ", "nan", "illisible"), ("Ea_kJ", "inf", "illisible"), ("age", "1e308", "Âge"), ("optim/delai", True, "illisible"),
         ("comparateur/classes/9", True, "Classe comparée"), ("comparateur/classes/-1", True, "Classe comparée"),
         ("comparateur/classes/x", True, "Classe comparée")]
mal = []
for chemin, v, mot in REFUS:
    o = saisir(e0, chemin, v)
    if not o["message"] or mot.lower() not in o["message"].lower() or o["etat"] != e0:
        mal.append((chemin, v, o["message"]))
chk(f"{len(REFUS)} saisies hors du domaine de l'écran refusées avec un message qui nomme le champ ; la note reste intacte", not mal, mal[:3])
o = saisir(e0, "seuil/pct", "50")
chk("le pourcentage du seuil se choisit dans le menu (« 50 » → 50)", o["etat"]["coulages"][0]["seuil"]["pct"] == 50)
o = saisir(e0, "date_coulage", "2/11/2026")
chk("la date de coulage JJ/MM/AAAA est lue, réécrite au format du moteur et affichée JJ/MM/AAAA",
    o["etat"]["coulages"][0]["date_coulage"] == "2026-11-02" and 'value="02/11/2026"' in o["donnees"])
o = saisir(o["etat"], "date_coulage", "")
chk("une date effacée revient à « sans date »", o["etat"]["coulages"][0]["date_coulage"] is None)
# les classes comparées : l'ordre du multiselect (la classe cochée va au bout)
o = saisir(e0, "comparateur/classes/0", False)
o = saisir(o["etat"], "comparateur/classes/0", True)
cl = o["etat"]["coulages"][0]["comparateur"]["classes"]
chk("décocher puis recocher C20/25 la remet AU BOUT de la liste (comme le multiselect : l'ordre fixe les couleurs)", cl == CLASSES[1:] + ["C20/25"], cl)
o2 = saisir(o["etat"], "comparateur/classes/0", True)
chk("recocher une classe déjà comparée ne change rien", o2["etat"] == o["etat"])
# la borne de l'âge suit le relevé
e1 = note(coulage(age=20))
o = saisir(e1, "temperature/mode", "releve")
o = saisir(o["etat"], "temperature/releve", "10\n11\n12\n13\n14")
chk("un relevé de 5 jours ramène l'âge 20 à min(28, 5) = 5 (la borne du curseur de l'écran)", o["etat"]["coulages"][0]["age"] == 5, o["etat"]["coulages"][0]["age"])
o = saisir(o["etat"], "temperature/mode", "constante")
chk("revenir à la température constante garde l'âge (5 ≤ 28)", o["etat"]["coulages"][0]["age"] == 5)
# un relevé refusé : gardé à l'écran avec son erreur, le calcul garde le précédent
e2 = note(coulage(temperature__mode="releve", temperature__releve=RELEVE, age=6))
o = saisir(e2, "temperature/releve", RELEVE.replace("6,5", "six"))
c2 = o["etat"]["coulages"][0]
chk("relevé mal écrit : refusé avec la ligne et ce qu'elle porte ; le texte reste à l'écran avec l'erreur ; le calcul garde l'ancien",
    o["message"] and "ligne 5" in o["message"] and "six" in o["message"] and c2["temperature"]["releve"] == RELEVE
    and A.BROUILLON in c2 and "six" in o["donnees"] and 'class="msg grave"' in o["donnees"], o["message"])
o3 = saisir(o["etat"], "temperature/releve", RELEVE.replace("6,5", "6,0"))
chk("le relevé corrigé est pris et l'erreur disparaît", A.BROUILLON not in o3["etat"]["coulages"][0] and o3["message"] is None
    and "6,0" in o3["etat"]["coulages"][0]["temperature"]["releve"])
o4 = req(module="age", action="calculer", etat=json.loads(json.dumps(o["etat"])))
chk("le relevé refusé survit à l'enregistrement (.stab) et à la relecture", o4["ok"] and A.BROUILLON in o4["etat"]["coulages"][0])
o = saisir(e0, ["temperature", "fichier"], {"nom": "releve_<b>.csv", "texte": RELEVE})
c3 = o["etat"]["coulages"][0]
chk("un fichier lu : le relevé est pris, le mode passe à « relevé », le nom du fichier est gardé (et échappé à l'écran)",
    c3["temperature"]["mode"] == "releve" and c3["temperature"]["releve"] == RELEVE and c3["temperature"]["fichier"] == "releve_<b>.csv"
    and "releve_&lt;b&gt;.csv" in o["donnees"] and "releve_<b>" not in o["donnees"])
o = saisir(note(coulage(temperature__mode="releve")), ["temperature", "fichier"], {"nom": "faux.csv", "texte": "douze"})
chk("un fichier illisible est refusé avec son nom et sa ligne", o["message"] and "ligne 1" in o["message"] and "Fichier « faux.csv »" in o["donnees"])
# ⚙️ : le champ E_a grisé tant que la case n'est pas cochée
chk("E_a est grisé sans ⚙️ et s'ouvre avec", 'data-chemin="Ea_kJ"' in A.formulaire(e0, M.calculer(e0["coulages"][0]))
    and re.search(r'data-chemin="Ea_kJ"[^>]*disabled', req(module="age", action="nouveau")["donnees"])
    and not re.search(r'data-chemin="Ea_kJ"[^>]*disabled', saisir(e0, "avance", True)["donnees"]))

# ------------------------------------------------------------------ 4. le sélecteur de classe
print("\n=== 4. Le sélecteur de classe (③) ===")
fautes = []
for c in CAS:
    s = req(module="age", action="selecteur", etat=note(c), role="classe")
    cands = M.candidats_classes(c)
    noms = re.findall(r'data-choix="([^"]+)"', s["html"])
    cur = re.findall(r'class="pk-row cur[^"]*" data-choix="([^"]+)"', s["html"])
    opt = re.findall(r'class="pk-row[^"]* opt" data-choix="([^"]+)"', s["html"])
    if noms != CLASSES or cur != [c["classe"]] or opt != [x["classe"] for x in cands if x["optimal"]]:
        fautes.append((noms, cur, opt))
    for x in cands:
        if f'{C.fr(x["fck"], 1).replace("-", "−")}</span>' not in s["html"]:
            fautes.append(("f", x["classe"]))
chk(f"{len(CAS)} sélecteurs : les sept classes par résistance, ✓ la classe du coulage, ★ la plus petite qui tient (moteur), f_ck au délai",
    not fautes, fautes[:3])
s = req(module="age", action="choisir", etat=note(coulage()), role="classe", nom="C35/45")
chk("un clic retient la classe : C35/45 devient la classe du coulage, et la ligne ③ dit qu'elle suffit",
    s["ok"] and s["etat"]["coulages"][0]["classe"] == "C35/45" and "classe du coulage C35/45 : 30,8 MPa à 14 j — elle suffit" in s["resultat"])
s = req(module="age", action="choisir", etat=note(coulage()), role="classe", nom="C99/99")
chk("une classe inconnue est refusée (erreur lisible)", not s["ok"] and "C99/99" in s["erreur"])
s = req(module="age", action="selecteur", etat=note(coulage(optim__mpa=60.0, optim__delai=2.0)), role="classe")
chk("aucune classe ne tient : pas d'étoile, toutes en rouge", s["ok"] and " opt" not in s["html"] and s["html"].count('class="chip r"') == 7)

# ------------------------------------------------------------------ 5. éléments, échappement
print("\n=== 5. Les coulages de la note, l'échappement ===")
e = A.nouvel_etat()
a1 = req(module="age", action="ajouter", etat=e)
a2 = req(module="age", action="dupliquer", etat=saisir(a1["etat"], "classe", "C45/55")["etat"])
chk("ajouter, copier : trois coulages, la copie ouverte porte la classe copiée",
    len(a2["etat"]["coulages"]) == 3 and a2["etat"]["actif"] == 2 and a2["etat"]["coulages"][2]["classe"] == "C45/55"
    and a2["etat"]["coulages"][2]["nom"] == "Coulage 2 (copie)")
a3 = req(module="age", action="supprimer", etat=a2["etat"], index=0)
chk("supprimer le premier : deux coulages, l'ouvert reste le même", len(a3["etat"]["coulages"]) == 2 and a3["etat"]["coulages"][a3["etat"]["actif"]]["nom"] == "Coulage 2 (copie)")
a4 = req(module="age", action="activer", etat=a3["etat"], index=0)
chk("le résumé de la note nomme le coulage ouvert", a4["resume"]["gouvernant"].startswith("Coulage 2 ·"))
PIEGE = '<img src=x onerror=alert(1)>'
fuites = []
for chemin in ("nom", "temperature/lieu"):
    base = note(coulage(temperature__mode="releve", temperature__releve=RELEVE, age=6))
    o = saisir(base, chemin, PIEGE)
    for k_, h in (("données", o["donnees"]), ("résultat", o["resultat"]), ("fenêtres", o["volets"]), ("éléments", o["elements"]),
                  ("vues", "".join(v["svg"] for v in o["vues"])), ("phrase", o["phrase"])):
        if "<img" in h:
            fuites.append((chemin, k_))
o = saisir(note(coulage(temperature__mode="releve")), "temperature/releve", "12\n" + PIEGE)
if "<img" in o["donnees"]:
    fuites.append(("relevé refusé", "données"))
chk("un nom, un lieu, un relevé refusé piégés sont échappés dans tous les fragments", not fuites, fuites)
abime = req(module="age", action="calculer", etat={"version": 1, "coulages": [{"classe": "C99"}], "actif": 0})
chk("un coulage abîmé (.stab) rend une erreur lisible", not abime["ok"] and "classe" in abime["erreur"])

# ------------------------------------------------------------------ 5 bis. ce que la relecture a ajouté
print("\n=== 5 bis. Le verdict de l'essai, le signe moins, la courbe lisible, les valeurs extrêmes admises ===")
# ⑤ : le second bloc de l'écran (cohérent / au-dessus / sous, l'écart) et la lecture en moyenne (défaut non conservatif)
mal = []
for mesure, verdict, grave in ((21.6, "cohérent avec C30/37", False), (35.0, "au-dessus de C30/37", False), (12.0, "sous C30/37", True)):
    c = coulage(essai__mpa=mesure, essai__age=7.0)
    r = req(module="age", action="calculer", etat=note(c))
    m = M.calculer(c)["essai"]
    msgs = [x for x in re.findall(r'<div class="msg( grave)?">(.*?)</div>', r["resultat"]) if "Essai ⑤" in x[1]]
    att = f"{verdict} (écart {m['ecart']:+.0f} %)".replace("(écart -", "(écart −")
    if len(msgs) != 2 or att not in texte(msgs[0][1]) or (msgs[0][0] == " grave") != grave or msgs[1][0] \
            or f"fck,28 ≈ {C.fr(m['fck28_est_moyenne'], 1)} MPa." not in texte(msgs[1][1]) or "à trancher par le bureau" not in msgs[1][1]:
        mal.append((mesure, msgs))
chk("⑤ : le verdict de l'écran (cohérent / au-dessus / sous, écart ; « sous » en grave) et, à part, la lecture en moyenne à trancher",
    not mal, mal)
chk("⑤ : 21,6 MPa à 7 j — le module dit C30/37 « cohérent », la lecture en moyenne donne 19,7 MPa (défaut non conservatif, écrit)",
    "19,7 MPa" in req(module="age", action="calculer", etat=note(coulage(essai__mpa=21.6, essai__age=7.0)))["resultat"])
r = req(module="age", action="calculer", etat=note(coulage(temperature__T=-5.0, age=3)))
chk("une température négative s'écrit avec le vrai signe moins (verdict, phrase, éléments, fenêtres)",
    "−5 °C" in r["resultat"] and "−5 °C" in r["phrase"] and "−5 °C" in r["elements"] and "-5 °C" not in r["resultat"] + r["phrase"] + r["elements"] + r["volets"])
# la courbe : légende AU-DESSUS du tracé, noms entiers dans le dessin, axe des résistances serré sous zéro, étiquette ① sans rien cacher
svg = DA.courbe("t", M.calculer(coulage(**TOUT)))
y_trace = float(re.search(r'<rect x="[^"]+" y="([^"]+)" width="[^"]+" height="[^"]+" fill="#FBFCFE"/>', svg).group(1))
y_leg = [float(y) for y in re.findall(r'<text x="[^"]+" y="([^"]+)"[^>]*>[^<]*(?:seuil ②|essai ⑤|comparateur|classe minimale)', svg)]
chk("courbe : la légende est au-dessus du tracé (elle ne cache plus les courbes des petites classes)", y_leg and max(y_leg) < y_trace, (y_leg, y_trace))
largeur = float(re.search(r'viewBox="0 0 ([0-9.]+) ', svg).group(1))
noms = re.findall(r'<text x="([0-9.]+)" y="[^"]+" text-anchor="start" font-size="([0-9.]+)"[^>]*font-weight="650">([^<]+)</text>', svg)
coupe = [(n, float(x) + DA._largeur(n, float(t), True)) for x, t, n in noms if float(x) + DA._largeur(n, float(t), True) > largeur]
chk("courbe : les noms au bout des courbes (« C35/45 ③ ») tiennent dans le dessin", noms and not coupe, coupe)
svg_f = DA.courbe("t", M.calculer(coulage(temperature__T=-5.0, age=1)))
chk("courbe : une résistance négative (−4,0 MPa à 1 j, −5 °C) n'étire pas l'axe jusqu'à −10 : il s'arrête juste dessous",
    ">−5<" in svg_f and ">−10<" not in svg_f and "−4,0 MPa à 1 j" in svg_f)
m_t = M.calculer(coulage(**TOUT))
rect = re.search(r'<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="([0-9.]+)" rx="6"', svg)
bx, by, bw, bh = (float(v) for v in rect.groups())
carre = re.search(r'<rect x="([0-9.]+)" y="([0-9.]+)" width="12" height="12" fill="#b45309"', svg)
cx, cy = float(carre.group(1)) + 6, float(carre.group(2)) + 6
chk("courbe : l'étiquette du point ① ne couvre pas le repère du seuil voisin (7,9 j contre 7 j)",
    not (bx - 6 <= cx <= bx + bw + 6 and by - 6 <= cy <= by + bh + 6), (bx, by, bw, bh, cx, cy))
# ② : jours froids d'abord, le seuil annoncé à la température moyenne est trop tôt — avertissement grave, et la fenêtre le montre
froid_chaud = note(coulage(temperature__mode="releve", temperature__releve="\n".join(["2"] * 7 + ["20"] * 7), seuil__pct=50, age=5))
r = req(module="age", action="calculer", etat=froid_chaud)
chaud_froid = req(module="age", action="calculer", etat=note(coulage(temperature__mode="releve", temperature__releve="\n".join(["20"] * 7 + ["2"] * 7),
                                                                     seuil__pct=50, age=5)))
chk("② NON CONSERVATIF (jours froids d'abord) : avertissement grave « il n'est atteint qu'au jour 8 », et la fenêtre du seuil le montre ; rien quand "
    "les jours chauds viennent d'abord",
    re.search(r'<div class="msg grave">Seuil ② NON CONSERVATIF.*?f<sub>ck</sub> au jour 5 = 11,36 MPa &lt; 15,0 MPa : il n\'est atteint qu\'au jour 8', r["resultat"])
    and "premier jour où f" in r["volets"] and "jour 8" in texte(r["volets"]) and "NON CONSERVATIF" not in chaud_froid["resultat"],
    re.findall(r'<div class="msg grave">(Seuil[^<]*)', r["resultat"]))
# les valeurs extrêmes ADMISES restent du JSON strict (pas d'infini) : 10 ans à 60 °C, 200 MPa
fautes = []
for k in (dict(temperature__T=60.0, optim__delai=3650.0, optim__afficher=True), dict(temperature__T=60.0, essai__age=3650.0, essai__mpa=200.0, essai__afficher=True),
          dict(optim__mpa=200.0, optim__afficher=True, seuil__mode="mpa", seuil__mpa=30.0), dict(temperature__T=-5.0, essai__mpa=200.0, essai__age=0.5)):
    try:
        t_ = R.traiter(json.dumps({"module": "age", "action": "calculer", "etat": note(coulage(**k))}))
        if not json.loads(t_, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))["ok"]:
            fautes.append((k, "refus"))
    except ValueError as x:
        fautes.append((k, x))
chk("les valeurs extrêmes admises (délai et âge d'essai de 10 ans à 60 °C, 200 MPa) se calculent en JSON strict", not fautes, fautes)

# ------------------------------------------------------------------ 6. pureté
print("\n=== 6. Pureté : le module du site se charge et répond sans streamlit ni matplotlib ===")
SOUS = r'''
import json, sys
INTERDITS = ("streamlit", "pandas", "reportlab", "matplotlib", "scipy", "requests", "fitz", "PIL")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
sys.path.insert(0, sys.argv[1])
from api_web import routeur as R
q = json.loads(sys.stdin.read())
out = [json.loads(R.traiter(json.dumps(x)))["ok"] for x in q]
print(json.dumps({"ok": out, "interdits": sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)}))
'''
reqs = [{"module": "age", "action": "nouveau"}, {"module": "age", "action": "calculer", "etat": e_rel},
        {"module": "age", "action": "selecteur", "etat": e_rel, "role": "classe"}]
pr = subprocess.run([sys.executable, "-I", "-c", SOUS, RACINE], input=json.dumps(reqs), capture_output=True, text=True, check=False)
res = json.loads(pr.stdout) if pr.returncode == 0 else {"ok": [], "interdits": [pr.stderr[-300:]]}
chk("le module « age » répond (nouveau, relevé, sélecteur) avec streamlit, matplotlib, pandas, requests REFUSÉS", res["ok"] == [True] * 3 and not res["interdits"], res)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

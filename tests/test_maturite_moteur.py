# -*- coding: utf-8 -*-
"""Moteur « Maturité du béton » (beton/maturite) : la PARITÉ avec l'écran Streamlit modules/age_beton.py v3.2, qui
reste la référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Données : listes, constantes et valeurs par défaut — nouvel_element() est l'écran au PREMIER affichage, lu en
     rejouant show() sans aucune saisie (chaque widget rend sa valeur par défaut).
  2. À la main : β_cc, f_ck(t), âge équivalent, seuil, essai recalculés ici avec math seul, formules écrites.
  3. Parité (plus de 60 coulages) : chaque coulage est traduit en saisies de l'écran, show() est exécuté avec un faux
     « st » qui rend ces saisies et ENREGISTRE ce qui s'affiche ; des espions relèvent _bloc (titre, texte, état),
     _mini_metrics, build_figure (toutes ses données : abscisses, courbes, repères), le tableau « Données météo »,
     t_real_for_target, classe_min_pour_delai, classe_estimee_depuis_mesure, age_equiv_from_series, et le dict
     « ab_rapport » ; le mode Météo est rejoué avec la météo déposée dans session_state, comme après « Récupérer ».
     Les nombres sont comparés AU BIT PRÈS ; les textes de l'écran sont réécrits depuis le moteur et comparés.
  4. Couverture : les cas atteignent chaque état, chaque branche (seuil atteint / non, classe actuelle / supérieure /
     aucune, essai cohérent / dessus / sous, t_e ≥ 28, hors domaine, relevé mesuré / prévu / prolongé / sans donnée…).
  5. Le relevé saisi (le site remplace Open-Meteo) : formats, refus, aller-retour.
  6. Format : JSON strict, normalisation (défauts, ValueError hors du domaine de l'écran), formules recalculables.
  7. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, matplotlib,
     pandas, requests, scipy… et n'y charge que la bibliothèque standard et numpy, sous 3.12 et 3.13.
  8. Mutations : onze défauts plausibles glissés dans le moteur font tous rougir la parité.
  9. De bout en bout : ce que l'écran Streamlit AFFICHE (blocs, légendes, mesures, tableau, figure) est lu, au même
     arrondi, dans ce que le SITE affiche (api_web/routeur.traiter → fragments HTML et SVG), sur plus de 30 coulages.

    /usr/bin/python3.12 tests/test_maturite_moteur.py       (streamlit, matplotlib et pandas requis pour la parité)
"""
import arrondi_affichage as AA  # noqa: E402  (tests/, le même dossier)
import ast
import contextlib
import copy
import datetime as _dt
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from beton.maturite import ec2 as EC  # noqa: E402
from beton.maturite import entree as EN  # noqa: E402
from beton.maturite import moteur as M  # noqa: E402
from beton.maturite import releve as RL  # noqa: E402

OK, KO = [], []
ECART = {"n": 0, "bit": 0}


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as f:
        return f.read()


def egal(a, b):
    """Égalité AU BIT PRÈS pour les nombres (NaN = NaN), stricte pour le reste."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        ECART["n"] += 1
        same = (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        ECART["bit"] += same
        return same
    return a == b


def ecarts(a, b, chemin="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append(f"{chemin} : clés {sorted(set(a) ^ set(b))}")
        for k in sorted(set(a) & set(b)):
            ecarts(a[k], b[k], f"{chemin}.{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{chemin} : {len(a)} ≠ {len(b)} éléments")
        for i, (x, y) in enumerate(zip(a, b)):
            ecarts(x, y, f"{chemin}[{i}]", out)
    elif not egal(a, b):
        out.append(f"{chemin} : {a!r} ≠ {b!r}")
    return out


# ------------------------------------------------------------------ le module Streamlit et son faux « st »
try:
    logging.disable(logging.WARNING)
    import numpy as np  # noqa: E402
    import streamlit  # noqa: E402,F401
    import modules.age_beton as AB  # noqa: E402
    import modules.navigation as NAV  # noqa: E402
except ImportError as e:  # pragma: no cover — la parité EXIGE le module
    AB = None
    chk("streamlit, matplotlib, pandas et modules/age_beton.py se chargent (indispensables à la parité)", False, e)


class FauxSt:
    """Un « st » qui rend les SAISIES d'un scénario (par clé de widget, sinon par libellé) et ENREGISTRE ce que l'écran
    afficherait ; sans saisie, chaque widget rend sa valeur par défaut — l'écran au premier affichage."""

    def __init__(self, saisies, session=None):
        self.saisies = dict(saisies)
        self.session_state = dict(session or {})
        self.ev = []
        self.widgets = {}

    def _val(self, key, label, defaut, **bornes):
        cle = key or label
        self.widgets[cle] = dict(defaut=defaut, **bornes)
        v = self.saisies.get(cle, defaut)
        if key:
            self.session_state[key] = v
        return v

    def markdown(self, body="", *a, **k):
        self.ev.append(("md", str(body)))

    def caption(self, body="", *a, **k):
        self.ev.append(("caption", str(body)))

    def warning(self, body="", *a, **k):
        self.ev.append(("warning", str(body)))

    def error(self, body="", *a, **k):
        self.ev.append(("error", str(body)))

    def latex(self, body="", *a, **k):
        self.ev.append(("latex", str(body)))

    def expander(self, label, *a, **k):
        self.ev.append(("exp", label))
        return contextlib.nullcontext()

    def container(self, *a, **k):
        return contextlib.nullcontext()

    def columns(self, spec, *a, **k):
        return [contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def divider(self, *a, **k):
        pass

    def pyplot(self, *a, **k):
        pass

    def dataframe(self, df, *a, **k):
        self.ev.append(("df", df.to_dict("records")))

    def button(self, *a, **k):
        return False

    def toggle(self, label, value=False, key=None, **k):
        return self._val(key, label, value)

    def selectbox(self, label, options, index=0, key=None, **k):
        return self._val(key, label, list(options)[index], options=list(options))

    def radio(self, label, options, index=0, key=None, **k):
        return self._val(key, label, list(options)[index], options=list(options))

    def multiselect(self, label, options, default=None, key=None, **k):
        return list(self._val(key, label, list(default or []), options=list(options)))

    def text_input(self, label, value="", key=None, **k):
        return self._val(key, label, value)

    def date_input(self, label, value=None, key=None, **k):
        if key and key in self.session_state and key not in self.saisies:
            return self.session_state[key]
        return self._val(key, label, value)

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None, **k):
        return self._val(key, label, value, mini=min_value, maxi=max_value)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        return self._val(key, label, value, mini=min_value, maxi=max_value, pas=step)

    def checkbox(self, label, value=False, key=None, on_change=None, **k):
        v = self._val(key, label, value)
        if on_change is not None and key in self.saisies and v != value:
            on_change()                                  # le clic : st.session_state[key] est déjà posé
        return v


CIMENT_LIB = EN.CIMENT_LIBELLES
COUL_RELEVE = _dt.date(2026, 10, 1)                 # coulage des relevés sans date (le module date chaque jour)


def saisies(e):
    """Le coulage (format du moteur) traduit en saisies de l'écran Streamlit, et la session qui va avec (la météo
    « récupérée » en mode Météo, les cases déjà cochées)."""
    t = e["temperature"]
    s = {"ab_adv": e["avance"], "Classe béton": e["classe"], "Classe ciment": CIMENT_LIB[e["ciment"]],
         "Date de coulage": _dt.date.fromisoformat(e["date_coulage"]) if e["date_coulage"] else None,
         "Température": "Manuel" if t["mode"] == "constante" else "Météo", "T moyenne (°C)": t["T"],
         "Ea [kJ/mol]": e["Ea_kJ"], "ab_age": e["age"],
         "ab_s_mode": "Pourcentage de fck(28)" if e["seuil"]["mode"] == "pct" else "Résistance (MPa)",
         "ab_s_pct": e["seuil"]["pct"], "ab_s_mpa": e["seuil"]["mpa"], "ab_chk_seuil": e["seuil"]["afficher"],
         "ab_o_mpa": e["optim"]["mpa"], "ab_o_del": e["optim"]["delai"], "ab_chk_opt": e["optim"]["afficher"],
         "ab_cmp_cls": list(e["comparateur"]["classes"]), "ab_cmp_cim": CIMENT_LIB[e["comparateur"]["ciment"]],
         "ab_chk_cmp": e["comparateur"]["afficher"],
         "ab_e_mpa": e["essai"]["mpa"], "ab_e_age": e["essai"]["age"], "ab_e_T": e["essai"]["T"], "ab_chk_lab": e["essai"]["afficher"]}
    session = {}
    if t["mode"] == "releve":
        lieu = t["lieu"] or "Bruxelles, Belgique"
        s["Lieu du chantier"] = lieu
        rel = RL.lire(t["releve"])
        if rel["temps"]:
            dates = rel["dates"] or [(COUL_RELEVE + _dt.timedelta(days=i)).isoformat() for i in range(len(rel["temps"]))]
            d0, d1 = _dt.date.fromisoformat(dates[0]), _dt.date.fromisoformat(dates[-1])
            s["ab_m_coul"] = d0
            session["ab_m_ctrl"] = d1
            session["ab_meteo"] = {"d_dates": dates, "d_temps": list(rel["temps"]), "d_prev": rel["prev"], "d_extra": rel["extra"],
                                   "n_extra": sum(rel["extra"]), "label": lieu, "lat": 50.85, "lon": 4.35,
                                   "req": [lieu, d0.isoformat(), d1.isoformat()]}
        else:
            s["ab_m_coul"] = COUL_RELEVE
            session["ab_m_ctrl"] = COUL_RELEVE + _dt.timedelta(days=5)
    return s, session


ESPIONS = ("_bloc", "_mini_metrics", "build_figure", "t_real_for_target", "classe_min_pour_delai",
           "classe_estimee_depuis_mesure", "age_equiv_from_series")


def ecran(sais, session=None):
    """show() exécuté avec le faux « st » : ce que l'écran affiche, et ce que ses fonctions ont rendu."""
    faux = FauxSt(sais, session)
    traces = {k: [] for k in ESPIONS}
    vrais = {k: getattr(AB, k) for k in ESPIONS}

    def espion(nom):
        def f(*a, **k):
            if nom == "build_figure":
                traces[nom].append(dict(k, t_real=a[0], ref=a[1]))
                return None
            r = vrais[nom](*a, **k)
            traces[nom].append((a, k, r))
            return r
        return f

    vrai_st, vrai_bouton = AB.st, NAV.bouton_accueil
    AB.st = faux
    NAV.bouton_accueil = lambda *a, **k: None
    for k in ESPIONS:
        setattr(AB, k, espion(k))
    try:
        AB.show()
    finally:
        AB.st, NAV.bouton_accueil = vrai_st, vrai_bouton
        for k, f in vrais.items():
            setattr(AB, k, f)
    return faux, traces


def textes(evs, sorte):
    return [x[1] for x in evs if x[0] == sorte]


# ------------------------------------------------------------------ les cas, au format du moteur
def coulage(nom="Cas", **k):
    e = EN.nouvel_element(nom)
    for cle, v in k.items():
        if "__" in cle:
            a, b = cle.split("__")
            e[a][b] = v
        else:
            e[cle] = v
    return EN.normaliser(e)


def releve(temps, sources=None, dates=False, debut=_dt.date(2026, 10, 12)):
    """Un relevé collé : une ligne par jour, « — » pour un jour sans donnée."""
    lignes = []
    for i, t in enumerate(temps):
        x = "—" if t is None else repr(float(t)).replace(".", ",")
        src = (sources or "")[i:i + 1] if sources else ""
        mot = {"m": "mesuré", "p": "prévision", "x": "prolongé"}.get(src, "")
        lignes.append(";".join(v for v in ((debut + _dt.timedelta(days=i)).isoformat() if dates else "", x, mot) if v))
    return "\n".join(lignes)


CAS = [
    ("défauts de l'écran (nouvel_element)", EN.nouvel_element()),
    ("C30/37 N, 7 j à 20 °C", coulage(age=7)),
    ("C25/30 R, 3 j à 20 °C : borne du domaine (t_e = 3 j pile, ≤ : hors domaine)", coulage(classe="C25/30", ciment="R", age=3)),
    ("C20/25 S, 2 j à 5 °C : hors domaine", coulage(classe="C20/25", ciment="S", age=2, temperature__T=5.0)),
    ("C50/60 R, 1 j à 60 °C (borne haute de T)", coulage(classe="C50/60", ciment="R", age=1, temperature__T=60.0)),
    ("C40/50 S, 28 j à −5 °C (borne basse de T) : t_e < 28 j, f_ck(t) < f_ck", coulage(classe="C40/50", ciment="S", temperature__T=-5.0)),
    ("C35/45 N, 14 j à 12,5 °C, seuil 50 %", coulage(classe="C35/45", age=14, temperature__T=12.5, seuil__pct=50)),
    ("seuil 100 % à 20 °C (cible = f_ck)", coulage(seuil__pct=100, seuil__afficher=True)),
    ("seuil 100 % à −5 °C : non atteignable en 90 j", coulage(temperature__T=-5.0, seuil__pct=100, seuil__afficher=True)),
    ("seuil 30 % à 20 °C : atteint avec t_e ≤ 3 j", coulage(seuil__pct=30, seuil__afficher=True)),
    ("seuil en MPa (12,5 MPa), date de coulage : le seuil est daté",
     coulage(seuil__mode="mpa", seuil__mpa=12.5, date_coulage="2026-11-02", seuil__afficher=True)),
    ("seuil en MPa au-dessus de f_ck (valeur gardée après un changement de classe) : non atteignable",
     coulage(classe="C20/25", seuil__mode="mpa", seuil__mpa=25.0, seuil__afficher=True)),
    ("seuil en MPa = f_ck pile, date de coulage", coulage(seuil__mode="mpa", seuil__mpa=30.0, date_coulage="2026-02-27")),
    ("optimisation : la classe actuelle suffit (cible 20 MPa en 7 j)", coulage(optim__mpa=20.0, optim__delai=7.0, optim__afficher=True)),
    ("optimisation : une classe inférieure suffit (« actuelle » affiché)", coulage(classe="C45/55", optim__mpa=15.0, optim__delai=5.0, optim__afficher=True)),
    ("optimisation : aucune classe (60 MPa en 2 j)", coulage(optim__mpa=60.0, optim__delai=2.0, optim__afficher=True)),
    ("optimisation : délai 0,5 j à 35 °C", coulage(optim__mpa=8.0, optim__delai=0.5, temperature__T=35.0, optim__afficher=True)),
    ("optimisation : délai long (45 j) — courbe et repère au-delà de l'axe", coulage(optim__mpa=40.0, optim__delai=45.0, optim__afficher=True)),
    ("comparateur affiché, sept classes, même ciment", coulage(comparateur__afficher=True)),
    ("comparateur : trois classes dans un ordre libre, ciment S", coulage(comparateur__afficher=True, comparateur__classes=["C50/60", "C20/25", "C30/37"],
                                                                         comparateur__ciment="S")),
    ("comparateur coché sans classe : rien n'est tracé", coulage(comparateur__afficher=True, comparateur__classes=[])),
    ("essai cohérent (21,6 MPa à 7 j, C30/37)", coulage(essai__mpa=21.6, essai__age=7.0, essai__afficher=True)),
    ("essai au-dessus (35 MPa à 7 j)", coulage(essai__mpa=35.0, essai__age=7.0, essai__afficher=True)),
    ("essai sous la classe (12 MPa à 7 j)", coulage(essai__mpa=12.0, essai__age=7.0, essai__afficher=True)),
    ("essai à 28 j et plus (t_e ≥ 28 : f_ck,28 = mesure)", coulage(essai__mpa=33.0, essai__age=35.0, essai__afficher=True)),
    ("essai à 1 j, T de cure 5 °C : hors domaine", coulage(essai__mpa=4.0, essai__age=1.0, essai__T=5.0, essai__afficher=True)),
    ("essai à la limite des classes (32,5 MPa à 28 j : la plus proche, à égalité la première)", coulage(essai__mpa=32.5, essai__age=28.0)),
    ("essai non coché : pas sur la figure", coulage(essai__mpa=25.0, essai__age=10.0)),
    ("énergie d'activation ajustée à 45 kJ/mol, 8 °C", coulage(avance=True, Ea_kJ=45.0, temperature__T=8.0, age=10)),
    ("énergie d'activation 20 kJ/mol (borne basse), 30 °C", coulage(avance=True, Ea_kJ=20.0, temperature__T=30.0, age=5)),
    ("énergie d'activation 60 kJ/mol (borne haute), 2 °C", coulage(avance=True, Ea_kJ=60.0, temperature__T=2.0, age=20)),
    ("⚙️ ouvert sans changer E_a (33,256 kJ/mol)", coulage(avance=True, age=9)),
    ("valeur saisie non entière : T = 17,3 °C, E_a 38,7", coulage(temperature__T=17.3, avance=True, Ea_kJ=38.7, age=11)),
    ("délai 0,5 j à −5 °C : f_ck(délai) ≤ 0 pour toutes les classes (taux sans objet, JSON strict)",
     coulage(temperature__T=-5.0, optim__mpa=5.0, optim__delai=0.5, optim__afficher=True, age=1)),
    ("tout affiché : seuil, optimisation, comparateur, essai", coulage(classe="C25/30", ciment="R", age=6, temperature__T=11.0, seuil__afficher=True,
                                                                     optim__afficher=True, comparateur__afficher=True, essai__mpa=18.0,
                                                                     essai__age=6.0, essai__T=11.0, essai__afficher=True, date_coulage="2026-12-30")),
    ("relevé mesuré seul (14 j, sans date)", coulage(temperature__mode="releve", temperature__releve=releve([12, 11, 9.5, 10, 13, 14, 12, 8, 7, 9, 10, 11, 12, 13]), age=10)),
    ("relevé daté avec prévisions (défavorable)", coulage(temperature__mode="releve", temperature__releve=releve(
        [6.5, 7, 8, 5.5, 4, 3.5, 6, 8, 9, 10], "mmmmmmpppp", dates=True), age=8, seuil__afficher=True, optim__afficher=True)),
    ("relevé avec jours prolongés et un jour sans donnée", coulage(temperature__mode="releve", temperature__releve=releve(
        [10, None, 12, 11, 9, 9, 9, 9], "mmmppxxx", dates=True), age=8, comparateur__afficher=True, essai__mpa=10.0, essai__afficher=True)),
    ("relevé froid : t_e ≤ 3 j à l'âge choisi (hors domaine) et défavorable", coulage(temperature__mode="releve", temperature__releve=releve(
        [1, 0.5, -1, 2, 3], "mmppp"), age=5)),
    ("relevé d'un seul jour : curseur borné à 2 j, le 2e jour n'existe pas", coulage(temperature__mode="releve", temperature__releve=releve([15]), age=2)),
    ("relevé long (45 j) : âge au-delà de l'axe de la figure", coulage(temperature__mode="releve", temperature__releve=releve(
        [8 + (i % 9) for i in range(45)], dates=True), age=45, temperature__lieu="Namur", essai__mpa=30.0, essai__age=42.0, essai__afficher=True)),
    ("relevé sans aucune donnée exploitable : T = 20 °C (le module)", coulage(temperature__mode="releve", temperature__releve=releve([None, None, None]))),
    ("relevé vide : T = 20 °C par défaut", coulage(temperature__mode="releve", temperature__releve="")),
    ("relevé collé depuis le tableau de l'écran (en-tête, T min h / T max h)", coulage(temperature__mode="releve", temperature__releve=(
        "Date\tT moy (°C)\tSource\tT min h\tT max h\n2026-11-02\t9.4\tmesuré\t4.1\t13.0\n2026-11-03\t—\tmesuré\t\t\n"
        "2026-11-04\t11.2\tprévision\t7.0\t15.5\n2026-11-05\t11.2\tprolongé\t\t"), age=4)),
    ("relevé, lieu saisi, E_a ajusté (le t_e du tableau garde E_a de l'EC2)", coulage(temperature__mode="releve", temperature__lieu="Liège",
     temperature__releve=releve([14, 15, 16, 12, 11, 10], "mmmmpp", dates=True), avance=True, Ea_kJ=40.0, age=6)),
    ("relevé froid puis chaud (7 j à 2 °C, 7 j à 20 °C), seuil 50 % daté : ② à la moyenne l'annonce trop tôt (défaut non conservatif)",
     coulage(temperature__mode="releve", temperature__releve=releve([2] * 7 + [20] * 7, dates=True), seuil__pct=50, seuil__afficher=True, age=5,
             date_coulage="2026-10-12")),
]
for i, (cl, ci) in enumerate([(c, k) for c in EC.CLASSES for k in "RNS"]):
    CAS.append((f"{cl} ciment {ci}, {3 + i % 25} j à {(-5 + 3 * i) % 41} °C", coulage(classe=cl, ciment=ci, age=3 + i % 25, temperature__T=float((-5 + 3 * i) % 41))))


def aleatoire(rng, k):
    """Un coulage tiré au sort (graine fixe) : béton, cure, âge, et chaque analyse."""
    e = EN.nouvel_element(f"Aléa {k}")
    e["classe"], e["ciment"] = rng.choice(EC.CLASSES), rng.choice("RNS")
    e["avance"] = rng.random() < 0.3
    e["Ea_kJ"] = round(rng.uniform(20, 60), 3)
    if rng.random() < 0.35:
        n = rng.randint(1, 40)
        temps = [None if rng.random() < 0.08 else round(rng.uniform(-5, 30), 1) for _ in range(n)]
        src = "".join(rng.choice("mmmpx") for _ in range(n))
        e["temperature"].update(mode="releve", releve=releve(temps, src, dates=rng.random() < 0.5))
        e["age"] = EN.age_defaut(EN.age_max(e))
        if rng.random() < 0.7:
            e["age"] = rng.randint(1, EN.age_max(e))
    else:
        e["temperature"]["T"] = round(rng.uniform(-5, 60), rng.choice((0, 1, 2)))
        e["age"] = rng.randint(1, 28)
    e["date_coulage"] = (_dt.date(2026, 1, 1) + _dt.timedelta(days=rng.randint(0, 600))).isoformat() if rng.random() < 0.4 else None
    fck = EC.fck_classe(e["classe"])
    e["seuil"].update(mode=rng.choice(("pct", "mpa")), pct=rng.choice(EN.POURCENTAGES), mpa=round(rng.uniform(1, fck), 1),
                      afficher=rng.random() < 0.5)
    e["optim"].update(mpa=round(rng.uniform(1, 70), 1), delai=round(rng.uniform(0.5, 50), 1), afficher=rng.random() < 0.5)
    e["comparateur"].update(classes=rng.sample(EC.CLASSES, rng.randint(0, 7)), ciment=rng.choice("RNS"), afficher=rng.random() < 0.5)
    e["essai"].update(mpa=0.0 if rng.random() < 0.3 else round(rng.uniform(1, 70), 1), age=round(rng.uniform(0.5, 60), 1),
                      T=round(rng.uniform(-5, 60), 1), afficher=rng.random() < 0.5)
    return EN.normaliser(e)


RNG = random.Random(20261107)
CAS += [(f"tirage {k:02d} (graine 20261107)", aleatoire(RNG, k)) for k in range(1, 41)]


# ------------------------------------------------------------------ 1. données
print("=== 1. Données : listes, constantes, valeurs par défaut de l'écran ===")
if AB is not None:
    chk("CLASSES, CIMENTS, CIMENT_DETAIL sont ceux du module", EC.CLASSES == AB.CLASSES and EC.CIMENTS == AB.CIMENTS
        and EC.CIMENT_DETAIL == AB.CIMENT_DETAIL and list(EC.CIMENTS) == list(AB.CIMENTS))
    chk("R, E_a de l'EC2 (4000·R), T_ref, marge défavorable (3 °C), horizon (16 j) sont ceux du module",
        (EC.R_GAZ, EC.EA_EC2, EC.T_REF_K, EC.MARGE_PREVISION_C, EC.HORIZON_PREVISION_J)
        == (AB.R_GAZ, AB.EA_EC2, AB.T_REF_K, AB.MARGE_PREVISION_C, AB.HORIZON_PREVISION_J))
    chk("la référence de parité est l'en-tête du module (VERSION 3.2)", "VERSION 3.2" in lire("modules", "age_beton.py")[:300]
        and M.REFERENCE == "modules/age_beton.py v3.2")
    chk("les couleurs de la figure sont celles du module", M.COULEURS == {"ref": AB.COL_REF, "defav": AB.COL_DEFAV, "seuil": AB.COL_SEUIL,
                                                                          "opt": AB.COL_OPT, "mes": AB.COL_MES} and M.COULEURS_CMP == AB.COL_CMP)
    faux, tr = ecran({})
    faux_mpa, _ = ecran({"ab_s_mode": "Résistance (MPa)"})        # le champ en MPa n'existe qu'une fois ce mode choisi
    w, ss = dict(faux.widgets, ab_s_mpa=faux_mpa.widgets["ab_s_mpa"]), faux.session_state
    lu = {"classe": w["Classe béton"]["defaut"], "ciment": next(k for k, v in CIMENT_LIB.items() if v == w["Classe ciment"]["defaut"]),
          "date_coulage": w["Date de coulage"]["defaut"], "mode": {"Manuel": "constante"}[w["Température"]["defaut"]],
          "T": w["T moyenne (°C)"]["defaut"], "avance": w["ab_adv"]["defaut"], "age": w["ab_age"]["defaut"],
          "seuil": {"mode": {"Pourcentage de fck(28)": "pct"}[w["ab_s_mode"]["defaut"]], "pct": w["ab_s_pct"]["defaut"],
                    "mpa": w["ab_s_mpa"]["defaut"], "afficher": w["ab_chk_seuil"]["defaut"]},
          "optim": {"mpa": w["ab_o_mpa"]["defaut"], "delai": w["ab_o_del"]["defaut"], "afficher": w["ab_chk_opt"]["defaut"]},
          "comparateur": {"classes": w["ab_cmp_cls"]["defaut"], "ciment": next(k for k, v in CIMENT_LIB.items() if v == w["ab_cmp_cim"]["defaut"]),
                          "afficher": w["ab_chk_cmp"]["defaut"]},
          "essai": {"mpa": w["ab_e_mpa"]["defaut"], "age": w["ab_e_age"]["defaut"], "T": w["ab_e_T"]["defaut"], "afficher": w["ab_chk_lab"]["defaut"]}}
    d = EN.nouvel_element()
    moteur = {"classe": d["classe"], "ciment": d["ciment"], "date_coulage": d["date_coulage"], "mode": d["temperature"]["mode"],
              "T": d["temperature"]["T"], "avance": d["avance"], "age": d["age"], "seuil": d["seuil"], "optim": d["optim"],
              "comparateur": d["comparateur"], "essai": d["essai"]}
    chk("nouvel_element() = l'écran au premier affichage (C30/37, N, 20 °C, 28 j, 75 %, cible 30,0 MPa en 14,0 j, 7 classes, essai 0)",
        not ecarts(moteur, lu), ecarts(moteur, lu))
    faux2, _ = ecran({"ab_adv": True})
    chk("E_a par défaut (⚙️ ouvert) = EA_EC2/1000 = 33,256 kJ/mol, bornes 20 à 60 ; et ×1000 rend EA_EC2 au bit près",
        egal(faux2.widgets["Ea [kJ/mol]"]["defaut"], d["Ea_kJ"]) and (faux2.widgets["Ea [kJ/mol]"]["mini"], faux2.widgets["Ea [kJ/mol]"]["maxi"])
        == (EN.EA_MIN, EN.EA_MAX) and d["Ea_kJ"] * 1000.0 == EC.EA_EC2)
    chk("bornes des champs = celles de l'écran (T −5 à 60 ; âge 1 à 28 ; % 30 à 100 pas 5 ; seuil ≥ 1 ; délai ≥ 0,5 ; essai ≥ 0, ≥ 0,5)",
        (w["T moyenne (°C)"]["mini"], w["T moyenne (°C)"]["maxi"], w["ab_age"]["mini"], w["ab_age"]["maxi"], w["ab_s_pct"]["mini"],
         w["ab_s_pct"]["maxi"], w["ab_s_pct"]["pas"], w["ab_s_mpa"]["mini"], w["ab_o_mpa"]["mini"], w["ab_o_del"]["mini"],
         w["ab_e_mpa"]["mini"], w["ab_e_age"]["mini"], w["ab_e_T"]["mini"], w["ab_e_T"]["maxi"])
        == (EN.T_MIN, EN.T_MAX, 1, EN.AGE_DEFAUT_MAX, 30, 100, 5, 1.0, 1.0, 0.5, 0.0, 0.5, EN.T_MIN, EN.T_MAX)
        and list(EN.POURCENTAGES) == list(range(w["ab_s_pct"]["mini"], w["ab_s_pct"]["maxi"] + 1, w["ab_s_pct"]["pas"])))

# ------------------------------------------------------------------ 2. à la main
print("\n=== 2. Contrôles à la main (math seul, formules de l'EC2 écrites ici) ===")


def proche(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


r = M.calculer(coulage(age=7))
chk("C30/37 N, 7 j à 20 °C : t_e = 7 j, β = exp(0,25·(1 − √4)) = 0,7788, f_ck(t) = 0,7788 × 38 − 8 = 21,59 MPa (72 %)",
    proche(r["age"]["t_e"], 7.0) and proche(r["age"]["fck"], math.exp(-0.25) * 38 - 8) and proche(r["age"]["pct"], (math.exp(-0.25) * 38 - 8) / 30 * 100),
    r["age"])
fac10 = math.exp(-4000.0 * (1 / 283.15 - 1 / 293.15))
r = M.calculer(coulage(age=10, temperature__T=10.0, classe="C25/30", ciment="R"))
b = math.exp(0.20 * (1 - math.sqrt(28 / (10 * fac10))))
chk(f"C25/30 R, 10 j à 10 °C : facteur = exp(−4000·(1/283,15 − 1/293,15)) = {fac10:.4f}, t_e = {10 * fac10:.3f} j, "
    f"f_ck(t) = {b * 33 - 8:.3f} MPa", proche(r["temperature"]["facteur"], fac10) and proche(r["age"]["t_e"], 10 * fac10)
    and proche(r["age"]["fck"], b * 33 - 8), (r["temperature"]["facteur"], r["age"]["fck"]))
r = M.calculer(coulage(temperature__mode="releve", temperature__releve="5\n10\n15", age=3))
te = sum(math.exp(-4000.0 * (1 / (t + 273.15) - 1 / 293.15)) for t in (5, 10, 15))
chk(f"relevé 5 / 10 / 15 °C : t_e = Σ facteurs = {te:.4f} j, T de calcul = moyenne = 10 °C",
    proche(r["age"]["t_e"], te) and r["temperature"]["T"] == 10.0, r["age"]["t_e"])
r = M.calculer(coulage(seuil__pct=75))
te_c = 28 / (1 - math.log(30.5 / 38) / 0.25) ** 2
chk(f"seuil 75 % (22,5 MPa) à 20 °C : t_e = 28/(1 − ln(30,5/38)/0,25)² = {te_c:.4f} j ; la dichotomie du module tombe à 0,001 MPa près",
    proche(r["seuil"]["controle"]["t_e"], te_c) and abs(r["seuil"]["fck_t"] - 22.5) < 1e-3 and abs(r["seuil"]["t"] - te_c) < 1e-3, r["seuil"]["t"])
r = M.calculer(coulage(essai__mpa=21.6, essai__age=7.0))
est = (21.6 + 8) / math.exp(-0.25) - 8
chk(f"essai 21,6 MPa à 7 j : f_ck,28 estimé = (21,6 + 8)/e^(−0,25) − 8 = {est:.3f} MPa → C30/37, écart {(est - 30) / 30 * 100:+.1f} % : cohérent",
    proche(r["essai"]["fck28_est"], est) and r["essai"]["classe"] == "C30/37" and r["essai"]["verdict"] == "coherent", r["essai"])
chk(f"essai 21,6 MPa à 7 j lu comme une MOYENNE (le site, pour information) : 21,6/e^(−0,25) − 8 = {21.6 / math.exp(-0.25) - 8:.2f} MPa, "
    "soit 8/β_cc de moins que le module", proche(r["essai"]["fck28_est_moyenne"], 21.6 / math.exp(-0.25) - 8)
    and proche(r["essai"]["fck28_est"] - r["essai"]["fck28_est_moyenne"], 8 / math.exp(-0.25), 1e-9))
r = M.calculer(coulage(optim__mpa=30.0, optim__delai=14.0))
f14 = {c: math.exp(0.25 * (1 - math.sqrt(2))) * (EC.fck_classe(c) + 8) - 8 for c in EC.CLASSES}
chk(f"classe minimale pour 30 MPa en 14 j : C30/37 n'a que {f14['C30/37']:.2f} MPa, C35/45 {f14['C35/45']:.2f} MPa → C35/45",
    r["optim"]["classe"] == "C35/45" and not r["optim"]["ok_retenue"] and proche(r["optim"]["fck"], f14["C35/45"]), r["optim"]["classe"])

# ② à la température MOYENNE du relevé (le module) face au relevé jour par jour (comme ①) : jours froids d'abord, le
# seuil est annoncé trop tôt — défaut NON CONSERVATIF du module, que le moteur nomme (seuil.releve) et que le site signale
froid_chaud = coulage(temperature__mode="releve", temperature__releve="\n".join(["2"] * 7 + ["20"] * 7), seuil__pct=50, age=5)
r = M.calculer(froid_chaud)
fac = lambda T: math.exp(-4000.0 * (1 / (T + 273.15) - 1 / 293.15))  # noqa: E731
te5 = 5 * fac(2.0)
f5 = math.exp(0.25 * (1 - math.sqrt(28 / te5))) * 38 - 8
jours = [d for d in range(1, 15) if math.exp(0.25 * (1 - math.sqrt(28 / (min(d, 7) * fac(2.0) + max(0, d - 7) * fac(20.0))))) * 38 - 8 >= 15.0]
chk(f"② NON CONSERVATIF, 7 j à 2 °C puis 7 j à 20 °C, seuil 15 MPa : le module (à 11 °C) l'annonce après {r['seuil']['t']:.2f} j ; jour par jour, "
    f"f_ck(5 j) = {f5:.2f} MPa < 15 MPa, atteint au jour {jours[0]} — le moteur le nomme (seuil.releve.optimiste)",
    r["seuil"]["t"] < 5 and r["seuil"]["releve"]["optimiste"] and r["seuil"]["releve"]["jour"] == jours[0] == 8
    and r["seuil"]["releve"]["jour_annonce"] == 5 and proche(r["seuil"]["releve"]["fck_jour_annonce"], f5, 1e-9)
    and any(a_["code"] == "seuil_optimiste" for a_ in r["avertissements"]), r["seuil"]["releve"])
r = M.calculer(coulage(temperature__mode="releve", temperature__releve="\n".join(["20"] * 7 + ["2"] * 7), seuil__pct=50, age=5))
chk("② avec les jours chauds d'abord : le relevé atteint le seuil au plus tard au jour annoncé — pas d'avertissement",
    not r["seuil"]["releve"]["optimiste"] and not any(a_["code"] == "seuil_optimiste" for a_ in r["avertissements"]), r["seuil"]["releve"])
incoherents = []
for desc, e in [x for x in CAS if x[1]["temperature"]["mode"] == "releve"]:
    r = M.calculer(e)
    sv, sr_ = r["seuil"]["releve"], r["seuil"]
    if r["temperature"]["source"] != "releve" or sr_["t"] is None:
        if sv is not None:
            incoherents.append((desc, "releve sans relevé ni seuil"))
        continue
    lignes = r["tableau"]["lignes"]
    j = math.ceil(sr_["t"])
    premier = next((li["age"] for li in lignes if li["fck"] >= sr_["cible"]), None)
    attendu = (j <= len(lignes)) and lignes[j - 1]["fck"] < sr_["cible"]
    if sv["jour"] != premier or sv["optimiste"] != attendu:
        incoherents.append((desc, sv, premier, attendu))
chk("seuil.releve : sur chaque relevé des cas, le premier jour atteint et le drapeau « optimiste » se relisent dans le tableau jour par jour",
    not incoherents, incoherents[:2])

# ------------------------------------------------------------------ 3. parité
FMT = {"ok": "ok", "warn": "warn", "nok": "nok", "info": "info"}


def attendus_blocs(r):
    """Les blocs que l'écran doit afficher, ÉCRITS avec les nombres du moteur et les f-strings du module."""
    a, T = r["age"], r["temperature"]["T"]
    out = [(f"fck({a['t']} j à {T:.0f} °C)", f"{a['fck']:.2f} MPa · {a['pct']:.0f} % de fck(28)", a["etat"])]
    s = r["seuil"]
    cible_txt = (f"{s['pct']} % = {s['cible']:.1f} MPa" if s["mode"] == "pct"
                 else f"{s['cible']:.1f} MPa ({s['cible'] / r['materiau']['fck28'] * 100:.0f} %)")
    if s["t"] is None:
        out.append((f"Seuil {cible_txt}", "non atteignable", "nok"))
    else:
        dt = f" — le {_dt.date.fromisoformat(s['date']).strftime('%d/%m/%Y')}" if s["date"] else ""
        out.append((f"Seuil {cible_txt}", f"après {s['t']:.1f} j{dt}", "ok"))
    o = r["optim"]
    if o["classe"] is None:
        out.append((f"Aucune classe pour {o['mpa']:.1f} MPa en {o['delai']:.1f} j", "", "nok"))
    else:
        out.append((f"Classe minimale : {o['classe']}{' (actuelle)' if o['actuelle'] else ''}", f"{o['fck']:.1f} MPa à {o['delai']:.1f} j",
                    "info" if o["actuelle"] else "ok"))
    e = r["essai"]
    if e["actif"]:
        out.append(("Classe estimée", f"≈ {e['classe']} (fck28 ≈ {e['fck28_est']:.1f} MPa)", "info"))
        lbl = r["materiau"]["classe"]
        out.append(({"coherent": f"Cohérent avec {lbl}", "dessus": f"Au-dessus de {lbl}", "sous": f"Sous {lbl}"}[e["verdict"]],
                    f"écart {e['ecart']:+.0f} %", e["etat"]))
    return out


def comparer(desc, e):
    """Le moteur contre l'écran sur un coulage ; rend les écarts."""
    r = M.calculer(e)
    sais, session = saisies(e)
    faux, tr = ecran(sais, session)
    ev = faux.ev
    out = []
    rap = faux.session_state["ab_rapport"]
    a, T = r["age"], r["temperature"]
    ec = lambda nom, x, y: out.extend(ecarts(x, y, nom))  # noqa: E731
    # le rapport (les nombres exacts de ① et la température)
    ec("rapport", {"beton": r["materiau"]["classe"], "fck28": r["materiau"]["fck28"], "ciment": r["materiau"]["ciment_libelle"],
                   "s": r["materiau"]["s"], "mode_temp": sais["Température"], "T_moyenne": T["T"],
                   "source_temp": "manuelle" if T["source"] == "manuelle" else f"météo ({sais['Lieu du chantier']})",
                   "profil_temps": T["profil"], "Ea_kJ_mol": T["Ea_kJ"], "age_sel_j": a["t"], "t_e_sel_j": a["t_e"], "fck_sel_MPa": a["fck"],
                   "pct_fck28": a["pct"], "date_coulage": e["date_coulage"],
                   "defav": None if a["defav"] is None else {"T_moy": a["defav"]["T_moy"], "n_prev": a["defav"]["n_prev"],
                                                              "n_extra": a["defav"]["n_extra"], "marge": a["defav"]["marge"],
                                                              "t_e_sel_j": a["defav"]["t_e"], "fck_sel_MPa": a["defav"]["fck"]}},
       {k: v for k, v in rap.items() if k not in ("module", "version")})
    # les blocs (titre, texte, état) et leur ordre
    ec("blocs", attendus_blocs(r), [(x[0][0], x[0][1] if len(x[0]) > 1 else x[1].get("right", ""), x[0][2] if len(x[0]) > 2 else x[1].get("etat", "ok"))
                                    for x in tr["_bloc"]])
    # les légendes de l'écran : t_e de ①, défavorable, détail du ciment, hors domaine de l'essai
    cap = textes(ev, "caption")
    src = f"  ({rap['source_temp']})" if rap["source_temp"] != "manuelle" else ""
    att = f"t_e = {a['t_e']:.1f} j" + src + ("  — ⚠️ ≤ 3 j : hors domaine EC2" if a["hors_domaine"] else "")
    if att not in cap:
        out.append(f"légende t_e de ① absente : {att!r}")
    if f"{r['materiau']['ciment_detail']} — s = {r['materiau']['s']:.2f}" not in cap:
        out.append("légende du ciment")
    if a["defav"] is not None:
        d = a["defav"]
        att = (f"🌡️ Défavorable (jours prévus − {d['marge']:.0f} °C) : fck({a['t']} j) ≈ {d['fck']:.2f} MPa · t_e = {d['t_e']:.1f} j — "
               "la météo prévue n'est pas acquise ; recalcule au jour J.")
        if att not in cap:
            out.append("légende défavorable")
    elif any("Défavorable" in c for c in cap):
        out.append("légende défavorable en trop")
    if (r["essai"]["actif"] and r["essai"]["hors_domaine"]) != ("⚠️ t_e ≤ 3 j : hors domaine EC2." in cap):
        out.append("légende hors domaine de l'essai")
    # les fonctions du module : arguments et résultats
    s = r["seuil"]
    (args, _, res), = tr["t_real_for_target"]
    ec("t_real_for_target", (r["materiau"]["fck28"], r["materiau"]["s"], s["cible"], T["T"], T["Ea"], s["t"]), args + (res,))
    (args, _, res), = tr["classe_min_pour_delai"]
    o = r["optim"]
    ec("classe_min_pour_delai", (o["mpa"], r["materiau"]["s"], T["T"], o["delai"], T["Ea"],
                                 None if o["classe"] is None else (o["classe"], o["fck28"], o["fck"])), args + (res,))
    es = r["essai"]
    if es["actif"]:
        (args, _, res), = tr["classe_estimee_depuis_mesure"]
        ec("classe_estimee_depuis_mesure", (es["mpa"], es["age"], r["materiau"]["s"], es["T"], T["Ea"],
                                            {"t_e": es["t_e"], "fck28_est": es["fck28_est"], "classe": es["classe"],
                                             "fck_classe": es["fck_classe"], "ecart_pct": es["ecart_classe"]}), args + (res,))
    elif tr["classe_estimee_depuis_mesure"]:
        out.append("essai calculé alors que la mesure est nulle")
    # la figure : toutes les données de build_figure
    (fig,), c = tr["build_figure"], r["courbes"]
    ec("figure.t", c["t"], [float(x) for x in fig["t_real"]])
    ec("figure.ref", c["ref"], {"label": fig["ref"]["label"], "y": [float(x) for x in fig["ref"]["y"]]})
    ec("figure.point", c["point"], list(fig["point"]))
    ec("figure.t28", c["t28"], fig["t28_real"])
    ec("figure.titre", c["titre"], fig["titre"])
    ec("figure.seuil", c["seuil"], fig["seuil"])
    ec("figure.lab", c["lab"], fig["mesure_lab"])
    ec("figure.opt", c["opt"], None if fig["opt"] is None else {"label": fig["opt"]["label"], "y": [float(x) for x in fig["opt"]["y"]],
                                                                "t": fig["opt"]["t"], "fck": fig["opt"]["fck"]})
    ec("figure.defav", c["defav"], None if fig["defav"] is None else {"label": fig["defav"]["label"], "y": [float(x) for x in fig["defav"]["y"]]})
    ec("figure.comparateur", c["comparateur"], None if fig["comparateur"] is None else
       [{"label": x["label"], "y": [float(v) for v in x["y"]]} for x in fig["comparateur"]])
    # le mode Météo : la synthèse (comptes, T min / moy / max, avertissements, tableau, t_e du tableau)
    rel = T["releve"]
    mm = [it for (args, _, _) in tr["_mini_metrics"] for it in [args[0]]]
    if rel is not None:
        ec("mini T", [("T min", f"{rel['T_min']:.1f} °C"), ("T moy", f"{T['T']:.1f} °C"), ("T max", f"{rel['T_max']:.1f} °C")], mm[0])
        items = [("Jours", f"{rel['n_jours']}"), ("t_e", f"{rel['t_e_releve']:.1f} j")]
        if rel["t_e_releve_defav"] is not None:
            items.append(("t_e défavorable", f"{rel['t_e_releve_defav']:.1f} j"))
        ec("mini t_e", items, mm[1])
        series = [x for x in tr["age_equiv_from_series"] if x[0][1] == EC.EA_EC2 and len(x[0][0]) == rel["n_jours"]]
        ec("t_e du tableau (E_a de l'EC2, au bit)", rel["t_e_releve"], series[0][2] if series else None)
        badge = f"**{sais['Lieu du chantier']}** — {rel['n_mes']} j mesurés" + (f" · {rel['n_prev']} j de prévision" if rel["n_prev"] else "") \
            + (f" · {rel['n_extra']} j prolongés" if rel["n_extra"] else "")
        if badge not in cap:
            out.append(f"badge du relevé : {badge!r}")
        if bool(rel["n_extra"]) != any("prolongés à" in w for w in textes(ev, "warning")):
            out.append("avertissement de prolongation")
        if rel["n_extra"] and not any(f"T ≈ {rel['t_prol']:.1f} °C" in w for w in textes(ev, "warning")):
            out.append("T prolongée")
        if rel["n_nan"] and f"{rel['n_nan']} jour(s) sans donnée, remplacé(s) par la moyenne ({rel['T_rempl']:.1f} °C)." not in cap:
            out.append("légende des jours sans donnée")
        df = textes(ev, "df")[0]
        jours = M.calculer(e)["tableau"]["lignes"]
        ec("tableau Données météo", [{"Date": sais and session["ab_meteo"]["d_dates"][i],
                                      "T moy (°C)": f"{j['T_brut']:.1f}" if j["T_brut"] is not None else "—",
                                      "Source": {"prolonge": "prolongé", "prevision": "prévision", "mesure": "mesuré"}[j["source"]]}
                                     for i, j in enumerate(jours)], df)
    elif e["temperature"]["mode"] == "releve":
        if mm:
            out.append("synthèse affichée sans relevé exploitable")
        vide = not RL.lire(e["temperature"]["releve"])["temps"]
        if vide != any("T = 20 °C par défaut" in x for x in cap) or (not vide) != ("Aucune température exploitable sur la période." in textes(ev, "error")):
            out.append("message du relevé vide ou inexploitable")
    # le curseur de l'âge : ses bornes
    w = faux.widgets["ab_age"]
    ec("curseur âge (min, max)", (1, a["age_max"]), (w["mini"], w["maxi"]))
    return out, r, faux


COUV = set()
if AB is not None:
    print(f"\n=== 3. Parité avec modules/age_beton.py, coulage par coulage ({len(CAS)} cas) ===")
    for k, (desc, e) in enumerate(CAS):
        diff, r, faux = comparer(desc, e)
        n_cap = len(textes(faux.ev, "caption"))
        chk(f"cas {k + 1:02d} · {desc} — {len(r['courbes']['t'])} abscisses, {n_cap} légendes", not diff, diff[:6])
        a, s, o, es = r["age"], r["seuil"], r["optim"], r["essai"]
        COUV |= {f"âge {a['etat']}", f"seuil {s['mode']}", f"seuil {s['etat']}", f"optim {o['etat']}", f"ciment {r['materiau']['ciment']}",
                 f"classe {r['materiau']['classe']}", f"mode {r['temperature']['mode']}", f"source {r['temperature']['source']}"}
        COUV |= {"t_e ≥ 28" for _ in [0] if a["t_e"] >= 28} | {"seuil daté" for _ in [0] if s["date"]} | {"E_a ajusté" for _ in [0] if r["entree"]["avance"]}
        COUV |= {f"essai {es['verdict']}" for _ in [0] if es["actif"]} | {"essai t_e ≥ 28" for _ in [0] if es["actif"] and es["t_e"] >= 28}
        COUV |= {"essai hors domaine" for _ in [0] if es["actif"] and es["hors_domaine"]}
        COUV |= {"seuil hors domaine" for _ in [0] if s["hors_domaine"]} | {"comparateur tracé" for _ in [0] if r["courbes"]["comparateur"]}
        COUV |= {"repère t28 dans l'axe" for _ in [0] if r["courbes"]["t28"] and 1 < r["courbes"]["t28"] < 40}
        COUV |= {"repère t28 hors de l'axe" for _ in [0] if r["courbes"]["t28"] and not 1 < r["courbes"]["t28"] < 40}
        rel = r["temperature"]["releve"]
        if rel:
            COUV |= {"relevé prévision" for _ in [0] if rel["n_prev"]} | {"relevé prolongé" for _ in [0] if rel["n_extra"]}
            COUV |= {"relevé sans donnée" for _ in [0] if rel["n_nan"]} | {"relevé défavorable" for _ in [0] if a["defav"]}
            COUV |= {"âge > axe" for _ in [0] if a["t"] > 40}
        if r["temperature"]["mode"] == "releve" and not rel:
            COUV.add("relevé vide ou inexploitable")
    print(f"     {ECART['n']} comparaisons numériques : {ECART['bit']} identiques AU BIT PRÈS")
    chk("toutes les comparaisons numériques sont identiques au bit près (c'est le même calcul)", ECART["n"] == ECART["bit"] and ECART["n"] > 50000,
        (ECART["n"], ECART["bit"]))
    chk(f"{len(CAS)} coulages comparés : au moins 60", len(CAS) >= 60)

    print("\n=== 4. Couverture : ce que les cas atteignent réellement ===")
    ATTENDUS = (["âge ok", "âge warn", "seuil pct", "seuil mpa", "seuil ok", "seuil nok", "optim ok", "optim info", "optim nok",
                 "ciment R", "ciment N", "ciment S", "mode constante", "mode releve", "source manuelle", "source releve", "t_e ≥ 28",
                 "seuil daté", "E_a ajusté", "essai coherent", "essai dessus", "essai sous", "essai t_e ≥ 28", "essai hors domaine",
                 "seuil hors domaine", "comparateur tracé", "repère t28 dans l'axe", "repère t28 hors de l'axe", "relevé prévision",
                 "relevé prolongé", "relevé sans donnée", "relevé défavorable", "âge > axe", "relevé vide ou inexploitable"]
                + [f"classe {c}" for c in EC.CLASSES])
    manque = [x for x in ATTENDUS if x not in COUV]
    chk(f"les cas atteignent les {len(ATTENDUS)} situations attendues", not manque, manque)
    # une borne exacte : t_e = 3 j pile est « hors domaine » (≤, pas <) et t_e = 28 j pile donne f_ck
    r3 = M.calculer(coulage(age=3))
    r28 = M.calculer(coulage(age=28))
    chk("bornes exactes : t_e = 3 j pile → hors domaine (≤) ; t_e = 28 j pile → f_ck(t) = f_ck (t_e < 28 sinon)",
        r3["age"]["t_e"] == 3.0 and r3["age"]["hors_domaine"] and r28["age"]["t_e"] == 28.0 and r28["age"]["fck"] == 30.0)

# ------------------------------------------------------------------ 5. le relevé saisi
print("\n=== 5. Le relevé saisi (le site ne peut pas appeler Open-Meteo) ===")
r = RL.lire("12,5\n13.0\n—\n14")
chk("une température par ligne, virgule ou point ; « — » = jour sans donnée ; tout « mesuré » par défaut",
    r["temps"][:2] == [12.5, 13.0] and math.isnan(r["temps"][2]) and r["dates"] is None and not any(r["prev"]))
r = RL.lire("Date;T moy (°C);Source\n02/11/2026;8;mesuré\n2026-11-03;9,5;prévision\n04.11.2026;10;prolongé")
chk("en-tête ignoré, dates JJ/MM/AAAA, AAAA-MM-JJ, JJ.MM.AAAA ; sources : prévision → prévu, prolongé → prévu ET prolongé",
    r["dates"] == ["2026-11-02", "2026-11-03", "2026-11-04"] and r["prev"] == [False, True, True] and r["extra"] == [False, False, True])
r = RL.lire("2026-11-02,8.5,mesuré\n2026-11-03,9.5,p")
chk("un CSV à virgules (nombres à point) se lit aussi", r["temps"] == [8.5, 9.5] and r["prev"] == [False, True])
REFUS_RELEVE = (("ligne illisible au milieu", "12\nabc\n13"), ("deux températures sur une ligne", "12 13"),
                ("dates non consécutives", "2026-11-02;8\n2026-11-04;9"), ("dates dans le désordre", "2026-11-03;8\n2026-11-02;9"),
                ("date sur une ligne sur deux", "2026-11-02;8\n9"), ("source inconnue", "8;demain"),
                ("température invraisemblable", "180"), ("date seule", "2026-11-02"), ("trop de jours", "10\n" * 401),
                ("date impossible", "31/02/2026;8"))
mal = []
for nom, txt in REFUS_RELEVE:
    try:
        RL.lire(txt)
        mal.append(f"{nom} : accepté")
    except ValueError as x:
        if "ligne" not in str(x) and "jours" not in str(x):
            mal.append(f"{nom} : message sans numéro de ligne ({x})")
chk(f"{len(REFUS_RELEVE)} relevés mal formés refusés, chacun avec sa ligne (jamais lus de travers)", not mal, mal)
rr = RL.lire(releve([10, None, 11.25, -3], "mpxm", dates=True))
chk("aller-retour : lire(ecrire(r)) = r", ecarts(RL.lire(RL.ecrire(rr)), rr) == [] or all("nan" in x for x in ecarts(RL.lire(RL.ecrire(rr)), rr)),
    ecarts(RL.lire(RL.ecrire(rr)), rr))
chk("le tableau « Données météo » de l'écran Streamlit se colle tel quel (Date, T moy, Source, T min h, T max h)",
    RL.lire("Date\tT moy (°C)\tSource\tT min h\tT max h\n2026-11-02\t9.4\tmesuré\t4.1\t13.0\n2026-11-03\t—\tprévision\t\t")["prev"] == [False, True])

# ------------------------------------------------------------------ 6. format
print("\n=== 6. Format : JSON, normalisation, domaine de l'écran, formules ===")
tous = [M.calculer(e) for _, e in CAS]
chk("la sortie de calculer() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(x, allow_nan=False)) == x for x in tous))
chk("normaliser() est idempotente et {} donne nouvel_element()",
    all(EN.normaliser(EN.normaliser(e)) == EN.normaliser(e) for _, e in CAS) and EN.normaliser({}) == EN.nouvel_element())
entree = copy.deepcopy(CAS[2][1])
M.calculer(entree)
M.candidats_classes(entree)
chk("calculer() et candidats_classes() ne modifient pas le coulage reçu", entree == CAS[2][1])
chk("les libellés du module sont admis (ciment « Classe R — durcissement rapide »)", EN.normaliser({"ciment": "Classe R — durcissement rapide"})["ciment"] == "R")
REFUS = (("classe inconnue", {"classe": "C16/20"}), ("ciment inconnu", {"ciment": "X"}), ("T sous −5 °C", {"temperature": {"T": -6}}),
         ("T au-dessus de 60 °C", {"temperature": {"T": 61}}), ("T en texte", {"temperature": {"T": "20"}}), ("T infinie", {"temperature": {"T": float("inf")}}),
         ("mode inconnu", {"temperature": {"mode": "météo"}}), ("E_a 19 kJ/mol", {"Ea_kJ": 19.0}), ("E_a 61", {"Ea_kJ": 61.0}),
         ("âge 0", {"age": 0}), ("âge 29 sans relevé", {"age": 29}), ("âge non entier", {"age": 7.5}), ("âge booléen", {"age": True}),
         ("âge 6 pour un relevé de 5 jours", {"age": 6, "temperature": {"mode": "releve", "releve": "1\n2\n3\n4\n5"}}),
         ("pourcentage 72", {"seuil": {"pct": 72}}), ("pourcentage 25", {"seuil": {"pct": 25}}), ("seuil 0,5 MPa", {"seuil": {"mpa": 0.5}}),
         ("cible 0,9 MPa", {"optim": {"mpa": 0.9}}), ("délai 0,4 j", {"optim": {"delai": 0.4}}), ("mesure négative", {"essai": {"mpa": -1}}),
         ("âge d'essai 0,4 j", {"essai": {"age": 0.4}}), ("T de cure 70 °C", {"essai": {"T": 70}}), ("classe comparée inconnue", {"comparateur": {"classes": ["C12/15"]}}),
         ("classe comparée en double", {"comparateur": {"classes": ["C20/25", "C20/25"]}}), ("clé inconnue", {"beton": "C30/37"}),
         ("clé de seuil inconnue", {"seuil": {"cible": 3}}), ("date illisible", {"date_coulage": "demain"}), ("date en nombre", {"date_coulage": 20261102}),
         ("relevé illisible", {"temperature": {"mode": "releve", "releve": "douze"}}), ("case en texte", {"seuil": {"afficher": "oui"}}),
         ("coulage non objet", None),
         # les bornes hautes du SITE (l'écran n'en a pas) : 10 ans, 200 MPa — au-delà, rien de physique, et 1e308 j déborde en infini
         ("délai 3651 j", {"optim": {"delai": 3651.0}}), ("délai 1e308 j", {"optim": {"delai": 1e308}}),
         ("âge d'essai 3651 j", {"essai": {"age": 3651.0}}), ("cible 201 MPa", {"optim": {"mpa": 201.0}}),
         ("mesure 201 MPa", {"essai": {"mpa": 201.0}}), ("seuil 201 MPa", {"seuil": {"mpa": 201.0}}))
refusees = []
for nom, champ in REFUS:
    try:
        EN.normaliser(champ if champ is None else {**EN.nouvel_element(), **{k: ({**EN.nouvel_element()[k], **v} if isinstance(v, dict) else v) for k, v in champ.items()}})
        refusees.append(f"{nom} : accepté")
    except ValueError:
        pass
    except Exception as x:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
        refusees.append(f"{nom} : {type(x).__name__} {x}")
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError (jamais de valeur ramenée en silence)", not refusees, refusees)
# « formule = valeurs = résultat » : chaque résultat se retrouve à partir des seuls termes que le moteur donne
FORMULES = {
    "T_abs": lambda t, x: t["T"] + 273.15,
    "facteur": lambda t, x: math.exp(-t["E_a/R"] * (1.0 / t["T_abs"] - 1.0 / t["T_ref"])),
    "t_e": lambda t, x: t["t"] * t["facteur"],
    "t_e_releve": lambda t, x: x["releve_te"](t["n"]),
    "beta": lambda t, x: float(np.exp(t["s"] * (1.0 - np.sqrt(28.0 / max(t["t_e"], 1e-6))))),
    "fcm": lambda t, x: t["fck"] + 8.0,
    "fck_t": lambda t, x: t["beta"] * t["fcm"] - 8.0,
    "fck_t_28": lambda t, x: t["fck"],
    "pct": lambda t, x: t["fck_t"] / t["fck"] * 100.0,
    "cible_pct": lambda t, x: t["fck"] * t["p"] / 100.0,
    "t_e_cible": lambda t, x: 28.0 / (1.0 - math.log((t["cible"] + 8.0) / t["fcm"]) / t["s"]) ** 2,
    "t_cible": lambda t, x: t["t_e"] / t["facteur"],
    "t_seuil": lambda t, x: x["t_seuil"],
    "fck_seuil": lambda t, x: x["fck_de"](t["t_e"]),
    "t_e_delai": lambda t, x: t["delai"] * t["facteur"],
    "fck28_est": lambda t, x: (t["f_mes"] + 8.0) / t["beta"] - 8.0,
    "fck28_est_28": lambda t, x: t["f_mes"],
    "ecart_classe": lambda t, x: (t["fck28_est"] - t["fck"]) / t["fck"] * 100.0,
    "ecart": lambda t, x: (t["fck28_est"] - t["fck"]) / t["fck"] * 100.0,
}
fautes, n_f = [], 0
for (desc, e), r in zip(CAS, tous):
    prof = r["temperature"]["profil"]
    x = {"releve_te": lambda n, p=prof, Ea=r["temperature"]["Ea"]: EC.age_equivalent_serie(p[:n], Ea), "t_seuil": r["seuil"]["t"],
         "fck_de": lambda te, rr=r: float(EC.fck_age_equivalent(rr["materiau"]["fck28"], rr["materiau"]["s"], te))}
    for bloc in ("age", "seuil", "optim", "essai"):
        for f in r[bloc].get("formules", []):
            n_f += 1
            t = {y["symbole"]: y["valeur"] for y in f["termes"]}
            v = FORMULES[f["cle"]](t, x)
            if not proche(v, f["resultat"], 1e-12):
                fautes.append(f"{desc}/{bloc}/{f['cle']} : {v!r} ≠ {f['resultat']!r}")
chk(f"« formule = valeurs = résultat » : les {n_f} formules se recalculent à partir de leurs seuls termes", not fautes and n_f > 400, fautes[:4])
c = M.candidats_classes(CAS[1][1])
chk("candidats_classes : les 7 classes dans l'ordre de résistance, la retenue = la classe du coulage, l'optimale = classe_min_pour_delai",
    [x["classe"] for x in c] == EC.CLASSES and [x["classe"] for x in c if x["retenue"]] == [CAS[1][1]["classe"]]
    and [x["classe"] for x in c if x["optimal"]] == [M.calculer(CAS[1][1])["optim"]["classe"]])
doc = lire("docs", "beton", "MATURITE_MOTEUR.md") if os.path.exists(os.path.join(RACINE, "docs", "beton", "MATURITE_MOTEUR.md")) else ""
cles = ["t_e", "fck", "pct", "hors_domaine", "cible", "t_seuil", "classe", "actuelle", "fck28_est", "ecart", "verdict", "T_moy", "defav", "facteur",
        "t28_reel", "t_e_releve"]
sans_doc = [k for k in cles if f"`{k}`" not in doc]
chk(f"les {len(cles)} grandeurs principales sont nommées dans docs/beton/MATURITE_MOTEUR.md", not sans_doc, sans_doc)

# ------------------------------------------------------------------ 7. pureté
print("\n=== 7. Pureté : le moteur se charge et calcule SANS streamlit ni matplotlib, avec la bibliothèque standard et numpy ===")
non_std = []
for dossier, _, fichiers in os.walk(os.path.join(RACINE, "beton", "maturite")):
    for f in fichiers:
        if f.endswith(".py"):
            arbre = ast.parse(lire(os.path.relpath(os.path.join(dossier, f), RACINE)))
            for n in ast.walk(arbre):
                noms = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                    [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
                non_std += [f"{f}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"beton", "numpy"}]
chk("aucun fichier de beton/maturite n'importe autre chose que la bibliothèque standard et numpy (lecture des sources)", not non_std, non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "matplotlib", "scipy", "requests", "pymupdf", "fitz", "PIL")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
try:
    import matplotlib  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from beton.maturite import moteur as M
coulages = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.calculer(e) for e in coulages],
          "cands": M.candidats_classes(coulages[1])}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"beton", "numpy"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
coulages = [e for _, e in CAS]
attendu = {"res": json.loads(json.dumps(tous)), "cands": json.loads(json.dumps(M.candidats_classes(coulages[1])))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(coulages), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, matplotlib, pandas, requests… REFUSÉS, et le moteur calcule quand même ({len(out['res'])} coulages)",
        out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard et numpy", not out["tiers"], out["tiers"])
    diff = ecarts({"res": out["res"], "cands": out["cands"]}, attendu)
    chk(f"Python {v} : résultats identiques à ceux de ce processus (au bit près)", not diff, diff[:3])

# ------------------------------------------------------------------ 8. mutations
# Chaque mutation fausse le moteur d'une façon plausible (une constante, une borne, une formule, une branche) ; la parité
# avec l'écran Streamlit DOIT la voir : on rejoue les coulages jusqu'au premier écart. Une mutation que la parité ne voit
# pas serait un trou de la preuve.
if AB is not None:
    import functools
    print("\n=== 8. Mutations : chaque défaut glissé dans le moteur fait rougir la parité ===")

    def _beta_faux(t, s):
        t = np.maximum(np.asarray(t, dtype=float), 1e-6)
        return np.exp(s * (1.0 - np.sqrt(27.0 / t)))

    def _fck_faux(fck28, s, t_e):
        t_e = np.asarray(t_e, dtype=float)
        return np.where(t_e < 28.0, EC.beta_cc(t_e, s) * (fck28 + 7.0) - 7.0, float(fck28))

    def _classe_plus_basse(res, age, s, T, Ea=EC.EA_EC2):
        d = EC.classe_estimee_depuis_mesure(res, age, s, T, Ea)
        i = max(0, EC.CLASSES.index(d["classe"]) - 1)
        return dict(d, classe=EC.CLASSES[i], fck_classe=EC.fck_classe(EC.CLASSES[i]))

    MUTATIONS = [
        ("β_cc : 27 au lieu de 28 sous la racine", [(M, "beta_cc", _beta_faux), (EC, "beta_cc", _beta_faux)]),
        ("f_cm = f_ck + 7 au lieu de + 8", [(M, "fck_age_equivalent", _fck_faux)]),
        ("T_ref = 293,0 K au lieu de 293,15 K", [(EC, "T_REF_K", 293.0)]),
        ("domaine : t_e < 3 j au lieu de ≤ 3 j", [(M, "SEUIL_DOMAINE", 2.999999)]),
        ("essai cohérent dans ± 10 % au lieu de ± 8 %", [(M, "ECART_COHERENT", 10.0)]),
        ("marge défavorable 2,5 °C au lieu de 3 °C", [(RL, "profil_defavorable", lambda t, p: EC.profil_defavorable(t, p, 2.5))]),
        ("① à la température moyenne au lieu du relevé jour par jour",
         [(M, "age_equivalent_point", lambda t, T, prof, Ea: float(EC.age_equivalent(t, T, Ea)))]),
        ("seuil : tolérance 0,01 MPa au lieu de 0,001", [(M, "temps_reel_pour_cible", functools.partial(EC.temps_reel_pour_cible, tol=1e-2))]),
        ("classe minimale : E_a de l'EC2 même ajustée", [(M, "classe_min_pour_delai", lambda c, s, T, d, Ea=EC.EA_EC2: EC.classe_min_pour_delai(c, s, T, d))]),
        ("essai : la classe inférieure au lieu de la plus proche", [(M, "classe_estimee_depuis_mesure", _classe_plus_basse)]),
        ("jour sans donnée : 0 °C au lieu de la moyenne", [(RL, "synthese", lambda rel: RL_SYNTHESE(dict(rel, temps=[0.0 if math.isnan(t) else t for t in rel["temps"]])))]),
    ]
    RL_SYNTHESE = RL.synthese
    vues = []
    for nom, patchs in MUTATIONS:
        anciens = [(mod, att, getattr(mod, att)) for mod, att, _ in patchs]
        for mod, att, v in patchs:
            setattr(mod, att, v)
        trouve = None
        try:
            for k, (desc, e) in enumerate(CAS):
                try:
                    diff = comparer(desc, e)[0]
                except Exception as x:  # noqa: BLE001 — une mutation qui fait planter le moteur est vue aussi
                    diff = [f"{type(x).__name__} : {x}"]
                if diff:
                    trouve = (k + 1, diff[0])
                    break
        finally:
            for mod, att, v in anciens:
                setattr(mod, att, v)
        vues.append(trouve is not None)
        chk(f"mutation « {nom} » : vue par la parité" + (f" (cas {trouve[0]:02d} : {str(trouve[1])[:70]})" if trouve else ""), trouve is not None)
    chk(f"les {len(MUTATIONS)} mutations rougissent toutes ; après elles, le moteur rendu est intact (cas 01 identique)",
        all(vues) and not comparer(*CAS[0])[0])

# ------------------------------------------------------------------ 9. le site affiche les chiffres de l'écran
# De bout en bout : ce que l'ÉCRAN STREAMLIT AFFICHE (ses blocs, ses légendes, ses mesures, son tableau, sa figure,
# rejoués avec le faux « st ») face à ce que le SITE AFFICHE (api_web/routeur.traiter → fragments HTML et SVG), texte
# contre texte, à l'arrondi d'affichage de l'écran. Les sections 3 et 6 prouvent moteur = écran ; celle-ci prouve que
# rien ne se perd ni ne se déforme entre le moteur et ce que l'utilisateur lit.
if AB is not None:
    import html as _html
    import re as _re

    from api_web import routeur as ROUTEUR
    print("\n=== 9. Le site affiche les chiffres de l'écran Streamlit (de bout en bout, texte contre texte) ===")

    def _site(e):
        return json.loads(ROUTEUR.traiter(json.dumps({"module": "age", "action": "calculer", "etat": {"version": 1, "coulages": [e], "actif": 0}})))

    def _txt(h):
        return _html.unescape(_re.sub(r"<[^>]+>", "", h))

    def _f(s):
        """Un nombre affiché par l'écran (« -3.99 ») écrit comme le site l'écrit (« −3,99 »)."""
        return s.replace("-", "−").replace(".", ",")

    def _nb(s):
        """« 14.0 » (l'écran) → « 14 » (le site écrit une saisie sans zéro inutile)."""
        return _f(s[:-2] if s.endswith(".0") else s)

    def bout_a_bout(e):
        faux, tr = ecran(*saisies(e))
        s = _site(e)
        if not s["ok"]:
            return [f"site : {s['erreur']}"], 0
        res, vol, don = _txt(s["resultat"]), _txt(s["volets"]), _txt(s["donnees"])
        svg = next(v["svg"] for v in s["vues"] if v["cle"] == "courbe")
        tab = _txt(next(v["svg"] for v in s["vues"] if v["cle"] == "tableau"))
        blocs = [(x[0][0], x[0][1] if len(x[0]) > 1 else "", x[0][2] if len(x[0]) > 2 else "ok") for x in tr["_bloc"]]
        cap = textes(faux.ev, "caption")
        out, n = [], 0

        def voit(texte, ou, nom):
            nonlocal n
            n += 1
            if not AA.present(texte, ou):     # au demi exact près (le site arrondit demi vers l'extérieur, l'écran au pair)
                out.append(f"{nom} : « {texte} » absent du site")
        # ① « fck(t j à T °C) » — « x MPa · y % de fck(28) » ; la légende « t_e = x j »
        m1 = _re.match(r"([-\d.]+) MPa · ([-\d]+) % de fck\(28\)$", blocs[0][1])
        voit(f"{_f(m1.group(1))} MPa · te", res, "① résistance")
        voit(f"{_f(m1.group(2))} % de fck = ", res, "① pourcentage")
        te = next(_re.match(r"t_e = ([\d.]+) j", c).group(1) for c in cap if c.startswith("t_e = "))
        voit(f"te = {_f(te)} j", res, "① t_e")
        if blocs[0][2] == "warn":
            voit("hors du domaine de l'EC2 (te ≤ 3 j)", res, "① hors domaine")
        dfv = next((c for c in cap if c.startswith("🌡️ Défavorable")), None)
        if dfv:
            md = _re.search(r"≈ ([-\d.]+) MPa · t_e = ([\d.]+) j", dfv)
            voit(f"Défavorable à {e['age']} j{_f(md.group(1))} MPa · te", res, "① défavorable")
            te_site = float(_re.search(r"prévisions − 3 °C · te = ([\d,]+) j", vol).group(1).replace(",", "."))
            n += 1
            if abs(te_site - float(md.group(2))) > 0.05 + 0.005:
                out.append(f"① t_e défavorable : {te_site} ≠ {md.group(2)}")
        # ② « Seuil 75 % = 22.5 MPa » — « après 7.9 j — le 09/11/2026 » / « non atteignable »
        titre, droite = blocs[1][0], blocs[1][1]
        voit(f"seuil {_f(titre[len('Seuil '):])}", vol, "② cible")
        if droite == "non atteignable":
            voit("MPa · non atteignable", res, "② non atteignable")
        else:
            ms = _re.match(r"après ([\d.]+) j(?: — le (\S+))?$", droite)
            voit(f"MPa · après {_f(ms.group(1))} j" + (f" · {ms.group(2)}" if ms.group(2) else ""), res, "② temps et date")
        # ③ « Classe minimale : X (actuelle) » — « f MPa à d j » / « Aucune classe pour f MPa en d j »
        titre, droite = blocs[2][0], blocs[2][1]
        if titre.startswith("Aucune classe"):
            ma = _re.match(r"Aucune classe pour ([\d.]+) MPa en ([\d.]+) j", titre)
            voit("Classe minimaleaucune ne suffit", res, "③ aucune")
            voit(f"classe pour {_nb(ma.group(1))} MPa en {_nb(ma.group(2))} j→ aucune classe", vol, "③ aucune : cible et délai")
        else:
            cl = titre.split(" : ")[1].split(" ")[0]
            mo = _re.match(r"([-\d.]+) MPa à ([\d.]+) j", droite)
            voit(f"Classe minimale{cl} · {_f(mo.group(1))} MPa", res, "③ classe minimale")
            voit(f"classe pour", vol, "③ fenêtre")
            n_d = _nb(mo.group(2))
            voit(f"MPa en {n_d} j→ {cl}", vol, "③ délai")
            if "(actuelle)" in titre:
                voit(f"→ {cl} (la classe du coulage suffit)", vol, "③ actuelle")
        # ⑤ « Classe estimée » — « ≈ X (fck28 ≈ f MPa) » ; « Cohérent avec / Au-dessus de / Sous Y » — « écart ±e % »
        if len(blocs) > 3:
            me = _re.match(r"≈ (\S+) \(fck28 ≈ ([-\d.]+) MPa\)", blocs[3][1])
            voit(f"≈ {me.group(1)} · fck,28 ≈ {_f(me.group(2))} MPa", res, "⑤ classe estimée")
            verdict = blocs[4][0][0].lower() + blocs[4][0][1:]
            voit(f"{verdict} (écart {_f(blocs[4][1].split(' ')[1])} %)", res, "⑤ verdict")
            if "⚠️ t_e ≤ 3 j : hors domaine EC2." in cap:
                voit("Essai : te ≤ 3 j, hors du domaine de l'EC2.", res, "⑤ hors domaine")   # t_e écrit en indice
        # le relevé : les mesures, le badge, la prolongation, les jours sans donnée, le tableau « Données météo »
        mm = [args[0] for (args, _, _) in tr["_mini_metrics"]]
        if mm:
            for lbl, val in mm[0]:
                voit(f"{lbl}{_f(val)}", don, f"mesure {lbl}")
            for lbl, val in mm[1]:
                if lbl == "Jours":
                    voit(f"relevé de {val} j", don, "jours du relevé")
                else:
                    voit(f"{'te' if lbl == 't_e' else 'te défav.'}{_f(val)}", don, f"mesure {lbl}")
            badge = next(c for c in cap if c.startswith("**"))
            nombres = _re.findall(r"(\d+) j (mesurés|de prévision|prolongés)", badge)
            for k_, mot in nombres:
                voit(f"{k_} j {'mesuré' if mot == 'mesurés' else mot.rstrip('s') if mot == 'prolongés' else mot}", don, f"badge {mot}")
            for w in textes(faux.ev, "warning"):
                mp = _re.search(r"T ≈ ([\d.-]+) °C", w)
                if mp:
                    voit(f"T ≈ {_f(mp.group(1))} °C", res, "prolongation")
            for c_ in cap:
                mn = _re.match(r"(\d+) jour\(s\) sans donnée, remplacé\(s\) par la moyenne \(([-\d.]+) °C\)", c_)
                if mn:
                    voit(f"{mn.group(1)} jour(s) sans donnée, remplacé(s) par la moyenne ({_f(mn.group(2))} °C)", res, "jours sans donnée")
            for row in textes(faux.ev, "df")[0]:
                date = _dt.date.fromisoformat(row["Date"]).strftime("%d/%m/%Y") if e["temperature"]["releve"].count("-") >= 2 else ""
                voit(f"{date}{_f(row['T moy (°C)'])}{row['Source']}", tab, "tableau Données météo")
        # la figure : l'annotation du point, les légendes du seuil, de l'essai, de la classe minimale, du défavorable
        (fig,) = tr["build_figure"]
        ts, fs = fig["point"]
        if ts <= 120:
            voit(f"{_f(f'{fs:.1f}')} MPa à {ts:.0f} j", svg, "figure : point ①")
        if fig["seuil"] and fig["seuil"].get("t") is not None:
            voit(f"seuil ② : {_f(f'{fig['seuil']['fck']:.1f}')} MPa à {_f(f'{fig['seuil']['t']:.1f}')} j", svg, "figure : seuil")
        if fig["mesure_lab"] and fig["mesure_lab"].get("fck", 0) > 0:
            voit(f"essai ⑤ : {_f(f'{fig['mesure_lab']['fck']:.1f}')} MPa à {_f(f'{fig['mesure_lab']['age']:.1f}')} j", svg, "figure : essai")
        if fig["opt"]:
            voit(fig["opt"]["label"].replace(" (optim.)", " · classe minimale ③"), svg, "figure : classe minimale")
        if fig["defav"]:
            voit(fig["defav"]["label"], svg, "figure : défavorable")
        for cmp_ in fig["comparateur"] or []:
            voit(f">{cmp_['label']}", svg, "figure : classe comparée")
        return out, n

    CHOISIS = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 19, 21, 22, 23, 24, 25, 28, 33, 34, 36, 37, 38, 39, 43, 44, 45, 70, 80, 96]
    total, fautes9 = 0, []
    for k in CHOISIS:
        desc, e = CAS[k]
        d, n = bout_a_bout(e)
        total += n
        if d:
            fautes9.append((k + 1, desc, d[:3]))
    chk(f"{len(CHOISIS)} coulages variés (défauts, hors domaine, −5 et 60 °C, seuil non atteignable / daté / en MPa, classe actuelle / inférieure / "
        f"aucune, comparateur, essais cohérent / dessus / sous / hors domaine, E_a, f_ck ≤ 0, relevés prévus / prolongés / sans donnée, tirages) : "
        f"les {total} valeurs que l'écran Streamlit affiche sont lues sur le site, au même arrondi", not fautes9 and len(CHOISIS) >= 15, fautes9[:3])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Moteur « Taux d'armature » (beton/taux) : la PARITÉ avec l'écran Streamlit modules/taux_armature.py, qui reste la
référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Données : listes, libellés, bornes de chaque champ et défauts LUS dans le module (source et état initial).
  2. À la main : deux poutres recalculées ici, en m, kg et m³.
  3. Parité (plus de 50 poutres) : chaque poutre est chargée dans l'état du module, puis son écran ENTIER (show) est
     exécuté avec un faux « st » qui enregistre au lieu d'afficher, et des espions sur build_geometry,
     build_rebar_families, compute_stirrups_for_zones, compute_quantities et draw_section. On compare au moteur, AU BIT
     PRÈS : géométrie, polygone, aire, chaque barre dessinée (x, y, Ø, couleur), codes, longueurs, étriers, volumes
     par catégorie, masses, taux ; puis les CHAÎNES affichées : les trois chiffres, la synthèse, chaque ligne du
     tableau récapitulatif.
  4. Couverture : chaque forme, position, rôle, Ø, nombre de retours, mode de saisie, défaut reproduit… est atteint.
  5. Effets de bord de la saisie (beton/taux/saisie.py) contre l'écran : boutons d'ajout, forme, retours, types.
  6. Constats sur l'écran Streamlit (AppTest, le vrai moteur de Streamlit) : chaque défaut écrit dans la doc est
     MONTRÉ ici ; si l'écran est corrigé, le constat rougit et la doc est à reprendre.
  7. Mutations : chaque altération plausible du moteur fait rougir la parité.
  8. Format : JSON strict, normalisation (défauts, ValueError hors du domaine), formules recalculables.
  9. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, pandas,
     numpy, matplotlib… sous Python 3.12 et 3.13 (celui de Pyodide).

    /usr/bin/python3.12 tests/test_taux_moteur.py       (streamlit, pandas et matplotlib requis pour la parité)
"""
import ast
import contextlib
import copy
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

from beton.taux import entree as EN  # noqa: E402
from beton.taux import geometrie as GE  # noqa: E402
from beton.taux import moteur as M  # noqa: E402
from beton.taux import saisie as SA  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as fh:
        return fh.read()


STATS = {"n": 0, "bit": 0}


def egal(a, b):
    """Égalité STRICTE (au bit près pour les nombres) ; le compte des comparaisons numériques est tenu."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        STATS["n"] += 1
        if a == b:
            STATS["bit"] += 1
            return True
        return False
    return a == b


# ------------------------------------------------------------------ les cas, au format du moteur
def fam(role, position=None, n=2, d=16, r=0, hr=0.0, de=0.0, a=100.0):
    f = {"role": role, "position": position or {"principal": "ame_inf", "renfort": "ame_sup", "lateral": "lat_g"}[role],
         "n": n, "d": d, "retours": r, "h_retour": hr, "de": de, "a": a}
    return f


def poutre(nom, familles=(), types=(), zones=(), metres=None, **kw):
    """Une poutre ; ``metres`` : {index de famille: (De en m, À en m)} SAISIS en mètres (mode « metre »)."""
    p = EN.nouvelle_poutre()
    p["nom"] = nom
    p.update({k: v for k, v in kw.items() if k not in ("tg", "td")})
    if "tg" in kw:
        p["talon_gauche"] = dict(zip(("b", "h"), kw["tg"]))
    if "td" in kw:
        p["talon_droit"] = dict(zip(("b", "h"), kw["td"]))
    p["familles"] = [dict(f) for f in familles]
    p["types_etriers"] = [{"nom": n, "d": d} for n, d in types]
    p["zones"] = [{"nom": n, "de": de, "a": a, "s": s, "type": t} for n, de, a, s, t in zones]
    p = EN.normaliser(p)
    p["_metres"] = metres or {}
    if metres:
        p["longueurs"] = "metre"
        for i, (m1, m2) in metres.items():
            SA.longueur_en_metres(p, p["familles"][i], "de", m1)
            SA.longueur_en_metres(p, p["familles"][i], "a", m2)
    return p


def sans(p):
    """La poutre sans les annotations du test."""
    return {k: v for k, v in p.items() if not k.startswith("_")}


E1 = (("E1", 8.0),)
CAS = [
    ("défauts de l'écran (aucune armature)", poutre("P0")),
    ("le cas des constats : 2Ø16 en bas, E1 Ø8 de 0 à 33 % à 15 cm", poutre("P1", [fam("principal")], E1, [("Z1", 0, 33, 15, "E1")])),
    ("poutre complète : lit bas, renfort haut, latérales 2 côtés, deux zones",
     poutre("P2", [fam("principal", n=3, d=20), fam("renfort", n=2, d=12, de=0, a=30), fam("lateral", "lat_2", n=2, d=10, r=2, hr=24)],
            E1, [("Z1", 0, 25, 10, "E1"), ("Z2", 25, 75, 20, "E1"), ("Z3", 75, 100, 10, "E1")])),
    ("talon gauche 25 × 20 : renfort de talon en bas et en haut", poutre(
        "TG", [fam("principal", n=4, d=25), fam("renfort", "talon_g_inf", n=3, d=16), fam("renfort", "talon_g_sup", n=2, d=12)],
        forme="talon_gauche", tg=(25.0, 20.0))),
    ("talon droit 15 × 30, lit principal posé sur le talon", poutre(
        "TD", [fam("principal", "talon_d_inf", n=2, d=14), fam("principal", n=3, d=16)], forme="talon_droit", td=(15.0, 30.0))),
    ("deux talons, familles sur l'âme seulement (talons IGNORÉS par le module)", poutre(
        "DT", [fam("principal", n=3, d=16), fam("renfort", n=2, d=12)], E1, [("Z1", 0, 100, 20, "E1")], forme="deux_talons",
        tg=(20.0, 25.0), td=(30.0, 35.0))),
    ("trois lits en bas (rangs de lit 0, 1, 2), deux en haut", poutre(
        "LITS", [fam("principal", n=4, d=25), fam("principal", n=3, d=20), fam("renfort", n=2, d=16, de=20, a=80),
                 fam("renfort", n=2, d=14), fam("principal", "ame_sup", n=2, d=10)])),
    ("une barre seule (au milieu), vingt barres", poutre("N", [fam("principal", n=1, d=32), fam("renfort", n=20, d=6)])),
    ("latérales : une barre à mi-hauteur, gauche, droite", poutre(
        "LAT", [fam("lateral", "lat_g", n=1, d=12), fam("lateral", "lat_d", n=3, d=10), fam("lateral", "lat_2", n=4, d=8, r=1, hr=18)])),
    ("retours : un et deux, hauteurs différentes", poutre(
        "RET", [fam("principal", n=3, d=16, r=1, hr=54), fam("renfort", n=2, d=12, r=2, hr=37.5)])),
    ("À ≤ De : longueur nulle (0 kg) ; À = De", poutre("NUL", [fam("principal", de=60, a=40), fam("renfort", de=50, a=50), fam("principal", n=3)])),
    ("chaque Ø de l'écran", poutre("PHI", [fam("principal", n=2, d=d) for d in EN.DIAMETRES[:5]] + [fam("renfort", n=2, d=d) for d in EN.DIAMETRES[5:]],
                                    b=60.0, h=120.0)),
    ("troncature des étriers : L = 1 m, 0 à 30 %, s = 10 cm (3 étriers au lieu de 4)",
     poutre("TRONC", [fam("principal")], E1, [("Z1", 0, 30, 10, "E1")], L=1.0)),
    ("zones qui se recouvrent (comptées deux fois)", poutre("REC", [fam("principal")], E1, [("Z1", 0, 50, 15, "E1"), ("Z2", 30, 100, 20, "E1")])),
    ("zone À ≤ De (ignorée)", poutre("ZNUL", [fam("principal")], E1, [("Z1", 40, 40, 15, "E1"), ("Z2", 0, 100, 25, "E1")])),
    ("zones sans type (aucun type : ignorées)", poutre("ZST", [fam("principal")], (), [("Z1", 0, 100, 15, "")])),
    ("plusieurs types d'étriers (le site les permet)", poutre(
        "TYP", [fam("principal", n=4, d=20)], (("E1", 8.0), ("E2", 12.0), ("E3", 6.0)),
        [("Z1", 0, 20, 8, "E2"), ("Z2", 20, 80, 25, "E1"), ("Z3", 80, 100, 8, "E3")])),
    ("étrier Ø16 et Ø7,5 (l'écran admet tout nombre de 6 à 16)", poutre(
        "ETR", [fam("principal")], (("E1", 16.0), ("E2", 7.5)), [("Z1", 0, 50, 12.5, "E1"), ("Z2", 50, 100, 30, "E2")])),
    ("majoration 0 %", poutre("M0", [fam("principal", n=3, d=20)], E1, [("Z1", 0, 100, 20, "E1")], majoration=0.0)),
    ("majoration 50 %", poutre("M50", [fam("principal", n=3, d=20)], E1, [("Z1", 0, 100, 20, "E1")], majoration=50.0)),
    ("majoration 7,5 %, acier B500C", poutre("M75", [fam("principal", n=3, d=20)], majoration=7.5, acier="B500C")),
    ("portée minimale 0,5 m", poutre("L05", [fam("principal")], E1, [("Z1", 0, 100, 10, "E1")], L=0.5)),
    ("portée maximale 60 m", poutre("L60", [fam("principal", n=6, d=32)], E1, [("Z1", 0, 100, 20, "E1")], L=60.0)),
    ("petite section 12 × 10, enrobage 1 cm", poutre("XS", [fam("principal", n=2, d=8)], (("E1", 6.0),), [("Z1", 0, 100, 10, "E1")],
                                                       b=12.0, h=10.0, enrobage=1.0)),
    ("grande section 200 × 300, enrobage 10 cm", poutre("XL", [fam("principal", n=12, d=40), fam("lateral", "lat_2", n=8, d=16)],
                                                         E1, [("Z1", 0, 100, 15, "E1")], b=200.0, h=300.0, enrobage=10.0)),
    ("valeurs non entières (27,5 × 52,5 ; L = 7,35 m ; c = 2,5)", poutre(
        "DEC", [fam("principal", n=3, d=14, de=2.5, a=97.5), fam("renfort", n=2, d=10, r=1, hr=12.25)], E1, [("Z1", 12.5, 87.5, 17.5, "E1")],
        b=27.5, h=52.5, L=7.35, enrobage=2.5)),
    ("saisie en mètres (De, À en m)", poutre("MET", [fam("principal", n=3, d=16), fam("renfort", n=2, d=12)], E1, [("Z1", 0, 33, 15, "E1")],
                                             metres={0: (0.0, 6.0), 1: (0.35, 1.98)})),
    ("saisie en mètres, L = 7,35 m", poutre("MET2", [fam("principal", n=2, d=20), fam("lateral", "lat_2", n=2, d=10)], L=7.35,
                                             metres={0: (0.1, 7.35), 1: (1.0, 6.2)})),
    ("talons à la limite (300 × 300)", poutre("TGXL", [fam("renfort", "talon_g_inf", n=6, d=25)], forme="talon_gauche", tg=(300.0, 300.0))),
    ("talons de 0,5 cm (enrobage plus grand : barres hors du talon, sans effet sur les quantités)", poutre(
        "TG05", [fam("renfort", "talon_d_inf", n=2, d=10), fam("renfort", "talon_d_sup", n=2, d=10)], forme="talon_droit", td=(0.5, 0.5))),
]
RNG = random.Random(20261107)


def aleatoire(k):
    """Une poutre tirée au sort (graine fixe), dans le domaine de l'écran."""
    forme = RNG.choice(list(EN.FORMES))
    b, h = RNG.choice((20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 22.5)), RNG.choice((30.0, 40.0, 50.0, 60.0, 75.0, 90.0, 47.5))
    c = RNG.choice((1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0))
    p = {"forme": forme, "b": b, "h": h, "enrobage": c, "L": RNG.choice((2.5, 4.0, 5.4, 6.0, 7.2, 8.0, 11.5, 13.25)),
         "majoration": RNG.choice((0.0, 3.0, 5.0, 10.0, 12.5)), "tg": (RNG.choice((10.0, 15.0, 22.5)), RNG.choice((15.0, 20.0))),
         "td": (RNG.choice((10.0, 20.0)), RNG.choice((12.0, 25.0)))}
    fams = []
    for _ in range(RNG.randint(0, 7)):
        role = RNG.choice(("principal", "principal", "renfort", "lateral"))
        opts = EN.options_positions(forme, role)
        if forme == "deux_talons":
            opts = [o for o in opts if not o.startswith("talon")]
        de = RNG.choice((0.0, 0.0, 5.0, 10.0, 25.0, 33.0, 60.0))
        a = RNG.choice((100.0, 100.0, 95.0, 75.0, 40.0, 66.6))
        r = RNG.choice((0, 0, 1, 2))
        fams.append(fam(role, RNG.choice(opts), RNG.randint(1, 8), RNG.choice(EN.DIAMETRES), r, RNG.choice((10.0, 24.0, 35.5)) if r else 0.0, de, a))
    types = [(f"E{i + 1}", RNG.choice((6.0, 8.0, 10.0, 12.0))) for i in range(RNG.choice((0, 1, 1, 2)))]
    zones = []
    for i in range(RNG.randint(0, 4) if types else 0):
        de = RNG.choice((0.0, 10.0, 20.0, 25.0, 33.0, 50.0))
        zones.append((f"Z{i + 1}", de, min(100.0, de + RNG.choice((10.0, 20.0, 33.0, 50.0, 80.0))), RNG.choice((7.5, 10.0, 12.0, 15.0, 20.0, 25.0)),
                      RNG.choice(types)[0]))
    return poutre(f"Aléa {k}", fams, types, zones, **p)


CAS += [(f"tirage {k:02d} (graine 20261107)", aleatoire(k)) for k in range(1, 31)]


def cas(debut):
    return next(p for d, p in CAS if d.startswith(debut))


# ------------------------------------------------------------------ 1. données
print("=== 1. Données : listes, libellés, bornes et défauts lus dans le module ===")
try:
    logging.disable(logging.WARNING)
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import streamlit  # noqa: E402,F401
    import modules.taux_armature as T  # noqa: E402
    import modules.navigation as NAV  # noqa: E402
except ImportError as e:  # pragma: no cover — la parité EXIGE le module
    T = None
    chk("streamlit, matplotlib, pandas et modules/taux_armature.py se chargent (indispensables à la parité)", False, e)

SRC = lire("modules", "taux_armature.py")
ARBRE = ast.parse(SRC)


def appels(nom_widget):
    """Les appels st.<nom_widget>(…) du module : [(libellé, {mot-clé: valeur constante}, nœud)]."""
    out = []
    for n in ast.walk(ARBRE):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == nom_widget and n.args \
                and isinstance(n.args[0], ast.Constant):
            kw = {}
            for k in n.keywords:
                try:
                    kw[k.arg] = ast.literal_eval(k.value)
                except ValueError:
                    kw[k.arg] = None
            out.append((n.args[0].value, kw, n))
    return out


if T is not None:
    chk("les libellés des positions sont ceux de l'écran (POSITION_LABELS)", {v[0]: v[1] for v in EN.POSITIONS.values()} == T.POSITION_LABELS)
    chk("les libellés des rôles sont ceux de l'écran (ROLE_LABELS)", {v[0]: v[1] for v in EN.ROLES.values()} == T.ROLE_LABELS)
    selects = {lab: n for lab, _, n in appels("selectbox")}
    chk("les formes sont celles du menu « Section », dans l'ordre", list(EN.FORMES.values()) == ast.literal_eval(selects["Section"].args[1]))
    chk("les Ø des familles sont ceux du menu « Ø (mm) »", list(EN.DIAMETRES) == ast.literal_eval(selects["Ø (mm)"].args[1]))
    chk("les retours sont ceux du menu « Retours »", list(EN.RETOURS) == ast.literal_eval(selects["Retours"].args[1]))
    chk("les qualités d'acier sont celles du menu « Qualité d'acier »", list(EN.ACIERS) == ast.literal_eval(selects["Qualité d’acier"].args[1]))
    BORNES_ECRAN = {"Enrobage c (cm)": "enrobage", "Portée L (m)": "L", "Largeur b (cm)": "b", "Hauteur h (cm)": "h",
                    "Talon gauche – largeur (cm)": "talon", "Talon gauche – hauteur (cm)": "talon", "Talon droit – largeur (cm)": "talon",
                    "Talon droit – hauteur (cm)": "talon", "n barres": "n", "Hauteur retour (cm)": "h_retour", "De (%L)": "pct", "À (%L)": "pct",
                    "Ø (mm)": "d_etrier", "Espacement s (cm)": "s", "Majoration globale (%)": "majoration"}
    lues, fausses = 0, []
    for lab, kw, _ in appels("number_input"):
        if lab in BORNES_ECRAN:
            lues += 1
            if (kw.get("min_value"), kw.get("max_value")) != EN.BORNES[BORNES_ECRAN[lab]]:
                fausses.append((lab, kw.get("min_value"), kw.get("max_value")))
    chk(f"les bornes des {lues} champs numériques à bornes fixes sont celles de l'écran (min_value, max_value)", lues >= 17 and not fausses, fausses)
    maj = next(kw for lab, kw, _ in appels("number_input") if lab == "Majoration globale (%)")
    chk("la majoration par défaut est celle de l'écran (5 %)", maj["value"] == EN.DEFAUTS_POUTRE["majoration"])
    chk("les masses volumiques sont celles du module (7 850 et 2 500 kg/m³)",
        T.STEEL_DENSITY_KG_M3 == M.RHO_S == 7850.0 and T.CONCRETE_DENSITY_KG_M3 == M.RHO_C == 2500.0)


class Etat(dict):
    """Un st.session_state minimal : un dictionnaire lisible aussi par attribut."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v


class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran afficherait. Un widget à clé lit et écrit l'état comme Streamlit (une
    valeur d'état hors des bornes est rejetée : le widget reprend sa valeur par défaut) ; sans clé, il rend sa valeur
    par défaut — sauf la majoration, qui reçoit celle du cas."""

    def __init__(self, ss, majoration=5.0):
        self.session_state = ss
        self.maj = majoration
        self.ev = []

    def _w(self, key, defaut):
        if key is None:
            return defaut
        if key not in self.session_state:
            self.session_state[key] = defaut
        return self.session_state[key]

    def radio(self, label, options, index=0, key=None, **_):
        return self._w(key, options[index])

    def selectbox(self, label, options, index=0, key=None, **_):
        v = self._w(key, options[index])
        if v not in options:
            v = options[index]
            if key is not None:
                self.session_state[key] = v
        self.ev.append(("selectbox", label, v))
        return v

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **_):
        if label == "Majoration globale (%)":
            return self.maj
        v = self._w(key, value)
        if key is not None and ((min_value is not None and v < min_value) or (max_value is not None and v > max_value)):
            v = value
            self.session_state[key] = v
        return v

    def text_input(self, label, value="", key=None, **_):
        return self._w(key, value)

    def button(self, *_, **__):
        return False

    def columns(self, spec, **_):
        return [Colonne(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def container(self, **_):
        return Colonne(self)

    def metric(self, label, value, *_, **__):
        self.ev.append(("metric", label, value))

    def table(self, data, **_):
        self.ev.append(("table", data))

    def dataframe(self, df, **_):
        self.ev.append(("dataframe", df.to_dict("records")))

    def pyplot(self, fig, **_):
        self.ev.append(("pyplot", fig))

    def info(self, body, **_):
        self.ev.append(("info", body))

    def _texte(self, *a, **_):
        self.ev.append(("texte", " ".join(str(x) for x in a)))

    markdown = write = subheader = title = text = caption = _texte

    def rerun(self):  # pragma: no cover — aucun bouton n'est cliqué
        raise RuntimeError("rerun")


class Colonne(contextlib.AbstractContextManager):
    """Une colonne (ou un conteneur) de Streamlit : un bloc « with », et aussi un objet qui porte les widgets
    (``m1.metric(…)``) — tout est renvoyé au faux « st »."""

    def __init__(self, st):
        self._st = st

    def __exit__(self, *_):
        return None

    def __getattr__(self, nom):
        return getattr(self._st, nom)


def session(p):
    """L'état du module pour la poutre p (au format du moteur)."""
    ss = Etat()
    ss.update({"L_beam_m": p["L"], "b_web_cm": p["b"], "h_total_cm": p["h"], "cover_cm": p["enrobage"],
               "flange_left_b_cm": p["talon_gauche"]["b"], "flange_left_h_cm": p["talon_gauche"]["h"],
               "flange_right_b_cm": p["talon_droit"]["b"], "flange_right_h_cm": p["talon_droit"]["h"],
               "shape_section": EN.FORMES[p["forme"]], "steel_grade": p["acier"],
               "rebar_families": [{"id": f"F{i + 1}", "role": EN.MODULE_ROLE[f["role"]], "position": EN.MODULE_POSITION[f["position"]],
                                   "n_bars": f["n"], "dia_mm": f["d"], "hook_count": f["retours"], "hook_height_cm": f["h_retour"],
                                   "x_start_pct": f["de"], "x_end_pct": f["a"]} for i, f in enumerate(p["familles"])],
               "stirrup_types": [{"name": t["nom"], "phi_mm": t["d"], "type": "full"} for t in p["types_etriers"]],
               "stirrup_zones": [{"name": z["nom"], "x_start_pct": z["de"], "x_end_pct": z["a"], "spacing_cm": z["s"], "type_name": z["type"]}
                                 for z in p["zones"]],
               "family_counter": len(p["familles"]), "st_type_counter": len(p["types_etriers"]), "st_zone_counter": len(p["zones"]),
               "range_mode": "percent" if p["longueurs"] == "pourcent" else "meter",
               "range_mode_radio": "En % de la portée" if p["longueurs"] == "pourcent" else "En mètres (m)"})
    for i, (m1, m2) in p.get("_metres", {}).items():           # les champs « De (m) », « À (m) » tels que saisis
        ss[f"x1m_{i}"], ss[f"x2m_{i}"] = m1, m2
    return ss


DESSINS = {"n": 0}


def rejouer(p, dessiner=False):
    """L'écran ENTIER du module (show) sur la poutre p : ce qu'il calcule (espions) et ce qu'il affiche."""
    ss = session(p)
    faux = FauxSt(ss, p["majoration"])
    espion = {}
    vrais = {k: getattr(T, k) for k in ("st", "build_geometry", "build_rebar_families", "compute_stirrups_for_zones", "compute_quantities",
                                         "draw_section")}
    vrai_accueil = NAV.bouton_accueil

    def garder(nom):
        def f(*a, **k):
            r = vrais[nom](*a, **k)
            espion.setdefault(nom, []).append(r)
            return r
        return f

    def dessin(section, **k):
        espion["section"] = section
        if dessiner:
            DESSINS["n"] += 1
            fig = vrais["draw_section"](section, **k)
            plt.close(fig)
        return "figure"

    T.st = faux
    for nom in ("build_geometry", "build_rebar_families", "compute_stirrups_for_zones", "compute_quantities"):
        setattr(T, nom, garder(nom))
    T.draw_section = dessin
    NAV.bouton_accueil = lambda *a, **k: None
    erreur = None
    try:
        T.show()
    except ValueError as e:
        erreur = str(e)
    finally:
        for k, v in vrais.items():
            setattr(T, k, v)
        NAV.bouton_accueil = vrai_accueil
    return {"espion": espion, "ev": faux.ev, "ss": ss, "erreur": erreur}


def comparer(r, md):
    """Le moteur (r) contre le module rejoué (md) : la liste des écarts."""
    ecarts = []

    def cmp(nom, a, b):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                ecarts.append(f"{nom} : {len(a)} ≠ {len(b)} éléments")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(f"{nom}[{i}]", x, y)
        elif not egal(a, b):
            ecarts.append(f"{nom} : moteur {a!r} ≠ module {b!r}")
    es = md["espion"]
    geo = es["build_geometry"][-1]                       # la géométrie du calcul (colonne de droite)
    g = r["geometrie"]
    for k, km in (("b", "b_web"), ("h", "h_total"), ("c", "cover"), ("bg", "flange_left_b"), ("hg", "flange_left_h"),
                  ("bd", "flange_right_b"), ("hd", "flange_right_h")):
        cmp(f"géométrie {k}", g[k], getattr(geo, km))
    cmp("aire", g["aire_cm2"], geo.area_cm2())
    cmp("polygone", g["polygone"], [list(x) for x in geo.concrete_polygon()])
    fams = es["build_rebar_families"][-1]
    cmp("nombre de familles", len(r["familles"]), len(fams))
    L = md["ss"]["L_beam_m"]
    codes = T.compute_family_codes(md["ss"]["rebar_families"])
    cols = T.compute_family_colors_for_families(fams)
    for f, mf in zip(r["familles"], fams):
        k = f["code"]
        cmp(f"{k} code", k, codes[mf.id])
        cmp(f"{k} couleur", f["couleur"], cols[mf.id])
        cmp(f"{k} position", EN.MODULE_POSITION[f["position"]], mf.position)
        cmp(f"{k} rôle", EN.MODULE_ROLE[f["role"]], mf.role)
        cmp(f"{k} n_eff", f["n_eff"], mf.effective_n_bars())
        cmp(f"{k} pourcentages", [f["de"], f["a"]], [mf.x_start_pct, mf.x_end_pct])
        cmp(f"{k} h_retour", f["h_retour"], mf.hook_height_cm)
        cmp(f"{k} L_active", f["L_active"], mf.active_length_m(L))
        cmp(f"{k} L_tot", f["L_tot"], mf.total_bar_length_m(L))
        cmp(f"{k} As", f["As"], mf.steel_area_mm2())
        cmp(f"{k} m_lin", f["m_lin"], T.steel_linear_mass_kg_m(mf.dia_mm))
        cmp(f"{k} volume", f["volume"], T.rebar_volume_m3_from_length(mf.total_bar_length_m(L), mf.dia_mm))
        # la colonne « kg tot » du tableau : l'expression de show(), mot pour mot
        cmp(f"{k} kg tot", f["masse_majoree"], mf.total_bar_length_m(L) * T.steel_linear_mass_kg_m(mf.dia_mm) * (1.0 + md["maj"] / 100.0))
    sec = es.get("section")
    barres_module = [[b.x, b.y, b.phi_mm, b.family_id, b.color] for b in sec.rebars] if sec else []
    barres_moteur = [[x, y, f["d"], fams[f["index"]].id, f["couleur"]] for f in r["familles"] for x, y in f["barres"]]
    cmp("barres dessinées (x, y, Ø, famille, couleur)", barres_moteur, barres_module)
    st_res = es["compute_stirrups_for_zones"][-1]
    cmp("zones calculées", len(r["etriers"]), len(st_res))
    for e, sr in zip(r["etriers"], st_res):
        z = e["zone"]
        cmp(f"zone {z} nom/type", [e["zone"], e["type"]], [sr.zone.name, sr.stirrup_type.name])
        cmp(f"zone {z} n", e["n"], sr.n_stirrups)
        cmp(f"zone {z} longueur d'un étrier", e["L_un"], sr.length_per_stirrup_m)
        cmp(f"zone {z} longueur totale", e["L_tot"], sr.total_length_m)
        cmp(f"zone {z} kg tot", e["masse_majoree"], sr.total_length_m * T.steel_linear_mass_kg_m(sr.stirrup_type.phi_mm) * (1.0 + md["maj"] / 100.0))
    qr = es["compute_quantities"][-1]
    q = r["quantites"]
    for k, km in (("Vc", "Vc_m3"), ("masse_beton", "mass_concrete_kg"), ("V_acier", "V_steel_m3"), ("masse_acier", "mass_steel_kg"),
                  ("kg_m3", "kg_per_m3")):
        cmp(f"quantité {k}", q[k], getattr(qr, km))
    cmp("volumes par catégorie", [q["par_categorie"][c] for c in M.CATEGORIES],
        [qr.by_category_m3[c] for c in ("main", "reinforcement", "lateral", "stirrups")])
    # ---- ce que l'écran AFFICHE
    mets = [(lab, val) for t, lab, val in [e for e in md["ev"] if e[0] == "metric"]]
    m_maj, t_maj = q["masse_acier_majoree"], q["kg_m3_majore"]
    cmp("les trois chiffres", mets, [("Volume béton Vc", f"{q['Vc']:.3f} m³"), ("Masse acier", f"{m_maj:.1f} kg"),
                                      ("Taux d’armature", f"{t_maj:.1f} kg/m³")])
    table = next(e[1] for e in md["ev"] if e[0] == "table")
    cmp("la synthèse", [x["Valeur"] for x in table], [f"{m_maj:.1f} kg", f"{q['Vc']:.3f} m³", f"{q['masse_beton']:.0f} kg", f"{t_maj:.1f} kg/m³"])
    df = next((e[1] for e in md["ev"] if e[0] == "dataframe"), [])
    icone = {"bas": "🔴", "haut": "🔵", "cote": "⚫️"}
    lignes = [{"Couleur": icone[f["groupe"]], "Code": f["code"], "Position": f["libelle_position"], "n barres": f["n_eff"], "Ø (mm)": int(f["d"]),
               "L active (m)": f"{f['L_active']:.2f}", "L tot (m)": f"{f['L_tot']:.2f}", "kg/m": f"{f['m_lin']:.3f}", "kg tot": f"{f['masse_majoree']:.2f}"}
              for f in r["familles"]]
    lignes += [{"Couleur": "🟡", "Code": e["type"], "Position": f"Zone {e['zone']}", "n barres": e["n"], "Ø (mm)": int(e["d"]),
                "L active (m)": f"{e['L_un']:.2f}", "L tot (m)": f"{e['L_tot']:.2f}", "kg/m": f"{e['m_lin']:.3f}", "kg tot": f"{e['masse_majoree']:.2f}"}
               for e in r["etriers"]]
    cmp("le tableau récapitulatif (chaque ligne, chaque colonne)", [sorted(x.items()) for x in lignes], [sorted(x.items()) for x in df])
    vide = any(e[0] == "info" and "Ajoute des armatures" in e[1] for e in md["ev"])
    cmp("« Ajoute des armatures… » quand le tableau est vide", vide, not lignes)
    return ecarts


def faits(p, r):
    """Ce que le cas atteint (couverture), lu dans les sorties contrôlées."""
    out = {f"forme {p['forme']}", f"longueurs {p['longueurs']}", f"acier {p['acier']}"}
    for f in r["familles"]:
        out |= {f"position {f['position']}", f"rôle {f['role']}", f"Ø{f['d']}", f"retours {f['retours']}", f"lit {f['lit']}"}
        if f["n"] == 1:
            out.add("une barre")
        if f["L_active"] == 0.0:
            out.add("longueur nulle")
        if 0 < f["de"] or f["a"] < 100:
            out.add("barre partielle")
    out |= {f"zones {len(r['etriers'])}"} | {f"zone ignorée {z['motif']}" for z in r["zones_ignorees"]}
    out |= {f"avertissement {a['code']}" for a in r["avertissements"]}
    if len(p["types_etriers"]) > 1:
        out.add("plusieurs types")
    if p["majoration"] in (0.0, 50.0):
        out.add(f"majoration {p['majoration']:g}")
    return out


if T is not None:
    print("\n=== 2. À la main (m, kg, m³) ===")
    r1 = M.calculer(sans(cas("le cas des constats")))
    q1 = r1["quantites"]
    rho = 7850.0
    mlin16, mlin8 = math.pi * 0.008 ** 2 * rho, math.pi * 0.004 ** 2 * rho
    L_et = (2 * ((300 - 60) + (600 - 60)) + 2 * 8 * 8) / 1000           # 1,688 m : contour de l'âme à l'enrobage + 2 × 8 Ø
    n_et = math.floor(6 * 0.33 / 0.15) + 1                                 # 1,98 m / 0,15 = 13,2 → 14 étriers
    m_tot = (2 * 6 * mlin16 + n_et * L_et * mlin8) * 1.05
    chk(f"H1 : Vc = 0,30 × 0,60 × 6 = 1,080 m³ ; 2Ø16 sur 6 m = {2 * 6 * mlin16:.2f} kg ; {n_et} étriers Ø8 de {L_et:.3f} m",
        abs(q1["Vc"] - 1.08) < 1e-12 and r1["etriers"][0]["n"] == 14 and abs(r1["etriers"][0]["L_un"] - L_et) < 1e-12)
    chk(f"H1 : masse majorée de 5 % = {m_tot:.2f} kg, taux = {m_tot / 1.08:.2f} kg/m³ (l'écran : 29,7 kg et 27,5 kg/m³)",
        abs(q1["masse_acier_majoree"] - m_tot) < 1e-9 and abs(q1["kg_m3_majore"] - m_tot / 1.08) < 1e-9
        and f"{q1['masse_acier_majoree']:.1f}" == "29.7" and f"{q1['kg_m3_majore']:.1f}" == "27.5")
    # H2 : talon gauche 25 × 20 (aire 30 × 60 + 25 × 20 = 2 300 cm²), 4Ø25 + 3Ø16 + 2Ø12 sur 6 m, sans majoration
    r2 = M.calculer(dict(sans(cas("talon gauche 25")), majoration=0.0))
    q2 = r2["quantites"]
    m2 = sum(n * 6 * math.pi * (d / 2000) ** 2 * rho for n, d in ((4, 25), (3, 16), (2, 12)))
    chk(f"H2 : aire 30 × 60 + 25 × 20 = 2 300 cm², Vc = 1,380 m³, acier {m2:.2f} kg, taux {m2 / 1.38:.2f} kg/m³",
        q2["aire_cm2"] == 2300.0 and abs(q2["Vc"] - 1.38) < 1e-12 and abs(q2["masse_acier"] - m2) < 1e-9 and abs(q2["kg_m3_majore"] - m2 / 1.38) < 1e-9)
    f2 = r2["familles"][1]
    chk("H2 : le renfort de talon inférieur est à c + Ø/2 = 3,8 cm du bas, de x = −15 − 25 + 3 = −37 à −15 − 3 = −18 cm",
        f2["barres"] == [[-37.0, 3.8], [-27.5, 3.8], [-18.0, 3.8]], f2["barres"])
    r3 = M.calculer(sans(cas("latérales : une barre")))
    chk("H3 : latérales — une barre à mi-hauteur (30 cm) ; trois barres à 1/4, 2/4, 3/4 de c à h − c ; « 2 côtés » : 2 × 4",
        r3["familles"][0]["barres"] == [[-12.0, 30.0]] and [b[1] for b in r3["familles"][1]["barres"]] == [16.5, 30.0, 43.5]
        and r3["familles"][2]["n_eff"] == 8 and len(r3["familles"][2]["barres"]) == 8, [f["barres"] for f in r3["familles"]])

    print("\n=== 3. Parité avec modules/taux_armature.py : l'écran ENTIER rejoué, cas par cas ===")
    COUV, REJEUX = set(), []
    for k, (desc, p) in enumerate(CAS):
        r = M.calculer(sans(p))
        md = rejouer(p, dessiner=k < 4)
        md["maj"] = p["majoration"]
        REJEUX.append(md)
        ecarts = comparer(r, md) if md["erreur"] is None else [f"le module s'arrête : {md['erreur']}"]
        COUV |= faits(p, r)
        n_b = sum(len(f["barres"]) for f in r["familles"])
        chk(f"cas {k + 1:02d} · {desc} — {len(r['familles'])} famille(s), {n_b} barres, {len(r['etriers'])} zone(s)", not ecarts, ecarts[:5])
    chk(f"{len(CAS)} poutres comparées : au moins 50", len(CAS) >= 50)
    chk(f"le dessin de l'écran (draw_section, matplotlib) a tourné sur {DESSINS['n']} cas", DESSINS["n"] == 4)
    print(f"     {STATS['n']} comparaisons numériques, {STATS['bit']} identiques AU BIT PRÈS")
    chk("toutes les comparaisons numériques sont identiques au bit près (c'est le même calcul)", STATS["n"] == STATS["bit"] and STATS["n"] > 5000,
        STATS)

    print("\n=== 4. Couverture : ce que les cas atteignent réellement ===")
    ATTENDUS = ([f"forme {f}" for f in EN.FORMES] + [f"position {x}" for x in EN.POSITIONS] + [f"rôle {x}" for x in EN.ROLES]
                + [f"Ø{d}" for d in EN.DIAMETRES] + [f"retours {r}" for r in EN.RETOURS] + ["longueurs pourcent", "longueurs metre"]
                + ["lit 0", "lit 1", "lit 2", "une barre", "longueur nulle", "barre partielle", "zones 0", "zones 1", "zones 2", "zones 3",
                   "zone ignorée sans_type", "zone ignorée longueur_nulle", "plusieurs types", "majoration 0", "majoration 50",
                   "acier B500A", "acier B500C"]
                + [f"avertissement {c}" for c in ("talons_ignores", "sans_armature", "longueur_nulle", "troncature_etriers",
                                                   "zones_chevauchement", "zones_trou", "zone_sans_type", "zone_nulle", "etriers_talon")])
    manque = [a for a in ATTENDUS if a not in COUV]
    chk(f"les cas atteignent les {len(ATTENDUS)} situations attendues (formes, positions, Ø, retours, zones, avertissements…)", not manque, manque)
    tronc = M.calculer(sans(cas("troncature")))
    chk("la troncature est REPRODUITE : 3 étriers (l'écran aussi), et dite", tronc["etriers"][0]["n"] == 3
        and [a["code"] for a in tronc["avertissements"]].count("troncature_etriers") == 1)

    print("\n=== 5. Effets de bord de la saisie (beton/taux/saisie.py) contre l'écran ===")
    # forme : une famille dont la position n'est plus offerte passe à la première du menu (edit_family)
    ecarts = []
    for de, vers in (("talon_gauche", "rectangulaire"), ("talon_gauche", "talon_droit"), ("talon_droit", "talon_gauche"),
                     ("deux_talons", "rectangulaire"), ("talon_gauche", "deux_talons"), ("rectangulaire", "talon_droit")):
        opts = EN.options_positions(de, "renfort")
        p = poutre("F", [fam("renfort", pos) for pos in opts] + [fam("lateral", "lat_2")], forme=de)
        attendu = copy.deepcopy(sans(p))
        SA.changer_forme(attendu, vers)
        p2 = dict(sans(p), forme=vers)                     # l'écran : la forme change, edit_family corrige les positions
        md = rejouer(p2)
        lues = [EN.POSITION_MODULE[f["position"]] for f in md["ss"]["rebar_families"]]
        if lues != [f["position"] for f in attendu["familles"]]:
            ecarts.append((de, vers, lues, [f["position"] for f in attendu["familles"]]))
    chk("forme changée : les positions retirées passent à « Armatures inférieures (âme) », comme edit_family (6 changements)", not ecarts, ecarts)
    p = poutre("R", [fam("principal", r=1, hr=30.0)])
    p2 = copy.deepcopy(sans(p))
    p2["familles"][0]["retours"] = 0
    md = rejouer(dict(p2, _metres={}))
    q = copy.deepcopy(sans(p))
    SA.changer_retours(q["familles"][0], 0)
    chk("« Retours » = 0 remet la hauteur de retour à 0, comme edit_family", md["ss"]["rebar_families"][0]["hook_height_cm"] == 0.0
        == q["familles"][0]["h_retour"])
    # un type d'étrier supprimé : ses zones prennent le premier type restant (edit_stirrup_zone)
    p = poutre("T", [fam("principal")], (("E1", 8.0), ("E2", 10.0), ("E3", 12.0)), [("Z1", 0, 50, 15, "E2"), ("Z2", 50, 100, 15, "E3")])
    for i in (1, 0, 2):
        q = copy.deepcopy(sans(p))
        SA.supprimer_type(q, i)
        ss = session(p)
        ss["stirrup_types"].pop(i)
        faux = FauxSt(ss)
        vrai = T.st
        T.st = faux
        try:
            for j, d in enumerate(ss["stirrup_zones"]):
                T.edit_stirrup_zone(j, d)
        finally:
            T.st = vrai
        chk(f"type E{i + 1} supprimé : les zones prennent le premier type restant, comme edit_stirrup_zone",
            [z["type"] for z in q["zones"]] == [d["type_name"] for d in ss["stirrup_zones"]], ([z["type"] for z in q["zones"]], ss["stirrup_zones"]))
    q = poutre("T1", [], (("E1", 8.0),), [("Z1", 0, 50, 15, "E1")])
    SA.renommer_type(q, 0, "Cadre")
    ss = session(q)
    ss["stirrup_zones"][0]["type_name"] = "E1"               # l'écran : le type renommé, la zone garde l'ancien nom…
    vrai, T.st = T.st, FauxSt(ss)
    try:
        T.edit_stirrup_zone(0, ss["stirrup_zones"][0])        # … et edit_stirrup_zone la remet sur le premier type
    finally:
        T.st = vrai
    chk("un type (seul) renommé : sa zone le suit — c'est ce que fait l'écran avec un seul type", q["zones"][0]["type"] == ss["stirrup_zones"][0]["type_name"] == "Cadre")

    print("\n=== 6. Constats sur l'écran Streamlit (AppTest) : chaque défaut de la doc, montré ===")
    from streamlit.testing.v1 import AppTest
    APP = ("import sys\nsys.path.insert(0, %r)\nfrom modules import taux_armature\ntaux_armature.show()\n" % RACINE)

    def ecran():
        at = AppTest.from_string(APP, default_timeout=60)
        at.run()
        return at

    def bouton(at, lab):
        return next(b for b in at.button if b.label == lab)

    def metriques(at):
        return [m.value for m in at.metric]

    def selon(at, lab):
        return next(s for s in at.selectbox if s.label == lab)

    at = ecran()
    chk("défauts de l'écran = nouvelle_poutre() (30 × 60, L 6 m, c 3 cm, B500A, majoration 5 %, aucune armature) : 1,080 m³, 0 kg",
        [(n.label, n.value) for n in at.number_input] == [("Enrobage c (cm)", 3.0), ("Portée L (m)", 6.0), ("Largeur b (cm)", 30.0),
                                                           ("Hauteur h (cm)", 60.0), ("Majoration globale (%)", 5.0)]
        and metriques(at) == ["1.080 m³", "0.0 kg", "0.0 kg/m³"] and selon(at, "Qualité d’acier").value == "B500A"
        and at.session_state["rebar_families"] == [] and at.session_state["stirrup_zones"] == [])
    lib = [x for x in at.table[0].value["Élément"]][2]
    chk("D1 : la synthèse affiche « Poids béton (≈{CONCRETE_DENSITY_KG_M3:.0f} kg/m³) » (accolades, chaîne non formatée)",
        lib == "Poids béton (≈{CONCRETE_DENSITY_KG_M3:.0f} kg/m³)", lib)
    selon(at, "Section").set_value("Rectangulaire + talon gauche").run()
    talons = [(n.label, n.value) for n in at.number_input if n.label.startswith("Talon")]
    chk("D4 : talon gauche — les champs du talon affichent 0, le calcul prend 10 × 15 cm (1,080 + 0,090 = 1,170 m³)",
        talons == [("Talon gauche – largeur (cm)", 0.0), ("Talon gauche – hauteur (cm)", 0.0)] and metriques(at)[0] == "1.170 m³"
        and M.calculer(dict(EN.nouvelle_poutre(), forme="talon_gauche"))["quantites"]["Vc"] == 1.17, (talons, metriques(at)))
    selon(at, "Section").set_value("Rectangulaire + deux talons").run()
    premier = selon(at, "Section").value
    selon(at, "Section").set_value("Rectangulaire + deux talons").run()
    chk("D7 : le menu « Section » perd le premier choix qui suit un changement (il faut choisir deux fois)",
        premier == "Rectangulaire + talon gauche" and selon(at, "Section").value == "Rectangulaire + deux talons", premier)
    n_talons = len([n for n in at.number_input if n.label.startswith("Talon")])
    chk("D2 : deux talons — quatre champs de talon affichés, mais le volume est celui de l'âme seule (1,080 m³) ; le moteur aussi",
        n_talons == 4 and metriques(at)[0] == "1.080 m³" and M.calculer(dict(EN.nouvelle_poutre(), forme="deux_talons"))["quantites"]["Vc"] == 1.08,
        (n_talons, metriques(at)))
    bouton(at, "+ Renfort talon").click().run()
    err = [e.value for e in at.exception]
    try:
        M.calculer({**EN.nouvelle_poutre(), "forme": "deux_talons", "familles": [EN.nouvelle_famille({}, "renfort", "talon_g_inf")]})
        moteur_err = None
    except ValueError as e:
        moteur_err = str(e)
    chk("D3 : deux talons + « + Renfort talon » : l'écran s'arrête sur « Talon gauche non défini. » ; le moteur lève la même erreur",
        err == ["Talon gauche non défini."] and moteur_err == "Talon gauche non défini.", (err, moteur_err))

    at = ecran()
    bouton(at, "+ Lit principal").click().run()
    f0 = at.session_state["rebar_families"][0]
    p = EN.nouvelle_poutre()
    f_moteur = SA.ajouter_famille(p, "principal")
    chk("« + Lit principal » : 2Ø16, âme en bas, 0 à 100 %, sans retour — la famille de saisie.ajouter_famille",
        {k: f0[k] for k in ("n_bars", "dia_mm", "hook_count", "x_start_pct", "x_end_pct")} == {"n_bars": 2, "dia_mm": 16, "hook_count": 0, "x_start_pct": 0.0, "x_end_pct": 100.0}
        and (EN.ROLE_MODULE[f0["role"]], EN.POSITION_MODULE[f0["position"]], f0["hook_height_cm"]) == (f_moteur["role"], f_moteur["position"], f_moteur["h_retour"])
        and (f0["n_bars"], f0["dia_mm"], f0["x_start_pct"], f0["x_end_pct"]) == (f_moteur["n"], f_moteur["d"], f_moteur["de"], f_moteur["a"]), f0)
    next(s for s in at.selectbox if s.key == "hook_count_0").set_value(1).run()
    hr = next(n for n in at.number_input if n.key == "h_hook_0").value
    chk("D5 : « Retours » = 1 → hauteur de retour 0 (la valeur h − 2c proposée par add_family est perdue)", hr == 0.0, hr)
    for lab, sorte in (("+ Renfort local", "renfort"), ("+ Latéral", "lateral")):
        bouton(at, lab).click().run()
        f = at.session_state["rebar_families"][-1]
        fm = SA.ajouter_famille(p, sorte)
        chk(f"« {lab} » : la famille de saisie.ajouter_famille ({fm['position']})",
            (EN.ROLE_MODULE[f["role"]], EN.POSITION_MODULE[f["position"]], f["n_bars"], f["dia_mm"], f["hook_height_cm"])
            == (fm["role"], fm["position"], fm["n"], fm["d"], fm["h_retour"]), f)
    bouton(at, "+ Étrier").click().run()
    bouton(at, "+ Étrier").click().run()
    SA.ajouter_etrier(p)
    SA.ajouter_etrier(p)
    chk("« + Étrier » deux fois : un type E1 Ø8, deux zones Z1 et Z2 de 0 à 33 % à 15 cm — saisie.ajouter_etrier",
        [(t["name"], t["phi_mm"]) for t in at.session_state["stirrup_types"]] == [(t["nom"], t["d"]) for t in p["types_etriers"]]
        and [(z["name"], z["x_start_pct"], z["x_end_pct"], z["spacing_cm"], z["type_name"]) for z in at.session_state["stirrup_zones"]]
        == [(z["nom"], z["de"], z["a"], z["s"], z["type"]) for z in p["zones"]])
    # D6 : supprimer une famille décale les clés des widgets : la suivante hérite des valeurs de la supprimée
    at = ecran()
    bouton(at, "+ Lit principal").click().run()
    bouton(at, "+ Renfort local").click().run()
    next(n for n in at.number_input if n.key == "nb_1").set_value(5).run()
    next(s for s in at.selectbox if s.key == "phi_1").set_value(25).run()
    next(b for b in at.button if b.key == "del_fam_0").click().run()
    reste = at.session_state["rebar_families"]
    chk("D6 : supprimer la 1re famille (2Ø16 en bas) change la 2e (5Ø25 en haut) en 2Ø16 en bas : clés de widgets par rang",
        [(f["id"], f["position"], f["n_bars"], f["dia_mm"]) for f in reste] == [("F2", "web_bottom", 2, 16)], reste)
    # D8 : en mètres, une portée changée garde les mètres (s'ils tiennent dans la nouvelle portée), sinon le pourcentage
    at = ecran()
    bouton(at, "+ Lit principal").click().run()
    at.radio[0].set_value("En mètres (m)").run()
    p = EN.normaliser(dict(EN.nouvelle_poutre(), longueurs="metre", familles=[EN.nouvelle_famille({}, "principal")]))
    suite = []
    for L in (8.0, 4.0, 10.0):
        next(n for n in at.number_input if n.key == "L_beam_m").set_value(L).run()
        SA.changer_portee(p, L)
        f = at.session_state["rebar_families"][0]
        suite.append(((f["x_start_pct"], f["x_end_pct"]), (p["familles"][0]["de"], p["familles"][0]["a"])))
    chk("D8 : en mètres, L 6 → 8 m garde 6 m (75 %) ; 8 → 4 m : 6 m dépasse, l'écran reprend 75 % (3 m) ; 4 → 10 m garde 3 m (30 %)",
        [x for x, _ in suite] == [(0.0, 75.0), (0.0, 75.0), (0.0, 30.0)] and all(a == b for a, b in suite), suite)

    print("\n=== 7. Mutations : chaque altération du moteur rougit la parité ===")

    def rougit(nom, poser, enlever):
        poser()
        try:
            n_cas = 0
            for (desc, p), md in zip(CAS, REJEUX):
                try:
                    r = M.calculer(sans(p))
                except Exception:  # noqa: BLE001 — une mutation qui fait planter compte comme détectée
                    n_cas += 1
                    continue
                if md["erreur"] is None and comparer(r, md):
                    n_cas += 1
        finally:
            enlever()
        chk(f"mutation « {nom} » : la parité rougit ({n_cas} cas)", n_cas > 0)

    def remplacer(objet, nom, valeur):
        vrai = getattr(objet, nom)
        return (lambda: setattr(objet, nom, valeur)), (lambda: setattr(objet, nom, vrai))

    _vrai = {k: getattr(M, k) for k in ("n_effectif", "longueur_totale", "longueur_etrier", "_zones", "couleurs", "codes", "lits", "longueur_active")}
    _talons, _centres = GE.talons, GE.centres

    def zones_arrondies(p, g):
        out, ign = _vrai["_zones"](p, g)
        for z in out:
            z["n"] = round(z["L_zone"] / z["s_m"]) + 1
            z["L_tot"] = z["n"] * z["L_un"]
        return out, ign

    def deux_talons_comptes(p):
        bg, hg, bd, hd = _talons(p)
        if p["forme"] == "deux_talons":
            return p["talon_gauche"]["b"], p["talon_gauche"]["h"], p["talon_droit"]["b"], p["talon_droit"]["h"]
        return bg, hg, bd, hd

    def pas_25(g, pos, n, phi, lit=0):
        pts = _centres(g, pos, n, phi, lit)
        return [(x, y + (0.5 * lit if pos in GE.GROUPES_BAS else -0.5 * lit)) for x, y in pts]

    MUTATIONS = [
        ("acier à 7 800 kg/m³", *remplacer(M, "RHO_S", 7800.0)),
        ("béton à 2 400 kg/m³", *remplacer(M, "RHO_C", 2400.0)),
        ("« Latérales 2 côtés » comptées une fois", *remplacer(M, "n_effectif", lambda f: f["n"])),
        ("retours oubliés", *remplacer(M, "longueur_totale", lambda f, L: M.longueur_active(f, L) * M.n_effectif(f))),
        ("retours comptés une fois par famille", *remplacer(M, "longueur_totale", lambda f, L: _vrai["longueur_totale"](f, L)
                                                            - (f["retours"] * f["h_retour"] / 100.0) * (M.n_effectif(f) - 1) if M.longueur_active(f, L) > 0 else 0.0)),
        ("crochets d'étrier de 10 Ø", *remplacer(M, "longueur_etrier", lambda d, g: _vrai["longueur_etrier"](d, g) + 4 * d / 1000.0)),
        ("étrier au nu de l'enrobage + Ø", *remplacer(M, "longueur_etrier", lambda d, g: _vrai["longueur_etrier"](d, g) - 8 * d / 1000.0)),
        ("nombre d'étriers arrondi au lieu de tronqué", *remplacer(M, "_zones", zones_arrondies)),
        ("deux talons comptés (le défaut corrigé en silence)", *remplacer(GE, "talons", deux_talons_comptes)),
        ("lits espacés de Ø + 2,5 cm", *remplacer(GE, "centres", pas_25)),
        ("couleurs : dégradé inversé", *remplacer(M, "couleurs", lambda fs: list(reversed(_vrai["couleurs"](fs))))),
        ("codes numérotés sans groupes", *remplacer(M, "codes", lambda fs: [f"INF{i + 1}" for i in range(len(fs))])),
        ("rang de lit par groupe au lieu de par position", *remplacer(M, "lits", M.rangs)),
        ("longueur active sans borne À ≤ De", *remplacer(M, "longueur_active", lambda f, L: L * (f["a"] - f["de"]) / 100.0)),
    ]
    for nom, poser, enlever in MUTATIONS:
        rougit(nom, poser, enlever)

# ------------------------------------------------------------------ 8. format
print("\n=== 8. Format : JSON, normalisation, domaine de l'écran, formules ===")
tous = [M.calculer(sans(p)) for _, p in CAS]
chk("la sortie de calculer() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(r, allow_nan=False)) == r for r in tous))
chk("normaliser() est idempotente et {} donne nouvelle_poutre()",
    all(M.normaliser(M.normaliser(sans(p))) == M.normaliser(sans(p)) for _, p in CAS) and M.normaliser({}) == M.nouvelle_poutre())
entree = copy.deepcopy(sans(cas("poutre complète")))
M.calculer(entree)
M.candidats_diametre(entree, 0)
chk("calculer() et candidats_diametre() ne modifient pas la poutre reçue", entree == sans(cas("poutre complète")))
cands = M.candidats_diametre(sans(cas("poutre complète")), 0)
ref = M.calculer(sans(cas("poutre complète")))
chk("candidats_diametre : les dix Ø de l'écran, triés, le retenu marqué et identique au calcul de la poutre",
    [c["d"] for c in cands] == list(EN.DIAMETRES) and [c["d"] for c in cands if c["retenu"]] == [20]
    and next(c for c in cands if c["retenu"])["kg_m3_majore"] == ref["quantites"]["kg_m3_majore"])
REFUS = (("largeur 4 cm", {"b": 4.0}), ("hauteur 301 cm", {"h": 301.0}), ("enrobage 0,5", {"enrobage": 0.5}), ("portée 0,4 m", {"L": 0.4}),
         ("portée 61 m", {"L": 61.0}), ("2c ≥ b", {"b": 6.0, "enrobage": 3.0}), ("2c ≥ h", {"h": 10.0, "enrobage": 5.0}),
         ("majoration 51 %", {"majoration": 51.0}), ("majoration négative", {"majoration": -1.0}), ("forme inconnue", {"forme": "en T"}),
         ("talon nul (l'écran lirait 10 cm)", {"talon_gauche": {"b": 0.0, "h": 15.0}}), ("talon de 301 cm", {"talon_droit": {"b": 10.0, "h": 301.0}}),
         ("largeur en texte", {"b": "30"}), ("hauteur booléenne", {"h": True}), ("portée infinie", {"L": float("inf")}), ("clé inconnue", {"largeur": 30}),
         ("acier B400", {"acier": "B400"}), ("mode « cm »", {"longueurs": "cm"}),
         ("Ø18 (non proposé)", {"familles": [fam("principal", d=18)]}), ("zéro barre", {"familles": [fam("principal", n=0)]}),
         ("21 barres", {"familles": [fam("principal", n=21)]}), ("2,5 barres", {"familles": [fam("principal", n=2.5)]}),
         ("trois retours", {"familles": [fam("principal", r=3)]}), ("retour de 1 001 cm", {"familles": [fam("principal", r=1, hr=1001.0)]}),
         ("De à −5 %", {"familles": [fam("principal", de=-5.0)]}), ("À à 101 %", {"familles": [fam("principal", a=101.0)]}),
         ("talon sur une section rectangulaire", {"familles": [fam("renfort", "talon_g_inf")]}),
         ("latérale en bas d'âme", {"familles": [fam("lateral", "ame_inf")]}), ("lit en latérale", {"familles": [fam("principal", "lat_g")]}),
         ("rôle inconnu", {"familles": [fam("principal") | {"role": "chapeau"}]}), ("clé de famille inconnue", {"familles": [fam("principal") | {"x": 1}]}),
         ("étrier Ø5", {"types_etriers": [{"nom": "E1", "d": 5.0}]}), ("étrier Ø18", {"types_etriers": [{"nom": "E1", "d": 18.0}]}),
         ("deux types du même nom", {"types_etriers": [{"nom": "E1", "d": 8.0}, {"nom": "E1", "d": 10.0}]}),
         ("type sans nom", {"types_etriers": [{"nom": " ", "d": 8.0}]}),
         ("espacement 1 cm", {"types_etriers": [{"nom": "E1", "d": 8.0}], "zones": [{"nom": "Z1", "de": 0, "a": 50, "s": 1.0, "type": "E1"}]}),
         ("zone d'un type inconnu", {"types_etriers": [{"nom": "E1", "d": 8.0}], "zones": [{"nom": "Z1", "de": 0, "a": 50, "s": 15, "type": "E9"}]}),
         ("familles en objet", {"familles": {"a": 1}}))
refusees = []
for nom, champ in REFUS:
    try:
        M.calculer(dict(M.nouvelle_poutre(), **champ))
        refusees.append(f"{nom} : accepté")
    except ValueError:
        pass
    except Exception as e:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
        refusees.append(f"{nom} : {type(e).__name__} {e}")
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError (jamais de substitution silencieuse)", not refusees, refusees)
chk("retours = 0 : la hauteur de retour est remise à 0 par normaliser (edit_family)",
    M.normaliser(dict(M.nouvelle_poutre(), familles=[fam("principal", r=0, hr=25.0)]))["familles"][0]["h_retour"] == 0.0)
FORMULES = {
    "L_active": lambda t: t["L"] * (t["a"] - t["de"]) / 100.0 if t["a"] > t["de"] else 0.0,
    "n_eff": lambda t: t["n"] * t["k"],
    "L_tot": lambda t: (t["L_active"] * t["n_eff"] + t["r"] * t["h_r"] / 100.0 * t["n_eff"]) if t["L_active"] > 0 else 0.0,
    "m_lin": lambda t: math.pi * (t["d"] / 2000.0) ** 2 * t["rho"],
    "masse": lambda t: t["L_tot"] * t["m_lin"] * (1 + t["maj"] / 100.0),
    "L_zone": lambda t: t["L"] * (t["a"] - t["de"]) / 100.0,
    "n": lambda t: math.floor(t["L_zone"] / (t["s"] / 100.0)) + 1,
    "L_un": lambda t: (2 * ((t["b"] - 2 * t["c"]) * 10 + (t["h"] - 2 * t["c"]) * 10) + 2 * 8 * t["d"]) / 1000.0,
    "L_tot_etriers": lambda t: t["n"] * t["L_un"],
    "A_c": lambda t: t["b"] * t["h"] + t["b_g"] * t["h_g"] + t["b_d"] * t["h_d"],
    "V_c": lambda t: t["A_c"] / 1e4 * t["L"],
    "M_c": lambda t: t["V_c"] * t["rho"],
    "V_s": lambda t: t["V_long"] + t["V_et"],
    "M_s": lambda t: t["V_s"] * t["rho"],
    "M_s_maj": lambda t: t["M_s"] * (1 + t["maj"] / 100.0),
    "taux": lambda t: t["M_s_maj"] / t["V_c"],
}
fautes, n_f = [], 0
for (desc, p), r in zip(CAS, tous):
    blocs = [f["formules"] for f in r["familles"]] + [e["formules"] for e in r["etriers"]] + [r["quantites"]["formules"]]
    for bloc in blocs:
        for f in bloc:
            n_f += 1
            t = {x["symbole"]: x["valeur"] for x in f["termes"]}
            v = FORMULES[f["cle"]](t)
            if not (v == f["resultat"] or abs(v - f["resultat"]) <= 1e-12 * max(1.0, abs(v))):
                fautes.append(f"{p['nom']}/{f['cle']} : {v!r} ≠ {f['resultat']!r}")
chk(f"« formule = valeurs = résultat » : les {n_f} formules se recalculent à partir de leurs seuls termes", not fautes and n_f > 1000, fautes[:4])
doc = lire("docs", "beton", "TAUX_ARMATURE_MOTEUR.md") if os.path.exists(os.path.join(RACINE, "docs", "beton", "TAUX_ARMATURE_MOTEUR.md")) else ""
cles_q = sorted(k for k in tous[2]["quantites"] if k != "formules")
cles_f = sorted(k for k in tous[2]["familles"][0] if k not in ("formules",))
sans_doc = [k for k in cles_q + cles_f if f"`{k}`" not in doc or f"``{k}``" not in M.__doc__]
chk(f"les {len(cles_q) + len(cles_f)} grandeurs de sortie sont nommées dans la docstring du moteur ET dans docs/beton/TAUX_ARMATURE_MOTEUR.md",
    not sans_doc, sans_doc)

# ------------------------------------------------------------------ 9. pureté
print("\n=== 9. Pureté : le moteur se charge et calcule SANS streamlit, avec la seule bibliothèque standard ===")
non_std = []
for f in sorted(os.listdir(os.path.join(RACINE, "beton", "taux"))):
    if f.endswith(".py"):
        for n in ast.walk(ast.parse(lire("beton", "taux", f))):
            noms = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
            non_std += [f"{f}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"beton"}]
chk("aucun fichier de beton/taux n'importe autre chose que la bibliothèque standard (lecture des sources)", not non_std, non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "numpy", "matplotlib", "scipy", "pymupdf", "fitz", "PIL")
class Refus:
    """Refuse tout chargement des bibliothèques interdites."""
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
try:
    import numpy  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from beton.taux import moteur as M
poutres = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.calculer(p) for p in poutres]}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"beton"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
poutres = [sans(p) for _, p in CAS]
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(poutres), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans bibliothèque tierce", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : numpy, streamlit, pandas… REFUSÉS par sys.meta_path, et le moteur calcule quand même ({len(out['res'])} poutres)",
        out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    chk(f"Python {v} : résultats identiques à ceux de ce processus, au bit près", out["res"] == json.loads(json.dumps(tous)))
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

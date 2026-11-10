# -*- coding: utf-8 -*-
"""Moteur « Enrobage » (beton/enrobage) : la PARITÉ avec l'écran Streamlit modules/enrobage.py (VERSION 2.0), qui reste
la référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

TROIS ÉCARTS VOULUS avec l'écran Streamlit depuis le 10/11/2026 (lot Q, docs/site/LOT_Q_QUATRE_MODULES.md § 4 : le site
ne reproduit plus une valeur fausse). Ils sont NOMMÉS (``ECARTS_VOULUS``), la parité les EXCLUT — et seulement eux —, et
chaque élément qu'ils touchent est recalculé ici d'après la norme (``controle_voulu``) :
  D1  poteaux au feu : les distances a sont lues dans le tableau 5.2a de l'EN 1992-1-2:2004, colonne μ_fi = 0,7
      (beton/bibliotheque/feu.py) — l'écran porte la colonne μ_fi = 0,2 sous le libellé « μ_fi = 0,7 » ; R240 : la
      case du tableau est vide, « non couvert par le tableau », plus de valeur inventée ;
  D2  voiles R180 et R240 : 50 et 60 mm (tableau 5.4, μ_fi = 0,7, une face) — l'écran écrit 45 et 55 ;
  D6  dalle ou voile : le Ø d'étrier n'est plus retiré de la distance à l'axe.
Tout le reste de la parité est gardé (durabilité, poutres, dalles sans étrier, voiles jusqu'à R120 sans étrier, feu
non vérifié, effets de bord des widgets).

  1. Données : chaque tableau et chaque liste du moteur est RELU dans le source du module (arbre syntaxique, sans
     l'importer) ; les trois fonctions de calcul (a_feu, c_durabilite, c_feu) sont les siennes, mot pour mot (arbres
     syntaxiques comparés).
  2. Fonctions : a_feu, c_durabilite et c_feu du moteur et du module donnent les MÊMES valeurs, de même type, au bit
     près, sur toute la grille utile (classes, durées, variantes, Ø et Ø d'étrier de 0 à 60 mm, pairs et impairs).
  3. Écran (faux « st ») : show() est exécuté avec un faux st qui ENREGISTRE ce que l'écran afficherait et répond aux
     widgets ; des espions relèvent les flottants exacts passés et rendus par les trois fonctions. Plus de 10 000
     éléments : les widgets (libellés, options, index par défaut, aides, bornes), chaque encadré (titre, valeur,
     état), chaque ligne de markdown, chaque caption — égaux à ce que le moteur calcule, écrit au format de l'écran.
  4. Couverture : chaque type, durée, variante, terme gouvernant c_min, feu couvert (= 0 et < 0), écart feu /
     durabilité de 2 mm pile (encadré) et de 2,5 mm (pas d'encadré), égalité c_nom,dur = c_nom,feu (la durabilité
     gouverne), demi-millimètres.
  5. AppTest (le VRAI écran Streamlit) : défauts, effets de bord des widgets sans clé (classe structurale, largeur au
     feu) rejoués pas à pas et comparés à ``appliquer`` ; résultats affichés comparés au moteur.
  6. À la main : quelques éléments recalculés ici avec l'EN 1992-1-1 §4.4.1 et les tableaux de l'écran.
  7. Mutations : on abîme le moteur (une valeur de tableau, la borne à 0, ≥ devenu >, ≤ 2 devenu < 2, Δc_dev oublié,
     effet de bord retiré ou forcé) : la parité DOIT rougir ; et chaque écart voulu REVENU (la table de l'écran pour les
     poteaux, 45 mm pour un voile R180, l'étrier retiré d'une dalle, une valeur inventée à R240) fait rougir son contrôle.
  8. Format : normalisation (défauts, ValueError hors du domaine de l'écran), JSON, saisie.
  9. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, numpy, pandas…
     sous Python 3.12 et 3.13 (celui de Pyodide) s'il est installé.

    /usr/bin/python3.12 tests/test_enrobage_moteur.py       (streamlit requis pour 2, 3, 5 et 7)
"""
import ast
import contextlib
import copy
import inspect
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from beton.bibliotheque import feu as FEU  # noqa: E402
from beton.enrobage import donnees as D  # noqa: E402
from beton.enrobage import entree as EN  # noqa: E402
from beton.enrobage import moteur as M  # noqa: E402

OK, KO = [], []
SOURCE_MODULE = os.path.join(RACINE, "modules", "enrobage.py")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:700]}]" if info and not cond else ""))


def meme(a, b):
    """Égalité stricte : même valeur ET même type (int ≠ float), récursivement — « au bit près »."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(meme(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


# =================================================================== 1. les données, relues dans le source
print("\n=== 1. Données : tableaux, listes et fonctions relus dans modules/enrobage.py ===")
with open(SOURCE_MODULE, encoding="utf-8") as f:
    ARBRE = ast.parse(f.read())
LUS = {}
for n in ARBRE.body:
    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
        try:
            LUS[n.targets[0].id] = ast.literal_eval(n.value)
        except ValueError:
            pass
for nom in ("EXPOSITION_INFO", "CLASSE_STRUCTURALE_DEFAULT", "C_MIN_DUR", "R_LEVELS", "FEU_DALLE", "FEU_POUTRE"):
    chk(f"{nom} du moteur = celui du module (valeurs, ordre des clés)",
        nom in LUS and LUS[nom] == getattr(D, nom) and list(LUS[nom]) == list(getattr(D, nom)), (LUS.get(nom), getattr(D, nom)))

# --------------------------------------------------- les TROIS ÉCARTS VOULUS avec l'écran Streamlit depuis le 10/11/2026
ECARTS_VOULUS = {
    "D1": "poteaux : distances au feu du tableau 5.2a de l'EN 1992-1-2 (μ_fi = 0,7), plus celles de l'écran (colonne μ_fi = 0,2) ; R240 non couvert",
    "D2": "voiles R180 et R240 : 50 et 60 mm (tableau 5.4, μ_fi = 0,7, une face), l'écran écrit 45 et 55",
    "D6": "dalle ou voile : le Ø d'étrier n'est plus retiré de la distance à l'axe",
}
# la SECONDE copie des deux colonnes de la norme (R2), transcrite ici du § 4.5 de docs/site/LOT_Q_QUATRE_MODULES.md :
# tableau 5.2a, colonne μ_fi = 0,7 (poteau exposé sur plus d'un côté) ; tableau 5.4, colonne μ_fi = 0,7, une face
POTEAU_NORME = {"R30": [(200, 32), (300, 27)], "R60": [(250, 46), (350, 40)], "R90": [(350, 53), (450, 40)],
                "R120": [(350, 57), (450, 51)], "R180": [(450, 70)], "R240": []}
VOILE_NORME = {"R30": 10, "R60": 10, "R90": 25, "R120": 35, "R180": 50, "R240": 60}
chk("écart voulu D1 : FEU_POTEAU = la colonne μ_fi = 0,7 du tableau 5.2a (transcrite ici), LUE dans beton/bibliotheque/feu.py — ce n'est plus la table de l'écran",
    D.FEU_POTEAU == POTEAU_NORME == {r: [(b, a) for b, a, _ in FEU.POTEAU_5_2A[r]["mu07"]] for r in D.R_LEVELS}
    and LUS["FEU_POTEAU"] != D.FEU_POTEAU and list(D.FEU_POTEAU) == D.R_LEVELS, D.FEU_POTEAU)
chk("… ce que l'écran portait sous le libellé « μ_fi = 0,7 » : la colonne μ_fi = 0,2 du tableau (R90 : 200/31, 300/25 ; R120 : 250/40, 350/35 ; R180 : 350/45)",
    all([tuple(v) for v in LUS["FEU_POTEAU"][r]][:len(FEU.POTEAU_5_2A[r]["mu02"])] == [(b, a) for b, a, _ in FEU.POTEAU_5_2A[r]["mu02"]] for r in ("R90", "R120", "R180")))
chk("écart voulu D2 : FEU_VOILE = le tableau 5.4, μ_fi = 0,7, une face (transcrit ici), LU dans feu.py ; il ne diffère de l'écran qu'à R180 (45 → 50) et R240 (55 → 60)",
    D.FEU_VOILE == VOILE_NORME == {r: FEU.VOILE_5_4[r]["mu07_1"][1] for r in D.R_LEVELS}
    and {r: (LUS["FEU_VOILE"][r], D.FEU_VOILE[r]) for r in D.R_LEVELS if LUS["FEU_VOILE"][r] != D.FEU_VOILE[r]} == {"R180": (45, 50), "R240": (55, 60)})
chk("les deux tables que l'écran et la norme partagent sont liées au moteur du feu : FEU_DALLE = tableau 5.8 (une direction), FEU_POUTRE = tableau 5.5",
    D.FEU_DALLE == {r: FEU.DALLE_5_8[r]["une"][0] for r in D.R_LEVELS} and D.FEU_POUTRE == {r: [(b, a) for b, a, _ in FEU.POUTRE_5_5[r]] for r in D.R_LEVELS})
chk("écart voulu D6 : seuls la poutre et le poteau ont un étrier à retirer de a", D.TYPES_A_ETRIER == ("Poutre", "Poteau"))


def ecart_voulu(el):
    """Le nom de l'écart voulu (D1, D2 ou D6) qui touche cet élément, ou None : la parité avec l'écran vaut."""
    if not el["feu"]:
        return None
    if el["type"] == "Poteau":
        return "D1"
    if el["type"] == "Voile" and el["r"] in ("R180", "R240"):
        return "D2"
    if el["type"] in ("Dalle", "Voile") and el["diam_etrier"] > 0:
        return "D6"
    return None


def controle_voulu(el) -> list:
    """Un élément touché par un écart voulu, recalculé ICI d'après la norme — les deux colonnes transcrites ci-dessus,
    la table des dalles de l'écran, et l'enchaînement de l'écran SANS l'étrier pour une dalle ou un voile (c_feu,min
    = max(a − Ø/2 − Ø_étrier ; 0) ; c_nom,feu = c_feu,min + Δc_dev ; c_nom = max(durabilité ; feu)) : les écarts avec ce
    que le moteur rend ([] : identique). La durabilité est celle de l'écran, recalculée de même."""
    res = M.calculer(el)
    d, f = res["durabilite"], res["feu"]
    cmd = LUS["C_MIN_DUR"][el["classe_struct"]][list(LUS["EXPOSITION_INFO"]).index(el["classe_expo"])]
    cmin = max(el["diam_barre"], cmd, 10)
    cnd = cmin + el["delta_dev"]
    e = []
    if (d["c_min_dur"], d["c_min"], d["c_nom"], res["c_nom_dur"]) != (cmd, cmin, cnd, cnd):
        e.append(f"durabilité {d} ≠ ({cmd}, {cmin}, {cnd})")
    if el["type"] == "Poteau" and not POTEAU_NORME[el["r"]]:
        # R240 : la case du tableau est vide — aucune valeur, l'enrobage est celui de la durabilité, et c'est dit en rouge
        if not (f["sans_valeur"] and f["a"] is None and f["c_nom"] is None and f["b_min"] is None and res["c_nom"] == cnd
                and res["gouvernant"] == "durabilite" and not res["proches"]
                and any(a["code"] == "feu_non_couvert" and a["grave"] and "non couvert par le tableau" in a["message"] for a in res["avertissements"])):
            e.append(f"poteau {el['r']} : attendu « non couvert par le tableau », reçu {f}")
        return e
    if el["type"] == "Poteau":
        a, etrier = dict(POTEAU_NORME[el["r"]])[el["b_min"]], el["diam_etrier"]
    elif el["type"] == "Voile":
        a, etrier = VOILE_NORME[el["r"]], 0
    else:
        a, etrier = LUS["FEU_DALLE"][el["r"]], 0
    cmin_f = max(a - el["diam_barre"] / 2.0 - etrier, 0.0)
    cnom_f = cmin_f + el["delta_dev"]
    attendu_ = (a, cmin_f, cnom_f, max(cnd, cnom_f), "durabilite" if cnd >= cnom_f else "feu", abs(cnd - cnom_f) <= 2, cmin_f <= 0)
    recu = (f["a"], f["c_min"], f["c_nom"], res["c_nom"], res["gouvernant"], res["proches"], f["couvert"])
    if recu != attendu_ or f["sans_valeur"] or f["diam_etrier"] != etrier:
        e.append(f"{el['type']} {el['r']} : moteur {recu} ≠ norme {attendu_}")
    return e
chk("EXPO_CLASSES = les clés de EXPOSITION_INFO, dans l'ordre", D.EXPO_CLASSES == list(LUS["EXPOSITION_INFO"]))
chk("CLASSES_STRUCTURALES = les clés de C_MIN_DUR, dans l'ordre", D.CLASSES_STRUCTURALES == list(LUS["C_MIN_DUR"]))
FONCTIONS_MODULE = {n.name: n for n in ARBRE.body if isinstance(n, ast.FunctionDef)}


class SansD(ast.NodeTransformer):
    """« D.FEU_DALLE » (le moteur lit ses tableaux dans donnees) → « FEU_DALLE » (le module les a en global)."""
    def visit_Attribute(self, n):
        self.generic_visit(n)
        if isinstance(n.value, ast.Name) and n.value.id == "D":
            return ast.copy_location(ast.Name(id=n.attr, ctx=n.ctx), n)
        return n


for nom in ("a_feu", "c_durabilite", "c_feu"):
    moteur = SansD().visit(ast.parse(textwrap.dedent(inspect.getsource(getattr(M, nom)))).body[0])
    chk(f"{nom} : la fonction du moteur est celle du module, mot pour mot (arbres syntaxiques égaux)",
        ast.dump(moteur) == ast.dump(FONCTIONS_MODULE[nom]))

# =================================================================== le module, importé (streamlit requis)
try:
    logging.disable(logging.WARNING)
    import streamlit as st  # noqa: F401
    from modules import enrobage as E
    from modules import navigation as NAV
except Exception as e:  # noqa: BLE001
    E = None
    print(f"     (streamlit absent : parité écran non jouée — {e})")


# =================================================================== 2. les fonctions, au bit près
DIAMS = list(range(6, 61))
ETRIERS = list(range(0, 21))
if E is not None:
    print("\n=== 2. Fonctions : a_feu, c_durabilite, c_feu du moteur = celles du module, au bit près ===")
    n, n_voulus, ecarts = 0, 0, []
    for t in D.TYPES + ["Autre"]:
        for r in D.R_LEVELS + ["R15"]:
            for v in [None] + EN.variantes(t, r) + [(999, 7)]:
                n += 1
                if t == "Poteau" and r in D.R_LEVELS and (v is None or not POTEAU_NORME[r]):
                    # écart voulu D1 : sans largeur choisie, la première variante est celle du tableau 5.2a (μ_fi = 0,7) ;
                    # R240 : la liste est vide, la fonction de l'écran rend (0, "") quelle que soit la largeur passée — le
                    # moteur ne l'appelle d'ailleurs pas dans ce cas (calculer : « non couvert par le tableau »)
                    n_voulus += 1
                    att = (POTEAU_NORME[r][0][1], f"Poteau exposé sur >1 face, μ_fi≈0,7, b_min={POTEAU_NORME[r][0][0]} mm (tab. 5.2a).") if POTEAU_NORME[r] else (0, "")
                    if not meme(M.a_feu(t, r, v), att):
                        ecarts.append(("D1", t, r, v, M.a_feu(t, r, v), att))
                elif t == "Voile" and r in ("R180", "R240"):
                    n_voulus += 1               # écart voulu D2 : 50 et 60 mm
                    if not meme(M.a_feu(t, r, v), (VOILE_NORME[r], E.a_feu(t, r, v)[1])):
                        ecarts.append(("D2", t, r, v, M.a_feu(t, r, v)))
                elif not meme(M.a_feu(t, r, v), E.a_feu(t, r, v)):
                    ecarts.append((t, r, v, M.a_feu(t, r, v), E.a_feu(t, r, v)))
    chk(f"a_feu : {n} appels (types, durées, variantes, cas hors liste) identiques à l'écran, sauf les {n_voulus} des écarts voulus D1 et D2 — "
        "qui rendent la valeur de la norme", not ecarts and n_voulus == 6 + 1 + 2 * 2, ecarts[:3])
    n, ecarts = 0, []
    for s in D.CLASSES_STRUCTURALES:
        for x in D.EXPO_CLASSES:
            for d in DIAMS + [6.5, 20.0]:
                for dd in D.DELTAS_DEV:
                    n += 1
                    if not meme(M.c_durabilite(s, x, d, dd), E.c_durabilite(s, x, d, dd)):
                        ecarts.append((s, x, d, dd))
    chk(f"c_durabilite : {n} appels identiques (valeurs et types)", not ecarts, ecarts[:3])
    A_TOUS = sorted({0} | set(D.FEU_DALLE.values()) | set(D.FEU_VOILE.values())
                    | {a for t in (D.FEU_POTEAU, D.FEU_POUTRE) for vs in t.values() for _, a in vs})
    n, ecarts = 0, []
    for a in A_TOUS:
        for d in DIAMS:
            for e in ETRIERS:
                for dd in D.DELTAS_DEV:
                    n += 1
                    if not meme(M.c_feu(a, d, e, dd), E.c_feu(a, d, e, dd)):
                        ecarts.append((a, d, e, dd))
    chk(f"c_feu : {n} appels identiques (toutes les distances a des tableaux, Ø 6–60, Ø étrier 0–20)", not ecarts, ecarts[:3])


# =================================================================== 3. l'écran, rejoué avec un faux st
class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran afficherait et répond aux widgets avec les valeurs de l'élément."""

    def __init__(self, reponses):
        self.rep, self.ev = reponses, []

    def _ctx(self):
        return contextlib.nullcontext()

    def markdown(self, body="", unsafe_allow_html=False, **_):
        self.ev.append(("md", str(body)))

    def caption(self, body="", **_):
        self.ev.append(("caption", str(body)))

    def columns(self, spec, **_):
        return [self._ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def expander(self, label, expanded=False, **_):
        self.ev.append(("exp", label, expanded))
        return self._ctx()

    def divider(self):
        self.ev.append(("divider",))

    def selectbox(self, label, options, index=0, format_func=str, help=None, **_):
        options = list(options)
        self.ev.append(("selectbox", label, [format_func(o) for o in options], index, help))
        voulu = self.rep[label]
        trouve = [o for o in options if o == voulu]
        if not trouve:  # le moteur répond une valeur que l'écran ne propose pas : écart enregistré
            self.ev.append(("hors_liste", label, voulu))
        return trouve[0] if trouve else options[0]

    def radio(self, label, options, index=0, horizontal=False, help=None, **_):
        self.ev.append(("radio", label, list(options), index, horizontal, help))
        return next(o for o in options if o == self.rep[label])

    def number_input(self, label, value=None, step=None, min_value=None, help=None, **_):
        self.ev.append(("number_input", label, value, step, min_value, help))
        return self.rep[label]

    def checkbox(self, label, value=False, **_):
        self.ev.append(("checkbox", label, value))
        return self.rep[label]

    def select_slider(self, label, options, value=None, **_):
        self.ev.append(("select_slider", label, list(options), value))
        return self.rep[label]


L_TYPE, L_POS, L_D, L_DE, L_DEV = ("Type d'élément", "Position dans l'ouvrage", "Ø max armatures principales [mm]",
                                   "Ø étrier / cadre [mm]", "Tolérance d'exécution Δc_dev [mm]")
L_EXPO, L_STRUCT, L_R, L_B = "Classe d'exposition", "Classe structurale", "Durée de résistance requise", "Largeur minimale de l'élément b [mm]"


def reponses(el):
    v = next((x for x in EN.variantes(el["type"], el["r"]) if x[0] == el["b_min"]), None)  # pas M._variante : il se mute
    return {L_TYPE: el["type"], L_POS: el["position"], L_D: el["diam_barre"], L_DE: el["diam_etrier"], L_DEV: el["delta_dev"],
            L_EXPO: el["classe_expo"], L_STRUCT: el["classe_struct"], D.LIBELLE_FEU: el["feu"], L_R: el["r"], L_B: v}


def ecran(el):
    """show() du module sur l'élément : (événements, encadrés, espions)."""
    faux = FauxSt(reponses(el))
    blocs, espions = [], {"c_durabilite": [], "c_feu": [], "a_feu": []}
    vrais = {k: getattr(E, k) for k in ("st", "_bloc", "c_durabilite", "c_feu", "a_feu")}

    def espion(nom):
        def f(*a):
            r = vrais[nom](*a)
            espions[nom].append((a, r))
            return r
        return f
    E.st, E._bloc = faux, (lambda left, right="", etat="ok": blocs.append((left, right, etat)))
    for k in espions:
        setattr(E, k, espion(k))
    vrai_nav = NAV.bouton_accueil
    NAV.bouton_accueil = lambda key, **kw: faux.ev.append(("accueil", key))
    try:
        E.show()
    finally:
        for k, v in vrais.items():
            setattr(E, k, v)
        NAV.bouton_accueil = vrai_nav
    return faux.ev, blocs, espions


GOUV = {"durabilite": "durabilité", "feu": "feu"}


def attendu(res):
    """Ce que l'écran DOIT afficher pour ce résultat du moteur : les widgets, les encadrés, le markdown, les captions —
    écrits avec les formats du module (:.0f…) à partir des seuls nombres du moteur."""
    el, d, f, x = res["element"], res["durabilite"], res["feu"], res["exposition"]
    wid = [("selectbox", L_TYPE, D.TYPES, 0, None), ("selectbox", L_POS, D.POSITIONS, 0, None),
           ("number_input", L_D, D.DIAM_BARRE_DEFAUT, 2, D.DIAM_BARRE_MIN, None),
           ("number_input", L_DE, D.DIAM_ETRIER_DEFAUT, 2, D.DIAM_ETRIER_MIN, D.AIDE_ETRIER),
           ("radio", L_DEV, D.DELTAS_DEV, 0, True, D.AIDE_DELTA_DEV),
           ("selectbox", L_EXPO, [f"{k} — {D.EXPOSITION_INFO[k][0]}" for k in D.EXPO_CLASSES], 0, None),
           ("selectbox", L_STRUCT, D.CLASSES_STRUCTURALES, D.CLASSES_STRUCTURALES.index(d["classe_structurale_proposee"]), D.AIDE_CLASSE_STRUCT),
           ("checkbox", D.LIBELLE_FEU, False)]
    if el["feu"]:
        wid.append(("select_slider", L_R, D.R_LEVELS, D.R_DEFAUT))
        if el["type"] in D.TYPES_VARIANTES:
            wid.append(("selectbox", L_B, [f"b ≥ {b} mm  →  a = {a} mm" for b, a in EN.variantes(el["type"], el["r"])], 0, D.AIDE_LARGEUR))
    blocs = [(f"{x['classe']} — {x['titre']}", "", "info"), ("A — Durabilité", f"c_nom = {d['c_nom']:.0f} mm", "ok")]
    md = ["## 🧱 " + D.TITRE, f"**Mécanisme** : {x['description']}", f"**Quand la choisir** : {x['quand']}", f"*Exemples* : {x['exemples']}",
          "### Résultats",
          f"- c_min,dur = **{d['c_min_dur']} mm** (classe {d['classe_structurale']} / {d['classe_exposition']})  \n"
          f"- c_min,b = **{d['c_min_b']:.0f} mm** (= Ø barre)  \n"
          f"- c_min = max(c_min,b ; c_min,dur ; 10) = **{d['c_min']:.0f} mm**  \n"
          f"- Δc_dev = **{d['delta_dev']} mm**  \n"
          f"- **c_nom,durabilité = {d['c_nom']:.0f} mm**"]
    cap = [D.CAPTION_TITRE] + ([D.CAPTION_FEU] if el["feu"] else [])
    if f is not None and f["sans_valeur"]:
        # (un poteau R240 : hors parité, écart voulu D1 — l'écran n'a pas d'équivalent à « non couvert par le tableau »)
        blocs.append((f"B — Feu ({f['r']})", "non couvert par le tableau", "ok"))
    elif f is not None:
        blocs.append((f"B — Feu ({f['r']})", f"c_nom = {f['c_nom']:.0f} mm", "ok"))
        cap.append(f["note"])
        md.append(f"- Distance à l'axe requise a = **{f['a']:.0f} mm**  \n"
                  f"- c_feu,min = a − Ø/2 − Ø_étrier = {f['a']:.0f} − {f['demi_barre']:.0f} − {f['diam_etrier']:.0f} = **{f['c_min']:.0f} mm**  \n"
                  f"- Δc_dev = **{f['delta_dev']} mm**  \n"
                  f"- **c_nom,feu = {f['c_nom']:.0f} mm**")
        if f["couvert"]:
            blocs.append(("Distance à l'axe couverte par la durabilité", "feu non dimensionnant", "info"))
    else:
        blocs.append(("B — Feu", "non vérifié", "info"))
        md.append(D.INVITE_FEU)
    blocs.append(("C — Enrobage nominal retenu", f"c_nom = {res['c_nom']:.0f} mm", "ok"))
    md.append(f"Valeur gouvernée par la **{GOUV[res['gouvernant']]}**.")
    if res["proches"]:
        blocs.append(("Durabilité et feu proches", "vérifier le cas réel", "warn"))
    blocs.append(D.GUIDE_REGLE + ("info",))
    md += [D.GUIDE_INTRO, D.GUIDE_CLASSES, D.GUIDE_PLUSIEURS]
    return wid, blocs, md, cap


def comparer(el):
    """Les écarts entre le moteur et l'écran du module pour un élément ([] : identiques)."""
    res = M.calculer(el)
    ev, blocs, esp = ecran(el)
    wid, blocs_att, md_att, cap_att = attendu(res)
    ecarts = []
    widgets = [e for e in ev if e[0] in ("selectbox", "radio", "number_input", "checkbox", "select_slider")]
    if widgets != wid:
        ecarts.append(f"widgets : écran {widgets} ≠ attendu {wid}")
    if blocs != blocs_att:
        ecarts.append(f"encadrés : écran {blocs} ≠ moteur {blocs_att}")
    md = [e[1] for e in ev if e[0] == "md"]
    if md != md_att:
        ecarts.append(f"markdown : écran {[m for m in md if m not in md_att]} ≠ moteur {[m for m in md_att if m not in md]}")
    cap = [e[1] for e in ev if e[0] == "caption"]
    if cap != cap_att:
        ecarts.append(f"captions : écran {cap} ≠ moteur {cap_att}")
    # les flottants exacts passés et rendus par les fonctions du module, pendant show()
    d, f = res["durabilite"], res["feu"]
    (args_d, ret_d), = esp["c_durabilite"]
    if not (meme(ret_d, (d["c_min_dur"], d["c_min_b"], d["c_min"], d["c_nom"])) and args_d == (el["classe_struct"], el["classe_expo"], el["diam_barre"], el["delta_dev"])):
        ecarts.append(f"c_durabilite : écran {args_d} → {ret_d} ≠ moteur {d}")
    if f is None:
        if esp["c_feu"] or esp["a_feu"]:
            ecarts.append("l'écran calcule le feu, pas le moteur")
    else:
        if not esp["c_feu"]:
            ecarts.append("le moteur calcule le feu, pas l'écran")
        else:
            (args_a, ret_a), = esp["a_feu"]
            (args_f, ret_f), = esp["c_feu"]
            if not (meme(ret_a, (f["a"], f["note"])) and meme(ret_f, (f["c_min"], f["c_nom"]))
                    and args_f == (f["a"], el["diam_barre"], el["diam_etrier"], el["delta_dev"])):
                ecarts.append(f"feu : écran a_feu{args_a} → {ret_a}, c_feu{args_f} → {ret_f} ≠ moteur {f}")
    c_ecran = max(ret_d[3], esp["c_feu"][0][1][1] if esp["c_feu"] else 0.0)
    if not meme(res["c_nom"], c_ecran):
        ecarts.append(f"c_nom : moteur {res['c_nom']!r} ≠ max de l'écran {c_ecran!r}")
    return ecarts, res


def element(**k):
    return EN.normaliser({**EN.nouvel_element(), "b_min": None, **k})


def grille():
    """Les éléments de la parité : la grille de la durabilité, la grille du feu, des tirages, des bornes exactes."""
    out = []
    for x in D.EXPO_CLASSES:                                     # durabilité : chaque case de C_MIN_DUR, Ø et Δc_dev
        for s in D.CLASSES_STRUCTURALES:
            for dd in D.DELTAS_DEV:
                for d in (6, 8, 9, 10, 12, 16, 20, 25, 32, 40):
                    out.append(element(classe_expo=x, classe_struct=s, delta_dev=dd, diam_barre=d))
    for t in D.TYPES:                                            # le type et la position, feu non vérifié
        for p in D.POSITIONS:
            for x in D.EXPO_CLASSES:
                out.append(element(type=t, position=p, classe_expo=x, classe_struct=D.CLASSE_STRUCTURALE_DEFAULT[x]))
    rng = random.Random(1907)
    for t in D.TYPES:                                            # feu : chaque durée, chaque variante
        for r in D.R_LEVELS:
            for v in (EN.variantes(t, r) or [(None, None)]):
                for d in (6, 7, 10, 12, 16, 20, 25, 32, 40):
                    for e in (0, 6, 8, 10, 12):
                        for dd in D.DELTAS_DEV:
                            x = rng.choice(D.EXPO_CLASSES)
                            out.append(element(type=t, r=r, b_min=v[0], feu=True, diam_barre=d, diam_etrier=e, delta_dev=dd,
                                               classe_expo=x, classe_struct=rng.choice(D.CLASSES_STRUCTURALES),
                                               position=rng.choice(D.POSITIONS)))
    for _ in range(3000):                                        # des tirages : Ø impairs, grands, étriers de 0 à 20
        t, r = rng.choice(D.TYPES), rng.choice(D.R_LEVELS)
        vs = EN.variantes(t, r)
        out.append(element(type=t, r=r, b_min=rng.choice(vs)[0] if vs else None, feu=rng.random() < 0.8,
                           diam_barre=rng.randint(6, 60), diam_etrier=rng.randint(0, 20), delta_dev=rng.choice(D.DELTAS_DEV),
                           classe_expo=rng.choice(D.EXPO_CLASSES), classe_struct=rng.choice(D.CLASSES_STRUCTURALES),
                           position=rng.choice(D.POSITIONS), nom=f"T{_}"))
    # les bornes exactes : écart feu / durabilité de 0, ±2, ±2,5 mm ; c_feu,min nul pile
    bornes = [  # (description, élément)
        ("égalité c_nom,dur = c_nom,feu = 35 (XC4/S4, Ø20, poutre R60 b ≥ 160 : 35 − 10 − 0 = 25)",
         element(classe_expo="XC4", classe_struct="S4", diam_barre=20, diam_etrier=0, feu=True, r="R60", b_min=160)),
        ("écart +2 pile : c_nom,dur 40, c_nom,feu 38 (XC4/S4, Ø20, R90 b ≥ 200 : 45 − 10 − 7 = 28)",
         element(classe_expo="XC4", classe_struct="S4", diam_barre=20, diam_etrier=7, feu=True, r="R90", b_min=200)),
        ("écart −2 pile : c_nom,dur 40, c_nom,feu 42 (R90 b ≥ 200 : 45 − 10 − 3)",
         element(classe_expo="XC4", classe_struct="S4", diam_barre=20, diam_etrier=3, feu=True, r="R90", b_min=200)),
        ("écart 2,5 : c_nom,dur 40, c_nom,feu 37,5 (Ø21 : 45 − 10,5 − 7)",
         element(classe_expo="XC4", classe_struct="S4", diam_barre=21, diam_etrier=7, feu=True, r="R90", b_min=200)),
        ("c_feu,min nul pile : dalle R30, Ø20, Ø étrier 0 (10 − 10 − 0 = 0)",
         element(type="Dalle", diam_barre=20, diam_etrier=0, feu=True, r="R30")),
        # (ces deux bornes portaient sur un voile avec étrier et sur un poteau : les écarts voulus D6 et D1 les sortent
        #  de la parité ; les mêmes situations sont tenues sur une dalle sans étrier et sur une poutre)
        ("c_feu,min négatif borné à 0 : dalle R30, Ø25, étrier 0 (10 − 12,5 − 0 = −2,5)",
         element(type="Dalle", diam_barre=25, diam_etrier=0, feu=True, r="R30")),
        ("demi-millimètre arrondi au pair par l'écran : poutre R60 b ≥ 120, Ø25, étrier 8 (40 − 12,5 − 8 = 19,5 → « 20 »)",
         element(diam_barre=25, diam_etrier=8, feu=True, r="R60", b_min=120)),
        ("demi-millimètre arrondi au pair par l'écran : poutre R60 b ≥ 120, Ø25, étrier 9 (40 − 12,5 − 9 = 18,5 → « 18 »)",
         element(diam_barre=25, diam_etrier=9, feu=True, r="R60", b_min=120)),
    ]
    return out, bornes


def faits(el, res):
    """Les situations qu'un élément atteint (la couverture)."""
    d, f = res["durabilite"], res["feu"]
    out = {f"terme {d['terme_c_min']}", f"Δc_dev {el['delta_dev']}", f"gouvernant {res['gouvernant']}"}
    if d["c_min_b"] == d["c_min_dur"] == d["c_min"]:
        out.add("égalité c_min,b = c_min,dur")
    if d["c_min"] == 10 and d["c_min_b"] < 10:
        out.add("plancher 10 mm atteint")
    if f is not None and f["sans_valeur"]:
        out |= {f"feu {el['type']}", f"feu {el['type']} {el['r']}", "poteau non couvert par le tableau"}
    elif f is not None:
        out |= {f"feu {el['type']}", f"feu {el['type']} {el['r']}", f"variante {el['type']} {el['r']} {el['b_min']}"}
        if f["c_brut"] == 0:
            out.add("c_feu,min nul pile")
        if f["c_brut"] < 0:
            out.add("c_feu,min négatif borné")
        if res["c_nom_dur"] == res["c_nom_feu"]:
            out.add("égalité c_nom,dur = c_nom,feu")
        ecart = abs(res["c_nom_dur"] - res["c_nom_feu"])
        if ecart == 2:
            out.add("écart 2 pile")
        if ecart == 2.5:
            out.add("écart 2,5")
        if res["c_nom"] != int(res["c_nom"]):
            out.add("demi-millimètre")
        if f"{f['c_min']:.0f}" != str(int(f["c_min"] + 0.5)):
            out.add("arrondi au pair de l'écran (x,5 → pair)")
    return out


if E is not None:
    print("\n=== 3. Écran : show() rejoué avec un faux st, comparé au moteur ===")
    CAS, BORNES = grille()
    PARITE = [el for el in CAS if ecart_voulu(el) is None]          # la parité stricte avec l'écran
    VOULUS = [el for el in CAS if ecart_voulu(el) is not None]      # les trois écarts voulus, tenus contre la norme
    n_ok, ecarts_tous, COUV = 0, [], set()
    for el in PARITE:
        ecarts, res = comparer(el)
        COUV |= faits(el, res)
        if ecarts:
            ecarts_tous.append((el, ecarts))
        else:
            n_ok += 1
    chk(f"{len(PARITE)} éléments en PARITÉ : widgets, encadrés, markdown, captions et flottants de l'écran = moteur ({n_ok} identiques)",
        not ecarts_tous, ecarts_tous[:2])
    par_ecart, ecarts_voulus = {k: 0 for k in ECARTS_VOULUS}, []
    for el in VOULUS:
        par_ecart[ecart_voulu(el)] += 1
        ecarts = controle_voulu(el)
        COUV |= faits(el, M.calculer(el))
        if ecarts:
            ecarts_voulus.append((el, ecarts))
    chk(f"{len(VOULUS)} éléments touchés par un ÉCART VOULU depuis le 10/11/2026 ({', '.join(f'{k} : {v}' for k, v in par_ecart.items())}) : "
        "le moteur = la norme, recalculée ici (colonnes transcrites, étrier non retiré d'une dalle ou d'un voile)",
        not ecarts_voulus and all(par_ecart.values()), ecarts_voulus[:2])
    for k, v in ECARTS_VOULUS.items():
        print(f"         écart voulu {k} — {v} : {par_ecart[k]} éléments")
    # … et ces éléments sont bien des ÉCARTS : l'écran, lui, afficherait autre chose (sinon l'exclusion ne servirait à rien)
    temoins = {"D1": element(type="Poteau", feu=True, r="R30", b_min=300), "D2": element(type="Voile", feu=True, r="R180", diam_etrier=0),
               "D6": element(type="Dalle", feu=True, r="R90", diam_etrier=8)}
    chk("un témoin de chaque écart voulu : l'écran Streamlit affiche bien AUTRE CHOSE que le moteur (poteau R30 b ≥ 300 : 25 mm contre 27 ; "
        "voile R180 : 45 contre 50 ; dalle R90, étrier 8 : c_feu,min 12 contre 20)",
        all(ecart_voulu(el) == k and comparer(el)[0] and not controle_voulu(el) for k, el in temoins.items())
        and M.calculer(temoins["D1"])["feu"]["a"] == 27 and M.calculer(temoins["D2"])["feu"]["a"] == 50 and M.calculer(temoins["D6"])["feu"]["c_min"] == 20.0)
    for desc, el in BORNES:
        ecarts, res = comparer(el)
        COUV |= faits(el, res)
        chk(f"borne · {desc} → c_nom {res['c_nom']!r} ({res['gouvernant']}, proches {res['proches']})", not ecarts and ecart_voulu(el) is None, ecarts[:3])
    chk("au moins 10 000 éléments passés, dont plus de 6 000 en parité stricte avec l'écran", len(CAS) >= 10000 and len(PARITE) > 6000, (len(CAS), len(PARITE)))

    print("\n=== 4. Couverture : ce que les cas atteignent réellement ===")
    ATTENDUS = (["terme c_min_b", "terme c_min_dur", "égalité c_min,b = c_min,dur", "plancher 10 mm atteint",
                 "gouvernant durabilite", "gouvernant feu", "c_feu,min nul pile", "c_feu,min négatif borné",
                 "égalité c_nom,dur = c_nom,feu", "écart 2 pile", "écart 2,5", "demi-millimètre", "arrondi au pair de l'écran (x,5 → pair)"]
                + [f"Δc_dev {x}" for x in D.DELTAS_DEV]
                + [f"feu {t} {r}" for t in D.TYPES for r in D.R_LEVELS]
                + [f"variante {t} {r} {b}" for t in D.TYPES_VARIANTES for r in D.R_LEVELS for b, _ in EN.variantes(t, r)]
                + ["poteau non couvert par le tableau"])
    manque = [a for a in ATTENDUS if a not in COUV]
    chk(f"les cas atteignent les {len(ATTENDUS)} situations attendues (termes, gouvernants, bornes, durées, variantes)",
        not manque, manque)
    chk("le plancher de 10 mm ne gouverne jamais SEUL (c_min,dur ≥ 10 dans tout le tableau) : il n'est atteint qu'à égalité",
        "terme plancher" not in COUV and min(min(v) for v in D.C_MIN_DUR.values()) == 10)


# =================================================================== 5. AppTest : le vrai écran Streamlit
def lire_bloc(h):
    """L'encadré HTML de _bloc → (titre, valeur, état)."""
    gauche = re.search(r'<div style="font-weight:700;">(.*?)</div>', h, re.S)
    droite = re.search(r"white-space:nowrap;'>(.*?)</div>", h, re.S)
    icone = re.search(r'<div style="font-size:20px;line-height:1;">(.*?)</div>', h, re.S)
    etats = {v: k for k, v in E.C_ICONES.items()}
    return (gauche.group(1) if gauche else "", droite.group(1) if droite else "", etats.get(icone.group(1) if icone else "", "?"))


def appli():
    import sys as _s
    _s.path.insert(0, "/home/user/DEV_Etudes_structure")
    from modules import enrobage
    enrobage.show()


def etat_apptest(at):
    """L'élément que l'écran affiche (valeurs des widgets) et ses encadrés de résultat."""
    sb = {s.label: s.value for s in at.selectbox}
    el = {"type": sb[L_TYPE], "position": sb[L_POS], "diam_barre": at.number_input[0].value,
          "diam_etrier": at.number_input[1].value, "delta_dev": at.radio[0].value, "classe_expo": sb[L_EXPO],
          "classe_struct": sb[L_STRUCT], "feu": at.checkbox[0].value}
    if el["feu"]:
        el["r"] = at.select_slider[0].value
        el["b_min"] = sb[L_B][0] if L_B in sb else None
    blocs = [lire_bloc(m.value) for m in at.markdown if m.value.lstrip().startswith("<div style=\"background:")]
    return el, blocs


if E is not None:
    print("\n=== 5. AppTest : le VRAI écran Streamlit, pas à pas, contre le moteur ===")
    from streamlit.testing.v1 import AppTest

    def sel(at, label):
        return next(s for s in at.selectbox if s.label == label)

    compares = {"parite": 0, "voulus": 0}

    def rejouer(sequence, appliquer=EN.appliquer):
        """Joue ``sequence`` [(champ, valeur), …] sur le vrai écran ET sur le moteur ; rend la liste des écarts."""
        at = AppTest.from_function(appli, default_timeout=30).run()
        el = EN.nouvel_element()
        ecarts = []
        for k, (champ, valeur) in enumerate([(None, None)] + list(sequence)):
            if champ is not None:
                if champ == "feu":
                    at.checkbox[0].set_value(valeur).run()
                elif champ == "r":
                    at.select_slider[0].set_value(valeur).run()
                elif champ == "b_min":
                    # (la variante de l'ÉCRAN : pour un poteau, ses couples ne sont plus ceux du moteur — écart voulu D1)
                    ecran_ = {"Poutre": E.FEU_POUTRE, "Poteau": E.FEU_POTEAU}[el["type"]][el["r"]]
                    sel(at, L_B).set_value(next(tuple(v) for v in ecran_ if v[0] == valeur)).run()
                elif champ == "diam_barre":
                    at.number_input[0].set_value(valeur).run()
                elif champ == "diam_etrier":
                    at.number_input[1].set_value(valeur).run()
                elif champ == "delta_dev":
                    at.radio[0].set_value(valeur).run()
                else:
                    sel(at, {"type": L_TYPE, "position": L_POS, "classe_expo": L_EXPO, "classe_struct": L_STRUCT}[champ]).set_value(valeur).run()
                el = appliquer(el, champ, valeur)
            if at.exception:
                return [f"exception de l'écran : {at.exception}"]
            if ecart_voulu(el) is not None:
                # un pas touché par un écart voulu (D1, D2, D6) : l'écran et le moteur ne disent plus la même chose —
                # le pas est joué des deux côtés, pas comparé ; le moteur y est tenu contre la norme
                ecarts += [f"pas {k} ({champ} = {valeur!r}), écart voulu : {x}" for x in controle_voulu(el)]
                compares["voulus"] += 1
                continue
            compares["parite"] += 1
            vu, blocs = etat_apptest(at)
            attendu_el = {c: el[c] for c in vu}
            if vu != attendu_el:
                ecarts.append(f"pas {k} ({champ} = {valeur!r}) : écran {vu} ≠ moteur {attendu_el}")
            _, blocs_att, _, _ = attendu(M.calculer(el))
            if blocs != blocs_att:
                ecarts.append(f"pas {k} ({champ} = {valeur!r}) : encadrés écran {blocs} ≠ moteur {blocs_att}")
        return ecarts

    SEQ_STRUCT = [("classe_expo", "XC1"), ("classe_struct", "S5"), ("classe_expo", "XC2/XC3"), ("classe_struct", "S6"),
                  ("classe_expo", "XC4"), ("classe_expo", "XD1/XS1"), ("classe_struct", "S3"), ("classe_expo", "XD2/XS2"),
                  ("classe_expo", "XD3/XS3"), ("classe_expo", "X0")]
    # (jusqu'au 10/11/2026 la séquence passait par un poteau R30 → R60, seul cas de l'écran où deux durées ont la même
    #  liste de largeurs ; les poteaux sont sortis de la parité — écart voulu D1 — : la largeur « gardée quand la liste
    #  est la même » est tenue par une durée re-saisie à l'identique, et la séquence finit par les pas des écarts voulus)
    SEQ_FEU = [("feu", True), ("b_min", 200), ("r", "R90"), ("b_min", 300), ("r", "R90"), ("diam_barre", 25), ("diam_etrier", 10),
               ("delta_dev", 5), ("type", "Dalle"), ("diam_etrier", 0), ("r", "R240"), ("type", "Poutre"), ("type", "Voile"), ("r", "R120"),
               ("delta_dev", 0), ("position", "Extérieur"), ("r", "R180"), ("type", "Poteau"), ("r", "R30"), ("b_min", 300), ("r", "R60")]
    SEQ_DIAM = [("diam_barre", 7), ("diam_etrier", 0), ("diam_barre", 41), ("classe_expo", "XC4"), ("delta_dev", 0),
                ("feu", True), ("r", "R240"), ("b_min", 700), ("diam_etrier", 12)]
    for nom, seq in (("classe structurale remise à la classe proposée quand elle change, gardée sinon (10 pas)", SEQ_STRUCT),
                     ("largeur au feu remise à la première variante quand la liste change, gardée sinon (21 pas, dont 7 d'écarts voulus)", SEQ_FEU),
                     ("Ø, Δc_dev, R240 et la plus grande largeur (9 pas)", SEQ_DIAM)):
        ecarts = rejouer(seq)
        chk(f"AppTest · {nom} : widgets et encadrés de l'écran = moteur à chaque pas", not ecarts, ecarts[:3])
    chk("AppTest : 36 pas comparés à l'écran en parité, 7 pas d'écarts voulus (dalle avec étrier, voile R240 et R180, poteau) tenus contre la norme",
        compares == {"parite": 36, "voulus": 7}, compares)
    # l'écart VOULU : décocher puis recocher le feu — l'écran oublie la durée et la largeur (widgets effacés), le site
    # les garde (docs/beton/ENROBAGE_MOTEUR.md § 5) ; le contrôle rougit si l'écran change de comportement
    at = AppTest.from_function(appli, default_timeout=30).run()
    at.checkbox[0].check().run()
    at.select_slider[0].set_value("R120").run()
    at.checkbox[0].uncheck().run()
    at.checkbox[0].check().run()
    el = EN.nouvel_element()
    for c, v in (("feu", True), ("r", "R120"), ("feu", False), ("feu", True)):
        el = EN.appliquer(el, c, v)
    chk("écart voulu : l'écran remet R60 après décocher / recocher le feu, le moteur garde R120",
        at.select_slider[0].value == "R60" and el["r"] == "R120", (at.select_slider[0].value, el["r"]))

# =================================================================== 6. à la main
print("\n=== 6. À la main (EN 1992-1-1 §4.4.1, tableaux de l'écran) ===")
MAIN = [  # (élément, c_min,dur, c_min, c_nom,dur, a, c_feu,min, c_nom retenu, gouvernant)
    (dict(classe_expo="XC4", classe_struct="S4", diam_barre=20, diam_etrier=8, delta_dev=10, feu=True, r="R90", b_min=200),
     30, 30, 40, 45, 27.0, 40, "durabilite"),          # 45 − 20/2 − 8 = 27 ; +10 = 37 < 40
    (dict(classe_expo="XC1", classe_struct="S2", diam_barre=25, diam_etrier=10, delta_dev=10, feu=True, r="R120", b_min=200),
     10, 25, 35, 65, 42.5, 52.5, "feu"),               # 65 − 12,5 − 10 = 42,5 ; +10 = 52,5 > 35
    (dict(type="Dalle", classe_expo="XC2/XC3", classe_struct="S4", diam_barre=12, diam_etrier=0, delta_dev=5, feu=True, r="R60"),
     25, 25, 30, 20, 14.0, 30, "durabilite"),          # 20 − 6 − 0 = 14 ; +5 = 19 < 30
    (dict(type="Voile", classe_expo="XD3/XS3", classe_struct="S6", diam_barre=16, diam_etrier=0, delta_dev=0, feu=True, r="R240"),
     55, 55, 55, 60, 52.0, 55, "durabilite"),          # écart voulu D2 : a = 60 (l'écran : 55) ; 60 − 8 = 52 < 55
    # --- les trois écarts voulus depuis le 10/11/2026, à la main
    (dict(type="Poteau", classe_expo="XC1", classe_struct="S2", diam_barre=20, diam_etrier=8, delta_dev=10, feu=True, r="R90", b_min=350),
     10, 20, 30, 53, 35.0, 45.0, "feu"),               # D1 : tab. 5.2a, μ_fi = 0,7 : 350/53 ; 53 − 10 − 8 = 35 ; +10 = 45 > 30
    (dict(type="Poteau", classe_expo="XC1", classe_struct="S2", diam_barre=20, diam_etrier=8, delta_dev=10, feu=True, r="R90", b_min=450),
     10, 20, 30, 40, 22.0, 32.0, "feu"),               # D1 : 450/40 ; 40 − 10 − 8 = 22 ; +10 = 32 > 30
    (dict(type="Voile", classe_expo="XC1", classe_struct="S2", diam_barre=12, diam_etrier=8, delta_dev=10, feu=True, r="R180"),
     10, 12, 22, 50, 44.0, 54.0, "feu"),               # D2 et D6 : a = 50, aucun étrier retiré : 50 − 6 = 44 ; +10 = 54
    (dict(type="Dalle", classe_expo="XC2/XC3", classe_struct="S4", diam_barre=12, diam_etrier=8, delta_dev=5, feu=True, r="R60"),
     25, 25, 30, 20, 14.0, 30, "durabilite"),          # D6 : 20 − 6 = 14 (l'écran : 20 − 6 − 8 = 6) ; +5 = 19 < 30
    (dict(type="Poteau", classe_expo="X0", classe_struct="S1", diam_barre=8, diam_etrier=6, delta_dev=10, feu=False),
     10, 10, 20, None, None, 20, "durabilite"),        # max(8 ; 10 ; 10) = 10 ; +10
]
for k, (el, cmd, cmin, cnd, a, cfm, cn, gv) in enumerate(MAIN, 1):
    r = M.calculer(element(**el))
    d, f = r["durabilite"], r["feu"]
    ok = (d["c_min_dur"], d["c_min"], d["c_nom"], r["c_nom"], r["gouvernant"]) == (cmd, cmin, cnd, cn, gv) \
        and ((f is None and a is None) or (f["a"], f["c_min"]) == (a, cfm))
    chk(f"à la main {k} : c_min,dur {cmd}, c_min {cmin}, c_nom,dur {cnd}, a {a}, c_feu,min {cfm} → c_nom {cn} ({gv})", ok,
        (d, f and {x: f[x] for x in ("a", "c_min", "c_nom")}, r["c_nom"], r["gouvernant"]))
r240 = M.calculer(element(type="Poteau", classe_expo="XC4", classe_struct="S4", feu=True, r="R240"))
chk("à la main · poteau R240 (écart voulu D1) : le tableau 5.2a n'a pas de valeur à μ_fi = 0,7 — « non couvert par le tableau », aucune largeur, "
    "aucun enrobage de feu, c_nom = durabilité (40), avertissement GRAVE ; l'encadré « feu couvert par la durabilité » ne s'affiche pas",
    r240["feu"]["sans_valeur"] and r240["feu"]["a"] is None and r240["feu"]["c_nom"] is None and r240["feu"]["b_min"] is None and r240["c_nom"] == 40
    and r240["gouvernant"] == "durabilite" and [a["code"] for a in r240["avertissements"] if a["grave"]] == ["feu_non_couvert"]
    and not any(a["code"] in ("feu_couvert", "feu_proche") for a in r240["avertissements"]) and "non couvert par le tableau" in r240["feu"]["note"]
    and EN.variantes("Poteau", "R240") == [] and M.candidats_feu(element(type="Poteau", feu=True, r="R240")) == [])
chk("à la main · poteau R90 b ≥ 450 : « au moins 8 barres » (le repère ** du tableau), dit ; b ≥ 350 : non",
    any(a["code"] == "huit_barres" for a in M.calculer(element(type="Poteau", feu=True, r="R90", b_min=450))["avertissements"])
    and not any(a["code"] == "huit_barres" for a in M.calculer(element(type="Poteau", feu=True, r="R90", b_min=350))["avertissements"]))
c = M.candidats_feu(element(feu=True, r="R90", b_min=200))
chk("sélecteur poutre R90 : quatre largeurs triées, la retenue (200) et l'optimale (400 : la durabilité gouverne, c_nom 30)",
    [x["b_min"] for x in c] == [150, 200, 300, 400] and [x["retenue"] for x in c] == [False, True, False, False]
    and [x["optimale"] for x in c] == [False, False, False, True] and [x["c_nom"] for x in c] == [47.0, 37.0, 32.0, 30], c)
c = M.candidats_feu(element(classe_expo="XD3/XS3", classe_struct="S6", feu=True, r="R30"))
chk("sélecteur poutre R30 en XD3/S6 : la durabilité gouverne partout, l'optimale est la PLUS PETITE largeur (80)",
    [x["optimale"] for x in c] == [True, False, False, False] and len({x["c_nom"] for x in c}) == 1, c)
chk("sélecteur : rien pour une dalle ou un voile", M.candidats_feu(element(type="Dalle")) == [] == M.candidats_feu(element(type="Voile")))


# =================================================================== 7. mutations
def mutant(fonction, avant, apres, module=M):
    """Une copie de ``fonction`` dont le source a ``avant`` remplacé par ``apres`` (le texte doit y être)."""
    src = textwrap.dedent(inspect.getsource(fonction))
    assert avant in src, (fonction.__name__, avant)
    espace = dict(vars(module))
    exec(compile(src.replace(avant, apres, 1), f"<mutant {fonction.__name__}>", "exec"), espace)
    return espace[fonction.__name__]


@contextlib.contextmanager
def remplace(objet, nom, valeur):
    vrai = getattr(objet, nom)
    setattr(objet, nom, valeur)
    try:
        yield
    finally:
        setattr(objet, nom, vrai)


if E is not None:
    print("\n=== 7. Mutations : la parité DOIT rougir ===")
    # (retirer le plancher de 10 mm est une mutation ÉQUIVALENTE : c_min,dur ≥ 10 dans tout le tableau, le plancher ne
    # change jamais rien — la section 4 le prouve ; elle n'est donc pas jouée ici)
    CAS_M = [el for desc, el in BORNES] + PARITE[::7]

    def rougit():
        return sum(1 for el in CAS_M if comparer(el)[0])
    for desc, objet, nom, valeur in (
            ("une distance du tableau 5.5 changée (R90, b ≥ 200 : 45 → 44)", D, "FEU_POUTRE",
             {**D.FEU_POUTRE, "R90": [(150, 55), (200, 44), (300, 40), (400, 35)]}),
            ("une case de c_min,dur changée (S4 / XC4 : 30 → 25)", D, "C_MIN_DUR", {**D.C_MIN_DUR, "S4": [15, 20, 25, 25, 35, 40, 45]}),
            ("la borne à 0 de c_feu,min retirée", M, "c_feu", mutant(M.c_feu, "max(a_mm - diam_barre / 2.0 - diam_etrier, 0.0)",
                                                                       "a_mm - diam_barre / 2.0 - diam_etrier")),
            ("Ø/2 devenu Ø//2 (entier)", M, "c_feu", mutant(M.c_feu, "diam_barre / 2.0", "diam_barre // 2")),
            ("Δc_dev oublié dans c_nom,dur", M, "c_durabilite", mutant(M.c_durabilite, "c_min + delta_dev", "c_min")),
            ("à égalité, le feu gouverne (≥ devenu >)", M, "calculer", mutant(M.calculer, "cnom_d >= cnom_f", "cnom_d > cnom_f")),
            ("l'encadré « proches » à moins de 2 mm (≤ devenu <)", M, "calculer", mutant(M.calculer, "<= 2)", "< 2)")),
            ("le feu calculé même non coché", M, "calculer", mutant(M.calculer, 'if el["feu"] and el["r"]:', 'if el["r"]:')),
            ("la largeur retenue ignorée (toujours la première)", M, "_variante", lambda el: (EN.variantes(el["type"], el["r"]) or [None])[0])):
        with remplace(objet, nom, valeur):
            n = rougit()
        chk(f"mutation · {desc} → {n} élément(s) en écart", n > 0)
    # --- chaque écart voulu REVENU (le défaut d'hier) fait rougir son contrôle contre la norme
    def voulus_rouges(elements):
        n = 0
        for el in elements:
            try:
                n += bool(controle_voulu(el))
            except (ValueError, KeyError):      # une largeur de la norme absente de la table abîmée : refusée, c'est rougir aussi
                n += 1
        return n
    ecran_poteau = {r: [tuple(v) for v in LUS["FEU_POTEAU"][r]] for r in D.R_LEVELS}
    for desc, objet, nom, valeur, touches in (
            ("D1 revenu : FEU_POTEAU remis à la table de l'écran (colonne μ_fi = 0,2)", D, "FEU_POTEAU", ecran_poteau, "D1"),
            ("D1 : une case de la colonne μ_fi = 0,7 changée (R90 : 350/43)", D, "FEU_POTEAU", {**D.FEU_POTEAU, "R90": [(350, 43), (450, 40)]}, "D1"),
            ("D1 : une valeur inventée pour un poteau R240 (450/40, la variante de l'écran)", D, "FEU_POTEAU", {**D.FEU_POTEAU, "R240": [(450, 40)]}, "D1"),
            ("D2 revenu : voiles R180 et R240 à 45 et 55 mm", D, "FEU_VOILE", {**D.FEU_VOILE, "R180": 45, "R240": 55}, "D2"),
            ("D6 revenu : l'étrier de nouveau retiré de a pour une dalle ou un voile", D, "TYPES_A_ETRIER", ("Poutre", "Poteau", "Dalle", "Voile"), "D6")):
        elements = [el for el in VOULUS if ecart_voulu(el) == touches][::5]
        with remplace(objet, nom, valeur):
            n = voulus_rouges(elements)
        chk(f"mutation · {desc} → {n} élément(s) de l'écart voulu {touches} en écart avec la norme", n > 0 and voulus_rouges(elements) == 0)
    for desc, mut in (("la classe structurale jamais remise à la classe proposée", mutant(EN.appliquer, "if D.CLASSE_STRUCTURALE_DEFAULT[valeur] != ",
                                                                                            "if False and D.CLASSE_STRUCTURALE_DEFAULT[valeur] != ", EN)),
                      ("la classe structurale TOUJOURS remise", mutant(EN.appliquer, "if D.CLASSE_STRUCTURALE_DEFAULT[valeur] != ",
                                                                       "if True or D.CLASSE_STRUCTURALE_DEFAULT[valeur] != ", EN)),
                      ("la largeur TOUJOURS remise", mutant(EN.appliquer, "if nouvelles != anciennes:", "if True:", EN)),
                      ("la largeur jamais remise", mutant(EN.appliquer, "if nouvelles != anciennes:", "if False:", EN))):
        try:
            n = len(rejouer(SEQ_STRUCT if "structurale" in desc else SEQ_FEU, appliquer=mut))
        except (ValueError, StopIteration):  # une largeur jamais remise sort du tableau : refusée, c'est rougir aussi
            n = 1
        chk(f"mutation · {desc} → l'AppTest rougit ({n} écart(s))", n > 0)

# =================================================================== 8. format
print("\n=== 8. Format : normalisation, JSON, saisie ===")
e0 = EN.nouvel_element()
chk("nouvel_element : les défauts de l'écran (Poutre, Intérieur, Ø20, Ø8, Δc_dev 10, X0, S1, sans feu, R60, b ≥ 120)",
    e0 == {"nom": "Élément 1", "type": "Poutre", "position": "Intérieur", "diam_barre": 20, "diam_etrier": 8, "delta_dev": 10,
           "classe_expo": "X0", "classe_struct": "S1", "feu": False, "r": "R60", "b_min": 120}, e0)
chk("normaliser est idempotent et complète un élément vide", EN.normaliser({}) == e0 == EN.normaliser(EN.normaliser(e0)))
chk("une classe structurale absente prend la classe proposée pour l'exposition (XD3/XS3 → S6)",
    EN.normaliser({"classe_expo": "XD3/XS3"})["classe_struct"] == "S6")
REFUS = [({"zz": 1}, "clé inconnue"), ({"type": "Mur"}, "type inconnu"), ({"diam_barre": 5}, "Ø sous le minimum (6)"),
         ({"diam_barre": 20.5}, "Ø non entier"), ({"diam_barre": "20"}, "Ø en texte"), ({"diam_barre": True}, "Ø booléen"),
         ({"diam_etrier": -2}, "Ø d'étrier négatif"), ({"delta_dev": 7}, "Δc_dev hors liste"), ({"classe_expo": "XF1"}, "XF non couvert"),
         ({"classe_struct": "S7"}, "classe structurale inconnue"), ({"r": "R45"}, "durée inconnue"), ({"feu": "oui"}, "feu non booléen"),
         ({"b_min": 130}, "largeur absente du tableau"), ({"type": "Dalle", "b_min": 120}, "largeur pour une dalle"),
         ({"nom": 3}, "nom non textuel"), ({"diam_barre": float("inf")}, "Ø infini"), ([], "pas un dict"),
         # garde-fou du site (relecture du 07/11/2026) : l'écran n'a pas de maximum, 10⁴⁰⁰ y ferait déborder les flottants
         ({"diam_barre": 61}, "Ø au-delà de 60 mm"), ({"diam_barre": 10 ** 400}, "Ø démesuré (10⁴⁰⁰)"),
         ({"diam_barre": 1e300}, "Ø flottant démesuré"), ({"diam_etrier": 41}, "Ø d'étrier au-delà de 40 mm")]
for el, desc in REFUS:
    try:
        EN.normaliser(el)
        refuse = False
    except ValueError:
        refuse = True
    chk(f"refusé (ValueError) : {desc}", refuse)
chk("les bornes du garde-fou sont admises : Ø60 et étrier Ø40 se calculent",
    M.calculer({"diam_barre": 60, "diam_etrier": 40, "feu": True, "r": "R240"})["c_nom"] == 70)
try:
    EN.normaliser({"diam_barre": 10 ** 400})
    msg = ""
except ValueError as ex:
    msg = str(ex)
chk("Ø de 10⁴⁰⁰ : refusé par un message lisible (pas un OverflowError)", "au plus 60 mm" in msg and "démesuré" in msg, msg)
chk("Ø entier écrit en flottant (20.0) accepté et rendu entier", EN.normaliser({"diam_barre": 20.0})["diam_barre"] == 20
    and type(EN.normaliser({"diam_barre": 20.0})["diam_barre"]) is int)
r = M.calculer(element(feu=True, r="R120", type="Poteau", diam_barre=25))
chk("la sortie de calculer est du JSON strict et se relit à l'identique", json.loads(json.dumps(r, allow_nan=False)) == json.loads(json.dumps(r)))
try:
    EN.appliquer(e0, "diam_barre", 4)
    refuse = False
except ValueError:
    refuse = True
chk("appliquer refuse un Ø de 4 mm (ValueError) et ne touche pas l'élément reçu", refuse and e0["diam_barre"] == 20)
chk("chaque terme porte une règle (EN 1992-1-1, EN 1992-1-2 ou convention de l'écran)",
    all(t["regle"].startswith(("EN 1992-1-", "Convention de l'écran", "Écran")) for t in r["termes"]), r["termes"])
codes = {a["code"] for a in r["avertissements"]}
chk("poteau au feu : la SOURCE est dite (tableau 5.2a, colonne μ_fi = 0,7) ; les avertissements rouges D1, D2 et D6 d'avant le 10/11/2026 ont disparu",
    "source_feu" in codes and not codes & {"tableau_poteau", "tableau_voile", "etrier_decompte"} and not any(a["grave"] for a in r["avertissements"])
    and "tableau 5.2a" in r["feu"]["source"] and "μ_fi = 0,7" in r["feu"]["source"] and "EN 1992-1-2:2004" in r["feu"]["source"])
rv = M.calculer(element(type="Voile", feu=True, r="R240", diam_etrier=8))
chk("voile au feu : la source (tableau 5.4, μ_fi = 0,7, une face) ; l'étrier saisi n'est pas retiré, et c'est dit (non grave)",
    "tableau 5.4" in rv["feu"]["source"] and {"source_feu", "etrier_non_retire"} <= {a["code"] for a in rv["avertissements"]}
    and not any(a["grave"] for a in rv["avertissements"]) and rv["feu"]["diam_etrier"] == 0 and rv["feu"]["c_min"] == 60 - 10.0)

# =================================================================== 9. pureté
print("\n=== 9. Pureté : bibliothèque standard seule, Python 3.12 et 3.13 ===")
SCRIPT = r"""
import json, sys
INTERDITS = {"streamlit", "numpy", "pandas", "matplotlib", "scipy", "reportlab", "fitz", "PIL"}
avant = set(sys.modules)
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
sys.path.insert(0, sys.argv[1])
from beton.enrobage import moteur as M
e = M.nouvel_element(); e.update(feu=True, r="R90", b_min=200, classe_expo="XC4", classe_struct="S4")
r = M.calculer(e)
print(json.dumps({"c_nom": r["c_nom"], "feu": r["feu"]["c_nom"], "charges": sorted({m.split(".")[0] for m in set(sys.modules) - avant})}))
"""
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
sorties = []
for py in interpretes:
    p = subprocess.run([py, "-I", "-c", SCRIPT, RACINE], capture_output=True, text=True, timeout=60)
    v = subprocess.run([py, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True).stdout.strip()
    try:
        s = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        s = None
    sorties.append(s)
    chk(f"Python {v} : streamlit, numpy, pandas… REFUSÉS par sys.meta_path, et le moteur calcule quand même (c_nom 40, feu 37)",
        s is not None and s["c_nom"] == 40 and s["feu"] == 37.0 and "beton" in s["charges"], (p.stderr[-600:], s))
if len(sorties) > 1:
    chk("résultats identiques sous les deux interprètes", sorties[0] == sorties[1])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""Moteur « Choix de profilé » (acier/choix_profil) : la PARITÉ avec l'écran Streamlit modules/choix_profile.py, qui
reste la référence. Chaque contrôle rougit si l'on retire ce qu'il protège (``--mutations`` le prouve).

  1. Données : profils() est load_profiles() du module ; les familles proposées, les défauts et les bornes de chaque
     champ sont ceux des widgets de l'écran, RELEVÉS en l'exécutant.
  2. À la main : des résistances et des taux recalculés ici avec les formules EC3 écrites en N et mm ; les valeurs
     PUBLIÉES de χ (EN 1993-1-1, tableau des courbes, λ̄ = 1,0) ; le palier χ = 1.
  3. Parité : chaque cas est rejoué dans l'ÉCRAN du module (show() exécuté avec un faux « st » qui enregistre au lieu
     d'afficher) et comparé au moteur :
       a) le tableau trié (toutes les colonnes, au bit près), ses couleurs de ligne (optimal, ≤ 100 %, > 100 %) ;
       b) le menu « Sélectionner un profilé » : ses options (l'ordre) et son index par défaut (l'optimal du module) ;
       c) la colonne de droite : le titre, les deux tableaux de la fiche, CHAQUE formule LaTeX (les nombres écrits),
          chaque message (st.info, st.warning, st.caption), dans l'ordre ;
       d) les taux exacts : les fonctions du module appelées sur chaque profilé (résistances, taux, χ) — au bit près.
  4. Couverture : chaque acier, chaque courbe, chaque famille, chaque avertissement, chaque gouvernant, les bornes…
     sont effectivement atteints par les cas.
  5. Sélecteur : ses lignes (autres familles, triées par masse) sont repassées par l'écran du module.
  6. Format : JSON strict, normalisation (défauts, ValueError hors du domaine de l'écran), formules recalculées à
     partir de leurs seuls termes, toutes les clés documentées.
  7. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, pandas,
     reportlab, matplotlib, scipy, PIL, fitz — et n'y charge que la bibliothèque standard et numpy (le catalogue des
     profilés), sous Python 3.12 et sous 3.13 (celui de Pyodide) — résultats identiques.

    /usr/bin/python3.12 tests/test_choix_profil_moteur.py              (streamlit et pandas requis pour la parité)
    /usr/bin/python3.12 tests/test_choix_profil_moteur.py --mutations  (la campagne de mutations : chacune doit rougir)
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
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.choix_profil import calcul as CA  # noqa: E402
from acier.choix_profil import entree as EN  # noqa: E402
from acier.choix_profil import moteur as M  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def lire(*chemin):
    f = os.path.join(RACINE, *chemin)
    if not os.path.exists(f):
        return ""
    with open(f, encoding="utf-8") as fh:
        return fh.read()


ECART = {"n": 0, "bit": 0, "max": 0.0}


def egal(a, b):
    """Égalité STRICTE (au bit près) ; NaN de pandas et None se valent ; le compte est tenu."""
    na = a is None or (isinstance(a, float) and math.isnan(a))
    nb_ = b is None or (isinstance(b, float) and math.isnan(b))
    if na or nb_:
        return na and nb_
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        ECART["n"] += 1
        if a == b:
            ECART["bit"] += 1
            return True
        ECART["max"] = max(ECART["max"], abs(a - b) / max(abs(a), abs(b)))
        return False
    return a == b


def proche(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


# ------------------------------------------------------------------ les cas, au format du moteur
def el(nom="P", **k):
    e = EN.nouvel_element(nom)
    e.update(k)
    return EN.normaliser(e)


CAS = [
    ("défauts de l'écran (aucun effort)", el()),
    ("M_y = 45, V_z = 30, HEA, S235", el(My=45.0, Vz=30.0)),
    ("même chose en S275", el(My=45.0, Vz=30.0, acier="S275")),
    ("même chose en S355", el(My=45.0, Vz=30.0, acier="S355")),
    ("M_y et V_z négatifs (valeurs absolues)", el(My=-120.0, Vz=-80.0)),
    ("IPE seuls, M_y = 5 000 kN·m : aucun ne tient", el(familles=["IPE"], My=5000.0, Vz=100.0)),
    ("aucune famille cochée : toutes les familles", el(familles=[], My=300.0, Vz=200.0)),
    ("IPE + HEA + HEB : l'optimal du module n'est pas le plus léger", el(familles=["HEA", "HEB", "IPE"], My=150.0, Vz=60.0)),
    ("V_z gouverne (fort cisaillement, faible moment)", el(familles=["IPE"], My=5.0, Vz=400.0)),
    ("N + M_y + V_z, flambement y et z, courbe b", el(familles=["HEB"], My=60.0, Vz=40.0, avec_N=True, N=600.0, flambement=True,
                                                     Lcr_y=6.0, Lcr_z=3.0)),
    ("flambement gouverne (poteau élancé, courbe c)", el(familles=["HEA"], My=5.0, avec_N=True, N=900.0, flambement=True, Lcr_y=8.0,
                                                        Lcr_z=8.0, courbe="c")),
    ("flambement courbe a0", el(familles=["HEB"], avec_N=True, N=800.0, flambement=True, Lcr_y=5.0, Lcr_z=5.0, courbe="a0")),
    ("flambement courbe a", el(familles=["HEM"], avec_N=True, N=1500.0, flambement=True, Lcr_y=7.0, Lcr_z=3.5, courbe="a")),
    ("flambement courbe d", el(familles=["HD"], avec_N=True, N=3000.0, My=100.0, flambement=True, Lcr_y=9.0, Lcr_z=4.5, courbe="d")),
    ("flambement selon y seul (L_cr,z = 0)", el(familles=["IPE"], avec_N=True, N=250.0, My=20.0, flambement=True, Lcr_y=5.0)),
    ("flambement selon z seul (L_cr,y = 0)", el(familles=["IPE"], avec_N=True, N=250.0, My=20.0, flambement=True, Lcr_z=2.5)),
    ("flambement coché, L_cr nuls : impossible", el(avec_N=True, N=300.0, My=30.0, flambement=True)),
    ("flambement coché, N = 0 : rien à calculer", el(My=30.0, flambement=True, Lcr_y=4.0, Lcr_z=4.0)),
    ("flambement coché, N en traction", el(My=30.0, avec_N=True, N=-150.0, flambement=True, Lcr_y=4.0, Lcr_z=4.0)),
    ("traction N = -300 : η_N retranché de la flexion", el(familles=["IPE"], My=80.0, Vz=20.0, avec_N=True, N=-300.0)),
    ("M_z ajouté (flexion déviée)", el(familles=["HEB"], My=80.0, Vz=40.0, avec_Mz=True, Mz=25.0)),
    ("V_y ajouté : A_v,y absent de la base lue", el(My=40.0, Vz=20.0, avec_Vy=True, Vy=150.0)),
    ("T ajouté : la torsion n'entre dans aucun calcul", el(My=40.0, Vz=20.0, avec_T=True, T=12.0)),
    ("déversement coché (L_LT saisi, non calculé)", el(My=40.0, Vz=20.0, deversement=True, L_LT=5.0)),
    ("tous les efforts, tous les contrôles", el(familles=["HEA", "HEB"], My=90.0, Vz=70.0, avec_N=True, N=400.0, avec_Mz=True, Mz=15.0,
                                                avec_Vy=True, Vy=30.0, avec_T=True, T=3.0, flambement=True, Lcr_y=5.0, Lcr_z=2.5, courbe="c",
                                                deversement=True, L_LT=4.0, gamma_M0=1.05, gamma_M1=1.1)),
    ("I_y min = 5 000 cm⁴ : les petits profilés sortent", el(My=60.0, Vz=40.0, Iy_min=5000.0)),
    ("I_y min au-delà du plus fort profilé (10⁷ cm⁴) : aucun profilé", el(My=60.0, Iy_min=1e7)),
    ("γ_M0 = 1,1 et γ_M1 = 1,15", el(familles=["IPE"], My=100.0, Vz=50.0, avec_N=True, N=100.0, gamma_M0=1.1, gamma_M1=1.15,
                                      flambement=True, Lcr_y=6.0, Lcr_z=3.0)),
    ("bornes de l'écran : γ_M0 = 0,8 et γ_M1 = 1,5", el(My=70.0, avec_N=True, N=200.0, gamma_M0=0.8, gamma_M1=1.5, flambement=True,
                                                         Lcr_y=4.0, Lcr_z=4.0)),
    ("profilé choisi (HEA 300), non optimal", el(My=45.0, Vz=30.0, profil="HEA 300")),
    ("profilé choisi qui ne tient pas (HEA 100)", el(My=45.0, Vz=30.0, profil="HEA 100")),
    ("profilé choisi hors des familles cochées (IPE 300 avec HEA)", el(My=45.0, Vz=30.0, profil="IPE 300")),
    ("profilé choisi sorti par I_y min", el(My=45.0, Vz=30.0, profil="HEA 120", Iy_min=1000.0)),
    ("I_y min = I_y(HEA 200) exactement : HEA 200 reste (le filtre est « I_y < I_y min »)", el(My=45.0, Vz=30.0, Iy_min=CA.profil("HEA 200")["Iy"])),
    ("efforts infimes (M_y = 10⁻⁵ kN·m) : utilisations arrondies à 0, l'ordre du catalogue", el(familles=["HEA", "IPE"], My=1e-5)),
    ("efforts saisis puis décochés (N, M_z, V_y, T gardés), flambement décoché (L_cr gardés)",
     el(My=45.0, Vz=30.0, N=500.0, Mz=20.0, Vy=40.0, T=5.0, Lcr_y=6.0, Lcr_z=3.0, courbe="d", L_LT=4.0)),
    ("HD 320x300.0 choisi (paroi de 48 mm) : f_y non réduit par l'écran", el(familles=["HD"], My=250.0, Vz=150.0, profil="HD 320x300.0")),
    ("HL 920x970 en S355 (paroi de 89,9 mm, hors tableau 3.1)", el(familles=["HL"], My=2500.0, Vz=900.0, acier="S355", profil="HL 920x970")),
]
for fam in EN.familles_disponibles():   # chaque famille seule
    CAS.append((f"famille {fam} seule, M_y = 250, V_z = 150", el(familles=[fam], My=250.0, Vz=150.0)))


def aleatoire(rng, k):
    """Un élément tiré au sort (graine fixe) : familles, acier, efforts, γ, instabilités, filtre, choix."""
    fams = rng.sample(EN.familles_disponibles(), rng.choice((0, 1, 1, 2, 3)))
    e = dict(familles=sorted(fams), acier=rng.choice(EN.ACIERS), My=round(rng.uniform(-400, 900), 2), Vz=round(rng.uniform(-300, 600), 1),
             gamma_M0=rng.choice((1.0, 1.0, 1.05, 1.1)), gamma_M1=rng.choice((1.0, 1.1)))
    if rng.random() < 0.5:
        e.update(avec_N=True, N=round(rng.uniform(-500, 4000), 1))
    if rng.random() < 0.3:
        e.update(avec_Mz=True, Mz=round(rng.uniform(-80, 80), 2))
    if rng.random() < 0.2:
        e.update(avec_Vy=True, Vy=round(rng.uniform(0, 200), 1))
    if rng.random() < 0.5:
        e.update(flambement=True, Lcr_y=rng.choice((0.0, 3.0, 4.5, 6.0, 9.0)), Lcr_z=rng.choice((0.0, 2.0, 3.0, 4.5)),
                 courbe=rng.choice([c for c, _, _ in EN.COURBES]))
    if rng.random() < 0.2:
        e.update(Iy_min=rng.choice((500.0, 2000.0, 20000.0)))
    if rng.random() < 0.15:
        e.update(deversement=True, L_LT=rng.choice((0.0, 3.0, 6.0)))
    if rng.random() < 0.25:
        e.update(profil=rng.choice(CA.noms()))
    return el(f"Aléa {k}", **e)


RNG = random.Random(20261107)
CAS += [(f"tirage {k:02d} (graine 20261107)", aleatoire(RNG, k)) for k in range(1, 25)]


def cas(debut):
    return next(e for d, e in CAS if d.startswith(debut))


# ------------------------------------------------------------------ 1. données
print("=== 1. Données : le catalogue, les familles, les défauts de l'écran ===")
try:
    logging.disable(logging.WARNING)
    import streamlit as st  # noqa: F401,E402
    import modules.choix_profile as CP  # noqa: E402
    import modules.navigation as NAV  # noqa: E402
except ImportError as e:  # pragma: no cover — la parité EXIGE le module
    CP = None
    chk("streamlit, pandas et modules/choix_profile.py se chargent (indispensables à la parité)", False, e)

# les libellés des widgets de l'écran ↔ les clés de l'élément
W = {"familles": "Types de profilés à inclure :", "My": "My,Ed [kN·m] (défaut)", "Vz": "Vz,Ed [kN] (défaut)", "acier": "Acier",
     "avec_N": "Ajouter N,Ed (compression +)", "avec_Mz": "Ajouter Mz,Ed", "avec_Vy": "Ajouter Vy,Ed", "avec_T": "Ajouter T,Ed (torsion)",
     "N": "N,Ed [kN]", "Mz": "Mz,Ed [kN·m]", "Vy": "Vy,Ed [kN]", "T": "T,Ed [kN·m]", "gamma_M0": "γM0", "gamma_M1": "γM1",
     "Iy_min": "Iy min. [cm⁴] (optionnel)", "flambement": "Vérifier flambement (si N>0)",
     "deversement": "Vérifier déversement (LTB) — nécessite It/Iw/Iz", "Lcr_y": "Lcr,y [m]", "Lcr_z": "Lcr,z [m]",
     "courbe": "Courbe flambement (α)", "L_LT": "L_LT [m] (longueur non contreventée)", "profil": "Sélectionner un profilé :"}
TABLE = "Afficher tous les profilés ✓/✗"


class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran afficherait ; chaque widget rend la valeur demandée (``vals``, par
    libellé) ou sa valeur par défaut, comme Streamlit à la première exécution."""

    def __init__(self, vals=None):
        self.vals = vals or {}
        self.ev = []
        self.widgets = {}
        self.lus = set()

    def _w(self, sorte, label, defaut, **info):
        self.widgets[label] = dict(sorte=sorte, defaut=defaut, **info)
        self.ev.append(("w", label))
        if label in self.vals:
            self.lus.add(label)
            return self.vals[label]
        return defaut

    def title(self, *a, **k):
        pass

    def columns(self, spec, **_):
        return [contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def expander(self, label, expanded=False, **_):
        self.ev.append(("exp", label))
        return contextlib.nullcontext()

    def multiselect(self, label, options, default=None, **_):
        return self._w("multiselect", label, list(default or []), options=list(options))

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **_):
        return self._w("number", label, value, mini=min_value, maxi=max_value)

    def selectbox(self, label, options, index=0, **_):
        options = list(options)
        self.ev.append(("select", label, options, index))
        return self._w("select", label, options[index], options=options, index=index)

    def checkbox(self, label, value=False, **_):
        return self._w("checkbox", label, value)

    def markdown(self, t, **_):
        self.ev.append(("md", t))

    def subheader(self, t, **_):
        self.ev.append(("sub", t))

    def caption(self, t, **_):
        self.ev.append(("caption", t))

    def warning(self, t, **_):
        self.ev.append(("warning", t))

    def info(self, t, **_):
        self.ev.append(("info", t))

    def error(self, t, **_):
        self.ev.append(("error", t))

    def latex(self, t, **_):
        self.ev.append(("latex", t))

    def dataframe(self, d, **_):
        self.ev.append(("df", d))


def vers_widgets(e):
    """L'élément du moteur, saisi dans les widgets de l'écran."""
    v = {W[k]: e[k] for k in ("familles", "My", "Vz", "acier", "avec_N", "avec_Mz", "avec_Vy", "avec_T", "N", "Mz", "Vy", "T",
                              "gamma_M0", "gamma_M1", "Iy_min", "flambement", "deversement", "Lcr_y", "Lcr_z", "L_LT")}
    v[W["courbe"]] = EN.LIBELLE_COURBE[e["courbe"]]
    if e["profil"] is not None:
        v[W["profil"]] = e["profil"]
    return v


def ecran(e, choix=None):
    """L'écran du module exécuté sur l'élément ``e`` ; ``choix`` : le profilé choisi dans le menu (sinon celui de
    ``e``, s'il y est ; sinon le défaut de l'écran)."""
    vals = vers_widgets(e)
    faux = FauxSt(vals)
    menu = {}

    def selectbox(label, options, index=0, **k):
        options = list(options)
        if label == W["profil"]:
            menu.update(options=options, index=index)
            voulu = choix or e["profil"]
            faux.vals[label] = voulu if voulu in options else options[index]
        return FauxSt.selectbox(faux, label, options, index, **k)
    faux.selectbox = selectbox
    vrai_st, vrai_nav = CP.st, NAV.bouton_accueil
    CP.st, NAV.bouton_accueil = faux, (lambda *a, **k: None)
    try:
        CP.show()
    finally:
        CP.st, NAV.bouton_accueil = vrai_st, vrai_nav
    return faux, menu


if CP is not None:
    chk("profils() = load_profiles() du module (les mêmes profilés, champs et valeurs, dans le même ordre)",
        list(CA.profils().items()) == list(CP.load_profiles().items()), len(CA.profils()))
    chk("les familles chargées sont celles que le module passe au catalogue",
        "CAT.legacy((" + ", ".join(f'"{f}"' for f in CA.FAMILLES_CHARGEES) + "))" in lire("modules", "choix_profile.py"))
    copie = CA.profils()
    copie["HEA 200"]["Wel_y"] = 1.0
    chk("profils() rend une copie : la modifier ne change pas le calcul",
        M.verifier(el(My=10.0))["lignes"] == M.verifier(el(My=10.0))["lignes"] and CA.profil("HEA 200")["Wel_y"] != 1.0)
    # les défauts : une exécution sans rien saisir, puis une avec chaque case cochée (les champs cachés apparaissent)
    f0, _ = ecran(EN.nouvel_element())
    tout = {W[k]: True for k in ("avec_N", "avec_Mz", "avec_Vy", "avec_T", "flambement", "deversement")}
    f1 = FauxSt(tout)
    CP.st, vrai_nav = f1, NAV.bouton_accueil
    NAV.bouton_accueil = lambda *a, **k: None
    try:
        CP.show()
    finally:
        CP.st, NAV.bouton_accueil = st, vrai_nav
    wd = {**f1.widgets, **f0.widgets}
    defaut = EN.nouvel_element()
    lus = {k: wd[W[k]]["defaut"] for k in W if k != "profil"}
    lus["courbe"] = {t: c for c, _, t in EN.COURBES}[lus["courbe"]]
    attendu = {k: defaut[k] for k in lus}
    chk("nouvel_element() = les valeurs des widgets à l'ouverture de l'écran (familles HEA, S235, γ 1,0, courbe b, cases décochées…)",
        lus == attendu, {k: (lus[k], attendu[k]) for k in lus if lus[k] != attendu[k]})
    chk("les familles proposées sont celles du menu de l'écran", wd[W["familles"]]["options"] == EN.familles_disponibles(),
        wd[W["familles"]]["options"])
    chk("les aciers et les courbes (libellés et α) sont ceux des menus de l'écran",
        wd[W["acier"]]["options"] == list(EN.ACIERS) and wd[W["courbe"]]["options"] == [t for _, _, t in EN.COURBES]
        and all(f"\"{t}\":{a}".replace(" ", "") in lire("modules", "choix_profile.py").replace(" ", "") for _, a, t in EN.COURBES))
    bornes = {k: (wd[W[k]]["mini"], wd[W[k]]["maxi"]) for k in EN.REELS}
    chk("les bornes de l'écran sont celles du moteur : γ dans [0,8 ; 1,5], I_y min, L_cr, L_LT ≥ 0, efforts libres",
        all(bornes[k] == (EN.GAMMA_MIN, EN.GAMMA_MAX) for k in ("gamma_M0", "gamma_M1"))
        and all(bornes[k] == (0.0, None) for k in EN.POSITIFS) and all(bornes[k] == (None, None) for k in ("My", "Vz", "N", "Mz", "Vy", "T")),
        bornes)
    chk("le tableau de tous les profilés est affiché par défaut (la case de l'écran est cochée)", wd[TABLE]["defaut"] is True)

# ------------------------------------------------------------------ 1 bis. les fonctions de l'écran, une à une
print("\n=== 1 bis. Les fonctions recopiées contre celles de l'écran, sur des entrées construites (bords compris) ===")
if CP is not None:
    # load_profiles sur des enregistrements construits : W_el nul, négatif, absent, « None », texte ; « Poids » avant
    # « masse » ; « Iv » à défaut de « Iy » ; « Wely » avant « Wel » ; A_v,y présent ; b manquant ; un nombre illisible
    BRUT = {
        "W0": {"type": "X", "h": 100, "Wel": 0, "A": 10, "Avz": 5},
        "Wneg": {"type": "X", "h": 100, "Wel": -3.0, "A": 10},
        "Wnone": {"type": "X", "h": 100, "Wel": None, "A": 10},
        "Wtxt": {"type": "X", "h": 100, "Wel": "None", "A": 10},
        "Wvide": {"type": "X", "h": 100, "Wel": "", "A": 10},
        "Poids": {"type": "Y", "h": "120", "Wel": "12.5", "Poids": 10.0, "masse": 12.0, "A": "8,5", "Iv": 99.0, "Avy": 3.0, "Avz": 2.0},
        "Wely": {"type": "Y", "h": 80, "Wely": 7.0, "Wel": 9.0, "Wel_y": 11.0, "Welz": 1.0, "Wel_z": 2.0, "masse": 4.0, "Iy": 50.0, "Iv": 60.0,
                 "It": "x", "Iw": 3},
        "sans_b": {"h": None, "Wel": 1.0},
    }
    import acier.profiles.catalogue as CAT_
    vrai_legacy = CAT_.legacy
    CAT_.legacy = lambda familles=None: copy.deepcopy(BRUT)
    try:
        CP.load_profiles.clear()
        lu_module = CP.load_profiles()
    finally:
        CAT_.legacy = vrai_legacy
        CP.load_profiles.clear()
    lu_moteur = CA.nettoyer(copy.deepcopy(BRUT))
    chk("load_profiles et nettoyer donnent la même chose sur des enregistrements construits (W_el ≤ 0 ou illisible écartés, "
        "« Poids » avant « masse », « Iv », « Wely », A_v,y lu s'il existe, b manquant → 0)",
        list(lu_module.items()) == list(lu_moteur.items()) and list(lu_moteur) == ["Poids", "Wely", "sans_b"]
        and lu_moteur["Poids"]["Poids"] == 10.0 and lu_moteur["Wely"]["Wel_y"] == 7.0 and lu_moteur["Poids"]["Avy"] == 3.0,
        (list(lu_module), list(lu_moteur)))
    # résistances, taux et flambement : les fonctions de l'écran et celles du moteur, sur 3 000 tirages (bords compris)
    rng = random.Random(7)
    diff = []
    for k in range(3000):
        prof = {c: rng.choice((None, 0.0, round(rng.uniform(0.5, 900.0), 3))) for c in ("A", "Wel_y", "Wel_z", "Avy", "Avz")}
        fy_ = rng.choice((235, 275, 355))
        g0 = rng.choice((0.8, 1.0, 1.05, 1.5))
        ra, rb = CP.ec3_section_resistance(prof, fy=fy_, gamma_M0=g0), CA.resistances_section(prof, fy=fy_, gamma_M0=g0)
        ef_ = [rng.choice((0.0, round(rng.uniform(-900, 900), 2))) for _ in range(5)]
        ta, tb = CP.utilisation_ratio_section(*ef_, ra), CA.taux_section(*ef_, rb)
        A_, I_ = rng.choice((None, round(rng.uniform(1, 500), 2))), rng.choice((None, round(rng.uniform(1, 2e5), 1)))
        L_, al = rng.choice((0.0, -1.0, round(rng.uniform(0.1, 20), 2))), rng.choice((0.13, 0.21, 0.34, 0.49, 0.76))
        fa, fb = CP.ec3_buckling_chi(A_, I_, L_, fy=fy_, alpha=al), CA.flambement_chi(A_, I_, L_, fy=fy_, alpha=al)
        if fb is not None:
            fb = {x: fb[x] for x in ("chi", "lambdabar", "Ncr_N")}
        if ra != rb or ta != tb or fa != fb:
            diff.append((k, prof, ef_))
    chk("ec3_section_resistance, utilisation_ratio_section, ec3_buckling_chi : 3 000 tirages, mêmes résultats au bit près "
        "(A_v,y présent, données nulles ou absentes, efforts négatifs, L_cr ≤ 0)", not diff, diff[:2])
    chk("fmt_no_trailing_zeros : le même texte (entiers, décimales, None, texte)",
        all(CP.fmt_no_trailing_zeros(x) == CA.fmt_sans_zeros(x) for x in (None, 0, 3.0, 26.2455, 1294070.0, 0.1 + 0.2, -4.5, "abc", 1e-9)))

# ------------------------------------------------------------------ 2. à la main
print("\n=== 2. Contrôles à la main (EC3 écrit en N et en mm) et valeurs publiées de χ ===")
P = CA.profil("HEA 200")
fy = 235
NRd = P["A"] * 100 * fy / 1.0                                   # N
MyRd = P["Wel_y"] * 1000 * fy / 1.0                             # N·mm
VzRd = P["Avz"] * 100 * fy / math.sqrt(3)                       # N
r = M.verifier(el(My=45.0, Vz=30.0, profil="HEA 200"))
R = r["profil"]["resistances"]
chk(f"HEA 200 S235 : N_Rd = A·f_y = {P['A']:.3f} cm² × 235 = {NRd / 1e3:.1f} kN ; M_y,Rd = W_el,y·f_y = {MyRd / 1e6:.2f} kN·m ; "
    f"V_z,Rd = A_v,z·f_y/√3 = {VzRd / 1e3:.1f} kN",
    proche(R["N_Rd"], NRd / 1e3) and proche(R["My_Rd"], MyRd / 1e6) and proche(R["Vz_Rd"], VzRd / 1e3), R)
eta = max(45e6 / MyRd, 30e3 / VzRd)
chk(f"HEA 200 : η = max(45 / M_y,Rd ; 30 / V_z,Rd) = {eta:.4f}, la flexion gouverne", proche(r["taux_max"], eta)
    and r["gouvernant"]["cle"] == "NM", (r["taux_max"], r["gouvernant"]))
# flambement à la main : HEB 200 S235, L_cr = 4 m selon z, courbe c
P2 = CA.profil("HEB 200")
Ncr = math.pi ** 2 * 210000 * P2["Iz"] * 1e4 / 4000 ** 2
lam = math.sqrt(P2["A"] * 100 * 235 / Ncr)
phi = 0.5 * (1 + 0.49 * (lam - 0.2) + lam ** 2)
chi = 1 / (phi + math.sqrt(phi ** 2 - lam ** 2))
r2 = M.verifier(el(familles=["HEB"], avec_N=True, N=500.0, flambement=True, Lcr_z=4.0, courbe="c", profil="HEB 200"))
ax = r2["profil"]["axes"][0]
chk(f"HEB 200, L_cr,z = 4 m, courbe c : N_cr = {Ncr / 1e3:.1f} kN, λ̄ = {lam:.4f}, χ = {chi:.4f}, N_b,Rd = {chi * P2['A'] * 23.5:.1f} kN",
    ax["axe"] == "z" and proche(ax["Ncr"], Ncr / 1e3) and proche(ax["lambdabar"], lam) and proche(ax["chi"], chi)
    and proche(r2["profil"]["Nb_Rd"], chi * P2["A"] * 100 * 235 / 1e3) and proche(r2["taux_max"], max(500e3 / (chi * P2["A"] * 100 * 235),
                                                                                                         r2["profil"]["ratios"]["util"])), ax)
# les valeurs publiées : EN 1993-1-1 §6.3.1.2, χ à λ̄ = 1,0 (tableaux des courbes : a 0,6656 ; b 0,5970 ; c 0,5399 ; d 0,4671)
PUBLIE = {"a": 0.6656, "b": 0.5970, "c": 0.5399, "d": 0.4671}
lus = {c: CA.chi_courbe(1.0, EN.ALPHA[c])[1] for c in PUBLIE}
chk("χ(λ̄ = 1,0) = valeurs publiées des courbes a, b, c, d à 10⁻⁴ (0,6656 · 0,5970 · 0,5399 · 0,4671)",
    all(abs(lus[c] - v) < 5e-5 for c, v in PUBLIE.items()), lus)
# f_y selon l'épaisseur (EN 1993-1-1 tableau 3.1) : l'écran ne réduit pas ; le moteur le DIT, avec l'écart
av_hd = {a["code"]: a["message"] for a in M.verifier(el(familles=["HD"], My=250.0, profil="HD 320x300.0"))["avertissements"]}
av_hd355 = {a["code"]: a["message"] for a in M.verifier(el(familles=["HD"], My=250.0, acier="S355", profil="HD 320x300.0"))["avertissements"]}
av_hl = {a["code"]: a["message"] for a in M.verifier(el(familles=["HL"], My=250.0, profil="HL 920x970"))["avertissements"]}
av_hea = {a["code"] for a in M.verifier(el(My=250.0, profil="HEA 200"))["avertissements"]}
chk("paroi > 40 mm : f_y non réduit DIT — HD 320x300.0 (t_f = 48 mm) : 235 au lieu de 215 N/mm² (+9,3 %), en S355 355 au lieu de 335 (+6 %) ; "
    "HL 920x970 (89,9 mm) hors du tableau 3.1 ; HEA 200 : rien",
    "215 N/mm²" in av_hd.get("fy_epaisseur", "") and "9,3 %" in av_hd["fy_epaisseur"] and "335 N/mm²" in av_hd355.get("fy_epaisseur", "")
    and "6 %" in av_hd355["fy_epaisseur"] and "hors du tableau 3.1" in av_hl.get("fy_epaisseur", "") and "fy_epaisseur" not in av_hea
    and CA.profil("HD 320x300.0")["tf"] == 48.0, (av_hd, av_hd355, av_hl))
# la classe des parois (EN 1993-1-1 tableau 5.2) : l'écran ne classe pas (D5) ; le moteur le DIT, sans rien recalculer
def _cl4(**k):
    return [a["message"] for a in M.verifier(el(**k))["avertissements"] if a["code"] == "classe_4"]


def _ame_c_t(nom):
    p_ = CA.profil(nom)
    return (p_["h"] - 2 * p_["tf"] - 2 * p_["r"]) / p_["tw"]


ames4 = [n for n in CA.noms() if _ame_c_t(n) > 42.0]
chk("classe 4 DITE : semelle de l'HP 305x79 en S355 (c/t_f = 12,0 > 14ε = 11,4), pas en S235 ; âme de l'IPE 600 comprimé (42,8 > 42ε), "
    "pas sans compression ; l'IPE 500 comprimé (41,8) et l'HEA 200 : rien ; 28 âmes de classe 4 en compression en S235 (dont IPE 550, IPE 600)",
    any("c/t_f = 12,0 > 14ε = 11,4" in x for x in _cl4(familles=["HP"], My=50.0, acier="S355", profil="HP 305x79"))
    and not _cl4(familles=["HP"], My=50.0, profil="HP 305x79")
    and any("c/t_w = 42,8 > 42ε = 42,0" in x for x in _cl4(familles=["IPE"], avec_N=True, N=100.0, profil="IPE 600"))
    and not _cl4(familles=["IPE"], My=100.0, profil="IPE 600") and not _cl4(familles=["IPE"], avec_N=True, N=100.0, profil="IPE 500")
    and not _cl4(avec_N=True, N=100.0, My=50.0, profil="HEA 200") and len(ames4) == 28 and {"IPE 550", "IPE 600"} <= set(ames4),
    (_cl4(familles=["HP"], My=50.0, acier="S355", profil="HP 305x79"), len(ames4)))
r600 = M.verifier(el(familles=["IPE"], avec_N=True, N=100.0, profil="IPE 600"))
chk("classe 4 : DITE, jamais recalculée — le taux du retenu reste celui de sa ligne du tableau de l'écran (parité entière)",
    r600["taux_max"] == next(li["taux"] for li in r600["lignes"] if li["nom"] == "IPE 600"))
chk("χ = 1 sous λ̄ = 0,2 (palier) pour les cinq courbes", all(CA.chi_courbe(lb, a)[1] == 1.0 for a in EN.ALPHA.values() for lb in (0.0, 0.1, 0.2)))

# ------------------------------------------------------------------ 3. parité
LATEX_RES = {"N_Rd": "N", "My_Rd": "My", "Mz_Rd": "Mz", "Vz_Rd": "Vz", "Vy_Rd": "Vy"}


def ecran_attendu(res):
    """Ce que l'écran du module DOIT afficher pour la sortie ``res`` du moteur : les textes et les formules LaTeX
    reconstruits à partir des nombres du moteur, avec les formats du module (mot pour mot)."""
    e = res["element"]
    ev = [("md", "### Efforts (ELU)"), ("md", "### Paramètres EC3")]
    if e["flambement"]:
        ev.append(("md", "**Flambement**"))
    if e["deversement"]:
        ev += [("md", "**Déversement (LTB)**"), ("caption", M.LEGENDE_LTB)]
    ev.append(("sub", "📌 Profilé optimal :"))
    if not res["lignes"]:
        return ev + [("warning", M.TEXTES_MODULE["aucun_profil"])]
    ev.append(("md", f"### {res['retenu']}"))
    ev.append(("sub", "Formules EC3 — vérification section"))
    p = CA.profil(res["retenu"])
    R, rt = res["profil"]["resistances"], res["profil"]["ratios"]
    fy, g0 = res["acier"]["fy"], e["gamma_M0"]
    if R["N_Rd"] is not None:
        ev.append(("latex", rf"N_{{Rd}} = \frac{{A f_y}}{{\gamma_{{M0}}}} = \frac{{{p['A']:.2f}\times 10^2 \cdot {fy}}}{{{g0:.2f}}}"
                            rf" = {R['N_Rd']:.1f}\ \text{{kN}}"))
    else:
        ev.append(("info", M.TEXTES_MODULE["N_Rd_absent"]))
    if R["My_Rd"] is not None:
        ev.append(("latex", rf"M_{{y,Rd}} = \frac{{W_{{el,y}} f_y}}{{\gamma_{{M0}}}} = \frac{{{p['Wel_y']:.2f}\times 10^3 \cdot {fy}}}{{{g0:.2f}}}"
                            rf" = {R['My_Rd']:.1f}\ \text{{kN·m}}"))
    else:
        ev.append(("info", M.TEXTES_MODULE["My_Rd_absent"]))
    if e["avec_Mz"]:
        if R["Mz_Rd"] is not None:
            ev.append(("latex", rf"M_{{z,Rd}} = \frac{{W_{{el,z}} f_y}}{{\gamma_{{M0}}}} = \frac{{{p['Wel_z']:.2f}\times 10^3 \cdot {fy}}}{{{g0:.2f}}}"
                                rf" = {R['Mz_Rd']:.1f}\ \text{{kN·m}}"))
        else:
            ev.append(("warning", M.TEXTES_MODULE["Mz_Rd_absent"]))
    if R["Vz_Rd"] is not None:
        ev.append(("latex", rf"V_{{z,Rd}} = \frac{{A_{{vz}} f_y}}{{\sqrt{{3}}\gamma_{{M0}}}} = \frac{{{p['Avz']:.2f}\times 10^2 \cdot {fy}}}"
                            rf"{{\sqrt{{3}}\cdot {g0:.2f}}} = {R['Vz_Rd']:.1f}\ \text{{kN}}"))
    else:
        ev.append(("info", M.TEXTES_MODULE["Vz_Rd_absent"]))
    if e["avec_Vy"]:
        if R["Vy_Rd"] is not None:
            ev.append(("latex", rf"V_{{y,Rd}} = \frac{{A_{{vy}} f_y}}{{\sqrt{{3}}\gamma_{{M0}}}} = {R['Vy_Rd']:.1f}\ \text{{kN}}"))
        else:
            ev.append(("warning", M.TEXTES_MODULE["Vy_Rd_absent"]))
    ev.append(("sub", "Utilisations (ratios)"))
    ev.append(("latex", rf"\eta_N = \frac{{N_{{Ed}}}}{{N_{{Rd}}}} = {rt['rN']:.3f}"))
    ev.append(("latex", rf"\eta_{{My}} = \frac{{|M_{{y,Ed}}|}}{{M_{{y,Rd}}}} = {rt['rMy']:.3f}"))
    if e["avec_Mz"]:
        ev.append(("latex", rf"\eta_{{Mz}} = \frac{{|M_{{z,Ed}}|}}{{M_{{z,Rd}}}} = {rt['rMz']:.3f}"))
    if e["avec_Vy"]:
        ev.append(("latex", rf"\eta_{{Vy}} = \frac{{|V_{{y,Ed}}|}}{{V_{{y,Rd}}}} = {rt['rVy']:.3f}"))
    ev.append(("latex", rf"\eta_{{Vz}} = \frac{{|V_{{z,Ed}}|}}{{V_{{z,Rd}}}} = {rt['rVz']:.3f}"))
    ev.append(("latex", rf"\eta_{{NM}} = \eta_N + \eta_{{My}} + \eta_{{Mz}} = {rt['util_NM']:.3f}"))
    ev.append(("latex", rf"\eta_{{section}} = \max(\eta_{{NM}},\eta_{{Vy}},\eta_{{Vz}}) = {rt['util']:.3f}"))
    if res["flambement"]["calcule"]:
        ev.append(("sub", "Instabilité — flambement (si données dispo)"))
        axes = {a["axe"]: a for a in res["profil"]["axes"]}
        if not axes:
            ev.append(("warning", M.TEXTES_MODULE["flambement_impossible"]))
        else:
            for k in ("y", "z"):
                if k in axes:
                    ev.append(("latex", rf"\chi_{k} = {axes[k]['chi']:.3f}\quad;\quad \bar\lambda_{k} = {axes[k]['lambdabar']:.3f}"))
            ev.append(("caption", M.LEGENDE_FLAMBEMENT))
    if e["deversement"]:
        ev += [("sub", "Instabilité — déversement (LTB)"), ("info", M.TEXTES_MODULE["deversement_non_calcule"])]
    return ev


COLONNES = (("Utilisation [%]", "utilisation"), ("h [mm]", "h"), ("Wel_y [cm³]", "Wel_y"), ("Wel_z [cm³]", "Wel_z"), ("Avy [cm²]", "Avy"),
            ("Avz [cm²]", "Avz"), ("A [cm²]", "A"), ("Iy [cm⁴]", "Iy"), ("Iz [cm⁴]", "Iz"), ("Poids [kg/m]", "masse"),
            ("u_section [%]", "u_section"), ("u_flamb [%]", "u_flamb"))
COULEURS = {"optimal": "#b7f7c1", "tient": "#eafaf0", "ne tient pas": "#ffeaea"}


def taux_module(e, nom):
    """Le taux d'un profilé par les FONCTIONS du module (le corps de sa boucle, appelé ici) : (section, flambement)."""
    prof = CP.load_profiles()[nom]
    fy_ = int(e["acier"][1:])
    ef = M._efforts(e)
    fl = M._flambement(e)
    res_ = CP.ec3_section_resistance(prof, fy=fy_, gamma_M0=e["gamma_M0"])
    rat = CP.utilisation_ratio_section(ef["N"], ef["My"], ef["Mz"], ef["Vy"], ef["Vz"], res_)
    ub = None
    if e["flambement"] and ef["N"] > 0:
        cands = []
        for I, L in ((prof.get("Iy"), fl["Lcr_y"]), (prof.get("Iz"), fl["Lcr_z"])):
            b = CP.ec3_buckling_chi(prof.get("A"), I, L, fy=fy_, alpha=fl["alpha"]) if (L > 0 and I) else None
            if b:
                cands.append(b["chi"] * (prof.get("A") * 1e2) * fy_ / e["gamma_M1"])
        if cands:
            ub = (ef["N"] * 1e3) / min(cands)
    return res_, rat, ub


def comparer(desc, e, res, ecarts):
    def cmp(nom, a, b):
        if not egal(a, b):
            ecarts.append(f"{nom} : moteur {a!r} ≠ module {b!r}")
    faux, menu = ecran(e)
    sortie = [x for x in faux.ev if x[0] in ("md", "sub", "latex", "info", "warning", "caption", "error")]
    attendu = ecran_attendu(res)
    if sortie != attendu:
        diff = next((i for i, (a, b) in enumerate(zip(sortie, attendu)) if a != b), min(len(sortie), len(attendu)))
        ecarts.append(f"écran, ligne {diff} : module {sortie[diff:diff + 1]} ≠ attendu {attendu[diff:diff + 1]} "
                      f"({len(sortie)} / {len(attendu)} lignes)")
    if not res["lignes"]:
        cmp("aucun tableau sans profilé", [x for x in faux.ev if x[0] == "df"], [])
        return faux
    # b) le menu du profilé : l'ordre du tableau, l'index par défaut = l'optimal du module
    cmp("options du menu (l'ordre du tableau)", [li["nom"] for li in res["lignes"]], menu.get("options"))
    idx = [li["nom"] for li in res["lignes"]].index(res["optimal"]) if res["optimal"] else 0
    cmp("index par défaut du menu (l'optimal du module)", idx, menu.get("index"))
    # a) le tableau : chaque colonne, au bit près ; les couleurs des lignes
    dfs = [x[1] for x in faux.ev if x[0] == "df"]
    sty, dims, props = dfs
    df = sty.data
    cmp("profilés du tableau, dans l'ordre", [li["nom"] for li in res["lignes"]], list(df.index))
    cmp("colonnes du tableau", [c for c, _ in COLONNES], list(df.columns))
    for i, li in enumerate(res["lignes"]):
        for col, k in COLONNES:
            v = df.iloc[i][col]
            v = None if v is None else (v.item() if hasattr(v, "item") else v)
            cmp(f"{li['nom']}.{col}", li[k], v)
    sty._compute()
    couleurs = [sty.ctx[(i, 0)][0][1] for i in range(len(res["lignes"]))]
    voulu = [COULEURS["optimal"] if li["nom"] == res["optimal"] else COULEURS["tient" if li["tient"] else "ne tient pas"] for li in res["lignes"]]
    cmp("couleurs des lignes (optimal, tient, ne tient pas)", voulu, couleurs)
    # c) la fiche : les deux tableaux, au format de l'écran
    cmp("fiche · dimensions", [[a, CA.fmt_sans_zeros(b)] for a, b in res["profil"]["dimensions"]],
        [list(x) for x in zip(dims["Dimensions"], dims["Valeur"])])
    cmp("fiche · propriétés", [[a, CA.fmt_sans_zeros(b)] for a, b in res["profil"]["proprietes"]],
        [list(x) for x in zip(props["Propriété"], props["Valeur"])])
    # d) les taux EXACTS, par les fonctions du module : chaque profilé du tableau, puis le retenu en détail
    for li in res["lignes"]:
        _, rat, ub = taux_module(e, li["nom"])
        cmp(f"{li['nom']} η section exact", li["taux_section"], rat["util"])
        cmp(f"{li['nom']} η flambement exact", li["taux_flambement"], ub)
        cmp(f"{li['nom']} η exact", li["taux"], max(rat["util"], ub) if ub is not None else rat["util"])
    res_, rat, ub = taux_module(e, res["retenu"])
    for k, kk, u in (("N_Rd", "N_Rd_N", 1e3), ("My_Rd", "My_Rd_Nmm", 1e6), ("Mz_Rd", "Mz_Rd_Nmm", 1e6), ("Vy_Rd", "Vy_Rd_N", 1e3), ("Vz_Rd", "Vz_Rd_N", 1e3)):
        cmp(f"retenu {k}", res["profil"]["resistances"][k], res_[kk] / u if res_[kk] is not None else None)
    for k in ("rN", "rMy", "rMz", "rVy", "rVz", "util_NM", "util"):
        cmp(f"retenu {k}", res["profil"]["ratios"][k], rat[k])
    prof = CP.load_profiles()[res["retenu"]]
    fl = M._flambement(e)
    for a in res["profil"]["axes"]:
        b = CP.ec3_buckling_chi(prof["A"], prof["I" + a["axe"]], a["Lcr"], fy=res["acier"]["fy"], alpha=fl["alpha"])
        for k, kk, u in (("chi", "chi", 1), ("lambdabar", "lambdabar", 1), ("Ncr", "Ncr_N", 1e3)):
            cmp(f"retenu flambement {a['axe']}.{k}", a[k], b[kk] / u)
    return faux


def faits(e, res):
    """Ce que le cas atteint (couverture), lu dans les sorties CONTRÔLÉES contre le module."""
    out = {f"acier {e['acier']}"} | {f"famille {f}" for f in res["familles"]} | {f"avertissement {a['code']}" for a in res["avertissements"]}
    if not e["familles"]:
        out.add("toutes les familles")
    if len(e["familles"]) > 1:
        out.add("plusieurs familles")
    if e["flambement"]:
        out.add(f"courbe {e['courbe']}")
    if res["profil"]:
        out.add(f"gouvernant {res['gouvernant']['cle']}")
        axes = "".join(a["axe"] for a in res["profil"]["axes"])
        if axes:
            out.add(f"flambement {axes}")
        if res["profil"]["taux_flambement"] is not None:
            out.add("η flambement calculé")
        if res["taux_max"] > 1:
            out.add("retenu > 100 %")
        if not res["retenu_auto"] and not res["choix_hors_tableau"]:
            out.add("profilé choisi")
        if res["profil"]["axes"] and any(a["lambdabar"] <= 0.2 for a in res["profil"]["axes"]):
            out.add("λ̄ ≤ 0,2")
    if res["optimal"] and res["plus_leger"] and res["optimal"] != res["plus_leger"]:
        out.add("optimal ≠ plus léger")
    if res["optimal"] and res["optimal"] == res["plus_leger"]:
        out.add("optimal = plus léger")
    if 0 < len(res["lignes"]) and e["Iy_min"] > 0:
        out.add("filtre I_y partiel")
    if any(li["tient"] and li["taux"] > 1 for li in res["lignes"]):
        out.add("borne : arrondi à 100,000 % retenu comme tenant")
    if any(li["taux"] == 1.0 for li in res["lignes"]):
        out.add("borne : η = 1 exactement")
    us = [li["utilisation"] for li in res["lignes"]]
    if len(us) != len(set(us)):
        out.add("égalités dans le tri")
    if e["gamma_M0"] != 1.0 or e["gamma_M1"] != 1.0:
        out.add("γ ≠ 1")
    if (e["My"] < 0 or e["Vz"] < 0):
        out.add("efforts négatifs")
    return out


if CP is not None:
    print("\n=== 3. Parité avec modules/choix_profile.py, cas par cas (écran exécuté) ===")
    # les bornes : M_y qui met HEA 200 à η = 1 exactement, puis juste au-dessus (l'écran arrondit à 100,000 % : il tient)
    base = el(familles=["HEA"], profil="HEA 200")
    MyRd_ = CP.ec3_section_resistance(CP.load_profiles()["HEA 200"], fy=235, gamma_M0=1.0)["My_Rd_Nmm"]
    x = MyRd_ / 1e6
    essais = [x] + [x := math.nextafter(x, math.inf) for _ in range(40)]
    x = MyRd_ / 1e6
    essais += [x := math.nextafter(x, -math.inf) for _ in range(40)]
    pile = next((m for m in essais if abs(m * 1e6) / MyRd_ == 1.0), None)
    if pile is not None:
        CAS.append(("borne exacte : η(HEA 200) = 1 (M_y construit avec le module)", el(familles=["HEA"], My=pile)))
    CAS.append(("borne de l'arrondi : η(HEA 200) = 1 + 2·10⁻⁶ s'écrit 100,000 % et « tient »",
                el(familles=["HEA"], My=MyRd_ / 1e6 * (1 + 2e-6))))
    COUV = set()
    for k, (desc, e) in enumerate(CAS):
        res = M.verifier(e)
        ecarts = []
        comparer(desc, e, res, ecarts)
        COUV |= faits(e, res)
        chk(f"cas {k + 1:02d} · {desc} — {res['n_profils']} profilés, retenu {res['retenu']}, η {res['taux_max'] if res['taux_max'] is None else round(res['taux_max'], 4)}",
            not ecarts, ecarts[:5])
    chk(f"{len(CAS)} cas comparés : au moins 50", len(CAS) >= 50)
    print(f"     {ECART['n']} comparaisons numériques : {ECART['bit']} identiques au bit près, écart relatif maximal {ECART['max']:.2e}")
    chk("toutes les comparaisons numériques sont identiques au bit près (c'est le même calcul)", ECART["n"] == ECART["bit"] and ECART["n"] > 10000,
        (ECART["n"], ECART["bit"]))

    print("\n=== 4. Couverture : ce que les cas atteignent réellement ===")
    ATTENDUS = ([f"acier {a}" for a in EN.ACIERS] + [f"courbe {c}" for c, _, _ in EN.COURBES] + [f"famille {f}" for f in EN.familles_disponibles()]
                + ["toutes les familles", "plusieurs familles", "flambement yz", "flambement y", "flambement z", "η flambement calculé",
                   "gouvernant NM", "gouvernant Vz", "gouvernant flambement", "retenu > 100 %", "profilé choisi", "optimal ≠ plus léger",
                   "optimal = plus léger", "filtre I_y partiel", "borne : arrondi à 100,000 % retenu comme tenant", "égalités dans le tri",
                   "γ ≠ 1", "efforts négatifs"]
                + [f"avertissement {c}" for c in ("deversement_non_calcule", "torsion_ignoree", "aucun_profil", "choix_hors_tableau", "fy_epaisseur",
                                                  "Vy_Rd_absent", "traction_retranchee", "flambement_sans_compression", "flambement_impossible",
                                                  "aucun_ne_tient", "sans_effort")])
    if pile is not None:
        ATTENDUS.append("borne : η = 1 exactement")
    manque = [a for a in ATTENDUS if a not in COUV]
    chk(f"les cas atteignent les {len(ATTENDUS)} situations attendues (aciers, courbes, familles, avertissements, gouvernants, bornes…)",
        not manque, manque)

    print("\n=== 5. Sélecteur : ses lignes repassées par l'écran du module ===")
    for desc, fams in (("M_y = 45", ["HEA", "IPE"]), ("N + M_y + V_z, flambement", None), ("I_y min = 5 000", ["HEB", "HEM"]),
                       ("tous les efforts", ["HD"])):
        e = cas(desc)
        cands = M.candidats(e, fams)
        e2 = dict(e, familles=fams if fams is not None else e["familles"], profil=None)
        res2 = M.verifier(e2)
        par_nom = {li["nom"]: li for li in res2["lignes"]}
        ecarts = []
        for c in cands:
            li = par_nom.get(c["nom"])
            if li is None:
                ecarts.append(f"{c['nom']} absent du tableau de l'écran")
                continue
            for k in ("taux", "utilisation", "masse"):
                if not egal(c[k], li[k]):
                    ecarts.append(f"{c['nom']}.{k}")
        masses = [c["masse"] for c in cands]
        forme = (masses == sorted(masses) and len(cands) == len(res2["lignes"]) and sum(c["retenu"] for c in cands) <= 1
                 and sum(c["plus_leger"] for c in cands) == (1 if any(c["taux"] <= 1 for c in cands) else 0))
        leger = [c for c in cands if c["plus_leger"]]
        forme = forme and (not leger or all(c["masse"] >= leger[0]["masse"] for c in cands if c["taux"] <= 1))
        # trois lignes prises au hasard : choisies dans le menu de l'écran, la colonne de droite est celle du moteur
        rng = random.Random(len(cands))
        for c in rng.sample(cands, min(3, len(cands))):
            e3 = dict(e2, profil=c["nom"])
            ecarts_c = []
            comparer(f"{desc} · {c['nom']}", EN.normaliser(e3), M.verifier(e3), ecarts_c)
            ecarts += ecarts_c[:2]
        chk(f"sélecteur « {desc} » ({', '.join(fams) if fams else 'familles de l’élément'}) : {len(cands)} lignes triées par masse, "
            f"taux = ceux de l'écran ; trois choix repassés par l'écran", not ecarts and forme, (ecarts[:4], forme))

# ------------------------------------------------------------------ 6. format
print("\n=== 6. Format : JSON, normalisation, domaine de l'écran, formules ===")
tous = [M.verifier(e) for _, e in CAS]
chk("la sortie de verifier() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(r_, allow_nan=False)) == r_ for r_ in tous))
r0 = M.verifier(cas("défauts de l'écran"))
chk("sans effort, le gouvernant est l'interaction N + M : max(η_NM ; η_Vy ; η_Vz) rend le PREMIER maximum",
    r0["gouvernant"]["cle"] == "NM" and r0["gouvernant"]["taux"] == 0.0, r0["gouvernant"])
r_dec = M.verifier(cas("efforts saisis puis décochés"))
chk("cases décochées : le calcul reçoit ce que l'écran lui passe (N = M_z = V_y = T = 0 ; L_cr = 0 et α = 0,34 sans flambement)",
    r_dec["efforts"] == {"N": 0.0, "My": 45.0, "Mz": 0.0, "Vy": 0.0, "Vz": 30.0, "T": 0.0}
    and {k: r_dec["flambement"][k] for k in ("Lcr_y", "Lcr_z", "alpha", "courbe")} == {"Lcr_y": 0.0, "Lcr_z": 0.0, "alpha": 0.34, "courbe": None}
    and r_dec["deversement"]["L_LT"] == 0.0, (r_dec["efforts"], r_dec["flambement"]))
chk("normaliser() est idempotente et {} donne nouvel_element()",
    all(EN.normaliser(EN.normaliser(e)) == EN.normaliser(e) for _, e in CAS) and EN.normaliser({}) == EN.nouvel_element())
entree = copy.deepcopy(cas("tous les efforts"))
M.verifier(entree)
M.candidats(entree, ["HEB"])
chk("verifier() et candidats() ne modifient pas l'élément reçu", entree == cas("tous les efforts"))
chk("le libellé de courbe du module (« c (0.49) ») est admis", EN.normaliser({"courbe": "c (0.49)"})["courbe"] == "c")
chk("les familles se rangent dans l'ordre de l'écran, sans doublon", EN.normaliser({"familles": ["IPE", "HEA", "IPE"]})["familles"] == ["HEA", "IPE"])
REFUS = (("famille inconnue", {"familles": ["UPN"]}), ("familles en texte", {"familles": "HEA"}), ("acier S450", {"acier": "S450"}),
         ("γM0 = 0,5", {"gamma_M0": 0.5}), ("γM1 = 1,6", {"gamma_M1": 1.6}), ("I_y min négatif", {"Iy_min": -1.0}),
         ("L_cr,y négatif", {"Lcr_y": -2.0}), ("L_LT négatif", {"L_LT": -0.1}), ("courbe e", {"courbe": "e"}),
         ("M_y en texte", {"My": "45"}), ("V_z booléen", {"Vz": True}), ("N infini", {"N": float("inf")}), ("M_z NaN", {"Mz": float("nan")}),
         ("case en nombre", {"avec_N": 1}), ("clé inconnue", {"Myd": 3.0}), ("profilé inconnu", {"profil": "HEA 205"}),
         ("profilé en nombre", {"profil": 200}), ("nom en nombre", {"nom": 12}), ("élément en liste", None),
         # le DOMAINE DU SITE : l'écran les accepte mais son calcul y déborde (OverflowError, division par zéro) ou écrit
         # des nombres de cent chiffres — refusés avec le nom du champ (entree.hors_domaine)
         ("M_y = 1e303 (OverflowError dans l'écran)", {"My": 1e303}), ("N = -2e6", {"N": -2e6, "avec_N": True}), ("V_z = 1,000001e6", {"Vz": 1.000001e6}),
         ("T = 1e308", {"T": 1e308}), ("L_cr,y = 1e-200 (division par zéro)", {"Lcr_y": 1e-200, "flambement": True, "avec_N": True, "N": 5.0}),
         ("L_cr,z = 1e150 (OverflowError)", {"Lcr_z": 1e150}), ("L_LT = 0,5 mm", {"L_LT": 5e-4}), ("I_y min = 1e9", {"Iy_min": 1e9}))
refusees = []
for nom, champ in REFUS:
    try:
        M.verifier([] if champ is None else dict(EN.nouvel_element(), **champ))
        refusees.append(f"{nom} : accepté")
    except ValueError:
        pass
    except Exception as ex:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
        refusees.append(f"{nom} : {type(ex).__name__} {ex}")
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError (jamais de substitution silencieuse)", not refusees, refusees)
ADMIS = ({"My": 1e6}, {"N": -1e6, "avec_N": True}, {"Lcr_y": 1e-3, "Lcr_z": 1e3, "flambement": True, "avec_N": True, "N": 1e6}, {"Iy_min": 1e8},
         {"Lcr_y": 0.0, "L_LT": 0.0})
admis_ko = []
for champ in ADMIS:
    try:
        r_ = M.verifier(dict(EN.nouvel_element(), **champ))
        json.dumps(r_, allow_nan=False)
    except Exception as ex:  # noqa: BLE001
        admis_ko.append((champ, f"{type(ex).__name__} {ex}"))
chk("les bornes du site elles-mêmes (±10⁶ kN ; 1 mm et 1 000 m ; 10⁸ cm⁴) sont admises et se calculent (JSON strict)", not admis_ko, admis_ko)
try:
    M.candidats(EN.nouvel_element(), ["UPN"])
    chk("candidats : famille inconnue → ValueError", False)
except ValueError:
    chk("candidats : famille inconnue → ValueError", True)
# « formule = valeurs = résultat » : chaque résultat se retrouve à partir des seuls termes que le moteur donne
FORMULES = {
    "N_Rd": lambda t: t["A"] * 1e2 * t["fy"] / t["gamma_M0"] / 1e3,
    "My_Rd": lambda t: t["Wel_y"] * 1e3 * t["fy"] / t["gamma_M0"] / 1e6,
    "Mz_Rd": lambda t: t["Wel_z"] * 1e3 * t["fy"] / t["gamma_M0"] / 1e6,
    "Vz_Rd": lambda t: t["Avz"] * 1e2 * t["fy"] / (math.sqrt(3) * t["gamma_M0"]) / 1e3,
    "Vy_Rd": lambda t: t["Avy"] * 1e2 * t["fy"] / (math.sqrt(3) * t["gamma_M0"]) / 1e3,
    "rN": lambda t: t["N_Ed"] / t["N_Rd"] if t else 0.0, "rMy": lambda t: t["My_Ed"] / t["My_Rd"] if t else 0.0,
    "rMz": lambda t: t["Mz_Ed"] / t["Mz_Rd"] if t else 0.0, "rVz": lambda t: t["Vz_Ed"] / t["Vz_Rd"] if t else 0.0,
    "rVy": lambda t: t["Vy_Ed"] / t["Vy_Rd"] if t else 0.0,
    "util_NM": lambda t: t["rN"] + t["rMy"] + t["rMz"], "util": lambda t: max(t["util_NM"], t["rVy"], t["rVz"]),
    "Ncr": lambda t, a: math.pi ** 2 * t["E"] * t[f"I_{a}"] * 1e4 / (t[f"Lcr_{a}"] * 1e3) ** 2 / 1e3,
    "lambda": lambda t, a: math.sqrt(t["A"] * 1e2 * t["fy"] / (t[f"Ncr_{a}"] * 1e3)),
    "phi": lambda t, a: 0.5 * (1 + t["alpha"] * (t[f"lambda_{a}"] - 0.2) + t[f"lambda_{a}"] ** 2),
    "chi": lambda t, a: min(1.0, 1 / (t[f"phi_{a}"] + math.sqrt(max(t[f"phi_{a}"] ** 2 - t[f"lambda_{a}"] ** 2, 0.0)))),
    "Nb": lambda t, a: t[f"chi_{a}"] * t["A"] * 1e2 * t["fy"] / t["gamma_M1"] / 1e3,
    "util_buck": lambda t: t["N_Ed"] / t["Nb_Rd"],
    "util_global": lambda t: max(t["util"], t["util_buck"]) if t["util_buck"] is not None else t["util"],
}
fautes, n_formules = [], 0
for (desc, e), res in zip(CAS, tous):
    if not res["profil"]:
        continue
    for f in res["profil"]["formules"]:
        n_formules += 1
        t = {x["symbole"]: x["valeur"] for x in f["termes"]}
        k = f["cle"]
        if k.startswith(("Ncr_", "lambda_", "phi_", "chi_", "Nb_Rd_")):
            racine, axe = k.rsplit("_", 1)
            v = FORMULES[racine.replace("_Rd", "")](t, axe)
        else:
            v = FORMULES[k](t)
        if not proche(v, f["resultat"], 1e-12) and not (v == 0 and f["resultat"] == 0):
            fautes.append(f"{e['nom']}/{k} : {v!r} ≠ {f['resultat']!r}")
    for vv in res["verifications"]:
        if vv["limite"] and vv["cle"] in ("N", "My", "Mz", "Vz", "Vy", "flambement") and not proche(vv["valeur"] / vv["limite"], vv["taux"], 1e-12):
            fautes.append(f"{e['nom']}/{vv['cle']} : valeur / limite ≠ taux")
chk(f"« formule = valeurs = résultat » : les {n_formules} formules se recalculent à partir de leurs seuls termes, et valeur / limite = taux",
    not fautes and n_formules > 500, fautes[:4])
doc = lire("docs", "acier", "CHOIX_PROFIL_MOTEUR.md")
r_tout = M.verifier(cas("tous les efforts"))
cles = sorted(set(r_tout) | set(r_tout["lignes"][0]) | set(r_tout["profil"]) | set(r_tout["profil"]["axes"][0]) | set(r_tout["verifications"][0])
              | set(r_tout["efforts"]) | set(r_tout["flambement"]) | set(r_tout["gouvernant"]))
sans_doc = [k for k in cles if f"`{k}`" not in doc or f"``{k}``" not in M.__doc__]
chk(f"les {len(cles)} clés de la sortie sont nommées dans la docstring du moteur ET dans docs/acier/CHOIX_PROFIL_MOTEUR.md", not sans_doc, sans_doc)

# ------------------------------------------------------------------ 7. pureté
print("\n=== 7. Pureté : le moteur se charge et calcule SANS streamlit ni pandas ===")
non_std = []
for dossier, _, fichiers in os.walk(os.path.join(RACINE, "acier", "choix_profil")):
    for f in fichiers:
        if f.endswith(".py"):
            arbre = ast.parse(lire(os.path.relpath(os.path.join(dossier, f), RACINE)))
            for n in ast.walk(arbre):
                noms_ = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
                non_std += [f"{f}: {x}" for x in noms_ if x.split(".")[0] not in sys.stdlib_module_names | {"acier"}]
chk("aucun fichier d'acier/choix_profil n'importe autre chose que la bibliothèque standard et le paquet acier (lecture des sources)",
    not non_std, non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "matplotlib", "scipy", "PIL", "fitz", "pymupdf")
class Refus:
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("chargement refusé : " + nom)
        return None
sys.meta_path.insert(0, Refus())
try:
    import pandas  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from acier.choix_profil import moteur as M
elements = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.verifier(e) for e in elements],
          "cands": M.candidats(elements[1], ["HEA", "IPE"])}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"acier"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
elements = [e for _, e in CAS]
attendu = {"res": json.loads(json.dumps(tous)), "cands": json.loads(json.dumps(M.candidats(elements[1], ["HEA", "IPE"])))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(elements), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit ni pandas", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, pandas, matplotlib… REFUSÉS par sys.meta_path, et le moteur calcule quand même ({len(out['res'])} éléments, le sélecteur)",
        out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard et numpy (le catalogue des profilés)", set(out["tiers"]) <= {"numpy"}, out["tiers"])
    diff = [i for i, (a, b) in enumerate(zip(out["res"], attendu["res"])) if a != b] + ([-1] if out["cands"] != attendu["cands"] else [])
    chk(f"Python {v} : résultats identiques à ceux de ce processus (au bit près)", not diff, diff[:3])
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")


# ------------------------------------------------------------------ la campagne de mutations (--mutations)
MUTATIONS = [  # (fichier d'acier/choix_profil, texte, remplacement, ce que la mutation casse)
    ("calcul.py", "if Wel_y is None or Wel_y <= 0:", "if Wel_y is None or Wel_y < 0:", "le filtre W_el,y > 0 du chargement"),
    ("calcul.py", 'lire_cle(p, "Avy")', 'lire_cle(p, "Avz")', "A_v,y lu à la place de rien (le défaut du module corrigé en douce)"),
    ("calcul.py", 'lire_cle(p, "Poids", "masse")', 'lire_cle(p, "masse", "A")', "la masse"),
    ("calcul.py", "N_Rd = (A_mm2 * fy / gamma_M0)", "N_Rd = (A_mm2 * fy / 1.0)", "γ_M0 sur N_Rd"),
    ("calcul.py", "My_Rd = (Wy_mm3 * fy / gamma_M0)", "My_Rd = (Wy_mm3 * fy * 1.0001 / gamma_M0)", "M_y,Rd à 10⁻⁴ près"),
    ("calcul.py", "(math.sqrt(3) * gamma_M0)) if Avz_mm2", "(math.sqrt(2) * gamma_M0)) if Avz_mm2", "le √3 de V_z,Rd"),
    ("calcul.py", 'rN = (NEd / resist["N_Rd_N"])', 'rN = (abs(NEd) / resist["N_Rd_N"])', "η_N sans valeur absolue (le défaut du module)"),
    ("calcul.py", "interaction_NM = rN + rMy + rMz", "interaction_NM = rN + rMy", "M_z dans l'interaction"),
    ("calcul.py", "interaction = max(interaction_NM, rVy, rVz)", "interaction = max(interaction_NM, rVz)", "V_y dans le taux de section"),
    ("calcul.py", "phi = 0.5 * (1.0 + alpha * (lambdabar - 0.2) + lambdabar**2)", "phi = 0.5 * (1.0 + alpha * (lambdabar - 0.25) + lambdabar**2)",
     "le plateau 0,2 de φ"),
    ("calcul.py", "chi = min(chi, 1.0)", "chi = min(chi, 1.01)", "χ ≤ 1"),
    ("calcul.py", 'E_ACIER = 210000.0', 'E_ACIER = 205000.0', "E du flambement"),
    ("calcul.py", "Ncr = (math.pi**2 * E * I_mm4) / (L_mm**2)", "Ncr = (math.pi**2 * E * I_mm4) / (L_mm**2.0001)", "N_cr"),
    ("moteur.py", "Nb_Rd = buck[\"chi\"] * (A * 1e2) * fy / gamma_M1", "Nb_Rd = buck[\"chi\"] * (A * 1e2) * fy / gamma_M0", "γ_M1 du flambement"),
    ("moteur.py", "if el[\"flambement\"] and NEd > 0:", "if el[\"flambement\"] and NEd >= 0:", "le flambement seulement en compression"),
    ("moteur.py", "Nb_Rd_min = min(candidates)", "Nb_Rd_min = max(candidates)", "l'axe le plus défavorable"),
    ("moteur.py", "util_global = max(util_global, util_buck)", "util_global = util_global", "le flambement dans η"),
    ("moteur.py", "utilisation = round(util_global * 100, 3)", "utilisation = round(util_global * 100, 2)", "l'arrondi de la colonne Utilisation"),
    ("moteur.py", "\"tient\": utilisation <= 100.0}", "\"tient\": utilisation < 100.0}", "« ≤ 100 » du module"),
    ("moteur.py", "if li[\"utilisation\"] <= 100.0:", "if li[\"taux\"] <= 1.0:", "l'optimal sur la colonne ARRONDIE (le défaut du module)"),
    ("moteur.py", "if ecart is None or e < ecart:", "if ecart is None or e <= ecart:", "le premier à égalité (idxmin)"),
    ("moteur.py", "key=lambda x: x[\"utilisation\"])", "key=lambda x: x[\"taux\"])", "le tri sur la colonne arrondie"),
    ("moteur.py", "Iv_min > 0 and v[\"Iy\"] < Iv_min", "Iv_min > 0 and v[\"Iy\"] <= Iv_min", "le filtre I_y min"),
    ("moteur.py", "if familles else profiles", "if familles else {}", "aucune famille = toutes"),
    ("moteur.py", "\"N\": el[\"N\"] if el[\"avec_N\"] else 0.0", "\"N\": el[\"N\"]", "N ignoré case décochée"),
    ("moteur.py", "(optimal if optimal in noms else noms[0])", "(noms[0])", "le profilé retenu par défaut = l'optimal"),
    ("moteur.py", "\"h\": int(prof[\"h\"]) if prof.get(\"h\") else None", "\"h\": prof.get(\"h\")", "la colonne h en entier"),
    ("moteur.py", "if x > t:", "if x >= t:", "le premier maximum (gouvernant)"),
    ("moteur.py", "\"Vy_Rd_absent\"", "\"Vy_Rd_absent_\"", "l'avertissement V_y non vérifié"),
    ("moteur.py", "if t_max > EPAISSEUR_LIMITE:", "if t_max > 400.0:", "l'avertissement f_y non réduit au-delà de 40 mm"),
    ("moteur.py", "FY_EPAIS = {\"S235\": 215,", "FY_EPAIS = {\"S235\": 225,", "la valeur réduite du tableau 3.1"),
    ("moteur.py", "return {\"Lcr_y\": 0.0, \"Lcr_z\": 0.0, \"alpha\": EN.ALPHA_SANS_FLAMBEMENT, \"courbe\": None}",
     "return {\"Lcr_y\": el[\"Lcr_y\"], \"Lcr_z\": 0.0, \"alpha\": EN.ALPHA_SANS_FLAMBEMENT, \"courbe\": None}", "L_cr ignorés case décochée"),
    ("entree.py", "GAMMA_MIN, GAMMA_MAX = 0.8, 1.5", "GAMMA_MIN, GAMMA_MAX = 0.8, 2.0", "les bornes de γ"),
    ("entree.py", "(\"b\", 0.34, \"b (0.34)\")", "(\"b\", 0.35, \"b (0.34)\")", "α de la courbe b"),
    ("entree.py", "COURBE_DEFAUT = \"b\"", "COURBE_DEFAUT = \"c\"", "la courbe par défaut"),
    ("entree.py", "return [\"HEA\"] if \"HEA\" in dispo else dispo[:1]", "return [\"HEB\"]", "la famille par défaut"),
    ("entree.py", "        if el[k] < 0:\n", "        if el[k] < -1:\n", "le refus des longueurs négatives"),
    ("entree.py", "    el[\"familles\"] = [f for f in dispo if f in fams]", "    el[\"familles\"] = list(fams)", "l'ordre des familles"),
    ("entree.py", "EFFORT_MAX = 1e6 ", "EFFORT_MAX = 1e400 ", "la borne des efforts (le calcul de l'écran déborde)"),
    ("moteur.py", "LIMITE_SEMELLE_CL3, LIMITE_AME_COMPRESSION_CL3 = 14.0, 42.0", "LIMITE_SEMELLE_CL3, LIMITE_AME_COMPRESSION_CL3 = 14.0, 43.0",
     "la limite 42ε de l'âme comprimée (classe 4 dite)"),
    ("moteur.py", "if flechi and c_f / tf >", "if False and c_f / tf >", "la semelle de classe 4 dite"),
    ("entree.py", "x > LONGUEUR_MAX or 0 < x < LONGUEUR_MIN", "x > LONGUEUR_MAX", "la longueur minimale (division par zéro)"),
]
TEMOIN = ("moteur.py", "REFERENCE = \"modules/choix_profile.py", "REFERENCE = \"modules/choix_profile.py", "témoin : rien ne change")


def campagne():
    """Chaque mutation dans un ARBRE MIROIR (liens vers le dépôt, copie d'acier/choix_profil modifiée) : ce test y est
    rejoué et doit rougir ; le témoin doit rester vert. Le dépôt n'est jamais modifié."""
    import time
    resultats = []
    for fichier, avant_, apres, quoi in [TEMOIN] + MUTATIONS:
        tmp = tempfile.mkdtemp(prefix="mutation_profil_")
        try:
            for nom in os.listdir(RACINE):
                if nom not in ("acier", "tests", "docs"):
                    os.symlink(os.path.join(RACINE, nom), os.path.join(tmp, nom))
            for d in ("acier", "tests", "docs"):
                os.mkdir(os.path.join(tmp, d))
                for nom in os.listdir(os.path.join(RACINE, d)):
                    if (d, nom) not in (("acier", "choix_profil"), ("tests", "test_choix_profil_moteur.py")):
                        os.symlink(os.path.join(RACINE, d, nom), os.path.join(tmp, d, nom))
            shutil.copytree(os.path.join(RACINE, "acier", "choix_profil"), os.path.join(tmp, "acier", "choix_profil"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy(os.path.join(RACINE, "tests", "test_choix_profil_moteur.py"), os.path.join(tmp, "tests"))
            cible = os.path.join(tmp, "acier", "choix_profil", fichier)
            src = open(cible, encoding="utf-8").read()
            if src.count(avant_) != 1:
                resultats.append((quoi, "texte introuvable", False))
                continue
            with open(cible, "w", encoding="utf-8") as fh:
                fh.write(src.replace(avant_, apres))
            t0 = time.time()
            pr = subprocess.run([sys.executable, os.path.join(tmp, "tests", "test_choix_profil_moteur.py")], capture_output=True, text=True, check=False)
            rouges = [ligne.strip() for ligne in pr.stdout.splitlines() if ligne.startswith("  ECHEC")]
            temoin = quoi.startswith("témoin")
            resultats.append((quoi, f"{len(rouges)} échec(s), {time.time() - t0:.0f} s" + (f" — {rouges[0][8:90]}" if rouges else ""),
                              (pr.returncode == 0) if temoin else (pr.returncode != 0)))
            print(("  OK    " if resultats[-1][2] else "  ECHEC ") + f"mutation « {quoi} » : {resultats[-1][1]}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for *_, ok in resultats)
    print(f"\nMUTATIONS : {n_ok} / {len(resultats)} conformes (chaque mutation rougit, le témoin reste vert)")
    return n_ok == len(resultats)


if "--mutations" in sys.argv:
    sys.exit(0 if campagne() else 1)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

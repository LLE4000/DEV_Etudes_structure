# -*- coding: utf-8 -*-
"""Moteur « Cornières » (acier/corniere) : la PARITÉ avec l'écran Streamlit modules/corniere.py, qui reste la
référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Données : les défauts et les bornes du moteur (entree.CHAMPS, la section proposée, le type de charge) sont LUS
     dans l'écran lui-même — ses st.number_input, st.radio, st.toggle, st.selectbox, rejoués avec un faux « st ».
  2. À la main : deux cornières recalculées ici, formules écrites en N et en mm.
  3. Parité : l'écran est EXÉCUTÉ (show()) avec un faux « st » qui sert les saisies du cas et enregistre ce qui
     s'affiche ; une trace (sys.settrace) relève TOUTES les variables locales de show() à son retour. On compare au
     moteur, AU BIT PRÈS (égalité stricte des flottants) : chaque grandeur, chaque ✅ / ❌, et ce que l'écran écrit
     (pourcentages, épaisseur, moments affichés). Une variable ajoutée à l'écran fait rougir le test tant qu'elle
     n'est pas comparée. Les 41 sections de la liste de l'écran, la saisie libre, les charges linéaires et
     ponctuelles, des tirages à graine fixe et des bornes exactes (≤ et non <).
  4. Couverture : chaque verdict ✅ et ❌, chaque borne, chaque type de charge… est effectivement atteint.
  5. Catalogue : la liste de l'écran (ANGLES_STD) face au catalogue acier/profiles — sections absentes, masses
     différentes : le constat est ÉCRIT dans docs/acier/CORNIERE_MOTEUR.md, et le test exige qu'il y soit.
  6. Sélecteur : chaque cornière du catalogue repassée par l'écran (saisie libre, même t) ; tri par masse, optimal.
  7. Format : JSON strict, normalisation (défauts, ValueError hors du domaine de l'écran), formules recalculées à
     partir de leurs seuls termes, grandeurs documentées.
  8. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, pandas,
     reportlab, matplotlib, scipy, PIL, fitz : seuls la bibliothèque standard et numpy (le catalogue) y sont chargés,
     sous Python 3.12 et sous 3.13 (celui de Pyodide) s'il est installé — résultats identiques.
  9. Mutations : le moteur rechargé depuis ses sources, modifié d'UNE ligne (formule, coefficient, unité, « ≤ »,
     gouvernant, sélecteur, défaut, borne, refus), est rejeté par les contrôles ci-dessus ; un témoin ne l'est pas.

    /usr/bin/python3.12 tests/test_corniere_moteur.py       (streamlit requis pour la parité)
"""
import ast
import copy
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.corniere import entree as E  # noqa: E402
from acier.corniere import moteur as M  # noqa: E402

OK, KO = [], []


def lire(*chemin):
    f = os.path.join(RACINE, *chemin)
    if not os.path.exists(f):
        return ""
    with open(f, encoding="utf-8") as fh:
        return fh.read()


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


ECART = {"n": 0, "bit": 0}


def egal(a, b):
    """Égalité STRICTE (au bit près pour les flottants) ; les booléens et les textes aussi."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        ECART["n"] += 1
        if a == b and math.copysign(1, a) == math.copysign(1, b):
            ECART["bit"] += 1
            return True
        return False
    return a == b and type(a) is type(b)


# ------------------------------------------------------------------ l'écran, rejoué
# la liste de l'écran, relue dans SA source (une variable locale de show())
ARBRE = ast.parse(lire("modules", "corniere.py"))
TABLE = next(ast.literal_eval(n.value) for n in ast.walk(ARBRE)
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "ANGLES_STD")

try:
    logging.disable(logging.WARNING)
    import streamlit  # noqa: F401,E402 — l'écran l'importe
    import modules.corniere as MC  # noqa: E402
    import modules.navigation as NAV  # noqa: E402
except ImportError as e:  # pragma: no cover — la parité EXIGE l'écran
    MC = None
    chk("streamlit et modules/corniere.py se chargent (indispensables à la parité)", False, e)

L_STD = "Utiliser une cornière **standard**"
L_SEC, L_TYPE = "Section", "Type de charge"
L_A, L_B, L_T = "Largeur aile A (mm)", "Largeur aile B (mm)", "Épaisseur t (mm)"
L_VLIN, L_VPON = "Effort tranchant V (kN/m)", "Effort tranchant V (kN)"
L_E, L_G = "Excentricité e (cm) – distance charge/cornière", "Coefficient global (ELU)"
L_S, L_BORD = "Espacement des fixations s (cm)", "Distance de l’âme au bord du mur (cm)"
L_SIG, L_TAU, L_SIGU = "σ_lim, ELS (N/mm²)", "τ_lim, ELS (N/mm²)", "σ_eq lim, ELU (N/mm²)"
L_L = "Longueur d’influence L pour charge ponctuelle (m)"
RADIO = {"Linéaire (kN/m)": "lineaire", "Ponctuelle (kN)": "ponctuelle"}
# champ du moteur → libellé du champ de l'écran (le V a deux libellés selon le type de charge)
LIBELLE = {"A": L_A, "B": L_B, "t": L_T, "V": L_VLIN, "L": L_L, "e": L_E, "gamma_ELU": L_G, "s": L_S, "bord": L_BORD,
           "sig_lim_ELS": L_SIG, "tau_lim_ELS": L_TAU, "sig_lim_ELU": L_SIGU}


class Colonne:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def metric(self, label, value, *a, **k):
        self.st.ev.append(("metric", label, value))

    def write(self, *a, **k):
        self.st.ev.append(("write",) + a)


class FauxSt:
    """Un « st » qui SERT les saisies d'un cas (par libellé de champ) et ENREGISTRE ce que l'écran afficherait.
    Les bornes des champs ne sont pas appliquées ici : le moteur les applique (normaliser), le test le vérifie."""
    AFFICHAGES = ("header", "subheader", "divider", "caption", "info", "success", "markdown", "write", "warning", "error",
                  "set_page_config")

    def __init__(self, saisies):
        self.saisies = dict(saisies)
        self.ev, self.widgets = [], []
        self.sidebar = self

    def __getattr__(self, nom):
        if nom in FauxSt.AFFICHAGES:
            return lambda *a, **k: self.ev.append((nom,) + a)
        raise AttributeError(nom)

    def columns(self, spec, **_):
        return [Colonne(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def button(self, *_, **__):
        return False

    def toggle(self, label, value=False, **_):
        self.widgets.append(("toggle", label, value))
        return self.saisies.get(label, value)

    def selectbox(self, label, options, index=0, **_):
        self.widgets.append(("selectbox", label, list(options), index))
        return self.saisies.get(label, list(options)[index])

    def radio(self, label, options, index=0, **_):
        self.widgets.append(("radio", label, list(options), index))
        return self.saisies.get(label, list(options)[index])

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **_):
        self.widgets.append(("number_input", label, min_value, max_value, value, step))
        return self.saisies.get(label, value)


def ecran(saisies):
    """show() exécuté avec un faux st : (ce qui s'affiche, les variables locales de show() à son retour)."""
    faux, pris = FauxSt(saisies), {}
    code = MC.show.__code__

    def trace(frame, event, arg):
        if event == "call":
            return trace if frame.f_code is code else None
        if event == "return" and frame.f_code is code:
            pris.update(frame.f_locals)
        return trace
    vrai, vrai_nav = MC.st, NAV.st
    MC.st, NAV.st = faux, faux
    sys.settrace(trace)
    try:
        MC.show()
    finally:
        sys.settrace(None)
        MC.st, NAV.st = vrai, vrai_nav
    return faux, pris


def saisies_de(c, cle_ecran=None):
    """Les saisies de l'écran pour la cornière c (format du moteur). ``cle_ecran`` : la section de la liste de l'écran
    à choisir (sinon : saisie libre avec les dimensions de la section du moteur)."""
    sec = M.section(M.normaliser(c))
    s = {L_STD: cle_ecran is not None, L_TYPE: {v: k for k, v in RADIO.items()}[c["charge"]]}
    if cle_ecran is not None:
        s[L_SEC] = cle_ecran
    else:
        s.update({L_A: sec["A"], L_B: sec["B"], L_T: sec["t"]})
    s[L_VLIN if c["charge"] == "lineaire" else L_VPON] = c["V"]
    s.update({L_E: c["e"], L_G: c["gamma_ELU"], L_S: c["s"], L_BORD: c["bord"], L_SIG: c["sig_lim_ELS"],
              L_TAU: c["tau_lim_ELS"], L_SIGU: c["sig_lim_ELU"], L_L: c["L"]})
    return s


# ------------------------------------------------------------------ les cas, au format du moteur
def corn(nom="C", profil=None, A=60.0, B=60.0, t=6.0, charge="lineaire", **k):
    c = dict(M.nouvelle_corniere(), nom=nom, A=A, B=B, t=t, charge=charge)
    c["standard"] = profil is not None
    if profil:
        c["profil"] = profil
    c.update(k)
    return M.normaliser(c)


def nom_catalogue(cle):
    v = TABLE[cle]
    return f"L {v['A']:g}x{v['B']:g}x{v['t']:g}"


CAT = {p["nom"] for p in M.cornieres()}
CAS = []      # (description, cornière du moteur, section de la liste de l'écran ou None)
CAS.append(("défauts de l'écran (nouvelle_corniere)", M.nouvelle_corniere(), "100x100x10"))
for cle in TABLE:   # chaque section de la liste de l'écran : au catalogue (même A, B, t) ou en saisie libre
    v = TABLE[cle]
    if nom_catalogue(cle) in CAT:
        CAS.append((f"liste de l'écran {cle} = catalogue {nom_catalogue(cle)}", corn(cle, nom_catalogue(cle), V=12.0, e=4.0), cle))
    else:
        CAS.append((f"liste de l'écran {cle} (absente du catalogue) = saisie libre", corn(cle, None, v["A"], v["B"], v["t"], V=12.0, e=4.0), cle))
CAS += [
    ("saisie libre 60 × 60 × 6 (défauts de la saisie)", corn("S", None), None),
    ("saisie libre 150 × 90 × 7,5, charge forte", corn("S2", None, 150, 90, 7.5, V=40.0, e=6.5), None),
    ("saisie libre t = 2 mm (borne basse)", corn("S3", None, 30, 30, 2.0, V=1.0, e=1.0), None),
    ("saisie libre t = 20 mm (borne haute)", corn("S4", None, 300, 300, 20.0, V=150.0, e=12.0), None),
    ("ponctuelle 8 kN sur L = 0,6 m, catalogue L 80x80x8", corn("P1", "L 80x80x8", charge="ponctuelle", V=8.0, L=0.6), None),
    ("ponctuelle sur L = 0,1 m (borne basse)", corn("P2", "L 60x60x6", charge="ponctuelle", V=2.0, L=0.1, e=3.0), None),
    ("ponctuelle sur L = 5 m (borne haute)", corn("P3", None, 70, 70, 7.0, charge="ponctuelle", V=60.0, L=5.0, e=5.0), None),
    ("V = 0 : tout est nul", corn("V0", "L 50x50x5", V=0.0), None),
    ("e = 0 : cisaillement seul", corn("E0", "L 50x50x5", e=0.0, V=3.0), None),
    ("γ_ELU = 1 (borne basse)", corn("G1", "L 70x70x7", gamma_ELU=1.0, V=20.0, e=5.0), None),
    ("γ_ELU = 1,5", corn("G15", "L 70x70x7", gamma_ELU=1.5, V=20.0, e=5.0), None),
    ("s = 10 cm (borne basse), ailes inégales L 100x65x8", corn("S10", "L 100x65x8", s=10.0, V=15.0, e=4.5), None),
    ("s = 150 cm, L 150x100x12", corn("S150", "L 150x100x12", s=150.0, V=30.0, e=7.0), None),
    ("ELS non conforme par σ seulement (τ admis)", corn("NS", "L 40x40x4", V=3.0, e=12.0, tau_lim_ELS=5.0), None),
    ("ELS non conforme par τ seulement (σ admis)", corn("NT", "L 40x40x4", V=4.0, e=0.5), None),
    ("ELS non conforme par σ et par τ", corn("NST", "L 30x30x3", V=6.0, e=10.0), None),
    ("ELS admis, ELU non conforme (σ_lim,ELS haut)", corn("NU", "L 50x50x6", V=6.0, e=10.0, sig_lim_ELS=400.0, tau_lim_ELS=5.0), None),
    ("tout non conforme, grande excentricité", corn("NX", "L 60x60x6", V=25.0, e=20.0), None),
    ("critères au minimum de l'écran (10 ; 0,1 ; 10)", corn("CM", "L 120x120x12", V=1.0, e=1.0, sig_lim_ELS=10.0, tau_lim_ELS=0.1, sig_lim_ELU=10.0), None),
    ("critères Eurocode S355 (σ 355, τ 205, σ_eq 355)", corn("S355", "L 90x90x9", V=35.0, e=6.0, sig_lim_ELS=355.0, tau_lim_ELS=205.0, sig_lim_ELU=355.0), None),
    ("distance au bord 0 et 25 cm (sans effet sur le calcul)", corn("BD", "L 80x80x10", bord=25.0), None),
    ("valeurs non rondes (V = 7,35 ; e = 3,15 ; γ = 1,43 ; s = 27,5)", corn("NR", "L 75x75x8", V=7.35, e=3.15, gamma_ELU=1.43, s=27.5), None),
    ("catalogue t = 25 mm (au-delà de la saisie libre de l'écran : même code, borne de champ contournée)",
     corn("T25", "L 200x200x25", V=200.0, e=8.0), None),
    ("catalogue t = 35 mm, la plus lourde", corn("T35", "L 250x250x35", V=300.0, e=10.0), None),
]
# bornes EXACTES : la valeur égale la limite au flottant près ; ✅ attendu (≤ et non <)
CAS += [
    ("borne exacte τ_ELS = τ_lim,ELS = 0,8 (V = 8 kN/m, t = 10)", corn("BT", "L 100x100x10", V=8.0, e=1.0), None),
]
_g = M.calculer(10.0, 30.0, "lineaire", 1.0, 6.0, 1.35, 30.0, 155.0, 0.8, 186.0)
CAS += [
    ("borne exacte σ_eq,ELS = σ_lim,ELS (limite construite)", corn("BS", "L 100x100x10", V=30.0, e=6.0, sig_lim_ELS=_g["sigma_eq_ELS"], tau_lim_ELS=5.0), None),
    ("borne exacte σ_eq,ELU = σ_eq,lim,ELU (limite construite)", corn("BU", "L 100x100x10", V=30.0, e=6.0, sig_lim_ELU=_g["sigma_eq_ELU"], tau_lim_ELS=5.0), None),
]


def aleatoire(rng, k):
    """Une cornière tirée au sort (graine fixe) : catalogue ou saisie libre, charge, excentricité, critères."""
    cat = rng.random() < 0.6
    p = rng.choice(M.cornieres()) if cat else None
    charge = rng.choice(("lineaire", "ponctuelle"))
    t = round(rng.uniform(2.0, 20.0) * 2) / 2
    A, B = round(rng.uniform(10, 300), 1), round(rng.uniform(10, 300), 1)
    A, B = (max(A, 2 * t), max(B, 2 * t)) if t >= min(A, B) else (A, B)     # une cornière : ailes plus larges que t
    return corn(f"Aléa {k}", p["nom"] if cat and p["t"] <= 20 else None, A, B, t,
                charge, V=round(rng.uniform(0, 60), 2), L=round(rng.uniform(0.1, 5.0), 2), e=round(rng.uniform(0, 25), 2),
                gamma_ELU=rng.choice((1.0, 1.35, 1.5, round(rng.uniform(1, 2), 2))), s=float(rng.choice(range(10, 121, 5))),
                bord=round(rng.uniform(0, 20), 1), sig_lim_ELS=rng.choice((155.0, 160.0, 235.0)),
                tau_lim_ELS=rng.choice((0.8, 2.5, 90.0, round(rng.uniform(0.1, 120), 2))), sig_lim_ELU=rng.choice((186.0, 235.0, 355.0)))


RNG = random.Random(20261107)
CAS += [(f"tirage {k:02d} (graine 20261107)", aleatoire(RNG, k), None) for k in range(1, 31)]

# variable locale de show() ↔ grandeur du moteur
CORRESPONDANCE = (
    ("b_trans_mm", "b_trans"), ("t_mm", "t"), ("I_trans", "I_trans"), ("W_trans", "W_trans"), ("Az_trans", "Az_trans"),
    ("V_ELS", "V_ELS"), ("V_ELU", "V_ELU"), ("e_m", "e_m"), ("M_ELS_Nmm", "M_ELS_Nmm"), ("M_ELU_Nmm", "M_ELU_Nmm"),
    ("sigma_ELS", "sigma_ELS"), ("sigma_ELU", "sigma_ELU"), ("tau_ELS", "tau_ELS"), ("tau_ELU", "tau_ELU"),
    ("sigma_eq_ELS", "sigma_eq_ELS"), ("sigma_eq_ELU", "sigma_eq_ELU"), ("b_long_mm", "b_long"), ("I_long", "I_long"),
    ("W_long", "W_long"), ("Az_long", "Az_long"), ("Vloc_ELS", "Vloc_ELS"), ("Vloc_ELU", "Vloc_ELU"),
    ("Mlong_ELS_kNm", "Mlong_ELS"), ("Mlong_ELU_kNm", "Mlong_ELU"), ("sigma_long_ELS", "sigma_long_ELS"),
    ("sigma_long_ELU", "sigma_long_ELU"), ("tau_long_ELS", "tau_long_ELS"), ("tau_long_ELU", "tau_long_ELU"),
    ("sigma_eq_long_ELS", "sigma_eq_long_ELS"), ("sigma_eq_long_ELU", "sigma_eq_long_ELU"),
    ("U_sigma", "U_sigma"), ("U_tau", "U_tau"), ("U_ELU", "U_ELU"), ("U_long_ELU", "U_long_ELU"))
# les entrées : variable de show() ↔ champ de la cornière normalisée
ENTREES = (("V_char", "V"), ("e", "e"), ("gamma_ELU", "gamma_ELU"), ("s", "s"), ("edge", "bord"), ("sig_lim_ELS", "sig_lim_ELS"),
           ("tau_lim_ELS", "tau_lim_ELS"), ("sig_lim_ELU", "sig_lim_ELU"))
VERDICTS = (("ok_els", "els"), ("ok_elu", "elu"), ("ok_long", "long"))
# ce que show() garde et qui n'est pas une grandeur : les colonnes, la liste, les aides, les choix des widgets
HORS_CALCUL = {"left", "right", "c1", "c2", "c3", "navigation", "to_m", "to_mm", "ANGLES_STD", "use_std", "sel", "charge_type"}
COMPAREES = {a for a, _ in CORRESPONDANCE} | {a for a, _ in ENTREES} | {a for a, _ in VERDICTS} | {"A", "B", "t", "kg_m", "V_unit", "L"}


def comparer(c, cle_ecran, ev, loc, mod=None):
    """La cornière c (moteur, ou ``mod`` : un mutant) contre l'écran : variables locales et affichage. Rend la liste
    des écarts."""
    ecarts = []
    res = (mod or M).verifier(c)
    g, sec = res["grandeurs"], res["section"]

    def cmp(nom, a, b):
        if not egal(a, b):
            ecarts.append(f"{nom} : moteur {a!r} ≠ écran {b!r}")
    for a, k in CORRESPONDANCE:
        cmp(k, g[k], loc[a])
    for a, k in ENTREES:
        cmp(k, res["corniere"][k], float(loc[a]))
    for a, k in VERDICTS:
        cmp(f"verdict {k}", res["verdicts"][k], loc[a] == "✅")
        cmp(f"grandeur {a}", g[a], loc[a] == "✅")
    cmp("t (section)", float(sec["t"]), float(loc["t"]))
    cmp("A (section)", float(sec["A"]), float(loc["A"]))
    cmp("B (section)", float(sec["B"]), float(loc["B"]))
    cmp("type de charge", "kN/m" if c["charge"] == "lineaire" else "kN", loc["V_unit"])
    if c["charge"] == "ponctuelle":
        cmp("L", c["L"], float(loc["L"]))
    if not c["standard"] and cle_ecran is None:
        cmp("masse en saisie libre (aucune, comme l'écran)", sec["masse"], loc["kg_m"])
    cmp("M_ELS (kN·m/m) × 10⁶ = M_ELS_Nmm", g["M_ELS"] * 1e6, loc["M_ELS_Nmm"])
    cmp("M_ELU (kN·m/m) × 10⁶ = M_ELU_Nmm", g["M_ELU"] * 1e6, loc["M_ELU_Nmm"])
    # les variables de show() : toutes comparées (une nouvelle fait rougir)
    autres = set(loc) - HORS_CALCUL - COMPAREES
    if autres:
        ecarts.append(f"variables de show() non comparées : {sorted(autres)}")
    # l'affichage
    succes = [e[1] for e in ev if e[0] == "success"]
    infos = [e[1] for e in ev if e[0] == "info"]
    v = {x["cle"]: x for x in res["verifications"]}
    ico = {True: "✅", False: "❌"}
    cmp("bandeau ELS", f"ELS : σ_eq/lim = {v['els_sigma']['taux'] * 100:.0f}% ; τ/lim = {v['els_tau']['taux'] * 100:.0f}%  {ico[res['verdicts']['els']]}",
        succes[0] if succes else None)
    cmp("bandeau ELU", f"ELU : σ_eq/lim = {v['elu']['taux'] * 100:.0f}%  {ico[res['verdicts']['elu']]}", succes[1] if len(succes) > 1 else None)
    cmp("bandeau longitudinal", f"ELU (longitudinal) : σ_eq/lim = {v['long_elu']['taux'] * 100:.0f}% {ico[res['verdicts']['long']]}", infos[-1] if infos else None)
    metr = {e[1]: e[2] for e in ev if e[0] == "metric"}
    cmp("épaisseur affichée", f"{sec['t']:.1f} mm", metr.get("Épaisseur t"))
    cmp("largeur de lame affichée", "1000 mm", metr.get("Largeur de lame b"))
    md = "\n".join(e[1] for e in ev if e[0] == "markdown")
    for txt in (f"M (ELS) = **{g['M_ELS']:.3f} kN·m/m**", f"M (ELU) = **{g['M_ELU']:.3f} kN·m/m**", f"V (ELU) = **{g['V_ELU']:.2f} kN/m**",
                f"σ_eq(ELU)=**{g['sigma_eq_ELU']:.1f}**", f"V_loc(ELU)=**{g['Vloc_ELU']:.2f} kN**", f"σ_eq_long(ELU)=**{g['sigma_eq_long_ELU']:.1f}** N/mm²"):
        if txt not in md:
            ecarts.append(f"l'écran n'affiche pas « {txt} »")
    return ecarts, res


def faits(c, res, cle_ecran):
    """Ce que le cas atteint (couverture), lu dans les sorties CONTRÔLÉES contre l'écran."""
    out = {f"charge {c['charge']}", "catalogue" if c["standard"] else "saisie libre"}
    out |= {f"verdict {k} {'✅' if x else '❌'}" for k, x in res["verdicts"].items()}
    out |= {f"{v['cle']} {v['etat']}" for v in res["verifications"]}
    if cle_ecran is not None:
        out.add("liste de l'écran")
    if res["section"]["famille"]:
        out.add(f"famille {res['section']['famille']}")
    g = res["grandeurs"]
    if not res["verdicts"]["els"]:
        out.add("ELS ❌ par " + "+".join(k for k, x in (("σ", g["sigma_eq_ELS"] > c["sig_lim_ELS"]), ("τ", g["tau_ELS"] > c["tau_lim_ELS"])) if x))
    if c["V"] == 0:
        out.add("V = 0")
    if c["e"] == 0:
        out.add("e = 0")
    if res["section"]["t"] > 20:
        out.add("t > 20 mm")
    for v in res["verifications"]:
        if v["taux"] == 1.0 and v["ok"]:
            out.add(f"borne exacte {v['cle']}")
    return out


# ------------------------------------------------------------------ 1. données : défauts et bornes LUS dans l'écran
print("=== 1. Données : les défauts et les bornes du moteur sont ceux de l'écran ===")
if MC is not None:
    ev0, loc0 = ecran({})
    w0 = ecran({})[0].widgets
    ev1, _ = ecran({L_STD: False, L_TYPE: "Ponctuelle (kN)"})
    widgets = {w[1]: w for w in ev1.widgets}
    widgets.update({w[1]: w for w in w0})
    WIDGETS = widgets

    def ecarts_bornes(E_):
        """Les défauts et bornes du module d'entrée ``E_`` face aux st.number_input de l'écran, et la section proposée."""
        out = []
        for k, (defaut, mini, maxi, _u, _l) in E_.CHAMPS.items():
            w = WIDGETS.get(LIBELLE[k])
            if w is None or w[0] != "number_input":
                out.append(f"{k} : champ « {LIBELLE[k]} » introuvable")
            elif (w[4], w[2], w[3]) != (defaut, mini, maxi):
                out.append(f"{k} : écran (défaut {w[4]}, min {w[2]}, max {w[3]}) ≠ moteur ({defaut}, {mini}, {maxi})")
        sel_ = WIDGETS.get(L_SEC)
        if sel_ is None or nom_catalogue(sel_[2][sel_[3]]) != E_.PROFIL_DEFAUT or E_.nouvelle_corniere()["profil"] != E_.PROFIL_DEFAUT:
            out.append("section proposée à l'ouverture")
        return out
    bornes = ecarts_bornes(E)
    chk(f"les {len(E.CHAMPS)} champs numériques : défaut, minimum et maximum du moteur = ceux des st.number_input de l'écran",
        not bornes, bornes)
    vpon = widgets.get(L_VPON)
    chk("le V d'une charge ponctuelle a le même défaut et la même borne que celui d'une charge linéaire",
        vpon is not None and (vpon[4], vpon[2], vpon[3]) == (E.CHAMPS["V"][0], E.CHAMPS["V"][1], E.CHAMPS["V"][2]), vpon)
    radio = widgets.get(L_TYPE)
    chk("le type de charge : les deux options de l'écran, linéaire par défaut",
        radio is not None and [RADIO[x] for x in radio[2]] == list(E.CHARGES) and E.nouvelle_corniere()["charge"] == RADIO[radio[2][radio[3]]],
        radio)
    tog = widgets.get(L_STD)
    sel = widgets.get(L_SEC)
    chk("à l'ouverture : cornière standard, la première de la liste de l'écran (« 100x100x10 », tri alphabétique) = PROFIL_DEFAUT",
        tog is not None and tog[2] is True and E.nouvelle_corniere()["standard"] is True and sel is not None
        and sel[2] == sorted(TABLE) and nom_catalogue(sel[2][sel[3]]) == E.PROFIL_DEFAUT, (tog, sel and sel[2][:3]))
    chk("la liste de l'écran relue dans sa source est celle qu'il propose (41 sections)", sel is not None and sorted(sel[2]) == sorted(TABLE) and len(TABLE) == 41)
    ecarts, _ = comparer(M.nouvelle_corniere(), "100x100x10", ev0.ev, loc0)
    chk("nouvelle_corniere() calculée = l'écran au premier affichage, au bit près", not ecarts, ecarts[:4])

# ------------------------------------------------------------------ 2. à la main
print("\n=== 2. Contrôles à la main (N et mm) ===")


def proche(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-300)


# H1 : L 100x100x10 (t = 10 mm), V = 5 kN/m, e = 2,3 cm, γ = 1,35 — bande de 1 000 mm
r1 = M.verifier(M.nouvelle_corniere())["grandeurs"]
W = 1000 * 10 ** 2 / 6                        # 16 666,7 mm³ par mètre
M_els = 5.0 * 23.0                            # N·mm/mm = 5 N/mm × 23 mm → × 1000 mm : 115 000 N·mm
sig = M_els * 1000 / W
tau = 5.0 * 1000 / (1000 * 10)
chk(f"H1 σ_ELS = V·e / (b·t²/6) = 115 000 / 16 667 = {sig:.3f} N/mm² ; τ_ELS = V/(b·t) = {tau:.3f} N/mm²",
    proche(r1["sigma_ELS"], sig) and proche(r1["tau_ELS"], tau), (r1["sigma_ELS"], r1["tau_ELS"]))
chk("H1 σ_eq,ELS = √(σ² + 3τ²) et U_τ = 0,5 / 0,8 = 62,5 % (le taux gouvernant de la cornière neuve)",
    proche(r1["sigma_eq_ELS"], math.sqrt(sig ** 2 + 3 * tau ** 2)) and r1["U_tau"] == 0.625)
sig_u, tau_u = 1.35 * sig, 1.35 * tau
chk(f"H1 ELU : σ = {sig_u:.4f}, τ = {tau_u:.4f}, σ_eq / 186 = {math.sqrt(sig_u ** 2 + 3 * tau_u ** 2) / 186:.5f}",
    proche(r1["U_ELU"], math.sqrt(sig_u ** 2 + 3 * tau_u ** 2) / 186))
chk("H1 longitudinal : sur un panneau de s = 300 mm, V_loc = 1,5 kN et σ_long,ELU = σ_ELU (la largeur s se simplifie)",
    proche(r1["Vloc_ELS"], 1.5) and proche(r1["sigma_long_ELU"], sig_u) and proche(r1["U_long_ELU"], r1["U_ELU"]))
# H2 : charge ponctuelle 8 kN répartie sur 0,6 m, L 80x80x8, e = 2,3 cm
r2 = M.verifier(corn("P", "L 80x80x8", charge="ponctuelle", V=8.0, L=0.6))["grandeurs"]
q = 8.0 / 0.6                                  # kN/m = N/mm
chk(f"H2 ponctuelle : V_ELS = 8 / 0,6 = {q:.4f} kN/m, V_ELU = 1,35 × 8 / 0,6, σ_ELS = q·e / (t²/6) = {q * 23 / (64 / 6):.3f} N/mm²",
    proche(r2["V_ELS"], q) and proche(r2["V_ELU"], 1.35 * 8 / 0.6) and proche(r2["sigma_ELS"], q * 23 / (8 ** 2 / 6)), r2["sigma_ELS"])

# ------------------------------------------------------------------ 3. parité
COUV = set()
ECRANS = []        # (cornière, section de la liste, affichage, variables de show()) : l'écran rejoué, gardé pour les mutants
SELECTEUR_ECRAN = {}   # cornière du catalogue → (taux, ok) de l'écran, cornière neuve (défauts)
if MC is not None:
    print("\n=== 3. Parité avec modules/corniere.py, cas par cas (calcul + affichage), au bit près ===")
    _, loc = ecran({})
    rendues = set(loc) - HORS_CALCUL
    chk(f"chacune des {len(rendues)} variables de show() (hors colonnes et widgets) est comparée au moteur",
        rendues <= COMPAREES and COMPAREES - rendues <= {"L"}, sorted(rendues ^ COMPAREES))
    for k, (desc, c, cle_ecran) in enumerate(CAS):
        ev, loc = ecran(saisies_de(c, cle_ecran))
        ECRANS.append((c, cle_ecran, ev.ev, loc))
        ecarts, res = comparer(c, cle_ecran, ev.ev, loc)
        COUV |= faits(c, res, cle_ecran)
        chk(f"cas {k + 1:02d} · {desc}", not ecarts, ecarts[:4])
    chk(f"{len(CAS)} cornières comparées : au moins 80", len(CAS) >= 80, len(CAS))
    print(f"     {ECART['n']} comparaisons numériques : {ECART['bit']} identiques au bit près")
    chk("toutes les comparaisons numériques sont identiques AU BIT PRÈS (c'est le même calcul)", ECART["n"] == ECART["bit"] and ECART["n"] > 4000,
        (ECART["n"], ECART["bit"]))

    print("\n=== 4. Couverture : ce que les cas atteignent réellement ===")
    ATTENDUS = (["charge lineaire", "charge ponctuelle", "catalogue", "saisie libre", "liste de l'écran", "famille L", "famille LI",
                 "V = 0", "e = 0", "t > 20 mm", "ELS ❌ par σ", "ELS ❌ par τ", "ELS ❌ par σ+τ"]
                + [f"verdict {k} {x}" for k in ("els", "elu", "long") for x in ("✅", "❌")]
                + [f"{k} {e}" for k in ("els_sigma", "els_tau", "elu", "long_elu") for e in ("ok", "nok")]
                + ["borne exacte els_tau", "borne exacte els_sigma", "borne exacte elu"])
    manque = [a for a in ATTENDUS if a not in COUV]
    chk(f"les cas atteignent les {len(ATTENDUS)} situations attendues (verdicts, charges, bornes exactes…)", not manque, manque)

# ------------------------------------------------------------------ 5. la liste de l'écran face au catalogue
print("\n=== 5. La liste de l'écran (ANGLES_STD) face au catalogue acier/profiles ===")
absentes = [k for k in TABLE if nom_catalogue(k) not in CAT]
ecarts_masse = {k: TABLE[k]["kg_m"] / M.section_catalogue(nom_catalogue(k))["masse"] - 1 for k in TABLE if nom_catalogue(k) in CAT}
fausses = sorted(k for k, x in ecarts_masse.items() if abs(x) > 0.05)
doc = lire("docs", "acier", "CORNIERE_MOTEUR.md")
chk(f"{len(absentes)} sections de la liste de l'écran sont absentes du catalogue, toutes nommées dans la doc ({', '.join(absentes)})",
    all(f"`{k}`" in doc for k in absentes) and len(absentes) == 10, [k for k in absentes if f"`{k}`" not in doc])
chk(f"masses de la liste de l'écran : {len(ecarts_masse)} présentes au catalogue, écart courant +1,7 à +2,4 % ; "
    f"fausse(s) de plus de 5 % : {fausses} — écrite(s) dans la doc",
    fausses == ["90x90x10"] and "18,54" in doc and all(-0.001 < x < 0.025 for k, x in ecarts_masse.items() if k not in fausses),
    {k: round(x, 4) for k, x in ecarts_masse.items()})
chk("toutes les sections de la liste de l'écran ont t ≤ 20 mm et A, B dans les bornes de la saisie libre (10 à 300 mm)",
    all(2 <= v["t"] <= 20 and 10 <= v["A"] <= 300 and 10 <= v["B"] <= 300 for v in TABLE.values()))

# ------------------------------------------------------------------ 6. sélecteur
print("\n=== 6. Sélecteur : chaque cornière du catalogue repassée par l'écran ===")
for desc, c in (("défauts", M.nouvelle_corniere()), ("ponctuelle forte, S355", corn("X", "L 60x60x6", charge="ponctuelle", V=30.0, L=0.5, e=6.0,
                                                                                         sig_lim_ELS=355.0, tau_lim_ELS=205.0, sig_lim_ELU=355.0))):
    cands = M.candidats(c)
    ecarts = []
    if MC is not None:
        for x in cands:
            s = saisies_de(c)                       # l'écran en saisie libre, aux dimensions de la cornière candidate
            s.update({L_A: x["A"], L_B: x["B"], L_T: x["t"]})
            _, loc = ecran(s)
            taux = max(loc["U_sigma"], loc["U_tau"], loc["U_ELU"], loc["U_long_ELU"])
            ok = loc["ok_els"] == loc["ok_elu"] == loc["ok_long"] == "✅"
            if desc == "défauts":
                SELECTEUR_ECRAN[x["nom"]] = (taux, ok)
            if not (egal(x["taux"], taux) and x["ok"] == ok):
                ecarts.append(f"{x['nom']} : {x['taux']!r}/{x['ok']} ≠ {taux!r}/{ok}")
    masses = [x["masse"] for x in cands]
    premier = next((x for x in cands if x["ok"] and x["B"] >= c["e"] * 10.0), None)
    marques = (sum(x["optimal"] for x in cands) == (premier is not None) and (premier is None or premier["optimal"])
               and all(x["aile"] == (x["B"] >= c["e"] * 10.0) for x in cands)
               and [x["nom"] for x in cands if x["retenue"]] == ([c["profil"]] if c["standard"] else []))
    chk(f"sélecteur ({desc}) : {len(cands)} cornières L et LI triées par masse, taux et verdicts = l'écran ; "
        f"la plus légère qui tient = {premier and premier['nom']}",
        not ecarts and masses == sorted(masses) and len(cands) == len(M.cornieres()) and marques and
        {x["famille"] for x in cands} == {"L", "LI"}, (ecarts[:3], marques))
C_HORS = corn("H", "L 100x100x10", e=8.0)          # e = 80 mm : les petites cornières qui « tiennent » ont B < 80 mm
CANDS_HORS = M.candidats(C_HORS)
leg = next(x for x in CANDS_HORS if x["ok"])
opt_h = M.optimal(C_HORS)
chk(f"e = 80 mm : la plus légère qui tient à l'écran ({leg['nom']}, B = {leg['B']:g} mm) n'est PAS proposée, la charge tomberait "
    f"au-delà de son aile ; l'optimal est {opt_h and opt_h['nom']} (B ≥ e)",
    leg["B"] < 80 and not leg["optimal"] and not leg["aile"] and opt_h is not None and opt_h["B"] >= 80 and opt_h["ok"]
    and all(not (x["ok"] and x["aile"]) for x in CANDS_HORS[:CANDS_HORS.index(opt_h)]), (leg["nom"], opt_h and opt_h["nom"]))
chk("candidats(familles=['LI']) : les seules cornières à ailes inégales",
    {x["famille"] for x in M.candidats(M.nouvelle_corniere(), ["LI"])} == {"LI"})
chk("aucune cornière ne tient : aucun optimal (V = 500 kN/m, e = 25 cm)",
    not any(x["optimal"] for x in M.candidats(corn("Z", "L 60x60x6", V=500.0, e=25.0))) and M.optimal(corn("Z", "L 60x60x6", V=500.0, e=25.0)) is None)

# ------------------------------------------------------------------ 7. format
print("\n=== 7. Format : JSON, normalisation, domaine de l'écran, formules ===")
tous = [M.verifier(c) for _, c, _ in CAS]
chk("la sortie de verifier() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(r, allow_nan=False)) == r for r in tous))
chk("« -0 » est relu 0 (le champ n'écrit jamais « -0 »)", math.copysign(1, M.normaliser({"V": -0.0})["V"]) == 1.0)
chk("normaliser() est idempotente et {} donne nouvelle_corniere()",
    all(M.normaliser(M.normaliser(c)) == M.normaliser(c) for _, c, _ in CAS) and M.normaliser({}) == M.nouvelle_corniere())
entree = copy.deepcopy(CAS[5][1])
M.verifier(entree)
M.candidats(entree)
chk("verifier() et candidats() ne modifient pas la cornière reçue", entree == CAS[5][1])
chk("les libellés de l'écran pour le type de charge sont admis (même sens)",
    M.normaliser({"charge": "Ponctuelle (kN)"})["charge"] == "ponctuelle" and M.normaliser({"charge": "Linéaire (kN/m)"})["charge"] == "lineaire")
REFUS = (("aile A 9,9 mm", {"A": 9.9}), ("aile B 300,5 mm", {"B": 300.5}), ("t 1,9 mm", {"t": 1.9}), ("t 20,5 mm", {"t": 20.5}),
         ("V négatif", {"V": -0.1}), ("L 0,05 m", {"L": 0.05}), ("L 5,1 m", {"L": 5.1}), ("e négative", {"e": -0.1}),
         ("γ 0,99", {"gamma_ELU": 0.99}), ("s 9,9 cm", {"s": 9.9}), ("bord négatif", {"bord": -1}), ("σ_lim,ELS 9,9", {"sig_lim_ELS": 9.9}),
         ("τ_lim,ELS 0,09", {"tau_lim_ELS": 0.09}), ("σ_eq,lim,ELU 9", {"sig_lim_ELU": 9}), ("V en texte", {"V": "5"}),
         ("t booléen", {"t": True}), ("e infinie", {"e": float("inf")}), ("s indéterminé", {"s": float("nan")}), ("clé inconnue", {"epaisseur": 6}),
         ("profil inconnu", {"profil": "L 99x99x9"}), ("profil qui n'est pas une cornière", {"profil": "IPE 200"}),
         ("type de charge « répartie »", {"charge": "repartie"}), ("standard « oui »", {"standard": "oui"}), ("nom numérique", {"nom": 3}),
         ("épaisseur ≥ aile A (saisie libre)", {"standard": False, "A": 12, "t": 12}),
         ("épaisseur ≥ aile B", {"standard": False, "B": 15, "t": 16}),
         ("calcul qui déborde (V = 1e200)", {"V": 1e200}), ("calcul qui déborde (e = 1e300)", {"e": 1e300}),
         ("panneau sans fin (s = 1e308)", {"s": 1e308}), ("γ = 1e308", {"gamma_ELU": 1e308}))


def refus_manques(M_):
    """Les saisies hors du domaine que le moteur ``M_`` n'a PAS refusées par un ValueError."""
    out = []
    for nom, champ in REFUS:
        try:
            M_.verifier(dict(M_.nouvelle_corniere(), **champ))
            out.append(f"{nom} : accepté")
        except ValueError:
            pass
        except Exception as e:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
            out.append(f"{nom} : {type(e).__name__} {e}")
    return out


refusees = refus_manques(M)
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError (jamais de substitution silencieuse)", not refusees, refusees)
for nom, champ in (("aile A 10 mm", {"A": 10}), ("aile 300 mm", {"B": 300}), ("t 2 mm", {"t": 2}), ("t 20 mm", {"t": 20}), ("V nul", {"V": 0}),
                   ("L 0,1 m", {"L": 0.1}), ("L 5 m", {"L": 5}), ("γ 1", {"gamma_ELU": 1}), ("s 10 cm", {"s": 10}), ("τ_lim 0,1", {"tau_lim_ELS": 0.1}),
                   ("t juste sous l'aile (A = 10,5, t = 10)", {"standard": False, "A": 10.5, "t": 10}), ("V = 1e12 (grand, fini)", {"V": 1e12})):
    try:
        M.verifier(dict(M.nouvelle_corniere(), **champ))
        chk(f"borne admise comme à l'écran : {nom}", True)
    except ValueError as e:
        chk(f"borne admise comme à l'écran : {nom}", False, e)
# « formule = valeurs = résultat » : chaque résultat se retrouve à partir des seuls termes que le moteur donne
FORMULES = {
    "V_ELS_lin": lambda t: t["V"],
    "V_ELU_lin": lambda t: t["V"] * t["γ_ELU"],
    "V_ELS_ponct": lambda t: t["V"] / t["L"],
    "V_ELU_ponct": lambda t: (t["V"] * t["γ_ELU"]) / t["L"],
    "M": lambda t: list(t.values())[0] * (t["e"] / 100.0),
    "I": lambda t: t["b"] * t["t"] ** 3 / 12.0,
    "W": lambda t: t["b"] * t["t"] ** 2 / 6.0,
    "Az": lambda t: t["b"] * t["t"],
    "sigma": lambda t: t["M"] * 1e6 / t["W"],
    "tau": lambda t: t["V"] * 1e3 / t["A_z"],
    "sigma_eq": lambda t: math.sqrt(t["σ"] ** 2 + 3.0 * t["τ"] ** 2),
    "taux": lambda t: list(t.values())[0] / list(t.values())[1],
    "b_long": lambda t: t["s"] * 10.0,
    "Vloc": lambda t: list(t.values())[0] * (t["s"] / 100.0),
    "Mlong": lambda t: list(t.values())[0] * (t["e"] / 100.0),
    "sigma_long": lambda t: t["M_long"] * 1e6 / t["W_long"],
    "tau_long": lambda t: t["V_loc"] * 1e3 / t["A_z,long"],
}


def fautes_formules(resultats):
    """Les formules dont le résultat ne se retrouve pas à partir de leurs seuls termes, et les vérifications dont
    valeur / limite / taux / ok ne se tiennent pas ; rend (fautes, nombre de formules)."""
    out, n = [], 0
    for res in resultats:
        c = res["corniere"]
        for groupe, blocs in res["calcul"].items():
            for b in blocs:
                for f in b["formules"]:
                    n += 1
                    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
                    if f["cle"] not in FORMULES or FORMULES[f["cle"]](t) != f["resultat"]:
                        out.append(f"{c['nom']}/{groupe}/{f['symbole']} : {f['resultat']!r}")
        v = {x["cle"]: x for x in res["verifications"]}
        g = res["grandeurs"]
        if not (v["els_sigma"]["valeur"] == g["sigma_eq_ELS"] and v["els_tau"]["valeur"] == g["tau_ELS"] and v["elu"]["valeur"] == g["sigma_eq_ELU"]
                and v["long_elu"]["valeur"] == g["sigma_eq_long_ELU"] and all(x["taux"] == x["valeur"] / x["limite"] for x in v.values())
                and all(x["ok"] == (x["valeur"] <= x["limite"]) for x in v.values())):
            out.append(f"{c['nom']} : valeur / limite / taux / ok des vérifications")
        gov = max(res["verifications"], key=lambda x: x["taux"])
        if not (res["gouvernant"]["cle"] == gov["cle"] and res["taux_max"] == gov["taux"]
                and res["statut"] == ("ok" if all(x["ok"] for x in res["verifications"]) else "nok")
                and (res["statut"] == "ok") == all(res["verdicts"].values())):
            out.append(f"{c['nom']} : gouvernant, taux maximal ou statut")
    return out, n


fautes, n_f = fautes_formules(tous)
chk(f"« formule = valeurs = résultat » : les {n_f} formules se recalculent AU BIT PRÈS à partir de leurs seuls termes ; "
    "valeur / limite = taux, ok = (valeur ≤ limite)", not fautes and n_f > 2000, fautes[:4])
chk("le gouvernant est la vérification au plus grand taux (la première de l'ordre de l'écran à égalité) ; statut nok dès qu'une ne tient pas",
    all(r["gouvernant"]["cle"] == max(r["verifications"], key=lambda x: x["taux"])["cle"] and r["taux_max"] == max(x["taux"] for x in r["verifications"])
        and r["statut"] == ("ok" if all(x["ok"] for x in r["verifications"]) else "nok") for r in tous))
chk("statut ok ⇔ les trois ✅ de l'écran", all((r["statut"] == "ok") == all(r["verdicts"].values()) for r in tous))
cles = sorted(tous[0]["grandeurs"])
sans_doc = [k for k in cles if f"`{k}`" not in doc or k not in M.__doc__]
chk(f"les {len(cles)} grandeurs sont nommées dans la docstring du moteur ET dans docs/acier/CORNIERE_MOTEUR.md", not sans_doc, sans_doc)

ESSAIS_NOMS = (("L 80×80×8", "L 80x80x8"), ("80x80x8", "L 80x80x8"), ("l80*80*8", "L 80x80x8"), (" 100 x 65 x 8 ", "L 100x65x8"),
               ("65x100x8", "L 100x65x8"), ("L 45×45×4,5", "L 45x45x4.5"), ("L 45x45x4.5", "L 45x45x4.5"), ("L 99x99x9", None),
               ("IPE 200", None), ("", None), (None, None), ("80x80", None))
mauvais_noms = [(x, E.nom_catalogue(x), y) for x, y in ESSAIS_NOMS if E.nom_catalogue(x) != y]
chk(f"une désignation tapée se relit au catalogue ({len(ESSAIS_NOMS)} essais : ×, *, espaces, ailes inversées, virgule ; inconnue → None)",
    not mauvais_noms, mauvais_noms)
chk("la désignation du bureau : « L 45×45×4,5 » (signe ×, virgule) ; chaque cornière du catalogue se relit depuis sa désignation",
    M.designation("L 45x45x4.5") == "L 45×45×4,5" and all(E.nom_catalogue(M.designation(p["nom"])) == p["nom"] for p in M.cornieres()))
hors = [(r["corniere"]["nom"], r["corniere"]["e"], r["section"]["B"]) for r in tous
        if ("excentricite_hors_aile" in {a["code"] for a in r["avertissements"]}) != (r["corniere"]["e"] * 10.0 > r["section"]["B"])]
n_hors = sum("excentricite_hors_aile" in {a["code"] for a in r["avertissements"]} for r in tous)
chk(f"avertissement « charge au-delà de l'aile B » ⇔ e > B ({n_hors} cornières sur {len(tous)}) ; un constat, aucun chiffre ne change "
    "(les grandeurs sont celles de l'écran, contrôlées plus haut)", not hors and 0 < n_hors < len(tous), hors[:3])
chk("chaque résultat porte les deux avertissements de l'écran : ancrages non vérifiés, longitudinal provisoire",
    all({"ancrages_non_verifies", "longitudinal_provisoire"} <= {a["code"] for a in r["avertissements"]} for r in tous))

# ------------------------------------------------------------------ 8. pureté
print("\n=== 8. Pureté : le moteur se charge et calcule SANS streamlit ===")
non_std = []
for dossier, _, fichiers in os.walk(os.path.join(RACINE, "acier", "corniere")):
    for f in fichiers:
        if f.endswith(".py"):
            arbre = ast.parse(lire(os.path.relpath(os.path.join(dossier, f), RACINE)))
            for n in ast.walk(arbre):
                noms = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                    [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
                non_std += [f"{f}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names and not x.startswith("acier.profiles")]
chk("acier/corniere n'importe que la bibliothèque standard et le catalogue acier.profiles (lecture des sources)", not non_std, non_std)
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
    import streamlit  # noqa: F401
    refus = False
except ImportError:
    refus = True
sys.path.insert(0, sys.argv[1])
from acier.corniere import moteur as M
cs = json.loads(sys.stdin.read())
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": [M.verifier(c) for c in cs], "cands": M.candidats(cs[0])}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"acier"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
cs = [c for _, c, _ in CAS]
attendu = {"res": json.loads(json.dumps(tous)), "cands": json.loads(json.dumps(M.candidats(cs[0])))}
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(cs), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, pandas, reportlab… REFUSÉS par sys.meta_path, et le moteur calcule quand même ({len(out['res'])} cornières, "
        "le sélecteur)", out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : seuls la bibliothèque standard et numpy (le catalogue acier/profiles) sont chargés", set(out["tiers"]) <= {"numpy"}, out["tiers"])
    diff = [i for i, (a, b) in enumerate(zip(out["res"], attendu["res"])) if a != b] + (["cands"] if out["cands"] != attendu["cands"] else [])
    chk(f"Python {v} : résultats identiques à ceux de ce processus", not diff, diff[:3])
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

# ------------------------------------------------------------------ 9. mutations : chaque contrôle rougit si l'on retire ce qu'il protège
print("\n=== 9. Mutations : le moteur modifié d'UNE ligne est rejeté par les contrôles ci-dessus ===")
PAQUET = sys.modules["acier.corniere"]


def charger_mutant(mutations):
    """Le moteur (entree + moteur) rechargé depuis ses SOURCES, après des remplacements [(fichier, avant, après)] ;
    rend (E, M) mutants. Chaque « avant » doit figurer UNE fois dans la source (sinon la mutation est périmée)."""
    noms = ("entree", "moteur")
    sauve = {n: sys.modules[f"acier.corniere.{n}"] for n in noms}
    sauve_attr = {n: getattr(PAQUET, n) for n in noms}
    mods = {}
    try:
        for n in noms:
            chemin = os.path.join(RACINE, "acier", "corniere", n + ".py")
            src = lire("acier", "corniere", n + ".py")
            for fichier, avant, apres in mutations:
                if fichier == n:
                    if src.count(avant) != 1:
                        raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {n}.py)")
                    src = src.replace(avant, apres)
            m = types.ModuleType(f"acier.corniere.{n}")
            m.__package__, m.__file__ = "acier.corniere", chemin
            sys.modules[m.__name__] = m
            setattr(PAQUET, n, m)
            exec(compile(src, chemin, "exec"), m.__dict__)
            mods[n] = m
    finally:
        sys.modules.update({f"acier.corniere.{n}": sauve[n] for n in noms})
        for n in noms:
            setattr(PAQUET, n, sauve_attr[n])
    return mods["entree"], mods["moteur"]


def rejets(E_, M_):
    """Ce que les contrôles de ce fichier reprochent au moteur (E_, M_) : bornes et défauts lus dans l'écran, parité
    avec l'écran rejoué (les 99 cornières), refus, formules et statuts, sélecteur. Rend la liste des reproches."""
    out = [f"bornes : {x}" for x in ecarts_bornes(E_)]
    res_ = []
    for c, cle_ecran, ev, loc in ECRANS:
        try:
            ecarts, r = comparer(c, cle_ecran, ev, loc, M_)
            res_.append(r)
            out += [f"parité {c['nom']} : {x}" for x in ecarts[:1]]
        except Exception as e:  # noqa: BLE001 — un mutant qui casse est rejeté
            out.append(f"parité {c['nom']} : {type(e).__name__}")
    out += [f"refus : {x}" for x in refus_manques(M_)]
    out += [f"formules : {x}" for x in fautes_formules(res_)[0][:2]]
    try:
        cands = M_.candidats(M_.nouvelle_corniere())
        masses = [x["masse"] for x in cands]
        premier = next((x for x in cands if x["ok"]), None)
        if masses != sorted(masses) or not (premier and premier["optimal"]) or sum(x["optimal"] for x in cands) != 1:
            out.append("sélecteur : tri ou optimal")
        o_h = M_.optimal(C_HORS)
        if not (o_h and o_h["B"] >= C_HORS["e"] * 10.0 and o_h["nom"] == opt_h["nom"]):
            out.append("sélecteur : optimal dont l'aile B ne reçoit pas la charge")
        out += [f"sélecteur {x['nom']}" for x in cands
                if x["nom"] in SELECTEUR_ECRAN and (not egal(x["taux"], SELECTEUR_ECRAN[x["nom"]][0]) or x["ok"] != SELECTEUR_ECRAN[x["nom"]][1])][:1]
    except Exception as e:  # noqa: BLE001
        out.append(f"sélecteur : {type(e).__name__}")
    return out


MUTATIONS = (
    ("I = b·t³/12 → /12,5", [("moteur", "I_trans = b_trans_mm * t_mm**3 / 12.0", "I_trans = b_trans_mm * t_mm**3 / 12.5")]),
    ("W = b·t²/6 → /6,5", [("moteur", "W_trans = b_trans_mm * t_mm**2 / 6.0", "W_trans = b_trans_mm * t_mm**2 / 6.5")]),
    ("A_z = b·t → b·t/2", [("moteur", "Az_trans = b_trans_mm * t_mm\n", "Az_trans = b_trans_mm * t_mm / 2\n")]),
    ("bande de 1 000 mm → 1 000,5", [("moteur", "B_TRANS = 1000.0", "B_TRANS = 1000.5")]),
    ("γ_ELU oublié (linéaire)", [("moteur", "V_ELU = V_char * gamma_ELU\n", "V_ELU = V_char\n")]),
    ("charge ponctuelle non répartie sur L", [("moteur", "V_ELS = V_char / L", "V_ELS = V_char")]),
    ("e en cm lu en dm", [("moteur", "e_m = e / 100.0", "e_m = e / 10.0")]),
    ("√(σ² + 3τ²) → √(σ² + τ²)", [("moteur", "sigma_eq_ELS = math.sqrt(sigma_ELS**2 + 3.0 * tau_ELS**2)", "sigma_eq_ELS = math.sqrt(sigma_ELS**2 + tau_ELS**2)")]),
    ("τ de l'ELU dans σ_eq,ELU → celui de l'ELS", [("moteur", "sigma_eq_ELU = math.sqrt(sigma_ELU**2 + 3.0 * tau_ELU**2)", "sigma_eq_ELU = math.sqrt(sigma_ELU**2 + 3.0 * tau_ELS**2)")]),
    ("panneau b = 10·s → 100·s", [("moteur", "b_long_mm = s * 10.0", "b_long_mm = s * 100.0")]),
    ("V_loc = V·s sans /100", [("moteur", "Vloc_ELU = V_ELU * (s / 100.0)", "Vloc_ELU = V_ELU * s")]),
    ("σ_eq,long avec τ_long,ELS", [("moteur", "sigma_eq_long_ELU = math.sqrt(sigma_long_ELU**2 + 3.0 * tau_long_ELU**2)", "sigma_eq_long_ELU = math.sqrt(sigma_long_ELU**2 + 3.0 * tau_long_ELS**2)")]),
    ("U_τ avec τ_ELU", [("moteur", "U_tau = tau_ELS / tau_lim_ELS", "U_tau = tau_ELU / tau_lim_ELS")]),
    ("ELS : « ≤ » devenu « < »", [("moteur", "ok_els = sigma_eq_ELS <= sig_lim_ELS and tau_ELS <= tau_lim_ELS", "ok_els = sigma_eq_ELS < sig_lim_ELS and tau_ELS < tau_lim_ELS")]),
    ("ELS sans le contrôle de τ", [("moteur", "ok_els = sigma_eq_ELS <= sig_lim_ELS and tau_ELS <= tau_lim_ELS", "ok_els = sigma_eq_ELS <= sig_lim_ELS")]),
    ("longitudinal comparé à σ_lim,ELS", [("moteur", "ok_long = sigma_eq_long_ELU <= sig_lim_ELU", "ok_long = sigma_eq_long_ELU <= sig_lim_ELS")]),
    ("épaisseur du catalogue remplacée par l'aile", [("moteur", '"A": p["h"], "B": p["b"], "t": p["t"]', '"A": p["h"], "B": p["b"], "t": p["b"] / 10')]),
    ("ailes A et B échangées", [("moteur", '"A": p["h"], "B": p["b"], "t": p["t"]', '"A": p["b"], "B": p["h"], "t": p["t"]')]),
    ("valeur de la vérification τ : τ_ELU", [("moteur", '"els_tau": (g["tau_ELS"]', '"els_tau": (g["tau_ELU"]')]),
    ("terme faux dans la formule de W", [("moteur", '_f("W", "W", [("b", b, "mm"), ("t", t, "mm")]', '_f("W", "W", [("b", b, "mm"), ("t", t + 1, "mm")]')]),
    ("gouvernant : le plus petit taux", [("moteur", 'gov = max(ver, key=lambda v: v["taux"])', 'gov = min(ver, key=lambda v: v["taux"])')]),
    ("sélecteur trié par nom", [("entree", 'key=lambda p: (p["masse"], p["nom"])', 'key=lambda p: (p["nom"], p["masse"])')]),
    ("optimal : la plus LOURDE qui tient", [("moteur", 'premier = next((x for x in out if x["ok"] and x["aile"]), None)', 'premier = next((x for x in reversed(out) if x["ok"] and x["aile"]), None)')]),
    ("optimal : l'aile B oubliée (la charge peut tomber dans le vide)", [("moteur", 'premier = next((x for x in out if x["ok"] and x["aile"]), None)', 'premier = next((x for x in out if x["ok"]), None)')]),
    ("épaisseur ≥ aile admise", [("entree", "    if out[\"t\"] >= aile:", "    if out[\"t\"] > 1e9:")]),
    ("calcul qui déborde admis", [("moteur", "        if not fini:\n", "        if False:\n")]),
    ("défaut de e : 2,5 cm", [("entree", '"e": (2.3, 0.0, None', '"e": (2.5, 0.0, None')]),
    ("borne haute de t : 25 mm", [("entree", '"t": (6.0, 2.0, 20.0', '"t": (6.0, 2.0, 25.0')]),
    ("section proposée : L 20x20x3", [("entree", 'PROFIL_DEFAUT = "L 100x100x10"', 'PROFIL_DEFAUT = "L 20x20x3"')]),
    ("une saisie négative ramenée à la borne au lieu d'être refusée", [("entree", "        v = _nombre(k, v)", "        v = max(_nombre(k, v) if not (isinstance(v, (int, float)) and v < CHAMPS[k][1]) else CHAMPS[k][1], CHAMPS[k][1] or 0.0)")]),
)
TEMOIN = ("témoin : un commentaire changé", [("moteur", "# Taux\n", "# Taux (témoin)\n")])
if ECRANS:
    survivants, n_rejets = [], []
    for nom, muts in MUTATIONS:
        try:
            E_m, M_m = charger_mutant(muts)
            r = rejets(E_m, M_m)
        except AssertionError as e:
            r, survivants = [], survivants + [f"{nom} : {e}"]
            continue
        n_rejets.append(len(r))
        if not r:
            survivants.append(nom)
    chk(f"{len(MUTATIONS)} mutations volontaires du moteur (formules, coefficients, unités, « ≤ », gouvernant, sélecteur, défauts, "
        f"bornes, refus) : TOUTES rejetées (de {min(n_rejets or [0])} à {max(n_rejets or [0])} reproches chacune)", not survivants, survivants)
    E_t, M_t = charger_mutant(TEMOIN[1])
    chk("la mutation témoin (un commentaire) n'est rejetée par aucun contrôle : les rejets viennent bien des mutations", not rejets(E_t, M_t))
    chk("après les mutations, le moteur chargé est intact (mêmes résultats)", [M.verifier(c) for _, c, _ in CAS] == tous)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

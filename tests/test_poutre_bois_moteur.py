# -*- coding: utf-8 -*-
"""Moteur « Poutre bois » (bois/poutre) : la PARITÉ avec l'écran Streamlit modules/poutre_bois.py, qui reste la
référence. Chaque contrôle rougit si l'on retire ce qu'il protège.

  1. Données : la base des classes, les sections, γ_M, k_mod, k_def sont ceux du module (ses dictionnaires) ; les
     menus, leurs options et leurs défauts, les bornes et défauts de chaque champ numérique sont LUS dans son écran
     (un faux « st » enregistre chaque widget) et comparés à bois/poutre/donnees.py.
  2. À la main : deux poutres recalculées ici, en N et en mm, sans le moteur.
  3. Parité (plus de 700 poutres : chaque classe × service × durée, chaque section standard, sections sur mesure,
     bornes de chaque champ, appui coché, tirages au sort) : l'écran est EXÉCUTÉ avec un faux « st » qui sert les
     saisies par libellé et enregistre ce qu'il affiche ; on compare
       a) chaque grandeur du calcul (les variables locales de show(), relevées à son retour) au BIT près ;
       b) ce que l'écran AFFICHE (métriques, lignes ELU/ELS, pastilles vertes ou rouges, légende) au texte près,
          reconstruit à partir des seules valeurs du moteur.
  4. Couverture : chaque vérification est atteinte conforme ET non conforme, le taux exactement égal à 100 % (borne
     ≤), k_h au-dessus, à et sous 1, l'appui coché et décoché, chaque classe, service, durée, limite et section.
  5. Mutations : douze altérations du moteur (γ_M, k_h « corrigé », k_cr, appui à l'ELU, flèche finale EC5, unité
     de la flèche, ordre des opérations…) font chacune rougir la parité.
  6. Sélecteur de sections : chaque candidate est repassée par l'écran ; tri par aire, retenue, plus légère qui tient.
  7. Format : JSON strict, normalisation idempotente, ValueError hors du domaine de l'écran (jamais de substitution).
  8. Courbes des dessins : leurs maxima sont ceux du calcul.
  9. Pureté : le moteur se charge et calcule dans un sous-processus où sys.meta_path REFUSE streamlit, numpy,
     pandas… et n'y charge que la bibliothèque standard (Python 3.12, et 3.13 — celui de Pyodide — s'il est là).

    /usr/bin/python3.12 tests/test_poutre_bois_moteur.py       (streamlit et numpy requis pour la parité)
"""
import ast
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from bois.poutre import diagrammes as DG  # noqa: E402
from bois.poutre import donnees as D  # noqa: E402
from bois.poutre import entree as E  # noqa: E402
from bois.poutre import moteur as M  # noqa: E402

OK, KO = [], []
T0 = time.time()


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as fh:
        return fh.read()


try:
    logging.disable(logging.WARNING)
    import modules.navigation as NAV  # noqa: E402
    import modules.poutre_bois as PB  # noqa: E402
except ImportError as e:  # pragma: no cover — la parité EXIGE le module
    chk("streamlit, numpy et modules/poutre_bois.py se chargent (indispensables à la parité)", False, e)
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    sys.exit(1)

# ------------------------------------------------------------------ le faux st
# les libellés des champs de l'écran (copiés de modules/poutre_bois.py : un libellé changé là-bas fait rougir ici)
L_CLASSE, L_SC, L_DUREE = "Classe de résistance", "Classe de service", "Durée dominante"
L_STD, L_SEC, L_B, L_H, L_L = "Section **standard**", "Section (mm)", "Largeur b (mm)", "Hauteur h (mm)", "Portée L (m)"
L_G, L_Q, L_GG, L_GQ, L_PSI = "Charge permanente G (kN/m)", "Charge d’exploitation Q (kN/m)", "γ_G", "γ_Q", "ψ₂ (quasi-permanent)"
L_LI, L_LF, L_APP, L_A = "Limite instantanée", "Limite finale", "Compression ⟂ fil à l’appui", "Largeur d’appui a (mm)"
NUMERIQUES = {"b": L_B, "h": L_H, "L": L_L, "G": L_G, "Q": L_Q, "gamma_G": L_GG, "gamma_Q": L_GQ, "psi2": L_PSI, "a": L_A}


class Colonne:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def metric(self, label, value, *a, **k):
        self.st.ev.append(("metric", label, value))


class FauxSt:
    """Un « st » qui SERT les saisies d'une poutre (par libellé de champ) et ENREGISTRE ce que l'écran afficherait.
    Les bornes ne sont pas appliquées ici : le moteur les applique (normaliser), le test le vérifie (§ 7)."""
    AFFICHAGES = ("header", "subheader", "markdown", "write", "success", "error", "caption", "info", "warning")

    def __init__(self, saisies):
        self.saisies = dict(saisies)
        self.ev, self.widgets = [], []

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

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, disabled=False, **_):
        self.widgets.append(("number_input", label, min_value, max_value, value, step, disabled))
        return self.saisies.get(label, value)


def ecran(saisies):
    """show() exécuté avec un faux st : (le faux st — affichage et widgets —, les variables locales de show())."""
    faux, pris = FauxSt(saisies), {}
    code = PB.show.__code__

    def trace(frame, event, arg):
        if event == "call":
            return trace if frame.f_code is code else None
        if event == "return" and frame.f_code is code:
            pris.update(frame.f_locals)
        return trace
    vrai, vrai_nav = PB.st, NAV.st
    PB.st, NAV.st = faux, faux
    sys.settrace(trace)
    try:
        PB.show()
    finally:
        sys.settrace(None)
        PB.st, NAV.st = vrai, vrai_nav
    return faux, pris


def saisies_de(p):
    """Les saisies de l'écran pour la poutre p (format du moteur)."""
    s = {L_CLASSE: p["classe"], L_SC: p["service"], L_DUREE: p["duree"], L_STD: p["standard"], L_LI: p["lim_inst"],
         L_LF: p["lim_fin"], L_APP: p["appui"]}
    if p["standard"]:
        s[L_SEC] = p["section"]
    for k, lab in NUMERIQUES.items():
        s[lab] = p[k]
    return s


# ------------------------------------------------------------------ 1. données
print("=== 1. Données : la base, les sections, les coefficients, les menus et les champs de l'écran ===")
chk("la base des classes (TIMBER_BDD) : mêmes classes, mêmes valeurs, mêmes types, même ordre",
    D.CLASSES == PB.TIMBER_BDD and list(D.CLASSES) == list(PB.TIMBER_BDD)
    and all(type(D.CLASSES[c][k]) is type(PB.TIMBER_BDD[c][k]) for c in D.CLASSES for k in D.CLASSES[c]))
chk("les sections standard (SECTIONS_STD) : mêmes étiquettes, mêmes dimensions entières, même ordre", D.SECTIONS_STD == PB.SECTIONS_STD
    and all(type(a["b"]) is type(b["b"]) and type(a["h"]) is type(b["h"]) for a, b in zip(D.SECTIONS_STD, PB.SECTIONS_STD)))
chk("γ_M, k_mod (3 classes × 5 durées) et k_def sont ceux du module", D.GAMMA_M == PB.GAMMA_M and D.KMOD == PB.KMOD and D.KDEF == PB.KDEF)
chk("les fonctions du module et du moteur rendent les mêmes bits (sect_rect, defl_simply, shear_tau_rect, kh_depth)",
    all(M.section_rect(b, h) == PB.sect_rect(b, h) and M.kh_hauteur(h) == PB.kh_depth(h) and M.tau_rect(7.3, b, h) == PB.shear_tau_rect(7.3, b, h)
        and M.fleche_isostatique(3.7, 4.1, 11000, M.section_rect(b, h)[1]) == PB.defl_simply(3.7, 4.1, 11000, PB.sect_rect(b, h)[1])
        for b in (30.0, 38, 45, 63.5, 300.0) for h in (1.0, 80.0, 120, 150, 151.3, 499.9, 500.0, 2000.0)))
f0, _ = ecran({})
W0 = {w[1]: w for w in f0.widgets}
attendus = {
    L_CLASSE: ("selectbox", list(D.CLASSES), list(D.CLASSES).index(D.DEFAUTS["classe"])),
    L_SC: ("radio", list(D.CLASSES_SERVICE), D.CLASSES_SERVICE.index(D.DEFAUTS["service"])),
    L_DUREE: ("selectbox", list(D.DUREES), D.DUREES.index(D.DEFAUTS["duree"])),
    L_SEC: ("selectbox", [s["tag"] for s in D.SECTIONS_STD], [s["tag"] for s in D.SECTIONS_STD].index(D.DEFAUTS["section"])),
    L_LI: ("selectbox", list(D.LIMITES_INST), D.LIMITES_INST.index(D.DEFAUTS["lim_inst"])),
    L_LF: ("selectbox", list(D.LIMITES_FIN), D.LIMITES_FIN.index(D.DEFAUTS["lim_fin"])),
}
chk("les six menus de l'écran : mêmes options, même ordre, même défaut que le moteur",
    all(W0.get(lab, ())[:4] == (k, lab, opts, idx) for lab, (k, opts, idx) in attendus.items()),
    [(lab, W0.get(lab)) for lab in attendus if W0.get(lab, ())[:4] != (attendus[lab][0], lab, attendus[lab][1], attendus[lab][2])])
chk("les deux bascules : « Section standard » allumée, « Compression ⟂ fil à l'appui » éteinte (les défauts du moteur)",
    W0.get(L_STD) == ("toggle", L_STD, D.DEFAUTS["standard"]) and W0.get(L_APP) == ("toggle", L_APP, D.DEFAUTS["appui"]))
f_mes, _ = ecran({L_STD: False})
W1 = {w[1]: w for w in f_mes.widgets}
champs_ecran = {k: W1.get(lab) for k, lab in NUMERIQUES.items()}
chk("chaque champ numérique de l'écran (9) : défaut, minimum et maximum = donnees.CHAMPS",
    all(w and w[0] == "number_input" and (w[4], w[2], w[3]) == D.CHAMPS[k] for k, w in champs_ecran.items()),
    {k: (w[2:5] if w else None) for k, w in champs_ecran.items() if not w or (w[4], w[2], w[3]) != D.CHAMPS[k]})
chk("le champ « a » est grisé tant que l'appui n'est pas coché, et seulement alors",
    W1[L_A][6] is True and {w[1]: w for w in ecran({L_APP: True})[0].widgets}[L_A][6] is False)
chk("section standard : l'écran propose le menu des sections et pas b, h ; sur mesure : b et h, pas le menu",
    L_SEC in W0 and L_B not in W0 and L_H not in W0 and L_B in W1 and L_H in W1 and L_SEC not in W1)
chk("nouvelle_poutre() est l'écran au premier affichage (chaque menu, bascule et champ à son défaut)",
    M.nouvelle_poutre() == E.normaliser({}) and all(M.nouvelle_poutre()[k] == v for k, v in D.DEFAUTS.items()))

# ------------------------------------------------------------------ 2. à la main
print("\n=== 2. À la main : deux poutres recalculées en N et en mm, sans le moteur ===")


def proche(a, b, tol=1e-12):
    return a == b or abs(a - b) <= tol * max(abs(a), abs(b))


# 63 × 200, C24, classe de service 2, moyen terme, L = 3,6 m, G = 1,5 et Q = 2,5 kN/m, appui a = 80 mm
p1 = M.normaliser({"classe": "C24", "service": 2, "duree": "moyen", "section": "63x200", "L": 3.6, "G": 1.5, "Q": 2.5,
                   "appui": True, "a": 80.0, "lim_inst": "L/400", "lim_fin": "L/250"})
r1 = M.verifier(p1)
b, h, L = 63.0, 200.0, 3600.0                                # mm
q_elu = 1.35 * 1.5 + 1.5 * 2.5                               # N/mm (= kN/m)
I = b * h ** 3 / 12                                          # mm⁴
main = {
    "M_Ed": q_elu * L ** 2 / 8 / 1e6,                        # kN·m
    "sigma_m_Ed": (q_elu * L ** 2 / 8) / (b * h ** 2 / 6),   # N/mm²
    "fm_d": 0.80 * 24 * (150 / 200) ** 0.2 / 1.3,            # k_h de l'écran : < 1 au-dessus de 150 mm
    "tau_Ed": 1.5 * (q_elu * L / 2) / (b * h),
    "fv_d": 0.80 * 4.0 / 1.3,
    "R_appui": (1.5 + 2.5) * L / 2 / 1000,                   # kN, sous G + Q NON pondérées (comme l'écran)
    "sigma_c90_Ed": (1.5 + 2.5) * L / 2 / (b * 80),
    "fc90_d": 0.80 * 2.7 / 1.3,
    # la flèche de l'ÉCRAN : 5·(q·1000)·L⁴/(384·E·I), soit 1 000 fois la flèche de q en N/mm (écart d'unité)
    "w_inst": 1000 * 5 * (1.5 + 2.5) * L ** 4 / (384 * 11000 * I),
    "w_fin": (1 + 0.8) * 1000 * 5 * (1.5 + 0.3 * 2.5) * L ** 4 / (384 * 11000 * I),
    "lim_inst_mm": L / 400, "lim_fin_mm": L / 250,
}
ecarts_main = {k: (r1["grandeurs"][k], v) for k, v in main.items() if not proche(r1["grandeurs"][k], v)}
chk("63 × 200 C24, SC 2, moyen terme, L = 3,6 m, appui 80 mm : M, σ, f_m,d, τ, f_v,d, R, σ_c,90, f_c,90,d, w_inst, w_fin, limites",
    not ecarts_main, ecarts_main)
chk("… et la flèche de l'écran vaut 1 000 fois 5qL⁴/(384EI) avec q en N/mm (l'écart d'unité est REPRODUIT, pas corrigé)",
    proche(r1["grandeurs"]["w_inst"] / 1000, 5 * 4.0 * L ** 4 / (384 * 11000 * I)) and r1["grandeurs"]["w_inst"] > 10_000)
# sur mesure 80 × 120 (k_h > 1), C30, SC 3, instantanée, L = 2,4 m
p2 = M.normaliser({"classe": "C30", "service": 3, "duree": "instant", "standard": False, "b": 80.0, "h": 120.0, "L": 2.4, "G": 0.8,
                   "Q": 1.2, "gamma_G": 1.0, "gamma_Q": 1.0, "psi2": 0.6})
r2 = M.verifier(p2)
kh2 = (150 / 120) ** 0.2
chk("80 × 120 sur mesure, C30, SC 3, instantanée : k_h = (150/120)^0,2 = 1,0456 > 1, f_m,d = 0,9 × 30 × k_h / 1,3",
    proche(r2["grandeurs"]["kh"], kh2) and kh2 > 1 and proche(r2["grandeurs"]["fm_d"], 0.9 * 30 * kh2 / 1.3)
    and proche(r2["grandeurs"]["sigma_m_Ed"], (2.0 * 2400 ** 2 / 8) / (80 * 120 ** 2 / 6)) and r2["grandeurs"]["kdef"] == 2.0)
# la combinaison γ_G·G SEULE, avec k_mod permanent : l'écran ne la fait pas (écart « combinaison_G », non conservatif).
# 63 × 200 C24, SC 1, moyen terme (k_mod 0,8), L = 3 m, G = 3, Q = 0,5 kN/m : l'écran dit la flexion conforme (92 %),
# la combinaison 1,35·G avec k_mod 0,6 donne 104 % — le moteur le chiffre À TITRE INDICATIF, sans toucher au verdict.
p3 = M.normaliser({"section": "63x200", "duree": "moyen", "L": 3.0, "G": 3.0, "Q": 0.5})
r3 = M.verifier(p3)
cg = r3["indicatif"]["combinaison_G"]
kh3 = (150 / 200) ** 0.2
u_G_main = (1.35 * 3.0 * 3000 ** 2 / 8) / (63 * 200 ** 2 / 6) / (0.6 * 24 * kh3 / 1.3)
v_G_main = 1.5 * (1.35 * 3.0 * 3000 / 2) / (63 * 200) / (0.6 * 4.0 / 1.3)
chk("combinaison 1,35·G seule, k_mod permanent (63 × 200, moyen terme, G = 3, Q = 0,5) : recalculée à la main, flexion 104 % "
    "alors que l'écran dit 92 % — l'écart « combinaison_G » est cité",
    cg is not None and proche(cg["u_flex"], u_G_main, 1e-12) and proche(cg["u_shear"], v_G_main, 1e-12) and cg["kmod"] == 0.6
    and cg["u_flex"] > 1.0 > r3["grandeurs"]["u_flex"] and "combinaison_G" in r3["ecarts"], (cg, r3["grandeurs"]["u_flex"]))
chk("… elle reste INDICATIVE : ni les vérifications, ni le taux de flexion, ni le verdict ne changent (les grandeurs de l'écran)",
    [v["cle"] for v in r3["verifications"]] == ["flexion", "cisaillement", "fleche_inst", "fleche_fin"]
    and r3["verifications"][0]["taux"] == r3["grandeurs"]["u_flex"] and "combinaison_G" not in r3["grandeurs"])
r3p = M.verifier(dict(p3, duree="permanent"))
r3z = M.verifier(dict(p3, G=0.0))
chk("sans objet (None, écart non cité) quand la durée choisie est déjà « permanent » ou quand G = 0",
    r3p["indicatif"]["combinaison_G"] is None and "combinaison_G" not in r3p["ecarts"]
    and r3z["indicatif"]["combinaison_G"] is None and "combinaison_G" not in r3z["ecarts"])
chk("les flèches sans l'écart d'unité (indicatives) valent celles de l'écran divisées par 1 000",
    r1["indicatif"]["w_inst_unite"] == r1["grandeurs"]["w_inst"] / 1000.0 and r1["indicatif"]["w_fin_unite"] == r1["grandeurs"]["w_fin"] / 1000.0
    and proche(r1["indicatif"]["w_inst_unite"], 5 * 4.0 * L ** 4 / (384 * 11000 * I)))

# ------------------------------------------------------------------ 3. parité
print("\n=== 3. Parité : l'écran exécuté avec un faux « st » contre le moteur, au bit près ===")
CORRESPONDANCE = (("A", "A"), ("I", "I"), ("W", "W"), ("kh", "kh"), ("kmod", "kmod"), ("kdef", "kdef"), ("fm_d", "fm_d"),
                  ("fv_d", "fv_d"), ("fc90_d", "fc90_d"), ("q_ELU", "q_ELU"), ("M_Ed", "M_ELU"), ("V_Ed", "V_ELU"),
                  ("q_inst", "q_SLS_inst"), ("q_qp", "q_SLS_quasi"), ("sigma_m_Ed", "sigma_m_Ed"), ("tau_Ed", "tau_Ed"),
                  ("w_inst", "w_inst"), ("w_fin", "w_fin"), ("L_mm", "L_mm"), ("lim_inst_mm", "lim_inst_mm"),
                  ("lim_fin_mm", "lim_fin_mm"), ("u_flex", "u_flex"), ("u_shear", "u_shear"), ("u_wi", "u_wi"), ("u_wf", "u_wf"),
                  ("u_c90", "fc90_util"))
CPT = {"bits": 0, "textes": 0}


def affichage_attendu(r):
    """Ce que l'écran affiche, reconstruit à partir des SEULES valeurs du moteur (les formats de l'écran)."""
    g, geo, mat, co = r["grandeurs"], r["geometrie"], r["materiau"], r["coefficients"]
    p = r["poutre"]
    v = {x["cle"]: x for x in r["verifications"]}
    ok = lambda c: "success" if v[c]["etat"] == M.OK else "error"  # noqa: E731
    ev = [("metric", "Section (mm)", f"{int(geo['b'])}×{int(geo['h'])}"), ("metric", "Portée L (m)", f"{geo['L']:.2f}"),
          ("metric", "E0,mean (MPa)", f"{mat['E0_mean']:.0f}"), ("metric", "ρ (kg/m³)", f"{mat['rho_mean']:.0f}"),
          ("write", f"- **M_Ed** = {g['M_Ed']:.2f} kN·m ; **V_Ed** = {g['V_Ed']:.2f} kN"),
          ("write", f"- **σ_m,Ed** = {g['sigma_m_Ed']:.1f} MPa  |  **f_m,d** = {g['fm_d']:.1f} MPa"),
          (ok("flexion"), f"Flexion : **{g['u_flex']*100:.0f}%**"),
          ("write", f"- **τ_Ed** = {g['tau_Ed']:.2f} MPa  |  **f_v,d** = {g['fv_d']:.2f} MPa"),
          (ok("cisaillement"), f"Cisaillement : **{g['u_shear']*100:.0f}%**")]
    if "appui" in v:
        ev += [("write", f"- **σ_c,90,Ed** = {g['sigma_c90_Ed']:.2f} MPa  |  **f_c,90,d** = {g['fc90_d']:.2f} MPa"),
               (ok("appui"), f"Compression ⟂ fil (appui) : **{g['u_c90']*100:.0f}%**")]
    ev += [("write", f"Instantanée : w = {g['w_inst']:.1f} mm | lim. {p['lim_inst']} = {g['lim_inst_mm']:.1f} mm"),
           (ok("fleche_inst"), f"Taux : **{g['u_wi']*100:.0f}%**"),
           ("write", f"Finale : w = {g['w_fin']:.1f} mm | lim. {p['lim_fin']} = {g['lim_fin_mm']:.1f} mm"),
           (ok("fleche_fin"), f"Taux : **{g['u_wf']*100:.0f}%**"),
           ("caption", f"{p['classe']}: fm,k={mat['fm_k']} MPa, fv,k={mat['fv_k']} MPa, fc,90,k={mat['fc90_k']} MPa, "
                       f"E0,mean={mat['E0_mean']} MPa, Gmean={mat['G_mean']} MPa | "
                       f"kmod={co['kmod']:.2f} (SC {co['service']}, {co['duree']}), kdef={co['kdef']:.2f}, kh={co['kh']:.2f}, γM={co['gamma_M']:.2f}")]
    return ev


def comparer(p, moteur=M):
    """Les écarts entre le moteur et l'écran pour la poutre p : grandeurs au bit près, affichage au texte près."""
    ecarts = []
    r = moteur.verifier(p)
    faux, loc = ecran(saisies_de(p))
    g = r["grandeurs"]
    for cm, ce in CORRESPONDANCE:
        a, b_ = g[cm], loc.get(ce)
        if cm == "u_c90" and not p["appui"]:
            b_ = loc.get("fc90_util")
        CPT["bits"] += 1
        if not (a == b_ and type(a) is type(b_)) and not (a is None and b_ is None):
            ecarts.append(f"{cm} : moteur {a!r} ≠ écran {b_!r}")
    for cm, ce in (("b", "b_mm"), ("h", "h_mm")):
        if not (r["geometrie"][cm] == loc[ce] and type(r["geometrie"][cm]) is type(loc[ce])):
            ecarts.append(f"{cm} : moteur {r['geometrie'][cm]!r} ≠ écran {loc[ce]!r}")
    if p["appui"]:
        for cm, ce in (("R_appui", "R_sup"), ("sigma_c90_Ed", "sigma_c90_Ed")):
            if g[cm] != loc[ce]:
                ecarts.append(f"{cm} : moteur {g[cm]!r} ≠ écran {loc[ce]!r}")
    vu = [e for e in faux.ev if e[0] in ("metric", "write", "success", "error", "caption")][:-1]   # la dernière : les hypothèses
    attendu = affichage_attendu(r)
    CPT["textes"] += len(attendu)
    if vu != attendu:
        diff = [(a, b_) for a, b_ in zip(vu, attendu) if a != b_][:2] or [(len(vu), len(attendu))]
        ecarts.append(f"affichage : {diff}")
    return ecarts


def poutre(**k):
    return M.normaliser(k)


CAS = [("défauts de l'écran", M.nouvelle_poutre())]
for c in D.CLASSES:
    for sc in D.CLASSES_SERVICE:
        for du in D.DUREES:
            CAS.append((f"{c} · SC {sc} · {du}", poutre(classe=c, service=sc, duree=du, section="63x200", L=3.2, G=1.2, Q=1.5)))
for s in D.SECTIONS_STD:
    for li in D.LIMITES_INST:
        for lf in D.LIMITES_FIN:
            CAS.append((f"{s['tag']} · {li} · {lf}", poutre(section=s["tag"], lim_inst=li, lim_fin=lf, L=2.5, G=0.6, Q=0.9)))
for b_ in (30.0, 45.5, 80.0, 175.25, 300.0):
    for h_ in (80.0, 120.0, 149.99, 150.0, 150.01, 240.0, 333.3, 500.0):
        CAS.append((f"sur mesure {b_} × {h_}", poutre(standard=False, b=b_, h=h_, L=4.4, G=2.2, Q=1.7, appui=True, a=95.0)))
# chaque champ à ses bornes (les autres à leur défaut)
for k, (dflt, mini, maxi) in D.CHAMPS.items():
    for x in (mini, maxi):
        CAS.append((f"borne {k} = {x}", poutre(**{k: x, "standard": k not in ("b", "h"), "appui": k == "a"})))
CAS += [
    ("sans charge : tout est conforme (flèches nulles)", poutre(G=0.0, Q=0.0, appui=True)),
    ("charge minuscule : les flèches de l'écran passent", poutre(section="100x250", L=0.5, G=0.01, Q=0.0, classe="C40")),
    ("taux exactement 100 % : compression à l'appui (≤ et non <)",
     poutre(classe="C14", service=3, duree="moyen", section="100x250", G=1.0, Q=2.0, L=2.0, appui=True, a=30.0)),
    ("appui coché, petite largeur d'appui", poutre(appui=True, a=30.0, section="45x220", L=5.0, G=3.0, Q=4.0)),
    ("appui coché, grande largeur", poutre(appui=True, a=200.0, section="75x225", L=6.0, G=1.0, Q=1.0)),
    ("appui coché mais section sur mesure", poutre(appui=True, standard=False, b=120.0, h=360.0, L=7.5, G=4.0, Q=3.5)),
    ("ψ₂ = 1 : flèche finale = (1 + k_def)·w_inst", poutre(psi2=1.0, service=3)),
    ("γ_G = γ_Q = 1", poutre(gamma_G=1.0, gamma_Q=1.0, section="63x175")),
    ("valeurs décimales non rondes", poutre(standard=False, b=63.7, h=211.3, L=3.37, G=1.234, Q=2.718, gamma_G=1.35, gamma_Q=1.5,
                                            psi2=0.37, appui=True, a=73.5, classe="C27", service=2, duree="court")),
]
RNG = random.Random(20261107)
for k in range(420):
    std = RNG.random() < 0.6
    CAS.append((f"tirage {k:03d} (graine 20261107)", poutre(
        classe=RNG.choice(list(D.CLASSES)), service=RNG.choice(D.CLASSES_SERVICE), duree=RNG.choice(D.DUREES), standard=std,
        section=RNG.choice([s["tag"] for s in D.SECTIONS_STD]), b=round(RNG.uniform(30, 300), RNG.choice((0, 1, 2))),
        h=round(RNG.uniform(80, 500), RNG.choice((0, 1, 2))), L=round(RNG.uniform(0.5, 12.0), RNG.choice((1, 2, 3))),
        G=round(RNG.uniform(0, 50) if RNG.random() < 0.3 else RNG.uniform(0, 5), 2),
        Q=round(RNG.uniform(0, 50) if RNG.random() < 0.3 else RNG.uniform(0, 5), 2),
        gamma_G=RNG.choice((1.0, 1.35, 1.5, 2.0, round(RNG.uniform(1, 2), 3))), gamma_Q=RNG.choice((1.0, 1.5, 2.0, round(RNG.uniform(1, 2), 3))),
        psi2=RNG.choice((0.1, 0.3, 0.6, 0.8, 1.0, round(RNG.uniform(0.1, 1.0), 3))), lim_inst=RNG.choice(D.LIMITES_INST),
        lim_fin=RNG.choice(D.LIMITES_FIN), appui=RNG.random() < 0.4, a=round(RNG.uniform(30, 200), RNG.choice((0, 1))))))
chk(f"{len(CAS)} poutres au banc de parité", len(CAS) > 700, len(CAS))
fautes = []
for desc, p in CAS:
    e = comparer(p)
    if e:
        fautes.append((desc, e[:3]))
chk(f"les {len(CAS)} poutres : {CPT['bits']} grandeurs identiques au BIT près (valeur et type), "
    f"{CPT['textes']} lignes affichées identiques au texte près", not fautes, fautes[:3])
RES = [M.verifier(p) for _, p in CAS]

# ------------------------------------------------------------------ 4. couverture
print("\n=== 4. Couverture : ce que les cas atteignent ===")
etats = {(v["cle"], v["etat"]) for r in RES for v in r["verifications"]}
chk("chaque vérification est atteinte conforme ET non conforme (flexion, cisaillement, appui, deux flèches)",
    all((c, e) in etats for c in ("flexion", "cisaillement", "appui", "fleche_inst", "fleche_fin") for e in (M.OK, M.NOK)), sorted(etats))
chk("un taux EXACTEMENT égal à 100 % est atteint, et il est conforme (≤, pas <)",
    any(v["taux"] == 1.0 and v["etat"] == M.OK for r in RES for v in r["verifications"]))
khs = [r["grandeurs"]["kh"] for r in RES]
chk("k_h atteint > 1 (h < 150), = 1 (h = 150) et < 1 (h > 150 : l'écart « kh_reduit »)",
    any(x > 1 for x in khs) and 1.0 in khs and any(x < 1 for x in khs)
    and all(("kh_reduit" in r["ecarts"]) == (r["grandeurs"]["kh"] < 1) for r in RES))
chk("chaque classe, classe de service, durée, limite et section standard est jouée ; sur mesure et appui aussi",
    {r["poutre"]["classe"] for r in RES} == set(D.CLASSES) and {r["poutre"]["service"] for r in RES} == set(D.CLASSES_SERVICE)
    and {r["poutre"]["duree"] for r in RES} == set(D.DUREES) and {r["poutre"]["lim_inst"] for r in RES} == set(D.LIMITES_INST)
    and {r["poutre"]["lim_fin"] for r in RES} == set(D.LIMITES_FIN)
    and {r["geometrie"]["tag"] for r in RES} >= {s["tag"] for s in D.SECTIONS_STD}
    and any(not r["poutre"]["standard"] for r in RES) and sum(r["poutre"]["appui"] for r in RES) > 50)

# ------------------------------------------------------------------ 5. mutations
print("\n=== 5. Mutations : chaque altération du moteur fait rougir la parité ===")
SOURCE = lire("bois", "poutre", "moteur.py")
MUTATIONS = (
    ("γ_M 1,30 → 1,25 pour la flexion", 'kmod * mat["fm_k"] * kh / GAMMA_M', 'kmod * mat["fm_k"] * kh / 1.25'),
    ("k_h « corrigé » selon l'EC5 (1 au-delà de 150 mm)", "return min(max(kh, D.KH_BORNES[0]), D.KH_BORNES[1])",
     "return 1.0 if h_mm >= 150 else min(kh, 1.3)"),
    ("τ avec k_cr = 0,67", "return 1.5 * (V_kN*1000.0) / A", "return 1.5 * (V_kN*1000.0) / (0.67*A)"),
    ("appui sous la réaction ELU", "R_sup = (q_SLS_inst * L_m) / 2.0", "R_sup = (q_ELU * L_m) / 2.0"),
    ("flèche finale de l'EC5 (w_inst,G·(1+k_def) + w_inst,Q·(1+ψ₂·k_def))", "w_fin = (1.0 + kdef) * w_qp", "w_fin = w_inst + kdef * w_qp"),
    ("flèche avec q en N/mm (l'unité « corrigée »)", "(5.0 * (q_kN_m*1000.0) * (L_m*1000.0)**4)", "(5.0 * (q_kN_m) * (L_m*1000.0)**4)"),
    ("f_m,d dans un autre ordre d'opérations", 'kmod * mat["fm_k"] * kh / GAMMA_M', 'kmod * (mat["fm_k"] * kh / GAMMA_M)'),
    ("W = b·h²/6 au lieu de I/(h/2)", "W = I/(h_mm/2.0)", "W = b_mm*h_mm**2/6.0"),
    ("k_def de la classe de service 1 partout", "kdef = KDEF[sc]", "kdef = KDEF[1]"),
    ("limite finale lue sur la limite instantanée", "lim_fin_mm = L_mm / n_fin", "lim_fin_mm = L_mm / n_inst"),
    ("ψ₂ oublié dans la combinaison quasi permanente", "q_SLS_quasi = qG + psi2*qQ", "q_SLS_quasi = qG + qQ"),
    ("« ≤ 100 % » devenu « < 100 % »", 'v["taux"] <= 1.0 else NOK', 'v["taux"] < 1.0 else NOK'),
)


def moteur_mute(source):
    mod = types.ModuleType("bois.poutre._mute")
    mod.__package__ = "bois.poutre"
    mod.__file__ = M.__file__
    exec(compile(source, M.__file__, "exec"), mod.__dict__)
    return mod


EXTRAIT = [p for i, (_, p) in enumerate(CAS) if i % 3 == 0 or "100 %" in CAS[i][0] or "sur mesure" in CAS[i][0]]
for nom, avant, apres in MUTATIONS:
    if SOURCE.count(avant) != 1:
        chk(f"mutation « {nom} » : le texte visé est dans le moteur, une fois", False, SOURCE.count(avant))
        continue
    mm = moteur_mute(SOURCE.replace(avant, apres))
    n_rouges = sum(1 for p in EXTRAIT if comparer(p, mm))
    chk(f"mutation « {nom} » : la parité rougit ({n_rouges}/{len(EXTRAIT)} poutres)", n_rouges > 0)
chk("le moteur non muté, chargé de la même façon, reste au vert", not any(comparer(p, moteur_mute(SOURCE)) for p in EXTRAIT[:40]))
# la combinaison G seule est INDICATIVE (hors parité) : c'est le contrôle à la main du § 2 qui la garde
MUT_G = ('km_p = KMOD[sc]["permanent"]', 'km_p = KMOD[sc]["long"]')
if SOURCE.count(MUT_G[0]) == 1:
    cg_m = moteur_mute(SOURCE.replace(*MUT_G)).verifier(p3)["indicatif"]["combinaison_G"]
    chk("mutation « combinaison G seule avec k_mod long terme » : le contrôle à la main la voit (et la parité, hors sujet, non)",
        cg_m is None or not proche(cg_m["u_flex"], u_G_main, 1e-9))
else:
    chk("mutation « combinaison G seule » : le texte visé est dans le moteur, une fois", False, SOURCE.count(MUT_G[0]))

# ------------------------------------------------------------------ 6. sélecteur
print("\n=== 6. Sélecteur de sections : chaque candidate repassée par l'écran ===")
for desc, p in (("défauts", M.nouvelle_poutre()), ("charge légère, L = 1,2 m", poutre(L=1.2, G=0.0, Q=0.0005, section="63x175")),
                ("sans charge", poutre(G=0.0, Q=0.0, section="45x90")), ("sur mesure", poutre(standard=False, G=0.0, Q=0.0))):
    cands = M.candidats_sections(p)
    ecarts = []
    for c in cands:
        q = dict(p, standard=True, section=c["nom"])
        _, loc = ecran(saisies_de(q))
        t = max(x for x in (loc["u_flex"], loc["u_shear"], loc["u_wi"], loc["u_wf"]) + ((loc["fc90_util"],) if loc["fc90_util"] is not None else ()))
        if c["taux"] != t or c["A"] != loc["A"]:
            ecarts.append((c["nom"], c["taux"], t))
    aires = [c["A"] for c in cands]
    best = next((c for c in cands if c["ok"]), None)
    chk(f"sélecteur ({desc}) : les {len(cands)} sections de la liste, triées par aire, taux maximal = celui de l'écran ; "
        f"retenue = {[c['nom'] for c in cands if c['retenue']]} ; plus légère qui tient = {best['nom'] if best else 'aucune'}",
        not ecarts and len(cands) == len(D.SECTIONS_STD) and aires == sorted(aires)
        and [c["nom"] for c in cands if c["retenue"]] == ([p["section"]] if p["standard"] else [])
        and [c["nom"] for c in cands if c["optimal"]] == ([best["nom"]] if best else [])
        and all(c["ok"] == (c["taux"] <= 1.0) for c in cands), ecarts[:3])

# ------------------------------------------------------------------ 7. format
print("\n=== 7. Format : JSON, normalisation, domaine de l'écran ===")
chk("la sortie de verifier() est du JSON strict (sans NaN) et se relit à l'identique",
    all(json.loads(json.dumps(r, allow_nan=False)) == r for r in RES))
chk("normaliser() est idempotente et {} donne nouvelle_poutre()",
    all(M.normaliser(M.normaliser(p)) == M.normaliser(p) for _, p in CAS) and M.normaliser({}) == M.nouvelle_poutre())
entree = dict(CAS[5][1])
M.verifier(entree)
M.candidats_sections(entree)
chk("verifier() et candidats_sections() ne modifient pas la poutre reçue", entree == CAS[5][1])
REFUS = [("classe inconnue", {"classe": "C45"}), ("classe GL24h (absente de la base)", {"classe": "GL24h"}), ("service 4", {"service": 4}),
         ("service 1.0 (réel)", {"service": 1.0}), ("service vrai", {"service": True}), ("durée inconnue", {"duree": "longue"}),
         ("section absente", {"section": "50x200"}), ("standard en texte", {"standard": "oui"}), ("limite L/500", {"lim_inst": "L/500"}),
         ("limite finale L/400", {"lim_fin": "L/400"}), ("appui en nombre", {"appui": 1}), ("nom numérique", {"nom": 12}),
         ("clé inconnue", {"portee": 4.0}), ("b en texte", {"b": "80"}), ("L infinie", {"L": float("inf")}), ("G NaN", {"G": float("nan")})]
for k, (dflt, mini, maxi) in D.CHAMPS.items():
    REFUS += [(f"{k} sous son minimum", {k: mini - 0.01}), (f"{k} au-dessus de son maximum", {k: maxi + 0.01})]
REFUS += [("G négative (l'écran ne la ramène pas à 0 : refus)", {"G": -1.0}), ("ψ₂ nul (sous le minimum de l'écran, 0,1)", {"psi2": 0.0})]
mal = []
for nom, champ in REFUS:
    try:
        M.verifier(dict(M.nouvelle_poutre(), **champ))
        mal.append(f"{nom} : accepté")
    except ValueError as e:
        if not str(e) or "Traceback" in str(e):
            mal.append(f"{nom} : message vide")
    except Exception as e:  # noqa: BLE001 — une autre exception n'est pas un refus : c'est une panne
        mal.append(f"{nom} : {type(e).__name__} {e}")
chk(f"{len(REFUS)} saisies hors du domaine de l'écran lèvent ValueError avec un message (jamais de substitution)", not mal, mal)
try:
    M.normaliser(dict(M.nouvelle_poutre(), L=0.2))
except ValueError as e:
    chk("le message de refus est en français et donne le domaine : « … entre 0,5 et 12 m … »", "entre 0,5 et 12 m" in str(e), str(e))
chk("les bornes elles-mêmes sont admises (le domaine de l'écran est fermé)",
    all(M.verifier(dict(M.nouvelle_poutre(), **{k: x})) for k, (_, mi, ma) in D.CHAMPS.items() for x in (mi, ma)))
chk("les écarts EC5 nommés : huit codes, chacun avec son texte et son sens, et chaque poutre ne cite que des codes connus",
    len(M.ECARTS) == 8 and all(len(t) == 2 and t[0] and t[1] for t in M.ECARTS.values()) and all(set(r["ecarts"]) <= set(M.ECARTS) for r in RES))
# « formule = valeurs = résultat » : chaque résultat se retrouve à partir des seuls termes que le moteur donne
FORMULES = {
    "q_ELU": lambda t: t["gamma_G"] * t["G"] + t["gamma_Q"] * t["Q"],
    "M_Ed": lambda t: t["q_ELU"] * t["L"] ** 2 / 8.0,
    "V_Ed": lambda t: t["q_ELU"] * t["L"] / 2.0,
    "I": lambda t: t["b"] * t["h"] ** 3 / 12.0,
    "W": lambda t: t["I"] / (t["h"] / 2.0),
    "sigma_m_Ed": lambda t: t["M_Ed"] * 1e6 / t["W"],
    "kmod": lambda t: D.KMOD[t["service"]][t["duree"]],
    "kh": lambda t: min(max((150.0 / t["h"]) ** 0.2, 0.6), 1.3),
    "fm_d": lambda t: t["kmod"] * t["fm_k"] * t["kh"] / t["gamma_M"],
    "tau_Ed": lambda t: 1.5 * t["V_Ed"] * 1000.0 / (t["b"] * t["h"]),
    "fv_d": lambda t: t["kmod"] * t["fv_k"] / t["gamma_M"],
    "R_appui": lambda t: t["q_inst"] * t["L"] / 2.0,
    "sigma_c90_Ed": lambda t: t["R_appui"] * 1000.0 / (t["b"] * t["a"]),
    "fc90_d": lambda t: t["kmod"] * t["fc90_k"] / t["gamma_M"],
    "q_inst": lambda t: t["G"] + t["Q"],
    "q_qp": lambda t: t["G"] + t["psi2"] * t["Q"],
    "w_inst": lambda t: 5.0 * t["q"] * 1000.0 * (t["L"] * 1000.0) ** 4 / (384.0 * t["E"] * t["I"]),
    "w_qp": lambda t: 5.0 * t["q"] * 1000.0 * (t["L"] * 1000.0) ** 4 / (384.0 * t["E"] * t["I"]),
    "kdef": lambda t: D.KDEF[t["service"]],
    "w_fin": lambda t: (1.0 + t["kdef"]) * t["w_qp"],
    "lim_inst": lambda t: t["L"] / t["n"],
    "lim_fin": lambda t: t["L"] / t["n"],
    "u": lambda t: t["valeur"] / t["limite"],
}
fautes, n_f = [], 0
for r in RES:
    for v in r["verifications"]:
        for f in v["formules"]:
            n_f += 1
            t = {x["symbole"]: x["valeur"] for x in f["termes"]}
            if not proche(FORMULES[f["cle"]](t), f["resultat"], 1e-12):
                fautes.append((r["poutre"]["nom"], v["cle"], f["cle"]))
        if v["formules"][-1]["resultat"] != v["taux"] or v["taux"] != v["valeur"] / v["limite"]:
            fautes.append((v["cle"], "taux"))
chk(f"« formule = valeurs = résultat » : les {n_f} formules se recalculent à partir de leurs seuls termes ; la dernière est le taux",
    not fautes and n_f > 10_000, fautes[:4])
doc = lire("docs", "bois", "POUTRE_BOIS_MOTEUR.md") if os.path.exists(os.path.join(RACINE, "docs", "bois", "POUTRE_BOIS_MOTEUR.md")) else ""
cles = sorted(RES[0]["grandeurs"])
sans_doc = [k for k in cles if f"`{k}`" not in doc or f"``{k}``" not in M.__doc__]
chk(f"les {len(cles)} grandeurs sont nommées dans la docstring du moteur ET dans docs/bois/POUTRE_BOIS_MOTEUR.md", not sans_doc, sans_doc)
chk("chaque écart EC5 du moteur (code) est discuté dans la doc", all(f"`{k}`" in doc for k in M.ECARTS), [k for k in M.ECARTS if f"`{k}`" not in doc])

# ------------------------------------------------------------------ 8. courbes des dessins
print("\n=== 8. Courbes des dessins : les maxima sont ceux du calcul ===")
mauvaises = []
for r in RES[::7]:
    c = DG.courbes(r, 49)
    g = r["grandeurs"]
    if not (c["V"][0] == g["V_Ed"] and c["V"][-1] == -g["V_Ed"] and proche(c["M"][24], g["M_Ed"], 1e-15) and c["M"][0] == 0
            and proche(max(c["w_inst"]), g["w_inst"], 1e-15) and proche(c["w_fin"][24], g["w_fin"], 1e-15) and c["x"][-1] == r["geometrie"]["L"]
            and max(c["M"]) == c["M"][24]):
        mauvaises.append(r["poutre"]["nom"])
chk("V(0) = V_Ed, V(L) = −V_Ed, M(L/2) = M_Ed (le maximum), w(L/2) = w_inst et w_fin du calcul", not mauvaises, mauvaises[:3])
try:
    DG.courbes(RES[0], 10)
    chk("courbes : un nombre de points pair est refusé (la mi-portée doit être un point)", False)
except ValueError:
    chk("courbes : un nombre de points pair est refusé (la mi-portée doit être un point)", True)

# ------------------------------------------------------------------ 9. pureté
print("\n=== 9. Pureté : le moteur se charge et calcule SANS streamlit ni numpy, avec la seule bibliothèque standard ===")
non_std = []
for dossier, _, fichiers in os.walk(os.path.join(RACINE, "bois")):
    for f in fichiers:
        if f.endswith(".py"):
            arbre = ast.parse(lire(os.path.relpath(os.path.join(dossier, f), RACINE)))
            for n in ast.walk(arbre):
                noms = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                    [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else []
                non_std += [f"{f}: {x}" for x in noms if x.split(".")[0] not in sys.stdlib_module_names | {"bois"}]
chk("aucun fichier de bois/ n'importe autre chose que la bibliothèque standard (lecture des sources)", not non_std, non_std)
SOUS_PROCESSUS = r'''
import json, sys
avant = set(sys.modules)
INTERDITS = ("streamlit", "pandas", "reportlab", "numpy", "matplotlib", "scipy", "pymupdf", "fitz")
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
from bois.poutre import moteur as M, diagrammes as DG
poutres = json.loads(sys.stdin.read())
res = [M.verifier(p) for p in poutres]
sortie = {"refus": refus, "version": list(sys.version_info[:2]), "res": res, "cands": M.candidats_sections(poutres[1]),
          "courbes": DG.courbes(res[1])}
charges = {m.split(".")[0] for m in set(sys.modules) - avant}
sortie["tiers"] = sorted(charges - set(sys.stdlib_module_names) - {"bois"})
sortie["interdits"] = sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)
print(json.dumps(sortie))
'''
lot = [p for _, p in CAS[::5]]
attendu = json.loads(json.dumps({"res": [M.verifier(p) for p in lot], "cands": M.candidats_sections(lot[1]),
                                 "courbes": DG.courbes(M.verifier(lot[1]))}))
interpretes = [sys.executable] + [x for x in (shutil.which("python3.13"), "/usr/bin/python3.13")
                                  if x and os.path.exists(x) and os.path.realpath(x) != os.path.realpath(sys.executable)][:1]
for exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(lot), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"{exe} : le moteur se charge et calcule sans streamlit ni numpy", False, pr.stderr.strip().splitlines()[-3:])
        continue
    out = json.loads(pr.stdout)
    v = ".".join(map(str, out["version"]))
    chk(f"Python {v} : streamlit, numpy, pandas… REFUSÉS par sys.meta_path, et le moteur calcule quand même ({len(out['res'])} poutres, "
        "le sélecteur, les courbes)", out["refus"] and not out["interdits"], (out["refus"], out["interdits"]))
    chk(f"Python {v} : le moteur n'a chargé que la bibliothèque standard", not out["tiers"], out["tiers"])
    chk(f"Python {v} : résultats identiques à ceux de ce processus, au bit près",
        {k: out[k] for k in ("res", "cands", "courbes")} == attendu)
if len(interpretes) == 1:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)   ({time.time() - T0:.1f} s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

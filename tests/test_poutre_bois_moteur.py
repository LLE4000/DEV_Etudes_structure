# -*- coding: utf-8 -*-
"""Moteur « Poutre bois » (bois/poutre), refait le 10/11/2026 sur l'Eurocode 5 (docs/site/LOT_Q_QUATRE_MODULES.md § 1) :
plus aucune parité avec l'écran Streamlit — chaque formule de la conception est recalculée ICI, avec ``math`` seul,
et chaque contrôle rougit si l'on retire ce qu'il protège (§ 14, mutations).

  1. TABLES : chaque table du moteur (EN 338:2016 tableau 1, EN 14080:2013, γ_M, k_mod, k_def, k_cr, k_sys, k_h, les
     natures de la charge variable, les limites, les sections) relue contre une transcription INDÉPENDANTE écrite ici
     (par classe, d'après le rapport de sources du lot) ; les relations de cohérence de l'EN 384 entre les propriétés
     d'une classe (contrôle de transcription, pas une source).
  2. EXEMPLE 1 DÉROULÉ À LA MAIN : la poutre de l'ouverture (C24, 75 × 225, L = 4 m, G = 0,6 et Q = 0,8 kN/m,
     habitation, service 1, a = 100 mm, maintien continu) — tout, de la section aux flèches, à 1e-9 près.
  3. LA RÉFÉRENCE : le calcul de la conception § 1.4 refait ici en N et en mm (une autre écriture que le moteur),
     comparé au moteur sur les exemples de la consigne — G seule qui gouverne (G = 3, Q = 0,5), le déversement
     (45 × 245 sur 5 m, maintien aux appuis), un lamellé (GL24h 120 × 400 sur 8 m), l'appui qui gouverne (a = 40 mm),
     la classe de service 3 — puis sur 400 poutres tirées au sort.
  4. LES DEUX COMBINAISONS : G seule gouverne quand Q est petit, G + Q sinon ; chaque vérification nomme la sienne ;
     sans Q, une seule combinaison ; le défaut de l'ancien écran (92 % affichés pour 104 %) est vu.
  5. k_h (massif 150 mm, lamellé 600 mm, plafonds), k_crit aux trois branches et à leurs raccords, l_ef (appuis,
     entretoises), k_c,90 (1,5 ; 1,75 ; 1,0 avec avertissement), la longueur efficace d'appui.
  6. UNITÉS : la flèche recalculée en unités SI (N, m, Pa) par le théorème de la charge unité, intégrée
     numériquement — l'erreur × 1 000 de l'ancien écran serait prise.
  7. ELS : w_inst, w_fin et w_net,fin, k_def, ψ₂ de chaque nature, la contre-flèche, les limites.
  8. CHARGES : la saisie surfacique, le poids propre, la nature « autre ».
  9. INVARIANTS : b doublé (contraintes divisées par deux), charges doublées, portée, k_sys, monotonie.
 10. FORMULES : chaque formule de chaque bloc RÉÉVALUÉE à partir de son gabarit et de ses seuls termes, par un petit
     interpréteur écrit ici ; chaque vérification finit par son taux.
 11. SÉLECTEUR : les sections du type du bois, triées par aire, la retenue, la plus légère qui tient.
 12. REFUS : chaque champ hors bornes, type faux, clé inconnue, contradiction — ValueError en français qui nomme le
     champ, l'entrée intacte ; les bornes admises ; idempotence ; JSON strict.
 13. DIAGRAMMES et PURETÉ : les maxima des courbes sont ceux du calcul ; bibliothèque standard seule (sources lues,
     puis sous-processus où sys.meta_path refuse numpy, streamlit, pandas… sous Python 3.12 et 3.13).
 14. MUTATIONS : le moteur rechargé depuis ses sources, modifié d'UNE formule ou d'UNE valeur, est rejeté ; un
     témoin ne l'est pas.

    /usr/bin/python3.12 tests/test_poutre_bois_moteur.py
"""
import ast
import copy
import json
import math
import os
import random
import re
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
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:700]}]" if info and not cond else ""))


def pres(a, b, tol=1e-9):
    """Égalité relative (tol), absolue sous 1."""
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def dec(a, b, nd):
    """a écrit avec nd décimales vaut b."""
    return abs(a - b) <= 0.5 * 10 ** (-nd) + 1e-12


def lire(*chemin):
    with open(os.path.join(RACINE, *chemin), encoding="utf-8") as fh:
        return fh.read()


def V(res, cle):
    return next(v for v in res["verifications"] if v["cle"] == cle)


def poutre(**k):
    """La poutre neuve avec ``k`` : une clé « charges__G » écrit dans le groupe « charges »."""
    p = M.nouvelle_poutre()
    for cle, val in k.items():
        if "__" in cle:
            g, c = cle.split("__")
            p[g][c] = val
        else:
            p[cle] = val
    if "section" in k or k.get("standard") is False or "classe" in k:
        if p["standard"]:
            p["b"] = p["h"] = None                    # déduits de la section standard
    if "charges__nature" in k and k["charges__nature"] != "autre":
        p["charges"]["duree"] = p["charges"]["psi2"] = None
    return p


# ====================================================================== 1. les tables
print("=== 1. Tables : chaque valeur de norme relue contre une transcription indépendante ===")
# Transcription INDÉPENDANTE (par classe : f_m,k, f_t,0,k, f_c,0,k, f_c,90,k, f_v,k, E_0,mean, E_0,05, G_mean, ρ_k,
# ρ_mean), d'après le rapport de sources du lot Q (10/11/2026) : EN 338:2016 tableau 1, EN 14080:2013 tableaux 4 et 5.
T = {
    "C14": (14, 7.2, 16, 2.0, 3.0, 7000, 4700, 440, 290, 350),
    "C16": (16, 8.5, 17, 2.2, 3.2, 8000, 5400, 500, 310, 370),
    "C18": (18, 10, 18, 2.2, 3.4, 9000, 6000, 560, 320, 380),
    "C20": (20, 11.5, 19, 2.3, 3.6, 9500, 6400, 590, 330, 400),
    "C22": (22, 13, 20, 2.4, 3.8, 10000, 6700, 630, 340, 410),
    "C24": (24, 14.5, 21, 2.5, 4.0, 11000, 7400, 690, 350, 420),
    "C27": (27, 16.5, 22, 2.5, 4.0, 11500, 7700, 720, 360, 430),
    "C30": (30, 19, 24, 2.7, 4.0, 12000, 8000, 750, 380, 460),
    "C35": (35, 22.5, 25, 2.7, 4.0, 13000, 8700, 810, 390, 470),
    "C40": (40, 26, 27, 2.8, 4.0, 14000, 9400, 880, 400, 480),
    "C45": (45, 30, 29, 2.9, 4.0, 15000, 10100, 940, 410, 490),
    "C50": (50, 33.5, 30, 3.0, 4.0, 16000, 10700, 1000, 430, 520),
    "GL24h": (24, 19.2, 24, 2.5, 3.5, 11500, 9600, 650, 385, 420),
    "GL28h": (28, 22.3, 28, 2.5, 3.5, 12600, 10500, 650, 425, 460),
    "GL32h": (32, 25.6, 32, 2.5, 3.5, 14200, 11800, 650, 440, 490),
    "GL24c": (24, 17, 21.5, 2.5, 3.5, 11000, 9100, 650, 365, 400),
    "GL28c": (28, 19.5, 24, 2.5, 3.5, 12500, 10400, 650, 390, 420),
    "GL32c": (32, 19.5, 24.5, 2.5, 3.5, 13500, 11200, 650, 400, 440),
}
NOMS_T = ("fm_k", "ft0_k", "fc0_k", "fc90_k", "fv_k", "E0_mean", "E0_05", "G_mean", "rho_k", "rho_mean")
TABLE = {c: dict(zip(NOMS_T, map(float, v))) for c, v in T.items()}


def fautes_tables(d):
    """Ce que les tables de ``d`` (bois/poutre/donnees) ont de différent de la transcription du test."""
    f = []
    if list(d.CLASSES) != list(T):
        f.append(f"classes : {list(d.CLASSES)}")
    for c, vals in TABLE.items():
        for k, v in vals.items():
            if d.CLASSES.get(c, {}).get(k) != v:
                f.append(f"{c} {k} : {d.CLASSES.get(c, {}).get(k)} ≠ {v}")
        typ = "massif" if c.startswith("C") else "lamelle"
        if d.CLASSES.get(c, {}).get("type") != typ:
            f.append(f"{c} : type {d.CLASSES.get(c, {}).get('type')}")
    if d.GAMMA_M != {"massif": 1.30, "lamelle": 1.25}:
        f.append(f"γ_M : {d.GAMMA_M}")
    kmod = {1: (0.60, 0.70, 0.80, 0.90, 1.10), 2: (0.60, 0.70, 0.80, 0.90, 1.10), 3: (0.50, 0.55, 0.65, 0.70, 0.90)}
    for s, vals in kmod.items():
        if tuple(d.KMOD[s][k] for k in ("permanent", "long", "moyen", "court", "instantane")) != vals:
            f.append(f"k_mod classe {s} : {d.KMOD[s]}")
    if d.KDEF != {1: 0.60, 2: 0.80, 3: 2.00}:
        f.append(f"k_def : {d.KDEF}")
    if (d.K_CR, d.K_SYS) != (0.67, 1.1):
        f.append(f"k_cr, k_sys : {d.K_CR}, {d.K_SYS}")
    if (d.K_H["massif"]["h_ref"], d.K_H["massif"]["exposant"], d.K_H["massif"]["plafond"]) != (150.0, 0.2, 1.3) \
            or (d.K_H["lamelle"]["h_ref"], d.K_H["lamelle"]["exposant"], d.K_H["lamelle"]["plafond"]) != (600.0, 0.1, 1.1):
        f.append(f"k_h : {d.K_H}")
    if (d.COEF_SIGMA_CRIT, d.LAMBDA_PALIER, d.LAMBDA_ELANCE, d.K_CRIT_ORIGINE, d.K_CRIT_PENTE, d.LEF_CHARGE_REPARTIE, d.LEF_FIBRE_COMPRIMEE) \
            != (0.78, 0.75, 1.4, 1.56, 0.75, 0.9, 2.0):
        f.append("déversement : une constante a changé")
    if (d.K_C90, d.K_C90_DEFAUT, d.MAJORATION_APPUI, d.L1_SUR_H_MIN, d.A_MAX_K_C90) != ({"massif": 1.5, "lamelle": 1.75}, 1.0, 30.0, 2.0, {"lamelle": 400.0}):
        f.append("compression d'appui : une constante a changé")
    if (d.COEF_TAU_RECTANGLE, d.FLECHE_NUM, d.FLECHE_DEN, d.FORME_TRANCHANT, d.G_PESANTEUR, d.GAMMA_G, d.GAMMA_Q) != (1.5, 5.0, 384.0, 1.2, 9.81, 1.35, 1.5):
        f.append("cisaillement, flèche, pesanteur ou γ : une constante a changé")
    natures = {"A": ("moyen", 0.3), "B": ("moyen", 0.3), "C": ("moyen", 0.6), "D": ("moyen", 0.6), "E": ("long", 0.8), "H": ("court", 0.0),
               "neige": ("court", 0.0), "chantier": ("court", 0.0), "autre": (None, None)}
    if {k: (d.DUREE_NATURE[k], d.PSI2_NATURE[k]) for k in d.CLES_NATURE} != natures or list(d.CLES_NATURE) != list(natures):
        f.append(f"natures : {[(k, d.DUREE_NATURE[k], d.PSI2_NATURE[k]) for k in d.CLES_NATURE]}")
    if d.LIMITES_INST != (200, 250, 300, 350, 400, 500) or d.LIMITES_FIN != (150, 200, 250, 300, 350) \
            or (d.LIMITE_INST_DEFAUT, d.LIMITE_FIN_DEFAUT) != (300, 250):
        f.append("limites de flèche")
    return f


chk("les 18 classes (C14 à C50, GL24h à GL32c) et leurs dix propriétés ; γ_M, k_mod, k_def, k_cr, k_sys, k_h, déversement, appui, flèche, natures, "
    "limites : identiques à la transcription du test", not fautes_tables(D), fautes_tables(D)[:4])
coh = []
for c in [x for x in T if x.startswith("C")]:
    m = TABLE[c]
    if abs(m["E0_05"] - 0.67 * m["E0_mean"]) > 50.01:
        coh.append(f"{c} : E_0,05 ≠ 0,67·E_0,mean")
    if abs(m["G_mean"] - m["E0_mean"] / 16.0) > 5.01:
        coh.append(f"{c} : G_mean ≠ E_0,mean/16")
    if abs(m["fv_k"] - min(1.6 + 0.1 * m["fm_k"], 4.0)) > 1e-9:
        coh.append(f"{c} : f_v,k")
    if abs(m["fc90_k"] - 0.007 * m["rho_k"]) > 0.0501:
        coh.append(f"{c} : f_c,90,k ≠ 0,007·ρ_k")
    if abs(m["ft0_k"] - (-3.07 + 0.73 * m["fm_k"])) > 0.25:
        coh.append(f"{c} : f_t,0,k")
    if abs(m["fc0_k"] - 4.3 * math.sqrt(m["fm_k"])) > 0.5:
        coh.append(f"{c} : f_c,0,k")
chk("EN 338:2016 : les relations de l'EN 384 tiennent pour les 12 classes C (E_0,05 = 0,67·E, G = E/16, f_v,k = min(1,6 + 0,1·f_m,k ; 4), f_c,90,k = 0,007·ρ_k, "
    "f_t,0,k = −3,07 + 0,73·f_m,k, f_c,0,k = 4,3·√f_m,k) — aucune valeur d'une autre édition", not coh, coh)
chk("EN 14080:2013 : GL28h f_t,0,k = 22,3 (pas 22,4) ; classes h : f_c,0,k = f_m,k ; f_v,k = 3,5 et G_mean = 650 partout",
    D.CLASSES["GL28h"]["ft0_k"] == 22.3 and all(D.CLASSES[c]["fc0_k"] == D.CLASSES[c]["fm_k"] for c in ("GL24h", "GL28h", "GL32h"))
    and all(D.CLASSES[c]["fv_k"] == 3.5 and D.CLASSES[c]["G_mean"] == 650.0 for c in T if c.startswith("GL")))
MASSIF_ATTENDU = ["38x89", "38x140", "38x184", "38x235", "45x95", "45x120", "45x145", "45x170", "45x195", "45x220", "45x245", "63x150", "63x175",
                  "63x200", "63x225", "75x150", "75x175", "75x200", "75x225", "100x200", "100x225", "100x250", "100x300", "150x150", "150x300", "200x200"]
sm, sl = [s["nom"] for s in E.sections("massif")], E.sections("lamelle")
chk("sections du bois massif : les 26 de la consigne, dans l'ordre", sm == MASSIF_ATTENDU, sm)
chk("sections du lamellé-collé : 7 largeurs × 10 hauteurs, limitées à h/b ≤ 6 → 67 ; aucune au-delà",
    len(sl) == 67 and all(s["h"] <= 6 * s["b"] for s in sl) and {s["b"] for s in sl} == {80, 100, 120, 140, 160, 180, 200}
    and {s["h"] for s in sl} == {160, 200, 240, 280, 320, 360, 400, 480, 560, 640} and "80x560" not in [s["nom"] for s in sl], len(sl))
P0 = M.nouvelle_poutre()
chk("la poutre neuve : C24, 75 × 225 standard, L = 4 m, G = 0,6, Q = 0,8 kN/m (habitation, moyen terme, ψ₂ = 0,3), poids propre compté, service 1, "
    "appui de 100 mm vérifié, maintien continu, γ 1,35 et 1,5, L/300 et L/250, sans contre-flèche",
    (P0["classe"], P0["standard"], P0["section"], P0["b"], P0["h"], P0["L"]) == ("C24", True, "75x225", 75.0, 225.0, 4.0)
    and P0["charges"] == {"mode": "lineaire", "G": 0.6, "Q": 0.8, "g": 1.5, "q": 2.0, "entraxe": 0.4, "poids_propre": True, "nature": "A", "duree": "moyen", "psi2": 0.3}
    and P0["appui"] == {"verifier": True, "a": 100.0, "debord": 0.0} and P0["maintien"] == {"type": "continu", "entraxe": 1.0}
    and (P0["service"], P0["gamma_G"], P0["gamma_Q"], P0["systeme"], P0["lim_inst"], P0["lim_fin"], P0["contre_fleche"]) == (1, 1.35, 1.5, False, 300, 250, 0.0), P0)
chk("la poutre neuve est déjà normalisée, et la normalisation est idempotente", M.normaliser(P0) == P0 and M.normaliser(M.normaliser(P0)) == P0)

# ====================================================================== 2. l'exemple 1, déroulé à la main
print("\n=== 2. Exemple 1 déroulé à la main : C24, 75 × 225, L = 4 m, G = 0,6 et Q = 0,8 kN/m, habitation, service 1, a = 100 mm ===")
R0 = M.verifier(P0)
# --- les valeurs, en clair (C24, EN 338:2016 : f_m,k 24 · f_v,k 4,0 · f_c,90,k 2,5 · E_0,mean 11 000 · G_mean 690 N/mm² · ρ_mean 420 kg/m³)
b1, h1, L1 = 75.0, 225.0, 4000.0                                   # mm
A1 = 75.0 * 225.0                                                  # 16 875 mm²
W1 = 75.0 * 225.0 ** 2 / 6.0                                       # 632 812,5 mm³
I1 = 75.0 * 225.0 ** 3 / 12.0                                      # 71 191 406,25 mm⁴
gpp1 = 420.0 * 9.81 * (0.075 * 0.225) / 1000.0                     # 420 kg/m³ × 9,81 N/kg × 0,016875 m² = 69,53 N/m = 0,0695 kN/m
g1, q1 = 0.6 + gpp1, 0.8                                           # kN/m = N/mm
pG1 = 1.35 * g1                                                    # combinaison G seule, k_mod permanent 0,60
pGQ1 = 1.35 * g1 + 1.5 * 0.8                                       # combinaison G + Q, k_mod moyen terme 0,80
MG1, MGQ1 = pG1 * L1 ** 2 / 8.0, pGQ1 * L1 ** 2 / 8.0              # N·mm
VG1, VGQ1 = pG1 * L1 / 2.0, pGQ1 * L1 / 2.0                        # N
# flexion : h = 225 ≥ 150 → k_h = 1 ; maintien continu → k_crit = 1 ; γ_M = 1,30
fmdG1, fmdGQ1 = 0.60 * 1.0 * 24.0 / 1.30, 0.80 * 1.0 * 24.0 / 1.30
sigG1, sigGQ1 = MG1 / W1, MGQ1 / W1
# cisaillement : k_cr = 0,67
fvdG1, fvdGQ1 = 0.60 * 4.0 / 1.30, 0.80 * 4.0 / 1.30
tauG1, tauGQ1 = 1.5 * VG1 / (0.67 * 75.0 * 225.0), 1.5 * VGQ1 / (0.67 * 75.0 * 225.0)
# appui : a = 100, débord 0 → l_ef = 100 + min(30 ; 100 ; 3900/2) + min(30 ; 100 ; 0) = 130 ; l_1 = 3900 ≥ 2·225 → k_c,90 = 1,5
lefa1 = 100.0 + 30.0 + 0.0
fc90G1, fc90GQ1 = 0.60 * 2.5 / 1.30, 0.80 * 2.5 / 1.30
s90G1, s90GQ1 = VG1 / (75.0 * lefa1), VGQ1 / (75.0 * lefa1)
# flèches : k_def = 0,60 ; ψ₂ = 0,3
wgm1 = 5.0 * g1 * L1 ** 4 / (384.0 * 11000.0 * I1)
wgv1 = 1.2 * g1 * L1 ** 2 / (8.0 * 690.0 * A1)
wqm1 = 5.0 * q1 * L1 ** 4 / (384.0 * 11000.0 * I1)
wqv1 = 1.2 * q1 * L1 ** 2 / (8.0 * 690.0 * A1)
wg1, wq1 = wgm1 + wgv1, wqm1 + wqv1
winst1 = wg1 + wq1
wfin1 = wg1 * (1.0 + 0.60) + wq1 * (1.0 + 0.3 * 0.60)
ex1 = [
    ("A", R0["geometrie"]["A"], A1), ("W", R0["geometrie"]["W"], W1), ("I", R0["geometrie"]["I"], I1),
    ("g_pp", R0["charges"]["poids_propre"], gpp1), ("g", R0["charges"]["g"], g1), ("q", R0["charges"]["q"], q1),
    ("p_d G", R0["combinaisons"][0]["p_d"], pG1), ("p_d G+Q", R0["combinaisons"][1]["p_d"], pGQ1),
    ("M_d G", R0["combinaisons"][0]["M_d"], MG1 / 1e6), ("M_d G+Q", R0["combinaisons"][1]["M_d"], MGQ1 / 1e6),
    ("V_d G", R0["combinaisons"][0]["V_d"], VG1 / 1e3), ("V_d G+Q", R0["combinaisons"][1]["V_d"], VGQ1 / 1e3),
    ("k_mod G", R0["combinaisons"][0]["kmod"], 0.60), ("k_mod G+Q", R0["combinaisons"][1]["kmod"], 0.80),
    ("k_h", R0["coefficients"]["kh"], 1.0), ("k_crit", R0["deversement"]["kcrit"], 1.0), ("γ_M", R0["coefficients"]["gamma_M"], 1.30),
    ("f_m,d G", V(R0, "flexion")["detail"][0]["resistance"], fmdG1), ("f_m,d G+Q", V(R0, "flexion")["detail"][1]["resistance"], fmdGQ1),
    ("σ_m,d G", V(R0, "flexion")["detail"][0]["valeur"], sigG1), ("σ_m,d G+Q", V(R0, "flexion")["detail"][1]["valeur"], sigGQ1),
    ("taux flexion G", V(R0, "flexion")["detail"][0]["taux"], sigG1 / fmdG1), ("taux flexion", V(R0, "flexion")["taux"], sigGQ1 / fmdGQ1),
    ("f_v,d G+Q", V(R0, "cisaillement")["detail"][1]["resistance"], fvdGQ1), ("τ_d G", V(R0, "cisaillement")["detail"][0]["valeur"], tauG1),
    ("τ_d G+Q", V(R0, "cisaillement")["valeur"], tauGQ1), ("taux cisaillement G", V(R0, "cisaillement")["detail"][0]["taux"], tauG1 / fvdG1),
    ("taux cisaillement", V(R0, "cisaillement")["taux"], tauGQ1 / fvdGQ1),
    ("l_ef,a", R0["appui"]["l_ef"], lefa1), ("l_1", R0["appui"]["l1"], 3900.0), ("k_c,90", R0["appui"]["kc90"], 1.5),
    ("σ_c,90,d G+Q", V(R0, "appui")["valeur"], s90GQ1), ("limite appui", V(R0, "appui")["limite"], 1.5 * fc90GQ1),
    ("taux appui G", V(R0, "appui")["detail"][0]["taux"], s90G1 / (1.5 * fc90G1)), ("taux appui", V(R0, "appui")["taux"], s90GQ1 / (1.5 * fc90GQ1)),
    ("w_g,m", R0["fleches"]["w_g_m"], wgm1), ("w_g,v", R0["fleches"]["w_g_v"], wgv1), ("w_q,m", R0["fleches"]["w_q_m"], wqm1),
    ("w_q,v", R0["fleches"]["w_q_v"], wqv1), ("w_inst", R0["fleches"]["w_inst"], winst1), ("w_fin", R0["fleches"]["w_fin"], wfin1),
    ("w_net,fin", V(R0, "fleche_fin")["valeur"], wfin1), ("limite inst", V(R0, "fleche_inst")["limite"], 4000.0 / 300.0),
    ("limite fin", V(R0, "fleche_fin")["limite"], 4000.0 / 250.0), ("taux w_inst", V(R0, "fleche_inst")["taux"], winst1 / (4000.0 / 300.0)),
    ("taux w_net,fin", V(R0, "fleche_fin")["taux"], wfin1 / (4000.0 / 250.0)), ("taux max", R0["taux_max"], wfin1 / 16.0),
    ("masse linéique", R0["masse_lineique"], 420.0 * 0.075 * 0.225), ("masse", R0["masse"], 420.0 * 0.075 * 0.225 * 4.1),
]
faux = [(n, a, b) for n, a, b in ex1 if not pres(a, b)]
chk(f"les {len(ex1)} grandeurs de l'exemple 1 (section, charges, deux combinaisons, flexion, cisaillement, appui, flèches, masse) : le moteur = le calcul à la main, à 1e-9",
    not faux, faux[:4])
chk("exemple 1, les chiffres : g_pp = 0,0695 kN/m ; M_d = 4,208 kN·m ; flexion 45,02 % ; cisaillement 22,68 % ; appui 18,70 % ; w_inst = 6,56 mm (49,19 %) ; "
    "w_net,fin = 8,99 mm (56,21 %) : conforme, la flèche finale nette gouverne",
    dec(gpp1, 0.0695, 4) and dec(MGQ1 / 1e6, 4.208, 3) and dec(100 * V(R0, "flexion")["taux"], 45.02, 2) and dec(100 * V(R0, "cisaillement")["taux"], 22.68, 2)
    and dec(100 * V(R0, "appui")["taux"], 18.70, 2) and dec(R0["fleches"]["w_inst"], 6.56, 2) and dec(100 * V(R0, "fleche_inst")["taux"], 49.19, 2)
    and dec(R0["fleches"]["w_net_fin"], 8.99, 2) and dec(100 * R0["taux_max"], 56.21, 2) and R0["statut"] == "ok" and R0["gouvernant"]["cle"] == "fleche_fin",
    (R0["taux_max"], R0["gouvernant"]))
chk("exemple 1 : les trois vérifications ELU nomment G + Q (k_mod moyen terme) ; cinq vérifications dans l'ordre de la note, deux groupes",
    [v["combinaison"] for v in R0["verifications"]] == ["GQ", "GQ", "GQ", None, None]
    and [(v["cle"], v["groupe"]) for v in R0["verifications"]] == [("flexion", "ELU"), ("cisaillement", "ELU"), ("appui", "ELU"), ("fleche_inst", "ELS"), ("fleche_fin", "ELS")])


# ====================================================================== 3. la référence : la conception § 1.4 refaite en N et en mm
print("\n=== 3. La référence : la méthode de la conception refaite ici en N et en mm, comparée au moteur ===")
KMOD_REF = {1: {"permanent": 0.60, "long": 0.70, "moyen": 0.80, "court": 0.90, "instantane": 1.10},
            2: {"permanent": 0.60, "long": 0.70, "moyen": 0.80, "court": 0.90, "instantane": 1.10},
            3: {"permanent": 0.50, "long": 0.55, "moyen": 0.65, "court": 0.70, "instantane": 0.90}}
KDEF_REF = {1: 0.60, 2: 0.80, 3: 2.00}
NATURES_REF = {"A": ("moyen", 0.3), "B": ("moyen", 0.3), "C": ("moyen", 0.6), "D": ("moyen", 0.6), "E": ("long", 0.8), "H": ("court", 0.0),
               "neige": ("court", 0.0), "chantier": ("court", 0.0)}


def reference(p):
    """Le calcul de la conception § 1.4 pour une poutre NORMALISÉE, écrit ici en N et en mm (p en N/mm, L en mm, M en
    N·mm, V en N), avec ``math`` et les tables transcrites dans ce fichier — rien du moteur."""
    m = TABLE[p["classe"]]
    massif = p["classe"].startswith("C")
    b, h, L = p["b"], p["h"], p["L"] * 1000.0
    A, W, I = b * h, b * h * h / 6.0, b * h ** 3 / 12.0
    c = p["charges"]
    G, Q = (c["g"] * c["entraxe"], c["q"] * c["entraxe"]) if c["mode"] == "surfacique" else (c["G"], c["Q"])
    g_pp = m["rho_mean"] * 9.81 * A * 1e-9                         # kg/m³ × N/kg × mm² × 10⁻⁹ = N/mm
    g, q = G + (g_pp if c["poids_propre"] else 0.0), Q
    duree, psi2 = (c["duree"], c["psi2"]) if c["nature"] == "autre" else NATURES_REF[c["nature"]]
    gamma_M = 1.30 if massif else 1.25
    ksys = 1.1 if p["systeme"] else 1.0
    if massif:
        kh = min((150.0 / h) ** 0.2, 1.3) if h < 150.0 else 1.0
    else:
        kh = min((600.0 / h) ** 0.1, 1.1) if h < 600.0 else 1.0
    # déversement
    mt = p["maintien"]
    if mt["type"] == "continu":
        l_ef, s_crit, lam, kcrit = None, None, None, 1.0
    else:
        l_ef = (0.9 * L if mt["type"] == "appuis" else min(mt["entraxe"] * 1000.0, 0.9 * L)) + 2.0 * h
        s_crit = 0.78 * b * b * m["E0_05"] / (h * l_ef)
        lam = math.sqrt(m["fm_k"] / s_crit)
        kcrit = 1.0 if lam <= 0.75 else (1.56 - 0.75 * lam if lam <= 1.4 else 1.0 / (lam * lam))
    # appui
    a, deb = p["appui"]["a"], p["appui"]["debord"]
    l1 = L - a
    l_ef_a = a + min(30.0, a, l1 / 2.0) + min(30.0, a, deb)
    kc90 = 1.5 if massif else 1.75
    if l1 < 2.0 * h or (not massif and a > 400.0):
        kc90 = 1.0
    combs = [("G", "permanent", p["gamma_G"] * g)] + ([("GQ", duree, p["gamma_G"] * g + p["gamma_Q"] * q)] if q > 0 else [])
    out = {"g_pp": g_pp, "g": g, "q": q, "kh": kh, "kcrit": kcrit, "l_ef": l_ef, "sigma_crit": s_crit, "lambda_rel": lam, "l_ef_a": l_ef_a,
           "kc90": kc90, "combs": {}}
    for cle, d, pd in combs:
        kmod = KMOD_REF[p["service"]][d]
        Md, Vd = pd * L * L / 8.0, pd * L / 2.0
        fmd, fvd, fc90d = kmod * kh * ksys * m["fm_k"] / gamma_M, kmod * ksys * m["fv_k"] / gamma_M, kmod * m["fc90_k"] / gamma_M
        sig, tau, s90 = Md / W, 1.5 * Vd / (0.67 * b * h), Vd / (b * l_ef_a)
        out["combs"][cle] = {"kmod": kmod, "p_d": pd, "M_d": Md / 1e6, "V_d": Vd / 1e3, "fm_d": fmd, "sigma_m": sig, "taux_m": sig / (kcrit * fmd),
                             "fv_d": fvd, "tau": tau, "taux_v": tau / fvd, "fc90_d": fc90d, "sigma_c90": s90, "taux_c90": s90 / (kc90 * fc90d)}

    def w(pp):
        return 5.0 * pp * L ** 4 / (384.0 * m["E0_mean"] * I) + 1.2 * pp * L * L / (8.0 * m["G_mean"] * A)
    kdef = KDEF_REF[p["service"]]
    w_inst = w(g) + w(q)
    w_fin = w(g) * (1.0 + kdef) + w(q) * (1.0 + psi2 * kdef)
    w_net = w_fin - p["contre_fleche"]
    out.update(w_g=w(g), w_q=w(q), w_inst=w_inst, w_fin=w_fin, w_net_fin=w_net, taux_inst=w_inst / (L / p["lim_inst"]),
               taux_fin=max(w_net, 0.0) / (L / p["lim_fin"]))
    taux = {"flexion": max(x["taux_m"] for x in out["combs"].values()), "cisaillement": max(x["taux_v"] for x in out["combs"].values())}
    if p["appui"]["verifier"]:
        taux["appui"] = max(x["taux_c90"] for x in out["combs"].values())
    taux["fleche_inst"], taux["fleche_fin"] = out["taux_inst"], out["taux_fin"]
    out["taux"] = taux
    out["taux_max"] = max(taux.values())
    out["statut"] = "nok" if out["taux_max"] > 1.0 else "ok"
    out["masse_lineique"] = m["rho_mean"] * A * 1e-6
    return out


def ecarts(mod, p, tol=1e-9):
    """Ce que le moteur ``mod`` rend de différent de la référence pour la poutre ``p``."""
    pn = mod.normaliser(p)
    r, ref = mod.verifier(p), reference(pn)
    f = []

    def cmp_(nom, a, b):
        if (a is None) != (b is None) or (a is not None and not pres(a, b, tol)):
            f.append(f"{nom} : {a} ≠ {b}")
    cmp_("g_pp", r["charges"]["poids_propre"], ref["g_pp"])
    cmp_("g", r["charges"]["g"], ref["g"])
    cmp_("q", r["charges"]["q"], ref["q"])
    cmp_("k_h", r["coefficients"]["kh"], ref["kh"])
    for k in ("kcrit", "l_ef", "sigma_crit", "lambda_rel"):
        cmp_(k, r["deversement"][k], ref[k])
    cmp_("l_ef,a", r["appui"]["l_ef"], ref["l_ef_a"])
    cmp_("k_c,90", r["appui"]["kc90"], ref["kc90"])
    if [c["cle"] for c in r["combinaisons"]] != list(ref["combs"]):
        f.append(f"combinaisons : {[c['cle'] for c in r['combinaisons']]} ≠ {list(ref['combs'])}")
    for c in r["combinaisons"]:
        x = ref["combs"].get(c["cle"], {})
        for k in ("kmod", "p_d", "M_d", "V_d"):
            cmp_(f"{k} {c['cle']}", c[k], x.get(k))
    for cle, (k_res, k_val, k_taux) in {"flexion": ("fm_d", "sigma_m", "taux_m"), "cisaillement": ("fv_d", "tau", "taux_v"),
                                        "appui": ("fc90_d", "sigma_c90", "taux_c90")}.items():
        v = next((v for v in r["verifications"] if v["cle"] == cle), None)
        if (v is None) != (cle not in ref["taux"]):
            f.append(f"{cle} : présence")
        if v is None:
            continue
        for d in v["detail"]:
            x = ref["combs"].get(d["combinaison"], {})
            cmp_(f"{cle} résistance {d['combinaison']}", d["resistance"], x.get(k_res))
            cmp_(f"{cle} valeur {d['combinaison']}", d["valeur"], x.get(k_val))
            cmp_(f"{cle} taux {d['combinaison']}", d["taux"], x.get(k_taux))
        cmp_(f"{cle} taux", v["taux"], ref["taux"][cle])
        gouv = max(ref["combs"], key=lambda k: ref["combs"][k][k_taux])
        if v["combinaison"] != gouv:
            f.append(f"{cle} : combinaison {v['combinaison']} ≠ {gouv}")
    for k in ("w_g", "w_q", "w_inst", "w_fin", "w_net_fin"):
        cmp_(k, r["fleches"][k], ref[k])
    cmp_("taux w_inst", V(r, "fleche_inst")["taux"], ref["taux_inst"])
    cmp_("taux w_net,fin", V(r, "fleche_fin")["taux"], ref["taux_fin"])
    cmp_("taux max", r["taux_max"], ref["taux_max"])
    cmp_("masse linéique", r["masse_lineique"], ref["masse_lineique"])
    if r["statut"] != ref["statut"]:
        f.append(f"statut {r['statut']} ≠ {ref['statut']}")
    gov = max(ref["taux"], key=lambda k: ref["taux"][k])
    if not pres(ref["taux"][r["gouvernant"]["cle"]], ref["taux"][gov]):
        f.append(f"gouvernant {r['gouvernant']['cle']} ≠ {gov}")
    return f


EXEMPLES = {
    "1 · la poutre de l'ouverture": P0,
    "2 · G seule gouverne (G = 3, Q = 0,5 kN/m)": poutre(charges__G=3.0, charges__Q=0.5),
    "3 · le déversement joue (45 × 245 sur 5 m, maintien aux appuis)": poutre(section="45x245", L=5.0, maintien__type="appuis"),
    "4 · lamellé-collé GL24h 120 × 400 sur 8 m": poutre(classe="GL24h", section="120x400", L=8.0, charges__G=3.0, charges__Q=4.0),
    "5 · l'appui gouverne (a = 40 mm)": poutre(section="100x300", L=3.0, charges__G=8.0, charges__Q=12.0, appui__a=40.0),
    "6 · classe de service 3": poutre(service=3),
}
for nom, p in EXEMPLES.items():
    ec = ecarts(M, p)
    chk(f"exemple {nom} : le moteur = la référence (chaque grandeur, chaque combinaison, à 1e-9)", not ec, ec[:3])
r2, r3, r4, r5, r6 = (M.verifier(EXEMPLES[k]) for k in list(EXEMPLES)[1:])
# à la main : g = 3 + 0,0695 = 3,0695 ; G seule : 1,35 × 3,0695 × 4²/8 = 8,288 kN·m → 13,10 / (0,6 × 24/1,3 = 11,08) = 118,2 % ;
# G + Q : (4,144 + 0,75) × 2 = 9,788 kN·m → 15,47 / (0,8 × 24/1,3 = 14,77) = 104,7 %
chk("exemple 2 : G seule gouverne la flexion (118,2 % contre 104,7 % pour G + Q : non conforme), le cisaillement et l'appui ; G + Q reste calculée et montrée",
    V(r2, "flexion")["combinaison"] == "G" and V(r2, "cisaillement")["combinaison"] == "G" and V(r2, "appui")["combinaison"] == "G"
    and pres(V(r2, "flexion")["taux"], (1.35 * (3.0 + gpp1) * 4000.0 ** 2 / 8.0 / W1) / (0.6 * 24.0 / 1.3))
    and dec(100 * V(r2, "flexion")["detail"][0]["taux"], 118.2, 1) and dec(100 * V(r2, "flexion")["detail"][1]["taux"], 104.7, 1) and r2["statut"] == "nok",
    [(d["combinaison"], 100 * d["taux"]) for d in V(r2, "flexion")["detail"]])
chk("exemple 3 : l_ef = 0,9 × 5000 + 2 × 245 = 4990 mm ; σ_m,crit = 0,78 × 45² × 7400 / (245 × 4990) = 9,56 N/mm² ; λ_rel,m = √(24/9,56) = 1,584 > 1,4 → "
    "k_crit = 1/λ² = 0,398 ; la flexion passe de 97,4 % (continu) à 244,4 % : non conforme",
    r3["deversement"]["l_ef"] == 4990.0 and dec(r3["deversement"]["sigma_crit"], 9.56, 2) and dec(r3["deversement"]["lambda_rel"], 1.584, 3)
    and r3["deversement"]["branche"] == "elance" and dec(r3["deversement"]["kcrit"], 0.398, 3) and dec(100 * V(r3, "flexion")["taux"], 244.4, 1)
    and pres(r3["deversement"]["sigma_crit"], 0.78 * 45.0 ** 2 * 7400.0 / (245.0 * 4990.0))
    and dec(100 * V(M.verifier(poutre(section="45x245", L=5.0)), "flexion")["taux"], 97.4, 1) and r3["statut"] == "nok",
    (r3["deversement"], 100 * V(r3, "flexion")["taux"]))
chk("exemple 4 : lamellé-collé — γ_M = 1,25 ; k_h = min((600/400)^0,1 ; 1,1) = 1,0414 ; k_c,90 = 1,75 ; ρ_mean 420 → g_pp = 0,198 kN/m",
    r4["coefficients"]["gamma_M"] == 1.25 and pres(r4["coefficients"]["kh"], (600.0 / 400.0) ** 0.1) and dec(r4["coefficients"]["kh"], 1.0414, 4)
    and r4["appui"]["kc90"] == 1.75 and dec(r4["charges"]["poids_propre"], 0.198, 3) and r4["materiau"]["type"] == "lamelle", r4["coefficients"])
chk("exemple 5 : a = 40 mm → l_ef,a = 40 + 30 + 0 = 70 mm ; la compression à l'appui gouverne et dépasse 100 %",
    r5["appui"]["l_ef"] == 70.0 and r5["gouvernant"]["cle"] == "appui" and r5["taux_max"] > 1.0 and r5["statut"] == "nok", (r5["gouvernant"], r5["appui"]))
chk("exemple 6 : classe de service 3 — k_mod 0,50 (permanent) et 0,65 (moyen terme), k_def = 2,0 : w_fin = 3·w_g + 1,6·w_q",
    [c["kmod"] for c in r6["combinaisons"]] == [0.50, 0.65] and r6["coefficients"]["kdef"] == 2.0
    and pres(r6["fleches"]["w_fin"], 3.0 * r6["fleches"]["w_g"] + 1.6 * r6["fleches"]["w_q"]), r6["combinaisons"])


def tirage(rng, k):
    """Une poutre tirée au sort dans le domaine (les contrôles croisés tenus)."""
    classe = rng.choice(list(T))
    typ = "massif" if classe.startswith("C") else "lamelle"
    p = {"nom": f"T{k}", "classe": classe, "L": round(rng.uniform(0.8, 14.0), 2)}
    if rng.random() < 0.6:
        s = rng.choice(E.sections(typ))
        p.update(standard=True, section=s["nom"])
    else:
        p.update(standard=False, b=float(rng.randint(30, 300)), h=float(rng.randint(60, 900)))
    nature = rng.choice(list(D.CLES_NATURE))
    ch = {"mode": rng.choice(["lineaire", "surfacique"]), "G": round(rng.uniform(0, 8), 2), "Q": rng.choice([0.0, round(rng.uniform(0, 10), 2)]),
          "g": round(rng.uniform(0, 4), 2), "q": round(rng.uniform(0, 6), 2), "entraxe": round(rng.uniform(0.3, 1.2), 2),
          "poids_propre": rng.random() < 0.8, "nature": nature}
    if nature == "autre":
        ch.update(duree=rng.choice(list(D.DUREES)), psi2=round(rng.uniform(0, 1), 2))
    p["charges"] = ch
    p["appui"] = {"verifier": rng.random() < 0.85, "a": float(rng.randint(20, 500)), "debord": float(rng.choice([0, 0, 10, 25, 60, 300]))}
    typ_m = rng.choice(list(D.MAINTIENS))
    p["maintien"] = {"type": typ_m, "entraxe": round(rng.uniform(0.2, p["L"]), 2)}
    p.update(service=rng.choice([1, 2, 3]), gamma_G=rng.choice([1.0, 1.35]), gamma_Q=rng.choice([1.0, 1.5]), systeme=rng.random() < 0.3,
             lim_inst=rng.choice(list(D.LIMITES_INST)), lim_fin=rng.choice(list(D.LIMITES_FIN)), contre_fleche=rng.choice([0.0, 0.0, 5.0, 40.0]))
    return p


rng = random.Random(20261110)
TIRAGES = [tirage(rng, k) for k in range(400)]
ec_t = [(p["nom"], e[0]) for p in TIRAGES for e in [ecarts(M, p)] if e]
chk(f"{len(TIRAGES)} poutres tirées au sort (18 classes, sections standard et sur mesure, deux modes de charge, 9 natures, 3 maintiens, appui, effet système, "
    "contre-flèche) : le moteur = la référence partout", not ec_t, ec_t[:3])
res_t = [M.verifier(p) for p in TIRAGES]
couv = {
    "G seule gouverne une vérification": any(v["combinaison"] == "G" and len(v["detail"]) == 2 for r in res_t for v in r["verifications"]),
    "G + Q gouverne": any(v["combinaison"] == "GQ" for r in res_t for v in r["verifications"]),
    "sans Q (une seule combinaison)": any(len(r["combinaisons"]) == 1 for r in res_t),
    "k_crit palier": any(r["deversement"]["branche"] == "palier" for r in res_t), "k_crit linéaire": any(r["deversement"]["branche"] == "lineaire" for r in res_t),
    "k_crit élancé": any(r["deversement"]["branche"] == "elance" for r in res_t), "k_h > 1": any(r["coefficients"]["kh"] > 1 for r in res_t),
    "k_c,90 ramené à 1 (distance)": any(r["appui"]["raison"] == "distance" for r in res_t),
    "k_c,90 ramené à 1 (lamellé, a > 400)": any(r["appui"]["raison"] == "longueur" for r in res_t),
    "appui non vérifié": any(not r["appui"]["verifie"] for r in res_t), "flèche nette négative": any(r["fleches"]["w_net_fin"] < 0 for r in res_t),
    "conforme": any(r["statut"] == "ok" for r in res_t), "non conforme": any(r["statut"] == "nok" for r in res_t),
    "chaque vérification gouverne au moins une fois": {r["gouvernant"]["cle"] for r in res_t} >= {"flexion", "cisaillement", "appui", "fleche_inst", "fleche_fin"},
}
chk("les tirages atteignent chaque branche de la méthode (" + ", ".join(couv) + ")", all(couv.values()), [k for k, v in couv.items() if not v])

# ====================================================================== 4. les deux combinaisons
print("\n=== 4. Les deux combinaisons ELU, chacune avec son k_mod ===")
anc = M.verifier(poutre(section="63x200", L=3.0, charges__G=3.0, charges__Q=0.5, charges__poids_propre=False))
# à la main : W = 63 × 200²/6 = 420 000 mm³ ; G seule : 1,35 × 3 × 3²/8 = 4,556 kN·m → 10,85 / (0,6 × 24/1,3 = 11,08) = 97,9 % ;
# G + Q : (4,05 + 0,75) × 9/8 = 5,4 kN·m → 12,86 / (0,8 × 24/1,3 = 14,77) = 87,1 %
chk("le défaut n° 2 de l'ancien écran (63 × 200 C24, L = 3 m, G = 3, Q = 0,5, sans poids propre) : G seule gouverne la flexion à 97,9 % "
    "(4,556 × 10⁶/420 000 = 10,85 contre 0,6 × 24/1,3 = 11,08), G + Q ne donne que 87,1 % ; le moteur nomme G seule",
    V(anc, "flexion")["combinaison"] == "G" and pres(V(anc, "flexion")["taux"], (1.35 * 3.0 * 3000.0 ** 2 / 8.0 / 420000.0) / (0.6 * 24.0 / 1.3))
    and dec(100 * V(anc, "flexion")["taux"], 97.9, 1) and dec(100 * V(anc, "flexion")["detail"][1]["taux"], 87.1, 1),
    [(d["combinaison"], 100 * d["taux"]) for d in V(anc, "flexion")["detail"]])
sansq = M.verifier(poutre(charges__Q=0.0))
chk("sans charge variable (Q = 0) : une seule combinaison, G seule, k_mod permanent — pas de combinaison G + Q au k_mod plus favorable",
    [c["cle"] for c in sansq["combinaisons"]] == ["G"] and sansq["combinaisons"][0]["kmod"] == 0.6 and all(len(v["detail"]) == 1 for v in sansq["verifications"][:3]))
court = M.verifier(poutre(charges__nature="neige"))
chk("la nature de Q fixe son k_mod : neige → court terme → 0,90 ; stockage → long terme → 0,70 ; G seule reste à 0,60",
    court["combinaisons"][1]["kmod"] == 0.9 and M.verifier(poutre(charges__nature="E"))["combinaisons"][1]["kmod"] == 0.7
    and court["combinaisons"][0]["kmod"] == 0.6)
chk("la combinaison ELU qui gouverne (diagrammes V et M) est celle du taux ELU le plus fort : G + Q à l'ouverture, G seule dans l'exemple 2",
    R0["combinaison_elu"] == "GQ" and r2["combinaison_elu"] == "G" and pres(r2["efforts"]["M_d"], r2["combinaisons"][0]["M_d"])
    and pres(R0["efforts"]["R"], R0["combinaisons"][1]["V_d"]))

# ====================================================================== 5. k_h, k_crit, l_ef, k_c,90
print("\n=== 5. k_h, k_crit aux trois branches, l_ef, k_c,90 et la longueur efficace d'appui ===")
chk("k_h du bois massif (référence 150 mm) : h = 100 → (150/100)^0,2 = 1,0845 ; h = 40 → plafonné à 1,3 ; h = 150 et au-delà → 1 (jamais < 1 : l'ancien "
    "écran réduisait au-dessus de 150 mm)",
    pres(M.k_h("massif", 100.0), 1.5 ** 0.2) and dec(M.k_h("massif", 100.0), 1.0845, 4) and M.k_h("massif", 40.0) == 1.3
    and M.k_h("massif", 150.0) == 1.0 and M.k_h("massif", 225.0) == 1.0 and M.k_h("massif", 149.0) > 1.0)
chk("k_h du lamellé-collé (référence 600 mm) : h = 300 → (600/300)^0,1 = 1,0718 ; h = 200 → plafonné à 1,1 ; h = 600 et au-delà → 1 ; "
    "il n'est PAS celui du massif (à h = 300 : 1,0718 contre 1)",
    pres(M.k_h("lamelle", 300.0), 2.0 ** 0.1) and dec(M.k_h("lamelle", 300.0), 1.0718, 4) and M.k_h("lamelle", 200.0) == 1.1
    and M.k_h("lamelle", 600.0) == 1.0 and M.k_h("lamelle", 640.0) == 1.0 and M.k_h("massif", 300.0) == 1.0)
chk("k_crit (6.34) : 1 jusqu'à λ = 0,75 ; 1,56 − 0,75·λ jusqu'à 1,4 (λ = 1 → 0,81) ; 1/λ² au-delà (λ = 2 → 0,25) ; continu aux deux raccords "
    "(0,9975 à 0,75 ; 0,51 à 1,4)",
    M.k_crit(0.5) == 1.0 and M.k_crit(0.75) == 1.0 and pres(M.k_crit(1.0), 0.81) and pres(M.k_crit(2.0), 0.25) and pres(M.k_crit(1.4), 1.56 - 0.75 * 1.4)
    and abs(M.k_crit(0.7500001) - 0.9975) < 1e-6 and abs(M.k_crit(1.4000001) - 1.0 / 1.4 ** 2) < 1e-6 and abs(M.k_crit(1.4) - 0.51) < 1e-12)
pal = M.verifier(poutre(section="100x200", L=3.0, maintien__type="appuis"))
lin = M.verifier(poutre(section="63x225", L=4.0, maintien__type="appuis"))
# à la main, 63 × 225 sur 4 m : l_ef = 3600 + 450 = 4050 ; σ_crit = 0,78 × 63² × 7400/(225 × 4050) = 25,14 ; λ = √(24/25,14) = 0,977 ; k_crit = 1,56 − 0,733 = 0,827
# 100 × 200 sur 3 m : l_ef = 2700 + 400 = 3100 ; σ_crit = 0,78 × 100² × 7400/(200 × 3100) = 93,10 ; λ = √(24/93,10) = 0,508
chk("les trois branches par des poutres : 100 × 200 sur 3 m (λ = 0,508 : palier, k_crit = 1) ; 63 × 225 sur 4 m (λ = 0,977 : k_crit = 0,827) ; "
    "45 × 245 sur 5 m (λ = 1,584 : 0,398)",
    pal["deversement"]["branche"] == "palier" and pal["deversement"]["kcrit"] == 1.0 and dec(pal["deversement"]["lambda_rel"], 0.508, 3)
    and lin["deversement"]["branche"] == "lineaire" and dec(lin["deversement"]["lambda_rel"], 0.977, 3) and dec(lin["deversement"]["kcrit"], 0.827, 3)
    and pres(lin["deversement"]["sigma_crit"], 0.78 * 63.0 ** 2 * 7400.0 / (225.0 * 4050.0)), (pal["deversement"], lin["deversement"]))
ent = M.verifier(poutre(section="45x245", L=5.0, maintien__type="entretoises", maintien__entraxe=1.25))
ent_l = M.verifier(poutre(section="45x245", L=5.0, maintien__type="entretoises", maintien__entraxe=4.8))
# à la main : σ_crit = 0,78 × 45² × 7400/(245 × 1740) = 27,42 ; λ = √(24/27,42) = 0,9356 ; k_crit = 1,56 − 0,75 × 0,9356 = 0,858
chk("entretoises à 1,25 m : l_ef = 1250 + 2 × 245 = 1740 mm (moment pris constant), k_crit remonte de 0,398 à 0,858 ; un entraxe de 4,8 m (plus que "
    "0,9·L) ne rend pas la poutre plus élancée que sans entretoise : l_ef = min(4800 ; 4500) + 490 = 4990 mm",
    ent["deversement"]["l_ef"] == 1740.0 and dec(ent["deversement"]["kcrit"], 0.858, 3) and ent_l["deversement"]["l_ef"] == 4990.0
    and pres(ent_l["deversement"]["kcrit"], r3["deversement"]["kcrit"]), (ent["deversement"], ent_l["deversement"]["l_ef"]))
chk("maintien continu : pas de longueur efficace, k_crit = 1 — et l'hypothèse (appuis maintenus en torsion) est écrite",
    R0["deversement"]["l_ef"] is None and R0["deversement"]["kcrit"] == 1.0 and any("maintenus en torsion" in h for h in R0["hypotheses"]))
proche = M.verifier(poutre(section="100x300", L=0.65, appui__a=100.0))
chk("k_c,90 : appuis rapprochés (l_1 = 650 − 100 = 550 mm < 2·h = 600 mm) → ramené de 1,5 à 1,0, avec l'avertissement « kc90_distance » qui donne les "
    "deux longueurs", proche["appui"]["kc90"] == 1.0 and proche["appui"]["raison"] == "distance"
    and any(a["code"] == "kc90_distance" and "550" in a["message"] and "600" in a["message"] for a in proche["avertissements"]), proche["avertissements"])
gl_long = M.verifier(poutre(classe="GL24h", section="120x400", L=8.0, appui__a=450.0))
gl_ok = M.verifier(poutre(classe="GL24h", section="120x400", L=8.0, appui__a=400.0))
chk("k_c,90 du lamellé-collé : 1,75 jusqu'à a = 400 mm compris ; a = 450 mm → 1,0 avec l'avertissement « kc90_longueur » ; le massif garde 1,5 à 450 mm",
    gl_ok["appui"]["kc90"] == 1.75 and gl_long["appui"]["kc90"] == 1.0 and any(a["code"] == "kc90_longueur" for a in gl_long["avertissements"])
    and M.verifier(poutre(section="100x300", L=8.0, appui__a=450.0))["appui"]["kc90"] == 1.5)
deb = M.verifier(poutre(appui__debord=50.0))
deb_c = M.verifier(poutre(appui__debord=12.0))
petit = M.verifier(poutre(appui__a=20.0, appui__debord=50.0))
chk("longueur efficace d'appui : sans débord 100 + 30 = 130 ; débord de 50 → 100 + 30 + 30 = 160 ; débord de 12 → 142 ; a = 20 → 20 + 20 + 20 = 60 "
    "(jamais plus que a de chaque côté)", R0["appui"]["l_ef"] == 130.0 and deb["appui"]["l_ef"] == 160.0 and deb_c["appui"]["l_ef"] == 142.0
    and petit["appui"]["l_ef"] == 60.0, (deb["appui"]["l_ef"], deb_c["appui"]["l_ef"], petit["appui"]["l_ef"]))
court_l1 = M.verifier(poutre(section="38x89", L=0.5, appui__a=460.0))
chk("côté travée, la majoration ne dépasse pas l_1/2 : L = 500, a = 460 → l_1 = 40 → 460 + 20 + 0 = 480 mm", court_l1["appui"]["l_ef"] == 480.0, court_l1["appui"])
sans_ap = M.verifier(poutre(appui__verifier=False))
chk("l'appui retiré de la note : quatre vérifications, et l'avertissement « appui_non_verifie » le dit",
    [v["cle"] for v in sans_ap["verifications"]] == ["flexion", "cisaillement", "fleche_inst", "fleche_fin"]
    and any(a["code"] == "appui_non_verifie" for a in sans_ap["avertissements"]) and not any(a["code"] == "appui_non_verifie" for a in R0["avertissements"]))


# ====================================================================== 6. les unités : la flèche refaite en SI
print("\n=== 6. Unités : la flèche refaite en unités SI par le théorème de la charge unité ===")
def simpson(f, a, b, n=400):
    hh = (b - a) / n
    return hh / 3.0 * (f(a) + f(b) + 4.0 * sum(f(a + (2 * i - 1) * hh) for i in range(1, n // 2 + 1)) + 2.0 * sum(f(a + 2 * i * hh) for i in range(1, n // 2)))


def fleche_si(p_kN_m, L_m, E_MPa, I_mm4, G_MPa, A_mm2):
    """w(L/2) en mm, par ∫ M·m̄/(E·I) dx + 1,2·∫ V·v̄/(G·A) dx (charge unité à mi-portée), tout en N, m et Pa."""
    p, EI, GA = p_kN_m * 1000.0, E_MPa * 1e6 * I_mm4 * 1e-12, G_MPa * 1e6 * A_mm2 * 1e-6
    flexion = 2.0 * simpson(lambda x: (p * x * (L_m - x) / 2.0) * (x / 2.0) / EI, 0.0, L_m / 2.0)
    tranchant = 2.0 * simpson(lambda x: 1.2 * (p * (L_m / 2.0 - x)) * 0.5 / GA, 0.0, L_m / 2.0)
    return 1000.0 * (flexion + tranchant)


w_si = fleche_si(g1, 4.0, 11000.0, I1, 690.0, A1) + fleche_si(0.8, 4.0, 11000.0, I1, 690.0, A1)
chk(f"w_inst de l'exemple 1 refaite en N, m et Pa par intégration numérique : {w_si:.4f} mm = le moteur ({R0['fleches']['w_inst']:.4f} mm) — quelques "
    "millimètres, pas des mètres (l'ancien écran : 1 000 fois plus)", abs(w_si - R0["fleches"]["w_inst"]) < 1e-6 * w_si and 1.0 < R0["fleches"]["w_inst"] < 20.0,
    (w_si, R0["fleches"]["w_inst"]))
si_t = []
for p, r in list(zip(TIRAGES, res_t))[:60]:
    w = fleche_si(r["charges"]["g"], r["geometrie"]["L"], r["materiau"]["E0_mean"], r["geometrie"]["I"], r["materiau"]["G_mean"], r["geometrie"]["A"]) \
        + fleche_si(r["charges"]["q"], r["geometrie"]["L"], r["materiau"]["E0_mean"], r["geometrie"]["I"], r["materiau"]["G_mean"], r["geometrie"]["A"])
    if abs(w - r["fleches"]["w_inst"]) > 1e-6 * max(w, 1e-9):
        si_t.append((p["nom"], w, r["fleches"]["w_inst"]))
chk("… et sur 60 tirages : la flèche instantanée du moteur = l'intégration en SI", not si_t, si_t[:2])
chk("1 kN/m = 1 N/mm : M_d (kN·m) = p_d·L²/8 avec L en m, et σ_m,d = M_d × 10⁶ / W — 4,208 kN·m sur 632 812 mm³ = 6,65 N/mm²",
    dec(R0["combinaisons"][1]["M_d"] * 1e6 / R0["geometrie"]["W"], 6.65, 2) and pres(V(R0, "flexion")["valeur"], R0["combinaisons"][1]["M_d"] * 1e6 / R0["geometrie"]["W"]))

# ====================================================================== 7. ELS
print("\n=== 7. ELS : flèche instantanée, flèche finale, contre-flèche, limites ===")
fl = R0["fleches"]
chk("w_fin = w_g·(1 + k_def) + w_q·(1 + ψ₂·k_def) — et non (1 + k_def)·w(G + ψ₂·Q) de l'ancien écran, qui oubliait (1 − ψ₂)·w_q : "
    f"{fl['w_fin']:.3f} mm contre {(1 + 0.6) * (fl['w_g'] + 0.3 * fl['w_q']):.3f}",
    pres(fl["w_fin"], fl["w_g"] * 1.6 + fl["w_q"] * 1.18) and fl["w_fin"] - 1.6 * (fl["w_g"] + 0.3 * fl["w_q"]) > 0.7 * fl["w_q"] - 1e-9)
psi = {k: M.verifier(poutre(charges__nature=k))["fleches"] for k in ("A", "C", "E", "H")}
chk("ψ₂ suit la nature : habitation 0,3, réunion 0,6, stockage 0,8, toiture 0 — w_fin croît avec ψ₂, w_inst ne bouge pas",
    psi["H"]["w_fin"] < psi["A"]["w_fin"] < psi["C"]["w_fin"] < psi["E"]["w_fin"] and len({round(x["w_inst"], 12) for x in psi.values()}) == 1
    and pres(psi["H"]["w_fin"], psi["H"]["w_g"] * 1.6 + psi["H"]["w_q"]) and pres(psi["E"]["w_fin"], psi["E"]["w_g"] * 1.6 + psi["E"]["w_q"] * 1.48))
cf = M.verifier(poutre(contre_fleche=5.0))
chk("contre-flèche de 5 mm : w_net,fin = w_fin − 5 ; w_inst n'en dépend pas", pres(cf["fleches"]["w_net_fin"], fl["w_fin"] - 5.0)
    and pres(V(cf, "fleche_fin")["taux"], (fl["w_fin"] - 5.0) / 16.0) and cf["fleches"]["w_inst"] == fl["w_inst"])
cf2 = M.verifier(poutre(contre_fleche=20.0))
chk("contre-flèche de 20 mm (plus que la flèche finale) : w_net,fin < 0, le taux est compté nul et l'avertissement « contre_fleche » le dit",
    cf2["fleches"]["w_net_fin"] < 0 and V(cf2, "fleche_fin")["taux"] == 0.0 and V(cf2, "fleche_fin")["valeur"] < 0
    and any(a["code"] == "contre_fleche" for a in cf2["avertissements"]))
lims = [(n, V(M.verifier(poutre(lim_inst=n)), "fleche_inst")["limite"]) for n in D.LIMITES_INST] + [(n, V(M.verifier(poutre(lim_fin=n)), "fleche_fin")["limite"]) for n in D.LIMITES_FIN]
chk("les limites : L/n pour chaque diviseur des deux menus (4000/200 … 4000/500 ; 4000/150 … 4000/350)", all(pres(lim, 4000.0 / n) for n, lim in lims), lims)
sc2 = M.verifier(poutre(service=2))
chk("classe de service 2 : mêmes k_mod que la classe 1, k_def = 0,8", [c["kmod"] for c in sc2["combinaisons"]] == [0.6, 0.8] and sc2["coefficients"]["kdef"] == 0.8)

# ====================================================================== 8. les charges
print("\n=== 8. Charges : surfacique, poids propre, nature « autre » ===")
surf = M.verifier(poutre(charges__mode="surfacique"))
chk("saisie surfacique : G = g_s·e = 1,5 × 0,40 = 0,6 kN/m et Q = 2,0 × 0,40 = 0,8 kN/m — la poutre neuve donne les mêmes taux dans les deux modes",
    pres(surf["charges"]["G"], 0.6) and pres(surf["charges"]["Q"], 0.8) and pres(surf["taux_max"], R0["taux_max"]) and surf["charges"]["entraxe"] == 0.4
    and R0["charges"]["entraxe"] is None)
surf2 = M.verifier(poutre(charges__mode="surfacique", charges__g=2.0, charges__q=3.0, charges__entraxe=0.6))
chk("… g_s = 2, q_s = 3 kN/m², entraxe 0,60 m → G = 1,2 et Q = 1,8 kN/m ; les charges linéaires gardées en réserve ne jouent pas",
    pres(surf2["charges"]["G"], 1.2) and pres(surf2["charges"]["Q"], 1.8) and surf2["poutre"]["charges"]["G"] == 0.6)
sans_pp = M.verifier(poutre(charges__poids_propre=False))
chk("poids propre : compté par défaut (g = G + 0,0695) ; décoché, g = G, et l'avertissement « poids_propre » donne sa valeur",
    pres(R0["charges"]["g"], 0.6 + R0["charges"]["poids_propre"]) and sans_pp["charges"]["g"] == 0.6 and pres(sans_pp["charges"]["poids_propre"], gpp1)
    and any(a["code"] == "poids_propre" and "0,070" in a["message"] for a in sans_pp["avertissements"]), sans_pp["avertissements"])
autre = M.verifier(poutre(charges__nature="autre", charges__duree="instantane", charges__psi2=0.5))
chk("nature « autre » : la durée et ψ₂ saisis jouent (instantané → k_mod 1,10 ; ψ₂ = 0,5)",
    autre["combinaisons"][1]["kmod"] == 1.1 and autre["coefficients"]["psi2"] == 0.5 and autre["coefficients"]["nature_libre"] is True)
vide = M.verifier(poutre(charges__G=0.0, charges__Q=0.0, charges__poids_propre=False))
chk("aucune charge : tous les taux nuls, conforme, et l'avertissement « sans_charge »", vide["taux_max"] == 0.0 and vide["statut"] == "ok"
    and any(a["code"] == "sans_charge" for a in vide["avertissements"]))
sysm = M.verifier(poutre(systeme=True))
chk("effet système : f_m,d et f_v,d × 1,1 (pas f_c,90,d) ; l'hypothèse est écrite",
    pres(V(sysm, "flexion")["limite"], 1.1 * V(R0, "flexion")["limite"]) and pres(V(sysm, "cisaillement")["limite"], 1.1 * V(R0, "cisaillement")["limite"])
    and pres(V(sysm, "appui")["limite"], V(R0, "appui")["limite"]) and any("k_sys" in h for h in sysm["hypotheses"]))
sys_e = M.verifier(poutre(systeme=True, charges__mode="surfacique", charges__entraxe=0.9))
chk("effet système avec un entraxe de 0,90 m : avertissement « systeme_entraxe » ; pas à 0,40 m",
    any(a["code"] == "systeme_entraxe" for a in sys_e["avertissements"])
    and not any(a["code"] == "systeme_entraxe" for a in M.verifier(poutre(systeme=True, charges__mode="surfacique"))["avertissements"]))

# ====================================================================== 9. les invariants
print("\n=== 9. Invariants ===")
i1 = M.verifier(poutre(standard=False, b=60.0, h=220.0, charges__poids_propre=False))
i2 = M.verifier(poutre(standard=False, b=120.0, h=220.0, charges__poids_propre=False))
chk("b doublé (sans poids propre) : σ_m,d, τ_d, σ_c,90,d et les flèches divisées par deux",
    all(pres(V(i2, k)["valeur"], V(i1, k)["valeur"] / 2.0) for k in ("flexion", "cisaillement", "appui", "fleche_inst", "fleche_fin")))
i3 = M.verifier(poutre(standard=False, b=60.0, h=220.0, charges__poids_propre=False, charges__G=1.2, charges__Q=1.6))
chk("charges doublées (sans poids propre) : tous les taux doublés", all(pres(V(i3, v["cle"])["taux"], 2.0 * v["taux"]) for v in i1["verifications"]))
i4 = M.verifier(poutre(standard=False, b=60.0, h=220.0, charges__poids_propre=False, L=8.0))
chk("portée doublée : M_d × 4, V_d × 2, flèche de flexion × 16, flèche d'effort tranchant × 4",
    pres(i4["combinaisons"][1]["M_d"], 4 * i1["combinaisons"][1]["M_d"]) and pres(i4["combinaisons"][1]["V_d"], 2 * i1["combinaisons"][1]["V_d"])
    and pres(i4["fleches"]["w_g_m"], 16 * i1["fleches"]["w_g_m"]) and pres(i4["fleches"]["w_g_v"], 4 * i1["fleches"]["w_g_v"]))
mono = [M.verifier(poutre(charges__Q=q))["taux_max"] for q in (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)]
chk("le taux maximal ne baisse jamais quand Q croît", all(a <= b + 1e-12 for a, b in zip(mono, mono[1:])), mono)
chk("le taux à 100 % exactement est CONFORME, juste au-dessus NON conforme (seuil du code couleur du bureau)", M._etat(1.0) == "ok" and M._etat(1.0 + 1e-12) == "nok")
chk("statut : « nok » dès qu'une vérification dépasse 100 % ; le gouvernant est la vérification au taux le plus fort, avec sa combinaison",
    all((r["statut"] == "nok") == any(v["taux"] > 1.0 for v in r["verifications"]) and r["taux_max"] == max(v["taux"] for v in r["verifications"])
        and V(r, r["gouvernant"]["cle"])["taux"] == r["taux_max"] and r["gouvernant"]["combinaison"] == V(r, r["gouvernant"]["cle"])["combinaison"] for r in res_t))


# ====================================================================== 10. les formules : réévaluées à partir de leurs termes
print("\n=== 10. Formules : chaque résultat se recalcule à partir de son gabarit et de ses seuls termes ===")
def evaluer(gabarit, termes):
    """Un gabarit (« {γ_G} × {g} + {γ_Q} × {q} », « min((150 / {h})^0,2 ; 1,3) », « 5 × {g} × {L}⁴ / (384 × … ) ») évalué avec
    ses termes par un petit interpréteur écrit ici : virgules décimales, ×, −, ², ³, ⁴, ^, √, 10³, 10⁶, 10⁹, min(a ; b)."""
    vals = {t["symbole"]: t["valeur"] for t in termes}
    # les constantes à la virgule (HORS des accolades : « {E_0,05} » est un nom), puis chaque terme par sa valeur
    morceaux = re.split(r"(\{[^{}]+\})", gabarit)
    s = "".join(m_ if m_.startswith("{") else re.sub(r"(\d),(\d)", r"\1.\2", m_).replace("10⁹", "1e9").replace("10⁶", "1e6").replace("10³", "1e3")
                for m_ in morceaux)
    s = re.sub(r"\{([^{}]+)\}", lambda m_: "(" + repr(float(vals[m_.group(1)])) + ")", s)
    s = s.replace("×", "*").replace("−", "-").replace(";", ",").replace("^", "**").replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    s = s.replace("√(", "sqrt(")
    return eval(s, {"__builtins__": {}}, {"sqrt": math.sqrt, "min": min})  # noqa: S307 — des nombres et des opérateurs


def fautes_formules(resultats):
    f, n, lus = [], 0, 0
    for r in resultats:
        cles = [b["cle"] for b in r["calcul"]]
        if len(set(cles)) != len(cles):
            f.append(f"blocs en double : {cles}")
        for b in r["calcul"]:
            for fx in b["formules"]:
                if fx["gabarit"] is None:
                    lus += 1
                    if not fx["expression"] or not isinstance(fx["resultat"], (int, float)):
                        f.append(f"{b['cle']} {fx['symbole']} : valeur lue sans origine")
                    continue
                n += 1
                noms = set(re.findall(r"\{([^{}]+)\}", fx["gabarit"]))
                if noms != {t["symbole"] for t in fx["termes"]}:
                    f.append(f"{b['cle']} {fx['symbole']} : termes {sorted(t['symbole'] for t in fx['termes'])} ≠ gabarit {sorted(noms)}")
                    continue
                try:
                    val = evaluer(fx["gabarit"], fx["termes"])
                except Exception as e:  # noqa: BLE001 — un gabarit illisible est une faute
                    f.append(f"{b['cle']} {fx['symbole']} : illisible ({e})")
                    continue
                if not pres(val, fx["resultat"]):
                    f.append(f"{b['cle']} {fx['symbole']} = {fx['gabarit']} → {val} ≠ {fx['resultat']}")
                if fx["expression"] != re.sub(r"\{([^{}]+)\}", r"\1", fx["gabarit"]).replace(" × ", "·"):
                    f.append(f"{b['cle']} {fx['symbole']} : expression ≠ gabarit")
        par = {b["cle"]: b for b in r["calcul"]}
        for v in r["verifications"]:
            if any(k not in par for k in v["blocs"]):
                f.append(f"{v['cle']} : bloc absent {v['blocs']}")
                continue
            if v["formules"] != [fx for k in v["blocs"] for fx in par[k]["formules"]]:
                f.append(f"{v['cle']} : formules ≠ blocs")
            dernier = v["formules"][-1]
            if dernier["cle"] != "taux" or not pres(dernier["resultat"], v["taux"]):
                f.append(f"{v['cle']} : la dernière formule n'est pas son taux ({dernier['symbole']} = {dernier['resultat']} ≠ {v['taux']})")
            # les termes d'une formule sont les résultats des formules d'avant, quand elles portent le même symbole
            connus = {}
            for fx in v["formules"]:
                for t in fx["termes"]:
                    if t["symbole"] in connus and not pres(t["valeur"], connus[t["symbole"]]) and t["symbole"] not in ("L", "w_lim", "n", "l_ef"):
                        f.append(f"{v['cle']} {fx['symbole']} : terme {t['symbole']} = {t['valeur']} ≠ {connus[t['symbole']]} calculé plus haut")
                connus[fx["symbole"]] = fx["resultat"]
    return f, n, lus


CAS_FORMULES = [R0, r2, r3, r4, r5, r6, surf2, sans_pp, sysm, ent, lin, pal, proche, gl_long, cf2, autre, sansq,
                M.verifier(poutre(standard=False, b=50.0, h=100.0, L=2.0, maintien__type="appuis")),
                M.verifier(poutre(classe="GL28c", section="100x240", L=6.0, maintien__type="entretoises", maintien__entraxe=2.0))] + res_t[:80]
ff, n_fx, n_lu = fautes_formules(CAS_FORMULES)
chk(f"{n_fx} formules réévaluées par l'interpréteur du test à partir de leur gabarit et de leurs termes ({n_lu} valeurs lues dans un tableau ou fixées par une "
    f"condition, avec leur origine) sur {len(CAS_FORMULES)} poutres : chaque résultat retrouvé à 1e-9 ; chaque vérification finit par son taux ; chaque terme "
    "est le résultat calculé plus haut", not ff and n_fx > 3000, ff[:4])
blocs0 = [b["cle"] for b in R0["calcul"]]
chk("les blocs du détail, dans l'ordre de la note : charges, deux combinaisons, section, déversement, flexion × 2, cisaillement × 2, appui, appui × 2, flèches",
    blocs0 == ["charges", "combinaison_G", "combinaison_GQ", "section", "deversement", "flexion_G", "flexion_GQ", "cisaillement_G", "cisaillement_GQ",
               "appui", "appui_G", "appui_GQ", "fleche_inst", "fleche_fin"], blocs0)
syms = {fx["symbole"] for r in CAS_FORMULES for b in r["calcul"] for fx in b["formules"]}
chk("les symboles du détail : charges (g_pp, g, q, p_d), efforts (M_d, V_d), section (A, W, I, k_h), déversement (l_ef, σ_m,crit, λ_rel,m, k_crit), "
    "résistances (f_m,d, f_v,d, f_c,90,d), contraintes, appui (l_1, l_ef,a, k_c,90), flèches (w_g,m, w_g,v, w_inst, w_fin, w_net,fin, w_lim, k_def, ψ_2)",
    syms >= {"G", "Q", "g_pp", "g", "q", "k_mod", "p_d", "M_d", "V_d", "A", "W", "I", "k_h", "l_ef", "σ_m,crit", "λ_rel,m", "k_crit", "f_m,d", "σ_m,d", "f_v,d",
             "τ_d", "l_1", "l_ef,a", "k_c,90", "f_c,90,d", "σ_c,90,d", "w_g,m", "w_g,v", "w_g", "w_q,m", "w_q,v", "w_q", "w_inst", "w_lim", "k_def", "ψ_2",
             "w_fin", "w_net,fin", "η"}, sorted(syms))
chk("k_sys n'apparaît dans f_m,d et f_v,d que quand l'effet système est demandé",
    not any("k_sys" in fx["expression"] for b in R0["calcul"] for fx in b["formules"])
    and sum("k_sys" in fx["expression"] for b in sysm["calcul"] for fx in b["formules"]) == 4)


def textes(r):
    return [a["message"] for a in r["avertissements"]] + r["hypotheses"] + r["non_couvert"] + [fx["expression"] for b in r["calcul"] for fx in b["formules"]] \
        + [v["enonce"] for v in r["verifications"]] + [r["geometrie"]["designation"]]


REFERENCES_NORME = re.compile(r"§[\d.]+(?:\(\d+\))?|\(\d\.\d+\)|tableau \d\.\d")     # « §6.1.7(3) », « (6.10) », « tableau 3.1 » : pas des nombres


def a_point(t):
    """Un nombre écrit « à point » dans un texte (hors des renvois aux articles et aux tableaux de la norme)."""
    return re.search(r"\d\.\d", REFERENCES_NORME.sub("", t)) is not None


points = [t for r in CAS_FORMULES + [M.verifier(poutre(standard=False, b=62.5, h=187.5))] for t in textes(r) if a_point(t)]
chk("aucun nombre « à point » dans les textes du moteur (avertissements, hypothèses, origines, énoncés, désignation « 62,5 × 187,5 ») : virgule décimale partout",
    not points and M.verifier(poutre(standard=False, b=62.5, h=187.5))["geometrie"]["designation"] == "62,5 × 187,5", points[:3])
chk("ce qui n'est pas couvert est rendu (charges concentrées, porte-à-faux, entailles, vibrations, flexion déviée, feu, perçages, bois humide)",
    len(R0["non_couvert"]) == 8 and all(any(m in t for t in R0["non_couvert"]) for m in ("concentrées", "porte-à-faux", "entailles", "vibrations", "déviée",
                                                                                         "feu", "perçages", "humide")), R0["non_couvert"])

# ====================================================================== 11. le sélecteur
print("\n=== 11. Sélecteur : les sections standard du type du bois, par aire, la retenue et la plus légère qui tient ===")
C0 = M.candidats_sections(P0)


def fautes_selecteur(mod, p, n_attendu):
    f = []
    cs = mod.candidats_sections(p)
    pn = mod.normaliser(p)
    if len(cs) != n_attendu:
        f.append(f"{len(cs)} candidats")
    if [c["A"] for c in cs] != sorted(c["A"] for c in cs):
        f.append("pas triés par aire")
    if any(not pres(c["A"], c["b"] * c["h"]) or not pres(c["masse"], TABLE[pn["classe"]]["rho_mean"] * c["b"] * c["h"] * 1e-6) for c in cs):
        f.append("aire ou masse")
    for c in cs[:: max(1, len(cs) // 9)]:
        r = mod.verifier(dict(copy.deepcopy(p), standard=True, section=c["nom"], b=None, h=None))
        if not pres(c["taux"], r["taux_max"]) or c["ok"] != (r["statut"] == "ok") or c["gouvernant"] != r["gouvernant"]["cle"]:
            f.append(f"{c['nom']} : taux {c['taux']} ≠ {r['taux_max']}")
    tiennent = [c for c in cs if c["ok"]]
    if [c["nom"] for c in cs if c["optimal"]] != [c["nom"] for c in tiennent[:1]]:
        f.append(f"optimal : {[c['nom'] for c in cs if c['optimal']]} ≠ {[c['nom'] for c in tiennent[:1]]}")
    if [c["nom"] for c in cs if c["retenue"]] != ([pn["section"]] if pn["standard"] else []):
        f.append(f"retenue : {[c['nom'] for c in cs if c['retenue']]}")
    return f


chk("poutre neuve : 26 sections du bois massif triées par aire, chacune au taux de son calcul complet, 75 × 225 retenue, optimale = la première qui tient",
    not fautes_selecteur(M, P0, 26), fautes_selecteur(M, P0, 26))
opt0 = M.optimal(P0)
chk("la note neuve propose plus léger : ★ sur 38 × 235 (95,2 %, 3,75 kg/m contre 7,09 kg/m pour 75 × 225)",
    opt0["nom"] == "38x235" and dec(100 * opt0["taux"], 95.2, 1) and dec(opt0["masse"], 3.75, 2) and opt0["A"] < R0["geometrie"]["A"] and opt0["designation"] == "38 × 235",
    opt0)
PGL = poutre(classe="GL24h", section="120x400", L=8.0, charges__G=3.0, charges__Q=4.0)
chk("lamellé-collé : les 67 sections du lamellé (aucune du massif), mêmes règles", not fautes_selecteur(M, PGL, 67)
    and all(c["nom"] in [s["nom"] for s in sl] for c in M.candidats_sections(PGL)), fautes_selecteur(M, PGL, 67))
chk("section sur mesure : aucune retenue ; le reste inchangé", not fautes_selecteur(M, poutre(standard=False, b=80.0, h=240.0), 26))
lourd = poutre(L=12.0, charges__G=20.0, charges__Q=30.0)
chk("aucune section ne tient (12 m, 20 + 30 kN/m) : pas d'optimal", M.optimal(lourd) is None and not any(c["ok"] for c in M.candidats_sections(lourd)))
chk("à aire égale (100 × 225 et 150 × 150 : 22 500 mm²), la plus étroite d'abord", [c["nom"] for c in C0 if c["A"] == 22500.0] == ["100x225", "150x150"])
chk("la section la plus proche d'une section sur mesure (retour à « standard ») : la même si elle existe, sinon l'aire la plus voisine (80 × 240 = 19 200 mm² → "
    "100 × 200 ; 75 × 225 en lamellé → 80 × 200, à aire égale avec 100 × 160 la hauteur la plus voisine)",
    E.section_proche("massif", 75.0, 225.0) == "75x225" and E.section_proche("massif", 80.0, 240.0) == "100x200" and E.section_proche("lamelle", 75.0, 225.0) == "80x200",
    (E.section_proche("massif", 80.0, 240.0), E.section_proche("lamelle", 75.0, 225.0)))

# ====================================================================== 12. les refus
print("\n=== 12. Refus : ValueError en français qui nomme le champ, l'entrée intacte ===")


def refuse(mod, p, *mots):
    """La poutre ``p`` est refusée par ``mod.normaliser`` ET ``mod.verifier``, avec un message qui porte ``mots``."""
    avant = copy.deepcopy(p)
    for fn in (mod.normaliser, mod.verifier):
        try:
            fn(p)
            return False
        except ValueError as e:
            if not all(m in str(e) for m in mots) or p != avant:
                return False
    return True


def refus_manques(mod):
    """Les refus attendus que ``mod`` ne fait pas (ou fait sans nommer le champ)."""
    d = mod.D if hasattr(mod, "D") else D
    f = []
    for chemin, (mini, maxi, unite, lib) in D.BORNES.items():
        for x in (mini - 0.01, maxi + 0.01):
            p = poutre(standard=False) if chemin in ("b", "h") else poutre(charges__nature="autre", charges__duree="moyen", charges__psi2=0.3)
            if "/" in chemin:
                g, c = chemin.split("/")
                p[g][c] = x
            else:
                p[chemin] = x
            if not refuse(mod, p, lib[3:] if lib.startswith(("la ", "le ")) else lib.replace("l'", ""), "doit rester entre"):
                f.append(f"{chemin} = {x} admis")
    cas = [
        (dict(P0, classe="C99"), ("classe de résistance",)), (dict(P0, classe="D30"), ("classe",)), (dict(P0, inconnu=1), ("inconnu",)),
        (dict(P0, section="80x160"), ("80 × 160", "bois massif")), (poutre(classe="GL24h", section="75x225"), ("75 × 225", "lamellé-collé")),
        (dict(P0, b=80.0), ("largeur b", "section standard")), (dict(P0, h=200), ("hauteur h", "section standard")),
        (dict(P0, L="4"), ("portée L", "nombre")), (dict(P0, L=float("nan")), ("portée L",)), (dict(P0, L=float("inf")), ("portée L",)), (dict(P0, L=True), ("portée L",)),
        (dict(P0, standard=1), ("section standard", "vrai ou faux")), (dict(P0, systeme="oui"), ("effet système",)),
        (dict(P0, service=1.0), ("classe de service",)), (dict(P0, service=4), ("classe de service",)), (dict(P0, service=True), ("classe de service",)),
        (dict(P0, lim_inst=275), ("flèche instantanée", "L/300")), (dict(P0, lim_fin=500), ("flèche finale",)), (dict(P0, lim_inst="L/300"), ("flèche instantanée",)),
        (dict(P0, nom=12), ("nom",)), (dict(P0, nom="x" * 61), ("nom", "61")),
        (dict(P0, charges=dict(P0["charges"], psi2=0.7)), ("ψ₂", "habitation", "autre")), (dict(P0, charges=dict(P0["charges"], duree="court")), ("durée", "autre")),
        (dict(P0, charges=dict(P0["charges"], nature="vent")), ("nature",)), (dict(P0, charges=dict(P0["charges"], mode="ponctuelle")), ("mode",)),
        (dict(P0, charges=dict(P0["charges"], poids_propre="oui")), ("poids propre",)), (dict(P0, charges=dict(P0["charges"], truc=1)), ("truc",)),
        (dict(P0, charges=[1, 2]), ("charges",)), (dict(P0, appui=dict(P0["appui"], verifier=None, a="cent")), ("longueur d'appui",)),
        (dict(P0, appui=dict(P0["appui"], verifier="non")), ("vérifier l'appui",)), (dict(P0, maintien=dict(P0["maintien"], type="libre")), ("maintien",)),
        (dict(P0, L=0.5, appui=dict(P0["appui"], a=500.0)), ("longueur d'appui", "portée")),
        (dict(P0, maintien={"type": "entretoises", "entraxe": 4.5}), ("entretoises", "portée")),
        (dict(P0, charges=dict(P0["charges"], nature="autre", duree="eternel")), ("durée",)),
    ]
    for p, mots in cas:
        if not refuse(mod, p, *mots):
            f.append(f"admis ou mal dit : {mots}")
    for mauvais in (None, [], "poutre", 3):
        try:
            mod.normaliser(mauvais)
            f.append(f"{mauvais!r} admis")
        except ValueError:
            pass
    return f


rm = refus_manques(M)
chk("chaque champ numérique refusé juste sous son minimum et juste au-dessus de son maximum (15 champs × 2), les types faux, les clés inconnues, les menus, "
    "les contradictions (b contre la section standard, ψ₂ et durée contre la nature, section d'un autre type, a ≥ L, entretoises > L) : ValueError qui nomme "
    "le champ, l'entrée intacte", not rm, rm[:4])
bornes_ok = []
for chemin, (mini, maxi, unite, lib) in D.BORNES.items():
    for x in (mini, maxi):
        p = poutre(standard=False, L=30.0 if chemin in ("appui/a", "maintien/entraxe") else 4.0) if chemin != "L" else poutre(standard=False)
        p["charges"].update(nature="autre", duree="moyen", psi2=0.3)
        if "/" in chemin:
            g, c = chemin.split("/")
            p[g][c] = x
        else:
            p[chemin] = x
        try:
            json.dumps(M.verifier(p), allow_nan=False)
        except ValueError as e:
            bornes_ok.append((chemin, x, str(e)))
chk("les bornes elles-mêmes sont admises (fermées) et le calcul y reste fini (JSON strict)", not bornes_ok, bornes_ok[:3])
try:
    M.normaliser(dict(P0, L=0.2))
    msg = ""
except ValueError as e:
    msg = str(e)
chk("le message d'un refus : « La portée L doit rester entre 0,5 et 30 m (reçu : 0,2) » — en français, à la virgule", msg == "La portée L doit rester entre 0,5 et 30 m (reçu : 0,2)", msg)
part = {"nom": "Seule", "classe": "GL24h", "L": 6}
pn = M.normaliser(part)
chk("une poutre partielle est complétée : la section par défaut de SON type (lamellé : 120 × 400), b et h déduits, les groupes aux défauts ; un entier devient un flottant",
    pn["section"] == "120x400" and (pn["b"], pn["h"]) == (120.0, 400.0) and pn["L"] == 6.0 and isinstance(pn["L"], float) and pn["charges"] == P0["charges"]
    and part == {"nom": "Seule", "classe": "GL24h", "L": 6}, pn)
chk("la nature fixe la durée et ψ₂ absents (stockage → long terme, 0,8) ; la section standard fixe b et h absents",
    M.normaliser(dict(P0, charges={"nature": "E"}))["charges"]["duree"] == "long" and M.normaliser(dict(P0, charges={"nature": "E"}))["charges"]["psi2"] == 0.8
    and M.normaliser({"section": "45x220"})["h"] == 220.0)
idem = [p["nom"] for p in TIRAGES[:120] if M.normaliser(M.normaliser(p)) != M.normaliser(p) or json.loads(json.dumps(M.normaliser(p))) != M.normaliser(p)]
chk("normaliser est idempotente et survit à un aller-retour JSON (120 tirages)", not idem, idem[:3])
strict = []
for p in TIRAGES[:120] + list(EXEMPLES.values()):
    r = M.verifier(p)
    try:
        if json.loads(json.dumps(r, allow_nan=False)) != r:
            strict.append(p["nom"])
    except ValueError:
        strict.append(p["nom"])
chk("la sortie de verifier est du JSON strict (ni NaN ni infini) et se relit à l'identique", not strict, strict[:3])
zero = M.normaliser(dict(P0, contre_fleche=-0.0, appui=dict(P0["appui"], debord=-0.0)))
chk("jamais de « -0 » dans la poutre relue", str(zero["contre_fleche"]) == "0.0" and str(zero["appui"]["debord"]) == "0.0")
intact = copy.deepcopy(TIRAGES[3])
M.verifier(TIRAGES[3])
M.candidats_sections(TIRAGES[3])
chk("verifier et candidats_sections ne modifient pas la poutre reçue", TIRAGES[3] == intact)
chk("lire une section telle qu'on l'écrit : « 75 × 225 », « 75x225 », « 75 X 225 » → « 75x225 » ; la désignation à l'écran : « 75 × 225 »",
    E.lire_tag("75 × 225") == E.lire_tag("75x225") == E.lire_tag("75 X 225") == "75x225" and E.designation("75x225") == "75 × 225")

# ====================================================================== 13. diagrammes, pureté
print("\n=== 13. Diagrammes : les maxima des courbes sont ceux du calcul ; pureté : bibliothèque standard seule ===")


def fautes_diagrammes(mod_dg, r):
    f = []
    cb = mod_dg.courbes(r, 49)
    n = len(cb["x"])
    if n != 49 or cb["x"][0] != 0.0 or not pres(cb["x"][-1], r["geometrie"]["L"]) or not pres(cb["x"][24], r["geometrie"]["L"] / 2):
        f.append("abscisses")
    ef = r["efforts"]
    if not pres(cb["V"][0], ef["V_d"]) or not pres(cb["V"][-1], -ef["V_d"]) or abs(cb["V"][24]) > 1e-9 or not pres(max(cb["V"]), ef["V_d"]):
        f.append(f"V : {cb['V'][0]} ≠ {ef['V_d']}")
    if not pres(cb["M"][24], ef["M_d"]) or not pres(max(cb["M"]), ef["M_d"]) or abs(cb["M"][0]) > 1e-9 or abs(cb["M"][-1]) > 1e-9:
        f.append(f"M : {cb['M'][24]} ≠ {ef['M_d']}")
    if any(not pres(cb["M"][i], cb["M"][n - 1 - i]) for i in range(n)):
        f.append("M non symétrique")
    # le moment est l'intégrale de l'effort tranchant EN CHAQUE POINT (trapèzes, exacts pour une droite) : une parabole
    pas, aire = r["geometrie"]["L"] / (n - 1), 0.0
    for i in range(n - 1):
        aire += (cb["V"][i] + cb["V"][i + 1]) / 2.0 * pas
        if abs(aire - cb["M"][i + 1]) > 1e-9 * max(1.0, ef["M_d"]):
            f.append(f"∫V ≠ M en x = {cb['x'][i + 1]} : {aire} ≠ {cb['M'][i + 1]}")
            break
    if not pres(cb["M"][12], 0.75 * ef["M_d"]):
        f.append("M(L/4) ≠ 3/4·M_d")
    # la déformée au quart de la portée, à la main : flexion 57·p·L⁴/(6144·E·I), effort tranchant 1,2·3·p·L²/(32·G·A)
    Lmm, pp = r["geometrie"]["L_mm"], r["charges"]["g"] + r["charges"]["q"]
    w_quart = 57.0 * pp * Lmm ** 4 / (6144.0 * r["materiau"]["E0_mean"] * r["geometrie"]["I"]) \
        + 1.2 * 3.0 * pp * Lmm ** 2 / (32.0 * r["materiau"]["G_mean"] * r["geometrie"]["A"])
    if not pres(cb["w_inst"][12], w_quart):
        f.append(f"w_inst(L/4) = {cb['w_inst'][12]} ≠ {w_quart}")
    fl_ = r["fleches"]
    for k, attendu in (("w_inst", fl_["w_inst"]), ("w_fin", fl_["w_fin"]), ("w_net_fin", fl_["w_net_fin"])):
        if not pres(cb[k][24], attendu):
            f.append(f"{k}(L/2) = {cb[k][24]} ≠ {attendu}")
        if abs(cb[k][0]) > 1e-9 or abs(cb[k][-1]) > 1e-9:
            f.append(f"{k} non nul aux appuis")
    if not pres(max(cb["w_inst"]), fl_["w_inst"]) or not pres(max(cb["w_fin"]), fl_["w_fin"]):
        f.append("le maximum de la déformée n'est pas à mi-portée")
    if cb["combinaison"] != r["combinaison_elu"]:
        f.append("combinaison")
    return f


fd = [(k, f) for k, r in (("ouverture", R0), ("G seule", r2), ("lamellé", r4), ("contre-flèche", cf), ("service 3", r6)) for f in fautes_diagrammes(DG, r)]
chk("V(0) = V_d, V(L/2) = 0, M(x) = ∫V en chaque point (parabole : M(L/4) = 3/4·M_d), M(L/2) = M_d, w_inst(L/4) à la main, w_inst(L/2), w_fin(L/2), "
    "w_net,fin(L/2) : ceux du calcul ; la combinaison est celle qui gouverne l'ELU",
    not fd, fd[:3])
try:
    DG.courbes(R0, 10)
    pair = False
except ValueError:
    pair = True
chk("un nombre de points pair est refusé (la mi-portée doit être un point)", pair)

PERMIS = {"math", "re", "copy", "typing"}
imports = {}
for n in ("donnees", "entree", "moteur", "diagrammes", "__init__"):
    arbre = ast.parse(lire("bois", "poutre", n + ".py"))
    for nd in ast.walk(arbre):
        if isinstance(nd, ast.Import):
            imports.setdefault(n, set()).update(a.name.split(".")[0] for a in nd.names)
        elif isinstance(nd, ast.ImportFrom) and nd.level == 0:
            imports.setdefault(n, set()).add((nd.module or "").split(".")[0])
chk("les sources de bois/poutre n'importent que math, re, copy et typing (et le paquet lui-même)", all(v <= PERMIS for v in imports.values()), imports)
chk("aucune trace de l'ancien moteur : ni « Streamlit » comme référence de parité, ni ECARTS, ni grandeur « indicatif »",
    not hasattr(M, "ECARTS") and "indicatif" not in R0 and "modules/poutre_bois.py" not in D.REFERENCE)
SOUS_PROCESSUS = r"""
import importlib.abc, json, sys
racine = sys.argv[1]
sys.path.insert(0, racine)
REFUSES = ("streamlit", "numpy", "pandas", "scipy", "matplotlib", "reportlab", "PIL", "fitz", "acier", "beton", "api_web", "modules")
class Refus(importlib.abc.MetaPathFinder):
    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in REFUSES:
            raise ImportError("refusé par le test : " + nom)
        return None
sys.meta_path.insert(0, Refus())
from bois.poutre import moteur as M, diagrammes as DG
lot = json.loads(sys.stdin.read())
res = [M.verifier(p) for p in lot]
cands = M.candidats_sections(lot[0])
cb = DG.courbes(res[0], 9)
charges = sorted(m.split(".")[0] for m in sys.modules if m.split(".")[0] in REFUSES)
print(json.dumps({"res": res, "cands": cands, "cb": cb, "charges": charges}))
"""
LOT = [P0] + list(EXEMPLES.values())[1:] + TIRAGES[:40]
attendu = json.loads(json.dumps({"res": [M.verifier(p) for p in LOT], "cands": M.candidats_sections(LOT[0]), "cb": DG.courbes(R0, 9)}))
interpretes = [(v, exe) for v, exe in (("3.12", "/usr/bin/python3.12"), ("3.13", shutil.which("python3.13") or "/usr/bin/python3.13")) if os.path.exists(exe)]
for v, exe in interpretes:
    pr = subprocess.run([exe, "-I", "-c", SOUS_PROCESSUS, RACINE], input=json.dumps(LOT), capture_output=True, text=True, check=False)
    if pr.returncode != 0:
        chk(f"Python {v} : le moteur se charge et calcule sans numpy ni streamlit", False, pr.stderr[-600:])
        continue
    out = json.loads(pr.stdout)
    chk(f"Python {v} : numpy, streamlit, pandas, scipy… et le reste du dépôt REFUSÉS par sys.meta_path, le moteur calcule quand même ({len(out['res'])} poutres, "
        "le sélecteur, les courbes), aucun module refusé chargé", not out["charges"], out["charges"])
    diff = [i for i, (a, b) in enumerate(zip(out["res"], attendu["res"])) if a != b] + (["cands"] if out["cands"] != attendu["cands"] else []) \
        + (["cb"] if out["cb"] != attendu["cb"] else [])
    chk(f"Python {v} : résultats identiques à ceux de ce processus", not diff, diff[:3])
if len(interpretes) < 2:
    print("     (python3.13 absent : le calcul sous la version de Pyodide n'est pas rejoué sur cette machine)")

# ====================================================================== 14. mutations
print("\n=== 14. Mutations : le moteur modifié d'UNE formule ou d'UNE valeur est rejeté ===")
PAQUET = sys.modules["bois.poutre"]
NOMS_MODULES = ("donnees", "entree", "moteur", "diagrammes")


def charger_mutant(mutations):
    """Les quatre modules rechargés depuis leurs SOURCES après des remplacements [(module, avant, après)] ; chaque
    « avant » doit figurer UNE fois dans la source (sinon la mutation est périmée). Rend {nom: module}."""
    sauve = {n: sys.modules[f"bois.poutre.{n}"] for n in NOMS_MODULES}
    sauve_attr = {n: getattr(PAQUET, n) for n in NOMS_MODULES}
    mods = {}
    try:
        for n in NOMS_MODULES:
            chemin = os.path.join(RACINE, "bois", "poutre", n + ".py")
            src = lire("bois", "poutre", n + ".py")
            for module, avant, apres in mutations:
                if module == n:
                    if src.count(avant) != 1:
                        raise AssertionError(f"mutation périmée : « {avant} » ({src.count(avant)} fois dans {n}.py)")
                    src = src.replace(avant, apres)
            m = types.ModuleType(f"bois.poutre.{n}")
            m.__package__, m.__file__ = "bois.poutre", chemin
            sys.modules[m.__name__] = m
            setattr(PAQUET, n, m)
            exec(compile(src, chemin, "exec"), m.__dict__)  # noqa: S102 — le moteur lui-même, muté
            mods[n] = m
    finally:
        sys.modules.update({f"bois.poutre.{n}": sauve[n] for n in NOMS_MODULES})
        for n in NOMS_MODULES:
            setattr(PAQUET, n, sauve_attr[n])
    return mods


def rejets(mods):
    """Ce que les contrôles de ce fichier reprochent au moteur ``mods`` : les tables, les exemples face à la référence,
    des tirages, les formules réévaluées, les unités, le sélecteur, les refus, les diagrammes, les défauts."""
    mod, dg, d = mods["moteur"], mods["diagrammes"], mods["donnees"]
    out = []
    try:
        out += [f"table : {x}" for x in fautes_tables(d)[:2]]
        if mod.nouvelle_poutre() != P0:
            out.append("défauts de la poutre neuve")
        for nom, p in EXEMPLES.items():
            ec = ecarts(mod, p)
            if ec:
                out.append(f"exemple {nom[:1]} : {ec[0]}")
        for p in TIRAGES[:120]:
            ec = ecarts(mod, p)
            if ec:
                out.append(f"tirage {p['nom']} : {ec[0]}")
                break
        r0 = mod.verifier(P0)
        if abs(fleche_si(r0["charges"]["g"], 4.0, 11000.0, I1, 690.0, A1) + fleche_si(0.8, 4.0, 11000.0, I1, 690.0, A1) - r0["fleches"]["w_inst"]) > 1e-5:
            out.append("unités : la flèche n'est pas celle de l'intégration en SI")
        ff_, _, _ = fautes_formules([mod.verifier(p) for p in list(EXEMPLES.values()) + TIRAGES[:25]]
                                    + [mod.verifier(poutre(charges__mode="surfacique", systeme=True, maintien__type="entretoises", section="45x245", L=5.0)),
                                       mod.verifier(poutre(standard=False, b=50.0, h=100.0, L=2.0, maintien__type="appuis"))])
        out += [f"formules : {x}" for x in ff_[:2]]
        out += [f"sélecteur : {x}" for x in fautes_selecteur(mod, P0, 26)[:1] + fautes_selecteur(mod, PGL, 67)[:1]]
        o = mod.optimal(P0)
        if not o or o["nom"] != "38x235":
            out.append("optimal de la note neuve")
        out += [f"refus : {x}" for x in refus_manques(mod)[:2]]
        out += [f"diagrammes : {x}" for x in fautes_diagrammes(dg, r0)[:1] + fautes_diagrammes(dg, mod.verifier(poutre(contre_fleche=5.0, service=3)))[:1]]
        if mod._etat(1.0) != "ok" or mod._etat(1.0 + 1e-9) != "nok":
            out.append("seuil de conformité")
        g = mod.verifier(EXEMPLES["5 · l'appui gouverne (a = 40 mm)"])["gouvernant"]
        if g["cle"] != "appui":
            out.append("gouvernant")
        if not any(a["code"] == "kc90_distance" for a in mod.verifier(poutre(section="100x300", L=0.65))["avertissements"]):
            out.append("avertissement k_c,90")
        if any(a_point(t) for t in textes(mod.verifier(poutre(charges__poids_propre=False, standard=False, b=62.5, h=187.5)))):
            out.append("texte à point décimal")
    except AssertionError:
        raise
    except Exception as err:  # noqa: BLE001 — un mutant qui casse est rejeté
        out.append(f"{type(err).__name__} : {err}")
    return out


MUTATIONS = (
    ("k_cr retiré du cisaillement (défaut n° 3 de l'ancien écran)", [("moteur", "/ (D.K_CR * b * h)", "/ (b * h)")]),
    ("G + Q calculée avec le k_mod permanent", [("moteur", '"nom": "G + Q", "duree": duree,', '"nom": "G + Q", "duree": "permanent",')]),
    ("G seule calculée avec le k_mod de la charge variable", [("moteur", '"nom": "G seule", "duree": "permanent",', '"nom": "G seule", "duree": duree,')]),
    ("la combinaison G seule jamais retenue (défaut n° 2 de l'ancien écran)", [("moteur", 'c = max(x["combs"], key=lambda c: c[k])', 'c = x["combs"][-1]')]),
    ("poids propre oublié", [("moteur", 'g = G + g_pp if ch["poids_propre"] else G', "g = G")]),
    ("poids propre calculé avec ρ_k", [("moteur", 'g_pp = mat["rho_mean"] * b * h', 'g_pp = mat["rho_k"] * b * h')]),
    ("5/384 changé en 5/348", [("donnees", "FLECHE_NUM, FLECHE_DEN = 5.0, 384.0", "FLECHE_NUM, FLECHE_DEN = 5.0, 348.0")]),
    ("flèche × 1 000 (défaut n° 1 de l'ancien écran : q en kN/m multiplié par 1 000)",
     [("moteur", "return D.FLECHE_NUM * p * L_mm ** 4 /", "return D.FLECHE_NUM * (p * 1000.0) * L_mm ** 4 /")]),
    ("flèche d'effort tranchant oubliée", [("moteur", "w_g, w_q = w_g_m + w_g_v, w_q_m + w_q_v", "w_g, w_q = w_g_m, w_q_m")]),
    ("ψ₂ oublié dans la flèche finale", [("moteur", "w_q * (1.0 + psi2 * kdef)", "w_q * (1.0 + kdef)")]),
    ("flèche finale de l'ancien écran : (1 + k_def)·w(G + ψ₂·Q)",
     [("moteur", "w_fin = w_g * (1.0 + kdef) + w_q * (1.0 + psi2 * kdef)", "w_fin = (1.0 + kdef) * (w_g + psi2 * w_q)")]),
    ("la contre-flèche oubliée", [("moteur", "w_net_fin = w_fin - w_c", "w_net_fin = w_fin")]),
    ("la limite finale prise pour l'instantanée", [("moteur", 'lim_fin = L_mm / p["lim_fin"]', 'lim_fin = L_mm / p["lim_inst"]')]),
    ("k_h appliqué au-dessus de la hauteur de référence (défaut de l'ancien écran)", [("moteur", '    if h >= r["h_ref"]:\n        return 1.0\n', "")]),
    ("k_h du lamellé-collé pris à 150 mm", [("donnees", '"h_ref": 600.0', '"h_ref": 150.0')]),
    ("k_h sans plafond", [("moteur", 'return min((r["h_ref"] / h) ** r["exposant"], r["plafond"])', 'return (r["h_ref"] / h) ** r["exposant"]')]),
    ("γ_M du lamellé-collé = 1,30", [("donnees", "GAMMA_M = {MASSIF: 1.30, LAMELLE: 1.25}", "GAMMA_M = {MASSIF: 1.30, LAMELLE: 1.30}")]),
    ("déversement avec E_0,mean au lieu de E_0,05", [("moteur", 'sigma_crit = D.COEF_SIGMA_CRIT * b ** 2 * mat["E0_05"]', 'sigma_crit = D.COEF_SIGMA_CRIT * b ** 2 * mat["E0_mean"]')]),
    ("l_ef sans les 2·h de la charge sur la fibre comprimée", [("moteur", "return D.LEF_CHARGE_REPARTIE * L_mm + D.LEF_FIBRE_COMPRIMEE * h", "return D.LEF_CHARGE_REPARTIE * L_mm")]),
    ("k_crit : la branche linéaire remplacée par 1", [("moteur", "return D.K_CRIT_ORIGINE - D.K_CRIT_PENTE * lambda_rel", "return 1.0")]),
    ("k_crit oublié dans la limite de flexion", [("moteur", 'c["lim_m"] = kcrit * c["fm_d"]', 'c["lim_m"] = c["fm_d"]')]),
    ("M_d = p·L²/10", [("moteur", 'c["M_d"] = c["p_d"] * L ** 2 / 8.0', 'c["M_d"] = c["p_d"] * L ** 2 / 10.0')]),
    ("τ = V/A (sans 1,5)", [("donnees", "COEF_TAU_RECTANGLE = 1.5", "COEF_TAU_RECTANGLE = 1.0")]),
    ("compression d'appui sous G + Q non pondérées (défaut n° 5 de l'ancien écran)",
     [("moteur", 'c["sigma_c90"] = c["V_d"] * 1e3 / (b * l_ef_a)', 'c["sigma_c90"] = (g + q) * L / 2.0 * 1e3 / (b * l_ef_a)')]),
    ("appui sans longueur efficace (a seul)", [("moteur", "l_ef_a = a + cote_travee + cote_about", "l_ef_a = a")]),
    ("la majoration côté about ignore le débord", [("moteur", "cote_about = min(D.MAJORATION_APPUI, a, debord)", "cote_about = min(D.MAJORATION_APPUI, a)")]),
    ("k_c,90 gardé à 1,5 quand les appuis sont rapprochés", [("moteur", "    if l1 < D.L1_SUR_H_MIN * h:\n        raison = \"distance\"\n    elif", "    if")]),
    ("k_c,90 du massif = 1,75", [("donnees", "K_C90 = {MASSIF: 1.5, LAMELLE: 1.75}", "K_C90 = {MASSIF: 1.75, LAMELLE: 1.75}")]),
    ("k_sys appliqué à la compression d'appui", [("moteur", 'c["fc90_d"] = c["kmod"] * mat["fc90_k"] / gamma_M', 'c["fc90_d"] = c["kmod"] * ksys * mat["fc90_k"] / gamma_M')]),
    ("k_sys oublié sur f_v,d", [("moteur", 'c["fv_d"] = c["kmod"] * ksys * mat["fv_k"] / gamma_M', 'c["fv_d"] = c["kmod"] * mat["fv_k"] / gamma_M')]),
    ("charge surfacique sans l'entraxe", [("moteur", 'G = ch["g"] * ch["entraxe"] if surfacique else ch["G"]', 'G = ch["g"] if surfacique else ch["G"]')]),
    ("EN 338 : f_v,k de C24 = 2,5 (édition 2003)", [("donnees", '"fv_k":     (3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),', '"fv_k":     (3.0, 3.2, 3.4, 3.6, 3.8, 2.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),')]),
    ("EN 338 : f_c,90,k de C30 = 3,0 (base de l'ancien écran)", [("donnees", '"fc90_k":   (2.0, 2.2, 2.2, 2.3, 2.4, 2.5, 2.5, 2.7, 2.7, 2.8, 2.9, 3.0),', '"fc90_k":   (2.0, 2.2, 2.2, 2.3, 2.4, 2.5, 2.5, 3.0, 2.7, 2.8, 2.9, 3.0),')]),
    ("EN 14080 : E_0,05 de GL24h = 9 400", [("donnees", '"E0_05":    (9600, 10500, 11800, 9100, 10400, 11200),', '"E0_05":    (9400, 10500, 11800, 9100, 10400, 11200),')]),
    ("k_def de la classe de service 2 = 0,6", [("donnees", "KDEF = {1: 0.60, 2: 0.80, 3: 2.00}", "KDEF = {1: 0.60, 2: 0.60, 3: 2.00}")]),
    ("k_mod moyen terme de la classe 3 = 0,70", [("donnees", '3: {"permanent": 0.50, "long": 0.55, "moyen": 0.65,', '3: {"permanent": 0.50, "long": 0.55, "moyen": 0.70,')]),
    ("ψ₂ de l'habitation = 0,6", [("donnees", '("A", "A · habitation", "moyen", 0.3),', '("A", "A · habitation", "moyen", 0.6),')]),
    ("la neige en moyen terme", [("donnees", '("neige", "Neige (Belgique, ≤ 1 000 m)", "court", 0.0),', '("neige", "Neige (Belgique, ≤ 1 000 m)", "moyen", 0.0),')]),
    ("conforme jusqu'à 105 %", [("moteur", "return NOK if taux > 1.0 else OK", "return NOK if taux > 1.05 else OK")]),
    ("gouvernant : le plus petit taux", [("moteur", 'gov = max(verifs, key=lambda v: v["taux"])', 'gov = min(verifs, key=lambda v: v["taux"])')]),
    ("sélecteur trié par largeur", [("moteur", 'out.sort(key=lambda c: (c["A"], c["b"]))', 'out.sort(key=lambda c: (c["b"], c["A"]))')]),
    ("optimal : la plus LOURDE qui tient", [("moteur", "premier = next((c for c in out if c[\"ok\"]), None)", "premier = next((c for c in reversed(out) if c[\"ok\"]), None)")]),
    ("un champ hors bornes admis", [("entree", "    if v < mini or v > maxi:", "    if False and (v < mini or v > maxi):")]),
    ("ψ₂ contradictoire avec la nature corrigé en silence", [("entree", "if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or float(x) != psi2):", "if False:")]),
    ("longueur d'appui ≥ portée admise", [("entree", '    if out["appui"]["a"] >= L_mm:', '    if False and out["appui"]["a"] >= L_mm:')]),
    ("une clé inconnue admise", [("entree", "    inconnues = sorted(str(k) for k in poutre if k not in CLES)\n    if inconnues:", "    inconnues = sorted(str(k) for k in poutre if k not in CLES)\n    if False:")]),
    ("défaut de la poutre neuve : appui non vérifié (l'ancien écran)", [("donnees", '"appui": {"verifier": True, "a": 100.0, "debord": 0.0},', '"appui": {"verifier": False, "a": 100.0, "debord": 0.0},')]),
    ("diagramme : M(x) triangulaire", [("diagrammes", 'out["M"].append(ef["p_d"] * x * (L - x) / 2.0)', 'out["M"].append(ef["p_d"] * min(x, L - x) * L / 4.0)')]),
    ("diagramme : la déformée sans l'effort tranchant", [("diagrammes", "    return flexion + tranchant", "    return flexion")]),
    ("diagramme : la déformée nette sans la contre-flèche", [("diagrammes", 'out["w_net_fin"].append(w_fin - fl["w_c"] * 4.0 * xi * (1.0 - xi))', 'out["w_net_fin"].append(w_fin)')]),
    ("le texte d'une formule faux (M_d écrit p·L²/10, calculé juste)", [("moteur", '_fx("M_d", "M_d", "{p_d} × {L}² / 8"', '_fx("M_d", "M_d", "{p_d} × {L}² / 10"')]),
    ("un terme faux dans une formule (W à la place de I dans la flèche)", [("moteur", '_t("I", x["I"], "mm⁴")], w_m, "mm"),', '_t("I", x["W"], "mm⁴")], w_m, "mm"),')]),
    ("un nombre à point décimal dans un avertissement", [("moteur", 'f"Poids propre non compté ({_n(x[\'g_pp\'], 3)} kN/m)', 'f"Poids propre non compté ({x[\'g_pp\']:.3f} kN/m)')]),
)
TEMOIN = ("témoin : un commentaire changé", [("moteur", "# les charges (kN/m = N/mm)", "# les charges (kN/m = N/mm), témoin")])
survivants, n_rejets = [], []
for nom, muts in MUTATIONS:
    try:
        rej = rejets(charger_mutant(muts))
    except AssertionError as err:
        survivants.append(f"{nom} : {err}")
        continue
    n_rejets.append(len(rej))
    if not rej:
        survivants.append(nom)
chk(f"{len(MUTATIONS)} mutations volontaires (k_cr, k_mod de la mauvaise durée × 2, G seule oubliée, poids propre × 2, 5/384, flèche × 1 000, effort tranchant, "
    "ψ₂, flèche finale de l'ancien écran, contre-flèche, limite, k_h × 3, γ_M, déversement × 4, M_d, τ, appui × 4, k_c,90 × 2, k_sys × 2, surfacique, tables × 6, "
    "natures × 2, seuil, gouvernant, sélecteur × 2, refus × 4, défaut, diagrammes × 3, textes de formule × 2, point décimal) : TOUTES rejetées "
    f"(de {min(n_rejets or [0])} à {max(n_rejets or [0])} reproches chacune)", not survivants, survivants)
chk("la mutation témoin (un commentaire) n'est rejetée par aucun contrôle : les rejets viennent bien des mutations", not rejets(charger_mutant(TEMOIN[1])),
    rejets(charger_mutant(TEMOIN[1]))[:3])
chk("après les mutations, le moteur chargé est intact (mêmes résultats)", M.verifier(P0) == R0 and M.candidats_sections(P0) == C0)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)   ({time.time() - T0:.1f} s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:400])
sys.exit(1 if KO else 0)

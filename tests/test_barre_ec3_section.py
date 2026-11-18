# -*- coding: utf-8 -*-
"""Barres en acier — EUROCODE 3, la SECTION (acier/barre/ec3_classe.py, ec3_section.py) : la classe (EN 1993-1-1
tableau 5.2), la section efficace d'une classe 4 (EN 1993-1-5 § 4.4), les résistances et leurs interactions (§ 6.2).
La spécification : docs/acier/BARRES_ACIER.md § 5.1 à 5.4.

Il n'y a pas d'écran dont reproduire les chiffres. Les références sont :
  1. LA NORME : les limites du tableau 5.2 aux points où le texte les donne en clair (72ε, 83ε, 124ε, 33ε, 38ε, 42ε),
     les k_σ des tableaux 4.1 et 4.2, les seuils de ρ ; pour ψ = 1, l'accord avec le moteur du flambement en service.
  2. LES CLASSES de profilés connus (IPE 300, HEA 260, HEA 300, IPE 600…), l'âme fléchie et comprimée calculée à la
     main, et — sur tout le catalogue — la classe et l'aire efficace en compression du module Flambement.
  3. LA SECTION EFFICACE recalculée AUTREMENT : par assemblage des parties efficaces (le moteur retire des trous à la
     section brute), sur des sections sans congé dont les propriétés brutes sont exactes.
  4. LES RÉSISTANCES et les interactions calculées À LA MAIN sur les valeurs brutes du catalogue (formules réécrites ici).
  5. LES FORMULES : chaque bloc du moteur se recalcule par son gabarit à 10⁻⁹ ; aucun gabarit mort.
  6. LE CATALOGUE : chaque I, H et tube, quatre nuances, six jeux d'efforts — JSON strict, invariants.
  7. MUTATIONS : chaque altération plausible du moteur — écrite sur son TEXTE — fait rougir un contrôle des § 1 à 4.
  8. PURETÉ : le moteur se charge et calcule sans streamlit, pandas, matplotlib, scipy, reportlab.

    /usr/bin/python3.12 tests/test_barre_ec3_section.py
"""
import copy
import importlib.util
import json
import math
import os
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.barre import ec3_classe as CL  # noqa: E402
from acier.barre import ec3_section as ES  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.flambement import moteur as FM  # noqa: E402
from acier.flambement import normes as NO  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []
CHEMINS = {"classe": os.path.join(RACINE, "acier", "barre", "ec3_classe.py"),
           "section": os.path.join(RACINE, "acier", "barre", "ec3_section.py")}
R3 = math.sqrt(3.0)
NUANCES = ("S235", "S275", "S355", "S460")


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def brut(nom):
    """L'enregistrement du catalogue (cm, cm², cm³, cm⁴ ; dimensions en mm) : la matière des calculs à la main."""
    return CAT.par_nom()[nom]


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


# ------------------------------------------------------------------ 3. la section efficace, recalculée autrement
def assemblage(parties):
    """Aire, centre de gravité et inertie d'un assemblage de rectangles [(largeur, z1, z2)] autour de l'axe horizontal."""
    A = sum(w * (z2 - z1) for w, z1, z2 in parties)
    zc = sum(w * (z2 - z1) * (z1 + z2) / 2.0 for w, z1, z2 in parties) / A
    inertie = sum(w * (z2 - z1) ** 3 / 12.0 + w * (z2 - z1) * ((z1 + z2) / 2.0 - zc) ** 2 for w, z1, z2 in parties)
    return A, zc, inertie


def ks_interne(psi):
    return 4.0 if psi >= 1 else 8.2 / (1.05 + psi) if psi > 0 else 7.81 if psi == 0 else 7.81 - 6.29 * psi + 9.78 * psi ** 2 if psi > -1 \
        else 23.9 if psi == -1 else 5.98 * (1 - psi) ** 2


def rho_i(c_t, eps, psi):
    lp = c_t / (28.4 * eps * math.sqrt(ks_interne(psi)))
    return 1.0 if lp <= 0.5 + math.sqrt(0.085 - 0.055 * psi) else (lp - 0.055 * (3 + psi)) / lp ** 2


def rho_c(c_t, eps, ks=0.43):
    lp = c_t / (28.4 * eps * math.sqrt(ks))
    return 1.0 if lp <= 0.748 else (lp - 0.188) / lp ** 2


def ame_efficace(z_bas, z_haut, zc, t, eps):
    """Les parties efficaces d'une âme (partie droite de z_bas à z_haut, comprimée en haut) pour l'axe neutre zc."""
    c = z_haut - z_bas
    psi = -(zc - z_bas) / (z_haut - zc)
    rho = rho_i(c / t, eps, psi)
    bc = z_haut - zc
    beff = rho * bc
    return [(z_bas, zc + 0.6 * beff), (z_haut - 0.4 * beff, z_haut)], psi, rho


def i_sans_conge(h, b, tf, tw):
    """Un I soudé sans congé : propriétés brutes exactes (mm)."""
    A, zc, Iy = assemblage([(b, 0, tf), (tw, tf, h - tf), (b, h - tf, h)])
    Iz = 2 * tf * b ** 3 / 12.0 + (h - 2 * tf) * tw ** 3 / 12.0
    return SE.saisie("I", "soude_oxycoupe", nom=f"I {h}x{b}x{tf}x{tw}", h=h, b=b, tf=tf, tw=tw, r=0.0, A=A, Iy=Iy, Iz=Iz,
                     iy=math.sqrt(Iy / A), iz=math.sqrt(Iz / A), Wel_y=Iy / (h / 2.0), Wel_z=Iz / (b / 2.0),
                     Wpl_y=b * tf * (h - tf) + tw * (h - 2 * tf) ** 2 / 4.0, Wpl_z=tf * b ** 2 / 2.0 + (h - 2 * tf) * tw ** 2 / 4.0)


def weff_y_I(h, b, tf, tw, fy):
    """W_eff,y d'un I sans congé par ASSEMBLAGE des parties efficaces : semelle comprimée (console, ψ = 1), puis âme avec
    le ψ de la section à semelle réduite ; rapporté à la fibre comprimée de la section brute."""
    eps = math.sqrt(235.0 / fy)
    cf = (b - tw) / 2.0
    bf = tw + 2.0 * rho_c(cf / tf, eps) * cf                    # largeur efficace de la semelle comprimée
    _, zc1, _ = assemblage([(b, 0, tf), (tw, tf, h - tf), (bf, h - tf, h)])
    morceaux, psi, rho = ame_efficace(tf, h - tf, zc1, tw, eps)
    _, zc, inertie = assemblage([(b, 0, tf), (bf, h - tf, h)] + [(tw, z1, z2) for z1, z2 in morceaux])
    return inertie / (h - zc), psi, rho


def weff_z_I(h, b, tf, tw, fy):
    """W_eff,z d'un I sans congé : le bout des deux consoles comprimées retiré (bord libre le plus comprimé, k_σ du
    tableau 4.2 avec le ψ de la section brute) ; rapporté au bord de la section BRUTE (le choix du moteur, du côté sûr)."""
    eps = math.sqrt(235.0 / fy)
    cf = (b - tw) / 2.0
    psi = (tw / 2.0) / (b / 2.0)
    rho = rho_c(cf / tf, eps, 0.57 - 0.21 * psi + 0.07 * psi ** 2)
    # l'axe z est vertical : on assemble le long de y (de 0 à b), la zone comprimée en haut
    _, yc, inertie = assemblage([(2 * tf, 0, b - (1 - rho) * cf), (h - 2 * tf, (b - tw) / 2.0, (b + tw) / 2.0)])
    return inertie / (b - yc), rho


def tube_sans_arrondi(h, b, t):
    """Un caisson à angles vifs : propriétés brutes exactes (mm)."""
    A, _, Iy = assemblage([(b, 0, t), (2 * t, t, h - t), (b, h - t, h)])
    _, _, Iz = assemblage([(h, 0, t), (2 * t, t, b - t), (h, b - t, b)])
    return SE.saisie("tube_rect", "caisson", nom=f"caisson {h}x{b}x{t}", h=h, b=b, t=t, A=A, Iy=Iy, Iz=Iz, iy=math.sqrt(Iy / A),
                     iz=math.sqrt(Iz / A), Wel_y=Iy / (h / 2.0), Wel_z=Iz / (b / 2.0),
                     Wpl_y=b * t * (h - t) + 2 * t * (h - 2 * t) ** 2 / 4.0, Wpl_z=h * t * (b - t) + 2 * t * (b - 2 * t) ** 2 / 4.0)


def weff_y_tube(h, b, t, fy):
    """W_eff,y d'un caisson à angles vifs : paroi comprimée (interne, ψ = 1, largeur b − 3t, trou au milieu), puis les
    deux âmes (partie droite h − 3t centrée)."""
    eps = math.sqrt(235.0 / fy)
    cb, ch = b - 3 * t, h - 3 * t
    trou = (1 - rho_i(cb / t, eps, 1.0)) * cb
    haut = [((b - trou) / 2.0, h - t, h)] * 2
    _, zc1, _ = assemblage([(b, 0, t), (2 * t, t, h - t)] + haut)
    morceaux, psi, rho = ame_efficace(1.5 * t, h - 1.5 * t, zc1, t, eps)
    ames = [(2 * t, t, 1.5 * t), (2 * t, h - 1.5 * t, h - t)] + [(2 * t, z1, z2) for z1, z2 in morceaux]
    _, zc, inertie = assemblage([(b, 0, t)] + ames + haut)
    return inertie / (h - zc), psi, rho


# ------------------------------------------------------------------ les contrôles de référence (rejoués par les mutations)
def controles(C_=CL, S_=ES):
    """[(nom, ok, info)] — ``C_`` et ``S_`` : les modules à contrôler (les mutations y passent leurs copies altérées)."""
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    def v(nom, nuance, **eff):
        return S_.verifier({"section": SE.du_catalogue(nom), "nuance": nuance, **eff})

    def taux(r, id_):
        x = verif(r, id_)
        return None if x is None else x["taux"]

    def limite(r, id_):
        x = verif(r, id_)
        return None if x is None else x["limite"]

    # ---- 1. la norme
    e355 = math.sqrt(235.0 / 355.0)
    lim, _ = C_.limites_flexion_composee(0.5, -1.0, 1.0)
    c("tableau 5.2 : âme fléchie sans effort normal (α = 0,5 ; ψ = −1) → 72ε, 83ε, 124ε", [round(x, 9) for x in lim] == [72.0, 83.0, 124.0], lim)
    lim, _ = C_.limites_flexion_composee(1.0, 1.0, 1.0)
    c("tableau 5.2 : âme entièrement comprimée (α = 1 ; ψ = 1) → 33ε, 38ε, 42ε", [round(x, 9) for x in lim] == [33.0, 38.0, 42.0], lim)
    lim, _ = C_.limites_flexion_composee(0.75, 0.0, e355)
    c("tableau 5.2 : α = 0,75 et ψ = 0 en S355 → 396ε/8,75 = 36,82 ; 456ε/8,75 = 42,40 ; 42ε/0,67 = 51,00",
      proche(lim[0], 396 * e355 / 8.75) and proche(lim[1], 456 * e355 / 8.75) and proche(lim[2], 42 * e355 / 0.67)
      and [round(x, 2) for x in lim] == [36.82, 42.40, 51.00], lim)
    lim, _ = C_.limites_flexion_composee(0.25, -2.0, 1.0)
    c("tableau 5.2 : α = 0,25 et ψ = −2 → 36ε/α = 144 ; 41,5ε/α = 166 ; 62ε·(1 − ψ)·√(−ψ) = 263,0",
      proche(lim[0], 144.0) and proche(lim[1], 166.0) and proche(lim[2], 62 * 3 * math.sqrt(2.0)), lim)
    lim, _ = C_.limites_flexion_composee(0.0, -3.0, 1.0)
    c("tableau 5.2 : âme tendue sur toute sa hauteur (α = 0) → pas de limite de classe 1 ni 2", lim[0] is None and lim[1] is None, lim)
    c("tableau 5.2 : α et ψ d'une âme — α = 0,5 + N/(2·c·t·f_y), ψ = 2·N/(A·f_y) − 1, bornés (IPE 600 S355, N = 2 000 kN : 0,9567 et −0,2776)",
      all(proche(x, y) for x, y in zip(C_.alpha_psi(2000e3, 514.0, 12.0, 15599.0, 355.0),
                                       (0.5 + 2000e3 / (2 * 514.0 * 12.0 * 355.0), 2 * 2000e3 / (15599.0 * 355.0) - 1)))
      and C_.alpha_psi(1e9, 514.0, 12.0, 15599.0, 355.0) == (1.0, 1.0) and C_.alpha_psi(-1e9, 514.0, 12.0, 15599.0, 355.0)[0] == 0.0)
    ks = [(1.0, 4.0), (0.5, 8.2 / 1.55), (0.0, 7.81), (-0.5, 7.81 + 3.145 + 2.445), (-1.0, 23.9), (-2.0, 5.98 * 9)]
    c("EN 1993-1-5 tableau 4.1 : k_σ d'une paroi interne pour ψ = 1 ; 0,5 ; 0 ; −0,5 ; −1 ; −2 → 4 ; 5,29 ; 7,81 ; 13,40 ; 23,9 ; 53,82",
      all(proche(C_.k_sigma_interne(p), k) for p, k in ks), [(p, C_.k_sigma_interne(p)) for p, _ in ks])
    c("EN 1993-1-5 tableau 4.2 : k_σ d'une console (bord libre le plus comprimé) pour ψ = 1 ; 0 ; −1 → 0,43 ; 0,57 ; 0,85",
      all(proche(C_.k_sigma_console(p), k) for p, k in ((1.0, 0.43), (0.0, 0.57), (-1.0, 0.85))))
    c("EN 1993-1-5 § 4.4 : ρ d'une paroi interne — 1 jusqu'à 0,673 (ψ = 1) et jusqu'à 0,874 (ψ = −1) ; λ̄_p = 1 → 0,78 (ψ = 1), 0,89 (ψ = −1)",
      C_.rho_interne(0.673, 1.0) == 1.0 and C_.rho_interne(0.874, -1.0) == 1.0 and proche(C_.rho_interne(1.0, 1.0), 0.78)
      and proche(C_.rho_interne(1.0, -1.0), 0.89) and C_.rho_interne(0.70, 1.0) < 1.0)
    c("EN 1993-1-5 § 4.4 : ρ d'une console — 1 jusqu'à 0,748 ; λ̄_p = 1 → 0,812 ; λ̄_p = 2 → 0,453",
      C_.rho_console(0.748) == 1.0 and proche(C_.rho_console(1.0), 0.812) and proche(C_.rho_console(2.0), 0.453) and C_.rho_console(0.76) < 1.0)
    ecarts = []
    for i in range(1, 201):
        ct, eps = 4.0 + 0.5 * i, 0.70 + 0.0015 * i
        for typ in ("interne", "console"):
            lp = NO.elancement_paroi(typ, ct, eps)
            mien = C_.lambda_p(ct, eps, 4.0 if typ == "interne" else C_.K_SIGMA_CONSOLE)
            rho = C_.rho_interne(mien, 1.0) if typ == "interne" else C_.rho_console(mien)
            if not (proche(lp, mien) and proche(NO.rho(typ, lp), rho)):
                ecarts.append((typ, ct, eps))
    c("ψ = 1 : λ̄_p et ρ sont ceux du moteur du flambement en service (acier/flambement/normes.py), 400 parois", not ecarts, ecarts[:3])

    # ---- 2. les classes
    def classes(nom, nuance, N=0.0, My=0.0, Mz=0.0):
        """La classe comme le moteur la fait : f_y de calcul (la plus forte paroi) pour α et ψ, ε de chaque paroi selon
        SON épaisseur (brief § 7.1 ; tableau 5.2) — lot A, 16/11/2026."""
        s = SE.du_catalogue(nom)
        d = S_.lire({"section": s, "nuance": nuance})
        cl = C_.classer(s, d["fy"], N, My, Mz, d["fy_parois"])
        return cl["classe"], {p["role"]: p["classe"] for p in cl["parois"]}, cl

    # ---- 2 bis. la f_y selon l'épaisseur et l'ε paroi par paroi (lot A : table unique, option a de l'ANB:2018 ; brief § 7.1)
    d300 = S_.lire({"section": SE.du_catalogue("HEB 300"), "nuance": "S235"})
    cl300 = C_.classer(SE.du_catalogue("HEB 300"), d300["fy"], 0.0, 1.0, 0.0, d300["fy_parois"])
    eps_f, eps_w = (next(q["epsilon"] for q in cl300["parois"] if q["role"] == r_) for r_ in ("semelle", "ame"))
    c("HEB 300 S235 (t_f = 19, t_w = 11 mm) — test 25 du brief : f_y de calcul 225 N/mm² (tableau 3.3 ANB, 16 < t ≤ 40) ; ε des semelles "
      "= √(235/225) = 1,022, ε de l'âme = √(235/235) = 1,000 — chaque limite c/t avec l'ε de SA paroi",
      d300["fy"] == 225.0 and d300["fy_parois"] == {"semelle": 225.0, "ame": 235.0} and round(eps_f, 3) == 1.022 and eps_w == 1.0
      and proche(next(q for q in cl300["parois"] if q["role"] == "semelle")["limites"][0], 9.0 * math.sqrt(235.0 / 225.0))
      and proche(next(q for q in cl300["parois"] if q["role"] == "ame")["limites"][0], 72.0)
      and "tableau 3.3 ANB" in d300["fy_source"] and "16 < t ≤ 40 mm" in d300["fy_source"], (d300["fy_parois"], eps_f, eps_w))
    srcs = {nom: S_.lire({"section": SE.du_catalogue(nom), "nuance": "S355"})["fy_source"] for nom in ("CHS 168.3x6.3", "SHS 200x200x6.3", "IPE 300")}
    froid = next(q["nom"] for q in CAT.profils() if q["famille"] == "SHS-F")
    srcs[froid] = S_.lire({"section": SE.du_catalogue(froid), "nuance": "S355"})["fy_source"]
    c("le PRODUIT selon la fabrication : un tube fini à chaud lit le tableau 3.8 ANB (EN 10210-1), un tube formé à froid le tableau 3.10 ANB "
      "(EN 10219-1), un profilé laminé le tableau 3.3 ANB (EN 10025-2)",
      "tableau 3.8 ANB" in srcs["CHS 168.3x6.3"] and "tableau 3.8 ANB" in srcs["SHS 200x200x6.3"] and "tableau 3.10 ANB" in srcs[froid]
      and "tableau 3.3 ANB" in srcs["IPE 300"], srcs)

    c("IPE 300 S235 : classe 1 en flexion ; classe 2 en compression (âme c/t = 35,0 entre 33ε et 38ε)",
      classes("IPE 300", "S235", My=1.0)[0] == 1 and classes("IPE 300", "S235", N=1.0)[:2] == (2, {"semelle": 1, "ame": 2}),
      (classes("IPE 300", "S235", My=1.0)[:2], classes("IPE 300", "S235", N=1.0)[:2]))
    c("HEA 260 et HEA 300 en S355 : classe 3 par la semelle (c/t = 8,18 et 8,48 > 10ε = 8,14) ; classe 1 en S235 (c/t ≤ 9)",
      classes("HEA 260", "S355", My=1.0)[:2] == (3, {"semelle": 3, "ame": 1}) and classes("HEA 300", "S355", My=1.0)[0] == 3
      and classes("HEA 300", "S235", My=1.0)[0] == 1, (classes("HEA 260", "S355", My=1.0)[:2], classes("HEA 300", "S235", My=1.0)[:2]))
    c("HEB 200 S355 : classe 1 en compression comme en flexion", classes("HEB 200", "S355", N=1.0)[0] == 1 and classes("HEB 200", "S355", My=1.0)[0] == 1)
    c("IPE 600 S355 : classe 4 en compression (âme c/t = 42,8 > 42ε = 34,2) ; classe 1 en flexion",
      classes("IPE 600", "S355", N=1.0)[:2] == (4, {"semelle": 1, "ame": 4}) and classes("IPE 600", "S355", My=1.0)[0] == 1)
    # IPE 600 S355, âme fléchie et comprimée, à la main : c = 514, t_w = 12, A = 155,99 cm²
    p = brut("IPE 600")
    cw, A6 = p["cw_tw"] * p["tw"], p["A"] * 100.0
    fyd6 = 345.0                                   # IPE 600 S355 : t_f = 19 mm → 345 (tableau 3.3 ANB) ; l'âme (12 mm) garde 355, son ε = e355
    attendus = []
    for N in (500e3, 1000e3, 2000e3, 4000e3):
        al = min(0.5 + N / (2 * cw * p["tw"] * fyd6), 1.0)
        ps = 2 * N / (A6 * fyd6) - 1
        l1, l2 = (396 * e355 / (13 * al - 1), 456 * e355 / (13 * al - 1)) if al > 0.5 else (36 * e355 / al, 41.5 * e355 / al)
        l3 = 42 * e355 / (0.67 + 0.33 * ps) if ps > -1 else 62 * e355 * (1 - ps) * math.sqrt(-ps)
        attendus.append(1 if p["cw_tw"] <= l1 else 2 if p["cw_tw"] <= l2 else 3 if p["cw_tw"] <= l3 else 4)
    obtenus = [classes("IPE 600", "S355", N=N, My=1.0)[1]["ame"] for N in (500e3, 1000e3, 2000e3, 4000e3)]
    c("IPE 600 S355, âme fléchie ET comprimée, calculée à la main : N = 500, 1 000, 2 000, 4 000 kN → classes 1, 2, 3, 4",
      obtenus == attendus == [1, 2, 3, 4], (obtenus, attendus))
    cl = classes("IPE 600", "S355", N=2000e3, My=1.0)[2]
    ame = next(q for q in cl["parois"] if q["role"] == "ame")
    c("IPE 600 S355, N = 2 000 kN : l'âme est dite « flexion composée », α = 0,970 et ψ = −0,257 (f_y de calcul 345), limite de classe 3 "
      "= 42·ε_w/(0,67 + 0,33·ψ) = 58,4 (ε de l'âme : 355)",
      ame["etat"] == C_.FLEXION_COMPOSEE and round(ame["alpha"], 3) == 0.970 and round(ame["psi"], 3) == -0.257 and round(ame["limites"][2], 1) == 58.4
      and proche(ame["limites"][2], 42 * e355 / (0.67 + 0.33 * ame["psi"])) and ame["epsilon"] == e355, ame)
    c("traction seule : classe 1, aucune paroi comprimée ; traction + flexion : l'âme profite de la traction (α < 0,5)",
      classes("IPE 600", "S355", N=-1000e3)[0] == 1 and classes("IPE 600", "S355", N=-1000e3)[2]["parois"][1]["etat"] == C_.TENDUE
      and classes("IPE 600", "S355", N=-1000e3, My=1.0)[2]["parois"][1]["alpha"] < 0.5)
    c("IPE 600 S355 sous M_z + N : l'âme est classée en compression uniforme (classe 4) ; sous M_z seul, elle n'est pas comprimée",
      classes("IPE 600", "S355", N=1.0, Mz=1.0)[1]["ame"] == 4 and classes("IPE 600", "S355", Mz=1.0)[1]["ame"] == 1)
    c("tube rond : d/t contre 50ε², 70ε², 90ε² — CHS 168,3x6,3 S235 classe 1 (26,7) ; CHS 406,4x6,3 classe 3 en S235 (64,5 ≤ 70) et 4 en S355 (> 59,6)",
      classes("CHS 168.3x6.3", "S235", N=1.0)[0] == 1 and classes("CHS 406.4x6.3", "S235", My=1.0)[0] == 2
      and classes("CHS 406.4x6.3", "S355", My=1.0)[0] == 4, (classes("CHS 406.4x6.3", "S235", My=1.0)[0], classes("CHS 406.4x6.3", "S355", My=1.0)[0]))
    c("RHS 200x100x5 S355 (parois b − 3t = 85 et h − 3t = 185) : compression → paroi h classe 4 (37 > 34,2) ; flexion d'axe fort → classe 1 "
      "(paroi h fléchie) ; flexion d'axe faible → classe 4 (paroi h comprimée) ; flexion déviée → classe 4 (du côté sûr)",
      classes("RHS 200x100x5", "S355", N=1.0)[1] == {"b": 1, "h": 4} and classes("RHS 200x100x5", "S355", My=1.0)[0] == 1
      and classes("RHS 200x100x5", "S355", Mz=1.0)[1] == {"b": 1, "h": 4} and classes("RHS 200x100x5", "S355", My=1.0, Mz=1.0)[0] == 4)
    c("tube carré : les DEUX âmes portent l'effort normal — α = 0,5 + N/(4·c·t·f_y) (SHS 200x200x6,3 S355, N = 400 kN : 0,656)",
      proche(next(q for q in classes("SHS 200x200x6.3", "S355", N=400e3, My=1.0)[2]["parois"] if q["role"] == "h")["alpha"],
             0.5 + 400e3 / (4 * (200 - 3 * 6.3) * 6.3 * 355.0)))
    # tout le catalogue : la compression pure donne la classe et l'aire efficace du module Flambement
    ecarts, n, n_par = [], 0, 0
    for pr in CAT.profils():
        if pr["famille"] not in SE.FAMILLES or SE.FAMILLES[pr["famille"]][0] not in ES.FORMES:
            continue
        s = SE.du_catalogue(pr["nom"])
        for nu in NUANCES:
            try:
                ref = FM._section(pr, nu)
            except NO.HorsDomaine:
                continue                       # épaisseur au-delà de 80 mm (domaine du module), ou tube rond de classe 4 : contrôlés plus bas
            n += 1
            try:
                d_ = S_.lire({"section": s, "nuance": nu})
            except RE.Refus as e_:
                ecarts.append((pr["nom"], nu, "refusé par la barre, calculé par le Flambement", str(e_)[:60]))
                continue
            fyp = d_["fy_parois"]
            cl = C_.classer(s, ref["fy"], 1.0, 0.0, 0.0, fyp)
            aeff = C_.aire_efficace(s, {q["role"]: q["epsilon"] for q in cl["parois"]})["A_eff"] if cl["classe"] == 4 else s["A"]
            n_par += len({fyp.get(q["role"]) for q in cl["parois"]}) > 1
            if d_["fy"] != ref["fy"] or cl["classe"] != ref["classe"] or not proche(aeff, ref["A_eff"]):
                ecarts.append((pr["nom"], nu, d_["fy"], ref["fy"], cl["classe"], ref["classe"], aeff, ref["A_eff"]))
    # (lot A, deuxième passe : le module Flambement prend lui aussi l'ε de CHAQUE paroi, avec la f_y de SON épaisseur — la
    #  parité redevient une ÉGALITÉ, parois de même f_y ou non ; un écart, dans un sens ou dans l'autre, rougit)
    c(f"tout le catalogue en compression pure, quatre nuances ({n} couples profilé-nuance) : même f_y de calcul, même classe et même A_eff "
      f"que le module Flambement en service — y compris quand l'âme et les semelles n'ont pas la même f_y ({n_par} couples : chacune son ε)",
      n > 3500 and n_par > 300 and not ecarts, (n, n_par, ecarts[:3]))

    # ---- 3. la section efficace, recalculée par assemblage
    faux = []
    for (h, b, tf, tw, fy) in ((600.0, 300.0, 10.0, 6.0, 355.0), (800.0, 250.0, 8.0, 5.0, 235.0), (500.0, 400.0, 10.0, 8.0, 460.0),
                              (1000.0, 300.0, 20.0, 6.0, 355.0), (400.0, 300.0, 8.0, 6.0, 355.0)):
        s = i_sans_conge(h, b, tf, tw)
        eps = math.sqrt(235.0 / fy)
        w = C_.module_efficace(s, eps, "y")
        ref, psi, rho = weff_y_I(h, b, tf, tw, fy)
        if not (proche(w["W_eff"], ref, 1e-9) and proche(w["ame"]["psi"], psi, 1e-9) and proche(w["ame"]["rho"], rho, 1e-9)):
            faux.append(("y", h, b, tf, tw, fy, w["W_eff"], ref))
        wz = C_.module_efficace(s, eps, "z")
        refz, rhoz = weff_z_I(h, b, tf, tw, fy)
        if not (proche(wz["W_eff"], refz, 1e-9) and proche(wz["membrure"]["rho"], rhoz, 1e-9)):
            faux.append(("z", h, b, tf, tw, fy, wz["W_eff"], refz))
    c("I sans congé (5 sections, semelles et âmes de classe 4) : W_eff,y et W_eff,z du moteur (trous retirés de la section brute) = "
      "ceux de l'assemblage des parties efficaces, à 10⁻⁹ ; même ψ et même ρ de l'âme", not faux, faux[:2])
    s = i_sans_conge(600.0, 300.0, 10.0, 6.0)
    w = C_.module_efficace(s, e355, "y")
    c("I 600x300x10x6 S355 : la semelle réduite déplace l'axe neutre vers la fibre tendue (e > 0), ψ de l'âme entre −1 et 0, W_eff < W_el",
      w["reduit"] and w["e"] > 0 and -1.0 < w["ame"]["psi"] < 0.0 and w["W_eff"] < s["Wel_y"] and w["membrure"]["rho"] < 1.0, w)
    faux = []
    for (h, b, t, fy) in ((300.0, 300.0, 5.0, 355.0), (400.0, 200.0, 4.0, 355.0), (500.0, 300.0, 6.0, 460.0), (600.0, 200.0, 5.0, 235.0)):
        s = tube_sans_arrondi(h, b, t)
        eps = math.sqrt(235.0 / fy)
        for axe, (hh, bb) in (("y", (h, b)), ("z", (b, h))):
            w = C_.module_efficace(s, eps, axe)
            ref, psi, rho = weff_y_tube(hh, bb, t, fy)
            if not (proche(w["W_eff"], ref, 1e-9) and proche(w["ame"]["psi"], psi, 1e-9)):
                faux.append((axe, h, b, t, fy, w["W_eff"], ref))
    c("caisson à angles vifs (4 sections, 2 axes) : W_eff du moteur = celui de l'assemblage des parties efficaces (paroi comprimée "
      "trouée au milieu, âmes réduites), à 10⁻⁹", not faux, faux[:2])
    s = SE.du_catalogue("IPE 600")
    a = C_.aire_efficace(s, e355)
    lp = p["cw_tw"] / (28.4 * e355 * 2.0)
    c("IPE 600 S355 : λ̄_p = 42,83/(28,4·ε·2) = 0,927 ; ρ = (λ̄_p − 0,22)/λ̄_p² = 0,823 ; A_eff = A − (1 − ρ)·c·t_w = 145,1 cm² ; "
      "pas de retrait dans les semelles",
      proche(a["A_eff"], A6 - (1 - (lp - 0.22) / lp ** 2) * cw * p["tw"]) and round(lp, 3) == 0.927 and round((lp - 0.22) / lp ** 2, 3) == 0.823
      and round(a["A_eff"] / 100, 1) == 145.1 and a["parois"][0]["retrait"] == 0.0, a["A_eff"])
    faux = []
    for pr in CAT.profils():
        if pr["famille"] not in SE.FAMILLES or SE.FAMILLES[pr["famille"]][0] not in ("I", "tube_carre", "tube_rect"):
            continue
        s = SE.du_catalogue(pr["nom"])
        for eps in (1.0, e355, math.sqrt(235.0 / 460.0)):
            for axe in ("y", "z"):
                w = C_.module_efficace(s, eps, axe)
                wel = s[f"Wel_{axe}"]
                if not (0.5 * wel < w["W_eff"] <= wel * (1 + 1e-12)) or (w["reduit"] is False and w["W_eff"] != wel) or w["e"] < 0:
                    faux.append((pr["nom"], round(eps, 3), axe, w["W_eff"] / wel))
            ae = C_.aire_efficace(s, eps)["A_eff"]
            if not 0.5 * s["A"] < ae <= s["A"]:
                faux.append((pr["nom"], round(eps, 3), "A", ae / s["A"]))
    c("tout le catalogue (I, H, tubes carrés et rectangulaires ; S235, S355, S460) : 0,5·W_el < W_eff ≤ W_el, W_eff = W_el quand rien "
      "n'est réduit, axe neutre déplacé vers la fibre tendue, 0,5·A < A_eff ≤ A", not faux, faux[:4])
    try:
        C_.aire_efficace(SE.du_catalogue("CHS 406.4x6.3"), e355)
        nc = False
    except RE.NonCouvert:
        nc = True
    r = v("CHS 406.4x6.3", "S355", NEd=100e3, MyEd=10e6)
    c("tube rond de classe 4 : non couvert (EN 1993-1-6) — la section n'est jamais rendue « ok »",
      nc and r["statut"] == RE.NON_COUVERT and r["verifications"][0]["statut"] == RE.NON_COUVERT and any(x["code"] == "classe4" for x in r["alertes"]),
      (nc, r["statut"]))

    # ---- 4. les résistances, à la main (valeurs brutes du catalogue : cm², cm³ ; mm)
    p = brut("IPE 300")
    Npl, Mply, Mplz = p["A"] * 100 * 235.0, p["Wpl_y"] * 1e3 * 235.0, p["Wpl_z"] * 1e3 * 235.0
    Vplz, Vply = p["Avz"] * 100 * 235.0 / R3, 2.0 * p["b"] * p["tf"] * 235.0 / R3
    r = v("IPE 300", "S235", NEd=100e3, MyEd=10e6, MzEd=1e6, VzEd=10e3, VyEd=10e3)
    c("IPE 300 S235 : N_c,Rd = A·f_y = 1 264,6 kN ; M_pl,y,Rd = 147,7 kN·m ; M_pl,z,Rd = 29,4 kN·m ; V_pl,z,Rd = 348,5 kN ; V_pl,y,Rd = "
      "2·b·t_f·f_y/√3 = 435,5 kN (DÉCISION BUREAU n° 3, brief § 7.5 ; l'analogie A − h_w·t_w donnait 461,8 kN)",
      proche(limite(r, "E-N"), Npl / 1e3) and proche(limite(r, "E-My"), Mply / 1e6) and proche(limite(r, "E-Mz"), Mplz / 1e6)
      and proche(limite(r, "E-Vz"), Vplz / 1e3) and proche(limite(r, "E-Vy"), Vply / 1e3)
      and [round(limite(r, k), 1) for k in ("E-N", "E-My", "E-Mz", "E-Vz", "E-Vy")] == [1264.6, 147.7, 29.4, 348.5, 435.5]
      and any(x["code"] == "Avy" and "2·b·t_f" in x["message"] for x in r["alertes"]),
      [limite(r, k) for k in ("E-N", "E-My", "E-Mz", "E-Vz", "E-Vy")])
    r = v("HEB 200", "S235", VyEd=100e3)
    sw = i_sans_conge(400.0, 200.0, 12.0, 8.0)
    rw = S_.verifier({"section": sw, "fy": 235.0, "VyEd": 100e3})
    c("A_v,y (test 14 du § 11.5) : HEB 200 laminé → 2·b·t_f = 6 000 mm² (V_pl,y,Rd = 814,1 kN) ; un I SOUDÉ garde A − h_w·t_w (6.2.6(3) e))",
      proche(r["grandeurs"]["A_v"]["y"], 6000.0) and round(limite(r, "E-Vy"), 1) == 814.1
      and proche(rw["grandeurs"]["A_v"]["y"], sw["A"] - (400.0 - 24.0) * 8.0), (r["grandeurs"]["A_v"], rw["grandeurs"]["A_v"]))
    # troisième passe : un U — V_y portée par les SEMELLES, jamais l'A_v,z de l'âme (UPN 380 : 52,43 cm², +57 % sur A − h_w·t_w)
    faux_u, vus = [], {}
    for q in CAT.profils():
        if q["famille"] not in ("UPN", "UPE"):
            continue
        av = S_.aires_cisaillement(SE.du_catalogue(q["nom"]), 1.0)
        d_ = q["dims"]
        semelles, geo = 2.0 * d_["b"] * d_["tf"], q["A"] * 100 - (d_["h"] - 2.0 * d_["tf"]) * d_["tw"]
        vus[q["nom"]] = round(av["y"] / 100, 2)
        if not (proche(av["y"], min(semelles, geo)) and proche(av["z"], q["Avz"] * 100)):
            faux_u.append((q["nom"], av["y"], semelles, geo))
    su = SE.saisie("U", "soude_oxycoupe", h=200.0, b=80.0, tw=6.0, tf=11.0, A=2900.0, Iy=1.9e7, Iz=1.9e6, iy=81.0, iz=25.0)
    try:
        S_.aires_cisaillement(SE.du_catalogue("L 100x100x10"), 1.0)
        refus_l = False
    except S_.Refus:
        refus_l = True
    c("A_v,y d'un U laminé (les 32 UPN et UPE du catalogue) : min(2·b·t_f, A − h_w·t_w) — UPN 200 17,18 (A − h_w·t_w) · UPN 380 32,64 · UPE 200 17,60 cm² (2·b·t_f) ; "
      "A_v,z inchangée (celle du catalogue) ; U soudé : A − h_w·t_w (6.2.6(3) e)) ; cornière : refusée (2A/π d'un tube rond, avant)",
      len(vus) == 32 and not faux_u and [vus[k] for k in ("UPN 200", "UPN 380", "UPE 200")] == [17.18, 32.64, 17.6]
      and proche(S_.aires_cisaillement(su, 1.0)["y"], 2900.0 - 178.0 * 6.0) and refus_l, (faux_u[:3], {k: vus.get(k) for k in ("UPN 200", "UPN 380", "UPE 200")}, refus_l))
    r = v("IPE 500", "S235", NEd=500e3, MyEd=200e6, MzEd=25e6, VzEd=100e3, VyEd=50e3)
    cl5 = {q["role"]: q["classe"] for q in r["grandeurs"]["classification"]["parois"]}
    ref21 = {"E-N": 2715.0, "E-My": 516.0, "E-Mz": 78.9, "E-Vz": 814.0}
    ecarts21 = {k: limite(r, k) / x - 1 for k, x in ref21.items()}
    p5 = brut("IPE 500")
    vy_scia = (p5["A"] * 100 - (p5["h"] - 2 * p5["tf"]) * p5["tw"]) * 235.0 / R3 / 1e3
    c("exemple 5 de l'ECCS N°119 (benchmark 21 de l'aide de SCIA Engineer — IPE 500 S235, N_Ed = 500 kN, flexion déviée) : semelles et âme "
      "de classe 1 ; N_c,Rd 2 715 kN, M_c,y,Rd 516 kN·m, M_c,z,Rd 78,9 kN·m, V_pl,z,Rd 814 kN à moins de 0,3 % ; V_pl,y,Rd : SCIA donne 912 kN "
      "avec A − h_w·t_w (919,8 kN à 0,9 %) — le module prend 2·b·t_f (DÉCISION BUREAU n° 3) : 868,3 kN",
      cl5 == {"semelle": 1, "ame": 1} and all(abs(ecarts21[k]) < 0.003 for k in ("E-N", "E-My", "E-Mz", "E-Vz")) and abs(vy_scia / 912.0 - 1) < 0.01
      and proche(limite(r, "E-Vy"), 2 * p5["b"] * p5["tf"] * 235.0 / R3 / 1e3) and round(limite(r, "E-Vy"), 1) == 868.3,
      (cl5, {k: round(x, 4) for k, x in ecarts21.items()}, limite(r, "E-Vy")))
    r = v("HEB 200", "S235", NEd=432.72e3)
    c("HEB 200 S235 (la barre du classeur du bureau, 432,72 kN) : N_c,Rd = 78,08 cm²·235 = 1 835,0 kN, taux 23,6 % ; une seule vérification",
      [x["id"] for x in r["verifications"]] == ["E-N"] and proche(limite(r, "E-N"), brut("HEB 200")["A"] * 100 * 0.235) and round(taux(r, "E-N"), 3) == 0.236, r["verifications"])
    r = v("HEA 300", "S355", MyEd=100e6, MzEd=0.0)
    r1 = v("HEA 300", "S235", MyEd=100e6)
    ph = brut("HEA 300")
    c("HEA 300 : classe 3 en S355 → M_c,y,Rd = W_el,y·f_y = 447,2 kN·m ; classe 1 en S235 → W_pl,y·f_y = 325,1 kN·m",
      proche(limite(r, "E-My"), ph["Wel_y"] * 0.355) and proche(limite(r1, "E-My"), ph["Wpl_y"] * 0.235) and r["grandeurs"]["classe"] == 3, (limite(r, "E-My"), limite(r1, "E-My")))
    r = v("IPE 600", "S355", NEd=3000e3)
    c("IPE 600 S355 en compression : classe 4 → N_c,Rd = A_eff·f_y = 145,1 cm² × 345 = 5 005 kN (A_eff avec l'ε de l'âme, 12 mm → 355 ; f_y de "
      "calcul 345 : t_f = 19 mm) ; l'alerte « classe 4 » est émise",
      proche(limite(r, "E-N"), a["A_eff"] * 0.345) and round(limite(r, "E-N")) == 5005 and any(x["code"] == "classe4" for x in r["alertes"]), limite(r, "E-N"))
    r = v("IPE 600", "S355", NEd=-3000e3)
    c("IPE 600 S355 en traction : N_t,Rd = A·f_y = 15 599 × 345 = 5 381,7 kN (section brute, f_y de la semelle de 19 mm), classe 1",
      proche(limite(r, "E-N"), A6 * 0.345) and verif(r, "E-N")["titre"] == "Traction" and r["grandeurs"]["classe"] == 1, limite(r, "E-N"))
    pt = brut("IPE 750x147")
    hw = pt["h"] - 2 * pt["tf"]
    r = v("IPE 750x147", "S235", VzEd=500e3)
    c("IPE 750x147 (laminé, η = 1,00 — NBN EN 1993-1-5 ANB:2011, 5.1(2), note complémentaire ; test 22 du § 11.5) : le plancher h_w·t_w = "
      "94,9 cm² est SOUS l'aire du catalogue (105,4 cm²) → V_pl,z,Rd = 105,4 cm² × 225/√3 = 1 369,4 kN (t_f = 17 mm → f_y = 225)",
      proche(limite(r, "E-Vz"), pt["Avz"] * 100 * 225.0 / R3 / 1e3) and hw * pt["tw"] < pt["Avz"] * 100
      and "Av_z" not in verif(r, "E-Vz")["formules"] and round(limite(r, "E-Vz"), 1) == 1369.4, limite(r, "E-Vz"))
    r = v("IPE 750x147", "S235", VzEd=500e3, parametres={"eta": 1.2})
    c("le η d'un laminé est le paramètre « eta » : 1,2 donné (l'ancienne valeur), le plancher 1,2·h_w·t_w = 113,9 cm² gouverne (1 479,5 kN)",
      proche(limite(r, "E-Vz"), 1.2 * hw * pt["tw"] * 225.0 / R3 / 1e3) and "Av_z" in verif(r, "E-Vz")["formules"], limite(r, "E-Vz"))
    soude = i_sans_conge(1000.0, 300.0, 20.0, 6.0)
    rw = S_.verifier({"section": soude, "fy": 235.0, "VzEd": 300e3})
    rw1 = S_.verifier({"section": soude, "fy": 235.0, "VzEd": 300e3, "parametres": {"eta_soude": 1.0}})
    c("un I SOUDÉ garde η = 1,20 (sections soudées, jusqu'au S460) : A_v,z = 1,2·960·6 = 6 912 mm² > A − 2·b·t_f + t_w·t_f = 5 880 → "
      "V_pl,z,Rd = 937,8 kN ; seuil de voilement 72·ε/η = 60 (h_w/t_w = 160 : A-Vb) ; « eta_soude » = 1,0 : l'aire de base revient",
      proche(limite(rw, "E-Vz"), 1.2 * 960 * 6 * 235.0 / R3 / 1e3) and "Av_z" in verif(rw, "E-Vz")["formules"] and verif(rw, "A-Vb") is not None
      and proche(next(f for f in rw["calcul"] if f["cle"] == "lim_cis")["resultat"], 60.0)
      and proche(limite(rw1, "E-Vz"), (soude["A"] - 2 * 300 * 20 + 6 * 20) * 235.0 / R3 / 1e3), (limite(rw, "E-Vz"), limite(rw1, "E-Vz")))
    r = v("IPE 750x147", "S355", VzEd=500e3)
    c("IPE 750x147 S355 : h_w/t_w = 54,5 ≤ 72·ε_w/η = 72·0,814/1,00 = 58,6 (ε de l'âme, 13,2 mm → 355) — l'âme n'est plus élancée "
      "(avec η = 1,20 elle l'était : 48,8) ; pas de ligne A-Vb",
      r["statut"] == RE.OK and verif(r, "A-Vb") is None and verif(r, "E-Vz")["statut"] == RE.OK, [x["id"] for x in r["verifications"]])
    pk = brut("IPE 750x134")
    hwk = pk["h"] - 2 * pk["tf"]
    lwk = hwk / (86.4 * pk["tw"] * e355)
    r = v("IPE 750x134", "S355", VzEd=500e3)
    c("IPE 750x134 S355 : h_w/t_w = 59,9 > 58,6 → le voilement par cisaillement est VÉRIFIÉ (A-Vb, EN 1993-1-5 § 5 : λ̄_w = h_w/(86,4·t·ε) "
      "= 0,867, χ_w = 0,83/λ̄_w, V_bw,Rd = 1 722,0 kN sous V_pl,z,Rd = 1 965,4 kN) ; une âme trapue (IPE 300) n'a pas cette ligne",
      r["statut"] == RE.OK and verif(r, "A-Vb")["statut"] == RE.OK and proche(limite(r, "A-Vb"), 0.83 / lwk * 355.0 * hwk * pk["tw"] / R3 / 1e3)
      and round(limite(r, "A-Vb"), 1) == 1722.0 and limite(r, "A-Vb") < limite(r, "E-Vz")
      and verif(r, "E-Vb") is None and verif(r, "E-Vz")["statut"] == RE.OK and verif(v("IPE 300", "S355", VzEd=100e3), "A-Vb") is None
      and v("IPE 300", "S355", VzEd=100e3)["statut"] == RE.OK, (r["statut"], limite(r, "A-Vb")))
    pt = pk
    r = v("IPE 750x134", "S355", MyEd=1500e6, VzEd=1200e3)
    Mf_, Mpl_ = pt["b"] * pt["tf"] * (pt["h"] - pt["tf"]) * 355.0 / 1e6, pt["Wpl_y"] * 1e3 * 355.0 / 1e6
    c("… sous V_z,Ed = 1 200 kN > 0,5·V_bw,Rd et M_y,Ed = 1 500 kN·m > M_f,Rd : l'interaction (7.1) de l'EN 1993-1-5 est vérifiée avec le moment "
      "d'AXE FORT (A-VbM = η̄₁ + (1 − M_f,Rd/M_pl,Rd)·(2·η̄₃ − 1)²)",
      verif(r, "A-VbM") is not None and proche(verif(r, "A-VbM")["valeur"], 1500.0 / Mpl_ + (1.0 - Mf_ / Mpl_) * (2.0 * 1200.0 / limite(r, "A-Vb") - 1.0) ** 2),
      verif(r, "A-VbM"))
    # flexion + effort tranchant (6.30) : IPE 300 S235, V = 250 kN
    rho = (2 * 250e3 / Vplz - 1) ** 2
    Aw = (p["h"] - 2 * p["tf"]) * p["tw"]
    MyV = (p["Wpl_y"] * 1e3 - rho * Aw ** 2 / (4 * p["tw"])) * 235.0
    r = v("IPE 300", "S235", MyEd=100e6, VzEd=250e3)
    c("IPE 300 S235, M_y = 100 kN·m et V_z = 250 kN (> 0,5·V_pl) : ρ = 0,189 ; M_y,V,Rd = (W_pl,y − ρ·A_w²/(4·t_w))·f_y = 141,5 kN·m (6.30)",
      proche(limite(r, "E-MV"), MyV / 1e6) and round(r["grandeurs"]["rho_z"], 3) == 0.189 and round(limite(r, "E-MV"), 1) == 141.5
      and proche(taux(r, "E-MV"), 100e6 / MyV), (limite(r, "E-MV"), r["grandeurs"]["rho_z"]))
    r = v("IPE 300", "S235", MyEd=100e6, VzEd=0.5 * Vplz)
    c("V_Ed = 0,5·V_pl,Rd tout juste : pas de réduction, pas de vérification « flexion et effort tranchant »",
      verif(r, "E-MV") is None and r["grandeurs"]["rho"] == 0.0, [x["id"] for x in r["verifications"]])
    r = v("IPE 300", "S235", MyEd=100e6, VzEd=0.55 * Vplz)
    c("V_Ed = 0,55·V_pl,Rd : la réduction commence dès 0,5 — ρ = (2·0,55 − 1)² = 0,01",
      verif(r, "E-MV") is not None and proche(r["grandeurs"]["rho_z"], 0.01), r["grandeurs"]["rho_z"])
    pr5 = brut("RHS 200x100x5")
    r = v("RHS 200x100x5", "S355", VzEd=100e3, VyEd=50e3)
    c("RHS 200x100x5 S355 : A_v,z = A·h/(b + h) = 19,15 cm² → V_pl,z,Rd = 392,6 kN ; A_v,y = A·b/(b + h) = 9,58 cm² → V_pl,y,Rd = 196,3 kN",
      proche(limite(r, "E-Vz"), pr5["A"] * 100 * 200 / 300.0 * 355.0 / R3 / 1e3, 1e-5) and proche(limite(r, "E-Vy"), pr5["A"] * 100 * 100 / 300.0 * 355.0 / R3 / 1e3, 1e-5)
      and [round(limite(r, k), 1) for k in ("E-Vz", "E-Vy")] == [392.6, 196.3], [limite(r, k) for k in ("E-Vz", "E-Vy")])
    r = v("HEA 300", "S355", MyEd=100e6, VzEd=600e3)
    VplH = ph["Avz"] * 100 * 355.0 / R3
    rhoH = (2 * 600e3 / VplH - 1) ** 2
    c("HEA 300 S355 (classe 3) sous effort tranchant élevé : (1 − ρ)·M_c,y,Rd, du côté sûr — et c'est dit dans les hypothèses",
      proche(limite(r, "E-MV"), (1 - rhoH) * ph["Wel_y"] * 0.355) and any("du côté sûr" in h_ for h_ in r["hypotheses"]), limite(r, "E-MV"))
    r = v("IPE 300", "S235", MyEd=10e6, VzEd=1.05 * Vplz)
    c("IPE 300, V_Ed ≥ V_pl,Rd : l'effort tranchant est refusé ; ρ plafonné à 1 — l'âme est épuisée, les semelles seules portent le moment "
      "((W_pl,y − A_w²/(4·t_w))·f_y = 115,3 kN·m)",
      verif(r, "E-Vz")["statut"] == RE.NOK and r["statut"] == RE.NOK and r["grandeurs"]["rho_z"] == 1.0
      and proche(limite(r, "E-MV"), (p["Wpl_y"] * 1e3 - Aw ** 2 / (4 * p["tw"])) * 235.0 / 1e6) and round(limite(r, "E-MV"), 1) == 115.3, limite(r, "E-MV"))
    r = v("HEA 300", "S355", MyEd=10e6, VzEd=1.05 * VplH)
    c("HEA 300 S355 (classe 3), V_Ed ≥ V_pl,Rd : la flexion n'a plus de réserve — refus sans taux, jamais de division par zéro",
      verif(r, "E-Vz")["statut"] == RE.NOK and verif(r, "E-MV")["statut"] == RE.NOK and verif(r, "E-MV")["taux"] is None and r["statut"] == RE.NOK)
    # flexion composée, I de classe 1 : IPE 300 S235
    a_ = (p["A"] * 100 - 2 * p["b"] * p["tf"]) / (p["A"] * 100)
    n = 400e3 / Npl
    MNy = Mply * (1 - n) / (1 - 0.5 * a_)
    r = v("IPE 300", "S235", NEd=400e3, MyEd=100e6)
    c("IPE 300 S235, N = 400 kN et M_y = 100 kN·m : n = 0,316 ; a = 0,404 ; M_N,y,Rd = M_pl,y·(1 − n)/(1 − 0,5·a) = 126,5 kN·m (6.36) ; taux 79,1 %",
      proche(limite(r, "E-NM"), MNy / 1e6) and round(n, 3) == 0.316 and round(a_, 3) == 0.404 and round(limite(r, "E-NM"), 1) == 126.5
      and round(taux(r, "E-NM"), 3) == 0.791, (limite(r, "E-NM"), taux(r, "E-NM")))
    r = v("IPE 300", "S235", NEd=100e3, MyEd=100e6)
    c("N faible (n = 0,079 ≤ a/2) : M_N,y,Rd = M_pl,y,Rd, non réduit — et c'est dit",
      proche(limite(r, "E-NM"), Mply / 1e6) and verif(r, "E-NM")["messages"] == ["Effort normal faible : moment plastique non réduit."], verif(r, "E-NM"))
    n6 = 0.6
    MNz = Mplz * (1 - ((n6 - a_) / (1 - a_)) ** 2)
    r = v("IPE 300", "S235", NEd=n6 * Npl, MzEd=10e6)
    r1 = v("IPE 300", "S235", NEd=0.3 * Npl, MzEd=10e6)
    c("IPE 300 S235 en flexion d'axe faible : n = 0,6 > a → M_N,z,Rd = M_pl,z·[1 − ((n − a)/(1 − a))²] = 26,2 kN·m (6.38) ; n = 0,3 ≤ a → M_pl,z (6.37)",
      proche(limite(r, "E-NM"), MNz / 1e6) and round(limite(r, "E-NM"), 1) == 26.2 and proche(limite(r1, "E-NM"), Mplz / 1e6), (limite(r, "E-NM"), limite(r1, "E-NM")))
    n = 400e3 / Npl
    dev = (60e6 / MNy) ** 2 + (10e6 / Mplz) ** (5 * n)
    r = v("IPE 300", "S235", NEd=400e3, MyEd=60e6, MzEd=10e6)
    c("IPE 300 S235, flexion composée déviée (6.41) : α = 2, β = 5n = 1,58 → (60/126,5)² + (10/29,4)^1,58 = 40,6 %",
      proche(taux(r, "E-NM"), dev) and proche(r["grandeurs"]["beta"], 5 * n) and r["grandeurs"]["alpha"] == 2.0 and round(dev, 3) == 0.406
      and round(5 * n, 2) == 1.58, (taux(r, "E-NM"), dev))
    r = v("IPE 300", "S235", MyEd=60e6, MzEd=10e6)
    c("flexion déviée sans effort normal : β = 1 (5n < 1) → (M_y/M_pl,y)² + M_z/M_pl,z",
      proche(taux(r, "E-NM"), (60e6 / Mply) ** 2 + 10e6 / Mplz) and r["grandeurs"]["beta"] == 1.0 and verif(r, "E-NM")["titre"] == "Flexion déviée", taux(r, "E-NM"))
    r = v("IPE 300", "S235", NEd=1.0 * Npl, MyEd=1e6)
    c("N_Ed = N_pl,Rd avec un moment : la flexion composée est refusée (aucune réserve), sans division par zéro",
      verif(r, "E-NM")["statut"] == RE.NOK and verif(r, "E-NM")["taux"] is None and r["statut"] == RE.NOK)
    r = v("IPE 300", "S235", NEd=-400e3, MyEd=100e6)
    c("traction et flexion : même interaction avec |N| (M_N,y,Rd = 126,5 kN·m), titre « Traction et flexion »",
      proche(limite(r, "E-NM"), MNy / 1e6) and verif(r, "E-NM")["titre"] == "Traction et flexion")
    # N + M + V : (1 − ρ) sur tout
    r = v("IPE 300", "S235", NEd=400e3, MyEd=60e6, VzEd=250e3)
    nV = 400e3 / ((1 - rho) * Npl)
    c("IPE 300 S235, N + M_y + V_z élevé (§ 6.2.10) : (1 − ρ) sur N_pl,Rd et M_pl,y,Rd, puis (6.36) — du côté sûr",
      proche(limite(r, "E-NM"), (1 - rho) * Mply * (1 - nV) / (1 - 0.5 * a_) / 1e6) and "avec effort tranchant" in verif(r, "E-NM")["titre"]
      and verif(r, "E-MV") is None, limite(r, "E-NM"))
    # tubes
    pr_ = brut("SHS 200x200x6.3")
    NplS, MplS = pr_["A"] * 100 * 355.0, pr_["Wpl_y"] * 1e3 * 355.0
    aw = min((pr_["A"] * 100 - 2 * 200 * 6.3) / (pr_["A"] * 100), 0.5)
    nS = 600e3 / NplS
    MN = min(MplS * (1 - nS) / (1 - 0.5 * aw), MplS)
    ab = 1.66 / (1 - 1.13 * nS ** 2)
    r = v("SHS 200x200x6.3", "S355", NEd=600e3, MyEd=50e6, MzEd=30e6)
    c("SHS 200x200x6,3 S355 (classe 2), N = 600 kN, M_y = 50, M_z = 30 kN·m : a_w = a_f = 0,479 ; M_N = M_pl·(1 − n)/(1 − 0,5·a_w) (6.39, 6.40) ; "
      "n = 0,349 ; M_N = 106,4 kN·m ; α = β = 1,66/(1 − 1,13·n²) = 1,93 (6.41) → 0,470^1,93 + 0,282^1,93 = 32,1 %",
      proche(taux(r, "E-NM"), (50e6 / MN) ** ab + (30e6 / MN) ** ab) and round(aw, 3) == 0.479 and round(ab, 2) == 1.93 and r["grandeurs"]["classe"] == 2
      and round(nS, 3) == 0.349 and round(MN / 1e6, 1) == 106.4 and round(taux(r, "E-NM"), 3) == 0.321, (taux(r, "E-NM"), aw, ab, r["grandeurs"]["classe"]))
    n5 = 200e3 / (pr5["A"] * 100 * 235.0)
    r = v("RHS 200x100x5", "S235", NEd=200e3, MyEd=20e6)
    c("RHS 200x100x5 S235, N = 200 kN et M_y = 20 kN·m : a_w = (A − 2·b·t)/A = 0,652, PLAFONNÉ à 0,5 → M_N,y,Rd = M_pl,y·(1 − n)/0,75 = 40,8 kN·m",
      (pr5["A"] * 100 - 2 * 100 * 5) / (pr5["A"] * 100) > 0.5 and proche(limite(r, "E-NM"), pr5["Wpl_y"] * 0.235 * (1 - n5) / 0.75)
      and r["grandeurs"]["a_w"] == 0.5 and round(limite(r, "E-NM"), 1) == 40.8 and r["grandeurs"]["classe"] == 1, (limite(r, "E-NM"), r["grandeurs"].get("a_w")))
    r = v("SHS 200x200x6.3", "S235", NEd=0.95 * pr_["A"] * 100 * 235.0, MyEd=1e6, MzEd=1e6)
    c("tube carré, n = 0,95 : α = β plafonnés à 6 (1 − 1,13·n² < 0 ne donne jamais un exposant négatif)", r["grandeurs"]["alpha"] == 6.0 and r["grandeurs"]["beta"] == 6.0, r["grandeurs"].get("alpha"))
    pc = brut("CHS 168.3x6.3")
    r = v("CHS 168.3x6.3", "S235", NEd=200e3, MyEd=10e6, MzEd=10e6, VzEd=30e3, VyEd=40e3)
    n_c = 200e3 / (pc["A"] * 100 * 235.0)
    c("CHS 168,3x6,3 S235 : la RÉSULTANTE des moments (14,14 kN·m) et celle des efforts tranchants (50 kN) ; M_N,Rd = M_pl,Rd·(1 − n^1,7) "
      "(§ 6.2.9.1(6), AC:2009 — le critère linéaire n + M/M_pl,Rd, 62,9 %, n'est plus employé : brief v12 § 7.5) → taux 40,6 % ; "
      "V_pl,Rd = (2A/π)·f_y/√3 = 276,9 kN",
      proche(taux(r, "E-NM"), math.hypot(10e6, 10e6) / (pc["Wpl_y"] * 1e3 * 235.0 * (1 - n_c ** 1.7)))
      and proche(verif(r, "E-My")["valeur"], math.hypot(10.0, 10.0)) and proche(verif(r, "E-Vz")["valeur"], 50.0)
      and proche(limite(r, "E-Vz"), 2 * pc["A"] * 100 / math.pi * 235.0 / R3 / 1e3, 1e-5) and verif(r, "E-Mz") is None and verif(r, "E-Vy") is None
      and round(taux(r, "E-NM"), 3) == 0.406, (taux(r, "E-NM"), limite(r, "E-Vz")))
    # ---- § 11.5, test 27 : CHS sous N + M (classes 1 et 2), M_N,Rd/M_pl,Rd = 1 − n^1,7
    rap = []
    for n_ in (0.5, 0.8):
        r = v("CHS 168.3x6.3", "S235", NEd=n_ * pc["A"] * 100 * 235.0, MyEd=1e6)
        rap.append(r["grandeurs"]["M_N_y"] / r["grandeurs"]["M_y_Rd"])
    c("§ 11.5 test 27 — CHS sous N + M : n = 0,5 → M_N,Rd/M_pl,Rd = 1 − 0,5^1,7 = 0,692 ; n = 0,8 → 0,316",
      [round(x, 3) for x in rap] == [0.692, 0.316] and proche(rap[0], 1 - 0.5 ** 1.7, 1e-12), rap)
    # ---- § 11.5, test 35 : RHS sous N + M, les plafonds de (6.39) et (6.40)
    pr2 = brut("RHS 200x100x5")
    r = v("RHS 200x100x5", "S235", MyEd=10e6, MzEd=5e6)
    aw_b, af_b = (pr2["A"] * 100 - 2 * 100 * 5) / (pr2["A"] * 100), (pr2["A"] * 100 - 2 * 200 * 5) / (pr2["A"] * 100)
    c("§ 11.5 test 35 — RHS 200x100x5 S235 (classes 1 et 2) sous M_y + M_z, n = 0 et a_w = 0,5 : M_N,y,Rd = M_pl,y,Rd (plafond de (6.39)), et non 1,333·M_pl,y,Rd ; "
      "idem M_N,z,Rd = M_pl,z,Rd (6.40), a_f < 0,5",
      aw_b > 0.5 and af_b < 0.5 and r["grandeurs"]["classe"] <= 2 and r["grandeurs"]["M_N_y"] == r["grandeurs"]["M_y_Rd"] and r["grandeurs"]["M_N_z"] == r["grandeurs"]["M_z_Rd"]
      and r["grandeurs"]["a_w"] == 0.5, (r["grandeurs"].get("M_N_y"), r["grandeurs"].get("M_y_Rd"), r["grandeurs"].get("M_N_z"), r["grandeurs"].get("M_z_Rd")))
    # ---- § 11.5, test 28 : le contrôle élastique (6.1), pour information
    pi3 = brut("IPE 300")
    Iy3, h3, tf3, b3, tw3 = pi3["Iy"] * 1e4, pi3["dims"]["h"], pi3["dims"]["tf"], pi3["dims"]["b"], pi3["dims"]["tw"]
    r = v("IPE 300", "S235", MyEd=140e6)
    crit = (140e6 * h3 / 2 / Iy3 / 235.0) ** 2
    r2 = v("IPE 300", "S235", MyEd=100e6, VzEd=300e3)
    sig, tau = 100e6 * (h3 / 2 - tf3) / Iy3, 300e3 * b3 * tf3 * (h3 - tf3) / 2 / (Iy3 * tw3)
    crit2 = (sig / 235.0) ** 2 + 3 * (tau / 235.0) ** 2
    c("§ 11.5 test 28 — contrôle élastique (6.1) : IPE 300 S235, M_y = 140 kN·m (≤ M_pl,y,Rd = 147,7) : (σ/f_y)² = 1,14 > 1 à la fibre extrême — "
      "verdict « vérifié », le contrôle affiché pour information ; M_y = 100 kN·m et V_z = 300 kN : 1,25 à la jonction âme-semelle (σ et τ au même point, "
      "τ = V·S/(I·t_w))",
      r["statut"] == RE.OK and r2["statut"] == RE.OK and proche(r["grandeurs"]["controle_elastique"]["critere"], crit, 1e-12) and round(crit, 2) == 1.14
      and r["grandeurs"]["controle_elastique"]["point"] == "fibre haute" and any(a["code"] == "elastique" for a in r["alertes"])
      and proche(r2["grandeurs"]["controle_elastique"]["critere"], crit2, 1e-12) and round(crit2, 2) == 1.25
      and r2["grandeurs"]["controle_elastique"]["point"] == "jonction haute" and not any(x["id"].startswith("E-EL") for x in r["verifications"])
      and not any(a["code"] == "elastique" for a in v("IPE 300", "S235", MyEd=100e6)["alertes"]),
      (r["grandeurs"].get("controle_elastique", {}).get("critere"), crit, r2["grandeurs"].get("controle_elastique", {}).get("critere"), crit2))
    # ---- § 11.5, test 29 : semelle en console fléchie et comprimée
    l_c = C_.limites_console_flechie(0.5, "comprimee", 1.0, 0.43)
    l_t = C_.limites_console_flechie(0.5, "tendue", 1.0, 0.43)
    c("§ 11.5 test 29 — semelle en console fléchie et comprimée, α = 0,5 : classe 1 si c/t ≤ 18ε (extrémité comprimée, 9ε/α) ou ≤ 25,46ε (extrémité "
      "tendue, 9ε/(α·√α)) ; classe 3 : 21ε·√k_σ",
      round(l_c[0], 2) == 18.0 and round(l_t[0], 2) == 25.46 and round(l_c[1], 2) == 20.0 and proche(l_c[2], 21 * math.sqrt(0.43)), (l_c, l_t))
    # … et ce que CLASSER prend : la borne prudente de ces limites (α et ψ de la console non calculés)
    e_ = math.sqrt(235.0 / 355.0)
    fyp_ = {"semelle": 355.0, "ame": 355.0}
    sz = next(p for p in C_.classer(SE.du_catalogue("HEA 300"), 355.0, 500e3, 150e6, 30e6, fyp_)["parois"] if p["role"] == "semelle")
    sy = next(p for p in C_.classer(SE.du_catalogue("HEA 300"), 355.0, 500e3, 150e6, 0.0, fyp_)["parois"] if p["role"] == "semelle")
    dessous = all(sz["limites"][i] <= C_.limites_console_flechie(a_, x_, e_, k_)[i] + 1e-12 for i in range(3) for a_ in (0.1, 0.5, 1.0)
                  for x_ in ("comprimee", "tendue") for k_ in (0.43, 0.6, 1.7))
    I11 = SE.saisie("I", "lamine", "I c/t = 11,3", h=420.0, b=234.0, tw=8.0, tf=10.0, A=8000.0, Iy=2.4e8, Iz=2.1e7, iy=173.0, iz=51.0,
                    Wel_y=1.14e6, Wel_z=1.8e5, Wpl_y=1.3e6, Wpl_z=2.8e5, cf_tf=11.3, cw_tw=50.0)
    k3 = [C_.classer(I11, 355.0, 0.0, 100e6, mz_, fyp_)["parois"][0]["classe"] for mz_ in (0.0, 10e6)]
    c("§ 11.5 test 29 — ce que classer PREND pour une semelle en console sous M_z (HEA 300 S355, N + M_y + M_z) : 9ε et 10ε (sous 9ε/α et 9ε/(α·√α) "
      "pour tout α) ; classe 3 : 21ε·√0,43 = 13,77ε (sous 21ε·√k_σ pour tout k_σ du tableau 4.2, et sous les 14ε de la compression uniforme) ; sans "
      "M_z : 14ε — une semelle de c/t = 11,3 (entre 13,77ε et 14ε) passe de la classe 3 à la classe 4 quand M_z s'ajoute",
      proche(sz["limites"][0], 9 * e_) and proche(sz["limites"][1], 10 * e_) and proche(sz["limites"][2], 21 * e_ * math.sqrt(0.43))
      and sz["regles"][2][0] == C_.GABARIT_LIM3_CONSOLE and dessous and proche(sy["limites"][2], 14 * e_) and k3 == [3, 4],
      (sz["limites"], sy["limites"], k3))
    # A_v,y : le message dit la règle PRISE (aires_cisaillement, « regle_y »)
    def msg_avy(r_):
        return [a["message"] for a in r_["alertes"] if a["code"] == "Avy"]
    Iso = SE.saisie("I", "soude_oxycoupe", "I soudé", h=420.0, b=234.0, tw=8.0, tf=10.0, A=8000.0, Iy=2.4e8, Iz=2.1e7, iy=173.0, iz=51.0)
    ma = [msg_avy(v("HEB 200", "S235", VyEd=100e3)), msg_avy(v("IPN 200", "S235", VyEd=100e3)),
          msg_avy(S_.verifier({"section": Iso, "nuance": "S235", "VyEd": 100e3}))]
    c("A_v,y : le message dit la règle prise — HEB 200 : 2·b·t_f (décision du bureau) ; IPN 200 : min(2·b·t_f ; A − h_w·t_w) ; un I soudé : A − h_w·t_w",
      ma == [[S_.MSG_AVY[S_.AVY_LAMINE]], [S_.MSG_AVY[S_.AVY_MIN]], [S_.MSG_AVY[S_.AVY_SOUDE]]] and "min(2·b·t_f" in ma[1][0], ma)
    # un I saisi par ses c/t et ses propriétés, sans h ni b : le contrôle élastique n'a pas de points — il n'est pas fait
    Inu = SE.saisie("I", "lamine", "I sans dimensions", A=5381.0, Iy=83.56e6, Iz=6.04e6, iy=124.6, iz=33.5, Wpl_y=628.4e3, Wel_y=557.1e3,
                    cf_tf=5.3, cw_tw=35.0, tf=10.7, tw=7.1)
    try:
        rnu = S_.verifier({"section": Inu, "nuance": "S235", "MyEd": 100e6})
        nu_ok = rnu["statut"] == RE.OK and verif(rnu, "E-My") is not None and "controle_elastique" not in rnu["grandeurs"]
    except Exception as e:  # noqa: BLE001
        nu_ok, rnu = False, repr(e)
    c("un I saisi sans h ni b (ses c/t et ses propriétés seulement) : vérifié (E-My), sans contrôle élastique (pour information, il n'a pas de points)",
      nu_ok, rnu if not nu_ok else "")
    # ---- § 19.2 : A_v,y des IPN = la plus petite de 2·b·t_f et de l'aire géométrique A − (h − 2·t_f)·t_w
    faux_ipn, vus_ipn = [], {}
    for q in CAT.profils():
        if q["famille"] != "IPN":
            continue
        d_ = q["dims"]
        attendu = min(2.0 * d_["b"] * d_["tf"], q["A"] * 100 - (d_["h"] - 2.0 * d_["tf"]) * d_["tw"])
        av = S_.aires_cisaillement(SE.du_catalogue(q["nom"]), 1.0)["y"]
        vus_ipn[q["nom"]] = (round(av / 100, 2), round(2.0 * d_["b"] * d_["tf"] / 100, 2))
        if not proche(av, attendu):
            faux_ipn.append((q["nom"], av, attendu))
    c("§ 19.2 — A_v,y des IPN : la plus petite de 2·b·t_f et de A − (h − 2·t_f)·t_w, comme pour les U (IPN 200 : 20,13 cm² au lieu de 20,34)",
      len(vus_ipn) > 10 and not faux_ipn and vus_ipn["IPN 200"] == (20.13, 20.34) and sum(1 for a_, b_ in vus_ipn.values() if a_ < b_) >= 1,
      (faux_ipn[:3], vus_ipn.get("IPN 200")))
    # classe 3 et classe 4
    r = v("HEA 300", "S355", NEd=500e3, MyEd=150e6, MzEd=30e6)
    sig = 500e3 / (ph["A"] * 100) + 150e6 / (ph["Wel_y"] * 1e3) + 30e6 / (ph["Wel_z"] * 1e3)
    c("HEA 300 S355 (classe 3) : σ_x = N/A + M_y/W_el,y + M_z/W_el,z = 44,4 + 119,1 + 71,3 = 234,8 N/mm² ≤ f_y = 355 (6.42), taux 66,2 %",
      proche(verif(r, "E-NM")["valeur"], sig) and proche(limite(r, "E-NM"), 355.0) and round(sig, 1) == 234.8 and round(taux(r, "E-NM"), 3) == 0.662
      and [round(x, 1) for x in (500e3 / (ph["A"] * 100), 150e6 / (ph["Wel_y"] * 1e3), 30e6 / (ph["Wel_z"] * 1e3))] == [44.4, 119.1, 71.3], verif(r, "E-NM"))
    r = v("IPE 600", "S355", NEd=4000e3, MyEd=200e6)
    c("IPE 600 S355, N = 4 000 kN et M_y = 200 kN·m : âme de classe 4 → N/(A_eff·f_y) + M_y/(W_el,y·f_y) = 4 000/5 005 + 200/1 059 = 98,8 % "
      "(6.44, f_y = 345 ; l'âme et les semelles tiennent en flexion seule : W_eff,y = W_el,y)",
      r["grandeurs"]["classe"] == 4 and proche(taux(r, "E-NM"), 4000e3 / (a["A_eff"] * 345.0) + 200e6 / (brut("IPE 600")["Wel_y"] * 1e3 * 345.0))
      and round(taux(r, "E-NM"), 3) == 0.988, (r["grandeurs"]["classe"], taux(r, "E-NM")))
    s = SE.du_catalogue("HEAA 300")
    e460 = math.sqrt(235.0 / 460.0)
    wy = C_.module_efficace(s, e460, "y")
    r = v("HEAA 300", "S460", MyEd=200e6)
    c("HEAA 300 S460 en flexion : semelle de classe 4 (c/t = 11,4 > 14ε = 10,0) → M_c,y,Rd = W_eff,y·f_y, W_eff,y < W_el,y, formules de la paroi nommées",
      r["grandeurs"]["classe"] == 4 and proche(limite(r, "E-My"), wy["W_eff"] * 460.0 / 1e6) and wy["W_eff"] < s["Wel_y"] and wy["reduit"]
      and {"lambda_p_My_m", "rho_My_m", "W_eff_y"} <= set(verif(r, "E-My")["formules"]), (r["grandeurs"]["classe"], limite(r, "E-My")))
    r = v("RHS 200x100x5", "S355", NEd=200e3, MyEd=30e6, MzEd=10e6)
    c("RHS 200x100x5 S355 en flexion composée déviée : classe 4 (du côté sûr, dit dans les hypothèses) → (6.44) à trois termes",
      r["grandeurs"]["classe"] == 4 and any("du côté sûr" in h_ for h_ in r["hypotheses"])
      and proche(taux(r, "E-NM"), 200.0 / limite(r, "E-N") + 30.0 / limite(r, "E-My") + 10.0 / limite(r, "E-Mz")), taux(r, "E-NM"))
    # matériau, paramètres, domaine
    r = v("HEM 300", "S355", NEd=1000e3) if "HEM 300" in CAT.par_nom() else None
    tf_hem = brut("HEM 300")["tf"]
    c(f"HEM 300 S355 (t_f = {tf_hem:g} mm, 16 < t ≤ 40) : f_y = 345 (tableau 3.3 ANB) ; HD 400x1086 (t_f > 80 mm) : hors du domaine du module, "
      "refusé avec un message ; f_y donné passe devant la table",
      r["grandeurs"]["fy"] == 345.0 and _refuse(S_, {"section": SE.du_catalogue("HD 400x1086.0"), "nuance": "S355", "NEd": 1.0})
      and S_.verifier({"section": SE.du_catalogue("HD 400x1086.0"), "fy": 300.0, "NEd": 1e6})["grandeurs"]["fy"] == 300.0)
    epais = next((q["nom"] for q in CAT.profils() if q["famille"] in ("HD", "HEM", "HL") and 40 < (q.get("tf") or 0) <= 80), None)
    r = v(epais, "S355", NEd=1e6)
    tf_ep = brut(epais)["tf"]
    attendu_ep = 335.0 if tf_ep <= 63 else 325.0
    c(f"{epais} S355 (t_f = {tf_ep:g} mm) : f_y = {attendu_ep:g} N/mm² (tableau 3.3 ANB : 335 de 40 à 63 mm, 325 de 63 à 80)",
      r["grandeurs"]["fy"] == attendu_ep, r["grandeurs"]["fy"])
    r = v("IPE 300", "S235", NEd=400e3, MyEd=100e6, parametres={"gamma_M0": 1.1})
    c("γ_M0 = 1,10 (autre annexe) : toutes les résistances divisées par 1,1", proche(limite(r, "E-N"), Npl / 1.1 / 1e3) and proche(limite(r, "E-My"), Mply / 1.1 / 1e6))
    refus = [k for k, b_ in (("section absente", {"nuance": "S235"}), ("nuance inconnue", {"section": SE.du_catalogue("IPE 300"), "nuance": "S999"}),
                             ("effort texte", {"section": SE.du_catalogue("IPE 300"), "nuance": "S235", "NEd": "12"}),
                             ("effort infini", {"section": SE.du_catalogue("IPE 300"), "nuance": "S235", "MyEd": float("inf")}),
                             ("f_y négatif", {"section": SE.du_catalogue("IPE 300"), "fy": -1.0}),
                             ("paramètre inconnu", {"section": SE.du_catalogue("IPE 300"), "nuance": "S235", "parametres": {"gama_M0": 1.0}}),
                             ("paramètre retiré (couple « sans maintiens »)", {"section": SE.du_catalogue("IPE 300"), "nuance": "S235",
                                                                             "parametres": {"lambda_LT0_sans_maintiens": 0.4}}),
                             ("γ_M0 nul", {"section": SE.du_catalogue("IPE 300"), "nuance": "S235", "parametres": {"gamma_M0": 0.0}}))
             if not _refuse(S_, b_)]
    c("domaine : section absente, nuance inconnue, effort texte ou infini, f_y négatif, paramètre inconnu ou retiré, γ_M0 nul — refusés avec un "
      "message", not refus, refus)
    r = S_.verifier({"section": SE.du_catalogue("UPN 200"), "nuance": "S235", "NEd": 100e3})
    r1 = S_.verifier({"section": SE.du_catalogue("L 100x100x10"), "nuance": "S235", "NEd": 100e3})
    c("U et cornière : hors du module, statut « non couvert » (jamais « ok »)", r["statut"] == RE.NON_COUVERT and r1["statut"] == RE.NON_COUVERT)
    r = v("IPE 300", "S235")
    c("aucun effort : aucune vérification, statut « ok », taux None", r["verifications"] == [] and r["statut"] == RE.OK and r["taux_max"] is None)
    r, r1 = v("IPE 300", "S235", NEd=300e3, MyEd=80e6, MzEd=5e6, VzEd=200e3, VyEd=20e3), v("IPE 300", "S235", NEd=300e3, MyEd=-80e6, MzEd=-5e6, VzEd=-200e3, VyEd=-20e3)
    c("le signe des moments et des efforts tranchants ne change rien (section à deux axes de symétrie)",
      [(x["id"], x["taux"]) for x in r["verifications"]] == [(x["id"], x["taux"]) for x in r1["verifications"]])
    return C


def _refuse(S_, barre):
    try:
        S_.verifier(barre)
    except RE.Refus as e:
        return bool(str(e))
    return False


# ------------------------------------------------------------------ 5 et 6. formules, catalogue
def cas_formules():
    cat = SE.du_catalogue
    return [
        {"section": cat("IPE 300"), "nuance": "S235", "NEd": 400e3, "MyEd": 60e6, "MzEd": 10e6, "VzEd": 100e3, "VyEd": 50e3},
        {"section": cat("IPE 300"), "nuance": "S235", "MyEd": 100e6, "VzEd": 250e3},
        {"section": cat("IPE 300"), "nuance": "S235", "NEd": 400e3, "MyEd": 60e6, "VzEd": 250e3, "VyEd": 300e3},
        {"section": cat("IPE 300"), "nuance": "S235", "NEd": 800e3, "MzEd": 10e6},
        {"section": cat("IPE 300"), "nuance": "S235", "NEd": -300e3, "MyEd": 50e6},
        {"section": cat("IPE 600"), "nuance": "S355", "NEd": 1000e3, "MyEd": 300e6},
        {"section": cat("IPE 600"), "nuance": "S355", "NEd": 2000e3, "MyEd": 300e6, "VzEd": 1300e3},
        {"section": cat("IPE 600"), "nuance": "S355", "NEd": 4000e3, "MyEd": 200e6, "MzEd": 20e6, "VzEd": 1300e3},
        {"section": cat("IPE 600"), "nuance": "S355", "NEd": -2500e3, "MyEd": 600e6},
        {"section": cat("IPE 750x147"), "nuance": "S235", "MyEd": 500e6, "VzEd": 1200e3},
        {"section": cat("HEA 300"), "nuance": "S355", "NEd": 500e3, "MyEd": 150e6, "MzEd": 30e6, "VzEd": 600e3},
        {"section": cat("HEA 300"), "nuance": "S355", "MyEd": 150e6, "VzEd": 600e3},
        {"section": cat("HEAA 300"), "nuance": "S460", "NEd": 800e3, "MyEd": 200e6, "MzEd": 40e6},
        {"section": cat("HEAA 300"), "nuance": "S460", "MyEd": 200e6, "MzEd": 40e6, "VzEd": 600e3},
        {"section": cat("SHS 200x200x6.3"), "nuance": "S355", "NEd": 600e3, "MyEd": 50e6, "MzEd": 30e6, "VzEd": 100e3},
        {"section": cat("RHS 200x100x5"), "nuance": "S355", "NEd": 200e3, "MyEd": 30e6, "MzEd": 10e6},
        {"section": cat("RHS 200x100x5"), "nuance": "S355", "NEd": 300e3, "MyEd": 30e6, "VzEd": 300e3},
        {"section": cat("SHS 300x300x6.3"), "nuance": "S355", "NEd": 500e3, "MyEd": 100e6},
        {"section": cat("CHS 168.3x6.3"), "nuance": "S235", "NEd": 200e3, "MyEd": 10e6, "MzEd": 10e6, "VzEd": 150e3, "VyEd": 150e3},
        {"section": cat("CHS 406.4x6.3"), "nuance": "S235", "NEd": 500e3, "MyEd": 100e6},
        {"section": cat("CHS 168.3x6.3"), "nuance": "S235", "MyEd": 20e6, "VzEd": 200e3},
        {"section": i_sans_conge(600.0, 300.0, 10.0, 6.0), "fy": 355.0, "NEd": 500e3, "MyEd": 300e6, "MzEd": 20e6},
        {"section": i_sans_conge(1000.0, 300.0, 20.0, 6.0), "fy": 355.0, "NEd": -500e3, "MyEd": 500e6},     # âme de classe 4 sous traction : ψ < −1
        {"section": i_sans_conge(1000.0, 300.0, 20.0, 6.0), "fy": 235.0, "VzEd": 300e3},                     # I soudé : le plancher η·h_w·t_w
        {"section": cat("HEA 300"), "nuance": "S355", "NEd": 500e3, "MyEd": 150e6, "MzEd": 30e6},
        # (mise à jour 1 : le harnais suit) I soudé, âme de classe 3, semelles de classe 1 : la classe 2 efficace (décision n° 11)
        {"section": i_sans_conge(1000.0, 300.0, 20.0, 10.0), "fy": 235.0, "MyEd": 1500e6},
    ]


def mutations():
    """Chaque altération plausible du moteur — écrite sur son TEXTE — fait rougir au moins un contrôle de référence."""
    cas = [
        ("classe", "classe 1 d'une âme fléchie et comprimée : 396 → 456", "FC_HAUT = (396.0, 456.0)", "FC_HAUT = (456.0, 456.0)"),
        ("classe", "α ≤ 0,5 : 36ε/α → 33ε/α", "FC_BAS = (36.0, 41.5)", "FC_BAS = (33.0, 41.5)"),
        ("classe", "classe 3 : 42ε/(0,67 + 0,33ψ) sans le terme en ψ", "l3, r3 = 42.0 * eps / (0.67 + 0.33 * psi), (\"e_lim3_haut\", None)",
         "l3, r3 = 42.0 * eps / 0.67, (\"e_lim3_haut\", None)"),
        ("classe", "classe 3, ψ ≤ −1 : sans la racine", "62.0 * eps * (1.0 - psi) * math.sqrt(-psi)", "62.0 * eps * (1.0 - psi)"),
        ("classe", "α sans le facteur 2 du dénominateur", "alpha = min(max(0.5 + N / (2.0 * c * t_ames * fy), 0.0), 1.0)",
         "alpha = min(max(0.5 + N / (c * t_ames * fy), 0.0), 1.0)"),
        ("classe", "ψ sans le facteur 2", "psi = min(2.0 * N / (A * fy) - 1.0, 1.0)", "psi = min(N / (A * fy) - 1.0, 1.0)"),
        ("classe", "tube : une seule âme porte l'effort normal", 'alpha, psi = alpha_psi(N, p["c"], p["n"] * p["t"], A, fy)',
         'alpha, psi = alpha_psi(N, p["c"], p["t"], A, fy)'),
        ("classe", "limites de flexion à la place de la compression pour une âme comprimée", "etat = COMPRESSION if N > 0 else TENDUE",
         "etat = FLEXION if N > 0 else TENDUE"),
        ("classe", "tube rond : ε au lieu de ε²", 'k = eps ** 2 if p["type"] == "tube" else eps', "k = eps"),
        ("classe", "paroi d'un tube : b − 2t au lieu de b − 3t", "B_MOINS_3T = 3.0 ", "B_MOINS_3T = 2.0 "),
        ("classe", "la classe de la section : la MEILLEURE paroi", 'return {"classe": max(p["classe"] for p in out)', 'return {"classe": min(p["classe"] for p in out)'),
        ("classe", "tube en flexion déviée : la paroi-semelle classée comme une âme", "            if semelle_de:\n                etat = COMPRESSION\n            elif ame_de:",
         "            if semelle_de and not ame_de:\n                etat = COMPRESSION\n            elif ame_de:"),
        ("classe", "k_σ interne, −1 < ψ < 0 : sans le terme en ψ²", "return 7.81 - 6.29 * psi + 9.78 * psi * psi", "return 7.81 - 6.29 * psi"),
        ("classe", "k_σ d'une console comprimée : 0,57 au lieu de 0,43", "K_SIGMA_CONSOLE = 0.43", "K_SIGMA_CONSOLE = 0.57"),
        ("classe", "ρ interne : 0,055·(3 + ψ) → 0,055·(3 − ψ)", "return min((lp - 0.055 * (3.0 + psi)) / lp ** 2, 1.0)", "return min((lp - 0.055 * (3.0 - psi)) / lp ** 2, 1.0)"),
        ("classe", "ρ console : 0,188 → 0,22", "return min((lp - 0.188) / lp ** 2, 1.0)", "return min((lp - 0.22) / lp ** 2, 1.0)"),
        ("classe", "λ̄_p : 28,4 → 21", "return c_t / (28.4 * eps * math.sqrt(k_sigma))", "return c_t / (21.0 * eps * math.sqrt(k_sigma))"),
        ("classe", "âme efficace : b_e1 = 0,6·b_eff (les deux parts échangées)", "BE1_PSI_NEGATIF = 0.4 ", "BE1_PSI_NEGATIF = 0.6 "),
        ("classe", "âme efficace : ψ de la section BRUTE (−1) au lieu de la section à semelle réduite", "        psi_w = -bas / haut\n", "        psi_w = -1.0\n"),
        ("classe", "âme efficace : b_c = toute la hauteur", "        if psi_w < 0:\n            bc = haut\n", "        if psi_w < 0:\n            bc = c\n"),
        ("classe", "inertie efficace : le terme de transport du trou oublié", "I_eff = inertie - sum(a_ * z * z + i0 for a_, z, i0 in trous) - A_net * zG * zG",
         "I_eff = inertie - sum(i0 for a_, z, i0 in trous) - A_net * zG * zG"),
        ("classe", "W_eff rapporté à la fibre brute SANS le déplacement de l'axe neutre", "W_eff = min(I_eff / (v - zG), Wel)", "W_eff = min(I_eff / v, Wel)"),
        ("classe", "semelle d'un I en flexion d'axe faible : k_σ de la compression uniforme", "            ks = k_sigma_console(psi_m)\n", "            ks = K_SIGMA_CONSOLE\n"),
        ("classe", "aire efficace : une seule console retirée par semelle", '"retrait": p["n"] * (1.0 - rho) * p["c"] * p["t"]', '"retrait": (1.0 - rho) * p["c"] * p["t"]'),
        ("classe", "tube rond de classe 4 calculé comme une paroi", '            raise NonCouvert("Tube rond de classe 4 : EN 1993-1-6, hors du module.")\n', "            continue\n"),
        ("section", "classe 4 en compression : A au lieu de A_eff", 'A_N = ef["A_eff"] if (ef is not None and comprime) else A', "A_N = A"),
        ("section", "classe 3 : W_pl au lieu de W_el", '    elif classe == 3:\n        Wy, Wz = s.get("Wel_y"), s.get("Wel_z")', '    elif classe == 3:\n        Wy, Wz = s.get("Wpl_y"), s.get("Wpl_z")'),
        ("section", "classe 2 : W_el au lieu de W_pl", '    if classe <= 2:\n        Wy, Wz = s.get("Wpl_y"), s.get("Wpl_z")', '    if classe <= 2:\n        Wy, Wz = s.get("Wel_y"), s.get("Wel_z")'),
        ("section", "V_pl,Rd sans √3", "Vpl = av[a] * fy / (math.sqrt(3.0) * gM0)", "Vpl = av[a] * fy / gM0"),
        ("section", "le plancher η·h_w·t_w oublié", '"z": max(base, plancher)', '"z": base'),
        ("section", "aires de cisaillement d'un tube échangées", '"z": avz if avz is not None else A * s["h"] / (s["b"] + s["h"]),\n                "y": avy if avy is not None else A * s["b"] / (s["b"] + s["h"])',
         '"z": avy if avy is not None else A * s["h"] / (s["b"] + s["h"]),\n                "y": avz if avz is not None else A * s["b"] / (s["b"] + s["h"])'),
        ("section", "seuil de l'effort tranchant : 0,6·V_pl,Rd", "SEUIL_V = 0.5 ", "SEUIL_V = 0.6 "),
        ("section", "ρ sans le carré", "rho[a] = min((2.0 * V / Vpl - 1.0) ** 2, 1.0)", "rho[a] = min(2.0 * V / Vpl - 1.0, 1.0)"),
        ("section", "(6.30) : A_w² / (2·t_w)", "MV = min((s[\"Wpl_y\"] - r * Aw ** 2 / (4.0 * s[\"tw\"])) * fy / gM0, MRd)", "MV = min((s[\"Wpl_y\"] - r * Aw ** 2 / (2.0 * s[\"tw\"])) * fy / gM0, MRd)"),
        ("section", "(6.30) appliquée à une classe 3", 'if a == "y" and s["forme"] == "I" and classe <= 2 and rho["y"] == 0.0:', 'if a == "y" and s["forme"] == "I" and rho["y"] == 0.0:'),
        # (le seuil 72·ε/η et la résistance V_bw,Rd sont dans acier/barre/ec3_ame.py : leurs mutations, dans tests/test_barre_ame.py)
        ("section", "voilement par cisaillement passé sous silence", '            if d["voilement"]: ', "            if False: "),
        ("section", "voilement par cisaillement : le moment de l'autre axe", "AM.cisaillement(J, s, fy, P, V, My if a == \"z\" else Mz, N, direction=a,",
         "AM.cisaillement(J, s, fy, P, V, Mz if a == \"z\" else My, N, direction=a,"),
        ("section", "(6.36) : 1 − a au lieu de 1 − 0,5·a (I)", "            f_ = (1.0 - n) / (1.0 - 0.5 * a)\n", "            f_ = (1.0 - n) / (1.0 - a)\n"),
        ("section", "(6.36) : plafond de a porté à 1", "A_MAX = 0.5 ", "A_MAX = 1.0 "),
        ("section", "(6.38) : sans le carré", "MNz = MzRd * (1.0 - ((n - a) / (1.0 - a)) ** 2)", "MNz = MzRd * (1.0 - ((n - a) / (1.0 - a)))"),
        ("section", "(6.41) I : β = 5n sans le plancher 1", "alpha, beta = ALPHA_I, max(5.0 * n, 1.0)", "alpha, beta = ALPHA_I, 5.0 * n"),
        ("section", "(6.41) I : α = 1", "ALPHA_I = 2.0 ", "ALPHA_I = 1.0 "),
        ("section", "(6.41) tubes : 1,66/(1 − 1,13·n) sans le carré", "den = 1.0 - 1.13 * n * n", "den = 1.0 - 1.13 * n"),
        ("section", "(6.40) : a_f avec b au lieu de h", 'af = coefficient("a_f", "a_f", s["h"], t, "h", "t") if N != 0 else min((A - 2.0 * s["h"] * t) / A, A_MAX)',
         'af = coefficient("a_f", "a_f", s["b"] / 2.0, t, "h", "t") if N != 0 else min((A - 2.0 * s["b"] / 2.0 * t) / A, A_MAX)'),
        ("section", "tube rond : moments ajoutés au lieu de la résultante", "        My, Mz = math.hypot(My, Mz), 0.0\n", "        My, Mz = My + Mz, 0.0\n"),
        ("section", "tube rond : effort normal oublié dans l'interaction (brief v12 : M_N,Rd = M_pl,Rd·(1 − n^1,7))", "        MN = MyRd * (1.0 - n ** CHS_EXPOSANT)\n",
         "        MN = MyRd\n"),
        ("section", "tube rond : 1 − n² au lieu de 1 − n^1,7", "CHS_EXPOSANT = 1.7 ", "CHS_EXPOSANT = 2.0 "),
        ("section", "RHS : plafond de (6.39) retiré", "(1.0 - 0.5 * aw)\n            MNy = MyRd * f_ if f_ < 1.0 else MyRd", "(1.0 - 0.5 * aw)\n            MNy = MyRd * f_"),
        ("section", "RHS : plafond de (6.40) retiré", "(1.0 - 0.5 * af)\n            MNz = MzRd * f_ if f_ < 1.0 else MzRd", "(1.0 - 0.5 * af)\n            MNz = MzRd * f_"),
        ("section", "contrôle élastique : 3·τ² → τ²", "critere=(sx / f) ** 2 + 3.0 * (tau / f) ** 2", "critere=(sx / f) ** 2 + 1.0 * (tau / f) ** 2"),
        ("section", "contrôle élastique : τ avec le moment statique de toute la demi-section", "        Sh = Sb = b * tf * (h - tf) / 2.0\n", "        Sh = Sb = b * h * h / 8.0\n"),
        ("section", "contrôle élastique au-delà de 1 passé sous silence", '    if pire["critere"] > 1.0:\n', '    if pire["critere"] > 2.0:\n'),
        ("classe", "console fléchie, extrémité tendue : 9ε/α au lieu de 9ε/(α·√α)", '    div = alpha if extremite == "comprimee" else alpha * math.sqrt(alpha)', "    div = alpha"),
        ("section", "§ 19.2 — IPN : A_v,y = 2·b·t_f seul", 'FAMILLES_AVY_MIN = ("IPN",)', "FAMILLES_AVY_MIN = ()"),
        ("section", "§ 6.2.10 : effort tranchant élevé oublié dans la flexion composée", "    k = 1.0 - r\n", "    k = 1.0\n"),
        ("section", "classe 3 : σ_x sans le terme de l'effort normal", '("N_Ed", abs(N) / 1e3, A), ("M_y,Ed"', '("N_Ed", 0.0, A), ("M_y,Ed"'),
        ("section", "classe 4 traitée par l'interaction plastique", "    if classe == 4:\n        # (6.44)", "    if classe == 5:\n        # (6.44)"),
        ("section", "f_y : épaisseur de l'âme seule", 'return max(float(s.get(k) or 0.0) for k in ("tf", "tw", "t"))', 'return max(float(s.get(k) or 0.0) for k in ("tw", "t"))'),
        ("section", "une forme non couverte rendue sans vérification", '        J.verif("E-SECTION", "Résistance de la section", NORME, "", None, None, "", [], statut=NON_COUVERT,\n                messages=[J.alerte("forme"',
         '        J.verif("E-SECTION", "Résistance de la section", NORME, "", 0.0, 1.0, "", [],\n                messages=[J.alerte("forme"'),
        ("section", "lot A — le produit oublié : tout est « laminé »", '    return PRODUITS.get(s.get("fabrication"), "lamine")', '    return "lamine"'),
        ("section", "lot A — f_y d'une paroi : celle de calcul (un seul ε)", '        return {"semelle": limite_elastique(nuance, s, s["tf"]), "ame": limite_elastique(nuance, s, s["tw"])}',
         '        return {"semelle": limite_elastique(nuance, s), "ame": limite_elastique(nuance, s)}'),
        # (deuxième passe : plus SÉVÈRE que la norme sur les seuls I dont l'âme et les semelles n'ont pas la même f_y — l'ancienne
        #  parité avec le module Flambement, une inégalité sur ces couples, la laissait passer ; l'égalité retrouvée l'attrape)
        ("classe", "lot A — les semelles classées avec l'ε de l'âme (trop petit pour une semelle plus épaisse que l'âme)",
         '        fyp = float((fy_parois or {}).get(role, fy))\n', '        fyp = float((fy_parois or {}).get("ame" if role == "semelle" else role, fy))\n'),
        # (lot C, C0g — le harnais suit : la DÉCISION BUREAU n° 3 est lue dans la table ; la même altération sur la ligne neuve)
        ("section", "lot A — A_v,y d'un I laminé : A − h_w·t_w",
         '(2.0 * s["b"] * s["tf"], AVY_LAMINE) if decision == "2btf"', '(A - hw * s["tw"], AVY_LAMINE) if decision == "2btf"'),
        ("section", "lot C — A_v,y d'un I laminé : la décision n° 3 n'est plus lue dans la table (la variante toujours)",
         '        decision = PA.valeur("avy_lamine") if avy is None else avy', '        decision = "A_moins_hwtw"'),
        ("section", "lot A — η des sections soudées pour un laminé", 'gM0, eta = P["gamma_M0"], AM.eta_de(s, P)', 'gM0, eta = P["gamma_M0"], P["eta_soude"]'),
        ("section", "lot A — section efficace avec l'ε de la f_y de calcul", '    ep = eps_parois or eps\n', '    ep = eps\n'),
        ("classe", "lot A — ε de chaque paroi oublié", '        fyp = float((fy_parois or {}).get(role, fy))\n', '        fyp = fy\n'),
        # (troisième passe : un U — la dernière branche lui donnait l'A_v,z de l'âme pour V_y aussi)
        ("section", "troisième passe — A_v,y d'un U : l'A_v,z de l'âme", 'avy = geo if s.get("fabrication") in AM.SOUDES else min(2.0 * s["b"] * s["tf"], geo)',
         "avy = avz"),
        ("section", "troisième passe — A_v,y d'un U laminé : 2·b·t_f seul (plus grand que A − h_w·t_w sur 13 UPN)",
         'avy = geo if s.get("fabrication") in AM.SOUDES else min(2.0 * s["b"] * s["tf"], geo)',
         'avy = geo if s.get("fabrication") in AM.SOUDES else 2.0 * s["b"] * s["tf"]'),
        ("section", "troisième passe — une cornière prend l'aire d'un tube rond (2A/π)", '    if forme != "tube_rond":\n        raise Refus(',
         '    if False:\n        raise Refus('),
        ("section", "le signe du moment compte", 'My, Mz, Vz, Vy = abs(d["MyEd"]), abs(d["MzEd"]), abs(d["VzEd"]), abs(d["VyEd"])', 'My, Mz, Vz, Vy = d["MyEd"], d["MzEd"], d["VzEd"], d["VyEd"]'),
        # (lot B, reprise du bloc B2)
        ("classe", "lot B — semelle en console sous M_z : classe 3 à 14ε (la compression uniforme)", '            if p["type"] == "console" and fz:', "            if False:"),
        ("section", "lot B — A_v,y : le message de la fabrication, pas de la règle prise", '            J.alerte("Avy", "info", MSG_AVY[av["regle_y"]])',
         '            J.alerte("Avy", "info", MSG_AVY[AVY_SOUDE if s.get("fabrication") in AM.SOUDES else AVY_LAMINE])'),
        ("section", "lot B — IPN : le message de 2·b·t_f", "            avy, regle = min(avy, A - hw * s[\"tw\"]), AVY_MIN", "            avy = min(avy, A - hw * s[\"tw\"])"),
        ("section", "lot B — contrôle élastique d'un I sans dimensions", ' and (PR.est_prs(s) or all(s.get(k) for k in ("h", "b", "tf", "tw"))):', ":"),
    ]
    sources = {k: open(ch, encoding="utf-8").read() for k, ch in CHEMINS.items()}
    res = []
    for fichier, nom, ancien, nouveau in cas:
        source = sources[fichier]
        if source.count(ancien) != 1:
            res.append((nom, None))
            continue
        try:
            mods = {}
            for k in ("classe", "section"):
                spec = importlib.util.spec_from_file_location(f"acier.barre.ec3_{k}_mute", CHEMINS[k])
                mod = importlib.util.module_from_spec(spec)
                mod.__package__ = "acier.barre"
                src = sources[k].replace(ancien, nouveau) if k == fichier else sources[k]
                if k == "section":          # la section mutée (ou non) doit employer la classe mutée
                    src = src.replace("from . import ec3_classe as CL\n", "")
                    mod.__dict__["CL"] = mods["classe"]
                exec(compile(src, CHEMINS[k], "exec"), mod.__dict__)
                mods[k] = mod
            rouges = [n for n, ok, _ in controles(mods["classe"], mods["section"]) if not ok]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
            rouges = [f"exception {type(e).__name__}: {e}"]
        res.append((nom, rouges))
    return res


def main():
    print("== 1 à 4. la norme, les classes, la section efficace, les résistances")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 5. les formules")
    faux, n, utilises = [], 0, set()
    for b in cas_formules():
        r = ES.verifier(b)
        cles = [f["cle"] for f in r["calcul"]]
        if len(cles) != len(set(cles)):
            faux.append((b["section"]["nom"], "clé répétée", [k for k in cles if cles.count(k) > 1][:3]))
        for f in r["calcul"]:
            n += 1
            utilises.add(f["gabarit"])
            if f["gabarit"] not in FO.GABARITS:
                faux.append((b["section"]["nom"], f["cle"], "gabarit absent"))
            elif not proche(FO.evaluer(f), f["resultat"]):
                faux.append((b["section"]["nom"], f["cle"], FO.evaluer(f), f["resultat"]))
        attendues = set(cles)
        for v in r["verifications"]:
            manquent = [k for k in v["formules"] if k not in attendues]
            if manquent:
                faux.append((b["section"]["nom"], v["id"], "formules absentes", manquent))
        for groupe in [r["grandeurs"].get("classification", {}).get("formules", [])] + list((r["grandeurs"].get("efficace") or {}).get("formules", {}).values()):
            if any(k not in attendues for k in groupe):
                faux.append((b["section"]["nom"], "classification / section efficace", "formules absentes"))
    chk(f"chaque bloc de formule du moteur se recalcule par son gabarit, sur ses seuls termes, à 10⁻⁹ ({n} formules, {len(cas_formules())} sections) ; "
        "clés uniques ; chaque vérification nomme des formules qui existent", n > 200 and not faux, faux[:4])
    morts = set(FO.GABARITS_EC3_SECTION) - utilises
    chk("chaque gabarit de la section (Eurocode 3) sert à un contrôle (aucun n'est mort)", not morts, sorted(morts))
    chk("les gabarits des deux méthodes ne se marchent pas dessus (clés distinctes)", not set(FO.GABARITS_EC3) & set(FO.GABARITS_HISTORIQUE))
    r = ES.verifier({"section": SE.du_catalogue("IPE 300"), "nuance": "S235", "NEd": 400e3, "MyEd": 100e6})
    npl = next(f for f in r["calcul"] if f["cle"] == "N_Rd")
    chk("la résistance s'écrit en unités d'usage : N_c,Rd = A·f_y/γ_M0 avec A en mm², f_y en N/mm² → 1 264,6 kN",
        [(t["symbole"], t["unite"]) for t in npl["termes"]] == [("A", "mm²"), ("f_y", "N/mm²"), ("γ_M0", "")] and round(npl["resultat"], 1) == 1264.6
        and npl["unite"] == "kN", npl)

    print("== 6. le catalogue, la sortie")
    plantes, n, statuts, faux = [], 0, set(), []
    jeux = (lambda s, fy: {"NEd": 0.3 * s["A"] * fy}, lambda s, fy: {"MyEd": 0.5 * s["Wel_y"] * fy, "VzEd": 0.1 * s["A"] * fy},
            lambda s, fy: {"NEd": 0.2 * s["A"] * fy, "MyEd": 0.3 * s["Wel_y"] * fy, "MzEd": 0.2 * s["Wel_z"] * fy, "VzEd": 0.05 * s["A"] * fy, "VyEd": 0.05 * s["A"] * fy},
            lambda s, fy: {"NEd": -0.4 * s["A"] * fy, "MzEd": 0.4 * s["Wel_z"] * fy},
            lambda s, fy: {"NEd": 0.7 * s["A"] * fy, "MyEd": 0.2 * s["Wel_y"] * fy, "VzEd": 0.3 * s["A"] * fy},
            lambda s, fy: {"MyEd": 0.6 * s["Wel_y"] * fy, "MzEd": 0.3 * s["Wel_z"] * fy, "VyEd": 0.3 * s["A"] * fy})
    for p in CAT.profils():
        if p["famille"] not in SE.FAMILLES:
            continue
        s = SE.du_catalogue(p["nom"])
        if s["forme"] not in ES.FORMES:
            continue
        for nu in NUANCES:
            for jeu in jeux:
                try:
                    r = ES.verifier({"section": s, "nuance": nu, **jeu(s, 235.0)})
                except RE.Refus:
                    continue                    # épaisseur au-delà de 80 mm (domaine du module)
                except Exception as e:  # noqa: BLE001
                    plantes.append((p["nom"], nu, type(e).__name__, str(e)[:80]))
                    continue
                n += 1
                json.dumps(r, allow_nan=False)
                statuts.add(r["statut"])
                for x in r["verifications"]:
                    if x["taux"] is not None and not (x["taux"] >= 0 and math.isfinite(x["taux"])):
                        faux.append((p["nom"], nu, x["id"], x["taux"]))
                    if x["statut"] == RE.OK and x["taux"] is not None and x["taux"] > 1.0:
                        faux.append((p["nom"], nu, x["id"], "ok au-delà de 1"))
                g = r["grandeurs"]
                if g.get("M_N_y") is not None and g["M_N_y"] > g["M_y_Rd"] * (1 + 1e-12):
                    faux.append((p["nom"], nu, "M_N,y > M_pl,y"))
    chk(f"chaque I, H et tube du catalogue, 4 nuances, 6 jeux d'efforts ({n} calculs) : aucun plantage, JSON strict, taux finis et positifs, "
        "M_N,y,Rd ≤ M_pl,y,Rd ; les trois statuts se rencontrent", n > 20000 and not plantes and not faux and statuts == {RE.OK, RE.NOK, RE.NON_COUVERT},
        (n, plantes[:3], faux[:3], statuts))
    # la flexion composée ne s'allège jamais quand l'effort normal augmente (changements de classe compris)
    recule = []
    for nom, nu in (("IPE 300", "S235"), ("IPE 600", "S355"), ("HEA 300", "S355"), ("HEAA 300", "S460"), ("SHS 200x200x6.3", "S355"),
                    ("RHS 200x100x5", "S355"), ("CHS 168.3x6.3", "S235"), ("SHS 300x300x6.3", "S355")):
        s = SE.du_catalogue(nom)
        avant = 0.0
        for i in range(0, 96):
            r = ES.verifier({"section": s, "nuance": nu, "NEd": i / 100.0 * s["A"] * 235.0, "MyEd": 0.3 * s["Wel_y"] * 235.0})
            x = verif(r, "E-NM") or verif(r, "E-My")
            if x["taux"] is None:
                break
            if x["taux"] < avant - 1e-12:
                recule.append((nom, nu, i, avant, x["taux"]))
            avant = x["taux"]
    chk("8 sections, effort normal croissant de 0 à 95 % de A·235 sous un moment fixe : le taux de la flexion composée ne recule jamais "
        "(changements de classe compris)", not recule, recule[:3])
    b0 = {"section": SE.du_catalogue("IPE 600"), "nuance": "S355", "NEd": 4000e3, "MyEd": 200e6, "VzEd": 100e3, "parametres": {"eta": 1.0}}
    b1 = copy.deepcopy(b0)
    ES.verifier(b1)
    chk("verifier ne modifie pas la barre reçue", b1 == b0)
    r = ES.verifier(b0)
    chk("la sortie a la forme commune aux deux méthodes (acier/barre/resultat.py)",
        set(r) == {"methode", "norme", "statut", "taux_max", "gouvernant", "verifications", "grandeurs", "calcul", "alertes", "hypotheses"}
        # (mise à jour 1, étape 2 — le harnais suit : une vérification peut rendre les grandeurs de SA section, « section », pour le
        #  dessin des contraintes — la flexion en élastique de l'Eurocode 3 : N, M_y, V_z, σ, τ au moins)
        and r["methode"] == "ec3" and all(set(x) - {"section"} == {"id", "methode", "titre", "reference", "critere", "valeur", "limite", "unite", "taux",
                                                                   "statut", "formules", "messages"}
                                          and ("section" not in x or {"N", "My", "Vz", "sigma", "tau"} <= set(x["section"])) for x in r["verifications"]),
        sorted(r))

    print("== 7. mutations")
    for nom, rouges in mutations():
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans le moteur, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")

    print("== 8. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab', 'PIL', 'fitz'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from acier.barre import ec3_section as E, section as S\n"
            "r = E.verifier({'section': S.du_catalogue('IPE 600'), 'nuance': 'S355', 'NEd': 4000e3, 'MyEd': 200e6, 'VzEd': 100e3})\n"
            "print(r['statut'], round(r['taux_max'], 3), sorted(m for m in sys.modules if m.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab')))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le moteur se charge et calcule sans streamlit, pandas, matplotlib, scipy, reportlab (refusés ; numpy ne sert qu'au catalogue)",
        p.returncode == 0 and p.stdout.strip() == "ok 0.988 []", (p.stdout.strip(), p.stderr[-300:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

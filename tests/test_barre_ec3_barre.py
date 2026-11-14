# -*- coding: utf-8 -*-
"""Barres en acier — EUROCODE 3, la BARRE (acier/barre/mcr.py, ec3_flambement.py, ec3_deversement.py,
ec3_interaction.py, ec3.py) : flambement (§ 6.3.1), déversement (§ 6.3.2, M_cr numérique), flexion composée avec
instabilité (§ 6.3.3, annexe A). La spécification : docs/acier/BARRES_ACIER.md § 5.5 à 5.8 et § 6.4.

Les références :
  1. FLAMBEMENT — les 20 valeurs de χ du brief ; le HEB 200 du classeur du bureau recalculé à la main ; les charges de
     flambement du module Flambement EN SERVICE sur tout le catalogue ; la torsion à la main.
  2. M_cr — la formule fermée du moment constant ; les SEPT valeurs R1 d'un solveur indépendant (numpy/scipy, deuxième
     avis du 14/11/2026) ; les C₁ connus du moment linéaire ; la console classique (4,013) ; un maintien de la semelle
     comprimée à mi-portée vaut un maintien complet.
  3. DÉVERSEMENT — les valeurs R3 du deuxième avis (IPE 300, charge répartie + charge centrale) avec les deux couples
     (λ̄_LT,0 ; β) ; k_c, f, les plafonds, les courbes du tableau 6.5.
  4. ANNEXE A — recalculée ICI, d'une seule traite, sur les valeurs brutes du catalogue, pour neuf barres ; les valeurs
     du benchmark 21 de l'aide de SCIA Engineer (ECCS N°119, exemple 5) qui ne dépendent pas du diagramme de M_y.
  5. FORMULES — chaque bloc se recalcule par son gabarit à 10⁻⁹ ; aucun gabarit mort.
  6. CATALOGUE — un I ou H sur deux et un tube sur quatre : JSON strict, invariants.
  7. MUTATIONS — chaque altération plausible du texte du moteur fait rougir un contrôle des § 1 à 4.
  8. PURETÉ.

    /usr/bin/python3.12 tests/test_barre_ec3_barre.py
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

from acier.barre import ec3 as EC  # noqa: E402
from acier.barre import ec3_deversement as DV  # noqa: E402
from acier.barre import ec3_flambement as FL  # noqa: E402
from acier.barre import ec3_interaction as IN  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from acier.barre import mcr as MC  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.flambement import moteur as FM  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []
FICHIERS = ("mcr", "ec3_flambement", "ec3_deversement", "ec3_interaction", "ec3")
CHEMINS = {k: os.path.join(RACINE, "acier", "barre", k + ".py") for k in FICHIERS}
E_, G_ = 210000.0, 81000.0
ALPHA = {"a0": 0.13, "a": 0.21, "b": 0.34, "c": 0.49, "d": 0.76}
PI2 = math.pi ** 2


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def brut(nom):
    return CAT.par_nom()[nom]


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


def chi_de(lb, alpha, lb0=0.2, beta=1.0):
    """χ (ou χ_LT) réécrit ici."""
    ph = 0.5 * (1 + alpha * (lb - lb0) + beta * lb * lb)
    return min(1.0, 1.0 / (ph + math.sqrt(ph * ph - beta * lb * lb)))


def ipe300_du_brief():
    """L'IPE 300 des cas R1 à R3 du brief : les propriétés des catalogues (I_t 20,12 cm⁴, I_w 125,9·10³ cm⁶)."""
    return SE.saisie("I", "lamine", nom="IPE 300 (brief)", h=300.0, b=150.0, tw=7.1, tf=10.7, r=15.0, A=5381.0, Iy=8356e4, Iz=603.8e4,
                     iy=124.6, iz=33.5, Wel_y=557.1e3, Wel_z=80.5e3, Wpl_y=628.4e3, Wpl_z=125.2e3, It=20.12e4, Iw=125.9e9, Avz=2568.0)


# ------------------------------------------------------------------ 4. l'annexe A, recalculée d'une traite
def annexe_A(q):
    """Les facteurs de l'annexe A et les deux expressions (6.61), (6.62), depuis des NOMBRES (N, mm) : A (ou A_eff), Iy,
    It, Wel_y, Wel_z, Wpl_y, Wpl_z, W_y, W_z (ceux de la classe), fy, N, My, Mz, Ncry, Ncrz, NcrT (None), chiy, chiz,
    chiLT, lby, lbz, lb0, C1, Cmy0, Cmz0, classe, Wel_eps (le module de ε_y)."""
    N, My, Mz, fy = q["N"], q["My"], q["Mz"], q["fy"]
    ry, rz = N / q["Ncry"], N / q["Ncrz"]
    rT = N / q["NcrT"] if q.get("NcrT") else 0.0
    muy = (1 - ry) / (1 - q["chiy"] * ry)
    muz = (1 - rz) / (1 - q["chiz"] * rz)
    aLT = max(1 - q["It"] / q["Iy"], 0.0)
    lim = 0.2 * math.sqrt(q["C1"]) * ((1 - rz) * (1 - rT)) ** 0.25
    Cmy, Cmz, CmLT = q["Cmy0"], q["Cmz0"], 1.0
    if q["lb0"] > lim and My > 0:
        if N > 0:
            rac = math.sqrt(My / N * q["A"] / q["Wel_eps"]) * aLT
            Cmy = q["Cmy0"] + (1 - q["Cmy0"]) * rac / (1 + rac)
        else:
            Cmy = 1.0 if aLT > 0 else q["Cmy0"]
        CmLT = max(Cmy ** 2 * aLT / math.sqrt((1 - rz) * (1 - rT)), 1.0)
    out = {"mu_y": muy, "mu_z": muz, "a_LT": aLT, "C_my": Cmy, "C_mz": Cmz, "C_mLT": CmLT, "lambda_0_lim": lim}
    NRk = q["A"] * fy
    if q["classe"] <= 2:
        wy, wz = min(q["Wpl_y"] / q["Wel_y"], 1.5), min(q["Wpl_z"] / q["Wel_z"], 1.5)
        npl = N / NRk
        lm = max(q["lby"], q["lbz"])
        ty = My / (q["chiLT"] * q["Wpl_y"] * fy)
        tz = Mz / (q["Wpl_z"] * fy)
        bLT = 0.5 * aLT * q["lb0"] ** 2 * ty * tz
        cLT = 10 * aLT * q["lb0"] ** 2 / (5 + q["lbz"] ** 4) * ty / Cmy
        dLT = 2 * aLT * q["lb0"] / (0.1 + q["lbz"] ** 4) * ty / Cmy * tz / Cmz
        eLT = 1.7 * aLT * q["lb0"] / (0.1 + q["lbz"] ** 4) * ty / Cmy
        Cyy = max(1 + (wy - 1) * ((2 - 1.6 / wy * Cmy ** 2 * lm - 1.6 / wy * Cmy ** 2 * lm ** 2) * npl - bLT), q["Wel_y"] / q["Wpl_y"])
        Cyz = max(1 + (wz - 1) * ((2 - 14 * Cmz ** 2 * lm ** 2 / wz ** 5) * npl - cLT), 0.6 * math.sqrt(wz / wy) * q["Wel_z"] / q["Wpl_z"])
        Czy = max(1 + (wy - 1) * ((2 - 14 * Cmy ** 2 * lm ** 2 / wy ** 5) * npl - dLT), 0.6 * math.sqrt(wy / wz) * q["Wel_y"] / q["Wpl_y"])
        Czz = max(1 + (wz - 1) * ((2 - 1.6 / wz * Cmz ** 2 * lm - 1.6 / wz * Cmz ** 2 * lm ** 2) * npl - eLT), q["Wel_z"] / q["Wpl_z"])
        kyy = Cmy * CmLT * muy / (1 - ry) / Cyy
        kyz = Cmz * muy / (1 - rz) / Cyz * 0.6 * math.sqrt(wz / wy)
        kzy = Cmy * CmLT * muz / (1 - ry) / Czy * 0.6 * math.sqrt(wy / wz)
        kzz = Cmz * muz / (1 - rz) / Czz
        out.update({"w_y": wy, "w_z": wz, "b_LT": bLT, "c_LT": cLT, "d_LT": dLT, "e_LT": eLT, "C_yy": Cyy, "C_yz": Cyz, "C_zy": Czy, "C_zz": Czz})
    else:
        kyy = Cmy * CmLT * muy / (1 - ry)
        kyz = Cmz * muy / (1 - rz)
        kzy = Cmy * CmLT * muz / (1 - ry)
        kzz = Cmz * muz / (1 - rz)
    MyRk, MzRk = q["W_y"] * fy, q["W_z"] * fy
    out.update({"k_yy": kyy, "k_yz": kyz, "k_zy": kzy, "k_zz": kzz,
                "t61": N / (q["chiy"] * NRk) + kyy * My / (q["chiLT"] * MyRk) + kyz * Mz / MzRk,
                "t62": N / (q["chiz"] * NRk) + kzy * My / (q["chiLT"] * MyRk) + kzz * Mz / MzRk})
    return out


def cm0_de(diag, psi, r):
    return {"lineaire": 0.79 + 0.21 * psi + 0.36 * (psi - 0.33) * r, "constant": 1.0 + 0.36 * 0.67 * r, "reparti": 1 + 0.03 * r,
            "ponctuel": 1 - 0.18 * r}[diag]


def barre_annexe(nom, fy, L, N, My, Mz, dy, dz, alphas, classe, torsion=True, section=None, AN=None, Wy=None, Wz=None, alpha_LT=None,
                 lb0_beta=(0.4, 0.75), kc=1.0, borne=True):
    """Les nombres d'une barre du catalogue pour ``annexe_A`` : tout est recalculé ici (N_cr, χ, M_cr,0, χ_LT avec le
    M_cr NUMÉRIQUE passé par ``section`` = (M_cr, forme du f))."""
    p = brut(nom)
    A, Iy, Iz = p["A"] * 100, p["Iy"] * 1e4, p["Iz"] * 1e4
    It, Iw = p["It"] * 1e4, (p.get("Iw") or 0.0) * 1e6
    AN = AN or A
    Ncry, Ncrz = PI2 * E_ * Iy / L ** 2, PI2 * E_ * Iz / L ** 2
    NcrT = (G_ * It + PI2 * E_ * Iw / L ** 2) / ((p["iy"] * 10) ** 2 + (p["iz"] * 10) ** 2) if torsion else None
    lby, lbz = math.sqrt(AN * fy / Ncry), math.sqrt(AN * fy / Ncrz)
    q = {"A": AN, "Iy": Iy, "It": It, "Wel_y": p["Wel_y"] * 1e3, "Wel_z": p["Wel_z"] * 1e3, "Wpl_y": p["Wpl_y"] * 1e3, "Wpl_z": p["Wpl_z"] * 1e3,
         "fy": fy, "N": N, "My": My, "Mz": Mz, "Ncry": Ncry, "Ncrz": Ncrz, "NcrT": NcrT, "chiy": chi_de(lby, alphas[0]), "chiz": chi_de(lbz, alphas[1]),
         "lby": lby, "lbz": lbz, "classe": classe, "C1": 1.0 / kc ** 2}
    q["W_y"] = Wy or (q["Wpl_y"] if classe <= 2 else q["Wel_y"])
    q["W_z"] = Wz or (q["Wpl_z"] if classe <= 2 else q["Wel_z"])
    q["Wel_eps"] = q["Wel_y"]
    if section is not None and My > 0:                       # déversement : M_cr numérique donné, le reste recalculé
        Mcr = section
        Mcr0 = math.pi / L * math.sqrt(E_ * Iz * G_ * It) * math.sqrt(1 + PI2 * E_ * Iw / (L ** 2 * G_ * It))
        lbLT = math.sqrt(q["W_y"] * fy / Mcr)
        q["lb0"] = math.sqrt(q["W_y"] * fy / Mcr0)
        if lbLT <= lb0_beta[0] or My / Mcr <= lb0_beta[0] ** 2:
            q["chiLT"] = 1.0
        else:
            x = min(chi_de(lbLT, alpha_LT, *lb0_beta), 1 / lbLT ** 2)
            f = min(1 - 0.5 * (1 - kc) * (1 - 2 * (lbLT - 0.8) ** 2), 1.0)
            q["chiLT"] = min(x / f, 1.0, 1 / lbLT ** 2)
    else:
        q["lb0"], q["chiLT"] = 0.0, 1.0
    ry, rz = N / Ncry, N / Ncrz
    q["Cmy0"] = cm0_de(dy[0], dy[1], ry) if My > 0 else 1.0
    q["Cmz0"] = cm0_de(dz[0], dz[1], rz) if Mz > 0 else 1.0
    if borne:
        q["Cmy0"], q["Cmz0"] = max(q["Cmy0"], 1 - ry) if My > 0 else 1.0, max(q["Cmz0"], 1 - rz) if Mz > 0 else 1.0
    return q


# ------------------------------------------------------------------ les contrôles de référence (rejoués par les mutations)
def controles(M=None):
    """[(nom, ok, info)] — ``M`` : les modules à contrôler {"mcr", "ec3_flambement", "ec3_deversement",
    "ec3_interaction", "ec3"} (les mutations y passent leurs copies altérées)."""
    M = M or {"mcr": MC, "ec3_flambement": FL, "ec3_deversement": DV, "ec3_interaction": IN, "ec3": EC}
    mc, ec, dv = M["mcr"], M["ec3"], M["ec3_deversement"]
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    def v(nom, nuance, L=None, **k):
        b = {"section": nom if isinstance(nom, dict) else SE.du_catalogue(nom), "nuance": nuance, **k}
        if L is not None:
            b.setdefault("axes", {"y": {"L": L}, "z": {"L": L}})
        return ec.verifier(b)

    def taux(r, id_):
        x = verif(r, id_)
        return None if x is None else x["taux"]

    def limite(r, id_):
        x = verif(r, id_)
        return None if x is None else x["limite"]

    # ---- 1. le flambement
    brief = {"a0": (0.9513, 0.7253, 0.3953, 0.2323), "a": (0.9243, 0.6656, 0.3724, 0.2229), "b": (0.8842, 0.5970, 0.3422, 0.2095),
             "c": (0.8430, 0.5399, 0.3145, 0.1962), "d": (0.7793, 0.4671, 0.2766, 0.1766)}
    s = SE.du_catalogue("HEB 200")
    ecarts = []
    for courbe, valeurs in brief.items():
        for lb, ref in zip((0.5, 1.0, 1.5, 2.0), valeurs):
            L = lb * math.pi * math.sqrt(E_ * s["Iz"] / (s["A"] * 235.0))          # la longueur qui donne λ̄_z = lb
            r = v(s, "S235", NEd=1e3, axes={"y": {"L": L, "courbe": courbe}, "z": {"L": L, "courbe": courbe}})
            x = r["grandeurs"]["flambement"]["z"]["chi"]
            if round(x, 4) != ref:
                ecarts.append((courbe, lb, x, ref))
    c("χ(λ̄) des cinq courbes pour λ̄ = 0,5 ; 1,0 ; 1,5 ; 2,0 : les 20 valeurs du brief (§ 11.2), au dix-millième", not ecarts, ecarts[:3])
    p = brut("HEB 200")
    Ncr = PI2 * E_ * p["Iz"] * 1e4 / 7200.0 ** 2
    lb = math.sqrt(p["A"] * 100 * 235.0 / Ncr)
    r = v("HEB 200", "S235", 7200.0, NEd=432.72e3)
    F = r["grandeurs"]["flambement"]
    c("HEB 200 S235, 7,20 m (la barre du classeur du bureau) : N_cr,z = π²·E·I_z/L² = 801 kN ; λ̄_z = √(A·f_y/N_cr) = 1,514 ; courbe c ; "
      "χ_z = 0,310 ; N_b,z,Rd = 569,2 kN — taux 76,0 % (le brief donnait 570,1 kN avec i = 5,07 cm arrondi : écart E4)",
      proche(F["z"]["N_cr"], Ncr) and proche(F["z"]["lambda_bar"], lb) and F["z"]["courbe"] == "c" and F["y"]["courbe"] == "b"
      and proche(limite(r, "E-Nb-z"), chi_de(lb, 0.49) * p["A"] * 100 * 0.235) and round(limite(r, "E-Nb-z"), 1) == 569.2
      and round(Ncr / 1e3) == 801 and round(lb, 3) == 1.514 and round(taux(r, "E-Nb-z"), 3) == 0.760 and r["gouvernant"] == "E-Nb-z",
      (F["z"], limite(r, "E-Nb-z")))
    r = v("HEB 200", "S235", NEd=432.72e3, axes={"y": {"L": 7200.0, "u": 2.0}, "z": {"L": 7200.0, "u": 0.5}})
    F = r["grandeurs"]["flambement"]
    c("la longueur de flambement est u·L, par axe : u_y = 2 → N_cr,y divisé par 4 ; u_z = 0,5 → N_cr,z multiplié par 4",
      proche(F["y"]["N_cr"], PI2 * E_ * p["Iy"] * 1e4 / 14400.0 ** 2) and proche(F["z"]["N_cr"], PI2 * E_ * p["Iz"] * 1e4 / 3600.0 ** 2))
    r = v("HEB 200", "S235", 7200.0, NEd=432.72e3, parametres={"gamma_M1": 1.1})
    c("γ_M1 = 1,10 (autre annexe) : N_b,Rd divisé par 1,1 ; la résistance de la section, elle, garde γ_M0",
      proche(limite(r, "E-Nb-z"), chi_de(lb, 0.49) * p["A"] * 100 * 0.235 / 1.1) and proche(limite(r, "E-N"), p["A"] * 100 * 0.235))
    r = v("HEB 200", "S235", NEd=432.72e3, axes={"y": {"L": 7200.0}, "z": {"L": 0.0}})
    c("un axe tenu sur toute la longueur (L = 0) : pas de flambement autour de cet axe, pas de vérification",
      verif(r, "E-Nb-z") is None and verif(r, "E-Nb-y") is not None and r["grandeurs"]["flambement"]["z"]["chi"] == 1.0)
    r = v("HEB 200", "S235", 7200.0, NEd=-432.72e3)
    r0 = v("HEB 200", "S235", 7200.0, MyEd=10e6)
    c("traction, ou pas d'effort normal : aucune vérification de flambement",
      not any(x["id"].startswith("E-Nb") for x in r["verifications"]) and not any(x["id"].startswith("E-Nb") for x in r0["verifications"]))
    r = v("HEB 200", "S235", NEd=800e3, MyEd=20e6, axes={"y": {"L": 9000.0}, "z": {"L": 3000.0}})
    c("à défaut d'autre indication, la torsion et le déversement prennent la longueur de l'axe FAIBLE (3 m ici, l'axe fort flambant sur 9 m)",
      r["grandeurs"]["flambement"]["T"]["Lcr"] == 3000.0 and r["grandeurs"]["deversement"]["troncon"]["L"] == 3000.0)
    r = v("HEB 200", "S235", 500.0, NEd=432.72e3)
    c("barre courte (λ̄ ≤ 0,2) : flambement dit négligeable, χ = 1", verif(r, "E-Nb-z")["messages"] and r["grandeurs"]["flambement"]["z"]["chi"] == 1.0)
    # tout le catalogue contre le module Flambement en service
    ecarts, n = [], 0
    for pr in CAT.profils():
        if pr["famille"] not in SE.FAMILLES or SE.FAMILLES[pr["famille"]][0] not in ("I",) + SE.TUBES:
            continue
        for nu in ("S235", "S355"):
            ref = FM.verifier({"nom": "x", "profil": pr["nom"], "nuance": nu, "N": 50.0, "L": 3.0, "gamma_M0": 1.0, "gamma_M1": 1.0,
                               "axes": {a: {"appuis": "AA", "k": 1.0, "Lcr": 3.0, "maintiens": 0} for a in ("y", "z")}})
            if ref["statut"] == FM.IMPOSSIBLE:
                continue
            n += 1
            r = v(pr["nom"], nu, 3000.0, NEd=50e3)
            F = r["grandeurs"]["flambement"]
            if not all(proche(F[a]["N_b_Rd"] / 1e3, ref["axes"][a]["NbRd"], 1e-9) and F[a]["courbe"] == ref["axes"][a]["courbe"] for a in ("y", "z")):
                ecarts.append((pr["nom"], nu, F["z"]["N_b_Rd"] / 1e3, ref["axes"]["z"]["NbRd"]))
    c(f"tout le catalogue, 3 m, S235 et S355 ({n} barres) : N_b,y,Rd, N_b,z,Rd et les courbes sont ceux du module Flambement EN SERVICE "
      "(classe 4 comprise : A_eff)", n > 1700 and not ecarts, (n, ecarts[:3]))
    s = SE.saisie("I", "soude_oxycoupe", h=400.0, b=300.0, tf=20.0, tw=10.0, r=0.0, A=15600.0, Iy=4.6e8, Iz=9.0e7, iy=171.7, iz=75.96, It=1.7e6, Iw=3.2e12)
    s2 = dict(s, tf=45.0)
    c("I soudé (tableau 6.2) : courbes b et c jusqu'à 40 mm de semelle, c et d au-delà ; caisson soudé : b et b",
      (FL.courbes(s, "S355", 355.0)["y"], FL.courbes(s, "S355", 355.0)["z"]) == ("b", "c") == tuple(M["ec3_flambement"].courbes(s, "S355", 355.0)[a] for a in "yz")
      and tuple(M["ec3_flambement"].courbes(s2, "S355", 355.0)[a] for a in "yz") == ("c", "d")
      and M["ec3_flambement"].courbes(SE.saisie("tube_rect", "caisson", h=300.0, b=200.0, t=8.0, A=7700.0, Iy=9e7, Iz=5e7, iy=108.0, iz=80.0), "S355", 355.0)["y"] == "b")
    # torsion : HEB 200 tenu latéralement tous les 2 m (axe faible) mais libre en torsion sur 8 m
    i02 = (p["iy"] * 10) ** 2 + (p["iz"] * 10) ** 2
    NcrT = (G_ * p["It"] * 1e4 + PI2 * E_ * p["Iw"] * 1e6 / 8000.0 ** 2) / i02
    lbT = math.sqrt(p["A"] * 100 * 235.0 / NcrT)
    r = v("HEB 200", "S235", NEd=800e3, axes={"y": {"L": 8000.0}, "z": {"L": 2000.0}}, torsion={"L": 8000.0})
    T = r["grandeurs"]["flambement"]["T"]
    c("flambement par torsion (HEB 200, axe faible tenu tous les 2 m, torsion libre sur 8 m) : N_cr,T = (G·I_t + π²·E·I_w/L_T²)/i₀² = "
      f"{NcrT / 1e3:.0f} kN < N_cr,z → vérification « E-Nb-T » avec la courbe de l'axe faible",
      proche(T["N_cr"], NcrT) and proche(limite(r, "E-Nb-T"), chi_de(lbT, 0.49) * p["A"] * 100 * 0.235) and T["gouverne"] and T["courbe"] == "c"
      and NcrT < r["grandeurs"]["flambement"]["z"]["N_cr"],
      (T["N_cr"], NcrT))
    r = v("HEB 200", "S235", 7200.0, NEd=432.72e3)
    c("même longueur pour l'axe faible et la torsion : N_cr,T > N_cr,z, la torsion ne peut pas gouverner et n'est pas rendue",
      verif(r, "E-Nb-T") is None and r["grandeurs"]["flambement"]["T"]["N_cr"] > r["grandeurs"]["flambement"]["z"]["N_cr"])

    # ---- 2. le moment critique
    L = 6000.0
    Iz, It, Iw = 603.8e4, 20.12e4, 125.9e9
    EIz, EIw, GIt = E_ * Iz, E_ * Iw, G_ * It

    def mcr(moment, **k):
        return mc.moment_critique(k.pop("L", L), EIz, EIw, GIt, moment, **k)["M_cr"]

    exact = math.pi / L * math.sqrt(EIz * GIt) * math.sqrt(1 + PI2 * EIw / (L ** 2 * GIt))
    c("R1, moment constant (IPE 300, 6 m, fourches) : le solveur redonne la formule fermée, 90,47 kN·m",
      proche(mcr(lambda x: 1e6), exact, 1e-7) and proche(mc.moment_constant(L, EIz, EIw, GIt), exact) and round(exact / 1e6, 2) == 90.47, mcr(lambda x: 1e6) / 1e6)

    def Mq(x):
        return x * (L - x) / 2.0

    def Mp(x):
        return (x if x <= L / 2 else L - x) / 2.0

    r1 = [round(mcr(Mq, reparties=[(0, L, 1.0, zg)]) / 1e6, 2) for zg in (0.0, 150.0, -150.0)] + \
         [round(mcr(Mp, ponctuelles=[(L / 2, 1.0, zg)]) / 1e6, 2) for zg in (0.0, 150.0, -150.0)]
    c("R1, charge répartie puis charge centrale, appliquées au centre de cisaillement, sur la semelle haute, sous la semelle basse : "
      "102,31 ; 78,83 ; 132,69 ; 123,12 ; 89,19 ; 168,95 kN·m (solveur indépendant numpy/scipy du deuxième avis)",
      r1 == [102.31, 78.83, 132.69, 123.12, 89.19, 168.95], r1)
    # (la mémoire des chargements proportionnels est VIDÉE entre les deux calculs : c'est le solveur qui est jugé ici)
    mc._GARDES.clear()
    fort = mcr(lambda x: 37.0 * Mq(x), reparties=[(0, L, 37.0, 150.0)])
    mc._GARDES.clear()
    c("le moment critique ne dépend pas de l'intensité de la charge (M_cr = λ_cr·|M|max) ni du maillage (16 ou 64 éléments : 10⁻⁴)",
      proche(fort, mcr(Mq, reparties=[(0, L, 1.0, 150.0)]), 1e-8)
      and proche(mcr(Mp, ponctuelles=[(L / 2, 1.0, 150.0)], elements=16), mcr(Mp, ponctuelles=[(L / 2, 1.0, 150.0)], elements=64), 1e-4))
    # la recherche du multiplicateur critique : le nombre de décompositions LDLᵀ, compté
    compte = [0]
    decomposer = mc._decomposition

    def espion(*a):
        compte[0] += 1
        return decomposer(*a)

    mc._decomposition = espion
    try:
        couts = []
        for k in ({"reparties": [(0, L, 1.0, 150.0)]}, {"reparties": [(0, L, 1.0, -150.0)]}, {"maintiens": [(L / 2, -144.65)]}):
            mc._GARDES.clear()
            compte[0] = 0
            mcr(Mq, **k)
            couts.append(compte[0])
        c("la sécante sur le déterminant trouve le multiplicateur critique à 10⁻¹⁰ en moins de 16 décompositions (la bissection "
          "seule en demande 35)", all(0 < n < 16 for n in couts), couts)
        compte[0] = 0
        encore = mcr(lambda x: 37.0 * Mq(x), reparties=[(0, L, 37.0, 150.0)], M_max=37.0 * L * L / 8.0)
        mc._GARDES.clear()
        frais = mcr(lambda x: 37.0 * Mq(x), reparties=[(0, L, 37.0, 150.0)], M_max=37.0 * L * L / 8.0)
        gratuit = compte[0]
        compte[0] = 0
        mcr(Mq, reparties=[(0, L, 1.0, 150.0)], M_max=L * L / 8.0)                       # la même forme, 37 fois plus faible
        c("deux chargements PROPORTIONNELS : le second ne coûte aucune décomposition et rend le M_cr du premier ; la mémoire "
          "vidée, le solveur redonne la même valeur", gratuit > 0 and compte[0] == 0 and proche(encore, frais, 1e-9)
          and proche(mcr(Mq, reparties=[(0, L, 1.0, 150.0)], M_max=L * L / 8.0), frais, 1e-9), (gratuit, compte[0]))
    finally:
        mc._decomposition = decomposer
    c1 = {psi: mcr(lambda x, psi=psi: 1e6 * (1 - (1 - psi) * x / L)) / exact for psi in (0.5, 0.0, -0.5, -1.0)}
    c("moment linéaire : C₁ = M_cr/M_cr,0 croît quand ψ décroît — 1,30 (ψ = 0,5) ; 1,75 à 1,90 (ψ = 0) ; plus de 2,2 (ψ = −0,5) ; 2,5 à 2,8 (ψ = −1)",
      1.28 < c1[0.5] < 1.34 and 1.75 < c1[0.0] < 1.90 and c1[-0.5] > 2.2 and 2.5 < c1[-1.0] < 2.8 and c1[0.5] < c1[0.0] < c1[-0.5],
      {k: round(x, 3) for k, x in c1.items()})
    Lc = 3000.0
    EIw_petit = (0.01 ** 2) * GIt * Lc ** 2 / PI2
    con = mc.moment_critique(Lc, EIz, EIw_petit, GIt, lambda x: -(Lc - x), ponctuelles=[(Lc, 1.0, 0.0)], appuis=("encastre", "libre"), elements=64)["M_cr"]
    c("console, charge en bout au centre de cisaillement, gauchissement presque nul : P_cr·L²/√(E·I_z·G·I_t) = 4,04 (4,013 sans gauchissement, "
      "valeur classique)", 4.013 < con * Lc / math.sqrt(EIz * GIt) < 4.06, con * Lc / math.sqrt(EIz * GIt))
    demi = math.pi / (L / 2) * math.sqrt(EIz * GIt) * math.sqrt(1 + PI2 * EIw / ((L / 2) ** 2 * GIt))
    haut = mcr(lambda x: 1e6, maintiens=[(L / 2, 144.65)])
    bas = mcr(lambda x: 1e6, maintiens=[(L / 2, -144.65)])
    c("moment constant, maintien d'UNE semelle à mi-portée : la semelle COMPRIMÉE vaut un maintien complet (formule fermée sur 3 m, 250,95 kN·m) ; "
      "la semelle tendue aide peu (105,85 kN·m contre 90,47 sans maintien)",
      proche(haut, demi, 1e-5) and round(demi / 1e6, 2) == 250.95 and exact < bas < 1.25 * exact and round(bas / 1e6, 2) == 105.85, (haut / 1e6, bas / 1e6))
    c("le signe du moment retourne la semelle comprimée : moment NÉGATIF et maintien de la semelle BASSE = moment positif et maintien de la haute",
      proche(mcr(lambda x: -1e6, maintiens=[(L / 2, -144.65)]), haut, 1e-8) and proche(mcr(lambda x: -1e6, maintiens=[(L / 2, 144.65)]), bas, 1e-8))
    serre = mc.moment_critique(L, EIz, EIw, GIt, lambda x: 1e6, maintiens=[(L * i / 12.0, 144.65) for i in range(1, 12)])["M_cr"]
    c("semelle comprimée tenue tous les 50 cm : le moment critique passe au-delà de 30 fois celui de la poutre nue (ou il n'y en a plus)",
      serre is None or serre > 30 * exact, None if serre is None else serre / exact)
    c("tronçon sans moment : aucun moment critique ; données absurdes refusées",
      mc.moment_critique(L, EIz, EIw, GIt, lambda x: 0.0)["M_cr"] is None and _leve(lambda: mc.moment_critique(-1.0, EIz, EIw, GIt, lambda x: 1.0))
      and _leve(lambda: mc.moment_critique(L, EIz, EIw, GIt, lambda x: 1.0, appuis=("fourche", "rotule"))))
    faux = []
    for nom in ("IPE 200", "IPE 400", "IPE 600", "HEA 200", "HEB 300", "HEM 300", "HEA 600", "IPN 300"):
        s = SE.du_catalogue(nom)
        for Lx in (2000.0, 5000.0, 10000.0):
            a_, b_, g_ = E_ * s["Iz"], E_ * s["Iw"], G_ * s["It"]
            if not proche(mc.moment_critique(Lx, a_, b_, g_, lambda x: 5e6)["M_cr"], mc.moment_constant(Lx, a_, b_, g_), 1e-6):
                faux.append((nom, Lx))
    c("8 profilés du catalogue, 2, 5 et 10 m, moment constant : le solveur redonne la formule fermée à 10⁻⁶", not faux, faux)

    # ---- 3. le déversement
    s = ipe300_du_brief()
    qEd, PEd = 3.267, 36.0e3

    def M23(x):
        return qEd * x * (L - x) / 2.0 + PEd * (x if x <= L / 2 else L - x) / 2.0

    def r3(zg, **k):
        tr = {"L": L, "moment": M23, "reparties": [(0.0, L, qEd, zg)], "ponctuelles": [(L / 2, PEd, zg)]}
        return v(s, "S235", L, MyEd=68.70e6, deversement={"troncon": tr}, **k)

    autre = {"lambda_LT0_sans_maintiens": 0.2, "beta_sans_maintiens": 1.0}
    lignes = []
    for zg in (0.0, 150.0):
        for par in ({}, autre):
            r = r3(zg, parametres=par)
            D = r["grandeurs"]["deversement"]
            lignes.append((round(D["M_cr"] / 1e6, 2), round(D["lambda_LT"], 3), round(D["chi_LT"], 4), round(limite(r, "E-Mb"), 2), round(taux(r, "E-Mb"), 3)))
    c("R3 (IPE 300, 6 m, charge répartie + charge centrale, M_Ed = 68,70 kN·m) — au centre de cisaillement : M_cr 118,18, λ̄_LT 1,118, "
      "χ_LT 0,6278 → M_b,Rd 92,71, taux 0,741 avec (0,4 ; 0,75) ; 0,5246 → 77,47, 0,887 avec (0,2 ; 1,0) — sur la semelle haute : 86,98, "
      "1,303, 0,5220 → 77,09, 0,891 ; 0,4254 → 62,83, 1,094 (diagramme hors tableau 6.6 : f = 1 ; le brief imprimait 77,48 et 1,093, au dernier "
      "chiffre près)",
      lignes == [(118.18, 1.118, 0.6278, 92.71, 0.741), (118.18, 1.118, 0.5246, 77.47, 0.887), (86.98, 1.303, 0.5220, 77.09, 0.891),
                 (86.98, 1.303, 0.4254, 62.83, 1.094)], lignes)
    D = r3(0.0)["grandeurs"]["deversement"]
    c("R3 : courbe b (laminé, h/b = 2 ≤ 2), couple (0,4 ; 0,75) nommé « à confirmer », k_c = 1 hors tableau, λ̄_0 du moment constant = "
      "√(W_pl,y·f_y/90,47) = 1,278",
      D["courbe"] == "b" and (D["lambda_LT0"], D["beta"]) == (0.4, 0.75) and D["a_confirmer"] == ["lambda_LT0_sans_maintiens", "beta_sans_maintiens"]
      and D["kc"] == 1.0 and not D["kc_tabule"] and round(D["lambda_0"], 3) == 1.278 and proche(D["M_cr0"], exact, 1e-9), D)
    # les quatre diagrammes du mode « efforts imposés »
    lb_kc = {}
    for diag, psi, kc_ref in (("constant", 1.0, 1.0), ("reparti", 1.0, 0.94), ("ponctuel", 1.0, 0.86), ("lineaire", 0.0, 1 / 1.33), ("lineaire", -1.0, 1 / 1.66)):
        r = v(s, "S235", L, MyEd=60e6, flexion={"y": {"diagramme": diag, "psi": psi}})
        D = r["grandeurs"]["deversement"]
        x = min(chi_de(D["lambda_LT"], 0.34, 0.4, 0.75), 1 / D["lambda_LT"] ** 2)
        f = min(1 - 0.5 * (1 - kc_ref) * (1 - 2 * (D["lambda_LT"] - 0.8) ** 2), 1.0)
        lb_kc[(diag, psi)] = (proche(D["kc"], kc_ref) and proche(D["f"], f) and proche(D["chi_LT"], min(x / f, 1.0, 1 / D["lambda_LT"] ** 2))
                              and proche(limite(r, "E-Mb"), D["chi_LT"] * 628.4 * 0.235), round(D["M_cr"] / 1e6, 2), round(D["f"], 4))
    c("tableau 6.6 — k_c : moment constant 1 ; charge répartie 0,94 ; charge concentrée 0,86 (et non 0,90 : écart E1 du brief) ; linéaire "
      "1/(1,33 − 0,33·ψ) ; f = 1 − 0,5·(1 − k_c)·[1 − 2·(λ̄_LT − 0,8)²] ≤ 1 ; χ_LT,mod = χ_LT/f ; M_cr des quatre diagrammes : 90,47 ; 102,31 ; "
      "123,12 kN·m…", all(x[0] for x in lb_kc.values()) and [lb_kc[k][1] for k in (("constant", 1.0), ("reparti", 1.0), ("ponctuel", 1.0))] == [90.47, 102.31, 123.12]
      and lb_kc[("lineaire", -1.0)][2] < lb_kc[("ponctuel", 1.0)][2] < lb_kc[("reparti", 1.0)][2] < 1.0, lb_kc)
    # le facteur que M_b,Rd et (6.61) emploient porte le NOM et la VALEUR de la ligne qui le donne
    def facteur(r):
        par = {f["cle"]: f for f in r["calcul"]}
        mb = par["M_b_Rd"]["termes"][0]
        ligne = next((par[k] for k in verif(r, "E-Mb")["formules"] if par[k]["symbole"] == mb["symbole"]), None)
        t661 = next((t for t in par["eq_661"]["termes"] if t["symbole"].startswith("χ_LT")), None) if "eq_661" in par else None
        lb = r["grandeurs"]["deversement"]["lambda_LT"]
        return {"nom": mb["symbole"], "valeur": mb["valeur"], "ligne": ligne and (ligne["gabarit"], ligne["resultat"]),
                "messages": list(verif(r, "E-Mb")["messages"]), "661": t661 and (t661["symbole"], t661["valeur"]),
                "critere": (verif(r, "E-661") or {}).get("critere", ""), "lb": lb, "chi": min(chi_de(lb, 0.34, 0.4, 0.75), 1 / lb ** 2)}

    def f_de(kc, lb):
        return min(1 - 0.5 * (1 - kc) * (1 - 2 * (lb - 0.8) ** 2), 1.0)

    fc = facteur(v(s, "S235", L, MyEd=60e6))                                                                    # moment constant : f = 1
    fm = facteur(v(s, "S235", L, MyEd=60e6, NEd=50e3, flexion={"y": {"diagramme": "reparti"}}))                 # χ_LT,mod = χ_LT/f
    fp = facteur(v(s, "S235", L, MyEd=60e6, deversement={"kc": 0.3}))                                           # plafond 1/λ̄_LT²
    fu = facteur(v(s, "S235", 2000.0, MyEd=140e6, flexion={"y": {"diagramme": "ponctuel"}}))                    # plafond 1
    mod = fm["chi"] / f_de(0.94, fm["lb"])
    c("le facteur écrit dans M_b,Rd porte le nom et la valeur de la ligne qui le donne : « χ_LT » quand f = 1 ; « χ_LT,mod » = χ_LT/f ; "
      "« χ_LT,mod » = 1/λ̄_LT² quand ce plafond gouverne ; « χ_LT,mod » = 1 avec son message quand χ_LT/f ≥ 1 — et (6.61) l'écrit du même nom",
      fc["nom"] == "χ_LT" and proche(fc["valeur"], fc["chi"]) and fc["ligne"][0] == "e_chi_LT" and proche(fc["ligne"][1], fc["chi"]) and not fc["messages"]
      and fm["nom"] == "χ_LT,mod" and proche(fm["valeur"], mod) and mod < 1 / fm["lb"] ** 2 and fm["ligne"][0] == "e_chi_LT_mod"
      and proche(fm["ligne"][1], mod) and fm["661"][0] == "χ_LT,mod" and proche(fm["661"][1], mod) and "χ_LT,mod·M_y,Rk" in fm["critere"]
      and fp["nom"] == "χ_LT,mod" and fp["chi"] / f_de(0.3, fp["lb"]) > 1 / fp["lb"] ** 2 and proche(fp["valeur"], 1 / fp["lb"] ** 2)
      and fp["ligne"][0] == "e_chi_LT_plafond" and proche(fp["ligne"][1], 1 / fp["lb"] ** 2) and not fp["messages"]
      and fu["nom"] == "χ_LT,mod" and fu["chi"] / f_de(0.86, fu["lb"]) > 1 and fu["valeur"] == 1.0 and fu["ligne"] is None
      and fu["messages"] == ["χ_LT/f ≥ 1 : χ_LT,mod = 1."], (fc, fm, fp, fu))
    D0 = v(s, "S235", L, MyEd=60e6, flexion={"y": {"diagramme": "lineaire", "psi": 0.0}})["grandeurs"]["deversement"]
    c("moment linéaire ψ = 0 par le chemin des efforts imposés : M_cr = C₁·M_cr,0 avec C₁ entre 1,75 et 1,90 ; λ̄_0 reste celui du moment constant",
      1.75 < D0["M_cr"] / exact < 1.90 and proche(D0["M_cr0"], exact), D0["M_cr"] / exact)
    r = v(s, "S235", L, MyEd=60e6, flexion={"y": {"diagramme": "reparti"}}, parametres={"gamma_M1": 1.1})
    D1 = r["grandeurs"]["deversement"]
    c("γ_M1 = 1,10 : M_b,Rd = χ_LT·W_y·f_y/1,1", proche(limite(r, "E-Mb"), D1["chi_LT"] * 628.4 * 0.235 / 1.1))
    r_h, r_c, r_b = (v(s, "S235", L, MyEd=60e6, flexion={"y": {"diagramme": "reparti"}}, deversement={"niveau": n_}) for n_ in ("haut", "centre", "bas"))
    c("le niveau d'application de la charge : semelle haute (z_g = +h/2) 78,83 kN·m < centre 102,31 < semelle basse 132,69 — et la résistance suit",
      [round(x["grandeurs"]["deversement"]["M_cr"] / 1e6, 2) for x in (r_h, r_c, r_b)] == [78.83, 102.31, 132.69]
      and limite(r_h, "E-Mb") < limite(r_c, "E-Mb") < limite(r_b, "E-Mb"))
    r = v(s, "S235", 12000.0, MyEd=20e6)
    D = r["grandeurs"]["deversement"]
    c("grand élancement (12 m, λ̄_LT = 1,9) : χ_LT plafonné à 1/λ̄_LT² (avec β = 0,75 la formule donnerait plus)",
      D["lambda_LT"] > 1.8 and proche(D["chi_LT"], 1 / D["lambda_LT"] ** 2) and chi_de(D["lambda_LT"], 0.34, 0.4, 0.75) > 1 / D["lambda_LT"] ** 2, D["chi_LT"])
    r = v(s, "S235", 1000.0, MyEd=100e6)
    D = r["grandeurs"]["deversement"]
    c("tronçon court (1 m, λ̄_LT ≤ 0,4) : déversement négligeable, χ_LT = 1, M_b,Rd = M_c,Rd — et c'est dit",
      D["cas"] == dv.NEGLIGE and D["chi_LT"] == 1.0 and proche(limite(r, "E-Mb"), 628.4 * 0.235) and verif(r, "E-Mb")["messages"])
    r1_, r2_ = v(s, "S235", L, MyEd=60e6, deversement={"Lc": 0.0}), v(s, "S235", L, MyEd=60e6, deversement={"actif": False})
    c("semelle comprimée tenue en continu (L_c = 0, ou déversement désactivé) : pas de vérification de déversement, et c'est dit",
      all(verif(x, "E-Mb") is None and x["grandeurs"]["deversement"]["cas"] == dv.EMPECHE and any("empêché" in h for h in x["hypotheses"]) for x in (r1_, r2_)))
    r = v(s, "S235", L, MyEd=60e6, deversement={"Lc": 2000.0})
    c("L_c = 2 m entre sections tenues (au lieu des 6 m de la barre) : M_cr est celui d'un tronçon de 2 m",
      proche(r["grandeurs"]["deversement"]["M_cr"], math.pi / 2000.0 * math.sqrt(EIz * GIt) * math.sqrt(1 + PI2 * EIw / (2000.0 ** 2 * GIt)), 1e-6))
    r = v(s, "S235", L, MyEd=60e6, deversement={"Mcr": 100e6})
    c("moment critique DONNÉ (100 kN·m) : il remplace le calcul, et c'est dit", proche(r["grandeurs"]["deversement"]["M_cr"], 100e6)
      and proche(r["grandeurs"]["deversement"]["lambda_LT"], math.sqrt(628.4e3 * 235 / 100e6)) and any("donné" in h for h in r["hypotheses"]))
    tr = {"L": L, "moment": lambda x: 60e6, "maintiens": [(L / 2, -144.65)]}
    r = v(s, "S235", L, MyEd=60e6, deversement={"troncon": tr})
    D = r["grandeurs"]["deversement"]
    c("un maintien d'une seule semelle DANS le tronçon : M_cr en tient compte (105,85 kN·m) et le couple devient (0,2 ; 1,0) — à confirmer",
      round(D["M_cr"] / 1e6, 2) == 105.85 and (D["lambda_LT0"], D["beta"]) == (0.2, 1.0) and D["a_confirmer"] == ["lambda_LT0_avec_maintiens", "beta_avec_maintiens"]
      and proche(D["chi_LT"], chi_de(D["lambda_LT"], 0.34, 0.2, 1.0)), D)
    courbes_LT = {n_: v(n_, "S235", 6000.0, MyEd=20e6)["grandeurs"]["deversement"]["courbe"] for n_ in ("IPE 300", "IPE 400", "HEA 300", "HEA 800")}
    ws = SE.saisie("I", "soude_oxycoupe", h=600.0, b=200.0, tf=15.0, tw=8.0, r=0.0, A=10560.0, Iy=6.0e8, Iz=2.0e7, iy=238.0, iz=43.5, It=5.5e5, Iw=1.7e12,
                   Wel_y=2.0e6, Wpl_y=2.3e6, Wel_z=2.0e5, Wpl_z=3.1e5)
    c("tableau 6.5 : laminé h/b ≤ 2 → b (IPE 300, HEA 300) ; > 2 → c (IPE 400, HEA 800) ; soudé h/b = 3 → d",
      courbes_LT == {"IPE 300": "b", "IPE 400": "c", "HEA 300": "b", "HEA 800": "c"}
      and v(ws, None, 6000.0, fy=355.0, MyEd=100e6)["grandeurs"]["deversement"]["courbe"] == "d", courbes_LT)
    r = v("HEA 300", "S355", 8000.0, MyEd=200e6)
    D = r["grandeurs"]["deversement"]
    c("HEA 300 S355 (classe 3) : λ̄_LT et M_b,Rd avec W_el,y", proche(D["lambda_LT"], math.sqrt(brut("HEA 300")["Wel_y"] * 1e3 * 355.0 / D["M_cr"]))
      and proche(limite(r, "E-Mb"), D["chi_LT"] * brut("HEA 300")["Wel_y"] * 0.355))
    r_c_, r_r_, r_t_ = v("SHS 200x200x6.3", "S355", 6000.0, MyEd=50e6), v("CHS 168.3x6.3", "S235", 6000.0, MyEd=10e6), v("RHS 300x100x8", "S355", 4000.0, MyEd=80e6)
    Dt = r_t_["grandeurs"]["deversement"]
    Mpl8 = brut("RHS 300x100x8")["Wpl_y"] * 1e3 * 355.0
    D8 = v("RHS 300x100x8", "S355", 8000.0, MyEd=0.9 * Mpl8)["grandeurs"]["deversement"]
    D8b = v("RHS 300x100x8", "S355", 8000.0, MyEd=0.5 * Mpl8)["grandeurs"]["deversement"]
    c("tube carré, tube rond : pas de déversement ; tube rectangulaire 300x100x8 : cas général (courbe d) — négligeable sur 4 m (λ̄_LT = 0,32 ≤ 0,4) ; "
      "sur 8 m (λ̄_LT = 0,46) : réduit sous 0,9·M_pl (χ_LT = 0,81 avec Φ_LT = 0,5·[1 + 0,76·(λ̄_LT − 0,2) + λ̄_LT²]), négligé sous 0,5·M_pl "
      "(M_Ed/M_cr ≤ 0,4²)",
      verif(r_c_, "E-Mb") is None and verif(r_r_, "E-Mb") is None and Dt["courbe"] == "d" and Dt["cas"] == dv.NEGLIGE and round(Dt["lambda_LT"], 2) == 0.32
      and D8["cas"] == dv.CALCULE and round(D8["lambda_LT"], 2) == 0.46 and proche(D8["chi_LT"], chi_de(D8["lambda_LT"], 0.76)) and round(D8["chi_LT"], 2) == 0.81
      and D8b["cas"] == dv.NEGLIGE and D8b["lambda_LT"] > 0.4, (Dt.get("lambda_LT"), D8.get("lambda_LT"), D8.get("chi_LT"), D8b.get("cas")))

    # ---- 4. la flexion composée avec instabilité (annexe A)
    def compare(nom, r, q, classe):
        att = annexe_A(q)
        A_ = r["grandeurs"]["annexe_A"]
        cles = ["mu_y", "mu_z", "a_LT", "C_my", "C_mz", "C_mLT", "k_yy", "k_yz", "k_zy", "k_zz"] + \
               (["w_y", "w_z", "b_LT", "c_LT", "d_LT", "e_LT", "C_yy", "C_yz", "C_zy", "C_zz"] if classe <= 2 else [])
        faux = [(k, A_[k], att[k]) for k in cles if not proche(A_[k], att[k], 1e-9)]
        if not (proche(taux(r, "E-661"), att["t61"], 1e-9) and proche(taux(r, "E-662"), att["t62"], 1e-9)):
            faux.append(("taux", taux(r, "E-661"), att["t61"], taux(r, "E-662"), att["t62"]))
        return faux

    bilan = {}
    # (a) IPE 500 S235, 3,75 m : compression, flexion déviée, déversement
    r = v("IPE 500", "S235", 3750.0, NEd=500e3, MyEd=200e6, MzEd=25e6, flexion={"y": {"diagramme": "reparti"}, "z": {"diagramme": "lineaire", "psi": 0.0}})
    q = barre_annexe("IPE 500", 235.0, 3750.0, 500e3, 200e6, 25e6, ("reparti", 1.0), ("lineaire", 0.0), (0.21, 0.34), 1,
                     section=r["grandeurs"]["deversement"]["M_cr"], alpha_LT=0.49, kc=0.94)
    bilan["IPE 500 S235, N + M_y (répartie) + M_z (triangulaire)"] = compare("a", r, q, 1)
    r_a, q_a = r, q
    # (b) HEA 300 S355, 5 m : classe 3, facteurs élastiques
    r = v("HEA 300", "S355", 5000.0, NEd=500e3, MyEd=150e6, MzEd=30e6)
    q = barre_annexe("HEA 300", 355.0, 5000.0, 500e3, 150e6, 30e6, ("constant", 1.0), ("constant", 1.0), (0.34, 0.49), 3,
                     section=r["grandeurs"]["deversement"]["M_cr"], alpha_LT=0.34, kc=1.0)
    bilan["HEA 300 S355 (classe 3)"] = compare("b", r, q, 3)
    # (c) tube carré : pas de déversement, pas de torsion
    r = v("SHS 200x200x6.3", "S355", 4000.0, NEd=600e3, MyEd=50e6, MzEd=30e6, flexion={"y": {"diagramme": "lineaire", "psi": -0.5}, "z": {"diagramme": "ponctuel"}})
    q = barre_annexe("SHS 200x200x6.3", 355.0, 4000.0, 600e3, 50e6, 30e6, ("lineaire", -0.5), ("ponctuel", 1.0), (0.21, 0.21), 2, torsion=False)
    bilan["SHS 200x200x6,3 S355"] = compare("c", r, q, 2)
    # (d) et (e) : un seul moment
    r = v("IPE 300", "S235", 4000.0, NEd=200e3, MyEd=50e6, flexion={"y": {"diagramme": "ponctuel"}})
    q = barre_annexe("IPE 300", 235.0, 4000.0, 200e3, 50e6, 0.0, ("ponctuel", 1.0), ("constant", 1.0), (0.21, 0.34), 1,
                     section=r["grandeurs"]["deversement"]["M_cr"], alpha_LT=0.34, kc=0.86)
    bilan["IPE 300 S235, N + M_y"] = compare("d", r, q, 1)
    r = v("IPE 300", "S235", 4000.0, NEd=200e3, MzEd=10e6, flexion={"z": {"diagramme": "reparti"}})
    q = barre_annexe("IPE 300", 235.0, 4000.0, 200e3, 0.0, 10e6, ("constant", 1.0), ("reparti", 1.0), (0.21, 0.34), 1)
    bilan["IPE 300 S235, N + M_z"] = compare("e", r, q, 1)
    # (f) flexion déviée sans effort normal
    r = v("IPE 300", "S235", 4000.0, MyEd=60e6, MzEd=8e6, flexion={"y": {"diagramme": "ponctuel"}})
    q = barre_annexe("IPE 300", 235.0, 4000.0, 0.0, 60e6, 8e6, ("ponctuel", 1.0), ("constant", 1.0), (0.21, 0.34), 1,
                     section=r["grandeurs"]["deversement"]["M_cr"], alpha_LT=0.34, kc=0.86)
    bilan["IPE 300 S235, M_y + M_z sans effort normal"] = compare("f", r, q, 1)
    # (g) HEB 200 élancé sous forte compression : des planchers de C_ij gouvernent
    r = v("HEB 200", "S235", 7200.0, NEd=550e3, MyEd=10e6, MzEd=3e6)
    q = barre_annexe("HEB 200", 235.0, 7200.0, 550e3, 10e6, 3e6, ("constant", 1.0), ("constant", 1.0), (0.34, 0.49), 1,
                     section=r["grandeurs"]["deversement"]["M_cr"], alpha_LT=0.34, kc=1.0)
    bilan["HEB 200 S235, 7,20 m"] = compare("g", r, q, 1)
    planchers = r["grandeurs"]["annexe_A"]["planchers"]
    # (h) et (i) : deux tronçons courts dont λ̄_0 tombe de part et d'autre de la limite 0,2·√C₁·[…]^¼ (charge concentrée : C₁ = 1/0,86²)
    pi3 = brut("IPE 300")
    a3, b3, g3 = E_ * pi3["Iz"] * 1e4, E_ * pi3["Iw"] * 1e6, G_ * pi3["It"] * 1e4
    cotes = {}
    for cible in (0.225, 0.25):
        lo, hi = 300.0, 4000.0
        for _ in range(60):                                   # la longueur qui donne λ̄_0 = cible
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if math.sqrt(pi3["Wpl_y"] * 1e3 * 235.0 / mc.moment_constant(mid, a3, b3, g3)) < cible else (lo, mid)
        Lx = 0.5 * (lo + hi)
        r = v("IPE 300", "S235", Lx, NEd=100e3, MyEd=80e6, MzEd=5e6, flexion={"y": {"diagramme": "ponctuel"}})
        q = barre_annexe("IPE 300", 235.0, Lx, 100e3, 80e6, 5e6, ("ponctuel", 1.0), ("constant", 1.0), (0.21, 0.34), 1,
                         section=r["grandeurs"]["deversement"]["M_cr"], alpha_LT=0.34, kc=0.86)
        bilan[f"IPE 300 court, λ̄_0 = {cible}"] = compare("h", r, q, 1)
        cotes[cible] = (r["grandeurs"]["annexe_A"]["lambda_0"] > r["grandeurs"]["annexe_A"]["lambda_0_lim"], r["grandeurs"]["annexe_A"]["C_mLT"])
    c("annexe A recalculée ici d'une traite, sur les valeurs brutes du catalogue, pour neuf barres (I de classe 1 avec et sans déversement, "
      "classe 3, tube, un seul moment, flexion déviée sans effort normal, tronçons courts) : μ, w, a_LT, C_my, C_mz, C_mLT, b_LT à e_LT, C_yy à C_zz, "
      "k_yy à k_zz et les deux taux — égaux à 10⁻⁹", not any(bilan.values()), {k: x[:2] for k, x in bilan.items() if x})
    c("HEB 200 de 7,20 m sous forte compression : un plancher de C_ij gouverne (le cas existe dans les barres recalculées)", bool(planchers), planchers)
    c("la limite λ̄_0 ≤ 0,2·√C₁·[(1 − N/N_cr,z)·(1 − N/N_cr,T)]^¼, avec C₁ = 1/k_c² = 1,352 (charge concentrée) : λ̄_0 = 0,225 est en deçà "
      "(C_mLT = 1), λ̄_0 = 0,25 au-delà (C_mLT > 1)", cotes[0.225] == (False, 1.0) and cotes[0.25][0] and cotes[0.25][1] > 1.0, cotes)
    # le benchmark 21 de l'aide de SCIA Engineer (ECCS N°119, exemple 5) — sans la borne de C_mi,0 (l'EN seule)
    r = v("IPE 500", "S235", 3750.0, NEd=500e3, MyEd=200e6, MzEd=25e6, flexion={"y": {"diagramme": "reparti"}, "z": {"diagramme": "lineaire", "psi": 0.0}},
          parametres={"borne_Cmi0": "aucune"})
    A_, D = r["grandeurs"]["annexe_A"], r["grandeurs"]["deversement"]
    ref = {"mu_y": (A_["mu_y"], 1.00, 0.005), "w_y": (A_["w_y"], 1.138, 0.001), "mu_z": (A_["mu_z"], 0.937, 0.001), "w_z": (A_["w_z"], 1.5, 0.0),
           "C_mz,0": (A_["C_mz0"], 0.771, 0.001), "M_cr,0": (D["M_cr0"] / 1e6, 899.0, 0.002), "lambda_0": (D["lambda_0"], 0.757, 0.001),
           "a_LT": (A_["a_LT"], 0.998, 0.001)}
    ecarts = {k: (round(x, 4), y) for k, (x, y, tol) in ref.items() if abs(x / y - 1) > tol + 1e-12}
    c("exemple 5 de l'ECCS N°119 (benchmark 21 de l'aide de SCIA Engineer : IPE 500 S235, 3,75 m, N_Ed = 500 kN, M_z triangulaire) — les valeurs "
      "de référence qui ne dépendent pas du diagramme de M_y : μ_y 1,00 ; w_y 1,138 ; μ_z 0,937 ; w_z 1,5 ; C_mz,0 0,771 ; M_cr,0 899 kN·m ; "
      "λ̄_0 0,757 ; a_LT 0,998 — à moins de 0,2 %", not ecarts, ecarts)
    c("la borne inférieure de C_mi,0 (annexe belge, supposée 1 − N_Ed/N_cr,i, à confirmer) est active par défaut : C_mz,0 = 0,842 au lieu de 0,771",
      round(r_a["grandeurs"]["annexe_A"]["C_mz0"], 3) == 0.842 and proche(r_a["grandeurs"]["annexe_A"]["C_mz0"], 1 - 500e3 / q_a["Ncrz"])
      and {x["cle"] for x in r_a["grandeurs"]["a_confirmer"]} == {"lambda_LT0_sans_maintiens", "beta_sans_maintiens", "methode_interaction", "borne_Cmi0"})
    # tableau A.2 : la formule générale redonne les deux cas chargés (écart E2 du brief)
    ph = brut("HEB 200")
    EI = E_ * ph["Iy"] * 1e4
    cas = {}
    for nom, delta, Mmax in (("reparti", 5 * 10.0 * 6000.0 ** 4 / (384 * EI), 10.0 * 6000.0 ** 2 / 8), ("ponctuel", 1e4 * 6000.0 ** 3 / (48 * EI), 1e4 * 6000.0 / 4)):
        rg = v("HEB 200", "S235", 6000.0, NEd=400e3, MyEd=Mmax, flexion={"y": {"diagramme": "general", "delta": delta, "L": 6000.0}}, deversement={"actif": False})
        rt = v("HEB 200", "S235", 6000.0, NEd=400e3, MyEd=Mmax, flexion={"y": {"diagramme": nom}}, deversement={"actif": False})
        cas[nom] = (rg["grandeurs"]["annexe_A"]["C_my0"], rt["grandeurs"]["annexe_A"]["C_my0"])
    ry = 400e3 / (PI2 * EI / 6000.0 ** 2)
    c("tableau A.2 : la formule générale 1 + (π²·E·I·|δ|/(L²·|M|) − 1)·N_Ed/N_cr redonne, avec la flèche et le moment exacts, la charge répartie "
      "(1 + 0,03·N/N_cr) et la charge concentrée (1 − 0,18·N/N_cr) à 0,2 % — les deux cas que le brief avait échangés (écart E2)",
      proche(cas["reparti"][1], 1 + 0.03 * ry) and proche(cas["ponctuel"][1], 1 - 0.18 * ry) and abs(cas["reparti"][0] / cas["reparti"][1] - 1) < 0.002
      and abs(cas["ponctuel"][0] / cas["ponctuel"][1] - 1) < 0.002 and cas["reparti"][1] > 1 > cas["ponctuel"][1], cas)
    r = v("IPE 300", "S235", 6000.0, NEd=360e3, MyEd=5e6)
    c("N_Ed au-delà de la charge critique de l'axe faible (IPE 300, 6 m, 360 kN > N_cr,z = 348 kN) : flambement refusé, flexion composée refusée SANS taux",
      verif(r, "E-Nb-z")["statut"] == RE.NOK and verif(r, "E-661")["statut"] == RE.NOK and verif(r, "E-661")["taux"] is None and r["statut"] == RE.NOK)
    r = v("IPE 300", "S235", 4000.0, NEd=-200e3, MyEd=50e6)
    c("traction et flexion : pas de (6.61) ni (6.62) ; le déversement reste vérifié sans le bénéfice de la traction",
      verif(r, "E-661") is None and verif(r, "E-Mb") is not None)
    rt, r0 = (v("IPE 300", "S235", 4000.0, NEd=N_, MyEd=60e6, MzEd=8e6, flexion={"y": {"diagramme": "ponctuel"}}) for N_ in (-200e3, 0.0))
    c("traction et flexion DÉVIÉE : la barre est vérifiée comme si elle n'était pas tendue — (6.61) et (6.62) de la flexion déviée seule",
      taux(rt, "E-661") is not None and proche(taux(rt, "E-661"), taux(r0, "E-661")) and proche(taux(rt, "E-662"), taux(r0, "E-662")))
    r = v("IPE 600", "S355", 2000.0, NEd=4000e3, MyEd=150e6)
    A_ = r["grandeurs"]["annexe_A"]
    c("IPE 600 S355 de classe 4 (2 m, N = 4 000 kN) : N_Rk = A_eff·f_y (5 150 kN), facteurs élastiques, λ̄ calculés avec A_eff",
      r["grandeurs"]["classe"] == 4 and round(A_["N_Rk"] / 1e3) == 5150 and not A_["plastique"]
      and proche(r["grandeurs"]["flambement"]["z"]["lambda_bar"], math.sqrt(A_["N_Rk"] / r["grandeurs"]["flambement"]["z"]["N_cr"])), A_["N_Rk"])
    refus = [k for k, b_ in (("u nul", {"axes": {"y": {"L": 3000.0, "u": 0.0}}}), ("L négatif", {"axes": {"y": {"L": -1.0}}}),
                             ("courbe inconnue", {"axes": {"y": {"L": 3000.0, "courbe": "e"}}}), ("diagramme inconnu", {"flexion": {"y": {"diagramme": "sinus"}}}),
                             ("ψ hors de [−1 ; 1]", {"flexion": {"y": {"diagramme": "lineaire", "psi": 1.5}}}), ("L_c négatif", {"deversement": {"Lc": -1.0}}),
                             ("niveau inconnu", {"deversement": {"niveau": "milieu"}}), ("M_cr nul", {"deversement": {"Mcr": 0.0}}),
                             ("tronçon sans moment", {"deversement": {"troncon": {"L": 3000.0}}}))
             if not _refuse(ec, {"section": SE.du_catalogue("IPE 300"), "nuance": "S235", "NEd": 1e4, "MyEd": 1e6, **b_})]
    c("domaine : u nul, L négatif, courbe ou diagramme inconnus, ψ hors de [−1 ; 1], L_c négatif, niveau inconnu, M_cr nul, tronçon sans moment — "
      "refusés avec un message", not refus, refus)
    return C


def _leve(f):
    try:
        f()
    except ValueError:
        return True
    return False


def _refuse(ec, barre):
    try:
        ec.verifier(barre)
    except RE.Refus as e:
        return bool(str(e))
    return False


# ------------------------------------------------------------------ 5. les formules
def cas_formules():
    cat = SE.du_catalogue
    L = 6000.0
    return [
        {"section": cat("HEB 200"), "nuance": "S235", "NEd": 432.72e3, "axes": {"y": {"L": 7200.0}, "z": {"L": 7200.0}}},
        {"section": cat("HEB 200"), "nuance": "S235", "NEd": 800e3, "axes": {"y": {"L": 8000.0}, "z": {"L": 2000.0}}, "torsion": {"L": 8000.0}},
        {"section": cat("HEB 200"), "nuance": "S235", "NEd": 100e3, "axes": {"y": {"L": 500.0}, "z": {"L": 500.0}}},
        {"section": cat("IPE 500"), "nuance": "S235", "NEd": 500e3, "MyEd": 200e6, "MzEd": 25e6, "axes": {"y": {"L": 3750.0}, "z": {"L": 3750.0}},
         "flexion": {"y": {"diagramme": "reparti"}, "z": {"diagramme": "lineaire", "psi": 0.0}}},
        {"section": cat("IPE 500"), "nuance": "S235", "NEd": 500e3, "MyEd": 200e6, "MzEd": 25e6, "axes": {"y": {"L": 3750.0}, "z": {"L": 3750.0}},
         "flexion": {"y": {"diagramme": "ponctuel"}, "z": {"diagramme": "ponctuel"}}, "parametres": {"borne_Cmi0": "aucune"}},
        {"section": cat("HEA 300"), "nuance": "S355", "NEd": 500e3, "MyEd": 150e6, "MzEd": 30e6, "axes": {"y": {"L": 5000.0}, "z": {"L": 5000.0}}},
        {"section": cat("SHS 200x200x6.3"), "nuance": "S355", "NEd": 600e3, "MyEd": 50e6, "MzEd": 30e6, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}},
         "flexion": {"y": {"diagramme": "lineaire", "psi": -0.5}, "z": {"diagramme": "ponctuel"}}},
        {"section": cat("IPE 300"), "nuance": "S235", "NEd": 200e3, "MyEd": 50e6, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}},
         "flexion": {"y": {"diagramme": "lineaire", "psi": 0.3}}},
        {"section": cat("IPE 300"), "nuance": "S235", "NEd": 200e3, "MzEd": 10e6, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}},
         "flexion": {"z": {"diagramme": "reparti"}}},
        {"section": cat("IPE 300"), "nuance": "S235", "MyEd": 60e6, "MzEd": 8e6, "axes": {"y": {"L": 4000.0}, "z": {"L": 4000.0}},
         "flexion": {"y": {"diagramme": "ponctuel"}, "z": {"diagramme": "lineaire", "psi": -1.0}}},
        {"section": cat("IPE 300"), "nuance": "S235", "MyEd": 20e6, "axes": {"y": {"L": 12000.0}, "z": {"L": 12000.0}}},
        {"section": cat("HEB 200"), "nuance": "S235", "NEd": 350e3, "MyEd": 30e6, "MzEd": 10e6, "axes": {"y": {"L": 7200.0}, "z": {"L": 7200.0}}},
        {"section": cat("HEB 200"), "nuance": "S235", "NEd": 400e3, "MyEd": 45e6, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}},
         "flexion": {"y": {"diagramme": "general", "delta": 12.0, "L": 6000.0}}, "deversement": {"actif": False}},
        {"section": cat("IPE 600"), "nuance": "S355", "NEd": 4000e3, "MyEd": 150e6, "axes": {"y": {"L": 2000.0}, "z": {"L": 2000.0}}},
        {"section": cat("RHS 300x100x8"), "nuance": "S355", "NEd": 300e3, "MyEd": 80e6, "axes": {"y": {"L": 8000.0}, "z": {"L": 8000.0}}},
        {"section": cat("RHS 400x200x8") if "RHS 400x200x8" in CAT.par_nom() else cat("RHS 300x100x8"), "nuance": "S355", "MyEd": 150e6,
         "axes": {"y": {"L": 30000.0}, "z": {"L": 30000.0}}},
        {"section": ipe300_du_brief(), "nuance": "S235", "MyEd": 60e6, "axes": {"y": {"L": L}, "z": {"L": L}},
         "deversement": {"troncon": {"L": L, "moment": lambda x: 60e6, "maintiens": [(L / 2, -144.65)]}}},
    ]


def charger(mutation=None):
    """Les cinq modules de la barre, chargés depuis leur TEXTE (une mutation appliquée à l'un d'eux) et liés entre eux."""
    mods = {}
    for k in FICHIERS:
        src = open(CHEMINS[k], encoding="utf-8").read()
        if mutation and mutation[0] == k:
            if src.count(mutation[1]) != 1:
                return None
            src = src.replace(mutation[1], mutation[2])
        spec = importlib.util.spec_from_file_location(f"acier.barre.{k}_mute", CHEMINS[k])
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "acier.barre"
        exec(compile(src, CHEMINS[k], "exec"), mod.__dict__)
        mods[k] = mod
    mods["ec3_deversement"].MC = mods["mcr"]
    mods["ec3_interaction"].DV = mods["ec3_deversement"]
    e = mods["ec3"]
    e.FL, e.DV, e.IN, e.MC = mods["ec3_flambement"], mods["ec3_deversement"], mods["ec3_interaction"], mods["mcr"]
    return mods


MUTATIONS = [
    ("mcr", "couplage flexion-torsion : signe retourné (la semelle « comprimée » devient l'autre)", "c = -wt * M * ddN[a] * N[b]", "c = wt * M * ddN[a] * N[b]"),
    ("mcr", "hauteur d'application d'une charge répartie : signe retourné", "Kg[ip[a]][ip[b]] -= wt * qz * N[a] * N[b]", "Kg[ip[a]][ip[b]] += wt * qz * N[a] * N[b]"),
    ("mcr", "hauteur d'application d'une charge ponctuelle oubliée", "Kg[4 * noeud(x) + 2][4 * noeud(x) + 2] -= P * zg", "Kg[4 * noeud(x) + 2][4 * noeud(x) + 2] -= 0.0"),
    ("mcr", "torsion de Saint-Venant oubliée", "K[ip[a]][ip[b]] += wt * (EIw * ddN[a] * ddN[b] + GIt * dN[a] * dN[b])", "K[ip[a]][ip[b]] += wt * (EIw * ddN[a] * ddN[b])"),
    ("mcr", "gauchissement oublié", "K[ip[a]][ip[b]] += wt * (EIw * ddN[a] * ddN[b] + GIt * dN[a] * dN[b])", "K[ip[a]][ip[b]] += wt * (1e-6 * EIw * ddN[a] * ddN[b] + GIt * dN[a] * dN[b])"),
    ("mcr", "fourche : la torsion n'est pas tenue", "            fixes |= {4 * i, 4 * i + 2}\n", "            fixes |= {4 * i}\n"),
    ("mcr", "maintien d'une semelle : cote z retournée", "    for x, z in maintiens:  ", "    for x, z in [(x_, -z_) for x_, z_ in maintiens]:  "),
    ("mcr", "M_cr rendu sans |M|max", 'out.update({"M_cr": lam * M_max, "lambda": lam})', 'out.update({"M_cr": lam, "lambda": lam})'),
    ("mcr", "formule fermée : π² oublié devant E·I_w", "math.sqrt(1.0 + math.pi ** 2 * EIw / (L ** 2 * GIt))", "math.sqrt(1.0 + EIw / (L ** 2 * GIt))"),
    ("mcr", "bissection grossière (tolérance 10 %)", "TOLERANCE = 1e-10 ", "TOLERANCE = 1e-1 "),
    ("mcr", "la sécante retirée (bissection seule : 35 décompositions)", "        if f_haut is not None and pas < SECANTES:", "        if False:"),
    ("mcr", "mémoire : la hauteur d'application des charges réparties oubliée dans la signature",
     "tuple(round(qz * L / M_max, CHIFFRES) for _, qz in gauss),", "(),"),
    ("mcr", "mémoire : la forme du diagramme oubliée dans la signature", "tuple(round(M / M_max, CHIFFRES) for M, _ in gauss),", "(),"),
    ("mcr", "mémoire : la cote des maintiens oubliée dans la signature", "tuple((round(x, 6), z) for x, z in maintiens),", "(),"),
    ("mcr", "mémoire : la hauteur d'application des charges ponctuelles oubliée dans la signature",
     "tuple((noeud(x), round(P * zg / M_max, CHIFFRES)) for x, P, zg in ponctuelles))", "())"),
    ("ec3_flambement", "N_cr avec L au lieu de u·L", "Ncr = math.pi ** 2 * E * inertie / Lcr ** 2", 'Ncr = math.pi ** 2 * E * inertie / ax["L"] ** 2'),
    ("ec3_flambement", "λ̄ avec A au lieu de A_eff (classe 4)", "        lb = math.sqrt(A_N * fy / Ncr)\n        Phi, chi = NO.phi_chi(lb, alpha)\n        NbRd",
     '        lb = math.sqrt(s["A"] * fy / Ncr)\n        Phi, chi = NO.phi_chi(lb, alpha)\n        NbRd'),
    ("ec3_flambement", "N_b,Rd avec γ_M0", "        NbRd = chi * A_N * fy / gM1\n        g.update(", '        NbRd = chi * A_N * fy / P["gamma_M0"]\n        g.update('),
    ("ec3_flambement", "I soudé : courbes du laminé", 'cy, cz = ("b", "c") if mince else ("c", "d")', 'cy, cz = ("a", "b") if mince else ("b", "c")'),
    ("ec3_flambement", "torsion : gauchissement oublié", 'NcrT = (Gc * s["It"] + math.pi ** 2 * E * s["Iw"] / LT ** 2) / i0 ** 2', 'NcrT = (Gc * s["It"]) / i0 ** 2'),
    ("ec3_flambement", "torsion : i₀ = i_y", 'i0 = math.sqrt(s["iy"] ** 2 + s["iz"] ** 2)', 'i0 = s["iy"]'),
    ("ec3_flambement", "torsion rendue quand elle ne gouverne pas, cachée quand elle gouverne", 'gouverne = F["z"]["N_cr"] is None or NcrT < F["z"]["N_cr"]',
     'gouverne = F["z"]["N_cr"] is None or NcrT > F["z"]["N_cr"]'),
    ("ec3_flambement", "torsion : courbe de l'axe fort", '        courbe = F["z"]["courbe"]\n', '        courbe = F["y"]["courbe"]\n'),
    ("ec3_flambement", "flambement vérifié aussi en traction", "        if comprime:\n            J.verif(f\"E-Nb-{a}\"", "        if True:\n            J.verif(f\"E-Nb-{a}\""),
    ("ec3_deversement", "k_c de la charge concentrée : 0,90 (l'erreur du brief)", "KC_REPARTI, KC_PONCTUEL = 0.94, 0.86", "KC_REPARTI, KC_PONCTUEL = 0.94, 0.90"),
    ("ec3_deversement", "k_c de la charge répartie : 0,90", "KC_REPARTI, KC_PONCTUEL = 0.94, 0.86", "KC_REPARTI, KC_PONCTUEL = 0.90, 0.86"),
    ("ec3_deversement", "k_c linéaire : signe de ψ", "return 1.0 / (1.33 - 0.33 * psi), True", "return 1.0 / (1.33 + 0.33 * psi), True"),
    ("ec3_deversement", "tableau 6.5 : limite h/b à 1,2", "ELANCE = 2.0 ", "ELANCE = 1.2 "),
    ("ec3_deversement", "tableau 6.5 : courbes du cas général (a, b)", 'COURBES_LT = {"lamine": ("b", "c"), "soude": ("c", "d")}', 'COURBES_LT = {"lamine": ("a", "b"), "soude": ("c", "d")}'),
    ("ec3_deversement", "I soudé : courbes du laminé", 'COURBES_LT = {"lamine": ("b", "c"), "soude": ("c", "d")}', 'COURBES_LT = {"lamine": ("b", "c"), "soude": ("b", "c")}'),
    ("ec3_deversement", "Φ_LT sans β", "Phi = 0.5 * (1.0 + alpha * (lbLT - lbLT0) + beta * lbLT ** 2)", "Phi = 0.5 * (1.0 + alpha * (lbLT - lbLT0) + lbLT ** 2)"),
    ("ec3_deversement", "Φ_LT avec 0,2 au lieu de λ̄_LT,0", "Phi = 0.5 * (1.0 + alpha * (lbLT - lbLT0) + beta * lbLT ** 2)", "Phi = 0.5 * (1.0 + alpha * (lbLT - 0.2) + beta * lbLT ** 2)"),
    ("ec3_deversement", "plafond 1/λ̄_LT² oublié", "plafond = min(1.0, 1.0 / lbLT ** 2)", "plafond = 1.0"),
    ("ec3_deversement", "f : sans le carré", "f = min(1.0 - 0.5 * (1.0 - kc) * (1.0 - 2.0 * (lbLT - 0.8) ** 2), 1.0)", "f = min(1.0 - 0.5 * (1.0 - kc) * (1.0 - 2.0 * (lbLT - 0.8)), 1.0)"),
    ("ec3_deversement", "χ_LT,mod = χ_LT·f", "chi = min(chi0 / f, plafond)", "chi = min(chi0 * f, plafond)"),
    ("ec3_deversement", "M_b,Rd écrit « χ_LT » alors que sa valeur est χ_LT,mod", '            nom_chi = "χ_LT,mod"\n', '            nom_chi = "χ_LT"\n'),
    ("ec3_deversement", "la ligne du plafond 1/λ̄_LT² de χ_LT,mod oubliée", "            elif plafond < 1.0:\n", "            elif False:\n"),
    ("ec3_deversement", "χ_LT,mod plafonné à 1 sans le dire", '                messages.append("χ_LT/f ≥ 1 : χ_LT,mod = 1.")\n', "                pass\n"),
    ("ec3_deversement", "le nom du facteur non transmis à la flexion composée", '"chi_LT": chi, "symbole_chi": nom_chi})', '"chi_LT": chi})'),
    ("ec3_interaction", "(6.61) écrit « χ_LT » alors que sa valeur est χ_LT,mod", 'nom_chi = D.get("symbole_chi", "χ_LT")', 'nom_chi = "χ_LT"'),
    ("ec3_deversement", "déversement jamais négligé", "if lbLT <= seuil or My / Mcr <= seuil ** 2:", "if False:"),
    ("ec3_deversement", "couples (λ̄_LT,0 ; β) échangés", 'lbLT0 = P["lambda_LT0_avec_maintiens" if avec else "lambda_LT0_sans_maintiens"]',
     'lbLT0 = P["lambda_LT0_sans_maintiens" if avec else "lambda_LT0_avec_maintiens"]'),
    ("ec3_deversement", "λ̄_LT avec W_el,y quelle que soit la classe", "    lbLT = math.sqrt(Wy * fy / Mcr)\n", '    lbLT = math.sqrt(s["Wel_y"] * fy / Mcr)\n'),
    ("ec3_deversement", "λ̄_0 pris sur le diagramme réel (et non sur le moment constant)", "    lb0 = math.sqrt(Wy * fy / Mcr0)\n", "    lb0 = math.sqrt(Wy * fy / Mcr)\n"),
    ("ec3_deversement", "niveau d'application de la charge répartie ignoré", 'charges = {"reparties": [(0.0, L, 8.0 * M / L ** 2, zg)]}', 'charges = {"reparties": [(0.0, L, 8.0 * M / L ** 2, 0.0)]}'),
    ("ec3_deversement", "charge répartie de moitié (q = 4M/L²)", 'charges = {"reparties": [(0.0, L, 8.0 * M / L ** 2, zg)]}', 'charges = {"reparties": [(0.0, L, 4.0 * M / L ** 2, zg)]}'),
    ("ec3_deversement", "diagramme linéaire pris constant", "            return M * (1.0 - (1.0 - psi) * x / L)\n", "            return M\n"),
    ("ec3_deversement", "tube carré vérifié au déversement comme un I", 'if not My > 0 or s["forme"] not in ("I", "tube_rect"):', 'if not My > 0 or s["forme"] not in ("I", "tube_rect", "tube_carre"):'),
    ("ec3_deversement", "C₁ = 1/k_c", '"C1": 1.0 / kc ** 2', '"C1": 1.0 / kc'),
    ("ec3_deversement", "M_b,Rd avec γ_M0 = 1 en dur", "    MbRd = chi * Wy * fy / gM1\n", "    MbRd = chi * Wy * fy\n"),
    ("ec3_interaction", "charge répartie : 1 − 0,18 (les deux cas échangés, l'erreur du brief)", "        c = 1.0 + 0.03 * r\n", "        c = 1.0 - 0.18 * r\n"),
    ("ec3_interaction", "charge concentrée : 1 + 0,03", "        c = 1.0 - 0.18 * r\n", "        c = 1.0 + 0.03 * r\n"),
    ("ec3_interaction", "moment linéaire : ψ + 0,33", 'c = 0.79 + 0.21 * fl["psi"] + 0.36 * (fl["psi"] - 0.33) * r', 'c = 0.79 + 0.21 * fl["psi"] + 0.36 * (fl["psi"] + 0.33) * r'),
    ("ec3_interaction", "borne inférieure de C_mi,0 retirée", "    if borne and c < 1.0 - r:", "    if False:"),
    ("ec3_interaction", "formule générale : sans π²", 'c = 1.0 + (math.pi ** 2 * E * inertie * abs(fl["delta"]) / (fl["L"] ** 2 * M) - 1.0) * r',
     'c = 1.0 + (E * inertie * abs(fl["delta"]) / (fl["L"] ** 2 * M) - 1.0) * r'),
    ("ec3_interaction", "μ_y = 1", "    muy = (1.0 - ry) / (1.0 - chiy * ry)\n", "    muy = 1.0\n"),
    ("ec3_interaction", "μ_z avec χ_y", "    muz = (1.0 - rz) / (1.0 - chiz * rz)\n", "    muz = (1.0 - rz) / (1.0 - chiy * rz)\n"),
    ("ec3_interaction", "plafond de w porté à 2", "W_MAX = 1.5 ", "W_MAX = 2.0 "),
    ("ec3_interaction", "a_LT = 1 − I_t/I_z", 'aLT = max(1.0 - s["It"] / s["Iy"], 0.0) if s.get("It") else 0.0', 'aLT = max(1.0 - s["It"] / s["Iz"], 0.0) if s.get("It") else 0.0'),
    ("ec3_interaction", "limite de λ̄_0 : C₁ au lieu de √C₁", "lim = 0.2 * math.sqrt(C1) * ((1.0 - rz) * (1.0 - rT)) ** 0.25", "lim = 0.2 * C1 * ((1.0 - rz) * (1.0 - rT)) ** 0.25"),
    ("ec3_interaction", "limite de λ̄_0 jamais franchie (C_mLT = 1 toujours)", "    susceptible = lb0 > lim and My > 0\n", "    susceptible = False\n"),
    ("ec3_interaction", "C_my : sans le dénominateur", "Cmy = Cmy0 + (1.0 - Cmy0) * racine / (1.0 + racine)", "Cmy = Cmy0 + (1.0 - Cmy0) * racine"),
    ("ec3_interaction", "C_mLT sans le plancher 1", "        CmLT = max(brut, 1.0)\n", "        CmLT = brut\n"),
    ("ec3_interaction", "b_LT : coefficient 1 au lieu de 0,5", "bLT = 0.5 * aLT * lb0 ** 2 * ty * tz", "bLT = 1.0 * aLT * lb0 ** 2 * ty * tz"),
    ("ec3_interaction", "c_LT : λ̄_z² au lieu de λ̄_z⁴", "cLT = 10.0 * aLT * lb0 ** 2 / (5.0 + lbz ** 4) * ty / Cmy", "cLT = 10.0 * aLT * lb0 ** 2 / (5.0 + lbz ** 2) * ty / Cmy"),
    ("ec3_interaction", "d_LT : coefficient 1 au lieu de 2", "dLT = 2.0 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy * tz / Cmz", "dLT = 1.0 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy * tz / Cmz"),
    ("ec3_interaction", "e_LT : coefficient 1 au lieu de 1,7", "eLT = 1.7 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy", "eLT = 1.0 * aLT * lb0 / (0.1 + lbz ** 4) * ty / Cmy"),
    ("ec3_interaction", "C_yy : λ̄_max au lieu de λ̄_max² dans le second terme", "Cyy = 1.0 + (wy - 1.0) * ((2.0 - 1.6 / wy * Cmy ** 2 * lbmax - 1.6 / wy * Cmy ** 2 * lbmax ** 2) * npl - bLT)",
     "Cyy = 1.0 + (wy - 1.0) * ((2.0 - 1.6 / wy * Cmy ** 2 * lbmax - 1.6 / wy * Cmy ** 2 * lbmax) * npl - bLT)"),
    ("ec3_interaction", "C_yz : w_z⁴ au lieu de w_z⁵", "Cyz = 1.0 + (wz - 1.0) * ((2.0 - 14.0 * Cmz ** 2 * lbmax ** 2 / wz ** 5) * npl - cLT)",
     "Cyz = 1.0 + (wz - 1.0) * ((2.0 - 14.0 * Cmz ** 2 * lbmax ** 2 / wz ** 4) * npl - cLT)"),
    ("ec3_interaction", "C_zy : 1,6 au lieu de 14", "Czy = 1.0 + (wy - 1.0) * ((2.0 - 14.0 * Cmy ** 2 * lbmax ** 2 / wy ** 5) * npl - dLT)",
     "Czy = 1.0 + (wy - 1.0) * ((2.0 - 1.6 * Cmy ** 2 * lbmax ** 2 / wy ** 5) * npl - dLT)"),
    ("ec3_interaction", "planchers des C_ij retirés", 'Cyy, Cyz, Czy, Czz = (max(bruts[k], planchers[k]) for k in ("C_yy", "C_yz", "C_zy", "C_zz"))',
     'Cyy, Cyz, Czy, Czz = (bruts[k] for k in ("C_yy", "C_yz", "C_zy", "C_zz"))'),
    ("ec3_interaction", "k_yz sans le facteur 0,6", "kyz = Cmz * muy / r_(rz) / Cyz * 0.6 * math.sqrt(wz / wy)", "kyz = Cmz * muy / r_(rz) / Cyz * math.sqrt(wz / wy)"),
    ("ec3_interaction", "k_zy avec N_cr,z", "kzy = Cmy * CmLT * muz / r_(ry) / Czy * 0.6 * math.sqrt(wy / wz)", "kzy = Cmy * CmLT * muz / r_(rz) / Czy * 0.6 * math.sqrt(wy / wz)"),
    ("ec3_interaction", "k_zz avec μ_y", "kzz = Cmz * muz / r_(rz) / Czz", "kzz = Cmz * muy / r_(rz) / Czz"),
    ("ec3_interaction", "classe 3 traitée en plastique", "    plastique = classe <= 2\n", "    plastique = classe <= 3\n"),
    ("ec3_interaction", "(6.62) avec χ_y", '("E-662", "6.62", chiz, "z", kzy, kzz)', '("E-662", "6.62", chiy, "z", kzy, kzz)'),
    ("ec3_interaction", "χ_LT oublié dans les deux expressions", "taux += ky * My / (chiLT * MyRk / gM1)", "taux += ky * My / (MyRk / gM1)"),
    ("ec3_interaction", "N_Ed ≥ N_cr non arrêté", "    if ry >= 1.0 or rz >= 1.0 or rT >= 1.0:", "    if False:"),
    ("ec3_interaction", "traction comptée comme une compression négative dans (6.61)", '    N = max(d["NEd"], 0.0)  ', '    N = d["NEd"]  '),
    ("ec3", "torsion : longueur de l'axe fort par défaut", 'LT, uT = _nombre(t.get("L", d["axes"]["z"]["L"]), "torsion.L")', 'LT, uT = _nombre(t.get("L", d["axes"]["y"]["L"]), "torsion.L")'),
    ("ec3", "déversement : L_c par défaut = longueur de l'axe fort", '    Lc = d["axes"]["z"]["L"] if Lc is None else _nombre(Lc, "deversement.Lc")', '    Lc = d["axes"]["y"]["L"] if Lc is None else _nombre(Lc, "deversement.Lc")'),
    ("ec3", "niveau « haut » sous le centre de cisaillement", 'NIVEAUX = {"centre": 0.0, "haut": 1.0, "bas": -1.0}', 'NIVEAUX = {"centre": 0.0, "haut": -1.0, "bas": 1.0}'),
    ("ec3", "diagramme par défaut : charge répartie (et non le moment constant, du côté sûr)", 'diag = f.get("diagramme", DV.CONSTANT)', 'diag = f.get("diagramme", DV.REPARTI)'),
]


def main():
    print("== 1 à 4. le flambement, le moment critique, le déversement, l'annexe A")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 5. les formules")
    faux, n, utilises = [], 0, set()
    for b in cas_formules():
        r = EC.verifier(b)
        cles = [f["cle"] for f in r["calcul"]]
        if len(cles) != len(set(cles)):
            faux.append((b["section"]["nom"], "clé répétée", sorted({k for k in cles if cles.count(k) > 1})[:3]))
        for f in r["calcul"]:
            n += 1
            utilises.add(f["gabarit"])
            if f["gabarit"] not in FO.GABARITS:
                faux.append((b["section"]["nom"], f["cle"], "gabarit absent"))
            elif not proche(FO.evaluer(f), f["resultat"]):
                faux.append((b["section"]["nom"], f["cle"], FO.evaluer(f), f["resultat"]))
        for v in r["verifications"]:
            manquent = [k for k in v["formules"] if k not in set(cles)]
            if manquent:
                faux.append((b["section"]["nom"], v["id"], "formules absentes", manquent))
    chk(f"chaque bloc de formule du moteur se recalcule par son gabarit, sur ses seuls termes, à 10⁻⁹ ({n} formules, {len(cas_formules())} barres) ; "
        "clés uniques ; chaque vérification nomme des formules qui existent", n > 400 and not faux, faux[:4])
    morts = set(FO.GABARITS_EC3_BARRE) - utilises
    chk("chaque gabarit de la barre (Eurocode 3) sert à un contrôle (aucun n'est mort)", not morts, sorted(morts))
    chk("les tables de gabarits ne se marchent pas dessus (clés distinctes)",
        not set(FO.GABARITS_EC3_BARRE) & set(FO.GABARITS_EC3_SECTION) and not set(FO.GABARITS_EC3) & set(FO.GABARITS_HISTORIQUE))

    print("== 6. le catalogue, la sortie")
    plantes, n, faux = [], 0, []
    i_ = t_ = 0
    for p in CAT.profils():
        if p["famille"] not in SE.FAMILLES:
            continue
        s = SE.du_catalogue(p["nom"])
        if s["forme"] == "I":
            i_ += 1
            if i_ % 2:
                continue
        elif s["forme"] in SE.TUBES:
            t_ += 1
            if t_ % 4:
                continue
        else:
            continue
        for nu, L, eff in (("S235", 3000.0, {"NEd": 0.15 * s["A"] * 235, "MyEd": 0.25 * s["Wel_y"] * 235, "MzEd": 0.15 * s["Wel_z"] * 235}),
                           ("S355", 6000.0, {"NEd": 0.05 * s["A"] * 235, "MyEd": 0.3 * s["Wel_y"] * 235}),
                           ("S355", 4000.0, {"MyEd": 0.3 * s["Wel_y"] * 235, "MzEd": 0.1 * s["Wel_z"] * 235})):
            try:
                r = EC.verifier({"section": s, "nuance": nu, "axes": {"y": {"L": L}, "z": {"L": L}},
                                 "flexion": {"y": {"diagramme": "reparti"}}, "deversement": {"niveau": "haut"}, **eff})
            except RE.Refus:
                continue
            except Exception as e:  # noqa: BLE001
                plantes.append((p["nom"], nu, type(e).__name__, str(e)[:80]))
                continue
            n += 1
            json.dumps(r, allow_nan=False)
            for x in r["verifications"]:
                if x["taux"] is not None and not (x["taux"] >= 0 and math.isfinite(x["taux"])):
                    faux.append((p["nom"], nu, x["id"], x["taux"]))
            g = r["grandeurs"]
            if g.get("couvert"):
                F = g.get("flambement") or {}
                for a in ("y", "z"):
                    if a in F and not (0 < F[a]["chi"] <= 1 and F[a]["N_b_Rd"] <= g["A_N"] * g["fy"] * (1 + 1e-12)):
                        faux.append((p["nom"], nu, "χ ou N_b,Rd", a))
                D = g.get("deversement") or {}
                if D.get("M_b_Rd") is not None and not (0 < D["chi_LT"] <= 1 and D["M_b_Rd"] <= g["W_y"] * g["fy"] * (1 + 1e-12)):
                    faux.append((p["nom"], nu, "χ_LT ou M_b,Rd"))
    chk(f"un I ou H sur deux et un tube sur quatre, trois jeux d'efforts ({n} barres) : aucun plantage, JSON strict, taux finis et positifs, "
        "χ ≤ 1, N_b,Rd ≤ A·f_y, M_b,Rd ≤ W_y·f_y", n > 800 and not plantes and not faux, (n, plantes[:3], faux[:3]))
    b0 = cas_formules()[3]
    b1 = copy.deepcopy(b0)
    EC.verifier(b1)
    chk("verifier ne modifie pas la barre reçue", b1 == b0)

    print("== 7. mutations")
    for fichier, nom, ancien, nouveau in MUTATIONS:
        mods = charger((fichier, ancien, nouveau))
        if mods is None:
            chk(f"mutation « {nom} » : le texte à muter est dans le moteur, une seule fois", False, "texte absent ou répété")
            continue
        try:
            rouges = [n_ for n_, ok, _ in controles(mods) if not ok]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
            rouges = [f"exception {type(e).__name__}: {e}"]
        chk(f"mutation « {nom} » : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")

    print("== 8. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab', 'PIL', 'fitz'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from acier.barre import ec3 as E, section as S\n"
            "r = E.verifier({'section': S.du_catalogue('IPE 300'), 'nuance': 'S235', 'NEd': 200e3, 'MyEd': 50e6, 'axes': {'y': {'L': 4000.0}, 'z': {'L': 4000.0}},\n"
            "                'flexion': {'y': {'diagramme': 'ponctuel'}}})\n"
            "print(r['statut'], r['gouvernant'], sorted(m for m in sys.modules if m.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab')))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le moteur se charge et calcule (M_cr compris) sans streamlit, pandas, matplotlib, scipy, reportlab (refusés ; numpy ne sert qu'au catalogue)",
        p.returncode == 0 and p.stdout.strip().endswith("[]") and p.stdout.split()[0] in ("ok", "nok"), (p.stdout.strip(), p.stderr[-300:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

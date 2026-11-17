# -*- coding: utf-8 -*-
"""Barres en acier — les PROFILÉS RECONSTITUÉS SOUDÉS (PRS) en I, de bout en bout côté Eurocode 3 (acier/barre/prs.py
et les règles PRS de ec3_classe, ec3_section, ec3_flambement, ec3_deversement, ec3_interaction, mcr — terme de Wagner).
La spécification : brief v12 § 18 (sauf 18.6 et 18.7), § 7.1 à 7.5 ; docs/acier/BARRES_ACIER.md § 5.10.

Les références :
  1. PRS-A (bisymétrique) et PRS-B (monosymétrique) du brief § 11.8, chaque valeur à la tolérance du brief :
     propriétés, classification, cisaillement, flambement, M_cr (solveur, avec le terme de Wagner ; (E.1) de l'annexe E
     ANB), M_b,Rd, statuts.
  2. « Autres tests PRS » 1 à 5 (gorge non saisie, section soudée équivalente, bords non saisis, bibliothèque, I_t).
  3. § 11.5, tests 22 (η = 1,20 partout pour un PRS) et 25 (f_y et ε plat par plat).
  4. LA SECTION EFFICACE (classe 4) recalculée AUTREMENT : un PRS à semelles égales sans gorge redonne celle d'un I saisi
     (l'autre écriture du module) ; un PRS monosymétrique, par assemblage des parties efficaces, ici.
  5. LE MOTEUR : un PRS par ses plats, par son nom en bibliothèque ; barre chargée (charges et maintiens repérés depuis
     le centre de cisaillement) ; les laminés inchangés.
  6. FORMULES — chaque bloc des PRS se recalcule par son gabarit à 10⁻⁹ ; aucun gabarit PRS mort.
  7. MUTATIONS — chaque altération plausible du texte du moteur (prs.py, mcr.py, les règles PRS) fait rougir un
     contrôle des § 1 à 5. Le module muté est remis dans le paquet, ceux qui l'emploient rechargés, puis tout est rendu.
  8. PURETÉ — prs.py se charge et calcule sans numpy, scipy, pandas, streamlit.
  9, 10. Les reprises (contre-vérifications) : le sens de la charge en efforts imposés (M_y,Ed < 0 : le plus petit M_cr des
     deux sens, recoupé par un Ritz indépendant), le Catalogue (un PRS monosymétrique écarté seul ; I_t, I_w et A_v
     d'un PRS pris ici), le PRS-C du § 11.9 (Eurocode 3 du propriétaire, 7 / 8 / 9 maintiens).

    /usr/bin/python3.12 tests/test_barre_prs.py
"""
import importlib
import json
import math
import os
import re
import subprocess
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import acier.barre as PAQUET  # noqa: E402
from acier.barre import ec3 as _ec3  # noqa: E402,F401 — chargés une fois : la mutation les recharge
from acier.barre import formules as FO  # noqa: E402
from acier.barre import moteur as _moteur  # noqa: E402,F401
from acier.profiles import consultation as CO  # noqa: E402
import api_web.catalogue as _catalogue  # noqa: E402,F401 — l'autre lecteur des profilés personnalisés (une mutation le remplace)

OK, KO = [], []
DOSSIER = os.path.join(RACINE, "acier", "barre")
E_, G_ = 210000.0, 81000.0
L10 = 10000.0


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def m(nom):
    """Le module du paquet tel qu'il est MAINTENANT (une mutation le remplace) : jamais une référence gardée."""
    return sys.modules[f"acier.barre.{nom}"]


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def a_pres(a, b, ecart):
    """|a − b| ≤ ecart (la tolérance du brief : la dernière décimale qu'il imprime, à un demi près)."""
    return a is not None and abs(a - b) <= ecart


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


def prs_a(**k):
    return m("prs").section_prs(300.0, 30.0, 540.0, 15.0, 300.0, 30.0, **{"a": 6.0, "nuance": "S355", "bords": "oxycoupes", **k})


def prs_b(**k):
    return m("prs").section_prs(300.0, 30.0, 540.0, 15.0, 200.0, 30.0, **{"a": 6.0, "nuance": "S355", "bords": "oxycoupes", **k})


def mcr(s, moment, **k):
    """M_cr (kN·m) du solveur sur 10 m entre fourches, avec le z_j de la section."""
    r = m("mcr").moment_critique(L10, E_ * s["Iz"], E_ * s["Iw"], G_ * s["It"], moment, zj=k.pop("zj", m("prs").zj_haut(s)), **k)
    return r["M_cr"] / 1e6


def reparti(s, zg):
    q = 1.0
    return mcr(s, lambda x: q * x * (L10 - x) / 2.0, reparties=[(0.0, L10, q, zg)])


def barre_dev(s, niveau="haut", **k):
    """La barre de 10 m, charge répartie au niveau donné (efforts imposés, ec3.verifier)."""
    return m("ec3").verifier({"section": s, "nuance": "S355", "MyEd": k.pop("MyEd", 500e6), "axes": {"y": {"L": L10}, "z": {"L": L10}},
                              "flexion": {"y": {"diagramme": k.pop("diagramme", "reparti")}}, "deversement": {"niveau": niveau}, **k})


# ------------------------------------------------------------------ la section efficace recalculée autrement (§ 4)
def efficace_assemblee(p, sv, fy_parois, semelle_haut_comprimee=True):
    """W_eff,y d'un PRS sous M_y seul, par assemblage des parties EFFICACES (rectangles) : semelle comprimée réduite aux
    bouts (console, k_σ = 0,43), puis l'âme avec le ψ de cette section (tableau 4.1) — écrit ici sans le moteur."""
    b1, t1, hw, tw, b2, t2 = (p[k] for k in ("b1", "t1", "hw", "tw", "b2", "t2"))
    h = t1 + hw + t2
    eps = {k: math.sqrt(235.0 / v) for k, v in fy_parois.items()}
    parts = [(b2, 0.0, t2), (tw, t2, t2 + hw), (b1, t2 + hw, h)]
    # les parties retirées, en rectangles [(largeur, z1, z2)] de largeur négative
    if semelle_haut_comprimee:
        bf, tf, zf, ef = b1, t1, (t2 + hw, h), eps["semelle_haut"]
    else:
        bf, tf, zf, ef = b2, t2, (0.0, t2), eps["semelle_bas"]
    c = (bf - tw) / 2.0 - sv
    lp = (c / tf) / (28.4 * ef * math.sqrt(0.43))
    rho = 1.0 if lp <= 0.748 else min((lp - 0.188) / lp ** 2, 1.0)
    trous = [(-2.0 * (1.0 - rho) * c, zf[0], zf[1])] if rho < 1.0 else []

    def props(rs):
        A = sum(w * (z2 - z1) for w, z1, z2 in rs)
        zc = sum(w * (z2 - z1) * (z1 + z2) / 2.0 for w, z1, z2 in rs) / A
        return A, zc, sum(w * (z2 - z1) ** 3 / 12.0 + w * (z2 - z1) * ((z1 + z2) / 2.0 - zc) ** 2 for w, z1, z2 in rs)

    _, z1, _ = props(parts + trous)
    zb, zt = t2 + sv, t2 + hw - sv
    cw = zt - zb
    if semelle_haut_comprimee:
        sc, so, sens = zt - z1, zb - z1, -1.0
        bord = zt
    else:
        sc, so, sens = z1 - zb, z1 - zt, 1.0
        bord = zb
    psi = so / sc
    ks = 7.81 - 6.29 * psi + 9.78 * psi ** 2 if -1 < psi < 0 else 23.9 if psi == -1 else 5.98 * (1 - psi) ** 2 if psi < -1 else 8.2 / (1.05 + psi)
    lpw = (cw / tw) / (28.4 * eps["ame"] * math.sqrt(ks))
    rw = 1.0 if lpw <= 0.5 + math.sqrt(0.085 - 0.055 * psi) else min((lpw - 0.055 * (3 + psi)) / lpw ** 2, 1.0)
    bc = sc if psi < 0 else cw
    beff = rw * bc
    be1 = 0.4 * beff if psi < 0 else 2.0 / (5.0 - psi) * beff
    if bc - beff > 0:
        u, v = sorted((bord + sens * be1, bord + sens * (be1 + bc - beff)))
        trous.append((-tw, u, v))
    A, zc, inertie = props(parts + trous)
    _, zg0, I0 = props(parts)
    # (plafonné au W_el,y de la section brute, la plus petite des deux fibres : la convention du module, du côté sûr)
    return min(inertie / max(h - zc, zc), I0 / max(h - zg0, zg0)), A, rho, rw


def aire_assemblee(p, sv, fy_parois):
    """A_eff et e_N d'un PRS sous compression seule, par assemblage des parties efficaces."""
    b1, t1, hw, tw, b2, t2 = (p[k] for k in ("b1", "t1", "hw", "tw", "b2", "t2"))
    h = t1 + hw + t2
    eps = {k: math.sqrt(235.0 / v) for k, v in fy_parois.items()}
    parts = [(b2, 0.0, t2), (tw, t2, t2 + hw), (b1, t2 + hw, h)]
    A0 = sum(w * (z2 - z1) for w, z1, z2 in parts)
    zg = sum(w * (z2 - z1) * (z1 + z2) / 2.0 for w, z1, z2 in parts) / A0
    trous = []
    for bf, tf, z1, z2, e in ((b1, t1, t2 + hw, h, eps["semelle_haut"]), (b2, t2, 0.0, t2, eps["semelle_bas"])):
        c = (bf - tw) / 2.0 - sv
        lp = (c / tf) / (28.4 * e * 0.6557438524302)
        rho = 1.0 if lp <= 0.748 else min((lp - 0.188) / lp ** 2, 1.0)
        if rho < 1:
            trous.append((-2.0 * (1.0 - rho) * c, z1, z2))
    cw = hw - 2.0 * sv
    lpw = (cw / tw) / (28.4 * eps["ame"] * 2.0)
    rw = 1.0 if lpw <= 0.673 else min((lpw - 0.22) / lpw ** 2, 1.0)
    if rw < 1:
        mi = t2 + sv + cw / 2.0
        trous.append((-tw, mi - (1.0 - rw) * cw / 2.0, mi + (1.0 - rw) * cw / 2.0))
    rs = parts + trous
    A = sum(w * (z2 - z1) for w, z1, z2 in rs)
    zc = sum(w * (z2 - z1) * (z1 + z2) / 2.0 for w, z1, z2 in rs) / A
    return A, zc - zg


# ------------------------------------------------------------------ 1 à 5. les contrôles
def controles():
    """[(nom, ok, info)] — sur les modules du paquet tels qu'ils sont (une mutation en remplace un)."""
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    PR, ES, CL, MC, AM = m("prs"), m("ec3_section"), m("ec3_classe"), m("mcr"), m("ec3_ame")
    A, B = prs_a(), prs_b()

    # ---- PRS-A : matériau, propriétés
    d = ES.lire({"section": A, "nuance": "S355"})
    fp = d["fy_parois"]
    c("PRS-A, matériau (brief § 7.1, 18.3) : semelles de 30 mm f_y = 345 (ε = 0,8253), âme de 15 mm f_y = 355 (ε = 0,8136) ; f_y de calcul 345",
      d["fy"] == 345.0 and fp["semelle_haut"] == fp["semelle_bas"] == 345.0 and fp["ame"] == 355.0
      and round(math.sqrt(235 / fp["semelle_haut"]), 4) == 0.8253 and round(math.sqrt(235 / fp["ame"]), 4) == 0.8136, (d["fy"], fp))
    c("PRS-A, propriétés (§ 11.8) : A = 26 100 mm² ; I_y = 1 660,23·10⁶ ; I_z = 135,152·10⁶ mm⁴ ; i_y = 252,21 ; i_z = 71,96 mm ; W_el,y = 5 534,1·10³ ; "
      "W_pl,y = 6 223,5·10³ ; W_el,z = 901,0·10³ ; W_pl,z = 1 380,4·10³ mm³",
      round(A["A"], 6) == 26100.0 and round(A["Iy"] / 1e6, 2) == 1660.23 and round(A["Iz"] / 1e6, 3) == 135.152 and round(A["iy"], 2) == 252.21
      and round(A["iz"], 2) == 71.96 and round(A["Wel_y"] / 1e3, 1) == 5534.1 and round(A["Wpl_y"] / 1e3, 1) == 6223.5
      and round(A["Wel_z"] / 1e3, 1) == 901.0 and round(A["Wpl_z"] / 1e3, 1) == 1380.4 and A["Avz"] == 9720.0 and A["Avy"] == 18000.0 and A["eta"] == 1.2,
      {k: A[k] for k in ("A", "Iy", "Iz", "iy", "iz", "Wel_y", "Wpl_y", "Wel_z", "Wpl_z", "Avz", "Avy")})
    c("PRS-A : I_w = 1,0965·10¹³ mm⁶ (18.4) ; (E.3) : 1,0978·10¹³ (+0,11 %), pour information",
      round(A["Iw"] / 1e13, 4) == 1.0965 and round(PR.iw_annexe_e(A) / 1e13, 4) == 1.0978 and round((PR.iw_annexe_e(A) / A["Iw"] - 1) * 100, 2) == 0.11,
      (A["Iw"], PR.iw_annexe_e(A)))
    st = sum(b * t ** 3 / 3 for b, t in ((300, 30), (540, 15), (300, 30)))
    semelles = 2 * 300 * 30 ** 3 / 3 * (1 - 0.63 * 30 / 300) + 540 * 15 ** 3 / 3
    c("PRS-A : I_t = 5,657·10⁶ mm⁴ (18.4 : (b·t³/3)·(1 − 0,63·t/b) sur les TROIS plats) — ni Σ b·t³/3 (6,008·10⁶), ni la correction des seules semelles (5,667·10⁶)",
      round(A["It"] / 1e6, 3) == 5.657 and round(st / 1e6, 3) == 6.008 and round(semelles / 1e6, 3) == 5.667 and not proche(A["It"], semelles, 1e-4),
      A["It"])
    # ---- PRS-A : classification, cisaillement, flambement
    cl = CL.classer(A, 345.0, 0.0, 1.0, 0.0, fp)
    par = {p["role"]: p for p in cl["parois"]}
    c("PRS-A, classe (§ 11.8) : s = 8,485 ; âme c = 523,03 mm, c/t = 34,87 ≤ 72·0,8136 = 58,58 ; semelle c = 134,01 mm, c/t = 4,467 ≤ 9·0,8253 = 7,43 : classe 1",
      round(A["s"], 3) == 8.485 and round(par["ame"]["c"], 2) == 523.03 and round(par["ame"]["c_t"], 2) == 34.87
      and round(par["ame"]["limites"][0], 2) == 58.58 and round(par["semelle_haut"]["c"], 2) == 134.01 and round(par["semelle_haut"]["c_t"], 3) == 4.467
      and round(par["semelle_haut"]["limites"][0], 2) == 7.43 and cl["classe"] == 1, {k: (p["c"], p["c_t"], p["limites"]) for k, p in par.items()})
    r = ES.verifier({"section": A, "nuance": "S355", "VzEd": 100e3, "VyEd": 100e3})
    el = AM.elancement(A, 345.0, m("parametres").jeu(), "z", fy_ame=355.0)
    c("PRS-A, cisaillement : A_v,z = 1,2·540·15 = 9 720 mm², V_pl,z,Rd = 1 936,1 kN ; A_v,y = A − h_w·t_w = 18 000 mm², V_pl,y,Rd = 3 585,3 kN ; "
      "h_w/t_w = 36,0 ≤ 72·0,8136/1,2 = 48,82 : pas de voilement par cisaillement",
      proche(r["grandeurs"]["A_v"]["z"], 9720.0) and round(verif(r, "E-Vz")["limite"], 1) == 1936.1 and proche(r["grandeurs"]["A_v"]["y"], 18000.0)
      and round(verif(r, "E-Vy")["limite"], 1) == 3585.3 and el is not None and round(el["rapport"], 1) == 36.0 and round(el["limite"], 2) == 48.82
      and not el["elancee"] and el["eta"] == 1.2, (r["grandeurs"].get("A_v"), el))
    fl = m("ec3_flambement").courbes(A, "S355", 345.0)
    r = m("ec3").verifier({"section": A, "nuance": "S355", "NEd": -1000e3, "axes": {"y": {"L": 0.0}, "z": {"L": 0.0}}})
    rc = ES.verifier({"section": A, "nuance": "S355", "NEd": 1000e3})
    c("PRS-A, flambement : t_f = 30 ≤ 40 mm → y : b, z : c (tableau 6.2, I soudé) ; N_pl,Rd = A·f_y = 9 004,5 kN (en compression seule, l'âme passe en "
      "classe 4 : c/t = 34,87 > 42·ε = 34,17 — A_eff)",
      (fl["y"], fl["z"]) == ("b", "c") and round(verif(r, "E-N")["limite"], 1) == 9004.5 and rc["grandeurs"]["classe"] == 4
      and verif(rc, "E-N")["limite"] < 9004.5, (fl, verif(r, "E-N"), rc["grandeurs"]["classe"]))
    # ---- PRS-A : M_cr et déversement
    m0 = mcr(A, lambda x: 1e6)
    ferme = MC.moment_constant(L10, E_ * A["Iz"], E_ * A["Iw"], G_ * A["It"]) / 1e6
    vals = [reparti(A, 0.0), reparti(A, 300.0), reparti(A, -300.0)]
    c("PRS-A, M_cr : moment constant 1 385,68 kN·m (formule exacte et solveur) ; charge répartie, solveur : centre de cisaillement 1 567,12 ; "
      "semelle haute 1 191,42 ; semelle basse 2 059,63 (à 0,01 kN·m)",
      a_pres(m0, 1385.68, 0.01) and a_pres(ferme, 1385.68, 0.01) and all(a_pres(v, w, 0.01) for v, w in zip(vals, (1567.12, 1191.42, 2059.63))),
      (m0, ferme, vals))
    e1 = [MC.annexe_e(L10, E_ * A["Iz"], E_ * A["Iw"], G_ * A["It"], 1.12, 0.45, 0.525, zg, 0.0) / 1e6 for zg in (0.0, 300.0, -300.0)]
    c("PRS-A, annexe E ANB (E.8), C₁ = 1,12, C₂ = 0,45 : 1 551,96 (−0,97 %) ; 1 185,17 (−0,52 %) ; 2 032,25 (−1,33 %)",
      all(a_pres(v, w, 0.01) for v, w in zip(e1, (1551.96, 1185.17, 2032.25)))
      and [round((v / w - 1) * 100, 2) for v, w in zip(e1, vals)] == [-0.97, -0.52, -1.33], e1)
    r = barre_dev(A)
    D, v = r["grandeurs"]["deversement"], verif(r, "E-Mb")
    c("PRS-A, déversement, charge sur la semelle haute (§ 11.8) : M_y,Rk = 2 147,11 kN·m ; λ̄_LT = 1,3424 ; § 6.3.2.2, courbe c (h/b = 2,0) : χ_LT = 0,3714, "
      "M_b,Rd = 797,39 kN·m ; § 6.3.2.3 affiché pour information : f = 0,9877, χ_LT,mod = 0,3760, M_b,Rd = 807,36 kN·m ; contrôle (E.1) 1 185,2 kN·m",
      v is not None and round(D["W_y"] * 345.0 / 1e6, 2) == 2147.11 and round(D["lambda_LT"], 4) == 1.3424 and D["courbe"] == "c"
      and D["methode"] == "§ 6.3.2.2" and round(D["chi_LT"], 4) == 0.3714 and a_pres(v["limite"], 797.39, 0.01)
      and D["variante"]["methode"] == "§ 6.3.2.3" and round(D["variante"]["f"], 4) == 0.9877 and round(D["variante"]["chi_LT"], 4) == 0.3760
      and a_pres(D["variante"]["M_b_Rd"] / 1e6, 807.36, 0.01) and any("807,4" in x for x in v["messages"])
      and a_pres(D["annexe_E"]["M_cr"] / 1e6, 1185.17, 0.01), (D.get("lambda_LT"), D.get("chi_LT"), v and v["limite"], D.get("variante"), v and v["messages"]))
    # ---- PRS-B : propriétés
    zp = PR.zj_psi(B, "haut")
    c("PRS-B, propriétés (§ 11.8) : A = 23 100 ; centre de gravité à 337,013 mm ; I_y = 1 384,684·10⁶ ; I_z = 87,652·10⁶ ; W_el,y = 5 265,2·10³ (haute) "
      "et 4 108,7·10³ (basse) ; axe neutre plastique à 400,0 mm ; W_pl,y = 5 218,5·10³ ; W_pl,z = 1 005,4·10³",
      round(B["A"], 6) == 23100.0 and round(B["z_g"], 3) == 337.013 and round(B["Iy"] / 1e6, 3) == 1384.684 and round(B["Iz"] / 1e6, 3) == 87.652
      and round(B["Wel_y_haut"] / 1e3, 1) == 5265.2 and round(B["Wel_y_bas"] / 1e3, 1) == 4108.7 and B["Wel_y"] == B["Wel_y_bas"]
      and round(B["z_pl"], 1) == 400.0 and round(B["Wpl_y"] / 1e3, 1) == 5218.5 and round(B["Wpl_z"] / 1e3, 1) == 1005.4,
      {k: B[k] for k in ("A", "z_g", "Iy", "Iz", "Wel_y_haut", "Wel_y_bas", "z_pl", "Wpl_y", "Wpl_z")})
    c("PRS-B, semelle haute comprimée : ψ_f = 0,5429 ; I_w = 5,0127·10¹² ((E.3) 5,0214·10¹², +0,17 %) ; centre de cisaillement à 454,714 mm, z_s = +117,701 ; "
      "∫z·(y² + z²)dA = −7,4992·10¹⁰ mm⁵ ; z_j = +144,78 mm ((E.4) : 123,77) ; I_t = 4,757·10⁶ mm⁴",
      round(zp["psi_f"], 4) == 0.5429 and round(B["Iw"] / 1e12, 4) == 5.0127 and round(PR.iw_annexe_e(B) / 1e12, 4) == 5.0214
      and round(PR.cotes(B)["z_cisaillement"], 3) == 454.714 and round(B["z_s"], 3) == 117.701 and round(B["int_z_y2z2"] / 1e10, 4) == -7.4992
      and a_pres(zp["z_j"], 144.7805, 0.0005) and round(zp["z_j_approche"], 2) == 123.77 and zp["formule_approche"] == "(E.4)"
      and round(B["It"] / 1e6, 3) == 4.757, (zp, B["Iw"], B["z_s"], B["int_z_y2z2"], B["It"]))
    zb = PR.zj_psi(B, "bas")
    c("PRS-B, petite semelle comprimée : ψ_f et z_j changent de signe (annexe E ANB, 2(7)) — ψ_f = −0,5429, z_j = −144,78 mm, (E.5)",
      proche(zb["psi_f"], -zp["psi_f"]) and proche(zb["z_j"], -zp["z_j"]) and zb["formule_approche"] == "(E.5)", zb)
    # ---- PRS-B : classification
    dB = ES.lire({"section": B, "nuance": "S355", "MyEd": 1e6})
    cl = CL.classer(B, 345.0, 0.0, 1.0, 0.0, dB["fy_parois"], "haut")
    par = {p["role"]: p for p in cl["parois"]}
    psi = -(B["z_g"] - (30 + 6 * math.sqrt(2))) / ((600 - 30 - 6 * math.sqrt(2)) - B["z_g"])
    c("PRS-B, classe, semelle haute comprimée : âme fléchie et comprimée (même sans N), α = 0,3088 ; c/t = 34,87 ≤ 36·ε/α = 94,85 ; semelle 4,467 ≤ 7,43 : "
      "classe 1 ; la semelle basse, tendue, n'a pas de limite ; ψ élastique aux pieds des cordons = −298,53/224,50 = −1,3297 (classe 3 : 62ε·(1 − ψ)·√(−ψ))",
      par["ame"]["etat"] == CL.FLEXION_COMPOSEE and round(par["ame"]["alpha"], 4) == 0.3088 and round(par["ame"]["limites"][0], 2) == 94.85
      and proche(par["ame"]["psi"], psi, 1e-12) and round(psi, 4) == -1.3297
      and proche(par["ame"]["limites"][2], 62 * math.sqrt(235 / 355) * (1 - psi) * math.sqrt(-psi), 1e-12)
      and round(par["semelle_haut"]["c_t"], 3) == 4.467 and par["semelle_bas"]["etat"] == CL.TENDUE and cl["classe"] == 1,
      {k: (p["etat"], p.get("alpha"), p["limites"]) for k, p in par.items()})
    # ---- PRS-B : M_cr, le terme de Wagner
    g, p_ = mcr(B, lambda x: 1e6), mcr(B, lambda x: -1e6)
    sans = mcr(B, lambda x: 1e6, zj=0.0)
    sans_m = mcr(B, lambda x: -1e6, zj=0.0)
    e4 = mcr(B, lambda x: 1e6, zj=zp["z_j_approche"])
    f1 = [MC.moment_constant(L10, E_ * B["Iz"], E_ * B["Iw"], G_ * B["It"], z) / 1e6 for z in (zp["z_j"], -zp["z_j"])]
    e1 = [MC.annexe_e(L10, E_ * B["Iz"], E_ * B["Iw"], G_ * B["It"], 1.0, 0.0, 1.0, 0.0, z) / 1e6 for z in (zp["z_j"], -zp["z_j"])]
    c("PRS-B, M_cr sous moment constant, TERME DE WAGNER : grande semelle comprimée 1 241,73 kN·m ; petite semelle comprimée 715,69 ((E.1) avec "
      "C₁ = C₃ = 1, formule fermée et solveur, à 0,01 kN·m) ; sans le terme : 942,71 dans les deux sens ; avec le z_j de (E.4) : 1 194,01",
      a_pres(g, 1241.73, 0.01) and a_pres(p_, 715.69, 0.01) and a_pres(sans, 942.71, 0.01) and a_pres(sans_m, 942.71, 0.01)
      and a_pres(e4, 1194.01, 0.01) and all(a_pres(x, y, 0.01) for x, y in zip(f1 + e1, (1241.73, 715.69, 1241.73, 715.69))),
      (g, p_, sans, sans_m, e4, f1, e1))
    cot = PR.cotes(B)
    vals = [reparti(B, 0.0), reparti(B, cot["face_haut"]), reparti(B, cot["face_bas"])]
    zj = zp["z_j"]
    e1 = [MC.annexe_e(L10, E_ * B["Iz"], E_ * B["Iw"], G_ * B["It"], 1.12, 0.45, 0.525, zg, zj) / 1e6 for zg in (0.0, cot["face_haut"], cot["face_bas"])]
    c("PRS-B, charge répartie : solveur 1 234,80 (centre de cisaillement) ; 1 086,25 (semelle haute, z_g = +145,286) ; 1 803,30 (semelle basse, z_g = −454,714) ; "
      "(E.1), C₁ = 1,12, C₂ = 0,45, C₃ = 0,525, z_j exact : 1 221,75 ; 1 077,68 ; 1 771,34",
      all(a_pres(x, y, 0.01) for x, y in zip(vals, (1234.80, 1086.25, 1803.30))) and all(a_pres(x, y, 0.01) for x, y in zip(e1, (1221.75, 1077.68, 1771.34)))
      and round(cot["face_haut"], 3) == 145.286 and round(cot["face_bas"], 3) == -454.714, (vals, e1, cot))
    r = barre_dev(B)
    D, v = r["grandeurs"]["deversement"], verif(r, "E-Mb")
    chi_c = m("ec3_deversement").chi_LT(D["lambda_LT"], 0.49, 0.2, 1.0, 0.2, 1.0, 2.0)[0]
    c("PRS-B, déversement, charge sur la semelle haute : M_y,Rk = 1 800,38 ; λ̄_LT = 1,2874 ; § 6.3.2.2, b = 200 mm, h/b = 3,0 → courbe d : χ_LT = 0,3430, "
      "M_b,Rd = 617,55 kN·m (b = 300 et courbe c : 709,7, ce que le module ne prend pas) ; § 6.3.2.3 refusé ; (E.1) 1 077,7",
      v is not None and round(D["W_y"] * 345 / 1e6, 2) == 1800.38 and round(D["lambda_LT"], 4) == 1.2874 and D["b_h_b"] == 200.0 and D["courbe"] == "d"
      and round(D["chi_LT"], 4) == 0.3430 and a_pres(v["limite"], 617.55, 0.01) and round(chi_c * D["W_y"] * 345 / 1e6, 1) == 709.7
      and D["methode"] == "§ 6.3.2.2" and "variante" not in D and any("refusé" in x for x in v["messages"])
      and a_pres(D["annexe_E"]["M_cr"] / 1e6, 1077.68, 0.01), (D.get("lambda_LT"), D.get("courbe"), v and v["limite"], v and v["messages"]))
    r = barre_dev(B, niveau="centre", MyEd=-500e6)
    D = r["grandeurs"]["deversement"]
    direct = reparti_signe = mcr(B, lambda x: -x * (L10 - x) / 2.0, reparties=[(0.0, L10, -1.0, 0.0)])
    e1n = MC.annexe_e(L10, E_ * B["Iz"], E_ * B["Iw"], G_ * B["It"], 1.12, 0.45, 0.525, 0.0, -zp["z_j"]) / 1e6
    c("PRS-B, moment NÉGATIF (petite semelle comprimée), charge répartie au centre de cisaillement : le M_cr du solveur sous le diagramme négatif "
      "(le z_j de la semelle haute, M_y < 0 : Wagner déstabilisant) — sous celui de la grande semelle comprimée (1 234,80) ; (E.1) avec z_j = −144,78 à 3 % "
      "près ; la section se classe avec la petite semelle comprimée",
      D["cas"] == "calcule" and proche(D["M_cr"] / 1e6, direct, 1e-9) and direct < 1234.80 - 300.0 and abs(D["annexe_E"]["M_cr"] / 1e6 / direct - 1) < 0.03
      and proche(D["annexe_E"]["M_cr"] / 1e6, e1n, 1e-9) and D["annexe_E"]["z_j"] < 0
      and r["grandeurs"]["classification"]["parois"][2]["etat"] == CL.COMPRESSION and r["grandeurs"]["classification"]["parois"][0]["etat"] == CL.TENDUE,
      (D.get("M_cr"), direct, e1n, [p_["etat"] for p_ in r["grandeurs"]["classification"]["parois"]]))
    del reparti_signe
    # ---- PRS-B : statuts
    r = m("ec3").verifier({"section": B, "nuance": "S355", "NEd": 500e3, "MyEd": 300e6, "axes": {"y": {"L": L10}, "z": {"L": L10}},
                           "flexion": {"y": {"diagramme": "reparti"}}, "deversement": {"niveau": "haut"}})
    c("PRS-B, statuts : N ≠ 0 → flambement par flexion-torsion « non couvert » (E-Nb-T) et flexion composée avec instabilité « non couvert » (E-661) ; "
      "la section reste vérifiée (sommation linéaire (6.2), 6.2.9.1(5) ne vaut qu'à semelles égales)",
      verif(r, "E-Nb-T")["statut"] == m("resultat").NON_COUVERT and verif(r, "E-661")["statut"] == m("resultat").NON_COUVERT
      and r["statut"] == m("resultat").NON_COUVERT and "(6.2)" in verif(r, "E-NM")["reference"] and verif(r, "E-NM")["statut"] == "ok"
      and proche(verif(r, "E-NM")["taux"], 500e3 / (23100 * 345.0) + 300e6 / (5218.5e3 * 345.0), 1e-6),
      [(x["id"], x["statut"], x["reference"]) for x in r["verifications"]])
    r = m("ec3").verifier({"section": A, "nuance": "S355", "NEd": 500e3, "MyEd": 300e6, "axes": {"y": {"L": L10}, "z": {"L": L10}},
                           "flexion": {"y": {"diagramme": "reparti"}}, "deversement": {"niveau": "haut"}})
    c("PRS-A (bisymétrique) sous N + M : flambement par torsion (F.3) calculé, (6.61) et (6.62) calculés comme pour un laminé ; (6.36) en section",
      verif(r, "E-661")["statut"] in ("ok", "nok") and verif(r, "E-662") is not None and "T" in r["grandeurs"]["flambement"]
      and verif(r, "E-NM")["reference"].endswith("6.2.9.1"), [(x["id"], x["statut"], x["reference"]) for x in r["verifications"]])
    # ---- M + V : (6.30) à semelles égales, (6.29) sinon
    rA = ES.verifier({"section": A, "nuance": "S355", "MyEd": 1000e6, "VzEd": 1500e3})
    rB = ES.verifier({"section": B, "nuance": "S355", "MyEd": 1000e6, "VzEd": 1500e3})
    rho = (2 * 1500e3 / (9720 * 345 / math.sqrt(3)) - 1) ** 2
    WV = PR.module_plastique_ame_reduite(B, rho)
    Wa = 6223.5e3 - rho * (540 * 15) ** 2 / (4 * 15)
    c("M + V (brief § 18.5) : PRS-A à semelles égales → (6.30) ; PRS-B → (6.29), âme à (1 − ρ)·f_y, axe neutre plastique recalculé ; la forme exacte de "
      "(6.29) redonne (6.30) à semelles égales",
      "(6.30)" in verif(rA, "E-MV")["reference"] and "(6.29)" in verif(rB, "E-MV")["reference"]
      and proche(verif(rA, "E-MV")["limite"], Wa * 345 / 1e6, 1e-9) and proche(verif(rB, "E-MV")["limite"], WV * 345 / 1e6, 1e-9)
      and proche(PR.module_plastique_ame_reduite(A, rho), Wa, 1e-9) and WV < B["Wpl_y"],
      (verif(rA, "E-MV"), verif(rB, "E-MV")))

    # ---- 2. autres tests PRS
    S0 = PR.section_prs(300, 30, 540, 15, 300, 30, a=None, nuance="S355", bords="oxycoupes")
    par = {p["role"]: p for p in CL.parois(S0)}
    r = ES.verifier({"section": S0, "nuance": "S355", "MyEd": 1e6})
    c("autre test 1 — gorge non saisie : s = 0, âme c = 540 mm (c/t = 36,0), semelle c = 142,5 mm (c/t = 4,75), avec un message",
      S0["s"] == 0.0 and proche(par["ame"]["c"], 540.0) and round(par["ame"]["c_t"], 1) == 36.0 and proche(par["semelle_haut"]["c"], 142.5)
      and round(par["semelle_haut"]["c_t"], 2) == 4.75 and PR.MSG_GORGE in S0["messages"] and any(a["code"] == "prs_gorge" for a in r["alertes"]),
      (S0["s"], par["ame"]["c"], par["semelle_haut"]["c"], S0["messages"]))
    rd = barre_dev(A, parametres={"prs_soudee_equivalente": True})
    Dd, vd = rd["grandeurs"]["deversement"], verif(rd, "E-Mb")
    rm = barre_dev(B, parametres={"prs_soudee_equivalente": True})
    c("autre test 2 — « section soudée équivalente » (DÉCISION BUREAU, défaut non coché) : cochée, § 6.3.2.3 décide (PRS-A : 807,36 kN·m) et la note "
      "l'imprime, § 6.3.2.2 pour information (797,39) ; non cochée, l'inverse ; un PRS monosymétrique : § 6.3.2.2 seul, même coché",
      m("parametres").valeur("prs_soudee_equivalente") is False and Dd["methode"] == "§ 6.3.2.3" and a_pres(vd["limite"], 807.36, 0.01)
      and Dd["variante"]["methode"] == "§ 6.3.2.2" and a_pres(Dd["variante"]["M_b_Rd"] / 1e6, 797.39, 0.01)
      and any("soudée équivalente" in h for h in rd["hypotheses"]) and rm["grandeurs"]["deversement"]["methode"] == "§ 6.3.2.2"
      and a_pres(verif(rm, "E-Mb")["limite"], 617.55, 0.01), (Dd.get("methode"), vd and vd["limite"], Dd.get("variante")))
    Sb = PR.section_prs(300, 30, 540, 15, 300, 30, a=6.0, nuance="S355", bords=None)
    rb = m("moteur").verifier({"prs": {"b1": 300, "t1": 30, "hw": 540, "tw": 15, "b2": 300, "t2": 30, "a": 6}, "nuance": "S355", "mode": "efforts",
                               "efforts": {"service": {"My": 300.0}, "elu": {"My": 400.0}}, "flambement": {"y": {"L": 10.0}, "z": {"L": 10.0}}})
    h_ = rb["methodes"]["historique"]
    Bb = PR.section_prs(300, 30, 540, 15, 200, 30, a=6.0, nuance="S355", bords=None)
    c("autre test 3 — bords des semelles non saisis : « fabrication » None, verdict de la méthode ELS BLOQUÉ (message) ; l'Eurocode 3 se calcule, en "
      "section SOUDÉE (courbe c de déversement, A_v,y = A − h_w·t_w = 15 000 mm² pour PRS-B, et non 2·b·t_f = 12 000)",
      Sb["fabrication"] is None and PR.MSG_BORDS in Sb["messages"] and h_["statut"] == m("resultat").NON_COUVERT
      and any(a["message"] == PR.MSG_BORDS for a in h_["alertes"]) and rb["methodes"]["ec3"]["statut"] in ("ok", "nok")
      and verif(rb["methodes"]["ec3"], "E-Mb") is not None and rb["methodes"]["ec3"]["grandeurs"]["deversement"]["courbe"] == "c"
      and proche(ES.aires_cisaillement(Bb, 1.2)["y"], 15000.0), (h_["statut"], h_["alertes"], rb["methodes"]["ec3"]["statut"]))
    cles_prop = {"A", "Iy", "Iz", "It", "Iw", "Wel_y", "Wpl_y", "Avz", "Avy", "iy", "iz"}
    lu = []
    for s in (A, B, S0):
        e = json.loads(json.dumps(PR.vers_bibliotheque(s)))
        s2 = PR.de_bibliotheque(e)
        lu.append((s["nom"], e, all(proche(s[k], s2[k], 1e-12) for k in cles_prop | {"z_g", "z_s", "int_z_y2z2"}) and s2["prs"] == s["prs"],
                   not (cles_prop & set(e) | cles_prop & set(e["dims"]))))
    cat = PR.de_bibliotheque({"famille": "PRS", "dims": {"h": 600.0, "b": 300.0, "tw": 15.0, "tf": 30.0}}, "S355")
    c("autre test 4 — bibliothèque (le format des profilés personnalisés du Catalogue) : un PRS enregistré puis rechargé (JSON) redonne les mêmes "
      "propriétés, RECALCULÉES — aucune n'est stockée ; un PRS du Catalogue (semelles égales) se relit ; le nom porte « PRS » (celui du Catalogue à "
      "semelles égales)",
      all(x[2] and x[3] for x in lu) and all("PRS" in x[0] for x in lu) and lu[0][1]["famille"] == "PRS" and set(lu[0][1]["dims"]) == {"h", "b", "tw", "tf", "a"}
      and set(lu[1][1]["dims"]) == {"h", "b", "tw", "tf", "b2", "tf2", "a"} and lu[1][1]["nuance"] == "S355"
      and A["nom"] == CO.nom_perso("PRS", {"h": 600, "b": 300, "tw": 15, "tf": 30}) and proche(cat["It"], S0["It"]) and cat["s"] == 0.0,
      [(x[0], x[1], x[2], x[3]) for x in lu])
    T5 = PR.section_prs(100, 5, 540, 30, 100, 5, a=None, nuance="S235", bords="oxycoupes")
    seule = 2 * 100 * 5 ** 3 / 3 * (1 - 0.63 * 5 / 100) + 540 * 30 ** 3 / 3
    c("autre test 5 — I_t, contre-exemple du tour 8 (semelles 100 × 5, âme 540 × 30) : formule de 18.4 = 4,698·10⁶ mm⁴ (exacte 4,787·10⁶ : −1,9 %, côté "
      "sûr) ; la correction des seules semelles donnerait 4,868·10⁶ (+1,7 %, côté dangereux)",
      round(T5["It"] / 1e6, 3) == 4.698 and round(seule / 1e6, 3) == 4.868 and round((T5["It"] / 4.787e6 - 1) * 100, 1) == -1.9, (T5["It"], seule))
    # contrôles bloquants de la saisie (brief § 18.2)
    try:
        PR.section_prs(15, 30, 540, 15, 300, 30)
        msg_tw = ""
    except m("resultat").Refus as e:
        msg_tw = str(e)
    refus = ["t_w < b1" in msg_tw]
    for args, k in (((300, 30, 540, -15, 300, 30), {}), ((300, 30, 540, 15, 300, 0), {}), ((10, 30, 540, 15, 300, 30), {}),
                    ((300, 30, 540, 15, 300, 30), {"a": 300.0}), ((300, 30, 540, 15, 300, 30), {"bords": "lamine"}),
                    ((300, 30, 540, 15, 300, 30), {"nuance": "S500"})):
        try:
            PR.section_prs(*args, **k)
            refus.append(False)
        except m("resultat").Refus:
            refus.append(True)
    c("saisie (brief § 18.2) : contrôles BLOQUANTS — dimension négative ou nulle, t_w ≥ b, c ≤ 0 (gorge trop forte), bords inconnus, nuance inconnue : refusés",
      all(refus), refus)

    # ---- § 11.5, test 15 : un I ou un H avec I_y < I_z → verdict bloqué, avec un message
    try:
        m("section").saisie("I", "lamine", "I importé", h=300.0, b=150.0, tw=7.1, tf=10.7, A=5381.0, Iy=6.04e6, Iz=83.56e6, iy=33.5, iz=124.6)
        bloque = ""
    except ValueError as e:
        bloque = str(e)
    c("§ 11.5 test 15 — import : un I ou un H avec I_y < I_z → verdict bloqué, message « axes inversés »",
      "axes inversés" in bloque and "604" in bloque and "8 356" in bloque, bloque)
    # ---- 3. § 11.5, tests 22 et 25
    P = m("parametres").jeu()
    ipe = m("section").du_catalogue("IPE 300")
    av_ipe = ES.aires_cisaillement(ipe, AM.eta_de(ipe, P))
    c("§ 11.5 test 22 — η : PRS → 1,20 partout (A_v,z = 1,2·h_w·t_w ; seuil 72·ε/1,2 = 60·ε), bords saisis ou non ; I laminé → 1,00 (A_v,z du catalogue)",
      AM.eta_de(A, P) == 1.2 and AM.eta_de(Sb, P) == 1.2 and AM.eta_de(ipe, P) == 1.0 and proche(ES.aires_cisaillement(Sb, 1.2)["z"], 1.2 * 540 * 15)
      and not av_ipe["plancher_gouverne"] and round(72.0 / AM.eta_de(A, P), 6) == 60.0, (AM.eta_de(A, P), AM.eta_de(Sb, P), av_ipe))
    fy = {(n, t): ES.limite_elastique(n, A, t) for n, t in (("S235", 20.0), ("S235", 30.0), ("S355", 30.0), ("S355", 16.0), ("S235", 16.0))}
    heb3 = ES.lire({"section": m("section").du_catalogue("HEB 300"), "nuance": "S235"})
    heb2 = ES.lire({"section": m("section").du_catalogue("HEB 200"), "nuance": "S235"})
    c("§ 11.5 test 25 — f_y (tableau 3.3 ANB) : plat de 20 mm S235 → 225 ; plats de 30 mm : S235 → 225, S355 → 345 ; 16 mm ou moins : 235 et 355 ; "
      "HEB 200 S235 → 235 ; HEB 300 S235 → 225, ε de l'âme (11 mm) = 1,000, ε des semelles = 1,022",
      fy == {("S235", 20.0): 225.0, ("S235", 30.0): 225.0, ("S355", 30.0): 345.0, ("S355", 16.0): 355.0, ("S235", 16.0): 235.0}
      and heb2["fy"] == 235.0 and heb3["fy"] == 225.0 and round(math.sqrt(235 / heb3["fy_parois"]["ame"]), 3) == 1.0
      and round(math.sqrt(235 / heb3["fy_parois"]["semelle"]), 3) == 1.022, (fy, heb3["fy_parois"]))
    C_ = m("prs").section_prs(300, 30, 1420, 25, 500, 50, a=None, nuance="S235", bords="oxycoupes")
    dC = ES.lire({"section": C_, "nuance": "S235"})
    c("PRS-C (§ 11.9), plats de 30, 25 et 50 mm en S235 : f_y de chaque plat 225, 225, 215 ; f_y de calcul 215 (la plus petite, brief § 18.3) ; "
      "h_w = 1 420 mm (saisie, et non h − 2·t_f) ; A_v,z = 1,2·1 420·25 = 42 600 mm² ; W_el,y,min = 24,368·10⁶ mm³ (fibre haute)",
      dC["fy"] == 215.0 and (dC["fy_parois"]["semelle_haut"], dC["fy_parois"]["ame"], dC["fy_parois"]["semelle_bas"]) == (225.0, 225.0, 215.0)
      and m("section").hauteur_ame(C_) == 1420.0 and proche(ES.aires_cisaillement(C_, 1.2)["z"], 42600.0)
      and round(C_["Wel_y"] / 1e6, 3) == 24.368 and C_["Wel_y"] == C_["Wel_y_haut"], (dC["fy"], dC["fy_parois"], C_["Wel_y"]))

    # ---- 4. la section efficace (classe 4)
    K = PR.section_prs(400, 12, 1200, 8, 400, 12, a=None, nuance="S355", bords="oxycoupes")
    I_ = m("section").saisie("I", "soude_oxycoupe", "I saisi", h=K["h"], b=400.0, tw=8.0, tf=12.0, r=0.0, A=K["A"], Iy=K["Iy"], Iz=K["Iz"], iy=K["iy"],
                             iz=K["iz"], Wel_y=K["Wel_y"], Wel_z=K["Wel_z"], Wpl_y=K["Wpl_y"], Wpl_z=K["Wpl_z"])
    eps = {"semelle_haut": 0.8136, "semelle_bas": 0.8136, "ame": 0.8136, "semelle": 0.8136}
    aK, aI = CL.aire_efficace(K, eps), CL.aire_efficace(I_, eps)
    wK, wI = CL.module_efficace(K, eps, "y"), CL.module_efficace(I_, eps, "y")
    zK, zI = CL.module_efficace(K, eps, "z"), CL.module_efficace(I_, eps, "z")
    c("section efficace (EN 1993-1-5 § 4.4) d'un PRS 400×12 / 1 200×8 en S355, classe 4, sans gorge : A_eff, W_eff,y et W_eff,z, e_N = 0 — la MÊME que "
      "celle d'un I saisi (l'autre écriture du module : semelles égales, sans congé)",
      proche(aK["A_eff"], aI["A_eff"], 1e-12) and aK["A_eff"] < K["A"] and abs(aK["e_N"]) < 1e-9 and proche(wK["W_eff"], wI["W_eff"], 1e-9)
      and wK["W_eff"] < K["Wel_y"] and proche(zK["W_eff"], zI["W_eff"], 1e-9) and zK["W_eff"] < K["Wel_z"],
      (aK["A_eff"], aI["A_eff"], wK["W_eff"], wI["W_eff"], zK["W_eff"], zI["W_eff"]))
    KM = PR.section_prs(400, 12, 1200, 8, 250, 25, a=5.0, nuance="S355", bords="oxycoupes")
    fyp = ES.lire({"section": KM, "nuance": "S355"})["fy_parois"]
    e_ = {k: math.sqrt(235 / v) for k, v in fyp.items()}
    rh = [CL.module_efficace(KM, e_, "y", sc) for sc in ("haut", "bas")]
    ah = [efficace_assemblee(KM["prs"], KM["s"], fyp, sc == "haut") for sc in ("haut", "bas")]
    ae, eN = aire_assemblee(KM["prs"], KM["s"], fyp)
    aKM = CL.aire_efficace(KM, e_)
    c("section efficace d'un PRS MONOSYMÉTRIQUE (semelle haute 400 × 12, basse 250 × 25, âme 1 200 × 8, a = 5) recalculée par assemblage des parties "
      "efficaces : W_eff,y semelle haute puis basse comprimée ; A_eff et le décalage e_N du centre de gravité efficace",
      all(proche(r_["W_eff"], a_[0], 1e-9) for r_, a_ in zip(rh, ah)) and proche(aKM["A_eff"], ae, 1e-9) and proche(aKM["e_N"], eN, 1e-9)
      and abs(eN) > 1.0 and rh[0]["W_eff"] < KM["Wel_y"], ([r_["W_eff"] for r_ in rh], [a_[0] for a_ in ah], aKM["A_eff"], ae, aKM["e_N"], eN))
    r = ES.verifier({"section": KM, "nuance": "S355", "NEd": 1000e3})
    x = verif(r, "E-NM")
    c("classe 4 d'un PRS monosymétrique COMPRIMÉ : (6.44) avec ΔM_y = N_Ed·|e_N| (§ 6.2.2.5(4)), même sans moment",
      r["grandeurs"]["classe"] == 4 and x is not None and proche(r["grandeurs"]["delta_M_y"], 1000e3 * abs(aKM["e_N"]), 1e-9)
      and x["taux"] > verif(r, "E-N")["taux"], (x, r["grandeurs"].get("delta_M_y")))

    # ---- 5. le moteur
    pl = {"b1": 300, "t1": 30, "hw": 540, "tw": 15, "b2": 200, "t2": 30, "a": 6, "bords": "oxycoupes"}
    base = {"nuance": "S355", "mode": "efforts", "efforts": {"service": {"My": 300.0}, "elu": {"My": 400.0}}, "flambement": {"y": {"L": 10.0}, "z": {"L": 10.0}},
            "niveau": "haut", "diagramme": {"y": {"forme": "reparti"}}, "methodes": ["ec3"]}
    r1 = m("moteur").verifier({**base, "prs": pl})
    bib = [PR.vers_bibliotheque(PR.section_prs(**pl, nuance="S355"))]
    r2 = m("moteur").verifier({**base, "profil": bib[0]["nom"], "bibliotheque": bib})
    D1 = r1["methodes"]["ec3"]["grandeurs"]["deversement"]
    c("moteur, efforts imposés : un PRS par ses plats, ou par son nom en bibliothèque — le même calcul (PRS-B, charge sur la semelle haute : M_cr = 1 086,25, "
      "M_b,Rd = 617,55 kN·m)",
      r1["profil"] == r2["profil"] == B["nom"] and a_pres(D1["M_cr"] / 1e6, 1086.25, 0.01)
      and a_pres(verif(r1["methodes"]["ec3"], "E-Mb")["limite"], 617.55, 0.01) and r1["methodes"]["ec3"]["taux_max"] == r2["methodes"]["ec3"]["taux_max"],
      (r1["profil"], r2["profil"], D1.get("M_cr")))
    try:
        m("moteur").verifier({**base, "profil": "PRS 1x2x3x4", "bibliotheque": bib})
        inconnu = False
    except m("resultat").Refus:
        inconnu = True
    c("moteur : un nom absent du catalogue ET de la bibliothèque est refusé", inconnu)
    ch = {"nuance": "S355", "mode": "chargee", "modele": "appuis", "travees": [10.0], "poids_propre": False, "niveau": "haut", "methodes": ["ec3"],
          "cas": [{"nom": "G", "nature": "permanent", "charges": [{"type": "q", "v": 30.0, "a": 0.0, "b": 10.0}]}]}
    rc = m("moteur").verifier({**ch, "prs": pl})
    Dc = rc["methodes"]["ec3"]["grandeurs"]["deversement"]
    rcm = m("moteur").verifier({**ch, "prs": pl, "maintiens": [{"x": 5.0, "genre": "haut"}]})
    Dm = rcm["methodes"]["ec3"]["grandeurs"]["deversement"]
    tr = m("statique").troncons({**ch, "maintiens": [{"x": 5.0, "genre": "haut"}]}, B)
    c("moteur, barre chargée (deux appuis, 10 m, charge répartie sur la semelle haute) : M_cr = 1 086,25 kN·m (z_g = +145,286 depuis le centre de "
      "cisaillement, Wagner compris) ; un maintien de la semelle haute à mi-portée, au plan moyen de CETTE semelle (+130,286 mm), relève M_cr ; (E.1) calculé",
      a_pres(Dc["M_cr"] / 1e6, 1086.25, 0.01) and Dc["annexe_E"]["diagramme"] == "reparti" and a_pres(Dc["annexe_E"]["M_cr"] / 1e6, 1077.68, 0.01)
      and proche(tr[0]["maintiens"][0][1], PR.cotes(B)["semelle_haut"]) and round(tr[0]["maintiens"][0][1], 3) == 130.286 and Dm["M_cr"] > 1.5 * Dc["M_cr"],
      (Dc.get("M_cr"), Dm.get("M_cr"), tr[0]["maintiens"]))
    bib3 = [dict(bib[0], nuance="S275")]
    r3 = m("moteur").verifier({**base, "profil": bib[0]["nom"], "bibliotheque": bib3})
    c("moteur : la nuance de l'enregistrement de la bibliothèque passe devant celle de la barre (une nuance par PRS) — S275 : f_y = 265 N/mm²",
      r3["barre"]["nuance"] == "S275" and r3["methodes"]["ec3"]["grandeurs"]["fy"] == 265.0, (r3["barre"]["nuance"], r3["methodes"]["ec3"]["grandeurs"].get("fy")))
    rl = m("moteur").verifier({**ch, "profil": "IPE 600", "cas": [{"nom": "G", "nature": "permanent", "charges": [{"type": "q", "v": 30.0, "a": 0.0, "b": 10.0}]}]})
    Dl = rl["methodes"]["ec3"]["grandeurs"]["deversement"]
    ipe6 = m("section").du_catalogue("IPE 600")
    ref = m("mcr").moment_critique(L10, E_ * ipe6["Iz"], E_ * ipe6["Iw"], G_ * ipe6["It"], lambda x: 30.0 * x * (L10 - x) / 2.0 * 1.35,
                                   reparties=[(0.0, L10, 30.0 * 1.35, ipe6["h"] / 2.0)])["M_cr"]
    c("un laminé (IPE 600) dans la même barre : z_j = 0, z_g = h/2 — M_cr inchangé (le solveur seul, sans Wagner), aucune grandeur PRS",
      proche(Dl["M_cr"], ref, 1e-9) and "annexe_E" not in Dl and "b_h_b" not in Dl, (Dl.get("M_cr"), ref))

    # ---- 9. reprise du bloc (contre-vérification) : le sens des charges, la colonne C₃, les règles sans contrôle
    rn = barre_dev(B, niveau="haut", MyEd=-500e6)
    Dn, vn = rn["grandeurs"]["deversement"], verif(rn, "E-Mb")
    c("efforts imposés, PRS-B, M_y,Ed = −500 kN·m, charge sur la semelle haute : la charge DESCEND, comme pour un laminé — M_cr = 806,49 kN·m (Ritz "
      "indépendant : 806,49 ; le soulèvement en donnerait 1 033,76) ; (E.1) avec z_g = +145,286 (2(8) : la charge va vers le centre de cisaillement), "
      "+0,02 % ; l'hypothèse dit « charge descendante »",
      a_pres(Dn["M_cr"] / 1e6, 806.49, 0.01) and proche(Dn["annexe_E"]["z_g"], PR.cotes(B)["face_haut"]) and abs(Dn["annexe_E"]["ecart"]) < 0.001
      and any("charge descendante sur la semelle haute" in h for h in rn["hypotheses"]) and vn is not None,
      (Dn.get("M_cr"), Dn.get("annexe_E"), rn["hypotheses"]))
    rs = m("moteur").verifier({**ch, "prs": pl, "cas": [{"nom": "W", "nature": "permanent", "charges": [{"type": "q", "v": -30.0, "a": 0.0, "b": 10.0}]}]})
    Ds = rs["methodes"]["ec3"]["grandeurs"]["deversement"]
    c("barre chargée, PRS-B, SOULÈVEMENT de 30 kN/m appliqué sur la semelle haute (la semelle basse comprimée) : M_cr = 1 033,76 kN·m (Ritz : 1 033,75) ; "
      "(E.1) avec z_g = −145,286 (la charge s'éloigne du centre de cisaillement), à 0,1 % près",
      a_pres(Ds["M_cr"] / 1e6, 1033.76, 0.01) and proche(Ds["annexe_E"]["z_g"], -PR.cotes(B)["face_haut"]) and abs(Ds["annexe_E"]["ecart"]) < 0.001
      and Ds["annexe_E"]["z_j"] < 0, (Ds.get("M_cr"), Ds.get("annexe_E")))
    co_ = {**ch, "modele": "console", "travees": [4.0], "cas": [{"nom": "G", "nature": "permanent", "charges": [{"type": "q", "v": 30.0, "a": 0.0, "b": 4.0}]}]}
    rk = m("moteur").verifier({**co_, "prs": pl})
    Ek = rk["methodes"]["ec3"]
    Dk, park = Ek["grandeurs"]["deversement"], {p_["role"]: p_["etat"] for p_ in Ek["grandeurs"]["classification"]["parois"]}
    c("barre chargée, console PRS-B de 4 m sous 30 kN/m (moment négatif, la semelle BASSE comprimée) : M_cr = 4 000,51 kN·m (Ritz : 4 000,51) ; la "
      "section se classe avec la semelle basse comprimée, l'haute tendue ; hypothèse « semelle basse comprimée », z_j = −144,78 ; pas de (E.1) (console)",
      a_pres(Dk["M_cr"] / 1e6, 4000.51, 0.01) and park == {"semelle_haut": CL.TENDUE, "ame": CL.FLEXION_COMPOSEE, "semelle_bas": CL.COMPRESSION}
      and any("semelle basse comprimée" in h and "-144,78" in h.replace("−", "-") for h in Ek["hypotheses"]) and "annexe_E" not in Dk,
      (Dk.get("M_cr"), park, Ek["hypotheses"]))
    # la colonne C₃ du tableau E.1 ANB (k_z = 1)
    pf = zp["psi_f"]
    c3 = {(p_, sg): MC.coefficients_annexe_e("lineaire", p_, sg * pf)["C3"] for p_ in (1.0, 0.0, -0.25, -0.5, -0.75, -1.0) for sg in (1, -1)}
    c("(E.1), tableau E.1 ANB, colonne C₃ (k_z = 1), PRS-B (ψ_f = ±0,5429) : 1 de ψ = 1 à 0 ; ψ = −1/4 : 0,850 (ψ_f > 0) ; −1/2 : 1,3 − 1,2·ψ_f = 0,6486 ; "
      "−3/4 : 0,55 − ψ_f ; −1 : −ψ_f (les deux colonnes) ; 1 pour ψ_f ≤ 0 (sauf ψ = −1)",
      all(c3[(p_, sg)] == 1.0 for p_ in (1.0, 0.0) for sg in (1, -1)) and c3[(-0.25, 1)] == 0.85 and proche(c3[(-0.5, 1)], 1.3 - 1.2 * pf)
      and round(c3[(-0.5, 1)], 4) == 0.6486 and proche(c3[(-0.75, 1)], 0.55 - pf) and proche(c3[(-1.0, 1)], -pf) and proche(c3[(-1.0, -1)], pf)
      and all(c3[(p_, -1)] == 1.0 for p_ in (-0.25, -0.5, -0.75)), c3)
    r0 = barre_dev(B, niveau="centre", diagramme="lineaire", flexion={"y": {"diagramme": "lineaire", "psi": 0.0}})
    D0 = r0["grandeurs"]["deversement"]
    e10 = MC.annexe_e(L10, E_ * B["Iz"], E_ * B["Iw"], G_ * B["It"], 1.77 / 1.05, 0.0, 1.0, 0.0, zp["z_j"]) / 1e6
    c("(E.1) n'est plus bloqué sous moments d'extrémité : PRS-B, ψ = 0, grande semelle comprimée — C₁ = 1,77/1,05 (note b, K_ωt = 0,52), C₃ = 1 : "
      "2 093,2 kN·m, sous le solveur (2 274,32 ; Ritz 2 274,31), du côté sûr",
      D0["annexe_E"]["bloque"] is None and proche(D0["annexe_E"]["M_cr"] / 1e6, e10, 1e-9) and round(e10, 1) == 2093.2
      and a_pres(D0["M_cr"] / 1e6, 2274.32, 0.01) and D0["annexe_E"]["M_cr"] < D0["M_cr"], (D0.get("annexe_E"), D0.get("M_cr"), e10))
    nb = [MC.coefficients_annexe_e(dg, p_, 0.0, 0.5)["C1"] for dg, p_ in (("constant", 1.0), ("lineaire", 0.75), ("lineaire", 0.0))]
    c("annexe E ANB, note b (moments d'extrémité, K_ωt ≤ 1) : C₁/1,05 sans descendre sous 1 — ψ = 1 : 1,00 (et non 0,952) ; ψ = 0,75 : 1,086 ; ψ = 0 : 1,686",
      nb[0] == 1.0 and round(nb[1], 3) == 1.086 and round(nb[2], 3) == 1.686, nb)
    F_ = PR.section_prs(300, 30, 540, 15, 100, 10, a=6.0, nuance="S355", bords="oxycoupes")
    rf = m("ec3").verifier({"section": F_, "nuance": "S355", "MyEd": 300e6, "axes": {"y": {"L": L10}, "z": {"L": L10}}, "flexion": {"y": {"diagramme": "reparti"}},
                            "deversement": {"niveau": "haut"}})
    Df = rf["grandeurs"]["deversement"]
    c("annexe E ANB, note c : un PRS de ψ_f = 0,976 > 0,9 (semelles 300 × 30 et 100 × 10) — contrôle (E.1) bloqué, avec le motif ; le solveur décide",
      round(PR.zj_psi(F_, "haut")["psi_f"], 3) == 0.976 and "hors du domaine" in (Df["annexe_E"]["bloque"] or "") and "M_cr" not in Df["annexe_E"]
      and any("(E.1) bloqué" in x for x in verif(rf, "E-Mb")["messages"]) and Df["M_cr"] > 0, (Df.get("annexe_E"), verif(rf, "E-Mb")["messages"]))
    rA0 = m("ec3").verifier({"section": A, "nuance": "S355", "MyEd": 500e6, "axes": {"y": {"L": L10}, "z": {"L": L10}}, "flexion": {"y": {"diagramme": "constant"}},
                             "deversement": {"niveau": "centre"}})
    c("PRS-A, moment constant : (E.1) est exacte — l'écart s'écrit « +0,00 % », jamais « −0,00 % »",
      any("(+0,00 %)" in x for x in verif(rA0, "E-Mb")["messages"]) and not any("−0,00" in x for x in verif(rA0, "E-Mb")["messages"]),
      verif(rA0, "E-Mb")["messages"])
    trb = m("statique").troncons({**ch, "maintiens": [{"x": 5.0, "genre": "bas"}]}, B)
    c("un maintien de la semelle BASSE de PRS-B se place au plan moyen de cette semelle : −439,714 mm sous le centre de cisaillement",
      round(trb[0]["maintiens"][0][1], 3) == -439.714 and proche(trb[0]["maintiens"][0][1], PR.cotes(B)["semelle_bas"]), trb[0]["maintiens"])
    KB = PR.section_prs(250, 15, 1200, 5, 250, 15, a=None, nuance="S355", bords="oxycoupes")
    rK = ES.verifier({"section": KB, "nuance": "S355", "NEd": 500e3})
    c("PRS bisymétrique de classe 4 (250 × 15 / 1 200 × 5) en compression seule : e_N = 0 exactement (pas un bruit d'arrondi), aucune vérification "
      "E-NM (6.44) ajoutée",
      rK["grandeurs"]["classe"] == 4 and rK["grandeurs"]["efficace"]["e_N"] == 0.0 and verif(rK, "E-NM") is None and verif(rK, "E-N") is not None,
      (rK["grandeurs"]["efficace"].get("e_N"), [x["id"] for x in rK["verifications"]]))
    T40 = PR.section_prs(300, 40, 540, 15, 300, 40, a=6.0, nuance="S355", bords="oxycoupes")
    c("tableau 6.2, I soudé : t_f = 40 mm reste « t_f ≤ 40 » — courbes b (y) et c (z)",
      (m("ec3_flambement").courbes(T40, "S355", 335.0)["y"], m("ec3_flambement").courbes(T40, "S355", 335.0)["z"]) == ("b", "c"))
    D30 = PR.section_prs(300, 30.5, 540, 15, 200, 25, a=6.0, nuance="S355", bords="oxycoupes")
    r30 = m("ec3").verifier({"section": D30, "nuance": "S355", "NEd": 100e3, "MyEd": 100e6, "axes": {"y": {"L": 5000.0}, "z": {"L": 5000.0}},
                             "deversement": {"niveau": "haut"}})
    c("une épaisseur non entière s'écrit avec une virgule : « t_f = 30,5 mm », « semelle haute 300 × 30,5 » (jamais 30.5, ni 31)",
      any("t_f = 30,5 mm" in h for h in r30["hypotheses"]) and any("semelle haute 300 × 30,5," in h for h in r30["hypotheses"])
      and not any("30.5" in h for h in r30["hypotheses"]), [h for h in r30["hypotheses"] if "30" in h])
    Pj = m("parametres").jeu()
    c("η, une seule valeur : le « Avz » et le « eta » du contrat (jeu par défaut) sont ceux que l'Eurocode 3 de la barre recalcule (ec3_ame.eta_de)",
      all(s_["eta"] == AM.eta_de(s_, Pj) and proche(s_["Avz"], ES.aires_cisaillement(s_, AM.eta_de(s_, Pj))["z"]) for s_ in (A, B, S0)),
      [(s_["eta"], AM.eta_de(s_, Pj)) for s_ in (A, B, S0)])
    so = [PR.section_prs(300, 30, 540, 15, 200, 30, a=6.0, nuance="S355", bords=b_) for b_ in ("soude_oxycoupe", "soude_larges_plats")]
    c("contrat § 4 : les bords se donnent aussi par la fabrication (« soude_oxycoupe », « soude_larges_plats ») — la même section",
      [x["fabrication"] for x in so] == ["soude_oxycoupe", "soude_larges_plats"] and [x["bords"] for x in so] == ["oxycoupes", "larges_plats"]
      and proche(so[0]["Wpl_y"], B["Wpl_y"]), [(x["fabrication"], x["bords"]) for x in so])
    rv = ES.verifier({"section": Bb, "nuance": "S355", "VyEd": 100e3})
    mv = [a["message"] for a in rv["alertes"] if a["code"] == "Avy"]
    c("PRS-B sans bords saisis, V_y = 100 kN : A_v,y = A − h_w·t_w = 15 000 mm² et le message dit cette règle (section soudée), pas « 2·b·t_f : laminé »",
      proche(rv["grandeurs"]["A_v"]["y"], 15000.0) and mv == [ES.MSG_AVY[ES.AVY_SOUDE]] and "A − h_w·t_w" in mv[0], mv)
    # le Catalogue relit le même enregistrement (R2 : un enregistrement, deux lecteurs d'accord, ou un refus)
    KAT = sys.modules["api_web.catalogue"]
    eA, eB = PR.vers_bibliotheque(A), PR.vers_bibliotheque(B)
    lu_cat = KAT._def_perso(json.loads(json.dumps(eA)))
    refus_cat = []
    for e_ in (eB, {"famille": "PRS", "dims": {**eA["dims"], "zz": 1.0}}):
        try:
            KAT._def_perso(json.loads(json.dumps(e_)))
            refus_cat.append("")
        except ValueError as x_:
            refus_cat.append(str(x_))
    c("bibliothèque : le Catalogue relit un PRS bisymétrique du module Barres sous le MÊME nom et aux mêmes dimensions (la gorge admise) ; il REFUSE un "
      "PRS monosymétrique (il le relirait bisymétrique) et une dimension inconnue",
      CO.nom_perso("PRS", lu_cat["dims"]) == A["nom"] and lu_cat["dims"] == {"h": 600.0, "b": 300.0, "tw": 15.0, "tf": 30.0}
      and refus_cat[0] == KAT.MSG_PRS_MONO and "inconnue" in refus_cat[1], (lu_cat, refus_cat))
    C += controles_fin(A, B, pl, ch)
    return C


# ------------------------------------------------------------------ 10. fin du bloc (deuxième contre-vérification)
# M_cr d'un solveur de Ritz INDÉPENDANT (séries de sinus pour v et φ, terme de Wagner, hauteur des charges, maintiens par
# l'espace nul des liaisons ; numpy/scipy, hors du dépôt), PRS-B sur 10 m, M_y,Ed < 0 (semelle basse comprimée), kN·m :
# {(diagramme, niveau): (charge descendante, charge montante)}
RITZ_PRS_B_NEGATIF = {("reparti", "haut"): (806.49, 1033.75), ("reparti", "centre"): (912.30, 912.30), ("reparti", "bas"): (1343.80, 628.60),
                      ("ponctuel", "haut"): (965.87, 1328.40), ("ponctuel", "centre"): (1131.52, 1131.52), ("ponctuel", "bas"): (1841.20, 705.91)}
# PRS-C (brief § 11.9, demande du propriétaire du 16/11/2026), 28 m, n maintiens de la semelle haute à L/(n + 1), le poids de
# l'acier au centre de gravité : M_cr du même Ritz (220 termes, convergé à 0,1 kN·m) ; taux attendus du propriétaire
RITZ_PRS_C = {7: 23079.3, 8: 28279.2, 9: 34035.4}
RITZ_PRS_C_CS = {7: 23337.6, 8: 28529.9, 9: 34279.1}       # (les mêmes charges ramenées au centre de cisaillement)
TAUX_PRS_C = {7: 1.007, 8: 0.967, 9: 0.937}
MCR_PRS_C_8 = 28279.0                    # « M_cr ≈ 28 279 kN·m » (la note du propriétaire)
M_ED_PRS_C = 4199.4


def prs_c_troncon(C_, n, L=28000.0):
    """Le tronçon de PRS-C décrit en entier (fonctions du moteur : la barre chargée n'a qu'un niveau de charge) :
    1,35·5,456 kN/m au centre de gravité ; 1,5·(20,80 + 2,40) kN/m et 1,5·0,75·3,0 kN/m de 12,5 à 15,5 m sur la face
    haute ; n maintiens latéraux au plan moyen de la semelle haute."""
    q1, q2, q3, a, b = 1.35 * 5.456, 1.5 * (20.80 + 2.40), 1.5 * 0.75 * 3.0, 12500.0, 15500.0
    RA3 = q3 * (b - a) * (L - (a + b) / 2.0) / L

    def moment(x):
        mo = (q1 + q2) * x * (L - x) / 2.0 + RA3 * x
        return mo - (0.0 if x <= a else q3 * (x - a) ** 2 / 2.0 if x <= b else q3 * (b - a) * (x - (a + b) / 2.0))
    zt = m("section").cote(C_, "haut")
    return {"L": L, "moment": moment, "reparties": [(0.0, L, q1, -C_["z_s"]), (0.0, L, q2, zt), (a, b, q3, zt)],
            "maintiens": [(L * i / (n + 1), m("section").cote(C_, "semelle_haut")) for i in range(1, n + 1)]}, moment(L / 2.0)


def controles_fin(A, B, pl, ch):
    """Le sens de la charge en efforts imposés (M_y,Ed < 0), les charges ponctuelles de sens_charges, les décimales d'une
    dimension, le Catalogue (un PRS monosymétrique écarté seul ; I_t, I_w, A_v d'ici), le PRS-C de bout en bout."""
    C = []

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    PR, MC = m("prs"), m("mcr")
    # ---- efforts imposés, M_y,Ed < 0 : le sens de la charge inconnu — le plus petit M_cr des deux sens (À VÉRIFIER)
    vus, bon = {}, True
    for (dg, niv), (desc, mont) in RITZ_PRS_B_NEGATIF.items():
        r = barre_dev(B, niveau=niv, MyEd=-500e6, diagramme=dg)
        D = r["grandeurs"]["deversement"]
        sc, hyp = D.get("sens_charge"), [h for h in r["hypotheses"] if "sens de la charge inconnu" in h]
        retenu = "montante" if mont < desc - 0.01 else "descendante"
        ok = a_pres(D["M_cr"] / 1e6, min(desc, mont), 0.015)
        if niv == "centre":                          # (au centre de cisaillement, les deux sens se valent : rien à dire)
            ok = ok and sc is None and not hyp
        else:
            sc = sc or {}
            ok = (ok and a_pres((sc.get("descendante") or 0) / 1e6, desc, 0.015) and a_pres((sc.get("montante") or 0) / 1e6, mont, 0.015)
                  and sc.get("retenu") == retenu
                  and len(hyp) == 1 and "À VÉRIFIER" in hyp[0] and any(f"charge {retenu} " in h for h in r["hypotheses"])
                  and proche(D["annexe_E"]["z_g"], (1.0 if retenu == "descendante" else -1.0) * m("section").cote(B, niv)))
        vus[(dg, niv)] = (round(D["M_cr"] / 1e6, 2), (sc or {}).get("retenu"))
        bon = bon and ok
    c("efforts imposés, PRS-B, M_y,Ed = −500 kN·m : le sens de la charge est inconnu — le module prend le PLUS PETIT M_cr des deux sens, "
      "à chaque niveau (À VÉRIFIER, une ligne) ; Ritz indépendant : répartie 806,49 (haut, descendante) / 912,30 / 628,60 (bas, montante) ; "
      "ponctuelle 965,87 / 1 131,52 / 705,91 ; (E.1) avec le z_g du sens retenu", bon, vus)
    r = barre_dev(B, niveau="bas", MyEd=-500e6)
    v = verif(r, "E-Mb")
    c("PRS-B, M_y,Ed = −500 kN·m, charge sous la semelle basse : M_b,Rd = 415,0 kN·m, taux 1,205 (ne passe pas) — et non 708,8 kN·m, taux "
      "0,705, d'une charge toujours descendante", a_pres(v["limite"], 415.02, 0.01) and round(v["taux"], 3) == 1.205 and v["statut"] == "nok",
      (v["limite"], v["taux"]))
    ipe = m("section").du_catalogue("IPE 300")

    def lam(M, niv):
        x = m("ec3").verifier({"section": ipe, "nuance": "S235", "MyEd": M, "axes": {"y": {"L": 6000.0}, "z": {"L": 6000.0}},
                               "flexion": {"y": {"diagramme": "reparti"}}, "deversement": {"niveau": niv}})
        return x["grandeurs"]["deversement"]["M_cr"] / 1e6, x["hypotheses"]
    (hp, hh), (bp, bh), (hn, _), (bn, bnh), (cn, cnh) = lam(60e6, "haut"), lam(60e6, "bas"), lam(-60e6, "haut"), lam(-60e6, "bas"), lam(-60e6, "centre")
    c("un laminé (IPE 300, 6 m, charge répartie) : M_y,Ed > 0 inchangé (haut 78,28 ; bas 132,14 kN·m) ; M_y,Ed < 0 sous la semelle basse : "
      "le soulèvement gouverne, comme sur la semelle haute (78,28) ; au centre, rien à dire",
      a_pres(hp, 78.28, 0.01) and a_pres(bp, 132.14, 0.01) and proche(hn, hp) and proche(bn, hp) and not any("inconnu" in h for h in hh + bh + cnh)
      and any("inconnu" in h for h in bnh), (hp, bp, hn, bn, cn))
    # ---- sens_charges compte les charges PONCTUELLES : un soulèvement ponctuel retourne le z_g de (E.1)
    rp = m("moteur").verifier({**ch, "prs": pl, "cas": [{"nom": "W", "nature": "permanent", "charges": [{"type": "P", "v": -50.0, "a": 5.0}]}]})
    Dp = rp["methodes"]["ec3"]["grandeurs"]["deversement"]
    c("barre chargée, PRS-B, soulèvement PONCTUEL de 50 kN à mi-portée sur la semelle haute : M_cr = 1 328,40 kN·m (Ritz 1 328,40) ; (E.1) avec "
      "z_g = −145,286 (la charge s'éloigne du centre de cisaillement)",
      a_pres(Dp["M_cr"] / 1e6, 1328.40, 0.01) and Dp["annexe_E"]["diagramme"] == "ponctuel"
      and proche(Dp["annexe_E"]["z_g"], -PR.cotes(B)["face_haut"]), (Dp.get("M_cr"), Dp.get("annexe_E")))
    # ---- une dimension s'écrit avec ses décimales (le nom et les phrases donnent le même plat)
    D25 = PR.section_prs(300, 30.25, 540, 12.25, 200, 30, a=6.0, nuance="S355", bords="oxycoupes")
    h25 = m("ec3_section").verifier({"section": D25, "nuance": "S355", "MyEd": 100e6})["hypotheses"]
    c("une dimension au centième s'écrit au centième : « semelle haute 300 × 30,25, âme 540 × 12,25 » (le nom porte 30.25), jamais « 30,3 »",
      any("semelle haute 300 × 30,25, âme 540 × 12,25," in h for h in h25) and not any("30,3" in h or "12,3" in h for h in h25)
      and "30.25" in D25["nom"], [h for h in h25 if "semelle haute" in h])
    # ---- le Catalogue : un fichier qui porte un PRS monosymétrique — cet enregistrement seul est écarté, les autres lus
    KAT = sys.modules["api_web.catalogue"]
    tube = {"famille": "TUBE", "dims": {"d": 168.3, "t": 5.0}}
    note = {"recherches": [{"persos": [PR.vers_bibliotheque(A), PR.vers_bibliotheque(B), tube], "fiche": A["nom"]}]}
    texte_r = ""
    try:
        lu = KAT.normaliser(json.loads(json.dumps(note)))
        r0, relu = lu["recherches"][0], KAT.normaliser(json.loads(json.dumps(lu)))["recherches"][0]
        res = KAT.resultat(lu)
        texte_r = res["resultat"]
        ok = ([d["famille"] for d in r0["persos"]] == ["PRS", "TUBE"] and r0.get("persos_ecartes") == [PR.vers_bibliotheque(B)]
              and relu.get("persos_ecartes") == r0["persos_ecartes"] and [d["famille"] for d in relu["persos"]] == ["PRS", "TUBE"]
              and KAT.MSG_PRS_ECARTE.format(nom=B["nom"]) in res["donnees"])
        info = (r0["persos"], r0.get("persos_ecartes"))
    except ValueError as x_:
        ok, info = False, f"fichier entier refusé : {x_}"
    c("Catalogue : un fichier de profilés personnalisés qui porte un PRS monosymétrique — CET enregistrement seul est écarté (gardé tel quel, "
      "une ligne le dit), le PRS bisymétrique et le tube sont lus ; relu, rien ne change", ok, info)
    # ---- le Catalogue prend I_t (18.4), I_w, A_v,z et A_v,y d'un PRS ICI (une seule source, R2) ; le reste : le même calcul
    pc = next(q for q in KAT.persos_calcules(r0) if q["famille"] == "PRS") if ok else KAT.persos_calcules(
        {"persos": [{"famille": "PRS", "dims": {"h": 600.0, "b": 300.0, "tw": 15.0, "tf": 30.0}}]})[0]
    S0_ = PR.section_prs(300, 30, 540, 15, 300, 30, a=None, nuance="S355", bords="oxycoupes")
    lies = {"A": 1e2, "Iy": 1e4, "Iz": 1e4, "Wel_y": 1e3, "Wpl_y": 1e3, "Wpl_z": 1e3}
    pk = PR.proprietes_catalogue({"h": 600.0, "b": 300.0, "tw": 15.0, "tf": 30.0})      # (ce que le Catalogue reçoit d'ici)
    fx = {li["cle"]: li["parts"][1] for _t, ls in KAT.formules(pc) for li in ls if len(li["parts"]) > 1}   # (la fenêtre de calcul)
    c("Catalogue, PRS 600x300x15x30 : I_t = 565,67 cm⁴ (18.4 : (b·t³/3)·(1 − 0,63·t/b), trois plats) et non Σ b·t³/3 = 600,8 cm⁴ ; I_w, "
      "A_v,z = 1,2·h_w·t_w = 97,2 cm², A_v,y = A − h_w·t_w = 180 cm² du module Barres ; A, I_y, I_z, W les mêmes (deux calculs liés) ; "
      "les réserves de l'écran sur I_t et A_v,z ne s'affichent plus",
      proche(pc["It"], S0_["It"] / 1e4) and round(pc["It"], 2) == 565.67 and proche(pc["Iw"], S0_["Iw"] / 1e6) and proche(pc["Avz"], 97.2)
      and proche(pc["Avy"], 180.0) and pc["methode_It"] == PR.METHODE_IT_CATALOGUE and all(proche(pc[k], S0_[k] / f, 1e-9) for k, f in lies.items())
      and KAT._prs_barres(pc) and set(pk) == {"It", "Iw", "Avz", "Avy", "methode_It"} and all(proche(pc[k], pk[k]) for k in ("It", "Iw", "Avz", "Avy"))
      and [fx.get(k) for k in ("Avz", "Avy", "Iw")] == ["η·h_w·t_w", "A − h_w·t_w", "(h − t_f)²·t_f·b³/24"]
      and "0,63" in texte_r and "surestime" not in texte_r and "formule des laminés" not in texte_r,
      ({k: (pc.get(k), S0_.get(k)) for k in ("It", "Iw", "Avz", "Avy", "A", "Iy")}, [x for x in ("surestime", "formule des laminés") if x in texte_r]))
    # ---- PRS-C (brief § 11.9) : l'Eurocode 3 du propriétaire, § 6.3.2.2, courbe d, 7, 8 et 9 maintiens de la semelle haute
    C_ = PR.section_prs(300, 30, 1420, 25, 500, 50, a=None, nuance="S235", bords="oxycoupes")
    fonc, bout, cs = {}, {}, {}
    for n in (7, 8, 9):
        tr, MEd = prs_c_troncon(C_, n)
        r = m("ec3").verifier({"section": C_, "nuance": "S235", "MyEd": MEd, "axes": {"y": {"L": 28000.0}, "z": {"L": 28000.0}},
                               "deversement": {"troncon": tr}})
        D, v = r["grandeurs"]["deversement"], verif(r, "E-Mb")
        fonc[n] = (MEd / 1e6, D["M_cr"] / 1e6, v["taux"], r["grandeurs"]["classe"], r["grandeurs"]["fy"], D["W_y"], D["courbe"], D["methode"])
        tr_cs = {**tr, "reparties": [(x1, x2, q, 0.0) for x1, x2, q, _z in tr["reparties"]]}
        r_cs = m("ec3").verifier({"section": C_, "nuance": "S235", "MyEd": MEd, "axes": {"y": {"L": 28000.0}, "z": {"L": 28000.0}},
                                  "deversement": {"troncon": tr_cs}})
        cs[n] = r_cs["grandeurs"]["deversement"]["M_cr"] / 1e6
        rb_ = m("moteur").verifier({"nuance": "S235", "mode": "chargee", "modele": "appuis", "travees": [28.0], "poids_propre": False, "niveau": "haut",
                                    "methodes": ["ec3"], "prs": {"b1": 300, "t1": 30, "hw": 1420, "tw": 25, "b2": 500, "t2": 50, "bords": "oxycoupes"},
                                    "maintiens": [{"x": 28.0 * i / (n + 1), "genre": "haut"} for i in range(1, n + 1)],
                                    "cas": [{"nom": "G", "nature": "permanent", "charges": [{"type": "q", "v": 5.456, "a": 0.0, "b": 28.0}]},
                                            {"nom": "Q", "nature": "exploitation_A", "charges": [{"type": "q", "v": 23.2, "a": 0.0, "b": 28.0},
                                                                                              {"type": "q", "v": 2.25, "a": 12.5, "b": 15.5}]}]})
        vb = verif(rb_["methodes"]["ec3"], "E-Mb")
        bout[n] = (vb["valeur"], rb_["methodes"]["ec3"]["grandeurs"]["deversement"]["M_cr"] / 1e6, vb["taux"])
    c("PRS-C (§ 11.9), Eurocode 3 par les fonctions du moteur (le poids de l'acier au centre de gravité, béton et travail sur la semelle haute) : "
      "M_Ed = 4 199,4 kN·m ; classe 3, W_el,y de la fibre haute (24,368·10⁶ mm³), f_y = 215 (plat de 50 mm), § 6.3.2.2, courbe d ; M_cr à 0,2 % "
      "du Ritz indépendant (7, 8, 9 maintiens : 23 079,3 ; 28 279,2 ; 34 035,4 — le propriétaire : 28 279 à 8) ; taux 1,007 / 0,967 / 0,937",
      all(a_pres(f[0], M_ED_PRS_C, 0.1) and f[3] == 3 and f[4] == 215.0 and proche(f[5], C_["Wel_y_haut"]) and round(f[5] / 1e6, 3) == 24.368
          and f[6] == "d" and f[7] == "§ 6.3.2.2" and abs(f[1] / RITZ_PRS_C[n] - 1.0) <= 0.002 and a_pres(f[2], TAUX_PRS_C[n], 0.001)
          for n, f in fonc.items()) and abs(fonc[8][1] / MCR_PRS_C_8 - 1.0) <= 0.002, fonc)
    c("PRS-C, la hauteur des charges compte : les mêmes charges au centre de cisaillement donnent un M_cr plus grand d'au moins 0,5 % "
      "(7, 8, 9 maintiens : +1,1 / +0,9 / +0,7 % ; Ritz indépendant 23 337,6 ; 28 529,9 ; 34 279,1)",
      all(cs[n] / fonc[n][1] - 1.0 >= 0.005 and abs(cs[n] / RITZ_PRS_C_CS[n] - 1.0) <= 0.002 for n in fonc),
      {n: (round(fonc[n][1], 1), round(cs[n], 1)) for n in fonc})
    c("PRS-C de bout en bout par le moteur de la barre (barre chargée, 1,35·G + 1,5·Q, maintiens « haut ») : M_Ed = 4 199,4 kN·m ; taux "
      "1,007 / 0,967 / 0,937 — le modèle n'a qu'un niveau de charge : l'acier aussi sur la semelle haute (côté sûr, M_cr −0,1 %)",
      all(a_pres(b_[0], M_ED_PRS_C, 0.1) and a_pres(b_[2], TAUX_PRS_C[n], 0.001) and fonc[n][1] * 0.997 <= b_[1] <= fonc[n][1] for n, b_ in bout.items()),
      bout)
    return C


# ------------------------------------------------------------------ 6. les formules
def cas_formules():
    PR = m("prs")
    A, B = prs_a(), prs_b()
    KM = PR.section_prs(400, 12, 1200, 8, 250, 25, a=5.0, nuance="S355", bords="oxycoupes")
    return [
        ("ec3_section", {"section": B, "nuance": "S355", "NEd": 200e3, "MyEd": 900e6, "VzEd": 1500e3}),
        ("ec3_section", {"section": B, "nuance": "S355", "MyEd": -600e6, "VzEd": 300e3}),
        ("ec3_section", {"section": KM, "nuance": "S355", "NEd": 1000e3}),
        ("ec3_section", {"section": KM, "nuance": "S355", "NEd": 500e3, "MyEd": 200e6, "MzEd": 10e6}),
        ("ec3", {"section": A, "nuance": "S355", "MyEd": 500e6, "axes": {"y": {"L": L10}, "z": {"L": L10}}, "flexion": {"y": {"diagramme": "reparti"}},
                 "deversement": {"niveau": "haut"}}),
        ("ec3", {"section": B, "nuance": "S355", "MyEd": 500e6, "axes": {"y": {"L": L10}, "z": {"L": L10}}, "flexion": {"y": {"diagramme": "constant"}},
                 "deversement": {"niveau": "centre"}}),
        # (E.1) sous moments d'extrémité, C₃ du tableau E.1 ANB ≠ 1 (ψ = −1/2, ψ_f > 0) ; moment négatif, charge descendante
        ("ec3", {"section": B, "nuance": "S355", "MyEd": 500e6, "axes": {"y": {"L": L10}, "z": {"L": L10}},
                 "flexion": {"y": {"diagramme": "lineaire", "psi": -0.5}}, "deversement": {"niveau": "centre"}}),
        ("ec3", {"section": B, "nuance": "S355", "MyEd": -500e6, "axes": {"y": {"L": L10}, "z": {"L": L10}}, "flexion": {"y": {"diagramme": "reparti"}},
                 "deversement": {"niveau": "haut"}}),
    ]


# ------------------------------------------------------------------ 7. les mutations
def _ordre():
    """Les modules du paquet et ceux de chacun qu'ils importent (« from . import x », « from .x import »)."""
    dep = {}
    for f in sorted(os.listdir(DOSSIER)):
        if f.endswith(".py") and f != "__init__.py":
            src = open(os.path.join(DOSSIER, f), encoding="utf-8").read()
            dep[f[:-3]] = set(re.findall(r"^\s*from \. import (\w+)", src, re.M)) | set(re.findall(r"^\s*from \.(\w+) import", src, re.M))
    return dep


DEP = _ordre()


def _dependants(cible):
    """Les modules qui emploient ``cible`` (directement ou non), dans un ordre où chacun vient après ce qu'il importe."""
    touches, change = {cible}, True
    while change:
        change = False
        for k, v in DEP.items():
            if k not in touches and v & touches:
                touches.add(k)
                change = True
    touches.discard(cible)
    ordre, faits = [], {cible}
    while touches - set(ordre):
        for k in sorted(touches - set(ordre)):
            if DEP[k] & touches <= set(ordre) | faits:
                ordre.append(k)
    return ordre


def _installer(nom, mod):
    sys.modules[f"acier.barre.{nom}"] = mod
    setattr(PAQUET, nom, mod)
    for k in _dependants(nom):
        if f"acier.barre.{k}" in sys.modules:
            importlib.reload(sys.modules[f"acier.barre.{k}"])


def muter(fichier, ancien, nouveau):
    """Les contrôles qui rougissent quand le texte ``ancien`` de ``fichier`` devient ``nouveau`` ; None si le texte n'y
    est pas une seule fois. Le module d'origine est remis, et ceux qui l'emploient rechargés, quoi qu'il arrive.
    ``fichier`` : un module du paquet (« prs »), ou un chemin depuis la racine (« api_web/catalogue ») — celui-là n'a
    pas de dépendant dans le paquet : il est seulement remplacé."""
    externe = "/" in fichier
    chemin = os.path.join(RACINE, fichier + ".py") if externe else os.path.join(DOSSIER, fichier + ".py")
    nom_mod = fichier.replace("/", ".") if externe else f"acier.barre.{fichier}"
    src = open(chemin, encoding="utf-8").read()
    if src.count(ancien) != 1:
        return None
    orig = sys.modules[nom_mod]
    mod = types.ModuleType(nom_mod)
    mod.__file__, mod.__package__ = chemin, nom_mod.rpartition(".")[0]
    installer = (lambda m_: sys.modules.__setitem__(nom_mod, m_)) if externe else (lambda m_: _installer(fichier, m_))
    try:
        exec(compile(src.replace(ancien, nouveau), chemin, "exec"), mod.__dict__)
        installer(mod)
        rouges = [n for n, ok, _ in controles() if not ok]
    except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
        rouges = [f"exception {type(e).__name__}: {e}"]
    finally:
        installer(orig)
    return rouges


MUTATIONS = [
    ("prs", "I_t : Σ b·t³/3 sans la correction (1 − 0,63·t/b)", "    return b * t ** 3 / 3.0 * (1.0 - COEF_IT * t / b)", "    return b * t ** 3 / 3.0"),
    ("prs", "I_t : la correction sur les seules semelles", '"It": torsion_rectangle(b1, t1) + torsion_rectangle(hw, tw) + torsion_rectangle(b2, t2),',
     '"It": torsion_rectangle(b1, t1) + hw * tw ** 3 / 3.0 + torsion_rectangle(b2, t2),'),
    ("prs", "I_w avec h au lieu de h_s", '"Iw": hs ** 2 * I1 * I2 / (I1 + I2),', '"Iw": h ** 2 * I1 * I2 / (I1 + I2),'),
    ("prs", "centre de cisaillement : I₁ au lieu de I₂ (côté de la mauvaise semelle)", "    z_sc = h - t1 / 2.0 - hs * I2 / (I1 + I2)", "    z_sc = h - t1 / 2.0 - hs * I1 / (I1 + I2)"),
    ("prs", "z_j : le signe de l'intégrale retourné", '    zj_haut = s["z_s"] - 0.5 * s["int_z_y2z2"] / s["Iy"]', '    zj_haut = s["z_s"] + 0.5 * s["int_z_y2z2"] / s["Iy"]'),
    ("prs", "z_j : ∫z·y²dA en b³/12 au lieu de b³/24", "    integrale = sum(b ** 3 * ((z2 - zg) ** 2 - (z1 - zg) ** 2) / 24.0", "    integrale = sum(b ** 3 * ((z2 - zg) ** 2 - (z1 - zg) ** 2) / 12.0"),
    ("prs", "ψ_f : semelle tendue moins semelle comprimée", "    psi = (Ifc - Ift) / (Ifc + Ift)", "    psi = (Ift - Ifc) / (Ifc + Ift)"),
    ("prs", "s = a (sans √2)", "        s = a * RACINE2", "        s = a"),
    ("prs", "W_pl,y autour du centre de gravité (et non de l'axe neutre plastique)", '"Wpl_y": module_plastique(rects, zp),', '"Wpl_y": module_plastique(rects, zg),'),
    ("prs", "A_v,z avec η = 1", '"Avz": eta * hw * tw,', '"Avz": hw * tw,'),
    ("prs", "W_el,y : la plus GRANDE des deux fibres", '"Wel_y": min(Iy / (h - zg), Iy / zg),', '"Wel_y": max(Iy / (h - zg), Iy / zg),'),
    ("prs", "h/b du déversement avec la plus GRANDE largeur", '"h": h, "b": min(b1, b2),', '"h": h, "b": max(b1, b2),'),
    ("prs", "t_f du tableau 6.2 : la plus MINCE des semelles", '"tw": tw, "tf": max(t1, t2),', '"tw": tw, "tf": min(t1, t2),'),
    ("prs", "α de l'âme : semelles comprimée et tendue échangées", "    brut = 0.5 + (N / fy - Afc + Aft) / (2.0 * c * p[\"tw\"])", "    brut = 0.5 + (N / fy + Afc - Aft) / (2.0 * c * p[\"tw\"])"),
    ("prs", "ψ de l'âme : v_t et v_c échangés", "    psi = (sN - (fy - sN) * vt / vc) / fy", "    psi = (sN - (fy - sN) * vc / vt) / fy"),
    ("prs", "bords non saisis pris « oxycoupés » (rien ne bloque la méthode ELS)", '"fabrication": BORDS.get(bords),', '"fabrication": BORDS.get(bords, "soude_oxycoupe"),'),
    ("prs", "gorge non saisie : pas de message", "        messages.append(MSG_GORGE)\n", "        pass\n"),
    ("prs", "contrôle bloquant t_w < b oublié", "    if not (tw < b1 and tw < b2):", "    if False:"),
    ("prs", "section efficace : e_N de signe retourné", "    eN = (s[\"A\"] * s[\"z_g\"] - sum(a * z for a, z in trous)) / A_eff - s[\"z_g\"] if trous and",
     "    eN = s[\"z_g\"] - (s[\"A\"] * s[\"z_g\"] - sum(a * z for a, z in trous)) / A_eff if trous and"),
    ("prs", "section efficace : le trou de l'âme pris depuis l'autre bord", "            debut = zc_bord + sens * be1 ", "            debut = zt_bord - sens * be1 "),
    ("prs", "W_eff : la fibre la plus PROCHE de l'axe neutre efficace", "    if haut_d >= bas_d:", "    if haut_d < bas_d:"),
    ("prs", "(6.29) : l'âme entière gardée", 'rects = _rectangles(p["b1"], p["t1"], p["hw"], (1.0 - rho) * p["tw"], p["b2"], p["t2"])',
     'rects = _rectangles(p["b1"], p["t1"], p["hw"], p["tw"], p["b2"], p["t2"])'),
    ("prs", "bibliothèque : la semelle basse d'un monosymétrique oubliée", '        dims.update({"b2": p["b2"], "tf2": p["t2"]})', "        pass"),
    ("prs", "cotes : depuis le centre de gravité (et non du centre de cisaillement)", '    zc = s["z_g"] + s["z_s"]\n', '    zc = s["z_g"]\n'),
    ("mcr", "terme de Wagner retiré", "                        Kg[ip[a]][ip[b]] += 2.0 * wt * M * zj * dN[a] * dN[b]", "                        pass"),
    ("mcr", "terme de Wagner de signe inversé", "                        Kg[ip[a]][ip[b]] += 2.0 * wt * M * zj * dN[a] * dN[b]", "                        Kg[ip[a]][ip[b]] -= 2.0 * wt * M * zj * dN[a] * dN[b]"),
    ("mcr", "terme de Wagner compté à moitié (½·∫M·z_j·φ′²)", "                        Kg[ip[a]][ip[b]] += 2.0 * wt * M * zj * dN[a] * dN[b]", "                        Kg[ip[a]][ip[b]] += wt * M * zj * dN[a] * dN[b]"),
    ("mcr", "mémoire : z_j oublié dans la signature (un M_cr sans Wagner rendu pour la même poutre)", "    cle = (L, EIz, EIw, GIt, round(zj, 9), ", "    cle = (L, EIz, EIw, GIt, "),
    ("mcr", "formule fermée monosymétrique : − z_j", "        return Pz * (zj + math.sqrt(zj * zj + EIw / EIz + GIt / Pz))", "        return Pz * (-zj + math.sqrt(zj * zj + EIw / EIz + GIt / Pz))"),
    ("mcr", "(E.1) : C₃·z_j de signe retourné", "    x = C2 * zg - C3 * zj\n", "    x = C2 * zg + C3 * zj\n"),
    ("mcr", "tableau E.2 ANB : C₃ de la charge répartie à 0,411", 'E2 = {"reparti": (1.12, 0.45, 0.525),', 'E2 = {"reparti": (1.12, 0.45, 0.411),'),
    ("ec3_deversement", "le z_j du PRS non passé au solveur", "    zj = PR.zj_haut(s)\n", "    zj = 0.0\n"),
    ("ec3_deversement", "PRS : § 6.3.2.3 par défaut", '        if prs and not (P["prs_soudee_equivalente"] and not s["monosymetrique"]):', "        if False:"),
    ("ec3_deversement", "PRS monosymétrique : § 6.3.2.3 admis quand il est déclaré", '        if prs and not (P["prs_soudee_equivalente"] and not s["monosymetrique"]):',
     '        if prs and not P["prs_soudee_equivalente"]:'),
    ("ec3_deversement", "§ 6.3.2.2 avec le facteur f", '        if s["forme"] == "I" and methode != METHODE_GENERALE:', '        if s["forme"] == "I":'),
    ("ec3_deversement", "PRS sans bords saisis : courbes d'un laminé", '        famille = "soude" if (s["fabrication"] in SOUDES or SE.soude(s)) else "lamine"',
     '        famille = "soude" if s["fabrication"] in SOUDES else "lamine"'),
    ("ec3_deversement", "moment négatif d'un PRS monosymétrique pris positif", "    signe = -1.0 if (PR.monosymetrique(s) and d.get(\"semelle_comprimee\") == PR.BAS) else 1.0", "    signe = 1.0"),
    ("ec3_flambement", "PRS monosymétrique comprimé : flambement par torsion calculé comme un bisymétrique", "    if PR.monosymetrique(s):\n        # PRS monosymétrique (brief § 7.6",
     "    if False:\n        # PRS monosymétrique (brief § 7.6"),
    ("ec3_flambement", "PRS sans bords saisis : pas de courbe (refus)", '        elif forme == "I" and (fab in ("soude_oxycoupe", "soude_larges_plats") or SE.soude(s)):',
     '        elif forme == "I" and fab in ("soude_oxycoupe", "soude_larges_plats"):'),
    ("ec3_interaction", "PRS monosymétrique : (6.61), (6.62) calculées", "    if PR.monosymetrique(s):\n        # PRS monosymétrique (brief § 7.8", "    if False:\n        # PRS monosymétrique (brief § 7.8"),
    ("ec3_section", "PRS : A_v,z d'un laminé", "    if PR.est_prs(s):\n        # PRS (brief § 7.5 et 18.5)", "    if False:\n        # PRS (brief § 7.5 et 18.5)"),
    ("ec3_section", "PRS monosymétrique sous N + M : (6.36)", "    if PR.monosymetrique(s):\n        # PRS à semelles différentes : 6.2.9.1(5)", "    if False:\n        # PRS à semelles différentes : 6.2.9.1(5)"),
    ("ec3_section", "PRS monosymétrique sous M + V : (6.30)", '    if a == "y" and PR.monosymetrique(s) and classe <= 2 and rho["y"] == 0.0:', '    if False:'),
    ("ec3_section", "classe 4 : ΔM_y = N_Ed·e_N oublié", "        if eN and N > 0:\n", "        if False:\n"),
    ("ec3_section", "f_y d'une semelle de PRS : celle de l'autre semelle", '        out = {"semelle_haut": limite_elastique(nuance, s, p["t1"]), "semelle_bas": limite_elastique(nuance, s, p["t2"]),',
     '        out = {"semelle_haut": limite_elastique(nuance, s, p["t2"]), "semelle_bas": limite_elastique(nuance, s, p["t1"]),'),
    ("ec3_section", "le signe du moment perdu (la semelle haute toujours comprimée)", '    d["semelle_comprimee"] = PR.BAS if d["MyEd"] < 0 else PR.HAUT', '    d["semelle_comprimee"] = PR.HAUT'),
    ("ec3_classe", "âme d'un PRS monosymétrique fléchi : les limites de la flexion seule (72ε)", '        elif role == "ame" and PR.est_prs(s) and fy_ and (mono or N != 0):',
     '        elif role == "ame" and PR.est_prs(s) and fy_ and N != 0:'),
    ("ec3_classe", "semelle tendue d'un PRS classée comprimée, la semelle comprimée tendue", '            comprimee_par_M = fy_ and role == f"semelle_{semelle_comprimee}"',
     '            comprimee_par_M = fy_ and role != f"semelle_{semelle_comprimee}"'),
    ("section", "§ 11.5 test 15 — I_y < I_z accepté", '    if forme == "I" and s["Iy"] < s["Iz"]:', "    if False:"),
    ("section", "h_w d'un PRS : h − 2·t_f", '    if s.get("prs"):\n        return s["prs"]["hw"]', '    if False:\n        return s["prs"]["hw"]'),
    ("section", "cote d'une charge sur un PRS : ±h/2 (depuis le centre de gravité)", "    if s.get(\"prs\"):\n        from . import prs as PR", "    if False:\n        from . import prs as PR"),
    ("ec3", "niveau des charges d'un PRS : ±h/2", "        zg = SE.cote(s, niveau) if s.get(\"prs\") else", "        zg = NIVEAUX[niveau] * (s.get(\"h\") or 0.0) / 2.0 if True else"),
    ("ec3_ame", "η = 1,00 pour un PRS sans bords saisis", '    if s.get("prs"):\n        return P["eta_soude"]', '    if False:\n        return P["eta_soude"]'),
    ("moteur", "bords non saisis : la méthode ELS calculée quand même", "    if PR.est_prs(s) and s.get(\"fabrication\") is None:\n        raise RE.Refus(PR.MSG_BORDS)",
     "    if False:\n        raise RE.Refus(PR.MSG_BORDS)"),
    ("moteur", "barre chargée : les charges d'un PRS à ±h/2", "    if PR.est_prs(s):                                          # un PRS : au-dessus de SON centre de cisaillement",
     "    if False:                                          # un PRS : au-dessus de SON centre de cisaillement"),
    ("statique", "maintien d'une semelle de PRS au plan moyen d'un I bisymétrique", '        dedans = [((x - a) * 1e3, SE.cote(s, "semelle_" + genre) if prs else',
     '        dedans = [((x - a) * 1e3, SE.cote(s, "semelle_" + genre) if False else'),
    ("entree", "bibliothèque : la nuance de l'enregistrement ignorée", '        b["nuance"] = s["nuance"]\n', "        pass\n"),
    # ---- reprise du bloc (contre-vérification)
    ("ec3_deversement", "efforts imposés, M < 0 : la charge retournée (ascendante)", "    sens, M = (-1.0 if M < 0 else 1.0), abs(M)  ", "    sens = 1.0  "),
    ("ec3_deversement", "efforts imposés, M < 0 : le moment pris positif (semelle haute comprimée)", "            return -brut(x)\n", "            return brut(x)\n"),
    ("ec3_deversement", "(E.1) : z_g sans le sens de la charge", '            zg = dev["zg"] * sens_charges(tr) if charge else 0.0', '            zg = dev["zg"] if charge else 0.0'),
    ("ec3_deversement", "(E.1) : z_g retourné quand la semelle basse est comprimée (l'ancienne règle)",
     '            zg = dev["zg"] * sens_charges(tr) if charge else 0.0', '            zg = (dev["zg"] if signe > 0 else -dev["zg"]) if charge else 0.0'),
    ("ec3_deversement", "hypothèse : le sens de la charge non dit", '    sens = " montante" if sens_charges(tr) < 0 else (" descendante" if PR.monosymetrique(s) or inconnu else "")',
     '    sens = ""'),
    ("ec3_deversement", "écart de (E.1) : signe pris avant l'arrondi (« −0,00 % »)", '            ecart = round(E["ecart"] * 100, 2) + 0.0', '            ecart = E["ecart"] * 100'),
    ("moteur", "barre chargée : le signe du moment d'un PRS monosymétrique perdu", "                if PR.monosymetrique(s) and _moment_extreme(C, tr, xs) < 0:",
     "                if False:"),
    ("mcr", "tableau E.1 ANB : C₃ de ψ = −1/2 (ψ_f > 0) pris à 1", "(_UN, (1.3, -1.2))", "(_UN, _UN)"),
    ("mcr", "tableau E.1 ANB : les deux colonnes de C₃ échangées", "        a, b = E1_C3[i][psi_f > 0]", "        a, b = E1_C3[i][psi_f < 0]"),
    ("mcr", "note b : C₁/1,05 sans le plancher 1", "        c1 = max(c1 / NOTE_B, 1.0)", "        c1 = c1 / NOTE_B"),
    ("mcr", "domaine −0,9 ≤ ψ_f ≤ 0,9 retiré", "    if abs(psi_f) > PSI_F_MAX:", "    if False:"),
    ("prs", "cote du plan moyen de la semelle basse = face basse", '"semelle_bas": p["t2"] / 2.0 - zc,', '"semelle_bas": -zc,'),
    ("prs", "e_N d'un PRS bisymétrique : le bruit d'arrondi gardé", 'if trous and s.get("monosymetrique") else 0.0', "if trous else 0.0"),
    ("prs", "une dimension écrite sans ses décimales", "    return fr(x, n)\n", "    return fr(x)\n"),
    ("prs", "une dimension au centième arrondie au dixième (30,25 → 30,3)", "DECIMALES_MM = 3 ", "DECIMALES_MM = 1 "),
    ("prs", "contrat : la fabrication « soude_oxycoupe » refusée comme bords", "    bords = {v: k for k, v in BORDS.items()}.get(bords, bords)", "    pass"),
    ("prs", "η du contrat : celui des laminés", '    eta = PA.valeur("eta_soude")', '    eta = PA.valeur("eta")'),
    ("ec3_flambement", "tableau 6.2 : t_f = 40 pris épais", "            mince = s[\"tf\"] <= SEMELLE_EPAISSE", "            mince = s[\"tf\"] < SEMELLE_EPAISSE"),
    ("ec3_section", "A_v,y d'un PRS sans bords : le message d'un laminé", '            J.alerte("Avy", "info", MSG_AVY[av["regle_y"]])',
     '            J.alerte("Avy", "info", MSG_AVY[AVY_SOUDE if s.get("fabrication") in AM.SOUDES else AVY_LAMINE])'),
    ("api_web/catalogue", "Catalogue : un PRS monosymétrique relu bisymétrique", "        raise ValueError(MSG_PRS_MONO)", "        pass"),
    ("api_web/catalogue", "Catalogue : une dimension inconnue admise", "    if inconnues:\n", "    if False:\n"),
    # ---- fin du bloc (deuxième contre-vérification)
    ("ec3_deversement", "efforts imposés, M < 0 : la charge toujours DESCENDANTE (les deux sens non comparés)", "        if inconnu:\n", "        if False:\n"),
    ("ec3_deversement", "efforts imposés, M < 0 : la charge toujours MONTANTE",
     '            if r_m["M_cr"] is not None and (r["M_cr"] is None or r_m["M_cr"] < r["M_cr"]):', '            if r_m["M_cr"] is not None:'),
    ("ec3_deversement", "efforts imposés, M < 0 : la ligne À VÉRIFIER oubliée", '    if "sens_charge" in D:\n', "    if False:\n"),
    ("ec3_deversement", "sens_charges : les charges ponctuelles ignorées", '    total += sum(p[1] for p in tr.get("ponctuelles") or ())', "    total += 0.0"),
    ("mcr", "charges RÉPARTIES au centre de cisaillement (hauteur ignorée)", "                    if qz:\n", "                    if False:\n"),
    ("mcr", "charges RÉPARTIES : hauteur comptée à moitié", "                        Kg[ip[a]][ip[b]] -= wt * qz * N[a] * N[b]",
     "                        Kg[ip[a]][ip[b]] -= 0.5 * wt * qz * N[a] * N[b]"),
    ("mcr", "charges PONCTUELLES au centre de cisaillement (hauteur ignorée)", "        Kg[4 * noeud(x) + 2][4 * noeud(x) + 2] -= P * zg", "        pass"),
    ("prs", "Catalogue : A_v,z et A_v,y d'un laminé gardés pour un PRS",
     '    out.update({"It": s["It"] / 1e4, "Iw": s["Iw"] / 1e6, "Avz": s["Avz"] / 1e2, "Avy": s["Avy"] / 1e2})', '    out.update({"It": s["It"] / 1e4, "Iw": s["Iw"] / 1e6})'),
    ("api_web/catalogue", "Catalogue : le PRS de l'écran (I_t = Σ b·t³/3, non conservatif)", "    if fam == PR.FAMILLE and prs_barres:", "    if False:"),
    ("api_web/catalogue", "Catalogue : les réserves de l'écran gardées pour un PRS du module Barres", "        if p is not None and reserve(p) and not _prs_barres(p):",
     "        if p is not None and reserve(p):"),
    ("api_web/catalogue", "Catalogue : un PRS monosymétrique fait perdre TOUT le fichier", "                if str(e) != MSG_PRS_MONO:\n                    raise\n", "                raise\n"),
    ("api_web/catalogue", "Catalogue : le PRS écarté perdu à la relecture", '        for d in (r.get("persos") or []) + (r.get("persos_ecartes") or []):',
     '        for d in (r.get("persos") or []):'),
    ("api_web/catalogue", "Catalogue : le PRS écarté sans un mot", '    for d in r.get("persos_ecartes") or []:', "    for d in []:"),
    ("api_web/catalogue", "Catalogue : la formule de A_v,z d'un laminé écrite pour le PRS du module Barres",
     "    if prs_b:                                    # EN 1993-1-1", "    if False:                                    # EN 1993-1-1"),
    ("api_web/catalogue", "Catalogue : la formule de I_w d'un laminé écrite pour le PRS du module Barres",
     "    if prs_b:                                    # semelles seules", "    if False:                                    # semelles seules"),
]


def main():
    print("== 1 à 5. PRS-A, PRS-B, autres tests PRS, § 11.5, section efficace, moteur")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 6. les formules")
    faux, n, utilises = [], 0, set()
    for mod, b in cas_formules():
        r = m(mod).verifier(b)
        cles = [f["cle"] for f in r["calcul"]]
        if len(cles) != len(set(cles)):
            faux.append((mod, "clé répétée", [k for k in cles if cles.count(k) > 1][:3]))
        for f in r["calcul"]:
            n += 1
            utilises.add(f["gabarit"])
            if f["gabarit"] not in FO.GABARITS:
                faux.append((mod, f["cle"], "gabarit absent"))
            elif not proche(FO.evaluer(f), f["resultat"]):
                faux.append((mod, f["cle"], FO.evaluer(f), f["resultat"]))
        for v in r["verifications"]:
            manquent = [k for k in v["formules"] if k not in set(cles)]
            if manquent:
                faux.append((mod, v["id"], "formules absentes", manquent))
        json.dumps(r, allow_nan=False)
    chk(f"chaque bloc de formule des PRS se recalcule par son gabarit, sur ses seuls termes, à 10⁻⁹ ({n} formules) ; clés uniques ; JSON strict",
        n > 80 and not faux, faux[:4])
    morts = set(FO.GABARITS_EC3_PRS) - utilises
    chk("chaque gabarit des PRS sert à un contrôle (aucun n'est mort)", not morts, sorted(morts))

    print("== 7. mutations")
    for fichier, nom, ancien, nouveau in MUTATIONS:
        rouges = muter(fichier, ancien, nouveau)
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans {fichier}.py, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » ({fichier}.py) : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("après les mutations, le paquet est rendu : les contrôles repassent tous", all(ok for _, ok, _ in controles()))

    print("== 8. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('numpy', 'scipy', 'pandas', 'streamlit', 'matplotlib', 'reportlab'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from acier.barre import prs as P, mcr as M\n"
            "s = P.section_prs(300, 30, 540, 15, 200, 30, a=6, nuance='S355', bords='oxycoupes')\n"
            "r = M.moment_critique(10000.0, 210000.0 * s['Iz'], 210000.0 * s['Iw'], 81000.0 * s['It'], lambda x: 1e6, zj=P.zj_haut(s))\n"
            "print(round(r['M_cr'] / 1e6, 2))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("prs.py et le solveur de M_cr se chargent et calculent sans numpy, scipy, pandas, streamlit (refusés) : PRS-B, 1 241,73 kN·m",
        p.returncode == 0 and p.stdout.strip() == "1241.73", (p.stdout.strip(), p.stderr[-300:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

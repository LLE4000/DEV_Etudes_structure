# -*- coding: utf-8 -*-
"""Barres en acier — L'ÂME ET SES RAIDISSEURS selon l'EN 1993-1-5 (acier/barre/ec3_ame.py, raidisseurs.py, ame.py) :
voilement par cisaillement, âme sous une force concentrée, interactions, décision et dimensionnement des raidisseurs.
La spécification : docs/acier/BARRES_ACIER.md § 5.9 ; brief § 8 et § 11.3 (cas R4).

Les références :
  1. R4 — IPE 300 S235 sous charge concentrée, types (a) et (c) : les valeurs du brief (calculées indépendamment), et
     un témoin écrit ici depuis le texte de la norme, comparé au moteur sur tous les I du catalogue.
  2. CISAILLEMENT — IPE 750x147 S355 à la main ; l'interaction (7.1) ; les deux âmes d'un tube ; ce qui n'est pas couvert.
  3. RAIDISSEUR — à la main (IPE 300, HEB 300) ; la forme du plat sur tout le catalogue ; l'âme disponible ; le palier.
  4. LA DÉCISION — lieu par lieu : appuis posés ou par l'âme, charges ponctuelles et leur niveau, type déduit de la
     géométrie, interaction avec la flexion, force transmise d'une semelle à l'autre, raidisseur.
  5. LA BARRE — le bloc dans le résultat du moteur : statut d'ensemble, efforts imposés, tubes, données, anciennes notes.
  6. SORTIE — chaque formule recalculée par son gabarit et relue sur ses nombres imprimés ; JSON strict.
  7. MUTATIONS — chaque altération plausible du texte fait rougir un contrôle.
  8. PURETÉ.

    /usr/bin/python3.12 tests/test_barre_ame.py
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

from acier.barre import ame as AME  # noqa: E402
from acier.barre import ec3_ame as AM  # noqa: E402
from acier.barre import ec3_section as ES  # noqa: E402
from acier.barre import ecriture as W  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import formules as FO  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import parametres as PA  # noqa: E402
from acier.barre import raidisseurs as RA  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.flambement import normes as NO  # noqa: E402
from acier.profiles import catalogue as CAT  # noqa: E402

OK, KO = [], []
FICHIERS = ("ec3_ame", "raidisseurs", "ame", "entree", "moteur")
CHEMINS = {k: os.path.join(RACINE, "acier", "barre", k + ".py") for k in FICHIERS}
R3 = math.sqrt(3.0)
E_ACIER = 210000.0


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:500]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def verif(r, id_):
    return next((v for v in r["verifications"] if v["id"] == id_), None)


def lieu(r, nom):
    return next((L for L in r["lieux"] if L["nom"] == nom), None)


def q(v, a, b):
    return {"type": "q", "v": v, "a": a, "b": b}


def P_(v, a, ss=None):
    return {"type": "P", "v": v, "a": a} if ss is None else {"type": "P", "v": v, "a": a, "ss": ss}


def barre(**k):
    b = EN.nouvelle_barre("B")
    b.update(k)
    return b


def brut(nom):
    return CAT.par_nom()[nom]


# ------------------------------------------------------------------ les témoins, écrits depuis le texte de la norme
def frd_temoin(h, b, tw, tf, fy, ss, type_, c=0.0, gM1=1.0):
    """F_Rd (kN) d'une âme sous une force concentrée — EN 1993-1-5 § 6, sans une fonction du moteur."""
    hw = h - 2.0 * tf
    ss = min(ss, hw)
    kF = {"a": 6.0, "b": 3.5}.get(type_) or min(2.0 + 6.0 * (ss + c) / hw, 6.0)
    Fcr = 0.9 * kF * E_ACIER * tw ** 3 / hw
    m1 = b / tw

    def ly(m2):
        l1 = ss + 2.0 * tf * (1.0 + math.sqrt(m1 + m2))
        if type_ != "c":
            return l1
        le = min(kF * E_ACIER * tw ** 2 / (2.0 * fy * hw), ss + c)
        return min(l1, le + tf * math.sqrt(m1 / 2.0 + (le / tf) ** 2 + m2), le + tf * math.sqrt(m1 + m2))

    longueur = ly(0.02 * (hw / tf) ** 2)
    lam = math.sqrt(longueur * tw * fy / Fcr)
    if lam <= 0.5:
        longueur = ly(0.0)
        lam = math.sqrt(longueur * tw * fy / Fcr)
    return fy * min(0.5 / lam, 1.0) * longueur * tw / gM1 / 1e3


def frd_profil(nom, fy, ss, type_, c=0.0):
    p = brut(nom)
    return frd_temoin(p["h"], p["b"], p["tw"], p["tf"], fy, ss, type_, c)


def raidisseur_temoin(nom, fy, ts, cotes=None):
    """Une paire de plats d'épaisseur ts dans un I du catalogue — EN 1993-1-5 § 9, à la main : (bs, chanfrein, N_c,Rd kN,
    A_st, I_st, N_b,Rd kN)."""
    p = brut(nom)
    hw = p["h"] - 2.0 * p["tf"]
    eps = math.sqrt(235.0 / fy)
    ch = 5.0 * math.ceil((p.get("r") or p["r1"]) / 5.0 - 1e-9)                # (un IPN porte son congé sous « r1 »)
    admis = min(14.0 * eps, math.sqrt(E_ACIER / (5.3 * fy) - 0.25))
    bs = min(5.0 * math.floor((p["b"] - p["tw"]) / 2.0 / 5.0 + 1e-9), 5.0 * math.floor(ts * admis / 5.0 + 1e-9))
    Nc = 2.0 * (bs - ch) * ts * fy / 1e3
    g, d = cotes if cotes is not None else (15.0 * eps * p["tw"], 15.0 * eps * p["tw"])
    lw = ts + g + d
    Ast = 2.0 * bs * ts + lw * p["tw"]
    Ist = (ts * (2.0 * bs + p["tw"]) ** 3 + (lw - ts) * p["tw"] ** 3) / 12.0
    lam = math.sqrt(Ast * fy / (math.pi ** 2 * E_ACIER * Ist / (0.75 * hw) ** 2))
    chi = 1.0
    if lam > 0.2:
        phi = 0.5 * (1.0 + 0.49 * (lam - 0.2) + lam ** 2)
        chi = min(1.0 / (phi + math.sqrt(phi ** 2 - lam ** 2)), 1.0)
    return bs, ch, Nc, Ast, Ist, chi * Ast * fy / 1e3


R2 = dict(poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "charges": [q(2.42, 0.0, 6.0), P_(10.0, 3.0, 100.0)]},
                                   {"nom": "Q", "nature": "exploitation_A", "charges": [P_(15.0, 3.0, 100.0)]}])
CONTINUE = dict(profil="HEA 200", modele="continue", travees=[5.0, 5.0], poids_propre=False,
                cas=[{"nom": "G", "nature": "permanent", "charges": [q(6.0, 0.0, 10.0)]},
                     {"nom": "Q1", "nature": "exploitation_B", "charges": [q(8.0, 0.0, 5.0)]},
                     {"nom": "Q2", "nature": "exploitation_B", "charges": [q(8.0, 5.0, 10.0)]}])
TOUT = "1,35·G + 1,5·Q1 + 1,5·Q2"
I_DU_CATALOGUE = [p["nom"] for p in CAT.par_nom().values() if SE.FAMILLES.get(p["famille"], ("",))[0] == "I"]


def _refuse(f):
    try:
        f()
    except RE.Refus as e:
        return bool(str(e))
    return False


# ------------------------------------------------------------------ les contrôles de référence (rejoués par les mutations)
def controles(M=None):
    M = M or {"ec3_ame": AM, "raidisseurs": RA, "ame": AME, "entree": EN, "moteur": MO}
    am, ra, mo, en = M["ec3_ame"], M["raidisseurs"], M["moteur"], M["entree"]
    C = []
    vus = []                                             # les résultats « ame » rencontrés : la sortie les relit tous

    def c(nom, cond, info=""):
        C.append((nom, bool(cond), info))

    def ame(**k):
        r = mo.verifier(barre(**k))
        vus.append(r["ame"])
        return r["ame"]

    Pm = PA.jeu()
    s300 = SE.du_catalogue("IPE 300")
    a300 = am.ame_de(s300)

    # ---- 1. R4 : la charge concentrée
    def R(ss, t, cc=0.0):
        return am.resistance(a300, 235.0, Pm, ss, t, cc)

    c("R4 : IPE 300 — h_w = 278,6 mm, m₁ = 21,13, F_cr du type (a) = 1 456,8 kN",
      proche(a300["hw"], 278.6) and round(R(50, "a")["m1"], 2) == 21.13 and round(R(50, "a")["Fcr"] / 1e3, 1) == 1456.8, a300)
    r = R(50, "a")
    c("R4 (a) s_s = 50 : λ̄_F = 0,441 ≤ 0,5 → m₂ = 0, l_y = 169,8 mm, χ_F = 1, F_Rd = 283,2 kN",
      r["m2"] == 0.0 and round(r["ly"], 1) == 169.8 and round(r["lambda_F"], 3) == 0.441 and r["chi_F"] == 1.0 and round(r["FRd"] / 1e3, 1) == 283.2, r)
    r = R(100, "a")
    c("R4 (a) s_s = 100 : m₂ = 13,56, l_y = 247,4 mm, λ̄_F = 0,532, χ_F = 0,939, F_Rd = 387,8 kN",
      round(r["m2"], 2) == 13.56 and round(r["ly"], 1) == 247.4 and round(r["lambda_F"], 3) == 0.532 and round(r["chi_F"], 3) == 0.939
      and round(r["FRd"] / 1e3, 1) == 387.8, r)
    r = R(150, "a")
    c("R4 (a) s_s = 150 : l_y = 297,4 mm, λ̄_F = 0,584, χ_F = 0,857, F_Rd = 425,1 kN",
      round(r["ly"], 1) == 297.4 and round(r["lambda_F"], 3) == 0.584 and round(r["chi_F"], 3) == 0.857 and round(r["FRd"] / 1e3, 1) == 425.1, r)
    r = R(100, "c", 0.0)
    c("R4 (c) s_s = 100, c = 0 : k_F = 4,154, F_cr = 1 008,5 kN, l_e = 100 (plafonné à s_s + c), l_y = 163,0 mm, λ̄_F = 0,519, χ_F = 0,963, "
      "F_Rd = 261,9 kN",
      round(r["kF"], 3) == 4.154 and round(r["Fcr"] / 1e3, 1) == 1008.5 and r["le"] == 100.0 and r["le_plafonne"] and round(r["ly"], 1) == 163.0
      and round(r["lambda_F"], 3) == 0.519 and round(r["chi_F"], 3) == 0.963 and round(r["FRd"] / 1e3, 1) == 261.9, r)
    r = R(100, "c", 50.0)
    c("R4 (c) s_s = 100, c = 50 : k_F = 5,230, F_cr = 1 270,0 kN, l_e = 150, l_y = 213,0 mm, λ̄_F = 0,529, χ_F = 0,945, F_Rd = 335,9 kN",
      round(r["kF"], 2) == 5.23 and round(r["Fcr"] / 1e3, 1) == 1270.0 and r["le"] == 150.0 and round(r["ly"], 1) == 213.0
      and round(r["lambda_F"], 3) == 0.529 and round(r["chi_F"], 3) == 0.945 and round(r["FRd"] / 1e3, 1) == 335.9, r)
    r = R(100, "c", 50.0)
    c("type (c) : l_y est le plus petit de ses TROIS expressions (ici la troisième : 213,0 < 247,4 < 308,9)",
      r["ly"] == min(r["ly1"], r["ly2"], r["ly3"]) and r["ly"] == r["ly3"] and [round(r[k], 1) for k in ("ly1", "ly2", "ly3")] == [247.4, 308.9, 213.0],
      (r["ly1"], r["ly2"], r["ly3"]))
    sK = SE.du_catalogue("IPE 750x147")
    r = am.resistance(am.ame_de(sK), 355.0, Pm, 200.0, "c", 200.0)
    kF = 2.0 + 6.0 * 400.0 / 719.0
    c("type (c) sur une âme haute (IPE 750x147 S355, s_s = c = 200 mm) : l_e = k_F·E·t_w²/(2·f_y·h_w) = 382,6 mm < s_s + c — il n'est PAS "
      "plafonné ; ici la première expression de l_y gouverne",
      not r["le_plafonne"] and proche(r["le"], kF * E_ACIER * 13.2 ** 2 / (2.0 * 355.0 * 719.0)) and round(r["le"], 1) == 382.6
      and r["ly"] == r["ly1"] and proche(r["FRd"] / 1e3, frd_profil("IPE 750x147", 355.0, 200.0, "c", 200.0)), r)
    ecarts = []
    for nom in I_DU_CATALOGUE:
        s = SE.du_catalogue(nom)
        a = am.ame_de(s)
        for fy in (235.0, 355.0):
            for ss, t, cc in ((0, "a", 0), (50, "a", 0), (150, "a", 0), (400, "a", 0), (5000, "a", 0), (0, "b", 0), (120, "b", 0), (0, "c", 0),
                              (100, "c", 0), (100, "c", 30), (60, "c", 500), (5000, "c", 20)):
                if not proche(am.resistance(a, fy, Pm, ss, t, cc)["FRd"] / 1e3, frd_profil(nom, fy, ss, t, cc), 1e-12):
                    ecarts.append((nom, fy, ss, t, cc))
    c(f"le témoin écrit ici depuis la norme = le moteur, sur les {len(I_DU_CATALOGUE)} I et H du catalogue, deux nuances, douze charges "
      "(types a, b, c ; s_s de 0 à plus que h_w ; c de 0 à 500)", len(I_DU_CATALOGUE) > 150 and not ecarts, ecarts[:4])
    c("s_s est bornée à h_w (§ 6.3(1)) : 5 000 mm saisis ne valent que 278,6",
      R(5000, "a")["ss"] == a300["hw"] and R(5000, "a")["FRd"] == R(278.6, "a")["FRd"] and R(-5, "a")["ss"] == 0.0)
    c("type (c) loin du bout : k_F plafonné à 6 et la première expression de l_y — la résistance du type (a)",
      R(100, "c", 1e4)["kF"] == 6.0 and R(100, "c", 1e4)["kF_plafonne"] and proche(R(100, "c", 1e4)["FRd"], R(100, "a")["FRd"]))
    c("la force transmise d'une semelle à l'autre (b, k_F = 3,5) résiste moins que la force reprise par cisaillement (a, k_F = 6) dès que "
      "χ_F < 1 ; près du bout (c), moins encore",
      R(100, "b")["kF"] == 3.5 and R(100, "b")["FRd"] < R(100, "a")["FRd"] and R(100, "c", 0)["FRd"] < R(100, "b")["FRd"])
    c("F_Rd croît avec la longueur d'appui (0, 50, 100, 150, 250 mm)",
      all(R(u, "a")["FRd"] < R(v, "a")["FRd"] for u, v in ((0, 50), (50, 100), (100, 150), (150, 250))))
    try:
        am.resistance(a300, 235.0, Pm, 100, "d")
        c("un type inconnu est refusé", False)
    except ValueError:
        c("un type inconnu est refusé", True)
    c("η₁ élastique : |N|/(f_y·A) + |M|/(f_y·W_el) — 100 kN et 50 kN·m sur l'IPE 300 : 0,079 + 0,382",
      proche(am.eta1(100e3, 50e6, s300["A"], s300["Wel_y"], 235.0, 1.0), 100e3 / (s300["A"] * 235.0) + 50e6 / (s300["Wel_y"] * 235.0))
      and proche(am.eta1(-100e3, -50e6, s300["A"], s300["Wel_y"], 235.0, 1.0), am.eta1(100e3, 50e6, s300["A"], s300["Wel_y"], 235.0, 1.0))
      and proche(am.interaction(0.6, 0.5), (0.6 + 0.8 * 0.5) / 1.4))

    # ---- 2. le voilement par cisaillement
    s750 = SE.du_catalogue("IPE 750x147")
    p750 = brut("IPE 750x147")
    hw, tw = p750["h"] - 2 * p750["tf"], p750["tw"]
    eps355 = math.sqrt(235.0 / 355.0)
    e = am.elancement(s750, 355.0, Pm)
    c("IPE 750x147 S355 : h_w/t_w = 54,5 > 72·ε/η = 48,8 — âme élancée ; en S235 (limite 60,0) elle ne l'est pas ; l'IPE 300 jamais",
      e["elancee"] and round(e["rapport"], 1) == 54.5 and round(e["limite"], 1) == 48.8 and not am.elancement(s750, 235.0, Pm)["elancee"]
      and not am.elancement(s300, 355.0, Pm)["elancee"], e)
    lw = hw / (86.4 * tw * eps355)
    Vbw = 0.83 / lw * 355.0 * hw * tw / R3 / 1e3
    J = RE.Journal("ame", "x")
    am.cisaillement(J, s750, 355.0, Pm, 500e3)
    v = verif(J.rendre(), "A-Vb")
    c("… λ̄_w = h_w/(86,4·t·ε) = 0,775, χ_w = 0,83/λ̄_w = 1,071, V_bw,Rd = 1,071·355·719·13,2/√3 = 2 083,7 kN ; 500 kN : 24,0 %",
      v is not None and round(lw, 3) == 0.775 and round(Vbw, 1) == 2083.7 and round(0.83 / lw, 3) == 1.071 and proche(v["limite"], Vbw) and proche(v["valeur"], 500.0)
      and v["statut"] == RE.OK and proche(v["taux"], 500.0 / Vbw) and proche(J.G["voilement_cisaillement"]["chi_w"], 0.83 / lw), v)
    c("V_bw,Rd < V_pl,Rd (2 334 kN, avec le plancher η·h_w·t_w) : le voilement réduit bien la résistance au cisaillement",
      Vbw < 1.2 * hw * tw * 355.0 / R3 / 1e3)
    vides = []
    for sec, fy, V in ((s300, 355.0, 300e3), (s750, 235.0, 500e3), (s750, 355.0, 0.0)):
        J = RE.Journal("ame", "x")
        am.cisaillement(J, sec, fy, Pm, V)
        vides.append(len(J.verifs))
    c("rien n'est écrit pour une âme qui n'est pas élancée, ni sans effort tranchant", vides == [0, 0, 0], vides)
    Mf = p750["b"] * p750["tf"] * (p750["h"] - p750["tf"]) * 355.0 / 1e6
    Mpl = p750["Wpl_y"] * 1e3 * 355.0 / 1e6
    J = RE.Journal("ame", "x")
    am.cisaillement(J, s750, 355.0, Pm, 1200e3, 1500e6)
    v = verif(J.rendre(), "A-VbM")
    attendu = 1500.0 / Mpl + (1.0 - Mf / Mpl) * (2.0 * 1200.0 / Vbw - 1.0) ** 2
    c("interaction (7.1), V_Ed = 1 200 kN > 0,5·V_bw,Rd et M_Ed = 1 500 kN·m > M_f,Rd : η̄₁ + (1 − M_f,Rd/M_pl,Rd)·(2·η̄₃ − 1)² à la main",
      v is not None and proche(v["valeur"], attendu) and proche(v["taux"], attendu) and v["limite"] == 1.0 and 0.8 < attendu < 0.9
      and round(Mf, 0) == round(265.0 * 17.0 * 736.0 * 355.0 / 1e6, 0), (v, attendu, Mf, Mpl))
    sans = []
    for V, Mx, Nx in ((1000e3, 1500e6, 0.0), (1200e3, 1000e6, 0.0), (1200e3, 0.0, 0.0)):
        J = RE.Journal("ame", "x")
        am.cisaillement(J, s750, 355.0, Pm, V, Mx, Nx)
        sans.append([x["id"] for x in J.verifs])
    c("… sans objet tant que V_Ed ≤ 0,5·V_bw,Rd, ou que M_Ed ≤ M_f,Rd (les semelles seules portent le moment), ou sans moment",
      sans == [["A-Vb"], ["A-Vb"], ["A-Vb"]], sans)
    J = RE.Journal("ame", "x")
    am.cisaillement(J, s750, 355.0, Pm, 1200e3, 1500e6, 300e3)
    v = verif(J.rendre(), "A-VbM")
    c("… avec un effort normal : « non couvert », et dit — jamais « ok »",
      v is not None and v["statut"] == RE.NON_COUVERT and v["taux"] is None and J.alertes and "non vérifiée" in J.alertes[0]["message"], v)
    tube = SE.du_catalogue("RHS 350x150x5")
    J = RE.Journal("ame", "x")
    am.cisaillement(J, tube, 235.0, Pm, 200e3)
    am.cisaillement(J, tube, 235.0, Pm, 200e3, direction="y")
    r = J.rendre()
    hwt = 350.0 - 3.0 * 5.0
    Vt = 2.0 * 0.83 / (hwt / (86.4 * 5.0)) * 235.0 * hwt * 5.0 / R3 / 1e3
    c("un tube RHS 350x150x5 S235 : (h − 3t)/t = 67,0 > 60 — ses DEUX âmes portent V_z (n = 2) ; sous V_y, les parois de 150 ne sont pas élancées",
      [x["id"] for x in r["verifications"]] == ["A-Vb"] and proche(verif(r, "A-Vb")["limite"], Vt)
      and any(f["gabarit"] == "a_V_bw_n" for f in r["calcul"]) and am.elancement(tube, 235.0, Pm, "y")["elancee"] is False, r["verifications"])
    c("ni tube rond, ni cornière : la question ne se pose pas",
      am.elancement(SE.du_catalogue("CHS 219.1x6.3"), 235.0, Pm) is None and am.elancement(SE.du_catalogue("L 100x100x10"), 235.0, Pm) is None
      and am.ame_de(tube) is None)
    sm = am.semelle(s300, 235.0, Pm)
    c("voilement de l'âme induit par la semelle (§ 8), k = 0,4 : IPE 300 — h_w/t_w = 39,2 ≤ 0,4·(E/f_y)·√(A_w/A_fc) = 397",
      sm["ok"] and round(sm["rapport"], 1) == 39.2 and proche(sm["limite"], 0.4 * E_ACIER / 235.0 * math.sqrt(278.6 * 7.1 / (150.0 * 10.7)))
      and round(sm["limite"]) == 397 and am.semelle(tube, 235.0, Pm) is None, sm)
    c("… sur tout le catalogue et jusqu'au S460, aucun I laminé ne le dépasse",
      all(am.semelle(SE.du_catalogue(n_), 460.0, Pm)["ok"] for n_ in I_DU_CATALOGUE))

    # ---- 3. le raidisseur
    p = ra.dimensionner(s300, 235.0, Pm, 300e3)
    bs, ch, Nc, Ast, Ist, Nb = raidisseur_temoin("IPE 300", 235.0, 12.0)
    c("IPE 300, 300 kN : 2 plats 70 × 12 (8 mm : 206,8 kN ; 10 mm : 258,5 kN ; 12 mm : 310,2 kN), chanfrein 15 — écrasement 96,7 %",
      p["ok"] and (p["ts"], p["bs"], p["chanfrein"]) == (12.0, 70.0, 15.0) and proche(p["NcRd"] / 1e3, 310.2) and proche(p["NcRd"] / 1e3, Nc)
      and proche(p["taux"], 300.0 / 310.2) and ra.designation(p) == "2 plats 70 × 12 mm", p)
    c("… flambement : A_st = 2·70·12 + (12 + 30·7,1)·7,1 = 3 277,5 mm², I_st, λ̄ = 0,071 ≤ 0,2 → χ = 1, N_b,Rd = 770,2 kN",
      proche(p["Ast"], 3277.5) and proche(p["Ast"], Ast) and proche(p["Ist"], Ist) and p["Phi"] is None and p["chi"] == 1.0
      and proche(p["NbRd"] / 1e3, Nb) and round(p["NbRd"] / 1e3, 1) == 770.2 and round(p["lambda"], 3) == 0.071, p)
    c("la première épaisseur qui porte : 200 kN → 8 mm ; 207 kN → 10 mm ; 259 kN → 12 mm ; 400 kN → 20 mm",
      [ra.dimensionner(s300, 235.0, Pm, F * 1e3)["ts"] for F in (200, 207, 259, 400)] == [8.0, 10.0, 12.0, 20.0])
    p = ra.dimensionner(s300, 235.0, Pm, 600e3)
    c("aucun plat ne porte 600 kN (20 mm : 517 kN) : le plus épais est rendu, « ok » faux, taux 116 %",
      p["ts"] == 20.0 and not p["ok"] and proche(p["taux"], 600.0 / 517.0), p)
    sH = SE.du_catalogue("HEB 300")
    c("un plat trop élancé est RÉTRÉCI : HEB 300 S235 (demi-semelle 140) — 100 mm pour 8, 125 pour 10, 140 dès 12 ; en S355 : 80, 105, 125, 140",
      [ra.plat(sH, 235.0, Pm, t)["bs"] for t in ra.EPAISSEURS] == [100.0, 125.0, 140.0, 140.0, 140.0]
      and [ra.plat(sH, 355.0, Pm, t)["bs"] for t in ra.EPAISSEURS] == [80.0, 105.0, 125.0, 140.0, 140.0]
      and proche(ra.elancement_admis(235.0, E_ACIER), math.sqrt(E_ACIER / (5.3 * 235.0) - 0.25))
      and proche(ra.elancement_admis(460.0, E_ACIER), math.sqrt(E_ACIER / (5.3 * 460.0) - 0.25)), [ra.plat(sH, 355.0, Pm, t)["bs"] for t in ra.EPAISSEURS])
    fautes, n = [], 0
    for nom in I_DU_CATALOGUE:
        s = SE.du_catalogue(nom)
        pr = brut(nom)
        for fy in (235.0, 355.0, 460.0):
            for ts in ra.EPAISSEURS:
                pl = ra.plat(s, fy, Pm, ts)
                if pl is None:
                    continue
                n += 1
                t_, b_ = pl["ts"], pl["bs"]
                if not (b_ / t_ <= 14.0 * math.sqrt(235.0 / fy) + 1e-9 and (b_ * t_ ** 3 / 3.0) / (b_ ** 3 * t_ / 3.0 + b_ * t_ ** 3 / 12.0) >= 5.3 * fy / E_ACIER - 1e-12
                        and b_ <= (pr["b"] - pr["tw"]) / 2.0 + 1e-9 and pl["chanfrein"] >= (pr.get("r") or pr["r1"]) - 1e-9 and b_ - pl["chanfrein"] >= 5.0
                        and b_ % 5.0 == 0 and proche(pl["torsion"], t_ ** 2 / (b_ ** 2 + t_ ** 2 / 4.0)) and pl["elancement"] <= pl["elancement_lim"] + 1e-9
                        and pl["torsion"] >= pl["torsion_lim"] - 1e-12 and proche(pl["elancement_lim"], 14.0 * math.sqrt(235.0 / fy))
                        and proche(pl["torsion_lim"], 5.3 * fy / E_ACIER)):
                    fautes.append((nom, fy, ts, b_))
    c(f"sur tout le catalogue ({n} plats : chaque I, trois nuances, cinq épaisseurs) : b_s/t_s ≤ 14·ε, I_T/I_p ≥ 5,3·f_y/E, le plat ne dépasse "
      "pas la semelle, le chanfrein dégage le congé, il reste 5 mm de contact au moins, largeur au pas de 5 mm", n > 2000 and not fautes, fautes[:4])
    pl = ra.plat(s300, 235.0, Pm, 10.0, (0.0, None))
    pl2 = ra.plat(s300, 235.0, Pm, 10.0, (50.0, 60.0))
    c("l'âme participante est bornée par l'âme disponible : en bout de barre (0 mm d'un côté) l_w = t_s + 15·ε·t_w ; à 50 et 60 mm de deux "
      "voisins, l_w = t_s + 45 + 55",
      pl["limite"] and pl["cotes"] == [0.0, 106.5] and proche(pl["lw"], 116.5) and pl2["cotes"] == [45.0, 55.0] and proche(pl2["lw"], 110.0)
      and not ra.plat(s300, 235.0, Pm, 10.0)["limite"] and proche(ra.plat(s300, 235.0, Pm, 10.0)["lw"], 10.0 + 30.0 * 7.1), (pl["cotes"], pl2["cotes"]))
    haut = SE.saisie("I", "lamine", h=1500.0, b=100.0, tw=8.0, tf=20.0, r=0.0, A=15680.0, Iy=4.0e9, Iz=3.4e6, iy=505.0, iz=14.7)
    pl = ra.plat(haut, 355.0, Pm, 8.0)
    lam = math.sqrt(pl["Ast"] * 355.0 / (math.pi ** 2 * E_ACIER * pl["Ist"] / (0.75 * 1460.0) ** 2))
    phi = 0.5 * (1.0 + 0.49 * (lam - 0.2) + lam ** 2)
    c("au-delà du palier (une âme de 1 460 mm, des plats de 45 × 8) : λ̄ = 0,871, courbe c — Φ = 1,043, χ = 0,618, N_b,Rd = χ·A_st·f_y",
      pl["Phi"] is not None and proche(pl["lambda"], lam) and proche(pl["Phi"], phi) and proche(pl["chi"], 1.0 / (phi + math.sqrt(phi ** 2 - lam ** 2)))
      and proche(pl["NbRd"], pl["chi"] * pl["Ast"] * 355.0) and (round(lam, 3), round(phi, 3), round(pl["chi"], 3)) == (0.871, 1.043, 0.618),
      (lam, phi, pl["chi"]))
    c("appui rigide d'une âme élancée : I_st ≥ 0,75·h_w·t_w³ (IPE 750x147 : 124 cm⁴ demandés) ; un tube ne reçoit pas de raidisseur",
      ra.dimensionner(s750, 355.0, Pm, 100e3, appui_rigide=True)["rigide"] is True and round(ra.plat(s750, 355.0, Pm, 8.0)["I_min"] / 1e4) == 124
      and ra.dimensionner(s750, 355.0, Pm, 100e3)["rigide"] is None and ra.gabarit(tube) is None and ra.dimensionner(tube, 235.0, Pm, 1e3) is None)

    # ---- 4. la décision, lieu par lieu
    g300 = brut("IPE 300")["masse"] * 9.81 / 1000.0
    a = ame()
    Fed = (1.35 * (5.0 + g300) + 1.5 * 5.0) * 3.0
    c("la barre neuve (IPE 300, 6 m, G + Q, posée, s_s non saisie) : deux lieux — Appui A, Appui B —, type (c), F_Ed = 44,4 kN ≤ F_Rd = 58,0 kN : "
      "77 %, pas de raidisseur",
      [L["nom"] for L in a["lieux"]] == ["Appui A", "Appui B"] and all(L["type"] == "c" and L["decision"] == "aucun" and L["raidisseur"] is None
                                                                      and L["ss"] == 0.0 for L in a["lieux"])
      and proche(a["lieux"][0]["F_Ed"], Fed) and proche(a["lieux"][0]["sans"]["F_Rd"], frd_profil("IPE 300", 235.0, 0, "c", 0))
      and proche(a["taux_max"], Fed / frd_profil("IPE 300", 235.0, 0, "c", 0)) and round(a["taux_max"], 2) == 0.77 and a["statut"] == RE.OK
      and [v["id"] for v in a["verifications"]] == ["A-L00-F", "A-L01-F"] and a["combinaison"] == "1,35·Poids propre + 1,35·G + 1,5·Q", a["lieux"])
    c("… la longueur d'appui non saisie est DITE (hypothèses, et message de la vérification)",
      any("non saisie" in h for h in a["hypotheses"]) and any("non saisie" in m for m in verif(a, "A-L00-F")["messages"]), a["hypotheses"])
    a = ame(appuis={"mode": "poses", "ss": 100.0, "c": 0.0})
    b_ = ame(appuis={"mode": "poses", "ss": 100.0, "c": 50.0})
    c("s_s = 100 mm aux appuis : F_Rd = 261,9 kN (R4, type c, c = 0) ; avec 50 mm de débord : 335,9 kN",
      round(a["lieux"][0]["sans"]["F_Rd"], 1) == 261.9 and round(b_["lieux"][0]["sans"]["F_Rd"], 1) == 335.9 and a["lieux"][0]["ss"] == 100.0
      and not any("non saisie" in h for h in a["hypotheses"]), (a["lieux"][0], b_["lieux"][0]))
    a = ame(appuis={"mode": "poses", "ss": 100.0, "c": 400.0})
    c("un débord assez long (s_s + c ≥ 2·h_w/3) : la réaction de rive n'est plus « près du bout » — type (a)",
      a["lieux"][0]["type"] == "a" and proche(a["lieux"][0]["sans"]["F_Rd"], frd_profil("IPE 300", 235.0, 100, "a")), a["lieux"][0])
    a = ame(**R2)
    L = lieu(a, "Charge à 3,00 m")
    eta1 = 68.70e6 / (brut("IPE 300")["Wel_y"] * 1e3 * 235.0)
    v = verif(a, L["base"] + "-FM") if L else None
    c("R2/R4 : la charge de 36 kN à mi-portée (s_s = 100 mm, G et Q à la même abscisse : UNE force), type (a) — η₂ = 36/387,8 = 0,093 : "
      "pas de raidisseur",
      L is not None and L["genre"] == "charge" and L["type"] == "a" and proche(L["F_Ed"], 36.0) and round(L["sans"]["F_Rd"], 1) == 387.8
      and round(L["sans"]["eta2"], 3) == 0.093 and L["decision"] == "aucun" and len([x for x in a["lieux"] if x["genre"] == "charge"]) == 1, L)
    c("… avec la flexion (7.2) : η₁ = M_Ed/(W_el,y·f_y) = 68,70/130,9 = 0,525 (élastique) ; (η₂ + 0,8·η₁)/1,4 = 36,6 % — c'est elle qui gouverne "
      "le lieu",
      v is not None and proche(v["valeur"], (36.0 / L["sans"]["F_Rd"] + 0.8 * eta1) / 1.4, 1e-4) and v["limite"] == 1.0 and round(v["taux"], 3) == 0.366
      and v["critere"] == "(η₂ + 0,8·η₁)/1,4 ≤ 1"
      and L["gouvernant"] == v["id"] and proche(L["taux"], v["taux"]) and proche(L["sans"]["interaction"], v["taux"]) and round(eta1, 3) == 0.525, v)
    c("… aux appuis, sans moment : pas d'interaction à vérifier (une seule vérification par appui)",
      [x["id"] for x in a["verifications"]] == ["A-L00-F", "A-L01-F", "A-L01-FM", "A-L02-F"], [x["id"] for x in a["verifications"]])
    a = ame(**CONTINUE)
    L = lieu(a, "Appui B")
    pH = brut("HEA 200")
    RB, MB = 1.25 * 20.1 * 5.0, 20.1 * 25.0 / 8.0
    FRd = frd_profil("HEA 200", 235.0, 0, "a")
    attendu = (RB / FRd + 0.8 * MB * 1e6 / (pH["Wel_y"] * 1e3 * 235.0)) / 1.4
    c("poutre continue HEA 200, 2 × 5 m : l'appui du milieu est de type (a) ; R_B = 1,25·q·L = 125,6 kN et M_B = q·L²/8 = 62,8 kN·m sous les "
      "deux travées chargées — η₂ = 62,8 %, mais (η₂ + 0,8·η₁)/1,4 = 84,2 % : l'interaction gouverne",
      L is not None and L["type"] == "a" and proche(L["F_Ed"], RB) and proche(L["sans"]["F_Rd"], FRd) and proche(L["sans"]["eta2"], RB / FRd)
      and proche(L["taux"], attendu, 1e-6) and L["gouvernant"] == "A-L01-FM" and verif(a, "A-L01-FM")["combinaison"] == TOUT
      and round(attendu, 3) == 0.842 and L["decision"] == "aucun" and [x["type"] for x in a["lieux"]] == ["c", "a", "c"], (L, attendu))
    plus = copy.deepcopy(CONTINUE)
    plus["cas"][0]["charges"].append(P_(50.0, 5.0))
    a = ame(**plus)
    L = lieu(a, "Appui B")
    FRb = frd_profil("HEA 200", 235.0, 0, "b")
    bs, ch, Nc, Ast, Ist, Nb = raidisseur_temoin("HEA 200", 235.0, 8.0)
    vR, vB = verif(a, "A-L01-R"), verif(a, "A-L01-Rb")
    c("… avec une charge de 50 kN posée AU DROIT de l'appui du milieu : un seul lieu ; la réaction (193,1 kN) passe seule (96,6 %), mais avec la "
      "flexion (η₂ + 0,8·η₁ = 1,52 > 1,4) il faut un raidisseur — et la raison le dit",
      L is not None and len(a["lieux"]) == 3 and proche(L["F_Ed"], RB + 67.5) and L["decision"] == "raidisseur" and "η₂ + 0,8·η₁" in L["raison"]
      and "1,52" in L["raison"] and L["sans"]["eta2"] < 1.0 < L["sans"]["interaction"], L)
    c("… le raidisseur : 2 plats 95 × 8 mm, chanfrein 20 — écrasement 193,1 kN ≤ 282,0 kN, flambement ≤ N_b,Rd ; l'âme seule n'est plus jugée",
      L["raidisseur"] == {"bs": 95.0, "ts": 8.0, "chanfrein": 20.0, "designation": "2 plats 95 × 8 mm"} and vR is not None and vB is not None
      and proche(vR["valeur"], RB + 67.5) and proche(vR["limite"], Nc) and round(Nc, 1) == 282.0 and proche(vB["limite"], Nb)
      and L["verifications"] == ["A-L01-R", "A-L01-Rb"] and verif(a, "A-L01-F") is None and verif(a, "A-L01-FM") is None
      and a["statut"] == RE.OK and any("soudures" in h for h in a["hypotheses"]) and any("Sans raidisseur" in m for m in vR["messages"]), (L, vR))
    moins = copy.deepcopy(CONTINUE)
    moins["cas"][0]["charges"].append(P_(20.0, 5.0))
    a = ame(**moins)
    vB = verif(a, "A-L01-B")
    c("… une charge plus petite (27 kN de calcul) : pas de raidisseur — la force qui passe d'une semelle à l'autre est vérifiée en type (b), "
      "k_F = 3,5",
      vB is not None and proche(vB["valeur"], 27.0) and proche(vB["limite"], FRb) and lieu(a, "Appui B")["decision"] == "aucun"
      and any("Type (b)" in m for m in vB["messages"]) and verif(a, "A-L01-F") is not None, vB)
    lourd = dict(profil="HEB 300", travees=[4.0], appuis={"mode": "poses", "ss": 200.0, "c": 0.0},
                 cas=[{"nom": "G", "nature": "permanent", "charges": [P_(500.0, 2.0)]}])
    a = ame(**lourd)
    L = lieu(a, "Charge à 2,00 m")
    bs, ch, Nc, Ast, Ist, Nb = raidisseur_temoin("HEB 300", 235.0, 15.0)
    c("HEB 300, 675 kN à mi-portée (s_s non saisie) : F_Rd = 611,2 kN seulement — raidisseur 2 plats 140 × 15 mm, écrasement 675/775,5 = 87,0 %",
      L is not None and L["decision"] == "raidisseur" and round(L["sans"]["F_Rd"], 1) == 611.2 and "F_Ed = 675,0 kN > F_Rd = 611,2 kN" in L["raison"]
      and L["raidisseur"]["designation"] == "2 plats 140 × 15 mm" and proche(verif(a, L["base"] + "-R")["limite"], Nc) and round(Nc, 1) == 775.5
      and proche(L["taux"], 675.0 / 775.5) and proche(verif(a, L["base"] + "-Rb")["limite"], Nb) and (bs, ch) == (140.0, 30.0), L)
    c("… aux appuis (s_s = 200 mm, 337,5 kN + le poids propre) : l'âme suffit, pas de raidisseur",
      all(x["decision"] == "aucun" and x["statut"] == RE.OK for x in a["lieux"] if x["genre"] == "appui") and a["statut"] == RE.OK, a["lieux"])
    a = ame(appuis={"mode": "ame", "ss": 0.0, "c": 0.0})
    c("appuis PAR L'ÂME (platine, cornières) : aucune force concentrée aux appuis — aucun lieu, rien à vérifier, et l'hypothèse le dit",
      a["lieux"] == [] and a["verifications"] == [] and a["statut"] == RE.OK and a["taux_max"] is None
      and any("par l'âme" in h for h in a["hypotheses"]), a["hypotheses"])
    lieux_niveau = {}
    for niv, val in (("haut", 30.0), ("centre", 30.0), ("bas", 30.0), ("bas", -30.0), ("haut", -30.0)):
        x = ame(niveau=niv, appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, cas=[{"nom": "G", "nature": "permanent", "charges": [P_(val, 2.0, 60.0)]}])
        lieux_niveau[(niv, val)] = [L["nom"] for L in x["lieux"]]
    c("une charge ponctuelle n'appuie sur l'âme que si elle POUSSE la semelle où elle est posée : vers le bas sur la semelle haute, vers le haut "
      "sous la semelle basse ; au centre de cisaillement elle entre par l'âme",
      lieux_niveau == {("haut", 30.0): ["Charge à 2,00 m"], ("centre", 30.0): [], ("bas", 30.0): [], ("bas", -30.0): ["Charge à 2,00 m"],
                       ("haut", -30.0): []}, lieux_niveau)
    a = ame(modele="console", travees=[2.0], cas=[{"nom": "G", "nature": "permanent", "charges": [P_(20.0, 2.0, 80.0), P_(20.0, 1.0, 80.0)]}])
    c("console : l'encastrement n'est pas un lieu (un assemblage, hors du module) ; la charge au bout libre est « près d'une extrémité » — type "
      "(c), c = 0 ; celle du milieu, type (a)",
      [(L["nom"], L["type"]) for L in a["lieux"]] == [("Charge à 1,00 m", "a"), ("Charge à 2,00 m", "c")]
      and proche(lieu(a, "Charge à 2,00 m")["sans"]["F_Rd"], frd_profil("IPE 300", 235.0, 80, "c", 0)), a["lieux"])
    a = ame(modele="enc_app")
    c("encastrée – appuyée : seul l'appui simple B est un lieu", [L["nom"] for L in a["lieux"]] == ["Appui B"], a["lieux"])
    a = ame(modele="continue", travees=[3.0, 6.0, 3.0], poids_propre=False, cas=[{"nom": "G", "nature": "permanent", "charges": [q(20.0, 3.0, 9.0)]}])
    c("poutre continue chargée sur sa seule travée centrale : les appuis de rive se SOULÈVENT — pas de force sur l'âme, et un avertissement "
      "(ancrage hors du module)",
      [L["nom"] for L in a["lieux"]] == ["Appui B", "Appui C"] and len([x for x in a["alertes"] if "soulèvement" in x["message"]]) == 2, a["alertes"])
    a = ame(cas=[{"nom": "G", "nature": "permanent", "charges": [P_(10.0, 2.0, 120.0)]}, {"nom": "Q", "nature": "exploitation_A", "charges": [P_(20.0, 2.0, 40.0)]},
                 {"nom": "S", "nature": "neige", "charges": [P_(5.0, 4.0, 0.0)]}])
    L = lieu(a, "Charge à 2,00 m")
    c("deux cas à la même abscisse : UNE force (1,35·10 + 1,5·20 = 43,5 kN), la plus courte des deux longueurs d'appui (40 mm) ; une autre "
      "abscisse, un autre lieu",
      L is not None and proche(L["F_Ed"], 43.5) and L["ss"] == 40.0 and lieu(a, "Charge à 4,00 m") is not None
      and proche(lieu(a, "Charge à 4,00 m")["F_Ed"], 7.5), L)
    pres = {d_: ame(appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, cas=[{"nom": "G", "nature": "permanent", "charges": [P_(20.0, 3.0, 50.0), P_(20.0, 3.0 + d_, 50.0)]}])
            for d_ in (0.10, 0.16, 0.18, 0.50)}
    ly50 = am.resistance(a300, 235.0, Pm, 50.0, "a")["ly"]
    c("deux charges RAPPROCHÉES sur la même semelle — leurs longueurs chargées l_y (169,8 mm) se recouvrent : à 100 et 160 mm d'entraxe — "
      "devraient être vérifiées aussi ensemble (§ 6.3(2)) : le bloc est « non couvert », et le dit ; à 180 et 500 mm, plus rien",
      round(ly50, 1) == 169.8 and all(verif(pres[d_], "A-GR") is not None and verif(pres[d_], "A-GR")["statut"] == RE.NON_COUVERT
                                       and pres[d_]["statut"] == RE.NON_COUVERT and any("ENSEMBLE" in x["message"] for x in pres[d_]["alertes"])
                                       for d_ in (0.10, 0.16))
      and all(verif(pres[d_], "A-GR") is None and pres[d_]["statut"] == RE.OK for d_ in (0.18, 0.50))
      and "charge à 3,00 m et charge à 3,10 m" in pres[0.10]["alertes"][0]["message"], {d_: pres[d_]["statut"] for d_ in pres})
    a = ame(modele="enc_app")
    c("un encastrement : l'hypothèse dit qu'il n'est pas vérifié ; une console n'a pas d'appui posé — pas de phrase sur la longueur d'appui",
      any("Encastrement" in h for h in a["hypotheses"]) and any("posée sur ses appuis" in h for h in a["hypotheses"])
      and not any("posée sur ses appuis" in h for h in ame(modele="console", travees=[2.0], cas=[{"nom": "G", "nature": "permanent",
                                                                                           "charges": [q(5.0, 0.0, 2.0)]}])["hypotheses"])
      and not any("Encastrement" in h for h in ame()["hypotheses"]))
    a = ame(profil="IPE 750x147", nuance="S355", travees=[8.0], maintiens=[{"x": 2.0}, {"x": 4.0}, {"x": 6.0}],
            cas=[{"nom": "G", "nature": "permanent", "charges": [q(20.0, 0.0, 8.0)]}])
    g750 = p750["masse"] * 9.81 / 1000.0
    Ved = 1.35 * (20.0 + g750) * 4.0
    v = verif(a, "A-Vb")
    c("IPE 750x147 S355, 8 m, 20 kN/m : l'âme élancée est VÉRIFIÉE au cisaillement (V_Ed = 115,8 kN ≤ V_bw,Rd = 2 083,7 kN), à l'appui",
      v is not None and proche(v["valeur"], Ved) and proche(v["limite"], Vbw) and v["statut"] == RE.OK and v["x"] in (0.0, 8.0), v)
    c("… et des raidisseurs sont posés à ses deux appuis (§ 5.1(2)), bien que l'âme seule y porte la réaction ; ils ont la rigidité d'un appui",
      all(L["decision"] == "raidisseur" and "§ 5.1(2)" in L["raison"] and L["sans"]["eta2"] < 1.0 for L in a["lieux"]) and len(a["lieux"]) == 2
      and any(f["gabarit"] == "r_I_min" for f in a["calcul"]) and a["statut"] == RE.OK, a["lieux"])
    a = ame(profil="IPE 750x147", nuance="S355", travees=[8.0], appuis={"mode": "poses", "ss": 200.0, "c": 200.0},
            cas=[{"nom": "G", "nature": "permanent", "charges": [q(20.0, 0.0, 8.0)]}])
    c("… avec s_s = c = 200 mm : la résistance de l'âme seule (type c, l_e non plafonné) accompagne le raidisseur dans sa fenêtre de calcul",
      any(f["gabarit"] == "a_le" for f in a["calcul"]) and round(a["lieux"][0]["sans"]["F_Rd"], 1) == round(frd_profil("IPE 750x147", 355.0, 200, "c", 200), 1)
      and a["lieux"][0]["type"] == "c", a["lieux"][0])
    a = ame(profil="SHS 200x200x8", cas=[{"nom": "G", "nature": "permanent", "charges": [q(5.0, 0.0, 6.0), P_(20.0, 3.0)]}])
    c("un tube posé et chargé : la force concentrée sur sa paroi n'est PAS vérifiée ici (EN 1993-1-8) — le bloc le signale, sans rien juger",
      a["verifications"] == [] and a["lieux"] == [] and [x["code"] for x in a["alertes"]] == ["tube_charge"] and a["statut"] == RE.OK, a["alertes"])
    garde = M["raidisseurs"].gabarit
    try:
        M["raidisseurs"].gabarit = lambda s: None
        a = ame(**lourd)
    finally:
        M["raidisseurs"].gabarit = garde
    L = lieu(a, "Charge à 2,00 m")
    vF = verif(a, "A-L01-F")
    c("un raidisseur nécessaire dans un profilé qui n'en reçoit pas : la vérification de l'âme seule est rendue NON CONFORME (675/611,2 = "
      "110 %), et dite",
      L is not None and L["decision"] == "aucun" and L["statut"] == RE.NOK and a["statut"] == RE.NOK and vF["statut"] == RE.NOK
      and round(vF["taux"], 2) == 1.10 and any("trop étroit" in x["message"] for x in a["alertes"]), L)

    # ---- 5. la barre
    r = mo.verifier(barre(mode="efforts", profil="IPE 750x147", nuance="S355", methodes=["ec3"],
                          efforts={"service": {}, "elu": {"Vz": 2200.0}}, flambement={"y": {"L": 6.0, "u": 1.0}, "z": {"L": 6.0, "u": 1.0}}))
    vus.append(r["ame"])
    c("efforts imposés, V_z,Ed = 2 200 kN sur l'IPE 750x147 S355 : l'Eurocode 3 passe (V_pl,z,Rd = 2 334 kN) mais l'âme voile "
      "(V_bw,Rd = 2 083,7 kN) — la barre est NON CONFORME",
      r["methodes"]["ec3"]["statut"] == RE.OK and r["ame"]["statut"] == RE.NOK and r["statut"] == RE.NOK
      and proche(verif(r["ame"], "A-Vb")["taux"], 2200.0 / Vbw) and verif(r["methodes"]["ec3"], "A-Vb") is None,
      (r["methodes"]["ec3"]["statut"], r["ame"]["statut"], r["statut"]))
    c("… efforts imposés : aucun lieu (les appuis et les charges ne sont pas décrits), et l'hypothèse le dit",
      r["ame"]["lieux"] == [] and any("Efforts imposés" in h for h in r["ame"]["hypotheses"]) and r["ame"]["combinaisons"] == [])
    r = mo.verifier(barre(methodes=["historique"]))
    vus.append(r["ame"])
    c("méthode historique SEULE : le bloc de l'âme est calculé quand même (ELU), et compte dans le statut de la barre",
      list(r["methodes"]) == ["historique"] and len(r["ame"]["lieux"]) == 2 and r["ame"]["norme"] == "NBN EN 1993-1-5"
      and r["ame"]["methode"] == "ame")
    r = mo.verifier(barre(profil="HD 400x818.0", nuance="S355"))
    c("une épaisseur hors du tableau 3.1 (HD 400x818) : le bloc de l'âme est « non couvert », avec son motif",
      r["ame"]["statut"] == RE.NON_COUVERT and "Tableau 3.1" in r["ame"]["alertes"][0]["message"] and r["ame"]["lieux"] == [], r["ame"]["statut"])
    vieux = barre()
    del vieux["appuis"]
    nb = en.normaliser(vieux)
    c("une note d'avant le lot 4 (sans « appuis », sans « ss ») se relit : barre posée, s_s = 0, c = 0 ; chaque charge ponctuelle reçoit s_s = 0",
      nb["appuis"] == {"mode": "poses", "ss": 0.0, "c": 0.0}
      and en.normaliser(barre(cas=[{"nom": "G", "nature": "permanent", "charges": [P_(5.0, 1.0), q(1.0, 0.0, 6.0)]}]))["cas"][0]["charges"]
      == [{"type": "P", "v": 5.0, "a": 1.0, "ss": 0.0}, {"type": "q", "v": 1.0, "a": 0.0, "b": 6.0}], nb["appuis"])
    c("données refusées : mode d'appui inconnu ; s_s négative, au-delà de 2 000 mm ou qui n'est pas un nombre ; débord négatif",
      all(_refuse(lambda k=k: en.normaliser(barre(**k))) for k in (
          {"appuis": {"mode": "soude"}}, {"appuis": {"ss": -1.0}}, {"appuis": {"ss": 2001.0}}, {"appuis": {"c": -5.0}}, {"appuis": {"ss": "100"}},
          {"cas": [{"nom": "G", "nature": "permanent", "charges": [P_(5.0, 1.0, -2.0)]}]},
          {"cas": [{"nom": "G", "nature": "permanent", "charges": [P_(5.0, 1.0, 1e9)]}]})))
    b0 = barre(**plus)
    avant = copy.deepcopy(b0)
    r1, r2 = mo.verifier(b0), mo.verifier(b0)
    c("la barre reçue n'est pas modifiée ; deux calculs rendent le même bloc, à l'octet",
      b0 == avant and json.dumps(r1["ame"], sort_keys=True) == json.dumps(r2["ame"], sort_keys=True))

    # ---- 6. la sortie : chaque formule de chaque bloc rencontré
    faux, illisibles, orphelines, n_f = [], [], [], 0
    for x in vus:
        cles = {f["cle"] for f in x["calcul"]}
        for f in x["calcul"]:
            n_f += 1
            if not proche(FO.evaluer(f), f["resultat"]):
                faux.append((f["cle"], FO.evaluer(f), f["resultat"]))
            if not W.relecture_juste(f):
                illisibles.append((f["cle"], W.relire(f), f["resultat"]))
        orphelines += [(v["id"], k) for v in x["verifications"] for k in v["formules"] if k not in cles]
    c(f"les {n_f} formules des blocs rencontrés : chaque gabarit, évalué sur les termes du moteur, redonne son résultat à 10⁻⁹ près",
      n_f > 300 and not faux, faux[:3])
    c("… et, refaite à la main avec les nombres IMPRIMÉS, chaque formule redonne le résultat imprimé", not illisibles, illisibles[:3])
    c("… chaque vérification ne cite que des formules rendues", not orphelines, orphelines[:3])
    gab = {f["gabarit"] for x in vus for f in x["calcul"]}
    manquent = sorted(set(FO.GABARITS_EC3_AME) - gab - {"a_eta3", "a_M_f", "a_eta1_bar", "a_71", "a_V_bw_n", "a_semelle", "a_eta1", "a_eta1_N"})
    c("… les gabarits de l'âme et des raidisseurs servent tous dans ces barres (hors l'interaction 7.1, les tubes, la semelle et η₁ sous effort "
      "normal, contrôlés à part)", not manquent, manquent)
    try:
        json.dumps(vus, allow_nan=False)
        c("JSON strict (ni NaN ni infini)", True)
    except ValueError as e_:
        c("JSON strict (ni NaN ni infini)", False, str(e_))
    return C


def charger(mutation=None):
    """Les cinq modules, chargés depuis leur TEXTE (une mutation appliquée à l'un d'eux) et liés entre eux."""
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
    mods["ame"].AM, mods["ame"].RA = mods["ec3_ame"], mods["raidisseurs"]
    mods["moteur"].AME, mods["moteur"].EN = mods["ame"], mods["entree"]
    return mods


MUTATIONS = [
    ("ec3_ame", "k_F du type (a) : 6 → 3,5", 'KF = {"a": 6.0, "b": 3.5}', 'KF = {"a": 3.5, "b": 3.5}'),
    ("ec3_ame", "k_F du type (b) : 3,5 → 6", 'KF = {"a": 6.0, "b": 3.5}', 'KF = {"a": 6.0, "b": 6.0}'),
    ("ec3_ame", "k_F du type (c) sans plafond", "    kF = KF[type_] if type_ != \"c\" else min(brut, KF_C_MAX)", "    kF = KF[type_] if type_ != \"c\" else brut"),
    ("ec3_ame", "k_F du type (c) : 2 + 6·s_s/h_w, le débord oublié", "    brut = 2.0 + 6.0 * (ss + c) / hw", "    brut = 2.0 + 6.0 * ss / hw"),
    ("ec3_ame", "F_cr : 0,9 oublié", "    Fcr = 0.9 * kF * E * tw ** 3 / hw", "    Fcr = kF * E * tw ** 3 / hw"),
    ("ec3_ame", "F_cr en t_w²", "    Fcr = 0.9 * kF * E * tw ** 3 / hw", "    Fcr = 0.9 * kF * E * tw ** 2 / hw"),
    ("ec3_ame", "m₁ = b_f/t_f", "    m1 = bf / tw ", "    m1 = bf / tf "),
    ("ec3_ame", "m₂ gardé même quand λ̄_F ≤ 0,5", "    if lF <= SEUIL_M2:\n        m2 = 0.0", "    if False:\n        m2 = 0.0"),
    ("ec3_ame", "m₂ toujours nul", "    m2 = 0.02 * (hw / tf) ** 2\n    L = longueurs(m2)", "    m2 = 0.0\n    L = longueurs(m2)"),
    ("ec3_ame", "m₂ : 0,02·(h_w/t_f)", "    m2 = 0.02 * (hw / tf) ** 2\n    L = longueurs(m2)", "    m2 = 0.02 * (hw / tf)\n    L = longueurs(m2)"),
    ("ec3_ame", "l_y : t_f au lieu de 2·t_f", "        ly1 = ss + 2.0 * tf * (1.0 + math.sqrt(m1 + m2))", "        ly1 = ss + tf * (1.0 + math.sqrt(m1 + m2))"),
    ("ec3_ame", "l_e sans le facteur 2", "        le_libre = kF * E * tw ** 2 / (2.0 * fy * hw)", "        le_libre = kF * E * tw ** 2 / (fy * hw)"),
    ("ec3_ame", "l_e non plafonné à s_s + c", "        le = min(le_libre, ss + c)", "        le = le_libre"),
    ("ec3_ame", "type (c) : la deuxième expression de l_y oubliée", '"ly": min(ly1, ly2, ly3)}', '"ly": min(ly1, ly3)}'),
    ("ec3_ame", "type (c) : la plus GRANDE des trois", '"ly": min(ly1, ly2, ly3)}', '"ly": max(ly1, ly2, ly3)}'),
    ("ec3_ame", "l_y,2 : m₁ au lieu de m₁/2", "        ly2 = le + tf * math.sqrt(m1 / 2.0 + (le / tf) ** 2 + m2)", "        ly2 = le + tf * math.sqrt(m1 + (le / tf) ** 2 + m2)"),
    ("ec3_ame", "χ_F non plafonné à 1", "    chi = min(CHI_F / lF, 1.0)", "    chi = CHI_F / lF"),
    ("ec3_ame", "χ_F = 1/λ̄_F", "CHI_F = 0.5  ", "CHI_F = 1.0  "),
    ("ec3_ame", "s_s non bornée à h_w", "    ss = min(max(float(ss), 0.0), hw) ", "    ss = max(float(ss), 0.0) "),
    ("ec3_ame", "interaction (7.2) : 0,8 → 1,0", "INTERACTION_72 = (0.8, 1.4)", "INTERACTION_72 = (1.0, 1.4)"),
    ("ec3_ame", "interaction (7.2) : limite 1,4 → 1,0", "INTERACTION_72 = (0.8, 1.4)", "INTERACTION_72 = (0.8, 1.0)"),
    ("ec3_ame", "η₁ : le moment oublié", "    return abs(N) / (A * fy / gM0) + abs(M) / (W * fy / gM0)", "    return abs(N) / (A * fy / gM0)"),
    ("ec3_ame", "limite d'élancement de l'âme : 72 → 124", "VOILEMENT_AME = 72.0 ", "VOILEMENT_AME = 124.0 "),
    ("ec3_ame", "limite d'élancement de l'âme : 72·ε sans η", '    lim = VOILEMENT_AME * NO.epsilon(fy) / P["eta"]', "    lim = VOILEMENT_AME * NO.epsilon(fy)"),
    ("ec3_ame", "λ̄_w : 86,4 → 37,4", "LAMBDA_W = 86.4 ", "LAMBDA_W = 37.4 "),
    ("ec3_ame", "χ_w : 0,83 → 1,0", "CHI_W = 0.83 ", "CHI_W = 1.0 "),
    ("ec3_ame", "V_bw,Rd sans √3", '"Vbw": e["n"] * chi * fy * e["hw"] * e["tw"] / (math.sqrt(3.0) * P["gamma_M1"])}', '"Vbw": e["n"] * chi * fy * e["hw"] * e["tw"] / P["gamma_M1"]}'),
    ("ec3_ame", "un tube : une seule âme", '"n": 1 if s["forme"] == "I" else 2,', '"n": 1,'),
    ("ec3_ame", "interaction (7.1) vérifiée dès 0 (seuil 0,5 perdu)", "SEUIL_ETA3 = 0.5 ", "SEUIL_ETA3 = 0.0 "),
    ("ec3_ame", "interaction (7.1) : (η̄₃ − 1)²", "    v = eta1b + (1.0 - Mf / Mpl) * (2.0 * eta3 - 1.0) ** 2", "    v = eta1b + (1.0 - Mf / Mpl) * (eta3 - 1.0) ** 2"),
    ("ec3_ame", "interaction (7.1) : M_f,Rd avec h", '    Mf = s["b"] * s["tf"] * (s["h"] - s["tf"]) * fy / gM0', '    Mf = s["b"] * s["tf"] * s["h"] * fy / gM0'),
    ("ec3_ame", "interaction (7.1) sous effort normal : rendue sans le dire", '    if s["forme"] != "I" or N != 0:', '    if s["forme"] != "I":'),
    ("ec3_ame", "voilement induit par la semelle : k = 4", "K_SEMELLE = 0.4 ", "K_SEMELLE = 0.004 "),
    ("raidisseurs", "épaisseurs : 6 mm proposée", "EPAISSEURS = (8.0, 10.0, 12.0, 15.0, 20.0)", "EPAISSEURS = (6.0, 8.0, 10.0, 12.0, 15.0, 20.0)"),
    ("raidisseurs", "chanfrein oublié dans l'aire de contact", '    Ac = 2.0 * (bs - g["chanfrein"]) * ts', "    Ac = 2.0 * bs * ts"),
    ("raidisseurs", "un seul plat au contact", '    Ac = 2.0 * (bs - g["chanfrein"]) * ts', '    Ac = (bs - g["chanfrein"]) * ts'),
    ("raidisseurs", "âme participante : 30·ε·t_w de chaque côté", "AME_PARTICIPANTE = 15.0 ", "AME_PARTICIPANTE = 30.0 "),
    ("raidisseurs", "âme participante non bornée par l'âme disponible", "    cotes = [plein if d is None else min(plein, max(float(d) - ts / 2.0, 0.0)) for d in dispo]", "    cotes = [plein for d in dispo]"),
    ("raidisseurs", "longueur de flambement : h_w", "LONGUEUR = 0.75 ", "LONGUEUR = 1.0 "),
    ("raidisseurs", "courbe a au lieu de c", 'COURBE = "c" ', 'COURBE = "a" '),
    ("raidisseurs", "palier de flambement ignoré (χ = 1 toujours)", "    if lam > PALIER:\n        Phi", "    if lam > 1e9:\n        Phi"),
    ("raidisseurs", "inertie du raidisseur : la largeur d'UN plat", "    Ist = (ts * (2.0 * bs + tw) ** 3 + (lw - ts) * tw ** 3) / 12.0", "    Ist = (ts * (bs + tw) ** 3 + (lw - ts) * tw ** 3) / 12.0"),
    ("raidisseurs", "aire du raidisseur : un seul plat", "    Ast = 2.0 * bs * ts + lw * tw", "    Ast = bs * ts + lw * tw"),
    ("raidisseurs", "plat : 14·ε → 20·ε", "ELANCEMENT_PLAT = 14.0 ", "ELANCEMENT_PLAT = 20.0 "),
    ("raidisseurs", "torsion du plat oubliée", "    return min(ELANCEMENT_PLAT * NO.epsilon(fy), math.sqrt(E / (TORSION * fy) - 0.25))", "    return ELANCEMENT_PLAT * NO.epsilon(fy)"),
    ("raidisseurs", "le plat dépasse la semelle (largeur b/2)", '    bs_max = PAS * math.floor((s["b"] - s["tw"]) / 2.0 / PAS + 1e-9)', '    bs_max = PAS * math.floor(s["b"] / 2.0 / PAS + 1e-9) + PAS'),
    ("raidisseurs", "chanfrein arrondi vers le bas", "    chanfrein = PAS * math.ceil(conge / PAS - 1e-9)", "    chanfrein = PAS * math.floor(conge / PAS)"),
    ("raidisseurs", "un IPN sans chanfrein (son congé n'est pas sous « r »)", '    conge = s.get("r") or s["tw"] ', '    conge = s.get("r") or 0.0 '),
    ("raidisseurs", "la première épaisseur rendue sans la vérifier", "        if p[\"ok\"]:\n            return p", "        if True:\n            return p"),
    ("raidisseurs", "le flambement oublié dans le choix de l'épaisseur", 'p.update({"F": F, "taux": max(F / p["NcRd"], F / p["NbRd"]), "rigide": rigide})', 'p.update({"F": F, "taux": F / p["NbRd"], "rigide": rigide})'),
    ("raidisseurs", "rigidité d'appui jamais demandée", '        rigide = (p["Ist"] >= p["I_min"]) if appui_rigide else None', "        rigide = None"),
    ("ame", "appuis par l'âme vérifiés comme posés", '    poses = ap["mode"] == "poses"', "    poses = True"),
    ("ame", "encastrement pris pour un appui posé", '            if a["type"] == "appui":\n                out.append({"x": a["x"], "appui": i,', '            if True:\n                out.append({"x": a["x"], "appui": i,'),
    ("ame", "charges ponctuelles vérifiées quel que soit leur niveau", '    if b["niveau"] in ("haut", "bas"):', "    if True:"),
    ("ame", "une charge vers le haut sur la semelle haute vérifiée", '    return {"haut": P if (niveau == "haut" and P > EPS) else 0.0,', '    return {"haut": abs(P) if niveau == "haut" else 0.0,'),
    ("ame", "une charge suspendue vérifiée comme si elle appuyait", '"bas": (R if R > EPS else 0.0) + (-P if (niveau == "bas" and P < -EPS) else 0.0),', '"bas": (R if R > EPS else 0.0) + (abs(P) if niveau == "bas" else 0.0),'),
    ("ame", "réaction non pondérée par les combinaisons", '        R = sum(f * sol.appuis[L["appui"]]["R"] for f, sol in zip(facteurs, sols) if f)', '        R = sum(sol.appuis[L["appui"]]["R"] for f, sol in zip(facteurs, sols) if f)'),
    ("ame", "charge ponctuelle non pondérée", '    P = sum(facteurs[ci] * v for ci, v in L["charges"])', '    P = sum(v for ci, v in L["charges"])'),
    ("ame", "longueur d'appui d'un groupe : la plus longue", '                hote["ss"][b["niveau"]] = min(g["ss"])', '                hote["ss"][b["niveau"]] = max(g["ss"])'),
    ("ame", "le débord de rive oublié", '            return ap["ss"] / 2.0 + ap["c"]', '            return ap["ss"] / 2.0'),
    ("ame", "type (a) partout (le bout de la barre ignoré)", '        type_ = "c" if 2.0 + 6.0 * (min(ss, hw) + c) / hw < AM.KF_C_MAX else "a"', '        type_ = "a"'),
    ("ame", "type (c) partout", '        type_ = "c" if 2.0 + 6.0 * (min(ss, hw) + c) / hw < AM.KF_C_MAX else "a"', '        type_ = "c"'),
    ("ame", "force transmise (type b) jamais vérifiée", "    if len(out) == 2:\n        out[\"b\"]", "    if len(out) == 99:\n        out[\"b\"]"),
    ("ame", "force transmise : la plus GRANDE des deux", '                essais.append(("b", min(o["haut"], o["bas"])))', '                essais.append(("b", max(o["haut"], o["bas"])))'),
    ("ame", "moment de l'interaction lu d'un seul côté", '"M": max(abs(e["M_g"]), abs(e["M_d"]))})', '"M": 0.0})'),
    ("ame", "raidisseur décidé sur η₂ seul (l'interaction oubliée)", '        elif pire["inter"] is not None and pire["inter"] > 1.0:', '        elif False:'),
    ("ame", "raidisseur jamais décidé sur η₂", '        if pire["eta2"] > 1.0:', '        if pire["eta2"] > 1e9:'),
    ("ame", "âme élancée : pas de raidisseur d'appui", '        elif el is not None and el["elancee"] and L["genre"] == "appui":\n            raison', '        elif False:\n            raison'),
    ("ame", "raidisseur dimensionné pour la force d'une combinaison quelconque", '            pire["F"] = max(pire["F"], o["Fmax"])', '            pire["F"] = o["Fmax"]'),
    ("ame", "deux forces rapprochées rendues « ok » sans le dire", "    if proches:\n        J = RE.Journal(METHODE, NORME)", "    if False:\n        J = RE.Journal(METHODE, NORME)"),
    ("ame", "forces rapprochées : cherchées sur l_y entier (deux fois trop loin)", "        if any(ecart < (r1[k][\"ly\"] + r2[k][\"ly\"]) / 2.0 for k", "        if any(ecart < (r1[k][\"ly\"] + r2[k][\"ly\"]) for k"),
    ("ame", "un encastrement passé sous silence", '    if genres - {"appui"}:', "    if False:"),
    ("ame", "voilement par cisaillement jamais cherché", '    if el is not None and el["elancee"]:\n        for C in combs:', '    if False:\n        for C in combs:'),
    ("ame", "un raidisseur impossible rendu conforme", '                impossible = bool(etude["raison"]) ', "                impossible = False "),
    ("ame", "un tube chargé rendu sans avertissement", '        if charges and s["forme"] in ("tube_carre", "tube_rect", "tube_rond"):', "        if False:"),
    ("ame", "réaction de soulèvement non signalée", '            if f["soulevement"] > 0:', "            if False:"),
    ("ame", "le bloc d'une barre aux efforts imposés : sans le cisaillement", '    AM.cisaillement(J, s, fy, P, abs(e["Vz"]) * 1e3, abs(e["My"]) * 1e6, e["N"] * 1e3)\n', ""),
    ("entree", "appuis par défaut « par l'âme » (rien n'est vérifié sans le dire)", '"appuis": {"mode": "poses", "ss": 0.0, "c": 0.0},', '"appuis": {"mode": "ame", "ss": 0.0, "c": 0.0},'),
    ("entree", "longueur d'appui d'une charge ponctuelle perdue à la relecture", '                ch["ss"] = _longueur_appui(saisie.get("ss", 0.0), f"cas « {nom} », longueur d\'appui s_s")', '                ch["ss"] = 0.0'),
    ("entree", "longueur d'appui sans borne", "    if not 0.0 <= x <= SS_MAX:", "    if False:"),
    ("entree", "mode d'appui non contrôlé", '    if ap is None or ap["mode"] not in MODES_APPUI:', "    if ap is None:"),
    ("moteur", "le statut de la barre ignore l'âme", '    statuts = [r["statut"] for r in out["methodes"].values()] + [out["ame"]["statut"]]', '    statuts = [r["statut"] for r in out["methodes"].values()]'),
]


def main():
    print("== 1 à 6. références")
    for nom, ok, info in controles():
        chk(nom, ok, info)

    print("== 6 bis. la section vérifiée seule (acier/barre/ec3_section.py)")
    s750 = SE.du_catalogue("IPE 750x147")
    r = ES.verifier({"section": s750, "nuance": "S355", "VzEd": 500e3})
    chk("ES.verifier, IPE 750x147 S355 sous V_z : l'âme élancée est vérifiée (A-Vb, 24 %) — plus « non couvert »",
        verif(r, "A-Vb") is not None and verif(r, "A-Vb")["statut"] == RE.OK and r["statut"] == RE.OK and verif(r, "E-Vb") is None, r["statut"])
    r = ES.verifier({"section": s750, "nuance": "S355", "VzEd": 500e3, "voilement": False})
    chk("… « voilement » faux (le moteur des barres, qui le range dans le bloc de l'âme) : la section ne le porte pas",
        verif(r, "A-Vb") is None and verif(r, "E-Vz") is not None)
    r = ES.verifier({"section": s750, "nuance": "S355", "VzEd": 2200e3})
    chk("… 2 200 kN : E-Vz passe (94 %), A-Vb non (106 %) — la section est non conforme",
        verif(r, "E-Vz")["statut"] == RE.OK and verif(r, "A-Vb")["statut"] == RE.NOK and r["statut"] == RE.NOK, r["statut"])

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
    sains = charger()
    chk("les modules rechargés SANS mutation passent tous les contrôles (le banc des mutations est sain)",
        sains is not None and all(ok for _, ok, _ in controles(sains)))

    print("== 8. pureté")
    code = ("import sys\n"
            "class Refus:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab', 'PIL', 'fitz'):\n"
            "            raise ImportError('refusé : ' + name)\n"
            "sys.meta_path.insert(0, Refus())\n"
            f"sys.path.insert(0, {RACINE!r})\n"
            "from acier.barre import moteur as M, entree as E\n"
            "r = M.verifier(E.nouvelle_barre())\n"
            "print(r['ame']['statut'], sorted(m for m in sys.modules if m.split('.')[0] in ('streamlit', 'pandas', 'matplotlib', 'scipy', 'reportlab')))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=RACINE)
    chk("le bloc de l'âme se calcule sans streamlit, pandas, matplotlib, scipy, reportlab (refusés)",
        p.returncode == 0 and p.stdout.strip().endswith("[]"), (p.stdout.strip(), p.stderr[-300:]))

    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())

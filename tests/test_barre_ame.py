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
def frd_temoin(h, b, tw, tf, fy, ss, type_, c=0.0, gM1=1.0, contact=False, fyw=None):
    """F_Rd (kN) d'une âme sous une force concentrée — EN 1993-1-5 § 6, sans une fonction du moteur, écrit depuis le texte
    et les règles déclarées du brief (§ 8.2, lot A du 16/11/2026) :
    - m₂ : les DEUX branches — m₂ = 0 retenue seulement si λ̄_F ≤ 0,5, m₂ = 0,02·(h_w/t_f)² seulement si λ̄_F > 0,5 ; la plus
      petite F_Rd des branches cohérentes ;
    - type (c), contact d'extrémité non établi (« inconnu », DÉCISION BUREAU n° 2) : s_s = 0 dans (6.10) SEULEMENT — k_F et le
      plafond s_s + c de l_e gardent s_s (commentaire JRC EUR 22898, p. 79) ;
    - λ̄_F avec la f_y de l'âme ``fyw`` (brief § 7.1), F_Rd et l_e avec la f_y de calcul ``fy``."""
    fyw = fy if fyw is None else fyw
    hw = h - 2.0 * tf
    ss = min(ss, hw)
    kF = {"a": 6.0, "b": 3.5}.get(type_) or min(2.0 + 6.0 * (ss + c) / hw, 6.0)
    Fcr = 0.9 * kF * E_ACIER * tw ** 3 / hw
    m1 = b / tw
    s610 = ss if (type_ != "c" or contact) else 0.0

    def ly(m2):
        l1 = s610 + 2.0 * tf * (1.0 + math.sqrt(m1 + m2))
        if type_ != "c":
            return l1
        le = min(kF * E_ACIER * tw ** 2 / (2.0 * fy * hw), ss + c)
        return min(l1, le + tf * math.sqrt(m1 / 2.0 + (le / tf) ** 2 + m2), le + tf * math.sqrt(m1 + m2))

    frd = []
    for m2 in (0.0, 0.02 * (hw / tf) ** 2):
        longueur = ly(m2)
        lam = math.sqrt(longueur * tw * fyw / Fcr)
        if (lam <= 0.5) == (m2 == 0.0):                       # la branche est cohérente
            frd.append(fy * min(0.5 / lam, 1.0) * longueur * tw / gM1 / 1e3)
    return min(frd)


def frd_profil(nom, fy, ss, type_, c=0.0, contact=False, fyw=None):
    p = brut(nom)
    return frd_temoin(p["h"], p["b"], p["tw"], p["tf"], fy, ss, type_, c, contact=contact, fyw=fyw)


def raidisseur_temoin(nom, fy, ts, cotes=None, fy_plat=None, fy_ame=None):
    """Une paire de plats d'épaisseur ts dans un I du catalogue — EN 1993-1-5 § 9, à la main : (bs, chanfrein, N_c,Rd kN,
    A_st, I_st, N_b,Rd kN). ``fy_plat`` : la f_y du plat selon son épaisseur, ``fy_ame`` : celle de l'âme (à défaut ``fy``) —
    ε_p (14·ε, torsion) et N_c,Rd avec le plat ; 15·ε·t_w avec l'âme ; λ̄ et N_b,Rd avec min(plat ; âme) (brief § 7.1, § 8.4)."""
    p = brut(nom)
    fyp, fyw = fy_plat or fy, fy_ame or fy
    fyd = min(fyp, fyw)
    hw = p["h"] - 2.0 * p["tf"]
    ep, ew = math.sqrt(235.0 / fyp), math.sqrt(235.0 / fyw)
    ch = 5.0 * math.ceil((p.get("r") or p["r1"]) / 5.0 - 1e-9)                # (un IPN porte son congé sous « r1 »)
    admis = min(14.0 * ep, math.sqrt(E_ACIER / (5.3 * fyp) - 0.25))
    bs = min(5.0 * math.floor((p["b"] - p["tw"]) / 2.0 / 5.0 + 1e-9), 5.0 * math.floor(ts * admis / 5.0 + 1e-9))
    Nc = 2.0 * (bs - ch) * ts * fyp / 1e3
    g, d = cotes if cotes is not None else (15.0 * ew * p["tw"], 15.0 * ew * p["tw"])
    lw = ts + g + d
    Ast = 2.0 * bs * ts + lw * p["tw"]
    Ist = (ts * (2.0 * bs + p["tw"]) ** 3 + (lw - ts) * p["tw"] ** 3) / 12.0
    lam = math.sqrt(Ast * fyd / (math.pi ** 2 * E_ACIER * Ist / (0.75 * hw) ** 2))
    chi = 1.0
    if lam > 0.2:
        phi = 0.5 * (1.0 + 0.49 * (lam - 0.2) + lam ** 2)
        chi = min(1.0 / (phi + math.sqrt(phi ** 2 - lam ** 2)), 1.0)
    return bs, ch, Nc, Ast, Ist, chi * Ast * fyd / 1e3


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
    def R(ss, t, cc=0.0, contact=False):
        return am.resistance(a300, 235.0, Pm, ss, t, cc, contact=contact)

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
    rk = R(100, "c", 0.0, contact=True)
    c("R4 (c) s_s = 100, c = 0, contact « inconnu » (DÉCISION BUREAU n° 2, défaut ; tests 4 et 13 du § 11.5) : k_F = 4,154 et F_cr = 1 008,5 kN "
      "avec s_s ; s_s = 0 dans (6.10) seulement : l_y = 119,76 mm, λ̄_F = 0,4451, χ_F = 1 → VERDICT F_Rd = 199,82 kN ; avec contact : "
      "248,91 kN (m₂ = 0, l_y = 149,2 mm) — l'autre branche (m₂ = 13,56 : 261,9 kN) est aussi cohérente : la plus petite est retenue, et dite",
      round(r["kF"], 3) == 4.154 and round(r["Fcr"] / 1e3, 1) == 1008.5 and r["le"] == 100.0 and r["le_plafonne"] and r["ss_610"] == 0.0
      and round(r["ly"], 2) == 119.76 and round(r["lambda_F"], 4) == 0.4451 and r["chi_F"] == 1.0 and round(r["FRd"] / 1e3, 2) == 199.82
      and round(r["FRd_contact"] / 1e3, 2) == 248.91 and rk["deux_branches"] and round(rk["FRd_autre"] / 1e3, 1) == 261.9 and rk["m2"] == 0.0
      and round(rk["ly"], 1) == 149.2 and round(rk["FRd"] / 1e3, 2) == 248.91 and not r["deux_branches"]
      and proche(r["FRd"] / 1e3, frd_profil("IPE 300", 235.0, 100, "c", 0.0)) and proche(rk["FRd"] / 1e3, frd_profil("IPE 300", 235.0, 100, "c", 0.0, contact=True))
      and any("deux branches" in m and "261,9" in m for m in am.mots_resistance(rk))
      and any("contact d'extrémité non établi" in m and "248,9" in m for m in am.mots_resistance(r)), (r, rk))
    r = R(100, "c", 50.0)
    rk = R(100, "c", 50.0, contact=True)
    c("R4 (c) s_s = 100, c = 50 : k_F = 5,230, F_cr = 1 270,0 kN, l_e = 150 ; sans contact, (6.10) avec s_s = 0 gouverne : 199,8 kN (comme c = 0) ; "
      "avec contact, l_y = 213,0 mm, λ̄_F = 0,529, χ_F = 0,945, F_Rd = 335,9 kN, pour information",
      round(r["kF"], 2) == 5.23 and round(r["Fcr"] / 1e3, 1) == 1270.0 and r["le"] == 150.0 and round(r["FRd"] / 1e3, 1) == 199.8
      and round(rk["ly"], 1) == 213.0 and round(rk["lambda_F"], 3) == 0.529 and round(rk["chi_F"], 3) == 0.945 and round(rk["FRd"] / 1e3, 1) == 335.9
      and proche(r["FRd_contact"], rk["FRd"]), (r, rk))
    c("type (c), contact établi : l_y est le plus petit de ses TROIS expressions (ici la troisième : 213,0 < 247,4 < 308,9)",
      rk["ly"] == min(rk["ly1"], rk["ly2"], rk["ly3"]) and rk["ly"] == rk["ly3"] and [round(rk[k], 1) for k in ("ly1", "ly2", "ly3")] == [247.4, 308.9, 213.0],
      (rk["ly1"], rk["ly2"], rk["ly3"]))
    # (lot A) les deux branches de m₂ : HEAA 1000 S235, appui de type (c), s_s = 0 — la plus forte discontinuité du catalogue
    sA = SE.du_catalogue("HEAA 1000")
    rA = am.resistance(am.ame_de(sA), 225.0, Pm, 0.0, "c", 0.0, fy_ame=235.0)
    c("deux branches de m₂ cohérentes (HEAA 1000 S235, f_y = 225, type (c), s_s = c = 0) : m₂ = 0 → λ̄_F = 0,381 ≤ 0,5 et F_Rd = 231,5 kN ; "
      "m₂ = 39,1 → λ̄_F > 0,5 et 458,4 kN (λ̄_F avec l'âme de 16 mm : 235) — la PLUS PETITE est retenue (l'ancien moteur gardait la seconde : "
      "478,7 kN en f_y = 235)",
      rA["deux_branches"] and rA["m2"] == 0.0 and round(rA["lambda_F"], 3) == 0.381 and round(rA["FRd"] / 1e3, 1) == 231.5
      and round(rA["FRd_autre"] / 1e3, 1) == 458.4 and proche(rA["FRd"] / 1e3, frd_profil("HEAA 1000", 225.0, 0.0, "c", 0.0, fyw=235.0)), rA)
    # (lot A) λ̄_F avec la f_y de l'ÂME (brief § 7.1) : HEB 300 S235 — f_y de calcul 225 (t_f = 19), âme de 11 mm à 235
    aH = am.ame_de(SE.du_catalogue("HEB 300"))
    rH = am.resistance(aH, 225.0, Pm, 0.0, "a", fy_ame=235.0)
    c("HEB 300 S235 sous une charge de type (a), s_s = 0 : λ̄_F = √(l_y·t_w·f_yw/F_cr) avec f_yw = 235 (l'âme), F_Rd = f_y·χ_F·l_y·t_w avec la "
      "f_y de calcul 225 — 585,2 kN",
      proche(rH["lambda_F"], math.sqrt(rH["ly"] * aH["tw"] * 235.0 / rH["Fcr"])) and proche(rH["FRd"], 225.0 * rH["chi_F"] * rH["ly"] * aH["tw"])
      and round(rH["FRd"] / 1e3, 1) == 585.2 and proche(rH["FRd"] / 1e3, frd_profil("HEB 300", 225.0, 0.0, "a", fyw=235.0)), rH)
    sK = SE.du_catalogue("IPE 750x147")
    r = am.resistance(am.ame_de(sK), 355.0, Pm, 200.0, "c", 200.0)
    kF = 2.0 + 6.0 * 400.0 / 719.0
    c("type (c) sur une âme haute (IPE 750x147 S355, s_s = c = 200 mm) : l_e = k_F·E·t_w²/(2·f_y·h_w) = 382,6 mm < s_s + c — il n'est PAS "
      "plafonné (k_F et l_e gardent s_s) ; ici la première expression de l_y gouverne (s_s = 0 dans (6.10) : contact non établi)",
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
    for nom in I_DU_CATALOGUE[::7]:                                    # le contact établi et la f_y de l'âme, sur un I sur sept
        a = am.ame_de(SE.du_catalogue(nom))
        for ss, t, cc in ((100, "c", 0), (60, "c", 500), (0, "a", 0)):
            if not proche(am.resistance(a, 225.0, Pm, ss, t, cc, contact=True, fy_ame=235.0)["FRd"] / 1e3,
                          frd_profil(nom, 225.0, ss, t, cc, contact=True, fyw=235.0), 1e-12):
                ecarts.append((nom, "contact", ss, t, cc))
    c(f"le témoin écrit ici depuis la norme = le moteur, sur les {len(I_DU_CATALOGUE)} I et H du catalogue, deux nuances, douze charges "
      "(types a, b, c ; s_s de 0 à plus que h_w ; c de 0 à 500) ; et, sur un I sur sept, le contact établi et une âme à f_y propre",
      len(I_DU_CATALOGUE) > 150 and not ecarts, ecarts[:4])
    c("s_s est bornée à h_w (§ 6.3(1)) : 5 000 mm saisis ne valent que 278,6",
      R(5000, "a")["ss"] == a300["hw"] and R(5000, "a")["FRd"] == R(278.6, "a")["FRd"] and R(-5, "a")["ss"] == 0.0)
    c("type (c) loin du bout, contact établi : k_F plafonné à 6 et la première expression de l_y — la résistance du type (a) ; sans contact, "
      "s_s ne compte pas dans (6.10) : moins",
      R(100, "c", 1e4, True)["kF"] == 6.0 and R(100, "c", 1e4, True)["kF_plafonne"] and proche(R(100, "c", 1e4, True)["FRd"], R(100, "a")["FRd"])
      and R(100, "c", 1e4)["FRd"] < R(100, "a")["FRd"])
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

    # ---- 2. le voilement par cisaillement (lot A : η = 1,00 pour un laminé — l'IPE 750x147 S355, élancé avec η = 1,20, ne l'est
    #      plus ; l'IPE 750x134 S355 l'est encore)
    s750 = SE.du_catalogue("IPE 750x134")
    p750 = brut("IPE 750x134")
    hw, tw = p750["h"] - 2 * p750["tf"], p750["tw"]
    eps355 = math.sqrt(235.0 / 355.0)
    e = am.elancement(s750, 355.0, Pm)
    e147 = am.elancement(SE.du_catalogue("IPE 750x147"), 345.0, Pm, fy_ame=355.0)
    c("IPE 750x134 S355 : h_w/t_w = 59,9 > 72·ε/η = 72·0,814/1,00 = 58,6 — âme élancée ; en S235 (limite 72) elle ne l'est pas ; l'IPE 300 "
      "jamais ; l'IPE 750x147 S355 (54,5, ε de son âme de 13,2 mm) non plus — avec η = 1,20 il l'était (48,8)",
      e["elancee"] and round(e["rapport"], 1) == 59.9 and round(e["limite"], 1) == 58.6 and e["eta"] == 1.0 and not am.elancement(s750, 235.0, Pm)["elancee"]
      and not am.elancement(s300, 355.0, Pm)["elancee"] and not e147["elancee"] and round(e147["limite"], 1) == 58.6, (e, e147))
    lw = hw / (86.4 * tw * eps355)
    Vbw = 0.83 / lw * 355.0 * hw * tw / R3 / 1e3
    J = RE.Journal("ame", "x")
    am.cisaillement(J, s750, 355.0, Pm, 500e3)
    v = verif(J.rendre(), "A-Vb")
    c("… λ̄_w = h_w/(86,4·t·ε) = 0,852, χ_w = 0,83/λ̄_w = 0,974 (λ̄_w ≥ 0,83/η), V_bw,Rd = 0,974·355·719·12/√3 = 1 722,0 kN ; 500 kN : 29,0 %",
      v is not None and round(lw, 3) == 0.852 and round(Vbw, 1) == 1722.0 and round(0.83 / lw, 3) == 0.974 and proche(v["limite"], Vbw) and proche(v["valeur"], 500.0)
      and v["statut"] == RE.OK and proche(v["taux"], 500.0 / Vbw) and proche(J.G["voilement_cisaillement"]["chi_w"], 0.83 / lw), v)
    c("V_bw,Rd < V_pl,Rd (1 965,4 kN : l'aire du catalogue, au-dessus du plancher h_w·t_w) : le voilement réduit bien la résistance au cisaillement",
      Vbw < p750["Avz"] * 100 * 355.0 / R3 / 1e3 and p750["Avz"] * 100 > hw * tw)
    soude = SE.saisie("I", "soude_oxycoupe", h=1000.0, b=300.0, tf=20.0, tw=6.0, r=0.0, A=17760.0, Iy=3.3e9, Iz=9.0e7, iy=431.0, iz=71.2)
    es = am.elancement(soude, 235.0, Pm)
    c("une section SOUDÉE garde η = 1,20 (NBN EN 1993-1-5 ANB:2011, 5.1(2)) : seuil 72·ε/η = 60 — test 22 du § 11.5",
      es["eta"] == 1.2 and proche(es["limite"], 60.0) and am.eta_de(soude, Pm) == 1.2 and am.eta_de(s750, Pm) == 1.0
      and am.eta_de(SE.du_catalogue("RHS 350x150x5"), Pm) == 1.0, es)
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
      v is not None and proche(v["valeur"], attendu) and proche(v["taux"], attendu) and v["limite"] == 1.0 and 0.9 < attendu < 1.0
      and round(Mf, 0) == round(264.0 * 15.5 * 734.5 * 355.0 / 1e6, 0), (v, attendu, Mf, Mpl))
    sans = []
    for V, Mx, Nx in ((800e3, 1500e6, 0.0), (1200e3, 1000e6, 0.0), (1200e3, 0.0, 0.0)):
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
    am.cisaillement(J, tube, 355.0, Pm, 200e3)
    am.cisaillement(J, tube, 355.0, Pm, 200e3, direction="y")
    r = J.rendre()
    hwt = 350.0 - 3.0 * 5.0
    Vt = 2.0 * 0.83 / (hwt / (86.4 * 5.0 * eps355)) * 355.0 * hwt * 5.0 / R3 / 1e3
    c("un tube RHS 350x150x5 S355 : (h − 3t)/t = 67,0 > 72·ε/η = 58,6 (un tube : η = 1,00 ; en S235, 67 ≤ 72 : pas élancé) — ses DEUX âmes "
      "portent V_z (n = 2) ; sous V_y, les parois de 150 ne sont pas élancées",
      [x["id"] for x in r["verifications"]] == ["A-Vb"] and proche(verif(r, "A-Vb")["limite"], Vt) and not am.elancement(tube, 235.0, Pm)["elancee"]
      and any(f["gabarit"] == "a_V_bw_n" for f in r["calcul"]) and am.elancement(tube, 355.0, Pm, "y")["elancee"] is False, r["verifications"])
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
    c("aucun plat ne porte 600 kN (20 mm : 517 kN avec une f_y donnée de 235 pour tous les plats) : le plus épais est rendu, « ok » faux, taux 116 %",
      p["ts"] == 20.0 and not p["ok"] and proche(p["taux"], 600.0 / 517.0), p)
    # ---- 3 bis. (lot A) la f_y de CHAQUE plat selon son épaisseur — test 18 du § 11.5 du brief (§ 7.1, § 8.4)
    p18 = ra.plat(s300, 235.0, Pm, 20.0, nuance="S235", fy_ame=235.0)
    bs, ch, Nc, Ast, Ist, Nb = raidisseur_temoin("IPE 300", 235.0, 20.0, fy_plat=225.0, fy_ame=235.0)
    c("test 18 du brief : IPE 300 S235, F = 450 kN, 2 plats 70 × 20 — f_y = 225 pour les plats de 20 mm (tableau 3.3 ANB), f_y de calcul de "
      "la section du raidisseur min(225 ; 235) = 225 ; A_st = 2·70·20 + (30·ε·t_w + t_s)·t_w = 4 454,3 mm² (ε de l'âme = 1) ; λ̄ = 0,063, χ = 1, "
      "N_b,Rd = 1 002,2 kN, taux 0,449 ; contact 2 200 mm² : 204,5 N/mm² ≤ 225, N_c,Rd = 495,0 kN, taux 0,909",
      p18["fy_plat"] == 225.0 and p18["fy_calcul"] == 225.0 and p18["fy_ame"] == 235.0 and (p18["bs"], p18["chanfrein"]) == (70.0, 15.0)
      and proche(p18["Ast"], 4454.3) and proche(p18["Ast"], Ast) and round(p18["lambda"], 3) == 0.063 and p18["chi"] == 1.0
      and round(p18["NbRd"] / 1e3, 1) == 1002.2 and proche(p18["NbRd"] / 1e3, Nb) and round(450.0 / (p18["NbRd"] / 1e3), 3) == 0.449
      and p18["Ac"] == 2200.0 and round(450e3 / p18["Ac"], 1) == 204.5 and proche(p18["NcRd"] / 1e3, 495.0) and proche(p18["NcRd"] / 1e3, Nc)
      and round(450.0 / 495.0, 3) == 0.909 and proche(p18["elancement_lim"], 14.0 * math.sqrt(235.0 / 225.0))
      and proche(p18["torsion_lim"], 5.3 * 225.0 / E_ACIER), p18)
    p6 = ra.dimensionner(s300, 235.0, Pm, 600e3, nuance="S235", fy_ame=235.0)
    p8 = ra.plat(s300, 235.0, Pm, 12.0, nuance="S235", fy_ame=235.0)
    c("… avec la nuance : 600 kN, aucun plat ne porte (20 mm : 495,0 kN, taux 121 %) ; un plat de 12 mm garde 235 (t ≤ 16) — N_c,Rd 310,2 kN",
      p6["ts"] == 20.0 and not p6["ok"] and proche(p6["taux"], 600.0 / 495.0) and p8["fy_plat"] == 235.0 and round(p8["NcRd"] / 1e3, 1) == 310.2, (p6["taux"], p8["fy_plat"]))
    sM = SE.du_catalogue("HEM 300")
    pM = ra.plat(sM, 225.0, Pm, 15.0, nuance="S235", fy_ame=225.0)
    c("HEM 300 S235 (âme de 21 mm : 225) : un plat de 15 mm (235) — la f_y de calcul du raidisseur est celle de l'âme, min(235 ; 225) = 225, "
      "dans λ̄ et N_b,Rd ; l'écrasement du plat, et 14·ε_p, gardent celle du plat",
      pM["fy_plat"] == 235.0 and pM["fy_calcul"] == 225.0 and proche(pM["NbRd"], pM["chi"] * pM["Ast"] * 225.0)
      and proche(pM["NcRd"], pM["Ac"] * 235.0) and proche(pM["elancement_lim"], 14.0) and proche(pM["lw"], 15.0 + 30.0 * math.sqrt(235.0 / 225.0) * 21.0), pM)
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
    c("appui rigide d'une âme élancée : I_st ≥ 0,75·h_w·t_w³ (IPE 750x134 : 93 cm⁴ demandés) ; un tube ne reçoit pas de raidisseur",
      ra.dimensionner(s750, 355.0, Pm, 100e3, appui_rigide=True)["rigide"] is True and round(ra.plat(s750, 355.0, Pm, 8.0)["I_min"] / 1e4) == 93
      and ra.dimensionner(s750, 355.0, Pm, 100e3)["rigide"] is None and ra.gabarit(tube) is None and ra.dimensionner(tube, 235.0, Pm, 1e3) is None)

    # ---- 4. la décision, lieu par lieu
    g300 = 78.6 * brut("IPE 300")["A"] * 1e-4               # le poids propre : 78,6 kN/m³ × A (brief H13 ; lot A)
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
    c("s_s = 100 mm aux appuis (contact d'extrémité inconnu) : F_Rd = 199,8 kN (R4, type c, verdict sans contact) — avec contact 248,9 kN, dit "
      "pour information ; avec 50 mm de débord : 199,8 kN (avec contact 335,9)",
      round(a["lieux"][0]["sans"]["F_Rd"], 1) == 199.8 and round(b_["lieux"][0]["sans"]["F_Rd"], 1) == 199.8 and a["lieux"][0]["ss"] == 100.0
      and any("avec contact, F_Rd = 248,9 kN" in m for m in verif(a, "A-L00-F")["messages"])
      and any("avec contact, F_Rd = 335,9 kN" in m for m in verif(b_, "A-L00-F")["messages"])
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
    bs, ch, Nc, Ast, Ist, Nb = raidisseur_temoin("HEB 300", 235.0, 15.0, fy_plat=235.0, fy_ame=235.0)
    c("HEB 300 S235, 675 kN à mi-portée (s_s non saisie) : F_Rd = 585,2 kN seulement (f_y de calcul 225 : t_f = 19 mm ; λ̄_F avec l'âme, 235) — "
      "raidisseur 2 plats 140 × 15 mm (plats de 15 mm : 235 ; âme : 235), écrasement 675/775,5 = 87,0 %",
      L is not None and L["decision"] == "raidisseur" and round(L["sans"]["F_Rd"], 1) == 585.2 and "F_Ed = 675,0 kN > F_Rd = 585,2 kN" in L["raison"]
      and proche(L["sans"]["F_Rd"], frd_profil("HEB 300", 225.0, 0.0, "a", fyw=235.0))
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
    a = ame(profil="IPE 750x134", nuance="S355", travees=[8.0], maintiens=[{"x": 2.0}, {"x": 4.0}, {"x": 6.0}],
            cas=[{"nom": "G", "nature": "permanent", "charges": [q(20.0, 0.0, 8.0)]}])
    g750 = 78.6 * p750["A"] * 1e-4
    Ved = 1.35 * (20.0 + g750) * 4.0
    v = verif(a, "A-Vb")
    c("IPE 750x134 S355, 8 m, 20 kN/m : l'âme élancée est VÉRIFIÉE au cisaillement (V_Ed = 115,2 kN ≤ V_bw,Rd = 1 722,0 kN), à l'appui",
      v is not None and proche(v["valeur"], Ved) and proche(v["limite"], Vbw) and v["statut"] == RE.OK and v["x"] in (0.0, 8.0), v)
    c("… et des raidisseurs sont posés à ses deux appuis (§ 5.1(2)), bien que l'âme seule y porte la réaction ; ils ont la rigidité d'un appui",
      all(L["decision"] == "raidisseur" and "§ 5.1(2)" in L["raison"] and L["sans"]["eta2"] < 1.0 for L in a["lieux"]) and len(a["lieux"]) == 2
      and any(f["gabarit"] == "r_I_min" for f in a["calcul"]) and a["statut"] == RE.OK, a["lieux"])
    a = ame(profil="IPE 750x134", nuance="S355", travees=[8.0], appuis={"mode": "poses", "ss": 200.0, "c": 200.0},
            cas=[{"nom": "G", "nature": "permanent", "charges": [q(20.0, 0.0, 8.0)]}])
    c("… avec s_s = c = 200 mm : la résistance de l'âme seule (type c, l_e non plafonné) accompagne le raidisseur dans sa fenêtre de calcul",
      any(f["gabarit"] == "a_le" for f in a["calcul"]) and round(a["lieux"][0]["sans"]["F_Rd"], 1) == round(frd_profil("IPE 750x134", 355.0, 200, "c", 200), 1)
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
    c("un raidisseur nécessaire dans un profilé qui n'en reçoit pas : la vérification de l'âme seule est rendue NON CONFORME (675/585,2 = "
      "115 %), et dite",
      L is not None and L["decision"] == "aucun" and L["statut"] == RE.NOK and a["statut"] == RE.NOK and vF["statut"] == RE.NOK
      and round(vF["taux"], 2) == 1.15 and any("trop étroit" in x["message"] for x in a["alertes"]), L)

    # ---- 5. la barre
    r = mo.verifier(barre(mode="efforts", profil="IPE 750x134", nuance="S355", methodes=["ec3"],
                          efforts={"service": {}, "elu": {"Vz": 1800.0}}, flambement={"y": {"L": 6.0, "u": 1.0}, "z": {"L": 6.0, "u": 1.0}}))
    vus.append(r["ame"])
    c("efforts imposés, V_z,Ed = 1 800 kN sur l'IPE 750x134 S355 : l'Eurocode 3 passe (V_pl,z,Rd = 1 965,4 kN) mais l'âme voile "
      "(V_bw,Rd = 1 722,0 kN) — la barre est NON CONFORME",
      r["methodes"]["ec3"]["statut"] == RE.OK and r["ame"]["statut"] == RE.NOK and r["statut"] == RE.NOK
      and proche(verif(r["ame"], "A-Vb")["taux"], 1800.0 / Vbw) and verif(r["methodes"]["ec3"], "A-Vb") is None,
      (r["methodes"]["ec3"]["statut"], r["ame"]["statut"], r["statut"]))
    c("… efforts imposés : aucun lieu (les appuis et les charges ne sont pas décrits), et l'hypothèse le dit",
      r["ame"]["lieux"] == [] and any("Efforts imposés" in h for h in r["ame"]["hypotheses"]) and r["ame"]["combinaisons"] == [])
    r = mo.verifier(barre(methodes=["historique"]))
    vus.append(r["ame"])
    c("méthode historique SEULE : le bloc de l'âme est calculé quand même (ELU), et compte dans le statut de la barre",
      list(r["methodes"]) == ["historique"] and len(r["ame"]["lieux"]) == 2 and r["ame"]["norme"] == "NBN EN 1993-1-5"
      and r["ame"]["methode"] == "ame")
    r = mo.verifier(barre(profil="HD 400x818.0", nuance="S355"))
    c("une épaisseur hors du domaine du module (HD 400x818, t_f = 97 mm > 80 mm) : le bloc de l'âme est « non couvert », avec son motif",
      r["ame"]["statut"] == RE.NON_COUVERT and "80 mm" in r["ame"]["alertes"][0]["message"] and r["ame"]["lieux"] == [], r["ame"]["statut"])
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
    ("ec3_ame", "m₂ gardé même quand λ̄_F ≤ 0,5 (la branche m₂ = 0 jamais cohérente)", '(sans, sans["lambda_F"] <= SEUIL_M2)', "(sans, False)"),
    ("ec3_ame", "lot A — deux branches cohérentes : la PLUS GRANDE F_Rd retenue", 'B = min(coherentes, key=lambda x: x["FRd"])',
     'B = max(coherentes, key=lambda x: x["FRd"])'),
    ("ec3_ame", "m₂ toujours nul", "avec, sans = branche(0.02 * (hw / tf) ** 2), branche(0.0)", "avec, sans = branche(0.0), branche(0.0)"),
    ("ec3_ame", "m₂ : 0,02·(h_w/t_f)", "avec, sans = branche(0.02 * (hw / tf) ** 2), branche(0.0)", "avec, sans = branche(0.02 * (hw / tf)), branche(0.0)"),
    ("ec3_ame", "l_y : t_f au lieu de 2·t_f", "        ly1 = ss610 + 2.0 * tf * (1.0 + math.sqrt(m1 + m2))", "        ly1 = ss610 + tf * (1.0 + math.sqrt(m1 + m2))"),
    ("ec3_ame", "lot A — type (c) : le contact supposé établi (s_s dans (6.10))", '    ss610 = ss if (type_ != "c" or contact) else 0.0', "    ss610 = ss"),
    ("ec3_ame", "lot A — type (c) : s_s ôté aussi de k_F", "    brut = 2.0 + 6.0 * (ss + c) / hw", "    brut = 2.0 + 6.0 * (ss610 + c) / hw"),
    ("ec3_ame", "lot A — λ̄_F avec la f_y de calcul (et non celle de l'âme)", '        lF = math.sqrt(L["ly"] * tw * fyw / Fcr)', '        lF = math.sqrt(L["ly"] * tw * fy / Fcr)'),
    ("ec3_ame", "lot A — η des sections soudées pour un laminé", '    return P["eta_soude"] if s.get("fabrication") in SOUDES_ETA else P["eta"]', '    return P["eta_soude"]'),
    ("ec3_ame", "lot A — η = 1 pour une section soudée", '    return P["eta_soude"] if s.get("fabrication") in SOUDES_ETA else P["eta"]', '    return P["eta"]'),
    ("ec3_ame", "lot A — 72·ε/η avec l'ε de la f_y de calcul", "    fyw = fy if fy_ame is None else float(fy_ame)\n    eta, eps = eta_de(s, P), NO.epsilon(fyw)",
     "    fyw = fy\n    eta, eps = eta_de(s, P), NO.epsilon(fyw)"),
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
    ("ec3_ame", "limite d'élancement de l'âme : 72·ε sans η", "    lim = VOILEMENT_AME * eps / eta", "    lim = VOILEMENT_AME * eps"),
    ("ec3_ame", "λ̄_w : 86,4 → 37,4", "LAMBDA_W = 86.4 ", "LAMBDA_W = 37.4 "),
    ("ec3_ame", "χ_w : 0,83 → 1,0", "CHI_W = 0.83 ", "CHI_W = 1.0 "),
    ("ec3_ame", "V_bw,Rd sans √3", '"Vbw": e["n"] * chi * fy * e["hw"] * e["tw"] / (math.sqrt(3.0) * P["gamma_M1"])}', '"Vbw": e["n"] * chi * fy * e["hw"] * e["tw"] / P["gamma_M1"]}'),
    ("ec3_ame", "un tube : une seule âme", '"n": 1 if s["forme"] == "I" else 2,', '"n": 1,'),
    ("ec3_ame", "interaction (7.1) vérifiée dès 0 (seuil 0,5 perdu)", "SEUIL_ETA3 = 0.5 ", "SEUIL_ETA3 = 0.0 "),
    ("ec3_ame", "interaction (7.1) : (η̄₃ − 1)²", "    v = eta1b + (1.0 - Mf / Mpl) * (2.0 * eta3 - 1.0) ** 2", "    v = eta1b + (1.0 - Mf / Mpl) * (eta3 - 1.0) ** 2"),
    ("ec3_ame", "interaction (7.1) : M_f,Rd avec h", '    Mf = s["b"] * s["tf"] * (s["h"] - s["tf"]) * fy / gM0', '    Mf = s["b"] * s["tf"] * s["h"] * fy / gM0'),
    ("ec3_ame", "interaction (7.1) sous effort normal : rendue sans le dire", '    if s["forme"] != "I" or N != 0:', '    if s["forme"] != "I":'),
    ("ec3_ame", "voilement induit par la semelle : k = 4", "K_SEMELLE = 0.4 ", "K_SEMELLE = 0.004 "),
    ("raidisseurs", "lot A — le plat à la f_y de la barre (épaisseur oubliée)", '    fyp = fy if nuance is None else NO.limite_elastique(nuance, ts, produit="lamine")', "    fyp = fy"),
    ("raidisseurs", "lot A — f_y de calcul du raidisseur : celle du plat seul", '"fy_calcul": min(fyp, fyw)', '"fy_calcul": fyp'),
    ("raidisseurs", "lot A — 15·ε·t_w avec l'ε du plat", '    plein = AME_PARTICIPANTE * m["eps_ame"] * tw', '    plein = AME_PARTICIPANTE * m["eps_plat"] * tw'),
    ("raidisseurs", "lot A — écrasement à la f_y de calcul (et non à celle du plat)", "NcRd=Ac * fyp / P[\"gamma_M0\"]", "NcRd=Ac * fyd / P[\"gamma_M0\"]"),
    ("ame", "lot A — le raidisseur dimensionné sans la nuance (une f_y pour tout)", 'appui_rigide=rigide, nuance=b["nuance"], fy_ame=fyw)', "appui_rigide=rigide)"),
    ("ame", "lot A — l'âme prise à la f_y de calcul", '    return d["fy"], d["fy_parois"]["ame"], d["fy_source"]', '    return d["fy"], d["fy"], d["fy_source"]'),
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
    ("ame", "le bloc d'une barre aux efforts imposés : sans le cisaillement", '    AM.cisaillement(J, s, fy, P, abs(e["Vz"]) * 1e3, abs(e["My"]) * 1e6, e["N"] * 1e3, fy_ame=fyw)\n', ""),
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
    s750 = SE.du_catalogue("IPE 750x134")
    r = ES.verifier({"section": s750, "nuance": "S355", "VzEd": 500e3})
    chk("ES.verifier, IPE 750x134 S355 sous V_z : l'âme élancée est vérifiée (A-Vb, 29 %) — plus « non couvert »",
        verif(r, "A-Vb") is not None and verif(r, "A-Vb")["statut"] == RE.OK and r["statut"] == RE.OK and verif(r, "E-Vb") is None, r["statut"])
    r = ES.verifier({"section": s750, "nuance": "S355", "VzEd": 500e3, "voilement": False})
    chk("… « voilement » faux (le moteur des barres, qui le range dans le bloc de l'âme) : la section ne le porte pas",
        verif(r, "A-Vb") is None and verif(r, "E-Vz") is not None)
    r = ES.verifier({"section": s750, "nuance": "S355", "VzEd": 1800e3})
    chk("… 1 800 kN : E-Vz passe (92 %), A-Vb non (105 %) — la section est non conforme",
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

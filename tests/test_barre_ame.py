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
from acier.barre import prs as PRS  # noqa: E402
from acier.barre import raidisseurs as RA  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402
from acier.barre import section as SE  # noqa: E402
from acier.barre import statique as ST  # noqa: E402
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
def frd_temoin(h, b, tw, tf, fy, ss, type_, c=0.0, gM1=1.0, contact=False, fyw=None, hw=None, borne=False):
    """F_Rd (kN) d'une âme sous une force concentrée — EN 1993-1-5 § 6, sans une fonction du moteur, écrit depuis le texte
    et les règles déclarées du brief (§ 8.2, lot A du 16/11/2026) :
    - m₂ : les DEUX branches — m₂ = 0 retenue seulement si λ̄_F ≤ 0,5, m₂ = 0,02·(h_w/t_f)² seulement si λ̄_F > 0,5 ; la plus
      petite F_Rd des branches cohérentes ;
    - type (c), contact d'extrémité non établi (« inconnu », DÉCISION BUREAU n° 2) : s_s = 0 dans (6.10) SEULEMENT — k_F et le
      plafond s_s + c de l_e gardent s_s (commentaire JRC EUR 22898, p. 79) ;
    - λ̄_F avec la f_y de l'âme ``fyw`` (brief § 7.1), F_Rd et l_e avec la f_y de calcul ``fy`` ;
    - ``borne`` (lot B : s_s ou c non saisi, pris à 0) : au type (c), la branche m₂ = 0 même incohérente, dès qu'elle le
      deviendrait avec k_F = 6 et sa l_y — sinon la règle des branches."""
    fyw = fy if fyw is None else fyw
    hw = h - 2.0 * tf if hw is None else hw                # un PRS : l'âme saisie entre ses semelles
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
    l0 = ly(0.0)
    if borne and type_ == "c" and math.sqrt(l0 * tw * fyw / (0.9 * 6.0 * E_ACIER * tw ** 3 / hw)) <= 0.5:
        return fy * min(0.5 / math.sqrt(l0 * tw * fyw / Fcr), 1.0) * l0 * tw / gM1 / 1e3
    return min(frd)


def frd_profil(nom, fy, ss, type_, c=0.0, contact=False, fyw=None, borne=False):
    p = brut(nom)
    return frd_temoin(p["h"], p["b"], p["tw"], p["tf"], fy, ss, type_, c, contact=contact, fyw=fyw, borne=borne)


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


def prs_section(b1, t1, hw, tw, b2, t2, a=None, fabrication=None):
    """Une section PRS en I : celle de acier/barre/prs.py (bloc B2), le contrat des règles du lot B, § 4. Avant
    l'intégration, ce test écrivait ici une copie d'après le contrat ; elle manquait des clés que prs.py lit (la gorge
    « s » : ec3_classe → prs.parois), et c'était une seconde source des propriétés (R2). ``fabrication`` None : les
    bords des semelles ne sont pas saisis."""
    return PRS.section_prs(b1, t1, hw, tw, b2, t2, a=a, nuance="S355", bords=fabrication)


R2 = dict(poids_propre=False, appuis={"mode": "poses", "ss": None, "c": None},                 # (le brief ne donne pas s_s aux appuis)
          cas=[{"nom": "G", "nature": "permanent", "charges": [q(2.42, 0.0, 6.0), P_(10.0, 3.0, 100.0)]},
               {"nom": "Q", "nature": "exploitation_A", "charges": [P_(15.0, 3.0, 100.0)]}])
R2M = dict(R2, maintiens=[{"x": 3.0, "genre": "haut"}])          # R2, la semelle haute tenue au droit de la charge (§ 6.1(1))
CONTINUE = dict(profil="HEA 200", modele="continue", travees=[5.0, 5.0], poids_propre=False, appuis={"mode": "poses", "ss": 0.0, "c": 0.0},
                cas=[{"nom": "G", "nature": "permanent", "charges": [q(6.0, 0.0, 10.0)]},
                     {"nom": "Q1", "nature": "exploitation_B", "charges": [q(8.0, 0.0, 5.0)]},
                     {"nom": "Q2", "nature": "exploitation_B", "charges": [q(8.0, 5.0, 10.0)]}])
TOUT = "1,35·G + 1,5·Q1 + 1,5·Q2"
I_DU_CATALOGUE = [p["nom"] for p in CAT.par_nom().values() if SE.FAMILLES.get(p["famille"], ("",))[0] == "I"]


def balayage(ame_mod, Pm, sections, fys, ss_l, c_l):
    """La BORNE contre le réel (2e reprise du lot B ; brief § 8.3, point 5) : pour chaque section, f_y, semelle, contact et chaque
    (s_s, c) RÉEL, F_Rd à la borne d'une donnée non saisie — s_s et c (0, 0), s_s seule (0, c), c seul (s_s, 0) — ≤ F_Rd réel ;
    par le chemin du moteur (``ame.resistances`` : le type se déduit de la géométrie) ; et la force transmise (type b), s_s non
    saisie. Rend (cas, violations, cas où la règle des branches SEULE, à la borne, dépasse le réel)."""
    G = 1e7

    def F(s_, cote, ss, c, ct, m, fy, fyw):
        L_ = {"dispo": (ss / 2.0 + c, G), "ss": {"haut": None, "bas": None, cote: ss}, "contact": {cote: ct}, "manque": {cote: m}, "manque_bout": []}
        return ame_mod.resistances(L_, s_, fy, Pm, fyw)[cote]["FRd"]

    n, viol, naif = 0, [], 0
    for nom_, s_ in sections:
        for fy, fyw in fys:
            for cote in (("haut", "bas") if s_.get("prs") else ("bas",)):
                for ct in ("oui", "non", "inconnu"):
                    for ss in ss_l:
                        for cc in c_l:
                            reel = F(s_, cote, ss, cc, ct, [], fy, fyw)
                            for b_ss, b_c, m in ((0.0, 0.0, ["s_s", "c"]), (0.0, cc, ["s_s"]), (ss, 0.0, ["c"])):
                                n += 1
                                if F(s_, cote, b_ss, b_c, ct, m, fy, fyw) > reel * (1.0 + 1e-12):
                                    viol.append((nom_, fy, cote, ct, ss, cc, m))
                                if F(s_, cote, b_ss, b_c, ct, [], fy, fyw) > reel * (1.0 + 1e-12):
                                    naif += 1
            # une charge près d'un bout (son c vient de l'appui de rive : e = s_s,ap/2 + c,ap) — s_s seule, l'appui seul, les deux
            for cote in (("haut", "bas") if s_.get("prs") else ("bas",)):
                for x in (0.0, 50.0, 200.0):
                    for ss in ss_l[:-1]:
                        for e in (0.0, 25.0, 100.0):
                            def Lc(ss_, e_, m, mb):
                                return {"dispo": (x + e_, G), "ss": {"haut": None, "bas": None, cote: ss_}, "contact": {cote: "inconnu"},
                                        "manque": {cote: m}, "manque_bout": mb}
                            reel = ame_mod.resistances(Lc(ss, e, [], []), s_, fy, Pm, fyw)[cote]["FRd"]
                            for b_ss, b_e, m, mb in ((0.0, e, ["s_s"], []), (ss, 0.0, [], ["s_s", "c"]), (0.0, 0.0, ["s_s"], ["c"])):
                                n += 1
                                if ame_mod.resistances(Lc(b_ss, b_e, m, mb), s_, fy, Pm, fyw)[cote]["FRd"] > reel * (1.0 + 1e-12):
                                    viol.append((nom_, fy, "charge", x, ss, e, m, mb))
            for ss in ss_l:
                n += 1
                reel = ame_mod.resistances({"dispo": (G, G), "ss": {"haut": ss, "bas": ss}, "manque": {"haut": [], "bas": []}}, s_, fy, Pm, fyw)["b"]["FRd"]
                b0 = ame_mod.resistances({"dispo": (G, G), "ss": {"haut": 0.0, "bas": 0.0}, "manque": {"haut": ["s_s"], "bas": ["s_s"]}}, s_, fy, Pm, fyw)
                if b0["b"]["FRd"] > reel * (1.0 + 1e-12):
                    viol.append((nom_, fy, "b", ss))
    return n, viol, naif


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
    sN0 = prs_section(300.0, 20.0, 1000.0, 6.0, 250.0, 20.0)

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
      and any("hypothèse de contact non établie" in m and "248,9" in m for m in am.mots_resistance(r)), (r, rk))
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

    # ---- 1 bis. (lot B) les PRS en I — brief § 18.5 dernier point, § 8.2, test 22 du § 11.5 (partie PRS), PRS-A du § 11.8
    sA = prs_section(300.0, 30.0, 540.0, 15.0, 300.0, 30.0, a=6.0)
    eA = am.elancement(sA, 345.0, Pm, fy_ame=355.0)
    c("PRS-A (300 × 30, 540 × 15, S355 ; bords des semelles NON saisis : « fabrication » None) : section soudée → η = 1,20 (NBN EN 1993-1-5 "
      "ANB:2011, 5.1(2) et sa note) ; h_w/t_w = 36,0 ≤ 72·ε_w/η = 72·0,8136/1,2 = 48,82 (ε de l'âme de 15 mm, 355) : pas de voilement par "
      "cisaillement à vérifier (A_v,z = 9 720 mm² : le contrôle de acier/barre/prs.py, bloc B2)",
      am.eta_de(sA, Pm) == 1.2 and am.est_soude(sA) and eA["eta"] == 1.2 and eA["rapport"] == 36.0
      and round(eA["limite"], 2) == 48.82 and not eA["elancee"] and proche(sA["A"], 26100.0) and round(sA["Iy"] / 1e6, 2) == 1660.23, eA)
    sC = prs_section(300.0, 30.0, 1420.0, 25.0, 500.0, 50.0)
    aH, aB = am.ame_de(sC, "haut"), am.ame_de(sC, "bas")
    rH = am.resistance(aH, 215.0, Pm, 100.0, "a", fy_ame=225.0)
    rB = am.resistance(aB, 215.0, Pm, 100.0, "a", fy_ame=225.0)
    c("PRS monosymétrique (PRS-C : 300 × 30, 1 420 × 25, 500 × 50) : h_w = 1 420 mm, l'âme ENTRE semelles (et non h − 2·t_f = 1 400) ; sous la "
      "semelle haute, b_f = 300 et t_f = 30 (m₁ = 12) ; sous la basse, 500 et 50 (m₁ = 20) — F_Rd = le témoin, semelle par semelle",
      am.hw_de(sC) == 1420.0 and (aH["bf"], aH["tf"], aB["bf"], aB["tf"]) == (300.0, 30.0, 500.0, 50.0) and rH["m1"] == 12.0 and rB["m1"] == 20.0
      and proche(rH["FRd"] / 1e3, frd_temoin(sC["h"], 300.0, 25.0, 30.0, 215.0, 100.0, "a", fyw=225.0, hw=1420.0))
      and proche(rB["FRd"] / 1e3, frd_temoin(sC["h"], 500.0, 25.0, 50.0, 215.0, 100.0, "a", fyw=225.0, hw=1420.0)) and rB["FRd"] > rH["FRd"], (rH, rB))
    Lb = {"dispo": (5000.0, 5000.0), "ss": {"haut": 100.0, "bas": 100.0}}
    rb = M["ame"].resistances(Lb, sC, 215.0, Pm, 225.0)
    c("… la force transmise d'une semelle à l'autre (type b) prend la semelle qui donne la PLUS PETITE F_Rd (choix prudent déclaré, brief § 8.2) : "
      "la semelle haute (300 × 30)",
      rb["b"]["type"] == "b" and rb["b"]["ame"]["bf"] == 300.0 and proche(rb["b"]["FRd"], am.resistance(aH, 215.0, Pm, 100.0, "b", fy_ame=225.0)["FRd"])
      and rb["b"]["FRd"] < am.resistance(aB, 215.0, Pm, 100.0, "b", fy_ame=225.0)["FRd"] and rb["haut"]["ame"]["tf"] == 30.0 and rb["bas"]["ame"]["tf"] == 50.0, rb["b"])
    mf = am.moment_semelles(sC, 215.0, 1.0)
    smC = am.semelle(sC, 215.0, Pm, fy_semelle=235.0)
    c("… M_f,Rd des seules semelles (§ 7.1) : la plus petite (300 × 30 = 9 000 mm²) sur h_f = 1 500 − 15 − 25 = 1 460 mm ; un laminé garde "
      "b·t_f·(h − t_f) ; voilement induit par la semelle (§ 8) avec la PLUS GRANDE semelle (500 × 50 : NBN EN 1993-1-5 ANB:2011, 8(2)) et f_yf",
      mf["Af"] == 9000.0 and mf["hf"] == 1460.0 and proche(mf["Mf"], 9000.0 * 1460.0 * 215.0)
      and proche(am.moment_semelles(s300, 235.0, 1.0)["Mf"], 150.0 * 10.7 * (300.0 - 10.7) * 235.0)
      and (smC["bf"], smC["tf"]) == (500.0, 50.0) and proche(smC["limite"], 0.4 * E_ACIER / 235.0 * math.sqrt(1420.0 * 25.0 / (500.0 * 50.0))), (mf, smC))
    gA, gC = ra.gabarit(sA), ra.gabarit(sC)
    c("… le raidisseur d'un PRS : chanfrein qui dégage le cordon (a·√2 = 8,5 → 10 mm ; gorge non saisie : t_w = 25) ; largeur prise sur la plus "
      "ÉTROITE semelle ((300 − 25)/2 → 135 mm) ; hauteur h_w = 1 420",
      (gA["chanfrein"], gA["bs_max"], gA["hw"]) == (10.0, 140.0, 540.0) and (gC["chanfrein"], gC["bs_max"], gC["hw"]) == (25.0, 135.0, 1420.0), (gA, gC))

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
    # (lot A, deuxième passe) le CAISSON soudé, saisi (le catalogue n'en a pas) : η = 1,20 comme un I soudé (NBN EN 1993-1-5
    # ANB:2011, 5.1(2) : sections soudées) — parois de 8 mm, h_w = h − 3·t (EN 1993-1-5 § 4.4(2))
    def caisson(h, t, fab="caisson"):
        A, Ii = 4.0 * (h - t) * t, (h ** 4 - (h - 2.0 * t) ** 4) / 12.0
        return SE.saisie("tube_carre", fab, h=h, b=h, t=t, A=A, Iy=Ii, Iz=Ii, iy=math.sqrt(Ii / A), iz=math.sqrt(Ii / A))
    k600, k424 = caisson(600.0, 8.0), caisson(424.0, 8.0)
    ek = [am.elancement(caisson(600.0, 8.0, f), fy, Pm) for f in ("caisson", "caisson_recuit") for fy in (235.0, 355.0)]
    rk = am.resistance_cisaillement(am.elancement(k424, 235.0, Pm), 235.0, Pm)
    J = RE.Journal("ame", "x")
    am.cisaillement(J, k600, 235.0, Pm, 500e3)
    c("un CAISSON soudé (saisi, recuit ou non) garde η = 1,20 : 72·ε/η = 60·ε — 60 en S235, 60 × 0,814 = 48,8 en S355 (paroi de 8 mm : 355) ; "
      "600 × 8 : h_w/t = 576/8 = 72 > 60, âme élancée et « A-Vb » écrit (avec η = 1,00 : 72 ≤ 72, rien) ; 424 × 8 (h_w/t = 50) : λ̄_w = "
      "50/86,4 = 0,579 < 0,83/η = 0,692 → plateau χ_w = η = 1,20 (1,00 avec η = 1,00)",
      all(e_["eta"] == 1.2 and e_["elancee"] and e_["rapport"] == 72.0 for e_ in ek) and proche(ek[0]["limite"], 60.0)
      and proche(ek[1]["limite"], 60.0 * math.sqrt(235.0 / 355.0)) and round(ek[1]["limite"], 1) == 48.8 and proche(ek[2]["limite"], 60.0)
      and am.eta_de(k600, Pm) == 1.2 and rk["plafonne"] and rk["chi_w"] == 1.2 and round(rk["lambda_w"], 3) == 0.579
      and verif(J.rendre(), "A-Vb") is not None, (ek, rk))
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
    p3 = brut("IPE 300")
    S300, I300 = 150.0 * 10.7 * (300.0 - 10.7) / 2.0, p3["Iy"] * 1e4
    crit0 = (Fed * 1e3 / (2.0 * 10.7 * 7.1) / 235.0) ** 2 + 3.0 * (Fed * 1e3 * S300 / (I300 * 7.1) / 235.0) ** 2
    c("la barre neuve (IPE 300, 6 m, G + Q, posée, s_s et c NON SAISIS) : deux lieux — Appui A, Appui B — vérifiés à la BORNE s_s = c = 0 "
      "(la borne est sous F_Rd réel, prouvé par le balayage ; brief § 8.3, point 5) : type (c), F_Ed = 44,4 kN ≤ F_Rd = 58,0 kN (77 %), σ_x = 0 "
      "→ les deux contrôles, σ_z,Ed = F_Ed/(2·t_f·t_w) = 292,6 N/mm², critère 6.2.1(5) 1,57 > 1 : la borne ne passe pas — « DÉCISION "
      "IMPOSSIBLE : s_s à saisir » (seul le critère, qui ne lit pas c, ne passe pas : c n'est pas demandé — le harnais suit : « s_s et c » à la "
      "2e reprise), « bloque » (lot C, C0c : une donnée à saisir — le harnais suit, « non couvert » avant), le taux à la borne (157 %) sur la "
      "ligne ; la borne dite UNE fois dans les hypothèses, marquée À VÉRIFIER",
      [L["nom"] for L in a["lieux"]] == ["Appui A", "Appui B"] and all(L["type"] == "c" and L["decision"] == "impossible" and L["ss"] == 0.0
                                                                      and L["statut"] == RE.BLOQUE and L["raidisseur"] is None
                                                                      and round(L["taux"], 3) == 1.567 and L["borne"] is None
                                                                      and L["raison"] == "Décision impossible : s_s à saisir."
                                                                      and L["sans"]["semelle"] == ["les_deux"] for L in a["lieux"])
      and proche(a["lieux"][0]["F_Ed"], Fed) and proche(a["lieux"][0]["sans"]["F_Rd"], frd_profil("IPE 300", 235.0, 0, "c", 0))
      and round(a["lieux"][0]["sans"]["eta2"], 2) == 0.77 and proche(a["lieux"][0]["sans"]["critere"], crit0, 1e-6) and round(crit0, 2) == 1.57
      and round(a["lieux"][0]["sans"]["sigma_z"], 1) == 292.6 and a["statut"] == RE.BLOQUE and a["taux_max"] is None
      and [v["id"] for v in a["verifications"]] == ["A-L00-EL", "A-L00-F", "A-L01-EL", "A-L01-F"]
      and all(v["statut"] == RE.BLOQUE and v["taux"] is not None for v in a["verifications"])
      and [x["message"] for x in a["alertes"]] == ["Appui A, appui B : décision impossible : s_s à saisir."]
      and a["hypotheses"].count(AME.BORNE_HYPOTHESE) == 1 and "À VÉRIFIER" in AME.BORNE_HYPOTHESE, (a["lieux"][0], crit0))
    a00 = ame(appuis={"mode": "poses", "ss": 0.0, "c": 0.0})
    c("… les données non saisies sont DITES (hypothèses, message de la vérification) ; une note ANCIENNE (s_s = c = 0 SAISIS) rend les MÊMES "
      "chiffres — F_Rd, taux, critère — et décide (raidisseur, critère 1,57) : le libellé seul dit la différence",
      any("s_s non saisie, débord en rive c non saisi" in h for h in a["hypotheses"])
      and any("Non saisis : s_s = 0, c = 0 (borne prudente)." in m for m in verif(a, "A-L00-F")["messages"])
      and all(L["decision"] == "raidisseur" and "critère 6.2.1(5)" in L["raison"] for L in a00["lieux"])
      and all(proche(x["sans"]["F_Rd"], y["sans"]["F_Rd"]) and proche(x["sans"]["critere"], y["sans"]["critere"]) and proche(x["sans"]["eta2"], y["sans"]["eta2"])
              for x, y in zip(a["lieux"], a00["lieux"])), a["hypotheses"])
    a = ame(appuis={"mode": "poses", "ss": 100.0, "c": None})
    c("… s_s saisie (100 mm), c non : la borne c = 0 passe (22 %) — « sans raidisseur », la borne dite (« c = 0 : borne prudente ») ; une "
      "poutre continue : l'appui du milieu n'a pas de débord — pas de borne",
      all(L["decision"] == "aucun" and L["borne"] == "c = 0 : borne prudente" and L["statut"] == RE.OK and round(L["taux"], 3) == 0.223 for L in a["lieux"])
      and any("Non saisi : c = 0 (borne prudente)." in m for m in verif(a, "A-L00-F")["messages"])
      and [L["borne"] for L in ame(**dict(CONTINUE, appuis={"mode": "poses", "ss": 100.0, "c": None}))["lieux"]]
      == ["c = 0 : borne prudente", None, "c = 0 : borne prudente"], [L["borne"] for L in a["lieux"]])
    lourd_c = dict(poids_propre=False, appuis={"mode": "poses", "ss": 180.0, "c": None}, cas=[{"nom": "G", "nature": "permanent", "charges": [q(52.0, 0.0, 6.0)]}])
    a, a4 = ame(**lourd_c), ame(**dict(lourd_c, appuis={"mode": "poses", "ss": 180.0, "c": 400.0}))
    L, L4 = a["lieux"][0], a4["lieux"][0]
    c("… une réaction de 210,6 kN, s_s = 180 mm, c non saisi : à la borne c = 0, type (c), F_Rd = 199,8 kN < 210,6 kN (le critère 6.2.1(5), lui, "
      "passe) — « décision impossible : c à saisir », le taux à la borne (105 %) sur la ligne, « bloque » (lot C) ; c = 400 mm saisi : type (a) — sans "
      "raidisseur",
      L["decision"] == "impossible" and L["raison"] == "Décision impossible : c à saisir." and L["type"] == "c" and round(L["sans"]["F_Rd"], 1) == 199.8
      and L["sans"]["critere"] < 1.0 and proche(L["taux"], 1.35 * 52.0 * 3.0 / 199.8240, 1e-4) and L["statut"] == RE.BLOQUE and L4["type"] == "a"
      and L4["decision"] == "aucun", (L["raison"], L["taux"], L["sans"]["critere"], L4["decision"]))
    L0 = ame(**dict(lourd_c, appuis={"mode": "poses", "ss": None, "c": None}))["lieux"][0]
    c("… s_s et c non saisis, F_Rd à la borne qui ne passe pas (η₂ lit s_s et c) : « décision impossible : s_s et c à saisir » — les deux sont "
      "demandés (seul le critère 6.2.1(5) ne lit pas c)",
      L0["decision"] == "impossible" and L0["raison"] == "Décision impossible : s_s et c à saisir." and L0["sans"]["eta2"] > 1.0, (L0["raison"], L0["sans"]))
    a = ame(appuis={"mode": "poses", "ss": 100.0, "c": 0.0})
    b_ = ame(appuis={"mode": "poses", "ss": 100.0, "c": 50.0})
    c("s_s = 100 mm aux appuis (contact d'extrémité inconnu) : F_Rd = 199,8 kN (R4, type c, verdict sans contact) — avec contact 248,9 kN, dit "
      "pour information ; avec 50 mm de débord : 199,8 kN (avec contact 335,9)",
      round(a["lieux"][0]["sans"]["F_Rd"], 1) == 199.8 and round(b_["lieux"][0]["sans"]["F_Rd"], 1) == 199.8 and a["lieux"][0]["ss"] == 100.0
      and any("avec contact, F_Rd = 248,9 kN" in m for m in verif(a, "A-L00-F")["messages"])
      and any("avec contact, F_Rd = 335,9 kN" in m for m in verif(b_, "A-L00-F")["messages"])
      and not any("non saisie" in h for h in a["hypotheses"]) and AME.BORNE_HYPOTHESE not in a["hypotheses"], (a["lieux"][0], b_["lieux"][0]))
    a = ame(appuis={"mode": "poses", "ss": 100.0, "c": 400.0})
    c("un débord assez long (s_s + c ≥ 2·h_w/3) : la réaction de rive n'est plus « près du bout » — type (a)",
      a["lieux"][0]["type"] == "a" and proche(a["lieux"][0]["sans"]["F_Rd"], frd_profil("IPE 300", 235.0, 100, "a")), a["lieux"][0])
    a0 = ame(**R2)
    L0 = lieu(a0, "Charge à 3,00 m")
    a = ame(**R2M)
    L = lieu(a, "Charge à 3,00 m")
    eta1 = 68.70e6 / (brut("IPE 300")["Wel_y"] * 1e3 * 235.0)
    v = verif(a, L["base"] + "-FM") if L else None
    c("R2/R4 : la charge de 36 kN à mi-portée (s_s = 100 mm, G et Q à la même abscisse : UNE force), type (a) — η₂ = 36/387,8 = 0,093 ; la "
      "semelle haute tenue au droit de la charge (maintien « haut » à 3 m) : pas de raidisseur",
      L is not None and L["genre"] == "charge" and L["type"] == "a" and proche(L["F_Ed"], 36.0) and round(L["sans"]["F_Rd"], 1) == 387.8
      and round(L["sans"]["eta2"], 3) == 0.093 and L["decision"] == "aucun" and len([x for x in a["lieux"] if x["genre"] == "charge"]) == 1, L)
    c("… R2 tel quel (sans maintien intermédiaire) : la semelle chargée n'est pas tenue au droit de la charge (§ 6.1(1)) ; un maintien n'a "
      "pas de borne — non couvert (règle prudente gardée, À TRANCHER), et la raison dit en une ligne le fait, le taux obtenu comme si elle "
      "était tenue et le remède : « Semelle haute sans maintien (EN 1993-1-5, 6.1(1)) — 37 % si elle est tenue ; ajouter un maintien "
      "(À VÉRIFIER). » (le harnais suit : « maintien … à préciser ») ; les MÊMES taux (36,6 %, sur la ligne) ; l'avertissement le redit ; un "
      "maintien de la semelle BASSE ne suffit pas ; un « complet », oui ; aux appuis (s_s, c non saisis), la borne passe : sans raidisseur (61 %)",
      L0["decision"] == "impossible" and L0["statut"] == RE.NON_COUVERT
      and L0["raison"] == "Semelle haute sans maintien (EN 1993-1-5, 6.1(1)) — 37 % si elle est tenue ; ajouter un maintien (À VÉRIFIER)."
      and [x["message"] for x in a0["alertes"] if x["code"].startswith("decision_impossible")]
      == ["Charge à 3,00 m : semelle haute sans maintien (EN 1993-1-5, 6.1(1)) — 37 % si elle est tenue ; ajouter un maintien (À VÉRIFIER)."]
      and proche(L0["taux"], L["taux"]) and round(L0["taux"], 3) == 0.366 and proche(L0["sans"]["interaction"], L["sans"]["interaction"])
      and proche(L0["sans"]["eta2"], L["sans"]["eta2"])
      and all(x["decision"] == "aucun" and x["borne"] == "s_s = 0, c = 0 : borne prudente" and round(x["taux"], 3) == 0.613
              for x in a0["lieux"] if x["genre"] == "appui")
      and a0["statut"] == RE.NON_COUVERT
      and lieu(ame(**dict(R2, maintiens=[{"x": 3.0, "genre": "bas"}])), "Charge à 3,00 m")["decision"] == "impossible"
      and lieu(ame(**dict(R2, maintiens=[{"x": 3.0, "genre": "complet"}])), "Charge à 3,00 m")["decision"] == "aucun", L0)
    # (lot C, C4 — le harnais suit : le critère est écrit comme la norme, « η₂ + 0,8·η₁ ≤ 1,4 » ; la valeur est la somme
    #  0,513, la limite 1,4, le taux 0,366 ne change pas ; avant : valeur 0,366, limite 1, « (η₂ + 0,8·η₁)/1,4 ≤ 1 »)
    fm = [f for f in a["calcul"] if v is not None and f["cle"] in v["formules"]]
    c("… avec la flexion (7.2) : η₁ = M_Ed/(W_el,y·f_y) = 68,70/130,9 = 0,525 (élastique) ; η₂ + 0,8·η₁ = 0,513 ≤ 1,4, taux 36,6 % — c'est elle "
      "qui gouverne le lieu ; le taux est la formule (η₂ + 0,8·η₁)/1,4, écrite UNE fois",
      v is not None and proche(v["valeur"], 36.0 / L["sans"]["F_Rd"] + 0.8 * eta1, 1e-4) and round(v["valeur"], 3) == 0.513 and v["limite"] == 1.4
      and round(v["taux"], 3) == 0.366 and v["critere"] == "η₂ + 0,8·η₁ ≤ 1,4"
      and [f["gabarit"] for f in fm].count("a_72") == 1 and "taux" not in [f["gabarit"] for f in fm]
      and L["gouvernant"] == v["id"] and proche(L["taux"], v["taux"]) and proche(L["sans"]["interaction"], v["taux"]) and round(eta1, 3) == 0.525, v)
    c("… aux appuis, sans moment : pas d'interaction (7.2) ; σ_x = 0 : le critère 6.2.1(5) à la jonction (« A-Lnn-EL », lot B) — il passe",
      [x["id"] for x in a["verifications"]] == ["A-L00-EL", "A-L00-F", "A-L01-F", "A-L01-FM", "A-L02-EL", "A-L02-F"]
      and all(verif(a, i)["statut"] == RE.OK for i in ("A-L00-EL", "A-L02-EL")), [x["id"] for x in a["verifications"]])
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
    plus["cas"][0]["charges"].append(P_(50.0, 5.0, 0.0))
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
    moins["cas"][0]["charges"].append(P_(20.0, 5.0, 200.0))
    moins["appuis"] = {"mode": "poses", "ss": 200.0, "c": 0.0}
    a = ame(**moins)
    vB = verif(a, "A-L01-B")
    FRb = frd_profil("HEA 200", 235.0, 200.0, "b")
    c("… une charge plus petite (27 kN de calcul ; s_s = 200 mm, bornée à h_w) : pas de raidisseur — la force qui passe d'une semelle à l'autre "
      "est vérifiée en type (b), k_F = 3,5 (le harnais suit : s_s = 0 avant le lot B ; avec s_s = 0, la semelle haute TENDUE au droit de "
      "l'appui demande un raidisseur — test 31, plus bas)",
      vB is not None and proche(vB["valeur"], 27.0) and proche(vB["limite"], FRb) and lieu(a, "Appui B")["decision"] == "aucun"
      and any("Type (b)" in m for m in vB["messages"]) and verif(a, "A-L01-F") is not None, vB)
    moins["cas"][0]["charges"][-1] = P_(20.0, 5.0)
    vB0 = verif(ame(**moins), "A-L01-B")
    c("… la même charge SANS s_s : la force transmise (type b) est calculée à la borne s_s = 0 — et le dit (« Non saisi : s_s = 0 »)",
      vB0 is not None and vB0["messages"][:1] == ["Non saisi : s_s = 0 (borne prudente)."] and proche(vB0["limite"], frd_profil("HEA 200", 235.0, 0.0, "b")),
      vB0 and vB0["messages"])
    lourd = dict(profil="HEB 300", travees=[4.0], appuis={"mode": "poses", "ss": 200.0, "c": 0.0}, maintiens=[{"x": 2.0, "genre": "complet"}],
                 cas=[{"nom": "G", "nature": "permanent", "charges": [P_(500.0, 2.0, 0.0)]}])
    a = ame(**lourd)
    L = lieu(a, "Charge à 2,00 m")
    bs, ch, Nc, Ast, Ist, Nb = raidisseur_temoin("HEB 300", 235.0, 15.0, fy_plat=235.0, fy_ame=235.0)
    c("HEB 300 S235, 675 kN à mi-portée (s_s = 0 saisie ; les deux semelles tenues) : F_Rd = 585,2 kN seulement (f_y de calcul 225 : t_f = 19 mm ; λ̄_F avec l'âme, 235) — "
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
    pres = {d_: ame(appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 3.0}, {"x": 3.0 + d_}],
                    cas=[{"nom": "G", "nature": "permanent", "charges": [P_(20.0, 3.0, 50.0), P_(20.0, 3.0 + d_, 50.0)]}])
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
    # (lot C, C1 point 15 — défaut NON CONSERVATIF : un appui « par l'âme » d'une âme élancée ne demandait aucun raidisseur d'appui,
    #  alors que le voilement par cisaillement A-Vb est calculé avec eux ; mesuré sur la base : lieux [], âme « ok »)
    a = ame(profil="IPE 750x134", nuance="S355", travees=[8.0], maintiens=[{"x": 4.0}], appuis={"mode": "ame", "ss": None, "c": None},
            cas=[{"nom": "G", "nature": "permanent", "charges": [q(30.0, 0.0, 8.0)]}, {"nom": "Q", "nature": "exploitation_A", "charges": [q(40.0, 0.0, 8.0)]}])
    Rk = (1.35 * (30.0 + g750) + 1.5 * 40.0) * 4.0
    c("(lot C, C1 point 15) appuis PAR L'ÂME, âme élancée (IPE 750x134 S355) : le raidisseur d'appui du § 5.1(2) est dû à chaque appui — une "
      "paire de plats dimensionnée sous la réaction entière (À VÉRIFIER), ses vérifications R et Rb, jamais « sans raidisseur » ; l'hypothèse dite "
      "une fois",
      [L["nom"] for L in a["lieux"]] == ["Appui A", "Appui B"] and all(L["decision"] == "raidisseur" and L["raison"] == AME.RAISON_APPUI_ELANCEE
                                                                        and L["par_ame"] and proche(L["F_Ed"], Rk) for L in a["lieux"])
      and {"A-L00-R", "A-L00-Rb", "A-L01-R", "A-L01-Rb"} <= {v["id"] for v in a["verifications"]}
      and a["hypotheses"].count(AME.HYP_APPUI_AME) == 1 and any(f["gabarit"] == "r_I_min" for f in a["calcul"]), a["lieux"])
    a = ame(profil="IPE 300", appuis={"mode": "ame", "ss": None, "c": None}, maintiens=[{"x": 3.0}])
    c("… une âme qui n'est pas élancée (IPE 300) tenue par l'âme : aucun lieu, aucun raidisseur (rien ne change)",
      a["lieux"] == [] and AME.HYP_APPUI_AME not in a["hypotheses"], a["lieux"])
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

    # ---- 4 bis. (lot B) brief § 8.2 à § 8.4 et § 15.4 : seuil local, η₁ plastique, contact saisi, semelle tendue, σ_z, vue avant/après
    a = ame(**R2M)
    L = lieu(a, "Charge à 3,00 m")
    vFM = verif(a, L["base"] + "-FM")
    c("R4 / test 17 : SEUIL de résistance locale à M_Ed = 68,70 kN·m imposé — 380,08 kN avec η₁ élastique (l'interaction gouverne), 387,76 kN "
      "avec η₁ plastique (η₂ ≤ 1 gouverne) ; le message le dit, avec « ce n'est pas la capacité de la barre »",
      abs(L["sans"]["seuil"]["elastique"] - 380.08) <= 0.01 and abs(L["sans"]["seuil"]["plastique"] - 387.76) <= 0.01
      and round(L["sans"]["seuil"]["M_Ed"], 2) == 68.70 and any("380,1 kN" in m and "ce n'est pas la capacité de la barre" in m for m in vFM["messages"]),
      L["sans"]["seuil"])
    r6, r0 = am.resistance(a300, 235.0, Pm, 100.0, "a", panneau=6000.0), am.resistance(a300, 235.0, Pm, 100.0, "a")
    e1, e1p = 68.70e6 / (s300["Wel_y"] * 235.0), 68.70e6 / (s300["Wpl_y"] * 235.0)
    c("… à M_Ed = 68,70 kN·m exactement : 380,08 et 387,76 kN ; avec a = 6,00 m (k_F = 6 + 2·(h_w/a)²) : 380,22 et 387,90 kN",
      [round(am.seuil_local(x["FRd"], e) / 1e3, 2) for x in (r0, r6) for e in (e1, e1p)] == [380.08, 387.76, 380.22, 387.90]
      and proche(r6["kF"], 6.0 + 2.0 * (278.6 / 6000.0) ** 2) and r6["ly"] <= 6000.0, (r6["kF"], r6["FRd"]))
    c("η₁ PLASTIQUE en variante (DÉCISION BUREAU n° 1, classes 1 et 2 : commentaire JRC p. 98) — R2 : η₁ = 68,70/(W_pl,y·f_y) = 0,465, taux "
      "33,2 % pour information ; le verdict reste l'élastique (36,6 %)",
      proche(L["sans"]["interaction_plastique"], (L["sans"]["eta2"] + 0.8 * 68.7015e6 / (s300["Wpl_y"] * 235.0)) / 1.4, 1e-9)
      and round(L["sans"]["interaction_plastique"], 3) == 0.332 and proche(vFM["taux"], L["sans"]["interaction"])
      and any("η₁ plastique (classe 1, variante)" in m and "33,2 %" in m for m in vFM["messages"]), L["sans"])
    a4 = ame(**dict(R2M, profil="IPE 400"))
    L4 = lieu(a4, "Charge à 3,00 m")
    m4 = " ".join(verif(a4, L4["base"] + "-FM")["messages"])
    c("… la classe est prise SOUS LES EFFORTS de l'occurrence : l'IPE 400 est de classe 3 en compression pure, de classe 1 sous ce moment — sa "
      "variante plastique est calculée (17,6 % < 19,3 % élastique)",
      ES.verifier({"section": SE.du_catalogue("IPE 400"), "nuance": "S235", "NEd": 1.0})["grandeurs"]["classe"] == 3
      and L4["sans"]["interaction_plastique"] < L4["sans"]["interaction"] and round(L4["sans"]["interaction_plastique"], 3) == 0.176
      and round(L4["sans"]["interaction"], 3) == 0.193 and "η₁ plastique (classe 1, variante)" in m4, (L4["sans"], m4))
    ct = {k: ame(appuis={"mode": "poses", "ss": 100.0, "c": 0.0, "contact": k}) for k in ("oui", "non", "inconnu")}
    mct = {k: " ".join(verif(ct[k], "A-L00-F")["messages"]) for k in ct}
    c("test 13 : contact d'extrémité SAISI (R4, type c, c = 0) — « oui » : 248,91 kN ; « non » : 199,82 kN, sans variante ; « inconnu » : "
      "les deux, verdict sur 199,82 kN, « hypothèse de contact non établie »",
      round(ct["oui"]["lieux"][0]["sans"]["F_Rd"], 2) == 248.91 and round(ct["non"]["lieux"][0]["sans"]["F_Rd"], 2) == 199.82
      and round(ct["inconnu"]["lieux"][0]["sans"]["F_Rd"], 2) == 199.82 and "hypothèse de contact non établie" in mct["inconnu"]
      and "248,9" in mct["inconnu"] and "pour information" not in mct["non"] and "contact d'extrémité établi" in mct["oui"]
      and [ct[k]["lieux"][0]["sans"]["contact"] for k in ("oui", "non", "inconnu")] == ["oui", "non", "inconnu"]
      and _refuse(lambda: en.normaliser(barre(appuis={"mode": "poses", "ss": 0.0, "c": 0.0, "contact": "peut-être"})))
      and "contact" not in en.normaliser(barre())["appuis"], mct)
    a = ame(cas=[{"nom": "G", "nature": "permanent", "charges": [P_(20.0, 3.0, 300.0)]}], appuis={"mode": "ame", "ss": 0.0, "c": 0.0})
    c("test 32 : s_s saisie 300 mm sous l'IPE 300 → s_s retenue = h_w = 278,6 mm (§ 6.3(1)), avec un message",
      a["lieux"][0]["ss"] == 278.6 and any("s_s bornée à h_w = 278,6 mm" in m for v_ in a["verifications"] for m in v_["messages"]), a["lieux"][0])
    sz = am.sigma_z(100e3, 100.0, 10.7, 7.1)
    c("test 33 : σ_z,Ed (EN 1993-1-5 § 3.2.3) — IPE 300, F_Ed = 100 kN, s_s = 100 mm, sans raidisseur : s_e = s_s + 2·t_f = 121,4 mm ; à la "
      "jonction (z = 0) σ_z,Ed = F_Ed/(s_e·t_w) = 116,0 N/mm² ; plus bas dans l'âme, b_eff s'élargit",
      round(sz["se"], 1) == 121.4 and round(sz["sigma_z"], 1) == 116.0 and proche(sz["beff"], 121.4)
      and am.sigma_z(100e3, 100.0, 10.7, 7.1, z=100.0)["sigma_z"] < sz["sigma_z"] and proche(am.sigma_z(100e3, 100.0, 10.7, 7.1, ast1=7.1)["n"], 0.636 * math.sqrt(1.878)), sz)
    t31 = copy.deepcopy(CONTINUE)
    t31["cas"][0]["charges"].append(P_(20.0, 5.0, 200.0))
    t31["appuis"] = {"mode": "ame", "ss": 0.0, "c": 0.0}
    a = ame(**t31)
    L = lieu(a, "Charge à 5,00 m")
    q_, zj, Sf = 1.35 * 6.0 + 1.5 * 8.0, 95.0 - 10.0, 200.0 * 10.0 * 90.0
    MB, VB, IH = q_ * 25.0 / 8.0, 5.0 * q_ * 5.0 / 8.0, pH["Iy"] * 1e4
    sx_, sz_, tau_ = -MB * 1e6 * zj / IH, 27e3 / ((170.0 + 20.0) * 6.5), VB * 1e3 * Sf / (IH * 6.5)
    crit31 = (sx_ / 235.0) ** 2 + (sz_ / 235.0) ** 2 - (sx_ / 235.0) * (sz_ / 235.0) + 3.0 * (tau_ / 235.0) ** 2
    vEL = verif(a, L["base"] + "-EL") if L else None
    c("test 31 : poutre continue HEA 200, charge de 27 kN sur la semelle HAUTE au droit de l'appui B (moment négatif) — semelle chargée TENDUE : "
      "pas d'interaction (7.2) ; η₂ ≤ 1 et le critère 6.2.1(5) à la jonction, σ_x = −144,6, σ_z,Ed = 21,9 (§ 3.2.3), τ = 47,1 N/mm² → 56,5 %",
      L is not None and L["sans"]["semelle"] == ["tendue"] and verif(a, L["base"] + "-FM") is None and vEL is not None
      and proche(vEL["valeur"], crit31, 1e-9) and round(crit31, 3) == 0.565 and vEL["combinaison"] == TOUT and L["decision"] == "aucun"
      and L["sans"]["interaction"] is None and proche(L["sans"]["critere"], crit31, 1e-9) and verif(a, L["base"] + "-F") is not None, (L, crit31))
    t31b = copy.deepcopy(t31)
    t31b["cas"][0]["charges"][-1] = P_(20.0, 5.0, 0.0)
    a = ame(**t31b)
    L = lieu(a, "Charge à 5,00 m")
    c("… la même charge sans longueur d'appui (s_s = 0 : σ_z,Ed = 207,7 N/mm²) : le critère dépasse 1 → raidisseur, et la raison le dit",
      L is not None and L["decision"] == "raidisseur" and "critère 6.2.1(5)" in L["raison"] and "tendue" in L["raison"] and L["sans"]["critere"] > 1.0, L)
    sig = [am.sigma_x(-500e3, 20e6, s300["A"], s300["Iy"], 150.0 - 5.35, Mz, s300["Iz"], y) for Mz in (0.0, 10e6) for y in (-75.0, 75.0)]
    c("test 36 : semelle haute d'une poutre sur deux appuis avec une TRACTION (N/A = 92,9 N/mm² > la compression de M_y, 34,6) → semelle "
      "tendue ; un M_z qui fait changer σ_x de signe sur la largeur → « les deux » (les deux contrôles, le plus défavorable retenu)",
      am.etat_semelle(sig[:2]) == am.TENDUE and am.etat_semelle(sig[2:]) == am.LES_DEUX and am.etat_semelle([30.0, 1.0]) == am.COMPRIMEE
      and am.etat_semelle([0.0]) == am.LES_DEUX and round(-sig[0], 1) == 58.3, sig)
    a = ame(travees=[2.0], poids_propre=False, appuis={"mode": "ame", "ss": 0.0, "c": 0.0},
            cas=[{"nom": "G", "nature": "permanent", "N": -400.0, "charges": [P_(20.0, 1.0, 50.0)]}])
    L = a["lieux"][0]
    c("… dans la barre : IPE 300, 2 m, N = −400 kN (traction), 20 kN au milieu sur la semelle haute — semelle tendue : « -EL », pas de « -FM »",
      L["sans"]["semelle"] == ["tendue"] and L["base"] + "-EL" in L["verifications"] and L["base"] + "-FM" not in L["verifications"], L)
    c("test 2 : I_st,min d'un raidisseur intermédiaire — 1,5·h_w³·t³/a² si a/h_w < √2, 0,75·h_w·t³ sinon : les deux se raccordent à a/h_w = √2",
      proche(ra.inertie_minimale(1000.0, 8.0, 1000.0 * math.sqrt(2.0)), 0.75 * 1000.0 * 8.0 ** 3)
      and proche(1.5 * 1000.0 ** 3 * 8.0 ** 3 / (1000.0 * math.sqrt(2.0)) ** 2, 0.75 * 1000.0 * 8.0 ** 3)
      and proche(ra.inertie_minimale(1000.0, 8.0, 1200.0), 1.5 * 1000.0 ** 3 * 8.0 ** 3 / 1200.0 ** 2)
      and proche(ra.inertie_minimale(1000.0, 8.0, 2000.0), 0.75 * 1000.0 * 8.0 ** 3) and proche(ra.inertie_minimale(1000.0, 8.0), 0.75 * 1000.0 * 8.0 ** 3))
    mince = SE.saisie("I", "soude_oxycoupe", h=1040.0, b=300.0, tw=6.0, tf=20.0, r=0.0, A=18000.0, Iy=3.3e9, Iz=9.0e7, iy=428.0, iz=70.7)
    em = am.elancement(mince, 355.0, Pm)
    lw_ = am.resistance_cisaillement(em, 355.0, Pm)["lambda_w"]
    Nst = ra.effort_cisaillement(400e3, 1000.0, 6.0, 355.0, lw_, 1.0)
    pN = ra.dimensionner(mince, 355.0, Pm, 150e3, Nst=Nst)
    J = RE.Journal("ame", "x")
    ra.ecrire(J, "A-L00", "Charge", mince, 355.0, Pm, dict(pN, cisaillement={"V": 400e3, "fyw": 355.0, "lambda_w": lw_}))
    vRb = verif(J.rendre(), "A-L00-Rb")
    vu = ra.vue(pN, 150e3, Nst)
    c("test 34 : raidisseur qui sert aussi au cisaillement (âme de 1 000 × 6 en S355, λ̄_w = 2,371) — N_st = V_Ed − h_w·t·f_yw/(√3·λ̄_w²·γ_M1) "
      "= 400 − 218,8 = 181,2 kN ; le flambement porte N = F_Ed + N_st (150 + 181,2), l'écrasement F_Ed seule",
      round(lw_, 3) == 2.371 and round(Nst / 1e3, 1) == 181.2 and ra.effort_cisaillement(100e3, 1000.0, 6.0, 355.0, lw_, 1.0) == 0.0
      and proche(pN["N"], 150e3 + Nst) and proche(pN["taux"], max(150e3 / pN["NcRd"], (150e3 + Nst) / pN["NbRd"]))
      and proche(vRb["valeur"], (150e3 + Nst) / 1e3) and vRb["critere"] == "N_Ed ≤ N_b,Rd" and proche(vu["sigma"], (150e3 + Nst) / pN["Ast"])
      and proche(vu["sigma_contact"], 150e3 / pN["Ac"]) and {f["gabarit"] for f in J.calcul} >= {"r_N_st", "r_N_Ed"}
      and all(proche(FO.evaluer(f), f["resultat"]) and W.relecture_juste(f) for f in J.calcul), (Nst, pN["taux"]))
    J = RE.Journal("ame", "x")
    sP = prs_section(300.0, 30.0, 1400.0, 10.0, 300.0, 30.0)
    am.cisaillement(J, sP, 355.0, Pm, 1500e3, 5000e6)
    f_prs = [f for f in J.calcul if f["gabarit"] == "a_M_f_prs"]
    c("… et l'interaction (7.1) d'un PRS élancé (300 × 30, 1 400 × 10, S355) : M_f,Rd = A_f·h_f·f_y/γ_M0 (gabarit des PRS), recalculé et relu",
      len(f_prs) == 1 and proche(FO.evaluer(f_prs[0]), f_prs[0]["resultat"]) and W.relecture_juste(f_prs[0])
      and proche(f_prs[0]["resultat"], 9000.0 * 1430.0 * 355.0 / 1e6) and verif(J.rendre(), "A-VbM") is not None, [f["gabarit"] for f in J.calcul])
    b18 = dict(travees=[2.0], poids_propre=False, appuis={"mode": "poses", "ss": 100.0, "c": 0.0}, maintiens=[{"x": 1.0, "genre": "complet"}],
               cas=[{"nom": "Q", "nature": "exploitation_A", "charges": [P_(300.0, 1.0, 100.0)]}])
    a = ame(**b18)
    L = lieu(a, "Charge à 1,00 m")
    av = L["avec"] if L else {}
    c("test 18 / § 15.4 : vue AVANT/APRÈS raidisseur — IPE 300 S235, type (a), F = 450 kN, s_s = 100 : sans raidisseur F_Rd = 387,8 kN, "
      "η₂ = 1,160 ; avec 2 plats 70 × 20 : f_y = 225 (plats de 20 mm), A_st = 4 454,3 mm², σ = 101,0 N/mm², N_b,Rd = 1 002,2 kN, taux 0,449 ; "
      "contact 2 200 mm², 204,5 ≤ 225 N/mm² (taux 0,909) ; rappel : un raidisseur ne réduit pas τ",
      L is not None and round(L["sans"]["F_Rd"], 1) == 387.8 and abs(L["sans"]["eta2"] - 1.160) <= 1e-3 and L["raidisseur"]["designation"] == "2 plats 70 × 20 mm"
      and round(av["A_st"], 1) == 4454.3 and round(av["sigma"], 1) == 101.0 and round(av["N_b_Rd"], 1) == 1002.2 and round(av["taux_flambement"], 3) == 0.449
      and av["A_c"] == 2200.0 and round(av["sigma_contact"], 1) == 204.5 and av["f_y_contact"] == 225.0 and round(av["taux_contact"], 3) == 0.909
      and L["rappel"] == ra.RAPPEL_TAU and round(L["sans"]["sigma_z"], 1) == round(450e3 / (121.4 * 7.1), 1)
      and sum(ra.RAPPEL_TAU in h for h in a["hypotheses"]) == 1 and not any(ra.RAPPEL_TAU in m for v_ in a["verifications"] for m in v_["messages"]), L)
    c("… sans raidisseur, pas de vue « avec » ni de rappel (R2, la charge de 36 kN)",
      lieu(ame(**R2M), "Charge à 3,00 m")["avec"] is None and lieu(ame(**R2M), "Charge à 3,00 m")["rappel"] is None
      and not any(ra.RAPPEL_TAU in h for h in ame(**R2M)["hypotheses"]))

    # ---- 4 ter. (reprise du lot B) les données manquantes, les cotes d'une semelle de PRS, le contact d'une charge, N_st par panneau
    nH, nB = am.niveaux_semelle(sC, "haut"), am.niveaux_semelle(sC, "bas")
    zgC = sC["z_g"]
    c("PRS-C monosymétrique : les cotes de la semelle chargée depuis SON centre de gravité z_g (contrat de prs.py), z vers le haut — haute : plan "
      "moyen h − z_g − t₁/2, jonction h − z_g − t₁, S_f = b₁·t₁·z ; basse : −(z_g − t₂/2), −(z_g − t₂), b₂·t₂ (et non h/2, ni la semelle haute)",
      abs(zgC - 750.0) > 50.0 and proche(nH["z_moyen"], 1500.0 - zgC - 15.0) and proche(nH["z_jonction"], 1500.0 - zgC - 30.0)
      and proche(nH["S"], 300.0 * 30.0 * (1500.0 - zgC - 15.0)) and (nH["bf"], nH["tf"]) == (300.0, 30.0)
      and proche(nB["z_moyen"], -(zgC - 25.0)) and proche(nB["z_jonction"], -(zgC - 50.0)) and proche(nB["S"], 500.0 * 50.0 * (zgC - 25.0))
      and (nB["bf"], nB["tf"]) == (500.0, 50.0), (zgC, nH, nB))

    def bout(*contacts):
        """Une charge de 10 kN par cas à 0,05 m du bout (s_s = 100 : c = 0, type c), chacune avec son contact (None : non saisi)."""
        chs = []
        for i_, k_ in enumerate(contacts):
            ch_ = P_(10.0, 0.05, 100.0)
            if k_:
                ch_["contact"] = k_
            chs.append({"nom": f"C{i_}", "nature": "permanent", "charges": [ch_]})
        return ame(appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 0.05, "genre": "haut"}], poids_propre=False, cas=chs)["lieux"][0]
    k1, k2, k3, k4, k5 = bout("oui"), bout("non"), bout("oui", "non"), bout("oui", "oui"), bout("oui", None)
    c("le CONTACT saisi sur une charge ponctuelle près du bout (type c, c = 0) : « oui » → 248,91 kN, « non » → 199,82 kN ; plusieurs charges au "
      "même endroit : établi seulement s'il l'est pour toutes, « non » dès qu'une le refuse, « inconnu » sinon",
      [x["type"] for x in (k1, k2, k3)] == ["c"] * 3 and round(k1["sans"]["F_Rd"], 2) == 248.91 and round(k2["sans"]["F_Rd"], 2) == 199.82
      and [x["sans"]["contact"] for x in (k1, k2, k3, k4, k5)] == ["oui", "non", "non", "oui", "inconnu"] and round(k4["sans"]["F_Rd"], 2) == 248.91
      and round(k5["sans"]["F_Rd"], 2) == 199.82 and en.normaliser(barre(cas=[{"nom": "G", "nature": "permanent", "charges": [
          dict(P_(1.0, 1.0, 10.0), contact="oui")]}]))["cas"][0]["charges"][0]["contact"] == "oui", [x["sans"] for x in (k1, k2, k3, k4, k5)])
    Pm_ = 87.5
    am_ = ame(travees=[2.0], poids_propre=False, appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 1.0, "genre": "complet"}],
              cas=[{"nom": "G", "nature": "permanent", "N": -400.0, "charges": [P_(Pm_, 1.0, 50.0)]}])
    Lm = am_["lieux"][0]
    sm_ = [-1.35 * 400e3 / s300["A"] + 1.35 * Pm_ * 2.0 / 4.0 * 1e6 * z_ / s300["Iy"] for z_ in (150.0 - 5.35, 150.0 - 10.7)]
    c("l'état de la semelle chargée se lit au PLAN MOYEN (brief § 8.2) : IPE 300 tendu (N = −540 kN) et fléchi (59,1 kN·m) — σ_x = +1,9 N/mm² "
      "au plan moyen, −1,8 à la jonction : semelle comprimée, interaction (7.2)",
      sm_[0] > 0.0 > sm_[1] and Lm["sans"]["semelle"] == ["comprimee"] and Lm["base"] + "-FM" in Lm["verifications"] and Lm["decision"] == "aucun", (sm_, Lm["sans"]))
    hog = {g_: lieu(ame(profil="HEA 200", modele="continue", travees=[5.0, 5.0], poids_propre=False, appuis={"mode": "poses", "ss": 100.0, "c": 0.0},
                        maintiens=[{"x": 4.6, "genre": g_}], cas=[{"nom": "G", "nature": "permanent", "charges": [q(6.0, 0.0, 10.0), P_(5.0, 4.6, 100.0)]}]),
                   "Charge à 4,60 m") for g_ in ("haut", "complet")}
    c("une charge sur la semelle haute TENDUE (moment négatif, poutre continue) : la semelle comprimée, la basse, doit être tenue aussi (§ 6.1(1)) "
      "— maintien « haut » seul : décision impossible, « Semelle basse sans maintien (EN 1993-1-5, 6.1(1)) — 5 % si elle est tenue ; ajouter un "
      "maintien (À VÉRIFIER). » (le harnais suit : « maintien de la semelle basse à préciser ») ; « complet » : décidé",
      hog["haut"]["sans"]["semelle"] == ["tendue"] and hog["haut"]["decision"] == "impossible"
      and hog["haut"]["raison"] == "Semelle basse sans maintien (EN 1993-1-5, 6.1(1)) — 5 % si elle est tenue ; ajouter un maintien (À VÉRIFIER)."
      and proche(hog["haut"]["taux"], hog["complet"]["taux"]) and hog["complet"]["decision"] == "aucun", hog)
    pr_ = [lieu(ame(appuis=k_, maintiens=[{"x": 0.1, "genre": "haut"}], poids_propre=False,
                    cas=[{"nom": "G", "nature": "permanent", "charges": [P_(10.0, 0.1, 50.0)]}]), "Charge à 0,10 m")
           for k_ in ({"mode": "poses", "ss": None, "c": None}, {"mode": "poses", "ss": 100.0, "c": 0.0})]
    c("une charge du type (c) près d'un appui de rive sans s_s ni c : son c dépend d'eux — la borne les prend à 0 (elle passe : décidée, et "
      "dit) ; saisis : décidée, sans borne",
      pr_[0]["type"] == "c" and pr_[0]["decision"] == "aucun" and pr_[0]["borne"] == "s_s et c de l'appui de rive = 0 : borne prudente"
      and pr_[1]["decision"] == "aucun" and pr_[1]["borne"] is None, [(x["decision"], x["borne"]) for x in pr_])
    a = ame(**dict(b18, maintiens=[{"x": 1.0, "genre": "haut"}]))
    L = lieu(a, "Charge à 1,00 m")
    c("un raidisseur sous une charge : 0,75·h_w suppose les DEUX semelles tenues (§ 9.4(2)) — la semelle haute seule : décision impossible, pas "
      "de plats proposés ; les appuis (des fourches) gardent les leurs",
      L["decision"] == "impossible" and L["raidisseur"] is None
      and L["raison"] == "Semelle basse sans maintien (EN 1993-1-5, 9.4(2)) : raidisseur nécessaire ; ajouter un maintien complet (À VÉRIFIER)."
      and all(x["decision"] == "raidisseur" for x in a["lieux"] if x["genre"] == "appui"), L["raison"])

    def ame_s(s_, **k):
        """Le bloc de l'âme d'une barre chargée faite d'une SECTION donnée (un PRS selon le contrat de prs.py), sans le catalogue."""
        b_ = en.normaliser(barre(**k))
        cas_ = ST.cas_complets(b_, s_)
        sols_ = ST.resoudre(b_, cas_, Pm["E"] * s_["Iy"] / 1e9)
        r_ = M["ame"].chargee(b_, s_, cas_, sols_, [c_.get("N", 0.0) for c_ in cas_], ST.ruptures(sols_))
        vus.append(r_)
        return r_
    sN = prs_section(300.0, 20.0, 1000.0, 6.0, 250.0, 20.0)
    aN = ame_s(sN, travees=[10.0], poids_propre=False, nuance="S355", appuis={"mode": "poses", "ss": 100.0, "c": 50.0},
               maintiens=[{"x": 5.0, "genre": "complet"}], cas=[{"nom": "G", "nature": "permanent", "charges": [q(50.0 / 1.35, 0.0, 10.0),
                                                                                                                    P_(300.0 / 1.35, 5.0, 100.0)]}])
    LN = lieu(aN, "Charge à 5,00 m")
    lwN = 1000.0 / (86.4 * 6.0 * math.sqrt(235.0 / 355.0))
    NstN = 375e3 - 1000.0 * 6.0 * 345.0 / (R3 * lwN ** 2)
    vN = verif(aN, LN["base"] + "-Rb") if LN else None
    c("test 34 dans une BARRE (PRS 300 × 20 / 1 000 × 6 / 250 × 20, S355, 10 m, q_Ed = 50 kN/m, F_Ed = 300 kN à mi-portée) : raidisseur ; N_st "
      "avec V_Ed à 0,5·h_w du bord du panneau le plus sollicité — l'appui : 400 − 50·0,5 = 375 kN (et non 150 kN au droit de la charge, qui "
      "donnerait N_st = 0) ; N = 300 + 162,4 kN",
      LN is not None and LN["decision"] == "raidisseur" and vN is not None and proche(vN["valeur"], (300e3 + NstN) / 1e3, 1e-6)
      and round(NstN / 1e3, 1) == 162.4 and vN["critere"] == "N_Ed ≤ N_b,Rd" and proche(LN["avec"]["sigma"], (300e3 + NstN) / LN["avec"]["A_st"], 1e-6)
      and all(x["decision"] == "raidisseur" and "§ 5.1(2)" in x["raison"] for x in aN["lieux"] if x["genre"] == "appui"), (vN, NstN))
    r = mo.verifier(barre(mode="efforts", profil="IPE 300", methodes=["ec3"], efforts={"service": {}, "elu": {"Vz": 400.0}},
                          flambement={"y": {"L": 6.0, "u": 1.0}, "z": {"L": 6.0, "u": 1.0}}))
    r1 = mo.verifier(barre(mode="efforts", profil="IPE 300", methodes=["ec3"], efforts={"service": {}, "elu": {"Vz": 100.0}},
                           flambement={"y": {"L": 6.0, "u": 1.0}, "z": {"L": 6.0, "u": 1.0}}))
    r2 = mo.verifier(barre(travees=[0.6], poids_propre=False, appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[],
                           cas=[{"nom": "G", "nature": "permanent", "charges": [q(2.0, 0.0, 0.6)]}]))
    c("brief § 8.3, point 8 : un LAMINÉ que V_pl,Rd fait échouer (IPE 300, V_z = 400 kN > 348,4 kN) — « le cisaillement gouverne : un raidisseur "
      "ne réduit pas τ — plaque de doublage d'âme ou profil plus lourd » ; à 100 kN, il passe : AUCUNE force concentrée n'appuie sur l'âme (efforts "
      "imposés, aucun lieu), pas de question de raidisseur — rien (le harnais suit : « gouverne » y était dit, sans le remède) ; de même une barre "
      "chargée par l'âme, sans charge ponctuelle (aucun lieu, E-Vz seul à 0,2 %)",
      verif(r["methodes"]["ec3"], "E-Vz")["statut"] == RE.NOK and AM.MOT_CISAILLEMENT in verif(r["methodes"]["ec3"], "E-Vz")["messages"]
      and AM.MOT_CISAILLEMENT_GOUVERNE not in verif(r["methodes"]["ec3"], "E-Vz")["messages"] and r1["ame"]["lieux"] == []
      and verif(r1["methodes"]["ec3"], "E-Vz")["statut"] == RE.OK
      and not any("cisaillement gouverne" in m_ for m_ in verif(r1["methodes"]["ec3"], "E-Vz")["messages"])
      and r2["ame"]["lieux"] == [] and not any("cisaillement gouverne" in m_ for m_ in verif(r2["methodes"]["ec3"], "E-Vz")["messages"]),
      (verif(r["methodes"]["ec3"], "E-Vz")["messages"], verif(r1["methodes"]["ec3"], "E-Vz")["messages"], verif(r2["methodes"]["ec3"], "E-Vz")["messages"]))
    a = ame(**R2)
    c("« -EL » : titre « critère élastique à la jonction (6.2.1(5)) » ; à un appui d'extrémité (σ_x = 0), « les deux contrôles » ; semelle tendue "
      "(test 31) : « le critère 6.2.1(5) décide »",
      verif(a, "A-L00-EL")["titre"] == "Appui A : critère élastique à la jonction (6.2.1(5))"
      and verif(a, "A-L00-EL")["messages"] == [am.MOTS_ETAT[am.LES_DEUX]] and vEL["messages"] == [am.MOTS_ETAT[am.TENDUE]], verif(a, "A-L00-EL"))

    # ---- 4 quater. (2e reprise du lot B) s_s et c NON SAISIS : la BORNE 0 (brief § 8.3, point 5 ; § 8.2), sa preuve et ses suites
    n_b, viol_b, naif_b = balayage(M["ame"], Pm, [("IPE 300", s300), ("IPE 750x147", SE.du_catalogue("IPE 750x147")), ("PRS N", sN0)],
                                   ((235.0, 235.0), (355.0, 355.0)), (0.0, 20.0, 60.0, 100.0, 200.0, 1e4), (0.0, 5.0, 20.0, 50.0, 200.0, 1000.0))
    c(f"PREUVE (balayage, {n_b} cas : IPE 300, IPE 750x147, un PRS ; S235, S355 ; types a, b, c déduits de la géométrie ; contact oui, non, "
      "inconnu ; s_s de 0 à plus que h_w, c de 0 à 1 000 mm) : F_Rd à la BORNE ≤ F_Rd réel partout — s_s et c non saisis (0, 0), s_s seul (0, c), "
      f"c seul (s_s, 0) ; la règle des branches seule, à la borne, dépasse le réel ({naif_b} cas) : la borne la complète",
      n_b > 2000 and not viol_b and naif_b > 0, (viol_b[:3], naif_b))
    aK = am.ame_de(SE.du_catalogue("IPE 750x147"))
    rK0, rK20, rKb = (am.resistance(aK, 235.0, Pm, 200.0, "c", 0.0), am.resistance(aK, 235.0, Pm, 200.0, "c", 20.0),
                      am.resistance(aK, 235.0, Pm, 200.0, "c", 0.0, borne=True))
    c("… pourquoi la règle des branches ne suffit pas (IPE 750x147 S235, type (c), s_s = 200 mm) : c = 0 → 704,0 kN (m₂ > 0 seule cohérente), "
      "c = 20 mm → 578,0 kN (m₂ = 0 redevient cohérente) ; à la borne c = 0 : la branche m₂ = 0, 566,2 kN ≤ 578,0 — et dite",
      round(rK0["FRd"] / 1e3, 1) == 704.0 and round(rK20["FRd"] / 1e3, 1) == 578.0 and round(rKb["FRd"] / 1e3, 1) == 566.2 and rKb["borne_m2"]
      and rKb["m2"] == 0.0 and rKb["lambda_F"] > 0.5 and not rKb["deux_branches"] and not rK0["borne_m2"]
      and proche(rKb["FRd"] / 1e3, frd_profil("IPE 750x147", 235.0, 200.0, "c", 0.0, borne=True))
      and any("borne : m₂ = 0, la plus petite des deux branches" in m for m in am.mots_resistance(rKb)), (rK0["FRd"], rK20["FRd"], rKb["FRd"]))
    a134, a200 = am.ame_de(SE.du_catalogue("IPE 750x134")), am.ame_de(SE.du_catalogue("IPE 200"))
    r134, r134b = am.resistance(a134, 355.0, Pm, 100.0, "c", 0.0), am.resistance(a134, 355.0, Pm, 100.0, "c", 0.0, borne=True)
    r200, r200b = am.resistance(a200, 355.0, Pm, 0.0, "b"), am.resistance(a200, 355.0, Pm, 0.0, "b", borne=True)
    ecarts = []
    for nom in I_DU_CATALOGUE[::5]:
        a_ = am.ame_de(SE.du_catalogue(nom))
        for fy in (235.0, 355.0):
            for ss, cc in ((0.0, 0.0), (100.0, 0.0), (0.0, 50.0)):
                if not proche(am.resistance(a_, fy, Pm, ss, "c", cc, borne=True)["FRd"] / 1e3, frd_profil(nom, fy, ss, "c", cc, borne=True), 1e-12):
                    ecarts.append((nom, fy, ss, cc))
    c("… la borne ne s'écarte de la règle des branches que si la branche m₂ = 0 PEUT redevenir cohérente (avec k_F = 6) : l'IPE 750x134 S355, "
      "type (c), s_s = 100, c = 0 — non : 555,4 kN, la valeur d'une note ancienne ; au type (b), jamais (IPE 200 S355, s_s = 0 : 81,8 kN) ; et le "
      f"témoin = le moteur à la borne, sur {len(I_DU_CATALOGUE[::5])} I du catalogue",
      proche(r134b["FRd"], r134["FRd"]) and round(r134["FRd"] / 1e3, 1) == 555.4 and r134["m2"] > 0 and not r134b["borne_m2"]
      and proche(r200b["FRd"], r200["FRd"]) and r200["m2"] > 0 and not ecarts, (r134b["FRd"], r200b["FRd"], ecarts[:3]))
    sur_ = ame(appuis={"mode": "poses", "ss": 100.0, "c": None}, maintiens=[{"x": 0.5, "genre": "complet"}],
               cas=[{"nom": "G", "nature": "permanent", "charges": [q(5.0, 0.0, 6.0), P_(150.0, 0.5, 100.0)]}])
    LA = lieu(sur_, "Appui A")
    c("… c non saisi, mais ce qui ne passe pas n'en dépend pas — le critère 6.2.1(5) à l'appui A (s_s = 100 saisie, σ_x = 0) dépasse 1 : le "
      "raidisseur est sûr, quel que soit c — décidé, jamais « c à saisir » (son âme disponible prise à c = 0)",
      LA["decision"] == "raidisseur" and LA["sans"]["critere"] > 1.0 and LA["raidisseur"] is not None and LA["manque"] is None, LA["raison"])
    grp = lieu(ame(appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 2.0, "genre": "haut"}],
                   cas=[{"nom": "G", "nature": "permanent", "charges": [P_(10.0, 2.0, 100.0)]}, {"nom": "Q", "nature": "exploitation_A", "charges": [P_(20.0, 2.0)]}]),
               "Charge à 2,00 m")
    c("plusieurs charges au MÊME endroit, l'une sans s_s (l'autre 100 mm) : la borne s_s = 0 pour le groupe — jamais la plus courte des SAISIES "
      "(la manquante peut être plus courte) : F_Rd = 199,8 kN (et non 387,8), « s_s = 0 : borne prudente »",
      grp["ss"] == 0.0 and proche(grp["sans"]["F_Rd"], frd_profil("IPE 300", 235.0, 0.0, "a")) and round(grp["sans"]["F_Rd"], 1) == 199.8
      and grp["decision"] == "aucun" and grp["borne"] == "s_s = 0 : borne prudente", grp)
    hb = ame(**dict(lourd, appuis={"mode": "poses", "ss": 200.0, "c": None}))
    c("c non saisi, mais la force de l'appui de rive y est du type (a) à c = 0 (HEB 300, s_s = 200 : 2 + 6·200/262 ≥ 6) : c n'y change rien — il "
      "n'est pas demandé, pas de borne ; décidé comme avec c = 0 saisi",
      all(x["type"] == "a" and x["decision"] == "aucun" and x["borne"] is None for x in hb["lieux"] if x["genre"] == "appui")
      and [x["decision"] for x in hb["lieux"]] == [x["decision"] for x in ame(**lourd)["lieux"]], [(x["type"], x["borne"]) for x in hb["lieux"]])
    rv = lieu(ame(appuis={"mode": "poses", "ss": None, "c": None}, maintiens=[{"x": 0.1, "genre": "haut"}], poids_propre=False,
                  cas=[{"nom": "G", "nature": "permanent", "charges": [P_(10.0, 0.1, 50.0)]}]), "Charge à 0,10 m")
    c("… une charge du type (c) près d'un appui de rive sans s_s ni c : son c en dépend — la borne les prend à 0, et le dit",
      rv["type"] == "c" and rv["decision"] == "aucun" and rv["borne"] == "s_s et c de l'appui de rive = 0 : borne prudente", rv["borne"])
    nst = {x_: lieu(ame_s(sN, travees=[10.0], poids_propre=False, nuance="S355", appuis={"mode": "poses", "ss": 100.0, "c": 50.0},
                          maintiens=[{"x": x_, "genre": "complet"}], cas=[{"nom": "G", "nature": "permanent", "charges": [
                              q(50.0 / 1.35, 0.0, 10.0), P_(300.0 / 1.35, x_, 100.0)]}]), f"Charge à {RE.fr(x_, 2)} m") for x_ in (8.0, 9.6)}
    vN8 = verif(vus[-2], nst[8.0]["base"] + "-Rb")
    vN96 = verif(vus[-1], nst[9.6]["base"] + "-Rb")
    c("test 34, la charge HORS du milieu (à 8 m) : N_st prend le PLUS GRAND des deux panneaux — le droit, V = 490 − 50·0,5 = 465 kN à 0,5·h_w de "
      "l'appui B : N = 300 + 465 − 212,6 = 552,4 kN (le panneau gauche seul : 372,4) ; à 9,6 m, le panneau droit (0,4 m < h_w) pris à "
      "mi-longueur : V = 528 kN, N = 615,4 kN",
      vN8 is not None and round(vN8["valeur"], 1) == 552.4 and vN96 is not None and round(vN96["valeur"], 1) == 615.4, (vN8 and vN8["valeur"], vN96 and vN96["valeur"]))
    pointe = {g_: lieu(ame(modele="console", travees=[2.0], poids_propre=False, maintiens=[{"x": 2.0, "genre": g_}],
                           cas=[{"nom": "G", "nature": "permanent", "charges": [P_(20.0, 2.0, 80.0)]}]), "Charge à 2,00 m") for g_ in ("haut", "complet")}
    loin = lieu(ame(**dict(R2, maintiens=[{"x": 3.1, "genre": "haut"}])), "Charge à 3,00 m")
    c("la règle du maintien (§ 6.1(1)), ses deux branches : au bout libre d'une console (M = 0 : σ_x nul, « les deux »), la semelle haute "
      "seule tenue ne suffit pas — « maintien de la semelle basse à préciser » ; « complet » : décidé ; un maintien à 0,10 m de la charge ne "
      "la tient pas (la même abscisse)",
      pointe["haut"]["sans"]["semelle"] == ["les_deux"] and pointe["haut"]["decision"] == "impossible"
      and pointe["haut"]["raison"].startswith("Semelle basse sans maintien (EN 1993-1-5, 6.1(1)) — ")
      and pointe["haut"]["raison"].endswith(" % si elle est tenue ; ajouter un maintien (À VÉRIFIER).")
      and pointe["complet"]["decision"] != "impossible" and loin["decision"] == "impossible" and loin["raison"].startswith("Semelle haute sans maintien"),
      (pointe["haut"]["raison"], pointe["complet"]["decision"], loin["raison"]))
    # (2e finition du lot B) les autres formes du message du maintien : le pluriel ; une donnée ET le maintien ; au-delà de 100 % ; sans le taux
    rk50, rk0 = (ame(modele="console", travees=[2.0], maintiens=[], cas=[{"nom": "G", "nature": "permanent", "charges": [P_(30.0, 2.0, ss_)]}])
                 for ss_ in (50.0, None))
    k50, k0 = lieu(rk50, "Charge à 2,00 m"), lieu(rk0, "Charge à 2,00 m")
    deux = "emelles haute et basse sans maintien (EN 1993-1-5, 6.1(1))"
    c("le message du maintien au PLURIEL (console de 2 m, 30 kN au bout libre, sans maintien, s_s = 50 mm : M = 0, σ_x nul — les deux semelles à "
      "tenir) : « Semelles haute et basse sans maintien (EN 1993-1-5, 6.1(1)) — 24 % si elles sont tenues ; ajouter un maintien (À VÉRIFIER). », "
      "redit par l'avertissement ; la ligne du lieu et la fenêtre -F, SANS le taux (leur pastille le donne)",
      k50 is not None and k50["decision"] == "impossible" and round(k50["taux"], 3) == 0.245
      and k50["raison"] == f"S{deux} — 24 % si elles sont tenues ; ajouter un maintien (À VÉRIFIER)."
      and k50["ligne"] == f"S{deux} ; ajouter un maintien (À VÉRIFIER)." and verif(rk50, k50["base"] + "-F")["messages"][0] == k50["ligne"]
      and [x["message"] for x in rk50["alertes"] if x["code"].startswith("decision_impossible")]
      == [f"Charge à 2,00 m : s{deux} — 24 % si elles sont tenues ; ajouter un maintien (À VÉRIFIER)."], (k50 and (k50["raison"], k50["ligne"])))
    c("… la même console, s_s NON saisie (la borne ne passe pas : critère 6.2.1(5) à 130 %) : la donnée ET le maintien, dans la même phrase — "
      "« Décision impossible : s_s à saisir ; semelles haute et basse sans maintien (EN 1993-1-5, 6.1(1)). » (saisir s_s seule ne suffirait pas)",
      k0 is not None and k0["decision"] == "impossible" and k0["raison"] == f"Décision impossible : s_s à saisir ; s{deux}." and k0["ligne"] is None
      and verif(rk0, k0["base"] + "-F")["messages"][0] == k0["raison"], k0 and k0["raison"])
    q0, qh, qc = (lieu(ame(maintiens=m_, appuis={"mode": "poses", "ss": 100.0, "c": 50.0},
                           cas=[{"nom": "G", "nature": "permanent", "charges": [P_(120.0, 3.0, 100.0)]}]), "Charge à 3,00 m")
                  for m_ in ([], [{"x": 3.0, "genre": "haut"}], [{"x": 3.0, "genre": "complet"}]))
    c("… au-delà de 100 % (IPE 300, 6 m, 120 kN à mi-portée, tout saisi, sans maintien : 137 % si elle est tenue) : le remède dit le raidisseur ET "
      "le maintien COMPLET que sa longueur demande (§ 9.4(2)) — « … ; ajouter un maintien complet et un raidisseur (À VÉRIFIER). » ; la preuve : un "
      "maintien « haut » seul ne suffit pas (la longueur du raidisseur), « complet » pose le raidisseur",
      q0["decision"] == "impossible"
      and q0["raison"] == "Semelle haute sans maintien (EN 1993-1-5, 6.1(1)) — 137 % si elle est tenue ; ajouter un maintien complet et un raidisseur (À VÉRIFIER)."
      and q0["ligne"] == "Semelle haute sans maintien (EN 1993-1-5, 6.1(1)) ; ajouter un maintien complet et un raidisseur (À VÉRIFIER)."
      and qh["decision"] == "impossible" and qc["decision"] == "raidisseur"
      # (intégration du lot B, point 29 d : le fait, la décision et le remède, sur le modèle de sans_maintien — le harnais suit)
      and qh["raison"] == "Semelle basse sans maintien (EN 1993-1-5, 9.4(2)) : raidisseur nécessaire ; ajouter un maintien complet (À VÉRIFIER).",
      (q0["raison"], qh["raison"], qc["decision"]))
    cis = mo.verifier(barre(travees=[1.5], poids_propre=False, appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 0.75, "genre": "complet"}],
                            cas=[{"nom": "G", "nature": "permanent", "charges": [q(200.0, 0.0, 1.5), P_(5.0, 0.75, 100.0)]}]))
    vz, vz0 = verif(cis["methodes"]["ec3"], "E-Vz"), verif(mo.verifier(barre())["methodes"]["ec3"], "E-Vz")
    peu = mo.verifier(barre(maintiens=[], appuis={"mode": "poses", "ss": 200.0, "c": 100.0}))
    vzp = verif(peu["methodes"]["ec3"], "E-Vz")
    tous = [v_["taux"] for r_ in (cis["methodes"]["ec3"], cis["ame"]) for v_ in r_["verifications"] if v_["id"] != "E-Vz" and v_["taux"] is not None]
    c("brief § 8.3, point 8 : le cisaillement GOUVERNE (IPE 300, 1,5 m, 200 kN/m) — E-Vz passe (58 %) et a le plus fort taux de l'Eurocode 3 "
      "et de l'âme : « le cisaillement gouverne », dit, sans le remède ; la barre neuve (E-Vz 13 %) : rien ; une barre peu chargée où E-Vz "
      "(13 %) dépasse les forces concentrées (10 %) mais que le déversement gouverne (114 %) : rien (le harnais suit : le message y était posé)",
      vz["statut"] == RE.OK and vz["messages"].count(AM.MOT_CISAILLEMENT_GOUVERNE) == 1 and AM.MOT_CISAILLEMENT not in vz["messages"]
      and vz["taux"] >= max(tous) and AM.MOT_CISAILLEMENT_GOUVERNE not in vz0["messages"]
      and round(vzp["taux"], 3) == 0.128 and max(v_["taux"] for v_ in peu["ame"]["verifications"] if v_["taux"]) < vzp["taux"]
      and peu["methodes"]["ec3"]["taux_max"] > 1.0 and not any("cisaillement gouverne" in m for m in vzp["messages"]),
      (vz["taux"], max(tous), vzp["messages"], peu["methodes"]["ec3"]["taux_max"]))
    sP6 = prs_section(300.0, 60.0, 200.0, 10.0, 300.0, 60.0)
    rP6 = ame_s(sP6, modele="console", travees=[2.0], poids_propre=False, appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 1.0, "genre": "complet"}],
                cas=[{"nom": "G", "nature": "permanent", "N": 560.0, "charges": [q(150.0, 0.0, 2.0), P_(1.5, 1.0)]}])
    LP6 = rP6["lieux"][0]
    fse = next((f for f in rP6["calcul"] if f["cle"] == LP6["base"] + "-EL/se"), None)
    pire6 = M["ame"].critere_pire(2025.0, [0.0, 200.0], 60.0, 10.0, 4.0, 0.0, 345.0, 1.0)
    c("le critère 6.2.1(5), s_s non saisie : au PIRE de ses deux bouts (quadratique en σ_z) — 0 d'ordinaire ; h_w quand σ_x à la jonction dépasse "
      "σ_z (PRS 300 × 60 / 200 × 10, console comprimée et fléchie, semelle haute tendue au plan moyen : s_e = h_w + 2·t_f = 320 mm)",
      LP6["sans"]["semelle"] == ["tendue"] and fse is not None and fse["resultat"] == 320.0 and pire6["ss"] == 200.0
      and M["ame"].critere_pire(2025.0, [0.0, 200.0], 60.0, 10.0, 0.0, 0.0, 345.0, 1.0)["ss"] == 0.0, (fse and fse["resultat"], pire6))
    vP6 = verif(rP6, LP6["base"] + "-EL")
    rP50 = ame_s(sP6, modele="console", travees=[2.0], poids_propre=False, appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 1.0, "genre": "complet"}],
                 cas=[{"nom": "G", "nature": "permanent", "N": 560.0, "charges": [q(150.0, 0.0, 2.0), P_(1.5, 1.0, 50.0)]}])
    L50 = rP50["lieux"][0]
    f50 = next((f for f in rP50["calcul"] if f["cle"] == L50["base"] + "-EL/se"), None)
    v50 = verif(rP50, L50["base"] + "-EL")
    neuve_el = verif(ame(), "A-L00-EL")
    c("… la fenêtre le DIT quand elle imprime s_s = h_w (s_s non saisie, le bout h_w le pire) : « s_s non saisie : le pire de 0 et h_w. » ; la "
      "barre neuve (le bout 0, s_s = 0 imprimée) ne le dit pas ; s_s SAISIE (50 mm) : le critère à la saisie seule, jamais au bout h_w — "
      "s_e = 50 + 2·60 = 170 mm, sans le message",
      vP6 is not None and AME.MOT_PIRE_HW in vP6["messages"] and neuve_el is not None and AME.MOT_PIRE_HW not in neuve_el["messages"]
      and f50 is not None and f50["resultat"] == 170.0 and v50 is not None and AME.MOT_PIRE_HW not in v50["messages"]
      and L50["sans"]["semelle"] == ["tendue"], (vP6 and vP6["messages"], f50 and f50["resultat"], v50 and v50["messages"]))

    # ---- (finition du lot B) le filtre « critère 6.2.1(5) seul en cause », ses deux clauses ; § 8.3 point 8 contre l'âme aussi
    L9 = ame(poids_propre=False, appuis={"mode": "poses", "ss": 100.0, "c": None}, maintiens=[{"x": 3.0, "genre": "complet"}],
             cas=[{"nom": "G", "nature": "permanent", "N": 900.0, "charges": [q(40.0, 0.0, 6.0)]}])["lieux"][0]
    c("le filtre « critère 6.2.1(5) seul en cause » regarde AUSSI l'interaction : IPE 300, 6 m, N = 900 kN, 40 kN/m, s_s = 100 mm, c non saisi — "
      "la semelle est comprimée, η₂ = 0,81 passe mais (η₂ + 0,8·η₁)/1,4 = 1,13 non, à la borne c = 0 : « décision impossible : c à saisir », "
      "jamais un raidisseur décidé sans c",
      L9["nom"] == "Appui A" and L9["decision"] == "impossible" and L9["raison"] == "Décision impossible : c à saisir." and L9["raidisseur"] is None
      and L9["sans"]["eta2"] <= 1.0 and L9["sans"]["interaction"] > 1.0 and L9["sans"]["semelle"] == ["comprimee"],
      (L9["decision"], L9["raison"], L9["sans"]["eta2"], L9["sans"]["interaction"]))
    Lb = lieu(ame(poids_propre=False, appuis={"mode": "poses", "ss": None, "c": None}, maintiens=[{"x": 0.1, "genre": "complet"}],
                  cas=[{"nom": "G", "nature": "permanent", "N": -500.0, "charges": [q(1.0, 0.0, 6.0), P_(40.0, 0.1)]}]), "Charge à 0,10 m")
    c("… et il vide les données de l'appui de rive : une charge de 40 kN à 0,10 m de l'appui A (type c), rien de saisi, la barre TENDUE "
      "(N = −500 kN : semelle chargée tendue) — seul le critère 6.2.1(5) ne passe pas (η₂ = 0,27) : « s_s à saisir », sans redemander s_s et c "
      "de l'appui de rive (le critère ne les lit pas)",
      Lb is not None and Lb["type"] == "c" and Lb["decision"] == "impossible" and Lb["raison"] == "Décision impossible : s_s à saisir."
      and Lb["sans"]["eta2"] <= 1.0 and Lb["sans"]["critere"] > 1.0 and Lb["sans"]["semelle"] == ["tendue"], Lb and (Lb["raison"], Lb["sans"]))
    p8 = mo.verifier(barre(travees=[1.0], poids_propre=False, appuis={"mode": "poses", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 0.5, "genre": "complet"}],
                           cas=[{"nom": "G", "nature": "permanent", "charges": [q(150.0, 0.0, 1.0)]}]))
    vz8 = verif(p8["methodes"]["ec3"], "E-Vz")
    ec8 = [v_["taux"] for v_ in p8["methodes"]["ec3"]["verifications"] if v_["id"] != "E-Vz" and v_["taux"] is not None]
    c("§ 8.3 point 8 : E-Vz comparé à l'Eurocode 3 ET à l'âme — IPE 300, 1 m, 150 kN/m, posée (s_s = c = 0) : E-Vz (29 %) dépasse le reste "
      "de l'Eurocode 3, mais les raidisseurs des appuis portent 49 % : le cisaillement ne gouverne pas, rien n'est dit (un raidisseur y est "
      "proposé)",
      vz8["statut"] == RE.OK and round(vz8["taux"], 2) == 0.29 and vz8["taux"] > max(ec8)
      and all(x["decision"] == "raidisseur" and round(x["taux"], 2) == 0.49 for x in p8["ame"]["lieux"])
      and not any("cisaillement gouverne" in m_ for m_ in vz8["messages"]), (vz8["taux"], max(ec8), vz8["messages"]))
    prox = {k_: ame(appuis={"mode": "ame", "ss": 0.0, "c": 0.0}, maintiens=[{"x": 3.0}, {"x": 3.3}],
                    cas=[{"nom": "G", "nature": "permanent", "charges": [P_(20.0, 3.0, k_), P_(20.0, 3.3, k_)]}]) for k_ in (None, 50.0)}
    c("deux charges à 300 mm, s_s non saisie : leurs l_y pourraient se recouvrir (la plus grande : s_s = h_w, 426 mm) — « A-GR » non couvert ; "
      "s_s = 50 saisie (l_y = 169,8) : rien",
      verif(prox[None], "A-GR") is not None and verif(prox[50.0], "A-GR") is None
      and round(am.ly_plafond(a300, a300["hw"]), 1) == 426.0, [x["id"] for x in prox[None]["verifications"]])

    # ---- 5. la barre
    r = mo.verifier(barre(mode="efforts", profil="IPE 750x134", nuance="S355", methodes=["ec3"],
                          efforts={"service": {}, "elu": {"Vz": 1800.0}}, flambement={"y": {"L": 6.0, "u": 1.0}, "z": {"L": 6.0, "u": 1.0}}))
    vus.append(r["ame"])
    c("efforts imposés, V_z,Ed = 1 800 kN sur l'IPE 750x134 S355 : l'Eurocode 3 passe (V_pl,z,Rd = 1 965,4 kN) mais l'âme voile "
      "(V_bw,Rd = 1 722,0 kN) — la barre est NON CONFORME",
      r["methodes"]["ec3"]["statut"] == RE.OK and r["ame"]["statut"] == RE.NOK and r["statut"] == RE.NOK
      and proche(verif(r["ame"], "A-Vb")["taux"], 1800.0 / Vbw) and verif(r["methodes"]["ec3"], "A-Vb") is None
      and any("ne réduit pas τ" in m for m in verif(r["ame"], "A-Vb")["messages"]),
      (r["methodes"]["ec3"]["statut"], r["ame"]["statut"], r["statut"]))
    c("… efforts imposés : aucun lieu (les appuis et les charges ne sont pas décrits), et l'hypothèse le dit",
      r["ame"]["lieux"] == [] and any("Efforts imposés" in h for h in r["ame"]["hypotheses"]) and r["ame"]["combinaisons"] == [])
    r = mo.verifier(barre(methodes=["historique"]))
    vus.append(r["ame"])
    # (lot C, C0d — le harnais suit : les DEUX méthodes sont toujours calculées, « methodes » ne décide plus du calcul)
    c("méthode historique « seule » (une ancienne barre) : les deux méthodes et le bloc de l'âme sont calculés (ELU), et comptent dans le statut "
      "de la barre",
      list(r["methodes"]) == ["historique", "ec3"] and len(r["ame"]["lieux"]) == 2 and r["ame"]["norme"] == "NBN EN 1993-1-5"
      and r["ame"]["methode"] == "ame")
    r = mo.verifier(barre(profil="HD 400x818.0", nuance="S355"))
    c("une épaisseur hors du domaine du module (HD 400x818, t_f = 97 mm > 80 mm) : le bloc de l'âme est « non couvert », avec son motif",
      r["ame"]["statut"] == RE.NON_COUVERT and "80 mm" in r["ame"]["alertes"][0]["message"] and r["ame"]["lieux"] == [], r["ame"]["statut"])
    vieux = barre()
    del vieux["appuis"]
    nb = en.normaliser(vieux)
    c("une note d'avant le lot 4 (sans « appuis », sans « ss ») se relit : barre posée, s_s et c NON SAISIS (None, jamais 0 — le harnais suit : "
      "0 avant la reprise) ; une charge ponctuelle sans « ss » : None ; une valeur saisie, même 0, est gardée",
      nb["appuis"] == {"mode": "poses", "ss": None, "c": None, "detail": None}           # (lot C, C0e : les appuis un par un, vides)
      and en.normaliser(barre(cas=[{"nom": "G", "nature": "permanent", "charges": [P_(5.0, 1.0), q(1.0, 0.0, 6.0), P_(5.0, 2.0, 0.0)]}]))["cas"][0]["charges"]
      == [{"type": "P", "v": 5.0, "a": 1.0, "ss": None, "niveau": None, "introduction": "auto"}, {"type": "q", "v": 1.0, "a": 0.0, "b": 6.0, "niveau": None},
          {"type": "P", "v": 5.0, "a": 2.0, "ss": 0.0, "niveau": None, "introduction": "auto"}]      # (lot C, C0e : niveau et introduction, leurs défauts)
      and en.normaliser(barre(appuis={"mode": "poses", "ss": 0.0, "c": 0.0}))["appuis"] == {"mode": "poses", "ss": 0.0, "c": 0.0, "detail": None}, nb["appuis"])
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
    textes = [t for x in vus for t in (x.get("hypotheses") or []) + [a_["message"] for a_ in x.get("alertes") or []]
              + [m_ for v_ in x["verifications"] for m_ in v_.get("messages") or []] + [L_.get("raison") or "" for L_ in x.get("lieux") or []]]
    c("… aucun texte rendu (hypothèses, avertissements, messages, raisons) ne cite le brief, document interne — la référence reste en commentaire "
      "(« s_s ou c non saisi : calculé à sa borne 0 (À VÉRIFIER). »)",
      textes and not [t for t in textes if "brief" in t] and any(t == "s_s ou c non saisi : calculé à sa borne 0 (À VÉRIFIER)." for t in textes),
      [t for t in textes if "brief" in t][:2])
    gab = {f["gabarit"] for x in vus for f in x["calcul"]}
    manquent = sorted(set(FO.GABARITS_EC3_AME) - gab - {"a_eta3", "a_M_f", "a_eta1_bar", "a_71", "a_V_bw_n", "a_semelle", "a_eta1", "a_eta1_N",
                                                        "a_M_f_prs", "r_N_st", "r_N_Ed"})
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
    mods["raidisseurs"].AM = mods["ec3_ame"]
    mods["moteur"].AME, mods["moteur"].EN = mods["ame"], mods["entree"]
    return mods


MUTATIONS = [
    ("ec3_ame", "k_F du type (a) : 6 → 3,5", 'KF = {"a": 6.0, "b": 3.5}', 'KF = {"a": 3.5, "b": 3.5}'),
    ("ec3_ame", "k_F du type (b) : 3,5 → 6", 'KF = {"a": 6.0, "b": 3.5}', 'KF = {"a": 6.0, "b": 6.0}'),
    ("ec3_ame", "k_F du type (c) sans plafond", "    kF = KF[type_] + voisin if type_ != \"c\" else min(brut, KF_C_MAX)", "    kF = KF[type_] + voisin if type_ != \"c\" else brut"),
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
    ("ec3_ame", "lot A — η des sections soudées pour un laminé", '    return P["eta_soude"] if est_soude(s) else P["eta"]', '    return P["eta_soude"]'),
    ("ec3_ame", "lot A — η = 1 pour une section soudée", '    return P["eta_soude"] if est_soude(s) else P["eta"]', '    return P["eta"]'),
    ("ec3_ame", "lot B — un PRS aux bords non saisis pris pour un laminé (η = 1,00)", '    return bool(s.get("prs")) or s.get("fabrication") in SOUDES_ETA',
     '    return s.get("fabrication") in SOUDES_ETA'),
    # (intégration du lot B : hw_de lit section.hauteur_ame, la seule source du h_w d'un PRS — l'altération porte sur son appel)
    ("ec3_ame", "lot B — h_w d'un PRS : h − 2·t_f", "    return SE.hauteur_ame(s)\n",
     '    return s["h"] - 2.0 * s["tf"] if s.get("forme") == "I" and s.get("tf") else SE.hauteur_ame(s)\n'),
    ("ec3_ame", "lot B — les semelles d'un PRS ignorées (b et t_f de la section)", '    if p:\n        return {"haut": {"bf": float(p["b1"])',
     '    if False:\n        return {"haut": {"bf": float(p["b1"])'),
    ("ec3_ame", "lot B — M_f,Rd avec la plus GRANDE semelle", '    Af = min(x["bf"] * x["tf"] for x in sm.values())', '    Af = max(x["bf"] * x["tf"] for x in sm.values())'),
    ("ec3_ame", "lot B — § 8 avec la plus petite semelle (ANB 8(2) oubliée)", '    fl = max(semelles_de(s).values(), key=lambda x: x["bf"] * x["tf"])',
     '    fl = min(semelles_de(s).values(), key=lambda x: x["bf"] * x["tf"])'),
    ("ec3_ame", "lot B — § 8 avec la f_y de calcul (et non celle des semelles)", "    fyf = fy if fy_semelle is None else float(fy_semelle)", "    fyf = fy"),
    ("ame", "lot B — force transmise d'un PRS : la semelle qui donne la plus GRANDE F_Rd", '        out["b"] = min(essais, key=lambda r: r["FRd"])',
     '        out["b"] = max(essais, key=lambda r: r["FRd"])'),
    ("raidisseurs", "lot B — raidisseur d'un PRS : largeur prise sur la plus large semelle", '        b = min(prs["b1"], prs["b2"])', '        b = max(prs["b1"], prs["b2"])'),
    ("raidisseurs", "lot B — raidisseur d'un PRS : chanfrein sans le cordon", '        conge = prs["a"] * math.sqrt(2.0) if prs.get("a") else s["tw"]', '        conge = s["tw"]'),
    ("ec3_ame", "lot A — les caissons soudés comptés parmi les laminés (η = 1,00)", 'SOUDES_ETA = SOUDES + ("caisson", "caisson_recuit")', "SOUDES_ETA = SOUDES"),
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
    ("ec3_ame", "interaction (7.1) : M_f,Rd avec h", '    hf = s["h"] - sm["haut"]["tf"] / 2.0 - sm["bas"]["tf"] / 2.0', '    hf = s["h"]'),
    ("ec3_ame", "interaction (7.1) sous effort normal : rendue sans le dire", '    if s["forme"] != "I" or N != 0:', '    if s["forme"] != "I":'),
    ("ec3_ame", "voilement induit par la semelle : k = 4", "K_SEMELLE = 0.4 ", "K_SEMELLE = 0.004 "),
    # lot C, C1b : la f_y du plat est lue dans la liste du module Barres (nuances.py, S420 et la norme produit compris)
    ("raidisseurs", "lot A — le plat à la f_y de la barre (épaisseur oubliée)", '    fyp = fy if nuance is None else NU.fy(nuance, ts, "lamine", norme_produit)', "    fyp = fy"),
    ("raidisseurs", "lot A — f_y de calcul du raidisseur : celle du plat seul", '"fy_calcul": min(fyp, fyw)', '"fy_calcul": fyp'),
    ("raidisseurs", "lot A — 15·ε·t_w avec l'ε du plat", '    plein = AME_PARTICIPANTE * m["eps_ame"] * tw', '    plein = AME_PARTICIPANTE * m["eps_plat"] * tw'),
    ("raidisseurs", "lot A — écrasement à la f_y de calcul (et non à celle du plat)", "NcRd=Ac * fyp / P[\"gamma_M0\"]", "NcRd=Ac * fyd / P[\"gamma_M0\"]"),
    ("ame", "lot A — le raidisseur dimensionné sans la nuance (une f_y pour tout)", 'appui_rigide=rigide, nuance=b["nuance"], fy_ame=fyw,', "appui_rigide=rigide,"),
    ("ame", "lot A — l'âme prise à la f_y de calcul", '    return d["fy"], d["fy_parois"]["ame"], d["fy_source"]', '    return d["fy"], d["fy"], d["fy_source"]'),
    ("ec3_ame", "lot B — semelle tendue jamais reconnue", "    if all(x < -EPS_SIGMA for x in v):\n        return TENDUE", "    if False:\n        return TENDUE"),
    ("ame", "lot B — interaction (7.2) appliquée à une semelle tendue", '            comprimee = any(x["etat"] != AM.TENDUE for x in ctrl)', "            comprimee = True"),
    ("ame", "lot B — critère 6.2.1(5) de la semelle tendue oublié", '            o["el"] = max((x for x in ctrl if x["crit"] is not None), key=lambda x: x["crit"], default=None)',
     '            o["el"] = None'),
    ("ame", "lot B — raidisseur jamais décidé sur le critère 6.2.1(5)", '        elif pire["crit"] is not None and pire["crit"] > 1.0:', "        elif False:"),
    ("ec3_ame", "lot B — σ_z,Ed : s_e sans 2·t_f", "    se = ss + 2.0 * tf", "    se = ss + tf"),
    ("ec3_ame", "lot B — critère 6.2.1(5) : terme croisé de signe inversé", "    return (sx / f) ** 2 + (sz / f) ** 2 - (sx / f) * (sz / f) + 3.0 * (tau / f) ** 2",
     "    return (sx / f) ** 2 + (sz / f) ** 2 + (sx / f) * (sz / f) + 3.0 * (tau / f) ** 2"),
    ("ec3_ame", "lot B — critère 6.2.1(5) sans τ", "    return (sx / f) ** 2 + (sz / f) ** 2 - (sx / f) * (sz / f) + 3.0 * (tau / f) ** 2",
     "    return (sx / f) ** 2 + (sz / f) ** 2 - (sx / f) * (sz / f)"),
    ("ec3_ame", "lot B — seuil local : F_Rd seule (l'interaction oubliée)", "    return max(0.0, min(FRd, (lim - k * e1) * FRd))", "    return FRd"),
    ("ame", "lot B — variante η₁ plastique : W_el", '                Wp = s["Wpl_y"] if a_pl else W', "                Wp = W"),
    ("ec3_ame", "lot B — k_F sans le raidisseur voisin (a ignoré)", "    voisin = 0.0 if panneau is None else 2.0 * (hw / float(panneau)) ** 2", "    voisin = 0.0"),
    ("ec3_ame", "lot B — contact « oui » ignoré", '    contact = etat == "oui"', "    contact = False"),
    ("ec3_ame", "lot B — contact « non » : la variante avec contact donnée quand même", '    if type_ == "c" and etat == "inconnu" and ss > 0:',
     '    if type_ == "c" and etat != "oui" and ss > 0:'),
    ("ame", "lot B — contact saisi perdu au lieu", '        contact = (L.get("contact") or {}).get(cote) or "inconnu"', '        contact = "inconnu"'),
    ("entree", "lot B — contact des appuis perdu à la relecture", '        b["appuis"]["contact"] = ap["contact"]', "        pass"),
    ("raidisseurs", "lot B — N_st toujours nul", "    return max(0.0, V - hw * t * fyw / (math.sqrt(3.0) * lambda_w ** 2 * gM1))", "    return 0.0"),
    ("raidisseurs", "lot B — flambement du raidisseur sans N_st", '"taux": max(F / p["NcRd"], (F + Nst) / p["NbRd"])', '"taux": max(F / p["NcRd"], F / p["NbRd"])'),
    ("raidisseurs", "lot B — I_st,min : seuil a/h_w = 1 au lieu de √2", "    if a is not None and a / hw < math.sqrt(2.0):", "    if a is not None and a / hw < 1.0:"),
    ("raidisseurs", "lot B — vue avec raidisseur : σ sous F seule", '"sigma": N / p["Ast"]', '"sigma": F / p["Ast"]'),
    ("ame", "lot B — rappel « ne réduit pas τ » oublié (une fois par bloc, dans les hypothèses)", "(EN 1993-1-8). \"\n                   + RA.RAPPEL_TAU)",
     "(EN 1993-1-8).\")"),
    ("ec3_ame", "lot B — cisaillement qui gouverne : rien n'est dit", "    if V > r[\"Vbw\"]:", "    if False:"),
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
    ("raidisseurs", "le plat dépasse la semelle (largeur b/2)", '    bs_max = PAS * math.floor((b - s["tw"]) / 2.0 / PAS + 1e-9)', '    bs_max = PAS * math.floor(b / 2.0 / PAS + 1e-9) + PAS'),
    ("raidisseurs", "chanfrein arrondi vers le bas", "    chanfrein = PAS * math.ceil(conge / PAS - 1e-9)", "    chanfrein = PAS * math.floor(conge / PAS)"),
    ("raidisseurs", "un IPN sans chanfrein (son congé n'est pas sous « r »)", '    conge = s.get("r") or s["tw"] ', '    conge = s.get("r") or 0.0 '),
    ("raidisseurs", "la première épaisseur rendue sans la vérifier", "        if p[\"ok\"]:\n            return p", "        if True:\n            return p"),
    ("raidisseurs", "le flambement oublié dans le choix de l'épaisseur", '"taux": max(F / p["NcRd"], (F + Nst) / p["NbRd"])', '"taux": (F + Nst) / p["NbRd"]'),
    ("raidisseurs", "rigidité d'appui jamais demandée", '        rigide = (p["Ist"] >= p["I_min"]) if appui_rigide else None', "        rigide = None"),
    ("ame", "appuis par l'âme vérifiés comme posés", '    poses = ap["mode"] == "poses"', "    poses = True"),
    ("ame", "encastrement pris pour un appui posé", '            if a["type"] == "appui":                 # un appui : une fourche', '            if True:                 # un appui : une fourche'),
    # (lot C, C1 : une hauteur PAR CHARGE — les textes visés ont bougé, le harnais suit)
    ("ame", "charges ponctuelles vérifiées quel que soit leur niveau", 'if ch["type"] == "P" and ch["v"] and niv in ("haut", "bas"):', 'if ch["type"] == "P" and ch["v"]:'),
    ("ame", "une charge vers le haut sur la semelle haute vérifiée", '    return {"haut": P["haut"] if P["haut"] > EPS else 0.0,', '    return {"haut": abs(P["haut"]),'),
    ("ame", "une charge suspendue vérifiée comme si elle appuyait", '"bas": (R if R > EPS else 0.0) + (-P["bas"] if P["bas"] < -EPS else 0.0),', '"bas": (R if R > EPS else 0.0) + abs(P["bas"]),'),
    ("ame", "réaction non pondérée par les combinaisons", '        R = sum(f * sol.appuis[L["appui"]]["R"] for f, sol in zip(facteurs, sols) if f)', '        R = sum(sol.appuis[L["appui"]]["R"] for f, sol in zip(facteurs, sols) if f)'),
    ("ame", "charge ponctuelle non pondérée", '            P[niv] += facteurs[part[0]] * part[1]', '            P[niv] += part[1]'),
    ("ame", "longueur d'appui d'un groupe : la plus longue", '            hote["ss"][niv] = min(g["ss"][niv]) if connue', '            hote["ss"][niv] = max(g["ss"][niv]) if connue'),
    ("ame", "le débord de rive oublié", '        return (donnee(i, "ss") or 0.0) / 2.0 + (donnee(i, "c") or 0.0)', '        return (donnee(i, "ss") or 0.0) / 2.0'),
    ("ame", "type (a) partout (le bout de la barre ignoré)", '        type_ = "c" if 2.0 + 6.0 * (min(ss, hw) + c) / hw < AM.KF_C_MAX else "a"', '        type_ = "a"'),
    ("ame", "type (c) partout", '        type_ = "c" if 2.0 + 6.0 * (min(ss, hw) + c) / hw < AM.KF_C_MAX else "a"', '        type_ = "c"'),
    ("ame", "force transmise (type b) jamais vérifiée", "    if len(out) == 2:\n        manque", "    if len(out) == 99:\n        manque"),
    ("ame", "force transmise : la plus GRANDE des deux", '                essais.append(("b", min(o["haut"], o["bas"])))', '                essais.append(("b", max(o["haut"], o["bas"])))'),
    ("ame", "moment de l'interaction lu d'un seul côté", '"M": max(abs(e["M_g"]), abs(e["M_d"])), "M_s": Ms,', '"M": 0.0, "M_s": Ms,'),
    ("ame", "raidisseur décidé sur η₂ seul (l'interaction oubliée)", '        elif pire["inter"] is not None and pire["inter"] > 1.0:', '        elif False:'),
    ("ame", "raidisseur jamais décidé sur η₂", '        if pire["eta2"] > 1.0:', '        if pire["eta2"] > 1e9:'),
    ("ame", "âme élancée : pas de raidisseur d'appui", '        certain = el is not None and el["elancee"] and L["genre"] == "appui"', "        certain = False"),
    ("ame", "lot C, point 15 — appui par l'âme d'une âme élancée : aucun raidisseur d'appui (NON CONSERVATIF)",
     'endroits = lieux(b, cas, sols, ame_elancee=el is not None and el["elancee"] and a is not None)', "endroits = lieux(b, cas, sols)"),
    ("ame", "lot C, point 15 — appui par l'âme : le raidisseur sous la plus petite réaction", "        F, C_F = max(Rs, key=lambda t: t[0])",
     "        F, C_F = min(Rs, key=lambda t: t[0])"),
    ("ame", "raidisseur dimensionné pour la force d'une combinaison quelconque", '            pire["F"] = max(pire["F"], o["Fmax"])', '            pire["F"] = o["Fmax"]'),
    ("ame", "deux forces rapprochées rendues « ok » sans le dire", "    if proches:\n        J = RE.Journal(METHODE, NORME)", "    if False:\n        J = RE.Journal(METHODE, NORME)"),
    ("ame", "forces rapprochées : cherchées sur l_y entier (deux fois trop loin)", "        if any(ecart < (portee(r1[k]) + portee(r2[k])) / 2.0 for k", "        if any(ecart < (portee(r1[k]) + portee(r2[k])) for k"),
    ("ame", "un encastrement passé sous silence", '    if genres - {"appui"}:', "    if False:"),
    ("ame", "voilement par cisaillement jamais cherché", '    if el is not None and el["elancee"]:\n        for C in combs:', '    if False:\n        for C in combs:'),
    ("ame", "un raidisseur impossible rendu conforme", '                trop_etroit = bool(etude["raison"]) and not manque ', "                trop_etroit = False "),
    ("ame", "un tube chargé rendu sans avertissement", '        if charges and s["forme"] in ("tube_carre", "tube_rect", "tube_rond"):', "        if False:"),
    ("ame", "réaction de soulèvement non signalée", '            if f["soulevement"] > 0:', "            if False:"),
    ("ame", "le bloc d'une barre aux efforts imposés : sans le cisaillement", '    AM.cisaillement(J, s, fy, P, abs(e["Vz"]) * 1e3, abs(e["My"]) * 1e6, e["N"] * 1e3, fy_ame=fyw)\n', ""),
    ("entree", "appuis par défaut « par l'âme » (rien n'est vérifié sans le dire)", '"appuis": {"mode": "poses", "ss": None, "c": None, "detail": None},',
     '"appuis": {"mode": "ame", "ss": None, "c": None, "detail": None},'),
    ("entree", "longueur d'appui d'une charge ponctuelle perdue à la relecture", '                ch["ss"] = _longueur_appui(saisie.get("ss"), f"cas « {nom} », longueur d\'appui s_s")', '                ch["ss"] = 0.0'),
    ("entree", "longueur d'appui sans borne", "    if not 0.0 <= x <= SS_MAX:", "    if False:"),
    ("entree", "mode d'appui non contrôlé", '    if ap is None or ap["mode"] not in MODES_APPUI:', "    if ap is None:"),
    # (reprise du lot B) les données manquantes, le maintien, la classe de η₁, N_st par panneau, les cotes d'un PRS, le contact d'une charge
    ("entree", "reprise — s_s non saisie prise pour 0", "    if v is None:\n        return None", "    if v is None:\n        return 0.0"),
    ("ame", "2e reprise — la borne qui ne passe pas : décidé quand même (sur s_s = 0)", "        a_saisir = bool(donnees or bout) and echec and not certain",
     "        a_saisir = False"),
    ("ame", "2e reprise — la borne qui passe : décision impossible quand même", "        a_saisir = bool(donnees or bout) and echec and not certain",
     "        a_saisir = bool(donnees or bout) and not certain"),
    ("ame", "reprise — c d'un appui de rive non exigé", '(["c"] if c_i is None and rive(a["x"]) else [])', "[]"),
    ("ame", "reprise — c d'une force voisine d'un appui de rive non saisi : ignoré", '        rive = list(L.get("manque_bout") or []) if type_ == "c" else []',
     "        rive = []"),
    ("ame", "2e reprise — une décision impossible perd son taux (la ligne n'a plus le taux à la borne)", '"taux": gouv["taux"], "statut": statut})',
     '"taux": None if manque else gouv["taux"], "statut": statut})'),
    # (intégration du lot B, point 29 d) le raidisseur sous une charge, une semelle libre : la semelle nommée
    ("ame", "intégration, point 29 d — la semelle libre mal nommée (haute pour basse)", "f\"Semelle {'haute' if libres[0] == 'haut' else 'basse'}\"",
     "f\"Semelle {'basse' if libres[0] == 'haut' else 'haute'}\""),
    ("ame", "reprise — maintien de la semelle chargée supposé", '        maintiens = [k for k in ("haut", "bas") if k in requis and not L["tenue"][k]]', "        maintiens = []"),
    ("ame", "reprise — semelle tendue : la semelle comprimée non exigée", "        requis = set(charges_) | {AUTRE[k] for k in charges_ if etats_[k] - {AM.COMPRIMEE}}",
     "        requis = set(charges_)"),
    ("ame", "reprise — un appui ne tient pas les semelles", "        if any(abs(a[\"x\"] - x) < MEME_ABSCISSE for a in appuis):\n            return True", "        if False:\n            return True"),
    ("ame", "reprise — un maintien de l'autre semelle compté", 'm.get("genre", "complet") in ("complet", cote)', 'm.get("genre", "complet") in ("complet", "haut", "bas")'),
    ("ame", "reprise — raidisseur sous une charge : une seule semelle tenue suffit", '        if manque is None and raison and L["genre"] == "charge" and not all(L["tenue"].values()):',
     "        if False:"),
    ("ame", "reprise — η₁ plastique : la classe sous N = 1 N", '"NEd": N * 1e3, "MyEd": M * 1e6, "parametres": b["parametres"],\n',
     '"NEd": 1.0, "MyEd": 0.0, "parametres": b["parametres"],\n'),
    ("ame", "reprise — N_st : V au droit du raidisseur", '            cis = {"V": cisaillement_panneaux(L["x"], bords, combs, el["hw"] / 1e3) * 1e3, "fyw": fy, "lambda_w": lw}',
     '            cis = {"V": etude["pire"]["V"] * 1e3, "fyw": fy, "lambda_w": lw}'),
    ("ame", "reprise — N_st : V au bord même du panneau (0,5·h_w oublié)", "        d = min(0.5 * hw, (v - u) / 2.0)", "        d = 0.0"),
    ("ame", "reprise — N_st : le bord le MOINS sollicité", "            e = C.lire(u + d if Vu >= Vv else v - d)", "            e = C.lire(u + d if Vu < Vv else v - d)"),
    ("ec3_ame", "reprise — PRS : z_g ignoré (h/2)", '    zg = s.get("z_g") or s["h"] / 2.0', '    zg = s["h"] / 2.0'),
    ("ec3_ame", "reprise — semelle basse prise pour la haute", "    sm = semelles_de(s)[cote]\n    zg = ", '    sm = semelles_de(s)["haut"]\n    zg = '),
    ("entree", "reprise — contact d'une charge ponctuelle perdu", '                    ch["contact"] = saisie["contact"]', "                    pass"),
    ("ame", "reprise — contact de plusieurs charges : « non » ignoré", '"non" if "non" in k_ else "inconnu"', '"inconnu"'),
    ("ame", "reprise — état de la semelle lu à la jonction", '        etat = AM.etat_semelle([AM.sigma_x(o["N"] * 1e3, o["M_s"] * 1e6, A, I, nv["z_moyen"])])',
     '        etat = AM.etat_semelle([AM.sigma_x(o["N"] * 1e3, o["M_s"] * 1e6, A, I, nv["z_jonction"])])'),
    ("ame", "reprise — τ de la jonction : la plus petite des deux valeurs", '"V": max(abs(e["V_g"]), abs(e["V_d"]))})', '"V": min(abs(e["V_g"]), abs(e["V_d"]))})'),
    ("ec3_ame", "reprise — seuil local sans le plafond F_Rd", "    return max(0.0, min(FRd, (lim - k * e1) * FRd))", "    return max(0.0, (lim - k * e1) * FRd)"),
    ("ame", "reprise — vue « sans » : σ_z avec s_s = 0", '"sigma_z": AM.sigma_z(pire["F"] * 1e3, r0["ss"], r0["ame"]["tf"], s["tw"])["sigma_z"]',
     '"sigma_z": AM.sigma_z(pire["F"] * 1e3, 0.0, r0["ame"]["tf"], s["tw"])["sigma_z"]'),
    ("raidisseurs", "reprise — raidisseur d'un PRS : h_w = h − 2·t_f", "    hw = AM.hw_de(s)", '    hw = s["h"] - 2.0 * s["tf"]'),
    ("ec3_ame", "reprise — h_w d'un PRS pour le raidisseur (relié par charger)", "    return SE.hauteur_ame(s)\n",
     '    return SE.hauteur_ame(s) - (1.0 if s.get("prs") else 0.0)\n'),
    ("moteur", "reprise — un laminé que V_pl,Rd fait échouer : rien n'est dit (ou sans le remède)", '            if v["id"] == "E-Vz" and v["statut"] == RE.NOK:',
     "            if False:"),
    # (2e reprise du lot B) la borne 0 de s_s et c, sa preuve ; N_st sur deux panneaux ; le groupe ; le maintien ; § 8.3 point 8
    ("ec3_ame", "2e reprise — borne : la règle des branches gardée au type (c)", "    if borne_m2:\n        B = sans", "    if False:\n        B = sans"),
    ("ec3_ame", "2e reprise — borne : m₂ = 0 même quand elle ne peut redevenir cohérente", "                and math.sqrt(sans[\"ly\"] * tw * fyw / (0.9 * KF_C_MAX * E * tw ** 3 / hw)) <= SEUIL_M2)",
     "                and True)"),
    ("ec3_ame", "2e reprise — borne : m₂ = 0 au type (b) aussi", '    borne_m2 = (borne and type_ == "c" and B is not sans', "    borne_m2 = (borne and B is not sans"),
    ("ame", "2e reprise — c manquant : « à saisir » même quand ce qui ne passe pas n'en dépend pas",
     '        sur = pire["crit"] is not None and pire["crit"] > 1.0 and "s_s" not in donnees', "        sur = False"),
    ("ame", "2e reprise — la résistance jamais prise à la borne", "borne=bool(manque or rive)), ame=a,", "borne=False), ame=a,"),
    ("ame", "2e reprise — c exigé quand la force est du type (a) à c = 0", 'if not (m == "c" and type_ == "a")]', "]"),
    ("ame", "2e reprise — s_s d'un groupe : la plus courte des SAISIES", '            connue = None not in g["ss"][niv]\n            hote["ss"][niv] = min(g["ss"][niv]) if connue else 0.0',
     '            connue = any(v is not None for v in g["ss"][niv])\n            hote["ss"][niv] = min(v for v in g["ss"][niv] if v is not None) if connue else 0.0'),
    ("ame", "2e reprise — le critère 6.2.1(5) au seul s_s = 0", '        essais = [r["ss"]] + ([r["ame"]["hw"]] if "s_s" in (r.get("manque") or []) else [])', '        essais = [r["ss"]]'),
    ("ame", "2e reprise — critère au pire : le plus PETIT des bouts", '        if pire is None or crit > pire["crit"] + EPS:', '        if pire is None or crit < pire["crit"] - EPS:'),
    ("ame", "2e reprise — forces rapprochées : l_y à la borne (la plus courte)", '    if not (r.get("manque") or r.get("rive")):\n        return r["ly"]', "    if True:\n        return r[\"ly\"]"),
    ("ec3_ame", "2e reprise — l_y la plus grande : sans m₂", "2.0 * tf * (1.0 + math.sqrt(a[\"bf\"] / a[\"tw\"] + 0.02 * (hw / tf) ** 2))", "2.0 * tf * (1.0 + math.sqrt(a[\"bf\"] / a[\"tw\"]))"),
    ("ame", "2e reprise — la borne non dite", '"borne": f"{b_} : borne prudente" if b_ else None, "bloque"', '"borne": None, "bloque"'),
    ("ame", "2e reprise — N_st sur un seul panneau", "    for u, v in ((gauche, x0), (x0, droite)):", "    for u, v in ((gauche, x0),):"),
    ("ame", "2e reprise — N_st : un panneau plus court que h_w non borné", "        d = min(0.5 * hw, (v - u) / 2.0)", "        d = 0.5 * hw"),
    ("ame", "2e reprise — maintien : « les deux » n'exige pas l'autre semelle", "etats_[k] - {AM.COMPRIMEE}}", "etats_[k] & {AM.TENDUE}}"),
    ("ame", "2e reprise — maintien compté à 0,5 m près", 'any(abs(m["x"] - x) < MEME_ABSCISSE and m.get', 'any(abs(m["x"] - x) < 0.5 and m.get'),
    ("moteur", "2e reprise — § 8.3 point 8 : rien quand le cisaillement gouverne sans échouer",
     '            elif v["id"] == "E-Vz" and out["ame"]["lieux"] and v["taux"] is not None and v["taux"] >= max(autres, default=0.0):', "            elif False:"),
    # (3e reprise du lot B) § 8.3 point 8 « gouverne » ; c demandé pour rien ; la borne dite dans les hypothèses ; le message de la force transmise
    ("moteur", "3e reprise — § 8.3 point 8 : E-Vz comparé aux seules forces concentrées (la barre peu chargée reçoit le conseil)",
     'for x in r_["verifications"] if x["id"] != "E-Vz" and x["taux"] is not None]', 'for x in r_["verifications"] if x["id"].startswith("A-L") and x["taux"] is not None]'),
    ("moteur", "3e reprise — § 8.3 point 8 : le remède donné quand le cisaillement passe", "                v[\"messages\"].append(AME.AM.MOT_CISAILLEMENT_GOUVERNE)",
     "                v[\"messages\"].append(AME.AM.MOT_CISAILLEMENT)"),
    ("ame", "3e reprise — le critère 6.2.1(5) seul en cause : c demandé quand même", '            donnees, bout = [m for m in donnees if m == "s_s"], []', "            pass"),
    ("ame", "3e reprise — critère seul : le filtre appliqué même quand η₂ ne passe pas",
     '        if echec and pire["eta2"] <= 1.0 and not (pire["inter"] is not None and pire["inter"] > 1.0):', "        if echec:"),
    ("ame", "3e reprise — la borne non dite dans les hypothèses du bloc", "        hyp.append(BORNE_HYPOTHESE) ", "        pass "),
    ("ame", "3e reprise — la force transmise ne dit pas sa borne", 'messages=_mots_force(res["b"], _saisie(L, "b"))', 'messages=AM.mots_resistance(res["b"])'),
    # (finition du lot B) les clauses de la 3e reprise, figées ; le message du maintien ; la fenêtre au bout h_w ; « gouverne » sans lieu
    ("ame", "finition — critère seul : le filtre ignore l'interaction (un raidisseur décidé sans c)",
     '        if echec and pire["eta2"] <= 1.0 and not (pire["inter"] is not None and pire["inter"] > 1.0):', '        if echec and pire["eta2"] <= 1.0:'),
    ("ame", "finition — critère seul : les données de l'appui de rive redemandées", '            donnees, bout = [m for m in donnees if m == "s_s"], []',
     '            donnees = [m for m in donnees if m == "s_s"]'),
    ("moteur", "finition — § 8.3 point 8 : E-Vz comparé à l'Eurocode 3 seul (l'âme oubliée)", 'for r_ in (ec, out["ame"]) for x in', 'for r_ in (ec,) for x in'),
    ("moteur", "finition — § 8.3 point 8 : « gouverne » dit sans force concentrée (aucun lieu)", ' and out["ame"]["lieux"] and v["taux"]', ' and v["taux"]'),
    ("ame", "finition — le critère 6.2.1(5) au bout h_w aussi quand s_s est saisie",
     '        essais = [r["ss"]] + ([r["ame"]["hw"]] if "s_s" in (r.get("manque") or []) else [])', '        essais = [r["ss"], r["ame"]["hw"]]'),
    ("ame", "finition — la fenêtre imprime s_s = h_w sans le dire", '[AM.MOTS_ETAT[x["etat"]]] + ([MOT_PIRE_HW] if x["pire_hw"] else [])', '[AM.MOTS_ETAT[x["etat"]]]'),
    ("ame", "finition — maintien seul : le taux pris sur η₂ seul (l'interaction oubliée)", '        tenu = max(pire["eta2"], pire["inter"] or 0.0, pire["crit"] or 0.0)',
     '        tenu = pire["eta2"]'),
    ("ame", "finition — maintien seul : le remède oublié", '{taux_} ; {remede} (À VÉRIFIER)."', '{taux_} (À VÉRIFIER)."'),
    ("ame", "finition — maintien seul : l'ancienne phrase « maintien … à préciser », sans taux", 'impossible(donnees, maintiens, bout) if a_saisir else sans_maintien(maintiens, tenu))',
     'impossible(donnees if a_saisir else [], maintiens, bout if a_saisir else []))'),
    ("ame", "finition — la borne dite en citant le brief (document interne)", 'BORNE_HYPOTHESE = "s_s ou c non saisi : calculé à sa borne 0 (À VÉRIFIER)."',
     'BORNE_HYPOTHESE = "s_s ou c non saisi : calculé à sa borne 0 (À VÉRIFIER, brief § 8.3, point 5)."'),
    # (2e finition du lot B) les formes du message du maintien, toutes figées
    ("ame", "2e finition — maintien : « si elles sont tenues » jamais dit (le pluriel)", '    tenue = "si elles sont tenues" if len(cotes) == 2 else "si elle est tenue"',
     '    tenue = "si elle est tenue"'),
    ("ame", "2e finition — maintien : « semelles haute et basse » réduit à la première", '    quoi = "semelles haute et basse" if len(cotes) == 2 else',
     '    quoi = "semelle haute" if len(cotes) == 2 else'),
    ("ame", "2e finition — une donnée ET le maintien manquent : le maintien tu",
     "    parts = ([f\"{' ; '.join(quoi)} à saisir\"] if quoi else []) + ([_sans_maintien(maintiens)] if maintiens else [])",
     "    parts = ([f\"{' ; '.join(quoi)} à saisir\"] if quoi else [])"),
    ("ame", "2e finition — au-delà de 100 % : le raidisseur et le maintien complet non dits",
     'remede = "ajouter un maintien complet et un raidisseur" if taux > 1.0 else "ajouter un maintien"', 'remede = "ajouter un maintien"'),
    ("ame", "2e finition — la ligne du lieu redit le taux de sa pastille", '    taux_ = "" if ligne else f" — ', '    taux_ = f" — '),
    ("ame", "2e finition — la ligne sans taux jamais donnée", "        ligne = sans_maintien(maintiens, tenu, ligne=True) if manque and not a_saisir else None",
     "        ligne = None"),
    ("ame", "2e finition — la ligne donnée aussi quand une donnée manque", "ligne=True) if manque and not a_saisir else None",
     "ligne=True) if maintiens and not certain else None"),
    ("ame", "2e finition — la fenêtre -F redit le taux d'une autre vérification", '[etude["ligne"] or manque]', "[manque]"),
    ("moteur", "le statut de la barre ignore l'âme", '    statuts = [r["statut"] for r in out["methodes"].values()] + [out["ame"]["statut"]]', '    statuts = [r["statut"] for r in out["methodes"].values()]'),
    # (lot C, C4) -FM écrit comme la norme : la somme et 1,4 — l'ancienne écriture (taux / 1) revient, ou le taux est écrit deux fois
    ("ame", "lot C — -FM : la valeur redevient le taux, la limite 1", "                            o[\"eta2\"] + k72 * o[\"eta1\"], lim72, \"\", cles,",
     "                            o[\"inter\"], 1.0, \"\", cles,"),
    ("ame", "lot C — -FM : la formule du taux écrite une seconde fois", 'cles.append(J.f(f"taux_{L[\'base\']}-FM", "η"', 'cles.append(J.f("i72", "η"'),
]


# (lot B) les exemples du commentaire JRC, EUR 22898 EN (brief § 11.4) : BLOQUÉS — le site eurocodes.jrc.ec.europa.eu (et ses
# copies) n'était pas joignable depuis la session du lot B. Aucune donnée n'est inventée : à fournir (données d'entrée et
# résultats publiés de chaque exemple), puis à écrire ici comme contrôles, à la tolérance du brief (règle 7 : 1 %).
JRC_BLOQUES = ("cisaillement (§ 5)", "charges transversales (§ 6)", "raidisseurs (§ 9)")


def main():
    for quoi in JRC_BLOQUES:
        print(f"  BLOQUÉ  JRC EUR 22898, exemples de {quoi} : données à fournir")
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

    print("== 6 ter. (2e reprise du lot B) la preuve de la borne, balayage large (une fois)")
    def prs_(b1, t1, hw, tw, b2, t2):
        return prs_section(b1, t1, hw, tw, b2, t2)
    secs = [(n_, SE.du_catalogue(n_)) for n_ in ("IPE 300", "IPE 750x147", "IPE 750x134", "HEAA 1000", "HEB 300", "HEA 200", "HEM 300", "HL 1100 A")]
    secs += [("PRS 300×20/1000×6/250×20", prs_(300.0, 20.0, 1000.0, 6.0, 250.0, 20.0)), ("PRS-C", prs_(300.0, 30.0, 1420.0, 25.0, 500.0, 50.0))]
    n_b, viol_b, naif_b = balayage(AME, PA.jeu(), secs, ((235.0, 235.0), (355.0, 355.0), (225.0, 235.0)),
                                   (0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1e4),
                                   (0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 400.0, 1000.0, 5000.0))
    chk(f"F_Rd à la borne ≤ F_Rd réel sur {n_b} cas (8 laminés et 2 PRS, trois f_y, appuis et charges près d'un bout, types a, b, c, contact "
        f"oui, non, inconnu) ; la règle des branches seule, à la borne, dépasse le réel dans {naif_b} cas",
        n_b > 50000 and not viol_b and naif_b > 0, (viol_b[:3], naif_b))

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

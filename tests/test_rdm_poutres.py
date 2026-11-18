# -*- coding: utf-8 -*-
"""Le moteur du FORMULAIRE DE POUTRES (rdm/poutres.py, docs/site/BIBLIOTHEQUE_AUTRES.md § 3).

    /usr/bin/python3.12 tests/test_rdm_poutres.py

Ce que la suite prouve, par des calculs INDÉPENDANTS du moteur :
  1. chaque cas d'école contre sa formule de formulaire, RÉÉCRITE ICI (copie nommée : si le moteur change un
     coefficient, la copie ne le suit pas) ;
  2. les valeurs publiées des poutres continues à travées égales (tables des formulaires : fractions exactes) ;
  3. un cas composé (quatre travées inégales, charge partielle, ponctuelle, couple, triangle) contre une INTÉGRATION
     NUMÉRIQUE de E·I·f″ = −M écrite dans ce fichier (le moteur, lui, n'intègre rien : il évalue des formules fermées) ;
  4. l'équilibre, la superposition, les symétries, les refus ;
  5. des mutations du moteur, qui rougissent ;
  6. les DÉBORDS (porte-à-faux aux rives, « appuis » et « continue ») : formulaire réécrit ici contre les paramètres
     initiaux, équilibre écrit depuis les données, symétrie, débord nul = résultats d'avant les débords (valeurs
     figées le 17/11/2026 à 1e-12), refus, et leurs mutations.
"""
import math
import os
import sys

RACINE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RACINE)

from rdm import poutres as P  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def pres(a, b, rel=1e-9, ech=1.0):
    return abs(a - b) <= rel * max(abs(a), abs(b), ech * 1e-6)


def R(sol, nom):
    return next(a for a in sol.appuis if a["nom"] == nom)["R"]


def Mapp(sol, nom):
    return next(a for a in sol.appuis if a["nom"] == nom)["M"]


Q = lambda v, a, b: {"type": "q", "v": v, "a": a, "b": b}                                      # noqa: E731
T = lambda v, a, b, s="croissante": {"type": "tri", "v": v, "a": a, "b": b, "sens": s}         # noqa: E731
F = lambda v, a: {"type": "P", "v": v, "a": a}                                                  # noqa: E731
Cp = lambda v, a: {"type": "C", "v": v, "a": a}                                                 # noqa: E731

# ================================================================================================ 1 · cas d'école
print("=== 1 · Cas d'école : la formule du formulaire, réécrite ici")
q, L, EI = 10.0, 6.0, 21000.0
s = P.resoudre("appuis", [L], [Q(q, 0, L)], EI)
chk("deux appuis, q totale : R = q·L/2 = 30, M_max = q·L²/8 = 45 à L/2, f = 5·q·L⁴/(384·E·I)",
    pres(R(s, "A"), q * L / 2) and pres(R(s, "B"), 30.0) and pres(s.extremes["M_max"][0], 45.0) and pres(s.extremes["M_max"][1], 3.0)
    and pres(s.extremes["EIf_max"][0], 5 * q * L ** 4 / 384) and pres(s.evaluer(3.0)["f"], 5 * q * L ** 4 / 384 / EI), s.extremes)
Pp, a = 20.0, 2.0
b = L - a
s = P.resoudre("appuis", [L], [F(Pp, a)])
chk("deux appuis, P en a : R_A = P·b/L, R_B = P·a/L, M_max = P·a·b/L sous la charge, E·I·f(a) = P·a²·b²/(3·L)",
    pres(R(s, "A"), Pp * b / L) and pres(R(s, "B"), Pp * a / L) and pres(s.extremes["M_max"][0], Pp * a * b / L)
    and pres(s.extremes["M_max"][1], a) and pres(s.evaluer(a)["EIf"], Pp * a * a * b * b / (3 * L)))
chk("deux appuis, P en a < L/2 : flèche maximale P·a·(L² − a²)^1,5/(9·√3·L) à x = L − √((L² − a²)/3)",
    pres(s.extremes["EIf_max"][0], Pp * a * (L * L - a * a) ** 1.5 / (9 * math.sqrt(3) * L))
    and pres(s.extremes["EIf_max"][1], L - math.sqrt((L * L - a * a) / 3)), s.extremes["EIf_max"])
s = P.resoudre("appuis", [L], [F(Pp, L / 2)])
chk("deux appuis, P au milieu : M = P·L/4 = 30, E·I·f = P·L³/48 = 90", pres(s.extremes["M_max"][0], 30.0) and pres(s.extremes["EIf_max"][0], 90.0))
s = P.resoudre("appuis", [L], [T(12.0, 0, L)])
chk("deux appuis, triangle 0 → q : R_A = q·L/6, R_B = q·L/3, M_max = q·L²/(9·√3) à L/√3, E·I·f_max = 0,006522·q·L⁴ à 0,5193·L",
    pres(R(s, "A"), 12.0) and pres(R(s, "B"), 24.0) and pres(s.extremes["M_max"][0], 12 * 36 / (9 * math.sqrt(3)))
    and pres(s.extremes["M_max"][1], L / math.sqrt(3)) and abs(s.extremes["EIf_max"][0] / (12 * L ** 4) - 0.006522) < 1e-6
    and abs(s.extremes["EIf_max"][1] / L - 0.5193) < 1e-4, s.extremes)
Cc = 15.0
s = P.resoudre("appuis", [L], [Cp(Cc, a)])
e = s.evaluer(a)
chk("deux appuis, couple C horaire en a : R_A = −C/L, R_B = +C/L, M saute de −C·a/L à +C·b/L",
    pres(R(s, "A"), -Cc / L) and pres(R(s, "B"), Cc / L) and pres(e["M_g"], -Cc * a / L) and pres(e["M_d"], Cc * b / L), e)
Lc = 3.0
s = P.resoudre("console", [Lc], [Q(5.0, 0, Lc)])
chk("console, q totale : R = q·L = 15, M_A = −q·L²/2 = −22,5, E·I·f(L) = q·L⁴/8 = 50,625",
    pres(R(s, "A"), 15.0) and pres(Mapp(s, "A"), -22.5) and pres(s.extremes["EIf_max"][0], 50.625) and pres(s.extremes["EIf_max"][1], Lc))
s = P.resoudre("console", [Lc], [F(8.0, 2.0)])
chk("console, P en a : M_A = −P·a = −16, E·I·f(a) = P·a³/3, E·I·f(L) = P·a²·(3·L − a)/6",
    pres(Mapp(s, "A"), -16.0) and pres(s.evaluer(2.0)["EIf"], 8 * 8 / 3) and pres(s.extremes["EIf_max"][0], 8 * 4 * (9 - 2) / 6))
s = P.resoudre("console", [Lc], [T(9.0, 0, Lc, "decroissante")])
chk("console, triangle q → 0 : R = q·L/2, M_A = −q·L²/6, E·I·f(L) = q·L⁴/30",
    pres(R(s, "A"), 13.5) and pres(Mapp(s, "A"), -13.5) and pres(s.extremes["EIf_max"][0], 9 * 81 / 30))
s = P.resoudre("console", [Lc], [Cp(6.0, 2.0)])
chk("console, couple C en a : M = −C de A à a, E·I·f(L) = C·a·(2·L − a)/2", pres(Mapp(s, "A"), -6.0) and pres(s.evaluer(1.0)["M_d"], -6.0)
    and pres(s.evaluer(2.5)["M_d"], 0.0, ech=6) and pres(s.evaluer(Lc)["EIf"], 6 * 2 * (6 - 2) / 2))
q, L = 12.0, 5.0
s = P.resoudre("enc_app", [L], [Q(q, 0, L)])
chk("encastrée–appuyée, q : M_A = −q·L²/8, R_A = 5·q·L/8, R_B = 3·q·L/8, M⁺ = 9·q·L²/128 à 5·L/8, f = q·L⁴/(184,6·E·I) à 0,5785·L",
    pres(Mapp(s, "A"), -q * L * L / 8) and pres(R(s, "A"), 5 * q * L / 8) and pres(R(s, "B"), 3 * q * L / 8)
    and pres(s.extremes["M_max"][0], 9 * q * L * L / 128) and pres(s.extremes["M_max"][1], 5 * L / 8)
    and abs(q * L ** 4 / s.extremes["EIf_max"][0] - 184.6) < 0.05 and abs(s.extremes["EIf_max"][1] / L - 0.5785) < 1e-4, s.extremes)
s = P.resoudre("enc_app", [L], [F(30.0, L / 2)])
chk("encastrée–appuyée, P au milieu : M_A = −3·P·L/16, R_A = 11·P/16, R_B = 5·P/16, M sous P = 5·P·L/32, E·I·f_max = P·L³/(48·√5)",
    pres(Mapp(s, "A"), -3 * 30 * L / 16) and pres(R(s, "A"), 11 * 30 / 16) and pres(R(s, "B"), 5 * 30 / 16)
    and pres(s.extremes["M_max"][0], 5 * 30 * L / 32) and pres(s.extremes["EIf_max"][0], 30 * L ** 3 / (48 * math.sqrt(5))))
s = P.resoudre("bi_enc", [L], [Q(q, 0, L)])
chk("bi-encastrée, q : M_A = M_B = −q·L²/12, M au milieu = q·L²/24, E·I·f = q·L⁴/384",
    pres(Mapp(s, "A"), -25.0) and pres(Mapp(s, "B"), -25.0) and pres(s.extremes["M_max"][0], 12.5) and pres(s.extremes["EIf_max"][0], q * L ** 4 / 384))
a, b = 2.0, 3.0
s = P.resoudre("bi_enc", [L], [F(30.0, a)])
chk("bi-encastrée, P en a : M_A = −P·a·b²/L², M_B = −P·a²·b/L², M sous P = 2·P·a²·b²/L³, E·I·f(a) = P·a³·b³/(3·L³)",
    pres(Mapp(s, "A"), -30 * a * b * b / L ** 2) and pres(Mapp(s, "B"), -30 * a * a * b / L ** 2)
    and pres(s.extremes["M_max"][0], 2 * 30 * a * a * b * b / L ** 3) and pres(s.evaluer(a)["EIf"], 30 * a ** 3 * b ** 3 / (3 * L ** 3)))
s = P.resoudre("bi_enc", [L], [F(30.0, L / 2)])
chk("bi-encastrée, P au milieu : |M| = P·L/8 aux encastrements et sous la charge, E·I·f = P·L³/192",
    pres(Mapp(s, "A"), -30 * L / 8) and pres(s.extremes["M_max"][0], 30 * L / 8) and pres(s.extremes["EIf_max"][0], 30 * L ** 3 / 192))

# ================================================================================================ 2 · poutres continues
print("\n=== 2 · Poutres continues : valeurs des tables (fractions exactes)")
q, L = 10.0, 4.0
qL, qL2 = q * L, q * L * L
s = P.resoudre("continue", [L, L], [Q(q, 0, 2 * L)])
chk("2 travées égales, q partout : M_B = −q·L²/8, R_A = 3·q·L/8, R_B = 10·q·L/8, M en travée = 9·q·L²/128 à 3·L/8",
    pres(Mapp(s, "B"), -qL2 / 8) and pres(R(s, "A"), 3 * qL / 8) and pres(R(s, "B"), 10 * qL / 8) and pres(s.extremes["travees"][0]["M_max"][0], 9 * qL2 / 128)
    and pres(s.extremes["travees"][0]["M_max"][1], 3 * L / 8))
s = P.resoudre("continue", [L, L], [Q(q, 0, L)])
chk("2 travées égales, q sur la première seule : M_B = −q·L²/16, R_A = 7·q·L/16, R_C = −q·L/16",
    pres(Mapp(s, "B"), -qL2 / 16) and pres(R(s, "A"), 7 * qL / 16) and pres(R(s, "C"), -qL / 16))
s = P.resoudre("continue", [L, L], [F(20.0, L / 2), F(20.0, 1.5 * L)])
chk("2 travées égales, P au milieu de chacune : M_B = −3·P·L/16, R_A = 5·P/16, R_B = 22·P/16",
    pres(Mapp(s, "B"), -3 * 20 * L / 16) and pres(R(s, "A"), 5 * 20 / 16) and pres(R(s, "B"), 22 * 20 / 16))
s = P.resoudre("continue", [L] * 3, [Q(q, 0, 3 * L)])
chk("3 travées égales, q partout : M_B = M_C = −q·L²/10, R_A = 0,4·q·L, R_B = 1,1·q·L, M₁ = 0,08·q·L², M₂ = 0,025·q·L²",
    pres(Mapp(s, "B"), -qL2 / 10) and pres(Mapp(s, "C"), -qL2 / 10) and pres(R(s, "A"), 0.4 * qL) and pres(R(s, "B"), 1.1 * qL)
    and pres(s.extremes["travees"][0]["M_max"][0], 0.08 * qL2) and pres(s.extremes["travees"][1]["M_max"][0], 0.025 * qL2))
s = P.resoudre("continue", [L] * 3, [Q(q, 0, L), Q(q, 2 * L, 3 * L)])
chk("3 travées égales, q sur les travées de rive : M_B = −q·L²/20, R_A = 0,45·q·L", pres(Mapp(s, "B"), -qL2 / 20) and pres(R(s, "A"), 0.45 * qL))
s = P.resoudre("continue", [L] * 4, [Q(q, 0, 4 * L)])
chk("4 travées égales, q partout : M_B = −3·q·L²/28, M_C = −2·q·L²/28, R_A = 11·q·L/28, R_B = 32·q·L/28, R_C = 26·q·L/28",
    pres(Mapp(s, "B"), -3 * qL2 / 28) and pres(Mapp(s, "C"), -2 * qL2 / 28) and pres(R(s, "A"), 11 * qL / 28) and pres(R(s, "B"), 32 * qL / 28)
    and pres(R(s, "C"), 26 * qL / 28))
s = P.resoudre("continue", [L] * 5, [Q(q, 0, 5 * L)])
chk("5 travées égales, q partout : M_B = −4·q·L²/38, M_C = −3·q·L²/38, R_A = 15·q·L/38, R_B = 43·q·L/38, R_C = 37·q·L/38",
    pres(Mapp(s, "B"), -4 * qL2 / 38) and pres(Mapp(s, "C"), -3 * qL2 / 38) and pres(R(s, "A"), 15 * qL / 38) and pres(R(s, "B"), 43 * qL / 38)
    and pres(R(s, "C"), 37 * qL / 38))

# ================================================================================================ 3 · intégration numérique
print("\n=== 3 · Cas composé contre une intégration numérique écrite ici")
Ls = [4.0, 5.0, 3.5, 6.0]
ch = [Q(10.0, 1.0, 11.0), F(25.0, 6.5), Cp(15.0, 4.0), T(8.0, 9.0, 18.5, "decroissante"), T(6.0, 0.5, 3.0), Q(-3.0, 13.0, 16.0)]
s = P.resoudre("continue", Ls, ch, 1.0)


def integrer(sol, n=4000):
    """E·I·f par double intégration numérique (Simpson composé) de −M sur chaque travée, f(0) = 0 imposé, la rotation
    initiale tirée de f(L) = 0 — rien du moteur que M(x), lu par ``evaluer``."""
    pire = 0.0
    for t in sol.travees:
        h = t.L / n
        xs = [t.x0 + i * h for i in range(n + 1)]
        Mv = []
        for i, x in enumerate(xs):
            e_ = sol.evaluer(x)
            Mv.append(e_["M_d"] if i < n else e_["M_g"])
        # les sauts de M (couples) sont rares : la somme de Simpson les lisse à h près, ce qui borne l'écart admis
        th = [0.0]
        for i in range(n):
            th.append(th[-1] - (Mv[i] + Mv[i + 1]) / 2 * h)
        f = [0.0]
        for i in range(n):
            f.append(f[-1] + (th[i] + th[i + 1]) / 2 * h)
        w0 = -f[-1] / t.L
        for i in (n // 4, n // 2, 3 * n // 4):
            num = f[i] + w0 * (xs[i] - t.x0)
            pire = max(pire, abs(num - sol.evaluer(xs[i])["EIf"]))
    return pire


ech = max(abs(s.extremes["EIf_max"][0]), abs(s.extremes["EIf_min"][0]))
ecart = integrer(s)
chk(f"4 travées inégales, 6 charges : E·I·f des formules fermées = double intégration numérique de −M (écart {ecart / ech:.1e} de la flèche max)",
    ecart < 2e-3 * ech, (ecart, ech))
# le moment lui-même, recalculé à la main dans une section : somme des forces à gauche × bras
x = 7.3
Mg = (R(s, "A") * x + R(s, "B") * (x - 4.0) - 10.0 * (x - 1.0) ** 2 / 2 - 25.0 * (x - 6.5) + 15.0
      - (6.0 * 2.5 / 2) * (x - (0.5 + 2 * 2.5 / 3)))
chk("4 travées inégales : M(7,3) = somme des moments des forces à gauche, écrite à la main (réactions, q partielle, P, C, triangle)",
    pres(s.evaluer(x)["M_d"], Mg, 1e-9), (s.evaluer(x)["M_d"], Mg))
chk("4 travées inégales : chaque contrôle interne est tenu (équilibre, flèches nulles, rotations continues, formulaire = paramètres initiaux)",
    all(c["ok"] for c in s.controles) and len(s.controles) >= 2 + 5 * 4 + 3, [c for c in s.controles if not c["ok"]])
tot = sum(P.resultante(c)[0] for c in s.charges)
chk("4 travées inégales : Σ R = Σ charges (calculée ici depuis les données)", pres(sum(a["R"] for a in s.appuis), 10 * 10 + 25 + 8 * 9.5 / 2 + 6 * 2.5 / 2 - 9)
    and pres(tot, sum(a["R"] for a in s.appuis)))

# ================================================================================================ 4 · propriétés
print("\n=== 4 · Superposition, symétrie, refus")
c1, c2 = [Q(7.0, 0.5, 4.0), F(12.0, 3.0)], [T(9.0, 1.0, 5.0), Cp(-8.0, 2.0)]
for modele in ("appuis", "console", "enc_app", "bi_enc"):
    s1, s2, s12 = P.resoudre(modele, [5.0], c1), P.resoudre(modele, [5.0], c2), P.resoudre(modele, [5.0], c1 + c2)
    ok = all(pres(s12.evaluer(x)[k], s1.evaluer(x)[k] + s2.evaluer(x)[k], 1e-9, 100) for x in (0.7, 2.0, 3.3, 4.6) for k in ("V_d", "M_d", "EIf", "EIrot"))
    chk(f"superposition ({modele}) : V, M, E·I·θ, E·I·f de deux chargements s'additionnent", ok)
s = P.resoudre("continue", [4.0, 6.0, 4.0], [Q(5.0, 0, 14.0), F(10.0, 7.0)])
chk("symétrie : poutre et charges symétriques → réactions et moments symétriques, rotation nulle au milieu",
    pres(R(s, "A"), R(s, "D")) and pres(R(s, "B"), R(s, "C")) and pres(Mapp(s, "B"), Mapp(s, "C")) and abs(s.evaluer(7.0)["EIrot"]) < 1e-9 * 1000)
s = P.resoudre("appuis", [6.0], [Q(-10.0, 0, 6.0)])
chk("charge vers le haut : tout change de signe (M_min = −45, flèche vers le haut)", pres(s.extremes["M_min"][0], -45.0) and pres(s.extremes["EIf_min"][0], -168.75))
s = P.resoudre("appuis", [6.0], [F(10.0, 0.0), F(7.0, 6.0)])
chk("charges ponctuelles SUR les appuis : elles passent dans les réactions, aucun moment", pres(R(s, "A"), 10.0) and pres(R(s, "B"), 7.0)
    and s.extremes["M_max"][0] == 0.0 and s.extremes["M_min"][0] == 0.0)
e = P.resoudre("appuis", [6.0], [F(20.0, 2.0)]).evaluer(2.0)
chk("sous une charge ponctuelle, V saute de P (13,33 → −6,67)", pres(e["V_g"], 40 / 3) and pres(e["V_d"], 40 / 3 - 20))
# un appui à une somme de portées INEXACTE en binaire (3,37 + 2,18 = 5,550000000000001), lu à 5,55 (la barre arrondit ses abscisses à 10⁻⁹)
L1, L2, L3, qi = 3.37, 2.18, 3.45, 30.0
a11, a22, m1, m2 = 2 * (L1 + L2), 2 * (L2 + L3), -qi * (L1 ** 3 + L2 ** 3) / 4, -qi * (L2 ** 3 + L3 ** 3) / 4     # trois moments, à la main
MB, MC = (m1 * a22 - L2 * m2) / (a11 * a22 - L2 * L2), (a11 * m2 - L2 * m1) / (a11 * a22 - L2 * L2)
VCg, VCd = qi * L2 / 2 + (MC - MB) / L2 - qi * L2, qi * L3 / 2 - MC / L3


def appui_inexact():
    e_ = P.resoudre("continue", [L1, L2, L3], [Q(qi, 0, L1 + L2 + L3)]).evaluer(round(L1 + L2, 9))
    return L1 + L2 != 5.55 and pres(e_["V_g"], VCg) and pres(e_["V_d"], VCd) and pres(e_["M_g"], MC) and pres(e_["M_d"], MC)


chk("appui à 3,37 + 2,18 = 5,550000000000001 m, lu à 5,55 : V à gauche dans la travée BC (−33,54 kN), à droite dans la travée CD (+60,15 kN), "
    "M = M_C des deux côtés (trois moments, à la main)", appui_inexact(), P.resoudre("continue", [L1, L2, L3], [Q(qi, 0, L1 + L2 + L3)]).evaluer(5.55))
chk("sans E·I : les efforts sont rendus, la flèche reste E·I·f (f = None)", P.resoudre("appuis", [6.0], [Q(1.0, 0, 6.0)]).evaluer(3.0)["f"] is None)
co = P.resoudre("continue", [4.0, 4.0], [Q(10.0, 0, 8.0)]).courbes()
chk("diagrammes : par travée, des points de l'abscisse x0 à x0 + L, M nul aux appuis de rive, M_B aux deux côtés de l'appui",
    len(co) == 2 and co[0]["points"][0][0] == 0.0 and co[1]["points"][-1][0] == 8.0 and abs(co[0]["points"][0][2]) < 1e-9
    and pres(co[0]["points"][-1][2], -20.0) and pres(co[1]["points"][0][2], -20.0))
for nom, args in (("portée nulle", ("appuis", [0.0], [])), ("charge hors de la poutre", ("appuis", [5.0], [F(1.0, 6.0)])),
                  ("a ≥ b", ("appuis", [5.0], [Q(1.0, 3.0, 3.0)])), ("modèle inconnu", ("portique", [5.0], [])),
                  ("une seule travée continue", ("continue", [5.0], [])), ("six travées", ("continue", [1.0] * 6, [])),
                  ("neuf charges", ("appuis", [5.0], [F(1.0, 1.0)] * 9)), ("valeur non finie", ("appuis", [5.0], [F(float("nan"), 1.0)])),
                  ("E·I nul", ("appuis", [5.0], [], 0.0))):
    try:
        P.resoudre(*args)
        chk(f"refus : {nom}", False, "accepté")
    except P.ErreurDonnees as ex:
        chk(f"refus : {nom} ({ex})", True)
chk("poutre sans charge : tout est nul, aucun cas d'école", P.resoudre("bi_enc", [5.0], []).extremes["M_max"][0] == 0.0
    and P.cas_ecole(P.resoudre("bi_enc", [5.0], [])) is None)
cles = set()
for modele, Lx, chx in (("appuis", [6.0], [Q(1, 0, 6)]), ("appuis", [6.0], [F(1, 2)]), ("appuis", [6.0], [F(1, 3)]), ("appuis", [6.0], [T(1, 0, 6)]),
                        ("console", [3.0], [Q(1, 0, 3)]), ("console", [3.0], [F(1, 2)]), ("console", [3.0], [T(1, 0, 3, "decroissante")]),
                        ("enc_app", [5.0], [Q(1, 0, 5)]), ("enc_app", [5.0], [F(1, 2)]), ("enc_app", [5.0], [F(1, 2.5)]),
                        ("bi_enc", [5.0], [Q(1, 0, 5)]), ("bi_enc", [5.0], [F(1, 2)]), ("bi_enc", [5.0], [F(1, 2.5)]),
                        ("continue", [4.0, 4.0], [Q(1, 0, 8)]), ("continue", [4.0] * 3, [Q(1, 0, 12)])):
    ce = P.cas_ecole(P.resoudre(modele, Lx, chx))
    cles.add(ce["cle"] if ce else None)
chk("cas d'école : quinze cas reconnus, chacun comparé par le moteur à la résolution générale", len(cles) == 15 and None not in cles, cles)
chk("cas d'école : une charge négative est reconnue aussi (les « max » suivent le signe)",
    pres(P.cas_ecole(P.resoudre("appuis", [6.0], [Q(-10, 0, 6)]))["valeurs"]["M_max"], -45.0))
chk("cas d'école : deux charges → None (la feuille montre la superposition)", P.cas_ecole(P.resoudre("appuis", [6.0], [Q(1, 0, 6), F(1, 2)])) is None)

# ================================================================================================ 5 · mutations
print("\n=== 5 · Mutations (le moteur abîmé doit être refusé)")


def muter(nom, poser, defaire, essai):
    poser()
    try:
        try:
            essai()
            chk(f"mutation « {nom} » : refusée", False, "le calcul abîmé est passé")
        except (P.ErreurControle, AssertionError) as ex:
            chk(f"mutation « {nom} » : refusée ({str(ex)[:70]}…)", True)
    finally:
        defaire()


def attendu(cond):
    if not cond:
        raise AssertionError("valeur du formulaire non retrouvée")


orig_F1, orig_rf, orig_tri = P._F1, P.rotations_formulaire, P._resoudre_tridiagonal
muter("F₁ du formulaire faux (ξ⁵/5 → ξ⁵/4)", lambda: setattr(P, "_F1", lambda x, L: 2 * L * L * x ** 3 / 3 - 3 * L * x ** 4 / 4 + x ** 5 / 4),
      lambda: setattr(P, "_F1", orig_F1), lambda: P.resoudre("appuis", [6.0], [T(5.0, 1.0, 4.0)]))
muter("rotation du formulaire sous P fausse (L + b → L + a)",
      lambda: setattr(P, "rotations_formulaire", lambda m, L: (orig_rf(m, L)[0], orig_rf(m, L)[1], orig_rf(m, L)[3], orig_rf(m, L)[2])),
      lambda: setattr(P, "rotations_formulaire", orig_rf), lambda: P.resoudre("appuis", [6.0], [F(5.0, 1.0)]))
muter("trois moments : système mal résolu (solution × 1,01)", lambda: setattr(P, "_resoudre_tridiagonal", lambda Ls_, sec: [1.01 * x for x in orig_tri(Ls_, sec)]),
      lambda: setattr(P, "_resoudre_tridiagonal", orig_tri), lambda: P.resoudre("continue", [4.0, 5.0], [Q(10.0, 0, 9.0)]))
orig_Q = P._Q
muter("terme de flèche d'une charge ponctuelle faux (⟨x − a⟩³/6 → /5)",
      lambda: setattr(P, "_Q", lambda t, j, x, d=True: orig_Q(t, j, x, d) * (1.2 if j == 4 and any(g == "P" for g, _, _ in t) else 1.0)),
      lambda: setattr(P, "_Q", orig_Q), lambda: attendu(pres(P.resoudre("console", [3.0], [F(8.0, 2.0)]).extremes["EIf_max"][0], 8 * 4 * 7 / 6)))
orig_tol = P.TOL_APPUI
muter("appui comparé sans tolérance (5,55 ≠ 5,550000000000001 : ses deux côtés dans la même travée)", lambda: setattr(P, "TOL_APPUI", 0.0),
      lambda: setattr(P, "TOL_APPUI", orig_tol), lambda: attendu(appui_inexact()))
src = open(os.path.join(RACINE, "rdm", "poutres.py"), encoding="utf-8").read()
for nom, avant, apres, essai in (
        ("encastrée–appuyée : M_A = −3·E·I·ω_A/L devenu −2", "t0.fermer(-3.0 * t0.wA0 / t0.L, 0.0)", "t0.fermer(-2.0 * t0.wA0 / t0.L, 0.0)",
         lambda M: M.resoudre("enc_app", [5.0], [Q(12.0, 0, 5.0)])),
        ("bi-encastrée : coefficient 2 devenu 3", "t0.fermer(-2.0 * (2.0 * t0.wA0 - t0.wB0) / t0.L", "t0.fermer(-3.0 * (2.0 * t0.wA0 - t0.wB0) / t0.L",
         lambda M: M.resoudre("bi_enc", [5.0], [Q(12.0, 0, 5.0)])),
        ("trois moments : second membre −6 devenu −5", "seconds = [-6.0 * (g.wB0 + d.wA0)", "seconds = [-5.0 * (g.wB0 + d.wA0)",
         lambda M: M.resoudre("continue", [4.0, 4.0], [Q(10.0, 0, 8.0)])),
        ("console : moment d'encastrement sans le bras", "t0.M_A = _Q(t0.termes, 2, L, True) - t0.R_A * L", "t0.M_A = -t0.R_A * L",
         lambda M: M.resoudre("console", [3.0], [F(8.0, 2.0)]))):
    assert src.count(avant) == 1, nom
    espace = {"__name__": "rdm_poutres_mute"}
    exec(compile(src.replace(avant, apres), "poutres_mute.py", "exec"), espace)
    M_ = type("M", (), {k: staticmethod(v) if callable(v) and not isinstance(v, type) else v for k, v in espace.items() if not k.startswith("__")})
    try:
        essai(M_)
        chk(f"mutation du source « {nom} » : refusée", False, "le calcul abîmé est passé")
    except Exception as ex:                                # ErreurControle de l'espace muté (une autre classe que P.ErreurControle)
        chk(f"mutation du source « {nom} » : refusée ({type(ex).__name__})", type(ex).__name__ == "ErreurControle", ex)
chk("après les mutations, le moteur est revenu (q·L²/8 = 45)", pres(P.resoudre("appuis", [6.0], [Q(10.0, 0, 6.0)]).extremes["M_max"][0], 45.0))

# ================================================================================================ 6 · débords
print("\n=== 6 · Débords : porte-à-faux aux rives (« appuis », « continue ») ; abscisses depuis le bout gauche de la barre")


def Ap(sol, nom):
    return next(a for a in sol.appuis if a["nom"] == nom)


# 6.1 · formulaire (Roark, tableau 8.1 ; formulaire des poutres à porte-à-faux), réécrit ici, contre les paramètres initiaux
q, g, L, EI = 10.0, 2.0, 6.0, 21000.0


def debord_gauche_q(M=P):
    """Débord gauche seul chargé (q sur [0 ; g]) : M_A = −q·g²/2 en x = g, R_B = −q·g²/(2·L), E·I·f du bout libre
    = q·g³·(3·g + 4·L)/24, E·I·f au milieu de la travée = M_A·L²/16 (vers le haut)."""
    s_ = M.resoudre("appuis", [L], [Q(q, 0, g)], EI, debords=(g, 0.0))
    return (pres(Ap(s_, "A")["x"], g) and pres(Ap(s_, "A")["M"], -q * g * g / 2) and pres(R(s_, "A"), q * g + q * g * g / (2 * L))
            and pres(R(s_, "B"), -q * g * g / (2 * L)) and pres(s_.evaluer(0.0)["EIf"], q * g ** 3 * (3 * g + 4 * L) / 24)
            and pres(s_.evaluer(g + L / 2)["EIf"], -q * g * g / 2 * L * L / 16) and pres(s_.evaluer(g + L / 2)["M_d"], -q * g * g / 4))


chk("débord gauche seul chargé (q = 10, g = 2, L = 6) : M_A = −q·g²/2 = −20 en x = g, R_B = −q·g²/(2·L), "
    "E·I·f du bout libre = q·g³·(3·g + 4·L)/24 = 100, au milieu M_A·L²/16", debord_gauche_q())
Pp, dr = 8.0, 1.5
s = P.resoudre("appuis", [L], [F(Pp, L + dr)], EI, debords=(0.0, dr))
chk("débord droit, P au bout libre : M_B = −P·d = −12, R_A = −P·d/L, R_B = P·(L + d)/L, E·I·f du bout = P·d²·(L + d)/3 = 45",
    pres(Mapp(s, "B"), -Pp * dr) and pres(R(s, "A"), -Pp * dr / L) and pres(R(s, "B"), Pp * (L + dr) / L)
    and pres(s.evaluer(L + dr)["EIf"], Pp * dr * dr * (L + dr) / 3) and pres(s.evaluer(L + dr)["f"], Pp * dr * dr * (L + dr) / 3 / EI)
    and pres(s.extremes["EIf_max"][0], Pp * dr * dr * (L + dr) / 3) and pres(s.extremes["EIf_max"][1], L + dr), s.appuis)
Pc, gc, Lc2 = 5.0, 1.2, 4.0
s = P.resoudre("continue", [Lc2, Lc2], [F(Pc, 0.0)], EI, debords=(gc, 0.0))
chk("continue 2 × 4 m, P au bout du débord gauche (g = 1,2) : M_A = −P·g, trois moments M_B = +P·g/4, "
    "E·I·f du bout = P·g³/3 + 7·P·g²·L/24",
    pres(Mapp(s, "A"), -Pc * gc) and pres(Mapp(s, "B"), Pc * gc / 4) and pres(Mapp(s, "C"), 0.0, ech=Pc)
    and pres(s.evaluer(0.0)["EIf"], Pc * gc ** 3 / 3 + 7 * Pc * gc * gc * Lc2 / 24), s.appuis)
Cd, qt, dt = 6.0, 9.0, 1.8
s = P.resoudre("appuis", [L], [Cp(Cd, 0.7), T(qt, 1.0 + L, 1.0 + L + dt)], EI, debords=(1.0, dt))
chk("moments de rive : couple C horaire sur le débord gauche → M_A = +C ; triangle 0 → q sur le débord droit → M_B = −q·d²/3",
    pres(Mapp(s, "A"), Cd) and pres(Mapp(s, "B"), -qt * dt * dt / 3) and pres(s.evaluer(1.0 + L + dt)["M_g"], 0.0, ech=qt * dt * dt), s.appuis)

# 6.2 · équilibre, écrit depuis les données (pas depuis le moteur)
Ls6, g6, d6 = [4.0, 5.0, 3.0], 1.0, 1.5
Lt6 = g6 + sum(Ls6) + d6
ch6 = [Q(10.0, 0.0, Lt6), F(20.0, 0.4), F(12.0, Lt6), Cp(6.0, 13.7), T(8.0, 9.0, Lt6, "decroissante")]
s = P.resoudre("continue", Ls6, ch6, EI, debords=(g6, d6))
F6 = 10.0 * Lt6 + 20.0 + 12.0 + 8.0 * 5.5 / 2
M6 = 10.0 * Lt6 ** 2 / 2 + 20.0 * 0.4 + 12.0 * Lt6 + 6.0 + 8.0 * 5.5 / 2 * (9.0 + 5.5 / 3)
chk("équilibre (continue 4 + 5 + 3 m, débords 1 et 1,5 m, cinq charges dont trois sur les débords) : Σ R = Σ charges = 199 kN, "
    "Σ R·x = Σ F·x + Σ C (par rapport au bout gauche)",
    pres(sum(a["R"] for a in s.appuis), F6) and pres(F6, 199.0) and pres(sum(a["R"] * a["x"] for a in s.appuis), M6)
    and [a["x"] for a in s.appuis] == [1.0, 5.0, 10.0, 13.0], (sum(a["R"] for a in s.appuis), F6))
e0, eL = s.evaluer(0.0), s.evaluer(Lt6)
chk("bouts libres : rien à gauche du bout gauche ni à droite du bout droit (V = M = 0), le tranchant saute de P au bout droit",
    e0["V_g"] == 0.0 and e0["M_g"] == 0.0 and pres(e0["M_d"], 0.0, ech=100) and eL["V_d"] == 0.0 and eL["M_d"] == 0.0
    and pres(eL["V_g"], 12.0) and pres(eL["M_g"], 0.0, ech=100), (e0, eL))
chk("contrôles internes tenus, dont ceux des débords (moment de rive des résultantes, continuité, bout libre)",
    all(c["ok"] for c in s.controles) and sum("débord" in c["nom"] for c in s.controles) == 2 * 3 + 2 + 2 * 3, [c["nom"] for c in s.controles])

# 6.3 · symétrie
q7, g7, L7 = 10.0, 1.5, 6.0
s = P.resoudre("appuis", [L7], [Q(q7, 0.0, L7 + 2 * g7)], EI, debords=(g7, g7))
chk("deux appuis, débords égaux, q partout : R_A = R_B = q·(L + 2·g)/2, M_A = M_B = −q·g²/2, M au milieu = q·L²/8 − q·g²/2, "
    "E·I·f du bout = q·g·(3·g³ + 6·g²·L − L³)/24, au milieu 5·q·L⁴/384 − q·g²·L²/16",
    pres(R(s, "A"), q7 * (L7 + 2 * g7) / 2) and pres(R(s, "B"), R(s, "A")) and pres(Mapp(s, "A"), -q7 * g7 * g7 / 2) and pres(Mapp(s, "B"), Mapp(s, "A"))
    and pres(s.evaluer(g7 + L7 / 2)["M_d"], q7 * L7 * L7 / 8 - q7 * g7 * g7 / 2)
    and pres(s.evaluer(0.0)["EIf"], q7 * g7 * (3 * g7 ** 3 + 6 * g7 * g7 * L7 - L7 ** 3) / 24) and pres(s.evaluer(L7 + 2 * g7)["EIf"], s.evaluer(0.0)["EIf"])
    and pres(s.evaluer(g7 + L7 / 2)["EIf"], 5 * q7 * L7 ** 4 / 384 - q7 * g7 * g7 / 2 * L7 * L7 / 8)
    and abs(s.evaluer(g7 + L7 / 2)["EIrot"]) < 1e-9 * q7 * L7 ** 3, (s.appuis, s.evaluer(0.0)))
s = P.resoudre("continue", [4.0, 6.0, 4.0], [Q(5.0, 0.0, 17.0), F(10.0, 8.5)], EI, debords=(1.5, 1.5))
chk("continue symétrique à débords égaux : R_A = R_D, R_B = R_C, M_A = M_D = −q·g²/2, M_B = M_C, flèches des bouts égales, rotation nulle au milieu",
    pres(R(s, "A"), R(s, "D")) and pres(R(s, "B"), R(s, "C")) and pres(Mapp(s, "A"), -5.0 * 1.5 ** 2 / 2) and pres(Mapp(s, "D"), Mapp(s, "A"))
    and pres(Mapp(s, "B"), Mapp(s, "C")) and pres(s.evaluer(0.0)["EIf"], s.evaluer(17.0)["EIf"]) and abs(s.evaluer(8.5)["EIrot"]) < 1e-9 * 1000, s.appuis)

# 6.4 · débord nul = le modèle d'avant les débords
# valeurs FIGÉES le 17/11/2026 sur f012e40 (rdm/poutres.py sans débords) : copie nommée, comparée à 1e-12
FIGE_R = [4.744061705765324, 68.67122871701828, 74.21054183806658, 12.278093541935117, 1.5960741972147092]
FIGE_M = [0.0, -39.773753176938705, -32.697301063020674, -2.739344290395947, 0.0]
FIGE_EXT = {"M_max": (33.764472880020314, 6.5), "M_min": (-39.773753176938705, 4.0), "EIf_max": (56.72642253065569, 6.4507115887768816),
            "EIf_min": (-7.988977546271162, 3.1745947970714354)}
FIGE_73 = (19.29670521824721, 45.9331919980324)


def fige():
    s_ = P.resoudre("continue", [4.0, 5.0, 3.5, 6.0], ch, 1.0, debords=(0.0, 0.0))
    e_ = s_.evaluer(7.3)
    return (all(pres(a["R"], r_, 1e-12) for a, r_ in zip(s_.appuis, FIGE_R)) and all(pres(a["M"], m_, 1e-12, 100) for a, m_ in zip(s_.appuis, FIGE_M))
            and all(pres(s_.extremes[k][0], v_[0], 1e-12) and pres(s_.extremes[k][1], v_[1], 1e-12) for k, v_ in FIGE_EXT.items())
            and pres(e_["M_d"], FIGE_73[0], 1e-12) and pres(e_["EIf"], FIGE_73[1], 1e-12))


chk("débord nul : le cas composé (4 travées, 6 charges) rend les valeurs figées avant les débords, à 1e-12 (réactions, moments, extremums, M et f à 7,3 m)", fige())


def empreinte(sol):
    return (repr(sol.appuis), repr(sol.extremes), repr(sol.equations), repr([(c["nom"], c["ecart"]) for c in sol.controles]),
            repr([sol.evaluer(x) for x in (0.0, 0.37 * sol.L, 0.5 * sol.L, sol.L)]), repr(sol.courbes(8)))


pareil = True
for modele, Lx, chx in (("appuis", [5.0], c1 + c2), ("console", [5.0], c1 + c2), ("enc_app", [5.0], c1), ("bi_enc", [5.0], c2),
                        ("continue", [4.0, 5.0, 3.5, 6.0], ch)):
    ref = empreinte(P.resoudre(modele, Lx, chx, 3.0))
    pareil = pareil and all(empreinte(P.resoudre(modele, Lx, chx, 3.0, debords=dx)) == ref for dx in ((0.0, 0.0), [0, 0], None))
chk("débord nul, sous toutes ses écritures ((0, 0), [0, 0], None) et pour les cinq modèles : résultat identique au bit près à l'appel sans débord", pareil)
g8, d8, L8 = 1.2, 0.8, [4.0, 5.0, 3.5]
ch8 = [Q(10.0, 1.0, 11.0), F(25.0, 6.5), Cp(15.0, 4.0), T(6.0, 0.5, 3.0)]
decale = [dict(c, a=c["a"] + g8, **({"b": c["b"] + g8} if "b" in c else {})) for c in ch8]
s0, s8 = P.resoudre("continue", L8, ch8, EI), P.resoudre("continue", L8, decale, EI, debords=(g8, d8))
th_A, th_D = s0.evaluer(0.0)["EIrot"], s0.evaluer(12.5)["EIrot"]
chk("débords NON chargés : travées identiques au modèle sans débord décalé de g (réactions, moments, M, V, f à 1e-12) ; "
    "débords : M = V = 0, f du bout = −θ_A·g à gauche, +θ_D·d à droite",
    all(pres(a["R"], b_["R"], 1e-12, 100) and pres(a["M"], b_["M"], 1e-12, 100) and pres(a["x"], b_["x"] + g8, 1e-12) for a, b_ in zip(s8.appuis, s0.appuis))
    and all(pres(s8.evaluer(x + g8)[k], s0.evaluer(x)[k], 1e-12, 1e5) for x in (0.7, 4.0, 6.5, 9.1, 12.5) for k in ("M_g", "M_d", "V_g", "V_d", "EIf"))
    and pres(s8.evaluer(0.0)["EIf"], -th_A * g8, 1e-12) and pres(s8.evaluer(12.5 + g8 + d8)["EIf"], th_D * d8, 1e-12)
    and s8.evaluer(0.5)["M_d"] == 0.0 and s8.evaluer(0.5)["V_d"] == 0.0, (s8.appuis, s0.appuis))

# 6.5 · refus, repère, tronçons
for nom, args in (("débord d'une console", ("console", [3.0], [], None, (1.0, 0.0))), ("débord d'une encastrée–appuyée", ("enc_app", [3.0], [], None, (0.0, 1.0))),
                  ("débord d'une bi-encastrée", ("bi_enc", [3.0], [], None, (1.0, 1.0))), ("débord négatif", ("appuis", [3.0], [], None, (-0.5, 0.0))),
                  ("une seule longueur", ("appuis", [3.0], [], None, (1.0,))), ("débord non fini", ("continue", [3.0, 3.0], [], None, (float("inf"), 0.0))),
                  ("charge au-delà du bout droit", ("appuis", [3.0], [F(1.0, 4.6)], None, (1.0, 0.5)))):
    try:
        P.resoudre(*args)
        chk(f"refus : {nom}", False, "accepté")
    except P.ErreurDonnees as ex:
        chk(f"refus : {nom} ({ex})", True)
s = P.resoudre("appuis", [3.4], [Q(2.0, 0.0, round(0.3 + 3.4 + 0.5, 12)), F(1.0, 0.3 + 3.4 + 0.5 + 1e-12)], None, (0.3, 0.5))
chk("repère : la barre va de 0 à g + L + d (0,3 + 3,4 + 0,5), la charge par défaut la couvre, un bout tapé est ramené au bout",
    pres(s.L, 4.2) and [round(a["x"], 12) for a in s.appuis] == [0.3, 3.7] and [t.debord for t in s.travees] == ["gauche", None, "droite"]
    and pres(sum(a["R"] for a in s.appuis), 2.0 * 4.2 + 1.0)
    and pres(sum(a["R"] for a in P.resoudre("appuis", [3.4], [{"type": "q", "v": 2.0, "a": 0.0}], None, (0.3, 0.5)).appuis), 8.4), s.appuis)
chk("une poutre à débord n'est pas un cas d'école (None) ; sans débord, le même cas l'est",
    P.cas_ecole(P.resoudre("appuis", [6.0], [Q(1.0, 1.0, 7.0)], None, (1.0, 0.0))) is None
    and P.cas_ecole(P.resoudre("appuis", [6.0], [Q(1.0, 0.0, 6.0)]))["cle"] == "appuis_q")



def debord_minuscule(M=P):
    """(intégration du lot C) Un débord d'au plus TOL_APPUI·max(1, L) — un résidu de soustraction — est un débord NUL : la
    charge du bout reste à l'appui, et le tranchant lu en 0 est celui de la poutre sans débord (30 kN, et non −5 kN d'un
    tronçon que _travee n'atteint pas, ni 30 kN lus dans la travée PAR ERREUR à côté d'un débord gardé)."""
    s0 = M.resoudre("appuis", [6.0], [Q(10.0, 0.0, 6.0), F(5.0, 0.0), F(5.0, 6.0)], None)
    for e in (1e-13, 1e-10):
        s_ = M.resoudre("appuis", [6.0], [Q(10.0, 0.0, 6.0 + 2 * e), F(5.0, 0.0), F(5.0, 6.0 + 2 * e)], None, debords=(e, e))
        if not ([t.debord for t in s_.travees] == [None] and s_.debords == (0.0, 0.0) and s_.appuis[0]["x"] == 0.0
                and all(pres(a["R"], b_["R"], 1e-9) for a, b_ in zip(s_.appuis, s0.appuis))
                and all(pres(s_.evaluer(0.0)[k], s0.evaluer(0.0)[k], 1e-9) for k in ("V_g", "V_d", "M_d"))):
            return False
    s_ = M.resoudre("appuis", [6.0], [F(5.0, 0.0)], None, debords=(1e-8, 0.0))       # au-delà de la tolérance : un vrai débord
    return [t.debord for t in s_.travees] == ["gauche", None] and pres(s_.evaluer(0.0)["V_d"], -5.0)


chk("débord minuscule (10⁻¹³ et 10⁻¹⁰ m) = débord nul : un seul tronçon, réactions et V, M en 0 de la poutre sans débord ; "
    "10⁻⁸ m reste un débord (V_d(0) = −5)", debord_minuscule())

# 6.6 · mutations des débords (le source muté doit être refusé : contrôle interne ou formulaire de ce fichier)
eq_cont = lambda M: M.resoudre("continue", Ls6, ch6, EI, debords=(g6, d6))                     # noqa: E731
for nom, avant, apres, essai in (
        ("moment de rive oublié (trois moments : M_A = 0 au lieu du moment du débord)",
         "Mi = [M_g] + _resoudre_tridiagonal(Ls, connus) + [M_d]", "Mi = [0.0] + _resoudre_tridiagonal(Ls, connus) + [0.0]", eq_cont),
        ("moment de rive oublié au second membre des trois moments", "connus[0] -= Ls[0] * M_g", "connus[0] -= 0.0", eq_cont),
        ("moment de rive oublié (deux appuis)", "t0.fermer(M_g, M_d)", "t0.fermer(0.0, 0.0)", lambda M: attendu(debord_gauche_q(M))),
        ("débord compté dans la portée (l'appui A au bout de la barre)", "tr = _repartir(Ls, ch, g_, d_)",
         "tr = _repartir([g_ + Ls[0]] + Ls[1:], ch, 0.0, d_)", lambda M: attendu(debord_gauche_q(M))),
        ("flèche du débord sans la rotation de l'appui", "dg.raccorder(t0.EIrot(0.0))", "dg.raccorder(0.0)", lambda M: attendu(debord_gauche_q(M))),
        ("charge du débord droit absente de la réaction de rive", '"R": tn.R_B if dd is None else tn.R_B + dd.R_A', '"R": tn.R_B', eq_cont),
        ("débord minuscule gardé (un tronçon que _travee n'atteint pas)", "g, d = (0.0 if g <= tol else g), (0.0 if d <= tol else d)",
         "g, d = g, d", lambda M: attendu(debord_minuscule(M)))):
    src = open(os.path.join(RACINE, "rdm", "poutres.py"), encoding="utf-8").read()
    assert src.count(avant) == 1, nom
    espace = {"__name__": "rdm_poutres_mute"}
    exec(compile(src.replace(avant, apres), "poutres_mute.py", "exec"), espace)
    M_ = type("M", (), {k: staticmethod(v) if callable(v) and not isinstance(v, type) else v for k, v in espace.items() if not k.startswith("__")})
    try:
        essai(M_)
        chk(f"mutation du source « {nom} » : refusée", False, "le calcul abîmé est passé")
    except Exception as ex:                                # ErreurControle de l'espace muté, ou le formulaire non retrouvé
        chk(f"mutation du source « {nom} » : refusée ({type(ex).__name__})", type(ex).__name__ in ("ErreurControle", "AssertionError"), ex)
chk("après les mutations des débords, le moteur est revenu (débord gauche : M_A = −q·g²/2)", debord_gauche_q())

# ================================================================================================ 7 · sans dépendance
import subprocess  # noqa: E402

sonde = ("import sys\nclass R:\n    def find_spec(self, n, p=None, t=None):\n"
         "        if n.split('.')[0] in ('numpy','scipy','streamlit','matplotlib','reportlab','pandas'):\n"
         "            raise ImportError(n)\nsys.meta_path.insert(0, R())\n"
         f"sys.path.insert(0, {RACINE!r})\nfrom rdm import poutres as P\n"
         "print(P.resoudre('appuis', [6.0], [{'type': 'q', 'v': 10.0, 'a': 0.0, 'b': 6.0}]).extremes['M_max'][0])")
r = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True)
chk("le moteur charge et résout sans numpy ni scipy (bibliothèque standard seule)", r.returncode == 0 and r.stdout.strip() == "45.0", r.stderr[-300:])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)

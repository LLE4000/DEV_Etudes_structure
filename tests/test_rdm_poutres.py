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
  5. des mutations du moteur, qui rougissent.
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

# ================================================================================================ 6 · sans dépendance
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

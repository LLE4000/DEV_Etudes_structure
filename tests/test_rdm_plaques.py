# -*- coding: utf-8 -*-
"""Le moteur des PLAQUES RECTANGULAIRES (rdm/plaques.py, docs/site/BIBLIOTHEQUE_AUTRES.md § 4).

    /usr/bin/python3.12 tests/test_rdm_plaques.py

Ce que la suite prouve, par des contrôles INDÉPENDANTS des séries du moteur :
  1. les valeurs PUBLIÉES (Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2ᵉ éd., ν = 0,3) : copies
     nommées, avec leur tableau ;
  2. chaque solution VÉRIFIE le problème : D·∇⁴w = q à l'intérieur et les conditions de chaque bord (flèche, rotation,
     moment, effort tranchant de Kirchhoff), mesurés par DIFFÉRENCES FINIES sur la seule fonction w(x, y) — une fonction
     qui satisfait l'équation et toutes les conditions aux limites EST la solution (unicité) ;
  3. les méthodes s'accordent où elles se recouvrent : Navier = Lévy (plaque appuyée), Lévy = superposition (plaques de
     Lévy sans bord libre), Lévy en x = Lévy en y ;
  4. les limites de poutre (plaque très allongée), les symétries, la linéarité, les refus ;
  5. des mutations du moteur, que les contrôles 1 et 2 voient.
"""
import math
import os
import sys

RACINE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RACINE)

from rdm import plaques as PL  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def B(code):
    """« AAEL » → bords gauche (x = 0), droit (x = a), bas (y = 0), haut (y = b)."""
    return dict(zip(PL.COTES, code))


def calc(code, a=1.0, b=1.0, nu=0.3, q=1.0):
    return PL.calculer(a, b, B(code), q, 210000.0, nu, 10.0)


# ================================================================================================ 1 · valeurs publiées
print("=== 1 · Valeurs publiées (Timoshenko & Woinowsky-Krieger, ν = 0,3)")
# (nom, bords, a, b, {grandeur: (valeur publiée, tolérance absolue)}) — la tolérance est UNE unité du dernier chiffre publié ;
# DEUX pour deux moments d'encastrement (−0,0571 et −0,0697) : les tableaux de 1959 sont calculés avec peu de termes, et la
# solution du moteur, qui vérifie l'équation et les bords (contrôle 2), donne −0,05699 et −0,06984
PUBLIE = [
    ("tableau 8 — quatre bords appuyés, carrée", "AAAA", 1, 1, {"alpha": (0.00406, 1e-5), "bx": (0.0479, 1e-4), "by": (0.0479, 1e-4)}),
    ("tableau 8 — quatre bords appuyés, b/a = 2", "AAAA", 1, 2, {"alpha": (0.01013, 1e-5), "bx": (0.1017, 1e-4), "by": (0.0464, 1e-4)}),
    ("tableau 8 — quatre bords appuyés, b/a = 1,5", "AAAA", 1, 1.5, {"alpha": (0.00772, 1e-5), "bx": (0.0812, 1e-4), "by": (0.0498, 1e-4)}),
    ("tableau 35 — quatre bords encastrés, carrée", "EEEE", 1, 1, {"alpha": (0.00126, 1e-5), "enc_x0": (-0.0513, 1e-4), "enc_y0": (-0.0513, 1e-4)}),
    ("tableau 35 — quatre bords encastrés, b/a = 2", "EEEE", 1, 2, {"alpha": (0.00254, 1e-5), "enc_x0": (-0.0829, 1e-4), "enc_y0": (-0.0571, 2e-4),
                                                                   "bx": (0.0412, 1e-4), "by": (0.0158, 1e-4)}),
    ("tableau 29 — deux bords opposés appuyés, deux encastrés, carrée", "AAEE", 1, 1, {"alpha": (0.00192, 1e-5), "enc_y0": (-0.0697, 2e-4),
                                                                                    "bx": (0.0244, 1e-4), "by": (0.0332, 1e-4)}),
    ("tableau 30 — trois bords appuyés, un encastré, carrée", "AAEA", 1, 1, {"alpha_c": (0.0028, 1e-4), "enc_y0": (-0.084, 1e-3),
                                                                           "bx": (0.034, 1e-3), "by": (0.039, 1e-3)}),
    ("§ 46 — trois bords appuyés, un libre, carrée (milieu du bord libre)", "AAAL", 1, 1, {"lib_w": (0.01286, 1e-5), "lib_M": (0.112, 1e-3)}),
    ("§ 47 — deux appuyés, un encastré, le bord opposé libre, carrée", "AAEL", 1, 1, {"lib_w": (0.0113, 1e-4), "lib_M": (0.0972, 1e-4),
                                                                                    "enc_y0": (-0.119, 1e-3)}),
]
for nom, code, a, b, att in PUBLIE:
    r = calc(code, a, b)
    obt = {"alpha": r["w_max"]["alpha"], "alpha_c": r["centre"]["w"] * r["D"] / r["L"] ** 4, "bx": r["centre"]["beta_x"], "by": r["centre"]["beta_y"]}
    for e in r["encastrements"]:
        obt["enc_" + e["cote"]] = e["milieu"] / r["L"] ** 2
    for e in r["libres"]:
        obt["lib_w"], obt["lib_M"] = e["alpha"], e["beta"]
    ecarts = {k: (round(obt[k], 5), v) for k, (v, tol) in att.items() if abs(obt[k] - v) > tol * 1.01}
    chk(f"{nom} : " + ", ".join(f"{k} = {v}" for k, (v, _) in att.items()), not ecarts, ecarts)
# la plaque encastrée carrée : la valeur moderne du moment au centre (0,02291 ; le tableau 35 de 1959 porte 0,0231)
r = calc("EEEE")
chk("quatre bords encastrés, carrée : w = 0,0012653·q·a⁴/D, M_bord = −0,051334·q·a², M_centre = 0,022905·q·a² "
    "(Taylor & Govindjee, 2004 ; le tableau 35 arrondit le centre à 0,0231)",
    abs(r["w_max"]["alpha"] - 0.0012653) < 2e-7 and abs(r["encastrements"][0]["beta"] + 0.051334) < 2e-5 and abs(r["centre"]["beta_x"] - 0.022905) < 2e-6,
    (r["w_max"]["alpha"], r["encastrements"][0]["beta"], r["centre"]["beta_x"]))

# ================================================================================================ 2 · l'équation et les bords
print("\n=== 2 · Chaque solution vérifie D·∇⁴w = q et ses conditions aux limites (différences finies sur w seul)")


def residus(code, a, b, nu=0.3):
    """Les plus grands écarts, RELATIFS, de la solution au problème posé : équation (13 points), puis par bord la flèche,
    la rotation (encastré), le moment (appuyé, libre) et l'effort tranchant de Kirchhoff (libre). q = D = 1."""
    S = PL._resoudre_plaque(a, b, B(code), nu)
    w = lambda x, y: S.brut(x, y)["w"]                                                                   # noqa: E731
    L = min(a, b)
    h = L / 50.0
    ech_w = max(abs(w(a * i / 8, b * j / 8)) for i in range(9) for j in range(9)) or 1.0
    out = {"equation": 0.0}
    for fx, fy in ((0.5, 0.5), (0.3, 0.6), (0.7, 0.25), (0.2, 0.2), (0.8, 0.8)):
        x, y = fx * a, fy * b
        lap2 = (20 * w(x, y) - 8 * (w(x + h, y) + w(x - h, y) + w(x, y + h) + w(x, y - h))
                + 2 * (w(x + h, y + h) + w(x - h, y + h) + w(x + h, y - h) + w(x - h, y - h))
                + w(x + 2 * h, y) + w(x - 2 * h, y) + w(x, y + 2 * h) + w(x, y - 2 * h)) / h ** 4
        out["equation"] = max(out["equation"], abs(lap2 - 1.0))
    for cote in PL.COTES:
        typ = S.bords[cote]
        le_long_x = cote in ("y0", "yb")
        Lb = a if le_long_x else b
        pire = {"w": 0.0, "rotation": 0.0, "moment": 0.0, "tranchant": 0.0}
        for fs in (0.25, 0.5, 0.7):
            s = fs * Lb

            def pt(n_, t_=0.0):
                """Le point à la distance n_ du bord (vers l'intérieur), décalé de t_ le long du bord."""
                if cote == "y0":
                    return s + t_, n_
                if cote == "yb":
                    return s + t_, b - n_
                if cote == "x0":
                    return n_, s + t_
                return a - n_, s + t_
            # dérivées normales (vers l'intérieur) par différences décentrées d'ordre 4 ; dérivée tangentielle centrée d'ordre 4
            W = [w(*pt(k * h)) for k in range(7)]
            d1 = (-25 * W[0] + 48 * W[1] - 36 * W[2] + 16 * W[3] - 3 * W[4]) / (12 * h)
            d2 = (45 * W[0] - 154 * W[1] + 214 * W[2] - 156 * W[3] + 61 * W[4] - 10 * W[5]) / (12 * h ** 2)
            d3 = (-49 * W[0] + 232 * W[1] - 461 * W[2] + 496 * W[3] - 307 * W[4] + 104 * W[5] - 15 * W[6]) / (8 * h ** 3)

            def w_tt(n_):
                return (-w(*pt(n_, 2 * h)) + 16 * w(*pt(n_, h)) - 30 * w(*pt(n_)) + 16 * w(*pt(n_, -h)) - w(*pt(n_, -2 * h))) / (12 * h ** 2)
            wtt = w_tt(0.0)
            tt = [w_tt(k * h) for k in range(5)]
            wttn = (-25 * tt[0] + 48 * tt[1] - 36 * tt[2] + 16 * tt[3] - 3 * tt[4]) / (12 * h)        # la dérivée normale de w_tt
            if typ in ("A", "E"):
                pire["w"] = max(pire["w"], abs(W[0]) / ech_w)
            if typ == "E":
                pire["rotation"] = max(pire["rotation"], abs(d1) / (ech_w / L))
            if typ in ("A", "L"):
                pire["moment"] = max(pire["moment"], abs(d2 + nu * wtt) / (ech_w / L ** 2))
            if typ == "L":
                pire["tranchant"] = max(pire["tranchant"], abs(d3 + (2 - nu) * wttn) / (ech_w / L ** 3))
        for k, v in pire.items():
            out[f"{cote}:{k}"] = v
    return out


TOL = {"equation": 5e-3, "w": 1e-6, "rotation": 2e-3, "moment": 1e-2, "tranchant": 5e-2}
for code, a, b in (("AAAA", 1, 1.4), ("EEEE", 1, 1), ("EEEE", 1.6, 1), ("AAEE", 1, 1.3), ("EEAA", 1.2, 1), ("AAEA", 1, 1), ("EAAA", 1, 1.5),
                   ("AAAL", 1, 1), ("AALA", 1, 1.2), ("AAEL", 1, 1), ("AALE", 1.3, 1), ("AALL", 1, 0.8), ("LAAA", 1, 1), ("ELAA", 1, 1.2),
                   ("LLAA", 0.9, 1), ("EAEA", 1, 1), ("AEAE", 1, 1.4), ("EEEA", 1, 1), ("EAEE", 1.5, 1), ("AEEA", 1, 1.2)):
    rs = residus(code, a, b)
    faux = {k: f"{v:.1e}" for k, v in rs.items() if v > TOL[k.split(":")[-1]]}
    chk(f"{code} {a} × {b} : équation (écart {rs['equation']:.0e}) et conditions des quatre bords tenues", not faux, faux)

# ================================================================================================ 3 · méthodes
print("\n=== 3 · Les méthodes s'accordent où elles se recouvrent")
for a, b in ((1, 1), (1, 1.7), (2.2, 1)):
    r = calc("AAAA", a, b)
    nv = r["controles"]["navier"]
    chk(f"plaque appuyée {a} × {b} : Navier (double série) = Lévy au centre (w à {nv['ecart_w']:.0e}, M à {nv['ecart_M']:.0e})",
        nv["ecart_w"] < 1e-6 and nv["ecart_M"] < 1e-4, nv)
for code, a, b in (("AAEE", 1, 1), ("AAEA", 1, 1.3), ("EEAA", 1.4, 1), ("AEAA", 1, 1), ("AAAA", 1, 1.2)):
    S1 = PL._resoudre_plaque(a, b, B(code), 0.3)
    S2 = PL._resoudre_plaque(a, b, B(code), 0.3, forcer="superposition")
    pire = 0.0
    for fx, fy in ((0.5, 0.5), (0.3, 0.7), (0.8, 0.2)):
        c1, c2 = S1.champ(fx * a, fy * b), S2.champ(fx * a, fy * b)
        pire = max(pire, abs(c1["w"] - c2["w"]) / abs(c1["w"]), abs(c1["Mx"] - c2["Mx"]) / 0.05, abs(c1["My"] - c2["My"]) / 0.05)
    chk(f"{code} {a} × {b} : Lévy (méthode {S1.methode}) = superposition (méthode {S2.methode}), w et moments à {pire:.0e}",
        S1.methode == "levy" and S2.methode == "superposition" and pire < 2e-4, pire)
r1, r2 = calc("AAEL", 1, 1.3), calc("ELAA", 1.3, 1)
chk("la plaque tournée d'un quart de tour (Lévy en x ↔ Lévy en y) donne les mêmes coefficients",
    abs(r1["w_max"]["alpha"] - r2["w_max"]["alpha"]) < 1e-9 and abs(r1["centre"]["beta_x"] - r2["centre"]["beta_y"]) < 1e-9
    and abs(r1["encastrements"][0]["beta"] - r2["encastrements"][0]["beta"]) < 1e-9, (r1["w_max"]["alpha"], r2["w_max"]["alpha"]))
r1, r2 = calc("EAEA", 1, 1.4), calc("AEAE", 1, 1.4)
chk("plaque retournée (encastrements aux bords opposés) : même flèche maximale, au point symétrique",
    abs(r1["w_max"]["alpha"] - r2["w_max"]["alpha"]) < 1e-8 and abs(r1["w_max"]["x"] + r2["w_max"]["x"] - 1) < 1e-3
    and abs(r1["w_max"]["y"] + r2["w_max"]["y"] - 1.4) < 1e-3)

# ================================================================================================ 4 · limites, propriétés, refus
print("\n=== 4 · Limites de poutre, linéarité, refus")
r = calc("AAAA", 1, 8)
chk("plaque appuyée très allongée (b/a = 8) : la bande sur deux appuis — w = 5/384·q·a⁴/D, M = q·a²/8",
    abs(r["centre"]["w"] * r["D"] - 5 / 384) < 2e-5 and abs(r["centre"]["Mx"] - 1 / 8) < 2e-4, (r["centre"]["w"] * r["D"], r["centre"]["Mx"]))
r = calc("EEAA", 1, 8)
chk("bande encastrée (longs bords encastrés, b/a = 8) : w = 1/384·q·a⁴/D, M_enc = −q·a²/12, M_centre = q·a²/24",
    abs(r["centre"]["w"] * r["D"] - 1 / 384) < 2e-5 and abs(r["encastrements"][0]["milieu"] + 1 / 12) < 2e-4 and abs(r["centre"]["Mx"] - 1 / 24) < 2e-4,
    (r["centre"]["w"] * r["D"], r["encastrements"][0]["milieu"], r["centre"]["Mx"]))
r = calc("AALL", 1, 1, nu=0.0)
chk("deux bords appuyés, deux libres, ν = 0 : flexion cylindrique — exactement la poutre (5/384, 1/8) en tout point",
    abs(r["centre"]["w"] * r["D"] - 5 / 384) < 1e-9 and abs(r["libres"][0]["w"] * r["D"] - 5 / 384) < 1e-9 and abs(r["centre"]["Mx"] - 1 / 8) < 1e-9
    and abs(r["centre"]["My"]) < 1e-9, (r["centre"]["w"] * r["D"], r["centre"]["My"]))
r1, r3 = calc("EAEL", 1, 1, q=1.0) if False else calc("AAEL"), calc("AAEL", q=3.0)
chk("linéarité : q × 3 → flèches et moments × 3, coefficients inchangés",
    abs(r3["w_max"]["valeur"] - 3 * r1["w_max"]["valeur"]) < 1e-12 and abs(r3["centre"]["Mx"] - 3 * r1["centre"]["Mx"]) < 1e-12
    and abs(r3["w_max"]["alpha"] - r1["w_max"]["alpha"]) < 1e-12)
rn = calc("AAEL", q=-2.0)
chk("charge vers le haut : flèche et moments changent de signe, coefficients inchangés",
    rn["w_max"]["valeur"] < 0 and abs(rn["w_max"]["alpha"] - r1["w_max"]["alpha"]) < 1e-12 and abs(rn["centre"]["Mx"] + 2 * r1["centre"]["Mx"]) < 1e-12)
r = PL.calculer(2.0, 3.0, B("AAAA"), 5.0, 210000.0, 0.3, 12.0)
D_main = 210000.0 * 12.0 ** 3 / (12 * (1 - 0.09)) * 1e-6
chk("unités : D = E·t³/(12·(1 − ν²)) = 33,23 kN·m (210 000 MPa, 12 mm) ; σ = 6·M/t² ; w = α·q·L⁴/D (calcul à la main)",
    abs(r["D"] - D_main) < 1e-9 and abs(D_main - 33.2308) < 1e-3 and abs(r["sigma_max"] - 6 * r["M_abs_max"] * 1000 / 144) < 1e-9
    and abs(r["w_max"]["valeur"] - r["w_max"]["alpha"] * 5.0 * 2.0 ** 4 / D_main) < 1e-12, (r["D"], r["sigma_max"]))
chk("avertissements : flèche > t/2 signalée (hors petites flèches) ; plaque épaisse (t > L/10) signalée",
    "grande_fleche" in r["avertissements"] and "epaisse" in PL.calculer(0.5, 0.5, B("AAAA"), 1.0, 210000.0, 0.3, 80.0)["avertissements"]
    and not PL.calculer(1.0, 1.0, B("AAAA"), 1.0, 210000.0, 0.3, 20.0)["avertissements"], r["avertissements"])
for code, mot in (("ELLL", "libre"), ("EAEL", "libre"), ("EELE", "libre"), ("LLLL", "libres"), ("ALEA", "libre"), ("ELAL", "libres")):
    try:
        calc(code)
        chk(f"refus : {code}", False, "accepté")
    except PL.NonPrisEnCharge as ex:
        chk(f"refus dit : {code} — {str(ex)[:60]}…", mot in str(ex) and "Lévy" in str(ex))
for nom, args in (("a nul", (0.0, 1.0, B("AAAA"), 1.0, 210000.0, 0.3, 10.0)), ("ν = 0,5", (1.0, 1.0, B("AAAA"), 1.0, 210000.0, 0.5, 10.0)),
                  ("bord inconnu", (1.0, 1.0, B("AAAX"), 1.0, 210000.0, 0.3, 10.0)), ("rapport 20", (20.0, 1.0, B("AAAA"), 1.0, 210000.0, 0.3, 10.0)),
                  ("q non fini", (1.0, 1.0, B("AAAA"), float("inf"), 210000.0, 0.3, 10.0))):
    try:
        PL.calculer(*args)
        chk(f"refus : {nom}", False, "accepté")
    except PL.ErreurDonnees as ex:
        chk(f"refus : {nom} ({ex})", True)
compte = {"levy": 0, "superposition": 0, "refus": 0}
for c in (x + y + z + t for x in "AEL" for y in "AEL" for z in "AEL" for t in "AEL"):
    try:
        compte[PL.classer(B(c))] += 1
    except PL.NonPrisEnCharge:
        compte["refus"] += 1
# (à la main : sans bord libre 2⁴ = 16 plaques, dont 4 + 4 − 1 = 7 de Lévy → 9 par superposition ; avec bord libre, Lévy
#  demande deux bords opposés appuyés : (3² − 2²) × 2 = 10 ; reste 81 − 17 − 9 = 55)
chk("les 81 combinaisons de bords : 17 par Lévy, 9 par superposition, 55 refusées (bord libre sans deux voisins appuyés) — aucune sans réponse",
    compte == {"levy": 17, "superposition": 9, "refus": 55}, compte)

# ================================================================================================ 5 · mutations
print("\n=== 5 · Mutations (les contrôles 1 et 2 doivent les voir)")
src = open(os.path.join(RACINE, "rdm", "plaques.py"), encoding="utf-8").read()
MUT = [
    ("bord libre : (2 − ν) devenu (1 − ν) dans l'effort tranchant", "d3[k] - (2 - nu) * a2 * d1[k]", "d3[k] - (1 - nu) * a2 * d1[k]", ("AAAL", 1, 1), "tranchant"),
    ("bord libre : moment sans le terme en ν", "[d2[k] - nu * a2 * f[k] for k in range(4)], nu * a2 * p)", "[d2[k] for k in range(4)], 0.0)", ("AAAL", 1, 1), "moment"),
    ("bord encastré : rotation remplacée par la courbure", 'return [(f, -p), (d1, 0.0)]', 'return [(f, -p), (d2, 0.0)]', ("AAEA", 1, 1), "rotation"),
    ("solution de la bande : 4/π⁵ devenu 4/π⁴", "p = 4.0 * Ls ** 4 / (math.pi ** 5 * mm ** 5)", "p = 4.0 * Ls ** 4 / (math.pi ** 4 * mm ** 5)", ("AAEA", 1, 1), "w"),
    ("superposition : couplage des bords perpendiculaires au carré manquant", "g = sg * be[n] * (2.0 / a) * al[k] / (al[k] ** 2 + be[n] ** 2) ** 2",
     "g = sg * be[n] * (2.0 / a) * al[k] / (al[k] ** 2 + be[n] ** 2)", ("EEEE", 1, 1), "rotation"),
    ("superposition : signe du bord opposé perdu", "A[li][idx[(\"xa\", n)]] -= g * (-1.0) ** h", "A[li][idx[(\"xa\", n)]] -= g", ("EEEE", 1, 1.3), "rotation"),
]
espace_test = {"B": B, "math": math}
for nom, avant, apres, (code, a, b), quoi in MUT:
    assert src.count(avant) == 1, nom
    espace = {"__name__": "rdm_plaques_mute"}
    exec(compile(src.replace(avant, apres), "plaques_mute.py", "exec"), espace)
    reel = PL._resoudre_plaque
    try:
        PL._resoudre_plaque = espace["_resoudre_plaque"]
        try:
            rs = residus(code, a, b)
            vu = any(v > TOL[k.split(":")[-1]] for k, v in rs.items() if k.split(":")[-1] == quoi or k == "equation")
        except ArithmeticError:
            vu = True
    finally:
        PL._resoudre_plaque = reel
    chk(f"mutation « {nom} » : le contrôle de {quoi} la voit ({code})", vu)
chk("après les mutations, le moteur est revenu (0,00406)", abs(calc("AAAA")["w_max"]["alpha"] - 0.00406) < 1e-5)

import subprocess  # noqa: E402

sonde = ("import sys\nclass R:\n    def find_spec(self, n, p=None, t=None):\n"
         "        if n.split('.')[0] in ('numpy','scipy','streamlit','matplotlib','reportlab','pandas'):\n"
         "            raise ImportError(n)\nsys.meta_path.insert(0, R())\n"
         f"sys.path.insert(0, {RACINE!r})\nfrom rdm import plaques as P\n"
         "print(round(P.calculer(1.0, 1.0, dict(zip(P.COTES, 'EEEE')), 1.0, 210000.0, 0.3, 10.0)['w_max']['alpha'], 5))")
rr = subprocess.run([sys.executable, "-c", sonde], capture_output=True, text=True)
chk("le moteur charge et résout sans numpy ni scipy (bibliothèque standard seule)", rr.returncode == 0 and rr.stdout.strip() == "0.00127", rr.stderr[-300:])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)

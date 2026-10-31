# -*- coding: utf-8 -*-
"""Tests du CALCUL MANUEL de la platine (lot D1) — chaque garantie rougit si
on la retire, et chaque nombre attendu est recalculé À LA MAIN dans le test :

  1. GÉOMÉTRIE : lamelles d'aire exacte d'un tube, appui le plus proche
     (face, angle, cercle), raidisseur, grille régulière reconnue ou non.
  2. EFFORTS : la même répartition que la vérification complète sur la même
     disposition (fjd, c, Aeff, kc identiques ; T identiques à kb égal) ;
     suspente : T = N/4 ; élément excentré : C = N sous l'élément ; résidus
     d'équilibre ; cisaillement partagé avec torsion.
  3. CÔTÉ TENDU à la main : boulon d'angle (bras diagonal, leff plafonnée à
     m, Lb*, mode 1 → tp = √(T·m/(leff·fy))) ; boulons devant l'aile (section
     au nu, largeur L + 2m, tp = √(4·M/(b·fy))) ; élastique = √1,5 × plastique
     sans effet de levier ; raidisseur qui raccourcit le bras.
  4. CÔTÉ COMPRIMÉ : pression uniforme → m = p·c²/2 exactement ; épaisseur
     requise = celle où la bande c porte N à fjd ; règle de Voronoï.
  5. TROUS : pression diamétrale du Tableau 3.4 à la main.
  6. INTUITIONS : rond fin ≫ RHS ; toute la platine ≠ bande c ; critères.
  7. VALIDITÉ : boulon dans l'élément, pince, entraxe, élément hors platine,
     aucun boulon, équilibre impossible ; temps de calcul.

Lancement : python3 tests/test_platine_manuelle.py (depuis la racine).
"""
import math
import os
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.pied_poteau.platine_beton import moteur as MC  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton import equilibre as EQ  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton import geometrie as GE  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.manuel import moteur as MM  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.manuel import elements as EL  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.manuel import console as CO  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.manuel import conversion as CV  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.entrees import (  # noqa: E402
    TYPE_RHS, TYPE_ROND, TYPE_CHS, TYPE_PLAT, CRIT_ELAS, APPUI_PLEIN, MODE_COMPLET, MODE_MANUEL,
    element_defaut, defaults, charger_json)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def tiges(*xy):
    return [dict(x=x, y=y) for x, y in xy]


CORD = 0.8 * 8 * math.sqrt(2)                      # cordon a = 8 déduit du bras

# ================================================================
print("=== 1. Géométrie ===")
chs = EL.normaliser(dict(element_defaut(TYPE_CHS), d=219.1, t=8))
lam = EL.bandes(chs, 20.0, (-1000, -1000, 1000, 1000))
A_anneau = math.pi * ((219.1 / 2 + 20) ** 2 - (219.1 / 2 - 8 - 20) ** 2)
chk("tube CHS 219,1×8 élargi de c = 20 : les lamelles ont l'aire EXACTE de l'anneau", pres(GE.aire_rects(lam), A_anneau, 1e-9),
    f"{GE.aire_rects(lam):.1f} / {A_anneau:.1f}")
a = EL.appui_proche(chs, 0, 200, 0)
chk("appui le plus proche d'un tube : radial, d = 200 − R", pres(a["d"], 200 - 219.1 / 2, 1e-12) and a.get("cercle") and not a["coin"])
hea = EL.normaliser(element_defaut())
chk("HEA 300 du catalogue : h 290, b 300, tw 8,5, tf 14 lus dans la bibliothèque (source unique)",
    (hea["h"], hea["b"], hea["tw"], hea["tf"]) == (290, 300, 8.5, 14))
a = EL.appui_proche(hea, 0, -200, -200)
chk("boulon (−200 ; −200) : angle de l'aile inférieure, d = √(50² + 55²), deux faces (−y, −x)",
    a["coin"] and pres(a["d"], math.hypot(50, 55), 1e-12) and [f["nom"].rsplit(" ", 1)[1] for f in a["faces"]] == ["−y", "−x"])
a = EL.appui_proche(hea, 0, 100, 220)
chk("boulon (100 ; 220) : devant l'aile supérieure, d = 220 − 145, une face +y de longueur 300",
    not a["coin"] and pres(a["d"], 75, 1e-12) and len(a["faces"]) == 1 and pres(a["faces"][0]["L"], 300, 1e-12))
a = EL.appui_proche(hea, 0, 100, 0)
chk("boulon (100 ; 0) entre les ailes : contre l'âme, d = 100 − 4,25", pres(a["d"], 100 - 4.25, 1e-12) and "ame" in a["face"])
r = EL.normaliser_raid(dict(x=150, y=0, dir="+x", L=80, t=10))
chk("raidisseur +x depuis (150 ; 0), L 80, t 10 : rectangle (150, −5, 230, 5)", EL.rect_raid(r) == (150, -5, 230, 5))
chk("rond : l'emprise contient (0 ; 0) et pas (30 ; 0) pour Ø20",
    EL.dedans(EL.normaliser(dict(element_defaut(TYPE_ROND), d=20)), 0, 0) and not EL.dedans(EL.normaliser(dict(element_defaut(TYPE_ROND), d=20)), 30, 0))
d = CV.disposition_reguliere(defaults())
chk("la disposition par défaut est une grille 2 × 2 de 400 × 400 autour du HEA 300 : vérification complète possible",
    d["ok"] and d["params"]["n_ax"] == 2 and d["params"]["n_ay"] == 2 and pres(d["params"]["w_x"], 400, 1e-9) and d["params"]["prof_c"] == "HEA 300")
u6 = dict(tiges=tiges((-200, -200), (200, -200), (-200, 0), (200, 0), (-200, 200), (200, 200)))
chk("six boulons en 2 files × 3 rangées : grille reconnue (n_ay = 3)", CV.disposition_reguliere(u6)["params"]["n_ay"] == 3)
u_irr = dict(tiges=tiges((-200, -200), (200, -200), (-200, 200), (150, 200)))
chk("un boulon déplacé : pas une grille, raison dite", not CV.disposition_reguliere(u_irr)["ok"] and "grille" in CV.disposition_reguliere(u_irr)["raison"])
chk("un raidisseur : vérification complète impossible, raison dite",
    not CV.disposition_reguliere(dict(raid=[dict(x=150, y=0, dir="+x", L=80, t=10)]))["ok"])
v = CV.vers_manuel(dict(defaults(), mode=MODE_COMPLET, n_ax=2, n_ay=2, w_x=400, w_y=400))
chk("complet → manuel : les quatre boulons de la grille, le profilé à l'axe",
    sorted((t["x"], t["y"]) for t in v["tiges"]) == sorted((x, y) for x, y in ((-200, -200), (200, -200), (-200, 200), (200, 200)))
    and v["elems"][0]["prof"] == "HEA 300" and v["mode"] == MODE_MANUEL)
anc = charger_json({"bp_u": 500, "n_ax": 2})
chk("un fichier d'avant le mode manuel (sans clé « mode ») s'ouvre en vérification complète", anc["mode"] == MODE_COMPLET)
chk("un fichier neuf garde son mode", charger_json({"mode": MODE_MANUEL})["mode"] == MODE_MANUEL)

# ================================================================
print("\n=== 2. Efforts ===")
R = MM.compute()
Rc = MC.compute()
chk("même disposition → même fjd, même bande c, même aire efficace, même kc que la vérification complète",
    pres(R.fjd, Rc.fjd, 1e-12) and pres(R.c, Rc.c, 1e-12) and pres(R.A_eff, Rc.A_eff, 1e-9) and pres(R.k_c, Rc.k_c, 1e-9))
el = EQ.elastique(R.A_eff_rects, [(t["x"], t["y"]) for t in R.tiges], R.k_c, Rc.k_b, 300e3, 50e6, 30e6)
chk("à raideur de tige égale, les efforts des tiges sont ceux de la vérification complète (même solveur)",
    all(pres(a / 1e3, t["T"], 1e-9) for a, t in zip(el["T"], Rc.tiges)))
chk("le mode manuel conclut à l'effet de levier (Lb ≤ Lb* avec le bras diagonal) → kb = 1,6·E·As/Lb",
    R.levier and pres(R.k_b, 1.6 * 210000 * 353 / R.Lb, 1e-9))
Rs = MM.compute(dict(N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0))
chk("suspente (N = −100 kN, rien d'autre) : T = 25 kN sur chaque boulon, aucun contact",
    all(pres(t["T"], 25.0, 1e-9) for t in Rs.tiges) and Rs.C < 1e-9 and Rs.A_c < 1e-9)
Re = MM.compute(dict(elems=[dict(element_defaut(), x=60)], xN=60, Mx_Ed=0, My_Ed=0, Vy_Ed=0))
chk("élément et efforts excentrés de 60 mm : My = N·e = 18 kNm, résultante C = N sous l'élément (xC = 60)",
    pres(Re.My_o, 18.0, 1e-9) and pres(Re.C, 300.0, 1e-9) and pres(Re.xC, 60.0, 1e-6) and all(t["T"] < 1e-9 for t in Re.tiges))
u_irr = dict(tiges=tiges((-215, -180), (210, -200), (-200, 190), (180, 210), (0, -215), (40, 205)),
             elems=[dict(element_defaut(), x=-40, y=20), dict(element_defaut(TYPE_PLAT), b=150, t=12, x=160, y=-60, rot=90)],
             raid=[dict(x=110, y=165, dir="+y", L=60, t=10)], N_Ed=250, Mx_Ed=70, My_Ed=-45, Vx_Ed=30, Vy_Ed=20, T_Ed=5, xN=-40, yN=20)
Ri = MM.compute(u_irr)
a_, bx_, by_ = Ri.axe
Fp, Sx, Sy, _ = EQ._pression(Ri.A_eff_rects, a_, bx_, by_)
rN = Ri.k_c * Fp - sum(t["T"] * 1e3 for t in Ri.tiges) - 250e3
rMx = Ri.k_c * Sy - sum(t["T"] * 1e3 * t["y"] for t in Ri.tiges) - Ri.Mx_o * 1e6
rMy = Ri.k_c * Sx - sum(t["T"] * 1e3 * t["x"] for t in Ri.tiges) - Ri.My_o * 1e6
chk("disposition quelconque (6 boulons, HEA + plat, raidisseur, efforts excentrés) : équilibre exact (N, Mx, My)",
    Ri.eq_ok and abs(rN) < 1e-3 * 250e3 and abs(rMx) < 1e-3 * abs(Ri.Mx_o * 1e6) and abs(rMy) < 1e-3 * abs(Ri.My_o * 1e6),
    f"{rN:.3f} {rMx:.1f} {rMy:.1f}")
Rv = MM.compute(dict(N_Ed=0, Mx_Ed=0, My_Ed=0, Vx_Ed=100, Vy_Ed=0, T_Ed=20))
chk("Vx = 100 kN et T = 20 kNm sur 4 boulons à ±200 : v = 25 ± 12,5 (torsion) → max √(37,5² + 12,5²) = 39,5 kN",
    pres(Rv.V_t_ec3, math.hypot(37.5, 12.5), 1e-9))
Rv2 = MM.compute(dict(N_Ed=0, Mx_Ed=0, My_Ed=0, Vx_Ed=0, Vy_Ed=100, T_Ed=0, xN=200))
chk("V appliqué hors du centre du groupe (xN = 200) : torsion V·e = 20 kNm autour du centre du groupe",
    pres(Rv2.T_G, 20.0, 1e-9))

# ================================================================
print("\n=== 3. Côté tendu, à la main ===")
t1 = Rs.tiges[0]
m = math.hypot(50, 55) - CORD
chk("boulon d'angle : bras = diagonale jusqu'à l'angle − 0,8·a·√2 = 65,3 mm, leff plafonnée à m (modèle d'ingénieur)",
    pres(t1["m"], m, 1e-9) and pres(t1["leff1"], m, 1e-9) and t1["coin"] and any(a.id == "coin" for a in Rs.alerts))
e = (250 - 200) / (55 / math.hypot(50, 55))                      # pince dans la direction du bras : sortie par y
chk("pince perpendiculaire e = 67,6 mm (sortie de la platine dans la direction du bras), e_lat = 67,6, s = 400",
    pres(t1["e_perp"], e, 1e-9) and pres(t1["e_lat"], e, 1e-9) and pres(t1["s_min"], 400, 1e-9))
Mpl1 = 0.25 * m * 25 ** 2 * 345
chk("Mpl,1 = 0,25·leff·tp²·fy = 3,52 kNm ; Lb* = 8,8·m³·As/(leff·tp³) = 847 mm > Lb = 261 → effet de levier",
    pres(t1["Mpl1"], Mpl1 / 1e6, 1e-9) and pres(t1["Lb_star"], 8.8 * m ** 3 * 353 / (m * 25 ** 3), 1e-9) and t1["levier"])
FT1 = 4 * Mpl1 / m
n = min(e, 1.25 * m)
FT3 = min(0.9 * 800 * 353 / 1.25, 353 * 800 / 1.5)
FT2 = (2 * Mpl1 + n * FT3) / (m + n)
chk("FT,Rd = min(4·Mpl/m = 215,6 ; (2·Mpl + n·Ft)/(m + n) = 148,7 ; Ft = 188,3) = 148,7 kN, mode 2",
    pres(t1["FT_Rd"], min(FT1, FT2, FT3) / 1e3, 1e-9) and t1["mode_ts"] == "2")
chk("épaisseur requise du tronçon en T (mode 1 aux faibles épaisseurs, fy = 355) : tp = √(T·m/(leff·fy)) = √(T/fy) = 8,39 mm",
    pres(Rs.plaque["t_T"], math.sqrt(25e3 / 355), 0.005), f"{Rs.plaque['t_T']:.3f}")
sec = {s["nom"].rsplit(" ", 1)[1]: s for s in Rs.sections if "angle" not in s["nom"]}
chk("suspente, boulons d'angle : sections au nu des deux faces de l'angle — aile (b = 300 + 2·46 = 392, M = 2·T·46) "
    "et about d'aile (b = 14 + 2·41 = 96, M = T·41)",
    pres(sec["−y"]["b"], 300 + 2 * (55 - CORD), 1e-9) and pres(sec["−y"]["M"], 2 * 25 * (55 - CORD) / 1e3, 1e-9)
    and pres(sec["−x"]["b"], 14 + 2 * (50 - CORD), 1e-9) and pres(sec["−x"]["M"], 25 * (50 - CORD) / 1e3, 1e-9),
    str({k: (round(s["b"], 1), round(s["M"], 3)) for k, s in sec.items()}))
t_about = math.sqrt(4 * 25e3 * (50 - CORD) / ((14 + 2 * (50 - CORD)) * 355))
chk("l'about d'aile gouverne la suspente : tS = √(4·M/(b·fy)) = 10,97 mm", pres(Rs.plaque["t_S"], t_about, 0.005),
    f"{Rs.plaque['t_S']:.3f} / {t_about:.3f}")
# boulons DEVANT l'aile : (±100 ; ±215) — pince 35 ≥ 1,2·d0 = 31,2
DEVANT = tiges((-100, -215), (100, -215), (-100, 215), (100, 215))
Rf = MM.compute(dict(tiges=DEVANT, N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0))
tf_ = Rf.tiges[0]
m2 = 215 - 145 - CORD
cp = min(math.pi * m2, math.pi * m2 / 2 + 100, math.pi * m2 / 2 + 150)
nc = min(2 * m2 + 0.625 * 35, 150 / 2 + m2 + 0.3125 * 35, 200 / 4 + m2 + 0.3125 * 35)
chk("boulon devant l'aile : m = 215 − 145 − cordon = 61,0 ; e = 35 ; e_lat = 150 ; s = 200 ; leff,1 = min(cp, nc, L/2 + m) = 121,9",
    pres(tf_["m"], m2, 1e-9) and pres(tf_["e_perp"], 35, 1e-9) and pres(tf_["e_lat"], 150, 1e-9) and pres(tf_["s_min"], 200, 1e-9)
    and pres(tf_["leff1"], min(cp, nc, 150 + m2), 1e-9) and not tf_["coin"], f"{tf_['leff1']:.2f} / {min(cp, nc):.2f}")
chk("section au nu de l'aile : deux boulons, b = min(bp ; 300 + 2·m) = 421,9 mm, M = 2·25·61,0 = 3,05 kNm",
    len(Rf.sections) == 2 and all(pres(s["b"], 300 + 2 * m2, 1e-9) and pres(s["M"], 2 * 25e3 * m2 / 1e6, 1e-9) and len(s["idx"]) == 2
                                   for s in Rf.sections), str([(s["nom"], round(s["b"], 1), round(s["M"], 3), s["idx"]) for s in Rf.sections]))
t_sec = math.sqrt(4 * 2 * 25e3 * m2 / ((300 + 2 * m2) * 355))
t_loc = math.sqrt(25e3 * m2 / (min(cp, nc) * 355))
chk("épaisseurs requises à la main : section tS = √(4·M/(b·fy)) = 9,02 mm > tronçon en T tT = √(T·m/(leff·fy)) = 5,93 mm",
    pres(Rf.plaque["t_S"], t_sec, 0.005) and pres(Rf.plaque["t_T"], t_loc, 0.005) and pres(Rf.plaque["t_req"], t_sec, 0.005),
    f"{Rf.plaque['t_S']:.3f}/{t_sec:.3f} {Rf.plaque['t_T']:.3f}/{t_loc:.3f}")
Rf40 = MM.compute(dict(tiges=tiges((-100, -200), (100, -200), (-100, 200), (100, 200)), tp_u=40, N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0))
t0 = Rf40.tiges[0]
chk("platine de 40 : Lb > Lb* → sans effet de levier ; l'épaisseur élastique vaut √1,5 fois la plastique (même charnière 2·leff)",
    not t0["levier"] and pres(t0["t_el"] / t0["t_pl"], math.sqrt(1.5), 1e-9)
    and pres(t0["svm"], math.sqrt((6 * 25e3 * t0["m"] / (2 * t0["leff1"] * 40 ** 2)) ** 2 + 3 * (25e3 / (2 * t0["leff1"] * 40)) ** 2), 1e-9))
Rfe = MM.compute(dict(tiges=DEVANT, N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0, crit=CRIT_ELAS))
chk("critère élastique : la section au nu demande √1,5 × l'épaisseur plastique (τ négligeable ici)",
    pres(Rfe.plaque["t_S"], t_sec * math.sqrt(1.5), 0.01) and Rfe.elastique, f"{Rfe.plaque['t_S']:.3f} / {t_sec * math.sqrt(1.5):.3f}")
Rr = MM.compute(dict(tiges=DEVANT, N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0,
                     raid=[dict(x=130, y=145, dir="+y", L=100, t=10), dict(x=-130, y=145, dir="+y", L=100, t=10),
                           dict(x=130, y=-145, dir="−y", L=100, t=10), dict(x=-130, y=-145, dir="−y", L=100, t=10)]))
m_r = 25 - CORD
b_r = min(500, 100 + 2 * m_r)
chk("raidisseurs de 100 le long des boulons (à 25 mm) : le bras tombe à 25 − cordon = 15,9 mm, l'appui est la face du "
    "raidisseur (L = 100), section b = L + 2m = 132 → tp = √(4·T·m/(b·fy)) = 5,8 mm : l'épaisseur requise baisse",
    pres(Rr.tiges[0]["m"], m_r, 1e-9) and "raidisseur" in Rr.tiges[0]["appui"]["face"] and not Rr.tiges[0]["coin"]
    and pres(Rr.plaque["t_S"], math.sqrt(4 * 25e3 * m_r / (b_r * 355)), 0.005) and Rr.plaque["t_req"] < Rf.plaque["t_req"],
    f"{Rr.plaque['t_req']:.2f} < {Rf.plaque['t_req']:.2f} ; m = {Rr.tiges[0]['m']:.2f} ; {Rr.tiges[0]['appui']['face']}")
Rr2 = MM.compute(dict(tiges=DEVANT, N_Ed=-100, Mx_Ed=0, My_Ed=0, Vy_Ed=0,
                      raid=[dict(x=100, y=145, dir="+y", L=50, t=10), dict(x=-100, y=145, dir="+y", L=50, t=10),
                            dict(x=100, y=-145, dir="−y", L=50, t=10), dict(x=-100, y=-145, dir="−y", L=50, t=10)]))
chk("un boulon juste AU-DELÀ DU BOUT d'un raidisseur de 10 mm n'est tenu que par ce bout : bande étroite b = 10 + 2m = 32 mm, "
    "l'épaisseur requise MONTE (le modèle le dit, c'est voulu)",
    pres(Rr2.tiges[0]["m"], 20 - CORD, 1e-9) and pres(Rr2.pire_section["b"], 10 + 2 * (20 - CORD), 1e-9)
    and Rr2.plaque["t_req"] > Rf.plaque["t_req"], f"{Rr2.plaque['t_req']:.2f} > {Rf.plaque['t_req']:.2f}")

# ================================================================
print("\n=== 4. Côté comprimé ===")
Rn = MM.compute(dict(N_Ed=300, Mx_Ed=0, My_Ed=0, Vy_Ed=0))
p = 300e3 / Rn.A_eff
chk("compression centrée sur la bande c : pression uniforme N/Aeff = 2,32 MPa, bande la plus sollicitée m = p·c²/2 exactement",
    pres(Rn.pmax, p, 1e-9) and pres(Rn.console["m_Ed"], p * Rn.c ** 2 / 2 / 1e3, 1e-6) and all(t["T"] < 1e-9 for t in Rn.tiges),
    f"{Rn.console['m_Ed']:.4f} / {p * Rn.c ** 2 / 2 / 1e3:.4f}")
R2 = MM.compute(dict(N_Ed=2000, Mx_Ed=0, My_Ed=0, Vy_Ed=0))
c_req = (-1763 + math.sqrt(1763 ** 2 + 16 * (2000e3 / R2.fjd - 10627))) / 8      # Aeff(c) = 2(300 + 2c)(14 + 2c) + (8,5 + 2c)(262 − 2c) = N/fjd
t_c = c_req / math.sqrt(345 / (3 * R2.fjd))
chk("N = 2 000 kN : l'épaisseur requise côté comprimé est celle dont la bande c porte N à fjd (c = 45,0 → tp = 18,9 mm)",
    pres(R2.plaque["t_c"], t_c, 0.005) and pres(R2.plaque["t_req"], t_c, 0.005), f"{R2.plaque['t_c']:.3f} / {t_c:.3f}")
chk("sous la pression uniforme, les bandes intérieures (entre les ailes) ne dépassent pas la console c : règle de Voronoï",
    Rn.console["m_Ed"] <= p * Rn.c ** 2 / 2 / 1e3 * (1 + 1e-9))
L = CO.arret(0.0, 131.0, 0.0, -1.0, (-250, -250, 250, 250), [(-4.25, -131, 4.25, 131)])
chk("bande partie de l'aile vers l'âme (x = 0) : elle s'arrête où l'âme devient plus proche — la bande d'x = 0 est "
    "tout de suite dans l'âme (L = 0) ; à x = 60 elle s'arrête à mi-distance", L < 1e-6
    and pres(CO.arret(60.0, 131.0, 0.0, -1.0, (-250, -250, 250, 250), [(-4.25, -131, 4.25, 131)]), 60 - 4.25, 1e-6))
Rp = MM.compute(dict(appui=APPUI_PLEIN))
chk("toute la platine : la console va jusqu'au bord (105 mm au-delà de l'aile), moins de pression mais plus de bras ; "
    "efforts des tiges plus faibles qu'avec la bande c",
    pres(Rp.console["L"], 105, 1e-6) and Rp.console["m_Ed"] > R.console["m_Ed"] and Rp.T_max < R.T_max and any(a.id == "plein" for a in Rp.alerts))

# ================================================================
print("\n=== 5. Trous ===")
Rb = MM.compute(dict(N_Ed=0, Mx_Ed=0, My_Ed=0, Vx_Ed=0, Vy_Ed=200))
ap = Rb.appuis[2]                                                       # boulon (−200 ; 200) : le bord à 50 devant lui
ar = Rb.appuis[0]                                                       # boulon (−200 ; −200) : l'autre boulon à 400 devant
ab = min(50 / (3 * 26), 800 / Rb.fu_p, 1.0)
k1 = min(2.8 * 50 / 26 - 1.7, 1.4 * 400 / 26 - 1.7, 2.5)
chk("Vy = 200 kN sur 4 boulons : 50 kN chacun ; boulon vers le bord : e1 = 50, p1 = ∞, e2 = 50, p2 = 400 → αb = 0,64, k1 = 2,5 ; "
    "boulon arrière : e1 = 450, p1 = 400 → αb = 1",
    pres(ap["Fv"], 50, 1e-9) and pres(ap["e1"], 50, 1e-9) and ap["p1"] == float("inf") and pres(ap["e2"], 50, 1e-9)
    and pres(ap["p2"], 400, 1e-9) and pres(ap["ab"], ab, 1e-9) and pres(ap["k1"], k1, 1e-9)
    and pres(ar["e1"], 450, 1e-9) and pres(ar["p1"], 400, 1e-9) and pres(ar["ab"], 1.0, 1e-9),
    str([(a["i"], round(a["e1"]), a["p1"], round(a["e2"]), a["p2"], round(a["ab"], 3), round(a["k1"], 2)) for a in Rb.appuis]))
chk("Fb,Rd = k1·αb·fu·d·tp/γM2 et l'épaisseur qu'il faut : tb = Fv·γM2/(k1·αb·fu·d)",
    pres(ap["Fb"], k1 * ab * Rb.fu_p * 24 * 25 / 1.25 / 1e3, 1e-9) and pres(ap["t_b"], 50e3 * 1.25 / (k1 * ab * Rb.fu_p * 24), 1e-9)
    and Rb.ck["pb"].active and pres(Rb.ck["pb"].Rd, ap["Fb"], 1e-9))
chk("avec N = 300 kN, le frottement Cf·N = 60 kN absorbe Vy = 40 kN : pas de pression diamétrale (EN 1993-1-8 §6.2.2)",
    not R.ck["pb"].active and pres(R.Ff_Rd / 1e3, 60, 1e-9))

# ================================================================
print("\n=== 6. Intuitions ===")
Rrhs = MM.compute(dict(elems=[dict(element_defaut(TYPE_RHS), h=300, b=300, t=10)]))
Rrond = MM.compute(dict(elems=[dict(element_defaut(TYPE_ROND), d=20)]))
chk("un rond Ø20 à l'axe : bras de 264 mm au lieu de 62, la platine travaille bien plus que sur un RHS 300 "
    "(épaisseur requise infinie : la pression dépasse fjd, la console radiale ne tient pas)",
    Rrond.tiges[0]["m"] > 4 * Rrhs.tiges[0]["m"] and Rrond.plaque["t_T"] > Rrhs.plaque["t_T"]
    and Rrond.plaque["t_req"] == float("inf") and not Rrond.verified and any(a.id == "fjd_info" for a in Rrond.alerts))
chk("RHS 300×300×10 et HEA 300 à la même place : bras plus court sur le RHS (angle à 150 ; 150), épaisseur requise plus faible",
    Rrhs.tiges[0]["m"] < R.tiges[0]["m"] and Rrhs.plaque["t_req"] < R.plaque["t_req"])
Rchs = MM.compute(dict(elems=[dict(element_defaut(TYPE_CHS), d=219.1, t=8)]))
chk("CHS 219,1×8 : bras radial 200·√2 − 109,6 − cordon, aire d'appui = anneau élargi de c découpé par la platine",
    pres(Rchs.tiges[0]["m"], 200 * math.sqrt(2) - 219.1 / 2 - CORD, 1e-9) and Rchs.tiges[0]["cercle"]
    and pres(Rchs.A_eff, math.pi * ((219.1 / 2 + Rchs.c) ** 2 - (219.1 / 2 - 8 - Rchs.c) ** 2), 1e-6))
chk("le verdict suit l'épaisseur requise : t_req ≤ tp ⟺ VÉRIFIÉE (défaut, RHS, suspente, rond)",
    all((X.plaque["t_req"] <= X.u.tp_u + 1e-9) == X.verified for X in (R, Rrhs, Rs, Rrond, Rf)))
Rel = MM.compute(dict(crit=CRIT_ELAS))
chk("critère élastique sur le cas par défaut : plus d'épaisseur qu'en EN 1993-1-8 (l'effet de levier n'est pas compté)",
    Rel.plaque["t_req"] > R.plaque["t_req"] and Rel.ck["ptl"].unit == "MPa")
chk("le point d'application par défaut est l'axe de la platine ; à l'axe, aucun moment ajouté",
    pres(R.Mx_o, 50, 1e-12) and pres(R.My_o, 30, 1e-12) and not any(a.id == "exc" for a in R.alerts))

# ================================================================
print("\n=== 7. Validité et temps ===")
Rx = MM.compute(dict(tiges=tiges((0, 0), (200, -200), (-200, 200), (200, 200))))
chk("boulon dans l'emprise de l'élément : bloquant, non vérifié", any(a.id == "bin_0" and a.block for a in Rx.alerts) and not Rx.verified)
Rx = MM.compute(dict(tiges=tiges((-235, -200), (200, -200), (-200, 200), (200, 200))))
chk("pince 15 mm < 1,2·d0 = 31,2 : Tableau 3.3 non conforme, bloquant",
    any(a.id == "dist_e" and a.block for a in Rx.alerts) and not Rx.dist_ok and not Rx.verified)
Rx = MM.compute(dict(tiges=tiges((-200, -200), (-160, -200), (-200, 200), (200, 200))))
chk("entraxe 40 mm < 2,2·d0 = 57,2 : bloquant", any(a.id == "dist_p" and a.block for a in Rx.alerts) and not Rx.verified)
Rx = MM.compute(dict(elems=[dict(element_defaut(), x=150)]))
chk("élément hors de la platine : bloquant", any(a.id == "hors_0" and a.block for a in Rx.alerts) and not Rx.verified)
Rx = MM.compute(dict(tiges=[]))
chk("aucun boulon : bloquant, épaisseur requise infinie", any(a.id == "aucun_b" for a in Rx.alerts) and Rx.plaque["t_req"] == float("inf"))
Rx = MM.compute(dict(tiges=tiges((-100, 200), (100, 200)), N_Ed=20, Mx_Ed=80, My_Ed=0, Vy_Ed=0))
chk("deux boulons du côté comprimé seulement et un moment qui soulève l'autre côté : équilibre introuvable → bloquant",
    not Rx.eq_ok and any(a.id == "eq" and a.block for a in Rx.alerts) and not Rx.verified)
Rx = MM.compute(dict(tiges=tiges((-200, 0), (200, 0)), N_Ed=20, Mx_Ed=80, My_Ed=0, Vy_Ed=0))
chk("deux boulons SUR l'axe du moment : ils tirent tous deux autant, contre la zone comprimée (équilibre trouvé, tiges "
    "très tendues → information)", Rx.eq_ok and pres(Rx.tiges[0]["T"], Rx.tiges[1]["T"], 1e-6) and Rx.tiges[0]["T"] > 100
    and any(a.id == "tige_info" for a in Rx.alerts))
Rx = MM.compute(dict(tiges=tiges((-200, 0), (200, 0)), N_Ed=20, Mx_Ed=0, My_Ed=40, Vy_Ed=0))
chk("les mêmes deux boulons avec My : le boulon du côté soulevé reprend, l'autre est détendu", Rx.eq_ok and Rx.tiges[0]["T"] > 1
    and Rx.tiges[1]["T"] < 1e-9)
t0 = time.time()
MM.compute()
dt = time.time() - t0
chk("un calcul manuel complet (répartition recalculée à chaque épaisseur du balayage) en moins de 1,5 s", dt < 1.5, f"{dt:.2f} s")

# ================================================================
print("\n=== 8. Benchmark de l'écran ===")
# L'onglet Benchmark rejoue des calculs à la main (les mêmes qu'ici) et affiche « VALIDÉ » ou « À CONTRÔLER ».
# Le 31/10/2026, le harnais d'écran a trouvé VAL-M4 « À CONTRÔLER » : sa valeur à la main prenait fu = 490 MPa
# au lieu des 510 MPa de la table des aciers (S355, EN 1993-1-1 Tab. 3.1) — Fb,Rd 4 % trop bas. Ce contrôle
# rougit si une valeur à la main du benchmark s'écarte de plus de 0,5 % du moteur.
from acier.assemblages.pied_poteau.platine_beton.manuel.benchmark import run_bench_manuel  # noqa: E402
B = run_bench_manuel()
chk("le benchmark de l'onglet (VAL-M1 à VAL-M4) est VALIDÉ : chaque ligne à 0,5 % du calcul à la main",
    B["ok"] and B["statut"] == "VALIDÉ (calculs manuels)"
    and [c["id"] for c in B["cas"]] == ["VAL-M1", "VAL-M2", "VAL-M3", "VAL-M4"],
    str([(c["id"], r["lab"], round(r["ec"] * 100, 2)) for c in B["cas"] for r in c["rows"] if not r["ok"]]))
chk("VAL-M4 : Fb,Rd = 2,5 × 0,641 × 510 × 24 × 25 / 1,25 = 392,3 kN ; tb = 3,19 mm",
    pres(B["cas"][3]["rows"][3]["val"], 392.3, 0.05) and pres(B["cas"][3]["rows"][4]["val"], 3.19, 0.005))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

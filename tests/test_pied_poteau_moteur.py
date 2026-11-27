# -*- coding: utf-8 -*-
"""Tests du MOTEUR pied de poteau (lot A) — chaque garantie rougit si on la retire :

  1. GÉOMÉTRIE sans maillage : union exacte de rectangles, découpage par un
     demi-plan, intégrales de Green contre les formules analytiques ; aire
     efficace du HEA 300 calculée à la main.
  2. ÉQUILIBRE élastique : compression pure → pression uniforme ; résidus
     d'équilibre nuls ; reproduction de la répartition de l'étude (script
     indépendant) ; soulèvement pur → quatre tiges égales.
  3. DOMAINE PLASTIQUE : convexité de la frontière ; niveau 1 ≤ niveau 2 sur
     chaque axe et en taux ; capacités uniaxiales de niveau 1 à la main
     (modèle Table 6.7) ; N seul retrouve fjd·Aeff.
  4. COMPOSANTS à la main : fjd, c, Aeff, NRd, tronçon en T de coin (Lb*,
     mode sans effet de levier), tiges acier (EC3 et EN 1992-4), cône,
     arrachement, bord, pry-out, cordons, borne de la platine comprimée.
  5. CONFIGURATIONS : RHS à six tiges (classes ext / coin), tiges entre les
     ailes (int), géométries invalides (tige dans le vide du tube, platine
     hors massif, pince trop courte), mortier hors conditions.
  6. BENCHMARK : VAL-A et VAL-B au vert ; temps de calcul.

Lancement : python3 tests/test_pied_poteau_moteur.py (depuis la racine).
"""
import math
import os
import random
import sys
import time

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.pied_poteau.platine_beton import geometrie as GE  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton import equilibre as EQ  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton import composants as CP  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.moteur import compute  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.benchmark import run_bench  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.entrees import (FORME_RHS, TYPE_CROSSE, TYPE_ETA, NON,  # noqa: E402
                                                                  PERIM_TOUT)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


# ================================================================
print("=== 1. Géométrie ===")
U = GE.union_rectangles([(0, 0, 10, 10), (5, 5, 15, 15), (20, 0, 30, 5)])
chk("union de rectangles : aire exacte (100 + 100 − 25 + 50)", pres(GE.aire_rects(U), 225, 1e-12)
    and all(GE.intersection(a, b) is None for i, a in enumerate(U) for b in U[i + 1:]), str(U))
carre = GE.rect(-1, -1, 1, 1)
moitie = GE.clip(carre, 1, 1, 0)
chk("découpage d'un carré par sa diagonale : moitié de l'aire", pres(GE.aire(moitie), 2.0, 1e-12))
A, Sx, Sy, Ixx, Ixy, Iyy = GE.moments(GE.rect(1, 2, 4, 6))
chk("intégrales de Green sur un rectangle 3 × 4 en (1..4, 2..6)",
    pres(A, 12, 1e-12) and pres(Sx, 12 * 2.5, 1e-12) and pres(Sy, 12 * 4, 1e-12)
    and pres(Ixx, 4 * (4 ** 3 - 1) / 3, 1e-12) and pres(Iyy, 3 * (6 ** 3 - 2 ** 3) / 3, 1e-12) and pres(Ixy, 12 * 2.5 * 4, 1e-12))
c = 25 * math.sqrt(345 / (3 * 20.4))
rects = GE.aire_efficace(GE.parois_I(290, 300, 8.5, 14), c, (-250, -250, 250, 250))
A_main = 2 * (300 + 2 * c) * (14 + 2 * c) + (8.5 + 2 * c) * (290 - 28 - 2 * c)
chk("aire efficace du HEA 300 (c = 59,36) = 1 293,7 cm², à la main", pres(GE.aire_rects(rects), A_main, 1e-9))
rects_c = GE.aire_efficace(GE.parois_I(290, 300, 8.5, 14), c, (-150, -150, 150, 150))
chk("…découpée par une platine 300 × 300", GE.aire_rects(rects_c) < A_main and pres(GE.aire_rects(rects_c), 300 * 300 - 0, 0.5)
    and all(-150 - 1e-9 <= r[0] and r[2] <= 150 + 1e-9 for r in rects_c))
_r, A_t, bras = GE.tronçon_comprime(GE.parois_I(290, 300, 8.5, 14), c, (-250, -250, 250, 250), (0, 1), "I")
chk("tronçon comprimé +y : bande d'aile, bras au centre de l'aile (138 mm)", pres(A_t, (300 + 2 * c) * (14 + 2 * c), 1e-9)
    and pres(bras, 138, 1e-9))
_r, A_ty, bras_y = GE.tronçon_comprime(GE.parois_I(290, 300, 8.5, 14), c, (-250, -250, 250, 250), (1, 0), "I")
chk("tronçon comprimé +x : demi-bandes des deux ailes, bras (b/2 + c)/2", pres(A_ty, 2 * (150 + c) * (14 + 2 * c), 1e-9)
    and pres(bras_y, (150 + c) / 2, 1e-9))
pos = GE.positions_tiges(2, 3, 400, 360)
chk("grille 2 × 3 : six tiges, rangées à −180 / 0 / 180", len(pos) == 6 and sorted({p[1] for p in pos}) == [-180, 0, 180]
    and sorted({p[0] for p in pos}) == [-200, 200])
dims = dict(h=290, b=300, tw=8.5, tf=14, r=27)
chk("classement : coin / ext / int / lat", GE.classer_tige(200, 200, "I", dims, 8)["classe"] == "coin"
    and GE.classer_tige(100, 200, "I", dims, 8)["classe"] == "ext"
    and GE.classer_tige(60, 0, "I", dims, 8)["classe"] == "int"
    and GE.classer_tige(200, 0, "I", dims, 8)["classe"] == "lat"
    and GE.classer_tige(50, 50, "RHS", dict(h=250, b=250, t=10), 6)["classe"] == "dedans")

# ================================================================
print("\n=== 2. Équilibre élastique ===")
tiges = [(-200, -200), (200, -200), (-200, 200), (200, 200)]
el = EQ.elastique(rects, tiges, 100.0, 5e5, 1e6, 0.0, 0.0)
chk("compression pure : pression uniforme N/Aeff, aucune tige tendue",
    el["ok"] and pres(el["pmax"], 1e6 / A_main, 1e-9) and max(el["T"]) == 0 and pres(el["C"], 1e6, 1e-9))
el = EQ.elastique(rects, tiges, 127.0, 568e3, 300e3, 50e6, 30e6)
res = (el["C"] - sum(el["T"]) - 300e3, el["C"] * el["yC"] - sum(t * y for (x, y), t in zip(tiges, el["T"])) - 50e6,
       el["C"] * el["xC"] - sum(t * x for (x, y), t in zip(tiges, el["T"])) - 30e6)
chk("N + Mx + My : résidus d'équilibre < 1e-6 relatif", el["ok"] and abs(res[0]) < 1 and abs(res[1]) < 1e2 and abs(res[2]) < 1e2, str(res))
R = compute(dict(bc_u=3000, lc_u=3000, hc_u=2000))
chk("reproduction du script de l'étude (massif large) : fjd 34, c 46, pmax 14,9, C 349,6 en (57 ; 115), T 49,6",
    pres(R.fjd, 34.0, 1e-9) and pres(R.c, 46.0, 0.002) and pres(R.pmax, 14.90, 0.002) and pres(R.C, 349.6, 0.002)
    and pres(R.xC, 57, 0.02) and pres(R.yC, 115, 0.01) and pres(R.tiges[0]["T"], 49.6, 0.003)
    and sum(t["T"] for t in R.tiges[1:]) < 1e-6, f"{R.fjd} {R.c} {R.pmax} {R.C} {R.xC} {R.yC} {[t['T'] for t in R.tiges]}")
Ru = compute(dict(N_Ed=-200, Mx_Ed=0, My_Ed=0, Vy_Ed=0))
chk("soulèvement pur : quatre tiges à 50 kN, aucun contact, pression nulle",
    Ru.eq_ok and all(pres(t["T"], 50, 1e-6) for t in Ru.tiges) and Ru.A_c < 1e-6 and Ru.pmax < 1e-9)

# ================================================================
print("\n=== 3. Domaine plastique, niveau 1 / niveau 2 ===")
R0 = compute()


def convexe(pts):
    n = len(pts)
    signes = set()
    for i in range(n):
        (x0, y0), (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cr = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(cr) > 1e-6 * (abs(x1) + abs(y1) + 1):
            signes.add(cr > 0)
    return len(signes) == 1


chk("la frontière du domaine (120 points) est convexe", len(R0.dom) >= 100 and convexe(R0.dom), str(len(R0.dom)))
chk("niveau 1 ≤ niveau 2 sur les quatre demi-axes, et taux niveau 1 ≥ taux niveau 2",
    all(R0.MRd1[k] <= R0.MRd2[k] + 1e-6 for k in R0.MRd1) and R0.eta1 >= R0.eta2)
FT = R0.rangees[0]["FT_tige"] / 1e3
M1x = ((300 + 2 * FT) * 138 + 2 * FT * 200) / 1e3
M1y = ((300 + 2 * FT) * (150 + R0.c) / 2 + 2 * FT * 200) / 1e3
chk("niveau 1 à la main : Mx,Rd = (N + ΣT)·zC + ΣT·zT = 160,4 kNm ; My,Rd = 138,6 kNm",
    pres(R0.MRd1["x+"], M1x, 1e-6) and pres(R0.MRd1["y+"], M1y, 1e-6) and pres(M1x, 160.36, 0.001),
    f"{R0.MRd1} {M1x} {M1y}")
rng = random.Random(7)
ok_conv = True
for _ in range(4):
    # entraxes ≤ 430 : pince ≥ 35 mm ≥ 1,2·d0 (une géométrie invalide rend des capacités nulles, à dessein) ;
    # wy ≥ 340 : à wy = 300 les tiges sont à 5 mm de l'aile du HEA 300, trou DANS le cordon — refusé depuis le lot de
    # clôture (R13, alerte « tige_acier »), contrôlé en section 8
    Rr = compute(dict(w_x=rng.choice([300, 380, 430]), w_y=rng.choice([340, 380, 430]), n_ay=rng.choice([2, 3, 4]),
                      N_Ed=rng.choice([0, 300, 1200]), Mx_Ed=rng.choice([20, 80]), My_Ed=rng.choice([0, 40, 80]),
                      bc_u=1200, lc_u=1200))
    ok_conv &= bool(Rr.dom) and convexe(Rr.dom) and all(Rr.MRd1[k] <= Rr.MRd2[k] + 1e-6 for k in Rr.MRd1) and Rr.eta1 >= Rr.eta2 - 1e-9
chk("quatre configurations tirées au sort : convexité et niveau 1 ≤ niveau 2", ok_conv)
Rn = compute(dict(Mx_Ed=0, My_Ed=0, N_Ed=2000))
chk("N seul : les deux niveaux rendent N/(fjd·Aeff)", pres(Rn.eta1, 2000 / (Rn.N_Rd / 1e3), 1e-9) and Rn.eta1 == Rn.eta2
    and Rn.ck["ncp"].active and pres(Rn.ck["ncp"].eta, Rn.eta1, 1e-9))
Rb = compute(dict(N_Ed=5000, Mx_Ed=10))
chk("compression dominante (N > 2·FC,Rd) : niveau 1 non applicable → alerte, le niveau 2 renseigne",
    any(a.id == "comp_dom" for a in Rb.alerts) and Rb.MRd1["x+"] == 0 and not Rb.ck["fc1"].ok)

# ================================================================
print("\n=== 4. Composants à la main ===")
fcd = 0.85 * 30 / 1.5
fjd = 2 / 3 * fcd * 1.8
chk("fjd = 20,40 MPa (Ac0 = platine, Ac1 = massif 900²), fcd = 17", pres(R0.fjd, fjd, 1e-12) and pres(R0.fcd, fcd, 1e-12))
chk("c = 59,36 mm et NRd = fjd·Aeff = 2 639 kN", pres(R0.c, c, 1e-12) and pres(R0.N_Rd / 1e3, fjd * A_main / 1e3, 1e-9))
rg = R0.rangees[0]
m = 55 - 0.8 * 8 * math.sqrt(2)
Mpl = 0.25 * 150 * 625 * 345
chk("tronçon en T de coin : m = 45,95, Σleff,1 = 150 (plafond 0,25·b), Lb > Lb* → FT = 2·Mpl/m = 352 kN",
    pres(rg["m"], m, 1e-9) and pres(rg["leff1"], 150, 1e-9) and not rg["levier"]
    and pres(rg["FT_Rd"], 2 * Mpl / m, 1e-9) and rg["mode"].startswith("1-2"),
    f"{rg['m']} {rg['leff1']} {rg['levier']} {rg['FT_Rd']}")
Re = compute(dict(w_x=200))
rg_e = Re.rangees[0]
leff_row = 2 * min(min(math.pi * rg_e["m"], 0.5 * math.pi * rg_e["m"] + 100, 0.5 * math.pi * rg_e["m"] + 150),
                   min(2 * rg_e["m"] + 0.625 * 50, 75 + rg_e["m"] + 0.3125 * 50, 50 + rg_e["m"] + 0.3125 * 50))
chk("tiges dans la largeur d'aile (ext) : Σleff,1 = 2·min(motifs par tige) — Tableau 6.6 retrouvé",
    all(t["classe"] == "ext" for t in Re.tiges) and pres(rg_e["leff1"], leff_row, 1e-9), f"{rg_e['leff1']} {leff_row}")
chk("tiges : Ft,Rd = 203,3 kN (EC3), NRd,s = 188,3 kN (EN 1992-4, γMs = 1,5), F2,vb,Rd = 56,0 kN",
    pres(R0.Ft_EC3 / 1e3, 0.9 * 800 * 353 / 1.25 / 1e3, 1e-9) and pres(R0.N_Rd_s / 1e3, 353 * 800 / 1.5 / 1e3, 1e-9)
    and pres(R0.F2vb / 1e3, 0.248 * 800 * 353 / 1.25 / 1e3, 1e-9))
N0 = 7.7 * math.sqrt(30) * 300 ** 1.5
chk("cône (une tige tendue, c = 250) = 76,6 kN", pres(R0.ck["cone"].Rd, N0 * 490000 / 810000 * (0.7 + 0.3 * 250 / 450) / 1.5 / 1e3, 1e-9))
chk("arrachement de la tête = 120,6 kN", pres(R0.ck["pull"].Rd, 7.5 * math.pi / 4 * (1600 - 576) * 30 / 1.5 / 1e3, 1e-9))
lf = 288
al_, be_ = 0.1 * math.sqrt(lf / 250), 0.1 * (24 / 250) ** 0.2
V0 = 1.7 * 24 ** al_ * lf ** be_ * math.sqrt(30) * 250 ** 1.5
chk("bord (c1 = 250, rangée de 2) = 53,1 kN", pres(R0.ck["edge"].Rd, V0 * 1.2 * 0.9 / 1.5 / 1e3, 1e-9))
chk("pry-out = 2·NRk,c(groupe)/γMc = 253,2 kN", pres(R0.ck["pry"].Rd, 2 * N0 * (0.7 + 0.3 * 250 / 450) / 1.5 / 1e3, 1e-9))
Rt = compute(dict(perim=PERIM_TOUT))
chk("périmètre complet : l'interaction béton βN^1,5 + βV^1,5 gouverne le cas par défaut (massif 900), non vérifié",
    Rt.gov.key == "cv" and not Rt.verified and Rt.ck["edge"].eta > 0.7 and "cv" not in Rt.info_keys)
chk("périmètre EC3 (défaut) : le béton d'ancrage est en information — même calcul, hors verdict, alerte « ancr_info »",
    R0.perim_ec3 and R0.verified and R0.ck["cv"].eta == Rt.ck["cv"].eta and "cv" in R0.info_keys
    and R0.gov.key == "tpq" and any(a.id == "ancr_info" for a in R0.alerts)
    and not any(a.id == "ancr_info" for a in Rt.alerts), f"{R0.gov.key} {R0.verified}")
P = R0.plaque
chk("épaisseur requise : max des quatre critères, console comprimée dimensionnante (mEd = 11,5 kNm/m sur c = 59)",
    P["t_req"] == max(P["t_c"], P["t_N"], P["t_M"], P["t_T"]) and P["t_c"] == P["t_req"]
    and pres(P["m_Ed"], R0.ck["pc"].Ed, 1e-12) and P["ligne"] == "y = +h/2" and 13.5 < P["t_c"] < 14.5
    and pres(R0.ck["tpq"].Ed, P["t_req"], 1e-12) and R0.ck["tpq"].Rd == 25.0, str(P))
Tm = P["tendu"]
chk("tronçon en T : moment pris M = T·m (43,2 kN × 45,9 mm), épaisseur √(2·T·m·γM0/(Σleff,1·fy)) = 8,8 mm",
    pres(Tm["M"], Tm["T"] * Tm["m"] / 1e3, 1e-12) and pres(Tm["T"], R0.T_max, 1e-9)
    and pres(Tm["t_formule"], math.sqrt(2 * Tm["T"] * 1e3 * Tm["m"] / (150 * 345)), 1e-9) and 8 < P["t_T"] < 9.5)
chk("Navier sur toute la platine : σ = 1,2 ± 2,4 ± 1,44 MPa → soulèvement, repère sans objet",
    pres(P["navier"]["sN"], 1.2, 1e-12) and pres(P["navier"]["sMx"], 2.4, 1e-12) and pres(P["navier"]["sMy"], 1.44, 1e-12)
    and not P["navier"]["applicable"])
Rn = compute(dict(Mx_Ed=0, My_Ed=0, Vy_Ed=0))
chk("N seul : Navier applicable (σ = 1,2 MPa uniforme, L = 105), t = √(6·σ·L²/2·γM0/fy) ; EC3 moins exigeant",
    Rn.plaque["navier"]["applicable"] and pres(Rn.plaque["navier"]["t"], math.sqrt(6 * 1.2 * 105 ** 2 / 2 / 345), 1e-9)
    and Rn.plaque["t_req"] < Rn.plaque["navier"]["t"])
for cas in (dict(), dict(tp_u=10), dict(tp_u=15, N_Ed=600, Mx_Ed=120), dict(N_Ed=1200, Mx_Ed=200), dict(tp_u=90, N_Ed=1200, Mx_Ed=200)):
    Rx = compute(cas)
    ok4 = all(Rx.ck[k].ok for k in ("pc", "ncp", "fc1", "pt") if Rx.ck[k].active)
    if Rx.ck["tpq"].ok != ok4:
        chk("tp ≥ épaisseur requise ⟺ les vérifications de la platine passent", False, str(cas))
        break
else:
    chk("tp ≥ épaisseur requise ⟺ les vérifications de la platine passent (5 cas, dont platine très épaisse : "
        "niveau 1 non monotone, t_M = ∞)", compute(dict(tp_u=90, N_Ed=1200, Mx_Ed=200)).plaque["t_M"] == float("inf"))
w = R0.cordons
chk("cordons : σ⊥ = σn/√2, σeq = √(2σn² + 3τ∥²), limites fu/(βw·γM2) et 0,9·fu/γM2",
    pres(w["s_perp"], w["sig_n"] / math.sqrt(2), 1e-9) and pres(w["s_eq"], math.sqrt(2 * w["sig_n"] ** 2 + 3 * w["tau_par"] ** 2), 1e-9)
    and pres(w["f_eq"], R0.fu_w / (R0.bw_w * 1.25), 1e-9) and pres(w["f_perp"], 0.9 * R0.fu_w / 1.25, 1e-9))
chk("platine comprimée : le taux du débord ne dépasse jamais pmax/fjd (la bande c est élastique par construction)",
    R0.ck["pc"].eta <= R0.ck["bet"].eta + 1e-9 and R.ck["pc"].eta <= R.ck["bet"].eta + 1e-9)
Rc = compute(dict(typ_a=TYPE_CROSSE))
fbd = 2.25 * 0.7 * 0.7 * 0.30 * 30 ** (2 / 3) / 1.5
chk("tige à crosse : Ft,bond = π·d·hef·fbd (fbd = 2,25·0,7·fctd)", pres(Rc.Ft_bond, math.pi * 24 * 300 * fbd, 1e-9)
    and Rc.ck["pull"].label.startswith("Adhérence"))
chk("rigidité : Sj,ini > 0, seuil 30·E·Ic/Lc, pied semi-rigide pour tp = 25 et Lc = 4 m",
    R0.Sj > 0 and R0.rig is not None and not R0.rig["rigide"] and pres(R0.Sj_seuil, 30 * 210000 * R0.I_c / 4000 / 1e6, 1e-9))

# ================================================================
print("\n=== 5. Configurations ===")
Rr = compute(dict(forme=FORME_RHS, prof_c="Personnalisé", h_u=250, b_u=250, t_u=10, bp_u=450, lp_u=450, tp_u=20,
                  n_ay=3, w_x=350, w_y=360, N_Ed=-150, Mx_Ed=40, My_Ed=0, Vy_Ed=0, bc_u=1200, lc_u=1200))
chk("RHS 250 × 250 × 10, six tiges : coins en haut et en bas, ext au milieu ; soulèvement avec moment",
    Rr.forme == "RHS" and sorted(t["classe"] for t in Rr.tiges) == ["coin"] * 4 + ["ext"] * 2 and Rr.eq_ok
    and sum(1 for t in Rr.tiges if t["T"] > 1e-6) >= 4 and Rr.ck["fc1"].active, str([t["classe"] for t in Rr.tiges]))
Ri = compute(dict(w_x=120, w_y=180))
chk("tiges entre les ailes : classe int, alerte « int », tronçon en T non raidi", all(t["classe"] == "int" for t in Ri.tiges)
    and any(a.id == "int" for a in Ri.alerts) and Ri.rangees and Ri.rangees[0]["classe"] == "int")
Rd = compute(dict(forme=FORME_RHS, prof_c="Personnalisé", h_u=250, b_u=250, t_u=10, w_x=100, w_y=100))
chk("tige dans le vide du tube → géométrie invalide (alerte bloquante)", any(a.id == "dedans" and a.block for a in Rd.alerts)
    and not Rd.geo_ok and not Rd.verified)
Rm = compute(dict(bc_u=400))
chk("platine hors massif → alerte bloquante", any(a.id == "massif" and a.block for a in Rm.alerts) and not Rm.verified)
Rp = compute(dict(w_x=460))
chk("pince des tiges < 1,2·d0 → alerte dist_e, Tableau 3.3 KO", any(a.id == "dist_e" for a in Rp.alerts) and not Rp.dist_ok)
Rg = compute(dict(t_g=120))
chk("mortier trop épais (> 0,2·min(bp, lp)) → alerte bloquante « mortier »", any(a.id == "mortier" and a.block for a in Rg.alerts))
Rok = compute(dict(bc_u=1400, lc_u=1400, hc_u=800))
chk("massif 1 400 × 1 400 : le pied par défaut est vérifié (bord et cône desserrés)", Rok.verified, f"{Rok.gov.key} {Rok.eta_max}")
chk("mêmes saisies → mêmes résultats (déterminisme)", compute().eta_max == R0.eta_max and compute().ck["edge"].Rd == R0.ck["edge"].Rd)

Rz = compute(dict(ancr="M20", N_Ed=-400, Mx_Ed=120, Vy_Ed=50))
chk("tige déjà à sa résistance en traction avec bras de levier (MRk,s = 0 → VRd,s = 0) : pas de division par zéro, "
    "cisaillement et interaction hors limite (défaut trouvé par la validation croisée du lot C)",
    Rz.VRd_s == 0.0 and Rz.ck["tv"].eta >= 1e8 and Rz.ck["tvn"].eta >= 1e8 and not Rz.verified)
Re_ = compute(dict(typ_a=TYPE_ETA, eta_tau=12, eta_k1=8.5, perim=PERIM_TOUT))
ad = Re_.pull
chk("tige post-installée (ETA) : arrachement combiné §7.2.1.6 — N⁰Rk,p = π·d·hef·τRk, scr,Np = 7,3·d·√τRk ≤ 3·hef, "
    "ψg = 1 ; k1 de l'ETA dans le cône ; la tige seule bornée par π·d·hef·τRk/γMc",
    Re_.eta and Re_.ck["pull"].label.startswith("Arrachement combiné") and ad["actif"]
    and pres(ad["N0"], math.pi * 24 * 300 * 12, 1e-9) and pres(ad["scr"], min(7.3 * 24 * math.sqrt(12), 900), 1e-9)
    and ad["psi_g"] == 1.0 and pres(Re_.cone["k1"], 8.5, 1e-12)
    and pres(Re_.Ft_bond, math.pi * 24 * 300 * 12 / 1.5, 1e-9) and Re_.Ft_anc <= Re_.Ft_bond + 1e-9
    and any(a.id == "eta_info" for a in Re_.alerts))
Re2 = compute(dict(typ_a=TYPE_ETA, eta_tau=12, eta_nrks=150, eta_vrks=80))
chk("ETA : NRk,s et VRk,s saisis remplacent les valeurs calculées (mêmes γMs)",
    pres(Re2.N_Rd_s, 150e3 / Re2.g_Ms_N, 1e-9) and pres(Re2.VRd_s0, 80e3 / Re2.g_Ms_V, 1e-9))
chk("sans ETA : le cône garde k1 = 7,7 (fissuré) et l'arrachement de la tête", pres(R0.cone["k1"], 7.7, 1e-12)
    and R0.ck["pull"].label.startswith("Arrachement de la tête"))
from acier.assemblages.pied_poteau.platine_beton import benchmark as BM  # noqa: E402
chk("fiches d'exemples publiés : lues, COMPLÈTES depuis le lot de clôture → cinq rejouées (section 8), aucune « à "
    "compléter » ; validation croisée datée",
    len(BM.fiches_publiees()) == 5 and all(f["complet"] for f in BM.fiches_publiees())
    and [x["id"] for x in BM.BENCH] == ["BM1", "BM2", "BM3", "BM4", "BM5"]
    and BM.VALIDATION_CROISEE["configurations"] == 40 and BM.VALIDATION_CROISEE["date"] == "31/10/2026")
chk("une fiche complète se rejoue : la lecture des grandeurs du moteur (fjd, ck:, FC_Rd:, MRd1:, rangee:)",
    pres(BM._valeur(R0, "fjd"), R0.fjd, 1e-12) and pres(BM._valeur(R0, "ck:ncp"), R0.ck["ncp"].Rd, 1e-12)
    and pres(BM._valeur(R0, "FC_Rd:x+"), R0.FC_Rd["x+"] / 1e3, 1e-12) and pres(BM._valeur(R0, "MRd1:x+"), R0.MRd1["x+"], 1e-12)
    and pres(BM._valeur(R0, "rangee:0:FT_Rd"), R0.rangees[0]["FT_Rd"] / 1e3, 1e-12))
# R2 : les écarts affichés dans l'onglet Benchmark sont une COPIE NOMMÉE du rapport daté ; ce contrôle les lie.
import re  # noqa: E402
VC = BM.VALIDATION_CROISEE
rap = open(VC["rapport"], encoding="utf-8").read()
mesures = re.findall(r"^\| [^|]+ \| ([0-9,]+) (%|mm|‰) \| [^|]+ \|$", rap, re.M)


def _lie(copie, mesure):
    """La copie (« 0,012 % ») est la mesure (« 0,0121 », « % ») arrondie à la précision de la copie."""
    val, unite = copie.rsplit(" ", 1)
    dec = len(val.split(",")[1]) if "," in val else 0
    return unite == mesure[1] and abs(round(float(mesure[0].replace(",", ".")), dec) - float(val.replace(",", "."))) < 1e-12


chk("validation croisée : les écarts de l'onglet Benchmark sont ceux du rapport daté (arrondis), "
    "configurations, directions et figures compris",
    len(mesures) == 6 and all(_lie(e[1], m) for e, m in zip(VC["ecarts"][:6], mesures))
    and f"{VC['configurations']} configurations" in rap and f"{VC['directions']} directions" in rap
    and f"**0** / {VC['directions']}" in rap and VC["ecarts"][6][1] == f"0 direction / {VC['directions']}"
    and all(os.path.exists(os.path.join("docs", "validation", "figures", f)) and f"figures/{f}" in rap
            for f in VC["figures"]),
    str(mesures))

# ================================================================
print("\n=== 6. Benchmark et temps ===")
t0 = time.time()
b = run_bench()
dt = time.time() - t0
chk("VAL-A et VAL-B au vert", b["ok"], f"{b['statut']} pire écart {b['worst']:.4%} "
    + str([(r['lab'][:40], r['val'], r['main']) for v in b['valid'] for r in v['rows'] if not r['ok']]))
t0 = time.time()
compute()
chk("un calcul complet en moins de 1,5 s", time.time() - t0 < 1.5, f"{time.time() - t0:.2f} s")

# ================================================================
print("\n=== 7. Lot A : f_y ET f_u selon l'épaisseur (tableaux 3.3 et 3.4 ANB) ; fy_epaisseur délègue à la table ===")
# NBN EN 1993-1-1 ANB:2018 3.2.1(1) (option a). La règle « −10 / −20 » écrite dans entrees.py était fausse en S460
# (450 au lieu de 440 de 16 à 40 mm) et au-delà de 63 mm (S355 de 70 mm : 335 au lieu de 325), et gardait f_u = 510.
sys.path.insert(0, os.path.join(RACINE, "tests"))
import lot_a_materiau as LA  # noqa: E402
from acier import materiaux as MA  # noqa: E402
from acier.bibliotheques import PERSO  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.entrees import acier_epaisseur  # noqa: E402
_P0 = dict(fy=300, fu=420, bw=0.85)
chk("acier_epaisseur : S355 70 mm → 325, 150 mm (borne du balayage) → 295 ; S460 N/NL 20 mm → 440, 70 mm → 410 "
    "(la table, plus « −10 / −20 »)",
    [acier_epaisseur("S355", 70, _P0)["fy"], acier_epaisseur("S355", 150, _P0)["fy"], acier_epaisseur("S460 N/NL", 20, _P0)["fy"],
     acier_epaisseur("S460 N/NL", 70, _P0)["fy"]] == [325, 295, 440, 410])
try:
    acier_epaisseur("S355", 300, _P0, piece="platine")
    _refus = None
except MA.HorsTable as e:
    _refus = str(e)
chk("acier_epaisseur : au-delà du tableau (300 mm) → refus nommé", bool(_refus) and "platine" in _refus, str(_refus))
LA.jouer(chk, compute, mutations=[LA.MUTATIONS[0], LA.MUTATIONS[1], LA.MUTATION_REGLE_PIED], attendus=[
    ("note neuve (HEA 300, platine 25 mm S355)", {}, {"c": (14, LA.FY_S355, LA.FU_S355), "p": (25, LA.FY_S355_40, LA.FU_S355)}),
    ("poteau HEB 300 (tf = 19 mm)", dict(prof_c="HEB 300"), {"c": (19, LA.FY_S355_40, LA.FU_S355)}),
    ("S460 N/NL, platine 25 mm (tableau 3.4 ANB : 440, l'ancienne règle disait 450)", dict(nu_c="S460 N/NL", nu_pl="S460 N/NL"),
     {"p": (25, LA.FY_S460_40, LA.FU_S460)}),
    ("platine 70 mm S355 (325, l'ancienne règle disait 335)", dict(tp_u=70), {"p": (70, LA.FY_S355_80, LA.FU_S355)}),
], ligne_fu=("note neuve", {}, "w"),
   perso=(dict(nu_pl=PERSO, fy_u=300, fu_u=420, bw_u=0.85), "p", 290, 420,
          "platine de 25 mm : la réduction historique du module, −10 MPa de 16 à 40 mm, sur la valeur saisie"),
   hors_table=(dict(tp_u=300), "platine"))

# ================================================================
print("\n=== 8. Lot de clôture (27/11/2026) : l'ancien et le nouveau de chaque correction ===")
from acier.assemblages.pied_poteau.platine_beton.entrees import OUI, TYPE_TETE, PERIM_EC3  # noqa: E402
S2 = math.sqrt(2)
# --- R05 : aire de diffusion SEMBLABLE à l'aire chargée (EN 1992-1-1 §6.7)
bcN = CP.beton_comprime(30, 0.85, 1.5, 500, 500, 600, 1500, 600, 30, 30)
b2a, l2a = CP.diffusion_independante(500, 500, 600, 1500, 600)
fjd_ancien = 2 / 3 * 17.0 * min(3.0, math.sqrt(b2a * l2a / 250000.0))
chk("R05 — massif allongé 600 × 1 500 sous une platine 500 × 500 : un seul rapport k = min(3 ; 600/500 ; 1100/500) = 1,2 "
    "→ fjd = 13,60 MPa ; l'ancienne diffusion, direction par direction, rendait 18,41 MPa (+35 %)",
    pres(bcN["k"], 1.2, 1e-12) and pres(bcN["fjd"], 13.6, 1e-9) and pres(bcN["Ac1"], 1.44 * 250000, 1e-9)
    and pres(fjd_ancien, 18.41, 5e-4) and pres(compute(dict(bc_u=600, lc_u=1500)).fjd, 13.6, 1e-9),
    f"{bcN['fjd']:.3f} / ancien {fjd_ancien:.3f}")
chk("R05 — massif carré : rien ne change (900 × 900 : k = 1,8 = √(Ac1/Ac0) de l'ancienne règle, fjd = 20,40 MPa) ; la "
    "hauteur du massif borne toujours (hc = 200 : k = 1,4)",
    pres(R0.fjd, 20.4, 1e-9) and pres(R0.k_diff, 1.8, 1e-12)
    and pres(CP.beton_comprime(30, 0.85, 1.5, 500, 500, 900, 900, 200, 30, 30)["k"], 1.4, 1e-12))
# --- mortier de plus de 50 mm (EN 1993-1-8 §6.2.5(7))
chk("mortier : conditions de βj = 2/3 — 60 mm / 20 MPa refusé (plus de 50 mm : fg ≥ fck = 30), 60 / 30 admis, 50 / 20 "
    "admis (seul 0,2·fck compte), 30 / 5 refusé, 110 mm refusé (> 0,2 × 500)",
    [CP.mortier_conforme(t, f, 500, 500, 30) for t, f in ((60, 20), (60, 30), (50, 20), (30, 5), (110, 40))]
    == [False, True, True, False, False])
# --- R12 : une alerte bloquante refuse le pied
Rm3 = compute(dict(f_g=3))
Rm60 = compute(dict(t_g=60, f_g=20))
chk("R12 — mortier de 3 MPa : toutes les vérifications passent, l'alerte « mortier » est bloquante → PIED NON VÉRIFIÉ "
    "(avant : « VÉRIFIÉ », l'alerte ne touchait pas le verdict) ; mortier 60 mm / 20 MPa : refusé aussi",
    Rm3.all_ok and Rm3.geo_ok and Rm3.bloquant == ["mortier"] and not Rm3.verified
    and Rm3.statut.startswith("PIED DE POTEAU NON VÉRIFIÉ") and Rm60.bloquant == ["mortier"] and not Rm60.verified
    and R0.bloquant == [], f"{Rm3.statut} {Rm3.bloquant}")
# --- R13 : le trou d'une tige ne mord ni sur le profilé ni sur son cordon
Rtu = compute(dict(forme=FORME_RHS, h_u=300, b_u=300, t_u=10, w_x=290, w_y=290))
R330, R340, R360 = (compute(dict(w_x=200, w_y=w)) for w in (330, 340, 360))
ids = lambda R: [a.id for a in R.alerts]  # noqa: E731
chk("R13 — tube 300 × 300, grille de 290 : les tiges sont DANS la paroi → refus « tige_acier » (avant : m forcé à 1 mm, "
    "tronçon en T énorme, pied « vérifié »)",
    "tige_acier" in Rtu.bloquant and not Rtu.verified and all(t["d_acier"] == 0.0 for t in Rtu.tiges), str(ids(Rtu)))
chk("R13 — HEA 300, a = 8, M24 (d0 = 26, rondelle 44) : il faut 11,3 + 13 = 24,3 mm de l'aile pour le trou, 33,3 mm pour "
    "la rondelle. À 20 mm (wy = 330) : refus ; à 25 mm (wy = 340) : avis « rondelle » seulement ; à 35 mm (wy = 360) : rien",
    "tige_acier" in R330.bloquant and not R330.verified
    and "tige_acier" not in ids(R340) and "rondelle" in ids(R340) and R340.bloquant == []
    and "tige_acier" not in ids(R360) and "rondelle" not in ids(R360)
    and pres(R340.z_cordon, 8 * S2, 1e-12) and pres(R340.jeu_trou, 25 - 8 * S2 - 13, 1e-9),
    f"{ids(R330)} | {ids(R340)} | {ids(R360)}")
HEA = dict(h=290, b=300, tw=8.5, tf=14, r=27)
cl = lambda x, y: GE.classer_tige(x, y, "I", HEA, 8)  # noqa: E731
chk("R13 — distance de l'axe à l'acier : devant l'aile, la cote ; au coin, la diagonale jusqu'à l'about ; entre les "
    "ailes, près du congé de 27 mm, la distance à l'arc (r − distance au centre du congé)",
    pres(cl(100, 200)["d_acier"], 55, 1e-12) and pres(cl(200, 200)["d_acier"], math.hypot(50, 55), 1e-12)
    and pres(cl(24.25, 111)["d_acier"], 27 - math.hypot(27 - 20, 27 - 20), 1e-12)
    and pres(cl(100, 0)["d_acier"], 100 - 4.25, 1e-12))
# --- bras d'une tige entre les ailes : le cordon, pas le congé (Figure 6.2 d'EN 1993-1-8, cas soudé)
ci = cl(100, 0)
chk("tige entre les ailes (HEA 300, x = 100) : m = 100 − tw/2 − 0,8·a·√2 = 86,70 mm ; l'ancien bras déduisait 0,8·r = "
    "21,6 mm (cas « laminé ») → 74,15 mm, plus court de 12,5 mm, donc plus résistant",
    ci["classe"] == "int" and pres(ci["m"], 100 - 4.25 - 0.8 * 8 * S2, 1e-12)
    and pres(GE.m_int_ancien(100, HEA), 100 - 4.25 - 21.6, 1e-12) and ci["m"] > GE.m_int_ancien(100, HEA) + 12.5,
    f"{ci['m']:.2f} / {GE.m_int_ancien(100, HEA):.2f}")
# --- Tableau 6.6, motif 0,5·bp : la charnière droite qui traverse la platine
Rw = compute(dict(w_x=250, w_y=430))
rw = Rw.rangees[0]
m_w, ex_w = 215 - 145 - 0.8 * 8 * S2, 250 - 215
nc_w = 2 * min(2 * m_w + 0.625 * ex_w, 0.5 * 125 + m_w + 0.3125 * ex_w, 0.25 * 250 + m_w + 0.3125 * ex_w)
anc = CP.tstub_rangee(rw["liste"], rw["m"], rw["e_x"], 25, Rw.fy_p, 1.0, Rw.Lb, Rw.As, Rw.Ft_anc, rw["cap"])
chk("Tableau 6.6 — rangée devant l'aile, tiges à 250 d'entraxe et à 70 mm de l'aile, platine de 500 : les motifs par "
    "tige donnent 268,8 mm ; la charnière droite 0,5·bp = 250 mm gouverne (−7,0 % sur Mpl et sur le mode 1) — motif "
    "absent avant le lot",
    Rw.geo_ok and rw["classe"] == "ext" and pres(nc_w, 268.8, 5e-4) and pres(rw["leff_nc_tiges"], nc_w, 1e-9) and rw["droite"]
    and pres(rw["leff1"], 250, 1e-12) and pres(rw["leff2"], 250, 1e-12) and pres(anc["leff1"], nc_w, 1e-9)
    and pres(rw["FT1"] / anc["FT1"], 250 / nc_w, 1e-9) and rw["FT1"] < anc["FT1"],
    f"{rw['leff_nc_tiges']:.1f} → {rw['leff2']:.1f} ; FT1 {anc['FT1'] / 1e3:.1f} → {rw['FT1'] / 1e3:.1f} kN")
chk("Tableau 6.6 — le cas par défaut ne change pas (tiges de coin : 2 × 0,25·b = 150 mm < 0,5·bp = 250 mm) ; une tige "
    "entre les ailes d'un profilé en I n'a pas ce motif",
    pres(R0.rangees[0]["leff1"], 150, 1e-12) and not R0.rangees[0]["droite"]
    and all(r["l_droite"] is None for r in compute(dict(w_x=200, w_y=400, n_ay=3)).rangees if r["classe"] == "int"))
# --- crosse : fyb ≤ 300 MPa (EN 1993-1-8 §6.2.6.12(5))
chk("crosse — classe 8.8 (fyb = 640) : refus « crosse_fyb » ; classe 5.6 (fyb = 300) : admise ; une tige à tête 8.8 : "
    "pas concernée",
    "crosse_fyb" in compute(dict(typ_a=TYPE_CROSSE)).bloquant and not compute(dict(typ_a=TYPE_CROSSE)).verified
    and "crosse_fyb" not in ids(compute(dict(typ_a=TYPE_CROSSE, classe="5.6"))) and "crosse_fyb" not in ids(R0))
# --- statut : torsion, fendage, éclatement latéral ; la raison n'existe que sur un pied vérifié
Rt = compute(dict(Vx_Ed=0, Vy_Ed=0, T_Ed=20, bc_u=3000, lc_u=3000))
Rvt = compute(dict(T_Ed=20, bc_u=3000, lc_u=3000))
Rtm = compute(dict(Vx_Ed=0, Vy_Ed=0, T_Ed=-20, bc_u=3000, lc_u=3000))
chk("torsion seule (T = 20 kNm) : 17,68 kN par tige (deux composantes de 12,5), mais ni la rupture de bord ni le pry-out "
    "ne s'activent (V résultant nul) → PIED À VÉRIFIER « torsion : bord et pry-out du béton non vérifiés », avis "
    "« tors_beton » ; avec V en plus : le taux du bord ne bouge pas, même statut",
    pres(Rt.V_t_en, math.hypot(12.5, 12.5), 1e-9) and not Rt.ck["edge"].active and not Rt.ck["pry"].active
    and Rt.verified and "torsion" in Rt.a_verifier and "tors_beton" in ids(Rt) and Rt.statut.startswith("PIED DE POTEAU À VÉRIFIER")
    and pres(Rvt.ck["edge"].eta, compute(dict(bc_u=3000, lc_u=3000)).ck["edge"].eta, 1e-12) and "torsion" in Rvt.a_verifier
    and "tors_beton" not in ids(compute(dict(bc_u=3000, lc_u=3000))), f"{Rt.statut} | {Rvt.statut}")
Rc14 = compute(dict(perim=PERIM_TOUT, bc_u=1400, lc_u=1400, Vy_Ed=10))
Rc30 = compute(dict(perim=PERIM_TOUT, bc_u=3000, lc_u=3000))
Re14 = compute(dict(perim=PERIM_EC3, bc_u=1400, lc_u=1400, Vy_Ed=10))
Rc07 = compute(dict(perim=PERIM_TOUT, bc_u=700, lc_u=700, hef=300))
chk("fendage — périmètre COMPLET, bord à 500 mm < 1,5·hef = 450 ? non : à 1 400 de massif le bord est à 500 mm ≥ 450 → "
    "rien ; à 900 (défaut, 250 mm) la raison « fendage du massif non vérifié » est nommée ; au périmètre EC3 elle reste "
    "un avis (« fend »), sans changer le statut",
    Rc30.justifier == [] and Rc14.justifier == [] and "fendage du massif non vérifié" in compute(dict(perim=PERIM_TOUT)).justifier
    and "fend" in ids(R0) and not any("fendage" in x for x in R0.justifier), f"{Rc14.justifier} {Rc14.c_min}")
chk("éclatement latéral — tige à tête à 150 mm du bord = 0,5·hef, périmètre complet : raison nommée ; sans objet pour "
    "une tige scellée",
    "éclatement latéral non vérifié" in Rc07.justifier
    and "éclatement latéral non vérifié" not in compute(dict(perim=PERIM_TOUT, bc_u=700, lc_u=700, typ_a=TYPE_ETA)).justifier,
    str(Rc07.justifier))
Rcd = compute(dict(perim=PERIM_TOUT))
chk("la raison « à vérifier » n'existe que sur un pied VÉRIFIÉ : périmètre complet par défaut (interaction du béton "
    "107,7 %) → NON VÉRIFIÉ, a_verifier vide, la liste des raisons reste lisible (justifier)",
    not Rcd.verified and Rcd.a_verifier == "" and Rcd.justifier and Rcd.statut == "PIED DE POTEAU NON VÉRIFIÉ"
    and Rc14.verified and Rc14.a_verifier == "" and Re14.statut == "PIED DE POTEAU VÉRIFIÉ (EC3 — béton d'ancrage en information)",
    f"{Rcd.statut} | {Rc14.statut} | {Re14.statut}")
chk("la portée du verdict est écrite (R.portee) : périmètre EC3 — béton d'ancrage en information ; complet — il décide ; "
    "dans les deux, fendage, éclatement latéral et torsion sur le béton nommés « non calculés »",
    "en information, hors verdict" in R0.portee and "décident." in Rcd.portee
    and all("Non calculés : fendage, éclatement latéral, effet de la torsion sur le béton." in R.portee for R in (R0, Rcd)))
chk("torsion de signe inversé (T = −20 kNm) : la même raison, le même statut — le signe ne la fait pas disparaître",
    Rtm.verified and Rtm.a_verifier == Rt.a_verifier and Rtm.statut == Rt.statut, Rtm.statut)
# --- γMs,V de l'EN 1992-4 (Tableau 4.1) : 1,5 dès que fuk > 800 MPa ou fyk/fuk > 0,8
v109 = CP.tige_cisaillement_EN1992_4(353, 1000, 900, 24, 10, 40, 0, 1e9)
v88, v46 = CP.tige_cisaillement_EN1992_4(353, 800, 640, 24, 10, 40, 0, 1e9), CP.tige_cisaillement_EN1992_4(353, 400, 240, 24, 10, 40, 0, 1e9)
chk("γMs,V de l'EN 1992-4 (Tableau 4.1, seconde branche citée de mémoire, du côté sûr) : tige 10.9 → 1,5 (avant : "
    "1,25, VRd,s surestimé de 20 %) ; 8.8 → 1,25 ; 4.6 → fuk/fyk = 1,667 ; VRd,s = k6·As·fuk/γMs,V = 117,7 kN en 10.9",
    pres(v109[4], 1.5, 1e-12) and pres(v88[4], 1.25, 1e-12) and pres(v46[4], 400 / 240, 1e-12)
    and pres(v109[1], 0.5 * 353 * 1000 / 1.5, 1e-9) and not v109[3], f"{v109[4]} {v88[4]} {v46[4]} {v109[1]:.0f}")
# --- A11 : appui direct du poteau (option explicite, « Non » par défaut)
w0 = CP.cordons("I", HEA, 8, 470, 0.9, 1.25, 300e3, 50e6, 30e6, 0, 40e3, 0)
w1 = CP.cordons("I", HEA, 8, 470, 0.9, 1.25, 300e3, 50e6, 30e6, 0, 40e3, 0, contact=True)
sn1 = 50e6 * 145 / w0["Ix"] + 30e6 * 150 / w0["Iy"] - 300e3 / w0["A"]
chk("appui direct — cordons du cas par défaut : sans contact σn = N/A + M·r/I = 119,8 MPa ; avec contact la compression "
    "passe par l'appui, σn = max(0 ; M·r/I − N/A) = 75,8 MPa ; le cisaillement ne change pas",
    pres(w0["sig_n"], 300e3 / w0["A"] + 50e6 * 145 / w0["Ix"] + 30e6 * 150 / w0["Iy"], 1e-9) and pres(w0["sig_n"], 119.8, 1e-3)
    and pres(w1["sig_n"], sn1, 1e-9) and pres(w1["sig_n"], 75.8, 1e-3) and pres(w1["tau_par"], w0["tau_par"], 1e-12)
    and w1["contact"] and not w0["contact"], f"{w0['sig_n']:.2f} → {w1['sig_n']:.2f}")
Rad = compute(dict(contact=OUI))
chk("appui direct — moteur : défaut « Non » (cordons à 40,5 %) ; « Oui » : 25,7 %, avis « appui_direct » ; sous "
    "N = 2 000 kN les cordons ne gardent que le cisaillement (σn = 0) ; en TRACTION (N = −100 kN) l'option ne change rien",
    not R0.appui_direct and pres(R0.ck["w"].eta, 0.405, 2e-3) and Rad.appui_direct and pres(Rad.ck["w"].eta, 0.257, 2e-3)
    and "appui_direct" in ids(Rad) and "appui_direct" not in ids(R0)
    and compute(dict(contact=OUI, N_Ed=2000)).cordons["sig_n"] == 0.0
    and pres(compute(dict(contact=OUI, N_Ed=-100)).ck["w"].eta, compute(dict(N_Ed=-100)).ck["w"].eta, 1e-12),
    f"{R0.ck['w'].eta:.4f} → {Rad.ck['w'].eta:.4f}")
# --- k1 du cône : le libellé dit d'où vient la valeur
chk("cône — k1 = 7,7 gardé (valeur des fixations post-installées, prudente pour une tige coulée en place), et la ligne "
    "le dit ; avec une ETA : « valeur de l'ETA »",
    pres(R0.cone["k1"], 7.7, 1e-12) and "valeur des fixations post-installées" in R0.ck["cone"].vals
    and "valeur de l'ETA" in compute(dict(typ_a=TYPE_ETA, eta_k1=8.5)).ck["cone"].vals)
# --- les cinq exemples publiés
bm = {x["id"]: x for x in b["bench"]}
e1 = {r["lab"][:12]: r for r in bm["BM2"]["rows"]}
chk("exemples publiés — cinq fiches complètes rejouées (MSB §6.5 ; SCI P358 ex. 1 à 3 ; SCI P398 E.1), aucune ligne "
    "« À vérifier » ni « À expliquer » : chaque écart de plus de 1 % porte son explication",
    sorted(bm) == ["BM1", "BM2", "BM3", "BM4", "BM5"] and b["fiches"] == []
    and all(r["st"] in ("OK", "Écart expliqué", "Écart > 3 % documenté") for x in b["bench"] for r in x["rows"])
    and b["statut"] == "VALIDÉ" and b["worst"] <= 0.01, f"{b['statut']} {b['worst']:.4f}")
chk("SCI P398 E.1 — c = 112 mm, Fc,pl,Rd = 2 193 kN, ℓeff,1 = 0,5·bbp = 300 mm et Mpl,1,Rd = 47,8 kNm retrouvés à moins "
    "de 0,3 % ; FT,3 = 753 kN contre 812 (la tige est bornée par EN 1992-4 : écart documenté, du côté sûr)",
    all(abs(r["ec"]) <= 0.003 for r in bm["BM2"]["rows"] if r["lab"].startswith(("c =", "Fc,pl", "ℓeff", "Mpl")))
    and sum(1 for r in bm["BM2"]["rows"] if r["lab"].startswith(("c =", "Fc,pl", "ℓeff", "Mpl"))) == 4
    and any(r["lab"].startswith("FT,3") and pres(r["val"], 753.07, 1e-4) and r["exp"] for r in bm["BM2"]["rows"]))
chk("pieds articulés publiés (appui direct) — le critère d'aire du moteur retrouve tp,min à moins de 1,5 % (45 / 47 / "
    "89 / 19 mm) ; l'épaisseur RETENUE par le moteur est du côté sûr (jusqu'à +4 %)",
    all(abs(r["ec"]) <= 0.015 for k in ("BM1", "BM3", "BM4", "BM5") for r in bm[k]["rows"] if "critère d'aire" in r["lab"])
    and all(-0.001 <= r["ec"] <= 0.041 for k in ("BM1", "BM3", "BM4", "BM5") for r in bm[k]["rows"] if "RETIENT" in r["lab"]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

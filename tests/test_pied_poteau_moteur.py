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
from acier.assemblages.pied_poteau.platine_beton.entrees import FORME_RHS, TYPE_CROSSE, NON, PERIM_TOUT  # noqa: E402

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
    # entraxes ≤ 430 : pince ≥ 35 mm ≥ 1,2·d0 (une géométrie invalide rend des capacités nulles, à dessein)
    Rr = compute(dict(w_x=rng.choice([300, 380, 430]), w_y=rng.choice([300, 380, 430]), n_ay=rng.choice([2, 3, 4]),
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

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

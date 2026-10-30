# -*- coding: utf-8 -*-
"""Moteur de l'assemblage par platine d'about de hauteur partielle.

Chaque garantie rougit si on la retire :
  1. EXEMPLE PUBLIÉ (MSB Part 5 §2.4, Worked Example: Partial depth end
     plate — IPE A 550 S275, platine 430×200×12 S275, 12 M20 8.8, p3 = 140,
     VEd = 550 kN, tying 275 kN) : les valeurs imprimées, EN DIRECT, effort
     tranchant ET tying.
  2. RÈGLES DU GUIDE : groupe de boulons en trois branches (le 0,8 ne
     s'applique que si le cisaillement des boulons gouverne), rupture de
     bloc k = 1 / 0,5 selon hp ≥ 1,36·p3, coefficients de pleine résistance
     0,46 / 0,48 / 0,55, ductilité tp ≤ 0,36·d·√(fub/fy).
  3. TRONÇON EN T : les DEUX ailes portées par une seule résistance (pas de
     ×2), mode 1 par la méthode 2, Σleff = hp sur les détails courants ;
     ELU (fy/γM0) ≠ tying (fu/γMu) ; tying vérifié SEUL.
  4. CAS PAR DÉFAUT : ancres partagées avec les deux autres modules.
  5. ALERTES ET DOMAINE ; 6. PRÉDIM ; 7. ROBUSTESSE ; 8. RÉGRESSION FIGÉE.

Lancement : python3 tests/test_platine_about_moteur.py
"""
import json
import math
import os
import random
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.assemblages.poutre_poutre.platine_about import moteur  # noqa: E402
from acier.assemblages.poutre_poutre.platine_about.benchmark import WE, run_bench  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * abs(b)


S3 = math.sqrt(3)

# ================================================================
print("=== 1. Exemple publié : MSB Part 5 §2.4 (platine d'about partielle) ===")
R = moteur.compute(WE)
chk("géométrie : e2 = (200 − 140)/2 = 30, e1' = 40, 12 boulons", R.e_2 == 30 and R.e1bot == 40 and R.n_b == 12)
chk("âme portée : hp·tw·fy/√3 = 614 kN (Av = 3870 mm²)", pres(R.ck["vwS"].Rd, 614, 0.01) and R.Av_w == 3870,
    str(R.ck["vwS"].Rd))
chk("un boulon : Fv,Rd = 94 kN", pres(R.Fv_Rd, 94, 0.01), str(R.Fv_Rd))
chk("pression diamétrale, boulons d'extrémité = 107 kN (source : αb arrondi à 0,61)", pres(R.Fbb_L, 107, 0.01),
    str(R.Fbb_L))
chk("pression diamétrale, boulons intérieurs = 142 kN", pres(R.Fbi_L, 142, 0.01), str(R.Fbi_L))
chk("groupe : Fv,Rd ≤ (Fb,Rd)min → 0,8·ns·Fv,Rd = 902 kN", pres(R.ck["bg"].Rd, 902, 0.01)
    and R.branche_g.startswith("0,8"), str(R.ck["bg"].Rd))
chk("platine : min(brut ; net ; bloc) = rupture de bloc 1182 kN",
    pres(min(R.ck["pcg"].Rd, R.ck["pcn"].Rd, R.ck["pcb"].Rd), 1182, 0.01)
    and R.ck["pcb"].Rd < R.ck["pcg"].Rd < R.ck["pcn"].Rd)
chk("bloc avec k = 1 car hp = 430 ≥ 1,36·p3 = 190,4 (avec k = 0,5 on tomberait à ≈ 1103)",
    R.k_bloc == 1 and R.hp_136)
chk("tying – platine en flexion = 493 kN (source : m arrondi à 59)", pres(R.ck["ptT"].Rd, 493, 0.01),
    str(R.ck["ptT"].Rd))
chk("tying – âme de la poutre en traction tw·hp·fu/γMu = 1513 kN", pres(R.ck["tnS"].Rd, 1513, 0.01),
    str(R.ck["tnS"].Rd))
chk("pleine résistance S275 : a ≥ 0,48·tw (4,32 mm ≤ 5,6)", pres(R.k_plein, 0.48, 0.005)
    and pres(R.a_plein, 0.48 * 9, 0.005) and not any(a.id == "a_rec" for a in R.alerts))
chk("dimensionnante en cisaillement : âme de la poutre, 550/614 = 0,90",
    R.gov.key == "vwS" and pres(R.eta_max, 550 / 614.4450, 1e-4) and R.verified, R.gov.key)
chk("tying 275/493 = 0,56 (sans interaction avec VEd : vérifié seul)",
    pres(R.ck["ptT"].eta, 0.559, 0.01) and not R.ck["bvt"].active and R.Ft_Ed == 0)
chk("ductilité : tp = 12 ≤ 0,36·20·√(800/275) = 12,28 mm (de justesse, comme le guide)",
    pres(R.tp_max, 12.28, 0.001) and R.rec_tp)
B = run_bench()
chk("benchmark : statut VALIDÉ, pire écart ≤ 1 %", B["ok"] and B["statut"] == "VALIDÉ" and B["worst"] <= 0.01,
    f"{B['statut']} {B['worst']}")

# ================================================================
print("\n=== 2. Règles du guide ===")
# trois branches du groupe de boulons
chk("branche ΣFb,Rd si (Fb,Rd)max ≤ Fv,Rd",
    moteur.groupe_boulons(100, 60, 80, 2, 4) == (2 * 60 + 4 * 80, "ΣFb,Rd (pression diamétrale)"))
chk("branche ns·(Fb,Rd)min entre les deux",
    moteur.groupe_boulons(70, 60, 80, 2, 4) == (6 * 60, "ns·(Fb,Rd)min"))
chk("branche 0,8·ns·Fv,Rd si Fv,Rd ≤ (Fb,Rd)min (le 0,8 seulement ici)",
    moteur.groupe_boulons(50, 60, 80, 2, 4)[0] == 0.8 * 6 * 50)
R0 = moteur.compute(dict())
chk("cas par défaut : Fb,ext 91,66 < Fv,Rd 94,08 < Fb,int 113,9 → ns·(Fb,Rd)min = 549,96 kN",
    R0.branche_g == "ns·(Fb,Rd)min" and pres(R0.ck["bg"].Rd, 6 * R0.Fbb_L, 1e-12)
    and R0.Fbb_L < R0.Fv_Rd < R0.Fbi_L)
Rk = moteur.compute(dict(p3_u=140, bp_u=200, hp_u=180, n1_u=2, p1_u=80, e1_u=50))
chk("rupture de bloc : hp = 180 < 1,36·140 → k = 0,5 (§3.10.2(3))", Rk.k_bloc == 0.5 and not Rk.hp_136)
Ant, Anv = Rk.Ant_p, Rk.Anv_p
chk("… VRd,b = 2·(0,5·fu·Ant/γM2 + fy·Anv/√3)",
    pres(Rk.ck["pcb"].Rd, 2 * (0.5 * 510 * Ant / 1.25 + 355 * Anv / S3) / 1000, 1e-12))
chk("coefficients de pleine résistance du guide : 0,46 / 0,48 / 0,55 (S235 / S275 / S355)",
    all(pres(moteur.coef_pleine_resistance(fy, fu, bw, 1.25), k, 0.01)
        for fy, fu, bw, k in ((235, 360, 0.8, 0.46), (275, 430, 0.85, 0.48), (355, 510, 0.9, 0.55))))
chk("ductilité : 0,36·d·√(fub/fy) — 10,8 mm en S355, 12,28 mm en S275 (M20 8.8)",
    pres(moteur.tp_ductile(20, 800, 355), 10.808, 1e-3) and pres(moteur.tp_ductile(20, 800, 275), 12.280, 1e-3))
chk("cisaillement brut de la platine : 2·(hp·tp/1,27)·fy/√3 (exemple : 1290 kN)",
    pres(R.ck["pcg"].Rd, 2 * 430 * 12 / 1.27 * 275 / S3 / 1000, 1e-12) and pres(R.ck["pcg"].Rd, 1290, 0.001))
chk("cisaillement net : 2·tp·(hp − n1·d0)·fu/(√3·γM2) (exemple : 1420 kN)",
    pres(R.ck["pcn"].Rd, 1420, 0.001))

# ================================================================
print("\n=== 3. Tronçon en T (traction) ===")
m = (140 - 9) / 2 - 0.8 * 5.6 * math.sqrt(2)
chk("m = (p3 − tw)/2 − 0,8·a·√2 ; n = e2 = 30 ; ew = dw/4", pres(R.m_T, m, 1e-12) and R.nT == 30
    and R.ew_T == 37 / 4)
chk("Σleff = hp = 430 sur l'exemple (rangées en groupe, bords libres)", R.leff_T == 430)
Mpl = 0.25 * 430 * 144 * 430 / 1.1
FT1 = (8 * 30 - 2 * 9.25) * Mpl / (2 * m * 30 - 9.25 * (m + 30)) / 1000
chk("mode 1 par la méthode 2, résistance des DEUX ailes (pas de ×2) = 491,86 kN",
    pres(R.FT1, FT1, 1e-12) and R.FT_Rd == min(R.FT1, R.FT2, R.FT3) and pres(R.FT_Rd, FT1, 1e-12))
chk("mode 3 = ΣFt,Rd des 12 boulons avec γMu", pres(R.FT3, 12 * 0.9 * 800 * 245 / 1.1 / 1000, 1e-12))
Re = moteur.compute(dict(WE, nat_N="ELU fondamental"))
chk("ELU : fy/γM0 et γM2 (FT,1 = 491,86 × 275/430 × 1,1 ; FT,3 avec γM2)",
    pres(Re.FT1, FT1 * 275 / 430 * 1.1, 1e-9) and pres(Re.FT3, 12 * 0.9 * 800 * 245 / 1.25 / 1000, 1e-12))
chk("ELU : interaction cisaillement + traction active, tying inactif",
    Re.ck["bvt"].active and not Re.ck["wT"].active
    and pres(Re.ck["bvt"].Ed, (550 / 12) / Re.Fv_Rd + (275 / 12) / (1.4 * Re.Ft_Rd), 1e-12))
chk("ELU : âme portée tw·hp·fy/γM0 ; cordons sous la résultante √(qz² + qx²)",
    pres(Re.ck["tnS"].Rd, 9 * 430 * 275 / 1000, 1e-12)
    and pres(Re.ck["w"].Ed, math.hypot(550000 / 860, 275000 / 860), 1e-12))
chk("tying : cordon sous NEd seul avec γMu", R.ck["wT"].active
    and pres(R.ck["wT"].Ed, 275000 / 860, 1e-12) and pres(R.ck["w"].Ed, 550000 / 860, 1e-12))
chk("NEd en compression : sans effet (transmis par contact)",
    not any(moteur.compute(dict(N_Ed=-80)).ck[k].active for k in ("bvt", "ptT", "pp", "tnS")))

# ================================================================
print("\n=== 4. Cas par défaut : ancres partagées ===")
chk("défaut VÉRIFIÉ ; la gorge minimale (60 %) n'est PAS dimensionnante", R0.verified and R0.gov.key != "wa"
    and R0.ck["wa"].eta == 0.6, R0.gov.key)
chk("grugeage : MEd = VEd·(tp + c) = 125 × 0,160 = 20 kNm ; Mv,N,Rd = 49,479 (ancre des deux autres modules)",
    pres(R0.ck["mN"].Ed, 20.0, 1e-12) and pres(R0.ck["mN"].Rd, 49.4789861999044, 1e-9))
chk("âme portée : 8,5 × 190 × 355/√3 = 331,0 kN", pres(R0.ck["vwS"].Rd, 8.5 * 190 * 355 / S3 / 1000, 1e-12))
chk("âme porteuse : cisaillement local Av = tw·hp = 2090 mm², Ed = VEd/2",
    R0.Av_P == 2090 and R0.ck["vlP"].Ed == 62.5)
chk("pression diamétrale de l'âme porteuse : αb = p1/3d0 − 1/4, k1 = 2,5 (pas de bord libre)",
    pres(R0.abi_P, 60 / 66 - 0.25, 1e-12) and R0.k1_P == 2.5)
chk("pleine résistance S355 (platine) / S355 (âme) : 0,554·8,5 = 4,71 mm ≤ a = 5", pres(R0.a_plein, 4.7067, 1e-3)
    and not R0.alerts)

# ================================================================
print("\n=== 5. Alertes et domaine ===")


def ids(u):
    return {a.id for a in moteur.compute(u).alerts}


chk("hp = 140 : pince basse < 1,2·d0 → bloquant", "e1bot" in ids(dict(hp_u=140)))
chk("bp = 140 : e2 = 25 < 26,4 → bloquant (Tableau 3.3)", "e2p" in ids(dict(bp_u=140)))
chk("p3 = 55 : la rondelle chevauche le cordon → bloquant", "p3_cordon" in ids(dict(p3_u=55, bp_u=120)))
chk("platine hors de l'âme droite de la porteuse → bloquant", "pl_web" in ids(dict(z_C=50, hp_u=320, d_nb=0))
    or "h_dispo" in ids(dict(z_C=50, hp_u=320)))
chk("c < (b − tw)/2 − tp : grugeage trop court → bloquant", "ln_min" in ids(dict(l_n=120)))
chk("tp = 12 en S355 (> 10,8 mm) : ductilité (recommandation, non bloquante)",
    "tp_duct" in ids(dict(tp_u=12)) and moteur.compute(dict(tp_u=12)).geo_ok)
chk("tp = 6 : platine < 8 mm déconseillée", "tp_min" in ids(dict(tp_u=6)))
chk("hp < 0,6·h : maintien en torsion", "hp_rec" in ids(dict(hp_u=170)))
chk("a = 4 < 4,71 : gorge sous la pleine résistance", "a_rec" in ids(dict(a_w=4)))
chk("NEd > 0 : flexion transversale de l'âme porteuse signalée (non vérifiée)", "NEd" in ids(dict(N_Ed=40)))
chk("Tableau 3.3 : e1, e2, p1 et p3 (= p2) contrôlés", [x.id for x in R0.dist] == ["e1p", "e2p", "p1", "p3"]
    and all(x.ok for x in R0.dist))

# ================================================================
print("\n=== 6. Prédimensionnement ===")
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT", V_Ed=250))
chk("proposition complète (V = 250 kN)", Rp.pd.found and Rp.pd.msg == "Proposition complète", Rp.pd.msg)
chk("au moins deux rangées, hp ≥ 0,6·h, p3 ≥ 90, 8 mm ≤ tp ≤ limite de ductilité",
    Rp.pd.n >= 2 and Rp.pd.hp >= 0.6 * Rp.h_S and Rp.pd.p3 >= 90
    and 8 <= Rp.pd.tp <= moteur.tp_ductile(Rp.d_b, Rp.f_ub, Rp.fy_L) + 1e-9)
u2 = moteur.apply_solution(dict(V_Ed=250), Rp.pd.pick)
chk("apply_solution : mode VÉRIFICATION, boulon, rangées, p3, hp, tp, bp, a recopiés",
    u2["mode_calc"] == "VÉRIFICATION" and u2["boulon_u"] == Rp.pd.boulon and u2["n1_u"] == Rp.pd.n
    and u2["p3_u"] == Rp.pd.p3 and u2["hp_u"] == Rp.pd.hp and u2["tp_u"] == Rp.pd.tp and u2["a_w"] == Rp.pd.a)
Rv = moteur.compute(u2)
chk("la solution recopiée est vérifiée en détail, sans alerte", Rv.verified and not Rv.alerts,
    (Rv.gov.key if Rv.gov else "") + str([a.id for a in Rv.alerts]))

# ================================================================
print("\n=== 7. Robustesse (300 tirages aléatoires) ===")
rng = random.Random(11)
nerr = 0
for i in range(300):
    try:
        moteur.compute(dict(
            hp_u=rng.choice([0, 60, 140, 190, 260, 400]), tp_u=rng.choice([0, 6, 8, 10, 15, 20]),
            bp_u=rng.choice([0, 60, 100, 150, 220]), z_C=rng.choice([0, 20, 50, 90]),
            n1_u=rng.randint(0, 8), p1_u=rng.choice([0, 40, 60, 90]), p3_u=rng.choice([0, 50, 90, 140]),
            e1_u=rng.choice([0, 25, 35, 60]), a_w=rng.choice([0, 3, 5, 8]),
            d_nt=rng.choice([0, 30, 50]), d_nb=rng.choice([0, 40]), l_n=rng.choice([0, 100, 150]),
            V_Ed=rng.choice([0, 50, 125, 400, 900]), N_Ed=rng.choice([-60, 0, 80]),
            nat_N=rng.choice(["ELU fondamental", "Tying (accidentel)"]),
            boulon_u=rng.choice(["M12", "M16", "M20", "M24", "M30"]), classe=rng.choice(["4.6", "8.8", "10.9"]),
            cat=rng.choice(["A", "B", "C"]), trou=rng.choice(["Normal", "Surdimensionné"]),
            prof_S=rng.choice(["IPE 200", "HEA 300", "HEB 500", "IPE 600"]),
            prof_P=rng.choice(["HEA 400", "HEB 600", "IPE 300"]),
            mode_calc=rng.choice(["VÉRIFICATION", "PRÉDIMENSIONNEMENT"])))
    except Exception as e:  # noqa: BLE001
        nerr += 1
        if nerr == 1:
            print("      première exception :", type(e).__name__, e, "au tirage", i)
chk("aucune exception sur 300 tirages", nerr == 0, str(nerr))

# ================================================================
print("\n=== 8. Régression figée (tests/ref_platine_about.json) ===")
with open("tests/ref_platine_about.json", encoding="utf-8") as fh:
    REF = json.load(fh)
ndiff = 0
nverif = 0
for nom, cas in REF.items():
    Rr = moteur.compute(cas["inp"])
    actives = {c.key for c in Rr.checks if c.active}
    if actives != set(cas["checks"]):
        ndiff += 1
        print("   ", nom, "vérifications actives divergentes", sorted(actives ^ set(cas["checks"])))
    for k, (ed, rd, eta) in cas["checks"].items():
        nverif += 1
        c = Rr.ck.get(k)
        if c is None or not c.active or abs(c.Ed - ed) > 1e-9 * max(1, abs(ed)) \
           or abs(c.Rd - rd) > 1e-9 * max(1, abs(rd)) or abs(c.eta - eta) > 1e-12 * max(1, abs(eta)):
            ndiff += 1
            print("   ", nom, k, "attendu", (ed, rd, eta), "obtenu",
                  (c.Ed, c.Rd, c.eta) if c and c.active else "inactif")
    for k, v in cas["scal"].items():
        if abs(Rr[k] - v) > 1e-9 * max(1, abs(v)):
            ndiff += 1
            print("   ", nom, "scalaire", k, "attendu", v, "obtenu", Rr[k])
    if (Rr.gov.key if Rr.gov else None) != cas["gov"] or Rr.verified != cas["verified"] \
       or sorted(a.id for a in Rr.alerts) != cas["alerts"]:
        ndiff += 1
        print("   ", nom, "synthèse divergente")
chk(f"RÉGRESSION CALCUL : 0 différence sur les {len(REF)} cas figés ({nverif} vérifications)", ndiff == 0,
    str(ndiff))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

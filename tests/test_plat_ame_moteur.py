# -*- coding: utf-8 -*-
"""Moteur de l'assemblage par plat d'âme soudé (fin plate).

Chaque garantie rougit si on la retire :
  1. EXEMPLE PUBLIÉ (MSB Part 5 §3.4, Worked Example: Fin Plate — IPE A 550
     S275, plat 360×160×10 S275, 2 files de 5 M20 8.8, z = 80 mm) : les
     valeurs du guide, EN DIRECT (1 plan de cisaillement, sans la
     transposition ×2 du benchmark du module doubles cornières).
  2. CAS PAR DÉFAUT : les grandeurs partagées avec le module doubles
     cornières (même z, même groupe, même âme) tombent sur les MÊMES
     nombres — dont le taux de pression diamétrale de l'âme 0,739914…
     ancré par le calcul manuel VAL-A du module 1.
  3. DÉVERSEMENT DU PLAT LONG : courbe BS 5950-1 Annexe B.2.1 (Table 17),
     E = 205 000 MPa — valeurs manuelles indépendantes.
  4. ALERTES ET DOMAINE : pince basse, plat hors âme droite, ductilité
     tp ≤ 0,5·d, plat long non maintenu, hp ≥ 0,6·h.
  5. PRÉDIM : proposition complète, apply_solution recopie tout.
  6. ROBUSTESSE : 200 tirages aléatoires sans exception.

Lancement : python3 tests/test_plat_ame_moteur.py
"""
import math
import os
import random
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.bibliotheques import PERSO  # noqa: E402
from acier.assemblages.poutre_poutre.plat_ame import moteur  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol):
    return abs(a - b) <= tol * abs(b)


# ================================================================
print("=== 1. Exemple publié : MSB Part 5 §3.4 (fin plate) ===")
WE = dict(prof_S=PERSO, hS_u=547, bS_u=210, twS_u=9, tfS_u=15.7, rS_u=24, nu_S="S275",
          prof_P=PERSO, hP_u=800, bP_u=300, twP_u=15, tfP_u=30, rP_u=27, nu_P="S275",
          nu_pl="S275", d_top=60, d_nt=0, d_nb=0, l_n=0, g_h=10, z_C=50,
          hp_u=360, tp_u=10, bp_u=160, boulon_u="M20", classe="8.8",
          n1_u=5, n2_u=2, p1_u=70, p2_u=60, e1_u=40, e2b_u=40, V_Ed=350, g_M1=1.0)
R = moteur.compute(WE)


def vr(k):
    return R.u.V_Ed / R.ck[k].eta


chk("géométrie : z = 80 mm (gh + e2,b + p2/2), e2 plat = 50, e1,b âme = 90",
    R.z_p == 80 and R.e2_p == 50 and R.e1b_S == 90)
chk("inertie polaire du groupe Ip = 107 000 mm² (p. 5-40)", R.Ip == 107000, str(R.Ip))
chk("groupe excentré – cisaillement des boulons VRd = 584 kN (p. 5-40, arrondis de la source)",
    pres(vr("bv"), 584, 0.01), str(vr("bv")))
chk("pression diamétrale du plat VRd = 605 kN (p. 5-41, arrondis de la source ≤ 1 %)",
    pres(vr("pdL"), 605, 0.011), str(vr("pdL")))
chk("pression diamétrale de l'âme VRd = 624 kN (p. 5-42)", pres(vr("pdS"), 624, 0.01), str(vr("pdS")))
chk("plat – cisaillement section brute VRd = 450 kN (p. 5-43)", pres(R.ck["pcg"].Rd, 450, 0.01), str(R.ck["pcg"].Rd))
chk("plat – cisaillement section nette VRd = 497 kN (p. 5-43)", pres(R.ck["pcn"].Rd, 497, 0.01), str(R.ck["pcn"].Rd))
chk("plat – rupture de bloc VRd = 483 kN (p. 5-43)", pres(R.ck["pcb"].Rd, 483, 0.01), str(R.ck["pcb"].Rd))
chk("âme – cisaillement brut VRd = 953 kN (p. 5-45 ; aire calculée +0,25 %)",
    pres(R.ck["vgS"].Rd, 953, 0.01), str(R.ck["vgS"].Rd))
chk("âme – cisaillement net VRd = 995 kN (p. 5-45 ; même écart d'aire)",
    pres(R.ck["vnS"].Rd, 995, 0.01), str(R.ck["vnS"].Rd))
chk("âme – rupture de bloc VRd = 507 kN (p. 5-45, poutre NON grugée : (n1 − 1)·d0)",
    pres(R.ck["vbS"].Rd, 507, 0.01), str(R.ck["vbS"].Rd))
chk("flexion du plat sans objet (hp = 360 ≥ 2,73·z = 218,4)", not R.ck["pfl"].active and R.court)
chk("plat LONG (z = 80 > tp/0,15 = 66,7) : déversement actif", R.long_p and R.ck["plt"].active)
chk("VÉRIFIÉ à VEd = 350 kN ; dimensionnante = cisaillement brut du plat (450 kN, "
    "la plus petite résistance du guide)", R.verified and R.gov.key == "pcg", R.gov.key)

print("\n=== 2. Cas par défaut : ancrages partagés avec le module 1 ===")
R0 = moteur.compute(dict())
chk("défaut : z = 50, Ip = 7200, effort de boulon 66,699… kN (mêmes nombres que doubles cornières)",
    R0.z_p == 50 and R0.Ip == 7200 and pres(R0.F_b, 66.6992108065922, 1e-9), str(R0.F_b))
chk("taux de pression diamétrale de l'âme = 0,739914371796275 (ancre VAL-A du module 1)",
    abs(R0.ck["pdS"].eta - 0.739914371796275) < 1e-12, str(R0.ck["pdS"].eta))
chk("grugeage : A_Tee 6121, Av 2358,5, Wel,N 139377,43, Mv,N,Rd 49,479 kNm, ln,max 290 (VAL-A)",
    R0.A_Tee == 6121 and R0.Av_S == 2358.5 and pres(R0.W_N, 139377.425915224, 1e-9)
    and pres(R0.ck["mN"].Rd, 49.4789861999044, 1e-9) and R0.ln_max == 290)
chk("plat court par défaut (z = 50 ≤ tp/0,15 = 66,7) : ni flexion ni déversement",
    not R0.ck["pfl"].active and not R0.ck["plt"].active)
chk("soudure : qz = VEd/(2hp) = 328,9 N/mm ; qx = 3·MS/hp² = 519,4 N/mm ; Fw,Rd = fvw·a",
    pres(R0.qz_w, 125000 / 380, 1e-9) and pres(R0.qx_w, 3 * 6.25e6 / 36100, 1e-9)
    and pres(R0.ck["w"].Rd, R0.fvw * 5, 1e-9))
chk("cisaillement local âme porteuse : Av = tw·hp = 11 × 190 = 2090 mm², Ed = VEd/2",
    R0.Av_P == 2090 and R0.ck["vlP"].Ed == 62.5)
chk("cas par défaut VÉRIFIÉ, dimensionnante = pression diamétrale de l'âme",
    R0.verified and R0.gov.key == "pdS")
chk("γM1 par défaut = 1,10 (ANB belge)", R0.u.g_M1 == 1.1)

print("\n=== 3. Déversement du plat long (BS 5950-1 Annexe B.2.1) ===")
chk("pb = py pour λ ≤ λL0 (plateau)", moteur.pb_bs5950(20, 275) == 275)
lam = 2.8 * math.sqrt(80 * 360 / (1.5 * 100))
chk("λLT = 2,8·√(z·hp/(1,5·tp²)) = 38,80 sur l'exemple publié", pres(R.lam_LT, lam, 1e-12), str(R.lam_LT))
pE = math.pi ** 2 * 205000 / lam ** 2
lam0 = 0.4 * math.sqrt(math.pi ** 2 * 205000 / 275)
eta = 0.007 * (lam - lam0)
phi = (275 + (eta + 1) * pE) / 2
pb = pE * 275 / (phi + math.sqrt(phi * phi - pE * 275))
chk("pb (Perry-Robertson, E = 205 000) recalculé indépendamment", pres(R.pb_LT, pb, 1e-12),
    f"{R.pb_LT} vs {pb}")
chk("Mb,Rd = Wel·pb/γM1 (γM1 = 1,0 sur l'exemple)", pres(R.ck["plt"].Rd, 216000 * pb / 1e6, 1e-12))
Rg = moteur.compute(dict(WE, g_M1=1.1))
chk("γM1 = 1,10 (ANB) réduit Mb,Rd de 1/1,1", pres(Rg.ck["plt"].Rd, R.ck["plt"].Rd / 1.1, 1e-12))

print("\n=== 4. Alertes et domaine ===")
Ra = moteur.compute(dict(hp_u=300))
chk("hp = 300 avec 3 rangées : pince basse OK mais plat hors âme droite → alerte pl_web",
    any(a.id == "pl_web" for a in Ra.alerts))
Rb = moteur.compute(dict(hp_u=140))
chk("hp = 140 : pince basse < 1,2·d0 → alerte bloquante e1bot",
    any(a.id == "e1bot" and a.block for a in Rb.alerts) and not Rb.verified)
Rc = moteur.compute(dict(tp_u=15))
chk("tp = 15 > 0,5·d = 10 : alerte ductilité (non bloquante)",
    any(a.id == "tp_duct" and not a.block for a in Rc.alerts))
Rd_ = moteur.compute(dict(e2b_u=70, lt_ok="Non"))
chk("z = 80 > tp/0,15 et poutre non maintenue : alerte plat long",
    Rd_.long_p and any(a.id == "long_nr" for a in Rd_.alerts))
chk("hp < 0,6·h : recommandation signalée", any(a.id == "hp_rec" for a in moteur.compute(dict(hp_u=170)).alerts))
chk("Tableau 3.3 : 6 lignes actives par défaut (e1p, e2p, e1b, e2b, p1 ; p2 inactif)",
    len(R0.dist) == 5 and all(x.ok for x in R0.dist), str([x.id for x in R0.dist]))

print("\n=== 5. Prédimensionnement ===")
Rp = moteur.compute(dict(mode_calc="PRÉDIMENSIONNEMENT"))
chk("proposition complète au cas par défaut", Rp.pd.found and Rp.pd.msg == "Proposition complète", Rp.pd.msg)
chk("épaisseur proposée ≤ 0,5·d (ductilité)", Rp.pd.tp <= 0.5 * Rp.d_b + 1e-9, str(Rp.pd.tp))
u2 = moteur.apply_solution(dict(), Rp.pd.pick)
chk("apply_solution : mode VÉRIFICATION, boulon, rangées, hp, tp, bp, a recopiés",
    u2["mode_calc"] == "VÉRIFICATION" and u2["boulon_u"] == Rp.pd.boulon
    and u2["n1_u"] == Rp.pd.n and u2["hp_u"] == Rp.pd.hp and u2["tp_u"] == Rp.pd.tp
    and u2["a_w"] == Rp.pd.a)
Rv = moteur.compute(u2)
chk("la solution recopiée est vérifiée en détail", Rv.verified, Rv.gov.key if Rv.gov else "")

print("\n=== 6. Robustesse (200 tirages aléatoires) ===")
rng = random.Random(7)
nerr = 0
for i in range(200):
    try:
        moteur.compute(dict(
            hp_u=rng.choice([0, 60, 140, 190, 260, 400]), tp_u=rng.choice([0, 6, 8, 10, 15, 20]),
            bp_u=rng.choice([0, 60, 100, 160, 220]), z_C=rng.choice([0, 20, 50, 90]),
            n1_u=rng.randint(1, 8), n2_u=rng.choice([1, 2]), p1_u=rng.choice([0, 40, 60, 90]),
            p2_u=rng.choice([0, 50, 70]), e1_u=rng.choice([0, 25, 35, 60]),
            e2b_u=rng.choice([0, 30, 40, 80]), g_h=rng.choice([0, 10, 25]),
            d_nt=rng.choice([0, 30, 50]), d_nb=rng.choice([0, 40]), l_n=rng.choice([0, 100, 150]),
            V_Ed=rng.choice([0, 50, 125, 400, 900]), N_Ed=rng.choice([-60, 0, 80]),
            M_Ed=rng.choice([0, 5]), boulon_u=rng.choice(["M12", "M16", "M20", "M24"]),
            classe=rng.choice(["4.6", "8.8", "10.9"]), cat=rng.choice(["A", "B", "C"]),
            trou=rng.choice(["Normal", "Surdimensionné"]),
            prof_S=rng.choice(["IPE 200", "HEA 300", "HEB 500"]),
            prof_P=rng.choice(["HEA 400", "HEB 600"])))
    except Exception as e:  # noqa: BLE001
        nerr += 1
        if nerr == 1:
            print("      première exception :", type(e).__name__, e, "au tirage", i)
chk("aucune exception sur 200 tirages", nerr == 0, str(nerr))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)

# -*- coding: utf-8 -*-
"""
Tests du noyau géotechnique (modules/sol_theorie.py).

Ces contrôles ne dépendent d'aucune bibliothèque externe : ils opposent
l'implémentation à des VALEURS EXACTES et à des propriétés de premier
principe. Une constante fausse dans le moteur les fait rougir.

Lancer :  python3 tests/test_sol_theorie.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import sol_theorie as ST  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   {info}" if info else ""))


def presque(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


# =====================================================================
print("=" * 72)
print(" 1. FACTEUR D'INFLUENCE DE NEWMARK — tables publiées")
print("=" * 72)
# Valeurs tabulées du facteur d'influence sous le coin d'un rectangle
# chargé (tables de Newmark, reprises dans Das, Bowles, Terzaghi-Peck).
for m, n, attendu in [(0.5, 0.5, 0.0840), (0.5, 1.0, 0.1202), (1.0, 1.0, 0.1752),
                      (1.0, 2.0, 0.1999), (2.0, 2.0, 0.2325), (1.0, 10.0, 0.2046),
                      (10.0, 10.0, 0.2498)]:
    v = ST.I_coin(m, n, 1.0)
    chk(f"I_coin(m={m}, n={n}) = {attendu}", presque(v, attendu, 5e-4), f"calculé {v:.5f}")

print()
print(" Cas limites et symétries")
chk("I_coin → 0,25 quand z → 0", presque(ST.I_coin(1, 1, 1e-9), 0.25, 1e-9))
chk("I_centre → 1 quand z → 0", presque(ST.I_position(2, 3, 1e-9, "centre"), 1.0, 1e-6))
chk("I_bord → 0,5 quand z → 0", presque(ST.I_position(2, 3, 1e-9, "bord_long"), 0.5, 1e-6))
chk("I_angle ≡ I_coin", presque(ST.I_position(2, 3, 1.5, "angle"), ST.I_coin(2, 3, 1.5), 1e-12))
chk("symétrie I_coin(B,L) = I_coin(L,B)",
    presque(ST.I_coin(2, 5, 1.7), ST.I_coin(5, 2, 1.7), 1e-12))
chk("décroissance monotone de I avec z",
    all(ST.I_position(3, 4, z, "centre") > ST.I_position(3, 4, z + 0.25, "centre")
        for z in [0.25 * i for i in range(1, 40)]))
chk("ordre centre > bord > angle à toute profondeur",
    all(ST.I_position(3, 5, z, "centre") > ST.I_position(3, 5, z, "bord_long")
        > ST.I_position(3, 5, z, "angle") for z in (0.3, 1, 2, 5, 10, 25)))

# Piège de branche : atan simple donnerait un résultat NÉGATIF ici.
# La frontière pour m = n est m = sqrt(1+sqrt(2)) ≈ 1,5538.
chk("branche de l'arctangente (m=n=2) : résultat positif",
    ST.I_coin(2, 2, 1.0) > 0.23, f"I = {ST.I_coin(2, 2, 1.0):.6f}")
chk("I_coin borné par 0,25 pour tout m,n",
    all(ST.I_coin(m, n, 1.0) <= 0.2500001
        for m in (0.1, 1, 5, 50, 500) for n in (0.1, 1, 5, 50, 500)))

print()
print(" Champ lointain = charge ponctuelle 3Q/(2πz²)")
for z in (30.0, 60.0, 150.0):
    B = L = 2.0
    exact = 3.0 * B * L / (2 * math.pi * z ** 2)
    calc = ST.delta_sigma(1.0, B, L, z, "centre")
    chk(f"z = {z:.0f} m", abs(calc - exact) / exact < 0.02,
        f"Newmark {calc:.4e} / ponctuel {exact:.4e}")

print()
print(" Intégrale de I : forme analytique EXACTE")
print("   ∫I_centre dz = (2B/π)·[asinh(L/B) + (L/B)·asinh(B/L)]")
print("   (identité : cette intégrale vaut Is·B, facteur de tassement à ν = 0)")


def integrale_exacte(B, L):
    return (2.0 * B / math.pi) * (math.asinh(L / B) + (L / B) * math.asinh(B / L))


def integrale_num(B, L, zmax=600.0, dz=5e-4):
    tot, z = 0.0, dz / 2.0
    while z < zmax:
        tot += ST.I_position(B, L, z, "centre") * dz
        z += dz
    return tot + 3.0 * B * L / (2.0 * math.pi * zmax)   # queue analytique


for B, L in ((1.0, 1.0), (1.0, 2.0), (2.0, 3.0), (1.0, 10.0)):
    e, n_ = integrale_exacte(B, L), integrale_num(B, L)
    chk(f"∫I dz — rectangle {B:g}×{L:g}", abs(n_ - e) / e < 2e-4,
        f"numérique {n_:.6f} / exact {e:.6f}")
chk("carré : ∫I dz = (4/π)·ln(1+√2) = 1,1222·B",
    presque(integrale_exacte(1, 1), 4 * math.log(1 + math.sqrt(2)) / math.pi, 1e-12))
chk("cohérence avec Is publié du carré souple (1,12)",
    presque(integrale_exacte(1, 1), 1.12, 3e-3), f"{integrale_exacte(1,1):.5f}")

# =====================================================================
print()
print("=" * 72)
print(" 2. MODULES ÉLASTIQUES")
print("=" * 72)
chk("M = E(1−ν)/((1+ν)(1−2ν)) — ν = 0 donne M = E",
    presque(ST.module_oedometrique(10.0, 0.0), 10.0, 1e-12))
chk("ν = 1/3 : M = 1,5·E", presque(ST.module_oedometrique(12.0, 1 / 3), 18.0, 1e-9))
chk("ν = 0,30 : M = 1,3462·E", presque(ST.module_oedometrique(10.0, 0.30), 13.4615, 1e-3))
chk("M ≥ E sur tout [0 ; 0,5[",
    all(ST.module_oedometrique(10.0, nu) >= 10.0 - 1e-12
        for nu in [i / 100 for i in range(0, 50)]))
chk("réciprocité module_young ∘ module_oedometrique",
    all(presque(ST.module_young(ST.module_oedometrique(25.0, nu), nu), 25.0, 1e-9)
        for nu in (0.0, 0.15, 0.30, 0.45)))
chk("ν ≥ 0,5 renvoie 0 (garde-fou, pas de division par zéro)",
    ST.module_oedometrique(10.0, 0.5) == 0.0 and ST.module_oedometrique(10.0, 0.6) == 0.0)

# =====================================================================
print()
print("=" * 72)
print(" 3. TASSEMENT ET COEFFICIENT DE RÉACTION")
print("=" * 72)
# Colonne confinée : contrainte uniforme sur H -> k = M/H exactement.
# On l'obtient en forçant B et L très grands devant H (I ≈ 1).
couches = [{"h": 2.0, "gamma": 20.0, "M": 10.0, "nom": "test"}]
r = ST.tassement(couches, 2000.0, 2000.0, 100.0, D=0.0, critere=0.0,
                 dz=0.005, q_net=False)
chk("colonne confinée : k = M/H = 5 MN/m³",
    presque(r["k_MNm3"], 5.0, 0.02), f"{r['k_MNm3']:.4f}")
chk("colonne confinée : w = q·H/M = 20 mm",
    presque(r["w_mm"], 20.0, 0.1), f"{r['w_mm']:.3f}")

# Ressorts en série (contrainte uniforme) : 1/k = Σh/M
couches2 = [{"h": 2.0, "gamma": 20.0, "M": 10.0}, {"h": 3.0, "gamma": 20.0, "M": 30.0}]
r2 = ST.tassement(couches2, 4000.0, 4000.0, 100.0, D=0.0, critere=0.0,
                  dz=0.005, q_net=False)
attendu = 1.0 / (2.0 / 10.0 + 3.0 / 30.0)
chk(f"deux couches en série : k = {attendu:.4f} MN/m³",
    presque(r2["k_MNm3"], attendu, 0.02), f"{r2['k_MNm3']:.4f}")

# k ∝ 1/B : propriété de premier principe non négociable
prof = [{"h": 80.0, "gamma": 19.0, "M": 30.0}]
ks = [(B, ST.tassement(prof, B, B, 150.0, critere=0.0, dz=0.05,
                       q_net=False, z_max_mult=30.0)["k_MNm3"])
      for B in (1.0, 2.0, 4.0)]
chk("k décroît quand B croît", ks[0][1] > ks[1][1] > ks[2][1],
    " ; ".join(f"B={b:g} k={k:.2f}" for b, k in ks))
chk("k ≈ proportionnel à 1/B (doublement de B → k/2, ±8 %)",
    all(0.42 < ks[i + 1][1] / ks[i][1] < 0.58 for i in range(2)),
    " ; ".join(f"{ks[i+1][1]/ks[i][1]:.3f}" for i in range(2)))

# Zonage : le centre tasse plus que les bords, donc k y est plus faible
z = ST.k_zone([{"h": 25.0, "gamma": 19.0, "M": 20.0}], 4.0, 6.0, 120.0, D=0.0)
chk("k centre < k bord < k angle",
    z["centre"]["k_MNm3"] < z["bord_long"]["k_MNm3"] < z["angle"]["k_MNm3"],
    " ; ".join(f"{p}={z[p]['k_MNm3']:.2f}" for p in ("centre", "bord_long", "angle")))
chk("tassement centre > tassement angle",
    z["centre"]["w_mm"] > z["angle"]["w_mm"])

# Profondeur d'influence : plus le critère est strict, plus on descend
prof3 = [{"h": 60.0, "gamma": 19.0, "M": 25.0}]
z10 = ST.tassement(prof3, 3.0, 3.0, 150.0, critere=0.10, dz=0.05)["z_influence"]
z20 = ST.tassement(prof3, 3.0, 3.0, 150.0, critere=0.20, dz=0.05)["z_influence"]
z30 = ST.tassement(prof3, 3.0, 3.0, 150.0, critere=0.30, dz=0.05)["z_influence"]
chk("profondeur d'influence : critère 10 % > 20 % > 30 %", z10 > z20 > z30,
    f"{z10:.2f} / {z20:.2f} / {z30:.2f} m")
chk("un k plus profond donne un k plus faible",
    ST.tassement(prof3, 3.0, 3.0, 150.0, critere=0.10)["k_MNm3"]
    < ST.tassement(prof3, 3.0, 3.0, 150.0, critere=0.30)["k_MNm3"])

# Pression nette
rn = ST.tassement(prof3, 3.0, 3.0, 150.0, D=2.0, q_net=True)
rb = ST.tassement(prof3, 3.0, 3.0, 150.0, D=2.0, q_net=False)
chk("pression nette < pression brute → tassement plus faible",
    rn["w_mm"] < rb["w_mm"] and presque(rn["q_net_kPa"], 150.0 - 19.0 * 2.0, 1e-6),
    f"q_net = {rn['q_net_kPa']:.1f} kPa")

# Le niveau d'assise écarte réellement les couches supérieures (défaut F2)
mou = [{"h": 2.0, "gamma": 17.0, "M": 1.5, "nom": "vase"},
       {"h": 30.0, "gamma": 20.0, "M": 50.0, "nom": "sable"}]
sans_D = ST.tassement(mou, 2.0, 2.0, 150.0, D=0.0, q_net=False)
avec_D = ST.tassement(mou, 2.0, 2.0, 150.0, D=2.0, q_net=False)
chk("fonder SOUS une couche molle augmente fortement k",
    avec_D["k_MNm3"] > 3 * sans_D["k_MNm3"],
    f"D=0 : {sans_D['k_MNm3']:.2f} → D=2 m : {avec_D['k_MNm3']:.2f} MN/m³")

# --- k ne doit PAS dépendre de q (modèle élastique linéaire) ---
# Défaut trouvé en revue : le critère d'arrêt était testé AVANT
# d'accumuler la tranche et sans plancher de profondeur ; k passait de
# 7,4 à 62,6 puis à 0,00 quand q diminuait, en affichant du vert.
sol_lin = [{"h": 25.0, "gamma": 19.0, "M": 25.0, "nom": "sable"}]
ks_q = [ST.tassement(sol_lin, 4.0, 4.0, float(q), D=3.0)["k_MNm3"]
        for q in (300, 200, 150, 120, 100, 80, 70, 60)]
chk("k reste positif quel que soit q (même faiblement chargé)",
    all(v > 0 for v in ks_q), " ; ".join(f"{v:.2f}" for v in ks_q))
chk("k varie peu avec q (rapport < 1,5)",
    max(ks_q) / min(ks_q) < 1.5, f"{min(ks_q):.2f} à {max(ks_q):.2f}")
chk("aucune discontinuité de k avec q",
    all(abs(a - b) / max(a, b) < 0.15 for a, b in zip(ks_q, ks_q[1:])),
    " ; ".join(f"{v:.2f}" for v in ks_q))
chk("plancher d'intégration : au moins une largeur de fondation",
    ST.tassement(sol_lin, 4.0, 4.0, 60.0, D=3.0)["z_influence"] >= 4.0 - 1e-9,
    f"{ST.tassement(sol_lin, 4.0, 4.0, 60.0, D=3.0)['z_influence']:.2f} m")

# --- une couche sans module n'est PAS incompressible ---
troue = [{"h": 2.0, "gamma": 19.0, "M": None, "nom": "non classé"},
         {"h": 18.0, "gamma": 19.0, "M": 20.0, "nom": "sable"}]
plein = [{"h": 2.0, "gamma": 19.0, "M": 20.0, "nom": "remblai"},
         {"h": 18.0, "gamma": 19.0, "M": 20.0, "nom": "sable"}]
rt = ST.tassement(troue, 3.0, 3.0, 200.0)
chk("profil troué → refus de calculer (et non un k flatteur)",
    rt["k_MNm3"] == 0.0 and "incomplet" in rt["convergence"], rt["convergence"])
chk("le refus indique l'épaisseur manquante", rt["h_sans_module"] > 1.9,
    f"{rt['h_sans_module']:.2f} m")
chk("le même profil complété se calcule normalement",
    ST.tassement(plein, 3.0, 3.0, 200.0)["k_MNm3"] > 0)

# Robustesse
chk("charge nulle → pas de division par zéro",
    ST.tassement(prof3, 2.0, 2.0, 0.0)["k_MNm3"] == 0.0)
chk("q inférieur au poids des terres → message explicite",
    "nette" in ST.tassement(prof3, 2.0, 2.0, 10.0, D=3.0, q_net=True)["convergence"])

# =====================================================================
print()
print("=" * 72)
print(" 4. CLASSIFICATION CPT (Robertson)")
print("=" * 72)
CAS = [("sable dense", 15.0, 75.0, 160.0, 111.0, 1.31, 2.05),
       ("limon", 2.0, 50.0, 57.0, 57.0, 2.05, 2.95),
       ("argile molle", 0.8, 28.0, 102.0, 53.0, 2.60, 3.60),
       ("tourbe", 0.3, 15.0, 22.0, 7.3, 2.95, 4.20)]
for nom, qc, fs, sv0, svp0, ic_lo, ic_hi in CAS:
    Ic, Qtn, Fr, n, conv = ST.ic_robertson(qc * 1000.0, fs, sv0, svp0)
    zone, lib = ST.sbt_from_ic(Ic)
    chk(f"Ic — {nom} dans [{ic_lo} ; {ic_hi}]", ic_lo <= Ic <= ic_hi,
        f"Ic = {Ic:.3f} → {lib}")
    chk(f"convergence de n — {nom}", conv and 0.0 <= n <= 1.0, f"n = {n:.3f}")

chk("plafond CN actif près de la surface (sinon tourbe mal classée)",
    ST.ic_robertson(300.0, 15.0, 22.0, 7.3)[0] >
    ST.ic_robertson(300.0, 15.0, 22.0, 7.3, cn_max=None)[0],
    f"avec plafond {ST.ic_robertson(300.,15.,22.,7.3)[0]:.2f} / "
    f"sans {ST.ic_robertson(300.,15.,22.,7.3,cn_max=None)[0]:.2f}")
chk("frottement nul → pas de classification (au lieu d'un Ic faux)",
    ST.ic_robertson(5000.0, 0.0, 100.0, 60.0)[0] is None)
chk("qt ≤ σv0 → pas de classification", ST.ic_robertson(50.0, 10.0, 100.0, 60.0)[0] is None)
chk("bornes SBT strictement croissantes",
    all(ST.SBT_ZONES[i][0] < ST.SBT_ZONES[i + 1][0] for i in range(len(ST.SBT_ZONES) - 1)))
chk("Ic croissant → zone décroissante (sable vers argile)",
    [ST.sbt_from_ic(v)[0] for v in (1.2, 1.8, 2.3, 2.8, 3.2, 3.9)] == [7, 6, 5, 4, 3, 2])

print()
print(" Module oedométrique déduit du CPT")
# α_M grenu : formule de Robertson, coefficient 0,0188 (et non 0,03)
chk("α_M(Ic = 1,615) ≈ 6,96 (coefficient 0,0188)",
    presque(ST.alpha_M_robertson(1.615, 100.0), 6.9568, 1e-3),
    f"{ST.alpha_M_robertson(1.615, 100.0):.4f}")
chk("α_M plafonné à 14 pour les sols fins",
    ST.alpha_M_robertson(3.0, 250.0) == 14.0)
chk("α_M = Qt sous le plafond", presque(ST.alpha_M_robertson(3.0, 9.0), 9.0, 1e-12))
for nom, qc, fs, sv0, svp0, lo, hi in [("sable dense", 15.0, 75.0, 160.0, 111.0, 40.0, 130.0),
                                       ("argile raide", 3.0, 108.0, 240.0, 142.0, 15.0, 45.0),
                                       ("tourbe", 0.3, 15.0, 22.0, 7.3, 0.3, 5.0)]:
    Ic, Qtn, Fr, n, _ = ST.ic_robertson(qc * 1000.0, fs, sv0, svp0)
    Mb, Mh = ST.modules_depuis_cpt(Ic, qc * 1000.0, sv0, svp0, qc, Qtn=Qtn)
    chk(f"M encadré — {nom} dans [{lo} ; {hi}] MPa",
        Mb is not None and lo <= Mb <= hi and lo <= Mh <= hi and Mb <= Mh,
        f"[{Mb:.1f} ; {Mh:.1f}]")
chk("borne basse ≤ borne haute pour tous les cas",
    all((lambda t: t[0] is None or t[0] <= t[1])(
        ST.modules_depuis_cpt(*(lambda a: (a[0], q * 1000.0, s, sp, q))(
            ST.ic_robertson(q * 1000.0, f, s, sp)), Qtn=ST.ic_robertson(q * 1000.0, f, s, sp)[1]))
        for q, f, s, sp in ((15.0, 75.0, 160.0, 111.0), (2.0, 50.0, 57.0, 57.0),
                            (0.8, 28.0, 102.0, 53.0), (0.3, 15.0, 22.0, 7.3))))

# =====================================================================
print()
print("=" * 72)
print(" 5. AUTRES THÉORIES")
print("=" * 72)
chk("Terzaghi sable : k plaque > k semelle", ST.k_terzaghi_taille(50.0, 2.0, nature="sable") < 50.0)
chk("Terzaghi sable B = 0,30 m redonne la valeur de plaque",
    presque(ST.k_terzaghi_taille(50.0, 0.30, nature="sable"), 50.0, 1e-9))
chk("Terzaghi argile B = 0,30 m redonne la valeur de plaque",
    presque(ST.k_terzaghi_taille(50.0, 0.30, nature="argile"), 50.0, 1e-9))
chk("Terzaghi argile : k ∝ 1/B",
    presque(ST.k_terzaghi_taille(60.0, 6.0, nature="argile") * 2,
            ST.k_terzaghi_taille(60.0, 3.0, nature="argile"), 1e-9))
chk("correction de forme : semelle filante = 0,667 × carré",
    presque(ST.k_terzaghi_taille(50.0, 2.0, L=1e6, nature="sable") /
            ST.k_terzaghi_taille(50.0, 2.0, L=2.0, nature="sable"), 2.0 / 3.0, 1e-3))
chk("Vesić sans raideur = E/(B(1−ν²))",
    presque(ST.k_vesic(25.0, 2.0, 0.30), 25.0 / (2.0 * (1 - 0.09)), 1e-9))
chk("k_elastique ∝ 1/B",
    presque(ST.k_elastique(25.0, 4.0) * 2, ST.k_elastique(25.0, 2.0), 1e-9))

# =====================================================================
print()
print("=" * 72)
print(" 6. DÉCOUPAGE AUTOMATIQUE EN COUCHES")
print("=" * 72)
pts = []
z = 0.02
while z <= 18.0:
    if z < 4.0:
        qc, rf = 2.0, 2.6
    elif z < 9.0:
        qc, rf = 13.0, 0.6
    else:
        qc, rf = 3.2, 3.4
    pts.append((round(z, 2), qc, rf / 100.0 * qc * 1000.0))
    z += 0.02
cou = ST.profil_depuis_cpt(pts, nappe_m=3.0)
chk("trois couches retrouvées", len(cou) == 3, f"{len(cou)} couches")
if len(cou) == 3:
    chk("limite 1 ≈ 4,0 m", presque(cou[0]["z1"], 4.0, 0.25), f"{cou[0]['z1']:.2f}")
    chk("limite 2 ≈ 9,0 m", presque(cou[1]["z1"], 9.0, 0.25), f"{cou[1]['z1']:.2f}")
    chk("couche 2 identifiée sableuse", cou[1]["Ic"] < 2.05, f"Ic = {cou[1]['Ic']:.2f}")
    chk("couche 2 la plus raide", cou[1]["M_haut"] > cou[0]["M_haut"]
        and cou[1]["M_haut"] > cou[2]["M_haut"])
chk("couches jointives et croissantes",
    all(presque(a["z1"], b["z0"], 1e-9) for a, b in zip(cou, cou[1:]))
    and cou[0]["z0"] == 0.0)
chk("épaisseurs cohérentes",
    all(presque(c["z1"] - c["z0"], c["h"], 1e-9) for c in cou))
chk("profil vide → liste vide", ST.profil_depuis_cpt([]) == [])

# =====================================================================
print()
print("=" * 72)
print(f" RÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
print("=" * 72)
for nom, info in KO:
    print("   -", nom, "|", info)
sys.exit(0 if not KO else 1)

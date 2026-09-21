# -*- coding: utf-8 -*-
"""Tirage aléatoire d'un jeu d'entrées réaliste — partagé par les tests
d'oracle (parité numérique et parité des dessins)."""
from acier.bibliotheques import DB, PERSO, ORI_P, ORI_S


def tirage(rng):
    """Un jeu d'entrées réaliste, tiré au hasard."""
    u = {}
    if rng.random() < 0.25:
        u["id_projet"] = rng.choice(["Halle A", "Bât. 12", "Passerelle", ""])
        u["id_rep"] = rng.choice(["A-1", "P3/S7", "rep 42", ""])
    u["mode_calc"] = "PRÉDIMENSIONNEMENT" if rng.random() < 0.2 else "VÉRIFICATION"
    u["fix_P"] = rng.choice(["Boulonnée", "Boulonnée", "Boulonnée", "Soudée"])
    u["fix_S"] = rng.choice(["Boulonnée", "Boulonnée", "Boulonnée", "Soudée"])
    for X in ("P", "S"):
        if rng.random() < 0.12:
            u["prof_" + X] = PERSO
            h = rng.choice([250, 300, 360, 420, 500, 547, 620, 800])
            u["h" + X + "_u"] = h
            u["b" + X + "_u"] = rng.choice([120, 150, 180, 210, 250, 300])
            u["tw" + X + "_u"] = rng.choice([6, 7.5, 9, 10.5, 12, 15])
            u["tf" + X + "_u"] = rng.choice([9, 11, 13.5, 15.7, 19, 25, 30])
            u["r" + X + "_u"] = rng.choice([12, 15, 18, 21, 24, 27])
        else:
            u["prof_" + X] = rng.choice(DB["profils"])["n"]
        u["nu_" + X] = rng.choice([a["n"] for a in DB["aciers"]] + [PERSO])
    u["nu_C"] = rng.choice([a["n"] for a in DB["aciers"]] + [PERSO])
    if PERSO in (u["nu_P"], u["nu_S"], u["nu_C"]):
        u["fy_u"] = rng.choice([235, 275, 355, 420, 460])
        u["fu_u"] = u["fy_u"] + rng.choice([80, 120, 155, 175])
        u["bw_u"] = rng.choice([0.8, 0.85, 0.9, 1.0])
    u["d_top"] = rng.choice([0, 0, 0, 10, 20, 40, 60, 100])
    u["d_nt"] = rng.choice([0, 0, 30, 40, 50, 60, 80])
    u["d_nb"] = rng.choice([0, 0, 0, 0, 30, 40, 50])
    u["l_n"] = rng.choice([0, 80, 100, 120, 150, 180, 220, 260])
    u["r_n"] = rng.choice([8, 10, 12])
    u["g_h"] = rng.choice([5, 10, 10, 15, 20])
    u["lt_ok"] = rng.choice(["Oui", "Oui", "Non"])
    if rng.random() < 0.15:
        u["corn_u"] = PERSO
        u["k1_u"] = rng.choice([70, 90, 100, 120, 150, 160])
        u["k2_u"] = rng.choice([70, 75, 90, 100, 120])
        u["kt_u"] = rng.choice([7, 8, 9, 10, 12, 15])
        u["kr_u"] = rng.choice([9, 10, 11, 12, 13, 16])
    else:
        u["corn_u"] = rng.choice(DB["cornieres"])["n"]
    u["orient"] = rng.choice([ORI_P, ORI_S])
    u["LC_u"] = rng.choice([100, 130, 160, 190, 220, 260, 300, 360, 430, 500])
    u["z_C"] = rng.choice([30, 40, 50, 60, 80, 100, 120])
    u["boulon_u"] = rng.choice(DB["boulons"])["n"]
    u["classe"] = rng.choice(DB["classes"])["n"]
    u["trou"] = rng.choice(["Normal", "Normal", "Normal", "Surdimensionné"])
    if u["trou"] == "Surdimensionné" and rng.random() < 0.8:
        d = [b for b in DB["boulons"] if b["n"] == u["boulon_u"]][0]["d"]
        u["d0_u"] = d + rng.choice([3, 4, 6])
    u["filet"] = rng.choice(["Oui", "Oui", "Non"])
    u["cat"] = rng.choice(["A", "A", "A", "B", "C"])
    u["mu_s"] = rng.choice([0.2, 0.3, 0.4, 0.5])
    u["k_s"] = rng.choice([0.85, 1, 1])
    u["k_ser"] = rng.choice([0.6, 0.7, 1])
    for X, p2 in (("S", "p2_S"), ("P", "p2_P")):
        u[f"n1{X}_u"] = rng.choice([1, 2, 2, 3, 3, 3, 4, 5, 6, 8])
        n2 = rng.choice([1, 1, 1, 2])
        u[f"n2{X}_u"] = str(n2) if rng.random() < 0.5 else n2   # chaîne ou nombre
        u[f"p1{X}_u"] = rng.choice([40, 50, 55, 60, 65, 70, 80, 100])
        u[p2] = rng.choice([50, 60, 70, 80, 90])
        u[f"e1{X}_u"] = rng.choice([25, 30, 35, 40, 45, 50, 60])
    u["e2b_u"] = rng.choice([30, 35, 40, 45, 50, 60, 70])
    u["gA_u"] = rng.choice([40, 45, 50, 55, 60, 65, 70, 80])
    u["a_P"] = rng.choice([3, 4, 5, 6, 8]); u["lh_P"] = rng.choice([0, 0, 30, 40, 60])
    u["a_S"] = rng.choice([3, 4, 5, 6, 8]); u["lh_S"] = rng.choice([0, 30, 40, 40, 60])
    u["V_Ed"] = rng.choice([-300, 0, 25, 60, 125, 125, 200, 300, 450, 600])
    u["N_Ed"] = rng.choice([0, 0, 0, -120, -50, 30, 50, 150, 370])
    u["H_Ed"] = rng.choice([0, 0, 0, -20, 10, 20, 40, 80])
    u["M_Ed"] = rng.choice([0, 0, 0, -8, 3, 5, 12, 30])
    u["g_M0"] = rng.choice([1, 1, 1, 1.1]); u["g_M2n"] = rng.choice([1.25, 1.25, 1.1])
    u["g_M2"] = rng.choice([1.25, 1.25, 1.1]); u["g_M3"] = rng.choice([1.25, 1.1]); u["g_M3s"] = rng.choice([1.1, 1])
    u["opt_exc"] = rng.choice(["Non", "Non", "Oui"]); u["k_rot"] = rng.choice([0.8, 0.8, 1])
    u["opt_blf"] = rng.choice(["Non", "Non", "Oui"]); u["expo"] = rng.choice(["Non", "Non", "Oui"])
    u["eta_c"] = rng.choice([0.7, 0.8, 0.9, 1]); u["k_e1"] = rng.choice([1.2, 1.5, 1.6, 2])
    u["k_p1"] = rng.choice([2.2, 2.5, 2.7, 3, 3.5]); u["k_e2"] = rng.choice([1.2, 1.5, 1.8, 2.5])
    u["pd_dmin"] = rng.choice(DB["boulons"])["n"]; u["pd_dmax"] = rng.choice(DB["boulons"])["n"]
    return u

# -*- coding: utf-8 -*-
"""Outils communs de la lecture critique du classeur REALEX « Cornières ».

- ``classeur(...)`` recopie la feuille P8 FORMULE PAR FORMULE (chaque ligne cite sa cellule) : c'est la
  « méthode Excel historique », sans aucune correction ;
- la géométrie exacte du congé d'un L laminé (congé intérieur de rayon r tangent aux deux faces intérieures) ;
- les données du cas de contrôle de l'ingénieur.

Repère (celui du schéma image1.jpeg) : origine au talon extérieur de la cornière (coin bas-gauche, contre le mur),
x horizontal vers la charge (le long de l'aile horizontale), y vertical vers le haut (le long de l'aile verticale,
plaquée au mur). Le mur est le plan x = 0. Unités : mm, N, MPa, sauf mention (kN, kN·m comme le classeur).
"""
import math

# ------------------------------------------------------------------ le cas de contrôle (P8!B3…C17, C71…C76)
CAS = dict(larg=120.0, haut=120.0, e=15.0, r=13.0, b_cm=30.0, gM=1.5, R=42.76, X=75.0, Xj=0.0,
           Y=85.0, Yj=0.0, fy=355.0, n_anc=3, d_anc=16.0, jeu_anc_cm=1.0, R_elu=68.9)
# résultats historiques donnés par l'ingénieur
HISTO = dict(M1=2.00972, M2=2.56560, M3=2.35870, M4=2.89665, taux=0.8886)


def classeur(larg, haut, e, r, b_cm, gM, R, X, Xj, Y, Yj, fy, n_anc=3, d_anc=16.0, jeu_anc_cm=1.0,
             R_elu=68.9):
    """La feuille P8, cellule par cellule. Unités du classeur : mm sauf b_cm (C8, en cm) et jeu_anc_cm (C76,
    en cm) ; R en kN ; M en kN·m ; σ en MPa."""
    c = {}
    c["C19_Xe"] = X - r - e                      # P8!C19 = $C$13-$C$7-$C$6
    c["C20_Xer"] = X - e                         # P8!C20 = $C$13-$C$6
    c["C23_Ye"] = Y - e - r                      # P8!C23 = $C$16-$C$6-$C$7
    c["C24_Yer"] = Y - e                         # P8!C24 = C16-$C$6
    c["C25_YH"] = Y - e / 2                      # P8!C25 = $C$16-$C$6/2
    c["A20_larg_moins_X"] = larg - X             # P8!A20 = C4-C13 (calculé, jamais utilisé)
    c["A21_haut_moins_Y"] = haut - Y             # P8!A21 = C5-C16 (calculé ; repris par C90)
    b = b_cm * 10                                # « $C$8*10 » : la longueur reprise passe de cm en mm
    adm = fy / gM                                # P8!H31 = C3/$C$10
    tadm = fy / gM / math.sqrt(3)                # P8!H34 = C3/$C$10/SQRT(3)
    c["adm"], c["tadm"], c["b"] = adm, tadm, b

    def sect(nom, t, M, V):
        W = b * t ** 2 / 6                       # C32, C42, C52, C62 = $C$8*10*t^2/6
        s = M * 1e6 / W                          # G31, G41, G51, G61 = F*10^6/W
        tau = V * 1000 / (b * t)                 # G34, G44, G54, G64 = R*1000/(C8*10*t)
        eq = (s ** 2 + 3 * tau ** 2) ** 0.5      # G37, G47, G57, G67
        c[nom] = dict(t=t, W=W, M=M, sigma=s, tau=tau, sigma_eq=eq, taux_sigma=s / adm, taux_tau=tau / tadm,
                      taux=eq / adm)

    # section 1 — aile horizontale, épaisseur e, bras Xe + Xj
    sect("S1", e, R * ((c["C19_Xe"] + Xj) / 1000), R)              # C35 = $C$12*(($C$33+$C$34)/1000)
    # section 2 — aile horizontale, « e + r », bras Xe+r + Xj
    sect("S2", e + r, R * ((c["C20_Xer"] + Xj) / 1000), R)         # C45 = $C$12*(($C$43+$C$44)/1000)
    MMax = (X + Xj) / 1000 * R                                     # C55 = C63 = C78 = ($C$13+$C$14)/1000*$C$12
    c["MMax"] = MMax
    c["C56_rapport3"] = (c["C23_Ye"] + Yj) / (c["C25_YH"] + Yj)    # C56 = ($C$23+$C$17)/($C$25+$C$17)
    c["C64_rapport4"] = (c["C24_Yer"] + Yj) / (c["C25_YH"] + Yj)   # C64 = ($C$24+$C$17)/($C$25+$C$17)
    sect("S3", e, c["C56_rapport3"] * MMax, R)                     # C57 = C56*C55
    sect("S4", e + r, c["C64_rapport4"] * MMax, R)                 # C65 = C64*C63
    c["D3_taux_max"] = max(c[k]["taux"] for k in ("S1", "S2", "S3", "S4"))   # D3 = MAX(I37,I47,I57,I67)
    # ancrages (P8!C70…C90)
    C75 = Y / 10                                                   # C75 = C16/10 (cm)
    c["z_ancrage_mm"] = (C75 * 0.01 - jeu_anc_cm * 0.01) * 1000   # le bras de C82, en mm
    c["N_ELS"] = MMax / (C75 * 0.01 - jeu_anc_cm * 0.01) / n_anc   # C82 = C78/(C75*0.01-C76*0.01)/C73
    c["V_ELS"] = R / n_anc                                         # C83 = C70/C73
    c["N_ELU"] = (R_elu / R) * c["N_ELS"]                          # C85 = (C71/C70)*C82
    c["V_ELU"] = (R_elu / R) * c["V_ELS"]                          # C86 = (C71/C70)*C83
    c["C88_pression"] = c["V_ELS"] * 1e3 / (d_anc * e)             # C88 = C83*10^3/(C74*C6)
    c["C90_arrachement"] = c["V_ELS"] * 1000 / (2 * (haut - Y) * e) if haut != Y else float("inf")  # C90
    return c


# ------------------------------------------------------------------ géométrie exacte du congé
def epaisseur_verticale(x, e, r):
    """Hauteur de matière de l'aile horizontale sur la verticale d'abscisse x (mm), congé exact.
    Centre du congé (e+r, e+r), rayon r ; la matière est SOUS l'arc pour e ≤ x ≤ e+r."""
    if x >= e + r:
        return e
    if x <= e:
        raise ValueError("x ≤ e : on est dans l'aile verticale, la coupe verticale n'est plus une coupe d'aile")
    u = (e + r) - x
    return (e + r) - math.sqrt(r * r - u * u)


def coupe_radiale(alpha_deg, e, r):
    """Coupe radiale issue du centre du congé, d'angle alpha (0° = vers le mur, horizontale ; 90° = vers le bas,
    verticale). Rend (P, Q, t) : point de l'arc, point de sortie sur la face extérieure (x = 0 ou y = 0), longueur."""
    a = math.radians(alpha_deg)
    cx = cy = e + r
    dx, dy = -math.cos(a), -math.sin(a)
    P = (cx + r * dx, cy + r * dy)
    sx = (cx / math.cos(a)) if math.cos(a) > 1e-12 else float("inf")    # distance du centre au plan x = 0
    sy = (cy / math.sin(a)) if math.sin(a) > 1e-12 else float("inf")    # distance du centre au plan y = 0
    s = min(sx, sy)
    Q = (cx + s * dx, cy + s * dy)
    return P, Q, s - r


def fmt(x, n=2):
    """Nombre au format du bureau (virgule décimale)."""
    return f"{x:.{n}f}".replace(".", ",")

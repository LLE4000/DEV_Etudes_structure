# -*- coding: utf-8 -*-
"""Validation croisée INDÉPENDANTE du moteur « Charges sur plancher » (lot 4).

Hors moteur et hors chemin de production (numpy, matplotlib pour les
figures). Cinq contrôles, chacun par une méthode ÉCRITE À PART de celle du
moteur — grilles fines, éléments finis de poutre, formules reprises de la
norme ou du Heft 240 :

 A. UNE ROUE SUR UN HOURDIS — une roue de nacelle à 2 m d'un appui ; le
    moment ELU (γG sur les poids propres, γQ sur les charges) est cherché sur
    une grille de 7 201 sections, le tranchant aux appuis par statique ; les
    éléments non chargés ne portent que leur poids propre et les charges
    présentes.
 B. UN STABILISATEUR SUR TÔLE À CHEVAL SUR DEUX HOURDIS — bande plastique
    c = t·√(f/(3·p_Ed)) itérée en point fixe (EN 1993-1-8 §6.2.5), aire
    efficace, règle du levier (moitié-moitié sur le joint), moments des deux
    éléments.
 C. UN MANITOU SUR UNE DALLE DE 20 CM — largeurs efficaces du Heft 240
    (t_y, b_v) écrites ici, superposition des quatre roues par une grille
    transversale au pas de 5 mm et une grille longitudinale de 7 201 sections,
    effets locaux de l'EN 1992-1-1 (§6.7 et §6.4) recalculés depuis la norme.
 D. FILIÈRE SUR RESSORTS — le moteur résout une poutre d'Euler sur appuis
    élastiques par la méthode des déplacements sur ses seuls nœuds utiles ;
    ici des ÉLÉMENTS FINIS de poutre (Hermite cubique, 400 éléments), sur
    des configurations tirées au sort : réactions, moment maximal et flèches
    aux nœuds (le moteur compte w positif vers le bas, les éléments finis
    vers le haut).
 E. CRÉNEAUX — la superposition des largeurs efficaces (maximum sur la
    largeur) contre une évaluation exhaustive aux arêtes, sur des boîtes
    tirées au sort, plancher et fenêtres de bande.

Produit : un rapport daté dans docs/validation/ et trois figures. Cette
page est une MESURE : elle ne se réécrit pas (R6) ; une nouvelle campagne
produit un nouveau rapport daté.

Lancement : python3 tools/validation_charges_plancher.py [--rapide]
"""
import math
import os
import random
import sys
import time
from datetime import date

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
os.chdir(RACINE)

import numpy as np  # noqa: E402

from plancher.charges import bibliotheque as BIB  # noqa: E402
from plancher.charges import diffusion as DF  # noqa: E402
from plancher.charges import modele as M  # noqa: E402
from plancher.charges import repartiteurs as RP  # noqa: E402
from plancher.charges import verification as V  # noqa: E402
from plancher.charges.parametres import Parametres  # noqa: E402

GG, GQ = 1.35, 1.5                  # coefficients partiels de l'ELU (P-2), écrits ici à part du moteur
INK, ACC, KO, OK, ATT, MUT = "#15181F", "#33415C", "#9C3341", "#2E7D46", "#D98A00", "#6E7480"
DOSSIER = os.path.join("docs", "validation")
FIG = os.path.join(DOSSIER, "figures")
# le plancher d'essai des cas A et B : résistances d'ESSAI (pas un catalogue), nature Rd
L, NEL, BEL, H_EL, G_EL, Q_PRES, M_RD, V_RD = 7.2, 8, 1.2, 0.20, 2.9, 1.0, 60.0, 80.0


def fr(x, n):
    """Nombre en écriture française (espace des milliers, virgule)."""
    return f"{x:,.{n}f}".replace(",", " ").replace(".", ",")


def ecart_rel(a, b):
    return abs(a - b) / max(1e-12, abs(b))


# ----------------------------------------------------------------- statique
def m_uniforme(w, x, L_):
    return w * x * (L_ - x) / 2.0


def m_ponctuel(P, a, x, L_):
    return P * x * (L_ - a) / L_ if x <= a else P * a * (L_ - x) / L_


def reactions(P, a, L_):
    return P * (L_ - a) / L_, P * a / L_


def hourdis_essai(**kw):
    base = dict(nom="SP 200 essai", L=L, n_elements=NEL, b_element=BEL, h=H_EL, g=G_EL, q_presentes=Q_PRES,
                M_Rd=M_RD, V_Rd=V_RD, nature=M.RD, I=1.05e-3, etat=M.CLAVETES, p_loc_adm=0.7)
    base.update(kw)
    return M.Hourdis(**base)


# ============================================================ A · une roue sur un hourdis
def cas_A():
    P, phi, a, b, xr, yr = 12.0, 1.4, 0.15, 0.10, 2.0, 3.0      # roue de nacelle, à 2 m de l'appui A, sur H3
    H = hourdis_essai()
    res = V.verifier(H, [M.Empreinte("R1", xr, yr, a, b, P, phi=phi)], {})
    e3 = next(e for e in res["elements"] if e["nom"] == "H3")
    # --- indépendant
    wg, wq, Pd = G_EL * BEL, Q_PRES * BEL, P * phi
    xs = np.linspace(0.0, L, 7201)
    m_ed = GG * np.array([m_uniforme(wg, x, L) for x in xs]) + GQ * np.array([m_uniforme(wq, x, L) + m_ponctuel(Pd, xr, x, L) for x in xs])
    i = int(np.argmax(m_ed))
    RA, RB = reactions(Pd, xr, L)
    v_ed = max(GG * wg * L / 2 + GQ * (wq * L / 2 + RA), GG * wg * L / 2 + GQ * (wq * L / 2 + RB))
    m_vide = GG * wg * L * L / 8 + GQ * wq * L * L / 8
    lignes = [("M_Ed de H3 (kN·m)", e3["M_max"], float(m_ed[i]), 5e-4),
              ("abscisse du maximum (m)", e3["x_M"], float(xs[i]), 0.02),
              ("V_Ed de H3 (kN)", e3["V_max"], v_ed, 1e-9),
              ("η_M de H3", e3["taux_M"], float(m_ed[i]) / M_RD, 5e-4),
              ("η_V de H3", e3["taux_V"], v_ed / V_RD, 1e-9),
              ("M_Ed d'un élément non chargé, H1 (kN·m)", next(e["M_max"] for e in res["elements"] if e["nom"] == "H1"), m_vide, 1e-9)]
    return {"lignes": lignes, "xs": xs, "m_ed": m_ed, "x_charge": xr, "e3": e3,
            "m_a_la_charge": float(m_ed[int(round(xr / L * 7200))]), "m_mi_portee": float(m_ed[3600])}


# ============================================================ B · stabilisateur sur tôle à cheval
def cas_B():
    P, a, t, A, f = 95.0, 0.30, 20.0, 600.0, 235.0
    H = hourdis_essai()
    joint = 3 * BEL                                               # 3,60 m : joint H3 | H4
    e = M.Empreinte("P2", L / 2, joint, a, a, P)
    plat = M.Plateau("tôle", f, t, A, A)
    res = V.verifier(H, [e], {"P2": plat})
    pt = res["patches"][0]
    vp = next(v for v in res["verifications"] if v["famille"] == "plateau")
    # --- indépendant : point fixe sur c (EN 1993-1-8 §6.2.5), pression ELU
    c, a_mm = 0.0, a * 1e3
    for _ in range(200):
        a_eff = min(A, a_mm + 2 * c)
        p_ed = GQ * P * 1e3 / (a_eff * a_eff)
        c = min(t * math.sqrt(f / (3.0 * p_ed)), (A - a_mm) / 2.0)
    a_eff = min(A, a_mm + 2 * c)
    # règle du levier : l'empreinte efficace, centrée sur le joint, se partage par moitié
    part = 0.5
    wg, wq = G_EL * BEL, Q_PRES * BEL
    m_ed = max(GG * m_uniforme(wg, x, L) + GQ * (m_uniforme(wq, x, L) + m_ponctuel(part * P, L / 2, x, L)) for x in np.linspace(0, L, 7201))
    v_ed = GG * wg * L / 2 + GQ * (wq * L / 2 + part * P / 2)
    lignes = [("bande plastique c (mm)", vp["E"], c, 1e-6),
              ("aire efficace a_eff (mm)", pt["a"] * 1e3, a_eff, 1e-6),
              ("pression de service sous l'aire efficace (N/mm²)", pt["pression"], P * 1e3 / (a_eff * a_eff), 1e-6),
              ("moment de la part de H3 : M_Q − w_q·L²/8 = (P/2)·L/4 (kN·m)",
               next(x for x in res["elements"] if x["nom"] == "H3")["M_Q"] - m_uniforme(wq, L / 2, L), part * P * L / 4, 1e-9),
              ("M_Ed de H3 = M_Ed de H4 (kN·m)", next(x for x in res["elements"] if x["nom"] == "H3")["M_max"], m_ed, 5e-4),
              ("V_Ed de H3 (kN)", next(x for x in res["elements"] if x["nom"] == "H3")["V_max"], v_ed, 1e-9),
              ("H3 et H4 identiques (kN·m)", next(x for x in res["elements"] if x["nom"] == "H3")["M_max"],
               next(x for x in res["elements"] if x["nom"] == "H4")["M_max"], 1e-12)]
    return {"lignes": lignes, "c": c, "a_eff": a_eff}


# ============================================================ C · Manitou sur dalle 20 cm
def cas_C():
    D = M.DalleBA(nom="D20", L=L, largeur=9.6, h=0.20, d=0.17, beton="C30/37", As_inf=524.0, fyk=500.0, q_presentes=Q_PRES)
    gab = BIB.gabarits()["Télescopique à roues (type MT 1840) · en déplacement"]
    charges = {"R1": 60.0, "R2": 60.0, "R3": 40.0, "R4": 40.0}
    for ap in gab.appuis:
        ap.P = charges[ap.nom]
    eng = M.Engin("E1", gab, L / 2, 4.8, 0)
    res = V.verifier(D, [eng], {})
    vf = next(v for v in res["verifications"] if v["famille"] == "flexion")
    vt = next(v for v in res["verifications"] if v["famille"] == "tranchant")
    # --- indépendant : positions des roues (angle 0 : dx le long de la portée), Heft 240, créneaux sur grille
    roues = [(L / 2 + ap.dx, 4.8 + ap.dy, ap.a, ap.b, ap.P * ap.phi) for ap in gab.appuis]
    B, h, h1 = D.largeur, D.h, D.h1
    w = 25.0 * h + Q_PRES
    ys = np.arange(0.0, B + 1e-9, 0.005)

    def profil(valeurs_largeurs):
        """Somme des créneaux (valeur / largeur sur [y − l/2, y + l/2], tronqué au plancher et concentré)."""
        tot = np.zeros_like(ys)
        for y, larg, val in valeurs_largeurs:
            lo, hi = max(0.0, y - larg / 2), min(B, y + larg / 2)
            tot[(ys >= lo - 1e-9) & (ys <= hi + 1e-9)] += val / (hi - lo)
        return tot

    xs = np.linspace(0.0, L, 7201)
    m_max, x_m = 0.0, 0.0
    for x in xs:
        boites = []
        for (xi, yi, a, b, Pi) in roues:
            ty = min(b + 2 * h1 + h, 0.8 * L)
            boites.append((yi, ty, m_ponctuel(Pi, xi, x, L)))
        m = profil(boites).max() + m_uniforme(w, x, L)
        if m > m_max:
            m_max, x_m = m, x
    vA = profil([(yi, min(b + 2 * h1 + h, 0.8 * L) + 0.5 * xi, reactions(Pi, xi, L)[0]) for (xi, yi, a, b, Pi) in roues]).max() + w * L / 2
    vB = profil([(yi, min(b + 2 * h1 + h, 0.8 * L) + 0.5 * (L - xi), reactions(Pi, xi, L)[1]) for (xi, yi, a, b, Pi) in roues]).max() + w * L / 2
    # --- effets locaux depuis la norme, sur la roue la plus chargée (R1)
    xi, yi, a, b, P1 = roues[0]
    fck, gc = 30.0, 1.5
    fcd = fck / gc
    a_mm, b_mm, h_mm, d_mm = a * 1e3, b * 1e3, h * 1e3, D.d * 1e3
    b2, d2 = min(3 * a_mm, a_mm + h_mm), min(3 * b_mm, b_mm + h_mm)
    F_Rdu = min(a_mm * b_mm * fcd * math.sqrt(b2 * d2 / (a_mm * b_mm)), 3.0 * fcd * a_mm * b_mm) / 1e3
    u1 = 2 * (a_mm + b_mm) + 2 * math.pi * 2 * d_mm
    v_Ed = GQ * P1 * 1e3 / (u1 * d_mm)
    k = min(1 + math.sqrt(200.0 / d_mm), 2.0)
    rho = 524.0 / (1000.0 * d_mm)
    v_Rdc = max(0.18 / gc * k * (100 * rho * fck) ** (1 / 3), 0.035 * k ** 1.5 * math.sqrt(fck))
    locaux = [v for v in res["verifications"] if v["famille"] == "local" and v["objet"].endswith("E1.R1")]
    pl = next(v for v in locaux if v["objet"].startswith("pression localisée"))
    po = next(v for v in locaux if v["objet"].startswith("poinçonnement sous"))
    lignes = [("M par mètre en service, bande gouvernante (kN·m/m)", vf["E"], m_max, 2e-3),
              ("abscisse du maximum (m)", vf["x"], x_m, 0.02),
              ("V par mètre en service à l'appui le plus chargé (kN/m)", vt["E"], max(vA, vB), 1e-6),
              ("§6.7 : F_Rdu sous R1 (kN)", pl["R"], F_Rdu, 1e-9),
              ("§6.7 : γQ·P sous R1 (kN)", pl["E"], GQ * P1, 1e-9),
              ("§6.4 : v_Ed sous R1 (N/mm²)", po["E"], v_Ed, 1e-9),
              ("§6.4 : v_Rd,c (N/mm²)", po["R"], v_Rdc, 1e-9)]
    # profil transversal à la section gouvernante, pour la figure
    boites = [(yi_, min(b_ + 2 * h1 + h, 0.8 * L), m_ponctuel(Pi_, xi_, x_m, L)) for (xi_, yi_, a_, b_, Pi_) in roues]
    return {"lignes": lignes, "ys": ys, "profil": profil(boites) + m_uniforme(w, x_m, L), "x_m": x_m, "roues": roues, "vf": vf, "B": B}


# ============================================================ D · filière sur ressorts
def poutre_ef(EI, Lf, ressorts, charges, n_el=400):
    """Éléments finis de poutre d'Euler-Bernoulli (Hermite cubique), extrémités
    libres, ressorts et charges aux nœuds (le maillage passe par leurs
    abscisses). Rend (positions des nœuds, flèches, réactions, M aux nœuds)."""
    pts = sorted({0.0, Lf} | {s for s, _ in ressorts} | {s for s, _ in charges})
    noeuds = [0.0]
    for p0, p1 in zip(pts, pts[1:]):
        n = max(1, int(round((p1 - p0) / Lf * n_el)))
        noeuds.extend(np.linspace(p0, p1, n + 1)[1:].tolist())
    x = np.array(noeuds)
    N = len(x)
    K = np.zeros((2 * N, 2 * N))
    for i in range(N - 1):
        le = x[i + 1] - x[i]
        ke = EI / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                      [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        idx = [2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3]
        K[np.ix_(idx, idx)] += ke
    F = np.zeros(2 * N)
    for s, k in ressorts:
        j = int(np.argmin(np.abs(x - s)))
        K[2 * j, 2 * j] += k
    for s, P in charges:
        j = int(np.argmin(np.abs(x - s)))
        F[2 * j] -= P
    u = np.linalg.solve(K, F)
    w = u[0::2]
    R = [k * (-w[int(np.argmin(np.abs(x - s)))]) for s, k in ressorts]
    # moments aux nœuds par les efforts d'extrémité des éléments
    Mn = np.zeros(N)
    for i in range(N - 1):
        le = x[i + 1] - x[i]
        ke = EI / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                      [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        fe = ke @ u[2 * i:2 * i + 4]
        Mn[i + 1] = fe[3]          # moment à l'extrémité droite de l'élément
    return x, w, R, Mn


def configurations_D(n, graine):
    rng = random.Random(graine)
    out = []
    for _ in range(n):
        Lf = rng.uniform(2.0, 6.0)
        EI = 10 ** rng.uniform(2.0, 5.0)
        nr = rng.randint(2, 6)
        ress = sorted({round(rng.uniform(0.0, Lf), 3) for _ in range(nr)})
        if len(ress) < 2:
            ress = [0.0, Lf]
        ressorts = [(s, 10 ** rng.uniform(2.0, 5.0)) for s in ress]
        nc = rng.randint(1, 3)
        charges = [(round(rng.uniform(0.0, Lf), 3), rng.uniform(10.0, 100.0)) for _ in range(nc)]
        out.append((Lf, EI, ressorts, charges))
    return out


def cas_D(n, graine=20261102):
    lignes, pires = [], (0.0, 0.0, 0.0)
    exemple = None
    for Lf, EI, ressorts, charges in configurations_D(n, graine):
        res = RP.poutre_sur_ressorts(EI, Lf, ressorts, charges)
        x, w, R, Mn = poutre_ef(EI, Lf, ressorts, charges)
        sP = sum(P for _, P in charges)
        dR = max(abs(a - b) for (_, a), b in zip(res["reactions"], R)) / sP
        dM = ecart_rel(res["M_max"], float(np.max(np.abs(Mn))))
        # flèches : le moteur compte w positif VERS LE BAS (R = k·w), les éléments finis vers le haut
        w_ef = np.interp(res["x"], x, -w)
        dW = float(np.max(np.abs(np.array(res["w"]) - w_ef)) / max(1e-12, np.max(np.abs(w_ef))))
        pires = (max(pires[0], dR), max(pires[1], dM), max(pires[2], dW))
        lignes.append((Lf, EI, len(ressorts), len(charges), dR, dM, dW))
        if exemple is None:
            exemple = (x, w, res, ressorts, charges, Lf)
    return {"lignes": lignes, "pires": pires, "exemple": exemple}


# ============================================================ E · créneaux
def max_exhaustif(boites, y_min, y_max, fenetre=None):
    """Évaluation aux arêtes (± 1e-9), tronquée au plancher avec concentration."""
    inter = []
    for yc, w, val in boites:
        lo, hi = max(y_min, yc - w / 2), min(y_max, yc + w / 2)
        if hi - lo <= 1e-12:
            lo, hi = max(y_min, min(yc, y_max)), max(y_min, min(yc, y_max)) + 1e-9
        inter.append((lo, hi, val / (hi - lo)))
    a, b = fenetre if fenetre else (y_min, y_max)
    best = 0.0
    for y in sorted({lo for lo, _, _ in inter} | {hi for _, hi, _ in inter} | {a, b}):
        for eps in (1e-9, -1e-9):
            yy = y + eps
            if a <= yy <= b:
                best = max(best, sum(v for lo, hi, v in inter if lo <= yy <= hi))
    return best


def cas_E(n, graine=20261102):
    rng = random.Random(graine)
    pire, n_fen = 0.0, 0
    for _ in range(n):
        B = rng.choice([4.8, 7.2, 9.6])
        boites = [(rng.uniform(0.0, B), rng.uniform(0.2, 2.5), rng.uniform(1.0, 100.0)) for _ in range(rng.randint(1, 5))]
        m, _ = DF.max_creneaux(boites, 0.0, B)
        pire = max(pire, ecart_rel(m, max_exhaustif(boites, 0.0, B)))
        a = rng.uniform(0.0, B - 1.0)
        mf, _ = DF.max_creneaux(boites, 0.0, B, fenetre=(a, a + 1.0))
        pire = max(pire, ecart_rel(mf, max_exhaustif(boites, 0.0, B, (a, a + 1.0))) if mf > 0 else abs(mf - max_exhaustif(boites, 0.0, B, (a, a + 1.0))))
        n_fen += 1
    return {"pire": pire, "n": n, "n_fen": n_fen}


# ------------------------------------------------------------------- figures
def _style(ax, titre):
    ax.set_title(titre, fontsize=10, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUT)
    ax.grid(True, color="#E5E7EE", lw=0.6)


def figures(A, C, D, chemins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # A : diagramme de M_Ed, les deux anciens points, le maximum
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(A["xs"], A["m_ed"], color=ACC, lw=1.6, label="M_Ed(x), grille de 7 201 sections")
    ax.plot([A["x_charge"]], [A["m_a_la_charge"]], "o", color=MUT, label="à la charge (x = 2,00 m)")
    ax.plot([L / 2], [A["m_mi_portee"]], "s", color=MUT, label="à mi-portée")
    ax.plot([A["e3"]["x_M"]], [A["e3"]["M_max"]], "D", color=KO, label=f"moteur : {fr(A['e3']['M_max'], 2)} kN·m en x = {fr(A['e3']['x_M'], 2)} m")
    ax.set_xlabel("x (m)", fontsize=8, color=MUT)
    ax.set_ylabel("M_Ed (kN·m)", fontsize=8, color=MUT)
    _style(ax, "A · une roue sur un hourdis : le maximum tombe entre la charge et la mi-portée")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(chemins[0], dpi=140)
    plt.close(fig)
    # C : profil transversal des créneaux à la section gouvernante
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(C["ys"], C["profil"], color=ACC, lw=1.4, label=f"m(y) à x = {fr(C['x_m'], 2)} m, grille de 5 mm")
    ax.axhline(C["vf"]["E"], color=KO, lw=1.0, ls="--", label=f"moteur : {fr(C['vf']['E'], 2)} kN·m/m")
    for (xi, yi, a, b, Pi) in C["roues"]:
        ax.axvline(yi, color=MUT, lw=0.6, ls=":")
    ax.set_xlabel("y (m)", fontsize=8, color=MUT)
    ax.set_ylabel("m (kN·m/m)", fontsize=8, color=MUT)
    _style(ax, "C · Manitou sur dalle : superposition des roues en travers (Heft 240, b_m = t_y)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(chemins[1], dpi=140)
    plt.close(fig)
    # D : flèche et réactions d'un exemple
    x, w, res, ressorts, charges, Lf = D["exemple"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(x, -w * 1e3, color=ACC, lw=1.4, label="flèche, éléments finis (400 éléments)")
    # le moteur compte w positif vers le bas, les éléments finis vers le haut : les deux sont tracés vers le bas
    ax.plot(res["x"], np.array(res["w"]) * 1e3, "o", ms=3, color=KO, label="flèche aux nœuds du moteur")
    for s, _ in ressorts:
        ax.axvline(s, color=MUT, lw=0.6, ls=":")
    for s, P in charges:
        ax.axvline(s, color=KO, lw=0.6, ls="--", alpha=0.6)
        ax.annotate(f"{fr(P, 0)} kN", (s, 0.97), xycoords=("data", "axes fraction"), fontsize=7, color=KO, ha="center", va="top")
    ax.set_xlabel("s (m)", fontsize=8, color=MUT)
    ax.set_ylabel("w (mm, vers le bas)", fontsize=8, color=MUT)
    _style(ax, f"D · filière de {fr(Lf, 2)} m sur {len(ressorts)} ressorts : moteur contre éléments finis")
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(chemins[2], dpi=140)
    plt.close(fig)


# -------------------------------------------------------------------- rapport
def pourcent(v):
    """Pourcentage lisible : quatre décimales jusqu'à 10⁻⁴ %, puissance de dix en dessous
    (une tolérance de 10⁻⁹ s'écrirait sinon « 0,0000 % », ce qui est faux)."""
    if v == 0:
        return "0 %"
    if v * 100 >= 1e-4:
        return f"{fr(v * 100, 4)} %"
    ex = int(math.floor(math.log10(v * 100) + 1e-9))
    mant = v * 100 / 10 ** ex
    sup = str(-ex).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return (f"{fr(mant, 0)}·" if abs(mant - 1) > 1e-6 else "") + f"10⁻{sup} %"


def tableau(lignes):
    out = ["| Grandeur | Moteur | Indépendant | Écart | Tolérance |", "|---|---|---|---|---|"]
    ok = True
    for nom, a, b, tol in lignes:
        e = ecart_rel(a, b) if abs(b) > 1e-9 else abs(a - b)
        ok = ok and e <= tol
        out.append(f"| {nom} | {fr(a, 3)} | {fr(b, 3)} | {pourcent(e)} | {pourcent(tol)} |")
    return out, ok


def main(rapide=False):
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    nD, nE = (6, 60) if rapide else (24, 300)
    A, Bc, C, D, E = cas_A(), cas_B(), cas_C(), cas_D(nD), cas_E(nE)
    tA, okA = tableau(A["lignes"])
    tB, okB = tableau(Bc["lignes"])
    tC, okC = tableau(C["lignes"])
    okD = D["pires"][0] <= 5e-3 and D["pires"][1] <= 1e-2 and D["pires"][2] <= 1e-2
    okE = E["pire"] <= 1e-6
    verdict = okA and okB and okC and okD and okE
    sfx = "_rapide" if rapide else ""
    chemins = [os.path.join(FIG, f"plancher_validation_A_roue{sfx}.png"), os.path.join(FIG, f"plancher_validation_C_dalle{sfx}.png"),
               os.path.join(FIG, f"plancher_validation_D_filiere{sfx}.png")]
    figures(A, C, D, chemins)
    duree = time.time() - t0
    jour = date.today().strftime("%Y-%m-%d")
    Lr = [f"# Validation croisée des charges sur plancher — {jour}", "",
          "*Lot 4. Méthodes indépendantes du moteur (numpy, hors chemin de production) : grilles fines de sections, "
          "superposition des créneaux sur grille, point fixe de la bande plastique, éléments finis de poutre, formules de "
          "l'EN 1992-1-1 et du Heft 240 réécrites à part. Script : `tools/validation_charges_plancher.py`. Cette page est "
          "une MESURE datée : elle ne se réécrit pas.*", "",
          f"**Verdict : {'VALIDÉ' if verdict else 'NON VALIDÉ'}** — cinq contrôles, {nD} filières et {nE} jeux de créneaux tirés au sort "
          f"(graine 20261102), {fr(duree, 0)} s" + (" (mode rapide)" if rapide else "") + ".", "",
          "Les cas A à C emploient des résistances d'ESSAI (M_Rd 60 kN·m, V_Rd 80 kN par hourdis ; dalle de 20 cm à 524 mm²/m), "
          "pas un catalogue : ils valident les EFFORTS, les largeurs efficaces, les répartiteurs et les effets locaux, pas une "
          "valeur de fabricant.", "",
          "## A · Une roue sur un hourdis", "",
          "Roue de nacelle 150 × 100 mm, 12 kN, φ 1,4, à 2,00 m de l'appui A sur H3 d'un plancher de 8 hourdis de 1,20 m, portée 7,20 m, "
          "g 2,90 kN/m², charges présentes 1,00 kN/m². M_Ed = 1,35·M_G + 1,5·M_Q cherché sur 7 201 sections ; V par statique.", ""]
    Lr += tA + ["",
                f"Le maximum tombe en x = {fr(A['e3']['x_M'], 2)} m : {fr(A['e3']['M_max'], 2)} kN·m, contre {fr(A['m_a_la_charge'], 2)} à la charge "
                f"et {fr(A['m_mi_portee'], 2)} à mi-portée — c'est le défaut non conservatif corrigé le 02/11/2026 (sections au pas L/60).", "",
                "## B · Un stabilisateur sur tôle à cheval sur deux hourdis", "",
                "Semelle 300 × 300 mm, 95 kN, à mi-portée, centrée sur le joint H3 | H4 ; tôle S235 600 × 600 × 20. Bande plastique "
                "itérée en point fixe sous la pression ELU, aire efficace, règle du levier (moitié-moitié), moments des deux éléments.", ""]
    Lr += tB + ["", "## C · Un Manitou sur une dalle de 20 cm", "",
                "Gabarit MT 1840 (roues 300 × 250 mm) à (3,60 ; 4,80) m sur une dalle 7,20 × 9,60 m, h 200 mm, d 170 mm, C30/37, "
                "As 524 mm²/m ; charges d'essai 60 · 60 · 40 · 40 kN. Heft 240 : t_y = b + 2·h1 + h, b_v = t_y + 0,5·a, b_m = t_y ; "
                "superposition des quatre roues sur une grille transversale de 5 mm, 7 201 sections ; §6.7 et §6.4 depuis la norme.", ""]
    Lr += tC + ["", "## D · Filière sur ressorts : moteur contre éléments finis", "",
                f"{nD} configurations tirées au sort (longueur 2–6 m, EI 10²–10⁵ kN·m², 2–6 ressorts de 10²–10⁵ kN/m, 1–3 charges "
                "de 10–100 kN) ; éléments finis d'Euler-Bernoulli, Hermite cubique, 400 éléments.", "",
                "| Grandeur | Écart maximal | Tolérance |", "|---|---|---|",
                f"| Réaction d'un ressort (rapportée à Σ P) | {fr(D['pires'][0] * 100, 4)} % | 0,5 % |",
                f"| Moment maximal | {fr(D['pires'][1] * 100, 4)} % | 1 % |",
                f"| Flèche aux nœuds du moteur (rapportée à la flèche maximale) | {fr(D['pires'][2] * 100, 4)} % | 1 % |", "",
                "| # | L (m) | EI (kN·m²) | ressorts | charges | écart R / Σ P | écart M_max | écart w |", "|---|---|---|---|---|---|---|---|"]
    for i, (Lf, EI, nr, nc, dR, dM, dW) in enumerate(D["lignes"], 1):
        Lr.append(f"| {i} | {fr(Lf, 2)} | {fr(EI, 0)} | {nr} | {nc} | {fr(dR * 100, 4)} % | {fr(dM * 100, 4)} % | {fr(dW * 100, 4)} % |")
    Lr += ["", "## E · Créneaux : maximum sur la largeur et fenêtres de bande", "",
           f"{nE} jeux de 1 à 5 boîtes tirés au sort sur des planchers de 4,80, 7,20 et 9,60 m, avec pour chacun une fenêtre de bande "
           "d'un mètre ; évaluation exhaustive aux arêtes, troncature au plancher avec concentration.", "",
           "| Grandeur | Écart maximal | Tolérance |", "|---|---|---|", f"| Maximum des créneaux (plancher et fenêtres) | {fr(E['pire'] * 100, 6)} % | 0,0001 % |", "",
           "## Figures", "", f"![Roue sur hourdis](figures/plancher_validation_A_roue{sfx}.png)", "",
           f"![Manitou sur dalle](figures/plancher_validation_C_dalle{sfx}.png)", "", f"![Filière](figures/plancher_validation_D_filiere{sfx}.png)", "",
           "## Ce que cette validation ne couvre pas", "",
           "- les formules béton armé de l'application (M_adm, V_adm du module Poutre) : elles sont reproduites, pas jugées ;",
           "- le coefficient k du Heft 240 en flexion, les parts de l'annexe C de l'EN 1168 et la résistance du joint : non transcrits, "
           "le moteur les signale « à transcrire » ;",
           "- les valeurs de catalogue des hourdis : les cas d'essai en emploient de fictives ;",
           "- l'hypothèse du tranchant à 100 % sur l'élément chargé (P-3) et celle des éléments indépendants sous une filière : décisions du bureau, pas des mesures."]
    chemin = os.path.join(DOSSIER, f"charges_plancher_validation_{jour}{sfx}.md")
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write("\n".join(Lr) + "\n")
    print("\n".join(Lr[:8]))
    print("…\nrapport :", chemin, "| verdict :", "VALIDÉ" if verdict else "NON VALIDÉ", f"| {duree:.0f} s")
    for nom, ok in (("A", okA), ("B", okB), ("C", okC), ("D", okD), ("E", okE)):
        print(f"   {nom} : {'OK' if ok else 'KO'}")
    return verdict


if __name__ == "__main__":
    sys.exit(0 if main(rapide="--rapide" in sys.argv) else 1)

# -*- coding: utf-8 -*-
"""Validation croisée INDÉPENDANTE du moteur de pied de poteau (lot C).

Hors moteur et hors chemin de production (numpy + scipy ; le moteur reste en
bibliothèque standard). Trois contrôles, chacun par une méthode numérique
DIFFÉRENTE de celle du moteur :

 A. RÉPARTITION ÉLASTIQUE — le moteur découpe des polygones et intègre
    exactement (Green) ; ici une GRILLE de ressorts (cellules de l'aire
    efficace au pas de 4 mm, sans traction) et des tiges-ressorts, Newton sur
    le plan de pénétration (a, bx, by) avec jacobien analytique. Comparés :
    résultante C, sa position, pression maximale, effort de chaque tige.
 B. DOMAINE RÉSISTANT (niveau 2) — le moteur balaie l'axe neutre avec un champ
    « tout ou rien » ; ici une PROGRAMMATION LINÉAIRE (scipy, HiGHS) sur les
    cellules (0 ≤ p ≤ fjd) et les tiges (0 ≤ T ≤ FT,Rd), équilibre de N :
    fonction d'appui h(φ) = max (Mx·cos φ + My·sin φ) et capacité sur le rayon
    r(φ) = max t tel que (Mx, My) = t·(cos φ, sin φ). Le domaine exact d'un
    champ statiquement admissible borné EST la solution de ce programme : le
    balayage du moteur doit le retrouver.
 C. NIVEAU 1 ≤ NIVEAU 2 — la capacité de la Table 6.7 avec interaction
    linéaire, dans chaque direction, reste sous le rayon du programme linéaire
    (borne statique) ; la capacité sur le rayon du moteur est intérieure (sûre)
    et proche de celle du programme linéaire.

Produit : un rapport daté dans docs/validation/ et trois figures.

Lancement : python3 tools/validation_pied_poteau.py [--rapide]
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
from scipy.optimize import linprog  # noqa: E402

from acier.assemblages.pied_poteau.platine_beton import moteur, equilibre as EQ  # noqa: E402
from acier.assemblages.pied_poteau.platine_beton.entrees import FORME_RHS, PERIM_TOUT  # noqa: E402

PAS_A, PAS_B = 4.0, 5.0            # pas de la grille (mm) : répartition, programme linéaire
INK, ACC, KO, OK, ATT, MUT = "#15181F", "#33415C", "#9C3341", "#2E7D46", "#D98A00", "#6E7480"
DOSSIER = os.path.join("docs", "validation")
FIG = os.path.join(DOSSIER, "figures")


# ----------------------------------------------------------------- grille
def cellules(rects, pas):
    """Centres et aires des cellules de l'aire efficace (pas ≈ ``pas``)."""
    X, Y, A = [], [], []
    for (x0, y0, x1, y1) in rects:
        nx = max(1, int(round((x1 - x0) / pas)))
        ny = max(1, int(round((y1 - y0) / pas)))
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        xs = x0 + dx * (np.arange(nx) + 0.5)
        ys = y0 + dy * (np.arange(ny) + 0.5)
        gx, gy = np.meshgrid(xs, ys)
        X.append(gx.ravel()); Y.append(gy.ravel()); A.append(np.full(gx.size, dx * dy))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(A)


def coins(rects, pas):
    """Les coins des rectangles (où une fonction linéaire est extrémale)."""
    pts = [(x, y) for (x0, y0, x1, y1) in rects for (x, y) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def winkler(rects, tiges, kc, kb, N, Mx, My, pas=PAS_A):
    """Répartition élastique sur une grille de ressorts : Newton avec
    jacobien analytique (le résidu est linéaire par morceaux)."""
    X, Y, dA = cellules(rects, pas)
    xb = np.array([t[0] for t in tiges]); yb = np.array([t[1] for t in tiges])
    cx, cy = coins(rects, pas)
    A_tot = float(dA.sum())
    u = np.array([N / (kc * A_tot) if (N > 0 and A_tot > 0) else 1e-3, 0.0, 0.0])
    cibles = np.array([N, Mx, My])
    ech = np.array([max(abs(N), 1e3), max(abs(Mx), 1e5), max(abs(My), 1e5)])

    def resid_jac(u):
        a, bx, by = u
        w = a + bx * X + by * Y
        m = w > 0
        wb = a + bx * xb + by * yb
        mb = wb < 0
        F = kc * np.sum(w[m] * dA[m]); Sx = kc * np.sum(w[m] * X[m] * dA[m]); Sy = kc * np.sum(w[m] * Y[m] * dA[m])
        T = np.where(mb, -kb * wb, 0.0)
        r = np.array([F - T.sum() - N, Sy - np.sum(T * yb) - Mx, Sx - np.sum(T * xb) - My])
        # jacobien : d/da, d/dbx, d/dby
        J = np.zeros((3, 3))
        for col, (g, gb) in enumerate(((np.ones_like(X), np.ones_like(xb)), (X, xb), (Y, yb))):
            J[0, col] = kc * np.sum(g[m] * dA[m]) + kb * np.sum(gb[mb])
            J[1, col] = kc * np.sum(g[m] * Y[m] * dA[m]) + kb * np.sum(gb[mb] * yb[mb])
            J[2, col] = kc * np.sum(g[m] * X[m] * dA[m]) + kb * np.sum(gb[mb] * xb[mb])
        return r, J

    def norme(r):
        return float(np.max(np.abs(r) / ech))

    r, J = resid_jac(u)
    for _ in range(200):
        if norme(r) < 1e-10:
            break
        du = np.linalg.solve(J, -r)
        pas_, n0 = 1.0, norme(r)
        for _ in range(40):
            un = u + pas_ * du
            rn, Jn = resid_jac(un)
            if norme(rn) < n0:
                u, r, J = un, rn, Jn
                break
            pas_ *= 0.5
        else:
            break
    a, bx, by = u
    w = a + bx * X + by * Y
    m = w > 0
    C = kc * np.sum(w[m] * dA[m])
    xC = kc * np.sum(w[m] * X[m] * dA[m]) / C if C > 0 else 0.0
    yC = kc * np.sum(w[m] * Y[m] * dA[m]) / C if C > 0 else 0.0
    wc = a + bx * cx + by * cy
    pmax = kc * float(np.max(wc)) if np.any(wc > 0) else 0.0
    T = np.maximum(0.0, -kb * (a + bx * xb + by * yb))
    return dict(ok=norme(r) < 1e-6, C=C, xC=xC, yC=yC, pmax=pmax, T=T, a=a, bx=bx, by=by, X=X, Y=Y, w=w, dA=dA)


# ------------------------------------------------------- programme linéaire
def lp_appui(X, Y, dA, tiges, FtRd, N, fjd, phi):
    """h(φ) = max (Mx·cos φ + My·sin φ) sur le domaine des champs admissibles."""
    xb = np.array([t[0] for t in tiges]); yb = np.array([t[1] for t in tiges])
    n, nb = X.size, xb.size
    cphi, sphi = math.cos(phi), math.sin(phi)
    # variables : p_i (MPa) puis T_j (N) ; Mx = Σ p dA y − Σ T y ; My = Σ p dA x − Σ T x
    c = -np.concatenate([dA * (cphi * Y + sphi * X), -(cphi * yb + sphi * xb)])
    A_eq = np.concatenate([dA, -np.ones(nb)])[None, :]
    bounds = [(0.0, fjd)] * n + [(0.0, float(f)) for f in FtRd]
    res = linprog(c, A_eq=A_eq, b_eq=[N], bounds=bounds, method="highs")
    return (-res.fun if res.success else None), res


def lp_rayon(X, Y, dA, tiges, FtRd, N, fjd, phi):
    """r(φ) = max t tel que (Mx, My) = t·(cos φ, sin φ) — la capacité sur le rayon."""
    xb = np.array([t[0] for t in tiges]); yb = np.array([t[1] for t in tiges])
    n, nb = X.size, xb.size
    cphi, sphi = math.cos(phi), math.sin(phi)
    c = np.zeros(n + nb + 1); c[-1] = -1.0
    A_eq = np.zeros((3, n + nb + 1))
    A_eq[0, :n] = dA; A_eq[0, n:n + nb] = -1.0
    A_eq[1, :n] = dA * Y; A_eq[1, n:n + nb] = -yb; A_eq[1, -1] = -cphi
    A_eq[2, :n] = dA * X; A_eq[2, n:n + nb] = -xb; A_eq[2, -1] = -sphi
    bounds = [(0.0, fjd)] * n + [(0.0, float(f)) for f in FtRd] + [(0.0, None)]
    res = linprog(c, A_eq=A_eq, b_eq=[N, 0.0, 0.0], bounds=bounds, method="highs")
    return (res.x[-1] if res.success else None), res


def appui_polygone(dom, phi):
    """Fonction d'appui du polygone du moteur (sommets en kNm → N·mm)."""
    return max(1e6 * (mx * math.cos(phi) + my * math.sin(phi)) for mx, my in dom)


def niveau1_rayon(R, phi):
    """Capacité du niveau 1 sur le rayon φ : interaction linéaire des capacités uniaxiales."""
    kx = "x+" if math.cos(phi) >= 0 else "x-"
    ky = "y+" if math.sin(phi) >= 0 else "y-"
    s = 0.0
    if abs(math.cos(phi)) > 1e-12:
        s += abs(math.cos(phi)) / max(R.MRd1[kx] * 1e6, 1e-9)
    if abs(math.sin(phi)) > 1e-12:
        s += abs(math.sin(phi)) / max(R.MRd1[ky] * 1e6, 1e-9)
    return 1.0 / s if s > 0 else 0.0


# ------------------------------------------------------------ configurations
PROFILS = ["HEA 300", "HEB 240", "IPE 400", "HEA 200", "HEB 400", "HEM 240"]


def configurations(n, graine=20261031):
    rng = random.Random(graine)
    out = [("défaut (exemple du bureau)", dict(perim=PERIM_TOUT)),
           ("massif large, soulèvement", dict(N_Ed=-150, Mx_Ed=60, My_Ed=20, bc_u=1400, lc_u=1400, hc_u=800)),
           ("RHS 250×250×10, six tiges", dict(forme=FORME_RHS, prof_c="Personnalisé", h_u=250, b_u=250, t_u=10, bp_u=450,
                                               lp_u=450, tp_u=20, n_ay=3, w_x=350, w_y=360, N_Ed=200, Mx_Ed=40, My_Ed=25,
                                               bc_u=1200, lc_u=1200, hc_u=800)),
           ("HEB 400, 2 × 3 tiges M30, N fort", dict(prof_c="HEB 400", bp_u=700, lp_u=800, tp_u=40, w_x=560, w_y=660,
                                                      n_ay=3, ancr="M30", bc_u=1600, lc_u=1800, hc_u=1000, hef=500,
                                                      d_h=60, N_Ed=1200, Mx_Ed=350, My_Ed=60))]
    while len(out) < n:
        prof = rng.choice(PROFILS)
        bp = rng.choice([400, 450, 500, 600, 700]); lp = rng.choice([400, 500, 600, 700, 800])
        tp = rng.choice([15, 20, 25, 30, 40])
        n_ay = rng.choice([2, 2, 3, 4])
        wx = rng.choice([0.6, 0.7, 0.8]) * bp; wy = rng.choice([0.6, 0.7, 0.8]) * lp
        inp = dict(prof_c=prof, bp_u=bp, lp_u=lp, tp_u=tp, n_ay=n_ay, w_x=round(wx), w_y=round(wy),
                   ancr=rng.choice(["M20", "M24", "M30"]), N_Ed=rng.choice([-100, 50, 200, 400, 800, 1200]),
                   Mx_Ed=rng.choice([-120, -40, 0, 30, 80, 150]), My_Ed=rng.choice([-60, -20, 0, 15, 40, 90]),
                   bc_u=max(bp + 400, 1200), lc_u=max(lp + 400, 1200), hc_u=800, hef=rng.choice([250, 300, 400]))
        R = moteur.compute(inp)
        if not R.geo_ok or not R.eq_ok or not R.dist_ok:
            continue
        out.append((f"aléatoire : {prof}, {bp}×{lp}×{tp}, {2 * n_ay} tiges {inp['ancr']}, N = {inp['N_Ed']}, "
                    f"Mx = {inp['Mx_Ed']}, My = {inp['My_Ed']}", inp))
    return out


# ------------------------------------------------------------------- figures
def _style(ax, titre):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUT)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.set_title(titre, fontsize=9.5, color=INK, loc="left", pad=8)
    ax.grid(alpha=0.18, color=MUT, linewidth=0.5)


def figure_repartition(R, W, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.4))
    u = R.u
    # grille : pression par cellule
    p = np.where(W["w"] > 0, R.k_c * W["w"], np.nan)
    sc = a1.scatter(W["X"], W["Y"], c=p, s=5.5, marker="s", cmap="Reds", vmin=0, vmax=max(R.pmax, 1e-6), linewidths=0)
    a1.add_patch(Rectangle((-u.bp_u / 2, -u.lp_u / 2), u.bp_u, u.lp_u, fc="none", ec=INK, lw=1.0))
    for (_n, x0, y0, x1, y1) in R.parois:
        a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#C3CED7", ec=INK, lw=0.7, alpha=0.9))
    for poly in R.contact:
        a1.add_patch(Polygon(poly, closed=True, fc="none", ec=ACC, lw=1.3, ls="--"))
    for t, Tg in zip(R.tiges, W["T"]):
        a1.plot(t["x"], t["y"], "o", ms=7, mfc="#fff", mec=INK, mew=1)
        if t["T"] > 1e-6:
            a1.annotate(f"T = {fr(t['T'], 1)} kN (moteur)\n{fr(Tg / 1e3, 1)} kN (grille)", (t["x"], t["y"]),
                        (t["x"] + (14 if t["x"] <= 0 else -14), t["y"] + (16 if t["y"] <= 0 else -16)),
                        fontsize=7.5, color=INK, ha="left" if t["x"] <= 0 else "right",
                        va="bottom" if t["y"] <= 0 else "top")
    a1.plot([R.xC], [R.yC], "x", ms=9, mew=2, color=KO)
    a1.plot([W["xC"]], [W["yC"]], "+", ms=11, mew=1.4, color=ACC)
    a1.set_aspect("equal")
    a1.set_xlim(-u.bp_u / 2 - 40, u.bp_u / 2 + 40); a1.set_ylim(-u.lp_u / 2 - 40, u.lp_u / 2 + 40)
    _style(a1, f"A · Pression sur la grille de ressorts (pas {PAS_A:.0f} mm)\ncontour de contact du moteur en tireté")
    a1.text(0.0, -0.16, f"C : moteur {fr(R.C, 1)} kN en ({R.xC:.0f} ; {R.yC:.0f}) ×, grille {fr(W['C'] / 1e3, 1)} kN en "
                        f"({W['xC']:.0f} ; {W['yC']:.0f}) + · p max {fr(R.pmax, 2)} / {fr(W['pmax'], 2)} MPa",
            transform=a1.transAxes, fontsize=8, color=INK)
    cb = fig.colorbar(sc, ax=a1, shrink=0.8, pad=0.02); cb.set_label("p (MPa)", fontsize=8, color=MUT); cb.ax.tick_params(labelsize=7, colors=MUT)
    a1.set_xlabel("x (mm)", fontsize=8, color=MUT); a1.set_ylabel("y (mm)", fontsize=8, color=MUT)
    # profil de pression le long de l'axe de la paroi comprimée
    xs = np.linspace(-u.bp_u / 2, u.bp_u / 2, 400)
    ys = np.linspace(-u.lp_u / 2, u.lp_u / 2, 400)
    a_, bx_, by_ = R.axe
    pm = R.k_c * np.maximum(0.0, a_ + bx_ * 0 + by_ * ys)
    pg = W["w"]
    a2.plot(ys, pm, color=KO, lw=1.6, label="moteur : p(0, y) = kc·max(0, a + by·y)")
    sel = np.abs(W["X"]) < PAS_A
    if sel.any():
        o = np.argsort(W["Y"][sel])
        a2.plot(W["Y"][sel][o], R.k_c * np.maximum(0.0, pg[sel][o]), "o", ms=3.5, color=ACC, label="grille : cellules de la colonne x ≈ 0")
    a2.axhline(R.fjd, color=ATT, lw=1, ls="--", label=f"fjd = {fr(R.fjd, 1)} MPa")
    a2.set_xlabel("y (mm)", fontsize=8, color=MUT); a2.set_ylabel("p (MPa)", fontsize=8, color=MUT)
    a2.legend(fontsize=7.5, frameon=False)
    _style(a2, "Pression le long de l'axe x = 0 (âme)\nles deux méthodes")
    fig.suptitle("Validation A — répartition élastique : polygones exacts (moteur) contre grille de ressorts (indépendante)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(chemin, dpi=140)
    plt.close(fig)


def fr(x, n):
    """Nombre en écriture française (virgule)."""
    return f"{x:.{n}f}".replace(".", ",")


def figure_domaines(cas, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cas)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.2))
    if n == 1:
        axes = [axes]
    for ax, (nom, R, phis, hlp, rlp, rmot, m1) in zip(axes, cas):
        xs = [p[0] for p in R.dom] + [R.dom[0][0]]; ys = [p[1] for p in R.dom] + [R.dom[0][1]]
        ax.plot(xs, ys, color=ACC, lw=1.6, label="moteur : balayage de l'axe neutre (niveau 2)")
        ax.plot([r * math.cos(f) / 1e6 for r, f in zip(rlp, phis)], [r * math.sin(f) / 1e6 for r, f in zip(rlp, phis)],
                "o", ms=5, mfc="#fff", mec=KO, mew=1.3, label="programme linéaire : capacité sur le rayon r(φ)")
        ax.plot([r * math.cos(f) / 1e6 for r, f in zip(m1, phis)], [r * math.sin(f) / 1e6 for r, f in zip(m1, phis)],
                "s", ms=3.5, color=ATT, label="niveau 1 (Table 6.7, interaction linéaire)")
        M1 = R.MRd1
        ax.plot([M1["x+"], 0, -M1["x-"], 0, M1["x+"]], [0, M1["y+"], 0, -M1["y-"], 0], color=ATT, lw=0.9, ls="--")
        ax.plot([R.u.Mx_Ed], [R.u.My_Ed], "o", ms=6, color=INK, label="(Mx, My) appliqués")
        ax.axhline(0, color=MUT, lw=0.5); ax.axvline(0, color=MUT, lw=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("Mx (kNm)", fontsize=8, color=MUT); ax.set_ylabel("My (kNm)", fontsize=8, color=MUT)
        ecart = max(abs(a - b) / max(abs(b), 1) for a, b in zip(rmot, rlp))
        _style(ax, f"{nom}\nN = {R.u.N_Ed:.0f} kN · écart max moteur / PL sur le rayon : {fr(ecart * 100, 2)} %")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Validation B et C — domaine (Mx, My) : balayage du moteur contre programme linéaire (HiGHS), niveau 1 à l'intérieur",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(chemin, dpi=140)
    plt.close(fig)


def figure_ecarts(lignes_A, lignes_B, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.4))
    Cm = [l["C_m"] for l in lignes_A]; Cg = [l["C_g"] for l in lignes_A]
    a1.plot([0, max(Cm + [1])], [0, max(Cm + [1])], color=MUT, lw=0.8)
    a1.plot(Cm, Cg, "o", ms=5, mfc=ACC, mec="none", alpha=0.85)
    a1.set_xlabel("C moteur (kN)", fontsize=8, color=MUT); a1.set_ylabel("C grille (kN)", fontsize=8, color=MUT)
    _style(a1, f"A · Résultante comprimée, {len(lignes_A)} configurations\nécart max {fr(max(l['dC'] for l in lignes_A) * 100, 3)} %")
    Tm = [t for l in lignes_A for t in l["T_m"]]; Tg = [t for l in lignes_A for t in l["T_g"]]
    a2.plot([0, max(Tm + [1])], [0, max(Tm + [1])], color=MUT, lw=0.8)
    a2.plot(Tm, Tg, "o", ms=4, mfc=KO, mec="none", alpha=0.75)
    a2.set_xlabel("T moteur (kN)", fontsize=8, color=MUT); a2.set_ylabel("T grille (kN)", fontsize=8, color=MUT)
    _style(a2, f"A · Effort de chaque tige ({len(Tm)} tiges)\nécart max {fr(max(l['dT'] for l in lignes_A) * 1e3, 2)} ‰ de T max")
    dh = [l["dh"] * 100 for l in lignes_B]; dr = [l["dr"] * 100 for l in lignes_B]
    a3.hist([dh, dr], bins=12, color=[ACC, KO], label=["fonction d'appui h(φ) : |polygone − PL| / PL",
                                                        "rayon r(φ) : |moteur − PL| / PL"], alpha=0.85)
    a3.set_xlabel("écart (%)", fontsize=8, color=MUT); a3.set_ylabel("directions", fontsize=8, color=MUT)
    a3.legend(fontsize=7, frameon=False)
    _style(a3, f"B · Domaine : {len(lignes_B)} directions × configurations\nécart max h {fr(max(dh), 2)} % · r {fr(max(dr), 2)} %")
    fig.suptitle("Écarts moteur / méthodes indépendantes", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(chemin, dpi=140)
    plt.close(fig)


# -------------------------------------------------------------------- rapport
def main(rapide=False):
    t0 = time.time()
    os.makedirs(FIG, exist_ok=True)
    n_conf = 12 if rapide else 40
    confs = configurations(n_conf)
    lignes_A, lignes_B, lignes_C = [], [], []
    cas_fig = []
    W0 = None
    for i, (nom, inp) in enumerate(confs):
        R = moteur.compute(inp)
        u = R.u
        N_, Mx, My = u.N_Ed * 1e3, u.Mx_Ed * 1e6, u.My_Ed * 1e6
        xy = [(t["x"], t["y"]) for t in R.tiges]
        # --- A
        W = winkler(R.A_eff_rects, xy, R.k_c, R.k_b, N_, Mx, My)
        if i == 0:
            W0 = W
        Tm = [t["T"] for t in R.tiges]; Tg = [float(x) / 1e3 for x in W["T"]]
        Tref = max(max(Tm), 1e-9)
        lignes_A.append(dict(nom=nom, ok=W["ok"], C_m=R.C, C_g=W["C"] / 1e3, dC=abs(R.C - W["C"] / 1e3) / max(R.C, 1e-9),
                             dx=abs(R.xC - W["xC"]), dy=abs(R.yC - W["yC"]), p_m=R.pmax, p_g=W["pmax"],
                             dp=abs(R.pmax - W["pmax"]) / max(R.pmax, 1e-9), T_m=Tm, T_g=Tg,
                             dT=max(abs(a - b) for a, b in zip(Tm, Tg)) / Tref))
        # --- B, C
        if math.hypot(Mx, My) > 1e-6 and R.dom:
            X, Y, dA = cellules(R.A_eff_rects, PAS_B)
            phis = [2 * math.pi * k / 12 for k in range(12)]
            hl, rl, rm, m1 = [], [], [], []
            for phi in phis:
                h, _ = lp_appui(X, Y, dA, xy, R.FtRd, N_, R.fjd, phi)
                r, _ = lp_rayon(X, Y, dA, xy, R.FtRd, N_, R.fjd, phi)
                hp = appui_polygone(R.dom, phi)
                pm = EQ.capacite_direction(R.A_eff_rects, xy, R.FtRd, N_, R.fjd, phi)
                rmo = math.hypot(pm["Mx"], pm["My"]) if pm else 0.0
                n1 = niveau1_rayon(R, phi)
                hl.append(h); rl.append(r); rm.append(rmo); m1.append(n1)
                lignes_B.append(dict(nom=nom, phi=phi, h_lp=h, h_poly=hp, dh=abs(hp - h) / max(abs(h), 1.0),
                                     r_lp=r, r_mot=rmo, dr=abs(rmo - r) / max(abs(r), 1.0)))
                lignes_C.append(dict(nom=nom, phi=phi, n1=n1, r_lp=r, r_mot=rmo, ok1=n1 <= r * (1 + 2e-3) + 1.0,
                                     ok2=rmo <= r * (1 + 2e-3) + 1.0))
            if len(cas_fig) < 3:
                cas_fig.append((nom.split(" : ")[0] if nom.startswith("aléatoire") else nom, R, phis, hl, rl, rm, m1))
    # --- figures (le mode rapide écrit à côté, sans toucher à la mesure complète)
    sfx = "_rapide" if rapide else ""
    R0 = moteur.compute(confs[0][1])
    figure_repartition(R0, W0, os.path.join(FIG, f"pied_validation_A_repartition{sfx}.png"))
    figure_domaines(cas_fig, os.path.join(FIG, f"pied_validation_B_domaines{sfx}.png"))
    figure_ecarts(lignes_A, lignes_B, os.path.join(FIG, f"pied_validation_ecarts{sfx}.png"))
    # --- rapport
    jour = date.today().strftime("%Y-%m-%d") if not rapide else "rapide"
    dC = max(l["dC"] for l in lignes_A); dT = max(l["dT"] for l in lignes_A); dp = max(l["dp"] for l in lignes_A)
    dxy = max(max(l["dx"], l["dy"]) for l in lignes_A)
    dh = max(l["dh"] for l in lignes_B); dr = max(l["dr"] for l in lignes_B)
    n_c1 = sum(1 for l in lignes_C if not l["ok1"]); n_c2 = sum(1 for l in lignes_C if not l["ok2"])
    verdict = (dC < 2e-3 and dT < 5e-3 and dp < 5e-3 and dxy < 1.0 and dh < 1e-2 and dr < 1e-2 and n_c1 == 0 and n_c2 == 0
               and all(l["ok"] for l in lignes_A))
    L = [f"# Validation croisée du pied de poteau — {jour}", "",
         "*Lot C. Méthodes indépendantes du moteur (numpy + scipy, hors chemin de production) : grille de ressorts "
         "pour la répartition élastique, programmation linéaire (HiGHS) pour le domaine résistant. Script : "
         "`tools/validation_pied_poteau.py`. Cette page est une MESURE datée : elle ne se réécrit pas.*", "",
         f"**Verdict : {'VALIDÉ' if verdict else 'À CONTRÔLER'}** — {len(confs)} configurations "
         f"({sum(1 for n, _ in confs if n.startswith('aléatoire'))} tirées au sort, graine 20261031), "
         f"{len(lignes_B)} directions de moment, {time.time() - t0:.0f} s.", "",
         "## A · Répartition élastique : polygones exacts (moteur) contre grille de ressorts",
         "", f"Pas de la grille : {PAS_A:.0f} mm. Les deux méthodes résolvent le MÊME modèle (platine rigide, lit sans "
         "traction sur l'aire efficace, tiges tendues) par deux discrétisations différentes ; l'écart mesure la "
         "justesse du découpage des polygones et des intégrales de Green du moteur.", "",
         "| Grandeur | Écart maximal | Tolérance |", "|---|---|---|",
         f"| Résultante comprimée C | {fr(dC * 100, 4)} % | 0,2 % |",
         f"| Position de C (xC, yC) | {fr(dxy, 3)} mm | 1 mm |",
         f"| Pression maximale | {fr(dp * 100, 4)} % | 0,5 % |",
         f"| Effort d'une tige (rapporté à T max de la configuration) | {fr(dT * 1000, 3)} ‰ | 5 ‰ |", "",
         "## B · Domaine résistant (niveau 2) : balayage de l'axe neutre contre programme linéaire", "",
         f"Cellules de {PAS_B:.0f} mm, 0 ≤ p ≤ fjd, 0 ≤ T ≤ FT,Rd, équilibre de N. La fonction d'appui du polygone "
         "du moteur (120 sommets) et la capacité sur chaque rayon sont comparées à l'optimum du programme linéaire "
         "— qui est, par le théorème statique, le domaine exact des champs admissibles bornés.", "",
         "| Grandeur | Écart maximal | Tolérance |", "|---|---|---|",
         f"| Fonction d'appui h(φ), polygone / PL | {fr(dh * 100, 3)} % | 1 % |",
         f"| Capacité sur le rayon r(φ), moteur / PL | {fr(dr * 100, 3)} % | 1 % |", "",
         "## C · Niveau 1 ≤ niveau 2, et rayon du moteur intérieur", "",
         f"Directions où le niveau 1 dépasse le rayon du programme linéaire : **{n_c1}** / {len(lignes_C)}. "
         f"Directions où le rayon du moteur dépasse celui du programme linéaire (au-delà de 0,2 % + 1 N·mm) : "
         f"**{n_c2}** / {len(lignes_C)}.", "",
         "## Configurations", "", "| # | Configuration | C moteur / grille (kN) | p max (MPa) | T max moteur / grille (kN) |",
         "|---|---|---|---|---|"]
    for i, l in enumerate(lignes_A):
        L.append(f"| {i + 1} | {l['nom']} | {fr(l['C_m'], 1)} / {fr(l['C_g'], 1)} | {fr(l['p_m'], 2)} / {fr(l['p_g'], 2)} | "
                 f"{fr(max(l['T_m']), 1)} / {fr(max(l['T_g']), 1)} |")
    L += ["", "## Figures", "", f"![Répartition](figures/pied_validation_A_repartition{sfx}.png)", "",
          f"![Domaines](figures/pied_validation_B_domaines{sfx}.png)", "", f"![Écarts](figures/pied_validation_ecarts{sfx}.png)", "",
          "## Ce que cette validation ne couvre pas", "",
          "- les hypothèses du modèle lui-même (platine rigide, bande c, raideurs k13 / k16) : elles sont celles de "
          "l'EN 1993-1-8 et de l'EN 1992-4 section 6 ;",
          "- la longueur efficace des tiges de coin et latérales (modèle d'ingénieur) : il y faut un calcul de plaque "
          "élasto-plastique hors outil ;",
          "- les exemples publiés (MSB Part 5 §6.5, SCI P398) : les documents ne sont pas accessibles depuis "
          "l'environnement de calcul — fiches à compléter dans `docs/validation/exemples_publies.json`."]
    chemin = os.path.join(DOSSIER, f"pied_poteau_validation_{jour}.md")
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:30]))
    print("…\nrapport :", chemin)
    return verdict


if __name__ == "__main__":
    sys.exit(0 if main(rapide="--rapide" in sys.argv) else 1)

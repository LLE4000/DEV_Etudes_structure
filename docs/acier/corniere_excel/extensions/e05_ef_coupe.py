# -*- coding: utf-8 -*-
"""E05 — Prototype de maillage de la COUPE (cas de contrôle) et comparaison au modèle poutre.

Cas calculés (tranche de 1 mm, q = R/L = 42 760/300 N/mm, déformation plane) :
  A-ref        configuration A, ancrage 30 kN/mm par cheville (3 sur 300 mm), mur quasi rigide
  A-mur        idem, mur souple (Winkler 1 000 N/mm³ : joint de mortier, béton local)
  A-rigide     ancrage infiniment raide (1e7 N/mm par mm) : effet de levier en tête d'aile
  B-Y85        configuration B (aile horizontale en haut), ancrage à 85 mm sous la face supérieure
  B-Y45        configuration B, ancrage remonté à 45 mm sous la face supérieure
Plus une étude de convergence (deux maillages) sur A-ref.

Lancer : /usr/bin/python3.12 -I e05_ef_coupe.py
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

import ef_l  # noqa: E402
import ext_commun as K  # noqa: E402

c = K.CAS
e, r, L, R = c["e"], c["r"], c["L"], c["R"]
q = R / L
X = c["X"]
ex = K.excel(c)
sadm = ex["sadm"]
COL = K.COUL
out = []
P = out.append
GEO = dict(B=c["larg"], H=c["haut"], e=e, r=r, Y=c["Y"])
MAIL_FIN = dict(n_t=10, n_c=16, n_leg=40, grad=4.0)
MAIL_GROS = dict(n_t=6, n_c=8, n_leg=20, grad=4.0)

# ---------------------------------------------------------------- utilitaires de dépouillement
def fibres_A(pb):
    """Sections 1 à 4 (configuration A) : résultantes FE et contraintes de fibre."""
    res = {}
    a = e + r
    cuts = {
        1: dict(debut=(a, 0.0), fin=(a, e), normale="x", t=e),
        2: dict(debut=(e, 0.0), fin=(e, a), normale="x", t=a),
        3: dict(debut=(0.0, a), fin=(e, a), normale="y", t=e),
        4: dict(debut=(0.0, e), fin=(a, e), normale="y", t=a),
    }
    for k, cu in cuts.items():
        cp = pb.coupe(cu["debut"], cu["fin"], n=60, normale=cu["normale"])
        t = cu["t"]
        # fibre intérieure (côté congé) et fibre extérieure
        eps = 1e-4
        if cu["normale"] == "x":
            s_int = pb.contrainte_en((cu["debut"][0], t - eps))[0]
            s_ext = pb.contrainte_en((cu["debut"][0], eps))[0]
        else:
            s_int = pb.contrainte_en((t - eps, cu["debut"][1]))[1]
            s_ext = pb.contrainte_en((eps, cu["debut"][1]))[1]
        W, A = t ** 2 / 6, t
        res[k] = dict(M=cp["M"] * L, N=cp["N"] * L, V=cp["V"] * L, s_int=s_int, s_ext=s_ext,
                      s_poutre=abs(cp["M"]) / W + cp["N"] / A, s_flex=abs(cp["M"]) / W)
    return res


def surface_interieure(pb, cfg="A", n_face=120, n_arc=120):
    """Contrainte tangentielle le long de la surface intérieure : face supérieure de l'aile
    horizontale (de la charge au pied du congé), arc du congé, face intérieure de l'aile verticale
    (jusqu'à l'ancrage). s = 0 au pied du congé de l'aile horizontale."""
    H = GEO["H"]
    a = e + r
    yA = GEO["Y"] if cfg == "A" else H - GEO["Y"]
    pts = []
    d = 1e-4
    # face de l'aile horizontale côté congé
    for u in np.linspace(X - 6, a, n_face):
        y = e - d if cfg == "A" else H - e + d
        s = pb.contrainte_en((u, y))
        pts.append((-(u - a), u, y, s[0], s))
    # arc
    Cc = (a, a) if cfg == "A" else (a, H - a)
    for k, th in enumerate(np.linspace(0, math.pi / 2, n_arc)):
        if cfg == "A":
            ang = 3 * math.pi / 2 - th
        else:
            ang = math.pi / 2 + th
        Pp = (Cc[0] + (r + d) * math.cos(ang), Cc[1] + (r + d) * math.sin(ang))
        s = pb.contrainte_en(Pp)
        tvec = np.array([-math.sin(ang), math.cos(ang)])
        sig_t = s[0] * tvec[0] ** 2 + s[1] * tvec[1] ** 2 + 2 * s[2] * tvec[0] * tvec[1]
        pts.append((r * th, Pp[0], Pp[1], sig_t, s))
    # face intérieure de l'aile verticale
    if cfg == "A":
        ys = np.linspace(a, yA - 8, n_face)
    else:
        ys = np.linspace(H - a, yA + 8, n_face)
    for y in ys:
        s = pb.contrainte_en((e - d, y))
        pts.append((r * math.pi / 2 + abs(y - (a if cfg == "A" else H - a)), e, y, s[1], s))
    return pts


def poutre_surface(cfg_pts, M_h, M_v, N_v=0.0):
    """Théorie des poutres aux mêmes points : coupe verticale côté aile horizontale,
    horizontale côté aile verticale (bascule au milieu de l'arc)."""
    a = e + r
    vals = []
    for s, u, y, *_ in cfg_pts:
        if s <= r * math.pi / 4:            # aile horizontale et première moitié de l'arc
            t = K.t_conge(u, e, r) if u <= a else e
            vals.append(M_h(u) / (t ** 2 / 6))
        else:
            yy = y if y >= e else e
            t = K.t_conge(yy, e, r) if yy <= a else e
            vals.append(M_v(yy) / (t ** 2 / 6) + N_v / t)
    return np.array(vals)


# ---------------------------------------------------------------- calculs
resultats = {}
cas = [
    ("A-ref", "A", dict(k_anc=300.0, k_mur=1e5, b_rond=3.0), GEO),
    ("A-mur", "A", dict(k_anc=300.0, k_mur=1e3, b_rond=3.0), GEO),
    ("A-rigide", "A", dict(k_anc=1e7, k_mur=1e5, b_rond=3.0), GEO),
    ("B-Y85", "B", dict(k_anc=300.0, k_mur=1e5, b_rond=3.0), GEO),
    ("B-Y45", "B", dict(k_anc=300.0, k_mur=1e5, b_rond=3.0), dict(GEO, Y=45.0)),
]
pbs = {}
P("# E05 — maillage de la coupe (prototype), cas de contrôle")
P("")
P("## 1. Maillages et temps (CPython 3.12, numpy seul, solveur bande)")
for nom_m, m in (("gros", MAIL_GROS), ("fin", MAIL_FIN)):
    t0 = time.perf_counter()
    pb = ef_l.Probleme(GEO, "A", maillage=m, solveur="bande")
    pb.resoudre(q, X, k_anc=300.0, k_mur=1e5, b_rond=3.0)
    tt = time.perf_counter() - t0
    P(f"maillage {nom_m}: {pb.n} nœuds, {len(pb.elements)} éléments Q9, {2 * pb.n} inconnues, demi-largeur de bande "
      f"{pb.b}, |det J| min {abs(pb.detmin):.3f} ; maillage+rigidités {pb.t_maillage:.2f} s, numérotation+assemblage "
      f"{pb.t_assemblage:.2f} s, résolution (contact, {pb.iterations} itérations) {pb.t_resolution:.2f} s ; total {tt:.2f} s")
    pbs[nom_m] = pb
# convergence
P("")
P("## 2. Convergence (cas A-ref)")
conv = {}
for nom_m in ("gros", "fin"):
    pb = pbs[nom_m]
    fb = fibres_A(pb)
    pts = surface_interieure(pb, "A", 60, 90)
    arr = np.array([p_[3] for p_ in pts])
    k = int(np.argmax(arr))
    conv[nom_m] = (fb, arr[k], pts[k][0])
    P(f"{nom_m}: H = {pb.reactions['H'] * L / 1e3:.3f} kN ; σ fibre int. S1 = {fb[1]['s_int']:.2f}, S3 = {fb[3]['s_int']:.2f} ; "
      f"pic de surface {arr[k]:.2f} N/mm² à s = {pts[k][0]:.2f} mm")
d1 = abs(conv['fin'][1] / conv['gros'][1] - 1) * 100
P(f"écart gros/fin sur le pic de surface : {d1:.2f} %")

P("")
P("## 3. Cas calculés (maillage fin)")
for nom, cfg, opts, geo in cas:
    pb = ef_l.Probleme(geo, cfg, maillage=MAIL_FIN, solveur="bande")
    pb.resoudre(q, X, **opts)
    pbs[nom] = pb
    rr = pb.reactions
    yv, fv = rr["mur_y"], rr["mur_f"]
    H = geo["H"]
    yA = geo["Y"] if cfg == "A" else H - geo["Y"]
    if cfg == "A":
        bas = yv < yA
    else:
        bas = yv < yA
    C_bas = fv[bas].sum()
    C_haut = fv[~bas].sum()
    y_bas = (fv[bas] * yv[bas]).sum() / max(C_bas, 1e-30)
    res = dict(H=rr["H"] * L, y_H=rr["y_H"], C_bas=C_bas * L, y_bas=y_bas, C_haut=C_haut * L,
               y_haut=(fv[~bas] * yv[~bas]).sum() / max(C_haut, 1e-30), V=rr["V"] * L, iterations=pb.iterations,
               x_charge=rr["x_charge"])
    P(f"{nom}: H = {res['H'] / 1e3:.3f} kN à y = {res['y_H']:.2f} ; contact sous l'ancrage {res['C_bas'] / 1e3:.3f} kN "
      f"à y = {res['y_bas']:.2f} ; contact au-dessus {res['C_haut'] / 1e3:.3f} kN (y = {res['y_haut']:.2f}) ; "
      f"V = {res['V'] / 1e3:.3f} kN ; charge en x = {res['x_charge']:.3f} ; itérations de contact {pb.iterations}")
    # corps libre équivalent : pivot au contact trouvé, V sur l'axe
    if cfg == "A":
        H_fb = R * (res["x_charge"] - e / 2) / (res["y_H"] - res["y_bas"]) if res["C_haut"] < 1e-6 * R else float("nan")
    else:
        H_fb = R * (res["x_charge"] - e / 2) / (res["y_H"] - res["y_bas"])
    res["H_corps_libre"] = H_fb
    P(f"   corps libre avec le pivot trouvé (V sur l'axe) : H = {H_fb / 1e3:.3f} kN")
    resultats[nom] = res

# ---------------------------------------------------------------- sections 1 à 4, configuration A
P("")
P("## 4. Sections 1 à 4 : classeur, poutre continue, maillage (contraintes de flexion, N/mm²)")
P("| section | classeur M (kN·m) | classeur σ | FE M (kN·m) A-ref | FE N (kN) | σ poutre avec M,N du FE | FE fibre intérieure | FE fibre extérieure | rapport fibre/poutre |")
tab4 = {}
for nom in ("A-ref", "A-mur", "A-rigide"):
    fb = fibres_A(pbs[nom])
    tab4[nom] = fb
    if nom != "A-ref":
        continue
    for k in (1, 2, 3, 4):
        f = fb[k]
        P(f"| {k} | {ex['M'][k - 1] / 1e6:.4f} | {ex['sig'][k - 1]:.2f} | {abs(f['M']) / 1e6:.4f} | {f['N'] / 1e3:.3f} | "
          f"{f['s_poutre']:.2f} | {f['s_int']:.2f} | {f['s_ext']:.2f} | {f['s_int'] / f['s_poutre']:.3f} |")
for nom in ("A-mur", "A-rigide"):
    fb = tab4[nom]
    P(f"{nom}: " + " ; ".join(f"S{k} M = {abs(fb[k]['M']) / 1e6:.4f} kN·m, fibre int. {fb[k]['s_int']:.1f}" for k in (1, 2, 3, 4)))
# contrôle de statique (aile horizontale) : M FE = R·(x_q − u)
for k, u in ((1, e + r), (2, e)):
    m_stat = R * (resultats["A-ref"]["x_charge"] - u)
    P(f"contrôle statique section {k}: M_FE = {abs(tab4['A-ref'][k]['M']) / 1e6:.5f} ; R·(x_q − u) = {m_stat / 1e6:.5f} kN·m ; "
      f"écart {abs(tab4['A-ref'][k]['M']) / m_stat * 100 - 100:+.3f} %")

# ---------------------------------------------------------------- surface intérieure
P("")
P("## 5. Contrainte le long de la surface intérieure (A-ref)")
pts = surface_interieure(pbs["A-ref"], "A")
sFE = np.array([p_[0] for p_ in pts])
tFE = np.array([p_[3] for p_ in pts])
fbE_H = ex["Mmax"] / (c["Y"] - e / 2)
M_h = lambda u: R / L * max(X - u, 0.0)
M_vE = lambda y: fbE_H / L * max(c["Y"] - y, 0.0)
rref = resultats["A-ref"]
M_vF = lambda y: rref["H"] / L * max(rref["y_H"] - y, 0.0)   # V sur l'axe : bras nul
sigE = poutre_surface(pts, M_h, M_vE)
sigF = poutre_surface(pts, M_h, M_vF, N_v=R / L)
kmax = int(np.argmax(tFE))
P(f"pic FE sur la surface : {tFE[kmax]:.2f} N/mm² à s = {sFE[kmax]:.2f} mm (point u = {pts[kmax][1]:.2f}, y = {pts[kmax][2]:.2f}) ; "
  f"taux {tFE[kmax] / sadm * 100:.2f} % de σ_adm (contrainte de surface, τ nul)")
# pic de chaque côté
cote_h = sFE <= r * math.pi / 4
for nm, msk in (("côté aile horizontale (s ≤ milieu de l'arc)", cote_h), ("côté aile verticale", ~cote_h)):
    j = np.nonzero(msk)[0][np.argmax(tFE[msk])]
    P(f"   {nm}: FE {tFE[j]:.2f} à s = {sFE[j]:.2f} ; poutre (M classeur) {sigE[j]:.2f} ; poutre (M, N du FE) {sigF[j]:.2f}")
vm_s = []
s_all, vm_all = pbs["A-ref"].contraintes_noeuds()
nd = pbs["A-ref"].noeuds
dist_anc = np.hypot(nd[:, 0] - e, nd[:, 1] - c["Y"])
dist_coin = np.hypot(nd[:, 0], nd[:, 1])
masque = (dist_anc > 8) & (dist_coin > 3)
j = int(np.argmax(np.where(masque, vm_all, 0)))
P(f"von Mises nodal max hors voisinage des appuis (8 mm de l'ancrage, 3 mm de l'arête de contact) : {vm_all[j]:.2f} N/mm² "
  f"en u = {nd[j, 0]:.2f}, y = {nd[j, 1]:.2f}")
j2 = int(np.argmax(vm_all))
P(f"von Mises nodal max absolu : {vm_all[j2]:.2f} N/mm² en u = {nd[j2, 0]:.2f}, y = {nd[j2, 1]:.2f} (singularité d'appui)")

# ---------------------------------------------------------------- configuration B
P("")
P("## 6. Configuration B (aile horizontale en haut)")
H = GEO["H"]
for nom in ("B-Y85", "B-Y45"):
    pb = pbs[nom]
    Yb = 85.0 if nom == "B-Y85" else 45.0
    a = e + r
    cuts = {3: ((0.0, H - a), (e, H - a)), 4: ((0.0, H - e), (a, H - e))}
    txt = []
    for k, (d0, d1_) in cuts.items():
        cp = pb.coupe(d0, d1_, n=60, normale="y")
        t = e if k == 3 else a
        s_int = pb.contrainte_en((t - 1e-4, d0[1]))[1]
        txt.append(f"S{k}: M = {abs(cp['M']) * L / 1e6:.4f} kN·m, N = {cp['N'] * L / 1e3:.3f} kN, fibre int. {s_int:.1f}")
    cp1 = pb.coupe((a, H - e), (a, H), n=60, normale="x")
    s1 = pb.contrainte_en((a, H - e + 1e-4))[0]
    rr = resultats[nom]
    fbB_M = R * (X - e / 2)
    P(f"{nom}: H = {rr['H'] / 1e3:.3f} kN ; contact {rr['C_bas'] / 1e3:.3f} kN à y = {rr['y_bas']:.2f} (bout libre en y = 0) ; "
      f"corps libre (pivot trouvé, V sur l'axe) H = {rr['H_corps_libre'] / 1e3:.3f} kN ; "
      f"S1: M = {abs(cp1['M']) * L / 1e6:.4f}, fibre (congé) {s1:.1f} ; " + " ; ".join(txt)
      + f" ; corps libre : M3 = M4 = R·(X − e/2) = {fbB_M / 1e6:.4f} kN·m")
    pts_b = surface_interieure(pb, "B")
    tb = np.array([p_[3] for p_ in pts_b])
    kk = int(np.argmax(np.abs(tb)))
    P(f"   surface intérieure : |σ| max {abs(tb[kk]):.1f} N/mm² (signe {np.sign(tb[kk]):+.0f}) à s = {pts_b[kk][0]:.2f}")
    resultats[nom]["pts"] = [(p_[0], p_[3]) for p_ in pts_b]

# ---------------------------------------------------------------- figures
blues = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
cmap_seq = LinearSegmentedColormap.from_list("bleus", blues)


def triangulation(pb, Ud=None, ech=0.0):  # noqa: D401
    xy = pb.noeuds.copy()
    if Ud is not None:
        xy[:, 0] += ech * Ud[0::2]
        xy[:, 1] += ech * Ud[1::2]
    tris = []
    for el in pb.elements:
        g = [[el[3 * j + i] for i in range(3)] for j in range(3)]
        for j in range(2):
            for i in range(2):
                a_, b_, c_, d_ = g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i]
                tris += [(a_, b_, c_), (a_, c_, d_)]
    return Triangulation(xy[:, 0], xy[:, 1], np.array(tris))


def contour_ext(pb, Ud=None, ech=0.0):
    """Arêtes du bord (arêtes d'éléments n'appartenant qu'à un élément)."""
    from collections import Counter
    cnt = Counter()
    for el in pb.elements:
        for a_ in ((0, 1, 2), (2, 5, 8), (8, 7, 6), (6, 3, 0)):
            k = tuple(sorted((el[a_[0]], el[a_[2]])))
            cnt[(k, el[a_[1]])] += 1
    xy = pb.noeuds.copy()
    if Ud is not None:
        xy[:, 0] += ech * Ud[0::2]
        xy[:, 1] += ech * Ud[1::2]
    segs = []
    for (k, m), n_ in cnt.items():
        if n_ == 1:
            segs.append([xy[k[0]], xy[m], xy[k[1]]])
    return segs


def carte(ax, pb, titre, vmax, ech=None, cfg="A"):
    s_n, vm = pb.contraintes_noeuds()
    if ech is None:                      # amplification : déplacement maximal dessiné ≈ 6 mm
        ech = 6.0 / np.abs(pb.U).max()
    titre = titre + f"\n(déformée ×{ech:.0f} ; déplacement max {K.fr(np.abs(pb.U).max(), 3)} mm)"
    tri = triangulation(pb, pb.U, ech)
    tpc = ax.tripcolor(tri, np.clip(vm, 0, vmax), shading="gouraud", cmap=cmap_seq, vmin=0, vmax=vmax)
    for sg in contour_ext(pb):
        sg = np.array(sg)
        ax.plot(sg[:, 0], sg[:, 1], color=COL["muet"], lw=0.6, ls=(0, (3, 2)))
    for sg in contour_ext(pb, pb.U, ech):
        sg = np.array(sg)
        ax.plot(sg[:, 0], sg[:, 1], color=COL["encre"], lw=0.8)
    ax.axvline(0, color=COL["encre2"], lw=1.2)
    ax.fill_betweenx([-10, 130], -14, 0, color=COL["beton"], zorder=0)
    # efforts de contact
    rr = pb.reactions
    f = rr["mur_f"]
    ymur = rr["mur_y"]
    fmax = max(f.max(), 1e-9)
    for yy, ff in zip(ymur, f):
        if ff > 0.02 * fmax:
            ax.annotate("", xy=(0, yy), xytext=(-12 * ff / fmax, yy),
                        arrowprops=dict(arrowstyle="-|>", color=COL["s2"], lw=1.0, mutation_scale=7))
    H = pb.geo["H"]
    yA = pb.geo["Y"] if cfg == "A" else H - pb.geo["Y"]
    ax.plot([e], [yA], marker=">", ms=7, color=COL["encre"])
    ax.text(e + 3, yA, "ancrage", fontsize=7.5, color=COL["encre"], va="center")
    yq = e if cfg == "A" else H
    ax.annotate("", xy=(X, yq + ech * pb.U[1::2].min() * 0), xytext=(X, yq + 22),
                arrowprops=dict(arrowstyle="-|>", color=COL["critique"], lw=1.6, mutation_scale=11))
    ax.set_aspect("equal")
    ax.set_xlim(-16, 128)
    ax.set_ylim(-8, 146)
    ax.axis("off")
    ax.set_title(titre, fontsize=9, color=COL["encre"], loc="left")
    return tpc


os.makedirs(K.FIGURES, exist_ok=True)
fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.6), dpi=150)
vmax = 260.0
for ax, nom, tit in zip(axs, ("A-ref", "A-rigide", "B-Y85"),
                        ("A — ancrage 30 kN/mm, mur raide", "A — ancrage infiniment raide (effet de levier)",
                         "B — aile horizontale en haut, ancrage à 85 mm")):
    tpc = carte(ax, pbs[nom], tit, vmax, cfg=nom[0])
cb = fig.colorbar(tpc, ax=axs, shrink=0.75, pad=0.01)
cb.set_label("von Mises (N/mm², déformation plane) — écrêté à 260", fontsize=8, color=COL["encre2"])
cb.ax.tick_params(labelsize=7.5, colors=COL["encre2"])
fig.suptitle("Maillage de la coupe — L120×120×15, q = 42,76 kN / 300 mm ; flèches orange : contact du mur ; pointillés : forme initiale",
             fontsize=10, color=COL["encre"])
fE1 = os.path.join(K.FIGURES, "e05_ef_vonmises.png")
fig.savefig(fE1, facecolor=COL["fond"], bbox_inches="tight")
plt.close(fig)

# figure 2 : contrainte le long de la surface intérieure, A-ref
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
K.style_axes(ax)
ax.axvspan(0, r * math.pi / 2, color=COL["neutre"], zorder=0)
ax.plot(sFE, tFE, color=COL["s1"], lw=2, label="maillage (contrainte de surface, A-ref)")
ax.plot(sFE, sigE, color=COL["s2"], lw=1.6, ls=(0, (5, 2)), label="poutre, moments du classeur (coupes droites)")
ax.plot(sFE, sigF, color=COL["s3"], lw=1.6, ls=(0, (2, 2)), label="poutre, efforts du maillage (M + N)")
ax.axhline(sadm, color=COL["critique"], lw=0.9, ls="--")
ax.text(sFE.min() + 1, sadm + 4, "σ_adm = 236,7", fontsize=7.5, color=COL["encre2"])
for s0, lab in ((0.0, "section 1\n(pied du congé)"), (r * math.pi / 2, "section 3\n(pied du congé)")):
    ax.axvline(s0, color=COL["muet"], lw=0.8)
    ax.text(s0 + 0.5, 15, lab, fontsize=7.5, color=COL["encre2"])
ax.plot([sFE[kmax]], [tFE[kmax]], "o", ms=8, color=COL["s1"], mec=COL["fond"], mew=2)
ax.annotate(f"pic {K.fr(tFE[kmax], 1)} N/mm²", (sFE[kmax], tFE[kmax]), xytext=(8, 8), textcoords="offset points",
            fontsize=8, color=COL["encre"])
ax.text(r * math.pi / 4, 280, "arc du congé", fontsize=7.5, color=COL["encre2"], ha="center")
ax.set_ylim(0, 300)
ax.set_xlabel("abscisse le long de la surface intérieure (mm) : négatif vers la charge, positif vers l'ancrage",
              fontsize=8.5, color=COL["encre2"])
ax.set_ylabel("contrainte de traction (N/mm²)", fontsize=8.5, color=COL["encre2"])
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set_title("Surface intérieure de la cornière : maillage et théorie des poutres", fontsize=9.5, color=COL["encre"],
             loc="left")
fig.tight_layout()
fE2 = os.path.join(K.FIGURES, "e05_surface_interieure.png")
fig.savefig(fE2, facecolor=COL["fond"])
plt.close(fig)

# figure zoom : contrainte principale signée (traction > 0) autour de l'angle, A-ref
pb = pbs["A-ref"]
s_n, vm = pb.contraintes_noeuds()
cen = 0.5 * (s_n[:, 0] + s_n[:, 1])
rad = np.sqrt((0.5 * (s_n[:, 0] - s_n[:, 1])) ** 2 + s_n[:, 2] ** 2)
s1, s2 = cen + rad, cen - rad
sig = np.where(np.abs(s1) >= np.abs(s2), s1, s2)
div = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a9", "#e34948", "#9b1c1c"])
fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=150)
tri = triangulation(pb)
tpc = ax.tripcolor(tri, np.clip(sig, -260, 260), shading="gouraud", cmap=div, norm=TwoSlopeNorm(0, -260, 260))
for sg in contour_ext(pb):
    sg = np.array(sg)
    ax.plot(sg[:, 0], sg[:, 1], color=COL["encre"], lw=0.8)
a_ = e + r
for lab, (x0, y0, x1, y1) in {"1": (a_, 0, a_, e), "2": (e, 0, e, a_), "3": (0, a_, e, a_), "4": (0, e, a_, e)}.items():
    ax.plot([x0, x1], [y0, y1], color=COL["encre"], lw=1.0, ls=(0, (4, 2)))
    ax.text(x0 + 0.7, y0 + 0.7, lab, fontsize=10, fontweight="bold", color=COL["encre"])
pts_z = surface_interieure(pb, "A", 40, 120)
kz = int(np.argmax([p_[3] for p_ in pts_z]))
ax.plot([pts_z[kz][1]], [pts_z[kz][2]], "o", ms=7, mfc="none", mec=COL["encre"], mew=1.5)
ax.annotate(f"pic de surface {K.fr(pts_z[kz][3], 0)} N/mm²", (pts_z[kz][1], pts_z[kz][2]), xytext=(34, 36),
            fontsize=8, color=COL["encre"], arrowprops=dict(arrowstyle="-", color=COL["muet"], lw=0.7))
ax.set_xlim(-2, 52)
ax.set_ylim(-2, 52)
ax.set_aspect("equal")
ax.axis("off")
cb = fig.colorbar(tpc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("contrainte principale dominante (N/mm²) : rouge traction, bleu compression", fontsize=8,
             color=COL["encre2"])
cb.ax.tick_params(labelsize=7.5, colors=COL["encre2"])
ax.set_title("Angle de la cornière (cas A-ref) : sections 1 à 4 du classeur sur le champ de contraintes",
             fontsize=9, color=COL["encre"], loc="left")
fig.tight_layout()
fE4 = os.path.join(K.FIGURES, "e05_ef_zoom_angle.png")
fig.savefig(fE4, facecolor=COL["fond"])
plt.close(fig)

# figure 3 : effet de levier en fonction de la raideur d'ancrage
P("")
P("## 7. Effet de levier selon la raideur de l'ancrage (maillage gros, A, mur raide)")
pbg = pbs["gros"]
ks = [30, 100, 200, 300, 500, 1000, 3000, 1e4, 1e5]
Hs, Qs = [], []
for kk in ks:
    pbg.resoudre(q, X, k_anc=kk * 1e3 / 100.0, k_mur=1e5, b_rond=3.0)   # kN/mm par cheville -> N/mm par mm (3 sur 300 mm)
    rr = pbg.reactions
    yv, fv = rr["mur_y"], rr["mur_f"]
    Hs.append(rr["H"] * L / 1e3)
    Qs.append(fv[yv > c["Y"]].sum() * L / 1e3)
    P(f"k = {kk:>7g} kN/mm par cheville : H = {Hs[-1]:.2f} kN ; contact en tête (y > Y) = {Qs[-1]:.2f} kN")
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
K.style_axes(ax)
ax.semilogx(ks, Hs, color=COL["s1"], lw=2, marker="o", ms=5, label="traction totale des ancrages H (maillage)")
ax.semilogx(ks, Qs, color=COL["s2"], lw=1.6, marker="s", ms=4, label="force de levier en tête d'aile (maillage)")
ax.axhline(ex["N_anc"] * 3 / 1e3, color=COL["muet"], lw=1, ls="--")
ax.text(32, ex["N_anc"] * 3 / 1e3 + 2, "classeur : 3 × 14,25 = 42,76 kN", fontsize=7.5, color=COL["encre2"])
ax.set_xlabel("raideur axiale d'une cheville (kN/mm), 3 chevilles sur 300 mm", fontsize=8.5, color=COL["encre2"])
ax.set_ylabel("kN (ELS)", fontsize=8.5, color=COL["encre2"])
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set_title("Effet de levier : il n'apparaît qu'avec un ancrage très raide", fontsize=9.5, color=COL["encre"], loc="left")
fig.tight_layout()
fE3 = os.path.join(K.FIGURES, "e05_effet_levier.png")
fig.savefig(fE3, facecolor=COL["fond"])
plt.close(fig)

P("")
P("## Figures")
for f in (fE1, fE2, fE3, fE4):
    P(f)
K.ecrire(os.path.join(K.ICI, "e05_sortie.txt"), out)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "pts"} for k, v in resultats.items()},
          open(os.path.join(K.ICI, "e05_resultats.json"), "w"), indent=1)
print("\n".join(out))

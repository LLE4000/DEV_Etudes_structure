# -*- coding: utf-8 -*-
"""Relecture adverse — 5. Mon propre maillage de la coupe (indépendant de ef_l.py) : quadrilatères à 4 nœuds,
déformation plane, patchs de Coons dans l'angle (congé exact), contact unilatéral avec le mur (pénalité),
ancrage = ressorts unilatéraux sous l'écrou ± 3 mm + appui vertical au nœud de mi-épaisseur.
Configuration A (classeur) et B (retournée). Unités : N, mm."""
import math
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import griddata

E_, NU = 210000.0, 0.3
e, r, B, H_ = 15.0, 13.0, 120.0, 120.0
a = e + r
LZ = 300.0                 # épaisseur hors plan = longueur reprise
R, X, Y = 42760.0, 75.0, 85.0


def maillage(h=0.5):
    nodes, elems, idx = [], [], {}

    def nid(p):
        k = (round(p[0], 6), round(p[1], 6))
        if k not in idx:
            idx[k] = len(nodes)
            nodes.append(k)
        return idx[k]

    def bloc(fun, n1, n2):
        g = [[nid(fun(i / n1, j / n2)) for j in range(n2 + 1)] for i in range(n1 + 1)]
        for i in range(n1):
            for j in range(n2):
                elems.append([g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]])

    nH, nE, nA = int(round((B - a) / h)), int(round(e / h)), int(round(a / h))
    bloc(lambda s, t: (a + (B - a) * s, e * t), nH, nE)                     # aile horizontale
    bloc(lambda s, t: (e * t, a + (H_ - a) * s), nH, nE)                    # aile verticale
    M = np.array([a - r * math.cos(math.pi / 4), a - r * math.sin(math.pi / 4)])

    def coons(cb, ct, cl, cr):
        P00, P10, P01, P11 = cb(0), cb(1), ct(0), ct(1)
        def f(s, t):
            v = ((1 - t) * np.array(cb(s)) + t * np.array(ct(s)) + (1 - s) * np.array(cl(t)) + s * np.array(cr(t))
                 - ((1 - s) * (1 - t) * np.array(P00) + s * (1 - t) * np.array(P10) + (1 - s) * t * np.array(P01) + s * t * np.array(P11)))
            return (float(v[0]), float(v[1]))
        return f
    arcA = lambda s: (a + r * math.cos(math.radians(225 + 45 * s)), a + r * math.sin(math.radians(225 + 45 * s)))
    bloc(coons(lambda s: (a * s, 0.0), arcA, lambda t: (M[0] * t, M[1] * t), lambda t: (a, e * t)), nA, nE)
    arcB = lambda s: (a + r * math.cos(math.radians(225 - 45 * s)), a + r * math.sin(math.radians(225 - 45 * s)))
    bloc(coons(lambda s: (0.0, a * s), arcB, lambda t: (M[0] * t, M[1] * t), lambda t: (e * t, a)), nA, nE)
    N = np.array(nodes, float)
    El = np.array(elems, int)
    # orientation directe
    for k, el in enumerate(El):
        p = N[el]
        aire = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        if aire < 0:
            El[k] = el[::-1]
    return N, El


GP = np.array([-1, 1]) / math.sqrt(3)
Dm = E_ / ((1 + NU) * (1 - 2 * NU)) * np.array([[1 - NU, NU, 0], [NU, 1 - NU, 0], [0, 0, (1 - 2 * NU) / 2]])


def dN(xi, eta):
    return 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])


def Bmat(p, xi, eta):
    d = dN(xi, eta)
    J = d @ p
    dx = np.linalg.solve(J, d)
    Bm = np.zeros((3, 8))
    Bm[0, 0::2] = dx[0]
    Bm[1, 1::2] = dx[1]
    Bm[2, 0::2] = dx[1]
    Bm[2, 1::2] = dx[0]
    return Bm, np.linalg.det(J)


def assembler(N, El):
    rows, cols, vals = [], [], []
    for el in El:
        p = N[el]
        Ke = np.zeros((8, 8))
        for xi in GP:
            for eta in GP:
                Bm, dJ = Bmat(p, xi, eta)
                Ke += Bm.T @ Dm @ Bm * dJ * LZ
        dofs = np.ravel([[2 * n, 2 * n + 1] for n in el])
        rows.append(np.repeat(dofs, 8))
        cols.append(np.tile(dofs, 8))
        vals.append(Ke.ravel())
    n = 2 * len(N)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def contraintes_nodales(N, El, U):
    acc = np.zeros((len(N), 3))
    cnt = np.zeros(len(N))
    ext = np.array([[1 + math.sqrt(3) / 2, -0.5, 1 - math.sqrt(3) / 2, -0.5],
                    [-0.5, 1 + math.sqrt(3) / 2, -0.5, 1 - math.sqrt(3) / 2],
                    [1 - math.sqrt(3) / 2, -0.5, 1 + math.sqrt(3) / 2, -0.5],
                    [-0.5, 1 - math.sqrt(3) / 2, -0.5, 1 + math.sqrt(3) / 2]])
    gpts = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for el in El:
        p = N[el]
        ue = np.ravel([[U[2 * n], U[2 * n + 1]] for n in el])
        s = []
        for (gx, gy) in gpts:
            Bm, _ = Bmat(p, gx / math.sqrt(3), gy / math.sqrt(3))
            s.append(Dm @ Bm @ ue)
        s = np.array(s)
        sn = ext @ s
        acc[el] += sn
        cnt[el] += 1
    return acc / cnt[:, None]


def resoudre(N, El, K, conf="A", k_anc_cheville=30e3, k_mur_s=None, washer=3.0, Yb=Y, pas=0.5, haut_ok=True):
    """k_mur_s : module de réaction (N/mm³) ; None = mur raide (pénalité)."""
    n = 2 * len(N)
    F = np.zeros(n)
    yl = e if conf == "A" else H_          # face chargée de l'aile horizontale
    sel = np.where((np.abs(N[:, 1] - yl) < 1e-6) & (N[:, 0] >= X - 5 - 1e-9) & (N[:, 0] <= X + 5 + 1e-9))[0]
    xs = np.sort(N[sel, 0])
    w = np.zeros(len(sel))
    for k, i in enumerate(sel):
        x = N[i, 0]
        w[k] = (min(x + pas / 2, X + 5) - max(x - pas / 2, X - 5))
    w *= R / w.sum()
    F[2 * sel + 1] -= w
    ya = Yb if conf == "A" else H_ - Yb
    anc = np.where((np.abs(N[:, 0] - e) < 1e-6) & (np.abs(N[:, 1] - ya) <= washer + 1e-9))[0]
    kan = 3 * k_anc_cheville / len(anc)
    vnode = np.argmin((N[:, 0] - e / 2) ** 2 + (N[:, 1] - ya) ** 2)
    mur = np.where(np.abs(N[:, 0]) < 1e-6)[0]
    if not haut_ok:
        mur = mur[N[mur, 1] < ya] if conf == "A" else mur[N[mur, 1] < ya]
    ymur = N[mur, 1]
    ordre = np.argsort(ymur)
    trib = np.zeros(len(mur))
    ys = ymur[ordre]
    t_ = np.zeros(len(ys))
    t_[1:-1] = (ys[2:] - ys[:-2]) / 2
    t_[0] = (ys[1] - ys[0]) / 2
    t_[-1] = (ys[-1] - ys[-2]) / 2
    trib[ordre] = t_
    kmur = (np.full(len(mur), 1e9) if k_mur_s is None else k_mur_s * trib * LZ)
    actif_m = np.ones(len(mur), bool)
    actif_a = np.ones(len(anc), bool)
    for it in range(60):
        d = np.zeros(n)
        d[2 * mur] += np.where(actif_m, kmur, 0.0)
        d[2 * anc] += np.where(actif_a, kan, 0.0)
        d[2 * vnode + 1] += 1e12
        Kt = K + sp.diags(d)
        U = spla.spsolve(Kt.tocsc(), F)
        um = U[2 * mur]
        ua = U[2 * anc]
        nm = um < 0          # pénètre le mur
        na = ua > 0          # l'écrou s'éloigne du mur : la cheville est tendue
        if np.array_equal(nm, actif_m) and np.array_equal(na, actif_a):
            break
        actif_m, actif_a = nm, na
    fm = -kmur * um * actif_m             # compression du mur (N, > 0)
    fa = kan * ua * actif_a               # traction des chevilles (N, > 0)
    return dict(U=U, fm=fm, ym=ymur, fa=fa, ya=N[anc, 1], Fv=-1e12 * U[2 * vnode + 1], it=it, mur=mur)


def resume(nom, N, res, ya):
    fm, ym = res["fm"], res["ym"]
    bas = ym < ya
    C = fm[bas].sum()
    yc = (fm[bas] * ym[bas]).sum() / max(C, 1e-30)
    Ht = res["fa"].sum()
    yH = (res["fa"] * res["ya"]).sum() / max(Ht, 1e-30)
    print(f"{nom}: H = {Ht/1e3:.3f} kN à y = {yH:.2f} ; mur sous l'ancrage {C/1e3:.3f} kN centré à y = {yc:.2f} ; "
          f"mur au-dessus {fm[~bas].sum()/1e3:.3f} kN ; appui vertical {res['Fv']/1e3:.3f} kN ; itérations {res['it']}")
    return Ht, C, yc, fm[~bas].sum()


if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    N, El = maillage(h)
    print(f"maillage h = {h} : {len(N)} nœuds, {len(El)} éléments")
    K = assembler(N, El)
    res = resoudre(N, El, K, "A", pas=h)
    Ht, C, yc, Ch = resume("A, mur raide, 30 kN/mm", N, res, Y)
    S = contraintes_nodales(N, El, res["U"])
    sx, sy, txy = S[:, 0], S[:, 1], S[:, 2]
    s1 = (sx + sy) / 2 + np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    s2 = (sx + sy) / 2 - np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    # sections
    def val(comp, pts):
        return griddata(N, comp, np.array(pts), method="linear")
    # section 2 : coupe u = e, y de 0 à a ; section 4 : coupe y = e, u de 0 à a
    print("section 2 : σx en bas (15, 0+) =", val(sx, [(e, 0.02)])[0], " en haut (15, 28-) =", val(sx, [(e, a - 0.02)])[0],
          " ; classeur σ2 =", 2.5656e6 / 39200)
    print("section 4 : σy au mur (0+, 15) =", val(sy, [(0.02, e)])[0], " côté congé (28-, 15) =", val(sy, [(a - 0.02, e)])[0],
          " ; classeur σ4 =", 2.8966452e6 / 39200)
    print("section 1 : σx haut (28, 15) =", val(sx, [(a, e - 0.01)])[0], " bas (28, 0) =", val(sx, [(a, 0.01)])[0],
          " ; poutre R·47/W =", R * 47 / 11250)
    print("section 3 : σy côté congé (15, 28) =", val(sy, [(e - 0.01, a)])[0], " côté mur (0, 28) =", val(sy, [(0.01, a)])[0],
          " ; poutre P0 M/W+N/A =", Ht * 57 / 11250 + R / 4500, " ; classeur σ3 = 209.662")
    # section 3 : résultantes par intégration des σy sur y = 28
    us = np.linspace(0.0, e, 301)
    syv = val(sy, [(u, a) for u in us])
    Nn = np.trapezoid(syv, us) * LZ
    Mm = np.trapezoid(syv * (us - e / 2), us) * LZ
    print(f"section 3 intégrée : N = {Nn/1e3:.2f} kN ; M = {abs(Mm)/1e6:.4f} kN·m (P0 : {Ht*57/1e6:.4f})")
    # pic à la surface du congé
    arc = np.where(np.abs(np.hypot(N[:, 0] - a, N[:, 1] - a) - r) < 1e-6)[0]
    k = arc[np.argmax(s1[arc])]
    th = math.degrees(math.atan2(N[k, 1] - a, N[k, 0] - a))
    print(f"pic à la surface du congé : σ1 = {s1[k]:.1f} N/mm² en ({N[k,0]:.2f} ; {N[k,1]:.2f}), angle {th:.1f}°,"
          f" à {a - N[k,0]:.2f} mm (horizontalement) du pied de l'aile horizontale, arc {math.radians(270-th)*r:.2f} mm")
    # poutre au même u
    uu = N[k, 0]
    tl = a - math.sqrt(r * r - (a - uu) ** 2)
    print(f"   contrainte de poutre au même u : {R*(X-uu)/(LZ*tl*tl/6):.1f} ; rapport {s1[k]/(R*(X-uu)/(LZ*tl*tl/6)):.3f}")
    # le long du pied du congé, vers la section 1 / 3
    print("σx fibre supérieure de l'aile horizontale (y = e) selon u :")
    for u_ in (28.0, 30.0, 32.0, 35.0, 40.0):
        sv = val(sx, [(u_, e - 0.01)])[0]
        print(f"   u = {u_}: σx = {sv:.1f} ; poutre {R*(X-u_)/11250:.1f} ; rapport {sv/(R*(X-u_)/11250):.3f}")
    print("max σ1 sur les surfaces libres intérieures :", float(np.max(s1)), "en", N[np.argmax(s1)])

    # configuration B
    NB = N.copy()
    NB[:, 1] = H_ - N[:, 1]
    ElB = El[:, ::-1].copy()
    KB = assembler(NB, ElB)
    resB = resoudre(NB, ElB, KB, "B", pas=h)
    resume("B, mur raide, 30 kN/mm", NB, resB, H_ - Y)
    print(f"   corps libre B : H = {R*(X-e/2)/(H_-Y)/1e3:.3f} kN")

    # mur souple
    for ks in (1e3, 30.0):
        rs = resoudre(N, El, K, "A", k_mur_s=ks, pas=h)
        Hs, Cs, ycs, _ = resume(f"A, mur Winkler {ks:g} N/mm³", N, rs, Y)
        Ss = contraintes_nodales(N, El, rs["U"])
        syv = griddata(N, Ss[:, 1], np.array([(u, a) for u in us]), method="linear")
        Nn = np.trapezoid(syv, us) * LZ
        Mm = np.trapezoid(syv * (us - e / 2), us) * LZ
        snom = abs(Mm) / (LZ * e * e / 6) + Nn / (LZ * e)
        print(f"   section 3 : N = {Nn/1e3:.2f} kN, M = {abs(Mm)/1e6:.4f} kN·m, σ nominal M/W+N/A = {snom:.2f} "
              f"(classeur σ3 = 209.66 : {snom/209.662-1:+.2%})")

    # effet de levier
    for kc in (30e3, 100e3, 200e3, 300e3, 1000e3, 1e8):
        rl = resoudre(N, El, K, "A", k_anc_cheville=kc, pas=h)
        resume(f"levier, cheville {kc/1e3:g} kN/mm", N, rl, Y)

# -*- coding: utf-8 -*-
"""Modèle longitudinal de la cornière (repris de extensions/e03_longitudinal.py, mêmes équations).

Poutre à section indéformable sur appuis élastiques : U, V (translations du centre de cisaillement), φ (rotation) ;
flexion biaxiale avec produit d'inertie ; torsion de Saint-Venant ; ancrages = tirant horizontal à la hauteur Y
(k_N en série avec k_pl) + appui vertical k_V ; mur = contact unilatéral continu à l'arête inférieure et, si
« levier », en tête d'aile. Propriétés lues dans le catalogue du site.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import s_commun as K  # noqa: E402

C = K.CAS
e, r, R = C["e"], C["r"], C["R"]
prof = [q for q in K.profils_site(("L",)) if q["nom"] == "L 120x120x15"][0]


def contour():
    B = Hh = 120.0
    pts = [(0, 0), (B, 0), (B, e), (e + r, e)]
    for a in np.linspace(-math.pi / 2, -math.pi, 41)[1:-1]:
        pts.append((e + r + r * math.cos(a), e + r + r * math.sin(a)))
    pts += [(e, e + r), (e, Hh), (0, Hh)]
    return np.array(pts, float)


def proprietes(pts):
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    Ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cr).sum() / 24 - A * cx * cy
    return A, cx, cy, Ixy


A_p, uG, yG, Iuy_p = proprietes(contour())
E_, G_ = 210000.0, 81000.0
Iuu = prof["Iy"] * 1e4
Iyy = prof["Iz"] * 1e4
Iuy = math.copysign(abs(prof["Iyz"]) * 1e4, Iuy_p)
It = prof["It"] * 1e4
uS = yS = e / 2
Wel = prof["Wel_y"] * 1e3


def modele(Ltot=2000.0, ancrages=(200.0, 1000.0, 1800.0), charges=((400.0, 300.0, R),), Xp=75.0, Y=85.0,
           yP=0.0, kN=30e3, kV=10e3, k_pl=None, kw=1e5, h=5.0, rigide=1.0, H_aile=120.0, levier=True):
    n_el = int(round(Ltot / h))
    zs = np.linspace(0, Ltot, n_el + 1)
    nn = len(zs)
    ndof = 5 * nn
    Kg = np.zeros((ndof, ndof))
    pat = np.array([[12, 6 * h, -12, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h],
                    [-12, -6 * h, 12, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]]) * E_ / h ** 3 * rigide
    for i in range(n_el):
        a, b = 5 * i, 5 * (i + 1)
        iu = [a, a + 1, b, b + 1]
        iv = [a + 2, a + 3, b + 2, b + 3]
        Kg[np.ix_(iu, iu)] += Iyy * pat
        Kg[np.ix_(iv, iv)] += Iuu * pat
        Kg[np.ix_(iu, iv)] += Iuy * pat
        Kg[np.ix_(iv, iu)] += Iuy * pat
        ip = [a + 4, b + 4]
        Kg[np.ix_(ip, ip)] += G_ * It / h * np.array([[1, -1], [-1, 1]]) * rigide
    F = np.zeros(ndof)
    for zc, larg, Ptot in charges:
        if larg <= 0:
            w = np.zeros(nn)
            w[int(np.argmin(abs(zs - zc)))] = 1.0
        else:
            w = np.clip(np.minimum(zs + h / 2, zc + larg / 2) - np.maximum(zs - h / 2, zc - larg / 2), 0, None)
            w /= w.sum()
        F[2::5] += -Ptot * w
        F[4::5] += -Ptot * w * (Xp - uS)
    kNe = kN if k_pl is None else 1.0 / (1.0 / kN + 1.0 / k_pl)
    idx_a = [int(np.argmin(abs(zs - za))) for za in ancrages]

    def g_u(yy):
        v = np.zeros(5)
        v[0] = 1.0
        v[4] = -(yy - yS)
        return v
    gv = np.zeros(5)
    gv[2] = 1.0
    wz = np.full(nn, h)
    wz[0] = wz[-1] = h / 2
    actif_bas = np.ones(nn, dtype=bool)
    actif_haut = np.zeros(nn, dtype=bool)
    actif_anc = np.ones(len(idx_a), dtype=bool)
    for it in range(50):
        Kt = Kg.copy()
        for j, k in enumerate(idx_a):
            sl = slice(5 * k, 5 * k + 5)
            if actif_anc[j]:
                Kt[sl, sl] += kNe * np.outer(g_u(Y), g_u(Y))
            Kt[sl, sl] += kV * np.outer(gv, gv)
        gb, gh = g_u(yP), g_u(H_aile)
        for k in range(nn):
            sl = slice(5 * k, 5 * k + 5)
            if actif_bas[k]:
                Kt[sl, sl] += kw * wz[k] * np.outer(gb, gb)
            if actif_haut[k]:
                Kt[sl, sl] += kw * wz[k] * np.outer(gh, gh)
        Un = np.linalg.solve(Kt, F).reshape(nn, 5)
        db, dh = Un @ gb, Un @ gh
        da = np.array([Un[k] @ g_u(Y) for k in idx_a])
        nb, nh, na = db < 0, (dh < 0) & levier, da > 0
        if np.array_equal(nb, actif_bas) and np.array_equal(nh, actif_haut) and np.array_equal(na, actif_anc):
            break
        actif_bas, actif_haut, actif_anc = nb, nh, na
    N_anc = np.where(actif_anc, kNe * da, 0.0)
    V_anc = np.array([-kV * Un[k] @ gv for k in idx_a])
    C_bas = np.where(actif_bas, -kw * wz * db, 0.0)
    C_haut = np.where(actif_haut, -kw * wz * dh, 0.0)
    Tz = G_ * It * np.gradient(Un[:, 4], zs) * rigide
    U2 = np.gradient(Un[:, 1], zs)
    V2 = np.gradient(Un[:, 3], zs)
    fibres = [(0, 0), (120, 0), (0, 120), (120, e), (e, 120)]
    sz = np.max([np.abs(E_ * rigide * ((uf - uG) * U2 + (yf - yG) * V2)) for uf, yf in fibres], axis=0)
    Mv = E_ * rigide * (Iuu * V2 + Iuy * U2)
    return dict(z=zs, N=N_anc, V=V_anc, C_bas=C_bas, C_haut=C_haut, T=Tz, sz=sz, Mv=Mv, it=it + 1)




# raideur locale de l'aile verticale autour d'un ancrage (analogie du tronçon en T, EN 1993-1-8, à confirmer)
m_tee = (C["Y"] - e) - 0.8 * r
leff = min(2 * math.pi * m_tee, 4 * m_tee + 1.25 * (120.0 - C["Y"]))
k_pl = 3 * E_ * (leff * e ** 3 / 12) / C["Y"] ** 3

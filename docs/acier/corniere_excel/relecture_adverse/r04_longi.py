# -*- coding: utf-8 -*-
"""Relecture adverse — 4. Cornière longue : mon propre modèle de poutre (indépendant de s_longi.py / e03).
Degrés de liberté par nœud : w (horizontal, vers l'extérieur du mur), w', v (vertical), v', φ (rotation autour du
centre de cisaillement, sens trigonométrique dans le plan (u, y)). Flexion biaxiale (tenseur centré, produit
d'inertie), torsion de Saint-Venant. Contact unilatéral au talon (u = 0, y = 0) continu ; ancrages discrets :
tirant horizontal à (u, 85), appui vertical au nœud de mi-épaisseur (u = e/2). Unités N, mm."""
import math
import json
import sys
import numpy as np

P = {p["nom"]: p for p in json.load(open(sys.argv[1]))["profils"]}["L 120x120x15"]
E, G = 210000.0, 81000.0
Iuu = Ivv = P["Iy"] * 1e4
Iuv = P["Iyz"] * 1e4
It = P["It"] * 1e4
e = 15.0
uS = yS = e / 2
Lt, h = 2000.0, 5.0
ne = int(Lt / h)
nn = ne + 1
z = np.linspace(0, Lt, nn)
R, X, Y = 42760.0, 75.0, 85.0
print(f"EI = {E*Iuu/1e12:.2f} kN·m² ; GIt = {G*It/1e12:.3f} kN·m² ; rapport {E*Iuu/(G*It):.1f}")


def k4(L):
    return np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                     [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]]) / L ** 3


def rigidite(GIt_fact=1.0):
    nd = 5 * nn
    K = np.zeros((nd, nd))
    kb = k4(h)
    I = {(0, 0): Iuu, (1, 1): Ivv, (0, 1): Iuv, (1, 0): Iuv}
    for el in range(ne):
        n1, n2 = el, el + 1
        idx = {0: [5 * n1, 5 * n1 + 1, 5 * n2, 5 * n2 + 1], 1: [5 * n1 + 2, 5 * n1 + 3, 5 * n2 + 2, 5 * n2 + 3]}
        for (a, b), Iab in I.items():
            K[np.ix_(idx[a], idx[b])] += E * Iab * kb
        it = [5 * n1 + 4, 5 * n2 + 4]
        K[np.ix_(it, it)] += G * It * GIt_fact / h * np.array([[1, -1], [-1, 1]])
    return K


def charge(z0=250.0, z1=550.0, Rt=R, Xc=X):
    F = np.zeros(5 * nn)
    q = Rt / (z1 - z0)
    for i in range(nn):
        lo, hi = max(z[i] - h / 2, z0), min(z[i] + h / 2, z1)
        if hi > lo:
            f = q * (hi - lo)
            F[5 * i + 2] -= f                     # vertical vers le bas
            F[5 * i + 4] += -(Xc - uS) * f        # couple autour du centre de cisaillement
    return F


def resoudre(K0, F, ancr=(200.0, 1000.0, 1800.0), kN=1e9, kV=1e9, kmur=1e7, haut=False, uV=e / 2, yA=Y):
    ia = [int(round(za / h)) for za in ancr]
    # vecteurs de liaison : Δu(point (u, y)) = w - φ (y - yS) ; Δy = v + φ (u - uS)
    def cu(i, y):
        return (np.array([5 * i, 5 * i + 4]), np.array([1.0, -(y - yS)]))
    def cy(i, u):
        return (np.array([5 * i + 2, 5 * i + 4]), np.array([1.0, (u - uS)]))
    def add(K, g, k):
        d, c = g
        K[np.ix_(d, d)] += k * np.outer(c, c)
    def dot(g, U):
        d, c = g
        return float(c @ U[d])
    Ga = [cu(i, yA) for i in ia]
    Gv = [cy(i, uV) for i in ia]
    Gm = [cu(i, 0.0) for i in range(nn)]
    Gh = [cu(i, 120.0) for i in range(nn)] if haut else []
    trib = np.full(nn, h); trib[0] = trib[-1] = h / 2
    act_a = np.ones(len(ia), bool)
    act_m = np.ones(nn, bool)
    act_h = np.zeros(nn, bool)
    for it in range(200):
        K = K0.copy()
        for g, a in zip(Ga, act_a):
            if a: add(K, g, kN)
        for g in Gv: add(K, g, kV)
        for i, (g, a) in enumerate(zip(Gm, act_m)):
            if a: add(K, g, kmur * trib[i])
        for i, (g, a) in enumerate(zip(Gh, act_h)):
            if a: add(K, g, kmur * trib[i])
        U = np.linalg.solve(K, F)
        da = np.array([dot(g, U) for g in Ga])
        dm = np.array([dot(g, U) for g in Gm])
        dh = np.array([dot(g, U) for g in Gh]) if haut else np.zeros(0)
        na, nm = da > 0, dm < 0
        nh = dh < 0 if haut else act_h
        if np.array_equal(na, act_a) and np.array_equal(nm, act_m) and np.array_equal(nh, act_h):
            break
        act_a, act_m, act_h = na, nm, nh
    N = kN * da * act_a
    V = np.array([-kV * dot(g, U) for g in Gv])
    C = -kmur * trib * dm * act_m
    Ch = (-kmur * trib * dh * act_h).sum() if haut else 0.0
    # torsion et moment longitudinal
    phi = U[4::5]
    T = G * It * np.diff(phi) / h
    w2 = np.zeros(ne); v2 = np.zeros(ne)
    for el in range(ne):
        n1 = el
        # courbure au milieu de l'élément (Hermite)
        def courb(d1, t1, d2, t2):
            return (t2 - t1) / h          # courbure moyenne de l'élément (exacte pour un cubique)
        w2[el] = courb(U[5*n1], U[5*n1+1], U[5*n1+5], U[5*n1+6])
        v2[el] = courb(U[5*n1+2], U[5*n1+3], U[5*n1+7], U[5*n1+8])
    Mx = E * (Ivv * v2 + Iuv * w2)          # moment de flexion « vertical » (plan y-z)
    My = E * (Iuu * w2 + Iuv * v2)
    # contrainte longitudinale aux fibres extrêmes : σ = -E (w'' (u-uc) + v'' (y-yc))
    uc = yc = P["yc"]
    pts = [(0, 120), (15, 120), (120, 0), (120, 15), (0, 0)]
    sig = max(abs(E * (w2 * (u - uc) + v2 * (y - yc))).max() for u, y in pts)
    return dict(N=N, V=V, C=C.sum(), Ch=Ch, T=T, tau=np.abs(T).max() * e / It, Mx=Mx, sig=sig, it=it)


K0 = rigidite()
F = charge()
print(f"contrôle : ΣF vertical = {-F[2::5].sum()/1e3:.3f} kN ; couple total = {-F[4::5].sum()/1e6:.4f} kN·m (R·(X−e/2) = {R*(X-e/2)/1e6:.4f})")
# règle du levier (poutre infiniment raide)
zs = np.array([200.0, 1000.0, 1800.0]); zg = zs.mean(); ez = 400.0 - zg
Vl = R / 3 + R * ez * (zs - zg) / ((zs - zg) ** 2).sum()
print("règle du levier V =", np.round(Vl / 1e3, 3))
H0 = R * (X - e / 2) / Y
print(f"R/n : V = {R/3e3:.3f} ; N = H0/3 = {H0/3e3:.3f}")
kloc = 81.4e3
for nom, kw in [("rigides", dict()),
                ("raides 100/30 + aile", dict(kN=1 / (1 / 100e3 + 1 / kloc), kV=30e3)),
                ("souples 10/5 + aile", dict(kN=1 / (1 / 10e3 + 1 / kloc), kV=5e3)),
                ("rigides, levier permis", dict(haut=True))]:
    o = resoudre(K0, F, **kw)
    print(f"{nom:24s} N = {np.round(o['N']/1e3,2)} V = {np.round(o['V']/1e3,2)} N1/(H/3) = {o['N'][0]/(H0/3):.2f}"
          f" τ = {o['tau']:.1f} ({o['tau']/(355/1.5/math.sqrt(3)):.0%}) |Mx| = {np.abs(o['Mx']).max()/1e6:.3f} kN·m"
          f" σz max = {o['sig']:.1f} ; ΣV = {o['V'].sum()/1e3:.3f} ; ΣN−ΣC−Ch = {(o['N'].sum()-o['C']-o['Ch'])/1e3:.4f} ; it {o['it']}")
o = resoudre(rigidite(1000.0), F)
print(f"torsion ×1000 : N = {np.round(o['N']/1e3,2)}")
# variante ancrage au droit de la charge
F2 = charge()
for nom, kw in [("400/1000/1600 rigides", dict(ancr=(400.0, 1000.0, 1600.0))),
                ("400/1000/1600 souples", dict(ancr=(400.0, 1000.0, 1600.0), kN=1 / (1 / 10e3 + 1 / kloc), kV=5e3))]:
    o = resoudre(K0, F2, **kw)
    print(f"{nom:24s} N = {np.round(o['N']/1e3,2)} V = {np.round(o['V']/1e3,2)} τ = {o['tau']:.1f}")
# poutre continue (flexion verticale seule, Iuv = 0) : réactions sur trois appuis rigides
def continue_seule():
    K = np.zeros((2 * nn, 2 * nn))
    for el in range(ne):
        idx = [2 * el, 2 * el + 1, 2 * el + 2, 2 * el + 3]
        K[np.ix_(idx, idx)] += E * Ivv * k4(h)
    Fv = -F[2::5]
    Fz = np.zeros(2 * nn); Fz[0::2] = -Fv
    ia = [int(round(za / h)) for za in (200, 1000, 1800)]
    for i in ia: K[2 * i, 2 * i] += 1e12
    U = np.linalg.solve(K, Fz)
    return np.array([-1e12 * U[2 * i] for i in ia])
print("poutre continue, flexion verticale seule, appuis rigides : V =", np.round(continue_seule() / 1e3, 3))

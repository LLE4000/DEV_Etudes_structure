# -*- coding: utf-8 -*-
"""Éléments finis PLANS de la coupe transversale d'une cornière ancrée (contrainte plane, 1 mm de longueur).

Maillage structuré en quadrilatères à 4 nœuds : aile horizontale (bloc A), aile verticale (bloc B), et le coin
(blocs C1, C2 symétriques par rapport à la bissectrice) maillé par interpolation transfinie (Coons) avec le
CONGÉ EXACT (arc de rayon r tangent aux deux faces intérieures). Bouts d'aile à angle vif (r2 négligé).

Conditions aux limites (la physique du classeur, rendue explicite) :
- charge q = R / b (N par mm de longueur) au nœud (X + Xj, e) de la face supérieure de l'aile horizontale ;
- mur : face x = 0, contact UNILATÉRAL (le mur pousse, ne tire pas), rigide ou élastique (Winkler k_w, N/mm³) ;
- cheville : la rondelle (diamètre d_w) appuie sur la face x = e de l'aile verticale autour de y = Y — contact
  unilatéral rigide, ou ressorts (raideur de la cheville) ; la tige reprend l'effort vertical (u_y = 0 sur la
  ligne y = Y, x ∈ [0, e]). Pas de frottement au mur.

Algorithme de contact : ensemble actif (on libère un nœud qui tire, on bloque un nœud qui pénètre), jusqu'à
stabilité. Résolution creuse (scipy). Contraintes aux points de Gauss extrapolées aux nœuds et moyennées.
"""
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E_ACIER, NU = 210000.0, 0.3


# ---------------------------------------------------------------------------------------------- maillage
def _grille(fonc, n_xi, n_eta):
    xi = np.linspace(0.0, 1.0, n_xi + 1)
    eta = np.linspace(0.0, 1.0, n_eta + 1)
    XI, ETA = np.meshgrid(xi, eta, indexing="ij")
    P = fonc(XI, ETA)                                     # (n_xi+1, n_eta+1, 2)
    idx = np.arange((n_xi + 1) * (n_eta + 1)).reshape(n_xi + 1, n_eta + 1)
    el = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
    return P.reshape(-1, 2), el


def mailler(larg, haut, e, r, h=0.5, n_arc=None):
    nt = max(4, int(round(e / h)))
    blocs = []
    # A : aile horizontale x ∈ [e+r, larg], y ∈ [0, e]
    nA = int(round((larg - e - r) / h))
    blocs.append(_grille(lambda XI, ETA: np.stack([e + r + XI * (larg - e - r), ETA * e], -1), nA, nt))
    # B : aile verticale x ∈ [0, e], y ∈ [e+r, haut]
    nB = int(round((haut - e - r) / h))
    blocs.append(_grille(lambda XI, ETA: np.stack([XI * e, e + r + ETA * (haut - e - r)], -1), nt, nB))
    # C1 : coin, sous la bissectrice
    c = e + r
    Mx = c - r / math.sqrt(2)
    nc = n_arc or max(8, int(round(c / (1.4 * h))))

    def coons(XI, ETA):
        P00, P10, P01, P11 = np.array([0, 0.]), np.array([c, 0.]), np.array([Mx, Mx]), np.array([c, e])
        bas = np.stack([XI * c, 0 * XI], -1)
        th = np.radians(225 + 45 * XI)
        haut_ = np.stack([c + r * np.cos(th), c + r * np.sin(th)], -1)
        gau = ETA[..., None] * P01
        dro = np.stack([c + 0 * ETA, ETA * e], -1)
        xi, et = XI[..., None], ETA[..., None]
        return ((1 - et) * bas + et * haut_ + (1 - xi) * gau + xi * dro
                - ((1 - xi) * (1 - et) * P00 + xi * (1 - et) * P10 + (1 - xi) * et * P01 + xi * et * P11))

    blocs.append(_grille(coons, nc, nt))
    P, el = blocs[-1]
    blocs.append((P[:, ::-1].copy(), el[:, ::-1].copy()))            # C2 = miroir (x ↔ y), orientation rétablie
    # fusion des nœuds communs
    pts, els, dec = [], [], 0
    for P, el in blocs:
        pts.append(P)
        els.append(el + dec)
        dec += len(P)
    pts = np.vstack(pts)
    els = np.vstack(els)
    cle = np.round(pts, 6)
    _, inv_first, inv = np.unique(cle, axis=0, return_index=True, return_inverse=True)
    nodes = pts[inv_first]
    els = inv.ravel()[els]
    return nodes, els


# ---------------------------------------------------------------------------------------------- éléments Q4
G = 1 / math.sqrt(3)
GP = np.array([[-G, -G], [G, -G], [G, G], [-G, G]])
NREF = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)


def _dN(xi, eta):
    return 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])


DN = np.array([_dN(*g) for g in GP])                                  # (4 gp, 2, 4)


def matrice_D(E=E_ACIER, nu=NU):
    return E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


def _B(nodes, els):
    Xe = nodes[els]                                                    # (ne, 4, 2)
    J = np.einsum("gij,ejk->egik", DN, Xe)                             # (ne, 4, 2, 2)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    inv = np.empty_like(J)
    inv[..., 0, 0], inv[..., 1, 1] = J[..., 1, 1] / det, J[..., 0, 0] / det
    inv[..., 0, 1], inv[..., 1, 0] = -J[..., 0, 1] / det, -J[..., 1, 0] / det
    dNx = np.einsum("egij,gjk->egik", inv, DN)                         # (ne, 4, 2, 4)
    ne = len(els)
    B = np.zeros((ne, 4, 3, 8))
    B[:, :, 0, 0::2] = dNx[:, :, 0, :]
    B[:, :, 1, 1::2] = dNx[:, :, 1, :]
    B[:, :, 2, 0::2] = dNx[:, :, 1, :]
    B[:, :, 2, 1::2] = dNx[:, :, 0, :]
    return B, det


def assembler(nodes, els, D):
    B, det = _B(nodes, els)
    if (det <= 0).any():
        raise RuntimeError(f"{(det <= 0).sum()} points de Gauss à jacobien négatif")
    Ke = np.einsum("egji,jk,egkl,eg->eil", B, D, B, det)               # (ne, 8, 8)
    dofs = np.stack([2 * els, 2 * els + 1], -1).reshape(len(els), 8)
    I = np.repeat(dofs, 8, axis=1).ravel()
    Jj = np.tile(dofs, (1, 8)).ravel()
    n = 2 * len(nodes)
    K = sp.coo_matrix((Ke.ravel(), (I, Jj)), shape=(n, n)).tocsr()
    return K, B, det


# ---------------------------------------------------------------------------------------------- le modèle
class Coupe:
    def __init__(self, larg, haut, e, r, h=0.5):
        self.larg, self.haut, self.e, self.r = larg, haut, e, r
        self.nodes, self.els = mailler(larg, haut, e, r, h)
        self.D = matrice_D()
        self.K, self.B, self.det = assembler(self.nodes, self.els, self.D)
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        self.mur = np.where(np.abs(x) < 1e-9)[0]
        self.mur = self.mur[np.argsort(y[self.mur])]
        ym = y[self.mur]
        w = np.zeros(len(ym))                                          # longueur tributaire de chaque nœud du mur
        w[:-1] += np.diff(ym) / 2
        w[1:] += np.diff(ym) / 2
        self.w_mur = w
        self.h = h

    def noeud(self, x, y):
        d = np.hypot(self.nodes[:, 0] - x, self.nodes[:, 1] - y)
        i = int(np.argmin(d))
        if d[i] > 1e-6:
            raise ValueError(f"pas de nœud en ({x}, {y}) : le plus proche est à {d[i]:.3g} mm")
        return i

    def resoudre(self, q, X, Y, d_w=30.0, k_tige=None, k_mur=None, x_V=None, max_iter=80):
        """Cas de charge et contact.

        q       charge en N par mm de longueur, au nœud (X, e) ;
        d_w     diamètre de la rondelle (mm) : elle appuie sur la face x = e, y ∈ [Y − d_w/2, Y + d_w/2] ;
        k_tige  raideur AXIALE des chevilles, en N/mm PAR MM de longueur (n·k₁/b) ; None = tige indéformable ;
        k_mur   module de réaction du mur (N/mm³, Winkler) ; None = mur rigide ;
        x_V     abscisse du point où la tige reprend l'effort vertical (défaut e/2 : milieu de l'appui dans le trou).

        La rondelle est un CORPS RIGIDE en translation (un degré de liberté maître u_w), relié au sol par la tige
        (ressort k_tige) ; le contact rondelle–aile et le contact aile–mur sont UNILATÉRAUX (ensemble actif).
        Aucun encastrement fictif : la tige reprend l'effort vertical en UN point.
        """
        e = self.e
        x, y = self.nodes[:, 0], self.nodes[:, 1]
        nn = 2 * len(self.nodes)
        n = nn + 1                                                      # + le degré de liberté de la rondelle
        w_ = nn
        F = np.zeros(n)
        F[2 * self.noeud(X, e) + 1] = -q
        xv = e / 2 if x_V is None else x_V
        d = np.hypot(x - xv, y - Y)
        i_tige = int(np.argmin(d))
        rond = np.where((np.abs(x - e) < 1e-9) & (y >= Y - d_w / 2 - 1e-9) & (y <= Y + d_w / 2 + 1e-9))[0]
        rond = rond[np.argsort(y[rond])]
        kw_ = np.zeros(n)
        if k_tige is not None:
            kw_[w_] = k_tige
        Kf = (sp.block_diag((self.K, sp.csr_matrix((1, 1))), format="csr") + sp.diags(kw_)).tocsr()
        Kmur = sp.csr_matrix((n, n))
        actif_mur = np.ones(len(self.mur), bool)
        actif_rond = np.ones(len(rond), bool)
        for it in range(max_iter):
            fixes = {2 * i_tige + 1}
            if k_tige is None:
                fixes.add(w_)
            K = Kf
            if k_mur is None:
                fixes |= set(int(2 * i) for i in self.mur[actif_mur])
            else:
                kd = np.zeros(n)
                kd[2 * self.mur] = np.where(actif_mur, k_mur * self.w_mur, 0.0)
                K = Kf + sp.diags(kd)
            lies = [int(2 * i) for i in rond[actif_rond]]
            # matrice de liaison T : u_complet = T · u_réduit
            exclus = np.zeros(n, bool)
            exclus[list(fixes)] = True
            exclus[lies] = True
            red = np.where(~exclus)[0].tolist()
            col = {j: k for k, j in enumerate(red)}
            rows, cols = list(red), [col[j] for j in red]
            if w_ in col:
                rows += lies
                cols += [col[w_]] * len(lies)
            Tm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(red)))
            Kr = (Tm.T @ K @ Tm).tocsc()
            ur = spla.spsolve(Kr, Tm.T @ F)
            u = Tm @ ur
            Rr = K @ u - F
            ux_m = u[2 * self.mur]
            if k_mur is None:
                nouveau_m = np.where(actif_mur, Rr[2 * self.mur] > -1e-9, ux_m < -1e-12)
            else:
                nouveau_m = ux_m < 0
            gap = u[2 * rond] - u[w_]                                   # > 0 : l'aile entrerait dans la rondelle
            nouveau_r = np.where(actif_rond, Rr[2 * rond] < 1e-9, gap > 1e-12)
            if (nouveau_m == actif_mur).all() and (nouveau_r == actif_rond).all():
                break
            actif_mur, actif_rond = nouveau_m, nouveau_r
        else:
            raise RuntimeError("contact : pas de convergence")
        if k_mur is None:
            fm = np.where(actif_mur, Rr[2 * self.mur], 0.0)
        else:
            fm = np.where(actif_mur, -k_mur * self.w_mur * u[2 * self.mur], 0.0)
        fr = np.where(actif_rond, Rr[2 * rond], 0.0)
        self.u = u[:nn]
        self.res = dict(iterations=it + 1, mur_y=y[self.mur], mur_f=fm, C=fm.sum(),
                        yc=(fm * y[self.mur]).sum() / fm.sum() if fm.sum() > 0 else float("nan"),
                        rond_y=y[rond], rond_f=fr, T=-fr.sum(), V=Rr[2 * i_tige + 1],
                        x_tige=float(x[i_tige]), u_rondelle=float(u[w_]),
                        T_ressort=(k_tige * u[w_]) if k_tige is not None else None)
        self._contraintes()
        return self.res

    # ------------------------------------------------------------------ contraintes
    def _contraintes(self):
        dofs = np.stack([2 * self.els, 2 * self.els + 1], -1).reshape(len(self.els), 8)
        ue = self.u[dofs]                                              # (ne, 8)
        sg = np.einsum("ij,egjk,ek->egi", self.D, self.B, ue)          # (ne, 4 gp, 3)
        # extrapolation des points de Gauss aux nœuds
        Ext = np.array([[0.25 * (1 + a * gx * math.sqrt(3)) * (1 + b * gy * math.sqrt(3))
                         for (gx, gy) in NREF] for (a, b) in NREF])    # (4 nœuds, 4 gp)
        sn = np.einsum("ng,egi->eni", Ext, sg)
        acc = np.zeros((len(self.nodes), 3))
        cnt = np.zeros(len(self.nodes))
        np.add.at(acc, self.els.ravel(), sn.reshape(-1, 3))
        np.add.at(cnt, self.els.ravel(), 1)
        self.sig = acc / cnt[:, None]                                  # σxx, σyy, τxy moyennés aux nœuds
        sx, sy, t = self.sig.T
        self.vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * t ** 2)
        c, rr = (sx + sy) / 2, np.hypot((sx - sy) / 2, t)
        self.s1, self.s2 = c + rr, c - rr
        Xe = self.nodes[self.els]
        self._bb = np.stack([Xe[..., 0].min(1), Xe[..., 0].max(1), Xe[..., 1].min(1), Xe[..., 1].max(1)], -1)

    def echantillon(self, pts):
        """σxx, σyy, τxy interpolés aux points (n, 2)."""
        out = np.full((len(pts), 3), np.nan)
        tol = 1e-7
        for k, p in enumerate(pts):
            cand = np.where((self._bb[:, 0] - tol <= p[0]) & (p[0] <= self._bb[:, 1] + tol)
                            & (self._bb[:, 2] - tol <= p[1]) & (p[1] <= self._bb[:, 3] + tol))[0]
            for el in cand:
                Xe = self.nodes[self.els[el]]
                xi = np.zeros(2)
                for _ in range(20):
                    Nn = 0.25 * (1 + xi[0] * NREF[:, 0]) * (1 + xi[1] * NREF[:, 1])
                    f = Nn @ Xe - p
                    J = _dN(*xi) @ Xe
                    dx = np.linalg.solve(J.T, f)
                    xi -= dx
                    if np.abs(dx).max() < 1e-12:
                        break
                if np.abs(xi).max() <= 1 + 1e-6:
                    Nn = 0.25 * (1 + xi[0] * NREF[:, 0]) * (1 + xi[1] * NREF[:, 1])
                    out[k] = Nn @ self.sig[self.els[el]]
                    break
        return out

    def coupe(self, A, Bp, n=200):
        """Résultantes sur la coupe [A, B] (par mm de longueur) : N (traction > 0, selon la normale), M (au milieu),
        V ; et la distribution de σ_n le long de la coupe. Normale = rotation de +90° de la direction A→B."""
        A, Bp = np.asarray(A, float), np.asarray(Bp, float)
        L = np.linalg.norm(Bp - A)
        d = (Bp - A) / L
        nrm = np.array([d[1], -d[0]])
        gx, gw = np.polynomial.legendre.leggauss(n)
        s = (gx + 1) / 2 * L
        w = gw / 2 * L
        pts = A + s[:, None] * d
        S = self.echantillon(pts)
        sx, sy, t = S.T
        sn = sx * nrm[0] ** 2 + sy * nrm[1] ** 2 + 2 * t * nrm[0] * nrm[1]
        tn = (sy - sx) * nrm[0] * nrm[1] + t * (nrm[0] ** 2 - nrm[1] ** 2)
        tn = sx * nrm[0] * d[0] + sy * nrm[1] * d[1] + t * (nrm[0] * d[1] + nrm[1] * d[0])
        N = (sn * w).sum()
        M = (sn * (s - L / 2) * w).sum()
        V = (tn * w).sum()
        # fibres extrêmes : un point à 1e-4 mm des bords
        ext = self.echantillon(np.array([A + 1e-4 * d, Bp - 1e-4 * d]))
        sn_ext = ext[:, 0] * nrm[0] ** 2 + ext[:, 1] * nrm[1] ** 2 + 2 * ext[:, 2] * nrm[0] * nrm[1]
        return dict(L=L, N=N, M=M, V=V, s=s, sn=sn, tn=tn, sn_ext=sn_ext,
                    lin_max=abs(N) / L + 6 * abs(M) / L ** 2, sn_max=np.nanmax(np.abs(np.r_[sn, sn_ext])))

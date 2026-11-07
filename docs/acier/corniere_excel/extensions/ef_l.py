# -*- coding: utf-8 -*-
"""Prototype d'éléments finis plans pour la COUPE d'une cornière (numpy seul).

Ce module ne charge que numpy : il doit pouvoir tourner dans le navigateur
(Pyodide 0.29.5 + numpy 2.2.5, sans scipy).

Modèle : tranche de 1 mm de la cornière, déformation plane (εz = 0), élément
quadrilatère à 9 nœuds (Lagrange), intégration de Gauss 3×3. Maillage
structuré par blocs (interpolation transfinie), congé exact.

Appuis (configuration A, aile horizontale en bas) :
  - mur : face extérieure de l'aile verticale (u = 0), contact UNILATÉRAL
    (ressorts actifs seulement en compression), raideur de Winkler k_mur
    (N/mm³) × longueur tributaire ;
  - ancrage, traction : ressorts unilatéraux sur la face intérieure (u = e)
    dans la zone d'appui de l'écrou ou de la rondelle (demi-largeur b_rond),
    raideur totale k_anc (N/mm par mm de longueur de cornière) ;
  - ancrage, cisaillement : appui vertical au nœud de mi-épaisseur à
    y = Y' (portance de la tige dans le trou).
Charge : q = R/L (N/mm) répartie sur 10 mm autour de u = X' sur la face
supérieure de l'aile horizontale.

Configuration B (aile horizontale en haut) : la géométrie est retournée
(y → H − y) ; la charge s'applique sur la face supérieure (y = H).

Solveur : 'bande' (Cholesky par panneaux sur une matrice bande, numérotation
de Cuthill–McKee inverse — numpy seul) ou 'scipy' (référence CPython).
"""
import math
import time

import numpy as np

# ------------------------------------------------------------------ Q9
_G = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
_W = np.array([5 / 9, 8 / 9, 5 / 9])


def _l(x):
    return np.array([x * (x - 1) / 2, 1 - x * x, x * (x + 1) / 2])


def _dl(x):
    return np.array([x - 0.5, -2 * x, x + 0.5])


def forme_q9(xi, eta):
    """N (9,), dN/dξ (9,), dN/dη (9,) ; nœud (i, j) -> indice 3·j + i."""
    lx, ly, dx, dy = _l(xi), _l(eta), _dl(xi), _dl(eta)
    N = np.outer(ly, lx).ravel()
    dNx = np.outer(ly, dx).ravel()
    dNy = np.outer(dy, lx).ravel()
    return N, dNx, dNy


# ------------------------------------------------------------------ maillage
def _grille(n, grad=1.0):
    """2n+1 abscisses dans [0, 1] (nœuds milieux au milieu de chaque élément) ;
    grad > 1 resserre près de 0 (progression géométrique)."""
    if abs(grad - 1.0) < 1e-12:
        a = np.linspace(0, 1, n + 1)
    else:
        q = grad ** (1.0 / max(n - 1, 1))
        h = np.array([q ** k for k in range(n)])
        a = np.concatenate([[0.0], np.cumsum(h)]) / h.sum()
    s = np.empty(2 * n + 1)
    s[0::2] = a
    s[1::2] = 0.5 * (a[:-1] + a[1:])
    return s


def _coons(bas, haut, gauche, droite, s, t):
    """Interpolation transfinie : bas(s), haut(s), gauche(t), droite(t)."""
    P00, P10 = bas(0.0), bas(1.0)
    P01, P11 = haut(0.0), haut(1.0)
    S, T = np.meshgrid(s, t, indexing="ij")
    out = np.zeros(S.shape + (2,))
    for a in range(S.shape[0]):
        for b in range(S.shape[1]):
            ss, tt = S[a, b], T[a, b]
            out[a, b] = ((1 - tt) * bas(ss) + tt * haut(ss) + (1 - ss) * gauche(tt) + ss * droite(tt)
                         - ((1 - ss) * (1 - tt) * P00 + ss * (1 - tt) * P10 + (1 - ss) * tt * P01 + ss * tt * P11))
    return out


def mailler(geo, n_t=8, n_c=12, n_leg=30, grad=4.0):
    """Maillage Q9 de la cornière (configuration A).

    geo : dict(B=larg. aile horizontale, H=haut. aile verticale, e, r, Y=ordonnée
    de l'ancrage). Renvoie noeuds (n,2), elements (ne,9), blocs.
    """
    B, H, e, r, Ya = geo["B"], geo["H"], geo["e"], geo["r"], geo["Y"]
    a = e + r
    Cc = np.array([a, a])
    Pm = Cc + r * np.array([-1, -1]) / math.sqrt(2)
    lin = lambda P, Q: (lambda s: np.asarray(P, float) + s * (np.asarray(Q, float) - np.asarray(P, float)))
    arc = lambda th0, th1: (lambda s: Cc + r * np.array([math.cos(th0 + s * (th1 - th0)),
                                                         math.sin(th0 + s * (th1 - th0))]))
    blocs = []
    st = _grille(n_t)
    sc = _grille(n_c)
    # aile horizontale : s le long de u (resserré près du pied du congé), t dans l'épaisseur
    sl = _grille(n_leg, grad)
    blocs.append(_coons(lin((a, 0), (B, 0)), lin((a, e), (B, e)), lin((a, 0), (a, e)), lin((B, 0), (B, e)), sl, st))
    # aile verticale, de y = a à l'ancrage puis de l'ancrage au bout
    n1 = max(4, int(round(n_leg * (Ya - a) / (H - a))))
    n2 = max(3, n_leg - n1)
    s1 = _grille(n1, grad)
    s2 = _grille(n2)
    blocs.append(_coons(lin((0, a), (0, Ya)), lin((e, a), (e, Ya)), lin((0, a), (e, a)), lin((0, Ya), (e, Ya)), s1, st))
    blocs.append(_coons(lin((0, Ya), (0, H)), lin((e, Ya), (e, H)), lin((0, Ya), (e, Ya)), lin((0, H), (e, H)), s2, st))
    # bloc d'angle sous la diagonale : (0,0) (a,0) (a,e) Pm
    blocs.append(_coons(lin((0, 0), (a, 0)), arc(5 * math.pi / 4, 3 * math.pi / 2), lin((0, 0), Pm), lin((a, 0), (a, e)),
                        sc, st))
    # bloc d'angle au-dessus de la diagonale : (0,0) (0,a) (e,a) Pm
    blocs.append(_coons(lin((0, 0), (0, a)), arc(5 * math.pi / 4, math.pi), lin((0, 0), Pm), lin((0, a), (e, a)),
                        sc, st))
    # fusion des nœuds
    cles = {}
    noeuds = []
    elements = []
    for g in blocs:
        idx = np.empty(g.shape[:2], dtype=int)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                k = (round(g[i, j, 0], 6), round(g[i, j, 1], 6))
                if k not in cles:
                    cles[k] = len(noeuds)
                    noeuds.append(g[i, j])
                idx[i, j] = cles[k]
        for i in range(0, g.shape[0] - 2, 2):
            for j in range(0, g.shape[1] - 2, 2):
                elements.append([idx[i + ii, j + jj] for jj in range(3) for ii in range(3)])
    return np.array(noeuds), np.array(elements, dtype=int)


# ------------------------------------------------------------------ rigidité
def matrice_D(E=210000.0, nu=0.3):
    f = E / ((1 + nu) * (1 - 2 * nu))
    return f * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])


def _B_elem(X, xi, eta):
    """X (ne,9,2) -> B (ne,3,18), detJ (ne,)."""
    N, dNx, dNy = forme_q9(xi, eta)
    dN = np.stack([dNx, dNy])                       # (2,9)
    J = np.einsum("ak,ekb->eab", dN, X)             # (ne,2,2) J[a,b] = d x_b / d ξ_a
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    inv = np.empty_like(J)
    inv[:, 0, 0] = J[:, 1, 1] / det
    inv[:, 1, 1] = J[:, 0, 0] / det
    inv[:, 0, 1] = -J[:, 0, 1] / det
    inv[:, 1, 0] = -J[:, 1, 0] / det
    dNxy = np.einsum("eba,ak->ebk", inv, dN)        # dN/dx_b = Σ_a (J⁻¹)[b,a] dN/dξ_a
    ne = X.shape[0]
    B = np.zeros((ne, 3, 18))
    B[:, 0, 0::2] = dNxy[:, 0]
    B[:, 1, 1::2] = dNxy[:, 1]
    B[:, 2, 0::2] = dNxy[:, 1]
    B[:, 2, 1::2] = dNxy[:, 0]
    return B, det


def rigidites(noeuds, elements, D):
    X = noeuds[elements]
    Ke = np.zeros((len(elements), 18, 18))
    detmin = np.inf
    for i, xi in enumerate(_G):
        for j, eta in enumerate(_G):
            B, det = _B_elem(X, xi, eta)
            detmin = min(detmin, det.min())
            Ke += np.einsum("eki,kl,elj->eij", B, D, B) * (np.abs(det) * _W[i] * _W[j])[:, None, None]
    return Ke, detmin


def ddl_elements(elements):
    d = np.empty((len(elements), 18), dtype=int)
    d[:, 0::2] = 2 * elements
    d[:, 1::2] = 2 * elements + 1
    return d


# ------------------------------------------------------------------ numérotation et solveur bande
def cuthill_mckee_inverse(n, elements, depart):
    voisins = [set() for _ in range(n)]
    for el in elements:
        for a in el:
            voisins[a].update(el)
    for a in range(n):
        voisins[a].discard(a)
    deg = [len(v) for v in voisins]
    ordre = []
    vu = np.zeros(n, dtype=bool)
    file = [depart]
    vu[depart] = True
    tete = 0
    while tete < len(file):
        a = file[tete]
        tete += 1
        ordre.append(a)
        nv = sorted((b for b in voisins[a] if not vu[b]), key=lambda b: deg[b])
        for b in nv:
            vu[b] = True
            file.append(b)
    if len(ordre) < n:
        raise ValueError("maillage non connexe")
    ordre = ordre[::-1]
    rang = np.empty(n, dtype=int)
    rang[np.array(ordre)] = np.arange(n)
    return rang


class Bande:
    """Matrice symétrique définie positive en stockage bande supérieure U[i, d] = A[i, i+d]."""

    def __init__(self, n, b):
        self.n, self.b = n, b
        self.U = np.zeros((n, b + 1))

    def assembler(self, ddl_perm, Ke):
        ne = Ke.shape[0]
        I = np.repeat(ddl_perm[:, :, None], 18, axis=2)
        J = np.repeat(ddl_perm[:, None, :], 18, axis=1)
        m = J >= I
        np.add.at(self.U, (I[m], (J - I)[m]), Ke[m])

    def cholesky(self, nb=64):
        """Factorisation A = Rᵀ R par panneaux (R triangulaire supérieure, même bande)."""
        n, b = self.n, self.b
        U = self.U.copy()
        for k0 in range(0, n, nb):
            k1 = min(k0 + nb, n)
            p = k1 - k0
            hi = min(n, k1 + b)
            m = hi - k0
            a = np.arange(m)
            A_, C_ = np.meshgrid(a, a, indexing="ij")
            lo = np.minimum(A_, C_)
            d = np.abs(A_ - C_)
            ok = d <= b
            W = np.where(ok, U[k0 + lo, np.minimum(d, b)], 0.0)
            W11 = W[:p, :p]
            L11 = np.linalg.cholesky(W11)
            W21 = W[p:, :p]
            if m > p:
                L21 = np.linalg.solve(L11, W21.T).T          # L21 = W21 L11⁻ᵀ
                W22 = W[p:, p:] - L21 @ L21.T
            # écrire R = Lᵀ pour les lignes du panneau
            Rp = np.zeros((p, m))
            Rp[:, :p] = L11.T
            if m > p:
                Rp[:, p:] = L21.T
            ii, jj = np.nonzero(np.triu(np.ones((p, m), dtype=bool)))
            dd = jj - ii
            sel = dd <= b
            U[k0 + ii[sel], dd[sel]] = Rp[ii[sel], jj[sel]]
            if m > p:
                q = m - p
                ii, jj = np.nonzero(np.triu(np.ones((q, q), dtype=bool)))
                dd = jj - ii
                sel = dd <= b
                U[k1 + ii[sel], dd[sel]] = W22[ii[sel], jj[sel]]
        self.R = U
        return U

    def resoudre(self, f):
        n, b, R = self.n, self.b, self.R
        y = np.array(f, dtype=float)
        # Rᵀ y = f : y_i = (f_i − Σ_{k<i} R[k,i] y_k)/R[i,i]  (colonne par colonne)
        for i in range(n):
            y[i] /= R[i, 0]
            j1 = min(n, i + b + 1)
            y[i + 1:j1] -= R[i, 1:j1 - i] * y[i]
        x = y
        for i in range(n - 1, -1, -1):
            j1 = min(n, i + b + 1)
            x[i] = (x[i] - R[i, 1:j1 - i] @ x[i + 1:j1]) / R[i, 0]
        return x


# ------------------------------------------------------------------ problème
class Probleme:
    def __init__(self, geo, config="A", maillage=None, E=210000.0, nu=0.3, solveur="bande", nb=64):
        self.geo, self.config, self.solveur, self.nb = dict(geo), config, solveur, nb
        t0 = time.perf_counter()
        g = dict(geo)                            # Y compté depuis la face extérieure de l'aile horizontale
        noeuds, elements = mailler(g, **(maillage or {}))
        if config == "B":
            noeuds = noeuds.copy()
            noeuds[:, 1] = geo["H"] - noeuds[:, 1]
        self.noeuds, self.elements = noeuds, elements
        self.D = matrice_D(E, nu)
        self.nu = nu
        self.Ke, self.detmin = rigidites(noeuds, elements, self.D)
        self.ddl = ddl_elements(elements)
        self.n = len(noeuds)
        self.t_maillage = time.perf_counter() - t0
        # numérotation
        t0 = time.perf_counter()
        depart = int(np.argmax(noeuds[:, 0] - 1e-3 * noeuds[:, 1]))
        rang = cuthill_mckee_inverse(self.n, elements, depart)
        self.perm_ddl = np.empty(2 * self.n, dtype=int)
        self.perm_ddl[0::2] = 2 * rang
        self.perm_ddl[1::2] = 2 * rang + 1
        dp = self.perm_ddl[self.ddl]
        self.b = int((dp.max(axis=1) - dp.min(axis=1)).max())
        if solveur == "bande":
            self.K0 = Bande(2 * self.n, self.b)
            self.K0.assembler(dp, self.Ke)
        else:
            import scipy.sparse as sp
            I = np.repeat(self.ddl[:, :, None], 18, axis=2).ravel()
            J = np.repeat(self.ddl[:, None, :], 18, axis=1).ravel()
            self.K0 = sp.coo_matrix((self.Ke.ravel(), (I, J)), shape=(2 * self.n, 2 * self.n)).tocsc()
        self.t_assemblage = time.perf_counter() - t0

    # ---- longueurs tributaires d'un ensemble de nœuds alignés
    @staticmethod
    def _tributaires(coord):
        o = np.argsort(coord)
        c = coord[o]
        w = np.zeros_like(c)
        if len(c) > 1:
            d = np.diff(c)
            w[:-1] += d / 2
            w[1:] += d / 2
        out = np.empty_like(w)
        out[o] = w
        return out

    def resoudre(self, q, X, k_mur=1e5, k_anc=1e7, b_rond=3.0, l_charge=10.0, it_max=40, depart="bas"):
        """q : charge linéique (N/mm), X : position de la charge depuis le nu du mur.
        k_mur : N/mm³ (Winkler), k_anc : N/mm par mm de cornière (raideur totale de l'ancrage)."""
        geo, nd = self.geo, self.noeuds
        e, H = geo["e"], geo["H"]
        tol = 1e-6
        yA = geo["Y"] if self.config == "A" else H - geo["Y"]
        if self.config == "A":
            y_face = e                    # face supérieure de l'aile horizontale
        else:
            y_face = H
        # charge : pression uniforme q/l_charge sur [X − l/2, X + l/2] de la face chargée,
        # forces nodales cohérentes (intégration de Gauss fine le long des arêtes quadratiques)
        F = np.zeros(2 * self.n)
        p = q / l_charge
        xg, wg = np.polynomial.legendre.leggauss(60)
        aretes = ((0, 1, 2), (6, 7, 8), (0, 3, 6), (2, 5, 8))
        for el in self.elements:
            for a in aretes:
                k = el[list(a)]
                if np.all(abs(nd[k, 1] - y_face) < tol):
                    xk = nd[k, 0]
                    for x_, w_ in zip(xg, wg):
                        Nl = _l(x_)
                        xx = Nl @ xk
                        if X - l_charge / 2 <= xx <= X + l_charge / 2:
                            jac = abs(_dl(x_) @ xk)
                            F[2 * k + 1] -= p * Nl * w_ * jac
        haut = np.nonzero(F[1::2] != 0)[0]
        F *= q / -F.sum()          # résultante exacte (la quadrature tronquée près des bornes)
        # mur
        mur = np.nonzero(abs(nd[:, 0]) < tol)[0]
        wm = self._tributaires(nd[mur, 1])
        # ancrage, traction : face intérieure u = e, |y − yA| ≤ b_rond
        ar = np.nonzero((abs(nd[:, 0] - e) < tol) & (abs(nd[:, 1] - yA) <= b_rond + tol))[0]
        if len(ar) == 1:
            wa = np.array([1.0])
        else:
            wa = self._tributaires(nd[ar, 1])
            wa = wa / wa.sum()
        # ancrage, cisaillement : nœud de mi-épaisseur à y = yA
        av = int(np.argmin((nd[:, 0] - e / 2) ** 2 + (nd[:, 1] - yA) ** 2))
        pen_v = 1e12
        if depart == "bas":
            # départ : contact présumé près de l'arête du bout comprimé (A : dessous ; B : bout libre)
            actif_mur = nd[mur, 1] <= (e if self.config == "A" else e)
        else:
            actif_mur = np.ones(len(mur), dtype=bool)
        actif_anc = np.ones(len(ar), dtype=bool)
        hist = []
        t0 = time.perf_counter()
        for it in range(it_max):
            diag = np.zeros(2 * self.n)
            diag[2 * mur[actif_mur]] += k_mur * wm[actif_mur]
            diag[2 * ar[actif_anc]] += k_anc * wa[actif_anc]
            diag[2 * av + 1] += pen_v
            Uu = self._solve(diag, F)
            ux = Uu[2 * mur]
            n_mur = (ux < 0)
            n_anc = (Uu[2 * ar] > 0)
            hist.append((int(actif_mur.sum()), int(actif_anc.sum())))
            if np.array_equal(n_mur, actif_mur) and np.array_equal(n_anc, actif_anc):
                break
            actif_mur, actif_anc = n_mur, n_anc
        self.t_resolution = time.perf_counter() - t0
        self.iterations = it + 1
        self.U = Uu
        fmur = np.where(actif_mur, -k_mur * wm * Uu[2 * mur], 0.0)       # > 0 : le mur pousse vers +u
        fanc = np.where(actif_anc, k_anc * wa * Uu[2 * ar], 0.0)        # > 0 : l'ancrage tire vers −u
        fv = -pen_v * Uu[2 * av + 1]
        self.reactions = dict(
            C=fmur.sum(), y_C=(fmur * nd[mur, 1]).sum() / max(fmur.sum(), 1e-30),
            mur_y=nd[mur, 1], mur_f=fmur, H=fanc.sum(),
            y_H=(fanc * nd[ar, 1]).sum() / max(fanc.sum(), 1e-30), V=fv,
            charge=-F.sum(), x_charge=(F[2 * haut + 1] * nd[haut, 0]).sum() / F.sum(),
        )
        return self.U

    def _solve(self, diag, F):
        if self.solveur == "bande":
            Kb = Bande(self.K0.n, self.K0.b)
            Kb.U = self.K0.U.copy()
            p = self.perm_ddl
            Kb.U[p, 0] += diag
            Kb.cholesky(self.nb)
            Fp = np.zeros_like(F)
            Fp[p] = F
            xp = Kb.resoudre(Fp)
            return xp[p]
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl
        return spl.spsolve(self.K0 + sp.diags(diag), F)

    # ---- contraintes
    def contrainte_point(self, el, xi, eta):
        X = self.noeuds[self.elements[el]][None]
        B, _ = _B_elem(X, xi, eta)
        ue = self.U[self.ddl[el]]
        s = self.D @ (B[0] @ ue)
        return s

    def contraintes_noeuds(self):
        """σx, σy, τxy moyennées aux nœuds ; von Mises en déformation plane."""
        acc = np.zeros((self.n, 3))
        cnt = np.zeros(self.n)
        pos = [-1.0, 0.0, 1.0]
        X = self.noeuds[self.elements]
        Ue = self.U[self.ddl]
        for j, eta in enumerate(pos):
            for i, xi in enumerate(pos):
                B, _ = _B_elem(X, xi, eta)
                s = np.einsum("kl,elj,ej->ek", self.D, B, Ue)
                k = self.elements[:, 3 * j + i]
                np.add.at(acc, k, s)
                np.add.at(cnt, k, 1)
        s = acc / cnt[:, None]
        sz = self.nu * (s[:, 0] + s[:, 1])
        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - sz) ** 2 + (sz - s[:, 0]) ** 2) + 3 * s[:, 2] ** 2)
        return s, vm

    def localiser(self, P, it=30):
        """Élément et coordonnées naturelles du point P (inversion de Newton)."""
        X = self.noeuds[self.elements]
        lo, hi = X.min(axis=1), X.max(axis=1)
        cand = np.nonzero((P[0] >= lo[:, 0] - 1e-6) & (P[0] <= hi[:, 0] + 1e-6)
                          & (P[1] >= lo[:, 1] - 1e-6) & (P[1] <= hi[:, 1] + 1e-6))[0]
        best = None
        for el in cand:
            xe = X[el]
            z = np.zeros(2)
            for _ in range(it):
                N, dNx, dNy = forme_q9(z[0], z[1])
                x = N @ xe
                J = np.array([dNx @ xe, dNy @ xe]).T     # dx/dξ, dx/dη en colonnes
                dz = np.linalg.solve(J, P - x)
                z += dz
                if np.abs(dz).max() < 1e-12:
                    break
            if np.abs(z).max() <= 1 + 1e-7:
                return el, z
            err = np.abs(z).max()
            if best is None or err < best[2]:
                best = (el, z, err)
        if best is not None and best[2] < 1 + 1e-4:
            return best[0], best[1]
        raise ValueError(f"point hors du maillage : {P}")

    def contrainte_en(self, P):
        el, z = self.localiser(np.asarray(P, float))
        return self.contrainte_point(el, z[0], z[1])

    def coupe(self, debut, fin, n=81, centre=None, normale="x"):
        """Résultantes sur une coupe droite (par mm de cornière) : N, V, M (autour du milieu).

        normale 'x' : coupe verticale (u = cte), σ = σx, τ = τxy, bras = y − y_milieu.
        normale 'y' : coupe horizontale (y = cte), σ = σy, τ = τxy, bras = u − u_milieu.
        """
        xg, wg = np.polynomial.legendre.leggauss(n)
        P0, P1 = np.asarray(debut, float), np.asarray(fin, float)
        Lc = np.linalg.norm(P1 - P0)
        mid = 0.5 * (P0 + P1) if centre is None else np.asarray(centre, float)
        Nn = Vv = Mm = 0.0
        prof = []
        for x, w in zip(xg, wg):
            P = P0 + (x + 1) / 2 * (P1 - P0)
            s = self.contrainte_en(P)
            sig = s[0] if normale == "x" else s[1]
            bras = (P[1] - mid[1]) if normale == "x" else (P[0] - mid[0])
            Nn += sig * w * Lc / 2
            Vv += s[2] * w * Lc / 2
            Mm += sig * bras * w * Lc / 2
            prof.append((P, s))
        return dict(N=Nn, V=Vv, M=Mm, profil=prof, longueur=Lc)

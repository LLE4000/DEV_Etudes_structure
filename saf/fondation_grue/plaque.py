# -*- coding: utf-8 -*-
"""Fondation de grue — la PLAQUE SUR SOL ÉLASTIQUE : une dalle en béton armé déformable, jamais rigide.

Le modèle est celui du calcul de référence (tests/fondation_grue_ref.py), aux mêmes hypothèses que le SAF et que SCIA :
- plaque mince (Kirchhoff), élément rectangle ACM à 12 degrés de liberté (w, ∂w/∂x, ∂w/∂y), grille régulière ; un
  élément est gardé si son centre est dans le contour ; une zone plus épaisse (plot coulé en place) donne son épaisseur
  aux éléments dont le centre est dedans ;
- sol : un ressort à chaque nœud (k × surface tributaire), plus C2 (Pasternak) ;
- sol en COMPRESSION SEULE : les nœuds qui se soulèvent sont libérés, et on recalcule jusqu'à stabilité ;
- charges nodales cohérentes, 8 × 8 points par élément.

CE QUI DIFFÈRE DE LA RÉFÉRENCE, ET SEULEMENT CELA : la résolution. Le navigateur n'a pas scipy ; la matrice est donc
résolue ici avec numpy seul.
- La grille numérotée ligne par ligne donne une matrice TRIDIAGONALE PAR BLOCS (un bloc par ligne de nœuds). Elle est
  factorisée une fois par géométrie et par k : S_0 = A_0, S_j = A_j − B_{j−1}ᵀ·S_{j−1}⁻¹·B_{j−1} ; on garde les S_j⁻¹.
  Tous les cas de charge sont ensuite de simples seconds membres. Un bloc B_j ne lie un nœud qu'à ses voisins : il est
  gardé par ses onze diagonales, et ses produits coûtent peu. Les blocs suivent le PETIT côté de la grille (des lignes
  de nœuds, ou des colonnes quand la dalle est plus longue que haute) : ils sont alors les plus petits possible.
- Libérer des ressorts ne refait pas la factorisation : K = K_0 − E·D·Eᵀ (E : les nœuds libérés, D : leurs ressorts),
  donc x = y + Z·(D⁻¹ − EᵀZ)⁻¹·(Eᵀy), avec y = K_0⁻¹F et Z = K_0⁻¹E (formule de Woodbury, exacte). Au-delà de
  ``SEUIL_WOODBURY`` nœuds libérés, on refactorise.
Les résultats sont ceux de la référence à l'arrondi de la machine près (tests/test_saf_grue_moteur.py le mesure).

Unités : kN, m, kPa, MN/m³. w positif vers le bas ; contrainte de contact σ = k·w.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

POINTS = 8                       # points de charge par côté d'élément (8 × 8)
_DEMI_BANDE = 5                  # un degré de liberté d'une ligne de nœuds ne touche, sur la ligne voisine, que ceux à ±5
ITERATIONS_MAX = 30              # du sol en compression seule
SEUIL_WOODBURY = 0.30            # part des nœuds libérés au-delà de laquelle on refactorise


# ------------------------------------------------------------------ l'élément ACM
def _base(x, y):
    p = np.array([1, x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3, x ** 3 * y, x * y ** 3])
    px = np.array([0, 1, 0, 2 * x, y, 0, 3 * x * x, 2 * x * y, y * y, 0, 3 * x * x * y, y ** 3])
    py = np.array([0, 0, 1, 0, x, 2 * y, 0, x * x, 2 * x * y, 3 * y * y, x ** 3, 3 * x * y * y])
    pxx = np.array([0, 0, 0, 2, 0, 0, 6 * x, 2 * y, 0, 0, 6 * x * y, 0])
    pyy = np.array([0, 0, 0, 0, 0, 2, 0, 0, 2 * x, 6 * y, 0, 6 * x * y])
    pxy = np.array([0, 0, 0, 0, 1, 0, 0, 2 * x, 2 * y, 0, 3 * x * x, 3 * y * y])
    return p, px, py, pxx, pyy, pxy


def element(a: float, b: float, D: float, nu: float, c2x: float, c2y: Optional[float] = None):
    """La rigidité de flexion (+ Pasternak C2x, C2y) d'un rectangle a × b, et la matrice C⁻¹ des coefficients du polynôme."""
    c2y = c2x if c2y is None else c2y
    C = np.zeros((12, 12))
    for i, (xn, yn) in enumerate([(0, 0), (a, 0), (a, b), (0, b)]):
        p, px, py, *_ = _base(xn, yn)
        C[3 * i], C[3 * i + 1], C[3 * i + 2] = p, px, py
    Ci = np.linalg.inv(C)
    g, wg = np.polynomial.legendre.leggauss(4)
    Dm = D * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    K = np.zeros((12, 12))
    for gi, wi in zip(g, wg):
        for gj, wj in zip(g, wg):
            x, y, w = a * (gi + 1) / 2, b * (gj + 1) / 2, wi * wj * a * b / 4
            p, px, py, pxx, pyy, pxy = _base(x, y)
            B = np.vstack([pxx, pyy, 2 * pxy]) @ Ci
            Nx, Ny = px @ Ci, py @ Ci
            K += (B.T @ Dm @ B + c2x * np.outer(Nx, Nx) + c2y * np.outer(Ny, Ny)) * w
    return K, Ci


def dans_polygone(X, Y, poly) -> np.ndarray:
    """Points dans un polygone (vectorisé, lancer de rayon — la règle de saf/fondation_grue/geometrie.py)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    c = np.zeros(X.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        m = (y1 > Y) != (y2 > Y)
        xi = x1 + (Y - y1) * (x2 - x1) / (y2 - y1)
        c ^= m & (X < xi)
    return c


# l'ordre des 12 degrés de liberté d'un élément, rangés par LIGNE de nœuds : la ligne du bas (nœuds 0 et 1), puis celle
# du haut dans le sens des x croissants (le nœud 3 avant le nœud 2) ; ou par COLONNE : celle de gauche (nœuds 0 et 3),
# puis celle de droite (nœuds 1 et 2)
_PAR_LIGNE = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
_PAR_COLONNE = [0, 1, 2, 9, 10, 11, 3, 4, 5, 6, 7, 8]


# ------------------------------------------------------------------ la dalle
class Dalle:
    """Une dalle maillée, prête à être chargée : ``charge`` rend un vecteur de forces, ``resoudre`` les résout."""

    def __init__(self, contour, h: float, k_MNm3: float, E_MPa: float = 32800.0, nu: float = 0.2, c2_MNm=0.0,
                 maille: float = 0.25, zones: Sequence = ()):
        """``zones`` : [(polygone, épaisseur totale), …] = les parties plus épaisses de la dalle (plots coulés en place).
        ``c2_MNm`` : C2 (Pasternak), une valeur ou le couple (C2x, C2y)."""
        c2x, c2y = c2_MNm if isinstance(c2_MNm, (tuple, list)) else (c2_MNm, c2_MNm)
        xs, ys = [p[0] for p in contour], [p[1] for p in contour]
        self.x0, self.y0 = min(xs), min(ys)
        Lx, Ly = max(xs) - self.x0, max(ys) - self.y0
        self.nx, self.ny = max(1, int(round(Lx / maille))), max(1, int(round(Ly / maille)))
        self.a, self.b = Lx / self.nx, Ly / self.ny
        self.k = k_MNm3 * 1000.0
        self.h, self.nu = h, nu
        self.D = E_MPa * 1000.0 * h ** 3 / (12 * (1 - nu * nu))
        Kb, self.Ci = element(self.a, self.b, 1.0, nu, 0.0)        # la flexion pour D = 1
        Kp, _ = element(self.a, self.b, 0.0, nu, c2x * 1000.0, c2y * 1000.0)  # Pasternak seul
        nnx = self.nnx = self.nx + 1
        I, J = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        I, J = I.ravel(), J.ravel()
        act = dans_polygone(self.x0 + (I + 0.5) * self.a, self.y0 + (J + 0.5) * self.b, contour)
        self.I, self.J = I[act], J[act]
        self.De = np.full(len(self.I), self.D)                     # la rigidité de flexion de chaque élément
        for poly, hz in zones:
            dz = dans_polygone(self.x0 + (self.I + 0.5) * self.a, self.y0 + (self.J + 0.5) * self.b, poly)
            self.De[dz] = E_MPa * 1000.0 * hz ** 3 / (12 * (1 - nu * nu))
        n = np.stack([self.J * nnx + self.I, self.J * nnx + self.I + 1, (self.J + 1) * nnx + self.I + 1, (self.J + 1) * nnx + self.I], 1)
        self.noeuds = n
        self.edofs = (3 * n[:, :, None] + np.arange(3)).reshape(-1, 12)
        nn = self.nn = nnx * (self.ny + 1)
        self.ndof = 3 * nn
        area = np.zeros(nn)
        np.add.at(area, n.ravel(), self.a * self.b / 4)
        self.area = area                                           # la surface tributaire de chaque nœud
        self.on0 = area > 0                                        # les nœuds de la dalle
        s = (np.arange(POINTS) + 0.5) / POINTS
        self.su, self.sv = [g.ravel() for g in np.meshgrid(s, s)]
        self.Ns = np.array([_base(u * self.a, v * self.b)[0] @ self.Ci for u, v in zip(self.su, self.sv)])
        # ---- la matrice, par blocs : A[j] sur la diagonale, B[j] entre les blocs j et j+1. Un bloc est une LIGNE de
        # nœuds, ou une COLONNE quand la grille a plus de colonnes que de lignes (le bloc le plus petit)
        colonnes = self.nx > self.ny
        gI, gJ = np.arange(nn) % nnx, np.arange(nn) // nnx         # la colonne et la ligne de chaque nœud
        bloc, rang = (gI, gJ) if colonnes else (gJ, gI)
        eb, er = (self.I, self.J) if colonnes else (self.J, self.I)   # le bloc et le rang de chaque élément
        par = _PAR_COLONNE if colonnes else _PAR_LIGNE
        nr = (self.ny if colonnes else self.nx) + 1                # nœuds par bloc
        m = self.m = 3 * nr
        nb = self.nb = (self.nx if colonnes else self.ny) + 1
        self._nr = nr
        self._noeud_ordre = np.argsort(bloc * nr + rang, kind="stable")      # rang dans les blocs → nœud
        self._ordre = (3 * self._noeud_ordre[:, None] + np.arange(3)).ravel()  # … → degré de liberté
        Kb_l, Kp_l = Kb[np.ix_(par, par)], Kp[np.ix_(par, par)]
        A = np.zeros((nb, m, m))
        B = np.zeros((max(nb - 1, 1), m, m))
        for i in range(nr - 1):
            ici = er == i
            if not ici.any():
                continue
            js = eb[ici]
            Ke = self.De[ici][:, None, None] * Kb_l[None] + Kp_l[None]
            s6 = slice(3 * i, 3 * i + 6)
            A[js, s6, s6] += Ke[:, :6, :6]
            A[js + 1, s6, s6] += Ke[:, 6:, 6:]
            B[js, s6, s6] += Ke[:, :6, 6:]
        # un nœud hors de la dalle (contour non rectangulaire) garde ses trois degrés de liberté, bloqués à zéro
        hors = np.flatnonzero(~self.on0)
        for d in range(3):
            A[bloc[hors], 3 * rang[hors] + d, 3 * rang[hors] + d] = 1.0
        # B[j] ne lie un degré de liberté qu'à ceux des nœuds voisins : onze diagonales (décalages −5 à +5)
        Bd = np.zeros((max(nb - 1, 1), 2 * _DEMI_BANDE + 1, m))
        for o in range(-_DEMI_BANDE, _DEMI_BANDE + 1):
            r0 = max(0, -o)
            Bd[:, o + _DEMI_BANDE, r0:r0 + m - abs(o)] = np.diagonal(B, offset=o, axis1=1, axis2=2)
        assert np.isclose(np.abs(B).sum(), np.abs(Bd).sum()), "B a des termes hors de ses onze diagonales"
        self._A, self._Bd = A, Bd
        self._fact = None                                          # la factorisation « tous les ressorts actifs »
        self._colonnes: Dict[int, int] = {}                        # nœud libéré → sa colonne de Z = K₀⁻¹E
        self._Z = np.zeros((self.ndof, 0))

    # ---------------------------------------------------------------- les charges
    def charge(self, q_uniforme: float = 0.0, plaques: Sequence = ()) -> np.ndarray:
        """Le vecteur des forces nodales : q uniforme sur la dalle + [(polygone, q en kN/m²), …]."""
        F = np.zeros(self.ndof)
        if q_uniforme:
            Fe = np.full(len(self.I), float(q_uniforme))[:, None] * self.Ns.sum(axis=0)[None, :] * self.a * self.b / len(self.su)
            np.add.at(F, self.edofs.ravel(), Fe.ravel())
        for poly, qp in plaques:
            # seuls les éléments que la boîte du polygone touche peuvent porter de sa charge
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            pres = np.flatnonzero((self.x0 + (self.I + 1) * self.a >= min(xs)) & (self.x0 + self.I * self.a <= max(xs))
                                  & (self.y0 + (self.J + 1) * self.b >= min(ys)) & (self.y0 + self.J * self.b <= max(ys)))
            if not len(pres):
                continue
            X = self.x0 + (self.I[pres][:, None] + self.su[None, :]) * self.a
            Y = self.y0 + (self.J[pres][:, None] + self.sv[None, :]) * self.b
            q = qp * dans_polygone(X, Y, poly)
            Fe = (q @ self.Ns) * self.a * self.b / len(self.su)
            np.add.at(F, self.edofs[pres].ravel(), Fe.ravel())
        return F

    def charge_unitaire(self, poly) -> np.ndarray:
        """La charge d'un polygone NORMALISÉE : la somme des forces verticales vaut exactement 1 kN, même si le polygone
        n'est pas aligné sur le maillage. ValueError si le polygone ne touche pas la dalle."""
        f = self.charge(plaques=[(poly, 1.0)])
        total = f[0::3].sum()
        if not total > 0:
            raise ValueError("une surface d'appui est hors de la dalle")
        return f / total

    # ---------------------------------------------------------------- la résolution
    def _B_fois(self, j: int, X: np.ndarray) -> np.ndarray:
        """B[j] @ X, par les onze diagonales de B[j]."""
        m, Bd = self.m, self._Bd[j]
        out = np.zeros_like(X)
        for o in range(-_DEMI_BANDE, _DEMI_BANDE + 1):
            r0, r1 = max(0, -o), m - max(0, o)
            out[r0:r1] += Bd[o + _DEMI_BANDE, r0:r1, None] * X[r0 + o:r1 + o]
        return out

    def _Bt_fois(self, j: int, X: np.ndarray) -> np.ndarray:
        """B[j]ᵀ @ X, par les onze diagonales de B[j]."""
        m, Bd = self.m, self._Bd[j]
        out = np.zeros_like(X)
        for o in range(-_DEMI_BANDE, _DEMI_BANDE + 1):
            r0, r1 = max(0, -o), m - max(0, o)
            out[r0 + o:r1 + o] += Bd[o + _DEMI_BANDE, r0:r1, None] * X[r0:r1]
        return out

    def _factoriser(self, ressorts: np.ndarray) -> np.ndarray:
        """La factorisation par blocs pour ces ressorts (kN/m par nœud) : les S_j⁻¹."""
        nb, m, nr = self.nb, self.m, self._nr
        Si = np.empty((nb, m, m))
        diag = np.arange(0, m, 3)
        ressorts = ressorts[self._noeud_ordre]                     # dans l'ordre des blocs
        for j in range(nb):
            S = self._A[j].copy()
            if j:                                                  # − B_{j−1}ᵀ·S_{j−1}⁻¹·B_{j−1} (S⁻¹ est symétrique)
                S -= self._Bt_fois(j - 1, self._Bt_fois(j - 1, Si[j - 1]).T)
            S[diag, diag] += ressorts[j * nr:(j + 1) * nr]
            Si[j] = np.linalg.inv(S)
        return Si

    def _substituer(self, Si: np.ndarray, F: np.ndarray) -> np.ndarray:
        """K⁻¹·F pour une factorisation ; ``F`` : (ndof,) ou (ndof, n seconds membres)."""
        nb, m = self.nb, self.m
        seul = F.ndim == 1
        Y = (F[:, None] if seul else F)[self._ordre].reshape(nb, m, -1)        # dans l'ordre des blocs
        T = np.empty(Y.shape)
        T[0] = Si[0] @ Y[0]
        for j in range(1, nb):
            T[j] = Si[j] @ (Y[j] - self._Bt_fois(j - 1, T[j - 1]))
        for j in range(nb - 2, -1, -1):
            T[j] -= Si[j] @ self._B_fois(j, T[j + 1])
        X = np.empty((self.ndof, T.shape[2]))
        X[self._ordre] = T.reshape(self.ndof, -1)
        return X[:, 0] if seul else X

    def _lineaire(self):
        if self._fact is None:
            self._fact = self._factoriser(np.where(self.on0, self.k * self.area, 0.0))
        return self._fact

    def _colonnes_de(self, noeuds: np.ndarray) -> np.ndarray:
        """Les colonnes de Z = K₀⁻¹E pour ces nœuds (calculées une fois, gardées)."""
        manque = [int(n) for n in noeuds if int(n) not in self._colonnes]
        if manque:
            E = np.zeros((self.ndof, len(manque)))
            E[3 * np.array(manque), np.arange(len(manque))] = 1.0
            Z = self._substituer(self._lineaire(), E)
            base = self._Z.shape[1]
            for rang, n in enumerate(manque):
                self._colonnes[n] = base + rang
            self._Z = np.hstack([self._Z, Z])
        return np.array([self._colonnes[int(n)] for n in noeuds], dtype=int)

    def resoudre(self, Fs: Sequence[np.ndarray], sans_traction: bool = True, maxit: int = ITERATIONS_MAX) -> List[Dict]:
        """Résout plusieurs cas. Rend pour chacun : sigma (kPa, aux nœuds), sigma_lin (le calcul linéaire, traction
        possible ; NaN hors de la dalle), w (m ; NaN hors de la dalle), decolle (la part de surface sans contact),
        iterations."""
        on0 = self.on0
        n_dalle = int(on0.sum())
        Y = self._substituer(self._lineaire(), np.array(Fs, dtype=float).T.reshape(self.ndof, -1))
        res = []
        for c in range(Y.shape[1]):
            y = Y[:, c]
            w_lin = y[0::3].copy()
            on, x, it = on0.copy(), y, 0
            for it in range(maxit):
                lib = np.flatnonzero(on0 & ~on)
                if not len(lib):
                    x = y
                elif len(lib) > SEUIL_WOODBURY * n_dalle:
                    x = self._substituer(self._factoriser(np.where(on, self.k * self.area, 0.0)), np.asarray(Fs[c], dtype=float))
                else:
                    cols = self._colonnes_de(lib)
                    Z = self._Z[:, cols]
                    C = np.diag(1.0 / (self.k * self.area[lib])) - Z[3 * lib, :]
                    x = y + Z @ np.linalg.solve(C, y[3 * lib])
                w = x[0::3]
                if not sans_traction:
                    break
                nouveau = on0 & (w > 0)
                if np.array_equal(nouveau, on):
                    break
                on = nouveau
            w = x[0::3]
            res.append({"sigma": np.where(on, self.k * w, 0.0), "sigma_lin": np.where(on0, self.k * w_lin, np.nan),
                        "w": np.where(on0, w, np.nan), "decolle": float(self.area[on0 & ~on].sum() / self.area[on0].sum()),
                        "iterations": it + 1})
        return res

    # ---------------------------------------------------------------- ce que l'écran dessine
    def grille(self, valeurs: np.ndarray) -> List[List[Optional[float]]]:
        """Des valeurs aux nœuds rangées en lignes (du bas vers le haut) : None hors de la dalle."""
        g = np.where(self.on0, valeurs, np.nan).reshape(self.ny + 1, self.nnx)
        return [[None if np.isnan(v) else float(v) for v in ligne] for ligne in g]

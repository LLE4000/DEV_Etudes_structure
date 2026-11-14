"""Fondation de grue - calcul de reference du module (verification rapide avant SCIA).

Dalle en beton arme deformable : plaque mince (Kirchhoff), element rectangle ACM 12 ddl
(w, dw/dx, dw/dy), grille reguliere, contour quelconque. Sol de Winkler (ressorts k x surface
tributaire aux noeuds) + Pasternak C2 en option. Sol en COMPRESSION SEULE par defaut
(le sol ne retient pas la dalle : les zones qui se soulevent sont liberees, iterations).
Unites : kN, m, kPa, MN/m3. w positif vers le bas ; contrainte de contact sigma = k.w.
Moments aux centres des elements : m > 0 = fibre inferieure tendue.

Charge de chaque plaque normalisee : la somme des forces verticales vaut exactement R, meme si la
plaque n'est pas alignee sur le maillage (sinon jusqu'a quelques % d'erreur).

Plots sous les pieds (option) : prefabriques (poids du plot + reaction repartie sur le plot, rigidite
de la dalle inchangee) ou coules en place (poids du plot + dalle plus epaisse sous le plot).

Usage : python fondation_grue_ref.py   -> tests T1 a T13.
Dependances : numpy, scipy.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

KG_CM2 = 98.0665  # 1 kg/cm2 en kPa


# ---------------------------------------------------------------- element ACM
def _basis(x, y):
    p = np.array([1, x, y, x*x, x*y, y*y, x**3, x*x*y, x*y*y, y**3, x**3*y, x*y**3])
    px = np.array([0, 1, 0, 2*x, y, 0, 3*x*x, 2*x*y, y*y, 0, 3*x*x*y, y**3])
    py = np.array([0, 0, 1, 0, x, 2*y, 0, x*x, 2*x*y, 3*y*y, x**3, 3*x*y*y])
    pxx = np.array([0, 0, 0, 2, 0, 0, 6*x, 2*y, 0, 0, 6*x*y, 0])
    pyy = np.array([0, 0, 0, 0, 0, 2, 0, 0, 2*x, 6*y, 0, 6*x*y])
    pxy = np.array([0, 0, 0, 0, 1, 0, 0, 2*x, 2*y, 0, 3*x*x, 3*y*y])
    return p, px, py, pxx, pyy, pxy


def element(a, b, D, nu, c2):
    """Rigidite de flexion (+ Pasternak) d'un rectangle a x b, et matrice C^-1 des coefficients."""
    C = np.zeros((12, 12))
    for i, (xn, yn) in enumerate([(0, 0), (a, 0), (a, b), (0, b)]):
        p, px, py, *_ = _basis(xn, yn)
        C[3*i], C[3*i+1], C[3*i+2] = p, px, py
    Ci = np.linalg.inv(C)
    g, wg = np.polynomial.legendre.leggauss(4)
    Dm = D*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    K = np.zeros((12, 12))
    for gi, wi in zip(g, wg):
        for gj, wj in zip(g, wg):
            x, y, w = a*(gi+1)/2, b*(gj+1)/2, wi*wj*a*b/4
            p, px, py, pxx, pyy, pxy = _basis(x, y)
            B = np.vstack([pxx, pyy, 2*pxy]) @ Ci
            Nx, Ny = px @ Ci, py @ Ci
            K += (B.T @ Dm @ B + c2*(np.outer(Nx, Nx) + np.outer(Ny, Ny)))*w
    return K, Ci


# ---------------------------------------------------------------- geometrie
def in_poly(X, Y, poly):
    """Point dans polygone (vectorise, lancer de rayon)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    c = np.zeros(X.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if y1 == y2:
            continue
        m = (y1 > Y) != (y2 > Y)
        xi = x1 + (Y-y1)*(x2-x1)/(y2-y1)
        c ^= m & (X < xi)
    return c


def _rot(cx, cy, pts, beta_deg):
    t = np.radians(beta_deg)
    return [(cx + x*np.cos(t) - y*np.sin(t), cy + x*np.sin(t) + y*np.cos(t)) for x, y in pts]


def legs(cx, cy, a, b, beta_deg=0.0):
    """Centres des pieds P1 (haut-gauche), P2 (haut-droite), P3 (bas-droite), P4 (bas-gauche),
    sens horaire vu du dessus, chassis non tourne. a = entraxe en x, b = entraxe en y."""
    return _rot(cx, cy, [(-a/2, b/2), (a/2, b/2), (a/2, -b/2), (-a/2, -b/2)], beta_deg)


def plate_poly(x, y, ap, bp, beta_deg=0.0):
    """Coins d'une plaque : haut-gauche puis sens horaire (avant rotation), comme NDC_TYPE.xlsx."""
    return _rot(x, y, [(-ap/2, bp/2), (ap/2, bp/2), (ap/2, -bp/2), (-ap/2, -bp/2)], beta_deg)


def aire(poly):
    """Aire d'un polygone (formule du lacet)."""
    n = len(poly)
    return 0.5*abs(sum(poly[i][0]*poly[(i+1) % n][1] - poly[(i+1) % n][0]*poly[i][1] for i in range(n)))


def _croise(p1, p2, q1, q2):
    """Vrai si les segments p1-p2 et q1-q2 se coupent franchement."""
    def o(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    return o(q1, q2, p1)*o(q1, q2, p2) < 0 and o(p1, p2, q1)*o(p1, p2, q2) < 0


def dans_dalle(contour, poly, tol=1e-3):
    """Polygone (plaque ou plot) entierement dans la dalle. Bord sur bord admis : le polygone est
    d'abord retreci de tol (1 mm) vers son centre. Puis : tous ses sommets dans le contour et aucun
    de ses cotes ne coupe un cote du contour (coin rentrant)."""
    P = np.asarray(poly, float)
    c = P.mean(axis=0)
    Q = [tuple(c + (q - c)*(1 - tol/max(np.linalg.norm(q - c), tol))) for q in P]
    if not in_poly([q[0] for q in Q], [q[1] for q in Q], contour).all():
        return False
    n, m = len(contour), len(Q)
    return not any(_croise(Q[i], Q[(i+1) % m], contour[j], contour[(j+1) % n])
                   for i in range(m) for j in range(n))


def _d_point_segment(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    t = np.clip(np.dot(p-a, b-a)/np.dot(b-a, b-a), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t*(b-a))))


def distance_plaques_bord(contour, plaques):
    """Distance mini entre plaques et contour (coins de plaque -> bords du contour et sommets du
    contour -> bords des plaques), et plaques entierement dans la dalle."""
    d = np.inf
    n = len(contour)
    for pl in plaques:
        for q in pl:
            for i in range(n):
                d = min(d, _d_point_segment(q, contour[i], contour[(i+1) % n]))
        for v in contour:
            for i in range(len(pl)):
                d = min(d, _d_point_segment(v, pl[i], pl[(i+1) % len(pl)]))
    dedans = all(dans_dalle(contour, pl) for pl in plaques)
    return d, dedans


def _clip_ligne(contour, p0, d, t_plaques):
    """Segment de la droite p0 + t.d a l'interieur du contour, qui contient les plaques."""
    p0, d = np.asarray(p0, float), np.asarray(d, float)
    ts = []
    n = len(contour)
    for i in range(n):
        a, b = np.asarray(contour[i], float), np.asarray(contour[(i+1) % n], float)
        e = b - a
        den = d[0]*(-e[1]) - d[1]*(-e[0])
        if abs(den) < 1e-12:
            continue
        r = a - p0
        t = (r[0]*(-e[1]) - r[1]*(-e[0]))/den
        s = (d[0]*r[1] - d[1]*r[0])/den
        if -1e-9 <= s <= 1+1e-9:
            ts.append(t)
    ts = sorted(set(round(t, 9) for t in ts))
    tm = float(np.mean(t_plaques))
    for t1, t2 in zip(ts[:-1], ts[1:]):
        mid = p0 + d*(t1+t2)/2
        if t1 <= tm <= t2 and in_poly([mid[0]], [mid[1]], contour)[0]:
            return tuple(p0 + d*t1), tuple(p0 + d*t2)
    return None


def bandes_integration(contour, cx, cy, a, b, beta_deg=0.0, largeur=1.0, centres=None):
    """4 bandes d'integration passant par les plaques, le long des axes du chassis, limitees a la
    dalle : BX1 (P1-P2), BX2 (P4-P3), BY1 (P1-P4), BY2 (P2-P3). Largeur totale symetrique.
    centres : centres des 4 surfaces d'appui (P1..P4) s'ils different des axes d'appui
    (excentrement radial de la bibliotheque des chassis)."""
    P = centres if centres is not None else legs(cx, cy, a, b, beta_deg)
    t = np.radians(beta_deg)
    ux, uy = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    out = []
    for nom, (i, j), u in (("BX1", (0, 1), ux), ("BX2", (3, 2), ux), ("BY1", (0, 3), uy), ("BY2", (1, 2), uy)):
        p0 = np.asarray(P[i])
        tj = float(np.dot(np.asarray(P[j]) - p0, u))
        seg = _clip_ligne(contour, p0, u, (0.0, tj))
        if seg:
            out.append({"nom": nom, "debut": seg[0], "fin": seg[1], "largeur": largeur})
    return out


# ---------------------------------------------------------------- positions de charge
def positions(R, chassis_carre=True):
    """Toutes les positions physiques des 4 reactions (rotations + miroirs), doublons retires.
    Pied i recoit R[(i+k)%4] (rotation de k x 90 deg) ou R[(k-i)%4] (miroir).
    Chassis rectangulaire : rotations 0/180 deg et miroirs selon les 2 axes seulement."""
    ks_rot = (0, 1, 2, 3) if chassis_carre else (0, 2)
    ks_mir = (0, 1, 2, 3) if chassis_carre else (1, 3)
    out, seen = [], set()
    cand = [tuple(R[(i+k) % 4] for i in range(4)) for k in ks_rot]
    cand += [tuple(R[(k-i) % 4] for i in range(4)) for k in ks_mir]
    for r in cand:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ---------------------------------------------------------------- modele
class Dalle:
    def __init__(self, contour, h, k_MNm3, E_MPa=32800.0, nu=0.2, c2_MNm=0.0, maille=0.25, zones=()):
        """zones : [(polygone, epaisseur totale), ...] = parties plus epaisses de la dalle (plots
        coules en place). Un element prend l'epaisseur de la zone si son centre est dedans."""
        xs, ys = [p[0] for p in contour], [p[1] for p in contour]
        self.x0, self.y0 = min(xs), min(ys)
        Lx, Ly = max(xs)-self.x0, max(ys)-self.y0
        self.nx, self.ny = max(1, int(round(Lx/maille))), max(1, int(round(Ly/maille)))
        self.a, self.b = Lx/self.nx, Ly/self.ny
        self.k = k_MNm3*1000.0
        self.h, self.nu = h, nu
        self.D = E_MPa*1000.0*h**3/(12*(1-nu*nu))
        Kb, self.Ci = element(self.a, self.b, 1.0, nu, 0.0)        # flexion pour D = 1
        Kp, _ = element(self.a, self.b, 0.0, nu, c2_MNm*1000.0)    # Pasternak seul
        nnx = self.nx+1
        I, J = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        I, J = I.ravel(), J.ravel()
        act = in_poly(self.x0+(I+0.5)*self.a, self.y0+(J+0.5)*self.b, contour)
        self.I, self.J = I[act], J[act]
        self.De = np.full(len(self.I), self.D)                     # rigidite de flexion par element
        for poly, hz in zones:
            dz = in_poly(self.x0+(self.I+0.5)*self.a, self.y0+(self.J+0.5)*self.b, poly)
            self.De[dz] = E_MPa*1000.0*hz**3/(12*(1-nu*nu))
        n = np.stack([self.J*nnx+self.I, self.J*nnx+self.I+1, (self.J+1)*nnx+self.I+1, (self.J+1)*nnx+self.I], 1)
        self.edofs = (3*n[:, :, None] + np.arange(3)).reshape(-1, 12)
        nn = nnx*(self.ny+1)
        self.ndof = 3*nn
        r = np.repeat(self.edofs, 12, 1).ravel()
        c = np.tile(self.edofs, (1, 12)).ravel()
        Ke = self.De[:, None]*Kb.ravel()[None, :] + Kp.ravel()[None, :]
        self.K0 = sp.csc_matrix((Ke.ravel(), (r, c)), shape=(self.ndof, self.ndof))
        area = np.zeros(nn)
        np.add.at(area, n.ravel(), self.a*self.b/4)
        self.area = area
        self.used = np.unique(self.edofs)
        s = (np.arange(8)+0.5)/8
        self.su, self.sv = [g.ravel() for g in np.meshgrid(s, s)]
        self.Ns = np.array([_basis(u*self.a, v*self.b)[0] @ self.Ci for u, v in zip(self.su, self.sv)])
        _, _, _, pxx, pyy, pxy = _basis(self.a/2, self.b/2)
        self.Bc = np.vstack([pxx, pyy, pxy]) @ self.Ci   # w_xx, w_yy, w_xy au centre

    def charge(self, q_uniforme=0.0, plaques=()):
        """Vecteur de charge : q uniforme sur la dalle + [(polygone, q kN/m2), ...]."""
        X = self.x0 + (self.I[:, None] + self.su[None, :])*self.a
        Y = self.y0 + (self.J[:, None] + self.sv[None, :])*self.b
        q = np.full(X.shape, float(q_uniforme))
        for poly, qp in plaques:
            q += qp*in_poly(X, Y, poly)
        Fe = (q @ self.Ns)*self.a*self.b/len(self.su)
        F = np.zeros(self.ndof)
        np.add.at(F, self.edofs.ravel(), Fe.ravel())
        return F

    def moments(self, x):
        """mx, my aux centres des elements (kNm/m), m > 0 = fibre inferieure tendue."""
        c = x[self.edofs] @ self.Bc.T
        mx = -self.De*(c[:, 0] + self.nu*c[:, 1])
        my = -self.De*(c[:, 1] + self.nu*c[:, 0])
        return mx, my

    def resoudre(self, Fs, sans_traction=True, maxit=30):
        """Resout plusieurs cas. Une seule factorisation tant que tous les ressorts sont actifs.
        Renvoie pour chaque cas un dict : sigma (noeuds), sigma_lin (calcul lineaire, traction
        possible), w (m), decolle (fraction de surface sans contact), mx, my, iterations."""
        u = self.used
        res = []
        on0 = self.area > 0
        lu_lin = None
        for F in Fs:
            on = on0.copy()
            for it in range(maxit):
                if lu_lin is not None and np.array_equal(on, on0):
                    lu = lu_lin
                else:
                    ks = np.where(on, self.k*self.area, 0.0)
                    Ks = sp.csc_matrix((ks, (3*np.arange(len(ks)), 3*np.arange(len(ks)))), shape=(self.ndof, self.ndof))
                    lu = spla.splu((self.K0 + Ks)[u][:, u].tocsc())
                    if np.array_equal(on, on0):
                        lu_lin = lu
                x = np.zeros(self.ndof)
                x[u] = lu.solve(F[u])
                w = x[0::3]
                if it == 0:
                    w_lin = w.copy()
                if not sans_traction:
                    break
                new = on0 & (w > 0)
                if np.array_equal(new, on):
                    break
                on = new
            mx, my = self.moments(x)
            res.append({"sigma": np.where(on, self.k*w, 0.0),
                        "sigma_lin": np.where(on0, self.k*w_lin, np.nan),
                        "w": np.where(on0, w, np.nan),
                        "decolle": float(self.area[on0 & ~on].sum()/self.area[on0].sum()),
                        "mx": mx, "my": my, "iterations": it+1})
        return res


def verifier(contour, h, k, chassis, plaque, R_jeux, centre=None, beta=0.0, maille=0.25,
             E=32800.0, c2=0.0, gamma_beton=25.0, sans_traction=True, plaques=None, plots=None):
    """Toutes les positions de tous les jeux (ELS caracteristique : PP + reactions).
    plaques : 4 polygones (P1..P4, coordonnees globales, n sommets chacun) pour des surfaces
    d'appui quelconques (bibliotheque des chassis) ; sinon rectangles ap x bp centres sur les pieds.
    plots : None, ou (type, a, b, hauteur) = un plot a x b sous chaque pied, centre sur la surface
    d'appui, cotes paralleles aux axes du chassis.
      "prefabrique" : poids du plot + reaction repartie sur tout le plot ; dalle inchangee.
      "coule"       : poids du plot + dalle d'epaisseur h + hauteur sous le plot ; reaction sur la
                      surface d'appui.
    Renvoie une liste de dicts par cas : jeu, R, sigma_max, sigma_min_lin (kPa), w_max (mm),
    decolle (%), m_pos, m_neg (kNm/m), charge_totale, reaction_sol (kN), iterations."""
    if centre is None:
        xs, ys = [p[0] for p in contour], [p[1] for p in contour]
        centre = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2)
    a, b = chassis
    ap, bp = plaque
    if plaques is not None:
        pl = plaques
    else:
        pl = [plate_poly(x, y, ap, bp, beta) for x, y in legs(centre[0], centre[1], a, b, beta)]
    pp, zones = [], ()
    if plots is not None:
        typ, a_pl, b_pl, h_pl = plots
        pp = [plate_poly(*np.mean(np.asarray(p, float), axis=0), a_pl, b_pl, beta) for p in pl]
        if typ == "coule":
            zones = [(q, h + h_pl) for q in pp]
        elif typ == "prefabrique":
            pl = pp                                   # la reaction est repartie sur le plot
        else:
            raise ValueError("plots : type 'prefabrique' ou 'coule'")
    d = Dalle(contour, h, k, E_MPa=E, c2_MNm=c2, maille=maille, zones=zones)
    Fg = d.charge(q_uniforme=gamma_beton*h)
    for q in pp:                                      # poids de chaque plot, somme exacte
        f = d.charge(plaques=[(q, 1.0)])
        Fg = Fg + f*(gamma_beton*h_pl*aire(q)/f[0::3].sum())
    Fp = []
    for p in pl:
        f = d.charge(plaques=[(p, 1.0)])
        Fp.append(f/f[0::3].sum())   # charge unitaire EXACTE : somme des forces verticales = 1 kN
    cas = [(j, r) for j, R in enumerate(R_jeux, 1) for r in positions(R, chassis_carre=(abs(a-b) < 1e-9))]
    Fs = [Fg + sum(r[i]*Fp[i] for i in range(4)) for _, r in cas]
    sol = d.resoudre(Fs, sans_traction)
    out = []
    for (j, r), s, F in zip(cas, sol, Fs):
        out.append({"jeu": j, "R": r, "sigma_max": float(s["sigma"].max()),
                    "sigma_min_lin": float(np.nanmin(s["sigma_lin"])),
                    "w_max": float(np.nanmax(s["w"])*1000), "decolle": 100*s["decolle"],
                    "m_pos": float(max(s["mx"].max(), s["my"].max())),
                    "m_neg": float(min(s["mx"].min(), s["my"].min())),
                    "charge_totale": float(F[0::3].sum()),
                    "reaction_sol": float((s["sigma"]*d.area).sum()),
                    "iterations": s["iterations"]})
    return out


def sensibilite_k(contour, h, k_liste, chassis, plaque, R_jeux, **kw):
    """Valeurs dimensionnantes (toutes positions, tous jeux) pour chaque k."""
    lignes = []
    for k in k_liste:
        r = verifier(contour, h, k, chassis, plaque, R_jeux, **kw)
        lignes.append({"k": k,
                       "sigma_max": max(x["sigma_max"] for x in r),
                       "w_max": max(x["w_max"] for x in r),
                       "decolle": max(x["decolle"] for x in r),
                       "m_pos": max(x["m_pos"] for x in r),
                       "m_neg": min(x["m_neg"] for x in r)})
    return lignes


def proposer_rectangle(Lx, Ly, h, k, chassis, plaque, R_jeux, ecart=(0.0, 0.0), beta=0.0,
                       lim=0.5*KG_CM2, pas=0.05, dmax=6.0, maille=0.25, **kw):
    """Plus petit agrandissement d (multiple de pas, ajoute de chaque cote) tel que sigma max <= lim.
    Le chassis garde sa position par rapport au centre. Maille 0,25 m (une maille de 0,50 m peut
    sous-estimer de 4 % quand la plaque fait 0,50 m). Avec plusieurs k : prendre le k le plus raide."""
    def smax(d):
        L1, L2 = Lx + 2*d, Ly + 2*d
        r = verifier([(0, 0), (L1, 0), (L1, L2), (0, L2)], h, k, chassis, plaque, R_jeux,
                     centre=(L1/2 + ecart[0], L2/2 + ecart[1]), beta=beta, maille=maille, **kw)
        return max(x["sigma_max"] for x in r)
    lo, hi = 0, int(round(dmax/pas))
    if smax(0) <= lim:
        return 0.0, smax(0)
    if smax(hi*pas) > lim:
        return None, smax(hi*pas)
    while hi - lo > 1:
        mid = (lo + hi)//2
        if smax(mid*pas) <= lim:
            hi = mid
        else:
            lo = mid
    return hi*pas, smax(hi*pas)


# ---------------------------------------------------------------- tests
def _rect(Lx, Ly):
    return [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]


if __name__ == "__main__":
    import time
    E, nu, k, h = 32800.0, 0.2, 12.0, 0.5
    D = E*1000*h**3/(12*(1-nu*nu))
    jeux = [(350, 440, 440, 350), (150, 150, 550, 550), (135, 100, 800, 135)]

    # T1 - charge uniforme : sigma = q partout
    d = Dalle(_rect(8, 8), h, k)
    s = d.resoudre([d.charge(q_uniforme=12.5)])[0]["sigma"]
    print(f"T1  uniforme 12,5 kPa : sigma = {s.min():.3f} a {s.max():.3f} kPa (attendu 12,5 +/- 0,2 %)")

    # T2 - charge ponctuelle sur grande dalle : Hertz w0 = P/(8 sqrt(kD)), sigma = P/(8 l^2)
    d = Dalle(_rect(28, 28), h, k)
    F = np.zeros(d.ndof)
    c = (d.ny//2)*(d.nx+1) + d.nx//2
    F[3*c] = 1000.0
    s = d.resoudre([F], sans_traction=False)[0]["sigma"]   # theorie elastique lineaire
    hertz = 1000.0/(8*np.sqrt(k*1000*D))*k*1000
    print(f"T2  Hertz P = 1000 kN : sigma max = {s[c]:.2f} kPa ; theorie {hertz:.2f} kPa ({100*(s[c]/hertz-1):+.2f} %)")

    # T3 - limite rigide (E x 1000), calcul LINEAIRE (traction admise) : 77,4 / -15,9 kPa a la main
    r = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), [jeux[2]], E=E*1000, sans_traction=False)
    print(f"T3  rigide lineaire jeu 3 : sigma max = {r[0]['sigma_max']:.1f} ; min = {r[0]['sigma_min_lin']:.1f} kPa (attendu 77,4 / -15,9)")

    # T4 - dalle type NDC_TYPE : 8 x 8 x 0,50, k = 12, chassis 6 x 6 centre, plaques 0,5 x 0,5
    t = time.time()
    r = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), jeux)
    dt = time.time() - t
    print(f"T4  dalle type ({len(r)} cas, {dt:.2f} s) :")
    for j in (1, 2, 3):
        rr = [x for x in r if x["jeu"] == j]
        m = max(x["sigma_max"] for x in rr)
        print(f"    jeu {j} ({len(rr)} positions) : sigma max = {m:.1f} kPa = {m/KG_CM2:.2f} kg/cm2 ; "
              f"w max = {max(x['w_max'] for x in rr):.2f} mm ; decolle = {max(x['decolle'] for x in rr):.1f} %")
    pl_type = [plate_poly(x, y, 0.5, 0.5) for x, y in legs(4, 4, 6, 6)]
    print(f"    distance mini plaque-bord = {distance_plaques_bord(_rect(8, 8), pl_type)[0]:.3f} m")
    equilibre = list(r)

    # T4c - idem T4 jeu 3 avec C2x = C2y = k/10 = 1,2 MN/m (valeurs exportees dans le SAF)
    r = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), [jeux[2]], c2=k/10)
    print(f"T4c dalle type jeu 3 avec C2 = k/10 : sigma max = {max(x['sigma_max'] for x in r):.1f} kPa")

    # T4b - grue decalee de +0,25 m en x et en y, jeu 3
    r = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), [jeux[2]], centre=(4.25, 4.25))
    print(f"T4b grue decalee (+0,25 ; +0,25), jeu 3 : sigma max = {max(x['sigma_max'] for x in r):.1f} kPa sur {len(r)} positions")

    # T5 - poids propre seul (reactions nulles) : sigma = 25 x h partout
    r = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), [(0, 0, 0, 0)])
    print(f"T5  poids propre seul, dalle type : {len(r)} cas ; sigma = {r[0]['sigma_min_lin']:.2f} a "
          f"{r[0]['sigma_max']:.2f} kPa (25 x h = {25*h:.1f} kPa)")

    # T7 - dalle en L, chassis 4,5 x 4,5 tourne de 30 deg, plaques 0,6 x 0,6
    Lc = [(0, 0), (12, 0), (12, 7), (7, 7), (7, 12), (0, 12)]
    r = verifier(Lc, h, k, (4.5, 4.5), (0.6, 0.6), jeux, centre=(4.2, 4.2), beta=30.0)
    pl = [plate_poly(x, y, 0.6, 0.6, 30.0) for x, y in legs(4.2, 4.2, 4.5, 4.5, 30.0)]
    dmin, dedans = distance_plaques_bord(Lc, pl)
    print(f"T7  dalle en L, chassis tourne 30 deg : {len(r)} positions ; sigma max = {max(x['sigma_max'] for x in r):.1f} kPa ; "
          f"plaques dedans = {dedans} ; distance mini = {dmin:.3f} m")
    equilibre += r

    # T8 - limite rigide en compression seule
    r = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), [jeux[2]], E=E*1000)
    print(f"T8  rigide compression seule jeu 3 : sigma max = {r[0]['sigma_max']:.1f} kPa ; decolle = {r[0]['decolle']:.1f} % ; "
          f"{r[0]['iterations']} iterations (calcul rigide sans traction independant : 79,2 kPa, 7 %)")
    equilibre += r

    # T12 - surfaces d'appui de forme quelconque (comme la bibliotheque des chassis), dalle type, 3 jeux
    def platine_45(xc, yc, sx, sy, a_trav, b_long):
        u = np.array([sx, sy])/np.sqrt(2); v = np.array([-sy, sx])/np.sqrt(2)
        c = np.array([xc, yc])
        return [tuple(c + i*b_long/2*u + j*a_trav/2*v) for i, j in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    signes = [(-1, 1), (1, 1), (1, -1), (-1, -1)]          # P1 haut-gauche ... P4 bas-gauche
    centres = legs(4, 4, 6, 6)
    pl45 = [platine_45(x, y, sx, sy, 0.58, 0.40) for (x, y), (sx, sy) in zip(centres, signes)]
    r45 = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), jeux, plaques=pl45)
    ang = np.linspace(0, 2*np.pi, 49)[:-1]
    plc = [[(x + 0.25*np.cos(t), y + 0.25*np.sin(t)) for t in ang] for x, y in centres]
    rc = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), jeux, plaques=plc)
    aire48 = aire(plc[0])
    print(f"T12 platines 0,58 x 0,40 a 45 deg : sigma max = {max(x['sigma_max'] for x in r45):.1f} kPa ; "
          f"distance mini = {distance_plaques_bord(_rect(8, 8), pl45)[0]:.3f} m")
    print(f"T12 platines rondes D 0,50 (48 cotes, aire {aire48:.4f} m2) : sigma max = {max(x['sigma_max'] for x in rc):.1f} kPa ; "
          f"distance mini = {distance_plaques_bord(_rect(8, 8), plc)[0]:.3f} m")
    equilibre += r45 + rc
    e = 0.10                                                # excentrement radial (vers l'exterieur)
    cen_e = [(x + sx*e/np.sqrt(2), y + sy*e/np.sqrt(2)) for (x, y), (sx, sy) in zip(centres, signes)]
    bs = bandes_integration(_rect(8, 8), 4, 4, 6, 6, 0.0, 1.0, centres=cen_e)
    f3 = lambda v: f"{round(v, 6) + 0.0:.3f}"
    print("T12 bandes, excentrement radial 0,10 m : " + " ; ".join(
        f"{x['nom']} ({f3(x['debut'][0])}, {f3(x['debut'][1])}) -> ({f3(x['fin'][0])}, {f3(x['fin'][1])})" for x in bs))

    # T13 - plots 2,00 x 2,00 x 0,35 m sous les pieds, dalle type, 3 jeux
    for nom, typ in (("prefabriques", "prefabrique"), ("coules en place", "coule")):
        r = verifier(_rect(8, 8), h, k, (6, 6), (0.5, 0.5), jeux, plots=(typ, 2.0, 2.0, 0.35))
        sj = [max(x["sigma_max"] for x in r if x["jeu"] == j) for j in (1, 2, 3)]
        print(f"T13 plots {nom} : sigma max = {sj[0]:.1f} / {sj[1]:.1f} / {sj[2]:.1f} kPa (jeux 1 / 2 / 3) ; "
              f"w max = {max(x['w_max'] for x in r):.2f} mm ; decolle = {max(x['decolle'] for x in r):.1f} %")
        equilibre += r
    # un cas ou les plots font baisser sigma max : grande dalle, sol raide (maximum sous le pied)
    v = [max(x["sigma_max"] for x in verifier(_rect(12, 12), 0.4, 30.0, (6, 6), (0.5, 0.5), jeux, plots=pt))
         for pt in (None, ("prefabrique", 2.0, 2.0, 0.35), ("coule", 2.0, 2.0, 0.35))]
    print(f"T13 dalle 12 x 12 x 0,40, k = 30 : sans plot = {v[0]:.1f} ; prefabriques = {v[1]:.1f} ; "
          f"coules en place = {v[2]:.1f} kPa")
    # controles : zone epaissie sur toute la dalle = dalle de 0,85 m ; plot bord sur bord
    ch = [(q, R/0.25) for q, R in zip(pl_type, jeux[2])]
    s1, s2 = [dd.resoudre([dd.charge(q_uniforme=25*0.85, plaques=ch)])[0]["sigma"].max()
              for dd in (Dalle(_rect(8, 8), h, k, zones=[(_rect(8, 8), 0.85)]), Dalle(_rect(8, 8), 0.85, k))]
    print(f"T13 controle : zone de 0,85 m sur toute la dalle = {s1:.2f} kPa ; dalle de 0,85 m = {s2:.2f} kPa")
    print("T13 controle : plot dans la dalle = "
          + " ; ".join(f"{dans_dalle(_rect(8, 8), plate_poly(1, 1, c, c))} (cote {c:.2f} m)" for c in (2.0, 2.2)))

    # T11 - equilibre : reaction du sol = charges, pour chaque cas de T4, T7, T8, T12 et T13
    ecart = max(abs(x["reaction_sol"] - x["charge_totale"])/x["charge_totale"] for x in equilibre)
    totaux = sorted(set(round(x["charge_totale"], 1) for x in equilibre))
    print(f"T11 equilibre sur {len(equilibre)} cas : ecart max reaction sol / charge = {100*ecart:.4f} % ; charges totales = {totaux} kN")

    # T9 - tableau par k (dalle type, 3 jeux, toutes positions)
    print("T9  dalle type, valeurs dimensionnantes par k :")
    for ln in sensibilite_k(_rect(8, 8), h, (2, 5, 10, 20, 30), (6, 6), (0.5, 0.5), jeux):
        print(f"    k = {ln['k']:>2} MN/m3 : sigma max = {ln['sigma_max']:6.1f} kPa ({ln['sigma_max']/KG_CM2:.2f} kg/cm2) ; "
              f"w max = {ln['w_max']:5.1f} mm ; decolle = {ln['decolle']:4.1f} %")

    # T10 - bandes d'integration (largeur 1,00 m)
    for nom, cont, cen, ch, be in (("dalle type", _rect(8, 8), (4, 4), (6, 6), 0.0), ("dalle en L", Lc, (4.2, 4.2), (4.5, 4.5), 30.0)):
        bs = bandes_integration(cont, cen[0], cen[1], ch[0], ch[1], be, 1.0)
        f3 = lambda v: f"{round(v, 6) + 0.0:.3f}"
        txt = " ; ".join(f"{x['nom']} ({f3(x['debut'][0])}, {f3(x['debut'][1])}) -> ({f3(x['fin'][0])}, {f3(x['fin'][1])})" for x in bs)
        print(f"T10 {nom} : {txt}")

    # T6 - dimensions proposees (dalle type, jeu 3, k = 12, limite 0,5 kg/cm2)
    t = time.time()
    for hh in (0.4, 0.5, 0.6, 0.8, 1.0):
        d_, s_ = proposer_rectangle(8, 8, hh, k, (6, 6), (0.5, 0.5), [jeux[2]])
        L = 8 + 2*d_
        print(f"T6  h = {hh:.2f} m -> L = {L:.2f} m ; sigma max = {s_:.1f} kPa ; beton {L*L*hh:.1f} m3")
    print(f"    ({time.time()-t:.1f} s)")

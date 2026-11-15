"""Fondation de grue - calcul de reference du module (verification rapide avant SCIA).

Dalle en beton arme deformable : plaque mince (Kirchhoff), element rectangle ACM 12 ddl
(w, dw/dx, dw/dy), grille reguliere, contour quelconque. Sol de Winkler (ressorts k x surface
tributaire aux noeuds) + Pasternak C2 en option. Sol en COMPRESSION SEULE par defaut
(le sol ne retient pas la dalle : les zones qui se soulevent sont liberees, iterations).
Unites : kN, m, kPa, MN/m3. w positif vers le bas ; contrainte de contact sigma = k.w.
Moments aux centres des elements : m > 0 = fibre inferieure tendue.

Charge de chaque plaque normalisee : la somme des forces verticales vaut exactement R, meme si la
plaque n'est pas alignee sur le maillage (sinon jusqu'a quelques % d'erreur).

Plots sous les pieds (option) : prefabriques (poids du plot + reaction diffusee a 45 deg dans
l'epaisseur du plot, carre centre sur le pied, rigidite de la dalle inchangee) ou coules en place
(poids du plot + dalle plus epaisse sous le plot).

Efforts pour le ferraillage (ELS, memes cas que la contrainte de contact) : moments mx, my, mxy et
efforts tranchants vx, vy aux noeuds, integres sur les bandes (largeur exacte), enveloppe de tous les
cas et de tous les k.

Usage : python fondation_grue_ref.py   -> tests T1 a T17.
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


def centre_poly(poly):
    """Centre de gravite d'un polygone."""
    P = np.asarray(poly, float); x, y = P[:, 0], P[:, 1]; x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x*y1 - x1*y; A = c.sum()/2
    return ((x + x1)*c).sum()/(6*A), ((y + y1)*c).sum()/(6*A)


def diffusion_45(p, h_pl, a_pl, b_pl, beta_deg=0.0):
    """Plot prefabrique : zone ou la reaction arrive sur la dalle. Diffusion a 45 deg dans l'epaisseur
    du plot : carre centre sur la surface d'appui, cotes paralleles aux axes du chassis (donc aux cotes
    du plot), cote = cote du carre de meme aire que la surface d'appui + 2 x hauteur du plot, sans
    depasser le plot. Plaque 0,50 x 0,50 et plot de 0,35 m : 1,20 x 1,20 m."""
    xc, yc = centre_poly(p)
    c = np.sqrt(aire(p)) + 2*h_pl
    return plate_poly(xc, yc, min(c, a_pl), min(c, b_pl), beta_deg)


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
        self._garde = np.zeros((self.ny, self.nx), bool)          # elements gardes (grille)
        self._garde[self.J, self.I] = True
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
                        "mx": mx, "my": my, "iterations": it+1, "x": x})
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
    m = _modele(contour, h, k, chassis, plaque, R_jeux, centre, beta, maille, E, c2, gamma_beton,
                plaques, plots)
    d, cas, Fs = m["dalle"], m["cas"], m["Fs"]
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


def _modele(contour, h, k, chassis, plaque, R_jeux, centre=None, beta=0.0, maille=0.25, E=32800.0,
            c2=0.0, gamma_beton=25.0, plaques=None, plots=None):
    """Dalle, surfaces d'appui, plots et vecteurs de charge de tous les cas (ELS : PP + plots + R)."""
    if centre is None:
        xs, ys = [p[0] for p in contour], [p[1] for p in contour]
        centre = ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2)
    a, b = chassis
    ap, bp = plaque
    if plaques is not None:
        pl = plaques
    else:
        pl = [plate_poly(x, y, ap, bp, beta) for x, y in legs(centre[0], centre[1], a, b, beta)]
    appuis = pl                                       # surfaces d'appui reelles (centres des bandes)
    pp, zones, h_pl = [], (), 0.0
    if plots is not None:
        typ, a_pl, b_pl, h_pl = plots
        pp = [plate_poly(*centre_poly(p), a_pl, b_pl, beta) for p in pl]
        if typ == "coule":
            zones = [(q, h + h_pl) for q in pp]
            charge = pl
        elif typ == "prefabrique":
            charge = [diffusion_45(p, h_pl, a_pl, b_pl, beta) for p in pl]
        else:
            raise ValueError("plots : type 'prefabrique' ou 'coule'")
    else:
        typ, charge = None, pl
    d = Dalle(contour, h, k, E_MPa=E, c2_MNm=c2, maille=maille, zones=zones)
    Fg = d.charge(q_uniforme=gamma_beton*h)
    for q in pp:                                      # poids de chaque plot, somme exacte
        f = d.charge(plaques=[(q, 1.0)])
        Fg = Fg + f*(gamma_beton*h_pl*aire(q)/f[0::3].sum())
    Fp = []
    for p in charge:
        f = d.charge(plaques=[(p, 1.0)])
        Fp.append(f/f[0::3].sum())   # charge unitaire EXACTE : somme des forces verticales = 1 kN
    cas = [(j, r) for j, R in enumerate(R_jeux, 1) for r in positions(R, chassis_carre=(abs(a-b) < 1e-9))]
    Fs = [Fg + sum(r[i]*Fp[i] for i in range(4)) for _, r in cas]
    # zone chargee de chaque pied, pour l'effort tranchant a d : plaque, carre de diffusion ou plot coule
    zch = pp if typ == "coule" else charge
    return {"dalle": d, "cas": cas, "Fs": Fs, "appuis": appuis, "charge": charge, "plots": pp,
            "type_plot": typ, "zones_chargees": zch, "centre": centre, "chassis": (a, b), "beta": beta}


# ---------------------------------------------------------------- efforts pour le ferraillage
_COINS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def champs_noeuds(d, x):
    """Moments mx, my, mxy (kNm/m) et efforts tranchants vx, vy (kN/m) aux noeuds.
    Moments : valeurs aux coins des elements, moyennees sur les elements voisins. Efforts tranchants :
    vx = -D (w,xxx + w,xyy), vy = -D (w,yyy + w,xxy) au centre des elements, moyennes aux noeuds.
    Convention : m > 0 = fibre inferieure tendue. Noeuds hors dalle : 0."""
    nn = d.ndof//3
    en = d.edofs[:, 0::3]//3
    X, Y = d.a/2, d.b/2
    z = np.zeros(12)
    pxxx, pxxy, pxyy, pyyy = z.copy(), z.copy(), z.copy(), z.copy()
    pxxx[6], pxxx[10] = 6, 6*Y
    pxxy[7], pxxy[10] = 2, 6*X
    pxyy[8], pxyy[11] = 2, 6*Y
    pyyy[9], pyyy[11] = 6, 6*X
    ue = x[d.edofs]
    vx = -d.De*(ue @ ((pxxx + pxyy) @ d.Ci))
    vy = -d.De*(ue @ ((pyyy + pxxy) @ d.Ci))
    M = np.zeros((nn, 3)); V = np.zeros((nn, 2)); cnt = np.zeros(nn)
    for ci, (u, v) in enumerate(_COINS):
        _, _, _, pxx, pyy, pxy = _basis(u*d.a, v*d.b)
        c = ue @ (np.vstack([pxx, pyy, pxy]) @ d.Ci).T
        m = np.stack([-d.De*(c[:, 0] + d.nu*c[:, 1]), -d.De*(c[:, 1] + d.nu*c[:, 0]),
                      -d.De*(1 - d.nu)*c[:, 2]], 1)
        np.add.at(M, en[:, ci], m); np.add.at(V, en[:, ci], np.stack([vx, vy], 1)); np.add.at(cnt, en[:, ci], 1)
    ok = cnt > 0
    M[ok] /= cnt[ok, None]; V[ok] /= cnt[ok, None]
    return M, V


def _interp(d, val, X, Y):
    """Interpolation bilineaire d'un champ nodal aux points (X, Y). Point hors des elements de la
    dalle : poids nul (renvoie 0)."""
    nnx = d.nx + 1
    fx = (np.asarray(X, float) - d.x0)/d.a; fy = (np.asarray(Y, float) - d.y0)/d.b
    i = np.clip(np.floor(fx).astype(int), 0, d.nx - 1); j = np.clip(np.floor(fy).astype(int), 0, d.ny - 1)
    u, v = fx - i, fy - j
    dedans = (fx >= -1e-9) & (fx <= d.nx + 1e-9) & (fy >= -1e-9) & (fy <= d.ny + 1e-9) & d._garde[j, i]
    n00 = j*nnx + i
    r = ((1-u)*(1-v)*val[n00] + u*(1-v)*val[n00+1] + u*v*val[n00+1+nnx] + (1-u)*v*val[n00+nnx])
    return np.where(dedans, r, 0.0)


def integrer_bande(d, val, p0, u, t0, t1, s, nt=None):
    """Integrale, sur la largeur t in [t0, t1] (exacte, bords interpoles), du champ nodal val, pour
    chaque abscisse s le long de l'axe p0 + s.u. Renvoie un tableau de meme taille que s."""
    n = np.array([-u[1], u[0]])
    if nt is None:
        nt = 2*int(np.ceil((t1 - t0)/min(d.a, d.b)*8)) + 1
    t = np.linspace(t0, t1, nt)
    S, T = np.meshgrid(s, t, indexing="ij")
    X = p0[0] + S*u[0] + T*n[0]; Y = p0[1] + S*u[1] + T*n[1]
    F = _interp(d, val, X.ravel(), Y.ravel()).reshape(S.shape)
    return np.trapezoid(F, t, axis=1)


def _etendue(contour, p, n, smax=50.0):
    """Distance du point p au contour dans la direction n (premier bord rencontre)."""
    best = smax
    m = len(contour)
    for i in range(m):
        a, b = np.asarray(contour[i], float), np.asarray(contour[(i+1) % m], float)
        e = b - a
        den = n[0]*(-e[1]) - n[1]*(-e[0])
        if abs(den) < 1e-12:
            continue
        r = a - p
        tt = (r[0]*(-e[1]) - r[1]*(-e[0]))/den
        uu = (n[0]*r[1] - n[1]*r[0])/den
        if tt > 1e-9 and -1e-9 <= uu <= 1 + 1e-9:
            best = min(best, tt)
    return best


def zones_renfort(contour, bandes):
    """Zone de renfort de chaque bande : la bande, prolongee jusqu'au bord de la dalle d'un cote si,
    sur toute la longueur de la bande, la dalle s'arrete a moins d'une largeur de bande au-dela de son
    bord (pas de lisiere etroite avec le seul treillis). Renvoie [(t_min, t_max), ...] par bande."""
    out = []
    for bd in bandes:
        p0, p1 = np.asarray(bd["debut"], float), np.asarray(bd["fin"], float)
        L = np.linalg.norm(p1 - p0); u = (p1 - p0)/L; n = np.array([-u[1], u[0]]); b = bd["largeur"]
        lim = []
        for sg in (-1, 1):
            ext = [_etendue(contour, p0 + s*u + sg*(b/2)*n, sg*n) for s in np.linspace(0.02*L, 0.98*L, 25)]
            e = max(ext)
            lim.append(sg*(b/2 + e) if e < b - 1e-9 else sg*b/2)
        out.append((lim[0], lim[1]))
    return out


def _rot_moments(M, u):
    """Moments dans le repere (u, n) : m_uu, m_nn, m_un."""
    c, s_ = u[0], u[1]
    mx, my, mxy = M[:, 0], M[:, 1], M[:, 2]
    muu = mx*c*c + my*s_*s_ + 2*mxy*c*s_
    mnn = mx*s_*s_ + my*c*c - 2*mxy*c*s_
    mun = (my - mx)*c*s_ + mxy*(c*c - s_*s_)
    return muu, mnn, mun


def efforts(contour, h, k_liste, chassis, plaque, R_jeux, centre=None, beta=0.0, maille=0.20,
            E=32800.0, c2_rapport=0.1, gamma_beton=25.0, plaques=None, plots=None, largeur=1.0, d_v=None):
    """Efforts ELS pour le ferraillage : memes cas que la contrainte de contact (PP + plots + reactions,
    toutes les positions de tous les jeux, sol en compression seule), pour chaque k de k_liste
    (C2 = c2_rapport x k, comme le module), puis enveloppe.
    Renvoie un dict :
      bandes : par bande (BX1, BX2, BY1, BY2) : s (m), Mmax, Mmin (kNm, integres sur la largeur),
               Vabs (kN), M_plus, M_moins, V_d (max |V| a d ou plus du bord des zones chargees),
               zone (t_min, t_max de la zone de renfort), prol_inf, prol_sup (moments de calcul
               par m, torsion comprise, dans le prolongement de la zone au-dela de la bande) ;
      courante : moments de calcul par m (torsion comprise : m + |mxy| en bas, m - |mxy| en haut),
               en x et en y, hors zones de renfort : inf_x, inf_y (max), sup_x, sup_y (min), et les memes
               dans les plots coules (section plus haute) : plot_inf, plot_sup ;
      champ : enveloppes au centre des elements (x, y, epaisseur h de l'element, masques zone_renfort
               et plot, moments de calcul inf_x, inf_y, sup_x, sup_y et moments sans torsion mx_max,
               my_max, mx_min, my_min), pour les cartes, le treillis et les renforts locaux.
    Bandes : valeurs aux noeuds integrees sur la largeur. Treillis et renforts locaux : valeurs au
    centre des elements, chaque element avec sa propre epaisseur (pas de moyenne a travers le bord
    d'un plot coule)."""
    if d_v is None:
        d_v = h - 0.06
    res = None
    for k in k_liste:
        m = _modele(contour, h, k, chassis, plaque, R_jeux, centre, beta, maille, E, c2_rapport*k,
                    gamma_beton, plaques, plots)
        d = m["dalle"]
        cen = [centre_poly(p) for p in m["appuis"]]
        a, b = m["chassis"]
        bandes = bandes_integration(contour, m["centre"][0], m["centre"][1], a, b, beta, largeur, centres=cen)
        zr = zones_renfort(contour, bandes)
        pieds = {"BX1": (0, 1), "BX2": (3, 2), "BY1": (0, 3), "BY2": (1, 2)}
        geo = []
        for bd, z in zip(bandes, zr):
            p0, p1 = np.asarray(bd["debut"], float), np.asarray(bd["fin"], float)
            L = np.linalg.norm(p1 - p0); u = (p1 - p0)/L
            s = np.arange(0.0, L + 1e-9, maille/4)
            inter = []
            for i in pieds[bd["nom"]]:
                proj = [(np.asarray(v, float) - p0) @ u for v in m["zones_chargees"][i]]
                inter.append((min(proj), max(proj)))
            ok_v = np.all([(s <= s1 - d_v + 1e-9) | (s >= s2 + d_v - 1e-9) for s1, s2 in inter], axis=0)
            geo.append((bd, z, p0, u, s, ok_v))
        # controles du treillis : valeurs au centre des elements (pas de moyenne a travers un
        # changement d'epaisseur), chaque element avec sa propre epaisseur
        xc = d.x0 + (d.I + 0.5)*d.a; yc = d.y0 + (d.J + 0.5)*d.b
        dans_zone = np.zeros(len(xc), bool)
        prol = []
        for bd, z, p0, u, s, ok_v in geo:
            n = np.array([-u[1], u[0]])
            S = (xc - p0[0])*u[0] + (yc - p0[1])*u[1]; T = (xc - p0[0])*n[0] + (yc - p0[1])*n[1]
            dz = (S >= -1e-6) & (S <= s[-1] + 1e-6) & (T >= z[0] - 1e-6) & (T <= z[1] + 1e-6)
            dans_zone |= dz
            prol.append(dz & (np.abs(T) > bd["largeur"]/2 + 1e-6))
        dans_plot = np.zeros(len(xc), bool)
        for q in (m["plots"] if m["type_plot"] == "coule" else []):
            dans_plot |= in_poly(xc, yc, q)
        cour = ~dans_zone
        sol = d.resoudre(m["Fs"])
        if res is None:
            res = {"bandes": [{"nom": bd["nom"], "s": s, "Mmax": np.full(len(s), -np.inf),
                               "Mmin": np.full(len(s), np.inf), "Vabs": np.zeros(len(s)), "ok_v": ok_v,
                               "zone": z, "prol_inf": 0.0, "prol_sup": 0.0}
                              for bd, z, p0, u, s, ok_v in geo],
                   "courante": {"inf_x": 0.0, "inf_y": 0.0, "sup_x": 0.0, "sup_y": 0.0,
                                "plot_inf": 0.0, "plot_sup": 0.0},
                   "champ": {"x": xc, "y": yc,
                             "h": h + np.where(dans_plot, plots[3] if dans_plot.any() else 0.0, 0.0),
                             "zone_renfort": dans_zone, "plot": dans_plot,
                             "inf_x": np.zeros(len(xc)), "inf_y": np.zeros(len(xc)),
                             "sup_x": np.zeros(len(xc)), "sup_y": np.zeros(len(xc)),
                             "mx_max": np.zeros(len(xc)), "my_max": np.zeros(len(xc)),
                             "mx_min": np.zeros(len(xc)), "my_min": np.zeros(len(xc))}}
        for sl in sol:
            M, V = champs_noeuds(d, sl["x"])
            for rb, (bd, z, p0, u, s, ok_v) in zip(res["bandes"], geo):
                muu, mnn, mun = _rot_moments(M, u)
                Mi = integrer_bande(d, muu, p0, u, -bd["largeur"]/2, bd["largeur"]/2, s)
                Vi = integrer_bande(d, V[:, 0]*u[0] + V[:, 1]*u[1], p0, u, -bd["largeur"]/2, bd["largeur"]/2, s)
                rb["Mmax"] = np.maximum(rb["Mmax"], Mi); rb["Mmin"] = np.minimum(rb["Mmin"], Mi)
                rb["Vabs"] = np.maximum(rb["Vabs"], np.abs(Vi))
            c = sl["x"][d.edofs] @ d.Bc.T
            Me = np.stack([-d.De*(c[:, 0] + d.nu*c[:, 1]), -d.De*(c[:, 1] + d.nu*c[:, 0]),
                           -d.De*(1 - d.nu)*c[:, 2]], 1)
            for rb, (bd, z, p0, u, s, ok_v), pr in zip(res["bandes"], geo, prol):
                if pr.any():
                    muu, mnn, mun = _rot_moments(Me, u)
                    rb["prol_inf"] = max(rb["prol_inf"], float((muu + np.abs(mun))[pr].max()))
                    rb["prol_sup"] = min(rb["prol_sup"], float((muu - np.abs(mun))[pr].min()))
            ax = np.abs(Me[:, 2])
            ch = res["champ"]
            ch["inf_x"] = np.maximum(ch["inf_x"], Me[:, 0] + ax); ch["inf_y"] = np.maximum(ch["inf_y"], Me[:, 1] + ax)
            ch["sup_x"] = np.minimum(ch["sup_x"], Me[:, 0] - ax); ch["sup_y"] = np.minimum(ch["sup_y"], Me[:, 1] - ax)
            ch["mx_max"] = np.maximum(ch["mx_max"], Me[:, 0]); ch["my_max"] = np.maximum(ch["my_max"], Me[:, 1])
            ch["mx_min"] = np.minimum(ch["mx_min"], Me[:, 0]); ch["my_min"] = np.minimum(ch["my_min"], Me[:, 1])
            c_ = res["courante"]
            for zone_, plot_ in ((cour & ~dans_plot, False), (cour & dans_plot, True)):
                if not zone_.any():
                    continue
                if not plot_:
                    c_["inf_x"] = max(c_["inf_x"], float((Me[:, 0] + ax)[zone_].max()))
                    c_["inf_y"] = max(c_["inf_y"], float((Me[:, 1] + ax)[zone_].max()))
                    c_["sup_x"] = min(c_["sup_x"], float((Me[:, 0] - ax)[zone_].min()))
                    c_["sup_y"] = min(c_["sup_y"], float((Me[:, 1] - ax)[zone_].min()))
                else:
                    c_["plot_inf"] = max(c_["plot_inf"], float((np.maximum(Me[:, 0], Me[:, 1]) + ax)[zone_].max()))
                    c_["plot_sup"] = min(c_["plot_sup"], float((np.minimum(Me[:, 0], Me[:, 1]) - ax)[zone_].min()))
    for rb in res["bandes"]:
        rb["M_plus"] = float(rb["Mmax"].max()); rb["M_moins"] = float(rb["Mmin"].min())
        rb["V_d"] = float(rb["Vabs"][rb["ok_v"]].max()) if rb["ok_v"].any() else 0.0
    return res


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

    # T14 - efforts ELS pour le ferraillage (dalle type, k = 2 ; 5 ; 10, C2 = k/10, maille 0,20 m,
    #       bandes de 1,00 m, d = 0,44 m) : enveloppe des 16 positions et des 3 k
    t = time.time()
    for nom, pt in (("sans plot", None), ("plots prefabriques", ("prefabrique", 2.0, 2.0, 0.35)),
                    ("plots coules", ("coule", 2.0, 2.0, 0.35))):
        ef = efforts(_rect(8, 8), h, (2, 5, 10), (6, 6), (0.5, 0.5), jeux, plots=pt, d_v=0.44)
        B = ef["bandes"]; c_ = ef["courante"]
        sym = max(max(x["M_plus"] for x in B) - min(x["M_plus"] for x in B),
                  max(x["M_moins"] for x in B) - min(x["M_moins"] for x in B),
                  max(x["V_d"] for x in B) - min(x["V_d"] for x in B))
        print(f"T14 {nom} : bandes M+ = {B[0]['M_plus']:6.1f} kNm ; M- = {B[0]['M_moins']:6.1f} kNm ; "
              f"V a d = {B[0]['V_d']:6.1f} kN (ecart entre les 4 bandes {sym:.2f})")
        txt = (f"inf x / y = {c_['inf_x']:5.1f} / {c_['inf_y']:5.1f} ; sup x / y = {c_['sup_x']:6.1f} / {c_['sup_y']:6.1f}")
        if pt and pt[0] == "coule":
            txt += f" ; dans les plots : inf {c_['plot_inf']:5.1f} / sup {c_['plot_sup']:6.1f}"
        print(f"    zone courante, centres d'elements (kNm/m, torsion comprise) : {txt}")
        print(f"    prolongements des bandes jusqu'au bord (kNm/m) : inf {max(x['prol_inf'] for x in B):5.1f} ; "
              f"sup {min(x['prol_sup'] for x in B):6.1f}")
    ef = efforts(_rect(8, 8), h, (2, 5, 10), (6, 6), (0.5, 0.5), jeux, largeur=0.5, d_v=0.44)
    B = ef["bandes"][0]
    print(f"T14 bandes de 0,50 m, sans plot : M+ = {B['M_plus']:5.1f} kNm ({B['M_plus']/0.5:5.1f} kNm/m) ; "
          f"M- = {B['M_moins']:6.1f} kNm ({B['M_moins']/0.5:6.1f} kNm/m) ; V a d = {B['V_d']:5.1f} kN ({B['V_d']/0.5:5.1f} kN/m)")
    print(f"    ({time.time()-t:.1f} s)")

    # T15 - integration sur la largeur exacte : champ lineaire f = 3 + y sur la bande y = 1,00 +/- 0,50
    #       (integrale exacte = 4,000), maille 0,20 (bords de bande au milieu des elements) et 0,25
    for mm in (0.20, 0.25):
        dd = Dalle(_rect(8, 8), h, k, maille=mm)
        f = 3 + dd.y0 + (np.arange(dd.ndof//3)//(dd.nx+1))*dd.b
        v = integrer_bande(dd, f, (0.0, 1.0), np.array([1.0, 0.0]), -0.5, 0.5, np.array([2.0, 4.1, 7.0]))
        yc = dd.y0 + (dd.J + 0.5)*dd.b; sel = (dd.I == 10) & (np.abs(yc - 1.0) < 0.5)
        naif = ((3 + yc[sel])*dd.b).sum()
        print(f"T15 maille {mm:.2f} : integrale = {v[0]:.3f} ; {v[1]:.3f} ; {v[2]:.3f} (exact 4,000) ; "
              f"centres d'elements strictement dans la bande : {naif:.3f}")

    # T16 - plots prefabriques : carre de diffusion a 45 deg (cote = carre de meme aire + 2 x 0,35)
    for nom, pq in (("plaque 0,50 x 0,50", plate_poly(1, 7, 0.5, 0.5)),
                    ("plaque 0,58 x 0,40 a 45 deg", platine_45(1, 7, -1, 1, 0.58, 0.40)),
                    ("plaque ronde D 0,50 (48 cotes)", plc[0])):
        q = diffusion_45(pq, 0.35, 2.0, 2.0)
        xs_, ys_ = [v[0] for v in q], [v[1] for v in q]
        cx, cy = centre_poly(pq)
        print(f"T16 {nom} : carre {max(xs_)-min(xs_):.3f} x {max(ys_)-min(ys_):.3f} m, centre ({cx:.3f} ; {cy:.3f})")
    q = diffusion_45(plate_poly(1, 7, 0.5, 0.5), 0.35, 1.0, 1.0)
    print(f"T16 plot de 1,00 x 1,00 : carre limite au plot {max(v[0] for v in q)-min(v[0] for v in q):.3f} m")

    # T17 - zones de renfort : bande prolongee jusqu'au bord si la lisiere restante est plus etroite que la bande
    for nom, L_ in (("dalle 8 x 8", 8.0), ("dalle 10 x 10", 10.0)):
        bs = bandes_integration(_rect(L_, L_), L_/2, L_/2, 6, 6, 0.0, 1.0)
        zr = zones_renfort(_rect(L_, L_), bs)
        print(f"T17 {nom} : " + " ; ".join(f"{x['nom']} t = {z[0]:+.2f} a {z[1]:+.2f}" for x, z in zip(bs, zr)))

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

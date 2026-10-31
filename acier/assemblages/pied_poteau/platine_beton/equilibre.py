# -*- coding: utf-8 -*-
"""Équilibre d'une platine rigide sous N + Mx + My — deux modèles, aucun maillage.

ÉLASTIQUE (répartition) : lit de ressorts de raideur ``kc`` SANS TRACTION sur
l'aire efficace, tiges = ressorts de raideur ``kb`` EN TRACTION SEULEMENT ;
inconnue : le plan de pénétration w = a + bx·x + by·y ; trois équations
(N, Mx, My) résolues par Newton, intégrales exactes sur les polygones de
contact. C'est le modèle qu'EN 1992-4 (section 6) impose pour dériver les
efforts dans les fixations : platine rigide, théorie de l'élasticité.

PLASTIQUE (résistance) : bloc à ``fjd`` sur la partie comprimée de l'aire
efficace, tiges du côté tendu À LEUR RÉSISTANCE (champ extrémal du théorème
statique : contraintes bornées, équilibre linéaire → le domaine des charges
équilibrables est convexe et sa frontière est tracée par ces champs
« tout ou rien ») ; balayage de l'orientation de l'axe neutre → frontière du
domaine (Mx, My) résistant à N. Une tige située exactement sur l'axe neutre
porte la fraction que l'équilibre de N impose.

NIVEAU 1 (verdict) : capacité uniaxiale au modèle de la Table 6.7
d'EN 1993-1-8 — résultante comprimée au centre de la bande de la paroi
comprimée, tiges du côté opposé réparties linéairement depuis ce centre,
plafonnées — puis interaction LINÉAIRE des deux axes, conservative par
convexité.

Unités internes : N, mm, N·mm, MPa, N/mm³. Signes : N > 0 compression ;
Mx > 0 comprime y > 0 ; My > 0 comprime x > 0.
"""
import math

from .geometrie import clip, moments

TWO_PI = 2 * math.pi


# ------------------------------------------------------------ élastique
def _pression(rects, a, bx, by):
    """Pour w = a + bx·x + by·y : (∫w, ∫w·x, ∫w·y) sur la partie w ≥ 0 de
    chaque rectangle, et les polygones de contact."""
    nn = math.hypot(bx, by)
    F = Sx = Sy = 0.0
    polys = []
    for (x0, y0, x1, y1) in rects:
        poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        part = (poly if a > 0 else []) if nn < 1e-15 else clip(poly, bx, by, -a)
        if len(part) < 3:
            continue
        A, sx, sy, ixx, ixy, iyy = moments(part)
        F += a * A + bx * sx + by * sy
        Sx += a * sx + bx * ixx + by * ixy
        Sy += a * sy + bx * ixy + by * iyy
        polys.append(part)
    return F, Sx, Sy, polys


def _resoudre3(J, r):
    """Résolution de J·du = r (3 × 3) par élimination de Gauss avec pivot."""
    M = [row[:] + [r[i]] for i, row in enumerate(J)]
    for c in range(3):
        p = max(range(c, 3), key=lambda i: abs(M[i][c]))
        if abs(M[p][c]) < 1e-300:
            raise ZeroDivisionError("jacobien singulier")
        M[c], M[p] = M[p], M[c]
        for i in range(3):
            if i != c:
                f = M[i][c] / M[c][c]
                for j in range(c, 4):
                    M[i][j] -= f * M[c][j]
    return [M[i][3] / M[i][i] for i in range(3)]


def elastique(rects, tiges, kc, kb, N, Mx, My, iter_max=300):
    """Répartition élastique. ``rects`` : aire efficace ; ``tiges`` : [(x, y)] ;
    ``kc`` (N/mm³), ``kb`` (N/mm). Retourne un dict : ``ok``, ``a, bx, by``,
    ``contact`` (polygones), ``A_c``, ``C`` (N), ``xC, yC``, ``T`` (N par tige),
    ``pmax`` (MPa), ``iter``."""
    A_tot = sum((x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in rects)
    res = dict(ok=True, a=0.0, bx=0.0, by=0.0, contact=[], A_c=0.0, C=0.0, xC=0.0, yC=0.0,
               T=[0.0] * len(tiges), pmax=0.0, iter=0)
    if abs(N) < 1e-9 and abs(Mx) < 1e-9 and abs(My) < 1e-9:
        return res
    cibles = (N, Mx, My)
    ech = (max(abs(N), 1e3), max(abs(Mx), 1e5), max(abs(My), 1e5))

    def resid(u):
        a, bx, by = u
        F, Sx, Sy, _ = _pression(rects, a, bx, by)
        r0, r1, r2 = kc * F, kc * Sy, kc * Sx
        for (x, y) in tiges:
            w = a + bx * x + by * y
            if w < 0:
                T = -kb * w
                r0 -= T
                r1 -= T * y
                r2 -= T * x
        return [r0 - cibles[0], r1 - cibles[1], r2 - cibles[2]]

    def norme(r):
        return max(abs(r[i]) / ech[i] for i in range(3))

    u = [N / (kc * A_tot) if (N > 0 and A_tot > 0) else 1e-3, 0.0, 0.0]
    r = resid(u)
    k = 0
    for k in range(1, iter_max + 1):
        if norme(r) < 1e-10:
            break
        J = [[0.0] * 3 for _ in range(3)]
        for j in range(3):
            hj = 1e-7 * max(1.0, abs(u[j])) if j == 0 else 1e-9 * max(1.0, abs(u[j]) * 1e4)
            up = u[:]; up[j] += hj
            um = u[:]; um[j] -= hj
            rp, rm = resid(up), resid(um)
            for i in range(3):
                J[i][j] = (rp[i] - rm[i]) / (2 * hj)
        try:
            du = _resoudre3(J, [-x for x in r])
        except ZeroDivisionError:
            res["ok"] = False
            break
        # recherche linéaire : on n'accepte qu'un pas qui réduit le résidu
        pas, n0 = 1.0, norme(r)
        for _ in range(30):
            un = [u[i] + pas * du[i] for i in range(3)]
            rn = resid(un)
            if norme(rn) < n0:
                u, r = un, rn
                break
            pas *= 0.5
        else:
            res["ok"] = False
            break
    if norme(r) > 1e-6:
        res["ok"] = False
    a, bx, by = u
    F, Sx, Sy, polys = _pression(rects, a, bx, by)
    C = kc * F
    res.update(a=a, bx=bx, by=by, contact=polys, iter=k,
               A_c=sum(moments(p)[0] for p in polys), C=C,
               xC=(kc * Sx / C) if C > 1e-9 else 0.0, yC=(kc * Sy / C) if C > 1e-9 else 0.0,
               T=[max(0.0, -kb * (a + bx * x + by * y)) for (x, y) in tiges],
               pmax=max([kc * (a + bx * x + by * y) for p in polys for (x, y) in p] or [0.0]))
    return res


# ------------------------------------------------------------- plastique
def _bloc(rects, nx, ny, d, fjd):
    """Bloc à fjd sur la partie nx·x + ny·y ≥ d de l'aire efficace : (C, ∫σx, ∫σy)."""
    C = Sx = Sy = 0.0
    for (x0, y0, x1, y1) in rects:
        part = clip([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], nx, ny, d)
        if len(part) >= 3:
            A, sx, sy, *_ = moments(part)
            C += fjd * A
            Sx += fjd * sx
            Sy += fjd * sy
    return C, Sx, Sy


def _tirants(tiges, FtRd, nx, ny, d):
    """Tiges du côté tendu (nx·x + ny·y < d, strictement) à leur résistance —
    champ extrémal. Le saut de N(d) au passage d'une tige est exactement en
    d = projection de la tige ; ``point_frontiere`` y place la fraction de
    force qu'impose l'équilibre de N."""
    return [FtRd[i] if d - (nx * x + ny * y) > 0.0 else 0.0 for i, (x, y) in enumerate(tiges)]


def point_frontiere(rects, tiges, FtRd, N, fjd, theta, iter_bis=48):
    """Pour une normale d'axe neutre ``theta`` (vers le côté comprimé) : la
    profondeur ``d`` qui équilibre N, puis (Mx, My) résistants. None si N ne
    peut pas être équilibré dans cette orientation."""
    nx, ny = math.cos(theta), math.sin(theta)
    vals = [nx * x + ny * y for (x0, y0, x1, y1) in rects for (x, y) in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    vals += [nx * x + ny * y for (x, y) in tiges]
    lo, hi = min(vals) - 1.0, max(vals) + 1.0

    def Nres(d):
        C, _sx, _sy = _bloc(rects, nx, ny, d, fjd)
        return C - sum(_tirants(tiges, FtRd, nx, ny, d))

    if N > Nres(lo) + 1e-6 or N < Nres(hi) - 1e-6:
        return None
    for _ in range(iter_bis):
        mid = 0.5 * (lo + hi)
        if Nres(mid) > N:
            lo = mid
        else:
            hi = mid
    d = 0.5 * (lo + hi)
    C, Sx, Sy = _bloc(rects, nx, ny, d, fjd)
    # tiges franchement tendues, franchement comprimées, ou SUR l'axe neutre
    # (à 1e-6 mm près — la bissection converge à 1e-11) : l'équilibre de N
    # tombe alors dans le saut, et ces tiges portent la fraction qui manque
    T, axe = [], []
    for i, (x, y) in enumerate(tiges):
        e = d - (nx * x + ny * y)
        if e > 1e-6:
            T.append(FtRd[i])
        else:
            T.append(0.0)
            if e > -1e-6:
                axe.append(i)
    reste = C - sum(T) - N
    for i in axe:
        part = min(max(reste, 0.0), FtRd[i])
        T[i] = part
        reste -= part
    Mx = Sy - sum(t * y for (x, y), t in zip(tiges, T))
    My = Sx - sum(t * x for (x, y), t in zip(tiges, T))
    return dict(theta=theta, d=d, Mx=Mx, My=My, C=C, T=T)


def domaine(rects, tiges, FtRd, N, fjd, n_theta=180):
    """La frontière du domaine (Mx, My) résistant à N : liste de points."""
    pts = []
    for k in range(n_theta):
        p = point_frontiere(rects, tiges, FtRd, N, fjd, TWO_PI * k / n_theta)
        if p is not None:
            pts.append(p)
    return pts


def _intersection_rayon(pa, pb, phi):
    """Intersection du rayon de direction ``phi`` (depuis l'origine) avec la
    corde [pa, pb] de la frontière — la corde d'un domaine convexe est à
    l'intérieur du domaine : le point rendu est admissible."""
    ux, uy = math.cos(phi), math.sin(phi)
    ax, ay = pa["Mx"], pa["My"]
    dx, dy = pb["Mx"] - ax, pb["My"] - ay
    det = ux * (-dy) - uy * (-dx)
    if abs(det) < 1e-12:
        return pa
    r = (ax * (-dy) - ay * (-dx)) / det
    t = (ux * ay - uy * ax) / det
    t = min(1.0, max(0.0, t))
    p = dict(pa)
    p.update(Mx=ax + t * dx, My=ay + t * dy, theta=(pa["theta"], pb["theta"]), corde=t, r=r)
    return p


def capacite_direction(rects, tiges, FtRd, N, fjd, phi, n=180):
    """Le point de la frontière dans la direction ``phi`` du plan (Mx, My).

    La direction du moment tourne avec l'axe neutre, par arcs (bloc qui
    glisse) et par SAUTS (une tige change de côté : face droite du domaine).
    On encadre ``phi`` entre deux points échantillonnés, on resserre
    l'encadrement par bissection sur l'angle de l'axe neutre, puis on coupe
    le rayon par la corde restante — exacte sur une face, intérieure (donc
    sûre) sur un arc."""
    def pt(theta):
        return point_frontiere(rects, tiges, FtRd, N, fjd, theta % TWO_PI)

    def ecart(p):
        return (math.atan2(p["My"], p["Mx"]) - phi + math.pi) % TWO_PI - math.pi

    ech = []
    for k in range(n):
        p = pt(TWO_PI * k / n)
        if p is not None:
            ech.append((TWO_PI * k / n, p))
    if not ech:
        return None
    for i in range(len(ech)):
        (ta, pa), (tb, pb) = ech[i], ech[(i + 1) % len(ech)]
        ea, eb = ecart(pa), ecart(pb)
        if abs(ea) < 1e-12:
            return pa
        if ea * eb < 0 and abs(ea - eb) < math.pi:
            lo, hi = ta, (tb if tb > ta else tb + TWO_PI)
            plo, phi_ = pa, pb
            elo = ea
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                pm = pt(mid)
                if pm is None:
                    break
                em = ecart(pm)
                if abs(em) < 1e-12:
                    return pm
                if (em < 0) == (elo < 0):
                    lo, plo, elo = mid, pm, em
                else:
                    hi, phi_ = mid, pm
            return _intersection_rayon(plo, phi_, phi)
    return min(ech, key=lambda e: abs(ecart(e[1])))[1]


# --------------------------------------------------------------- niveau 1
def capacite_niveau1(sens, zC, FC_Rd, tiges, FtRd, N):
    """Capacité uniaxiale au modèle de la Table 6.7 (EN 1993-1-8 §6.2.8.3),
    généralisé à plusieurs rangées : résultante comprimée au centre de la
    bande comprimée (bras ``zC``, résistance ``FC_Rd``), tiges du côté opposé
    réparties linéairement depuis ce centre, la plus éloignée plafonnée à sa
    résistance ; si N > FC,Rd, les deux parois comprimées se partagent N.
    ``sens`` = (nx, ny) du côté comprimé. Retourne ``(M_Rd, branche)`` en N·mm."""
    nx, ny = sens
    proj = [nx * x + ny * y for (x, y) in tiges]
    tendues = [i for i, p in enumerate(proj) if p < 0]
    if tendues:
        p_far = min(proj[i] for i in tendues)
        Tcap = [0.0] * len(tiges)
        for i in tendues:
            Tcap[i] = min(FtRd[i], FtRd[i] * (zC - proj[i]) / (zC - p_far)) if zC - p_far > 0 else FtRd[i]
    else:
        Tcap = [0.0] * len(tiges)
    S_T = sum(Tcap)
    M_T = sum(Tcap[i] * (-proj[i]) for i in range(len(tiges)))
    if N <= FC_Rd:
        lam_max = 1.0 if S_T <= 0 else min(1.0, (FC_Rd - N) / S_T)
        lam_min = 0.0 if N >= 0 else (float("inf") if S_T <= 0 else -N / S_T)
        if lam_min > lam_max + 1e-12:
            return 0.0, "tiges insuffisantes pour le soulèvement"
        lam = lam_max
        M = (N + lam * S_T) * zC + lam * M_T
        return M, ("tiges à leur résistance" if lam >= 1 - 1e-9 else "béton comprimé (une paroi)")
    if N <= 2 * FC_Rd:
        return (2 * FC_Rd - N) * zC, "béton comprimé (deux parois)"
    return 0.0, "compression supérieure à 2·FC,Rd"

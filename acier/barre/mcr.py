# -*- coding: utf-8 -*-
"""Le MOMENT CRITIQUE de déversement élastique M_cr d'un tronçon de poutre à section en I (ou en caisson) à deux axes
de symétrie — calcul NUMÉRIQUE, bibliothèque standard seule (docs/acier/BARRES_ACIER.md § 5.6).

Deux champs le long du tronçon : v, le déplacement latéral du centre de cisaillement ; φ, la rotation de torsion.
Repère : x le long de la poutre, z vers le HAUT, φ dans le sens direct autour de x — le déplacement latéral d'un point
de cote z vaut u(z) = v − z·φ. Moment M_y > 0 quand la fibre HAUTE est comprimée (poutre sur deux appuis sous charge
descendante). L'énergie potentielle au second ordre :

    Π = ½·∫(E·I_z·v″² + E·I_w·φ″² + G·I_t·φ′²)dx − ∫M_y·v″·φ dx − ½·∫q·z_g·φ² dx − ½·Σ P_i·z_g,i·φ(x_i)²

q, P : charges descendantes ; z_g : la cote de leur point d'application au-dessus du centre de cisaillement (une
charge posée sur la semelle haute déstabilise, z_g > 0 ; suspendue à la semelle basse, elle stabilise).

Éléments finis d'Hermite (cubiques) pour v et pour φ : quatre inconnues par nœud (v, v′, φ, φ′) ; intégration de Gauss
à quatre points, exacte pour un moment du second degré. Bouts du tronçon : « fourche » (v = φ = 0 ; gauchissement et
rotation latérale libres), « encastre » (v = v′ = φ = φ′ = 0), « libre ». Un MAINTIEN latéral à la cote z, à l'abscisse
x — une panne sur la semelle haute, un bracon sur la basse — est la liaison EXACTE v − z·φ = 0.

Le chargement est multiplié par λ ; la matrice K + λ·K_g cesse d'être définie positive au premier multiplicateur
critique. Le nombre de pivots négatifs de sa décomposition LDLᵀ (en bande) compte les multiplicateurs critiques entre
0 et λ : une bissection trouve le premier. M_cr = λ_cr·max|M_y|.

Contrôles (tests/test_barre_ec3_barre.py) : la formule fermée du moment constant ; les valeurs d'un solveur
indépendant (numpy/scipy) pour la charge répartie et la charge centrale, aux trois niveaux d'application ; la console
classique ; un maintien de la semelle comprimée à mi-portée vaut un maintien complet.
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

FOURCHE, ENCASTRE, LIBRE = "fourche", "encastre", "libre"
APPUIS = (FOURCHE, ENCASTRE, LIBRE)
ELEMENTS = 32                    # éléments au moins sur le tronçon (16 donnent déjà cinq chiffres)
TOLERANCE = 1e-10                # précision relative de la bissection
DOUBLEMENTS = 40                 # au-delà de 2⁴⁰ fois l'estimation de départ : pas de déversement
# quatre points de Gauss sur [0 ; 1] : (abscisse, poids)
_G4 = ((0.5 - 0.5 * 0.8611363115940526, 0.5 * 0.3478548451374538), (0.5 - 0.5 * 0.3399810435848563, 0.5 * 0.6521451548625461),
       (0.5 + 0.5 * 0.3399810435848563, 0.5 * 0.6521451548625461), (0.5 + 0.5 * 0.8611363115940526, 0.5 * 0.3478548451374538))


def moment_constant(L: float, EIz: float, EIw: float, GIt: float) -> float:
    """M_cr d'un tronçon entre fourches sous moment CONSTANT — la formule fermée :
    (π/L)·√(E·I_z·G·I_t)·√(1 + π²·E·I_w/(L²·G·I_t))."""
    return math.pi / L * math.sqrt(EIz * GIt) * math.sqrt(1.0 + math.pi ** 2 * EIw / (L ** 2 * GIt))


def _hermite(xi: float, Le: float):
    """Les quatre fonctions d'Hermite sur un élément de longueur Le, et leurs dérivées première et seconde en x."""
    N = (1 - 3 * xi * xi + 2 * xi ** 3, Le * (xi - 2 * xi * xi + xi ** 3), 3 * xi * xi - 2 * xi ** 3, Le * (-xi * xi + xi ** 3))
    dN = ((-6 * xi + 6 * xi * xi) / Le, 1 - 4 * xi + 3 * xi * xi, (6 * xi - 6 * xi * xi) / Le, -2 * xi + 3 * xi * xi)
    ddN = ((-6 + 12 * xi) / Le ** 2, (-4 + 6 * xi) / Le, (6 - 12 * xi) / Le ** 2, (-2 + 6 * xi) / Le)
    return N, dN, ddN


def _noeuds(L: float, points: Sequence[float], n_min: int) -> List[float]:
    """Les abscisses des nœuds : les points imposés (charges ponctuelles, maintiens, bornes des charges réparties),
    puis chaque intervalle recoupé pour ne pas dépasser L/n_min."""
    pts = sorted({0.0, L} | {min(max(float(x), 0.0), L) for x in points})
    fusion = [pts[0]]
    for x in pts[1:]:
        if x - fusion[-1] > 1e-9 * L:
            fusion.append(x)
    fusion[-1] = L
    xs = [fusion[0]]
    pas = L / n_min
    for a, b in zip(fusion, fusion[1:]):
        k = max(1, int(math.ceil((b - a) / pas - 1e-9)))
        xs += [a + (b - a) * i / k for i in range(1, k + 1)]
    return xs


def _pivots_negatifs(bande: List[List[float]], b: int) -> int:
    """Le nombre de pivots négatifs de la décomposition LDLᵀ d'une matrice symétrique en bande
    (bande[i][k] = A[i][i − k], k = 0…b) — par la loi d'inertie, son nombre de valeurs propres négatives."""
    n = len(bande)
    bas = [[0.0] * (b + 1) for _ in range(n)]
    d = [0.0] * n
    negatifs = 0
    for i in range(n):
        j0 = max(0, i - b)
        ligne = bas[i]
        for j in range(j0, i):
            s = bande[i][i - j]
            lj = bas[j]
            for k in range(max(j0, j - b), j):
                s -= ligne[i - k] * d[k] * lj[j - k]
            ligne[i - j] = s / d[j]
        s = bande[i][0]
        for k in range(j0, i):
            s -= ligne[i - k] ** 2 * d[k]
        if s == 0.0:
            s = -1e-300                       # pivot nul : on est SUR un multiplicateur critique, compté comme franchi
        d[i] = s
        if s < 0:
            negatifs += 1
    return negatifs


def moment_critique(L: float, EIz: float, EIw: float, GIt: float, moment: Callable[[float], float],
                    reparties: Sequence[Tuple[float, float, float, float]] = (), ponctuelles: Sequence[Tuple[float, float, float]] = (),
                    appuis: Tuple[str, str] = (FOURCHE, FOURCHE), maintiens: Sequence[Tuple[float, float]] = (),
                    M_max: Optional[float] = None, elements: int = ELEMENTS) -> Dict:
    """Le moment critique du tronçon : {"M_cr" (N·mm ; None : pas de déversement), "lambda" (le multiplicateur critique),
    "M_max" (N·mm), "noeuds"}.

    ``L`` (mm) ; ``EIz`` (N·mm²), ``EIw`` (N·mm⁴), ``GIt`` (N·mm²) ; ``moment(x)`` : M_y en N·mm, x de 0 à L ;
    ``reparties`` : [(x1, x2, q, z_g)] ou [(x1, x2, q1, q2, z_g)] — q en N/mm, vers le bas, linéaire de x1 à x2 ;
    ``ponctuelles`` : [(x, P, z_g)] (P en N) ;
    ``appuis`` : les deux bouts ; ``maintiens`` : [(x, z)] — déplacement latéral nul à la cote z ; ``M_max`` : le plus
    grand |M_y| du tronçon (échantillonné si absent)."""
    if not (L > 0 and EIz > 0 and GIt > 0 and EIw >= 0):
        raise ValueError("tronçon : L, E·I_z, G·I_t positifs et E·I_w positif ou nul")
    if any(a not in APPUIS for a in appuis) or len(appuis) != 2:
        raise ValueError(f"appuis : deux bouts parmi {APPUIS}")
    rep = [(r[0], r[1], r[2], r[2], r[3]) if len(r) == 4 else tuple(r) for r in reparties]      # (x1, x2, q1, q2, z_g)
    if any(len(r) != 5 or not r[1] > r[0] for r in rep):
        raise ValueError("charge répartie : (x1, x2, q, z_g) ou (x1, x2, q1, q2, z_g), avec x2 > x1")
    xs = _noeuds(L, [x for x, _, _ in ponctuelles] + [x for x, _ in maintiens] + [x for r in rep for x in (r[0], r[1])], elements)
    nn = len(xs)
    nd = 4 * nn
    K = [[0.0] * nd for _ in range(nd)]
    Kg = [[0.0] * nd for _ in range(nd)]
    for e in range(nn - 1):
        x0, Le = xs[e], xs[e + 1] - xs[e]
        iv = (4 * e, 4 * e + 1, 4 * e + 4, 4 * e + 5)
        ip = (4 * e + 2, 4 * e + 3, 4 * e + 6, 4 * e + 7)
        xm = x0 + 0.5 * Le
        dessus = [r for r in rep if r[0] - 1e-9 * L <= xm <= r[1] + 1e-9 * L]
        for xi, w in _G4:
            N, dN, ddN = _hermite(xi, Le)
            wt = w * Le
            x = x0 + xi * Le
            M = moment(x)
            qz = sum((q1 + (q2 - q1) * (x - a) / (b - a)) * zg for a, b, q1, q2, zg in dessus)
            for a in range(4):
                for b in range(4):
                    K[iv[a]][iv[b]] += wt * EIz * ddN[a] * ddN[b]
                    K[ip[a]][ip[b]] += wt * (EIw * ddN[a] * ddN[b] + GIt * dN[a] * dN[b])
                    c = -wt * M * ddN[a] * N[b]                     # − ∫ M_y·v″·φ
                    Kg[iv[a]][ip[b]] += c
                    Kg[ip[b]][iv[a]] += c
                    if qz:
                        Kg[ip[a]][ip[b]] -= wt * qz * N[a] * N[b]   # − ½·∫ q·z_g·φ²

    def noeud(x: float) -> int:
        return min(range(nn), key=lambda k: abs(xs[k] - x))

    for x, P, zg in ponctuelles:
        Kg[4 * noeud(x) + 2][4 * noeud(x) + 2] -= P * zg            # − ½·P·z_g·φ(x)²
    fixes = set()
    for bout, i in ((appuis[0], 0), (appuis[1], nn - 1)):
        if bout == FOURCHE:
            fixes |= {4 * i, 4 * i + 2}
        elif bout == ENCASTRE:
            fixes |= {4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3}
    for x, z in maintiens:                                          # v = z·φ : la ligne et la colonne de φ reçoivent z fois celles de v
        i = noeud(x)
        iv_, ip_ = 4 * i, 4 * i + 2
        if iv_ in fixes:
            continue                                                # déjà tenu (un bout en fourche)
        for A in (K, Kg):
            for j in range(nd):
                A[ip_][j] += z * A[iv_][j]
            for j in range(nd):
                A[j][ip_] += z * A[j][iv_]
        fixes.add(iv_)
    libres = [i for i in range(nd) if i not in fixes]
    n = len(libres)
    b = 0
    for r, i in enumerate(libres):
        for c_, j in enumerate(libres[:r]):
            if r - c_ > b and (K[i][j] != 0.0 or Kg[i][j] != 0.0):
                b = r - c_
    Kb = [[K[libres[r]][libres[r - k]] if r - k >= 0 else 0.0 for k in range(b + 1)] for r in range(n)]
    Gb = [[Kg[libres[r]][libres[r - k]] if r - k >= 0 else 0.0 for k in range(b + 1)] for r in range(n)]

    def franchis(lam: float) -> int:
        return _pivots_negatifs([[Kb[r][k] + lam * Gb[r][k] for k in range(b + 1)] for r in range(n)], b)

    if M_max is None:
        M_max = max(max(abs(moment(L * i / 800.0)) for i in range(801)), max(abs(moment(x)) for x in xs))
    out = {"M_cr": None, "lambda": None, "M_max": M_max, "noeuds": nn}
    if not M_max > 0:
        return out
    haut = moment_constant(L, EIz, EIw, GIt) / M_max
    bas_ = 0.0
    k = 0
    while franchis(haut) == 0:
        bas_, haut = haut, 2.0 * haut
        k += 1
        if k > DOUBLEMENTS:
            return out                                              # aucun multiplicateur critique : le tronçon ne déverse pas
    while haut - bas_ > TOLERANCE * haut:
        milieu = 0.5 * (bas_ + haut)
        if franchis(milieu) == 0:
            bas_ = milieu
        else:
            haut = milieu
    lam = 0.5 * (bas_ + haut)
    out.update({"M_cr": lam * M_max, "lambda": lam})
    return out

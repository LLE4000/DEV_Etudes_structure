# -*- coding: utf-8 -*-
"""Le MOMENT CRITIQUE de déversement élastique M_cr d'un tronçon de poutre à section en I (ou en caisson) — deux axes
de symétrie, ou un seul (PRS monosymétrique, terme de Wagner) — calcul NUMÉRIQUE, bibliothèque standard seule
(docs/acier/BARRES_ACIER.md § 5.6 et 5.10).

Deux champs le long du tronçon : v, le déplacement latéral du centre de cisaillement ; φ, la rotation de torsion.
Repère : x le long de la poutre, z vers le HAUT, φ dans le sens direct autour de x — le déplacement latéral d'un point
de cote z vaut u(z) = v − z·φ. Moment M_y > 0 quand la fibre HAUTE est comprimée (poutre sur deux appuis sous charge
descendante). L'énergie potentielle au second ordre :

    Π = ½·∫(E·I_z·v″² + E·I_w·φ″² + G·I_t·φ′²)dx − ∫M_y·v″·φ dx + ∫M_y·z_j·φ′² dx − ½·∫q·z_g·φ² dx − ½·Σ P_i·z_g,i·φ(x_i)²

q, P : charges descendantes ; z_g : la cote de leur point d'application au-dessus du centre de cisaillement (une
charge posée sur la semelle haute déstabilise, z_g > 0 ; suspendue à la semelle basse, elle stabilise).
∫M_y·z_j·φ′² (WAGNER, brief § 7.7) : nul pour une section bisymétrique ; pour un PRS monosymétrique, z_j est compté
vers la semelle HAUTE (acier/barre/prs.zj_haut) et M_y > 0 quand elle est comprimée — le même z_j sert pour les deux
signes du moment. Sous moment constant, la solution exacte est (E.1) avec C₁ = C₃ = 1 :
M_cr = (π²·E·I_z/L²)·[z_j + √(z_j² + I_w/I_z + L²·G·I_t/(π²·E·I_z))].

Éléments finis d'Hermite (cubiques) pour v et pour φ : quatre inconnues par nœud (v, v′, φ, φ′) ; intégration de Gauss
à quatre points, exacte pour un moment du second degré. Bouts du tronçon : « fourche » (v = φ = 0 ; gauchissement et
rotation latérale libres), « encastre » (v = v′ = φ = φ′ = 0), « libre ». Un MAINTIEN latéral à la cote z, à l'abscisse
x — une panne sur la semelle haute, un bracon sur la basse — est la liaison EXACTE v − z·φ = 0.

Le chargement est multiplié par λ ; la matrice K + λ·K_g cesse d'être définie positive au premier multiplicateur
critique. Le nombre de pivots négatifs de sa décomposition LDLᵀ (en bande) compte les multiplicateurs critiques entre
0 et λ : il ENCADRE le premier (doublements, puis bissection tant que plus d'un est franchi). Dans l'encadrement, le
déterminant — le produit des pivots — change de signe une seule fois : la sécante (fausse position « Illinois ») y
converge en une dizaine de décompositions au lieu de trente-cinq ; le COMPTE des pivots reste le juge du côté, et un pas
de sécante douteux est remplacé par une bissection. M_cr = λ_cr·max|M_y|.

Deux chargements PROPORTIONNELS ont le même M_cr (λ_cr est inversement proportionnel au chargement) : le résultat est
gardé sous la signature du chargement ramené à max|M_y| = 1 — 1,35·G et 1,00·G, ou G et Q de même forme, ne se
calculent qu'une fois.

Contrôles (tests/test_barre_ec3_barre.py) : la formule fermée du moment constant ; les valeurs d'un solveur
indépendant (numpy/scipy) pour la charge répartie et la charge centrale, aux trois niveaux d'application ; la console
classique ; un maintien de la semelle comprimée à mi-portée vaut un maintien complet.
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .resultat import fr

FOURCHE, ENCASTRE, LIBRE = "fourche", "encastre", "libre"
APPUIS = (FOURCHE, ENCASTRE, LIBRE)
ELEMENTS = 32                    # éléments au moins sur le tronçon (16 donnent déjà cinq chiffres)
TOLERANCE = 1e-10                # précision relative de l'encadrement du multiplicateur critique
DOUBLEMENTS = 40                 # au-delà de 2⁴⁰ fois l'estimation de départ : pas de déversement
SECANTES = 60                    # pas de sécante au plus ; au-delà, bissection pure (jamais atteint en pratique)
MEMOIRE = 512                    # chargements gardés (au-delà, la mémoire est vidée)
CHIFFRES = 10                    # décimales de la signature d'un chargement ramené à max|M_y| = 1
_GARDES: Dict[Tuple, float] = {}
# quatre points de Gauss sur [0 ; 1] : (abscisse, poids)
_G4 = ((0.5 - 0.5 * 0.8611363115940526, 0.5 * 0.3478548451374538), (0.5 - 0.5 * 0.3399810435848563, 0.5 * 0.6521451548625461),
       (0.5 + 0.5 * 0.3399810435848563, 0.5 * 0.6521451548625461), (0.5 + 0.5 * 0.8611363115940526, 0.5 * 0.3478548451374538))


def moment_constant(L: float, EIz: float, EIw: float, GIt: float, zj: float = 0.0) -> float:
    """M_cr d'un tronçon entre fourches sous moment CONSTANT — la formule fermée :
    (π/L)·√(E·I_z·G·I_t)·√(1 + π²·E·I_w/(L²·G·I_t)) ; avec ``zj`` (mm, positif vers la semelle COMPRIMÉE) :
    P_z·[z_j + √(z_j² + E·I_w/(E·I_z) + G·I_t/P_z)], P_z = π²·E·I_z/L² (la même valeur quand z_j = 0)."""
    if zj:
        Pz = math.pi ** 2 * EIz / L ** 2
        return Pz * (zj + math.sqrt(zj * zj + EIw / EIz + GIt / Pz))
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


def _decomposition(bande: List[List[float]], b: int) -> Tuple[int, float, int]:
    """La décomposition LDLᵀ d'une matrice symétrique en bande (bande[i][k] = A[i][i − k], k = 0…b) :
    (le nombre de pivots négatifs — par la loi d'inertie, son nombre de valeurs propres négatives —, puis le
    déterminant, produit des pivots, en mantisse et exposant de 2 pour ne pas déborder)."""
    n = len(bande)
    u = [[0.0] * (b + 1) for _ in range(n)]      # u[i][i − j] = L[i][j]·d[j]
    bas = [[0.0] * (b + 1) for _ in range(n)]    # bas[i][i − j] = L[i][j]
    d = [0.0] * n
    negatifs, mantisse, exposant = 0, 1.0, 0
    for i in range(n):
        j0 = i - b if i > b else 0
        ui, li, bi = u[i], bas[i], bande[i]
        for j in range(j0, i):
            # L[i][j]·d[j] = A[i][j] − Σ_k L[i][k]·d[k]·L[j][k], k de j0 à j − 1 (avant j0, la ligne i est hors de la bande) ;
            # avec m = j − k : L[i][k]·d[k] = ui[i − j + m], L[j][k] = bas[j][m]
            o = i - j
            s = bi[o]
            lj = bas[j]
            for m in range(1, j - j0 + 1):
                s -= ui[o + m] * lj[m]
            ui[o] = s
            li[o] = s / d[j]
        s = bi[0]
        for m in range(1, i - j0 + 1):
            s -= ui[m] * li[m]
        if s == 0.0:
            s = -1e-300                       # pivot nul : on est SUR un multiplicateur critique, compté comme franchi
        d[i] = s
        if s < 0:
            negatifs += 1
        m, e = math.frexp(s)
        mantisse *= m
        exposant += e
        if abs(mantisse) < 1e-150:            # (une mantisse est entre ½ et 1 : le produit se renormalise de loin en loin)
            m, e = math.frexp(mantisse)
            mantisse, exposant = m, exposant + e
    return negatifs, mantisse, exposant


def _rapport(fa: Tuple[float, int], fb: Tuple[float, int]) -> Optional[float]:
    """f(a) / f(b) pour deux déterminants (mantisse, exposant) ; None s'il déborde."""
    try:
        return math.ldexp(fa[0] / fb[0], fa[1] - fb[1])
    except (OverflowError, ZeroDivisionError):
        return None


def _premier(lire: Callable[[float], Tuple[int, float, int]], depart: float) -> Optional[float]:
    """Le premier multiplicateur critique : le plus petit λ > 0 où K + λ·K_g cesse d'être définie positive ; None s'il
    n'y en a pas. ``lire(λ)`` rend la décomposition de K + λ·K_g ; ``depart`` : une estimation, doublée tant que rien
    n'est franchi."""
    bas, f_bas, haut = 0.0, None, depart
    for _ in range(DOUBLEMENTS + 1):
        franchis, m, e = lire(haut)
        if franchis:
            break
        bas, f_bas, haut = haut, (m, e), 2.0 * haut
    else:
        return None                                                 # aucun multiplicateur critique : le tronçon ne déverse pas
    f_haut = (m, e) if franchis == 1 else None
    if f_bas is None:
        f_bas = lire(bas)[1:]
    cote, pas = 0, 0
    while haut - bas > TOLERANCE * haut:
        # un seul multiplicateur dans l'encadrement : le déterminant y change de signe une fois, la sécante le vise ;
        # plusieurs (ou un pas hors de l'encadrement) : bissection — le déterminant ne dirait pas lequel est le premier
        secante = False
        if f_haut is not None and pas < SECANTES:
            r = _rapport(f_bas, f_haut)                              # f(bas) / f(haut) < 0
            if r is not None and r < 0:
                essai = (bas - haut * r) / (1.0 - r)
                secante = bas < essai < haut
                pas += 1
        if not secante:
            essai = 0.5 * (bas + haut)
        fr, m, e = lire(essai)
        if fr == 0:                                                  # le COMPTE juge du côté ; le déterminant ne fait que guider
            bas, f_bas = essai, (m, e)
            if secante and cote == -1:
                f_haut = (0.5 * f_haut[0], f_haut[1])                # « Illinois » : le bout resté en place pèse moitié moins
            cote = -1 if secante else 0
        else:
            if secante and cote == 1 and fr == 1:
                f_bas = (0.5 * f_bas[0], f_bas[1])
            haut, f_haut = essai, ((m, e) if fr == 1 else None)
            cote = 1 if secante else 0
    return 0.5 * (bas + haut)


def _bandes(xs: Sequence[float], gauss: Sequence[Tuple[float, float]], EIz: float, EIw: float, GIt: float, ponctuelles: Sequence[Tuple],
            appuis: Tuple[str, str], maintiens: Sequence[Tuple[float, float]], noeud: Callable[[float], int],
            zj: float = 0.0) -> Tuple[List, List, int]:
    """K et K_g assemblées, liaisons posées, en bande : (Kb, Gb, demi-largeur) — bande[i][k] = A[i][i − k]."""
    nn = len(xs)
    nd = 4 * nn
    K = [[0.0] * nd for _ in range(nd)]
    Kg = [[0.0] * nd for _ in range(nd)]
    for e in range(nn - 1):
        Le = xs[e + 1] - xs[e]
        iv = (4 * e, 4 * e + 1, 4 * e + 4, 4 * e + 5)
        ip = (4 * e + 2, 4 * e + 3, 4 * e + 6, 4 * e + 7)
        for g, (xi, w) in enumerate(_G4):
            N, dN, ddN = _hermite(xi, Le)
            wt = w * Le
            M, qz = gauss[4 * e + g]
            for a in range(4):
                for b in range(4):
                    K[iv[a]][iv[b]] += wt * EIz * ddN[a] * ddN[b]
                    K[ip[a]][ip[b]] += wt * (EIw * ddN[a] * ddN[b] + GIt * dN[a] * dN[b])
                    c = -wt * M * ddN[a] * N[b]                     # − ∫ M_y·v″·φ
                    Kg[iv[a]][ip[b]] += c
                    Kg[ip[b]][iv[a]] += c
                    if qz:
                        Kg[ip[a]][ip[b]] -= wt * qz * N[a] * N[b]   # − ½·∫ q·z_g·φ²
                    if zj:
                        Kg[ip[a]][ip[b]] += 2.0 * wt * M * zj * dN[a] * dN[b]   # + ∫ M_y·z_j·φ′² (Wagner)
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
        Ki, Gi = K[i], Kg[i]
        for c_ in range(r - b):                                     # (au-delà de la demi-largeur déjà connue seulement)
            j = libres[c_]
            if Ki[j] != 0.0 or Gi[j] != 0.0:
                b = r - c_
                break
    Kb = [[K[libres[r]][libres[r - k]] if r - k >= 0 else 0.0 for k in range(b + 1)] for r in range(n)]
    Gb = [[Kg[libres[r]][libres[r - k]] if r - k >= 0 else 0.0 for k in range(b + 1)] for r in range(n)]
    return Kb, Gb, b


def moment_critique(L: float, EIz: float, EIw: float, GIt: float, moment: Callable[[float], float],
                    reparties: Sequence[Tuple[float, float, float, float]] = (), ponctuelles: Sequence[Tuple[float, float, float]] = (),
                    appuis: Tuple[str, str] = (FOURCHE, FOURCHE), maintiens: Sequence[Tuple[float, float]] = (),
                    M_max: Optional[float] = None, elements: int = ELEMENTS, zj: float = 0.0) -> Dict:
    """Le moment critique du tronçon : {"M_cr" (N·mm ; None : pas de déversement), "lambda" (le multiplicateur critique),
    "M_max" (N·mm), "noeuds"}. ``zj`` (mm) : le z_j de la section quand la semelle HAUTE est comprimée (0 : bisymétrique).

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

    def noeud(x: float) -> int:
        return min(range(nn), key=lambda k: abs(xs[k] - x))

    # le chargement aux points de Gauss — (M_y, q·z_g) —, lu une fois : il fait la signature, puis l'assemblage
    gauss = []
    for e in range(nn - 1):
        x0, Le = xs[e], xs[e + 1] - xs[e]
        xm = x0 + 0.5 * Le
        dessus = [r for r in rep if r[0] - 1e-9 * L <= xm <= r[1] + 1e-9 * L]
        for xi, _ in _G4:
            x = x0 + xi * Le
            gauss.append((moment(x), sum((q1 + (q2 - q1) * (x - a) / (b - a)) * zg for a, b, q1, q2, zg in dessus)))
    if M_max is None:
        M_max = max(max(abs(moment(L * i / 800.0)) for i in range(801)), max(abs(moment(x)) for x in xs))
    out = {"M_cr": None, "lambda": None, "M_max": M_max, "noeuds": nn}
    if not M_max > 0:
        return out
    cle = (L, EIz, EIw, GIt, round(zj, 9), tuple(appuis), tuple((round(x, 6), z) for x, z in maintiens), tuple(round(x, 6) for x in xs),
           tuple(round(M / M_max, CHIFFRES) for M, _ in gauss), tuple(round(qz * L / M_max, CHIFFRES) for _, qz in gauss),
           tuple((noeud(x), round(P * zg / M_max, CHIFFRES)) for x, P, zg in ponctuelles))
    garde = _GARDES.get(cle)                                        # λ_cr·max|M_y| d'un chargement proportionnel déjà résolu
    if garde is None:
        Kb, Gb, b = _bandes(xs, gauss, EIz, EIw, GIt, ponctuelles, appuis, maintiens, noeud, zj)
        lam = _premier(lambda t: _decomposition([[k_ + t * g_ for k_, g_ in zip(kr, gr)] for kr, gr in zip(Kb, Gb)], b),
                       moment_constant(L, EIz, EIw, GIt) / M_max)
        garde = -1.0 if lam is None else lam * M_max                # (−1 : le tronçon ne déverse pas)
        if len(_GARDES) >= MEMOIRE:
            _GARDES.clear()
        _GARDES[cle] = garde
    if garde > 0:
        lam = garde / M_max
        out.update({"M_cr": lam * M_max, "lambda": lam})
    return out


# ------------------------------------------------------------------ le contrôle analytique : annexe E ANB, (E.1)
# NBN EN 1993-1-1 ANB:2018, annexe E, k_z = 1 (brief § 7.7 et 18.5) — tableau E.1 ANB (moments d'extrémité, C₂·z_g = 0) :
# C₁ pour ψ = 1, 3/4, 1/2, 1/4, 0, −1/4, −1/2, −3/4, −1 ; C₃, colonne k_z = 1, en deux colonnes selon le signe de ψ_f
# (ψ_f ≤ 0 ; ψ_f > 0), chaque valeur écrite a + b·ψ_f : E1_C3[i] = ((a, b) pour ψ_f ≤ 0, (a, b) pour ψ_f > 0). Le module
# n'a que k_z = 1 (la colonne k_z = 0,5 n'est pas codée). La colonne C₃ n'est pas dans le brief : relue sur le texte de
# l'ANB (hors du dépôt) — À VÉRIFIER sur l'image du tableau. Tableau E.2 ANB (charges transversales, deux appuis) : (C₁, C₂, C₃).
E1_PSI = (1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0)
E1_C1 = (1.00, 1.14, 1.31, 1.52, 1.77, 2.06, 2.35, 2.60, 2.60)
_UN = (1.0, 0.0)
E1_C3 = ((_UN, _UN), (_UN, _UN), (_UN, _UN), (_UN, _UN), (_UN, _UN), (_UN, (0.850, 0.0)), (_UN, (1.3, -1.2)), (_UN, (0.55, -1.0)),
         ((0.0, -1.0), (0.0, -1.0)))
E2 = {"reparti": (1.12, 0.45, 0.525), "ponctuel": (1.35, 0.59, 0.411), "quarts": (1.04, 0.42, 0.562)}
PSI_F_MAX = 0.9                  # domaine des tableaux : −0,9 ≤ ψ_f ≤ 0,9 (annexe E ANB)
NOTE_B = 1.05                    # moments d'extrémité, K_ωt ≤ 1 : C₁/1,05, jamais sous 1


def kwt(L: float, EIw: float, GIt: float, kw: float = 1.0) -> float:
    """K_ωt = π/(k_ω·L)·√(E·I_ω/(G·I_t)) (annexe E ANB, note b des tableaux)."""
    return math.pi / (kw * L) * math.sqrt(EIw / GIt)


def coefficients_annexe_e(diagramme: str, psi: float = 1.0, psi_f: float = 0.0, Kwt: Optional[float] = None) -> Dict:
    """(C₁, C₂, C₃) de l'annexe E ANB pour k_z = 1 : {"C1", "C2", "C3", "source", "bloque" (None, ou le motif :
    diagramme hors des tableaux, ψ ou ψ_f hors des tableaux)}. ``diagramme`` : « lineaire » ou
    « constant » (moments d'extrémité, tableau E.1 ANB, ψ = M₂/M₁ — une valeur du tableau seulement), « reparti »,
    « ponctuel » (charge à mi-portée), « quarts » (deux charges aux quarts) : tableau E.2 ANB. ``psi_f`` : celui de la
    section (0 si bisymétrique) ; C₃ n'est demandé que pour ψ_f ≠ 0."""
    out = {"C1": None, "C2": None, "C3": None, "source": None, "bloque": None}
    if abs(psi_f) > PSI_F_MAX:
        out["bloque"] = f"ψ_f = {fr(psi_f, 3)} hors du domaine des tableaux (−0,9 ≤ ψ_f ≤ 0,9)"
        return out
    if diagramme in E2:
        out.update(dict(zip(("C1", "C2", "C3"), E2[diagramme])), source="tableau E.2 ANB")
        return out
    if diagramme not in ("constant", "lineaire"):
        out["bloque"] = "diagramme hors des tableaux E.1 ANB et E.2 ANB : le solveur seul"
        return out
    p = 1.0 if diagramme == "constant" else psi
    i = next((k for k, x in enumerate(E1_PSI) if abs(x - p) < 1e-9), None)
    if i is None:
        out["bloque"] = f"ψ = {fr(p, 2)} n'est pas une ligne du tableau E.1 ANB : le solveur seul"
        return out
    c1 = E1_C1[i]
    if Kwt is not None and Kwt <= 1.0:
        c1 = max(c1 / NOTE_B, 1.0)
    out.update({"C1": c1, "C2": 0.0, "source": "tableau E.1 ANB" + (" (note b : C₁/1,05)" if Kwt is not None and Kwt <= 1.0 and E1_C1[i] > 1.0 else "")})
    if psi_f:
        a, b = E1_C3[i][psi_f > 0]
        out["C3"] = a + b * psi_f
    else:
        out["C3"] = 0.0
    return out


def annexe_e(L: float, EIz: float, EIw: float, GIt: float, C1: float, C2: float, C3: float, zg: float, zj: float,
             kz: float = 1.0, kw: float = 1.0) -> float:
    """M_cr par (E.1) de l'annexe E ANB (N·mm) :
        C₁·π²·E·I_z/(k_z·L)²·{√[(k_z/k_ω)²·I_ω/I_z + (k_z·L)²·G·I_t/(π²·E·I_z) + (C₂·z_g − C₃·z_j)²] − (C₂·z_g − C₃·z_j)}
    z_g = z_a − z_s, positif quand la charge va VERS le centre de cisaillement depuis son point d'application (annexe E
    ANB, 2(8) : une charge de gravité au-dessus de lui) ; z_j positif vers la semelle comprimée (2(7)) ; bisymétrique :
    z_j = 0, c'est (E.8)."""
    Pz = math.pi ** 2 * EIz / (kz * L) ** 2
    x = C2 * zg - C3 * zj
    return C1 * Pz * (math.sqrt((kz / kw) ** 2 * EIw / EIz + (kz * L) ** 2 * GIt / (math.pi ** 2 * EIz) + x * x) - x)

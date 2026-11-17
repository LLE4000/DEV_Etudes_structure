# -*- coding: utf-8 -*-
"""La STATIQUE d'une barre chargée : chaque cas de charge résolu par le formulaire de poutres du dépôt (rdm/poutres.py,
écritures fermées), puis superposé — le comportement est linéaire. docs/acier/BARRES_ACIER.md § 9.

Unités de la barre : m, kN, kN/m, kN·m (celles de l'écran et du formulaire). Les moteurs de vérification travaillent en
N et mm : ``efforts`` convertit. Conventions du formulaire, reprises telles quelles : une charge vers le bas est
positive ; M > 0 tend la fibre inférieure (la fibre HAUTE est comprimée) ; flèche positive vers le bas.

    poids propre   g = 78,6 kN/m³ × A (kN/m), un cas permanent de plus — le poids volumique des classeurs du bureau
                   (brief H13 ; décision du propriétaire du 16/11/2026 : « 78,6 partout »)
    une combinaison  M(x) = Σ γ_i·M_i(x),  V(x) = Σ γ_i·V_i(x),  N = Σ γ_i·N_i,  f(x) = Σ γ_i·f_i(x)
    sections à vérifier : les appuis, chaque rupture de chargement, et les extremums du moment — là où l'effort
    tranchant de la combinaison s'annule. Entre deux ruptures, V est un polynôme de degré 2 au plus (charges
    uniformes et triangulaires) : trois lectures donnent ses racines, sans balayage.

Les lectures sont GARDÉES : toutes les combinaisons relisent les mêmes sections des mêmes cas.

LES TRONÇONS DE DÉVERSEMENT : entre deux sections tenues en déplacement latéral ET en torsion — les appuis (en fourche à
défaut, ou encastrés en torsion : ``appuis.detail``), l'encastrement d'une console, et chaque maintien « complet ». Un
maintien d'une seule semelle (« haut », « bas ») reste À L'INTÉRIEUR de son tronçon. Le bout libre d'une console, et
celui d'un débord, est libre. Pour M_cr (``pour_mcr``), les maintiens « négligés dans M_cr » ne comptent pas (brief § 4.1,
§ 7.7 : la variante (0,4 ; 0,75) de l'ANB) ; le flambement et la méthode ELS les gardent.

(Lot C, C1) LA BARRE CHARGÉE DE BOUT EN BOUT :
    une hauteur PAR CHARGE (``niveau`` ; à défaut celle de la barre) ; le poids propre au CENTRE DE GRAVITÉ (z_g = −z_s,
    0 pour un I bisymétrique) ;
    la POSITION de la barre pour le poids propre : horizontale (en flexion), verticale (q·L dans N, en compression, sur
    toute la barre — prudent, À VÉRIFIER : il croît en réalité de 0 à q·L), inclinée d'un angle α sur l'horizontale
    (q·cos α en flexion, q·L·sin α dans N) ;
    N PAR TRONÇON (``N_troncons`` d'un cas : [{"x2", "N"}…], il remplace son N constant) : N(x) de la combinaison ;
    les DÉBORDS (porte-à-faux aux rives, rdm/poutres.py) : la barre va de 0 à g + Σ portées + d, le premier appui en x = g.
"""
import functools
import math
from typing import Dict, List, Optional, Sequence, Tuple

from rdm import poutres as PO

from . import mcr as MC
from . import section as SE
from .resultat import Refus

POIDS_VOLUMIQUE = 78.6           # kN/m³ : l'acier des classeurs du bureau (brief H13, décision du 16/11/2026)
POIDS_PROPRE = "Poids propre"
GENRES_MAINTIEN = ("complet", "haut", "bas")
NEGLIGEABLE = 1e-9               # un effort tranchant plus petit que cette part du plus grand est pris nul
GRAVITE = "gravite"              # le « niveau » du poids propre : le centre de gravité de la section (jamais saisi)
RACINES_ITERATIONS = 60          # dichotomie d'une racine du moment (60 pas : 10⁻¹⁸ de la longueur de l'intervalle)


def poids_propre(s: Dict) -> float:
    """Le poids propre du profilé en kN/m : 78,6 kN/m³ × A (A en mm² → × 10⁻⁶ m²)."""
    if not s.get("A"):
        raise Refus("l'aire du profilé manque : le poids propre en dépend")
    return POIDS_VOLUMIQUE * s["A"] * 1e-6


def repartition(position: str, angle: Optional[float] = None) -> Tuple[float, float]:
    """(part dans N, part en flexion) du poids propre selon la POSITION de la barre (brief § 4.1, § 6.2 point 7) :
    horizontale (0, 1) ; verticale (1, 0) ; inclinée d'un angle α (degrés) sur l'horizontale (sin α, cos α). Une barre
    inclinée sans angle est refusée (aucune valeur supposée)."""
    if position in (None, "horizontale"):
        return 0.0, 1.0
    if position == "verticale":
        return 1.0, 0.0
    if position == "inclinee":
        if angle is None:
            raise Refus("barre inclinée : l'angle α sur l'horizontale est à saisir (0 à 90°)")
        a = math.radians(angle)
        return math.sin(a), math.cos(a)
    raise Refus(f"position de la barre : horizontale, verticale ou inclinée — reçu {position!r}")


def longueur(barre: Dict) -> float:
    """La longueur de la BARRE (m) : ses portées et ses débords (porte-à-faux aux rives, rdm/poutres.py)."""
    g, d = (barre.get("debords") or (0.0, 0.0))[:2]
    return g + sum(barre["travees"]) + d


def cote_gravite(s: Dict) -> float:
    """z_g du centre de GRAVITÉ, au-dessus du centre de cisaillement (mm) : −z_s d'un PRS monosymétrique, 0 sinon."""
    return -(s.get("z_s") or 0.0) if s.get("prs") else 0.0


def cote_charge(barre: Dict, s: Dict, ch: Dict) -> float:
    """La cote d'une charge (mm, AU-DESSUS du centre de cisaillement, section.cote) : son niveau — à défaut celui de la
    barre ; le poids propre (GRAVITE) au centre de gravité."""
    niv = ch.get("niveau") or barre["niveau"]
    return cote_gravite(s) if niv == GRAVITE else SE.cote(s, niv)


def cas_complets(barre: Dict, s: Dict) -> List[Dict]:
    """Les cas de charge de la barre, le poids propre en tête s'il est demandé (un cas permanent) : en flexion et dans N
    selon la position de la barre (``repartition``), appliqué au centre de gravité (``niveau`` GRAVITE)."""
    cas = [dict(c) for c in barre["cas"]]
    if barre.get("poids_propre", True):
        L = longueur(barre)
        g = poids_propre(s)
        fN, fM = repartition(barre.get("position"), barre.get("angle"))
        charges = [{"type": "q", "v": fM * g, "a": 0.0, "b": L, "niveau": GRAVITE}] if fM > 1e-12 else []
        cas.insert(0, {"nom": POIDS_PROPRE, "nature": "permanent", "N": fN * g * L, "N_troncons": None, "charges": charges})
    return cas


def N_du_cas(c: Dict):
    """L'effort normal d'un cas (kN, > 0 en compression) : un nombre (constant le long de la barre) ou, par tronçon,
    [(x2, N)…] — ``N_troncons`` remplace alors le N constant."""
    nt = c.get("N_troncons")
    if nt:
        return [(t["x2"], t["N"]) for t in nt]
    return c.get("N", 0.0)


def _N_en(n, x: float, droite: bool) -> float:
    """N d'un cas (``N_du_cas``) dans la section x, à droite (ou à gauche) d'un changement."""
    if not isinstance(n, list):
        return n
    for x2, v in n:
        if x < x2 - 1e-9 or (abs(x - x2) <= 1e-9 and not droite):
            return v
    return n[-1][1]


def cotes(e: Dict, eps: float = 1e-9) -> List[Tuple[float, float]]:
    """Les couples (M, V) de part et d'autre d'une section lue (``Combinaison.lire``) — un seul quand rien ne saute."""
    g, d = (e["M_g"], e["V_g"]), (e["M_d"], e["V_d"])
    return [d] if (abs(g[0] - d[0]) < eps and abs(g[1] - d[1]) < eps) else [g, d]


class Lecture:
    """Un cas résolu par le formulaire, dont les sections lues sont gardées (``evaluer``) ; le reste de la solution —
    travées, appuis, réactions — se lit comme sur elle."""

    def __init__(self, sol):
        self.sol = sol
        self.lus: Dict[float, Dict] = {}
        self.moments: Dict[float, float] = {}

    def evaluer(self, x: float) -> Dict:
        e = self.lus.get(x)
        if e is None:
            e = self.lus[x] = self.sol.evaluer(x)
        return e

    def moment(self, x: float) -> float:
        """M dans la section x — à droite ; à gauche au bout de la barre : la valeur de ``evaluer``, sans le reste
        (effort tranchant, rotation, flèche). Le moment critique lit le moment en une centaine de points par tronçon."""
        m = self.moments.get(x)
        if m is None:
            sol = self.sol
            xc = min(sol.L, max(0.0, x))
            droite = xc < sol.L
            t, xl = sol._travee(xc, droite)                 # la travée du formulaire : un appui reconnu à sa tolérance
            m = self.moments[x] = t.M(xl, droite)
        return m

    def __getattr__(self, nom: str):
        return getattr(self.sol, nom)


def resoudre(barre: Dict, cas: Sequence[Dict], EI: Optional[float] = None) -> List[Lecture]:
    """Une solution du formulaire par cas (``EI`` en kN·m² pour la flèche), débords compris."""
    out = []
    debords = tuple((barre.get("debords") or (0.0, 0.0))[:2])
    for c in cas:
        try:
            out.append(Lecture(PO.resoudre(barre["modele"], barre["travees"], c.get("charges") or [], EI, debords=debords)))
        except PO.ErreurDonnees as e:
            raise Refus(f"cas « {c['nom']} » : {e}") from None
    return out


class Combinaison:
    """Les efforts d'une combinaison le long de la barre (kN, m) : la somme pondérée des cas. ``N_cas`` : l'effort normal
    de chaque cas, constant (un nombre) ou par tronçon ([(x2, N)…], ``N_du_cas``). ``N`` : le PLUS GRAND effort normal
    de la combinaison le long de la barre — celui du flambement et de la flexion composée (prudent, À VÉRIFIER : la
    longueur de flambement n'est pas située) ; ``N_en(x, droite)`` : celui d'une section."""

    def __init__(self, nom: str, sols: Sequence, facteurs: Sequence[float], N_cas: Sequence):
        self.nom, self.facteurs = nom, list(facteurs)
        self.paires = [(f, s) for f, s in zip(facteurs, sols) if f]
        self.N_cas = [(f, n) for f, n in zip(facteurs, N_cas) if f]
        self.L = sols[0].L if sols else 0.0
        bornes = sorted({0.0} | {x2 for _, n in self.N_cas if isinstance(n, list) for x2, _ in n[:-1]})
        self.N = max((sum(f * _N_en(n, x, True) for f, n in self.N_cas) for x in bornes), default=0.0)
        self.variable = len(bornes) > 1          # N change le long de la barre
        self.lus: Dict[float, Dict] = {}

    def N_en(self, x: float, droite: bool = True) -> float:
        """L'effort normal de la combinaison dans la section x (kN), à droite ou à gauche d'un changement."""
        return sum(f * _N_en(n, x, droite) for f, n in self.N_cas)

    def lire(self, x: float) -> Dict:
        """M et V de part et d'autre de la section x, la flèche (None sans E·I). Le résultat est gardé : ne pas l'écrire."""
        o = self.lus.get(x)
        if o is not None:
            return o
        o = self.lus[x] = {"x": x, "M_g": 0.0, "M_d": 0.0, "V_g": 0.0, "V_d": 0.0, "f": 0.0}
        for f, s in self.paires:
            e = s.evaluer(x)
            for k in ("M_g", "M_d", "V_g", "V_d"):
                o[k] += f * e[k]
            o["f"] = None if (o["f"] is None or e["f"] is None) else o["f"] + f * e["f"]
        return o

    def M(self, x: float) -> float:
        """Le moment dans la section x (à droite ; à gauche au bout de la barre) — continu sauf sous un couple. C'est
        la valeur de ``lire`` (M_d, ou M_g au bout), lue sans l'effort tranchant ni la flèche."""
        return sum(f * s.moment(x) for f, s in self.paires)


def ruptures(sols: Sequence) -> List[float]:
    """Les abscisses globales où une expression change : appuis, débuts et fins de charge, charges ponctuelles."""
    pts = set()
    for s in sols:
        for t in s.travees:
            for x in t.ruptures():
                pts.add(round(t.x0 + x, 9))
            for m in t.morceaux:
                if m["type"] in ("P", "C"):
                    pts.add(round(t.x0 + m["a"], 9))
    return sorted(pts)


def _racines(y0: float, ym: float, y1: float, seuil: float) -> List[float]:
    """Les racines dans ]0 ; 1[ du polynôme de degré 2 au plus qui passe par (0, y0), (½, ym), (1, y1) ; aucune quand
    les trois valeurs sont sous ``seuil`` (un effort tranchant nul sur tout l'intervalle : le moment y est constant)."""
    ech = max(abs(y0), abs(ym), abs(y1))
    if ech <= seuil:
        return []
    A = 2.0 * (y0 + y1 - 2.0 * ym)
    B = y1 - y0 - A
    if abs(A) < 1e-12 * ech:
        r = [-y0 / B] if abs(B) > 1e-12 * ech else []
    else:
        d = B * B - 4.0 * A * y0
        if d < 0:
            return []
        q = -(B + math.copysign(math.sqrt(d), B)) / 2.0             # la forme stable
        r = [q / A] + ([y0 / q] if q != 0 else [])
    return sorted(t for t in r if 1e-12 < t < 1.0 - 1e-12)


def sections(barre: Dict, comb: Combinaison, pts: Sequence[float], racines: Optional[set] = None) -> List[float]:
    """Les sections à vérifier pour cette combinaison : ``pts`` (les ruptures, triées), les extremums du moment — les
    racines de l'effort tranchant entre deux ruptures —, les POINTS demandés (``barre["points"]``) et les RACINES du
    moment (brief § 10 : ses changements de signe), ajoutées aussi à ``racines`` quand il est donné (le dessin les lit)."""
    out = set(pts) | {round(x, 9) for x in barre.get("points") or []}
    lus = [comb.lire(x) for x in pts]
    seuil = NEGLIGEABLE * max((max(abs(e["V_g"]), abs(e["V_d"])) for e in lus), default=0.0)
    for u, v, eu, ev in zip(pts, pts[1:], lus, lus[1:]):
        if v - u <= 1e-9:
            continue
        for t in _racines(eu["V_d"], comb.lire(0.5 * (u + v))["V_d"], ev["V_g"], seuil):
            out.add(round(u + t * (v - u), 6))
    rac = racines_moment(comb, sorted(out))
    if racines is not None:
        racines.update(rac)
    return sorted(out | set(rac))


def racines_moment(comb: Combinaison, xs: Sequence[float]) -> List[float]:
    """Les abscisses où le moment de la combinaison change de signe, entre deux sections ``xs`` consécutives (triées) :
    entre deux ruptures et deux extremums, M est monotone — au plus une racine, trouvée par dichotomie (à 10⁻⁶ m)."""
    out = []
    ech = max((max(abs(comb.lire(x)["M_g"]), abs(comb.lire(x)["M_d"])) for x in xs), default=0.0)
    seuil = NEGLIGEABLE * ech
    for u, v in zip(xs, xs[1:]):
        if v - u <= 1e-9:
            continue
        mu, mv = comb.lire(u)["M_d"], comb.lire(v)["M_g"]
        if abs(mu) <= seuil or abs(mv) <= seuil or (mu > 0) == (mv > 0):
            continue
        a, b = u, v
        for _ in range(RACINES_ITERATIONS):
            m = 0.5 * (a + b)
            if (comb.M(m) > 0) == (mu > 0):
                a = m
            else:
                b = m
            if b - a < 1e-9:
                break
        out.append(round(0.5 * (a + b), 6))
    return out


def cotes_n(comb: Combinaison, x: float, eps: float = 1e-9) -> List[Tuple[float, float, float]]:
    """Les triplets (M, V, N) de part et d'autre de la section x — un seul quand rien ne saute (N compris)."""
    e = comb.lire(x)
    g = (e["M_g"], e["V_g"], comb.N_en(x, False))
    d = (e["M_d"], e["V_d"], comb.N_en(x, True))
    return [d] if all(abs(a - b) < eps for a, b in zip(g, d)) else [g, d]


def appuis_x(barre: Dict) -> List[float]:
    """Les abscisses des appuis (m), débord gauche compris : g, g + L₁, … ; la console : son encastrement en 0."""
    if barre["modele"] == "console":
        return [0.0]
    g = (barre.get("debords") or (0.0, 0.0))[0]
    xs, x = [round(g, 9)], g
    for Lt in barre["travees"]:
        x += Lt
        xs.append(round(x, 9))
    return xs


def torsion_appui(barre: Dict, i: int) -> str:
    """La condition de torsion de l'appui ``i`` pour M_cr : la fourche à défaut ; « encastre » quand ``appuis.detail`` le
    dit (brief § 4.1 : conditions de torsion aux appuis, fourche par défaut)."""
    detail = (barre.get("appuis") or {}).get("detail") or []
    d = detail[i] if i < len(detail) else None
    return MC.ENCASTRE if (d or {}).get("torsion") == "encastre" else MC.FOURCHE


EULER_ELEMENTS = 48              # éléments de poutre sur la barre (au moins EULER_MIN par morceau entre deux points tenus)
EULER_MIN = 6
EULER_ITERATIONS = 400           # itérations inverses au plus ; arrêt quand N_cr ne bouge plus de 10⁻¹² en valeur relative


def _cholesky_bande(K: List[List[float]], b: int) -> List[List[float]]:
    """La factorisation de Cholesky d'une matrice symétrique en BANDE (``K[i][j]`` = K(i, i + j), 0 ≤ j ≤ b) : le facteur
    L (``L[i][j]`` = L(i + j, i)), K = L·Lᵀ ; K doit être définie positive (la barre tenue, aucun mécanisme)."""
    n = len(K)
    L = [[0.0] * (b + 1) for _ in range(n)]
    for i in range(n):
        for j in range(i, min(n, i + b + 1)):
            v = K[i][j - i]
            for k in range(max(0, j - b), i):
                v -= L[k][i - k] * L[k][j - k]
            if j == i:
                if v <= 0.0:
                    raise Refus("flambement de la barre avec ses débords : la barre n'est pas tenue (mécanisme)")
                L[i][0] = math.sqrt(v)
            else:
                L[i][j - i] = v / L[i][0]
    return L


def _resoudre_bande(L: List[List[float]], b: int, y: List[float]) -> List[float]:
    """x tel que L·Lᵀ·x = y (``_cholesky_bande``)."""
    n = len(L)
    z = list(y)
    for i in range(n):
        v = z[i] - sum(L[k][i - k] * z[k] for k in range(max(0, i - b), i))
        z[i] = v / L[i][0]
    for i in reversed(range(n)):
        v = z[i] - sum(L[i][k - i] * z[k] for k in range(i + 1, min(n, i + b + 1)))
        z[i] = v / L[i][0]
    return z


def _produit_bande(K: List[List[float]], b: int, x: List[float]) -> List[float]:
    """K·x pour une matrice symétrique en bande (``_cholesky_bande``)."""
    n = len(K)
    y = [0.0] * n
    for i in range(n):
        for j in range(i, min(n, i + b + 1)):
            v = K[i][j - i]
            if v:
                y[i] += v * x[j]
                if j != i:
                    y[j] += v * x[i]
    return y


@functools.lru_cache(maxsize=64)
def longueur_euler(L: float, tenus: Tuple[float, ...]) -> float:
    """La longueur de flambement (m) d'une barre de longueur ``L`` (m), E·I et N constants, tenue en DÉPLACEMENT aux
    abscisses ``tenus`` (des articulations : la rotation y est libre), libre ailleurs — un bout qui n'est pas tenu est
    LIBRE (un débord, lot C). Le plus petit N_cr de (K − N·K_G)·φ = 0 : éléments de poutre d'Hermite (rigidité et
    rigidité géométrique cohérentes, E·I = 1), itération inverse (quotient de Rayleigh) ; L_cr = π/√N_cr. La méthode
    donne N_cr par excès (Rayleigh-Ritz) : l'écart est en h⁴, sous 10⁻⁶ ici — L_cr est arrondie au mm SUPÉRIEUR.
    Recoupée par l'équation caractéristique d'une travée articulée et de son débord (tests/test_barre_moteur.py)."""
    tt = [round(x, 9) for x in tenus if -1e-9 <= x <= L + 1e-9]
    pts = sorted({0.0, round(L, 9)} | set(tt))
    noeuds = [0.0]
    for a, c in zip(pts, pts[1:]):
        n = max(EULER_MIN, math.ceil(EULER_ELEMENTS * (c - a) / L))
        noeuds += [a + (c - a) * i / n for i in range(1, n + 1)]
    nd = 2 * len(noeuds)
    K = [[0.0] * 4 for _ in range(nd)]
    G = [[0.0] * 4 for _ in range(nd)]
    for e in range(len(noeuds) - 1):
        h = noeuds[e + 1] - noeuds[e]
        ke = [[12.0, 6 * h, -12.0, 6 * h], [6 * h, 4 * h * h, -6 * h, 2 * h * h], [-12.0, -6 * h, 12.0, -6 * h], [6 * h, 2 * h * h, -6 * h, 4 * h * h]]
        kg = [[36.0, 3 * h, -36.0, 3 * h], [3 * h, 4 * h * h, -3 * h, -h * h], [-36.0, -3 * h, 36.0, -3 * h], [3 * h, -h * h, -3 * h, 4 * h * h]]
        for p in range(4):
            for q in range(p, 4):
                K[2 * e + p][q - p] += ke[p][q] / h ** 3
                G[2 * e + p][q - p] += kg[p][q] / (30.0 * h)
    bloques = {2 * i for i, x in enumerate(noeuds) if any(abs(x - t) <= 1e-9 for t in tt)}
    if len(bloques) < 2:
        raise Refus("flambement de la barre avec ses débords : deux points tenus au moins")
    for c in bloques:                          # w = 0 : la ligne et la colonne effacées, 1 sur la diagonale de K
        for j in range(4):
            K[c][j] = G[c][j] = 0.0
            if c - j >= 0:
                K[c - j][j] = G[c - j][j] = 0.0
        K[c][0] = 1.0
    Lc = _cholesky_bande(K, 3)
    graine = 12345
    x = []
    for i in range(nd):                        # un départ pseudo-aléatoire, fixe : aucun mode n'y est orthogonal
        graine = (1103515245 * graine + 12345) % 2 ** 31
        x.append(0.0 if i in bloques else graine / 2 ** 31 - 0.5)
    lam = None
    for _ in range(EULER_ITERATIONS):
        y = _produit_bande(G, 3, x)
        z = _resoudre_bande(Lc, 3, y)
        zKz = sum(a * c for a, c in zip(z, y))          # zᵀ·K·z = zᵀ·y
        zGz = sum(a * c for a, c in zip(z, _produit_bande(G, 3, z)))
        nouveau = zKz / zGz
        norme = math.sqrt(sum(a * a for a in z))
        x = [a / norme for a in z]
        if lam is not None and abs(nouveau - lam) <= 1e-12 * lam:
            lam = nouveau
            break
        lam = nouveau
    return math.ceil(math.pi / math.sqrt(lam) * 1e3 - 1e-6) / 1e3


def troncons(barre: Dict, s: Dict, pour_mcr: bool = False) -> List[Dict]:
    """Les tronçons de déversement : [{"x1", "x2" (m), "appuis" (les deux bouts), "maintiens": [(x local en mm, cote z)]}].
    ``pour_mcr`` : les tronçons de M_cr — sans les maintiens « négligés dans M_cr » (``neglige``)."""
    L = longueur(barre)
    console = barre["modele"] == "console"
    g, d = (barre.get("debords") or (0.0, 0.0))[:2]
    tenus = {0.0: MC.ENCASTRE} if console else {x: torsion_appui(barre, i) for i, x in enumerate(appuis_x(barre))}
    interieurs = []
    for m in barre.get("maintiens") or []:
        if pour_mcr and m.get("neglige"):
            continue
        x, genre = PO.ramener(float(m["x"]), L), m.get("genre", "complet")
        if genre not in GENRES_MAINTIEN:
            raise Refus(f"maintien : genre parmi {', '.join(GENRES_MAINTIEN)} — reçu {genre!r}")
        if not 0.0 <= x <= L:
            raise Refus(f"maintien à {x:g} m : hors de la barre (0 à {L:g} m)")
        if genre == "complet":
            tenus.setdefault(round(x, 9), MC.FOURCHE)
        else:
            interieurs.append((x, genre))
    if console and round(L, 9) not in tenus:
        tenus[round(L, 9)] = MC.LIBRE
    if g > 1e-9 and 0.0 not in tenus:            # le bout d'un débord est libre (un maintien complet l'y tient)
        tenus[0.0] = MC.LIBRE
    if d > 1e-9 and round(L, 9) not in tenus:
        tenus[round(L, 9)] = MC.LIBRE
    bornes = sorted(tenus)
    cote = ((s.get("h") or 0.0) - (s.get("tf") or s.get("t") or 0.0)) / 2.0        # le plan moyen d'une semelle
    prs = bool(s.get("prs"))                     # un PRS : les plans moyens de SES semelles, au-dessus de SON centre de cisaillement
    out = []
    for a, b in zip(bornes, bornes[1:]):
        if b - a <= 1e-9:
            continue
        dedans = [((x - a) * 1e3, SE.cote(s, "semelle_" + genre) if prs else (cote if genre == "haut" else -cote))
                  for x, genre in interieurs if a + 1e-9 < x < b - 1e-9]
        out.append({"x1": a, "x2": b, "appuis": (tenus[a], tenus[b]), "maintiens": dedans})
    return out


def charges_du_troncon(cas: Sequence[Dict], facteurs: Sequence[float], tr: Dict, zg) -> Tuple[List, List]:
    """Les charges de la combinaison portées par le tronçon, en N et mm, abscisses locales, chacune à SA cote : (réparties
    [(x1, x2, q1, q2, zg)], ponctuelles [(x, P, zg)]). ``zg`` : une cote pour toutes (mm), ou ``zg(charge)`` — la cote de
    chaque charge, selon son niveau (lot C, C1 : une hauteur par charge). Les couples n'ont pas de hauteur d'application."""
    a, b = tr["x1"], tr["x2"]
    rep, pon = [], []
    cote = zg if callable(zg) else (lambda ch: zg)
    for f, c in zip(facteurs, cas):
        if not f:
            continue
        for ch in c.get("charges") or []:
            if ch["type"] == "P":
                if a - 1e-9 <= ch["a"] <= b + 1e-9:
                    pon.append(((ch["a"] - a) * 1e3, f * ch["v"] * 1e3, cote(ch)))
            elif ch["type"] in ("q", "tri"):
                ca, cb = ch.get("a", 0.0), ch.get("b", b)
                u, v = max(a, ca), min(b, cb)
                if v - u <= 1e-9:
                    continue
                if ch["type"] == "q":
                    q1 = q2 = ch["v"]
                else:
                    croissante = ch.get("sens", "croissante") == "croissante"
                    def q(x, ca=ca, cb=cb, v_=ch["v"], cr=croissante):
                        t = (x - ca) / (cb - ca)
                        return v_ * (t if cr else 1.0 - t)
                    q1, q2 = q(u), q(v)
                rep.append(((u - a) * 1e3, (v - a) * 1e3, f * q1, f * q2, cote(ch)))        # 1 kN/m = 1 N/mm
    return rep, pon

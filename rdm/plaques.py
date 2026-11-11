# -*- coding: utf-8 -*-
"""PLAQUES RECTANGULAIRES MINCES sous charge uniforme (docs/site/BIBLIOTHEQUE_AUTRES.md § 4) — bibliothèque standard.

Plaque a (selon x) × b (selon y), épaisseur t constante, matériau élastique isotrope, théorie de Kirchhoff-Love (petites
flèches) : D·∇⁴w = q, D = E·t³/(12·(1 − ν²)). Chaque bord est simplement appuyé (« A »), encastré (« E ») ou libre
(« L ») : x = 0 (gauche), x = a (droite), y = 0 (bas), y = b (haut).

MÉTHODES (Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2ᵉ éd.) :

  « levy »           deux bords OPPOSÉS simplement appuyés (§§ 30, 41-48) : w = Σ_m [p_m + Y_m(y)]·sin(mπx/a) ; p_m est la
                     solution de la bande, Y_m la solution homogène (quatre constantes par harmonique, fixées par les
                     conditions des deux autres bords : appuyé, encastré ou libre — système 4 × 4 exact) ;
  « superposition »  toute combinaison de bords appuyés et encastrés sans paire opposée appuyée (§§ 42-45, dont la plaque
                     encastrée sur ses quatre bords) : plaque appuyée sous q + moments répartis le long des bords
                     encastrés, développés en séries de sinus ; les rotations y sont annulées (système linéaire) ;
  « navier »         quatre bords appuyés (§ 28) : la double série de sinus, calculée EN PLUS de Lévy et comparée.

NON PRIS EN CHARGE : un bord libre dont les deux bords voisins ne sont pas tous deux simplement appuyés (console, coin
libre, trois bords encastrés et un libre…) — pas de solution de Lévy, et la superposition ci-dessus ne sait pas écrire
un bord libre. Le moteur le REFUSE (``NonPrisEnCharge``) ; il n'approche rien.

Unités : a, b en m ; t en mm ; q en kN/m² ; E en MPa ; w en m ; moments en kN·m/m ; contraintes en MPa.
Les fonctions de base de Y_m sont des exponentielles DÉCROISSANTES depuis chaque bord (e^{−αy}, αy·e^{−αy}, e^{−α(b−y)},
α(b−y)·e^{−α(b−y)}) : aucune ne déborde, quel que soit l'harmonique.
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

BORDS = ("A", "E", "L")
NOMS_BORD = {"A": "appui simple", "E": "encastrement", "L": "bord libre"}
COTES = ("x0", "xa", "y0", "yb")
K_LEVY = 30             # harmoniques impairs de la série de Lévy (m = 1, 3, …, 59)
N_SUPER = 36            # harmoniques par bord encastré dans la superposition (m = 1 … 36)
N_NAVIER = 79           # dernier harmonique (impair) de la double série de Navier


class ErreurDonnees(ValueError):
    """Une donnée hors du domaine : le message dit laquelle."""


class NonPrisEnCharge(ValueError):
    """La combinaison de bords n'a pas de solution classique dans le périmètre : le message dit pourquoi."""


# ================================================================================================ une harmonique
def _base(al: float, L: float, y: float):
    """Les quatre fonctions de base en y et leurs dérivées première, seconde et troisième."""
    u, v = al * y, al * (L - y)
    eu, ev = math.exp(-u), math.exp(-v)
    a2, a3 = al * al, al ** 3
    return ((eu, u * eu, ev, v * ev),
            (-al * eu, al * (1 - u) * eu, al * ev, -al * (1 - v) * ev),
            (a2 * eu, a2 * (u - 2) * eu, a2 * ev, a2 * (v - 2) * ev),
            (-a3 * eu, a3 * (3 - u) * eu, a3 * ev, -a3 * (3 - v) * ev))


def _resoudre(A: List[List[float]], b: List[float]) -> List[float]:
    """Élimination de Gauss à pivot partiel (les lignes sont d'abord ramenées à leur plus grand terme)."""
    n = len(b)
    M = []
    for i in range(n):
        s = max(abs(x) for x in A[i]) or 1.0
        M.append([x / s for x in A[i]] + [b[i] / s])
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-300:
            raise ArithmeticError("système singulier")
        M[k], M[p] = M[p], M[k]
        piv = M[k][k]
        for i in range(k + 1, n):
            f = M[i][k] / piv
            if f:
                Mi, Mk = M[i], M[k]
                for j in range(k, n + 1):
                    Mi[j] -= f * Mk[j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _lignes_bord(typ: str, al: float, L: float, y: float, p: float, nu: float):
    """Les deux conditions d'un bord (en y) pour l'harmonique α : [(coefficients des quatre constantes, second membre)].
    ``p`` : la solution particulière (constante en y) ajoutée à Y."""
    f, d1, d2, d3 = _base(al, L, y)
    a2 = al * al
    if typ == "A":                              # w = 0 ; M = 0 → Y″ = 0
        return [(f, -p), (d2, 0.0)]
    if typ == "E":                              # w = 0 ; rotation nulle
        return [(f, -p), (d1, 0.0)]
    # libre : M = 0 → Y″ − ν·α²·(Y + p) = 0 ; effort tranchant de Kirchhoff nul → Y‴ − (2 − ν)·α²·Y′ = 0
    return [([d2[k] - nu * a2 * f[k] for k in range(4)], nu * a2 * p), ([d3[k] - (2 - nu) * a2 * d1[k] for k in range(4)], 0.0)]


class Famille:
    """Une série Σ_m sin(α_m·s)·[p_m + Y_m(r)] : s court le long des bords appuyés (longueur Ls), r en travers (Lr).
    ``termes`` : [(α, p, (c1, c2, c3, c4))]. ``dans_x`` : s = x et r = y ; sinon s = y et r = x."""

    def __init__(self, Ls: float, Lr: float, dans_x: bool):
        self.Ls, self.Lr, self.dans_x = Ls, Lr, dans_x
        self.termes: List[Tuple[float, float, Sequence[float]]] = []
        self.bande = False            # la solution particulière est sommée sous forme fermée (bande sous q)

    def derivees(self, x: float, y: float) -> Dict[str, float]:
        """w et ses dérivées utiles en (x, y), en unités q/D = 1 : w, ws, wr, wss, wrr, wsr (s le long, r en travers),
        et les dérivées troisièmes wrrr, wssr (effort tranchant d'un bord libre)."""
        s, r = (x, y) if self.dans_x else (y, x)
        o = dict.fromkeys(("w", "ws", "wr", "wss", "wrr", "wsr", "wrrr", "wssr"), 0.0)
        for al, p, c in self.termes:
            f, d1, d2, d3 = _base(al, self.Lr, r)
            Y = sum(c[k] * f[k] for k in range(4))
            Y1 = sum(c[k] * d1[k] for k in range(4))
            Y2 = sum(c[k] * d2[k] for k in range(4))
            Y3 = sum(c[k] * d3[k] for k in range(4))
            si, co = math.sin(al * s), math.cos(al * s)
            pp = 0.0 if self.bande else p
            o["w"] += (pp + Y) * si
            o["ws"] += al * (pp + Y) * co
            o["wr"] += Y1 * si
            o["wss"] -= al * al * (pp + Y) * si
            o["wrr"] += Y2 * si
            o["wsr"] += al * Y1 * co
            o["wrrr"] += Y3 * si
            o["wssr"] -= al * al * Y1 * si
        if self.bande:                         # Σ p_m·sin = (s⁴ − 2·Ls·s³ + Ls³·s)/24 : la bande sur deux appuis, exacte
            Ls = self.Ls
            o["w"] += (s ** 4 - 2 * Ls * s ** 3 + Ls ** 3 * s) / 24.0
            o["ws"] += (4 * s ** 3 - 6 * Ls * s * s + Ls ** 3) / 24.0
            o["wss"] += -(s * (Ls - s)) / 2.0
        return o


def _levy(Ls: float, Lr: float, c0: str, c1: str, nu: float, dans_x: bool, K: int = K_LEVY) -> Famille:
    """La solution de Lévy sous charge uniforme (q/D = 1) : bords s = 0 et s = Ls appuyés ; r = 0 de type c0, r = Lr de
    type c1. Harmoniques impairs m = 1 … 2K − 1 (une charge uniforme n'excite que ceux-là)."""
    F = Famille(Ls, Lr, dans_x)
    F.bande = True
    for i in range(K):
        mm = 2 * i + 1
        al = mm * math.pi / Ls
        p = 4.0 * Ls ** 4 / (math.pi ** 5 * mm ** 5)
        lig = _lignes_bord(c0, al, Lr, 0.0, p, nu) + _lignes_bord(c1, al, Lr, Lr, p, nu)
        c = _resoudre([list(li[0]) for li in lig], [li[1] for li in lig])
        F.termes.append((al, p, c))
    return F


def _unitaires(al: float, Lr: float) -> Tuple[List[float], List[float]]:
    """Les deux solutions homogènes à w nul sur les deux bords : moment unité en r = 0 (−Y″(0) = 1) puis en r = Lr."""
    f0, _, d20, _ = _base(al, Lr, 0.0)
    f1, _, d21, _ = _base(al, Lr, Lr)
    A = [list(f0), list(d20), list(f1), list(d21)]
    return _resoudre([r[:] for r in A], [0.0, -1.0, 0.0, 0.0]), _resoudre([r[:] for r in A], [0.0, 0.0, 0.0, -1.0])


# ================================================================================================ les solutions
class Solution:
    """Une plaque résolue : ``champ(x, y)`` rend w (m), M_x, M_y, M_xy (kN·m/m) ; ``brut(x, y)`` les dérivées en q/D = 1."""

    def __init__(self, a: float, b: float, bords: Dict[str, str], nu: float, methode: str, familles: List[Famille]):
        self.a, self.b, self.bords, self.nu, self.methode, self.familles = a, b, bords, nu, methode, familles
        self.q = self.D = 1.0
        self.moments_bord: Dict[str, List[float]] = {}          # superposition : les coefficients e_m des bords encastrés

    def brut(self, x: float, y: float) -> Dict[str, float]:
        """w, w_x, w_y, w_xx, w_yy, w_xy, et les dérivées troisièmes w_yyy, w_xxy, w_xxx, w_xyy — unités q/D = 1."""
        o = dict.fromkeys(("w", "wx", "wy", "wxx", "wyy", "wxy", "wyyy", "wxxy", "wxxx", "wxyy"), 0.0)
        for F in self.familles:
            d = F.derivees(x, y)
            o["w"] += d["w"]
            o["wxy"] += d["wsr"]
            if F.dans_x:
                o["wx"] += d["ws"]; o["wy"] += d["wr"]; o["wxx"] += d["wss"]; o["wyy"] += d["wrr"]
                o["wyyy"] += d["wrrr"]; o["wxxy"] += d["wssr"]
            else:
                o["wy"] += d["ws"]; o["wx"] += d["wr"]; o["wyy"] += d["wss"]; o["wxx"] += d["wrr"]
                o["wxxx"] += d["wrrr"]; o["wxyy"] += d["wssr"]
        return o

    def champ(self, x: float, y: float) -> Dict[str, float]:
        d = self.brut(x, y)
        k = self.q / self.D
        nu = self.nu
        return {"w": k * d["w"], "Mx": -self.q * (d["wxx"] + nu * d["wyy"]), "My": -self.q * (d["wyy"] + nu * d["wxx"]),
                "Mxy": -self.q * (1 - nu) * d["wxy"]}


def classer(bords: Dict[str, str]) -> str:
    """La méthode qui convient à ces bords, ou NonPrisEnCharge avec la raison."""
    for c in COTES:
        if bords.get(c) not in BORDS:
            raise ErreurDonnees(f"bord {c} : condition attendue parmi {BORDS}")
    if all(bords[c] == "A" for c in COTES):
        return "levy"
    if bords["x0"] == bords["xa"] == "A" or bords["y0"] == bords["yb"] == "A":
        return "levy"
    if "L" not in bords.values():
        return "superposition"
    libres = [c for c in COTES if bords[c] == "L"]
    noms = {"x0": "gauche", "xa": "droit", "y0": "bas", "yb": "haut"}
    raise NonPrisEnCharge(
        "le bord " + " et le bord ".join(noms[c] for c in libres) + (" est libre" if len(libres) == 1 else " sont libres")
        + " sans que ses deux bords voisins soient simplement appuyés : il n'existe pas de solution de Lévy pour cette plaque, "
        "et la superposition de Timoshenko (moments de bord) ne sait pas écrire un bord libre")


def _resoudre_plaque(a: float, b: float, bords: Dict[str, str], nu: float, K: int = K_LEVY, N: int = N_SUPER,
                     forcer: Optional[str] = None) -> Solution:
    """La plaque résolue en unités q = D = 1. ``forcer="superposition"`` impose la superposition à une plaque de Lévy
    sans bord libre : les deux méthodes se recouvrent là, et les tests les comparent."""
    methode = classer(bords)
    if forcer == "superposition" and "L" not in bords.values():
        methode = "superposition"
    if methode == "levy":
        if bords["x0"] == bords["xa"] == "A":
            return Solution(a, b, bords, nu, "levy", [_levy(a, b, bords["y0"], bords["yb"], nu, True, K)])
        return Solution(a, b, bords, nu, "levy", [_levy(b, a, bords["x0"], bords["xa"], nu, False, K)])
    # ---- superposition : plaque appuyée + moments le long des bords encastrés, rotations nulles
    L1 = _levy(a, b, "A", "A", nu, True, K)                # la plaque appuyée, série en x
    L2 = _levy(b, a, "A", "A", nu, False, K)               # la même, série en y : ses rotations le long de x = 0 et x = a
    al = [m * math.pi / a for m in range(1, N + 1)]
    be = [n * math.pi / b for n in range(1, N + 1)]
    Uy = [_unitaires(x, b) for x in al]                    # familles en x (fonctions de y) : moments en y = 0, y = b
    Ux = [_unitaires(x, a) for x in be]                    # familles en y (fonctions de x) : moments en x = 0, x = a

    def pente(c: Sequence[float], alpha: float, Lr: float, r: float) -> float:
        return sum(c[k] * _base(alpha, Lr, r)[1][k] for k in range(4))

    def pente_charge(F: Famille, harmonique: int, r: float) -> float:
        if harmonique % 2 == 0:
            return 0.0
        i = (harmonique - 1) // 2
        if i >= len(F.termes):
            return 0.0
        alpha, _, c = F.termes[i]
        return pente(c, alpha, F.Lr, r)

    enc = [c for c in ("y0", "yb", "x0", "xa") if bords[c] == "E"]
    idx = {(c, k): i * N + k for i, c in enumerate(enc) for k in range(N)}
    nt = len(enc) * N
    A = [[0.0] * nt for _ in range(nt)]
    B = [0.0] * nt
    for c in enc:
        for k in range(N):
            li = idx[(c, k)]
            h = k + 1
            if c in ("y0", "yb"):
                r = 0.0 if c == "y0" else b
                B[li] = -pente_charge(L1, h, r)
                for cc, j in (("y0", 0), ("yb", 1)):
                    if (cc, k) in idx:
                        A[li][idx[(cc, k)]] += pente(Uy[k][j], al[k], b, r)
                for n in range(N):
                    sg = 1.0 if c == "y0" else (-1.0) ** (n + 1)
                    g = sg * be[n] * (2.0 / a) * al[k] / (al[k] ** 2 + be[n] ** 2) ** 2
                    if ("x0", n) in idx:
                        A[li][idx[("x0", n)]] += g
                    if ("xa", n) in idx:
                        A[li][idx[("xa", n)]] -= g * (-1.0) ** h
            else:
                r = 0.0 if c == "x0" else a
                B[li] = -pente_charge(L2, h, r)
                for cc, j in (("x0", 0), ("xa", 1)):
                    if (cc, k) in idx:
                        A[li][idx[(cc, k)]] += pente(Ux[k][j], be[k], a, r)
                for mth in range(N):
                    sg = 1.0 if c == "x0" else (-1.0) ** (mth + 1)
                    g = sg * al[mth] * (2.0 / b) * be[k] / (al[mth] ** 2 + be[k] ** 2) ** 2
                    if ("y0", mth) in idx:
                        A[li][idx[("y0", mth)]] += g
                    if ("yb", mth) in idx:
                        A[li][idx[("yb", mth)]] -= g * (-1.0) ** h
    sol = _resoudre(A, B)
    Fx, Fy = Famille(a, b, True), Famille(b, a, False)
    for k in range(N):
        e0 = sol[idx[("y0", k)]] if ("y0", k) in idx else 0.0
        e1 = sol[idx[("yb", k)]] if ("yb", k) in idx else 0.0
        if e0 or e1:
            Fx.termes.append((al[k], 0.0, [e0 * Uy[k][0][j] + e1 * Uy[k][1][j] for j in range(4)]))
        f0 = sol[idx[("x0", k)]] if ("x0", k) in idx else 0.0
        f1 = sol[idx[("xa", k)]] if ("xa", k) in idx else 0.0
        if f0 or f1:
            Fy.termes.append((be[k], 0.0, [f0 * Ux[k][0][j] + f1 * Ux[k][1][j] for j in range(4)]))
    S = Solution(a, b, bords, nu, "superposition", [L1, Fx, Fy])
    S.moments_bord = {c: [sol[idx[(c, k)]] for k in range(N)] for c in enc}
    return S


def navier_centre(a: float, b: float, nu: float, N: int = N_NAVIER) -> Dict[str, float]:
    """La plaque appuyée par la double série de Navier, au centre, en unités q/D = 1 (w) et q = 1 (moments) :
    w = Σ Σ 16/(π⁶·m·n·(m²/a² + n²/b²)²)·sin·sin."""
    w = mx = my = 0.0
    for m in range(1, N + 1, 2):
        for n in range(1, N + 1, 2):
            k = (m * m / (a * a) + n * n / (b * b))
            c = 16.0 / (math.pi ** 6 * m * n * k * k) * (-1) ** ((m - 1) // 2 + (n - 1) // 2)
            w += c
            mx += c * math.pi ** 2 * (m * m / (a * a) + nu * n * n / (b * b))
            my += c * math.pi ** 2 * (n * n / (b * b) + nu * m * m / (a * a))
    return {"w": w, "Mx": mx, "My": my}


# ================================================================================================ recherche d'extremums
def _or(f: Callable[[float], float], u: float, v: float, it: int = 40) -> float:
    """Le maximum de f sur [u ; v] par la section dorée (f unimodale près de son maximum)."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = v - g * (v - u), u + g * (v - u)
    fc, fd = f(c), f(d)
    for _ in range(it):
        if fc > fd:
            v, d, fd = d, c, fc
            c = v - g * (v - u)
            fc = f(c)
        else:
            u, c, fc = c, d, fd
            d = u + g * (v - u)
            fd = f(d)
    return (u + v) / 2.0


def _maximum(f: Callable[[float, float], float], a: float, b: float, grille: List[List[float]]) -> Tuple[float, float, float]:
    """(valeur, x, y) du maximum de f sur la plaque : le plus grand point de la GRILLE déjà calculée (``grille[j][i]`` =
    f(a·i/n, b·j/n)), affiné par la section dorée en x puis en y, deux tours."""
    n = len(grille) - 1
    _, i, j = max((grille[j][i], i, j) for j in range(n + 1) for i in range(n + 1))
    x, y = a * i / n, b * j / n
    hx, hy = a / n, b / n
    for _ in range(2):
        x = _or(lambda s: f(s, y), max(0.0, x - hx), min(a, x + hx), 22)
        y = _or(lambda s: f(x, s), max(0.0, y - hy), min(b, y + hy), 22)
    return f(x, y), x, y


# ================================================================================================ la porte
def rigidite(E: float, t: float, nu: float) -> float:
    """D = E·t³/(12·(1 − ν²)) en kN·m (E en MPa, t en mm)."""
    return E * t ** 3 / (12.0 * (1.0 - nu * nu)) * 1e-6


def calculer(a: float, b: float, bords: Dict[str, str], q: float, E: float, nu: float, t: float, grille: int = 20,
             K: int = K_LEVY, N: int = N_SUPER) -> Dict:
    """La plaque calculée : flèche maximale, moments au centre, aux bords encastrés et libres, contraintes, la grille
    des flèches (lignes de niveau), les coefficients α et β (w = α·q·L⁴/D, M = β·q·L², L = min(a ; b)) et les contrôles
    (convergence, Navier pour la plaque appuyée)."""
    for nom, v, lo, hi in (("a", a, 0.01, 1000.0), ("b", b, 0.01, 1000.0), ("t", t, 0.01, 5000.0), ("E", E, 1.0, 1e7)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ErreurDonnees(f"{nom} hors du domaine ({lo:g} à {hi:g})")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q):
        raise ErreurDonnees("q : nombre attendu")
    if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not 0.0 <= nu < 0.5:
        raise ErreurDonnees("ν hors du domaine (0 à 0,5 exclu)")
    if not 0.1 <= a / b <= 10.0:
        raise ErreurDonnees("rapport des côtés hors du domaine (0,1 à 10) : au-delà, la plaque porte dans un seul sens — une poutre")
    S = _resoudre_plaque(a, b, bords, nu, K, N)
    S.q, S.D = q, rigidite(E, t, nu)
    Lm = min(a, b)
    sg = 1.0 if q >= 0 else -1.0
    out: Dict = {"a": a, "b": b, "bords": dict(bords), "q": q, "E": E, "nu": nu, "t": t, "D": S.D, "L": Lm, "methode": S.methode, "solution": S}
    kq = abs(q) or 1.0
    # ---- flèche maximale (du côté de la charge), moments au centre
    G = [[S.champ(a * i / grille, b * j / grille) for i in range(grille + 1)] for j in range(grille + 1)]      # une fois, pour tout
    wm, xw, yw = _maximum(lambda x, y: sg * S.champ(x, y)["w"], a, b, [[sg * g["w"] for g in li] for li in G])
    out["w_max"] = {"valeur": sg * wm, "x": xw, "y": yw, "alpha": wm * S.D / (kq * Lm ** 4) if q else 0.0}
    c = S.champ(a / 2, b / 2)
    out["centre"] = {"w": c["w"], "Mx": c["Mx"], "My": c["My"], "beta_x": sg * c["Mx"] / (kq * Lm ** 2), "beta_y": sg * c["My"] / (kq * Lm ** 2)}
    # ---- moments maximaux en travée (positifs du côté de la charge)
    mx = _maximum(lambda x, y: sg * S.champ(x, y)["Mx"], a, b, [[sg * g["Mx"] for g in li] for li in G])
    my = _maximum(lambda x, y: sg * S.champ(x, y)["My"], a, b, [[sg * g["My"] for g in li] for li in G])
    out["Mx_max"] = {"valeur": sg * mx[0], "x": mx[1], "y": mx[2], "beta": mx[0] / (kq * Lm ** 2)}
    out["My_max"] = {"valeur": sg * my[0], "x": my[1], "y": my[2], "beta": my[0] / (kq * Lm ** 2)}
    # ---- bords : moment d'encastrement le plus fort le long de chaque bord encastré ; flèche et moment d'un bord libre
    out["encastrements"], out["libres"] = [], []
    for cote in COTES:
        le_long_x = cote in ("y0", "yb")
        Lb = a if le_long_x else b
        fixe = {"x0": 0.0, "xa": a, "y0": 0.0, "yb": b}[cote]
        pt = (lambda s: (s, fixe)) if le_long_x else (lambda s: (fixe, s))
        comp = "My" if le_long_x else "Mx"                # le moment NORMAL au bord
        tang = "Mx" if le_long_x else "My"                # le moment le long du bord
        if bords[cote] == "E":
            s = max((Lb * i / 32 for i in range(33)), key=lambda u: -sg * S.champ(*pt(u))[comp])
            s = _or(lambda u: -sg * S.champ(*pt(u))[comp], max(0.0, s - Lb / 32), min(Lb, s + Lb / 32), 30)
            v = S.champ(*pt(s))[comp]
            out["encastrements"].append({"cote": cote, "composante": comp, "valeur": v, "s": s, "x": pt(s)[0], "y": pt(s)[1],
                                         "beta": sg * v / (kq * Lm ** 2), "milieu": S.champ(*pt(Lb / 2))[comp]})
        elif bords[cote] == "L":
            s = max((Lb * i / 32 for i in range(33)), key=lambda u: sg * S.champ(*pt(u))["w"])
            s = _or(lambda u: sg * S.champ(*pt(u))["w"], max(0.0, s - Lb / 32), min(Lb, s + Lb / 32), 30)
            ch = S.champ(*pt(s))
            out["libres"].append({"cote": cote, "composante": tang, "w": ch["w"], "M": ch[tang], "s": s, "x": pt(s)[0], "y": pt(s)[1],
                                  "alpha": sg * ch["w"] * S.D / (kq * Lm ** 4) if q else 0.0, "beta": sg * ch[tang] / (kq * Lm ** 2)})
    # ---- contrainte maximale σ = 6·|M|/t²
    cand = [abs(out["Mx_max"]["valeur"]), abs(out["My_max"]["valeur"])] + [abs(e["valeur"]) for e in out["encastrements"]] + [abs(e["M"]) for e in out["libres"]]
    out["M_abs_max"] = max(cand)
    out["sigma_max"] = 6.0 * out["M_abs_max"] * 1000.0 / (t * t)
    # ---- la grille des flèches (lignes de niveau)
    out["grille"] = [[g["w"] for g in li] for li in G]
    # ---- contrôles : convergence (la moitié des termes), Navier pour la plaque appuyée
    S2 = _resoudre_plaque(a, b, bords, nu, max(4, K // 2), max(4, N // 2))
    S2.q, S2.D = S.q, S.D
    c2 = S2.champ(xw, yw)
    cm2 = S2.champ(a / 2, b / 2)
    ech_M = out["M_abs_max"] or 1.0
    out["controles"] = {"convergence_w": abs(c2["w"] - sg * wm) / (abs(wm) or 1.0),
                        "convergence_M": max(abs(cm2["Mx"] - c["Mx"]), abs(cm2["My"] - c["My"])) / ech_M}
    if all(bords[k] == "A" for k in COTES):
        nv = navier_centre(a, b, nu)
        out["controles"]["navier"] = {"w": nv["w"] * q / S.D, "Mx": nv["Mx"] * q, "My": nv["My"] * q,
                                      "ecart_w": abs(nv["w"] * q / S.D - c["w"]) / (abs(c["w"]) or 1.0),
                                      "ecart_M": max(abs(nv["Mx"] * q - c["Mx"]), abs(nv["My"] * q - c["My"])) / ech_M}
    out["avertissements"] = []
    if t > Lm * 1000.0 / 10.0:
        out["avertissements"].append("epaisse")
    if abs(out["w_max"]["valeur"]) * 1000.0 > t / 2.0:
        out["avertissements"].append("grande_fleche")
    return out

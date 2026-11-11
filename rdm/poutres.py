# -*- coding: utf-8 -*-
"""FORMULAIRE DE POUTRES DROITES (docs/site/BIBLIOTHEQUE_AUTRES.md § 3) — bibliothèque standard seule.

Modèles : poutre sur deux appuis, console, encastrée–appuyée, bi-encastrée, poutre continue de 2 à 5 travées (rigidité
E·I constante). Charges : répartie uniforme (totale ou partielle), triangulaire, ponctuelle, couple ; elles se
superposent (comportement linéaire).

CONVENTIONS — x depuis l'extrémité gauche ; une charge vers le bas est positive ; V(x) = somme des forces à gauche
(réaction vers le haut positive), V = dM/dx ; M(x) positif quand il tend la fibre inférieure ; flèche f positive vers
le bas, E·I·f″ = −M ; rotation θ = f′ ; un couple C est positif dans le sens horaire.

MÉTHODE — uniquement des écritures fermées, aucune intégration numérique :

  a) chaque travée, rendue isostatique, s'écrit par la méthode des PARAMÈTRES INITIAUX (Clebsch ; Macaulay) :
         V(x) = R_A − Q₁(x)                M(x) = M_A + R_A·x − Q₂(x)
         E·I·θ(x) = E·I·ω₀ − M_A·x − R_A·x²/2 + Q₃(x)
         E·I·f(x) = E·I·ω₀·x − M_A·x²/2 − R_A·x³/6 + Q₄(x)
     où Q_j additionne, charge par charge, les termes ⟨x − a⟩ⁿ (nuls à gauche de la charge) ;
  b) les rotations d'appui de la travée isostatique viennent du FORMULAIRE (q·L³/24, P·a·b·(L + b)/(6·L)… :
     ``rotations_formulaire``) — et, séparément, de la condition f(L) = 0 de l'écriture a) : les deux calculs sont
     comparés à chaque résolution (``controles``), un écart arrête le calcul ;
  c) les liaisons hyperstatiques s'écrivent par l'égalité des rotations : M_A = −3·E·I·ω_A/L (encastrée–appuyée),
     M_A = −2·(2·E·I·ω_A − E·I·ω_B)/L (bi-encastrée), théorème des TROIS MOMENTS de Clapeyron (poutre continue).

Les efforts ne dépendent pas de E·I (rigidité constante) : ``EI`` n'est demandé que pour la flèche ; sans lui le moteur
rend E·I·f (kN·m³). Unités : m, kN, kN/m, kN·m, E·I en kN·m².
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

MODELES = ("appuis", "console", "enc_app", "bi_enc", "continue")
TYPES_CHARGE = ("q", "tri", "P", "C")
SENS = ("croissante", "decroissante")
TRAVEES_MAX = 5
CHARGES_MAX = 8
TOL = 1e-9          # tolérance relative des contrôles internes (équilibre, compatibilité, formulaire = paramètres initiaux)


class ErreurDonnees(ValueError):
    """Une donnée hors du domaine (portée nulle, charge hors de la poutre, modèle inconnu) : le message dit laquelle."""


class ErreurControle(ArithmeticError):
    """Un contrôle interne a échoué (deux calculs indépendants ne s'accordent pas) : le résultat n'est pas rendu."""


# ================================================================================================ les termes de Macaulay
def _puiss(x: float, a: float, n: int, droite: bool = True) -> float:
    """⟨x − a⟩ⁿ : (x − a)ⁿ à droite de a, 0 à gauche. Pour n = 0 (un saut), ``droite`` dit de quel côté de a on lit."""
    d = x - a
    if n == 0:
        return 1.0 if (d > 0 or (droite and d >= 0)) else 0.0
    return d ** n if d > 0 else 0.0


def _Q(termes: Sequence[Tuple], j: int, x: float, droite: bool = True) -> float:
    """Q_j(x) : la j-ième primitive du chargement (j = 0 densité, 1 tranchant, 2 moment, 3 rotation, 4 flèche).
    Un terme est (genre, a, w) : genre 0 = densité w·⟨x − a⟩⁰, 1 = densité w·⟨x − a⟩¹, « P » = force, « C » = couple."""
    s = 0.0
    for g, a, w in termes:
        if g == "P":
            if j >= 1:
                s += w * _puiss(x, a, j - 1, droite) / math.factorial(j - 1)
        elif g == "C":
            if j >= 2:
                s -= w * _puiss(x, a, j - 2, droite) / math.factorial(j - 2)
        else:
            s += w * _puiss(x, a, g + j, droite) * math.factorial(g) / math.factorial(g + j)
    return s


# ================================================================================================ le formulaire
def _F(x: float, L: float) -> float:
    return L * L * x * x - L * x ** 3 + x ** 4 / 4.0


def _F1(x: float, L: float) -> float:
    return 2.0 * L * L * x ** 3 / 3.0 - 3.0 * L * x ** 4 / 4.0 + x ** 5 / 5.0


def _G(x: float, L: float) -> float:
    return L * L * x * x / 2.0 - x ** 4 / 4.0


def _G1(x: float, L: float) -> float:
    return L * L * x ** 3 / 3.0 - x ** 5 / 5.0


def rotations_formulaire(m: Dict, L: float) -> Tuple[float, float, float, float]:
    """(R_A, R_B, E·I·ω_A, E·I·ω_B) d'une charge sur la travée ISOSTATIQUE de portée L — les formules du formulaire
    (Techniques de l'ingénieur C 2 060 ; Roark, tableau 8.1, cas 1e, 2e, 3e), écrites ici telles qu'on les lit :

        P en a (b = L − a)   R_A = P·b/L        E·I·ω_A = P·a·b·(L + b)/(6·L)     E·I·ω_B = P·a·b·(L + a)/(6·L)
        C en a               R_A = −C/L         E·I·ω_A = C·(3·b² − L²)/(6·L)     E·I·ω_B = C·(L² − 3·a²)/(6·L)
        q de a à b           R_A = W·(L − x_c)/L        E·I·ω_A = q·[F(b) − F(a)]/(6·L)   E·I·ω_B = q·[G(b) − G(a)]/(6·L)
        triangle 0 → q       mêmes sommes avec F₁, G₁ (k = q/(b − a)) :
                             E·I·ω_A = k·{[F₁(b) − F₁(a)] − a·[F(b) − F(a)]}/(6·L)

    F(ξ) = L²ξ² − Lξ³ + ξ⁴/4, G(ξ) = L²ξ²/2 − ξ⁴/4, F₁(ξ) = 2L²ξ³/3 − 3Lξ⁴/4 + ξ⁵/5, G₁(ξ) = L²ξ³/3 − ξ⁵/5 : la charge
    répartie est la somme de charges ponctuelles élémentaires. ω_A = f′(0), ω_B = −f′(L), positives pour une charge
    vers le bas. ``m`` est un morceau de charge LOCAL (``morceaux``)."""
    t = m["type"]
    if t == "P":
        P, a = m["v"], m["a"]
        b = L - a
        return P * b / L, P * a / L, P * a * b * (L + b) / (6 * L), P * a * b * (L + a) / (6 * L)
    if t == "C":
        Cc, a = m["v"], m["a"]
        b = L - a
        return -Cc / L, Cc / L, Cc * (3 * b * b - L * L) / (6 * L), Cc * (L * L - 3 * a * a) / (6 * L)
    a, b, q1, q2 = m["a"], m["b"], m["q1"], m["q2"]
    k = (q2 - q1) / (b - a)
    W = (q1 + q2) / 2.0 * (b - a)
    Mo = (b - a) * (q1 * (2 * a + b) + q2 * (a + 2 * b)) / 6.0          # moment de la charge par rapport à A
    dF, dG = _F(b, L) - _F(a, L), _G(b, L) - _G(a, L)
    wA = (q1 * dF + k * ((_F1(b, L) - _F1(a, L)) - a * dF)) / (6 * L)
    wB = (q1 * dG + k * ((_G1(b, L) - _G1(a, L)) - a * dG)) / (6 * L)
    return W - Mo / L, Mo / L, wA, wB


# ================================================================================================ une travée
class Travee:
    """Une travée de portée L (abscisse locale x de 0 à L), ses morceaux de charge, ses paramètres initiaux."""

    def __init__(self, L: float, x0: float, nom_A: str, nom_B: str):
        self.L, self.x0, self.nom_A, self.nom_B = L, x0, nom_A, nom_B
        self.morceaux: List[Dict] = []
        self.termes: List[Tuple] = []
        self.R_A0 = self.wA0 = self.wB0 = 0.0          # travée isostatique : réaction en A, E·I·ω_A, E·I·ω_B
        self.M_A = self.M_B = 0.0                      # moments aux extrémités (négatifs : fibre supérieure tendue)
        self.R_A = self.w0 = 0.0                       # paramètres initiaux : tranchant et E·I·rotation en x = 0

    # -- les charges
    def ajouter(self, m: Dict) -> None:
        self.morceaux.append(m)
        L = self.L
        if m["type"] in ("P", "C"):
            self.termes.append((m["type"], m["a"], m["v"]))
            return
        a, b, q1, q2 = m["a"], m["b"], m["q1"], m["q2"]
        k = (q2 - q1) / (b - a)
        self.termes += [(0, a, q1), (1, a, k)]
        if b < L:                                      # au-delà de la travée, un terme ne joue plus
            self.termes += [(0, b, -q2), (1, b, -k)]

    def charge_totale(self) -> float:
        return _Q(self.termes, 1, self.L, True)

    # -- la travée isostatique (paramètres initiaux, f(0) = f(L) = 0, M(0) = M(L) = 0)
    def isostatique(self) -> None:
        L = self.L
        self.R_A0 = _Q(self.termes, 2, L, True) / L
        self.wA0 = (self.R_A0 * L ** 3 / 6.0 - _Q(self.termes, 4, L)) / L
        self.wB0 = -(self.wA0 - self.R_A0 * L * L / 2.0 + _Q(self.termes, 3, L))

    def fermer(self, M_A: float, M_B: float) -> None:
        """Les moments d'extrémité connus : R_A = R_A° + (M_B − M_A)/L ; E·I·ω₀ = E·I·ω_A + M_A·L/3 + M_B·L/6."""
        self.M_A, self.M_B = M_A, M_B
        self.R_A = self.R_A0 + (M_B - M_A) / self.L
        self.w0 = self.wA0 + M_A * self.L / 3.0 + M_B * self.L / 6.0

    @property
    def R_B(self) -> float:
        return self.charge_totale() - self.R_A

    # -- les expressions le long de la travée
    def p(self, x: float, droite: bool = True) -> float:
        return _Q(self.termes, 0, x, droite)

    def V(self, x: float, droite: bool = True) -> float:
        return self.R_A - _Q(self.termes, 1, x, droite)

    def M(self, x: float, droite: bool = True) -> float:
        return self.M_A + self.R_A * x - _Q(self.termes, 2, x, droite)

    def EIrot(self, x: float) -> float:
        return self.w0 - self.M_A * x - self.R_A * x * x / 2.0 + _Q(self.termes, 3, x)

    def EIf(self, x: float) -> float:
        return self.w0 * x - self.M_A * x * x / 2.0 - self.R_A * x ** 3 / 6.0 + _Q(self.termes, 4, x)

    def ruptures(self) -> List[float]:
        """Les abscisses où une expression change de forme : extrémités, débuts et fins de charge."""
        pts = {0.0, self.L}
        for m in self.morceaux:
            for c in ("a", "b"):
                if c in m and 0.0 < m[c] < self.L:
                    pts.add(m[c])
        return sorted(pts)


# ================================================================================================ les données
def _nombre(v, nom: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ErreurDonnees(f"{nom} : nombre attendu")
    return float(v)


def normaliser(modele: str, travees: Sequence[float], charges: Sequence[Dict]) -> Tuple[List[float], List[Dict]]:
    """Les données contrôlées : portées > 0, charges DANS la poutre (abscisses globales), valeurs finies."""
    if modele not in MODELES:
        raise ErreurDonnees(f"modèle inconnu : {modele!r}")
    Ls = [_nombre(x, "portée") for x in travees]
    if not Ls or any(x <= 0 for x in Ls):
        raise ErreurDonnees("chaque portée doit être strictement positive")
    if modele == "continue":
        if not 2 <= len(Ls) <= TRAVEES_MAX:
            raise ErreurDonnees(f"une poutre continue a de 2 à {TRAVEES_MAX} travées")
    elif len(Ls) != 1:
        raise ErreurDonnees("ce modèle n'a qu'une travée")
    Lt = sum(Ls)
    if len(charges) > CHARGES_MAX:
        raise ErreurDonnees(f"au plus {CHARGES_MAX} charges")
    out = []
    for i, c in enumerate(charges):
        ou = f"charge {i + 1}"
        if not isinstance(c, dict) or c.get("type") not in TYPES_CHARGE:
            raise ErreurDonnees(f"{ou} : type attendu parmi {TYPES_CHARGE}")
        t = c["type"]
        v = _nombre(c.get("v"), f"{ou}, valeur")
        a = _nombre(c.get("a", 0.0), f"{ou}, abscisse a")
        if t in ("P", "C"):
            if not 0.0 <= a <= Lt:
                raise ErreurDonnees(f"{ou} : l'abscisse a doit être sur la poutre (0 à {Lt:g} m)")
            out.append({"type": t, "v": v, "a": a})
            continue
        b = _nombre(c.get("b", Lt), f"{ou}, abscisse b")
        if not (0.0 <= a < b <= Lt):
            raise ErreurDonnees(f"{ou} : il faut 0 ≤ a < b ≤ {Lt:g} m")
        if t == "q":
            out.append({"type": "q", "v": v, "a": a, "b": b, "q1": v, "q2": v})
        else:
            sens = c.get("sens", "croissante")
            if sens not in SENS:
                raise ErreurDonnees(f"{ou} : sens attendu parmi {SENS}")
            out.append({"type": "tri", "v": v, "a": a, "b": b, "sens": sens,
                        "q1": 0.0 if sens == "croissante" else v, "q2": v if sens == "croissante" else 0.0})
    return Ls, out


def resultante(c: Dict) -> Tuple[float, float]:
    """(force résultante vers le bas, son moment par rapport à x = 0) d'une charge — écrits depuis la charge elle-même,
    sans passer par les termes de Macaulay (l'équilibre global les compare)."""
    if c["type"] == "P":
        return c["v"], c["v"] * c["a"]
    if c["type"] == "C":
        return 0.0, 0.0
    a, b, q1, q2 = c["a"], c["b"], c["q1"], c["q2"]
    return (q1 + q2) / 2.0 * (b - a), (b - a) * (q1 * (2 * a + b) + q2 * (a + 2 * b)) / 6.0


def _repartir(Ls: Sequence[float], charges: Sequence[Dict]) -> List[Travee]:
    """Les travées et, pour chacune, les MORCEAUX de charge en abscisses locales (une charge répartie qui enjambe un
    appui est coupée à l'appui ; une charge ponctuelle sur un appui intermédiaire revient à la travée de droite)."""
    noms = "ABCDEF"
    tr, x0 = [], 0.0
    for i, L in enumerate(Ls):
        tr.append(Travee(L, x0, noms[i], noms[i + 1]))
        x0 += L
    for i, c in enumerate(charges):
        if c["type"] in ("P", "C"):
            k = len(tr) - 1
            for j, t in enumerate(tr):
                if c["a"] < t.x0 + t.L:
                    k = j
                    break
            tr[k].ajouter({"i": i, "type": c["type"], "v": c["v"], "a": min(tr[k].L, max(0.0, c["a"] - tr[k].x0))})
            continue
        a, b, q1, q2 = c["a"], c["b"], c["q1"], c["q2"]
        for t in tr:
            u, v = max(a, t.x0), min(b, t.x0 + t.L)
            if v - u <= 1e-12 * max(1.0, t.L):
                continue
            qu = q1 + (q2 - q1) * (u - a) / (b - a)
            qv = q1 + (q2 - q1) * (v - a) / (b - a)
            t.ajouter({"i": i, "type": "q" if c["type"] == "q" else "trap", "a": u - t.x0, "b": min(t.L, v - t.x0), "q1": qu, "q2": qv,
                       "entier": abs(u - a) < 1e-12 and abs(v - b) < 1e-12, "sens": c.get("sens")})
    return tr


# ================================================================================================ la résolution
def _resoudre_tridiagonal(Ls: Sequence[float], seconds: Sequence[float]) -> List[float]:
    """Le système des trois moments (M aux appuis intermédiaires ; M = 0 aux appuis de rive), par élimination de Gauss
    sur la matrice tridiagonale — n ≤ 4 inconnues."""
    n = len(seconds)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = 2.0 * (Ls[i] + Ls[i + 1])
        if i > 0:
            A[i][i - 1] = Ls[i]
        if i < n - 1:
            A[i][i + 1] = Ls[i + 1]
    b = list(seconds)
    for i in range(1, n):
        f = A[i][i - 1] / A[i - 1][i - 1]
        A[i][i] -= f * A[i - 1][i]
        b[i] -= f * b[i - 1]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - (A[i][i + 1] * x[i + 1] if i < n - 1 else 0.0)) / A[i][i]
    return x


def _racines_quadratique(y0: float, ym: float, y1: float) -> List[float]:
    """Les racines dans ]0 ; 1[ du polynôme de degré ≤ 2 passant par (0, y0), (½, ym), (1, y1)."""
    A = 2.0 * (y0 + y1 - 2.0 * ym)
    B = y1 - y0 - A
    C = y0
    ech = max(abs(y0), abs(ym), abs(y1))
    if ech == 0.0:
        return []
    if abs(A) < 1e-12 * ech:
        if abs(B) < 1e-12 * ech:
            return []
        r = [-C / B]
    else:
        d = B * B - 4 * A * C
        if d < 0:
            return []
        s = math.sqrt(d)
        q = -(B + (s if B >= 0 else -s)) / 2.0             # forme stable
        r = [q / A] + ([C / q] if q != 0 else [])
    return sorted(t for t in r if 1e-12 < t < 1 - 1e-12)


def _dichotomie(f, u: float, v: float) -> float:
    fu = f(u)
    for _ in range(200):
        m = (u + v) / 2.0
        fm = f(m)
        if fm == 0.0 or (v - u) < 1e-14 * max(1.0, abs(v)):
            return m
        if (fm > 0) == (fu > 0):
            u, fu = m, fm
        else:
            v = m
    return (u + v) / 2.0


class Solution:
    """Le résultat d'une résolution : travées fermées, réactions, extremums, contrôles ; ``evaluer(x)`` lit une section."""

    def __init__(self, modele: str, travees: List[Travee], charges: List[Dict], EI: Optional[float]):
        self.modele, self.travees, self.charges, self.EI = modele, travees, charges, EI
        self.L = sum(t.L for t in travees)
        self.appuis: List[Dict] = []
        self.controles: List[Dict] = []
        self.extremes: Dict = {}
        self.equations: List[Dict] = []                # le système des trois moments, écrit (poutre continue)

    # -- lecture d'une section
    def _travee(self, x: float, droite: bool = True) -> Tuple[Travee, float]:
        for t in self.travees:
            fin = t.x0 + t.L
            if x < fin or (x == fin and (not droite or t is self.travees[-1])):
                return t, min(t.L, max(0.0, x - t.x0))
        t = self.travees[-1]
        return t, t.L

    def evaluer(self, x: float) -> Dict:
        """V à gauche et à droite, M à gauche et à droite, E·I·θ, E·I·f (et f, θ si E·I est donné) dans la section x."""
        x = min(self.L, max(0.0, x))
        tg, xg = self._travee(x, False)
        td, xd = self._travee(x, True)
        out = {"x": x, "V_g": tg.V(xg, False), "V_d": td.V(xd, True), "M_g": tg.M(xg, False), "M_d": td.M(xd, True),
               "EIf": td.EIf(xd), "EIrot": td.EIrot(xd)}
        if x <= 0.0:
            out["V_g"], out["M_g"] = 0.0, 0.0
        if x >= self.L:
            out["V_d"], out["M_d"] = 0.0, 0.0
        out["f"] = out["EIf"] / self.EI if self.EI else None
        out["rot"] = out["EIrot"] / self.EI if self.EI else None
        return out

    def courbes(self, n: int = 40) -> List[Dict]:
        """Par travée, les points des diagrammes (abscisse globale, V, M, E·I·f) : chaque tronçon entre deux ruptures
        est échantillonné sur l'expression fermée, les deux côtés d'un saut compris."""
        out = []
        for t in self.travees:
            pts = []
            r = t.ruptures()
            for u, v in zip(r, r[1:]):
                k = max(2, int(math.ceil(n * (v - u) / t.L)))
                for j in range(k + 1):
                    x = u + (v - u) * j / k
                    cote = j == 0 or 0 < j < k            # premier point : à droite de u ; dernier : à gauche de v
                    pts.append((t.x0 + x, t.V(x, cote), t.M(x, cote), t.EIf(x)))
            out.append({"x0": t.x0, "L": t.L, "points": pts})
        return out


def _extremes(sol: Solution) -> None:
    """Les extremums, cherchés sur les expressions fermées : V aux ruptures (et où la densité s'annule), M aux ruptures
    et où V s'annule (V est un polynôme de degré ≤ 2 sur un tronçon : racines exactes), f où la rotation s'annule
    (dichotomie sur l'expression)."""
    Vs, Ms, fs = [], [], []
    par_travee = []
    for t in sol.travees:
        mt, ft = [], []
        r = t.ruptures()
        for u, v in zip(r, r[1:]):
            pu, pv = t.p(u, True), t.p(v, False)
            Vs += [(t.V(u, True), t.x0 + u), (t.V(v, False), t.x0 + v)]
            if pu * pv < 0:
                xs = u + (v - u) * pu / (pu - pv)
                Vs.append((t.V(xs), t.x0 + xs))
            mt += [(t.M(u, True), t.x0 + u), (t.M(v, False), t.x0 + v)]
            for s in _racines_quadratique(t.V(u, True), t.V((u + v) / 2.0), t.V(v, False)):
                xs = u + (v - u) * s
                mt.append((t.M(xs), t.x0 + xs))
            ft += [(t.EIf(u), t.x0 + u), (t.EIf(v), t.x0 + v)]
            k = 32
            xs_ = [u + (v - u) * j / k for j in range(k + 1)]
            rs = [t.EIrot(x) for x in xs_]
            for j in range(k):
                if rs[j] == 0.0 and j > 0:
                    ft.append((t.EIf(xs_[j]), t.x0 + xs_[j]))
                elif rs[j] * rs[j + 1] < 0:
                    xr = _dichotomie(t.EIrot, xs_[j], xs_[j + 1])
                    ft.append((t.EIf(xr), t.x0 + xr))
        Ms += mt
        fs += ft
        par_travee.append({"M_max": max(mt), "M_min": min(mt), "f_max": max(ft), "f_min": min(ft)})
    ech_M = max((abs(m) for m, _ in Ms), default=0.0)
    ech_f = max((abs(f) for f, _ in fs), default=0.0)

    def net(p, ech):
        return (0.0 if abs(p[0]) < 1e-12 * max(ech, 1e-300) else p[0], p[1])

    sol.extremes = {"V_max": max(Vs), "V_min": min(Vs), "M_max": net(max(Ms), ech_M), "M_min": net(min(Ms), ech_M),
                    "EIf_max": net(max(fs), ech_f), "EIf_min": net(min(fs), ech_f),
                    "travees": [{k: net(v, ech_M if k[0] == "M" else ech_f) for k, v in d.items()} for d in par_travee]}


def _controler(sol: Solution) -> None:
    """Les contrôles internes : équilibre des forces et des moments (écrit depuis les charges, pas depuis les termes),
    flèche nulle aux appuis, rotation nulle aux encastrements, continuité aux appuis intermédiaires, et accord du
    FORMULAIRE avec les paramètres initiaux pour chaque travée isostatique. Un écart lève ErreurControle."""
    F = sum(resultante(c)[0] for c in sol.charges)
    Mo = sum(resultante(c)[1] for c in sol.charges) + sum(c["v"] for c in sol.charges if c["type"] == "C")
    ech_F = sum(abs(resultante(c)[0]) for c in sol.charges) + sum(abs(c["v"]) / sol.L for c in sol.charges if c["type"] == "C") or 1.0
    ech_M = ech_F * sol.L
    ech_w = ech_F * sol.L ** 2
    ech_f = ech_F * sol.L ** 3
    c = []
    c.append(("équilibre vertical : Σ R = Σ charges", sum(a["R"] for a in sol.appuis) - F, ech_F))
    c.append(("équilibre des moments par rapport à A", sum(a.get("M_enc_horaire", 0.0) for a in sol.appuis) + Mo
              - sum(a["R"] * a["x"] for a in sol.appuis), ech_M))
    for t in sol.travees:
        if sol.modele != "console":
            c.append((f"flèche nulle en {t.nom_B}", t.EIf(t.L), ech_f))
            c.append((f"moment en {t.nom_B} = M_{t.nom_B}", t.M(t.L, False) - t.M_B
                      + sum(m["v"] for m in t.morceaux if m["type"] == "C" and m["a"] >= t.L), ech_M))
        fo = [rotations_formulaire(m, t.L) for m in t.morceaux]
        c.append((f"travée {t.nom_A}{t.nom_B} : R_A du formulaire = paramètres initiaux", sum(x[0] for x in fo) - t.R_A0, ech_F))
        c.append((f"travée {t.nom_A}{t.nom_B} : E·I·ω_A du formulaire = paramètres initiaux", sum(x[2] for x in fo) - t.wA0, ech_w))
        c.append((f"travée {t.nom_A}{t.nom_B} : E·I·ω_B du formulaire = paramètres initiaux", sum(x[3] for x in fo) - t.wB0, ech_w))
    t0, tn = sol.travees[0], sol.travees[-1]
    if sol.modele in ("console", "enc_app", "bi_enc"):
        c.append(("rotation nulle à l'encastrement A", t0.EIrot(0.0), ech_w))
    if sol.modele == "bi_enc":
        c.append(("rotation nulle à l'encastrement B", tn.EIrot(tn.L), ech_w))
    if sol.modele == "console":
        c.append(("moment nul à l'extrémité libre", t0.M(t0.L, True), ech_M))
        c.append(("tranchant nul à l'extrémité libre", t0.V(t0.L, True), ech_F))
    for g, d in zip(sol.travees, sol.travees[1:]):
        c.append((f"rotation continue en {d.nom_A}", g.EIrot(g.L) - d.EIrot(0.0), ech_w))
    sol.controles = [{"nom": n, "ecart": e, "echelle": s, "ok": abs(e) <= 1e-7 * s} for n, e, s in c]
    faux = [k for k in sol.controles if not k["ok"]]
    if faux:
        raise ErreurControle("contrôle interne en défaut : " + " ; ".join(f"{k['nom']} (écart {k['ecart']:.3g})" for k in faux))


def resoudre(modele: str, travees: Sequence[float], charges: Sequence[Dict], EI: Optional[float] = None) -> Solution:
    """La poutre résolue. ``travees`` : les portées (m) ; ``charges`` : {type q|tri|P|C, v, a, b, sens} en abscisses
    globales ; ``EI`` (kN·m², facultatif) pour la flèche."""
    Ls, ch = normaliser(modele, travees, charges)
    if EI is not None:
        EI = _nombre(EI, "E·I")
        if EI <= 0:
            raise ErreurDonnees("E·I doit être strictement positif")
    tr = _repartir(Ls, ch)
    sol = Solution(modele, tr, ch, EI)
    for t in tr:
        t.isostatique()
    t0 = tr[0]
    if modele == "appuis":
        t0.fermer(0.0, 0.0)
    elif modele == "enc_app":
        t0.fermer(-3.0 * t0.wA0 / t0.L, 0.0)
    elif modele == "bi_enc":
        t0.fermer(-2.0 * (2.0 * t0.wA0 - t0.wB0) / t0.L, -2.0 * (2.0 * t0.wB0 - t0.wA0) / t0.L)
    elif modele == "console":
        L = t0.L
        t0.R_A = t0.charge_totale()
        t0.M_A = _Q(t0.termes, 2, L, True) - t0.R_A * L
        t0.M_B = 0.0
        t0.w0 = 0.0
    else:
        seconds = [-6.0 * (g.wB0 + d.wA0) for g, d in zip(tr, tr[1:])]
        Mi = [0.0] + _resoudre_tridiagonal(Ls, seconds) + [0.0]
        for i, t in enumerate(tr):
            t.fermer(Mi[i], Mi[i + 1])
        for i in range(1, len(tr)):
            sol.equations.append({"appui": tr[i].nom_A, "L_g": Ls[i - 1], "L_d": Ls[i], "wB_g": tr[i - 1].wB0, "wA_d": tr[i].wA0,
                                  "second": seconds[i - 1], "M_g": Mi[i - 1], "M": Mi[i], "M_d": Mi[i + 1]})
    # les appuis : réaction (vers le haut), moment d'encastrement (le moment fléchissant dans la section d'encastrement)
    if modele == "console":
        sol.appuis = [{"nom": "A", "x": 0.0, "R": t0.R_A, "M": t0.M_A, "M_enc_horaire": t0.M_A, "type": "encastrement"}]
    else:
        for i, t in enumerate(tr):
            R = t.R_A + (tr[i - 1].R_B if i > 0 else 0.0)
            a = {"nom": t.nom_A, "x": t.x0, "R": R, "M": t.M_A, "type": "appui"}
            if i == 0 and modele in ("enc_app", "bi_enc"):
                a.update(type="encastrement", M_enc_horaire=t.M_A)
            sol.appuis.append(a)
        tn = tr[-1]
        b = {"nom": tn.nom_B, "x": tn.x0 + tn.L, "R": tn.R_B, "M": tn.M_B, "type": "appui"}
        if modele == "bi_enc":
            b.update(type="encastrement", M_enc_horaire=-tn.M_B)
        sol.appuis.append(b)
    _controler(sol)
    _extremes(sol)
    return sol


# ================================================================================================ les cas d'école
def cas_ecole(sol: Solution) -> Optional[Dict]:
    """Si la poutre est un CAS D'ÉCOLE du formulaire (une seule charge, cas usuel), ses formules classiques ÉVALUÉES :
    {cle, valeurs: {nom: valeur}}. Chaque valeur est comparée à la résolution générale ; un écart lève ErreurControle
    (deux calculs indépendants). Sinon None : la feuille montre alors la superposition."""
    if len(sol.charges) != 1:
        return None
    c = sol.charges[0]
    m, t = sol.modele, c["type"]
    L = sol.travees[0].L
    v = c["v"]
    egales = all(abs(x.L - L) < 1e-12 * L for x in sol.travees)
    totale = t in ("q", "tri") and c["a"] == 0.0 and abs(c["b"] - sol.L) < 1e-12 * sol.L
    r3, r5, r33 = math.sqrt(3.0), math.sqrt(5.0), math.sqrt(33.0)
    cle, val = None, {}
    if m == "appuis" and t == "q" and totale:
        cle, val = "appuis_q", {"R_A": v * L / 2, "R_B": v * L / 2, "M_max": v * L * L / 8, "x_M": L / 2, "EIf_max": 5 * v * L ** 4 / 384, "x_f": L / 2}
    elif m == "appuis" and t == "P" and 0 < c["a"] < L:
        a = c["a"]
        b = L - a
        if a >= b:
            xf, ff = math.sqrt((L * L - b * b) / 3), v * b * (L * L - b * b) ** 1.5 / (9 * r3 * L)
        else:
            xf, ff = L - math.sqrt((L * L - a * a) / 3), v * a * (L * L - a * a) ** 1.5 / (9 * r3 * L)
        cle, val = "appuis_P", {"R_A": v * b / L, "R_B": v * a / L, "M_max": v * a * b / L, "x_M": a, "EIf_a": v * a * a * b * b / (3 * L),
                                "EIf_max": ff, "x_f": xf}
        if abs(a - b) < 1e-12 * L:
            cle = "appuis_P_milieu"
            val["EIf_max"] = v * L ** 3 / 48
    elif m == "appuis" and t == "tri" and totale:
        xm = L / r3 if c["sens"] == "croissante" else L - L / r3
        s = L * math.sqrt(1 - math.sqrt(8 / 15))
        xf = s if c["sens"] == "croissante" else L - s
        ff = v * s * (3 * s ** 4 - 10 * L * L * s * s + 7 * L ** 4) / (360 * L)
        ra, rb = (v * L / 6, v * L / 3) if c["sens"] == "croissante" else (v * L / 3, v * L / 6)
        cle, val = "appuis_tri", {"R_A": ra, "R_B": rb, "M_max": v * L * L / (9 * r3), "x_M": xm, "EIf_max": ff, "x_f": xf}
    elif m == "console" and t == "q" and totale:
        cle, val = "console_q", {"R_A": v * L, "M_A": -v * L * L / 2, "EIf_max": v * L ** 4 / 8, "x_f": L}
    elif m == "console" and t == "P" and c["a"] > 0:
        a = c["a"]
        cle, val = "console_P", {"R_A": v, "M_A": -v * a, "EIf_a": v * a ** 3 / 3, "EIf_max": v * a * a * (3 * L - a) / 6, "x_f": L}
    elif m == "console" and t == "tri" and totale and c["sens"] == "decroissante":
        cle, val = "console_tri", {"R_A": v * L / 2, "M_A": -v * L * L / 6, "EIf_max": v * L ** 4 / 30, "x_f": L}
    elif m == "enc_app" and t == "q" and totale:
        cle, val = "enc_app_q", {"M_A": -v * L * L / 8, "R_A": 5 * v * L / 8, "R_B": 3 * v * L / 8, "M_max": 9 * v * L * L / 128, "x_M": 5 * L / 8,
                                 "EIf_max": (39 + 55 * r33) / 65536 * v * L ** 4, "x_f": L - (1 + r33) * L / 16}
    elif m == "enc_app" and t == "P" and 0 < c["a"] < L:
        a = c["a"]
        b = L - a
        rb = v * a * a * (3 * L - a) / (2 * L ** 3)
        cle, val = "enc_app_P", {"M_A": -v * a * b * (L + b) / (2 * L * L), "R_A": v - rb, "R_B": rb, "M_max": rb * b, "x_M": a}
        if abs(a - b) < 1e-12 * L:
            cle = "enc_app_P_milieu"
            val.update(EIf_max=v * L ** 3 / (48 * r5), x_f=L - L / r5)
    elif m == "bi_enc" and t == "q" and totale:
        cle, val = "bi_enc_q", {"M_A": -v * L * L / 12, "M_B": -v * L * L / 12, "R_A": v * L / 2, "R_B": v * L / 2, "M_max": v * L * L / 24, "x_M": L / 2,
                                "EIf_max": v * L ** 4 / 384, "x_f": L / 2}
    elif m == "bi_enc" and t == "P" and 0 < c["a"] < L:
        a = c["a"]
        b = L - a
        cle, val = "bi_enc_P", {"M_A": -v * a * b * b / (L * L), "M_B": -v * a * a * b / (L * L), "R_A": v * b * b * (3 * a + b) / L ** 3,
                                "R_B": v * a * a * (a + 3 * b) / L ** 3, "M_max": 2 * v * a * a * b * b / L ** 3, "x_M": a,
                                "EIf_a": v * a ** 3 * b ** 3 / (3 * L ** 3)}
        if abs(a - b) < 1e-12 * L:
            cle = "bi_enc_P_milieu"
            val.update(EIf_max=v * L ** 3 / 192, x_f=L / 2)
    elif m == "continue" and t == "q" and totale and egales and len(sol.travees) == 2:
        cle, val = "continue_2_q", {"M_B": -v * L * L / 8, "R_A": 3 * v * L / 8, "R_B": 5 * v * L / 4, "R_C": 3 * v * L / 8, "M_max": 9 * v * L * L / 128,
                                    "x_M": 3 * L / 8, "EIf_max": (39 + 55 * r33) / 65536 * v * L ** 4, "x_f": (1 + r33) * L / 16}
    elif m == "continue" and t == "q" and totale and egales and len(sol.travees) == 3:
        cle, val = "continue_3_q", {"M_B": -v * L * L / 10, "M_C": -v * L * L / 10, "R_A": 0.4 * v * L, "R_B": 1.1 * v * L, "R_C": 1.1 * v * L, "R_D": 0.4 * v * L,
                                    "M_max": 0.08 * v * L * L, "x_M": 0.4 * L, "M_milieu": 0.025 * v * L * L}
    if cle is None:
        return None
    # --- la comparaison à la résolution générale (signe de la charge : les « max » suivent |v|)
    sg = 1.0 if v >= 0 else -1.0
    gen: Dict[str, float] = {}
    app = {a["nom"]: a for a in sol.appuis}
    for k in val:
        if k.startswith("R_"):
            gen[k] = app[k[2:]]["R"]
        elif k in ("M_A", "M_B", "M_C"):
            gen[k] = app[k[2:]]["M"]
        elif k == "M_max":
            gen[k] = (sol.extremes["M_max"] if sg > 0 else sol.extremes["M_min"])[0]
        elif k == "x_M":
            xs = (sol.extremes["M_max"] if sg > 0 else sol.extremes["M_min"])[1]
            gen[k] = xs if not (cle.startswith("continue") and xs > L) else sol.L - xs      # symétrie : la première travée
        elif k == "EIf_max":
            gen[k] = (sol.extremes["EIf_max"] if sg > 0 else sol.extremes["EIf_min"])[0]
        elif k == "x_f":
            xs = (sol.extremes["EIf_max"] if sg > 0 else sol.extremes["EIf_min"])[1]
            gen[k] = xs if not (cle.startswith("continue") and xs > L) else sol.L - xs
        elif k == "EIf_a":
            gen[k] = sol.evaluer(c["a"])["EIf"]
        elif k == "M_milieu":
            gen[k] = sol.evaluer(sol.L / 2)["M_d"]
    ech = {"R": abs(v) * (L if t in ("q", "tri") else 1.0), "M": abs(v) * (L * L if t in ("q", "tri") else L),
           "E": abs(v) * (L ** 4 if t in ("q", "tri") else L ** 3), "x": L}
    faux = [k for k in val if abs(val[k] - gen[k]) > 1e-6 * max(ech[k[0]], 1e-300)]
    if faux:
        raise ErreurControle("formule de formulaire ≠ résolution générale : " + ", ".join(f"{k} ({val[k]:.6g} ≠ {gen[k]:.6g})" for k in faux))
    return {"cle": cle, "valeurs": val, "general": gen}

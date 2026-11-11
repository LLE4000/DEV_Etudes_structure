# -*- coding: utf-8 -*-
"""Fondations de grues — LA POUTRE SUR DEUX APPUIS, résolue exactement.

Toute la statique des classeurs du bureau est celle-ci : une poutre droite posée sur deux appuis simples (deux pieux,
ou deux semelles), chargée par un ou deux pieds de la grue et par son poids propre, dont on déplace un appui ou la
charge de la TOLÉRANCE D'EXÉCUTION « e ». Ce fichier la résout une fois pour toutes, sans rien savoir des classeurs :

- il trace les diagrammes (M et V le long de la poutre, une courbe par position de tolérance, puis l'enveloppe) ;
- il REDÉMONTRE les formules des classeurs : ``tests/test_fondation_grue_moteur.py`` exige que ses réactions, ses
  moments et ses efforts tranchants soient ceux de ``classeur.py`` (qui recopie les cellules), et nomme les endroits où
  ils diffèrent (maximum échantillonné tous les 5 cm, par exemple).

Conventions : abscisses en m le long de la poutre, dans un repère NOMINAL commun à toutes les positions de tolérance
(les appuis bougent, le repère non) ; charges vers le bas positives (kN, kN/m) ; M > 0 quand la fibre INFÉRIEURE est
tendue ; V > 0 quand la résultante des forces à gauche de la coupe monte. Bibliothèque standard seulement.
"""
from typing import Dict, List, Optional, Sequence, Tuple

EPS = 1e-12


class Poutre:
    """Une poutre de ``x0`` à ``x1`` sur deux appuis simples ``xa`` < ``xb`` (porte-à-faux admis de chaque côté).

    ``ponctuelles`` : [(x, P)] ; ``reparties`` : [(xd, xf, q)] avec q en kN/m. Les réactions RA et RB (kN, vers le
    haut) sortent des deux équations d'équilibre ; une réaction négative est un SOULÈVEMENT de l'appui."""

    def __init__(self, x0: float, x1: float, xa: float, xb: float,
                 ponctuelles: Sequence[Tuple[float, float]] = (), reparties: Sequence[Tuple[float, float, float]] = ()):
        if not (x1 > x0 and xb > xa):
            raise ValueError("poutre : il faut x1 > x0 et xb > xa")
        if xa < x0 - EPS or xb > x1 + EPS:
            raise ValueError("poutre : les appuis doivent être sur la poutre")
        self.x0, self.x1, self.xa, self.xb = float(x0), float(x1), float(xa), float(xb)
        self.ponctuelles = [(float(x), float(p)) for x, p in ponctuelles]
        self.reparties = [(float(a), float(b), float(q)) for a, b, q in reparties if b > a]
        for x, _ in self.ponctuelles:
            if x < x0 - EPS or x > x1 + EPS:
                raise ValueError("poutre : une charge est hors de la poutre")
        charge = sum(p for _, p in self.ponctuelles) + sum(q * (b - a) for a, b, q in self.reparties)
        moment_a = (sum(p * (x - xa) for x, p in self.ponctuelles)
                    + sum(q * (b - a) * ((a + b) / 2 - xa) for a, b, q in self.reparties))
        self.RB = moment_a / (xb - xa)
        self.RA = charge - self.RB
        self.portee = xb - xa
        pts = {self.x0, self.x1, self.xa, self.xb}
        pts.update(x for x, _ in self.ponctuelles)
        for a, b, _ in self.reparties:
            pts.update((a, b))
        self.noeuds = sorted(pts)

    # ---- efforts
    def _forces(self):
        """Les forces concentrées, vers le haut positives : (x, F)."""
        return [(self.xa, self.RA), (self.xb, self.RB)] + [(x, -p) for x, p in self.ponctuelles]

    def V(self, x: float, gauche: bool = False) -> float:
        """Effort tranchant juste à droite de x (ou juste à gauche si ``gauche``)."""
        v = sum(f for xf, f in self._forces() if (xf < x - EPS if gauche else xf <= x + EPS))
        for a, b, q in self.reparties:
            v -= q * max(min(x, b) - a, 0.0)
        return v

    def M(self, x: float) -> float:
        """Moment fléchissant en x (kN·m), fibre inférieure tendue positive."""
        m = sum(f * (x - xf) for xf, f in self._forces() if xf <= x)
        for a, b, q in self.reparties:
            m -= q * (max(x - a, 0.0) ** 2 - max(x - b, 0.0) ** 2) / 2
        return m

    def _q(self, xg: float, xd: float) -> float:
        """Charge répartie totale sur le tronçon ]xg, xd[ (entre deux nœuds consécutifs)."""
        milieu = (xg + xd) / 2
        return sum(q for a, b, q in self.reparties if a < milieu < b)

    def points_singuliers(self) -> List[float]:
        """Les abscisses où M peut être extrême : les nœuds, et les points d'effort tranchant nul dans un tronçon."""
        pts = list(self.noeuds)
        for xg, xd in zip(self.noeuds, self.noeuds[1:]):
            q = self._q(xg, xd)
            if abs(q) > EPS:
                xs = xg + self.V(xg) / q
                if xg + EPS < xs < xd - EPS:
                    pts.append(xs)
        return sorted(pts)

    def extremes(self) -> Dict[str, Dict[str, float]]:
        """Les extremums EXACTS : M max, M min et le plus grand |V|, avec leur abscisse."""
        pts = self.points_singuliers()
        mmax = max(pts, key=self.M)
        mmin = min(pts, key=self.M)
        vs = [(abs(self.V(x, g)), x, self.V(x, g)) for x in self.noeuds for g in (True, False)]
        v = max(vs, key=lambda t: t[0])
        return {"M_max": {"valeur": self.M(mmax), "x": mmax}, "M_min": {"valeur": self.M(mmin), "x": mmin},
                "V_max": {"valeur": v[0], "x": v[1], "signe": 1.0 if v[2] >= 0 else -1.0}}

    def courbe(self, pas: Optional[float] = None) -> Dict[str, List[float]]:
        """Les deux diagrammes, prêts à tracer : chaque nœud est donné deux fois (à gauche, à droite) pour que les
        sauts de V soient verticaux ; entre deux nœuds, un point tous les ``pas`` (défaut : longueur / 120) ; les
        points d'effort tranchant nul sont ajoutés (les sommets de M)."""
        pas = pas or (self.x1 - self.x0) / 120
        xs, vs, ms = [], [], []
        singuliers = self.points_singuliers()
        for xg, xd in zip(self.noeuds, self.noeuds[1:]):
            n = max(int(round((xd - xg) / pas)), 1)
            interieurs = [xg + (xd - xg) * i / n for i in range(1, n)]
            interieurs += [x for x in singuliers if xg + EPS < x < xd - EPS]
            xs.append(xg), vs.append(self.V(xg)), ms.append(self.M(xg))
            for x in sorted(interieurs):
                xs.append(x), vs.append(self.V(x)), ms.append(self.M(x))
            xs.append(xd), vs.append(self.V(xd, gauche=True)), ms.append(self.M(xd))
        return {"x": xs, "V": vs, "M": ms}

    def decrire(self) -> dict:
        """Tout ce qu'il faut pour dessiner et pour relire : géométrie, charges, réactions, extremums, courbes."""
        return {"x0": self.x0, "x1": self.x1, "appuis": [self.xa, self.xb], "portee": self.portee,
                "ponctuelles": [list(c) for c in self.ponctuelles], "reparties": [list(c) for c in self.reparties],
                "RA": self.RA, "RB": self.RB, "extremes": self.extremes(), "courbe": self.courbe()}


def enveloppe(poutres: Sequence[Poutre], n: int = 160) -> Dict[str, List[Optional[float]]]:
    """L'enveloppe de plusieurs poutres dessinées dans le même repère : en chaque abscisse d'une grille commune, le
    plus grand et le plus petit M, le plus grand et le plus petit V. Hors de l'étendue d'une poutre, elle ne compte
    pas. La grille contient tous les nœuds (de toutes les poutres) et leurs points d'effort tranchant nul."""
    x0, x1 = min(p.x0 for p in poutres), max(p.x1 for p in poutres)
    pts = {x0 + (x1 - x0) * i / n for i in range(n + 1)}
    for p in poutres:
        pts.update(p.points_singuliers())
    xs = sorted(pts)
    out = {"x": [], "M_max": [], "M_min": [], "V_max": [], "V_min": []}
    for x in xs:
        dedans = [p for p in poutres if p.x0 - EPS <= x <= p.x1 + EPS]
        if not dedans:
            continue
        ms = [p.M(x) for p in dedans]
        vs = [v for p in dedans for v in (p.V(x, gauche=True), p.V(x))]
        out["x"].append(x)
        out["M_max"].append(max(ms)), out["M_min"].append(min(ms))
        out["V_max"].append(max(vs)), out["V_min"].append(min(vs))
    return out


# ================================================================== les positions de tolérance des classeurs
def poutre_un_pied(L: float, e: float, P: float, pp: float) -> Dict[str, Poutre]:
    """La poutre du cadre de 4 pieux et la poutre secondaire : longueur L de pieu à pieu, UN pied de la grue (P) à
    l'aplomb nominal du pieu A (x = 0), pieu B en x = L.
    CAS A : le pieu A est RENTRÉ de e (x = e) — le pied est en porte-à-faux, la poutre va de 0 à L ;
    CAS B : le pieu A est SORTI de e (x = −e) — la poutre va de −e à L, le pied est à e de l'appui."""
    return {"A": Poutre(0.0, L, e, L, [(0.0, P)], [(0.0, L, pp)]),
            "B": Poutre(-e, L, -e, L, [(0.0, P)], [(-e, L, pp)])}


def poutre_deux_pieux(Lp: float, s: float, e: float, P: float, pp: float, c: float = 0.0) -> Dict[str, Poutre]:
    """La poutre primaire et la semelle sur pieux : longueur Lp centrée sur ses deux pieux (entraxe s, en ± s/2), une
    charge P en son milieu nominal (x = 0), ponctuelle ou répartie sur la longueur c.
    CAS A : chaque pieu RENTRÉ de e/2 (portée s − e) et la charge décalée de e/2 vers le pieu A ;
    CAS B : chaque pieu SORTI de e/2 (portée s + e), la charge au milieu."""
    def charge(x):
        return ([(x, P)], []) if c <= 0 else ([], [(x - c / 2, x + c / 2, P / c)])

    pa, ra = charge(-e / 2)
    pb, rb = charge(0.0)
    poids = [(-Lp / 2, Lp / 2, pp)]
    return {"A": Poutre(-Lp / 2, Lp / 2, -(s - e) / 2, (s - e) / 2, pa, ra + poids),
            "B": Poutre(-Lp / 2, Lp / 2, -(s + e) / 2, (s + e) / 2, pb, rb + poids)}


def poutre_support(Lc: float, a: float, b: float, e: float, sous_cas: str, P1: float, P2: float, pp: float) -> Poutre:
    """La poutre support du châssis : appui A (axe de la semelle) en x = 0, pieds de la grue en a et a + Lc, appui B en
    a + Lc + b, dans le repère nominal. Le modèle du classeur s'arrête aux deux appuis (pas de porte-à-faux).
    A.a : l'appui A recule de e ; A.b : l'appui B recule de e ; B.a : le châssis est posé décalé de + e ;
    B.b : de − e."""
    Ln = Lc + a + b
    if sous_cas == "A.a":
        return Poutre(-e, Ln, -e, Ln, [(a, P1), (a + Lc, P2)], [(-e, Ln, pp)])
    if sous_cas == "A.b":
        return Poutre(0.0, Ln + e, 0.0, Ln + e, [(a, P1), (a + Lc, P2)], [(0.0, Ln + e, pp)])
    if sous_cas == "B.a":
        return Poutre(0.0, Ln, 0.0, Ln, [(a + e, P1), (a + e + Lc, P2)], [(0.0, Ln, pp)])
    if sous_cas == "B.b":
        return Poutre(0.0, Ln, 0.0, Ln, [(a - e, P1), (a - e + Lc, P2)], [(0.0, Ln, pp)])
    raise ValueError(f"sous-cas « {sous_cas} » inconnu")

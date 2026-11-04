# -*- coding: utf-8 -*-
"""Le cordon : un segment de gorge a dans un dièdre, et la décomposition
exacte d'une force par unité de longueur sur sa gorge (EN 1993-1-8 fig. 4.5).

Repère du groupe : le PLAN DE JOINT est z = 0 (x, y dedans), z sa normale.
Un cordon porte l'axe e_a (de P1 vers P2), et les directions u_A, u_B de ses
deux ailes, de la racine vers chaque face, perpendiculaires à l'axe et entre
elles. Le plan de gorge contient l'axe et la bissectrice (u_A + u_B)/√2 :
  n_t = (u_A − u_B)/√2   normale au plan de gorge  → σ⊥
  t_⊥ = (u_A + u_B)/√2   dans la gorge, en travers → τ⊥
  e_a                    dans la gorge, le long    → τ∥
Une force f (N/mm) sur le cordon donne σ⊥ = f·n_t/a, τ⊥ = f·t_⊥/a, τ∥ = f·e_a/a.

Cordon d'angle usuel : u_B = +z (le long de la face de la pièce rapportée),
u_A dans le plan de joint, perpendiculaire à l'axe, dirigé HORS de la pièce
rapportée. Soudure bout à bout à pénétration partielle : « ailes virtuelles »
u_A = (n + z)/√2, u_B = (z − n)/√2 avec n la normale au cordon dans le plan :
la gorge est alors perpendiculaire à n (σ⊥ = effort normal / (a·l)).
Unités : mm, N, N/mm, N/mm².
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

A_MIN = 3.0            # gorge minimale (EN 1993-1-8 §4.5.2)
L_MIN_ABS = 30.0       # longueur efficace minimale : max(30 mm ; 6a) (§4.5.1)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("vecteur nul")
    return v / n


@dataclass
class Cordon:
    nom: str
    p1: Tuple[float, float, float]
    p2: Tuple[float, float, float]
    a: float
    u_A: Tuple[float, float, float]
    u_B: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    retours: bool = False                 # retours d'angle exécutés → longueur entière (§4.5.1)
    bout_a_bout: bool = False             # pénétration partielle : gorge = pénétration
    continu: bool = False                 # tronçon d'un cordon continu (pourtour) : pas de longueur minimale propre
    note: str = ""
    _cache: Dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ géométrie
    @property
    def P1(self) -> np.ndarray:
        return np.asarray(self.p1, dtype=float)

    @property
    def P2(self) -> np.ndarray:
        return np.asarray(self.p2, dtype=float)

    @property
    def longueur(self) -> float:
        return float(np.linalg.norm(self.P2 - self.P1))

    @property
    def l_eff(self) -> float:
        """Longueur efficace : l − 2a sans retours d'angle (§4.5.1), bornée à 0 ;
        longueur entière pour un tronçon de cordon continu."""
        return self.longueur if (self.retours or self.continu) else max(0.0, self.longueur - 2.0 * self.a)

    @property
    def utile(self) -> bool:
        """Un cordon compte si l_eff ≥ max(30 mm ; 6a) et a ≥ 3 mm (§4.5.1, §4.5.2) ;
        un tronçon de cordon continu ne porte pas de longueur minimale propre."""
        if self.a < A_MIN - 1e-9:
            return False
        return self.continu or self.l_eff >= max(L_MIN_ABS, 6.0 * self.a) - 1e-9

    @property
    def e_a(self) -> np.ndarray:
        return _unit(self.P2 - self.P1)

    @property
    def n_t(self) -> np.ndarray:
        return _unit(np.asarray(self.u_A, float) - np.asarray(self.u_B, float))

    @property
    def t_perp(self) -> np.ndarray:
        return _unit(np.asarray(self.u_A, float) + np.asarray(self.u_B, float))

    def milieu(self) -> np.ndarray:
        return (self.P1 + self.P2) / 2.0

    def point(self, s: float) -> np.ndarray:
        """Point d'abscisse réduite s ∈ [0, 1] sur la longueur EFFICACE,
        centrée sur le cordon (les 2a de bout ne comptent pas)."""
        c = self.milieu()
        return c + (s - 0.5) * self.l_eff * self.e_a

    def verifier_geometrie(self) -> None:
        ea, ua, ub = self.e_a, _unit(self.u_A), _unit(self.u_B)
        if abs(float(np.dot(ea, ua))) > 1e-6 or abs(float(np.dot(ea, ub))) > 1e-6 or abs(float(np.dot(ua, ub))) > 1e-6:
            raise ValueError(f"{self.nom} : ailes non perpendiculaires à l'axe ou entre elles")

    # ------------------------------------------------------------ gorge
    def composantes(self, f) -> Dict[str, float]:
        """σ⊥, τ⊥, τ∥ (N/mm²) et |f| (N/mm) pour une force f (N/mm) sur le cordon."""
        f = np.asarray(f, dtype=float)
        return {"sigma_perp": float(np.dot(f, self.n_t)) / self.a,
                "tau_perp": float(np.dot(f, self.t_perp)) / self.a,
                "tau_par": float(np.dot(f, self.e_a)) / self.a,
                "f": float(np.linalg.norm(f))}


def cordon_angle(nom: str, p1, p2, a: float, cote: int = +1, **kw) -> Cordon:
    """Cordon d'angle dans le plan de joint : la pièce rapportée est à GAUCHE
    de l'axe (côté e_z × e_a) pour cote = +1, à droite pour −1 ; u_A pointe
    hors de la pièce, u_B = +z."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    if len(p1) == 2:
        p1 = np.array([p1[0], p1[1], 0.0])
    if len(p2) == 2:
        p2 = np.array([p2[0], p2[1], 0.0])
    ea = _unit(p2 - p1)
    gauche = np.cross(np.array([0.0, 0.0, 1.0]), ea)          # normale dans le plan, à gauche de l'axe
    u_A = -cote * gauche                                         # hors de la pièce rapportée
    return Cordon(nom, tuple(p1), tuple(p2), float(a), tuple(u_A), (0.0, 0.0, 1.0), **kw)


def cordon_bout_a_bout(nom: str, p1, p2, a: float, cote: int = +1, **kw) -> Cordon:
    """Soudure bout à bout à pénétration partielle, de gorge a = pénétration :
    ailes virtuelles telles que la gorge soit perpendiculaire à la normale n
    du cordon dans le plan (n = e_z × e_a pour cote = +1)."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    if len(p1) == 2:
        p1 = np.array([p1[0], p1[1], 0.0])
    if len(p2) == 2:
        p2 = np.array([p2[0], p2[1], 0.0])
    ea = _unit(p2 - p1)
    n = cote * np.cross(np.array([0.0, 0.0, 1.0]), ea)
    z = np.array([0.0, 0.0, 1.0])
    u_A = _unit(n + z)
    u_B = _unit(z - n)
    return Cordon(nom, tuple(p1), tuple(p2), float(a), tuple(u_A), tuple(u_B), bout_a_bout=True, **kw)


def beta_Lw(L_j: float, a: float) -> float:
    """Assemblage long par recouvrement (§4.11) : βLw,1 = 1,2 − 0,2·L_j/(150·a) ≤ 1,
    borné à 0 ; 1 si L_j ≤ 150·a."""
    if L_j <= 150.0 * a:
        return 1.0
    return max(0.0, min(1.0, 1.2 - 0.2 * L_j / (150.0 * a)))

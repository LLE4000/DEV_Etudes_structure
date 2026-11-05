# -*- coding: utf-8 -*-
"""Le groupe de cordons : propriétés des gorges traitées comme des lignes
d'aire a·dl, et répartition ÉLASTIQUE des six composantes d'effort.

Efforts au point de référence R (N, N·mm), dans le repère du plan de joint :
  F = (V_x, V_y, N)   avec N normal au plan (traction > 0 : arrachement)
  M = (M_x, M_y, M_z) avec M_x, M_y de flexion (axes du plan), M_z de torsion.
Transport en G (centre des gorges) : M_G = M + (R − G) × F.
Force par unité de longueur en un point P d'un cordon de gorge a :
  f = a·[ F/A_w  +  σ_flexion(P)·e_z  +  (M_z/J)·e_z × GP ]
avec, pour des axes quelconques (I_xy ≠ 0 compris),
  σ_flexion = [ (M_x·I_y + M_y·I_xy)·(y − y_G) − (M_y·I_x + M_x·I_xy)·(x − x_G) ] / (I_x·I_y − I_xy²)
(M_x > 0 tend la fibre y > 0 ; M_y > 0 comprime la fibre x > 0). Les gorges
différentes sont pondérées par a : chaque cordon prend une force
proportionnelle à sa gorge, à contrainte égale. f varie linéairement le long
d'un cordon : on l'évalue aux deux extrémités efficaces et au milieu.
Unités : mm, N, N·mm, N/mm, mm², mm⁴.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .cordon import Cordon


@dataclass
class Efforts:
    """Six composantes au point de référence (N et N·mm)."""
    N: float = 0.0
    Vx: float = 0.0
    Vy: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0
    reference: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def F(self) -> np.ndarray:
        return np.array([self.Vx, self.Vy, self.N], dtype=float)

    def M(self) -> np.ndarray:
        return np.array([self.Mx, self.My, self.Mz], dtype=float)


class Groupe:
    def __init__(self, cordons: Sequence[Cordon], beta_Lw: float = 1.0):
        self.cordons = [c for c in cordons]
        self.beta_Lw = beta_Lw
        for c in self.cordons:
            c.verifier_geometrie()

    # ------------------------------------------------------------ propriétés
    @property
    def utiles(self) -> List[Cordon]:
        return [c for c in self.cordons if c.utile]

    def proprietes(self) -> Dict[str, float]:
        """A_w, centre G, I_x, I_y, I_xy, J (polaire) des gorges utiles."""
        cs = self.utiles
        if not cs:
            raise ValueError("aucun cordon utile (gorge ≥ 3 mm et longueur efficace ≥ max(30 mm ; 6a))")
        A = sum(c.a * c.l_eff for c in cs)
        G = sum(c.a * c.l_eff * c.milieu() for c in cs) / A
        Ix = Iy = Ixy = 0.0
        for c in cs:
            p1, p2 = c.point(0.0) - G, c.point(1.0) - G
            L, a = c.l_eff, c.a
            # intégrales d'un segment droit : ∫ y² dl = L·(y1² + y1·y2 + y2²)/3, ∫ x·y dl = L·(2x1y1 + x1y2 + x2y1 + 2x2y2)/6
            Ix += a * L * (p1[1] ** 2 + p1[1] * p2[1] + p2[1] ** 2) / 3.0
            Iy += a * L * (p1[0] ** 2 + p1[0] * p2[0] + p2[0] ** 2) / 3.0
            Ixy += a * L * (2 * p1[0] * p1[1] + p1[0] * p2[1] + p2[0] * p1[1] + 2 * p2[0] * p2[1]) / 6.0
        return {"A_w": A, "xG": float(G[0]), "yG": float(G[1]), "Ix": Ix, "Iy": Iy, "Ixy": Ixy, "J": Ix + Iy,
                "l_total": sum(c.l_eff for c in cs), "n_cordons": len(cs)}

    # ------------------------------------------------------------ répartition
    def forces(self, e: Efforts, s_list: Sequence[float] = (0.0, 0.5, 1.0)) -> List[Dict]:
        """Force par unité de longueur (N/mm) aux points s de chaque cordon utile,
        et ses composantes sur la gorge. Lève ValueError si un moment de
        flexion sollicite un groupe sans inertie autour de cet axe."""
        pr = self.proprietes()
        G = np.array([pr["xG"], pr["yG"], 0.0])
        F = e.F()
        MG = e.M() + np.cross(np.asarray(e.reference, float) - G, F)
        Ix, Iy, Ixy, J, A = pr["Ix"], pr["Iy"], pr["Ixy"], pr["J"], pr["A_w"]
        det = Ix * Iy - Ixy * Ixy
        if abs(MG[0]) > 1e-9 or abs(MG[1]) > 1e-9:
            if det <= 1e-9 * max(1.0, (Ix + Iy) ** 2):
                # cordons tous alignés : flexion possible seulement autour de l'axe perpendiculaire à la ligne
                if Ix <= 1e-9 and abs(MG[0]) > 1e-9:
                    raise ValueError("cordons alignés sur x : le groupe n'a pas d'inertie pour M_x (flexion autour de son propre axe)")
                if Iy <= 1e-9 and abs(MG[1]) > 1e-9:
                    raise ValueError("cordons alignés sur y : le groupe n'a pas d'inertie pour M_y (flexion autour de son propre axe)")
        if abs(MG[2]) > 1e-9 and J <= 1e-9:
            raise ValueError("un seul point de cordon : pas d'inertie polaire pour M_z")
        out = []
        for c in self.utiles:
            for s in s_list:
                P = c.point(s)
                r = P - G
                if det > 1e-9 * max(1.0, (Ix + Iy) ** 2):
                    sig = ((MG[0] * Iy + MG[1] * Ixy) * r[1] - (MG[1] * Ix + MG[0] * Ixy) * r[0]) / det
                else:
                    sig = (MG[0] * r[1] / Ix if Ix > 1e-9 else 0.0) - (MG[1] * r[0] / Iy if Iy > 1e-9 else 0.0)
                tors = (MG[2] / J) * np.cross(np.array([0.0, 0.0, 1.0]), r) if J > 1e-9 else np.zeros(3)
                f = c.a * (F / A + np.array([0.0, 0.0, sig]) + tors) / self.beta_Lw
                comp = c.composantes(f)
                out.append({"cordon": c, "s": s, "point": P, "f_vec": f, **comp})
        return out

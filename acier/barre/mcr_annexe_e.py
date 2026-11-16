# -*- coding: utf-8 -*-
"""Le CONTRÔLE ANALYTIQUE du moment critique de déversement : l'annexe E ANB de la NBN EN 1993-1-1 ANB:2018, pour une
section à deux axes de symétrie (z_j = 0). Le solveur aux éléments finis (acier/barre/mcr.py) reste la RÉFÉRENCE :
l'annexe E admet elle-même les méthodes avancées (1(1) et 2(1)) ; ses formules sont approchées et données du côté sûr
(annexe E ANB, 2(1)) — le cas R1 montre une exception de +0,94 %. docs/acier/BARRES_ACIER.md § 5.6 ; brief § 7.7, cas R1
du § 11.3, tests 20 et 21 du § 11.5.

    (E.8)  M_cr = C₁·π²·E·I_z/(k_z·L)²·{√[(k_z/k_ω)²·I_w/I_z + (k_z·L)²·G·I_t/(π²·E·I_z) + (C₂·z_g)²] − C₂·z_g}
           z_g = z_a − z_s : le point d'application de la charge au-dessus du centre de cisaillement (> 0 : déstabilisant)

Coefficients, k_z = k_ω = 1 seulement (les conditions d'extrémité courantes, 2(3)) :
    tableau E.1 ANB, moments d'extrémité (C₂·z_g = 0), C₁ selon ψ = 1 ; 0,75 ; 0,5 ; 0,25 ; 0 ; −0,25 ; −0,5 ; −0,75 ; −1 :
        1,00 ; 1,14 ; 1,31 ; 1,52 ; 1,77 ; 2,06 ; 2,35 ; 2,60 ; 2,60 — entre deux lignes, l'équation (E.2) de 2(6) :
        C₁ = 1,77 − 1,04·ψ + 0,27·ψ² ≤ 2,60 (expression approchée, annexe E ANB 2(6), valable pour les cas du tableau E.1 ANB)
    tableau E.2 ANB, charges transversales : répartie C₁ = 1,12, C₂ = 0,45 ; concentrée à mi-portée 1,35 et 0,59 ;
        deux charges aux quarts 1,04 et 0,42
    note b) des tableaux : en moments d'extrémité seulement, C₁ divisé par 1,05 si K_ωt = π/(k_ω·L)·√(E·I_w/(G·I_t)) ≤ 1,
        sans descendre sous 1
    note c) : valables pour les I avec −0,9 ≤ ψ_f ≤ 0,9 (ψ_f = 0 pour une section bisymétrique) — pas pour les T.

Unités : N et mm (E·I en N·mm², M_cr en N·mm). Un autre k_z, k_ω, ou un diagramme hors des tableaux : ValueError (le
solveur seul ; aucune moyenne de C₁ et C₂ n'est faite — brief § 7.7).
"""
import math
from typing import Dict, Optional

MOMENTS = "moments"                              # moments d'extrémité, ψ = M₂/M₁
REPARTI, PONCTUEL, DEUX_QUARTS = "reparti", "ponctuel", "deux_quarts"
TABLEAU_E1 = {1.0: 1.00, 0.75: 1.14, 0.5: 1.31, 0.25: 1.52, 0.0: 1.77, -0.25: 2.06, -0.5: 2.35, -0.75: 2.60, -1.0: 2.60}
TABLEAU_E2 = {REPARTI: (1.12, 0.45), PONCTUEL: (1.35, 0.59), DEUX_QUARTS: (1.04, 0.42)}
C1_MAX_E2 = 2.60                                 # le plafond de l'équation (E.2)
DIVISEUR_NOTE_B = 1.05


def k_wt(L: float, EIw: float, GIt: float, kw: float = 1.0) -> float:
    """K_ωt = π/(k_ω·L)·√(E·I_w/(G·I_t)) (note b des tableaux E.1 ANB et E.2 ANB)."""
    return math.pi / (kw * L) * math.sqrt(EIw / GIt)


def coefficients(cas: str, psi: Optional[float] = None, Kwt: Optional[float] = None) -> Dict:
    """{"C1", "C2", "source"} pour ``cas`` (MOMENTS avec ``psi``, ou REPARTI, PONCTUEL, DEUX_QUARTS), k_z = k_ω = 1.
    En moments d'extrémité, ``Kwt`` donné : la note b (C₁/1,05 si K_ωt ≤ 1, jamais sous 1)."""
    if cas == MOMENTS:
        if psi is None or not -1.0 <= psi <= 1.0:
            raise ValueError("moments d'extrémité : ψ entre −1 et 1")
        cle = next((k for k in TABLEAU_E1 if abs(k - psi) < 1e-9), None)
        if cle is not None:
            C1, source = TABLEAU_E1[cle], "tableau E.1 ANB"
        else:
            C1, source = min(1.77 - 1.04 * psi + 0.27 * psi ** 2, C1_MAX_E2), "équation (E.2) de l'annexe E ANB"
        if Kwt is not None and Kwt <= 1.0:
            C1, source = max(C1 / DIVISEUR_NOTE_B, 1.0), source + ", note b (C₁/1,05, K_ωt ≤ 1)"
        return {"C1": C1, "C2": 0.0, "source": source}
    if cas in TABLEAU_E2:
        C1, C2 = TABLEAU_E2[cas]
        return {"C1": C1, "C2": C2, "source": "tableau E.2 ANB"}
    raise ValueError(f"diagramme hors des tableaux E.1 ANB et E.2 ANB : {cas!r} — le solveur seul")


def moment_critique(L: float, EIz: float, EIw: float, GIt: float, C1: float, C2: float = 0.0, zg: float = 0.0,
                    kz: float = 1.0, kw: float = 1.0) -> float:
    """(E.8) : M_cr d'une poutre bisymétrique de portée ``L`` entre restreintes latérales (N·mm)."""
    if kz != 1.0 or kw != 1.0:
        raise ValueError("k_z = k_ω = 1 seulement (les autres valeurs des tableaux ne sont pas codées)")
    if not (L > 0 and EIz > 0 and GIt > 0 and EIw >= 0):
        raise ValueError("L, E·I_z, G·I_t positifs ; E·I_w positif ou nul")
    a = C2 * zg
    return C1 * math.pi ** 2 * EIz / (kz * L) ** 2 * (math.sqrt((kz / kw) ** 2 * EIw / EIz + (kz * L) ** 2 * GIt / (math.pi ** 2 * EIz) + a ** 2) - a)


def controle(cas: str, L: float, EIz: float, EIw: float, GIt: float, psi: Optional[float] = None, zg: float = 0.0) -> Dict:
    """Le contrôle complet d'un cas des tableaux : {"M_cr", "C1", "C2", "K_wt", "source"} (note b appliquée aux moments
    d'extrémité ; z_g sans effet sur eux : C₂·z_g = 0)."""
    Kwt = k_wt(L, EIw, GIt)
    co = coefficients(cas, psi, Kwt)
    Mcr = moment_critique(L, EIz, EIw, GIt, co["C1"], co["C2"], zg if cas != MOMENTS else 0.0)
    return {"M_cr": Mcr, "C1": co["C1"], "C2": co["C2"], "K_wt": Kwt, "source": co["source"]}

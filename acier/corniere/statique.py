# -*- coding: utf-8 -*-
"""Cornière d'appui — la STATIQUE d'une combinaison : une position de la charge X′, une position de l'ancrage Y′,
et les efforts M, N, V le long des deux ailes qui en découlent, pour chaque mode et chaque configuration
(``docs/acier/CORNIERE_DECISIONS.md`` § 3.1 à 3.4 ; équations de ``docs/acier/corniere_excel/synthese``).

Bibliothèque standard (math) seulement. Unités : mm, kN, kN·mm pour les moments internes, N/mm² pour les contraintes.

Repère commun aux deux configurations : sur chaque aile, ``s`` est la DISTANCE AU DOS DE L'AUTRE AILE — sur l'aile
horizontale, depuis le nu du mur (dos de l'aile verticale) ; sur l'aile verticale, depuis la face extérieure de l'aile
horizontale (le dessous en A, le dessus en B). La charge est à s = X′, l'ancrage à s = Y′ dans les deux
configurations (c'est pour cela que Y se compte depuis la face extérieure de l'aile horizontale), le bout libre de
l'aile verticale à s = haut, le contact du mur en B à s = haut − c. Sous s = e : le bloc d'angle, hors théorie des
poutres. Le congé occupe e ≤ s ≤ e + r, où l'épaisseur vaut t(s) = e + r − √(r² − (e + r − s)²).

Signe des moments : M > 0 quand la face côté congé (l'intérieur de l'angle) est tendue. En A, les deux ailes ont
M ≥ 0 (uV = e/2) ; en B, tout est négatif (face du congé comprimée). N > 0 : traction. Les contraintes se calculent
sur la fibre la plus sollicitée : σ = |M|/W + |N|/A, τ = |V|/A, σ_éq = √(σ² + 3τ²).

Les trois statiques :
- A, mode historique (P8 mot pour mot) : aile horizontale M = R·(X′ − s), V = R ; aile verticale M = H_s·(Y′ − s)
  avec H_s = Mmax/(YH + Yj), Mmax = R·X′, YH = Y − e/2 (Y nominal), Yj = Y′ − Y ; le « τ » du classeur R/(L·t) dans
  les deux ailes (V = R), pas d'effort normal ; la traction totale des ancrages Mmax/(Y − jeu_ancrage) (C82·n).
- A, mode équilibre (un corps libre, pivot y_c, cisaillement d'ancrage à u_V) : H = R·(X′ − u_V)/(Y′ − y_c) ;
  aile verticale M = H·(Y′ − s) − R·(e/2 − u_V), N = +R, V = H sous l'ancrage ; rien au-dessus.
- B (toujours l'équilibre) : contact au bout libre, à c = y_c du bout ; y_a = haut − Y′ ; H = R·(X′ − u_V)/(y_a − c) ;
  aile verticale entre l'angle et l'ancrage (e ≤ s ≤ Y′) : M = −R·(X′ − e/2) constant (statique de la partie
  au-dessus de la coupe, où seule la charge agit — avec u_V = e/2 c'est le R·(X′ − u_V) des décisions, le saut
  R·(e/2 − u_V) à l'ancrage étant le couple du cisaillement d'ancrage excentré), N = −R, V = 0 ; entre l'ancrage et
  le contact (Y′ < s ≤ haut − c) : M = −H·(haut − c − s), V = H, N = 0 ; sous le contact : rien.
"""
import math
from typing import Callable, Dict, List, Optional, Tuple

HISTORIQUE, EQUILIBRE = "historique", "equilibre"
HORIZONTALE, VERTICALE = "horizontale", "verticale"
PAS_CONGE = 0.1                    # mm : pas d'exploration du maximum dans le congé (≤ 0,1 mm exigé), puis la section dorée
PAS_DROIT = 1.0                    # mm : pas d'exploration dans la partie droite (les efforts y sont linéaires)


# ------------------------------------------------------------------ géométrie
def t_conge(s: float, e: float, r: float) -> Optional[float]:
    """Épaisseur de l'aile à la distance ``s`` du dos de l'autre aile : e au-delà du congé (s ≥ e + r),
    e + r − √(r² − (e + r − s)²) dans le congé (e ≤ s ≤ e + r), None sous s = e (bloc d'angle)."""
    if s >= e + r:
        return e
    if s < e:
        return None
    w = e + r - s
    return e + r - math.sqrt(max(r * r - w * w, 0.0))


def module_aire(largeur: float, t: float) -> Tuple[float, float]:
    """W = largeur·t²/6 (mm³) et A = largeur·t (mm²) d'une coupe rectangulaire."""
    return largeur * t * t / 6.0, largeur * t


def contraintes(M: float, N: float, V: float, largeur: float, t: float) -> Tuple[float, float, float]:
    """(σ, τ, σ_éq) en N/mm² sur la fibre la plus sollicitée : σ = |M|/W + |N|/A, τ = |V|/A, σ_éq = √(σ² + 3τ²).
    ``M`` en kN·mm, ``N`` et ``V`` en kN, ``largeur`` et ``t`` en mm."""
    W, A = module_aire(largeur, t)
    sigma = abs(M) * 1e3 / W + abs(N) * 1e3 / A
    tau = abs(V) * 1e3 / A
    return sigma, tau, math.sqrt(sigma * sigma + 3.0 * tau * tau)


# ------------------------------------------------------------------ la statique d'une combinaison
def statique(g: Dict, mode: str, configuration: str, Xp: float, Yp: float) -> Dict:
    """Les efforts d'une combinaison. ``g`` : e, r, larg, haut, R (kN), Y (nominal), yc, uV, jeu_ancrage.
    Rend un dict : Xp, Yp, Mmax (kN·mm), H (kN, traction totale des ancrages = compression du mur C), bras (mm),
    et les fonctions M_h, N_h, V_h (aile horizontale) et M_v, N_v, V_v (aile verticale) de ``s`` ; en historique
    aussi H_sections (le H implicite des sections, Mmax/(YH + Yj)), YH, Yj, rho3, rho4 ; en B aussi ya, M_angle
    (positif, kN·mm), s_contact."""
    e, r, R = g["e"], g["r"], g["R"]
    Mmax = R * Xp
    signe = 1.0 if configuration == "A" else -1.0
    st: Dict = {"Xp": Xp, "Yp": Yp, "Mmax": Mmax, "mode": mode, "configuration": configuration,
                "M_h": lambda s: signe * R * (Xp - s) if s <= Xp else 0.0,
                "V_h": lambda s: R if s <= Xp else 0.0,
                "N_h": lambda s: 0.0}
    if configuration == "A" and mode == HISTORIQUE:
        YH = g["Y"] - e / 2.0
        Yj = Yp - g["Y"]
        Hs = Mmax / (YH + Yj)
        st.update(H_sections=Hs, YH=YH, Yj=Yj, rho3=(g["Y"] - e - r + Yj) / (YH + Yj), rho4=(g["Y"] - e + Yj) / (YH + Yj),
                  bras=g["Y"] - g["jeu_ancrage"], H=Mmax / (g["Y"] - g["jeu_ancrage"]),
                  M_v=lambda s: Hs * (Yp - s) if s < Yp else 0.0,
                  V_v=lambda s: R if s < Yp else 0.0,
                  N_v=lambda s: 0.0)
    elif configuration == "A":
        uV, yc = g["uV"], g["yc"]
        bras = Yp - yc
        H = R * (Xp - uV) / bras
        st.update(H=H, bras=bras,
                  M_v=lambda s: H * (Yp - s) - R * (e / 2.0 - uV) if s < Yp else 0.0,
                  N_v=lambda s: R if s < Yp else 0.0,
                  V_v=lambda s: H if s < Yp else 0.0)
    else:
        uV, c = g["uV"], g["yc"]
        ya = g["haut"] - Yp
        bras = ya - c
        H = R * (Xp - uV) / bras
        M_angle = R * (Xp - e / 2.0)
        s_contact = g["haut"] - c

        def M_v(s: float) -> float:
            if s <= Yp:
                return -M_angle
            if s <= s_contact:
                return -H * (s_contact - s)
            return 0.0

        st.update(H=H, bras=bras, ya=ya, M_angle=M_angle, s_contact=s_contact, M_v=M_v,
                  N_v=lambda s: -R if s <= Yp else 0.0,
                  V_v=lambda s: H if Yp < s <= s_contact else 0.0)
    st["C"] = st["H"]
    return st


def efforts(st: Dict, aile: str, s: float) -> Tuple[float, float, float]:
    """(M, N, V) de la combinaison ``st`` sur l'aile ``aile`` à l'abscisse ``s``."""
    if aile == HORIZONTALE:
        return st["M_h"](s), st["N_h"](s), st["V_h"](s)
    return st["M_v"](s), st["N_v"](s), st["V_v"](s)


# ------------------------------------------------------------------ le maximum le long d'une aile
def grille(a: float, b: float, pas: float) -> List[float]:
    """Les abscisses de a à b au pas ≤ ``pas`` (bornes comprises)."""
    if b <= a:
        return [a]
    n = max(1, int(math.ceil((b - a) / pas - 1e-9)))
    h = (b - a) / n
    return [a + i * h for i in range(n)] + [b]


def maximum(f: Callable[[float], float], abscisses: List[float], affiner: bool = True) -> Tuple[float, float]:
    """(s, f(s)) du maximum de ``f`` : le meilleur point d'une liste d'abscisses triées, puis — si ``affiner`` — la
    section dorée entre ses deux voisins (f y est lisse ou unimodale : le congé, ou un coude des efforts)."""
    vals = [f(x) for x in abscisses]
    i = max(range(len(abscisses)), key=vals.__getitem__)
    meilleur = (vals[i], abscisses[i])
    if affiner and len(abscisses) > 1:
        lo, hi = abscisses[max(i - 1, 0)], abscisses[min(i + 1, len(abscisses) - 1)]
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        f1, f2 = f(x1), f(x2)
        for _ in range(60):
            if f1 < f2:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + phi * (hi - lo)
                f2 = f(x2)
            else:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - phi * (hi - lo)
                f1 = f(x1)
        xm = (lo + hi) / 2.0
        meilleur = max(meilleur, (f(xm), xm), (f1, x1), (f2, x2))
    return meilleur[1], meilleur[0]


def abscisses_exploration(e: float, r: float, fin: float, coudes: List[float]) -> List[float]:
    """Les abscisses où chercher le maximum sur une aile : le congé au pas PAS_CONGE, la partie droite au pas
    PAS_DROIT, et les coudes des efforts (charge, ancrage, contact) — triées, sans doublon."""
    pts = grille(e, e + r, PAS_CONGE) if r > 0 else [e]
    pts += grille(e + r, fin, PAS_DROIT)
    pts += [x for x in coudes if e <= x <= fin]
    pts += [x + 1e-9 for x in coudes if e <= x + 1e-9 <= fin]   # juste après un coude (la discontinuité de V ou M)
    pts += [x - 1e-9 for x in coudes if e <= x - 1e-9 <= fin]
    return sorted(set(round(p, 9) for p in pts))

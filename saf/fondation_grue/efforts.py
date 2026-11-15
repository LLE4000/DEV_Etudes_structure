# -*- coding: utf-8 -*-
"""Fondation de grue — les EFFORTS ELS pour le principe d'armatures (lot 5, cahier des charges du 15/11/2026, § 7 et
annexe A, ``efforts``).

Aucun calcul en plus : les moments et les efforts tranchants se LISENT dans les solutions déjà calculées pour la contrainte
de contact (``verification.calculer`` : poids propre + plots + réactions, toutes les positions de tous les jeux, sol en
compression seule) — pour chaque k coché « SCIA », puis l'ENVELOPPE. Pas de superposition : chaque cas est un calcul non
linéaire complet. Les formules sont celles du calcul de référence (tests/fondation_grue_ref.py, annexe A : ``champs_noeuds``,
``integrer_bande``, ``efforts``) ; tests/test_saf_grue_efforts.py vérifie l'égalité cas par cas. PAS DE TORSION : les
moments sont mx et my, comme la carte 2D de SCIA et les bandes (lot 5, § 2 : la torsion est retirée, décision en attente).

    champs_noeuds(d, x)                         -> (M (nn × 3 : mx, my, mxy), V (nn × 2 : vx, vy)) aux nœuds
    integrer_bande(d, val, p0, u, t0, t1, s)    -> l'intégrale sur la largeur EXACTE (bords interpolés), à chaque abscisse
    zones_chargees(modele)                      -> la zone chargée de chaque pied (pour V à d)
    enveloppe(modele, resultats, d_v)           -> les efforts des bandes, les moments au point des nœuds, les extrêmes

Convention : m > 0 = fibre INFÉRIEURE tendue. Unités : kNm/m, kN/m aux nœuds ; kNm, kN intégrés sur une bande.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometrie as G
from .plaque import _base, dans_polygone

_COINS = [(0, 0), (1, 0), (1, 1), (0, 1)]
PAS_S = 0.25                     # le pas en s le long d'une bande : maille / 4 (§ 4.3)


def _trapeze(F: np.ndarray, t: np.ndarray) -> np.ndarray:
    """L'intégrale par les trapèzes le long du dernier axe (numpy.trapezoid, sans dépendre de sa version)."""
    dt = np.diff(t)
    return ((F[..., 1:] + F[..., :-1]) * dt).sum(axis=-1) / 2.0


def _garde(d) -> np.ndarray:
    """La grille des éléments GARDÉS de la dalle (ny × nx)."""
    g = getattr(d, "_garde_grille", None)
    if g is None:
        g = np.zeros((d.ny, d.nx), bool)
        g[d.J, d.I] = True
        d._garde_grille = g
    return g


def champs_noeuds(d, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Moments mx, my, mxy (kNm/m) et efforts tranchants vx, vy (kN/m) aux NŒUDS (§ 4.2). Les moments sont pris aux coins
    des éléments, puis moyennés sur les éléments voisins ; les efforts tranchants au centre des éléments
    (vx = −D (w,xxx + w,xyy), vy = −D (w,yyy + w,xxy)), puis moyennés aux nœuds. Nœuds hors de la dalle : 0."""
    nn = d.ndof // 3
    en = d.edofs[:, 0::3] // 3
    X, Y = d.a / 2, d.b / 2
    z = np.zeros(12)
    pxxx, pxxy, pxyy, pyyy = z.copy(), z.copy(), z.copy(), z.copy()
    pxxx[6], pxxx[10] = 6, 6 * Y
    pxxy[7], pxxy[10] = 2, 6 * X
    pxyy[8], pxyy[11] = 2, 6 * Y
    pyyy[9], pyyy[11] = 6, 6 * X
    ue = x[d.edofs]
    vx = -d.De * (ue @ ((pxxx + pxyy) @ d.Ci))
    vy = -d.De * (ue @ ((pyyy + pxxy) @ d.Ci))
    M = np.zeros((nn, 3))
    V = np.zeros((nn, 2))
    cnt = np.zeros(nn)
    for ci, (u, v) in enumerate(_COINS):
        _, _, _, pxx, pyy, pxy = _base(u * d.a, v * d.b)
        c = ue @ (np.vstack([pxx, pyy, pxy]) @ d.Ci).T
        m = np.stack([-d.De * (c[:, 0] + d.nu * c[:, 1]), -d.De * (c[:, 1] + d.nu * c[:, 0]), -d.De * (1 - d.nu) * c[:, 2]], 1)
        np.add.at(M, en[:, ci], m)
        np.add.at(V, en[:, ci], np.stack([vx, vy], 1))
        np.add.at(cnt, en[:, ci], 1)
    ok = cnt > 0
    M[ok] /= cnt[ok, None]
    V[ok] /= cnt[ok, None]
    return M, V


def _interp(d, val: np.ndarray, X, Y) -> np.ndarray:
    """L'interpolation bilinéaire d'un champ aux nœuds, aux points (X, Y) ; un point hors des éléments de la dalle : 0."""
    nnx = d.nx + 1
    fx = (np.asarray(X, float) - d.x0) / d.a
    fy = (np.asarray(Y, float) - d.y0) / d.b
    i = np.clip(np.floor(fx).astype(int), 0, d.nx - 1)
    j = np.clip(np.floor(fy).astype(int), 0, d.ny - 1)
    u, v = fx - i, fy - j
    dedans = (fx >= -1e-9) & (fx <= d.nx + 1e-9) & (fy >= -1e-9) & (fy <= d.ny + 1e-9) & _garde(d)[j, i]
    n00 = j * nnx + i
    r = (1 - u) * (1 - v) * val[n00] + u * (1 - v) * val[n00 + 1] + u * v * val[n00 + 1 + nnx] + (1 - u) * v * val[n00 + nnx]
    return np.where(dedans, r, 0.0)


def integrer_bande(d, val: np.ndarray, p0, u, t0: float, t1: float, s: np.ndarray, nt=None) -> np.ndarray:
    """L'intégrale, sur la largeur t ∈ [t0, t1] — EXACTE, les bords interpolés (§ 4.3, test T15 : la méthode « centres
    d'éléments dans la bande » perd 20 % à la maille de 0,20 m) —, du champ aux nœuds ``val``, à chaque abscisse s le long
    de l'axe p0 + s·u. Rend un tableau de la taille de ``s``."""
    u = np.asarray(u, float)
    n = np.array([-u[1], u[0]])
    if nt is None:
        nt = 2 * int(np.ceil((t1 - t0) / min(d.a, d.b) * 8)) + 1
    t = np.linspace(t0, t1, nt)
    S, T = np.meshgrid(np.asarray(s, float), t, indexing="ij")
    X = p0[0] + S * u[0] + T * n[0]
    Y = p0[1] + S * u[1] + T * n[1]
    F = _interp(d, val, X.ravel(), Y.ravel()).reshape(S.shape)
    return _trapeze(F, t)


def rot_moments(M: np.ndarray, u) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Les moments dans le repère (u, n) : m_uu, m_nn, m_un (§ 4.3 : m_uu = mx·cos² + my·sin² + 2·mxy·sin·cos)."""
    c, s_ = u[0], u[1]
    mx, my, mxy = M[:, 0], M[:, 1], M[:, 2]
    muu = mx * c * c + my * s_ * s_ + 2 * mxy * c * s_
    mnn = mx * s_ * s_ + my * c * c - 2 * mxy * c * s_
    mun = (my - mx) * c * s_ + mxy * (c * c - s_ * s_)
    return muu, mnn, mun


def zones_chargees(modele: Dict) -> List:
    """La zone chargée de chaque pied (V à d, lot 5, § 6) : la surface d'appui ; sous un plot préfabriqué, le carré de
    diffusion ; sous un plot coulé en place, le plot entier (V à d du nu du plot)."""
    pl = modele.get("plots")
    if pl and pl["type"] == "coule":
        return pl["polys"]
    if pl:
        return pl["diffusion"]
    return modele["plaques"]


PIEDS = {"BX1": (0, 1), "BX2": (3, 2), "BY1": (0, 3), "BY2": (1, 2)}
FACES = ("inf_x", "inf_y", "sup_x", "sup_y")


def lecture_v(s: np.ndarray, charge: Sequence[Tuple[float, float]], d_v: float) -> np.ndarray:
    """Les abscisses où V se lit : à d_v ou plus du bord de chaque zone chargée [(s1, s2)] de la bande."""
    return np.all([(s <= s1 - d_v + 1e-9) | (s >= s2 + d_v - 1e-9) for s1, s2 in charge], axis=0)


def geometrie_bandes(modele: Dict, d_v: float) -> List[Dict]:
    """Pour chaque bande : son axe (p0, u), sa longueur, ses abscisses s (pas maille / 4), les abscisses où V se lit (à d
    ou plus du bord des zones chargées de ses deux pieds), et celles où l'axe est dans un plot coulé en place."""
    zch = zones_chargees(modele)
    pl = modele.get("plots")
    coules = pl["polys"] if pl and pl["type"] == "coule" else []
    pas = modele.get("maille", 0.25) * PAS_S
    out = []
    for bd in modele["bandes"]:
        p0, p1 = np.asarray(bd["debut"], float), np.asarray(bd["fin"], float)
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / L
        s = np.arange(0.0, L + 1e-9, pas)
        inter = []
        for i in PIEDS[bd["nom"]]:
            proj = [float((np.asarray(v, float) - p0) @ u) for v in zch[i]]
            inter.append((min(proj), max(proj)))
        dans_pl = np.zeros(len(s), bool)                 # l'axe de la bande dans un plot coulé en place
        for q in coules:
            dans_pl |= dans_polygone(p0[0] + s * u[0], p0[1] + s * u[1], q)
        out.append({"bande": bd, "p0": p0, "u": u, "L": L, "s": s, "ok_v": lecture_v(s, inter, d_v), "charge": inter,
                    "dans_plot": dans_pl})
    return out


def _actifs(modele: Dict, d) -> np.ndarray:
    """Les nœuds de la dalle hors des plots coulés en place (nny × nnx) : un nœud STRICTEMENT dans un plot coulé (le plot
    rétréci de 1e-6 m) est exclu — la section y est plus haute (annexe A)."""
    nnx, nny = d.nx + 1, d.ny + 1
    actif = np.zeros(nnx * nny, bool)
    actif[np.unique(d.edofs[:, 0::3] // 3)] = True
    pl = modele.get("plots")
    if pl and pl["type"] == "coule":
        X = d.x0 + (np.arange(nnx * nny) % nnx) * d.a
        Y = d.y0 + (np.arange(nnx * nny) // nnx) * d.b
        for q in pl["polys"]:
            actif &= ~dans_polygone(X, Y, G.rectangle(*G.centre_gravite(q), pl["a"] - 1e-6, pl["b"] - 1e-6, modele.get("beta", 0.0)))
    return actif.reshape(nny, nnx)


EGALITE_CAS = 1e-6              # deux cas « égaux » à 10⁻⁶ près (relatif) : le bruit du calcul entre deux positions symétriques


def _mieux(rb: Dict, cle: str, v: float, plus: bool, cas: str, k: float) -> None:
    """Retient (valeur, cas, k) si ``v`` dépasse franchement la valeur gardée — à égalité (10⁻⁶ près, relatif : deux positions
    symétriques diffèrent du bruit du calcul, qui n'est pas celui de l'annexe), le PREMIER cas dans l'ordre des jeux, des
    positions et des k garde la place, comme dans l'annexe A."""
    ref = rb["dim"][cle][0]
    tol = EGALITE_CAS * max(1.0, abs(ref)) if np.isfinite(ref) else 0.0
    if (v > ref + tol) if plus else (v < ref - tol):
        rb["dim"][cle] = (float(v), cas, k)


def enveloppe(modele: Dict, resultats: Sequence[Dict], d_v: float) -> Optional[Dict]:
    """Les efforts ELS de la fondation, en ENVELOPPE de tous les cas de tous les ``resultats`` (un par k coché « SCIA » :
    ``verification.calculer``, avec ses cas et les déplacements de chaque cas). ``d_v`` : la distance à laquelle V se lit
    du bord des zones chargées (h − 6 cm, la valeur de l'annexe A).

    Rend {"bandes": [par bande : "nom", "debut", "fin", "s" (m), "Mmax", "Mmin" (kNm, intégrés sur la largeur), "Vabs" (kN),
           "ok_v", "dans_plot", "charge", "largeur", "p0", "u", "L" ; "M_plus", "M_moins" (hors des plots coulés : au nu du
           plot), "M_plus_plot", "M_moins_plot" (dans les plots coulés, pour information, ou None), "V_d" (le plus grand |V|
           à d_v ou plus des zones chargées ; None quand AUCUNE abscisse n'y est — V à d ne se lit pas, ce qui n'est pas
           « V = 0 »), "cas_M_plus", "cas_M_moins", "cas_V_d" ((nom du cas, k) de chaque valeur, ou None)],
          "noeuds": {"inf_x" (mx maxi), "sup_x" (mx mini), "inf_y", "sup_y"} (kNm/m, nny × nnx : les moments au point, comme
                    la carte 2D de SCIA),
          "extremes": {"inf_x", "inf_y" (maxi), "sup_x", "sup_y" (mini)} (hors des plots coulés), "cas_extremes" (idem :
                    (cas, k)),
          "grille": {"nx", "ny" (nœuds par ligne et par colonne), "x0", "y0", "a", "b", "actif"}, "pieds", "largeur",
          "d_v", "ks"} — None sans résultat."""
    res = None
    for r in resultats:
        d = r["dalle"]
        geo = geometrie_bandes(modele, d_v)
        nnx, nny = d.nx + 1, d.ny + 1
        actif = _actifs(modele, d)
        if res is None:
            plus, moins = np.full((nny, nnx), -np.inf), np.full((nny, nnx), np.inf)
            res = {"bandes": [{"nom": g["bande"]["nom"], "debut": g["bande"]["debut"], "fin": g["bande"]["fin"], "s": g["s"],
                               "Mmax": np.full(len(g["s"]), -np.inf), "Mmin": np.full(len(g["s"]), np.inf),
                               "Vabs": np.zeros(len(g["s"])), "ok_v": g["ok_v"], "dans_plot": g["dans_plot"], "charge": g["charge"],
                               "largeur": g["bande"]["largeur"], "p0": g["p0"], "u": g["u"], "L": g["L"],
                               "dim": {"M_plus": (-np.inf, None, None), "M_moins": (np.inf, None, None), "V_d": (-np.inf, None, None)}}
                              for g in geo],
                   "noeuds": {"inf_x": plus.copy(), "sup_x": moins.copy(), "inf_y": plus.copy(), "sup_y": moins.copy()},
                   "extremes": {"inf_x": -np.inf, "inf_y": -np.inf, "sup_x": np.inf, "sup_y": np.inf},
                   "cas_extremes": {k_: None for k_ in FACES},
                   "grille": {"nx": nnx, "ny": nny, "x0": d.x0, "y0": d.y0, "a": d.a, "b": d.b, "actif": actif},
                   "pieds": [G.centre_gravite(p) for p in modele["plaques"]], "largeur": modele["bandes"][0]["largeur"] if modele["bandes"] else None,
                   "d_v": d_v, "ks": []}
        res["ks"].append(r["k"])
        no, ex, cx = res["noeuds"], res["extremes"], res["cas_extremes"]
        # (les cas parcourus JEU PAR JEU, dans l'ordre des positions — l'ordre de l'annexe A : à égalité, deux positions
        # symétriques (3.b et 3.g) donnent la même valeur, et le cas dimensionnant retenu est le premier de cet ordre)
        for c, x in sorted(zip(r["cas"], r["deplacements"]), key=lambda t: (t[0]["jeu"], t[0]["nom"].split(".")[-1])):
            M, V = champs_noeuds(d, x)
            for rb, g in zip(res["bandes"], geo):
                u, p0, b = g["u"], g["p0"], g["bande"]["largeur"]
                muu, _, _ = rot_moments(M, u)
                Mi = integrer_bande(d, muu, p0, u, -b / 2, b / 2, g["s"])
                Vi = integrer_bande(d, V[:, 0] * u[0] + V[:, 1] * u[1], p0, u, -b / 2, b / 2, g["s"])
                rb["Mmax"] = np.maximum(rb["Mmax"], Mi)
                rb["Mmin"] = np.minimum(rb["Mmin"], Mi)
                rb["Vabs"] = np.maximum(rb["Vabs"], np.abs(Vi))
                hp = ~g["dans_plot"]
                if hp.any():
                    _mieux(rb, "M_plus", float(Mi[hp].max()), True, c["nom"], r["k"])
                    _mieux(rb, "M_moins", float(Mi[hp].min()), False, c["nom"], r["k"])
                if g["ok_v"].any():
                    _mieux(rb, "V_d", float(np.abs(Vi)[g["ok_v"]].max()), True, c["nom"], r["k"])
            mx, my = M[:, 0].reshape(nny, nnx), M[:, 1].reshape(nny, nnx)
            no["inf_x"] = np.maximum(no["inf_x"], mx)
            no["sup_x"] = np.minimum(no["sup_x"], mx)
            no["inf_y"] = np.maximum(no["inf_y"], my)
            no["sup_y"] = np.minimum(no["sup_y"], my)
            if actif.any():
                for cle, m_, plus_ in (("inf_x", mx, True), ("sup_x", mx, False), ("inf_y", my, True), ("sup_y", my, False)):
                    v = float(m_[actif].max() if plus_ else m_[actif].min())
                    if (v > ex[cle] + 1e-9) if plus_ else (v < ex[cle] - 1e-9):
                        ex[cle], cx[cle] = v, (c["nom"], r["k"])
    if res is None:
        return None
    for rb in res["bandes"]:
        hp, dp = ~rb["dans_plot"], rb["dans_plot"]
        rb["M_plus"] = float(rb["Mmax"][hp].max()) if hp.any() else None
        rb["M_moins"] = float(rb["Mmin"][hp].min()) if hp.any() else None
        rb["M_plus_plot"] = float(rb["Mmax"][dp].max()) if dp.any() else None
        rb["M_moins_plot"] = float(rb["Mmin"][dp].min()) if dp.any() else None
        rb["V_d"] = float(rb["Vabs"][rb["ok_v"]].max()) if rb["ok_v"].any() else None
        dim = rb.pop("dim")
        rb["cas_M_plus"], rb["cas_M_moins"], rb["cas_V_d"] = (dim[c][1:] if dim[c][1] is not None else None for c in ("M_plus", "M_moins", "V_d"))
    return res

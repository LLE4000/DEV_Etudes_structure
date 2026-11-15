# -*- coding: utf-8 -*-
"""Fondation de grue — les EFFORTS ELS pour le ferraillage (lot 4, § 4 du cahier des charges du 15/11/2026).

Aucun calcul en plus : les moments et les efforts tranchants se LISENT dans les solutions déjà calculées pour la contrainte
de contact (``verification.calculer`` : poids propre + plots + réactions, toutes les positions de tous les jeux, sol en
compression seule) — pour chaque k coché « SCIA », puis l'ENVELOPPE. Pas de superposition : chaque cas est un calcul non
linéaire complet. Les formules sont celles du calcul de référence (tests/fondation_grue_ref.py, annexe A : ``champs_noeuds``,
``integrer_bande``, ``zones_renfort``, ``efforts``) ; tests/test_saf_grue_efforts.py vérifie l'égalité cas par cas.

    champs_noeuds(d, x)                         -> (M (nn × 3 : mx, my, mxy), V (nn × 2 : vx, vy)) aux nœuds
    integrer_bande(d, val, p0, u, t0, t1, s)    -> l'intégrale sur la largeur EXACTE (bords interpolés), à chaque abscisse
    zones_renfort(contour, bandes)              -> [(t_min, t_max)] : la bande, prolongée jusqu'au bord si la lisière
                                                   restante est plus étroite qu'une bande
    zones_chargees(modele)                      -> la zone chargée de chaque pied (pour V à d)
    enveloppe(modele, resultats, d_v, torsion)  -> les efforts des bandes, de la zone courante et le champ des moments

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


def moments_centres(d, x: np.ndarray) -> np.ndarray:
    """mx, my, mxy au CENTRE des éléments (kNm/m), chaque élément avec sa propre épaisseur (n_el × 3)."""
    Bc = getattr(d, "_Bc", None)
    if Bc is None:
        _, _, _, pxx, pyy, pxy = _base(d.a / 2, d.b / 2)
        Bc = d._Bc = np.vstack([pxx, pyy, pxy]) @ d.Ci
    c = x[d.edofs] @ Bc.T
    return np.stack([-d.De * (c[:, 0] + d.nu * c[:, 1]), -d.De * (c[:, 1] + d.nu * c[:, 0]), -d.De * (1 - d.nu) * c[:, 2]], 1)


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


def _etendue(contour: Sequence, p, n, smax: float = 50.0) -> float:
    """La distance du point p au contour dans la direction n (le premier bord rencontré)."""
    best = smax
    m = len(contour)
    for i in range(m):
        a, b = np.asarray(contour[i], float), np.asarray(contour[(i + 1) % m], float)
        e = b - a
        den = n[0] * (-e[1]) - n[1] * (-e[0])
        if abs(den) < 1e-12:
            continue
        r = a - p
        tt = (r[0] * (-e[1]) - r[1] * (-e[0])) / den
        uu = (n[0] * r[1] - n[1] * r[0]) / den
        if tt > 1e-9 and -1e-9 <= uu <= 1 + 1e-9:
            best = min(best, tt)
    return best


def zones_renfort(contour: Sequence, bandes: Sequence[Dict]) -> List[Tuple[float, float]]:
    """La zone de renfort de chaque bande (§ 4.5, test T17) : la bande, prolongée jusqu'au bord de la dalle d'un côté si,
    sur toute la longueur de la bande, la dalle s'arrête à moins d'une largeur de bande au-delà de son bord (pas de
    lisière étroite couverte par le seul treillis). Rend [(t_min, t_max)] par bande (t : la distance à l'axe, vers la
    gauche de l'axe positive)."""
    out = []
    for bd in bandes:
        p0, p1 = np.asarray(bd["debut"], float), np.asarray(bd["fin"], float)
        L = np.linalg.norm(p1 - p0)
        u = (p1 - p0) / L
        n = np.array([-u[1], u[0]])
        b = bd["largeur"]
        lim = []
        for sg in (-1, 1):
            ext = [_etendue(contour, p0 + s * u + sg * (b / 2) * n, sg * n) for s in np.linspace(0.02 * L, 0.98 * L, 25)]
            e = max(ext)
            lim.append(sg * (b / 2 + e) if e < b - 1e-9 else sg * b / 2)
        out.append((lim[0], lim[1]))
    return out


def rot_moments(M: np.ndarray, u) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Les moments dans le repère (u, n) : m_uu, m_nn, m_un (§ 4.3 : m_uu = mx·cos² + my·sin² + 2·mxy·sin·cos)."""
    c, s_ = u[0], u[1]
    mx, my, mxy = M[:, 0], M[:, 1], M[:, 2]
    muu = mx * c * c + my * s_ * s_ + 2 * mxy * c * s_
    mnn = mx * s_ * s_ + my * c * c - 2 * mxy * c * s_
    mun = (my - mx) * c * s_ + mxy * (c * c - s_ * s_)
    return muu, mnn, mun


def zones_chargees(modele: Dict) -> List:
    """La zone chargée de chaque pied (§ 4.3, V à d) : la surface d'appui ; sous un plot préfabriqué, le carré de
    diffusion ; sous un plot coulé en place, le plot."""
    pl = modele.get("plots")
    if pl and pl["type"] == "coule":
        return pl["polys"]
    if pl and pl["type"] == "prefabrique":
        return pl["diffusion"]
    return modele["plaques"]


PIEDS = {"BX1": (0, 1), "BX2": (3, 2), "BY1": (0, 3), "BY2": (1, 2)}


def geometrie_bandes(modele: Dict, d, d_v: float) -> List[Dict]:
    """Pour chaque bande : son axe (p0, u), sa longueur, ses abscisses s (pas maille / 4), sa zone de renfort, et les
    abscisses où V se lit (à d ou plus du bord des zones chargées de ses deux pieds)."""
    zr = zones_renfort(modele["contour"], modele["bandes"])
    zch = zones_chargees(modele)
    pas = modele.get("maille", 0.25) * PAS_S
    out = []
    for bd, z in zip(modele["bandes"], zr):
        p0, p1 = np.asarray(bd["debut"], float), np.asarray(bd["fin"], float)
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / L
        s = np.arange(0.0, L + 1e-9, pas)
        inter = []
        for i in PIEDS[bd["nom"]]:
            proj = [float((np.asarray(v, float) - p0) @ u) for v in zch[i]]
            inter.append((min(proj), max(proj)))
        ok_v = lecture_v(s, inter, d_v)
        out.append({"bande": bd, "zone": z, "p0": p0, "u": u, "L": L, "s": s, "ok_v": ok_v, "charge": inter})
    return out


def lecture_v(s: np.ndarray, charge: Sequence[Tuple[float, float]], d_v: float) -> np.ndarray:
    """Les abscisses où V se lit (§ 4.3) : à d_v ou plus du bord de chaque zone chargée [(s1, s2)] de la bande."""
    return np.all([(s <= s1 - d_v + 1e-9) | (s >= s2 + d_v - 1e-9) for s1, s2 in charge], axis=0)


def v_a_d(bande: Dict, d_v: float) -> Optional[float]:
    """V à d d'une bande de l'enveloppe, RELU pour une autre hauteur utile d_v (celle que le moteur poutre donne pour la
    section de la bande, § 4.3) : le plus grand |V(s)| aux abscisses à d_v ou plus du bord des zones chargées ; None
    quand AUCUNE abscisse n'y est (des plots coulés plus proches que 2d, sans porte-à-faux) — V à d ne se lit pas, ce
    qui n'est pas « V = 0 »."""
    ok = lecture_v(bande["s"], bande["charge"], d_v)
    return float(bande["Vabs"][ok].max()) if ok.any() else None


def enveloppe(modele: Dict, resultats: Sequence[Dict], d_v: float, torsion: bool = True) -> Dict:
    """Les efforts ELS de la fondation, en ENVELOPPE de tous les cas de tous les ``resultats`` (un par k coché « SCIA » :
    ``verification.calculer``, avec les déplacements de chaque cas). ``d_v`` : la hauteur utile qui borne la lecture de
    V à d (le moteur poutre la donne pour la section de la bande). ``torsion`` : les moments de calcul du treillis avec
    |mxy| (§ 4.4, oui par défaut).

    Rend {"bandes": [par bande : "nom", "s" (m), "Mmax", "Mmin" (kNm, intégrés), "Vabs" (kN), "M_plus", "M_moins", "V_d",
           "zone" (t_min, t_max), "prol_inf", "prol_sup" (kNm/m, dans le prolongement jusqu'au bord), "largeur", "p0", "u",
           "L", "ok_v", "charge" (les intervalles chargés le long de l'axe)],
          "courante": {"inf_x", "inf_y" (max), "sup_x", "sup_y" (min), "plot_inf", "plot_sup"} (kNm/m, hors zones de renfort —
          l'annexe A ; le ferraillage, lui, contrôle chaque direction hors des seules zones de SA direction : zone_x, zone_y),
          "champ": {"x", "y", "h", "zone_renfort", "zone_x", "zone_y" (les zones des BX, des BY), "plot", "inf_x", "inf_y", "sup_x", "sup_y", "mx_max", "my_max",
                    "mx_min", "my_min"} (au centre des éléments), "d_v", "torsion", "ks"}"""
    res = None
    pl = modele.get("plots")
    for r in resultats:
        d = r["dalle"]
        geo = geometrie_bandes(modele, d, d_v)
        xc = d.x0 + (d.I + 0.5) * d.a
        yc = d.y0 + (d.J + 0.5) * d.b
        dans_zone = np.zeros(len(xc), bool)
        zone_sens = {"x": np.zeros(len(xc), bool), "y": np.zeros(len(xc), bool)}
        prol = []
        for g in geo:
            u, p0, z = g["u"], g["p0"], g["zone"]
            n = np.array([-u[1], u[0]])
            S = (xc - p0[0]) * u[0] + (yc - p0[1]) * u[1]
            T = (xc - p0[0]) * n[0] + (yc - p0[1]) * n[1]
            dz = (S >= -1e-6) & (S <= g["s"][-1] + 1e-6) & (T >= z[0] - 1e-6) & (T <= z[1] + 1e-6)
            dans_zone |= dz
            # (une bande BX ne pose que des barres en x, une BY qu'en y : dans la zone d'une BX, les barres en y restent
            # au treillis — elles se contrôlent élément par élément comme la zone courante)
            zone_sens["x" if g["bande"]["nom"].startswith("BX") else "y"] |= dz
            prol.append(dz & (np.abs(T) > g["bande"]["largeur"] / 2 + 1e-6))
        dans_plot = np.zeros(len(xc), bool)
        if pl and pl["type"] == "coule":
            for q in pl["polys"]:
                dans_plot |= dans_polygone(xc, yc, q)
        cour = ~dans_zone
        if res is None:
            nul = lambda: np.zeros(len(xc))  # noqa: E731
            res = {"bandes": [{"nom": g["bande"]["nom"], "s": g["s"], "Mmax": np.full(len(g["s"]), -np.inf), "Mmin": np.full(len(g["s"]), np.inf),
                               "Vabs": np.zeros(len(g["s"])), "ok_v": g["ok_v"], "zone": g["zone"], "prol_inf": 0.0, "prol_sup": 0.0,
                               "largeur": g["bande"]["largeur"], "p0": g["p0"], "u": g["u"], "L": g["L"], "charge": g["charge"],
                               "debut": g["bande"]["debut"], "fin": g["bande"]["fin"]} for g in geo],
                   "courante": {"inf_x": 0.0, "inf_y": 0.0, "sup_x": 0.0, "sup_y": 0.0, "plot_inf": 0.0, "plot_sup": 0.0},
                   "champ": {"x": xc, "y": yc, "h": modele["h"] + np.where(dans_plot, pl["hauteur"] if dans_plot.any() else 0.0, 0.0),
                             "zone_renfort": dans_zone, "zone_x": zone_sens["x"], "zone_y": zone_sens["y"], "plot": dans_plot, "inf_x": nul(), "inf_y": nul(), "sup_x": nul(), "sup_y": nul(),
                             "mx_max": nul(), "my_max": nul(), "mx_min": nul(), "my_min": nul(), "I": d.I, "J": d.J,
                             "a": d.a, "b": d.b, "x0": d.x0, "y0": d.y0},
                   "d_v": d_v, "torsion": torsion, "ks": []}
        res["ks"].append(r["k"])
        for x in r["deplacements"]:
            M, V = champs_noeuds(d, x)
            for rb, g in zip(res["bandes"], geo):
                u, p0, b = g["u"], g["p0"], g["bande"]["largeur"]
                muu, _, _ = rot_moments(M, u)
                Mi = integrer_bande(d, muu, p0, u, -b / 2, b / 2, g["s"])
                Vi = integrer_bande(d, V[:, 0] * u[0] + V[:, 1] * u[1], p0, u, -b / 2, b / 2, g["s"])
                rb["Mmax"] = np.maximum(rb["Mmax"], Mi)
                rb["Mmin"] = np.minimum(rb["Mmin"], Mi)
                rb["Vabs"] = np.maximum(rb["Vabs"], np.abs(Vi))
            Me = moments_centres(d, x)
            for rb, g, pr in zip(res["bandes"], geo, prol):
                if pr.any():
                    muu, _, mun = rot_moments(Me, g["u"])
                    t_ = np.abs(mun) if torsion else 0.0
                    rb["prol_inf"] = max(rb["prol_inf"], float((muu + t_)[pr].max()))
                    rb["prol_sup"] = min(rb["prol_sup"], float((muu - t_)[pr].min()))
            ax = np.abs(Me[:, 2]) if torsion else np.zeros(len(Me))
            ch = res["champ"]
            ch["inf_x"] = np.maximum(ch["inf_x"], Me[:, 0] + ax)
            ch["inf_y"] = np.maximum(ch["inf_y"], Me[:, 1] + ax)
            ch["sup_x"] = np.minimum(ch["sup_x"], Me[:, 0] - ax)
            ch["sup_y"] = np.minimum(ch["sup_y"], Me[:, 1] - ax)
            ch["mx_max"] = np.maximum(ch["mx_max"], Me[:, 0])
            ch["my_max"] = np.maximum(ch["my_max"], Me[:, 1])
            ch["mx_min"] = np.minimum(ch["mx_min"], Me[:, 0])
            ch["my_min"] = np.minimum(ch["my_min"], Me[:, 1])
            c_ = res["courante"]
            for zone_, plot_ in ((cour & ~dans_plot, False), (cour & dans_plot, True)):
                if not zone_.any():
                    continue
                if not plot_:
                    c_["inf_x"] = max(c_["inf_x"], float((Me[:, 0] + ax)[zone_].max()))
                    c_["inf_y"] = max(c_["inf_y"], float((Me[:, 1] + ax)[zone_].max()))
                    c_["sup_x"] = min(c_["sup_x"], float((Me[:, 0] - ax)[zone_].min()))
                    c_["sup_y"] = min(c_["sup_y"], float((Me[:, 1] - ax)[zone_].min()))
                else:
                    c_["plot_inf"] = max(c_["plot_inf"], float((np.maximum(Me[:, 0], Me[:, 1]) + ax)[zone_].max()))
                    c_["plot_sup"] = min(c_["plot_sup"], float((np.minimum(Me[:, 0], Me[:, 1]) - ax)[zone_].min()))
    if res is None:
        return None
    for rb in res["bandes"]:
        rb["M_plus"] = float(rb["Mmax"].max())
        rb["M_moins"] = float(rb["Mmin"].min())
        rb["V_d"] = float(rb["Vabs"][rb["ok_v"]].max()) if rb["ok_v"].any() else None
    return res

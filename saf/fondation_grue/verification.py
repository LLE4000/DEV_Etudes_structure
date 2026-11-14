# -*- coding: utf-8 -*-
"""Fondation de grue — la VÉRIFICATION RAPIDE : la contrainte de contact et le tassement de la dalle, pour toutes les
positions de la grue et plusieurs raideurs de sol. Une ESTIMATION pour fixer la dalle avant SCIA, qui reste la référence.

Charges ELS caractéristiques, comme les combinaisons du SAF : le poids propre (γ·h sur toute la dalle), le poids des
plots, et chaque réaction répartie sur sa surface d'appui — ou sur tout le plot quand il est préfabriqué. Chaque charge
est NORMALISÉE : sa somme vaut exactement la réaction (ou le poids du plot), quelle que soit la forme de la surface.

    calculer(modele, k, c2x=0, c2y=0, sans_traction=True) -> le résultat pour ce k
    par_k(modele, ks, rapport_c2=(0.1, 0.1))               -> [le résultat de chaque k]
    proposer_un(modele_de, ks, limite, …)                   -> le plus petit agrandissement qui tient (« Proposer »)

LE MODÈLE (saf/fondation_grue/projet.py le bâtit depuis le projet) :
    {"contour": [(x, y)…], "h", "E" (MPa), "nu", "maille", "gamma" (kN/m³),
     "plaques": [P1…P4] (les surfaces d'appui), "plots": None | {"type": "prefabrique" | "coule", "polys": [P1…P4],
     "hauteur"}, "cas": [{"nom", "jeu", "R": (R1…R4)}…]}
Sans aucun cas (aucune réaction saisie), le poids propre est calculé seul (cas « PP »).

LE RÉSULTAT d'un k :
    {"k", "cas": [{"nom", "jeu", "R", "sigma_max" (kPa), "w_max" (mm), "decolle" (%), "iterations", "charge_totale",
     "reaction_sol" (kN), "x_max", "y_max" (où σ est maximale)}], "sigma_max", "w_max", "decolle", "cas_max" (le nom
     du cas qui donne σ max), "ecart_equilibre" (le plus grand écart relatif entre la réaction du sol et les charges),
     "dalle" (la dalle maillée), "champs" ([(σ aux nœuds, w aux nœuds)] par cas : pour les cartes)}
"""
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometrie as G
from .plaque import Dalle

KG_CM2 = 98.0665                 # 1 kg/cm² en kPa
EQUILIBRE = 1e-4                 # l'écart admis entre la réaction du sol et les charges (0,01 %)
NOEUDS_MAX = 60000               # au-delà, la maille est trop fine pour un calcul en direct
CACHE_OCTETS = 300e6             # la mémoire laissée aux dalles déjà factorisées
PP = "PP"                        # le nom du cas « poids propre seul »
PREMIER_PAS = 0.5                # « Proposer » : le premier agrandissement essayé (m), doublé tant qu'il ne suffit pas
# Deux grandeurs que la SYMÉTRIE rend égales (deux positions miroirs de la grue, deux nœuds symétriques de la dalle) ne
# diffèrent que par le bruit du calcul en virgule flottante — et ce bruit n'est pas le même d'une machine à l'autre :
# CPython nommait « 3.f » le cas le plus défavorable de la dalle type, le navigateur (Pyodide) « 3.g » (vu par
# tests/test_moteur_partage.py, 14/11/2026). À EGALITE (relative) près, c'est le PREMIER qui l'emporte : le même partout.
EGALITE = 1e-9

_CACHE: "OrderedDict[tuple, Dalle]" = OrderedDict()


def premier_max(valeurs: Sequence[float]) -> int:
    """L'indice de la plus grande valeur ; à égalité (à ``EGALITE`` relative près), le PREMIER — le même résultat sur
    toutes les machines (voir ``EGALITE``)."""
    vals = [float(v) for v in valeurs]
    haut = max(vals)
    seuil = haut - EGALITE * max(1.0, abs(haut))
    return next(i for i, v in enumerate(vals) if v >= seuil)


def premier_argmax(tableau) -> int:
    """``premier_max`` pour un tableau numpy (aplati) : le premier nœud qui atteint le maximum, à ``EGALITE`` près."""
    a = np.asarray(tableau, dtype=float).ravel()
    haut = float(np.max(a))
    return int(np.flatnonzero(a >= haut - EGALITE * max(1.0, abs(haut)))[0])


def _cle(modele: Dict, k: float, c2x: float, c2y: float, zones) -> tuple:
    return (tuple(tuple(p) for p in modele["contour"]), modele["h"], k, modele["E"], modele.get("nu", 0.2), c2x, c2y,
            modele.get("maille", 0.25), tuple((tuple(tuple(q) for q in poly), hz) for poly, hz in zones))


def _octets(d: Dalle) -> float:
    return 8.0 * (2 * d.nb * d.m * d.m + d._Z.size)


def dalle(modele: Dict, k: float, c2x: float = 0.0, c2y: float = 0.0) -> Dalle:
    """La dalle maillée du modèle pour ce k — gardée : déplacer la grue ou changer une réaction ne la refait pas."""
    pl = modele.get("plots")
    zones = [(q, modele["h"] + pl["hauteur"]) for q in pl["polys"]] if pl and pl["type"] == "coule" else []
    cle = _cle(modele, k, c2x, c2y, zones)
    d = _CACHE.get(cle)
    if d is not None:
        _CACHE.move_to_end(cle)
        return d
    xs, ys = [p[0] for p in modele["contour"]], [p[1] for p in modele["contour"]]
    maille = modele.get("maille", 0.25)
    if (round((max(xs) - min(xs)) / maille) + 1) * (round((max(ys) - min(ys)) / maille) + 1) > NOEUDS_MAX:
        raise ValueError("la maille du calcul est trop fine pour cette dalle : l'agrandir dans les options")
    d = Dalle(modele["contour"], modele["h"], k, E_MPa=modele["E"], nu=modele.get("nu", 0.2), c2_MNm=(c2x, c2y), maille=maille, zones=zones)
    _CACHE[cle] = d
    while len(_CACHE) > 1 and sum(_octets(x) for x in _CACHE.values()) > CACHE_OCTETS:
        _CACHE.popitem(last=False)
    return d


def vider_cache() -> None:
    _CACHE.clear()


def calculer(modele: Dict, k: float, c2x: float = 0.0, c2y: float = 0.0, sans_traction: bool = True) -> Dict:
    """Toutes les positions de tous les jeux pour ce k (voir l'en-tête du fichier)."""
    d = dalle(modele, k, c2x, c2y)
    gamma, h = modele.get("gamma", 25.0), modele["h"]
    pl = modele.get("plots")
    surfaces = pl["polys"] if pl and pl["type"] == "prefabrique" else modele["plaques"]
    Fg = d.charge(q_uniforme=gamma * h)
    if pl:
        for q in pl["polys"]:                                      # le poids de chaque plot, somme exacte
            Fg = Fg + d.charge_unitaire(q) * (gamma * pl["hauteur"] * G.aire(q))
    cas = list(modele.get("cas") or [])
    if cas:
        Fp = [d.charge_unitaire(p) for p in surfaces]              # 1 kN exactement sur chaque surface
        Fs = [Fg + sum(float(c["R"][i]) * Fp[i] for i in range(4)) for c in cas]
    else:
        cas, Fs = [{"nom": PP, "jeu": 0, "R": (0.0, 0.0, 0.0, 0.0)}], [Fg]
    sol = d.resoudre(Fs, sans_traction)
    lignes, champs = [], []
    for c, s, F in zip(cas, sol, Fs):
        n = premier_argmax(s["sigma"])
        total = float(F[0::3].sum())
        lignes.append({"nom": c["nom"], "jeu": c["jeu"], "R": tuple(c["R"]), "sigma_max": float(s["sigma"].max()),
                       "sigma_min_lin": float(np.nanmin(s["sigma_lin"])), "w_max": float(np.nanmax(s["w"]) * 1000),
                       "decolle": 100 * s["decolle"], "iterations": s["iterations"], "charge_totale": total,
                       "reaction_sol": float((s["sigma"] * d.area).sum()),
                       "x_max": d.x0 + (n % d.nnx) * d.a, "y_max": d.y0 + (n // d.nnx) * d.b})
        champs.append((s["sigma"], s["w"]))
    pire = lignes[premier_max([x["sigma_max"] for x in lignes])]
    return {"k": k, "cas": lignes, "sigma_max": pire["sigma_max"], "w_max": max(x["w_max"] for x in lignes),
            "decolle": max(x["decolle"] for x in lignes), "cas_max": pire["nom"],
            "ecart_equilibre": max(abs(x["reaction_sol"] - x["charge_totale"]) / x["charge_totale"] for x in lignes),
            "dalle": d, "champs": champs}


def par_k(modele: Dict, ks: Sequence[float], rapport_c2: Tuple[float, float] = (0.1, 0.1), sans_traction: bool = True) -> List[Dict]:
    """Le résultat de chaque k ; C2x = k × rapport_c2[0], C2y = k × rapport_c2[1] (les valeurs écrites dans le SAF)."""
    return [calculer(modele, k, k * rapport_c2[0], k * rapport_c2[1], sans_traction) for k in ks]


def sigma_moyen(modele: Dict) -> Optional[Dict]:
    """Le repère « σ moyen » : (poids de la dalle + poids des plots + somme des réactions du jeu le plus lourd) ÷ aire
    de la dalle → {"sigma" (kPa), "dalle", "plots", "reactions" (kN), "aire" (m²), "jeu"}."""
    A = G.aire(modele["contour"])
    gamma = modele.get("gamma", 25.0)
    pl = modele.get("plots")
    poids_plots = sum(gamma * pl["hauteur"] * G.aire(q) for q in pl["polys"]) if pl else 0.0
    cas = modele.get("cas") or []
    lourd = max(cas, key=lambda c: sum(c["R"])) if cas else None
    reactions = float(sum(lourd["R"])) if lourd else 0.0
    dalle_ = gamma * modele["h"] * A
    return {"sigma": (dalle_ + poids_plots + reactions) / A, "dalle": dalle_, "plots": poids_plots, "reactions": reactions,
            "aire": A, "jeu": lourd["jeu"] if lourd else None}


# ------------------------------------------------------------------ « Proposer »
def _sigma_max(modele: Dict, ks: Sequence[float], limite: float, rapport_c2) -> float:
    """σ max sur les k, du plus raide au plus souple ; on s'arrête au premier qui dépasse la limite."""
    pire = 0.0
    for k in sorted(ks, reverse=True):
        pire = max(pire, calculer(modele, k, k * rapport_c2[0], k * rapport_c2[1])["sigma_max"])
        if pire > limite:
            break
    return pire


def proposer_un(modele_de: Callable[[float], Optional[Dict]], ks: Sequence[float], limite: float, rapport_c2=(0.0, 0.0),
                pas: float = 0.05, dmax: float = 6.0) -> Tuple[Optional[float], float]:
    """Le plus petit agrandissement d (multiple de ``pas``) tel que σ max ≤ limite pour tous les ``ks`` — recherche par
    dichotomie (σ max diminue quand d augmente). ``modele_de(d)`` rend le modèle de la dalle agrandie de d de chaque
    côté (None si sa géométrie n'est plus valable). Rend (d, σ max) ; d = None si même ``dmax`` ne suffit pas.

    La borne haute de la dichotomie est cherchée en DOUBLANT d (0,50 ; 1 ; 2 ; 4 m… jusqu'à ``dmax``) : le résultat est
    celui d'une dichotomie partie de ``dmax``, sans calculer les très grandes dalles qui ne servent qu'à l'encadrer —
    c'est ce qui coûte dans le navigateur."""
    def smax(i: int) -> float:
        m = modele_de(i * pas)
        return float("inf") if m is None else _sigma_max(m, ks, limite, rapport_c2)
    n = int(round(dmax / pas))
    s0 = smax(0)
    if s0 <= limite:
        return 0.0, s0
    lo, hi = 0, min(n, max(1, int(round(PREMIER_PAS / pas))))
    s_hi = smax(hi)
    while s_hi > limite:
        if hi >= n:
            return None, s_hi
        lo, hi = hi, min(n, 2 * hi)
        s_hi = smax(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        s_mid = smax(mid)
        if s_mid <= limite:
            hi, s_hi = mid, s_mid
        else:
            lo = mid
    return hi * pas, s_hi


def volume(modele: Dict) -> float:
    """Le béton de la dalle et des plots (m³)."""
    pl = modele.get("plots")
    return G.aire(modele["contour"]) * modele["h"] + (sum(G.aire(q) for q in pl["polys"]) * pl["hauteur"] if pl else 0.0)

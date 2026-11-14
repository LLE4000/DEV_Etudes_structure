# -*- coding: utf-8 -*-
"""La STATIQUE d'une barre chargée : chaque cas de charge résolu par le formulaire de poutres du dépôt (rdm/poutres.py,
écritures fermées), puis superposé — le comportement est linéaire. docs/acier/BARRES_ACIER.md § 9.

Unités de la barre : m, kN, kN/m, kN·m (celles de l'écran et du formulaire). Les moteurs de vérification travaillent en
N et mm : ``efforts`` convertit. Conventions du formulaire, reprises telles quelles : une charge vers le bas est
positive ; M > 0 tend la fibre inférieure (la fibre HAUTE est comprimée) ; flèche positive vers le bas.

    poids propre   g = masse linéique × 9,81 / 1000 (kN/m), un cas permanent de plus
    une combinaison  M(x) = Σ γ_i·M_i(x),  V(x) = Σ γ_i·V_i(x),  N = Σ γ_i·N_i,  f(x) = Σ γ_i·f_i(x)
    sections à vérifier : les appuis, chaque rupture de chargement, et l'extremum du moment de chaque travée
    (cherché sur la somme, par balayage puis resserrement)

LES TRONÇONS DE DÉVERSEMENT : entre deux sections tenues en déplacement latéral ET en torsion — les appuis (pris en
fourche : hypothèse dite dans la note), l'encastrement d'une console, et chaque maintien « complet ». Un maintien d'une
seule semelle (« haut », « bas ») reste À L'INTÉRIEUR de son tronçon. Le bout libre d'une console est libre.
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rdm import poutres as PO

from . import mcr as MC
from .resultat import Refus

G_PESANTEUR = 9.81               # m/s²
POIDS_PROPRE = "Poids propre"
GENRES_MAINTIEN = ("complet", "haut", "bas")
BALAYAGE = 48                    # points par travée pour situer l'extremum du moment d'une combinaison


def poids_propre(s: Dict) -> float:
    """Le poids propre du profilé en kN/m."""
    if not s.get("masse"):
        raise Refus("la masse linéique du profilé manque : le poids propre en dépend")
    return s["masse"] * G_PESANTEUR / 1000.0


def cas_complets(barre: Dict, s: Dict) -> List[Dict]:
    """Les cas de charge de la barre, le poids propre en tête s'il est demandé (un cas permanent)."""
    cas = [dict(c) for c in barre["cas"]]
    if barre.get("poids_propre", True):
        L = sum(barre["travees"])
        cas.insert(0, {"nom": POIDS_PROPRE, "nature": "permanent", "N": 0.0,
                       "charges": [{"type": "q", "v": poids_propre(s), "a": 0.0, "b": L}]})
    return cas


def resoudre(barre: Dict, cas: Sequence[Dict], EI: Optional[float] = None) -> List:
    """Une solution du formulaire par cas (``EI`` en kN·m² pour la flèche)."""
    out = []
    for c in cas:
        try:
            out.append(PO.resoudre(barre["modele"], barre["travees"], c.get("charges") or [], EI))
        except PO.ErreurDonnees as e:
            raise Refus(f"cas « {c['nom']} » : {e}") from None
    return out


class Combinaison:
    """Les efforts d'une combinaison le long de la barre (kN, m) : la somme pondérée des cas."""

    def __init__(self, nom: str, sols: Sequence, facteurs: Sequence[float], N_cas: Sequence[float]):
        self.nom, self.facteurs = nom, list(facteurs)
        self.paires = [(f, s) for f, s in zip(facteurs, sols) if f]
        self.N = sum(f * n for f, n in zip(facteurs, N_cas))
        self.L = sols[0].L if sols else 0.0

    def lire(self, x: float) -> Dict:
        """M et V de part et d'autre de la section x, la flèche (None sans E·I)."""
        o = {"x": x, "M_g": 0.0, "M_d": 0.0, "V_g": 0.0, "V_d": 0.0, "f": 0.0}
        for f, s in self.paires:
            e = s.evaluer(x)
            for k in ("M_g", "M_d", "V_g", "V_d"):
                o[k] += f * e[k]
            o["f"] = None if (o["f"] is None or e["f"] is None) else o["f"] + f * e["f"]
        return o

    def M(self, x: float) -> float:
        """Le moment dans la section x (à droite ; à gauche au bout de la barre) — continu sauf sous un couple."""
        e = self.lire(x)
        return e["M_g"] if x >= self.L else e["M_d"]


def ruptures(sols: Sequence) -> List[float]:
    """Les abscisses globales où une expression change : appuis, débuts et fins de charge, charges ponctuelles."""
    pts = set()
    for s in sols:
        for t in s.travees:
            for x in t.ruptures():
                pts.add(round(t.x0 + x, 9))
            for m in t.morceaux:
                if m["type"] in ("P", "C"):
                    pts.add(round(t.x0 + m["a"], 9))
    return sorted(pts)


def _resserrer(f: Callable[[float], float], a: float, b: float) -> float:
    """Le maximum de f sur [a ; b] (f unimodale entre deux points du balayage) : section dorée."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
    return 0.5 * (a + b)


def sections(barre: Dict, comb: Combinaison, pts: Sequence[float]) -> List[float]:
    """Les sections à vérifier pour cette combinaison : ``pts`` (les ruptures) et, par travée, les extremums du moment."""
    out = set(pts)
    x0 = 0.0
    for L in barre["travees"]:
        xs = [x0 + L * i / BALAYAGE for i in range(BALAYAGE + 1)]
        for signe in (1.0, -1.0):
            vals = [signe * comb.M(x) for x in xs]
            i = max(range(len(xs)), key=lambda k: vals[k])
            if 0 < i < BALAYAGE and vals[i] > 0:
                out.add(round(_resserrer(lambda x, s=signe: s * comb.M(x), xs[i - 1], xs[i + 1]), 6))
        x0 += L
    return sorted(out)


def troncons(barre: Dict, s: Dict) -> List[Dict]:
    """Les tronçons de déversement : [{"x1", "x2" (m), "appuis" (les deux bouts), "maintiens": [(x local en mm, cote z)]}]."""
    L = sum(barre["travees"])
    console = barre["modele"] == "console"
    tenus = {0.0: MC.ENCASTRE if console else MC.FOURCHE}
    if not console:
        x = 0.0
        for Lt in barre["travees"]:
            x += Lt
            tenus[round(x, 9)] = MC.FOURCHE
    interieurs = []
    for m in barre.get("maintiens") or []:
        x, genre = float(m["x"]), m.get("genre", "complet")
        if genre not in GENRES_MAINTIEN:
            raise Refus(f"maintien : genre parmi {', '.join(GENRES_MAINTIEN)} — reçu {genre!r}")
        if not 0.0 <= x <= L:
            raise Refus(f"maintien à {x:g} m : hors de la barre (0 à {L:g} m)")
        if genre == "complet":
            tenus.setdefault(round(x, 9), MC.FOURCHE)
        else:
            interieurs.append((x, genre))
    if console and round(L, 9) not in tenus:
        tenus[round(L, 9)] = MC.LIBRE
    bornes = sorted(tenus)
    cote = ((s.get("h") or 0.0) - (s.get("tf") or s.get("t") or 0.0)) / 2.0        # le plan moyen d'une semelle
    out = []
    for a, b in zip(bornes, bornes[1:]):
        if b - a <= 1e-9:
            continue
        dedans = [((x - a) * 1e3, cote if genre == "haut" else -cote) for x, genre in interieurs if a + 1e-9 < x < b - 1e-9]
        out.append({"x1": a, "x2": b, "appuis": (tenus[a], tenus[b]), "maintiens": dedans})
    return out


def charges_du_troncon(cas: Sequence[Dict], facteurs: Sequence[float], tr: Dict, zg: float) -> Tuple[List, List]:
    """Les charges de la combinaison portées par le tronçon, en N et mm, abscisses locales, appliquées à la cote ``zg`` :
    (réparties [(x1, x2, q1, q2, zg)], ponctuelles [(x, P, zg)]). Les couples n'ont pas de hauteur d'application."""
    a, b = tr["x1"], tr["x2"]
    rep, pon = [], []
    for f, c in zip(facteurs, cas):
        if not f:
            continue
        for ch in c.get("charges") or []:
            if ch["type"] == "P":
                if a - 1e-9 <= ch["a"] <= b + 1e-9:
                    pon.append(((ch["a"] - a) * 1e3, f * ch["v"] * 1e3, zg))
            elif ch["type"] in ("q", "tri"):
                ca, cb = ch.get("a", 0.0), ch.get("b", b)
                u, v = max(a, ca), min(b, cb)
                if v - u <= 1e-9:
                    continue
                if ch["type"] == "q":
                    q1 = q2 = ch["v"]
                else:
                    croissante = ch.get("sens", "croissante") == "croissante"
                    def q(x, ca=ca, cb=cb, v_=ch["v"], cr=croissante):
                        t = (x - ca) / (cb - ca)
                        return v_ * (t if cr else 1.0 - t)
                    q1, q2 = q(u), q(v)
                rep.append(((u - a) * 1e3, (v - a) * 1e3, f * q1, f * q2, zg))        # 1 kN/m = 1 N/mm
    return rep, pon

# -*- coding: utf-8 -*-
"""La géométrie d'un garde-corps : sa forme en plan (droit, en L, en U : un, deux ou trois
tronçons droits), les entraxes LIBRES de chaque tronçon (réguliers à partir d'une longueur
et d'un entraxe maximal, ou saisis un à un : « un poteau central, un à gauche, un à droite,
pas à la même distance »), ses extrémités (un montant d'extrémité, ou la lisse ancrée dans un
mur qui reprend la demi-travée), et la liste des MONTANTS qui en découle, chacun avec sa
longueur tributaire par direction : un montant courant reçoit la demi-somme des deux entraxes
voisins dans une seule direction ; un montant d'ANGLE (L, U) reçoit une demi-travée de chaque
tronçon, dans deux directions perpendiculaires.

Le même découpage sert au système « lisse ancrée dans un mur » : les montants deviennent les
consoles de fixation, aux mêmes entraxes, avec les mêmes longueurs tributaires.

SANS MONTANT : une seule travée ancrée dans un mur à chaque bout (le garde-corps d'une loggia
entre deux murs) — la main courante porte de mur à mur, chaque mur reprend une demi-travée.
RAMPANT (escalier) : la pente α ne change pas la mécanique — les entraxes sont mesurés le long
de la main courante, les montants restent verticaux ; elle incline l'élévation et se lit dans
les hypothèses.
"""
import math
from typing import Dict, List

FORMES = {"droit": "droit", "L": "en L", "U": "en U"}
N_TRONCONS = {"droit": 1, "L": 2, "U": 3}
EXTREMITES = {"montant": "montant d'extrémité", "mur": "lisse ancrée dans le mur"}
SYSTEMES = {"montants": "montants verticaux", "mur": "lisse ancrée au mur"}
SYSTEMES_LONG = {"montants": "montants verticaux sur platine", "mur": "lisse horizontale ancrée dans un mur (consoles)"}
H_DEFAUT = 1100.0
S_MAX_DEFAUT = 1000.0


def entraxes_reguliers(L: float, s_max: float) -> List[float]:
    """n travées égales, n = ⌈L / s_max⌉ (au moins une)."""
    L, s_max = float(L), float(s_max)
    if L <= 0:
        return []
    n = max(1, int(math.ceil(L / s_max - 1e-9))) if s_max > 0 else 1
    return [L / n] * n


def troncon_defaut(L: float = 3600.0, s_max: float = S_MAX_DEFAUT) -> Dict:
    return {"L": float(L), "s_max": float(s_max), "entraxes": []}


def geometrie_defaut() -> Dict:
    return {"systeme": "montants", "forme": "droit", "H": H_DEFAUT, "pente": 0.0,
            "troncons": [troncon_defaut(3600.0), troncon_defaut(2400.0), troncon_defaut(3600.0)],
            "debut": "montant", "fin": "montant", "e_console": 100.0}


def entraxes_du_troncon(tr: Dict) -> List[float]:
    """Les travées d'un tronçon : la liste saisie si elle est renseignée, sinon la répartition
    régulière de sa longueur à l'entraxe maximal."""
    saisis = [float(v) for v in (tr.get("entraxes") or []) if v is not None and float(v) > 0]
    if saisis:
        return saisis
    return entraxes_reguliers(tr.get("L", 0.0), tr.get("s_max", S_MAX_DEFAUT))


def troncons_actifs(geom: Dict) -> List[Dict]:
    n = N_TRONCONS.get(geom.get("forme", "droit"), 1)
    trs = list(geom.get("troncons") or [])
    while len(trs) < n:
        trs.append(troncon_defaut())
    return trs[:n]


def entraxes_tous(geom: Dict) -> List[List[float]]:
    return [entraxes_du_troncon(tr) for tr in troncons_actifs(geom)]


def analyser(geom: Dict) -> Dict:
    """Les montants (ou consoles) et les travées d'une géométrie :

    ``montants`` : [{nom, x (abscisse développée, mm), troncon, coin, trib1, trib2, s_g, s_d}]
    avec trib1 la longueur tributaire (mm) dans la direction normale au tronçon du montant et
    trib2 celle dans la direction normale au tronçon suivant (montant d'angle seulement) ;
    ``travees`` : [{troncon, i, x0, L}] ; ``s_max`` ; ``L_total`` ; ``messages``.
    Une extrémité « mur » n'a pas de montant : le mur reprend la demi-travée (``reprises_mur``).
    """
    forme = geom.get("forme", "droit")
    ents = entraxes_tous(geom)
    debut = geom.get("debut", "montant")
    fin = geom.get("fin", "montant")
    montants: List[Dict] = []
    travees: List[Dict] = []
    reprises = []
    x = 0.0
    n_tr = len(ents)
    for i, ent in enumerate(ents):
        if not ent:
            continue
        for j, s in enumerate(ent):
            premier = (i == 0 and j == 0)
            if premier:
                if debut == "montant":
                    montants.append({"nom": "", "x": x, "troncon": i, "coin": False, "trib1": s / 2, "trib2": 0.0, "s_g": 0.0, "s_d": s})
                else:
                    reprises.append({"ou": "début", "x": x, "trib": s / 2})
            else:
                m = montants[-1]
                if m.get("_fin_troncon"):
                    m["coin"] = True
                    m["trib2"] = s / 2
                    m["s_d"] = s
                    del m["_fin_troncon"]
                else:
                    m["trib1"] += s / 2
                    m["s_d"] = s
            travees.append({"troncon": i, "i": j, "x0": x, "L": s})
            x += s
            dernier = (i == n_tr - 1 and j == len(ent) - 1)
            if dernier:
                if fin == "montant":
                    montants.append({"nom": "", "x": x, "troncon": i, "coin": False, "trib1": s / 2, "trib2": 0.0, "s_g": s, "s_d": 0.0})
                else:
                    reprises.append({"ou": "fin", "x": x, "trib": s / 2})
            else:
                fin_tr = (j == len(ent) - 1)
                montants.append({"nom": "", "x": x, "troncon": i, "coin": False, "trib1": s / 2, "trib2": 0.0, "s_g": s, "s_d": 0.0,
                                 **({"_fin_troncon": True} if fin_tr else {})})
    for k, m in enumerate(montants, 1):
        m["nom"] = f"M{k}"
        m.pop("_fin_troncon", None)
        m["trib"] = m["trib1"] + m["trib2"]
    s_max = max((t["L"] for t in travees), default=0.0)
    messages = []
    if not travees:
        messages.append("Refus : aucune travée (longueur ou entraxes nuls)")
    sans_montant = bool(travees) and not montants          # une travée de mur à mur : la main courante porte, sans montant
    return {"forme": forme, "montants": montants, "travees": travees, "s_max": s_max, "L_total": x, "reprises_mur": reprises,
            "entraxes": ents, "messages": messages, "n_montants": len(montants), "n_coins": sum(1 for m in montants if m["coin"]),
            "sans_montant": sans_montant, "pente": float(geom.get("pente") or 0.0)}


def longueurs_troncons(geom: Dict) -> List[float]:
    return [sum(e) for e in entraxes_tous(geom)]


def phrase(geom: Dict, ana: Dict) -> str:
    """« droit, 3,60 m en 3 travées de 1,20 m, 4 montants » ; « en L, 3,60 + 2,40 m, … »."""
    def m(v):
        return f"{v / 1000:.2f}".replace(".", ",") + " m"
    parts = []
    for ent in ana["entraxes"]:
        if not ent:
            continue
        L = sum(ent)
        if len(set(round(e, 3) for e in ent)) == 1:
            parts.append(f"{m(L)} en {len(ent)} travée{'s' if len(ent) > 1 else ''} de {m(ent[0])}")
        else:
            parts.append(f"{m(L)} en travées de " + " + ".join(m(e) for e in ent))
    s = FORMES.get(geom.get("forme", "droit"), "droit") + " : " + " ; ".join(parts)
    ext = []
    if geom.get("debut") == "mur":
        ext.append("début ancré dans le mur")
    if geom.get("fin") == "mur":
        ext.append("fin ancrée dans le mur")
    n = ana["n_montants"]
    nom = "console" if geom.get("systeme") == "mur" else "montant"
    if n:
        s += f" · {n} {nom}{'s' if n > 1 else ''}" + (f" dont {ana['n_coins']} d'angle" if ana["n_coins"] else "")
    else:
        s += " · sans montant, main courante de mur à mur"
    if ext:
        s += " · " + ", ".join(ext)
    pente = float(geom.get("pente") or 0.0)
    if pente:
        s += f" · rampant, pente {pente:g}° (entraxes mesurés suivant la pente)"
    return s

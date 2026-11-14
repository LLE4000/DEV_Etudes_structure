# -*- coding: utf-8 -*-
"""La bibliothèque des châssis de grue à tour — l'ACCÈS pour les autres modules, en LECTURE SEULE.

La base est ``donnees/chassis_grues.json`` (21 châssis Liebherr et Wolffkran), telle qu'elle a été livrée : rien n'y est
modifié, rien n'en est recopié ailleurs. Ce fichier traduit en Python, sans changer un résultat, les fonctions d'accès du
moteur de dessin d'origine (``src/chassis_dessin.js`` : ``appuis``, ``emprise``, ``emprises``, ``horsToutEmprises``) —
``tests/test_saf_grue_export.py`` compare les deux sur tous les châssis (node rejoue le moteur d'origine).

Longueurs en MILLIMÈTRES. Repère : origine au centre du châssis, x et y parallèles aux côtés du carré d'appuis. Les
appuis sont numérotés 1 à 4 dans le sens trigonométrique depuis (−x, −y).

    liste()                 -> [{id, constructeur, famille, nom, entraxe_x, entraxe_y, appui (ou None)}]
    chassis(id)             -> le châssis de la base (une COPIE : la base n'est jamais modifiée) ; KeyError s'il manque
    appuis(ch)              -> les 4 appuis : {n, x, y, r (vers l'extérieur, le long du bras), t}
    emprises(ch)            -> les 4 surfaces d'appui : {n, forme, centre, sommets, …} ; None où la source ne dit rien
    avec_appui(ch, appui)   -> une copie du châssis dont la surface d'appui est remplacée (un rectangle saisi)
    valeur(o), origine(o)   -> la valeur d'un relevé {valeur, origine, note} et son origine (cote, mesure, calcul…)
"""
import copy
import json
import math
import os
from functools import lru_cache
from typing import Dict, List, Optional

ICI = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(ICI, "donnees", "chassis_grues.json")
COTES_CERCLE = 48                # un cercle est rendu par un polygone à 48 côtés (un sommet tous les 7,5°)


@lru_cache(maxsize=1)
def _base() -> Dict:
    with open(FICHIER, encoding="utf-8") as f:
        return json.load(f)


def valeur(o):
    """La valeur d'un relevé ``{valeur, origine, note}`` ; une valeur nue est rendue telle quelle ; None si absente."""
    if isinstance(o, dict) and "valeur" in o:
        return o["valeur"]
    return o


def origine(o) -> Optional[str]:
    """L'origine d'un relevé : « cote », « mesure », « calcul », « indicatif » ; None pour une valeur nue."""
    return o.get("origine") if isinstance(o, dict) else None


def liste() -> List[Dict]:
    """Les châssis de la base, dans son ordre : de quoi faire la liste (groupée par constructeur, avec recherche)."""
    out = []
    for c in _base()["chassis"]:
        out.append({"id": c["id"], "constructeur": c["constructeur"], "famille": c.get("famille"), "nom": c["nom"],
                    "entraxe_x": valeur(c["appuis"]["entraxe_x"]), "entraxe_y": valeur(c["appuis"]["entraxe_y"]),
                    "appui": copy.deepcopy((c.get("pied") or {}).get("appui"))})
    return out


def chassis(ident: str) -> Dict:
    """Le châssis d'un ``id`` — une copie profonde : ce qu'on en fait ne touche jamais la base."""
    for c in _base()["chassis"]:
        if c["id"] == ident:
            return copy.deepcopy(c)
    raise KeyError(f"châssis inconnu : {ident!r}")


def appuis(ch: Dict) -> List[Dict]:
    """Les 4 appuis : position, direction radiale r (vers l'extérieur, le long du bras) et tangentielle t."""
    ex, ey = valeur(ch["appuis"]["entraxe_x"]), valeur(ch["appuis"]["entraxe_y"])
    out = []
    for i, (x, y) in enumerate(((-ex / 2, -ey / 2), (ex / 2, -ey / 2), (ex / 2, ey / 2), (-ex / 2, ey / 2))):
        L = math.hypot(x, y)
        r = [x / L, y / L]
        out.append({"n": i + 1, "x": x, "y": y, "r": r, "t": [-r[1], r[0]]})
    return out


def emprise(ch: Dict, a: Dict) -> Optional[Dict]:
    """L'emprise au sol d'un appui (la surface qui charge la fondation), en mm dans le repère du châssis :
    {n, forme, centre [x, y], sommets [[x, y], …]} ; un cercle porte aussi ``rayon`` (ses sommets : 48 côtés).
    None si la source ne donne pas la dimension d'appui."""
    ap = (ch.get("pied") or {}).get("appui")
    if not ap:
        return None
    exc = valeur(ap.get("excentrement_radial")) or 0
    c = [a["x"] + a["r"][0] * exc, a["y"] + a["r"][1] * exc]
    rot = ap.get("rotation_deg") or 0
    s = []
    if ap["forme"] == "rectangle":
        da, db = valeur(ap["a"]) / 2, valeur(ap["b"]) / 2
        u, w = (a["t"], a["r"]) if rot == 45 else ([1, 0], [0, 1])
        for qa, qb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            s.append([c[0] + qa * da * u[0] + qb * db * w[0], c[1] + qa * da * u[1] + qb * db * w[1]])
        return {"n": a["n"], "forme": "rectangle", "centre": c, "sommets": s, "a": valeur(ap["a"]), "b": valeur(ap["b"]),
                "rotation_deg": rot}
    if ap["forme"] == "cercle":
        R = valeur(ap["diametre"]) / 2
        for i in range(COTES_CERCLE):
            s.append([c[0] + R * math.cos(i * math.pi / 24), c[1] + R * math.sin(i * math.pi / 24)])
        return {"n": a["n"], "forme": "cercle", "centre": c, "rayon": R, "sommets": s}
    if ap["forme"] == "octogone":
        Rc = valeur(ap["cote_sur_plats"]) / 2 / math.cos(math.pi / 8)
        for i in range(8):
            s.append([c[0] + Rc * math.cos(math.pi / 8 + i * math.pi / 4), c[1] + Rc * math.sin(math.pi / 8 + i * math.pi / 4)])
        return {"n": a["n"], "forme": "octogone", "centre": c, "sommets": s, "cote_sur_plats": valeur(ap["cote_sur_plats"])}
    return None


def emprises(ch: Dict) -> List[Optional[Dict]]:
    """Les 4 surfaces d'appui, dans l'ordre des appuis (1 à 4)."""
    return [emprise(ch, a) for a in appuis(ch)]


def hors_tout_emprises(ch: Dict) -> Optional[Dict]:
    """Le hors-tout des surfaces d'appui, calculé : {x, y} ; None sans surface d'appui."""
    E = emprises(ch)
    if not E[0]:
        return None
    mx = max(abs(p[0]) for e in E for p in e["sommets"])
    my = max(abs(p[1]) for e in E for p in e["sommets"])
    return {"x": 2 * mx, "y": 2 * my}


def avec_appui(ch: Dict, appui: Optional[Dict]) -> Dict:
    """Une COPIE du châssis dont ``pied.appui`` est remplacé par ``appui`` (un rectangle saisi : {forme, a, b,
    rotation_deg, excentrement_radial}) — pour en lire les sommets par ``emprises``. La base n'est jamais modifiée."""
    c = copy.deepcopy(ch)
    c.setdefault("pied", {})
    c["pied"] = dict(c["pied"] or {}, appui=copy.deepcopy(appui))
    return c

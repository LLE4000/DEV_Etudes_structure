# -*- coding: utf-8 -*-
"""Fondation de grue — le CHÂSSIS d'un projet : celui de la bibliothèque des châssis de grue (grues/chassis), ou un
châssis saisi. Rien n'est recopié ni recalculé : on lit la bibliothèque par sa fonction d'accès, on convertit (mm → m),
on renumérote (appui n de la bibliothèque → pied P(5 − n) du module). La bibliothèque n'est jamais modifiée.

    lire(grue, rond="carre") -> le châssis prêt à être posé :
        {"id" (None pour un châssis saisi), "nom", "constructeur", "a", "b" (entraxes, m), "carre",
         "platine": None | {"forme", "a", "b", "diametre", "cote_sur_plats", "rotation", "excentrement", "aire",
                            "cote_carre", "origines": {champ: origine}}   — ce que dit la bibliothèque, en m et m²
         "appuis": "reels" | "rectangle" (le choix RETENU), "impose" (True : rectangle d'office, platine non indiquée),
         "forme" (de la surface retenue : « rectangle », « octogone », « cercle », « carré de même aire »), "angle" (0 | 45),
         "dims" (le texte court de la surface : « 0,58 × 0,40 m à 45° »), "saisie" (True : dimensions saisies),
         "surfaces": [P1…P4] (polygones, m, repère du châssis — origine à son centre, non tourné ; sommets dans le
                     sens HORAIRE, un rectangle parallèle aux axes commençant par son coin haut-gauche : l'ordre du SAF),
         "centres": [P1…P4] (le centre de chaque surface : l'excentrement radial y est), "aire" (d'une surface, m²),
         "note_fondation" (la note de la bibliothèque sur la partie centrale, ou vide)}

Numérotation : la bibliothèque numérote ses appuis 1 à 4 dans le sens trigonométrique depuis (−x, −y) ; le module, P1
(haut-gauche), P2 (haut-droite), P3 (bas-droite), P4 (bas-gauche) — appui n → P(5 − n).
"""
import math
from typing import Dict, List, Optional

from grues.chassis import acces as BIB

from . import geometrie as G

SAISIE = "saisie"                # la clé d'un châssis saisi (sans bibliothèque)
RECT_DEFAUT = (0.50, 0.50)       # appuis « Rectangle » : a × b par défaut (m)
ENTRAXE_DEFAUT = (6.0, 6.0)      # châssis saisi : a × b par défaut (m)
RONDS = ("carre", "48")          # une platine ronde : carré de même aire, ou polygone à 48 côtés


def liste() -> List[Dict]:
    """Les châssis de la bibliothèque, pour la liste (groupée par constructeur) : [{id, constructeur, nom}]."""
    return [{"id": c["id"], "constructeur": c["constructeur"], "nom": c["nom"]} for c in BIB.liste()]


def existe(ident) -> bool:
    return isinstance(ident, str) and any(c["id"] == ident for c in BIB.liste())


def _nombre(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}".rstrip("0").rstrip(".").replace(".", ",") if dec else f"{x:.0f}"


def _m(x: float) -> str:
    """Une dimension en mètres, à deux décimales au moins : 0,58 ; 0,443."""
    s = f"{x:.3f}"
    return (s[:-1] if s.endswith("0") else s).replace(".", ",")


def _platine(ap: Optional[Dict]) -> Optional[Dict]:
    """La platine de la bibliothèque, en mètres, avec l'origine de chaque valeur."""
    if not ap:
        return None
    champs = ("a", "b", "diametre", "cote_sur_plats", "excentrement_radial", "aire_m2", "cote_carre_meme_aire")
    v = {k: BIB.valeur(ap.get(k)) for k in champs}
    return {"forme": ap["forme"], "a": None if v["a"] is None else v["a"] / 1000, "b": None if v["b"] is None else v["b"] / 1000,
            "diametre": None if v["diametre"] is None else v["diametre"] / 1000,
            "cote_sur_plats": None if v["cote_sur_plats"] is None else v["cote_sur_plats"] / 1000,
            "rotation": ap.get("rotation_deg") or 0, "excentrement": (v["excentrement_radial"] or 0) / 1000,
            "aire": v["aire_m2"], "cote_carre": None if v["cote_carre_meme_aire"] is None else v["cote_carre_meme_aire"] / 1000,
            "origines": {k: BIB.origine(ap.get(k)) for k in champs if BIB.origine(ap.get(k))}}


def _en_pieds(valeurs: List) -> List:
    """Les quatre valeurs de la bibliothèque (appuis 1 à 4) dans l'ordre du module (P1 à P4) : appui n → P(5 − n)."""
    return [valeurs[3], valeurs[2], valeurs[1], valeurs[0]]


def lire(grue: Dict, rond: str = "carre") -> Dict:
    """Le châssis d'un projet (voir l'en-tête du fichier). ``grue`` : {"chassis": id | « saisie », "a", "b" (châssis
    saisi), "appuis": « reels » | « rectangle », "rect": {"a", "b", "angle": None | 0 | 45}}."""
    ident = grue.get("chassis")
    rect = grue.get("rect") or {}
    if ident and ident != SAISIE:
        ch = BIB.chassis(ident)                                   # KeyError si l'id n'est pas dans la bibliothèque
        nom, constructeur = ch["nom"], ch["constructeur"]
        note = (ch.get("fondation") or {}).get("note") or ""
    else:
        ident = None
        a, b = float(grue.get("a") or ENTRAXE_DEFAUT[0]), float(grue.get("b") or ENTRAXE_DEFAUT[1])
        ch = {"appuis": {"entraxe_x": a * 1000.0, "entraxe_y": b * 1000.0}, "pied": {"appui": None}}
        nom, constructeur, note = "", "", ""
    a, b = BIB.valeur(ch["appuis"]["entraxe_x"]) / 1000.0, BIB.valeur(ch["appuis"]["entraxe_y"]) / 1000.0
    reel = (ch.get("pied") or {}).get("appui")
    platine = _platine(reel)
    impose = platine is None                                      # pas de platine : appuis « Rectangle » d'office
    choix = "rectangle" if impose or grue.get("appuis") == "rectangle" else "reels"
    saisie = False
    if choix == "rectangle":
        ra, rb = float(rect.get("a") or RECT_DEFAUT[0]), float(rect.get("b") or RECT_DEFAUT[1])
        angle = rect.get("angle")
        if angle not in (0, 45):                                  # par défaut : l'orientation de la platine réelle
            angle = 45 if platine and platine["rotation"] == 45 else 0
        exc = BIB.valeur((reel or {}).get("excentrement_radial")) or 0
        copie = BIB.avec_appui(ch, {"forme": "rectangle", "a": ra * 1000.0, "b": rb * 1000.0, "rotation_deg": angle,
                                    "excentrement_radial": exc})
        emprises = BIB.emprises(copie)
        forme, dims, saisie = "rectangle", f"{_m(ra)} × {_m(rb)} m" + (" à 45°" if angle == 45 else ""), True
        surfaces = [[(x / 1000.0, y / 1000.0) for x, y in e["sommets"]] for e in emprises]
        centres = [(e["centre"][0] / 1000.0, e["centre"][1] / 1000.0) for e in emprises]
    else:
        emprises = BIB.emprises(ch)
        angle = platine["rotation"]
        centres = [(e["centre"][0] / 1000.0, e["centre"][1] / 1000.0) for e in emprises]
        if platine["forme"] == "cercle" and rond != "48":         # carré de même aire, côtés parallèles aux axes
            c = platine["cote_carre"]
            forme, dims = "carré de même aire", f"Ø {_m(platine['diametre'])} m → carré {_m(c)} m"
            surfaces = [[(x - c / 2, y - c / 2), (x + c / 2, y - c / 2), (x + c / 2, y + c / 2), (x - c / 2, y + c / 2)] for x, y in centres]
        else:
            surfaces = [[(x / 1000.0, y / 1000.0) for x, y in e["sommets"]] for e in emprises]
            forme = platine["forme"]
            if forme == "rectangle":
                dims = f"{_m(platine['a'])} × {_m(platine['b'])} m" + (" à 45°" if angle == 45 else "")
            elif forme == "cercle":
                dims = f"Ø {_m(platine['diametre'])} m (48 côtés)"
            else:
                dims = f"octogone de {_m(platine['cote_sur_plats'])} m sur plats"
    # la bibliothèque rend ses sommets dans le sens trigonométrique (un rectangle : depuis son coin bas-gauche) ; le SAF
    # les veut dans le sens horaire, un rectangle depuis son coin haut-gauche : c'est la même liste, renversée
    surfaces = [list(reversed(s)) for s in surfaces]
    return {"id": ident, "nom": nom, "constructeur": constructeur, "a": a, "b": b, "carre": abs(a - b) < 1e-9,
            "platine": platine, "appuis": choix, "impose": impose and ident is not None, "forme": forme, "angle": angle,
            "dims": dims, "saisie": saisie, "surfaces": _en_pieds(surfaces), "centres": _en_pieds(centres),
            "aire": G.aire(surfaces[0]), "note_fondation": note}


def poser(surfaces: List[List], cx: float, cy: float, beta_deg: float) -> List[List]:
    """Des polygones du repère du châssis posés dans le repère de la dalle : (cx, cy) + le point tourné de β."""
    return [G.tourner(cx, cy, s, beta_deg) for s in surfaces]

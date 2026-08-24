# -*- coding: utf-8 -*-
# ============================================================
#  treillis.py — Base des treillis soudés courants (module Dalle)
#
#  Partagé par modules/dalle.py (interface) et
#  modules/export_pdf_dalle.py (note de calcul).
#
#  Nomenclature : Øl/Øt/el/et
#    Øl = diamètre des fils porteurs (mm)      — direction calculée
#    Øt = diamètre des fils de répartition (mm)
#    el = espacement des fils porteurs (mm)
#    et = espacement des fils de répartition (mm)
#
#  La section d'acier est TOUJOURS calculée depuis la nomenclature
#  (aire du fil × 1000 / espacement) : une seule source, pas de valeur
#  recopiée à la main.
#
#  POUR AJOUTER UN TREILLIS : ajouter un tuple (Øl, Øt, el, et) à
#  TREILLIS_STANDARDS — tout le reste (liste de choix, sections,
#  libellés, PDF) suit automatiquement.
# ============================================================
import math

# (Ø longitudinal mm, Ø transversal mm, espacement long. mm, espacement transv. mm)
TREILLIS_STANDARDS = [
    (5, 5, 150, 150),      # As = 131 mm²/m
    (6, 6, 150, 150),      # As = 188 mm²/m
    (7, 7, 150, 150),      # As = 257 mm²/m
    (6, 6, 100, 100),      # As = 283 mm²/m
    (8, 8, 150, 150),      # As = 335 mm²/m
    (8, 8, 100, 100),      # As = 503 mm²/m
    (10, 10, 150, 150),    # As = 524 mm²/m
    (12, 12, 150, 150),    # As = 754 mm²/m
    (10, 10, 100, 100),    # As = 785 mm²/m
    (12, 12, 100, 100),    # As = 1131 mm²/m
]

TREILLIS_DEFAUT = "10/10/100/100"


def aire_barre_mm2(d_mm: float) -> float:
    """Aire d'un fil / d'une barre (mm²)."""
    return math.pi * (float(d_mm) / 2.0) ** 2


def designation(t) -> str:
    """(Øl, Øt, el, et) -> 'Øl/Øt/el/et'."""
    return f"{int(t[0])}/{int(t[1])}/{int(t[2])}/{int(t[3])}"


def parse_designation(des: str):
    """'10/10/100/100' -> (10, 10, 100, 100) ou None si invalide."""
    try:
        parts = [int(float(p.strip())) for p in str(des).split("/")]
    except Exception:
        return None
    if len(parts) != 4 or any(p <= 0 for p in parts):
        return None
    return tuple(parts)


def as_treillis_mm2_m(des: str) -> float:
    """Section des fils porteurs (mm²/m) d'un treillis, depuis sa nomenclature."""
    t = parse_designation(des)
    if t is None:
        return 0.0
    dl, _, el, _ = t
    return aire_barre_mm2(dl) * 1000.0 / el


def as_barres_mm2_m(d_mm: float, esp_mm: float) -> float:
    """Section (mm²/m) de barres Ød espacées de esp_mm."""
    esp = float(esp_mm)
    if esp <= 0:
        return 0.0
    return aire_barre_mm2(d_mm) * 1000.0 / esp


def liste_choix():
    """Désignations proposées dans l'interface, triées par section croissante."""
    return sorted((designation(t) for t in TREILLIS_STANDARDS), key=as_treillis_mm2_m)

# -*- coding: utf-8 -*-
"""Familles de profilés : ce que chacune a comme dimensions, comment son
contour se construit, quelles propriétés s'en déduisent et par quelle
méthode. Une seule fonction, ``calculer(famille, dims)``, sert au
catalogue livré (``tools/transcrire_profiles.py``) et au profilé
personnalisé saisi à l'écran : les deux chemins donnent les mêmes nombres.

Chaque famille est décrite par un dictionnaire FAMILLES[code] :
``libelle`` (nom long), ``groupe`` (I/H, U, L, creux, plein), ``norme``
(texte de la norme de dimensions), ``dims`` (clés des dimensions en mm),
``forme`` (constructeur de contour), ``courant`` (série usuelle en
Belgique, True/False), ``non_standard`` (profilé défini par l'utilisateur).
"""
import math
from typing import Dict, List, Optional

import numpy as np

from . import geometrie as G

PAS_RELATIF = 20          # pas de la grille de Saint-Venant : épaisseur minimale / 20 (≈ 0,5 %)
PAS_MAX = 2.0             # mm — borne haute du pas (grandes sections)

# ------------------------------------------------------------------ registre
FAMILLES: Dict[str, Dict] = {
    "IPE":   {"libelle": "Poutrelles IPE", "groupe": "I/H", "norme": "EN 10365:2017 (Euronorm 19-57)",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": True,
              "description": "Poutrelles à ailes parallèles moyennes, la poutre courante."},
    "IPE A": {"libelle": "Poutrelles IPE A (allégées)", "groupe": "I/H", "norme": "EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": False,
              "description": "Variante allégée des IPE, même hauteur nominale, semelles et âme plus minces."},
    "IPE O": {"libelle": "Poutrelles IPE O (renforcées)", "groupe": "I/H", "norme": "EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": False,
              "description": "Variante renforcée des IPE, hauteur légèrement supérieure."},
    "IPN":   {"libelle": "Poutrelles IPN (ailes inclinées)", "groupe": "I/H", "norme": "DIN 1025-1 / EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r1", "r2"), "forme": "I_conique", "courant": False,
              "description": "Poutrelles anciennes à semelles inclinées à 14 % ; t_f mesurée à b/4 de l'âme."},
    "HEA":   {"libelle": "Poutrelles HE A", "groupe": "I/H", "norme": "EN 10365:2017 (Euronorm 53-62)",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": True,
              "description": "Poutrelles à larges ailes, série légère."},
    "HEB":   {"libelle": "Poutrelles HE B", "groupe": "I/H", "norme": "EN 10365:2017 (Euronorm 53-62)",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": True,
              "description": "Poutrelles à larges ailes, série normale."},
    "HEM":   {"libelle": "Poutrelles HE M", "groupe": "I/H", "norme": "EN 10365:2017 (Euronorm 53-62)",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": True,
              "description": "Poutrelles à larges ailes, série renforcée."},
    "HEAA":  {"libelle": "Poutrelles HE AA (extra-légères)", "groupe": "I/H", "norme": "EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": False,
              "description": "Série extra-légère des HE, hors norme de dimensions pour certains tonnages."},
    "HL":    {"libelle": "Poutrelles HL (très larges ailes)", "groupe": "I/H", "norme": "EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": False,
              "description": "Poutrelles de grande hauteur à très larges ailes (920 à 1100 mm)."},
    "HD":    {"libelle": "Poteaux HD (larges ailes)", "groupe": "I/H", "norme": "EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": False,
              "description": "Poteaux à larges ailes, séries lourdes jusqu'à 1 299 kg/m."},
    "HP":    {"libelle": "Pieux HP (larges ailes)", "groupe": "I/H", "norme": "EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "I", "courant": False,
              "description": "Pieux à larges ailes, âme et semelles d'égale épaisseur."},
    "UPE":   {"libelle": "Profilés U UPE (ailes parallèles)", "groupe": "U", "norme": "EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r"), "forme": "U", "courant": True,
              "description": "U à ailes parallèles, le U courant."},
    "UPN":   {"libelle": "Profilés U UPN (ailes inclinées)", "groupe": "U", "norme": "DIN 1026-1 / EN 10365:2017",
              "dims": ("h", "b", "tw", "tf", "r1", "r2"), "forme": "U_conique", "courant": True,
              "description": "U à semelles inclinées à 8 % (5 % au-delà de 300) ; t_f mesurée à b/2 du bout d'aile."},
    "L":     {"libelle": "Cornières à ailes égales", "groupe": "L", "norme": "EN 10056-1:2017",
              "dims": ("h", "b", "t", "r1", "r2"), "forme": "L", "courant": True,
              "description": "Cornières à ailes égales, congé r1 au talon, arrondis r2 aux bouts."},
    "LI":    {"libelle": "Cornières à ailes inégales", "groupe": "L", "norme": "EN 10056-1:2017",
              "dims": ("h", "b", "t", "r1", "r2"), "forme": "L", "courant": True,
              "description": "Cornières à ailes inégales (h grande aile, b petite aile)."},
    "CHS":   {"libelle": "Tubes ronds finis à chaud", "groupe": "creux", "norme": "EN 10210-2:2019",
              "dims": ("d", "t"), "forme": "creux_rond", "courant": True,
              "description": "Profils creux circulaires finis à chaud."},
    "SHS":   {"libelle": "Tubes carrés finis à chaud", "groupe": "creux", "norme": "EN 10210-2:2019",
              "dims": ("h", "b", "t"), "forme": "creux_rect", "courant": True, "norme_creux": "EN 10210",
              "description": "Profils creux carrés finis à chaud, coins r_ext = 1,5·t."},
    "RHS":   {"libelle": "Tubes rectangulaires finis à chaud", "groupe": "creux", "norme": "EN 10210-2:2019",
              "dims": ("h", "b", "t"), "forme": "creux_rect", "courant": True, "norme_creux": "EN 10210",
              "description": "Profils creux rectangulaires finis à chaud, coins r_ext = 1,5·t."},
    "CHS-F": {"libelle": "Tubes ronds formés à froid", "groupe": "creux", "norme": "EN 10219-2:2019",
              "dims": ("d", "t"), "forme": "creux_rond", "courant": True,
              "description": "Profils creux circulaires formés à froid (gamme britannique EN 10219)."},
    "SHS-F": {"libelle": "Tubes carrés formés à froid", "groupe": "creux", "norme": "EN 10219-2:2019",
              "dims": ("h", "b", "t"), "forme": "creux_rect", "courant": True, "norme_creux": "EN 10219",
              "description": "Profils creux carrés formés à froid, coins r_ext = 2·t à 3·t selon t."},
    "RHS-F": {"libelle": "Tubes rectangulaires formés à froid", "groupe": "creux", "norme": "EN 10219-2:2019",
              "dims": ("h", "b", "t"), "forme": "creux_rect", "courant": True, "norme_creux": "EN 10219",
              "description": "Profils creux rectangulaires formés à froid, coins r_ext = 2·t à 3·t selon t."},
    # ---- profilés définis par l'utilisateur (non standard)
    "PRS":   {"libelle": "PRS — I reconstitué soudé", "groupe": "I/H", "norme": "défini par l'utilisateur",
              "dims": ("h", "b", "tw", "tf"), "forme": "I", "courant": False, "non_standard": True,
              "description": "I soudé de trois plats, sans congé ; semelles égales."},
    "CAISSON": {"libelle": "Caisson soudé", "groupe": "creux", "norme": "défini par l'utilisateur",
              "dims": ("h", "b", "t"), "forme": "creux_rect", "courant": False, "non_standard": True, "norme_creux": None,
              "description": "Tube rectangulaire reconstitué de quatre plats d'épaisseur t, coins vifs."},
    "TUBE":  {"libelle": "Tube rond quelconque", "groupe": "creux", "norme": "défini par l'utilisateur",
              "dims": ("d", "t"), "forme": "creux_rond", "courant": False, "non_standard": True,
              "description": "Tube rond de diamètre et d'épaisseur libres."},
    "PLAT":  {"libelle": "Plat", "groupe": "plein", "norme": "défini par l'utilisateur",
              "dims": ("h", "b"), "forme": "plein_rect", "courant": False, "non_standard": True,
              "description": "Plat de hauteur h (dans le sens de la flexion forte) et de largeur b."},
    "ROND":  {"libelle": "Rond plein", "groupe": "plein", "norme": "défini par l'utilisateur",
              "dims": ("d",), "forme": "plein_rond", "courant": False, "non_standard": True,
              "description": "Barre ronde pleine."},
}
GROUPES = ["I/H", "U", "L", "creux", "plein"]
LIBELLE_DIM = {"h": "h (mm)", "b": "b (mm)", "tw": "tw (mm)", "tf": "tf (mm)", "r": "r (mm)", "r1": "r1 (mm)",
               "r2": "r2 (mm)", "t": "t (mm)", "d": "d (mm)"}


def pente_upn(h: float) -> float:
    return G.PENTE_UPN if h <= 300.0 else G.PENTE_UPN_GRAND


# ------------------------------------------------------------------ contours
def contours(famille: str, dims: Dict[str, float]) -> List[np.ndarray]:
    """Contours (mm) d'un profilé : extérieur trigonométrique, trous horaires."""
    f = FAMILLES[famille]
    forme = f["forme"]
    d = {k: float(dims[k]) for k in f["dims"]}
    if forme == "I":
        return [G.contour_I(d["h"], d["b"], d["tw"], d["tf"], d.get("r", 0.0))]
    if forme == "I_conique":
        return [G.contour_I_conique(d["h"], d["b"], d["tw"], d["tf"], d["r1"], d["r2"], G.PENTE_IPN)]
    if forme == "U":
        return [G.contour_U(d["h"], d["b"], d["tw"], d["tf"], d.get("r", 0.0))]
    if forme == "U_conique":
        u = d["b"] / 2.0 if d["h"] <= 300.0 else (d["b"] - d["tw"]) / 2.0
        return [G.contour_U_conique(d["h"], d["b"], d["tw"], d["tf"], d["r1"], d["r2"], pente_upn(d["h"]), u)]
    if forme == "L":
        return [G.contour_L(d["h"], d["b"], d["t"], d.get("r1", 0.0), d.get("r2", 0.0))]
    if forme == "creux_rond":
        return G.contours_creux_rond(d["d"], d["t"])
    if forme == "creux_rect":
        re, ri = rayons(famille, d["t"])
        return G.contours_creux_rect(d["b"], d["h"], d["t"], re, ri)
    if forme == "plein_rect":
        return [G.contour_rectangle(d["b"], d["h"])]
    if forme == "plein_rond":
        return [G.cercle(d["d"] / 2)]
    raise ValueError(f"forme inconnue : {forme!r}")


def rayons(famille: str, t: float):
    """Rayons extérieur / intérieur des coins d'un creux rectangulaire selon
    la norme de la famille ; (0, 0) pour un caisson soudé."""
    norme = FAMILLES[famille].get("norme_creux")
    return G.rayons_creux(t, norme) if norme else (0.0, 0.0)


# ------------------------------------------------------------------ calcul complet
def calculer(famille: str, dims: Dict[str, float], torsion_exacte: bool = True,
             solveur=None) -> Dict[str, Optional[float]]:
    """Toutes les propriétés d'un profilé, en unités de catalogue (cm², cm⁴,
    cm³, cm, cm⁶, kg/m, m²/m ; distances en mm) :

    - géométrie (``geometrie.proprietes``) ;
    - A_v,z, A_v,y (EN 1993-1-1 §6.2.6) ;
    - I_t : Saint-Venant sur le contour exact pour les sections ouvertes
      laminées (``torsion_exacte``), Σ b·t³/3 (« mince ») sinon ; Bredt
      (EN 10210-2 annexe B) pour les creux, exact pour les tubes ronds ;
    - I_w et centre de cisaillement : I_z·(h − t_f)²/4 pour les I/H,
      Vlasov sur la ligne moyenne pour les U ; 0 pour les creux et les
      pleins ; non défini (None) pour les cornières (gauchissement
      secondaire négligé, comme les catalogues) ;
    - élancements de parois c/t pour la classification (EN 1993-1-1 Tab. 5.2) ;
    - ``methode_It`` dit la méthode employée pour I_t.
    """
    f = FAMILLES[famille]
    forme = f["forme"]
    d = {k: float(dims[k]) for k in f["dims"]}
    rings = contours(famille, dims)
    p = G.proprietes(rings)
    h, b = d.get("h"), d.get("b")
    tw, tf, t = d.get("tw"), d.get("tf"), d.get("t")
    r = d.get("r", d.get("r1", 0.0))
    # aires de cisaillement
    if forme in ("I", "I_conique"):
        p.update(G.aires_cisaillement("I", h, b, tw, tf, r, p["A"]))
    elif forme in ("U", "U_conique"):
        p.update(G.aires_cisaillement("U", h, b, tw, tf, r, p["A"]))
    elif forme == "creux_rect":
        p.update(G.aires_cisaillement("creux_rect", h, b, t, t, 0.0, p["A"]))
    elif forme == "creux_rond":
        p.update(G.aires_cisaillement("creux_rond", d["d"], d["d"], t, t, 0.0, p["A"]))
    else:
        p.update({"Avz": None, "Avy": None})
    # torsion et gauchissement
    p["Wt"] = None
    if forme == "creux_rond":
        p["It"], p["Wt"] = G.torsion_creux_rond(d["d"], t)
        p["Iw"], p["methode_It"] = 0.0, "exact (tube rond)"
    elif forme == "creux_rect":
        re, ri = rayons(famille, t)
        p["It"], p["Wt"] = G.torsion_creux_rect(b, h, t, re, ri)
        p["Iw"], p["methode_It"] = 0.0, "Bredt, EN 10210-2 annexe B"
    elif forme == "plein_rond":
        p["It"] = math.pi * d["d"] ** 4 / 32.0
        p["Wt"] = p["It"] / (d["d"] / 2)
        p["Iw"], p["methode_It"] = 0.0, "exact (rond plein)"
    else:
        segs = _ligne_moyenne(forme, d)
        if torsion_exacte and forme != "plein_rect" and not f.get("non_standard"):
            e_min = min(v for k, v in d.items() if k in ("tw", "tf", "t") and v > 0)
            pas = min(PAS_MAX, e_min / PAS_RELATIF)
            p["It"] = G.torsion_saint_venant(rings, pas, solveur=solveur)
            p["methode_It"] = f"Saint-Venant, différences finies, pas {pas:.3g} mm"
        elif forme == "plein_rect":
            p["It"] = _torsion_rectangle(b, h)
            p["methode_It"] = "série exacte du rectangle"
        else:
            p["It"] = G.torsion_mince(segs)
            p["methode_It"] = "Σ b·t³/3 (parois minces, sans congé)"
        if forme in ("I", "I_conique"):
            p["Iw"] = p["Iz"] * (h - tf) ** 2 / 4.0
        elif forme in ("U", "U_conique"):
            g = G.gauchissement(segs)
            p["Iw"] = g["Iw"]
            p["y_s"], p["z_s"] = g["y_s"] - p["yc"], g["z_s"] - p["zc"]    # depuis le centre de gravité
        else:
            p["Iw"] = None if forme == "L" else 0.0
    # élancements de parois
    if forme in ("I", "I_conique"):
        p["cf_tf"] = ((b - tw) / 2 - r) / tf
        p["cw_tw"] = (h - 2 * tf - 2 * r) / tw
    elif forme in ("U", "U_conique"):
        p["cf_tf"] = (b - tw - r) / tf
        p["cw_tw"] = (h - 2 * tf - 2 * r) / tw
    elif forme == "L":
        p["cf_tf"] = h / d["t"]
        p["cw_tw"] = b / d["t"]
    elif forme == "creux_rect":
        re, _ = rayons(famille, t)
        p["cf_tf"] = (b - 2 * re) / t
        p["cw_tw"] = (h - 2 * re) / t
    elif forme == "creux_rond":
        p["cf_tf"] = d["d"] / t
        p["cw_tw"] = None
    else:
        p["cf_tf"] = p["cw_tw"] = None
    out = G.en_catalogue(p)
    out["famille"] = famille
    for k, v in d.items():
        out[k] = v
    if forme in ("I", "I_conique"):
        out["d_ame"] = h - 2 * tf - 2 * r                 # hauteur droite de l'âme (mm)
    return out


def _ligne_moyenne(forme: str, d: Dict[str, float]):
    if forme in ("I", "I_conique"):
        return G.ligne_moyenne_I(d["h"], d["b"], d["tw"], d["tf"])
    if forme == "U_conique":
        u = d["b"] / 2.0 if d["h"] <= 300.0 else (d["b"] - d["tw"]) / 2.0
        return G.ligne_moyenne_U_conique(d["h"], d["b"], d["tw"], d["tf"], pente_upn(d["h"]), u)
    if forme == "U":
        return G.ligne_moyenne_U(d["h"], d["b"], d["tw"], d["tf"])
    if forme == "L":
        return G.ligne_moyenne_L(d["h"], d["b"], d["t"])
    return []


def _torsion_rectangle(b: float, h: float) -> float:
    """I_t d'un rectangle plein a × c (a ≥ c), série exacte de Saint-Venant :
    a·c³/3 · (1 − 192·c/(π⁵·a) · Σ_{n impair} tanh(n·π·a/(2c))/n⁵)."""
    a, c = max(b, h), min(b, h)
    s = sum(math.tanh(n * math.pi * a / (2 * c)) / n ** 5 for n in range(1, 60, 2))
    return a * c ** 3 / 3.0 * (1.0 - 192.0 * c / (math.pi ** 5 * a) * s)

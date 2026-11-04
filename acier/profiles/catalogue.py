# -*- coding: utf-8 -*-
"""Le catalogue livré (``acier/donnees/profiles.json``, source unique des
profilés de l'application) et ce qu'une page en fait : colonnes nommées,
filtres, tri, préférences d'affichage enregistrées, et la forme « ancienne »
des enregistrements pour les consommateurs écrits avant le catalogue
(``choix_profile``, filières du plancher, bibliothèques des assemblages).

Sans Streamlit. Unités : celles du fichier (mm ; cm², cm⁴, cm³, cm, cm⁶ ;
kg/m ; m²/m ; degrés).
"""
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from .familles import FAMILLES, GROUPES

_DOSSIER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "donnees")
CHEMIN = os.path.join(_DOSSIER, "profiles.json")
CHEMIN_PREFS = os.path.join(_DOSSIER, "preferences_profiles.json")

# ------------------------------------------------------------------ colonnes
# (clé, libellé, unité, groupe, décimales)
COLONNES: List[Tuple[str, str, str, str, int]] = [
    ("h", "h", "mm", "Dimensions", 1), ("b", "b", "mm", "Dimensions", 1), ("d", "d", "mm", "Dimensions", 1),
    ("tw", "tw", "mm", "Dimensions", 1), ("tf", "tf", "mm", "Dimensions", 1), ("t", "t", "mm", "Dimensions", 1),
    ("r", "r", "mm", "Dimensions", 1), ("r1", "r1", "mm", "Dimensions", 1), ("r2", "r2", "mm", "Dimensions", 1),
    ("d_ame", "d âme", "mm", "Dimensions", 1),
    ("masse", "Masse", "kg/m", "Section", 1), ("A", "A", "cm²", "Section", 2), ("perimetre", "Surface", "m²/m", "Section", 3),
    ("Avz", "Av,z", "cm²", "Section", 2), ("Avy", "Av,y", "cm²", "Section", 2),
    ("Iy", "Iy", "cm⁴", "Flexion y-y", 1), ("Wel_y", "Wel,y", "cm³", "Flexion y-y", 1), ("Wpl_y", "Wpl,y", "cm³", "Flexion y-y", 1),
    ("iy", "iy", "cm", "Flexion y-y", 2),
    ("Iz", "Iz", "cm⁴", "Flexion z-z", 1), ("Wel_z", "Wel,z", "cm³", "Flexion z-z", 1), ("Wpl_z", "Wpl,z", "cm³", "Flexion z-z", 1),
    ("iz", "iz", "cm", "Flexion z-z", 2),
    ("It", "It", "cm⁴", "Torsion", 2), ("Wt", "Wt", "cm³", "Torsion", 1), ("Iw", "Iw", "cm⁶", "Torsion", 0),
    ("Iu", "Iu", "cm⁴", "Axes principaux", 1), ("Iv", "Iv", "cm⁴", "Axes principaux", 1), ("Wel_u", "Wel,u", "cm³", "Axes principaux", 1),
    ("Wel_v", "Wel,v", "cm³", "Axes principaux", 1), ("iu", "iu", "cm", "Axes principaux", 2), ("iv", "iv", "cm", "Axes principaux", 2),
    ("alpha_deg", "α", "°", "Axes principaux", 1),
    ("yc", "yc", "mm", "Position", 1), ("zc", "zc", "mm", "Position", 1), ("y_s", "ys", "mm", "Position", 1), ("z_s", "zs", "mm", "Position", 1),
    ("cf_tf", "cf/tf", "", "Classification", 2), ("cw_tw", "cw/tw", "", "Classification", 2),
]
GROUPES_COLONNES = ["Dimensions", "Section", "Flexion y-y", "Flexion z-z", "Torsion", "Axes principaux", "Position", "Classification"]
AIDE_COLONNES = {
    "d_ame": "Hauteur droite de l'âme entre congés : h − 2·tf − 2·r.",
    "perimetre": "Surface extérieure par mètre (peinture), périmètre du contour extérieur.",
    "Avz": "Aire de cisaillement, effort parallèle à l'âme — EN 1993-1-1 §6.2.6(3), sans le plancher η·hw·tw.",
    "Avy": "Aire de cisaillement, effort parallèle aux semelles — EN 1993-1-1 §6.2.6(3).",
    "Wel_y": "Module élastique minimal (fibre la plus éloignée).",
    "Wel_z": "Module élastique minimal (fibre la plus éloignée : bout d'aile pour un U).",
    "It": "Constante de torsion : Saint-Venant sur le contour exact (laminés), EN 10210-2 annexe B (creux).",
    "Wt": "Module de torsion (creux seulement).",
    "Iw": "Constante de gauchissement : Iz·(h − tf)²/4 (I/H), Vlasov sur la ligne moyenne (U) ; — pour les cornières.",
    "alpha_deg": "Angle de l'axe principal fort u par rapport à y-y.",
    "yc": "Centre de gravité depuis l'origine du contour (dos de l'âme pour un U, talon pour un L).",
    "y_s": "Centre de cisaillement depuis le centre de gravité (U).",
    "cf_tf": "Élancement de la semelle (I/H, U), de la grande aile (L), de la paroi b (creux) — EN 1993-1-1 Tab. 5.2.",
    "cw_tw": "Élancement de l'âme (I/H, U), de la petite aile (L), de la paroi h (creux).",
}
DEFAUT_COLONNES = ["h", "b", "tw", "tf", "masse", "A", "Iy", "Wel_y", "Wpl_y", "iy", "Iz", "Wel_z", "iz", "It", "Avz"]
LIBELLE = {c[0]: c[1] for c in COLONNES}
UNITE = {c[0]: c[2] for c in COLONNES}
DECIMALES = {c[0]: c[4] for c in COLONNES}
GROUPE_DE = {c[0]: c[3] for c in COLONNES}


def colonnes_du_groupe(groupe: str) -> List[str]:
    return [c[0] for c in COLONNES if c[3] == groupe]


# ------------------------------------------------------------------ chargement
@lru_cache(maxsize=1)
def charger() -> Dict:
    with open(CHEMIN, "r", encoding="utf-8") as fh:
        return json.load(fh)


def meta() -> Dict:
    return charger()["meta"]


@lru_cache(maxsize=1)
def profils() -> List[Dict]:
    """Enregistrements À PLAT : les dimensions remontent au premier niveau
    (``rec["h"]``), ``rec["dims"]`` reste disponible."""
    out = []
    for rec in charger()["profils"]:
        plat = dict(rec)
        plat.update(rec["dims"])
        out.append(plat)
    return out


@lru_cache(maxsize=1)
def par_nom() -> Dict[str, Dict]:
    return {p["nom"]: p for p in profils()}


def familles_presentes() -> List[Tuple[str, int]]:
    """[(famille, nombre)] dans l'ordre du registre."""
    n: Dict[str, int] = {}
    for p in profils():
        n[p["famille"]] = n.get(p["famille"], 0) + 1
    return [(f, n[f]) for f in FAMILLES if f in n]


def groupes_presents() -> List[Tuple[str, List[str]]]:
    """[(groupe, [familles présentes])] dans l'ordre des groupes."""
    pres = [f for f, _ in familles_presentes()]
    return [(g, [f for f in pres if FAMILLES[f]["groupe"] == g]) for g in GROUPES
            if any(FAMILLES[f]["groupe"] == g for f in pres)]


def colonnes_utiles(liste: Sequence[Dict], colonnes: Sequence[str]) -> List[str]:
    """Les colonnes demandées qui portent au moins une valeur dans la liste."""
    return [c for c in colonnes if any(p.get(c) is not None for p in liste)]


# ------------------------------------------------------------------ filtres et tri
def filtrer(liste: Sequence[Dict], familles: Optional[Sequence[str]] = None, recherche: str = "",
            bornes: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
            courants_seulement: bool = False) -> List[Dict]:
    """Familles retenues (None = toutes), recherche texte insensible à la
    casse et aux espaces (« ipe200 », « 100x100 »), bornes {clé: (min, max)}
    (None = pas de borne ; un profilé sans la valeur est exclu quand une
    borne porte sur elle), séries courantes seulement."""
    mots = [m for m in recherche.lower().replace(",", ".").split() if m]
    out = []
    for p in liste:
        if familles is not None and p["famille"] not in familles:
            continue
        if courants_seulement and not p.get("courant"):
            continue
        if mots:
            cible = (p["nom"] + " " + p["famille"]).lower()
            compact = cible.replace(" ", "")
            if not all(m in cible or m.replace(" ", "") in compact for m in mots):
                continue
        ok = True
        for cle, (lo, hi) in (bornes or {}).items():
            if lo is None and hi is None:
                continue
            v = p.get(cle)
            if v is None or (lo is not None and v < lo) or (hi is not None and v > hi):
                ok = False
                break
        if ok:
            out.append(p)
    return out


def trier(liste: Sequence[Dict], cle: str, decroissant: bool = False) -> List[Dict]:
    """Tri par une colonne ; les valeurs absentes vont à la fin."""
    avec = [p for p in liste if p.get(cle) is not None]
    sans = [p for p in liste if p.get(cle) is None]
    return sorted(avec, key=lambda p: p[cle], reverse=decroissant) + sans


def ordre_catalogue(liste: Sequence[Dict]) -> List[Dict]:
    """Ordre du fichier (famille puis dimensions croissantes)."""
    rang = {p["nom"]: i for i, p in enumerate(profils())}
    return sorted(liste, key=lambda p: rang.get(p["nom"], 10 ** 9))


# ------------------------------------------------------------------ préférences
PREFS_DEFAUT = {"colonnes": list(DEFAUT_COLONNES), "familles": ["IPE", "HEA", "HEB", "HEM", "UPE", "UPN", "L", "LI", "CHS", "SHS", "RHS"],
                "tri": "", "decroissant": False, "courants_seulement": False, "decimales_fixes": None}


def preferences() -> Dict:
    """Préférences d'affichage enregistrées (fichier) complétées des défauts."""
    d = dict(PREFS_DEFAUT)
    try:
        with open(CHEMIN_PREFS, "r", encoding="utf-8") as fh:
            d.update(json.load(fh))
    except (OSError, ValueError):
        pass
    d["colonnes"] = [c for c in d["colonnes"] if c in LIBELLE] or list(DEFAUT_COLONNES)
    d["familles"] = [f for f in d["familles"] if f in FAMILLES]
    return d


def enregistrer_preferences(d: Dict) -> None:
    """Écrit le fichier des préférences (ce que la page garde par défaut)."""
    propre = {k: d.get(k, PREFS_DEFAUT[k]) for k in PREFS_DEFAUT}
    with open(CHEMIN_PREFS, "w", encoding="utf-8") as fh:
        json.dump(propre, fh, ensure_ascii=False, indent=1)


def retablir_preferences() -> None:
    enregistrer_preferences(PREFS_DEFAUT)


# ------------------------------------------------------------------ forme ancienne
def legacy(familles: Optional[Sequence[str]] = None) -> Dict[str, Dict]:
    """Les enregistrements sous la forme de l'ancien ``profiles_test.json``
    (masse, h, b, tw, tf, r, hi, d, A, Iy, Wel, Wpl_y, iy, Avz, Iz, Wel_z,
    Wpl_z, iz, type, It, Iw) pour les consommateurs écrits avant le
    catalogue. ``type`` = famille. Seules les familles I/H et U portent
    hi et d."""
    out = {}
    for p in profils():
        if familles is not None and p["famille"] not in familles:
            continue
        tf = p.get("tf")
        out[p["nom"]] = {"masse": p.get("masse"), "h": p.get("h"), "b": p.get("b"), "tw": p.get("tw"), "tf": tf,
                         "r": p.get("r", p.get("r1")), "hi": (p["h"] - 2 * tf) if tf and p.get("h") else None,
                         "d": p.get("d_ame"), "A": p.get("A"), "Iy": p.get("Iy"), "Wel": p.get("Wel_y"),
                         "Wpl_y": p.get("Wpl_y"), "iy": p.get("iy"), "Avz": p.get("Avz"), "Iz": p.get("Iz"),
                         "Wel_z": p.get("Wel_z"), "Wpl_z": p.get("Wpl_z"), "iz": p.get("iz"), "type": p["famille"],
                         "It": p.get("It"), "Iw": p.get("Iw")}
    return out


# ------------------------------------------------------------------ formatage
def formater(v, cle: str, decimales: Optional[int] = None) -> str:
    """Nombre en français (virgule, espace fine des milliers), décimales de
    la colonne ou imposées ; « — » si absent."""
    if v is None:
        return "—"
    n = DECIMALES.get(cle, 2) if decimales is None else decimales
    if abs(v) < 0.5 * 10 ** (-n):
        v = 0.0                                   # jamais « -0,0 »
    if cle == "Iw" and v >= 1e5:
        return f"{v / 1e3:,.1f}".replace(",", " ").replace(".", ",") + " ×10³"
    s = f"{v:,.{n}f}".replace(",", " ").replace(".", ",")
    return s

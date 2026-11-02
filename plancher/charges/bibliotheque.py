# -*- coding: utf-8 -*-
"""Bibliothèques du module « Charges sur plancher » : gabarits d'engins,
types de hourdis, plateaux, filières. Sources uniques :

- ``plancher/donnees/engins.json``, ``hourdis.json``, ``plateaux.json`` :
  fiches livrées, GÉNÉRIQUES et nommées comme telles (P-13) ; chaque fiche
  porte ``source`` ; une charge par appui absente vaut 0 et bloque le calcul ;
- les profilés de filière viennent de ``profiles_test.json`` de
  l'application (Iy en cm⁴, Wel en cm³, masse en kg/m), lu directement ;
- les nuances d'acier viennent de ``acier/donnees/bibliotheques.json``.

Les ajouts d'une étude vivent dans le fichier du cas (P-12) : ce module ne
réécrit jamais ses fichiers.
"""
import json
import os
from typing import Dict, List, Optional

from .modele import AppuiGabarit, Gabarit, Plateau, Filiere, Hourdis, Chape, RD, EN_TRAVERS

_ICI = os.path.dirname(os.path.abspath(__file__))
_RACINE = os.path.dirname(os.path.dirname(_ICI))
_DONNEES = os.path.join(os.path.dirname(_ICI), "donnees")


def _lire(nom):
    with open(os.path.join(_DONNEES, nom), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _lire_racine(nom):
    with open(os.path.join(_RACINE, nom), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- engins
def gabarits(extra: Optional[List[Dict]] = None) -> Dict[str, Gabarit]:
    """Gabarits livrés + gabarits du cas (``extra`` : même forme JSON)."""
    out = {}
    for d in _lire("engins.json") + list(extra or []):
        out[d["nom"]] = gabarit_depuis(d)
    return out


def gabarit_depuis(d: Dict) -> Gabarit:
    appuis = [AppuiGabarit(a["nom"], a["dx"], a["dy"], a["a"], a["b"], a.get("P", 0.0), a.get("phi", 1.0))
              for a in d.get("appuis", [])]
    return Gabarit(d["nom"], d.get("type_appuis", "pieds"), appuis, d.get("source", ""),
                   bool(d.get("generique", True)), d.get("notes", ""))


# ---------------------------------------------------------------- hourdis
def types_hourdis(extra: Optional[List[Dict]] = None) -> Dict[str, Dict]:
    out = {}
    for d in _lire("hourdis.json") + list(extra or []):
        out[d["nom"]] = d
    return out


def hourdis_depuis(d: Dict, **surcharges) -> Hourdis:
    """Construit un ``Hourdis`` depuis une fiche de catalogue ; les champs de
    cas (portée, nombre d'éléments, état, chape, parts…) passent en
    ``surcharges``."""
    champs = {
        "b_element": d.get("b_element", 1.2), "h": d["h"], "g": d["g"], "E": d.get("E", 36000.0),
        "I": d.get("I"), "M_Rd": d.get("M_Rd", 0.0), "V_Rd": d.get("V_Rd", 0.0), "nature": d.get("nature", RD),
        "M_Rd_chape": d.get("M_Rd_chape"), "V_Rd_chape": d.get("V_Rd_chape"), "M_cr": d.get("M_cr"),
        "p_loc_adm": d.get("p_loc_adm"), "source": d.get("source", ""), "nom": d["nom"],
    }
    champs.update(surcharges)
    return Hourdis(**champs)


# ---------------------------------------------------------------- plateaux
def plateaux(extra: Optional[List[Dict]] = None) -> Dict[str, Plateau]:
    out = {}
    for d in _lire("plateaux.json") + list(extra or []):
        out[d["nom"]] = Plateau(d["nom"], d.get("f"), d["t"], d["A"], d["Bp"], bool(d.get("rigide_declare", False)),
                                d.get("p_adm_declare"), 0.0, 0.0, d.get("source", ""), d.get("masse", 0.0))
    return out


# ---------------------------------------------------------------- filières (profilés de l'application)
def aciers() -> Dict[str, Dict]:
    return {a["n"]: a for a in _lire_racine(os.path.join("acier", "donnees", "bibliotheques.json"))["aciers"]}


def profils(familles=("IPE", "HEA", "HEB")) -> Dict[str, Dict]:
    """Profilés de ``profiles_test.json`` portant Iy (cm⁴) et Wel (cm³)."""
    raw = _lire_racine("profiles_test.json")
    out = {}
    for nom, p in raw.items():
        if p.get("type") in familles and p.get("Iy") and p.get("Wel"):
            out[nom] = p
    return out


def filiere_profil(nom: str, longueur: float, x: float, y: float, orientation: str = EN_TRAVERS,
                   nuance: str = "S235", refs=None, E: float = 210000.0, gamma_M0: float = 1.0) -> Filiere:
    """Une filière faite d'un profilé du catalogue : EI (kN·m²) = E·Iy, W_el
    (cm³), fy de la nuance, largeur de semelle b (m)."""
    p = profils()[nom]
    fy = aciers()[nuance]["fy"]
    EI = E * 1e3 * p["Iy"] * 1e-8                     # MPa → kN/m² ; cm⁴ → m⁴
    return Filiere(f"{nom} · {longueur:.2f} m", EI, p["Wel"], fy, longueur, x, y, orientation,
                   p["b"] / 1000.0, list(refs or []), gamma_M0, p.get("masse", 0.0), "profiles_test.json")


def candidats_filieres(longueur: float, familles=("IPE", "HEA", "HEB")) -> List[str]:
    """Noms des profilés candidats, par masse croissante."""
    ps = profils(familles)
    return sorted(ps, key=lambda n: (ps[n].get("masse", 0.0), n))

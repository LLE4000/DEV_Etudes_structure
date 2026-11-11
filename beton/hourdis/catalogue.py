# -*- coding: utf-8 -*-
"""Les catalogues de hourdis alvéolés précontraints des fabricants.

La SOURCE UNIQUE d'un catalogue est son fichier ``beton/donnees/hourdis/<fabricant>_<édition>.json``, transcrit hors
production (``tools/transcrire_hourdis_ergon.py`` pour Ergon : transcription à la main, relecture du texte du PDF et
relevé des coupes, qui s'accordent) et mesuré (``docs/validation/hourdis_ergon_transcription_2026-11-11.md``).
``CATALOGUES`` nomme les fichiers livrés : une autre marque, c'est un fichier de plus et une ligne ici (un test refuse
un fichier du dossier que cette liste ne nomme pas).

Ce module LIT : aucune valeur du fabricant n'y est recalculée. Il en tire, sous des noms, trois grandeurs dont les
modules de calcul ont besoin :

- l'inertie d'un élément en m⁴ (I_x du fabricant en mm⁴ × 10⁻¹²) ;
- la largeur d'un élément en m (la largeur NOMINALE du module : 1200 mm chez Ergon, joint compris — les valeurs du
  fabricant sont données par élément) ;
- le module E_cm de la classe de béton du fabricant (EN 1992-1-1 tableau 3.1, ``beton.donnees.ec2.Ecm``), pour la
  raideur relative des éléments sous une filière.
"""
import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from beton.donnees import ec2 as EC

DOSSIER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "donnees", "hourdis")
CATALOGUES = ("ergon_2018.json",)


@lru_cache(maxsize=None)
def _lire(fichier: str) -> Dict:
    with open(os.path.join(DOSSIER, fichier), "r", encoding="utf-8") as fh:
        return json.load(fh)


def cle(fichier: str) -> str:
    """La clé d'un catalogue : le nom de son fichier sans extension (« ergon_2018 »)."""
    return os.path.splitext(fichier)[0]


def catalogues() -> List[Tuple[str, Dict]]:
    """Les catalogues livrés, dans l'ordre de ``CATALOGUES`` : (clé, contenu)."""
    return [(cle(f), _lire(f)) for f in CATALOGUES]


def catalogue(c: str) -> Dict:
    for k, d in catalogues():
        if k == c:
            return d
    raise KeyError(f"catalogue de hourdis inconnu : {c!r}")


def titre(cat: Dict) -> str:
    """« Ergon 2018 »."""
    return f"{cat['meta']['fabricant']} {cat['meta']['edition']}"


def profil(cat: Dict, nom: str) -> Dict:
    for p in cat["profils"]:
        if p["nom"] == nom:
            return p
    raise KeyError(f"profil inconnu dans {titre(cat)} : {nom!r}")


def variante(cat: Dict, nom: str) -> Tuple[Dict, Dict]:
    """(profil, variante) d'une désignation du fabricant (« SP 200-4-2X »)."""
    for p in cat["profils"]:
        for v in p["variantes"]:
            if v["nom"] == nom:
                return p, v
    raise KeyError(f"variante inconnue dans {titre(cat)} : {nom!r}")


# ------------------------------------------------------------------ grandeurs tirées, nommées
def torons(cat: Dict, nom: str) -> List[Tuple[int, float]]:
    """Les torons d'une désignation, lus selon la notation du fabricant (``meta.notation.torons`` : indice → Ø en mm) :
    « SP 200-4-2X » → [(4, 12,5), (2, 9,3)] ; « SP 320-5 » → [(5, 12,5)]. L'indice est lu sans casse (« 6x »)."""
    notation = {k.upper(): float(v) for k, v in cat["meta"]["notation"]["torons"].items()}
    p, _ = variante(cat, nom)
    reste = nom[len(p["nom"]):]
    if not reste.startswith("-"):
        raise ValueError(f"désignation illisible : {nom!r}")
    out = []
    for morceau in reste[1:].split("-"):
        m = re.fullmatch(r"(\d+)([A-Za-z]?)", morceau)
        if not m or m.group(2).upper() not in notation:
            raise ValueError(f"désignation illisible : {nom!r}")
        out.append((int(m.group(1)), notation[m.group(2).upper()]))
    return out


def inertie_element(p: Dict) -> float:
    """m⁴ par élément : I_x du fabricant (mm⁴) × 10⁻¹²."""
    return p["Ix"] * 1.0e-12


def largeur_element(cat: Dict) -> float:
    """m : la largeur nominale du module (les valeurs du fabricant sont données par élément)."""
    return cat["meta"]["largeur_nominale"]["valeur"] / 1000.0


def E_cm(cat: Dict) -> float:
    """MPa : E_cm de la classe du fabricant, EN 1992-1-1 tableau 3.1 (``ec2.Ecm`` en GPa), arrondi au MPa."""
    return float(round(EC.Ecm(cat["meta"]["beton"]["fck"]) * 1000.0))


def nom_type(cat: Dict, v: Dict) -> str:
    """Le nom d'une variante comme type de hourdis d'un autre module : « SP 200-4-2X · Ergon 2018 » (la désignation du
    fabricant d'abord : elle est ce qui reste quand le nom est coupé au « · »)."""
    return f"{v['nom']} · {titre(cat)}"


def source(cat: Dict, p: Dict) -> str:
    """La provenance d'une variante, page par page."""
    m = cat["meta"]
    return (f"{m['fabricant']} · guide technique {m['edition']}, p. {p['page']} (M_Rd, V_Rd,c par élément ; M_Rd avec "
            f"{m['chape']['epaisseur_min']} mm de couche collaborante {m['chape']['classe']}) et p. "
            f"{m['pages']['caracteristiques']} (poids avec joints, I_x) ; béton {m['beton']['classe']}")


def fiche_plancher(c: str, nom: str) -> Dict:
    """La fiche d'une variante pour le module Charges sur plancher (la forme de ``plancher/donnees/hourdis.json``) :
    valeurs de CALCUL par élément (nature Rd) ; M_Rd avec chape, et les conditions du fabricant pour l'admettre
    (``e_chape`` en m, ``fck_exigee`` en MPa) ; V_Rd avec chape NON publié (``None``) ; poids avec joints ; inertie et
    E_cm pour la raideur. Une nouvelle fiche à chaque appel."""
    cat = catalogue(c)
    p, v = variante(cat, nom)
    m = cat["meta"]
    return {"nom": nom_type(cat, v), "generique": False, "catalogue": c, "profil": p["nom"],
            "b_element": largeur_element(cat), "h": p["h"] / 1000.0, "g": p["poids_kN_m2"], "E": E_cm(cat),
            "I": inertie_element(p), "M_Rd": float(v["M_Rd"]), "V_Rd": float(v["V_Rd"]), "nature": m["valeurs"]["nature"],
            "M_Rd_chape": float(v["M_Rd_chape"]) if v.get("M_Rd_chape") is not None else None,
            "V_Rd_chape": float(v["V_Rd_chape"]) if v.get("V_Rd_chape") is not None else None,
            "e_chape": m["chape"]["epaisseur_min"] / 1000.0, "fck_exigee": float(m["chape"]["fck"]),
            "M_cr": None, "p_loc_adm": None, "source": source(cat, p)}


def fiches_plancher() -> List[Dict]:
    """Toutes les variantes de tous les catalogues, comme fiches du module Charges sur plancher, dans l'ordre des
    catalogues puis du fabricant (profil par profil, chaque tableau dans son ordre)."""
    return [fiche_plancher(k, v["nom"]) for k, cat in catalogues() for p in cat["profils"] for v in p["variantes"]]


def groupes_plancher() -> List[Tuple[str, List[str]]]:
    """Les noms de type groupés pour un menu : (« Ergon 2018 · SP 200 », [« SP 200-4X · Ergon 2018 », …])."""
    return [(f"{titre(cat)} · {p['nom']}", [nom_type(cat, v) for v in p["variantes"]])
            for _, cat in catalogues() for p in cat["profils"]]


def chercher_type(nom_type_: str) -> Optional[Tuple[str, Dict, Dict]]:
    """(clé, profil, variante) d'un nom de type « SP 200-4-2X · Ergon 2018 », ou ``None``."""
    for k, cat in catalogues():
        for p in cat["profils"]:
            for v in p["variantes"]:
                if nom_type(cat, v) == nom_type_:
                    return k, p, v
    return None

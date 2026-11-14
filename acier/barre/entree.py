# -*- coding: utf-8 -*-
"""La BARRE en acier telle que l'écran la garde : ses données, leurs défauts, leur relecture (docs/acier/BARRES_ACIER.md
§ 9). Unités de la saisie : m, kN, kN/m, kN·m — celles du bureau.

    {"nom": "B1", "profil": "IPE 300", "nuance": "S235",
     "mode": "chargee" | "efforts",
     # --- barre CHARGÉE : la statique est faite par le module
     "modele": "appuis" | "console" | "enc_app" | "bi_enc" | "continue",      "travees": [6.0],
     "cas": [{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]}, …],
     "poids_propre": True,
     "niveau": "haut" | "centre" | "bas"      où les charges sont appliquées (déversement)
     "maintiens": [{"x": 3.0, "genre": "complet" | "haut" | "bas"}]
     # --- EFFORTS IMPOSÉS : le mode des classeurs du bureau
     "efforts": {"service": {"N", "My", "Mz", "Vz", "Vy"}, "elu": {…}}      N > 0 en compression
     "diagramme": {"y": {"forme": "constant" | "lineaire" | "reparti" | "ponctuel", "psi": 1.0}, "z": {…}}
     # --- commun
     "flambement": {"y": {"L": None, "u": 1.0}, "z": {"L": None, "u": 1.0}}   L None : la plus grande travée (y) ;
                                                                              la plus grande distance entre maintiens (z)
     "Lc": None                               distance entre sections tenues (efforts imposés ; None : L de l'axe faible)
     "methodes": ["historique", "ec3"],
     "historique": {"s": 1.5, "C": "unite" | "norme", "deversement": True, "tolerance": False},
     "parametres": {}}                        d'autres coefficients que ceux de la table (acier/barre/parametres.py)
"""
import copy
import math
from typing import Dict, List

from acier.flambement import normes as NO
from rdm import poutres as PO

from . import combinaisons as CO
from . import ec3_deversement as DV
from . import section as SE
from . import statique as ST
from .resultat import Refus

MODES = ("chargee", "efforts")
METHODES = ("historique", "ec3")
NIVEAUX = ("haut", "centre", "bas")
C_MODES = ("unite", "norme")        # C = 1 (les classeurs du bureau) ; ou C selon le diagramme (doc p. 20, 21)
EFFORTS = ("N", "My", "Mz", "Vz", "Vy")
CAS_MAX, MAINTIENS_MAX = 6, 12
L_MIN, L_MAX = 0.05, 60.0                 # m
FORMES_DIAGRAMME = (DV.CONSTANT, DV.LINEAIRE, DV.REPARTI, DV.PONCTUEL)


def nouvelle_barre(nom: str = "B1") -> Dict:
    return {"nom": nom, "profil": "IPE 300", "nuance": "S235", "mode": "chargee",
            "modele": "appuis", "travees": [6.0],
            "cas": [{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]},
                    {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]}],
            "poids_propre": True, "niveau": "haut", "maintiens": [],
            "efforts": {"service": {k: 0.0 for k in EFFORTS}, "elu": {k: 0.0 for k in EFFORTS}},
            "diagramme": {"y": {"forme": DV.CONSTANT, "psi": 1.0}, "z": {"forme": DV.CONSTANT, "psi": 1.0}},
            "flambement": {"y": {"L": None, "u": 1.0}, "z": {"L": None, "u": 1.0}}, "Lc": None,
            "methodes": list(METHODES), "historique": {"s": 1.5, "C": "unite", "deversement": True, "tolerance": False},
            "parametres": {}}


def _nombre(v, nom: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise Refus(f"{nom} : nombre attendu, reçu {v!r}")
    return float(v)


def normaliser(barre: Dict) -> Dict:
    """La barre complétée de ses défauts et contrôlée (une COPIE) ; Refus dit ce qui ne va pas."""
    if not isinstance(barre, dict):
        raise Refus("la barre doit être un dictionnaire")
    b = nouvelle_barre(str(barre.get("nom") or "B1"))
    src = copy.deepcopy(barre)
    for k in ("profil", "nuance", "mode", "modele", "poids_propre", "niveau", "Lc"):
        if k in src:
            b[k] = src[k]
    if b["mode"] not in MODES:
        raise Refus(f"mode inconnu : {b['mode']!r}")
    if b["nuance"] not in NO.NUANCES:
        raise Refus(f"nuance inconnue : {b['nuance']!r}")
    if not isinstance(b["profil"], str) or SE.CAT.par_nom().get(b["profil"]) is None:
        raise Refus(f"profilé inconnu du catalogue : {b['profil']!r}")
    if b["niveau"] not in NIVEAUX:
        raise Refus(f"niveau des charges : {', '.join(NIVEAUX)} — reçu {b['niveau']!r}")
    b["poids_propre"] = bool(b["poids_propre"])
    if b["Lc"] is not None and not _nombre(b["Lc"], "L_c") >= 0:
        raise Refus("la distance L_c entre sections tenues doit être positive ou nulle")
    meth = src.get("methodes", list(METHODES))
    if not isinstance(meth, (list, tuple)) or not meth or any(m not in METHODES for m in meth):
        raise Refus(f"méthodes : une ou deux parmi {', '.join(METHODES)}")
    b["methodes"] = [m for m in METHODES if m in meth]
    h = dict(b["historique"], **(src.get("historique") or {}))
    if not 1.0 <= _nombre(h["s"], "s") <= 3.0:
        raise Refus("le coefficient de sécurité s doit être entre 1 et 3")
    if h["C"] not in C_MODES:
        raise Refus(f"coefficient C : {', '.join(C_MODES)}")
    b["historique"] = {"s": float(h["s"]), "C": h["C"], "deversement": bool(h["deversement"]), "tolerance": bool(h["tolerance"])}
    if not isinstance(src.get("parametres", {}), dict):
        raise Refus("« parametres » doit être un dictionnaire")
    b["parametres"] = dict(src.get("parametres") or {})
    fl = src.get("flambement") or {}
    for a in ("y", "z"):
        ax = dict(b["flambement"][a], **(fl.get(a) or {}))
        if ax["L"] is not None and not _nombre(ax["L"], f"longueur de flambement {a}") >= 0:
            raise Refus(f"axe {a} : la longueur de flambement doit être positive ou nulle")
        if not _nombre(ax["u"], f"coefficient u de l'axe {a}") > 0:
            raise Refus(f"axe {a} : le coefficient u doit être positif")
        b["flambement"][a] = {"L": None if ax["L"] is None else float(ax["L"]), "u": float(ax["u"])}
    if b["mode"] == "efforts":
        ef = src.get("efforts") or {}
        for jeu in ("service", "elu"):
            for k in EFFORTS:
                b["efforts"][jeu][k] = _nombre((ef.get(jeu) or {}).get(k, 0.0), f"{k} ({jeu})")
        dg = src.get("diagramme") or {}
        for a in ("y", "z"):
            d = dict(b["diagramme"][a], **(dg.get(a) or {}))
            if d["forme"] not in FORMES_DIAGRAMME:
                raise Refus(f"diagramme du moment {a} : {', '.join(FORMES_DIAGRAMME)}")
            if not -1.0 <= _nombre(d["psi"], "ψ") <= 1.0:
                raise Refus("ψ = M₂/M₁ doit être entre −1 et 1")
            b["diagramme"][a] = {"forme": d["forme"], "psi": float(d["psi"])}
        for a in ("y", "z"):
            if b["flambement"][a]["L"] is None:
                raise Refus(f"efforts imposés : la longueur de flambement de l'axe {a} est à donner")
        return b
    # ---- barre chargée
    if b["modele"] not in PO.MODELES:
        raise Refus(f"modèle inconnu : {b['modele']!r}")
    tr = src.get("travees", b["travees"])
    if not isinstance(tr, (list, tuple)) or not tr:
        raise Refus("la barre a au moins une travée")
    b["travees"] = [_nombre(x, "portée") for x in tr]
    if any(not L_MIN <= x <= L_MAX for x in b["travees"]):
        raise Refus(f"chaque portée doit être entre {L_MIN:g} et {L_MAX:g} m")
    if b["modele"] == "continue":
        if not 2 <= len(b["travees"]) <= PO.TRAVEES_MAX:
            raise Refus(f"une poutre continue a de 2 à {PO.TRAVEES_MAX} travées")
    elif len(b["travees"]) != 1:
        raise Refus("ce modèle n'a qu'une travée")
    Lt = sum(b["travees"])
    cas = src.get("cas", b["cas"])
    if not isinstance(cas, (list, tuple)) or not cas or len(cas) > CAS_MAX:
        raise Refus(f"de 1 à {CAS_MAX} cas de charge")
    b["cas"] = []
    noms = set()
    for i, c in enumerate(cas):
        if not isinstance(c, dict):
            raise Refus(f"cas {i + 1} : dictionnaire attendu")
        nom = str(c.get("nom") or f"C{i + 1}")
        if nom in noms or nom == ST.POIDS_PROPRE:
            raise Refus(f"deux cas de charge portent le nom « {nom} »")
        noms.add(nom)
        if c.get("nature") not in CO.NATURES:
            raise Refus(f"cas « {nom} » : nature inconnue ({c.get('nature')!r})")
        try:
            _, charges = PO.normaliser(b["modele"], b["travees"], c.get("charges") or [])
        except PO.ErreurDonnees as e:
            raise Refus(f"cas « {nom} » : {e}") from None
        charges = [{k: ch[k] for k in ("type", "v", "a", "b", "sens") if k in ch} for ch in charges]
        b["cas"].append({"nom": nom, "nature": c["nature"], "N": _nombre(c.get("N", 0.0), f"effort normal du cas « {nom} »"), "charges": charges})
    mts = src.get("maintiens", [])
    if not isinstance(mts, (list, tuple)) or len(mts) > MAINTIENS_MAX:
        raise Refus(f"au plus {MAINTIENS_MAX} maintiens")
    b["maintiens"] = []
    for m in mts:
        x = _nombre((m or {}).get("x"), "abscisse d'un maintien")
        genre = (m or {}).get("genre", "complet")
        if genre not in ST.GENRES_MAINTIEN:
            raise Refus(f"maintien : genre parmi {', '.join(ST.GENRES_MAINTIEN)}")
        if not 0.0 <= x <= Lt:
            raise Refus(f"maintien à {x:g} m : hors de la barre (0 à {Lt:g} m)")
        b["maintiens"].append({"x": x, "genre": genre})
    b["maintiens"].sort(key=lambda m: m["x"])
    return b

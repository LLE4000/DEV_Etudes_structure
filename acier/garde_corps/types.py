# -*- coding: utf-8 -*-
"""Les TYPES de garde-corps du menu : chacun fixe le système (montants verticaux ou lisse
ancrée au mur) et le remplissage (barreaux, lisses, panneau, aucun) — le reste du cas
(hauteur, tronçons, charges, sections, fixation) se règle ensuite. Un type est un préréglage,
pas une contrainte : les champs restent libres après le choix."""
from typing import Dict

TYPES: Dict[str, Dict] = {
    "barreaudage": {"libelle": "Barreaudage vertical", "systeme": "montants", "remplissage": "barreaux",
                    "description": "Montants verticaux, main courante, lisse basse et barreaux verticaux entre les deux."},
    "lisses": {"libelle": "Lisses horizontales", "systeme": "montants", "remplissage": "lisses",
               "description": "Montants verticaux, main courante et lisses intermédiaires horizontales."},
    "panneau": {"libelle": "Remplissage en panneau (verre, tôle)", "systeme": "montants", "remplissage": "panneau",
                "description": "Montants verticaux, main courante, lisse basse et un panneau justifié par son fournisseur."},
    "main_courante": {"libelle": "Main courante seule sur montants", "systeme": "montants", "remplissage": "aucun",
                      "description": "Rampe ou lisse de protection : montants verticaux et main courante, sans remplissage."},
    "murale": {"libelle": "Main courante murale sur consoles", "systeme": "mur", "remplissage": "aucun",
               "description": "Main courante portée par des consoles ancrées dans un mur (rampe d'escalier)."},
}
TYPE_DEFAUT = "barreaudage"


def appliquer_type(cas: Dict, cle: str) -> None:
    """Pose le système et le remplissage du type sur le cas (les autres champs sont gardés)."""
    t = TYPES.get(cle) or TYPES[TYPE_DEFAUT]
    cas["type"] = cle if cle in TYPES else TYPE_DEFAUT
    cas["geometrie"]["systeme"] = t["systeme"]
    cas["remplissage"]["type"] = t["remplissage"]


def type_de(cas: Dict) -> str:
    """Le type d'un cas : celui enregistré s'il est cohérent, sinon déduit du système et du remplissage."""
    cle = cas.get("type")
    sys_, rp = cas["geometrie"].get("systeme", "montants"), cas["remplissage"].get("type", "barreaux")
    if cle in TYPES and TYPES[cle]["systeme"] == sys_ and TYPES[cle]["remplissage"] == rp:
        return cle
    for k, t in TYPES.items():
        if t["systeme"] == sys_ and t["remplissage"] == rp:
            return k
    return "murale" if sys_ == "mur" else TYPE_DEFAUT


def libelle(cas: Dict) -> str:
    return TYPES[type_de(cas)]["libelle"]

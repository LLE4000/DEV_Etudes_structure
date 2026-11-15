# -*- coding: utf-8 -*-
"""Les ÉCUSSONS (cahier § 9) : un petit bouclier, un pictogramme et un mot — jamais la couleur seule.

| Écusson         | Quand                                                                                               |
|-----------------|-----------------------------------------------------------------------------------------------------|
| À jour          | la date du jour est dans la période de validité donnée par le loueur ; sans période, l'année du prix |
|                 | est au moins l'année en cours                                                                       |
| Pas à jour      | période dépassée, ou année du prix avant l'année en cours                                           |
| À vérifier      | année inconnue, valeur « sur demande », ou case « à vérifier » cochée à la main                     |
| Manquant        | prix vide alors qu'on en a besoin                                                                    |
| Corrigé         | la valeur vient de nous ; elle prend l'année de la correction                                        |
| Demande envoyée | sur le dossier d'un loueur, entre l'envoi du fichier et l'import de sa réponse                      |

Un ENSEMBLE de prix (un dossier de loueur, une grue, un calcul) : « Pas à jour » si au moins un prix l'est ; sinon
« À vérifier » si au moins un l'est ; sinon « À jour ». Une correction compte selon son année (elle est récente)."""
import datetime as dt
from typing import Dict, Iterable, Optional, Tuple

from . import valeurs as V
from .periode import lire_date

A_JOUR = "a_jour"
PAS_A_JOUR = "pas_a_jour"
A_VERIFIER = "a_verifier"
MANQUANT = "manquant"
CORRIGE = "corrige"
DEMANDE = "demande"

LIBELLES = {A_JOUR: "À jour", PAS_A_JOUR: "Pas à jour", A_VERIFIER: "À vérifier", MANQUANT: "Manquant",
            CORRIGE: "Corrigé", DEMANDE: "Demande envoyée"}
ORDRE = (A_JOUR, A_VERIFIER, PAS_A_JOUR, MANQUANT, CORRIGE, DEMANDE)


def _validite(p: Dict) -> Tuple[Optional[dt.date], Optional[dt.date]]:
    v = p.get("validite") or {}
    return lire_date(v.get("du")), lire_date(v.get("au"))


def selon_annee(annee, aujourdhui: dt.date) -> str:
    """L'écusson d'une année de prix : À jour (au moins l'année en cours), Pas à jour (avant), À vérifier (inconnue)."""
    try:
        a = int(annee)
    except (TypeError, ValueError):
        return A_VERIFIER
    return A_JOUR if a >= aujourdhui.year else PAS_A_JOUR


def ecusson(p: Optional[Dict], aujourdhui: dt.date, besoin: bool = True) -> Optional[str]:
    """L'écusson d'un PRIX (``base.prix_*`` : {valeur, annee, validite, corrige, a_verifier…}).

    ``besoin`` : la case doit être remplie (un prix vide est alors « Manquant ») ; une case « non disponible » n'a pas
    d'écusson (rien à demander)."""
    if p is None:
        return MANQUANT if besoin else None
    n = V.nature(p.get("valeur"))
    if p.get("corrige"):
        return CORRIGE
    if n == V.INCONNU:
        return MANQUANT if besoin else None
    if n == V.NON_DISPONIBLE:
        return None
    if n == V.SUR_DEMANDE or p.get("a_verifier"):
        return A_VERIFIER
    du, au = _validite(p)
    if du or au:
        if au and aujourdhui > au:
            return PAS_A_JOUR
        if du and aujourdhui < du:
            return A_VERIFIER
        return A_JOUR
    return selon_annee(p.get("annee"), aujourdhui)


def pour_ensemble(e: Optional[str], p: Optional[Dict], aujourdhui: dt.date) -> Optional[str]:
    """Ce qu'un écusson pèse dans un ensemble : une correction compte selon son année."""
    if e == CORRIGE:
        return selon_annee((p or {}).get("annee"), aujourdhui) if (p or {}).get("annee") else A_JOUR
    return e


def agreger(ecussons: Iterable[Optional[str]], vide: str = PAS_A_JOUR) -> str:
    """L'écusson d'un ENSEMBLE de prix (déjà passés par ``pour_ensemble``) : Pas à jour > À vérifier > À jour. Un
    ensemble SANS AUCUN prix connu (un loueur dont on n'a encore rien reçu) prend ``vide``."""
    vus = [e for e in ecussons if e and e != MANQUANT]
    if not vus:
        return vide
    if PAS_A_JOUR in vus:
        return PAS_A_JOUR
    if A_VERIFIER in vus:
        return A_VERIFIER
    return A_JOUR


def compter(ecussons: Iterable[Optional[str]]) -> Dict[str, int]:
    """Le nombre de prix par écusson."""
    out = {k: 0 for k in ORDRE}
    for e in ecussons:
        if e:
            out[e] = out.get(e, 0) + 1
    return out

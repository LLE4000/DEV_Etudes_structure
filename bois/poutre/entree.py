# -*- coding: utf-8 -*-
"""Poutre bois — l'ENTRÉE du moteur : la poutre neuve (l'écran Streamlit au premier affichage) et la normalisation
stricte d'une poutre reçue (l'écran du site, un fichier .stab).

Règle de ``normaliser`` : une clé ABSENTE (ou ``null``) prend le défaut de l'écran ; une valeur HORS DU DOMAINE de
l'écran (classe inconnue, section absente de la liste, nombre hors des bornes de son ``st.number_input``, texte au
lieu d'un nombre, clé inconnue…) lève ``ValueError`` avec un message en français. L'écran Streamlit refuse lui aussi
une saisie hors bornes (son champ revient à la borne) : le moteur ne ramène JAMAIS une valeur en silence.
"""
import copy
import math
from typing import Dict

from . import donnees as D

CLES = frozenset(D.DEFAUTS)
LIBELLES = {
    "nom": "le nom", "classe": "la classe de résistance", "service": "la classe de service", "duree": "la durée dominante",
    "standard": "le choix de section", "section": "la section standard", "b": "la largeur b", "h": "la hauteur h",
    "L": "la portée L", "G": "la charge permanente G", "Q": "la charge d'exploitation Q", "gamma_G": "γ_G",
    "gamma_Q": "γ_Q", "psi2": "ψ₂", "lim_inst": "la limite instantanée", "lim_fin": "la limite finale",
    "appui": "l'option d'appui", "a": "la largeur d'appui a",
}
UNITES = {"b": "mm", "h": "mm", "L": "m", "G": "kN/m", "Q": "kN/m", "a": "mm", "gamma_G": "", "gamma_Q": "", "psi2": ""}
TAGS = tuple(s["tag"] for s in D.SECTIONS_STD)


def nouvelle_poutre() -> Dict:
    """La poutre de l'écran au premier affichage : C24, classe de service 1, durée « long », section standard
    38x150 (la première du menu), L = 4 m, G = 2 et Q = 3 kN/m, γ_G 1,35, γ_Q 1,5, ψ₂ 0,3, L/300 et L/200, appui
    non vérifié (a = 60 mm), section sur mesure 80 × 240 mm gardée en réserve."""
    return copy.deepcopy(D.DEFAUTS)


def section_standard(tag: str) -> Dict:
    """La section standard ``tag`` (« 45x220 ») de la liste de l'écran : {tag, b, h} (mm, entiers)."""
    for s in D.SECTIONS_STD:
        if s["tag"] == tag:
            return dict(s)
    raise ValueError(f"section « {tag} » absente de la liste ({', '.join(TAGS)})")


def _fr(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _nombre(x, cle: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{LIBELLES[cle]} : nombre attendu (reçu {x!r})")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{LIBELLES[cle]} : nombre fini attendu (reçu {x!r})")
    _, mini, maxi = D.CHAMPS[cle]
    if v < mini or v > maxi:
        u = f" {UNITES[cle]}" if UNITES[cle] else ""
        raise ValueError(f"{LIBELLES[cle][:1].upper() + LIBELLES[cle][1:]} doit rester entre {_fr(mini)} et {_fr(maxi)}{u}, "
                         f"comme dans l'écran Streamlit (reçu {_fr(v)})")
    return v


def _parmi(x, cle: str, permis) -> object:
    # le type compte : 1.0 ou True ne sont pas la classe de service 1 (une clé de k_mod, un entier de l'écran)
    if isinstance(x, bool) or not any(type(x) is type(q) and x == q for q in permis):
        raise ValueError(f"{LIBELLES[cle]} : {x!r} n'est pas proposé par l'écran ({', '.join(str(p) for p in permis)})")
    return x


def normaliser(poutre: Dict) -> Dict:
    """Une copie COMPLÈTE et CONTRÔLÉE de la poutre (voir la règle en tête du fichier).
    ``normaliser(normaliser(p)) == normaliser(p)``."""
    if not isinstance(poutre, dict):
        raise ValueError(f"poutre : dictionnaire attendu (reçu {type(poutre).__name__})")
    inconnues = sorted(str(k) for k in poutre if k not in CLES)
    if inconnues:
        raise ValueError(f"poutre : clé(s) inconnue(s) {', '.join(inconnues)}")

    def val(k):
        v = poutre.get(k)
        return D.DEFAUTS[k] if v is None else v

    nom = val("nom")
    if not isinstance(nom, str):
        raise ValueError(f"le nom : texte attendu (reçu {nom!r})")
    p = {"nom": nom,
         "classe": _parmi(val("classe"), "classe", tuple(D.CLASSES)),
         "service": _parmi(val("service"), "service", D.CLASSES_SERVICE),
         "duree": _parmi(val("duree"), "duree", D.DUREES)}
    if not isinstance(val("standard"), bool):
        raise ValueError(f"le choix de section : vrai ou faux attendu (reçu {val('standard')!r})")
    p["standard"] = val("standard")
    p["section"] = _parmi(val("section"), "section", TAGS)
    for k in ("b", "h", "L", "G", "Q", "gamma_G", "gamma_Q", "psi2"):
        p[k] = _nombre(val(k), k)
    p["lim_inst"] = _parmi(val("lim_inst"), "lim_inst", D.LIMITES_INST)
    p["lim_fin"] = _parmi(val("lim_fin"), "lim_fin", D.LIMITES_FIN)
    if not isinstance(val("appui"), bool):
        raise ValueError(f"l'option d'appui : vrai ou faux attendu (reçu {val('appui')!r})")
    p["appui"] = val("appui")
    p["a"] = _nombre(val("a"), "a")
    return p

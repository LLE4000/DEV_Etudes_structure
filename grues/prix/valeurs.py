# -*- coding: utf-8 -*-
"""Ce qu'une case de prix contient (cahier § 5.3) :

- vide ou ``None`` = INCONNU (jamais 0 € : un prix inconnu reste vide et il est signalé) ;
- ``0`` = COMPRIS (on écrit « compris ») ;
- ``"non disponible"`` = n'existe pas pour ce modèle ;
- ``"sur demande"`` = prix à demander au loueur ;
- un nombre = le prix ;
- tout autre texte (« voir fondations ») = pas un prix : il se lit comme une information.

``lire_mot`` lit ce qu'un loueur ou un chantier écrit dans une case (« 9 000 € », « 1.234,56 », « compris »…) — les mêmes
mots partout (cahier § 7.2, § 7.3)."""
import re
from typing import Any, Optional, Tuple

INCONNU = "inconnu"
COMPRIS = "compris"
NON_DISPONIBLE = "non_disponible"
SUR_DEMANDE = "sur_demande"
NOMBRE = "nombre"
TEXTE = "texte"

MOT_NON_DISPONIBLE = "non disponible"
MOT_SUR_DEMANDE = "sur demande"

_COMPRIS = ("compris", "inclus", "inbegrepen", "inclusief")
_NON_DISPO = ("non disponible", "nd", "n.d.", "-", "—", "nb", "niet beschikbaar")
_SUR_DEMANDE = ("sur demande", "sur devis", "op aanvraag", "op aanvraag.", "sur demande.")


def est_nombre(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def nature(v: Any) -> str:
    """La nature d'une valeur de la base : INCONNU, COMPRIS, NON_DISPONIBLE, SUR_DEMANDE, NOMBRE ou TEXTE."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return INCONNU
    if est_nombre(v):
        return COMPRIS if v == 0 else NOMBRE
    if isinstance(v, str):
        t = v.strip().lower()
        if t in _NON_DISPO:
            return NON_DISPONIBLE
        if t in _SUR_DEMANDE:
            return SUR_DEMANDE
        if t in _COMPRIS:
            return COMPRIS
    return TEXTE


def montant(v: Any) -> Optional[float]:
    """Le montant d'une valeur : un nombre, 0 si « compris », ``None`` sinon (inconnu, non disponible, sur demande…)."""
    n = nature(v)
    if n == NOMBRE:
        return float(v)
    if n == COMPRIS:
        return 0.0
    return None


_NB = re.compile(r"^[+-]?\d[\d\s  .,']*$")


def lire_nombre(texte: str) -> Optional[float]:
    """« 9 000 € », « 1.234,56 », « 6T », « 27,5 » → un nombre ; ``None`` si ce n'est pas un nombre lisible."""
    t = str(texte).strip().replace("€", "").replace("EUR", "").strip()
    t = re.sub(r"\s*(?:t|T|m|h|kg|m³|m²|j|/j|/jour|/h)\s*$", "", t).strip()
    if not t or not _NB.match(t):
        return None
    t = t.replace(" ", "").replace(" ", "").replace(" ", "").replace("'", "")
    if "," in t and "." in t:
        # le dernier séparateur est la virgule décimale : « 1.234,56 » ; sinon « 1,234.56 »
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        t = t.replace(",", ".") if t.count(",") == 1 else t.replace(",", "")
    elif t.count(".") > 1:
        t = t.replace(".", "")
    elif "." in t and len(t.split(".")[1]) == 3 and len(t.split(".")[0]) <= 3 and t.split(".")[0] not in ("0", ""):
        # « 1.234 » : un séparateur de milliers (à la belge), pas une décimale
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def lire_mot(texte: Any) -> Tuple[str, Any]:
    """Ce qu'on lit dans une case remplie par un loueur ou un chantier : (nature, valeur de la base).

    Un nombre → (NOMBRE, x) ; « compris »/« inclus » → (COMPRIS, 0) ; « non disponible »/« nd »/« - » → (NON_DISPONIBLE,
    "non disponible") ; « sur demande »/« sur devis » → (SUR_DEMANDE, "sur demande") ; vide → (INCONNU, None) ; tout le
    reste (« ± 30 ») → (TEXTE, le texte) : une valeur NON LUE, montrée telle quelle, à saisir à la main."""
    if texte is None:
        return INCONNU, None
    if est_nombre(texte):
        return (COMPRIS, 0) if texte == 0 else (NOMBRE, float(texte))
    t = str(texte).strip()
    if not t:
        return INCONNU, None
    b = t.lower()
    if b in _COMPRIS:
        return COMPRIS, 0
    if b in _NON_DISPO:
        return NON_DISPONIBLE, MOT_NON_DISPONIBLE
    if b in _SUR_DEMANDE:
        return SUR_DEMANDE, MOT_SUR_DEMANDE
    x = lire_nombre(t)
    if x is not None:
        return (COMPRIS, 0) if x == 0 else (NOMBRE, x)
    return TEXTE, t

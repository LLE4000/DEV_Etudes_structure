# -*- coding: utf-8 -*-
"""Ce qu'un élément « Choix de profilé » contient : les listes de l'écran Streamlit (``modules/choix_profile.py``),
ses valeurs par défaut, et la NORMALISATION STRICTE d'un élément reçu (le site, un fichier .stab).

Un élément (unités de l'écran : kN, kN·m, m, cm⁴) :

    {"nom": "P1", "familles": ["HEA"], "acier": "S235",
     "My": 0.0, "Vz": 0.0,                                   # M_y,Ed et V_z,Ed, les efforts « par défaut »
     "avec_N": false, "N": 0.0, "avec_Mz": false, "Mz": 0.0,  # les efforts « optionnels » : une case, une valeur
     "avec_Vy": false, "Vy": 0.0, "avec_T": false, "T": 0.0,
     "gamma_M0": 1.0, "gamma_M1": 1.0, "Iy_min": 0.0,
     "flambement": false, "Lcr_y": 0.0, "Lcr_z": 0.0, "courbe": "b",
     "deversement": false, "L_LT": 0.0,
     "profil": null}                                          # null : le profilé que l'écran retient (son optimal)

Une valeur hors du domaine de l'écran est REFUSÉE (ValueError, avec le nom du champ) : γ hors de [0,8 ; 1,5], I_y min,
L_cr ou L_LT négatifs, famille, acier ou courbe inconnus, clé inconnue, texte ou booléen à la place d'un nombre, nombre
infini. Les efforts peuvent être négatifs (l'écran l'accepte : N positif en compression).

Le site refuse EN PLUS ce que l'écran accepte mais ne sait pas calculer (``hors_domaine``) : un effort au-delà de
10⁶ kN ou kN·m, une longueur non nulle sous 1 mm ou au-delà de 1 000 m, I_y min au-delà de 10⁸ cm⁴ — l'écran y
déborde (OverflowError, division par zéro) ou écrit des taux de cent chiffres.
"""
import copy
import math
from typing import Dict, List, Optional

from . import calcul as CA

ACIERS = ("S235", "S275", "S355")
# les courbes de flambement de l'écran : clé, α, libellé du module
COURBES = (("a0", 0.13, "a0 (0.13)"), ("a", 0.21, "a (0.21)"), ("b", 0.34, "b (0.34)"), ("c", 0.49, "c (0.49)"), ("d", 0.76, "d (0.76)"))
ALPHA = {k: a for k, a, _ in COURBES}
LIBELLE_COURBE = {k: t for k, _, t in COURBES}
COURBE_DEFAUT = "b"                      # index=2 du selectbox
ALPHA_SANS_FLAMBEMENT = 0.34             # alpha_buck quand la case est décochée (sans effet : rien n'est calculé)
GAMMA_MIN, GAMMA_MAX = 0.8, 1.5          # min_value / max_value de γM0 et γM1
NOM_DEFAUT = "P1"

REELS = ("My", "Vz", "N", "Mz", "Vy", "T", "gamma_M0", "gamma_M1", "Iy_min", "Lcr_y", "Lcr_z", "L_LT")
POSITIFS = ("Iy_min", "Lcr_y", "Lcr_z", "L_LT")          # min_value=0.0 dans l'écran
EFFORTS = ("My", "Vz", "N", "Mz", "Vy", "T")
LONGUEURS = ("Lcr_y", "Lcr_z", "L_LT")
# Le DOMAINE DU SITE, plus étroit que celui de l'écran (qui ne borne ni les efforts ni les longueurs par le haut) : au-delà,
# le calcul de l'écran DÉBORDE — M_y = 1e303 kN·m : OverflowError ; L_cr = 1e150 m : OverflowError ; L_cr = 1e-200 m :
# division par zéro (N_cr) — ou rend des taux de cent chiffres et des textes de trois cents (I_y min = 1e308). Ces
# saisies sont REFUSÉES avec le nom du champ (règle 2 du contrat : jamais ramenées en silence). Les bornes restent loin
# de tout cas réel : 10⁶ kN est seize fois le N_Rd du plus fort profilé du catalogue (HL 920x1377 en S355 : 62 257 kN).
EFFORT_MAX = 1e6           # kN, kN·m, en valeur absolue
LONGUEUR_MIN = 1e-3        # m : une longueur non nulle d'au moins 1 mm (0 : pas de flambement selon cet axe)
LONGUEUR_MAX = 1e3         # m
IY_MIN_MAX = 1e8           # cm⁴ (1 m⁴ ; le plus fort profilé du catalogue n'en a pas 2 %)
CASES = ("avec_N", "avec_Mz", "avec_Vy", "avec_T", "flambement", "deversement")
CLES = ("nom", "familles", "acier") + REELS + CASES + ("courbe", "profil")


def familles_disponibles() -> List[str]:
    """Les familles proposées par l'écran : celles des profilés chargés, triées (``sorted``)."""
    return sorted({p["type"] for p in CA._profils().values() if p.get("type")})


def familles_defaut() -> List[str]:
    dispo = familles_disponibles()
    return ["HEA"] if "HEA" in dispo else dispo[:1]


def nouvel_element(nom: str = NOM_DEFAUT) -> Dict:
    """L'élément tel que l'écran l'affiche à l'ouverture."""
    return {"nom": nom, "familles": familles_defaut(), "acier": "S235", "My": 0.0, "Vz": 0.0,
            "avec_N": False, "N": 0.0, "avec_Mz": False, "Mz": 0.0, "avec_Vy": False, "Vy": 0.0, "avec_T": False, "T": 0.0,
            "gamma_M0": 1.0, "gamma_M1": 1.0, "Iy_min": 0.0,
            "flambement": False, "Lcr_y": 0.0, "Lcr_z": 0.0, "courbe": COURBE_DEFAUT, "deversement": False, "L_LT": 0.0,
            "profil": None}


def _reel(cle: str, x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"« {cle} » doit être un nombre : {x!r}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"« {cle} » doit être un nombre fini : {x!r}")
    return x


def hors_domaine(cle: str, x: float) -> Optional[str]:
    """Pourquoi la valeur ``x`` du champ ``cle`` sort du DOMAINE DU SITE (``EFFORT_MAX``…) ; None si elle y est."""
    if cle in EFFORTS and abs(x) > EFFORT_MAX:
        return f"« {cle} » vaut {x:g} : au plus {EFFORT_MAX:g} en valeur absolue (borne du site : au-delà, le calcul déborde)"
    if cle in LONGUEURS and (x > LONGUEUR_MAX or 0 < x < LONGUEUR_MIN):
        return (f"« {cle} » vaut {x:g} m : 0, ou de {LONGUEUR_MIN:g} à {LONGUEUR_MAX:g} m (borne du site : au-delà, "
                f"le calcul déborde)")
    if cle == "Iy_min" and x > IY_MIN_MAX:
        return f"« {cle} » vaut {x:g} cm⁴ : au plus {IY_MIN_MAX:g} cm⁴ (borne du site)"
    return None


def hors_domaine_site(el: Dict) -> Optional[str]:
    """La première valeur de l'élément (déjà lue en réels) qui sort du domaine du site ; None s'il n'y en a pas."""
    for k in EFFORTS + LONGUEURS + ("Iy_min",):
        m = hors_domaine(k, el[k])
        if m:
            return m
    return None


def normaliser(element: Dict) -> Dict:
    """L'élément complété (une clé absente ou null prend la valeur de l'écran) et contrôlé ; une COPIE."""
    if not isinstance(element, dict):
        raise ValueError(f"un élément est un dictionnaire, pas {type(element).__name__}")
    inconnues = sorted(set(element) - set(CLES))
    if inconnues:
        raise ValueError(f"clé(s) inconnue(s) : {', '.join(inconnues)}")
    el = nouvel_element()
    for k, v in element.items():
        if v is not None:
            el[k] = copy.deepcopy(v)
    if not isinstance(el["nom"], str):
        raise ValueError(f"« nom » doit être un texte : {el['nom']!r}")
    dispo = familles_disponibles()
    fams = el["familles"]
    if not isinstance(fams, list) or not all(isinstance(f, str) for f in fams):
        raise ValueError(f"« familles » doit être une liste de familles : {fams!r}")
    hors = [f for f in fams if f not in dispo]
    if hors:
        raise ValueError(f"famille(s) inconnue(s) : {', '.join(hors)} (l'écran propose {', '.join(dispo)})")
    el["familles"] = [f for f in dispo if f in fams]          # sans doublon, dans l'ordre de l'écran
    if el["acier"] not in ACIERS:
        raise ValueError(f"acier inconnu : {el['acier']!r} (l'écran propose {', '.join(ACIERS)})")
    for k in REELS:
        el[k] = _reel(k, el[k])
    for k in POSITIFS:
        if el[k] < 0:
            raise ValueError(f"« {k} » ne peut pas être négatif (borne de l'écran) : {el[k]!r}")
    message = hors_domaine_site(el)
    if message:
        raise ValueError(message)
    for k in ("gamma_M0", "gamma_M1"):
        if not GAMMA_MIN <= el[k] <= GAMMA_MAX:
            raise ValueError(f"« {k} » doit être entre {GAMMA_MIN} et {GAMMA_MAX} (bornes de l'écran) : {el[k]!r}")
    for k in CASES:
        if not isinstance(el[k], bool):
            raise ValueError(f"« {k} » doit être vrai ou faux : {el[k]!r}")
    courbe = {t: k for k, _, t in COURBES}.get(el["courbe"], el["courbe"])     # le libellé du module est admis
    if courbe not in ALPHA:
        raise ValueError(f"courbe de flambement inconnue : {el['courbe']!r}")
    el["courbe"] = courbe
    if el["profil"] is not None and (not isinstance(el["profil"], str) or el["profil"] not in CA._profils()):
        raise ValueError(f"profilé inconnu : {el['profil']!r}")
    return el

# -*- coding: utf-8 -*-
"""L'ENTRÉE du moteur « Maturité du béton » : un coulage (un béton, une cure, les analyses de l'écran), ses valeurs par
défaut — celles de l'écran Streamlit ``modules/age_beton.py`` au premier affichage — et sa normalisation stricte.

Les bornes sont celles des champs de l'écran (min_value / max_value) ; une valeur hors de ces bornes, une clé inconnue
ou un type faux lèvent ValueError avec le chemin du champ : le moteur ne ramène jamais une saisie en silence.
"""
import copy
import datetime as _dt
import math

from . import meteo as MT
from . import releve as RL
from .ec2 import CIMENTS, CLASSES, EA_EC2

# les classes de ciment : la clé courte du site → le libellé du module (la clé de CIMENTS)
CIMENT_LIBELLES = {"R": "Classe R — durcissement rapide", "N": "Classe N — durcissement normal", "S": "Classe S — durcissement lent"}
assert set(CIMENT_LIBELLES.values()) == set(CIMENTS)
# « Manuel » et « Météo » de l'écran : la température constante ; le relevé saisi (collé, lu dans un fichier) ; la météo du
# lieu récupérée chez Open-Meteo (le navigateur fait les appels, beton/maturite/meteo.py en lit les réponses)
MODES_TEMPERATURE = ("constante", "releve", "meteo")
MODES_RELEVE = ("releve", "meteo")                    # les modes où la cure suit un relevé jour par jour
MODES_SEUIL = ("pct", "mpa")                          # « Pourcentage de fck(28) » et « Résistance (MPa) »
POURCENTAGES = tuple(range(30, 101, 5))              # le curseur « Pourcentage » : 30 à 100 %, pas de 5
# bornes des champs de l'écran
T_MIN, T_MAX = -5.0, 60.0                             # « T moyenne (°C) », « T cure (°C) »
EA_MIN, EA_MAX = 20.0, 60.0                           # « Ea [kJ/mol] »
AGE_DEFAUT_MAX = 28                                   # le curseur « Âge (jours réels) » sans relevé : 1 à 28 j
# bornes HAUTES du site, que l'écran n'a pas (« Délai (jours) », « Âge (j) » de l'essai, résistances saisies : sans
# max_value) : au-delà, la saisie n'a pas de sens physique, et un délai de 1e308 j à 60 °C déborde en infini (le JSON
# du navigateur ne le porte pas). Aucun cas réel n'en approche : 10 ans, 200 MPa.
JOURS_MAX = 3650.0                                    # délai ③, âge de l'essai ⑤
MPA_MAX = 200.0                                       # seuil ② en MPa, cible ③, mesure ⑤


def nouvel_element(nom: str = "Coulage 1") -> dict:
    """Un coulage aux valeurs de l'écran Streamlit au premier affichage : C30/37, ciment N, T = 20 °C constante, âge
    28 j, seuil 75 %, cible de l'optimisation = f_ck(28 j) = 30,0 MPa en max(3 ; 28/2) = 14,0 j, les sept classes au
    comparateur, aucun essai (mesure 0), T de cure de l'essai = T."""
    return {"nom": nom, "classe": "C30/37", "ciment": "N", "date_coulage": None,
            "temperature": {"mode": "constante", "T": 20.0, "releve": "", "fichier": "", "lieu": "",
                            "debut": None, "fin": None, "meteo": None},
            "avance": False, "Ea_kJ": EA_EC2 / 1000.0,
            "age": 28,
            "seuil": {"mode": "pct", "pct": 75, "mpa": 20.0, "afficher": False},
            "optim": {"mpa": 30.0, "delai": 14.0, "afficher": False},
            "comparateur": {"classes": list(CLASSES), "ciment": "N", "afficher": False},
            "essai": {"mpa": 0.0, "age": 7.0, "T": 20.0, "afficher": False}}


def lire_date(x):
    """Une date de coulage : None, « AAAA-MM-JJ » ou « JJ/MM/AAAA » → « AAAA-MM-JJ » (ValueError sinon)."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    d = RL.lire_date(x.strip()) if isinstance(x, str) else None
    if d is None:
        raise ValueError(f"date illisible « {x} » (JJ/MM/AAAA)")
    return d.isoformat()


def age_max(element: dict) -> int:
    """La borne haute du curseur « Âge » : le nombre de jours du relevé (au moins 2), 28 sans relevé exploitable —
    ``int(max(age_max, 2))`` du module."""
    t = element["temperature"]
    if t["mode"] in MODES_RELEVE and t.get("releve"):
        rel = RL.lire(t["releve"])
        if rel["temps"] and RL.synthese(rel) is not None:
            return int(max(len(rel["temps"]), 2))
    return AGE_DEFAUT_MAX


def age_defaut(borne: int) -> int:
    """La position du curseur quand sa borne change (le module le recrée) : min(28, borne)."""
    return min(28, int(borne))


# ------------------------------------------------------------------ la normalisation
def _nombre(v, chemin, mini=None, maxi=None, entier=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{chemin} : un nombre est attendu (reçu {v!r})")
    if entier:
        if float(v) != int(v):
            raise ValueError(f"{chemin} : un nombre entier est attendu (reçu {v!r})")
        v = int(v)
    else:
        v = float(v)
    if mini is not None and v < mini:
        raise ValueError(f"{chemin} : {v:g} est sous le minimum {mini:g}")
    if maxi is not None and v > maxi:
        raise ValueError(f"{chemin} : {v:g} dépasse le maximum {maxi:g}")
    return v


def _booleen(v, chemin):
    if not isinstance(v, bool):
        raise ValueError(f"{chemin} : vrai ou faux attendu (reçu {v!r})")
    return v


def _texte(v, chemin, maxi=2000):
    if not isinstance(v, str):
        raise ValueError(f"{chemin} : un texte est attendu (reçu {v!r})")
    if len(v) > maxi:
        raise ValueError(f"{chemin} : texte trop long")
    return v


def _ciment(v, chemin):
    if v in CIMENT_LIBELLES:
        return v
    for k, lib in CIMENT_LIBELLES.items():                 # le libellé du module est admis
        if v == lib:
            return k
    raise ValueError(f"{chemin} : ciment inconnu {v!r} (R, N ou S)")


def _sous(d, defaut, chemin):
    """Un sous-dictionnaire : complété par les défauts, clés inconnues refusées."""
    if d is None:
        return copy.deepcopy(defaut)
    if not isinstance(d, dict):
        raise ValueError(f"{chemin} : un objet est attendu")
    inconnues = set(d) - set(defaut)
    if inconnues:
        raise ValueError(f"{chemin} : clé(s) inconnue(s) {sorted(inconnues)}")
    out = copy.deepcopy(defaut)
    out.update({k: v for k, v in d.items() if v is not None or k == "date_coulage"})
    return out


def normaliser(element) -> dict:
    """Une copie complétée et contrôlée du coulage ; ValueError (avec le chemin) hors du domaine de l'écran."""
    if not isinstance(element, dict):
        raise ValueError("un coulage est un objet")
    d0 = nouvel_element()
    e = _sous(element, d0, "coulage")
    e["nom"] = _texte(e["nom"], "nom", 200)
    if e["classe"] not in CLASSES:
        raise ValueError(f"classe : {e['classe']!r} n'est pas une classe de l'écran ({', '.join(CLASSES)})")
    e["ciment"] = _ciment(e["ciment"], "ciment")
    if e["date_coulage"] is not None and not isinstance(e["date_coulage"], str):
        raise ValueError(f"date_coulage : une date est attendue (reçu {e['date_coulage']!r})")
    e["date_coulage"] = lire_date(e["date_coulage"])
    t = e["temperature"] = _sous(e["temperature"], d0["temperature"], "temperature")
    if t["mode"] not in MODES_TEMPERATURE:
        raise ValueError(f"temperature/mode : {t['mode']!r} (constante ou releve)")
    t["T"] = _nombre(t["T"], "temperature/T", T_MIN, T_MAX)
    t["releve"] = _texte(t["releve"], "temperature/releve", RL.MAX_TEXTE)
    t["fichier"] = _texte(t["fichier"], "temperature/fichier", 300)
    t["lieu"] = _texte(t["lieu"], "temperature/lieu", 200)
    RL.lire(t["releve"])                                           # un relevé illisible lève ici, avec sa ligne
    # les dates de la récupération météo (« Coulage », « Fin (décoffrage) » de l'écran) : None = celles de l'écran au
    # premier affichage ; la récupération gardée, relue strictement (meteo.normaliser)
    for k in ("debut", "fin"):
        if t[k] is not None and not isinstance(t[k], str):
            raise ValueError(f"temperature/{k} : une date est attendue (reçu {t[k]!r})")
        t[k] = lire_date(t[k])
    t["meteo"] = MT.normaliser(t["meteo"])
    e["avance"] = _booleen(e["avance"], "avance")
    e["Ea_kJ"] = _nombre(e["Ea_kJ"], "Ea_kJ", EA_MIN, EA_MAX)
    e["age"] = _nombre(e["age"], "age", 1, age_max(e), entier=True)
    s = e["seuil"] = _sous(e["seuil"], d0["seuil"], "seuil")
    if s["mode"] not in MODES_SEUIL:
        raise ValueError(f"seuil/mode : {s['mode']!r} (pct ou mpa)")
    s["pct"] = _nombre(s["pct"], "seuil/pct", 30, 100, entier=True)
    if s["pct"] not in POURCENTAGES:
        raise ValueError(f"seuil/pct : {s['pct']} (30 à 100 %, par pas de 5)")
    # la borne haute du champ (f_ck) n'est pas contrôlée ici : l'écran garde la valeur quand la classe baisse
    s["mpa"] = _nombre(s["mpa"], "seuil/mpa", 1.0, MPA_MAX)
    s["afficher"] = _booleen(s["afficher"], "seuil/afficher")
    o = e["optim"] = _sous(e["optim"], d0["optim"], "optim")
    o["mpa"] = _nombre(o["mpa"], "optim/mpa", 1.0, MPA_MAX)
    o["delai"] = _nombre(o["delai"], "optim/delai", 0.5, JOURS_MAX)
    o["afficher"] = _booleen(o["afficher"], "optim/afficher")
    c = e["comparateur"] = _sous(e["comparateur"], d0["comparateur"], "comparateur")
    if not isinstance(c["classes"], list) or any(x not in CLASSES for x in c["classes"]) or len(set(c["classes"])) != len(c["classes"]):
        raise ValueError(f"comparateur/classes : une liste de classes distinctes de l'écran est attendue (reçu {c['classes']!r})")
    c["ciment"] = _ciment(c["ciment"], "comparateur/ciment")
    c["afficher"] = _booleen(c["afficher"], "comparateur/afficher")
    x = e["essai"] = _sous(e["essai"], d0["essai"], "essai")
    x["mpa"] = _nombre(x["mpa"], "essai/mpa", 0.0, MPA_MAX)
    x["age"] = _nombre(x["age"], "essai/age", 0.5, JOURS_MAX)
    x["T"] = _nombre(x["T"], "essai/T", T_MIN, T_MAX)
    x["afficher"] = _booleen(x["afficher"], "essai/afficher")
    return e


def date_seuil(date_coulage, t_seuil):
    """Le jour où le seuil est atteint : date de coulage + timedelta(days=t) — la fraction de jour est IGNORÉE par
    l'arithmétique des dates (module, ligne 709), le jour affiché est donc coulage + ⌊t⌋."""
    if not date_coulage or t_seuil is None:
        return None
    return (_dt.date.fromisoformat(date_coulage) + _dt.timedelta(days=float(t_seuil))).isoformat()

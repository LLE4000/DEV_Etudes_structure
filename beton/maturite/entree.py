# -*- coding: utf-8 -*-
"""L'ENTRÉE du moteur « Maturité du béton » : un COULAGE — son béton, sa fin de bétonnage, ses températures, ses
objectifs, ses hypothèses —, le coulage neuf et la normalisation STRICTE.

    {"nom": "Coulage 1",
     "classe": "C30/37",                                  C16/20 … C50/60 (lois.CLASSES)
     "durcissement": {"mode": "ciment" | "beton" | "r",   la lecture retenue : le ciment (EC2), le béton (bon de
                      "ciment": "R" | "N" | "S",          livraison, NBN EN 206) ou le rapport r connu
                      "beton": "rapide" | "moyen" | "lent",
                      "r": 0.30},                         r = f_cm,2j / f_cm,28j (0,15 à 0,70), lu en mode « r »
     "date": null | "AAAA-MM-JJ", "heure": 12,            la FIN DU BÉTONNAGE : t = 0 (heure entière, 0 à 23 ; midi)
     "temperature": {"mode": "constante" | "releve" | "meteo",
                     "T": 20.0,                           °C, −20 à 60 (mode constante ; et tant qu'un relevé manque)
                     "releve": "", "fichier": "", "lieu": "",       le relevé (releve.py), le nom du fichier lu, le lieu
                     "debut": null, "fin": null,          les dates de la météo (null : automatiques)
                     "meteo": null | {...}},              la récupération gardée (meteo.py)
     "marge": 3.0,                                        °C retirés aux jours prévus et prolongés (variante prudente), 0 à 10
     "objectifs": [{"nom": "Décoffrage",
                    "mode": "pct" | "mpa", "pct": 70.0, "mpa": 21.0,   la cible : % de f_ck (5 à 100) ou MPa (0,5 à 100)
                    "base": "fck" | "fcm",                ce à quoi elle se compare (en %, toujours « fck »)
                    "age": null | 3.0}, …],               l'âge PRÉVU de l'opération, jours (0,1 à 365) ; null : non daté
     "loi": {"mode": "nbn" | "ec2" | "perso", "Ea_kJ": 33.256},      la loi de température ; E_a de 20 à 60 kJ/mol
     "essai": {"actif": false, "age": 7.0, "mpa": 20.0,   la moyenne d'éprouvettes de chantier, à un âge réel
               "eprouvette": "cube" | "cylindre", "caler": false}}   … et « caler la courbe sur l'essai »

Une valeur hors de son domaine, une clé inconnue ou un type faux lèvent ValueError avec le chemin du champ : le moteur
ne ramène JAMAIS une saisie à une borne en silence. Une clé absente prend sa valeur par défaut.
"""
import copy
import math

from . import lois as L
from . import meteo as MT
from . import releve as RL

NOM_MAX = 80                                          # caractères, le nom d'un coulage
NOM_OBJECTIF_MAX = 40                                 # … et celui d'un objectif (il s'écrit sur la courbe)
MODES_TEMPERATURE = ("constante", "releve", "meteo")
MODES_RELEVE = ("releve", "meteo")                    # les modes où les températures suivent un relevé jour par jour
MODES_CIBLE = ("pct", "mpa")
EPROUVETTES = ("cube", "cylindre")
HEURE_DEFAUT = 12                                     # « midi » : la fin du bétonnage quand l'heure n'est pas donnée
T_MIN, T_MAX = -20.0, 60.0                            # °C, la température constante
MARGE_MIN, MARGE_MAX, MARGE_DEFAUT = 0.0, 10.0, 3.0   # °C, la marge sur les prévisions
AGE_MIN, AGE_MAX = 0.1, 365.0                         # jours, l'âge prévu d'un objectif et l'âge d'un essai
MPA_MIN, MPA_MAX = 0.5, 100.0                         # MPa, une cible
PCT_MIN, PCT_MAX, PCT_DEFAUT = 5.0, 100.0, 70.0       # % de f_ck, une cible
ESSAI_MPA_MAX = 150.0                                 # MPa, la résistance mesurée (un cube de béton C50/60 dépasse 60)
MAX_OBJECTIFS = 8
# les clés d'un coulage du module d'AVANT le 10/11/2026 (version 1 de la note : ciment, seuil ②, classe ③, comparateur ④)
ANCIENNES_CLES = ("ciment", "date_coulage", "avance", "Ea_kJ", "age", "seuil", "optim", "comparateur")


def nouvel_objectif(nom: str = "Décoffrage", fck: float = 30.0) -> dict:
    """Un objectif neuf : 70 % de f_ck, non daté. La cible en MPa (lue si l'on passe en MPa) part de la même valeur."""
    return {"nom": nom, "mode": "pct", "pct": PCT_DEFAUT, "mpa": round(PCT_DEFAUT * fck / 100.0, 1), "base": "fck", "age": None}


def nouvel_element(nom: str = "Coulage 1") -> dict:
    """Le coulage NEUF : C30/37, béton à durcissement moyen (le béton courant en Belgique, pris à la borne basse de sa
    classe), fin du bétonnage à midi sans date, 20 °C constante, loi NBN B 15-400, marge 3 °C, un objectif
    « Décoffrage » à 70 % de f_ck, non daté ; aucun essai."""
    return {"nom": nom, "classe": "C30/37",
            "durcissement": {"mode": "beton", "ciment": "N", "beton": "moyen", "r": 0.30},
            "date": None, "heure": HEURE_DEFAUT,
            "temperature": {"mode": "constante", "T": 20.0, "releve": "", "fichier": "", "lieu": "",
                            "debut": None, "fin": None, "meteo": None},
            "marge": MARGE_DEFAUT,
            "objectifs": [nouvel_objectif("Décoffrage", L.fck_classe("C30/37"))],
            "loi": {"mode": "nbn", "Ea_kJ": round(L.EA_EC2_KJ, 3)},
            "essai": {"actif": False, "age": 7.0, "mpa": 20.0, "eprouvette": "cube", "caler": False}}


def est_ancien(element) -> bool:
    """Un coulage du module d'avant le 10/11/2026 (ses clés : ciment, date_coulage, seuil, optim, comparateur…) ?"""
    return isinstance(element, dict) and any(k in element for k in ANCIENNES_CLES)


def depuis_ancien(element, defaut: str = "Coulage 1") -> dict:
    """Un coulage d'avant relu : son NOM est gardé (coupé à NOM_MAX), tout le reste est celui d'un coulage neuf — le
    module a été refait, les données sont à ressaisir (aucune conversion : les formules ont changé)."""
    nom = element.get("nom") if isinstance(element, dict) else None
    nom = nom.strip() if isinstance(nom, str) and nom.strip() else defaut
    return nouvel_element(nom[:NOM_MAX])


def lire_date(x):
    """Une date : None, « AAAA-MM-JJ » ou « JJ/MM/AAAA » → « AAAA-MM-JJ » (ValueError sinon)."""
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    d = RL.lire_date(x.strip()) if isinstance(x, str) else None
    if d is None:
        raise ValueError(f"date illisible « {str(x)[:40]} » (JJ/MM/AAAA)")
    return d.isoformat()


# ------------------------------------------------------------------ la normalisation
def _nombre(v, chemin, mini=None, maxi=None, entier=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{chemin} : un nombre est attendu (reçu {str(v)[:40]!r})")
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
        raise ValueError(f"{chemin} : vrai ou faux attendu (reçu {str(v)[:40]!r})")
    return v


def _texte(v, chemin, maxi=2000):
    if not isinstance(v, str):
        raise ValueError(f"{chemin} : un texte est attendu (reçu {str(v)[:40]!r})")
    if len(v) > maxi:
        raise ValueError(f"{chemin} : texte trop long ({len(v)} caractères, au plus {maxi})")
    return v


def _choix(v, chemin, permis):
    if not isinstance(v, str) or v not in permis:
        raise ValueError(f"{chemin} : {str(v)[:40]!r} n'est pas un choix ({', '.join(permis)})")
    return v


def _sous(d, defaut, chemin, nuls=()):
    """Un sous-dictionnaire : complété par les défauts, clés inconnues refusées ; ``nuls`` : les clés où null est une
    valeur (une date absente, un âge non daté) — ailleurs, null vaut « absent » et prend le défaut."""
    if d is None:
        return copy.deepcopy(defaut)
    if not isinstance(d, dict):
        raise ValueError(f"{chemin} : un objet est attendu")
    inconnues = set(d) - set(defaut)
    if inconnues:
        raise ValueError(f"{chemin} : clé(s) inconnue(s) {sorted(str(k)[:30] for k in inconnues)}")
    out = copy.deepcopy(defaut)
    out.update({k: v for k, v in d.items() if v is not None or k in nuls})
    return out


def _date(v, chemin):
    if v is not None and not isinstance(v, str):
        raise ValueError(f"{chemin} : une date est attendue (reçu {str(v)[:40]!r})")
    try:
        return lire_date(v)
    except ValueError as e:
        raise ValueError(f"{chemin} : {e}") from None


def normaliser_objectif(o, chemin: str, fck: float) -> dict:
    """Un objectif relu : complété, contrôlé ; en %, sa base est FORCÉE à « fck » (un pourcentage de f_ck se compare à
    la résistance caractéristique)."""
    o = _sous(o, nouvel_objectif("", fck), chemin, nuls=("age",))
    o["nom"] = _texte(o["nom"], f"{chemin}/nom", NOM_OBJECTIF_MAX).strip()
    o["mode"] = _choix(o["mode"], f"{chemin}/mode", MODES_CIBLE)
    o["pct"] = _nombre(o["pct"], f"{chemin}/pct", PCT_MIN, PCT_MAX)
    o["mpa"] = _nombre(o["mpa"], f"{chemin}/mpa", MPA_MIN, MPA_MAX)
    o["base"] = _choix(o["base"], f"{chemin}/base", L.BASES)
    if o["mode"] == "pct":
        o["base"] = "fck"
    if o["age"] is not None:
        o["age"] = _nombre(o["age"], f"{chemin}/age", AGE_MIN, AGE_MAX)
    return o


def normaliser(element) -> dict:
    """Une copie complétée et contrôlée du coulage ; ValueError (avec le chemin du champ) hors domaine."""
    if not isinstance(element, dict):
        raise ValueError("un coulage est un objet")
    d0 = nouvel_element()
    e = _sous(element, d0, "coulage", nuls=("date",))
    e["nom"] = _texte(e["nom"], "nom", NOM_MAX)
    e["classe"] = _choix(e["classe"], "classe", tuple(L.CLASSES))
    fck = L.fck_classe(e["classe"])
    du = e["durcissement"] = _sous(e["durcissement"], d0["durcissement"], "durcissement")
    du["mode"] = _choix(du["mode"], "durcissement/mode", L.MODES_DURCISSEMENT)
    du["ciment"] = _choix(du["ciment"], "durcissement/ciment", tuple(L.CIMENTS))
    du["beton"] = _choix(du["beton"], "durcissement/beton", tuple(L.BETONS))
    du["r"] = _nombre(du["r"], "durcissement/r", L.R_MIN, L.R_MAX)
    e["date"] = _date(e["date"], "date")
    e["heure"] = _nombre(e["heure"], "heure", 0, 23, entier=True)
    t = e["temperature"] = _sous(e["temperature"], d0["temperature"], "temperature", nuls=("debut", "fin", "meteo"))
    t["mode"] = _choix(t["mode"], "temperature/mode", MODES_TEMPERATURE)
    t["T"] = _nombre(t["T"], "temperature/T", T_MIN, T_MAX)
    t["releve"] = _texte(t["releve"], "temperature/releve", RL.MAX_TEXTE)
    t["fichier"] = _texte(t["fichier"], "temperature/fichier", 300)
    t["lieu"] = _texte(t["lieu"], "temperature/lieu", 200)
    try:
        RL.lire(t["releve"])                                       # un relevé illisible lève ici, avec sa ligne
    except ValueError as x:
        raise ValueError(f"temperature/releve : {x}") from None
    for k in ("debut", "fin"):
        t[k] = _date(t[k], f"temperature/{k}")
    t["meteo"] = MT.normaliser(t["meteo"])
    e["marge"] = _nombre(e["marge"], "marge", MARGE_MIN, MARGE_MAX)
    objs = e["objectifs"]
    if not isinstance(objs, list) or not objs:
        raise ValueError("objectifs : au moins un objectif est attendu")
    if len(objs) > MAX_OBJECTIFS:
        raise ValueError(f"objectifs : au plus {MAX_OBJECTIFS} objectifs ({len(objs)} reçus)")
    e["objectifs"] = [normaliser_objectif(o, f"objectifs/{i}", fck) for i, o in enumerate(objs)]
    lo = e["loi"] = _sous(e["loi"], d0["loi"], "loi")
    lo["mode"] = _choix(lo["mode"], "loi/mode", tuple(L.LOIS))
    lo["Ea_kJ"] = _nombre(lo["Ea_kJ"], "loi/Ea_kJ", L.EA_MIN, L.EA_MAX)
    x = e["essai"] = _sous(e["essai"], d0["essai"], "essai")
    x["actif"] = _booleen(x["actif"], "essai/actif")
    x["age"] = _nombre(x["age"], "essai/age", AGE_MIN, AGE_MAX)
    x["mpa"] = _nombre(x["mpa"], "essai/mpa", MPA_MIN, ESSAI_MPA_MAX)
    x["eprouvette"] = _choix(x["eprouvette"], "essai/eprouvette", EPROUVETTES)
    x["caler"] = _booleen(x["caler"], "essai/caler")
    return e

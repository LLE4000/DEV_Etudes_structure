# -*- coding: utf-8 -*-
"""Le module « Maturité du béton » du site, refait le 10/11/2026 sur la norme (docs/site/LOT_Q_QUATRE_MODULES.md § 2 ;
le moteur : ``beton/maturite``, docs/beton/MATURITE_MOTEUR.md), sur l'ÉTABLI COMPACT — le format de la poutre béton
armé, des cornières et des poutres bois (api_web/etabli_compact.py, js/compact.js) :

- à gauche, la ligne du coulage ouvert (son nom, la liste des coulages de la note, copier, supprimer) puis ses quatre
  rubriques : 1 Béton (la classe ; le DURCISSEMENT, un menu en deux groupes — le bon de livraison, rapide, moyen, lent
  ou r connu, et le ciment de l'EC2, R, N ou S — et dessous le r qu'il suppose ; la fin du bétonnage, date et heure),
  2 Températures (constante, relevé collé ou lu, météo du lieu récupérée chez Open-Meteo à la demande), 3 Objectifs
  (une carte par objectif : son nom, sa cible en MPa ou en % de f_ck, la résistance à laquelle elle se compare, l'âge
  prévu facultatif ; « + Objectif » et les six modèles), 4 Hypothèses (secondaire : la loi de température, la marge
  sur les prévisions, l'essai sur éprouvettes et son calage) ;
- au centre, trois VUES sous leur en-tête : la courbe des résistances, les températures jour par jour
  (api_web/dessin_age.py), le tableau jour par jour ;
- à droite, la tête (le nom ; « C30/37 · durcissement moyen · 20 °C »), le VERDICT (le premier objectif : son âge et sa
  date, sa variante prudente ; son taux s'il est daté), les blocs THÉMATIQUES « Objectifs », « Températures » et
  « Essai », les avertissements du moteur, la méthode, ses sources et ce qui n'est pas couvert ;
- les FENÊTRES DE CALCUL : chaque ligne « symbole = expression = valeurs = résultat » est le gabarit du MOTEUR
  (``moteur.EXPRESSIONS``) où chaque terme est remplacé par sa valeur (le test la réévalue) ;
- le SÉLECTEUR « classe » : chaque classe avec le même durcissement et les mêmes températures face au premier
  objectif — ✓ la classe du coulage, ★ la plus petite qui tient l'âge prévu.

Ce fichier MET EN FORME : il ne calcule rien et n'écrit aucune formule ; il CHOISIT parmi ce que le moteur a nommé et
COMPOSE les phrases. Toute saisie passe par ``entree.normaliser`` : hors domaine, elle est REFUSÉE avec le message du
moteur (le champ nommé) et le coulage reste intact. La météo est une IMPORTATION de données à la demande : le moteur
fait le plan des appels (beton/maturite/meteo.py), le bouton le porte (C.bouton_meteo), le navigateur fait les appels
et rend les réponses brutes, que ce module lit et GARDE dans la note ; le calcul se fait hors ligne. Une note du module
d'avant (version 1) se rouvre sur des coulages NEUFS qui gardent leurs noms, et l'AVIS le dit une fois (CLE_AVIS).
"""
import copy
import datetime as _dt
import html
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from beton.maturite import entree as EN
from beton.maturite import lois as L
from beton.maturite import meteo as MT
from beton.maturite import moteur as M
from beton.maturite import releve as RL

from . import commun as C
from . import dessin_age as DA
from . import etabli_compact as E

MODULE = "age"
PAGE = "maturite-beton"
VERSION = 2                     # l'état de la note : {"version": 2, "coulages": [...], "actif": 0} (1 : le module d'avant)
GENRE = E.Genre("Coulages", "le coulage", "ce coulage", "un coulage")
BROUILLON = "saisie_refusee"    # le site : un relevé refusé, gardé à l'écran avec son erreur (pas un champ du moteur)
# L'AVIS dit UNE fois à l'ouverture : ``normaliser`` (une note d'avant) le pose sous CLE_AVIS ; ``resultat`` le montre
# en tête du résultat et le RETIRE de l'état.
CLE_AVIS = "avis_ouverture"
AVIS_V1 = ("Note d'avant le 10/11/2026 : le module « Maturité du béton » a été refait sur la norme (objectifs datés, heure du "
           "bétonnage, loi de température belge) ; les données sont à ressaisir (les noms des coulages sont gardés).")

SOURCES_T = (("constante", "Constante"), ("releve", "Relevé"), ("meteo", "Météo du lieu"))
SEGMENTS_LOI = (("nbn", "NBN B 15-400"), ("ec2", "EC2 (B.10)"), ("perso", "Eₐ saisie"))
SEGMENTS_CIBLE = (("mpa", "MPa"), ("pct", "% de fck"))
SEGMENTS_BASE = (("fcm", "Moyenne"), ("fck", "Caractéristique"))
SEGMENTS_EPROUVETTE = (("cube", "Cube 150"), ("cylindre", "Cylindre"))
MOTS_SOURCE = {"mesure": "mesuré", "prevision": "prévision", "prolonge": "prolongé", "constante": "constante", "sans_donnee": "sans donnée"}
JOURS_SEMAINE = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")
COURT_BETON = {"rapide": "rapide", "moyen": "moyen", "lent": "lent"}
# les noms des champs, pour les messages (le moteur nomme le chemin : « temperature/T : -25 est sous le minimum -20 »)
NOMS_CHAMPS = {"nom": ("Nom", ""), "classe": ("Classe", ""), "durcissement/mode": ("Durcissement", ""), "durcissement/r": ("Rapport r", ""),
               "date": ("Fin du bétonnage", ""), "heure": ("Heure de fin du bétonnage", "h"), "temperature/mode": ("Températures", ""),
               "temperature/T": ("Température", "°C"), "temperature/lieu": ("Lieu", ""), "temperature/releve": ("Relevé", ""),
               "temperature/debut": ("Début de la météo", ""), "temperature/fin": ("Fin de la météo", ""), "marge": ("Marge sur les prévisions", "°C"),
               "loi/mode": ("Loi de température", ""), "loi/Ea_kJ": ("Énergie d'activation", "kJ/mol"), "essai/age": ("Âge de l'essai", "j"),
               "essai/mpa": ("Résistance mesurée", "MPa"), "essai/eprouvette": ("Éprouvette", ""), "objectifs": ("Objectifs", "")}
NOMS_OBJECTIF = {"nom": ("nom", ""), "mode": ("cible", ""), "pct": ("cible", "% de f_ck"), "mpa": ("cible", "MPa"), "base": ("comparée à", ""),
                 "age": ("âge prévu", "j")}
_CHEMIN_OBJECTIF = re.compile(r"^objectifs/(\d+)/(\w+)$")
NOMBRES = ("temperature/T", "marge", "loi/Ea_kJ", "essai/age", "essai/mpa", "durcissement/r")
CASES = ("essai/actif", "essai/caler")
CHOIX = ("classe", "temperature/mode", "loi/mode", "essai/eprouvette")

# un relevé d'exemple (le parcours de l'écran ; les notes d'exemple du gel) : six jours d'automne à Namur, une ligne sans
# donnée, deux jours de prévision et deux prolongés
EXEMPLE_RELEVE = ("Date;T moy (°C);Source\n2026-11-02;9,5;mesuré\n2026-11-03;8,0;mesuré\n2026-11-04;—;mesuré\n2026-11-05;6,5;mesuré\n"
                  "2026-11-06;7,0;prévision\n2026-11-07;8,5;prévision\n2026-11-08;8,0;prolongé\n2026-11-09;8,0;prolongé")


def _etat_releve(**k) -> Dict:
    """La note d'exemple « relevé collé » : un voile coulé le 02/11/2026 à 16 h, le relevé de Namur ; ``k`` :
    « essai__actif » écrit dans le groupe « essai »…"""
    e = EN.nouvel_element("Voile R+1")
    e["temperature"].update(mode="releve", releve=EXEMPLE_RELEVE, lieu="Namur")
    e.update(date="2026-11-02", heure=16)
    for cle, v in k.items():
        a, _, b = cle.partition("__")
        if b:
            e[a][b] = v
        else:
            e[a] = v
    return {"version": VERSION, "coulages": [EN.normaliser(e)], "actif": 0}


def aujourdhui() -> _dt.date:
    """Le jour où l'écran est rendu (celui du navigateur, dans Pyodide) : il partage les dates de la météo entre
    l'archive et les prévisions (le plan du bouton). Le gel le fige (tests/test_reference_refonte.py)."""
    return _dt.date.today()


def maintenant() -> str:
    """Le jour et l'heure d'une récupération de la météo (AAAA-MM-JJTHH:MM), écrits dans la note. Le gel le fige."""
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M")


# une récupération d'exemple (le parcours de l'écran, aucun appel) : Bruxelles, du 27/10 au 25/11/2026, récupérée le
# 07/11/2026 — l'archive jusqu'au 31/10, les prévisions du 01/11 au 23/11, deux jours prolongés. Les réponses ont le format
# d'Open-Meteo ; les températures sont inventées.
EXEMPLE_METEO = {"lieu": MT.LIEU_DEFAUT, "debut": "2026-10-27", "fin": "2026-11-25", "aujourdhui": "2026-11-07"}
_GEO_EXEMPLE = {"results": [{"id": 2800866, "name": "Bruxelles", "latitude": 50.85045, "longitude": 4.34878, "country_code": "BE",
                             "admin1": "Bruxelles-Capitale", "country": "Belgique", "timezone": "Europe/Brussels"}], "generationtime_ms": 0.5}


def _marque(p: Dict, coulage: Optional[int]) -> Dict:
    """Le plan porte le COULAGE dont le bouton vient (son rang dans la note) : une récupération qui revient alors qu'un
    autre coulage est ouvert est refusée — elle allait dans le coulage OUVERT, celui qu'on regardait au retour : une copie
    (même lieu, mêmes dates) perdait ses corrections à la main, et celui du clic restait sans météo."""
    return p if "erreur" in p or coulage is None else dict(p, coulage=coulage)


def _meteo_exemple(trouve: bool = True, horaire: bool = False) -> Dict:
    """Ce que le navigateur rendrait pour la récupération d'exemple : {"lieu", "plan", "geocodage", "reponses"}."""
    x = EXEMPLE_METEO
    pos = {"latitude": 50.85045, "longitude": 4.34878}
    p = _marque(MT.plan(x["lieu"], x["debut"], x["fin"], x["aujourdhui"], horaire=horaire, position=pos if horaire else None), 0)
    reps = []
    for a in p["appels"]:
        d0 = _dt.date.fromisoformat(a["debut"])
        jours = [d0 + _dt.timedelta(days=i) for i in range((_dt.date.fromisoformat(a["fin"]) - d0).days + 1)]
        if horaire:
            reps.append({"hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
                         "hourly": {"time": [f"{d.isoformat()}T{h:02d}:00" for d in jours for h in range(24)],
                                    "temperature_2m": [round(6 + d.toordinal() % 5 + 3 * math.sin((h - 9) / 24 * 2 * math.pi), 1)
                                                       for d in jours for h in range(24)]}})
        else:
            reps.append({"daily_units": {"time": "iso8601", "temperature_2m_mean": "°C"},
                         "daily": {"time": [d.isoformat() for d in jours], "temperature_2m_mean": [round(6 + d.toordinal() % 5 + 0.3, 1) for d in jours]}})
    return {"lieu": None if horaire else x["lieu"], "plan": p, "geocodage": None if horaire else (_GEO_EXEMPLE if trouve else {"generationtime_ms": 0.4}),
            "reponses": reps if trouve else []}


def _etat_meteo(importee: bool = False) -> Dict:
    """La note d'exemple en mode météo ; ``importee`` : la récupération d'exemple déjà faite et gardée (bétonnage le
    27/10/2026 à 15 h, décoffrage prévu à J + 3)."""
    e = EN.nouvel_element("Dalle R+2")
    x = EXEMPLE_METEO
    e["temperature"].update(mode="meteo", lieu=x["lieu"], debut=x["debut"], fin=x["fin"])
    e.update(date=x["debut"], heure=15)
    if importee:
        v = _meteo_exemple()
        e["temperature"]["meteo"] = MT.enregistrement(v["plan"], v["geocodage"], v["reponses"], "2026-11-07T08:30")
        e["temperature"]["releve"] = MT.releve(MT.lire_journalier(v["plan"], v["reponses"]))
        e["objectifs"][0]["age"] = 3.0
    return {"version": VERSION, "coulages": [EN.normaliser(e)], "actif": 0}


# une note du module d'avant (version 1), telle qu'un fichier .stab d'avant le 10/11/2026 la porte : le parcours la rejoue
# (la relecture, l'avis), tests/test_api_web_age.py la contrôle
NOTE_V1 = {"version": 1, "actif": 0,
           "coulages": [{"nom": "Voile R+1", "classe": "C30/37", "ciment": "N", "date_coulage": "2026-11-02", "age": 7,
                         "temperature": {"mode": "constante", "T": 12.0}, "seuil": {"mode": "pct", "pct": 70, "afficher": True}},
                        {"nom": "Dalle R+2", "classe": "C25/30", "ciment": "R", "age": 3}]}


# les requêtes de plus du parcours de l'écran (api_web/parcours.py) : le relevé collé ou lu, un relevé refusé, la date,
# un objectif ajouté (neuf, un modèle) puis retiré, un âge prévu, un essai calé, le sélecteur de classe, la note d'avant ;
# la météo du lieu : récupérée, lieu introuvable, le détail horaire, une réponse qui ne suit pas le plan
PARCOURS = [
    {"action": "modifier", "chemin": ["temperature", "mode"], "valeur": "releve", "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["temperature", "releve"], "valeur": EXEMPLE_RELEVE.replace("9,5", "10,5"), "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["temperature", "releve"], "valeur": "12\nabc\n13", "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["temperature", "fichier"], "valeur": {"nom": "releve.csv", "texte": EXEMPLE_RELEVE}},
    {"action": "modifier", "chemin": ["temperature", "fichier"], "valeur": {"nom": "faux.csv", "texte": "douze"}},
    {"action": "modifier", "chemin": ["temperature", "releve"], "valeur": "", "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["date"], "valeur": "02/11/2026"},
    {"action": "modifier", "chemin": ["heure"], "valeur": "7"},
    {"action": "modifier", "chemin": ["temperature", "T"], "valeur": "-5"},
    {"action": "modifier", "chemin": ["objectifs", "+"], "valeur": "nouveau"},
    {"action": "modifier", "chemin": ["objectifs", "+"], "valeur": "hors_gel"},
    {"action": "modifier", "chemin": ["objectifs", "0", "age"], "valeur": "3"},
    {"action": "modifier", "chemin": ["objectifs", "0", "mode"], "valeur": "mpa"},
    {"action": "modifier", "chemin": ["objectifs", "0", "-"], "valeur": "1", "etat": _etat_releve(objectifs=[EN.nouvel_objectif("A"), EN.nouvel_objectif("B")])},
    {"action": "modifier", "chemin": ["durcissement", "choix"], "valeur": "r"},
    {"action": "modifier", "chemin": ["essai", "mpa"], "valeur": "21,6", "etat": _etat_releve(essai__actif=True)},
    {"action": "modifier", "chemin": ["essai", "caler"], "valeur": True, "etat": _etat_releve(essai__actif=True)},
    {"action": "modifier", "chemin": ["loi", "Ea_kJ"], "valeur": "45", "etat": _etat_releve(loi={"mode": "perso", "Ea_kJ": 40.0})},
    {"action": "choisir", "role": "classe", "nom": "C35/45"},
    {"action": "calculer", "etat": NOTE_V1},
    {"action": "modifier", "chemin": ["temperature", "mode"], "valeur": "meteo"},
    {"action": "modifier", "chemin": ["temperature", "meteo"], "valeur": _meteo_exemple(), "etat": _etat_meteo()},
    {"action": "modifier", "chemin": ["temperature", "meteo"], "valeur": _meteo_exemple(trouve=False), "etat": _etat_meteo()},
    {"action": "modifier", "chemin": ["temperature", "meteo"], "valeur": dict(_meteo_exemple(), reponses=[]), "etat": _etat_meteo()},
    {"action": "modifier", "chemin": ["temperature", "meteo_horaire"], "valeur": _meteo_exemple(horaire=True), "etat": _etat_meteo(True)},
    {"action": "modifier", "chemin": ["temperature", "releve"], "valeur": "8\n9\n10", "etat": _etat_meteo(True)},
    {"action": "modifier", "chemin": ["temperature", "fin"], "valeur": "30/12/2027", "etat": _etat_meteo(True)},
]


# ------------------------------------------------------------------ l'état de la note : des coulages
def nouvel_etat() -> Dict:
    """Une note neuve : un coulage aux défauts de la conception § 2.5 (C30/37, béton à durcissement moyen, 20 °C
    constante, objectif « Décoffrage » à 70 % de f_ck, non daté)."""
    return {"version": VERSION, "coulages": [EN.nouvel_element("Coulage 1")], "actif": 0}


def _indice(x) -> int:
    """Un rang lu sans faire tomber la note : un nombre entier fini, sinon 0."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        try:
            return int(str(x))
        except (TypeError, ValueError):
            return 0
    return int(x)


def _moteur(c: Dict) -> Dict:
    """Le coulage tel que le moteur le lit (sans ce que le site garde à côté)."""
    return {k: v for k, v in c.items() if k != BROUILLON}


def _relire(c, i: int) -> Dict:
    if not isinstance(c, dict):
        raise ValueError(f"coulage n° {i + 1} illisible : un groupe de champs est attendu")
    try:
        e = EN.normaliser(_moteur(c))
    except ValueError as x:
        raise ValueError(f"coulage « {str(c.get('nom', i + 1))[:80]} » : {x}") from None
    br = c.get(BROUILLON)
    if isinstance(br, dict) and isinstance(br.get("texte"), str) and isinstance(br.get("erreur"), str) and len(br["texte"]) <= RL.MAX_TEXTE:
        e[BROUILLON] = {"texte": br["texte"], "erreur": br["erreur"][:400]}
    return e


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; un coulage illisible lève une erreur claire.
    Une note sans coulage est une note neuve ; une note d'une version antérieure (ou dont un coulage porte les clés du
    module d'avant) est une note du module d'AVANT : ses coulages sont NEUFS et gardent leurs noms, l'avis le dit ; une
    note d'une version plus récente est refusée (rien n'est deviné)."""
    coulages = etat.get("coulages") if isinstance(etat, dict) else None
    if not coulages:
        return nouvel_etat()
    if not isinstance(coulages, list):
        raise ValueError("note abîmée : « coulages » doit être une liste")
    version = etat.get("version")
    actif = _indice(etat.get("actif", 0))
    if isinstance(version, (int, float)) and not isinstance(version, bool) and version > VERSION:
        raise ValueError(f"note enregistrée par une version plus récente du module (version {version}) : elle ne se relit pas ici")
    if version != VERSION or any(EN.est_ancien(c) for c in coulages):
        out = {"version": VERSION, "coulages": [EN.depuis_ancien(c, f"Coulage {i + 1}") for i, c in enumerate(coulages)]}
        out["actif"] = min(max(0, actif), len(out["coulages"]) - 1)
        out[CLE_AVIS] = "v1"
        return out
    out = {"version": VERSION, "coulages": [_relire(c, i) for i, c in enumerate(coulages)]}
    out["actif"] = min(max(0, actif), len(out["coulages"]) - 1)
    return out


def _coulage(etat: Dict) -> Dict:
    return etat["coulages"][etat["actif"]]


def _nom_libre(cs: List[Dict]) -> str:
    pris = {c.get("nom") for c in cs}
    k = len(cs) + 1
    while f"Coulage {k}" in pris:
        k += 1
    return f"Coulage {k}"


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les coulages de la note : en ouvrir un, en ajouter, en copier un (``index`` ; sans : l'ouvert — la copie s'insère
    juste après), en supprimer un (jamais le dernier)."""
    cs = etat["coulages"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(cs) - 1)
    elif action == "ajouter":
        cs.append(EN.nouvel_element(_nom_libre(cs)))
        etat["actif"] = len(cs) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, _indice(index)), len(cs) - 1)
        c = copy.deepcopy(cs[i])
        c["nom"] = f"{c.get('nom') or 'Coulage'} (copie)"[:EN.NOM_MAX]
        cs.insert(i + 1, c)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(cs) > 1:
        i = min(max(0, _indice(index)), len(cs) - 1)
        del cs[i]
        if etat["actif"] > i or etat["actif"] >= len(cs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _moins(s: str) -> str:
    return s.replace("-", "−")


def _message(e: Exception) -> str:
    """Le message du moteur, le chemin du champ remplacé par son nom : « temperature/T : -25 est sous le minimum -20 » →
    « Température : −25 est sous le minimum −20 (°C) » ; « objectifs/1/mpa : … » → « Objectif 2, cible : … (MPa) »."""
    s = str(e)
    tete, sep, reste = s.partition(" : ")
    if sep:
        m = _CHEMIN_OBJECTIF.match(tete)
        if m and m.group(2) in NOMS_OBJECTIF:
            nom, u = NOMS_OBJECTIF[m.group(2)]
            return f"Objectif {int(m.group(1)) + 1}, {nom} : {_moins(reste)}" + (f" ({u})" if u else "")
        if tete in NOMS_CHAMPS:
            nom, u = NOMS_CHAMPS[tete]
            return f"{nom} : {_moins(reste)}" + (f" ({u})" if u else "")
    return s[:1].upper() + s[1:]


def _releve(c: Dict, texte: str, nom: Optional[str] = None) -> Optional[str]:
    """Un relevé collé ou lu : lu par le moteur ; refusé, il reste à l'écran avec son erreur (le calcul garde le relevé
    précédent)."""
    if len(texte) > RL.MAX_TEXTE:
        return f"Relevé refusé : plus de {RL.MAX_TEXTE // 1000} ko"
    try:
        RL.lire(texte)
    except ValueError as e:
        c[BROUILLON] = {"texte": texte, "erreur": f"{'Fichier « ' + nom + ' » : ' if nom else ''}{e}"}
        return f"Relevé refusé — {e}. Le calcul garde le relevé précédent ; corrigez la ligne indiquée."
    c["temperature"]["releve"] = texte
    c.pop(BROUILLON, None)
    if nom is not None:
        c["temperature"]["fichier"] = nom[:300]
        c["temperature"]["mode"] = "releve"
    return None


def _dates_meteo(c: Dict, auj: _dt.date):
    """Les dates de la récupération : celles saisies, sinon celles par défaut (le début = le jour de la fin du
    bétonnage, sinon aujourd'hui − 14 j ; la fin = max(aujourd'hui, début + 5 j))."""
    t = c["temperature"]
    debut = _dt.date.fromisoformat(t["debut"]) if t["debut"] else MT.dates_par_defaut(c["date"], auj)[0]
    fin = _dt.date.fromisoformat(t["fin"]) if t["fin"] else MT.dates_par_defaut(debut.isoformat(), auj)[1]
    return debut, fin


def plan_meteo(c: Dict, auj: Optional[_dt.date] = None, coulage: Optional[int] = None) -> Dict:
    """Le plan de la récupération du coulage (le bouton « Récupérer la météo ») ; {"erreur": message} si elle est
    impossible (pas de lieu, fin avant le début, dates hors de portée). ``coulage`` : son rang dans la note (_marque)."""
    auj = auj or aujourdhui()
    debut, fin = _dates_meteo(c, auj)
    try:
        return _marque(MT.plan(c["temperature"]["lieu"], debut, fin, auj), coulage)
    except ValueError as e:
        return {"erreur": str(e)}


def plan_horaire(c: Dict, auj: Optional[_dt.date] = None, coulage: Optional[int] = None) -> Dict:
    """Le plan du détail horaire (le second bouton) : le lieu trouvé, les dates de la récupération gardée."""
    try:
        return _marque(MT.plan_horaire(c["temperature"]["meteo"], auj or aujourdhui()), coulage)
    except ValueError as e:
        return {"erreur": str(e)}


def _plan_recu(valeur) -> Optional[_dt.date]:
    """Le jour du plan que le navigateur a exécuté (il le renvoie avec les réponses) ; None s'il est illisible."""
    try:
        return _dt.date.fromisoformat(valeur["plan"]["aujourdhui"])
    except (TypeError, KeyError, ValueError):
        return None


MODIFIEE = "Lieu ou dates modifiés pendant la récupération : cliquez à nouveau sur « Récupérer la météo »."
AUTRE_COULAGE = ("Un autre coulage a été ouvert pendant la récupération : rien n'est importé. Revenez au coulage voulu et cliquez à "
                 "nouveau sur « Récupérer la météo ».")


def _autre_coulage(valeur, coulage: int) -> bool:
    """La récupération vient-elle d'un autre coulage que celui qui est ouvert (le plan porte son rang) ?"""
    return isinstance(valeur.get("plan"), dict) and valeur["plan"].get("coulage") != coulage


def _importer_meteo(c: Dict, valeur, coulage: int) -> Optional[str]:
    """La météo que le navigateur a récupérée ({"lieu", "plan", "geocodage", "reponses"}, BRUTS) : relue par le moteur
    (beton/maturite/meteo.py) contre le plan du coulage, écrite en relevé (modifiable ensuite), et GARDÉE dans la note
    (la récupération : lieu, dates, jour et heure, réponses). Un refus rend un message ; le coulage reste intact."""
    auj = _plan_recu(valeur) if isinstance(valeur, dict) else None
    if auj is None:
        return "Météo illisible : le navigateur n'a pas rendu le plan de la récupération. Rechargez la page et recommencez."
    if _autre_coulage(valeur, coulage):
        return AUTRE_COULAGE
    t = c["temperature"]
    p = plan_meteo(c, auj, coulage)
    if "erreur" in p:
        return f"Météo non récupérée — {p['erreur']}"
    if valeur.get("plan") != p:
        return MODIFIEE
    try:
        geo = MT.lire_geocodage(valeur.get("geocodage"))
        if geo is None:
            return f"Lieu introuvable : « {t['lieu']} ». Précisez la ville et le pays (ex. « Namur, Belgique »)."
        j = MT.lire_journalier(p, valeur.get("reponses"))
        texte = MT.releve(j)
        rec = MT.enregistrement(p, valeur["geocodage"], valeur["reponses"], maintenant())
    except ValueError as e:
        return f"Météo refusée — {e}."
    t.update(releve=texte, meteo=rec, debut=p["debut"], fin=p["fin"], mode="meteo")
    c.pop(BROUILLON, None)
    return None


def _importer_horaire(c: Dict, valeur, coulage: int) -> Optional[str]:
    """Le détail horaire récupéré (à la demande) : relu contre le plan horaire de la récupération gardée, et gardé."""
    rec = c["temperature"]["meteo"]
    if not rec:
        return "Récupérez d'abord la météo du lieu : le détail horaire la complète."
    auj = _plan_recu(valeur) if isinstance(valeur, dict) else None
    if auj is None:
        return "Détail horaire illisible : le navigateur n'a pas rendu le plan de la récupération."
    if _autre_coulage(valeur, coulage):
        return AUTRE_COULAGE
    p = plan_horaire(c, auj, coulage)
    if "erreur" in p:
        return f"Détail horaire non récupéré — {p['erreur']}"
    if valeur.get("plan") != p:
        return MODIFIEE
    try:
        MT.lire_horaire(p, valeur.get("reponses"))
        rec["horaire"] = MT.enregistrement_horaire(p, valeur["reponses"], maintenant())
        c["temperature"]["meteo"] = MT.normaliser(rec)
    except ValueError as e:
        return f"Détail horaire refusé — {e}."
    return None


def _nombre(valeur, quoi: str) -> float:
    x = C.lire_nombre(valeur)
    if x is None:
        raise ValueError(f"{quoi} : nombre illisible « {str(valeur)[:40]} »")
    return x


def _case(valeur) -> bool:
    if isinstance(valeur, bool):
        return valeur
    if str(valeur).lower() in ("true", "1", "oui"):
        return True
    if str(valeur).lower() in ("false", "0", "non", ""):
        return False
    raise ValueError(f"case à cocher : vrai ou faux est attendu (reçu « {str(valeur)[:40]} »)")


def _poser(c: Dict, ch: List[str], v) -> None:
    d = c
    for x in ch[:-1]:
        d = d[x]
    if ch[-1] not in d:
        raise KeyError(ch[-1])
    d[ch[-1]] = v


def _rang_objectif(c: Dict, x: str) -> int:
    if not x.isdigit() or int(x) >= len(c["objectifs"]):
        raise ValueError(f"objectifs : objectif inconnu « {x[:20]} »")
    return int(x)


def _objectif(c: Dict, ch: List[str], valeur) -> Optional[str]:
    """Les saisies d'un objectif : l'ajouter (« objectifs/+ » : « nouveau » ou un modèle), le retirer (« objectifs/i/- »,
    jamais le dernier), son nom, sa cible (le mode, la valeur), la résistance à laquelle elle se compare, son âge prévu
    (vide : non daté). D'un mode de cible à l'autre, la cible garde sa valeur en MPa (70 % de f_ck du C30/37 → 21 MPa)."""
    fck = L.fck_classe(c["classe"])
    objs = c["objectifs"]
    if ch == ["objectifs", "+"]:
        if len(objs) >= EN.MAX_OBJECTIFS:
            return f"Objectifs : au plus {EN.MAX_OBJECTIFS} objectifs par coulage"
        cle = str(valeur or "nouveau")
        objs.append(EN.nouvel_objectif(f"Objectif {len(objs) + 1}", fck) if cle == "nouveau" else M.modele_objectif(cle, fck))
        return None
    if len(ch) != 3:
        raise ValueError(f"objectifs : chemin inconnu « {'/'.join(ch)[:40]} »")
    i = _rang_objectif(c, ch[1])
    o, champ = objs[i], ch[2]
    if champ == "-":
        if len(objs) == 1:
            return "Objectifs : un coulage garde au moins un objectif"
        del objs[i]
    elif champ == "nom":
        nom = str(valeur if valeur is not None else "").strip()
        if len(nom) > EN.NOM_OBJECTIF_MAX:
            return f"Objectif {i + 1}, nom : trop long ({len(nom)} caractères, au plus {EN.NOM_OBJECTIF_MAX})"
        o["nom"] = nom
    elif champ == "mode":
        v = str(valeur)
        if v == "mpa" and o["mode"] == "pct":
            o["mpa"] = round(o["pct"] * fck / 100.0, 1)
        elif v == "pct" and o["mode"] == "mpa":
            p = round(o["mpa"] / fck * 100.0, 1)
            if EN.PCT_MIN <= p <= EN.PCT_MAX:
                o["pct"] = p
        o["mode"] = v
    elif champ in ("pct", "mpa"):
        o[champ] = _nombre(valeur, f"objectifs/{i}/{champ}")
    elif champ == "base":
        o["base"] = str(valeur)
    elif champ == "age":
        try:
            o["age"] = C.lire_auto(valeur)
        except ValueError as e:
            return f"Objectif {i + 1}, âge prévu : {e}"
    else:
        raise ValueError(f"objectifs : champ inconnu « {champ[:20]} »")
    return None


def _appliquer(etat: Dict, c: Dict, ch: List[str], valeur) -> Optional[str]:
    """Écrit la saisie ``ch`` dans le coulage ``c`` (une copie : l'appelant la fait relire au moteur) ; un message
    quand elle est refusée avant le moteur."""
    k = "/".join(ch)
    if k in ("temperature/meteo", "temperature/meteo_horaire"):
        return (_importer_meteo if k == "temperature/meteo" else _importer_horaire)(c, valeur, etat["actif"])
    if k == "temperature/releve":
        return _releve(c, "" if valeur is None else str(valeur))
    if k == "temperature/fichier":
        if not isinstance(valeur, dict) or not isinstance(valeur.get("texte"), str):
            return "Fichier illisible"
        return _releve(c, valeur["texte"], str(valeur.get("nom") or "fichier"))
    if ch[0] == "objectifs":
        return _objectif(c, ch, valeur)
    texte = "" if valeur is None else str(valeur).strip()
    if k == "nom":
        if len(texte) > EN.NOM_MAX:
            return f"Nom trop long : {len(texte)} caractères (au plus {EN.NOM_MAX})"
        c["nom"] = texte
    elif k == "date":
        c["date"] = EN.lire_date(texte)
    elif k == "heure":
        x = _nombre(valeur, "heure")
        if not float(x).is_integer():
            raise ValueError(f"heure : une heure entière est attendue (« {texte[:20]} »)")
        c["heure"] = int(x)
    elif k == "durcissement/choix":
        mode, _, cle = texte.partition(":")
        du = c["durcissement"]
        if mode == "beton" and cle in L.BETONS:
            du.update(mode="beton", beton=cle)
        elif mode == "ciment" and cle in L.CIMENTS:
            du.update(mode="ciment", ciment=cle)
        elif mode == "r" and not cle:
            du["mode"] = "r"
        else:
            raise ValueError(f"durcissement/mode : « {texte[:30]} » n'est pas un choix")
    elif k in ("temperature/lieu",):
        c["temperature"]["lieu"] = texte[:200]
    elif k in ("temperature/debut", "temperature/fin"):
        t = c["temperature"]
        t[ch[1]] = EN.lire_date(texte)
        if t["debut"] and t["fin"] and t["fin"] < t["debut"]:
            if k == "temperature/fin":
                raise ValueError("temperature/fin : elle précède le début")
            t["fin"] = None          # le début passe après la fin : elle redevient automatique
    elif k in CHOIX:
        _poser(c, ch, texte)
        if k == "temperature/mode" and texte == "meteo" and not c["temperature"]["lieu"]:
            c["temperature"]["lieu"] = MT.LIEU_DEFAUT
    elif k in CASES:
        _poser(c, ch, _case(valeur))
    elif k in NOMBRES:
        _poser(c, ch, _nombre(valeur, k))
    else:
        raise ValueError(f"champ inconnu : « {k[:40]} »")
    return None


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans le coulage ouvert. Le moteur relit le coulage : une valeur hors de son domaine est REFUSÉE
    avec son message (le champ nommé), jamais ramenée en silence, et le coulage reste intact. Un relevé refusé reste à
    l'écran avec son erreur (le calcul garde le précédent)."""
    c = _coulage(etat)
    avant = copy.deepcopy(c)
    ch = [str(x) for x in chemin] or [""]
    try:
        msg = _appliquer(etat, c, ch, valeur)
        if msg:
            if BROUILLON in c and "/".join(ch) in ("temperature/releve", "temperature/fichier"):
                return msg                                         # le relevé refusé reste à l'écran, le reste est intact
            etat["coulages"][etat["actif"]] = avant
            return msg
        br = c.pop(BROUILLON, None)
        neuf = EN.normaliser(c)
        if br is not None:
            neuf[BROUILLON] = br
        etat["coulages"][etat["actif"]] = neuf
    except (ValueError, KeyError, IndexError, TypeError) as e:
        etat["coulages"][etat["actif"]] = avant
        return f"Saisie refusée — {_message(e)}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de classe : la classe cliquée devient celle du coulage (les objectifs en % la suivent)."""
    if role != "classe" or nom not in L.CLASSES:
        raise ValueError(f"choix inconnu : {role} = {nom}")
    c = _coulage(etat)
    br = c.get(BROUILLON)
    neuf = EN.normaliser(dict(_moteur(c), classe=nom))
    if br is not None:
        neuf[BROUILLON] = br
    etat["coulages"][etat["actif"]] = neuf


# ------------------------------------------------------------------ les nombres, les dates
NB, FINE = C.NBSP, C.FINE


def _lie(t: str) -> str:
    """Un texte d'un seul tenant (« J + 10,7 », « sam. 21/11 vers 5 h ») : ses espaces deviennent insécables."""
    return t.replace(" ", NB)


def _fr(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else _moins(C.fr(x, nd))


def _mpa(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else f"{_fr(x, nd)}{NB}MPa"


def _deg(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else f"{_fr(x, nd)}{NB}°C"


def _jours(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else f"{_fr(x, nd)}{NB}j"


def _jplus(t: Optional[float], prevu: bool = False) -> str:
    """Un âge réel depuis la fin du bétonnage : « J + 10,7 » ; ``prevu`` : l'âge SAISI d'une opération, écrit tel quel
    (« J + 3 », « J + 2,5 »)."""
    if t is None:
        return "—"
    return _lie(f"J + {C.nb(t, 1) if prevu else C.fr(t, 1)}")


def _taux_mots(t: Optional[float]) -> str:
    """Un taux en TEXTE, comme la pastille l'écrit (« 87 % », « > 999 % » au-delà du plafond)."""
    if t is None:
        return "—"
    x = C.pct(t)
    return f">{NB}{C.PLAFOND_TAUX}{NB}%" if x > C.PLAFOND_TAUX else f"{C.pc(x)}{NB}%"


def _cible_txt(x: float) -> str:
    """Une cible en MPa : entière si elle l'est (« 21 MPa »), sinon au dixième (« 10,5 MPa »)."""
    return f"{C.nb(x, 1)}{NB}MPa"


def _date_fr(iso: Optional[str], annee: bool = True) -> str:
    if not iso:
        return ""
    d = _dt.date.fromisoformat(iso[:10])
    return d.strftime("%d/%m/%Y" if annee else "%d/%m")


def _instant(iso: Optional[str], mot: str = "vers") -> str:
    """« 2026-11-21T05:00 » → « sam. 21/11 vers 5 h » (d'un seul tenant) ; l'année est dite quand elle n'est pas celle
    de l'année en cours de la note (``_ANNEE``)."""
    if not iso:
        return ""
    d = _dt.datetime.fromisoformat(iso)
    an = "" if d.year == _ANNEE[0] else f"/{d.year}"
    return _lie(f"{JOURS_SEMAINE[d.weekday()]} {d:%d/%m}{an} {mot} {d.hour} h")


_ANNEE = [None]            # l'année de la fin du bétonnage du coulage rendu (les dates de la même année l'omettent)


def _base_txt(base: str, t: str = "t") -> str:
    """La résistance à laquelle une cible se compare : « f<sub>ck</sub>(t) » ou « f<sub>cm</sub>(t) » (HTML)."""
    return f"f<sub>{'ck' if base == 'fck' else 'cm'}</sub>({t})"


def _v(x: float) -> str:
    """Une VALEUR substituée dans une formule, à quatre chiffres significatifs au moins ; un nombre négatif entre
    parenthèses (il suit un signe)."""
    a = abs(x)
    if a >= 1000:
        s = C.fr(x, 0 if float(x).is_integer() else 1)
    elif a >= 100:
        s = C.nb(round(x, 2), 2)
    elif a >= 10:
        s = C.nb(round(x, 3), 3)
    elif a >= 1:
        s = C.nb(round(x, 4), 4)
    elif a == 0:
        s = "0"
    else:
        nd = 4
        while nd < 8 and a < 10 ** (-(nd - 4) - 1):
            nd += 1
        s = C.nb(round(x, nd + 1), nd + 1)
    s = _moins(s)
    return f"({s})" if x < 0 else s


DECIMALES = {"MPa": 2, "j": 3, "°C": 1, "%": 1, "": 4, "kJ/mol": 2}


def _res(x: float, unite: str) -> str:
    nd = DECIMALES.get(unite, 3)
    return _moins(C.fr(x, nd)) + (f"{NB}{unite}" if unite else "")


# ------------------------------------------------------------------ les rubriques
def _menu_durcissement(c: Dict) -> str:
    """Le DURCISSEMENT en un menu de deux groupes : le bon de livraison (béton rapide, moyen, lent ; r connu), puis le
    ciment de l'EC2 (R, N, S). La brique C.choix n'a pas de groupes : le menu est écrit comme elle l'écrit (bois.py)."""
    du = c["durcissement"]
    v = {"beton": f"beton:{du['beton']}", "ciment": f"ciment:{du['ciment']}", "r": "r"}[du["mode"]]
    groupes = (("Bon de livraison · NBN EN 206", [(f"beton:{b}", f"Béton {b}") for b in L.BETONS] + [("r", "r connu")]),
               ("Ciment · EN 1992-1-1", [(f"ciment:{k}", f"Ciment {k}") for k in L.CIMENTS]))
    opts = "".join(f'<optgroup label="{C.esc(g)}">' + "".join(f'<option value="{C.esc(k)}"{" selected" if k == v else ""}>{C.esc(t)}</option>'
                                                             for k, t in items) + "</optgroup>" for g, items in groupes)
    return (f'<div class="field select"><select data-chemin="durcissement/choix" data-sorte="choix">{opts}</select>'
            f'{C.icon("chevron-down")}</div>')


def _aide_durcissement() -> str:
    ciments = " ; ".join(f"{k} : {', '.join(x['ciments'])} (s = {C.nb(x['s'], 2)})" for k, x in L.CIMENTS.items())
    betons = " ; ".join(f"{b} : {L.durcissement({'mode': 'beton', 'beton': b})['detail']}" for b in L.BETONS)
    return (f"Le durcissement fixe s de l'EC2 (3.2). Par le BON DE LIVRAISON (NBN EN 206, r = f_cm,2j / f_cm,28j) — {betons} ; s est "
            f"pris à la borne basse de la classe, du côté sûr. Par le CIMENT (EN 1992-1-1 §3.1.2(6)) — {ciments}. Les ciments au "
            "laitier (CEM III/A 42,5 N LA, le plus courant ici) sont N par la lettre de l'EC2 mais « moyen » ou « lent » sur le bon "
            "de livraison : préférer le bon, la lecture par le ciment est alors optimiste.")


HEURES = [(h, f"{h} h" + (" (midi)" if h == 12 else "")) for h in range(24)]


def _rubrique_beton(c: Dict, r: Dict) -> str:
    m = r["materiau"]
    du = m["durcissement"]
    s = C.ligne("Classe", C.choix(["classe"], c["classe"], [(k, k) for k in L.CLASSES]), "",
                f"{c['classe']} : f_ck = {C.nb(m['fck'])} MPa (cylindre), f_ck,cube = {C.nb(m['fck_cube'])} MPa, f_cm = f_ck + "
                f"{C.nb(L.DELTA_FCM)} = {C.nb(m['fcm'])} MPa (EN 1992-1-1, tableau 3.1).", full=True)
    s += C.ligne("Durcissement", _menu_durcissement(c), "", _aide_durcissement(), full=True)
    if c["durcissement"]["mode"] == "r":
        s += C.ligne("r connu", C.num(["durcissement", "r"], c["durcissement"]["r"], "", mini=L.R_MIN, maxi=L.R_MAX), "r",
                     f"r = f_cm,2j / f_cm,28j, donné par la centrale ou par des essais : {C.nb(L.R_MIN)} à {C.nb(L.R_MAX)}.")
    cale = (f" · calé sur l'essai : s = {C.nb(m['s'], 3)}, r = {C.nb(m['r'], 2)}" if m["cale"] else "")
    detail = f"{du['detail']} · " if du["detail"] else ""
    s += C.aide(f"{detail}f<sub>cm</sub>(2{NB}j) = {C.fr(du['r'] * 100, 0)}{NB}% de f<sub>cm</sub>(28{NB}j) · s = {C.nb(du['s'], 3)}{cale}")
    s += C.ligne("Fin du bétonnage", C.texte(["date"], _date_fr(c["date"]), "JJ/MM/AAAA"), "",
                 "La date de la FIN du bétonnage : t = 0. Vide : les échéances sont données en jours (J + …), sans date ; avec un relevé "
                 "daté, sa première ligne la donne.")
    s += C.ligne("à l'heure", C.choix(["heure"], c["heure"], HEURES), "",
                 "L'heure de la fin du bétonnage : le premier jour ne compte que ce qu'il en reste (midi par défaut).")
    return s


AIDE_FORMAT = ("Format : la température moyenne du jour (°C), précédée de sa date si l'on veut (AAAA-MM-JJ ou JJ/MM/AAAA), suivie de sa "
               "source (mesuré, prévision, prolongé) ; « — » : jour sans donnée (aucun durcissement compté). La première ligne est le jour "
               "du bétonnage. Les jours prévus et prolongés donnent la variante prudente (− la marge des hypothèses).")
AIDE_METEO = ("La météo est une importation de données, à la demande : le navigateur interroge Open-Meteo (géocodage, archive, "
              "prévisions) ; la note garde les températures reçues, et le calcul se fait hors ligne. Sans réseau : une "
              "température constante, ou un relevé.")


def _quand(iso: str) -> str:
    """« 2026-11-07T14:32 » → « 07/11/2026 à 14:32 »."""
    d = _dt.datetime.fromisoformat(iso)
    return f"{d:%d/%m/%Y} à {d:%H:%M}"


def _position(mt: Dict) -> str:
    """La position du lieu trouvé, à la française : « 50,85° N · 4,35° E »."""
    lat, lon = mt["latitude"], mt["longitude"]
    return (f"{C.fr(abs(lat), 2)}°{NB}{'N' if lat >= 0 else 'S'}{NB}·{NB}"
            f"{C.fr(abs(lon), 2)}°{NB}{'E' if lon >= 0 else 'O'}")


def _synthese_releve(etiquette: str, r: Dict) -> str:
    """Le badge (« lieu — n j mesurés · n j de prévision · n j prolongés ») et la rangée de mesures (T min, T moy,
    T max du relevé, t_T à 28 jours réels) d'un relevé exploitable. ``etiquette`` est du HTML (échappé par l'appelant)."""
    rel, sy = r["temperature"]["releve"], r["synthese"]
    badge = f"<b>{etiquette}</b> — {rel['n_mes']} j mesuré{'s' if rel['n_mes'] > 1 else ''}"
    if rel["n_prev"]:
        badge += f" · {rel['n_prev']} j de prévision"
    if rel["n_extra"]:
        badge += f" · {rel['n_extra']} j prolongé{'s' if rel['n_extra'] > 1 else ''}"
    mini = (f'<div class="age-mini"><div><span>T min</span><b>{_deg(rel["T_min"])}</b></div>'
            f'<div><span>T moy</span><b>{_deg(rel["T_moy"])}</b></div><div><span>T max</span><b>{_deg(rel["T_max"])}</b></div>'
            f'<div><span>t<sub>T</sub>(28{NB}j)</span><b>{_jours(sy["t_T_28"])}</b></div></div>')
    return C.aide(badge) + mini


def _zone_releve(c: Dict, indice: str) -> str:
    """La zone du relevé (une ligne par jour), avec le relevé refusé gardé à l'écran et son erreur."""
    br = c.get(BROUILLON)
    out = C.zone_texte(["temperature", "releve"], br["texte"] if br else c["temperature"]["releve"], 7, indice)
    if br:
        out += C.message(f"{C.esc(br['erreur'])} — le calcul garde le relevé précédent.", True)
    return out


def _section_meteo(c: Dict, r: Dict, coulage: int) -> str:
    """« Météo du lieu » : le lieu, les dates, le bouton (le plan du moteur), puis ce qui a été récupéré — le lieu
    trouvé, le jour et l'heure, le badge et les mesures, le détail horaire à la demande, les températures reçues,
    modifiables."""
    t, tp = c["temperature"], r["temperature"]
    auj = aujourdhui()
    d0 = _dates_meteo(dict(c, temperature=dict(t, debut=None, fin=None)), auj)[0]
    f0 = _dates_meteo(dict(c, temperature=dict(t, fin=None)), auj)[1]
    mt = tp["meteo"]
    s = (C.ligne("Lieu", C.texte(["temperature", "lieu"], t["lieu"], "ex. Namur, Belgique"), "",
                 "La ville et le pays du chantier : le premier lieu trouvé par Open-Meteo est retenu, et nommé sous le bouton.", full=True)
         + C.ligne2(("Début", C.texte(["temperature", "debut"], _date_fr(t["debut"]), f"auto · {d0:%d/%m/%Y}"),
                     "Le premier jour du relevé : le jour du bétonnage. Vide : la date de fin du bétonnage, sinon il y a 14 jours."),
                    ("Fin", C.texte(["temperature", "fin"], _date_fr(t["fin"]), f"auto · {f0:%d/%m/%Y}"),
                     f"Le dernier jour du relevé. Vide : aujourd'hui, ou le début + 5 j s'il est à venir. Prévisions jusqu'à "
                     f"+ {MT.HORIZON_PREVISION_J} j ; au-delà, les jours sont prolongés."))
         + C.bouton_meteo(["temperature", "meteo"], "Récupérer à nouveau la météo" if mt else "Récupérer la météo", plan_meteo(c, auj, coulage)))
    if mt:
        if mt["demande_changee"]:
            s += C.message("Lieu ou dates modifiés depuis la récupération — cliquez à nouveau sur « Récupérer la météo ».")
        s += C.aide(f'<span class="age-prov">Open-Meteo · récupérée le {_quand(mt["recupere"])} · {_position(mt)}</span>')
    if tp["releve"]:
        s += _synthese_releve(C.esc(mt["label"]) if mt else (C.esc(t["lieu"]) or "Relevé"), r)
    if mt:
        if mt["modifie"]:
            s += C.aide("Températures corrigées à la main depuis la récupération ; « Récupérer à nouveau la météo » remet celles du service.")
        if mt["horaire"] is None:
            s += C.bouton_meteo(["temperature", "meteo_horaire"], "Charger le détail horaire (facultatif)", plan_horaire(c, auj, coulage), petit=True)
        else:
            s += C.aide(f"Détail horaire chargé le {_quand(mt['horaire_recupere'])} : la minimale horaire sert à l'alerte de gel ; "
                        "T min h et T max h dans la vue « Tableau ».")
    if mt or t["releve"] or c.get(BROUILLON):
        s += ('<div class="grp">Températures journalières (modifiables)</div>'
              + _zone_releve(c, "Une ligne par jour : 2026-11-03;8,0;mesuré"))
    else:
        s += C.aide(f"Renseignez le lieu et les dates — passées (chantier en cours) ou à venir (prévisions jusqu'à + {MT.HORIZON_PREVISION_J} j) — "
                    f"puis récupérez la météo. En attendant : {_deg(t['T'])} constante.")
    return s + C.aide(AIDE_METEO)


def _rubrique_temperatures(c: Dict, r: Dict, coulage: int) -> str:
    t, tp = c["temperature"], r["temperature"]
    s = C.ligne("Source", C.segments(["temperature", "mode"], t["mode"], SOURCES_T), "",
                "Constante : une température pour toute la durée. Relevé : les températures journalières, collées ou lues dans un "
                "fichier. Météo du lieu : récupérées chez Open-Meteo (archive et prévisions), à la demande.", full=True)
    if t["mode"] == "constante":
        s += C.ligne("température", C.num(["temperature", "T"], t["T"], "°C", mini=EN.T_MIN, maxi=EN.T_MAX), "T",
                     f"La température moyenne, constante ({_moins(C.nb(EN.T_MIN))} à {C.nb(EN.T_MAX)} °C).")
    elif t["mode"] == "meteo":
        s += _section_meteo(c, r, coulage)
    else:
        s += (C.ligne("Lieu", C.texte(["temperature", "lieu"], t["lieu"], "ex. Namur"), "", "Le lieu du relevé (il se lit dans la note).", full=True)
              + '<div class="grp">Températures moyennes journalières</div>'
              + _zone_releve(c, "Une ligne par jour, depuis le jour du bétonnage :\n9,5\n2026-11-03;8,0\n04/11/2026;7,0;prévision")
              + f'<div class="age-fichier">{C.fichier(["temperature", "fichier"], "Lire un fichier", ".csv,.txt,.tsv")}'
              + (f'<span>{C.esc(t["fichier"])}</span>' if t["fichier"] else "<span>.csv ou .txt</span>") + "</div>")
        if tp["releve"]:
            s += _synthese_releve(C.esc(t["lieu"]) or "Relevé", r)
        s += C.aide(AIDE_FORMAT)
    return s + C.aide("Température du béton ; à défaut, celle de l'air (un élément mince exposé peut être plus froid la nuit).")


def _resume_objectif(o: Dict, ro: Dict) -> str:
    """La ligne courte d'une carte d'objectif : « 70 % f_ck · 21 MPa » ou « 5 MPa · moyenne », et « J + 3 » s'il est daté."""
    if o["mode"] == "pct":
        txt = f"{C.nb(o['pct'], 1)}{NB}% f<sub>ck</sub>"
    else:
        txt = f"{_cible_txt(o['mpa'])} {'f<sub>cm</sub>' if o['base'] == 'fcm' else 'f<sub>ck</sub>'}"
    if o["age"] is not None:
        txt += f" · {_jplus(o['age'], True)}"
    return txt


def _carte_objectif(i: int, o: Dict, ro: Dict, seul: bool) -> str:
    """La carte d'un objectif : sa ligne (le rang, le nom, la cible en bref, le chevron), puis son détail — la cible (MPa
    ou % de f_ck, et sa valeur), la résistance à laquelle elle se compare, l'âge prévu (vide : non daté), « Retirer »."""
    ch = ["objectifs", i]
    tete = (f'<div class="et-l1"><button type="button" class="et-tag" data-deplier="1" title="Objectif {i + 1} : détail">{i + 1}</button>'
            f'<div class="age-onom">{C.texte(ch + ["nom"], o["nom"], f"Objectif {i + 1}")}</div>'
            f'<span class="et-txt age-ores">{_resume_objectif(o, ro)}</span>'
            f'<button type="button" class="et-dep" data-deplier="1" title="Cible, comparaison, âge prévu">{C.icon("chevron-down")}</button></div>')
    if o["mode"] == "pct":
        valeur = C.num(ch + ["pct"], o["pct"], "% f<sub>ck</sub>", mini=EN.PCT_MIN, maxi=EN.PCT_MAX)
    else:
        valeur = C.num(ch + ["mpa"], o["mpa"], "MPa", mini=EN.MPA_MIN, maxi=EN.MPA_MAX)
    det = C.ligne("Cible", C.segments(ch + ["mode"], o["mode"], SEGMENTS_CIBLE), "",
                  "En % de f_ck : une exigence de calcul, comparée à la résistance caractéristique f_ck(t), qui suit la classe. En MPa : un "
                  "seuil fixe (un seuil de chantier se compare à la résistance MOYENNE, ce que mesurent des cubes de chantier).", full=True)
    det += C.ligne("valeur", valeur, "f<sub>c</sub>")
    if o["mode"] == "mpa":
        det += C.ligne("Comparée à", C.segments(ch + ["base"], o["base"], SEGMENTS_BASE), "",
                       "Moyenne f_cm(t) : la base des seuils de chantier (des cubes conservés sur chantier). Caractéristique f_ck(t) = f_cm(t) − 8 : "
                       "le côté sûr, par défaut pour une cible saisie à la main.", full=True)
    else:
        det += C.aide(f"Comparée à la résistance caractéristique f<sub>ck</sub>(t) : {_cible_txt(ro['cible'])} pour le {C.esc(ro['classe'])}.")
    det += C.ligne("âge prévu", C.num(ch + ["age"], o["age"], "j", auto="non daté", mini=EN.AGE_MIN, maxi=EN.AGE_MAX), "",
                   f"L'âge réel de l'opération (décoffrage, mise en charge), en jours depuis la fin du bétonnage ({C.nb(EN.AGE_MIN)} à "
                   f"{C.nb(EN.AGE_MAX)} j) : le résultat dit alors son taux. Vide : non daté, le résultat dit quand la cible est atteinte.")
    if not seul:
        det += E.retirer(ch + ["-"], "Retirer cet objectif")
    return f'<div class="et-piece age-obj" data-cible="o{i}">{tete}<div class="et-det">{det}</div></div>'


def _rubrique_objectifs(c: Dict, r: Dict) -> str:
    objs = c["objectifs"]
    cartes = "".join(_carte_objectif(i, o, dict(r["objectifs"][i], classe=c["classe"]), len(objs) == 1) for i, o in enumerate(objs))
    if len(objs) >= EN.MAX_OBJECTIFS:
        return cartes + C.aide(f"{EN.MAX_OBJECTIFS} objectifs au plus par coulage.")
    modeles = "".join(E.action(f'data-chemin="objectifs/+" data-valeur="{C.esc(k)}" title="{C.esc(src)}"', "plus",
                               f"{nom} · {C.nb(v)} MPa" if mode == "mpa" else nom)
                      for k, nom, mode, v, _b, src in M.MODELES_OBJECTIF)
    menu = E.menu(modeles, "Ajouter un objectif d'un modèle (sa source au survol)", f'{C.icon("plus")}<span>Modèles</span>', "age-modeles")
    aide = " · ".join(f"{nom} : {src}" for _k, nom, _m, _v, _b, src in M.MODELES_OBJECTIF)
    return (cartes + f'<div class="et-pied age-ajouts">{E.plus(["objectifs", "+"], "nouveau", "Objectif")}{menu}'
            f'<span class="aide" title="{C.esc("Les modèles — " + aide)}">{C.icon("info")}</span></div>')


def _aide_loi() -> str:
    pts = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    lignes = []
    for cle, lib in (("nbn", "NBN B 15-400"), ("ec2", "EC2 (B.10)"), ("perso", "E_a = 45 kJ/mol")):
        loi = {"mode": cle, "Ea_kJ": 45.0}
        lignes.append(f"{lib} : " + ", ".join(f"{C.nb(L.k_T(T, loi), 2)} à {_moins(C.nb(T, 0))} °C" for T in pts))
    return ("k(T), ce que vaut un jour à T par rapport à un jour à 20 °C — " + " ; ".join(lignes) + ". La NBN B 15-400 (complément belge "
            "à l'EN 13670, « âge fictif ») est la loi par défaut : belge, et la plus prudente par temps froid ; elle compte encore un peu "
            "sous 0 °C (prolongée jusqu'à 0 à −10 °C) et rien de plus au-dessus de 20 °C. (B.10) est écrite pour la cure thermique, et "
            "compte 0 sous 0 °C. E_a saisie : l'énergie d'activation donnée par la centrale (20 à 60 kJ/mol).")


def _rubrique_hypotheses(c: Dict, r: Dict) -> str:
    lo, x = c["loi"], c["essai"]
    s = C.ligne("Loi k(T)", C.segments(["loi", "mode"], lo["mode"], SEGMENTS_LOI), "", _aide_loi(), full=True)
    if lo["mode"] == "perso":
        s += C.ligne("énergie d'activation", C.num(["loi", "Ea_kJ"], lo["Ea_kJ"], "kJ/mol", mini=L.EA_MIN, maxi=L.EA_MAX), "E<sub>a</sub>",
                     f"La loi d'Arrhenius avec l'énergie d'activation de la centrale (R = {C.nb(L.R_GAZ)} J/(mol·K)) : {C.nb(L.EA_MIN)} à {C.nb(L.EA_MAX)} kJ/mol "
                     f"(l'EC2 suppose {C.nb(L.EA_EC2_KJ, 1)} ; les ciments au laitier montent à 45–60).")
    s += C.ligne("marge sur les prévisions", C.num(["marge"], c["marge"], "°C", mini=EN.MARGE_MIN, maxi=EN.MARGE_MAX), "ΔT",
                 "La VARIANTE PRUDENTE : les jours prévus, prolongés ou à température constante abaissés de cette marge (les prévisions "
                 f"sont souvent trop douces) ; chaque objectif donne ses deux dates. {C.nb(EN.MARGE_MIN)} à {C.nb(EN.MARGE_MAX)} °C.")
    s += C.case(["essai", "actif"], x["actif"], "Essai sur éprouvettes de chantier")
    if x["actif"]:
        s += C.ligne2(("âge", C.num(["essai", "age"], x["age"], "j", mini=EN.AGE_MIN, maxi=EN.AGE_MAX),
                       "L'âge réel des éprouvettes à l'essai, en jours depuis la fin du bétonnage (conservées sur chantier)."),
                      ("f<sub>mes</sub>", C.num(["essai", "mpa"], x["mpa"], "MPa", mini=EN.MPA_MIN, maxi=EN.ESSAI_MPA_MAX),
                       "La résistance MOYENNE mesurée sur les éprouvettes (elle se compare à f_cm(t), jamais à f_ck)."))
        s += C.ligne("Éprouvette", C.segments(["essai", "eprouvette"], x["eprouvette"], SEGMENTS_EPROUVETTE), "",
                     "Un cube de 150 mm est ramené au cylindre par f_ck / f_ck,cube de la classe (0,78 à 0,83).", full=True)
        s += C.case(["essai", "caler"], x["caler"], "Caler la courbe sur l'essai (recalcule s)")
    return s


def formulaire(etat: Dict, r: Optional[Dict] = None) -> str:
    """Les quatre rubriques du coulage ouvert : 1 Béton, 2 Températures, 3 Objectifs, 4 Hypothèses (secondaire)."""
    c = _coulage(etat)
    r = r or M.calculer(_moteur(c))
    m, tp = r["materiau"], r["temperature"]
    r1 = f"{C.esc(c['classe'])} · {_dur_court(m)}" + (f" · {_date_fr(c['date'])}" if c["date"] else "")
    n = len(c["objectifs"])
    r3 = f"{n} objectif{'s' if n > 1 else ''}"
    r4 = f"{C.esc(m['loi']['libelle'])} · ΔT {C.nb(c['marge'])}{NB}°C" + (" · essai" if c["essai"]["actif"] else "")
    return (C.section(1, "Béton", r1, _rubrique_beton(c, r), True, "bt")
            + C.section(2, "Températures", C.esc(_temperatures_txt(r)), _rubrique_temperatures(c, r, etat["actif"]), False, "tp")
            + C.section(3, "Objectifs", r3, _rubrique_objectifs(c, r), False, "ob")
            + C.section(4, "Hypothèses", r4, _rubrique_hypotheses(c, r), False, "hy", secondaire=True))


def _dur_court(m: Dict) -> str:
    """« durcissement moyen », « ciment N », « r = 0,40 » (HTML court)."""
    du = m["durcissement"]
    if du["mode"] == "beton":
        return f"durcissement {COURT_BETON[du['cle']]}"
    if du["mode"] == "ciment":
        return f"ciment {du['cle']}"
    return f"r = {C.nb(du['r'], 2)}"


def _temperatures_txt(r: Dict) -> str:
    """Les températures en quelques mots (TEXTE : le nom du lieu vient d'Open-Meteo, celui qui l'écrit en HTML l'échappe)."""
    tp = r["temperature"]
    if tp["source"] == "releve":
        n = tp["releve"]["n_jours"]
        if tp["mode"] == "meteo" and tp["meteo"]:
            return f"météo de {tp['meteo']['nom']} · {n} j"
        return f"relevé de {n} j"
    T = _deg(tp["T"]).replace(NB, " ")
    if tp["mode"] == "releve":
        return f"relevé à saisir · {T}"
    if tp["mode"] == "meteo":
        return f"températures effacées · {T}" if tp["meteo"] else f"météo à récupérer · {T}"
    return f"{T} constante"


def _dims(r: Dict) -> str:
    """« · C30/37 · moyen » : un coulage dans la liste de la note (HTML)."""
    if "materiau" not in r:
        return ""
    return f"· {C.esc(r['materiau']['classe'])} · {_dur_court(r['materiau'])}"


def arbre(etat: Dict, r: Optional[Dict] = None, resultats: Optional[List[Dict]] = None) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact) : la ligne du coulage ouvert (son nom, qu'un
    clic modifie ; la liste des coulages « nom · classe · durcissement » ; copier, supprimer), puis ses rubriques."""
    resultats = resultats or [_calculer(x) for x in etat["coulages"]]
    elements = [(c.get("nom") or f"Coulage {i + 1}", _dims(x), False) for i, (c, x) in enumerate(zip(etat["coulages"], resultats))]
    return E.arbre(GENRE, elements, etat["actif"], formulaire(etat, r), "coulage", compact=True)


# ------------------------------------------------------------------ le résultat
def _calculer(c: Dict) -> Dict:
    try:
        return M.calculer(_moteur(c))
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


NIV_ETAT = {"ok": "p", "attention": "f", "nok": "r"}


def _niv(o: Dict) -> str:
    """La couleur d'un objectif DATÉ : rouge s'il n'est pas tenu à l'âge prévu, vert foncé si sa variante prudente ne le
    tient pas (« conforme, attention »), sinon le code couleur de son taux ; « c » (consultation) sans âge prévu."""
    if o["age_prevu"] is None:
        return "r" if o["etat"] in ("non_atteignable", "jamais") else "c"
    if o["etat"] == "nok" or o["taux"] is None:
        return "r"
    return "f" if o["etat"] == "attention" else C.niveau(o["taux"])


def _taux(o: Dict):
    return C.pct(o["taux"]) if o["taux"] is not None else None


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : son objectif daté le plus chargé (le moteur : ``resume``) ; sans objectif daté, une
    consultation — le coulage ouvert et l'âge de son premier objectif."""
    dates = [(x["resume"]["taux"] if x["resume"]["taux"] is not None else float("inf"), i) for i, x in enumerate(resultats)
             if "resume" in x and x["resume"]["date"]]
    if dates:
        _, i = max(dates)
        x = resultats[i]
        o = x["objectifs"][x["resume"]["objectif"]]
        rang = {"nok": 2, "attention": 1, "ok": 0}
        pire = max((y["resume"]["etat"] for y in resultats if "resume" in y and y["resume"]["date"]), key=lambda k: rang[k])
        niv = NIV_ETAT[pire]
        t = _taux(o)
        return {"taux": t, "niveau": niv, "statut": C.STATUTS[niv][1],
                "gouvernant": f"{etat['coulages'][i].get('nom') or 'Coulage'} · {o['nom']} à J + {C.fr(o['age_prevu'], 1)}"}
    i = etat["actif"]
    x = resultats[i]
    if "objectifs" not in x:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    o = x["objectifs"][0]
    quand = f"J + {C.fr(o['t'], 1)}" if o["t"] is not None else ("non atteignable" if o["etat"] == "non_atteignable" else "jamais")
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1],
            "gouvernant": f"{etat['coulages'][i].get('nom') or 'Coulage'} · {o['nom']} {C.nb(o['cible'], 1)} MPa : {quand}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les coulages de la note sont la LIGNE de l'élément et les rubriques de la colonne de gauche (``arbre``)."""
    return ""


def _quand_atteint(o: Dict, prudent: bool = False) -> str:
    """« J + 10,7 · sam. 21/11 vers 5 h » (HTML d'un seul tenant par morceau) ; « non atteignable », « jamais »."""
    if o["etat"] == "non_atteignable" or not o["atteignable"]:
        return "non atteignable"
    t, inst = (o["prudent"]["t"], o["prudent"]["instant"]) if prudent else (o["t"], o["instant"])
    if t is None:
        return "jamais atteint"
    return _jplus(t) + (f" · {_instant(inst)}" if inst else "")


def _prudent_txt(r: Dict, o: Dict) -> str:
    """La variante prudente d'une échéance : « prudent (− 3 °C) : J + 12,1 · lun. 23/11 vers 8 h » ; rien si elle ne
    diffère pas."""
    if not r["synthese"]["prudent_distinct"] or o["t"] is None or o["prudent"]["t"] == o["t"]:
        return ""
    return f"prudent (−{NB}{C.nb(r['temperature']['marge'])}{NB}°C) : {_quand_atteint(o, True)}"


def _verdict(r: Dict) -> str:
    """Le VERDICT : le PREMIER objectif. Daté : son taux en grand, « f(J + 3) = 18,4 MPa pour 21 MPa demandés », la date
    de l'opération et quand la cible est atteinte. Non daté : l'âge où il est atteint, en grand, sa date et sa variante
    prudente."""
    o = r["objectifs"][0]
    titre = f"{o['nom']} · {C.nb(o['cible'], 1)} MPa"
    pr = _prudent_txt(r, o)
    if o["age_prevu"] is not None:
        quand = f"{_instant(o['instant_prevu'], 'à')} : " if o["instant_prevu"] else ""
        cmp_ = (f"{quand}{_base_txt(o['base'])} = {_mpa(o['f_prevu'])} pour {_cible_txt(o['cible'])}"
                f"<br>prudent {_mpa(o['f_prevu_prudent'])} · {_cible_txt(o['cible'])} atteints {_quand_atteint(o)}")
        return (f'<div class="verdict" data-calcul="o0" title="Voir le calcul">{C.chip(_taux(o), _niv(o), xl=True)}<div class="gov">'
                f'<b>{C.esc(o["nom"])} · {_jplus(o["age_prevu"], True)}</b>{cmp_}</div></div>')
    if o["t"] is None:
        cmp_ = ("la cible dépasse la résistance de la classe" if o["etat"] == "non_atteignable"
                else "jamais atteint : la loi ne compte aucun durcissement à cette température")
        return C.verdict_valeur(None, "j", titre, cmp_, "r", calcul="o0", symbole="J +", classe="age-v")
    cmp_ = (f"{_instant(o['instant']) + ' · ' if o['instant'] else ''}comparée à {_base_txt(o['base'])}"
            + (f"<br>{pr}" if pr else ""))
    return C.verdict_valeur(o["t"], "j", titre, cmp_, "c", calcul="o0", symbole="J +", classe="age-v")


def _ligne_objectif(r: Dict, i: int, o: Dict, gov: bool) -> str:
    """Une ligne du bloc « Objectifs » (deux lignes) : le nom, la cible (la classe pour le premier : elle ouvre le
    sélecteur), le taux s'il est daté, sinon l'âge atteint ; puis ce qu'on compare, en clair."""
    cible = f"{_cible_txt(o['cible'])} · {'f<sub>ck</sub>' if o['base'] == 'fck' else 'f<sub>cm</sub>'}"
    if o["age_prevu"] is not None:
        cmp_ = (f"{_base_txt(o['base'], _jplus(o['age_prevu'], True))} = {_mpa(o['f_prevu'])} · "
                f"atteint {_jplus(o['t']) if o['t'] is not None else 'jamais'}")
        etat_txt = ""
    else:
        cmp_ = _instant(o["instant"]) if o["instant"] else (_quand_atteint(o) if o["t"] is None else f"âge équivalent {_jours(o['t_T'])}")
        if _prudent_txt(r, o):
            cmp_ += f" · prudent {_jplus(o['prudent']['t'])}"
        etat_txt = (_lie(f"J + {C.fr(o['t'], 1)}") if o["t"] is not None else ("non atteignable" if o["etat"] == "non_atteignable" else "jamais"))
    piece = C.esc(r["materiau"]["classe"]) + f" · {cible}" if i == 0 else cible
    return C.ligne_verif(o["nom"], cmp_, _taux(o), _niv(o), calcul=f"o{i}", pk="classe" if i == 0 else "", piece=piece, gov=gov,
                         etat_txt=etat_txt)


def _etat_bloc_objectifs(r: Dict) -> Tuple[str, str]:
    datés = [o for o in r["objectifs"] if o["age_prevu"] is not None]
    n = len(r["objectifs"])
    if not datés:
        if any(o["etat"] in ("non_atteignable", "jamais") for o in r["objectifs"]):
            return "nok", "Non atteint"
        return "inc", f"Consultation · {n} objectif{'s' if n > 1 else ''}"
    etat = r["resume"]["etat"]
    t = r["resume"]["taux"]
    pc = f" · {_taux_mots(t)}" if t is not None else ""
    return {"ok": "ok", "attention": "warn", "nok": "nok"}[etat], C.STATUTS[NIV_ETAT[etat]][1] + pc


def _bloc_objectifs(r: Dict) -> str:
    gov = r["resume"]["objectif"] if r["resume"]["date"] else 0
    rows = "".join(_ligne_objectif(r, i, o, i == gov) for i, o in enumerate(r["objectifs"]))
    m = r["materiau"]
    efforts = (f"{C.esc(m['classe'])} · f<sub>ck</sub> {_mpa(m['fck'], 0)} · f<sub>cm</sub> {_mpa(m['fcm'], 0)} · s = {C.nb(m['s'], 3)}"
               + (" (calé)" if m["cale"] else ""))
    cls, etat_txt = _etat_bloc_objectifs(r)
    return E.bloc(0, "Objectifs", cls, etat_txt, efforts, rows, theme=True)


def _bloc_temperatures(r: Dict) -> str:
    sy, tp, gel = r["synthese"], r["temperature"], r["gel"]
    n = sy["n"]
    jours = " · ".join(f"{v} j {MOTS_SOURCE[k]}{'s' if v > 1 and k in ('mesure', 'prolonge') else ''}" for k, v in n.items()) or "température constante"
    efforts = (f"T moy {_deg(sy['T_moy'])} · min {_deg(sy['T_min'])} · max {_deg(sy['T_max'])} · loi {C.esc(r['materiau']['loi']['libelle'])}")
    rows = C.ligne_verif("Jours", jours + (f" · au-delà : {_deg(tp['T_fin'])} {'prolongée' if tp['source_fin'] == 'prolonge' else 'constante'}"
                                         if tp["source"] == "releve" else ""), None, "c", calcul="s1",
                         etat_txt=f"{sy['n_jours']} j connus" if sy["n_jours"] else _deg(tp["T"]))
    rows += C.ligne_verif("Âge équivalent", f"t<sub>T</sub> à 28{NB}j réels" + (f" · prudent {_jours(sy['t_T_28_prudent'])}" if sy["prudent_distinct"] else ""),
                          None, "c", calcul="s1", etat_txt=_jours(sy["t_T_28"]))
    if gel["alerte"]:
        quand = f"le {_date_fr(gel['date'])}" if gel["date"] else f"au jour {gel['premier_jour']}"
        cmp_ = f"gel {quand} ({_deg(gel['T'])}) avant 5{NB}MPa ({_jplus(gel['t_5MPa'])})"
        g_txt, g_niv = "protéger", "r"
    else:
        cmp_ = (f"5{NB}MPa (f<sub>cm</sub>) à {_jplus(gel['t_5MPa'])}" if gel["t_5MPa"] is not None else "5 MPa jamais atteints")
        cmp_ += " · " + ("gel ensuite" if gel["premier_jour"] is not None else "aucun jour de gel")
        if gel["prudent"]["alerte"]:
            cmp_ += " · gel possible en variante prudente"
        g_txt, g_niv = ("à surveiller", "f") if gel["prudent"]["alerte"] else ("hors gel", "p")
    if gel["base"] == "minimale":
        cmp_ += " · minimale horaire"
    rows += C.ligne_verif("Gel avant 5 MPa", cmp_, None, g_niv, calcul="gel", etat_txt=g_txt)
    cls = "nok" if gel["alerte"] else ("warn" if gel["prudent"]["alerte"] or r["temps_froid"]["rappel"] else "ok")
    etat_txt = "Gel avant 5 MPa" if gel["alerte"] else ("Temps froid" if r["temps_froid"]["rappel"] else _temperatures_txt(r))
    return E.bloc(1, "Températures", cls, etat_txt, efforts, rows, theme=True)


def _bloc_essai(r: Dict) -> str:
    x = r["essai"]
    if x is None:
        return ""
    ec = x["ecart"]
    cube = f" (cube {_mpa(x['mpa'])} × {C.nb(x['rapport_cube'], 3)})" if x["eprouvette"] == "cube" else ""
    cmp_ = f"mesuré {_mpa(x['f_cyl'])}{cube} pour {_mpa(x['f_prevu'])} prévus (f<sub>cm</sub>)"
    niv = "c" if ec is None else ("f" if ec < 0 else "p")
    row = C.ligne_verif(f"Essai à {_jplus(x['age']).replace(NB, ' ')}", cmp_, None, niv, calcul="s2",
                        etat_txt="—" if ec is None else f"{'+' if ec >= 0 else '−'}{C.fr(abs(ec), 0)} %")
    if x["cale"]:
        cal = f"courbe calée : s = {C.nb(x['s_cale'], 3)} · r = {C.nb(x['r_cale'], 2)}"
        row += C.ligne_verif("Calage", cal, None, "c", calcul="s2", etat_txt="calée")
    elif x["caler"]:
        row += C.ligne_verif("Calage", C.esc(x["calage_refuse"] or ""), None, "r", calcul="s2", etat_txt="refusé")
    cls = "warn" if (ec is not None and ec < 0 and not x["cale"]) or (x["caler"] and not x["cale"]) else "ok"
    etat_txt = "Calée sur l'essai" if x["cale"] else ("Sous la prévision" if ec is not None and ec < 0 else "Au-dessus de la prévision")
    efforts = f"t<sub>T</sub> = {_jours(x['t_T'], 2)}" + (" · indicatif (≤ 3 j équivalents)" if x["indicatif"] else "")
    return E.bloc(2, "Essai", cls, etat_txt, efforts, row, theme=True)


def _messages(r: Dict) -> str:
    """Les avertissements du moteur : le début (jusqu'aux deux-points) en gras, les symboles avec leurs indices ; les
    textes venus de l'état (le nom d'un objectif) échappés par ``symb_math``."""
    out = ""
    for a in r["avertissements"]:
        tete, sep, suite = a["message"].partition(" : ")
        h = f"<b>{C.symb_math(tete)}</b>{C.symb_math(sep + suite)}" if sep and len(tete) < 90 else C.symb_math(a["message"])
        out += C.message(h, a["niveau"] == M.GRAVE)
    return out


def _methode(r: Dict) -> str:
    """Sous les blocs : ce que l'outil est (une estimation), la méthode et ses sources, ce qui n'est pas couvert."""
    m = r["materiau"]
    lignes = [C.aide("<b>Estimation</b> : la loi de l'EC2 et la température saisie (celle de l'air à défaut de celle du béton). Pour une "
                     "opération critique (désétaiement, mise en charge, précontrainte), confirmer la résistance par des éprouvettes "
                     "conservées sur chantier ou une maturométrie étalonnée, et valider le délai avec le bureau d'études."),
              C.aide(f"<b>Méthode</b> : la résistance au jeune âge de l'EN 1992-1-1 §3.1.2 en âge équivalent ; durcissement : "
                     f"{C.esc(m['durcissement']['libelle'])} — {C.esc(m['durcissement']['source'])} ; âge équivalent : "
                     f"{C.esc(m['loi']['source'])} ; gel avant 5 MPa : EN 13670 §8.5(12)."),
              C.aide("<b>Non couvert</b> : " + C.symb_math(" ; ".join(r["non_couvert"])) + ".")]
    return f'<div class="et-hyps">{"".join(lignes)}</div>'


# ------------------------------------------------------------------ les fenêtres de calcul
def _fx(f: Dict) -> Dict:
    """Une formule du moteur → « symbole = expression = valeurs = résultat » : son gabarit (``moteur.EXPRESSIONS``), où
    chaque terme est remplacé par son symbole, puis par sa valeur ; une valeur posée par la loi (``moteur.LUES``) dit
    seulement son texte."""
    termes = f["termes"]
    cle = f["cle"]
    if cle == "beta" and termes[1]["valeur"] <= 0:
        cle = "beta_nul"
    if cle == "taux":
        return {"parts": [f["symbole"], *_fx_parts(cle, termes)], "res": _taux_mots(f["resultat"]).replace(NB, " "), "niveau": C.niveau(f["resultat"])}
    return {"parts": [f["symbole"], *_fx_parts(cle, termes)], "res": _res(f["resultat"], f["unite"]), "niveau": ""}


def _fx_parts(cle: str, termes: List[Dict]) -> List[str]:
    if cle in M.LUES:
        return [M.LUES[cle]]
    g = M.EXPRESSIONS[cle]
    sym = re.sub(r"\{(\d+)\}", lambda x: termes[int(x.group(1))]["symbole"], g).replace(" × ", "·")
    val = re.sub(r"\{(\d+)\}", lambda x: _v(termes[int(x.group(1))]["valeur"]), g)
    return [sym, val]


def _groupe(titre: str, formules: List[Dict], sous: str = "") -> str:
    st = f'<span class="ck-comb">{sous}</span>' if sous else ""
    return f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}{st}</div>{C.lignes_fx([_fx(f) for f in formules])}</div>'


def _hyps(r: Dict) -> List[str]:
    """Les hypothèses d'une fenêtre, en pastilles : la classe, le durcissement, la loi, les températures, la fin du
    bétonnage, la marge."""
    m, tp, tm = r["materiau"], r["temperature"], r["temps"]
    fin = (f"{_date_fr(tm['date'])} à {tm['heure']}{NB}h" if tm["date"] else f"{tm['heure']}{NB}h (sans date)")
    out = [f"Béton <b>{C.esc(m['classe'])}</b> · f<sub>ck</sub> = {_mpa(m['fck'], 0)} · f<sub>cm</sub> = {_mpa(m['fcm'], 0)}",
           f"{C.esc(m['durcissement']['libelle'].capitalize())} · s = {C.nb(m['s'], 3)} · r = {C.nb(m['r'], 3)}" + (" · calé sur l'essai" if m["cale"] else ""),
           f"Loi k(T) : {C.esc(m['loi']['libelle'])}" + (f" · E<sub>a</sub> = {C.nb(m['loi']['Ea_kJ'], 1)}{NB}kJ/mol" if m["loi"]["Ea_kJ"] else ""),
           f"Températures : {C.esc(_temperatures_txt(r))}",
           f"Fin du bétonnage : {fin}"]
    if r["synthese"]["prudent_distinct"]:
        out.append(f"Variante prudente : − {C.nb(tp['marge'])}{NB}°C sur les jours non mesurés")
    return out


COLONNES_OBJ = ("Objectif", "Comparée à", "Âge équivalent requis", "Atteint", "Taux")


def _ligne_tab(r: Dict, o: Dict, gov: bool) -> Dict:
    pr = _prudent_txt(r, o)
    sous = f"cible {_cible_txt(o['cible'])}" + (f" · {C.nb(o['pct'], 1)}{NB}% de f<sub>ck</sub>" if o["mode"] == "pct" else "")
    if o["age_prevu"] is not None:
        sous += f" · prévu à {_jplus(o['age_prevu'], True)}"
    li = {"nom": C.esc(o["nom"]), "sous": sous, "formule": f"{_base_txt(o['base'])} ≥ {_cible_txt(o['cible'])}",
          "valeur": f"t<sub>T,req</sub> = {_jours(o['t_T'], 2)}" if o["t_T"] is not None else "non atteignable",
          "limite": _quand_atteint(o) + (f"<br><small>{pr}</small>" if pr else ""), "gov": gov}
    if o["age_prevu"] is not None:
        li.update(taux=_taux(o), niveau=_niv(o))
    else:
        li.update(etat_txt="—" if o["t"] is None else f"J + {C.fr(o['t'], 1)}", niveau=_niv(o))
    return li


def _pied_fenetre(r: Dict) -> str:
    return (f'<div class="ck-autres age-pied-ck"><b>Non couvert par ce module</b> : {C.symb_math(" ; ".join(r["non_couvert"]))}.</div>')


def _volets(r: Dict, nom: str) -> str:
    """Les fenêtres de calcul : une par objectif (o0, o1…), « tout le calcul » des objectifs (s0), les températures et
    l'âge équivalent (s1), le gel (gel), l'essai (s2)."""
    hyps, pied = _hyps(r), _pied_fenetre(r)
    m = r["materiau"]
    base = f"{C.esc(m['classe'])} · {C.esc(m['durcissement']['libelle'])} · {C.esc(m['loi']['libelle'])}"
    g_mat = _groupe("Le béton", m["formules"], C.esc(m["durcissement"]["source"]))
    gov = r["resume"]["objectif"] if r["resume"]["date"] else 0
    out = ""
    for i, o in enumerate(r["objectifs"]):
        detail = g_mat + _groupe(f"« {o['nom']} »", o["formules"], "EN 1992-1-1 §3.1.2 · (3.2)")
        out += C.fenetre_calcul(f"o{i}", f"{nom} · {C.esc(o['nom'])}", f"{base} · âge où la cible est atteinte", hyps,
                                [_ligne_tab(r, o, False)], detail, pied, COLONNES_OBJ)
    detail = g_mat + "".join(_groupe(f"« {o['nom']} »", o["formules"]) for o in r["objectifs"])
    out += C.fenetre_calcul("s0", f"{nom} · objectifs", f"{base} · tout le calcul", hyps,
                            [_ligne_tab(r, o, i == gov) for i, o in enumerate(r["objectifs"])], detail, pied, COLONNES_OBJ)
    # les températures : k au-delà des jours connus, et chaque jour connu dans le tableau
    tp, sy = r["temperature"], r["synthese"]
    lignes_t = [{"nom": f"Jour {j['jour']}", "sous": _date_fr(j["date"]) or MOTS_SOURCE[j["source"]],
                 "formule": f"{_deg(j['T'])} · {MOTS_SOURCE[j['source']]}", "valeur": f"k = {C.fr(j['k'], 3)}",
                 "limite": f"t<sub>T</sub> = {_jours(j['t_T'], 2)}", "etat_txt": f"{C.fr(j['fin'], 2)}{NB}j", "niveau": "c"}
                for j in r["profil"][:sy["n_jours"] or 3]]
    detail = _groupe("Au-delà des jours connus" if tp["source"] == "releve" else "La température constante", [tp["formule_k"]],
                     f"{_deg(tp['T_fin'])} · {C.esc(m['loi']['libelle'])}")
    out += C.fenetre_calcul("s1", f"{nom} · températures", f"{base} · âge équivalent t<sub>T</sub> = Σ Δt·k(T)", hyps, lignes_t, detail, pied,
                            ("Jour", "Température", "k(T)", "t<sub>T</sub> en fin de jour", "Âge réel"))
    gel = r["gel"]
    lig_gel = [{"nom": "Gel avant 5 MPa", "sous": "EN 13670 §8.5(12)", "formule": f"f<sub>cm</sub>(t) ≥ 5{NB}MPa avant le gel",
                "valeur": f"5{NB}MPa à {_jplus(gel['t_5MPa'])}",
                "limite": (f"gel le {_date_fr(gel['date'])}" if gel["date"] else (f"gel au jour {gel['premier_jour']}" if gel["premier_jour"] is not None
                                                                                 else "aucun jour de gel")),
                "etat_txt": "protéger" if gel["alerte"] else "hors gel", "niveau": "r" if gel["alerte"] else "p"}]
    out += C.fenetre_calcul("gel", f"{nom} · gel", f"{base} · le gel avant 5 MPa", hyps, lig_gel,
                            _groupe("5 MPa en résistance moyenne", gel["formules"]), pied)
    x = r["essai"]
    if x:
        lig = [{"nom": f"Essai à {_jplus(x['age'])}", "sous": "cube de 150 mm" if x["eprouvette"] == "cube" else "cylindre",
                "formule": "f<sub>mes</sub> face à f<sub>cm</sub>(t)", "valeur": f"{_mpa(x['f_cyl'])} pour {_mpa(x['f_prevu'])}",
                "limite": f"t<sub>T</sub> = {_jours(x['t_T'], 2)}",
                "etat_txt": "—" if x["ecart"] is None else f"{'+' if x['ecart'] >= 0 else '−'}{C.fr(abs(x['ecart']), 1)} %", "niveau": "c"}]
        detail = _groupe("L'essai face à la prévision", x["formules"],
                         "calage : " + ("retenu" if x["cale"] else (C.esc(x["calage_refuse"]) if x["calage_refuse"] else "non demandé")))
        out += C.fenetre_calcul("s2", f"{nom} · essai", f"{base} · essai sur éprouvettes", hyps, lig, detail, pied,
                                ("Essai", "Comparaison", "Mesuré / prévu", "Âge équivalent", "Écart"))
    return out


# ------------------------------------------------------------------ les vues
def _vue_tableau(r: Dict) -> str:
    """Le profil jour par jour (le moteur : ``tableau``) — un tableau HTML posé dans la feuille : jour, date, T, source,
    (T min h, T max h), k, t_T en fin de jour, β_cc, f_cm(t), f_ck(t)."""
    tb = r["tableau"]
    date = r["temps"]["date"] is not None
    hor = tb["horaire"]
    tete = "<th>Jour</th>" + ("<th>Date</th>" if date else "") + '<th>T<small>°C</small></th><th class="g">Source</th>'
    if hor:
        tete += "<th>T min h<small>°C</small></th><th>T max h<small>°C</small></th>"
    tete += ("<th>k</th><th>t<sub>T</sub><small>j</small></th><th>β<sub>cc</sub></th><th>f<sub>cm</sub>(t)<small>MPa</small></th>"
             "<th>f<sub>ck</sub>(t)<small>MPa</small></th>")
    o0 = r["objectifs"][0]
    corps = ""
    for li in tb["lignes"]:
        cls = " ".join(x for x in ("on" if o0["t"] is not None and li["debut"] < o0["t"] <= li["fin"] + 1e-12 else "",
                                   "dom" if li["indicatif"] else "", "gel" if li["gel"] else "") if x)
        tr = f'<tr class="{cls}">' if cls else "<tr>"
        tr += f'<th>{li["jour"]}</th>'
        if date:
            tr += f'<td class="d">{_date_fr(li["date"])}</td>'
        tr += f'<td>{_fr(li["T"]) if li["T"] is not None else "—"}</td><td class="s {li["source"]}">{MOTS_SOURCE[li["source"]]}</td>'
        if hor:
            tr += f'<td>{_fr(li["T_min_h"])}</td><td>{_fr(li["T_max_h"])}</td>'
        tr += (f'<td>{C.fr(li["k"], 3)}</td><td>{C.fr(li["t_T"], 2)}</td><td>{C.fr(li["beta"], 3)}</td>'
               f'<td>{C.fr(li["fcm"], 1)}</td><td class="f">{C.fr(li["fck"], 1)}</td>')
        corps += tr + "</tr>"
    pied = ("Jour 0 : le jour du bétonnage (compté pour ce qui en reste après l'heure de fin) ; t<sub>T</sub> : l'âge équivalent à 20 °C "
            "en fin de jour ; en gris, t<sub>T</sub> ≤ 3 j (valeurs indicatives, EC2 §3.1.2(5)) ; en bleu, le jour où le premier objectif "
            "est atteint.")
    if hor:
        pied += " T min h, T max h : les extrêmes horaires du jour (Open-Meteo, détail horaire) ; « — » : aucune valeur horaire."
    return (f'<div class="age-vue"><div class="age-defile"><table class="age-tab"><thead><tr>{tete}</tr></thead><tbody>{corps}</tbody></table></div>'
            f'<div class="age-pied">{pied}</div></div>')


def _entete(r: Dict, titre: str, legende: str) -> str:
    """L'en-tête technique d'une vue : la classe et le durcissement ; la fin du bétonnage et les températures ; les
    objectifs en bref."""
    m, tm = r["materiau"], r["temps"]
    fin = f"fin du bétonnage le {_date_fr(tm['date'])} à {tm['heure']} h" if tm["date"] else f"fin du bétonnage à {tm['heure']} h (sans date)"
    objs = " ; ".join(f"{o['nom']} {C.nb(o['cible'], 1)} MPa : " + (f"J + {C.fr(o['t'], 1)}" if o["t"] is not None else "—") for o in r["objectifs"])
    sous = f"{C.esc(m['classe'])} · {C.esc(m['durcissement']['libelle'])} · s{NB}={NB}{C.nb(m['s'], 3)}"
    return E.entete(titre, sous, [f"{fin} · {_temperatures_txt(r)} · loi {m['loi']['libelle']}", objs], None, legende)


def _vues(r: Dict, ident: str) -> List[Dict]:
    tp = r["temperature"]
    if tp["mode"] == "meteo" and tp["meteo"]:
        mt = tp["meteo"]
        t_titre = "Météo du lieu"
        t_leg = f"{mt['label']} · Open-Meteo, {_quand(mt['recupere'])}" + (" · corrigée à la main" if mt["modifie"] else "")
    elif tp["source"] == "releve":
        t_titre, t_leg = "Relevé des températures", (tp["lieu"] or "relevé") + f" · {tp['releve']['n_jours']} j"
    else:
        t_titre, t_leg = "Températures", _temperatures_txt(r)
    t_leg += " · barres : la température du jour, par source · courbe : l'âge équivalent cumulé"
    c_leg = "f_cm(t) et f_ck(t) en âge réel · objectifs atteints · variante prudente en tirets · t_T ≤ 3 j grisé"
    vues = [("courbe", "Courbe des résistances", "Courbe", DA.courbe, c_leg),
            ("temperatures", t_titre, "Températures", DA.temperatures, t_leg),
            ("tableau", "Valeurs jour par jour", "Tableau", None, "jour, température, k, âge équivalent, résistances")]
    out = []
    for cle, titre, court, fn, leg in vues:
        svg = fn(f"{ident}_{cle}", r) if fn else _vue_tableau(r)
        out.append({"cle": cle, "titre": titre, "court": court, "svg": svg, "entete": _entete(r, titre.upper(), leg), "legende": leg,
                    "echelle": False})
    return out


def _avis(cle: Optional[str]) -> List[str]:
    return [AVIS_V1] if cle == "v1" else []


def resultat(etat: Dict) -> Dict:
    """Le résultat du coulage ouvert : la tête, l'avis d'ouverture (dit une fois, retiré de l'état), le verdict, les
    blocs « Objectifs », « Températures », « Essai », les avertissements, la méthode ; les fenêtres de calcul, les vues,
    la colonne de gauche, le résumé de la note."""
    avis = _avis(etat.pop(CLE_AVIS, None))
    c = _coulage(etat)
    r = M.calculer(_moteur(c))
    _ANNEE[0] = int(r["temps"]["date"][:4]) if r["temps"]["date"] else None
    resultats = [r if i == etat["actif"] else _calculer(x) for i, x in enumerate(etat["coulages"])]
    nom = C.esc(c.get("nom") or f"Coulage {etat['actif'] + 1}")
    m = r["materiau"]
    tete = E.tete_resultats(nom, f"{C.esc(m['classe'])} · {_dur_court(m)} · {C.esc(_temperatures_txt(r))}")
    blocs = _bloc_objectifs(r) + _bloc_temperatures(r) + _bloc_essai(r)
    pied = (f'<div class="pfoot"><div class="mass"><span>Âge équivalent · loi {C.esc(m["loi"]["libelle"])}</span>'
            f'<b>t<sub>T</sub>(28{NB}j) = {_jours(r["synthese"]["t_T_28"])}</b></div><div class="legend">{C.legende()}</div></div>')
    corps = (f'<div class="pbody">{tete}{"".join(C.message(C.esc(a)) for a in avis)}{_verdict(r)}<div class="et-blocs">{blocs}</div>'
             f'{_messages(r)}{_methode(r)}</div>{pied}')
    rs = resume_note(etat, resultats)
    statut = C.statut(rs["niveau"]) if rs["niveau"] in C.STATUTS else C.statut("i")
    return {"resultat": corps, **({"avis": avis} if avis else {}), "volets": _volets(r, nom), "vues": _vues(r, f"ag{etat['actif']}"),
            "phrase": "", "statut_html": statut, "resume": rs, "donnees": arbre(etat, r, resultats), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de classe
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Chaque classe (C16/20 à C50/60) avec le même durcissement et les mêmes températures, face au PREMIER objectif
    (``moteur.candidats_classes``) : l'âge où il est atteint, son taux à l'âge prévu s'il est daté ; ✓ la classe du
    coulage, ★ la plus petite qui tient l'âge prévu (aucune étoile sans âge prévu : il n'y a rien à tenir)."""
    if role != "classe":
        raise ValueError(f"sélecteur inconnu : {role}")
    c = _coulage(etat)
    cands = M.candidats_classes(_moteur(c))
    o = c["objectifs"][0]
    date = o["age"] is not None
    i_o = next((k for k, x in enumerate(cands) if x["optimal"]), 0)
    i_c = next((k for k, x in enumerate(cands) if x["retenue"]), i_o)

    def prop(x):
        quand = f"atteint à {_jplus(x['t'])}" if x["t"] is not None else ("non atteignable" if not x["atteignable"] else "jamais atteint")
        s = f"{C.esc(x['classe'])} · f<sub>ck</sub> = {_mpa(x['fck'], 0)} · cible {_cible_txt(x['cible'])} · {quand}"
        if date:
            s += f" · f({_jplus(x['age_prevu'], True)}) = {_mpa(x['f_prevu'])}" + ("" if x["ok"] else " · <b>ne tient pas</b>")
        return s

    rows = []
    for x in cands:
        row = {"nom": x["classe"], "col": C.fr(x["t"], 1) if x["t"] is not None else "—", "cur": x["retenue"], "opt": x["optimal"], "prop": prop(x)}
        if date:
            row.update(taux=C.pct(x["taux"]) if x["taux"] is not None else None, niveau="r" if not x["ok"] else ("f" if x["etat"] == "attention" else C.niveau(x["taux"])))
            if x["taux"] is None:
                row.update(etat="non", niveau="r")
        else:
            row.update(taux=None, niveau="r" if x["t"] is None else "c", etat="non" if x["t"] is None else "")
        rows.append(row)
    titre = f"Classe de béton · {C.esc(o['nom'] or 'Objectif 1')}" + (f" à {_jplus(o['age'], True)}" if date else "")
    sous = "triées par résistance · le même durcissement, les mêmes températures"
    return C.selecteur("classe", titre, sous, "", rows, max(0, min(i_o, i_c) - 4), "j : âge réel où l'objectif est atteint",
                       etoile="la plus petite classe qui tient l'âge prévu" if date else "", pastille="taux" if date else "")

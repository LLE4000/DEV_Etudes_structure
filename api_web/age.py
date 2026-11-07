# -*- coding: utf-8 -*-
"""Le module « Maturité du béton » du site : une note de plusieurs COULAGES (chacun : un béton, une cure, les cinq
analyses de l'écran Streamlit), les champs de saisie, le résultat, les fenêtres de calcul, le sélecteur de classe, la
courbe f_ck(t), le relevé des températures et le tableau des valeurs — rendus en fragments HTML.

Le calcul est celui du moteur ``beton/maturite`` (extrait de l'écran Streamlit ``modules/age_beton.py`` v3.2, dont il
reproduit les nombres au bit près — tests/test_maturite_moteur.py). Ce fichier MET EN FORME : il ne calcule rien.

Ce qui change par rapport à Streamlit, et pourquoi :
- la température de cure a TROIS sources : constante, relevé saisi (tableau collé ou fichier lu), ou météo du lieu
  (Open-Meteo, comme le mode « Météo » de l'écran). La météo est une IMPORTATION de données à la demande : le moteur
  fait le plan des appels (beton/maturite/meteo.py), le bouton le porte (C.bouton_meteo), le navigateur fait les appels
  vers les trois adresses d'Open-Meteo et rend les réponses brutes, que ce module lit et GARDE dans la note — les
  températures deviennent un relevé (modifiable à la main), et le calcul se fait hors ligne ;
- une saisie hors du domaine de l'écran (que Streamlit ramène en silence à sa borne) est REFUSÉE avec un message, et
  une saisie absurde aussi (délai ou âge d'essai au-delà de 10 ans, résistance au-delà de 200 MPa : l'écran n'a pas de
  borne haute) ;
- un relevé mal écrit n'est pas perdu : il reste affiché avec l'erreur et sa ligne, le calcul garde le précédent ;
- les défauts NON CONSERVATIFS de l'écran sont reproduits et SIGNALÉS (docs/beton/MATURITE_MOTEUR.md §0) : le seuil ②
  annoncé trop tôt par un relevé aux jours froids d'abord (avertissement grave), l'essai ⑤ lu comme une résistance
  caractéristique (la lecture en moyenne est donnée, à trancher).
"""
import copy
import datetime as _dt
import math
from typing import Dict, List, Optional, Sequence

from beton.maturite import entree as EN
from beton.maturite import meteo as MT
from beton.maturite import moteur as M
from beton.maturite import releve as RL
from beton.maturite.ec2 import CIMENT_DETAIL, CLASSES, HORIZON_PREVISION_J

from . import commun as C
from . import dessin_age as DA

MODULE = "age"
PAGE = "maturite-beton"

BROUILLON = "saisie_refusee"            # le site : un relevé refusé, gardé à l'écran avec son erreur (pas un champ du moteur)
CIMENTS_COURTS = {"R": "R · rapide", "N": "N · normal", "S": "S · lent"}
CIMENTS_SEGMENTS = {"R": "Rapide", "N": "Normal", "S": "Lent"}
SOURCES_CURE = [("constante", "Constante"), ("releve", "Relevé"), ("meteo", "Météo du lieu")]
AGES_USUELS = (3, 7, 14, 28)
NOMS_CHAMPS = {"nom": ("Nom", ""), "classe": ("Classe", ""), "ciment": ("Ciment", ""), "date_coulage": ("Date de coulage", ""),
               "temperature/mode": ("Cure", ""), "temperature/T": ("Température de cure", "°C"), "temperature/lieu": ("Lieu", ""),
               "temperature/releve": ("Relevé", ""), "temperature/debut": ("Début de la météo", ""), "temperature/fin": ("Fin de la météo", ""),
               "temperature/meteo": ("Météo du lieu", ""), "temperature/meteo_horaire": ("Détail horaire", ""),
               "avance": ("Énergie d'activation ajustée", ""),
               "Ea_kJ": ("Énergie d'activation", "kJ/mol"), "age": ("Âge", "j"), "seuil/mode": ("Cible du seuil", ""),
               "seuil/pct": ("Pourcentage du seuil", "%"), "seuil/mpa": ("Résistance du seuil", "MPa"), "seuil/afficher": ("Seuil affiché", ""),
               "optim/mpa": ("Résistance cible", "MPa"), "optim/delai": ("Délai", "j"), "optim/afficher": ("Classe affichée", ""),
               "comparateur/classes": ("Classes comparées", ""), "comparateur/ciment": ("Ciment du comparateur", ""),
               "comparateur/afficher": ("Comparateur affiché", ""), "essai/mpa": ("Mesure de l'essai", "MPa"),
               "essai/age": ("Âge de l'essai", "j"), "essai/T": ("T de cure de l'essai", "°C"), "essai/afficher": ("Essai affiché", "")}
TEXTES = {"nom", "temperature/lieu", "date_coulage", "temperature/debut", "temperature/fin"}
CHOIX = {"classe", "ciment", "temperature/mode", "seuil/mode", "comparateur/ciment"}
CASES = {"avance", "seuil/afficher", "optim/afficher", "comparateur/afficher", "essai/afficher"}
NOMBRES = {"temperature/T", "Ea_kJ", "seuil/mpa", "optim/mpa", "optim/delai", "essai/mpa", "essai/age", "essai/T"}
ENTIERS = {"age", "seuil/pct"}

# un relevé d'exemple (le tableau « Données météo » de l'écran Streamlit, collé) : il sert au parcours de l'écran
EXEMPLE_RELEVE = ("Date;T moy (°C);Source\n2026-11-02;9,5;mesuré\n2026-11-03;8,0;mesuré\n2026-11-04;—;mesuré\n2026-11-05;6,5;mesuré\n"
                  "2026-11-06;7,0;prévision\n2026-11-07;8,5;prévision\n2026-11-08;8,0;prolongé\n2026-11-09;8,0;prolongé")


def _etat_releve(**k) -> Dict:
    e = EN.nouvel_element("Voile R+1")
    e["temperature"].update(mode="releve", releve=EXEMPLE_RELEVE, lieu="Namur")
    e["age"] = 6
    for cle, v in k.items():
        a, b = cle.split("__")
        e[a][b] = v
    return {"version": 1, "coulages": [e], "actif": 0}


def aujourdhui() -> _dt.date:
    """Le jour où l'écran est rendu (celui du navigateur, dans Pyodide) : il partage les dates de la météo entre
    l'archive et les prévisions (le plan du bouton), comme l'écran Streamlit avec ``date.today()``."""
    return _dt.date.today()


def maintenant() -> str:
    """Le jour et l'heure d'une récupération de la météo (AAAA-MM-JJTHH:MM), écrits dans la note."""
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M")


# une récupération d'exemple (le parcours de l'écran, aucun appel) : Bruxelles, du 27/10 au 25/11/2026, récupérée le
# 07/11/2026 — l'archive jusqu'au 31/10, les prévisions du 01/11 au 23/11, deux jours prolongés. Les réponses ont le format
# d'Open-Meteo ; les températures sont inventées.
EXEMPLE_METEO = {"lieu": MT.LIEU_DEFAUT, "debut": "2026-10-27", "fin": "2026-11-25", "aujourdhui": "2026-11-07"}
_GEO_EXEMPLE = {"results": [{"id": 2800866, "name": "Bruxelles", "latitude": 50.85045, "longitude": 4.34878, "country_code": "BE",
                             "admin1": "Bruxelles-Capitale", "country": "Belgique", "timezone": "Europe/Brussels"}], "generationtime_ms": 0.5}


def _meteo_exemple(trouve: bool = True, horaire: bool = False) -> Dict:
    """Ce que le navigateur rendrait pour la récupération d'exemple : {"lieu", "plan", "geocodage", "reponses"}."""
    x = EXEMPLE_METEO
    pos = {"latitude": 50.85045, "longitude": 4.34878}
    p = MT.plan(x["lieu"], x["debut"], x["fin"], x["aujourdhui"], horaire=horaire, position=pos if horaire else None)
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
    """La note d'exemple en mode météo ; ``importee`` : la récupération d'exemple déjà faite et gardée."""
    e = EN.nouvel_element("Dalle R+2")
    x = EXEMPLE_METEO
    e["temperature"].update(mode="meteo", lieu=x["lieu"], debut=x["debut"], fin=x["fin"])
    if importee:
        v = _meteo_exemple()
        e["temperature"]["meteo"] = MT.enregistrement(v["plan"], v["geocodage"], v["reponses"], "2026-11-07T08:30")
        e["temperature"]["releve"] = MT.releve(MT.lire_journalier(v["plan"], v["reponses"]))
        e["age"] = 20
    return {"version": 1, "coulages": [e], "actif": 0}


# les requêtes de plus du parcours de l'écran (api_web/parcours.py) : le relevé collé ou lu, un relevé refusé, un essai,
# une date, le sélecteur de classe, une note où tout est affiché ; la météo du lieu : récupérée, lieu introuvable, le
# détail horaire, une réponse qui ne suit pas le plan
PARCOURS = [
    {"action": "modifier", "chemin": ["temperature", "mode"], "valeur": "releve", "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["temperature", "releve"], "valeur": EXEMPLE_RELEVE.replace("9,5", "10,5"), "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["temperature", "releve"], "valeur": "12\nabc\n13", "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["temperature", "fichier"], "valeur": {"nom": "releve.csv", "texte": EXEMPLE_RELEVE}},
    {"action": "modifier", "chemin": ["temperature", "fichier"], "valeur": {"nom": "faux.csv", "texte": "douze"}},
    {"action": "modifier", "chemin": ["temperature", "releve"], "valeur": "", "etat": _etat_releve()},
    {"action": "modifier", "chemin": ["essai", "mpa"], "valeur": "21,6"},
    {"action": "modifier", "chemin": ["date_coulage"], "valeur": "02/11/2026"},
    {"action": "modifier", "chemin": ["seuil", "mode"], "valeur": "mpa"},
    {"action": "modifier", "chemin": ["temperature", "T"], "valeur": "-5"},
    {"action": "choisir", "role": "classe", "nom": "C35/45"},
    {"action": "calculer", "etat": _etat_releve(seuil__afficher=True, optim__afficher=True, comparateur__afficher=True, essai__mpa=12.0,
                                                essai__afficher=True)},
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
    return {"version": 1, "coulages": [EN.nouvel_element("Coulage 1")], "actif": 0}


def _moteur(c: Dict) -> Dict:
    """Le coulage tel que le moteur le lit (sans ce que le site garde à côté)."""
    return {k: v for k, v in c.items() if k != BROUILLON}


def _relire(c) -> Dict:
    if not isinstance(c, dict):
        raise ValueError("un coulage de la note est illisible")
    e = EN.normaliser(_moteur(c))
    br = c.get(BROUILLON)
    if isinstance(br, dict) and isinstance(br.get("texte"), str) and isinstance(br.get("erreur"), str) and len(br["texte"]) <= RL.MAX_TEXTE:
        e[BROUILLON] = {"texte": br["texte"], "erreur": br["erreur"][:400]}
    return e


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; un coulage illisible lève une erreur claire."""
    coulages = etat.get("coulages") if isinstance(etat, dict) else None
    if not coulages:
        return nouvel_etat()
    if not isinstance(coulages, list):
        raise ValueError("la note est abîmée : « coulages » n'est pas une liste")
    out = {"version": 1, "coulages": [_relire(c) for c in coulages]}
    try:
        actif = int(etat.get("actif", 0) or 0)
    except (TypeError, ValueError):
        actif = 0
    out["actif"] = min(max(0, actif), len(out["coulages"]) - 1)
    return out


def _coulage(etat: Dict) -> Dict:
    return etat["coulages"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les coulages de la note : en ouvrir un, en ajouter, copier l'ouvert, en supprimer un (jamais le dernier)."""
    cs = etat["coulages"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(cs) - 1)
    elif action == "ajouter":
        cs.append(EN.nouvel_element(f"Coulage {len(cs) + 1}"))
        etat["actif"] = len(cs) - 1
    elif action == "dupliquer":
        c = copy.deepcopy(cs[etat["actif"]])
        c["nom"] = f"{c.get('nom') or 'Coulage'} (copie)"[:200]
        cs.append(c)
        etat["actif"] = len(cs) - 1
    elif action == "supprimer" and len(cs) > 1:
        i = min(max(0, int(index or 0)), len(cs) - 1)
        del cs[i]
        if etat["actif"] > i or etat["actif"] >= len(cs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _message(e: Exception) -> str:
    """Le message du moteur, le chemin du champ remplacé par son nom : « temperature/T : -6 est sous le minimum -5 » →
    « Température de cure : −6 est sous le minimum −5 (°C) »."""
    s = str(e)
    tete, sep, reste = s.partition(" : ")
    if sep and tete in NOMS_CHAMPS:
        nom, u = NOMS_CHAMPS[tete]
        return f"{nom} : {reste.replace('-', '−')}" + (f" ({u})" if u else "")
    return s[:1].upper() + s[1:]


def _bornes_age(c: Dict) -> None:
    """Quand la borne de l'âge change (un relevé plus court, le retour à la température constante), l'âge qui la
    dépasse revient à la position de l'écran : min(28, borne)."""
    borne = EN.age_max(_moteur(c))
    if c["age"] > borne:
        c["age"] = EN.age_defaut(borne)


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
    """Les dates de la récupération : celles saisies, sinon celles de l'écran au premier affichage (le coulage = la date
    de coulage du béton, sinon aujourd'hui − 14 j ; la fin = max(aujourd'hui, coulage + 5 j))."""
    t = c["temperature"]
    debut = _dt.date.fromisoformat(t["debut"]) if t["debut"] else MT.dates_par_defaut(c["date_coulage"], auj)[0]
    fin = _dt.date.fromisoformat(t["fin"]) if t["fin"] else MT.dates_par_defaut(debut.isoformat(), auj)[1]
    return debut, fin


def plan_meteo(c: Dict, auj: Optional[_dt.date] = None) -> Dict:
    """Le plan de la récupération du coulage (le bouton « Récupérer la météo ») ; {"erreur": message} si elle est
    impossible (pas de lieu, fin avant le coulage, dates hors de portée)."""
    auj = auj or aujourdhui()
    debut, fin = _dates_meteo(c, auj)
    try:
        return MT.plan(c["temperature"]["lieu"], debut, fin, auj)
    except ValueError as e:
        return {"erreur": str(e)}


def plan_horaire(c: Dict, auj: Optional[_dt.date] = None) -> Dict:
    """Le plan du détail horaire (le second bouton) : le lieu trouvé, les dates de la récupération gardée."""
    try:
        return MT.plan_horaire(c["temperature"]["meteo"], auj or aujourdhui())
    except ValueError as e:
        return {"erreur": str(e)}


def _plan_recu(valeur) -> Optional[_dt.date]:
    """Le jour du plan que le navigateur a exécuté (il le renvoie avec les réponses) ; None s'il est illisible."""
    try:
        return _dt.date.fromisoformat(valeur["plan"]["aujourdhui"])
    except (TypeError, KeyError, ValueError):
        return None


MODIFIEE = "Lieu ou dates modifiés pendant la récupération : cliquez à nouveau sur « Récupérer la météo »."


def _importer_meteo(c: Dict, valeur) -> Optional[str]:
    """La météo que le navigateur a récupérée ({"lieu", "plan", "geocodage", "reponses"}, BRUTS) : relue par le moteur
    (beton/maturite/meteo.py) contre le plan du coulage, écrite en relevé (modifiable ensuite), et GARDÉE dans la note
    (la récupération : lieu, dates, jour et heure, réponses). Un refus rend un message ; le coulage reste intact."""
    auj = _plan_recu(valeur) if isinstance(valeur, dict) else None
    if auj is None:
        return "Météo illisible : le navigateur n'a pas rendu le plan de la récupération. Rechargez la page et recommencez."
    t = c["temperature"]
    p = plan_meteo(c, auj)
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


def _importer_horaire(c: Dict, valeur) -> Optional[str]:
    """Le détail horaire récupéré (à la demande) : relu contre le plan horaire de la récupération gardée, et gardé."""
    rec = c["temperature"]["meteo"]
    if not rec:
        return "Récupérez d'abord la météo du lieu : le détail horaire la complète."
    auj = _plan_recu(valeur) if isinstance(valeur, dict) else None
    if auj is None:
        return "Détail horaire illisible : le navigateur n'a pas rendu le plan de la récupération."
    p = plan_horaire(c, auj)
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


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans le coulage ouvert. Le moteur relit le coulage : une valeur hors du domaine de l'écran est
    refusée avec un message, et le coulage reste intact."""
    c = _coulage(etat)
    avant = copy.deepcopy(c)
    ch = [str(x) for x in chemin]
    k = "/".join(ch)
    try:
        if k in ("temperature/meteo", "temperature/meteo_horaire"):
            msg = (_importer_meteo if k == "temperature/meteo" else _importer_horaire)(c, valeur)
            if msg:
                etat["coulages"][etat["actif"]] = avant
                return msg
        elif k == "temperature/releve":
            msg = _releve(c, "" if valeur is None else str(valeur))
            if msg:
                return msg
        elif k == "temperature/fichier":
            if not isinstance(valeur, dict) or not isinstance(valeur.get("texte"), str):
                return "Fichier illisible"
            msg = _releve(c, valeur["texte"], str(valeur.get("nom") or "fichier"))
            if msg:
                return msg
        elif ch[:2] == ["comparateur", "classes"] and len(ch) == 3:
            if not ch[2].isdigit() or int(ch[2]) >= len(CLASSES):
                return f"Classe comparée inconnue : « {ch[2]} »"
            cl = CLASSES[int(ch[2])]
            liste = c["comparateur"]["classes"]
            if valeur is True or valeur == "true":
                if cl not in liste:
                    liste.append(cl)                              # comme le multiselect : la classe ajoutée va au bout
            elif cl in liste:
                liste.remove(cl)
        elif k in TEXTES:
            texte = "" if valeur is None else str(valeur).strip()
            if k == "date_coulage":
                c[k] = EN.lire_date(texte)
            elif k in ("temperature/debut", "temperature/fin"):
                t = c["temperature"]
                t[ch[1]] = EN.lire_date(texte)
                if t["debut"] and t["fin"] and t["fin"] < t["debut"]:
                    if k == "temperature/fin":
                        raise ValueError("temperature/fin : elle précède le début")
                    t["fin"] = None          # le coulage passe après la fin : elle redevient automatique (l'écran la recale)
            elif k == "nom":
                c[k] = texte[:200]
            else:
                c["temperature"]["lieu"] = texte[:200]
        elif k in CHOIX:
            _poser(c, ch, str(valeur))
            if k == "temperature/mode" and c["temperature"]["mode"] == "meteo" and not c["temperature"]["lieu"]:
                c["temperature"]["lieu"] = MT.LIEU_DEFAUT          # « Lieu du chantier » de l'écran au premier affichage
        elif k in CASES:
            _poser(c, ch, bool(valeur) if isinstance(valeur, bool) else str(valeur).lower() in ("true", "1", "oui"))
        elif k in NOMBRES or k in ENTIERS:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            if k in ENTIERS:
                if abs(x - round(x)) > 1e-9:
                    return f"{NOMS_CHAMPS[k][0]} : un nombre entier est attendu (« {valeur} »)"
                x = int(round(x))
            if k == "seuil/mpa":
                fck = int(c["classe"].split("/")[0][1:])
                if x > fck:
                    return f"Résistance du seuil : {C.nb(x)} MPa dépasse f_ck = {fck} MPa (le champ de l'écran est borné à f_ck)"
            _poser(c, ch, x)
        else:
            return f"Champ inconnu : {k}"
        if k != "age":
            _bornes_age(c)
        br = c.pop(BROUILLON, None)
        neuf = EN.normaliser(c)
        if br is not None:
            neuf[BROUILLON] = br
        etat["coulages"][etat["actif"]] = neuf
    except (ValueError, KeyError, IndexError, TypeError) as e:
        etat["coulages"][etat["actif"]] = avant
        return f"Saisie refusée — {_message(e)}"
    return None


def _poser(c: Dict, ch: List[str], v) -> None:
    d = c
    for x in ch[:-1]:
        d = d[x]
    if ch[-1] not in d:
        raise KeyError(ch[-1])
    d[ch[-1]] = v


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de classe (③) : la classe cliquée devient celle du coulage."""
    if role != "classe" or nom not in CLASSES:
        raise ValueError(f"choix inconnu : {role} = {nom}")
    c = _coulage(etat)
    c["classe"] = nom
    br = c.pop(BROUILLON, None)
    neuf = EN.normaliser(c)
    if br is not None:
        neuf[BROUILLON] = br
    etat["coulages"][etat["actif"]] = neuf


# ------------------------------------------------------------------ les nombres
def _mpa(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else C.fr(x, nd).replace("-", "−")


def _j(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None else C.fr(x, nd).replace("-", "−")


def _v(x: float, nd: int = 4) -> str:
    """Une valeur d'une formule, sans zéros inutiles (4 décimales au plus)."""
    if x is None:
        return "—"
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    ent, _, dec = s.lstrip("-").partition(".")
    ent = f"{int(ent):,}".replace(",", " ")
    return ("−" if s.startswith("-") else "") + ent + ("," + dec if dec else "")


def _deg(x: float) -> str:
    """Une température saisie, écrite telle quelle, avec le vrai signe moins : « −5 », « 12,5 »."""
    return C.nb(x).replace("-", "−")


def _date_fr(iso: Optional[str]) -> str:
    return _dt.date.fromisoformat(iso).strftime("%d/%m/%Y") if iso else ""


def _cure(r: Dict) -> str:
    """La cure en une ligne (TEXTE : le nom du lieu vient d'Open-Meteo, celui qui l'écrit en HTML l'échappe)."""
    tp = r["temperature"]
    if tp["source"] == "releve":
        if tp["mode"] == "meteo" and tp["meteo"]:
            return f"météo de {tp['meteo']['nom']} · {tp['releve']['n_jours']} j · T moy {_mpa(tp['T'])} °C"
        return f"relevé de {tp['releve']['n_jours']} j · T moy {_mpa(tp['T'])} °C"
    if tp["mode"] == "releve":
        return "relevé à saisir · 20 °C"
    if tp["mode"] == "meteo":
        return "météo à récupérer · 20 °C"
    return f"{_deg(tp['T'])} °C constante"


def _cure_courte(r: Dict) -> str:
    """La cure en deux mots, pour la liste des coulages de la note (la ligne doit tenir sur une tablette)."""
    tp = r["temperature"]
    return {"releve": "relevé", "meteo": "météo"}.get(tp["mode"]) or f"{_deg(tp['T'])} °C"


def _quand(iso: str) -> str:
    """« 2026-11-07T14:32 » → « 07/11/2026 à 14:32 »."""
    d = _dt.datetime.fromisoformat(iso)
    return f"{d:%d/%m/%Y} à {d:%H:%M}"


def _position(mt: Dict) -> str:
    """La position du lieu trouvé, à la française : « 50,85° N · 4,35° E »."""
    lat, lon = mt["latitude"], mt["longitude"]
    return (f"{C.fr(abs(lat), 2)}°{C.NBSP}{'N' if lat >= 0 else 'S'}{C.NBSP}·{C.NBSP}"
            f"{C.fr(abs(lon), 2)}°{C.NBSP}{'E' if lon >= 0 else 'O'}")


def _synthese_releve(etiquette: str, tp: Dict) -> str:
    """Le badge de l'écran (« lieu — n j mesurés · n j de prévision · n j prolongés ») et la rangée de mesures (T min,
    T moy, T max, t_e, t_e défavorable) d'un relevé exploitable. ``etiquette`` est du HTML (échappé par l'appelant)."""
    rel = tp["releve"]
    badge = f"<b>{etiquette}</b> — {rel['n_mes']} j mesuré{'s' if rel['n_mes'] > 1 else ''}"
    if rel["n_prev"]:
        badge += f" · {rel['n_prev']} j de prévision"
    if rel["n_extra"]:
        badge += f" · {rel['n_extra']} j prolongé{'s' if rel['n_extra'] > 1 else ''}"
    mini = (f'<div class="age-mini"><div><span>T min</span><b>{_mpa(rel["T_min"])} °C</b></div>'
            f'<div><span>T moy</span><b>{_mpa(tp["T"])} °C</b></div><div><span>T max</span><b>{_mpa(rel["T_max"])} °C</b></div>'
            f'<div><span>t<sub>e</sub></span><b>{_j(rel["t_e_releve"])} j</b></div>'
            + (f'<div><span>t<sub>e</sub> défav.</span><b>{_j(rel["t_e_releve_defav"])} j</b></div>' if rel["t_e_releve_defav"] is not None else "")
            + "</div>")
    return C.aide(badge) + mini


def _zone_releve(c: Dict, indice: str) -> str:
    """La zone du relevé (une ligne par jour), avec le relevé refusé gardé à l'écran et son erreur."""
    br = c.get(BROUILLON)
    out = C.zone_texte(["temperature", "releve"], br["texte"] if br else c["temperature"]["releve"], 8, indice)
    if br:
        out += C.message(f"{C.esc(br['erreur'])} — le calcul garde le relevé précédent.", True)
    return out


AIDE_FORMAT = ("Format : la température moyenne du jour (°C), précédée de sa date si l'on veut (AAAA-MM-JJ ou JJ/MM/AAAA), "
               "suivie de sa source (mesuré, prévision, prolongé) ; « — » : jour sans donnée. Le tableau « Données météo » "
               "de Streamlit se colle tel quel. Les jours prévus et prolongés donnent la variante défavorable (− 3 °C).")
AIDE_METEO = ("La météo est une importation de données, à la demande : le navigateur interroge Open-Meteo (géocodage, archive, "
              "prévisions) ; la note garde les températures reçues, et le calcul se fait hors ligne. Sans réseau : une "
              "température constante, ou un relevé.")


def _section_meteo(c: Dict, r: Dict) -> str:
    """La cure « Météo du lieu » : le lieu, les dates, le bouton (le plan du moteur), puis ce qui a été récupéré — le
    lieu trouvé, le jour et l'heure, le badge et les mesures de l'écran, le détail horaire à la demande, et les
    températures reçues, modifiables."""
    L = C.ligne
    t, tp = c["temperature"], r["temperature"]
    auj = aujourdhui()
    d0, f0 = _dates_meteo(dict(c, temperature=dict(t, debut=None, fin=None)), auj)
    f0 = _dates_meteo(dict(c, temperature=dict(t, fin=None)), auj)[1]
    mt = tp["meteo"]
    s = (L("Lieu", C.texte(["temperature", "lieu"], t["lieu"], "ex. Bruxelles, Belgique"), full=True,
           aide="La ville et le pays du chantier : le premier lieu trouvé par Open-Meteo est retenu, et nommé sous le bouton.")
         + L("Début", C.texte(["temperature", "debut"], _date_fr(t["debut"]), f"auto · {d0:%d/%m/%Y}"), full=True,
             aide="Le premier jour du relevé : le coulage (« Coulage » de l'écran Streamlit), passé (chantier en cours) ou à venir. "
                  "Vide : la date de coulage du béton, sinon il y a 14 jours.")
         + L("Fin", C.texte(["temperature", "fin"], _date_fr(t["fin"]), f"auto · {f0:%d/%m/%Y}"), full=True,
             aide=f"Le dernier jour du relevé : le contrôle ou le décoffrage (« Fin (décoffrage) » de l'écran Streamlit). Vide : "
                  f"aujourd'hui, ou le début + 5 j s'il est à venir. Prévisions jusqu'à + {HORIZON_PREVISION_J} j ; au-delà, les jours "
                  "sont prolongés.")
         + C.bouton_meteo(["temperature", "meteo"], "Récupérer à nouveau la météo" if mt else "Récupérer la météo", plan_meteo(c, auj)))
    if mt:
        if mt["demande_changee"]:
            s += C.message("Lieu ou dates modifiés depuis la récupération — cliquez à nouveau sur « Récupérer la météo ».")
        s += C.aide(f'<span class="age-prov">Open-Meteo · récupérée le {_quand(mt["recupere"])} · {_position(mt)}</span>')
    if tp["releve"]:
        s += _synthese_releve(C.esc(mt["label"]) if mt else (C.esc(t["lieu"]) or "Relevé"), tp)
    if mt:
        if mt["modifie"]:
            s += C.aide("Températures corrigées à la main depuis la récupération ; « Récupérer à nouveau la météo » remet celles du service.")
        if mt["horaire"] is None:
            s += C.bouton_meteo(["temperature", "meteo_horaire"], "Charger le détail horaire (facultatif)", plan_horaire(c, auj), petit=True)
        else:
            s += C.aide(f"Détail horaire chargé le {_quand(mt['horaire_recupere'])} : T min h et T max h de chaque jour, dans la vue « Tableau ».")
    if mt or t["releve"] or c.get(BROUILLON):
        s += ('<div class="grp">Températures journalières (modifiables)</div>'
              + _zone_releve(c, "Une ligne par jour : 2026-11-03;8,0;mesuré"))
    else:
        s += C.aide(f"Renseignez le lieu et les dates — passées (chantier en cours) ou à venir (prévisions jusqu'à + {HORIZON_PREVISION_J} j) — "
                    "puis récupérez la météo. T = 20 °C par défaut.")
    return s + C.aide(AIDE_METEO)


# ------------------------------------------------------------------ le cadre Données
def formulaire(etat: Dict, r: Dict) -> str:
    c = _coulage(etat)
    L = C.ligne
    m, tp, a = r["materiau"], r["temperature"], r["age"]
    # 1 · le béton
    s1 = (L("Nom", C.texte(["nom"], c["nom"], "ex. Voile R+1"), full=True)
          + L("Classe", C.choix(["classe"], c["classe"], [(k, k) for k in CLASSES]), full=True)
          + L("Ciment", C.segments(["ciment"], c["ciment"], list(CIMENTS_SEGMENTS.items())), full=True)
          + C.aide(f"Classe {m['ciment']} : {C.esc(m['ciment_detail'])} — s = {C.nb(m['s'], 2)} (EC2 §3.1.2)")
          + L("Coulage", C.texte(["date_coulage"], _date_fr(c["date_coulage"]), "JJ/MM/AAAA"), full=True,
              aide="La date du coulage date le seuil ② ; vide : le seuil est donné en jours."))
    # 2 · la cure : constante, relevé saisi, ou météo du lieu
    t = c["temperature"]
    s2 = L("Cure", C.segments(["temperature", "mode"], t["mode"], SOURCES_CURE), full=True)
    if t["mode"] == "constante":
        s2 += (L("température", C.num(["temperature", "T"], t["T"], "°C", mini=EN.T_MIN, maxi=EN.T_MAX), "T",
                 "Température de cure, constante (−5 à 60 °C).")
               + C.aide("Une cure variable (chantier en cours, hiver) : « Relevé » pour coller les températures journalières, "
                        "« Météo du lieu » pour les récupérer chez Open-Meteo."))
    elif t["mode"] == "meteo":
        s2 += _section_meteo(c, r)
    else:
        s2 += (L("Lieu", C.texte(["temperature", "lieu"], t["lieu"], "ex. Bruxelles"), full=True)
               + '<div class="grp">Températures moyennes journalières</div>'
               + _zone_releve(c, "Une ligne par jour, du coulage au dernier jour :\n9,5\n2026-11-03;8,0\n04/11/2026;7,0;prévision")
               + f'<div class="age-fichier">{C.fichier(["temperature", "fichier"], "Lire un fichier", ".csv,.txt,.tsv")}'
               + (f'<span>{C.esc(t["fichier"])}</span>' if t["fichier"] else "<span>.csv ou .txt</span>") + "</div>")
        if tp["releve"]:
            s2 += _synthese_releve(C.esc(t["lieu"]) or "Relevé", tp)
        s2 += C.aide(AIDE_FORMAT)
        s2 += C.aide(f"Les températures du lieu peuvent aussi se récupérer chez Open-Meteo (prévisions à {HORIZON_PREVISION_J} j) : "
                     "« Météo du lieu ».")
    # 3 · l'âge choisi ①
    borne = a["age_max"]
    usuels = [(x, f"{x} j") for x in AGES_USUELS if x <= borne]
    s3 = L("âge réel", C.num(["age"], c["age"], "j", entier=True, mini=1, maxi=borne), "t",
           f"Âge réel du béton, en jours entiers : 1 à {borne} j" + (" (la durée du relevé)." if tp["source"] == "releve" else "."))
    if len(usuels) >= 2:
        s3 += L("Âges usuels", C.segments(["age"], c["age"], usuels), full=True)
    # 4 · le seuil ②
    sl = c["seuil"]
    s4 = L("Cible", C.segments(["seuil", "mode"], sl["mode"], [("pct", "% de fck(28)"), ("mpa", "Résistance")]), full=True)
    if sl["mode"] == "pct":
        s4 += L("pourcentage", C.choix(["seuil", "pct"], sl["pct"], [(p, f"{p} %") for p in EN.POURCENTAGES]), "p",
                "Le seuil en pourcentage de f_ck(28) : 30 à 100 %, par pas de 5 (décoffrage, mise en tension…).")
    else:
        s4 += L("résistance", C.num(["seuil", "mpa"], sl["mpa"], "MPa", mini=1, maxi=min(m["fck28"], EN.MPA_MAX)), "f<sub>s</sub>",
                f"La résistance à atteindre : 1 MPa à f_ck = {m['fck28']} MPa.")
    s4 += C.case(["seuil", "afficher"], sl["afficher"], "Afficher le seuil sur la courbe")
    # 5 · la classe pour un délai ③
    op = c["optim"]
    s5 = (L("résistance cible", C.num(["optim", "mpa"], op["mpa"], "MPa", mini=1, maxi=EN.MPA_MAX), "f<sub>c</sub>")
          + L("délai", C.num(["optim", "delai"], op["delai"], "j", mini=0.5, maxi=EN.JOURS_MAX), "t<sub>d</sub>",
              "Le délai réel, en jours (0,5 j au moins).")
          + C.case(["optim", "afficher"], op["afficher"], "Afficher la classe minimale sur la courbe")
          + C.aide("La plus petite classe (C20/25 à C50/60) qui atteint la cible au délai, avec le même ciment ; la classe "
                   "se change dans le résultat (★ : la plus petite qui tient)."))
    # 6 · le comparateur ④
    cp = c["comparateur"]
    cases = "".join(C.case(["comparateur", "classes", i], cl in cp["classes"], cl) for i, cl in enumerate(CLASSES))
    s6 = (f'<div class="grp">Classes comparées</div><div class="age-classes">{cases}</div>'
          + L("Ciment", C.segments(["comparateur", "ciment"], cp["ciment"], list(CIMENTS_SEGMENTS.items())), full=True)
          + C.case(["comparateur", "afficher"], cp["afficher"], "Afficher les classes sur la courbe")
          + C.aide("La classe du coulage n'est pas retracée : c'est la courbe de référence."))
    # 7 · l'essai ⑤
    es = c["essai"]
    s7 = (L("mesure", C.num(["essai", "mpa"], es["mpa"], "MPa", mini=0, maxi=EN.MPA_MAX), "f<sub>mes</sub>",
            "Résistance mesurée sur éprouvette ; 0 : pas d'essai.")
          + L("âge de l'essai", C.num(["essai", "age"], es["age"], "j", mini=0.5, maxi=EN.JOURS_MAX), "t")
          + L("cure de l'essai", C.num(["essai", "T"], es["T"], "°C", mini=EN.T_MIN, maxi=EN.T_MAX), "T", "Température de cure de l'éprouvette (−5 à 60 °C).")
          + C.case(["essai", "afficher"], es["afficher"], "Afficher l'essai sur la courbe"))
    # 8 · la méthode
    s8 = (C.case(["avance"], c["avance"], "Ajuster l'énergie d'activation (sur essais)")
          + L("énergie d'activation", C.num(["Ea_kJ"], c["Ea_kJ"] if c["avance"] else tp["Ea_kJ"], "kJ/mol", desactive=not c["avance"],
                                            mini=EN.EA_MIN, maxi=EN.EA_MAX), "E<sub>a</sub>",
              "EC2 annexe B : E_a/R = 4 000 K, soit 33,3 kJ/mol ; 20 à 60 kJ/mol sur essais.")
          + C.aide("Domaine de validité de l'EC2 : 3 j &lt; t<sub>e</sub> &lt; 28 j (§3.1.2(5)) ; en dessous, la résistance se "
                   "détermine par essais."))
    # les résumés des sections repliées
    sr = r["seuil"]
    r4 = f"{_mpa(sr['cible'])} MPa · " + (f"après {_j(sr['t'])} j" if sr["t"] is not None else "non atteignable")
    o = r["optim"]
    r5 = f"{C.nb(o['mpa'])} MPa en {C.nb(o['delai'])} j → {o['classe'] or 'aucune classe'}"
    r6 = f"{len(cp['classes'])} classe{'s' if len(cp['classes']) > 1 else ''} · ciment {cp['ciment']}" + (" · affiché" if cp["afficher"] else "")
    ex = r["essai"]
    r7 = f"{C.nb(es['mpa'])} MPa à {C.nb(es['age'])} j → ≈ {ex['classe']}" if ex["actif"] else "aucun essai"
    r8 = f"E<sub>a</sub> = {C.nb(tp['Ea_kJ'], 1)} kJ/mol" + (" · ajustée" if c["avance"] else " · EC2")
    return (C.section(1, "Béton", f"{m['classe']} · ciment {m['ciment']} · s = {C.nb(m['s'], 2)}", s1, True, "bt")
            + C.section(2, "Température de cure", C.esc(_cure(r)), s2, True, "tp")
            + C.section(3, "Résistance à un âge", f"{a['t']} j → {_mpa(a['fck'], 2)} MPa", s3, True, "ag")
            + C.section(4, "Seuil de résistance", r4, s4, False, "se")
            + C.section(5, "Classe pour un délai", r5, s5, False, "op")
            + C.section(6, "Comparateur de classes", r6, s6, False, "cp")
            + C.section(7, "Essai sur éprouvette", r7, s7, False, "es")
            + C.section(8, "Méthode", r8, s8, False, "me"))


# ------------------------------------------------------------------ le résultat
def _niv_part(x: Optional[float]) -> str:
    return "c" if x is None else C.niveau(max(x, 0.0))


def _taux_aff(x: Optional[float]):
    """Un taux à afficher : borné (une résistance nulle ou négative donne un taux sans objet : 999 %)."""
    return C.pct(9.99) if x is None else C.pct(min(max(x, 0.0), 9.99))


def _niv_classe(cl: Dict) -> str:
    """La couleur d'une classe au délai : rouge si elle ne tient pas (le critère du module, f + 10⁻⁶ ≥ cible), le
    code couleur sinon (le taux borné à 100 % : la tolérance du module ne fait pas virer au rouge une classe qui tient)."""
    if not cl["ok"] or cl["taux"] is None:
        return "r"
    return C.niveau(min(cl["taux"], 1.0))


def _taux_classe(cl: Dict):
    if cl["taux"] is None:
        return C.pct(9.99)
    return C.pct(min(cl["taux"], 1.0) if cl["ok"] else min(cl["taux"], 9.99))


def _cible_txt(sr: Dict, fck28: int) -> str:
    """La cible du seuil comme l'écran Streamlit l'écrit : « 75 % = 22,5 MPa » ou « 20,0 MPa (67 %) »."""
    if sr["mode"] == "pct":
        return f"{sr['pct']} % = {_mpa(sr['cible'])} MPa"
    return f"{_mpa(sr['cible'])} MPa ({C.fr(sr['cible'] / fck28 * 100, 0)} %)"


def _ecart_txt(x: float) -> str:
    return f"{x:+.0f} %".replace("-", "−")


VERDICTS_ESSAI = {"coherent": "cohérent avec", "dessus": "au-dessus de", "sous": "sous"}


def lignes_resultat(r: Dict) -> str:
    """Une ligne par analyse de l'écran Streamlit : ① (et sa variante défavorable), ② le seuil, ③ la classe minimale,
    ⑤ l'essai — chacune dit la réponse de l'écran (le texte de son bloc), la barre la situe."""
    a, sr, o, es, m = r["age"], r["seuil"], r["optim"], r["essai"], r["materiau"]
    rows = C.ligne_res(f"Résistance à {a['t']} j", f"{_mpa(a['fck'], 2)} MPa · t<sub>e</sub> {_j(a['t_e'])} j",
                       C.pct(max(a["pct"], 0.0) / 100), _niv_part(a["pct"] / 100), calcul="age", gov=True)
    if a["defav"]:
        d = a["defav"]
        rows += C.ligne_res(f"Défavorable à {a['t']} j", f"{_mpa(d['fck'], 2)} MPa · t<sub>e</sub> {_j(d['t_e'])} j",
                            C.pct(max(d["pct"], 0.0) / 100), _niv_part(d["pct"] / 100), calcul="age")
    # ② : la cible, le temps d'obtention et sa date (le bloc de l'écran) ; la barre : la cible face à f_ck à l'âge choisi
    if sr["t"] is None:
        piece = f"{_mpa(sr['cible'])} MPa · non atteignable"
    else:
        piece = f"{_mpa(sr['cible'])} MPa · après {_j(sr['t'])} j" + (f" · {_date_fr(sr['date'])}" if sr["date"] else "")
    rows += C.ligne_res(f"Seuil à {a['t']} j", piece, _taux_aff(sr["taux_age"]), "r" if sr["taux_age"] is None else C.niveau(sr["taux_age"]),
                        calcul="seuil")
    # ③ : la classe minimale et sa résistance au délai (le bloc de l'écran) ; ★ ; un clic ouvre les sept classes. Le délai
    # et la cible sont dans le titre de la fenêtre, du sélecteur et le résumé de la section (la ligne reste courte : tablette)
    ret = next(x for x in o["classes"] if x["classe"] == m["classe"])
    coulage = (f"classe du coulage {m['classe']} : {_mpa(ret['fck'])} MPa à {C.nb(o['delai'])} j"
               + (" — elle suffit" if ret["ok"] else " — elle ne suffit pas"))
    if o["classe"]:
        opt = next(x for x in o["classes"] if x["classe"] == o["classe"])
        rows += C.ligne_res("Classe minimale", f"{C.esc(o['classe'])} · {_mpa(o['fck'])} MPa", _taux_classe(opt),
                            _niv_classe(opt), calcul="optim", pk="classe", leger=f"La plus petite classe qui tient ; {coulage}")
    else:
        der = o["classes"][-1]
        rows += C.ligne_res("Classe minimale", "aucune ne suffit", _taux_classe(der), "r",
                            calcul="optim", pk="classe")
    # ⑤ : la classe estimée et f_ck,28 estimé (le premier bloc de l'écran) ; le verdict va dans le message
    if es["actif"]:
        rows += C.ligne_res(f"Essai à {C.nb(es['age'])} j", f"≈ {es['classe']} · f<sub>ck,28</sub> ≈ {_mpa(es['fck28_est'])} MPa",
                            C.pct(min(max(es["fck28_est"], 0.0) / m["fck28"], 9.99)), "c", calcul="essai")
    return rows


def message_essai(r: Dict) -> str:
    """Le verdict de l'essai ⑤ (le second bloc de l'écran : cohérent, au-dessus, sous, et l'écart), et la lecture en
    résistance MOYENNE que le module ne fait pas (défaut non conservatif, à trancher : docs/beton/MATURITE_MOTEUR.md)."""
    es, m = r["essai"], r["materiau"]
    if not es["actif"]:
        return ""
    verdict = (f"Essai ⑤ : <b>{VERDICTS_ESSAI[es['verdict']]} {C.esc(m['classe'])}</b> (écart {_ecart_txt(es['ecart'])}) — "
               f"f<sub>ck,28</sub> estimé ≈ {_mpa(es['fck28_est'])} MPa, classe ≈ {C.esc(es['classe'])}.")
    note = (f"Essai ⑤, à trancher par le bureau : comme l'écran Streamlit, la mesure est lue comme f<sub>ck</sub>(t) ; lue comme une "
            f"résistance moyenne (une éprouvette), f<sub>ck,28</sub> ≈ {_mpa(es['fck28_est_moyenne'])} MPa.")
    return C.message(verdict, es["verdict"] == "sous") + C.message(note)


GRAVES = {"hors_domaine_age", "seuil_non_atteignable", "aucune_classe", "essai_hors_domaine", "releve_inexploitable", "seuil_optimiste"}


def _verdict(r: Dict, nom: str) -> str:
    a = r["age"]
    niv = "c" if a["hors_domaine"] else C.niveau(max(a["pct"], 0.0) / 100)
    if niv == "f":
        niv = "p"
    tp = r["temperature"]
    cure = f"T<sub>moy</sub> {_mpa(tp['T'])} °C" if tp["source"] == "releve" else f"{_deg(tp['T'])} °C"
    gov = (f"{C.fr(a['pct'], 0).replace('-', '−')} % de f<sub>ck</sub> = {r['materiau']['fck28']} MPa"
           f"<br>t<sub>e</sub> = {_j(a['t_e'])} j · {cure}"
           + ("<br>hors du domaine de l'EC2 (t<sub>e</sub> ≤ 3 j)" if a["hors_domaine"] else ""))
    return (f'<div class="verdict age-v"><span class="chip xl {niv}">{_mpa(a["fck"], 1)}<small>{C.NBSP}MPa</small></span>'
            f'<div class="gov"><b>Résistance à {a["t"]} j · {C.esc(r["materiau"]["classe"])}</b>{gov}</div></div>')


def _hypotheses(r: Dict) -> List[str]:
    m, tp = r["materiau"], r["temperature"]
    if tp["source"] == "releve" and tp["meteo"]:
        cure = (f"Cure : <b>météo de {C.esc(tp['meteo']['nom'])}</b> (Open-Meteo, récupérée le {_quand(tp['meteo']['recupere'])}"
                f"{', corrigée à la main' if tp['meteo']['modifie'] else ''}) · {tp['releve']['n_jours']} j · T<sub>moy</sub> = {_mpa(tp['T'])} °C")
    elif tp["source"] == "releve":
        cure = f"Cure : <b>relevé</b> de {tp['releve']['n_jours']} j · T<sub>moy</sub> = {_mpa(tp['T'])} °C"
    else:
        cure = f"Cure à <b>{_deg(tp['T'])} °C</b> constante"
    return [f"Béton <b>{C.esc(m['classe'])}</b> · f<sub>ck</sub> = {m['fck28']} MPa · f<sub>cm</sub> = {C.nb(m['fcm'])} MPa",
            f"Ciment <b>{m['ciment']}</b> · s = {C.nb(m['s'], 2)} · {C.esc(m['ciment_detail'])}",
            cure, f"E<sub>a</sub> = {C.nb(tp['Ea_kJ'], 3)} kJ/mol · E<sub>a</sub>/R = {_v(tp['Ea_R'], 1)} K"]


def _fx(parts: List[str], res: str, niveau: str = "") -> Dict:
    return {"parts": parts, "res": res, "niveau": niveau}


def _groupe(titre: str, lignes: List[Dict]) -> str:
    return (f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div>'
            + C.lignes_fx(lignes).replace("β_cc", "β<sub>cc</sub>") + "</div>")


def _formules_age(r: Dict) -> List[Dict]:
    """Les lignes « symbole = formule = valeurs = résultat » de ① : chaque nombre vient des formules du moteur."""
    out = []
    for f in r["age"]["formules"]:
        t = {x["symbole"]: x["valeur"] for x in f["termes"]}
        k, res = f["cle"], f["resultat"]
        if k == "T_abs":
            out.append(_fx(["T_abs", "T + 273,15", f"{_v(t['T'])} + 273,15"], f"{_v(res, 2)} K"))
        elif k == "facteur":
            out.append(_fx(["k_T", "exp(−E_a/R·[1/T_abs − 1/293,15])", f"exp(−{_v(t['E_a/R'], 1)} × [1/{_v(t['T_abs'], 2)} − 1/293,15])"], _v(res, 4)))
        elif k == "t_e":
            out.append(_fx(["t_e", "t·k_T", f"{_v(t['t'])} × {_v(t['facteur'], 4)}"], f"{_v(res, 3)} j"))
        elif k == "t_e_releve":
            out.append(_fx(["t_e", "Σ exp(−E_a/R·[1/(273,15 + T_i) − 1/293,15])", f"somme sur les {int(t['n'])} premiers jours du relevé"],
                           f"{_v(res, 3)} j"))
        elif k == "beta":
            out.append(_fx(["β_cc(t_e)", "exp(s·[1 − √(28/t_e)])", f"exp({_v(t['s'], 2)} × [1 − √(28/{_v(t['t_e'], 3)})])"], _v(res, 4)))
        elif k == "fcm":
            out.append(_fx(["f_cm", "f_ck + 8", f"{_v(t['fck'])} + 8"], f"{_v(res)} MPa"))
        elif k == "fck_t":
            out.append(_fx(["f_ck(t)", "β_cc(t_e)·f_cm − 8", f"{_v(t['beta'], 4)} × {_v(t['fcm'])} − 8"], f"{_mpa(res, 2)} MPa"))
        elif k == "fck_t_28":
            out.append(_fx(["f_ck(t)", "f_ck (t_e ≥ 28 j)", f"{_v(t['fck'])}"], f"{_mpa(res, 2)} MPa"))
        elif k == "pct":
            out.append(_fx(["f_ck(t) / f_ck", f"{_mpa(t['fck_t'], 2)} / {_v(t['fck'])}"], f"{C.fr(res, 1).replace('-', '−')} %"))
    return out


FORMULES_EC2 = ('<div class="age-ec2">'
                '<div><i>f</i><sub>ck</sub>(<i>t</i>) = β<sub>cc</sub>(<i>t</i><sub>e</sub>)·<i>f</i><sub>cm</sub> − 8'
                '<span>pour <i>t</i><sub>e</sub> &lt; 28 j (éq. 3.1, 3.2) ; <i>f</i><sub>cm</sub> = <i>f</i><sub>ck</sub> + 8</span></div>'
                '<div><i>f</i><sub>ck</sub>(<i>t</i>) = <i>f</i><sub>ck</sub><span>pour <i>t</i><sub>e</sub> ≥ 28 j</span></div>'
                '<div>β<sub>cc</sub>(<i>t</i><sub>e</sub>) = exp(<i>s</i>·[1 − √(28/<i>t</i><sub>e</sub>)])'
                '<span><i>s</i> = 0,20 (R) · 0,25 (N) · 0,38 (S)</span></div>'
                '<div><i>t</i><sub>e</sub> = Σ Δ<i>t</i>·exp(−<i>E</i><sub>a</sub>/<i>R</i>·[1/(273,15 + <i>T</i>) − 1/293,15])'
                '<span><i>E</i><sub>a</sub>/<i>R</i> = 4 000 K (annexe B, éq. B.10)</span></div>'
                '<div class="dom">Domaine de validité : 3 j &lt; <i>t</i><sub>e</sub> &lt; 28 j (§3.1.2(5)) ; sous 3 jours, la résistance '
                'se détermine par essais.</div></div>')


def _fenetre_age(r: Dict, nom: str) -> str:
    a, m = r["age"], r["materiau"]
    lignes = [{"nom": f"Résistance à {a['t']} j", "sous": f"t<sub>e</sub> = {_j(a['t_e'], 2)} j", "formule": "f<sub>ck</sub>(t) = β<sub>cc</sub>(t<sub>e</sub>)·f<sub>cm</sub> − 8",
               "valeur": f"{_mpa(a['fck'], 2)} MPa", "limite": f"f<sub>ck</sub> = {m['fck28']} MPa", "taux": C.pct(max(a["pct"], 0.0) / 100),
               "niveau": _niv_part(a["pct"] / 100), "gov": True}]
    if a["defav"]:
        d = a["defav"]
        lignes.append({"nom": f"Défavorable à {a['t']} j", "sous": f"prévisions − {C.nb(d['marge'])} °C · t<sub>e</sub> = {_j(d['t_e'], 2)} j",
                       "formule": "jours prévus et prolongés abaissés de 3 °C", "valeur": f"{_mpa(d['fck'], 2)} MPa",
                       "limite": f"f<sub>ck</sub> = {m['fck28']} MPa", "taux": C.pct(max(d["pct"], 0.0) / 100), "niveau": _niv_part(d["pct"] / 100)})
    fx = _formules_age(r)
    n_te = next((i for i, x in enumerate(fx) if x["parts"][0] == "t_e"), 0) + 1
    detail = _groupe("Âge équivalent", fx[:n_te]) + _groupe("Résistance à l'âge choisi", fx[n_te:])
    if a["defav"]:
        d = a["defav"]
        detail += _groupe("Variante défavorable", [
            _fx(["t_e,déf", f"Σ jour par jour, jours prévus à T − {C.nb(d['marge'])} °C", f"{d['n_prev'] + d['n_extra']} jour(s) abaissé(s)"], f"{_v(d['t_e'], 3)} j"),
            _fx(["f_ck,déf(t)", "β_cc(t_e,déf)·f_cm − 8"], f"{_mpa(d['fck'], 2)} MPa")])
    sous = "EN 1992-1-1 §3.1.2 (éq. 3.1, 3.2) · annexe B (éq. B.10) · maturité"
    pied = f'<div class="ck-sub">Les formules de l\'EC2</div>{FORMULES_EC2}'
    return C.fenetre_calcul("age", f"{C.esc(nom)} · résistance à {a['t']} j", sous, _hypotheses(r), lignes, detail, pied)


def _fenetre_seuil(r: Dict, nom: str) -> str:
    a, sr, m, tp = r["age"], r["seuil"], r["materiau"], r["temperature"]
    quand = (f"atteint après {_j(sr['t'], 2)} j" + (f" · le {_date_fr(sr['date'])}" if sr["date"] else "")) if sr["t"] is not None else "non atteignable"
    lignes = [{"nom": f"Seuil à {a['t']} j", "sous": quand, "formule": "f<sub>s</sub> ≤ f<sub>ck</sub>(t)",
               "valeur": f"f<sub>s</sub> = {_mpa(sr['cible'], 2)} MPa", "limite": f"f<sub>ck</sub>({a['t']} j) = {_mpa(a['fck'], 2)} MPa",
               "taux": _taux_aff(sr["taux_age"]), "niveau": "r" if sr["taux_age"] is None else C.niveau(sr["taux_age"]), "gov": True}]
    fx = []
    for f in sr["formules"]:
        t = {x["symbole"]: x["valeur"] for x in f["termes"]}
        k, res = f["cle"], f["resultat"]
        if k == "cible_pct":
            fx.append(_fx(["f_s", "p·f_ck / 100", f"{_v(t['p'])} × {_v(t['fck'])} / 100"], f"{_mpa(res, 2)} MPa"))
        elif k == "t_e_cible":
            fx.append(_fx(["t_e,s", "28 / (1 − ln((f_s + 8) / f_cm) / s)²", f"28 / (1 − ln(({_v(t['cible'])} + 8) / {_v(t['fcm'])}) / {_v(t['s'], 2)})²"],
                          f"{_v(res, 3)} j"))
        elif k == "t_cible":
            fx.append(_fx(["t_s", "t_e,s / k_T", f"{_v(t['t_e'], 3)} / {_v(t['facteur'], 4)}"], f"{_v(res, 3)} j"))
        elif k == "t_seuil":
            fx.append(_fx(["t_s (module)", "dichotomie sur [0,05 ; 90] j, arrêt à 0,001 MPa près"], f"{_v(res, 3)} j"))
        elif k == "fck_seuil":
            fx.append(_fx(["f_ck(t_s)", "β_cc(t_e,s)·f_cm − 8"], f"{_mpa(res, 3)} MPa"))
    if sr["mode"] == "mpa":
        fx.insert(0, _fx(["f_s", "saisie"], f"{_mpa(sr['cible'], 2)} MPa"))
    comp = [_fx(["f_s / f_ck(t)", f"{_mpa(sr['cible'], 2)} / {_mpa(a['fck'], 2)}"], f"{C.pc(_taux_aff(sr['taux_age']))} %",
                "r" if sr["taux_age"] is None else C.niveau(sr["taux_age"]))]
    sv = sr.get("releve")
    if sv and sv["jour_annonce"] is not None:
        # le relevé jour par jour (les valeurs du tableau, celles de ①) : ce que le seuil donne vraiment
        jour = (f"jour {sv['jour']}" + (f" · le {_date_fr(sv['date'])}" if sv.get("date") else "")) if sv["jour"] is not None else "pas dans le relevé"
        comp += [_fx([f"f_ck,relevé(jour {sv['jour_annonce']})", "Σ jour par jour, comme ①"], f"{_mpa(sv['fck_jour_annonce'], 2)} MPa",
                     "r" if sv["optimiste"] else ""),
                 _fx(["premier jour où f_ck,relevé ≥ f_s"], jour, "r" if sv["optimiste"] else "")]
    detail = _groupe("Âge d'obtention du seuil", fx) + _groupe(f"Le seuil à l'âge choisi ({a['t']} j)" + (" et dans le relevé" if sv else ""), comp)
    note = (f"Comme l'écran Streamlit, le seuil se cherche à température CONSTANTE : {_deg(tp['T']) if tp['source'] != 'releve' else _mpa(tp['T'])} °C"
            + (" (la moyenne du relevé)." if tp["source"] == "releve" else ".")
            + " La date d'obtention ajoute les jours entiers au coulage (la fraction de jour est ignorée).")
    return C.fenetre_calcul("seuil", f"{C.esc(nom)} · seuil de résistance", f"seuil {_cible_txt(sr, m['fck28'])} · {quand}", _hypotheses(r), lignes, detail,
                            f'<div class="ck-autres">{note}</div>')


def _fenetre_optim(r: Dict, nom: str) -> str:
    o, m = r["optim"], r["materiau"]
    lignes = []
    for cl in o["classes"]:
        marque = " · ★ la plus petite qui tient" if cl["classe"] == o["classe"] else ""
        lignes.append({"nom": cl["classe"], "sous": f"f<sub>ck</sub> = {cl['fck28']} MPa{' · la classe du coulage' if cl['classe'] == m['classe'] else ''}{marque}",
                       "formule": "f<sub>c</sub> ≤ f<sub>ck</sub>(t<sub>d</sub>)", "valeur": f"{C.nb(o['mpa'])} MPa",
                       "limite": f"{_mpa(cl['fck'], 2)} MPa", "taux": _taux_classe(cl), "niveau": _niv_classe(cl), "gov": cl["classe"] == m["classe"]})
    fx = []
    for f in o["formules"]:
        t = {x["symbole"]: x["valeur"] for x in f["termes"]}
        if f["cle"] == "t_e_delai":
            fx.append(_fx(["t_e", "t_d·k_T", f"{_v(t['delai'])} × {_v(t['facteur'], 4)}"], f"{_v(f['resultat'], 3)} j"))
        elif f["cle"] == "beta":
            fx.append(_fx(["β_cc(t_e)", "exp(s·[1 − √(28/t_e)])", f"exp({_v(t['s'], 2)} × [1 − √(28/{_v(t['t_e'], 3)})])"], _v(f["resultat"], 4)))
    par = []
    for cl in o["classes"]:
        if o.get("beta") is not None:
            par.append(_fx([f"f_ck,{cl['classe'].split('/')[0]}(t_d)", f"{_v(o['beta'], 4)} × ({cl['fck28']} + 8) − 8"], f"{_mpa(cl['fck'], 2)} MPa",
                           _niv_classe(cl)))
        else:
            par.append(_fx([f"f_ck,{cl['classe'].split('/')[0]}(t_d)", "f_ck (t_e ≥ 28 j)"], f"{_mpa(cl['fck'], 2)} MPa", _niv_classe(cl)))
    detail = _groupe("Âge équivalent au délai", fx) + _groupe("Chaque classe au délai", par)
    res = f"→ {o['classe']}" + (" (la classe du coulage suffit)" if o["actuelle"] else "") if o["classe"] else "→ aucune classe de C20/25 à C50/60"
    pied = ('<div class="ck-autres">La première classe, de C20/25 à C50/60, dont f<sub>ck</sub>(t<sub>d</sub>) + 10<sup>−6</sup> ≥ f<sub>c</sub> '
            '(le critère de l\'écran Streamlit), avec le ciment du coulage et à température constante (la moyenne en mode relevé).</div>')
    return C.fenetre_calcul("optim", f"{C.esc(nom)} · classe pour {C.nb(o['mpa'])} MPa en {C.nb(o['delai'])} j", res, _hypotheses(r), lignes, detail, pied)


def _fenetre_essai(r: Dict, nom: str) -> str:
    es, m = r["essai"], r["materiau"]
    verdicts = {"coherent": f"cohérent avec {m['classe']}", "dessus": f"au-dessus de {m['classe']}", "sous": f"sous {m['classe']}"}
    lignes = [{"nom": "Classe estimée", "sous": f"≈ {es['classe']} · {verdicts[es['verdict']]}", "formule": "f<sub>ck,28,est</sub> comparé à f<sub>ck</sub>",
               "valeur": f"{_mpa(es['fck28_est'], 1)} MPa", "limite": f"f<sub>ck</sub> = {m['fck28']} MPa",
               "taux": C.pct(min(max(es["fck28_est"], 0.0) / m["fck28"], 9.99)), "niveau": "c", "gov": True}]
    fx = []
    for f in es["formules"]:
        t = {x["symbole"]: x["valeur"] for x in f["termes"]}
        k, res = f["cle"], f["resultat"]
        if k == "facteur":
            fx.append(_fx(["k_T", "exp(−E_a/R·[1/T_abs − 1/293,15])", f"exp(−{_v(t['E_a/R'], 1)} × [1/{_v(t['T_abs'], 2)} − 1/293,15])"], _v(res, 4)))
        elif k == "t_e":
            fx.append(_fx(["t_e", "t·k_T", f"{_v(t['t'])} × {_v(t['facteur'], 4)}"], f"{_v(res, 3)} j"))
        elif k == "beta":
            fx.append(_fx(["β_cc(t_e)", "exp(s·[1 − √(28/t_e)])", f"exp({_v(t['s'], 2)} × [1 − √(28/{_v(t['t_e'], 3)})])"], _v(res, 4)))
        elif k == "fck28_est":
            fx.append(_fx(["f_ck,28,est", "(f_mes + 8) / β_cc(t_e) − 8", f"({_v(t['f_mes'])} + 8) / {_v(t['beta'], 4)} − 8"], f"{_mpa(res, 2)} MPa"))
        elif k == "fck28_est_28":
            fx.append(_fx(["f_ck,28,est", "f_mes (t_e ≥ 28 j)", _v(t["f_mes"])], f"{_mpa(res, 2)} MPa"))
        elif k == "ecart_classe":
            fx.append(_fx([f"écart à {es['classe']}", "(f_ck,28,est − f_ck) / f_ck", f"({_mpa(t['fck28_est'], 2)} − {_v(t['fck'])}) / {_v(t['fck'])}"],
                          f"{res:+.1f} %".replace(".", ",").replace("-", "−")))
        elif k == "ecart":
            fx.append(_fx([f"écart à {m['classe']}", "(f_ck,28,est − f_ck) / f_ck", f"({_mpa(t['fck28_est'], 2)} − {_v(t['fck'])}) / {_v(t['fck'])}"],
                          f"{res:+.1f} %".replace(".", ",").replace("-", "−")))
    detail = _groupe("Âge équivalent de l'éprouvette", fx[:2]) + _groupe("Résistance à 28 jours estimée", fx[2:])
    pied = ('<div class="ck-autres">La classe estimée est la plus PROCHE de f<sub>ck,28,est</sub> (à égalité, la plus faible) ; '
            'l\'essai est « cohérent » si l\'écart à la classe du coulage reste dans ± 8 %, « sous » au-delà de − 8 % (l\'écran Streamlit).</div>'
            f'<div class="ck-autres"><b>À trancher par le bureau :</b> comme l\'écran Streamlit, la mesure est lue comme une résistance '
            f'CARACTÉRISTIQUE f<sub>ck</sub>(t). Une mesure sur éprouvette est plutôt une résistance moyenne (f<sub>cm</sub>(t) = '
            f'β<sub>cc</sub>·f<sub>cm</sub>, EC2 éq. 3.1) ; lue ainsi, f<sub>ck,28</sub> = f<sub>mes</sub>/β<sub>cc</sub> − 8 = '
            f'{_mpa(es["fck28_est_moyenne"], 1)} MPa, soit {_mpa(es["fck28_est"] - es["fck28_est_moyenne"], 1)} MPa de moins.</div>')
    return C.fenetre_calcul("essai", f"{C.esc(nom)} · essai {C.nb(es['mpa'])} MPa à {C.nb(es['age'])} j", f"cure à {_deg(es['T'])} °C",
                            _hypotheses(r), lignes, detail, pied)


def _vue_tableau(r: Dict) -> str:
    """Les valeurs que la courbe trace, jour par jour (le moteur : ``tableau``) — un tableau HTML posé dans la feuille."""
    tb = r["tableau"]
    rel = tb["releve"]
    hor = tb.get("horaire")
    tete = '<th>Âge<small>j</small></th>'
    if rel:
        tete += '<th>Date</th><th>T<small>°C</small></th><th class="g">Source</th>'
    if hor:
        tete += '<th>T min h<small>°C</small></th><th>T max h<small>°C</small></th>'
    tete += '<th>t<sub>e</sub><small>j</small></th><th>f<sub>ck</sub>(t)<small>MPa</small></th><th>f<sub>ck</sub>(t) / f<sub>ck</sub><small>%</small></th>'
    if rel:
        tete += '<th>à T<sub>moy</sub><small>MPa</small></th>'
    if tb["defav"]:
        tete += '<th>défavorable<small>MPa</small></th>'
    tete += "".join(f"<th>{C.esc(cl)}<small>MPa</small></th>" for cl in tb["classes"])
    sources = {"mesure": "mesuré", "prevision": "prévision", "prolonge": "prolongé"}
    corps = ""
    age = r["age"]["t"]
    for li in tb["lignes"]:
        dom = li["t_e"] <= 3.0
        cls = " ".join(x for x in ("on" if li["age"] == age else "", "dom" if dom else "") if x)
        tr = f'<tr class="{cls}">' if cls else "<tr>"
        tr += f'<th>{li["age"]}</th>'
        if rel:
            tr += f'<td class="d">{_date_fr(li["date"]) if li.get("date") else "—"}</td>'
            tr += f'<td>{_mpa(li["T_brut"]) if li["T_brut"] is not None else "—"}</td><td class="s {li["source"]}">{sources[li["source"]]}</td>'
        if hor:
            tr += f'<td>{_mpa(li["T_min_h"])}</td><td>{_mpa(li["T_max_h"])}</td>'
        tr += f'<td>{_j(li["t_e"], 2)}</td><td class="f">{_mpa(li["fck"], 2)}</td><td>{C.fr(li["pct"], 0).replace("-", "−")}</td>'
        if rel:
            tr += f'<td>{_mpa(li["fck_moyenne"], 2)}</td>'
        if tb["defav"]:
            tr += f'<td>{_mpa(li["fck_defav"], 2)}</td>'
        tr += "".join(f'<td>{_mpa(li["classes"][cl], 2)}</td>' for cl in tb["classes"])
        corps += tr + "</tr>"
    pied = ("Âges réels en jours ; t<sub>e</sub> : âge équivalent à 20 °C ; en gris, t<sub>e</sub> ≤ 3 j (hors du domaine de l'EC2) ; "
            "la ligne bleue : l'âge choisi ①.")
    if rel:
        pied += (" Avec un relevé : f<sub>ck</sub>(t) suit le relevé jour par jour (comme ①), « à T<sub>moy</sub> » est la valeur de la "
                 "courbe (la température moyenne, comme Streamlit).")
    if hor:
        pied += " T min h, T max h : les extrêmes horaires du jour (Open-Meteo, détail horaire) ; « — » : aucune valeur horaire."
    if tb["classes"]:
        pied += f" Les classes comparées ④ avec le ciment {C.esc(r['comparateur']['ciment'])}."
    return (f'<div class="age-vue"><div class="age-defile"><table class="age-tab"><thead><tr>{tete}</tr></thead><tbody>{corps}</tbody></table></div>'
            f'<div class="age-pied">{pied}</div></div>')


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : une consultation (pas de verdict) ; le coulage ouvert est nommé avec sa résistance."""
    i = etat["actif"]
    r = resultats[i]
    nom = etat["coulages"][i].get("nom") or f"Coulage {i + 1}"
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1],
            "gouvernant": f"{nom} · f_ck({r['age']['t']} j) = {_mpa(r['age']['fck'])} MPa · {r['materiau']['classe']}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (c, r) in enumerate(zip(etat["coulages"], resultats)):
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["coulages"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(c.get("nom") or f"Coulage {i + 1}")} '
                  f'<span>· {C.esc(c["classe"])} · {C.esc(_cure_courte(r))}</span></span><span class="pct">{_mpa(r["age"]["fck"])}{C.NBSP}MPa</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["coulages"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier le coulage ouvert">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter un coulage</div></div>')


def resultat(etat: Dict) -> Dict:
    c = _coulage(etat)
    r = M.calculer(_moteur(c))
    resultats = [r if i == etat["actif"] else M.calculer(_moteur(x)) for i, x in enumerate(etat["coulages"])]
    nom = c.get("nom") or f"Coulage {etat['actif'] + 1}"
    # les messages du moteur, échappés, les indices écrits en indice (t_e → t<sub>e</sub>, f_ck → f<sub>ck</sub>)
    messages = "".join(C.message(C.symb_math(a["message"]), a["code"] in GRAVES) for a in r["avertissements"]) + message_essai(r)
    tp, m = r["temperature"], r["materiau"]
    pied = (f'<div class="pfoot"><div class="mass"><span>EC2 §3.1.2 et annexe B</span>'
            f'<b>E<sub>a</sub>/R = {_v(tp["Ea_R"], 0)} K</b></div><div class="legend">{C.legende()}</div></div>')
    volets = _fenetre_age(r, nom) + _fenetre_seuil(r, nom) + _fenetre_optim(r, nom)
    if r["essai"]["actif"]:
        volets += _fenetre_essai(r, nom)
    titre_courbe = f"f_ck(t) · {m['classe']} · {_cure(r)}"
    # des graphiques, pas des dessins à l'échelle : « echelle » faux (la feuille n'écrit pas « mm · à l'échelle »)
    vues = [{"cle": "courbe", "titre": "Résistance au jeune âge", "court": "Courbe", "svg": DA.courbe("age_courbe", r), "legende": titre_courbe,
             "echelle": False}]
    if tp["source"] == "releve":
        rel, mt = tp["releve"], tp["meteo"]
        if mt:
            titre, leg = "Météo du lieu", f"{mt['label']} · Open-Meteo, {_quand(mt['recupere'])}" + (" · corrigée à la main" if mt["modifie"] else "")
        else:
            titre, leg = "Relevé des températures", tp["lieu"] or "relevé"
        vues.append({"cle": "releve", "titre": titre, "court": "Températures", "svg": DA.releve("age_releve", r), "echelle": False,
                     "legende": f"{leg} · {rel['n_jours']} j · T moy {_mpa(tp['T'])} °C · t_e {_j(rel['t_e_releve'])} j"})
    vues.append({"cle": "tableau", "titre": "Valeurs jour par jour", "court": "Tableau", "svg": _vue_tableau(r),
                 "legende": "âge, âge équivalent, résistance" + (" · classes comparées" if r["tableau"]["classes"] else "")})
    date = f" · coulé le {_date_fr(c['date_coulage'])}" if c["date_coulage"] else ""
    phrase = f"{C.esc(m['classe'])} · ciment {m['ciment']} (s = {C.nb(m['s'], 2)}) · {C.esc(_cure(r))}{date}"
    corps = (f'<div class="pbody">{_verdict(r, nom)}<div class="els">{lignes_resultat(r)}</div>{messages}</div>{pied}')
    return {"resultat": corps, "volets": volets, "vues": vues, "phrase": phrase, "statut_html": C.statut("i"),
            "resume": resume_note(etat, resultats), "donnees": formulaire(etat, r), "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de classe (③)
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les sept classes pour la cible de ③ : f_ck au délai, taux = cible / f_ck(délai), la retenue (✓, la classe du
    coulage) et l'optimale (★, la plus petite qui tient) — calculées par le moteur."""
    if role != "classe":
        raise ValueError(f"sélecteur inconnu : {role}")
    c = _coulage(etat)
    cands = M.candidats_classes(_moteur(c))
    o = c["optim"]
    i_o = next((k for k, x in enumerate(cands) if x["optimal"]), 0)
    i_c = next((k for k, x in enumerate(cands) if x["retenue"]), i_o)
    rows = [{"nom": x["classe"], "col": _mpa(x["fck"], 1), "taux": _taux_classe(x), "niveau": _niv_classe(x), "cur": x["retenue"], "opt": x["optimal"],
             "prop": (f"{x['classe']} · f<sub>ck</sub> = {x['fck28']} MPa · f<sub>ck</sub>({C.nb(o['delai'])} j) = {_mpa(x['fck'], 2)} MPa · "
                      + ("atteint la cible" if x["ok"] else "n'atteint pas la cible"))} for x in cands]
    return C.selecteur("classe", f"Classe de béton · {C.nb(o['mpa'])} MPa en {C.nb(o['delai'])} j",
                       "triées par résistance · un clic retient la classe", "", rows, max(0, min(i_o, i_c) - 4),
                       f"MPa à {C.nb(o['delai'])} j", etoile="la plus petite classe qui tient")

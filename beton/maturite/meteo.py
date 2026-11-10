# -*- coding: utf-8 -*-
"""La MÉTÉO DU LIEU d'un coulage (Open-Meteo), SANS RÉSEAU : le PLAN des appels et la LECTURE des réponses.

Ce module n'appelle jamais le service. C'est le navigateur qui le fait (site/web/js/meteo.js), à la demande de
l'ingénieur, vers les trois adresses d'Open-Meteo et elles seules (``ADRESSES`` ; la politique de sécurité du site,
site/web/_headers, n'en permet pas d'autre) : il rend les réponses JSON BRUTES, que ce module lit. La météo est donc une
IMPORTATION de données ; le calcul, lui, se fait hors ligne, sur les températures que la note garde.

La logique est celle de l'écran Streamlit ``modules/age_beton.py`` (v3.2), dont la LECTURE de la météo est gardée telle
quelle et prouvée par ``tests/test_maturite_moteur.py`` (l'écran rejoué avec un faux ``requests`` qui rend les MÊMES
réponses) :

- ``plages``          ← ``_plages_endpoints`` : l'archive (réanalyse, en retard de ~6 j) pour le passé, les prévisions
                        pour les jours récents ET futurs (jusqu'à 16 j), le reste PROLONGÉ ;
- ``plan``            ← les appels de ``meteo_journaliere`` / ``meteo_horaire`` / ``geocode_open_meteo`` : adresse,
                        dates, variables, fuseau — ce que le navigateur envoie, paramètre pour paramètre ;
- ``lire_geocodage``  ← ``geocode_open_meteo`` (le premier résultat ; nom, région, pays) ;
- ``lire_journalier`` ← ``meteo_journaliere`` (jours sans doublon, prévision = après le jour de la récupération,
                        prolongation à la moyenne des 5 derniers jours servis) ;
- ``lire_horaire``    ← ``meteo_horaire`` ; ``extremes_horaires`` ← les colonnes « T min h », « T max h » du tableau
                        « Données météo » ; ``minimales_apres`` : la plus basse valeur horaire de chaque jour, comptée à
                        partir d'une heure donnée le premier jour (l'alerte de gel du moteur) ;
- ``releve``          : les températures journalières écrites comme un relevé (``releve.py``) — la suite du calcul est
                        celle du relevé, que l'ingénieur peut ensuite corriger à la main ;
- ``enregistrement``, ``normaliser``, ``synthese`` : ce que la note GARDE d'une récupération (le lieu, les dates, le
                        jour et l'heure, les réponses réduites à ce qui est lu), relu strictement (un .stab vient
                        d'ailleurs), et ce que l'écran en dit.

Ce qui diffère de l'écran, et pourquoi : une réponse MAL FORMÉE (une température en texte, des dates qui ne sont pas des
dates, des listes de longueurs différentes, une date qui n'existe pas) est refusée avec un message, au lieu d'être lue
de travers ; une période de plus de 400 jours est refusée (le relevé n'en garde pas plus) ; chaque jour reçu doit être l'un
de ceux que l'appel a demandés, et les jours servis doivent être EXACTEMENT ceux du coulage au dernier jour servi (le
relevé est jour par jour, et son jour 1 est le coulage : une réponse qui commençait un jour plus tard, ou la veille du
coulage, décalait tous les âges sans un mot). Rien de ceci n'arrive avec une réponse conforme au format documenté
d'Open-Meteo, qui rend exactement les jours demandés.

Bibliothèque standard seule (la prolongation est une moyenne : somme / nombre — celle de l'écran, à 1e-12 près).
"""
import copy
import datetime as _dt
import math
import re

from . import releve as RL

# ------------------------------------------------------------------ les adresses et les constantes de l'écran
# Les trois adresses d'Open-Meteo (modules/age_beton.py : URL_FORECAST, URL_ARCHIVE, geocode_open_meteo). Copies NOMMÉES,
# liées par tests/test_site_web.py : site/web/js/meteo.js (ADRESSES) et la politique de sécurité (_headers, connect-src).
ADRESSES = {"geocodage": "https://geocoding-api.open-meteo.com/v1/search",
            "archive": "https://archive-api.open-meteo.com/v1/archive",
            "prevision": "https://api.open-meteo.com/v1/forecast"}
SOURCES = ("archive", "prevision")          # l'ordre des appels de l'écran : l'archive, puis les prévisions
RETARD_ARCHIVE_J = 6                        # l'archive est fiable jusqu'à ~J-6 (RETARD_ARCHIVE_J de l'écran)
HORIZON_PREVISION_J = 16                    # l'horizon de l'API de prévision d'Open-Meteo (cité dans les messages)
JOURNALIER = {"daily": "temperature_2m_mean"}
HORAIRE = {"hourly": "temperature_2m"}
GEOCODAGE = {"count": 1, "language": "fr"}  # geocode_open_meteo : le premier résultat, les noms en français
FUSEAU = "auto"
JOURS_PROLONGATION = 5                      # la prolongation : la moyenne des 5 derniers jours servis
LIEU_DEFAUT = "Bruxelles, Belgique"         # « Lieu du chantier » de l'écran au premier affichage
DEBUT_DEFAUT_J = 14                         # « Coulage » de l'écran sans date de coulage : aujourd'hui − 14 j
FIN_APRES_DEBUT_J = 5                       # « Fin (décoffrage) » de l'écran : max(aujourd'hui, coulage + 5 j)
MAX_JOURS = RL.MAX_JOURS                    # le relevé garde 400 jours au plus

_JOUR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HEURE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
_MAX_VALEURS_HORAIRES = 24 * (MAX_JOURS + 2)


def _date(x, quoi: str) -> _dt.date:
    if isinstance(x, _dt.date):
        return x
    if not isinstance(x, str) or not _JOUR.match(x):
        raise ValueError(f"{quoi} : une date AAAA-MM-JJ est attendue (reçu {str(x)[:40]!r})")
    try:
        return _dt.date.fromisoformat(x)
    except ValueError:
        raise ValueError(f"{quoi} : date impossible « {x} »") from None


def _fr(d: _dt.date) -> str:
    return d.strftime("%d/%m/%Y")


# ------------------------------------------------------------------ le plan des appels
def plages(debut: _dt.date, fin: _dt.date, aujourdhui: _dt.date):
    """Découpe [debut, fin] entre l'archive et l'API de prévision. Rend ``(plage_archive|None, plage_prevision|None,
    fin_dispo)``, ``fin_dispo`` étant le dernier jour servi par une API (le reste est prolongé) ; ``(None, None, None)``
    si la période est entièrement au-delà de l'horizon. Module : ``_plages_endpoints``, recopiée."""
    horizon = aujourdhui + _dt.timedelta(days=HORIZON_PREVISION_J)
    recente = aujourdhui - _dt.timedelta(days=RETARD_ARCHIVE_J)
    fin_dispo = min(fin, horizon)
    if fin_dispo < debut:
        return None, None, None
    if fin_dispo < recente:
        return (debut, fin_dispo), None, fin_dispo
    if debut >= recente:
        return None, (debut, fin_dispo), fin_dispo
    return (debut, recente - _dt.timedelta(days=1)), (recente, fin_dispo), fin_dispo


def dates_par_defaut(date_coulage, aujourdhui: _dt.date):
    """Les dates « Coulage » et « Fin (décoffrage) » de l'écran au premier affichage : le coulage = la date de coulage
    saisie, sinon aujourd'hui − 14 j ; la fin = max(aujourd'hui, coulage + 5 j)."""
    debut = _date(date_coulage, "date de coulage") if date_coulage else aujourdhui - _dt.timedelta(days=DEBUT_DEFAUT_J)
    return debut, max(aujourdhui, debut + _dt.timedelta(days=FIN_APRES_DEBUT_J))


def plan(lieu, debut, fin, aujourdhui, horaire: bool = False, position=None) -> dict:
    """Le PLAN d'une récupération : ce que le navigateur doit demander, et à qui. ``ValueError`` (message en français, à
    montrer tel quel) quand la récupération est impossible : pas de lieu, fin avant le coulage, période trop longue ou
    entièrement au-delà de l'horizon des prévisions (16 j).

    ``position`` ({"latitude", "longitude"}) : le lieu est déjà trouvé (le détail horaire) — pas de géocodage.

    Rend {"version", "nature": "journalier" | "horaire", "aujourdhui", "debut", "fin", "fin_dispo", "n_prolonges",
    "lieu" et "geocodage" (paramètres du géocodage) | "position", "appels": [{"source": "archive" | "prevision", "debut",
    "fin", "parametres": {"start_date", "end_date", "timezone", "daily" | "hourly"}}]} — les dates en AAAA-MM-JJ. Le
    navigateur ajoute la latitude et la longitude (``parametres``) et n'envoie rien d'autre."""
    debut, fin, aujourdhui = _date(debut, "coulage"), _date(fin, "fin"), _date(aujourdhui, "jour de la récupération")
    if position is None and (not isinstance(lieu, str) or not lieu.strip()):
        raise ValueError("Saisissez le lieu du chantier (ville, pays) avant de récupérer la météo.")
    if fin < debut:
        raise ValueError("La date de fin précède le coulage.")
    n = (fin - debut).days + 1
    if n > MAX_JOURS:
        raise ValueError(f"Période de {n} jours : le relevé en garde {MAX_JOURS} au plus — rapprochez la fin du coulage.")
    pa, pf, fin_dispo = plages(debut, fin, aujourdhui)
    if fin_dispo is None:
        raise ValueError(f"Période entièrement au-delà de l'horizon des prévisions ({HORIZON_PREVISION_J} j, jusqu'au "
                         f"{_fr(aujourdhui + _dt.timedelta(days=HORIZON_PREVISION_J))}) : rapprochez les dates, ou saisissez une "
                         "température constante ou un relevé.")
    variables = HORAIRE if horaire else JOURNALIER
    appels = []
    for source, pl in zip(SOURCES, (pa, pf)):
        if pl is None:
            continue
        appels.append({"source": source, "debut": pl[0].isoformat(), "fin": pl[1].isoformat(),
                       "parametres": {"start_date": pl[0].isoformat(), "end_date": pl[1].isoformat(), "timezone": FUSEAU, **variables}})
    p = {"version": 1, "nature": "horaire" if horaire else "journalier", "aujourdhui": aujourdhui.isoformat(),
         "debut": debut.isoformat(), "fin": fin.isoformat(), "fin_dispo": fin_dispo.isoformat(), "n_prolonges": (fin - fin_dispo).days}
    if position is None:
        p["lieu"] = lieu
        p["geocodage"] = {"name": lieu, **GEOCODAGE}
    else:
        p["position"] = {"latitude": position["latitude"], "longitude": position["longitude"]}
    p["appels"] = appels
    return p


def parametres(appel: dict, position: dict) -> dict:
    """Les paramètres d'un appel tels que le navigateur les envoie : la latitude et la longitude du lieu, puis ceux du
    plan — ``_appel_meteo`` de l'écran (le test de parité compare ce dictionnaire à celui que l'écran passe à requests)."""
    return {"latitude": position["latitude"], "longitude": position["longitude"], **appel["parametres"]}


# ------------------------------------------------------------------ la lecture des réponses
def _nombre(v, quoi):
    """Une valeur de réponse : un nombre, ou null (jour sans donnée) ; un texte ou un booléen est refusé."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"réponse du service météo illisible : {quoi} « {str(v)[:20]} » n'est pas une température")
    return v


def _paires(rep, bloc: str, variable: str, motif, source: str):
    """Les couples (instant, valeur) d'une réponse, comme ``zip(bloc.get("time", []), bloc.get(variable, []))`` de
    l'écran ; contrôlés : listes de même longueur, instants au format attendu ET qui existent (un 30 février, une
    25e heure sont refusés en français — Python les refusait plus loin, en anglais), valeurs numériques ou null."""
    if not isinstance(rep, dict):
        raise ValueError(f"réponse du service météo illisible ({source})")
    b = rep.get(bloc, {})
    if not isinstance(b, dict):
        raise ValueError(f"réponse du service météo illisible ({source} : « {bloc} »)")
    t, v = b.get("time", []), b.get(variable, [])
    if not isinstance(t, list) or not isinstance(v, list) or len(t) != len(v):
        raise ValueError(f"réponse du service météo illisible ({source} : les dates et les températures ne se répondent pas)")
    if len(t) > _MAX_VALEURS_HORAIRES:
        raise ValueError(f"réponse du service météo trop longue ({source})")
    lire = _dt.date.fromisoformat if motif is _JOUR else _dt.datetime.fromisoformat
    for x in t:
        if not isinstance(x, str) or not motif.match(x):
            raise ValueError(f"réponse du service météo illisible ({source} : instant « {str(x)[:20]} »)")
        try:
            lire(x)
        except ValueError:
            raise ValueError(f"réponse du service météo illisible ({source} : instant impossible « {x} »)") from None
    return [(x, _nombre(y, x)) for x, y in zip(t, v)]


def _dans_l_appel(appel: dict, jour: str) -> None:
    """Un jour reçu doit être l'un de ceux que l'appel a DEMANDÉS : une réponse qui déborde (un jour avant le coulage,
    une autre année) décalerait le relevé — le jour 1 du calcul ne serait plus le coulage."""
    if not appel["debut"] <= jour <= appel["fin"]:
        raise ValueError(f"réponse du service météo hors des dates demandées ({appel['source']} : le "
                         f"{_fr(_dt.date.fromisoformat(jour))}, pour une demande du {_fr(_dt.date.fromisoformat(appel['debut']))} au "
                         f"{_fr(_dt.date.fromisoformat(appel['fin']))})")


def lire_geocodage(rep):
    """Le lieu trouvé : le premier résultat du géocodage, ou None si le lieu est introuvable. Module :
    ``geocode_open_meteo`` (label = « nom, région, pays », les morceaux absents omis)."""
    if not isinstance(rep, dict):
        raise ValueError("réponse du géocodage illisible")
    res = rep.get("results")
    if not res:
        return None
    g = res[0] if isinstance(res, list) else None
    if not isinstance(g, dict):
        raise ValueError("réponse du géocodage illisible")
    noms = []
    for k in ("name", "admin1", "country"):
        x = g.get(k)
        if x is not None and not isinstance(x, str):
            raise ValueError(f"réponse du géocodage illisible (« {k} »)")
        noms.append(x[:200] if x else x)
    lat, lon = g.get("latitude"), g.get("longitude")
    for x, borne, k in ((lat, 90, "latitude"), (lon, 180, "longitude")):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > borne:
            raise ValueError(f"réponse du géocodage illisible ({k})")
    return {"latitude": lat, "longitude": lon, "label": ", ".join(filter(None, noms)), "nom": noms[0], "region": noms[1], "pays": noms[2]}


def _appels(p: dict, reponses) -> list:
    if not isinstance(reponses, list) or len(reponses) != len(p["appels"]):
        raise ValueError(f"le service météo devait rendre {len(p['appels'])} réponse(s), il en a rendu "
                         f"{len(reponses) if isinstance(reponses, list) else 'aucune'}")
    return list(zip(p["appels"], reponses))


def lire_journalier(p: dict, reponses) -> dict:
    """Les températures moyennes JOURNALIÈRES sur [debut, fin] : les réponses de l'archive puis des prévisions, dans
    l'ordre du plan ; un jour déjà reçu n'est pas repris ; au-delà de l'horizon, la série est PROLONGÉE à la moyenne des
    5 derniers jours servis (drapeau ``d_extra``) ; ``d_prev`` : le jour est postérieur au jour de la récupération.
    Module : ``meteo_journaliere``, recopiée. ValueError si aucune température n'est exploitable pour prolonger, et
    (le site seul) si un jour reçu n'a pas été demandé ou si les jours servis ne vont pas exactement du coulage au dernier
    jour servi."""
    aujourdhui = _date(p["aujourdhui"], "jour de la récupération")
    debut, fin, fin_dispo = _date(p["debut"], "coulage"), _date(p["fin"], "fin"), _date(p["fin_dispo"], "fin servie")
    dates, temps = [], []
    for appel, rep in _appels(p, reponses):
        for d, t in _paires(rep, "daily", JOURNALIER["daily"], _JOUR, appel["source"]):
            _dans_l_appel(appel, d)
            if d not in dates:
                dates.append(d)
                temps.append(float(t) if t is not None else float("nan"))
    # les jours servis sont EXACTEMENT ceux du coulage au dernier jour servi, dans l'ordre : le jour 1 du relevé est le
    # coulage (une réponse qui commence plus tard décalait tous les âges, sans un mot)
    attendus = [(debut + _dt.timedelta(days=i)).isoformat() for i in range((fin_dispo - debut).days + 1)]
    if dates != attendus:
        manque = next((d for d in attendus if d not in dates), None)
        raise ValueError(f"réponse incomplète du service météo : le {_fr(_dt.date.fromisoformat(manque))} manque" if manque
                         else "réponse du service météo illisible (les jours ne se suivent pas)")
    prevision = [(_dt.date.fromisoformat(d) > aujourdhui) for d in dates]
    extrapole = [False] * len(dates)
    n_extra = (fin - fin_dispo).days
    if n_extra > 0:
        valides = [t for t in temps[-JOURS_PROLONGATION:] if not math.isnan(t)] or \
                  [t for t in temps if not math.isnan(t)]
        if not valides:
            raise ValueError("Aucune donnée météo exploitable sur la période.")
        t_prolonge = sum(valides) / len(valides)
        for i in range(1, n_extra + 1):
            dates.append((fin_dispo + _dt.timedelta(days=i)).isoformat())
            temps.append(t_prolonge)
            prevision.append(True)
            extrapole.append(True)
    return {"d_dates": dates, "d_temps": temps, "d_prev": prevision, "d_extra": extrapole, "n_extra": n_extra}


def lire_horaire(p: dict, reponses) -> dict:
    """Les températures HORAIRES (le détail de contrôle, chargé à la demande), tronquées à ce que les API couvrent ; un
    instant qui ne suit pas le précédent n'est pas repris. Module : ``meteo_horaire``, recopiée ; le site refuse en plus
    un instant d'un jour que l'appel n'a pas demandé."""
    h_times, h_temps = [], []
    for appel, rep in _appels(p, reponses):
        for ht, hv in _paires(rep, "hourly", HORAIRE["hourly"], _HEURE, appel["source"]):
            _dans_l_appel(appel, ht[:10])
            if not h_times or ht > h_times[-1]:
                h_times.append(ht)
                h_temps.append(float(hv) if hv is not None else float("nan"))
    return {"h_times": h_times, "h_temps": h_temps}


def extremes_horaires(h: dict) -> dict:
    """{jour AAAA-MM-JJ: [T min h, T max h]} : les valeurs horaires VALIDES de chaque jour (un jour sans aucune n'y est
    pas) — les colonnes « T min h », « T max h » du tableau « Données météo » de l'écran."""
    par_jour = {}
    for ht, hv in zip(h["h_times"], h["h_temps"]):
        par_jour.setdefault(ht[:10], []).append(hv)
    out = {}
    for d, vals in par_jour.items():
        v = [x for x in vals if not math.isnan(x)]
        if v:
            out[d] = [min(v), max(v)]
    return out


def minimales_apres(h: dict, instant: str = None) -> dict:
    """{jour AAAA-MM-JJ: la plus basse valeur horaire VALIDE du jour} ; ``instant`` (AAAA-MM-JJTHH:MM) : les heures
    d'AVANT cet instant ne comptent pas (le jour du bétonnage : un gel du petit matin, avant que le béton soit en place,
    n'est pas un gel du béton). Un jour sans aucune valeur comptée n'y est pas."""
    out = {}
    for ht, hv in zip(h["h_times"], h["h_temps"]):
        if math.isnan(hv) or (instant is not None and ht < instant):
            continue
        d = ht[:10]
        out[d] = hv if d not in out else min(out[d], hv)
    return out


def releve(j: dict) -> str:
    """Les températures journalières écrites comme un RELEVÉ (une ligne par jour : date;T;source — ``releve.ecrire``),
    relu à l'identique. Les jours doivent se suivre (un relevé est jour par jour) : une réponse trouée est refusée avec
    le jour qui manque, et une réponse vide aussi."""
    if not j["d_dates"]:
        raise ValueError("le service météo n'a rendu aucune température pour ces dates")
    for a, b in zip(j["d_dates"], j["d_dates"][1:]):
        if (_dt.date.fromisoformat(b) - _dt.date.fromisoformat(a)).days != 1:
            manque = _dt.date.fromisoformat(a) + _dt.timedelta(days=1)
            raise ValueError(f"réponse incomplète du service météo : le {_fr(manque)} manque")
    rel = {"dates": list(j["d_dates"]), "temps": list(j["d_temps"]), "prev": list(j["d_prev"]), "extra": list(j["d_extra"])}
    texte = RL.ecrire(rel)
    if not memes_jours(RL.lire(texte), rel):                        # ecrire / lire : l'aller-retour est exact
        raise ValueError("les températures reçues ne s'écrivent pas en relevé")
    return texte


def memes_jours(a: dict, b: dict) -> bool:
    """Deux relevés lus portent-ils les mêmes jours (dates, températures au bit près — un jour sans donnée égal à un
    jour sans donnée —, sources) ?"""
    if (a.get("dates") or None) != (b.get("dates") or None) or a["prev"] != b["prev"] or a["extra"] != b["extra"]:
        return False
    return len(a["temps"]) == len(b["temps"]) and all(x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(a["temps"], b["temps"]))


# ------------------------------------------------------------------ ce que la note garde
def _reduite(rep, bloc: str, variable: str) -> dict:
    """Une réponse réduite à ce qui est lu (les instants et les températures) : le reste (unités, altitude, durée du
    calcul chez Open-Meteo…) n'est pas gardé."""
    b = rep.get(bloc, {}) if isinstance(rep, dict) else {}
    b = b if isinstance(b, dict) else {}
    return {bloc: {"time": list(b.get("time", [])), variable: list(b.get(variable, []))}}


def _lieu_garde(geo_brut: dict) -> dict:
    g = geo_brut["results"][0]
    return {"name": g.get("name"), "admin1": g.get("admin1"), "country": g.get("country"),
            "latitude": g["latitude"], "longitude": g["longitude"]}


def enregistrement(p: dict, geocodage_brut: dict, reponses, recupere: str) -> dict:
    """Ce que la note GARDE d'une récupération journalière réussie : le lieu demandé et trouvé, les dates, le jour et
    l'heure de la récupération, les réponses réduites à ce qui est lu. La note recalcule ensuite hors ligne."""
    rec = {"lieu": p["lieu"], "debut": p["debut"], "fin": p["fin"], "aujourdhui": p["aujourdhui"], "recupere": recupere,
           "geocodage": _lieu_garde(geocodage_brut),
           "reponses": [_reduite(r, "daily", JOURNALIER["daily"]) for r in reponses], "horaire": None}
    return normaliser(rec)


def enregistrement_horaire(p: dict, reponses, recupere: str) -> dict:
    """Le détail horaire gardé (chargé à la demande) : le jour et l'heure, les réponses réduites."""
    return {"aujourdhui": p["aujourdhui"], "recupere": recupere,
            "reponses": [_reduite(r, "hourly", HORAIRE["hourly"]) for r in reponses]}


def position(rec: dict) -> dict:
    return {"latitude": rec["geocodage"]["latitude"], "longitude": rec["geocodage"]["longitude"]}


def plan_enregistre(rec: dict) -> dict:
    """Le plan d'une récupération gardée, refait à partir de ses dates et de son jour : les réponses s'y relisent."""
    return plan(rec["lieu"], rec["debut"], rec["fin"], rec["aujourdhui"])


def plan_horaire(rec: dict, aujourdhui) -> dict:
    """Le plan du détail horaire d'une récupération gardée : le même lieu (sa position, sans géocodage), les mêmes
    dates ; le partage archive / prévisions suit le jour de la demande (comme l'écran)."""
    return plan(None, rec["debut"], rec["fin"], aujourdhui, horaire=True, position=position(rec))


def _texte(x, quoi, maxi=200, vide=True):
    if x is None and vide:
        return None
    if not isinstance(x, str) or len(x) > maxi:
        raise ValueError(f"temperature/meteo : {quoi} illisible")
    return x


def _recupere(x, quoi):
    if not isinstance(x, str) or not _HEURE.match(x):
        raise ValueError(f"temperature/meteo : {quoi} illisible (AAAA-MM-JJTHH:MM attendu)")
    _dt.datetime.fromisoformat(x)
    return x


def _reponses_gardees(reps, n: int, bloc: str, variable: str, motif, quoi: str) -> list:
    if not isinstance(reps, list) or len(reps) != n:
        raise ValueError(f"temperature/meteo : {quoi} — {n} réponse(s) attendue(s)")
    out = []
    for r in reps:
        if not isinstance(r, dict) or set(r) != {bloc} or not isinstance(r[bloc], dict) or set(r[bloc]) != {"time", variable}:
            raise ValueError(f"temperature/meteo : {quoi} — réponse illisible")
        try:
            paires = _paires(r, bloc, variable, motif, quoi)
        except ValueError as e:
            raise ValueError(f"temperature/meteo : {e}") from None
        out.append({bloc: {"time": [x for x, _ in paires], variable: [None if y is None else float(y) for _, y in paires]}})
    return out


def normaliser(rec):
    """La récupération gardée dans une note, relue STRICTEMENT (un .stab vient d'ailleurs) : clés connues, textes et
    nombres bornés, dates valides, réponses au format lu et qui se RELISENT (``lire_journalier``). ValueError sinon.
    None : aucune récupération."""
    if rec is None:
        return None
    if not isinstance(rec, dict):
        raise ValueError("temperature/meteo : un objet est attendu")
    cles = {"lieu", "debut", "fin", "aujourdhui", "recupere", "geocodage", "reponses", "horaire"}
    if set(rec) != cles:
        raise ValueError(f"temperature/meteo : clés attendues {sorted(cles)}")
    out = {"lieu": _texte(rec["lieu"], "le lieu", vide=False), "recupere": _recupere(rec["recupere"], "la date de récupération")}
    for k in ("debut", "fin", "aujourdhui"):
        out[k] = _date(rec[k], f"temperature/meteo/{k}").isoformat()
    g = rec["geocodage"]
    if not isinstance(g, dict) or set(g) != {"name", "admin1", "country", "latitude", "longitude"}:
        raise ValueError("temperature/meteo : le lieu trouvé est illisible")
    lu = lire_geocodage({"results": [g]})
    out["geocodage"] = {"name": lu["nom"], "admin1": lu["region"], "country": lu["pays"], "latitude": float(lu["latitude"]),
                        "longitude": float(lu["longitude"])}
    try:
        p = plan_enregistre(out)
    except ValueError as e:
        raise ValueError(f"temperature/meteo : {e}") from None
    out["reponses"] = _reponses_gardees(rec["reponses"], len(p["appels"]), "daily", JOURNALIER["daily"], _JOUR, "réponses journalières")
    try:
        lire_journalier(p, out["reponses"])
    except ValueError as e:
        raise ValueError(f"temperature/meteo : {e}") from None
    h = rec["horaire"]
    if h is None:
        out["horaire"] = None
    else:
        if not isinstance(h, dict) or set(h) != {"aujourdhui", "recupere", "reponses"}:
            raise ValueError("temperature/meteo/horaire : clés attendues aujourdhui, recupere, reponses")
        auj = _date(h["aujourdhui"], "temperature/meteo/horaire/aujourdhui").isoformat()
        try:
            ph = plan_horaire(out, auj)
        except ValueError as e:
            raise ValueError(f"temperature/meteo/horaire : {e}") from None
        out["horaire"] = {"aujourdhui": auj, "recupere": _recupere(h["recupere"], "la date du détail horaire"),
                          "reponses": _reponses_gardees(h["reponses"], len(ph["appels"]), "hourly", HORAIRE["hourly"], _HEURE,
                                                        "réponses horaires")}
        try:
            lire_horaire(ph, out["horaire"]["reponses"])          # elles se relisent (dates demandées, comme à l'import)
        except ValueError as e:
            raise ValueError(f"temperature/meteo/horaire : {e}") from None
    return out


def synthese(rec: dict, releve_texte: str, lieu: str = None, debut=None, fin=None) -> dict:
    """Ce que l'écran dit d'une récupération gardée : le lieu trouvé (« nom, région, pays ») et sa position, les dates,
    le jour et l'heure, les jours reçus et prolongés ; ``modifie`` : le relevé de la note n'est plus celui que la météo
    a donné (corrigé à la main) ; ``demande_changee`` : le lieu ou les dates de la note ne sont plus ceux de la
    récupération ; ``horaire`` : {jour: [T min h, T max h]} quand le détail horaire est chargé."""
    p = plan_enregistre(rec)
    j = lire_journalier(p, rec["reponses"])
    try:
        modifie = not memes_jours(RL.lire(releve(j)), RL.lire(releve_texte))
    except ValueError:
        modifie = True
    geo = lire_geocodage({"results": [rec["geocodage"]]})
    out = {"lieu": rec["lieu"], "label": geo["label"], "nom": geo["nom"] or rec["lieu"], "latitude": geo["latitude"], "longitude": geo["longitude"],
           "debut": rec["debut"], "fin": rec["fin"], "aujourdhui": rec["aujourdhui"], "recupere": rec["recupere"],
           "n_jours": len(j["d_dates"]), "n_prolonges": j["n_extra"],
           "sources": [{"source": a["source"], "debut": a["debut"], "fin": a["fin"]} for a in p["appels"]],
           "modifie": modifie,
           "demande_changee": (lieu, debut, fin) != (rec["lieu"], rec["debut"], rec["fin"]) if lieu is not None else False,
           "horaire": None, "horaire_recupere": None}
    h = rec.get("horaire")
    if h:
        out["horaire"] = extremes_horaires(lire_horaire(plan_horaire(rec, h["aujourdhui"]), h["reponses"]))
        out["horaire_recupere"] = h["recupere"]
    return copy.deepcopy(out)

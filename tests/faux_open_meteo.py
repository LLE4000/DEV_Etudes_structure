# -*- coding: utf-8 -*-
"""Un FAUX Open-Meteo, au FORMAT DOCUMENTÉ du service. Le réseau de cette machine est fermé, et les tests n'appellent
JAMAIS le vrai service : ce faux rend, à une demande donnée, la réponse qu'Open-Meteo rendrait (même structure, mêmes
clés, mêmes types), avec des températures inventées mais déterministes.

Le format (documentation d'Open-Meteo : open-meteo.com/en/docs, /en/docs/historical-weather-api, /en/docs/geocoding-api) :

- géocodage   GET https://geocoding-api.open-meteo.com/v1/search?name=…&count=1&language=fr
              → {"results": [{"id", "name", "latitude", "longitude", "elevation", "feature_code", "country_code",
                 "admin1_id", "timezone", "population", "country_id", "country", "admin1"}], "generationtime_ms"} ;
              un lieu inconnu : la clé « results » est ABSENTE ;
- archive     GET https://archive-api.open-meteo.com/v1/archive?latitude&longitude&start_date&end_date&timezone=auto
  prévisions  GET https://api.open-meteo.com/v1/forecast?… (mêmes paramètres)
              &daily=temperature_2m_mean → {"latitude", "longitude", "generationtime_ms", "utc_offset_seconds",
                 "timezone", "timezone_abbreviation", "elevation", "daily_units": {"time": "iso8601",
                 "temperature_2m_mean": "°C"}, "daily": {"time": ["AAAA-MM-JJ", …], "temperature_2m_mean": [x | null, …]}}
              &hourly=temperature_2m → … "hourly_units", "hourly": {"time": ["AAAA-MM-JJTHH:MM", …], "temperature_2m": […]}
              (la latitude et la longitude rendues sont celles de la maille, arrondies) ;
- une demande invalide : HTTP 400, {"error": true, "reason": "…"} — les prévisions ne vont ni au-delà de J+16, ni avant
  J−92 (« past_days ») ; un excès de demandes : HTTP 429.

    FauxService(aujourdhui)          le service ; .reponse(url, params) → (code HTTP, JSON) ; .appels : le journal
    FauxRequests(service)            remplace ``requests`` dans l'écran Streamlit (modules/age_beton._req)
    recuperer(plan, service)         ce que le NAVIGATEUR (site/web/js/meteo.js) rendrait au module pour ce plan
"""
import datetime as _dt
import math
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beton.maturite import meteo as MT  # noqa: E402

LIEUX = {
    "bruxelles": {"id": 2800866, "name": "Bruxelles", "latitude": 50.85045, "longitude": 4.34878, "elevation": 28.0, "feature_code": "PPLC",
                  "country_code": "BE", "admin1_id": 2800867, "timezone": "Europe/Brussels", "population": 1019022, "country_id": 2802361,
                  "country": "Belgique", "admin1": "Bruxelles-Capitale"},
    "namur": {"id": 2790471, "name": "Namur", "latitude": 50.4669, "longitude": 4.86746, "elevation": 85.0, "feature_code": "PPLA2",
              "country_code": "BE", "admin1_id": 3337387, "timezone": "Europe/Brussels", "population": 110939, "country_id": 2802361,
              "country": "Belgique", "admin1": "Wallonie"},
    # un lieu sans région (admin1 absent) : le nom trouvé s'écrit « nom, pays »
    "vaduz": {"id": 3042030, "name": "Vaduz", "latitude": 47.14151, "longitude": 9.52154, "elevation": 455.0, "feature_code": "PPLC",
              "country_code": "LI", "timezone": "Europe/Vaduz", "population": 5197, "country_id": 3042058, "country": "Liechtenstein"},
}


def _cle(nom: str) -> str:
    n = unicodedata.normalize("NFD", str(nom).split(",")[0].strip().lower())
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def t_jour(d: _dt.date, lat: float = 50.85) -> float:
    """La température moyenne d'un jour (°C, au dixième, comme Open-Meteo) : déterministe, variée (gel compris)."""
    k = d.toordinal()
    return round(5.5 + 7.0 * math.sin(k / 11.0) + 0.3 * (k % 5) - (lat - 50.0), 1)


def t_heure(d: _dt.date, h: int, lat: float = 50.85) -> float:
    return round(t_jour(d, lat) + 3.5 * math.sin((h - 9) * math.pi / 12.0), 1)


class FauxService:
    """Le service : il répond à chaque demande comme Open-Meteo, et journalise ``(url, params)``. ``nulls`` : les jours
    (AAAA-MM-JJ) rendus « null » ; ``pannes`` : une file de réponses imposées (code, JSON) servies d'abord."""

    def __init__(self, aujourdhui, nulls=(), nulls_horaires=(), pannes=None):
        self.aujourdhui = aujourdhui if isinstance(aujourdhui, _dt.date) else _dt.date.fromisoformat(aujourdhui)
        self.nulls = set(nulls)
        self.nulls_horaires = set(nulls_horaires)
        self.pannes = list(pannes or [])
        self.appels = []

    def reponse(self, url, params):
        p = dict(params or {})
        self.appels.append((url, p))
        if self.pannes:
            return self.pannes.pop(0)
        if url == MT.ADRESSES["geocodage"]:
            g = LIEUX.get(_cle(p.get("name", "")))
            return 200, ({"results": [dict(g)], "generationtime_ms": 0.61} if g else {"generationtime_ms": 0.43})
        if url not in (MT.ADRESSES["archive"], MT.ADRESSES["prevision"]):
            return 404, {"error": True, "reason": "Not Found"}
        try:
            d0, d1 = _dt.date.fromisoformat(p["start_date"]), _dt.date.fromisoformat(p["end_date"])
            lat, lon = float(p["latitude"]), float(p["longitude"])
        except (KeyError, ValueError):
            return 400, {"error": True, "reason": "Parameter 'start_date', 'end_date', 'latitude' or 'longitude' missing or invalid"}
        if url == MT.ADRESSES["prevision"]:
            lo, hi = self.aujourdhui - _dt.timedelta(days=92), self.aujourdhui + _dt.timedelta(days=16)
            if d0 < lo or d1 > hi:
                return 400, {"error": True, "reason": f"Parameter 'start_date' is out of allowed range from {lo} to {hi}"}
        if d1 < d0:
            return 400, {"error": True, "reason": "End-date must be larger or equal than start-date"}
        jours = [d0 + _dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
        tete = {"latitude": round(lat * 20) / 20, "longitude": round(lon * 20) / 20, "generationtime_ms": 0.21,
                "utc_offset_seconds": 3600, "timezone": "Europe/Brussels", "timezone_abbreviation": "GMT+1", "elevation": 28.0}
        if "hourly" in p:
            tete["hourly_units"] = {"time": "iso8601", p["hourly"]: "°C"}
            tete["hourly"] = {"time": [f"{d.isoformat()}T{h:02d}:00" for d in jours for h in range(24)],
                              p["hourly"]: [None if d.isoformat() in self.nulls_horaires else t_heure(d, h, lat) for d in jours for h in range(24)]}
        else:
            tete["daily_units"] = {"time": "iso8601", p["daily"]: "°C"}
            tete["daily"] = {"time": [d.isoformat() for d in jours], p["daily"]: [None if d.isoformat() in self.nulls else t_jour(d, lat) for d in jours]}
        return 200, tete


class FauxReponse:
    def __init__(self, code, contenu):
        self.status_code = code
        self.ok = code < 400
        self._c = contenu

    def json(self):
        return self._c


class FauxRequests:
    """``requests`` pour l'écran Streamlit : ``get(url, params, timeout)`` interroge le faux service."""

    def __init__(self, service):
        self.service = service

    def get(self, url, params=None, timeout=None):
        return FauxReponse(*self.service.reponse(url, params))


def recuperer(plan: dict, service: FauxService) -> dict:
    """Ce que le navigateur (site/web/js/meteo.js) rendrait au module pour ce plan : le géocodage (sauf si le plan donne
    la position) puis chaque appel, dans l'ordre — {"lieu", "plan", "geocodage", "reponses"} ; ou {"erreur": …} quand le
    service refuse (le navigateur le dit lui-même et n'envoie rien)."""
    geo, pos = None, plan.get("position")
    if pos is None:
        code, geo = service.reponse(MT.ADRESSES["geocodage"], plan["geocodage"])
        if code != 200:
            return {"erreur": f"géocodage {code}"}
        r0 = (geo.get("results") or [None])[0]
        if not r0:
            return {"lieu": plan.get("lieu"), "plan": plan, "geocodage": geo, "reponses": []}
        pos = {"latitude": r0["latitude"], "longitude": r0["longitude"]}
    reps = []
    for a in plan["appels"]:
        code, rep = service.reponse(MT.ADRESSES[a["source"]], MT.parametres(a, pos))
        if code != 200:
            return {"erreur": f"{a['source']} {code} {rep.get('reason', '')}"}
        reps.append(rep)
    return {"lieu": plan.get("lieu"), "plan": plan, "geocodage": geo, "reponses": reps}

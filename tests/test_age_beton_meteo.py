# -*- coding: utf-8 -*-
"""Météo du module « Âge béton » — SANS RÉSEAU (Open-Meteo est simulé).

Ce que la suite verrouille :
  1. le ROUTAGE archive/prévision (_plages_endpoints) : l'archive ERA5 est en
     retard de ~6 j, les jours récents ET futurs viennent de l'API forecast ;
  2. la PROLONGATION au-delà de l'horizon de prévision (16 j) : moyenne des
     5 derniers jours prévus, jours étiquetés « prolongé » ;
  3. le re-essai sur 429 et le message d'erreur clair (quota PAR adresse IP,
     partagée sur l'hébergeur mutualisé) ;
  4. la variante DÉFAVORABLE (jours prévus − 3 °C, jours mesurés intacts) ;
  5. des ancres de calcul EC2 (non-régression) ;
  6. l'écran complet (AppTest) : récupération, cache (zéro requête au 2e
     clic), détail horaire À LA DEMANDE, courbe et caption défavorables.

Lancement : python3 tests/test_age_beton_meteo.py (depuis la racine).
"""
import math
import os
import sys
import types
import datetime as vrai_dt

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import numpy as np  # noqa: E402
import modules.age_beton as ab  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom
          + (f"   [{info}]" if info and not cond else ""))


def pres(a, b, tol=1e-6):
    return abs(a - b) <= tol


# =====================================================================
#  SIMULATEUR Open-Meteo
# =====================================================================
def t_jour(d: vrai_dt.date) -> float:
    """Température déterministe d'un jour donné (même formule côté mock et
    côté attendu)."""
    return 8.0 + (d.toordinal() % 7)


class FauxResp:
    def __init__(self, status, payload):
        self.status_code = status
        self.ok = status == 200
        self._p = payload

    def json(self):
        return self._p


class FauxReq:
    """Remplace ``requests`` : génère archive/prévision/géocodage/horaire à
    partir des paramètres, et journalise chaque appel."""

    def __init__(self, file_reponses=None, nulls=()):
        self.appels = []            # (url, params)
        self.file = list(file_reponses or [])   # réponses imposées (429…)
        self.nulls = set(nulls)     # dates ISO renvoyées « null » (archive)

    def get(self, url, params=None, timeout=None):
        p = dict(params or {})
        self.appels.append((url, p))
        if self.file:
            r = self.file.pop(0)
            if r is not None:
                return r
        if "geocoding" in url:
            return FauxResp(200, {"results": [{
                "latitude": 50.85, "longitude": 4.35, "name": "Bruxelles",
                "admin1": "Bruxelles-Capitale", "country": "Belgique"}]})
        d0 = vrai_dt.date.fromisoformat(p["start_date"])
        d1 = vrai_dt.date.fromisoformat(p["end_date"])
        jours = [d0 + vrai_dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
        if "hourly" in p:
            return FauxResp(200, {"hourly": {
                "time": [f"{d.isoformat()}T{h:02d}:00" for d in jours for h in range(24)],
                "temperature_2m": [t_jour(d) + (h - 12) * 0.1
                                   for d in jours for h in range(24)]}})
        return FauxResp(200, {"daily": {
            "time": [d.isoformat() for d in jours],
            "temperature_2m_mean": [None if d.isoformat() in self.nulls else t_jour(d)
                                    for d in jours]}})


class FauxTime:
    def __init__(self):
        self.sommeils = []

    def sleep(self, s):
        self.sommeils.append(s)


def date_figee(a: vrai_dt.date):
    """Un module datetime de substitution dont date.today() est figé."""
    class DateFixe(vrai_dt.date):
        @classmethod
        def today(cls):
            return cls(a.year, a.month, a.day)
    return types.SimpleNamespace(date=DateFixe, timedelta=vrai_dt.timedelta)


def vider_caches():
    ab.meteo_journaliere.clear()
    ab.meteo_horaire.clear()
    ab.geocode_open_meteo.clear()


_DT_REEL, _REQ_REEL, _TIME_REEL = ab._dt, ab._req if ab._HAS_REQUESTS else None, ab._time
AUJ = vrai_dt.date(2026, 10, 24)
D = vrai_dt.date


# =====================================================================
#  1. ROUTAGE archive / prévision
# =====================================================================
print("=== 1. Routage archive/prévision (_plages_endpoints) ===")
# aujourd'hui = 24/10 → archive fiable jusqu'au 17/10 inclus, horizon 09/11
pa, pf, fd = ab._plages_endpoints(D(2026, 9, 1), D(2026, 9, 30), AUJ)
chk("passé lointain → archive seule", pa == (D(2026, 9, 1), D(2026, 9, 30)) and pf is None)
chk("passé lointain → fin_dispo = fin", fd == D(2026, 9, 30))

pa, pf, fd = ab._plages_endpoints(D(2026, 10, 20), D(2026, 11, 5), AUJ)
chk("récent + futur → forecast seule", pa is None and pf == (D(2026, 10, 20), D(2026, 11, 5)))

pa, pf, fd = ab._plages_endpoints(D(2026, 10, 1), D(2026, 10, 24), AUJ)
chk("période à cheval → archive jusqu'au 17/10", pa == (D(2026, 10, 1), D(2026, 10, 17)))
chk("période à cheval → forecast 18/10 → 24/10", pf == (D(2026, 10, 18), D(2026, 10, 24)))

pa, pf, fd = ab._plages_endpoints(D(2026, 10, 29), D(2026, 11, 20), AUJ)
chk("fin au-delà de l'horizon → tronquée au 09/11",
    pa is None and pf == (D(2026, 10, 29), D(2026, 11, 9)) and fd == D(2026, 11, 9))

pa, pf, fd = ab._plages_endpoints(D(2026, 11, 15), D(2026, 11, 20), AUJ)
chk("période entièrement au-delà → (None, None, None)", (pa, pf, fd) == (None, None, None))

# =====================================================================
#  2. meteo_journaliere : fenêtres exactes, drapeaux, prolongation
# =====================================================================
print("=== 2. meteo_journaliere (mock) ===")
ab._dt = date_figee(AUJ)
ab._HAS_REQUESTS = True

# --- chantier en cours : coulage 01/10, contrôle 24/10 (à cheval) ---
vider_caches()
fake = FauxReq()
ab._req = fake
m = ab.meteo_journaliere(50.85, 4.35, "2026-10-01", "2026-10-24")
chk("24 jours retournés", len(m["d_dates"]) == 24, str(len(m["d_dates"])))
chk("aucun doublon de date", len(set(m["d_dates"])) == 24)
chk("aucun jour marqué prévision (fin = aujourd'hui)", not any(m["d_prev"]))
chk("aucun jour prolongé", m["n_extra"] == 0 and not any(m["d_extra"]))
appels_daily = [(u, p) for u, p in fake.appels if "daily" in p]
chk("deux appels journaliers (archive + forecast)", len(appels_daily) == 2)
chk("archive appelée sur 01/10 → 17/10",
    any("archive" in u and p["start_date"] == "2026-10-01" and p["end_date"] == "2026-10-17"
        for u, p in appels_daily))
chk("forecast appelée sur 18/10 → 24/10",
    any("archive" not in u and p["start_date"] == "2026-10-18" and p["end_date"] == "2026-10-24"
        for u, p in appels_daily))
chk("températures = celles servies",
    all(pres(t, t_jour(vrai_dt.date.fromisoformat(d))) for d, t in zip(m["d_dates"], m["d_temps"])))

# --- coulage FUTUR dans l'horizon : 29/10 + décoffrage 5 j (03/11) ---
vider_caches()
fake = FauxReq()
ab._req = fake
m = ab.meteo_journaliere(50.85, 4.35, "2026-10-29", "2026-11-03")
chk("coulage futur : 6 jours, tous en prévision",
    len(m["d_dates"]) == 6 and all(m["d_prev"]) and m["n_extra"] == 0)
chk("coulage futur : un seul appel (forecast)",
    len([1 for u, p in fake.appels if "daily" in p]) == 1
    and all("archive" not in u for u, p in fake.appels))

# --- fin au-delà de l'horizon : prolongation ---
vider_caches()
fake = FauxReq()
ab._req = fake
m = ab.meteo_journaliere(50.85, 4.35, "2026-10-29", "2026-11-15")
chk("prolongation : 18 jours au total", len(m["d_dates"]) == 18, str(len(m["d_dates"])))
chk("prolongation : n_extra = 6 (10/11 → 15/11)", m["n_extra"] == 6, str(m["n_extra"]))
chk("prolongation : les 6 derniers jours étiquetés",
    m["d_extra"][-6:] == [True] * 6 and not any(m["d_extra"][:-6]))
chk("prolongation : aussi marqués prévision", all(m["d_prev"][-6:]))
t_attendu = float(np.mean([t_jour(D(2026, 11, 5) + vrai_dt.timedelta(days=i)) for i in range(5)]))
chk("prolongation : T = moyenne des 5 derniers jours prévus (05→09/11)",
    all(pres(t, t_attendu) for t in m["d_temps"][-6:]),
    f"{m['d_temps'][-1]:.2f} vs {t_attendu:.2f}")
chk("prolongation : dates prolongées correctes",
    m["d_dates"][-1] == "2026-11-15" and m["d_dates"][11] == "2026-11-09")

# --- période entièrement au-delà de l'horizon → erreur claire ---
vider_caches()
ab._req = FauxReq()
try:
    ab.meteo_journaliere(50.85, 4.35, "2026-11-15", "2026-11-20")
    chk("entièrement au-delà → RuntimeError", False, "aucune exception")
except RuntimeError as e:
    chk("entièrement au-delà → RuntimeError", True)
    chk("… message cite l'horizon et le mode Manuel",
        "horizon" in str(e) and "Manuel" in str(e), str(e)[:80])
except Exception as e:  # noqa: BLE001
    chk("entièrement au-delà → RuntimeError", False, f"{type(e).__name__}: {e}"[:80])

# --- jour « null » dans l'archive → NaN (pas de décalage) ---
vider_caches()
ab._req = FauxReq(nulls={"2026-10-05"})
m = ab.meteo_journaliere(50.85, 4.35, "2026-10-01", "2026-10-10")
i5 = m["d_dates"].index("2026-10-05")
chk("jour sans donnée → NaN à sa place", math.isnan(m["d_temps"][i5]))
chk("les autres jours restent alignés",
    pres(m["d_temps"][i5 + 1], t_jour(D(2026, 10, 6))))

# =====================================================================
#  3. meteo_horaire : à la demande, tronquée à l'horizon
# =====================================================================
print("=== 3. meteo_horaire (mock) ===")
vider_caches()
fake = FauxReq()
ab._req = fake
h = ab.meteo_horaire(50.85, 4.35, "2026-10-29", "2026-11-15")
chk("horaire tronquée à l'horizon : 12 j × 24 h", len(h["h_times"]) == 12 * 24,
    str(len(h["h_times"])))
chk("horaire : dernier pas le 09/11", h["h_times"][-1].startswith("2026-11-09"))
chk("horaire : un appel hourly", len([1 for u, p in fake.appels if "hourly" in p]) == 1)

# =====================================================================
#  4. Re-essai 429 et message d'erreur
# =====================================================================
print("=== 4. Gestion du 429 (quota par adresse IP) ===")
ft = FauxTime()
ab._time = ft
vider_caches()
ab._req = FauxReq(file_reponses=[FauxResp(429, {}), None])  # 429 puis normal
m = ab.meteo_journaliere(50.85, 4.35, "2026-10-20", "2026-10-24")
chk("429 puis succès : données obtenues", len(m["d_dates"]) == 5)
chk("429 puis succès : un temps d'attente avant le 2e essai", ft.sommeils == [2.0],
    str(ft.sommeils))

vider_caches()
ab._req = FauxReq(file_reponses=[FauxResp(429, {}), FauxResp(429, {})])
try:
    ab.meteo_journaliere(50.85, 4.35, "2026-10-20", "2026-10-24")
    chk("429 persistant → RuntimeError", False, "aucune exception")
except RuntimeError as e:
    chk("429 persistant → RuntimeError", True)
    chk("… message cite l'adresse IP partagée et le mode Manuel",
        "adresse IP" in str(e) and "Manuel" in str(e), str(e)[:80])

vider_caches()
ab._req = FauxReq(file_reponses=[FauxResp(500, {"reason": "panne simulée"})])
try:
    ab.meteo_journaliere(50.85, 4.35, "2026-10-20", "2026-10-24")
    chk("erreur HTTP → RuntimeError avec code et raison", False, "aucune exception")
except RuntimeError as e:
    chk("erreur HTTP → RuntimeError avec code et raison",
        "500" in str(e) and "panne simulée" in str(e), str(e)[:80])
ab._time = _TIME_REEL

# =====================================================================
#  5. Variante défavorable et ancres de calcul
# =====================================================================
print("=== 5. Profil défavorable + ancres EC2 ===")
prof = ab.profil_defavorable([10.0, float("nan"), 12.0, 13.0],
                             [False, False, True, True])
chk("jour mesuré intact", pres(prof[0], 10.0))
chk("NaN laissé intact", math.isnan(prof[1]))
chk("jours prévus abaissés de 3 °C", pres(prof[2], 9.0) and pres(prof[3], 10.0))
chk("marge = 3 °C (constante)", pres(ab.MARGE_PREVISION_C, 3.0))

chk("maturity_factor(20 °C) = 1", pres(ab.maturity_factor(20.0), 1.0, 1e-12))
chk("fck(7 j éq., C30/37, s=0,25) ≈ 21,59 MPa",
    pres(float(ab.fck_of_age_equiv(30, 0.25, 7.0)), math.exp(-0.25) * 38.0 - 8.0, 1e-9))
chk("fck plafonné à fck28 dès t_e = 28 j",
    pres(float(ab.fck_of_age_equiv(30, 0.25, 28.0)), 30.0)
    and pres(float(ab.fck_of_age_equiv(30, 0.25, 40.0)), 30.0))
chk("age_equiv_from_series : 7 j à 20 °C → t_e = 7 j",
    pres(ab.age_equiv_from_series([20.0] * 7), 7.0, 1e-9))
chk("plus froid → t_e plus petit (défavorable bien du côté sûr)",
    ab.age_equiv_from_series([17.0] * 7) < ab.age_equiv_from_series([20.0] * 7))
chk("_age_equiv_point sur profil constant 20 °C : t_sel = t_e",
    pres(ab._age_equiv_point(3, 20.0, [20.0] * 7, ab.EA_EC2), 3.0, 1e-9))

# =====================================================================
#  6. Écran complet (AppTest, réseau simulé)
# =====================================================================
print("=== 6. Écran Âge béton (AppTest) ===")
ab._dt = _DT_REEL          # l'écran travaille avec la vraie date du jour
vider_caches()
fake = FauxReq()
ab._req = fake
ab._HAS_REQUESTS = True

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
at.session_state["page"] = "Age béton"
at.run()
chk("mode Manuel : pas d'exception", not at.exception,
    str(at.exception[0].message)[:120] if at.exception else "")
chk("mode Manuel : pas de variante défavorable",
    not any("Défavorable" in c.value for c in at.caption))

[r for r in at.radio if r.label == "Température"][0].set_value("Météo")
at.run()
chk("mode Météo : pas d'exception", not at.exception,
    str(at.exception[0].message)[:120] if at.exception else "")
chk("mode Météo sans clic : aucune requête réseau", len(fake.appels) == 0,
    str(len(fake.appels)))

# Coulage à venir : J+2 → J+20 (dépasse l'horizon de 4 j)
auj = vrai_dt.date.today()
coul, fin = auj + vrai_dt.timedelta(days=2), auj + vrai_dt.timedelta(days=20)
at.date_input(key="ab_m_coul").set_value(coul)
at.date_input(key="ab_m_ctrl").set_value(fin)
at.run()
[b for b in at.button if "Récupérer" in b.label][0].click()
at.run()
chk("récupération : pas d'exception", not at.exception,
    str(at.exception[0].message)[:120] if at.exception else "")
meteo = at.session_state["ab_meteo"] if "ab_meteo" in at.session_state else None
chk("récupération : 19 jours en session",
    meteo is not None and len(meteo["d_dates"]) == 19,
    str(len(meteo["d_dates"])) if meteo else "absent")
chk("récupération : 4 jours prolongés", meteo is not None and meteo["n_extra"] == 4,
    str(meteo.get("n_extra")) if meteo else "absent")
chk("avertissement de prolongation affiché",
    any("prolongés" in w.value for w in at.warning))
chk("caption défavorable affichée (fck et t_e)",
    any("Défavorable" in c.value and "fck" in c.value for c in at.caption))
chk("badge mesuré/prévision/prolongé",
    any("prévision" in c.value and "prolongés" in c.value for c in at.caption))
chk("rapport : variante défavorable renseignée",
    at.session_state["ab_rapport"]["defav"] is not None
    and at.session_state["ab_rapport"]["defav"]["n_extra"] == 4)
chk("aucune requête horaire tant qu'on ne la demande pas",
    not any("hourly" in p for u, p in fake.appels))
n_avant = len(fake.appels)
chk("au moins un appel réseau a eu lieu (géocodage + journalier)", n_avant >= 2,
    str(n_avant))

# Re-clic « Récupérer » : tout doit sortir du cache (zéro requête de plus)
[b for b in at.button if "Récupérer" in b.label][0].click()
at.run()
chk("2e clic : zéro requête supplémentaire (cache)", len(fake.appels) == n_avant,
    f"{len(fake.appels)} vs {n_avant}")

# Détail horaire à la demande
at.button(key="ab_btn_h").click()
at.run()
chk("détail horaire : pas d'exception", not at.exception,
    str(at.exception[0].message)[:120] if at.exception else "")
chk("détail horaire chargé en session",
    "ab_meteo_h" in at.session_state and len(at.session_state["ab_meteo_h"]["h_times"]) > 0)
chk("détail horaire : exactement une requête hourly",
    len([1 for u, p in fake.appels if "hourly" in p]) == 1)

# Coulage passé « classique » (14 j) : pas de défavorable
vider_caches()
fake2 = FauxReq()
ab._req = fake2
at2 = AppTest.from_file(os.path.join(RACINE, "streamlit_app.py"), default_timeout=180)
at2.session_state["page"] = "Age béton"
at2.run()
[r for r in at2.radio if r.label == "Température"][0].set_value("Météo")
at2.run()
[b for b in at2.button if "Récupérer" in b.label][0].click()
at2.run()
chk("période passée : pas d'exception", not at2.exception,
    str(at2.exception[0].message)[:120] if at2.exception else "")
m2 = at2.session_state["ab_meteo"] if "ab_meteo" in at2.session_state else None
chk("période passée : 15 jours, rien de prolongé",
    m2 is not None and len(m2["d_dates"]) == 15 and m2["n_extra"] == 0,
    str(len(m2["d_dates"])) if m2 else "absent")
chk("période passée : pas de variante défavorable",
    at2.session_state["ab_rapport"]["defav"] is None
    and not any("Défavorable" in c.value for c in at2.caption))

# Fin avant coulage : recalée automatiquement (état écrit avant le widget)
at2.date_input(key="ab_m_coul").set_value(auj + vrai_dt.timedelta(days=10))
at2.run()
chk("fin < coulage : la fin est recalée à coulage + 5 j",
    at2.session_state["ab_m_ctrl"] == auj + vrai_dt.timedelta(days=15),
    str(at2.session_state["ab_m_ctrl"]))
chk("dates modifiées depuis la récupération : avertissement",
    any("modifiés depuis la récupération" in c.value for c in at2.caption))

# Restauration (hygiène si la suite est importée)
ab._dt, ab._time = _DT_REEL, _TIME_REEL
if _REQ_REEL is not None:
    ab._req = _REQ_REEL

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)

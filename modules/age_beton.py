# =============================================================
#  age_beton.py — Évolution de la résistance du béton (EC2)
#  VERSION 3.2 — ergonomie chantier
#
#  Base normative :
#   - EN 1992-1-1 §3.1.2 (éq. 3.1, 3.2), annexe B (éq. B.10)
#   - s : 0,20 (R) / 0,25 (N) / 0,38 (S) — classe de résistance ciment
#   - Maturité : t_e = Σ Δt_i · exp(−Ea/R · (1/T − 1/293,15))
# =============================================================

import math
import time as _time
import datetime as _dt
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

try:
    import requests as _req
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

# =============================================================
#  STYLE
# =============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6", "info": "#eef2ff"}
C_ICONES   = {"ok": "✅", "warn": "⚠️", "nok": "❌", "info": "ℹ️"}


def _bloc(left: str, right: str = "", etat: str = "ok"):
    rh = f"<div style='font-weight:600;opacity:.9;white-space:nowrap;'>{right}</div>" if right else ""
    st.markdown(
        f"<div style='background:{C_COULEURS.get(etat,'#f6f6f6')};padding:10px 14px;border-radius:10px;"
        f"border:1px solid #d9d9d9;margin:6px 0 4px 0;display:flex;justify-content:space-between;"
        f"align-items:center;gap:10px;'>"
        f"<div style='font-weight:700;'>{left}</div>"
        f"<div style='display:flex;align-items:center;gap:10px;'>{rh}"
        f"<div style='font-size:20px;line-height:1;'>{C_ICONES.get(etat,'')}</div></div></div>",
        unsafe_allow_html=True,
    )


def _mini_metrics(items):
    cells = "".join(
        f"<div style='flex:1;text-align:center;padding:5px 4px;'>"
        f"<div style='font-size:11px;color:#6b7280;'>{lbl}</div>"
        f"<div style='font-size:14px;font-weight:700;color:#374151;'>{val}</div></div>"
        for lbl, val in items
    )
    st.markdown(
        f"<div style='display:flex;gap:6px;background:#f8fafc;border:1px solid #e5e7eb;"
        f"border-radius:10px;padding:4px 6px;margin:4px 0;'>{cells}</div>",
        unsafe_allow_html=True,
    )


# =============================================================
#  DONNÉES
# =============================================================
CLASSES = ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60"]

CIMENTS = {
    "Classe R — durcissement rapide": 0.20,
    "Classe N — durcissement normal": 0.25,
    "Classe S — durcissement lent":   0.38,
}
CIMENT_DETAIL = {
    "Classe R — durcissement rapide": "CEM 42,5R · CEM 52,5N · CEM 52,5R",
    "Classe N — durcissement normal": "CEM 32,5R · CEM 42,5N",
    "Classe S — durcissement lent":   "CEM 32,5N",
}

R_GAZ   = 8.314
EA_EC2  = 4000.0 * R_GAZ
T_REF_K = 293.15


# =============================================================
#  CALCULS (fonctions pures)
# =============================================================
def parse_fck(label: str) -> int:
    return int(label.split("/")[0].replace("C", ""))


def beta_cc(t_days_equiv, s: float):
    t = np.maximum(np.asarray(t_days_equiv, dtype=float), 1e-6)
    return np.exp(s * (1.0 - np.sqrt(28.0 / t)))


def fck_of_age_equiv(fck28: float, s: float, t_days_equiv):
    fcm = fck28 + 8.0
    t_e = np.asarray(t_days_equiv, dtype=float)
    val = beta_cc(t_e, s) * fcm - 8.0
    return np.where(t_e < 28.0, val, float(fck28))


def age_equiv(t_days_real, T_celsius: float, Ea: float = EA_EC2):
    T_abs = float(T_celsius) + 273.15
    factor = math.exp((-Ea / R_GAZ) * (1.0 / T_abs - 1.0 / T_REF_K))
    return np.asarray(t_days_real, dtype=float) * factor


def maturity_factor(T_celsius: float, Ea: float = EA_EC2) -> float:
    T_abs = float(T_celsius) + 273.15
    return math.exp((-Ea / R_GAZ) * (1.0 / T_abs - 1.0 / T_REF_K))


def age_equiv_from_series(temps_celsius, Ea: float = EA_EC2, dt_days: float = 1.0):
    arr = np.asarray(temps_celsius, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sum([maturity_factor(T, Ea) for T in arr]) * dt_days)


def fck_target_from_pct(fck28: float, pct: float) -> float:
    return fck28 * float(pct) / 100.0


def t_real_for_target(fck28: float, s: float, target_MPa: float,
                      T_celsius: float, Ea: float = EA_EC2,
                      tmax: float = 90.0, tol: float = 1e-3):
    if target_MPa <= 0 or target_MPa > fck28 + 1e-9:
        return None
    lo, hi = 0.05, float(tmax)
    f_hi = float(fck_of_age_equiv(fck28, s, age_equiv(hi, T_celsius, Ea)))
    if f_hi + 1e-6 < target_MPa:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = float(fck_of_age_equiv(fck28, s, age_equiv(mid, T_celsius, Ea)))
        if abs(f_mid - target_MPa) < tol:
            return mid
        if f_mid < target_MPa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def classe_min_pour_delai(target_MPa: float, s: float, T_celsius: float,
                          delai_jours: float, Ea: float = EA_EC2):
    t_e = float(age_equiv(delai_jours, T_celsius, Ea))
    for label in CLASSES:
        fck28 = parse_fck(label)
        fck_atteint = float(fck_of_age_equiv(fck28, s, t_e))
        if fck_atteint + 1e-6 >= target_MPa:
            return label, fck28, fck_atteint
    return None


def classe_estimee_depuis_mesure(res_mesuree: float, age_jours: float, s: float,
                                 T_celsius: float, Ea: float = EA_EC2):
    t_e = float(age_equiv(age_jours, T_celsius, Ea))
    if t_e >= 28.0:
        fck28_est = res_mesuree
    else:
        b = float(beta_cc(t_e, s))
        fck28_est = (res_mesuree + 8.0) / b - 8.0
    fcks = [parse_fck(c) for c in CLASSES]
    idx = int(np.argmin([abs(f - fck28_est) for f in fcks]))
    classe = CLASSES[idx]
    fck_classe = fcks[idx]
    ecart = (fck28_est - fck_classe) / fck_classe * 100.0 if fck_classe else 0.0
    return {"t_e": t_e, "fck28_est": fck28_est, "classe": classe,
            "fck_classe": fck_classe, "ecart_pct": ecart}


# =============================================================
#  MÉTÉO (Open-Meteo)
#
#  Deux endpoints, un seul point d'entrée :
#   - prévisions  : https://api.open-meteo.com/v1/forecast
#       couvre [aujourd'hui − 92 j … aujourd'hui + 16 j] — jours récents
#       MESURÉS (sans le retard ~5 j de l'archive) ET jours FUTURS ;
#   - archive     : https://archive-api.open-meteo.com/v1/archive
#       pour l'historique plus ancien (réanalyse, en retard de ~5 jours).
#  Le quota gratuit d'Open-Meteo est compté PAR ADRESSE IP (600/min,
#  5000/h, 10000/j) : sur un hébergeur mutualisé (Streamlit Cloud),
#  l'adresse de sortie est partagée entre beaucoup d'applications — d'où
#  des « Too Many Requests » sans rapport avec notre propre usage. Parade :
#  cache (30 min), re-essai unique sur 429, données horaires seulement à
#  la demande, et message clair qui renvoie vers le mode Manuel.
# =============================================================
HORIZON_PREVISION_J = 16        # horizon de l'API forecast
RETARD_ARCHIVE_J = 6            # l'archive est fiable jusqu'à ~J-6
MARGE_PREVISION_C = 3.0         # variante défavorable : °C retirés aux jours prévus


def _get_json(url, params):
    """GET avec un re-essai unique sur 429 et des messages d'erreur en
    français (le 429 « concurrent » passe souvent au 2e essai ; les 429 de
    quota viennent de l'adresse IP PARTAGÉE de l'hébergeur)."""
    if not _HAS_REQUESTS:
        raise RuntimeError("Module 'requests' indisponible.")
    r = _req.get(url, params=params, timeout=20)
    if r.status_code == 429:
        _time.sleep(2.0)
        r = _req.get(url, params=params, timeout=20)
    if r.status_code == 429:
        raise RuntimeError(
            "Service météo saturé (Open-Meteo limite par adresse IP, et l'hébergeur de "
            "l'application partage la sienne avec d'autres applis). Réessaie dans quelques "
            "minutes — ou passe en mode Manuel.")
    if not r.ok:
        try:
            raison = r.json().get("reason", "")
        except Exception:
            raison = ""
        raise RuntimeError(f"Service météo : erreur {r.status_code}"
                           + (f" — {raison}" if raison else "") + ".")
    return r.json()


def _plages_endpoints(debut: _dt.date, fin: _dt.date, aujourdhui: _dt.date):
    """Découpe [debut, fin] entre l'archive et l'API de prévision.
    Retourne ``(plage_archive|None, plage_forecast|None, fin_dispo)`` où
    ``fin_dispo`` = dernier jour servi par une API (le reste est prolongé)."""
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


def _appel_meteo(url, lat, lon, debut: _dt.date, fin: _dt.date, variables: dict):
    params = {"latitude": lat, "longitude": lon,
              "start_date": debut.isoformat(), "end_date": fin.isoformat(),
              "timezone": "auto", **variables}
    return _get_json(url, params)


URL_FORECAST = "https://api.open-meteo.com/v1/forecast"
URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@st.cache_data(ttl=1800, show_spinner=False)
def meteo_journaliere(lat, lon, debut_iso: str, fin_iso: str):
    """Températures moyennes JOURNALIÈRES sur [debut, fin] — passé par
    l'archive et/ou la prévision selon les dates ; au-delà de l'horizon de
    prévision, la série est PROLONGÉE à la moyenne des 5 derniers jours
    prévus (drapeau ``extrapole``). Chaque jour porte ``prevision`` (True
    s'il est postérieur à aujourd'hui). Léger : pas d'horaire ici."""
    debut = _dt.date.fromisoformat(debut_iso); fin = _dt.date.fromisoformat(fin_iso)
    aujourdhui = _dt.date.today()
    pa, pf, fin_dispo = _plages_endpoints(debut, fin, aujourdhui)
    if fin_dispo is None:
        raise RuntimeError(
            f"Période entièrement au-delà de l'horizon de prévision ({HORIZON_PREVISION_J} j) : "
            "saisis une température manuelle (mode Manuel).")
    dates, temps = [], []
    for url, plage in ((URL_ARCHIVE, pa), (URL_FORECAST, pf)):
        if plage is None:
            continue
        data = _appel_meteo(url, lat, lon, plage[0], plage[1],
                            {"daily": "temperature_2m_mean"})
        daily = data.get("daily", {})
        for d, t in zip(daily.get("time", []), daily.get("temperature_2m_mean", [])):
            if d not in dates:
                dates.append(d)
                temps.append(float(t) if t is not None else float("nan"))
    prevision = [(_dt.date.fromisoformat(d) > aujourdhui) for d in dates]
    extrapole = [False] * len(dates)
    n_extra = (fin - fin_dispo).days
    if n_extra > 0:
        valides = [t for t in temps[-5:] if not math.isnan(t)] or \
                  [t for t in temps if not math.isnan(t)]
        if not valides:
            raise RuntimeError("Aucune donnée météo exploitable sur la période.")
        t_prolonge = float(np.mean(valides))
        for i in range(1, n_extra + 1):
            dates.append((fin_dispo + _dt.timedelta(days=i)).isoformat())
            temps.append(t_prolonge)
            prevision.append(True)
            extrapole.append(True)
    return {"d_dates": dates, "d_temps": temps, "d_prev": prevision,
            "d_extra": extrapole, "n_extra": n_extra}


@st.cache_data(ttl=1800, show_spinner=False)
def meteo_horaire(lat, lon, debut_iso: str, fin_iso: str):
    """Températures HORAIRES (détail de contrôle, chargé À LA DEMANDE : une
    requête horaire « pèse » beaucoup plus qu'une journalière dans le quota
    d'Open-Meteo). Tronquée à ce que les API couvrent."""
    debut = _dt.date.fromisoformat(debut_iso); fin = _dt.date.fromisoformat(fin_iso)
    pa, pf, fin_dispo = _plages_endpoints(debut, fin, _dt.date.today())
    if fin_dispo is None:
        return {"h_times": [], "h_temps": []}
    h_times, h_temps = [], []
    for url, plage in ((URL_ARCHIVE, pa), (URL_FORECAST, pf)):
        if plage is None:
            continue
        data = _appel_meteo(url, lat, lon, plage[0], plage[1],
                            {"hourly": "temperature_2m"})
        hourly = data.get("hourly", {})
        for ht, hv in zip(hourly.get("time", []), hourly.get("temperature_2m", [])):
            if not h_times or ht > h_times[-1]:
                h_times.append(ht)
                h_temps.append(float(hv) if hv is not None else float("nan"))
    return {"h_times": h_times, "h_temps": h_temps}


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_open_meteo(nom_lieu: str):
    res = _get_json("https://geocoding-api.open-meteo.com/v1/search",
                    {"name": nom_lieu, "count": 1, "language": "fr"}).get("results")
    if not res:
        return None
    g = res[0]
    label = ", ".join(filter(None, [g.get("name"), g.get("admin1"), g.get("country")]))
    return g["latitude"], g["longitude"], label


def profil_defavorable(temps, prevision, marge=MARGE_PREVISION_C):
    """La variante DÉFAVORABLE du profil : les jours mesurés sont certains,
    les jours prévus (et prolongés) sont abaissés de ``marge`` °C — plus
    froid = durcissement plus lent = côté sécurité pour un décoffrage."""
    return [t - marge if (p and not math.isnan(t)) else t
            for t, p in zip(temps, prevision)]


# =============================================================
#  GRAPHIQUE
# =============================================================
COL_REF   = "#2e6fb0"
COL_DEFAV = "#b91c1c"
COL_SEUIL = "#b45309"
COL_OPT   = "#7c3aed"
COL_MES   = "#1f8a70"
COL_GRID  = "#e5e7eb"
COL_TXT   = "#374151"
# Palette pour le comparateur multi-classes
COL_CMP   = ["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad", "#16a085", "#d35400"]


def build_figure(t_real, ref, comparateur=None, point=None, t28_real=None,
                 seuil=None, opt=None, mesure_lab=None, defav=None, titre=""):
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    # Courbe de référence
    ax.plot(t_real, ref["y"], color=COL_REF, lw=2.4, label=ref["label"], zorder=3)
    ax.fill_between(t_real, ref["y"], ref["y"].min(), color=COL_REF, alpha=0.06, zorder=1)

    # Variante défavorable (jours prévus abaissés de la marge)
    if defav is not None and defav.get("y") is not None:
        ax.plot(t_real, defav["y"], color=COL_DEFAV, lw=1.6, ls="--",
                label=defav["label"], zorder=2.5)

    # Comparateur multi-classes
    if comparateur:
        for i, cmp in enumerate(comparateur):
            col = COL_CMP[i % len(COL_CMP)]
            ax.plot(t_real, cmp["y"], color=col, lw=1.6, ls="-", alpha=0.85,
                    label=cmp["label"], zorder=3)

    # Optimisation classe
    if opt is not None and opt.get("y") is not None:
        ax.plot(t_real, opt["y"], color=COL_OPT, lw=2.0, ls="-", alpha=0.9,
                label=opt["label"], zorder=3)
        if opt.get("t") is not None:
            ax.scatter([opt["t"]], [opt["fck"]], s=55, marker="^",
                       color=COL_OPT, edgecolor="white", lw=1.3, zorder=6)
            ax.plot([opt["t"], opt["t"]], [ref["y"].min(), opt["fck"]],
                    color=COL_OPT, lw=1.0, ls="--", alpha=0.5, zorder=2)

    # Repère t_e = 28 j
    if t28_real is not None and t_real.min() < t28_real < t_real.max():
        ax.axvline(t28_real, color="#9ca3af", lw=1.0, ls=":", zorder=2)
        ax.annotate("t_e = 28 j", xy=(t28_real, ax.get_ylim()[0]),
                    xytext=(t28_real + 0.4, ref["y"].min() + 1),
                    fontsize=8, color="#6b7280")

    # Point sélectionné
    if point is not None:
        ts, fs = point
        ax.plot([ts, ts], [ref["y"].min(), fs], color=COL_REF, lw=1.0, ls="--", alpha=0.55, zorder=2)
        ax.plot([t_real.min(), ts], [fs, fs], color=COL_REF, lw=1.0, ls="--", alpha=0.55, zorder=2)
        ax.scatter([ts], [fs], s=55, color=COL_REF, edgecolor="white", lw=1.4, zorder=5)
        ax.annotate(f"{fs:.1f} MPa à {ts:.0f} j",
                    xy=(ts, fs), xytext=(8, 10), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color=COL_REF,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=COL_REF, lw=0.8, alpha=0.9),
                    zorder=6)

    # Seuil (date d'obtention)
    if seuil is not None and seuil.get("t") is not None:
        fs_, ts_ = seuil["fck"], seuil["t"]
        ax.axhline(fs_, color=COL_SEUIL, lw=1.1, ls="--", alpha=0.7, zorder=2)
        ax.plot([ts_, ts_], [ref["y"].min(), fs_], color=COL_SEUIL, lw=1.1, ls="--", alpha=0.7, zorder=2)
        ax.scatter([ts_], [fs_], s=55, marker="s", color=COL_SEUIL,
                   edgecolor="white", lw=1.3, zorder=6,
                   label=f"Seuil : {fs_:.1f} MPa @ {ts_:.1f} j")

    # Essai laboratoire
    if mesure_lab is not None and mesure_lab.get("fck", 0) > 0:
        ax.scatter([mesure_lab["age"]], [mesure_lab["fck"]], s=70, marker="P",
                   color=COL_MES, edgecolor="white", lw=1.4, zorder=6,
                   label=f"Essai : {mesure_lab['fck']:.1f} MPa @ {mesure_lab['age']:.1f} j")

    # Habillage
    ax.set_xlabel("Âge du béton (jours réels)", fontsize=10, color=COL_TXT)
    ax.set_ylabel("fck(t)  [MPa]", fontsize=10, color=COL_TXT)
    ax.set_title(titre, fontsize=11, color=COL_TXT, pad=10)
    ax.grid(True, color=COL_GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d1d5db")
    ax.tick_params(colors=COL_TXT, labelsize=9)
    ax.margins(x=0.02)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9, edgecolor="#e5e7eb")
    fig.tight_layout()
    return fig


# =============================================================
#  EXPORT PDF (structure prête)
# =============================================================
def build_rapport_data(ctx: dict) -> dict:
    return {"module": "age_beton", "version": "3.2", **ctx}


# =============================================================
#  HELPERS UI
# =============================================================
def _age_equiv_point(t_sel, T, profil_temps, Ea):
    if profil_temps:
        n = min(int(round(t_sel)), len(profil_temps))
        t_e = age_equiv_from_series(profil_temps[:n], Ea)
        reste = t_sel - n
        if reste > 0 and n < len(profil_temps):
            t_e += maturity_factor(profil_temps[n], Ea) * reste
        return t_e
    return float(age_equiv(t_sel, T, Ea))


def _saisie_temperature(mode_temp, date_coulage, Ea):
    """Retourne ``(T_moyenne, profil_temps|None, source_temp, defav|None)``.

    ``defav`` n'existe qu'en mode Météo quand la période contient des jours
    PRÉVUS ou PROLONGÉS : c'est la variante défavorable (jours non mesurés
    abaissés de ``MARGE_PREVISION_C``), sous la forme
    ``{"T_moy", "profil", "n_prev", "n_extra", "marge"}``."""
    if mode_temp == "Manuel":
        T = st.number_input("T moyenne (°C)", value=20.0, step=1.0, format="%.1f",
                            min_value=-5.0, max_value=60.0,
                            help="Température de cure, constante.")
        return T, None, "manuelle", None

    # ---- Mode Météo ----
    if not _HAS_REQUESTS:
        st.warning("Module réseau indisponible. `pip install requests` pour activer.")
    cA, cB, cC = st.columns([1.4, 1, 1])
    with cA:
        lieu = st.text_input("Lieu du chantier", value="Bruxelles, Belgique")
    with cB:
        d_coul = st.date_input("Coulage",
                               value=date_coulage or _dt.date.today() - _dt.timedelta(days=14),
                               key="ab_m_coul",
                               help="Passé (chantier en cours) ou futur (coulage à venir).")
    with cC:
        # fin par défaut : aujourd'hui, ou coulage + 5 j pour un coulage futur ;
        # si le coulage saisi dépasse une fin déjà mémorisée, on la recale
        # (état écrit AVANT l'instanciation du widget — pas de value=).
        defaut_fin = max(_dt.date.today(), d_coul + _dt.timedelta(days=5))
        if st.session_state.get("ab_m_ctrl") is None or st.session_state["ab_m_ctrl"] < d_coul:
            st.session_state["ab_m_ctrl"] = defaut_fin
        d_ctrl = st.date_input("Fin (contrôle/décoffrage)", key="ab_m_ctrl",
                               help="Jusqu'où construire le profil de température.")

    if d_ctrl < d_coul:
        st.error("La date de fin précède le coulage.")
        return 20.0, None, "manuelle", None

    if st.button("🔄 Récupérer la météo", use_container_width=True):
        try:
            geo = geocode_open_meteo(lieu)
            if geo is None:
                st.error("Lieu introuvable.")
            else:
                lat, lon, label = geo
                data = dict(meteo_journaliere(lat, lon, d_coul.isoformat(), d_ctrl.isoformat()))
                data.update(label=label, lat=lat, lon=lon,
                            req=[lieu, d_coul.isoformat(), d_ctrl.isoformat()])
                st.session_state["ab_meteo"] = data
                st.session_state.pop("ab_meteo_h", None)
        except Exception as e:
            st.error(f"Échec : {e}")

    meteo = st.session_state.get("ab_meteo")
    if meteo and meteo.get("d_temps"):
        if meteo.get("req") and meteo["req"] != [lieu, d_coul.isoformat(), d_ctrl.isoformat()]:
            st.caption("⚠️ Lieu ou dates modifiés depuis la récupération — "
                       "clique à nouveau sur « Récupérer la météo ».")
        temps = meteo["d_temps"]
        prev = meteo.get("d_prev", [False] * len(temps))
        extra = meteo.get("d_extra", [False] * len(temps))
        valides = [t for t in temps if not math.isnan(t)]
        if not valides:
            st.error("Aucune température exploitable sur la période.")
            return 20.0, None, "manuelle", None

        # Profil journalier ALIGNÉ sur les dates (un jour sans donnée est
        # remplacé par la moyenne, pour ne pas décaler les jours suivants).
        T_rempl = float(np.mean(valides))
        profil = [t if not math.isnan(t) else T_rempl for t in temps]
        n_nan = len(temps) - len(valides)
        n_extra = sum(1 for x in extra if x)
        n_prev = sum(1 for p, x in zip(prev, extra) if p and not x)
        n_mes = len(temps) - n_prev - n_extra
        T_moy = float(np.mean(profil))
        T_min, T_max = float(min(valides)), float(max(valides))

        badge = f"**{meteo.get('label', '')}** — {n_mes} j mesurés"
        if n_prev:
            badge += f" · {n_prev} j de prévision"
        if n_extra:
            badge += f" · {n_extra} j prolongés"
        st.caption(badge)
        _mini_metrics([("T min", f"{T_min:.1f} °C"), ("T moy", f"{T_moy:.1f} °C"),
                       ("T max", f"{T_max:.1f} °C")])
        if n_extra:
            t_prol = next(t for t, x in zip(profil, extra) if x)
            st.warning(f"La fin de période dépasse l'horizon de prévision "
                       f"({HORIZON_PREVISION_J} j) : {n_extra} j prolongés à "
                       f"T ≈ {t_prol:.1f} °C (moyenne des 5 derniers jours prévus). "
                       "À recouper à l'approche du décoffrage.")
        if n_nan:
            st.caption(f"{n_nan} jour(s) sans donnée, remplacé(s) par la moyenne "
                       f"({T_rempl:.1f} °C).")

        # Variante défavorable : la météo prévue n'est pas acquise.
        defav = None
        if n_prev + n_extra > 0:
            profil_def = profil_defavorable(profil, prev)
            defav = {"T_moy": float(np.mean(profil_def)), "profil": profil_def,
                     "n_prev": n_prev, "n_extra": n_extra, "marge": MARGE_PREVISION_C}

        # ---- Tableau des données (détail horaire À LA DEMANDE) ----
        with st.expander("📊 Données météo", expanded=False):
            import pandas as pd
            meteo_h = st.session_state.get("ab_meteo_h")
            if meteo_h is None and meteo.get("lat") is not None and _HAS_REQUESTS:
                if st.button("Charger le détail horaire (optionnel)", key="ab_btn_h",
                             help="Chargé à part : une requête horaire pèse beaucoup "
                                  "plus qu'une journalière dans le quota d'Open-Meteo."):
                    try:
                        st.session_state["ab_meteo_h"] = meteo_horaire(
                            meteo["lat"], meteo["lon"],
                            d_coul.isoformat(), d_ctrl.isoformat())
                        meteo_h = st.session_state["ab_meteo_h"]
                    except Exception as e:
                        st.error(f"Échec : {e}")
            hourly_by_day = {}
            if meteo_h:
                for ht, hv in zip(meteo_h.get("h_times", []), meteo_h.get("h_temps", [])):
                    hourly_by_day.setdefault(ht[:10], []).append(hv)

            rows = []
            for d, tmoy, p, x in zip(meteo["d_dates"], temps, prev, extra):
                row = {"Date": d,
                       "T moy (°C)": f"{tmoy:.1f}" if not math.isnan(tmoy) else "—",
                       "Source": "prolongé" if x else ("prévision" if p else "mesuré")}
                h_valid = [v for v in hourly_by_day.get(d, []) if not math.isnan(v)]
                if h_valid:
                    row["T min h"] = f"{min(h_valid):.1f}"
                    row["T max h"] = f"{max(h_valid):.1f}"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=240)

            t_e_meteo = age_equiv_from_series(profil, Ea)
            items = [("Jours", f"{len(profil)}"), ("t_e", f"{t_e_meteo:.1f} j")]
            if defav:
                items.append(("t_e défavorable",
                              f"{age_equiv_from_series(defav['profil'], Ea):.1f} j"))
            _mini_metrics(items)

        return T_moy, profil, f"météo ({meteo.get('label', '')})", defav

    st.caption("Renseigne le lieu et les dates — passées (chantier en cours) ou à venir "
               f"(prévisions jusqu'à +{HORIZON_PREVISION_J} j) — puis récupère la météo. "
               "T = 20 °C par défaut.")
    return 20.0, None, "manuelle", None


# =============================================================
#  PAGE PRINCIPALE
# =============================================================
def show():
    st.markdown("## Évolution de la résistance du béton — EC2")
    st.caption("fck(t) selon EN 1992-1-1 §3.1.2, effet de la température par âge équivalent (annexe B).")

    # Layout 3/5 – 2/5
    col_g, col_d = st.columns([3, 2])

    with col_g:
        # ==============================================================
        #  BLOC UNIQUE : BÉTON DE RÉFÉRENCE + ANALYSES
        # ==============================================================
        with st.container(border=True):

            # ---- Titre + ⚙️ sur la même ligne ----
            tc1, tc2 = st.columns([6, 1])
            with tc1:
                st.markdown("#### Béton de référence")
            with tc2:
                show_adv = st.toggle("⚙️", value=False, key="ab_adv",
                                     help="Paramètres avancés (énergie d'activation)")

            # ---- Ligne 1 : classe · ciment ----
            r1a, r1b = st.columns([1, 1.6])
            with r1a:
                beton_label = st.selectbox("Classe béton", CLASSES, index=2)
                fck28 = parse_fck(beton_label)
            with r1b:
                ciment = st.selectbox("Classe ciment", list(CIMENTS.keys()), index=1,
                                      help="Fixe s dans l'EC2.")
                s_ref = CIMENTS[ciment]
            st.caption(f"{CIMENT_DETAIL[ciment]} — s = {s_ref:.2f}")

            # ---- Ligne 2 : date coulage · température ----
            r2a, r2b, r2c = st.columns([1, 0.8, 1.8])
            with r2a:
                date_coulage = st.date_input("Date de coulage", value=None,
                                             help="Sert à dater les seuils (décoffrage…).")
            with r2b:
                mode_temp = st.radio("Température", ["Manuel", "Météo"], horizontal=True)
            with r2c:
                Ea = EA_EC2
                T, profil_temps, source_temp, defav = _saisie_temperature(mode_temp, date_coulage, Ea)

            # ---- Paramètres avancés (inline, sans nouveau bloc) ----
            if show_adv:
                Ea_kJ = st.number_input("Ea [kJ/mol]", value=EA_EC2 / 1000.0, step=1.0, format="%.1f",
                                        min_value=20.0, max_value=60.0,
                                        help="EC2 annexe B : 4000 K ⇔ 33,3 kJ/mol. Ajuster sur essais.")
                Ea = Ea_kJ * 1000.0

            st.divider()

            # ==============================================================
            #  ANALYSES
            # ==============================================================
            st.markdown("#### Analyses")

            overlays = st.session_state.setdefault("ab_ov", {})

            # ---- ① Résistance à un âge donné (toujours visible) --------
            st.markdown("##### ① Résistance à un âge donné")
            age_max = len(profil_temps) if profil_temps else 28
            t_sel = st.slider("Âge (jours réels)", 1, int(max(age_max, 2)),
                              min(28, int(max(age_max, 2))), key="ab_age")
            t_e_sel = _age_equiv_point(t_sel, T, profil_temps, Ea)
            fck_val = float(fck_of_age_equiv(fck28, s_ref, t_e_sel))
            pct_val = fck_val / fck28 * 100.0 if fck28 else 0.0

            etat = "warn" if t_e_sel <= 3.0 else "ok"
            _bloc(f"fck({t_sel} j à {T:.0f} °C)",
                  f"{fck_val:.2f} MPa · {pct_val:.0f} % de fck(28)", etat)
            st.caption(f"t_e = {t_e_sel:.1f} j"
                       + (f"  ({source_temp})" if source_temp != "manuelle" else "")
                       + ("  — ⚠️ ≤ 3 j : hors domaine EC2" if t_e_sel <= 3.0 else ""))
            overlays["point"] = (t_sel, fck_val)

            # Variante défavorable quand le profil contient des jours prévus
            t_e_def = fck_def = None
            if defav is not None:
                t_e_def = _age_equiv_point(t_sel, defav["T_moy"], defav["profil"], Ea)
                fck_def = float(fck_of_age_equiv(fck28, s_ref, t_e_def))
                st.caption(f"🌡️ Défavorable (jours prévus − {defav['marge']:.0f} °C) : "
                           f"fck({t_sel} j) ≈ {fck_def:.2f} MPa · t_e = {t_e_def:.1f} j — "
                           "la météo prévue n'est pas acquise ; recalcule au jour J.")

            st.divider()

            # ---- ② Recherche d'un seuil de résistance -----------------
            with st.expander("② Recherche d'un seuil de résistance", expanded=False):
                mode_cible = st.radio("Cible", ["Pourcentage de fck(28)", "Résistance (MPa)"],
                                      horizontal=True, key="ab_s_mode")
                if mode_cible == "Pourcentage de fck(28)":
                    pct_s = st.slider("Pourcentage", 30, 100, 75, step=5, key="ab_s_pct")
                    target_s = fck_target_from_pct(fck28, pct_s)
                    cible_txt = f"{pct_s} % = {target_s:.1f} MPa"
                else:
                    target_s = st.number_input("Résistance (MPa)", min_value=1.0,
                                               max_value=float(fck28),
                                               value=min(20.0, float(fck28)),
                                               step=1.0, format="%.1f", key="ab_s_mpa")
                    cible_txt = f"{target_s:.1f} MPa ({target_s / fck28 * 100:.0f} %)"

                t_seuil = t_real_for_target(fck28, s_ref, target_s, T, Ea)
                if t_seuil is None:
                    _bloc(f"Seuil {cible_txt}", "non atteignable", "nok")
                    overlays.pop("seuil", None)
                else:
                    date_txt = ""
                    if date_coulage:
                        jour = date_coulage + _dt.timedelta(days=float(t_seuil))
                        date_txt = f" — le {jour.strftime('%d/%m/%Y')}"
                    _bloc(f"Seuil {cible_txt}", f"après {t_seuil:.1f} j{date_txt}", "ok")
                    overlays["seuil"] = {"fck": target_s, "t": t_seuil}
                st.checkbox("Afficher sur le graphique", value=overlays.get("seuil_show", False),
                            key="ab_chk_seuil",
                            on_change=lambda: overlays.update(seuil_show=st.session_state["ab_chk_seuil"]))

            # ---- ③ Optimisation de la classe béton ---------------------
            with st.expander("③ Optimiser la classe (délai réduit)", expanded=False):
                o1, o2 = st.columns(2)
                with o1:
                    cible_opt = st.number_input("Résistance (MPa)", min_value=1.0,
                                                value=round(fck_val, 1), step=1.0,
                                                format="%.1f", key="ab_o_mpa")
                with o2:
                    delai_opt = st.number_input("Délai (jours)", min_value=0.5,
                                                value=max(3.0, round(t_sel / 2, 1)),
                                                step=0.5, format="%.1f", key="ab_o_del")
                res_opt = classe_min_pour_delai(cible_opt, s_ref, T, delai_opt, Ea)
                if res_opt is None:
                    _bloc(f"Aucune classe pour {cible_opt:.1f} MPa en {delai_opt:.1f} j", "", "nok")
                    overlays.pop("opt", None)
                else:
                    lbl_o, fck28_o, fck_att = res_opt
                    sur = " (actuelle)" if parse_fck(lbl_o) <= fck28 else ""
                    _bloc(f"Classe minimale : {lbl_o}{sur}",
                          f"{fck_att:.1f} MPa à {delai_opt:.1f} j",
                          "info" if sur else "ok")
                    overlays["opt"] = {"label": lbl_o, "s": s_ref, "fck28": fck28_o,
                                       "t": delai_opt, "fck": fck_att}
                st.checkbox("Afficher sur le graphique", value=overlays.get("opt_show", False),
                            key="ab_chk_opt",
                            on_change=lambda: overlays.update(opt_show=st.session_state["ab_chk_opt"]))

            # ---- ④ Comparateur d'équivalence (multi-classes) -----------
            with st.expander("④ Comparateur d'équivalence", expanded=False):
                st.caption("Comparer la montée en résistance de plusieurs classes.")
                classes_cmp = st.multiselect("Classes à comparer", CLASSES, default=CLASSES,
                                             key="ab_cmp_cls")
                ciment_cmp = st.selectbox("Ciment (comparateur)", list(CIMENTS.keys()),
                                          index=list(CIMENTS.keys()).index(ciment),
                                          key="ab_cmp_cim")
                s_cmp = CIMENTS[ciment_cmp]
                overlays["cmp_classes"] = classes_cmp
                overlays["cmp_s"] = s_cmp
                st.checkbox("Afficher sur le graphique", value=overlays.get("cmp_show", False),
                            key="ab_chk_cmp",
                            on_change=lambda: overlays.update(cmp_show=st.session_state["ab_chk_cmp"]))

            # ---- ⑤ Analyse d'une éprouvette ----------------------------
            with st.expander("⑤ Analyse d'un essai mesuré", expanded=False):
                e1, e2, e3 = st.columns(3)
                with e1:
                    res_lab = st.number_input("Mesure (MPa)", min_value=0.0, value=0.0,
                                              step=0.5, format="%.1f", key="ab_e_mpa")
                with e2:
                    age_lab = st.number_input("Âge (j)", min_value=0.5, value=7.0,
                                              step=0.5, format="%.1f", key="ab_e_age")
                with e3:
                    T_lab = st.number_input("T cure (°C)", value=float(T),
                                            step=1.0, format="%.1f",
                                            min_value=-5.0, max_value=60.0, key="ab_e_T")
                if res_lab > 0:
                    est = classe_estimee_depuis_mesure(res_lab, age_lab, s_ref, T_lab, Ea)
                    _bloc("Classe estimée",
                          f"≈ {est['classe']} (fck28 ≈ {est['fck28_est']:.1f} MPa)", "info")
                    ec = (est["fck28_est"] - fck28) / fck28 * 100.0
                    if abs(ec) <= 8.0:
                        _bloc(f"Cohérent avec {beton_label}", f"écart {ec:+.0f} %", "ok")
                    elif ec > 8.0:
                        _bloc(f"Au-dessus de {beton_label}", f"écart {ec:+.0f} %", "ok")
                    else:
                        _bloc(f"Sous {beton_label}", f"écart {ec:+.0f} %", "warn")
                    if est["t_e"] <= 3.0:
                        st.caption("⚠️ t_e ≤ 3 j : hors domaine EC2.")
                    overlays["lab"] = {"fck": res_lab, "age": age_lab}
                else:
                    overlays.pop("lab", None)
                st.checkbox("Afficher sur le graphique", value=overlays.get("lab_show", False),
                            key="ab_chk_lab",
                            on_change=lambda: overlays.update(lab_show=st.session_state["ab_chk_lab"]))

    # ==================================================================
    #  COLONNE DROITE — GRAPHIQUE + FORMULES
    # ==================================================================
    t_real = np.linspace(1, 40, 500)
    y_ref = fck_of_age_equiv(fck28, s_ref, age_equiv(t_real, T, Ea))
    ref = {"label": f"{beton_label} — {ciment.split(' — ')[0]}", "y": y_ref}

    # Comparateur multi-classes
    comparateur = None
    if overlays.get("cmp_show") and overlays.get("cmp_classes"):
        comparateur = []
        s_c = overlays.get("cmp_s", s_ref)
        for cl in overlays["cmp_classes"]:
            if cl == beton_label:
                continue  # pas de doublons avec la référence
            fc = parse_fck(cl)
            yc = fck_of_age_equiv(fc, s_c, age_equiv(t_real, T, Ea))
            comparateur.append({"label": cl, "y": yc})

    # Overlay seuil
    seuil_ov = overlays.get("seuil") if overlays.get("seuil_show") else None

    # Overlay optimisation
    opt_ov = None
    if overlays.get("opt_show") and overlays.get("opt"):
        o = overlays["opt"]
        y_o = fck_of_age_equiv(o["fck28"], o["s"], age_equiv(t_real, T, Ea))
        opt_ov = {"label": f"{o['label']} (optim.)", "y": y_o, "t": o["t"], "fck": o["fck"]}

    # Overlay essai labo
    lab_ov = overlays.get("lab") if overlays.get("lab_show") else None

    # Overlay variante défavorable (météo prévue abaissée)
    defav_ov = None
    if defav is not None:
        y_def = fck_of_age_equiv(fck28, s_ref, age_equiv(t_real, defav["T_moy"], Ea))
        defav_ov = {"label": f"défavorable (prévisions − {defav['marge']:.0f} °C)",
                    "y": y_def}

    factor = float(age_equiv(1.0, T, Ea))
    t28_real = 28.0 / factor if factor > 0 else None

    fig = build_figure(
        t_real, ref, comparateur=comparateur,
        point=overlays.get("point"), t28_real=t28_real,
        seuil=seuil_ov, opt=opt_ov, mesure_lab=lab_ov,
        defav=defav_ov,
        titre=f"fck(t) — {beton_label} — {T:.0f} °C",
    )

    with col_d:
        st.pyplot(fig, use_container_width=True)

        # ---- Formules EC2 (une par ligne, aérées) ----
        with st.expander("📘 Formules EC2", expanded=False):
            st.latex(r"f_{ck}(t) = \beta_{cc}(t_e)\,f_{cm} - 8")
            st.latex(r"f_{cm} = f_{ck} + 8")
            st.latex(r"f_{ck}(t) = f_{ck} \quad \text{pour } t_e \geq 28\,\text{j}")
            st.latex(r"\beta_{cc}(t_e) = \exp\!\Big(s\,\big[1 - \sqrt{28/t_e}\,\big]\Big)")
            st.caption("s = 0,20 (R)  /  0,25 (N)  /  0,38 (S)")
            st.latex(r"t_e = t \cdot \exp\!\left(\frac{-E_a}{R}"
                     r"\left[\frac{1}{273{,}15+T} - \frac{1}{293{,}15}\right]\right)")
            st.caption("Ea/R = 4000 K (EC2 annexe B, éq. B.10)")
            st.markdown("Domaine de validité : **3 j < t_e < 28 j** (§3.1.2(5)). "
                        "Sous 3 jours : résistance à déterminer par essais.")

    # ---- Export PDF (data prête) ----
    st.session_state["ab_rapport"] = build_rapport_data({
        "beton": beton_label, "fck28": fck28, "ciment": ciment, "s": s_ref,
        "mode_temp": mode_temp, "T_moyenne": T, "source_temp": source_temp,
        "profil_temps": profil_temps, "Ea_kJ_mol": Ea / 1000.0,
        "age_sel_j": t_sel, "t_e_sel_j": t_e_sel,
        "fck_sel_MPa": fck_val, "pct_fck28": pct_val,
        "date_coulage": date_coulage.isoformat() if date_coulage else None,
        "defav": ({"T_moy": defav["T_moy"], "n_prev": defav["n_prev"],
                   "n_extra": defav["n_extra"], "marge": defav["marge"],
                   "t_e_sel_j": t_e_def, "fck_sel_MPa": fck_def}
                  if defav else None),
    })


if __name__ == "__main__":
    st.set_page_config(page_title="Évolution fck(t) — EC2", page_icon="🧱", layout="wide")
    show()

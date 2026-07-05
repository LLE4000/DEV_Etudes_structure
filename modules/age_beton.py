# =============================================================
#  age_beton.py — Évolution de la résistance du béton (EC2)
#  VERSION 3.1 — refonte ergonomique
#
#  Base normative :
#   - EN 1992-1-1 §3.1.2 :
#       fcm(t) = βcc(t)·fcm          (éq. 3.1)
#       βcc(t) = exp( s·(1−√(28/t)) )  (éq. 3.2)
#       fck(t) = fcm(t) − 8  pour 3 < t < 28 j ; fck(t) = fck pour t ≥ 28 j
#     s selon la CLASSE DE RÉSISTANCE du ciment (et non le type CEM) :
#       s = 0,20 : classe R  (CEM 42,5R ; 52,5N ; 52,5R)
#       s = 0,25 : classe N  (CEM 32,5R ; 42,5N)
#       s = 0,38 : classe S  (CEM 32,5N)
#   - Effet de la température : âge équivalent (maturité),
#     EN 1992-1-1 annexe B éq. B.10 :
#       t_e = Σ Δt_i · exp( 13,65 − 4000 / (273 + T) )
#     4000 K ≡ Ea/R avec Ea ≈ 33,3 kJ/mol (Ea ajustable en avancé).
#
#  Évolutions v3.1 (ergonomie — le principe de calcul est inchangé) :
#   - Layout 2/3 (paramètres) — 1/3 (graphique).
#   - Bloc "Béton de référence" compacté (champs regroupés sur moins de lignes).
#   - Paramètres avancés (Ea) déplacés dans un expander ⚙️.
#   - Température : mode Manuel / Météo (Open-Meteo, Bruxelles par défaut) ;
#     min / max / moyenne affichés après récupération. Libellé "jour par jour" retiré.
#   - Les 4 analyses regroupées sous UN sélecteur unique (radio), présentées comme
#     des analyses d'un même béton, chacune avec sa case "Afficher sur le graphique".
#   - Le graphique superpose les analyses cochées (point, ligne, repère).
#
#  Le style, le graphique et l'encadré des formules EC2 restent identiques.
# =============================================================

import math
import datetime as _dt
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Import réseau optionnel (mode météo) — dégradation propre si absent.
try:
    import requests  # noqa: F401
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

# =============================================================
#  STYLE (aligné poutre.py)
# =============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6", "info": "#eef2ff"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌", "info": "ℹ️"}


def _bloc(left: str, right: str = "", etat: str = "ok"):
    right_html = f"<div style='font-weight:600;opacity:.9;white-space:nowrap;'>{right}</div>" if right else ""
    st.markdown(
        f"""
        <div style="background:{C_COULEURS.get(etat,'#f6f6f6')};padding:12px 14px;border-radius:10px;
             border:1px solid #d9d9d9;margin:8px 0 4px 0;display:flex;justify-content:space-between;
             align-items:center;gap:10px;">
          <div style="font-weight:700;">{left}</div>
          <div style="display:flex;align-items:center;gap:10px;">{right_html}
            <div style="font-size:20px;line-height:1;">{C_ICONES.get(etat,'')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mini_metrics(items):
    """Petite rangée de valeurs (label, valeur) compacte, sans dénaturer le style."""
    cells = "".join(
        f"<div style='flex:1;text-align:center;padding:6px 4px;'>"
        f"<div style='font-size:11px;color:#6b7280;'>{lbl}</div>"
        f"<div style='font-size:15px;font-weight:700;color:#374151;'>{val}</div></div>"
        for lbl, val in items
    )
    st.markdown(
        f"<div style='display:flex;gap:6px;background:#f8fafc;border:1px solid #e5e7eb;"
        f"border-radius:10px;padding:4px 6px;margin:6px 0;'>{cells}</div>",
        unsafe_allow_html=True,
    )


# =============================================================
#  DONNÉES
# =============================================================
CLASSES = ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60"]

# s selon la classe de résistance du ciment (EC2 §3.1.2(6))
CIMENTS = {
    "Classe R — durcissement rapide": 0.20,
    "Classe N — durcissement normal": 0.25,
    "Classe S — durcissement lent": 0.38,
}
CIMENT_DETAIL = {
    "Classe R — durcissement rapide": "CEM 42,5R · CEM 52,5N · CEM 52,5R",
    "Classe N — durcissement normal": "CEM 32,5R · CEM 42,5N",
    "Classe S — durcissement lent": "CEM 32,5N",
}

R_GAZ = 8.314          # J/(mol·K)
EA_EC2 = 4000.0 * R_GAZ  # ≈ 33,3 kJ/mol : valeur de l'éq. B.10 (4000 K)
T_REF_K = 293.15       # 20 °C


# =============================================================
#  CALCULS (fonctions pures)
# =============================================================
def parse_fck(label: str) -> int:
    return int(label.split("/")[0].replace("C", ""))


def beta_cc(t_days_equiv, s: float):
    """βcc(t) = exp( s·(1 − √(28/t)) ) — EC2 éq. 3.2 (t = âge équivalent, j)."""
    t = np.maximum(np.asarray(t_days_equiv, dtype=float), 1e-6)
    return np.exp(s * (1.0 - np.sqrt(28.0 / t)))


def fck_of_age_equiv(fck28: float, s: float, t_days_equiv):
    """fck(t) = βcc(t_e)·fcm − 8, plafonné à fck pour t_e ≥ 28 j (fcm = fck + 8)."""
    fcm = fck28 + 8.0
    t_e = np.asarray(t_days_equiv, dtype=float)
    val = beta_cc(t_e, s) * fcm - 8.0
    return np.where(t_e < 28.0, val, float(fck28))


def age_equiv(t_days_real, T_celsius: float, Ea: float = EA_EC2):
    """
    Âge équivalent à 20 °C pour température constante T (maturité, Arrhenius) :
        t_e = t · exp( −Ea/R · (1/T_abs − 1/293,15) )
    Avec Ea = 4000·R, identique à l'éq. B.10 de l'EN 1992-1-1.
    """
    T_abs = float(T_celsius) + 273.15
    factor = math.exp((-Ea / R_GAZ) * (1.0 / T_abs - 1.0 / T_REF_K))
    return np.asarray(t_days_real, dtype=float) * factor


def maturity_factor(T_celsius: float, Ea: float = EA_EC2) -> float:
    """Facteur de maturité journalier k(T) tel que t_e = Σ Δt·k(T)."""
    T_abs = float(T_celsius) + 273.15
    return math.exp((-Ea / R_GAZ) * (1.0 / T_abs - 1.0 / T_REF_K))


def age_equiv_from_series(temps_celsius, Ea: float = EA_EC2, dt_days: float = 1.0):
    """
    Âge équivalent cumulé pour un historique de températures (éq. B.10 sommée).
    temps_celsius : liste/array de températures moyennes par pas dt_days.
    Retourne l'âge équivalent total à 20 °C (jours).
    """
    arr = np.asarray(temps_celsius, dtype=float)
    if arr.size == 0:
        return 0.0
    facteurs = np.array([maturity_factor(T, Ea) for T in arr])
    return float(np.sum(facteurs) * dt_days)


def fck_target_from_pct(fck28: float, pct: float) -> float:
    """Résistance cible (MPa) correspondant à un pourcentage de fck(28)."""
    return fck28 * float(pct) / 100.0


def t_real_for_target(fck28: float, s: float, target_MPa: float,
                      T_celsius: float, Ea: float = EA_EC2,
                      tmax: float = 90.0, tol: float = 1e-3):
    """
    Temps réel t (à T constante) tel que fck(t_e(t,T)) = target.
    None si la cible dépasse fck28 ou n'est pas atteinte avant tmax.
    """
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
    """
    Classe de béton minimale telle que fck(t_e(delai, T)) ≥ target_MPa.
    Retourne (label, fck28, fck_atteint) ou None si aucune classe ne convient.
    """
    t_e = float(age_equiv(delai_jours, T_celsius, Ea))
    for label in CLASSES:
        fck28 = parse_fck(label)
        fck_atteint = float(fck_of_age_equiv(fck28, s, t_e))
        if fck_atteint + 1e-6 >= target_MPa:
            return label, fck28, fck_atteint
    return None


def classe_estimee_depuis_mesure(res_mesuree: float, age_jours: float, s: float,
                                 T_celsius: float, Ea: float = EA_EC2):
    """
    À partir d'une résistance mesurée à un âge donné, remonte à fck(28) probable,
    puis à la classe de béton normalisée la plus proche (par le fck).
    Retourne dict(t_e, fck28_est, classe, fck_classe, ecart_pct).
    """
    t_e = float(age_equiv(age_jours, T_celsius, Ea))
    if t_e >= 28.0:
        fck28_est = res_mesuree
    else:
        b = float(beta_cc(t_e, s))
        # fck_mes = b·(fck28+8) − 8  ⇒  fck28 = (fck_mes + 8)/b − 8
        fck28_est = (res_mesuree + 8.0) / b - 8.0
    # classe normalisée la plus proche par fck
    fcks = [parse_fck(c) for c in CLASSES]
    idx = int(np.argmin([abs(f - fck28_est) for f in fcks]))
    classe = CLASSES[idx]
    fck_classe = fcks[idx]
    ecart = (fck28_est - fck_classe) / fck_classe * 100.0 if fck_classe else 0.0
    return {"t_e": t_e, "fck28_est": fck28_est, "classe": classe,
            "fck_classe": fck_classe, "ecart_pct": ecart}


# =============================================================
#  MÉTÉO (Open-Meteo — historique)
# =============================================================
def fetch_temps_open_meteo(lat: float, lon: float, date_debut: str, date_fin: str):
    """
    Récupère la température moyenne journalière (°C) via l'API historique Open-Meteo.
    Retourne (dates, temps) ou lève une exception. Aucune clé API requise.
    """
    if not _HAS_REQUESTS:
        raise RuntimeError("Le module 'requests' n'est pas disponible dans cet environnement.")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": date_debut, "end_date": date_fin,
        "daily": "temperature_2m_mean", "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    temps = [float(t) if t is not None else float("nan") for t in temps]
    return dates, temps


def geocode_open_meteo(nom_lieu: str):
    """Géocodage simple via Open-Meteo. Retourne (lat, lon, label) ou None."""
    if not _HAS_REQUESTS:
        raise RuntimeError("Le module 'requests' n'est pas disponible dans cet environnement.")
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": nom_lieu, "count": 1, "language": "fr"}, timeout=20)
    r.raise_for_status()
    res = r.json().get("results")
    if not res:
        return None
    g = res[0]
    label = ", ".join(filter(None, [g.get("name"), g.get("admin1"), g.get("country")]))
    return g["latitude"], g["longitude"], label


# =============================================================
#  GRAPHIQUE
# =============================================================
COL_REF = "#2e6fb0"
COL_ALT = "#c0392b"
COL_MES = "#1f8a70"
COL_SEUIL = "#b45309"   # analyse "date d'obtention"
COL_OPT = "#7c3aed"     # analyse "optimisation classe"
COL_GRID = "#e5e7eb"
COL_TXT = "#374151"


def build_figure(t_real, ref, alt=None, point=None, mesure=None, t28_real=None,
                 seuil=None, opt=None, mesure_lab=None, titre=""):
    """
    ref / alt : dict(label, y) — courbes fck(t réels).
    point     : (t, fck) sélectionné sur la référence.
    mesure    : dict(fck, t_est) — résistance mesurée (comparateur historique).
    t28_real  : jours réels correspondant à t_e = 28 j (fin de montée).
    seuil     : dict(fck, t) — analyse "date d'obtention" à superposer.
    opt       : dict(label, y, t, fck) — analyse "optimisation classe".
    mesure_lab: dict(fck, age) — point d'essai laboratoire.
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    # Courbe de référence + remplissage léger
    ax.plot(t_real, ref["y"], color=COL_REF, lw=2.4, label=ref["label"], zorder=3)
    ax.fill_between(t_real, ref["y"], ref["y"].min(), color=COL_REF, alpha=0.06, zorder=1)

    # Courbe comparée (comparateur)
    if alt is not None:
        ax.plot(t_real, alt["y"], color=COL_ALT, lw=2.0, ls="-", alpha=0.9,
                label=alt["label"], zorder=3)

    # Courbe d'optimisation de classe
    if opt is not None and opt.get("y") is not None:
        ax.plot(t_real, opt["y"], color=COL_OPT, lw=2.0, ls="-", alpha=0.9,
                label=opt["label"], zorder=3)
        if opt.get("t") is not None:
            ax.scatter([opt["t"]], [opt["fck"]], s=55, marker="^",
                       color=COL_OPT, edgecolor="white", lw=1.3, zorder=6)
            ax.plot([opt["t"], opt["t"]], [ref["y"].min(), opt["fck"]],
                    color=COL_OPT, lw=1.0, ls="--", alpha=0.5, zorder=2)

    # Repère fin de montée (t_e = 28 j)
    if t28_real is not None and t_real.min() < t28_real < t_real.max():
        ax.axvline(t28_real, color="#9ca3af", lw=1.0, ls=":", zorder=2)
        ax.annotate("t_e = 28 j", xy=(t28_real, ax.get_ylim()[0]),
                    xytext=(t28_real + 0.4, ref["y"].min() + 1),
                    fontsize=8, color="#6b7280")

    # Point sélectionné : guides discrets + marqueur + étiquette
    if point is not None:
        t_sel, f_sel = point
        ax.plot([t_sel, t_sel], [ref["y"].min(), f_sel], color=COL_REF, lw=1.0, ls="--", alpha=0.55, zorder=2)
        ax.plot([t_real.min(), t_sel], [f_sel, f_sel], color=COL_REF, lw=1.0, ls="--", alpha=0.55, zorder=2)
        ax.scatter([t_sel], [f_sel], s=55, color=COL_REF, edgecolor="white", lw=1.4, zorder=5)
        ax.annotate(f"{f_sel:.1f} MPa à {t_sel:.0f} j",
                    xy=(t_sel, f_sel), xytext=(8, 10), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color=COL_REF,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=COL_REF, lw=0.8, alpha=0.9),
                    zorder=6)

    # Analyse "date d'obtention" : repère seuil (croix horizontale/verticale)
    if seuil is not None and seuil.get("t") is not None:
        f_s, t_s = seuil["fck"], seuil["t"]
        ax.axhline(f_s, color=COL_SEUIL, lw=1.1, ls="--", alpha=0.7, zorder=2)
        ax.plot([t_s, t_s], [ref["y"].min(), f_s], color=COL_SEUIL, lw=1.1, ls="--", alpha=0.7, zorder=2)
        ax.scatter([t_s], [f_s], s=55, marker="s", color=COL_SEUIL,
                   edgecolor="white", lw=1.3, zorder=6, label=f"Seuil : {f_s:.1f} MPa @ {t_s:.1f} j")

    # Comparateur : mesure de résistance
    if mesure is not None and mesure.get("fck", 0) > 0:
        ax.axhline(mesure["fck"], color=COL_MES, lw=1.3, ls=":", zorder=2,
                   label=f"Mesure : {mesure['fck']:.1f} MPa")
        if mesure.get("t_est"):
            ax.scatter([mesure["t_est"]], [mesure["fck"]], s=45, marker="D",
                       color=COL_MES, edgecolor="white", lw=1.2, zorder=5)
            ax.annotate(f"âge estimé ≈ {mesure['t_est']:.1f} j",
                        xy=(mesure["t_est"], mesure["fck"]), xytext=(8, -14),
                        textcoords="offset points", fontsize=8.5, color=COL_MES)

    # Analyse "essai laboratoire" : point mesuré (âge, fck)
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
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="#e5e7eb")
    fig.tight_layout()
    return fig


# =============================================================
#  EXPORT PDF — structure de données (à consommer plus tard)
# =============================================================
def build_rapport_data(contexte: dict) -> dict:
    """
    Rassemble en un dict sérialisable toutes les données affichées, prêtes
    pour un export_pdf ultérieur. Ne réalise aucun rendu : sépare calcul et sortie.
    """
    return {"module": "age_beton", "version": "3.1", **contexte}


# =============================================================
#  HELPERS UI
# =============================================================
def _overlay_check(key: str, overlays: dict):
    """Case 'Afficher sur le graphique' pilotant l'overlay 'key'."""
    val = st.checkbox("Afficher sur le graphique", value=overlays.get(f"{key}_show", False),
                      key=f"ab_chk_{key}")
    overlays[f"{key}_show"] = val


def _age_equiv_point(t_sel, T, profil_temps, Ea):
    """Âge équivalent au point sélectionné (profil météo si dispo, sinon T constante)."""
    if profil_temps:
        n = min(int(round(t_sel)), len(profil_temps))
        t_e = age_equiv_from_series(profil_temps[:n], Ea)
        reste = t_sel - n
        if reste > 0 and n < len(profil_temps):
            t_e += maturity_factor(profil_temps[n], Ea) * reste
        return t_e
    return float(age_equiv(t_sel, T, Ea))


def _saisie_temperature(mode_temp, date_coulage):
    """
    Gère la saisie de température selon le mode.
    Retourne (T_moyenne, profil_temps|None, source_temp).
    """
    if mode_temp == "Manuel":
        T = st.number_input("Température moyenne (°C)", value=20.0, step=1.0, format="%.1f",
                            min_value=-5.0, max_value=60.0,
                            help="Température moyenne de cure, supposée constante.")
        return T, None, "manuelle"

    # ---- Mode Météo ----
    if not _HAS_REQUESTS:
        st.warning("Module réseau indisponible ici. En production, `pip install requests` active la météo.")
    cA, cB, cC = st.columns([1.4, 1, 1])
    with cA:
        lieu = st.text_input("Lieu du chantier", value="Bruxelles, Belgique")
    with cB:
        d_coul = st.date_input("Coulage",
                               value=date_coulage or _dt.date.today() - _dt.timedelta(days=14),
                               key="ab_meteo_coul")
    with cC:
        d_ctrl = st.date_input("Contrôle", value=_dt.date.today(), key="ab_meteo_ctrl")

    if st.button("🔄 Récupérer la météo", use_container_width=True):
        try:
            geo = geocode_open_meteo(lieu)
            if geo is None:
                st.error("Lieu introuvable.")
            else:
                lat, lon, label = geo
                dates, temps = fetch_temps_open_meteo(lat, lon, d_coul.isoformat(), d_ctrl.isoformat())
                st.session_state["ab_meteo"] = {"label": label, "dates": dates, "temps": temps}
                st.success(f"{label} — {len(temps)} jours récupérés.")
        except Exception as e:
            st.error(f"Échec de récupération : {e}")

    meteo = st.session_state.get("ab_meteo")
    if meteo and meteo.get("temps"):
        valides = [t for t in meteo["temps"] if not math.isnan(t)]
        if valides:
            T = float(np.mean(valides))
            # min / max / moyenne
            _mini_metrics([
                ("T min", f"{min(valides):.1f} °C"),
                ("T moy", f"{T:.1f} °C"),
                ("T max", f"{max(valides):.1f} °C"),
            ])
            return T, valides, f"météo ({meteo['label']})"

    st.caption("Renseignez le lieu et les dates, puis récupérez la météo. T = 20 °C par défaut en attendant.")
    return 20.0, None, "manuelle"


# =============================================================
#  PAGE
# =============================================================
def show():
    st.markdown("## Évolution de la résistance du béton — EC2")
    st.caption("fck(t) selon EN 1992-1-1 §3.1.2, effet de la température par âge équivalent (annexe B).")

    # Layout : 2/3 paramètres — 1/3 graphique
    col_g, col_d = st.columns([2, 1])

    # =====================================================================
    #  COLONNE GAUCHE — PARAMÈTRES & ANALYSES
    # =====================================================================
    with col_g:
        # ---------------- Bloc "Béton de référence" (compact) ----------------
        with st.container(border=True):
            st.markdown("#### Béton de référence")

            # Ligne 1 : classe · ciment · date de coulage
            r1c1, r1c2, r1c3 = st.columns([1, 1.5, 1.2])
            with r1c1:
                beton_label = st.selectbox("Classe béton", CLASSES, index=2)
                fck28 = parse_fck(beton_label)
            with r1c2:
                ciment = st.selectbox("Classe ciment", list(CIMENTS.keys()), index=1,
                                      help="La classe de résistance du ciment (R/N/S) fixe s dans l'EC2, "
                                           "pas le type CEM.")
                s_ref = CIMENTS[ciment]
            with r1c3:
                date_coulage = st.date_input("Date de coulage", value=None,
                                             help="Sert à dater les seuils calculés (décoffrage…).")

            # Ligne 2 : mode température + saisie associée
            r2c1, r2c2 = st.columns([1, 2])
            with r2c1:
                mode_temp = st.radio("Température", ["Manuel", "Météo"], horizontal=True,
                                     help="Manuel : moyenne constante. Météo : import Open-Meteo.")
            with r2c2:
                T, profil_temps, source_temp = _saisie_temperature(mode_temp, date_coulage)

            st.caption(f"{CIMENT_DETAIL[ciment]} — s = {s_ref:.2f}")

            # Paramètres avancés (engrenage)
            with st.expander("⚙️ Paramètres avancés"):
                Ea_kJ = st.number_input("Énergie d'activation Ea [kJ/mol]",
                                        value=EA_EC2 / 1000.0, step=1.0, format="%.1f",
                                        min_value=20.0, max_value=60.0,
                                        help="EC2 annexe B (éq. B.10) ⇔ 4000 K ⇔ 33,3 kJ/mol. "
                                             "Ajuster uniquement si calibré sur essais.")
                Ea = Ea_kJ * 1000.0

        # =================================================================
        #  ANALYSES — un même béton, plusieurs analyses
        # =================================================================
        with st.container(border=True):
            st.markdown("#### Analyses")
            st.caption("Choisissez une analyse à réaliser sur le béton ci-dessus. "
                       "Cochez « Afficher sur le graphique » pour superposer plusieurs résultats.")

            analyse = st.radio(
                "Type d'analyse",
                ["① Résistance à un âge donné",
                 "② Date d'obtention d'une résistance",
                 "③ Optimiser la classe (plus vite)",
                 "④ Analyser un essai mesuré"],
                label_visibility="collapsed",
            )

            # état partagé des overlays graphiques
            overlays = st.session_state.setdefault("ab_overlays", {})

            # ---- Analyse ① : résistance à un âge donné (fonction historique) ----
            if analyse.startswith("①"):
                age_max = len(profil_temps) if profil_temps else 40
                t_sel = st.slider("Âge du béton (jours réels)", 1, int(max(age_max, 2)),
                                  min(14, int(max(age_max, 2))))
                t_e_sel = _age_equiv_point(t_sel, T, profil_temps, Ea)
                fck_val = float(fck_of_age_equiv(fck28, s_ref, t_e_sel))
                pct_val = fck_val / fck28 * 100.0 if fck28 else 0.0

                etat = "warn" if t_e_sel <= 3.0 else "ok"
                _bloc(f"fck({t_sel} j à {T:.0f} °C)",
                      f"{fck_val:.2f} MPa · {pct_val:.0f} % de fck(28)", etat)
                st.caption(f"Âge équivalent à 20 °C : t_e = {t_e_sel:.1f} j"
                           + (f"  ({source_temp})" if source_temp != "manuelle" else "")
                           + ("  —  ⚠️ t_e ≤ 3 j : hors domaine EC2, essais requis." if t_e_sel <= 3.0 else ""))
                overlays["point"] = (t_sel, fck_val)  # le point de référence est toujours tracé

            else:
                # hors analyse ①, on conserve un point de référence par défaut (14 j)
                t_sel = 14
                t_e_sel = _age_equiv_point(t_sel, T, profil_temps, Ea)
                fck_val = float(fck_of_age_equiv(fck28, s_ref, t_e_sel))
                pct_val = fck_val / fck28 * 100.0 if fck28 else 0.0
                overlays["point"] = (t_sel, fck_val)

            # ---- Analyse ② : date d'obtention d'une résistance ----
            if analyse.startswith("②"):
                mode_cible = st.radio("Définir la cible par", ["Pourcentage de fck(28)", "Résistance (MPa)"],
                                      horizontal=True)
                if mode_cible == "Pourcentage de fck(28)":
                    pct = st.slider("Pourcentage visé", 30, 100, 75, step=5)
                    target_seuil = fck_target_from_pct(fck28, pct)
                    cible_txt = f"{pct} % de fck(28) = {target_seuil:.1f} MPa"
                else:
                    target_seuil = st.number_input("Résistance visée (MPa)", min_value=1.0,
                                                   max_value=float(fck28), value=min(20.0, float(fck28)),
                                                   step=1.0, format="%.1f")
                    cible_txt = f"{target_seuil:.1f} MPa ({target_seuil / fck28 * 100:.0f} % de fck(28))"

                t_seuil = t_real_for_target(fck28, s_ref, target_seuil, T, Ea)
                if t_seuil is None:
                    _bloc(f"Seuil {cible_txt}", "non atteignable avec cette classe", "nok")
                    overlays.pop("seuil", None)
                else:
                    date_txt = ""
                    if date_coulage:
                        jour = date_coulage + _dt.timedelta(days=float(t_seuil))
                        date_txt = f" — le {jour.strftime('%d/%m/%Y')}"
                    _bloc(f"Seuil {cible_txt}", f"atteint après {t_seuil:.1f} j{date_txt}", "ok")
                    if not date_coulage:
                        st.caption("Renseignez une date de coulage pour dater le décoffrage.")
                    overlays["seuil"] = {"fck": target_seuil, "t": t_seuil}

                _overlay_check("seuil", overlays)

            # ---- Analyse ③ : optimiser la classe ----
            if analyse.startswith("③"):
                c1, c2 = st.columns(2)
                with c1:
                    cible_opt = st.number_input("Résistance à atteindre (MPa)", min_value=1.0,
                                                value=round(fck_val, 1), step=1.0, format="%.1f",
                                                help="Par défaut, la résistance du point de référence.")
                with c2:
                    delai_opt = st.number_input("Délai souhaité (jours)", min_value=0.5,
                                                value=max(3.0, round(t_sel / 2, 1)), step=0.5, format="%.1f")
                res_opt = classe_min_pour_delai(cible_opt, s_ref, T, delai_opt, Ea)
                if res_opt is None:
                    _bloc(f"Aucune classe n'atteint {cible_opt:.1f} MPa en {delai_opt:.1f} j",
                          "à cette température / ciment", "nok")
                    overlays.pop("opt", None)
                else:
                    lbl, fck28_o, fck_att = res_opt
                    etat_o = "ok" if parse_fck(lbl) > fck28 else "info"
                    sur_place = " (classe actuelle suffit)" if parse_fck(lbl) <= fck28 else ""
                    _bloc(f"Classe minimale : {lbl}{sur_place}",
                          f"{fck_att:.1f} MPa à {delai_opt:.1f} j", etat_o)
                    overlays["opt"] = {"label": lbl, "s": s_ref, "fck28": fck28_o,
                                       "t": delai_opt, "fck": fck_att}
                _overlay_check("opt", overlays)

            # ---- Analyse ④ : essai mesuré ----
            if analyse.startswith("④"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    res_lab = st.number_input("Mesure (MPa)", min_value=0.0, value=0.0, step=0.5, format="%.1f")
                with c2:
                    age_lab = st.number_input("Âge éprouvette (j)", min_value=0.5, value=7.0, step=0.5, format="%.1f")
                with c3:
                    T_lab = st.number_input("T cure (°C)", value=float(T), step=1.0, format="%.1f",
                                            min_value=-5.0, max_value=60.0)
                if res_lab > 0:
                    est = classe_estimee_depuis_mesure(res_lab, age_lab, s_ref, T_lab, Ea)
                    _bloc("Classe estimée", f"≈ {est['classe']} (fck28 ≈ {est['fck28_est']:.1f} MPa)", "info")
                    ecart_prevu = (est["fck28_est"] - fck28) / fck28 * 100.0
                    if abs(ecart_prevu) <= 8.0:
                        _bloc(f"Cohérent avec {beton_label} prévu", f"écart {ecart_prevu:+.0f} %", "ok")
                    elif ecart_prevu > 8.0:
                        _bloc(f"Au-dessus de {beton_label} prévu", f"écart {ecart_prevu:+.0f} %", "ok")
                    else:
                        _bloc(f"Sous {beton_label} prévu", f"écart {ecart_prevu:+.0f} %", "warn")
                    if est["t_e"] <= 3.0:
                        st.caption("⚠️ Âge équivalent ≤ 3 j : estimation hors domaine EC2, à confirmer par essais.")
                    overlays["mesure_lab"] = {"fck": res_lab, "age": age_lab}
                else:
                    overlays.pop("mesure_lab", None)
                _overlay_check("mesure_lab", overlays)

        # -------------------- Comparateur (conservé) --------------------
        with st.expander("🔁 Comparateur d'équivalence", expanded=False):
            st.caption("Quand une autre classe atteint-elle la même résistance, à la même température ?")
            alt_label = st.selectbox("Classe comparée", CLASSES, index=0)
            ciment_alt = st.selectbox("Ciment (classe comparée)", list(CIMENTS.keys()),
                                      index=list(CIMENTS.keys()).index(ciment))
            s_alt = CIMENTS[ciment_alt]
            fck28_alt = parse_fck(alt_label)
            target = fck_val
            t_eq = t_real_for_target(fck28_alt, s_alt, target, T, Ea)
            if t_eq is not None:
                _bloc(f"{alt_label} atteint {target:.1f} MPa", f"vers {t_eq:.1f} j à {T:.0f} °C", "ok")
            else:
                _bloc(f"{alt_label} n'atteint pas {target:.1f} MPa", f"fck(28) = {fck28_alt} MPa", "nok")
            show_alt = st.checkbox("Afficher sur le graphique", value=False, key="ab_show_alt")

    # =====================================================================
    #  COLONNE DROITE — GRAPHIQUE
    # =====================================================================
    t_real = np.linspace(1, 40, 500)
    y_ref = fck_of_age_equiv(fck28, s_ref, age_equiv(t_real, T, Ea))
    ref = {"label": f"{beton_label} — {ciment.split(' — ')[0]}", "y": y_ref}

    # Overlay comparateur
    alt = None
    if show_alt:
        y_alt = fck_of_age_equiv(fck28_alt, s_alt, age_equiv(t_real, T, Ea))
        alt = {"label": f"{alt_label} — {ciment_alt.split(' — ')[0]}", "y": y_alt}

    # Overlays des analyses (uniquement si cochés)
    seuil_ov = overlays.get("seuil") if overlays.get("seuil_show") else None
    mesure_lab_ov = overlays.get("mesure_lab") if overlays.get("mesure_lab_show") else None
    opt_ov = None
    if overlays.get("opt_show") and overlays.get("opt"):
        o = overlays["opt"]
        y_opt = fck_of_age_equiv(o["fck28"], o["s"], age_equiv(t_real, T, Ea))
        opt_ov = {"label": f"{o['label']} (optim.)", "y": y_opt,
                  "t": o["t"], "fck": o["fck"]}

    factor = float(age_equiv(1.0, T, Ea))
    t28_real = 28.0 / factor if factor > 0 else None
    point = overlays.get("point")

    fig = build_figure(
        t_real, ref, alt=alt, point=point, t28_real=t28_real,
        seuil=seuil_ov, opt=opt_ov, mesure_lab=mesure_lab_ov,
        titre=f"fck(t) — {beton_label} — {T:.0f} °C",
    )

    with col_d:
        st.pyplot(fig, use_container_width=True)

        with st.expander("📘 Formules (EC2)", expanded=False):
            st.latex(r"f_{ck}(t)=\beta_{cc}(t_e)\,f_{cm}-8 \quad ;\quad f_{cm}=f_{ck}+8"
                     r"\quad ;\quad f_{ck}(t)=f_{ck}\ \text{pour}\ t_e\geq 28\,\text{j}")
            st.latex(r"\beta_{cc}(t_e)=\exp\!\Big(s\,\big[1-\sqrt{28/t_e}\,\big]\Big)"
                     r"\qquad s=0{,}20\,(R)\ /\ 0{,}25\,(N)\ /\ 0{,}38\,(S)")
            st.latex(r"t_e = t\cdot\exp\!\Big(\tfrac{-E_a}{R}\Big[\tfrac{1}{273{,}15+T}-\tfrac{1}{293{,}15}\Big]\Big)"
                     r"\qquad E_a/R = 4000\ \text{K (EC2 ann. B)}")
            st.markdown("Domaine de validité : **3 j < t_e < 28 j** (EN 1992-1-1 §3.1.2(5)). "
                        "En dessous de 3 jours, la résistance doit être déterminée par essais. "
                        "Température supposée constante ; pour un historique T(t) variable, "
                        "sommer les âges équivalents par intervalles (éq. B.10).")

    # -------------------- Données prêtes pour export PDF --------------------
    st.session_state["ab_rapport"] = build_rapport_data({
        "beton": beton_label, "fck28": fck28, "ciment": ciment, "s": s_ref,
        "mode_temp": mode_temp, "T_moyenne": T, "source_temp": source_temp,
        "profil_temps": profil_temps, "Ea_kJ_mol": Ea / 1000.0,
        "age_sel_j": t_sel, "t_e_sel_j": t_e_sel,
        "fck_sel_MPa": fck_val, "pct_fck28": pct_val,
        "analyse_active": analyse,
        "date_coulage": date_coulage.isoformat() if date_coulage else None,
    })


if __name__ == "__main__":
    st.set_page_config(page_title="Évolution fck(t) — EC2", page_icon="🧱", layout="wide")
    show()

# =============================================================
#  age_beton.py — Évolution de la résistance du béton (EC2)
#  VERSION 2.0 — refonte
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
#  Corrections vs version précédente :
#   - "Recode" ciment corrigé (classes R/N/S, pas CEM I/III).
#   - Formule de maturité EC2 (4000 K) par défaut au lieu de Ea=40 kJ/mol.
#   - Avertissement t ≤ 3 j (hors domaine de validité de fck(t)).
#   - Construction du graphique regroupée en une fonction unique.
# =============================================================

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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


# =============================================================
#  GRAPHIQUE
# =============================================================
COL_REF = "#2e6fb0"
COL_ALT = "#c0392b"
COL_MES = "#1f8a70"
COL_GRID = "#e5e7eb"
COL_TXT = "#374151"


def build_figure(t_real, ref, alt=None, point=None, mesure=None, t28_real=None, titre=""):
    """
    ref / alt : dict(label, y) — courbes fck(t réels).
    point     : (t, fck) sélectionné sur la référence.
    mesure    : dict(fck, t_est) — résistance mesurée + âge estimé (optionnels).
    t28_real  : jours réels correspondant à t_e = 28 j (fin de montée).
    """
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    # Courbe de référence + remplissage léger
    ax.plot(t_real, ref["y"], color=COL_REF, lw=2.4, label=ref["label"], zorder=3)
    ax.fill_between(t_real, ref["y"], ref["y"].min(), color=COL_REF, alpha=0.06, zorder=1)

    # Courbe comparée
    if alt is not None:
        ax.plot(t_real, alt["y"], color=COL_ALT, lw=2.0, ls="-", alpha=0.9,
                label=alt["label"], zorder=3)

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

    # Mesure éventuelle
    if mesure is not None and mesure.get("fck", 0) > 0:
        ax.axhline(mesure["fck"], color=COL_MES, lw=1.3, ls=":", zorder=2,
                   label=f"Mesure : {mesure['fck']:.1f} MPa")
        if mesure.get("t_est"):
            ax.scatter([mesure["t_est"]], [mesure["fck"]], s=45, marker="D",
                       color=COL_MES, edgecolor="white", lw=1.2, zorder=5)
            ax.annotate(f"âge estimé ≈ {mesure['t_est']:.1f} j",
                        xy=(mesure["t_est"], mesure["fck"]), xytext=(8, -14),
                        textcoords="offset points", fontsize=8.5, color=COL_MES)

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
#  PAGE
# =============================================================
def show():
    st.markdown("## Évolution de la résistance du béton — EC2")
    st.caption("fck(t) selon EN 1992-1-1 §3.1.2, effet de la température par âge équivalent (annexe B).")

    col_g, col_d = st.columns([1, 1.4])

    # -------------------- Paramètres --------------------
    with col_g:
        with st.container(border=True):
            st.markdown("#### Béton de référence")
            c1, c2 = st.columns([1.4, 1])
            with c1:
                beton_label = st.selectbox("Classe de béton", CLASSES, index=2)
                fck28 = parse_fck(beton_label)
            with c2:
                T = st.number_input("Température (°C)", value=20.0, step=1.0, format="%.1f",
                                    min_value=-5.0, max_value=60.0,
                                    help="Température moyenne de cure, supposée constante.")

            ciment = st.selectbox("Classe de résistance du ciment", list(CIMENTS.keys()), index=1,
                                  help="C'est la classe de résistance du ciment (R/N/S) qui fixe s dans l'EC2, "
                                       "pas le type CEM.")
            st.caption(f"Ciments concernés : {CIMENT_DETAIL[ciment]} — s = {CIMENTS[ciment]:.2f}")
            s_ref = CIMENTS[ciment]

            t_sel = st.slider("Âge du béton (jours réels)", 1, 40, 14)

            with st.expander("⚙️ Avancé — énergie d'activation"):
                Ea_kJ = st.number_input("Ea [kJ/mol]", value=EA_EC2 / 1000.0, step=1.0, format="%.1f",
                                        min_value=20.0, max_value=60.0,
                                        help="EC2 annexe B (éq. B.10) ⇔ 4000 K ⇔ 33,3 kJ/mol. "
                                             "Ajuster uniquement si calibré sur essais.")
                Ea = Ea_kJ * 1000.0

        # Calcul du point sélectionné
        t_e_sel = float(age_equiv(t_sel, T, Ea))
        fck_val = float(fck_of_age_equiv(fck28, s_ref, t_e_sel))

        etat = "ok"
        if t_e_sel <= 3.0:
            etat = "warn"
        _bloc(f"fck({t_sel} j à {T:.0f} °C)", f"{fck_val:.2f} MPa", etat)
        st.caption(f"Âge équivalent à 20 °C : t_e = {t_e_sel:.1f} j"
                   + ("  —  ⚠️ t_e ≤ 3 j : hors domaine de validité EC2, essais requis." if t_e_sel <= 3.0 else ""))

        res_mesuree = st.number_input("Résistance mesurée (MPa, optionnel)", min_value=0.0,
                                      value=0.0, step=0.5, format="%.2f",
                                      help="Estime l'âge réel correspondant sur la courbe de référence.")

        # -------------------- Comparateur --------------------
        with st.expander("🔁 Comparateur d'équivalence", expanded=False):
            st.caption("Quand une autre classe atteint-elle la même résistance, à la même température ?")
            alt_label = st.selectbox("Classe comparée", CLASSES, index=0)
            ciment_alt = st.selectbox("Ciment (classe comparée)", list(CIMENTS.keys()),
                                      index=list(CIMENTS.keys()).index(ciment))
            s_alt = CIMENTS[ciment_alt]
            fck28_alt = parse_fck(alt_label)

            target = float(res_mesuree) if res_mesuree > 0 else fck_val
            t_eq = t_real_for_target(fck28_alt, s_alt, target, T, Ea)
            if t_eq is not None:
                _bloc(f"{alt_label} atteint {target:.1f} MPa",
                      f"vers {t_eq:.1f} j à {T:.0f} °C", "ok")
            else:
                _bloc(f"{alt_label} n'atteint pas {target:.1f} MPa",
                      f"fck(28) = {fck28_alt} MPa", "nok")
            show_alt = st.checkbox("Afficher sur le graphique", value=True)

    # -------------------- Courbes --------------------
    t_real = np.linspace(1, 40, 500)
    y_ref = fck_of_age_equiv(fck28, s_ref, age_equiv(t_real, T, Ea))
    ref = {"label": f"{beton_label} — {ciment.split(' — ')[0]}", "y": y_ref}

    alt = None
    if show_alt:
        y_alt = fck_of_age_equiv(fck28_alt, s_alt, age_equiv(t_real, T, Ea))
        alt = {"label": f"{alt_label} — {ciment_alt.split(' — ')[0]}", "y": y_alt}

    mesure = None
    if 0 < res_mesuree <= fck28 + 1e-9:
        t_est = t_real_for_target(fck28, s_ref, float(res_mesuree), T, Ea)
        mesure = {"fck": float(res_mesuree), "t_est": t_est}
    elif res_mesuree > fck28 + 1e-9:
        mesure = {"fck": float(res_mesuree), "t_est": None}

    # jours réels correspondant à t_e = 28 j
    factor = float(age_equiv(1.0, T, Ea))
    t28_real = 28.0 / factor if factor > 0 else None

    fig = build_figure(
        t_real, ref, alt=alt, point=(t_sel, fck_val), mesure=mesure, t28_real=t28_real,
        titre=f"fck(t) — {beton_label} — {T:.0f} °C",
    )

    with col_d:
        st.pyplot(fig, use_container_width=True)
        if res_mesuree > fck28 + 1e-9:
            _bloc("Mesure > fck(28) de la référence", "âge non estimable par cette loi", "warn")

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


if __name__ == "__main__":
    st.set_page_config(page_title="Évolution fck(t) — EC2", page_icon="🧱", layout="wide")
    show()

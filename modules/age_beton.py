import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# ==============================
# Utilitaires EC2 + Température
# ==============================

CLASSES = ["C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60"]

# Tes valeurs s (tu les avais dans cet ordre)
CIMENT_S = {
    "prise rapide": 0.20,
    "prise normale": 0.25,
    "prise lente":  0.38,
}

# Recode informatif simpliste (comme demandé)
RECODE_CIMENT = {
    "prise rapide": "≈ CEM I R / CEM II R (hydratation rapide)",
    "prise normale": "≈ CEM I / CEM II (hydratation normale)",
    "prise lente":  "≈ CEM III/IV (hydratation lente)",
}

R_GAZ = 8.314       # J/(mol·K)
EA_DEFAULT = 40000  # J/mol  (valeur typique ; ajuste si tu calibres)

def parse_fck(label: str) -> int:
    return int(label.split("/")[0].replace("C", ""))

def beta_cc(t_days_equiv, s: float):
    """
    Loi EC2 : βcc(t) = exp( s * (1 - sqrt(28/t)) )
    t_days_equiv = âge équivalent (en jours) à 20°C.
    """
    t = np.asarray(t_days_equiv, dtype=float)
    # Évite division par zéro
    t = np.maximum(t, 1e-6)
    return np.exp(s * (1.0 - np.sqrt(28.0 / t)))

def fck_of_age_equiv(fck28: float, s: float, t_days_equiv):
    """
    fck(t) = βcc(t_e)*fcm - 8  avec  fcm = fck + 8
    et plafonné à fck28 pour t_e >= 28 j.
    """
    fcm = fck28 + 8.0
    t_e = np.asarray(t_days_equiv, dtype=float)
    val = beta_cc(t_e, s) * fcm - 8.0
    return np.where(t_e < 28.0, val, fck28)

def age_equiv_arrhenius(t_days_real, T_celsius, Ea=EA_DEFAULT):
    """
    Âge équivalent (en jours) à 20°C pour une température constante T.
    t_e = t * exp( -Ea/R * (1/(T+273.15) - 1/293.15) )
    Accepte scalaire ou vecteur pour t_days_real.
    """
    T_abs = float(T_celsius) + 273.15
    T_ref = 293.15
    factor = math.exp((-Ea / R_GAZ) * ((1.0 / T_abs) - (1.0 / T_ref)))
    return np.asarray(t_days_real, dtype=float) * factor

def t_equivalent_for_target_with_T(fck28: float, s: float, target_MPa: float,
                                   T_celsius: float, tmax=90.0, tol=1e-3):
    """
    Trouve le temps réel t (à température T_celsius) tel que
    fck_of_age_equiv(fck28, s, age_equiv_arrhenius(t, T)) == target_MPa.
    Renvoie None si cible > fck28 ou si non atteint < tmax.
    """
    if target_MPa <= 0 or target_MPa > fck28 + 1e-9:
        return None

    lo, hi = 0.05, float(tmax)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        t_e_mid = age_equiv_arrhenius(mid, T_celsius)
        f_mid = float(fck_of_age_equiv(fck28, s, t_e_mid))
        if abs(f_mid - target_MPa) < tol:
            return mid
        if f_mid < target_MPa:
            lo = mid
        else:
            hi = mid

    # Vérifie dernière valeur
    t_e_hi = age_equiv_arrhenius(hi, T_celsius)
    if float(fck_of_age_equiv(fck28, s, t_e_hi)) + 1e-6 < target_MPa:
        return None
    return 0.5 * (lo + hi)

# ==============================
# Page
# ==============================

def show():
    st.markdown("## Évolution de la résistance du béton selon l'EC2")

    # Bouton Accueil (si tu l'utilises déjà)
    if st.button("🏠 Accueil", use_container_width=True, key="btn_accueil_age"):
        st.session_state.retour_accueil_demande = True
        st.rerun()

    col_g, col_d = st.columns([1, 1.4])

    # --------- Paramètres béton de référence (colonne gauche)
    with col_g:
        # Même ligne : classe béton + température
        c1, c2 = st.columns([2, 1])
        with c1:
            beton_label = st.selectbox("Choisir un type de béton (référence) :", CLASSES, index=0)
            fck28_ref = parse_fck(beton_label)
        with c2:
            temperature_c = st.number_input("Température (°C)", value=20.0, step=1.0, format="%.1f")

        type_ciment = st.selectbox("Choisir le type de ciment :", list(CIMENT_S.keys()), index=0)
        st.caption(f"Type (recode) : {RECODE_CIMENT.get(type_ciment, '')}")
        s_ref = CIMENT_S[type_ciment]

        # Âge réel sélectionné
        t_selected_real = st.slider("Âge du béton (en jours)", 1, 40, 14)

        # >>> Calcul : on passe par l'âge équivalent t_e(T)
        t_selected_equiv = float(age_equiv_arrhenius(t_selected_real, temperature_c))
        fck_val = float(fck_of_age_equiv(fck28_ref, s_ref, t_selected_equiv))

        # Affichage immédiat (même taille, en gras, texte inchangé)
        st.markdown(
            f"""
            <div style="margin-top:0.25rem">
              <div style="font-weight:700; font-size:1.1rem;">Résistance calculée fck(t)</div>
              <div style="font-weight:700; font-size:1.1rem;">{fck_val:.2f} MPa</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        res_mesuree = st.number_input(
            "Résistance mesurée (MPa, optionnel) :",
            min_value=0.0, value=0.0, step=0.1, format="%.2f"
        )

    # --------- Courbe de référence (dépend de T)
    # Axe en jours RÉELS (ce que tu manipules dans l'UI)
    t_real = np.linspace(1, 40, 500)
    t_equiv_curve = age_equiv_arrhenius(t_real, temperature_c)   # transforme en âge équivalent
    fck_curve_ref = fck_of_age_equiv(fck28_ref, s_ref, t_equiv_curve)

    # Éventuelle estimation d'âge (réel) depuis une mesure à la même T
    estimated_age_real = None
    if 0 < res_mesuree <= fck28_ref + 1e-9:
        # On cherche t_real tel que fck(t_e(t_real,T)) = res_mesuree
        estimated_age_real = t_equivalent_for_target_with_T(
            fck28=fck28_ref, s=s_ref, target_MPa=float(res_mesuree),
            T_celsius=temperature_c, tmax=90.0
        )

    # --------- Graphe (colonne droite)
    with col_d:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t_real, fck_curve_ref, label=f"{beton_label} ({type_ciment})", linewidth=2)
        ax.axvline(x=t_selected_real, linestyle='--', label=f"{t_selected_real} j")
        ax.axhline(y=fck_val, linestyle='--', label=f"fck = {fck_val:.2f} MPa")

        if res_mesuree > 0:
            ax.axhline(y=res_mesuree, linestyle=':', label=f"Mesure {res_mesuree:.2f} MPa")
            if estimated_age_real:
                ax.axvline(x=estimated_age_real, linestyle=':', label=f"Âge estimé {estimated_age_real:.1f} j")

    # --------- COMPARATEUR (UNE SEULE CLASSE)
    with col_g:
        st.divider()
        st.markdown("### Comparateur d'équivalence")

        # Cible = fck de la référence au jour sélectionné (ou la mesure si fournie)
        target = float(res_mesuree) if res_mesuree > 0 else fck_val

        alt_label = st.selectbox("Comparer avec :", CLASSES, index=2)
        type_ciment_alt = st.selectbox(
            "Ciment (classe comparée) :",
            list(CIMENT_S.keys()),
            index=list(CIMENT_S.keys()).index(type_ciment)
        )
        s_alt = CIMENT_S[type_ciment_alt]
        fck28_alt = parse_fck(alt_label)

        # Temps réel requis à la MÊME température pour atteindre la même résistance
        t_eq_real = t_equivalent_for_target_with_T(
            fck28=fck28_alt, s=s_alt, target_MPa=target,
            T_celsius=temperature_c, tmax=90.0
        )

        if t_eq_real is not None:
            st.success(f"{alt_label} ({type_ciment_alt}) atteint ≈ {target:.2f} MPa vers **{t_eq_real:.1f} j** à {temperature_c:.1f} °C.")
        else:
            st.warning(f"{alt_label} n’atteint pas {target:.2f} MPa (≤ fck(28) = {fck28_alt} MPa) aux conditions choisies.")

        # Ajoute la courbe comparée côté graphe (même température)
        with col_d:
            fck_curve_alt = fck_of_age_equiv(fck28_alt, s_alt, age_equiv_arrhenius(t_real, temperature_c))
            ax.plot(t_real, fck_curve_alt, label=f"{alt_label} ({type_ciment_alt})", linewidth=2)
            if t_eq_real is not None:
                ax.axvline(x=t_eq_real, linestyle='--', alpha=0.8, label=f"t_eq {alt_label} ≈ {t_eq_real:.1f} j")

    # --------- Finalisation graphe
    ax.set_xlabel("Âge du béton (jours réels)")
    ax.set_ylabel("Résistance fck(t) [MPa]")
    ax.set_title(f"Évolution de la résistance — {beton_label} — {type_ciment} — {temperature_c:.1f} °C")
    ax.grid(True)
    ax.legend()

    with col_d:
        st.pyplot(fig)

    # --------- Bloc court “Référence”
    with col_g:
        st.markdown("### Référence")
        st.markdown(
            f"**{beton_label}** ({type_ciment}) — {temperature_c:.1f} °C → "
            f"fck({t_selected_real} j réels) = **{fck_val:.2f} MPa**."
        )
        st.latex(
            r"f_{ck}(t)=\beta_{cc}(t)\,f_{cm}-8,\quad f_{cm}=f_{ck}+8,\quad "
            r"\beta_{cc}(t)=\exp\!\big(s\,\big[1-\sqrt{28/t}\big]\big),\ "
            r"t\ \equiv\ t_e(T)\ \text{(âge équivalent)}"
        )

# Si tu appelles directement ce module :
if __name__ == "__main__":
    st.set_page_config(page_title="Évolution fck(t) — EC2", page_icon="🧱", layout="wide")
    show()

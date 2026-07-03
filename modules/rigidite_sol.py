# =============================================================
#  raideur_sol.py — Raideur élastique des sols (modèle de Winkler)
#  VERSION 3.0 — refonte complète
#
#  Objectifs de la refonte :
#   - Calculs séparés du rendu : fonctions pures testables, aucun
#     effet de bord Streamlit dans le calcul.
#   - Physique corrigée / clarifiée :
#       * Cas 2 : k_serie (ressorts en série, tassement 1D d'une
#         colonne de sol) ET k_Boussinesq (semelle sur massif semi-
#         infini) sont deux MODÈLES DISTINCTS, jamais chaînés. Chacun
#         est affiché avec son domaine de validité.
#       * Cas 4 : contact plat/béton traité en compression 1D pure
#         (k = E/h). Le facteur (1−ν²) — valable pour un massif semi-
#         infini, pas pour une couche mince confinée — est proposé en
#         option explicite et non plus imposé.
#   - suggest_E_from_qc n'écrase plus silencieusement E : une colonne
#     "E suggéré" informe, l'utilisateur remplit "E" s'il le souhaite.
#   - set_page_config retiré (géré par l'app principale).
#   - State centralisé + defaults ; conversions d'unités robustes.
#
#  Winkler : q = k · w  ->  k = q / w
#    q [kPa = kN/m²], w [m], k [kN/m³]  (1 MN/m³ = 1000 kN/m³)
# =============================================================

import math
import pandas as pd
import streamlit as st


# =============================================================
#  CONSTANTES
# =============================================================
KGF_PER_CM2_TO_KPA = 98.0665          # 1 kgf/cm² = 98.0665 kPa
VERSION = "v3.0"

C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6", "info": "#eef2ff"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌", "info": "ℹ️"}

SOIL_TYPES = [
    "—",
    "Tourbe",
    "Argile très molle",
    "Argile molle à moyenne",
    "Argile ferme / raide",
    "Limon",
    "Sable lâche",
    "Sable moyennement compact",
    "Sable dense",
    "Sable graveleux / grave compacte",
    "Roche altérée",
    "Roche saine",
    "Personnalisé",
]

# Corrélation indicative qc -> E (E ≈ α·qc), α selon le type de sol.
_ALPHA_QC = {
    "Tourbe": 4.0,
    "Argile très molle": 4.0,
    "Argile molle": 5.0,
    "Argile ferme": 6.0,
    "Limon": 4.0,
    "Sable lâche": 3.5,
    "Sable moyennement compact": 5.0,
    "Sable dense": 6.0,
    "grave": 4.0,
    "Roche": 2.0,
}


# =============================================================
#  CONVERSIONS D'UNITÉS (fonctions pures)
# =============================================================
def to_kPa(value: float, unit: str) -> float:
    """Pression -> kPa."""
    return {"kPa": value, "MPa": value * 1000.0, "kg/cm²": value * KGF_PER_CM2_TO_KPA}.get(unit, value)


def from_kPa(value_kPa: float, unit: str) -> float:
    """kPa -> unité cible."""
    return {"kPa": value_kPa, "MPa": value_kPa / 1000.0, "kg/cm²": value_kPa / KGF_PER_CM2_TO_KPA}.get(unit, value_kPa)


def E_to_kPa(E: float, unit: str) -> float:
    """Module E (MPa ou GPa) -> kPa."""
    return {"MPa": E * 1000.0, "GPa": E * 1_000_000.0}.get(unit, E)


def kNpm3_to_MNpm3(v: float) -> float:
    return v / 1000.0


def suggest_E_from_qc(qc_MPa, soil_type: str):
    """E ≈ α·qc (MPa). Retourne None si qc invalide."""
    if qc_MPa is None or (isinstance(qc_MPa, float) and math.isnan(qc_MPa)) or qc_MPa <= 0:
        return None
    alpha = 3.0
    for key, a in _ALPHA_QC.items():
        if key.lower() in (soil_type or "").lower():
            alpha = a
            break
    return round(alpha * qc_MPa, 1)


# =============================================================
#  CALCULS (fonctions pures — aucun Streamlit)
# =============================================================
def k_from_qw(q_kPa: float, w_mm: float):
    """Winkler direct : k = q / w. Retourne (k_kNpm3, k_MNpm3, w_m)."""
    w_m = w_mm / 1000.0
    if w_m <= 0:
        return 0.0, 0.0, w_m
    k = q_kPa / w_m
    return k, kNpm3_to_MNpm3(k), w_m


def k_series(layers):
    """
    Ressorts en série (tassement 1D d'une colonne de sol chargée) :
        1/k_serie = Σ h_i / E_i
    layers : liste de (h_m, E_kPa) valides (>0).
    Retourne (k_kNpm3, k_MNpm3, H_m, E_moy_kPa).
    """
    denom = 0.0
    H = 0.0
    for h, E in layers:
        if h > 0 and E > 0:
            denom += h / E
            H += h
    if denom <= 0:
        return 0.0, 0.0, H, 0.0
    k = 1.0 / denom
    E_moy = k * H  # module oedométrique équivalent d'une colonne d'épaisseur H
    return k, kNpm3_to_MNpm3(k), H, E_moy


def k_boussinesq(E_kPa: float, B_m: float, nu: float):
    """
    Semelle rigide sur massif élastique semi-infini (ordre de grandeur) :
        k ≈ E / [B (1 − ν²)]
    Retourne (k_kNpm3, k_MNpm3).
    """
    if E_kPa <= 0 or B_m <= 0 or nu >= 1.0:
        return 0.0, 0.0
    k = E_kPa / (B_m * (1.0 - nu ** 2))
    return k, kNpm3_to_MNpm3(k)


def E_from_cpt(qt_MPa: float, sv0_kPa: float, alpha_E: float):
    """E = α_E (qt − σ'v0). Retourne (E_kPa, E_MPa, delta_kPa)."""
    delta = max(qt_MPa * 1000.0 - sv0_kPa, 0.0)
    E_kPa = alpha_E * delta
    return E_kPa, E_kPa / 1000.0, delta


def k_plate(B_mm, L_mm, alpha, Ec_GPa, use_nu, nu_c,
            has_grout=False, tg_mm=0.0, Eg_GPa=20.0):
    """
    Contact plat/béton. Épaisseur mobilisée h_c = α·min(B,L).
    Compression 1D pure : k = E / h_c. Option (1−ν²) si l'utilisateur
    l'assume (rarement justifié pour une couche mince confinée).
    Grout en série : 1/k_eq = 1/k_c + 1/k_g.
    Retourne un dict de résultats intermédiaires + k_eq.
    """
    B = B_mm / 1000.0
    L = L_mm / 1000.0
    hc = alpha * min(B, L)
    Ec_kPa = E_to_kPa(Ec_GPa, "GPa")

    if hc <= 0:
        return {"hc": 0.0, "kc": 0.0, "kg": 0.0, "keq_kNpm3": 0.0, "keq_MNpm3": 0.0}

    fac = (1.0 - nu_c ** 2) if use_nu else 1.0
    kc = Ec_kPa / (hc * fac)

    keq = kc
    kg = 0.0
    if has_grout and tg_mm > 0:
        tg = tg_mm / 1000.0
        Eg_kPa = E_to_kPa(Eg_GPa, "GPa")
        kg = Eg_kPa / tg if tg > 0 else 0.0
        if kc > 0 and kg > 0:
            keq = 1.0 / (1.0 / kc + 1.0 / kg)

    return {"hc": hc, "kc": kc, "kg": kg,
            "keq_kNpm3": keq, "keq_MNpm3": kNpm3_to_MNpm3(keq)}


# =============================================================
#  RENDU : helpers UI
# =============================================================
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


def _param_table(rows):
    df = pd.DataFrame(rows, columns=["Paramètre", "Description", "Valeur", "Unité"])
    df.index = [""] * len(df)
    st.table(df)


def _metric(col, label, value, unit=""):
    col.metric(label, f"{value:,.2f}".replace(",", " ") + (f" {unit}" if unit else ""))


# =============================================================
#  STATE
# =============================================================
def _init_state():
    d = {
        "press_unit": "kPa",
        "module_unit": "MPa",
        "detail_calc": True,
        "adv_open": False,
        "abaque_w": 20.0,
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)

    if "layers_df" not in st.session_state:
        st.session_state.layers_df = pd.DataFrame(
            [{"h [m]": 2.0, "Type de sol": "Sable moyennement compact",
              "qc moy [MPa]": 6.0, "Rf [%]": 1.0, "E [MPa]": 30.0}],
            index=[1],
        )


# =============================================================
#  PAGE
# =============================================================
def show():
    _init_state()

    st.markdown(
        "<style>.katex-display{text-align:left!important;margin:.2rem 0!important;}"
        ".katex-display>.katex{text-align:left!important;}"
        ".memo-chip{display:inline-block;padding:2px 8px;border-radius:999px;"
        "background:#eef2ff;color:#3730a3;font-size:.8rem;}"
        ".small{color:#64748b;font-size:.9rem;}</style>",
        unsafe_allow_html=True,
    )

    # ---------- Barre du haut ----------
    cols = st.columns([1, 1, 1, 1, 1, 1])
    with cols[0]:
        if st.button("🏠 Accueil", use_container_width=True, key="rs_home"):
            st.session_state.page = "Accueil"
            st.rerun()
    with cols[1]:
        if st.button("🧹 Réinitialiser", use_container_width=True, key="rs_reset"):
            keep = {"press_unit", "module_unit", "detail_calc", "adv_open", "page", "abaque_w"}
            for k in list(st.session_state.keys()):
                if k not in keep:
                    st.session_state.pop(k, None)
            st.rerun()
    with cols[2]:
        st.button("💾 Enregistrer", use_container_width=True, disabled=True, help="À connecter au système JSON")
    with cols[3]:
        st.button("📂 Ouvrir", use_container_width=True, disabled=True, help="Lecture de fichiers à venir")
    with cols[4]:
        st.button("📝 Générer PDF", use_container_width=True, disabled=True, help="Export PDF à développer")
    with cols[5]:
        st.markdown(f"<div style='text-align:right;padding-top:10px;'><span class='memo-chip'>{VERSION}</span></div>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("# Raideur élastique des sols")
    st.markdown("<span class='small'>Pré-dimensionnement — sols modélisés par des ressorts verticaux "
                "(modèle de Winkler).</span>", unsafe_allow_html=True)

    with st.expander("📘 Fiche mémo (k, unités et modèle de Winkler)", expanded=False):
        st.markdown(
            r"""
- **Winkler** : $q = k \cdot w \Rightarrow k = q/w$.
- **Unités** : $q$ en kPa = kN/m² · $w$ en m · $k$ en kN/m³ ou MN/m³ (1 MN/m³ = 1000 kN/m³).
- **Ressorts en série** (colonne de sol) : $1/k_{serie} = \sum_i h_i/E_i$.
- **Semelle sur massif semi-infini** (Boussinesq, ordre de grandeur) : $k \approx E/[B(1-\nu^2)]$.
  Ces deux modèles répondent à des questions différentes et **ne se chaînent pas**.
- **Contrainte admissible** pour un tassement de référence $w_{adm}$ :
  $q_{adm} = k \cdot w_{adm}$ ; en kgf/cm² : $q_{adm} \approx k(\text{MN/m}^3)\cdot w_{adm}(\text{mm})/98{,}07$.
- Valeurs à valider par l'**EN 1997 (Eurocode 7)** et le rapport géotechnique.
            """
        )

    col_left, col_right = st.columns(2)

    # =========================================================
    #  COLONNE GAUCHE — entrées
    # =========================================================
    with col_left:
        st.markdown("### Informations et entrées")

        st.session_state.adv_open = st.checkbox("⚙️ Configuration avancée (unités)",
                                                value=st.session_state.adv_open)
        if st.session_state.adv_open:
            c1, c2 = st.columns(2)
            with c1:
                old = st.session_state.press_unit
                new = st.selectbox("Pressions / contraintes", ["kPa", "MPa", "kg/cm²"],
                                   index=["kPa", "MPa", "kg/cm²"].index(old))
                if new != old:
                    for kk in ("solo_q", "solo_qad"):
                        if kk in st.session_state:
                            st.session_state[kk] = from_kPa(to_kPa(st.session_state[kk], old), new)
                    st.session_state.press_unit = new
            with c2:
                oldm = st.session_state.module_unit
                newm = st.selectbox("Modules E", ["MPa", "GPa"], index=0 if oldm == "MPa" else 1)
                if newm != oldm and "solo_E" in st.session_state:
                    st.session_state.solo_E *= (0.001 if newm == "GPa" else 1000.0)
                st.session_state.module_unit = newm

        cas = st.selectbox(
            "Quel cas souhaitez-vous traiter ?",
            ("1. Raideur d'un sol (q, w)",
             "2. Modélisation d'un sol (mono / multicouche / CPT interprété)",
             "3. Raideur d'un sol – formule empirique (CPT)",
             "4. Raideur d'un plat en béton",
             "5. Convertisseur k ↔ E ↔ (q, w)",
             "6. Abaque sols"),
            index=0,
        )

        pu = st.session_state.press_unit

        if cas.startswith("1."):
            st.markdown("**Raideur à partir d'un couple (q, w)**")
            st.caption("On connaît une pression de service q et un tassement w : k = q / w.")
            st.markdown("<span class='memo-chip'>Typiquement : q à l'ELS, w = 20 mm → k pour SCIA / RDM.</span>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.solo_q = st.number_input(f"q (pression au sol) [{pu}]", min_value=0.0,
                                                          value=float(st.session_state.get("solo_q", 60.0)), step=5.0)
            with c2:
                st.session_state.solo_w = st.number_input("w (tassement) [mm]", min_value=0.001,
                                                          value=float(st.session_state.get("solo_w", 20.0)), step=5.0)

        elif cas.startswith("2."):
            st.markdown("**Modélisation d'un sol (mono / multicouche / CPT interprété)**")
            st.caption("Une ligne = sol homogène. Plusieurs lignes = profil multicouche. "
                       "Calcul en ressorts en série : 1/k = Σ(hᵢ/Eᵢ).")
            cB, cNu = st.columns(2)
            with cB:
                st.session_state.multi_B = st.number_input("Largeur caractéristique B [m]", min_value=0.1,
                                                          value=float(st.session_state.get("multi_B", 2.0)), step=0.1,
                                                          help="Pour l'approximation Boussinesq k ≈ E/[B(1−ν²)].")
            with cNu:
                st.session_state.multi_nu = st.number_input("ν équivalent (Poisson)", min_value=0.0, max_value=0.49,
                                                          value=float(st.session_state.get("multi_nu", 0.30)), step=0.01)
            st.markdown("<span class='memo-chip'>Colonne « E suggéré » = α·qc, indicatif. "
                        "Remplis « E [MPa] » pour l'utiliser dans le calcul.</span>", unsafe_allow_html=True)

            df = st.session_state.layers_df.copy()
            col_cfg = {
                "h [m]": st.column_config.NumberColumn("h [m]", step=0.1, min_value=0.0),
                "Type de sol": st.column_config.SelectboxColumn("Type de sol", options=SOIL_TYPES, required=False, width="medium"),
                "qc moy [MPa]": st.column_config.NumberColumn("qc moy [MPa]", step=0.5, min_value=0.0),
                "Rf [%]": st.column_config.NumberColumn("Rf [%]", step=0.5, min_value=0.0),
                "E [MPa]": st.column_config.NumberColumn("E [MPa]", step=5.0, min_value=0.0),
            }
            st.markdown("#### Couches de sol")
            edited = st.data_editor(df, key="rs_layers_editor", num_rows="dynamic",
                                    use_container_width=True, column_config=col_cfg)

            # E suggéré = colonne informative (n'écrase pas E)
            if len(edited) > 0:
                edited = edited.copy()
                edited["E suggéré [MPa]"] = [
                    suggest_E_from_qc(r.get("qc moy [MPa]"), r.get("Type de sol") or "")
                    for _, r in edited.iterrows()
                ]
                edited.index = range(1, len(edited) + 1)
                st.dataframe(edited[["Type de sol", "qc moy [MPa]", "E [MPa]", "E suggéré [MPa]"]],
                             use_container_width=True)
                st.session_state.layers_df = edited.drop(columns=["E suggéré [MPa]"])

        elif cas.startswith("3."):
            st.markdown("**Raideur d'un sol – formule empirique (CPT)**")
            st.caption("Basée sur une valeur de qc : E = α_E (qₜ − σ'ᵥ₀), puis k ≈ E/[B(1−ν²)]. Sol supposé homogène.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.cpt_qt = st.number_input("qₜ (pointe nette) [MPa]", min_value=0.0,
                                                          value=float(st.session_state.get("cpt_qt", 5.0)), step=0.5)
            with c2:
                st.session_state.cpt_sv0 = st.number_input("σ'ᵥ₀ (contrainte eff.) [kPa]", min_value=0.0,
                                                          value=float(st.session_state.get("cpt_sv0", 100.0)), step=10.0)
            with c3:
                st.session_state.cpt_alphaE = st.number_input("α_E (CPT → E)", min_value=0.1,
                                                          value=float(st.session_state.get("cpt_alphaE", 2.5)), step=0.1)
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.cpt_B = st.number_input("B (largeur) [m]", min_value=0.1,
                                                          value=float(st.session_state.get("cpt_B", 2.0)), step=0.1)
            with c5:
                st.session_state.cpt_nu = st.number_input("ν (Poisson)", min_value=0.0, max_value=0.49,
                                                          value=float(st.session_state.get("cpt_nu", 0.30)), step=0.01)

        elif cas.startswith("4."):
            st.markdown("**Raideur d'un plat en béton (contact plat / béton / grout)**")
            st.caption("Contact assimilé à une compression 1D du béton (et du grout). Par défaut k = E/h.")
            st.markdown("**Géométrie du plat**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.plate_B = st.number_input("Largeur plat B [mm]", min_value=20.0,
                                                          value=float(st.session_state.get("plate_B", 200.0)), step=10.0)
            with c2:
                st.session_state.plate_L = st.number_input("Longueur plat L [mm]", min_value=20.0,
                                                          value=float(st.session_state.get("plate_L", 200.0)), step=10.0)
            with c3:
                st.session_state.plate_alpha = st.number_input("α (h_c = α·min(B,L))", min_value=0.05,
                                                          value=float(st.session_state.get("plate_alpha", 0.5)), step=0.05)
            st.markdown("**Béton support**")
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.plate_Ec = st.number_input("E_c béton [GPa]", min_value=5.0,
                                                          value=float(st.session_state.get("plate_Ec", 30.0)), step=1.0)
            with c5:
                st.session_state.plate_use_nu = st.checkbox("Appliquer le facteur (1−ν²)",
                                                          value=st.session_state.get("plate_use_nu", False),
                                                          help="Valable pour un massif semi-infini, "
                                                               "rarement justifié pour une couche mince confinée.")
            if st.session_state.plate_use_nu:
                st.session_state.plate_nu = st.number_input("ν béton", min_value=0.0, max_value=0.49,
                                                          value=float(st.session_state.get("plate_nu", 0.20)), step=0.01)
            else:
                st.session_state.plate_nu = st.session_state.get("plate_nu", 0.20)

            st.markdown("**Lit de mortier / grout (optionnel)**")
            st.session_state.plate_has_grout = st.checkbox("Présence d'un lit de mortier/grout",
                                                          value=st.session_state.get("plate_has_grout", False))
            if st.session_state.plate_has_grout:
                c6, c7 = st.columns(2)
                with c6:
                    st.session_state.plate_tg = st.number_input("Épaisseur grout t_g [mm]", min_value=1.0,
                                                          value=float(st.session_state.get("plate_tg", 20.0)), step=1.0)
                with c7:
                    st.session_state.plate_Eg = st.number_input("E_g grout [GPa]", min_value=5.0,
                                                          value=float(st.session_state.get("plate_Eg", 20.0)), step=1.0)
            else:
                st.session_state.plate_tg = st.session_state.get("plate_tg", 0.0)
                st.session_state.plate_Eg = st.session_state.get("plate_Eg", 20.0)

        elif cas.startswith("5."):
            st.markdown("**Convertisseur k ↔ E ↔ (q, w)**")
            st.caption("Choisis ce que tu connais ; l'outil déduit le reste.")
            st.session_state.conv_mode = st.radio(
                "Je connais…",
                ["k → q (pour un tassement w)", "q, w → k", "E, B, ν → k (Boussinesq)"],
                index=["k → q (pour un tassement w)", "q, w → k", "E, B, ν → k (Boussinesq)"].index(
                    st.session_state.get("conv_mode", "q, w → k")),
            )
            m = st.session_state.conv_mode
            if m.startswith("k →"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.conv_k = st.number_input("k [MN/m³]", min_value=0.0,
                                                          value=float(st.session_state.get("conv_k", 30.0)), step=1.0)
                with c2:
                    st.session_state.conv_w = st.number_input("w (tassement) [mm]", min_value=0.001,
                                                          value=float(st.session_state.get("conv_w", 20.0)), step=1.0)
            elif m.startswith("q,"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.conv_q = st.number_input(f"q [{pu}]", min_value=0.0,
                                                          value=float(st.session_state.get("conv_q", 60.0)), step=5.0)
                with c2:
                    st.session_state.conv_w = st.number_input("w [mm]", min_value=0.001,
                                                          value=float(st.session_state.get("conv_w", 20.0)), step=1.0)
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.conv_E = st.number_input(f"E [{st.session_state.module_unit}]", min_value=0.0,
                                                          value=float(st.session_state.get("conv_E", 30.0)), step=5.0)
                with c2:
                    st.session_state.conv_B = st.number_input("B [m]", min_value=0.1,
                                                          value=float(st.session_state.get("conv_B", 2.0)), step=0.1)
                with c3:
                    st.session_state.conv_nu = st.number_input("ν", min_value=0.0, max_value=0.49,
                                                          value=float(st.session_state.get("conv_nu", 0.30)), step=0.01)

        else:  # cas 6
            st.markdown("**Abaque sols – valeurs indicatives**")
            st.caption("Poids volumique γ, raideur k (MN/m³) et contrainte admissible qₐ (kg/cm²) "
                       "pour un tassement de référence. À confirmer par le géotechnicien.")

    # =========================================================
    #  COLONNE DROITE — résultats
    # =========================================================
    with col_right:
        st.markdown("### Dimensionnement / Résultats")
        st.session_state.detail_calc = st.checkbox("📘 Détail des calculs (formules + valeurs)",
                                                  value=st.session_state.detail_calc)
        detail = st.session_state.detail_calc
        pu = st.session_state.press_unit
        mu = st.session_state.module_unit

        # ---------- CAS 1 ----------
        if cas.startswith("1."):
            with st.container(border=True):
                q_kPa = to_kPa(st.session_state.get("solo_q", 0.0), pu)
                w_mm = st.session_state.get("solo_w", 20.0)
                k_kN, k_MN, w_m = k_from_qw(q_kPa, w_mm)
                etat = "ok" if k_MN > 0 else "nok"
                _bloc("Raideur de Winkler", f"k = {k_MN:,.2f} MN/m³".replace(",", " "), etat)
                if detail and k_MN > 0:
                    st.latex(r"k = \dfrac{q}{w}")
                    st.latex(f"k = \\dfrac{{{q_kPa:,.1f}}}{{{w_m:,.3f}}} "
                             f"= {k_kN:,.0f}\\,\\text{{kN/m³}} = {k_MN:,.2f}\\,\\text{{MN/m³}}")
                    _param_table([
                        ("q", "Pression de service", f"{st.session_state.get('solo_q', 0.0):,.2f}", pu),
                        ("w", "Tassement", f"{w_mm:,.2f}", "mm"),
                        ("k", "Raideur de sol", f"{k_MN:,.2f}", "MN/m³"),
                    ])

        # ---------- CAS 2 ----------
        elif cas.startswith("2."):
            with st.container(border=True):
                df = st.session_state.layers_df
                layers = []
                for _, r in df.iterrows():
                    h = r.get("h [m]")
                    E = r.get("E [MPa]")
                    if pd.notna(h) and pd.notna(E):
                        layers.append((float(h), E_to_kPa(float(E), "MPa")))
                k_kN, k_MN, H, E_moy_kPa = k_series(layers)

                _bloc("Ressorts en série (colonne 1D)", f"k = {k_MN:,.2f} MN/m³".replace(",", " "),
                      "ok" if k_MN > 0 else "nok")
                st.caption("Tassement d'une colonne de sol d'épaisseur H sous charge répartie.")
                if detail and k_MN > 0:
                    st.latex(r"k_{serie} = \left(\sum_i \dfrac{h_i}{E_i}\right)^{-1}")
                    st.latex(f"k_{{serie}} = {k_kN:,.0f}\\,\\text{{kN/m³}} = {k_MN:,.2f}\\,\\text{{MN/m³}}")
                    _param_table([
                        ("H", "Somme des épaisseurs", f"{H:,.2f}", "m"),
                        ("E_moy", "Module oedo. équivalent", f"{E_moy_kPa/1000:,.1f}", "MPa"),
                        ("k_serie", "Raideur (série)", f"{k_MN:,.2f}", "MN/m³"),
                    ])

                st.divider()
                B = st.session_state.get("multi_B", 2.0)
                nu = st.session_state.get("multi_nu", 0.30)
                kB_kN, kB_MN = k_boussinesq(E_moy_kPa, B, nu)
                _bloc("Semelle sur massif (Boussinesq)", f"k ≈ {kB_MN:,.2f} MN/m³".replace(",", " "),
                      "info" if kB_MN > 0 else "nok")
                st.caption("Approximation indépendante, valable pour une semelle sur massif semi-infini. "
                           "À ne pas confondre avec le modèle en série ci-dessus.")
                if detail and kB_MN > 0:
                    st.latex(r"k \approx \dfrac{E_{moy}}{B\,(1-\nu^2)}")
                    st.latex(f"k \\approx {kB_kN:,.0f}\\,\\text{{kN/m³}} = {kB_MN:,.2f}\\,\\text{{MN/m³}}")

        # ---------- CAS 3 ----------
        elif cas.startswith("3."):
            with st.container(border=True):
                qt = st.session_state.get("cpt_qt", 0.0)
                sv0 = st.session_state.get("cpt_sv0", 0.0)
                aE = st.session_state.get("cpt_alphaE", 2.5)
                E_kPa, E_MPa, delta = E_from_cpt(qt, sv0, aE)
                B = st.session_state.get("cpt_B", 2.0)
                nu = st.session_state.get("cpt_nu", 0.30)
                k_kN, k_MN = k_boussinesq(E_kPa, B, nu)
                _bloc("Raideur estimée (CPT)", f"k ≈ {k_MN:,.2f} MN/m³".replace(",", " "),
                      "ok" if k_MN > 0 else "nok")
                c1, c2 = st.columns(2)
                _metric(c1, "E estimé", E_MPa, "MPa")
                _metric(c2, "k", k_MN, "MN/m³")
                if detail and k_MN > 0:
                    st.latex(r"E = \alpha_E\,(q_t - \sigma'_{v0})")
                    st.latex(f"E = {aE:,.2f}\\,({qt*1000:,.0f}-{sv0:,.0f}) "
                             f"= {E_kPa:,.0f}\\,\\text{{kN/m²}} = {E_MPa:,.1f}\\,\\text{{MPa}}")
                    st.latex(r"k \approx \dfrac{E}{B(1-\nu^2)}")
                    st.latex(f"k \\approx {k_kN:,.0f}\\,\\text{{kN/m³}} = {k_MN:,.2f}\\,\\text{{MN/m³}}")

        # ---------- CAS 4 ----------
        elif cas.startswith("4."):
            with st.container(border=True):
                res = k_plate(
                    st.session_state.get("plate_B", 200.0), st.session_state.get("plate_L", 200.0),
                    st.session_state.get("plate_alpha", 0.5), st.session_state.get("plate_Ec", 30.0),
                    st.session_state.get("plate_use_nu", False), st.session_state.get("plate_nu", 0.20),
                    st.session_state.get("plate_has_grout", False),
                    st.session_state.get("plate_tg", 0.0), st.session_state.get("plate_Eg", 20.0),
                )
                _bloc("Raideur du contact plat/béton", f"k_eq = {res['keq_MNpm3']:,.1f} MN/m³".replace(",", " "),
                      "ok" if res["keq_MNpm3"] > 0 else "nok")
                if detail and res["hc"] > 0:
                    st.latex(r"h_c = \alpha\,\min(B,L)")
                    st.latex(f"h_c = {res['hc']*1000:,.1f}\\,\\text{{mm}}")
                    st.latex(r"k_c = \dfrac{E_c}{h_c}" +
                             (r"\,(1-\nu^2)^{-1}" if st.session_state.get("plate_use_nu") else ""))
                    st.latex(f"k_c = {res['kc']:,.0f}\\,\\text{{kN/m³}}")
                    if st.session_state.get("plate_has_grout") and res["kg"] > 0:
                        st.latex(r"\dfrac{1}{k_{eq}} = \dfrac{1}{k_c} + \dfrac{1}{k_g}")
                        st.latex(f"k_g = {res['kg']:,.0f}\\,\\text{{kN/m³}}")
                    _param_table([
                        ("h_c", "Épaisseur mobilisée", f"{res['hc']*1000:,.1f}", "mm"),
                        ("E_c", "Module béton", f"{st.session_state.get('plate_Ec',30.0):,.1f}", "GPa"),
                        ("k_eq", "Raideur équivalente", f"{res['keq_MNpm3']:,.1f}", "MN/m³"),
                    ])

        # ---------- CAS 5 : convertisseur ----------
        elif cas.startswith("5."):
            with st.container(border=True):
                m = st.session_state.get("conv_mode", "q, w → k")
                if m.startswith("k →"):
                    k_MN = st.session_state.get("conv_k", 0.0)
                    w_mm = st.session_state.get("conv_w", 20.0)
                    q_kPa = k_MN * 1000.0 * (w_mm / 1000.0)  # k[kN/m³]·w[m]
                    _bloc("Pression mobilisée", f"q = {from_kPa(q_kPa, pu):,.2f} {pu}".replace(",", " "),
                          "ok" if q_kPa > 0 else "nok")
                    if detail:
                        st.latex(r"q = k \cdot w")
                        st.latex(f"q = {k_MN:,.2f}\\cdot10^3 \\cdot {w_mm/1000:,.3f} "
                                 f"= {q_kPa:,.1f}\\,\\text{{kN/m²}} = {from_kPa(q_kPa,pu):,.2f}\\,\\text{{{pu}}}")
                elif m.startswith("q,"):
                    q_kPa = to_kPa(st.session_state.get("conv_q", 0.0), pu)
                    w_mm = st.session_state.get("conv_w", 20.0)
                    k_kN, k_MN, w_m = k_from_qw(q_kPa, w_mm)
                    _bloc("Raideur", f"k = {k_MN:,.2f} MN/m³".replace(",", " "), "ok" if k_MN > 0 else "nok")
                    if detail and k_MN > 0:
                        st.latex(r"k = q / w")
                        st.latex(f"k = {q_kPa:,.1f}/{w_m:,.3f} = {k_MN:,.2f}\\,\\text{{MN/m³}}")
                else:
                    E_kPa = E_to_kPa(st.session_state.get("conv_E", 0.0), mu)
                    B = st.session_state.get("conv_B", 2.0)
                    nu = st.session_state.get("conv_nu", 0.30)
                    k_kN, k_MN = k_boussinesq(E_kPa, B, nu)
                    _bloc("Raideur (Boussinesq)", f"k ≈ {k_MN:,.2f} MN/m³".replace(",", " "),
                          "ok" if k_MN > 0 else "nok")
                    if detail and k_MN > 0:
                        st.latex(r"k \approx \dfrac{E}{B(1-\nu^2)}")
                        st.latex(f"k \\approx {k_MN:,.2f}\\,\\text{{MN/m³}}")

        # ---------- CAS 6 : abaque ----------
        else:
            with st.container(border=True):
                st.markdown("#### Tassement de référence")
                st.session_state.abaque_w = st.number_input(
                    "w_adm [mm]", min_value=1.0, max_value=100.0,
                    value=float(st.session_state.abaque_w), step=5.0,
                    help="Convertit k (MN/m³) en qₐ (kg/cm²). En Belgique, 20 mm est courant en service.")
                w_adm = st.session_state.abaque_w
                factor_q = w_adm / KGF_PER_CM2_TO_KPA  # qₐ(kgf/cm²) ≈ k(MN/m³)·w(mm)/98.07

                soils = [
                    {"type": "Tourbe", "gamma": 10.0, "k_min": 1, "k_max": 5,
                     "desc": "Sol très organique, très compressible, souvent saturé, portance très faible. "
                             "On évite d'y fonder (remblais, pieux, substitution…)."},
                    {"type": "Argile très molle", "gamma": 16.0, "k_min": 2, "k_max": 10,
                     "desc": "Argile plastique peu consolidée, grande compressibilité, faibles résistances."},
                    {"type": "Argile molle à moyenne", "gamma": 18.0, "k_min": 10, "k_max": 40,
                     "desc": "Normalement à légèrement surconsolidée, tassements notables."},
                    {"type": "Argile ferme / surconsolidée", "gamma": 19.0, "k_min": 20, "k_max": 80,
                     "desc": "Argile raide, surconsolidée ou bien drainée, tassements plus limités."},
                    {"type": "Limon", "gamma": 18.0, "k_min": 15, "k_max": 60,
                     "desc": "Comportement intermédiaire argile/sable, sensible à l'eau et au compactage."},
                    {"type": "Sable lâche", "gamma": 18.0, "k_min": 10, "k_max": 30,
                     "desc": "Peu compacté, tassements importants, comportement peu rigide."},
                    {"type": "Sable moyennement compact", "gamma": 19.0, "k_min": 30, "k_max": 80,
                     "desc": "Sable courant sous bâtiments, portance correcte, tassements modérés."},
                    {"type": "Sable dense / graveleux", "gamma": 20.0, "k_min": 80, "k_max": 200,
                     "desc": "Très compact, très bonne portance, tassements faibles."},
                ]
                df = pd.DataFrame([{
                    "Type de sol": s["type"], "γ (kN/m³)": s["gamma"],
                    "k_min (MN/m³)": s["k_min"], "k_max (MN/m³)": s["k_max"],
                    "qₐ_min (kg/cm²)": round(s["k_min"] * factor_q, 2),
                    "qₐ_max (kg/cm²)": round(s["k_max"] * factor_q, 2),
                } for s in soils])
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown("#### Fiche sol")
                choix = st.selectbox("Type de sol :", [s["type"] for s in soils], index=6)
                sol = next(s for s in soils if s["type"] == choix)
                q_min = sol["k_min"] * factor_q
                q_max = sol["k_max"] * factor_q
                _bloc(sol["type"], f"qₐ ≈ {q_min:,.2f}–{q_max:,.2f} kg/cm²".replace(",", " "), "info")
                st.markdown(sol["desc"])
                st.markdown(f"- γ ≈ **{sol['gamma']} kN/m³**  \n"
                            f"- k ≈ **{sol['k_min']} à {sol['k_max']} MN/m³**  \n"
                            f"- pour w_adm = **{w_adm:.0f} mm** → qₐ ≈ **{q_min:.2f} à {q_max:.2f} kg/cm²**")

        st.divider()
        st.markdown("<div class='small'>Valeurs de k et qₐ indicatives, réservées au pré-dimensionnement. "
                    "Se référer au rapport géotechnique et à l'EN 1997 (Eurocode 7) pour le dimensionnement final.</div>",
                    unsafe_allow_html=True)

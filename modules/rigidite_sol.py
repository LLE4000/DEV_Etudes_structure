import math
import pandas as pd
import streamlit as st


def show():
    """
    Page Streamlit : calcul de la raideur de sol k (modèle de Winkler)
    """

    # -----------------------------
    # 🎨 Style
    # -----------------------------
    STYLES = """
    <style>
    h1, h2, h3 { margin: 0 0 .5rem 0; }
    .section-title { font-size: 1.25rem; font-weight: 700; margin: .25rem 0 .5rem 0; }
    .card { background: white; border-radius: 16px; padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(15,23,42,.05); border: 1px solid #EEF2F7; }
    .badge { display:inline-block; padding:.25rem .6rem; border-radius:999px;
             background:#F5F7FB; color:#334155; font-size:.85rem; }
    .metric-box { background:#F5F7FB; border-radius:12px; padding:.75rem 1rem;
                  border:1px solid #E2E8F0; }
    .small { color:#64748B; font-size:.9rem; }
    .topbar button { border-radius: 12px !important; height: 48px; font-weight: 600; }
    .katex-display { text-align:left !important; margin: .25rem 0 .5rem 0 !important; }
    .katex-display > .katex { text-align:left !important; }
    .memo-chip {
        display:inline-block; padding: 2px 8px; border-radius: 999px;
        background:#EEF2FF; color:#3730A3; font-size: .8rem;
    }
    </style>
    """
    st.markdown(STYLES, unsafe_allow_html=True)

    # =============================================================
    # 🧰 Helpers unités & affichage
    # =============================================================
    def to_kPa_from(value: float, unit: str) -> float:
        """Convertit une pression entrée (kPa, MPa, kg/cm²) en kPa."""
        if unit == "kPa":
            return value
        if unit == "MPa":
            return value * 1000.0
        if unit == "kg/cm²":
            return value * 98.0665
        return value

    def from_kPa_to(value_kPa: float, unit: str) -> float:
        """Convertit une pression depuis kPa vers l’unité souhaitée."""
        if unit == "kPa":
            return value_kPa
        if unit == "MPa":
            return value_kPa / 1000.0
        if unit == "kg/cm²":
            return value_kPa / 98.0665
        return value_kPa

    def E_MPa_to_kPa(E_MPa: float) -> float:
        """E en MPa (N/mm²) → kPa (kN/m²)."""
        return E_MPa * 1000.0

    def E_GPa_to_kPa(E_GPa: float) -> float:
        """E en GPa → kPa."""
        return E_GPa * 1_000_000.0

    def kNpm3_to_MNpm3(val_kNpm3: float) -> float:
        """k de kN/m³ → MN/m³."""
        return val_kNpm3 / 1000.0

    def MNpm3_to_kNpm3(val_MNpm3: float) -> float:
        """k de MN/m³ → kN/m³."""
        return val_MNpm3 * 1000.0

    def param_table(rows):
        """Affiche un tableau de paramètres (nom, description, valeur, unité)."""
        df = pd.DataFrame(rows, columns=["Paramètre", "Description", "Valeur", "Unité"])
        st.table(df)

    # =============================================================
    # 🧠 State & valeurs par défaut
    # =============================================================
    if "press_unit" not in st.session_state:
        st.session_state.press_unit = "kPa"
    if "module_unit" not in st.session_state:
        st.session_state.module_unit = "MPa"
    if "detail_calc" not in st.session_state:
        st.session_state.detail_calc = True
    if "adv_open" not in st.session_state:
        st.session_state.adv_open = False
    if "abaque_w" not in st.session_state:
        st.session_state.abaque_w = 20.0  # tassement de réf. pour l’abaque sols (mm)

    # =============================================================
    # 🧭 Barre du haut
    # =============================================================
    col_top = st.columns([1, 1, 1, 1, 1, 1])
    with col_top[0]:
        if st.button("🏠 Accueil", use_container_width=True, key="home_btn"):
            st.session_state.page = "Accueil"
            st.rerun()
    with col_top[1]:
        if st.button("🧹 Réinitialiser", use_container_width=True, key="reset_btn"):
            keep = {"press_unit", "module_unit", "detail_calc", "adv_open", "page", "abaque_w"}
            for k in list(st.session_state.keys()):
                if k not in keep:
                    st.session_state.pop(k, None)
            st.rerun()
    with col_top[2]:
        st.button("💾 Enregistrer", use_container_width=True, help="(À connecter à ton système JSON)")
    with col_top[3]:
        st.button("📂 Ouvrir", use_container_width=True, help="(Lecture de fichiers à venir)")
    with col_top[4]:
        st.button("📝 Générer PDF", use_container_width=True, help="(Export PDF à développer)")
    with col_top[5]:
        st.markdown("<span class='badge'>v1.6</span>", unsafe_allow_html=True)

    st.divider()

    # =============================================================
    # 🧱 En-tête
    # =============================================================
    st.markdown("# Raideur élastique des sols")
    st.markdown(
        "<span class='small'>Outil de pré-dimensionnement : on modélise le sol par des ressorts verticaux (modèle de Winkler).</span>",
        unsafe_allow_html=True,
    )

    # Fiche mémo générale
    with st.expander("📘 Fiche mémo (k, unités et modèle de Winkler)", expanded=False):
        st.markdown(
            """
            - On modélise le sol par un ressort vertical :  
              \\( q = k \\cdot w \\)  →  \\( k = q / w \\).  
            - Unités :
              - \\(q\\) : kPa = kN/m²  
              - \\(w\\) : m  
              - \\(k\\) : kN/m³ ou MN/m³ (1 MN/m³ = 1000 kN/m³)
            - \\(k\\) dépend de :
              - la **largeur B** de la fondation,
              - le **type de sol**,
              - le **niveau de charge** (ELS / ELU).
            - On peut relier \\(k\\) à une contrainte admissible \\(q_{adm}\\) pour un tassement choisi :  
              \\( q_{adm}(\\text{kg/cm}^2) \\approx k(\\text{MN/m}^3) \\cdot w(\\text{mm}) / 98{,}07 \\).
            - Les valeurs restent à valider par l’EN 1997 (Eurocode 7) et le rapport géotechnique.
            """
        )

    # =============================================================
    # 🧭 Deux colonnes
    # =============================================================
    col_left, col_right = st.columns([0.5, 0.5])

    # =============================================================
    # ================         COLONNE GAUCHE        ==============
    # =============================================================
    with col_left:
        st.markdown("### Informations et entrées")

        # --- Configuration avancée ---
        st.session_state.adv_open = st.checkbox(
            "Afficher la configuration avancée",
            value=st.session_state.adv_open,
        )
        if st.session_state.adv_open:
            c1, c2 = st.columns(2)
            with c1:
                old_unit = st.session_state.press_unit
                new_unit = st.selectbox(
                    "Pressions / contraintes",
                    ["kPa", "MPa", "kg/cm²"],
                    index=["kPa", "MPa", "kg/cm²"].index(st.session_state.press_unit),
                    help="Unité d’entrée des pressions. Les calculs sont faits en kPa en interne.",
                )
                # Conversion auto si changement d’unité
                if new_unit != old_unit and "solo_q" in st.session_state:
                    q_kPa = to_kPa_from(st.session_state.solo_q, old_unit)
                    st.session_state.solo_q = from_kPa_to(q_kPa, new_unit)
                if new_unit != old_unit and "solo_qad" in st.session_state:
                    qad_kPa = to_kPa_from(st.session_state.solo_qad, old_unit)
                    st.session_state.solo_qad = from_kPa_to(qad_kPa, new_unit)
                st.session_state.press_unit = new_unit

            with c2:
                old_munit = st.session_state.module_unit
                new_munit = st.selectbox(
                    "Modules E",
                    ["MPa", "GPa"],
                    index=0 if st.session_state.module_unit == "MPa" else 1,
                    help="Unité d’entrée pour E. Conversion automatique en kPa pour les calculs.",
                )
                if new_munit != old_munit and "solo_E" in st.session_state:
                    if old_munit == "MPa" and new_munit == "GPa":
                        st.session_state.solo_E /= 1000.0
                    elif old_munit == "GPa" and new_munit == "MPa":
                        st.session_state.solo_E *= 1000.0
                st.session_state.module_unit = new_munit

        # --- Choix du cas ---
        cas = st.selectbox(
            "Quel cas souhaitez-vous traiter ?",
            (
                "1. Sol homogène",
                "2. Sol multicouche",
                "3. CPT",
                "4. Plat sur béton",
                "5. Convertisseur & vérification",
                "6. Abaque sols",
            ),
            index=0,
        )

        # -------------------------
        # Formulaires selon le cas
        # -------------------------
        if cas.startswith("1"):
            # ----- CAS 1 : sol homogène -----
            st.markdown("**Sol homogène — choix de la méthode**")
            method = st.radio(
                "Méthode de calcul",
                [
                    "1. À partir d’un couple (q, w)",
                    "2. À partir d’une contrainte admissible (q_ad, s_ad)",
                    "3. À partir du module E du sol (E, B, ν)",
                ],
                horizontal=True,
            )

            st.markdown(
                "<span class='memo-chip'>Principe : k relie la pression q (kPa) au tassement w (m).</span>",
                unsafe_allow_html=True,
            )

            # (1) q, w
            if method.startswith("1."):
                st.caption("On connaît une pression de service q et un tassement w : on applique directement k = q / w.")
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.solo_q = st.number_input(
                        f"q (pression de service) [{st.session_state.press_unit}]",
                        min_value=0.0,
                        value=float(st.session_state.get("solo_q", 60.0)),
                        step=5.0,
                    )
                with c2:
                    st.session_state.solo_w = st.number_input(
                        "w (tassement) [mm]",
                        min_value=0.001,
                        value=float(st.session_state.get("solo_w", 20.0)),
                        step=5.0,
                    )

            # (2) q_ad, s_ad
            elif method.startswith("2."):
                st.caption(
                    "On connaît une contrainte admissible q_ad et un tassement admissible s_ad : "
                    "on prend k = q_ad / s_ad (avec correction SF si q_ad est une contrainte ultime)."
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.solo_qad = st.number_input(
                        f"q ad [{st.session_state.press_unit}]",
                        min_value=0.0,
                        value=float(st.session_state.get("solo_qad", 100.0)),
                        step=5.0,
                    )
                with c2:
                    st.session_state.solo_sadm = st.number_input(
                        "s adm [mm]",
                        min_value=0.1,
                        value=float(st.session_state.get("solo_sadm", 25.0)),
                        step=1.0,
                    )
                with c3:
                    st.session_state.solo_isult = st.toggle(
                        "q ad est une contrainte ultime ?",
                        value=st.session_state.get("solo_isult", False),
                        help="Si oui, q ad est multipliée par SF avant d’être utilisée.",
                    )
                    st.session_state.solo_sf = st.number_input(
                        "SF (si ultime)",
                        min_value=1.0,
                        value=float(st.session_state.get("solo_sf", 3.0)),
                        step=0.5,
                    )

            # (3) E, B, ν
            else:
                st.caption(
                    "On dispose d’un module de déformation E et d’une largeur B de semelle filante : "
                    "on prend k ≈ E / [B(1−ν²)]."
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.session_state.module_unit == "MPa":
                        st.session_state.solo_E = st.number_input(
                            "E du sol [MPa]",
                            min_value=0.0,
                            value=float(st.session_state.get("solo_E", 80.0)),
                            step=5.0,
                        )
                    else:
                        st.session_state.solo_E = st.number_input(
                            "E du sol [GPa]",
                            min_value=0.0,
                            value=float(st.session_state.get("solo_E", 0.08)),
                            step=0.01,
                        )
                with c2:
                    st.session_state.solo_B = st.number_input(
                        "B (largeur caractéristique) [m]",
                        min_value=0.01,
                        value=float(st.session_state.get("solo_B", 2.0)),
                        step=0.1,
                    )
                with c3:
                    st.session_state.solo_nu = st.number_input(
                        "ν (Poisson)",
                        min_value=0.0,
                        max_value=0.49,
                        value=float(st.session_state.get("solo_nu", 0.30)),
                        step=0.01,
                    )

        elif cas.startswith("2"):
            # ----- CAS 2 : multicouche -----
            st.markdown("**Sol multicouche — équivalence en série**")
            st.caption(
                "On approxime la raideur verticale par : "
                "1/k_eq = Σ(h_i / E_i) avec h_i en m et E_i en kPa."
            )

            n_layers = st.number_input(
                "Nombre de couches",
                min_value=1,
                max_value=6,
                value=int(st.session_state.get("multi_n_layers", 2)),
                step=1,
                key="multi_n_layers",
            )

            layers = []
            for i in range(int(n_layers)):
                c1, c2 = st.columns(2)
                idx = i + 1
                with c1:
                    h_i = st.number_input(
                        f"Épaisseur h{idx} [m]",
                        min_value=0.01,
                        value=float(st.session_state.get(f"multi_h_{i}", 1.0 if i == 0 else 2.0)),
                        step=0.1,
                        key=f"multi_h_{i}",
                    )
                with c2:
                    E_i = st.number_input(
                        f"E{idx} [MPa]",
                        min_value=0.1,
                        value=float(st.session_state.get(f"multi_E_{i}", 30.0 if i == 0 else 60.0)),
                        step=5.0,
                        key=f"multi_E_{i}",
                    )
                layers.append({"h": h_i, "E": E_i})

            st.session_state.multi_layers = layers

            st.session_state.multi_scale = st.checkbox(
                "Appliquer une largeur B et ν équivalents (fondation filante)",
                value=st.session_state.get("multi_scale", False),
            )
            if st.session_state.multi_scale:
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.multi_B = st.number_input(
                        "B équivalent [m]",
                        min_value=0.1,
                        value=float(st.session_state.get("multi_B", 2.0)),
                        step=0.1,
                    )
                with c2:
                    st.session_state.multi_nu = st.number_input(
                        "ν équivalent",
                        min_value=0.0,
                        max_value=0.49,
                        value=float(st.session_state.get("multi_nu", 0.30)),
                        step=0.01,
                    )

        elif cas.startswith("3"):
            # ----- CAS 3 : CPT -----
            st.markdown("**CPT — déduction de E puis de k**")
            st.caption(
                "On utilise une corrélation du type E = α_E (q_t − σ'ᵥ0), "
                "puis k ≈ E / [B(1−ν²)]."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.cpt_qt = st.number_input(
                    "qₜ (résistance de pointe nette) [MPa]",
                    min_value=0.0,
                    value=float(st.session_state.get("cpt_qt", 5.0)),
                    step=0.5,
                )
            with c2:
                st.session_state.cpt_sv0 = st.number_input(
                    "σ'ᵥ₀ (contrainte verticale effective) [kPa]",
                    min_value=0.0,
                    value=float(st.session_state.get("cpt_sv0", 100.0)),
                    step=10.0,
                )
            with c3:
                st.session_state.cpt_alphaE = st.number_input(
                    "α_E (facteur CPT → E)",
                    min_value=0.1,
                    value=float(st.session_state.get("cpt_alphaE", 2.5)),
                    step=0.1,
                )

            c4, c5 = st.columns(2)
            with c4:
                st.session_state.cpt_B = st.number_input(
                    "B (largeur influence / semelle) [m]",
                    min_value=0.1,
                    value=float(st.session_state.get("cpt_B", 2.0)),
                    step=0.1,
                )
            with c5:
                st.session_state.cpt_nu = st.number_input(
                    "ν (Poisson équivalent)",
                    min_value=0.0,
                    max_value=0.49,
                    value=float(st.session_state.get("cpt_nu", 0.30)),
                    step=0.01,
                )

        elif cas.startswith("4"):
            # ----- CAS 4 : plat sur béton -----
            st.markdown("**Plat métallique sur béton (ressort de contact)**")
            st.caption(
                "On assimile le contact à un ressort en compression du béton (et éventuellement du grout). "
                "Pour le béton seul : k_c ≈ E_c / [h_c(1−ν²)] ou k_c ≈ E_c / h_c suivant l’hypothèse."
            )

            st.markdown("**Géométrie du plat**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.plate_B = st.number_input(
                    "Largeur plat B [mm]",
                    min_value=20.0,
                    value=float(st.session_state.get("plate_B", 200.0)),
                    step=10.0,
                )
            with c2:
                st.session_state.plate_L = st.number_input(
                    "Longueur plat L [mm]",
                    min_value=20.0,
                    value=float(st.session_state.get("plate_L", 200.0)),
                    step=10.0,
                )
            with c3:
                st.session_state.plate_alpha = st.number_input(
                    "α (h_c = α·min(B,L))",
                    min_value=0.05,
                    value=float(st.session_state.get("plate_alpha", 0.5)),
                    step=0.05,
                )

            st.markdown("**Béton support**")
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.plate_Ec = st.number_input(
                    "E_c béton [GPa]",
                    min_value=5.0,
                    value=float(st.session_state.get("plate_Ec", 30.0)),
                    step=1.0,
                )
            with c5:
                st.session_state.plate_use_nu = st.checkbox(
                    "Tenir compte de ν du béton",
                    value=st.session_state.get("plate_use_nu", True),
                )

            if st.session_state.plate_use_nu:
                st.session_state.plate_nu = st.number_input(
                    "ν béton",
                    min_value=0.0,
                    max_value=0.49,
                    value=float(st.session_state.get("plate_nu", 0.20)),
                    step=0.01,
                )
            else:
                st.session_state.plate_nu = st.session_state.get("plate_nu", 0.20)

            st.markdown("**Lit de mortier / grout (optionnel)**")
            st.session_state.plate_has_grout = st.checkbox(
                "Présence d’un lit de mortier/grout",
                value=st.session_state.get("plate_has_grout", False),
            )

            if st.session_state.plate_has_grout:
                c6, c7 = st.columns(2)
                with c6:
                    st.session_state.plate_tg = st.number_input(
                        "Épaisseur grout t_g [mm]",
                        min_value=1.0,
                        value=float(st.session_state.get("plate_tg", 20.0)),
                        step=1.0,
                    )
                with c7:
                    st.session_state.plate_Eg = st.number_input(
                        "E_g grout [GPa]",
                        min_value=5.0,
                        value=float(st.session_state.get("plate_Eg", 20.0)),
                        step=1.0,
                    )
            else:
                st.session_state.plate_tg = st.session_state.get("plate_tg", 0.0)
                st.session_state.plate_Eg = st.session_state.get("plate_Eg", 20.0)

        elif cas.startswith("5"):
            # ----- CAS 5 : convertisseur -----
            st.markdown("**Convertisseur et vérification rapide**")
            st.info("Zone à compléter : conversions k ↔ E ↔ q,w.")

        else:
            # ----- CAS 6 : abaque sols (colonne gauche : rien à saisir) -----
            st.markdown("**Base de données / abaques sols**")
            st.caption(
                "Valeurs indicatives de poids volumique γ, raideur k (MN/m³) et contraintes "
                "admissibles qₐ (kg/cm²) associées à un tassement de référence w_adm. "
                "À confirmer par le géotechnicien."
            )

    # =============================================================
    # ================        COLONNE DROITE         ==============
    # =============================================================
    with col_right:
        st.markdown("### Dimensionnement / Résultats")

        st.session_state.detail_calc = st.checkbox(
            "📘 Détail des calculs (formules + valeurs numériques)",
            value=st.session_state.detail_calc,
        )

        # ----- CAS 1 : Sol homogène -----
        if cas.startswith("1"):
            with st.container(border=True):

                # On n'affiche que la méthode sélectionnée
                # ---------------------------------------

                # (1) k = q / w
                if method.startswith("1."):
                    if (
                        "solo_q" in st.session_state
                        and "solo_w" in st.session_state
                        and st.session_state.solo_w
                    ):
                        q_kPa = to_kPa_from(st.session_state.solo_q, st.session_state.press_unit)
                        w_m = st.session_state.solo_w / 1000.0
                        k_kNpm3 = q_kPa / w_m
                        ksA = kNpm3_to_MNpm3(k_kNpm3)
                        st.metric("k (MN/m³)", f"{ksA:,.2f}")
                        if st.session_state.detail_calc:
                            st.latex(r"k = \dfrac{q}{w}")
                            st.latex(
                                f"k = \\dfrac{{{q_kPa:,.1f}\\,\\text{{kN/m²}}}}{{{w_m:,.3f}\\,\\text{{m}}}}"
                                f" = {k_kNpm3:,.1f}\\,\\text{{kN/m³}} = {ksA:,.2f}\\,\\text{{MN/m³}}"
                            )
                            param_table(
                                [
                                    ("q", "Pression de service", f"{st.session_state.solo_q:,.3f}", st.session_state.press_unit),
                                    ("w", "Tassement", f"{st.session_state.solo_w:,.3f}", "mm"),
                                    ("k", "Raideur de sol", f"{ksA:,.3f}", "MN/m³"),
                                ]
                            )

                # (2) k = q_ad / s_ad
                elif method.startswith("2."):
                    if "solo_qad" in st.session_state and "solo_sadm" in st.session_state:
                        sadm_m = st.session_state.solo_sadm / 1000.0
                        qad_kPa = to_kPa_from(st.session_state.solo_qad, st.session_state.press_unit)
                        qad_used = qad_kPa * (st.session_state.solo_sf if st.session_state.solo_isult else 1.0)
                        if sadm_m > 0:
                            k_kNpm3_B = qad_used / sadm_m
                            ksB = kNpm3_to_MNpm3(k_kNpm3_B)
                            st.metric("k (MN/m³)", f"{ksB:,.2f}")
                            if st.session_state.detail_calc:
                                st.latex(r"k = \dfrac{q^{ad}}{s^{adm}}")
                                st.latex(
                                    f"k = \\dfrac{{{qad_used:,.1f}\\,\\text{{kN/m²}}}}{{{sadm_m:,.3f}\\,\\text{{m}}}}"
                                    f" = {k_kNpm3_B:,.1f}\\,\\text{{kN/m³}} = {ksB:,.2f}\\,\\text{{MN/m³}}"
                                )
                                param_table(
                                    [
                                        ("q ad", "Contrainte (admissible ou ultime×SF)",
                                         f"{from_kPa_to(qad_used, st.session_state.press_unit):,.3f}",
                                         st.session_state.press_unit),
                                        ("s adm", "Tassement admissible", f"{st.session_state.solo_sadm:,.3f}", "mm"),
                                        ("SF", "Facteur de sécurité",
                                         f"{st.session_state.solo_sf:,.2f}" if st.session_state.solo_isult else "—",
                                         "—"),
                                        ("k", "Raideur de sol", f"{ksB:,.3f}", "MN/m³"),
                                    ]
                                )

                # (3) k ≈ E / [B(1−ν²)]
                elif method.startswith("3."):
                    if "solo_E" in st.session_state and "solo_B" in st.session_state:
                        E_input = st.session_state.solo_E
                        E_MPa = E_input if st.session_state.module_unit == "MPa" else E_input * 1000.0
                        E_kPa = E_MPa_to_kPa(E_MPa)
                        B = max(st.session_state.solo_B, 1e-6)
                        nu = st.session_state.solo_nu
                        k_kNpm3_C = E_kPa / (B * (1 - nu ** 2))
                        ksC = kNpm3_to_MNpm3(k_kNpm3_C)
                        st.metric("k (MN/m³)", f"{ksC:,.2f}")
                        if st.session_state.detail_calc:
                            st.latex(r"k \approx \dfrac{E}{B(1-\nu^2)}")
                            st.latex(
                                f"k \\approx \\dfrac{{{E_kPa:,.0f}\\,\\text{{kN/m²}}}}"
                                f"{{{B:,.2f}\\,\\text{{m}}(1-{nu:.2f}^2)}}"
                                f" = {k_kNpm3_C:,.1f}\\,\\text{{kN/m³}} = {ksC:,.2f}\\,\\text{{MN/m³}}"
                            )
                            param_table(
                                [
                                    ("E", "Module de Young", f"{E_MPa:,.3f}", "MPa"),
                                    ("B", "Largeur caractéristique", f"{B:,.3f}", "m"),
                                    ("ν", "Coefficient de Poisson", f"{nu:,.3f}", "—"),
                                    ("k", "Raideur de sol", f"{ksC:,.3f}", "MN/m³"),
                                ]
                            )

        # ----- CAS 2 : Sol multicouche -----
        elif cas.startswith("2"):
            layers = st.session_state.get("multi_layers", [])
            denom, H = 0.0, 0.0
            for lay in layers:
                h = float(lay["h"])
                H += h
                E_MPa = float(lay["E"])
                E_kPa = E_MPa_to_kPa(E_MPa)
                if E_kPa > 0:
                    denom += h / E_kPa

            ks_eq = 0.0
            if denom > 0:
                k_kNpm3_eq = 1.0 / denom
                ks_eq = kNpm3_to_MNpm3(k_kNpm3_eq)
            st.metric("k_eq (MN/m³)", f"{ks_eq:,.2f}")

            if st.session_state.detail_calc:
                st.latex(r"k_{eq} = \left( \sum_i \dfrac{h_i}{E_i} \right)^{-1}")
                if denom > 0:
                    st.latex(
                        f"k_{{eq}} = \\left( \\sum_i \\dfrac{{h_i}}{{E_i}} \\right)^{{-1}}"
                        f" = {k_kNpm3_eq:,.1f}\\,\\text{{kN/m³}} = {ks_eq:,.2f}\\,\\text{{MN/m³}}"
                    )
                param_table(
                    [
                        ("H", "Somme des épaisseurs", f"{H:,.3f}", "m"),
                        ("k_eq", "Raideur équivalente", f"{ks_eq:,.3f}", "MN/m³"),
                    ]
                )

            if st.session_state.get("multi_scale"):
                H_eff = max(H, 1e-6)
                Eeq_kPa = (ks_eq * 1000.0) * H_eff
                Bm = st.session_state.get("multi_B", 2.0)
                nu = st.session_state.get("multi_nu", 0.30)
                k_kNpm3_B = Eeq_kPa / (Bm * (1 - nu ** 2))
                ksB = kNpm3_to_MNpm3(k_kNpm3_B)
                st.metric("k (avec B, ν) (MN/m³)", f"{ksB:,.2f}")
                if st.session_state.detail_calc:
                    st.latex(r"E_{eq} = k_{eq} \cdot H")
                    st.latex(r"k = \dfrac{E_{eq}}{B(1-\nu^2)}")
                    st.latex(
                        f"k = {k_kNpm3_B:,.1f}\\,\\text{{kN/m³}} = {ksB:,.2f}\\,\\text{{MN/m³}}"
                    )

        # ----- CAS 3 : CPT -----
        elif cas.startswith("3"):
            qt_MPa = st.session_state.get("cpt_qt", 0.0)
            qt_kPa = qt_MPa * 1000.0
            alphaE = st.session_state.get("cpt_alphaE", 2.5)
            sv0_kPa = st.session_state.get("cpt_sv0", 0.0)
            delta = max(qt_kPa - sv0_kPa, 0.0)
            E_kPa = alphaE * delta
            E_MPa = E_kPa / 1000.0

            B = max(st.session_state.get("cpt_B", 2.0), 1e-6)
            nu = st.session_state.get("cpt_nu", 0.30)
            k_kNpm3 = E_kPa / (B * (1 - nu ** 2))
            ks = kNpm3_to_MNpm3(k_kNpm3)

            c1, c2 = st.columns(2)
            c1.metric("E estimé (MPa)", f"{E_MPa:,.1f}")
            c2.metric("k (MN/m³)", f"{ks:,.2f}")

            if st.session_state.detail_calc:
                st.latex(r"E = \alpha_E \,(q_t - \sigma'_{v0})")
                st.latex(
                    f"E = {alphaE:,.2f}({qt_kPa:,.0f}-{sv0_kPa:,.0f})"
                    f" = {E_kPa:,.0f}\\,\\text{{kN/m²}} = {E_MPa:,.1f}\\,\\text{{MPa}}"
                )
                st.latex(r"k \approx \dfrac{E}{B(1-\nu^2)}")
                st.latex(
                    f"k \\approx {k_kNpm3:,.1f}\\,\\text{{kN/m³}} = {ks:,.2f}\\,\\text{{MN/m³}}"
                )

        # ----- CAS 4 : Plat sur béton -----
        elif cas.startswith("4"):
            Bp_mm = st.session_state.get("plate_B", 200.0)
            Lp_mm = st.session_state.get("plate_L", 200.0)
            alpha = st.session_state.get("plate_alpha", 0.5)
            Bp = Bp_mm / 1000.0
            Lp = Lp_mm / 1000.0
            hc = alpha * min(Bp, Lp)

            Ec_GPa = st.session_state.get("plate_Ec", 30.0)
            Ec_kPa = E_GPa_to_kPa(Ec_GPa)

            use_nu = st.session_state.get("plate_use_nu", True)
            nu_c = st.session_state.get("plate_nu", 0.20)

            if hc > 0:
                if use_nu:
                    kc_kNpm3 = Ec_kPa / (hc * (1 - nu_c ** 2))
                else:
                    kc_kNpm3 = Ec_kPa / hc
            else:
                kc_kNpm3 = 0.0

            has_grout = st.session_state.get("plate_has_grout", False)
            keq_kNpm3 = kc_kNpm3

            if has_grout and st.session_state.get("plate_tg", 0.0) > 0:
                tg_m = st.session_state.get("plate_tg", 20.0) / 1000.0
                Eg_GPa = st.session_state.get("plate_Eg", 20.0)
                Eg_kPa = E_GPa_to_kPa(Eg_GPa)
                kg_kNpm3 = Eg_kPa / tg_m if tg_m > 0 else 0.0
                if kc_kNpm3 > 0 and kg_kNpm3 > 0:
                    keq_kNpm3 = 1.0 / (1.0 / kc_kNpm3 + 1.0 / kg_kNpm3)

            keq = kNpm3_to_MNpm3(keq_kNpm3)
            st.metric("k_eq (MN/m³)", f"{keq:,.1f}")

            if st.session_state.detail_calc:
                st.latex(r"h_c = \alpha \,\min(B,L)")
                st.latex(
                    f"h_c = {alpha:,.2f} \\times "
                    f"\\min({Bp:,.3f},{Lp:,.3f}) = {hc:,.3f}\\,\\text{{m}}"
                )
                st.latex(r"k_c \approx \dfrac{E_c}{h_c(1-\nu^2)}")
                st.latex(
                    f"k_c \\approx {kc_kNpm3:,.1f}\\,\\text{{kN/m³}}"
                )
                if has_grout:
                    st.latex(r"\dfrac{1}{k_{eq}} = \dfrac{1}{k_c} + \dfrac{1}{k_g}")
                param_table(
                    [
                        ("B", "Largeur plat", f"{Bp_mm:,.0f}", "mm"),
                        ("L", "Longueur plat", f"{Lp_mm:,.0f}", "mm"),
                        ("h_c", "Épaisseur équivalente béton", f"{hc*1000:,.1f}", "mm"),
                        ("E_c", "Module béton", f"{Ec_GPa:,.1f}", "GPa"),
                        ("k_eq", "Raideur équivalente", f"{keq:,.1f}", "MN/m³"),
                    ]
                )

        # ----- CAS 5 : convertisseur -----
        elif cas.startswith("5"):
            st.info("À compléter : petits outils de conversion (k ↔ E ↔ q,w).")

        # ----- CAS 6 : abaque sols -----
        else:
            # Tassement de référence pour convertir k → qadm
            st.markdown("#### Réglage du tassement de référence")
            st.session_state.abaque_w = st.number_input(
                "Tassement de référence w_adm [mm]",
                min_value=1.0,
                max_value=100.0,
                value=float(st.session_state.abaque_w),
                step=5.0,
                help="Tassement admissible utilisé pour convertir k (MN/m³) en qₐ (kg/cm²). "
                     "En Belgique, 20 mm est une valeur courante pour les tassements de service.",
            )
            w_adm = st.session_state.abaque_w
            # facteur de conversion : q(kg/cm²) = k(MN/m³)*w(mm)/98.0665
            factor_q = w_adm / 98.0665

            soils = [
                {
                    "type": "Tourbe",
                    "gamma": 10.0,
                    "k_min": 1,
                    "k_max": 5,
                    "desc": "Sol très organique, très compressible, souvent saturé, capacité portante très faible. "
                            "On évite de fonder dedans (remblais, pieux, substitution...).",
                },
                {
                    "type": "Argile très molle",
                    "gamma": 16.0,
                    "k_min": 2,
                    "k_max": 10,
                    "desc": "Argile très plastique et peu consolidée, grande compressibilité et faibles résistances.",
                },
                {
                    "type": "Argile molle à moyenne",
                    "gamma": 18.0,
                    "k_min": 10,
                    "k_max": 40,
                    "desc": "Argile normalement consolidée ou légèrement surconsolidée, tassements notables.",
                },
                {
                    "type": "Argile ferme / surconsolidée",
                    "gamma": 19.0,
                    "k_min": 20,
                    "k_max": 80,
                    "desc": "Argile raide à très raide, surconsolidée ou bien drainée, meilleure tenue et tassements plus limités.",
                },
                {
                    "type": "Limon",
                    "gamma": 18.0,
                    "k_min": 15,
                    "k_max": 60,
                    "desc": "Silt / limon, comportement intermédiaire entre argiles et sables, sensibles à l’eau et au compactage.",
                },
                {
                    "type": "Sable lâche",
                    "gamma": 18.0,
                    "k_min": 10,
                    "k_max": 30,
                    "desc": "Sable peu compacté, tassements importants sous charges et comportement peu rigide.",
                },
                {
                    "type": "Sable moyennement compact",
                    "gamma": 19.0,
                    "k_min": 30,
                    "k_max": 80,
                    "desc": "Sable courant sous les bâtiments, portance correcte, tassements modérés.",
                },
                {
                    "type": "Sable dense / graveleux",
                    "gamma": 20.0,
                    "k_min": 80,
                    "k_max": 200,
                    "desc": "Sables très compacts ou graves denses, très bonne portance, tassements faibles.",
                },
            ]

            df = pd.DataFrame(
                [
                    {
                        "Type de sol": s["type"],
                        "γ (kN/m³)": s["gamma"],
                        "k_min (MN/m³)": s["k_min"],
                        "k_max (MN/m³)": s["k_max"],
                        "qₐ_min (kg/cm²)": s["k_min"] * factor_q,
                        "qₐ_max (kg/cm²)": s["k_max"] * factor_q,
                    }
                    for s in soils
                ]
            )
            st.dataframe(df, use_container_width=True)

            st.markdown("#### Fiche sol")

            choix = st.selectbox(
                "Afficher la fiche d’un type de sol :",
                [s["type"] for s in soils],
                index=6,
            )

            sol_sel = next(s for s in soils if s["type"] == choix)
            q_min = sol_sel["k_min"] * factor_q
            q_max = sol_sel["k_max"] * factor_q

            st.markdown(f"**{sol_sel['type']}**")
            st.markdown(sol_sel["desc"])
            st.markdown(
                f"- γ ≈ **{sol_sel['gamma']} kN/m³**  \n"
                f"- k ≈ **{sol_sel['k_min']} à {sol_sel['k_max']} MN/m³**  \n"
                f"- pour w_adm = **{w_adm:.0f} mm** :  \n"
                f"  → qₐ ≈ **{q_min:.2f} à {q_max:.2f} kg/cm²**"
            )

        # Bas de page
        st.divider()
        st.markdown(
            "<div style='color:#64748B;font-size:.9rem;'>"
            "Les valeurs de k et qₐ sont indicatives et réservées au pré-dimensionnement. "
            "Toujours se référer au rapport géotechnique et à l’EN 1997 (Eurocode 7) pour le dimensionnement final."
            "</div>",
            unsafe_allow_html=True,
        )

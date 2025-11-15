import math
import pandas as pd
import streamlit as st

def show():
    """
    Page Streamlit : calcul de la raideur de sol k (modèle de Winkler)
    avec plusieurs approches :
      1. Sol homogène (q-w, q_ad-s_ad, E-B-nu)
      2. Sol multicouche
      3. CPT → E → k
      4. Plat métallique sur béton
      5. Convertisseurs (placeholder)
      6. Abaques / base de connaissances (placeholder)
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
            keep = {"press_unit", "module_unit", "detail_calc", "adv_open", "page"}
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
        st.markdown("<span class='badge'>v1.3</span>", unsafe_allow_html=True)

    st.divider()

    # =============================================================
    # 🧱 En-tête
    # =============================================================
    st.markdown("# Raideur élastique des sols")
    st.markdown(
        "<span class='small'>Outil de pré-dimensionnement : on modélise le sol par des ressorts verticaux (modèle de Winkler).</span>",
        unsafe_allow_html=True,
    )

    # Petite fiche mémo générale (discrète)
    with st.expander("📘 Fiche mémo (k, unités et modèle de Winkler)", expanded=False):
        st.markdown(
            """
            - On cherche un **coefficient de réaction vertical** `k` tel que :  
              \\( q = k \\cdot w \\)  →  \\( k = q / w \\).  
            - `k` est exprimé typiquement en **MN/m³** (ou kN/m³).  
            - `k` **n’est pas** une propriété intrinsèque du sol, mais un **paramètre de modèle** qui dépend :
              - du type de fondation (semelle filante, radier, platine),
              - de la largeur influente \\(B\\),
              - du niveau de chargement (ELS / ELU).
            - Les formules s’appuient sur la **théorie de l’élasticité** et sur des **corrélations géotechniques** (CPT, essais in situ).
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

        # --- Bloc Configuration avancée ---
        st.session_state.adv_open = st.checkbox(
            "Afficher la configuration avancée",
            value=st.session_state.adv_open,
        )
        if st.session_state.adv_open:
            c1, c2 = st.columns(2)
            # Choix unités de pression
            with c1:
                old_unit = st.session_state.press_unit
                new_unit = st.selectbox(
                    "Pressions / contraintes",
                    ["kPa", "MPa", "kg/cm²"],
                    index=["kPa", "MPa", "kg/cm²"].index(st.session_state.press_unit),
                    help="Change uniquement l’unité d’entrée des pressions. Les calculs sont toujours faits en kPa en interne.",
                )
                # Conversion automatique si changement
                if new_unit != old_unit and "solo_q" in st.session_state:
                    q_kPa = to_kPa_from(st.session_state.solo_q, old_unit)
                    st.session_state.solo_q = from_kPa_to(q_kPa, new_unit)
                if new_unit != old_unit and "solo_qad" in st.session_state:
                    qad_kPa = to_kPa_from(st.session_state.solo_qad, old_unit)
                    st.session_state.solo_qad = from_kPa_to(qad_kPa, new_unit)
                st.session_state.press_unit = new_unit

            # Choix unités pour E
            with c2:
                old_munit = st.session_state.module_unit
                new_munit = st.selectbox(
                    "Modules E",
                    ["MPa", "GPa"],
                    index=0 if st.session_state.module_unit == "MPa" else 1,
                    help="Choix de l’unité d’entrée pour les modules E. Conversion interne en kPa.",
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
                "4. Plat métallique sur béton",
                "5. Convertisseur & vérification",
                "6. Abaque sols",
            ),
            index=0,
            help="Choisis la situation la plus proche de ton cas : semelle simple, profil multicouche, données CPT, platine…",
        )

        # -------------------------
        # Formulaires selon le cas
        # -------------------------
        # ----- CAS 1 : SOL HOMOGÈNE -----
        if cas.startswith("1"):
            st.markdown("**Sol homogène — choix de la méthode**")
            method = st.radio(
                "Méthode de calcul",
                [
                    "SLS direct (q, w)",
                    "Depuis contrainte admissible (q ad, s adm)",
                    "Depuis module E (E, B, ν)",
                ],
                horizontal=True,
            )

            # Mini-mémo sous forme de puce discrète
            st.markdown(
                "<span class='memo-chip'>k relie la pression q (kPa) au tassement w (m) : q = k·w.</span>",
                unsafe_allow_html=True,
            )

            # Explications sous chaque méthode
            if method.startswith("SLS"):
                st.caption(
                    "Méthode basée sur la déformation élastique mesurée ou estimée en service : "
                    r"on applique directement $k = q / w$."
                )
            elif method.startswith("Depuis contrainte"):
                st.caption(
                    "Méthode simplifiée à partir d’une contrainte admissible (ou ultime corrigée) et d’un tassement admissible : "
                    r"$k = q^{ad} / s^{adm}$."
                )
            else:
                st.caption(
                    "Méthode théorique à partir du module d’Young et d’une largeur caractéristique B (fondation filante) : "
                    r"$k \\approx E / \\big(B (1 - \\nu^2)\\big)$."
                )

            # --- Saisie selon la méthode ---
            if method.startswith("SLS"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.solo_q = st.number_input(
                        f"q (pression de service) [{st.session_state.press_unit}]",
                        min_value=0.0,
                        value=float(st.session_state.get("solo_q", 60.0)),
                        step=5.0,
                        help="Pression moyenne transmise au sol au stade de service.",
                    )
                with c2:
                    st.session_state.solo_w = st.number_input(
                        "w (tassement) [mm]",
                        min_value=0.001,
                        value=float(st.session_state.get("solo_w", 20.0)),
                        step=5.0,
                        help="Tassement vertical sous la fondation au stade de service.",
                    )

            elif method.startswith("Depuis contrainte"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.solo_qad = st.number_input(
                        f"q ad [{st.session_state.press_unit}]",
                        min_value=0.0,
                        value=float(st.session_state.get("solo_qad", 100.0)),
                        step=5.0,
                        help="Contrainte admissible donnée par le rapport de sol (ou contrainte ultime si coché).",
                    )
                with c2:
                    st.session_state.solo_sadm = st.number_input(
                        "s adm [mm]",
                        min_value=0.1,
                        value=float(st.session_state.get("solo_sadm", 25.0)),
                        step=1.0,
                        help="Tassement admissible retenu pour le calcul.",
                    )
                with c3:
                    st.session_state.solo_isult = st.toggle(
                        "q ad est une contrainte ultime ?",
                        value=st.session_state.get("solo_isult", False),
                        help="Si coché, q ad est interprété comme contrainte ultime et multiplié par SF.",
                    )
                    st.session_state.solo_sf = st.number_input(
                        "SF (si ultime)",
                        min_value=1.0,
                        value=float(st.session_state.get("solo_sf", 3.0)),
                        step=0.5,
                        help="Facteur appliqué à q ad si celle-ci est une contrainte ultime.",
                    )

            else:  # Depuis module E
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.session_state.module_unit == "MPa":
                        st.session_state.solo_E = st.number_input(
                            "E du sol [MPa]",
                            min_value=0.0,
                            value=float(st.session_state.get("solo_E", 80.0)),
                            step=5.0,
                            help="Module de Young du sol (en MPa).",
                        )
                    else:
                        st.session_state.solo_E = st.number_input(
                            "E du sol [GPa]",
                            min_value=0.0,
                            value=float(st.session_state.get("solo_E", 0.08)),
                            step=0.01,
                            help="Module de Young du sol (en GPa).",
                        )
                with c2:
                    st.session_state.solo_B = st.number_input(
                        "B (largeur caractéristique) [m]",
                        min_value=0.01,
                        value=float(st.session_state.get("solo_B", 2.0)),
                        step=0.1,
                        help="Largeur influente (ex. largeur de semelle filante).",
                    )
                with c3:
                    st.session_state.solo_nu = st.number_input(
                        "ν (Poisson)",
                        min_value=0.0,
                        max_value=0.49,
                        value=float(st.session_state.get("solo_nu", 0.30)),
                        step=0.01,
                        help="Coefficient de Poisson équivalent du sol.",
                    )

        # ----- CAS 2 : SOL MULTICOUCHE -----
        elif cas.startswith("2"):
            st.markdown("**Sol multicouche — équivalence en série**")
            st.caption(
                "On approxime la raideur verticale équivalente par une somme en série sur les couches : "
                r"$\\displaystyle \\frac{1}{k_{eq}} = \\sum_i \\dfrac{h_i}{E_i}$, "
                "avec $h_i$ en m et $E_i$ en kPa (E converti depuis MPa)."
            )

            # Nombre de couches
            n_layers = st.number_input(
                "Nombre de couches",
                min_value=1,
                max_value=6,
                value=int(st.session_state.get("multi_n_layers", 2)),
                step=1,
                key="multi_n_layers",
                help="Nombre de couches de sol superposées sous la fondation.",
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
                        help="Épaisseur de la couche i.",
                    )
                with c2:
                    E_i = st.number_input(
                        f"E{idx} [MPa]",
                        min_value=0.1,
                        value=float(st.session_state.get(f"multi_E_{i}", 30.0 if i == 0 else 60.0)),
                        step=5.0,
                        key=f"multi_E_{i}",
                        help="Module Young de la couche i (en MPa).",
                    )
                layers.append({"h": h_i, "E": E_i})

            st.session_state.multi_layers = layers

            # Échelle B & ν pour obtenir une raideur de fondation
            st.session_state.multi_scale = st.checkbox(
                "Appliquer une largeur B et un ν équivalents (fondation filante)",
                value=st.session_state.get("multi_scale", False),
                help="Permet de passer de k_eq (ressort global) à une raideur de fondation type Winkler k = E_eq / [B(1-ν²)].",
            )
            if st.session_state.multi_scale:
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.multi_B = st.number_input(
                        "B équivalent [m]",
                        min_value=0.1,
                        value=float(st.session_state.get("multi_B", 2.0)),
                        step=0.1,
                        help="Largeur influente de la fondation équivalente.",
                    )
                with c2:
                    st.session_state.multi_nu = st.number_input(
                        "ν équivalent",
                        min_value=0.0,
                        max_value=0.49,
                        value=float(st.session_state.get("multi_nu", 0.30)),
                        step=0.01,
                        help="Coefficient de Poisson équivalent global.",
                    )

        # ----- CAS 3 : CPT -----
        elif cas.startswith("3"):
            st.markdown("**CPT — déduction de E puis de k**")
            st.caption(
                "Approche classique à partir du pénétromètre statique : "
                r"$E = \\alpha_E \\big(q_t - \\sigma'_{v0}\\big)$, "
                "avec $q_t$ en MPa (résistance de pointe nette) et $\\sigma'_{v0}$ en kPa."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.cpt_qt = st.number_input(
                    "qₜ (résistance de pointe nette) [MPa]",
                    min_value=0.0,
                    value=float(st.session_state.get("cpt_qt", 5.0)),
                    step=0.5,
                    help="Résistance de pointe nette du CPT, en MPa.",
                )
            with c2:
                st.session_state.cpt_sv0 = st.number_input(
                    "σ'ᵥ₀ (contrainte verticale effective) [kPa]",
                    min_value=0.0,
                    value=float(st.session_state.get("cpt_sv0", 100.0)),
                    step=10.0,
                    help="Contrainte verticale effective au niveau considéré.",
                )
            with c3:
                st.session_state.cpt_alphaE = st.number_input(
                    "α_E (facteur CPT → E)",
                    min_value=0.1,
                    value=float(st.session_state.get("cpt_alphaE", 2.5)),
                    step=0.1,
                    help="Facteur empirique pour déduire E à partir du CPT (à caler sur la pratique locale).",
                )

            c4, c5 = st.columns(2)
            with c4:
                st.session_state.cpt_B = st.number_input(
                    "B (largeur influence / semelle) [m]",
                    min_value=0.1,
                    value=float(st.session_state.get("cpt_B", 2.0)),
                    step=0.1,
                    help="Largeur influente (par ex. largeur de semelle).",
                )
            with c5:
                st.session_state.cpt_nu = st.number_input(
                    "ν (Poisson équivalent)",
                    min_value=0.0,
                    max_value=0.49,
                    value=float(st.session_state.get("cpt_nu", 0.30)),
                    step=0.01,
                    help="Coefficient de Poisson équivalent global.",
                )

        # ----- CAS 4 : PLAT MÉTALLIQUE SUR BÉTON -----
        elif cas.startswith("4"):
            st.markdown("**Plat métallique sur béton (ressort de contact)**")
            st.caption(
                "On assimile le contact à un ressort de compression dans le béton (et éventuellement un lit de mortier/grout). "
                "Pour le béton seul : "
                r"$k_c \\approx \\dfrac{E_c}{h_c(1-\\nu^2)}$ ou $k_c \\approx \\dfrac{E_c}{h_c}$ suivant l’hypothèse."
            )

            st.markdown("**Géométrie du plat**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.plate_B = st.number_input(
                    "Largeur plat B [mm]",
                    min_value=20.0,
                    value=float(st.session_state.get("plate_B", 200.0)),
                    step=10.0,
                    help="Largeur de la platine métallique.",
                )
            with c2:
                st.session_state.plate_L = st.number_input(
                    "Longueur plat L [mm]",
                    min_value=20.0,
                    value=float(st.session_state.get("plate_L", 200.0)),
                    step=10.0,
                    help="Longueur de la platine métallique.",
                )
            with c3:
                st.session_state.plate_alpha = st.number_input(
                    "α (h_c = α·min(B,L))",
                    min_value=0.05,
                    value=float(st.session_state.get("plate_alpha", 0.5)),
                    step=0.05,
                    help="Facteur fixant l’épaisseur équivalente de béton comprimé sous le plat.",
                )

            st.markdown("**Béton support**")
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.plate_Ec = st.number_input(
                    "E_c béton [GPa]",
                    min_value=5.0,
                    value=float(st.session_state.get("plate_Ec", 30.0)),
                    step=1.0,
                    help="Module E du béton support, en GPa.",
                )
            with c5:
                st.session_state.plate_use_nu = st.checkbox(
                    "Tenir compte de ν du béton",
                    value=st.session_state.get("plate_use_nu", True),
                    help="Si coché, on utilise k_c = E_c / (h_c (1-ν²)), sinon k_c = E_c / h_c.",
                )

            if st.session_state.plate_use_nu:
                st.session_state.plate_nu = st.number_input(
                    "ν béton",
                    min_value=0.0,
                    max_value=0.49,
                    value=float(st.session_state.get("plate_nu", 0.20)),
                    step=0.01,
                    help="Coefficient de Poisson du béton.",
                )
            else:
                # Valeur par défaut si non utilisée
                st.session_state.plate_nu = st.session_state.get("plate_nu", 0.20)

            st.markdown("**Lit de mortier / grout (optionnel)**")
            st.session_state.plate_has_grout = st.checkbox(
                "Présence d’un lit de mortier/grout",
                value=st.session_state.get("plate_has_grout", False),
                help="Si coché, on modélise aussi la compressibilité du lit de mortier.",
            )

            if st.session_state.plate_has_grout:
                c6, c7 = st.columns(2)
                with c6:
                    st.session_state.plate_tg = st.number_input(
                        "Épaisseur grout t_g [mm]",
                        min_value=1.0,
                        value=float(st.session_state.get("plate_tg", 20.0)),
                        step=1.0,
                        help="Épaisseur du lit de mortier/grout.",
                    )
                with c7:
                    st.session_state.plate_Eg = st.number_input(
                        "E_g grout [GPa]",
                        min_value=5.0,
                        value=float(st.session_state.get("plate_Eg", 20.0)),
                        step=1.0,
                        help="Module E du grout, en GPa.",
                    )
            else:
                st.session_state.plate_tg = st.session_state.get("plate_tg", 0.0)
                st.session_state.plate_Eg = st.session_state.get("plate_Eg", 20.0)

        # ----- CAS 5 : CONVERTISSEUR -----
        elif cas.startswith("5"):
            st.markdown("**Convertisseur et vérification rapide**")
            st.info(
                "À terme : petits outils de conversion (k ↔ E, k ↔ q,w, etc.). "
                "Pour l’instant, utilise plutôt les cas 1 à 4 pour calculer k."
            )

        # ----- CAS 6 : ABAQUE / BDD -----
        else:
            st.markdown("**Base de données / abaques**")
            st.info(
                "Section prévue pour intégrer des valeurs recommandées issues de ton GTR / rapports géotechniques. "
                "Par exemple, des fourchettes de k par type de sol."
            )

    # =============================================================
    # ================        COLONNE DROITE         ==============
    # =============================================================
    with col_right:
        st.markdown("### Dimensionnement / Résultats")

        # Interrupteur détail des calculs
        st.session_state.detail_calc = st.checkbox(
            "📘 Détail des calculs (formules + valeurs numériques)",
            value=st.session_state.detail_calc,
        )

        # ----- CAS 1 : Sol homogène -----
        if cas.startswith("1"):
            with st.container(border=True):
                # (1) Méthode directe SLS : k = q / w
                if (
                    "solo_q" in st.session_state
                    and "solo_w" in st.session_state
                    and st.session_state.solo_w
                ):
                    q_kPa = to_kPa_from(
                        st.session_state.solo_q, st.session_state.press_unit
                    )
                    w_m = st.session_state.solo_w / 1000.0  # mm → m
                    k_kNpm3 = q_kPa / w_m          # q (kPa = kN/m²) / w (m) → kN/m³
                    ksA = kNpm3_to_MNpm3(k_kNpm3)  # MN/m³

                    st.metric("k (MN/m³)", f"{ksA:,.2f}")

                    if st.session_state.detail_calc:
                        st.latex(r"k = \\dfrac{q}{w}")
                        st.latex(rf"k = {k_kNpm3:,.1f}\\,\\text{{kN/m^3}} = {ksA:,.2f}\\,\\text{{MN/m^3}}")
                        param_table(
                            [
                                {
                                    "Paramètre": "q",
                                    "Description": "Pression de service",
                                    "Valeur": f"{st.session_state.solo_q:,.3f}",
                                    "Unité": st.session_state.press_unit,
                                },
                                {
                                    "Paramètre": "w",
                                    "Description": "Tassement",
                                    "Valeur": f"{st.session_state.solo_w:,.3f}",
                                    "Unité": "mm",
                                },
                                {
                                    "Paramètre": "k",
                                    "Description": "Raideur de sol",
                                    "Valeur": f"{ksA:,.3f}",
                                    "Unité": "MN/m³",
                                },
                            ]
                        )

                # (2) Depuis contrainte admissible : k = q_ad / s_ad (évent. SF)
                if "solo_qad" in st.session_state and "solo_sadm" in st.session_state:
                    sadm_m = st.session_state.solo_sadm / 1000.0
                    qad_kPa = to_kPa_from(
                        st.session_state.solo_qad, st.session_state.press_unit
                    )
                    qad_used_kPa = qad_kPa * (
                        st.session_state.solo_sf if st.session_state.solo_isult else 1.0
                    )
                    if sadm_m > 0:
                        k_kNpm3_B = qad_used_kPa / sadm_m
                        ksB = kNpm3_to_MNpm3(k_kNpm3_B)
                        st.metric("k (MN/m³)", f"{ksB:,.2f}")
                        if st.session_state.detail_calc:
                            st.latex(r"k = \\dfrac{q^{ad}}{s^{adm}}")
                            st.latex(rf"k = {k_kNpm3_B:,.1f}\\,\\text{{kN/m^3}} = {ksB:,.2f}\\,\\text{{MN/m^3}}")
                            param_table(
                                [
                                    {
                                        "Paramètre": "q ad",
                                        "Description": "Contrainte admissible (ou ultime × SF)",
                                        "Valeur": f"{from_kPa_to(qad_used_kPa, st.session_state.press_unit):,.3f}",
                                        "Unité": st.session_state.press_unit,
                                    },
                                    {
                                        "Paramètre": "s adm",
                                        "Description": "Tassement admissible",
                                        "Valeur": f"{st.session_state.solo_sadm:,.3f}",
                                        "Unité": "mm",
                                    },
                                    {
                                        "Paramètre": "SF",
                                        "Description": "Facteur de sécurité",
                                        "Valeur": f"{st.session_state.solo_sf:,.2f}"
                                        if st.session_state.solo_isult
                                        else "—",
                                        "Unité": "—",
                                    },
                                    {
                                        "Paramètre": "k",
                                        "Description": "Raideur de sol",
                                        "Valeur": f"{ksB:,.3f}",
                                        "Unité": "MN/m³",
                                    },
                                ]
                            )

                # (3) Depuis module E : k ≈ E / [B(1-ν²)]
                if "solo_E" in st.session_state and "solo_B" in st.session_state:
                    E_input = st.session_state.solo_E
                    # E en MPa ou GPa → MPa
                    E_MPa = (
                        E_input
                        if st.session_state.module_unit == "MPa"
                        else E_input * 1000.0
                    )
                    E_kPa = E_MPa_to_kPa(E_MPa)
                    B = max(st.session_state.solo_B, 1e-6)
                    nu = st.session_state.solo_nu
                    k_kNpm3_C = E_kPa / (B * (1 - nu**2))
                    ksC = kNpm3_to_MNpm3(k_kNpm3_C)
                    st.metric("k (MN/m³)", f"{ksC:,.2f}")
                    if st.session_state.detail_calc:
                        st.latex(r"k \\approx \\dfrac{E}{B(1-\\nu^2)}")
                        st.latex(rf"k \\approx {k_kNpm3_C:,.1f}\\,\\text{{kN/m^3}} = {ksC:,.2f}\\,\\text{{MN/m^3}}")
                        param_table(
                            [
                                {
                                    "Paramètre": "E",
                                    "Description": "Module de Young",
                                    "Valeur": f"{E_MPa:,.3f}",
                                    "Unité": "MPa",
                                },
                                {
                                    "Paramètre": "B",
                                    "Description": "Largeur caractéristique",
                                    "Valeur": f"{B:,.3f}",
                                    "Unité": "m",
                                },
                                {
                                    "Paramètre": "ν",
                                    "Description": "Poisson",
                                    "Valeur": f"{nu:,.3f}",
                                    "Unité": "—",
                                },
                                {
                                    "Paramètre": "k",
                                    "Description": "Raideur de sol",
                                    "Valeur": f"{ksC:,.3f}",
                                    "Unité": "MN/m³",
                                },
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
                st.latex(r"k_{eq} = \\left( \\sum_i \\dfrac{h_i}{E_i} \\right)^{-1}")
                param_table(
                    [
                        {
                            "Paramètre": "H",
                            "Description": "Somme des épaisseurs",
                            "Valeur": f"{H:,.3f}",
                            "Unité": "m",
                        },
                        {
                            "Paramètre": "k_eq",
                            "Description": "Raideur équivalente",
                            "Valeur": f"{ks_eq:,.3f}",
                            "Unité": "MN/m³",
                        },
                    ]
                )

            # Option : passage à un k pour fondation filante via E_eq et B,ν
            if st.session_state.get("multi_scale"):
                H_eff = max(H, 1e-6)
                # E_eq [kPa] = k_eq[kN/m³]*H[m]
                Eeq_kPa = (ks_eq * 1000.0) * H_eff
                Bm = st.session_state.get("multi_B", 2.0)
                nu = st.session_state.get("multi_nu", 0.30)
                k_kNpm3_B = Eeq_kPa / (Bm * (1 - nu**2))
                ksB = kNpm3_to_MNpm3(k_kNpm3_B)
                st.metric("k (avec échelle B) (MN/m³)", f"{ksB:,.2f}")
                if st.session_state.detail_calc:
                    st.latex(r"E_{eq} = k_{eq} \\cdot H")
                    st.latex(rf"k = \\dfrac{{E_{{eq}}}}{{B(1-\\nu^2)}} \\approx {k_kNpm3_B:,.1f}\\,\\text{{kN/m^3}} = {ksB:,.2f}\\,\\text{{MN/m^3}}")

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
            k_kNpm3 = E_kPa / (B * (1 - nu**2))
            ks = kNpm3_to_MNpm3(k_kNpm3)

            c1, c2 = st.columns(2)
            c1.metric("E estimé (MPa)", f"{E_MPa:,.1f}")
            c2.metric("k (MN/m³)", f"{ks:,.2f}")

            if st.session_state.detail_calc:
                st.latex(r"E = \\alpha_E (q_t - \\sigma'_{v0})")
                st.latex(rf"E = {E_MPa:,.1f}\\,\\text{{MPa}}")
                st.latex(r"k \\approx \\dfrac{E}{B(1-\\nu^2)}")
                st.latex(rf"k \\approx {k_kNpm3:,.1f}\\,\\text{{kN/m^3}} = {ks:,.2f}\\,\\text{{MN/m^3}}")
                param_table(
                    [
                        {
                            "Paramètre": "q_t",
                            "Description": "Résistance de pointe nette",
                            "Valeur": f"{qt_MPa:,.2f}",
                            "Unité": "MPa",
                        },
                        {
                            "Paramètre": "σ'ᵥ₀",
                            "Description": "Contrainte verticale effective",
                            "Valeur": f"{sv0_kPa:,.1f}",
                            "Unité": "kPa",
                        },
                        {
                            "Paramètre": "α_E",
                            "Description": "Facteur CPT → E",
                            "Valeur": f"{alphaE:,.2f}",
                            "Unité": "—",
                        },
                        {
                            "Paramètre": "E",
                            "Description": "Module estimé",
                            "Valeur": f"{E_MPa:,.2f}",
                            "Unité": "MPa",
                        },
                        {
                            "Paramètre": "B",
                            "Description": "Largeur influence",
                            "Valeur": f"{B:,.2f}",
                            "Unité": "m",
                        },
                        {
                            "Paramètre": "ν",
                            "Description": "Poisson équivalent",
                            "Valeur": f"{nu:,.2f}",
                            "Unité": "—",
                        },
                        {
                            "Paramètre": "k",
                            "Description": "Raideur de sol",
                            "Valeur": f"{ks:,.2f}",
                            "Unité": "MN/m³",
                        },
                    ]
                )

        # ----- CAS 4 : Plat métallique -----
        elif cas.startswith("4"):
            Bp_mm = st.session_state.get("plate_B", 200.0)
            Lp_mm = st.session_state.get("plate_L", 200.0)
            alpha = st.session_state.get("plate_alpha", 0.5)
            Bp = Bp_mm / 1000.0
            Lp = Lp_mm / 1000.0
            hc = alpha * min(Bp, Lp)  # m

            Ec_GPa = st.session_state.get("plate_Ec", 30.0)
            Ec_kPa = E_GPa_to_kPa(Ec_GPa)

            use_nu = st.session_state.get("plate_use_nu", True)
            nu_c = st.session_state.get("plate_nu", 0.20)

            if hc <= 0:
                kc_kNpm3 = 0.0
            else:
                if use_nu:
                    kc_kNpm3 = Ec_kPa / (hc * (1 - nu_c**2))
                else:
                    kc_kNpm3 = Ec_kPa / hc

            has_grout = st.session_state.get("plate_has_grout", False)
            keq_kNpm3 = kc_kNpm3
            if has_grout and st.session_state.get("plate_tg", 0.0) > 0:
                tg_m = st.session_state.get("plate_tg", 20.0) / 1000.0
                Eg_GPa = st.session_state.get("plate_Eg", 20.0)
                Eg_kPa = E_GPa_to_kPa(Eg_GPa)
                kg_kNpm3 = Eg_kPa / tg_m if tg_m > 0 else 0.0
                if kc_kNpm3 > 0 and kg_kNpm3 > 0:
                    # Ressorts en série : 1/k_eq = 1/kc + 1/kg
                    keq_kNpm3 = 1.0 / (1.0 / kc_kNpm3 + 1.0 / kg_kNpm3)

            keq = kNpm3_to_MNpm3(keq_kNpm3)
            st.metric("k_eq (MN/m³)", f"{keq:,.1f}")

            if st.session_state.detail_calc:
                st.latex(r"h_c = \\alpha \\cdot \\min(B,L)")
                st.latex(r"k_c \\approx \\dfrac{E_c}{h_c(1-\\nu^2)} \\text{ ou } k_c \\approx \\dfrac{E_c}{h_c}")
                if has_grout:
                    st.latex(r"\\dfrac{1}{k_{eq}} = \\dfrac{1}{k_c} + \\dfrac{1}{k_g}")
                param_table(
                    [
                        {
                            "Paramètre": "B",
                            "Description": "Largeur plat",
                            "Valeur": f"{Bp_mm:,.0f}",
                            "Unité": "mm",
                        },
                        {
                            "Paramètre": "L",
                            "Description": "Longueur plat",
                            "Valeur": f"{Lp_mm:,.0f}",
                            "Unité": "mm",
                        },
                        {
                            "Paramètre": "h_c",
                            "Description": "Épaisseur équivalente béton",
                            "Valeur": f"{hc*1000:,.1f}",
                            "Unité": "mm",
                        },
                        {
                            "Paramètre": "E_c",
                            "Description": "Module béton",
                            "Valeur": f"{Ec_GPa:,.1f}",
                            "Unité": "GPa",
                        },
                        {
                            "Paramètre": "k_eq",
                            "Description": "Raideur équivalente",
                            "Valeur": f"{keq:,.1f}",
                            "Unité": "MN/m³",
                        },
                    ]
                )

        # ----- CAS 5 : Convertisseur -----
        elif cas.startswith("5"):
            st.caption(
                "Les conversions détaillées (k ↔ E, k ↔ q,w, etc.) pourront être ajoutées ici. "
                "Pour l’instant, utilise les cas 1–4 pour dériver k."
            )
            st.info("Zone en construction — logique identique aux cas précédents, mais centrée sur les conversions.")

        # ----- CAS 6 : Base de connaissances / abaques -----
        else:
            st.caption(
                "Les sols sélectionnés sont surlignés en vert dans le tableau de gauche (une fois implémenté)."
            )
            st.warning(
                "Section prévue pour accueillir des tableaux/abaques issus de ton GTR ou de rapports géotechniques."
            )

        # Fin colonne droite
        st.divider()

        st.markdown(
            "<div style='color:#64748B;font-size:.9rem;'>"
            "Les valeurs de k sont à utiliser uniquement à titre indicatif pour le pré-dimensionnement. "
            "Se référer aux rapports géotechniques, aux essais in situ et aux prescriptions de l’EN 1997 (Eurocode 7) "
            "et de son Annexe Nationale pour le dimensionnement final."
            "</div>",
            unsafe_allow_html=True,
        )

# =============================================================
# Raideur élastique des sols
# =============================================================

import math
import pandas as pd
import streamlit as st

# =============================================================
# ===============  FONCTION D’ENTRÉE DE PAGE  =================
# =============================================================
def show():

    # -----------------------------
    # ⚙️ Page config
    # -----------------------------
    st.set_page_config(page_title="Raideur élastique des sols", page_icon="🧱", layout="wide")

    # -----------------------------
    # 🎨 Style
    # -----------------------------
    STYLES = """
    <style>
    h1, h2, h3 { margin: 0 0 .5rem 0; }
    .section-title { font-size: 1.25rem; font-weight: 700; margin: .25rem 0 .5rem 0; }
    .card { background: white; border-radius: 16px; padding: 1rem 1.25rem; box-shadow: 0 2px 10px rgba(15,23,42,.05); border: 1px solid #EEF2F7; }
    .badge { display:inline-block; padding:.25rem .6rem; border-radius:999px; background:#F5F7FB; color:#334155; font-size:.85rem; }
    .metric-box { background:#F5F7FB; border-radius:12px; padding:.75rem 1rem; border:1px solid #E2E8F0; }
    .small { color:#64748B; font-size:.9rem; }
    .topbar button { border-radius: 12px !important; height: 48px; font-weight: 600; }
    .katex-display { text-align:left !important; margin: .25rem 0 .5rem 0 !important; }
    .katex-display > .katex { text-align:left !important; }
    .emph { background: #E6F4EA !important; font-weight: 700 !important; color:#14532D !important; }
    </style>
    """
    st.markdown(STYLES, unsafe_allow_html=True)

    # =============================================================
    # 🧰 Helpers unités & affichage
    # =============================================================
    def to_kPa_from(value: float, unit: str) -> float:
        if unit == "kPa": return value
        if unit == "MPa": return value * 1000.0
        if unit == "kg/cm²": return value * 98.0665
        return value

    def from_kPa_to(value_kPa: float, unit: str) -> float:
        if unit == "kPa": return value_kPa
        if unit == "MPa": return value_kPa / 1000.0
        if unit == "kg/cm²": return value_kPa / 98.0665
        return value_kPa

    def E_MPa_to_kPa(E_MPa: float) -> float:
        return E_MPa * 1000.0

    def E_GPa_to_kPa(E_GPa: float) -> float:
        return E_GPa * 1_000_000.0

    def kNpm3_to_MNpm3(val_kNpm3: float) -> float:
        return val_kNpm3 / 1000.0

    def MNpm3_to_kNpm3(val_MNpm3: float) -> float:
        return val_MNpm3 * 1000.0

    def param_table(rows):
        df = pd.DataFrame(rows, columns=["Paramètre", "Description", "Valeur", "Unité"])
        st.table(df)

    # =============================================================
    # 🧠 State & valeurs par défaut
    # =============================================================
    if "press_unit" not in st.session_state:  st.session_state.press_unit = "kPa"
    if "module_unit" not in st.session_state: st.session_state.module_unit = "MPa"
    if "detail_calc" not in st.session_state: st.session_state.detail_calc = True
    if "adv_open" not in st.session_state:    st.session_state.adv_open = False

    # =============================================================
    # 🧭 Barre du haut
    # =============================================================
    col_top = st.columns([1,1,1,1,1,1])
    with col_top[0]:
        st.button("🏠 Accueil", use_container_width=True, key="home_btn")
    with col_top[1]:
        if st.button("🧹 Réinitialiser", use_container_width=True, key="reset_btn"):
            keep = {"press_unit","module_unit","detail_calc","adv_open"}
            for k in list(st.session_state.keys()):
                if k not in keep: st.session_state.pop(k, None)
            st.experimental_rerun()
    with col_top[2]:
        st.button("💾 Enregistrer", use_container_width=True)
    with col_top[3]:
        st.button("📂 Ouvrir", use_container_width=True)
    with col_top[4]:
        st.button("📝 Générer PDF", use_container_width=True)
    with col_top[5]:
        st.markdown("<span class='badge'>v1.2</span>", unsafe_allow_html=True)

    st.divider()

    # =============================================================
    # 🧱 En-tête
    # =============================================================
    st.markdown("# Raideur élastique des sols")
    st.markdown("\n")

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
        st.session_state.adv_open = st.checkbox("Afficher la configuration avancée", value=st.session_state.adv_open)
        if st.session_state.adv_open:
            c1, c2 = st.columns(2)
            with c1:
                old_unit = st.session_state.press_unit
                new_unit = st.selectbox("Pressions / contraintes", ["kPa","MPa","kg/cm²"],
                                        index=["kPa","MPa","kg/cm²"].index(st.session_state.press_unit))
                # Conversion automatique si changement
                if new_unit != old_unit and "solo_q" in st.session_state:
                    q_kPa = to_kPa_from(st.session_state.solo_q, old_unit)
                    st.session_state.solo_q = from_kPa_to(q_kPa, new_unit)
                st.session_state.press_unit = new_unit

            with c2:
                old_munit = st.session_state.module_unit
                new_munit = st.selectbox("Modules E", ["MPa","GPa"],
                                         index=0 if st.session_state.module_unit=="MPa" else 1)
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
        )

        # -------------------------
        # Formulaires selon le cas
        # -------------------------
        if cas.startswith("1"):
            st.markdown("**Sol homogène — méthode**")
            method = st.radio(
                "Sélectionnez la méthode",
                ["SLS direct (q, w)", "Depuis contrainte admissible (q ad, s adm)", "Depuis module E (E, B, ν)"],
                horizontal=True
            )

            # Explications sous chaque méthode
            if method.startswith("SLS"):
                st.caption("Méthode basée sur la déformation élastique du sol : la raideur est calculée directement par k = q / w.")
            elif method.startswith("Depuis contrainte"):
                st.caption("Méthode simplifiée utilisant les valeurs admissibles du sol : k = q_ad / s_ad, ou k = SF·q_ad / s_ad si q_ad est une contrainte ultime.")
            else:
                st.caption("Méthode théorique utilisant le module d’Young du sol et la largeur caractéristique B : k = E / [B·(1−ν²)].")

            # --- Saisie selon la méthode ---
            if method.startswith("SLS"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.solo_q = st.number_input(
                        f"q (pression de service) [{st.session_state.press_unit}]",
                        min_value=0.0, value=60.0, step=5.0
                    )
                with c2:
                    st.session_state.solo_w = st.number_input(
                        "w (tassement) [mm]", min_value=0.001, value=20.0, step=5.0
                    )

            elif method.startswith("Depuis contrainte"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.solo_qad = st.number_input(
                        f"q ad [{st.session_state.press_unit}]",
                        min_value=0.0, value=100.0, step=5.0
                    )
                with c2:
                    st.session_state.solo_sadm = st.number_input("s adm [mm]", min_value=0.1, value=25.0, step=1.0)
                with c3:
                    st.session_state.solo_isult = st.toggle("q ad est une contrainte ultime ?", value=False)
                    st.session_state.solo_sf = st.number_input("SF (si ultime)", min_value=1.0, value=3.0, step=0.5)

            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.session_state.module_unit == "MPa":
                        st.session_state.solo_E = st.number_input("E du sol [MPa]", min_value=0.0, value=80.0, step=5.0)
                    else:
                        st.session_state.solo_E = st.number_input("E du sol [GPa]", min_value=0.0, value=0.08, step=0.01)
                with c2:
                    st.session_state.solo_B = st.number_input("B (largeur caractéristique) [m]", min_value=0.01, value=2.0, step=0.1)
                with c3:
                    st.session_state.solo_nu = st.number_input("ν (Poisson)", min_value=0.0, max_value=0.49, value=0.30, step=0.01)

    # =============================================================
    # ================        COLONNE DROITE         ==============
    # =============================================================
    with col_right:
        st.markdown("### Dimensionnement / Résultats")

        # Interrupteur détail des calculs
        st.session_state.detail_calc = st.checkbox(
            "📘 Détail des calculs (formules + paramètres)",
            value=st.session_state.detail_calc
        )

        # ----- CAS 1 : Sol homogène -----
        if cas.startswith("1"):
            with st.container(border=True):
                if "solo_q" in st.session_state and "solo_w" in st.session_state and st.session_state.solo_w:
                    q_kPa = to_kPa_from(st.session_state.solo_q, st.session_state.press_unit)
                    w_m = st.session_state.solo_w / 1000.0
                    ksA = kNpm3_to_MNpm3(q_kPa / w_m)
                    st.metric("k (MN/m³)", f"{ksA:,.2f}")
                    if st.session_state.detail_calc:
                        st.latex(r"k = \dfrac{q}{w}")
                        param_table([
                            {"Paramètre":"q","Description":"Pression de service","Valeur":f"{st.session_state.solo_q:,.3f}","Unité":st.session_state.press_unit},
                            {"Paramètre":"w","Description":"Tassement","Valeur":f"{st.session_state.solo_w:,.3f}","Unité":"mm"},
                            {"Paramètre":"k","Description":"Raideur de sol","Valeur":f"{ksA:,.3f}","Unité":"MN/m³"},
                        ])

                if "solo_qad" in st.session_state and "solo_sadm" in st.session_state:
                    sadm_m = st.session_state.solo_sadm / 1000.0
                    qad_kPa = to_kPa_from(st.session_state.solo_qad, st.session_state.press_unit)
                    qad_used = qad_kPa * (st.session_state.solo_sf if st.session_state.solo_isult else 1.0)
                    if sadm_m > 0:
                        ksB = kNpm3_to_MNpm3(qad_used / sadm_m)
                        st.metric("k (MN/m³)", f"{ksB:,.2f}")
                        if st.session_state.detail_calc:
                            st.latex(r"k = \dfrac{q^{ad}}{s^{adm}} \text{ ou } k = \dfrac{SF \cdot q^{ad}}{s^{adm}}")
                            param_table([
                                {"Paramètre":"q ad","Description":"Contrainte admissible (ou ultime × SF)","Valeur":f"{from_kPa_to(qad_used, st.session_state.press_unit):,.3f}","Unité":st.session_state.press_unit},
                                {"Paramètre":"s adm","Description":"Tassement admissible","Valeur":f"{st.session_state.solo_sadm:,.3f}","Unité":"mm"},
                                {"Paramètre":"SF","Description":"Facteur de sécurité","Valeur":f"{st.session_state.solo_sf:,.2f}" if st.session_state.solo_isult else "—","Unité":"—"},
                                {"Paramètre":"k","Description":"Raideur de sol","Valeur":f"{ksB:,.3f}","Unité":"MN/m³"},
                            ])

                if "solo_E" in st.session_state and "solo_B" in st.session_state:
                    E_input = st.session_state.solo_E
                    E_MPa = E_input if st.session_state.module_unit=="MPa" else E_input*1000.0
                    E_kPa = E_MPa_to_kPa(E_MPa)
                    B = max(st.session_state.solo_B, 1e-6)
                    nu = st.session_state.solo_nu
                    ksC = kNpm3_to_MNpm3(E_kPa / (B * (1 - nu**2)))
                    st.metric("k (MN/m³)", f"{ksC:,.2f}")
                    if st.session_state.detail_calc:
                        st.latex(r"k = \dfrac{E}{B(1-\nu^2)}")
                        param_table([
                            {"Paramètre":"E","Description":"Module de Young","Valeur":f"{E_MPa:,.3f}","Unité":"MPa"},
                            {"Paramètre":"B","Description":"Largeur caractéristique","Valeur":f"{B:,.3f}","Unité":"m"},
                            {"Paramètre":"ν","Description":"Poisson","Valeur":f"{nu:,.3f}","Unité":"—"},
                            {"Paramètre":"k","Description":"Raideur de sol","Valeur":f"{ksC:,.3f}","Unité":"MN/m³"},
                        ])

        # ----- CAS 2 : Sol multicouche -----
        elif cas.startswith("2"):
            st.caption("Méthode équivalente par sommation série : 1/k_eq = Σ(h_i / E_i).")
            layers = st.session_state.multi_layers
            denom, H = 0.0, 0.0
            for lay in layers:
                h = float(lay['h']); H += h
                E_MPa = float(lay['E'])
                E_kPa = E_MPa_to_kPa(E_MPa)
                if E_kPa > 0: denom += h / E_kPa
            ks_eq = kNpm3_to_MNpm3((1.0/denom) if denom>0 else 0.0)
            st.metric("k_eq (MN/m³)", f"{ks_eq:,.2f}")
            if st.session_state.detail_calc:
                st.latex(r"k_{eq} = \left( \sum_i \dfrac{h_i}{E_i} \right)^{-1}")
                param_table([
                    {"Paramètre":"H","Description":"Somme des épaisseurs","Valeur":f"{H:,.3f}","Unité":"m"},
                    {"Paramètre":"k_eq","Description":"Raideur équivalente","Valeur":f"{ks_eq:,.3f}","Unité":"MN/m³"},
                ])
            if st.session_state.get("multi_scale"):
                Eeq_kPa = (ks_eq*1000.0) * H
                Bm, nu = st.session_state.multi_B, st.session_state.multi_nu
                ksB = kNpm3_to_MNpm3(Eeq_kPa / (Bm * (1 - nu**2)))
                st.metric("k (avec échelle B) (MN/m³)", f"{ksB:,.2f}")
                if st.session_state.detail_calc:
                    st.latex(r"E_{eq}=k_{eq}H \quad ; \quad k=\dfrac{E_{eq}}{B(1-\nu^2)}")

        # ----- CAS 3 : CPT -----
        elif cas.startswith("3"):
            st.caption("Méthode empirique à partir des résultats de pénétration statique (CPT) : E = α_E·(q_t − σ'v0).")
            qt_kPa = st.session_state.cpt_qt * 1000.0
            E_kPa = st.session_state.cpt_alphaE * max(qt_kPa - st.session_state.cpt_sv0, 0.0)
            E_MPa = E_kPa / 1000.0
            ks = kNpm3_to_MNpm3(E_kPa / (st.session_state.cpt_B * (1 - st.session_state.cpt_nu**2)))
            c1, c2 = st.columns(2)
            c1.metric("E estimé (MPa)", f"{E_MPa:,.1f}")
            c2.metric("k (MN/m³)", f"{ks:,.2f}")

        # ----- CAS 4 : Plat métallique -----
        elif cas.startswith("4"):
            st.caption("Calcul du ressort équivalent de contact béton/mortier/acier à partir des modules et épaisseurs.")
            Bp = st.session_state.plate_B / 1000.0
            Lp = st.session_state.plate_L / 1000.0
            hc = st.session_state.plate_alpha * min(Bp, Lp)
            Ec_kPa = E_GPa_to_kPa(st.session_state.plate_Ec)
            kc_kNpm3 = Ec_kPa / (hc * (1 - st.session_state.plate_nu**2)) if st.session_state.plate_use_nu else Ec_kPa / hc
            if st.session_state.plate_has_grout and st.session_state.plate_tg > 0:
                Eg_kPa = E_GPa_to_kPa(st.session_state.plate_Eg)
                tg_m = st.session_state.plate_tg / 1000.0
                kg_kNpm3 = Eg_kPa / tg_m
                keq_kNpm3 = 1.0 / (1.0/kc_kNpm3 + 1.0/kg_kNpm3)
            else:
                keq_kNpm3 = kc_kNpm3
            keq = kNpm3_to_MNpm3(keq_kNpm3)
            st.metric("k_eq (MN/m³)", f"{keq:,.1f}")

        # ----- CAS 5 : Convertisseur -----
        elif cas.startswith("5"):
            st.caption("Les flèches « ← depuis … » signifient : convertir les autres champs depuis cette unité.")
            st.info("Utilisez le panneau gauche pour les conversions et le contrôle q→w.")

        # ----- CAS 6 : Base de connaissances -----
        else:
            st.caption("Les sols sélectionnés sont surlignés en vert dans le tableau de gauche.")
            st.warning("Les lignes sont mises en évidence, sans index affiché.")

        # Fin colonne droite
        st.divider()

        st.markdown(
            "<div style='color:#64748B;font-size:.9rem;'>"
            "Valeurs à utiliser uniquement à titre indicatif ; se référer aux données géotechniques locales."
            "</div>",
            unsafe_allow_html=True
        )


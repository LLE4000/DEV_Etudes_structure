# modules/corniere.py
import math
import streamlit as st


def show():
    """
    Page: Dimensionnement de cornières ancrées
    - Colonne gauche: entrées (section, charges, ancrages, critères)
    - Colonne droite: dimensionnement (transversal, longitudinal) + TODO ancrages
    """
    from modules import navigation
    navigation.bouton_accueil("cor_btn_home")
    # ----------------------------
    # Données cornières standards (extrait – complète au besoin)
    # format: "A x B x t (mm)": {"A":..., "B":..., "t":..., "kg_m": ...}
    ANGLES_STD = {
        # égales
        "15x15x3": {"A": 15, "B": 15, "t": 3, "kg_m": 0.70},
        "20x20x3": {"A": 20, "B": 20, "t": 3, "kg_m": 0.90},
        "25x25x3": {"A": 25, "B": 25, "t": 3, "kg_m": 1.14},
        "25x25x4": {"A": 25, "B": 25, "t": 4, "kg_m": 1.48},
        "30x30x3": {"A": 30, "B": 30, "t": 3, "kg_m": 1.39},
        "30x30x4": {"A": 30, "B": 30, "t": 4, "kg_m": 1.81},
        "30x30x5": {"A": 30, "B": 30, "t": 5, "kg_m": 2.22},
        "35x35x4": {"A": 35, "B": 35, "t": 4, "kg_m": 2.13},
        "40x40x4": {"A": 40, "B": 40, "t": 4, "kg_m": 2.46},
        "40x40x5": {"A": 40, "B": 40, "t": 5, "kg_m": 3.03},
        "40x40x6": {"A": 40, "B": 40, "t": 6, "kg_m": 3.58},
        "45x45x5": {"A": 45, "B": 45, "t": 5, "kg_m": 3.44},
        "50x50x5": {"A": 50, "B": 50, "t": 5, "kg_m": 3.84},
        "50x50x6": {"A": 50, "B": 50, "t": 6, "kg_m": 4.57},
        "50x50x8": {"A": 50, "B": 50, "t": 8, "kg_m": 5.93},
        "60x60x6": {"A": 60, "B": 60, "t": 6, "kg_m": 5.53},
        "60x60x8": {"A": 60, "B": 60, "t": 8, "kg_m": 7.22},
        "70x70x7": {"A": 70, "B": 70, "t": 7, "kg_m": 7.52},
        "80x80x8": {"A": 80, "B": 80, "t": 8, "kg_m": 9.81},
        "80x80x10": {"A": 80, "B": 80, "t": 10, "kg_m": 12.09},
        "80x80x12": {"A": 80, "B": 80, "t": 12, "kg_m": 14.29},
        "90x90x9": {"A": 90, "B": 90, "t": 9, "kg_m": 12.42},
        # inégales et autres tailles
        "100x100x10": {"A": 100, "B": 100, "t": 10, "kg_m": 15.32},
        "100x100x12": {"A": 100, "B": 100, "t": 12, "kg_m": 18.17},
        "120x120x10": {"A": 120, "B": 120, "t": 10, "kg_m": 18.55},
        "120x120x12": {"A": 120, "B": 120, "t": 12, "kg_m": 22.03},
        "120x120x15": {"A": 120, "B": 120, "t": 15, "kg_m": 27.15},
        "150x150x10": {"A": 150, "B": 150, "t": 10, "kg_m": 23.42},
        "150x150x12": {"A": 150, "B": 150, "t": 12, "kg_m": 27.87},
        "150x150x15": {"A": 150, "B": 150, "t": 15, "kg_m": 34.42},
        "200x200x20": {"A": 200, "B": 200, "t": 20, "kg_m": 61.08},
        "40x20x4": {"A": 40, "B": 20, "t": 4, "kg_m": 1.80},
        "40x25x4": {"A": 40, "B": 25, "t": 4, "kg_m": 1.97},
        "50x30x5": {"A": 50, "B": 30, "t": 5, "kg_m": 3.02},
        "60x40x6": {"A": 60, "B": 40, "t": 6, "kg_m": 4.55},
        "70x50x6": {"A": 70, "B": 50, "t": 6, "kg_m": 5.51},
        "80x40x6": {"A": 80, "B": 40, "t": 6, "kg_m": 5.52},
        "80x60x7": {"A": 80, "B": 60, "t": 7, "kg_m": 7.51},
        "90x90x10": {"A": 90, "B": 90, "t": 10, "kg_m": 18.54},
        "150x100x10": {"A": 150, "B": 100, "t": 10, "kg_m": 19.36},
        "200x100x10": {"A": 200, "B": 100, "t": 10, "kg_m": 23.42},
    }

    # ----------------------------
    # UI – deux colonnes
    left, right = st.columns([1, 1.2])

    with left:
        st.header("Choix de la cornière")
        use_std = st.toggle("Utiliser une cornière **standard**", value=True)

        if use_std:
            sel = st.selectbox("Section", sorted(ANGLES_STD.keys()))
            A = ANGLES_STD[sel]["A"]
            B = ANGLES_STD[sel]["B"]
            t = ANGLES_STD[sel]["t"]
            kg_m = ANGLES_STD[sel]["kg_m"]
            st.info(f"Dimensions : {A} × {B} × {t} mm — {kg_m:.2f} kg/m")
        else:
            A = st.number_input("Largeur aile A (mm)", 10.0, 300.0, 60.0, step=1.0)
            B = st.number_input("Largeur aile B (mm)", 10.0, 300.0, 60.0, step=1.0)
            t = st.number_input("Épaisseur t (mm)", 2.0, 20.0, 6.0, step=0.5)
            kg_m = None

        st.divider()

        st.header("Charges")
        charge_type = st.radio(
            "Type de charge", ["Linéaire (kN/m)", "Ponctuelle (kN)"], horizontal=True
        )
        if charge_type.startswith("Linéaire"):
            V_char = st.number_input(
                "Effort tranchant V (kN/m)", min_value=0.0, value=5.0, step=0.1
            )
            V_unit = "kN/m"
        else:
            V_char = st.number_input(
                "Effort tranchant V (kN)", min_value=0.0, value=5.0, step=0.1
            )
            V_unit = "kN"

        e = st.number_input(
            "Excentricité e (cm) – distance charge/cornière",
            min_value=0.0,
            value=2.3,
            step=0.1,
        )
        gamma_ELU = st.number_input(
            "Coefficient global (ELU)", min_value=1.0, value=1.35, step=0.05
        )

        st.caption(
            "💡 Le moment transversal est pris **M = V × e** "
            "(avec V en kN par mètre de cornière considérée et e en m)."
        )

        st.divider()

        st.header("Ancrages")
        s = st.number_input(
            "Espacement des fixations s (cm)", min_value=10.0, value=30.0, step=1.0
        )
        edge = st.number_input(
            "Distance de l’âme au bord du mur (cm)", min_value=0.0, value=2.0, step=0.5
        )

        st.divider()

        st.header("Critères matériaux (admissibles)")
        st.caption(
            "Tu peux reprendre tes valeurs ELS/ELU de la note. Par défaut : acier S235."
        )
        sig_lim_ELS = st.number_input(
            "σ_lim, ELS (N/mm²)", min_value=10.0, value=155.0, step=5.0
        )
        tau_lim_ELS = st.number_input(
            "τ_lim, ELS (N/mm²)", min_value=0.1, value=0.8, step=0.1
        )
        sig_lim_ELU = st.number_input(
            "σ_eq lim, ELU (N/mm²)", min_value=10.0, value=186.0, step=5.0
        )

    # ----------------------------
    # CALCULS utilitaires
    def to_m(val_cm):  # cm -> m
        return val_cm / 100.0

    def to_mm(val_cm):  # cm -> mm
        return val_cm * 10.0

    # Modèle lame : b = 1000 mm (vérif par mètre courant)
    b_trans_mm = 1000.0
    t_mm = float(t)

    # Section géométrique (modèle lame)
    I_trans = b_trans_mm * t_mm**3 / 12.0  # mm^4
    W_trans = b_trans_mm * t_mm**2 / 6.0   # mm^3
    Az_trans = b_trans_mm * t_mm           # mm^2

    # Efforts (ELS et ELU)
    if V_unit == "kN/m":
        V_ELS = V_char              # kN/m
        V_ELU = V_char * gamma_ELU  # kN/m
    else:
        # Ponctuelle: ramener par mètre de cornière pour le check transversal
        L = st.sidebar.number_input(
            "Longueur d’influence L pour charge ponctuelle (m)",
            0.1, 5.0, 1.0, step=0.1
        )
        V_ELS = V_char / L            # kN/m
        V_ELU = (V_char * gamma_ELU) / L

    # Moment transversal M = V * e (par mètre) → N·mm/mm
    e_m = to_m(e)
    M_ELS_Nmm = V_ELS * e_m * 1e6
    M_ELU_Nmm = V_ELU * e_m * 1e6

    # Contraintes transversales
    sigma_ELS = M_ELS_Nmm / W_trans if W_trans > 0 else 0.0
    sigma_ELU = M_ELU_Nmm / W_trans if W_trans > 0 else 0.0
    tau_ELS = (V_ELS * 1e3) / Az_trans if Az_trans > 0 else 0.0
    tau_ELU = (V_ELU * 1e3) / Az_trans if Az_trans > 0 else 0.0
    sigma_eq_ELS = math.sqrt(sigma_ELS**2 + 3.0 * tau_ELS**2)
    sigma_eq_ELU = math.sqrt(sigma_ELU**2 + 3.0 * tau_ELU**2)

    # Longitudinal (entre fixations)
    b_long_mm = to_mm(s)  # mm
    I_long = b_long_mm * t_mm**3 / 12.0
    W_long = b_long_mm * t_mm**2 / 6.0
    Az_long = b_long_mm * t_mm

    Vloc_ELS = V_ELS * (s / 100.0)  # kN
    Vloc_ELU = V_ELU * (s / 100.0)  # kN

    # TODO: si poutre appuyée sous q = V_ELS (kN/m), utiliser Mmax = q*(s/100)**2/8.
    Mlong_ELS_kNm = Vloc_ELS * e_m
    Mlong_ELU_kNm = Vloc_ELU * e_m

    sigma_long_ELS = (Mlong_ELS_kNm * 1e6) / W_long if W_long > 0 else 0.0
    sigma_long_ELU = (Mlong_ELU_kNm * 1e6) / W_long if W_long > 0 else 0.0
    tau_long_ELS = (Vloc_ELS * 1e3) / Az_long if Az_long > 0 else 0.0
    tau_long_ELU = (Vloc_ELU * 1e3) / Az_long if Az_long > 0 else 0.0
    sigma_eq_long_ELS = math.sqrt(sigma_long_ELS**2 + 3.0 * tau_long_ELS**2)
    sigma_eq_long_ELU = math.sqrt(sigma_long_ELU**2 + 3.0 * tau_long_ELU**2)

    # Taux
    U_sigma = sigma_eq_ELS / sig_lim_ELS if sig_lim_ELS > 0 else 0.0
    U_tau = tau_ELS / tau_lim_ELS if tau_lim_ELS > 0 else 0.0
    U_ELU = sigma_eq_ELU / sig_lim_ELU if sig_lim_ELU > 0 else 0.0
    U_long_ELU = sigma_eq_long_ELU / sig_lim_ELU if sig_lim_ELU > 0 else 0.0

    # ----------------------------
    # AFFICHAGE
    with right:
        st.header("Dimensionnement de la cornière")

        st.subheader("1) Vérification **transversale** (par mètre courant)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Épaisseur t", f"{t_mm:.1f} mm")
        c2.metric("Largeur de lame b", f"{b_trans_mm:.0f} mm")
        if kg_m:
            c3.metric("Poids", f"{kg_m:.2f} kg/m")
        else:
            c3.write("")

        st.markdown(
            f"""
- **Efforts** :  
  • V (ELS) = **{V_ELS:.2f} {('kN/m' if V_unit=='kN/m' else 'kN/m')}** → V (ELU) = **{V_ELU:.2f} kN/m**  
  • e = **{e:.2f} cm** → M (ELS) = **{V_ELS*e_m:.3f} kN·m/m**, M (ELU) = **{V_ELU*e_m:.3f} kN·m/m**

- **Section (modèle lame)** : \(I = b t^3/12\), \(W = b t^2/6\), \(A_z = b t\)

- **Contraintes** :  
  • \(\\sigma = M/W\) → σ(ELS)=**{sigma_ELS:.1f}** N/mm² ; σ(ELU)=**{sigma_ELU:.1f}** N/mm²  
  • \(\\tau = V/A_z\) → τ(ELS)=**{tau_ELS:.2f}** N/mm² ; τ(ELU)=**{tau_ELU:.2f}** N/mm²  
  • \(\\sigma_{{eq}} = \\sqrt{{\\sigma^2 + 3\\tau^2}}\) → σ_eq(ELS)=**{sigma_eq_ELS:.1f}** ; σ_eq(ELU)=**{sigma_eq_ELU:.1f}** N/mm²
"""
        )

        ok_els = "✅" if (sigma_eq_ELS <= sig_lim_ELS and tau_ELS <= tau_lim_ELS) else "❌"
        ok_elu = "✅" if (sigma_eq_ELU <= sig_lim_ELU) else "❌"

        st.success(f"ELS : σ_eq/lim = {U_sigma*100:.0f}% ; τ/lim = {U_tau*100:.0f}%  {ok_els}")
        st.success(f"ELU : σ_eq/lim = {U_ELU*100:.0f}%  {ok_elu}")

        st.divider()
        st.subheader("2) Vérification **longitudinale** (entre fixations)")
        st.caption(
            "Hypothèse simple par panneau de largeur s. "
            "Remplace **M_long** dans le code par ta formule exacte (TODO)."
        )

        st.markdown(
            f"""
- **Panneau** : s = **{s:.0f} cm** ⇒ b = **{b_long_mm:.0f} mm**  
- **Efforts locaux** : V_loc(ELS)=**{Vloc_ELS:.2f} kN**, V_loc(ELU)=**{Vloc_ELU:.2f} kN**  
- **Moments (provisoirement)** : M_long(ELS)=**{Mlong_ELS_kNm:.3f} kN·m**, M_long(ELU)=**{Mlong_ELU_kNm:.3f} kN·m**

- **Contraintes** :  
  • σ_long(ELS)=**{sigma_long_ELS:.1f}** ; σ_long(ELU)=**{sigma_long_ELU:.1f}** N/mm²  
  • τ_long(ELU)=**{tau_long_ELU:.2f}** N/mm²  
  • σ_eq_long(ELU)=**{sigma_eq_long_ELU:.1f}** N/mm²
"""
        )

        ok_long = "✅" if (sigma_eq_long_ELU <= sig_lim_ELU) else "❌"
        st.info(f"ELU (longitudinal) : σ_eq/lim = {U_long_ELU*100:.0f}% {ok_long}")

        st.divider()
        st.subheader("3) Vérification des **ancrages**")
        st.caption(
            "À compléter selon ton logiciel/ETA (ex. HILTI) : cisaillement/traction, "
            "entraxes, profondeur, béton, interaction, distances au bord."
        )
        st.write("- Calculer V/anchorage, N/anchorage et interaction vs résistances ETA.")
        st.write("- Vérifier distances au bord et épaisseur du support.")

        st.divider()
        st.subheader("Notes & hypothèses")
        st.markdown(
            """
- Modèle **lame** (par mètre) : \(I=b t^3/12,\ v=t/2,\ W=b t^2/6,\ A_z=b t\).  
- **Transversal** : \(M = V \times e\).  
- **Longitudinal** : panneau de largeur **s**. Remplacer **M_long** par ta relation exacte (poutre entre fixations).  
- Critères **ELS/ELU** paramétrables.
"""
        )


# Permet d'exécuter ce fichier seul pour debug local
if __name__ == "__main__":
    st.set_page_config(page_title="Dimensionnement cornières ancrées", layout="wide")
    show()

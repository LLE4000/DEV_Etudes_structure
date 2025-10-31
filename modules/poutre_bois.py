# modules/poutre_bois.py
import math
import numpy as np
import streamlit as st

# ============================================================
# BDD intégrée (valeurs usuelles EN 338 – à ajuster si besoin)
# unités: MPa pour contraintes/modules, kg/m3 pour masses
TIMBER_BDD = {
    "C14": {"fm_k":14, "ft0_k":8,  "fc0_k":16, "fc90_k":2.0, "fv_k":2.0, "E0_mean":7000,  "G_mean":440, "rho_mean":350, "rho_k":290},
    "C16": {"fm_k":16, "ft0_k":10, "fc0_k":17, "fc90_k":2.2, "fv_k":2.5, "E0_mean":8000,  "G_mean":500, "rho_mean":370, "rho_k":310},
    "C18": {"fm_k":18, "ft0_k":11, "fc0_k":18, "fc90_k":2.2, "fv_k":2.5, "E0_mean":9000,  "G_mean":560, "rho_mean":380, "rho_k":320},
    "C20": {"fm_k":20, "ft0_k":12, "fc0_k":19, "fc90_k":2.3, "fv_k":2.5, "E0_mean":9500,  "G_mean":600, "rho_mean":390, "rho_k":330},
    "C22": {"fm_k":22, "ft0_k":13, "fc0_k":20, "fc90_k":2.5, "fv_k":3.0, "E0_mean":10000, "G_mean":625, "rho_mean":410, "rho_k":340},
    "C24": {"fm_k":24, "ft0_k":14, "fc0_k":21, "fc90_k":2.7, "fv_k":4.0, "E0_mean":11000, "G_mean":690, "rho_mean":420, "rho_k":350},
    "C27": {"fm_k":27, "ft0_k":16, "fc0_k":23, "fc90_k":2.9, "fv_k":4.0, "E0_mean":11500, "G_mean":720, "rho_mean":450, "rho_k":380},
    "C30": {"fm_k":30, "ft0_k":18, "fc0_k":24, "fc90_k":3.0, "fv_k":4.0, "E0_mean":12000, "G_mean":750, "rho_mean":460, "rho_k":380},
    "C35": {"fm_k":35, "ft0_k":21, "fc0_k":27, "fc90_k":3.4, "fv_k":4.5, "E0_mean":13000, "G_mean":810, "rho_mean":500, "rho_k":410},
    "C40": {"fm_k":40, "ft0_k":24, "fc0_k":30, "fc90_k":3.8, "fv_k":5.0, "E0_mean":14000, "G_mean":875, "rho_mean":520, "rho_k":450},
}
# Sections standard (mm)
SECTIONS_STD = [
    {"tag":"38x150","b":38,"h":150}, {"tag":"45x70","b":45,"h":70}, {"tag":"45x90","b":45,"h":90},
    {"tag":"45x120","b":45,"h":120}, {"tag":"45x145","b":45,"h":145}, {"tag":"45x220","b":45,"h":220},
    {"tag":"63x150","b":63,"h":150}, {"tag":"63x175","b":63,"h":175}, {"tag":"63x200","b":63,"h":200},
    {"tag":"75x225","b":75,"h":225}, {"tag":"100x250","b":100,"h":250},
]
# ============================================================

# EC5 paramètres (modifiables)
GAMMA_M = 1.30                          # bois massif
KMOD = {                                 # par classe de service et durée dominante
    1: {"permanent":0.60,"long":0.70,"moyen":0.80,"court":0.90,"instant":1.10},
    2: {"permanent":0.60,"long":0.70,"moyen":0.80,"court":0.90,"instant":1.10},
    3: {"permanent":0.50,"long":0.55,"moyen":0.65,"court":0.70,"instant":0.90},
}
KDEF = {1:0.60, 2:0.80, 3:2.00}

# --- géométrie rectangulaire
def sect_rect(b_mm, h_mm):
    A = b_mm*h_mm                    # mm²
    I = b_mm*(h_mm**3)/12.0          # mm^4
    W = I/(h_mm/2.0)                 # mm^3
    return A, I, W

# --- flèche poutre simplement appuyée sous q (kN/m)
def defl_simply(q_kN_m, L_m, E_MPa, I_mm4):
    return (5.0 * (q_kN_m*1000.0) * (L_m*1000.0)**4) / (384.0 * E_MPa * I_mm4)  # mm

# --- cisaillement rectangle τ ≈ 1.5 V/A
def shear_tau_rect(V_kN, b_mm, h_mm):
    A = b_mm*h_mm
    return 1.5 * (V_kN*1000.0) / A

# --- facteur de taille kh ≈ (150/h)^0.2 (borné)
def kh_depth(h_mm):
    kh = (150.0/max(h_mm,1.0))**0.2
    return min(max(kh,0.6),1.3)

def show():
    st.header("Dimensionnement **poutre en bois** (EC5)")

    left, right = st.columns([1, 1.25])

    # -------------------- Colonne gauche : entrées
    with left:
        st.subheader("1) Matériau & service")
        classes = list(TIMBER_BDD.keys())
        cls = st.selectbox("Classe de résistance", classes, index=classes.index("C24") if "C24" in classes else 0)
        mat = TIMBER_BDD[cls]

        sc = st.radio("Classe de service", [1,2,3], horizontal=True)
        duration = st.selectbox("Durée dominante", ["permanent","long","moyen","court","instant"], index=1)
        kmod = KMOD[sc][duration]
        kdef = KDEF[sc]

        st.subheader("2) Géométrie")
        use_std = st.toggle("Section **standard**", value=True)
        if use_std:
            tag = st.selectbox("Section (mm)", [s["tag"] for s in SECTIONS_STD])
            sel = next(s for s in SECTIONS_STD if s["tag"]==tag)
            b_mm, h_mm = sel["b"], sel["h"]
        else:
            b_mm = st.number_input("Largeur b (mm)", 30.0, 300.0, 80.0, step=1.0)
            h_mm = st.number_input("Hauteur h (mm)", 80.0, 500.0, 240.0, step=1.0)

        L_m = st.number_input("Portée L (m)", 0.5, 12.0, 4.0, step=0.1)

        st.subheader("3) Charges")
        qG = st.number_input("Charge permanente G (kN/m)", 0.0, 50.0, 2.0, step=0.1)
        qQ = st.number_input("Charge d’exploitation Q (kN/m)", 0.0, 50.0, 3.0, step=0.1)
        gammaG = st.number_input("γ_G", 1.0, 2.0, 1.35, step=0.05)
        gammaQ = st.number_input("γ_Q", 1.0, 2.0, 1.50, step=0.05)
        psi2   = st.number_input("ψ₂ (quasi-permanent)", 0.1, 1.0, 0.30, step=0.05)

        st.subheader("4) Limites ELS (flèche)")
        lim_inst = st.selectbox("Limite instantanée", ["L/300","L/350","L/400","L/200"], index=0)
        lim_fin  = st.selectbox("Limite finale",      ["L/200","L/250","L/300"], index=0)

        st.subheader("5) Appuis (option)")
        check_fc90 = st.toggle("Compression ⟂ fil à l’appui", value=False)
        a_app_mm = st.number_input("Largeur d’appui a (mm)", 30.0, 200.0, 60.0, step=5.0, disabled=not check_fc90)

    # -------------------- Calculs
    A, I, W = sect_rect(b_mm, h_mm)
    kh = kh_depth(h_mm)

    # résistances de calcul
    fm_d   = kmod * mat["fm_k"]  * kh / GAMMA_M
    fv_d   = kmod * mat["fv_k"]       / GAMMA_M
    fc90_d = kmod * mat["fc90_k"]     / GAMMA_M if mat.get("fc90_k") else None

    # Efforts pour poutre simplement appuyée
    q_ELU = gammaG*qG + gammaQ*qQ
    M_ELU = q_ELU * L_m**2 / 8.0   # kN·m
    V_ELU = q_ELU * L_m / 2.0      # kN

    q_SLS_inst  = qG + qQ
    q_SLS_quasi = qG + psi2*qQ

    # Contraintes ELU
    sigma_m_Ed = (M_ELU*1e6) / W                # MPa
    tau_Ed     = shear_tau_rect(V_ELU, b_mm, h_mm)

    # Flèches
    w_inst = defl_simply(q_SLS_inst,  L_m, mat["E0_mean"], I)
    w_fin  = (1.0 + kdef) * defl_simply(q_SLS_quasi, L_m, mat["E0_mean"], I)

    # Limites ELS
    L_mm = L_m*1000.0
    lim_inst_mm = L_mm / float(lim_inst.split("/")[1])
    lim_fin_mm  = L_mm / float(lim_fin.split("/")[1])

    # Compression perpendiculaire (option)
    fc90_util = None
    if check_fc90 and fc90_d:
        R_sup = (q_SLS_inst * L_m) / 2.0               # kN
        sigma_c90_Ed = (R_sup*1000.0) / (b_mm*a_app_mm)
        fc90_util = sigma_c90_Ed / fc90_d

    # Taux d’utilisation
    u_flex = sigma_m_Ed / fm_d if fm_d>0 else np.nan
    u_shear = tau_Ed / fv_d if fv_d>0 else np.nan
    u_wi = w_inst / lim_inst_mm if lim_inst_mm>0 else np.nan
    u_wf = w_fin  / lim_fin_mm  if lim_fin_mm>0  else np.nan

    # -------------------- Colonne droite : résultats
    with right:
        st.subheader("Résultats")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Section (mm)", f"{int(b_mm)}×{int(h_mm)}")
        c2.metric("Portée L (m)", f"{L_m:.2f}")
        c3.metric("E0,mean (MPa)", f"{mat['E0_mean']:.0f}")
        c4.metric("ρ (kg/m³)", f"{mat['rho_mean']:.0f}")

        st.markdown("#### ELU")
        st.write(f"- **M_Ed** = {M_ELU:.2f} kN·m ; **V_Ed** = {V_ELU:.2f} kN")
        st.write(f"- **σ_m,Ed** = {sigma_m_Ed:.1f} MPa  |  **f_m,d** = {fm_d:.1f} MPa")
        (st.success if u_flex<=1.0 else st.error)(f"Flexion : **{u_flex*100:.0f}%**")
        st.write(f"- **τ_Ed** = {tau_Ed:.2f} MPa  |  **f_v,d** = {fv_d:.2f} MPa")
        (st.success if u_shear<=1.0 else st.error)(f"Cisaillement : **{u_shear*100:.0f}%**")

        if fc90_util is not None:
            st.write(f"- **σ_c,90,Ed** = {(q_SLS_inst*L_m/2*1000)/(b_mm*a_app_mm):.2f} MPa  |  **f_c,90,d** = {fc90_d:.2f} MPa")
            (st.success if fc90_util<=1.0 else st.error)(f"Compression ⟂ fil (appui) : **{fc90_util*100:.0f}%**")

        st.markdown("#### ELS (flèche)")
        c5,c6 = st.columns(2)
        with c5:
            st.write(f"Instantanée : w = {w_inst:.1f} mm | lim. {lim_inst} = {lim_inst_mm:.1f} mm")
            (st.success if u_wi<=1.0 else st.error)(f"Taux : **{u_wi*100:.0f}%**")
        with c6:
            st.write(f"Finale : w = {w_fin:.1f} mm | lim. {lim_fin} = {lim_fin_mm:.1f} mm")
            (st.success if u_wf<=1.0 else st.error)(f"Taux : **{u_wf*100:.0f}%**")

        st.caption(
            f"{cls}: fm,k={mat['fm_k']} MPa, fv,k={mat['fv_k']} MPa, fc,90,k={mat['fc90_k']} MPa, "
            f"E0,mean={mat['E0_mean']} MPa, Gmean={mat['G_mean']} MPa | "
            f"kmod={kmod:.2f} (SC {sc}, {duration}), kdef={kdef:.2f}, kh={kh:.2f}, γM={GAMMA_M:.2f}"
        )

        st.markdown("---")
        st.caption("Hypothèses: poutre simplement appuyée, charge uniforme. Vérifs EC5 simplifiées (flexion, cisaillement, flèches). "
                   "Ajuste la BDD et les facteurs selon ton Annexe Nationale.")

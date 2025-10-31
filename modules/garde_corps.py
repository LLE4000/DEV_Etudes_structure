# modules/garde_corps.py
import math
import streamlit as st

E_STEEL = 210000.0  # MPa

# ------- sections standard
STD_RHS = ["40x40x3", "50x50x3", "60x40x3", "60x60x3", "80x40x3", "80x40x4", "80x80x4"]
STD_CHS = ["Ø33.7x2.6", "Ø42.4x2.6", "Ø48.3x3.2", "Ø60.3x3.2"]

def parse_rhs(tag): b,h,t = tag.split("x"); return float(b),float(h),float(t)
def parse_chs(tag): D,t = tag.replace("Ø","").split("x"); return float(D),float(t)

# ------- propriétés géométriques
def I_W_rect(b, h): I=b*h**3/12.0; W=I/(h/2.0); return I, W
def I_W_RHS(b, h, t):
    bi,hi=max(b-2*t,1e-6),max(h-2*t,1e-6)
    I=(b*h**3 - bi*hi**3)/12.0; W=I/(h/2.0); return I, W
def I_W_CHS(D, t):
    Do,Di=D,max(D-2*t,1e-6)
    I=(math.pi/64.0)*(Do**4 - Di**4); W=I/(Do/2.0); return I, W

def shear_area_rect(b,h): return b*h
def shear_area_rhs(b,h,t): return 2*t*(b+h)      # approximation parois minces
def shear_area_chs(D,t):   return math.pi*D*t

# ------- flèches
def defl_cantilever_tip(P_kN, H_mm, E, I):  # mm
    return (P_kN*1000.0)*(H_mm**3)/(3.0*E*I)
def defl_simple_q(q_kN_m, L_m, E, I):       # mm
    return (5.0*(q_kN_m*1000.0)*(L_m*1000.0)**4)/(384.0*E*I)
def defl_simple_Pmid(P_kN, L_m, E, I):      # mm
    return ((P_kN*1000.0)*(L_m*1000.0)**3)/(48.0*E*I)

def lim_fleche_mm(L_mm, mode, val):
    return L_mm/max(float(val),1.0) if mode=="L/x" else float(val)

def box_ok(ok, txt): (st.success if ok else st.error)(txt)

def show():
    st.markdown("## Garde-corps")

    left, right = st.columns([1, 1.25])

    # ============ ENTRÉES ============
    with left:
        st.markdown("### Informations")
        c1,c2 = st.columns(2)
        with c1:
            H = st.number_input("Hauteur utile (mm)", 800.0, 1400.0, 1100.0, step=10.0)
        with c2:
            s = st.number_input("Entraxe montants s (mm)", 400.0, 2500.0, 1000.0, step=50.0)

        st.markdown("### Charges (ELS)")
        r1,r2,r3 = st.columns(3)
        with r1:
            charge_mode = st.radio("Charge montant", ["P directe", "à partir de q×s"], horizontal=True)
        with r2:
            P_dir = st.number_input("P (kN)", 0.0, 10.0, 1.0, step=0.1, disabled=(charge_mode!="P directe"))
        with r3:
            q_line = st.number_input("q (kN/m)", 0.0, 5.0, 1.0, step=0.1, disabled=(charge_mode!="à partir de q×s"))

        r4,r5,r6 = st.columns(3)
        with r4:
            Q_point = st.number_input("Q ponctuelle main courante (kN)", 0.0, 5.0, 0.0, step=0.1)
        with r5:
            comb_mc = st.radio("Combinaison MC", ["max", "somme"], horizontal=True)
        with r6:
            E_mod = st.number_input("E acier (MPa)", 100000.0, 220000.0, E_STEEL, step=1000.0)

        st.markdown("### Matériau & limites")
        c1,c2 = st.columns(2)
        with c1:
            sigma_adm = st.number_input("σ admissible (MPa)", 50.0, 300.0, 160.0, step=5.0)
        with c2:
            tau_adm   = st.number_input("τ admissible (MPa)", 10.0, 200.0, 90.0, step=5.0)

        st.markdown("### Éléments à vérifier")
        e1,e2,e3 = st.columns(3)
        with e1: calc_post = st.checkbox("Montant principal", True)
        with e2: calc_mc   = st.checkbox("Main courante", True)
        with e3: calc_sec  = st.checkbox("Montant secondaire", False)

        # ---- Montant principal : section + limites
        if calc_post:
            st.markdown("### Montant principal – section")
            mod_montant = st.radio("Modèle", ["Encastré (charge en tête)", "Poutre simple (charge à mi-portée)"], horizontal=False)
            typ_p = st.radio("Type", ["RHS","CHS","Rectangulaire"], horizontal=True, key="p_type")
            use_std_p = st.toggle("Section standard", value=True, key="p_std")

            if typ_p=="RHS":
                if use_std_p:
                    tag = st.selectbox("RHS (b×h×t)", STD_RHS, key="p_rhs")
                    bp,hp,tp = parse_rhs(tag)
                else:
                    bp = st.number_input("b (mm)", 20.0, 200.0, 50.0, step=1.0, key="bp")
                    hp = st.number_input("h (mm)", 20.0, 200.0, 50.0, step=1.0, key="hp")
                    tp = st.number_input("t (mm)", 2.0, 12.0, 3.0, step=0.1, key="tp")
                I_post, W_post = I_W_RHS(bp,hp,tp); Av_post = shear_area_rhs(bp,hp,tp)
            elif typ_p=="CHS":
                if use_std_p:
                    tag = st.selectbox("CHS (Ø×t)", STD_CHS, key="p_chs")
                    Dp,tp = parse_chs(tag)
                else:
                    Dp = st.number_input("D (mm)", 21.3, 168.3, 48.3, step=0.1, key="Dp")
                    tp = st.number_input("t (mm)", 2.0, 10.0, 3.2, step=0.1, key="tp2")
                I_post, W_post = I_W_CHS(Dp,tp); Av_post = shear_area_chs(Dp,tp)
            else:
                bp = st.number_input("b (mm)", 5.0, 120.0, 40.0, step=1.0, key="bp_rect")
                hp = st.number_input("h (mm)", 5.0, 200.0, 8.0, step=1.0, key="hp_rect")
                I_post, W_post = I_W_rect(bp,hp); Av_post = shear_area_rect(bp,hp)

            c1,c2 = st.columns(2)
            with c1:
                mode_post = st.radio("Flèche Montant", ["L/x","mm"], horizontal=True, key="lim_post_mode")
            with c2:
                val_post  = st.number_input("Valeur", 1.0, 1000.0, 200.0, step=5.0, key="lim_post_val")

        # ---- Main courante : section + limites
        if calc_mc:
            st.markdown("### Main courante – section")
            typ_mc = st.radio("Type", ["RHS","CHS","Rectangulaire"], horizontal=True, key="mc_type")
            use_std_mc = st.toggle("Section standard", value=True, key="mc_std")

            if typ_mc=="RHS":
                if use_std_mc:
                    tag = st.selectbox("RHS (b×h×t)", STD_RHS, key="mc_rhs")
                    bmc,hmc,tmc = parse_rhs(tag)
                else:
                    bmc = st.number_input("b (mm)", 20.0, 200.0, 40.0, step=1.0, key="bmc")
                    hmc = st.number_input("h (mm)", 20.0, 200.0, 20.0, step=1.0, key="hmc")
                    tmc = st.number_input("t (mm)", 2.0, 12.0, 3.0, step=0.1, key="tmc")
                I_mc, W_mc = I_W_RHS(bmc,hmc,tmc); Av_mc = shear_area_rhs(bmc,hmc,tmc)
            elif typ_mc=="CHS":
                if use_std_mc:
                    tag = st.selectbox("CHS (Ø×t)", STD_CHS, key="mc_chs")
                    Dmc,tmc = parse_chs(tag)
                else:
                    Dmc = st.number_input("D (mm)", 21.3, 168.3, 42.4, step=0.1, key="Dmc")
                    tmc = st.number_input("t (mm)", 2.0, 10.0, 2.6, step=0.1, key="tmc2")
                I_mc, W_mc = I_W_CHS(Dmc,tmc); Av_mc = shear_area_chs(Dmc,tmc)
            else:
                bmc = st.number_input("b (mm)", 5.0, 120.0, 40.0, step=1.0, key="bmc_rect")
                hmc = st.number_input("h (mm)", 5.0, 200.0, 8.0, step=1.0, key="hmc_rect")
                I_mc, W_mc = I_W_rect(bmc,hmc); Av_mc = shear_area_rect(bmc,hmc)

            c1,c2 = st.columns(2)
            with c1:
                mode_mc = st.radio("Flèche MC", ["L/x","mm"], horizontal=True, key="lim_mc_mode")
            with c2:
                val_mc  = st.number_input("Valeur ", 1.0, 1000.0, 200.0, step=5.0, key="lim_mc_val")

        # ---- Montant secondaire (barreau)
        if calc_sec:
            st.markdown("### Montant secondaire (barreau)")
            orient = st.radio("Orientation", ["vertical","horizontal"], horizontal=True)
            c1,c2,c3 = st.columns(3)
            with c1:
                L_bar = st.number_input("Portée barreau L (mm)", 100.0, 3000.0, 900.0, step=10.0)
            with c2:
                spacing = st.number_input("Entraxe barreaux (mm)", 50.0, 400.0, 110.0, step=5.0)
            with c3:
                q_panel = st.number_input("Charge panneau (kN/m²)", 0.0, 5.0, 1.0, step=0.1)

            typ_b = st.radio("Type section", ["RHS","CHS","Rectangulaire"], horizontal=True, key="bar_type")
            if typ_b=="RHS":
                tag = st.selectbox("RHS standard", STD_RHS, key="rhs_bar")
                bb,hb,tb = parse_rhs(tag)
                I_bar, W_bar = I_W_RHS(bb,hb,tb); Av_bar = shear_area_rhs(bb,hb,tb)
            elif typ_b=="CHS":
                tag = st.selectbox("CHS standard", STD_CHS, key="chs_bar")
                Db,tb = parse_chs(tag)
                I_bar, W_bar = I_W_CHS(Db,tb); Av_bar = shear_area_chs(Db,tb)
            else:
                bb = st.number_input("b (mm)", 5.0, 120.0, 20.0, step=1.0, key="b_rect_bar")
                hb = st.number_input("h (mm)", 5.0, 120.0, 6.0, step=1.0, key="h_rect_bar")
                I_bar, W_bar = I_W_rect(bb,hb); Av_bar = shear_area_rect(bb,hb)

            c1,c2 = st.columns(2)
            with c1:
                mode_bar = st.radio("Flèche barreau", ["L/x","mm"], horizontal=True, key="lim_bar_mode")
            with c2:
                val_bar  = st.number_input("Valeur", 1.0, 1000.0, 200.0, step=5.0, key="lim_bar_val")

    # ============ CALCULS ============
    s_m = s/1000.0; H_m = H/1000.0

    # charge sur montant
    P_montant = P_dir if charge_mode=="P directe" else q_line * s_m

    # Montant principal
    post = None
    if calc_post:
        if "Encastré" in mod_montant:
            M_post = P_montant * H_m
            V_post = P_montant
            d_post = defl_cantilever_tip(P_montant, H, E_mod, I_post)
            formule_post = r"M = P\cdot H;\quad V=P;\quad d=\dfrac{P\,H^3}{3EI}"
        else:
            M_post = P_montant * H_m / 4.0
            V_post = P_montant / 2.0
            d_post = defl_simple_Pmid(P_montant, H_m, E_mod, I_post)
            formule_post = r"M = \dfrac{P\,L}{4};\quad V=\dfrac{P}{2};\quad d=\dfrac{P\,L^3}{48EI}"
        sigma_post = (M_post * 1e6) / W_post
        tau_post   = (V_post * 1000.0) / Av_post
        lim_post_mm = lim_fleche_mm(H, mode_post, val_post)
        post = dict(M=M_post,V=V_post,d=d_post,sigma=sigma_post,tau=tau_post,lim=lim_post_mm,formule=formule_post)

    # Main courante
    mc = None
    if calc_mc:
        L_mc = s_m
        Mq = q_line * L_mc**2 / 8.0
        MQ = Q_point * L_mc / 4.0
        dq = defl_simple_q(q_line, L_mc, E_mod, I_mc)
        dQ = defl_simple_Pmid(Q_point, L_mc, E_mod, I_mc)
        if comb_mc == "max":
            M_mc = max(Mq, MQ); d_mc = max(dq, dQ); V_mc = max(q_line*L_mc/2.0, Q_point/2.0)
            formule_mc = r"M=\max\!\left(\dfrac{qL^2}{8},\dfrac{QL}{4}\right),\ d=\max\!\left(\dfrac{5qL^4}{384EI},\dfrac{QL^3}{48EI}\right)"
        else:
            M_mc = Mq + MQ; d_mc = dq + dQ; V_mc = q_line*L_mc/2.0 + Q_point/2.0
            formule_mc = r"M=\dfrac{qL^2}{8}+\dfrac{QL}{4},\ d=\dfrac{5qL^4}{384EI}+\dfrac{QL^3}{48EI}"
        sigma_mc = (M_mc * 1e6) / W_mc
        tau_mc   = (V_mc * 1000.0) / Av_mc
        lim_mc_mm = lim_fleche_mm(s, mode_mc, val_mc)
        mc = dict(M=M_mc,V=V_mc,d=d_mc,sigma=sigma_mc,tau=tau_mc,lim=lim_mc_mm,formule=formule_mc)

    # Barreau secondaire
    bar = None
    if calc_sec:
        Lb_m = (L_bar/1000.0 if orient=="vertical" else spacing/1000.0)
        space_m = (spacing/1000.0 if orient=="vertical" else L_bar/1000.0)
        q_bar = q_panel * (space_m)  # kN/m
        M_bar = q_bar * Lb_m**2 / 8.0
        V_bar = q_bar * Lb_m / 2.0
        d_bar = defl_simple_q(q_bar, Lb_m, E_mod, I_bar)
        sigma_bar = (M_bar * 1e6) / W_bar
        tau_bar   = (V_bar * 1000.0) / Av_bar
        lim_bar_mm = lim_fleche_mm(L_bar if orient=="vertical" else spacing, mode_bar, val_bar)
        bar = dict(M=M_bar,V=V_bar,d=d_bar,sigma=sigma_bar,tau=tau_bar,lim=lim_bar_mm,
                   formule=r"M=\dfrac{q_b L^2}{8},\ V=\dfrac{q_b L}{2},\ d=\dfrac{5 q_b L^4}{384EI},\ q_b=q_{panneau}\times e")

    # ============ RÉSULTATS ============
    with right:
        st.markdown("### Dimensionnement")
        r1,r2,r3,r4,r5 = st.columns(5)
        r1.metric("H (mm)", f"{H:.0f}")
        r2.metric("s (mm)", f"{s:.0f}")
        r3.metric("q (kN/m)", f"{q_line:.2f}")
        r4.metric("Q (kN)", f"{Q_point:.2f}")
        r5.toggle("Détailler les calculs", key="show_details", value=False)

        if post:
            st.markdown("#### Montant principal")
            st.write(f"**P** = {P_montant:.2f} kN  |  **M** = {post['M']:.3f} kN·m  |  **V** = {post['V']:.3f} kN")
            box_ok(post["sigma"]<=sigma_adm, f"Contrainte σ = {post['sigma']:.1f} MPa ≤ {sigma_adm:.1f} MPa")
            box_ok(post["tau"]  <=tau_adm,   f"Cisaillement τ = {post['tau']:.1f} MPa ≤ {tau_adm:.1f} MPa")
            box_ok(post["d"]    <=post["lim"], f"Flèche d = {post['d']:.1f} mm ≤ {post['lim']:.1f} mm")
            if st.session_state.get("show_details"):
                with st.expander("Détails montant"):
                    st.latex(post["formule"])
                    st.latex(rf"M={post['M']:.3f}\ \text{{kN·m}},\ V={post['V']:.3f}\ \text{{kN}},\ "
                             rf"\sigma=\frac{{M\cdot10^6}}W={post['sigma']:.1f}\ \text{{MPa}},\ "
                             rf"\tau=\frac{{V\cdot10^3}}{{A_v}}={post['tau']:.1f}\ \text{{MPa}},\ "
                             rf"d={post['d']:.1f}\ \text{{mm}}")

            st.divider()

        if mc:
            st.markdown("#### Main courante")
            st.write(f"**M** = {mc['M']:.3f} kN·m  |  **V** = {mc['V']:.3f} kN")
            box_ok(mc["sigma"]<=sigma_adm, f"Contrainte σ = {mc['sigma']:.1f} MPa ≤ {sigma_adm:.1f} MPa")
            box_ok(mc["tau"]  <=tau_adm,   f"Cisaillement τ = {mc['tau']:.1f} MPa ≤ {tau_adm:.1f} MPa")
            box_ok(mc["d"]    <=mc["lim"], f"Flèche d = {mc['d']:.1f} mm ≤ {mc['lim']:.1f} mm")
            if st.session_state.get("show_details"):
                with st.expander("Détails main courante"):
                    st.latex(mc["formule"])
                    st.latex(rf"M={mc['M']:.3f}\ \text{{kN·m}},\ V={mc['V']:.3f}\ \text{{kN}},\ "
                             rf"\sigma=\frac{{M\cdot10^6}}W={mc['sigma']:.1f}\ \text{{MPa}},\ "
                             rf"\tau=\frac{{V\cdot10^3}}{{A_v}}={mc['tau']:.1f}\ \text{{MPa}},\ "
                             rf"d={mc['d']:.1f}\ \text{{mm}}")

            st.divider()

        if bar:
            st.markdown("#### Montant secondaire (barreau)")
            st.write(f"**M** = {bar['M']:.3f} kN·m  |  **V** = {bar['V']:.3f} kN")
            box_ok(bar["sigma"]<=sigma_adm, f"Contrainte σ = {bar['sigma']:.1f} MPa ≤ {sigma_adm:.1f} MPa")
            box_ok(bar["tau"]  <=tau_adm,   f"Cisaillement τ = {bar['tau']:.1f} MPa ≤ {tau_adm:.1f} MPa")
            box_ok(bar["d"]    <=bar["lim"], f"Flèche d = {bar['d']:.1f} mm ≤ {bar['lim']:.1f} mm")
            if st.session_state.get("show_details"):
                with st.expander("Détails barreau"):
                    st.latex(bar["formule"])
                    st.latex(rf"M={bar['M']:.3f}\ \text{{kN·m}},\ V={bar['V']:.3f}\ \text{{kN}},\ "
                             rf"\sigma=\frac{{M\cdot10^6}}W={bar['sigma']:.1f}\ \text{{MPa}},\ "
                             rf"\tau=\frac{{V\cdot10^3}}{{A_v}}={bar['tau']:.1f}\ \text{{MPa}},\ "
                             rf"d={bar['d']:.1f}\ \text{{mm}}")

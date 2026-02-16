import streamlit as st
import json
import math
import pandas as pd

# =========================================================
# Helpers
# =========================================================
def pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, "None", ""):
            return d[k]
    return default

def to_float(x, default=None):
    try:
        if x in (None, "None", ""):
            return default
        return float(x)
    except Exception:
        return default

def fmt_no_trailing_zeros(x, digits=3):
    if x is None:
        return "—"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if xf.is_integer():
        return str(int(round(xf)))
    return f"{xf:.{digits}f}".rstrip("0").rstrip(".")

# =========================================================
# Chargement profils (tolérant + plus de champs EC3 si dispo)
# =========================================================
@st.cache_data
def load_profiles():
    with open("profiles_test.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = {}
    for name, p in raw.items():
        # Dimensions (mm)
        h  = to_float(pick(p, "h"))
        b  = to_float(pick(p, "b"), default=0.0)
        tw = to_float(pick(p, "tw"), default=0.0)
        tf = to_float(pick(p, "tf"), default=0.0)
        r  = to_float(pick(p, "r"),  default=0.0)

        # Propriétés géométriques
        # Aires en cm² ; inerties en cm⁴ ; modules en cm³
        A   = to_float(pick(p, "A"), default=None)      # cm²
        Avy = to_float(pick(p, "Avy"), default=None)    # cm² (optionnel)
        Avz = to_float(pick(p, "Avz"), default=None)    # cm²

        Iy  = to_float(pick(p, "Iy", "Iv"), default=None)  # cm⁴
        Iz  = to_float(pick(p, "Iz"), default=None)        # cm⁴

        Wel_y = to_float(pick(p, "Wely", "Wel", "Wel_y"), default=None)  # cm³
        Wel_z = to_float(pick(p, "Welz", "Wel_z"), default=None)         # cm³

        # Torsion / gauchissement si dispo
        It = to_float(pick(p, "It"), default=None)      # cm⁴
        Iw = to_float(pick(p, "Iw"), default=None)      # cm⁶

        poids = to_float(pick(p, "Poids", "masse"), default=None)  # kg/m
        typ   = str(pick(p, "type", default=""))

        # Minimum vital pour faire au moins section en flexion/cisaillement
        if Wel_y is None or Wel_y <= 0:
            continue
        if Avz is None or Avz <= 0:
            # si Avz manque, on peut encore faire la flexion, mais pas le cisaillement
            pass

        cleaned[name] = {
            "type": typ,
            "h": h, "b": b, "tw": tw, "tf": tf, "r": r,
            "A": A,
            "Avy": Avy, "Avz": Avz,
            "Iy": Iy, "Iz": Iz,
            "Wel_y": Wel_y, "Wel_z": Wel_z,
            "It": It, "Iw": Iw,
            "Poids": poids
        }

    return cleaned

# =========================================================
# EC3 - Section resistance (simplifiée, conservatrice)
# Units:
# - NEd in kN, M in kN·m, V in kN
# - fy in MPa (N/mm²)
# - A in cm², W in cm³, Av in cm²
# =========================================================
def ec3_section_resistance(profile, fy, gamma_M0=1.0):
    # Convert to mm²/mm³
    A_mm2 = profile["A"] * 1e2 if profile.get("A") else None         # cm² -> mm²
    Wy_mm3 = profile["Wel_y"] * 1e3 if profile.get("Wel_y") else None # cm³ -> mm³
    Wz_mm3 = profile["Wel_z"] * 1e3 if profile.get("Wel_z") else None
    Avy_mm2 = profile["Avy"] * 1e2 if profile.get("Avy") else None
    Avz_mm2 = profile["Avz"] * 1e2 if profile.get("Avz") else None

    # Section resistances (ELASTIC, conservative; plastic needs Wpl)
    N_Rd = (A_mm2 * fy / gamma_M0) if A_mm2 else None                      # N
    My_Rd = (Wy_mm3 * fy / gamma_M0) if Wy_mm3 else None                   # N·mm
    Mz_Rd = (Wz_mm3 * fy / gamma_M0) if Wz_mm3 else None                   # N·mm
    Vy_Rd = (Avy_mm2 * fy / (math.sqrt(3) * gamma_M0)) if Avy_mm2 else None  # N
    Vz_Rd = (Avz_mm2 * fy / (math.sqrt(3) * gamma_M0)) if Avz_mm2 else None  # N

    return {
        "N_Rd_N": N_Rd,
        "My_Rd_Nmm": My_Rd,
        "Mz_Rd_Nmm": Mz_Rd,
        "Vy_Rd_N": Vy_Rd,
        "Vz_Rd_N": Vz_Rd
    }

def utilisation_ratio_section(NEd_kN, My_kNm, Mz_kNm, Vy_kN, Vz_kN, resist):
    # Convert actions to N / Nmm
    NEd = NEd_kN * 1e3
    My  = My_kNm * 1e6
    Mz  = Mz_kNm * 1e6
    Vy  = Vy_kN * 1e3
    Vz  = Vz_kN * 1e3

    # Individual ratios
    rN  = (NEd / resist["N_Rd_N"]) if (resist.get("N_Rd_N") and resist["N_Rd_N"] > 0) else 0.0
    rMy = (abs(My) / resist["My_Rd_Nmm"]) if (resist.get("My_Rd_Nmm") and resist["My_Rd_Nmm"] > 0) else 0.0
    rMz = (abs(Mz) / resist["Mz_Rd_Nmm"]) if (resist.get("Mz_Rd_Nmm") and resist["Mz_Rd_Nmm"] > 0) else 0.0
    rVy = (abs(Vy) / resist["Vy_Rd_N"]) if (resist.get("Vy_Rd_N") and resist["Vy_Rd_N"] > 0) else 0.0
    rVz = (abs(Vz) / resist["Vz_Rd_N"]) if (resist.get("Vz_Rd_N") and resist["Vz_Rd_N"] > 0) else 0.0

    # Interaction simple (conservative) for N+M
    # If no N, it reduces to bending only
    interaction_NM = rN + rMy + rMz

    # Shear check separately (could be enhanced with EC3 6.2.8)
    interaction = max(interaction_NM, rVy, rVz)

    return {
        "rN": rN, "rMy": rMy, "rMz": rMz, "rVy": rVy, "rVz": rVz,
        "util": interaction,
        "util_NM": interaction_NM
    }

# =========================================================
# EC3 - Buckling (compression) - requires Iy/Iz and Lcr
# =========================================================
def ec3_buckling_chi(A_cm2, I_cm4, Lcr_m, fy, E=210000.0, alpha=0.34):
    """
    A_cm2: area in cm²
    I_cm4: inertia in cm⁴ about buckling axis
    Lcr_m: buckling length in m
    fy, E in MPa (N/mm²)
    alpha: imperfection factor (curve a0..d)
    Returns chi, lambdabar, Ncr (N)
    """
    if A_cm2 is None or I_cm4 is None or Lcr_m <= 0:
        return None

    A_mm2 = A_cm2 * 1e2
    I_mm4 = I_cm4 * 1e4
    L_mm = Lcr_m * 1e3

    Ncr = (math.pi**2 * E * I_mm4) / (L_mm**2)  # N
    lambdabar = math.sqrt((A_mm2 * fy) / Ncr)

    phi = 0.5 * (1.0 + alpha * (lambdabar - 0.2) + lambdabar**2)
    chi = 1.0 / (phi + math.sqrt(max(phi**2 - lambdabar**2, 0.0)))
    chi = min(chi, 1.0)

    return {"chi": chi, "lambdabar": lambdabar, "Ncr_N": Ncr}

# =========================================================
# UI
# =========================================================
def show():
    st.title("Choix de profilé métallique optimisé (EC3)")

    profiles = load_profiles()
    if not profiles:
        st.error("Aucun profil n’a été chargé. Vérifie **profiles_test.json**.")
        return

    familles_disponibles = sorted({p["type"] for p in profiles.values() if p.get("type")})
    default_familles = ["HEA"] if "HEA" in familles_disponibles else familles_disponibles[:1]

    col_left, col_right = st.columns([1.35, 1.0])

    with col_left:
        familles_choisies = st.multiselect(
            "Types de profilés à inclure :", options=familles_disponibles, default=default_familles
        )

        st.markdown("### Efforts (ELU)")
        # Par défaut : My + Vz
        c1, c2, c3 = st.columns(3)
        with c1:
            My = st.number_input("My,Ed [kN·m] (défaut)", step=10.0, value=0.0)
        with c2:
            Vz = st.number_input("Vz,Ed [kN] (défaut)", step=10.0, value=0.0)
        with c3:
            acier = st.selectbox("Acier", ["S235", "S275", "S355"], index=0)

        fy = int(acier[1:])

        with st.expander("Ajouter d’autres efforts (optionnel)", expanded=False):
            add_N  = st.checkbox("Ajouter N,Ed (compression +)", value=False)
            add_Mz = st.checkbox("Ajouter Mz,Ed", value=False)
            add_Vy = st.checkbox("Ajouter Vy,Ed", value=False)
            add_T  = st.checkbox("Ajouter T,Ed (torsion)", value=False)

            NEd = st.number_input("N,Ed [kN]", step=50.0, value=0.0) if add_N else 0.0
            Mz  = st.number_input("Mz,Ed [kN·m]", step=10.0, value=0.0) if add_Mz else 0.0
            Vy  = st.number_input("Vy,Ed [kN]", step=10.0, value=0.0) if add_Vy else 0.0
            TEd = st.number_input("T,Ed [kN·m]", step=1.0, value=0.0) if add_T else 0.0

        st.markdown("### Paramètres EC3")
        c4, c5 = st.columns(2)
        with c4:
            gamma_M0 = st.number_input("γM0", min_value=0.8, max_value=1.5, value=1.0, step=0.05)
        with c5:
            gamma_M1 = st.number_input("γM1", min_value=0.8, max_value=1.5, value=1.0, step=0.05)

        Iv_min = st.number_input("Iy min. [cm⁴] (optionnel)", min_value=0.0, step=100.0, value=0.0)

        with st.expander("Instabilités (optionnel)", expanded=False):
            check_buckling = st.checkbox("Vérifier flambement (si N>0)", value=False)
            check_ltb = st.checkbox("Vérifier déversement (LTB) — nécessite It/Iw/Iz", value=False)

            # Flambement
            if check_buckling:
                st.markdown("**Flambement**")
                Lcr_y = st.number_input("Lcr,y [m]", min_value=0.0, step=0.1, value=0.0)
                Lcr_z = st.number_input("Lcr,z [m]", min_value=0.0, step=0.1, value=0.0)
                curve = st.selectbox("Courbe flambement (α)", ["a0 (0.13)", "a (0.21)", "b (0.34)", "c (0.49)", "d (0.76)"], index=2)
                alpha_map = {"a0 (0.13)":0.13, "a (0.21)":0.21, "b (0.34)":0.34, "c (0.49)":0.49, "d (0.76)":0.76}
                alpha_buck = alpha_map[curve]
            else:
                Lcr_y = Lcr_z = 0.0
                alpha_buck = 0.34

            # LTB (placeholder correct “données manquantes” si pas dispo)
            if check_ltb:
                st.markdown("**Déversement (LTB)**")
                Llt = st.number_input("L_LT [m] (longueur non contreventée)", min_value=0.0, step=0.1, value=0.0)
                st.caption("LTB complet nécessite : Iz, It, Iw + choix C1 / diagramme de moments. (Ajoutable dès que ta base JSON les contient.)")
            else:
                Llt = 0.0

        profils_filtres = (
            {k: v for k, v in profiles.items() if v["type"] in familles_choisies}
            if familles_choisies else profiles
        )

        rows = []
        for nom, prof in profils_filtres.items():
            # Filtre inertie (ici Iy)
            if prof.get("Iy") is not None and Iv_min > 0 and prof["Iy"] < Iv_min:
                continue

            resist = ec3_section_resistance(prof, fy=fy, gamma_M0=gamma_M0)
            ratios = utilisation_ratio_section(NEd, My, Mz, Vy, Vz, resist)

            # Option flambement (si demandé)
            buck_ok = None
            util_buck = None
            if check_buckling and NEd > 0:
                A = prof.get("A")
                Iy = prof.get("Iy")
                Iz = prof.get("Iz")
                buck_y = ec3_buckling_chi(A, Iy, Lcr_y, fy=fy, alpha=alpha_buck) if (Lcr_y > 0 and Iy) else None
                buck_z = ec3_buckling_chi(A, Iz, Lcr_z, fy=fy, alpha=alpha_buck) if (Lcr_z > 0 and Iz) else None

                # N_b,Rd = chi*A*fy/gamma_M1 (prendre le plus défavorable)
                candidates = []
                for buck in (buck_y, buck_z):
                    if buck:
                        Nb_Rd = buck["chi"] * (A*1e2) * fy / gamma_M1  # N
                        candidates.append(Nb_Rd)
                if candidates:
                    Nb_Rd_min = min(candidates)
                    util_buck = (NEd*1e3) / Nb_Rd_min
                    buck_ok = util_buck <= 1.0
                else:
                    util_buck = None
                    buck_ok = None

            # Utilisation globale : max(section, flambement) si flambement activé
            util_global = ratios["util"]
            if util_buck is not None:
                util_global = max(util_global, util_buck)

            rows.append({
                "Utilisation [%]": round(util_global * 100, 3),
                "Profilé": nom,
                "h [mm]": int(prof["h"]) if prof.get("h") else None,
                "Wel_y [cm³]": prof.get("Wel_y"),
                "Wel_z [cm³]": prof.get("Wel_z"),
                "Avy [cm²]": prof.get("Avy"),
                "Avz [cm²]": prof.get("Avz"),
                "A [cm²]": prof.get("A"),
                "Iy [cm⁴]": prof.get("Iy"),
                "Iz [cm⁴]": prof.get("Iz"),
                "Poids [kg/m]": prof.get("Poids"),
                "u_section [%]": round(ratios["util"]*100, 2),
                "u_flamb [%]": round(util_buck*100, 2) if util_buck is not None else None,
            })

        rows = sorted(rows, key=lambda x: x["Utilisation [%]"]) if rows else []

        st.subheader("📌 Profilé optimal :")
        if not rows:
            st.warning("Aucun profilé ne satisfait aux critères.")
            return

        df = pd.DataFrame(rows).set_index("Profilé")

        best_name = None
        util_series = df["Utilisation [%]"]
        le100 = util_series <= 100.0
        if le100.any():
            best_name = (100.0 - util_series[le100]).idxmin()

        noms = df.index.tolist()
        default_idx = noms.index(best_name) if best_name in noms else 0
        nom_selectionne = st.selectbox("Sélectionner un profilé :", options=noms, index=default_idx)

        def _row_style(row):
            u = row["Utilisation [%]"]
            if row.name == best_name:
                color = "#b7f7c1"
            elif u <= 100:
                color = "#eafaf0"
            else:
                color = "#ffeaea"
            return [f"background-color: {color}"] * len(row)

        if st.checkbox("Afficher tous les profilés ✓/✗", value=True):
            st.dataframe(
                df.style.apply(_row_style, axis=1)
                       .format(lambda v: fmt_no_trailing_zeros(v, digits=3)),
                use_container_width=True
            )

    # =========================================================
    # Colonne droite : fiche profil + formules
    # =========================================================
    with col_right:
        profil = profiles[nom_selectionne]
        st.markdown(f"### {nom_selectionne}")

        c1, c2 = st.columns(2)

        dims = [
            ("h [mm]",  profil.get("h")),
            ("b [mm]",  profil.get("b")),
            ("tw [mm]", profil.get("tw")),
            ("tf [mm]", profil.get("tf")),
            ("r [mm]",  profil.get("r")),
        ]
        props = [
            ("Poids [kg/m]", profil.get("Poids")),
            ("A [cm²]",      profil.get("A")),
            ("Wel_y [cm³]",  profil.get("Wel_y")),
            ("Wel_z [cm³]",  profil.get("Wel_z")),
            ("Avy [cm²]",    profil.get("Avy")),
            ("Avz [cm²]",    profil.get("Avz")),
            ("Iy [cm⁴]",     profil.get("Iy")),
            ("Iz [cm⁴]",     profil.get("Iz")),
        ]

        with c1:
            df_dims = pd.DataFrame({"Dimensions": [d[0] for d in dims],
                                    "Valeur": [fmt_no_trailing_zeros(d[1]) for d in dims]})
            st.dataframe(df_dims, hide_index=True, use_container_width=True)

        with c2:
            df_props = pd.DataFrame({"Propriété": [p[0] for p in props],
                                     "Valeur": [fmt_no_trailing_zeros(p[1]) for p in props]})
            st.dataframe(df_props, hide_index=True, use_container_width=True)

        st.subheader("Formules EC3 — vérification section")

        resist = ec3_section_resistance(profil, fy=fy, gamma_M0=gamma_M0)
        ratios = utilisation_ratio_section(NEd, My, Mz, Vy, Vz, resist)

        # LaTeX : résistances
        if resist.get("N_Rd_N") is not None:
            N_Rd_kN = resist["N_Rd_N"] / 1e3
            st.latex(
                rf"N_{{Rd}} = \frac{{A f_y}}{{\gamma_{{M0}}}}"
                rf" = \frac{{{profil['A']:.2f}\times 10^2 \cdot {fy}}}{{{gamma_M0:.2f}}}"
                rf" = {N_Rd_kN:.1f}\ \text{{kN}}"
            )
        else:
            st.info("N_Rd : A manquant dans la base (champ `A`).")

        # Flexion My
        if resist.get("My_Rd_Nmm") is not None:
            My_Rd = resist["My_Rd_Nmm"] / 1e6
            st.latex(
                rf"M_{{y,Rd}} = \frac{{W_{{el,y}} f_y}}{{\gamma_{{M0}}}}"
                rf" = \frac{{{profil['Wel_y']:.2f}\times 10^3 \cdot {fy}}}{{{gamma_M0:.2f}}}"
                rf" = {My_Rd:.1f}\ \text{{kN·m}}"
            )
        else:
            st.info("My,Rd : Wel_y manquant.")

        # Flexion Mz si dispo et si demandé
        if add_Mz:
            if resist.get("Mz_Rd_Nmm") is not None:
                Mz_Rd = resist["Mz_Rd_Nmm"] / 1e6
                st.latex(
                    rf"M_{{z,Rd}} = \frac{{W_{{el,z}} f_y}}{{\gamma_{{M0}}}}"
                    rf" = \frac{{{profil.get('Wel_z'):.2f}\times 10^3 \cdot {fy}}}{{{gamma_M0:.2f}}}"
                    rf" = {Mz_Rd:.1f}\ \text{{kN·m}}"
                )
            else:
                st.warning("Mz,Rd : Wel_z manquant dans la base (champ `Welz`/`Wel_z`).")

        # Cisaillement
        if resist.get("Vz_Rd_N") is not None:
            Vz_Rd = resist["Vz_Rd_N"] / 1e3
            st.latex(
                rf"V_{{z,Rd}} = \frac{{A_{{vz}} f_y}}{{\sqrt{{3}}\gamma_{{M0}}}}"
                rf" = \frac{{{profil['Avz']:.2f}\times 10^2 \cdot {fy}}}{{\sqrt{{3}}\cdot {gamma_M0:.2f}}}"
                rf" = {Vz_Rd:.1f}\ \text{{kN}}"
            )
        else:
            st.info("Vz,Rd : Avz manquant dans la base (champ `Avz`).")

        if add_Vy:
            if resist.get("Vy_Rd_N") is not None:
                Vy_Rd = resist["Vy_Rd_N"] / 1e3
                st.latex(
                    rf"V_{{y,Rd}} = \frac{{A_{{vy}} f_y}}{{\sqrt{{3}}\gamma_{{M0}}}}"
                    rf" = {Vy_Rd:.1f}\ \text{{kN}}"
                )
            else:
                st.warning("Vy,Rd : Avy manquant dans la base (champ `Avy`).")

        # LaTeX : utilisations
        st.subheader("Utilisations (ratios)")
        st.latex(rf"\eta_N = \frac{{N_{{Ed}}}}{{N_{{Rd}}}} = {ratios['rN']:.3f}")
        st.latex(rf"\eta_{{My}} = \frac{{|M_{{y,Ed}}|}}{{M_{{y,Rd}}}} = {ratios['rMy']:.3f}")
        if add_Mz:
            st.latex(rf"\eta_{{Mz}} = \frac{{|M_{{z,Ed}}|}}{{M_{{z,Rd}}}} = {ratios['rMz']:.3f}")
        if add_Vy:
            st.latex(rf"\eta_{{Vy}} = \frac{{|V_{{y,Ed}}|}}{{V_{{y,Rd}}}} = {ratios['rVy']:.3f}")
        st.latex(rf"\eta_{{Vz}} = \frac{{|V_{{z,Ed}}|}}{{V_{{z,Rd}}}} = {ratios['rVz']:.3f}")
        st.latex(rf"\eta_{{NM}} = \eta_N + \eta_{{My}} + \eta_{{Mz}} = {ratios['util_NM']:.3f}")
        st.latex(rf"\eta_{{section}} = \max(\eta_{{NM}},\eta_{{Vy}},\eta_{{Vz}}) = {ratios['util']:.3f}")

        # Instabilités affichées si activées
        if check_buckling and NEd > 0:
            st.subheader("Instabilité — flambement (si données dispo)")
            A = profil.get("A")
            Iy = profil.get("Iy")
            Iz = profil.get("Iz")

            buck_y = ec3_buckling_chi(A, Iy, Lcr_y, fy=fy, alpha=alpha_buck) if (Lcr_y > 0 and Iy) else None
            buck_z = ec3_buckling_chi(A, Iz, Lcr_z, fy=fy, alpha=alpha_buck) if (Lcr_z > 0 and Iz) else None

            if not buck_y and not buck_z:
                st.warning("Impossible de calculer le flambement : vérifier A, Iy/Iz et Lcr,y/Lcr,z.")
            else:
                if buck_y:
                    st.latex(rf"\chi_y = {buck_y['chi']:.3f}\quad;\quad \bar\lambda_y = {buck_y['lambdabar']:.3f}")
                if buck_z:
                    st.latex(rf"\chi_z = {buck_z['chi']:.3f}\quad;\quad \bar\lambda_z = {buck_z['lambdabar']:.3f}")
                st.caption("N_b,Rd = χ·A·fy/γM1 (axe le plus défavorable).")

        if check_ltb:
            st.subheader("Instabilité — déversement (LTB)")
            st.info("LTB complet à activer dès que ta base JSON contient Iz, It, Iw et qu’on implémente Mcr + χLT. (La structure UI est prête.)")


if __name__ == "__main__":
    show()

import streamlit as st
import json
import math
import pandas as pd

# ---------- Chargement / nettoyage des profils ----------
@st.cache_data
def load_profiles():
    def pick(d, *keys, default=None):
        for k in keys:
            if k in d and d[k] not in (None, "None", ""):
                return d[k]
        return default

    with open("profiles_test.json", "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaned = {}
    for name, p in raw.items():
        try:
            h      = float(pick(p, "h"))
            Wel    = float(pick(p, "Wel"))          # cm³
            Avz    = float(pick(p, "Avz"))          # cm²
            Iv     = pick(p, "Iv", "Iy")            # cm⁴ (optionnel)
            Iv     = None if Iv in (None, "None", "") else float(Iv)
            poids  = float(pick(p, "Poids", "masse"))
            typ    = str(pick(p, "type"))

            b      = float(pick(p, "b",  default=0.0))
            tw     = float(pick(p, "tw", default=0.0))
            tf     = float(pick(p, "tf", default=0.0))
            r      = float(pick(p, "r",  default=0.0))
            A      = float(pick(p, "A",  default=0.0))

            if Wel <= 0 or Avz <= 0:
                continue

            cleaned[name] = {
                "h": h, "Wel": Wel, "Avz": Avz, "Iv": Iv,
                "Poids": poids, "type": typ, "b": b, "tw": tw, "tf": tf, "r": r, "A": A
            }
        except Exception as e:
            print(f"⚠️ Profil ignoré ({name}) : {e}")

    return cleaned


# ---------- Calculs ----------
def calcul_contraintes(profile, M, V, fyk):
    Wel_mm3 = profile["Wel"] * 1e3   # cm³ → mm³
    Avz_mm2 = profile["Avz"] * 1e2   # cm² → mm²
    sigma_n = (M * 1e6) / Wel_mm3    # MPa
    tau     = (V * 1e3) / Avz_mm2    # MPa
    sigma_eq = math.sqrt(sigma_n**2 + 3 * tau**2)
    utilisation = sigma_eq / (fyk / 1.5)
    return sigma_n, tau, sigma_eq, utilisation


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


# ---------- UI ----------
def show():
    st.title("Choix de profilé métallique optimisé")

    profiles = load_profiles()
    if not profiles:
        st.error("Aucun profil n’a été chargé. Vérifie que **profiles_test.json** existe et contient `h`, `Wel`, `Avz`, `masse/Poids`, `type`.")
        return

    familles_disponibles = sorted({p["type"] for p in profiles.values()})
    default_familles = ["HEA"] if "HEA" in familles_disponibles else familles_disponibles[:1]

    col_left, col_right = st.columns([1.35, 1.0])

    with col_left:
        familles_choisies = st.multiselect(
            "Types de profilés à inclure :", options=familles_disponibles, default=default_familles
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            M = st.number_input("M [kN·m]", min_value=0.0, step=10.0, value=0.0)
        with c2:
            V = st.number_input("V [kN]", min_value=0.0, step=10.0, value=0.0)
        with c3:
            acier = st.selectbox("Acier", ["S235", "S275", "S355"], index=0)
        fyk = int(acier[1:])
        Iv_min = st.number_input("Iv min. [cm⁴] (optionnel)", min_value=0.0, step=100.0, value=0.0)

        profils_filtres = (
            {k: v for k, v in profiles.items() if v["type"] in familles_choisies}
            if familles_choisies else profiles
        )

        rows = []
        for nom, prof in profils_filtres.items():
            if prof["Iv"] is not None and Iv_min > 0 and prof["Iv"] < Iv_min:
                continue
            sigma_n, tau, sigma_eq, utilisation = calcul_contraintes(prof, M, V, fyk)
            rows.append({
                "Utilisation [%]": round(utilisation * 100, 3),
                "Profilé": nom,
                "h [mm]": int(prof["h"]),
                "Wel [cm³]": prof["Wel"],
                "Avz [cm²]": prof["Avz"],
                "Iv [cm⁴]": prof["Iv"],
                "Poids [kg/m]": prof["Poids"],
                "σ [MPa]": sigma_n,
                "τ [MPa]": tau,
                "σeq [MPa]": sigma_eq,
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

    with col_right:
        profil = profiles[nom_selectionne]
        st.markdown(f"### {nom_selectionne}")

        # 2 petits tableaux côte à côte SANS titres au-dessus
        c1, c2 = st.columns(2)

        dims = [
            ("h [mm]",  profil["h"]),
            ("b [mm]",  profil["b"]),
            ("tw [mm]", profil["tw"]),
            ("tf [mm]", profil["tf"]),
            ("r [mm]",  profil["r"]),
        ]
        props = [
            ("Poids [kg/m]", profil["Poids"]),
            ("A [cm²]",     profil["A"]),
            ("Wel [cm³]",   profil["Wel"]),
            ("Avz [cm²]",   profil["Avz"]),
            ("Iv [cm⁴]",    profil["Iv"]),
        ]

        with c1:
            df_dims = pd.DataFrame(
                {"Dimensions": [d[0] for d in dims],
                 "Valeur": [fmt_no_trailing_zeros(d[1]) for d in dims]}
            )
            st.dataframe(df_dims, hide_index=True, use_container_width=True)

        with c2:
            df_props = pd.DataFrame(
                {"Propriété": [p[0] for p in props],
                 "Valeur": [fmt_no_trailing_zeros(p[1]) for p in props]}
            )
            st.dataframe(df_props, hide_index=True, use_container_width=True)

        # Formules rendues comme avant (st.latex)
        sigma_n, tau, sigma_eq, utilisation = calcul_contraintes(profil, M, V, fyk)

        st.subheader("Formules de dimensionnement")
        st.latex(
            r"\sigma_n = \frac{M \times 10^6}{W_{el} \times 10^3}"
            rf" = \frac{{{M:.1f} \times 10^6}}{{{profil['Wel']:.1f} \times 10^3}}"
            rf" = {sigma_n:.2f}\ \text{{MPa}}"
        )
        st.latex(
            r"\tau = \frac{V \times 10^3}{A_{vz} \times 10^2}"
            rf" = \frac{{{V:.1f} \times 10^3}}{{{profil['Avz']:.2f} \times 10^2}}"
            rf" = {tau:.2f}\ \text{{MPa}}"
        )
        st.latex(
            rf"\sigma_{{eq}} = \sqrt{{\sigma_n^2 + 3\tau^2}}"
            rf" = \sqrt{{{sigma_n:.2f}^2 + 3 \times {tau:.2f}^2}}"
            rf" = {sigma_eq:.2f}\ \text{{MPa}}"
        )
        st.latex(
            rf"\text{{Utilisation}} = \frac{{\sigma_{{eq}}}}{{f_{{yk}}/1.5}}"
            rf" = \frac{{{sigma_eq:.2f}}}{{{(fyk/1.5):.1f}}}"
            rf" = {utilisation*100:.1f}\ \%"
        )


if __name__ == "__main__":
    show()

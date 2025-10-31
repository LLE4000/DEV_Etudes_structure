import streamlit as st

# Équivalences entre classes et descriptions
exposition_info = {
    "X0": (
        "Aucun risque de corrosion ni d’attaque",
        "Béton à l’intérieur de bâtiments où le taux d’humidité de l’air ambiant est très faible"
    ),
    "XC1": (
        "Sec ou humide en permanence",
        "Béton à l’intérieur de bâtiments où le taux d’humidité est faible"
    ),
    "XC2/XC3": (
        "Humidité modérée / rarement sec",
        "Béton extérieur abrité de la pluie ou en fondation"
    ),
    "XC4": (
        "Alternativement humide et sec",
        "Béton soumis à un contact fréquent avec l’eau"
    ),
    "XD1/XS1": (
        "Humidité modérée / Air salin",
        "Surfaces exposées aux chlorures ou près de la mer"
    ),
    "XD2/XS2": (
        "Humide, rarement sec / Immersion",
        "Piscines, structures marines immergées"
    ),
    "XD3/XS3": (
        "Alternativement humide et sec / Embruns",
        "Ponts ou dalles exposés aux chlorures ou embruns marins"
    )
}

classe_structurale_defaults = {
    "X0": "S1", "XC1": "S2", "XC2/XC3": "S4",
    "XC4": "S5", "XD1/XS1": "S5", "XD2/XS2": "S5", "XD3/XS3": "S6"
}

valeurs_cmin_dur = {
    "S1": [10, 10, 10, 15, 20, 25, 30],
    "S2": [10, 10, 15, 20, 25, 30, 35],
    "S3": [10, 15, 20, 25, 30, 35, 40],
    "S4": [15, 20, 25, 30, 35, 40, 45],
    "S5": [20, 25, 30, 35, 40, 45, 50],
    "S6": [25, 30, 35, 40, 45, 50, 55]
}

exposition_classes = list(exposition_info.keys())

def show():
    st.title("🧱 Calcul de l'enrobage du béton")
    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.header("Entrée des données")
        st.subheader("🔹 Paramètres généraux")

        type_element = st.selectbox("Type d'élément", ["Poutre", "Poteau", "Dalle", "Voile"])
        position = st.selectbox("Position dans l’ouvrage", ["Intérieur", "Extérieur"])

        # Menu déroulant unique avec info-bulle
        classe_exposition = st.selectbox(
            "Classe d’exposition",
            exposition_classes,
            format_func=lambda x: f"{x} — {exposition_info[x][0]}"
        )

        # Affichage info-bulle dynamique
        desc, exemple = exposition_info[classe_exposition]
        st.caption(f"**Description** : {desc}\n\n*Exemple* : _{exemple}_")

        classe_structurale = st.selectbox(
            "Classe structurale",
            list(valeurs_cmin_dur.keys()),
            index=list(valeurs_cmin_dur.keys()).index(classe_structurale_defaults[classe_exposition])
        )

        diametre_armature = st.number_input("Diamètre max des armatures [mm]", value=20, step=1, min_value=6)
        tolerance_constructive = st.radio("Tolérance constructive", [10, 7], index=0)

        st.subheader("🔹 Paramètres feu")
        check_feu = st.checkbox("Vérifier la résistance au feu")
        if check_feu:
            st.info("🔒 A venir dans la prochaine mise à jour.")

    # --- Résultats ---
    with col_result:
        st.header("Résultats")

        idx = exposition_classes.index(classe_exposition)
        c_min_dur = valeurs_cmin_dur[classe_structurale][idx]
        c_min_b = diametre_armature
        delta_c_dev = tolerance_constructive

        c_min = max(c_min_b, c_min_dur)
        c_nom = c_min + delta_c_dev

        st.subheader("🔹 Section A – Enrobage pour durabilité")
        st.markdown(f"- **c_min,dur** : {c_min_dur} mm")
        st.markdown(f"- **c_min,b** : {c_min_b} mm")
        st.markdown(f"- **Δc_dev** : {delta_c_dev} mm")
        st.markdown(f"- **c_nom,durabilité** = {c_nom} mm")

        st.subheader("🔹 Section B – Enrobage pour résistance au feu")
        if not check_feu:
            st.markdown("- Non requis")
        else:
            st.markdown("- 🔒 *À venir dans la prochaine mise à jour*")

        st.subheader("🔹 Section C – Résumé final")
        st.markdown(f"- **Enrobage requis** = {c_nom} mm")

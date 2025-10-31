import streamlit as st

def show():
    # --- Titre + bouton Accueil ---
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("## 📊 Tableau des sections d’armatures")
    with col2:
        if st.button("🏠 Accueil", key="retour_accueil_armatures"):
            st.session_state.page = "Accueil"

    st.markdown("---")

    # --- Affichage du tableau sans avertissement ---
    st.image("Tableau armature.png", caption="Tableau des sections d’armatures en mm² et espacement fixe (cm)", use_container_width=True)

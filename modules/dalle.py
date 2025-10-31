import streamlit as st

def show():
    # Ligne de titre avec bouton de retour
    col1, col2 = st.columns([7, 1])
    with col1:
        st.title("Dalle en béton armé")
    with col2:
        if st.button("↩️ Accueil"):
            st.session_state.page = "Accueil"

    # Saisie des données
    ep = st.number_input("Épaisseur (cm)", key="dalle_ep")
    portee = st.number_input("Portée (m)", key="dalle_portee")
    st.write("Épaisseur =", ep, "| Portée =", portee)

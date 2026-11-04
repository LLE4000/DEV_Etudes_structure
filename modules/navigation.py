# -*- coding: utf-8 -*-
"""Navigation partagée : le retour à l'accueil, une seule façon de faire.

Les tuiles de l'accueil ouvrent une page par l'URL (``?page=…``). Tant que
ce paramètre restait dans l'URL, ``streamlit_app.py`` le réappliquait à
CHAQUE exécution et aucun bouton « Accueil » ne pouvait tenir : la page
revenait à celle de l'URL au rerun suivant (défaut constaté le
04/11/2026 sur tous les modules). Le retour efface donc le paramètre, puis
demande la page d'accueil et relance ; ``streamlit_app.py`` n'applique
plus le paramètre d'URL qu'à son changement.
"""
import streamlit as st


def aller_accueil() -> None:
    """Efface le paramètre d'URL, demande l'accueil et relance."""
    try:
        st.query_params.clear()
    except Exception:  # noqa: BLE001 — anciennes versions sans query_params
        pass
    st.session_state["page"] = "Accueil"
    st.session_state["retour_accueil_demande"] = False
    st.session_state["_page_param_vu"] = None
    st.rerun()


def bouton_accueil(key: str, **kwargs) -> None:
    """Le bouton « 🏠 Accueil » d'un module ; ``kwargs`` passent à st.button
    (use_container_width, width…)."""
    if st.button("🏠 Accueil", key=key, **kwargs):
        aller_accueil()

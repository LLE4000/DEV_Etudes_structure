# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGES MÉTALLIQUES — page de sélection
# ===========================
#  assemblages.py (Streamlit) — v1.0
#
#  Point d'entrée « Acier → Assemblages métalliques » : présente les familles
#  du registre (acier/registre.py) en cartes avec vignette, puis aiguille vers
#  l'écran de l'assemblage choisi (clé de session `asm_courant`). Ajouter un
#  assemblage = une entrée dans le registre ; cette page ne change pas.
#
#  ESPACE DE NOMS DE SESSION : `asm_*` (assemblages), disjoint des clés
#  `b…` / `dal…` / `pre…` des modules béton.
# ===========================
import streamlit as st

from acier import registre

ASSEMBLAGES_VERSION = "1.0"


def _ouvrir(identifiant):
    st.session_state["asm_courant"] = identifiant


def retour_selection():
    """Revenir à la page de sélection (appelé par l'écran d'un assemblage)."""
    st.session_state.pop("asm_courant", None)


def _page_selection():
    if "retour_accueil_demande" not in st.session_state:
        st.session_state.retour_accueil_demande = False
    if st.session_state.retour_accueil_demande:
        st.session_state.page = "Accueil"
        st.session_state.retour_accueil_demande = False
        st.rerun()

    tH1, tH2, tH3 = st.columns([8, 1.6, 0.55], vertical_alignment="center")
    with tH1:
        st.markdown("## Assemblages métalliques")
    with tH2:
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {ASSEMBLAGES_VERSION}</div>",
            unsafe_allow_html=True,
        )
    with tH3:
        st.button("❔", key="asm_btn_version_hist", help="Bibliothèque d'assemblages métalliques — "
                  "EN 1993-1-8. Chaque assemblage est un module complet : données, "
                  "calcul, schémas, note de calcul.", use_container_width=True)

    btn1, _, _, _, _ = st.columns(5)
    with btn1:
        if st.button("🏠 Accueil", use_container_width=True, key="asm_btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()

    for fam in registre.FAMILLES:
        st.markdown(f"### {fam.titre}")
        st.caption(fam.pitch)
        cartes = list(fam.assemblages)
        for i in range(0, len(cartes), 4):
            cols = st.columns(4)
            for col, a in zip(cols, cartes[i:i + 4]):
                with col:
                    with st.container(border=True):
                        st.markdown(a.vignette, unsafe_allow_html=True)
                        st.markdown(f"**{a.titre}**")
                        st.caption(a.pitch)
                        if a.disponible:
                            st.button("Ouvrir", key=f"asm_btn_open_{a.id}", use_container_width=True,
                                      on_click=_ouvrir, args=(a.id,))
                        else:
                            st.markdown("<div style='text-align:center;color:#9ca3af;font-size:0.9em;'>"
                                        "À venir</div>", unsafe_allow_html=True)


def show():
    courant = st.session_state.get("asm_courant")
    if courant:
        trouve = registre.trouver(courant)
        if trouve and trouve[1].disponible:
            registre.charger(trouve[1]).show()
            return
        st.session_state.pop("asm_courant", None)
    _page_selection()

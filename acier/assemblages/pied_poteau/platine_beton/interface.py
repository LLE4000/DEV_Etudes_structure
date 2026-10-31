# -*- coding: utf-8 -*-
# ===========================
#  PIED DE POTEAU — PLATINE SUR BÉTON (N + Mx + My) — VERSION 1.0
# ===========================
#  interface.py (Streamlit)
#
#  Même écran que les assemblages poutre–poutre : barre d'outils, statut une
#  ligne, alertes courtes, puis DEUX COLONNES — à gauche le dessin (coupe A-A
#  puis vue en plan avec la répartition, colonne FIGÉE par CSS sticky), à
#  droite la carte compacte par objet, les replis et les onglets.
#
#  SOURCE UNIQUE : les entrées vivent dans st.session_state sous les clés
#  `ppb_<clé>` (espace disjoint des clés `asm_`, `fpl_`, `pab_` et `prj_`).
#  Le dessin est rendu AVANT la carte (messages appliqués avant widgets).
# ===========================
from datetime import datetime
from functools import lru_cache

import streamlit as st

from acier import projet_ui
from . import ecran_saisie, ecran_resultats, texte
from .entrees import CLES, defaults, charger_json, VISSERIE_DEFAUT
from .moteur import compute
from .benchmark import run_bench

MODULE_VERSION = "1.0"
PREFIXE = "ppb_"
_TRANSITOIRES = ("btn", "uploader", "pdf_bytes", "_ppb_", "ppb_cmp_", "ppb_cote_", "ppb_fix_")

_CSS_STICKY = """<style>
@media (min-width: 641px) {
  div[data-testid="stColumn"]:has(div.st-key-ppb_col_dessin) {
    position: sticky;
    top: 2.875rem;
    align-self: flex-start;
    max-height: calc(100vh - 3.5rem);
    overflow-y: auto;
    scrollbar-width: thin;
  }
  div[data-testid="stMainBlockContainer"] {
    padding-bottom: 1.5rem;
  }
}
</style>"""


def K(k):
    return PREFIXE + k


@lru_cache(maxsize=1)
def _bench():
    return run_bench()


def _transitoire(k):
    return any(m in k for m in _TRANSITOIRES)


def _init_etat():
    for k, v in defaults().items():
        st.session_state.setdefault(K(k), ecran_saisie.valeur_widget(k, v))
    st.session_state.setdefault("ppb_visserie", VISSERIE_DEFAUT)
    st.session_state.setdefault("ppb_ui_niveau", 1)
    st.session_state.setdefault("ppb_ui_alerte", None)
    st.session_state.setdefault("ppb_ui_onglet", ecran_resultats.ONGLETS[0])
    st.session_state.setdefault("ppb_ui_txt_complet", False)
    st.session_state.setdefault("ppb_ui_composant", True)
    st.session_state.setdefault("ppb_ui_dernier_clic", {})


def _epingler():
    """FIX PERSISTANCE : ré-affecter chaque clé persistante pour que
    Streamlit ne nettoie pas l'état des widgets non rendus."""
    for k in list(st.session_state.keys()):
        if k.startswith(PREFIXE) and not _transitoire(k):
            st.session_state[k] = st.session_state[k]


def lire_entrees():
    return {k: st.session_state.get(K(k)) for k in CLES}


def ecrire_entrees(u):
    for k in CLES:
        st.session_state[K(k)] = ecran_saisie.valeur_widget(k, u.get(k))


def _reinitialiser():
    for k in list(st.session_state.keys()):
        if k.startswith(PREFIXE):
            del st.session_state[k]
    projet_ui.entree_reinitialisee()      # l'entrée du projet suit (repère conservé)
    st.session_state["_ppb_toast"] = "Valeurs par défaut rétablies"


def _payload(u):
    return {"version": "assemblage-pied-poteau-1.0", "assemblage": "platine_beton",
            "values": u,
            "visserie": st.session_state.get("ppb_visserie", VISSERIE_DEFAUT)}


def charger_payload(data):
    if not isinstance(data, dict):
        raise ValueError("Structure JSON inattendue")
    src = data.get("values") if isinstance(data.get("values"), dict) else data
    return charger_json(src)


def _infos(u):
    return {"nom_projet": st.session_state.get("nom_projet", "") or u.get("id_projet", ""),
            "partie": st.session_state.get("partie", "") or u.get("id_rep", ""),
            "date": st.session_state.get("date", "") or u.get("id_date", "")
            or datetime.today().strftime("%d/%m/%Y"),
            "indice": st.session_state.get("indice", "0"),
            "visserie": st.session_state.get("ppb_visserie", VISSERIE_DEFAUT)}


# ------------------------------------------------------------------ écran
def show():
    _init_etat()
    _epingler()
    if "retour_accueil_demande" not in st.session_state:
        st.session_state.retour_accueil_demande = False
    if st.session_state.retour_accueil_demande:
        st.session_state.page = "Accueil"
        st.session_state.retour_accueil_demande = False
        st.rerun()
    msg = st.session_state.pop("_ppb_toast", None)
    if msg:
        st.toast(msg)

    tH1, tH2, tH3 = st.columns([8, 1.6, 0.55], vertical_alignment="center")
    with tH1:
        st.markdown("## Pied de poteau – platine sur béton")
    with tH2:
        st.markdown(f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {MODULE_VERSION}</div>",
                    unsafe_allow_html=True)
    with tH3:
        st.button("❔", key="ppb_btn_version", use_container_width=True,
                  help="Vérification ELU sous N + Mx + My (+ Vx, Vy, T) selon EN 1993-1-8 §6.2.5 / §6.2.8 (aire "
                       "efficace, tronçons en T, Table 6.7), EN 1992-1-1 §6.7 et EN 1992-4 (tiges coulées en "
                       "place) ; γM0 ANB belge. Répartition élastique sans éléments finis. Benchmark : calculs "
                       "manuels rejoués à chaque ouverture (exemples publiés au lot C).")

    # Barre d'outils : ◀ Projet · 🏠 · 🔄 · 💾 (projet entier) · [A2 ▾] · 📄 —
    # hors projet (écran ouvert directement), 💾 et 📂 gardent l'assemblage seul.
    b0, b1, b2, b3, b4, b5 = st.columns(6)
    with b0:
        projet_ui.bouton_retour("ppb_btn_retour")
    with b1:
        if st.button("🏠 Accueil", use_container_width=True, key="ppb_btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()
    with b2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="ppb_btn_reset",
                     disabled=projet_ui.verrouille()):
            _reinitialiser()
            st.rerun()
    u = lire_entrees()
    with b3:
        projet_ui.bouton_enregistrer("ppb_btn_save_dl", lambda: _payload(u), lambda: texte.nom_fichier(u, ".json"))
    with b4:
        if not projet_ui.selecteur(PREFIXE):
            projet_ui.bouton_ouvrir_seul(PREFIXE, charger_payload, ecrire_entrees)

    try:
        R = compute(u)
        erreur = None
    except Exception as e:  # noqa: BLE001 — l'écran doit rester utilisable
        R, erreur = None, f"{type(e).__name__}: {e}"

    with b5:
        if st.button("📄 Générer PDF", use_container_width=True, key="ppb_btn_pdf", disabled=R is None,
                     help="Note de calcul de 2 pages (A4 paysage) : synthèse + plan de principe."):
            from .note import generer_note
            try:
                st.session_state["ppb_pdf_bytes"] = generer_note(R, _infos(u))
                st.success("✅ Note de calcul générée")
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("ppb_pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")
        if st.session_state.get("ppb_pdf_bytes"):
            st.download_button("⬇️ Télécharger la note PDF", data=st.session_state["ppb_pdf_bytes"],
                               file_name=texte.nom_fichier(u, ".pdf"), mime="application/pdf",
                               use_container_width=True, key="ppb_btn_pdf_dl")

    projet_ui.bandeau_verrou(PREFIXE)
    if R is None:
        st.error(f"Données incomplètes : {erreur}")
        ecran_saisie.carte(u, set())
        return

    ecran_resultats.bandeau(R)
    ecran_resultats.alertes(R, u)
    chauds = set()
    sel = st.session_state.get("ppb_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            chauds = set(a.fields)
    st.markdown(_CSS_STICKY, unsafe_allow_html=True)
    c_dessin, c_droite = st.columns([1, 1.12], gap="medium")
    with c_dessin:
        with st.container(key="ppb_col_dessin"):
            e, p, hl = ecran_resultats.dessins(R, u)
    with c_droite:
        if not st.session_state.get("ppb_ui_composant", True):
            ecran_resultats.panneau_cotes(e, p, hl)
        ecran_saisie.carte(u, chauds)
        ecran_resultats.onglets(R, _bench(), u, ecrire_entrees)

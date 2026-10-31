# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGE POUTRE–POUTRE — PLATINE D'ABOUT DE HAUTEUR PARTIELLE — VERSION 1.0
# ===========================
#  interface.py (Streamlit)
#
#  Titre court et barre d'outils partagée (acier/projet_ui.py), statut une
#  ligne, alertes courtes, puis DEUX COLONNES — à gauche le dessin (élévation
#  puis vue en plan, colonne FIGÉE par CSS sticky), à droite la carte par
#  objet rendue par le kit (acier/assemblages/carte.py), les replis et les
#  onglets. Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md).
#
#  SOURCE UNIQUE : les entrées vivent dans st.session_state sous les clés
#  `pab_<clé>` (espace disjoint des clés `asm_` et `fpl_` des deux autres).
#  Le dessin est rendu AVANT la carte (messages appliqués avant widgets).
# ===========================
from datetime import datetime
from functools import lru_cache

import streamlit as st

from acier import projet_ui
from acier.assemblages import carte as kit
from . import ecran_saisie, ecran_resultats, texte
from .entrees import CLES, defaults, charger_json, VISSERIE_DEFAUT
from .moteur import compute
from .benchmark import run_bench

MODULE_VERSION = "1.0"
PREFIXE = "pab_"
TITRE_COURT = "Platine d'about partielle"
TITRE_LONG = "Assemblage poutre–poutre – platine d'about de hauteur partielle"
_TRANSITOIRES = ("btn", "uploader", "pdf_bytes", "_pab_", "pab_cmp_", "pab_cote_", "pab_fix_")

_CSS_STICKY = """<style>
@media (min-width: 641px) {
  div[data-testid="stColumn"]:has(div.st-key-pab_col_dessin) {
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
    st.session_state.setdefault("pab_visserie", VISSERIE_DEFAUT)
    st.session_state.setdefault("pab_ui_niveau", 1)
    st.session_state.setdefault("pab_ui_alerte", None)
    st.session_state.setdefault("pab_ui_onglet", ecran_resultats.ONGLETS[0])
    st.session_state.setdefault("pab_ui_txt_complet", False)
    st.session_state.setdefault("pab_ui_composant", True)
    st.session_state.setdefault("pab_ui_dernier_clic", {})


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
    st.session_state["_pab_toast"] = "Valeurs par défaut rétablies"


def _payload(u):
    return {"version": "assemblage-platine-about-1.0", "assemblage": "platine_about",
            "values": u,
            "visserie": st.session_state.get("pab_visserie", VISSERIE_DEFAUT)}


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
            "visserie": st.session_state.get("pab_visserie", VISSERIE_DEFAUT)}


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
    msg = st.session_state.pop("_pab_toast", None)
    if msg:
        st.toast(msg)

    projet_ui.en_tete(TITRE_COURT, TITRE_LONG + " · version " + MODULE_VERSION)
    u = lire_entrees()
    try:
        R = compute(u)
        erreur = None
    except Exception as e:  # noqa: BLE001 — l'écran doit rester utilisable
        R, erreur = None, f"{type(e).__name__}: {e}"

    def generer_pdf():
        from .note import generer_note
        return generer_note(R, _infos(u))

    # Barre d'outils partagée : ◀ Projet · 🏠 · 💾 (projet entier) · 📄 · [A2 ▾] · 🔄 —
    # hors projet (écran ouvert directement), 💾 et 📂 gardent l'assemblage seul.
    projet_ui.barre_outils(PREFIXE, lambda: _payload(u), lambda ext: texte.nom_fichier(u, ext), charger_payload,
                           ecrire_entrees, _reinitialiser, generer_pdf, pdf_disabled=R is None,
                           pdf_help="Note de calcul de 2 pages (A4 paysage) : synthèse + plan de principe.")
    projet_ui.bandeau_verrou(PREFIXE)
    st.markdown(_CSS_STICKY, unsafe_allow_html=True)
    kit.injecter_css(PREFIXE)
    if R is None:
        st.error(f"Données incomplètes : {erreur}")
        ecran_saisie.carte(u, set())
        return

    ecran_resultats.bandeau(R)
    ecran_resultats.alertes(R, u)
    chauds = set()
    sel = st.session_state.get("pab_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            chauds = set(a.fields)
    c_dessin, c_droite = st.columns([1, 1.12], gap="medium")
    with c_dessin:
        with st.container(key="pab_col_dessin"):
            e, p, hl = ecran_resultats.dessins(R, u)
    with c_droite:
        if not st.session_state.get("pab_ui_composant", True):
            ecran_resultats.panneau_cotes(e, p, hl)
        ecran_saisie.carte(u, chauds)
        ecran_resultats.onglets(R, _bench(), u, ecrire_entrees)

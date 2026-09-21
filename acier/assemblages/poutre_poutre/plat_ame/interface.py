# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGE POUTRE–POUTRE — PLAT D'ÂME SOUDÉ (FIN PLATE) — VERSION 1.0
# ===========================
#  interface.py (Streamlit)
#
#  Même écran que le module doubles cornières (finalisation du 21/09/2026) :
#  barre d'outils, statut une ligne, alertes courtes, puis DEUX COLONNES —
#  à gauche le dessin (élévation puis vue en plan, colonne FIGÉE par CSS
#  sticky), à droite la carte compacte par objet, les replis et les onglets.
#
#  SOURCE UNIQUE : les 67 entrées vivent dans st.session_state sous les clés
#  `fpl_<clé>` (espace disjoint des clés `asm_` du module doubles cornières).
#  Le dessin est rendu AVANT la carte (messages appliqués avant widgets).
# ===========================
import json
from datetime import datetime
from functools import lru_cache

import streamlit as st

from . import ecran_saisie, ecran_resultats, texte
from .entrees import CLES, defaults, charger_json, VISSERIE_DEFAUT
from .moteur import compute
from .benchmark import run_bench

MODULE_VERSION = "1.0"
PREFIXE = "fpl_"
_TRANSITOIRES = ("btn", "uploader", "pdf_bytes", "_fpl_", "fpl_cmp_", "fpl_cote_", "fpl_fix_")

_CSS_STICKY = """<style>
@media (min-width: 641px) {
  div[data-testid="stColumn"]:has(div.st-key-fpl_col_dessin) {
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
    st.session_state.setdefault("fpl_visserie", VISSERIE_DEFAUT)
    st.session_state.setdefault("fpl_ui_niveau", 1)
    st.session_state.setdefault("fpl_ui_alerte", None)
    st.session_state.setdefault("fpl_ui_onglet", ecran_resultats.ONGLETS[0])
    st.session_state.setdefault("fpl_ui_txt_complet", False)
    st.session_state.setdefault("fpl_ui_composant", True)
    st.session_state.setdefault("fpl_ui_dernier_clic", {})


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
    st.session_state["_fpl_toast"] = "Valeurs par défaut rétablies"


def _payload(u):
    return {"version": "assemblage-plat-ame-1.0", "assemblage": "plat_ame",
            "values": u,
            "visserie": st.session_state.get("fpl_visserie", VISSERIE_DEFAUT)}


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
            "visserie": st.session_state.get("fpl_visserie", VISSERIE_DEFAUT)}


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
    msg = st.session_state.pop("_fpl_toast", None)
    if msg:
        st.toast(msg)

    tH1, tH2, tH3 = st.columns([8, 1.6, 0.55], vertical_alignment="center")
    with tH1:
        st.markdown("## Assemblage poutre–poutre – plat d'âme soudé (fin plate)")
    with tH2:
        st.markdown(f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {MODULE_VERSION}</div>",
                    unsafe_allow_html=True)
    with tH3:
        st.button("❔", key="fpl_btn_version", use_container_width=True,
                  help="Vérification ELU selon EN 1993-1-8 et EN 1993-1-1 (γM1 : ANB belge) ; modèles "
                       "MSB Part 5 §3 (fin plate), SCI P358, ECCS n°126. Benchmark : Worked Example "
                       "publié du guide, rejoué à chaque ouverture.")

    b0, b1, b2, b3, b4, b5 = st.columns(6)
    with b0:
        if st.button("◀ Assemblages", use_container_width=True, key="fpl_btn_retour"):
            st.session_state.pop("asm_courant", None)
            st.rerun()
    with b1:
        if st.button("🏠 Accueil", use_container_width=True, key="fpl_btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()
    with b2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="fpl_btn_reset"):
            _reinitialiser()
            st.rerun()
    u = lire_entrees()
    with b3:
        st.download_button("💾 Enregistrer", data=json.dumps(_payload(u), indent=1, ensure_ascii=False).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".json"), mime="application/json",
                           use_container_width=True, key="fpl_btn_save_dl")
    with b4:
        if st.button("📂 Ouvrir", use_container_width=True, key="fpl_btn_open_toggle"):
            st.session_state["fpl_show_open_uploader"] = not st.session_state.get("fpl_show_open_uploader", False)
        if st.session_state.get("fpl_show_open_uploader", False):
            up = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed",
                                  key="fpl_open_uploader")
            if up is not None:
                try:
                    data = json.loads(up.read().decode("utf-8-sig"))
                    ecrire_entrees(charger_payload(data))
                    if isinstance(data.get("visserie"), str) and data["visserie"].strip():
                        st.session_state["fpl_visserie"] = data["visserie"].strip()
                    st.session_state["fpl_show_open_uploader"] = False
                    st.session_state["fpl_ui_alerte"] = None
                    st.session_state["_fpl_toast"] = "Données chargées"
                    st.rerun()
                except Exception:
                    st.error("Fichier invalide ou corrompu — chargement annulé.")

    try:
        R = compute(u)
        erreur = None
    except Exception as e:  # noqa: BLE001 — l'écran doit rester utilisable
        R, erreur = None, f"{type(e).__name__}: {e}"

    with b5:
        if st.button("📄 Générer PDF", use_container_width=True, key="fpl_btn_pdf", disabled=R is None,
                     help="Note de calcul de 2 pages (A4 paysage) : synthèse + plan de principe."):
            from .note import generer_note
            try:
                st.session_state["fpl_pdf_bytes"] = generer_note(R, _infos(u))
                st.success("✅ Note de calcul générée")
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("fpl_pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")
        if st.session_state.get("fpl_pdf_bytes"):
            st.download_button("⬇️ Télécharger la note PDF", data=st.session_state["fpl_pdf_bytes"],
                               file_name=texte.nom_fichier(u, ".pdf"), mime="application/pdf",
                               use_container_width=True, key="fpl_btn_pdf_dl")

    if R is None:
        st.error(f"Données incomplètes : {erreur}")
        ecran_saisie.carte(u, set())
        return

    ecran_resultats.bandeau(R)
    ecran_resultats.alertes(R, u)
    chauds = set()
    sel = st.session_state.get("fpl_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            chauds = set(a.fields)
    st.markdown(_CSS_STICKY, unsafe_allow_html=True)
    c_dessin, c_droite = st.columns([1, 1.12], gap="medium")
    with c_dessin:
        with st.container(key="fpl_col_dessin"):
            e, p, hl = ecran_resultats.dessins(R, u)
    with c_droite:
        if not st.session_state.get("fpl_ui_composant", True):
            ecran_resultats.panneau_cotes(e, p, hl)
        ecran_saisie.carte(u, chauds)
        ecran_resultats.onglets(R, _bench(), u, ecrire_entrees)

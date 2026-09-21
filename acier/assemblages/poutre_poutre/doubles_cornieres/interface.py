# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGE POUTRE–POUTRE — DOUBLES CORNIÈRES D'ÂME — VERSION 1.0
# ===========================
#  interface.py (Streamlit)
#
#  Écran de l'assemblage, construit comme les modules béton : même en-tête,
#  même barre d'outils (🏠 Accueil · 🔄 Réinitialiser · 💾 Enregistrer ·
#  📂 Ouvrir · 📄 Générer PDF), même disposition saisie à gauche / résultats
#  à droite (st.columns([2, 3])), mêmes encadrés.
#
#  SOURCE UNIQUE : les 84 entrées vivent dans st.session_state sous les clés
#  `asm_<clé>` ; le formulaire, le dessin cliquable, le panneau des cotes,
#  les chips d'alerte, l'export texte et la note PDF lisent et écrivent
#  ces clés — jamais une copie.
#
#  Ordre d'exécution : la colonne de droite est rendue AVANT la colonne de
#  saisie, pour qu'un clic sur une cote du dessin (composant) soit appliqué
#  à l'état avant l'instanciation des widgets du formulaire.
# ===========================
import json
from datetime import datetime
from functools import lru_cache

import streamlit as st

from acier.js import js_str
from . import ecran_saisie, ecran_resultats, texte
from .entrees import CLES, defaults, charger_json
from .moteur import compute
from .benchmark import run_bench

MODULE_VERSION = "1.0"
PREFIXE = "asm_"
_TRANSITOIRES = ("btn", "uploader", "pdf_bytes", "_asm_", "asm_cmp_", "asm_cote_", "asm_fix_")


def K(k):
    return PREFIXE + k


@lru_cache(maxsize=1)
def _bench():
    """Le benchmark ne dépend pas des saisies : calculé une fois."""
    return run_bench()


# ------------------------------------------------------------------- état
def _transitoire(k):
    return any(m in k for m in _TRANSITOIRES)


def _init_etat():
    for k, v in defaults().items():
        st.session_state.setdefault(K(k), ecran_saisie.valeur_widget(k, v))
    st.session_state.setdefault("asm_ui_niveau", 1)
    st.session_state.setdefault("asm_ui_alerte", None)
    st.session_state.setdefault("asm_ui_onglet", ecran_resultats.ONGLETS[0])
    st.session_state.setdefault("asm_ui_txt_complet", False)
    st.session_state.setdefault("asm_ui_composant", True)
    st.session_state.setdefault("asm_ui_dernier_clic", {})


def _epingler():
    """FIX PERSISTANCE (voir poutre.py / dalle.py) : ré-affecter chaque clé
    persistante pour que Streamlit ne nettoie pas l'état des widgets non
    rendus (champs conditionnels)."""
    for k in list(st.session_state.keys()):
        if k.startswith(PREFIXE) and not _transitoire(k):
            st.session_state[k] = st.session_state[k]


def lire_entrees():
    """Les 84 saisies courantes."""
    return {k: st.session_state.get(K(k)) for k in CLES}


def ecrire_entrees(u):
    """Écrit un jeu complet de saisies (chargement, solution, benchmark)."""
    for k in CLES:
        st.session_state[K(k)] = ecran_saisie.valeur_widget(k, u.get(k))


def _reinitialiser():
    for k in list(st.session_state.keys()):
        if k.startswith(PREFIXE) and k != "asm_courant":
            del st.session_state[k]
    st.session_state["_asm_toast"] = "Valeurs par défaut rétablies"


def _payload(u):
    return {"version": "assemblage-doubles-cornieres-1.0", "assemblage": "doubles_cornieres",
            "values": u}


def charger_payload(data):
    """Accepte le format du module (``values``) et le fichier brut de l'outil
    HTML (les 84 clés à la racine)."""
    if not isinstance(data, dict):
        raise ValueError("Structure JSON inattendue")
    src = data.get("values") if isinstance(data.get("values"), dict) else data
    return charger_json(src)


def _pdf_filename(u):
    return texte.nom_fichier(u, ".pdf")


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
    msg = st.session_state.pop("_asm_toast", None)
    if msg:
        st.toast(msg)

    tH1, tH2, tH3 = st.columns([8, 1.6, 0.55], vertical_alignment="center")
    with tH1:
        st.markdown("## Assemblage poutre–poutre – doubles cornières d'âme")
    with tH2:
        st.markdown(f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {MODULE_VERSION}</div>",
                    unsafe_allow_html=True)
    with tH3:
        st.button("❔", key="asm_btn_version_dc", use_container_width=True,
                  help="Vérification ELU selon EN 1993-1-8 et EN 1993-1-1 ; modèles complémentaires "
                       "MSB Part 5, ECCS n°126, SCI P358. Parité prouvée avec l'outil de référence.")

    b0, b1, b2, b3, b4, b5 = st.columns(6)
    with b0:
        if st.button("◀ Assemblages", use_container_width=True, key="asm_btn_retour"):
            st.session_state.pop("asm_courant", None)
            st.rerun()
    with b1:
        if st.button("🏠 Accueil", use_container_width=True, key="asm_btn_home_dc"):
            st.session_state.retour_accueil_demande = True
            st.rerun()
    with b2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="asm_btn_reset"):
            _reinitialiser()
            st.rerun()
    u = lire_entrees()
    with b3:
        st.download_button("💾 Enregistrer", data=json.dumps(_payload(u), indent=1, ensure_ascii=False).encode("utf-8"),
                           file_name=texte.nom_fichier(u, ".json"), mime="application/json",
                           use_container_width=True, key="asm_btn_save_dl")
    with b4:
        if st.button("📂 Ouvrir", use_container_width=True, key="asm_btn_open_toggle"):
            st.session_state["asm_show_open_uploader"] = not st.session_state.get("asm_show_open_uploader", False)
        if st.session_state.get("asm_show_open_uploader", False):
            up = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed",
                                  key="asm_open_uploader")
            if up is not None:
                try:
                    data = json.loads(up.read().decode("utf-8-sig"))
                    ecrire_entrees(charger_payload(data))
                    st.session_state["asm_show_open_uploader"] = False
                    st.session_state["asm_ui_alerte"] = None
                    st.session_state["_asm_toast"] = "Données chargées"
                    st.rerun()
                except Exception:
                    st.error("Fichier invalide ou corrompu — chargement annulé.")

    try:
        R = compute(u)
        erreur = None
    except Exception as e:  # noqa: BLE001 — l'écran doit rester utilisable
        R, erreur = None, f"{type(e).__name__}: {e}"

    with b5:
        if st.button("📄 Générer PDF", use_container_width=True, key="asm_btn_pdf", disabled=R is None):
            from .rapport import generer_pdf
            infos = {"nom_projet": st.session_state.get("nom_projet", "") or u.get("id_projet", ""),
                     "partie": st.session_state.get("partie", "") or u.get("id_rep", ""),
                     "date": st.session_state.get("date", "") or u.get("id_date", "")
                     or datetime.today().strftime("%d/%m/%Y"),
                     "indice": st.session_state.get("indice", "0")}
            try:
                st.session_state["asm_pdf_bytes"] = generer_pdf(R, infos)
                st.success("✅ Note de calcul générée")
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("asm_pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")
        if st.session_state.get("asm_pdf_bytes"):
            st.download_button("⬇️ Télécharger le rapport PDF", data=st.session_state["asm_pdf_bytes"],
                               file_name=_pdf_filename(u), mime="application/pdf",
                               use_container_width=True, key="asm_btn_pdf_dl")

    gauche, droite = st.columns([2, 3])
    with droite:
        cH1, cH2 = st.columns([18, 1.3], vertical_alignment="center")
        with cH1:
            st.markdown("### Résultats")
        with cH2:
            st.button("⚙️", key="asm_btn_toggle_param", help="Paramètres avancés", use_container_width=True,
                      on_click=lambda: st.session_state.__setitem__(
                          "asm_show_param", not st.session_state.get("asm_show_param", False)))
        if st.session_state.get("asm_show_param", False):
            with st.container(border=True):
                st.checkbox("Dessin interactif (cotes cliquables sur le schéma)", key="asm_ui_composant",
                            help="Désactivé : le schéma reste affiché et les cotes se modifient dans le panneau « Cotes ».")
        if R is None:
            st.error(f"Données incomplètes : {erreur}")
        else:
            ecran_resultats.render(R, _bench(), u, ecrire_entrees)
    with gauche:
        st.markdown("### Données")
        chauds = set()
        if R is not None:
            sel = st.session_state.get("asm_ui_alerte")
            for a in R.alerts:
                if a.id == sel:
                    chauds = set(a.fields)
        ecran_saisie.render(u, chauds)

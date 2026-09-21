# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGE POUTRE–POUTRE — DOUBLES CORNIÈRES D'ÂME — VERSION 1.2
# ===========================
#  interface.py (Streamlit)
#
#  Écran de l'assemblage : même en-tête et même barre d'outils que les
#  modules béton (🏠 Accueil · 🔄 Réinitialiser · 💾 Enregistrer · 📂 Ouvrir ·
#  📄 Générer PDF), puis — refonte du 21/09/2026 (docs/assemblages/REFONTE_UX.md) :
#
#    ● statut sur une ligne, taux par élément, alertes courtes ;
#    ● le DESSIN pleine largeur : TOUT se règle dessus — cotes cliquables,
#      poignées + / −, fenêtres de groupe, PANNEAUX DE PIÈCE (un clic sur
#      la poutre, la cornière, un boulon, un cordon ou l'étiquette VEd
#      ouvre tous ses paramètres), et la poutre portée se DÉPLACE à la
#      souris (jeu gh) — v2 du 21/09/2026 ;
#    ● en dessous, deux replis seulement : Paramètres avancés (réduits) et
#      Identification ; la carte de saisie complète ne revient qu'en repli
#      si le dessin interactif est désactivé ;
#    ● les onglets Vérifications · Prédim · Note · Benchmark · Méthode.
#
#  SOURCE UNIQUE : les 84 entrées vivent dans st.session_state sous les clés
#  `asm_<clé>` ; la carte, le dessin cliquable, le panneau des cotes (repli),
#  les chips d'alerte, l'export texte et la note PDF lisent et écrivent ces
#  clés — jamais une copie.
#
#  Ordre d'exécution : le dessin (composant) est rendu AVANT la carte, pour
#  qu'un message du dessin soit appliqué à l'état avant l'instanciation des
#  widgets qui portent les mêmes clés.
# ===========================
import json
from datetime import datetime
from functools import lru_cache

import streamlit as st

from . import ecran_saisie, ecran_resultats, texte
from .entrees import CLES, defaults, charger_json
from .moteur import compute
from .benchmark import run_bench

MODULE_VERSION = "1.2"
PREFIXE = "asm_"
_TRANSITOIRES = ("btn", "uploader", "pdf_bytes", "pdf_detail_bytes", "_asm_", "asm_cmp_", "asm_cote_", "asm_fix_")


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
    rendus (champs conditionnels, géométrie portée par le dessin)."""
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
    return {"version": "assemblage-doubles-cornieres-1.1", "assemblage": "doubles_cornieres",
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


def _infos(u):
    return {"nom_projet": st.session_state.get("nom_projet", "") or u.get("id_projet", ""),
            "partie": st.session_state.get("partie", "") or u.get("id_rep", ""),
            "date": st.session_state.get("date", "") or u.get("id_date", "")
            or datetime.today().strftime("%d/%m/%Y"),
            "indice": st.session_state.get("indice", "0")}


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
        if st.button("📄 Générer PDF", use_container_width=True, key="asm_btn_pdf", disabled=R is None,
                     help="Note de calcul d'une page (A4 paysage). Rapport détaillé : onglet Note."):
            from .note import generer_note
            try:
                st.session_state["asm_pdf_bytes"] = generer_note(R, _infos(u))
                st.success("✅ Note de calcul générée")
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("asm_pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")
        if st.session_state.get("asm_pdf_bytes"):
            st.download_button("⬇️ Télécharger la note PDF", data=st.session_state["asm_pdf_bytes"],
                               file_name=_pdf_filename(u), mime="application/pdf",
                               use_container_width=True, key="asm_btn_pdf_dl")

    if R is None:
        st.error(f"Données incomplètes : {erreur}")
        ecran_saisie.carte(u, set())
        return

    def generer_detaille():
        from .rapport import generer_pdf
        try:
            st.session_state["asm_pdf_detail_bytes"] = generer_pdf(R, _infos(u))
            st.success("✅ Rapport détaillé généré")
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("asm_pdf_detail_bytes", None)
            st.error(f"Erreur lors de la génération du PDF : {e}")

    ecran_resultats.bandeau(R)
    ecran_resultats.alertes(R, u)
    chauds = set()
    sel = st.session_state.get("asm_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            chauds = set(a.fields)
    if st.session_state.get("asm_ui_composant", True):
        # tout se règle sur le dessin : pleine largeur, puis les deux replis
        ecran_resultats.dessins(R, u)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            ecran_saisie.avances(u, chauds)
        with c2:
            ecran_saisie.identification(u, chauds)
    else:
        # repli sans composant : dessin statique + carte de saisie complète
        c_dessin, c_carte = st.columns([1.55, 1], gap="medium")
        with c_dessin:
            ecran_resultats.dessins(R, u)
        with c_carte:
            ecran_saisie.carte(u, chauds)
    ecran_resultats.onglets(R, _bench(), u, ecrire_entrees, generer_detaille)

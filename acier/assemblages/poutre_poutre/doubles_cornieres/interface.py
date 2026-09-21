# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGE POUTRE–POUTRE — DOUBLES CORNIÈRES D'ÂME — VERSION 1.3
# ===========================
#  interface.py (Streamlit)
#
#  Écran de l'assemblage : même en-tête et même barre d'outils que les
#  modules béton (🏠 Accueil · 🔄 Réinitialiser · 💾 Enregistrer · 📂 Ouvrir ·
#  📄 Générer PDF), puis — finalisation du 21/09/2026
#  (docs/assemblages/REFONTE_UX.md §12) :
#
#    ● statut sur une ligne, taux par élément, alertes courtes ;
#    ● DEUX COLONNES : à gauche le DESSIN (élévation puis vue en plan,
#      empilées), colonne FIGÉE à l'écran (CSS sticky — le schéma reste
#      visible pendant le défilement de la droite) ; TOUT se règle dessus —
#      cotes cliquables, poignées + / −, fenêtres de groupe, panneaux de
#      pièce, et la poutre portée se DÉPLACE à la souris (gh, Δz) ;
#    ● à droite : la carte compacte par objet (mode, profilés, cornières,
#      boulons, efforts), les replis Paramètres avancés et Identification,
#      puis les onglets Vérifications · Prédim · Note · Benchmark · Méthode ;
#    ● dessin interactif désactivé : mêmes colonnes, dessins statiques à
#      gauche, panneau des cotes en tête de la colonne de droite.
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
from .entrees import CLES, defaults, charger_json, VISSERIE_DEFAUT
from .moteur import compute
from .benchmark import run_bench

MODULE_VERSION = "1.3"
PREFIXE = "asm_"
_TRANSITOIRES = ("btn", "uploader", "pdf_bytes", "pdf_detail_bytes", "_asm_", "asm_cmp_", "asm_cote_", "asm_fix_")

# La colonne du dessin est FIGÉE (position: sticky) : le schéma reste visible
# pendant le défilement des paramètres et des vérifications. La colonne est
# reconnue par son CONTENU (:has, conteneur clé asm_col_dessin), jamais par sa
# position dans le DOM : les colonnes imbriquées de la droite ne sont pas
# touchées. Sous 641 px (téléphone), Streamlit empile les colonnes : le sticky
# est retiré pour que le dessin ne se peigne pas par-dessus la carte.
_CSS_STICKY = """<style>
@media (min-width: 641px) {
  div[data-testid="stColumn"]:has(div.st-key-asm_col_dessin) {
    position: sticky;
    top: 2.875rem;
    align-self: flex-start;
    max-height: calc(100vh - 3.5rem);
    overflow-y: auto;
    scrollbar-width: thin;
  }
  /* le padding bas par défaut (10rem) fait « garer » la colonne figée sous
     le haut de l'écran en fin de page : on le ramène à une respiration */
  div[data-testid="stMainBlockContainer"] {
    padding-bottom: 1.5rem;
  }
}
</style>"""


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
    st.session_state.setdefault("asm_visserie", VISSERIE_DEFAUT)
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
            "values": u,
            # annotation de fabrication (hors moteur) : visserie du cartouche
            "visserie": st.session_state.get("asm_visserie", VISSERIE_DEFAUT)}


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
            "indice": st.session_state.get("indice", "0"),
            "visserie": st.session_state.get("asm_visserie", VISSERIE_DEFAUT)}


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
                    if isinstance(data.get("visserie"), str) and data["visserie"].strip():
                        st.session_state["asm_visserie"] = data["visserie"].strip()
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
    # Deux colonnes : le DESSIN à gauche (élévation puis vue en plan, colonne
    # figée à l'écran), les paramètres, vérifications et commandes à droite.
    st.markdown(_CSS_STICKY, unsafe_allow_html=True)
    c_dessin, c_droite = st.columns([1, 1.12], gap="medium")
    with c_dessin:
        with st.container(key="asm_col_dessin"):
            e, p, hl = ecran_resultats.dessins(R, u)
    with c_droite:
        if not st.session_state.get("asm_ui_composant", True):
            ecran_resultats.panneau_cotes(e, p, hl)
        ecran_saisie.carte(u, chauds)
        ecran_resultats.onglets(R, _bench(), u, ecrire_entrees, generer_detaille)

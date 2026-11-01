# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGE POUTRE–POUTRE — DOUBLES CORNIÈRES D'ÂME — VERSION 1.3
# ===========================
#  interface.py (Streamlit)
#
#  Écran de l'assemblage : titre court et barre d'outils partagée
#  (acier/projet_ui.py : ◀ Projet · 🏠 Accueil · 💾 Enregistrer · 📄 PDF ·
#  [A1 ▾] ou 📂 Ouvrir · 🔄 Réinitialiser), puis — finalisation du 21/09/2026
#  (docs/assemblages/REFONTE_UX.md §12), charte ergonomique du 31/10/2026
#  (docs/assemblages/ERGONOMIE.md) :
#
#    ● statut sur une ligne, taux par élément, alertes courtes ;
#    ● DEUX COLONNES : à gauche le DESSIN (élévation puis vue en plan,
#      empilées), colonne FIGÉE à l'écran (CSS sticky — le schéma reste
#      visible pendant le défilement de la droite) ; TOUT se règle dessus —
#      cotes cliquables, poignées + / −, fenêtres de groupe, panneaux de
#      pièce, et la poutre portée se DÉPLACE à la souris (gh, Δz) ;
#    ● à droite : la carte par objet rendue par le kit
#      (acier/assemblages/carte.py : en-tête du mode, profilés, cornières,
#      boulons, efforts), les replis Paramètres avancés et Cartouche (note et
#      plan), puis les onglets Vérifications · Prédim · Note · Benchmark ·
#      Méthode ;
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
from functools import lru_cache

import streamlit as st

from acier import projet_ui
from acier.assemblages import carte as kit, ui_commun
from . import ecran_saisie, ecran_resultats, texte
from .entrees import CLES, defaults, charger_json, VISSERIE_DEFAUT
from .moteur import compute
from .benchmark import run_bench

MODULE_VERSION = "1.3"
PREFIXE = "asm_"
TITRE_COURT = "Doubles cornières d'âme"
TITRE_LONG = "Assemblage poutre–poutre – doubles cornières d'âme"
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
    projet_ui.entree_reinitialisee()      # l'entrée du projet suit (repère conservé)
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


def _infos(u):
    return projet_ui.infos_cartouche(PREFIXE, u, VISSERIE_DEFAUT)


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

    # Titre court + l'ancien titre long mot pour mot avec la version (plus de bouton ❔ ni de cellule
    # « Version » : leur texte vit dans Méthode › « Ce que l'outil vérifie »).
    projet_ui.en_tete(TITRE_COURT, TITRE_LONG + " · version " + MODULE_VERSION)
    u = lire_entrees()
    # dimensions vides (profil Personnalisé) : la vérification est suspendue, le dessin garde des dimensions
    # provisoires (kit.provisoire) — plus de taux absurdes ni d'exception (consolidation, points 8 et 19)
    vides = kit.dimensions_vides(ecran_saisie.D, u)
    try:
        R = compute(kit.provisoire(ecran_saisie.D, u) if vides else u)
        erreur = None
    except Exception as e:  # noqa: BLE001 — l'écran doit rester utilisable
        R, erreur = None, f"{type(e).__name__}: {e}"

    def generer_pdf():
        from .note import generer_note
        return generer_note(R, _infos(u))

    # Barre d'outils partagée : ◀ Projet · 🏠 · 💾 (projet entier) · 📄 · [A2 ▾] · 🔄 —
    # hors projet (écran ouvert directement), 💾 et 📂 gardent l'assemblage seul.
    projet_ui.barre_outils(PREFIXE, lambda: _payload(u), lambda ext: texte.nom_fichier(u, ext), charger_payload,
                           ecrire_entrees, _reinitialiser, generer_pdf, pdf_disabled=R is None or bool(vides),
                           pdf_help="Note de calcul de 2 pages (A4 paysage) : synthèse + plan de principe. "
                                    "Rapport détaillé : onglet Note.")
    projet_ui.bandeau_verrou(PREFIXE)
    # Deux colonnes : le DESSIN à gauche (élévation puis vue en plan, colonne
    # figée à l'écran), les paramètres, vérifications et commandes à droite.
    kit.injecter_css(PREFIXE)
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

    if vides:
        ui_commun.bloc_alerte_ligne("Données incomplètes", kit.message_dimensions(ecran_saisie.D, u), True)
    else:
        ecran_resultats.bandeau(R)
        ecran_resultats.alertes(R, u)
    chauds = set()
    sel = st.session_state.get("asm_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            chauds = set(a.fields)
    c_dessin, c_droite = st.columns([1, 1.12], gap="medium")
    with c_dessin:
        with st.container(key="asm_col_dessin"):
            e, p, hl = ecran_resultats.dessins(R, u)
    with c_droite:
        if not st.session_state.get("asm_ui_composant", True):
            ecran_resultats.panneau_cotes(R, u, e, p, hl)
        ecran_saisie.carte(u, chauds)
        if vides:
            st.caption("Vérifications, note et exports : dès que les dimensions sont saisies.")
        else:
            ecran_resultats.onglets(R, _bench(), u, ecrire_entrees, generer_detaille)

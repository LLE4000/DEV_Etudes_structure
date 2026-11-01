# -*- coding: utf-8 -*-
# ===========================
#  PIED DE POTEAU — PLATINE SUR BÉTON (N + Mx + My) — VERSION 1.0
# ===========================
#  interface.py (Streamlit)
#
#  Titre court et barre d'outils partagée (acier/projet_ui.py), statut une
#  ligne, alertes courtes, puis DEUX COLONNES — à gauche le dessin (coupe A-A
#  puis vue en plan avec la répartition, colonne FIGÉE par CSS sticky), à
#  droite la carte par objet rendue par le kit (acier/assemblages/carte.py),
#  les replis et les onglets. Deux modes, une seule source de clés : le
#  calcul manuel (défaut, manuel/ecran.py) et la vérification complète.
#  Charte ergonomique du 31/10/2026 (docs/assemblages/ERGONOMIE.md).
#
#  SOURCE UNIQUE : les entrées vivent dans st.session_state sous les clés
#  `ppb_<clé>` (espace disjoint des clés `asm_`, `fpl_`, `pab_` et `prj_`).
#  Le dessin est rendu AVANT la carte (messages appliqués avant widgets).
# ===========================
from functools import lru_cache

import streamlit as st

from acier import projet_ui
from acier.assemblages import carte as kit, ui_commun
from . import ecran_saisie, ecran_resultats, texte
from .entrees import CLES, defaults, charger_json, VISSERIE_DEFAUT, est_complet
from .calcul import compute                                    # selon le mode : manuel ou complet
from .benchmark import run_bench
from .manuel import ecran as ecran_manuel

MODULE_VERSION = "1.1"
PREFIXE = "ppb_"
TITRE_COURT = "Pied de poteau sur béton"
TITRE_LONG = "Platine sur béton – pied de poteau"
_TRANSITOIRES = ("btn", "uploader", "pdf_bytes", "_ppb_", "ppb_cmp_", "ppb_cote_", "ppb_fix_")

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
    st.session_state.setdefault("ppb_ui_mode_prec", st.session_state.get(K("mode")))


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
    msg = st.session_state.pop("_ppb_toast", None)
    if msg:
        st.toast(msg)

    # Titre court + l'ancien titre long et la version en caption ; plus de bouton ❔ (son texte : Méthode
    # « Ce que l'outil vérifie ») ni de cellule « Version » (charte §8).
    projet_ui.en_tete(TITRE_COURT, TITRE_LONG + " · version " + MODULE_VERSION)
    u = lire_entrees()
    # dimensions vides (profil Personnalisé, tube) : la vérification est suspendue, le dessin garde des
    # dimensions provisoires (kit.provisoire) — plus de ZeroDivisionError (consolidation, point 19)
    D = ecran_saisie.description(u)
    vides = kit.dimensions_vides(D, u)
    try:
        R = compute(kit.provisoire(D, u) if vides else u)
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
                           pdf_help="Note de calcul de 2 pages (A4 paysage) : synthèse + plan de principe.")
    projet_ui.bandeau_verrou(PREFIXE)
    manuel = not est_complet(u)
    # Le CSS du kit (colonne figée comprise) en UN élément, dans le même conteneur sans écart que le
    # bandeau : chaque élément vide coûtait 16 px d'écart, et au téléphone le dessin du calcul manuel tombait
    # sous le premier écran (945 px > 844, mesuré le 31/10/2026).
    with st.container(gap=None):
        kit.injecter_css(PREFIXE)
        if R is None:
            st.error(f"Données incomplètes : {erreur}")
        elif vides:
            ui_commun.bloc_alerte_ligne("Données incomplètes", kit.message_dimensions(D, u), True)
        elif manuel:
            ecran_manuel.bandeau(R)
        else:
            ecran_resultats.bandeau(R)
    if R is None:
        _carte(u, set(), manuel)
        return
    if not vides:
        # la ligne d'information hors verdict, sous le bandeau (hors du conteneur sans écart : elle le chevauchait)
        (ecran_manuel.information if manuel else ecran_resultats.information)(R)
        ecran_resultats.alertes(R, u)
    chauds = set()
    sel = st.session_state.get("ppb_ui_alerte")
    for a in R.alerts:
        if a.id == sel:
            chauds = set(a.fields) | set(a.elems)
    c_dessin, c_droite = st.columns([1, 1.12], gap="medium")
    with c_dessin:
        with st.container(key="ppb_col_dessin"):
            e, p, hl = ecran_manuel.dessins(R, u) if manuel else ecran_resultats.dessins(R, u)
    with c_droite:
        if not st.session_state.get("ppb_ui_composant", True):
            ecran_resultats.panneau_cotes(u, hl)
        _carte(u, chauds, manuel)
        if vides:
            st.caption("Vérifications, note et exports : dès que les dimensions sont saisies.")
        elif manuel:
            ecran_manuel.onglets(R, u, ecrire_entrees)
        else:
            ecran_resultats.onglets(R, _bench(), u, ecrire_entrees)


def _carte(u, chauds, manuel):
    """La carte du mode : calcul manuel (tables) ou vérification complète ;
    les deux s'ouvrent sur la même rangée d'en-tête (le mode et sa bascule)."""
    if manuel:
        ecran_manuel.carte(u, chauds, lire_entrees, ecrire_entrees)
    else:
        ecran_saisie.carte(u, chauds, lire_entrees, ecrire_entrees)

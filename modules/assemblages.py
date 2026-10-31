# -*- coding: utf-8 -*-
# ===========================
#  ASSEMBLAGES MÉTALLIQUES — page PROJET
# ===========================
#  assemblages.py (Streamlit) — v2.0
#
#  Point d'entrée « Acier → Assemblages métalliques ». Depuis la v2.0 la page
#  est un PROJET, comme Poutre et Dalle : informations du projet, LISTE des
#  assemblages créés (repère, type, pastille, taux, dimensionnante),
#  ➕ Nouvel assemblage (catalogue du registre → « Créer »), menu ⋯ par ligne
#  (repère, 📋 dupliquer, ↑ ↓ ordre du rapport, 🗑️ supprimer avec
#  confirmation), 💾 / 📂 sur le projet entier.
#
#  L'écran d'un assemblage reste celui de son type ; acier/projet_ui.ouvrir
#  charge l'entrée dans l'état du type, « ◀ Projet » et le sélecteur [A2 ▾]
#  y ramènent. Ajouter un type d'assemblage = une entrée dans le registre ;
#  cette page ne change pas.
#
#  ESPACE DE NOMS DE SESSION : `prj_*` (projet, acier/projet_ui.py) ;
#  `asm_courant` reste la clé de l'écran ouvert (lue par les trois écrans et
#  les tests). Disjoint des clés `b…` / `dal…` / `pre…` des modules béton.
# ===========================
import json

import streamlit as st

from acier import registre, projet_ui
from acier import projet as modele

ASSEMBLAGES_VERSION = "2.0"

_PASTILLE = {"ok": ("#1f7a3a", "vérifié"), "att": ("#b45309", "sous réserve"),
             "ko": ("#b91c1c", "non vérifié"), "err": ("#6b7280", "calcul impossible")}


def synchroniser_projet():
    """Appelée par streamlit_app.py au début de chaque run (voir la règle de
    synchronisation dans acier/projet_ui.py). Jamais bloquante : un défaut
    ici casserait toutes les pages de l'application."""
    try:
        projet_ui.synchroniser()
    except Exception:  # noqa: BLE001
        pass


def retour_selection():
    """Revenir à la page projet (conservé pour les anciens appelants)."""
    projet_ui.fermer()


# -------------------------------------------------------------- callbacks
def _toast(msg):
    st.session_state["_prj_toast"] = msg


def _creer(type_id):
    e = projet_ui.creer(type_id)
    st.session_state["prj_mode_catalogue"] = False
    if e:
        _toast(f"{modele.repere(e)} créé")


def _ouvrir(ident):
    projet_ui.ouvrir(ident)


def _dupliquer(ident):
    e = projet_ui.dupliquer(ident)
    if e:
        _toast(f"{modele.repere(e)} ajouté")


def _deplacer(ident, delta):
    modele.deplacer(projet_ui.projet(), ident, delta)


def _demander_suppression(ident):
    st.session_state["prj_confirm_suppr"] = ident


def _confirmer_suppression(ident):
    e = modele.trouver(projet_ui.projet(), ident)
    rep = modele.repere(e) if e else ""
    projet_ui.supprimer(ident)
    st.session_state["prj_confirm_suppr"] = None
    _toast(f"{rep} supprimé")


def _annuler_suppression():
    st.session_state["prj_confirm_suppr"] = None


def _renommer(ident, cle):
    modele.renommer(projet_ui.projet(), ident, st.session_state.get(cle, ""))


def _variante(ident):
    e = projet_ui.variante(ident)
    st.session_state["prj_mode_catalogue"] = False
    if e:
        _toast(f"{modele.repere(e)} ouvert")


def _basculer_verrou(ident):
    projet_ui.basculer_verrou(ident)


def _basculer_catalogue():
    st.session_state["prj_mode_catalogue"] = not st.session_state.get("prj_mode_catalogue", False)


def _vider():
    projet_ui.vider()
    st.session_state["prj_confirm_vider"] = False
    _toast("Projet vidé")


def _annuler_vider():
    st.session_state["prj_confirm_vider"] = False


def _demander_vider():
    st.session_state["prj_confirm_vider"] = True


# ------------------------------------------------------------------ rapport
def _empreinte():
    """Le rapport téléchargeable est celui du projet TEL QU'IL EST : un
    changement après la génération retire le bouton (pas de rapport périmé)."""
    return json.dumps(projet_ui.payload(), sort_keys=True, ensure_ascii=False, default=str)


def _bouton_rapport(liste):
    if st.button("📄 Générer PDF", use_container_width=True, key="prj_btn_pdf", disabled=not liste,
                 help="Rapport du projet : page de garde avec sommaire, puis la note de chaque assemblage "
                      "(un signet par assemblage)."):
        from acier.rapport_projet import generer_rapport
        try:
            st.session_state["prj_pdf_bytes"] = generer_rapport(projet_ui.projet(), projet_ui.infos())
            st.session_state["prj_pdf_empreinte"] = _empreinte()
            st.success("✅ Rapport généré")
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("prj_pdf_bytes", None)
            st.error(f"Erreur lors de la génération du PDF : {e}")
    if st.session_state.get("prj_pdf_bytes") and st.session_state.get("prj_pdf_empreinte") == _empreinte():
        st.download_button("⬇️ Télécharger le rapport PDF", data=st.session_state["prj_pdf_bytes"],
                           file_name=projet_ui.nom_fichier(".pdf"), mime="application/pdf",
                           use_container_width=True, key="prj_btn_pdf_dl")


# ------------------------------------------------------------------ page
def _ligne(e, i, n):
    r = projet_ui.resume(e)
    coul, lib = _PASTILLE.get(r.get("etat"), _PASTILLE["err"])
    ident = e["id"]
    rep = modele.repere(e) or "—"
    verrou = modele.est_verrouille(e)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([5, 2.6, 1.3, 0.7], vertical_alignment="center")
        with c1:
            st.markdown(f"{'🔒 ' if verrou else ''}**{rep}** — {projet_ui.titre_de(e['type'])}")
            if r.get("description"):
                st.caption(r["description"])
        with c2:
            eta = "" if r.get("eta") is None else f" · η = {r['eta'] * 100:.0f} %"
            gov = f"<br><span style='color:#6b7280;font-size:0.85em'>{r['gov']}</span>" if r.get("gov") else ""
            st.markdown(f"<span style='color:{coul};font-size:1.15em'>●</span> {lib}{eta}{gov}",
                        unsafe_allow_html=True)
        with c3:
            st.button("Ouvrir", key=f"prj_btn_ouvrir_{ident}", use_container_width=True,
                      on_click=_ouvrir, args=(ident,))
        with c4:
            with st.popover("⋯", use_container_width=True, help="Repère, dupliquer, ordre, supprimer"):
                cle = f"prj_rep_{ident}"
                st.session_state[cle] = modele.repere(e)
                st.text_input("Repère", key=cle, on_change=_renommer, args=(ident, cle), disabled=verrou)
                st.button("📋 Dupliquer", key=f"prj_btn_dup_{ident}", use_container_width=True,
                          on_click=_dupliquer, args=(ident,))
                st.button("🧬 Variante", key=f"prj_btn_var_{ident}", use_container_width=True,
                          on_click=_variante, args=(ident,),
                          help="Copie ouverte aussitôt, pour comparer deux solutions dans le même projet")
                m1, m2 = st.columns(2)
                with m1:
                    st.button("↑ Monter", key=f"prj_btn_up_{ident}", use_container_width=True,
                              disabled=i == 0, on_click=_deplacer, args=(ident, -1))
                with m2:
                    st.button("↓ Descendre", key=f"prj_btn_down_{ident}", use_container_width=True,
                              disabled=i >= n - 1, on_click=_deplacer, args=(ident, 1))
                st.button("🔓 Déverrouiller" if verrou else "🔒 Verrouiller", key=f"prj_btn_lock_{ident}",
                          use_container_width=True, on_click=_basculer_verrou, args=(ident,),
                          help="Verrouillé : les données, le repère et la suppression sont bloqués")
                st.button("🗑️ Supprimer", key=f"prj_btn_suppr_{ident}", use_container_width=True,
                          on_click=_demander_suppression, args=(ident,), disabled=verrou)
        if st.session_state.get("prj_confirm_suppr") == ident:
            w1, w2, w3 = st.columns([4, 1.3, 1.3], vertical_alignment="center")
            with w1:
                st.markdown(f"<span style='color:#b91c1c'>Supprimer **{rep}** ? Cette action est définitive.</span>",
                            unsafe_allow_html=True)
            with w2:
                st.button("Confirmer", key=f"prj_btn_suppr_ok_{ident}", use_container_width=True, type="primary",
                          on_click=_confirmer_suppression, args=(ident,))
            with w3:
                st.button("Annuler", key=f"prj_btn_suppr_non_{ident}", use_container_width=True,
                          on_click=_annuler_suppression)


def _catalogue(liste):
    st.markdown("### Nouvel assemblage" if liste else "### Premier assemblage du projet")
    st.caption("Choisir le type : l'assemblage est créé avec ses valeurs par défaut et s'ouvre aussitôt."
               + ("" if liste else " La liste du projet apparaîtra ici."))
    if liste:
        st.button("Annuler", key="prj_btn_catalogue_fermer", on_click=_basculer_catalogue)
    for fam in registre.FAMILLES:
        st.markdown(f"#### {fam.titre}")
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
                            st.button("Créer", key=f"asm_btn_open_{a.id}", use_container_width=True,
                                      on_click=_creer, args=(a.id,))
                        else:
                            st.markdown("<div style='text-align:center;color:#9ca3af;font-size:0.9em;'>"
                                        "À venir</div>", unsafe_allow_html=True)


def _page_projet():
    if "retour_accueil_demande" not in st.session_state:
        st.session_state.retour_accueil_demande = False
    if st.session_state.retour_accueil_demande:
        st.session_state.page = "Accueil"
        st.session_state.retour_accueil_demande = False
        st.rerun()
    msg = st.session_state.pop("_prj_toast", None)
    if msg:
        st.toast(msg)
    projet = projet_ui.projet()
    liste = modele.entrees(projet)

    tH1, tH2, tH3 = st.columns([8, 1.6, 0.55], vertical_alignment="center")
    with tH1:
        st.markdown("## Assemblages métalliques")
    with tH2:
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {ASSEMBLAGES_VERSION}</div>",
            unsafe_allow_html=True,
        )
    with tH3:
        st.button("❔", key="asm_btn_version_hist", use_container_width=True,
                  help="Projet d'assemblages métalliques — EN 1993-1-8. Un fichier = un projet : la liste "
                       "des assemblages créés, chacun ouvert dans l'écran de son type (données, calcul, "
                       "schémas, note). 💾 et 📂 portent sur le projet entier.")

    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        if st.button("🏠 Accueil", use_container_width=True, key="asm_btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()
    with b2:
        if st.session_state.get("prj_confirm_vider"):
            st.button("⚠️ Confirmer", key="prj_btn_vider_ok", use_container_width=True, type="primary",
                      on_click=_vider, help="Vide la liste des assemblages du projet")
            st.button("Annuler", key="prj_btn_vider_non", use_container_width=True, on_click=_annuler_vider)
        else:
            st.button("🔄 Réinitialiser", use_container_width=True, key="prj_btn_reset", on_click=_demander_vider,
                      help="Vide la liste des assemblages (les informations du projet sont conservées)")
    with b3:
        st.download_button("💾 Enregistrer", data=json.dumps(projet_ui.payload(), indent=1,
                                                             ensure_ascii=False).encode("utf-8"),
                           file_name=projet_ui.nom_fichier(".json"), mime="application/json",
                           use_container_width=True, key="prj_btn_save_dl",
                           help="Enregistre le projet entier : informations et tous les assemblages")
    with b4:
        if st.button("📂 Ouvrir", use_container_width=True, key="prj_btn_open_toggle",
                     help="Ouvre un fichier projet, ou ajoute au projet un fichier d'assemblage seul"):
            st.session_state["prj_show_open_uploader"] = not st.session_state.get("prj_show_open_uploader", False)
        if st.session_state.get("prj_show_open_uploader", False):
            up = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed",
                                  key="prj_open_uploader")
            if up is not None:
                try:
                    data = json.loads(up.read().decode("utf-8-sig"))
                    message = projet_ui.charger_fichier(data)
                    st.session_state["prj_show_open_uploader"] = False
                    _toast(message)
                    st.rerun()
                except Exception:  # noqa: BLE001
                    st.error("Fichier invalide ou corrompu — chargement annulé.")
    with b5:
        _bouton_rapport(liste)

    projet_ui.bloc_infos_projet()

    liste = modele.entrees(projet_ui.projet())
    n = len(liste)
    nb_ko = sum(1 for e in liste if projet_ui.resume(e).get("etat") in ("ko", "err"))
    st.markdown(f"### Assemblages ({n})" + (f" · {nb_ko} non vérifié{'s' if nb_ko > 1 else ''}" if nb_ko else ""))
    for i, e in enumerate(liste):
        _ligne(e, i, n)
    if liste:
        cN, _ = st.columns([2, 3])
        with cN:
            st.button("➕ Nouvel assemblage", key="prj_btn_nouveau", use_container_width=True,
                      on_click=_basculer_catalogue, type="primary")
    if not liste or st.session_state.get("prj_mode_catalogue", False):
        _catalogue(liste)


def show():
    projet_ui.synchroniser()
    courant = st.session_state.get("asm_courant")
    if courant:
        trouve = registre.trouver(courant)
        if trouve and trouve[1].disponible:
            registre.charger(trouve[1]).show()
            return
        st.session_state.pop("asm_courant", None)
    _page_projet()

# -*- coding: utf-8 -*-
"""Le projet d'assemblages dans la session Streamlit : ouverture d'une entrée
dans l'écran de son type, synchronisation, morceaux de barre d'outils
partagés par les trois écrans d'assemblage.

CLÉS DE SESSION : ``prj_*`` — disjointes de ``asm_`` (doubles cornières),
``fpl_``, ``pab_`` et des clés béton. Le bouton 🔄 des doubles cornières
efface tout ``asm_*`` : le projet n'y est pas, il survit.

RÈGLE DE SYNCHRONISATION. L'écran d'un type garde sa source unique (ses
clés ``<préfixe><clé>``) ; l'entrée du projet en est une VUE, recopiée au
début de CHAQUE run (``streamlit_app.py`` appelle ``synchroniser()``) tant
que l'état du type est VIVANT, c'est-à-dire tant que ses clés existent :
Streamlit les efface en fin de run quand les widgets ne sont pas rendus
(page Accueil), et ``_init_etat`` les recrée ensuite avec les défauts — une
entrée ne doit jamais être écrasée par ces défauts. La recopie en début de
run attrape aussi la valeur saisie dans le même geste que le clic 🏠 : l'état
est déjà à jour quand le script démarre, avant que la page Accueil ne
l'efface.

PIÈGES VERROUILLÉS (tests/test_projet_assemblages.py) :
  1. le projet n'est pas sous ``asm_`` — 🔄 des doubles cornières le garde ;
  2. deux entrées d'un même type partagent un écran : à l'ouverture, la note
     PDF déjà générée et l'alerte sélectionnée sont oubliées (sinon la note
     de A1 partirait sous le nom de A2) ; le dernier clic du dessin est
     CONSERVÉ (le composant garde son dernier message : l'oublier le ferait
     rejouer sur la nouvelle entrée).
"""
import json
from datetime import datetime

import streamlit as st

from acier import registre
from acier import projet as modele

CLE_PROJET = "prj_projet"
CLE_OUVERT = "prj_ouvert"
CLES_INFOS = modele.CLES_INFOS


# ------------------------------------------------------------------- état
def projet():
    p = st.session_state.get(CLE_PROJET)
    if not isinstance(p, dict):
        p = modele.nouveau()
        st.session_state[CLE_PROJET] = p
    return p


def entree_ouverte():
    ident = st.session_state.get(CLE_OUVERT)
    return modele.trouver(projet(), ident) if ident is not None else None


def module_de(type_id):
    """Le module d'interface du type (None si inconnu ou « à venir »)."""
    t = registre.trouver(type_id)
    return registre.charger(t[1]) if t and t[1].disponible else None


def titre_de(type_id):
    t = registre.trouver(type_id)
    return t[1].titre if t else str(type_id)


def _vivant(m):
    return all(m.K(k) in st.session_state for k in m.CLES)


def infos():
    return {"nom_projet": str(st.session_state.get("nom_projet", "") or ""),
            "partie": str(st.session_state.get("partie", "") or ""),
            "date": str(st.session_state.get("date", "") or datetime.today().strftime("%d/%m/%Y")),
            "indice": str(st.session_state.get("indice", "0") or "0")}


def epingler_infos():
    """Les informations du projet sont des widgets de la page projet ; elles
    doivent survivre aux écrans qui ne les rendent pas (même fix que Poutre)."""
    for k in CLES_INFOS:
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]


def _hydrater(m, e):
    """L'entrée → l'état du type (avant tout widget)."""
    m.ecrire_entrees(e["values"])
    st.session_state[m.PREFIXE + "visserie"] = e.get("visserie") or m.VISSERIE_DEFAUT


def synchroniser():
    """Recopie l'état VIVANT de l'écran ouvert dans son entrée (True si
    recopié). Si l'état n'est plus vivant — effacé par Streamlit après une
    page qui ne rendait pas l'écran, ou par 🔄 — et que la page des
    assemblages va se rendre, c'est l'ENTRÉE qui ré-hydrate l'écran : sans
    cela, ``_init_etat`` rendrait des valeurs par défaut que la recopie
    suivante écrirait dans l'entrée."""
    epingler_infos()
    e = entree_ouverte()
    if e is None:
        return False
    m = module_de(e["type"])
    if m is None:
        return False
    if not _vivant(m):
        if st.session_state.get("page") == "Assemblages métalliques":
            _hydrater(m, e)
        return False
    u = m.lire_entrees()
    if not str(u.get("id_rep") or "").strip():
        u["id_rep"] = modele.repere(e)          # un repère ne se vide pas
    e["values"] = u
    e["visserie"] = str(st.session_state.get(m.PREFIXE + "visserie", e.get("visserie", "")) or "")
    return True


def entree_reinitialisee():
    """Appelée par le 🔄 d'un écran après l'effacement de ses clés : l'entrée
    ouverte reprend les valeurs par défaut du type — repère conservé — et
    c'est elle qui ré-hydratera l'écran au run suivant."""
    e = entree_ouverte()
    if e is None:
        return False
    m = module_de(e["type"])
    if m is None:
        return False
    rep = modele.repere(e)
    e["values"] = {k: m.ecran_saisie.valeur_widget(k, v) for k, v in m.defaults().items()}
    e["values"]["id_rep"] = rep
    e["visserie"] = m.VISSERIE_DEFAUT
    return True


def ouvrir(ident):
    """Charge l'entrée dans l'état de son type et en fait l'écran courant.
    À appeler depuis un callback (avant l'instanciation des widgets)."""
    synchroniser()
    e = modele.trouver(projet(), ident)
    if e is None:
        return False
    m = module_de(e["type"])
    if m is None:
        return False
    m.ecrire_entrees(e["values"])
    p = m.PREFIXE
    st.session_state[p + "visserie"] = e.get("visserie") or m.VISSERIE_DEFAUT
    for k in ("pdf_bytes", "pdf_detail_bytes", "show_open_uploader"):
        st.session_state.pop(p + k, None)
    st.session_state[p + "ui_alerte"] = None
    st.session_state["asm_courant"] = e["type"]
    st.session_state[CLE_OUVERT] = e["id"]
    return True


def fermer():
    synchroniser()
    st.session_state.pop("asm_courant", None)
    st.session_state[CLE_OUVERT] = None


def creer(type_id):
    """Nouvelle entrée aux valeurs par défaut du type, repère libre, ouverte."""
    m = module_de(type_id)
    if m is None:
        return None
    values = {k: m.ecran_saisie.valeur_widget(k, v) for k, v in m.defaults().items()}
    e = modele.ajouter(projet(), type_id, values, m.VISSERIE_DEFAUT)
    ouvrir(e["id"])
    return e


def dupliquer(ident):
    synchroniser()
    return modele.dupliquer(projet(), ident)


def supprimer(ident):
    if st.session_state.get(CLE_OUVERT) == ident:
        st.session_state[CLE_OUVERT] = None
        st.session_state.pop("asm_courant", None)
    return modele.supprimer(projet(), ident)


def vider():
    st.session_state[CLE_PROJET] = modele.nouveau()
    st.session_state[CLE_OUVERT] = None
    st.session_state.pop("asm_courant", None)


def nom_fichier(ext):
    return modele.nom_fichier(infos(), ext)


def payload():
    synchroniser()
    return modele.payload(projet(), infos())


# ---------------------------------------------------------------- fichiers
def _normaliser(type_id, src):
    m = module_de(type_id)
    if m is None:
        raise ValueError(f"Type d'assemblage inconnu : {type_id}")
    return m.charger_payload(src)


def _detecter_type(src):
    """Fichier brut (clés à la racine, sans champ « assemblage ») : le type
    dont les clés sont le mieux représentées, s'il l'est à moitié au moins."""
    meilleur, score = None, 0.0
    cles_src = set(src.keys())
    for a in registre.disponibles():
        m = module_de(a.id)
        cles = set(m.CLES)
        s = len(cles & cles_src) / max(1, len(cles))
        if s > score:
            meilleur, score = a.id, s
    return meilleur if score >= 0.5 else None


def charger_fichier(data):
    """Fichier projet → remplace le projet et ses informations ; fichier
    d'assemblage seul → s'ajoute au projet. Retourne le message à afficher."""
    infos_, lot = modele.lire_fichier(data, _normaliser, _detecter_type)
    if infos_ is not None:
        p = modele.nouveau()
        modele.importer(p, lot)
        st.session_state[CLE_PROJET] = p
        st.session_state[CLE_OUVERT] = None
        st.session_state.pop("asm_courant", None)
        for k in CLES_INFOS:
            v = infos_.get(k, "")
            st.session_state[k] = v if (v or k != "indice") else "0"
        if any(infos_.get(k) for k in ("nom_projet", "partie")):
            st.session_state["chk_infos_projet"] = True
        return f"Projet chargé : {len(lot)} assemblage(s)"
    modele.importer(projet(), lot)
    return "Assemblage ajouté au projet"


# ----------------------------------------------------------------- résumé
@st.cache_data(show_spinner=False, max_entries=512)
def _resume_calcule(type_id, values_json):
    m = module_de(type_id)
    u = json.loads(values_json)
    u = {k: m.ecran_saisie.valeur_widget(k, u.get(k)) for k in m.CLES}
    try:
        R = m.compute(u)
    except Exception as exc:  # noqa: BLE001 — la liste doit rester lisible
        return {"etat": "err", "eta": None, "gov": type(exc).__name__, "statut": "calcul impossible"}
    etat = "ok" if R.verified and not (R.reserve or R.alerts) else ("att" if R.verified else "ko")
    return {"etat": etat, "eta": float(R.eta_max or 0.0), "gov": R.gov.label if R.gov else "",
            "statut": R.statut}


def resume(e):
    """État de l'entrée pour la liste : pastille, taux, dimensionnante,
    description courte. Calcul mis en cache sur les saisies."""
    try:
        r = dict(_resume_calcule(e["type"], json.dumps(e.get("values", {}), sort_keys=True, default=str)))
    except Exception as exc:  # noqa: BLE001
        r = {"etat": "err", "eta": None, "gov": type(exc).__name__, "statut": "calcul impossible"}
    r["description"] = modele.description(e.get("values"))
    return r


# ------------------------------------------------- barre d'outils des écrans
def bouton_retour(cle):
    if st.button("◀ Projet", use_container_width=True, key=cle,
                 help="Retour à la liste des assemblages du projet"):
        fermer()
        st.rerun()


def libelle_entree(e):
    return f"{modele.repere(e) or '—'} · {titre_de(e['type'])}"


def _libelles(projet_):
    """Un libellé par entrée, unique (un doublon de repère reçoit « (#id) ») ;
    les options du sélecteur sont ces libellés en clair — pas de format_func,
    que le harnais AppTest ne sait pas relire."""
    par_libelle = {}
    for x in modele.entrees(projet_):
        lab = libelle_entree(x)
        if lab in par_libelle:
            lab = f"{lab} (#{x['id']})"
        par_libelle[lab] = x["id"]
    return par_libelle


def _sur_selection(cle):
    ident = st.session_state.get(cle + "_map", {}).get(st.session_state.get(cle))
    if ident is not None and ident != st.session_state.get(CLE_OUVERT):
        ouvrir(ident)


def selecteur(prefixe):
    """Dans la barre d'un écran : passer à un autre assemblage du projet sans
    repasser par la liste. False (rien rendu) si l'écran est ouvert hors projet."""
    e = entree_ouverte()
    if e is None:
        return False
    par_libelle = _libelles(projet())
    cle = prefixe + "sel_projet"
    st.session_state[cle] = next(lab for lab, i in par_libelle.items() if i == e["id"])
    st.session_state[cle + "_map"] = par_libelle
    st.selectbox("Assemblage du projet", list(par_libelle), key=cle, label_visibility="collapsed",
                 on_change=_sur_selection, args=(cle,), help="Passer à un autre assemblage du projet")
    return True


def bouton_enregistrer(cle, payload_seul, nom_seul):
    """💾 : le projet entier quand une entrée est ouverte, sinon (écran ouvert
    hors projet) l'assemblage seul, au format que le module a toujours produit."""
    if entree_ouverte() is not None:
        data, nom, aide = payload(), nom_fichier(".json"), "Enregistre tout le projet (fichier JSON)"
    else:
        data, nom, aide = payload_seul(), nom_seul(), "Enregistre cet assemblage seul (fichier JSON)"
    st.download_button("💾 Enregistrer", data=json.dumps(data, indent=1, ensure_ascii=False).encode("utf-8"),
                       file_name=nom, mime="application/json", use_container_width=True, key=cle, help=aide)


def bouton_ouvrir_seul(prefixe, charger_payload, ecrire_entrees):
    """📂 d'un écran ouvert hors projet : charge un fichier d'assemblage seul
    dans l'écran (comportement d'avant le projet, mêmes clés)."""
    p = prefixe
    if st.button("📂 Ouvrir", use_container_width=True, key=p + "btn_open_toggle"):
        st.session_state[p + "show_open_uploader"] = not st.session_state.get(p + "show_open_uploader", False)
    if st.session_state.get(p + "show_open_uploader", False):
        up = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed",
                              key=p + "open_uploader")
        if up is not None:
            try:
                data = json.loads(up.read().decode("utf-8-sig"))
                ecrire_entrees(charger_payload(data))
                if isinstance(data.get("visserie"), str) and data["visserie"].strip():
                    st.session_state[p + "visserie"] = data["visserie"].strip()
                st.session_state[p + "show_open_uploader"] = False
                st.session_state[p + "ui_alerte"] = None
                st.session_state["_" + p + "toast"] = "Données chargées"
                st.rerun()
            except Exception:  # noqa: BLE001
                st.error("Fichier invalide ou corrompu — chargement annulé.")


# ------------------------------------------------- informations du projet
def _basculer_infos():
    st.session_state["chk_infos_projet"] = not st.session_state.get("chk_infos_projet", False)


def bloc_infos_projet():
    """Même bloc que Poutre et Dalle, mêmes clés (nom_projet, partie, date,
    indice) : les informations suivent l'utilisateur d'un module à l'autre."""
    st.session_state.setdefault("chk_infos_projet", False)
    st.session_state.setdefault("nom_projet", "")
    st.session_state.setdefault("partie", "")
    st.session_state.setdefault("date", datetime.today().strftime("%d/%m/%Y"))
    st.session_state.setdefault("indice", "0")
    affiche = bool(st.session_state.get("chk_infos_projet", False))
    cT, cBtn = st.columns([6, 0.6], vertical_alignment="center")
    with cT:
        st.markdown("### Informations sur le projet")
    with cBtn:
        st.button("➖" if affiche else "➕", key="prj_btn_toggle_infos", use_container_width=True,
                  help="Masquer les informations du projet" if affiche else "Ajouter les informations du projet",
                  on_click=_basculer_infos)
    if affiche:
        with st.container(border=True):
            st.text_input("Nom du projet", placeholder="Nom du projet", key="nom_projet", label_visibility="collapsed")
            st.text_input("Partie", placeholder="Partie", key="partie", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Date", placeholder="Date (jj/mm/aaaa)", key="date", label_visibility="collapsed")
            with c2:
                st.text_input("Indice", placeholder="Indice", key="indice", label_visibility="collapsed")

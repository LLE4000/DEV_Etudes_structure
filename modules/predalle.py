# -*- coding: utf-8 -*-
# ===========================
#  PRÉDALLE EN BÉTON ARMÉ — VERSION 1.0
# ===========================
#  predalle.py (Streamlit)
#
#  COPIE ADAPTÉE du module Dalle v2.1 (dalle.py), dans son propre espace
#  de clés : même écran, mêmes blocs, mêmes colonnes, mêmes boutons,
#  mêmes cartes de vérification, même logique Treillis / Barres /
#  « n barres » / renforts, même note de calcul. AUCUNE formule nouvelle :
#  hᵤ,min, Aₛ,req, Aₛ,min, Aₛ,max, τ ≤ τ_adm,I sont strictement les
#  expressions du module Dalle (donc du module Poutre).
#
#  CE QUI CHANGE — la GÉOMÉTRIE DES COUCHES, rien d'autre :
#   1. Une prédalle = une peau PRÉFABRIQUÉE en partie basse (h_pre,
#      6 cm par défaut) + du béton COULÉ EN PLACE au-dessus. Épaisseur
#      totale h (22 cm par défaut) ; coulé en place = h − h_pre, calculé,
#      jamais saisi. h et h_pre sont modifiables dans les
#      caractéristiques (champ « Prédalle (cm) » ajouté, seul ajout).
#   2. POSITIONS PAR DÉFAUT des armatures (_auto_dist_couche) :
#        - couche 1 INFÉRIEURE de la direction PRINCIPALE : DANS la
#          prédalle préfabriquée — enrobage + demi-Ø arrondi au 0,5 cm
#          sup. + jeu premier lit (règle Dalle inchangée) ;
#        - TOUTE autre couche inférieure (couche 1 de la direction
#          secondaire, et TOUS les renforts inférieurs, posés sur
#          chantier) : AU-DESSUS de la prédalle — h_pre + demi-Ø arrondi
#          au 0,5 cm sup. ;
#        - couches supérieures : règle Dalle inchangée (mesurées depuis
#          la face supérieure, coulée en place).
#      Chaque direction garde donc SA hauteur utile (d principal grand,
#      d secondaire réduit) via le CDG pondéré existant ; la colonne
#      « Dist. axe » saisissable permet toujours d'imposer un niveau.
#   3. NOTE DE CALCUL : la mise en page Dalle à l'identique, obtenue en
#      TRADUISANT l'état « pre{id}_ » vers l'espace « dal{id}_ »
#      qu'attend modules/export_pdf_dalle (_valeurs_pour_export), en
#      FIGEANT la distance d'axe effective de chaque couche — la règle
#      prédalle vit ici, l'export n'en connaît rien. Seule addition :
#      la clé h_pre, qui fait apparaître le trait de clivage
#      prédalle / coulé en place sur la coupe et les lignes
#      « dont prédalle préf. / dont coulé en place » dans DIMENSIONS.
#
#  ESPACE DE NOMS DE SESSION : préfixes "pre{id}_" / "meta_predalle_" /
#  "meta_pre{id}_", volontairement DISJOINTS des clés du module Dalle
#  ("dal{id}_") et du module Poutre ("b{id}_") — les trois modules
#  coexistent dans une même session sans échange d'état (les paramètres
#  globaux gamma_s, jeux, unités et infos projet restent partagés).
# ===========================
import streamlit as st
from datetime import datetime
from string import ascii_uppercase
import json
import math
import re
from copy import deepcopy

from modules import treillis as TR

# ============================================================
#  STYLES BLOCS (identiques à poutre.py)
# ============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌"}

# Données béton (chargées dans show())
BETON_DATA = {}

MAX_COUCHES = 4  # base + 3 renforts par face et par direction

PREDALLE_VERSION = "1.0"  # version affichée dans l'en-tête de l'application

# Directions d'une dalle : clé interne + libellé. Les faces deviennent
# "inf_x" / "sup_x" / "inf_y" / "sup_y" — suffixe opaque pour toute la
# machinerie des couches (aucune formule ne lit la direction).
DIR_KEYS = ("x", "y")
FACES_DIR = ("inf_x", "sup_x", "inf_y", "sup_y")

# Largeurs de colonnes du tableau des couches
# (Couche | Type | Treillis/Ø | Esp. | As/m | Dist. axe | CDG | Action)
# Sans colonne As (mm²/m) : redondante avec « Aₛ fourni » du bandeau
# (retour bureau du 31/08) — le bouton ＋ tient sur une colonne étroite,
# la première colonne ne porte plus qu'un numéro de couche.
COUCHE_COLS = [0.4, 1.5, 1.9, 1.05, 1.05, 1.05, 0.6]


def open_bloc_left_right(left: str, right: str = "", etat: str = "ok", pct=None):
    """Header de bloc : texte à gauche + (optionnel) texte à droite + pct + icône.
    HTML sur une seule ligne (voir fix v2.39.1 de poutre.py)."""
    parts = []
    if right:
        parts.append(f"<div style='font-weight:600;opacity:0.9;white-space:nowrap;'>{right}</div>")
    if pct is not None:
        try:
            parts.append(f"<div style='font-weight:700;white-space:nowrap;'>{float(pct):.0f} %</div>")
        except Exception:
            pass
    parts.append(f"<div style='font-size:20px;line-height:1;'>{C_ICONES.get(etat, '')}</div>")
    right_side = "".join(parts)
    bg = C_COULEURS.get(etat, '#f6f6f6')
    html = (
        f'<div style="background-color:{bg};padding:12px 14px 10px 14px;'
        f'border-radius:10px;border:1px solid #d9d9d9;margin:10px 0 12px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'gap:10px;margin-bottom:6px;">'
        f'<div style="font-weight:700;">{left}</div>'
        f'<div style="display:flex;align-items:center;gap:10px;">{right_side}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def close_bloc():
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
#  UTILITAIRES SESSION / CLÉS
# ============================================================
def KD(base: str, dalle_id: int) -> str:
    return f"pre{dalle_id}_{base}"


def KS(base: str, dalle_id: int, sec_id: int) -> str:
    return f"pre{dalle_id}_sec{sec_id}_{base}"


# Clés globales persistées (sauvegarde JSON + épinglage anti-nettoyage).
# Partagées avec le module Poutre : mêmes paramètres, même logiciel.
PERSISTED_GLOBAL_KEYS = {
    "units_len",
    "units_as",
    "gamma_s",              # coefficient acier γs (fyd = fyk / γs)
    "jeu_enrobage_cm",      # "Jeu premier lit (cm)"
    "nom_projet",
    "partie",
    "date",
    "indice",
    "chk_infos_projet",
}

# Clés à ne jamais épingler ni sauvegarder (transitoires / widgets boutons)
_TRANSIENT_MARKERS = ("btn", "open_uploader", "show_open_uploader", "pdf_bytes", "show_param_avances")


def _is_transient_key(k: str) -> bool:
    return any(m in k for m in _TRANSIENT_MARKERS)


def _is_dalle_key(k: str) -> bool:
    return bool(re.match(r"^pre\d+_", k)) or k.startswith("meta_predalle_nom_") or \
        (k.startswith("meta_pre") and "_nom_" in k)


def _pin_persistent_state():
    """
    FIX PERSISTANCE (voir poutre.py) : ré-affecter chaque clé persistante
    à elle-même en début de run pour empêcher Streamlit de nettoyer
    l'état des widgets non rendus (champs conditionnels).
    """
    for k in list(st.session_state.keys()):
        if _is_transient_key(k):
            continue
        if _is_dalle_key(k) or k.startswith("meta_") or k in PERSISTED_GLOBAL_KEYS:
            st.session_state[k] = st.session_state[k]
        elif k.endswith("_raw") and (_is_dalle_key(k[:-4]) or k[:-4] in PERSISTED_GLOBAL_KEYS):
            st.session_state[k] = st.session_state[k]


def _sync_float_raw_keys():
    """
    FIX BUG RECALCUL (voir poutre.py) : synchroniser toutes les clés
    *_raw vers leur clé numérique en tout début de run, AVANT tout
    calcul.
    """
    for k in list(st.session_state.keys()):
        if not k.endswith("_raw"):
            continue
        base = k[:-4]
        try:
            val = float(str(st.session_state[k]).strip().replace(",", "."))
        except Exception:
            continue
        st.session_state[base] = max(0.0, float(val))


def _fr(x, nd=1):
    """Format FR (virgule décimale)."""
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except Exception:
        return str(x)


def _ensure_global_defaults():
    """Défauts des paramètres globaux (partagés avec Poutre)."""
    st.session_state.setdefault("units_len", "cm")
    st.session_state.setdefault("units_as", "mm²")
    st.session_state.setdefault("jeu_enrobage_cm", 1.0)   # jeu premier lit

    # Coefficient acier γs (défaut 1.5)
    try:
        gs = float(st.session_state.get("gamma_s", 1.5) or 1.5)
    except Exception:
        gs = 1.5
    if gs <= 0:
        gs = 1.5
    st.session_state["gamma_s"] = gs


# ============================================================
#  RESET (limité aux données du module Prédalle)
# ============================================================
def _reset_module():
    """Réinitialise UNIQUEMENT les données du module Prédalle : les données
    du module Poutre et les paramètres globaux partagés sont conservés."""
    for k in list(st.session_state.keys()):
        if _is_dalle_key(k) or (k.endswith("_raw") and _is_dalle_key(k[:-4])) or k.startswith("predalle_"):
            del st.session_state[k]
    st.session_state.pop("predalles", None)
    st.rerun()


# ============================================================
#  SAISIE DÉCIMALE FR (texte) — identique à poutre.py
# ============================================================
def float_input_fr_simple(label, key, default=0.0, min_value=0.0, disabled: bool = False,
                          label_visibility: str = "visible"):
    if key not in st.session_state:
        st.session_state[key] = float(default)
    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = f"{float(st.session_state[key]):.2f}".replace(".", ",")

    raw = st.text_input(label, key=raw_key, disabled=disabled, label_visibility=label_visibility)

    try:
        val = float(str(raw).strip().replace(",", "."))
    except Exception:
        val = float(st.session_state[key])

    val = max(min_value, val)
    st.session_state[key] = float(val)
    return val


# ============================================================
#  COERCITIONS (pré-rendu uniquement)
# ============================================================
def _coerce_int_choice(key: str, options: list, default: int):
    cur = st.session_state.get(key, default)
    try:
        cur = int(float(cur))
    except Exception:
        cur = default
    if cur not in options:
        cur = default
    if st.session_state.get(key) != cur:
        st.session_state[key] = cur


def _coerce_str_choice(key: str, options: list, default: str):
    cur = str(st.session_state.get(key, default) or default)
    if cur not in options:
        cur = default
    if st.session_state.get(key) != cur:
        st.session_state[key] = cur


# ============================================================
#  NOMS DE SECTIONS PAR LETTRES (A, B, ..., Z, AA, AB, ...)
# ============================================================
def _letter_sequence():
    for c in ascii_uppercase:
        yield c
    for a in ascii_uppercase:
        for b in ascii_uppercase:
            yield a + b


def _next_section_name(dalle_id: int) -> str:
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    used = set()
    for s in dalle.get("sections", []):
        sid = int(s.get("id"))
        used.add(str(st.session_state.get(f"meta_pre{dalle_id}_nom_{sid}", s.get("nom", ""))).strip())
        used.add(str(s.get("nom", "")).strip())
    for L in _letter_sequence():
        if L not in used:
            return L
    return f"S{len(dalle.get('sections', [])) + 1}"  # repli improbable


# ============================================================
#  DALLES / SECTIONS : INIT / ADD / DELETE / DUPLICATE / COPY
# ============================================================
def _init_dalles_if_needed():
    if "predalles" not in st.session_state or not isinstance(st.session_state.predalles, list) or len(st.session_state.predalles) == 0:
        st.session_state.predalles = [{"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]}]

    for d in st.session_state.predalles:
        d["id"] = int(d.get("id", 0))
        d["nom"] = str(d.get("nom", f"Prédalle {d['id']}"))
        if "sections" not in d or not isinstance(d["sections"], list) or len(d["sections"]) == 0:
            d["sections"] = [{"id": 1, "nom": "A"}]
        for s in d["sections"]:
            s["id"] = int(s.get("id", 0))
            s["nom"] = str(s.get("nom", f"Section {s['id']}"))

    if not any(int(d.get("id", 0)) == 1 for d in st.session_state.predalles):
        st.session_state.predalles.insert(0, {"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]})

    for d in st.session_state.predalles:
        if not any(int(s.get("id", 0)) == 1 for s in d["sections"]):
            d["sections"].insert(0, {"id": 1, "nom": "A"})

    # Synchronisation des noms (labels d'expander à jour immédiatement)
    for d in st.session_state.predalles:
        did = int(d["id"])
        key_nom = f"meta_predalle_nom_{did}"
        if key_nom not in st.session_state:
            st.session_state[key_nom] = str(d.get("nom", f"Prédalle {did}"))
        d["nom"] = str(st.session_state.get(key_nom, d.get("nom")))

        for s in d.get("sections", []):
            sid = int(s["id"])
            key_snom = f"meta_pre{did}_nom_{sid}"
            if key_snom not in st.session_state:
                st.session_state[key_snom] = str(s.get("nom", "A"))
            s["nom"] = str(st.session_state.get(key_snom, s.get("nom", "A")))

    for d in st.session_state.predalles:
        _ensure_defaults_for_dalle(int(d["id"]))


def _next_dalle_id() -> int:
    ids = [int(d.get("id", 0)) for d in st.session_state.predalles]
    return (max(ids) + 1) if ids else 1


def _next_section_id(dalle_id: int) -> int:
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    ids = [int(s.get("id", 0)) for s in dalle["sections"]]
    return (max(ids) + 1) if ids else 1


DIAM_OPTS = [6, 8, 10, 12, 16, 20, 25, 32, 40]
# « Barres » = Ø à espacement régulier (Ø12/150) ; « n barres » = un
# NOMBRE de barres posées dans la bande (« 3 Ø12 ») — demandé par le
# bureau pour les renforts locaux. Ancien fichier : « Barres » inchangé.
TYPES_ARMATURE = ["Treillis", "Barres", "n barres"]


# Familles de clés de couches à migrer v1 -> v2 (une par face, une par
# face et par couche). Les clés d'affichage (as_disp_/dist_disp_) sont
# de purs widgets : elles sont purgées, jamais migrées.
_MIG_PAR_FACE = ("ncouches", "ycdg", "ycdg_auto", "ycdg_lastauto")
_MIG_PAR_COUCHE = ("arm_type", "treillis", "ø_barres", "esp_barres")


def _migrate_section_v2(dalle_id: int, sec_id: int):
    """
    Migration v1 -> v2 d'une section : l'unidirectionnel devient la
    DIRECTION X (M_inf -> Mx_inf, M_sup -> Mx_sup, couches inf/sup ->
    inf_x/sup_x) ; la direction Y démarre sur ses défauts. Idempotente :
    une valeur v1 n'est copiée que si la clé v2 n'existe pas encore,
    et les clés v1 sont retirées dans tous les cas.
    """
    ss = st.session_state

    def _bascule(old: str, new: str):
        if old in ss and new not in ss:
            ss[new] = ss[old]
        ss.pop(old, None)

    # Sollicitations (+ champs texte _raw associés)
    for old, new in (("M_inf", "Mx_inf"), ("M_sup", "Mx_sup")):
        _bascule(KS(old, dalle_id, sec_id), KS(new, dalle_id, sec_id))
        _bascule(KS(old, dalle_id, sec_id) + "_raw", KS(new, dalle_id, sec_id) + "_raw")

    # Couches : inf -> inf_x, sup -> sup_x
    for face in ("inf", "sup"):
        for fam in _MIG_PAR_FACE:
            _bascule(KS(f"{fam}_{face}", dalle_id, sec_id),
                     KS(f"{fam}_{face}_x", dalle_id, sec_id))
        for i in range(1, MAX_COUCHES + 1):
            for fam in _MIG_PAR_COUCHE:
                _bascule(KS(f"{fam}_{face}_c{i}", dalle_id, sec_id),
                         KS(f"{fam}_{face}_x_c{i}", dalle_id, sec_id))
            for fam in ("as_disp", "dist_disp"):
                ss.pop(KS(f"{fam}_{face}_{i}", dalle_id, sec_id), None)

    # v2.1 : plus d'étriers dans une dalle — purge des clés de cisaillement
    ss.pop(KS("shear_pas", dalle_id, sec_id), None)
    ss.pop(KS("shear_pas", dalle_id, sec_id) + "_raw", None)
    ss.pop(KS("shear_n_lines", dalle_id, sec_id), None)
    for i in range(8):
        for suf in ("type", "d"):
            ss.pop(KS(f"shear_line{i}_{suf}", dalle_id, sec_id), None)


def _ensure_defaults_for_dalle(dalle_id: int):
    # Prédalle : bande de 100 cm de large, 20 cm d'épaisseur par défaut
    st.session_state.setdefault(KD("b", dalle_id), 100)
    st.session_state.setdefault(KD("h", dalle_id), 22)
    # épaisseur de la peau préfabriquée — coulé en place = h − h_pre
    st.session_state.setdefault(KD("h_pre", dalle_id), 6.0)
    st.session_state.setdefault(KD("enrobage_beton", dalle_id), 3.0)

    if BETON_DATA:
        default_beton = "C30/37" if "C30/37" in BETON_DATA else list(BETON_DATA.keys())[0]
        st.session_state.setdefault(KD("beton", dalle_id), default_beton)
        if st.session_state.get(KD("beton", dalle_id)) not in BETON_DATA:
            st.session_state[KD("beton", dalle_id)] = default_beton

    # Acier par dalle : ENTIER 400 ou 500
    fkey = KD("fyk", dalle_id)
    cur = st.session_state.get(fkey, 500)
    try:
        cur = int(float(cur))
    except Exception:
        cur = 500
    if cur not in (400, 500):
        cur = 500
    st.session_state[fkey] = cur

    # Verrouillage par dalle (cadenas)
    st.session_state.setdefault(KD("lock_data", dalle_id), False)

    # Direction principale de la dalle : Y par défaut (convention bureau)
    dpk = KD("dir_principale", dalle_id)
    if str(st.session_state.get(dpk, "Y")).upper() not in ("X", "Y"):
        st.session_state[dpk] = "Y"
    st.session_state.setdefault(dpk, "Y")

    # Sections
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    for s in dalle.get("sections", []):
        sid = int(s["id"])
        _migrate_section_v2(dalle_id, sid)
        st.session_state.setdefault(KS("Mx_inf", dalle_id, sid), 0.0)
        st.session_state.setdefault(KS("Mx_sup", dalle_id, sid), 0.0)
        st.session_state.setdefault(KS("My_inf", dalle_id, sid), 0.0)
        st.session_state.setdefault(KS("My_sup", dalle_id, sid), 0.0)
        st.session_state.setdefault(KS("V", dalle_id, sid), 0.0)

        for which in FACES_DIR:
            nk = KS(f"ncouches_{which}", dalle_id, sid)
            try:
                nc = int(st.session_state.get(nk, 1) or 1)
            except Exception:
                nc = 1
            nc = max(1, min(MAX_COUCHES, nc))
            st.session_state[nk] = nc
            for i in range(1, nc + 1):
                # Couche 1 (base) : treillis par défaut ;
                # renforts : « n barres » (3 Ø12) par défaut.
                st.session_state.setdefault(
                    KS(f"arm_type_{which}_c{i}", dalle_id, sid),
                    "Treillis" if i == 1 else "n barres",
                )
                _coerce_str_choice(KS(f"arm_type_{which}_c{i}", dalle_id, sid), TYPES_ARMATURE,
                                   "Treillis" if i == 1 else "n barres")
                st.session_state.setdefault(KS(f"treillis_{which}_c{i}", dalle_id, sid), TR.TREILLIS_DEFAUT)
                st.session_state.setdefault(KS(f"ø_barres_{which}_c{i}", dalle_id, sid), 12)
                _coerce_int_choice(KS(f"ø_barres_{which}_c{i}", dalle_id, sid), DIAM_OPTS, 12)
                # Espacement : entier borné (le widget est un number_input
                # entier — un ancien JSON peut porter un flottant)
                ek = KS(f"esp_barres_{which}_c{i}", dalle_id, sid)
                try:
                    ev = int(float(st.session_state.get(ek, 150) or 150))
                except Exception:
                    ev = 150
                ev = max(25, min(500, ev))
                if st.session_state.get(ek) != ev:
                    st.session_state[ek] = ev
                # Nombre de barres (« n barres ») : entier borné
                nbk = KS(f"n_barres_{which}_c{i}", dalle_id, sid)
                try:
                    nv = int(float(st.session_state.get(nbk, 3) or 3))
                except Exception:
                    nv = 3
                nv = max(1, min(50, nv))
                if st.session_state.get(nbk) != nv:
                    st.session_state[nbk] = nv

        # v2.1 : plus d'étriers dans une dalle — les anciennes clés de
        # cisaillement (lignes, pas) sont purgées par la migration.


def _delete_dalle(dalle_id: int):
    if dalle_id == 1:
        return
    st.session_state.predalles = [d for d in st.session_state.predalles if int(d.get("id")) != dalle_id]
    prefix = f"pre{dalle_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_predalle_nom_{dalle_id}", None)
    for k in list(st.session_state.keys()):
        if k.startswith(f"meta_pre{dalle_id}_nom_"):
            del st.session_state[k]


def _duplicate_dalle(src_dalle_id: int):
    """Bouton 📋 'Copier la dalle' (toutes les données)."""
    src = next(d for d in st.session_state.predalles if int(d.get("id")) == src_dalle_id)
    new_id = _next_dalle_id()
    st.session_state.predalles.append({"id": new_id, "nom": f"{src.get('nom','Prédalle')} (copie)", "sections": deepcopy(src["sections"])})

    src_prefix = f"pre{src_dalle_id}_"
    dst_prefix = f"pre{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_predalle_nom_{new_id}"] = f"{st.session_state.get(f'meta_predalle_nom_{src_dalle_id}', src.get('nom','Prédalle'))} (copie)"
    for s in src.get("sections", []):
        sid = int(s.get("id"))
        st.session_state[f"meta_pre{new_id}_nom_{sid}"] = st.session_state.get(f"meta_pre{src_dalle_id}_nom_{sid}", s.get("nom", f"Section {sid}"))

    _ensure_defaults_for_dalle(new_id)


def _add_section(dalle_id: int):
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    new_id = _next_section_id(dalle_id)
    name = _next_section_name(dalle_id)
    dalle["sections"].append({"id": new_id, "nom": name})
    st.session_state[f"meta_pre{dalle_id}_nom_{new_id}"] = name
    _ensure_defaults_for_dalle(dalle_id)


def _copy_section(dalle_id: int, src_sec_id: int):
    """Copie intégrale d'une section (sollicitations, couches inf./sup.,
    cisaillement) vers une nouvelle section nommée avec la première
    lettre disponible. Callback on_click."""
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    new_id = _next_section_id(dalle_id)
    name = _next_section_name(dalle_id)
    dalle["sections"].append({"id": new_id, "nom": name})

    src_prefix = f"pre{dalle_id}_sec{src_sec_id}_"
    dst_prefix = f"pre{dalle_id}_sec{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_pre{dalle_id}_nom_{new_id}"] = name
    _ensure_defaults_for_dalle(dalle_id)


def _delete_section(dalle_id: int, sec_id: int):
    if sec_id == 1:
        return
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    dalle["sections"] = [s for s in dalle["sections"] if int(s.get("id")) != sec_id]
    prefix = f"pre{dalle_id}_sec{sec_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_pre{dalle_id}_nom_{sec_id}", None)


# ============================================================
#  SAVE / LOAD JSON (dalles + valeurs)
# ============================================================
def _build_save_payload():
    dalles = []
    for d in st.session_state.predalles:
        dalles.append(
            {
                "id": int(d.get("id")),
                "nom": str(d.get("nom")),
                "sections": [{"id": int(s.get("id")), "nom": str(s.get("nom"))} for s in d.get("sections", [])],
            }
        )

    values = {}
    for k in list(st.session_state.keys()):
        if _is_transient_key(k):
            continue
        if k in PERSISTED_GLOBAL_KEYS or (k.endswith("_raw") and k[:-4] in PERSISTED_GLOBAL_KEYS):
            values[k] = st.session_state[k]
        elif _is_dalle_key(k) or (k.endswith("_raw") and _is_dalle_key(k[:-4])):
            values[k] = st.session_state[k]

    return {"version": "predalle-1.0", "predalles": dalles, "values": values}


def _load_from_payload(payload: dict):
    dalles = payload.get("predalles", payload.get("dalles", None))
    values = payload.get("values", {})

    if isinstance(dalles, list) and len(dalles) > 0:
        cleaned = []
        for d in dalles:
            try:
                did = int(d.get("id"))
            except Exception:
                continue
            secs = d.get("sections", [])
            if not isinstance(secs, list) or len(secs) == 0:
                secs = [{"id": 1, "nom": "A"}]
            cleaned_secs = []
            for s in secs:
                try:
                    sid = int(s.get("id"))
                except Exception:
                    continue
                cleaned_secs.append({"id": sid, "nom": str(s.get("nom", f"Section {sid}"))})
            cleaned.append({"id": did, "nom": str(d.get("nom", f"Prédalle {did}")), "sections": cleaned_secs})
        st.session_state.predalles = cleaned if cleaned else [{"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]}]
    else:
        st.session_state.predalles = [{"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]}]

    if isinstance(values, dict):
        for k, v in values.items():
            if _is_transient_key(k):
                continue
            if k in PERSISTED_GLOBAL_KEYS or (k.endswith("_raw") and k[:-4] in PERSISTED_GLOBAL_KEYS):
                st.session_state[k] = v
            elif _is_dalle_key(k) or (k.endswith("_raw") and _is_dalle_key(k[:-4])):
                st.session_state[k] = v

    _ensure_global_defaults()
    _init_dalles_if_needed()


# ============================================================
#  OUTILS CALCUL (identiques à poutre.py)
# ============================================================
def _bar_area_mm2(diam_mm: float) -> float:
    return math.pi * (diam_mm / 2.0) ** 2


def _status_merge(*states: str) -> str:
    if any(s == "nok" for s in states):
        return "nok"
    if any(s == "warn" for s in states):
        return "warn"
    return "ok"


def _status_icon_label(state: str, label: str) -> str:
    if state == "ok":
        return f"🟢 {label}"
    if state == "warn":
        return f"🟡 {label}"
    return f"🔴 {label}"


def _get_fyk_and_mu_ref(dalle_id: int):
    try:
        fyk_i = int(float(st.session_state.get(KD("fyk", dalle_id), 500)))
    except Exception:
        fyk_i = 500
    if fyk_i not in (400, 500):
        fyk_i = 500
    return float(fyk_i), str(fyk_i)


def _get_gamma_s() -> float:
    try:
        gs = float(st.session_state.get("gamma_s", 1.5) or 1.5)
    except Exception:
        gs = 1.5
    return gs if gs > 0 else 1.5


def _round_up_to_half_cm(x_cm: float) -> float:
    try:
        return math.ceil(float(x_cm) * 2.0) / 2.0
    except Exception:
        return x_cm


# ============================================================
#  COUCHES D'ARMATURES (base + renforts)
# ============================================================
def _get_ncouches(dalle_id: int, sec_id: int, which: str) -> int:
    try:
        nc = int(st.session_state.get(KS(f"ncouches_{which}", dalle_id, sec_id), 1) or 1)
    except Exception:
        nc = 1
    return max(1, min(MAX_COUCHES, nc))


def _couche_type(dalle_id: int, sec_id: int, which: str, i: int) -> str:
    t = str(st.session_state.get(KS(f"arm_type_{which}_c{i}", dalle_id, sec_id),
                                 "Treillis" if i == 1 else "Barres"))
    return t if t in TYPES_ARMATURE else "Treillis"


def _couche_data(dalle_id: int, sec_id: int, which: str, i: int):
    """
    Données d'une couche : (type, désignation treillis, Ø barres,
    esp barres, nombre de barres).
    """
    typ = _couche_type(dalle_id, sec_id, which, i)
    des = str(st.session_state.get(KS(f"treillis_{which}_c{i}", dalle_id, sec_id), TR.TREILLIS_DEFAUT))
    try:
        d = int(float(st.session_state.get(KS(f"ø_barres_{which}_c{i}", dalle_id, sec_id), 12) or 12))
    except Exception:
        d = 12
    try:
        esp = float(st.session_state.get(KS(f"esp_barres_{which}_c{i}", dalle_id, sec_id), 150) or 150)
    except Exception:
        esp = 150.0
    try:
        n = int(float(st.session_state.get(KS(f"n_barres_{which}_c{i}", dalle_id, sec_id), 3) or 3))
    except Exception:
        n = 3
    return typ, des, d, esp, max(1, n)


def _couche_as_per_m(dalle_id: int, sec_id: int, which: str, i: int) -> float:
    """Section d'acier de la couche i (mm²/m), calculée automatiquement.
    « n barres » : n barres dans la bande de largeur b -> n·aire/(b/100),
    même aire de barre que partout ailleurs (π·Ø²/4)."""
    typ, des, d, esp, n = _couche_data(dalle_id, sec_id, which, i)
    if typ == "Treillis":
        return TR.as_treillis_mm2_m(des)
    if typ == "n barres":
        b = float(st.session_state.get(KD("b", dalle_id), 100) or 100)
        return n * (math.pi * d * d / 4.0) / max(0.01, b / 100.0)
    return TR.as_barres_mm2_m(d, esp)


def _couche_diam_mm(dalle_id: int, sec_id: int, which: str, i: int) -> float:
    """Ø des fils/barres de la couche (mm) — sert à la distance d'axe."""
    typ, des, d, esp, n = _couche_data(dalle_id, sec_id, which, i)
    if typ == "Treillis":
        t = TR.parse_designation(des)
        return float(t[0]) if t else 10.0
    return float(d)


def _couche_label(dalle_id: int, sec_id: int, which: str, i: int) -> str:
    """Libellé compact : 'Treillis 10/10/100/100', 'Ø12/150' ou '3 Ø12'."""
    typ, des, d, esp, n = _couche_data(dalle_id, sec_id, which, i)
    if typ == "Treillis":
        return f"Treillis {des}"
    if typ == "n barres":
        return f"{n} Ø{d}"
    esp_txt = f"{esp:.0f}" if abs(esp - round(esp)) < 1e-9 else _fr(esp, 1)
    return f"Ø{d}/{esp_txt}"


def _auto_dist_couche(dalle_id: int, sec_id: int, which: str, i: int) -> float:
    """
    Distance d'axe PAR DÉFAUT de la couche i (cm) — LA règle prédalle :

      - couche 1 INFÉRIEURE de la direction PRINCIPALE : DANS la prédalle
        préfabriquée -> enrobage + demi-Ø arrondi au 0,5 cm sup. + jeu
        premier lit (règle Dalle inchangée, ex. Ø10 : 3,0+0,5+1,0 = 4,5) ;
      - TOUTE autre couche INFÉRIEURE (couche 1 de la direction
        secondaire, et tous les renforts, posés sur chantier) :
        AU-DESSUS de la prédalle -> h_pre + demi-Ø arrondi au 0,5 cm sup.
        (ex. h_pre 6, Ø10 : 6,0 + 0,5 = 6,5 cm) ;
      - couches SUPÉRIEURES : règle Dalle inchangée (depuis la face
        supérieure, coulée en place).

    L'utilisateur peut toujours SAISIR une autre distance d'axe par
    couche (colonne « Dist. axe ») — voir _dist_couche_eff.
    """
    d = _couche_diam_mm(dalle_id, sec_id, which, i)
    if which.startswith("inf"):
        principale = str(st.session_state.get(KD("dir_principale", dalle_id), "Y") or "Y").lower()
        dk = which.rsplit("_", 1)[-1]
        if not (i == 1 and dk == principale):
            h_pre = float(st.session_state.get(KD("h_pre", dalle_id), 6.0) or 6.0)
            return h_pre + _round_up_to_half_cm(d / 20.0)
    enrob_beton = float(st.session_state.get(KD("enrobage_beton", dalle_id), 3.0) or 3.0)
    jeu1 = float(st.session_state.get("jeu_enrobage_cm", 1.0) or 0.0)
    return enrob_beton + _round_up_to_half_cm(d / 20.0) + jeu1


def _dist_couche_eff(dalle_id: int, sec_id: int, which: str, i: int) -> float:
    """Distance d'axe EFFECTIVE de la couche i (cm) : la saisie de la
    colonne « Dist. axe » si elle est valide, sinon l'automatique. Le
    CDG, la hauteur utile et les schémas de la note suivent cette valeur."""
    auto = _auto_dist_couche(dalle_id, sec_id, which, i)
    if bool(st.session_state.get(KS(f"dist_auto_{which}_c{i}", dalle_id, sec_id), True)):
        return auto
    raw = str(st.session_state.get(KS(f"dist_axe_{which}_c{i}", dalle_id, sec_id), "") or "").strip()
    try:
        v = float(raw.replace(",", "."))
        return v if v > 0 else auto
    except Exception:
        return auto


def _sync_dist_state(dalle_id: int, sec_id: int, which: str, i: int):
    """
    Synchronise le champ « Dist. axe » de la couche i avec la valeur
    automatique — même mécanisme que le CDG (_sync_ycdg_state) : tant
    que l'utilisateur n'a rien saisi, le champ suit l'automatique ; une
    vraie saisie fige la valeur ; champ vidé -> retour à l'automatique.
    À appeler AVANT le rendu du widget.
    """
    auto = _auto_dist_couche(dalle_id, sec_id, which, i)
    auto_txt = f"{auto:.1f}".replace(".", ",")

    dkey = KS(f"dist_axe_{which}_c{i}", dalle_id, sec_id)
    flag = KS(f"dist_auto_{which}_c{i}", dalle_id, sec_id)
    last = KS(f"dist_lastauto_{which}_c{i}", dalle_id, sec_id)

    if dkey not in st.session_state:
        st.session_state[dkey] = auto_txt
        st.session_state[flag] = True
    st.session_state.setdefault(flag, True)
    st.session_state.setdefault(last, auto_txt)

    cur_raw = str(st.session_state.get(dkey, "") or "").strip()
    last_auto = str(st.session_state.get(last, "") or "")

    if cur_raw == "":
        st.session_state[flag] = True
    elif bool(st.session_state.get(flag, False)) and cur_raw not in (last_auto, auto_txt):
        st.session_state[flag] = False

    if bool(st.session_state.get(flag, False)):
        st.session_state[dkey] = auto_txt
    st.session_state[last] = auto_txt


def _ycdg_manual(dalle_id: int, sec_id: int, which: str):
    """yG imposé par l'utilisateur (cm) ou None si vide/auto/invalide."""
    if bool(st.session_state.get(KS(f"ycdg_auto_{which}", dalle_id, sec_id), False)):
        return None
    raw = str(st.session_state.get(KS(f"ycdg_{which}", dalle_id, sec_id), "") or "").strip()
    if not raw:
        return None
    try:
        v = float(raw.replace(",", "."))
        return v if v > 0 else None
    except Exception:
        return None


def _sync_ycdg_state(dalle_id: int, sec_id: int, which: str):
    """
    Synchronise le champ 'CDG (cm)' avec la valeur calculée — même
    mécanisme que poutre.py v2.39 (comparaison à la dernière valeur
    auto affichée pour détecter une vraie saisie manuelle).
    Retourne (e_auto, is_auto). À appeler AVANT le rendu du widget.
    """
    _, e_auto, _, _ = _layers_geometry(dalle_id, sec_id, which, use_manual=False)
    e_auto_txt = f"{e_auto:.1f}".replace(".", ",")

    ykey = KS(f"ycdg_{which}", dalle_id, sec_id)
    auto_flag = KS(f"ycdg_auto_{which}", dalle_id, sec_id)
    last_key = KS(f"ycdg_lastauto_{which}", dalle_id, sec_id)

    if ykey not in st.session_state:
        st.session_state[ykey] = e_auto_txt
        st.session_state[auto_flag] = True
    st.session_state.setdefault(auto_flag, True)
    st.session_state.setdefault(last_key, e_auto_txt)

    cur_raw = str(st.session_state.get(ykey, "") or "").strip()
    last_auto = str(st.session_state.get(last_key, "") or "")

    if cur_raw == "":
        st.session_state[auto_flag] = True
    elif bool(st.session_state.get(auto_flag, False)) and cur_raw not in (last_auto, e_auto_txt):
        st.session_state[auto_flag] = False

    is_auto = bool(st.session_state.get(auto_flag, False))
    if is_auto:
        st.session_state[ykey] = e_auto_txt
    st.session_state[last_key] = e_auto_txt
    return e_auto, is_auto


def _layers_geometry(dalle_id: int, sec_id: int, which: str, use_manual: bool = True):
    """
    Pour une face :
      - As_total (mm²) sur la LARGEUR RÉELLE de la bande b
      - e_cdg (cm) = yG : distance parement -> c.d.g. pondéré des couches
        (Σ As·e / Σ As) ; un yG manuel saisi le REMPLACE (use_manual)
      - detail : 'Treillis 10/10/100/100 + Ø12/150'
      - As_per_m (mm²/m) : somme des couches au mètre
    """
    b = float(st.session_state.get(KD("b", dalle_id), 100))
    nc = _get_ncouches(dalle_id, sec_id, which)
    parts = []
    As_pm_tot = 0.0
    somme_As_e = 0.0

    for i in range(1, nc + 1):
        As_pm = _couche_as_per_m(dalle_id, sec_id, which, i)
        e = _dist_couche_eff(dalle_id, sec_id, which, i)
        As_pm_tot += As_pm
        somme_As_e += As_pm * e
        parts.append(_couche_label(dalle_id, sec_id, which, i))

    e_cdg = (somme_As_e / As_pm_tot) if As_pm_tot > 0 else _dist_couche_eff(dalle_id, sec_id, which, 1)
    if use_manual:
        man = _ycdg_manual(dalle_id, sec_id, which)
        if man is not None:
            e_cdg = man
    As_tot = As_pm_tot * (b / 100.0)
    return As_tot, e_cdg, " + ".join(parts), As_pm_tot


# ============================================================
#  COUCHES : callbacks ajout / suppression
# ============================================================
def _add_couche(dalle_id: int, sec_id: int, which: str):
    nk = KS(f"ncouches_{which}", dalle_id, sec_id)
    nc = _get_ncouches(dalle_id, sec_id, which)
    if nc >= MAX_COUCHES:
        return
    i = nc + 1
    # Renfort par défaut : 3 Ø12 posées dans la bande (« n barres ») —
    # modifiable en treillis ou en barres à espacement.
    st.session_state[KS(f"arm_type_{which}_c{i}", dalle_id, sec_id)] = "n barres"
    st.session_state.setdefault(KS(f"treillis_{which}_c{i}", dalle_id, sec_id), TR.TREILLIS_DEFAUT)
    st.session_state.setdefault(KS(f"ø_barres_{which}_c{i}", dalle_id, sec_id), 12)
    st.session_state.setdefault(KS(f"esp_barres_{which}_c{i}", dalle_id, sec_id), 150)
    st.session_state.setdefault(KS(f"n_barres_{which}_c{i}", dalle_id, sec_id), 3)
    st.session_state[nk] = i


def _delete_couche(dalle_id: int, sec_id: int, which: str, i: int):
    """Callback on_click : suppression du renfort i (i>=2) avec décalage."""
    nk = KS(f"ncouches_{which}", dalle_id, sec_id)
    nc = _get_ncouches(dalle_id, sec_id, which)
    if i < 2 or i > nc:
        return
    _SUFS = ("arm_type", "treillis", "ø_barres", "esp_barres", "n_barres",
             "dist_axe", "dist_auto", "dist_lastauto")
    for j in range(i, nc):
        for suf in _SUFS:
            st.session_state[KS(f"{suf}_{which}_c{j}", dalle_id, sec_id)] = st.session_state.get(
                KS(f"{suf}_{which}_c{j+1}", dalle_id, sec_id)
            )
    for suf in _SUFS:
        st.session_state.pop(KS(f"{suf}_{which}_c{nc}", dalle_id, sec_id), None)
    st.session_state[nk] = nc - 1


# ============================================================
#  UI : SECTIONS (SOLLICITATIONS) PAR DALLE
# ============================================================
def _render_section_inputs(dalle_id: int, sec_id: int, disabled: bool):
    """Cinq sollicitations sur une seule ligne compacte : Mx inf/sup,
    My inf/sup (kN·m par bande) et V max (kN — convention inchangée :
    un seul effort tranchant par section)."""
    st.markdown(
        "<div style='font-size:0.85em;color:#6b7280;margin-bottom:-6px;'>"
        "Sollicitations — moments en kN·m, effort tranchant en kN</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        float_input_fr_simple("Mx inf", key=KS("Mx_inf", dalle_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c2:
        float_input_fr_simple("Mx sup", key=KS("Mx_sup", dalle_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c3:
        float_input_fr_simple("My inf", key=KS("My_inf", dalle_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c4:
        float_input_fr_simple("My sup", key=KS("My_sup", dalle_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c5:
        float_input_fr_simple("V max", key=KS("V", dalle_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)


def render_solicitations_for_dalle(dalle_id: int, data_locked: bool = False):
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    st.markdown("#### Sections")

    for sec in dalle.get("sections", []):
        sec_id = int(sec.get("id"))
        sec_name_key = f"meta_pre{dalle_id}_nom_{sec_id}"
        st.session_state.setdefault(sec_name_key, sec.get("nom", f"Section {sec_id}"))

        with st.container(border=True):
            cN, cA, cC, cD = st.columns([4.4, 0.8, 0.8, 0.8], vertical_alignment="bottom")
            with cN:
                st.text_input(
                    "Section",
                    key=sec_name_key,
                    disabled=data_locked,
                )
            with cA:
                st.button(
                    "➕",
                    key=f"predalle_add_sec_btn_{dalle_id}_{sec_id}",
                    help="Ajouter une section",
                    use_container_width=True,
                    on_click=_add_section,
                    args=(dalle_id,),
                    disabled=data_locked,
                )
            with cC:
                st.button(
                    "📋",
                    key=f"predalle_copy_sec_btn_{dalle_id}_{sec_id}",
                    help="Copier la section (toutes les données)",
                    use_container_width=True,
                    on_click=_copy_section,
                    args=(dalle_id, sec_id),
                    disabled=data_locked,
                )
            with cD:
                if sec_id != 1:
                    st.button(
                        "🗑️",
                        key=f"predalle_del_sec_btn_{dalle_id}_{sec_id}",
                        help="Supprimer la section",
                        use_container_width=True,
                        on_click=_delete_section,
                        args=(dalle_id, sec_id),
                        disabled=data_locked,
                    )

            _render_section_inputs(dalle_id, sec_id, disabled=data_locked)

    if dalle_id != 1:
        st.button(
            "🗑️ Supprimer la dalle",
            key=f"predalle_del_beam_btn_{dalle_id}",
            on_click=_delete_dalle,
            args=(dalle_id,),
            disabled=data_locked,
            use_container_width=True,
        )


def _toggle_dalle_lock(dalle_id: int):
    k = KD("lock_data", dalle_id)
    st.session_state[k] = not bool(st.session_state.get(k, False))


def render_caracteristiques_dalle(dalle_id: int):
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)

    dalle_name_key = f"meta_predalle_nom_{dalle_id}"
    st.session_state.setdefault(dalle_name_key, dalle.get("nom", f"Prédalle {dalle_id}"))

    lock_key = KD("lock_data", dalle_id)
    st.session_state.setdefault(lock_key, False)
    data_locked = bool(st.session_state.get(lock_key, False))

    with st.expander(st.session_state.get(dalle_name_key, dalle.get("nom", f"Prédalle {dalle_id}")), expanded=True):
        t1, tC, tL = st.columns([6, 0.8, 0.8], vertical_alignment="center")
        with t1:
            st.markdown("#### Caractéristiques de la dalle")
        with tC:
            st.button(
                "📋",
                key=f"predalle_btn_copy_{dalle_id}",
                help="Copier la dalle",
                use_container_width=True,
                on_click=_duplicate_dalle,
                args=(dalle_id,),
            )
        with tL:
            st.button(
                "🔒" if data_locked else "🔓",
                key=f"predalle_btn_lock_{dalle_id}",
                help="Prédalle verrouillée — cliquer pour déverrouiller" if data_locked
                     else "Prédalle éditable — cliquer pour verrouiller",
                use_container_width=True,
                on_click=_toggle_dalle_lock,
                args=(dalle_id,),
            )

        c1, c2, c3 = st.columns([2.6, 1.6, 1.4], vertical_alignment="center")
        with c1:
            st.text_input("Nom de la dalle", key=dalle_name_key, disabled=data_locked)
        with c2:
            st.selectbox("Classe de béton", list(BETON_DATA.keys()), key=KD("beton", dalle_id), disabled=data_locked)
        with c3:
            st.selectbox("Qualité acier (B)", [400, 500], key=KD("fyk", dalle_id), disabled=data_locked)

        cB, cH, cP, cE, cD = st.columns([1.1, 0.9, 0.9, 1.0, 0.9])
        with cB:
            st.number_input(
                "Larg. bande (cm)", min_value=20, max_value=500, step=5, key=KD("b", dalle_id),
                disabled=data_locked,
                help="Largeur de la bande de dalle étudiée — 100 cm = 1 mètre courant.",
            )
        with cH:
            st.number_input("Épaisseur (cm)", min_value=5, max_value=100, step=1, key=KD("h", dalle_id), disabled=data_locked)
        with cP:
            _h_tot = float(st.session_state.get(KD("h", dalle_id), 22) or 22)
            _h_pre = float(st.session_state.get(KD("h_pre", dalle_id), 6.0) or 6.0)
            st.number_input(
                "Prédalle (cm)", min_value=3.0, max_value=15.0, step=0.5,
                key=KD("h_pre", dalle_id), disabled=data_locked,
                help="Épaisseur de la prédalle préfabriquée. Coulé en place = "
                     f"h − prédalle = {_h_tot - _h_pre:.0f} cm (calculé, jamais saisi).",
            )
        with cE:
            st.number_input("Enrob. béton (cm)", min_value=0.0, max_value=20.0, step=0.5, key=KD("enrobage_beton", dalle_id), disabled=data_locked)
        with cD:
            st.selectbox(
                "Dir. principale", ["Y", "X"], key=KD("dir_principale", dalle_id),
                disabled=data_locked,
                help="Direction principale de la dalle (Y par défaut). Pilote "
                     "l'ordre des cartes, la note de calcul et les schémas.",
            )

        render_solicitations_for_dalle(dalle_id, data_locked=data_locked)


# ============================================================
#  CALCUL DES ÉTATS D'UNE SECTION
#  (formules strictement identiques à poutre.py — seule la géométrie
#   des armatures provient des couches treillis/barres)
# ============================================================
def _dimensionnement_compute_states(dalle_id: int, sec_id: int, beton_data: dict):
    beton = str(st.session_state.get(KD("beton", dalle_id), "C30/37"))
    if beton not in beton_data:
        beton = list(beton_data.keys())[0]
    fck_cube = beton_data[beton]["fck_cube"]
    alpha_b = beton_data[beton]["alpha_b"]

    # fyd = fyk / γs
    fyk, mu_ref = _get_fyk_and_mu_ref(dalle_id)
    gamma_s = _get_gamma_s()
    fyd = fyk / gamma_s

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in beton_data[beton]:
        mu_key = "mu_a500" if "mu_a500" in beton_data[beton] else [k for k in beton_data[beton].keys() if k.startswith("mu_a")][0]
    mu_val = beton_data[beton][mu_key]

    b = float(st.session_state.get(KD("b", dalle_id), 100))
    h = float(st.session_state.get(KD("h", dalle_id), 20))

    # --- As min/max (formules Poutre inchangées, communes aux directions) ---
    fck_cyl = float(beton_data[beton].get("fck", 0.8 * fck_cube) or (0.8 * fck_cube))
    fctm = 0.30 * (fck_cyl ** (2.0 / 3.0)) if fck_cyl > 0 else 0.0
    As_min_ec = 0.26 * fctm / fyk * b * h * 1e2     # mm²  (b,h en cm -> ·1e2)
    As_min_plancher = 0.0013 * b * h * 1e2          # mm²
    As_min_base = max(As_min_ec, As_min_plancher)
    As_max = 0.04 * b * h * 1e2  # mm²

    # --- UNE PASSE PAR DIRECTION : mêmes expressions que la v1, la face
    #     opposée du critère 0,25·Aₛ,req est celle de la MÊME direction ---
    dirs = {}
    for dk in DIR_KEYS:
        w_inf, w_sup = f"inf_{dk}", f"sup_{dk}"
        _sync_ycdg_state(dalle_id, sec_id, w_inf)
        _sync_ycdg_state(dalle_id, sec_id, w_sup)

        As_inf_total, e_cdg_inf, inf_detail, As_inf_pm = _layers_geometry(dalle_id, sec_id, w_inf)
        As_sup_total, e_cdg_sup, sup_detail, As_sup_pm = _layers_geometry(dalle_id, sec_id, w_sup)

        d_utile_inf = h - e_cdg_inf  # cm
        d_utile_sup = h - e_cdg_sup  # cm
        geom_inf_ok = d_utile_inf > 0.0
        geom_sup_ok = d_utile_sup > 0.0
        d_calc_inf = max(d_utile_inf, 0.1)
        d_calc_sup = max(d_utile_sup, 0.1)

        M_inf_val = float(st.session_state.get(KS(f"M{dk}_inf", dalle_id, sec_id), 0.0) or 0.0)
        M_sup_val = float(st.session_state.get(KS(f"M{dk}_sup", dalle_id, sec_id), 0.0) or 0.0)

        As_formule_inf = (M_inf_val * 1e6) / (fyd * 0.9 * d_calc_inf * 10) if M_inf_val > 0 else 0.0
        As_formule_sup = (M_sup_val * 1e6) / (fyd * 0.9 * d_calc_sup * 10) if M_sup_val > 0 else 0.0

        As_min_inf_eff = max(As_min_base, 0.25 * As_formule_sup)
        As_min_sup_eff = max(As_min_base, 0.25 * As_formule_inf)

        As_req_inf_final = As_formule_inf
        As_req_sup_final = As_formule_sup

        etat_inf = "ok" if (geom_inf_ok and As_inf_total >= max(As_req_inf_final, As_min_inf_eff) and As_inf_total <= As_max) else "nok"
        etat_sup = "ok" if (geom_sup_ok and As_sup_total >= max(As_req_sup_final, As_min_sup_eff) and As_sup_total <= As_max) else "nok"

        dirs[dk] = {
            "M_inf_val": M_inf_val, "M_sup_val": M_sup_val,
            "As_inf_total": As_inf_total, "As_sup_total": As_sup_total,
            "As_inf_pm": As_inf_pm, "As_sup_pm": As_sup_pm,
            "inf_detail": inf_detail, "sup_detail": sup_detail,
            "e_cdg_inf": e_cdg_inf, "e_cdg_sup": e_cdg_sup,
            "d_utile_inf": d_utile_inf, "d_utile_sup": d_utile_sup,
            "geom_inf_ok": geom_inf_ok, "geom_sup_ok": geom_sup_ok,
            "As_formule_inf": As_formule_inf, "As_formule_sup": As_formule_sup,
            "As_min_inf_eff": As_min_inf_eff, "As_min_sup_eff": As_min_sup_eff,
            "As_req_inf_final": As_req_inf_final, "As_req_sup_final": As_req_sup_final,
            "etat_inf": etat_inf, "etat_sup": etat_sup,
        }

    # Direction principale : CHOIX de l'utilisateur (défaut Y) — pilote
    # l'ordre d'affichage, la note et les schémas (v2.1)
    principale = "x" if str(st.session_state.get(KD("dir_principale", dalle_id), "Y")).upper() == "X" else "y"

    V_val = float(st.session_state.get(KS("V", dalle_id, sec_id), 0.0) or 0.0)

    # --- Hauteur (formule Poutre inchangée) : M_max = max des 4 moments,
    #     d₁ = enrobage mécanique de la famille qui porte ce moment ---
    familles = [(dirs["x"]["M_inf_val"], dirs["x"]["e_cdg_inf"]),
                (dirs["x"]["M_sup_val"], dirs["x"]["e_cdg_sup"]),
                (dirs["y"]["M_inf_val"], dirs["y"]["e_cdg_inf"]),
                (dirs["y"]["M_sup_val"], dirs["y"]["e_cdg_sup"])]
    M_max = max(m for m, _ in familles)
    # première famille au moment maximal (ordre inf_x, sup_x, inf_y,
    # sup_y) — à égalité inf. l'emporte, comme dans la v1
    e_cdg_gov = next(e for m, e in familles if m == M_max)
    if M_max > 0:
        hmin_calc = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10  # cm
    else:
        hmin_calc = 0.0
    h_min_dalle = hmin_calc + e_cdg_gov
    etat_h = "ok" if (h_min_dalle <= h) else "nok"

    etat_inf = _status_merge(dirs["x"]["etat_inf"], dirs["y"]["etat_inf"])
    etat_sup = _status_merge(dirs["x"]["etat_sup"], dirs["y"]["etat_sup"])

    # --- Tranchant (v2.1) : une dalle ne reçoit pas d'étriers — SEULE la
    #     contrainte tangentielle est vérifiée, contre τ_adm,I (le seuil
    #     « pas besoin d'étriers » existant). Formule τ inchangée. ---
    tau_1 = 0.016 * fck_cube / 1.05
    tau_2 = 0.032 * fck_cube / 1.05
    tau_4 = 0.064 * fck_cube / 1.05

    if V_val > 0:
        tau = V_val * 1e3 / (0.75 * b * h * 100)
        etat_tau = "ok" if tau <= tau_1 else "nok"
    else:
        tau = 0.0
        etat_tau = "ok"

    etat_global = _status_merge(etat_h, etat_inf, etat_sup, etat_tau)

    return {
        "etat_global": etat_global,
        "etat_h": etat_h,
        "etat_inf": etat_inf,        # fusion X/Y (bandeaux de synthèse)
        "etat_sup": etat_sup,
        "etat_tau": etat_tau,
        "dirs": dirs,                # tout le détail par direction
        "principale": principale,    # "x" ou "y" (choix utilisateur)
        "V_val": V_val,
        "tau": tau,
        "tau_adm": tau_1,            # τ_adm,I : dalle sans étriers
        "M_max": M_max,
        "hmin_calc": hmin_calc,
        "e_cdg_gov": e_cdg_gov,
        "h_min_dalle": h_min_dalle,
        "tau_1": tau_1,
        "tau_2": tau_2,
        "tau_4": tau_4,
        "fyd": fyd,
        "gamma_s": gamma_s,
        "fyk": fyk,
        "alpha_b": alpha_b,
        "mu_val": mu_val,
        "beton": beton,
        "b": b,
        "h": h,
        "fctm": fctm,
        "As_min_ec": As_min_ec,
        "As_min_plancher": As_min_plancher,
        "As_max": As_max,
    }


# ============================================================
#  UI : TABLEAU DES COUCHES (Base / Renforts)
# ============================================================
_FACE_LABELS = {"inf_x": " (inf. X)", "sup_x": " (sup. X)",
                "inf_y": " (inf. Y)", "sup_y": " (sup. Y)"}


def _render_couche_row(dalle_id: int, sec_id: int, which: str, i: int, nc: int, disabled: bool):
    """Ligne du tableau des couches. i=1 : bouton '＋' (ajout de renfort) ;
    i>=2 : poubelle."""
    suffix = _FACE_LABELS.get(which, f" ({which})")

    type_key = KS(f"arm_type_{which}_c{i}", dalle_id, sec_id)
    treillis_key = KS(f"treillis_{which}_c{i}", dalle_id, sec_id)
    diam_key = KS(f"ø_barres_{which}_c{i}", dalle_id, sec_id)
    esp_key = KS(f"esp_barres_{which}_c{i}", dalle_id, sec_id)
    n_key = KS(f"n_barres_{which}_c{i}", dalle_id, sec_id)

    st.session_state.setdefault(type_key, "Treillis" if i == 1 else "n barres")
    st.session_state.setdefault(treillis_key, TR.TREILLIS_DEFAUT)
    st.session_state.setdefault(diam_key, 12)
    st.session_state.setdefault(esp_key, 150)
    st.session_state.setdefault(n_key, 3)

    typ = _couche_type(dalle_id, sec_id, which, i)

    c0, c1, c2, c3, c5, cG, c6 = st.columns(COUCHE_COLS, vertical_alignment="center")
    with c0:
        # Numéro seul (1, 2, 3…) : « Base / Renfort 1 » prenait trop de
        # place (retour bureau) — la couche 1 reste la nappe de base.
        st.markdown(f"{i}")
    with c1:
        st.selectbox(
            f"Type (couche {i}){suffix}",
            TYPES_ARMATURE,
            key=type_key,
            disabled=disabled,
            label_visibility="collapsed",
        )
    with c2:
        if typ == "Treillis":
            opts = TR.liste_choix()
            cur = str(st.session_state.get(treillis_key, TR.TREILLIS_DEFAUT))
            if cur not in opts:
                opts = opts + [cur]  # ancien fichier : la désignation reste utilisable
            st.selectbox(
                f"Treillis (couche {i}){suffix}",
                opts,
                key=treillis_key,
                disabled=disabled,
                label_visibility="collapsed",
            )
        else:
            st.selectbox(
                f"Ø (mm) (couche {i}){suffix}",
                DIAM_OPTS,
                key=diam_key,
                disabled=disabled,
                label_visibility="collapsed",
            )
    with c3:
        if typ == "Barres":
            st.number_input(
                f"Espacement (mm) (couche {i}){suffix}",
                min_value=25,
                max_value=500,
                step=25,
                key=esp_key,
                disabled=disabled,
                label_visibility="collapsed",
            )
        elif typ == "n barres":
            st.number_input(
                f"Nombre de barres (couche {i}){suffix}",
                min_value=1,
                max_value=50,
                step=1,
                key=n_key,
                disabled=disabled,
                label_visibility="collapsed",
            )
        else:
            st.markdown("<div style='text-align:center;opacity:0.5;'>—</div>", unsafe_allow_html=True)
    with c5:
        # Distance d'axe SAISISSABLE (retour bureau : « à quel niveau je
        # mets mes barres ») — suit l'automatique tant que rien n'est
        # saisi ; champ vidé = retour à l'automatique.
        st.text_input(
            f"Distance axe (cm) (couche {i}){suffix}",
            key=KS(f"dist_axe_{which}_c{i}", dalle_id, sec_id),
            disabled=disabled,
            label_visibility="collapsed",
        )
    with cG:
        # Champ CDG : uniquement sur la ligne de la dernière couche
        if i == nc:
            st.text_input(
                f"CDG {which} (cm)",
                key=KS(f"ycdg_{which}", dalle_id, sec_id),
                label_visibility="collapsed",
                disabled=disabled,
            )
    with c6:
        if i == 1:
            # Un gros « ＋ » seul : le libellé « Renfort » débordait sur
            # deux lignes (retour bureau) — l'infobulle dit ce qu'il fait.
            st.button(
                "＋",
                key=KS(f"btn_add_couche_{which}", dalle_id, sec_id),
                use_container_width=True,
                disabled=disabled or (nc >= MAX_COUCHES),
                help="Ajouter un renfort (treillis, barres à espacement ou n barres)",
                on_click=_add_couche,
                args=(dalle_id, sec_id, which),
            )
        else:
            st.button(
                "🗑️",
                key=KS(f"btn_del_couche_{which}_{i}", dalle_id, sec_id),
                use_container_width=True,
                disabled=disabled,
                help="Supprimer ce renfort",
                on_click=_delete_couche,
                args=(dalle_id, sec_id, which, i),
            )


def _render_couches_table(dalle_id: int, sec_id: int, which: str, disabled: bool):
    nc = _get_ncouches(dalle_id, sec_id, which)

    # Distances d'axe et CDG déjà synchronisés en tête de
    # render_dimensionnement_section — AVANT le calcul des états, pour
    # qu'une saisie se reflète dans les résultats du même rerun.

    h0, h1, h2, h3, h5, hG, h6 = st.columns(COUCHE_COLS, vertical_alignment="bottom")
    with h0:
        st.markdown("")
    with h1:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Type</div>", unsafe_allow_html=True,
                    help="« Barres » : Ø à espacement régulier (Ø12/150). "
                         "« n barres » : un nombre de barres posées dans la bande (3 Ø12).")
    with h2:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Treillis / Ø</div>", unsafe_allow_html=True)
    with h3:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Esp. / n</div>", unsafe_allow_html=True,
                    help="Espacement (mm) pour « Barres » ; nombre de barres pour « n barres ».")
    with h5:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Dist. axe (cm)</div>", unsafe_allow_html=True,
                    help="Distance parement -> axe de la couche, SAISISSABLE : par défaut "
                         "enrobage + demi-Ø arrondi au 0,5 cm sup. + jeu premier lit "
                         "(renfort dans le plan du treillis). Saisissez une autre valeur "
                         "pour poser la couche à un autre niveau — champ vidé = retour "
                         "à l'automatique.")
    with hG:
        st.markdown(
            "<div style='font-size:0.85em;font-weight:600;'>CDG (cm)</div>",
            unsafe_allow_html=True,
            help="Valeur calculée automatiquement (Σ As·e / Σ As). Si une valeur est "
                 "saisie, elle remplace le calcul automatique.",
        )
    with h6:
        st.markdown("")

    for i in range(1, nc + 1):
        _render_couche_row(dalle_id, sec_id, which, i, nc, disabled)


def _render_face_armatures(dalle_id: int, sec_id: int, dk: str, face: str,
                           states: dict, dim_locked: bool, units_as: str):
    """Bloc 'Armatures inférieures/supérieures — direction X/Y' complet.
    Toutes les valeurs viennent de states['dirs'][dk] — calcul par
    direction, formules inchangées."""
    which = f"{face}_{dk}"
    D = states["dirs"][dk]
    is_inf = (face == "inf")
    dir_lab = f"dir. {dk.upper()}" + (" (principale)" if states["principale"] == dk else " (secondaire)")
    titre = ("Armatures inférieures" if is_inf else "Armatures supérieures") + f" — {dir_lab}"
    unit_as_txt = "mm²" if units_as == "mm²" else "cm²"

    As_total = D["As_inf_total"] if is_inf else D["As_sup_total"]
    As_pm = D["As_inf_pm"] if is_inf else D["As_sup_pm"]
    detail = D["inf_detail"] if is_inf else D["sup_detail"]
    etat = D["etat_inf"] if is_inf else D["etat_sup"]
    As_req = D["As_req_inf_final"] if is_inf else D["As_req_sup_final"]
    As_min_eff = D["As_min_inf_eff"] if is_inf else D["As_min_sup_eff"]
    As_max = states["As_max"]
    geom_ok = D["geom_inf_ok"] if is_inf else D["geom_sup_ok"]

    As_disp = As_total if units_as == "mm²" else As_total / 100.0
    as_txt = f"{As_disp:.0f}" if units_as == "mm²" else f"{As_disp:.2f}"
    right = f"B{int(states['fyk'])} • {detail} ({as_txt} {unit_as_txt})"
    besoin_dim = max(As_req, As_min_eff)      # valeur dimensionnante
    pct_as = (besoin_dim / As_total * 100.0) if As_total > 0 else None

    open_bloc_left_right(titre, right, etat, pct=pct_as)

    # Infobulles pédagogiques : formule + valeurs numériques
    M_face = D["M_inf_val"] if is_inf else D["M_sup_val"]
    d_face = D["d_utile_inf"] if is_inf else D["d_utile_sup"]
    As_req_opp = D["As_formule_sup"] if is_inf else D["As_formule_inf"]
    fyk = states["fyk"]; gs = states["gamma_s"]; fyd = states["fyd"]
    b_cm = states["b"]

    help_req = (
        "**Aₛ,req = M / (fyd · 0,9 · d)**\n\n"
        f"M = {_fr(M_face, 1)} kN·m\n\n"
        f"fyd = {_fr(fyk, 0)} / {_fr(gs, 2)} = {_fr(fyd, 0)} N/mm²\n\n"
        f"d = {_fr(d_face, 1)} cm\n\n"
        f"→ Aₛ,req = {_fr(As_req, 0)} mm²"
    )
    help_min = (
        "**Aₛ,min = max( 0,26·fctm/fyk·b·h ; 0,0013·b·h ; 0,25·Aₛ,req face opposée, même direction )**\n\n"
        f"0,26 · {_fr(states['fctm'], 1)} / {_fr(fyk, 0)} · b · h = {_fr(states['As_min_ec'], 0)} mm²\n\n"
        f"0,0013 · b · h = {_fr(states['As_min_plancher'], 0)} mm²\n\n"
        f"0,25 · {_fr(As_req_opp, 0)} = {_fr(0.25 * As_req_opp, 0)} mm²\n\n"
        f"→ Aₛ,min = {_fr(As_min_eff, 0)} mm²"
    )
    help_max = (
        "**Aₛ,max = 0,04 · b · h**\n\n"
        f"0,04 · {_fr(states['b'] * 10, 0)} · {_fr(states['h'] * 10, 0)} = {_fr(As_max, 0)} mm²"
    )
    help_fourni = (
        "**Aₛ fourni = Σ couches × largeur de bande**\n\n"
        f"{detail}\n\n"
        f"Σ = {_fr(As_pm, 0)} mm²/m × {_fr(b_cm / 100.0, 2)} m = {_fr(As_total, 0)} mm²"
    )

    ca1, ca2, ca3, ca4 = st.columns(4)
    with ca1:
        st.markdown(f"**Aₛ,req = {As_req:.0f} mm²**", help=help_req)
    with ca2:
        st.markdown(f"**Aₛ,min = {As_min_eff:.0f} mm²**", help=help_min)
    with ca3:
        st.markdown(f"**Aₛ,max = {As_max:.0f} mm²**", help=help_max)
    with ca4:
        st.markdown(f"**Aₛ fourni = {As_pm:.0f} mm²/m**", help=help_fourni)

    if not geom_ok:
        st.markdown("❌ **Position des couches incompatible avec l'épaisseur : d utile ≤ 0.**")

    # ---- Tableau des couches : Base / Renforts ----
    _render_couches_table(dalle_id, sec_id, which, disabled=dim_locked)
    close_bloc()


# ============================================================
#  UI : DIMENSIONNEMENT D'UNE SECTION
# ============================================================
def _render_hauteur_details(states: dict, h: float):
    """
    Vérification de la hauteur en DEUX LIGNES (v2.0) : la formule de
    hᵤ,min (M_max = max des quatre moments), puis la vérification
    hᵤ,min + d₁ ≤ h sur une seule ligne. Formules Poutre inchangées —
    seule la présentation est compactée.
    """
    M_max = states["M_max"]
    hmin_calc = states["hmin_calc"]
    h_min_dalle = states["h_min_dalle"]
    ok_h = states["etat_h"] == "ok"
    if M_max > 0:
        m_txt = _fr(M_max, 0) if abs(M_max - round(M_max)) < 1e-9 else _fr(M_max, 1)
        st.markdown(
            f"hᵤ,min = √( {m_txt}·10⁶ / ({_fr(states['alpha_b'], 2)} · "
            f"{_fr(states['b'] * 10, 0)} · {_fr(states['mu_val'], 4)}) ) = "
            f"**{_fr(hmin_calc, 1)} cm**"
        )
    else:
        st.markdown("hᵤ,min = **0,0 cm** — aucun moment appliqué")
    st.markdown(
        f"hᵤ,min + d₁ = {_fr(hmin_calc, 1)} + {_fr(states['e_cdg_gov'], 1)} = "
        f"**{_fr(h_min_dalle, 1)} cm** {'≤' if ok_h else '>'} "
        f"h = **{_fr(h, 0)} cm** {'✅' if ok_h else '❌'}"
    )


def render_dimensionnement_section(dalle_id: int, sec_id: int, beton_data: dict):
    dalle_locked = bool(st.session_state.get(KD("lock_data", dalle_id), False))
    dalle = next(d for d in st.session_state.predalles if int(d.get("id")) == dalle_id)
    sec = next(s for s in dalle["sections"] if int(s.get("id")) == sec_id)
    sec_nom = str(st.session_state.get(f"meta_pre{dalle_id}_nom_{sec_id}", sec.get("nom", f"Section {sec_id}")))

    # Synchroniser les distances d'axe par couche PUIS le CDG de chaque
    # face AVANT le calcul des états : une saisie (niveau d'une couche,
    # CDG) se reflète ainsi dans les résultats du MÊME rerun — et les
    # widgets correspondants ne sont pas encore instanciés à ce stade.
    for which in FACES_DIR:
        for i in range(1, _get_ncouches(dalle_id, sec_id, which) + 1):
            _sync_dist_state(dalle_id, sec_id, which, i)
        _sync_ycdg_state(dalle_id, sec_id, which)

    states = _dimensionnement_compute_states(dalle_id, sec_id, beton_data)

    sec_label = sec_nom if sec_nom.lower().startswith("section") else f"Section {sec_nom}"
    title = _status_icon_label(states["etat_global"], sec_label)

    # NB : expanded=True pour tous — le libellé contient l'icône d'état,
    # et Streamlit remet un expander à son état par défaut dès que son
    # libellé change (voir poutre.py).
    with st.expander(title, expanded=True):
        dim_locked = dalle_locked

        units_len = st.session_state.get("units_len", "cm")
        units_as = st.session_state.get("units_as", "mm²")

        beton = states["beton"]
        b = states["b"]
        h = states["h"]
        fyd = states["fyd"]
        V_val = states["V_val"]

        # ---- Vérification de la hauteur (avec détail, comme le PDF) ----
        if units_len == "mm":
            right_h = f"{beton} • Section {b*10:.0f}×{h*10:.0f} mm"
        else:
            right_h = f"{beton} • Section {b:.0f}×{h:.0f} cm"
        h_min_dalle = states["h_min_dalle"]
        pct_h = (h_min_dalle / h * 100.0) if h > 0 else None
        open_bloc_left_right("Vérification de la hauteur", right_h, states["etat_h"], pct=pct_h)
        _render_hauteur_details(states, h)
        close_bloc()

        # ---- Armatures : quatre familles, direction PRINCIPALE d'abord ----
        ordre_dirs = (states["principale"], "y" if states["principale"] == "x" else "x")
        for dk in ordre_dirs:
            _render_face_armatures(dalle_id, sec_id, dk, "inf", states, dim_locked, units_as)
            _render_face_armatures(dalle_id, sec_id, dk, "sup", states, dim_locked, units_as)

        # ---- Effort tranchant (v2.1) : vérification de la contrainte
        #      tangentielle uniquement — une dalle ne reçoit pas d'étriers ----
        if V_val > 0:
            tau = states["tau"]
            tau_adm = states["tau_adm"]
            ok_tau = states["etat_tau"] == "ok"
            pct_tau = (tau / tau_adm * 100.0) if tau_adm > 0 else None
            open_bloc_left_right("Vérification de l'effort tranchant", "",
                                 states["etat_tau"], pct=pct_tau)
            st.markdown(
                f"τ = {tau:.2f} N/mm² {'≤' if ok_tau else '>'} "
                f"τ_adm,I = {tau_adm:.2f} N/mm² {'✅' if ok_tau else '❌'}",
                help="**τ = V / (0,75 · b · h)** — dalle sans armatures "
                     "d'effort tranchant : la contrainte tangentielle doit "
                     "rester sous τ_adm,I (seuil « pas besoin d'étriers »).",
            )
            close_bloc()


# ============================================================
#  UI : INFOS PROJET / PARAMÈTRES AVANCÉS
# ============================================================
def _valeurs_pour_export() -> dict:
    """Traduit l'état prédalle (« pre{id}_ ») vers l'espace de clés
    « dal{id}_ » qu'attend modules/export_pdf_dalle — la note réutilise
    la mise en page Dalle à l'identique. Les distances d'axe EFFECTIVES
    de chaque couche sont FIGÉES dans les clés exportées : la règle
    prédalle (_auto_dist_couche) vit dans ce module, l'export n'en
    connaît rien. La clé h_pre passe aussi (trait de clivage et lignes
    DIMENSIONS de la coupe)."""
    out = {}
    for k in list(st.session_state.keys()):
        if _is_transient_key(k):
            continue
        v = st.session_state[k]
        if k.startswith("meta_predalle_nom_"):
            out["meta_dalle_nom_" + k[len("meta_predalle_nom_"):]] = v
        elif re.match(r"^meta_pre\d+_nom_", k):
            out["meta_dal" + k[len("meta_pre"):]] = v
        elif re.match(r"^pre\d+_", k):
            out["dal" + k[3:]] = v
        elif not re.match(r"^(dal|b)\d+_", k) and not k.startswith("meta_"):
            out[k] = v  # clés globales (gamma_s, jeu, unités, infos projet…)
    # positions réelles figées, couche par couche
    for d in st.session_state.predalles:
        did = int(d.get("id"))
        for s in d.get("sections", []):
            sid = int(s.get("id"))
            for which in FACES_DIR:
                for i in range(1, _get_ncouches(did, sid, which) + 1):
                    e = _dist_couche_eff(did, sid, which, i)
                    out[f"dal{did}_sec{sid}_dist_axe_{which}_c{i}"] = f"{e:.2f}".replace(".", ",")
                    out[f"dal{did}_sec{sid}_dist_auto_{which}_c{i}"] = False
    return out


def _pdf_filename() -> str:
    """Nom automatique du rapport : AAA_NDC Partie#Indice_Date.pdf."""
    nom = str(st.session_state.get("nom_projet", "") or "")
    aaa = re.sub(r"[^A-Za-z0-9]", "", nom).upper()[:3] or "PRJ"
    partie = str(st.session_state.get("partie", "") or "").strip()
    indice = str(st.session_state.get("indice", "0") or "0").strip() or "0"
    date = str(st.session_state.get("date", "") or datetime.today().strftime("%d/%m/%Y")).strip().replace("/", "-")
    core = f"NDC {partie}".strip()
    name = f"{aaa}_{core}#{indice}_{date}.pdf"
    return re.sub(r'[\\/:*?"<>|]+', "-", name)


def _toggle_param_avances():
    st.session_state["predalle_show_param_avances"] = not bool(st.session_state.get("predalle_show_param_avances", False))


def _toggle_infos_projet():
    st.session_state["chk_infos_projet"] = not bool(st.session_state.get("chk_infos_projet", False))


def render_infos_projet():
    st.session_state.setdefault("chk_infos_projet", False)
    st.session_state.setdefault("nom_projet", "")
    st.session_state.setdefault("partie", "")
    st.session_state.setdefault("date", datetime.today().strftime("%d/%m/%Y"))
    st.session_state.setdefault("indice", "0")

    shown = bool(st.session_state.get("chk_infos_projet", False))
    cT, cBtn = st.columns([6, 0.6], vertical_alignment="center")
    with cT:
        st.markdown("### Informations sur le projet")
    with cBtn:
        st.button(
            "➖" if shown else "➕",
            key="predalle_btn_toggle_infos_projet",
            help="Masquer les informations du projet" if shown else "Ajouter les informations du projet",
            use_container_width=True,
            on_click=_toggle_infos_projet,
        )

    if bool(st.session_state.get("chk_infos_projet", False)):
        with st.container(border=True):
            st.text_input("Nom du projet", placeholder="Nom du projet", key="nom_projet", label_visibility="collapsed")
            st.text_input("Partie", placeholder="Partie", key="partie", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Date", placeholder="Date (jj/mm/aaaa)", key="date", label_visibility="collapsed")
            with c2:
                st.text_input("Indice", placeholder="Indice", key="indice", label_visibility="collapsed")


def render_parametres_avances():
    """Paramètres avancés, 3 colonnes (partagés avec le module Poutre)."""
    _ensure_global_defaults()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Affichage**")
        st.selectbox("Unité de longueur", ["cm", "mm"], key="units_len")
        st.selectbox("Unité d'armature", ["mm²", "cm²"], key="units_as")

    with c2:
        st.markdown("**Coefficients matériaux**")
        st.number_input(
            "Coefficient acier ELS",
            min_value=1.0,
            max_value=2.0,
            step=0.05,
            format="%.2f",
            key="gamma_s",
            help="fyd = fyk / coefficient — défaut 1,5 (méthode ancienne). "
                 "Ex. acier 500 : 500/1,5 = 333 MPa ; avec 1,15 : 435 MPa.",
        )

    with c3:
        st.markdown("**Jeux d'armatures**")
        st.number_input("Jeu premier lit (cm)", min_value=0.0, step=0.5, key="jeu_enrobage_cm",
                        help="Compté dans la distance d'axe des couches : "
                             "enrobage + demi-Ø arrondi au 0,5 cm sup. + jeu.")


# ============================================================
#  UI : COLONNES GAUCHE / DROITE
# ============================================================
def render_donnees_left(beton_data: dict):
    st.markdown("### Données")
    for d in st.session_state.predalles:
        did = int(d["id"])
        d["nom"] = str(st.session_state.get(f"meta_predalle_nom_{did}", d.get("nom", f"Prédalle {did}")))
        render_caracteristiques_dalle(did)


def render_dimensionnement_right(beton_data: dict):
    for d in st.session_state.predalles:
        did = int(d["id"])
        dnom = str(st.session_state.get(f"meta_predalle_nom_{did}", d.get("nom", f"Prédalle {did}")))
        d["nom"] = dnom

        sec_states = [
            _dimensionnement_compute_states(did, int(s["id"]), beton_data)["etat_global"]
            for s in d.get("sections", [])
        ]
        dalle_state = _status_merge(*sec_states) if sec_states else "ok"
        lock_icon = "🔒 " if bool(st.session_state.get(KD("lock_data", did), False)) else ""
        dalle_label = _status_icon_label(dalle_state, f"{lock_icon}{dnom}")

        # expanded=True : libellé dynamique (icône d'état) -> Streamlit
        # réinitialise l'expander à chaque changement de libellé.
        with st.expander(dalle_label, expanded=True):
            for s in d.get("sections", []):
                render_dimensionnement_section(did, int(s["id"]), beton_data)


# ============================================================
#  PAGE
# ============================================================
def show():
    # ---------- Données béton : chargées AVANT l'init ----------
    global BETON_DATA
    try:
        with open("beton_classes.json", "r", encoding="utf-8") as f:
            BETON_DATA = json.load(f)
    except Exception:
        st.error("Impossible de charger beton_classes.json — vérifie que le fichier est présent et valide.")
        st.stop()
    beton_data = BETON_DATA

    _ensure_global_defaults()
    _init_dalles_if_needed()

    # FIX PERSISTANCE : épingler toutes les clés persistantes AVANT tout rendu.
    _pin_persistent_state()

    # FIX BUG RECALCUL : synchroniser les saisies décimales FR (*_raw)
    # vers leurs valeurs numériques AVANT tout calcul.
    _sync_float_raw_keys()

    if "retour_accueil_demande" not in st.session_state:
        st.session_state.retour_accueil_demande = False

    if st.session_state.retour_accueil_demande:
        st.session_state.page = "Accueil"
        st.session_state.retour_accueil_demande = False
        st.rerun()

    tH1, tH2, tH3 = st.columns([8, 1.6, 0.55], vertical_alignment="center")
    with tH1:
        st.markdown("## Prédalle en béton armé")
    with tH2:
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {PREDALLE_VERSION}</div>",
            unsafe_allow_html=True,
        )
    with tH3:
        st.button("❔", key="predalle_btn_version_hist", help="Version actuelle du module Prédalle.",
                  use_container_width=True)

    btn1, btn2, btn3, btn4, btn5 = st.columns(5)

    with btn1:
        if st.button("🏠 Accueil", use_container_width=True, key="predalle_btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()

    with btn2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="predalle_btn_reset"):
            _reset_module()

    with btn3:
        payload = _build_save_payload()
        st.download_button(
            label="💾 Enregistrer",
            data=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="predalle_ba.json",
            mime="application/json",
            use_container_width=True,
            key="predalle_btn_save_dl",
        )

    with btn4:
        if st.button("📂 Ouvrir", use_container_width=True, key="predalle_btn_open_toggle"):
            st.session_state["predalle_show_open_uploader"] = not st.session_state.get("predalle_show_open_uploader", False)

        if st.session_state.get("predalle_show_open_uploader", False):
            uploaded = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed", key="predalle_open_uploader")
            if uploaded is not None:
                try:
                    data = json.load(uploaded)
                    if not isinstance(data, dict):
                        raise ValueError("Structure JSON inattendue")
                    _load_from_payload(data)
                    st.session_state["predalle_show_open_uploader"] = False
                    st.rerun()
                except Exception:
                    st.error("Fichier invalide ou corrompu — chargement annulé.")

    with btn5:
        if st.button("📄 Générer PDF", use_container_width=True, key="predalle_btn_pdf"):
            from modules.export_pdf_dalle import generer_rapport_pdf

            infos = {
                "nom_projet": st.session_state.get("nom_projet", ""),
                "partie": st.session_state.get("partie", ""),
                "date": st.session_state.get("date", datetime.today().strftime("%d/%m/%Y")),
                "indice": st.session_state.get("indice", "0"),
            }

            try:
                fichier_pdf = generer_rapport_pdf(
                    dalles=st.session_state.predalles,
                    values=_valeurs_pour_export(),
                    beton_data=beton_data,
                    infos=infos,
                )
                with open(fichier_pdf, "rb") as f:
                    st.session_state["predalle_pdf_bytes"] = f.read()
                st.success("✅ Note de calcul générée")
            except Exception as e:
                st.session_state.pop("predalle_pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")

        if st.session_state.get("predalle_pdf_bytes"):
            st.download_button(
                label="⬇️ Télécharger le rapport PDF",
                data=st.session_state["predalle_pdf_bytes"],
                file_name=_pdf_filename(),
                mime="application/pdf",
                use_container_width=True,
                key="predalle_btn_pdf_dl",
            )

    input_col_gauche, result_col_droite = st.columns([2, 3])

    with input_col_gauche:
        render_infos_projet()
        render_donnees_left(beton_data)

    with result_col_droite:
        st.session_state.setdefault("predalle_show_param_avances", False)

        cH1, cH2 = st.columns([18, 1.3], vertical_alignment="center")
        with cH1:
            st.markdown("### Dimensionnement")
        with cH2:
            st.button(
                "⚙️",
                key="predalle_btn_toggle_param_avances",
                help="Paramètres avancés",
                use_container_width=True,
                on_click=_toggle_param_avances,
            )

        if bool(st.session_state.get("predalle_show_param_avances", False)):
            with st.container(border=True):
                render_parametres_avances()

        render_dimensionnement_right(beton_data)

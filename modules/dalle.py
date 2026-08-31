# -*- coding: utf-8 -*-
# ===========================
#  DALLE EN BÉTON ARMÉ — VERSION 2.0
# ===========================
#  dalle.py (Streamlit)
#
#  Évolutions v2.0 (dalle BIDIRECTIONNELLE) :
#   1. SOLLICITATIONS : 5 valeurs par section — Mx inf, Mx sup, My inf,
#      My sup (kN·m par bande) et V max (kN, convention inchangée : un
#      seul effort tranchant par section, τ = V/(0,75·b·h)).
#   2. ARMATURES DANS LES DEUX DIRECTIONS : chaque direction (X, Y)
#      porte ses faces inf./sup. avec la MÊME machinerie de couches
#      (treillis / barres / renforts) — le suffixe de face `which`
#      devient "inf_x" / "sup_x" / "inf_y" / "sup_y". AUCUNE formule
#      modifiée : les quatre familles sont calculées indépendamment par
#      les expressions Poutre existantes (Aₛ,req, Aₛ,min avec 0,25·Aₛ,req
#      de la face opposée DE LA MÊME DIRECTION, Aₛ,max).
#   3. DIRECTION PRINCIPALE = celle dont le moment maximal est le plus
#      grand (X à égalité) — déterminée des sollicitations, affichée
#      dans les cartes ; l'ordre d'affichage suit (principale d'abord).
#   4. HAUTEUR : hᵤ,min inchangée, avec M_max = max des QUATRE moments ;
#      d₁ = enrobage mécanique de la famille dimensionnante. Affichage
#      compacté à deux lignes.
#   5. MIGRATION AUTOMATIQUE des données v1 (fichiers JSON et sessions) :
#      M_inf→Mx_inf, M_sup→Mx_sup, couches inf/sup → direction X ; la
#      direction Y démarre sur ses défauts. Idempotente.
#
#  Module construit comme une COPIE ADAPTÉE de poutre.py v2.40 :
#  même organisation de page, mêmes cartes, mêmes bandeaux verts/rouges,
#  même logique de vérification, même système de sections (ajout /
#  copie / suppression / verrouillage), même sauvegarde / ouverture
#  JSON, même génération PDF. Les FORMULES DE CALCUL sont strictement
#  celles du module Poutre (hᵤ,min, Aₛ,req, Aₛ,min, Aₛ,max, τ, pas des
#  étriers) appliquées à une bande de dalle de largeur b (100 cm par
#  défaut) et d'épaisseur h (20 cm par défaut).
#
#  Différences assumées avec Poutre (adaptées à une bande de dalle) :
#   1. ARMATURES PAR COUCHES au lieu de lits de barres comptées :
#      chaque face (inf. / sup.) porte une couche de BASE + des
#      RENFORTS (bouton "＋ Renfort"), chacun étant au choix :
#        - un TREILLIS (nomenclature Øl/Øt/el/et, base des treillis
#          courants dans modules/treillis.py, facilement extensible) ;
#        - des BARRES (Ø + espacement, ex. Ø12/150).
#      La section d'acier est calculée automatiquement en mm²/m puis
#      rapportée à la largeur réelle de la bande ; les sections des
#      couches s'ADDITIONNENT ("As fourni ≥ As requis", logique Poutre).
#   2. POSITION DES COUCHES : dans une dalle, les renforts sont posés
#      dans le plan du treillis (pas d'empilement en lits). Distance
#      d'axe d'une couche = enrobage + demi-Ø arrondi au 0,5 cm sup.
#      + jeu premier lit (paramètre avancé partagé avec Poutre). Le
#      Ø d'étrier n'intervient plus (pas d'étrier enveloppant les
#      armatures de flexion dans une dalle). Le CDG pondéré
#      (Σ As·e / Σ As) reste modifiable par face (champ CDG), comme
#      dans Poutre.
#   3. EFFORT TRANCHANT : moteur strictement identique à Poutre
#      (τ = V / (0,75·b·h), seuils τ_adm depuis beton_classes.json,
#      pas théorique Aₛₜ·fyd·d / V). Les lignes d'armatures n'ont plus
#      de position "de barre X à barre Y" (sans objet sur une dalle).
#   4. SURÉPAISSEURS : non couvertes par cette version. L'architecture
#      par sections (une section = une bande rectangulaire b×h avec ses
#      couches) est prête à accueillir des sections d'épaisseur
#      différente dans une étape ultérieure.
#
#  ESPACE DE NOMS DE SESSION : préfixes "dal{id}_" / "meta_dalle_" /
#  "meta_dal{id}_", volontairement DISJOINTS des clés "b{id}_" /
#  "meta_beam_" / "meta_b{id}_" du module Poutre — aucun échange d'état
#  entre les deux modules (les paramètres globaux gamma_s, jeux,
#  unités et infos projet restent partagés, comme un même logiciel).
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

DALLE_VERSION = "2.0"  # version affichée dans l'en-tête de l'application

# Directions d'une dalle : clé interne + libellé. Les faces deviennent
# "inf_x" / "sup_x" / "inf_y" / "sup_y" — suffixe opaque pour toute la
# machinerie des couches (aucune formule ne lit la direction).
DIR_KEYS = ("x", "y")
FACES_DIR = ("inf_x", "sup_x", "inf_y", "sup_y")

# Largeurs de colonnes du tableau des couches
# (Couche | Type | Treillis/Ø | Esp. | As/m | Dist. axe | CDG | Action)
COUCHE_COLS = [0.85, 1.45, 1.9, 1.05, 1.05, 1.05, 1.05, 1.0]


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
    return f"dal{dalle_id}_{base}"


def KS(base: str, dalle_id: int, sec_id: int) -> str:
    return f"dal{dalle_id}_sec{sec_id}_{base}"


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
    return bool(re.match(r"^dal\d+_", k)) or k.startswith("meta_dalle_nom_") or \
        (k.startswith("meta_dal") and "_nom_" in k)


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
#  RESET (limité aux données du module Dalle)
# ============================================================
def _reset_module():
    """Réinitialise UNIQUEMENT les données du module Dalle : les données
    du module Poutre et les paramètres globaux partagés sont conservés."""
    for k in list(st.session_state.keys()):
        if _is_dalle_key(k) or (k.endswith("_raw") and _is_dalle_key(k[:-4])) or k.startswith("dalle_"):
            del st.session_state[k]
    st.session_state.pop("dalles", None)
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
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
    used = set()
    for s in dalle.get("sections", []):
        sid = int(s.get("id"))
        used.add(str(st.session_state.get(f"meta_dal{dalle_id}_nom_{sid}", s.get("nom", ""))).strip())
        used.add(str(s.get("nom", "")).strip())
    for L in _letter_sequence():
        if L not in used:
            return L
    return f"S{len(dalle.get('sections', [])) + 1}"  # repli improbable


# ============================================================
#  DALLES / SECTIONS : INIT / ADD / DELETE / DUPLICATE / COPY
# ============================================================
def _init_dalles_if_needed():
    if "dalles" not in st.session_state or not isinstance(st.session_state.dalles, list) or len(st.session_state.dalles) == 0:
        st.session_state.dalles = [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]}]

    for d in st.session_state.dalles:
        d["id"] = int(d.get("id", 0))
        d["nom"] = str(d.get("nom", f"Dalle {d['id']}"))
        if "sections" not in d or not isinstance(d["sections"], list) or len(d["sections"]) == 0:
            d["sections"] = [{"id": 1, "nom": "A"}]
        for s in d["sections"]:
            s["id"] = int(s.get("id", 0))
            s["nom"] = str(s.get("nom", f"Section {s['id']}"))

    if not any(int(d.get("id", 0)) == 1 for d in st.session_state.dalles):
        st.session_state.dalles.insert(0, {"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]})

    for d in st.session_state.dalles:
        if not any(int(s.get("id", 0)) == 1 for s in d["sections"]):
            d["sections"].insert(0, {"id": 1, "nom": "A"})

    # Synchronisation des noms (labels d'expander à jour immédiatement)
    for d in st.session_state.dalles:
        did = int(d["id"])
        key_nom = f"meta_dalle_nom_{did}"
        if key_nom not in st.session_state:
            st.session_state[key_nom] = str(d.get("nom", f"Dalle {did}"))
        d["nom"] = str(st.session_state.get(key_nom, d.get("nom")))

        for s in d.get("sections", []):
            sid = int(s["id"])
            key_snom = f"meta_dal{did}_nom_{sid}"
            if key_snom not in st.session_state:
                st.session_state[key_snom] = str(s.get("nom", "A"))
            s["nom"] = str(st.session_state.get(key_snom, s.get("nom", "A")))

    for d in st.session_state.dalles:
        _ensure_defaults_for_dalle(int(d["id"]))


def _next_dalle_id() -> int:
    ids = [int(d.get("id", 0)) for d in st.session_state.dalles]
    return (max(ids) + 1) if ids else 1


def _next_section_id(dalle_id: int) -> int:
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
    ids = [int(s.get("id", 0)) for s in dalle["sections"]]
    return (max(ids) + 1) if ids else 1


DIAM_OPTS = [6, 8, 10, 12, 16, 20, 25, 32, 40]
SHEAR_DIAM_OPTS = [6, 8, 10, 12]
TYPES_ARMATURE = ["Treillis", "Barres"]


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


def _ensure_defaults_for_dalle(dalle_id: int):
    # Dalle : bande de 100 cm de large, 20 cm d'épaisseur par défaut
    st.session_state.setdefault(KD("b", dalle_id), 100)
    st.session_state.setdefault(KD("h", dalle_id), 20)
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

    # Sections
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
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
                # renforts : barres Ø12/150 par défaut.
                st.session_state.setdefault(
                    KS(f"arm_type_{which}_c{i}", dalle_id, sid),
                    "Treillis" if i == 1 else "Barres",
                )
                _coerce_str_choice(KS(f"arm_type_{which}_c{i}", dalle_id, sid), TYPES_ARMATURE,
                                   "Treillis" if i == 1 else "Barres")
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

        # Cisaillement : mêmes clés que Poutre, sans positions de barres
        st.session_state.setdefault(KS("shear_n_lines", dalle_id, sid), 1)
        st.session_state.setdefault(KS("shear_pas", dalle_id, sid), 30.0)
        n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", dalle_id, sid), 1) or 1))
        st.session_state[KS("shear_n_lines", dalle_id, sid)] = n_lines
        for i in range(n_lines):
            st.session_state.setdefault(KS(f"shear_line{i}_type", dalle_id, sid), "Étrier")
            st.session_state.setdefault(KS(f"shear_line{i}_d", dalle_id, sid), 10)
            _coerce_int_choice(KS(f"shear_line{i}_d", dalle_id, sid), SHEAR_DIAM_OPTS, 10)
            _tk = KS(f"shear_line{i}_type", dalle_id, sid)
            if str(st.session_state.get(_tk, "")) not in ("Étrier", "Épingle"):
                st.session_state[_tk] = "Étrier"


def _delete_dalle(dalle_id: int):
    if dalle_id == 1:
        return
    st.session_state.dalles = [d for d in st.session_state.dalles if int(d.get("id")) != dalle_id]
    prefix = f"dal{dalle_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_dalle_nom_{dalle_id}", None)
    for k in list(st.session_state.keys()):
        if k.startswith(f"meta_dal{dalle_id}_nom_"):
            del st.session_state[k]


def _duplicate_dalle(src_dalle_id: int):
    """Bouton 📋 'Copier la dalle' (toutes les données)."""
    src = next(d for d in st.session_state.dalles if int(d.get("id")) == src_dalle_id)
    new_id = _next_dalle_id()
    st.session_state.dalles.append({"id": new_id, "nom": f"{src.get('nom','Dalle')} (copie)", "sections": deepcopy(src["sections"])})

    src_prefix = f"dal{src_dalle_id}_"
    dst_prefix = f"dal{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_dalle_nom_{new_id}"] = f"{st.session_state.get(f'meta_dalle_nom_{src_dalle_id}', src.get('nom','Dalle'))} (copie)"
    for s in src.get("sections", []):
        sid = int(s.get("id"))
        st.session_state[f"meta_dal{new_id}_nom_{sid}"] = st.session_state.get(f"meta_dal{src_dalle_id}_nom_{sid}", s.get("nom", f"Section {sid}"))

    _ensure_defaults_for_dalle(new_id)


def _add_section(dalle_id: int):
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
    new_id = _next_section_id(dalle_id)
    name = _next_section_name(dalle_id)
    dalle["sections"].append({"id": new_id, "nom": name})
    st.session_state[f"meta_dal{dalle_id}_nom_{new_id}"] = name
    _ensure_defaults_for_dalle(dalle_id)


def _copy_section(dalle_id: int, src_sec_id: int):
    """Copie intégrale d'une section (sollicitations, couches inf./sup.,
    cisaillement) vers une nouvelle section nommée avec la première
    lettre disponible. Callback on_click."""
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
    new_id = _next_section_id(dalle_id)
    name = _next_section_name(dalle_id)
    dalle["sections"].append({"id": new_id, "nom": name})

    src_prefix = f"dal{dalle_id}_sec{src_sec_id}_"
    dst_prefix = f"dal{dalle_id}_sec{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_dal{dalle_id}_nom_{new_id}"] = name
    _ensure_defaults_for_dalle(dalle_id)


def _delete_section(dalle_id: int, sec_id: int):
    if sec_id == 1:
        return
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
    dalle["sections"] = [s for s in dalle["sections"] if int(s.get("id")) != sec_id]
    prefix = f"dal{dalle_id}_sec{sec_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_dal{dalle_id}_nom_{sec_id}", None)


# ============================================================
#  SAVE / LOAD JSON (dalles + valeurs)
# ============================================================
def _build_save_payload():
    dalles = []
    for d in st.session_state.dalles:
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

    return {"version": "dalle-2.0", "dalles": dalles, "values": values}


def _load_from_payload(payload: dict):
    dalles = payload.get("dalles", None)
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
            cleaned.append({"id": did, "nom": str(d.get("nom", f"Dalle {did}")), "sections": cleaned_secs})
        st.session_state.dalles = cleaned if cleaned else [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]}]
    else:
        st.session_state.dalles = [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]}]

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


def _brins_from_type(type_txt: str) -> int:
    t = str(type_txt)
    if "3 brins" in t:
        return 3
    if "pingle" in t or "1 brin" in t:
        return 1
    return 2


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
    Données d'une couche : (type, désignation treillis, Ø barres, esp barres).
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
    return typ, des, d, esp


def _couche_as_per_m(dalle_id: int, sec_id: int, which: str, i: int) -> float:
    """Section d'acier de la couche i (mm²/m), calculée automatiquement."""
    typ, des, d, esp = _couche_data(dalle_id, sec_id, which, i)
    if typ == "Treillis":
        return TR.as_treillis_mm2_m(des)
    return TR.as_barres_mm2_m(d, esp)


def _couche_diam_mm(dalle_id: int, sec_id: int, which: str, i: int) -> float:
    """Ø des fils/barres de la couche (mm) — sert à la distance d'axe."""
    typ, des, d, esp = _couche_data(dalle_id, sec_id, which, i)
    if typ == "Treillis":
        t = TR.parse_designation(des)
        return float(t[0]) if t else 10.0
    return float(d)


def _couche_label(dalle_id: int, sec_id: int, which: str, i: int) -> str:
    """Libellé compact : 'Treillis 10/10/100/100' ou 'Ø12/150'."""
    typ, des, d, esp = _couche_data(dalle_id, sec_id, which, i)
    if typ == "Treillis":
        return f"Treillis {des}"
    esp_txt = f"{esp:.0f}" if abs(esp - round(esp)) < 1e-9 else _fr(esp, 1)
    return f"Ø{d}/{esp_txt}"


def _auto_dist_couche(dalle_id: int, sec_id: int, which: str, i: int) -> float:
    """
    Distance d'axe automatique de la couche i (cm) :
      enrobage béton
      + demi-Ø de la couche arrondi au 0,5 cm sup.
      + jeu premier lit (paramètre avancé, partagé avec Poutre)
    Toutes les couches d'une face sont posées dans le même plan (les
    renforts d'une dalle se placent dans le plan du treillis — pas
    d'empilement en lits comme dans une poutre).
    Ex (treillis Ø10) : 3,0 + arr(0,5)=0,5 + 1,0 = 4,5 cm.
    """
    enrob_beton = float(st.session_state.get(KD("enrobage_beton", dalle_id), 3.0) or 3.0)
    jeu1 = float(st.session_state.get("jeu_enrobage_cm", 1.0) or 0.0)
    d = _couche_diam_mm(dalle_id, sec_id, which, i)
    return enrob_beton + _round_up_to_half_cm(d / 20.0) + jeu1


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
        e = _auto_dist_couche(dalle_id, sec_id, which, i)
        As_pm_tot += As_pm
        somme_As_e += As_pm * e
        parts.append(_couche_label(dalle_id, sec_id, which, i))

    e_cdg = (somme_As_e / As_pm_tot) if As_pm_tot > 0 else _auto_dist_couche(dalle_id, sec_id, which, 1)
    if use_manual:
        man = _ycdg_manual(dalle_id, sec_id, which)
        if man is not None:
            e_cdg = man
    As_tot = As_pm_tot * (b / 100.0)
    return As_tot, e_cdg, " + ".join(parts), As_pm_tot


# ============================================================
#  CISAILLEMENT : aires, résumé, callbacks (1 ligne = 1 étrier)
# ============================================================
def _shear_lines_total_Ast_mm2(dalle_id: int, sec_id: int) -> float:
    n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", dalle_id, sec_id), 1) or 1))
    Ast = 0.0
    for i in range(n_lines):
        typ = str(st.session_state.get(KS(f"shear_line{i}_type", dalle_id, sec_id), "Étrier"))
        diam = float(st.session_state.get(KS(f"shear_line{i}_d", dalle_id, sec_id), 8) or 8)
        Ast += _brins_from_type(typ) * _bar_area_mm2(diam)
    return Ast


def _shear_lines_summary(dalle_id: int, sec_id: int) -> str:
    n_lines = max(1, int(st.session_state.get(KS("shear_n_lines", dalle_id, sec_id), 1) or 1))
    order = []
    counts = {}
    for i in range(n_lines):
        typ = str(st.session_state.get(KS(f"shear_line{i}_type", dalle_id, sec_id), "Étrier"))
        diam = int(float(st.session_state.get(KS(f"shear_line{i}_d", dalle_id, sec_id), 8) or 8))
        base = "Épingle" if _brins_from_type(typ) == 1 else "Étrier"
        key = (base, diam)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    parts = []
    for (base, diam) in order:
        n = counts[(base, diam)]
        if n == 1:
            parts.append(f"{base} Ø{diam}")
        else:
            parts.append(f"{n} {base.lower()}s Ø{diam}")
    return " + ".join(parts)


def _delete_shear_line(dalle_id: int, sec_id: int, i: int):
    nk = KS("shear_n_lines", dalle_id, sec_id)
    prefix = "shear_line"
    n_lines = max(1, int(st.session_state.get(nk, 1) or 1))
    if n_lines <= 1 or i <= 0 or i >= n_lines:
        return
    for j in range(i, n_lines - 1):
        for suf in ("type", "d"):
            st.session_state[KS(f"{prefix}{j}_{suf}", dalle_id, sec_id)] = st.session_state.get(
                KS(f"{prefix}{j+1}_{suf}", dalle_id, sec_id)
            )
    for suf in ("type", "d"):
        st.session_state.pop(KS(f"{prefix}{n_lines-1}_{suf}", dalle_id, sec_id), None)
    st.session_state[nk] = n_lines - 1


def _add_shear_line(dalle_id: int, sec_id: int):
    nk = KS("shear_n_lines", dalle_id, sec_id)
    new_i = max(1, int(st.session_state.get(nk, 1) or 1))
    st.session_state[nk] = new_i + 1
    st.session_state.setdefault(KS(f"shear_line{new_i}_type", dalle_id, sec_id), "Étrier")
    st.session_state.setdefault(KS(f"shear_line{new_i}_d", dalle_id, sec_id), 10)


# ============================================================
#  COUCHES : callbacks ajout / suppression
# ============================================================
def _add_couche(dalle_id: int, sec_id: int, which: str):
    nk = KS(f"ncouches_{which}", dalle_id, sec_id)
    nc = _get_ncouches(dalle_id, sec_id, which)
    if nc >= MAX_COUCHES:
        return
    i = nc + 1
    # Renfort par défaut : barres Ø12/150 (modifiable en treillis).
    st.session_state[KS(f"arm_type_{which}_c{i}", dalle_id, sec_id)] = "Barres"
    st.session_state.setdefault(KS(f"treillis_{which}_c{i}", dalle_id, sec_id), TR.TREILLIS_DEFAUT)
    st.session_state.setdefault(KS(f"ø_barres_{which}_c{i}", dalle_id, sec_id), 12)
    st.session_state.setdefault(KS(f"esp_barres_{which}_c{i}", dalle_id, sec_id), 150)
    st.session_state[nk] = i


def _delete_couche(dalle_id: int, sec_id: int, which: str, i: int):
    """Callback on_click : suppression du renfort i (i>=2) avec décalage."""
    nk = KS(f"ncouches_{which}", dalle_id, sec_id)
    nc = _get_ncouches(dalle_id, sec_id, which)
    if i < 2 or i > nc:
        return
    for j in range(i, nc):
        for suf in ("arm_type", "treillis", "ø_barres", "esp_barres"):
            st.session_state[KS(f"{suf}_{which}_c{j}", dalle_id, sec_id)] = st.session_state.get(
                KS(f"{suf}_{which}_c{j+1}", dalle_id, sec_id)
            )
    for suf in ("arm_type", "treillis", "ø_barres", "esp_barres"):
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
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
    st.markdown("#### Sections")

    for sec in dalle.get("sections", []):
        sec_id = int(sec.get("id"))
        sec_name_key = f"meta_dal{dalle_id}_nom_{sec_id}"
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
                    key=f"dalle_add_sec_btn_{dalle_id}_{sec_id}",
                    help="Ajouter une section",
                    use_container_width=True,
                    on_click=_add_section,
                    args=(dalle_id,),
                    disabled=data_locked,
                )
            with cC:
                st.button(
                    "📋",
                    key=f"dalle_copy_sec_btn_{dalle_id}_{sec_id}",
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
                        key=f"dalle_del_sec_btn_{dalle_id}_{sec_id}",
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
            key=f"dalle_del_beam_btn_{dalle_id}",
            on_click=_delete_dalle,
            args=(dalle_id,),
            disabled=data_locked,
            use_container_width=True,
        )


def _toggle_dalle_lock(dalle_id: int):
    k = KD("lock_data", dalle_id)
    st.session_state[k] = not bool(st.session_state.get(k, False))


def render_caracteristiques_dalle(dalle_id: int):
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)

    dalle_name_key = f"meta_dalle_nom_{dalle_id}"
    st.session_state.setdefault(dalle_name_key, dalle.get("nom", f"Dalle {dalle_id}"))

    lock_key = KD("lock_data", dalle_id)
    st.session_state.setdefault(lock_key, False)
    data_locked = bool(st.session_state.get(lock_key, False))

    with st.expander(st.session_state.get(dalle_name_key, dalle.get("nom", f"Dalle {dalle_id}")), expanded=True):
        t1, tC, tL = st.columns([6, 0.8, 0.8], vertical_alignment="center")
        with t1:
            st.markdown("#### Caractéristiques de la dalle")
        with tC:
            st.button(
                "📋",
                key=f"dalle_btn_copy_{dalle_id}",
                help="Copier la dalle",
                use_container_width=True,
                on_click=_duplicate_dalle,
                args=(dalle_id,),
            )
        with tL:
            st.button(
                "🔒" if data_locked else "🔓",
                key=f"dalle_btn_lock_{dalle_id}",
                help="Dalle verrouillée — cliquer pour déverrouiller" if data_locked
                     else "Dalle éditable — cliquer pour verrouiller",
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

        cB, cH, cE = st.columns(3)
        with cB:
            st.number_input(
                "Larg. bande (cm)", min_value=20, max_value=500, step=5, key=KD("b", dalle_id),
                disabled=data_locked,
                help="Largeur de la bande de dalle étudiée — 100 cm = 1 mètre courant.",
            )
        with cH:
            st.number_input("Épaisseur (cm)", min_value=5, max_value=100, step=1, key=KD("h", dalle_id), disabled=data_locked)
        with cE:
            st.number_input("Enrob. béton (cm)", min_value=0.0, max_value=20.0, step=0.5, key=KD("enrobage_beton", dalle_id), disabled=data_locked)

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

    # Direction principale : celle du plus grand moment (X à égalité)
    principale = "x" if max(dirs["x"]["M_inf_val"], dirs["x"]["M_sup_val"]) >= \
        max(dirs["y"]["M_inf_val"], dirs["y"]["M_sup_val"]) else "y"

    # Distances couche 1 (cisaillement — logique Poutre : min des faces)
    dists_l1 = {w: _auto_dist_couche(dalle_id, sec_id, w, 1) for w in FACES_DIR}
    d_utile_for_shear = h - min(dists_l1.values())  # cm
    geom_shear_ok = d_utile_for_shear > 0.0
    d_calc_shear = max(d_utile_for_shear, 0.1)

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

    # --- Tranchant : τ = V / (0.75·b·h) (inchangé) ---
    tau_1 = 0.016 * fck_cube / 1.05
    tau_2 = 0.032 * fck_cube / 1.05
    tau_4 = 0.064 * fck_cube / 1.05

    def _shear_state(tau):
        if tau <= tau_1:
            return "ok"
        if tau <= tau_2:
            return "ok"
        if tau <= tau_4:
            return "warn"
        return "nok"

    if V_val > 0:
        tau = V_val * 1e3 / (0.75 * b * h * 100)
        etat_tau = _shear_state(tau)
        if not geom_shear_ok:
            etat_tau = "nok"
    else:
        etat_tau = "ok"

    def _pas_state(V_kn: float):
        pas = float(st.session_state.get(KS("shear_pas", dalle_id, sec_id), 30.0) or 30.0)
        Ast_e = _shear_lines_total_Ast_mm2(dalle_id, sec_id)
        pas_th = Ast_e * fyd * (d_calc_shear * 10.0) / (V_kn * 1e3) / 10.0
        s_max = min(0.75 * d_calc_shear, 30.0)
        pas_lim = min(pas_th, s_max)
        etat = "ok" if pas <= pas_lim else "nok"
        if not geom_shear_ok:
            etat = "nok"
        return etat

    etat_pas = _pas_state(V_val) if V_val > 0 else "ok"

    etat_global = _status_merge(etat_h, etat_inf, etat_sup, etat_tau, etat_pas)

    return {
        "etat_global": etat_global,
        "etat_h": etat_h,
        "etat_inf": etat_inf,        # fusion X/Y (bandeaux de synthèse)
        "etat_sup": etat_sup,
        "etat_tau": etat_tau,
        "etat_pas": etat_pas,
        "dirs": dirs,                # tout le détail par direction
        "principale": principale,    # "x" ou "y"
        "V_val": V_val,
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
        "d_utile_shear": d_utile_for_shear,
        "geom_shear_ok": geom_shear_ok,
    }


# ============================================================
#  UI : CISAILLEMENT (lignes, sans positions de barres)
# ============================================================
def _render_shear_lines_ui(dalle_id: int, sec_id: int, disabled: bool):
    n_key = KS("shear_n_lines", dalle_id, sec_id)
    pas_key = KS("shear_pas", dalle_id, sec_id)
    prefix = "shear_line"
    add_btn_key = KS("btn_add_shear_line", dalle_id, sec_id)
    del_btn_prefix = KS("btn_del_shear_line_", dalle_id, sec_id)

    n_lines = max(1, int(st.session_state.get(n_key, 1) or 1))
    st.session_state[n_key] = n_lines

    for i in range(n_lines):
        type_key = KS(f"{prefix}{i}_type", dalle_id, sec_id)
        st.session_state.setdefault(type_key, "Étrier")
        st.session_state.setdefault(KS(f"{prefix}{i}_d", dalle_id, sec_id), 10)

        va = "bottom" if i == 0 else "center"
        c0, c1, c3, c4 = st.columns([2.6, 1.3, 3.4, 0.65], vertical_alignment=va)

        with c0:
            st.selectbox(
                "Type",
                ["Étrier", "Épingle"],
                key=type_key,
                label_visibility="visible" if i == 0 else "collapsed",
                disabled=disabled,
                help="Un étrier = 2 brins · une épingle = 1 brin." if i == 0 else None,
            )
        with c1:
            st.selectbox(
                "Ø (mm)",
                SHEAR_DIAM_OPTS,
                key=KS(f"{prefix}{i}_d", dalle_id, sec_id),
                label_visibility="visible" if i == 0 else "collapsed",
                disabled=disabled,
            )
        with c3:
            if i == 0:
                float_input_fr_simple("Pas choisi (cm)", key=pas_key, default=30.0, min_value=1.0, disabled=disabled)
            else:
                st.markdown("")
        with c4:
            if i == 0:
                st.button(
                    "＋",
                    key=add_btn_key,
                    use_container_width=True,
                    disabled=disabled,
                    help="Ajouter une armature d'effort tranchant",
                    on_click=_add_shear_line,
                    args=(dalle_id, sec_id),
                )
            else:
                st.button(
                    "🗑️",
                    key=f"{del_btn_prefix}{i}",
                    use_container_width=True,
                    disabled=disabled,
                    on_click=_delete_shear_line,
                    args=(dalle_id, sec_id, i),
                )


# ============================================================
#  UI : TABLEAU DES COUCHES (Base / Renforts)
# ============================================================
_FACE_LABELS = {"inf_x": " (inf. X)", "sup_x": " (sup. X)",
                "inf_y": " (inf. Y)", "sup_y": " (sup. Y)"}


def _render_couche_row(dalle_id: int, sec_id: int, which: str, i: int, nc: int, disabled: bool):
    """Ligne du tableau des couches. i=1 : bouton '＋ Renfort' ; i>=2 : poubelle."""
    suffix = _FACE_LABELS.get(which, f" ({which})")

    type_key = KS(f"arm_type_{which}_c{i}", dalle_id, sec_id)
    treillis_key = KS(f"treillis_{which}_c{i}", dalle_id, sec_id)
    diam_key = KS(f"ø_barres_{which}_c{i}", dalle_id, sec_id)
    esp_key = KS(f"esp_barres_{which}_c{i}", dalle_id, sec_id)

    st.session_state.setdefault(type_key, "Treillis" if i == 1 else "Barres")
    st.session_state.setdefault(treillis_key, TR.TREILLIS_DEFAUT)
    st.session_state.setdefault(diam_key, 12)
    st.session_state.setdefault(esp_key, 150)

    typ = _couche_type(dalle_id, sec_id, which, i)
    As_pm = _couche_as_per_m(dalle_id, sec_id, which, i)
    dist = _auto_dist_couche(dalle_id, sec_id, which, i)

    c0, c1, c2, c3, c4, c5, cG, c6 = st.columns(COUCHE_COLS, vertical_alignment="center")
    with c0:
        st.markdown("Base" if i == 1 else f"Renfort {i - 1}")
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
        else:
            st.markdown("<div style='text-align:center;opacity:0.5;'>—</div>", unsafe_allow_html=True)
    with c4:
        # As de la couche (mm²/m) — calculée automatiquement, non modifiable
        st.text_input(
            f"As (mm²/m) (couche {i}){suffix}",
            value=f"{As_pm:.0f}",
            key=KS(f"as_disp_{which}_{i}", dalle_id, sec_id),
            disabled=True,
            label_visibility="collapsed",
        )
    with c5:
        st.text_input(
            f"Distance axe (cm) (couche {i}){suffix}",
            value=f"{dist:.1f}".replace(".", ","),
            key=KS(f"dist_disp_{which}_{i}", dalle_id, sec_id),
            disabled=True,
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
            st.button(
                "＋ Renfort",
                key=KS(f"btn_add_couche_{which}", dalle_id, sec_id),
                use_container_width=True,
                disabled=disabled or (nc >= MAX_COUCHES),
                help="Ajouter un renfort (treillis ou barres)",
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

    # Synchroniser le champ CDG AVANT le rendu du widget
    _sync_ycdg_state(dalle_id, sec_id, which)

    h0, h1, h2, h3, h4, h5, hG, h6 = st.columns(COUCHE_COLS, vertical_alignment="bottom")
    with h0:
        st.markdown("")
    with h1:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Type</div>", unsafe_allow_html=True)
    with h2:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Treillis / Ø</div>", unsafe_allow_html=True)
    with h3:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Esp. (mm)</div>", unsafe_allow_html=True)
    with h4:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>As (mm²/m)</div>", unsafe_allow_html=True)
    with h5:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Dist. axe (cm)</div>", unsafe_allow_html=True,
                    help="Distance d'axe = enrobage + demi-Ø arrondi au 0,5 cm sup. "
                         "+ jeu premier lit. Les couches d'une face sont posées dans "
                         "le même plan (renforts dans le plan du treillis).")
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
    dalle = next(d for d in st.session_state.dalles if int(d.get("id")) == dalle_id)
    sec = next(s for s in dalle["sections"] if int(s.get("id")) == sec_id)
    sec_nom = str(st.session_state.get(f"meta_dal{dalle_id}_nom_{sec_id}", sec.get("nom", f"Section {sec_id}")))

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

        # ---- Tranchant + étriers (moteur Poutre inchangé) ----
        tau_1, tau_2, tau_4 = states["tau_1"], states["tau_2"], states["tau_4"]

        def _shear_need_text(tau):
            if tau <= tau_1:
                return "Pas besoin d’étriers", "ok", "τ_adm_I", tau_1
            if tau <= tau_2:
                return "Besoin d’étriers", "ok", "τ_adm_II", tau_2
            if tau <= tau_4:
                return "Besoin de barres inclinées et d’étriers", "warn", "τ_adm_IV", tau_4
            return "Pas acceptable", "nok", "τ_adm_IV", tau_4

        def _bloc_pas(V_kn: float, pas_key_base: str, titre_tau: str, titre_pas: str, etat_pas_state: str):
            tau = V_kn * 1e3 / (0.75 * b * h * 100)
            besoin, etat_tau, nom_lim, tau_lim = _shear_need_text(tau)

            pct_tau = (tau / tau_lim * 100.0) if tau_lim > 0 else None
            open_bloc_left_right(titre_tau, "", etat_tau, pct=pct_tau)
            st.markdown(f"τ = {tau:.2f} N/mm² ≤ {nom_lim} = {tau_lim:.2f} N/mm² → {besoin}")
            close_bloc()

            pas = float(st.session_state.get(KS(pas_key_base, dalle_id, sec_id), 30.0) or 30.0)
            Ast_e = _shear_lines_total_Ast_mm2(dalle_id, sec_id)
            d_sh = max(states["d_utile_shear"], 0.1)
            pas_th = Ast_e * fyd * (d_sh * 10.0) / (V_kn * 1e3) / 10.0  # cm
            s_max = min(0.75 * d_sh, 30.0)
            pas_lim = min(pas_th, s_max)

            # Affichage TRONQUÉ vers le bas — même règle que Poutre v2.41 :
            # la valeur affichée est recopiable sans faire basculer le
            # verdict ; la comparaison reste sur la valeur exacte.
            pas_th_aff = math.floor(pas_th * 10.0) / 10.0
            s_max_aff = math.floor(s_max * 10.0) / 10.0
            help_pas = (
                "**s,th = Aₛₜ · fyd · d / V**\n\n"
                f"Aₛₜ = {_fr(Ast_e, 1)} mm²\n\n"
                f"fyd = {_fr(fyd, 0)} N/mm²\n\n"
                f"d = {_fr(d_sh, 1)} cm\n\n"
                f"V = {_fr(V_kn, 1)} kN\n\n"
                f"→ s,th = {_fr(pas_th_aff, 1)} cm (tronqué au mm inférieur)"
            )

            right_et = _shear_lines_summary(dalle_id, sec_id)
            pct_pas = (pas / pas_lim * 100.0) if pas_lim > 0 else None
            open_bloc_left_right(titre_pas, right_et, etat_pas_state, pct=pct_pas)
            a1, a2, a3 = st.columns(3)
            with a1:
                st.markdown(f"**Pas théorique = {pas_th_aff:.1f} cm**", help=help_pas)
            with a2:
                st.markdown(f"**Pas maximal = {s_max_aff:.1f} cm**", help="**s,max = min( 0,75 · d ; 30 cm )**")
            with a3:
                st.markdown(f"**Asw = {Ast_e:.0f} mm²**",
                            help="Section totale des armatures d'effort tranchant "
                                 "(Σ brins × aire du Ø).")
            close_bloc()

            _render_shear_lines_ui(dalle_id, sec_id, disabled=dim_locked)

        if V_val > 0:
            _bloc_pas(V_val, "shear_pas", "Vérification de l'effort tranchant", "Détermination des étriers", states["etat_pas"])


# ============================================================
#  UI : INFOS PROJET / PARAMÈTRES AVANCÉS
# ============================================================
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
    st.session_state["dalle_show_param_avances"] = not bool(st.session_state.get("dalle_show_param_avances", False))


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
            key="dalle_btn_toggle_infos_projet",
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
    for d in st.session_state.dalles:
        did = int(d["id"])
        d["nom"] = str(st.session_state.get(f"meta_dalle_nom_{did}", d.get("nom", f"Dalle {did}")))
        render_caracteristiques_dalle(did)


def render_dimensionnement_right(beton_data: dict):
    for d in st.session_state.dalles:
        did = int(d["id"])
        dnom = str(st.session_state.get(f"meta_dalle_nom_{did}", d.get("nom", f"Dalle {did}")))
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
        st.markdown("## Dalle en béton armé")
    with tH2:
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;font-size:0.9em;'>Version {DALLE_VERSION}</div>",
            unsafe_allow_html=True,
        )
    with tH3:
        st.button("❔", key="dalle_btn_version_hist", help="Version actuelle du module Dalle.",
                  use_container_width=True)

    btn1, btn2, btn3, btn4, btn5 = st.columns(5)

    with btn1:
        if st.button("🏠 Accueil", use_container_width=True, key="dalle_btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()

    with btn2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="dalle_btn_reset"):
            _reset_module()

    with btn3:
        payload = _build_save_payload()
        st.download_button(
            label="💾 Enregistrer",
            data=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="dalle_ba.json",
            mime="application/json",
            use_container_width=True,
            key="dalle_btn_save_dl",
        )

    with btn4:
        if st.button("📂 Ouvrir", use_container_width=True, key="dalle_btn_open_toggle"):
            st.session_state["dalle_show_open_uploader"] = not st.session_state.get("dalle_show_open_uploader", False)

        if st.session_state.get("dalle_show_open_uploader", False):
            uploaded = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed", key="dalle_open_uploader")
            if uploaded is not None:
                try:
                    data = json.load(uploaded)
                    if not isinstance(data, dict):
                        raise ValueError("Structure JSON inattendue")
                    _load_from_payload(data)
                    st.session_state["dalle_show_open_uploader"] = False
                    st.rerun()
                except Exception:
                    st.error("Fichier invalide ou corrompu — chargement annulé.")

    with btn5:
        if st.button("📄 Générer PDF", use_container_width=True, key="dalle_btn_pdf"):
            from modules.export_pdf_dalle import generer_rapport_pdf

            infos = {
                "nom_projet": st.session_state.get("nom_projet", ""),
                "partie": st.session_state.get("partie", ""),
                "date": st.session_state.get("date", datetime.today().strftime("%d/%m/%Y")),
                "indice": st.session_state.get("indice", "0"),
            }

            try:
                fichier_pdf = generer_rapport_pdf(
                    dalles=st.session_state.dalles,
                    values=dict(st.session_state),
                    beton_data=beton_data,
                    infos=infos,
                )
                with open(fichier_pdf, "rb") as f:
                    st.session_state["dalle_pdf_bytes"] = f.read()
                st.success("✅ Note de calcul générée")
            except Exception as e:
                st.session_state.pop("dalle_pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")

        if st.session_state.get("dalle_pdf_bytes"):
            st.download_button(
                label="⬇️ Télécharger le rapport PDF",
                data=st.session_state["dalle_pdf_bytes"],
                file_name=_pdf_filename(),
                mime="application/pdf",
                use_container_width=True,
                key="dalle_btn_pdf_dl",
            )

    input_col_gauche, result_col_droite = st.columns([2, 3])

    with input_col_gauche:
        render_infos_projet()
        render_donnees_left(beton_data)

    with result_col_droite:
        st.session_state.setdefault("dalle_show_param_avances", False)

        cH1, cH2 = st.columns([18, 1.3], vertical_alignment="center")
        with cH1:
            st.markdown("### Dimensionnement")
        with cH2:
            st.button(
                "⚙️",
                key="dalle_btn_toggle_param_avances",
                help="Paramètres avancés",
                use_container_width=True,
                on_click=_toggle_param_avances,
            )

        if bool(st.session_state.get("dalle_show_param_avances", False)):
            with st.container(border=True):
                render_parametres_avances()

        render_dimensionnement_right(beton_data)

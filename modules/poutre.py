# ===========================
#  VERSION 2.20
# ===========================
#  poutre.py (Streamlit)
#
#  Évolutions vs 2.10 :
#   1. ENROBAGES -> "Distance axe lit / parement (cm)" :
#      - Lit 1 (auto) = enrobage béton
#                     + Ø étrier arrondi au 0,5 cm sup.
#                     + demi-Ø barre lit 1 arrondi au 0,5 cm sup.
#                     + jeu premier lit
#      - Lit i (auto) = distance lit (i-1)
#                     + demi-Ø lit (i-1) arrondi au 0,5 cm sup.
#                     + jeu entre lits (paramètre global)
#                     + demi-Ø lit i arrondi au 0,5 cm sup.
#      - Chaque distance reste modifiable manuellement (override),
#        pour les inférieures (mesurée depuis le bas) comme pour les
#        supérieures (mesurée depuis le haut).
#   2. PARAMÈTRES AVANCÉS : Diamètre étrier (mm), Jeu premier lit (cm),
#      Jeu entre lits (cm), Tolérance dépassement (%).
#      (L'enrobage béton reste PAR POUTRE dans les caractéristiques,
#       pour ne pas perdre cette fonctionnalité.)
#   3. HAUTEUR UTILE : positions réelles de chaque lit (centre de
#      gravité pondéré par les aires) pour la flexion, inf. et sup.
#      Cisaillement : inchangé (lit 1, min des deux faces).
#   4. SECTIONS : nom éditable directement dans l'en-tête (plus de
#      ligne "Nom de la section"), nommage automatique par lettres
#      (Section A, B, C... premier nom libre), bouton 📋 copier la
#      section (toutes les données), bouton 🗑️ supprimer intégré.
#   5. fyd = fyk / 1.5 : VOLONTAIRE (méthode ancienne) — ne pas "corriger".
# ===========================
import streamlit as st
from datetime import datetime
from string import ascii_uppercase
import json
import math
import re
from copy import deepcopy

# ============================================================
#  STYLES BLOCS
# ============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌"}

# Données béton (chargées dans show())
BETON_DATA = {}

MAX_LITS = 4  # nombre maximal de lits d'armatures par face


def open_bloc_left_right(left: str, right: str = "", etat: str = "ok"):
    """
    Header de bloc : texte à gauche + texte à droite (aligné contre l'icône à droite).
    NB : le rendu (seule la barre d'en-tête est colorée) repose sur
    l'auto-fermeture du HTML par Streamlit. Ne pas modifier.
    """
    right_html = f"<div style='font-weight:600;opacity:0.9;white-space:nowrap;'>{right}</div>" if right else ""
    st.markdown(
        f"""
        <div style="
            background-color:{C_COULEURS.get(etat, '#f6f6f6')};
            padding:12px 14px 10px 14px;
            border-radius:10px;
            border:1px solid #d9d9d9;
            margin:10px 0 12px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="font-weight:700;">{left}</div>
            <div style="display:flex;align-items:center;gap:10px;">
              {right_html}
              <div style="font-size:20px;line-height:1;">{C_ICONES.get(etat, '')}</div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )


def close_bloc():
    st.markdown("</div>", unsafe_allow_html=True)


def small_italic_label_right(txt: str):
    """Libellé italique aligné à droite pour être collé visuellement à une checkbox."""
    st.markdown(
        f"<div style='text-align:right;font-style:italic;opacity:0.75;white-space:nowrap;padding-right:0px;margin-right:0px;'>{txt}</div>",
        unsafe_allow_html=True,
    )


# ============================================================
#  UTILITAIRES SESSION / CLÉS
# ============================================================
def KB(base: str, beam_id: int) -> str:
    return f"b{beam_id}_{base}"


def KS(base: str, beam_id: int, sec_id: int) -> str:
    return f"b{beam_id}_sec{sec_id}_{base}"


# Clés globales persistées (sauvegarde JSON + épinglage anti-nettoyage)
PERSISTED_GLOBAL_KEYS = {
    "units_len",
    "units_as",
    "tau_tolerance_percent",
    "diam_etrier_mm",       # Ø étrier pour la distance auto du lit 1
    "jeu_enrobage_cm",      # "Jeu premier lit (cm)" (clé conservée pour compat anciens JSON)
    "jeu_entre_lits_cm",    # "Jeu entre lits (cm)"
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


def _pin_persistent_state():
    """
    FIX PERSISTANCE :
    Streamlit supprime en fin de run l'état des widgets qui n'ont pas été
    rendus (champs conditionnels : M_sup masqué, lits repliés, cisaillement
    quand V=0, infos projet quand la case est décochée...).
    Ré-affecter chaque clé à elle-même en début de run la marque comme
    "gérée par l'application" et empêche ce nettoyage.
    """
    for k in list(st.session_state.keys()):
        if _is_transient_key(k):
            continue
        if re.match(r"^b\d+_", k) or k.startswith("meta_") or k in PERSISTED_GLOBAL_KEYS:
            st.session_state[k] = st.session_state[k]
        elif k.endswith("_raw") and (re.match(r"^b\d+_", k) or k[:-4] in PERSISTED_GLOBAL_KEYS):
            st.session_state[k] = st.session_state[k]


def _ensure_global_defaults():
    """Défauts des paramètres globaux (avant tout calcul / rendu)."""
    st.session_state.setdefault("units_len", "cm")
    st.session_state.setdefault("units_as", "mm²")
    st.session_state.setdefault("tau_tolerance_percent", 0)
    st.session_state.setdefault("diam_etrier_mm", 8)
    st.session_state.setdefault("jeu_enrobage_cm", 1.0)   # jeu premier lit
    st.session_state.setdefault("jeu_entre_lits_cm", 1.0)
    _coerce_int_choice("diam_etrier_mm", [6, 8, 10, 12], 8)


# ============================================================
#  RESET
# ============================================================
def _reset_module():
    current_page = st.session_state.get("page")
    st.session_state.clear()
    if current_page:
        st.session_state.page = current_page
    st.rerun()


# ============================================================
#  SAISIE DÉCIMALE FR (texte)
# ============================================================
def float_input_fr_simple(label, key, default=0.0, min_value=0.0, disabled: bool = False):
    """
    Champ décimal FR (virgule). La valeur numérique vit dans `key`,
    la saisie texte dans `key_raw`.
    """
    if key not in st.session_state:
        st.session_state[key] = float(default)
    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = f"{float(st.session_state[key]):.2f}".replace(".", ",")

    raw = st.text_input(label, key=raw_key, disabled=disabled)

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
    """Force une valeur de session à être un entier appartenant aux options.
    À n'appeler qu'AVANT l'instanciation du widget correspondant."""
    cur = st.session_state.get(key, default)
    try:
        cur = int(float(cur))
    except Exception:
        cur = default
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


def _next_section_name(beam_id: int) -> str:
    """Premier nom 'Section X' non utilisé dans la poutre (X = lettre)."""
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    used = set()
    for s in beam.get("sections", []):
        sid = int(s.get("id"))
        used.add(str(st.session_state.get(f"meta_b{beam_id}_nom_{sid}", s.get("nom", ""))))
        used.add(str(s.get("nom", "")))
    for L in _letter_sequence():
        candidate = f"Section {L}"
        if candidate not in used:
            return candidate
    return f"Section {len(beam.get('sections', [])) + 1}"  # repli improbable


# ============================================================
#  POUTRES / SECTIONS : INIT / ADD / DELETE / DUPLICATE / COPY
# ============================================================
def _init_beams_if_needed():
    if "beams" not in st.session_state or not isinstance(st.session_state.beams, list) or len(st.session_state.beams) == 0:
        st.session_state.beams = [{"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "Section A"}]}]

    for b in st.session_state.beams:
        b["id"] = int(b.get("id", 0))
        b["nom"] = str(b.get("nom", f"Poutre {b['id']}"))
        if "sections" not in b or not isinstance(b["sections"], list) or len(b["sections"]) == 0:
            b["sections"] = [{"id": 1, "nom": "Section A"}]
        for s in b["sections"]:
            s["id"] = int(s.get("id", 0))
            s["nom"] = str(s.get("nom", f"Section {s['id']}"))

    if not any(int(b.get("id", 0)) == 1 for b in st.session_state.beams):
        st.session_state.beams.insert(0, {"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "Section A"}]})

    for b in st.session_state.beams:
        if not any(int(s.get("id", 0)) == 1 for s in b["sections"]):
            b["sections"].insert(0, {"id": 1, "nom": "Section A"})

    # Synchronisation des noms (labels d'expander à jour immédiatement)
    for b in st.session_state.beams:
        bid = int(b["id"])
        key_nom = f"meta_beam_nom_{bid}"
        if key_nom not in st.session_state:
            st.session_state[key_nom] = str(b.get("nom", f"Poutre {bid}"))
        b["nom"] = str(st.session_state.get(key_nom, b.get("nom")))

        for s in b.get("sections", []):
            sid = int(s["id"])
            key_snom = f"meta_b{bid}_nom_{sid}"
            if key_snom not in st.session_state:
                st.session_state[key_snom] = str(s.get("nom", f"Section {sid}"))
            s["nom"] = str(st.session_state.get(key_snom, s.get("nom")))

    # Defaults + migrations
    for b in st.session_state.beams:
        _ensure_defaults_for_beam(int(b["id"]))


def _next_beam_id() -> int:
    ids = [int(b.get("id", 0)) for b in st.session_state.beams]
    return (max(ids) + 1) if ids else 1


def _next_section_id(beam_id: int) -> int:
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    ids = [int(s.get("id", 0)) for s in beam["sections"]]
    return (max(ids) + 1) if ids else 1


DIAM_OPTS = [6, 8, 10, 12, 16, 20, 25, 32, 40]
SHEAR_DIAM_OPTS = [6, 8, 10, 12]


def _migrate_second_lit(beam_id: int, sec_id: int, which: str):
    """
    Migration des anciennes clés 'second lit' (checkbox + *_2) vers le
    système multi-lits. Les barres sont reprises ; la position du lit 2
    est recalculée automatiquement avec la nouvelle logique de distances
    (l'ancien 'jeu' par lit n'existe plus : le jeu entre lits est global).
    Idempotent, fonctionne aussi pour les anciens fichiers JSON rechargés.
    """
    old_flag = KS(f"ajouter_second_lit_{which}", beam_id, sec_id)
    nkey = KS(f"nlits_{which}", beam_id, sec_id)
    if bool(st.session_state.get(old_flag, False)):
        if int(st.session_state.get(nkey, 1) or 1) < 2:
            st.session_state[nkey] = 2
            st.session_state[KS(f"n_as_{which}_l2", beam_id, sec_id)] = int(
                st.session_state.get(KS(f"n_as_{which}_2", beam_id, sec_id), 2) or 2
            )
            st.session_state[KS(f"ø_as_{which}_l2", beam_id, sec_id)] = int(
                st.session_state.get(KS(f"ø_as_{which}_2", beam_id, sec_id), 16) or 16
            )
        st.session_state[old_flag] = False


def _ensure_defaults_for_beam(beam_id: int):
    # Poutre
    st.session_state.setdefault(KB("b", beam_id), 20)
    st.session_state.setdefault(KB("h", beam_id), 40)
    st.session_state.setdefault(KB("enrobage_beton", beam_id), 3.0)

    if BETON_DATA:
        default_beton = "C30/37" if "C30/37" in BETON_DATA else list(BETON_DATA.keys())[0]
        st.session_state.setdefault(KB("beton", beam_id), default_beton)
        if st.session_state.get(KB("beton", beam_id)) not in BETON_DATA:
            st.session_state[KB("beton", beam_id)] = default_beton

    # Acier par poutre : ENTIER 400 ou 500
    fkey = KB("fyk", beam_id)
    cur = st.session_state.get(fkey, 500)
    try:
        cur = int(float(cur))
    except Exception:
        cur = 500
    if cur not in (400, 500):
        cur = 500
    st.session_state[fkey] = cur

    # Statut (En cours / Validé) + compat lock_data
    st.session_state.setdefault(KB("lock_data", beam_id), False)
    statut_key = KB("statut", beam_id)
    if statut_key not in st.session_state:
        st.session_state[statut_key] = "Validé" if bool(st.session_state.get(KB("lock_data", beam_id), False)) else "En cours"
    st.session_state[KB("lock_data", beam_id)] = (st.session_state.get(statut_key) == "Validé")

    # Sections
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    for s in beam.get("sections", []):
        sid = int(s["id"])
        st.session_state.setdefault(KS("M_inf", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("M_sup", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("V", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("V_lim", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("ajouter_moment_sup", beam_id, sid), False)
        st.session_state.setdefault(KS("ajouter_effort_reduit", beam_id, sid), False)

        for which in ("inf", "sup"):
            st.session_state.setdefault(KS(f"n_as_{which}", beam_id, sid), 2)
            st.session_state.setdefault(KS(f"ø_as_{which}", beam_id, sid), 16)
            st.session_state.setdefault(KS(f"nlits_{which}", beam_id, sid), 1)

            # Migration ancien "second lit"
            _migrate_second_lit(beam_id, sid, which)

            # Coercitions + defaults des lits existants
            _coerce_int_choice(KS(f"ø_as_{which}", beam_id, sid), DIAM_OPTS, 16)
            nl = max(1, min(MAX_LITS, int(st.session_state.get(KS(f"nlits_{which}", beam_id, sid), 1) or 1)))
            st.session_state[KS(f"nlits_{which}", beam_id, sid)] = nl
            for i in range(2, nl + 1):
                st.session_state.setdefault(KS(f"n_as_{which}_l{i}", beam_id, sid), 2)
                st.session_state.setdefault(KS(f"ø_as_{which}_l{i}", beam_id, sid), 16)
                st.session_state.setdefault(KS(f"dist_{which}_l{i}_override", beam_id, sid), False)
                _coerce_int_choice(KS(f"ø_as_{which}_l{i}", beam_id, sid), DIAM_OPTS, 16)

        # Distance axe lit 1 / parement (modifiable par face)
        # NB : clés 'enrob_calc_*' conservées pour compat anciens JSON.
        st.session_state.setdefault(KS("enrob_calc_inf", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("enrob_calc_sup", beam_id, sid), 0.0)
        st.session_state.setdefault(KS("enrob_calc_inf_override", beam_id, sid), False)
        st.session_state.setdefault(KS("enrob_calc_sup_override", beam_id, sid), False)

        # Lock dimensionnement
        st.session_state.setdefault(KS("lock_dim", beam_id, sid), False)

        # Cisaillement
        st.session_state.setdefault(KS("shear_n_lines", beam_id, sid), 1)
        st.session_state.setdefault(KS("shear_pas", beam_id, sid), 30.0)
        st.session_state.setdefault(KS("shear_n_lines_r", beam_id, sid), 1)
        st.session_state.setdefault(KS("shear_pas_r", beam_id, sid), 30.0)

        for prefix, nk in (("shear_line", "shear_n_lines"), ("shear_r_line", "shear_n_lines_r")):
            n_lines = max(1, int(st.session_state.get(KS(nk, beam_id, sid), 1) or 1))
            st.session_state[KS(nk, beam_id, sid)] = n_lines
            for i in range(n_lines):
                st.session_state.setdefault(KS(f"{prefix}{i}_type", beam_id, sid), "Étriers (2 brins)" if i == 0 else "Épingles (1 brin)")
                st.session_state.setdefault(KS(f"{prefix}{i}_n", beam_id, sid), 1)
                st.session_state.setdefault(KS(f"{prefix}{i}_d", beam_id, sid), 8)
                _coerce_int_choice(KS(f"{prefix}{i}_d", beam_id, sid), SHEAR_DIAM_OPTS, 8)


def _add_beam():
    new_id = _next_beam_id()
    st.session_state.beams.append({"id": new_id, "nom": f"Poutre {new_id}", "sections": [{"id": 1, "nom": "Section A"}]})
    st.session_state[f"meta_beam_nom_{new_id}"] = f"Poutre {new_id}"
    st.session_state[f"meta_b{new_id}_nom_1"] = "Section A"
    _ensure_defaults_for_beam(new_id)


def _delete_beam(beam_id: int):
    if beam_id == 1:
        return
    st.session_state.beams = [b for b in st.session_state.beams if int(b.get("id")) != beam_id]
    prefix = f"b{beam_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_beam_nom_{beam_id}", None)
    for k in list(st.session_state.keys()):
        if k.startswith(f"meta_b{beam_id}_nom_"):
            del st.session_state[k]


def _duplicate_beam(src_beam_id: int):
    """Disponible si tu veux brancher un bouton 'Dupliquer la poutre' plus tard."""
    src = next(b for b in st.session_state.beams if int(b.get("id")) == src_beam_id)
    new_id = _next_beam_id()
    st.session_state.beams.append({"id": new_id, "nom": f"{src.get('nom','Poutre')} (copie)", "sections": deepcopy(src["sections"])})

    src_prefix = f"b{src_beam_id}_"
    dst_prefix = f"b{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_beam_nom_{new_id}"] = f"{st.session_state.get(f'meta_beam_nom_{src_beam_id}', src.get('nom','Poutre'))} (copie)"
    for s in src.get("sections", []):
        sid = int(s.get("id"))
        st.session_state[f"meta_b{new_id}_nom_{sid}"] = st.session_state.get(f"meta_b{src_beam_id}_nom_{sid}", s.get("nom", f"Section {sid}"))

    _ensure_defaults_for_beam(new_id)


def _add_section(beam_id: int):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    new_id = _next_section_id(beam_id)
    name = _next_section_name(beam_id)
    beam["sections"].append({"id": new_id, "nom": name})
    st.session_state[f"meta_b{beam_id}_nom_{new_id}"] = name
    _ensure_defaults_for_beam(beam_id)


def _copy_section(beam_id: int, src_sec_id: int):
    """
    Copie intégrale d'une section (sollicitations, armatures inf./sup.,
    lits, distances, cisaillement, options) vers une nouvelle section
    nommée avec la première lettre disponible. Callback on_click.
    """
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    new_id = _next_section_id(beam_id)
    name = _next_section_name(beam_id)
    beam["sections"].append({"id": new_id, "nom": name})

    src_prefix = f"b{beam_id}_sec{src_sec_id}_"
    dst_prefix = f"b{beam_id}_sec{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix) and not _is_transient_key(k):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])

    st.session_state[f"meta_b{beam_id}_nom_{new_id}"] = name
    _ensure_defaults_for_beam(beam_id)


def _delete_section(beam_id: int, sec_id: int):
    if sec_id == 1:
        return
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    beam["sections"] = [s for s in beam["sections"] if int(s.get("id")) != sec_id]
    prefix = f"b{beam_id}_sec{sec_id}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        del st.session_state[k]
    st.session_state.pop(f"meta_b{beam_id}_nom_{sec_id}", None)


# ============================================================
#  SAVE / LOAD JSON (beams + valeurs)
# ============================================================
def _build_save_payload():
    beams = []
    for b in st.session_state.beams:
        beams.append(
            {
                "id": int(b.get("id")),
                "nom": str(b.get("nom")),
                "sections": [{"id": int(s.get("id")), "nom": str(s.get("nom"))} for s in b.get("sections", [])],
            }
        )

    values = {}
    for k in list(st.session_state.keys()):
        if _is_transient_key(k):
            continue
        if k in PERSISTED_GLOBAL_KEYS or (k.endswith("_raw") and k[:-4] in PERSISTED_GLOBAL_KEYS):
            values[k] = st.session_state[k]
        elif re.match(r"^b\d+_", k):
            values[k] = st.session_state[k]
        elif k.startswith("meta_beam_nom_") or (k.startswith("meta_b") and "_nom_" in k):
            values[k] = st.session_state[k]

    return {"version": "2.20", "beams": beams, "values": values}


def _load_from_payload(payload: dict):
    beams = payload.get("beams", None)
    values = payload.get("values", {})

    if isinstance(beams, list) and len(beams) > 0:
        cleaned = []
        for b in beams:
            try:
                bid = int(b.get("id"))
            except Exception:
                continue
            secs = b.get("sections", [])
            if not isinstance(secs, list) or len(secs) == 0:
                secs = [{"id": 1, "nom": "Section A"}]
            cleaned_secs = []
            for s in secs:
                try:
                    sid = int(s.get("id"))
                except Exception:
                    continue
                cleaned_secs.append({"id": sid, "nom": str(s.get("nom", f"Section {sid}"))})
            cleaned.append({"id": bid, "nom": str(b.get("nom", f"Poutre {bid}")), "sections": cleaned_secs})
        st.session_state.beams = cleaned if cleaned else [{"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "Section A"}]}]
    else:
        st.session_state.beams = [{"id": 1, "nom": "Poutre 1", "sections": [{"id": 1, "nom": "Section A"}]}]

    if isinstance(values, dict):
        for k, v in values.items():
            if _is_transient_key(k):
                continue
            if k in PERSISTED_GLOBAL_KEYS or (k.endswith("_raw") and k[:-4] in PERSISTED_GLOBAL_KEYS):
                st.session_state[k] = v
            elif re.match(r"^b\d+_", k):
                st.session_state[k] = v
            elif k.startswith("meta_beam_nom_") or (k.startswith("meta_b") and "_nom_" in k):
                st.session_state[k] = v

    _ensure_global_defaults()
    _init_beams_if_needed()


# ============================================================
#  OUTILS CALCUL
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


def _status_with_tolerance(value: float, limit: float, tol_percent: float):
    """Dans la tolérance => rester en VERT + texte 'Acceptable (tolérance ...)'."""
    if limit <= 0:
        return "nok", ""
    if value <= limit:
        return "ok", ""
    lim2 = limit * (1.0 + max(0.0, tol_percent) / 100.0)
    if value <= lim2:
        return "ok", f"Acceptable (tolérance +{tol_percent:.0f}%)"
    return "nok", ""


def _brins_from_type(type_txt: str) -> int:
    if "3 brins" in type_txt:
        return 3
    if "2 brins" in type_txt:
        return 2
    return 1


def _get_fyk_and_mu_ref(beam_id: int):
    """Acier par poutre : 400 ou 500 (lecture seule — coercition dans _ensure_defaults)."""
    try:
        fyk_i = int(float(st.session_state.get(KB("fyk", beam_id), 500)))
    except Exception:
        fyk_i = 500
    if fyk_i not in (400, 500):
        fyk_i = 500
    return float(fyk_i), str(fyk_i)


def _round_up_to_half_cm(x_cm: float) -> float:
    try:
        return math.ceil(float(x_cm) * 2.0) / 2.0
    except Exception:
        return x_cm


# ============================================================
#  DISTANCES AXE LIT / PAREMENT
# ============================================================
def _get_nlits(beam_id: int, sec_id: int, which: str) -> int:
    try:
        nl = int(st.session_state.get(KS(f"nlits_{which}", beam_id, sec_id), 1) or 1)
    except Exception:
        nl = 1
    return max(1, min(MAX_LITS, nl))


def _lit_bars(beam_id: int, sec_id: int, which: str, i: int):
    """Retourne (n, Ø) du lit i (i>=1)."""
    if i == 1:
        n = int(st.session_state.get(KS(f"n_as_{which}", beam_id, sec_id), 2) or 2)
        d = int(st.session_state.get(KS(f"ø_as_{which}", beam_id, sec_id), 16) or 16)
    else:
        n = int(st.session_state.get(KS(f"n_as_{which}_l{i}", beam_id, sec_id), 2) or 2)
        d = int(st.session_state.get(KS(f"ø_as_{which}_l{i}", beam_id, sec_id), 16) or 16)
    return n, d


def _dist_keys(beam_id: int, sec_id: int, which: str, i: int):
    """Clés (valeur, override) de la distance axe lit i / parement."""
    if i == 1:
        # clés historiques conservées pour la compatibilité des anciens JSON
        return KS(f"enrob_calc_{which}", beam_id, sec_id), KS(f"enrob_calc_{which}_override", beam_id, sec_id)
    return KS(f"dist_{which}_l{i}", beam_id, sec_id), KS(f"dist_{which}_l{i}_override", beam_id, sec_id)


def _auto_dist_lit(beam_id: int, sec_id: int, which: str, i: int) -> float:
    """
    Distance automatique axe lit i / parement (cm).

    Lit 1 :
      enrobage béton
      + Ø étrier (cm) arrondi au 0,5 cm sup.
      + demi-Ø barre lit 1 (cm) arrondi au 0,5 cm sup.
      + jeu premier lit
      Ex : 3,0 + arr(0,8)=1,0 + arr(0,8)=1,0 + 1,0 = 6,0 cm
           (avec jeu premier lit = 0 : 5,0 cm)

    Lit i (i >= 2) :
      distance lit (i-1)   [valeur réelle, override compris]
      + demi-Ø lit (i-1) arrondi au 0,5 cm sup.
      + jeu entre lits (paramètre global)
      + demi-Ø lit i arrondi au 0,5 cm sup.
      Ex (Ø16/Ø16, lit 1 à 5,0, jeu entre lits 1,0) :
           5,0 + 1,0 + 1,0 + 1,0 = 8,0 cm
    """
    if i == 1:
        enrob_beton = float(st.session_state.get(KB("enrobage_beton", beam_id), 3.0) or 3.0)
        d_etrier = float(st.session_state.get("diam_etrier_mm", 8) or 8)
        jeu1 = float(st.session_state.get("jeu_enrobage_cm", 1.0) or 0.0)
        _, d1 = _lit_bars(beam_id, sec_id, which, 1)
        return (
            enrob_beton
            + _round_up_to_half_cm(d_etrier / 10.0)
            + _round_up_to_half_cm(d1 / 20.0)
            + jeu1
        )

    prev_dist = _get_dist_lit(beam_id, sec_id, which, i - 1)
    _, d_prev = _lit_bars(beam_id, sec_id, which, i - 1)
    _, d_i = _lit_bars(beam_id, sec_id, which, i)
    jeuL = float(st.session_state.get("jeu_entre_lits_cm", 1.0) or 0.0)
    return (
        prev_dist
        + _round_up_to_half_cm(d_prev / 20.0)
        + jeuL
        + _round_up_to_half_cm(d_i / 20.0)
    )


def _get_dist_lit(beam_id: int, sec_id: int, which: str, i: int) -> float:
    """
    Distance axe lit i / parement effectivement utilisée :
    valeur auto, sauf override manuel de l'utilisateur.
    NB : peut écrire dans la clé du widget -> à n'appeler qu'AVANT le
    rendu des widgets de la section (les calculs précèdent toujours le
    rendu, et on ne réécrit que si la valeur change réellement).
    """
    key_val, key_ovr = _dist_keys(beam_id, sec_id, which, i)
    auto_val = _auto_dist_lit(beam_id, sec_id, which, i)

    if key_ovr not in st.session_state:
        st.session_state[key_ovr] = False
    if key_val not in st.session_state:
        st.session_state[key_val] = float(auto_val)

    if not bool(st.session_state.get(key_ovr, False)):
        if abs(float(st.session_state.get(key_val, auto_val) or auto_val) - float(auto_val)) > 1e-9:
            st.session_state[key_val] = float(auto_val)

    try:
        return float(st.session_state.get(key_val, auto_val) or auto_val)
    except Exception:
        return float(auto_val)


def _layers_geometry(beam_id: int, sec_id: int, which: str):
    """
    Pour une face :
      - As_total (mm²)
      - e_cdg (cm) : distance parement -> centre de gravité des lits
        (positions RÉELLES de chaque lit, overrides compris)
      - detail : chaîne '2Ø16 + 2Ø20 ...'
    """
    nl = _get_nlits(beam_id, sec_id, which)
    parts = []
    As_tot = 0.0
    somme_As_e = 0.0

    for i in range(1, nl + 1):
        n, d = _lit_bars(beam_id, sec_id, which, i)
        e = _get_dist_lit(beam_id, sec_id, which, i)
        As_i = n * _bar_area_mm2(d)
        As_tot += As_i
        somme_As_e += As_i * e
        parts.append(f"{n}Ø{d}")

    e_cdg = (somme_As_e / As_tot) if As_tot > 0 else _get_dist_lit(beam_id, sec_id, which, 1)
    return As_tot, e_cdg, " + ".join(parts)


# ============================================================
#  CISAILLEMENT : aires, résumé, callbacks
# ============================================================
def _shear_prefix_nkey(reduced: bool):
    return ("shear_r_line", "shear_n_lines_r") if reduced else ("shear_line", "shear_n_lines")


def _shear_lines_total_Ast_mm2(beam_id: int, sec_id: int, reduced: bool) -> float:
    prefix, nk = _shear_prefix_nkey(reduced)
    n_lines = max(1, int(st.session_state.get(KS(nk, beam_id, sec_id), 1) or 1))
    Ast = 0.0
    for i in range(n_lines):
        typ = str(st.session_state.get(KS(f"{prefix}{i}_type", beam_id, sec_id), "Étriers (2 brins)"))
        n_cadres = int(st.session_state.get(KS(f"{prefix}{i}_n", beam_id, sec_id), 1) or 1)
        diam = float(st.session_state.get(KS(f"{prefix}{i}_d", beam_id, sec_id), 8) or 8)
        Ast += n_cadres * _brins_from_type(typ) * _bar_area_mm2(diam)
    return Ast


def _shear_lines_summary(beam_id: int, sec_id: int, reduced: bool) -> str:
    prefix, nk = _shear_prefix_nkey(reduced)
    n_lines = max(1, int(st.session_state.get(KS(nk, beam_id, sec_id), 1) or 1))
    parts = []
    for i in range(n_lines):
        typ = str(st.session_state.get(KS(f"{prefix}{i}_type", beam_id, sec_id), "Étriers (2 brins)"))
        n_cadres = int(st.session_state.get(KS(f"{prefix}{i}_n", beam_id, sec_id), 1) or 1)
        diam = int(float(st.session_state.get(KS(f"{prefix}{i}_d", beam_id, sec_id), 8) or 8))
        parts.append(f"{n_cadres}× {typ} Ø{diam}")
    return " + ".join(parts)


def _delete_shear_line(beam_id: int, sec_id: int, reduced: bool, i: int):
    """Callback on_click : mutation légale des clés (avant instanciation des widgets)."""
    prefix, nk = _shear_prefix_nkey(reduced)
    n_lines = max(1, int(st.session_state.get(KS(nk, beam_id, sec_id), 1) or 1))
    if n_lines <= 1 or i <= 0 or i >= n_lines:
        return
    for j in range(i, n_lines - 1):
        for suf in ("type", "n", "d"):
            st.session_state[KS(f"{prefix}{j}_{suf}", beam_id, sec_id)] = st.session_state.get(
                KS(f"{prefix}{j+1}_{suf}", beam_id, sec_id)
            )
    for suf in ("type", "n", "d"):
        st.session_state.pop(KS(f"{prefix}{n_lines-1}_{suf}", beam_id, sec_id), None)
    st.session_state[KS(nk, beam_id, sec_id)] = n_lines - 1


def _add_shear_line(beam_id: int, sec_id: int, reduced: bool):
    prefix, nk = _shear_prefix_nkey(reduced)
    new_i = max(1, int(st.session_state.get(KS(nk, beam_id, sec_id), 1) or 1))
    st.session_state[KS(nk, beam_id, sec_id)] = new_i + 1
    st.session_state.setdefault(KS(f"{prefix}{new_i}_type", beam_id, sec_id), "Épingles (1 brin)")
    st.session_state.setdefault(KS(f"{prefix}{new_i}_n", beam_id, sec_id), 1)
    st.session_state.setdefault(KS(f"{prefix}{new_i}_d", beam_id, sec_id), 8)


# ============================================================
#  LITS : callbacks ajout / suppression
# ============================================================
def _add_lit(beam_id: int, sec_id: int, which: str):
    nk = KS(f"nlits_{which}", beam_id, sec_id)
    nl = _get_nlits(beam_id, sec_id, which)
    if nl >= MAX_LITS:
        return
    i = nl + 1
    _, prev_d = _lit_bars(beam_id, sec_id, which, nl)
    st.session_state.setdefault(KS(f"n_as_{which}_l{i}", beam_id, sec_id), 2)
    st.session_state[KS(f"ø_as_{which}_l{i}", beam_id, sec_id)] = prev_d
    st.session_state[KS(f"dist_{which}_l{i}_override", beam_id, sec_id)] = False
    st.session_state.pop(KS(f"dist_{which}_l{i}", beam_id, sec_id), None)  # sera recalculée en auto
    st.session_state[nk] = i


def _delete_lit(beam_id: int, sec_id: int, which: str, i: int):
    """Callback on_click : suppression du lit i (i>=2) avec décalage des suivants."""
    nk = KS(f"nlits_{which}", beam_id, sec_id)
    nl = _get_nlits(beam_id, sec_id, which)
    if i < 2 or i > nl:
        return
    for j in range(i, nl):
        for suf in ("n_as", "ø_as"):
            st.session_state[KS(f"{suf}_{which}_l{j}", beam_id, sec_id)] = st.session_state.get(
                KS(f"{suf}_{which}_l{j+1}", beam_id, sec_id)
            )
        st.session_state[KS(f"dist_{which}_l{j}", beam_id, sec_id)] = st.session_state.get(
            KS(f"dist_{which}_l{j+1}", beam_id, sec_id)
        )
        st.session_state[KS(f"dist_{which}_l{j}_override", beam_id, sec_id)] = bool(
            st.session_state.get(KS(f"dist_{which}_l{j+1}_override", beam_id, sec_id), False)
        )
    for suf in ("n_as", "ø_as", "dist"):
        st.session_state.pop(KS(f"{suf}_{which}_l{nl}", beam_id, sec_id), None)
    st.session_state.pop(KS(f"dist_{which}_l{nl}_override", beam_id, sec_id), None)
    st.session_state[nk] = nl - 1


# ============================================================
#  UI : SOLLICITATIONS PAR SECTION
# ============================================================
def _render_section_inputs(beam_id: int, sec_id: int, disabled: bool):
    c1, c2 = st.columns(2)
    with c1:
        float_input_fr_simple("Moment inférieur M (kNm)", key=KS("M_inf", beam_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c2:
        float_input_fr_simple("Effort tranchant V (kN)", key=KS("V", beam_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)

    c3, c4 = st.columns(2)
    with c3:
        m_sup_toggle = st.checkbox("Ajouter un moment supérieur", key=KS("ajouter_moment_sup", beam_id, sec_id), disabled=disabled)
    with c4:
        v_red_toggle = st.checkbox("Ajouter un effort tranchant réduit", key=KS("ajouter_effort_reduit", beam_id, sec_id), disabled=disabled)

    # Les valeurs masquées sont conservées (et sauvegardées) mais ignorées
    # dans les calculs tant que la case est décochée.
    c5, c6 = st.columns(2)
    with c5:
        if m_sup_toggle:
            float_input_fr_simple("Moment supérieur M_sup (kNm)", key=KS("M_sup", beam_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)
    with c6:
        if v_red_toggle:
            float_input_fr_simple("Effort tranchant réduit V_réduit (kN)", key=KS("V_lim", beam_id, sec_id), default=0.0, min_value=0.0, disabled=disabled)


def render_solicitations_for_beam(beam_id: int, data_locked: bool = False):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    st.markdown("### Sollicitations")

    for sec in beam.get("sections", []):
        sec_id = int(sec.get("id"))
        sec_name_key = f"meta_b{beam_id}_nom_{sec_id}"
        st.session_state.setdefault(sec_name_key, sec.get("nom", f"Section {sec_id}"))

        with st.expander(st.session_state.get(sec_name_key, sec.get("nom", f"Section {sec_id}")), expanded=True):
            # En-tête compact : nom éditable directement + copier + supprimer
            cN, cC, cD = st.columns([6, 0.8, 0.8], vertical_alignment="center")
            with cN:
                st.text_input(
                    "Nom de la section",
                    key=sec_name_key,
                    disabled=data_locked,
                    label_visibility="collapsed",
                )
            with cC:
                st.button(
                    "📋",
                    key=f"copy_sec_btn_{beam_id}_{sec_id}",
                    help="Copier la section (toutes les données)",
                    use_container_width=True,
                    on_click=_copy_section,
                    args=(beam_id, sec_id),
                    disabled=data_locked,
                )
            with cD:
                if sec_id != 1:
                    st.button(
                        "🗑️",
                        key=f"del_sec_btn_{beam_id}_{sec_id}",
                        help="Supprimer la section",
                        use_container_width=True,
                        on_click=_delete_section,
                        args=(beam_id, sec_id),
                        disabled=data_locked,
                    )

            _render_section_inputs(beam_id, sec_id, disabled=data_locked)

    cA, cD2 = st.columns([3, 1.4])
    with cA:
        st.button(
            "Ajouter une section à vérifier",
            key=f"add_sec_btn_{beam_id}",
            on_click=_add_section,
            args=(beam_id,),
            disabled=data_locked,
        )
    with cD2:
        if beam_id != 1:
            st.button(
                "Supprimer la poutre",
                key=f"del_beam_btn_{beam_id}",
                on_click=_delete_beam,
                args=(beam_id,),
                disabled=data_locked,
                use_container_width=True,
            )


def render_caracteristiques_beam(beam_id: int):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)

    beam_name_key = f"meta_beam_nom_{beam_id}"
    st.session_state.setdefault(beam_name_key, beam.get("nom", f"Poutre {beam_id}"))

    lock_key = KB("lock_data", beam_id)
    statut_key = KB("statut", beam_id)
    if statut_key not in st.session_state:
        st.session_state[statut_key] = "Validé" if bool(st.session_state.get(lock_key, False)) else "En cours"
    st.session_state[lock_key] = (st.session_state.get(statut_key) == "Validé")

    data_locked = bool(st.session_state.get(lock_key, False))

    with st.expander(st.session_state.get(beam_name_key, beam.get("nom", f"Poutre {beam_id}")), expanded=True):
        t1, t2 = st.columns([6, 1.6], vertical_alignment="center")
        with t1:
            st.markdown("#### Caractéristiques de la poutre")
        with t2:
            st.selectbox("Statut", ["En cours", "Validé"], key=statut_key, label_visibility="collapsed")

        c1, c2, c3 = st.columns([2.6, 1.6, 1.4], vertical_alignment="center")
        with c1:
            st.text_input("Nom de la poutre", key=beam_name_key, disabled=data_locked)
        with c2:
            st.selectbox("Classe de béton", list(BETON_DATA.keys()), key=KB("beton", beam_id), disabled=data_locked)
        with c3:
            st.selectbox("Qualité acier (B)", [400, 500], key=KB("fyk", beam_id), disabled=data_locked)

        cB, cH, cE = st.columns(3)
        with cB:
            st.number_input("Larg. (cm)", min_value=5, max_value=200, step=1, key=KB("b", beam_id), disabled=data_locked)
        with cH:
            st.number_input("Haut. (cm)", min_value=5, max_value=300, step=1, key=KB("h", beam_id), disabled=data_locked)
        with cE:
            st.number_input("Enrob. béton (cm)", min_value=0.0, max_value=20.0, step=0.5, key=KB("enrobage_beton", beam_id), disabled=data_locked)

        render_solicitations_for_beam(beam_id, data_locked=data_locked)


# ============================================================
#  CALCUL DES ÉTATS D'UNE SECTION
# ============================================================
def _dimensionnement_compute_states(beam_id: int, sec_id: int, beton_data: dict):
    beton = str(st.session_state.get(KB("beton", beam_id), "C30/37"))
    if beton not in beton_data:
        beton = list(beton_data.keys())[0]
    fck_cube = beton_data[beton]["fck_cube"]
    alpha_b = beton_data[beton]["alpha_b"]

    # Méthode ancienne : fyd = fyk / 1.5 (volontaire — validé par Khiao)
    fyk, mu_ref = _get_fyk_and_mu_ref(beam_id)
    fyd = fyk / 1.5

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in beton_data[beton]:
        mu_key = "mu_a500" if "mu_a500" in beton_data[beton] else [k for k in beton_data[beton].keys() if k.startswith("mu_a")][0]
    mu_val = beton_data[beton][mu_key]

    b = float(st.session_state.get(KB("b", beam_id), 20))
    h = float(st.session_state.get(KB("h", beam_id), 40))

    # --- Lits : aire totale + centre de gravité (positions réelles) ---
    As_inf_total, e_cdg_inf, inf_detail = _layers_geometry(beam_id, sec_id, "inf")
    As_sup_total, e_cdg_sup, sup_detail = _layers_geometry(beam_id, sec_id, "sup")

    # Distance lit 1 (pour hmin et cisaillement — logique conservée)
    dist_l1_inf = _get_dist_lit(beam_id, sec_id, "inf", 1)
    dist_l1_sup = _get_dist_lit(beam_id, sec_id, "sup", 1)

    # d utile FLEXION : c.d.g. des lits (inf. depuis le bas, sup. depuis le haut)
    d_utile_inf = h - e_cdg_inf  # cm
    d_utile_sup = h - e_cdg_sup  # cm
    # d utile CISAILLEMENT : inchangé (lit 1, min des deux faces)
    d_utile_for_shear = h - min(dist_l1_inf, dist_l1_sup)  # cm

    geom_inf_ok = d_utile_inf > 0.0
    geom_sup_ok = d_utile_sup > 0.0
    geom_shear_ok = d_utile_for_shear > 0.0
    d_calc_inf = max(d_utile_inf, 0.1)
    d_calc_sup = max(d_utile_sup, 0.1)
    d_calc_shear = max(d_utile_for_shear, 0.1)

    tol_tau = float(st.session_state.get("tau_tolerance_percent", 0.0) or 0.0)

    has_Msup = bool(st.session_state.get(KS("ajouter_moment_sup", beam_id, sec_id), False))
    has_Vred = bool(st.session_state.get(KS("ajouter_effort_reduit", beam_id, sec_id), False))

    M_inf_val = float(st.session_state.get(KS("M_inf", beam_id, sec_id), 0.0) or 0.0)
    M_sup_val = float(st.session_state.get(KS("M_sup", beam_id, sec_id), 0.0) or 0.0) if has_Msup else 0.0
    V_val = float(st.session_state.get(KS("V", beam_id, sec_id), 0.0) or 0.0)
    V_lim_val = float(st.session_state.get(KS("V_lim", beam_id, sec_id), 0.0) or 0.0) if has_Vred else 0.0
    has_Vlim = has_Vred and (V_lim_val > 0)

    # --- Hauteur (logique conservée : distance lit 1 inf.) ---
    M_max = max(M_inf_val, M_sup_val)
    if M_max > 0:
        hmin_calc = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10  # cm
    else:
        hmin_calc = 0.0
    etat_h = "ok" if (hmin_calc + dist_l1_inf <= h) else "nok"

    # --- As min/max ---
    As_min_formula = 0.0013 * b * h * 1e2  # mm²
    As_max = 0.04 * b * h * 1e2  # mm²

    As_formule_inf = (M_inf_val * 1e6) / (fyd * 0.9 * d_calc_inf * 10) if M_inf_val > 0 else 0.0
    As_formule_sup = (M_sup_val * 1e6) / (fyd * 0.9 * d_calc_sup * 10) if M_sup_val > 0 else 0.0

    As_min_inf_eff = max(As_min_formula, 0.25 * As_formule_sup)
    As_min_sup_eff = max(As_min_formula, 0.25 * As_formule_inf)

    As_req_inf_final = As_formule_inf
    As_req_sup_final = As_formule_sup

    # As,min effectif CONTRAIGNANT dans le statut
    etat_inf = "ok" if (geom_inf_ok and As_inf_total >= max(As_req_inf_final, As_min_inf_eff) and As_inf_total <= As_max) else "nok"
    etat_sup = "ok" if (geom_sup_ok and As_sup_total >= max(As_req_sup_final, As_min_sup_eff) and As_sup_total <= As_max) else "nok"

    # --- Tranchant : τ = V / (0.75·b·h) (inchangé) ---
    tau_1 = 0.016 * fck_cube / 1.05
    tau_2 = 0.032 * fck_cube / 1.05
    tau_4 = 0.064 * fck_cube / 1.05

    def _shear_need(tau):
        if tau <= tau_1:
            return "ok", tau_1
        if tau <= tau_2:
            return "ok", tau_2
        if tau <= tau_4:
            return "warn", tau_4
        return "nok", tau_4

    if V_val > 0:
        tau = V_val * 1e3 / (0.75 * b * h * 100)
        etat_tau_base, tau_lim = _shear_need(tau)
        if tau > tau_lim:
            etat_tau, _ = _status_with_tolerance(tau, tau_lim, tol_tau)
        else:
            etat_tau = etat_tau_base
        if not geom_shear_ok:
            etat_tau = "nok"
    else:
        etat_tau = "ok"

    def _pas_state(V_kn: float, pas_key: str, reduced: bool):
        pas = float(st.session_state.get(KS(pas_key, beam_id, sec_id), 30.0) or 30.0)
        Ast_e = _shear_lines_total_Ast_mm2(beam_id, sec_id, reduced=reduced)
        # s = Ast·fyd·d / V  (d en mm, V en N) -> résultat en mm, puis /10 -> cm
        pas_th = Ast_e * fyd * (d_calc_shear * 10.0) / (V_kn * 1e3) / 10.0
        s_max = min(0.75 * d_calc_shear, 30.0)
        pas_lim = min(pas_th, s_max)
        etat, _ = _status_with_tolerance(pas, pas_lim, tol_tau)
        if not geom_shear_ok:
            etat = "nok"
        return etat

    etat_pas = _pas_state(V_val, "shear_pas", reduced=False) if V_val > 0 else "ok"

    if has_Vlim:
        tau_r = V_lim_val * 1e3 / (0.75 * b * h * 100)
        etat_tau_r_base, tau_lim_r = _shear_need(tau_r)
        if tau_r > tau_lim_r:
            etat_tau_r, _ = _status_with_tolerance(tau_r, tau_lim_r, tol_tau)
        else:
            etat_tau_r = etat_tau_r_base
        if not geom_shear_ok:
            etat_tau_r = "nok"
        etat_pas_r = _pas_state(V_lim_val, "shear_pas_r", reduced=True)
    else:
        etat_tau_r = "ok"
        etat_pas_r = "ok"

    etat_global = _status_merge(etat_h, etat_inf, etat_sup, etat_tau, etat_pas, etat_tau_r, etat_pas_r)

    return {
        "etat_global": etat_global,
        "etat_h": etat_h,
        "etat_inf": etat_inf,
        "etat_sup": etat_sup,
        "etat_tau": etat_tau,
        "etat_pas": etat_pas,
        "etat_tau_r": etat_tau_r,
        "etat_pas_r": etat_pas_r,
        "has_Vlim": has_Vlim,
        "M_inf_val": M_inf_val,
        "M_sup_val": M_sup_val,
        "V_val": V_val,
        "V_lim_val": V_lim_val,
        "hmin_calc": hmin_calc,
        "tau_1": tau_1,
        "tau_2": tau_2,
        "tau_4": tau_4,
        "fyd": fyd,
        "beton": beton,
        "b": b,
        "h": h,
        "dist_l1_inf": dist_l1_inf,
        "dist_l1_sup": dist_l1_sup,
        "e_cdg_inf": e_cdg_inf,
        "e_cdg_sup": e_cdg_sup,
        "As_min_formula": As_min_formula,
        "As_max": As_max,
        "As_formule_inf": As_formule_inf,
        "As_formule_sup": As_formule_sup,
        "As_min_inf_eff": As_min_inf_eff,
        "As_min_sup_eff": As_min_sup_eff,
        "As_req_inf_final": As_req_inf_final,
        "As_req_sup_final": As_req_sup_final,
        "As_inf_total": As_inf_total,
        "As_sup_total": As_sup_total,
        "inf_detail": inf_detail,
        "sup_detail": sup_detail,
        "d_utile_inf": d_utile_inf,
        "d_utile_sup": d_utile_sup,
        "d_utile_shear": d_utile_for_shear,
        "geom_inf_ok": geom_inf_ok,
        "geom_sup_ok": geom_sup_ok,
        "geom_shear_ok": geom_shear_ok,
    }


# ============================================================
#  UI : CISAILLEMENT (lignes)
# ============================================================
def _render_shear_lines_ui(beam_id: int, sec_id: int, reduced: bool, disabled: bool):
    if reduced:
        n_key = KS("shear_n_lines_r", beam_id, sec_id)
        pas_key = KS("shear_pas_r", beam_id, sec_id)
        prefix = "shear_r_line"
        add_btn_key = KS("btn_add_shear_line_r", beam_id, sec_id)
        del_btn_prefix = KS("btn_del_shear_line_r_", beam_id, sec_id)
        type_label = "Type (réduit)"
        pas_label = "Pas choisi (cm) (réduit)"
        diam_label = "Ø (mm) (réduit)"
        nb_label = "Nbr. cadres (réduit)"
    else:
        n_key = KS("shear_n_lines", beam_id, sec_id)
        pas_key = KS("shear_pas", beam_id, sec_id)
        prefix = "shear_line"
        add_btn_key = KS("btn_add_shear_line", beam_id, sec_id)
        del_btn_prefix = KS("btn_del_shear_line_", beam_id, sec_id)
        type_label = "Type"
        pas_label = "Pas choisi (cm)"
        diam_label = "Ø (mm)"
        nb_label = "Nbr. cadres"

    n_lines = max(1, int(st.session_state.get(n_key, 1) or 1))
    st.session_state[n_key] = n_lines

    for i in range(n_lines):
        st.session_state.setdefault(KS(f"{prefix}{i}_type", beam_id, sec_id), "Étriers (2 brins)" if i == 0 else "Épingles (1 brin)")
        st.session_state.setdefault(KS(f"{prefix}{i}_n", beam_id, sec_id), 1)
        st.session_state.setdefault(KS(f"{prefix}{i}_d", beam_id, sec_id), 8)

        c0, c1, c2, c3, c4 = st.columns([3, 2, 2, 2, 1], vertical_alignment="center")
        with c0:
            st.selectbox(
                type_label,
                ["Étriers (2 brins)", "Épingles (1 brin)", "Étriers (3 brins)"],
                key=KS(f"{prefix}{i}_type", beam_id, sec_id),
                label_visibility="visible" if i == 0 else "collapsed",
                disabled=disabled,
            )
        with c1:
            st.number_input(
                nb_label,
                min_value=1,
                max_value=8,
                step=1,
                key=KS(f"{prefix}{i}_n", beam_id, sec_id),
                label_visibility="visible" if i == 0 else "collapsed",
                disabled=disabled,
            )
        with c2:
            st.selectbox(
                diam_label,
                SHEAR_DIAM_OPTS,
                key=KS(f"{prefix}{i}_d", beam_id, sec_id),
                label_visibility="visible" if i == 0 else "collapsed",
                disabled=disabled,
            )
        with c3:
            if i == 0:
                float_input_fr_simple(pas_label, key=pas_key, default=30.0, min_value=1.0, disabled=disabled)
            else:
                st.markdown("")
        with c4:
            if i > 0:
                st.button(
                    "🗑️",
                    key=f"{del_btn_prefix}{i}",
                    use_container_width=True,
                    disabled=disabled,
                    on_click=_delete_shear_line,
                    args=(beam_id, sec_id, reduced, i),
                )

    st.button(
        "➕ Ajouter armature d'effort tranchant" + (" (réduit)" if reduced else ""),
        key=add_btn_key,
        use_container_width=True,
        disabled=disabled,
        on_click=_add_shear_line,
        args=(beam_id, sec_id, reduced),
    )


# ============================================================
#  UI : LITS SUPPLÉMENTAIRES (2 à 4)
# ============================================================
def _render_extra_lits_ui(beam_id: int, sec_id: int, which: str, disabled: bool):
    suffix = " (sup.)" if which == "sup" else " (inf.)"
    nl = _get_nlits(beam_id, sec_id, which)

    for i in range(2, nl + 1):
        st.session_state.setdefault(KS(f"n_as_{which}_l{i}", beam_id, sec_id), 2)
        st.session_state.setdefault(KS(f"ø_as_{which}_l{i}", beam_id, sec_id), 16)

        # La valeur auto a déjà été synchronisée (hors override) par les
        # calculs qui précèdent le rendu ; on la recalcule ici uniquement
        # pour détecter l'override après saisie.
        auto_i = _auto_dist_lit(beam_id, sec_id, which, i)
        key_val, key_ovr = _dist_keys(beam_id, sec_id, which, i)
        st.session_state.setdefault(key_val, float(auto_i))
        st.session_state.setdefault(key_ovr, False)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 0.35], vertical_alignment="bottom")
        with c1:
            st.number_input(
                f"Nb barres (lit {i}){suffix}",
                min_value=1,
                max_value=50,
                step=1,
                key=KS(f"n_as_{which}_l{i}", beam_id, sec_id),
                disabled=disabled,
            )
        with c2:
            st.selectbox(
                f"Ø (mm) (lit {i}){suffix}",
                DIAM_OPTS,
                key=KS(f"ø_as_{which}_l{i}", beam_id, sec_id),
                disabled=disabled,
            )
        with c3:
            val_d = st.number_input(
                f"Distance axe lit {i} / parement (cm){suffix}",
                min_value=0.0,
                max_value=300.0,
                step=0.5,
                key=key_val,
                disabled=disabled,
            )
            st.session_state[key_ovr] = bool(abs(float(val_d) - float(auto_i)) > 1e-6)
        with c4:
            st.button(
                "🗑️",
                key=KS(f"btn_del_lit_{which}_{i}", beam_id, sec_id),
                use_container_width=True,
                disabled=disabled,
                on_click=_delete_lit,
                args=(beam_id, sec_id, which, i),
            )

    if nl < MAX_LITS:
        st.button(
            f"➕ Ajouter un lit{suffix}",
            key=KS(f"btn_add_lit_{which}", beam_id, sec_id),
            use_container_width=True,
            disabled=disabled,
            on_click=_add_lit,
            args=(beam_id, sec_id, which),
        )


def _render_face_armatures(beam_id: int, sec_id: int, which: str, states: dict, dim_locked: bool, units_as: str):
    """Bloc 'Armatures inférieures' / 'Armatures supérieures' complet."""
    is_inf = (which == "inf")
    titre = "Armatures inférieures" if is_inf else "Armatures supérieures"
    suffix = "" if is_inf else " (sup.)"
    unit_as_txt = "mm²" if units_as == "mm²" else "cm²"

    As_total = states["As_inf_total"] if is_inf else states["As_sup_total"]
    detail = states["inf_detail"] if is_inf else states["sup_detail"]
    etat = states["etat_inf"] if is_inf else states["etat_sup"]
    As_req = states["As_req_inf_final"] if is_inf else states["As_req_sup_final"]
    As_min_eff = states["As_min_inf_eff"] if is_inf else states["As_min_sup_eff"]
    As_max = states["As_max"]
    geom_ok = states["geom_inf_ok"] if is_inf else states["geom_sup_ok"]
    nl = _get_nlits(beam_id, sec_id, which)

    As_disp = As_total if units_as == "mm²" else As_total / 100.0
    right = f"{detail} — As={As_disp:.2f} {unit_as_txt}"

    open_bloc_left_right(titre, right, etat)

    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        st.markdown(f"**Aₛ,req,{'inf' if is_inf else 'sup'} = {As_req:.0f} mm²**")
    with ca2:
        st.markdown(f"**Aₛ,min,{'inf' if is_inf else 'sup'} = {As_min_eff:.0f} mm²**")
    with ca3:
        st.markdown(f"**Aₛ,max = {As_max:.0f} mm²**")

    if not geom_ok:
        st.markdown("❌ **Position des lits incompatible avec la hauteur : d utile ≤ 0.**")

    # ---- Lit 1 ----
    key_val1, key_ovr1 = _dist_keys(beam_id, sec_id, which, 1)
    auto_e1 = _auto_dist_lit(beam_id, sec_id, which, 1)

    r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
    with r1c1:
        st.number_input(
            "Nb barres" + suffix,
            min_value=1,
            max_value=50,
            step=1,
            key=KS(f"n_as_{which}", beam_id, sec_id),
            disabled=dim_locked,
        )
    with r1c2:
        st.selectbox("Ø (mm)" + suffix, DIAM_OPTS, key=KS(f"ø_as_{which}", beam_id, sec_id), disabled=dim_locked)
    with r1c3:
        val_e = st.number_input(
            "Distance axe lit / parement (cm)" + suffix,
            min_value=0.0,
            max_value=300.0,
            step=0.5,
            key=key_val1,
            disabled=dim_locked,
        )
        # L'override est une clé non-widget : mutation légale après rendu.
        st.session_state[key_ovr1] = bool(abs(float(val_e) - float(auto_e1)) > 1e-6)

    # ---- Lits 2..4 ----
    _render_extra_lits_ui(beam_id, sec_id, which, disabled=dim_locked)

    # ---- Récap : choix + d utile réduit si plusieurs lits ----
    As_total2, e_cdg2, detail2 = _layers_geometry(beam_id, sec_id, which)
    As_disp2 = As_total2 if units_as == "mm²" else As_total2 / 100.0
    if nl > 1:
        d_eff2 = states["h"] - e_cdg2
        st.markdown(
            f"<div style='margin-top:6px;font-weight:600;'>Choix : {detail2} — ( {As_disp2:.2f} {unit_as_txt} ) — "
            f"d utile = {d_eff2:.1f} cm (c.d.g. des {nl} lits)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='margin-top:6px;font-weight:600;'>Choix : {detail2} — ( {As_disp2:.2f} {unit_as_txt} )</div>",
            unsafe_allow_html=True,
        )
    close_bloc()


# ============================================================
#  UI : DIMENSIONNEMENT D'UNE SECTION
# ============================================================
def render_dimensionnement_section(beam_id: int, sec_id: int, beton_data: dict):
    beam_locked = bool(st.session_state.get(KB("lock_data", beam_id), False))
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    sec = next(s for s in beam["sections"] if int(s.get("id")) == sec_id)
    sec_nom = str(st.session_state.get(f"meta_b{beam_id}_nom_{sec_id}", sec.get("nom", f"Section {sec_id}")))

    states = _dimensionnement_compute_states(beam_id, sec_id, beton_data)
    title = _status_icon_label(states["etat_global"], sec_nom)

    with st.expander(title, expanded=True if sec_id == 1 else False):
        if beam_locked:
            st.caption("Poutre validée — dimensionnement figé.")
        else:
            st.checkbox("Bloquer le dimensionnement de cette section", key=KS("lock_dim", beam_id, sec_id))

        dim_locked = bool(st.session_state.get(KS("lock_dim", beam_id, sec_id), False)) or beam_locked

        units_len = st.session_state.get("units_len", "cm")
        units_as = st.session_state.get("units_as", "mm²")
        tol_tau = float(st.session_state.get("tau_tolerance_percent", 0.0) or 0.0)

        beton = states["beton"]
        b = states["b"]
        h = states["h"]
        fyd = states["fyd"]
        hmin_calc = states["hmin_calc"]
        dist_l1_inf = states["dist_l1_inf"]
        V_val = states["V_val"]
        V_lim_val = states["V_lim_val"]

        # ---- Vérification de la hauteur ----
        if units_len == "mm":
            right_h = f"{beton} — {b*10:.0f}×{h*10:.0f} mm — hmin={hmin_calc*10:.0f} mm"
        else:
            right_h = f"{beton} — {b:.0f}×{h:.0f} cm — hmin={hmin_calc:.1f} cm"

        open_bloc_left_right("Vérification de la hauteur", right_h, states["etat_h"])
        if units_len == "mm":
            st.markdown(
                f"**h,min** = {hmin_calc*10:.0f} mm  \n"
                f"h,min + distance axe lit 1 (inf.) = {(hmin_calc + dist_l1_inf)*10:.0f} mm ≤ h = {h*10:.0f} mm"
            )
        else:
            st.markdown(
                f"**h,min** = {hmin_calc:.1f} cm  \n"
                f"h,min + distance axe lit 1 (inf.) = {hmin_calc + dist_l1_inf:.1f} cm ≤ h = {h:.1f} cm"
            )
        close_bloc()

        # ---- Armatures inférieures / supérieures ----
        _render_face_armatures(beam_id, sec_id, "inf", states, dim_locked, units_as)
        _render_face_armatures(beam_id, sec_id, "sup", states, dim_locked, units_as)

        # ---- Tranchant + étriers ----
        tau_1, tau_2, tau_4 = states["tau_1"], states["tau_2"], states["tau_4"]

        def _shear_need_text(tau):
            if tau <= tau_1:
                return "Pas besoin d’étriers", "ok", "τ_adm_I", tau_1
            if tau <= tau_2:
                return "Besoin d’étriers", "ok", "τ_adm_II", tau_2
            if tau <= tau_4:
                return "Besoin de barres inclinées et d’étriers", "warn", "τ_adm_IV", tau_4
            return "Pas acceptable", "nok", "τ_adm_IV", tau_4

        def _bloc_pas(V_kn: float, pas_key_base: str, reduced: bool, titre_tau: str, titre_pas: str, etat_pas_state: str):
            tau = V_kn * 1e3 / (0.75 * b * h * 100)
            besoin, etat_tau_base, nom_lim, tau_lim = _shear_need_text(tau)
            if tau > tau_lim:
                etat_tau, suffix = _status_with_tolerance(tau, tau_lim, tol_tau)
            else:
                etat_tau, suffix = etat_tau_base, ""

            open_bloc_left_right(titre_tau, f"τ={tau:.2f} ≤ {nom_lim}={tau_lim:.2f}", etat_tau)
            extra = f" — {suffix}" if suffix else ""
            st.markdown(f"τ = {tau:.2f} N/mm² ≤ {nom_lim} = {tau_lim:.2f} N/mm² → {besoin}{extra}")
            close_bloc()

            _render_shear_lines_ui(beam_id, sec_id, reduced=reduced, disabled=dim_locked)

            pas = float(st.session_state.get(KS(pas_key_base, beam_id, sec_id), 30.0) or 30.0)
            Ast_e = _shear_lines_total_Ast_mm2(beam_id, sec_id, reduced=reduced)
            d_sh = max(states["d_utile_shear"], 0.1)
            pas_th = Ast_e * fyd * (d_sh * 10.0) / (V_kn * 1e3) / 10.0  # cm
            s_max = min(0.75 * d_sh, 30.0)
            pas_lim = min(pas_th, s_max)
            _, suffix_pas = _status_with_tolerance(pas, pas_lim, tol_tau)

            right_et = f"pas={pas:.1f} ≤ min({pas_th:.1f},{s_max:.1f})={pas_lim:.1f} cm"
            open_bloc_left_right(titre_pas, right_et, etat_pas_state)
            a1, a2, a3 = st.columns([1, 1, 2])
            with a1:
                st.markdown(f"**Pas théorique = {pas_th:.1f} cm**")
            with a2:
                st.markdown(f"**Pas maximal = {s_max:.1f} cm**")
            with a3:
                if suffix_pas:
                    st.markdown(f"**{suffix_pas}**")
            st.caption(_shear_lines_summary(beam_id, sec_id, reduced=reduced))
            close_bloc()

        if V_val > 0:
            _bloc_pas(V_val, "shear_pas", False, "Vérification de l'effort tranchant", "Détermination des étriers", states["etat_pas"])

        if states["has_Vlim"]:
            _bloc_pas(V_lim_val, "shear_pas_r", True, "Vérification effort tranchant réduit", "Détermination étriers réduits", states["etat_pas_r"])


# ============================================================
#  UI : INFOS PROJET / PARAMÈTRES AVANCÉS
# ============================================================
def render_infos_projet():
    st.session_state.setdefault("chk_infos_projet", False)
    st.session_state.setdefault("nom_projet", "")
    st.session_state.setdefault("partie", "")
    st.session_state.setdefault("date", datetime.today().strftime("%d/%m/%Y"))
    st.session_state.setdefault("indice", "0")

    cT, cLbl, cChk = st.columns([6, 2.2, 0.8], vertical_alignment="center")
    with cT:
        st.markdown("### Informations sur le projet")
    with cLbl:
        st.markdown("<div style='text-align:right;font-style:italic;opacity:0.75;'>Ajouter</div>", unsafe_allow_html=True)
    with cChk:
        st.checkbox("Ajouter les informations du projet", key="chk_infos_projet", label_visibility="collapsed")

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
    _ensure_global_defaults()

    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Affichage longueurs", ["cm", "mm"], key="units_len")
    with c2:
        st.selectbox("Affichage armatures", ["mm²", "cm²"], key="units_as")

    c3, c4, c5 = st.columns(3)
    with c3:
        st.selectbox("Diamètre étrier (mm)", [6, 8, 10, 12], key="diam_etrier_mm")
    with c4:
        st.number_input("Jeu premier lit (cm)", min_value=0.0, step=0.5, key="jeu_enrobage_cm")
    with c5:
        st.number_input("Jeu entre lits (cm)", min_value=0.0, step=0.5, key="jeu_entre_lits_cm")

    st.slider("Tolérance dépassement (%)", min_value=0, max_value=50, key="tau_tolerance_percent")


# ============================================================
#  UI : COLONNES GAUCHE / DROITE
# ============================================================
def render_donnees_left(beton_data: dict):
    st.markdown("### Données")
    for b in st.session_state.beams:
        bid = int(b["id"])
        b["nom"] = str(st.session_state.get(f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}")))
        render_caracteristiques_beam(bid)

    st.button("➕ Ajouter une poutre", use_container_width=True, key="btn_add_beam_simple", on_click=_add_beam)


def render_dimensionnement_right(beton_data: dict):
    for b in st.session_state.beams:
        bid = int(b["id"])
        bnom = str(st.session_state.get(f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}")))
        b["nom"] = bnom

        sec_states = [
            _dimensionnement_compute_states(bid, int(s["id"]), beton_data)["etat_global"]
            for s in b.get("sections", [])
        ]
        beam_state = _status_merge(*sec_states) if sec_states else "ok"
        beam_label = _status_icon_label(beam_state, bnom)

        with st.expander(beam_label, expanded=True if bid == 1 else False):
            for s in b.get("sections", []):
                render_dimensionnement_section(bid, int(s["id"]), beton_data)


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
    _init_beams_if_needed()

    # FIX PERSISTANCE : épingler toutes les clés persistantes AVANT tout rendu.
    _pin_persistent_state()

    if "retour_accueil_demande" not in st.session_state:
        st.session_state.retour_accueil_demande = False

    if st.session_state.retour_accueil_demande:
        st.session_state.page = "Accueil"
        st.session_state.retour_accueil_demande = False
        st.rerun()

    st.markdown("## Poutre en béton armé")

    btn1, btn2, btn3, btn4, btn5 = st.columns(5)

    with btn1:
        if st.button("🏠 Accueil", use_container_width=True, key="btn_home"):
            st.session_state.retour_accueil_demande = True
            st.rerun()

    with btn2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="btn_reset"):
            _reset_module()

    with btn3:
        payload = _build_save_payload()
        st.download_button(
            label="💾 Enregistrer",
            data=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="poutre_ba.json",
            mime="application/json",
            use_container_width=True,
            key="btn_save_dl",
        )

    with btn4:
        if st.button("📂 Ouvrir", use_container_width=True, key="btn_open_toggle"):
            st.session_state["show_open_uploader"] = not st.session_state.get("show_open_uploader", False)

        if st.session_state.get("show_open_uploader", False):
            uploaded = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed", key="open_uploader")
            if uploaded is not None:
                try:
                    data = json.load(uploaded)
                    if not isinstance(data, dict):
                        raise ValueError("Structure JSON inattendue")
                    _load_from_payload(data)
                    st.session_state["show_open_uploader"] = False
                    st.rerun()
                except Exception:
                    st.error("Fichier invalide ou corrompu — chargement annulé.")

    with btn5:
        if st.button("📄 Générer PDF", use_container_width=True, key="btn_pdf"):
            from modules.export_pdf import generer_rapport_pdf

            infos = {
                "nom_projet": st.session_state.get("nom_projet", ""),
                "partie": st.session_state.get("partie", ""),
                "date": st.session_state.get("date", datetime.today().strftime("%d/%m/%Y")),
                "indice": st.session_state.get("indice", "0"),
            }

            try:
                fichier_pdf = generer_rapport_pdf(
                    beams=st.session_state.beams,
                    values=dict(st.session_state),
                    beton_data=beton_data,
                    infos=infos,
                )
                with open(fichier_pdf, "rb") as f:
                    st.session_state["pdf_bytes"] = f.read()
                st.success("✅ Note de calcul générée")
            except Exception as e:
                st.session_state.pop("pdf_bytes", None)
                st.error(f"Erreur lors de la génération du PDF : {e}")

        if st.session_state.get("pdf_bytes"):
            st.download_button(
                label="⬇️ Télécharger le rapport PDF",
                data=st.session_state["pdf_bytes"],
                file_name="note_de_calcul_poutre.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="btn_pdf_dl",
            )

    input_col_gauche, result_col_droite = st.columns([2, 3])

    with input_col_gauche:
        render_infos_projet()
        render_donnees_left(beton_data)

    with result_col_droite:
        st.session_state.setdefault("show_param_avances", False)

        cH1, cH2, cH3 = st.columns([18, 2.6, 0.6], vertical_alignment="center")
        with cH1:
            st.markdown("### Dimensionnement")
        with cH2:
            small_italic_label_right("Paramètres avancés")
        with cH3:
            st.checkbox("Afficher paramètres avancés", key="show_param_avances", label_visibility="collapsed")

        if bool(st.session_state.get("show_param_avances", False)):
            with st.container(border=True):
                render_parametres_avances()

        render_dimensionnement_right(beton_data)

import streamlit as st
from datetime import datetime
import json
import math
import re

# ============================================================
#  STYLES BLOCS
# ============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}
C_ICONES   = {"ok": "✅",       "warn": "⚠️",      "nok": "❌"}

def open_bloc(titre: str, etat: str = "ok"):
    st.markdown(
        f"""
        <div style="
            background-color:{C_COULEURS.get(etat,'#f6f6f6')};
            padding:12px 14px 10px 14px;
            border-radius:10px;
            border:1px solid #d9d9d9;
            margin:10px 0 12px 0;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;">
            <div style="font-weight:700;">{titre}</div>
            <div style="font-size:20px;line-height:1;">{C_ICONES.get(etat,'')}</div>
          </div>
        """,
        unsafe_allow_html=True
    )

def close_bloc():
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
#  UTILITAIRES SESSION / CLÉS
# ============================================================
SECTION_KEY_PREFIX_RE = re.compile(r"^sec(\d+)_(.+)$")

def _is_section_key(k: str) -> bool:
    return bool(SECTION_KEY_PREFIX_RE.match(k))

def _is_raw_key(k: str) -> bool:
    return k.endswith("_raw")

def K(base: str, sec_id: int) -> str:
    """Section 1 = clés historiques inchangées. Section N => secN_<base>."""
    return base if sec_id == 1 else f"sec{sec_id}_{base}"

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
def float_input_fr_simple(label, key, default=0.0, min_value=0.0):
    """
    Champ texte qui accepte virgule/point.
    - Stocke le float dans st.session_state[key]
    - Stocke le texte brut dans st.session_state[f"{key}_raw"]
    """
    current = float(st.session_state.get(key, default) or 0.0)
    raw_default = st.session_state.get(f"{key}_raw", f"{current:.2f}".replace(".", ","))
    raw = st.text_input(label, value=raw_default, key=f"{key}_raw")

    try:
        val = float(str(raw).strip().replace(",", "."))
    except Exception:
        val = current

    val = max(min_value, val)
    st.session_state[key] = float(val)
    return val

# ============================================================
#  SECTIONS : INIT / ADD / DELETE
# ============================================================
SECTION_TYPES = [
    "Standard",
    "Travée",
    "Sur appui",
    "Extrémité",
    "Vérif recouvrement (à venir)",
    "Vérif second lit (à venir)",
]

def _init_sections_if_needed():
    if "sections" not in st.session_state or not isinstance(st.session_state.sections, list) or len(st.session_state.sections) == 0:
        st.session_state.sections = [{"id": 1, "nom": "Section A", "type": "Standard"}]

    # force présence section 1
    if not any(int(s.get("id", 0)) == 1 for s in st.session_state.sections):
        st.session_state.sections.insert(0, {"id": 1, "nom": "Section A", "type": "Standard"})

    # normalisation
    for s in st.session_state.sections:
        s["id"] = int(s.get("id", 0))
        s["nom"] = str(s.get("nom", f"Section {s['id']}"))
        s["type"] = str(s.get("type", "Standard"))

def _next_section_id() -> int:
    ids = [int(s.get("id", 0)) for s in st.session_state.sections]
    return (max(ids) + 1) if ids else 1

def _add_section():
    new_id = _next_section_id()
    st.session_state.sections.append({"id": new_id, "nom": f"Section {new_id}", "type": "Standard"})

def _delete_section(sec_id: int):
    if sec_id == 1:
        return
    st.session_state.sections = [s for s in st.session_state.sections if int(s.get("id")) != sec_id]

    prefix = f"sec{sec_id}_"
    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(prefix)]
    for k in keys_to_delete:
        del st.session_state[k]

# ============================================================
#  SAVE / LOAD JSON (sections + valeurs)
# ============================================================
BASE_SAVE_KEYS = {
    # infos projet
    "nom_projet", "partie", "date", "indice",
    # matériaux / géométrie
    "beton", "b", "h", "enrobage",
    # paramètres avancés
    "units_len", "units_as", "tau_tolerance_percent",
    "acier_non_standard", "fyk", "fyk_custom", "fyk_ref_for_mu",
    # SECTION 1 (clés historiques)
    "M_inf", "ajouter_moment_sup", "M_sup",
    "V", "ajouter_effort_reduit", "V_lim",
    "n_as_inf", "ø_as_inf", "n_as_sup", "ø_as_sup",
    "n_etriers", "ø_etrier", "pas_etrier", "type_etrier",
    "n_etriers_r", "ø_etrier_r", "pas_etrier_r", "type_etrier_r",
    # SECOND LIT (section 1)
    "ajouter_second_lit_inf", "n_as_inf_2", "ø_as_inf_2", "jeu_inf_2",
    "ajouter_second_lit_sup", "n_as_sup_2", "ø_as_sup_2", "jeu_sup_2",
    # UI
    "show_open_uploader",
}

def _build_save_payload():
    sections = [
        {"id": int(s.get("id")), "nom": str(s.get("nom")), "type": str(s.get("type", "Standard"))}
        for s in st.session_state.sections
    ]

    values = {}

    # base + raw associés
    for k in BASE_SAVE_KEYS:
        if k in st.session_state:
            values[k] = st.session_state[k]
        rk = f"{k}_raw"
        if rk in st.session_state:
            values[rk] = st.session_state[rk]

    # toutes les clés secN_* + raw
    for k in list(st.session_state.keys()):
        if _is_section_key(k) or (_is_raw_key(k) and _is_section_key(k[:-4])):
            values[k] = st.session_state[k]

    return {"sections": sections, "values": values}

def _load_from_payload(payload: dict):
    sections = payload.get("sections", None)
    values = payload.get("values", {})

    if isinstance(sections, list) and len(sections) > 0:
        cleaned = []
        for s in sections:
            try:
                sid = int(s.get("id"))
            except Exception:
                continue
            cleaned.append({
                "id": sid,
                "nom": str(s.get("nom", f"Section {sid}")),
                "type": str(s.get("type", "Standard")),
            })
        st.session_state.sections = cleaned if cleaned else [{"id": 1, "nom": "Section A", "type": "Standard"}]
    else:
        st.session_state.sections = [{"id": 1, "nom": "Section A", "type": "Standard"}]

    if isinstance(values, dict):
        for k, v in values.items():
            if (k in BASE_SAVE_KEYS) or _is_section_key(k) or (_is_raw_key(k) and _is_section_key(k[:-4])):
                st.session_state[k] = v

# ============================================================
#  OUTILS CALCUL
# ============================================================
def _bar_area_mm2(diam_mm: float) -> float:
    return math.pi * (diam_mm / 2.0) ** 2

def _status_merge(*states: str) -> str:
    # priorité: nok > warn > ok
    if any(s == "nok" for s in states):
        return "nok"
    if any(s == "warn" for s in states):
        return "warn"
    return "ok"

def _status_icon_label(state: str, label: str) -> str:
    # "vert" demandé => on utilise pastilles colorées (visibles même expander fermé)
    if state == "ok":
        return f"🟢 {label}"
    if state == "warn":
        return f"🟡 {label}"
    return f"🔴 {label}"

def _status_with_tolerance(value: float, limit: float, tol_percent: float):
    """
    Retourne (etat, msg_suffix)
    - ok: value <= limit
    - warn: value <= limit*(1+tol)
    - nok: au-delà
    """
    if limit <= 0:
        return "nok", ""
    if value <= limit:
        return "ok", ""
    lim2 = limit * (1.0 + max(0.0, tol_percent) / 100.0)
    if value <= lim2:
        return "warn", f"(dépassement toléré +{tol_percent:.0f}%)"
    return "nok", ""

def _get_fyk_and_mu_ref():
    """
    - fyk standard: 400/500 via st.session_state["fyk"]
    - acier non standard: fyk_custom, et mu_ref_for_mu=400/500 pour lire mu_aXXX dans la DB béton
    """
    acier_non_standard = bool(st.session_state.get("acier_non_standard", False))
    if acier_non_standard:
        fyk = float(st.session_state.get("fyk_custom", 500.0) or 500.0)
        mu_ref = str(st.session_state.get("fyk_ref_for_mu", "500"))
        return fyk, mu_ref
    else:
        fyk = float(st.session_state.get("fyk", "500") or 500.0)
        return fyk, str(int(fyk))

def _as_total_with_optional_second_layer(sec_id: int, which: str):
    """
    which = "inf" or "sup"
    Retourne (As_total_mm2, detail_str)
    """
    if which == "inf":
        n1 = int(st.session_state.get(K("n_as_inf", sec_id), 2) or 2)
        d1 = int(st.session_state.get(K("ø_as_inf", sec_id), 16) or 16)
        has2 = bool(st.session_state.get(K("ajouter_second_lit_inf", sec_id), False))
        n2 = int(st.session_state.get(K("n_as_inf_2", sec_id), 2) or 2)
        d2 = int(st.session_state.get(K("ø_as_inf_2", sec_id), d1) or d1)
        jeu = float(st.session_state.get(K("jeu_inf_2", sec_id), 0.0) or 0.0)
    else:
        n1 = int(st.session_state.get(K("n_as_sup", sec_id), 2) or 2)
        d1 = int(st.session_state.get(K("ø_as_sup", sec_id), 16) or 16)
        has2 = bool(st.session_state.get(K("ajouter_second_lit_sup", sec_id), False))
        n2 = int(st.session_state.get(K("n_as_sup_2", sec_id), 2) or 2)
        d2 = int(st.session_state.get(K("ø_as_sup_2", sec_id), d1) or d1)
        jeu = float(st.session_state.get(K("jeu_sup_2", sec_id), 0.0) or 0.0)

    As1 = n1 * _bar_area_mm2(d1)
    if has2:
        As2 = n2 * _bar_area_mm2(d2)
        AsT = As1 + As2
        detail = f"{n1}Ø{d1} + {n2}Ø{d2} (jeu {jeu:.1f} cm)"
    else:
        AsT = As1
        detail = f"{n1}Ø{d1}"
    return AsT, detail

# ============================================================
#  UI : SOLLICITATIONS PAR SECTION (UNIFORME)
# ============================================================
def _render_section_inputs(sec_id: int):
    cmom, cev = st.columns(2)

    with cmom:
        float_input_fr_simple("Moment inférieur M (kNm)", key=K("M_inf", sec_id), default=0.0, min_value=0.0)

        m_sup_toggle = st.checkbox(
            "Ajouter un moment supérieur",
            key=K("ajouter_moment_sup", sec_id),
            value=st.session_state.get(K("ajouter_moment_sup", sec_id), False)
        )
        if m_sup_toggle:
            float_input_fr_simple("Moment supérieur M_sup (kNm)", key=K("M_sup", sec_id), default=0.0, min_value=0.0)
        else:
            st.session_state[K("M_sup", sec_id)] = 0.0

    with cev:
        float_input_fr_simple("Effort tranchant V (kN)", key=K("V", sec_id), default=0.0, min_value=0.0)

        v_sup = st.checkbox(
            "Ajouter un effort tranchant réduit",
            key=K("ajouter_effort_reduit", sec_id),
            value=st.session_state.get(K("ajouter_effort_reduit", sec_id), False)
        )
        if v_sup:
            float_input_fr_simple("Effort tranchant réduit V_réduit (kN)", key=K("V_lim", sec_id), default=0.0, min_value=0.0)
        else:
            st.session_state[K("V_lim", sec_id)] = 0.0

def render_solicitations_all_sections():
    st.markdown("### Sollicitations")

    # Toutes les sections en expander (y compris section A)
    for s in st.session_state.sections:
        sid = int(s.get("id"))
        sec_nom = str(s.get("nom", f"Section {sid}"))
        sec_type = str(s.get("type", "Standard"))

        # section 1 ouverte par défaut, les autres fermées
        expanded_default = True if sid == 1 else False

        with st.expander(f"{sec_nom}", expanded=expanded_default):
            c1, c2, c3 = st.columns([3, 2, 1])

            with c1:
                new_nom = st.text_input("Nom de la section", value=sec_nom, key=f"meta_nom_{sid}")
            with c2:
                idx = SECTION_TYPES.index(sec_type) if sec_type in SECTION_TYPES else 0
                new_type = st.selectbox("Type", SECTION_TYPES, index=idx, key=f"meta_type_{sid}")
            with c3:
                if sid > 1:
                    if st.button("🗑️", key=f"del_sec_{sid}", use_container_width=True):
                        _delete_section(sid)
                        st.rerun()
                else:
                    st.markdown("")

            s["nom"] = new_nom
            s["type"] = new_type

            _render_section_inputs(sid)

    # Bouton TOUJOURS en dessous de tout
    st.markdown("---")
    if st.button("➕ Ajouter une section à vérifier", use_container_width=True, key="btn_add_section"):
        _add_section()
        st.rerun()

# ============================================================
#  DIMENSIONNEMENT PAR SECTION + STATUT GLOBAL
# ============================================================
def _dimensionnement_compute_states(sec_id: int, beton_data: dict):
    """
    Calcule les états (ok/warn/nok) sans rendre l'UI.
    Retourne: dict states + values utiles.
    """
    beton = st.session_state["beton"]
    fck_cube  = beton_data[beton]["fck_cube"]
    alpha_b   = beton_data[beton]["alpha_b"]

    fyk, mu_ref = _get_fyk_and_mu_ref()
    fyd = fyk / 1.5

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in beton_data[beton]:
        mu_key = "mu_a500" if "mu_a500" in beton_data[beton] else [k for k in beton_data[beton].keys() if k.startswith("mu_a")][0]
    mu_val = beton_data[beton][mu_key]

    b = float(st.session_state["b"])
    h = float(st.session_state["h"])
    enrobage = float(st.session_state["enrobage"])
    d_utile = h - enrobage  # cm

    tol_tau = float(st.session_state.get("tau_tolerance_percent", 0.0) or 0.0)

    M_inf_val = float(st.session_state.get(K("M_inf", sec_id), 0.0) or 0.0)
    M_sup_val = float(st.session_state.get(K("M_sup", sec_id), 0.0) or 0.0)
    V_val     = float(st.session_state.get(K("V", sec_id), 0.0) or 0.0)
    V_lim_val = float(st.session_state.get(K("V_lim", sec_id), 0.0) or 0.0)
    has_Vlim  = bool(st.session_state.get(K("ajouter_effort_reduit", sec_id), False)) and (V_lim_val > 0)

    # hauteur
    M_max = max(M_inf_val, M_sup_val)
    if M_max > 0:
        hmin_calc = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10  # cm
    else:
        hmin_calc = 0.0
    etat_h = "ok" if (hmin_calc + enrobage <= h) else "nok"

    # As min/max
    As_min_formula = 0.0013 * b * h * 1e2  # mm²
    As_max         = 0.04   * b * h * 1e2  # mm²

    As_req_inf = (M_inf_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_inf_val > 0 else 0.0
    As_req_sup = (M_sup_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_sup_val > 0 else 0.0
    As_min_inf_eff = max(As_min_formula, 0.25 * As_req_sup)
    As_min_sup_eff = max(As_min_formula, 0.25 * As_req_inf)

    As_inf_total, _ = _as_total_with_optional_second_layer(sec_id, "inf")
    As_sup_total, _ = _as_total_with_optional_second_layer(sec_id, "sup")
    etat_inf = "ok" if (As_inf_total >= max(As_req_inf, As_min_inf_eff) and As_inf_total <= As_max) else "nok"
    etat_sup = "ok" if (As_sup_total >= max(As_req_sup, As_min_sup_eff) and As_sup_total <= As_max) else "nok"

    # Tranchant
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

    # V
    if V_val > 0:
        tau = V_val * 1e3 / (0.75 * b * h * 100)
        etat_tau_base, tau_lim = _shear_need(tau)
        if tau > tau_lim:
            etat_tau, _ = _status_with_tolerance(tau, tau_lim, tol_tau)
        else:
            etat_tau = etat_tau_base
    else:
        etat_tau = "ok"

    # pas d'étriers V
    if V_val > 0:
        typ = str(st.session_state.get(K("type_etrier", sec_id), "Étriers (2 brins)"))
        brins = 2 if "2 brins" in typ else 1
        n_et = int(st.session_state.get(K("n_etriers", sec_id), 1) or 1)
        d_et = int(st.session_state.get(K("ø_etrier", sec_id), 8) or 8)
        pas = float(st.session_state.get(K("pas_etrier", sec_id), 30.0) or 30.0)

        Ast_e = n_et * brins * _bar_area_mm2(d_et)  # mm²
        pas_th = Ast_e * fyd * d_utile * 10 / (10 * V_val * 1e3)
        s_max = min(0.75 * d_utile, 30.0)
        pas_lim = min(pas_th, s_max)
        etat_pas, _ = _status_with_tolerance(pas, pas_lim, tol_tau)
    else:
        etat_pas = "ok"

    # V_lim
    if has_Vlim:
        tau_r = V_lim_val * 1e3 / (0.75 * b * h * 100)
        etat_tau_r_base, tau_lim_r = _shear_need(tau_r)
        if tau_r > tau_lim_r:
            etat_tau_r, _ = _status_with_tolerance(tau_r, tau_lim_r, tol_tau)
        else:
            etat_tau_r = etat_tau_r_base

        typ_r = str(st.session_state.get(K("type_etrier_r", sec_id), "Étriers (2 brins)"))
        brins_r = 2 if "2 brins" in typ_r else 1
        n_et_r = int(st.session_state.get(K("n_etriers_r", sec_id), 1) or 1)
        d_et_r = int(st.session_state.get(K("ø_etrier_r", sec_id), 8) or 8)
        pas_r  = float(st.session_state.get(K("pas_etrier_r", sec_id), 30.0) or 30.0)

        Ast_er = n_et_r * brins_r * _bar_area_mm2(d_et_r)
        pas_th_r = Ast_er * fyd * d_utile * 10 / (10 * V_lim_val * 1e3)
        s_max_r = min(0.75 * d_utile, 30.0)
        pas_lim_r = min(pas_th_r, s_max_r)
        etat_pas_r, _ = _status_with_tolerance(pas_r, pas_lim_r, tol_tau)
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
    }

def render_dimensionnement_section(sec_id: int, beton_data: dict):
    # --------- métadonnées section ----------
    sec = next(s for s in st.session_state.sections if int(s.get("id")) == sec_id)
    sec_nom = str(sec.get("nom", f"Section {sec_id}"))

    # --------- compute states (pour colorer le titre expander) ----------
    states = _dimensionnement_compute_states(sec_id, beton_data)
    etat_global = states["etat_global"]

    title = _status_icon_label(etat_global, sec_nom)
    expanded_default = True if sec_id == 1 else False

    with st.expander(title, expanded=expanded_default):
        # --------- paramètres matériaux / géométrie ----------
        beton = st.session_state["beton"]
        fck_cube  = beton_data[beton]["fck_cube"]
        alpha_b   = beton_data[beton]["alpha_b"]

        fyk, mu_ref = _get_fyk_and_mu_ref()
        fyd = fyk / 1.5

        mu_key = f"mu_a{mu_ref}"
        if mu_key not in beton_data[beton]:
            mu_key = "mu_a500" if "mu_a500" in beton_data[beton] else [k for k in beton_data[beton].keys() if k.startswith("mu_a")][0]
        mu_val = beton_data[beton][mu_key]

        b = float(st.session_state["b"])
        h = float(st.session_state["h"])
        enrobage = float(st.session_state["enrobage"])
        d_utile = h - enrobage  # cm

        # --------- options avancées d'affichage ----------
        units_len = st.session_state.get("units_len", "cm")  # "cm" ou "mm"
        units_as  = st.session_state.get("units_as", "mm²")  # "mm²" ou "cm²"
        tol_tau   = float(st.session_state.get("tau_tolerance_percent", 0.0) or 0.0)

        # --------- sollicitations ----------
        M_inf_val = float(st.session_state.get(K("M_inf", sec_id), 0.0) or 0.0)
        M_sup_val = float(st.session_state.get(K("M_sup", sec_id), 0.0) or 0.0)
        V_val     = float(st.session_state.get(K("V", sec_id), 0.0) or 0.0)
        V_lim_val = float(st.session_state.get(K("V_lim", sec_id), 0.0) or 0.0)

        # ---- Vérification de la hauteur ----
        M_max = max(M_inf_val, M_sup_val)
        if M_max > 0:
            hmin_calc = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10  # cm
        else:
            hmin_calc = 0.0

        open_bloc("Vérification de la hauteur", states["etat_h"])
        if units_len == "mm":
            st.markdown(
                f"**h,min** = {hmin_calc*10:.0f} mm  \n"
                f"h,min + enrobage = {(hmin_calc + enrobage)*10:.0f} mm ≤ h = {h*10:.0f} mm"
            )
        else:
            st.markdown(
                f"**h,min** = {hmin_calc:.1f} cm  \n"
                f"h,min + enrobage = {hmin_calc + enrobage:.1f} cm ≤ h = {h:.1f} cm"
            )
        close_bloc()

        # ---- Données section (communes) ----
        As_min_formula = 0.0013 * b * h * 1e2  # mm²
        As_max         = 0.04   * b * h * 1e2  # mm²

        As_req_inf = (M_inf_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_inf_val > 0 else 0.0
        As_req_sup = (M_sup_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_sup_val > 0 else 0.0

        As_min_inf_eff = max(As_min_formula, 0.25 * As_req_sup)
        As_min_sup_eff = max(As_min_formula, 0.25 * As_req_inf)

        unit_as_txt = "mm²" if units_as == "mm²" else "cm²"

        # ---- Armatures inférieures ----
        diam_opts = [6, 8, 10, 12, 16, 20, 25, 32, 40]
        open_bloc("Armatures inférieures", states["etat_inf"])
        ca1, ca2, ca3 = st.columns(3)
        with ca1: st.markdown(f"**Aₛ,req,inf = {As_req_inf:.0f} mm²**")
        with ca2: st.markdown(f"**Aₛ,min,inf = {As_min_inf_eff:.0f} mm²**")
        with ca3: st.markdown(f"**Aₛ,max = {As_max:.0f} mm²**")

        r1c1, r1c2 = st.columns([1, 1])
        with r1c1:
            st.number_input("Nb barres", min_value=1, max_value=50,
                            value=int(st.session_state.get(K("n_as_inf", sec_id), 2) or 2),
                            step=1, key=K("n_as_inf", sec_id))
        with r1c2:
            dcur = int(st.session_state.get(K("ø_as_inf", sec_id), 16) or 16)
            idx = diam_opts.index(dcur) if dcur in diam_opts else diam_opts.index(16)
            st.selectbox("Ø (mm)", diam_opts, index=idx, key=K("ø_as_inf", sec_id))

        has2 = st.checkbox("Ajouter un second lit (inf.)",
                           value=bool(st.session_state.get(K("ajouter_second_lit_inf", sec_id), False)),
                           key=K("ajouter_second_lit_inf", sec_id))
        if has2:
            s2c1, s2c2, s2c3 = st.columns([1, 1, 1])
            with s2c1:
                st.number_input("Nb barres (2e lit)", min_value=1, max_value=50,
                                value=int(st.session_state.get(K("n_as_inf_2", sec_id), 2) or 2),
                                step=1, key=K("n_as_inf_2", sec_id))
            with s2c2:
                dcur2 = int(st.session_state.get(K("ø_as_inf_2", sec_id), dcur) or dcur)
                idx2 = diam_opts.index(dcur2) if dcur2 in diam_opts else diam_opts.index(dcur)
                st.selectbox("Ø (mm) (2e lit)", diam_opts, index=idx2, key=K("ø_as_inf_2", sec_id))
            with s2c3:
                float_input_fr_simple("Jeu entre lits (cm)", key=K("jeu_inf_2", sec_id),
                                      default=float(st.session_state.get(K("jeu_inf_2", sec_id), 0.0) or 0.0),
                                      min_value=0.0)

        As_inf_total, inf_detail = _as_total_with_optional_second_layer(sec_id, "inf")
        As_inf_disp = As_inf_total if units_as == "mm²" else As_inf_total / 100.0
        st.markdown(
            f"<div style='margin-top:6px;font-weight:600;'>Choix : {inf_detail} — "
            f"( {As_inf_disp:.2f} {unit_as_txt} )</div>",
            unsafe_allow_html=True
        )
        close_bloc()

        # ---- Armatures supérieures ----
        open_bloc("Armatures supérieures", states["etat_sup"])
        cs1, cs2, cs3 = st.columns(3)
        with cs1: st.markdown(f"**Aₛ,req,sup = {As_req_sup:.0f} mm²**")
        with cs2: st.markdown(f"**Aₛ,min,sup = {As_min_sup_eff:.0f} mm²**")
        with cs3: st.markdown(f"**Aₛ,max = {As_max:.0f} mm²**")

        t1c1, t1c2 = st.columns([1, 1])
        with t1c1:
            st.number_input("Nb barres (sup.)", min_value=1, max_value=50,
                            value=int(st.session_state.get(K("n_as_sup", sec_id), 2) or 2),
                            step=1, key=K("n_as_sup", sec_id))
        with t1c2:
            dcurS = int(st.session_state.get(K("ø_as_sup", sec_id), 16) or 16)
            idxS = diam_opts.index(dcurS) if dcurS in diam_opts else diam_opts.index(16)
            st.selectbox("Ø (mm) (sup.)", diam_opts, index=idxS, key=K("ø_as_sup", sec_id))

        has2s = st.checkbox("Ajouter un second lit (sup.)",
                            value=bool(st.session_state.get(K("ajouter_second_lit_sup", sec_id), False)),
                            key=K("ajouter_second_lit_sup", sec_id))
        if has2s:
            u2c1, u2c2, u2c3 = st.columns([1, 1, 1])
            with u2c1:
                st.number_input("Nb barres (2e lit) (sup.)", min_value=1, max_value=50,
                                value=int(st.session_state.get(K("n_as_sup_2", sec_id), 2) or 2),
                                step=1, key=K("n_as_sup_2", sec_id))
            with u2c2:
                dcurS2 = int(st.session_state.get(K("ø_as_sup_2", sec_id), dcurS) or dcurS)
                idxS2 = diam_opts.index(dcurS2) if dcurS2 in diam_opts else diam_opts.index(dcurS)
                st.selectbox("Ø (mm) (2e lit) (sup.)", diam_opts, index=idxS2, key=K("ø_as_sup_2", sec_id))
            with u2c3:
                float_input_fr_simple("Jeu entre lits (cm) (sup.)", key=K("jeu_sup_2", sec_id),
                                      default=float(st.session_state.get(K("jeu_sup_2", sec_id), 0.0) or 0.0),
                                      min_value=0.0)

        As_sup_total, sup_detail = _as_total_with_optional_second_layer(sec_id, "sup")
        As_sup_disp = As_sup_total if units_as == "mm²" else As_sup_total / 100.0
        st.markdown(
            f"<div style='margin-top:6px;font-weight:600;'>Choix : {sup_detail} — "
            f"( {As_sup_disp:.2f} {unit_as_txt} )</div>",
            unsafe_allow_html=True
        )
        close_bloc()

        # ---- Tranchant + étriers ----
        tau_1 = 0.016 * fck_cube / 1.05
        tau_2 = 0.032 * fck_cube / 1.05
        tau_4 = 0.064 * fck_cube / 1.05

        def _shear_need_text(tau):
            if tau <= tau_1:
                return "Pas besoin d’étriers", "ok", "τ_adm_I", tau_1
            if tau <= tau_2:
                return "Besoin d’étriers", "ok", "τ_adm_II", tau_2
            if tau <= tau_4:
                return "Besoin de barres inclinées et d’étriers", "warn", "τ_adm_IV", tau_4
            return "Pas acceptable", "nok", "τ_adm_IV", tau_4

        if V_val > 0:
            tau = V_val * 1e3 / (0.75 * b * h * 100)
            besoin, etat_tau_base, nom_lim, tau_lim = _shear_need_text(tau)
            if tau > tau_lim:
                etat_tau, suffix = _status_with_tolerance(tau, tau_lim, tol_tau)
            else:
                etat_tau, suffix = etat_tau_base, ""
            open_bloc("Vérification de l'effort tranchant", etat_tau)
            extra = f" {suffix}" if suffix else ""
            st.markdown(f"τ = {tau:.2f} N/mm² ≤ {nom_lim} = {tau_lim:.2f} N/mm² → {besoin}{extra}")
            close_bloc()

            cE0, cE1, cE2, cE3 = st.columns([2, 2, 2, 2])
            with cE0:
                st.selectbox(
                    "Type",
                    ["Étriers (2 brins)", "Épingles (1 brin)"],
                    index=0 if str(st.session_state.get(K("type_etrier", sec_id), "Étriers (2 brins)")) == "Étriers (2 brins)" else 1,
                    key=K("type_etrier", sec_id),
                )
            with cE1:
                st.number_input("Nbr. cadres", min_value=1, max_value=8,
                                value=int(st.session_state.get(K("n_etriers", sec_id), 1) or 1),
                                step=1, key=K("n_etriers", sec_id))
            with cE2:
                diam_list = [6, 8, 10, 12]
                dcurE = int(st.session_state.get(K("ø_etrier", sec_id), 8) or 8)
                idxE = diam_list.index(dcurE) if dcurE in diam_list else diam_list.index(8)
                st.selectbox("Ø (mm)", diam_list, index=idxE, key=K("ø_etrier", sec_id))
            with cE3:
                float_input_fr_simple("Pas choisi (cm)", key=K("pas_etrier", sec_id),
                                      default=float(st.session_state.get(K("pas_etrier", sec_id), 30.0) or 30.0),
                                      min_value=1.0)

            typ = str(st.session_state.get(K("type_etrier", sec_id), "Étriers (2 brins)"))
            brins = 2 if "2 brins" in typ else 1
            n_et = int(st.session_state.get(K("n_etriers", sec_id), 1) or 1)
            d_et = int(st.session_state.get(K("ø_etrier", sec_id), 8) or 8)
            pas = float(st.session_state.get(K("pas_etrier", sec_id), 30.0) or 30.0)

            Ast_e = n_et * brins * _bar_area_mm2(d_et)  # mm²
            pas_th = Ast_e * fyd * d_utile * 10 / (10 * V_val * 1e3)
            s_max = min(0.75 * d_utile, 30.0)
            pas_lim = min(pas_th, s_max)
            etat_pas, suffix_pas = _status_with_tolerance(pas, pas_lim, tol_tau)

            open_bloc("Détermination des étriers", etat_pas)
            a1, a2, a3 = st.columns([1, 1, 2])
            with a1: st.markdown(f"**Pas théorique = {pas_th:.1f} cm**")
            with a2: st.markdown(f"**Pas maximal = {s_max:.1f} cm**")
            with a3:
                if suffix_pas:
                    st.markdown(f"**{suffix_pas}**")
            close_bloc()

        # ---- Tranchant réduit ----
        if bool(st.session_state.get(K("ajouter_effort_reduit", sec_id), False)) and V_lim_val > 0:
            tau_r = V_lim_val * 1e3 / (0.75 * b * h * 100)
            besoin_r, etat_r_base, nom_lim_r, tau_lim_r = _shear_need_text(tau_r)
            if tau_r > tau_lim_r:
                etat_r, suffix_r = _status_with_tolerance(tau_r, tau_lim_r, tol_tau)
            else:
                etat_r, suffix_r = etat_r_base, ""
            open_bloc("Vérification de l'effort tranchant réduit", etat_r)
            extra = f" {suffix_r}" if suffix_r else ""
            st.markdown(f"τ = {tau_r:.2f} N/mm² ≤ {nom_lim_r} = {tau_lim_r:.2f} N/mm² → {besoin_r}{extra}")
            close_bloc()

            cR0, cR1, cR2, cR3 = st.columns([2, 2, 2, 2])
            with cR0:
                st.selectbox(
                    "Type (réduit)",
                    ["Étriers (2 brins)", "Épingles (1 brin)"],
                    index=0 if str(st.session_state.get(K("type_etrier_r", sec_id), "Étriers (2 brins)")) == "Étriers (2 brins)" else 1,
                    key=K("type_etrier_r", sec_id),
                )
            with cR1:
                st.number_input("Nbr. cadres (réduit)", min_value=1, max_value=8,
                                value=int(st.session_state.get(K("n_etriers_r", sec_id), 1) or 1),
                                step=1, key=K("n_etriers_r", sec_id))
            with cR2:
                diam_list_r = [6, 8, 10, 12]
                dcurER = int(st.session_state.get(K("ø_etrier_r", sec_id), 8) or 8)
                idxER = diam_list_r.index(dcurER) if dcurER in diam_list_r else diam_list_r.index(8)
                st.selectbox("Ø (mm) (réduit)", diam_list_r, index=idxER, key=K("ø_etrier_r", sec_id))
            with cR3:
                float_input_fr_simple("Pas choisi (cm) (réduit)", key=K("pas_etrier_r", sec_id),
                                      default=float(st.session_state.get(K("pas_etrier_r", sec_id), 30.0) or 30.0),
                                      min_value=1.0)

            typ_r = str(st.session_state.get(K("type_etrier_r", sec_id), "Étriers (2 brins)"))
            brins_r = 2 if "2 brins" in typ_r else 1
            n_et_r = int(st.session_state.get(K("n_etriers_r", sec_id), 1) or 1)
            d_et_r = int(st.session_state.get(K("ø_etrier_r", sec_id), 8) or 8)
            pas_r  = float(st.session_state.get(K("pas_etrier_r", sec_id), 30.0) or 30.0)

            Ast_er = n_et_r * brins_r * _bar_area_mm2(d_et_r)
            pas_th_r = Ast_er * fyd * d_utile * 10 / (10 * V_lim_val * 1e3)
            s_max_r = min(0.75 * d_utile, 30.0)
            pas_lim_r = min(pas_th_r, s_max_r)
            etat_pas_r, suffix_pas_r = _status_with_tolerance(pas_r, pas_lim_r, tol_tau)

            open_bloc("Détermination des étriers réduits", etat_pas_r)
            b1, b2, b3 = st.columns([1, 1, 2])
            with b1: st.markdown(f"**Pas théorique = {pas_th_r:.1f} cm**")
            with b2: st.markdown(f"**Pas maximal = {s_max_r:.1f} cm**")
            with b3:
                if suffix_pas_r:
                    st.markdown(f"**{suffix_pas_r}**")
            close_bloc()

# ============================================================
#  SHOW()
# ============================================================
def show():
    _init_sections_if_needed()

    # ---------- État ----------
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "retour_accueil_demande" not in st.session_state:
        st.session_state.retour_accueil_demande = False

    if st.session_state.retour_accueil_demande:
        st.session_state.page = "Accueil"
        st.session_state.retour_accueil_demande = False
        st.rerun()

    st.markdown("## Poutre en béton armé")

    # ---------- Barre d’actions ----------
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
            key="btn_save_dl"
        )

    with btn4:
        if st.button("📂 Ouvrir", use_container_width=True, key="btn_open_toggle"):
            st.session_state["show_open_uploader"] = not st.session_state.get("show_open_uploader", False)

        if st.session_state.get("show_open_uploader", False):
            uploaded = st.file_uploader("Choisir un fichier JSON", type=["json"], label_visibility="collapsed", key="open_uploader")
            if uploaded is not None:
                data = json.load(uploaded)
                _load_from_payload(data)
                st.session_state["show_open_uploader"] = False
                st.success("Fichier chargé.")
                st.rerun()

    # Export PDF plus tard
    with btn5:
        st.button("📄 Générer PDF", use_container_width=True, key="btn_pdf_disabled", disabled=True)

    # ---------- Données béton ----------
    with open("beton_classes.json", "r", encoding="utf-8") as f:
        beton_data = json.load(f)

    input_col_gauche, result_col_droite = st.columns([2, 3])

    # ============================================================
    #  COLONNE GAUCHE
    # ============================================================
    with input_col_gauche:
        # -------- Paramètres avancés (déplacé AU-DESSUS) --------
        with st.expander("Paramètres avancés", expanded=False):
            # 3 lignes (pas en colonnes)
            st.selectbox("Affichage longueurs", ["cm", "mm"],
                         index=0 if st.session_state.get("units_len", "cm") == "cm" else 1,
                         key="units_len")
            st.selectbox("Affichage armatures", ["mm²", "cm²"],
                         index=0 if st.session_state.get("units_as", "mm²") == "mm²" else 1,
                         key="units_as")
            st.slider("Tolérance dépassement (%)", min_value=0, max_value=25,
                      value=int(st.session_state.get("tau_tolerance_percent", 0) or 0),
                      step=1, key="tau_tolerance_percent")

            st.markdown("---")
            st.checkbox("Qualité d'acier non standard", value=bool(st.session_state.get("acier_non_standard", False)),
                        key="acier_non_standard")
            if bool(st.session_state.get("acier_non_standard", False)):
                st.number_input("fyk custom [N/mm²]", min_value=200.0, max_value=2000.0,
                                value=float(st.session_state.get("fyk_custom", 500.0) or 500.0),
                                step=10.0, key="fyk_custom")
                st.selectbox("Référence mu (base béton)", ["400", "500"],
                             index=1 if str(st.session_state.get("fyk_ref_for_mu", "500")) == "500" else 0,
                             key="fyk_ref_for_mu")

        # -------- Informations sur le projet --------
        st.markdown("### Informations sur le projet")
        afficher_infos = st.checkbox("Ajouter les informations du projet", value=False, key="chk_infos_projet")
        if afficher_infos:
            st.text_input("", placeholder="Nom du projet", key="nom_projet")
            st.text_input("", placeholder="Partie", key="partie")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("", placeholder="Date (jj/mm/aaaa)",
                              value=st.session_state.get("date", datetime.today().strftime("%d/%m/%Y")), key="date")
            with c2:
                st.text_input("", placeholder="Indice", value=st.session_state.get("indice", "0"), key="indice")
        else:
            st.session_state.setdefault("date", datetime.today().strftime("%d/%m/%Y"))

        # -------- Caractéristiques (retour “comme avant”) --------
        st.markdown("### Caractéristiques de la poutre")
        cbet, cacier = st.columns(2)
        with cbet:
            options = list(beton_data.keys())
            default_beton = options[min(2, len(options)-1)]
            current_beton = st.session_state.get("beton", default_beton)
            st.selectbox("Classe de béton", options, index=options.index(current_beton), key="beton")

        with cacier:
            acier_non_standard = bool(st.session_state.get("acier_non_standard", False))
            if not acier_non_standard:
                acier_opts = ["400", "500"]
                cur_fyk = st.session_state.get("fyk", "500")
                st.selectbox("Qualité d'acier [N/mm²]", acier_opts,
                             index=acier_opts.index(cur_fyk) if cur_fyk in acier_opts else 1,
                             key="fyk")
                st.session_state.setdefault("fyk_custom", 500.0)
                st.session_state.setdefault("fyk_ref_for_mu", "500")
            else:
                # demandé: le champ fyk custom apparait ici quand non standard activé
                st.number_input("fyk (non standard) [N/mm²]", min_value=200.0, max_value=2000.0,
                                value=float(st.session_state.get("fyk_custom", 500.0) or 500.0),
                                step=10.0, key="fyk_custom")

        csec1, csec2, csec3 = st.columns(3)
        with csec1:
            st.number_input("Larg. [cm]", min_value=5, max_value=1000, value=st.session_state.get("b", 20), step=5, key="b")
        with csec2:
            st.number_input("Haut. [cm]", min_value=5, max_value=1000, value=st.session_state.get("h", 40), step=5, key="h")
        with csec3:
            st.number_input("Enrob. (cm)", min_value=0.0, max_value=100.0, value=st.session_state.get("enrobage", 5.0), step=0.5, key="enrobage")

        # -------- Sollicitations (uniforme) --------
        render_solicitations_all_sections()

    # ============================================================
    #  COLONNE DROITE
    # ============================================================
    with result_col_droite:
        st.markdown("### Dimensionnement")
        for s in st.session_state.sections:
            sid = int(s.get("id"))
            render_dimensionnement_section(sid, beton_data)

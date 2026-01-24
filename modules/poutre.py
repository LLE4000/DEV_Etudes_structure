# ===========================
#  PARTIE 1 / 2
#  poutre.py (Streamlit)
# ===========================
import streamlit as st
from datetime import datetime
import json
import math
import re
from copy import deepcopy

# ============================================================
#  STYLES BLOCS
# ============================================================
C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌"}


def open_bloc_left_right(left: str, right: str = "", etat: str = "ok"):
    """
    Header de bloc : texte à gauche + texte à droite (aligné contre l'icône à droite).
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


# ============================================================
#  UTILITAIRES SESSION / CLÉS
# ============================================================
SECTION_KEY_PREFIX_RE = re.compile(r"^b(\d+)_sec(\d+)_(.+)$")


def _is_beam_section_key(k: str) -> bool:
    return bool(SECTION_KEY_PREFIX_RE.match(k))


def _is_raw_key(k: str) -> bool:
    return k.endswith("_raw")


def KB(base: str, beam_id: int) -> str:
    return f"b{beam_id}_{base}"


def KS(base: str, beam_id: int, sec_id: int) -> str:
    return f"b{beam_id}_sec{sec_id}_{base}"


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
#  POUTRES / SECTIONS : INIT / ADD / DELETE / DUPLICATE
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


def _next_beam_id() -> int:
    ids = [int(b.get("id", 0)) for b in st.session_state.beams]
    return (max(ids) + 1) if ids else 1


def _next_section_id(beam_id: int) -> int:
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    ids = [int(s.get("id", 0)) for s in beam["sections"]]
    return (max(ids) + 1) if ids else 1


def _add_beam():
    new_id = _next_beam_id()
    st.session_state.beams.append({"id": new_id, "nom": f"Poutre {new_id}", "sections": [{"id": 1, "nom": "Section A"}]})


def _delete_beam(beam_id: int):
    if beam_id == 1:
        return
    st.session_state.beams = [b for b in st.session_state.beams if int(b.get("id")) != beam_id]
    prefix = f"b{beam_id}_"
    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(prefix)]
    for k in keys_to_delete:
        del st.session_state[k]


def _duplicate_beam(src_beam_id: int):
    src = next(b for b in st.session_state.beams if int(b.get("id")) == src_beam_id)
    new_id = _next_beam_id()
    new_beam = {"id": new_id, "nom": f"{src.get('nom','Poutre')} (copie)", "sections": deepcopy(src["sections"])}
    st.session_state.beams.append(new_beam)

    src_prefix = f"b{src_beam_id}_"
    dst_prefix = f"b{new_id}_"
    for k in list(st.session_state.keys()):
        if k.startswith(src_prefix):
            st.session_state[dst_prefix + k[len(src_prefix):]] = deepcopy(st.session_state[k])


def _add_section(beam_id: int):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    new_id = _next_section_id(beam_id)
    beam["sections"].append({"id": new_id, "nom": f"Section {new_id}"})


def _delete_section(beam_id: int, sec_id: int):
    if sec_id == 1:
        return
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    beam["sections"] = [s for s in beam["sections"] if int(s.get("id")) != sec_id]

    prefix = f"b{beam_id}_sec{sec_id}_"
    keys_to_delete = [k for k in list(st.session_state.keys()) if k.startswith(prefix)]
    for k in keys_to_delete:
        del st.session_state[k]


# ============================================================
#  SAVE / LOAD JSON (beams + valeurs)
# ============================================================
BASE_SAVE_KEYS_GLOBAL = {
    "units_len",
    "units_as",
    "tau_tolerance_percent",
    "show_open_uploader",
    "nom_projet",
    "partie",
    "date",
    "indice",
    "chk_infos_projet",
}


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
    for k in BASE_SAVE_KEYS_GLOBAL:
        if k in st.session_state:
            values[k] = st.session_state[k]
        rk = f"{k}_raw"
        if rk in st.session_state:
            values[rk] = st.session_state[rk]

    for k in list(st.session_state.keys()):
        if re.match(r"^b\d+_", k):
            values[k] = st.session_state[k]

    return {"beams": beams, "values": values}


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
            if k in BASE_SAVE_KEYS_GLOBAL or (k.endswith("_raw") and k[:-4] in BASE_SAVE_KEYS_GLOBAL):
                st.session_state[k] = v
                continue
            if re.match(r"^b\d+_", k):
                st.session_state[k] = v

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
    if limit <= 0:
        return "nok", ""
    if value <= limit:
        return "ok", ""
    lim2 = limit * (1.0 + max(0.0, tol_percent) / 100.0)
    if value <= lim2:
        return "warn", f"(dépassement toléré +{tol_percent:.0f}%)"
    return "nok", ""


def _brins_from_type(type_txt: str) -> int:
    if "3 brins" in type_txt:
        return 3
    if "2 brins" in type_txt:
        return 2
    return 1


def _get_fyk_and_mu_ref(beam_id: int):
    acier_non_standard = bool(st.session_state.get(KB("acier_non_standard", beam_id), False))
    if acier_non_standard:
        try:
            fyk = float(st.session_state.get(KB("fyk_custom", beam_id), 500.0))
        except Exception:
            fyk = 500.0
        mu_ref = str(st.session_state.get(KB("fyk_ref_for_mu", beam_id), "500"))
        if mu_ref not in ("400", "500"):
            mu_ref = "500"
        return fyk, mu_ref

    try:
        fyk = float(st.session_state.get(KB("fyk", beam_id), 500))
    except Exception:
        fyk = 500.0
    return fyk, str(int(fyk))


def _as_total_with_optional_second_layer(beam_id: int, sec_id: int, which: str):
    if which == "inf":
        n1 = int(st.session_state.get(KS("n_as_inf", beam_id, sec_id), 2) or 2)
        d1 = int(st.session_state.get(KS("ø_as_inf", beam_id, sec_id), 16) or 16)
        has2 = bool(st.session_state.get(KS("ajouter_second_lit_inf", beam_id, sec_id), False))
        n2 = int(st.session_state.get(KS("n_as_inf_2", beam_id, sec_id), 2) or 2)
        d2 = int(st.session_state.get(KS("ø_as_inf_2", beam_id, sec_id), d1) or d1)
        jeu = float(st.session_state.get(KS("jeu_inf_2", beam_id, sec_id), 0.0) or 0.0)
    else:
        n1 = int(st.session_state.get(KS("n_as_sup", beam_id, sec_id), 2) or 2)
        d1 = int(st.session_state.get(KS("ø_as_sup", beam_id, sec_id), 16) or 16)
        has2 = bool(st.session_state.get(KS("ajouter_second_lit_sup", beam_id, sec_id), False))
        n2 = int(st.session_state.get(KS("n_as_sup_2", beam_id, sec_id), 2) or 2)
        d2 = int(st.session_state.get(KS("ø_as_sup_2", beam_id, sec_id), d1) or d1)
        jeu = float(st.session_state.get(KS("jeu_sup_2", beam_id, sec_id), 0.0) or 0.0)

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
#  POUTRE ACTIVE (pour layout gauche/droite)
# ============================================================
def _get_active_beam_id() -> int:
    _init_beams_if_needed()
    if "active_beam_id" not in st.session_state:
        st.session_state.active_beam_id = int(st.session_state.beams[0]["id"])
    ids = [int(b["id"]) for b in st.session_state.beams]
    if int(st.session_state.active_beam_id) not in ids:
        st.session_state.active_beam_id = int(ids[0])
    return int(st.session_state.active_beam_id)


def _set_active_beam_id(bid: int):
    st.session_state.active_beam_id = int(bid)

# ===========================
#  PARTIE 2 / 2
#  poutre.py (Streamlit)
# ===========================

# ============================================================
#  UI : SOLLICITATIONS PAR SECTION (Solution B : poubelle visible)
# ============================================================
def _render_section_inputs(beam_id: int, sec_id: int):
    cmom, cev = st.columns(2)

    with cmom:
        float_input_fr_simple("Moment inférieur M (kNm)", key=KS("M_inf", beam_id, sec_id), default=0.0, min_value=0.0)

        m_sup_toggle = st.checkbox(
            "Ajouter un moment supérieur",
            key=KS("ajouter_moment_sup", beam_id, sec_id),
            value=bool(st.session_state.get(KS("ajouter_moment_sup", beam_id, sec_id), False)),
        )
        if m_sup_toggle:
            float_input_fr_simple("Moment supérieur M_sup (kNm)", key=KS("M_sup", beam_id, sec_id), default=0.0, min_value=0.0)
        else:
            st.session_state[KS("M_sup", beam_id, sec_id)] = 0.0

    with cev:
        float_input_fr_simple("Effort tranchant V (kN)", key=KS("V", beam_id, sec_id), default=0.0, min_value=0.0)

        v_sup = st.checkbox(
            "Ajouter un effort tranchant réduit",
            key=KS("ajouter_effort_reduit", beam_id, sec_id),
            value=bool(st.session_state.get(KS("ajouter_effort_reduit", beam_id, sec_id), False)),
        )
        if v_sup:
            float_input_fr_simple("Effort tranchant réduit V_réduit (kN)", key=KS("V_lim", beam_id, sec_id), default=0.0, min_value=0.0)
        else:
            st.session_state[KS("V_lim", beam_id, sec_id)] = 0.0


def render_solicitations_for_beam(beam_id: int):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)

    st.markdown("#### Sollicitations")
    for s in beam["sections"]:
        sid = int(s.get("id"))
        sec_nom = str(s.get("nom", f"Section {sid}"))

        cexp, cdel = st.columns([24, 2], vertical_alignment="center")

        with cexp:
            with st.expander(f"{sec_nom}", expanded=True if sid == 1 else False):
                new_nom = st.text_input("Nom de la section", value=sec_nom, key=f"meta_b{beam_id}_nom_{sid}")
                s["nom"] = new_nom
                _render_section_inputs(beam_id, sid)

        with cdel:
            if sid > 1:
                if st.button("🗑️", key=f"btn_del_b{beam_id}_sec_{sid}", use_container_width=True):
                    _delete_section(beam_id, sid)
                    st.rerun()

    st.markdown("---")
    if st.button("➕ Ajouter une section à vérifier", use_container_width=True, key=f"btn_add_section_b{beam_id}"):
        _add_section(beam_id)
        st.rerun()


# ============================================================
#  UI : CARACTÉRISTIQUES D'UNE POUTRE (gauche)
# ============================================================
def render_caracteristiques_beam(beam_id: int, beton_data: dict):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)

    st.markdown("#### Caractéristiques de la poutre")

    # Nom poutre
    beam["nom"] = st.text_input(
        "Nom de la poutre",
        value=str(beam.get("nom", f"Poutre {beam_id}")),
        key=f"meta_beam_nom_{beam_id}",
    )

    # béton / acier
    cbet, cacier = st.columns(2)
    with cbet:
        options = list(beton_data.keys())
        cur = str(st.session_state.get(KB("beton", beam_id), options[min(2, len(options) - 1)]))
        st.selectbox("Classe de béton", options, index=options.index(cur), key=KB("beton", beam_id))

    with cacier:
        st.checkbox(
            "Qualité d'acier non standard",
            value=bool(st.session_state.get(KB("acier_non_standard", beam_id), False)),
            key=KB("acier_non_standard", beam_id),
        )

        if not bool(st.session_state.get(KB("acier_non_standard", beam_id), False)):
            acier_opts = ["400", "500"]
            cur_fyk = str(st.session_state.get(KB("fyk", beam_id), "500"))
            st.selectbox("Qualité d'acier [N/mm²]", acier_opts, index=acier_opts.index(cur_fyk) if cur_fyk in acier_opts else 1, key=KB("fyk", beam_id))
            st.session_state.setdefault(KB("fyk_custom", beam_id), 500.0)
            st.session_state.setdefault(KB("fyk_ref_for_mu", beam_id), "500")
        else:
            st.number_input(
                "fyk (non standard) [N/mm²]",
                min_value=200.0,
                max_value=2000.0,
                value=float(st.session_state.get(KB("fyk_custom", beam_id), 500.0) or 500.0),
                step=10.0,
                key=KB("fyk_custom", beam_id),
            )
            st.selectbox(
                "Référence mu (base béton)",
                ["400", "500"],
                index=1 if str(st.session_state.get(KB("fyk_ref_for_mu", beam_id), "500")) == "500" else 0,
                key=KB("fyk_ref_for_mu", beam_id),
            )

    # section b/h/enrobage
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Larg. [cm]", min_value=5, max_value=1000, value=int(st.session_state.get(KB("b", beam_id), 20) or 20), step=5, key=KB("b", beam_id))
    with c2:
        st.number_input("Haut. [cm]", min_value=5, max_value=1000, value=int(st.session_state.get(KB("h", beam_id), 40) or 40), step=5, key=KB("h", beam_id))
    with c3:
        st.number_input("Enrob. (cm)", min_value=0.0, max_value=100.0, value=float(st.session_state.get(KB("enrobage", beam_id), 5.0) or 5.0), step=0.5, key=KB("enrobage", beam_id))


# ============================================================
#  DIMENSIONNEMENT / render_dimensionnement_section
#  IMPORTANT : tu gardes exactement ton code existant ici.
#  (Tu l’as déjà dans ton message — je ne le réécris pas pour éviter de casser.)
# ============================================================
# -> Conserve _dimensionnement_compute_states(...)
# -> Conserve render_dimensionnement_section(...)
# ============================================================
#  DIMENSIONNEMENT PAR SECTION + STATUT GLOBAL (inchangé en formules)
# ============================================================
def _dimensionnement_compute_states(beam_id: int, sec_id: int, beton_data: dict):
    beton = str(st.session_state.get(KB("beton", beam_id), "C30/37"))
    fck_cube = beton_data[beton]["fck_cube"]
    alpha_b = beton_data[beton]["alpha_b"]

    fyk, mu_ref = _get_fyk_and_mu_ref(beam_id)
    fyd = fyk / 1.5

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in beton_data[beton]:
        mu_key = "mu_a500" if "mu_a500" in beton_data[beton] else [k for k in beton_data[beton].keys() if k.startswith("mu_a")][0]
    mu_val = beton_data[beton][mu_key]

    b = float(st.session_state.get(KB("b", beam_id), 20))
    h = float(st.session_state.get(KB("h", beam_id), 40))
    enrobage = float(st.session_state.get(KB("enrobage", beam_id), 5.0))
    d_utile = h - enrobage  # cm

    tol_tau = float(st.session_state.get("tau_tolerance_percent", 0.0) or 0.0)

    M_inf_val = float(st.session_state.get(KS("M_inf", beam_id, sec_id), 0.0) or 0.0)
    M_sup_val = float(st.session_state.get(KS("M_sup", beam_id, sec_id), 0.0) or 0.0)
    V_val = float(st.session_state.get(KS("V", beam_id, sec_id), 0.0) or 0.0)
    V_lim_val = float(st.session_state.get(KS("V_lim", beam_id, sec_id), 0.0) or 0.0)
    has_Vlim = bool(st.session_state.get(KS("ajouter_effort_reduit", beam_id, sec_id), False)) and (V_lim_val > 0)

    # Hauteur
    M_max = max(M_inf_val, M_sup_val)
    if M_max > 0:
        hmin_calc = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10  # cm
    else:
        hmin_calc = 0.0
    etat_h = "ok" if (hmin_calc + enrobage <= h) else "nok"

    # As min/max
    As_min_formula = 0.0013 * b * h * 1e2  # mm²
    As_max = 0.04 * b * h * 1e2  # mm²

    As_req_inf = (M_inf_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_inf_val > 0 else 0.0
    As_req_sup = (M_sup_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_sup_val > 0 else 0.0
    As_min_inf_eff = max(As_min_formula, 0.25 * As_req_sup)
    As_min_sup_eff = max(As_min_formula, 0.25 * As_req_inf)

    As_inf_total, _ = _as_total_with_optional_second_layer(beam_id, sec_id, "inf")
    As_sup_total, _ = _as_total_with_optional_second_layer(beam_id, sec_id, "sup")
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
        typ = str(st.session_state.get(KS("type_etrier", beam_id, sec_id), "Étriers (2 brins)"))
        brins = _brins_from_type(typ)
        n_et = int(st.session_state.get(KS("n_etriers", beam_id, sec_id), 1) or 1)
        d_et = int(st.session_state.get(KS("ø_etrier", beam_id, sec_id), 8) or 8)
        pas = float(st.session_state.get(KS("pas_etrier", beam_id, sec_id), 30.0) or 30.0)

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

        typ_r = str(st.session_state.get(KS("type_etrier_r", beam_id, sec_id), "Étriers (2 brins)"))
        brins_r = _brins_from_type(typ_r)
        n_et_r = int(st.session_state.get(KS("n_etriers_r", beam_id, sec_id), 1) or 1)
        d_et_r = int(st.session_state.get(KS("ø_etrier_r", beam_id, sec_id), 8) or 8)
        pas_r = float(st.session_state.get(KS("pas_etrier_r", beam_id, sec_id), 30.0) or 30.0)

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
        "hmin_calc": hmin_calc,
        "tau_1": tau_1,
        "tau_2": tau_2,
        "tau_4": tau_4,
    }


def render_dimensionnement_section(beam_id: int, sec_id: int, beton_data: dict):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    sec = next(s for s in beam["sections"] if int(s.get("id")) == sec_id)
    sec_nom = str(sec.get("nom", f"Section {sec_id}"))

    states = _dimensionnement_compute_states(beam_id, sec_id, beton_data)
    etat_global = states["etat_global"]

    title = _status_icon_label(etat_global, sec_nom)

    # Solution B : expander + poubelle sur la même ligne (visible même fermé)
    cexp, cdel = st.columns([24, 2], vertical_alignment="center")
    with cexp:
        with st.expander(title, expanded=True if sec_id == 1 else False):
            beton = str(st.session_state.get(KB("beton", beam_id), "C30/37"))
            fck_cube = beton_data[beton]["fck_cube"]
            alpha_b = beton_data[beton]["alpha_b"]

            fyk, mu_ref = _get_fyk_and_mu_ref(beam_id)
            fyd = fyk / 1.5

            mu_key = f"mu_a{mu_ref}"
            if mu_key not in beton_data[beton]:
                mu_key = "mu_a500" if "mu_a500" in beton_data[beton] else [k for k in beton_data[beton].keys() if k.startswith("mu_a")][0]
            mu_val = beton_data[beton][mu_key]

            b = float(st.session_state.get(KB("b", beam_id), 20))
            h = float(st.session_state.get(KB("h", beam_id), 40))
            enrobage = float(st.session_state.get(KB("enrobage", beam_id), 5.0))
            d_utile = h - enrobage  # cm

            units_len = st.session_state.get("units_len", "cm")
            units_as = st.session_state.get("units_as", "mm²")
            tol_tau = float(st.session_state.get("tau_tolerance_percent", 0.0) or 0.0)

            M_inf_val = float(st.session_state.get(KS("M_inf", beam_id, sec_id), 0.0) or 0.0)
            M_sup_val = float(st.session_state.get(KS("M_sup", beam_id, sec_id), 0.0) or 0.0)
            V_val = float(st.session_state.get(KS("V", beam_id, sec_id), 0.0) or 0.0)
            V_lim_val = float(st.session_state.get(KS("V_lim", beam_id, sec_id), 0.0) or 0.0)

            # ---- Vérification de la hauteur ----
            M_max = max(M_inf_val, M_sup_val)
            if M_max > 0:
                hmin_calc = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10  # cm
            else:
                hmin_calc = 0.0

            # header demandé : "C30/37 — 40×60 cm — hmin=..."
            if units_len == "mm":
                right_h = f"{beton} — {b*10:.0f}×{h*10:.0f} mm — hmin={hmin_calc*10:.0f} mm"
            else:
                right_h = f"{beton} — {b:.0f}×{h:.0f} cm — hmin={hmin_calc:.1f} cm"

            open_bloc_left_right("Vérification de la hauteur", right_h, states["etat_h"])
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
            As_max = 0.04 * b * h * 1e2  # mm²

            As_req_inf = (M_inf_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_inf_val > 0 else 0.0
            As_req_sup = (M_sup_val * 1e6) / (fyd * 0.9 * d_utile * 10) if M_sup_val > 0 else 0.0

            As_min_inf_eff = max(As_min_formula, 0.25 * As_req_sup)
            As_min_sup_eff = max(As_min_formula, 0.25 * As_req_inf)

            unit_as_txt = "mm²" if units_as == "mm²" else "cm²"

            # ---- Armatures inférieures ----
            diam_opts = [6, 8, 10, 12, 16, 20, 25, 32, 40]
            As_inf_total, inf_detail = _as_total_with_optional_second_layer(beam_id, sec_id, "inf")
            As_inf_disp = As_inf_total if units_as == "mm²" else As_inf_total / 100.0
            right_inf = f"{inf_detail} — As={As_inf_disp:.2f} {unit_as_txt}"

            open_bloc_left_right("Armatures inférieures", right_inf, states["etat_inf"])
            ca1, ca2, ca3 = st.columns(3)
            with ca1:
                st.markdown(f"**Aₛ,req,inf = {As_req_inf:.0f} mm²**")
            with ca2:
                st.markdown(f"**Aₛ,min,inf = {As_min_inf_eff:.0f} mm²**")
            with ca3:
                st.markdown(f"**Aₛ,max = {As_max:.0f} mm²**")

            r1c1, r1c2 = st.columns([1, 1])
            with r1c1:
                st.number_input(
                    "Nb barres",
                    min_value=1,
                    max_value=50,
                    value=int(st.session_state.get(KS("n_as_inf", beam_id, sec_id), 2) or 2),
                    step=1,
                    key=KS("n_as_inf", beam_id, sec_id),
                )
            with r1c2:
                dcur = int(st.session_state.get(KS("ø_as_inf", beam_id, sec_id), 16) or 16)
                idx = diam_opts.index(dcur) if dcur in diam_opts else diam_opts.index(16)
                st.selectbox("Ø (mm)", diam_opts, index=idx, key=KS("ø_as_inf", beam_id, sec_id))

            has2 = st.checkbox(
                "Ajouter un second lit (inf.)",
                value=bool(st.session_state.get(KS("ajouter_second_lit_inf", beam_id, sec_id), False)),
                key=KS("ajouter_second_lit_inf", beam_id, sec_id),
            )
            if has2:
                s2c1, s2c2, s2c3 = st.columns([1, 1, 1])
                with s2c1:
                    st.number_input(
                        "Nb barres (2e lit)",
                        min_value=1,
                        max_value=50,
                        value=int(st.session_state.get(KS("n_as_inf_2", beam_id, sec_id), 2) or 2),
                        step=1,
                        key=KS("n_as_inf_2", beam_id, sec_id),
                    )
                with s2c2:
                    dcur2 = int(st.session_state.get(KS("ø_as_inf_2", beam_id, sec_id), dcur) or dcur)
                    idx2 = diam_opts.index(dcur2) if dcur2 in diam_opts else diam_opts.index(dcur)
                    st.selectbox("Ø (mm) (2e lit)", diam_opts, index=idx2, key=KS("ø_as_inf_2", beam_id, sec_id))
                with s2c3:
                    float_input_fr_simple("Jeu entre lits (cm)", key=KS("jeu_inf_2", beam_id, sec_id), default=0.0, min_value=0.0)

            As_inf_total2, inf_detail2 = _as_total_with_optional_second_layer(beam_id, sec_id, "inf")
            As_inf_disp2 = As_inf_total2 if units_as == "mm²" else As_inf_total2 / 100.0
            st.markdown(
                f"<div style='margin-top:6px;font-weight:600;'>Choix : {inf_detail2} — ( {As_inf_disp2:.2f} {unit_as_txt} )</div>",
                unsafe_allow_html=True,
            )
            close_bloc()

            # ---- Armatures supérieures ----
            As_sup_total, sup_detail = _as_total_with_optional_second_layer(beam_id, sec_id, "sup")
            As_sup_disp = As_sup_total if units_as == "mm²" else As_sup_total / 100.0
            right_sup = f"{sup_detail} — As={As_sup_disp:.2f} {unit_as_txt}"

            open_bloc_left_right("Armatures supérieures", right_sup, states["etat_sup"])
            cs1, cs2, cs3 = st.columns(3)
            with cs1:
                st.markdown(f"**Aₛ,req,sup = {As_req_sup:.0f} mm²**")
            with cs2:
                st.markdown(f"**Aₛ,min,sup = {As_min_sup_eff:.0f} mm²**")
            with cs3:
                st.markdown(f"**Aₛ,max = {As_max:.0f} mm²**")

            t1c1, t1c2 = st.columns([1, 1])
            with t1c1:
                st.number_input(
                    "Nb barres (sup.)",
                    min_value=1,
                    max_value=50,
                    value=int(st.session_state.get(KS("n_as_sup", beam_id, sec_id), 2) or 2),
                    step=1,
                    key=KS("n_as_sup", beam_id, sec_id),
                )
            with t1c2:
                dcurS = int(st.session_state.get(KS("ø_as_sup", beam_id, sec_id), 16) or 16)
                idxS = diam_opts.index(dcurS) if dcurS in diam_opts else diam_opts.index(16)
                st.selectbox("Ø (mm) (sup.)", diam_opts, index=idxS, key=KS("ø_as_sup", beam_id, sec_id))

            has2s = st.checkbox(
                "Ajouter un second lit (sup.)",
                value=bool(st.session_state.get(KS("ajouter_second_lit_sup", beam_id, sec_id), False)),
                key=KS("ajouter_second_lit_sup", beam_id, sec_id),
            )
            if has2s:
                u2c1, u2c2, u2c3 = st.columns([1, 1, 1])
                with u2c1:
                    st.number_input(
                        "Nb barres (2e lit) (sup.)",
                        min_value=1,
                        max_value=50,
                        value=int(st.session_state.get(KS("n_as_sup_2", beam_id, sec_id), 2) or 2),
                        step=1,
                        key=KS("n_as_sup_2", beam_id, sec_id),
                    )
                with u2c2:
                    dcurS2 = int(st.session_state.get(KS("ø_as_sup_2", beam_id, sec_id), dcurS) or dcurS)
                    idxS2 = diam_opts.index(dcurS2) if dcurS2 in diam_opts else diam_opts.index(dcurS)
                    st.selectbox("Ø (mm) (2e lit) (sup.)", diam_opts, index=idxS2, key=KS("ø_as_sup_2", beam_id, sec_id))
                with u2c3:
                    float_input_fr_simple("Jeu entre lits (cm) (sup.)", key=KS("jeu_sup_2", beam_id, sec_id), default=0.0, min_value=0.0)

            As_sup_total2, sup_detail2 = _as_total_with_optional_second_layer(beam_id, sec_id, "sup")
            As_sup_disp2 = As_sup_total2 if units_as == "mm²" else As_sup_total2 / 100.0
            st.markdown(
                f"<div style='margin-top:6px;font-weight:600;'>Choix : {sup_detail2} — ( {As_sup_disp2:.2f} {unit_as_txt} )</div>",
                unsafe_allow_html=True,
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

                right_tau = f"τ={tau:.2f} ≤ {nom_lim}={tau_lim:.2f}"
                open_bloc_left_right("Vérification de l'effort tranchant", right_tau, etat_tau)
                extra = f" {suffix}" if suffix else ""
                st.markdown(f"τ = {tau:.2f} N/mm² ≤ {nom_lim} = {tau_lim:.2f} N/mm² → {besoin}{extra}")
                close_bloc()

                cE0, cE1, cE2, cE3 = st.columns([2, 2, 2, 2])
                with cE0:
                    st.selectbox(
                        "Type",
                        ["Étriers (2 brins)", "Épingles (1 brin)", "Étriers (3 brins)"],
                        index=0
                        if str(st.session_state.get(KS("type_etrier", beam_id, sec_id), "Étriers (2 brins)")) == "Étriers (2 brins)"
                        else (2 if str(st.session_state.get(KS("type_etrier", beam_id, sec_id), "")) == "Étriers (3 brins)" else 1),
                        key=KS("type_etrier", beam_id, sec_id),
                    )
                with cE1:
                    st.number_input(
                        "Nbr. cadres",
                        min_value=1,
                        max_value=8,
                        value=int(st.session_state.get(KS("n_etriers", beam_id, sec_id), 1) or 1),
                        step=1,
                        key=KS("n_etriers", beam_id, sec_id),
                    )
                with cE2:
                    diam_list = [6, 8, 10, 12]
                    dcurE = int(st.session_state.get(KS("ø_etrier", beam_id, sec_id), 8) or 8)
                    idxE = diam_list.index(dcurE) if dcurE in diam_list else diam_list.index(8)
                    st.selectbox("Ø (mm)", diam_list, index=idxE, key=KS("ø_etrier", beam_id, sec_id))
                with cE3:
                    float_input_fr_simple("Pas choisi (cm)", key=KS("pas_etrier", beam_id, sec_id), default=30.0, min_value=1.0)

                typ = str(st.session_state.get(KS("type_etrier", beam_id, sec_id), "Étriers (2 brins)"))
                brins = _brins_from_type(typ)
                n_et = int(st.session_state.get(KS("n_etriers", beam_id, sec_id), 1) or 1)
                d_et = int(st.session_state.get(KS("ø_etrier", beam_id, sec_id), 8) or 8)
                pas = float(st.session_state.get(KS("pas_etrier", beam_id, sec_id), 30.0) or 30.0)

                Ast_e = n_et * brins * _bar_area_mm2(d_et)  # mm²
                pas_th = Ast_e * fyd * d_utile * 10 / (10 * V_val * 1e3)
                s_max = min(0.75 * d_utile, 30.0)
                pas_lim = min(pas_th, s_max)
                etat_pas, suffix_pas = _status_with_tolerance(pas, pas_lim, tol_tau)

                right_et = f"pas={pas:.1f} ≤ min({pas_th:.1f},{s_max:.1f})={pas_lim:.1f} cm"
                open_bloc_left_right("Détermination des étriers", right_et, etat_pas)
                a1, a2, a3 = st.columns([1, 1, 2])
                with a1:
                    st.markdown(f"**Pas théorique = {pas_th:.1f} cm**")
                with a2:
                    st.markdown(f"**Pas maximal = {s_max:.1f} cm**")
                with a3:
                    if suffix_pas:
                        st.markdown(f"**{suffix_pas}**")
                close_bloc()

            # ---- Tranchant réduit ----
            if bool(st.session_state.get(KS("ajouter_effort_reduit", beam_id, sec_id), False)) and V_lim_val > 0:
                tau_r = V_lim_val * 1e3 / (0.75 * b * h * 100)
                besoin_r, etat_r_base, nom_lim_r, tau_lim_r = _shear_need_text(tau_r)
                if tau_r > tau_lim_r:
                    etat_r, suffix_r = _status_with_tolerance(tau_r, tau_lim_r, tol_tau)
                else:
                    etat_r, suffix_r = etat_r_base, ""

                right_tau_r = f"τ={tau_r:.2f} ≤ {nom_lim_r}={tau_lim_r:.2f}"
                open_bloc_left_right("Vérification effort tranchant réduit", right_tau_r, etat_r)
                extra = f" {suffix_r}" if suffix_r else ""
                st.markdown(f"τ = {tau_r:.2f} N/mm² ≤ {nom_lim_r} = {tau_lim_r:.2f} N/mm² → {besoin_r}{extra}")
                close_bloc()

                cR0, cR1, cR2, cR3 = st.columns([2, 2, 2, 2])
                with cR0:
                    st.selectbox(
                        "Type (réduit)",
                        ["Étriers (2 brins)", "Épingles (1 brin)", "Étriers (3 brins)"],
                        index=0
                        if str(st.session_state.get(KS("type_etrier_r", beam_id, sec_id), "Étriers (2 brins)")) == "Étriers (2 brins)"
                        else (2 if str(st.session_state.get(KS("type_etrier_r", beam_id, sec_id), "")) == "Étriers (3 brins)" else 1),
                        key=KS("type_etrier_r", beam_id, sec_id),
                    )
                with cR1:
                    st.number_input(
                        "Nbr. cadres (réduit)",
                        min_value=1,
                        max_value=8,
                        value=int(st.session_state.get(KS("n_etriers_r", beam_id, sec_id), 1) or 1),
                        step=1,
                        key=KS("n_etriers_r", beam_id, sec_id),
                    )
                with cR2:
                    diam_list_r = [6, 8, 10, 12]
                    dcurER = int(st.session_state.get(KS("ø_etrier_r", beam_id, sec_id), 8) or 8)
                    idxER = diam_list_r.index(dcurER) if dcurER in diam_list_r else diam_list_r.index(8)
                    st.selectbox("Ø (mm) (réduit)", diam_list_r, index=idxER, key=KS("ø_etrier_r", beam_id, sec_id))
                with cR3:
                    float_input_fr_simple("Pas choisi (cm) (réduit)", key=KS("pas_etrier_r", beam_id, sec_id), default=30.0, min_value=1.0)

                typ_r = str(st.session_state.get(KS("type_etrier_r", beam_id, sec_id), "Étriers (2 brins)"))
                brins_r = _brins_from_type(typ_r)
                n_et_r = int(st.session_state.get(KS("n_etriers_r", beam_id, sec_id), 1) or 1)
                d_et_r = int(st.session_state.get(KS("ø_etrier_r", beam_id, sec_id), 8) or 8)
                pas_r = float(st.session_state.get(KS("pas_etrier_r", beam_id, sec_id), 30.0) or 30.0)

                Ast_er = n_et_r * brins_r * _bar_area_mm2(d_et_r)
                pas_th_r = Ast_er * fyd * d_utile * 10 / (10 * V_lim_val * 1e3)
                s_max_r = min(0.75 * d_utile, 30.0)
                pas_lim_r = min(pas_th_r, s_max_r)
                etat_pas_r, suffix_pas_r = _status_with_tolerance(pas_r, pas_lim_r, tol_tau)

                right_et_r = f"pas={pas_r:.1f} ≤ min({pas_th_r:.1f},{s_max_r:.1f})={pas_lim_r:.1f} cm"
                open_bloc_left_right("Détermination étriers réduits", right_et_r, etat_pas_r)
                b1, b2, b3 = st.columns([1, 1, 2])
                with b1:
                    st.markdown(f"**Pas théorique = {pas_th_r:.1f} cm**")
                with b2:
                    st.markdown(f"**Pas maximal = {s_max_r:.1f} cm**")
                with b3:
                    if suffix_pas_r:
                        st.markdown(f"**{suffix_pas_r}**")
                close_bloc()

    with cdel:
        if sec_id > 1:
            if st.button("🗑️", key=f"btn_del_dim_b{beam_id}_sec{sec_id}", use_container_width=True):
                _delete_section(beam_id, sec_id)
                st.rerun()

# ============================================================
#  COLONNE GAUCHE : choix poutre + actions
# ============================================================
def render_beam_picker_and_actions_left():
    _init_beams_if_needed()
    active_id = _get_active_beam_id()

    st.markdown("### Poutres")
    ids = [int(b["id"]) for b in st.session_state.beams]
    labels = [str(b.get("nom", f"Poutre {b['id']}")) for b in st.session_state.beams]
    idx = ids.index(active_id) if active_id in ids else 0

    chosen = st.selectbox("Poutre active", options=ids, format_func=lambda x: labels[ids.index(x)], index=idx, key="beam_picker")
    if int(chosen) != int(active_id):
        _set_active_beam_id(int(chosen))
        st.rerun()

    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("➕ Ajouter", use_container_width=True, key="btn_add_beam_left"):
            _add_beam()
            _set_active_beam_id(int(st.session_state.beams[-1]["id"]))
            st.rerun()
    with cB:
        if st.button("📄 Dupliquer", use_container_width=True, key="btn_dup_beam_left"):
            _duplicate_beam(active_id)
            _set_active_beam_id(int(st.session_state.beams[-1]["id"]))
            st.rerun()
    with cC:
        if active_id > 1:
            if st.button("🗑️ Suppr.", use_container_width=True, key="btn_del_beam_left"):
                _delete_beam(active_id)
                st.rerun()
        else:
            st.button("🔒", use_container_width=True, key="btn_del_beam_left_disabled", disabled=True)


def render_beam_inputs_left(beam_id: int, beton_data: dict):
    st.markdown("### Dimensionnement de la poutre")
    render_caracteristiques_beam(beam_id, beton_data)
    st.markdown("---")
    render_solicitations_for_beam(beam_id)


def render_beam_results_right(beam_id: int, beton_data: dict):
    beam = next(b for b in st.session_state.beams if int(b.get("id")) == beam_id)
    st.markdown("### Dimensionnement")
    st.markdown(f"**{beam.get('nom', f'Poutre {beam_id}')}**")
    for s in beam["sections"]:
        render_dimensionnement_section(beam_id, int(s["id"]), beton_data)


# ============================================================
#  SHOW() : layout final gauche/droite
# ============================================================
def show():
    _init_beams_if_needed()

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
                data = json.load(uploaded)
                _load_from_payload(data)
                st.session_state["show_open_uploader"] = False
                _get_active_beam_id()
                st.success("Fichier chargé.")
                st.rerun()

    with btn5:
        st.button("📄 Générer PDF", use_container_width=True, key="btn_pdf_disabled", disabled=True)

    with open("beton_classes.json", "r", encoding="utf-8") as f:
        beton_data = json.load(f)

    # colonnes écran
    input_col_gauche, result_col_droite = st.columns([2, 3])
    active_beam_id = _get_active_beam_id()

    with input_col_gauche:
        with st.expander("Paramètres avancés", expanded=False):
            st.selectbox("Affichage longueurs", ["cm", "mm"], index=0 if st.session_state.get("units_len", "cm") == "cm" else 1, key="units_len")
            st.selectbox("Affichage armatures", ["mm²", "cm²"], index=0 if st.session_state.get("units_as", "mm²") == "mm²" else 1, key="units_as")
            st.slider("Tolérance dépassement (%)", min_value=0, max_value=25, value=int(st.session_state.get("tau_tolerance_percent", 0) or 0), step=1, key="tau_tolerance_percent")

        st.markdown("### Informations sur le projet")
        afficher_infos = st.checkbox("Ajouter les informations du projet", value=bool(st.session_state.get("chk_infos_projet", False)), key="chk_infos_projet")
        if afficher_infos:
            st.text_input("", placeholder="Nom du projet", key="nom_projet")
            st.text_input("", placeholder="Partie", key="partie")
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("", placeholder="Date (jj/mm/aaaa)", value=st.session_state.get("date", datetime.today().strftime("%d/%m/%Y")), key="date")
            with c2:
                st.text_input("", placeholder="Indice", value=st.session_state.get("indice", "0"), key="indice")
        else:
            st.session_state.setdefault("date", datetime.today().strftime("%d/%m/%Y"))

        st.markdown("---")
        render_beam_picker_and_actions_left()
        st.markdown("---")
        render_beam_inputs_left(active_beam_id, beton_data)

    with result_col_droite:
        render_beam_results_right(active_beam_id, beton_data)

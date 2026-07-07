# =============================================================
#  raideur_sol.py — Raideur élastique des sols (modèle de Winkler)
#  VERSION 3.2
#
#  Évolutions vs 3.1 :
#   1. FIX PRÉREMPLISSAGE PEU FIABLE : le tableau "Couches de sol"
#      utilisait st.data_editor (num_rows="dynamic"). Ce widget ne
#      déclenche un rerun qu'à la perte de focus de la cellule éditée,
#      et affiche littéralement "None" sur les nouvelles lignes tant
#      qu'elles n'ont pas été éditées (défaut connu du widget). Le
#      préremplissage semblait donc ne pas fonctionner. -> Le tableau
#      est reconstruit avec des widgets natifs (selectbox/number_input,
#      un par cellule, même principe que le tableau des lits de
#      poutre.py) : chaque changement de type déclenche un rerun
#      immédiat et le préremplissage s'applique en une frappe, sans
#      "None" résiduel.
#   2. RÉORGANISATION LOGIQUE : "Largeur caractéristique B" et "ν
#      équivalent" ne servent QU'AU modèle Boussinesq (comparaison) et
#      n'ont jamais d'effet sur le modèle en ressorts en série (celui
#      à exporter vers SCIA). Ils étaient affichés tout en haut de la
#      colonne de gauche, comme s'ils faisaient partie du calcul
#      principal -> déplacés directement à côté du bloc Boussinesq,
#      avec une légende explicite ("sans effet sur le résultat SCIA").
#   3. Statut par ligne (pris en compte / ignorée, E suggéré qc->E,
#      avertissement rocher) affiché en légende sous chaque ligne du
#      tableau plutôt que dans un second tableau dupliqué en dessous
#      -> moins de redondance visuelle.
#   4. Ajout/suppression de couche par identifiant unique (comme
#      beam_id/sec_id dans poutre.py) : pas de décalage de clés à
#      gérer à la suppression, contrairement à une indexation par
#      position.
#
#  Winkler : q = k · w  ->  k = q / w
#    q [kPa = kN/m²], w [m], k [kN/m³]  (1 MN/m³ = 1000 kN/m³)
# =============================================================

import math
import pandas as pd
import streamlit as st


# =============================================================
#  CONSTANTES
# =============================================================
KGF_PER_CM2_TO_KPA = 98.0665          # 1 kgf/cm² = 98.0665 kPa
VERSION = "v3.2"

C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6", "info": "#eef2ff"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌", "info": "ℹ️"}

# Largeurs de colonnes du tableau des couches (h | Type | qc | Rf | E | Action)
LAYER_COLS = [0.8, 2.2, 1.0, 0.8, 1.0, 0.5]


# =============================================================
#  BASE DE DONNÉES SOLS (contexte belge) — SOURCE UNIQUE
#
#  Valeurs indicatives de littérature géotechnique courante (ordres de
#  grandeur type Bowles / DIN 4019 / retours d'expérience CSTC-SPW).
#  cpt_ok=False : qc n'a pas de sens physique pour ce matériau (refus de
#  pointe sur rocher sain, par exemple) -> pas de corrélation qc->E
#  proposée, E direct uniquement, à confirmer par le rapport géotechnique.
# =============================================================
SOIL_DB = {
    "Remblais / terre végétale": dict(
        category="Remblai", gamma=17.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=2.0, E_max=10.0, k_min=1, k_max=8, cpt_ok=False,
        desc="Matériau rapporté, hétérogène, non contrôlé. Ne jamais retenir "
             "comme assise de fondation sans reconnaissance spécifique (souvent à purger)."),
    "Tourbe": dict(
        category="Sol organique", gamma=10.0,
        qc_min=0.1, qc_max=0.5, alpha_qc=4.0, rf_typ=4.0,
        E_min=0.5, E_max=2.5, k_min=1, k_max=5, cpt_ok=True,
        desc="Sol très organique, très compressible, souvent saturé. Portance très "
             "faible : à éviter comme assise (substitution, pieux, colonnes)."),
    "Argile très molle": dict(
        category="Argile", gamma=16.0,
        qc_min=0.3, qc_max=1.0, alpha_qc=4.0, rf_typ=3.5,
        E_min=1.0, E_max=5.0, k_min=2, k_max=10, cpt_ok=True,
        desc="Argile plastique peu consolidée, forte compressibilité, faibles résistances."),
    "Argile molle à moyenne": dict(
        category="Argile", gamma=18.0,
        qc_min=1.0, qc_max=2.5, alpha_qc=5.0, rf_typ=3.0,
        E_min=5.0, E_max=15.0, k_min=10, k_max=40, cpt_ok=True,
        desc="Normalement à légèrement surconsolidée, tassements notables sous charge."),
    "Argile ferme / raide": dict(
        category="Argile", gamma=19.0,
        qc_min=2.5, qc_max=5.0, alpha_qc=6.0, rf_typ=2.5,
        E_min=15.0, E_max=40.0, k_min=20, k_max=80, cpt_ok=True,
        desc="Argile raide bien consolidée, tassements plus limités."),
    "Argile surconsolidée (Boom/Ypresienne)": dict(
        category="Argile", gamma=20.0,
        qc_min=3.0, qc_max=8.0, alpha_qc=7.0, rf_typ=2.5,
        E_min=25.0, E_max=80.0, k_min=40, k_max=150, cpt_ok=True,
        desc="Argiles profondes surconsolidées (bassin belge), raides, faible "
             "compressibilité résiduelle mais sensibles au gonflement/retrait si décomprimées."),
    "Limon (loess)": dict(
        category="Limon", gamma=18.0,
        qc_min=1.0, qc_max=3.0, alpha_qc=4.0, rf_typ=2.0,
        E_min=8.0, E_max=25.0, k_min=15, k_max=60, cpt_ok=True,
        desc="Très répandu en Hesbaye/Brabant. Comportement intermédiaire argile/sable, "
             "sensible à l'eau (collapsibilité possible à l'état non saturé)."),
    "Sable lâche": dict(
        category="Sable", gamma=18.0,
        qc_min=1.0, qc_max=5.0, alpha_qc=3.5, rf_typ=0.6,
        E_min=5.0, E_max=15.0, k_min=10, k_max=30, cpt_ok=True,
        desc="Peu compacté, tassements importants, risque de liquéfaction si saturé et sismique."),
    "Sable moyennement compact": dict(
        category="Sable", gamma=19.0,
        qc_min=5.0, qc_max=12.0, alpha_qc=5.0, rf_typ=0.5,
        E_min=15.0, E_max=40.0, k_min=30, k_max=80, cpt_ok=True,
        desc="Sable courant sous bâtiments, portance correcte, tassements modérés."),
    "Sable dense": dict(
        category="Sable", gamma=20.0,
        qc_min=12.0, qc_max=25.0, alpha_qc=6.0, rf_typ=0.4,
        E_min=40.0, E_max=80.0, k_min=80, k_max=150, cpt_ok=True,
        desc="Très compact, bonne portance, tassements faibles."),
    "Sable graveleux / grave compacte": dict(
        category="Sable/grave", gamma=21.0,
        qc_min=15.0, qc_max=30.0, alpha_qc=4.0, rf_typ=0.4,
        E_min=50.0, E_max=120.0, k_min=100, k_max=200, cpt_ok=True,
        desc="Granulométrie étalée bien compactée, très bonne portance."),
    "Craie altérée": dict(
        category="Craie", gamma=18.0,
        qc_min=1.5, qc_max=5.0, alpha_qc=3.0, rf_typ=1.5,
        E_min=15.0, E_max=60.0, k_min=30, k_max=100, cpt_ok=True,
        desc="Craie remaniée/fissurée (Hesbaye, Tournaisis) — comportement dispersé, "
             "attention aux dissolutions/cavités (karst crayeux)."),
    "Craie saine": dict(
        category="Craie", gamma=20.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=200.0, E_max=1500.0, k_min=150, k_max=500, cpt_ok=False,
        desc="Craie compacte non remaniée. Souvent refus au pénétromètre : caractériser "
             "par carottage/RQD ou essai de plaque plutôt que par CPT."),
    "Calcaire fracturé / altéré": dict(
        category="Calcaire", gamma=21.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=100.0, E_max=800.0, k_min=100, k_max=400, cpt_ok=False,
        desc="Massif calcaire fissuré ou altéré en surface. Grande dispersion : "
             "attention aux karst/cavités, RQD indispensable."),
    "Calcaire sain": dict(
        category="Calcaire", gamma=23.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=2000.0, E_max=15000.0, k_min=1000, k_max=3000, cpt_ok=False,
        desc="Massif rocheux sain, peu fracturé. Refus au pénétromètre — "
             "caractérisation par RQD/GSI/essai en place."),
    "Schiste houiller décomposé (W4-W5)": dict(
        category="Schiste houiller", gamma=18.0,
        qc_min=0.5, qc_max=3.0, alpha_qc=2.5, rf_typ=2.0,
        E_min=5.0, E_max=30.0, k_min=10, k_max=50, cpt_ok=True,
        desc="Roche entièrement à fortement décomposée (aspect de sol résiduel), "
             "classification ISO 14689 W4-W5. Comportement proche d'un sol fin ferme : "
             "un CPT reste indicatif, à recouper avec le log de sondage."),
    "Schiste houiller altéré (W3)": dict(
        category="Schiste houiller", gamma=20.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=100.0, E_max=800.0, k_min=100, k_max=400, cpt_ok=False,
        desc="Roche modérément altérée (W3), matrice affaiblie mais structure "
             "rocheuse conservée. Refus probable au CPT : caractériser par RQD/"
             "pressiomètre. Grande dispersion selon le degré de fracturation."),
    "Schiste houiller sain (W1-W2)": dict(
        category="Schiste houiller", gamma=25.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=1000.0, E_max=8000.0, k_min=800, k_max=3000, cpt_ok=False,
        desc="Roche saine à faiblement altérée (W1-W2), massif carbonifère typique "
             "des bassins wallons. Refus au pénétromètre — caractériser par RQD/GSI "
             "ou essai de plaque ; anisotropie de feuilletage à prendre en compte."),
    "Grès altéré": dict(
        category="Grès", gamma=20.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=150.0, E_max=1000.0, k_min=150, k_max=500, cpt_ok=False,
        desc="Grès fracturé/altéré en surface. RQD recommandé."),
    "Grès sain": dict(
        category="Grès", gamma=24.0,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=3000.0, E_max=20000.0, k_min=1500, k_max=4000, cpt_ok=False,
        desc="Massif rocheux sain. Refus au pénétromètre — RQD/GSI ou essai en place."),
    "Personnalisé": dict(
        category="—", gamma=None,
        qc_min=None, qc_max=None, alpha_qc=None, rf_typ=None,
        E_min=None, E_max=None, k_min=None, k_max=None, cpt_ok=True,
        desc="Valeurs saisies manuellement — aucune valeur suggérée."),
}

ROCK_CATEGORIES = {"Craie", "Calcaire", "Schiste houiller", "Grès"}


def soil_types_list():
    """Liste pour les menus déroulants : '—' (vide) + types de SOIL_DB."""
    return ["—"] + list(SOIL_DB.keys())


def _mid(lo, hi):
    if lo is None or hi is None:
        return None
    return round((lo + hi) / 2.0, 1)


def soil_default_qc(soil_type: str):
    """qc moyen typique (MPa) — None si non pertinent (rocher, remblai...)."""
    d = SOIL_DB.get(soil_type)
    if not d or not d.get("cpt_ok", False):
        return None
    return _mid(d.get("qc_min"), d.get("qc_max"))


def soil_default_Rf(soil_type: str):
    d = SOIL_DB.get(soil_type)
    if not d:
        return None
    return d.get("rf_typ")


def soil_default_E(soil_type: str):
    """E typique (MPa) — milieu de la plage de référence."""
    d = SOIL_DB.get(soil_type)
    if not d:
        return None
    return _mid(d.get("E_min"), d.get("E_max"))


def is_rock(soil_type: str) -> bool:
    d = SOIL_DB.get(soil_type)
    return bool(d and d.get("category") in ROCK_CATEGORIES)


# =============================================================
#  CONVERSIONS D'UNITÉS (fonctions pures)
# =============================================================
def to_kPa(value: float, unit: str) -> float:
    """Pression -> kPa."""
    return {"kPa": value, "MPa": value * 1000.0, "kg/cm²": value * KGF_PER_CM2_TO_KPA}.get(unit, value)


def from_kPa(value_kPa: float, unit: str) -> float:
    """kPa -> unité cible."""
    return {"kPa": value_kPa, "MPa": value_kPa / 1000.0, "kg/cm²": value_kPa / KGF_PER_CM2_TO_KPA}.get(unit, value_kPa)


def E_to_kPa(E: float, unit: str) -> float:
    """Module E (MPa ou GPa) -> kPa."""
    return {"MPa": E * 1000.0, "GPa": E * 1_000_000.0}.get(unit, E)


def kNpm3_to_MNpm3(v: float) -> float:
    return v / 1000.0


def suggest_E_from_qc(qc_MPa, soil_type: str):
    """E ≈ α·qc (MPa), selon SOIL_DB. Retourne None si qc invalide ou
    non pertinent pour ce type de sol (rocher)."""
    d = SOIL_DB.get(soil_type, {})
    alpha = d.get("alpha_qc")
    if alpha is None or qc_MPa is None or (isinstance(qc_MPa, float) and math.isnan(qc_MPa)) or qc_MPa <= 0:
        return None
    return round(alpha * qc_MPa, 1)


# =============================================================
#  CALCULS (fonctions pures — aucun Streamlit)
# =============================================================
def k_from_qw(q_kPa: float, w_mm: float):
    """Winkler direct : k = q / w. Retourne (k_kNpm3, k_MNpm3, w_m)."""
    w_m = w_mm / 1000.0
    if w_m <= 0:
        return 0.0, 0.0, w_m
    k = q_kPa / w_m
    return k, kNpm3_to_MNpm3(k), w_m


def k_series(layers):
    """
    Ressorts en série (tassement 1D d'une colonne de sol chargée) :
        1/k_serie = Σ h_i / E_i
    layers : liste de (h_m, E_kPa) valides (>0).
    Retourne (k_kNpm3, k_MNpm3, H_m, E_moy_kPa).
    """
    denom = 0.0
    H = 0.0
    for h, E in layers:
        if h > 0 and E > 0:
            denom += h / E
            H += h
    if denom <= 0:
        return 0.0, 0.0, H, 0.0
    k = 1.0 / denom
    E_moy = k * H  # module oedométrique équivalent d'une colonne d'épaisseur H
    return k, kNpm3_to_MNpm3(k), H, E_moy


def k_boussinesq(E_kPa: float, B_m: float, nu: float):
    """
    Semelle rigide sur massif élastique semi-infini (ordre de grandeur) :
        k ≈ E / [B (1 − ν²)]
    Retourne (k_kNpm3, k_MNpm3).
    """
    if E_kPa <= 0 or B_m <= 0 or nu >= 1.0:
        return 0.0, 0.0
    k = E_kPa / (B_m * (1.0 - nu ** 2))
    return k, kNpm3_to_MNpm3(k)


def E_from_cpt(qt_MPa: float, sv0_kPa: float, alpha_E: float):
    """E = α_E (qt − σ'v0). Retourne (E_kPa, E_MPa, delta_kPa)."""
    delta = max(qt_MPa * 1000.0 - sv0_kPa, 0.0)
    E_kPa = alpha_E * delta
    return E_kPa, E_kPa / 1000.0, delta


def k_plate(B_mm, L_mm, alpha, Ec_GPa, use_nu, nu_c,
            has_grout=False, tg_mm=0.0, Eg_GPa=20.0):
    """
    Contact plat/béton. Épaisseur mobilisée h_c = α·min(B,L).
    Compression 1D pure : k = E / h_c. Option (1−ν²) si l'utilisateur
    l'assume (rarement justifié pour une couche mince confinée).
    Grout en série : 1/k_eq = 1/k_c + 1/k_g.
    Retourne un dict de résultats intermédiaires + k_eq.
    """
    B = B_mm / 1000.0
    L = L_mm / 1000.0
    hc = alpha * min(B, L)
    Ec_kPa = E_to_kPa(Ec_GPa, "GPa")

    if hc <= 0:
        return {"hc": 0.0, "kc": 0.0, "kg": 0.0, "keq_kNpm3": 0.0, "keq_MNpm3": 0.0}

    fac = (1.0 - nu_c ** 2) if use_nu else 1.0
    kc = Ec_kPa / (hc * fac)

    keq = kc
    kg = 0.0
    if has_grout and tg_mm > 0:
        tg = tg_mm / 1000.0
        Eg_kPa = E_to_kPa(Eg_GPa, "GPa")
        kg = Eg_kPa / tg if tg > 0 else 0.0
        if kc > 0 and kg > 0:
            keq = 1.0 / (1.0 / kc + 1.0 / kg)

    return {"hc": hc, "kc": kc, "kg": kg,
            "keq_kNpm3": keq, "keq_MNpm3": kNpm3_to_MNpm3(keq)}


# =============================================================
#  RENDU : helpers UI
# =============================================================
def _bloc(left: str, right: str = "", etat: str = "ok"):
    right_html = f"<div style='font-weight:600;opacity:.9;white-space:nowrap;'>{right}</div>" if right else ""
    st.markdown(
        f"""
        <div style="background:{C_COULEURS.get(etat,'#f6f6f6')};padding:12px 14px;border-radius:10px;
             border:1px solid #d9d9d9;margin:8px 0 4px 0;display:flex;justify-content:space-between;
             align-items:center;gap:10px;">
          <div style="font-weight:700;">{left}</div>
          <div style="display:flex;align-items:center;gap:10px;">{right_html}
            <div style="font-size:20px;line-height:1;">{C_ICONES.get(etat,'')}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _param_table(rows):
    df = pd.DataFrame(rows, columns=["Paramètre", "Description", "Valeur", "Unité"])
    df.index = [""] * len(df)
    st.table(df)


def _metric(col, label, value, unit=""):
    col.metric(label, f"{value:,.2f}".replace(",", " ") + (f" {unit}" if unit else ""))


# =============================================================
#  ÉTAT DES COUCHES (identifiants uniques — pas de décalage à la
#  suppression, sur le même principe que beam_id/sec_id de poutre.py)
# =============================================================
def _layer_key(lid: int, field: str) -> str:
    return f"soil_layer_{lid}_{field}"


def _layer_ids():
    return list(st.session_state.get("layer_order", []))


def _new_layer_id() -> int:
    ids = _layer_ids()
    return (max(ids) + 1) if ids else 1


def _init_layer(lid: int, h=1.0, soil_type="—", qc=0.0, rf=0.0, E=0.0):
    st.session_state[_layer_key(lid, "h")] = float(h)
    st.session_state[_layer_key(lid, "type")] = soil_type
    st.session_state[_layer_key(lid, "type_prev")] = soil_type
    st.session_state[_layer_key(lid, "qc")] = float(qc)
    st.session_state[_layer_key(lid, "rf")] = float(rf)
    st.session_state[_layer_key(lid, "E")] = float(E)


def _add_layer():
    lid = _new_layer_id()
    st.session_state.layer_order.append(lid)
    _init_layer(lid)  # ligne vierge : l'utilisateur choisit un type -> préremplissage automatique


def _delete_layer(lid: int):
    ids = _layer_ids()
    if len(ids) <= 1 or lid not in ids:
        return
    st.session_state.layer_order.remove(lid)
    for f in ("h", "type", "type_prev", "qc", "rf", "E"):
        st.session_state.pop(_layer_key(lid, f), None)


def _get_layer_values(lid: int):
    return {
        "h": float(st.session_state.get(_layer_key(lid, "h"), 0.0) or 0.0),
        "type": st.session_state.get(_layer_key(lid, "type"), "—"),
        "qc": float(st.session_state.get(_layer_key(lid, "qc"), 0.0) or 0.0),
        "rf": float(st.session_state.get(_layer_key(lid, "rf"), 0.0) or 0.0),
        "E": float(st.session_state.get(_layer_key(lid, "E"), 0.0) or 0.0),
    }


# =============================================================
#  STATE
# =============================================================
def _init_state():
    d = {
        "press_unit": "kPa",
        "module_unit": "MPa",
        "detail_calc": True,
        "adv_open": False,
        "abaque_w": 20.0,
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)

    if "layer_order" not in st.session_state:
        st.session_state.layer_order = [1]
        _init_layer(1, h=2.0, soil_type="Sable moyennement compact", qc=8.5, rf=0.5, E=27.5)


# =============================================================
#  UI : TABLEAU DES COUCHES (widgets natifs — pas de data_editor)
# =============================================================
def _render_layer_row(lid: int, i: int, n: int, disabled: bool = False):
    """
    Une ligne = un widget par cellule (comme le tableau des lits de
    poutre.py). Chaque changement de valeur déclenche un rerun Streamlit
    immédiat : le préremplissage qc/Rf/E s'applique donc en une frappe,
    sans dépendre de la perte de focus (contrairement à data_editor).
    """
    h_key = _layer_key(lid, "h")
    type_key = _layer_key(lid, "type")
    prev_key = _layer_key(lid, "type_prev")
    qc_key = _layer_key(lid, "qc")
    rf_key = _layer_key(lid, "rf")
    E_key = _layer_key(lid, "E")

    st.session_state.setdefault(h_key, 1.0)
    st.session_state.setdefault(type_key, "—")
    st.session_state.setdefault(prev_key, st.session_state[type_key])
    st.session_state.setdefault(qc_key, 0.0)
    st.session_state.setdefault(rf_key, 0.0)
    st.session_state.setdefault(E_key, 0.0)

    lv = "visible" if i == 0 else "collapsed"
    va = "bottom" if i == 0 else "center"
    c_h, c_type, c_qc, c_rf, c_E, c_act = st.columns(LAYER_COLS, vertical_alignment=va)

    with c_h:
        st.number_input("h [m]", min_value=0.0, step=0.1, key=h_key,
                         disabled=disabled, label_visibility=lv)
    with c_type:
        st.selectbox("Type de sol", soil_types_list(), key=type_key,
                     disabled=disabled, label_visibility=lv)

    # ---- Préremplissage : mutation AVANT l'instanciation des widgets
    # qc/Rf/E ci-dessous (règle Streamlit : une clé de widget ne peut
    # être modifiée qu'avant que ce widget soit rendu dans le run). ----
    new_type = st.session_state.get(type_key, "—")
    type_changed = new_type != st.session_state.get(prev_key)
    if type_changed:
        qc_cur = float(st.session_state.get(qc_key, 0.0) or 0.0)
        E_cur = float(st.session_state.get(E_key, 0.0) or 0.0)
        if new_type not in ("—", "Personnalisé") and qc_cur <= 0 and E_cur <= 0:
            soil = SOIL_DB.get(new_type, {})
            if soil.get("cpt_ok", False):
                qc_def = soil_default_qc(new_type)
                if qc_def is not None:
                    st.session_state[qc_key] = qc_def
                rf_def = soil_default_Rf(new_type)
                if rf_def is not None:
                    st.session_state[rf_key] = rf_def
            E_def = soil_default_E(new_type)
            if E_def is not None:
                st.session_state[E_key] = E_def
        st.session_state[prev_key] = new_type

    with c_qc:
        st.number_input("qc moy [MPa]", min_value=0.0, step=0.5, key=qc_key,
                         disabled=disabled, label_visibility=lv)
    with c_rf:
        st.number_input("Rf [%]", min_value=0.0, step=0.5, key=rf_key,
                         disabled=disabled, label_visibility=lv)
    with c_E:
        st.number_input("E [MPa]", min_value=0.0, step=5.0, key=E_key,
                         disabled=disabled, label_visibility=lv)
    with c_act:
        if i == 0:
            st.button("＋", key=f"btn_add_layer_{lid}", use_container_width=True,
                       help="Ajouter une couche", disabled=disabled, on_click=_add_layer)
        else:
            st.button("🗑️", key=f"btn_del_layer_{lid}", use_container_width=True,
                       help="Supprimer cette couche", disabled=disabled,
                       on_click=_delete_layer, args=(lid,))

    # ---- Statut / info sous la ligne (remplace l'ancien tableau dupliqué) ----
    lv2 = _get_layer_values(lid)
    h_ok = lv2["h"] > 0
    E_ok = lv2["E"] > 0
    bits = []
    if not (h_ok and E_ok):
        bits.append("⚠️ ligne ignorée dans le calcul (h ou E manquant)")
    else:
        bits.append("✅ prise en compte")
    if new_type not in ("—", "Personnalisé"):
        e_sugg = suggest_E_from_qc(lv2["qc"], new_type)
        if e_sugg is not None:
            bits.append(f"E suggéré (qc→E) : {e_sugg:.1f} MPa")
        elif is_rock(new_type):
            bits.append("qc non pertinent pour ce type (refus de pointe probable) — seul E est utilisé")
    st.caption(" · ".join(bits))


def _render_layers_table(disabled: bool = False):
    ids = _layer_ids()
    n = len(ids)

    h0, h1, h2, h3, h4, h5 = st.columns(LAYER_COLS, vertical_alignment="bottom")
    with h1:
        st.markdown("<div style='font-size:0.85em;font-weight:600;'>Type de sol</div>", unsafe_allow_html=True)

    for i, lid in enumerate(ids):
        _render_layer_row(lid, i, n, disabled=disabled)


# =============================================================
#  PAGE
# =============================================================
def show():
    _init_state()

    st.markdown(
        "<style>.katex-display{text-align:left!important;margin:.2rem 0!important;}"
        ".katex-display>.katex{text-align:left!important;}"
        ".memo-chip{display:inline-block;padding:2px 8px;border-radius:999px;"
        "background:#eef2ff;color:#3730a3;font-size:.8rem;}"
        ".small{color:#64748b;font-size:.9rem;}</style>",
        unsafe_allow_html=True,
    )

    # ---------- Barre du haut ----------
    cols = st.columns([1, 1, 1, 1, 1, 1])
    with cols[0]:
        if st.button("🏠 Accueil", use_container_width=True, key="rs_home"):
            st.session_state.page = "Accueil"
            st.rerun()
    with cols[1]:
        if st.button("🧹 Réinitialiser", use_container_width=True, key="rs_reset"):
            keep = {"press_unit", "module_unit", "detail_calc", "adv_open", "page", "abaque_w"}
            for k in list(st.session_state.keys()):
                if k not in keep:
                    st.session_state.pop(k, None)
            st.rerun()
    with cols[2]:
        st.button("💾 Enregistrer", use_container_width=True, disabled=True, help="À connecter au système JSON")
    with cols[3]:
        st.button("📂 Ouvrir", use_container_width=True, disabled=True, help="Lecture de fichiers à venir")
    with cols[4]:
        st.button("📝 Générer PDF", use_container_width=True, disabled=True, help="Export PDF à développer")
    with cols[5]:
        st.markdown(f"<div style='text-align:right;padding-top:10px;'><span class='memo-chip'>{VERSION}</span></div>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("# Raideur élastique des sols")
    st.markdown("<span class='small'>Pré-dimensionnement — sols modélisés par des ressorts verticaux "
                "(modèle de Winkler).</span>", unsafe_allow_html=True)

    with st.expander("📘 Fiche mémo (k, unités et modèle de Winkler)", expanded=False):
        st.markdown(
            r"""
- **Winkler** : $q = k \cdot w \Rightarrow k = q/w$.
- **Unités** : $q$ en kPa = kN/m² · $w$ en m · $k$ en kN/m³ ou MN/m³ (1 MN/m³ = 1000 kN/m³).
- **Ressorts en série** (colonne de sol) : $1/k_{serie} = \sum_i h_i/E_i$ — modèle à privilégier pour
  exporter $k$ vers un logiciel de dalle sur sol élastique (SCIA...) quand le profil est connu.
- **Semelle sur massif semi-infini** (Boussinesq, ordre de grandeur) : $k \approx E/[B(1-\nu^2)]$.
  Ces deux modèles répondent à des questions différentes et **ne se chaînent pas** : B et ν
  n'ont aucun effet sur le modèle en série.
- **Contrainte admissible** pour un tassement de référence $w_{adm}$ :
  $q_{adm} = k \cdot w_{adm}$ ; en kgf/cm² : $q_{adm} \approx k(\text{MN/m}^3)\cdot w_{adm}(\text{mm})/98{,}07$.
- **Rocher (schiste, calcaire, craie saine, grès)** : le CPT est en général en **refus de pointe** —
  qc n'a pas de sens physique. Utiliser une plage de $E$ issue du degré d'altération (RQD/GSI/
  pressiomètre), jamais une corrélation qc→E.
- Valeurs à valider par l'**EN 1997 (Eurocode 7)** et le rapport géotechnique.
            """
        )

    col_left, col_right = st.columns(2)

    # =========================================================
    #  COLONNE GAUCHE — entrées
    # =========================================================
    with col_left:
        st.markdown("### Informations et entrées")

        st.session_state.adv_open = st.checkbox("⚙️ Configuration avancée (unités)",
                                                value=st.session_state.adv_open)
        if st.session_state.adv_open:
            c1, c2 = st.columns(2)
            with c1:
                old = st.session_state.press_unit
                new = st.selectbox("Pressions / contraintes", ["kPa", "MPa", "kg/cm²"],
                                   index=["kPa", "MPa", "kg/cm²"].index(old))
                if new != old:
                    for kk in ("solo_q", "solo_qad"):
                        if kk in st.session_state:
                            st.session_state[kk] = from_kPa(to_kPa(st.session_state[kk], old), new)
                    st.session_state.press_unit = new
            with c2:
                oldm = st.session_state.module_unit
                newm = st.selectbox("Modules E", ["MPa", "GPa"], index=0 if oldm == "MPa" else 1)
                if newm != oldm and "solo_E" in st.session_state:
                    st.session_state.solo_E *= (0.001 if newm == "GPa" else 1000.0)
                st.session_state.module_unit = newm

        cas = st.selectbox(
            "Quel cas souhaitez-vous traiter ?",
            ("1. Raideur d'un sol (q, w)",
             "2. Modélisation d'un sol (mono / multicouche / CPT interprété)",
             "3. Raideur d'un sol – formule empirique (CPT)",
             "4. Raideur d'un plat en béton",
             "5. Convertisseur k ↔ E ↔ (q, w)",
             "6. Abaque sols"),
            index=0,
        )

        pu = st.session_state.press_unit

        if cas.startswith("1."):
            st.markdown("**Raideur à partir d'un couple (q, w)**")
            st.caption("On connaît une pression de service q et un tassement w : k = q / w.")
            st.markdown("<span class='memo-chip'>Typiquement : q à l'ELS, w = 20 mm → k pour SCIA / RDM.</span>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.solo_q = st.number_input(f"q (pression au sol) [{pu}]", min_value=0.0,
                                                          value=float(st.session_state.get("solo_q", 60.0)), step=5.0)
            with c2:
                st.session_state.solo_w = st.number_input("w (tassement) [mm]", min_value=0.001,
                                                          value=float(st.session_state.get("solo_w", 20.0)), step=5.0)

        elif cas.startswith("2."):
            st.markdown("**Modélisation d'un sol (mono / multicouche / CPT interprété)**")
            st.caption("Une ligne = sol homogène. Plusieurs lignes = profil multicouche. "
                       "Calcul en ressorts en série : 1/k = Σ(hᵢ/Eᵢ).")
            st.markdown(
                "<span class='memo-chip'>Choisir un « Type de sol » préremplit qc / Rf / E avec des valeurs "
                "typiques (modifiables) — immédiatement, dès la sélection. Pour le rocher, seul E est "
                "proposé : qc n'a pas de sens en refus de pointe.</span>", unsafe_allow_html=True)

            st.markdown("#### Couches de sol")
            _render_layers_table(disabled=False)

        elif cas.startswith("3."):
            st.markdown("**Raideur d'un sol – formule empirique (CPT)**")
            st.caption("Basée sur une valeur de qc : E = α_E (qₜ − σ'ᵥ₀), puis k ≈ E/[B(1−ν²)]. Sol supposé homogène.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.cpt_qt = st.number_input("qₜ (pointe nette) [MPa]", min_value=0.0,
                                                          value=float(st.session_state.get("cpt_qt", 5.0)), step=0.5)
            with c2:
                st.session_state.cpt_sv0 = st.number_input("σ'ᵥ₀ (contrainte eff.) [kPa]", min_value=0.0,
                                                          value=float(st.session_state.get("cpt_sv0", 100.0)), step=10.0)
            with c3:
                st.session_state.cpt_alphaE = st.number_input("α_E (CPT → E)", min_value=0.1,
                                                          value=float(st.session_state.get("cpt_alphaE", 2.5)), step=0.1)
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.cpt_B = st.number_input("B (largeur) [m]", min_value=0.1,
                                                          value=float(st.session_state.get("cpt_B", 2.0)), step=0.1)
            with c5:
                st.session_state.cpt_nu = st.number_input("ν (Poisson)", min_value=0.0, max_value=0.49,
                                                          value=float(st.session_state.get("cpt_nu", 0.30)), step=0.01)
            st.caption("⚠️ Ne s'applique qu'aux sols meubles (le CPT est en refus sur du rocher).")

        elif cas.startswith("4."):
            st.markdown("**Raideur d'un plat en béton (contact plat / béton / grout)**")
            st.caption("Contact assimilé à une compression 1D du béton (et du grout). Par défaut k = E/h.")
            st.markdown("**Géométrie du plat**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.plate_B = st.number_input("Largeur plat B [mm]", min_value=20.0,
                                                          value=float(st.session_state.get("plate_B", 200.0)), step=10.0)
            with c2:
                st.session_state.plate_L = st.number_input("Longueur plat L [mm]", min_value=20.0,
                                                          value=float(st.session_state.get("plate_L", 200.0)), step=10.0)
            with c3:
                st.session_state.plate_alpha = st.number_input("α (h_c = α·min(B,L))", min_value=0.05,
                                                          value=float(st.session_state.get("plate_alpha", 0.5)), step=0.05)
            st.markdown("**Béton support**")
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.plate_Ec = st.number_input("E_c béton [GPa]", min_value=5.0,
                                                          value=float(st.session_state.get("plate_Ec", 30.0)), step=1.0)
            with c5:
                st.session_state.plate_use_nu = st.checkbox("Appliquer le facteur (1−ν²)",
                                                          value=st.session_state.get("plate_use_nu", False),
                                                          help="Valable pour un massif semi-infini, "
                                                               "rarement justifié pour une couche mince confinée.")
            if st.session_state.plate_use_nu:
                st.session_state.plate_nu = st.number_input("ν béton", min_value=0.0, max_value=0.49,
                                                          value=float(st.session_state.get("plate_nu", 0.20)), step=0.01)
            else:
                st.session_state.plate_nu = st.session_state.get("plate_nu", 0.20)

            st.markdown("**Lit de mortier / grout (optionnel)**")
            st.session_state.plate_has_grout = st.checkbox("Présence d'un lit de mortier/grout",
                                                          value=st.session_state.get("plate_has_grout", False))
            if st.session_state.plate_has_grout:
                c6, c7 = st.columns(2)
                with c6:
                    st.session_state.plate_tg = st.number_input("Épaisseur grout t_g [mm]", min_value=1.0,
                                                          value=float(st.session_state.get("plate_tg", 20.0)), step=1.0)
                with c7:
                    st.session_state.plate_Eg = st.number_input("E_g grout [GPa]", min_value=5.0,
                                                          value=float(st.session_state.get("plate_Eg", 20.0)), step=1.0)
            else:
                st.session_state.plate_tg = st.session_state.get("plate_tg", 0.0)
                st.session_state.plate_Eg = st.session_state.get("plate_Eg", 20.0)

        elif cas.startswith("5."):
            st.markdown("**Convertisseur k ↔ E ↔ (q, w)**")
            st.caption("Choisis ce que tu connais ; l'outil déduit le reste.")
            st.session_state.conv_mode = st.radio(
                "Je connais…",
                ["k → q (pour un tassement w)", "q, w → k", "E, B, ν → k (Boussinesq)"],
                index=["k → q (pour un tassement w)", "q, w → k", "E, B, ν → k (Boussinesq)"].index(
                    st.session_state.get("conv_mode", "q, w → k")),
            )
            m = st.session_state.conv_mode
            if m.startswith("k →"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.conv_k = st.number_input("k [MN/m³]", min_value=0.0,
                                                          value=float(st.session_state.get("conv_k", 30.0)), step=1.0)
                with c2:
                    st.session_state.conv_w = st.number_input("w (tassement) [mm]", min_value=0.001,
                                                          value=float(st.session_state.get("conv_w", 20.0)), step=1.0)
            elif m.startswith("q,"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.conv_q = st.number_input(f"q [{pu}]", min_value=0.0,
                                                          value=float(st.session_state.get("conv_q", 60.0)), step=5.0)
                with c2:
                    st.session_state.conv_w = st.number_input("w [mm]", min_value=0.001,
                                                          value=float(st.session_state.get("conv_w", 20.0)), step=1.0)
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.conv_E = st.number_input(f"E [{st.session_state.module_unit}]", min_value=0.0,
                                                          value=float(st.session_state.get("conv_E", 30.0)), step=5.0)
                with c2:
                    st.session_state.conv_B = st.number_input("B [m]", min_value=0.1,
                                                          value=float(st.session_state.get("conv_B", 2.0)), step=0.1)
                with c3:
                    st.session_state.conv_nu = st.number_input("ν", min_value=0.0, max_value=0.49,
                                                          value=float(st.session_state.get("conv_nu", 0.30)), step=0.01)

        else:  # cas 6
            st.markdown("**Abaque sols – valeurs indicatives**")
            st.caption("Poids volumique γ, raideur k (MN/m³) et contrainte admissible qₐ (kg/cm²) "
                       "pour un tassement de référence — basé sur la même base de données que le "
                       "tableau multicouche. À confirmer par le géotechnicien.")

    # =========================================================
    #  COLONNE DROITE — résultats
    # =========================================================
    with col_right:
        st.markdown("### Dimensionnement / Résultats")
        st.session_state.detail_calc = st.checkbox("📘 Détail des calculs (formules + valeurs)",
                                                  value=st.session_state.detail_calc)
        detail = st.session_state.detail_calc
        pu = st.session_state.press_unit
        mu = st.session_state.module_unit

        # ---------- CAS 1 ----------
        if cas.startswith("1."):
            with st.container(border=True):
                q_kPa = to_kPa(st.session_state.get("solo_q", 0.0), pu)
                w_mm = st.session_state.get("solo_w", 20.0)
                k_kN, k_MN, w_m = k_from_qw(q_kPa, w_mm)
                etat = "ok" if k_MN > 0 else "nok"
                _bloc("Raideur de Winkler", f"k = {k_MN:,.2f} MN/m³".replace(",", " "), etat)
                if detail and k_MN > 0:
                    st.latex(r"k = \dfrac{q}{w}")
                    st.latex(f"k = \\dfrac{{{q_kPa:,.1f}}}{{{w_m:,.3f}}} "
                             f"= {k_kN:,.0f}\\,\\text{{kN/m³}} = {k_MN:,.2f}\\,\\text{{MN/m³}}")
                    _param_table([
                        ("q", "Pression de service", f"{st.session_state.get('solo_q', 0.0):,.2f}", pu),
                        ("w", "Tassement", f"{w_mm:,.2f}", "mm"),
                        ("k", "Raideur de sol", f"{k_MN:,.2f}", "MN/m³"),
                    ])

        # ---------- CAS 2 ----------
        elif cas.startswith("2."):
            with st.container(border=True):
                ids = _layer_ids()

                H_saisi = 0.0
                layers = []
                lignes_ignorees = []
                for num, lid in enumerate(ids, start=1):
                    lv = _get_layer_values(lid)
                    h, E = lv["h"], lv["E"]
                    if h > 0:
                        H_saisi += h
                        if E > 0:
                            layers.append((h, E_to_kPa(E, "MPa")))
                        else:
                            lignes_ignorees.append(num)

                k_kN, k_MN, H, E_moy_kPa = k_series(layers)

                _bloc("Ressorts en série (colonne 1D)",
                      f"k = {k_MN:,.2f} MN/m³  ·  → à utiliser pour SCIA".replace(",", " "),
                      "ok" if k_MN > 0 else "nok")
                st.caption("Tassement d'une colonne de sol d'épaisseur H sous charge répartie. "
                           "Seul le tableau des couches (h, E) influence ce résultat.")

                if abs(H - H_saisi) > 1e-6:
                    lignes_txt = ", ".join(str(n) for n in lignes_ignorees) if lignes_ignorees else "?"
                    st.warning(
                        f"⚠️ Épaisseur saisie au total : {H_saisi:.2f} m, mais seulement "
                        f"{H:.2f} m pris en compte dans le calcul (E manquant sur la/les ligne(s) "
                        f"{lignes_txt}). Complète ou supprime ces lignes pour un résultat représentatif."
                    )

                if detail and k_MN > 0:
                    st.latex(r"k_{serie} = \left(\sum_i \dfrac{h_i}{E_i}\right)^{-1}")
                    st.latex(f"k_{{serie}} = {k_kN:,.0f}\\,\\text{{kN/m³}} = {k_MN:,.2f}\\,\\text{{MN/m³}}")
                    _param_table([
                        ("H", "Épaisseur prise en compte", f"{H:,.2f}", "m"),
                        ("E_moy", "Module oedo. équivalent", f"{E_moy_kPa/1000:,.1f}", "MPa"),
                        ("k_serie", "Raideur (série)", f"{k_MN:,.2f}", "MN/m³"),
                    ])

                st.divider()
                # B / ν déplacés ICI : ils ne servent QU'AU modèle Boussinesq
                # ci-dessous, jamais au résultat "à utiliser pour SCIA" ci-dessus.
                st.caption("Paramètres du modèle de comparaison ci-dessous uniquement "
                           "(sans effet sur le résultat SCIA ci-dessus) :")
                cB, cNu = st.columns(2)
                with cB:
                    st.session_state.multi_B = st.number_input("Largeur caractéristique B [m]", min_value=0.1,
                                                              value=float(st.session_state.get("multi_B", 2.0)), step=0.1,
                                                              key="multi_B_input")
                with cNu:
                    st.session_state.multi_nu = st.number_input("ν équivalent (Poisson)", min_value=0.0, max_value=0.49,
                                                              value=float(st.session_state.get("multi_nu", 0.30)), step=0.01,
                                                              key="multi_nu_input")
                B = st.session_state.get("multi_B", 2.0)
                nu = st.session_state.get("multi_nu", 0.30)
                kB_kN, kB_MN = k_boussinesq(E_moy_kPa, B, nu)
                _bloc("Semelle sur massif (Boussinesq)", f"k ≈ {kB_MN:,.2f} MN/m³".replace(",", " "),
                      "info" if kB_MN > 0 else "nok")
                st.caption("Ordre de grandeur de comparaison (massif semi-infini homogène). "
                           "À ne pas confondre avec le modèle en série ci-dessus, ni exporter tel quel vers SCIA.")
                if detail and kB_MN > 0:
                    st.latex(r"k \approx \dfrac{E_{moy}}{B\,(1-\nu^2)}")
                    st.latex(f"k \\approx {kB_kN:,.0f}\\,\\text{{kN/m³}} = {kB_MN:,.2f}\\,\\text{{MN/m³}}")

        # ---------- CAS 3 ----------
        elif cas.startswith("3."):
            with st.container(border=True):
                qt = st.session_state.get("cpt_qt", 0.0)
                sv0 = st.session_state.get("cpt_sv0", 0.0)
                aE = st.session_state.get("cpt_alphaE", 2.5)
                E_kPa, E_MPa, delta = E_from_cpt(qt, sv0, aE)
                B = st.session_state.get("cpt_B", 2.0)
                nu = st.session_state.get("cpt_nu", 0.30)
                k_kN, k_MN = k_boussinesq(E_kPa, B, nu)
                _bloc("Raideur estimée (CPT)", f"k ≈ {k_MN:,.2f} MN/m³".replace(",", " "),
                      "ok" if k_MN > 0 else "nok")
                c1, c2 = st.columns(2)
                _metric(c1, "E estimé", E_MPa, "MPa")
                _metric(c2, "k", k_MN, "MN/m³")
                if detail and k_MN > 0:
                    st.latex(r"E = \alpha_E\,(q_t - \sigma'_{v0})")
                    st.latex(f"E = {aE:,.2f}\\,({qt*1000:,.0f}-{sv0:,.0f}) "
                             f"= {E_kPa:,.0f}\\,\\text{{kN/m²}} = {E_MPa:,.1f}\\,\\text{{MPa}}")
                    st.latex(r"k \approx \dfrac{E}{B(1-\nu^2)}")
                    st.latex(f"k \\approx {k_kN:,.0f}\\,\\text{{kN/m³}} = {k_MN:,.2f}\\,\\text{{MN/m³}}")

        # ---------- CAS 4 ----------
        elif cas.startswith("4."):
            with st.container(border=True):
                res = k_plate(
                    st.session_state.get("plate_B", 200.0), st.session_state.get("plate_L", 200.0),
                    st.session_state.get("plate_alpha", 0.5), st.session_state.get("plate_Ec", 30.0),
                    st.session_state.get("plate_use_nu", False), st.session_state.get("plate_nu", 0.20),
                    st.session_state.get("plate_has_grout", False),
                    st.session_state.get("plate_tg", 0.0), st.session_state.get("plate_Eg", 20.0),
                )
                _bloc("Raideur du contact plat/béton", f"k_eq = {res['keq_MNpm3']:,.1f} MN/m³".replace(",", " "),
                      "ok" if res["keq_MNpm3"] > 0 else "nok")
                if detail and res["hc"] > 0:
                    st.latex(r"h_c = \alpha\,\min(B,L)")
                    st.latex(f"h_c = {res['hc']*1000:,.1f}\\,\\text{{mm}}")
                    st.latex(r"k_c = \dfrac{E_c}{h_c}" +
                             (r"\,(1-\nu^2)^{-1}" if st.session_state.get("plate_use_nu") else ""))
                    st.latex(f"k_c = {res['kc']:,.0f}\\,\\text{{kN/m³}}")
                    if st.session_state.get("plate_has_grout") and res["kg"] > 0:
                        st.latex(r"\dfrac{1}{k_{eq}} = \dfrac{1}{k_c} + \dfrac{1}{k_g}")
                        st.latex(f"k_g = {res['kg']:,.0f}\\,\\text{{kN/m³}}")
                    _param_table([
                        ("h_c", "Épaisseur mobilisée", f"{res['hc']*1000:,.1f}", "mm"),
                        ("E_c", "Module béton", f"{st.session_state.get('plate_Ec',30.0):,.1f}", "GPa"),
                        ("k_eq", "Raideur équivalente", f"{res['keq_MNpm3']:,.1f}", "MN/m³"),
                    ])

        # ---------- CAS 5 : convertisseur ----------
        elif cas.startswith("5."):
            with st.container(border=True):
                m = st.session_state.get("conv_mode", "q, w → k")
                if m.startswith("k →"):
                    k_MN = st.session_state.get("conv_k", 0.0)
                    w_mm = st.session_state.get("conv_w", 20.0)
                    q_kPa = k_MN * 1000.0 * (w_mm / 1000.0)  # k[kN/m³]·w[m]
                    _bloc("Pression mobilisée", f"q = {from_kPa(q_kPa, pu):,.2f} {pu}".replace(",", " "),
                          "ok" if q_kPa > 0 else "nok")
                    if detail:
                        st.latex(r"q = k \cdot w")
                        st.latex(f"q = {k_MN:,.2f}\\cdot10^3 \\cdot {w_mm/1000:,.3f} "
                                 f"= {q_kPa:,.1f}\\,\\text{{kN/m²}} = {from_kPa(q_kPa,pu):,.2f}\\,\\text{{{pu}}}")
                elif m.startswith("q,"):
                    q_kPa = to_kPa(st.session_state.get("conv_q", 0.0), pu)
                    w_mm = st.session_state.get("conv_w", 20.0)
                    k_kN, k_MN, w_m = k_from_qw(q_kPa, w_mm)
                    _bloc("Raideur", f"k = {k_MN:,.2f} MN/m³".replace(",", " "), "ok" if k_MN > 0 else "nok")
                    if detail and k_MN > 0:
                        st.latex(r"k = q / w")
                        st.latex(f"k = {q_kPa:,.1f}/{w_m:,.3f} = {k_MN:,.2f}\\,\\text{{MN/m³}}")
                else:
                    E_kPa = E_to_kPa(st.session_state.get("conv_E", 0.0), mu)
                    B = st.session_state.get("conv_B", 2.0)
                    nu = st.session_state.get("conv_nu", 0.30)
                    k_kN, k_MN = k_boussinesq(E_kPa, B, nu)
                    _bloc("Raideur (Boussinesq)", f"k ≈ {k_MN:,.2f} MN/m³".replace(",", " "),
                          "ok" if k_MN > 0 else "nok")
                    if detail and k_MN > 0:
                        st.latex(r"k \approx \dfrac{E}{B(1-\nu^2)}")
                        st.latex(f"k \\approx {k_MN:,.2f}\\,\\text{{MN/m³}}")

        # ---------- CAS 6 : abaque (source = SOIL_DB) ----------
        else:
            with st.container(border=True):
                st.markdown("#### Tassement de référence")
                st.session_state.abaque_w = st.number_input(
                    "w_adm [mm]", min_value=1.0, max_value=100.0,
                    value=float(st.session_state.abaque_w), step=5.0,
                    help="Convertit k (MN/m³) en qₐ (kg/cm²). En Belgique, 20 mm est courant en service.")
                w_adm = st.session_state.abaque_w
                factor_q = w_adm / KGF_PER_CM2_TO_KPA  # qₐ(kgf/cm²) ≈ k(MN/m³)·w(mm)/98.07

                rows = []
                for name, d in SOIL_DB.items():
                    if name == "Personnalisé" or d.get("k_min") is None:
                        continue
                    rows.append({
                        "Catégorie": d["category"],
                        "Type de sol": name,
                        "γ (kN/m³)": d["gamma"],
                        "k_min (MN/m³)": d["k_min"],
                        "k_max (MN/m³)": d["k_max"],
                        "qₐ_min (kg/cm²)": round(d["k_min"] * factor_q, 2),
                        "qₐ_max (kg/cm²)": round(d["k_max"] * factor_q, 2),
                    })
                df_ab = pd.DataFrame(rows)
                st.dataframe(df_ab, use_container_width=True, hide_index=True)

                st.markdown("#### Fiche sol")
                noms = [r["Type de sol"] for r in rows]
                default_idx = noms.index("Sable moyennement compact") if "Sable moyennement compact" in noms else 0
                choix = st.selectbox("Type de sol :", noms, index=default_idx)
                d = SOIL_DB[choix]
                q_min = d["k_min"] * factor_q
                q_max = d["k_max"] * factor_q
                _bloc(choix, f"qₐ ≈ {q_min:,.2f}–{q_max:,.2f} kg/cm²".replace(",", " "), "info")
                st.markdown(d["desc"])
                lignes = [f"- Catégorie : **{d['category']}**",
                          f"- γ ≈ **{d['gamma']} kN/m³**",
                          f"- k ≈ **{d['k_min']} à {d['k_max']} MN/m³**",
                          f"- E ≈ **{d['E_min']} à {d['E_max']} MPa**"]
                if d.get("cpt_ok") and d.get("qc_min") is not None:
                    lignes.append(f"- qc ≈ **{d['qc_min']} à {d['qc_max']} MPa** (α qc→E ≈ {d['alpha_qc']})")
                else:
                    lignes.append("- qc : **non pertinent** (refus de pointe probable) — caractériser par RQD/pressiomètre.")
                lignes.append(f"- pour w_adm = **{w_adm:.0f} mm** → qₐ ≈ **{q_min:.2f} à {q_max:.2f} kg/cm²**")
                st.markdown("  \n".join(lignes))

        st.divider()
        st.markdown("<div class='small'>Valeurs de k, E et qₐ indicatives (littérature géotechnique / retours "
                    "d'expérience), réservées au pré-dimensionnement — en particulier pour le rocher, où la "
                    "dispersion peut être très importante selon le degré de fracturation/altération. "
                    "Se référer systématiquement au rapport géotechnique et à l'EN 1997 (Eurocode 7) pour le "
                    "dimensionnement final.</div>",
                    unsafe_allow_html=True)

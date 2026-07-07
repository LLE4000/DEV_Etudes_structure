# =============================================================
#  raideur_sol.py — Raideur élastique des sols (modèle de Winkler)
#  VERSION 4.0
#
#  Évolutions vs 3.3 :
#   1. MULTI-SONDAGES : le Cas 2 gère plusieurs sondages par projet
#      (même principe que les poutres multiples de poutre.py : clés
#      d'état préfixées snd{id}_, ajout/copie/suppression par
#      identifiant unique). Chaque sondage a son nom (CPT01, CPT02...),
#      son niveau d'assise et son propre tableau de couches.
#   2. IMPORT PDF PAR IA : un rapport d'essais (CPT/forage) peut être
#      téléversé ; il est envoyé à l'API Anthropic (Claude) qui renvoie
#      les couches en JSON strict (nom du sondage, épaisseurs, types,
#      qc, E, nappe). Les types sont mappés vers SOIL_DB par mots-clés
#      et les valeurs manquantes préremplies par la corrélation α·qc ou
#      les valeurs typiques. TOUT est ensuite éditable : l'IA propose,
#      l'ingénieur dispose (bandeau d'avertissement systématique).
#      Clé API : st.secrets["ANTHROPIC_API_KEY"] ou saisie manuelle.
#   3. PROFONDEUR D'INFLUENCE : option "Limiter à 2·B" qui tronque
#      automatiquement le profil à la profondeur d'influence sous
#      l'assise (garde-fou contre le k artificiellement faible obtenu
#      en sommant tout le sondage). Avertissement si H saisi >> 2·B
#      quand l'option est désactivée.
#   4. PANNEAU "RAIDEURS À ENCODER DANS SCIA" : tableau récapitulatif
#      par sondage (k en MN/m³ ET kN/m³), enveloppe min/max entre
#      sondages, rappel de faire tourner la dalle dans les deux bornes.
#   5. RAPPORT PDF (reportlab, style sobre) : hypothèses et méthode,
#      tableau des couches par sondage, application numérique de
#      1/k = Σ hi/Ei, récapitulatif SCIA, références normatives
#      (EN 1997-1, ISO 22476-1, Winkler). Bouton dans le panneau SCIA.
#
#  Winkler : q = k · w  ->  k = q / w
#    q [kPa = kN/m²], w [m], k [kN/m³]  (1 MN/m³ = 1000 kN/m³)
# =============================================================

import base64
import io
import json as _json
import math
import re
from datetime import date

import pandas as pd
import streamlit as st

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors as _rl_colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


# =============================================================
#  CONSTANTES
# =============================================================
KGF_PER_CM2_TO_KPA = 98.0665          # 1 kgf/cm² = 98.0665 kPa
VERSION = "v4.0"
AI_MODEL = "claude-sonnet-4-6"        # modèle utilisé pour l'import PDF

C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6", "info": "#eef2ff"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌", "info": "ℹ️"}

# Largeurs de colonnes du tableau des couches (h | Type | qc | Rf | E | Action)
LAYER_COLS = [0.8, 2.2, 1.0, 0.8, 1.0, 0.5]

LAYER_FIELDS = ("h", "type", "type_prev", "qc", "rf", "E")


# =============================================================
#  BASE DE DONNÉES SOLS (contexte belge) — SOURCE UNIQUE
#  cpt_ok=False : qc sans sens physique (refus de pointe probable).
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
    "Sable argileux / argile sableuse": dict(
        category="Sable", gamma=19.0,
        qc_min=3.0, qc_max=8.0, alpha_qc=4.5, rf_typ=1.5,
        E_min=12.0, E_max=35.0, k_min=20, k_max=60, cpt_ok=True,
        desc="Mélange intermédiaire (fréquent dans les formations bruxelliennes/yprésiennes) : "
             "comportement plastique, drainage lent."),
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

# Mots-clés (minuscules, sans accents traités simplement) -> type SOIL_DB,
# utilisés pour mapper les libellés libres renvoyés par l'IA d'import PDF.
_SOIL_KEYWORDS = [
    (("remblai", "debris", "débris", "bricaillon", "revetement", "revêtement",
      "terre vegetale", "terre végétale", "asphalte"), "Remblais / terre végétale"),
    (("tourbe", "organique"), "Tourbe"),
    (("argile sableuse", "sable argileux", "argilo-sableux", "sablo-argileux"),
     "Sable argileux / argile sableuse"),
    (("boom", "ypres", "yprési", "surconsolid"), "Argile surconsolidée (Boom/Ypresienne)"),
    (("argile tres molle", "argile très molle"), "Argile très molle"),
    (("argile molle", "argile moyenne"), "Argile molle à moyenne"),
    (("argile ferme", "argile raide", "argile"), "Argile ferme / raide"),
    (("limon", "loess", "silt"), "Limon (loess)"),
    (("gravier", "grave", "graveleux"), "Sable graveleux / grave compacte"),
    (("sable lache", "sable lâche", "sable peu compact"), "Sable lâche"),
    (("sable dense", "sable tres compact", "sable très compact"), "Sable dense"),
    (("sable",), "Sable moyennement compact"),
    (("craie alter",), "Craie altérée"),
    (("craie",), "Craie saine"),
    (("calcaire alter", "calcaire fractur", "calcaire karst"), "Calcaire fracturé / altéré"),
    (("calcaire",), "Calcaire sain"),
    (("schiste decompos", "schiste décompos", "w4", "w5"), "Schiste houiller décomposé (W4-W5)"),
    (("schiste alter", "schiste altér", "w3"), "Schiste houiller altéré (W3)"),
    (("schiste",), "Schiste houiller sain (W1-W2)"),
    (("gres alter", "grès altér"), "Grès altéré"),
    (("gres", "grès"), "Grès sain"),
]


def soil_types_list():
    return ["—"] + list(SOIL_DB.keys())


def match_soil_type(label: str) -> str:
    """Mappe un libellé libre (log de forage, sortie IA) vers le type
    SOIL_DB le plus proche par mots-clés. '—' si aucun match."""
    low = (label or "").strip().lower()
    if not low:
        return "—"
    if label in SOIL_DB:
        return label
    for keys, target in _SOIL_KEYWORDS:
        if any(k in low for k in keys):
            return target
    return "Personnalisé"


def _mid(lo, hi):
    if lo is None or hi is None:
        return None
    return round((lo + hi) / 2.0, 1)


def soil_default_qc(soil_type: str):
    d = SOIL_DB.get(soil_type)
    if not d or not d.get("cpt_ok", False):
        return None
    return _mid(d.get("qc_min"), d.get("qc_max"))


def soil_default_Rf(soil_type: str):
    d = SOIL_DB.get(soil_type)
    return d.get("rf_typ") if d else None


def soil_default_E(soil_type: str):
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
    return {"kPa": value, "MPa": value * 1000.0, "kg/cm²": value * KGF_PER_CM2_TO_KPA}.get(unit, value)


def from_kPa(value_kPa: float, unit: str) -> float:
    return {"kPa": value_kPa, "MPa": value_kPa / 1000.0, "kg/cm²": value_kPa / KGF_PER_CM2_TO_KPA}.get(unit, value_kPa)


def E_to_kPa(E: float, unit: str) -> float:
    return {"MPa": E * 1000.0, "GPa": E * 1_000_000.0}.get(unit, E)


def kNpm3_to_MNpm3(v: float) -> float:
    return v / 1000.0


def suggest_E_from_qc(qc_MPa, soil_type: str):
    """E ≈ α·qc (MPa) selon SOIL_DB. None si non pertinent/invalide."""
    d = SOIL_DB.get(soil_type, {})
    alpha = d.get("alpha_qc")
    if alpha is None or qc_MPa is None or (isinstance(qc_MPa, float) and math.isnan(qc_MPa)) or qc_MPa <= 0:
        return None
    return round(alpha * qc_MPa, 1)


# =============================================================
#  CALCULS (fonctions pures — aucun Streamlit)
# =============================================================
def k_from_qw(q_kPa: float, w_mm: float):
    w_m = w_mm / 1000.0
    if w_m <= 0:
        return 0.0, 0.0, w_m
    k = q_kPa / w_m
    return k, kNpm3_to_MNpm3(k), w_m


def k_series(layers):
    """1/k_serie = Σ h_i/E_i. layers = [(h_m, E_kPa)].
    Retourne (k_kNpm3, k_MNpm3, H_m, E_moy_kPa)."""
    denom = 0.0
    H = 0.0
    for h, E in layers:
        if h > 0 and E > 0:
            denom += h / E
            H += h
    if denom <= 0:
        return 0.0, 0.0, H, 0.0
    k = 1.0 / denom
    return k, kNpm3_to_MNpm3(k), H, k * H


def k_boussinesq(E_kPa: float, B_m: float, nu: float):
    if E_kPa <= 0 or B_m <= 0 or nu >= 1.0:
        return 0.0, 0.0
    k = E_kPa / (B_m * (1.0 - nu ** 2))
    return k, kNpm3_to_MNpm3(k)


def E_from_cpt(qt_MPa: float, sv0_kPa: float, alpha_E: float):
    delta = max(qt_MPa * 1000.0 - sv0_kPa, 0.0)
    E_kPa = alpha_E * delta
    return E_kPa, E_kPa / 1000.0, delta


def k_plate(B_mm, L_mm, alpha, Ec_GPa, use_nu, nu_c,
            has_grout=False, tg_mm=0.0, Eg_GPa=20.0):
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
#  ÉTAT MULTI-SONDAGES
#  soundings = [{"id": 1, "nom": "CPT01"}, ...]
#  Clés par couche : snd{sid}_layer_{lid}_{champ}
#  Ordre des couches : snd{sid}_layer_order = [lid, ...]
# =============================================================
def _layer_key(sid: int, lid: int, field: str) -> str:
    return f"snd{sid}_layer_{lid}_{field}"


def _order_key(sid: int) -> str:
    return f"snd{sid}_layer_order"


def _layer_ids(sid: int):
    return list(st.session_state.get(_order_key(sid), []))


def _new_layer_id(sid: int) -> int:
    ids = _layer_ids(sid)
    return (max(ids) + 1) if ids else 1


def _init_layer(sid: int, lid: int, h=1.0, soil_type="—", qc=0.0, rf=0.0, E=0.0):
    st.session_state[_layer_key(sid, lid, "h")] = float(h)
    st.session_state[_layer_key(sid, lid, "type")] = soil_type
    # type_prev = type au moment de la création : le préremplissage ne se
    # déclenche que sur un changement ULTÉRIEUR, jamais sur des valeurs
    # importées (l'import IA fixe déjà qc/E — il ne faut pas les écraser).
    st.session_state[_layer_key(sid, lid, "type_prev")] = soil_type
    st.session_state[_layer_key(sid, lid, "qc")] = float(qc)
    st.session_state[_layer_key(sid, lid, "rf")] = float(rf)
    st.session_state[_layer_key(sid, lid, "E")] = float(E)


def _add_layer(sid: int):
    lid = _new_layer_id(sid)
    st.session_state[_order_key(sid)].append(lid)
    _init_layer(sid, lid)


def _delete_layer(sid: int, lid: int):
    ids = _layer_ids(sid)
    if len(ids) <= 1 or lid not in ids:
        return
    st.session_state[_order_key(sid)].remove(lid)
    for f in LAYER_FIELDS:
        st.session_state.pop(_layer_key(sid, lid, f), None)


def _get_layer_values(sid: int, lid: int):
    return {
        "h": float(st.session_state.get(_layer_key(sid, lid, "h"), 0.0) or 0.0),
        "type": st.session_state.get(_layer_key(sid, lid, "type"), "—"),
        "qc": float(st.session_state.get(_layer_key(sid, lid, "qc"), 0.0) or 0.0),
        "rf": float(st.session_state.get(_layer_key(sid, lid, "rf"), 0.0) or 0.0),
        "E": float(st.session_state.get(_layer_key(sid, lid, "E"), 0.0) or 0.0),
    }


def _sounding_ids():
    return [int(s["id"]) for s in st.session_state.get("soundings", [])]


def _sounding_name(sid: int) -> str:
    return str(st.session_state.get(f"snd{sid}_nom", f"Sondage {sid}"))


def _new_sounding_id() -> int:
    ids = _sounding_ids()
    return (max(ids) + 1) if ids else 1


def _add_sounding(nom: str = None, first_layer=True):
    sid = _new_sounding_id()
    nom = nom or f"CPT{sid:02d}"
    st.session_state.soundings.append({"id": sid, "nom": nom})
    st.session_state[f"snd{sid}_nom"] = nom
    st.session_state[f"snd{sid}_assise"] = 0.0
    st.session_state[_order_key(sid)] = []
    if first_layer:
        st.session_state[_order_key(sid)].append(1)
        _init_layer(sid, 1)
    return sid


def _delete_sounding(sid: int):
    if len(st.session_state.soundings) <= 1:
        return
    st.session_state.soundings = [s for s in st.session_state.soundings if int(s["id"]) != sid]
    prefix = f"snd{sid}_"
    for k in [k for k in list(st.session_state.keys()) if k.startswith(prefix)]:
        st.session_state.pop(k, None)


def _copy_sounding(src_sid: int):
    """Copie intégrale d'un sondage (nom + ' (copie)', couches, assise)."""
    sid = _new_sounding_id()
    nom = f"{_sounding_name(src_sid)} (copie)"
    st.session_state.soundings.append({"id": sid, "nom": nom})
    st.session_state[f"snd{sid}_nom"] = nom
    st.session_state[f"snd{sid}_assise"] = float(st.session_state.get(f"snd{src_sid}_assise", 0.0) or 0.0)
    st.session_state[_order_key(sid)] = list(_layer_ids(src_sid))
    for lid in _layer_ids(src_sid):
        for f in LAYER_FIELDS:
            st.session_state[_layer_key(sid, lid, f)] = st.session_state.get(_layer_key(src_sid, lid, f))


def _compute_sounding_k(sid: int, H_lim=None):
    """
    k_serie d'un sondage. Si H_lim (m) est fourni, le profil est tronqué
    à cette profondeur sous l'assise (la dernière couche est écrêtée) :
    garde-fou "profondeur d'influence" — sommer tout un sondage de 30 m
    donne un k artificiellement faible et faux pour une fondation.
    Retourne un dict complet pour l'affichage et le rapport PDF.
    """
    rows = []           # (num, h_utilisée, h_saisie, type, qc, E, statut)
    layers = []         # [(h, E_kPa)] pour k_series
    ignored = []
    H_saisi = 0.0
    cum = 0.0
    for num, lid in enumerate(_layer_ids(sid), start=1):
        lv = _get_layer_values(sid, lid)
        h, E = lv["h"], lv["E"]
        if h <= 0:
            continue
        H_saisi += h
        h_use = h
        clipped = False
        if H_lim is not None:
            if cum >= H_lim:
                h_use = 0.0
            elif cum + h > H_lim:
                h_use = H_lim - cum
                clipped = True
        cum += h
        if E > 0 and h_use > 0:
            layers.append((h_use, E_to_kPa(E, "MPa")))
            statut = "écrêtée à la prof. d'influence" if clipped else "prise en compte"
        elif E > 0:
            statut = "hors profondeur d'influence"
        else:
            statut = "ignorée (E manquant)"
            ignored.append(num)
        rows.append((num, h_use if E > 0 else 0.0, h, lv["type"], lv["qc"], E, statut))
    k_kN, k_MN, H, E_moy_kPa = k_series(layers)
    return {"k_kN": k_kN, "k_MN": k_MN, "H": H, "H_saisi": H_saisi,
            "E_moy_kPa": E_moy_kPa, "ignored": ignored, "rows": rows}


# =============================================================
#  IMPORT PDF PAR IA (API Anthropic)
# =============================================================
_AI_PROMPT = """Tu es un assistant géotechnique. Le PDF joint est un rapport d'essais de sol
(CPT, forages, coupes lithologiques). Extrais-en un profil de couches PAR SONDAGE/FORAGE,
en combinant si possible la lithologie du forage et les valeurs qc des CPT.

Réponds UNIQUEMENT avec un JSON valide, sans texte avant/après, sans balises markdown,
au format strict :
{"sondages":[{"nom":"CPT01","nappe_m":11.0,"couches":[
  {"de_m":0.0,"a_m":4.0,"type":"Remblais graviers et débris","qc_MPa":null,"E_MPa":null},
  {"de_m":4.0,"a_m":8.0,"type":"Sable jaune-orange","qc_MPa":9.0,"E_MPa":null}]}]}

Règles :
- "nom" : identifiant du sondage tel qu'il apparaît (CPT01, F01, BH01...). Si un forage
  et un CPT sont au même endroit, fusionne-les en un seul profil.
- "de_m"/"a_m" : profondeurs sous le terrain naturel, en mètres.
- "type" : description lithologique courte en français.
- "qc_MPa" : résistance de cône MOYENNE PRUDENTE de la couche lue sur les graphiques CPT
  (ignorer les pics isolés), null si non disponible.
- "E_MPa" : uniquement si le rapport donne explicitement un module, sinon null.
- "nappe_m" : profondeur de la nappe si mentionnée, sinon null.
- Regroupe en couches homogènes (4 à 8 couches max par sondage), n'invente aucune valeur."""


def _get_api_key():
    key = ""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    return key or st.session_state.get("ai_api_key", "")


def _extract_soundings_from_pdf(pdf_bytes: bytes, api_key: str):
    """Envoie le PDF à l'API Anthropic, retourne la liste 'sondages' du JSON.
    Lève une exception avec message lisible en cas d'échec."""
    if not _HAS_REQUESTS:
        raise RuntimeError("Le module 'requests' n'est pas disponible dans cet environnement.")
    b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")
    payload = {
        "model": AI_MODEL,
        "max_tokens": 3000,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
                {"type": "text", "text": _AI_PROMPT},
            ],
        }],
    }
    resp = _requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json=payload, timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API Anthropic : HTTP {resp.status_code} — {resp.text[:300]}")
    data = resp.json()
    txt = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    txt = re.sub(r"```(json)?", "", txt).strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise RuntimeError("Réponse IA sans JSON exploitable.")
    parsed = _json.loads(m.group(0))
    sondages = parsed.get("sondages", [])
    if not isinstance(sondages, list) or not sondages:
        raise RuntimeError("JSON reçu mais aucune clé 'sondages' exploitable.")
    return sondages


def _import_ai_soundings(sondages: list) -> list:
    """Crée les sondages/couches en session à partir du JSON IA.
    E : valeur du rapport si donnée, sinon α·qc, sinon valeur typique du
    type (à VÉRIFIER par l'utilisateur). Retourne les noms créés."""
    created = []
    for snd in sondages:
        nom = str(snd.get("nom", "") or f"Import {len(created) + 1}")
        sid = _add_sounding(nom=nom, first_layer=False)
        nappe = snd.get("nappe_m")
        if nappe is not None:
            try:
                st.session_state[f"snd{sid}_nappe"] = float(nappe)
            except Exception:
                pass
        couches = snd.get("couches", []) or []
        for c in couches:
            try:
                de = float(c.get("de_m", 0.0) or 0.0)
                a = float(c.get("a_m", 0.0) or 0.0)
            except Exception:
                continue
            h = max(0.0, a - de)
            if h <= 0:
                continue
            t = match_soil_type(str(c.get("type", "") or ""))
            qc = c.get("qc_MPa")
            try:
                qc = float(qc) if qc is not None else 0.0
            except Exception:
                qc = 0.0
            E = c.get("E_MPa")
            try:
                E = float(E) if E is not None else 0.0
            except Exception:
                E = 0.0
            if E <= 0:
                e_sugg = suggest_E_from_qc(qc, t)
                if e_sugg is not None:
                    E = e_sugg
                else:
                    E = soil_default_E(t) or 0.0
            rf = soil_default_Rf(t) or 0.0
            lid = _new_layer_id(sid)
            st.session_state[_order_key(sid)].append(lid)
            _init_layer(sid, lid, h=round(h, 2), soil_type=t, qc=qc, rf=rf, E=E)
        if not _layer_ids(sid):
            st.session_state[_order_key(sid)].append(1)
            _init_layer(sid, 1)
        created.append(nom)
    return created


# =============================================================
#  RAPPORT PDF (reportlab — style sobre)
# =============================================================
def build_soil_report_pdf(project: dict, soundings_data: list, B: float,
                          use_influence: bool, H_lim, w_adm_mm: float) -> bytes:
    """
    Rapport PDF : hypothèses/méthode, tableau des couches et application
    numérique 1/k = Σ hi/Ei par sondage, récapitulatif SCIA, références.
    soundings_data : [{"nom", "assise", "nappe", "res": dict de
    _compute_sounding_k}].
    """
    if not _HAS_REPORTLAB:
        raise RuntimeError("reportlab n'est pas installé (pip install reportlab).")

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=8,
                        spaceAfter=3, textColor=_rl_colors.HexColor("#1f2937"))
    BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=9.2, leading=12.5)
    SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=7.8,
                           textColor=_rl_colors.HexColor("#475569"))
    CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.2, leading=10.2)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")

    def P(t, b=False):
        return Paragraph(t, CELLB if b else CELL)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=13 * mm,
                            title="Note de calcul — Raideur élastique du sol")
    story = []
    story.append(Paragraph("Note de calcul — Raideur élastique du sol (modèle de Winkler)", H1))
    meta = []
    if project.get("nom"):
        meta.append(f"Projet : <b>{project['nom']}</b>")
    meta.append(f"Date : {project.get('date', date.today().strftime('%d/%m/%Y'))}")
    meta.append(f"Module raideur_sol {VERSION}")
    story.append(Paragraph(" — ".join(meta), SMALL))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. Méthode et hypothèses", H2))
    story.append(Paragraph(
        "Le sol est modélisé par des ressorts verticaux indépendants (modèle de Winkler) : "
        "q = k·w, avec q la pression de contact [kPa], w le tassement [m] et k le coefficient "
        "de réaction [kN/m³]. Pour un profil multicouche, le coefficient est obtenu par le "
        "modèle des ressorts en série (tassement œdométrique 1D d'une colonne de sol) : "
        "1/k = Σ (h<sub>i</sub>/E<sub>i</sub>), où h<sub>i</sub> est l'épaisseur et E<sub>i</sub> "
        "le module de déformation de la couche i. Les modules E sont issus du rapport d'essais "
        "lorsqu'ils y figurent ; à défaut, ils sont estimés par la corrélation indicative "
        "E ≈ α·q<sub>c</sub> (q<sub>c</sub> : résistance de cône CPT, α selon la nature du sol) "
        "ou par des valeurs typiques de littérature — ces estimations sont à valider par le "
        "géotechnicien.", BODY))
    infl_txt = (f"Le profil est limité à la profondeur d'influence H = 2·B = {H_lim:.2f} m sous "
                f"le niveau d'assise (B = {B:.2f} m : largeur caractéristique de la zone chargée)."
                if (use_influence and H_lim) else
                "Le profil n'est pas tronqué à une profondeur d'influence : l'épaisseur totale "
                "saisie est prise en compte (à justifier).")
    story.append(Paragraph(infl_txt, BODY))

    story.append(Paragraph("2. Sondages et calcul du coefficient de réaction", H2))
    for sd in soundings_data:
        res = sd["res"]
        story.append(Spacer(1, 3))
        titre = f"Sondage {sd['nom']}"
        extras = []
        if sd.get("assise"):
            extras.append(f"assise à {sd['assise']:.2f} m sous TN")
        if sd.get("nappe") is not None:
            extras.append(f"nappe ≈ {sd['nappe']:.1f} m sous TN")
        if extras:
            titre += " (" + ", ".join(extras) + ")"
        story.append(Paragraph(titre, ParagraphStyle(
            "H3", parent=BODY, fontName="Helvetica-Bold", fontSize=9.6, spaceAfter=2)))

        data = [[P("N°", True), P("h [m]", True), P("Nature", True),
                 P("qc [MPa]", True), P("E [MPa]", True), P("Statut", True)]]
        for (num, h_use, h, t, qc, E, statut) in res["rows"]:
            data.append([P(str(num)), P(f"{h:.2f}"), P(t),
                         P(f"{qc:.1f}" if qc > 0 else "—"),
                         P(f"{E:.0f}" if E > 0 else "—"), P(statut)])
        t_tbl = Table(data, colWidths=[10 * mm, 15 * mm, 68 * mm, 20 * mm, 20 * mm, 45 * mm])
        t_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, _rl_colors.HexColor("#94a3b8")),
            ("BACKGROUND", (0, 0), (-1, 0), _rl_colors.HexColor("#eef2f7")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_tbl)

        if res["k_MN"] > 0:
            terms = " + ".join(f"{h_use:.2f}/{E:.0f}" for (_, h_use, _, _, _, E, s) in res["rows"]
                               if h_use > 0 and E > 0)
            story.append(Paragraph(
                f"1/k = Σ h<sub>i</sub>/E<sub>i</sub> = {terms}  (h en m, E en MPa) "
                f"→ H pris en compte = {res['H']:.2f} m ; "
                f"E<sub>moy</sub> = {res['E_moy_kPa']/1000:.1f} MPa ; "
                f"<b>k = {res['k_kN']:,.0f} kN/m³ = {res['k_MN']:.2f} MN/m³</b>".replace(",", " "),
                BODY))
        else:
            story.append(Paragraph("Profil incomplet : k non calculable.", BODY))
        if res["ignored"]:
            story.append(Paragraph(
                f"Couches ignorées (E manquant) : {', '.join(map(str, res['ignored']))}.", SMALL))

    story.append(Paragraph("3. Récapitulatif — valeurs à encoder dans SCIA", H2))
    data = [[P("Sondage", True), P("k [MN/m³]", True), P("k [kN/m³]", True)]]
    k_vals = []
    for sd in soundings_data:
        res = sd["res"]
        if res["k_MN"] > 0:
            k_vals.append(res["k_MN"])
            data.append([P(sd["nom"]), P(f"{res['k_MN']:.2f}"),
                         P(f"{res['k_kN']:,.0f}".replace(",", " "))])
        else:
            data.append([P(sd["nom"]), P("—"), P("—")])
    t_rec = Table(data, colWidths=[70 * mm, 40 * mm, 40 * mm])
    t_rec.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, _rl_colors.HexColor("#94a3b8")),
        ("BACKGROUND", (0, 0), (-1, 0), _rl_colors.HexColor("#eef2f7")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_rec)
    if len(k_vals) >= 2:
        story.append(Paragraph(
            f"Enveloppe entre sondages : k = {min(k_vals):.2f} à {max(k_vals):.2f} MN/m³. "
            "Il est recommandé de calculer la dalle avec les deux bornes (étude de "
            "sensibilité) plutôt qu'avec une valeur unique — le coefficient de réaction "
            "n'est pas un paramètre de résistance et ne fait pas l'objet d'un facteur "
            "partiel dans l'EN 1997.", BODY))
    story.append(Paragraph(
        f"Pour information, la pression mobilisée pour un tassement de référence "
        f"w = {w_adm_mm:.0f} mm vaut q = k·w (ex. k = 5 MN/m³ → q ≈ "
        f"{5 * 1000 * w_adm_mm / 1000:.0f} kPa).", SMALL))

    story.append(Paragraph("4. Références", H2))
    story.append(Paragraph(
        "• NBN EN 1997-1 (Eurocode 7) + ANB — calcul géotechnique ; valeurs caractéristiques "
        "des paramètres de sol : §2.4.5.2 (estimation prudente).<br/>"
        "• ISO 22476-1 — essais de pénétration statique (CPT), norme d'exécution des essais "
        "du rapport géotechnique de référence.<br/>"
        "• Modèle de Winkler (1867) ; corrélations E ≈ α·q<sub>c</sub> : littérature "
        "géotechnique courante (indicatives, à valider par le géotechnicien).<br/>"
        "• Les valeurs de k du présent document relèvent du pré-dimensionnement et doivent "
        "être confrontées au rapport géotechnique du projet.", BODY))

    doc.build(story)
    return buf.getvalue()


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
#  GRAPHIQUES
# =============================================================
def _norm_log(x, lo, hi):
    if x is None or x <= 0:
        return 0.0
    lo = max(lo, 0.1)
    hi = max(hi, lo * 1.0001)
    v = (math.log10(x) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return min(1.0, max(0.0, v))


def _render_soil_profile_chart(sid: int):
    if not _HAS_MPL:
        st.info("Installe matplotlib (`pip install matplotlib`) pour afficher ce graphique.")
        return
    bands = []
    depth = 0.0
    for lid in _layer_ids(sid):
        lv = _get_layer_values(sid, lid)
        if lv["h"] <= 0:
            continue
        bands.append((depth, depth + lv["h"], lv["h"], lv["E"], lv["type"]))
        depth += lv["h"]
    if not bands:
        st.info("Renseigne au moins une couche avec h > 0.")
        return
    E_vals = [b[3] for b in bands if b[3] > 0]
    E_lo, E_hi = (min(E_vals), max(E_vals)) if E_vals else (1.0, 100.0)
    fig, ax = plt.subplots(figsize=(3.0, 5.0))
    cmap = plt.get_cmap("YlOrBr")
    for (z0, z1, h, E, t) in bands:
        color = cmap(0.12) if E <= 0 else cmap(0.25 + 0.65 * _norm_log(E, E_lo, E_hi))
        ax.barh((z0 + z1) / 2.0, width=1.0, height=h, left=0.0,
                color=color, edgecolor="black", linewidth=0.8)
        label = f"{t}\nh={h:.2f} m" + (f" · E={E:.0f} MPa" if E > 0 else "\n⚠ E manquant")
        ax.text(0.5, (z0 + z1) / 2.0, label, ha="center", va="center", fontsize=7, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(depth, 0)
    ax.set_ylabel("Profondeur sous assise (m)")
    ax.set_title(f"Profil {_sounding_name(sid)} (couleur ∝ E, échelle log)", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_layer_sensitivity_chart(sid: int, k_actual_MN: float):
    if not _HAS_MPL:
        st.info("Installe matplotlib (`pip install matplotlib`) pour afficher ce graphique.")
        return
    ids = _layer_ids(sid)
    if not ids:
        st.info("Ajoute au moins une couche.")
        return
    labels = [f"Couche {i + 1} — {_get_layer_values(sid, lid)['type']}" for i, lid in enumerate(ids)]
    sel_label = st.selectbox("Couche à faire varier", labels, key=f"sens_layer_choice_{sid}")
    sel_lid = ids[labels.index(sel_label)]
    param = st.radio("Paramètre à faire varier", ["E (module)", "h (épaisseur)"],
                     horizontal=True, key=f"sens_param_choice_{sid}")
    base = [{"lid": lid, **_get_layer_values(sid, lid)} for lid in ids]
    sel_lv = _get_layer_values(sid, sel_lid)
    fallback_note = None
    if param.startswith("E"):
        base_val = sel_lv["E"] if sel_lv["E"] > 0 else 20.0
        if sel_lv["h"] <= 0:
            fallback_note = "h de cette couche non renseigné — hypothèse 1,00 m pour la simulation."
        c1, c2 = st.columns(2)
        with c1:
            lo = st.number_input("E min [MPa]", min_value=0.1, value=round(max(0.1, base_val * 0.3), 1),
                                 step=1.0, key=f"sens_E_lo_{sid}")
        with c2:
            hi = st.number_input("E max [MPa]", min_value=lo + 0.1, value=round(base_val * 3.0, 1),
                                 step=1.0, key=f"sens_E_hi_{sid}")
        x_label = "E [MPa]"
    else:
        E_fixed = sel_lv["E"] if sel_lv["E"] > 0 else (soil_default_E(sel_lv["type"]) or 20.0)
        if sel_lv["E"] <= 0:
            fallback_note = f"E non renseigné — hypothèse {E_fixed:.1f} MPa pour la simulation."
        base_val = sel_lv["h"] if sel_lv["h"] > 0 else 1.0
        c1, c2 = st.columns(2)
        with c1:
            lo = st.number_input("h min [m]", min_value=0.05, value=round(max(0.05, base_val * 0.3), 2),
                                 step=0.1, key=f"sens_h_lo_{sid}")
        with c2:
            hi = st.number_input("h max [m]", min_value=lo + 0.05, value=round(base_val * 3.0, 2),
                                 step=0.1, key=f"sens_h_hi_{sid}")
        x_label = "h [m]"
    if fallback_note:
        st.caption(f"ℹ️ {fallback_note}")
    n_pts = 30
    xs = [lo + (hi - lo) * i / (n_pts - 1) for i in range(n_pts)]
    ys = []
    for x in xs:
        layers_kpa = []
        for bl in base:
            h, E = bl["h"], bl["E"]
            if bl["lid"] == sel_lid:
                if param.startswith("E"):
                    E = x
                    h = h if h > 0 else 1.0
                else:
                    h = x
                    E = E if E > 0 else (soil_default_E(bl["type"]) or 20.0)
            if h > 0 and E > 0:
                layers_kpa.append((h, E_to_kPa(E, "MPa")))
        _, kk, _, _ = k_series(layers_kpa)
        ys.append(kk)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(xs, ys, color="#2563eb", linewidth=2)
    ax.axvline(base_val, color="#94a3b8", linestyle="--", linewidth=1)
    ax.scatter([base_val], [k_actual_MN], color="#dc2626", zorder=5, label="Valeurs actuelles")
    ax.set_xlabel(x_label)
    ax.set_ylabel("k_serie [MN/m³]")
    ax.set_title(f"Sensibilité — {_sounding_name(sid)} / {sel_label}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Les autres couches restent fixées à leurs valeurs actuelles. Utile pour tester "
               "les bornes basse/haute d'un rapport géotechnique plutôt qu'un coefficient de "
               "sécurité arbitraire (EN 1997 ne prescrit pas de facteur partiel sur E/k).")


def _render_k_vs_B_chart(E_moy_kPa: float, nu: float):
    if not _HAS_MPL:
        st.info("Installe matplotlib (`pip install matplotlib`) pour afficher ce graphique.")
        return
    if E_moy_kPa <= 0:
        st.info("Renseigne au moins une couche valide (h et E) pour tracer cette courbe.")
        return
    B_cur = float(st.session_state.get("multi_B", 2.0))
    c1, c2 = st.columns(2)
    with c1:
        B_lo = st.number_input("B min [m]", min_value=0.1, value=round(max(0.1, B_cur * 0.3), 2),
                               step=0.1, key="kb_B_lo")
    with c2:
        B_hi = st.number_input("B max [m]", min_value=B_lo + 0.1, value=round(B_cur * 3.0, 2),
                               step=0.1, key="kb_B_hi")
    n_pts = 30
    xs = [B_lo + (B_hi - B_lo) * i / (n_pts - 1) for i in range(n_pts)]
    ys = [k_boussinesq(E_moy_kPa, B, nu)[1] for B in xs]
    _, k_cur_MN = k_boussinesq(E_moy_kPa, B_cur, nu)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(xs, ys, color="#7c3aed", linewidth=2)
    ax.axvline(B_cur, color="#94a3b8", linestyle="--", linewidth=1)
    ax.scatter([B_cur], [k_cur_MN], color="#dc2626", zorder=5, label="B actuel")
    ax.set_xlabel("B [m]")
    ax.set_ylabel("k (Boussinesq) [MN/m³]")
    ax.set_title("k ≈ E_moy / [B·(1−ν²)] en fonction de B", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Modèle de comparaison uniquement (décroissant en 1/B) — sans effet sur les "
               "résultats \"à encoder dans SCIA\" (ressorts en série).")


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
        "multi_B": 2.0,
        "multi_nu": 0.30,
        "use_influence": True,
        "projet_nom": "",
    }
    for k, v in d.items():
        st.session_state.setdefault(k, v)

    if "soundings" not in st.session_state:
        st.session_state.soundings = []
        sid = _add_sounding(nom="CPT01", first_layer=False)
        st.session_state[_order_key(sid)] = [1]
        _init_layer(sid, 1, h=2.0, soil_type="Sable moyennement compact",
                    qc=8.5, rf=0.5, E=27.5)


# =============================================================
#  UI : TABLEAU DES COUCHES D'UN SONDAGE (widgets natifs)
# =============================================================
def _render_layer_row(sid: int, lid: int, i: int, disabled: bool = False):
    h_key = _layer_key(sid, lid, "h")
    type_key = _layer_key(sid, lid, "type")
    prev_key = _layer_key(sid, lid, "type_prev")
    qc_key = _layer_key(sid, lid, "qc")
    rf_key = _layer_key(sid, lid, "rf")
    E_key = _layer_key(sid, lid, "E")

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

    # Préremplissage : mutation AVANT le rendu des widgets qc/Rf/E.
    new_type = st.session_state.get(type_key, "—")
    if new_type != st.session_state.get(prev_key):
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
            st.button("＋", key=f"btn_add_layer_{sid}_{lid}", use_container_width=True,
                      help="Ajouter une couche", disabled=disabled,
                      on_click=_add_layer, args=(sid,))
        else:
            st.button("🗑️", key=f"btn_del_layer_{sid}_{lid}", use_container_width=True,
                      help="Supprimer cette couche", disabled=disabled,
                      on_click=_delete_layer, args=(sid, lid))

    lv2 = _get_layer_values(sid, lid)
    bits = []
    if not (lv2["h"] > 0 and lv2["E"] > 0):
        bits.append("⚠️ ligne ignorée dans le calcul (h ou E manquant)")
    else:
        bits.append("✅ prise en compte")
    new_type = st.session_state.get(type_key, "—")
    if new_type not in ("—", "Personnalisé"):
        e_sugg = suggest_E_from_qc(lv2["qc"], new_type)
        if e_sugg is not None:
            bits.append(f"E suggéré (qc→E) : {e_sugg:.1f} MPa")
        elif is_rock(new_type):
            bits.append("qc non pertinent pour ce type (refus de pointe probable) — seul E est utilisé")
    st.caption(" · ".join(bits))


def _render_sounding_block(sid: int):
    """Bloc d'un sondage : nom + boutons + niveau d'assise + tableau des couches."""
    with st.container(border=True):
        cN, cA, cC, cD = st.columns([3.6, 0.7, 0.7, 0.7], vertical_alignment="bottom")
        with cN:
            st.text_input("Sondage", key=f"snd{sid}_nom")
        with cA:
            st.button("➕", key=f"btn_add_snd_{sid}", help="Ajouter un sondage",
                      use_container_width=True, on_click=_add_sounding)
        with cC:
            st.button("📋", key=f"btn_copy_snd_{sid}", help="Copier le sondage",
                      use_container_width=True, on_click=_copy_sounding, args=(sid,))
        with cD:
            if len(st.session_state.soundings) > 1:
                st.button("🗑️", key=f"btn_del_snd_{sid}", help="Supprimer le sondage",
                          use_container_width=True, on_click=_delete_sounding, args=(sid,))

        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Niveau d'assise sous TN [m]", min_value=0.0, step=0.5,
                            key=f"snd{sid}_assise",
                            help="Profondeur du dessous de fondation sous le terrain naturel. "
                                 "Les couches ci-dessous sont à saisir À PARTIR de ce niveau "
                                 "(les terrains au-dessus de l'assise ne participent pas au k).")
        with c2:
            nappe = st.session_state.get(f"snd{sid}_nappe")
            if nappe is not None:
                st.markdown(f"<div style='padding-top:28px;color:#475569;'>Nappe ≈ "
                            f"{float(nappe):.1f} m sous TN (import)</div>", unsafe_allow_html=True)

        st.markdown("**Couches (sous le niveau d'assise)**")
        for i, lid in enumerate(_layer_ids(sid)):
            _render_layer_row(sid, lid, i)


# =============================================================
#  UI : IMPORT PDF PAR IA
# =============================================================
def _render_pdf_import():
    with st.expander("📥 Importer un rapport d'essais (PDF) — extraction par IA", expanded=False):
        st.caption("Téléverse un rapport de sondages (CPT, forages). L'IA lit le document, "
                   "découpe le profil en couches et crée un sondage par essai. "
                   "**Tout reste modifiable : les valeurs proposées doivent être vérifiées "
                   "par l'ingénieur avant usage.**")
        api_key = _get_api_key()
        if not api_key:
            st.text_input("Clé API Anthropic (ou à placer dans st.secrets['ANTHROPIC_API_KEY'])",
                          type="password", key="ai_api_key")
            api_key = _get_api_key()
        up = st.file_uploader("Rapport PDF", type=["pdf"], key="ai_pdf_upload")
        if up is not None and st.button("🤖 Analyser et importer", key="btn_ai_import",
                                        disabled=not api_key, use_container_width=True,
                                        help=None if api_key else "Clé API requise"):
            try:
                with st.spinner("Analyse du rapport par l'IA…"):
                    sondages = _extract_soundings_from_pdf(up.getvalue(), api_key)
                created = _import_ai_soundings(sondages)
                st.success(f"Importé : {', '.join(created)}. Vérifie les couches, "
                           "les types mappés et les E proposés avant tout calcul.")
                st.rerun()
            except Exception as e:
                st.error(f"Import impossible : {e}")


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
            keep = {"press_unit", "module_unit", "detail_calc", "adv_open", "page",
                    "abaque_w", "ai_api_key"}
            for k in list(st.session_state.keys()):
                if k not in keep:
                    st.session_state.pop(k, None)
            st.rerun()
    with cols[2]:
        st.button("💾 Enregistrer", use_container_width=True, disabled=True,
                  help="À connecter au système JSON")
    with cols[3]:
        st.button("📂 Ouvrir", use_container_width=True, disabled=True,
                  help="Lecture de fichiers à venir")
    with cols[4]:
        st.button("📝 Générer PDF", use_container_width=True, disabled=True,
                  help="Rapport sols : bouton dans le panneau SCIA du Cas 2")
    with cols[5]:
        st.markdown(f"<div style='text-align:right;padding-top:10px;'>"
                    f"<span class='memo-chip'>{VERSION}</span></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("# Raideur élastique des sols")
    st.markdown("<span class='small'>Pré-dimensionnement — sols modélisés par des ressorts "
                "verticaux (modèle de Winkler).</span>", unsafe_allow_html=True)

    with st.expander("📘 Fiche mémo (k, unités et modèle de Winkler)", expanded=False):
        st.markdown(
            r"""
- **Winkler** : $q = k \cdot w \Rightarrow k = q/w$.
- **Unités** : $q$ en kPa = kN/m² · $w$ en m · $k$ en kN/m³ ou MN/m³ (1 MN/m³ = 1000 kN/m³).
- **Ressorts en série** (colonne de sol) : $1/k_{serie} = \sum_i h_i/E_i$ — modèle à privilégier pour
  exporter $k$ vers un logiciel de dalle sur sol élastique (SCIA...) quand le profil est connu.
- **Profondeur d'influence** : limiter le profil à $\approx 2 \cdot B$ sous l'assise — sommer tout un
  sondage de 30 m donne un k artificiellement faible.
- **Semelle sur massif semi-infini** (Boussinesq, ordre de grandeur) : $k \approx E/[B(1-\nu^2)]$.
  Ces deux modèles répondent à des questions différentes et **ne se chaînent pas**.
- **Rocher (schiste, calcaire, craie saine, grès)** : le CPT est en général en **refus de pointe** —
  utiliser une plage de $E$ issue du degré d'altération (RQD/GSI/pressiomètre), jamais qc→E.
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
             "2. Modélisation de sondages (multicouche / import PDF)",
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
            st.markdown("<span class='memo-chip'>Typiquement : q à l'ELS, w = 20 mm → k pour "
                        "SCIA / RDM.</span>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.solo_q = st.number_input(
                    f"q (pression au sol) [{pu}]", min_value=0.0,
                    value=float(st.session_state.get("solo_q", 60.0)), step=5.0)
            with c2:
                st.session_state.solo_w = st.number_input(
                    "w (tassement) [mm]", min_value=0.001,
                    value=float(st.session_state.get("solo_w", 20.0)), step=5.0)

        elif cas.startswith("2."):
            st.markdown("**Modélisation de sondages (multicouche / import PDF)**")
            st.caption("Un sondage = un profil de couches saisi À PARTIR du niveau d'assise. "
                       "Calcul par sondage en ressorts en série : 1/k = Σ(hᵢ/Eᵢ).")
            st.text_input("Nom du projet (pour le rapport PDF)", key="projet_nom")

            _render_pdf_import()

            c1, c2, c3 = st.columns([1.2, 1.0, 1.6], vertical_alignment="bottom")
            with c1:
                st.number_input("Largeur caractéristique B [m]", min_value=0.1, step=0.1,
                                key="multi_B",
                                help="Largeur de la zone chargée (semelle, bande de dalle). "
                                     "Sert à la profondeur d'influence (2·B) et au modèle "
                                     "de comparaison Boussinesq.")
            with c2:
                st.number_input("ν (Poisson)", min_value=0.0, max_value=0.49, step=0.01,
                                key="multi_nu", help="Uniquement pour la comparaison Boussinesq.")
            with c3:
                st.checkbox("Limiter le profil à la profondeur d'influence (2·B)",
                            key="use_influence",
                            help="Écrête automatiquement le profil à H = 2·B sous l'assise. "
                                 "Fortement recommandé : sommer tout un sondage profond donne "
                                 "un k artificiellement faible.")

            st.markdown("#### Sondages")
            for s in st.session_state.soundings:
                _render_sounding_block(int(s["id"]))

        elif cas.startswith("3."):
            st.markdown("**Raideur d'un sol – formule empirique (CPT)**")
            st.caption("Basée sur une valeur de qc : E = α_E (qₜ − σ'ᵥ₀), puis k ≈ E/[B(1−ν²)]. "
                       "Sol supposé homogène.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.cpt_qt = st.number_input(
                    "qₜ (pointe nette) [MPa]", min_value=0.0,
                    value=float(st.session_state.get("cpt_qt", 5.0)), step=0.5)
            with c2:
                st.session_state.cpt_sv0 = st.number_input(
                    "σ'ᵥ₀ (contrainte eff.) [kPa]", min_value=0.0,
                    value=float(st.session_state.get("cpt_sv0", 100.0)), step=10.0)
            with c3:
                st.session_state.cpt_alphaE = st.number_input(
                    "α_E (CPT → E)", min_value=0.1,
                    value=float(st.session_state.get("cpt_alphaE", 2.5)), step=0.1)
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.cpt_B = st.number_input(
                    "B (largeur) [m]", min_value=0.1,
                    value=float(st.session_state.get("cpt_B", 2.0)), step=0.1)
            with c5:
                st.session_state.cpt_nu = st.number_input(
                    "ν (Poisson)", min_value=0.0, max_value=0.49,
                    value=float(st.session_state.get("cpt_nu", 0.30)), step=0.01)
            st.caption("⚠️ Ne s'applique qu'aux sols meubles (le CPT est en refus sur du rocher).")

        elif cas.startswith("4."):
            st.markdown("**Raideur d'un plat en béton (contact plat / béton / grout)**")
            st.caption("Contact assimilé à une compression 1D du béton (et du grout). "
                       "Par défaut k = E/h.")
            st.markdown("**Géométrie du plat**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.plate_B = st.number_input(
                    "Largeur plat B [mm]", min_value=20.0,
                    value=float(st.session_state.get("plate_B", 200.0)), step=10.0)
            with c2:
                st.session_state.plate_L = st.number_input(
                    "Longueur plat L [mm]", min_value=20.0,
                    value=float(st.session_state.get("plate_L", 200.0)), step=10.0)
            with c3:
                st.session_state.plate_alpha = st.number_input(
                    "α (h_c = α·min(B,L))", min_value=0.05,
                    value=float(st.session_state.get("plate_alpha", 0.5)), step=0.05)
            st.markdown("**Béton support**")
            c4, c5 = st.columns(2)
            with c4:
                st.session_state.plate_Ec = st.number_input(
                    "E_c béton [GPa]", min_value=5.0,
                    value=float(st.session_state.get("plate_Ec", 30.0)), step=1.0)
            with c5:
                st.session_state.plate_use_nu = st.checkbox(
                    "Appliquer le facteur (1−ν²)",
                    value=st.session_state.get("plate_use_nu", False),
                    help="Valable pour un massif semi-infini, rarement justifié pour une "
                         "couche mince confinée.")
            if st.session_state.plate_use_nu:
                st.session_state.plate_nu = st.number_input(
                    "ν béton", min_value=0.0, max_value=0.49,
                    value=float(st.session_state.get("plate_nu", 0.20)), step=0.01)
            else:
                st.session_state.plate_nu = st.session_state.get("plate_nu", 0.20)
            st.markdown("**Lit de mortier / grout (optionnel)**")
            st.session_state.plate_has_grout = st.checkbox(
                "Présence d'un lit de mortier/grout",
                value=st.session_state.get("plate_has_grout", False))
            if st.session_state.plate_has_grout:
                c6, c7 = st.columns(2)
                with c6:
                    st.session_state.plate_tg = st.number_input(
                        "Épaisseur grout t_g [mm]", min_value=1.0,
                        value=float(st.session_state.get("plate_tg", 20.0)), step=1.0)
                with c7:
                    st.session_state.plate_Eg = st.number_input(
                        "E_g grout [GPa]", min_value=5.0,
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
                index=["k → q (pour un tassement w)", "q, w → k",
                       "E, B, ν → k (Boussinesq)"].index(
                    st.session_state.get("conv_mode", "q, w → k")),
            )
            m = st.session_state.conv_mode
            if m.startswith("k →"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.conv_k = st.number_input(
                        "k [MN/m³]", min_value=0.0,
                        value=float(st.session_state.get("conv_k", 30.0)), step=1.0)
                with c2:
                    st.session_state.conv_w = st.number_input(
                        "w (tassement) [mm]", min_value=0.001,
                        value=float(st.session_state.get("conv_w", 20.0)), step=1.0)
            elif m.startswith("q,"):
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state.conv_q = st.number_input(
                        f"q [{pu}]", min_value=0.0,
                        value=float(st.session_state.get("conv_q", 60.0)), step=5.0)
                with c2:
                    st.session_state.conv_w = st.number_input(
                        "w [mm]", min_value=0.001,
                        value=float(st.session_state.get("conv_w", 20.0)), step=1.0)
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.session_state.conv_E = st.number_input(
                        f"E [{st.session_state.module_unit}]", min_value=0.0,
                        value=float(st.session_state.get("conv_E", 30.0)), step=5.0)
                with c2:
                    st.session_state.conv_B = st.number_input(
                        "B [m]", min_value=0.1,
                        value=float(st.session_state.get("conv_B", 2.0)), step=0.1)
                with c3:
                    st.session_state.conv_nu = st.number_input(
                        "ν", min_value=0.0, max_value=0.49,
                        value=float(st.session_state.get("conv_nu", 0.30)), step=0.01)

        else:  # cas 6
            st.markdown("**Abaque sols – valeurs indicatives**")
            st.caption("Poids volumique γ, raideur k (MN/m³) et contrainte admissible qₐ "
                       "(kg/cm²) pour un tassement de référence — basé sur la même base de "
                       "données que le tableau multicouche. À confirmer par le géotechnicien.")

    # =========================================================
    #  COLONNE DROITE — résultats
    # =========================================================
    with col_right:
        st.markdown("### Dimensionnement / Résultats")
        st.session_state.detail_calc = st.checkbox(
            "📘 Détail des calculs (formules + valeurs)", value=st.session_state.detail_calc)
        detail = st.session_state.detail_calc
        pu = st.session_state.press_unit
        mu = st.session_state.module_unit

        # ---------- CAS 1 ----------
        if cas.startswith("1."):
            with st.container(border=True):
                q_kPa = to_kPa(st.session_state.get("solo_q", 0.0), pu)
                w_mm = st.session_state.get("solo_w", 20.0)
                k_kN, k_MN, w_m = k_from_qw(q_kPa, w_mm)
                _bloc("Raideur de Winkler", f"k = {k_MN:,.2f} MN/m³".replace(",", " "),
                      "ok" if k_MN > 0 else "nok")
                if detail and k_MN > 0:
                    st.latex(r"k = \dfrac{q}{w}")
                    st.latex(f"k = \\dfrac{{{q_kPa:,.1f}}}{{{w_m:,.3f}}} "
                             f"= {k_kN:,.0f}\\,\\text{{kN/m³}} = {k_MN:,.2f}\\,\\text{{MN/m³}}")
                    _param_table([
                        ("q", "Pression de service",
                         f"{st.session_state.get('solo_q', 0.0):,.2f}", pu),
                        ("w", "Tassement", f"{w_mm:,.2f}", "mm"),
                        ("k", "Raideur de sol", f"{k_MN:,.2f}", "MN/m³"),
                    ])

        # ---------- CAS 2 : MULTI-SONDAGES ----------
        elif cas.startswith("2."):
            B = float(st.session_state.get("multi_B", 2.0))
            nu = float(st.session_state.get("multi_nu", 0.30))
            use_infl = bool(st.session_state.get("use_influence", True))
            H_lim = 2.0 * B if use_infl else None

            results = []   # [{"sid","nom","assise","nappe","res"}]
            for s in st.session_state.soundings:
                sid = int(s["id"])
                res = _compute_sounding_k(sid, H_lim=H_lim)
                results.append({
                    "sid": sid, "nom": _sounding_name(sid),
                    "assise": float(st.session_state.get(f"snd{sid}_assise", 0.0) or 0.0),
                    "nappe": st.session_state.get(f"snd{sid}_nappe"),
                    "res": res,
                })

            # ---- Panneau SCIA (récapitulatif en tête) ----
            with st.container(border=True):
                st.markdown("#### 🎯 Raideurs à encoder dans SCIA (subsoil C)")
                rec_rows = []
                k_vals = []
                for r in results:
                    res = r["res"]
                    if res["k_MN"] > 0:
                        k_vals.append(res["k_MN"])
                        rec_rows.append({"Sondage": r["nom"],
                                         "k [MN/m³]": round(res["k_MN"], 2),
                                         "k [kN/m³]": round(res["k_kN"])})
                    else:
                        rec_rows.append({"Sondage": r["nom"],
                                         "k [MN/m³]": None, "k [kN/m³]": None})
                st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
                if len(k_vals) >= 2:
                    st.markdown(f"**Enveloppe : k = {min(k_vals):.2f} à {max(k_vals):.2f} MN/m³** "
                                "— calculer la dalle avec les deux bornes (sensibilité), "
                                "pas avec une valeur unique.")
                elif len(k_vals) == 1:
                    st.caption("Un seul sondage exploitable : pense à encadrer k avec les "
                               "bornes basse/haute des E (onglet Sensibilité).")
                if use_infl:
                    st.caption(f"Profils limités à la profondeur d'influence H = 2·B = "
                               f"{H_lim:.2f} m sous l'assise.")

                # ---- Rapport PDF ----
                if _HAS_REPORTLAB:
                    if st.button("📄 Générer le rapport PDF sols", key="btn_soil_pdf",
                                 use_container_width=True):
                        try:
                            pdf_bytes = build_soil_report_pdf(
                                project={"nom": st.session_state.get("projet_nom", ""),
                                         "date": date.today().strftime("%d/%m/%Y")},
                                soundings_data=results, B=B,
                                use_influence=use_infl, H_lim=H_lim,
                                w_adm_mm=float(st.session_state.get("abaque_w", 20.0)),
                            )
                            st.session_state["soil_pdf_bytes"] = pdf_bytes
                        except Exception as e:
                            st.session_state.pop("soil_pdf_bytes", None)
                            st.error(f"Erreur de génération du rapport : {e}")
                    if st.session_state.get("soil_pdf_bytes"):
                        nomp = re.sub(r"[^A-Za-z0-9]+", "_",
                                      st.session_state.get("projet_nom", "") or "Projet")[:20]
                        st.download_button(
                            "⬇️ Télécharger le rapport",
                            data=st.session_state["soil_pdf_bytes"],
                            file_name=f"{nomp}_Raideur_sol_{date.today().strftime('%d-%m-%Y')}.pdf",
                            mime="application/pdf", use_container_width=True,
                            key="btn_soil_pdf_dl")
                else:
                    st.caption("Rapport PDF indisponible : installe reportlab "
                               "(`pip install reportlab`).")

            # ---- Détail par sondage ----
            for r in results:
                res = r["res"]
                with st.expander(f"{'🟢' if res['k_MN'] > 0 else '🔴'} {r['nom']} — "
                                 f"k = {res['k_MN']:.2f} MN/m³" if res["k_MN"] > 0
                                 else f"🔴 {r['nom']} — profil incomplet", expanded=True):
                    _bloc("Ressorts en série (colonne 1D)",
                          f"k = {res['k_MN']:,.2f} MN/m³".replace(",", " "),
                          "ok" if res["k_MN"] > 0 else "nok")
                    if res["ignored"]:
                        st.warning(f"⚠️ Couche(s) {', '.join(map(str, res['ignored']))} "
                                   "ignorée(s) : E manquant. Complète ou supprime ces lignes.")
                    if use_infl and res["H_saisi"] > (H_lim or 0) + 1e-6:
                        st.info(f"Profil saisi : {res['H_saisi']:.2f} m — écrêté à "
                                f"{res['H']:.2f} m (profondeur d'influence 2·B).")
                    elif not use_infl and B > 0 and res["H_saisi"] > 2.5 * B:
                        st.warning(f"⚠️ H saisi = {res['H_saisi']:.2f} m >> 2·B = {2*B:.2f} m : "
                                   "k probablement sous-estimé. Active la limitation à la "
                                   "profondeur d'influence.")
                    if detail and res["k_MN"] > 0:
                        st.latex(r"k_{serie} = \left(\sum_i \dfrac{h_i}{E_i}\right)^{-1}")
                        st.latex(f"k_{{serie}} = {res['k_kN']:,.0f}\\,\\text{{kN/m³}} "
                                 f"= {res['k_MN']:,.2f}\\,\\text{{MN/m³}}")
                        _param_table([
                            ("H", "Épaisseur prise en compte", f"{res['H']:,.2f}", "m"),
                            ("E_moy", "Module oedo. équivalent",
                             f"{res['E_moy_kPa']/1000:,.1f}", "MPa"),
                            ("k_serie", "Raideur (série)", f"{res['k_MN']:,.2f}", "MN/m³"),
                        ])
                        kB_kN, kB_MN = k_boussinesq(res["E_moy_kPa"], B, nu)
                        if kB_MN > 0:
                            st.caption(f"Comparaison Boussinesq (massif semi-infini, "
                                       f"B = {B:.2f} m, ν = {nu:.2f}) : "
                                       f"k ≈ {kB_MN:.2f} MN/m³ — ordre de grandeur uniquement, "
                                       "ne pas exporter vers SCIA.")

            # ---- Graphiques ----
            st.divider()
            st.markdown("#### Graphiques")
            noms = [r["nom"] for r in results]
            sel = st.selectbox("Sondage", noms, key="chart_snd_choice")
            r_sel = results[noms.index(sel)]
            tab_profil, tab_sens, tab_kb = st.tabs(
                ["Profil du sol", "Sensibilité d'une couche", "k vs B (Boussinesq)"])
            with tab_profil:
                _render_soil_profile_chart(r_sel["sid"])
            with tab_sens:
                _render_layer_sensitivity_chart(r_sel["sid"], r_sel["res"]["k_MN"])
            with tab_kb:
                _render_k_vs_B_chart(r_sel["res"]["E_moy_kPa"], nu)

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
                    st.latex(f"k \\approx {k_kN:,.0f}\\,\\text{{kN/m³}} "
                             f"= {k_MN:,.2f}\\,\\text{{MN/m³}}")

        # ---------- CAS 4 ----------
        elif cas.startswith("4."):
            with st.container(border=True):
                res = k_plate(
                    st.session_state.get("plate_B", 200.0),
                    st.session_state.get("plate_L", 200.0),
                    st.session_state.get("plate_alpha", 0.5),
                    st.session_state.get("plate_Ec", 30.0),
                    st.session_state.get("plate_use_nu", False),
                    st.session_state.get("plate_nu", 0.20),
                    st.session_state.get("plate_has_grout", False),
                    st.session_state.get("plate_tg", 0.0),
                    st.session_state.get("plate_Eg", 20.0),
                )
                _bloc("Raideur du contact plat/béton",
                      f"k_eq = {res['keq_MNpm3']:,.1f} MN/m³".replace(",", " "),
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
                        ("E_c", "Module béton",
                         f"{st.session_state.get('plate_Ec',30.0):,.1f}", "GPa"),
                        ("k_eq", "Raideur équivalente", f"{res['keq_MNpm3']:,.1f}", "MN/m³"),
                    ])

        # ---------- CAS 5 : convertisseur ----------
        elif cas.startswith("5."):
            with st.container(border=True):
                m = st.session_state.get("conv_mode", "q, w → k")
                if m.startswith("k →"):
                    k_MN = st.session_state.get("conv_k", 0.0)
                    w_mm = st.session_state.get("conv_w", 20.0)
                    q_kPa = k_MN * 1000.0 * (w_mm / 1000.0)
                    _bloc("Pression mobilisée",
                          f"q = {from_kPa(q_kPa, pu):,.2f} {pu}".replace(",", " "),
                          "ok" if q_kPa > 0 else "nok")
                    if detail:
                        st.latex(r"q = k \cdot w")
                        st.latex(f"q = {k_MN:,.2f}\\cdot10^3 \\cdot {w_mm/1000:,.3f} "
                                 f"= {q_kPa:,.1f}\\,\\text{{kN/m²}} "
                                 f"= {from_kPa(q_kPa,pu):,.2f}\\,\\text{{{pu}}}")
                elif m.startswith("q,"):
                    q_kPa = to_kPa(st.session_state.get("conv_q", 0.0), pu)
                    w_mm = st.session_state.get("conv_w", 20.0)
                    k_kN, k_MN, w_m = k_from_qw(q_kPa, w_mm)
                    _bloc("Raideur", f"k = {k_MN:,.2f} MN/m³".replace(",", " "),
                          "ok" if k_MN > 0 else "nok")
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
                    help="Convertit k (MN/m³) en qₐ (kg/cm²). En Belgique, 20 mm est courant "
                         "en service.")
                w_adm = st.session_state.abaque_w
                factor_q = w_adm / KGF_PER_CM2_TO_KPA

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
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("#### Fiche sol")
                noms = [r["Type de sol"] for r in rows]
                default_idx = noms.index("Sable moyennement compact") \
                    if "Sable moyennement compact" in noms else 0
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
                    lignes.append(f"- qc ≈ **{d['qc_min']} à {d['qc_max']} MPa** "
                                  f"(α qc→E ≈ {d['alpha_qc']})")
                else:
                    lignes.append("- qc : **non pertinent** (refus de pointe probable) — "
                                  "caractériser par RQD/pressiomètre.")
                lignes.append(f"- pour w_adm = **{w_adm:.0f} mm** → qₐ ≈ "
                              f"**{q_min:.2f} à {q_max:.2f} kg/cm²**")
                st.markdown("  \n".join(lignes))

        st.divider()
        st.markdown("<div class='small'>Valeurs de k, E et qₐ indicatives (littérature "
                    "géotechnique / retours d'expérience), réservées au pré-dimensionnement. "
                    "Les valeurs importées par IA depuis un rapport PDF sont des propositions "
                    "à vérifier systématiquement par l'ingénieur. Se référer au rapport "
                    "géotechnique et à l'EN 1997 (Eurocode 7) pour le dimensionnement final."
                    "</div>", unsafe_allow_html=True)

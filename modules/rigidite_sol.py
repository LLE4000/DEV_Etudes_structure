# -*- coding: utf-8 -*-
# =============================================================
#  rigidite_sol.py — Raideur élastique des sols (Winkler)
#  VERSION 5.0
#
#  RÉÉCRITURE COMPLÈTE. Trois changements de fond vs la v4.0 :
#
#  1. LE CALCUL. La v4.0 posait 1/k = Σ hᵢ/Eᵢ, ce qui suppose la
#     contrainte UNIFORME sur toute la hauteur retenue et sous-estimait k
#     d'un facteur 2 à 2,5 (le résultat était en fait piloté par la règle
#     des 2·B, pas par le sol : pour un sol homogène la formule se réduit
#     exactement à k = E/2B). On calcule désormais le tassement comme il
#     se calcule :  w = Σ Δσᵢ·hᵢ/Mᵢ  avec Δσ issu de Boussinesq/Newmark,
#     puis k = q/w. La profondeur d'influence n'est plus décrétée à 2·B :
#     elle SORT du calcul (critère Δσ ≤ 0,20·σ'v0).
#
#  2. L'IMPORT. Plus d'intelligence artificielle, donc plus de clé API à
#     saisir dans l'application. Les données sont lues directement dans le
#     fichier : GEF, CSV, ou courbes vectorielles du PDF recalées sur les
#     traits de grille. Précision mesurée sur un rapport de contrôle :
#     écart médian 0,0005 % (contre 6,5 % en recalant sur les étiquettes
#     de texte, et sans commune mesure avec une lecture à l'œil).
#     Tout est exportable en CSV.
#
#  3. LE RÉSULTAT. k n'est pas une propriété du sol : il dépend de la
#     fondation. On rend donc un k ENCADRÉ (les corrélations CPT
#     divergent d'un facteur 2 à 4) et ZONÉ — centre, bord, angle — ce
#     qui reproduit à la main ce que fait un calcul itératif type Soilin
#     et permet d'encoder plusieurs zones de sol sous une dalle SCIA.
#
#  Corrections de défauts de la v4.0 :
#    F1  « craie altérée » était classée « Craie saine » (accents absents
#        des mots-clés) : module ×23 en silence.        -> sol_base.py
#    F2  le champ « Niveau d'assise » n'entrait dans aucun calcul.
#    F3  « Réinitialiser » vidait tout st.session_state, donc aussi les
#        poutres et les dalles des autres modules.
#    F4  un libellé de roche générique basculait sur la variante SAINE.
#    F7  les k de l'abaque sont des valeurs de PLAQUE 0,30 m : la
#        correction de taille de Terzaghi est désormais appliquée.
#
#  Validation externe (tests/test_sol_theorie.py) :
#    · facteur d'influence de Newmark : 7 valeurs des tables publiées,
#      écart < 5·10⁻⁴ ; ∫I dz = 1,1206·B contre Is = 1,12 publié.
#    · Robertson Ic et module oedométrique : concordance avec groundhog
#      (bibliothèque open source, Université de Gand) à 0,005 sur Ic et
#      au dixième de MPa sur M.
# =============================================================

import io
import json
import math
import re
from datetime import date

import pandas as pd
import streamlit as st

from modules import sol_base as SB
from modules import sol_import as SI
from modules import sol_theorie as ST

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    _HAS_MPL = True
except ImportError:                                   # pragma: no cover
    _HAS_MPL = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors as _rl
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    _HAS_REPORTLAB = True
except ImportError:                                   # pragma: no cover
    _HAS_REPORTLAB = False


VERSION = "v5.0"
KGF_PER_CM2_TO_KPA = 98.0665

C_COULEURS = {"ok": "#e6ffe6", "warn": "#fffbe6", "nok": "#ffe6e6", "info": "#eef2ff"}
C_ICONES = {"ok": "✅", "warn": "⚠️", "nok": "❌", "info": "ℹ️"}

# TOUTES les clés d'une couche, y compris celles issues d'un import CPT.
# Elles doivent y figurer sans exception : c'est cette liste qui est
# utilisée pour SUPPRIMER une couche et pour COPIER un sondage. Un champ
# oublié survit à la suppression et se retrouve hérité par la couche
# suivante portant le même identifiant (borne haute d'un ancien sol
# recollée à un nouveau), et disparaît d'une copie de sondage.
LAYER_FIELDS = ("h", "type", "type_prev", "qc", "rf", "M", "gamma",
                "M_haut", "Ic", "sbt")
LAYER_COLS = [0.7, 2.0, 0.9, 0.8, 0.9, 0.9, 0.5]

# Clés de session propres à ce module : tout ce qui commence par ces
# préfixes appartient au module sol et à personne d'autre. C'est ce qui
# permet un « Réinitialiser » qui ne détruit pas le travail des modules
# Poutre et Dalle (défaut F3 de la v4.0).
_PREFIXES = ("snd", "rs_", "sol_")


def _est_cle_module(k: str) -> bool:
    return (any(k.startswith(p) for p in _PREFIXES)
            or k in ("soundings", "rs_projet"))


# ---- écritures différées vers des clés de widgets --------------------
# Streamlit interdit de modifier st.session_state[k] quand le widget de
# clé k a déjà été instancié dans le run en cours. Un callback de bouton
# s'exécute au MILIEU du script : tout ce qui a été rendu avant lui est
# donc verrouillé. On empile ces écritures et on les applique au tout
# début du run suivant, avant qu'aucun widget n'existe.
def _differer(valeurs: dict):
    st.session_state.setdefault("rs_pending", {})
    st.session_state["rs_pending"].update(valeurs)


def _appliquer_differees():
    for k, v in (st.session_state.pop("rs_pending", None) or {}).items():
        st.session_state[k] = v


# =============================================================
#  HELPERS D'AFFICHAGE
# =============================================================
def _bloc(left, right="", etat="ok"):
    r = (f"<div style='font-weight:600;opacity:.9;white-space:nowrap;'>{right}</div>"
         if right else "")
    st.markdown(
        f'<div style="background:{C_COULEURS.get(etat, "#f6f6f6")};padding:12px 14px;'
        f'border-radius:10px;border:1px solid #d9d9d9;margin:8px 0 4px 0;display:flex;'
        f'justify-content:space-between;align-items:center;gap:10px;">'
        f'<div style="font-weight:700;">{left}</div>'
        f'<div style="display:flex;align-items:center;gap:10px;">{r}'
        f'<div style="font-size:20px;line-height:1;">{C_ICONES.get(etat, "")}</div></div></div>',
        unsafe_allow_html=True)


def _fr(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}".replace(",", " ").replace(".", ",")
    except Exception:
        return str(x)


# =============================================================
#  ÉTAT
# =============================================================
def _layer_key(sid, lid, champ):
    return f"snd{sid}_layer_{lid}_{champ}"


def _order_key(sid):
    return f"snd{sid}_layer_order"


def _layer_ids(sid):
    return list(st.session_state.get(_order_key(sid), []))


def _new_layer_id(sid):
    ids = _layer_ids(sid)
    return (max(ids) + 1) if ids else 1


def _init_layer(sid, lid, h=1.0, type_sol="—", qc=0.0, rf=0.0, M=0.0, gamma=19.0):
    st.session_state[_layer_key(sid, lid, "h")] = float(h)
    st.session_state[_layer_key(sid, lid, "type")] = type_sol
    st.session_state[_layer_key(sid, lid, "type_prev")] = type_sol
    st.session_state[_layer_key(sid, lid, "qc")] = float(qc)
    st.session_state[_layer_key(sid, lid, "rf")] = float(rf)
    st.session_state[_layer_key(sid, lid, "M")] = float(M)
    st.session_state[_layer_key(sid, lid, "gamma")] = float(gamma)


def _add_layer(sid):
    lid = _new_layer_id(sid)
    st.session_state[_order_key(sid)].append(lid)
    _init_layer(sid, lid)


def _delete_layer(sid, lid):
    ids = _layer_ids(sid)
    if len(ids) <= 1 or lid not in ids:
        return
    st.session_state[_order_key(sid)].remove(lid)
    for f in LAYER_FIELDS:
        st.session_state.pop(_layer_key(sid, lid, f), None)


def _get_layer(sid, lid):
    g = st.session_state.get(_layer_key(sid, lid, "gamma"), 19.0)
    return {
        "h": float(st.session_state.get(_layer_key(sid, lid, "h"), 0.0) or 0.0),
        "type": st.session_state.get(_layer_key(sid, lid, "type"), "—"),
        "qc": float(st.session_state.get(_layer_key(sid, lid, "qc"), 0.0) or 0.0),
        "rf": float(st.session_state.get(_layer_key(sid, lid, "rf"), 0.0) or 0.0),
        "M": float(st.session_state.get(_layer_key(sid, lid, "M"), 0.0) or 0.0),
        "gamma": float(g or 19.0),
    }


def _sounding_ids():
    return [int(s["id"]) for s in st.session_state.get("soundings", [])]


def _sounding_name(sid):
    return str(st.session_state.get(f"snd{sid}_nom", f"Sondage {sid}"))


def _new_sounding_id():
    ids = _sounding_ids()
    return (max(ids) + 1) if ids else 1


def _add_sounding(nom=None, first_layer=True):
    sid = _new_sounding_id()
    nom = nom or f"CPT{sid:02d}"
    st.session_state.soundings.append({"id": sid, "nom": nom})
    st.session_state[f"snd{sid}_nom"] = nom
    st.session_state[_order_key(sid)] = []
    if first_layer:
        # Profil de départ volontairement PROFOND : un profil court
        # tronquerait le tassement et donnerait un k trop élevé dès la
        # première ouverture, sans que l'utilisateur en soit averti.
        for i, (h, t, qc, rf, M, g) in enumerate((
                (2.0, "Limon (loess)", 2.0, 2.0, 22.0, 18.0),
                (4.0, "Sable moyennement compact", 8.5, 0.5, 37.0, 19.0),
                (14.0, "Argile ferme / raide", 3.5, 2.5, 37.0, 19.0)), start=1):
            st.session_state[_order_key(sid)].append(i)
            _init_layer(sid, i, h=h, type_sol=t, qc=qc, rf=rf, M=M, gamma=g)
    return sid


def _delete_sounding(sid):
    if len(st.session_state.soundings) <= 1:
        return
    st.session_state.soundings = [s for s in st.session_state.soundings
                                  if int(s["id"]) != sid]
    for k in [k for k in list(st.session_state.keys()) if k.startswith(f"snd{sid}_")]:
        st.session_state.pop(k, None)


def _copy_sounding(src):
    sid = _new_sounding_id()
    st.session_state.soundings.append({"id": sid, "nom": f"{_sounding_name(src)} (copie)"})
    st.session_state[f"snd{sid}_nom"] = f"{_sounding_name(src)} (copie)"
    st.session_state[_order_key(sid)] = list(_layer_ids(src))
    for lid in _layer_ids(src):
        for f in LAYER_FIELDS:
            st.session_state[_layer_key(sid, lid, f)] = \
                st.session_state.get(_layer_key(src, lid, f))
    for suf in ("nappe", "points", "source"):
        if f"snd{src}_{suf}" in st.session_state:
            st.session_state[f"snd{sid}_{suf}"] = st.session_state[f"snd{src}_{suf}"]


def _init_state():
    defauts = {
        "rs_B": 2.0, "rs_L": 2.0, "rs_q": 150.0, "rs_D": 1.0,
        "rs_nappe_active": False, "rs_nappe": 3.0,
        "rs_critere": 20, "rs_q_net": True, "rs_nu": 0.30,
        "rs_detail": True, "rs_projet": "", "rs_w_ref": 20.0,
        "rs_mode": "1. Sondage CPT (import de fichier)",
        "rs_pos_scia": "centre",
    }
    for k, v in defauts.items():
        st.session_state.setdefault(k, v)
    if "soundings" not in st.session_state:
        st.session_state.soundings = []
        _add_sounding(nom="CPT01")


def _reset_module():
    """Réinitialise UNIQUEMENT ce module (défaut F3 de la v4.0 : le bouton
    vidait tout st.session_state, donc aussi beams et dalles)."""
    for k in [k for k in list(st.session_state.keys()) if _est_cle_module(k)]:
        st.session_state.pop(k, None)
    st.rerun()


# =============================================================
#  PROFIL DE COUCHES <-> SESSION
# =============================================================
def _profil_session(sid):
    """Couches saisies, au format attendu par sol_theorie (depuis le TN)."""
    out = []
    for lid in _layer_ids(sid):
        lv = _get_layer(sid, lid)
        if lv["h"] <= 0:
            continue
        out.append({"h": lv["h"], "gamma": lv["gamma"], "M": lv["M"] or None,
                    "nom": lv["type"], "qc": lv["qc"]})
    return out


def _remplir_depuis_couches(sid, couches, nappe=None, source=""):
    """Remplit AUTOMATIQUEMENT le tableau des couches d'un sondage."""
    st.session_state[_order_key(sid)] = []
    for i, c in enumerate(couches, start=1):
        st.session_state[_order_key(sid)].append(i)
        libelle = c.get("sbt") or c.get("type") or "—"
        type_db = SB.match_soil_type(libelle)
        M = c.get("M_bas") or c.get("M") or 0.0
        rf = 0.0
        if c.get("fs") and c.get("qc"):
            rf = c["fs"] / (c["qc"] * 1000.0) * 100.0
        _init_layer(sid, i, h=round(c.get("h", 0.0), 2), type_sol=type_db,
                    qc=round(c.get("qc", 0.0) or 0.0, 2), rf=round(rf, 2),
                    M=round(M or 0.0, 1), gamma=round(c.get("gamma", 19.0), 1))
        st.session_state[_layer_key(sid, i, "M_haut")] = c.get("M_haut")
        st.session_state[_layer_key(sid, i, "Ic")] = c.get("Ic")
        st.session_state[_layer_key(sid, i, "sbt")] = c.get("sbt")
    if not _layer_ids(sid):
        st.session_state[_order_key(sid)].append(1)
        _init_layer(sid, 1)
    if nappe is not None:
        st.session_state[f"snd{sid}_nappe"] = float(nappe)
        # rs_nappe_active et rs_nappe sont des clés de WIDGETS déjà
        # instanciés plus haut dans le même run (panneau Fondation) :
        # les écrire ici lève une StreamlitAPIException. On diffère au
        # run suivant, où elles seront posées avant tout rendu.
        _differer({"rs_nappe_active": True, "rs_nappe": float(nappe)})
    if source:
        st.session_state[f"snd{sid}_source"] = source


def _profil_encadre(sid):
    """(couches_bas, couches_haut) : bornes basse et haute des modules."""
    bas, haut = [], []
    for lid in _layer_ids(sid):
        lv = _get_layer(sid, lid)
        if lv["h"] <= 0:
            continue
        Mb = lv["M"] or None
        Mh = st.session_state.get(_layer_key(sid, lid, "M_haut")) or Mb
        base = {"h": lv["h"], "gamma": lv["gamma"], "nom": lv["type"]}
        bas.append(dict(base, M=Mb))
        haut.append(dict(base, M=Mh))
    return bas, haut


# =============================================================
#  UI : IMPORT SANS IA
# =============================================================
def _render_import(sid):
    with st.expander("📥 Importer un sondage (GEF · CSV · PDF) — sans IA, sans clé API",
                     expanded=False):
        st.caption(
            "Les valeurs sont lues **directement dans le fichier**, jamais estimées. "
            "Par ordre de fidélité : fichier **GEF** (format d'échange belge/"
            "néerlandais, pas réel de 2 cm) → **CSV** → **PDF à courbes vectorielles** "
            "(les courbes sont relues et recalées sur les traits de grille). "
            "Un scan ou une photo ne contient aucune donnée lisible : demande le GEF "
            "ou le PDF d'origine au bureau d'essais.")

        up = st.file_uploader("Fichier de sondage",
                              type=["gef", "csv", "txt", "asc", "pdf"],
                              key=f"snd{sid}_upload")
        if up is None:
            return

        contenu = up.getvalue()
        nom = up.name

        if nom.lower().endswith(".pdf"):
            _render_import_pdf(sid, contenu, nom)
            return

        try:
            sondage = SI.importer(contenu, nom)
        except Exception as e:
            st.error(f"Lecture impossible : {e}")
            return
        _apercu_et_valider(sid, sondage)


def _render_import_pdf(sid, contenu, nom):
    """Import PDF : détection, recalage proposé, correction possible."""
    kpref = f"snd{sid}_pdf"
    try:
        analyse = SI.analyser_pdf(contenu, page_idx=int(st.session_state.get(f"{kpref}_page", 1)) - 1)
    except Exception as e:
        st.error(f"{e}")
        st.info("Astuce : si le rapport comporte un tableau de valeurs, il est souvent "
                "lisible même sans courbe vectorielle — essaie la page du tableau.")
        return

    calib = analyse["calibration"]
    c1, c2 = st.columns([1, 3])
    with c1:
        st.number_input("Page", min_value=1, max_value=max(1, calib.get("n_pages", 1)),
                        step=1, key=f"{kpref}_page")
    with c2:
        st.markdown(
            f"<div style='padding-top:30px;color:#475569;'>"
            f"{len(analyse['courbes'])} courbe(s) détectée(s) · "
            f"grille {calib['n_grille_v']}×{calib['n_grille_h']}</div>",
            unsafe_allow_html=True)

    for a in analyse["avertissements"]:
        st.warning(a)

    libelles = [f"Courbe {i+1} — {c['n']} points, amplitude {c['amplitude_x']:.0f} pt"
                for i, c in enumerate(analyse["courbes"])]
    cA, cB = st.columns(2)
    with cA:
        i_qc = st.selectbox("Courbe de résistance de pointe qc", range(len(libelles)),
                            format_func=lambda i: libelles[i], key=f"{kpref}_iqc")
    with cB:
        opts = [-1] + list(range(len(libelles)))
        i_fs = st.selectbox("Courbe de frottement fs (facultatif)", opts,
                            format_func=lambda i: "— aucune —" if i < 0 else libelles[i],
                            key=f"{kpref}_ifs")

    st.markdown("**Recalage des axes**")
    st.caption("Proposé automatiquement d'après les traits de grille et leurs étiquettes. "
               "Corrige-le si le rapport a une mise en page inhabituelle.")
    auto_ok = calib.get("auto_x") and calib.get("auto_y")
    if auto_ok:
        st.success(f"Recalage automatique — qc : R² = {calib['r2x']:.6f} · "
                   f"profondeur : R² = {calib['r2y']:.6f}")

    forcer = st.checkbox("Corriger le recalage à la main", key=f"{kpref}_manuel",
                         value=not auto_ok)
    calib_final = dict(calib)
    if forcer:
        st.caption("Donne la valeur portée par le bord gauche/droit du cadre (qc) et "
                   "par son bord haut/bas (profondeur).")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            qc_g = st.number_input("qc bord gauche", value=0.0, step=1.0, key=f"{kpref}_qcg")
        with g2:
            qc_d = st.number_input("qc bord droit", value=25.0, step=1.0, key=f"{kpref}_qcd")
        with g3:
            z_h = st.number_input("z bord haut [m]", value=0.0, step=1.0, key=f"{kpref}_zh")
        with g4:
            z_b = st.number_input("z bord bas [m]", value=20.0, step=1.0, key=f"{kpref}_zb")
        dx = calib["x1"] - calib["x0"]
        dy = calib["y1"] - calib["y0"]
        if abs(dx) > 1e-6 and abs(dy) > 1e-6 and qc_d != qc_g and z_b != z_h:
            calib_final["ax"] = (qc_d - qc_g) / dx
            calib_final["bx"] = qc_g - calib_final["ax"] * calib["x0"]
            calib_final["ay"] = (z_b - z_h) / dy
            calib_final["by"] = z_h - calib_final["ay"] * calib["y0"]

    # --- échelle propre de la courbe de frottement ---
    # Dans un rapport réel, fs n'est presque jamais tracé à la même échelle
    # que qc. Appliquer la calibration de qc à fs fausse Rf, donc Ic, donc
    # tout le classement du sol : le limon devient de la tourbe.
    facteur_fs = 1.0
    if i_fs >= 0:
        st.markdown("**Échelle de la courbe de frottement**")
        e1, e2 = st.columns([1, 2])
        with e1:
            facteur_fs = st.number_input(
                "fs tracé × ", min_value=0.001, max_value=1000.0, value=1.0,
                step=1.0, key=f"{kpref}_ffs",
                help="Beaucoup de rapports tracent fs sur l'axe des qc avec un "
                     "facteur d'agrandissement (« fs × 10 »). Indique-le ici. "
                     "Si fs a son propre axe, mets le rapport des deux pleines "
                     "échelles (ex. axe qc 0-25 MPa et axe fs 0-1 MPa → 25).")
        with e2:
            st.caption("Contrôle : le rapport de frottement Rf = fs/qc d'un sol réel "
                       "vaut 0,2 à 1 % dans un sable, 2 à 6 % dans une argile. "
                       "Une valeur hors de 0,05–12 % signale une échelle fausse.")

    try:
        sondage = SI.extraire_courbe(
            analyse, idx_qc=i_qc, idx_fs=(None if i_fs < 0 else i_fs),
            calib=calib_final, nom=re.sub(r"\.pdf$", "", nom, flags=re.I),
            facteur_fs=facteur_fs)
    except Exception as e:
        st.error(f"{e}")
        return

    rf = sondage.get("rf_median")
    if rf is not None:
        ok_rf = SI.RF_MIN_PLAUSIBLE <= rf <= SI.RF_MAX_PLAUSIBLE
        st.markdown(
            f"{'✅' if ok_rf else '❌'} Rapport de frottement médian obtenu : "
            f"**Rf = {rf:.2f} %**"
            + ("" if ok_rf else " — échelle de fs à corriger avant d'importer."))
    _apercu_et_valider(sid, sondage)


def _apercu_et_valider(sid, sondage):
    """Aperçu du sondage lu, découpage automatique, puis validation."""
    pts = sondage["points"]
    st.success(f"**{sondage['nom']}** — {sondage['source']} · "
               f"{len(pts)} points de {pts[0][0]:.2f} à {pts[-1][0]:.2f} m")
    for a in sondage.get("avertissements", []):
        st.warning(a)

    a_fs = any(p[2] is not None for p in pts)
    if not a_fs:
        st.warning(
            "Pas de frottement latéral fs dans ce fichier : la classification "
            "automatique du sol (Robertson) a besoin de qc **et** de fs. Les couches "
            "seront créées d'après qc seul et le type de sol restera à choisir.")

    c1, c2, c3 = st.columns(3)
    with c1:
        nappe = st.number_input("Nappe [m sous TN]", min_value=0.0, step=0.5,
                                value=float(sondage.get("nappe_m") or
                                            st.session_state.get("rs_nappe", 3.0)),
                                key=f"snd{sid}_imp_nappe")
    with c2:
        seuil = st.slider("Finesse du découpage", 0.10, 0.60, 0.25, 0.05,
                          key=f"snd{sid}_imp_seuil",
                          help="Écart d'indice Ic au-delà duquel on ouvre une nouvelle "
                               "couche. Plus petit = plus de couches.")
    with c3:
        h_min = st.number_input("Épaisseur minimale [m]", min_value=0.20, max_value=5.0,
                                value=0.50, step=0.10, key=f"snd{sid}_imp_hmin")

    couches = ST.profil_depuis_cpt(pts, nappe_m=nappe, seuil_ic=seuil, h_min=h_min,
                                   nu=float(st.session_state.get("rs_nu", 0.30)))
    if not couches:
        st.error("Aucune couche exploitable n'a pu être formée.")
        return

    lignes = []
    for c in couches:
        lignes.append({
            "de [m]": round(c["z0"], 2), "à [m]": round(c["z1"], 2),
            "h [m]": round(c["h"], 2),
            "Ic": None if c["Ic"] is None else round(c["Ic"], 2),
            "Type reconnu": c["sbt"],
            "qc moy [MPa]": round(c["qc"] or 0, 2),
            "M bas [MPa]": None if c["M_bas"] is None else round(c["M_bas"], 1),
            "M haut [MPa]": None if c["M_haut"] is None else round(c["M_haut"], 1),
        })
    st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)
    st.caption("Le type de sol est déduit de qc et fs par l'indice de comportement Ic "
               "(Robertson) — aucune saisie, aucune estimation. Le module M est encadré "
               "car les corrélations CPT divergent d'un facteur 2 à 4.")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("✅ Remplir le tableau", key=f"snd{sid}_valider",
                     use_container_width=True, type="primary"):
            _remplir_depuis_couches(sid, couches, nappe=nappe, source=sondage["source"])
            st.session_state[f"snd{sid}_points"] = pts
            # snd{sid}_nom porte un text_input rendu plus haut dans ce run :
            # écriture différée, sinon StreamlitAPIException.
            _differer({f"snd{sid}_nom": sondage["nom"]})
            st.rerun()
    with b2:
        st.download_button("⬇️ CSV des mesures", data=SI.vers_csv(sondage).encode("utf-8"),
                           file_name=f"{sondage['nom']}_mesures.csv", mime="text/csv",
                           use_container_width=True, key=f"snd{sid}_dl_pts")
    with b3:
        st.download_button("⬇️ CSV des couches",
                           data=SI.couches_vers_csv(sondage["nom"], couches).encode("utf-8"),
                           file_name=f"{sondage['nom']}_couches.csv", mime="text/csv",
                           use_container_width=True, key=f"snd{sid}_dl_cou")


# =============================================================
#  UI : TABLEAU DES COUCHES
# =============================================================
def _render_layer_row(sid, lid, i):
    kh, kt, kp, kq, kr, km, kg = (_layer_key(sid, lid, f)
                                  for f in ("h", "type", "type_prev", "qc", "rf", "M", "gamma"))
    st.session_state.setdefault(kh, 1.0)
    st.session_state.setdefault(kt, "—")
    st.session_state.setdefault(kp, st.session_state[kt])
    st.session_state.setdefault(kq, 0.0)
    st.session_state.setdefault(kr, 0.0)
    st.session_state.setdefault(km, 0.0)
    st.session_state.setdefault(kg, 19.0)

    lv = "visible" if i == 0 else "collapsed"
    va = "bottom" if i == 0 else "center"
    c_h, c_t, c_q, c_r, c_m, c_g, c_a = st.columns(LAYER_COLS, vertical_alignment=va)

    with c_h:
        st.number_input("h [m]", min_value=0.0, step=0.1, key=kh, label_visibility=lv)
    with c_t:
        st.selectbox("Type de sol", SB.soil_types_list(), key=kt, label_visibility=lv)

    # préremplissage sur changement de type (mutation avant rendu des widgets suivants)
    new_type = st.session_state.get(kt, "—")
    if new_type != st.session_state.get(kp):
        if new_type not in ("—", "Personnalisé") and \
                float(st.session_state.get(kq, 0) or 0) <= 0 and \
                float(st.session_state.get(km, 0) or 0) <= 0:
            d = SB.SOIL_DB.get(new_type, {})
            if d.get("cpt_ok"):
                q = SB.soil_default_qc(new_type)
                if q is not None:
                    st.session_state[kq] = q
                r = SB.soil_default_Rf(new_type)
                if r is not None:
                    st.session_state[kr] = r
            m = SB.soil_default_M(new_type)
            if m is not None:
                st.session_state[km] = m
        st.session_state[kg] = SB.soil_gamma(new_type, 19.0)
        st.session_state[kp] = new_type

    with c_q:
        st.number_input("qc [MPa]", min_value=0.0, step=0.5, key=kq, label_visibility=lv)
    with c_r:
        st.number_input("Rf [%]", min_value=0.0, step=0.1, key=kr, label_visibility=lv)
    with c_m:
        st.number_input("M [MPa]", min_value=0.0, step=5.0, key=km, label_visibility=lv,
                        help="Module OEDOMÉTRIQUE (déformation latérale empêchée). "
                             "C'est lui qui entre dans le tassement, pas le module de "
                             "Young E : M = E(1−ν)/((1+ν)(1−2ν)), soit ≈ 1,35·E pour "
                             "ν = 0,30." if i == 0 else None)
    with c_g:
        st.number_input("γ [kN/m³]", min_value=8.0, max_value=26.0, step=0.5, key=kg,
                        label_visibility=lv,
                        help="Sert au poids des terres σ'v0, donc à la profondeur "
                             "d'influence." if i == 0 else None)
    with c_a:
        if i == 0:
            st.button("＋", key=f"rs_add_l_{sid}_{lid}", use_container_width=True,
                      help="Ajouter une couche", on_click=_add_layer, args=(sid,))
        else:
            st.button("🗑️", key=f"rs_del_l_{sid}_{lid}", use_container_width=True,
                      help="Supprimer cette couche", on_click=_delete_layer, args=(sid, lid))

    lv2 = _get_layer(sid, lid)
    bits = []
    if lv2["h"] > 0 and lv2["M"] > 0:
        bits.append("✅ prise en compte")
    else:
        bits.append("⚠️ ignorée (h ou M manquant)")
    Ic = st.session_state.get(_layer_key(sid, lid, "Ic"))
    if Ic is not None:
        bits.append(f"Ic = {Ic:.2f}")
    sbt = st.session_state.get(_layer_key(sid, lid, "sbt"))
    if sbt:
        bits.append(f"CPT : {sbt}")
    Mh = st.session_state.get(_layer_key(sid, lid, "M_haut"))
    if Mh and lv2["M"] and Mh > lv2["M"]:
        bits.append(f"M encadré : {lv2['M']:.0f} – {Mh:.0f} MPa")
    elif lv2["type"] not in ("—", "Personnalisé"):
        sug = SB.suggest_M_from_qc(lv2["qc"], lv2["type"])
        if sug:
            bits.append(f"M suggéré (qc) : {sug:.0f} MPa")
        elif SB.is_rock(lv2["type"]):
            bits.append("roche : refus de pointe probable, qc sans objet")
    st.caption(" · ".join(bits))


def _render_sounding(sid):
    with st.container(border=True):
        cN, cA, cC, cD = st.columns([3.6, 0.7, 0.7, 0.7], vertical_alignment="bottom")
        with cN:
            st.text_input("Sondage", key=f"snd{sid}_nom")
        with cA:
            st.button("➕", key=f"rs_add_s_{sid}", help="Ajouter un sondage",
                      use_container_width=True, on_click=_add_sounding)
        with cC:
            st.button("📋", key=f"rs_cp_s_{sid}", help="Copier le sondage",
                      use_container_width=True, on_click=_copy_sounding, args=(sid,))
        with cD:
            if len(st.session_state.soundings) > 1:
                st.button("🗑️", key=f"rs_dl_s_{sid}", help="Supprimer le sondage",
                          use_container_width=True, on_click=_delete_sounding, args=(sid,))

        src = st.session_state.get(f"snd{sid}_source")
        if src:
            st.caption(f"Source : {src}")

        _render_import(sid)

        st.markdown("**Couches, depuis le TERRAIN NATUREL**")
        st.caption("Saisis tout le profil depuis la surface : le niveau d'assise est "
                   "appliqué par le calcul, il ne faut plus retirer les couches "
                   "supérieures à la main.")
        for i, lid in enumerate(_layer_ids(sid)):
            _render_layer_row(sid, lid, i)

        pts = st.session_state.get(f"snd{sid}_points")
        if pts:
            st.download_button(
                "⬇️ Exporter les mesures brutes en CSV",
                data=SI.vers_csv({"nom": _sounding_name(sid), "points": pts}).encode("utf-8"),
                file_name=f"{_sounding_name(sid)}_mesures.csv", mime="text/csv",
                use_container_width=True, key=f"rs_dlpts_{sid}")


# =============================================================
#  GRAPHIQUES
# =============================================================
def _fig_diffusion(couches, res, B, L, q, D, nappe, titre=""):
    """
    LE graphique : profil de sol, diffusion de la contrainte sous la
    fondation, contrainte en place, critère d'arrêt et profondeur
    d'influence. Il montre « comment ça bouge » avec la profondeur.
    """
    if not _HAS_MPL:
        return None
    tr = res.get("tranches") or []
    z_max = max(res.get("z_influence", 1.0) * 1.35, 1.0)

    fig, (axp, axs) = plt.subplots(
        1, 2, figsize=(9.2, 5.4), gridspec_kw={"width_ratios": [1, 2.1]}, sharey=True)

    # ---- panneau gauche : colonnes de sol ----
    cum = 0.0
    cmap = plt.get_cmap("YlOrBr")
    Ms = [c.get("M") for c in couches if c.get("M")]
    Mlo, Mhi = (min(Ms), max(Ms)) if Ms else (1.0, 100.0)
    for c in couches:
        h = c.get("h", 0.0)
        if h <= 0:
            continue
        z0, z1 = cum, cum + h
        cum = z1
        z0r, z1r = z0 - D, z1 - D
        if z1r <= 0 or z0r >= z_max:
            continue
        z0r, z1r = max(z0r, -D), min(z1r, z_max)
        M = c.get("M") or 0.0
        if M > 0 and Mhi > Mlo:
            t = (math.log10(max(M, .1)) - math.log10(max(Mlo, .1))) / \
                (math.log10(max(Mhi, .1)) - math.log10(max(Mlo, .1)) or 1)
        else:
            t = 0.0
        axp.axhspan(z0r, z1r, color=cmap(0.18 + 0.6 * min(max(t, 0), 1)),
                    ec="black", lw=0.7)
        if z1r - z0r > z_max * 0.05:
            axp.text(0.5, (z0r + z1r) / 2,
                     f"{(c.get('nom') or '—')[:22]}\nM = {M:.0f} MPa" if M else
                     f"{(c.get('nom') or '—')[:22]}\n⚠ M manquant",
                     ha="center", va="center", fontsize=7)
    axp.axhline(0, color="#111", lw=2.2)
    axp.text(0.5, -D * 0.5 if D > 0 else -0.02, "assise", ha="center", va="center",
             fontsize=8, color="#111",
             bbox=dict(fc="white", ec="none", alpha=.75, pad=1.5))
    if nappe is not None:
        zn = nappe - D
        if -D <= zn <= z_max:
            axp.axhline(zn, color="#2563eb", lw=1.3, ls=(0, (5, 3)))
            axp.text(0.02, zn, " nappe", color="#2563eb", fontsize=7.5, va="bottom")
    axp.set_xlim(0, 1)
    axp.set_xticks([])
    axp.set_ylabel("Profondeur sous l'assise [m]")
    axp.set_title("Profil", fontsize=10)

    # ---- panneau droit : contraintes ----
    if tr:
        zs = [t["z_sous_assise"] for t in tr]
        ds = [t["delta_sigma"] for t in tr]
        sv = [t["sigma_v0_eff"] for t in tr]
        crit = res.get("critere_pct", 20) / 100.0
        axs.fill_betweenx(zs, 0, ds, color="#b91c1c", alpha=.16,
                          label="Δσ mobilisée (comprime le sol)")
        axs.plot(ds, zs, color="#b91c1c", lw=2.1, label="Δσ apportée par la fondation")
        axs.plot(sv, zs, color="#334155", lw=1.3, ls="--", label="σ'v0 en place")
        axs.plot([c * crit for c in sv], zs, color="#059669", lw=1.3, ls=":",
                 label=f"critère d'arrêt : {crit:.0%} · σ'v0")
    zi = res.get("z_influence", 0.0)
    # L'étiquette doit dire POURQUOI on s'est arrêté : « profondeur
    # d'influence » n'est vrai que si le critère a réellement été atteint.
    # Un profil trop court tronque le tassement et SURESTIME k.
    tronque = "épuisé" in str(res.get("convergence", "")) or \
              "plafond" in str(res.get("convergence", ""))
    coul = "#b45309" if tronque else "#059669"
    fond = "#fffbeb" if tronque else "#ecfdf5"
    lib = (f" profil interrompu à {zi:.2f} m — k surestimé "
           if tronque else f" profondeur d'influence  {zi:.2f} m ")
    axs.axhline(zi, color=coul, lw=1.6, ls="--" if tronque else "-")
    axs.text(axs.get_xlim()[1] * .98, zi, lib,
             ha="right", va="bottom", fontsize=8.5, color=coul,
             bbox=dict(fc=fond, ec=coul, lw=.7, pad=2))
    axs.axhline(0, color="#111", lw=2.2)
    axs.set_xlabel("Contrainte [kPa]")
    axs.set_title(titre or "Diffusion de la contrainte sous la fondation", fontsize=10)
    axs.grid(alpha=.25)
    axs.legend(fontsize=7.5, loc="lower right", framealpha=.92)
    axs.set_ylim(z_max, -D if D > 0 else -0.05)
    fig.tight_layout()
    return fig


def _fig_k_vs_B(couches_bas, couches_haut, B, L, q, D, nappe, critere):
    if not _HAS_MPL:
        return None
    ratio = (L / B) if B > 0 else 1.0
    Bs, kb, kh = [], [], []
    for i in range(18):
        b = max(0.5, B * (0.25 + i * 0.15))
        Bs.append(b)
        rb = ST.tassement(couches_bas, b, b * ratio, q, D=D, nappe_m=nappe,
                          critere=critere, dz=0.10)
        rh = ST.tassement(couches_haut, b, b * ratio, q, D=D, nappe_m=nappe,
                          critere=critere, dz=0.10)
        kb.append(rb["k_MNm3"]); kh.append(rh["k_MNm3"])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.fill_between(Bs, kb, kh, color="#7c3aed", alpha=.16, label="encadrement")
    ax.plot(Bs, kb, color="#7c3aed", lw=1.6)
    ax.plot(Bs, kh, color="#7c3aed", lw=1.6, ls="--")
    ax.axvline(B, color="#94a3b8", ls="--", lw=1)
    ax.set_xlabel("Largeur de fondation B [m]")
    ax.set_ylabel("k [MN/m³]")
    ax.set_title("k n'est pas une propriété du sol : il dépend de la fondation",
                 fontsize=9.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fig_cpt(pts, couches, nappe=None):
    if not _HAS_MPL or not pts:
        return None
    z = [p[0] for p in pts]
    qc = [p[1] for p in pts]
    fs = [p[2] for p in pts if p[2] is not None]
    n = 2 if fs else 1
    fig, axes = plt.subplots(1, n + 1, figsize=(3.1 * (n + 1), 5.6), sharey=True)
    if n + 1 == 2:
        axes = list(axes)
    ax0 = axes[0]
    ax0.plot(qc, z, color="#1d4ed8", lw=.8)
    ax0.set_xlabel("qc [MPa]"); ax0.set_ylabel("Profondeur [m]")
    ax0.grid(alpha=.3)
    i = 1
    if fs:
        axf = axes[1]
        axf.plot([p[2] for p in pts if p[2] is not None],
                 [p[0] for p in pts if p[2] is not None], color="#b45309", lw=.8)
        axf.set_xlabel("fs [kPa]"); axf.grid(alpha=.3)
        i = 2
    axi = axes[i]
    zi = [c["z1"] for c in couches]
    Ici = [c["Ic"] for c in couches]
    for c in couches:
        if c["Ic"] is not None:
            axi.plot([c["Ic"], c["Ic"]], [c["z0"], c["z1"]], lw=3, color="#0f766e")
    for b, _, lab in ST.SBT_ZONES[:-1]:
        axi.axvline(b, color="#94a3b8", lw=.6, ls=":")
    axi.set_xlabel("Ic (Robertson)"); axi.grid(alpha=.3)
    axi.set_xlim(1.0, 4.0)
    for ax in axes:
        if nappe is not None:
            ax.axhline(nappe, color="#2563eb", lw=1.1, ls=(0, (5, 3)))
    ax0.set_ylim(max(z), 0)
    fig.tight_layout()
    return fig


def _fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =============================================================
#  RAPPORT PDF
# =============================================================
def _rapport_pdf(projet, resultats, params, images=None):
    if not _HAS_REPORTLAB:
        raise RuntimeError("reportlab n'est pas installé.")
    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=9,
                        spaceAfter=3, textColor=_rl.HexColor("#1f2937"))
    BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=9.2, leading=12.5)
    SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=7.8,
                           textColor=_rl.HexColor("#475569"))
    CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.2, leading=10.2)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")

    def P(t, b=False):
        return Paragraph(str(t), CELLB if b else CELL)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=13 * mm,
                            title="Note de calcul — Raideur élastique du sol")
    S = []
    S.append(Paragraph("Note de calcul — Coefficient de réaction du sol", H1))
    meta = []
    if projet.get("nom"):
        meta.append(f"Projet : <b>{projet['nom']}</b>")
    meta.append(f"Date : {projet.get('date')}")
    meta.append(f"Module rigidite_sol {VERSION}")
    S.append(Paragraph(" — ".join(meta), SMALL))
    S.append(Spacer(1, 5))

    S.append(Paragraph("1. Méthode", H2))
    S.append(Paragraph(
        "Le sol est représenté par des ressorts verticaux (Winkler) : q = k·w. Le "
        "coefficient de réaction n'est pas une propriété mesurable du sol — c'est le "
        "rapport entre la pression appliquée et le tassement qui en résulte. Il est donc "
        "obtenu en calculant d'abord le tassement.<br/><br/>"
        "<b>a.</b> La contrainte apportée par la fondation est diffusée dans le massif "
        "par la solution de Boussinesq, intégrée sur le rectangle chargé "
        "(facteur d'influence de Newmark) : Δσ(z) = q<sub>net</sub> · I(z).<br/>"
        "<b>b.</b> Le tassement est la somme des compressions de tranches : "
        "w = Σ Δσ<sub>i</sub>·h<sub>i</sub>/M<sub>i</sub>, où M est le module "
        "OEDOMÉTRIQUE de la couche.<br/>"
        "<b>c.</b> Le coefficient vaut k = q<sub>net</sub>/w.<br/>"
        "<b>d.</b> La profondeur d'influence n'est pas fixée a priori : l'intégration "
        "s'arrête lorsque la contrainte apportée devient négligeable devant la contrainte "
        "en place, Δσ ≤ " + f"{params['critere']:.0%}" + " · σ'<sub>v0</sub>.", BODY))
    S.append(Paragraph(
        "Les modules sont issus des essais lorsqu'ils y figurent ; à défaut ils sont "
        "déduits du CPT par l'indice de comportement I<sub>c</sub> (Robertson) et "
        "encadrés, les corrélations divergeant d'un facteur 2 à 4 selon les auteurs. "
        "Les deux bornes sont reportées : la structure doit être vérifiée aux deux.", BODY))

    S.append(Paragraph("2. Hypothèses de la fondation", H2))
    d = [[P("Grandeur", True), P("Valeur", True), P("Grandeur", True), P("Valeur", True)],
         [P("Largeur B"), P(f"{params['B']:.2f} m"),
          P("Longueur L"), P(f"{params['L']:.2f} m")],
         [P("Pression de contact q"), P(f"{params['q']:.0f} kPa"),
          P("Niveau d'assise D"), P(f"{params['D']:.2f} m")],
         [P("Nappe"), P("—" if params.get("nappe") is None else f"{params['nappe']:.2f} m"),
          P("Pression nette"), P(f"{params.get('q_net', 0):.0f} kPa")]]
    t = Table(d, colWidths=[38 * mm, 30 * mm, 38 * mm, 30 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, _rl.HexColor("#94a3b8")),
                           ("BACKGROUND", (0, 0), (-1, 0), _rl.HexColor("#eef2f7")),
                           ("TOPPADDING", (0, 0), (-1, -1), 2),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    S.append(t)

    for r in resultats:
        S.append(Paragraph(f"3. Sondage {r['nom']}", H2))
        data = [[P("N°", True), P("h [m]", True), P("Nature", True), P("γ", True),
                 P("M bas", True), P("M haut", True)]]
        for i, c in enumerate(r["couches"], 1):
            data.append([P(i), P(f"{c['h']:.2f}"), P(c.get("nom", "—")),
                         P(f"{c.get('gamma', 19):.1f}"),
                         P("—" if not c.get("M") else f"{c['M']:.0f}"),
                         P("—" if not c.get("M_haut") else f"{c['M_haut']:.0f}")])
        tt = Table(data, colWidths=[10 * mm, 16 * mm, 76 * mm, 16 * mm, 20 * mm, 20 * mm])
        tt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, _rl.HexColor("#94a3b8")),
                                ("BACKGROUND", (0, 0), (-1, 0), _rl.HexColor("#eef2f7")),
                                ("TOPPADDING", (0, 0), (-1, -1), 2),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        S.append(tt)
        S.append(Spacer(1, 4))
        S.append(Paragraph(
            f"Tassement calculé : <b>{r['w_bas']:.1f} à {r['w_haut']:.1f} mm</b> — "
            f"profondeur d'influence : <b>{r['z_infl']:.2f} m</b> sous l'assise — "
            f"coefficient de réaction au centre : "
            f"<b>k = {r['k_bas']:.2f} à {r['k_haut']:.2f} MN/m³</b>.", BODY))

        zr = [[P("Position", True), P("k bas [MN/m³]", True), P("k haut [MN/m³]", True),
               P("k bas [kN/m³]", True), P("k haut [kN/m³]", True)]]
        for pos, lab in ST.POSITIONS.items():
            z = r["zones"][pos]
            zr.append([P(lab), P(f"{z['k_bas_MNm3']:.2f}"), P(f"{z['k_haut_MNm3']:.2f}"),
                       P(f"{z['k_bas_MNm3']*1000:,.0f}".replace(",", " ")),
                       P(f"{z['k_haut_MNm3']*1000:,.0f}".replace(",", " "))])
        tz = Table(zr, colWidths=[46 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm])
        tz.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, _rl.HexColor("#94a3b8")),
                                ("BACKGROUND", (0, 0), (-1, 0), _rl.HexColor("#eef2f7")),
                                ("TOPPADDING", (0, 0), (-1, -1), 2),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        S.append(Spacer(1, 4))
        S.append(tz)
        S.append(Paragraph(
            "Le centre tasse davantage que les bords : k y est plus faible. Cette "
            "variation reproduit ce que produit un calcul itératif sol-structure et "
            "peut être encodée comme plusieurs zones de sol sous une même dalle.", SMALL))
        img = (images or {}).get(r["nom"])
        if img:
            S.append(Spacer(1, 5))
            S.append(RLImage(io.BytesIO(img), width=168 * mm, height=98 * mm))

    S.append(Paragraph("4. Références et limites", H2))
    S.append(Paragraph(
        "• NBN EN 1997-1 (Eurocode 7) et son ANB — calcul géotechnique. Le coefficient "
        "de réaction n'est pas un paramètre de résistance et ne fait l'objet d'aucun "
        "facteur partiel : l'encadrement remplace le coefficient de sécurité.<br/>"
        "• ISO 22476-1 — essais de pénétration statique (CPT).<br/>"
        "• Boussinesq (1885) ; Newmark (1935) pour l'intégration sur un rectangle.<br/>"
        "• Robertson (1990, 2009) — indice de comportement I<sub>c</sub> et corrélations "
        "vers le module oedométrique.<br/>"
        "• Terzaghi (1955) — correction de taille des coefficients mesurés à la plaque.<br/>"
        "• Le modèle de Winkler ignore le couplage entre ressorts et surestime les "
        "réactions en rive. Les valeurs de ce document relèvent du pré-dimensionnement "
        "et doivent être confrontées au rapport géotechnique du projet.", BODY))
    doc.build(S)
    return buf.getvalue()


# =============================================================
#  PAGE
# =============================================================
def show():
    # Les écritures différées par un callback du run précédent sont posées
    # AVANT tout rendu : c'est le seul moment où une clé de widget est
    # librement modifiable.
    _appliquer_differees()
    _init_state()

    st.markdown(
        "<style>.katex-display{text-align:left!important;margin:.2rem 0!important;}"
        ".memo-chip{display:inline-block;padding:2px 8px;border-radius:999px;"
        "background:#eef2ff;color:#3730a3;font-size:.8rem;}"
        ".small{color:#64748b;font-size:.9rem;}</style>", unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1, 1, 1])
    with cols[0]:
        if st.button("🏠 Accueil", use_container_width=True, key="rs_home"):
            st.session_state.page = "Accueil"
            st.rerun()
    with cols[1]:
        if st.button("🧹 Réinitialiser", use_container_width=True, key="rs_reset",
                     help="Ne réinitialise QUE ce module : les poutres et les dalles "
                          "des autres modules sont conservées."):
            _reset_module()
    with cols[2]:
        st.download_button("💾 Enregistrer", data=json.dumps(
            _payload(), indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="raideur_sol.json", mime="application/json",
            use_container_width=True, key="rs_save")
    with cols[3]:
        if st.button("📂 Ouvrir", use_container_width=True, key="rs_open_t"):
            st.session_state["rs_show_open"] = not st.session_state.get("rs_show_open", False)
    with cols[4]:
        st.markdown("<div style='padding-top:8px;text-align:center;color:#64748b;"
                    "font-size:.85rem;'>sans IA · sans clé API</div>",
                    unsafe_allow_html=True)
    with cols[5]:
        st.markdown(f"<div style='text-align:right;padding-top:10px;'>"
                    f"<span class='memo-chip'>{VERSION}</span></div>",
                    unsafe_allow_html=True)

    if st.session_state.get("rs_show_open"):
        up = st.file_uploader("Fichier JSON", type=["json"], key="rs_open_up",
                              label_visibility="collapsed")
        if up is not None:
            try:
                _charger(json.load(up))
                st.session_state["rs_show_open"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Fichier invalide : {e}")

    st.divider()
    st.markdown("# Raideur élastique des sols")
    st.markdown("<span class='small'>Coefficient de réaction de Winkler obtenu par "
                "calcul du tassement — diffusion de Boussinesq, module oedométrique, "
                "profondeur d'influence calculée.</span>", unsafe_allow_html=True)

    with st.expander("📘 Fiche mémo — ce que k est, et ce qu'il n'est pas", expanded=False):
        st.markdown(r"""
- **Winkler** : $q = k\,w \Rightarrow k = q/w$. Unités : $q$ en kPa, $w$ en m, $k$ en kN/m³
  (1 MN/m³ = 1000 kN/m³).
- **k n'est pas une propriété du sol.** Sur un même terrain, il varie d'un facteur 5 à 10
  entre une semelle isolée et un grand radier. Il faut donc le calculer POUR une fondation
  donnée — c'est pourquoi la géométrie est demandée ci-contre.
- **Méthode** : $\Delta\sigma(z) = q_{net}\,I(z)$ (Boussinesq/Newmark), puis
  $w = \sum \Delta\sigma_i h_i / M_i$, puis $k = q_{net}/w$.
- **M est le module OEDOMÉTRIQUE**, pas le module de Young :
  $M = E\,(1-\nu)/[(1+\nu)(1-2\nu)] \approx 1{,}35\,E$ pour $\nu = 0{,}30$.
- **Profondeur d'influence** : elle se calcule ($\Delta\sigma \le 20\,\% \cdot \sigma'_{v0}$),
  elle ne se décrète pas à $2B$. Sous un radier de 20 m, $2B = 40$ m n'a aucun sens.
- **Un k trop faible n'est pas « du côté de la sécurité »** : il étale la charge, augmente
  les tassements calculés mais diminue souvent les moments de pointe sous poteaux.
  D'où l'encadrement systématique, à passer dans les deux bornes.
- **Rocher** : le CPT est en refus de pointe — caractériser par RQD/pressiomètre, jamais
  par une corrélation sur qc.
        """)

    mode = st.selectbox(
        "Que veux-tu faire ?",
        ("1. Sondage CPT (import de fichier)",
         "2. Profil de sol saisi à la main",
         "3. Vérification rapide k = q / w",
         "4. Comparer les théories",
         "5. Abaque des sols",
         "6. Raideur d'un plat en béton"),
        key="rs_mode")

    gauche, droite = st.columns([1, 1.15])

    # =========================================================
    #  ENTRÉES
    # =========================================================
    with gauche:
        st.markdown("### Entrées")

        if mode.startswith(("1.", "2.")):
            with st.container(border=True):
                st.markdown("#### Fondation")
                st.caption("k dépend de la fondation : sans elle, le calcul n'a pas de sens.")
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.number_input("Largeur B [m]", min_value=0.2, step=0.1, key="rs_B",
                                    help="Petit côté de la zone chargée.")
                with f2:
                    st.number_input("Longueur L [m]", min_value=0.2, step=0.1, key="rs_L")
                with f3:
                    st.number_input("Assise D [m]", min_value=0.0, step=0.25, key="rs_D",
                                    help="Profondeur du dessous de fondation sous le "
                                         "terrain naturel. Les couches au-dessus sont "
                                         "automatiquement écartées du calcul.")
                g1, g2 = st.columns(2)
                with g1:
                    st.number_input("Pression q [kPa]", min_value=1.0, step=10.0, key="rs_q",
                                    help="Pression de contact à l'ELS sous la fondation.")
                with g2:
                    st.checkbox("Nappe", key="rs_nappe_active")
                    if st.session_state.rs_nappe_active:
                        st.number_input("Nappe [m sous TN]", min_value=0.0, step=0.5,
                                        key="rs_nappe", label_visibility="collapsed")

                with st.expander("Options de calcul", expanded=False):
                    o1, o2 = st.columns(2)
                    with o1:
                        st.select_slider(
                            "Critère de profondeur d'influence",
                            options=[10, 15, 20, 25, 30], key="rs_critere",
                            format_func=lambda v: f"Δσ ≤ {v} % · σ'v0",
                            help="20 % est l'usage courant ; 10 % est plus strict et "
                                 "descend plus profond (k plus faible).")
                    with o2:
                        st.number_input("ν (Poisson)", min_value=0.0, max_value=0.49,
                                        step=0.01, key="rs_nu",
                                        help="Sert à convertir E ↔ M et aux théories "
                                             "de comparaison.")
                    st.checkbox(
                        "Utiliser la pression NETTE (q − poids des terres excavées)",
                        key="rs_q_net",
                        help="Usage normal en fondation : le sol a déjà supporté le "
                             "poids des terres retirées.")

        if mode.startswith("1."):
            st.text_input("Nom du projet", key="rs_projet")
            st.markdown("#### Sondages")
            for s in list(st.session_state.soundings):
                _render_sounding(int(s["id"]))

        elif mode.startswith("2."):
            st.text_input("Nom du projet", key="rs_projet")
            st.markdown("#### Profil")
            st.caption("Saisis les couches depuis le terrain naturel, avec leur module "
                       "OEDOMÉTRIQUE M. Choisir un type de sol préremplit les valeurs.")
            for s in list(st.session_state.soundings):
                _render_sounding(int(s["id"]))

        elif mode.startswith("3."):
            st.markdown("**k à partir d'un couple (q, w)**")
            st.caption("Si tu disposes déjà d'un tassement calculé par le géotechnicien.")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("q [kPa]", min_value=0.0, step=5.0, value=150.0, key="rs3_q")
            with c2:
                st.number_input("w [mm]", min_value=0.01, step=1.0, value=20.0, key="rs3_w")

        elif mode.startswith("4."):
            st.markdown("**Comparaison des théories**")
            st.caption("Sol homogène. Sert à situer un ordre de grandeur, jamais à "
                       "produire la valeur retenue.")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("E [MPa]", min_value=0.5, step=5.0, value=25.0, key="rs4_E")
                st.number_input("B [m]", min_value=0.2, step=0.5, value=2.0, key="rs4_B")
            with c2:
                st.number_input("L [m]", min_value=0.2, step=0.5, value=2.0, key="rs4_L")
                st.number_input("ν", min_value=0.0, max_value=0.49, step=0.01,
                                value=0.30, key="rs4_nu")

        elif mode.startswith("5."):
            st.markdown("**Abaque des sols**")
            st.caption("Les coefficients tabulés sont des valeurs mesurées à la PLAQUE "
                       "de 0,30 m. La correction de taille de Terzaghi est appliquée "
                       "ci-contre pour la largeur choisie.")
            st.number_input("Largeur de fondation B [m]", min_value=0.3, step=0.5,
                            value=2.0, key="rs5_B")
            st.number_input("Tassement de référence [mm]", min_value=1.0, max_value=100.0,
                            step=5.0, key="rs_w_ref")

        else:
            st.markdown("**Raideur d'un plat en béton**")
            st.caption("Contact assimilé à une compression 1D : k = E/h_c.")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("B [mm]", min_value=20.0, step=10.0, value=200.0, key="rs6_B")
            with c2:
                st.number_input("L [mm]", min_value=20.0, step=10.0, value=200.0, key="rs6_L")
            with c3:
                st.number_input("α (h_c = α·min(B,L))", min_value=0.05, step=0.05,
                                value=0.5, key="rs6_a")
            c4, c5 = st.columns(2)
            with c4:
                st.number_input("E béton [GPa]", min_value=5.0, step=1.0, value=30.0,
                                key="rs6_E")
            with c5:
                st.checkbox("Lit de mortier", key="rs6_grout")
            if st.session_state.get("rs6_grout"):
                c6, c7 = st.columns(2)
                with c6:
                    st.number_input("Épaisseur [mm]", min_value=1.0, step=1.0, value=20.0,
                                    key="rs6_tg")
                with c7:
                    st.number_input("E mortier [GPa]", min_value=1.0, step=1.0, value=20.0,
                                    key="rs6_Eg")

    # =========================================================
    #  RÉSULTATS
    # =========================================================
    with droite:
        st.markdown("### Résultats")
        st.checkbox("📘 Détail des calculs", key="rs_detail")

        if mode.startswith(("1.", "2.")):
            _resultats_profil()
        elif mode.startswith("3."):
            q = st.session_state.get("rs3_q", 0.0)
            w = st.session_state.get("rs3_w", 20.0)
            k = q / (w / 1000.0) if w > 0 else 0.0
            _bloc("Coefficient de réaction", f"k = {_fr(k/1000, 2)} MN/m³",
                  "ok" if k > 0 else "nok")
            if st.session_state.rs_detail and k > 0:
                st.latex(r"k = \dfrac{q}{w}")
                st.latex(f"k = \\dfrac{{{q:.1f}}}{{{w/1000:.4f}}} = "
                         f"{k:,.0f}\\,\\text{{kN/m³}} = {k/1000:,.2f}\\,\\text{{MN/m³}}"
                         .replace(",", " "))
                st.caption("Valable si w provient d'un vrai calcul de tassement, pour "
                           "cette fondation et cette charge.")
        elif mode.startswith("4."):
            _resultats_theories()
        elif mode.startswith("5."):
            _resultats_abaque()
        else:
            _resultats_plat()

        st.divider()
        st.markdown("<div class='small'>Valeurs de pré-dimensionnement. Les modules "
                    "déduits d'un CPT sont des corrélations : elles divergent d'un "
                    "facteur 2 à 4 selon les auteurs, d'où l'encadrement systématique. "
                    "Se référer au rapport géotechnique et à l'EN 1997 pour le "
                    "dimensionnement final.</div>", unsafe_allow_html=True)


# =============================================================
#  RÉSULTATS — PROFIL
# =============================================================
def _resultats_profil():
    B = float(st.session_state.rs_B)
    L = float(st.session_state.rs_L)
    q = float(st.session_state.rs_q)
    D = float(st.session_state.rs_D)
    nappe = float(st.session_state.rs_nappe) if st.session_state.rs_nappe_active else None
    crit = float(st.session_state.rs_critere) / 100.0
    q_net = bool(st.session_state.rs_q_net)

    if L < B:
        st.info("B est le petit côté par convention : B et L ont été échangés.")
        B, L = L, B

    resultats, images, ecartes = [], {}, []
    for s in st.session_state.soundings:
        sid = int(s["id"])
        bas, haut = _profil_encadre(sid)
        if not bas or all(not c.get("M") for c in bas):
            ecartes.append(_sounding_name(sid))
            continue
        zones = {}
        for pos in ST.POSITIONS:
            rb = ST.tassement(bas, B, L, q, D=D, nappe_m=nappe, critere=crit,
                              position=pos, q_net=q_net)
            rh = ST.tassement(haut, B, L, q, D=D, nappe_m=nappe, critere=crit,
                              position=pos, q_net=q_net)
            zones[pos] = {
                "k_bas_MNm3": min(rb["k_MNm3"], rh["k_MNm3"]),
                "k_haut_MNm3": max(rb["k_MNm3"], rh["k_MNm3"]),
                "w_bas_mm": min(rb["w_mm"], rh["w_mm"]),
                "w_haut_mm": max(rb["w_mm"], rh["w_mm"]),
                "detail": rb,
            }
        c = zones["centre"]
        rb_c = c["detail"]
        rb_c["critere_pct"] = st.session_state.rs_critere
        resultats.append({
            "sid": sid, "nom": _sounding_name(sid), "zones": zones,
            "k_bas": c["k_bas_MNm3"], "k_haut": c["k_haut_MNm3"],
            "w_bas": c["w_bas_mm"], "w_haut": c["w_haut_mm"],
            "z_infl": rb_c["z_influence"], "detail": rb_c,
            "couches": [dict(cc, M_haut=hh.get("M")) for cc, hh in zip(bas, haut)],
            "q_net": rb_c["q_net_kPa"], "convergence": rb_c["convergence"],
        })

    if not resultats:
        st.warning("Renseigne au moins une couche avec une épaisseur et un module M, "
                   "ou importe un sondage.")
        return

    # Un profil troué (couche sans M dans la zone d'influence) ne donne PAS
    # un k : le moteur refuse désormais de le calculer plutôt que de traiter
    # la couche manquante comme incompressible.
    incalculables = [r for r in resultats if "incomplet" in r["convergence"]]

    # ---- panneau SCIA ----
    with st.container(border=True):
        st.markdown("#### 🎯 À encoder dans SCIA (paramètres de sol C1z)")

        if ecartes:
            st.warning(
                f"⚠️ **{', '.join(ecartes)} n'entre pas dans les résultats** : aucune "
                "couche ne porte de module M. Les valeurs ci-dessous ne couvrent que "
                f"{len(resultats)} sondage(s) sur {len(resultats) + len(ecartes)}.")
        for r in incalculables:
            st.error(
                f"❌ **{r['nom']} : k non calculable.** {r['convergence']}. Une couche "
                "sans module ne peut pas être traitée comme incompressible — ce serait "
                "l'hypothèse la plus favorable possible, et elle surestime k. Complète "
                "la colonne M de cette couche.")
        pos_lbl = st.radio("Position de référence", list(ST.POSITIONS.keys()),
                           format_func=lambda p: ST.POSITIONS[p], horizontal=True,
                           key="rs_pos_scia")
        rows = []
        for r in resultats:
            z = r["zones"][pos_lbl]
            rows.append({
                "Sondage": r["nom"],
                "k bas [MN/m³]": round(z["k_bas_MNm3"], 2),
                "k haut [MN/m³]": round(z["k_haut_MNm3"], 2),
                "k bas [kN/m³]": round(z["k_bas_MNm3"] * 1000),
                "k haut [kN/m³]": round(z["k_haut_MNm3"] * 1000),
                "w [mm]": f"{z['w_bas_mm']:.0f} – {z['w_haut_mm']:.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ks = [r["zones"][pos_lbl]["k_bas_MNm3"] for r in resultats] + \
             [r["zones"][pos_lbl]["k_haut_MNm3"] for r in resultats]
        ks = [v for v in ks if v > 0]
        lib_env = ("Enveloppe tous sondages" if not (ecartes or incalculables)
                   else f"Enveloppe des {len(resultats) - len(incalculables)} sondages "
                        f"calculés sur {len(resultats) + len(ecartes)}")
        if ks and max(ks) - min(ks) > 1e-6:
            st.markdown(f"**{lib_env} : k = {min(ks):.2f} à "
                        f"{max(ks):.2f} MN/m³** — la dalle doit être vérifiée aux "
                        "DEUX bornes : un k faible augmente les tassements, un k élevé "
                        "augmente les moments de pointe.")
        elif ks:
            st.markdown(f"**k = {ks[0]:.2f} MN/m³** (valeur unique : les modules ont été "
                        "saisis directement, il n'y a donc pas d'incertitude de "
                        "corrélation à reporter).")
            st.caption("Un module issu d'un CPT serait encadré : les corrélations "
                       "divergent d'un facteur 2 à 4. Pense à faire varier M à la main "
                       "pour mesurer la sensibilité de ta dalle.")

        # Profil trop court : le tassement est tronqué, donc k SURESTIMÉ.
        # C'est un défaut non conservatif : il doit être visible sans déplier.
        courts = [r for r in resultats
                  if "épuisé" in r["convergence"] or "plafond" in r["convergence"]]
        if courts:
            noms_c = ", ".join(r["nom"] for r in courts)
            manque = max(r["z_infl"] for r in courts)
            st.error(
                f"⚠️ **Profil trop court — k surestimé.** Pour {noms_c}, l'intégration "
                f"s'est arrêtée à {manque:.2f} m sous l'assise parce que les couches "
                f"saisies s'arrêtent là, et non parce que la contrainte était devenue "
                f"négligeable. Le tassement est donc tronqué et le k affiché est **trop "
                f"élevé** — c'est un écart du mauvais côté pour les moments de la dalle. "
                f"Prolonge le profil : sous une fondation de {B:.1f} m, il faut "
                f"typiquement descendre à {2.5 * B:.0f} m.")

        st.markdown("**Zonage de la dalle** — sans module itératif sol-structure, "
                    "c'est la façon d'approcher la raideur variable :")
        zr = []
        for pos, lab in ST.POSITIONS.items():
            vals = [r["zones"][pos] for r in resultats]
            zr.append({
                "Zone": lab,
                "k bas [MN/m³]": round(min(v["k_bas_MNm3"] for v in vals), 2),
                "k haut [MN/m³]": round(max(v["k_haut_MNm3"] for v in vals), 2),
            })
        st.dataframe(pd.DataFrame(zr), use_container_width=True, hide_index=True)
        st.caption("Le centre tasse plus que les bords, donc k y est plus faible. "
                   "Encoder un k unique concentre artificiellement les réactions en rive : "
                   "c'est le défaut connu du modèle de Winkler, pas de ce calcul.")

    # ---- graphique de diffusion ----
    st.markdown("#### Comment la contrainte se diffuse")
    noms = [r["nom"] for r in resultats]
    sel = st.selectbox("Sondage", noms, key="rs_chart_snd")
    r_sel = next(r for r in resultats if r["nom"] == sel)
    if _HAS_MPL:
        fig = _fig_diffusion(r_sel["couches"], r_sel["detail"], B, L, q, D, nappe,
                             titre=f"{r_sel['nom']} — {ST.POSITIONS['centre'].lower()}")
        if fig:
            png = _fig_bytes(fig)
            images[r_sel["nom"]] = png
            st.image(png, use_container_width=True)
        st.caption(
            "À gauche le profil, à droite les contraintes. L'aire rouge est ce qui "
            "comprime réellement le sol : elle décroît avec la profondeur au lieu de "
            "rester égale à q. Le calcul s'arrête où la courbe rouge croise la courbe "
            "verte — c'est la profondeur d'influence, elle n'est pas imposée.")
        with st.expander("k en fonction de la largeur de fondation", expanded=False):
            bas, haut = _profil_encadre(r_sel["sid"])
            f2 = _fig_k_vs_B(bas, haut, B, L, q, D, nappe, crit)
            if f2:
                st.image(_fig_bytes(f2), use_container_width=True)
            st.caption("À sol identique, k varie fortement avec la taille de la "
                       "fondation. C'est pourquoi une valeur de k n'a de sens qu'avec "
                       "la géométrie qui l'accompagne.")
        pts = st.session_state.get(f"snd{r_sel['sid']}_points")
        if pts:
            with st.expander("Courbes CPT et classification", expanded=False):
                cou = ST.profil_depuis_cpt(pts, nappe_m=nappe)
                f3 = _fig_cpt(pts, cou, nappe)
                if f3:
                    st.image(_fig_bytes(f3), use_container_width=True)

    # ---- détail ----
    for r in resultats:
        etat = "ok" if r["k_bas"] > 0 else "nok"
        with st.expander(f"{'🟢' if etat == 'ok' else '🔴'} {r['nom']} — "
                         f"k = {r['k_bas']:.2f} à {r['k_haut']:.2f} MN/m³", expanded=False):
            _bloc("Tassement calculé",
                  f"w = {r['w_bas']:.1f} à {r['w_haut']:.1f} mm", etat)
            m1, m2, m3 = st.columns(3)
            m1.metric("Pression nette", f"{r['q_net']:.0f} kPa")
            m2.metric("Profondeur d'influence", f"{r['z_infl']:.2f} m")
            m3.metric("k au centre", f"{r['k_bas']:.2f}–{r['k_haut']:.2f}")
            st.caption(f"Arrêt de l'intégration : {r['convergence']}.")
            if st.session_state.rs_detail:
                st.latex(r"\Delta\sigma(z) = q_{net}\cdot I(z)\quad;\quad"
                         r"w=\sum_i \frac{\Delta\sigma_i\,h_i}{M_i}\quad;\quad k=\frac{q_{net}}{w}")
                tr = r["detail"].get("tranches") or []
                if tr:
                    pas = max(1, len(tr) // 12)
                    df = pd.DataFrame([{
                        "z sous assise [m]": round(t["z_sous_assise"], 2),
                        "Δσ [kPa]": round(t["delta_sigma"], 1),
                        "σ'v0 [kPa]": round(t["sigma_v0_eff"], 1),
                        "Δσ/σ'v0": round(t["ratio"], 2) if t["ratio"] else None,
                        "M [MPa]": round(t["M"], 1),
                        "dw [mm]": round(t["dw_mm"], 3),
                        "couche": t["couche"],
                    } for t in tr[::pas]])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.caption(f"Une ligne sur {pas} — {len(tr)} tranches au total.")
            if r["detail"].get("h_sans_module", 0) > 0:
                st.warning(f"{r['detail']['h_sans_module']:.2f} m de profil sans module M "
                           "n'ont pas été comptés : complète ces couches.")

    # ---- rapport ----
    if _HAS_REPORTLAB:
        if st.button("📄 Générer la note de calcul", key="rs_pdf",
                     use_container_width=True):
            try:
                params = {"B": B, "L": L, "q": q, "D": D, "nappe": nappe,
                          "critere": crit, "q_net": resultats[0]["q_net"]}
                st.session_state["rs_pdf_bytes"] = _rapport_pdf(
                    {"nom": st.session_state.get("rs_projet", ""),
                     "date": date.today().strftime("%d/%m/%Y")},
                    resultats, params, images)
            except Exception as e:
                st.session_state.pop("rs_pdf_bytes", None)
                st.error(f"Erreur : {e}")
        if st.session_state.get("rs_pdf_bytes"):
            nomp = re.sub(r"[^A-Za-z0-9]+", "_",
                          st.session_state.get("rs_projet", "") or "Projet")[:20]
            st.download_button(
                "⬇️ Télécharger la note", data=st.session_state["rs_pdf_bytes"],
                file_name=f"{nomp}_Raideur_sol_{date.today().strftime('%d-%m-%Y')}.pdf",
                mime="application/pdf", use_container_width=True, key="rs_pdf_dl")


# =============================================================
#  RÉSULTATS — AUTRES MODES
# =============================================================
def _resultats_theories():
    E = float(st.session_state.get("rs4_E", 25.0))
    B = float(st.session_state.get("rs4_B", 2.0))
    L = float(st.session_state.get("rs4_L", 2.0))
    nu = float(st.session_state.get("rs4_nu", 0.30))
    M = ST.module_oedometrique(E, nu)

    couches = [{"h": 60.0, "gamma": 19.0, "M": M, "nom": "sol homogène"}]
    q = 150.0
    r = ST.tassement(couches, B, L, q, D=0.0, critere=0.20, q_net=False)

    rows = [
        {"Méthode": "Tassement calculé (retenue)", "k [MN/m³]": round(r["k_MNm3"], 2),
         "Domaine": "profil quelconque, fondation réelle"},
        {"Méthode": "Élastique — semelle rigide", "k [MN/m³]":
            round(ST.k_elastique(E, B, nu, Is=0.88), 2),
         "Domaine": "massif homogène semi-infini"},
        {"Méthode": "Élastique — souple, centre", "k [MN/m³]":
            round(ST.k_elastique(E, B, nu, Is=1.12), 2),
         "Domaine": "massif homogène semi-infini"},
        {"Méthode": "Vesić (1961)", "k [MN/m³]": round(ST.k_vesic(E, B, nu), 2),
         "Domaine": "sol homogène, ordre de grandeur"},
        {"Méthode": "Ancienne formule 1/k = Σh/E (2B)", "k [MN/m³]":
            round(E / (2 * B), 2),
         "Domaine": "⚠ contrainte supposée uniforme — biaisée"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Sol homogène E = {E:.0f} MPa (soit M = {M:.0f} MPa pour ν = {nu:.2f}), "
        f"semelle {B:.1f}×{L:.1f} m, q = {q:.0f} kPa. La dernière ligne est la formule "
        "de la version 4 : elle donne exactement E/(2B), donc un résultat piloté par la "
        "règle des 2B et non par le sol — d'où le facteur 2 à 2,5 d'écart.")
    _bloc("Valeur retenue", f"k = {r['k_MNm3']:.2f} MN/m³",
          "ok" if r["k_MNm3"] > 0 else "nok")
    st.caption(f"Tassement {r['w_mm']:.1f} mm, profondeur d'influence "
               f"{r['z_influence']:.2f} m.")


def _resultats_abaque():
    B = float(st.session_state.get("rs5_B", 2.0))
    w_ref = float(st.session_state.get("rs_w_ref", 20.0))
    rows = []
    for nom, d in SB.SOIL_DB.items():
        if nom == "Personnalisé" or d.get("kp_min") is None:
            continue
        kb = ST.k_terzaghi_taille(d["kp_min"], B, nature=d["nature"])
        kh = ST.k_terzaghi_taille(d["kp_max"], B, nature=d["nature"])
        rows.append({
            "Catégorie": d["category"], "Type de sol": nom,
            "γ [kN/m³]": d["gamma"],
            "M [MPa]": f"{d['M_min']:.0f} – {d['M_max']:.0f}" if d["M_min"] else "—",
            "k plaque 0,30 m": f"{d['kp_min']} – {d['kp_max']}",
            f"k pour B = {B:.1f} m": f"{kb:.1f} – {kh:.1f}",
            "q pour w réf [kPa]": f"{kb*1000*w_ref/1000:.0f} – {kh*1000*w_ref/1000:.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.warning(
        "**Les deux colonnes de k ne sont pas interchangeables.** Celle de gauche est "
        "la valeur mesurée sur une plaque de 0,30 m (tables de Terzaghi 1955) ; celle de "
        "droite lui applique la correction de taille pour la largeur choisie. Utiliser "
        "la colonne de gauche pour une fondation réelle surestime k d'un facteur 3 à 30. "
        "La version 4 de ce module ne faisait pas cette correction.")
    st.caption("Un abaque reste un ordre de grandeur : le mode « Sondage CPT » calcule "
               "la valeur pour ton sol et ta fondation.")


def _resultats_plat():
    Bmm = float(st.session_state.get("rs6_B", 200.0))
    Lmm = float(st.session_state.get("rs6_L", 200.0))
    a = float(st.session_state.get("rs6_a", 0.5))
    Ec = float(st.session_state.get("rs6_E", 30.0))
    hc = a * min(Bmm, Lmm) / 1000.0
    kc = (Ec * 1e6) / hc if hc > 0 else 0.0
    keq = kc
    kg = 0.0
    if st.session_state.get("rs6_grout"):
        tg = float(st.session_state.get("rs6_tg", 20.0)) / 1000.0
        Eg = float(st.session_state.get("rs6_Eg", 20.0))
        kg = (Eg * 1e6) / tg if tg > 0 else 0.0
        if kc > 0 and kg > 0:
            keq = 1.0 / (1.0 / kc + 1.0 / kg)
    _bloc("Raideur du contact", f"k = {keq/1000:,.0f} MN/m³".replace(",", " "),
          "ok" if keq > 0 else "nok")
    if st.session_state.rs_detail and hc > 0:
        st.latex(r"h_c = \alpha\cdot\min(B,L)")
        st.latex(f"h_c = {hc*1000:.1f}\\,\\text{{mm}}")
        st.latex(r"k_c = E_c / h_c")
        st.latex(f"k_c = {kc/1000:,.0f}\\,\\text{{MN/m³}}".replace(",", " "))
        if kg > 0:
            st.latex(r"1/k_{eq} = 1/k_c + 1/k_g")
            st.latex(f"k_g = {kg/1000:,.0f}\\,\\text{{MN/m³}}".replace(",", " "))


# =============================================================
#  SAUVEGARDE / OUVERTURE
# =============================================================
def _payload():
    vals = {k: v for k, v in st.session_state.items()
            if _est_cle_module(k) and isinstance(v, (int, float, str, bool, type(None)))}
    return {"version": VERSION,
            "soundings": [{"id": int(s["id"]), "nom": str(s["nom"])}
                          for s in st.session_state.get("soundings", [])],
            "orders": {str(int(s["id"])): _layer_ids(int(s["id"]))
                       for s in st.session_state.get("soundings", [])},
            "values": vals}


def _charger(payload):
    for k in [k for k in list(st.session_state.keys()) if _est_cle_module(k)]:
        st.session_state.pop(k, None)
    st.session_state.soundings = [{"id": int(s["id"]), "nom": str(s["nom"])}
                                  for s in payload.get("soundings", [])] or []
    for sid, order in (payload.get("orders") or {}).items():
        st.session_state[_order_key(int(sid))] = list(order)
    for k, v in (payload.get("values") or {}).items():
        st.session_state[k] = v
    _init_state()

# -*- coding: utf-8 -*-
"""Prédalle en béton armé — la lecture d'un FICHIER de l'écran Streamlit (« 💾 Enregistrer » : ``predalle_ba.json``,
format ``{"version": "predalle-1.0", "predalles": [...], "values": {...}}``), sans Streamlit.

La lecture refait ce que l'écran fait à l'ouverture (``_load_from_payload``) puis au run suivant, dans le même ordre :
paramètres globaux (``_ensure_global_defaults``), prédalle 1 et section 1 toujours présentes, puis pour chaque prédalle
(``_ensure_defaults_for_dalle``) : h = 22 et peau de 6 cm par défaut, peau RAMENÉE dans [3 ; min(15 ; h − 1)],
classe et nuance contrôlées, MIGRATION v1 → v2 de chaque section (M_inf → Mx_inf, couches inf/sup → direction X),
défauts et coercitions de chaque couche ; puis les textes « 28,00 » des sollicitations (``_sync_float_raw_keys``) ;
puis les champs « Dist. axe » et « CDG » (``_sync_dist_state``, ``_sync_ycdg_state`` : automatiques tant que rien n'est
saisi) — avec la RÈGLE PRÉDALLE des axes (``beton.predalle.moteur.distance_auto``), et non celle de la dalle.
``tests/test_predalle_moteur.py`` le prouve en relisant les fichiers que l'écran lui-même écrit et ouvre.

Ce que l'écran remplace sans rien dire est relevé dans ``remarques`` (peau ramenée dans ses bornes, moment négatif lu 0,
axe ou centre de gravité illisibles lus « automatique », type ou Ø inconnus ramenés au défaut…) ; ce que le moteur ne
peut pas représenter (treillis illisible, largeur hors des bornes de l'écran…) lève ValueError avec le nom de la clé.
Un fichier du module DALLE (clés « dal… ») ouvert ici ne donne que les noms des dalles et des sections, comme l'écran
(ses valeurs ne sont pas lues) : c'est relevé aussi.
"""
import math

from beton import treillis as TR
from beton.dalle import moteur as MD

from . import entree as E
from . import moteur as M

TYPES_ECRAN = {"Treillis": "treillis", "Barres": "barres", "n barres": "n_barres"}
_PAR_FACE_V1 = ("ncouches", "ycdg", "ycdg_auto", "ycdg_lastauto")
_PAR_COUCHE_V1 = ("arm_type", "treillis", "ø_barres", "esp_barres")


def _nombre(v):
    """float(str(v).replace(",", ".")) ou None — la lecture des champs texte de l'écran."""
    try:
        x = float(str(v).strip().replace(",", "."))
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _entier_borne(v, defaut: int, mini: int, maxi: int) -> int:
    try:
        x = int(float(v or defaut))
    except Exception:
        x = defaut
    return max(mini, min(maxi, x))


def _migrer(vals: dict, did: int, sid: int) -> None:
    """_migrate_section_v2 : l'unidirectionnel devient la direction X (une valeur v1 n'est copiée que si la clé v2
    n'existe pas) ; les clés v1 sont retirées."""
    def ks(base):
        return f"pre{did}_sec{sid}_{base}"

    def bascule(old, new):
        if old in vals and new not in vals:
            vals[new] = vals[old]
        vals.pop(old, None)
    for old, new in (("M_inf", "Mx_inf"), ("M_sup", "Mx_sup")):
        bascule(ks(old), ks(new))
        bascule(ks(old) + "_raw", ks(new) + "_raw")
    for face in ("inf", "sup"):
        for fam in _PAR_FACE_V1:
            bascule(ks(f"{fam}_{face}"), ks(f"{fam}_{face}_x"))
        for i in range(1, E.MAX_COUCHES + 1):
            for fam in _PAR_COUCHE_V1:
                bascule(ks(f"{fam}_{face}_c{i}"), ks(f"{fam}_{face}_x_c{i}"))


def lire(payload: dict) -> dict:
    """Le contenu d'un fichier de l'écran → ``{"predalles": [prédalle normalisée, …], "gamma_s", "jeu_premier_lit",
    "remarques": [texte, …]}`` (chaque prédalle porte γs et le jeu, paramètres globaux de l'écran)."""
    if not isinstance(payload, dict):
        raise ValueError("fichier : un objet JSON {\"predalles\": …, \"values\": …} attendu")
    remarques = []
    values = payload.get("values", {})
    vals = dict(values) if isinstance(values, dict) else {}
    # --- _load_from_payload : la liste des prédalles (ou, à défaut, des « dalles ») et de leurs sections
    liste = payload.get("predalles", payload.get("dalles", None))
    if "predalles" not in payload and isinstance(payload.get("dalles"), list):
        remarques.append("fichier du module Dalle : l'écran Prédalle n'en reprend que les noms (ses valeurs « dal… » ne sont pas lues)")
    predalles = []
    if isinstance(liste, list) and liste:
        for d in liste:
            try:
                did = int(d.get("id"))
            except Exception:
                continue
            secs = d.get("sections", [])
            if not isinstance(secs, list) or not secs:
                secs = [{"id": 1, "nom": "A"}]
            propres = []
            for s in secs:
                try:
                    sid = int(s.get("id"))
                except Exception:
                    continue
                propres.append({"id": sid, "nom": str(s.get("nom", f"Section {sid}"))})
            predalles.append({"id": did, "nom": str(d.get("nom", f"Prédalle {did}")), "sections": propres})
    if not predalles:
        predalles = [{"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]}]
    # seules les clés de la prédalle et les paramètres globaux sont repris (_is_dalle_key, PERSISTED_GLOBAL_KEYS)
    # --- _ensure_global_defaults
    jeu = _nombre(vals.get("jeu_enrobage_cm", 0.0))
    try:
        gs = float(vals.get("gamma_s", 1.5) or 1.5)
    except Exception:
        gs = 1.5
    if gs <= 0:
        gs = 1.5
    # --- _init_dalles_if_needed : prédalle 1 et section 1 toujours présentes
    for d in predalles:
        if not d["sections"]:
            d["sections"] = [{"id": 1, "nom": "A"}]
    if not any(d["id"] == 1 for d in predalles):
        predalles.insert(0, {"id": 1, "nom": "Prédalle 1", "sections": [{"id": 1, "nom": "A"}]})
    for d in predalles:
        if not any(s["id"] == 1 for s in d["sections"]):
            d["sections"].insert(0, {"id": 1, "nom": "A"})
    # --- _ensure_defaults_for_dalle (migration de chaque section comprise), PUIS _sync_float_raw_keys
    for d in predalles:
        for s in d["sections"]:
            _migrer(vals, d["id"], s["id"])
    for k in list(vals):
        if k.endswith("_raw"):
            x = _nombre(vals[k])
            if x is not None:
                if x < 0:
                    remarques.append(f"{k[:-4]} : {vals[k]!r} lu 0 (l'écran ramène un négatif à 0)")
                vals[k[:-4]] = max(0.0, x)
    if jeu is None or jeu < 0:
        raise ValueError(f"jeu_enrobage_cm : nombre ≥ 0 attendu (reçu {vals.get('jeu_enrobage_cm')!r})")
    out = [_predalle(vals, d, gs, jeu, remarques) for d in predalles]
    return {"predalles": out, "gamma_s": out[0]["gamma_s"], "jeu_premier_lit": out[0]["jeu_premier_lit"], "remarques": remarques}


def _predalle(vals: dict, d: dict, gs: float, jeu: float, remarques: list) -> dict:
    did = d["id"]

    def kd(base):
        return f"pre{did}_{base}"
    nom = str(vals.get(f"meta_predalle_nom_{did}", d["nom"]))
    beton = vals.get(kd("beton"), E.BETON_DEFAUT)
    if beton not in E.classes_beton():
        remarques.append(f"{kd('beton')} : {beton!r} inconnue, lue {E.BETON_DEFAUT} (comme l'écran)")
        beton = E.BETON_DEFAUT
    try:
        fyk = int(float(vals.get(kd("fyk"), 500)))
    except Exception:
        fyk = 500
    if fyk not in E.NUANCES:
        remarques.append(f"{kd('fyk')} : {vals.get(kd('fyk'))!r} lue 500 (comme l'écran)")
        fyk = 500
    dp = str(vals.get(kd("dir_principale"), "Y")).upper()
    p = {"nom": nom, "b": vals.get(kd("b"), E.DEFAUTS_PREDALLE["b"]), "h": vals.get(kd("h"), E.DEFAUTS_PREDALLE["h"]),
         "beton": beton, "fyk": fyk, "enrobage": vals.get(kd("enrobage_beton"), E.DEFAUTS_PREDALLE["enrobage"]),
         "dir_principale": dp if dp in ("X", "Y") else "Y", "gamma_s": gs, "jeu_premier_lit": jeu, "sections": []}
    for k in ("b", "h"):
        if isinstance(p[k], float) and p[k].is_integer():
            p[k] = int(p[k])
    # la peau : 6 cm par défaut (« or 6.0 »), RAMENÉE en silence dans [3 ; min(15 ; h − 1)] (audit I-5)
    try:
        h_tot = float(vals.get(kd("h"), 22) or 22)
        hp = float(vals.get(kd("h_pre"), 6.0) or 6.0)
        if kd("h_pre") in vals and not vals[kd("h_pre")]:
            remarques.append(f"{kd('h_pre')} : {vals[kd('h_pre')]!r} calculée 6 cm (« or 6.0 » de l'écran, qui garde 0 dans son "
                             "état et refuse de l'afficher)")
    except (TypeError, ValueError):
        raise ValueError(f"prédalle « {nom} » : épaisseur ou peau illisible") from None
    hp_max = E.h_pre_max(h_tot)
    if not (E.H_PRE_MIN <= hp <= hp_max):
        ramene = min(max(hp, E.H_PRE_MIN), hp_max)
        remarques.append(f"{kd('h_pre')} : peau de {hp:g} cm ramenée à {ramene:g} cm (bornes de l'écran pour h = {h_tot:g} cm)")
        hp = ramene
    p["h_pre"] = hp
    for s in d["sections"]:
        sid = s["id"]

        def ks(base):
            return f"pre{did}_sec{sid}_{base}"
        sec = {"nom": str(vals.get(f"meta_pre{did}_nom_{sid}", s["nom"]))}
        for k in E.MOMENTS + ("V",):
            x = _nombre(vals.get(ks(k), 0.0) or 0.0)
            if x is None or x < 0:
                remarques.append(f"{ks(k)} : {vals.get(ks(k))!r} lu 0 (comme l'écran)")
                x = 0.0
            sec[k] = x
        sec["faces"] = {}
        for which in E.FACES:
            nc = _entier_borne(vals.get(ks(f"ncouches_{which}"), 1), 1, 1, E.MAX_COUCHES)
            couches = []
            for i in range(1, nc + 1):
                t = vals.get(ks(f"arm_type_{which}_c{i}"), "Treillis" if i == 1 else "n barres")
                if t not in TYPES_ECRAN:
                    remarques.append(f"{ks(f'arm_type_{which}_c{i}')} : {t!r} lu « {'Treillis' if i == 1 else 'n barres'} »")
                    t = "Treillis" if i == 1 else "n barres"
                try:
                    dd = int(float(vals.get(ks(f"ø_barres_{which}_c{i}"), 12)))
                except Exception:
                    dd = 12
                if dd not in E.DIAMETRES:
                    remarques.append(f"{ks(f'ø_barres_{which}_c{i}')} : Ø{dd} lu Ø12 (comme l'écran)")
                    dd = 12
                des = str(vals.get(ks(f"treillis_{which}_c{i}"), TR.TREILLIS_DEFAUT))
                couches.append({"type": TYPES_ECRAN[t], "treillis": des, "d": dd,
                                "esp": _entier_borne(vals.get(ks(f"esp_barres_{which}_c{i}"), 150), 150, E.ESP_MIN, E.ESP_MAX),
                                "n": _entier_borne(vals.get(ks(f"n_barres_{which}_c{i}"), 3), 3, E.N_MIN, E.N_MAX),
                                "dist": None})
            sec["faces"][which] = {"couches": couches, "yG": None}
        p["sections"].append(sec)
    # contrôle du domaine (largeur, épaisseur, treillis…) AVANT les champs automatiques, qui en dépendent
    try:
        p = E.normaliser(p)
    except ValueError as e:
        raise ValueError(f"prédalle « {nom} » : {e}") from None
    # --- _sync_dist_state puis _sync_ycdg_state : l'axe de chaque couche (règle prédalle), puis le centre de gravité
    for s_out, s in zip(p["sections"], d["sections"]):
        sid = s["id"]

        def ks(base):
            return f"pre{did}_sec{sid}_{base}"
        for which in E.FACES:
            face = s_out["faces"][which]
            for i, c in enumerate(face["couches"], start=1):
                auto = M.distance_auto(c, i - 1, which, p["dir_principale"], p["enrobage"], p["jeu_premier_lit"], p["h_pre"])
                c["dist"] = _champ_auto(vals, ks(f"dist_axe_{which}_c{i}"), ks(f"dist_auto_{which}_c{i}"),
                                        ks(f"dist_lastauto_{which}_c{i}"), auto, remarques)
            # le CDG automatique : Σ As·e / Σ As sur les axes effectifs (_layers_geometry, use_manual=False)
            axes = [c["dist"] if c["dist"] is not None else
                    M.distance_auto(c, j, which, p["dir_principale"], p["enrobage"], p["jeu_premier_lit"], p["h_pre"])
                    for j, c in enumerate(face["couches"])]
            face["yG"] = _champ_auto(vals, ks(f"ycdg_{which}"), ks(f"ycdg_auto_{which}"), ks(f"ycdg_lastauto_{which}"),
                                     yG_calcule(face["couches"], axes, p["b"]), remarques)
    return E.normaliser(p)


def yG_calcule(couches: list, axes: list, b: float) -> float:
    """Σ As·e / Σ As des couches d'une face, dans l'ordre d'opérations de _layers_geometry (axe de la base si Σ As = 0)."""
    tot, somme = 0.0, 0.0
    for c, e in zip(couches, axes):
        a = MD.couche_as_par_m(c, b)
        tot += a
        somme += a * e
    return (somme / tot) if tot > 0 else axes[0]


def _champ_auto(vals: dict, cle: str, cle_auto: str, cle_dernier: str, auto: float, remarques: list):
    """Un champ « automatique tant que rien n'est saisi » de l'écran (_sync_dist_state, _sync_ycdg_state, puis
    _dist_couche_eff / _ycdg_manual) → la valeur imposée, ou None (automatique)."""
    auto_txt = f"{auto:.1f}".replace(".", ",")
    texte = vals.get(cle, auto_txt)
    drapeau = bool(vals.get(cle_auto, True)) if cle in vals else True
    dernier = str(vals.get(cle_dernier, auto_txt) or "")
    brut = str(texte or "").strip()
    if brut == "":
        drapeau = True
    elif drapeau and brut not in (dernier, auto_txt):
        drapeau = False
    if drapeau:
        return None
    x = _nombre(brut)
    if x is None or x <= 0:
        remarques.append(f"{cle} : {brut!r} illisible ou nul, lu automatique ({auto_txt} cm, comme l'écran)")
        return None
    return x

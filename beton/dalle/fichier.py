# -*- coding: utf-8 -*-
"""Dalle en béton armé — la lecture d'un FICHIER de l'écran Streamlit (« 💾 Enregistrer » : ``dalle_ba.json``,
format ``{"version": "dalle-2.0", "dalles": [...], "values": {...}}``), sans Streamlit.

La lecture refait ce que l'écran fait à l'ouverture (``_load_from_payload``) puis au run suivant, dans le même ordre :
paramètres globaux (``_ensure_global_defaults``), dalle 1 et section 1 toujours présentes (``_init_dalles_if_needed``),
MIGRATION v1 → v2 (``_migrate_section_v2`` : M_inf → Mx_inf, couches inf/sup → direction X), défauts et coercitions de
chaque couche (``_ensure_defaults_for_dalle``), textes « 28,00 » des sollicitations (``_sync_float_raw_keys``), puis les
champs « Dist. axe » et « CDG » (``_sync_dist_state``, ``_sync_ycdg_state`` : automatique tant que rien n'est saisi).
``tests/test_dalle_moteur.py`` le prouve en relisant les fichiers que l'écran lui-même écrit.

Ce que l'écran remplace sans rien dire est relevé dans ``remarques`` (moment négatif lu 0, axe ou centre de gravité
illisibles lus « automatique », type ou Ø inconnus ramenés au défaut…) ; ce que le moteur ne peut pas représenter
(treillis illisible, largeur hors des bornes de l'écran…) lève ValueError avec le nom de la clé.
"""
import math

from beton import treillis as TR

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


def _migrer(vals: dict, did: int, sid: int) -> None:
    """_migrate_section_v2 : l'unidirectionnel devient la direction X (une valeur v1 n'est copiée que si la clé v2
    n'existe pas) ; les clés v1 sont retirées."""
    def ks(base):
        return f"dal{did}_sec{sid}_{base}"

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
    """Le contenu d'un fichier de l'écran → ``{"dalles": [dalle normalisée, …], "gamma_s", "jeu_premier_lit",
    "remarques": [texte, …]}`` (chaque dalle porte γs et le jeu, paramètres globaux de l'écran)."""
    if not isinstance(payload, dict):
        raise ValueError("fichier : un objet JSON {\"dalles\": …, \"values\": …} attendu")
    remarques = []
    values = payload.get("values", {})
    vals = dict(values) if isinstance(values, dict) else {}
    # --- _load_from_payload : la liste des dalles et de leurs sections (autre chose qu'une liste : la dalle 1 neuve)
    dalles = []
    liste = payload.get("dalles")
    for d in liste if isinstance(liste, list) else []:
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
        dalles.append({"id": did, "nom": str(d.get("nom", f"Dalle {did}")), "sections": propres})
    if not dalles:
        dalles = [{"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]}]
    # --- _ensure_global_defaults
    jeu = _nombre(vals.get("jeu_enrobage_cm", 0.0))
    try:
        gs = float(vals.get("gamma_s", 1.5) or 1.5)
    except Exception:
        gs = 1.5
    if gs <= 0:
        gs = 1.5
    # --- _init_dalles_if_needed : dalle 1 et section 1 toujours présentes
    for d in dalles:
        if not d["sections"]:
            d["sections"] = [{"id": 1, "nom": "A"}]
    if not any(d["id"] == 1 for d in dalles):
        dalles.insert(0, {"id": 1, "nom": "Dalle 1", "sections": [{"id": 1, "nom": "A"}]})
    for d in dalles:
        if not any(s["id"] == 1 for s in d["sections"]):
            d["sections"].insert(0, {"id": 1, "nom": "A"})
    # --- _sync_float_raw_keys : le texte saisi l'emporte, un négatif devient 0
    for k in list(vals):
        if k.endswith("_raw"):
            x = _nombre(vals[k])
            if x is not None:
                if x < 0:
                    remarques.append(f"{k[:-4]} : {vals[k]!r} lu 0 (l'écran ramène un négatif à 0)")
                vals[k[:-4]] = max(0.0, x)
    if jeu is None or jeu < 0:
        raise ValueError(f"jeu_enrobage_cm : nombre ≥ 0 attendu (reçu {vals.get('jeu_enrobage_cm')!r})")
    out = []
    for d in dalles:
        out.append(_dalle(vals, d, gs, jeu, remarques))
    return {"dalles": out, "gamma_s": out[0]["gamma_s"], "jeu_premier_lit": out[0]["jeu_premier_lit"], "remarques": remarques}


def _entier_borne(v, defaut: int, mini: int, maxi: int) -> int:
    try:
        x = int(float(v or defaut))
    except Exception:
        x = defaut
    return max(mini, min(maxi, x))


def _dalle(vals: dict, d: dict, gs: float, jeu: float, remarques: list) -> dict:
    did = d["id"]

    def kd(base):
        return f"dal{did}_{base}"
    nom = str(vals.get(f"meta_dalle_nom_{did}", d["nom"]))
    beton = vals.get(kd("beton"), E.BETON_DEFAUT)
    if beton not in E._classes_lues():
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
    dalle = {"nom": nom, "b": vals.get(kd("b"), E.DEFAUTS_DALLE["b"]), "h": vals.get(kd("h"), E.DEFAUTS_DALLE["h"]),
             "beton": beton, "fyk": fyk, "enrobage": vals.get(kd("enrobage_beton"), E.DEFAUTS_DALLE["enrobage"]),
             "dir_principale": dp if dp in ("X", "Y") else "Y", "gamma_s": gs, "jeu_premier_lit": jeu, "sections": []}
    for k in ("b", "h", "enrobage"):
        if isinstance(dalle[k], float) and dalle[k].is_integer() and k != "enrobage":
            dalle[k] = int(dalle[k])
    for s in d["sections"]:
        sid = s["id"]
        _migrer(vals, did, sid)

        def ks(base):
            return f"dal{did}_sec{sid}_{base}"
        sec = {"nom": str(vals.get(f"meta_dal{did}_nom_{sid}", s["nom"]))}
        for k in E.MOMENTS + ("V",):
            x = _nombre(vals.get(ks(k), 0.0) or 0.0)
            if x is None or x < 0:
                # au premier rendu, le champ texte écrit la valeur, et le run suivant la relit bornée à 0
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
        dalle["sections"].append(sec)
    # contrôle du domaine (largeur, épaisseur, treillis…) AVANT les champs automatiques, qui en dépendent
    try:
        dalle = E.normaliser(dalle)
    except ValueError as e:
        raise ValueError(f"dalle « {nom} » : {e}") from None
    # --- _sync_dist_state puis _sync_ycdg_state : l'axe de chaque couche, puis le centre de gravité
    for s_out, s in zip(dalle["sections"], d["sections"]):
        sid = s["id"]

        def ks(base):
            return f"dal{did}_sec{sid}_{base}"
        for which in E.FACES:
            face = s_out["faces"][which]
            for i, c in enumerate(face["couches"], start=1):
                auto = M.distance_auto(c, dalle["enrobage"], dalle["jeu_premier_lit"])
                c["dist"] = _champ_auto(vals, ks(f"dist_axe_{which}_c{i}"), ks(f"dist_auto_{which}_c{i}"),
                                        ks(f"dist_lastauto_{which}_c{i}"), auto, remarques)
            geo = M._face(face["couches"], None, dalle["b"], dalle["enrobage"], dalle["jeu_premier_lit"])
            face["yG"] = _champ_auto(vals, ks(f"ycdg_{which}"), ks(f"ycdg_auto_{which}"), ks(f"ycdg_lastauto_{which}"),
                                     geo["yG_calcule"], remarques)
    return E.normaliser(dalle)


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

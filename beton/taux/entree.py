# -*- coding: utf-8 -*-
"""Taux d'armature — l'ENTRÉE : les listes, les libellés, les bornes et les valeurs par défaut de l'écran Streamlit
``modules/taux_armature.py`` (la référence de parité), et le contrôle d'une poutre.

Le format complet (entrée et sortie) est décrit dans la docstring de ``beton/taux/moteur.py``.

Règle de ``normaliser`` : une clé ABSENTE (ou ``null``) prend la valeur par défaut de l'écran ; une valeur HORS DU
DOMAINE de l'écran (nombre hors des bornes de son champ, Ø non proposé, position que la forme n'offre pas, clé
inconnue…) lève ``ValueError`` avec le chemin du champ. Jamais de substitution silencieuse. Les substitutions que
l'écran fait, lui, sont listées dans ``docs/beton/TAUX_ARMATURE_MOTEUR.md`` ; trois sont REFUSÉES ici parce qu'elles
changent la valeur saisie sans le dire (talon de 0 cm lu 10 ou 15 cm ; deux types d'étriers du même nom ; 2·c ≥ b ou
2·c ≥ h, où la longueur d'un étrier devient négative).
"""
import copy
import math

REFERENCE = "modules/taux_armature.py"

# ---- les formes de section : la clé du site → le libellé EXACT de l'écran (le calcul lit des sous-chaînes de ce
#      libellé, comme le module : « gauche », « droit », « deux talons » — voir geometrie.talons)
FORMES = {
    "rectangulaire": "Rectangulaire",
    "talon_gauche": "Rectangulaire + talon gauche",
    "talon_droit": "Rectangulaire + talon droit",
    "deux_talons": "Rectangulaire + deux talons",
}

# ---- les positions d'une famille : la clé du site → (code du module, libellé de l'écran : POSITION_LABELS)
POSITIONS = {
    "ame_inf": ("web_bottom", "Armatures inférieures (âme)"),
    "ame_sup": ("web_top", "Armatures supérieures (âme)"),
    "talon_g_inf": ("flange_left_bottom", "Talon gauche – inférieure"),
    "talon_g_sup": ("flange_left_top", "Talon gauche – supérieure"),
    "talon_d_inf": ("flange_right_bottom", "Talon droit – inférieure"),
    "talon_d_sup": ("flange_right_top", "Talon droit – supérieure"),
    "lat_g": ("side_left", "Latérales gauche"),
    "lat_d": ("side_right", "Latérales droite"),
    "lat_2": ("side_both", "Latérales 2 côtés"),
}
MODULE_POSITION = {k: v[0] for k, v in POSITIONS.items()}
POSITION_MODULE = {v[0]: k for k, v in POSITIONS.items()}

# ---- les rôles : la clé du site → (code du module, libellé de l'écran : ROLE_LABELS)
ROLES = {
    "principal": ("main", "Lit principal"),
    "renfort": ("reinforcement", "Renfort local"),
    "lateral": ("lateral", "Barres latérales"),
}
MODULE_ROLE = {k: v[0] for k, v in ROLES.items()}
ROLE_MODULE = {v[0]: k for k, v in ROLES.items()}

DIAMETRES = (6, 8, 10, 12, 14, 16, 20, 25, 32, 40)   # « Ø (mm) » d'une famille
RETOURS = (0, 1, 2)                                   # « Retours »
ACIERS = ("B500A", "B500B", "B500C")                  # « Qualité d'acier » : n'entre dans AUCUN calcul du module
DIAMETRES_ETRIERS = (6, 8, 10, 12, 14, 16)            # le menu du site ; l'écran : un nombre de 6 à 16 (pas de 2)
LONGUEURS = ("pourcent", "metre")                     # « Saisie des longueurs de lit » : en % de la portée, en mètres

# ---- les bornes des champs de l'écran (min_value, max_value des st.number_input)
BORNES = {
    "enrobage": (1.0, 10.0),     # « Enrobage c (cm) »
    "L": (0.5, 60.0),            # « Portée L (m) »
    "b": (5.0, 200.0),           # « Largeur b (cm) »
    "h": (10.0, 300.0),          # « Hauteur h (cm) »
    "talon": (0.0, 300.0),       # « Talon … – largeur / hauteur (cm) » (0 refusé ici : le module le lit 10 ou 15 cm)
    "n": (1, 20),                # « n barres »
    "h_retour": (0.0, 1000.0),   # « Hauteur retour (cm) »
    "pct": (0.0, 100.0),         # « De (%L) », « À (%L) » (familles et zones)
    "d_etrier": (6.0, 16.0),     # « Ø (mm) » d'un type d'étrier
    "s": (2.0, 100.0),           # « Espacement s (cm) »
    "majoration": (0.0, 50.0),   # « Majoration globale (%) »
}

# ---- les masses volumiques du module
MASSE_VOLUMIQUE_ACIER = 7850.0   # STEEL_DENSITY_KG_M3 (kg/m³)
MASSE_VOLUMIQUE_BETON = 2500.0   # CONCRETE_DENSITY_KG_M3 (kg/m³)

# ---- les valeurs par défaut (_init_state, add_family, add_stirrup_type, add_stirrup_zone)
TALON_DEFAUT = {"b": 10.0, "h": 15.0}   # build_geometry : « flange_left_b_cm or 10.0 », « … _h_cm or 15.0 »
DEFAUTS_POUTRE = {
    "nom": "P1",                 # le site seulement (l'écran ne nomme pas sa poutre)
    "forme": "rectangulaire",    # shape_section « Rectangulaire »
    "enrobage": 3.0,             # cover_cm
    "L": 6.0,                    # L_beam_m (m)
    "b": 30.0,                   # b_web_cm
    "h": 60.0,                   # h_total_cm
    "longueurs": "pourcent",     # range_mode « percent »
    "acier": "B500A",            # steel_grade
    "majoration": 5.0,           # « Majoration globale (%) » : value=5.0, sans clé
}
FAMILLE_DEFAUT = {"n": 2, "d": 16, "retours": 0, "h_retour": 0.0, "de": 0.0, "a": 100.0}
TYPE_DEFAUT = {"d": 8.0}                                   # add_stirrup_type : phi_mm 8.0, type « full »
ZONE_DEFAUT = {"de": 0.0, "a": 33.0, "s": 15.0}            # add_stirrup_zone

CLES_POUTRE = frozenset(DEFAUTS_POUTRE) | {"talon_gauche", "talon_droit", "familles", "types_etriers", "zones"}
CLES_TALON = frozenset({"b", "h"})
CLES_FAMILLE = frozenset({"role", "position"}) | frozenset(FAMILLE_DEFAUT)
CLES_TYPE = frozenset({"nom", "d"})
CLES_ZONE = frozenset({"nom", "de", "a", "s", "type"})


# ============================================================ les options de l'écran
def a_talon_gauche(forme: str) -> bool:
    """Les positions « talon gauche » sont OFFERTES (_position_options_for_family : « gauche » ou « deux talons »)."""
    lib = FORMES[forme]
    return "gauche" in lib or "deux talons" in lib


def a_talon_droit(forme: str) -> bool:
    lib = FORMES[forme]
    return "droit" in lib or "deux talons" in lib


def options_positions(forme: str, role: str) -> list:
    """Les positions qu'offre le menu « Position » d'une famille (_position_options_for_family), dans l'ordre."""
    if role == "lateral":
        return ["lat_g", "lat_d", "lat_2"]
    opts = ["ame_inf", "ame_sup"]
    if a_talon_gauche(forme):
        opts += ["talon_g_inf", "talon_g_sup"]
    if a_talon_droit(forme):
        opts += ["talon_d_inf", "talon_d_sup"]
    return opts


def renfort_talon_possible(forme: str) -> bool:
    """Le bouton « + Renfort talon » est actif (has_flange)."""
    lib = FORMES[forme]
    return "gauche" in lib or "droit" in lib or "deux talons" in lib


# ============================================================ éléments neufs
def nouvelle_poutre() -> dict:
    """La poutre de l'écran au premier affichage : rectangulaire 30 × 60 cm, portée 6 m, enrobage 3 cm, aucune
    armature, aucun étrier, B500A, majoration 5 %. Les talons valent 10 × 15 cm : ce que l'écran CALCULE pour une
    forme à talon (il affiche 0 dans le champ, mais lit « 0 or 10.0 »)."""
    p = copy.deepcopy(DEFAUTS_POUTRE)
    p["talon_gauche"] = dict(TALON_DEFAUT)
    p["talon_droit"] = dict(TALON_DEFAUT)
    p["familles"], p["types_etriers"], p["zones"] = [], [], []
    return p


def nouvelle_famille(poutre: dict, role: str, position: str = None) -> dict:
    """Une famille comme add_family la crée : 2Ø16 de 0 à 100 % sans retour. La position par défaut : âme en bas pour
    un lit principal, âme en haut pour un renfort, latérales à gauche. La hauteur de retour que add_family propose
    (h − 2c, ou b − 2c pour une latérale) est remise à 0 par edit_family dès l'affichage (« Retours » vaut 0) : c'est 0
    que le module garde, et c'est 0 qu'on garde (docs, défaut D5)."""
    if role not in ROLES:
        raise ValueError(f"rôle inconnu : {role!r}")
    if role == "lateral":
        pos = position or "lat_g"
    else:
        pos = position or ("ame_inf" if role == "principal" else "ame_sup")
    f = {"role": role, "position": pos}
    f.update(FAMILLE_DEFAUT)
    return f


def position_renfort_talon(forme: str) -> str:
    """La position d'un « + Renfort talon » : talon gauche inférieur si la forme a « gauche » ou « deux talons »,
    sinon talon droit inférieur."""
    lib = FORMES[forme]
    return "talon_g_inf" if ("gauche" in lib or "deux talons" in lib) else "talon_d_inf"


def _nom_libre(prefixe: str, pris) -> str:
    k = len(pris) + 1
    while f"{prefixe}{k}" in pris:
        k += 1
    return f"{prefixe}{k}"


def nouveau_type_etrier(poutre: dict) -> dict:
    """Un type d'étrier comme add_stirrup_type : « E<k> », Ø8."""
    return {"nom": _nom_libre("E", [t["nom"] for t in poutre.get("types_etriers") or []]), "d": TYPE_DEFAUT["d"]}


def nouvelle_zone(poutre: dict) -> dict:
    """Une zone comme add_stirrup_zone : « Z<k> », de 0 à 33 %, s = 15 cm, le premier type d'étrier."""
    types = poutre.get("types_etriers") or []
    z = {"nom": _nom_libre("Z", [x["nom"] for x in poutre.get("zones") or []])}
    z.update(ZONE_DEFAUT)
    z["type"] = types[0]["nom"] if types else ""
    return z


# ============================================================ contrôle et normalisation
def _cles(d, permises: frozenset, chemin: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{chemin} : un objet est attendu, pas {type(d).__name__}")
    inconnues = sorted(str(k) for k in d if k not in permises)
    if inconnues:
        raise ValueError(f"{chemin} : clé(s) inconnue(s) {', '.join(inconnues)} — clés admises : {', '.join(sorted(permises))}")


def _val(d: dict, cle: str, defaut):
    v = d.get(cle)
    return defaut if v is None else v


def _nombre(x, chemin: str, bornes, strict_min: bool = False) -> float:
    """Un nombre fini dans les bornes du champ de l'écran ; ``strict_min`` : la borne basse est exclue."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{chemin} : nombre attendu, reçu {x!r}")
    lo, hi = bornes
    if x < lo or (strict_min and x <= lo) or x > hi:
        bas = f"> {_t(lo)}" if strict_min else f"≥ {_t(lo)}"
        raise ValueError(f"{chemin} : {_t(x)} hors du domaine de l'écran ({bas} et ≤ {_t(hi)})")
    return float(x)


def _entier(x, chemin: str, bornes) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x != int(x):
        raise ValueError(f"{chemin} : nombre entier attendu, reçu {x!r}")
    lo, hi = bornes
    if not lo <= x <= hi:
        raise ValueError(f"{chemin} : {int(x)} hors du domaine de l'écran ({lo} à {hi})")
    return int(x)


def _t(x) -> str:
    return f"{x:g}".replace(".", ",")


def _texte(x, chemin: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"{chemin} : texte attendu, reçu {x!r}")
    return x


def normaliser(poutre: dict) -> dict:
    """La poutre complétée (clés absentes → défauts de l'écran) et contrôlée (ValueError hors du domaine de l'écran,
    avec le chemin du champ). Rend une COPIE ; idempotente."""
    _cles(poutre, CLES_POUTRE, "poutre")
    d = DEFAUTS_POUTRE
    p = {"nom": _texte(_val(poutre, "nom", d["nom"]), "nom")}
    p["forme"] = _val(poutre, "forme", d["forme"])
    if p["forme"] not in FORMES:
        raise ValueError(f"forme : {p['forme']!r} inconnue — formes : {', '.join(FORMES)}")
    for k in ("enrobage", "L", "b", "h", "majoration"):
        p[k] = _nombre(_val(poutre, k, d[k]), k, BORNES[k])
    if not 2 * p["enrobage"] < p["b"] or not 2 * p["enrobage"] < p["h"]:
        raise ValueError(f"enrobage : 2·c = {_t(2 * p['enrobage'])} cm ne laisse pas de place dans {_t(p['b'])} × {_t(p['h'])} cm "
                         "(le module calculerait des étriers de longueur négative)")
    for cote in ("talon_gauche", "talon_droit"):
        t = _val(poutre, cote, TALON_DEFAUT)
        _cles(t, CLES_TALON, cote)
        p[cote] = {k: _nombre(_val(t, k, TALON_DEFAUT[k]), f"{cote}.{k}", BORNES["talon"], strict_min=True) for k in ("b", "h")}
    p["longueurs"] = _val(poutre, "longueurs", d["longueurs"])
    if p["longueurs"] not in LONGUEURS:
        raise ValueError(f"longueurs : {p['longueurs']!r} — « pourcent » ou « metre »")
    p["acier"] = _val(poutre, "acier", d["acier"])
    if p["acier"] not in ACIERS:
        raise ValueError(f"acier : {p['acier']!r} — {', '.join(ACIERS)}")
    p["familles"] = [_famille(f, p["forme"], f"familles[{i}]") for i, f in enumerate(_liste(poutre, "familles"))]
    p["types_etriers"] = [_type(t, f"types_etriers[{i}]") for i, t in enumerate(_liste(poutre, "types_etriers"))]
    noms = [t["nom"] for t in p["types_etriers"]]
    for i, n in enumerate(noms):
        if not n.strip():
            raise ValueError(f"types_etriers[{i}].nom : un type d'étrier a un nom (les zones le désignent par son nom)")
        if n in noms[:i]:
            raise ValueError(f"types_etriers[{i}].nom : « {n} » est déjà pris (le module retiendrait le DERNIER type de ce nom "
                             "pour le calcul, et le premier dans le menu des zones)")
    p["zones"] = [_zone(z, noms, f"zones[{i}]") for i, z in enumerate(_liste(poutre, "zones"))]
    return p


def _liste(poutre: dict, cle: str) -> list:
    v = _val(poutre, cle, [])
    if not isinstance(v, list):
        raise ValueError(f"{cle} : une liste est attendue")
    return v


def _famille(f: dict, forme: str, chemin: str) -> dict:
    _cles(f, CLES_FAMILLE, chemin)
    role = _val(f, "role", "principal")
    if role not in ROLES:
        raise ValueError(f"{chemin}.role : {role!r} — {', '.join(ROLES)}")
    pos = _val(f, "position", "lat_g" if role == "lateral" else ("ame_inf" if role == "principal" else "ame_sup"))
    offertes = options_positions(forme, role)
    if pos not in offertes:
        raise ValueError(f"{chemin}.position : {pos!r} n'est pas offerte pour un « {ROLES[role][1]} » d'une section "
                         f"« {FORMES[forme]} » ({', '.join(offertes)})")
    out = {"role": role, "position": pos,
           "n": _entier(_val(f, "n", FAMILLE_DEFAUT["n"]), f"{chemin}.n", BORNES["n"])}
    dd = _val(f, "d", FAMILLE_DEFAUT["d"])
    if isinstance(dd, bool) or dd not in DIAMETRES:
        raise ValueError(f"{chemin}.d : Ø{dd!r} non proposé par l'écran ({', '.join(map(str, DIAMETRES))})")
    out["d"] = int(dd)
    r = _val(f, "retours", 0)
    if isinstance(r, bool) or r not in RETOURS:
        raise ValueError(f"{chemin}.retours : {r!r} — 0, 1 ou 2")
    out["retours"] = int(r)
    hr = _nombre(_val(f, "h_retour", 0.0), f"{chemin}.h_retour", BORNES["h_retour"])
    # edit_family : « Retours » = 0 remet la hauteur de retour à 0 (st.text « — »)
    out["h_retour"] = hr if out["retours"] else 0.0
    out["de"] = _nombre(_val(f, "de", 0.0), f"{chemin}.de", BORNES["pct"])
    out["a"] = _nombre(_val(f, "a", 100.0), f"{chemin}.a", BORNES["pct"])
    return out


def _type(t: dict, chemin: str) -> dict:
    _cles(t, CLES_TYPE, chemin)
    return {"nom": _texte(_val(t, "nom", ""), f"{chemin}.nom"),
            "d": _nombre(_val(t, "d", TYPE_DEFAUT["d"]), f"{chemin}.d", BORNES["d_etrier"])}


def _zone(z: dict, noms_types: list, chemin: str) -> dict:
    _cles(z, CLES_ZONE, chemin)
    out = {"nom": _texte(_val(z, "nom", ""), f"{chemin}.nom"),
           "de": _nombre(_val(z, "de", ZONE_DEFAUT["de"]), f"{chemin}.de", BORNES["pct"]),
           "a": _nombre(_val(z, "a", ZONE_DEFAUT["a"]), f"{chemin}.a", BORNES["pct"]),
           "s": _nombre(_val(z, "s", ZONE_DEFAUT["s"]), f"{chemin}.s", BORNES["s"])}
    typ = _texte(_val(z, "type", noms_types[0] if noms_types else ""), f"{chemin}.type")
    if noms_types and typ not in noms_types:
        raise ValueError(f"{chemin}.type : « {typ} » n'est pas un type d'étrier de la poutre ({', '.join(noms_types)})")
    out["type"] = typ
    return out


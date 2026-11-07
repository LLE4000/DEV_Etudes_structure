# -*- coding: utf-8 -*-
"""Dalle en béton armé — l'ENTRÉE : les listes, les bornes et les valeurs par défaut de l'écran Streamlit
(``modules/dalle.py`` v2.2, la référence de parité), les classes de béton (``beton_classes.json``, source unique, à la
racine du dépôt) et le contrôle d'une dalle.

Le format complet (entrée et sortie) est décrit dans la docstring de ``beton/dalle/moteur.py``.

Règle de ``normaliser`` : une clé ABSENTE (ou ``null``) prend la valeur par défaut de l'écran ; une valeur HORS DU
DOMAINE de l'écran (Ø non proposé, nuance autre que 400 / 500, classe inconnue, largeur hors de 20 à 500 cm, moment
négatif, désignation de treillis illisible, clé inconnue…) lève ``ValueError`` avec le chemin du champ. Jamais de
substitution silencieuse : l'écran, lui, en fait plusieurs (moment négatif lu 0, distance d'axe ou centre de gravité
illisibles ou nuls lus « automatique », treillis illisible lu 0 mm²/m…), listées dans ``docs/beton/DALLE_MOTEUR.md``.
La seule substitution reproduite est celle que l'écran applique à une saisie VALIDE : un enrobage nul est lu 3,0 cm.
"""
import copy
import functools
import json
import math
import os
from string import ascii_uppercase

from beton import treillis as TR

CHEMIN_CLASSES = os.path.join(os.path.dirname(__file__), "..", "..", "beton_classes.json")

# ---- les listes de l'écran (modules/dalle.py) ----
DIAMETRES = (6, 8, 10, 12, 16, 20, 25, 32, 40)      # DIAM_OPTS : Ø des barres (mm)
MAX_COUCHES = 4                                     # base + 3 renforts par face et par direction
NUANCES = (400, 500)                                # « Qualité acier (B) »
DIRECTIONS = ("x", "y")                             # DIR_KEYS
FACES = ("inf_x", "sup_x", "inf_y", "sup_y")        # FACES_DIR : face × direction (le « which » de l'écran)
# les types de couche : code du moteur ↔ libellé de l'écran (TYPES_ARMATURE)
TYPES = ("treillis", "barres", "n_barres")
LIBELLES_TYPES = {"treillis": "Treillis", "barres": "Barres", "n_barres": "n barres"}
_TYPES_LUS = {"treillis": "treillis", "barres": "barres", "n_barres": "n_barres", "n barres": "n_barres"}

# ---- les bornes des champs de l'écran (number_input : min_value / max_value) ----
B_MIN, B_MAX = 20, 500            # « Larg. bande (cm) », entier
H_MIN, H_MAX = 5, 100             # « Épaisseur (cm) », entier
ENROBAGE_MAX = 20.0               # « Enrob. béton (cm) », 0 à 20
GAMMA_S_MIN, GAMMA_S_MAX = 1.0, 2.0   # « Coefficient acier ELS »
ESP_MIN, ESP_MAX = 25, 500        # « Espacement (mm) », entier
N_MIN, N_MAX = 1, 50              # « Nombre de barres », entier

# ---- les valeurs par défaut (_ensure_global_defaults, _init_dalles_if_needed, _ensure_defaults_for_dalle) ----
BETON_DEFAUT = "C30/37"
DEFAUTS_DALLE = {
    "nom": "Dalle 1",
    "b": 100,                 # cm : la bande d'un mètre courant
    "h": 20,                  # cm
    "beton": BETON_DEFAUT,
    "fyk": 500,               # N/mm²
    "enrobage": 3.0,          # cm
    "dir_principale": "Y",    # « Dir. principale » : Y par défaut (convention du bureau)
    "gamma_s": 1.5,           # « Coefficient acier ELS » (paramètre global de l'écran, partagé avec Poutre)
    "jeu_premier_lit": 0.0,   # « Jeu premier lit (cm) » (paramètre global, 0 par défaut : retour bureau 01/09)
}
COUCHE_BASE = {"type": "treillis", "treillis": TR.TREILLIS_DEFAUT, "d": 12, "esp": 150, "n": 3, "dist": None}
COUCHE_RENFORT = {"type": "n_barres", "treillis": TR.TREILLIS_DEFAUT, "d": 12, "esp": 150, "n": 3, "dist": None}

CLES_DALLE = frozenset(DEFAUTS_DALLE) | {"sections"}
CLES_SECTION = frozenset({"nom", "Mx_inf", "Mx_sup", "My_inf", "My_sup", "V", "faces"})
MOMENTS = ("Mx_inf", "Mx_sup", "My_inf", "My_sup")
CLES_FACE = frozenset({"couches", "yG"})
CLES_COUCHE = frozenset(COUCHE_BASE)


# ============================================================ les classes de béton
@functools.lru_cache(maxsize=1)
def _classes_lues() -> dict:
    """Le fichier lu une fois (lecture seule : le moteur ne le modifie jamais)."""
    with open(CHEMIN_CLASSES, "r", encoding="utf-8") as f:
        return json.load(f)


def classes_beton() -> dict:
    """Les classes de béton de l'application, lues dans ``beton_classes.json`` (une copie)."""
    return copy.deepcopy(_classes_lues())


# ============================================================ dalles, sections, couches par défaut
def nouvelle_couche(base: bool = True) -> dict:
    """La couche 1 (base) : un treillis 10/10/100/100 ; un renfort ajouté (bouton « ＋ ») : 3 Ø12 (« n barres »)."""
    return dict(COUCHE_BASE if base else COUCHE_RENFORT)


def nouvelle_face() -> dict:
    """Une face d'une direction : une couche de base, centre de gravité calculé (yG = None)."""
    return {"couches": [nouvelle_couche()], "yG": None}


def nouvelle_section(nom: str = "A") -> dict:
    """Une section aux valeurs par défaut de l'écran : sans sollicitation, un treillis 10/10/100/100 sur chacune des
    quatre faces (inf. X, sup. X, inf. Y, sup. Y)."""
    return {"nom": nom, "Mx_inf": 0.0, "Mx_sup": 0.0, "My_inf": 0.0, "My_sup": 0.0, "V": 0.0,
            "faces": {f: nouvelle_face() for f in FACES}}


def nouvelle_dalle() -> dict:
    """Une dalle aux valeurs par défaut de l'écran : « Dalle 1 », bande de 100 cm, 20 cm, C30/37, B500, enrobage 3 cm,
    direction principale Y, γs 1,5, jeu 0, une section « A »."""
    d = copy.deepcopy(DEFAUTS_DALLE)
    d["sections"] = [nouvelle_section("A")]
    return d


def _lettres():
    yield from ascii_uppercase
    for a in ascii_uppercase:
        for b in ascii_uppercase:
            yield a + b


def nom_section_suivant(dalle: dict) -> str:
    """Le nom d'une section ajoutée ou copiée : la première lettre libre (A, B, …, Z, AA, AB…) — _next_section_name."""
    sections = dalle.get("sections") or []
    pris = {str(s.get("nom", "")).strip() for s in sections if isinstance(s, dict)}
    for lettre in _lettres():
        if lettre not in pris:
            return lettre
    return f"S{len(sections) + 1}"  # repli de l'écran, improbable


def libelle_section(nom: str) -> str:
    """Le titre d'une section comme l'écran l'écrit : « Section A » ; un nom qui commence déjà par « section » est
    gardé tel quel (« Section travée »)."""
    return nom if nom.lower().startswith("section") else f"Section {nom}"


# ============================================================ contrôle et normalisation
def _cles(d: dict, permises: frozenset, chemin: str) -> None:
    inconnues = sorted(str(k) for k in d if k not in permises)
    if inconnues:
        raise ValueError(f"{chemin} : clé(s) inconnue(s) {', '.join(inconnues)} — clés admises : {', '.join(sorted(permises))}")


def _val(d: dict, cle: str, defaut):
    v = d.get(cle)
    return defaut if v is None else v


def _nombre(x, chemin: str, minimum: float = 0.0, strict: bool = False, maximum=None) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{chemin} : nombre attendu (reçu {x!r})")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{chemin} : nombre fini attendu (reçu {x!r})")
    if (v <= minimum) if strict else (v < minimum):
        raise ValueError(f"{chemin} : nombre {'>' if strict else '≥'} {minimum:g} attendu (reçu {x!r})")
    if maximum is not None and v > maximum:
        raise ValueError(f"{chemin} : nombre ≤ {maximum:g} attendu (reçu {x!r})")
    return v


def _entier(x, chemin: str, permis=None, minimum=None, maximum=None) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, float) and not x.is_integer()):
        raise ValueError(f"{chemin} : entier attendu (reçu {x!r})")
    v = int(x)
    if permis is not None and v not in permis:
        raise ValueError(f"{chemin} : {v} n'est pas proposé par l'écran ({' '.join(map(str, permis))})")
    if minimum is not None and v < minimum:
        raise ValueError(f"{chemin} : entier ≥ {minimum} attendu (reçu {x!r})")
    if maximum is not None and v > maximum:
        raise ValueError(f"{chemin} : entier ≤ {maximum} attendu (reçu {x!r})")
    return v


def _texte(x, chemin: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"{chemin} : texte attendu (reçu {x!r})")
    return x


def _couche(c, chemin: str, base: bool) -> dict:
    if not isinstance(c, dict):
        raise ValueError(f"{chemin} : {{\"type\": …}} attendu (reçu {c!r})")
    _cles(c, CLES_COUCHE, chemin)
    defaut = COUCHE_BASE if base else COUCHE_RENFORT
    t = _val(c, "type", defaut["type"])
    if not isinstance(t, str) or t.strip().lower() not in _TYPES_LUS:
        raise ValueError(f"{chemin}.type : « treillis », « barres » ou « n_barres » attendu (reçu {t!r})")
    des = _texte(_val(c, "treillis", defaut["treillis"]), f"{chemin}.treillis")
    if not TR.est_canonique(des):
        raise ValueError(f"{chemin}.treillis : désignation « Øl/Øt/el/et » en entiers attendue (reçu {des!r})")
    out = {"type": _TYPES_LUS[t.strip().lower()], "treillis": des,
           "d": _entier(_val(c, "d", defaut["d"]), f"{chemin}.d", permis=DIAMETRES),
           "esp": _entier(_val(c, "esp", defaut["esp"]), f"{chemin}.esp", minimum=ESP_MIN, maximum=ESP_MAX),
           "n": _entier(_val(c, "n", defaut["n"]), f"{chemin}.n", minimum=N_MIN, maximum=N_MAX),
           "dist": None if c.get("dist") is None else _nombre(c["dist"], f"{chemin}.dist", strict=True)}
    return out


def _face(f, chemin: str) -> dict:
    if f is None:
        return nouvelle_face()
    if not isinstance(f, dict):
        raise ValueError(f"{chemin} : {{\"couches\": […], \"yG\": …}} attendu (reçu {f!r})")
    _cles(f, CLES_FACE, chemin)
    couches = f.get("couches")
    if couches is None:
        couches = [nouvelle_couche()]
    if not isinstance(couches, (list, tuple)) or not 1 <= len(couches) <= MAX_COUCHES:
        raise ValueError(f"{chemin}.couches : liste de 1 à {MAX_COUCHES} couche(s) attendue (reçu {couches!r})")
    return {"couches": [_couche(c, f"{chemin}.couches[{i}]", i == 0) for i, c in enumerate(couches)],
            "yG": None if f.get("yG") is None else _nombre(f["yG"], f"{chemin}.yG", strict=True)}


def _section(s, i: int, nom_defaut: str) -> dict:
    c = f"sections[{i}]"
    if not isinstance(s, dict):
        raise ValueError(f"{c} : dict attendu (reçu {s!r})")
    _cles(s, CLES_SECTION, c)
    out = {"nom": _texte(_val(s, "nom", nom_defaut), f"{c}.nom")}
    for k in MOMENTS + ("V",):
        out[k] = _nombre(_val(s, k, 0.0), f"{c}.{k}")
    faces = s.get("faces")
    if faces is None:
        faces = {}
    if not isinstance(faces, dict):
        raise ValueError(f"{c}.faces : {{\"inf_x\": …, \"sup_x\": …, \"inf_y\": …, \"sup_y\": …}} attendu (reçu {faces!r})")
    _cles(faces, frozenset(FACES), f"{c}.faces")
    out["faces"] = {f: _face(faces.get(f), f"{c}.faces.{f}") for f in FACES}
    return out


def normaliser(dalle: dict) -> dict:
    """Une copie COMPLÈTE et CONTRÔLÉE de la dalle : défauts de l'écran pour les clés absentes, ``ValueError`` (avec le
    chemin du champ) pour toute valeur hors du domaine de l'écran. ``normaliser(normaliser(d)) == normaliser(d)``."""
    if not isinstance(dalle, dict):
        raise ValueError(f"dalle : dict attendu (reçu {dalle!r})")
    _cles(dalle, CLES_DALLE, "dalle")
    D = DEFAUTS_DALLE
    d = {"nom": _texte(_val(dalle, "nom", D["nom"]), "nom"),
         "b": _entier(_val(dalle, "b", D["b"]), "b", minimum=B_MIN, maximum=B_MAX),
         "h": _entier(_val(dalle, "h", D["h"]), "h", minimum=H_MIN, maximum=H_MAX)}
    beton = _val(dalle, "beton", D["beton"])
    if beton not in _classes_lues():
        raise ValueError(f"beton : classe {beton!r} inconnue ({' '.join(_classes_lues())})")
    d["beton"] = beton
    d["fyk"] = _entier(_val(dalle, "fyk", D["fyk"]), "fyk", permis=NUANCES)
    d["enrobage"] = _nombre(_val(dalle, "enrobage", D["enrobage"]), "enrobage", maximum=ENROBAGE_MAX)
    dp = _val(dalle, "dir_principale", D["dir_principale"])
    if not isinstance(dp, str) or dp.strip().upper() not in ("X", "Y"):
        raise ValueError(f"dir_principale : « X » ou « Y » attendu (reçu {dp!r})")
    d["dir_principale"] = dp.strip().upper()
    d["gamma_s"] = _nombre(_val(dalle, "gamma_s", D["gamma_s"]), "gamma_s", minimum=GAMMA_S_MIN, maximum=GAMMA_S_MAX)
    d["jeu_premier_lit"] = _nombre(_val(dalle, "jeu_premier_lit", D["jeu_premier_lit"]), "jeu_premier_lit")
    sections = dalle.get("sections")
    if sections is None:
        d["sections"] = [nouvelle_section("A")]
        return d
    if not isinstance(sections, (list, tuple)) or not sections:
        raise ValueError(f"sections : liste d'au moins une section attendue (reçu {sections!r})")
    # une section sans nom prend la première lettre libre, comme une section ajoutée à l'écran
    noms = [s.get("nom") if isinstance(s, dict) else None for s in sections]
    pris = {"sections": [{"nom": n} for n in noms if isinstance(n, str)]}
    d["sections"] = []
    for i, s in enumerate(sections):
        defaut = "A"
        if noms[i] is None:
            defaut = nom_section_suivant(pris)
            pris["sections"].append({"nom": defaut})
        d["sections"].append(_section(s, i, defaut))
    return d

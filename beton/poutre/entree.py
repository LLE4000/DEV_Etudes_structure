# -*- coding: utf-8 -*-
"""Poutre en béton armé — l'ENTRÉE : les listes et les valeurs par défaut du module Streamlit
(``modules/poutre.py`` v2.40, la référence de parité), les classes de béton (``beton_classes.json``,
source unique, à la racine du dépôt) et le contrôle d'une poutre.

Le format complet (entrée et sortie) est décrit dans la docstring de ``beton/poutre/moteur.py``.

Règle de ``normaliser`` : une clé ABSENTE (ou ``null``) prend la valeur par défaut du module ; une
valeur HORS DU DOMAINE de l'écran (Ø non proposé, nuance autre que 400 / 500, classe de béton
inconnue, nombre négatif, clé inconnue…) lève ``ValueError`` avec le chemin du champ. Jamais de
substitution silencieuse : le module, lui, en fait plusieurs quand il relit un fichier JSON
(Ø inconnu → 16, nuance inconnue → 500, classe inconnue → C30/37…), elles sont listées dans
``docs/beton/POUTRE_MOTEUR.md``. La seule substitution reproduite est celle que le module applique à
une saisie VALIDE de son écran : un enrobage nul est lu 3,0 cm (voir ``geometrie.enrobage_de_calcul``).
"""
import copy
import functools
import json
import math
import os
from string import ascii_uppercase

from beton import position as P

from . import taux as T

CHEMIN_CLASSES = os.path.join(os.path.dirname(__file__), "..", "..", "beton_classes.json")

# ---- les listes de l'écran (modules/poutre.py) ----
DIAMETRES = (6, 8, 10, 12, 16, 20, 25, 32, 40)  # DIAM_OPTS : Ø des barres longitudinales (mm)
DIAMETRES_ETRIERS = (6, 8, 10, 12)              # SHEAR_DIAM_OPTS : Ø des étriers et des épingles (mm)
MAX_LITS = 4                                    # MAX_LITS : lits par face
NUANCES = (400, 500)                            # « Qualité acier (B) » : fyk (N/mm²)
BRINS = {"etrier": 2, "epingle": 1}             # _brins_from_type : un étrier = 2 brins, une épingle = 1
# les libellés admis pour le type d'une ligne d'effort tranchant (ceux du module compris)
_TYPES = {"etrier": "etrier", "étrier": "etrier", "epingle": "epingle", "épingle": "epingle"}

# ---- les valeurs par défaut (_ensure_global_defaults, _init_beams_if_needed, _ensure_defaults_for_beam) ----
BETON_DEFAUT = "C30/37"
DEFAUTS_POUTRE = {
    "nom": "Poutre 1",
    "b": 20.0,               # cm
    "h": 40.0,               # cm
    "beton": BETON_DEFAUT,
    "fyk": 500,              # N/mm²
    "gamma_s": 1.5,          # « Coefficient acier ELS » : fyd = fyk / γs
    "enrobage": 3.0,         # cm
    # le JEU DE POSE (cm) : la clé garde son nom d'avant (« jeu sous L1 »), sa règle est celle de beton/position.py
    # (ajouté à la position réelle de chaque lit AVANT l'arrondi) ; 1 cm par défaut, décision du propriétaire du
    # 13/11/2026 (c'était 0, « retour bureau 01/09 », ajouté après les demi-Ø arrondis)
    "jeu_premier_lit": P.JEU_POSE,
    "jeu_entre_lits": 1.0,   # cm : le jour libre entre deux lits (position réelle, sans arrondi)
    "arrondi": P.ARRONDI,    # cm : le pas d'arrondi de l'axe de calcul (beton/position.ARRONDIS : 0, 0,5 ou 1)
}
LIT_DEFAUT = {"n": 2, "d": 16}
ETRIER_DEFAUT = {"type": "etrier", "d": 10}
PAS_DEFAUT = 30.0            # cm

# ---- le FERRAILLAGE DÉTAILLÉ (site, 11/2026) : des clés FACULTATIVES, absentes du module Streamlit. Absentes (ou à leur
#      valeur par défaut), la poutre normalisée est celle d'avant, à l'octet : la parité avec l'écran est intacte.
#   lit     « pos »  : position imposée (cm) — lit 1 : du parement à l'axe ; lit i > 1 : entraxe VERTICAL depuis le lit
#                      précédent (axe à axe). Absente : la position automatique du module (enrobage, jeux).
#           « rep »  : répartition HORIZONTALE — « uniforme » (défaut : entre les montants de l'étrier), « aligne »
#                      (sur les barres du lit 1 de la face, les angles d'abord), « entraxe » (groupe centré, entraxe « sh »).
#           « sh »   : entraxe horizontal (cm, axe à axe) quand « rep » vaut « entraxe ».
#   étrier  « de », « a » : les barres du lit 1 INFÉRIEUR que le cadre entoure (1 à n ; absentes : toute la largeur) —
#                      une épingle n'a qu'une position, « de ». Comme les clés shear_line*_from/_to du module (dessin).
#   section « laterales » : {"n": barres par face latérale (absent : automatique, la règle des armatures de peau du
#                      PDF Streamlit), "d": Ø} — dessin, note et dispositions ; elles n'entrent pas dans le calcul.
#   poutre  « dg » (mm) : le plus gros granulat, pour l'espacement libre minimal (EC2 8.2(2)) ; « peau_d » (mm) et
#           « peau_smax » (cm) : les armatures de peau automatiques (paramètres avancés du module).
#   section « justifs » : {identifiant d'une observation : {"texte", "empreinte"}} — les dispositions « à examiner » que
#                      l'ingénieur a examinées et justifiées (beton/poutre/dispositions.py) ; absente : aucune.
#   poutre  « taux » : l'option « Taux d'armature » (beton/poutre/taux.py) — {"actif", "note", "majoration" (%),
#                      "retour" (cm), "arrondi" (kg/m³)} ; absente : option éteinte, réglages par défaut. « note » :
#                      le taux de la poutre s'écrit AUSSI dans la note de calcul (faux par défaut : on ne donne pas
#                      toujours le taux d'armature à qui reçoit la note) ;
#   section « longueur » (m) : la longueur de poutre que la section représente, pour le taux d'armature de la poutre ;
#                      absente : la section n'entre pas dans ce taux.
REPARTITIONS = ("uniforme", "aligne", "entraxe")
DIAMETRES_LATERAUX = (8, 10, 12, 16)            # « Armatures technologiques » du module : Ø (mm)
DEFAUTS_DETAIL = {"dg": 20.0, "peau_d": 10, "peau_smax": 30.0}

CLES_POUTRE = frozenset(DEFAUTS_POUTRE) | {"sections", "taux"} | frozenset(DEFAUTS_DETAIL)
CLES_SECTION = frozenset({"nom", "M_inf", "M_sup", "V", "inf", "sup", "etriers", "pas", "yG_inf", "yG_sup", "laterales",
                          "justifs", "longueur"})
CLES_TAUX = frozenset({"actif", "note", "majoration", "retour", "arrondi"})
JUSTIF_MAX = 600                                # caractères d'une justification
CLES_LIT = frozenset({"n", "d", "pos", "rep", "sh"})
CLES_ETRIER = frozenset({"type", "d", "de", "a"})
CLES_LATERALES = frozenset({"n", "d"})


# ============================================================ les classes de béton
@functools.lru_cache(maxsize=1)
def _classes_lues() -> dict:
    """Le fichier lu une fois (lecture seule : le moteur ne le modifie jamais)."""
    with open(CHEMIN_CLASSES, "r", encoding="utf-8") as f:
        return json.load(f)


def classes_beton() -> dict:
    """Les classes de béton de l'application, lues dans ``beton_classes.json`` :
    ``{"C30/37": {"fck", "fck_cube", "tau_lim", "alpha_b", "mu_a500", "mu_a400"}, …}``.
    Rend une copie : la modifier ne touche pas ce que le moteur utilise."""
    return copy.deepcopy(_classes_lues())


# ============================================================ poutres et sections par défaut
def nouvelle_section(nom: str = "A") -> dict:
    """Une section aux valeurs par défaut du module (_ensure_defaults_for_beam) : sans sollicitation,
    2Ø16 en bas et en haut, un étrier Ø10, pas de 30 cm, centres de gravité calculés."""
    return {"nom": nom, "M_inf": 0.0, "M_sup": 0.0, "V": 0.0,
            "inf": [dict(LIT_DEFAUT)], "sup": [dict(LIT_DEFAUT)],
            "etriers": [dict(ETRIER_DEFAUT)], "pas": PAS_DEFAUT,
            "yG_inf": None, "yG_sup": None}


def nouvelle_poutre() -> dict:
    """Une poutre aux valeurs par défaut du module : « Poutre 1 », 20 × 40 cm, C30/37, B500, γs = 1,5,
    enrobage 3 cm, jeu de pose 1 cm, 1 cm entre lits, arrondi 0,5 cm, une section « A »."""
    p = copy.deepcopy(DEFAUTS_POUTRE)
    p["sections"] = [nouvelle_section("A")]
    return p


def _lettres():
    yield from ascii_uppercase
    for a in ascii_uppercase:
        for b in ascii_uppercase:
            yield a + b


def nom_section_suivant(poutre: dict) -> str:
    """Le nom d'une section ajoutée : la première lettre libre (A, B, …, Z, AA, AB…) — _next_section_name."""
    sections = poutre.get("sections") or []
    pris = {str(s.get("nom", "")).strip() for s in sections if isinstance(s, dict)}
    for lettre in _lettres():
        if lettre not in pris:
            return lettre
    return f"S{len(sections) + 1}"  # repli du module, improbable


# ============================================================ contrôle et normalisation
def _cles(d: dict, permises: frozenset, chemin: str) -> None:
    inconnues = sorted(str(k) for k in d if k not in permises)
    if inconnues:
        raise ValueError(f"{chemin} : clé(s) inconnue(s) {', '.join(inconnues)} — "
                         f"clés admises : {', '.join(sorted(permises))}")


def _val(d: dict, cle: str, defaut):
    v = d.get(cle)
    return defaut if v is None else v


def _nombre(x, chemin: str, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{chemin} : nombre attendu (reçu {x!r})")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{chemin} : nombre fini attendu (reçu {x!r})")
    if (v <= minimum) if strict else (v < minimum):
        raise ValueError(f"{chemin} : nombre {'>' if strict else '≥'} {minimum:g} attendu (reçu {x!r})")
    return v


def _entier(x, chemin: str, permis=None, minimum=None) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, float) and not x.is_integer()):
        raise ValueError(f"{chemin} : entier attendu (reçu {x!r})")
    v = int(x)
    if permis is not None and v not in permis:
        raise ValueError(f"{chemin} : {v} n'est pas proposé par le module ({' '.join(map(str, permis))})")
    if minimum is not None and v < minimum:
        raise ValueError(f"{chemin} : entier ≥ {minimum} attendu (reçu {x!r})")
    return v


def _texte(x, chemin: str) -> str:
    if not isinstance(x, str):
        raise ValueError(f"{chemin} : texte attendu (reçu {x!r})")
    return x


def _liste(x, chemin: str, n_min: int, n_max=None) -> list:
    if not isinstance(x, (list, tuple)) or len(x) < n_min or (n_max is not None and len(x) > n_max):
        borne = f"de {n_min} à {n_max}" if n_max is not None else f"d'au moins {n_min}"
        raise ValueError(f"{chemin} : liste {borne} élément(s) attendue (reçu {x!r})")
    return list(x)


def _lits(x, chemin: str) -> list:
    if x is None:
        return [dict(LIT_DEFAUT)]
    out = []
    for i, lit in enumerate(_liste(x, chemin, 1, MAX_LITS)):
        c = f"{chemin}[{i}]"
        if not isinstance(lit, dict):
            raise ValueError(f"{c} : {{\"n\": …, \"d\": …}} attendu (reçu {lit!r})")
        _cles(lit, CLES_LIT, c)
        x = {"n": _entier(_val(lit, "n", LIT_DEFAUT["n"]), f"{c}.n", minimum=1),
             "d": _entier(_val(lit, "d", LIT_DEFAUT["d"]), f"{c}.d", permis=DIAMETRES)}
        # le détail : seulement s'il diffère du défaut (la poutre d'avant reste la même, à l'octet)
        if lit.get("pos") is not None:
            x["pos"] = _nombre(lit["pos"], f"{c}.pos", strict=True)
        rep = lit.get("rep")
        if rep is not None and rep != "uniforme":
            if rep not in REPARTITIONS:
                raise ValueError(f"{c}.rep : {' | '.join(REPARTITIONS)} attendu (reçu {rep!r})")
            x["rep"] = rep
        if lit.get("sh") is not None:
            x["sh"] = _nombre(lit["sh"], f"{c}.sh", strict=True)
        out.append(x)
    return out


def _etriers(x, chemin: str) -> list:
    if x is None:
        return [dict(ETRIER_DEFAUT)]
    out = []
    for i, e in enumerate(_liste(x, chemin, 1)):
        c = f"{chemin}[{i}]"
        if not isinstance(e, dict):
            raise ValueError(f"{c} : {{\"type\": …, \"d\": …}} attendu (reçu {e!r})")
        _cles(e, CLES_ETRIER, c)
        t = _val(e, "type", ETRIER_DEFAUT["type"])
        if not isinstance(t, str) or t.strip().lower() not in _TYPES:
            raise ValueError(f"{c}.type : « etrier » (2 brins) ou « epingle » (1 brin) attendu (reçu {t!r})")
        x = {"type": _TYPES[t.strip().lower()],
             "d": _entier(_val(e, "d", ETRIER_DEFAUT["d"]), f"{c}.d", permis=DIAMETRES_ETRIERS)}
        for k in ("de", "a"):
            if e.get(k) is not None:
                x[k] = _entier(e[k], f"{c}.{k}", minimum=1)
        out.append(x)
    return out


def _laterales(x, chemin: str):
    """Les armatures latérales imposées : {"n": par face, "d": Ø} ; None (absentes) : la règle automatique."""
    if x is None:
        return None
    if not isinstance(x, dict):
        raise ValueError(f"{chemin} : {{\"n\": …, \"d\": …}} attendu (reçu {x!r})")
    _cles(x, CLES_LATERALES, chemin)
    out = {}
    if x.get("n") is not None:
        out["n"] = _entier(x["n"], f"{chemin}.n", minimum=0)
    if x.get("d") is not None:
        out["d"] = _entier(x["d"], f"{chemin}.d", permis=DIAMETRES_LATERAUX)
    return out or None


def _taux(x, chemin: str):
    """L'option « Taux d'armature » de la poutre : {"actif" (bool), "note" (bool : le taux s'écrit aussi dans la note
    de calcul), "majoration" (%), "retour" (cm), "arrondi" (kg/m³)}, chaque réglage dans ses bornes
    (beton/poutre/taux.py). None quand l'option est éteinte ET que tous les réglages sont à leur défaut : la clé
    disparaît, et la poutre d'avant reste la même à l'octet. Éteinte avec un réglage changé : gardée (le réglage n'est
    pas perdu quand on rallume). « note » n'est écrite que vraie (fausse : absente)."""
    if x is None:
        return None
    if not isinstance(x, dict):
        raise ValueError(f"{chemin} : {{\"actif\", \"note\", \"majoration\", \"retour\", \"arrondi\"}} attendu (reçu {x!r})")
    _cles(x, CLES_TAUX, chemin)
    actif = x.get("actif")
    if actif is not None and not isinstance(actif, bool):
        raise ValueError(f"{chemin}.actif : vrai ou faux attendu (reçu {actif!r})")
    out = {"actif": bool(actif)}
    note = x.get("note")
    if note is not None and not isinstance(note, bool):
        raise ValueError(f"{chemin}.note : vrai ou faux attendu (reçu {note!r})")
    if note:
        out["note"] = True
    noms = {"majoration": "la majoration (%)", "retour": "le retour d'étrier (cm)", "arrondi": "l'arrondi (kg/m³)"}
    for k, defaut in T.DEFAUTS.items():
        v = x.get(k)
        if v is None:
            continue
        mini, maxi = T.BORNES[k]
        v = _entier(v, f"{chemin}.{k}") if k == "arrondi" else _nombre(v, f"{chemin}.{k}")
        if not mini <= v <= maxi:
            raise ValueError(f"{chemin}.{k} : {noms[k]} se saisit de {mini:g} à {maxi:g} (reçu {v:g})")
        if v != defaut:
            out[k] = v
    return out if (out["actif"] or len(out) > 1) else None


def _justifs(x, chemin: str):
    """Les justifications de l'ingénieur : {identifiant : {"texte" (≤ JUSTIF_MAX caractères, peut être vide),
    "empreinte"}} ; None ou vide : aucune (la clé disparaît, la section d'avant reste la même à l'octet)."""
    if not x:
        return None
    if not isinstance(x, dict):
        raise ValueError(f"{chemin} : {{identifiant : {{\"texte\", \"empreinte\"}}}} attendu (reçu {x!r})")
    out = {}
    for k, v in x.items():
        c = f"{chemin}[{k!r}]"
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"{chemin} : identifiant d'observation attendu (reçu {k!r})")
        if not isinstance(v, dict):
            raise ValueError(f"{c} : {{\"texte\", \"empreinte\"}} attendu (reçu {v!r})")
        _cles(v, frozenset({"texte", "empreinte"}), c)
        texte = _texte(v.get("texte") or "", f"{c}.texte").strip()
        if len(texte) > JUSTIF_MAX:
            raise ValueError(f"{c}.texte : {JUSTIF_MAX} caractères au plus (reçu {len(texte)})")
        out[k] = {"texte": texte, "empreinte": _texte(v.get("empreinte") or "", f"{c}.empreinte")}
    return out or None


def _section(s, i: int, nom_defaut: str) -> dict:
    c = f"sections[{i}]"
    if not isinstance(s, dict):
        raise ValueError(f"{c} : dict attendu (reçu {s!r})")
    _cles(s, CLES_SECTION, c)
    out = {"nom": _texte(_val(s, "nom", nom_defaut), f"{c}.nom")}
    for k in ("M_inf", "M_sup", "V"):
        out[k] = _nombre(_val(s, k, 0.0), f"{c}.{k}")
    out["inf"] = _lits(s.get("inf"), f"{c}.inf")
    out["sup"] = _lits(s.get("sup"), f"{c}.sup")
    out["etriers"] = _etriers(s.get("etriers"), f"{c}.etriers")
    out["pas"] = _nombre(_val(s, "pas", PAS_DEFAUT), f"{c}.pas", strict=True)
    for k in ("yG_inf", "yG_sup"):
        out[k] = None if s.get(k) is None else _nombre(s[k], f"{c}.{k}", strict=True)
    lat = _laterales(s.get("laterales"), f"{c}.laterales")
    if lat is not None:
        out["laterales"] = lat
    js = _justifs(s.get("justifs"), f"{c}.justifs")
    if js is not None:
        out["justifs"] = js
    if s.get("longueur") is not None:
        lg = _nombre(s["longueur"], f"{c}.longueur", strict=True)
        if lg > T.LONGUEUR_MAX:
            raise ValueError(f"{c}.longueur : la longueur d'une section se saisit en mètres, jusqu'à {T.LONGUEUR_MAX:g} m "
                             f"(reçu {lg:g})")
        out["longueur"] = lg
    return out


def normaliser(poutre: dict) -> dict:
    """Une copie COMPLÈTE et CONTRÔLÉE de la poutre : défauts du module pour les clés absentes,
    ``ValueError`` (avec le chemin du champ) pour toute valeur hors du domaine de l'écran.
    ``normaliser(normaliser(p)) == normaliser(p)``."""
    if not isinstance(poutre, dict):
        raise ValueError(f"poutre : dict attendu (reçu {poutre!r})")
    _cles(poutre, CLES_POUTRE, "poutre")
    p = {"nom": _texte(_val(poutre, "nom", DEFAUTS_POUTRE["nom"]), "nom")}
    p["b"] = _nombre(_val(poutre, "b", DEFAUTS_POUTRE["b"]), "b", strict=True)
    p["h"] = _nombre(_val(poutre, "h", DEFAUTS_POUTRE["h"]), "h", strict=True)
    beton = _val(poutre, "beton", DEFAUTS_POUTRE["beton"])
    if beton not in _classes_lues():
        raise ValueError(f"beton : classe {beton!r} inconnue ({' '.join(_classes_lues())})")
    p["beton"] = beton
    p["fyk"] = _entier(_val(poutre, "fyk", DEFAUTS_POUTRE["fyk"]), "fyk", permis=NUANCES)
    p["gamma_s"] = _nombre(_val(poutre, "gamma_s", DEFAUTS_POUTRE["gamma_s"]), "gamma_s", strict=True)
    for k in ("enrobage", "jeu_premier_lit", "jeu_entre_lits"):
        p[k] = _nombre(_val(poutre, k, DEFAUTS_POUTRE[k]), k)
    p["arrondi"] = P.lire_arrondi(_val(poutre, "arrondi", DEFAUTS_POUTRE["arrondi"]))
    # le détail du ferraillage : seulement s'il diffère du défaut
    if poutre.get("dg") is not None and float(poutre["dg"]) != DEFAUTS_DETAIL["dg"]:
        p["dg"] = _nombre(poutre["dg"], "dg", strict=True)
    if poutre.get("peau_d") is not None and poutre["peau_d"] != DEFAUTS_DETAIL["peau_d"]:
        p["peau_d"] = _entier(poutre["peau_d"], "peau_d", permis=DIAMETRES_LATERAUX)
    if poutre.get("peau_smax") is not None and float(poutre["peau_smax"]) != DEFAUTS_DETAIL["peau_smax"]:
        p["peau_smax"] = _nombre(poutre["peau_smax"], "peau_smax", strict=True)
    tx = _taux(poutre.get("taux"), "taux")
    if tx is not None:
        p["taux"] = tx
    sections = poutre.get("sections")
    if sections is None:
        p["sections"] = [nouvelle_section("A")]
        return p
    sections = _liste(sections, "sections", 1)
    # une section sans nom prend la première lettre libre, comme une section ajoutée à l'écran
    noms = [s.get("nom") if isinstance(s, dict) else None for s in sections]
    pris = {"sections": [{"nom": n} for n in noms if isinstance(n, str)]}
    p["sections"] = []
    for i, s in enumerate(sections):
        defaut = "A"
        if noms[i] is None:
            defaut = nom_section_suivant(pris)
            pris["sections"].append({"nom": defaut})
        p["sections"].append(_section(s, i, defaut))
    return p

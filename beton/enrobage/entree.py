# -*- coding: utf-8 -*-
"""Enrobage du béton — l'ENTRÉE : un élément (les widgets de l'écran ``modules/enrobage.py``), ses valeurs par défaut,
sa normalisation stricte et la SAISIE d'un champ avec les effets de bord de l'écran.

Un élément (unités de l'écran : mm) ::

    {"nom": "Élément 1", "type": "Poutre", "position": "Intérieur", "diam_barre": 20, "diam_etrier": 8,
     "delta_dev": 10, "classe_expo": "X0", "classe_struct": "S1", "feu": false, "r": "R60", "b_min": 120}

``b_min`` est la « Largeur minimale de l'élément b » choisie dans le tableau feu (poutre, poteau) : la largeur d'une
variante de la durée ``r`` ; ``null`` pour une dalle ou un voile (l'écran n'a pas ce menu).

Règle de ``normaliser`` : une clé ABSENTE (ou ``null``) prend la valeur par défaut de l'écran ; une valeur HORS DU
DOMAINE de l'écran (type inconnu, Ø non entier ou sous le minimum du champ, Δc_dev autre que 10 / 5 / 0, classe
inconnue, largeur absente du tableau, clé inconnue…) lève ``ValueError`` avec le nom du champ — jamais de substitution.
Un garde-fou du site s'y ajoute (écart voulu) : Ø ≤ 60 mm, Ø étrier ≤ 40 mm (donnees.DIAM_*_MAX) — l'écran n'a pas de
maximum, mais au-delà la saisie est absurde (et 10⁴⁰⁰ ferait déborder les flottants du calcul).

Les effets de bord de ``appliquer`` sont ceux des widgets SANS clé de l'écran, dont Streamlit calcule l'identité avec
leurs paramètres (prouvé par l'AppTest dans ``tests/test_enrobage_moteur.py``) :
- « Classe structurale » a pour index par défaut la classe proposée pour l'exposition : changer d'exposition la REMET à
  cette classe quand la classe proposée change (X0 → XC1 : S1 → S2), et la GARDE quand elle est la même (XC2/XC3 → XC4,
  toutes deux S4) ;
- « Largeur minimale » a pour options les variantes du tableau (type, durée) : changer de type ou de durée la REMET à
  la première variante quand la liste change, et la garde quand elle est la même (poteau R30 → R60).
"""
import copy
import math

from . import donnees as D

CLES = ("nom", "type", "position", "diam_barre", "diam_etrier", "delta_dev", "classe_expo", "classe_struct", "feu", "r",
        "b_min")
NOM_DEFAUT = "Élément 1"


def variantes(type_element: str, r_level: str) -> list:
    """Les variantes (b_min, a) du menu « Largeur minimale » : celles du tableau de la poutre ou du poteau pour la
    durée ; [] pour une dalle ou un voile (l'écran n'affiche pas ce menu)."""
    if type_element == "Poutre":
        return list(D.FEU_POUTRE.get(r_level, []))
    if type_element == "Poteau":
        return list(D.FEU_POTEAU.get(r_level, []))
    return []


def nouvel_element(nom: str = NOM_DEFAUT) -> dict:
    """Un élément aux valeurs par défaut de l'écran : poutre intérieure, Ø20, étrier Ø8, Δc_dev = 10 mm, X0 (le premier
    de la liste) et sa classe proposée S1, feu non vérifié, R60 et la première largeur du tableau 5.5 (120 mm)."""
    expo = D.EXPO_CLASSES[0]
    return {"nom": nom, "type": D.TYPES[0], "position": D.POSITIONS[0], "diam_barre": D.DIAM_BARRE_DEFAUT,
            "diam_etrier": D.DIAM_ETRIER_DEFAUT, "delta_dev": D.DELTAS_DEV[0], "classe_expo": expo,
            "classe_struct": D.CLASSE_STRUCTURALE_DEFAULT[expo], "feu": False, "r": D.R_DEFAUT,
            "b_min": variantes(D.TYPES[0], D.R_DEFAUT)[0][0]}


# ------------------------------------------------------------------ contrôles
def _parmi(x, permis, champ: str):
    if isinstance(x, bool) or x not in permis:
        raise ValueError(f"{champ} : {x!r} n'est pas proposé par l'écran ({' · '.join(map(str, permis))})")
    return x


def _entier(x, champ: str, minimum: int, maximum: int) -> int:
    """Un Ø de l'écran : un ENTIER (number_input à valeur entière) au moins égal au minimum du champ, et au plus égal
    au garde-fou du site (donnees.DIAM_*_MAX : au-delà, une saisie absurde)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or (isinstance(x, float) and not (math.isfinite(x) and x.is_integer())):
        raise ValueError(f"{champ} : un nombre entier de millimètres est attendu (reçu {x!r})")
    if x < minimum:
        raise ValueError(f"{champ} : au moins {minimum} mm (le minimum du champ de l'écran ; reçu {int(x)})")
    if x > maximum:
        recu = f"{int(x)} mm" if x < 1e12 else "un nombre démesuré"   # 10⁴⁰⁰ ne s'écrit pas en flottant
        raise ValueError(f"{champ} : au plus {maximum} mm (au-delà, la valeur n'a pas de sens pour une armature ; reçu {recu})")
    return int(x)


def normaliser(el: dict) -> dict:
    """Une copie COMPLÈTE et CONTRÔLÉE de l'élément ; ``normaliser(normaliser(e)) == normaliser(e)``."""
    if not isinstance(el, dict):
        raise ValueError(f"élément : dict attendu (reçu {el!r})")
    inconnues = sorted(str(k) for k in el if k not in CLES)
    if inconnues:
        raise ValueError(f"élément : clé(s) inconnue(s) {', '.join(inconnues)} — clés admises : {', '.join(CLES)}")
    d = nouvel_element()

    def val(k):
        return d[k] if el.get(k) is None else el[k]

    nom = val("nom")
    if not isinstance(nom, str):
        raise ValueError(f"nom : texte attendu (reçu {nom!r})")
    out = {"nom": nom,
           "type": _parmi(val("type"), D.TYPES, "type"),
           "position": _parmi(val("position"), D.POSITIONS, "position"),
           "diam_barre": _entier(val("diam_barre"), "diam_barre", D.DIAM_BARRE_MIN, D.DIAM_BARRE_MAX),
           "diam_etrier": _entier(val("diam_etrier"), "diam_etrier", D.DIAM_ETRIER_MIN, D.DIAM_ETRIER_MAX),
           "delta_dev": _parmi(val("delta_dev"), D.DELTAS_DEV, "delta_dev"),
           "classe_expo": _parmi(val("classe_expo"), D.EXPO_CLASSES, "classe_expo")}
    if "classe_struct" in el and el["classe_struct"] is not None:
        out["classe_struct"] = _parmi(el["classe_struct"], D.CLASSES_STRUCTURALES, "classe_struct")
    else:  # absente : la classe proposée pour l'exposition (l'index par défaut du widget)
        out["classe_struct"] = D.CLASSE_STRUCTURALE_DEFAULT[out["classe_expo"]]
    feu = val("feu")
    if not isinstance(feu, bool):
        raise ValueError(f"feu : vrai ou faux attendu (reçu {feu!r})")
    out["feu"] = feu
    out["r"] = _parmi(val("r"), D.R_LEVELS, "r")
    vs = variantes(out["type"], out["r"])
    b = el.get("b_min")
    if not vs:
        if b is not None:
            raise ValueError(f"b_min : sans objet pour un(e) {out['type'].lower()} (reçu {b!r}) — null attendu")
        out["b_min"] = None
    elif b is None:
        out["b_min"] = vs[0][0]
    else:
        out["b_min"] = _parmi(b, [v[0] for v in vs], f"b_min ({out['type'].lower()} {out['r']})")
    return out


# ------------------------------------------------------------------ la saisie d'un champ
def appliquer(el: dict, cle: str, valeur) -> dict:
    """L'élément après la saisie ``cle = valeur`` (une copie normalisée), avec les effets de bord de l'écran (voir la
    docstring du module). ``ValueError`` si la valeur est hors du domaine : l'élément reçu n'est pas touché."""
    if cle not in CLES:
        raise ValueError(f"champ inconnu : {cle!r}")
    avant = normaliser(el)
    e = copy.deepcopy(avant)
    e[cle] = valeur
    if cle == "classe_expo":
        _parmi(valeur, D.EXPO_CLASSES, "classe_expo")
        if D.CLASSE_STRUCTURALE_DEFAULT[valeur] != D.CLASSE_STRUCTURALE_DEFAULT[avant["classe_expo"]]:
            e["classe_struct"] = D.CLASSE_STRUCTURALE_DEFAULT[valeur]
    elif cle in ("type", "r"):
        _parmi(valeur, D.TYPES if cle == "type" else D.R_LEVELS, cle)
        anciennes = variantes(avant["type"], avant["r"])
        nouvelles = variantes(e["type"], e["r"])
        if nouvelles != anciennes:
            e["b_min"] = nouvelles[0][0] if nouvelles else None
    return normaliser(e)

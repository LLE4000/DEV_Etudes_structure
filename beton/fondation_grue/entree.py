# -*- coding: utf-8 -*-
"""Fondations de grues — les ENTRÉES : les trois schémas de fondation des classeurs du bureau, leurs champs (un champ
= une cellule de saisie du classeur, mêmes unités), leurs valeurs par défaut (celles des classeurs), la normalisation.

Unités des classeurs, gardées telles quelles : longueurs en cm ; enrobage, Ø et pas en mm ; efforts en kN, moments en
kN·m ; capacité d'un pieu en kN (variante 4 pieux) ou en tonnes (BASE). Bibliothèque standard seulement.

    TYPES                      les trois schémas : « semelles8 », « cadre4 », « diagonales8 »
    SAISIES[type]              chemin d'un champ → cellule de la fiche de calcul (source de la correspondance)
    EXEMPLES                   les quatre classeurs relus : BASE, GWW (Greenwood), VAR4, VAR8
    nouvelle_fondation(type)   une fondation aux valeurs du classeur de son schéma
    normaliser(f)              copie complétée et contrôlée ; ValueError (message clair) hors du domaine
    lire(f, chemin) / poser(f, chemin, valeur)   accès par chemin « support/inf/0/n »
"""
import copy

from . import donnees as D

SEMELLES8, CADRE4, DIAGONALES8 = "semelles8", "cadre4", "diagonales8"
TYPES = (SEMELLES8, CADRE4, DIAGONALES8)
LIBELLES_TYPE = {
    SEMELLES8: "8 pieux — 2 poutres support sur 4 semelles de 2 pieux",
    CADRE4: "4 pieux — 4 poutres en cadre",
    DIAGONALES8: "8 pieux — poutres primaires sur 2 pieux et poutres secondaires",
}
COURTS_TYPE = {SEMELLES8: "8 pieux · semelles", CADRE4: "4 pieux · cadre", DIAGONALES8: "8 pieux · primaires"}
CLASSEUR_TYPE = {SEMELLES8: "BASE", CADRE4: "VAR4", DIAGONALES8: "VAR8"}
NB_PIEUX = {SEMELLES8: 8, CADRE4: 4, DIAGONALES8: 8}

COTE_SEMELLES, COTE_PAROI = "semelles", "paroi"
COTES_B = (COTE_SEMELLES, COTE_PAROI)
LIBELLES_COTE_B = {COTE_SEMELLES: "semelles sur pieux des deux côtés (8 pieux)",
                   COTE_PAROI: "paroi de pieux sécants du côté b (4 pieux)"}
SERVICE, HORS = "service", "hors"
ETATS_CAS = (SERVICE, HORS)
LIBELLES_ETAT_CAS = {SERVICE: "en service", HORS: "hors service"}
PIEDS = ("A", "B", "C", "D")

DIAMETRES = (6, 8, 10, 12, 14, 16, 20, 25, 28, 32, 40)
NOM_MAX = 60
TEXTE_MAX = 80
N_CAS_MAX = 12

# ------------------------------------------------------------------ champ → cellule de la « Fiche de calcul »
SAISIES = {
    CADRE4: {
        "projet": "C3", "grue": "C4", "beton": "G10", "acier": "G11",
        "chassis": "C10", "poutre/b": "C11", "poutre/h": "C12", "pieux/D": "C13",
        "charges/P": "C24", "charges/H_hs": "C25", "charges/H_s": "G24", "charges/M_s": "G25", "e": "C29",
        "poutre/inf/n": "G56", "poutre/inf/d": "G57", "poutre/sup/n": "G60", "poutre/sup/d": "G61",
        "poutre/part_V": "G64",
        "poutre/etriers_appui/pas": "G66", "poutre/etriers_appui/brins": "G67", "poutre/etriers_appui/d": "G68",
        "poutre/etriers_travee/pas": "G70", "poutre/etriers_travee/brins": "G71", "poutre/etriers_travee/d": "G72",
        "pieux/capacite": "G84", "renfort/brins": "G115", "renfort/d": "G116", "perfos/n": "G134", "perfos/d": "G135",
    },
    DIAGONALES8: {
        "projet": "C3", "grue": "C4", "beton": "G10", "acier": "G11",
        "chassis": "C10", "secondaire/b": "C11", "secondaire/h": "C12", "primaire/b": "C13", "primaire/h": "C14",
        "primaire/L": "C15", "pieux/entraxe": "C16", "pieux/D": "C17",
        "charges/P": "C26", "charges/H_hs": "C27", "charges/H_s": "G26", "charges/M_s": "G27",
        "e_secondaire": "C31", "e_primaire": "C49",
        "secondaire/inf/n": "G71", "secondaire/inf/d": "G72", "secondaire/sup/n": "G75", "secondaire/sup/d": "G76",
        "secondaire/part_V": "G79",
        "secondaire/etriers/pas": "G81", "secondaire/etriers/brins": "G82", "secondaire/etriers/d": "G83",
        "primaire/inf/n": "G96", "primaire/inf/d": "G97", "primaire/sup/n": "G100", "primaire/sup/d": "G101",
        "primaire/etriers/pas": "G105", "primaire/etriers/brins": "G106", "primaire/etriers/d": "G107",
        "renfort/brins": "G140", "renfort/d": "G141", "perfos/n": "G159", "perfos/d": "G160",
    },
    SEMELLES8: {
        "projet": "C3", "grue": "C5", "niveau_beton": "G4", "fleche": "G5", "hauteur_crochet": "C6",
        "dimensions_chassis": "G6",
        "beton": "C13", "acier": "C14", "enrobage": "C15",
        "support/h": "C20", "support/b": "C21", "semelle/L": "C24", "semelle/h": "C25", "semelle/b": "C26",
        "pieux/entraxe": "C29", "pieux/D": "C30", "pieux/capacite": "C31",
        "chassis": "C34", "dist_a": "C35", "dist_b": "C37",
        "charges/H_s": "C60", "charges/H_hs": "G60", "charges/M_s": "C61",
        "e_support": "C66", "e_semelle": "C101", "c_appui": "C102",
        "support/inf/0/n": "G129", "support/inf/0/d": "G130", "support/inf/1/n": "G131", "support/inf/1/d": "G132",
        "support/sup/0/n": "G136", "support/sup/0/d": "G137", "support/sup/1/n": "G138", "support/sup/1/d": "G139",
        "support/part_V": "G143",
        "support/etriers_extremites/pas": "G145", "support/etriers_extremites/d": "G146",
        "support/etriers_extremites/brins": "G147",
        "support/etriers_centre/pas": "G152", "support/etriers_centre/d": "G153", "support/etriers_centre/brins": "G154",
        "semelle/inf/0/n": "G175", "semelle/inf/0/d": "G176", "semelle/inf/1/n": "G177", "semelle/inf/1/d": "G178",
        "semelle/sup/0/n": "G181", "semelle/sup/0/d": "G182", "semelle/sup/1/n": "G183", "semelle/sup/1/d": "G184",
        "semelle/etriers/pas": "G189", "semelle/etriers/d": "G190", "semelle/etriers/brins": "G191",
        "renfort/brins": "G234", "renfort/d": "G235", "renfort/brins_base": "G236",
        "perfos/n": "G258", "perfos/d": "G259",
    },
}
# les six cas de la fiche BASE : CAS 1 à 3 en service (C48:C59), CAS 4 à 6 hors service (G48:G59)
for _i in range(6):
    for _k, _pied in enumerate(PIEDS):
        SAISIES[SEMELLES8][f"charges/cas/{_i}/{_pied}"] = f"{'C' if _i < 3 else 'G'}{48 + 4 * (_i % 3) + _k}"
CHAMPS_TEXTE = ("projet", "grue", "niveau_beton", "fleche", "hauteur_crochet", "dimensions_chassis")


def _barres(n, d):
    return {"n": n, "d": d}


def _etriers(brins, d, pas):
    return {"brins": brins, "d": d, "pas": float(pas)}


def _cas(i, etat, A, B, C, D_):
    return {"nom": f"CAS {i}", "etat": etat, "A": float(A), "B": float(B), "C": float(C), "D": float(D_)}


def _semelles8(nom, projet, grue, textes, support, semelle, pieux, cas, H_s, H_hs, M_s, renfort, perfos):
    f = {"type": SEMELLES8, "nom": nom, "projet": projet, "grue": grue,
         "niveau_beton": "", "fleche": "", "hauteur_crochet": "", "dimensions_chassis": "",
         "beton": "C30/37", "acier": "BE500", "enrobage": 40.0, "cote_b": COTE_SEMELLES,
         "support": support, "semelle": semelle, "pieux": pieux,
         "chassis": 600.0, "dist_a": 25.0, "dist_b": 25.0,
         "charges": {"cas": [_cas(i + 1, SERVICE if i < 3 else HORS, *c) for i, c in enumerate(cas)],
                     "H_s": float(H_s), "H_hs": float(H_hs), "M_s": float(M_s)},
         "e_support": 20.0, "e_semelle": 20.0, "c_appui": 50.0, "renfort": renfort, "perfos": perfos}
    f.update(textes)
    return f


# ------------------------------------------------------------------ les quatre classeurs relus (saisies telles quelles)
EXEMPLES = {
    "BASE": _semelles8(
        "Fondation de grue 1", "GWW3 GT 11-05-2021", "280 ECH-12 Lit",
        {"niveau_beton": "/", "fleche": "75 m", "hauteur_crochet": "48,43 m", "dimensions_chassis": "6m x 6m"},
        {"b": 120.0, "h": 75.0, "inf": [_barres(10, 20), _barres(0, 16)], "sup": [_barres(6, 16), _barres(0, 16)],
         "part_V": 100.0, "etriers_extremites": _etriers(6, 12, 100), "etriers_centre": _etriers(4, 12, 750)},
        {"L": 250.0, "b": 90.0, "h": 75.0, "inf": [_barres(8, 20), _barres(0, 25)],
         "sup": [_barres(4, 16), _barres(0, 16)], "etriers": _etriers(4, 12, 150)},
        {"entraxe": 120.0, "D": 41.0, "capacite": 65.0},
        ((495, 889, 495, 101), (776, 776, 214, 214), (214, 776, 776, 214),
         (462, 828, 462, 95), (704, 704, 219, 219), (219, 704, 704, 219)),
        67, 141, 460, {"brins": 16, "d": 12, "brins_base": 5}, _barres(1, 25)),
    "GWW": _semelles8(
        "Greenwood", "GWW4 GT 11-04-25", "WOLF 7534",
        {"niveau_beton": "76,85", "fleche": "55 m", "hauteur_crochet": "47,7", "dimensions_chassis": "6m x 6m"},
        {"b": 140.0, "h": 70.0, "inf": [_barres(8, 20), _barres(0, 16)], "sup": [_barres(4, 16), _barres(0, 16)],
         "part_V": 100.0, "etriers_extremites": _etriers(6, 12, 100), "etriers_centre": _etriers(4, 12, 300)},
        {"L": 270.0, "b": 100.0, "h": 75.0, "inf": [_barres(8, 20), _barres(0, 25)],
         "sup": [_barres(4, 16), _barres(0, 16)], "etriers": _etriers(4, 12, 125)},
        {"entraxe": 170.0, "D": 53.0, "capacite": 75.0},
        ((369, 1004, 369, 0), (790, 790, 80, 80), (0, 0, 0, 0), (394, 770, 394, 18), (660, 660, 128, 128), (0, 0, 0, 0)),
        51, 129, 430, {"brins": 16, "d": 12, "brins_base": 5}, _barres(1, 25)),
    "VAR4": {
        "type": CADRE4, "nom": "Fondation de grue 1", "projet": "GSK - RX310", "grue": "132 ECH-8 Lit",
        "beton": "C30/37", "acier": "BE500", "chassis": 450.0,
        "poutre": {"b": 90.0, "h": 50.0, "inf": _barres(5, 16), "sup": _barres(5, 16), "part_V": 50.0,
                   "etriers_appui": _etriers(4, 12, 100), "etriers_travee": _etriers(4, 8, 300)},
        "pieux": {"D": 70.0, "capacite": 600.0},
        "charges": {"P": 474.0, "H_hs": 69.0, "H_s": 34.0, "M_s": 200.0},
        "e": 15.0, "renfort": {"brins": 16, "d": 12}, "perfos": _barres(1, 20)},
    "VAR8": {
        "type": DIAGONALES8, "nom": "Fondation de grue 1", "projet": "GWW2 - Greenwood", "grue": "280 ECH-12 Lit",
        "beton": "C30/37", "acier": "BE500", "chassis": 600.0,
        "secondaire": {"b": 40.0, "h": 50.0, "inf": _barres(6, 16), "sup": _barres(6, 16), "part_V": 10.0,
                       "etriers": _etriers(2, 12, 250)},
        "primaire": {"b": 90.0, "h": 70.0, "L": 240.0, "inf": _barres(7, 20), "sup": _barres(4, 20),
                     "etriers": _etriers(4, 12, 150)},
        "pieux": {"entraxe": 160.0, "D": 41.0},
        "charges": {"P": 820.0, "H_hs": 133.0, "H_s": 62.0, "M_s": 303.0},
        "e_secondaire": 15.0, "e_primaire": 20.0, "renfort": {"brins": 14, "d": 12}, "perfos": _barres(1, 20)},
}
LIBELLES_EXEMPLE = {"BASE": "classeur BASE (8 pieux, 4 semelles)", "GWW": "classeur Greenwood (GWW)",
                    "VAR4": "classeur variante 4 pieux", "VAR8": "classeur variante 8 pieux"}
EXEMPLE_TYPE = {SEMELLES8: "BASE", CADRE4: "VAR4", DIAGONALES8: "VAR8"}


def exemple(cle: str) -> dict:
    """Une copie d'un des quatre classeurs relus (« BASE », « GWW », « VAR4 », « VAR8 »)."""
    if cle not in EXEMPLES:
        raise ValueError(f"exemple « {cle} » inconnu : {', '.join(EXEMPLES)}")
    return copy.deepcopy(EXEMPLES[cle])


def nouvelle_fondation(type_: str = SEMELLES8, nom: str = "Fondation de grue 1") -> dict:
    """Une fondation neuve : les saisies du classeur de son schéma, sans le nom du projet ni de la grue."""
    if type_ not in TYPES:
        raise ValueError(f"schéma « {type_} » inconnu : {', '.join(TYPES)}")
    f = exemple(EXEMPLE_TYPE[type_])
    f["nom"] = nom
    for cle in CHAMPS_TEXTE:
        if cle in f:
            f[cle] = ""
    return f


# ------------------------------------------------------------------ accès par chemin
def _pas(noeud, cle):
    if isinstance(noeud, list):
        try:
            return noeud[int(cle)]
        except (ValueError, IndexError):
            raise KeyError(cle)
    return noeud[cle]


def lire(f: dict, chemin: str):
    """La valeur au chemin « support/inf/0/n » ; KeyError si le chemin n'existe pas."""
    noeud = f
    for cle in chemin.split("/"):
        noeud = _pas(noeud, cle)
    return noeud


def poser(f: dict, chemin: str, valeur) -> None:
    """Écrit une valeur au chemin (le chemin doit exister)."""
    cles = chemin.split("/")
    noeud = f
    for cle in cles[:-1]:
        noeud = _pas(noeud, cle)
    if isinstance(noeud, list):
        noeud[int(cles[-1])] = valeur
    else:
        if cles[-1] not in noeud:
            raise KeyError(chemin)
        noeud[cles[-1]] = valeur


# ------------------------------------------------------------------ normalisation
def _exiger(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nombre(nom: str, x, mini=None, strict=False, maxi=None) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{nom} : un nombre est attendu")
    x = float(x)
    _exiger(x == x and abs(x) != float("inf"), f"{nom} : un nombre fini est attendu")
    if mini is not None:
        _exiger(x > mini if strict else x >= mini, f"{nom} : doit être {'>' if strict else '≥'} {mini:g}")
    if maxi is not None:
        _exiger(x <= maxi, f"{nom} : doit être ≤ {maxi:g}")
    return x + 0.0


def _entier(nom: str, x, mini: int, maxi: int) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or float(x) != int(x):
        raise ValueError(f"{nom} : un nombre entier est attendu")
    x = int(x)
    _exiger(mini <= x <= maxi, f"{nom} : entre {mini} et {maxi}")
    return x


def _diametre(nom: str, x) -> int:
    d = _entier(nom, x, 1, 100)
    _exiger(d in DIAMETRES, f"{nom} : Ø parmi {', '.join(str(v) for v in DIAMETRES)}")
    return d


def _texte(nom: str, x, maxi: int = TEXTE_MAX) -> str:
    if x is None:
        return ""
    _exiger(isinstance(x, str), f"{nom} : un texte est attendu")
    x = " ".join(x.split())
    _exiger(len(x) <= maxi, f"{nom} : {maxi} caractères au plus")
    return x


def _lit(nom: str, x, n_mini: int = 1) -> dict:
    _exiger(isinstance(x, dict), f"{nom} : nombre et Ø attendus")
    return {"n": _entier(f"{nom}, nombre de barres", x.get("n"), n_mini, 60), "d": _diametre(f"{nom}, Ø", x.get("d"))}


def _deux_lits(nom: str, x) -> list:
    _exiger(isinstance(x, list) and len(x) == 2, f"{nom} : deux groupes de barres attendus (le second peut être à 0)")
    lits = [_lit(f"{nom} (groupe 1)", x[0], 0), _lit(f"{nom} (groupe 2)", x[1], 0)]
    _exiger(lits[0]["n"] + lits[1]["n"] >= 1, f"{nom} : au moins une barre")
    return lits


def _cadres(nom: str, x) -> dict:
    _exiger(isinstance(x, dict), f"{nom} : brins, Ø et pas attendus")
    return {"brins": _entier(f"{nom}, brins", x.get("brins"), 1, 40), "d": _diametre(f"{nom}, Ø", x.get("d")),
            "pas": _nombre(f"{nom}, pas", x.get("pas"), 0, strict=True, maxi=2000)}


def _section(nom: str, x, deux_groupes: bool) -> dict:
    _exiger(isinstance(x, dict), f"{nom} : données manquantes")
    lits = _deux_lits if deux_groupes else _lit
    return {"b": _nombre(f"{nom}, largeur", x.get("b"), 0, strict=True, maxi=1000),
            "h": _nombre(f"{nom}, hauteur", x.get("h"), 0, strict=True, maxi=1000),
            "inf": lits(f"{nom}, armatures inférieures", x.get("inf")),
            "sup": lits(f"{nom}, armatures supérieures", x.get("sup"))}


def _charges_simples(x) -> dict:
    _exiger(isinstance(x, dict), "charges de la grue : données manquantes")
    return {"P": _nombre("charge verticale par pied", x.get("P"), 0, strict=True),
            "H_hs": _nombre("charge horizontale hors service", x.get("H_hs"), 0),
            "H_s": _nombre("charge horizontale en service", x.get("H_s"), 0),
            "M_s": _nombre("moment en service", x.get("M_s"), 0)}


def _pourcent(nom: str, x) -> float:
    return _nombre(nom, x, 0, strict=True, maxi=100)


def normaliser(f: dict) -> dict:
    """Copie complète et contrôlée d'une fondation. Une clé absente prend la valeur du classeur de son schéma ; une
    valeur hors du domaine lève ValueError avec un message à afficher tel quel."""
    _exiger(isinstance(f, dict), "fondation illisible")
    type_ = f.get("type", SEMELLES8)
    _exiger(type_ in TYPES, f"schéma « {type_} » inconnu : {', '.join(TYPES)}")
    base = nouvelle_fondation(type_)
    inconnues = sorted(set(f) - set(base))
    _exiger(not inconnues, f"champ inconnu : {', '.join(inconnues)}")
    src = dict(base)
    src.update(f)
    out = {"type": type_, "nom": _texte("nom", src["nom"], NOM_MAX) or base["nom"]}
    for cle in CHAMPS_TEXTE:
        if cle in base:
            out[cle] = _texte(cle, src[cle])
    out["beton"] = D.nom_beton(src["beton"])
    _exiger(src["acier"] in D.ACIERS, f"acier « {src['acier']} » inconnu : {', '.join(D.ACIERS)}")
    out["acier"] = src["acier"]
    out["chassis"] = _nombre("largeur du châssis", src["chassis"], 0, strict=True, maxi=3000)
    _exiger(isinstance(src["pieux"], dict), "pieux : données manquantes")
    Dp = _nombre("diamètre des pieux", src["pieux"].get("D"), 0, strict=True, maxi=500)
    _exiger(isinstance(src["renfort"], dict), "renfort au droit du pieu : données manquantes")
    renfort = {"brins": _entier("renfort au pieu, brins", src["renfort"].get("brins"), 0, 80),
               "d": _diametre("renfort au pieu, Ø", src["renfort"].get("d"))}
    perfos = _lit("perfos", src["perfos"])

    if type_ == CADRE4:
        po = _section("poutre", src["poutre"], False)
        po["part_V"] = _pourcent("part de V reprise par la poutre", src["poutre"].get("part_V"))
        po["etriers_appui"] = _cadres("étriers à l'appui", src["poutre"].get("etriers_appui"))
        po["etriers_travee"] = _cadres("étriers en travée", src["poutre"].get("etriers_travee"))
        out["poutre"] = po
        out["pieux"] = {"D": Dp, "capacite": _nombre("capacité portante d'un pieu", src["pieux"].get("capacite"), 0, strict=True)}
        out["charges"] = _charges_simples(src["charges"])
        out["e"] = _nombre("excentricité e", src["e"], 0)
        _exiger(out["e"] < out["chassis"], "excentricité e : doit rester inférieure à la largeur du châssis")
        _exiger(po["h"] * 10 - 75 > 0, "poutre : hauteur trop faible (hauteur utile h − 75 mm ≤ 0)")
    elif type_ == DIAGONALES8:
        se = _section("poutre secondaire", src["secondaire"], False)
        se["part_V"] = _pourcent("part de V reprise par la poutre secondaire", src["secondaire"].get("part_V"))
        se["etriers"] = _cadres("étriers de la poutre secondaire", src["secondaire"].get("etriers"))
        pr = _section("poutre primaire", src["primaire"], False)
        pr["L"] = _nombre("poutre primaire, longueur", src["primaire"].get("L"), 0, strict=True, maxi=3000)
        pr["etriers"] = _cadres("étriers de la poutre primaire", src["primaire"].get("etriers"))
        out["secondaire"], out["primaire"] = se, pr
        out["pieux"] = {"entraxe": _nombre("entredistance des pieux", src["pieux"].get("entraxe"), 0, strict=True, maxi=3000),
                        "D": Dp}
        out["charges"] = _charges_simples(src["charges"])
        out["e_secondaire"] = _nombre("excentricité de la poutre secondaire", src["e_secondaire"], 0)
        out["e_primaire"] = _nombre("excentricité de la poutre primaire", src["e_primaire"], 0)
        _exiger(out["e_secondaire"] < out["chassis"],
                "excentricité de la poutre secondaire : doit rester inférieure à la largeur du châssis")
        _exiger(out["e_primaire"] < out["pieux"]["entraxe"] / 2,
                "excentricité de la poutre primaire : doit rester inférieure à la moitié de l'entredistance des pieux "
                "(la charge sortirait de la portée)")
        _exiger(pr["L"] >= out["pieux"]["entraxe"] + out["e_primaire"],
                "poutre primaire : sa longueur doit couvrir l'entredistance des pieux + l'excentricité")
        _exiger(se["h"] * 10 - 100 > 0 and pr["h"] * 10 - 75 > 0, "poutres : hauteur trop faible (hauteur utile ≤ 0)")
    else:
        out["enrobage"] = _nombre("enrobage", src["enrobage"], 0, maxi=200)
        _exiger(src["cote_b"] in COTES_B, f"appui du côté b : {' ou '.join(COTES_B)}")
        out["cote_b"] = src["cote_b"]
        su = _section("poutre support châssis", src["support"], True)
        su["part_V"] = _pourcent("part de V reprise par la poutre support", src["support"].get("part_V"))
        su["etriers_extremites"] = _cadres("étriers aux extrémités", src["support"].get("etriers_extremites"))
        su["etriers_centre"] = _cadres("étriers en partie centrale", src["support"].get("etriers_centre"))
        sm = _section("semelle sur pieux", src["semelle"], True)
        sm["L"] = _nombre("semelle sur pieux, longueur", src["semelle"].get("L"), 0, strict=True, maxi=3000)
        sm["etriers"] = _cadres("étriers de la semelle", src["semelle"].get("etriers"))
        out["support"], out["semelle"] = su, sm
        out["pieux"] = {"entraxe": _nombre("entredistance des pieux", src["pieux"].get("entraxe"), 0, strict=True, maxi=3000),
                        "D": Dp,
                        "capacite": _nombre("capacité portante d'un pieu", src["pieux"].get("capacite"), 0, strict=True)}
        out["dist_a"] = _nombre("distance a", src["dist_a"], 0, maxi=1000)
        out["dist_b"] = _nombre("distance b", src["dist_b"], 0, maxi=1000)
        ch = src["charges"]
        _exiger(isinstance(ch, dict) and isinstance(ch.get("cas"), list) and 1 <= len(ch["cas"]) <= N_CAS_MAX,
                f"charges de la grue : de 1 à {N_CAS_MAX} cas")
        cas = []
        for i, c in enumerate(ch["cas"]):
            _exiger(isinstance(c, dict), f"cas {i + 1} : données manquantes")
            _exiger(c.get("etat", SERVICE) in ETATS_CAS, f"cas {i + 1} : en service ou hors service")
            ligne = {"nom": _texte(f"cas {i + 1}, nom", c.get("nom"), 20) or f"CAS {i + 1}", "etat": c.get("etat", SERVICE)}
            for pied in PIEDS:
                ligne[pied] = _nombre(f"cas {i + 1}, charge verticale {pied}", c.get(pied), 0)
            cas.append(ligne)
        _exiger(any(c[p] > 0 for c in cas for p in PIEDS), "charges de la grue : toutes les charges verticales sont nulles")
        out["charges"] = {"cas": cas, "H_s": _nombre("charge horizontale en service", ch.get("H_s"), 0),
                          "H_hs": _nombre("charge horizontale hors service", ch.get("H_hs"), 0),
                          "M_s": _nombre("moment en service", ch.get("M_s"), 0)}
        out["e_support"] = _nombre("excentricité de la poutre support", src["e_support"], 0)
        out["e_semelle"] = _nombre("excentricité de la semelle", src["e_semelle"], 0)
        out["c_appui"] = _nombre("longueur d'appui de la charge", src["c_appui"], 0, strict=True)
        _exiger(out["dist_a"] - out["e_support"] >= 0 and out["dist_b"] - out["e_support"] >= 0,
                "excentricité de la poutre support : supérieure à la distance a ou b — le pied de la grue sortirait de "
                "la portée, ce que le classeur ne calcule pas")
        s, e, c, Ls = out["pieux"]["entraxe"], out["e_semelle"], out["c_appui"], sm["L"]
        _exiger(s / 2 - e - c / 2 >= 0,
                "semelle : entredistance des pieux trop courte pour l'excentricité et la longueur d'appui "
                "(la charge répartie sortirait de la portée : s/2 − e − c/2 < 0)")
        _exiger(Ls >= s + e, "semelle : sa longueur doit couvrir l'entredistance des pieux + l'excentricité")
        renfort["brins_base"] = _entier("renfort au pieu, brins des étriers de base", src["renfort"].get("brins_base"), 0, 80)
        for nom, sec, et in (("poutre support", su, max(su["etriers_extremites"]["d"], su["etriers_centre"]["d"])),
                             ("semelle", sm, sm["etriers"]["d"])):
            d = sec["h"] * 10 - out["enrobage"] - et - max(g["d"] if g["n"] > 0 else 0 for g in sec["inf"]) / 2
            _exiger(d - out["enrobage"] > 0, f"{nom} : hauteur trop faible pour l'enrobage et les armatures (hauteur utile ≤ 0)")
    out["renfort"], out["perfos"] = renfort, perfos
    return out

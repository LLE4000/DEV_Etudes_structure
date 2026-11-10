# -*- coding: utf-8 -*-
"""Taux d'armature — l'ENTRÉE : la poutre TYPE d'une note neuve, les listes, les bornes des champs et la normalisation
STRICTE d'une poutre (docs/site/LOT_Q_QUATRE_MODULES.md § 3 ; docs/beton/TAUX_ARMATURE_MOTEUR.md).

Une poutre (cm, m, mm — l'unité de chaque champ est écrite à côté de sa borne, ``BORNES``) :

    {"nom": "P1",
     "forme": "rectangulaire" | "talon_gauche" | "talon_droit" | "deux_talons",
     "b": 30.0, "h": 60.0,              l'âme (cm)            "L": 6.0   la longueur (m)      "enrobage": 3.0 (cm)
     "talon_gauche": {"b": 15.0, "h": 20.0}, "talon_droit": {...},    les talons BAS, à l'extérieur de l'âme (cm) ; gardés
                                                                      quelle que soit la forme, lus selon la forme
     "barres": [{"famille": "inf" | "sup" | "peau",
                 "lit": 1,              le rang du lit sur sa face (1 : contre le cadre) ; None pour la peau
                 "n": 3,                le nombre de barres (PAR FACE pour la peau)
                 "d": 20,               Ø (mm)
                 "de": None, "a": None, les abscisses des bouts (m, depuis l'about gauche) ; None : jusqu'à l'about — les
                                        deux à None : une barre FILANTE ; sinon un RENFORT
                 "retours": 0 | 1 | 2,  le nombre de bouts relevés (retours verticaux)
                 "l_retour": None}],    la longueur d'un retour (cm) ; None : automatique, h − 2c (b − 2c pour la peau)
     "types_etriers": [{"nom": "E1",
                        "lignes": [{"forme": "cadre" | "cadre_interieur" | "epingle", "d": 8,
                                    "largeur": None}],      la largeur hors tout d'un cadre intérieur (cm) ; None sinon
                        "cadre_talon": True}],              un cadre de talon à chaque cours (formes à talon seulement)
     "zones": [{"de": 0.0, "a": 1.5, "s": 15.0, "type": "E1"}],       (m, m, cm) — rangées de gauche à droite
     "majoration": 5.0,                 chutes et ligatures (%)
     "recouvrements": {"actifs": True, "l_com": 12.0, "k": 50.0},      longueur commerciale (m), recouvrement de k × Ø
     "rho_beton": 2500.0, "rho_acier": 7850.0,                         (kg/m³)
     "origine": None | {"module": "poutre", "nom": "V1", "date": "10/11/2026"}}     d'où la poutre a été reçue

Règle de ``normaliser`` : une clé ABSENTE (ou ``null``, là où ``null`` n'a pas de sens propre) prend sa valeur de
``DEFAUTS`` ; une clé INCONNUE, un type faux ou une valeur HORS DES BORNES lèvent ``ValueError`` avec un message en
français qui nomme le champ. Jamais une valeur ramenée à une borne. Les zones d'étriers sont rendues rangées de gauche
à droite (elles ne se recouvrent pas : l'ordre est sans ambiguïté) ; les barres gardent leur ordre, qui est celui de
leurs repères.
"""
import copy
import math

# ---- les formes de section : la clé → son libellé ; les talons que chaque forme porte
FORMES = ("rectangulaire", "talon_gauche", "talon_droit", "deux_talons")
LIBELLES_FORME = {"rectangulaire": "rectangulaire", "talon_gauche": "talon à gauche", "talon_droit": "talon à droite",
                  "deux_talons": "deux talons"}
TALONS = {"rectangulaire": (), "talon_gauche": ("talon_gauche",), "talon_droit": ("talon_droit",),
          "deux_talons": ("talon_gauche", "talon_droit")}
LIBELLES_TALON = {"talon_gauche": "talon gauche", "talon_droit": "talon droit"}

# ---- les barres longitudinales
FAMILLES = ("inf", "sup", "peau")
LIBELLES_FAMILLE = {"inf": "inférieur", "sup": "supérieur", "peau": "de peau"}
RETOURS = (0, 1, 2)
# Ø des barres (mm) : la série des aciers à béton (EN 10080 ; NBN A 24-302)
DIAMETRES = (6, 8, 10, 12, 14, 16, 20, 25, 28, 32, 40)

# ---- les étriers
FORMES_LIGNE = ("cadre", "cadre_interieur", "epingle")
LIBELLES_LIGNE = {"cadre": "cadre", "cadre_interieur": "cadre intérieur", "epingle": "épingle"}
# Ø des cadres et des épingles (mm) : au-delà de 20, un acier ne se façonne plus en cadre de poutre
DIAMETRES_ETRIERS = (6, 8, 10, 12, 14, 16, 20)

# ---- les bornes des champs : larges, mais réelles (min, max, unité)
BORNES = {
    "b": (10.0, 300.0, "cm"),
    "h": (10.0, 300.0, "cm"),
    "L": (0.5, 60.0, "m"),
    "enrobage": (1.0, 15.0, "cm"),
    "talon_b": (5.0, 150.0, "cm"),
    "talon_h": (5.0, 150.0, "cm"),
    "n": (1, 40, ""),
    "lit": (1, 6, ""),
    "l_retour": (0.0, 300.0, "cm"),
    "largeur": (5.0, 300.0, "cm"),
    "s": (5.0, 50.0, "cm"),
    "majoration": (0.0, 50.0, "%"),
    "l_com": (6.0, 18.0, "m"),
    "k": (20.0, 100.0, "Ø"),
    "rho_beton": (1500.0, 3000.0, "kg/m³"),
    "rho_acier": (7700.0, 8100.0, "kg/m³"),
}
MAX_BARRES, MAX_TYPES, MAX_LIGNES, MAX_ZONES = 40, 12, 8, 40
NOM_MAX = 60          # caractères du nom d'une poutre
NOM_TYPE_MAX = 16     # … du nom d'un type d'étriers
# deux abscisses à moins d'un micromètre sont la MÊME (deux zones jointives, une barre de longueur nulle) : les saisies
# sont des décimaux, leurs sommes en flottants diffèrent de 1e-16
EPS_X = 1e-6          # m
ORIGINES = ("poutre",)   # les modules qui savent envoyer une poutre (docs/site/LOT_Q_QUATRE_MODULES.md § 6.2)

# ---- les valeurs par défaut d'une clé absente (les scalaires de la poutre type ; des listes VIDES : une poutre lue
#      sans barres n'en reçoit pas)
DEFAUTS = {"nom": "P1", "forme": "rectangulaire", "b": 30.0, "h": 60.0, "L": 6.0, "enrobage": 3.0, "majoration": 5.0,
           "rho_beton": 2500.0, "rho_acier": 7850.0}
TALON_DEFAUT = {"b": 15.0, "h": 20.0}
RECOUVREMENTS_DEFAUT = {"actifs": True, "l_com": 12.0, "k": 50.0}
BARRE_DEFAUT = {"famille": "inf", "lit": 1, "n": 2, "d": 16, "de": None, "a": None, "retours": 0, "l_retour": None}
LIGNE_DEFAUT = {"forme": "cadre", "d": 8, "largeur": None}
ZONE_DEFAUT = {"s": 15.0}

CLES_POUTRE = frozenset(DEFAUTS) | {"talon_gauche", "talon_droit", "barres", "types_etriers", "zones", "recouvrements", "origine"}
CLES_TALON = frozenset(TALON_DEFAUT)
CLES_BARRE = frozenset(BARRE_DEFAUT)
CLES_TYPE = frozenset({"nom", "lignes", "cadre_talon"})
CLES_LIGNE = frozenset(LIGNE_DEFAUT)
CLES_ZONE = frozenset({"de", "a", "s", "type"})
CLES_RECOUVREMENTS = frozenset(RECOUVREMENTS_DEFAUT)
CLES_ORIGINE = frozenset({"module", "nom", "date"})


# ============================================================ la poutre type
def nouvelle_poutre(nom: str = "P1") -> dict:
    """La poutre TYPE d'une note neuve (conception § 3.5) : 30 × 60 cm, L = 6,00 m, c = 3 cm ; 3 Ø 20 inférieures
    filantes, un renfort 2 Ø 16 de 1,20 à 4,80 m en second lit, 2 Ø 12 supérieures filantes ; des cadres Ø 8 au pas de
    15 cm sur 1,50 m à chaque about et de 25 cm en travée ; majoration de 5 %, recouvrements de 50 Ø au-delà de 12 m."""
    return {
        "nom": nom, "forme": "rectangulaire", "b": 30.0, "h": 60.0, "L": 6.0, "enrobage": 3.0,
        "talon_gauche": dict(TALON_DEFAUT), "talon_droit": dict(TALON_DEFAUT),
        "barres": [
            {"famille": "inf", "lit": 1, "n": 3, "d": 20, "de": None, "a": None, "retours": 0, "l_retour": None},
            {"famille": "inf", "lit": 2, "n": 2, "d": 16, "de": 1.2, "a": 4.8, "retours": 0, "l_retour": None},
            {"famille": "sup", "lit": 1, "n": 2, "d": 12, "de": None, "a": None, "retours": 0, "l_retour": None},
        ],
        "types_etriers": [{"nom": "E1", "lignes": [{"forme": "cadre", "d": 8, "largeur": None}], "cadre_talon": True}],
        "zones": [{"de": 0.0, "a": 1.5, "s": 15.0, "type": "E1"}, {"de": 1.5, "a": 4.5, "s": 25.0, "type": "E1"},
                  {"de": 4.5, "a": 6.0, "s": 15.0, "type": "E1"}],
        "majoration": 5.0, "recouvrements": dict(RECOUVREMENTS_DEFAUT), "rho_beton": 2500.0, "rho_acier": 7850.0,
        "origine": None,
    }


# ============================================================ ce que la forme porte, l'étendue d'une barre
def talons(forme: str) -> tuple:
    """Les talons d'une forme : (« talon_gauche », « talon_droit ») pour deux talons, () pour un rectangle."""
    return TALONS[forme]


def a_talon(forme: str) -> bool:
    return bool(TALONS[forme])


def etendue(barre: dict, L: float, enrobage_cm: float) -> tuple:
    """(x₁, x₂) en m : les bouts d'une barre, BORNÉS aux abouts à l'enrobage — x₁ = max(de ; c), x₂ = min(à ; L − c),
    « de » et « à » absents valant l'about (0 et L). Une barre filante mesure donc L − 2c."""
    c = enrobage_cm / 100.0
    de = 0.0 if barre.get("de") is None else barre["de"]
    a = L if barre.get("a") is None else barre["a"]
    return max(de, c), min(a, L - c)


def filante(barre: dict) -> bool:
    """Une barre FILANTE court d'un about à l'autre (« de » et « à » vides) ; sinon c'est un renfort."""
    return barre.get("de") is None and barre.get("a") is None


# ============================================================ les mots des messages
def nombre_fr(x) -> str:
    """Un nombre dans un message : la virgule, sans zéro inutile (6 ; 1,5 ; 0,025)."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return (f"{x:.6f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)).replace(".", ",")


def libelle_barres(n: int, d: int) -> str:
    """« 3 Ø 20 »."""
    return f"{int(n)} Ø {int(d)}"


def nom_zone(i: int) -> str:
    """Le nom d'une zone d'étriers : son rang de gauche à droite, « Z1 », « Z2 »…"""
    return f"Z{i + 1}"


# ============================================================ contrôle et normalisation
def _cles(d, permises: frozenset, chemin: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{chemin} : un objet est attendu, reçu {type(d).__name__}")
    inconnues = sorted(str(k) for k in d if k not in permises)
    if inconnues:
        raise ValueError(f"{chemin} : clé(s) inconnue(s) {', '.join(inconnues)} (clés admises : {', '.join(sorted(permises))})")


def _val(d: dict, cle: str, defaut):
    v = d.get(cle)
    return defaut if v is None else v


def _nombre(x, chemin: str, borne: str) -> float:
    """Un nombre fini dans les bornes du champ (``BORNES[borne]``) ; un booléen n'est pas un nombre."""
    lo, hi, u = BORNES[borne]
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{chemin} : un nombre est attendu, reçu {x!r}")
    if x < lo or x > hi:
        u_ = f" {u}" if u else ""
        raise ValueError(f"{chemin} : {nombre_fr(float(x))}{u_} hors du domaine (de {nombre_fr(float(lo))} à {nombre_fr(float(hi))}{u_})")
    return float(x)


def _entier(x, chemin: str, borne: str) -> int:
    lo, hi, _ = BORNES[borne]
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x != int(x):
        raise ValueError(f"{chemin} : un nombre entier est attendu, reçu {x!r}")
    if not lo <= x <= hi:
        raise ValueError(f"{chemin} : {int(x)} hors du domaine (de {lo} à {hi})")
    return int(x)


def _diametre(x, chemin: str, permis: tuple) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x not in permis:
        raise ValueError(f"{chemin} : Ø {x!r} non permis (Ø {', '.join(map(str, permis))} mm)")
    return int(x)


def _texte(x, chemin: str, maxi: int) -> str:
    if not isinstance(x, str):
        raise ValueError(f"{chemin} : un texte est attendu, reçu {x!r}")
    t = x.strip()
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in t):
        raise ValueError(f"{chemin} : un texte d'une seule ligne est attendu (caractère de contrôle refusé)")
    if len(t) > maxi:
        raise ValueError(f"{chemin} : {len(t)} caractères, {maxi} au plus")
    return t


def _booleen(x, chemin: str) -> bool:
    if not isinstance(x, bool):
        raise ValueError(f"{chemin} : vrai ou faux est attendu, reçu {x!r}")
    return x


def _choix(x, chemin: str, permis: tuple, libelles=None) -> str:
    if not isinstance(x, str) or x not in permis:
        dits = ", ".join(permis if libelles is None else (f"{k} ({libelles[k]})" for k in permis))
        raise ValueError(f"{chemin} : {x!r} inconnu — {dits}")
    return x


def _liste(d: dict, cle: str, chemin: str, maxi: int) -> list:
    v = _val(d, cle, [])
    if not isinstance(v, list):
        raise ValueError(f"{chemin} : une liste est attendue, reçu {type(v).__name__}")
    if len(v) > maxi:
        raise ValueError(f"{chemin} : {len(v)} éléments, {maxi} au plus")
    return v


def _abscisse(x, chemin: str, L: float) -> float:
    """Une abscisse (m) sur la poutre : de 0 à L."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{chemin} : un nombre est attendu, reçu {x!r}")
    if x < -EPS_X or x > L + EPS_X:
        raise ValueError(f"{chemin} : {nombre_fr(float(x))} m hors de la poutre (de 0 à {nombre_fr(L)} m)")
    return min(max(float(x), 0.0), L)


def _barre(b, i: int, p: dict) -> dict:
    ch = f"barres n° {i + 1}"
    _cles(b, CLES_BARRE, ch)
    fam = _choix(_val(b, "famille", BARRE_DEFAUT["famille"]), f"{ch} · famille", FAMILLES, LIBELLES_FAMILLE)
    out = {"famille": fam,
           "lit": None if fam == "peau" else _entier(_val(b, "lit", BARRE_DEFAUT["lit"]), f"{ch} · lit", "lit"),
           "n": _entier(_val(b, "n", BARRE_DEFAUT["n"]), f"{ch} · nombre de barres", "n"),
           "d": _diametre(_val(b, "d", BARRE_DEFAUT["d"]), f"{ch} · Ø", DIAMETRES)}
    for k, quoi in (("de", "« de »"), ("a", "« à »")):
        out[k] = None if b.get(k) is None else _abscisse(b[k], f"{ch} · {quoi}", p["L"])
    if out["de"] is not None and out["a"] is not None and out["a"] <= out["de"] + EPS_X:
        raise ValueError(f"{ch} : « à » ({nombre_fr(out['a'])} m) doit dépasser « de » ({nombre_fr(out['de'])} m)")
    x1, x2 = etendue(out, p["L"], p["enrobage"])
    if x2 - x1 <= EPS_X:
        raise ValueError(f"{ch} : de {nombre_fr(0.0 if out['de'] is None else out['de'])} à "
                         f"{nombre_fr(p['L'] if out['a'] is None else out['a'])} m, la barre est tout entière dans l'enrobage d'about "
                         f"({nombre_fr(p['enrobage'])} cm) — elle n'a aucune longueur")
    r = _val(b, "retours", 0)
    if isinstance(r, bool) or r not in RETOURS:
        raise ValueError(f"{ch} · retours : {r!r} — 0, 1 ou 2 bouts relevés")
    out["retours"] = int(r)
    out["l_retour"] = None if b.get("l_retour") is None else _nombre(b["l_retour"], f"{ch} · longueur d'un retour", "l_retour")
    return out


def _ligne(li, ch: str, p: dict) -> dict:
    _cles(li, CLES_LIGNE, ch)
    forme = _choix(_val(li, "forme", LIGNE_DEFAUT["forme"]), f"{ch} · forme", FORMES_LIGNE, LIBELLES_LIGNE)
    out = {"forme": forme, "d": _diametre(_val(li, "d", LIGNE_DEFAUT["d"]), f"{ch} · Ø", DIAMETRES_ETRIERS), "largeur": None}
    if forme == "cadre_interieur":
        if li.get("largeur") is None:
            raise ValueError(f"{ch} · largeur : un cadre intérieur a une largeur (cm)")
        w = _nombre(li["largeur"], f"{ch} · largeur", "largeur")
        libre = p["b"] - 2.0 * p["enrobage"]
        if w > libre + 1e-9:
            raise ValueError(f"{ch} · largeur : {nombre_fr(w)} cm dépasse le cadre de l'âme (b − 2c = {nombre_fr(round(libre, 6))} cm)")
        out["largeur"] = w
    return out


def _type(t, i: int, p: dict) -> dict:
    ch = f"type d'étriers n° {i + 1}"
    _cles(t, CLES_TYPE, ch)
    nom = _texte(_val(t, "nom", ""), f"{ch} · nom", NOM_TYPE_MAX)
    if not nom:
        raise ValueError(f"{ch} · nom : un type d'étriers a un nom (les zones le désignent par lui)")
    lignes = _liste(t, "lignes", f"type « {nom} » · lignes", MAX_LIGNES)
    if not lignes:
        raise ValueError(f"type « {nom} » : il lui faut au moins une ligne (un cadre, un cadre intérieur ou une épingle)")
    return {"nom": nom, "lignes": [_ligne(li, f"type « {nom} » · ligne {j + 1}", p) for j, li in enumerate(lignes)],
            "cadre_talon": _booleen(_val(t, "cadre_talon", False), f"type « {nom} » · cadre de talon")}


def _zone(z, i: int, p: dict, noms: list) -> dict:
    ch = f"zone {nom_zone(i)}"
    _cles(z, CLES_ZONE, ch)
    for k in ("de", "a", "type"):
        if z.get(k) is None:
            raise ValueError(f"{ch} : « {'à' if k == 'a' else k} » manque")
    out = {"de": _abscisse(z["de"], f"{ch} · « de »", p["L"]), "a": _abscisse(z["a"], f"{ch} · « à »", p["L"]),
           "s": _nombre(_val(z, "s", ZONE_DEFAUT["s"]), f"{ch} · pas", "s")}
    if out["a"] <= out["de"] + EPS_X:
        raise ValueError(f"{ch} : « à » ({nombre_fr(out['a'])} m) doit dépasser « de » ({nombre_fr(out['de'])} m)")
    if not isinstance(z["type"], str) or z["type"] not in noms:
        raise ValueError(f"{ch} · type : « {z['type']} » n'est pas un type d'étriers de la poutre"
                         + (f" ({', '.join(noms)})" if noms else " (elle n'en a aucun)"))
    out["type"] = z["type"]
    return out


def _origine(o) -> dict:
    _cles(o, CLES_ORIGINE, "origine")
    module = _val(o, "module", "")
    if module not in ORIGINES:
        raise ValueError(f"origine · module : {module!r} inconnu ({', '.join(ORIGINES)})")
    return {"module": module, "nom": _texte(_val(o, "nom", ""), "origine · nom", NOM_MAX),
            "date": _texte(_val(o, "date", ""), "origine · date", 10)}


def normaliser(poutre: dict) -> dict:
    """La poutre complétée (clé absente → ``DEFAUTS``) et CONTRÔLÉE : ``ValueError`` en français, qui nomme le champ,
    pour une clé inconnue, un type faux ou une valeur hors du domaine. Rend une COPIE ; idempotente."""
    _cles(poutre, CLES_POUTRE, "poutre")
    p = {"nom": _texte(_val(poutre, "nom", DEFAUTS["nom"]), "nom", NOM_MAX),
         "forme": _choix(_val(poutre, "forme", DEFAUTS["forme"]), "forme", FORMES, LIBELLES_FORME)}
    for k, quoi in (("b", "largeur b"), ("h", "hauteur h"), ("L", "longueur L"), ("enrobage", "enrobage c")):
        p[k] = _nombre(_val(poutre, k, DEFAUTS[k]), quoi, k)
    for k, quoi in (("b", "largeur b"), ("h", "hauteur h")):
        if 2.0 * p["enrobage"] >= p[k]:
            raise ValueError(f"enrobage c : 2 × {nombre_fr(p['enrobage'])} cm ne laisse aucun cadre dans la {quoi} = {nombre_fr(p[k])} cm")
    for cote in ("talon_gauche", "talon_droit"):
        t = _val(poutre, cote, TALON_DEFAUT)
        _cles(t, CLES_TALON, LIBELLES_TALON[cote])
        p[cote] = {"b": _nombre(_val(t, "b", TALON_DEFAUT["b"]), f"{LIBELLES_TALON[cote]} · largeur", "talon_b"),
                   "h": _nombre(_val(t, "h", TALON_DEFAUT["h"]), f"{LIBELLES_TALON[cote]} · hauteur", "talon_h")}
    for cote in talons(p["forme"]):
        if p[cote]["h"] >= p["h"]:
            raise ValueError(f"{LIBELLES_TALON[cote]} · hauteur : {nombre_fr(p[cote]['h'])} cm — un talon est moins haut que la "
                             f"poutre (h = {nombre_fr(p['h'])} cm)")
    p["barres"] = [_barre(b, i, p) for i, b in enumerate(_liste(poutre, "barres", "barres", MAX_BARRES))]
    p["types_etriers"] = [_type(t, i, p) for i, t in enumerate(_liste(poutre, "types_etriers", "types d'étriers", MAX_TYPES))]
    noms = [t["nom"] for t in p["types_etriers"]]
    for i, n in enumerate(noms):
        if n in noms[:i]:
            raise ValueError(f"types d'étriers : deux types s'appellent « {n} » (les zones les désignent par leur nom)")
    if talons(p["forme"]) and any(t["cadre_talon"] for t in p["types_etriers"]):
        ht = max(p[cote]["h"] for cote in talons(p["forme"]))
        if 2.0 * p["enrobage"] >= ht:
            raise ValueError(f"cadre de talon : 2 × {nombre_fr(p['enrobage'])} cm d'enrobage ne laissent aucun cadre dans un talon de "
                             f"{nombre_fr(ht)} cm de haut (augmenter le talon, ou décocher « cadre de talon »)")
    zones = [_zone(z, i, p, noms) for i, z in enumerate(_liste(poutre, "zones", "zones d'étriers", MAX_ZONES))]
    zones.sort(key=lambda z: (z["de"], z["a"]))
    for i in range(len(zones) - 1):
        u, v = zones[i], zones[i + 1]
        if v["de"] < u["a"] - EPS_X:
            raise ValueError(f"zones d'étriers : {nom_zone(i)} (de {nombre_fr(u['de'])} à {nombre_fr(u['a'])} m) et {nom_zone(i + 1)} "
                             f"(de {nombre_fr(v['de'])} à {nombre_fr(v['a'])} m) se recouvrent de {nombre_fr(v['de'])} à "
                             f"{nombre_fr(min(u['a'], v['a']))} m — les étriers y seraient comptés deux fois")
    p["zones"] = zones
    p["majoration"] = _nombre(_val(poutre, "majoration", DEFAUTS["majoration"]), "majoration", "majoration")
    r = _val(poutre, "recouvrements", RECOUVREMENTS_DEFAUT)
    _cles(r, CLES_RECOUVREMENTS, "recouvrements")
    p["recouvrements"] = {"actifs": _booleen(_val(r, "actifs", RECOUVREMENTS_DEFAUT["actifs"]), "recouvrements · comptés"),
                          "l_com": _nombre(_val(r, "l_com", RECOUVREMENTS_DEFAUT["l_com"]), "recouvrements · longueur commerciale", "l_com"),
                          "k": _nombre(_val(r, "k", RECOUVREMENTS_DEFAUT["k"]), "recouvrements · longueur en Ø", "k")}
    for k, quoi in (("rho_beton", "masse volumique du béton"), ("rho_acier", "masse volumique de l'acier")):
        p[k] = _nombre(_val(poutre, k, DEFAUTS[k]), quoi, k)
    p["origine"] = None if poutre.get("origine") is None else _origine(poutre["origine"])
    return p


def copie(poutre: dict) -> dict:
    return copy.deepcopy(poutre)

# -*- coding: utf-8 -*-
"""Taux d'armature — les GESTES de la saisie : ce que l'écran propose quand on ajoute une ligne de barres, un type
d'étriers, une ligne à un type, une zone ; ce qui SUIT quand on déplace la limite de deux zones jointives ou qu'on
change la longueur de la poutre ; ce qui est refusé (retirer un type qu'une zone emploie).

Chaque fonction modifie la poutre reçue (déjà normalisée) et NE la renormalise pas : l'appelant le fait
(``entree.normaliser``), et c'est lui qui refuse ce qui sort du domaine. Les fonctions qui déplacent AUTRE CHOSE que la
valeur saisie rendent la liste de ce qui a suivi, en mots, pour que l'écran le DISE — rien de silencieux.
"""
from typing import List

from . import entree as EN


def _libre(prefixe: str, pris) -> str:
    k = len(pris) + 1
    while f"{prefixe}{k}" in pris:
        k += 1
    return f"{prefixe}{k}"


def _mm(x: float) -> float:
    """Une abscisse gardée au millimètre (les déplacements calculés ne laissent pas de 5,700000000000001)."""
    return round(x, 3)


# ============================================================ les barres
SORTES_BARRE = ("inf", "sup", "renfort", "peau")


def ajouter_barre(p: dict, sorte: str) -> dict:
    """« + Lit inférieur », « + Lit supérieur » : un lit FILANT de plus sur la face, posé au-dessus du dernier (son rang
    + 1), avec les barres du lit précédent (2 Ø 16 sur une face vide). « + Renfort » : 2 Ø 16 en second lit inférieur
    (premier lit s'il n'y en a pas), du cinquième aux quatre cinquièmes de la longueur. « + Barres de peau » : 1 Ø 10
    par face, filantes."""
    if sorte not in SORTES_BARRE:
        raise ValueError(f"barres : sorte {sorte!r} inconnue ({', '.join(SORTES_BARRE)})")
    if len(p["barres"]) >= EN.MAX_BARRES:
        raise ValueError(f"barres : {EN.MAX_BARRES} lignes au plus")
    b = dict(EN.BARRE_DEFAUT)
    if sorte in ("inf", "sup"):
        face = [x for x in p["barres"] if x["famille"] == sorte and EN.filante(x)]
        b.update(famille=sorte, lit=1)
        if face:
            dernier = max(face, key=lambda x: x["lit"])
            if dernier["lit"] >= EN.BORNES["lit"][1]:
                raise ValueError(f"barres : {EN.BORNES['lit'][1]} lits au plus sur une face")
            b.update(lit=dernier["lit"] + 1, n=dernier["n"], d=dernier["d"])
    elif sorte == "renfort":
        a_un_lit = any(x["famille"] == "inf" and x["lit"] == 1 for x in p["barres"])
        b.update(famille="inf", lit=2 if a_un_lit else 1, de=_mm(0.2 * p["L"]), a=_mm(0.8 * p["L"]))
    else:
        b.update(famille="peau", lit=None, n=1, d=10)
    p["barres"].append(b)
    return b


# ============================================================ les types d'étriers
def ajouter_type(p: dict) -> dict:
    """« + Type » : « E<k> », un cadre du Ø du premier type (Ø 8 sans type), avec son cadre de talon."""
    if len(p["types_etriers"]) >= EN.MAX_TYPES:
        raise ValueError(f"types d'étriers : {EN.MAX_TYPES} au plus")
    d = p["types_etriers"][0]["lignes"][0]["d"] if p["types_etriers"] else EN.LIGNE_DEFAUT["d"]
    t = {"nom": _libre("E", [x["nom"] for x in p["types_etriers"]]), "lignes": [{"forme": "cadre", "d": d, "largeur": None}],
         "cadre_talon": True}
    p["types_etriers"].append(t)
    return t


def largeur_interieure(p: dict) -> float:
    """La largeur proposée à un cadre intérieur neuf (cm) : la moitié du cadre de l'âme, au demi-centimètre."""
    return max(EN.BORNES["largeur"][0], round((p["b"] - 2.0 * p["enrobage"]) / 2.0 * 2.0) / 2.0)


def ajouter_ligne(p: dict, i: int) -> dict:
    """« + Ligne » : une épingle du Ø de la première ligne du type (la ligne la plus courante après le cadre)."""
    t = p["types_etriers"][i]
    if len(t["lignes"]) >= EN.MAX_LIGNES:
        raise ValueError(f"type « {t['nom']} » : {EN.MAX_LIGNES} lignes au plus")
    li = {"forme": "epingle", "d": t["lignes"][0]["d"], "largeur": None}
    t["lignes"].append(li)
    return li


def changer_forme_ligne(p: dict, i: int, j: int, forme: str) -> None:
    """La forme d'une ligne : un cadre intérieur neuf reçoit la largeur proposée (``largeur_interieure``) — le champ
    apparaît avec sa valeur."""
    li = p["types_etriers"][i]["lignes"][j]
    li["forme"] = forme
    if forme == "cadre_interieur" and li.get("largeur") is None:
        li["largeur"] = largeur_interieure(p)


def retirer_ligne(p: dict, i: int, j: int) -> None:
    t = p["types_etriers"][i]
    if len(t["lignes"]) <= 1:
        raise ValueError(f"type « {t['nom']} » : il lui faut au moins une ligne (retirer le type lui-même)")
    del t["lignes"][j]


def retirer_type(p: dict, i: int) -> None:
    """Un type qu'une zone emploie ne se retire pas : le refus nomme la ou les zones."""
    nom = p["types_etriers"][i]["nom"]
    zones = [EN.nom_zone(k) for k, z in enumerate(p["zones"]) if z["type"] == nom]
    if zones:
        un = len(zones) == 1
        raise ValueError(f"type « {nom} » : {'la zone' if un else 'les zones'} {', '.join(zones)} l'emploie{'' if un else 'nt'} — "
                         f"{'lui' if un else 'leur'} donner d'abord un autre type, ou {'la' if un else 'les'} retirer")
    del p["types_etriers"][i]


def renommer_type(p: dict, i: int, nom: str) -> None:
    """Un type renommé : les zones qui le désignaient le suivent (c'est le même type)."""
    ancien = p["types_etriers"][i]["nom"]
    nom = str(nom).strip()
    if nom != ancien and nom in [t["nom"] for t in p["types_etriers"]]:
        raise ValueError(f"types d'étriers : « {nom} » existe déjà")
    p["types_etriers"][i]["nom"] = nom
    for z in p["zones"]:
        if z["type"] == ancien:
            z["type"] = nom


# ============================================================ les zones
def trous(p: dict) -> list:
    """Les portions de la poutre SANS zone d'étriers : [(x₁, x₂)], de gauche à droite."""
    out, x = [], 0.0
    for z in sorted(p["zones"], key=lambda z: z["de"]):
        if z["de"] > x + EN.EPS_X:
            out.append((x, z["de"]))
        x = max(x, z["a"])
    if x < p["L"] - EN.EPS_X:
        out.append((x, p["L"]))
    return out


def ajouter_zone(p: dict) -> dict:
    """« + Zone » propose la SUITE : de la fin de la dernière zone à L, au pas et du type de la précédente ; si la
    dernière zone va déjà au bout, la première portion sans étriers ; sans aucun type, « E1 » (un cadre Ø 8) est créé.
    Quand les zones couvrent toute la poutre, il n'y a rien à proposer : refus, qui le dit."""
    if len(p["zones"]) >= EN.MAX_ZONES:
        raise ValueError(f"zones d'étriers : {EN.MAX_ZONES} au plus")
    libres = trous(p)
    if not libres:
        raise ValueError(f"zones d'étriers : elles couvrent déjà toute la poutre (de 0 à {EN.nombre_fr(p['L'])} m) — raccourcir une "
                         "zone pour en ajouter une")
    if not p["types_etriers"]:
        ajouter_type(p)
    zones = sorted(p["zones"], key=lambda z: z["de"])
    de, a = libres[-1] if (zones and abs(libres[-1][1] - p["L"]) <= EN.EPS_X and libres[-1][0] >= zones[-1]["a"] - EN.EPS_X) else libres[0]
    avant = [z for z in zones if z["a"] <= de + EN.EPS_X]
    modele = avant[-1] if avant else (zones[0] if zones else None)
    z = {"de": _mm(de), "a": _mm(a), "s": modele["s"] if modele else EN.ZONE_DEFAUT["s"],
         "type": modele["type"] if modele else p["types_etriers"][0]["nom"]}
    p["zones"].append(z)
    return z


def deplacer_borne(p: dict, i: int, cle: str, x: float) -> List[str]:
    """« de » ou « à » de la zone ``i`` saisi. La zone JOINTIVE de ce côté (celle qui finissait où celle-ci commençait,
    ou qui commençait où celle-ci finissait) SUIT : c'est la limite des deux zones qu'on déplace — sans quoi il
    faudrait d'abord ouvrir un trou, ou la saisie serait refusée pour recouvrement. Rend ce qui a suivi, en mots."""
    if cle not in ("de", "a"):
        raise ValueError(f"zone : borne {cle!r} inconnue")
    z = p["zones"][i]
    ancien = z[cle]
    x = _mm(x)
    suivis = []
    autre = "a" if cle == "de" else "de"
    for k, v in enumerate(p["zones"]):
        if k != i and abs(v[autre] - ancien) <= EN.EPS_X and abs(x - ancien) > EN.EPS_X:
            v[autre] = x
            suivis.append(f"{EN.nom_zone(k)} {'finit' if autre == 'a' else 'commence'} à {EN.nombre_fr(x)} m")
    z[cle] = x
    return suivis


# ============================================================ la longueur de la poutre
def changer_longueur(p: dict, L: float) -> List[str]:
    """Une longueur saisie. Ce qui se compte depuis l'about DROIT le suit : toute abscisse de la moitié droite (au-delà
    de L/2) — la fin d'un renfort, la limite d'une zone — garde sa distance à l'about droit ; celles de la moitié
    gauche gardent leur distance à l'about gauche. La zone de 1,50 m à chaque about reste à chaque about, la zone de
    travée s'allonge ou se raccourcit. Rend ce qui a suivi, en mots (l'appelant normalise : une zone ou une barre
    qui n'aurait plus de longueur est refusée)."""
    L0 = p["L"]
    delta = L - L0
    suivis = []
    if abs(delta) <= EN.EPS_X:
        p["L"] = L
        return suivis

    def suit(x):
        return _mm(x + delta) if x > L0 / 2.0 + EN.EPS_X else x
    for k, z in enumerate(p["zones"]):
        de, a = suit(z["de"]), suit(z["a"])
        if (de, a) != (z["de"], z["a"]):
            suivis.append(f"{EN.nom_zone(k)} de {EN.nombre_fr(de)} à {EN.nombre_fr(a)} m")
        z["de"], z["a"] = de, a
    for k, b in enumerate(p["barres"]):
        avant = (b["de"], b["a"])
        for cle in ("de", "a"):
            if b[cle] is not None:
                b[cle] = suit(b[cle])
        if (b["de"], b["a"]) != avant:
            suivis.append(f"repère {k + 1} de {EN.nombre_fr(0.0 if b['de'] is None else b['de'])} à "
                          f"{EN.nombre_fr(L if b['a'] is None else b['a'])} m")
    p["L"] = L
    return suivis

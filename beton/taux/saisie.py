# -*- coding: utf-8 -*-
"""Taux d'armature — les EFFETS DE BORD de l'écran Streamlit à la saisie (``modules/taux_armature.py``), écrits une
fois ici pour que le site les reproduise : ajouter une famille, un renfort de talon, une zone d'étriers ; changer la
forme, la portée, le nombre de retours ; supprimer ou renommer un type d'étrier. Chaque fonction modifie la poutre
reçue (déjà normalisée) et rend la liste des changements faits « en silence » par l'écran, pour que le site les DISE.

La parité de ces effets avec l'écran est prouvée dans ``tests/test_taux_moteur.py`` (section « effets de bord »).
"""
from typing import List

from . import entree as EN


def ajouter_famille(p: dict, sorte: str) -> dict:
    """Les boutons « + Lit principal », « + Renfort local », « + Latéral », « + Renfort talon » (add_family).
    « + Renfort talon » n'existe que pour une forme à talon (bouton grisé sinon) : ValueError."""
    if sorte == "talon":
        if not EN.renfort_talon_possible(p["forme"]):
            raise ValueError("« Renfort talon » : la section n'a pas de talon")
        f = EN.nouvelle_famille(p, "renfort", EN.position_renfort_talon(p["forme"]))
    elif sorte in EN.ROLES:
        f = EN.nouvelle_famille(p, sorte)
    else:
        raise ValueError(f"famille : sorte {sorte!r} inconnue")
    p["familles"].append(f)
    return f


def ajouter_etrier(p: dict) -> dict:
    """Le bouton « + Étrier » : un type d'étrier s'il n'y en a aucun (E1, Ø8), puis une zone (0 à 33 %, s = 15 cm)."""
    if not p["types_etriers"]:
        p["types_etriers"].append(EN.nouveau_type_etrier(p))
    z = EN.nouvelle_zone(p)
    p["zones"].append(z)
    return z


def ajouter_type(p: dict) -> dict:
    """Un type d'étrier de plus (le site seulement : l'écran n'en crée un que s'il n'y en a aucun)."""
    t = EN.nouveau_type_etrier(p)
    p["types_etriers"].append(t)
    return t


def changer_forme(p: dict, forme: str) -> List[str]:
    """Une forme choisie. Une famille dont la position n'est plus offerte (talon retiré) passe à la PREMIÈRE position
    du menu, « Armatures inférieures (âme) » (edit_family : i_sel = 0). Rend les familles déplacées (index)."""
    if forme not in EN.FORMES:
        raise ValueError(f"forme {forme!r} inconnue")
    p["forme"] = forme
    deplacees = []
    for i, f in enumerate(p["familles"]):
        opts = EN.options_positions(forme, f["role"])
        if f["position"] not in opts:
            f["position"] = opts[0]
            deplacees.append(i)
    return deplacees


def changer_portee(p: dict, L: float) -> List[int]:
    """Une portée changée. En % (le défaut), les longueurs des lits gardent leur pourcentage. En mètres, l'écran garde
    la valeur EN MÈTRES de chaque champ « De (m) », « À (m) » (son état de widget) et en déduit le pourcentage ; une
    valeur qui dépasse la nouvelle portée (max_value) est rejetée par Streamlit, qui reprend alors le pourcentage. Les
    zones d'étriers, toujours en %, gardent leur pourcentage. Rend les familles dont le pourcentage a changé."""
    ancien = p["L"]
    p["L"] = L
    changees = []
    if p["longueurs"] != "metre":
        return changees
    for i, f in enumerate(p["familles"]):
        avant = (f["de"], f["a"])
        for k in ("de", "a"):
            m = ancien * f[k] / 100.0
            if m <= L:
                f[k] = 100.0 * m / L if L > 0 else 0.0
        if (f["de"], f["a"]) != avant:
            changees.append(i)
    return changees


def longueur_en_metres(p: dict, f: dict, cle: str, m: float) -> None:
    """« De (m) » ou « À (m) » saisi : le pourcentage est 100·m / L (edit_family). Bornes de l'écran : 0 à L."""
    L = p["L"]
    if not 0.0 <= m <= L:
        raise ValueError(f"{'De' if cle == 'de' else 'À'} : {m:g} m hors de la portée (0 à {L:g} m)".replace(".", ","))
    f[cle] = 100.0 * m / L if L > 0 else 0.0


def changer_retours(f: dict, r: int) -> None:
    """« Retours » choisi ; 0 remet la hauteur de retour à 0 (edit_family : st.text « — »)."""
    if r not in EN.RETOURS:
        raise ValueError(f"retours : {r!r} — 0, 1 ou 2")
    f["retours"] = r
    if r == 0:
        f["h_retour"] = 0.0


def supprimer_type(p: dict, i: int) -> List[str]:
    """Un type d'étrier supprimé. Les zones qui l'utilisaient prennent le PREMIER type restant (edit_stirrup_zone) ; s'il
    n'en reste aucun, elles gardent son nom et ne sont plus comptées. Rend les zones réaffectées (noms)."""
    nom = p["types_etriers"][i]["nom"]
    del p["types_etriers"][i]
    noms = [t["nom"] for t in p["types_etriers"]]
    touchees = []
    for z in p["zones"]:
        if z["type"] == nom and nom not in noms:
            touchees.append(z["nom"])
            if noms:
                z["type"] = noms[0]
    return touchees


def renommer_type(p: dict, i: int, nom: str) -> None:
    """Un type d'étrier renommé : ses zones le suivent. (L'écran réaffecte une zone dont le type a disparu au PREMIER
    type : avec un seul type — le cas de l'écran, qui n'en crée pas d'autre — la zone suit donc le nouveau nom ; avec
    plusieurs, l'écran la changerait de type sans le dire, le site ne le fait pas.)"""
    ancien = p["types_etriers"][i]["nom"]
    if nom != ancien and nom in [t["nom"] for t in p["types_etriers"]]:
        raise ValueError(f"le type « {nom} » existe déjà")
    p["types_etriers"][i]["nom"] = nom
    for z in p["zones"]:
        if z["type"] == ancien:
            z["type"] = nom

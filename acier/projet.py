# -*- coding: utf-8 -*-
"""Modèle d'un PROJET d'assemblages métalliques : la liste ordonnée des
assemblages d'un même fichier. Indépendant de Streamlit, testable seul.

Un projet = ``{"assemblages": [entrée, …], "suivant": n}``. Une entrée =
``{"id": int, "type": identifiant du registre, "values": saisies du type,
"visserie": annotation du cartouche}``. Le REPÈRE d'une entrée est le champ
« Repère de l'assemblage » de ses saisies (``values["id_rep"]``) : une seule
source, jamais un second nom.

Le fichier projet (``payload``) porte la version, les informations du projet
et les entrées ; ``lire_fichier`` accepte aussi les fichiers d'assemblage seul
que les modules produisaient avant le projet, et les fichiers bruts de l'outil
HTML d'origine (type détecté par l'appelant).
"""
import re
from copy import deepcopy

VERSION = "projet-assemblages-1.0"
PREFIXE_REPERE = "A"
CLES_INFOS = ("nom_projet", "partie", "date", "indice")


def nouveau():
    return {"assemblages": [], "suivant": 1}


def entrees(projet):
    return projet.setdefault("assemblages", [])


def trouver(projet, ident):
    for e in entrees(projet):
        if e.get("id") == ident:
            return e
    return None


def position(projet, ident):
    for i, e in enumerate(entrees(projet)):
        if e.get("id") == ident:
            return i
    return -1


def repere(entree):
    return str((entree.get("values") or {}).get("id_rep") or "").strip()


def repere_libre(projet, prefixe=PREFIXE_REPERE):
    """Le premier repère « A1, A2, … » que le projet n'utilise pas encore."""
    pris = {repere(e) for e in entrees(projet)}
    n = 1
    while f"{prefixe}{n}" in pris:
        n += 1
    return f"{prefixe}{n}"


def _nouvel_id(projet):
    ident = int(projet.get("suivant", 1) or 1)
    projet["suivant"] = ident + 1
    return ident


def ajouter(projet, type_id, values, visserie="", repere_=None):
    """Ajoute une entrée en fin de liste et la renvoie. Un repère vide reçoit
    le premier libre ; ``repere_`` l'impose."""
    values = dict(values or {})
    if repere_ is not None:
        values["id_rep"] = str(repere_)
    elif not str(values.get("id_rep") or "").strip():
        values["id_rep"] = repere_libre(projet)
    e = {"id": _nouvel_id(projet), "type": str(type_id), "values": values, "visserie": str(visserie or "")}
    entrees(projet).append(e)
    return e


def dupliquer(projet, ident):
    """Copie complète (deepcopy) insérée juste après l'original, repère
    « X (copie) » — même geste que 📋 dans Poutre et Dalle."""
    src = trouver(projet, ident)
    if src is None:
        return None
    copie = deepcopy(src)
    copie["id"] = _nouvel_id(projet)
    copie["values"]["id_rep"] = f"{repere(src) or PREFIXE_REPERE} (copie)"
    entrees(projet).insert(position(projet, ident) + 1, copie)
    return copie


def supprimer(projet, ident):
    i = position(projet, ident)
    if i < 0:
        return False
    del entrees(projet)[i]
    return True


def deplacer(projet, ident, delta):
    """Déplace l'entrée de ``delta`` rangs (−1 : monter, +1 : descendre) ;
    l'ordre de la liste est l'ordre du rapport."""
    lst = entrees(projet)
    i = position(projet, ident)
    j = i + int(delta)
    if i < 0 or j < 0 or j >= len(lst):
        return False
    lst[i], lst[j] = lst[j], lst[i]
    return True


def renommer(projet, ident, nouveau_repere):
    e = trouver(projet, ident)
    if e is None:
        return False
    nouveau_repere = str(nouveau_repere or "").strip()
    if not nouveau_repere:
        return False
    e.setdefault("values", {})["id_rep"] = nouveau_repere
    return True


# ------------------------------------------------------------------ fichier
def payload(projet, infos):
    infos = infos or {}
    return {"version": VERSION,
            "projet": {k: str(infos.get(k, "") or "") for k in CLES_INFOS},
            "assemblages": [{"id": int(e["id"]), "type": e["type"], "repere": repere(e),
                             "values": e.get("values", {}), "visserie": e.get("visserie", "")}
                            for e in entrees(projet)]}


def est_projet(data):
    return (isinstance(data, dict) and isinstance(data.get("assemblages"), list)
            and str(data.get("version", "")).startswith("projet-assemblages"))


def lire_fichier(data, normaliser, detecter=None):
    """Lit un fichier projet OU un fichier d'assemblage seul.

    ``normaliser(type_id, src)`` → saisies validées du type (``charger_payload``
    du module) ; ``detecter(src)`` → type d'un fichier brut sans champ
    « assemblage », ou None.
    Retourne ``(infos ou None, [(type_id, values, visserie), …])`` ; lève
    ValueError si rien n'est lisible.
    """
    if not isinstance(data, dict):
        raise ValueError("Structure JSON inattendue")
    if est_projet(data):
        lot = []
        for e in data["assemblages"]:
            if not isinstance(e, dict) or not e.get("type"):
                raise ValueError("Entrée de projet invalide")
            values = normaliser(e["type"], e)
            rep = str(e.get("repere") or values.get("id_rep") or "").strip()
            if rep:
                values["id_rep"] = rep
            lot.append((str(e["type"]), values, str(e.get("visserie") or "")))
        infos = data.get("projet") if isinstance(data.get("projet"), dict) else {}
        return {k: str(infos.get(k, "") or "") for k in CLES_INFOS}, lot
    type_id = data.get("assemblage") or (detecter(data) if detecter else None)
    if not type_id:
        raise ValueError("Type d'assemblage non reconnu")
    values = normaliser(str(type_id), data)
    visserie = data.get("visserie") if isinstance(data.get("visserie"), str) else ""
    return None, [(str(type_id), values, visserie.strip())]


def importer(projet, lot):
    """Ajoute au projet les entrées d'un lot lu par ``lire_fichier``."""
    return [ajouter(projet, t, v, vis) for (t, v, vis) in lot]


def nom_fichier(infos, ext):
    """AAA_Assemblages Partie#Indice_Date.ext — même règle que la note Poutre."""
    infos = infos or {}
    nom = str(infos.get("nom_projet") or "")
    aaa = re.sub(r"[^A-Za-z0-9]", "", nom).upper()[:3] or "PRJ"
    partie = str(infos.get("partie") or "").strip()
    indice = str(infos.get("indice") or "0").strip() or "0"
    date = str(infos.get("date") or "").strip().replace("/", "-")
    core = f"Assemblages {partie}".strip()
    name = f"{aaa}_{core}#{indice}" + (f"_{date}" if date else "") + ext
    return re.sub(r'[\\/:*?"<>|]+', "-", name)

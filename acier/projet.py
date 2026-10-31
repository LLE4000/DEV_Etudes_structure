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


def description(values):
    """Ligne courte d'une entrée : profilés et efforts (liste, sommaire)."""
    u = values or {}
    morceaux = []
    if u.get("prof_S") or u.get("prof_P"):
        morceaux.append(f"{u.get('prof_S') or '?'} sur {u.get('prof_P') or '?'}")
    elif u.get("prof_c") or u.get("bp_u"):
        # pied de poteau : profilé (ou éléments soudés du calcul manuel), platine, massif
        def _n(k):
            try:
                return f"{float(u.get(k) or 0):g}"
            except (TypeError, ValueError):
                return "?"
        manuel = str(u.get("mode") or "").startswith("Calcul manuel")
        if manuel:
            elems = [e for e in (u.get("elems") or []) if isinstance(e, dict)]
            noms = []
            for e in elems:
                t = str(e.get("type") or "")
                if t.startswith("Profilé en I"):
                    noms.append(str(e.get("prof") or "I"))
                else:
                    noms.append(t.split(" (")[0].lower())
            prof = "calcul manuel · " + (" + ".join(noms) if noms else "aucun élément")
            n_b = len([t for t in (u.get("tiges") or []) if isinstance(t, dict)])
            morceaux.append(f"{prof} · platine {_n('bp_u')}×{_n('lp_u')}×{_n('tp_u')} · {n_b} boulons {u.get('ancr') or ''}")
        else:
            prof = u.get("prof_c") if u.get("prof_c") and u.get("prof_c") != "Personnalisé" else "Poteau"
            morceaux.append(f"{prof} · platine {_n('bp_u')}×{_n('lp_u')}×{_n('tp_u')} · massif {_n('bc_u')}×{_n('lc_u')}")
    for cle, lab in (("V_Ed", "V"), ("N_Ed", "N"), ("M_Ed", "M"), ("Mx_Ed", "Mx"), ("My_Ed", "My"),
                     ("Vx_Ed", "Vx"), ("Vy_Ed", "Vy")):
        try:
            v = float(u.get(cle) or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v:
            morceaux.append(f"{lab} = {v:g} {'kNm' if lab.startswith('M') else 'kN'}")
    return " · ".join(morceaux)


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


def ajouter(projet, type_id, values, visserie="", repere_=None, verrou=False):
    """Ajoute une entrée en fin de liste et la renvoie. Un repère vide reçoit
    le premier libre ; ``repere_`` l'impose."""
    values = dict(values or {})
    if repere_ is not None:
        values["id_rep"] = str(repere_)
    elif not str(values.get("id_rep") or "").strip():
        values["id_rep"] = repere_libre(projet)
    e = {"id": _nouvel_id(projet), "type": str(type_id), "values": values, "visserie": str(visserie or ""),
         "verrou": bool(verrou)}
    entrees(projet).append(e)
    return e


def dupliquer(projet, ident, suffixe=" (copie)"):
    """Copie complète (deepcopy) insérée juste après l'original, repère
    « X (copie) » — même geste que 📋 dans Poutre et Dalle. Une copie naît
    déverrouillée."""
    src = trouver(projet, ident)
    if src is None:
        return None
    copie = deepcopy(src)
    copie["id"] = _nouvel_id(projet)
    copie["values"]["id_rep"] = f"{repere(src) or PREFIXE_REPERE}{suffixe}"
    copie["verrou"] = False
    entrees(projet).insert(position(projet, ident) + 1, copie)
    return copie


def est_verrouille(entree):
    return bool(entree and entree.get("verrou"))


def verrouiller(projet, ident, etat=True):
    """🔒 : l'entrée ne se modifie plus (saisies, repère) et ne se supprime
    plus tant qu'elle est verrouillée — comme une dalle verrouillée."""
    e = trouver(projet, ident)
    if e is None:
        return False
    e["verrou"] = bool(etat)
    return True


def supprimer(projet, ident, forcer=False):
    i = position(projet, ident)
    if i < 0 or (est_verrouille(entrees(projet)[i]) and not forcer):
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
    if e is None or est_verrouille(e):
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
                             "values": e.get("values", {}), "visserie": e.get("visserie", ""),
                             "verrou": est_verrouille(e)}
                            for e in entrees(projet)]}


def est_projet(data):
    return (isinstance(data, dict) and isinstance(data.get("assemblages"), list)
            and str(data.get("version", "")).startswith("projet-assemblages"))


def lire_fichier(data, normaliser, detecter=None):
    """Lit un fichier projet OU un fichier d'assemblage seul.

    ``normaliser(type_id, src)`` → saisies validées du type (``charger_payload``
    du module) ; ``detecter(src)`` → type d'un fichier brut sans champ
    « assemblage », ou None.
    Retourne ``(infos ou None, [(type_id, values, visserie, verrou), …])`` ;
    lève ValueError si rien n'est lisible.
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
            lot.append((str(e["type"]), values, str(e.get("visserie") or ""), bool(e.get("verrou", False))))
        infos = data.get("projet") if isinstance(data.get("projet"), dict) else {}
        return {k: str(infos.get(k, "") or "") for k in CLES_INFOS}, lot
    type_id = data.get("assemblage") or (detecter(data) if detecter else None)
    if not type_id:
        raise ValueError("Type d'assemblage non reconnu")
    values = normaliser(str(type_id), data)
    visserie = data.get("visserie") if isinstance(data.get("visserie"), str) else ""
    return None, [(str(type_id), values, visserie.strip(), False)]


def importer(projet, lot):
    """Ajoute au projet les entrées d'un lot lu par ``lire_fichier``."""
    return [ajouter(projet, t, v, vis, verrou=verrou) for (t, v, vis, verrou) in lot]


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

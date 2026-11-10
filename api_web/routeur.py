# -*- coding: utf-8 -*-
"""Le point d'entrée unique du travailleur du navigateur : une requête JSON → une réponse JSON.

    {"module": "<clé d'un module du registre>", "action": "nouveau" | "calculer" | "modifier" | "choisir" |
     "selecteur" | "activer" | "ajouter" | "dupliquer" | "supprimer" | "telecharger" | "bibliotheque" | "fiches" |
     "transmettre",
     "etat": {...}, "chemin": [...], "valeur": ..., "role": "...", "nom": "...", "familles": [...], "froid": true,
     "index": 0, "quoi": "...", "rubrique": "...", "params": {...}, "fiche": {...}, "vers": "..."}

LA BIBLIOTHÈQUE D'UNE NOTE (docs/site/BIBLIOTHEQUES.md § 2.4, § 2.5) : « bibliotheque » rend la bibliothèque de la
famille du module pour cette note (préremplie, hôte « note », les cibles de « Utiliser cette valeur ») — sans toucher
à l'état ; « fiches » (quoi = ajouter | retirer | monter | descendre) gère les fiches incluses dans la note, et répond
comme « modifier » (l'état, les fragments, un message en cas de refus).

La réponse porte l'état de la note (à enregistrer), les fragments HTML des cadres (données, résultat, fenêtres de
calcul, vues du dessin) et le résumé (taux, niveau, statut). Une erreur du moteur est rendue, jamais avalée. La réponse
est du JSON STRICT : un nombre non fini (un taux infini) part comme null, jamais « Infinity » que le navigateur ne lit pas.
Les modules viennent du registre (api_web/registre.py) : chaque fichier qui déclare MODULE en est un. Les BIBLIOTHÈQUES
(action « consulter », sans module ni note) sont à part : ``BIBLIOTHEQUES`` ci-dessous.

TÉLÉCHARGER un fichier propre au module (le bouton ``commun.bouton_telecharger(quoi, …)``, ``data-telecharger``) :

    {"module", "action": "telecharger", "etat", "quoi"} → telecharger(etat normalisé, quoi) du module
                                                        → {"nom": str, "type": str, "texte": str}
    réponse : {"ok": true, "fichier": {"nom", "type", "texte"}} — ou {"nom", "type", "base64"} pour un fichier binaire
              (commun.TYPES_BINAIRES : la note PDF) ; un module qui attend « infos » (telecharger(etat, quoi, infos))
              reçoit celles de la requête (le nom, le chantier, l'indice de la note)
              {"ok": false, "erreur": "<message en français>"} : module sans ``telecharger``, ``quoi`` inconnu (le
              module rend None, ou lève ValueError avec son message), nom absurde, type hors liste, texte de plus de
              5 Mo (les règles : ``commun.refus_fichier``). Jamais une exception.

TRANSMETTRE l'élément ouvert à un AUTRE module, qui en fait une note neuve (le bouton ``data-transmettre="<clé>"`` :
« Détailler dans Taux d'armature » d'une poutre, docs/site/LOT_Q_QUATRE_MODULES.md § 6.2) :

    {"module", "action": "transmettre", "etat", "vers"[, "date"]} → transmettre(etat normalisé, vers[, date]) du module
                                                        → {"module": str, "nom": str, "etat": {...}}
    « date » : le jour de l'envoi, « JJ/MM/AAAA », donné par l'écran (le moteur n'a pas d'horloge) ; il ne part qu'au
              module qui l'attend (un paramètre « date »), et une date qui n'en est pas une part VIDE.
    réponse : {"ok": true, "note": {"module", "nom", "etat"}} — l'état RELU par le module de destination (son
              ``normaliser``) : jamais une note que sa page ne saurait pas ouvrir ; l'écran crée la note (même chantier,
              même dossier) et l'ouvre. Rien n'est écrit dans la note d'origine (son état n'est pas rendu).
              {"ok": false, "erreur": "<message en français>"} : module sans ``transmettre``, destination absente ou
              inconnue, refus du module (ValueError : ce qui manque, dit en clair). Jamais une exception.
"""
import importlib
import inspect
import json
import math
import re
import traceback

from . import commun as C
from . import registre

# les BIBLIOTHÈQUES du site (docs/site/BIBLIOTHEQUES.md) : hors du registre des modules — une consultation n'est pas
# une note. L'action « consulter » les sert : {"action": "consulter", "bibliotheque": "beton", "rubrique": "ancrages",
# "params": {...}, "hote": "page" | "note"} → les fragments de la rubrique (api_web/biblio_<clé>.consulter) ; la
# Bibliothèque Acier (BIB5) : {"bibliotheque": "acier", "rubrique": "profils" | "nuances" | "boulons", …}
BIBLIOTHEQUES = {"beton": "api_web.biblio_beton", "acier": "api_web.biblio_acier"}


def consulter(q: dict) -> dict:
    """Une rubrique d'une bibliothèque rendue pour ses paramètres (la béton armé par défaut) ; une bibliothèque
    inconnue est refusée (ValueError, que ``traiter`` rend en erreur lisible)."""
    nom = q.get("bibliotheque", "beton")
    if nom not in BIBLIOTHEQUES:
        raise ValueError(f"bibliothèque inconnue : {nom!r} ({', '.join(BIBLIOTHEQUES)})")
    mod = importlib.import_module(BIBLIOTHEQUES[nom])
    return {"ok": True, **mod.consulter(q.get("rubrique"), q.get("params") or {}, q.get("hote", "page"))}


def _sans_infini(x):
    """La réponse sans nombre non fini : un infini ou un NaN devient None (``null``, « pas de valeur »)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _sans_infini(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sans_infini(v) for v in x]
    return x


def _json(rep: dict) -> str:
    """La réponse en JSON STRICT. Un nombre non fini n'a pas d'écriture JSON : json.dumps écrivait « Infinity » (un taux
    infini que ``commun.pct`` rend tel quel, une division par zéro de numpy), que le navigateur refuse de lire — la
    requête restait sans réponse et la file du moteur bloquée (« Calcul… » sans fin). Il part comme null ; le HTML,
    lui, l'a déjà écrit (« > 999 % », « — »)."""
    try:
        return json.dumps(rep, ensure_ascii=False, allow_nan=False)
    except ValueError:
        return json.dumps(_sans_infini(rep), ensure_ascii=False, allow_nan=False)


def _refus(texte: str) -> dict:
    return {"ok": False, "erreur": texte}


def _court(quoi) -> str:
    t = str(quoi)
    return t if len(t) <= 60 else t[:59] + "…"


def telecharger(mod, q: dict) -> dict:
    """L'action « telecharger » : le fichier que le module prépare pour ``quoi``, contrôlé avant de partir."""
    fonction = getattr(mod, "telecharger", None)
    if not callable(fonction):
        return _refus("Ce module ne propose aucun fichier à télécharger.")
    quoi = q.get("quoi")
    if not isinstance(quoi, str) or not quoi.strip():
        return _refus("Le fichier à télécharger n'est pas précisé.")
    try:
        etat = mod.normaliser(q["etat"]) if q.get("etat") else mod.nouvel_etat()
    except Exception as e:  # noqa: BLE001 — une note abîmée : un message, pas une exception
        return _refus(f"La note ne se relit pas, le fichier n'est pas préparé : {e}")
    # « infos » (le nom, le chantier, l'indice de la note : la page de garde d'une note PDF) : passées au module qui
    # les attend (un paramètre « infos »), jamais aux autres
    kw = {"infos": q.get("infos") if isinstance(q.get("infos"), dict) else {}} \
        if "infos" in inspect.signature(fonction).parameters else {}
    try:
        f = fonction(etat, quoi, **kw)
    except ValueError as e:  # le refus du module, déjà dit en français
        return _refus(str(e) or f"Fichier inconnu : « {_court(quoi)} ».")
    except Exception as e:  # noqa: BLE001 — l'écran montre l'erreur ; la trace part dans la console
        return {"ok": False, "erreur": f"Le fichier « {_court(quoi)} » n'a pas pu être préparé : {type(e).__name__} : {e}",
                "trace": traceback.format_exc()}
    if f is None:
        return _refus(f"Fichier inconnu : « {_court(quoi)} ».")
    raison = C.refus_fichier(f)
    if raison:
        return _refus(f"Le fichier « {_court(quoi)} » est refusé : {raison}.")
    if f["type"] in C.TYPES_BINAIRES:
        return {"ok": True, "fichier": {"nom": f["nom"], "type": f["type"], "base64": f["base64"]}}
    return {"ok": True, "fichier": {"nom": f["nom"], "type": f["type"], "texte": f["texte"]}}


NOM_NOTE_MAX = 120   # caractères du nom de la note créée par « transmettre »
DATE_ENVOI = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")   # « JJ/MM/AAAA » : la date de l'envoi, telle que l'écran la donne


def date_envoi(q: dict) -> str:
    """La date de l'envoi lue dans la demande : « JJ/MM/AAAA », ou vide si l'écran a donné autre chose."""
    d = q.get("date")
    return d if isinstance(d, str) and DATE_ENVOI.fullmatch(d) else ""


def transmettre(mod, q: dict) -> dict:
    """L'action « transmettre » : ce que le module envoie à un autre (``mod.transmettre(etat, vers[, date])``), contrôlé avant
    de partir — la destination est un module du registre, et elle RELIT l'état qu'on lui envoie."""
    fonction = getattr(mod, "transmettre", None)
    if not callable(fonction):
        return _refus("Ce module n'envoie rien vers un autre module.")
    vers = q.get("vers")
    if not isinstance(vers, str) or not vers.strip():
        return _refus("Le module de destination n'est pas précisé.")
    try:
        etat = mod.normaliser(q["etat"]) if q.get("etat") else mod.nouvel_etat()
    except Exception as e:  # noqa: BLE001 — une note abîmée : un message, pas une exception
        return _refus(f"La note ne se relit pas, rien n'est envoyé : {e}")
    kw = {"date": date_envoi(q)} if "date" in inspect.signature(fonction).parameters else {}
    try:
        out = fonction(etat, vers, **kw)
    except ValueError as e:  # le refus du module, déjà dit en français
        return _refus(str(e) or f"Envoi vers « {_court(vers)} » impossible.")
    except Exception as e:  # noqa: BLE001 — l'écran montre l'erreur ; la trace part dans la console
        return {"ok": False, "erreur": f"L'envoi vers « {_court(vers)} » a échoué : {type(e).__name__} : {e}",
                "trace": traceback.format_exc()}
    if not isinstance(out, dict) or not isinstance(out.get("module"), str) or not isinstance(out.get("etat"), dict):
        return _refus(f"Envoi vers « {_court(vers)} » impossible.")
    try:
        cible = registre.module(out["module"])
        etat_cible = cible.normaliser(out["etat"])
    except Exception as e:  # noqa: BLE001 — la destination refuse ce qu'on lui envoie : rien ne part
        return _refus(f"« {_court(out['module'])} » ne relit pas ce qui lui est envoyé : {e}")
    nom = " ".join(str(out.get("nom") or "").split())[:NOM_NOTE_MAX]
    return {"ok": True, "note": {"module": out["module"], "nom": nom, "etat": etat_cible}}


def bibliotheque(mod, q: dict, etat: dict) -> dict:
    """L'action « bibliotheque » (docs/site/BIBLIOTHEQUES.md § 2.4) : la bibliothèque de la famille du module, rendue
    pour une NOTE (hôte « note ») avec les paramètres que la note connaît — ``mod.bibliotheque_prefill(etat)`` dit la
    bibliothèque, la rubrique à ouvrir d'abord et le préremplissage de chaque rubrique ; les paramètres de l'écran
    priment — et les CIBLES que « Utiliser cette valeur » peut écrire (``mod.bibliotheque_cibles(etat, rubrique,
    fiche)`` : le chemin de ``modifier``, la valeur, dite possible ou non). Un module sans bibliothèque est refusé
    doucement. L'état n'est pas rendu : consulter ne change rien à la note."""
    pre = getattr(mod, "bibliotheque_prefill", None)
    if not callable(pre):
        return _refus("Ce module n'a pas de bibliothèque.")
    p = pre(etat)
    rubrique = q.get("rubrique") or p.get("rubrique")
    params = {**(p.get("params", {}).get(rubrique) or {}), **(q.get("params") if isinstance(q.get("params"), dict) else {})}
    out = consulter({"bibliotheque": p["bibliotheque"], "rubrique": rubrique, "params": params, "hote": "note"})
    cib = getattr(mod, "bibliotheque_cibles", None)
    out["cibles"] = cib(etat, out["rubrique"], out["fiche"]) if callable(cib) else []
    return out


def traiter(requete: str) -> str:
    try:
        q = json.loads(requete)
        if q.get("action") == "consulter":
            return _json(consulter(q))
        mod = registre.module(q.get("module"))
        action = q.get("action", "calculer")
        if action == "telecharger":
            return _json(telecharger(mod, q))
        if action == "transmettre":
            return _json(transmettre(mod, q))
        etat = mod.normaliser(q["etat"]) if q.get("etat") and action != "nouveau" else mod.nouvel_etat()
        if action == "bibliotheque":
            return _json(bibliotheque(mod, q, etat))
        message = None
        if action == "modifier":
            message = mod.modifier(etat, q["chemin"], q.get("valeur"))
        elif action == "choisir":
            mod.choisir_section(etat, q["role"], q["nom"])
        elif action in ("activer", "ajouter", "dupliquer", "supprimer"):
            mod.gerer_elements(etat, action, q.get("index"))
        elif action == "selecteur":
            return _json({"ok": True, "html": mod.selecteur(etat, q["role"], q.get("familles"), q.get("froid", True))})
        elif action == "fiches":
            fonction = getattr(mod, "fiches", None)
            if not callable(fonction):
                return _json(_refus("Ce module ne garde pas de fiches de la bibliothèque."))
            message = fonction(etat, q.get("quoi"), q.get("fiche"), q.get("index"))
        out = mod.resultat(etat)
        out.update({"ok": True, "etat": etat, "message": message})
        return _json(out)
    except Exception as e:  # noqa: BLE001 — l'écran montre l'erreur ; la trace part dans la console
        return json.dumps({"ok": False, "erreur": f"{type(e).__name__} : {e}", "trace": traceback.format_exc()}, ensure_ascii=False)

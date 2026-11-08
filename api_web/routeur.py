# -*- coding: utf-8 -*-
"""Le point d'entrée unique du travailleur du navigateur : une requête JSON → une réponse JSON.

    {"module": "<clé d'un module du registre>", "action": "nouveau" | "calculer" | "modifier" | "choisir" |
     "selecteur" | "activer" | "ajouter" | "dupliquer" | "supprimer" | "telecharger", "etat": {...}, "chemin": [...],
     "valeur": ..., "role": "...", "nom": "...", "familles": [...], "froid": true, "index": 0, "quoi": "..."}

La réponse porte l'état de la note (à enregistrer), les fragments HTML des cadres (données, résultat, fenêtres de
calcul, vues du dessin) et le résumé (taux, niveau, statut). Une erreur du moteur est rendue, jamais avalée. La réponse
est du JSON STRICT : un nombre non fini (un taux infini) part comme null, jamais « Infinity » que le navigateur ne lit pas.
Les modules viennent du registre (api_web/registre.py) : chaque fichier qui déclare MODULE en est un.

TÉLÉCHARGER un fichier propre au module (le bouton ``commun.bouton_telecharger(quoi, …)``, ``data-telecharger``) :

    {"module", "action": "telecharger", "etat", "quoi"} → telecharger(etat normalisé, quoi) du module
                                                        → {"nom": str, "type": str, "texte": str}
    réponse : {"ok": true, "fichier": {"nom", "type", "texte"}} — ou {"nom", "type", "base64"} pour un fichier binaire
              (commun.TYPES_BINAIRES : la note PDF) ; un module qui attend « infos » (telecharger(etat, quoi, infos))
              reçoit celles de la requête (le nom, le chantier, l'indice de la note)
              {"ok": false, "erreur": "<message en français>"} : module sans ``telecharger``, ``quoi`` inconnu (le
              module rend None, ou lève ValueError avec son message), nom absurde, type hors liste, texte de plus de
              5 Mo (les règles : ``commun.refus_fichier``). Jamais une exception.
"""
import json
import math
import traceback

from . import commun as C
from . import registre


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
    import inspect
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


def traiter(requete: str) -> str:
    try:
        q = json.loads(requete)
        mod = registre.module(q.get("module"))
        action = q.get("action", "calculer")
        if action == "telecharger":
            return _json(telecharger(mod, q))
        etat = mod.normaliser(q["etat"]) if q.get("etat") and action != "nouveau" else mod.nouvel_etat()
        message = None
        if action == "modifier":
            message = mod.modifier(etat, q["chemin"], q.get("valeur"))
        elif action == "choisir":
            mod.choisir_section(etat, q["role"], q["nom"])
        elif action in ("activer", "ajouter", "dupliquer", "supprimer"):
            mod.gerer_elements(etat, action, q.get("index"))
        elif action == "selecteur":
            return _json({"ok": True, "html": mod.selecteur(etat, q["role"], q.get("familles"), q.get("froid", True))})
        out = mod.resultat(etat)
        out.update({"ok": True, "etat": etat, "message": message})
        return _json(out)
    except Exception as e:  # noqa: BLE001 — l'écran montre l'erreur ; la trace part dans la console
        return json.dumps({"ok": False, "erreur": f"{type(e).__name__} : {e}", "trace": traceback.format_exc()}, ensure_ascii=False)

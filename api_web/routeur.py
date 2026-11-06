# -*- coding: utf-8 -*-
"""Le point d'entrée unique du travailleur du navigateur : une requête JSON → une réponse JSON.

    {"module": "<clé d'un module du registre>", "action": "nouveau" | "calculer" | "modifier" | "choisir" |
     "selecteur" | "activer" | "ajouter" | "dupliquer" | "supprimer", "etat": {...}, "chemin": [...], "valeur": ...,
     "role": "...", "nom": "...", "familles": [...], "froid": true, "index": 0}

La réponse porte l'état de la note (à enregistrer), les fragments HTML des cadres (données, résultat, fenêtres de
calcul, vues du dessin) et le résumé (taux, niveau, statut). Une erreur du moteur est rendue, jamais avalée.
Les modules viennent du registre (api_web/registre.py) : chaque fichier qui déclare MODULE en est un.
"""
import json
import traceback

from . import registre


def traiter(requete: str) -> str:
    try:
        q = json.loads(requete)
        mod = registre.module(q.get("module"))
        action = q.get("action", "calculer")
        etat = mod.normaliser(q["etat"]) if q.get("etat") and action != "nouveau" else mod.nouvel_etat()
        message = None
        if action == "modifier":
            message = mod.modifier(etat, q["chemin"], q.get("valeur"))
        elif action == "choisir":
            mod.choisir_section(etat, q["role"], q["nom"])
        elif action in ("activer", "ajouter", "dupliquer", "supprimer"):
            mod.gerer_elements(etat, action, q.get("index"))
        elif action == "selecteur":
            return json.dumps({"ok": True, "html": mod.selecteur(etat, q["role"], q.get("familles"), q.get("froid", True))}, ensure_ascii=False)
        out = mod.resultat(etat)
        out.update({"ok": True, "etat": etat, "message": message})
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 — l'écran montre l'erreur ; la trace part dans la console
        return json.dumps({"ok": False, "erreur": f"{type(e).__name__} : {e}", "trace": traceback.format_exc()}, ensure_ascii=False)

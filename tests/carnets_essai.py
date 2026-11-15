# -*- coding: utf-8 -*-
"""Les CARNETS D'ESSAI des modules à carnet, pour les tests qui rejouent le routeur sans navigateur.

Un module qui déclare ``CARNET = True`` (api_web/routeur.py) tient des données HORS de la note, que l'écran garde dans le
navigateur et joint à chaque requête : la base des prix des grues (« Prix des grues »). Sans elle, son écran n'offre
qu'une action (importer la base). Les tests communs (tests/test_api_web_modules.py, tests/test_reference_refonte.py)
lui joignent donc, comme l'écran, son carnet d'ESSAI : la base de TEST du dépôt (prix fictifs, loueurs A/B/C).

    traiter(texte_json) -> la réponse du routeur, le carnet d'essai joint si la requête vise un module à carnet et n'en
                           porte pas"""
import copy
import json
import os

ICI = os.path.dirname(os.path.abspath(__file__))
CARNETS = {"prix_grues": os.path.join(ICI, "donnees", "prix_grues_base_test.json")}
_lus = {}


def carnet(cle: str):
    """Le carnet d'essai du module ``cle`` (une copie neuve), ou None."""
    if cle not in CARNETS:
        return None
    if cle not in _lus:
        with open(CARNETS[cle], encoding="utf-8") as f:
            _lus[cle] = json.load(f)
    return [copy.deepcopy(_lus[cle])]


def traiter(texte: str) -> str:
    from api_web import routeur as R
    q = json.loads(texte)
    if isinstance(q, dict) and q.get("module") in CARNETS and "carnet" not in q:
        q["carnet"] = carnet(q["module"])
        return R.traiter(json.dumps(q))
    return R.traiter(texte)

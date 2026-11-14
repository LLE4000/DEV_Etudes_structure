# -*- coding: utf-8 -*-
"""Ce que rend une MÉTHODE de vérification d'une barre (historique, Eurocode 3) — une seule forme pour les deux :

    {"methode", "norme", "statut" (ok | nok | non_couvert), "taux_max", "gouvernant",
     "verifications": [{"id", "methode", "titre", "reference", "critere", "valeur", "limite", "unite", "taux",
                        "statut", "formules" (clés de ``calcul``), "messages"}],
     "grandeurs", "calcul": [{"cle", "gabarit", "symbole", "termes": [{"symbole", "valeur", "unite"}], "resultat",
                              "unite"}], "alertes": [{"code", "niveau", "message"}], "hypotheses"}

Le moteur CALCULE et NOMME ; les gabarits des formules sont dans acier/barre/formules.py ; l'écran et la note
composent et formatent.
"""
from typing import Dict, List, Optional

OK, NOK, NON_COUVERT = "ok", "nok", "non_couvert"


class Refus(ValueError):
    """Une donnée hors du domaine de la méthode : le calcul est refusé, avec ce message."""


class NonCouvert(Exception):
    """Une donnée valide que la méthode ne couvre pas (une forme, un tube rond de classe 4) : la vérification est
    rendue « non_couvert », avec ce message — jamais « ok »."""


def terme(symbole: str, valeur, unite: str = "") -> Dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def formule(cle: str, symbole: str, termes: List[Dict], resultat, unite: str = "", gabarit: Optional[str] = None) -> Dict:
    """Une formule nommée : ``gabarit`` est la clé de acier/barre/formules.GABARITS (``cle`` à défaut)."""
    return {"cle": cle, "gabarit": gabarit or cle, "symbole": symbole, "termes": termes, "resultat": resultat, "unite": unite}


def fr(x: float, nd: int = 0) -> str:
    """Un nombre dans une alerte : virgule, espace fine insécable des milliers."""
    return f"{x:,.{nd}f}".replace(",", "\u202f").replace(".", ",")


class Journal:
    """Ce qu'une méthode accumule en calculant : grandeurs, blocs de formules, vérifications, alertes, hypothèses."""

    def __init__(self, methode: str, norme: str):
        self.methode, self.norme = methode, norme
        self.G: Dict = {}
        self.calcul: List[Dict] = []
        self.verifs: List[Dict] = []
        self.alertes: List[Dict] = []
        self.hyp: List[str] = []

    def f(self, cle: str, symbole: str, termes: List[Dict], resultat, unite: str = "", gabarit: Optional[str] = None) -> str:
        """Ajoute un bloc de formule et rend sa clé."""
        self.calcul.append(formule(cle, symbole, termes, resultat, unite, gabarit))
        return cle

    def alerte(self, code: str, niveau: str, message: str) -> str:
        self.alertes.append({"code": code, "niveau": niveau, "message": message})
        return message

    def verif(self, id_: str, titre: str, reference: str, critere: str, valeur, limite, unite: str, formules, taux=None,
              statut: Optional[str] = None, messages=()) -> Dict:
        """Une vérification ; sans ``statut``, le taux est valeur/limite et le statut s'en déduit."""
        if statut is None:
            if taux is None:
                taux = valeur / limite
            statut = OK if taux <= 1.0 else NOK
        v = {"id": id_, "methode": self.methode, "titre": titre, "reference": reference, "critere": critere, "valeur": valeur,
             "limite": limite, "unite": unite, "taux": taux, "statut": statut, "formules": [k for k in formules if k],
             "messages": list(messages)}
        self.verifs.append(v)
        return v

    def rendre(self) -> Dict:
        """Le résultat de la méthode : le bloc « taux » de chaque vérification chiffrée, puis la synthèse."""
        for v in self.verifs:
            if v["taux"] is not None and v["valeur"] is not None and f"taux_{v['id']}" not in v["formules"]:
                v["formules"].append(self.f(f"taux_{v['id']}", "η", [terme("valeur", v["valeur"], v["unite"]),
                                                                    terme("limite", v["limite"], v["unite"])], v["taux"], "", "taux"))
        s = synthese(self.verifs)
        return {"methode": self.methode, "norme": self.norme, "statut": s["statut"], "taux_max": s["taux_max"],
                "gouvernant": s["gouvernant"], "verifications": self.verifs, "grandeurs": self.G, "calcul": self.calcul,
                "alertes": self.alertes, "hypotheses": self.hyp}


def synthese(verifs: List[Dict]) -> Dict:
    """{"statut", "taux_max", "gouvernant"} : « nok » dès qu'une vérification ne passe pas ; « non_couvert » si rien ne
    dépasse mais qu'une vérification manque — jamais « ok » ; le gouvernant est le plus fort taux, ou la première
    vérification refusée sans taux (une instabilité)."""
    juges = [v for v in verifs if v["statut"] in (OK, NOK)]
    if any(v["statut"] == NOK for v in juges):
        statut = NOK
    elif any(v["statut"] == NON_COUVERT for v in verifs):
        statut = NON_COUVERT
    else:
        statut = OK
    avec_taux = [v for v in juges if v["taux"] is not None]
    gouv = max(avec_taux, key=lambda v: v["taux"]) if avec_taux else None
    sans_taux = [v for v in juges if v["taux"] is None]
    if sans_taux:
        gouv = sans_taux[0]
    return {"statut": statut, "taux_max": gouv["taux"] if gouv else None, "gouvernant": gouv["id"] if gouv else None}

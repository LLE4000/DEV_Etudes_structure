# -*- coding: utf-8 -*-
"""Ce que rend une MÉTHODE de vérification d'une barre (historique, Eurocode 3) — une seule forme pour les deux :

    {"methode", "norme", "statut" (ok | nok | bloque | non_couvert | hors_domaine), "taux_max", "gouvernant",
     "verifications": [{"id", "methode", "titre", "reference", "critere", "valeur", "limite", "unite", "taux",
                        "statut", "formules" (clés de ``calcul``), "messages"}],
     "grandeurs", "calcul": [{"cle", "gabarit", "symbole", "termes": [{"symbole", "valeur", "unite"}], "resultat",
                              "unite"}], "alertes": [{"code", "niveau", "message"}], "hypotheses"}

Le moteur CALCULE et NOMME ; les gabarits des formules sont dans acier/barre/formules.py ; l'écran et la note
composent et formatent.
"""
import decimal
import math
from typing import Dict, List, Optional

from .. import js as JS

OK, NOK, NON_COUVERT = "ok", "nok", "non_couvert"
# « hors domaine » (brief § 9) : le calcul est fait et montré, mais la méthode ne couvre pas le cas (méthode historique :
# nuance hors AE235–AE355, creux formé à froid de plus de 6 mm) — jamais « vérifié », jamais caché derrière le taux global
HORS_DOMAINE = "hors_domaine"
# « bloqué » (brief § 1 règle 1, § 9) : une donnée indispensable manque ou se contredit (les bords des semelles d'un PRS,
# la fabrication d'un tube, une classe de déversement à fournir, s_s d'un appui dont l'âme ne passe pas à la borne 0) —
# le verdict attend la donnée ; le calcul fait du côté défavorable reste montré pour information, jamais jugé
BLOQUE = "bloque"
# les statuts, du plus défavorable au moins défavorable : « ok » vient après eux tous (une seule source : pire_statut,
# synthese, l'enveloppe et l'écran la lisent)
GRAVITE = (NOK, BLOQUE, NON_COUVERT, HORS_DOMAINE)
STATUTS = GRAVITE + (OK,)
# les statuts JUGÉS : leur taux compte dans le taux le plus fort (hors domaine : calculé, montré, jamais « vérifié »)
JUGES = (OK, NOK, HORS_DOMAINE)


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


def fixe(x, nd: int) -> str:
    """``x`` écrit avec ``nd`` décimales (point décimal) — L'ARRONDI D'AFFICHAGE du module, le seul : DEMI VERS
    L'EXTÉRIEUR sur la valeur binaire exacte, la règle des moteurs et de l'écran (``acier.js.to_fixed``, celle de
    ``api_web.commun.fr``), appelée — pas recopiée. Une phrase, une formule et un tableau écrivent ainsi le même
    chiffre : 458,25 kN → « 458,3 » partout (l'arrondi de Python, au chiffre pair, écrivait « 458,2 » dans la formule,
    sous un tableau qui disait « 458,3 »). La précision est élargie pour qu'un très grand nombre s'écrive au lieu de
    lever ; le signe d'un zéro est ôté : jamais « -0 »."""
    with decimal.localcontext() as ctx:
        ctx.prec = 420                            # 309 chiffres pour le plus grand flottant, plus les décimales
        s = JS.to_fixed(x, int(nd))
    if s.startswith("-") and not s.strip("-0."):
        s = s[1:]
    return s


def fr(x: float, nd: int = 0) -> str:
    """Un nombre dans une phrase (hypothèse, alerte) : virgule, espace fine insécable des milliers ; « — » s'il n'est
    pas fini."""
    if not math.isfinite(x):
        return "—"
    ent, _, dec = fixe(x, nd).partition(".")
    signe, chiffres = ("-", ent[1:]) if ent.startswith("-") else ("", ent)
    return signe + f"{int(chiffres):,}".replace(",", "\u202f") + ("," + dec if dec else "")


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


def pire_statut(statuts) -> str:
    """Le plus défavorable de plusieurs statuts : nok, puis bloqué, puis non couvert, puis hors domaine, puis ok. Un
    statut inconnu est une erreur (ValueError) — jamais ignoré : une méthode qu'il juge sortirait « ok »."""
    statuts = list(statuts)
    inconnus = sorted(set(statuts) - set(STATUTS))
    if inconnus:
        raise ValueError(f"statut inconnu : {', '.join(map(repr, inconnus))}")
    return next((x for x in GRAVITE if x in statuts), OK)


def synthese(verifs: List[Dict]) -> Dict:
    """{"statut", "taux_max", "gouvernant"} : « nok » dès qu'une vérification ne passe pas ; « bloque » si rien ne
    dépasse mais qu'une donnée manque ; « non_couvert » si une vérification n'est pas couverte ; « hors_domaine » si une
    vérification est hors du domaine de la méthode — jamais « ok » dans ces trois cas ; un statut inconnu lève
    (ValueError). Le gouvernant est le plus fort taux JUGÉ (une vérification hors domaine compte : son taux est calculé ;
    un taux bloqué ou non couvert est pour information), ou la première vérification refusée sans taux (une instabilité)."""
    juges = [v for v in verifs if v["statut"] in JUGES]
    inconnus = sorted({v["statut"] for v in verifs} - set(STATUTS))
    if inconnus:
        raise ValueError(f"statut inconnu : {', '.join(map(repr, inconnus))}")
    if any(v["statut"] == NOK for v in juges):
        statut = NOK
    elif any(v["statut"] == BLOQUE for v in verifs):
        statut = BLOQUE
    elif any(v["statut"] == NON_COUVERT for v in verifs):
        statut = NON_COUVERT
    elif any(v["statut"] == HORS_DOMAINE for v in verifs):
        statut = HORS_DOMAINE
    else:
        statut = OK
    avec_taux = [v for v in juges if v["taux"] is not None]
    gouv = max(avec_taux, key=lambda v: v["taux"]) if avec_taux else None
    sans_taux = [v for v in juges if v["taux"] is None]
    if sans_taux:
        gouv = sans_taux[0]
    return {"statut": statut, "taux_max": gouv["taux"] if gouv else None, "gouvernant": gouv["id"] if gouv else None}

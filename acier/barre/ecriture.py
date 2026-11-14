# -*- coding: utf-8 -*-
"""Barres en acier — l'ÉCRITURE d'une formule : la même dans la fenêtre de calcul de l'écran et dans la note PDF.

Le moteur rend des blocs ``{cle, gabarit, symbole, termes [{symbole, valeur, unite}], resultat, unite}`` ;
``formules.py`` porte l'expression de chaque gabarit. Ce fichier en fait ce qui s'imprime :

    symbole = formule en symboles = formule en valeurs = résultat          (N_b,z,Rd = χ_z·A·f_y/γ_M1 = … = 569,2 kN)

    ecrire(bloc, symboles=None) -> {"symbole", "litteral", "numerique", "resultat"}        du texte, sans mise en forme
    morceaux(bloc, symboles=None) -> ([symbole, littéral, numérique], résultat)            les membres à joindre par « = »
                                     (sans le littéral quand il répète le symbole)
    relire(bloc) -> float            le gabarit évalué sur les valeurs TELLES QU'IMPRIMÉES
    relecture_juste(bloc) -> bool    … redonne-t-il le résultat imprimé ?
    nombre(v, nd=None) -> str        un nombre imprimé

Règles d'écriture :
- un terme s'imprime à CINQ chiffres significatifs, sans zéro inutile (un entier : en entier) — un calcul refait à la
  main avec les nombres imprimés retombe sur le résultat imprimé (la règle des fondations de grues) ;
- le résultat prend les décimales de son gabarit (−1 : un pourcentage, à une décimale) ;
- un facteur d'unités (``*1e3``, ``/1e5``) disparaît de la formule en symboles et s'imprime « × 10³ », « / 10⁵ » dans la
  formule en valeurs ; une valeur négative se met entre parenthèses ; un symbole composé (« N_Ed/N_cr,y ») aussi, là où
  la priorité des opérations le demande.

``relire`` est la preuve de l'écriture (tests/test_barre_ecriture.py) : chaque formule imprimée, réévaluée sur ses
nombres imprimés, redonne le résultat imprimé à l'arrondi d'affichage près.
"""
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import formules as FO

CHIFFRES = 5                     # chiffres significatifs d'un terme imprimé
DECIMALES_MAX = 12
_IDENT = re.compile(r"(?<![A-Za-z0-9_.])([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_(])")
_UNITE = re.compile(r"\s*([*/])\s*1e(\d)(?![0-9.])")                 # un facteur d'unités, tel que formules.py l'écrit
_DECIMALE = re.compile(r"(?<![A-Za-z_0-9.])(\d+)\.(\d+)")            # une constante de la norme : 0.5, 28.4
_PUISSANCE = re.compile(r"\*\*(\d+(?:,\d+)?|[^\s()*/+−-]+)")
_MOINS_UNAIRE = re.compile(r"(^|[(,]\s*)-(?=\S)")
_SEUL = re.compile(r"/\(([^()\s·×+−/*-]+)\)")                        # « /(symbole) » : des parenthèses devenues inutiles
_EXPOSANTS = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
_JETON = "⁣"                # borne d'un facteur d'unités pendant la substitution : aucun caractère d'identifiant
MOINS = "−"


def decimales(v) -> int:
    """Les décimales d'un terme imprimé : aucune pour un entier, sinon CHIFFRES chiffres significatifs."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v == 0:
        return 0
    if abs(v - round(v)) < 1e-9 * max(1.0, abs(v)):
        return 0
    return max(0, min(DECIMALES_MAX, CHIFFRES - 1 - int(math.floor(math.log10(abs(v))))))


def nombre(v, nd: Optional[int] = None) -> str:
    """Un nombre tel qu'il est imprimé : virgule décimale, sans zéro inutile, le vrai signe moins ; « — » s'il manque."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "—"
    s = f"{v:.{decimales(v) if nd is None else nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s.replace(".", ",").replace("-", MOINS)


def _lu(texte: str) -> float:
    """La valeur d'un nombre imprimé."""
    return float(texte.replace(MOINS, "-").replace(",", "."))


def _puissance(m) -> str:
    e = m.group(1)
    return _EXPOSANTS[e] if e in ("2", "3", "4", "5") else "^" + e


def _habiller(expr: str, numerique: bool) -> str:
    """Les opérateurs en typographie : ** → exposant, sqrt → √, * → · (symboles) ou × (valeurs), le vrai signe moins."""
    expr = _PUISSANCE.sub(_puissance, expr)
    expr = re.sub(r"sqrt\((\d+(?:,\d+)?)\)", r"√\1", expr).replace("sqrt(", "√(")
    expr = _MOINS_UNAIRE.sub(lambda m: m.group(1) + MOINS, expr)
    expr = expr.replace(" - ", f" {MOINS} ")
    if numerique:
        return re.sub(r"\s*/\s*", " / ", expr.replace("*", " × "))
    return _SEUL.sub(r"/\1", expr.replace("*", "·"))


def _gabarit(bloc: Dict) -> Tuple[Tuple[str, ...], str, int]:
    noms, expression, nd = FO.GABARITS[bloc["gabarit"]]
    if len(noms) != len(bloc["termes"]):
        raise ValueError(f"« {bloc['cle']} » : {len(bloc['termes'])} terme(s) pour le gabarit « {bloc['gabarit']} », qui en attend {len(noms)}")
    return noms, expression, nd


def _symbole(s: str, avant: str, apres: str) -> str:
    """Un symbole posé dans une formule : entre parenthèses s'il est composé et que ses voisins le demandent."""
    if any(c in s for c in ("+", MOINS, " - ")) or ("/" in s and (avant == "/" or apres[:1] == "/" or apres.startswith("**"))):
        return f"({s})"
    return s


def ecrire(bloc: Dict, symboles: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Le bloc écrit : {"symbole", "litteral", "numerique", "resultat"}. ``symboles`` remplace ceux des termes (le taux
    d'une vérification s'écrit avec les deux membres de son critère)."""
    noms, expression, nd = _gabarit(bloc)
    syms = list(symboles) if symboles is not None else [t["symbole"] for t in bloc["termes"]]
    par_sym = dict(zip(noms, syms))
    par_val = dict(zip(noms, (t["valeur"] for t in bloc["termes"])))
    expression = _DECIMALE.sub(r"\1,\2", expression)

    def lit(m) -> str:
        k = m.group(1)
        if k == "pi":
            return "π"
        if k not in par_sym:
            return k
        return _symbole(par_sym[k], m.string[m.start() - 1:m.start()], m.string[m.end():m.end() + 2])

    def num(m) -> str:
        k = m.group(1)
        if k == "pi":
            return "π"
        if k not in par_val:
            return k
        v = par_val[k]
        if par_sym[k].startswith("γ") and decimales(v) == 0:        # un coefficient partiel garde sa décimale : « 1,0 »
            return nombre(v) + ",0"
        s = nombre(v)
        return f"({s})" if s.startswith(MOINS) else s

    litteral = _habiller(_IDENT.sub(lit, _UNITE.sub("", expression)), False)
    brut = _UNITE.sub(lambda m: f" {_JETON}{m.group(1)}{m.group(2)}{_JETON}", expression)
    numerique = _habiller(_IDENT.sub(num, brut), True)
    numerique = re.sub(f" {_JETON}( × | / )(\\d){_JETON}", lambda m: m.group(1) + "10" + ("" if m.group(2) == "1" else _EXPOSANTS[m.group(2)]),
                       numerique)
    numerique = re.sub(r"\s+", " ", numerique).strip()
    if nd == -1:
        res = nombre(bloc["resultat"] * 100.0, 1) + " %"
    else:
        res = nombre(bloc["resultat"], nd) + (" " + bloc["unite"] if bloc["unite"] else "")
    return {"symbole": bloc["symbole"], "litteral": litteral, "numerique": numerique, "resultat": res}


def morceaux(bloc: Dict, symboles: Optional[Sequence[str]] = None) -> Tuple[List[str], str]:
    """``([symbole, littéral, numérique], résultat)`` : les membres à joindre par « = », puis le résultat. Quand la
    formule en symboles EST le symbole (le rapport « N_Ed/N_cr,y »), elle n'est pas écrite deux fois : ``([symbole,
    numérique], résultat)``."""
    e = ecrire(bloc, symboles)
    membres = [e["symbole"]] + ([e["litteral"]] if e["litteral"] != e["symbole"] else []) + [e["numerique"]]
    return membres, e["resultat"]


def relire(bloc: Dict) -> float:
    """Le gabarit évalué sur les valeurs des termes TELLES QU'IMPRIMÉES (cinq chiffres significatifs) : la valeur qu'un
    lecteur retrouve en refaisant le calcul de la note à la main."""
    noms, expression, _ = _gabarit(bloc)
    env = {n: _lu(nombre(t["valeur"])) for n, t in zip(noms, bloc["termes"])}
    return eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi, **env})  # noqa: S307 — gabarits du dépôt


RELECTURE = 5e-4                 # l'écart relatif admis entre le calcul refait à la main et le résultat imprimé


def relecture_juste(bloc: Dict) -> bool:
    """La formule imprimée, refaite à la main, redonne-t-elle le résultat imprimé ? Oui si l'écart tient dans le dernier
    chiffre imprimé du résultat, ou dans RELECTURE en valeur relative (un grand nombre imprimé à une décimale)."""
    _, _, nd = _gabarit(bloc)
    r, v = bloc["resultat"], relire(bloc)
    pas = 10.0 ** -(3 if nd == -1 else nd)                           # un pourcentage à une décimale : 10⁻³ du taux
    return abs(v - r) <= pas or abs(v - r) <= RELECTURE * abs(r)

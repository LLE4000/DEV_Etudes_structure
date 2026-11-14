# -*- coding: utf-8 -*-
"""Barres en acier — les GABARITS des formules que le moteur nomme (note de calcul, fenêtres de calcul).

Le moteur rend des blocs ``{cle, gabarit, symbole, termes [{symbole, valeur, unite}], resultat, unite}`` ; ce fichier
porte le SEUL savoir qui manque pour les écrire : l'expression de chaque gabarit, sur des identifiants pris DANS
L'ORDRE des termes du moteur. C'est la forme des notes acier existantes (acier/corniere/formules.py) : la note en fait
des substitutions « grandeur = formule = valeurs = résultat » avec le même kit.

Chaque gabarit est TESTÉ : évalué sur les termes non arrondis du moteur, il redonne le résultat du moteur à 10⁻⁹ près
(tests/test_barre_historique.py) — la preuve que la formule imprimée EST celle du calcul.

Unités imprimées de la méthode historique : celles des classeurs du bureau — kN, kN·m, cm, cm², cm⁴, cm³, mm², N/mm² ;
E en kN/cm² (21 000) dans la charge critique, qui s'écrit alors sans facteur : π²·E·I/(u·L)² en kN.
"""
import math
from typing import Dict, Tuple

# gabarit → (identifiants des termes, dans l'ordre du moteur ; expression Python ; décimales du résultat, −1 : pourcentage)
GABARITS: Dict[str, Tuple[Tuple[str, ...], str, int]] = {
    # ---- méthode historique (acier/barre/historique.py)
    "sigma_adm": (("fy", "s"), "fy/s", 1),
    "lambda_E": (("E", "fy"), "pi*sqrt(E/fy)", 1),
    "lambda": (("u", "L", "i"), "u*L/i", 1),
    "lambda_bar": (("lam", "lamE"), "lam/lamE", 3),
    "A_prime": (("alpha", "lb"), "0.5 + alpha*lb**2", 4),
    "B_prime": (("beta", "lb"), "beta*lb**2", 4),
    "phi": (("Ap", "Bp"), "1/(Ap + sqrt(Ap**2 - Bp))", 3),
    "P_adm": (("A", "phi", "fy", "s"), "A*phi*fy/s/1000", 1),
    "P_cr": (("E", "I", "u", "L"), "pi**2*E*I/(u*L)**2", 0),
    "PE": (("s", "P", "Pcr"), "1 - s*P/Pcr", 3),
    "I_d": (("tf", "b"), "tf*b**3/12", 0),
    "A_d": (("b", "tf", "hw", "tw"), "b*tf + hw*tw/6", 1),
    "r_d": (("Id", "Ad"), "sqrt(Id/Ad)", 2),
    "lambda_ed": (("Lc", "rd"), "Lc/rd", 1),
    "lambda_ed_ferme": (("Lc", "W", "Iz", "It"), "sqrt(pi*sqrt(2.6)*Lc*W/sqrt(Iz*It))", 1),
    "sigma_P": (("P", "A", "phi"), "P*1000/(A*phi)", 1),
    "sigma_My": (("C", "M", "phied", "W", "PE"), "C*M*1000/(phied*W*PE)", 1),
    "sigma_Mz": (("C", "M", "W", "PE"), "C*M*1000/(W*PE)", 1),
    "sigma": (("sP", "sMy", "sMz"), "sP + sMy + sMz", 1),
    "tau": (("V", "Av"), "V*1000/Av", 1),
    "sigma_eq": (("sig", "tau"), "sqrt(sig**2 + 3*tau**2)", 1),
    "taux": (("v", "lim"), "v/lim", -1),
}
_FONCTIONS = {"sqrt": math.sqrt, "pi": math.pi, "min": min, "max": max, "abs": abs}


def evaluer(formule: Dict) -> float:
    """Le gabarit d'un bloc du moteur, évalué sur SES termes (non arrondis) : doit redonner ``formule["resultat"]``."""
    noms, expression, _ = GABARITS[formule["gabarit"]]
    valeurs = [t["valeur"] for t in formule["termes"]]
    if len(valeurs) != len(noms):
        raise ValueError(f"« {formule['cle']} » : {len(valeurs)} terme(s) pour le gabarit « {formule['gabarit']} », qui en attend {len(noms)}")
    return eval(expression, {"__builtins__": {}}, {**_FONCTIONS, **dict(zip(noms, valeurs))})  # noqa: S307 — gabarits du dépôt

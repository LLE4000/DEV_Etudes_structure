# -*- coding: utf-8 -*-
"""Barres en acier — les GABARITS des formules que le moteur nomme (note de calcul, fenêtres de calcul).

Le moteur rend des blocs ``{cle, gabarit, symbole, termes [{symbole, valeur, unite}], resultat, unite}`` ; ce fichier
porte le SEUL savoir qui manque pour les écrire : l'expression de chaque gabarit, sur des identifiants pris DANS
L'ORDRE des termes du moteur. C'est la forme des notes acier existantes (acier/corniere/formules.py) : la note en fait
des substitutions « grandeur = formule = valeurs = résultat » avec le même kit.

Chaque gabarit est TESTÉ : évalué sur les termes non arrondis du moteur, il redonne le résultat du moteur à 10⁻⁹ près
(tests/test_barre_historique.py, tests/test_barre_ec3_section.py) — la preuve que la formule imprimée EST celle du calcul.

Unités imprimées de la méthode historique : celles des classeurs du bureau — kN, kN·m, cm, cm², cm⁴, cm³, mm², N/mm² ;
E en kN/cm² (21 000) dans la charge critique, qui s'écrit alors sans facteur : π²·E·I/(u·L)² en kN.
"""
import math
from typing import Dict, Tuple

# gabarit → (identifiants des termes, dans l'ordre du moteur ; expression Python ; décimales du résultat, −1 : pourcentage)
# ---- méthode historique (acier/barre/historique.py) ; « taux » sert aux deux méthodes
GABARITS_HISTORIQUE: Dict[str, Tuple[Tuple[str, ...], str, int]] = {
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
# ---- Eurocode 3 (acier/barre/ec3_*.py) : kN, kN·m, mm, mm², cm³, cm⁴, N/mm². Un gabarit n'a NI plafond NI plancher :
# le moteur n'écrit le bloc que là où la formule s'applique telle quelle (α entre 0 et 1, ρ < 1, a ≤ 0,5…).
GABARITS_EC3: Dict[str, Tuple[Tuple[str, ...], str, int]] = {
    # classe de la section (EN 1993-1-1 tableau 5.2)
    "e_eps": (("fy",), "sqrt(235/fy)", 3),
    "e_alpha": (("N", "c", "t", "fy"), "0.5 + N*1000/(2*c*t*fy)", 3),
    "e_psi": (("N", "A", "fy"), "2*N*1000/(A*fy) - 1", 3),
    "e_lim": (("k", "eps"), "k*eps", 1),
    "e_lim_tube": (("k", "eps"), "k*eps**2", 1),
    "e_lim12_haut": (("k", "eps", "alpha"), "k*eps/(13*alpha - 1)", 1),
    "e_lim12_bas": (("k", "eps", "alpha"), "k*eps/alpha", 1),
    "e_lim3_haut": (("eps", "psi"), "42*eps/(0.67 + 0.33*psi)", 1),
    "e_lim3_bas": (("eps", "psi"), "62*eps*(1 - psi)*sqrt(-psi)", 1),
    # section efficace (EN 1993-1-5 § 4.4)
    "e_lambda_p": (("ct", "eps", "ks"), "ct/(28.4*eps*sqrt(ks))", 3),
    "e_rho_interne": (("lp", "psi"), "(lp - 0.055*(3 + psi))/lp**2", 3),
    "e_rho_console": (("lp",), "(lp - 0.188)/lp**2", 3),
    "e_A_eff": (("A", "n1", "rho1", "c1", "t1", "n2", "rho2", "c2", "t2"), "A - n1*(1 - rho1)*c1*t1 - n2*(1 - rho2)*c2*t2", 0),
    "e_W_eff": (("Ieff", "v", "e"), "Ieff*10/(v + e)", 1),
    # résistances de la section (EN 1993-1-1 § 6.2)
    "e_N_Rd": (("A", "fy", "gM0"), "A*fy/gM0/1000", 1),
    "e_M_Rd": (("W", "fy", "gM0"), "W*fy/gM0/1000", 1),
    "e_Av_plancher": (("eta", "hw", "tw"), "eta*hw*tw", 0),
    "e_V_pl": (("Av", "fy", "gM0"), "Av*fy/(sqrt(3)*gM0)/1000", 1),
    "e_rho_V": (("V", "Vpl"), "(2*V/Vpl - 1)**2", 3),
    "e_M_yV": (("Wpl", "rho", "Aw", "tw", "fy", "gM0"), "(Wpl - rho*Aw**2/(4*tw)/1000)*fy/gM0/1000", 1),
    "e_reduit": (("rho", "R"), "(1 - rho)*R", 1),
    "e_resultante": (("a", "b"), "sqrt(a**2 + b**2)", 1),
    "e_n": (("N", "Npl"), "N/Npl", 3),
    "e_a": (("A", "b", "t"), "(A - 2*b*t)/A", 3),
    "e_M_N": (("Mpl", "n", "a"), "Mpl*(1 - n)/(1 - 0.5*a)", 1),
    "e_M_Nz_I": (("Mpl", "n", "a"), "Mpl*(1 - ((n - a)/(1 - a))**2)", 1),
    "e_beta_I": (("n",), "5*n", 2),
    "e_alpha_tube": (("n",), "1.66/(1 - 1.13*n**2)", 2),
    "e_deviee": (("My", "MNy", "al", "Mz", "MNz", "be"), "(My/MNy)**al + (Mz/MNz)**be", -1),
    "e_sigma2": (("F1", "S1", "F2", "S2"), "F1*1000/S1 + F2*1000/S2", 1),
    "e_sigma3": (("F1", "S1", "F2", "S2", "F3", "S3"), "F1*1000/S1 + F2*1000/S2 + F3*1000/S3", 1),
    "e_sigma_Rd": (("fy", "gM0"), "fy/gM0", 1),
    "e_sigma_Rd_V": (("rho", "fy", "gM0"), "(1 - rho)*fy/gM0", 1),
    "e_lineaire2": (("E1", "R1", "E2", "R2"), "E1/R1 + E2/R2", -1),
    "e_lineaire3": (("E1", "R1", "E2", "R2", "E3", "R3"), "E1/R1 + E2/R2 + E3/R3", -1),
}
GABARITS: Dict[str, Tuple[Tuple[str, ...], str, int]] = {**GABARITS_HISTORIQUE, **GABARITS_EC3}
_FONCTIONS = {"sqrt": math.sqrt, "pi": math.pi, "min": min, "max": max, "abs": abs}


def evaluer(formule: Dict) -> float:
    """Le gabarit d'un bloc du moteur, évalué sur SES termes (non arrondis) : doit redonner ``formule["resultat"]``."""
    noms, expression, _ = GABARITS[formule["gabarit"]]
    valeurs = [t["valeur"] for t in formule["termes"]]
    if len(valeurs) != len(noms):
        raise ValueError(f"« {formule['cle']} » : {len(valeurs)} terme(s) pour le gabarit « {formule['gabarit']} », qui en attend {len(noms)}")
    return eval(expression, {"__builtins__": {}}, {**_FONCTIONS, **dict(zip(noms, valeurs))})  # noqa: S307 — gabarits du dépôt

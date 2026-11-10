# -*- coding: utf-8 -*-
"""Cornière d'appui — les FORMULES de la note (et des fenêtres de calcul), écrites comme à la main.

Le moteur CALCULE et NOMME (R3) : ``verifier(c)["calcul"]`` rend des blocs de formules ``{cle, symbole, termes
[{symbole, valeur, unite}], resultat, unite, cellule}`` dont chaque résultat se retrouve à partir de ses seuls termes
(tests/test_corniere_moteur.py). Ce fichier ajoute le SEUL savoir qui manque pour les écrire : le GABARIT de chaque
clé (``GABARITS`` : l'expression Python des termes, dans l'ordre où le moteur les donne) — et en fait des substitutions
« grandeur = formule = valeurs = résultat » (``Sub`` du kit des assemblages, le même objet que les notes des
assemblages, des soudures et du garde-corps : ``morceaux``, ``rendre``, ``fmt``).

Chaque ligne est TESTABLE deux fois : ``verifier(sub)`` (le kit) ré-évalue le gabarit avec les nombres tels qu'ils sont
IMPRIMÉS et retrouve le résultat du moteur à l'arrondi d'affichage près ; ``exact(sub)`` l'évalue avec les termes non
arrondis et doit le retrouver à 1e-9 — c'est la preuve que le gabarit EST la formule du moteur. Rien n'est recalculé
pour l'affichage : les nombres sont ceux du moteur, les gabarits ne font que les relier (tests/test_note_corniere.py).

Unités imprimées : mm, kN, kN·m, N/mm², mm², mm³ ; les facteurs 10³ et 10⁶ des gabarits (``*1000``, ``*1e6``) sont
effacés de la forme littérale et imprimés dans la forme numérique par le kit.
"""
import math
from typing import Dict, List, Optional, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres.formules import Sub, evaluer, fmt, morceaux, nombre, rendre, verifier  # noqa: F401 (ré-exportés)

# clé du moteur → (identifiants des termes, DANS L'ORDRE des termes du moteur ; gabarit Python ; décimales du résultat :
# -1 = pourcentage). Les identifiants sont libres (le symbole imprimé est celui du terme du moteur) ; « sqrt », « min »,
# « max », « abs » sont les fonctions du kit.
GABARITS: Dict[str, Tuple[Tuple[str, ...], str, int]] = {
    "sigma_adm": (("fy", "coef"), "fy/coef", 1),
    "tau_adm": (("sa",), "sa/sqrt(3)", 1),
    "sigma_lim_ELU": (("fy", "g"), "fy/g", 1),                # ELU : f_y/γ_M0 (§ 9.5)
    "tau_lim_ELU": (("fy", "g"), "fy/(sqrt(3)*g)", 1),
    "N_ELU_direct": (("H", "n"), "H/n", 2),                   # ELU saisi : par ancrage, directement
    "V_ELU_direct": (("R", "n"), "R/n", 2),
    "Xp": (("X", "dX"), "X + dX", 1),
    "Yp": (("Y", "dY"), "Y + dY", 1),
    "Xe": (("X", "r", "e"), "X - r - e", 1),
    "Xer": (("X", "e"), "X - e", 1),
    "Ye": (("Y", "e", "r"), "Y - e - r", 1),
    "Yer": (("Y", "e"), "Y - e", 1),
    "YH": (("Y", "e"), "Y - e/2", 1),
    "M_bras_j": (("R", "bras", "Xj"), "R*(bras + Xj)/1000", 3),
    "Mmax_j": (("R", "X", "Xj"), "R*(X + Xj)/1000", 3),
    "rho": (("Yref", "Yj", "YH"), "(Yref + Yj)/(YH + Yj)", 4),
    "M_rho": (("rho", "Mmax"), "rho*Mmax", 3),
    "H_hist": (("Mmax", "YH", "Yj"), "Mmax*1000/(YH + Yj)", 2),
    "M_hist_v": (("Hs", "Yp", "s"), "Hs*(Yp - s)/1000", 3),
    "M_bras": (("R", "bras"), "R*bras/1000", 3),
    "M_lin": (("R", "Xp", "s"), "R*(Xp - s)/1000", 3),
    "bras_A": (("Yp", "yc"), "Yp - yc", 1),
    "H_eq": (("R", "Xp", "uV", "bras"), "R*(Xp - uV)/bras", 2),
    "M_eq_A": (("H", "Yp", "s", "R", "e", "uV"), "H*(Yp - s)/1000 - R*(e/2 - uV)/1000", 3),
    "y_a": (("haut", "Yp"), "haut - Yp", 1),
    "bras_B": (("ya", "c"), "ya - c", 1),
    "M_angle_B": (("R", "Xp", "e"), "R*(Xp - e/2)/1000", 3),
    "M_bas_B": (("H", "sc", "s"), "H*(sc - s)/1000", 3),
    "t_conge": (("e", "r", "s"), "e + r - sqrt(r**2 - (e + r - s)**2)", 2),
    "W": (("b", "t"), "b*t**2/6", 0),
    "A": (("b", "t"), "b*t", 0),
    "sigma_M": (("M", "W"), "M*1e6/W", 1),
    "sigma_MN": (("M", "W", "N", "A"), "abs(M)*1e6/W + abs(N)*1000/A", 1),
    "tau_RLt": (("R", "b", "t"), "R*1000/(b*t)", 2),
    "tau_VA": (("V", "A"), "abs(V)*1000/A", 2),
    "sigma_eq": (("s", "t"), "sqrt(s**2 + 3*t**2)", 1),
    "taux": (("v", "lim"), "v/lim", -1),
    "N_ELS_hist": (("Mmax", "Y", "jeu", "n"), "Mmax*1000/((Y - jeu)*n)", 2),
    "N_ELS_eq": (("H", "n"), "H/n", 2),
    "V_ELS": (("R", "n"), "R/n", 2),
    "rapport_ELU": (("RELU", "R"), "RELU/R", 3),
    "ELU": (("x", "rho"), "x*rho", 2),
    "R_ELU_auto": (("R",), "1.5*R", 2),
    "p_diam": (("V", "d", "e"), "V*1000/(d*e)", 1),
    "arrachement": (("V", "haut", "Y", "e"), "V*1000/(2*(haut - Y)*e)", 1),
    "L_eff": (("L", "a", "Xe"), "min(L, a + 2*Xe)", 1),
    "d0": (("d",), "d + 2", 0),
    "b_net": (("L", "n", "d0"), "L - n*d0", 0),
    "pression_support": (("C", "L"), "C*1000/L", 1),
}
# les symboles du moteur que la note écrit autrement : le taux s'appelle η ; le moment d'une section repère porte
# son numéro (le moteur dit « M(s) » ou « M » quand il le calcule à l'abscisse de la section)
SYMBOLES = {"taux": "η", "σ/σ_adm": "η_σ", "τ/τ_adm": "η_τ"}
MOMENTS_SECTIONS = {"S1": "M1", "S2": "M2", "S3": "M3", "S4": "M4"}
# les familles de clés : la chaîne qui mène au moment d'une coupe (positions, bras, H, M), la coupe elle-même
# (épaisseur, module, aire, contraintes, taux), la largeur efficace, la section nette
CHAINE = ("Xp", "Yp", "Xe", "Xer", "Ye", "Yer", "YH", "bras_A", "y_a", "bras_B", "Mmax_j", "M_bras_j", "rho", "M_rho",
          "H_hist", "M_hist_v", "M_bras", "M_lin", "H_eq", "M_eq_A", "M_angle_B", "M_bas_B")
COUPE = ("t_conge", "W", "A", "sigma_M", "sigma_MN", "tau_RLt", "tau_VA", "sigma_eq", "taux")
TAUX_DETAIL = ("η_σ", "η_τ")        # les taux σ/σ_adm et τ/τ_adm : des lignes du détail, hors du verdict (le classeur)


def sub(f: Dict, bloc: Optional[str] = None) -> Sub:
    """La substitution d'UNE formule du moteur ``f`` (``{cle, symbole, termes, resultat, unite, cellule}``) dans le bloc
    ``bloc`` (« S3 » renomme « M(s) » en « M3 »). KeyError si la clé n'a pas de gabarit, ValueError si le nombre de
    termes n'est pas celui du gabarit : une formule nouvelle du moteur ne passe pas en silence."""
    idents, gabarit, nd = GABARITS[f["cle"]]
    termes = f["termes"]
    if len(termes) != len(idents):
        raise ValueError(f"formule {f['cle']} : {len(termes)} terme(s) pour un gabarit à {len(idents)}")
    nom = SYMBOLES.get(f["symbole"], f["symbole"])
    if f["cle"] in ("M_eq_A", "M_hist_v", "M_angle_B", "M_bas_B", "M_lin") and bloc in MOMENTS_SECTIONS:
        nom = MOMENTS_SECTIONS[bloc]
    vals = {ident: (t["symbole"], t["valeur"]) for ident, t in zip(idents, termes)}
    return Sub(nom, gabarit, vals, f["resultat"], f["unite"], nd)


def subs(bloc: Dict) -> List[Sub]:
    """Toutes les substitutions d'un bloc du moteur, dans son ordre."""
    return [sub(f, bloc["cle"]) for f in bloc["formules"]]


def toutes(res: Dict) -> List[Tuple[Dict, List[Sub]]]:
    """``[(bloc, [Sub, …]), …]`` pour tous les blocs de ``res["calcul"]`` — ce que le test ré-évalue ligne à ligne."""
    return [(b, subs(b)) for b in res["calcul"]]


def bloc(res: Dict, cle: str) -> Optional[Dict]:
    """Le bloc de formules d'une clé (« S3 », « ancrages », « largeur_efficace », « materiau »…), ou None."""
    return next((b for b in res["calcul"] if b["cle"] == cle), None)


def chaine(b: Dict) -> List[Sub]:
    """La chaîne qui mène au moment d'une coupe : positions, bras de levier, H, M — les clés ``CHAINE`` du bloc."""
    return [sub(f, b["cle"]) for f in b["formules"] if f["cle"] in CHAINE]


def coupe(b: Dict, detail: bool = False) -> List[Sub]:
    """Les formules de la coupe elle-même (t(s), W, A, σ, τ, σ_éq, η) ; ``detail`` garde aussi η_σ et η_τ."""
    out = [sub(f, b["cle"]) for f in b["formules"] if f["cle"] in COUPE]
    return out if detail else [s for s in out if s.nom not in TAUX_DETAIL]


def cellules(b: Dict, cles=None) -> List[str]:
    """Les cellules du classeur des formules d'un bloc (mode historique ; vide sinon), sans doublon, dans l'ordre."""
    out: List[str] = []
    for f in b["formules"]:
        if cles is not None and f["cle"] not in cles:
            continue
        if f.get("cellule") and f["cellule"] not in out:
            out.append(f["cellule"])
    return out


def exact(s: Sub, tol: float = 1e-9) -> Tuple[bool, float]:
    """``(ok, valeur)`` : le gabarit évalué avec les termes NON arrondis redonne-t-il le résultat du moteur à ``tol``
    relatif ? C'est la preuve que le gabarit est la formule du moteur (``verifier`` du kit teste l'arrondi d'affichage)."""
    env = {k: float(v[1]) for k, v in s.vals.items()}
    env.update(sqrt=math.sqrt, min=min, max=max, abs=abs)
    v = eval(s.template, {"__builtins__": {}}, env)  # noqa: S307 — gabarits internes
    ref = s.res
    if abs(ref) < 1e-12:
        return abs(v) < 1e-9, v
    return abs(v - ref) / abs(ref) <= tol, v


def cle_identique(s: Sub) -> Tuple:
    """La clé d'une substitution déjà écrite (même nom, même gabarit, mêmes valeurs) : la note n'écrit pas deux fois
    Mmax (C55) pour S3 et S4."""
    return (s.nom, s.template, tuple((k, v[0], round(float(v[1]), 9)) for k, v in s.vals.items()))

# -*- coding: utf-8 -*-
"""Barres en acier — les GABARITS des formules que le moteur nomme (note de calcul, fenêtres de calcul).

Le moteur rend des blocs ``{cle, gabarit, symbole, termes [{symbole, valeur, unite}], resultat, unite}`` ; ce fichier
porte le SEUL savoir qui manque pour les écrire : l'expression de chaque gabarit, sur des identifiants pris DANS
L'ORDRE des termes du moteur. C'est la forme des notes acier existantes (acier/corniere/formules.py) : la note en fait
des substitutions « grandeur = formule = valeurs = résultat » avec le même kit.

Chaque gabarit est TESTÉ : évalué sur les termes non arrondis du moteur, il redonne le résultat du moteur à 10⁻⁹ près
(tests/test_barre_historique.py, tests/test_barre_ec3_section.py, tests/test_barre_ec3_barre.py) — la preuve que la
formule imprimée EST celle du calcul.

Unités imprimées de la méthode historique : celles des classeurs du bureau — kN, kN·m, cm, cm², cm⁴, cm³, mm², N/mm² ;
E en kN/cm² (21 000) dans la charge critique, qui s'écrit alors sans facteur : π²·E·I/(u·L)² en kN.

UN FACTEUR D'UNITÉS s'écrit ``1eN`` (``*1e3``, ``/1e5``…) — jamais ``1000`` : c'est ce qui le distingue d'un coefficient
de la norme (le 10 de c_LT, le 235 de ε). L'écriture (acier/barre/ecriture.py) l'ôte de la formule en symboles et
l'imprime « × 10³ » dans la formule en valeurs.
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
    "P_adm": (("A", "phi", "fy", "s"), "A*phi*fy/s/1e3", 1),
    "P_cr": (("E", "I", "u", "L"), "pi**2*E*I/(u*L)**2", 0),
    "PE": (("s", "P", "Pcr"), "1 - s*P/Pcr", 3),
    "I_d": (("tf", "b"), "tf*b**3/12", 0),
    "A_d": (("b", "tf", "hw", "tw"), "b*tf + hw*tw/6", 1),
    "r_d": (("Id", "Ad"), "sqrt(Id/Ad)", 2),
    "lambda_ed": (("Lc", "rd"), "Lc/rd", 1),
    "lambda_ed_ferme": (("Lc", "W", "Iz", "It"), "sqrt(pi*sqrt(2.6)*Lc*W/sqrt(Iz*It))", 1),
    "sigma_P": (("P", "A", "phi"), "P*1e3/(A*phi)", 1),
    "sigma_My": (("C", "M", "phied", "W", "PE"), "C*M*1e3/(phied*W*PE)", 1),
    "sigma_Mz": (("C", "M", "W", "PE"), "C*M*1e3/(W*PE)", 1),
    "sigma": (("sP", "sMy", "sMz"), "sP + sMy + sMz", 1),
    "tau": (("V", "Av"), "V*1e3/Av", 1),
    "sigma_eq": (("sig", "tau"), "sqrt(sig**2 + 3*tau**2)", 1),
    "taux": (("v", "lim"), "v/lim", -1),
}
# ---- Eurocode 3 (acier/barre/ec3_*.py) : kN, kN·m, mm, mm², cm³, cm⁴, N/mm². Un gabarit n'a NI plafond NI plancher :
# le moteur n'écrit le bloc que là où la formule s'applique telle quelle (α entre 0 et 1, ρ < 1, a ≤ 0,5…).
GABARITS_EC3_SECTION: Dict[str, Tuple[Tuple[str, ...], str, int]] = {
    # classe de la section (EN 1993-1-1 tableau 5.2)
    "e_eps": (("fy",), "sqrt(235/fy)", 3),
    "e_alpha": (("N", "c", "t", "fy"), "0.5 + N*1e3/(2*c*t*fy)", 3),
    "e_psi": (("N", "A", "fy"), "2*N*1e3/(A*fy) - 1", 3),
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
    "e_W_eff": (("Ieff", "v", "e"), "Ieff*1e1/(v + e)", 1),
    # résistances de la section (EN 1993-1-1 § 6.2)
    "e_N_Rd": (("A", "fy", "gM0"), "A*fy/gM0/1e3", 1),
    "e_M_Rd": (("W", "fy", "gM0"), "W*fy/gM0/1e3", 1),
    "e_Av_plancher": (("eta", "hw", "tw"), "eta*hw*tw", 0),
    "e_V_pl": (("Av", "fy", "gM0"), "Av*fy/(sqrt(3)*gM0)/1e3", 1),
    "e_rho_V": (("V", "Vpl"), "(2*V/Vpl - 1)**2", 3),
    "e_M_yV": (("Wpl", "rho", "Aw", "tw", "fy", "gM0"), "(Wpl - rho*Aw**2/(4*tw)/1e3)*fy/gM0/1e3", 1),
    "e_reduit": (("rho", "R"), "(1 - rho)*R", 1),
    "e_resultante": (("a", "b"), "sqrt(a**2 + b**2)", 1),
    "e_n": (("N", "Npl"), "N/Npl", 3),
    "e_a": (("A", "b", "t"), "(A - 2*b*t)/A", 3),
    "e_M_N": (("Mpl", "n", "a"), "Mpl*(1 - n)/(1 - 0.5*a)", 1),
    "e_M_Nz_I": (("Mpl", "n", "a"), "Mpl*(1 - ((n - a)/(1 - a))**2)", 1),
    "e_beta_I": (("n",), "5*n", 2),
    "e_alpha_tube": (("n",), "1.66/(1 - 1.13*n**2)", 2),
    "e_deviee": (("My", "MNy", "al", "Mz", "MNz", "be"), "(My/MNy)**al + (Mz/MNz)**be", -1),
    "e_sigma2": (("F1", "S1", "F2", "S2"), "F1*1e3/S1 + F2*1e3/S2", 1),
    "e_sigma3": (("F1", "S1", "F2", "S2", "F3", "S3"), "F1*1e3/S1 + F2*1e3/S2 + F3*1e3/S3", 1),
    "e_sigma_Rd": (("fy", "gM0"), "fy/gM0", 1),
    "e_sigma_Rd_V": (("rho", "fy", "gM0"), "(1 - rho)*fy/gM0", 1),
    "e_lineaire2": (("E1", "R1", "E2", "R2"), "E1/R1 + E2/R2", -1),
    "e_lineaire3": (("E1", "R1", "E2", "R2", "E3", "R3"), "E1/R1 + E2/R2 + E3/R3", -1),
}
GABARITS_EC3_BARRE: Dict[str, Tuple[Tuple[str, ...], str, int]] = {
    # flambement (EN 1993-1-1 § 6.3.1) : I en cm⁴, I_w en cm⁶, L en m, forces en kN
    "e_N_cr": (("E", "I", "Lcr"), "pi**2*E*I/Lcr**2/1e5", 0),
    "e_lambda_bar": (("A", "fy", "Ncr"), "sqrt(A*fy/(Ncr*1e3))", 3),
    "e_Phi": (("alpha", "lb"), "0.5*(1 + alpha*(lb - 0.2) + lb**2)", 3),
    "e_chi": (("Phi", "lb"), "1/(Phi + sqrt(Phi**2 - lb**2))", 3),
    "e_N_b_Rd": (("chi", "A", "fy", "gM1"), "chi*A*fy/gM1/1e3", 1),
    "e_i0": (("iy", "iz"), "sqrt(iy**2 + iz**2)", 1),
    "e_N_cr_T": (("G", "It", "E", "Iw", "LT", "i0"), "(G*It*1e4 + pi**2*E*Iw/LT**2)/i0**2/1e3", 0),
    # déversement (§ 6.3.2)
    "e_M_cr": (("lam", "Mmax"), "lam*Mmax", 1),
    "e_lambda_LT": (("W", "fy", "Mcr"), "sqrt(W*fy/(Mcr*1e3))", 3),
    "e_Phi_LT": (("alpha", "lb", "lb0", "beta"), "0.5*(1 + alpha*(lb - lb0) + beta*lb**2)", 3),
    "e_chi_LT": (("Phi", "beta", "lb"), "1/(Phi + sqrt(Phi**2 - beta*lb**2))", 3),
    "e_chi_LT_plafond": (("lb",), "1/lb**2", 3),
    "e_kc_lineaire": (("psi",), "1/(1.33 - 0.33*psi)", 3),
    "e_f": (("kc", "lb"), "1 - 0.5*(1 - kc)*(1 - 2*(lb - 0.8)**2)", 3),
    "e_chi_LT_mod": (("chi", "f"), "chi/f", 3),
    "e_M_b_Rd": (("chi", "W", "fy", "gM1"), "chi*W*fy/gM1/1e3", 1),
    # flexion composée avec instabilité (§ 6.3.3, annexe A)
    "e_rapport": (("N", "Ncr"), "N/Ncr", 4),
    "e_mu": (("r", "chi"), "(1 - r)/(1 - chi*r)", 3),
    "e_w": (("Wpl", "Wel"), "Wpl/Wel", 3),
    "e_npl": (("N", "NRk", "gM1"), "N/(NRk/gM1)", 3),
    "e_aLT": (("It", "Iy"), "1 - It/Iy", 3),
    "e_lambda_0_lim": (("C1", "rz", "rT"), "0.2*sqrt(C1)*((1 - rz)*(1 - rT))**0.25", 3),
    "e_Cm0_lineaire": (("psi", "N", "Ncr"), "0.79 + 0.21*psi + 0.36*(psi - 0.33)*N/Ncr", 3),
    "e_Cm0_lineaire_sans_N": (("psi",), "0.79 + 0.21*psi", 3),
    "e_Cm0_reparti": (("N", "Ncr"), "1 + 0.03*N/Ncr", 3),
    "e_Cm0_ponctuel": (("N", "Ncr"), "1 - 0.18*N/Ncr", 3),
    "e_Cm0_general": (("E", "I", "delta", "L", "M", "N", "Ncr"), "1 + (pi**2*E*I*delta/(L**2*M)/1e8 - 1)*N/Ncr", 3),
    "e_Cm0_borne": (("N", "Ncr"), "1 - N/Ncr", 3),
    "e_eps_y": (("M", "N", "A", "W"), "M*A/(N*W)", 3),
    "e_Cmy": (("Cm0", "epsy", "aLT"), "Cm0 + (1 - Cm0)*sqrt(epsy)*aLT/(1 + sqrt(epsy)*aLT)", 3),
    "e_CmLT": (("Cmy", "aLT", "rz", "rT"), "Cmy**2*aLT/sqrt((1 - rz)*(1 - rT))", 3),
    "e_bLT": (("aLT", "lb0", "My", "chiLT", "Mply", "Mz", "Mplz"), "0.5*aLT*lb0**2*My/(chiLT*Mply)*Mz/Mplz", 3),
    "e_cLT": (("aLT", "lb0", "lbz", "My", "Cmy", "chiLT", "Mply"), "10*aLT*lb0**2/(5 + lbz**4)*My/(Cmy*chiLT*Mply)", 3),
    "e_dLT": (("aLT", "lb0", "lbz", "My", "Cmy", "chiLT", "Mply", "Mz", "Cmz", "Mplz"),
              "2*aLT*lb0/(0.1 + lbz**4)*My/(Cmy*chiLT*Mply)*Mz/(Cmz*Mplz)", 3),
    "e_eLT": (("aLT", "lb0", "lbz", "My", "Cmy", "chiLT", "Mply"), "1.7*aLT*lb0/(0.1 + lbz**4)*My/(Cmy*chiLT*Mply)", 3),
    "e_Cyy": (("w", "Cm", "lbmax", "npl", "LT"), "1 + (w - 1)*((2 - 1.6/w*Cm**2*lbmax - 1.6/w*Cm**2*lbmax**2)*npl - LT)", 3),
    "e_Czz": (("w", "Cm", "lbmax", "npl", "LT"), "1 + (w - 1)*((2 - 1.6/w*Cm**2*lbmax - 1.6/w*Cm**2*lbmax**2)*npl - LT)", 3),
    "e_Cyz": (("w", "Cm", "lbmax", "npl", "LT"), "1 + (w - 1)*((2 - 14*Cm**2*lbmax**2/w**5)*npl - LT)", 3),
    "e_Czy": (("w", "Cm", "lbmax", "npl", "LT"), "1 + (w - 1)*((2 - 14*Cm**2*lbmax**2/w**5)*npl - LT)", 3),
    "e_kyy": (("Cmy", "CmLT", "mu", "r", "C"), "Cmy*CmLT*mu/(1 - r)/C", 3),
    "e_kzy": (("Cmy", "CmLT", "mu", "r", "C", "wy", "wz"), "Cmy*CmLT*mu/(1 - r)/C*0.6*sqrt(wy/wz)", 3),
    "e_kyz": (("Cmz", "mu", "r", "C", "wz", "wy"), "Cmz*mu/(1 - r)/C*0.6*sqrt(wz/wy)", 3),
    "e_kzz": (("Cmz", "mu", "r", "C"), "Cmz*mu/(1 - r)/C", 3),
    "e_k_el_y": (("Cmy", "CmLT", "mu", "r"), "Cmy*CmLT*mu/(1 - r)", 3),
    "e_k_el_z": (("Cmz", "mu", "r"), "Cmz*mu/(1 - r)", 3),
    "e_66_N_My_Mz": (("N", "chi", "NRk", "ky", "My", "chiLT", "MyRk", "kz", "Mz", "MzRk", "gM1"),
                     "N/(chi*NRk/gM1) + ky*My/(chiLT*MyRk/gM1) + kz*Mz/(MzRk/gM1)", -1),
    "e_66_N_My": (("N", "chi", "NRk", "ky", "My", "chiLT", "MyRk", "gM1"), "N/(chi*NRk/gM1) + ky*My/(chiLT*MyRk/gM1)", -1),
    "e_66_N_Mz": (("N", "chi", "NRk", "kz", "Mz", "MzRk", "gM1"), "N/(chi*NRk/gM1) + kz*Mz/(MzRk/gM1)", -1),
    "e_66_My_Mz": (("ky", "My", "chiLT", "MyRk", "kz", "Mz", "MzRk", "gM1"), "ky*My/(chiLT*MyRk/gM1) + kz*Mz/(MzRk/gM1)", -1),
}
GABARITS_EC3: Dict[str, Tuple[Tuple[str, ...], str, int]] = {**GABARITS_EC3_SECTION, **GABARITS_EC3_BARRE}
GABARITS: Dict[str, Tuple[Tuple[str, ...], str, int]] = {**GABARITS_HISTORIQUE, **GABARITS_EC3}
_FONCTIONS = {"sqrt": math.sqrt, "pi": math.pi, "min": min, "max": max, "abs": abs}


def evaluer(formule: Dict) -> float:
    """Le gabarit d'un bloc du moteur, évalué sur SES termes (non arrondis) : doit redonner ``formule["resultat"]``."""
    noms, expression, _ = GABARITS[formule["gabarit"]]
    valeurs = [t["valeur"] for t in formule["termes"]]
    if len(valeurs) != len(noms):
        raise ValueError(f"« {formule['cle']} » : {len(valeurs)} terme(s) pour le gabarit « {formule['gabarit']} », qui en attend {len(noms)}")
    return eval(expression, {"__builtins__": {}}, {**_FONCTIONS, **dict(zip(noms, valeurs))})  # noqa: S307 — gabarits du dépôt

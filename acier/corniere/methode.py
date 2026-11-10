# -*- coding: utf-8 -*-
"""Cornière d'appui — les TEXTES DE MÉTHODE, réduits à ce qu'un autre ingénieur doit lire pour comprendre la note
(révision du 10/11/2026 au soir, ``docs/acier/CORNIERE_DECISIONS.md`` § 9.1 : la page 2 est dominée par les dessins).

Ce qui reste du texte : un RÉSUMÉ de trois lignes au plus sous le titre de la page 2 (``resume`` : le modèle et
l'état limite avec ses limites, le mode et la configuration avec ses bras, les jeux et l'enveloppe — et la largeur
efficace quand elle joue), et la ligne des RÉFÉRENCES (``REFERENCES``). Les chiffres viennent du RÉSULTAT du moteur
(``verifier``) : rien n'est recalculé, les phrases ne font que les relier. Les LIMITES (chevilles non vérifiées, pic
au congé, effet de levier, support…) ne sont pas réécrites : ce sont les avertissements du moteur
(``res["avertissements"]``), que la note imprime abrégés si besoin — une seule source. L'écran peut lire ``resume`` au
même endroit.
"""
from typing import Dict, List

from acier.formats import F, f0

from .entree import ELS, HISTORIQUE, LIBELLES_CONFIGURATION

# les références de la note, courtes : la note les pose sur une ligne, séparées par « · »
REFERENCES = (
    "Classeur de calcul du bureau, feuille P8 (cornière d'appui, cas de contrôle L 120×120×15 S355)",
    "Décisions du module refondu, 10/11/2026 (configurations A et B, modes, jeux par l'enveloppe, état limite)",
    "EN 10056-1 (cornières) · EN 1993-1-1 Tab. 3.1 (f_y) et § 6.2.1(5) (critère élastique ELU, γ_M0 = 1,0 ANB)",
    "EN 1993-1-8 Tab. 3.3 (s ≥ 2,2·d0, e₁ ≥ 1,2·d0) · EN 1992-4 pour les chevilles (Hilti PROFIS, N et V par ancrage)",
)
LIGNES_RESUME = 3           # le résumé tient en trois lignes au plus : c'est la règle, et le test la garde


def _f1(x: float) -> str:
    return F(x, 1)


def resume(res: Dict) -> List[str]:
    """Les lignes du résumé de méthode d'une cornière (trois au plus), à partir du résultat du moteur : le modèle et
    l'état limite (ses limites σ et τ, le critère) ; le mode et la configuration (le pivot, les bras, ce qui est tendu
    ou comprimé) ; les jeux et l'enveloppe, la largeur efficace quand a est donnée. Les symboles indicés (σ_adm, u_V,
    L_eff…) sont composés par la note (``ndc_pdf.riche.auto``)."""
    g, mat, cfg, mode, c = res["geometrie"], res["materiau"], res["configuration"], res["mode"], res["corniere"]
    tranche = f"tranche courte (L = {f0(g['L'])} mm, {g['n']} ancrage{'s' if g['n'] > 1 else ''} qui se partagent R et la traction H)"
    if mat["etat_limite"] == ELS:
        l1 = (f"Cornière d'appui en {tranche}, méthode aux contraintes admissibles sur les efforts ELS : "
              f"σ_éq = √(σ² + 3τ²) ≤ σ_adm = f_y/{f0(mat['coef'])} = {_f1(mat['sigma_adm'])} N/mm², τ ≤ τ_adm = σ_adm/√3 = "
              f"{_f1(mat['tau_adm'])} N/mm² — pas une justification EC3 aux états limites.")
    else:
        l1 = (f"Cornière d'appui en {tranche}, critère élastique de von Mises à l'ELU (EN 1993-1-1 § 6.2.1(5)) sur la réaction R "
              f"saisie à l'ELU : σ_éq = √(σ² + 3τ²) ≤ f_y/γ_M0 = {_f1(mat['sigma_adm'])} N/mm², τ ≤ f_y/(√3·γ_M0) = "
              f"{_f1(mat['tau_adm'])} N/mm², γ_M0 = {F(mat['gamma_M0'], 1)} (ANB).")
    conf = f"Configuration {cfg} ({LIBELLES_CONFIGURATION[cfg]})"
    if cfg == "B":
        l2 = (f"{conf}, toujours l'équilibre : la cornière pivote autour du bout libre (contact à c = {f0(g['yc'])} mm), "
              f"H = R·(X′ − u_V)/(y_a − c) avec u_V = {f0(g['uV'])} mm ; aile verticale comprimée (N = −R), moment constant "
              f"R·(X′ − e/2) de l'ancrage à l'angle, section nette au trou ; la méthode du classeur ne s'applique pas.")
    elif mode == HISTORIQUE:
        l2 = (f"{conf}, méthode historique (feuille P8 du classeur) : sections S1 à S4 avec le pivot e/2 (YH = Y − e/2 = "
              f"{f0(c['Y'] - g['e'] / 2)} mm), τ = R/(L·t) dans les deux ailes, traction d'ancrage sur le bras Y − jeu = "
              f"{f0(c['Y'] - g['jeu_ancrage'])} mm (C76 = {f0(g['jeu_ancrage'])}) ; le maximum du congé en information ; "
              f"la garde calcule l'équilibre P1 à côté.")
    else:
        l2 = (f"{conf}, équilibre d'un seul corps libre, frottement nul au mur : compression du mur à y_c = {f0(g['yc'])} mm du "
              f"dessous, cisaillement d'ancrage à u_V = {f0(g['uV'])} mm du nu, H = R·(X′ − u_V)/(Y′ − y_c) ; aile verticale "
              f"tendue (N = +R), le maximum le long du congé entre dans le verdict.")
    if g["dx"] > 0 or g["dy"] > 0:
        l3 = (f"Jeux par l'enveloppe : charge à X ± {f0(g['dx'])}, ancrage à Y ± {f0(g['dy'])} — {len(res['combinaisons'])} combinaisons, "
              f"chaque grandeur garde la plus défavorable avec celle qui la gouverne ; les vues montrent une courbe par position de "
              f"la charge et l'enveloppe en gras.")
    else:
        l3 = "Sans jeu (dx = dy = 0) : une seule combinaison, la charge à X et l'ancrage à Y."
    if g["a"] is not None:
        l3 += (f" Largeur d'appui a = {f0(g['a'])} mm : L_eff = min(L ; a + 2·Xe) = {_f1(g['L_eff'])} mm sur l'aile horizontale "
               f"(σ × {F(g['facteur_L_eff'], 3)}).")
    return [l1, l2, l3]

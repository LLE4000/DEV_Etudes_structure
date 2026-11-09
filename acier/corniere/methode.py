# -*- coding: utf-8 -*-
"""Cornière d'appui — les TEXTES DE MÉTHODE, courts, écrits une fois : la page 2 de la note (``note.py``) les lit ici,
et l'écran peut les lire au même endroit. Ils disent le MODÈLE, la méthode aux contraintes admissibles, le mode de
calcul (historique ou équilibre), la configuration (A ou B) et ce qui change quand on retourne la cornière, les jeux
et l'enveloppe, les références. Les hypothèses CHIFFRÉES (σ_adm, bras, jeux…) sont imprimées par la note elle-même, en
page 1, à partir du résultat du moteur ; ici, seulement les règles (``docs/acier/CORNIERE_DECISIONS.md`` § 1 et § 3).
Les LIMITES (chevilles non vérifiées, pic au congé, effet de levier, support…) ne sont pas réécrites : ce sont les
avertissements du moteur (``res["avertissements"]``), que la note imprime tels quels — une seule source.
"""
from typing import Dict, List

from .entree import EQUILIBRE, HISTORIQUE

MODELE = {
    "titre": "Modèle",
    "paragraphes": [
        "Cornière d'appui en TRANCHE COURTE : la réaction R (ELS) pose sur l'aile horizontale à X du nu du mur, sur la "
        "longueur reprise L ; l'aile verticale est fixée au mur par n ancrages, à Y de la face extérieure de l'aile horizontale. "
        "L est la largeur de toutes les sections ; les n ancrages se partagent R/n et la traction H/n. Sections repères S1 à S4 "
        "au pied du congé (t = e) et au nu de l'autre aile (t = e + r) ; dans le congé t(s) = e + r − √(r² − (e + r − s)²) ; sous "
        "s = e, le bloc d'angle, hors théorie des poutres. Contraintes admissibles : σ_adm = f_y/coef, τ_adm = σ_adm/√3, "
        "σ_éq = √(σ² + 3τ²) ≤ σ_adm sur les efforts ELS non majorés — pas une justification EC3 aux états limites.",
    ]}

HISTORIQUE_A = {
    "titre": "Mode historique — la feuille P8 du classeur, mot pour mot",
    "paragraphes": [
        "Bras NOMINAUX : Xe = X − r − e (C19), Xe+r = X − e (C20), Ye = Y − e − r (C23), Ye+r = Y − e (C24), YH = Y − e/2 (C25) ; "
        "le jeu Xj de la charge s'ajoute dans les moments et Yj de l'ancrage dans ρ : M1 = R·(Xe + Xj), M2 = R·(Xe+r + Xj), "
        "Mmax = R·(X + Xj), ρ₃ = (Ye + Yj)/(YH + Yj), M3 = ρ₃·Mmax, ρ₄ = (Ye+r + Yj)/(YH + Yj), M4 = ρ₄·Mmax.",
        "σ = M/W et τ = R/(L·t) dans les DEUX ailes (le classeur, même dans l'aile verticale où c'est en réalité un effort "
        "normal) ; taux = σ_éq/σ_adm, verdict = le plus grand des quatre (D3). Ancrages : N_ELS = Mmax/((Y − jeu)·n) avec le jeu "
        "C76, V_ELS = R/n ; pression diamétrale V_ELS/(d·e) et arrachement V_ELS/(2·(haut − Y)·e) en information, sans limite.",
        "Le maximum de σ_éq le long du congé est calculé et montré (vue Contraintes) mais reste en INFORMATION : le classeur ne "
        "l'a pas, et la méthode historique rend son cas de contrôle exactement. Garde : l'équilibre P1 (pivot e/2, u_V = e/2) "
        "est calculé à côté ; s'il donne un taux plus grand sur l'aile verticale (ancrage bas), un avertissement rouge le dit.",
    ]}

EQUILIBRE_A = {
    "titre": "Mode équilibre — un seul corps libre",
    "paragraphes": [
        "Frottement nul au mur : la compression du mur C passe à y_c du dessous de la cornière, le cisaillement d'ancrage à u_V "
        "du nu du mur ; H = R·(X′ − u_V)/(Y′ − y_c) pour chaque combinaison (X′, Y′). Aile horizontale (e ≤ s ≤ X′) : "
        "M = R·(X′ − s), V = R, N = 0. Aile verticale sous l'ancrage : M = H·(Y′ − s) − R·(e/2 − u_V), N = +R (traction), "
        "V = H ; rien au-dessus de l'ancrage.",
        "Le maximum de σ_éq est cherché le long de t(s) dans chaque aile (pas ≤ 0,1 mm dans le congé) et ENTRE dans le "
        "verdict, avec S1 à S4. Ancrages : N_ELS = H/n, V_ELS = R/n ; compression du mur C = H, résultante à y_c (information).",
    ]}

CONFIGURATION_B = {
    "titre": "Configuration B — aile horizontale en haut (toujours l'équilibre)",
    "paragraphes": [
        "L'aile verticale DESCEND le long du mur, l'ancrage est au-dessous de l'angle ; la cornière pivote autour du bout libre "
        "de l'aile verticale, qui s'appuie sur le mur (contact à c du bout). y_a = haut − Y′ ; H = R·(X′ − u_V)/(y_a − c). "
        "Aile horizontale comme en A. Aile verticale entre l'angle et l'ancrage : moment CONSTANT et maximal M = R·(X′ − e/2) "
        "(statique de la partie au-dessus de la coupe, où seule la charge agit : il ne dépend pas de l'ancrage), "
        "N = −R (compression), V = 0 ; entre l'ancrage et le contact : M = H·(s_c − s), V = H.",
        "Le trou d'ancrage est dans le moment maximal : SECTION NETTE W_net = (L − n·d0)·e²/6, A_net = (L − n·d0)·e, les deux "
        "faces du trou vérifiées (au-dessus : M = R·(X′ − e/2), N = −R ; au-dessous : V = H). Pression du bout libre sur le "
        "support C/L en information et en avertissement : le support n'est pas connu du module. Les formules du classeur ne "
        "s'appliquent pas à B (elles donneraient −18 % sur M3) : le calcul est toujours en mode équilibre.",
    ]}

RETOURNEMENT = {
    "titre": "Ce qui change quand on retourne la cornière",
    "paragraphes": [
        "La cote Y désigne le MÊME trou dans les deux configurations (depuis la face extérieure de l'aile horizontale), mais la "
        "statique change : en A le talon s'appuie sur le mur, l'aile verticale est TENDUE (N = +R) et son moment croît de "
        "l'ancrage vers l'angle ; en B la cornière s'appuie par le bout libre, l'aile verticale est COMPRIMÉE (N = −R) et son "
        "moment est constant et maximal de l'ancrage à l'angle, là où est le trou (section nette). Le bras de la traction passe "
        "de Y′ − y_c à y_a − c = haut − Y′ − c : un ancrage haut en A devient, en B, un ancrage proche du bout libre, au bras court.",
    ]}

JEUX = {
    "titre": "Jeux et enveloppe",
    "paragraphes": [
        "La charge est calculée à X − dx, X et X + dx, l'ancrage à Y − dy, Y et Y + dy : neuf combinaisons (une seule sans jeu). "
        "Chaque grandeur vérifiée — chaque moment, aux sections et en tout point des diagrammes, la traction d'ancrage, la "
        "section nette, la largeur efficace — garde la plus défavorable des combinaisons, avec celle qui la gouverne "
        "(« charge à X + 20, ancrage à Y − 10 »). Les vues Moments montrent une courbe par position de la charge et l'ENVELOPPE "
        "en gras ; la tolérance d'ancrage se lit en bande.",
    ]}

LARGEUR_EFFICACE = {
    "titre": "Largeur efficace",
    "paragraphes": [
        "Quand la largeur d'appui a de la charge est plus courte que L, la section 1 ne reçoit que la bande diffusée à 45° "
        "depuis la charge jusqu'au pied du congé : L_eff = min(L ; a + 2·Xe) avec Xe = X′ − r − e, appliquée à toute l'aile "
        "horizontale (W et A), du côté sûr ; sans a, L_eff = L.",
    ]}

REFERENCES = [
    "Classeur de calcul du bureau, feuille P8 (cornière d'appui, cas de contrôle L 120×120×15 S355) : la méthode de référence, "
    "chaque grandeur du mode historique porte sa cellule",
    "Décisions de conception du module refondu du 10/11/2026 : configurations A et B, modes, jeux par l'enveloppe",
    "EN 10056-1 (cornières : dimensions, congé r₁, bouts r₂, masse) · EN 1993-1-1 Tableau 3.1 (limite élastique f_y)",
    "Efforts d'ancrage N et V par ancrage (ELS et ELU) pour un calcul des chevilles selon EN 1992-4 (Hilti PROFIS)",
]


def sections(mode: str, configuration: str, jeux: bool = True, largeur_efficace: bool = False) -> List[Dict]:
    """Les sections de méthode d'une cornière, dans l'ordre : le modèle (avec les contraintes admissibles), le mode
    (historique ou équilibre en A ; la configuration B impose l'équilibre), ce qui change au retournement, les jeux
    (quand il y en a), la largeur efficace (quand a est donnée). La note les répartit sur deux colonnes (``PARTAGE``)."""
    out = [MODELE]
    if configuration == "B":
        out += [CONFIGURATION_B, RETOURNEMENT]
    elif mode == HISTORIQUE:
        out += [HISTORIQUE_A, RETOURNEMENT]
    else:
        assert mode == EQUILIBRE
        out += [EQUILIBRE_A, RETOURNEMENT]
    if jeux:
        out.append(JEUX)
    if largeur_efficace:
        out.append(LARGEUR_EFFICACE)
    return out


# le nombre de sections de la première colonne de la note (le modèle et le mode) ; les suivantes (retournement, jeux,
# largeur efficace) vont dans la deuxième
PARTAGE = 2

# -*- coding: utf-8 -*-
"""Les grandeurs de l'EN 1992-1-1 que le module « Données béton » affiche À CÔTÉ des données du bureau. Chacune est
une FONCTION NOMMÉE (ou une constante nommée) avec sa source ; ``tests/test_donnees_beton_moteur.py`` confronte les
fonctions du béton au tableau 3.1 publié, classe par classe, de C12/15 à C90/105.

Béton — EN 1992-1-1:2004, tableau 3.1 (caractéristiques de résistance et de déformation du béton) ::

    f_cm        = f_ck + 8                                   (MPa)
    f_ctm       = 0,30·f_ck^(2/3)            jusqu'à C50/60  (MPa)
                = 2,12·ln(1 + f_cm/10)       au-delà de C50/60
    f_ctk,0,05  = 0,7·f_ctm                  fractile 5 %    (MPa)
    f_ctk,0,95  = 1,3·f_ctm                  fractile 95 %   (MPa)
    E_cm        = 22·(f_cm/10)^0,3           module sécant   (GPa)

Acier d'armature — EN 1992-1-1:2004 ::

    E_s   = 200 GPa                          §3.2.7(4) : valeur de calcul du module d'élasticité
    γ_s   = 1,15                             tableau 2.1N : ELU, situations durables et transitoires (valeur
                                             recommandée ; l'annexe nationale belge est à confirmer par le bureau)
    ε_yd  = f_yd / E_s                       allongement à la limite élastique de calcul (diagramme §3.2.7)

Le dépôt n'avait ni E_s ni le γ_s de l'EN 1992-1-1 (la méthode du bureau prend γ_s = 1,5 : ``beton/poutre/entree``) :
ils sont écrits ICI, une fois. La division f_yd = f_yk / γ_s n'est PAS refaite ici : c'est celle du moteur des poutres
(``beton/poutre/moteur.materiaux``), appelée par ``beton/donnees/consultation.py``.

Bibliothèque standard seule (math) : le module tourne dans le navigateur (Pyodide).
"""
import math

SOURCE_BETON = "EN 1992-1-1:2004, tableau 3.1"
SOURCE_E_S = "EN 1992-1-1:2004, §3.2.7(4)"
SOURCE_GAMMA_S = "EN 1992-1-1:2004, tableau 2.1N (ELU, situations durables et transitoires)"

FCK_LIMITE_FCTM = 50.0       # MPa : au-delà de C50/60, f_ctm suit la formule logarithmique (tableau 3.1)
E_S = 200.0                  # GPa : module d'élasticité de l'acier d'armature (§3.2.7(4))
GAMMA_S = 1.15               # coefficient partiel de l'acier, ELU durable et transitoire (tableau 2.1N)


def fcm(fck: float) -> float:
    """Résistance moyenne en compression sur cylindre (MPa) : f_cm = f_ck + 8 — tableau 3.1."""
    return fck + 8.0


def fctm(fck: float) -> float:
    """Résistance moyenne en traction directe (MPa) : 0,30·f_ck^(2/3) jusqu'à C50/60, 2,12·ln(1 + f_cm/10) au-delà —
    tableau 3.1."""
    if fck <= FCK_LIMITE_FCTM:
        return 0.30 * fck ** (2.0 / 3.0)
    return 2.12 * math.log(1.0 + fcm(fck) / 10.0)


def fctk_005(fck: float) -> float:
    """Résistance caractéristique en traction, fractile 5 % (MPa) : 0,7·f_ctm — tableau 3.1."""
    return 0.7 * fctm(fck)


def fctk_095(fck: float) -> float:
    """Résistance caractéristique en traction, fractile 95 % (MPa) : 1,3·f_ctm — tableau 3.1."""
    return 1.3 * fctm(fck)


def Ecm(fck: float) -> float:
    """Module d'élasticité sécant (GPa) : 22·(f_cm/10)^0,3 — tableau 3.1 (granulats de quartzite ; §3.1.3(2))."""
    return 22.0 * (fcm(fck) / 10.0) ** 0.3


def eps_yd(fyd: float) -> float:
    """Allongement de l'acier à la limite élastique de calcul (‰) : ε_yd = f_yd / E_s, f_yd en MPa, E_s en GPa."""
    return fyd / (E_S * 1000.0) * 1000.0


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



# ============================================================ déformations du tableau 3.1 et f_cd (bibliothèque béton armé)
# Les relations analytiques de la colonne de droite du tableau 3.1 (déformations en ‰) ; tests/test_bibliotheque_moteur.py
# les confronte aux valeurs imprimées, de C12/15 à C90/105.
SOURCE_FCD = "EN 1992-1-1:2004 §3.1.6(1)P, éq. (3.15) : f_cd = α_cc · f_ck / γ_c"
# les classes de résistance du tableau 3.1 : (f_ck, f_ck,cube) en MPa, de C12/15 à C90/105
CLASSES_3_1 = ((12, 15), (16, 20), (20, 25), (25, 30), (30, 37), (35, 45), (40, 50), (45, 55), (50, 60), (55, 67), (60, 75), (70, 85),
               (80, 95), (90, 105))
GAMMA_C = 1.5                # tableau 2.1N, ELU durable et transitoire
ALPHA_CC_RECOMMANDE = 1.0    # §3.1.6(1)P : valeur recommandée ; l'annexe belge (ANB) est à confirmer par le bureau (0,85 en usage)


def eps_c1(fck: float) -> float:
    """Déformation au pic de contrainte, diagramme non linéaire (‰) : 0,7·f_cm^0,31 ≤ 2,8 — tableau 3.1."""
    return min(2.8, 0.7 * fcm(fck) ** 0.31)


def eps_cu1(fck: float) -> float:
    """Déformation ultime, diagramme non linéaire (‰) : 3,5 jusqu'à C50/60, 2,8 + 27·((98 − f_cm)/100)^4 au-delà."""
    if fck <= FCK_LIMITE_FCTM:
        return 3.5
    return 2.8 + 27.0 * ((98.0 - fcm(fck)) / 100.0) ** 4


def eps_c2(fck: float) -> float:
    """Déformation au pic, diagramme parabole-rectangle (‰) : 2,0 jusqu'à C50/60, 2,0 + 0,085·(f_ck − 50)^0,53 au-delà."""
    if fck <= FCK_LIMITE_FCTM:
        return 2.0
    return 2.0 + 0.085 * (fck - 50.0) ** 0.53


def eps_cu2(fck: float) -> float:
    """Déformation ultime, parabole-rectangle (‰) : 3,5 jusqu'à C50/60, 2,6 + 35·((90 − f_ck)/100)^4 au-delà."""
    if fck <= FCK_LIMITE_FCTM:
        return 3.5
    return 2.6 + 35.0 * ((90.0 - fck) / 100.0) ** 4


def n_parabole(fck: float) -> float:
    """Exposant du diagramme parabole-rectangle : 2,0 jusqu'à C50/60, 1,4 + 23,4·((90 − f_ck)/100)^4 au-delà."""
    if fck <= FCK_LIMITE_FCTM:
        return 2.0
    return 1.4 + 23.4 * ((90.0 - fck) / 100.0) ** 4


def eps_c3(fck: float) -> float:
    """Déformation au pic, diagramme bilinéaire (‰) : 1,75 jusqu'à C50/60, 1,75 + 0,55·((f_ck − 50)/40) au-delà."""
    if fck <= FCK_LIMITE_FCTM:
        return 1.75
    return 1.75 + 0.55 * ((fck - 50.0) / 40.0)


def eps_cu3(fck: float) -> float:
    """Déformation ultime, diagramme bilinéaire (‰) : 3,5 jusqu'à C50/60, 2,6 + 35·((90 − f_ck)/100)^4 au-delà."""
    return eps_cu2(fck)


def fcd(fck: float, alpha_cc: float = ALPHA_CC_RECOMMANDE, gamma_c: float = GAMMA_C) -> float:
    """Résistance de calcul en compression (MPa) : f_cd = α_cc · f_ck / γ_c — §3.1.6(1)P."""
    return alpha_cc * fck / gamma_c

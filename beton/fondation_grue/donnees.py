# -*- coding: utf-8 -*-
"""Fondations de grues — les MATÉRIAUX des classeurs du bureau (feuille « Base de données », identique dans les
quatre classeurs), recalculés avec les mêmes formules et les mêmes arrondis. Bibliothèque standard seulement.

    R'wk                 donnée par classe de béton                                   (B3:F3)
    σ'b  = ARRONDI(R'wk / 2,7 ; 2)                                                    (B7:F7)
    μ    = ARRONDI(k·(1 − k/3)/2 ; 4),  k = 15·σ'b / (σs + 15·σ'b)                     (B11:F13)
           σs = 131 (BE220), 240 (BE400), 300 (BE500) — les constantes écrites dans les formules du classeur
    σadm = ARRONDI(fy / 1,5 ; 2)                                                      (B16:B18)

Ce sont les SEULS arrondis des classeurs : tout le reste se calcule sans arrondir.

La cinquième classe s'appelle « C45/50 » dans les classeurs ; avec R'wk = 47,5 c'est la C40/50 des autres modules du
site : elle porte ici ce nom, et « C45/50 » reste accepté à la relecture (``ALIAS_BETON``).
"""
import math
from decimal import ROUND_HALF_UP, Decimal

# R'wk (N/mm²) par classe — « Base de données »!B3:F3, dans l'ordre du classeur
RWK = {"C20/25": 23.5, "C25/30": 28.0, "C30/37": 35.0, "C35/45": 42.0, "C40/50": 47.5}
BETONS = tuple(RWK)
ALIAS_BETON = {"C45/50": "C40/50"}            # le libellé du classeur pour R'wk = 47,5
BETON_DEFAUT = "C30/37"

# σs écrit dans la formule de μ, et fy — « Base de données »!A11:A13, B16:B18
SIGMA_S_MU = {"BE220": 131.0, "BE400": 240.0, "BE500": 300.0}
FY = {"BE220": 220.0, "BE400": 400.0, "BE500": 500.0}
ACIERS = tuple(FY)
ACIER_DEFAUT = "BE500"

N_EQUIVALENCE = 15.0                           # coefficient d'équivalence acier / béton
DIVISEUR_BETON = 2.7                           # σ'b = R'wk / 2,7
DIVISEUR_ACIER = 1.5                           # σadm = fy / 1,5
TAUX_PERFOS = 150.0                            # N/mm² : taux de travail des perfos (effort horizontal)
POIDS_VOLUMIQUE = 25.0                         # kN/m³ : poids propre du béton armé
# les zones de cisaillement de la fiche BASE : τ < 0,016·R'wk (zone 1), < 0,032 (zone 2), < 0,048 (zone 3)
K_ZONES_TAU = (0.016, 0.032, 0.048)
# le seuil écrit EN DUR dans les variantes et dans le renfort au pieu de BASE (= 0,016 × 35, juste en C30/37)
TAU_SEUIL_EN_DUR = 0.56


def arrondi_excel(x: float, chiffres: int) -> float:
    """ARRONDI d'Excel : au plus proche, la moitié s'éloigne de zéro, sur l'écriture décimale à 15 chiffres
    significatifs (ce qu'Excel voit du nombre) — ni l'arrondi « au pair » de Python, ni son erreur binaire."""
    d = Decimal(format(float(x), ".15g"))
    return float(d.quantize(Decimal(1).scaleb(-chiffres), rounding=ROUND_HALF_UP))


def nom_beton(beton: str) -> str:
    """Le nom du site pour une classe (« C45/50 » du classeur → « C40/50 ») ; ValueError si inconnue."""
    b = ALIAS_BETON.get(beton, beton)
    if b not in RWK:
        raise ValueError(f"béton « {beton} » inconnu : {', '.join(BETONS)}")
    return b


def sigma_b(beton: str) -> float:
    """σ'b (N/mm²) = ARRONDI(R'wk / 2,7 ; 2)."""
    return arrondi_excel(RWK[nom_beton(beton)] / DIVISEUR_BETON, 2)


def mu(beton: str, acier: str) -> float:
    """μ = ARRONDI(15·σ'b/(σs + 15·σ'b)·(1 − 1/3·(15·σ'b/(σs + 15·σ'b)))/2 ; 4) — l'écriture du classeur."""
    if acier not in SIGMA_S_MU:
        raise ValueError(f"acier « {acier} » inconnu : {', '.join(ACIERS)}")
    sb, ss = sigma_b(beton), SIGMA_S_MU[acier]
    return arrondi_excel(15 * sb / (ss + (15 * sb)) * (1 - 1 / 3 * (15 * sb / (ss + (15 * sb)))) / 2, 4)


def sigma_adm(acier: str) -> float:
    """σ adm de l'acier (N/mm²) = ARRONDI(fy / 1,5 ; 2)."""
    if acier not in FY:
        raise ValueError(f"acier « {acier} » inconnu : {', '.join(ACIERS)}")
    return arrondi_excel(FY[acier] / DIVISEUR_ACIER, 2)


def materiaux(beton: str, acier: str) -> dict:
    """Les quatre grandeurs que les fiches lisent dans la base de données, plus les seuils de τ."""
    b = nom_beton(beton)
    rwk = RWK[b]
    return {"beton": b, "acier": acier, "Rwk": rwk, "sigma_b": sigma_b(b), "mu": mu(b, acier),
            "sigma_adm": sigma_adm(acier), "fy": FY[acier], "sigma_s_mu": SIGMA_S_MU[acier],
            "tau_zones": [k * rwk for k in K_ZONES_TAU]}


def aire(n: float, d: float) -> float:
    """Section de n barres Ø d (mm²) : π·(n·(Ø/2)²), l'écriture du classeur."""
    return math.pi * (n * (d / 2) ** 2)

# -*- coding: utf-8 -*-
"""Les CAS DE CHARGE d'une barre et leurs COMBINAISONS (docs/acier/BARRES_ACIER.md § 5.8).

Un cas : {"nom", "nature", …}. Natures : « permanent » (G) ; les charges d'exploitation par catégorie (A à H) ; la
neige, le vent, la température. Les coefficients ψ₀ viennent de la table des paramètres (acier/barre/parametres.py).

EUROCODE (NBN EN 1990, expression 6.10) — pour chaque ARRANGEMENT des cas variables (chacun présent ou absent : c'est
la recherche de la disposition défavorable, travée par travée) et chaque nature DOMINANTE parmi les présents :

    γ_G·ΣG + γ_Q·Q_dominante + Σ γ_Q·ψ₀·Q_autres        γ_G = 1,35 (défavorable) puis 1,00 (favorable), γ_Q = 1,5

Les cas d'une même nature sont une SEULE action disposée librement : ils reçoivent tous γ_Q quand leur nature domine.

MÉTHODE HISTORIQUE (charges de service) — pour chaque arrangement : G + ΣQ, tout à 100 % (la pratique du bureau,
à confirmer : Q4).
"""
from itertools import combinations
from typing import Dict, List, Sequence

from .resultat import Refus

PERMANENT = "permanent"
# nature → (libellé, clé du triplet ψ dans la table des paramètres)
NATURES = {
    PERMANENT: ("Permanente (G)", None),
    "exploitation_A": ("Exploitation A — habitation", "psi_A"), "exploitation_B": ("Exploitation B — bureaux", "psi_B"),
    "exploitation_C": ("Exploitation C — lieux de réunion", "psi_C"), "exploitation_D": ("Exploitation D — commerces", "psi_D"),
    "exploitation_E": ("Exploitation E — stockage", "psi_E"), "exploitation_F": ("Exploitation F — véhicules ≤ 30 kN", "psi_F"),
    "exploitation_G": ("Exploitation G — véhicules de 30 à 160 kN", "psi_G"), "exploitation_H": ("Exploitation H — toitures", "psi_H"),
    "neige": ("Neige", "psi_neige"), "vent": ("Vent", "psi_vent"), "temperature": ("Température", "psi_temperature"),
}
VARIABLES_MAX = 4               # 2⁴ arrangements, au plus quatre dominantes, deux γ_G : la liste reste lisible


def _fr(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def _nom(termes: Sequence) -> str:
    """« 1,35·G + 1,5·Q » ; un facteur 1 ne s'écrit pas."""
    return " + ".join(nom if abs(f - 1.0) < 1e-12 else f"{_fr(f)}·{nom}" for f, nom in termes if f != 0)


def _trier(cas: Sequence[Dict]):
    if not cas:
        raise Refus("la barre n'a aucun cas de charge")
    perm, var = [], []
    for i, c in enumerate(cas):
        if c.get("nature") not in NATURES:
            raise Refus(f"cas « {c.get('nom')} » : nature inconnue ({c.get('nature')!r})")
        (perm if c["nature"] == PERMANENT else var).append(i)
    if len(var) > VARIABLES_MAX:
        raise Refus(f"au plus {VARIABLES_MAX} cas de charge variables")
    return perm, var


def _arrangements(var: Sequence[int]):
    for k in range(len(var) + 1):
        for s in combinations(var, k):
            yield s


def elu(cas: Sequence[Dict], P: Dict) -> List[Dict]:
    """Les combinaisons fondamentales (6.10) : [{"nom", "facteurs": [γ de chaque cas], "dominante", "psi": {clés employées}}]."""
    perm, var = _trier(cas)
    out, vus = [], set()
    for gG in (P["gamma_G_sup"], P["gamma_G_inf"]):
        for s in _arrangements(var):
            natures = sorted({cas[i]["nature"] for i in s})
            if not s and gG != P["gamma_G_sup"]:
                continue                                   # G seul : la version défavorable suffit
            for dom in natures or [None]:
                f = [0.0] * len(cas)
                cles = set()
                for i in perm:
                    f[i] = gG
                for i in s:
                    if cas[i]["nature"] == dom:
                        f[i] = P["gamma_Q"]
                    else:
                        cle = NATURES[cas[i]["nature"]][1]
                        cles.add(cle)
                        f[i] = P["gamma_Q"] * P[cle][0]
                cle_f = tuple(round(x, 12) for x in f)
                if cle_f in vus or not any(f):
                    continue
                vus.add(cle_f)
                out.append({"nom": _nom([(f[i], cas[i]["nom"]) for i in range(len(cas))]), "facteurs": f, "dominante": dom,
                            "psi": sorted(cles)})
    return out


def service(cas: Sequence[Dict]) -> List[Dict]:
    """Les combinaisons de service de la méthode historique : G + ΣQ à 100 %, pour chaque arrangement des cas variables."""
    perm, var = _trier(cas)
    out = []
    for s in _arrangements(var):
        f = [0.0] * len(cas)
        for i in list(perm) + list(s):
            f[i] = 1.0
        if any(f):
            out.append({"nom": _nom([(f[i], cas[i]["nom"]) for i in range(len(cas))]), "facteurs": f, "dominante": None, "psi": []})
    return out

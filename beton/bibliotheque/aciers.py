# -*- coding: utf-8 -*-
"""Les ACIERS D'ARMATURE — EN 1992-1-1:2004 §3.2 et annexe C (tableau C.1) : nuances B400 et B500, classes de
ductilité A, B, C, et ce que le calcul en tire (f_yd pour γ_s de l'EC2 et du bureau, ε_yd). Les valeurs de l'annexe
C sont des EXIGENCES (des minimums) : le tableau les écrit comme telles (« ≥ »).

    k     = (f_t / f_y)_k : rapport caractéristique résistance en traction / limite d'élasticité
    ε_uk  : allongement caractéristique sous charge maximale (%)

Classe A : k ≥ 1,05, ε_uk ≥ 2,5 % ; classe B : k ≥ 1,08, ε_uk ≥ 5,0 % ; classe C : 1,15 ≤ k < 1,35, ε_uk ≥ 7,5 %.
f_yk de 400 à 600 MPa (§3.2.2(3)P) ; E_s = 200 GPa (§3.2.7(4)) ; masse volumique 7 850 kg/m³ (§3.2.7(3)).
La division f_yd = f_yk / γ_s est celle du moteur des poutres (``beton/poutre/moteur.materiaux``), par
``beton/donnees/consultation.aciers`` : elle n'est pas refaite ici.
"""
from beton.donnees import ec2 as EC

SOURCE = "EN 1992-1-1:2004 §3.2.2, §3.2.7 et annexe C, tableau C.1"
E_S = EC.E_S                            # 200 GPa
MASSE_VOLUMIQUE = 7850.0                # kg/m³, §3.2.7(3) (le tableau des armatures du bureau prend 7 860)
FYK_MIN, FYK_MAX = 400, 600             # §3.2.2(3)P : domaine d'application
NUANCES = (400, 500)                    # celles des modules béton (beton/poutre/entree.NUANCES)

# classe → (k minimal, k maximal (None : sans), ε_uk minimal en %)
DUCTILITE = {
    "A": {"k_min": 1.05, "k_max": None, "eps_uk_min": 2.5, "usage": "treillis et fils tréfilés à froid ; ductilité normale"},
    "B": {"k_min": 1.08, "k_max": None, "eps_uk_min": 5.0, "usage": "barres laminées à chaud ; ductilité élevée (le courant)"},
    "C": {"k_min": 1.15, "k_max": 1.35, "eps_uk_min": 7.5, "usage": "ductilité très élevée (zones sismiques, redistribution forte)"},
}
CLASSES = tuple(DUCTILITE.keys())


def nuances() -> list:
    """Les nuances × classes : [{nom, fyk, classe, k_min, k_max, eps_uk_min, usage}]."""
    out = []
    for fyk in NUANCES:
        for c, d in DUCTILITE.items():
            out.append({"nom": f"B{fyk}{c}", "fyk": fyk, "classe": c, **d})
    return out


def fiche(fyk: int, classe: str) -> dict:
    """La fiche d'une nuance : exigences de ductilité et grandeurs communes."""
    if fyk not in NUANCES:
        raise ValueError(f"nuance : {fyk} n'est pas une nuance des modules ({', '.join(str(n) for n in NUANCES)})")
    if classe not in DUCTILITE:
        raise ValueError(f"classe de ductilité : « {classe} » (A, B ou C)")
    d = DUCTILITE[classe]
    return {"nom": f"B{fyk}{classe}", "fyk": fyk, "classe": classe, "E_s": E_S, "masse_volumique": MASSE_VOLUMIQUE, **d,
            "source": SOURCE}

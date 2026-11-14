# -*- coding: utf-8 -*-
"""LA table des paramètres de l'Eurocode 3 et de ses annexes belges que les barres emploient : chaque valeur, sa
source et son STATUT (docs/acier/BARRES_ACIER.md § 5.8). Une seule table : changer d'annexe ou d'édition se fait ici.

Statuts :
- ``pointe``      : lue sur le texte (aucune pour l'instant : les textes ne sont pas au dépôt) ;
- ``a_pointer``   : la valeur de l'EN, connue, reprise par l'annexe belge — reste à lire sur le texte belge ;
- ``a_confirmer`` : la valeur change un résultat et sa source n'est pas le texte : la note l'imprime comme telle.
"""
from typing import Dict, List, Optional

A_POINTER, A_CONFIRMER, POINTE = "a_pointer", "a_confirmer", "pointe"
EDITION = "NBN EN 1993-1-1:2005 + AC:2009 + A1:2014, ANB:2018 ; NBN EN 1993-1-5:2006 ; NBN EN 1990 ANB"

PARAMETRES: Dict[str, Dict] = {
    # ---- coefficients partiels (EN 1993-1-1 § 6.1)
    "gamma_M0": {"valeur": 1.00, "nom": "γ_M0", "source": "EN 1993-1-1 § 6.1, valeur recommandée reprise par l'ANB", "statut": A_POINTER},
    "gamma_M1": {"valeur": 1.00, "nom": "γ_M1", "source": "EN 1993-1-1 § 6.1, valeur recommandée reprise par l'ANB", "statut": A_POINTER},
    "gamma_M2": {"valeur": 1.25, "nom": "γ_M2", "source": "EN 1993-1-1 § 6.1, valeur recommandée reprise par l'ANB", "statut": A_POINTER},
    # ---- matériau (EN 1993-1-1 § 3.2.6)
    "E": {"valeur": 210000.0, "nom": "E", "source": "EN 1993-1-1 § 3.2.6", "statut": A_POINTER},
    "G": {"valeur": 81000.0, "nom": "G", "source": "EN 1993-1-1 § 3.2.6", "statut": A_POINTER},
    # ---- déversement des profilés laminés (EN 1993-1-1 § 6.3.2.3) : deux couples selon la façon de calculer M_cr
    "lambda_LT0_sans_maintiens": {"valeur": 0.4, "nom": "λ̄_LT,0", "statut": A_CONFIRMER,
                                  "source": "ANB:2018, M_cr calculé en ignorant les maintiens latéraux — d'après l'aide d'un logiciel, pas le texte"},
    "beta_sans_maintiens": {"valeur": 0.75, "nom": "β", "statut": A_CONFIRMER,
                            "source": "ANB:2018, M_cr calculé en ignorant les maintiens latéraux — d'après l'aide d'un logiciel, pas le texte"},
    "lambda_LT0_avec_maintiens": {"valeur": 0.2, "nom": "λ̄_LT,0", "statut": A_CONFIRMER,
                                  "source": "ANB:2018, M_cr calculé avec les maintiens (section brute) — d'après l'aide d'un logiciel, pas le texte"},
    "beta_avec_maintiens": {"valeur": 1.0, "nom": "β", "statut": A_CONFIRMER,
                            "source": "ANB:2018, M_cr calculé avec les maintiens (section brute) — d'après l'aide d'un logiciel, pas le texte"},
    # ---- flexion composée avec instabilité (EN 1993-1-1 § 6.3.3)
    "methode_interaction": {"valeur": "annexe_A", "nom": "méthode du § 6.3.3", "statut": A_CONFIRMER,
                            "source": "ANB:2018 : méthode 1 (annexe A) — d'après l'aide d'un logiciel, pas le texte"},
    "borne_Cmi0": {"valeur": "1 - NEd/Ncr", "nom": "borne inférieure de C_mi,0", "statut": A_CONFIRMER,
                   "source": "ANB:2018 : une borne existe (tableau A.2) ; sa valeur est supposée, comme dans l'annexe française"},
    # ---- âme (EN 1993-1-5 § 5.1)
    "eta": {"valeur": 1.20, "nom": "η", "source": "EN 1993-1-5 § 5.1(2), valeur recommandée (aciers jusqu'au S460)", "statut": A_POINTER},
    # ---- combinaisons (NBN EN 1990 ANB, expression 6.10)
    "gamma_G_sup": {"valeur": 1.35, "nom": "γ_G,sup", "source": "NBN EN 1990 ANB, tableau A1.2(B)", "statut": A_POINTER},
    "gamma_G_inf": {"valeur": 1.00, "nom": "γ_G,inf", "source": "NBN EN 1990 ANB, tableau A1.2(B)", "statut": A_POINTER},
    "gamma_Q": {"valeur": 1.50, "nom": "γ_Q", "source": "NBN EN 1990 ANB, tableau A1.2(B)", "statut": A_POINTER},
    # ---- coefficients (ψ₀ ; ψ₁ ; ψ₂) des actions variables (NBN EN 1990 ANB, tableau A1.1)
    "psi_A": {"valeur": (0.7, 0.5, 0.3), "nom": "ψ, catégorie A", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_B": {"valeur": (0.7, 0.5, 0.3), "nom": "ψ, catégorie B", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_C": {"valeur": (0.7, 0.7, 0.6), "nom": "ψ, catégorie C", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_D": {"valeur": (0.7, 0.7, 0.6), "nom": "ψ, catégorie D", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_E": {"valeur": (1.0, 0.9, 0.8), "nom": "ψ, catégorie E", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_F": {"valeur": (0.7, 0.7, 0.6), "nom": "ψ, catégorie F", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_G": {"valeur": (0.7, 0.5, 0.3), "nom": "ψ, catégorie G", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_H": {"valeur": (0.0, 0.0, 0.0), "nom": "ψ, catégorie H", "source": "NBN EN 1990 ANB, tableau A1.1", "statut": A_POINTER},
    "psi_neige": {"valeur": (0.5, 0.2, 0.0), "nom": "ψ, neige", "statut": A_CONFIRMER,
                  "source": "valeurs de l'EN 1990 (altitude ≤ 1 000 m) — l'annexe belge est à lire"},
    "psi_vent": {"valeur": (0.6, 0.2, 0.0), "nom": "ψ, vent", "statut": A_CONFIRMER, "source": "valeurs de l'EN 1990 — l'annexe belge est à lire"},
    "psi_temperature": {"valeur": (0.6, 0.5, 0.0), "nom": "ψ, température", "statut": A_CONFIRMER,
                        "source": "valeurs de l'EN 1990 — l'annexe belge est à lire"},
}


def valeur(cle: str):
    """La valeur d'un paramètre ; KeyError s'il n'est pas dans la table."""
    return PARAMETRES[cle]["valeur"]


def jeu(surcharges: Optional[Dict] = None) -> Dict:
    """{clé: valeur} de toute la table, ``surcharges`` appliquées : un cas d'essai publié avec d'autres coefficients,
    une autre annexe. KeyError pour une clé qui n'est pas dans la table (une faute de frappe ne passe pas en silence)."""
    j = {k: p["valeur"] for k, p in PARAMETRES.items()}
    for k, v in (surcharges or {}).items():
        if k not in PARAMETRES:
            raise KeyError(k)
        j[k] = v
    return j


def a_confirmer(cles=None) -> List[Dict]:
    """Les paramètres « à confirmer » (parmi ``cles`` si donné) : [{cle, nom, valeur, source}] — ce que la note imprime
    dans ses hypothèses."""
    return [{"cle": k, "nom": p["nom"], "valeur": p["valeur"], "source": p["source"]} for k, p in PARAMETRES.items()
            if p["statut"] == A_CONFIRMER and (cles is None or k in cles)]

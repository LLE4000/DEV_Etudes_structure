# -*- coding: utf-8 -*-
"""LA table des paramètres de l'Eurocode 3 et de ses annexes belges que les barres emploient : chaque valeur, sa
source et son STATUT (docs/acier/BARRES_ACIER.md § 5.8). Une seule table : changer d'annexe ou d'édition se fait ici.

Statuts :
- ``pointe``      : lue sur le texte (le texte n'est pas au dépôt : la source dit la clause) ;
- ``a_pointer``   : la valeur de l'EN, connue, reprise par l'annexe belge — reste à lire sur le texte belge ;
- ``a_confirmer`` : la valeur change un résultat et sa source n'est pas le texte : la note l'imprime comme telle ;
- ``decision``    : une DÉCISION BUREAU (brief, règle 8) — le défaut est ici, l'autre variante est calculée et affichée.
"""
from typing import Dict, List, Optional

A_POINTER, A_CONFIRMER, POINTE, DECISION = "a_pointer", "a_confirmer", "pointe", "decision"
EDITION = "NBN EN 1993-1-1:2005 + AC:2009 + A1:2014, ANB:2018 ; NBN EN 1993-1-5:2006 ; NBN EN 1990 ANB"

PARAMETRES: Dict[str, Dict] = {
    # ---- coefficients partiels (EN 1993-1-1 § 6.1)
    "gamma_M0": {"valeur": 1.00, "nom": "γ_M0", "source": "EN 1993-1-1 § 6.1, valeur recommandée reprise par l'ANB", "statut": A_POINTER},
    "gamma_M1": {"valeur": 1.00, "nom": "γ_M1", "source": "EN 1993-1-1 § 6.1, valeur recommandée reprise par l'ANB", "statut": A_POINTER},
    "gamma_M2": {"valeur": 1.25, "nom": "γ_M2", "source": "EN 1993-1-1 § 6.1, valeur recommandée reprise par l'ANB", "statut": A_POINTER},
    # ---- matériau (EN 1993-1-1 § 3.2.6)
    "E": {"valeur": 210000.0, "nom": "E", "source": "EN 1993-1-1 § 3.2.6", "statut": A_POINTER},
    "G": {"valeur": 81000.0, "nom": "G", "source": "EN 1993-1-1 § 3.2.6", "statut": A_POINTER},
    # ---- déversement des profilés laminés (EN 1993-1-1 § 6.3.2.3) : le couple par défaut. NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) :
    # (0,2 ; 1,0) dès que M_cr est calculé sur la section brute avec les combinaisons, la distribution réelle des moments
    # et les maintiens — c'est le solveur du module (acier/barre/mcr.py), maintiens présents ou non. La variante
    # (0,4 ; 0,75) suppose les maintiens TOTALEMENT ignorés dans M_cr : clés « …_variante » ci-dessous (brief § 7.7).
    # Les clés gardent leur nom (« M_cr calculé avec les maintiens ») : une surcharge écrite avec elles reste lisible.
    "lambda_LT0_avec_maintiens": {"valeur": 0.2, "nom": "λ̄_LT,0", "statut": POINTE,
                                  "source": "NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : M_cr avec les combinaisons, la distribution réelle des moments et les maintiens"},
    "beta_avec_maintiens": {"valeur": 1.0, "nom": "β", "statut": POINTE,
                            "source": "NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : M_cr avec les combinaisons, la distribution réelle des moments et les maintiens"},
    # la VARIANTE du même 6.3.2.3(1) : poutre de BÂTIMENT avec maintiens, TOUS ignorés dans M_cr — appliquée seulement
    # sur demande et dans ce cas (ec3_deversement.couple_anb ; brief § 7.7, test 9 du § 11.5)
    "lambda_LT0_variante": {"valeur": 0.4, "nom": "λ̄_LT,0 (variante)", "statut": POINTE,
                            "source": "NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : poutre de bâtiment, maintiens totalement ignorés dans M_cr"},
    "beta_variante": {"valeur": 0.75, "nom": "β (variante)", "statut": POINTE,
                      "source": "NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : poutre de bâtiment, maintiens totalement ignorés dans M_cr"},
    # la méthode des I et H laminés : DÉCISION BUREAU n° 4 (l'ANB:2018 fixe les deux sans en imposer une) — § 6.3.2.3
    # par défaut ; le § 6.3.2.2 est calculé à côté et affiché pour information (brief § 7.7)
    "methode_deversement_lamines": {"valeur": "6.3.2.3", "nom": "méthode de déversement des I et H laminés", "statut": DECISION,
                                    "source": "NBN EN 1993-1-1 ANB:2018, 6.3.2.2(2) et 6.3.2.3 : DÉCISION BUREAU n° 4, § 6.3.2.3 par défaut"},
    # ---- flexion composée avec instabilité (EN 1993-1-1 § 6.3.3)
    "methode_interaction": {"valeur": "annexe_D_ANB", "nom": "méthode du § 6.3.3", "statut": POINTE,
                            "source": "NBN EN 1993-1-1 ANB:2018, 6.3.3(5) : méthode 1 normative, écrite par l'annexe D ANB (prise comme un tout)"},
    # n_pl : l'annexe D ANB écrit γ_M1, l'annexe A corrigée (AC:2009, modification 35 a) γ_M0 — même résultat tant que
    # γ_M0 = γ_M1 ; sinon point ouvert (brief § 7.8) : la note le dit, avec la variante (ec3_interaction)
    "npl_gamma": {"valeur": "gamma_M1", "nom": "γ_M de n_pl quand γ_M0 ≠ γ_M1", "statut": A_CONFIRMER,
                  "source": "annexe D ANB : γ_M1 (appliqué) ; annexe A corrigée par l'AC:2009 : γ_M0 (affiché) — équivalence non établie"},
    # ---- âme (EN 1993-1-5 § 5.1)
    # η : la note complémentaire de 5.1(2) de la NBN EN 1993-1-5 ANB:2011 (renvoi à 6.2.6(3) de l'EN 1993-1-1) réserve
    # η > 1,00 aux sections SOUDÉES chargées dans le plan de l'âme (brief § 7.5 et § 8) : 1,00 pour les I et H laminés
    # et les tubes ; 1,20 pour les I soudés et les caissons, jusqu'au S460 compris. Texte reçu au lot 4 : à pointer.
    "eta": {"valeur": 1.00, "nom": "η", "statut": A_POINTER,
            "source": "NBN EN 1993-1-5 ANB:2011, 5.1(2), note complémentaire : I et H laminés, tubes"},
    "eta_soude": {"valeur": 1.20, "nom": "η (section soudée)", "statut": A_POINTER,
                  "source": "NBN EN 1993-1-5 ANB:2011, 5.1(2) : sections soudées (I, caissons) jusqu'au S460"},
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


# les clés RETIRÉES de la table, et pourquoi : une surcharge qui les nomme est refusée avec ce motif (jamais ignorée)
RETIREES: Dict[str, str] = {
    "lambda_LT0_sans_maintiens": "retiré le 16/11/2026 — NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : le couple (0,4 ; 0,75) se lit dans "
                                 "« lambda_LT0_variante », « beta_variante » ; il ne s'applique que sur demande (deversement.couple), pour une "
                                 "poutre de bâtiment dont tous les maintiens sont ignorés dans M_cr",
    "beta_sans_maintiens": "retiré le 16/11/2026 — NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) : le couple (0,4 ; 0,75) se lit dans "
                           "« lambda_LT0_variante », « beta_variante » ; il ne s'applique que sur demande (deversement.couple), pour une "
                           "poutre de bâtiment dont tous les maintiens sont ignorés dans M_cr",
    "borne_Cmi0": "retiré le 16/11/2026 (lot B) — NBN EN 1993-1-1 ANB:2018, annexe D, tableau D.1 ANB : C_mi,0 n'a pas de borne "
                  "inférieure (ψ = −1 et N_Ed/N_cr = 0,1 donnent 0,532)",
}


class CleRetiree(KeyError):
    """Une surcharge qui nomme une clé retirée de la table : ``args[0]`` est la clé, ``motif`` dit pourquoi."""

    def __init__(self, cle: str):
        super().__init__(cle)
        self.motif = RETIREES[cle]


def jeu(surcharges: Optional[Dict] = None) -> Dict:
    """{clé: valeur} de toute la table, ``surcharges`` appliquées : un cas d'essai publié avec d'autres coefficients,
    une autre annexe. KeyError pour une clé qui n'est pas dans la table (une faute de frappe ne passe pas en silence) ;
    CleRetiree (un KeyError) pour une clé retirée, avec son motif."""
    j = {k: p["valeur"] for k, p in PARAMETRES.items()}
    for k, v in (surcharges or {}).items():
        if k in RETIREES:
            raise CleRetiree(k)
        if k not in PARAMETRES:
            raise KeyError(k)
        j[k] = v
    return j


def a_confirmer(cles=None) -> List[Dict]:
    """Les paramètres « à confirmer » (parmi ``cles`` si donné) : [{cle, nom, valeur, source}] — ce que la note imprime
    dans ses hypothèses."""
    return [{"cle": k, "nom": p["nom"], "valeur": p["valeur"], "source": p["source"]} for k, p in PARAMETRES.items()
            if p["statut"] == A_CONFIRMER and (cles is None or k in cles)]

# -*- coding: utf-8 -*-
"""Paramètres du moteur « Charges sur plancher » : les décisions du bureau
(P-1 à P-17 de l'étude) et les valeurs normatives non encore transcrites.

Chaque champ porte son défaut CONSERVATIF et la décision qu'il représente.
Un paramètre ``None`` marqué « à transcrire » désactive la vérification
qui en dépend, et le résultat le dit (statut « non vérifié ») : jamais une
valeur reconstituée de mémoire.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Parametres:
    # P-2 : coefficient uniforme sur G + Q pour les résistances de calcul (hourdis, effets locaux)
    gamma_f: float = 1.5
    # P-3 : part du tranchant qui reste sur l'élément chargé (1,0 = aucune répartition en tranchant)
    part_tranchant_element: float = 1.0
    # P-4 : coefficient k de la largeur efficace en flexion (Heft 240) — None = b_m = t_y
    k_flexion_heft240: Optional[float] = None
    # P-15 : joint longitudinal des hourdis — résistance admissible (kN par m de joint) et
    # longueur de joint mobilisée de part et d'autre de la charge (m) — None = non vérifié
    joint_adm_kN_m: Optional[float] = None
    joint_longueur_m: Optional[float] = None
    # P-9 : effets locaux EN 1992-1-1 — γc, αcc (ANB belge à confirmer), β du poinçonnement
    gamma_c: float = 1.5
    alpha_cc: float = 1.0
    beta_poinconnement: float = 1.0
    # v_Rd,max au nu de l'aire chargée (EN 1992-1-1 §6.4.5(3)) : coefficient de 0,4·ν·fcd,
    # valeur recommandée du corrigendum, à confirmer sur l'ANB — None = contrôle u0 non fait
    coef_vrd_max: Optional[float] = 0.4
    # Seuils d'affichage : orange au-delà, rouge au-delà
    seuil_attention: float = 0.85
    seuil_defaut: float = 1.0
    # Discrétisation des charges linéaires et surfaciques (m)
    pas_lineaire: float = 0.25
    pas_surfacique: float = 0.30
    # Balayage de position (m)
    pas_balayage: float = 0.10


DEFAUT = Parametres()

# -*- coding: utf-8 -*-
"""Les textes de méthode, écrits une fois : l'onglet Méthode de l'écran et la
page 2 de la note les lisent ici."""
from typing import Dict, List

from .criteres import G_KG, GAMMA_F_EQUIV, GAMMA_M2, K_BUREAU

SECTIONS: List[Dict] = [
    {"titre": "Le cordon et sa gorge",
     "paragraphes": [
         "Un cordon d'angle est un segment de gorge a posé dans le dièdre de deux faces perpendiculaires. Il reçoit une "
         "force par unité de longueur f (N/mm) que le calcul décompose sur sa gorge selon la figure 4.5 de l'EN 1993-1-8 : "
         "σ⊥ normale au plan de gorge, τ⊥ dans la gorge en travers de l'axe, τ∥ dans la gorge le long de l'axe. Le plan de "
         "gorge passe par la racine et coupe l'angle droit en deux ; la géométrie du cordon (ses deux ailes) décide des "
         "signes, pas une règle de pouce : une force en travers dans le plan de joint et un arrachement donnent toutes deux "
         "|σ⊥| = |τ⊥| = f/(a·√2), mais ne se combinent pas de la même façon.",
         "Longueur efficace l_eff = l − 2a sans retours d'angle (§4.5.1) ; un cordon ne compte que si l_eff ≥ max(30 mm ; 6a) "
         "et a ≥ 3 mm (§4.5.2). Bout à bout à pénétration partielle : cordon de gorge égale à la pénétration (§4.7.2), gorge "
         "perpendiculaire à l'effort normal.",
     ]},
    {"titre": "Le groupe de cordons",
     "paragraphes": [
         "Les cordons sont traités comme des lignes d'aire a·dl (méthode élastique classique) : aire A_w, centre G, inerties "
         "I_x, I_y, I_xy et inertie polaire J des gorges. Les six composantes d'effort données au point de référence sont "
         "transportées en G ; la force par unité de longueur en un point d'un cordon de gorge a vaut "
         "f = a·[F/A_w + σ_flexion·e_z + (M_z/J)·e_z × GP], avec σ_flexion linéaire (axes quelconques, I_xy compris) et la "
         "torsion proportionnelle au rayon. f varie linéairement le long d'un cordon : elle est évaluée aux deux extrémités "
         "efficaces et au milieu ; le point le plus chargé gouverne. Des gorges différentes prennent des forces proportionnelles "
         "à leur gorge, à contrainte égale.",
     ]},
    {"titre": "Eurocode : EN 1993-1-8 §4.5.3",
     "paragraphes": [
         f"Méthode directionnelle (§4.5.3.2) : √(σ⊥² + 3·(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) et σ⊥ ≤ 0,9·fu/γM2, avec fu et βw de "
         "l'acier de référence du cordon — la pièce la plus faible (Tab. 4.1 : βw = 0,8 pour S235, 0,85 pour S275, 0,9 pour "
         f"S355, 1,0 pour S420 et S460) — et γM2 = {GAMMA_M2:g} (ANB belge). Les efforts sont des efforts de calcul (ELU).",
         "Méthode simplifiée (§4.5.3.3) : |f| ≤ F_w,Rd = f_vw,d·a avec f_vw,d = fu/(√3·βw·γM2), quelle que soit la direction de "
         "f. Toujours du côté sûr par rapport à la directionnelle ; proposée en option.",
         "Pleine pénétration (§4.7.1) : la soudure vaut la pièce la plus faible ; l'outil vérifie la pièce élastiquement "
         "(√(σ² + 3τ²) ≤ fy/γM0). Assemblage long par recouvrement (§4.11) : βLw,1 = 1,2 − 0,2·L_j/(150a) ≤ 1 quand L_j > 150a.",
     ]},
    {"titre": "Méthode ELS du bureau",
     "paragraphes": [
         f"Efforts de service, sans coefficient de charge. La résultante par unité de longueur est comparée à k·a avec "
         f"k = {K_BUREAU:g} kg par mm de gorge et par cm de cordon par défaut, soit k·g/10 = {K_BUREAU * G_KG / 10:.1f} N/mm² "
         f"sur la gorge (g = {G_KG:g} N/kg ; 10 N/kg en option). La page affiche à côté ce que l'Eurocode donne au même niveau "
         f"de service, f_vw,d/γ_F avec γ_F = {GAMMA_F_EQUIV:g} : 138,6 N/mm² pour le S235 (141 kg/(mm·cm) à 9,81 N/kg, 139 à "
         "10 N/kg), 155,8 pour le S275, 174,5 pour le S355 — et un bouton pour prendre cette valeur. Sur du S235, la règle du "
         "bureau est plus sûre de 28 % ; sur du S355, de 43 %.",
         "Les deux méthodes ne se comparent qu'à efforts égaux : la carte résultat donne le taux selon l'autre méthode avec "
         "les mêmes efforts, à titre indicatif ; le bureau décide de la nature des efforts qu'il saisit.",
     ]},
    {"titre": "Ce que l'outil ne fait pas (V1)",
     "paragraphes": [
         "Pas de fatigue, pas d'aciers au-delà du S460 ni d'inoxydable, pas de cordons discontinus (les saisir comme cordons "
         "libres), pas de répartition plastique entre cordons, pas de vérification des pièces assemblées hors du bout à bout "
         "à pleine pénétration (la pièce est à vérifier dans son propre module). L'excentricité des plats d'un recouvrement "
         "simple n'est pas prise en compte (pratique usuelle).",
     ]},
]

REFERENCES = [
    "EN 1993-1-8:2005 §4.3 (cordons), §4.5.1 (longueur efficace), §4.5.2 (gorge), §4.5.3.2 (méthode directionnelle), "
    "§4.5.3.3 (méthode simplifiée), §4.7 (soudures bout à bout), §4.11 (assemblages longs), Tab. 4.1 (βw)",
    "NBN EN 1993-1-8 ANB : γM2 = 1,25",
    "Pratique du bureau : k = 100 kg par mm de gorge et par cm de cordon (efforts de service)",
    "Méthode élastique des groupes de cordons (répartition linéaire en flexion, polaire en torsion)",
]

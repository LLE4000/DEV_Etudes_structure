# -*- coding: utf-8 -*-
"""Les textes de méthode du garde-corps, écrits une fois : l'onglet Méthode de l'écran et la
page 2 de la note les lisent ici."""
from typing import Dict, List

from .charges import H_MAX_APPLICATION, QR_DEFAUT, QR_PONCT_DEFAUT, QV_DEFAUT
from .membres import FLECHE_X_DEFAUT, GAMMA_Q, SIGMA_ADM_DEFAUT, TAU_ADM_DEFAUT

SECTIONS: List[Dict] = [
    {"titre": "Le modèle",
     "paragraphes": [
         "Le garde-corps est une ligne en plan — droite, en L ou en U — portée par des montants verticaux encastrés en pied sur "
         "une platine, ou par une lisse horizontale ancrée dans un mur par des consoles. Les entraxes sont libres : réguliers à "
         "partir d'une longueur et d'un entraxe maximal, ou saisis un à un (un poteau central, un à gauche, un à droite, à des "
         "distances différentes). Chaque montant reçoit la charge de sa longueur tributaire, la demi-somme des deux entraxes "
         "voisins ; un montant d'angle reçoit une demi-travée de chaque tronçon, dans deux directions perpendiculaires, et une "
         "section rectangulaire y est vérifiée dans les deux orientations. Une extrémité ancrée dans un mur n'a pas de montant : "
         "le mur reprend la demi-travée, et la note dit l'effort à justifier avec la cheville.",
         "Le montant est une console de hauteur H (la main courante au-dessus de la fixation) chargée en tête : M = P·H, V = P, "
         "N = q_v·s_trib, flèche P·H³/(3EI). La lisse haute est une poutre sur deux appuis entre montants, de portée l'entraxe le "
         "plus grand, fléchie dans deux plans (q_h horizontal, q_v vertical, Q ponctuelle en option) : M = qL²/8 + QL/4, "
         "flèche 5qL⁴/(384EI) + QL³/(48EI). Un barreau est une poutre sur deux appuis entre la lisse basse et la lisse haute ; une "
         "lisse intermédiaire, une poutre entre montants ; chacun sous la pression du remplissage q_r × entraxe ou sous la charge "
         "ponctuelle Q_r à mi-portée, deux cas séparés dont le pire gouverne.",
     ]},
    {"titre": "Les charges",
     "paragraphes": [
         f"La charge linéaire horizontale q_h s'applique sur la lisse haute, à la hauteur de la main courante (au plus "
         f"{H_MAX_APPLICATION / 1000:.2f} m) : EN 1991-1-1 §6.4, Tableau 6.12, par catégorie d'usage — 0,5 kN/m en habitation (A), "
         "bureaux (B) et lieux de réunion avec tables (C1), 1,0 kN/m pour les catégories C2 à C4 et les commerces (D), 3,0 kN/m "
         "pour les foules (C5), 2,0 kN/m en stockage (E). Ce sont les valeurs RECOMMANDÉES de l'EN ; l'annexe nationale belge "
         "(NBN EN 1991-1-1 ANB) peut en fixer d'autres : la valeur retenue est toujours modifiable (valeur libre), et la note "
         "imprime la valeur et son origine. On peut aussi saisir directement la charge ponctuelle P qui revient à chaque montant.",
         f"S'y ajoutent une charge verticale de service q_v sur la main courante ({QV_DEFAUT:g} kN/m par défaut, usage du bureau), "
         f"une charge ponctuelle horizontale Q sur la lisse en option, et, pour le remplissage, une pression q_r "
         f"({QR_DEFAUT:g} kN/m² par défaut, usage du bureau) et une charge ponctuelle Q_r ({QR_PONCT_DEFAUT:g} kN : Q_kh,2 de la "
         "NBN B 03-004). Les charges horizontale et verticale agissent ensemble sur les montants et la lisse (côté sûr).",
     ]},
    {"titre": "Méthode ELS du bureau",
     "paragraphes": [
         f"Les « vieilles formules » : efforts de service, sans coefficient, et contraintes admissibles — σ = M/W_el + N/A ≤ σ_adm "
         f"({SIGMA_ADM_DEFAUT:g} N/mm² par défaut pour le S235), τ = V/A_v ≤ τ_adm ({TAU_ADM_DEFAUT:g} N/mm²), flèche ≤ L/x "
         f"(x = {FLECHE_X_DEFAUT:g} par défaut : le montant courant, CHS 48,3 × 3,2 à 1,20 m sous 0,5 kN/m, tient H/100 ; l'ancien "
         "module proposait L/200) ou une valeur en millimètres. L'aire de cisaillement A_v est celle du catalogue (EN 1993-1-1 "
         "§6.2.6 : 2A/π pour un tube rond, A·h/(b + h) pour un tube rectangulaire), l'aire entière pour un plat ou un rond.",
         "La soudure du montant sur la platine est vérifiée par le module Soudures avec la méthode ELS du bureau (résultante par "
         "unité de longueur ≤ k·a, k = 100 kg par mm de gorge et par cm de cordon) ; les boulons et les chevilles avec leurs "
         "résistances de calcul divisées par γ_F = 1,5 (l'équivalent de service, comme pour les cordons).",
     ]},
    {"titre": "Eurocode",
     "paragraphes": [
         f"Les mêmes efforts multipliés par γ_Q = {GAMMA_Q:g} (charges d'exploitation, EN 1990), les résistances élastiques de "
         "l'EN 1993-1-1 §6.2 : σ_Ed ≤ f_y/γ_M0 avec W_el (du côté sûr pour les classes 1 et 2, qui pourraient prendre W_pl), "
         "τ_Ed ≤ f_y/(√3·γ_M0), et la flèche sous les efforts de service. La soudure suit la méthode directionnelle de "
         "l'EN 1993-1-8 §4.5.3.2 (γ_M2 = 1,25), les boulons le Tableau 3.4 (F_t,Rd = 0,9·f_ub·A_s/γ_M2, F_v,Rd = α_v·f_ub·A_s/γ_M2, "
         "interaction F_v/F_v,Rd + F_t/(1,4·F_t,Rd) ≤ 1), les chevilles leurs valeurs de calcul de l'ETA avec l'interaction "
         "(β_N)^1,5 + (β_V)^1,5 ≤ 1 (EN 1992-4 §7.2.3, du côté sûr pour tous les modes de ruine).",
     ]},
    {"titre": "La fixation",
     "paragraphes": [
         "Platine rigide, deux rangées de n boulons à ±e_x/2 de l'axe du montant dans le sens de l'effort. Le moment est repris "
         "par la traction Z de la rangée tendue et l'appui au nu comprimé du montant : z = e_x/2 + h_p/2, Z = (M − N·h_p/2)/z "
         "(la compression N soulage) ; sous une traction N_t (console au mur), l'enveloppe des deux modèles — appui au nu, ou "
         "couple de boulons M/e_x + N_t/2 — gouverne. Traction par boulon F_t = Z/n, cisaillement F_v = V/(2n).",
         "La platine fléchit en console du nu du montant à la rangée tendue : bras m = e_x/2 − h_p/2 − 0,8·a·√2 "
         "(EN 1993-1-8 §6.2.6.5), largeur efficace b_eff = min(l_p ; Σ min(2πm ; 4m + 1,25e)) (Tableau 6.6, motifs circulaire et "
         "non circulaire), M_p = Z·m, σ_p = 6·M_p/(b_eff·t_p²) ≤ σ_lim, d'où l'épaisseur requise. La soudure du montant sur la "
         "platine est un cas du module Soudures : tube sur platine (C6) pour un tube du catalogue, console fléchie (C3) pour un "
         "plat, cordons libres sur le contour (C8) pour une section libre — mêmes dessins, mêmes formules.",
     ]},
    {"titre": "Ce que l'outil ne fait pas (V1)",
     "paragraphes": [
         "Le béton sous la platine et les modes de ruine du béton d'ancrage (cône, fendage, bord) ne sont pas vérifiés : les "
         "chevilles sont jugées sur les valeurs de calcul de leur ETA (module Pied de poteau pour le cas général). Pas de lisse "
         "basse, pas de panneau (verre, tôle : à justifier par le fournisseur), pas de choc ni de fatigue, pas de flambement du "
         "montant (N négligeable), pas d'aluminium ni d'inoxydable, pas de garde-corps de véhicules (EN 1991-1-1 annexe B).",
     ]},
]

REFERENCES = [
    "EN 1991-1-1 §6.4 et Tableau 6.12 (charges horizontales sur les garde-corps, valeurs recommandées) ; NBN EN 1991-1-1 ANB",
    "NBN B 03-004 : garde-corps de bâtiments (charge ponctuelle Q_kh,2 sur le remplissage)",
    "EN 1993-1-1 §6.2 (résistance élastique des sections), §6.2.6 (aires de cisaillement) ; EN 1993-1-8 Tab. 3.4 (boulons), §4.5.3 (cordons), §6.2.6.5 (bras m)",
    "EN 1992-4 §7.2.3 (interaction des ancrages) ; valeurs de calcul des chevilles : ETA du fabricant",
    "Pratique du bureau : σ_adm = 160 N/mm², τ_adm = 90 N/mm² sur efforts de service ; k = 100 kg/(mm·cm) pour les cordons",
]

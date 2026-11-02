# -*- coding: utf-8 -*-
"""Textes de MÉTHODE du module « Charges sur plancher » — source unique de
l'onglet Méthode de l'écran et de la page 2 de la note (R2). Chaque
section dit ce que le moteur fait, avec la décision du bureau qu'elle
applique (P-n de l'étude) et la référence normative. Aucun nombre n'y
est pris de mémoire : les valeurs citées sont celles du code (défauts de
``parametres.Parametres``) ou des formules transcrites.
"""
from typing import Dict, List

from .parametres import DEFAUT

SECTIONS: List[Dict] = [
    {"titre": "Charges et combinaisons",
     "paragraphes": [
         "Les charges sont saisies en valeurs de service : charge par appui d'un engin, relue sur le tableau de "
         "stabilité du constructeur pour l'opération prévue (P-13), avec son coefficient dynamique φ par appui "
         "(EN 1991-1-1 §6.3.2.3 pour les roues) ; charge d'un étançon ; charge linéaire d'une file d'étais ; "
         "charge surfacique d'un stock ; charges déjà présentes sur le plancher (EN 1991-1-6 §4.11).",
         "Les efforts sollicitants comptent les poids propres. G = poids propres de l'élément et de la couche de "
         "compression ; Q = charges présentes et charges concentrées multipliées par φ.",
         f"Quand la résistance est une valeur de CALCUL (M_Rd, V_Rd d'un catalogue, EN 1992-1-1 pour les effets "
         f"locaux, EN 1993 pour une filière ou un plateau) : M_Ed = γG·M_G + γQ·M_Q ≤ M_Rd, avec γG = {DEFAUT.gamma_G:g} "
         f"et γQ = {DEFAUT.gamma_Q:g} (P-2, tranché par le bureau le 02/11/2026). Quand elle est une valeur "
         "ADMISSIBLE de service (catalogue en service, méthode aux contraintes admissibles de l'application "
         "pour les dalles) : coefficients 1 sur G et Q.",
     ]},
    {"titre": "Répartiteurs : plateaux et filières",
     "paragraphes": [
         "Plateau : la charge se diffuse sur l'aire efficace (a + 2c)·(b + 2c), avec la bande plastique "
         "c = t·√(f/(3·p_Ed)) de chaque côté de la semelle (EN 1993-1-8 §6.2.5), p_Ed = γQ·P / aire, itérée "
         "jusqu'à cohérence et plafonnée par le bord du plateau ; une plaque déclarée rigide par son fournisseur "
         "compte entière, sa pression de service est comparée à la valeur publiée.",
         "Filière en travers des éléments : poutre d'Euler libre sur un ressort par élément croisé, de raideur "
         "k = 3·E·I·L/(a²·(L − a)²) à l'abscisse de la filière (flèche d'un élément sur deux appuis sous une charge "
         "ponctuelle) ; les réactions des ressorts sont les charges des éléments ; la filière est vérifiée en "
         "flexion à l'ELU, γQ·M_max ≤ W_el·fy/γM0 ; un ressort qui tire (élément soulevé) est signalé, le "
         "modèle linéaire n'étant plus exact. Filière dans le sens porteur : charge linéaire uniforme sous "
         "hypothèse rigide, sans répartition entre éléments.",
     ]},
    {"titre": "Hourdis",
     "paragraphes": [
         "Chaque élément reçoit la part de l'empreinte qui le recouvre (règle du levier). La flexion se partage "
         "avec les voisins selon la figure C.1 de l'EN 1168 seulement si le bureau a transcrit les parts et si "
         "les joints sont clavetés ou le plancher chapé ; sinon l'élément chargé porte tout.",
         f"Effort tranchant : {DEFAUT.part_tranchant_element:.0%} de la réaction d'appui de chaque charge "
         "concentrée reste sur l'élément chargé, sans partage par les joints (P-3). L'annexe C répartit la "
         "flexion à mi-portée, où les joints ont toute la portée pour transmettre ; près de l'appui, le joint "
         "n'a que quelques dizaines de centimètres et l'annexe ne dit rien de l'effort tranchant : garder tout "
         "sur l'élément chargé est prudent.",
         "M est cherché aux abscisses des charges, à mi-portée et sur une grille au pas L/60 — son maximum "
         "tombe entre la charge et la mi-portée ; V aux deux appuis. Les résistances sont celles "
         "du catalogue, dans leur nature (calcul ou service, P-1) ; les valeurs « avec couche de compression » "
         "ne sont admises que si la chape est collaborante et a atteint la classe exigée par le fabricant "
         "(P-17), sinon la chape ne compte que comme poids. La pression sur la membrure supérieure, diffusée à "
         "45° dans la chape, est comparée à la pression locale admissible du fabricant ; sans cette donnée, un "
         "répartiteur efficace est exigé (P-5). L'effort transmis par un joint longitudinal est vérifié quand "
         "sa résistance est transcrite de l'EN 1168 (P-15). Le moment de fissuration, s'il est renseigné, donne "
         "une ligne d'information en service (P-11).",
     ]},
    {"titre": "Dalle coulée en place et prédalle",
     "paragraphes": [
         "Largeurs efficaces du Heft 240 : t_y = b + 2·h1 + h ≤ 0,8·l ; en tranchant b_v = t_y + 0,5·a ; en "
         "flexion b_m = t_y tant que le coefficient k n'est pas transcrit (P-4, très conservatif sous charge "
         "concentrée), b_m = t_y + k·x·(1 − x/l) ensuite ; sans armature transversale ≥ 20 % de la principale "
         "(EN 1992-1-1 §9.3.1.1), pas de largeur au-delà de t_y. Les charges voisines se superposent en "
         "créneaux et le maximum est pris sur la largeur.",
         "Résistances par mètre : les formules de l'application (module Poutre) inversées, "
         "M_adm = min(αb·μ·b·d² ; As·fyd·0,9·d) et V_adm = τadm,I·0,75·b·h, sur la classe de béton saisie ou sur "
         "la classe arrondie vers le bas que fck(t) autorise ; flexion et tranchant par mètre en charges de "
         "service, coefficients 1. Effets locaux à l'ELU avec γQ sur la charge : pression localisée "
         "EN 1992-1-1 §6.7 (6.63), poinçonnement §6.4 sur le contour u1 à 2d avec v_Rd,c (6.47) et v_min.",
         "Prédalle : largeur efficace plafonnée à la prédalle quand les joints ne sont pas armés ou qu'une charge "
         "est à cheval sur un joint (P-16) ; étais en place = portée réduite avec avertissement, le report sur "
         "l'étage inférieur n'étant pas modélisé (P-8) ; avant durcissement de la chape, seules les charges "
         "surfaciques de bétonnage sont admises (P-7).",
     ]},
    {"titre": "Ce que l'outil ne fait pas (V1)",
     "paragraphes": [
         "Deux appuis seulement, sans continuité ni porte-à-faux ; filières à 0° ou 90° ; éléments supposés "
         "indépendants sous une filière (conservatif pour le plus chargé) ; pas de chaîne d'étaiement entre "
         "étages ; poinçonnement au sens de l'EN 1992-1-1 réservé aux dalles et prédalles — pour un hourdis, "
         "la membrure se juge sur la donnée du fabricant. Un résultat « non vérifié » n'est jamais masqué : la "
         "ligne dit quel paramètre manque.",
     ]},
]

A_TRANSCRIRE = [
    "le coefficient k de la largeur efficace en flexion (Heft 240) ;",
    "les parts de la figure C.1 de l'EN 1168 pour chaque type de hourdis ;",
    "la résistance du joint longitudinal (EN 1168) ;",
    "αcc et le coefficient de v_Rd,max de l'ANB belge ;",
    "les valeurs de catalogue des hourdis (M_Rd, V_Rd, nature, I, pression locale admissible, M_cr, valeurs "
    "avec chape et classe exigée), à relire sur la fiche du fabricant.",
]

REFERENCES = [
    "EN 1168 (annexe C, joints)", "Heft 240 du DAfStb (largeurs efficaces)", "EN 1993-1-8 §6.2.5 (bande plastique)",
    "EN 1992-1-1 §6.4, §6.7, §9.3.1.1", "EN 1991-1-6 §4.11", "EN 1991-1-1 §6.3.2.3",
    "formules BA de l'application (module Poutre)",
]


def phrase_a_transcrire() -> str:
    return "Reste à transcrire avant un usage en production : " + " ".join(A_TRANSCRIRE)

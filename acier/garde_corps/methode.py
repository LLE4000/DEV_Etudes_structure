# -*- coding: utf-8 -*-
"""Les textes de méthode du garde-corps, courts, écrits une fois : l'onglet Détail de l'écran
et la page 2 de la note les lisent ici. Les hypothèses chiffrées (charges, limites, sections)
sont imprimées par la note elle-même, en tête : ici, seulement les règles."""
from typing import Dict, List

from .membres import COEF_CONTINUE, FLECHE_X_DEFAUT, GAMMA_ADM, GAMMA_Q


def _fr(x: float) -> str:
    """Un nombre à une décimale, virgule française (156,7)."""
    return f"{x:.1f}".replace(".", ",")


def _fr3(x: float) -> str:
    """Un coefficient à trois décimales, virgule française (0,125)."""
    return f"{x:.3f}".replace(".", ",")


SECTIONS: List[Dict] = [
    {"titre": "Modèle",
     "paragraphes": [
         "Montant principal : console encastrée en pied, chargée en tête par P = q_h × s_trib (s_trib : demi-somme des entraxes "
         "voisins ; un montant d'angle reçoit une demi-travée de chaque tronçon dans deux directions, section rectangulaire "
         "vérifiée dans les deux orientations). M = P·H, V = P, f = P·H³/(3EI).",
         "Main courante : poutre de portée l'entraxe le plus grand, sur deux appuis (M = qL²/8, f = 5qL⁴/(384EI)) ou continue "
         f"(coefficients des tables : M = {_fr3(COEF_CONTINUE[2][0])} / {_fr3(COEF_CONTINUE[3][0])} / {_fr3(COEF_CONTINUE[4][0])} qL² pour 2 / 3 / ≥ 4 travées). "
         "La charge horizontale fléchit l'axe faible (h vertical), la charge verticale l'axe fort.",
         "Barreaux : poutre sur deux appuis entre la lisse basse et la main courante (portée déduite de H, h_bas et des sections) ; "
         "lisses intermédiaires : poutre entre montants, posée h vertical comme la main courante (la charge horizontale fléchit l'axe faible). Charge ponctuelle Q_r à mi-portée (cas 3) ou pression q_r × entraxe (cas 4).",
         "Main courante murale : la main courante est l'élément principal, poutre sur les consoles (portée l'entraxe le plus grand) ; chaque "
         "console est une console de longueur e scellée dans le mur, sous R_v = q_v × s_trib (flexion, cas 2) et R_h = q_h × s_trib (traction "
         "axiale qui tire la console hors du mur, cas 1) ; sa platine et ses chevilles sont vérifiées sous l'enveloppe des deux cas.",
         "Sans montant (les deux bouts ancrés dans un mur et une seule travée) : la main courante porte de mur à mur sur toute la longueur, "
         "ses réactions à chaque mur (H du q_h, V du q_v) sont données à justifier avec l'ancrage.",
         "Cas de charge : les quatre charges agissent séparément (cas 1 à 4) ; l'option « cumul » fait agir q_h et q_v ensemble.",
     ]},
    {"titre": "Vérification à l'ELS",
     "paragraphes": [
         f"Charges de service sans coefficient ; contraintes admissibles dérivées de la nuance : σ_adm = f_y/{GAMMA_ADM:g} "
         f"({_fr(235 / GAMMA_ADM)} N/mm² pour le S235), τ_adm = σ_adm/√3 ({_fr(235 / GAMMA_ADM / 3 ** 0.5)} N/mm²). σ = M/W_el + N/A ≤ σ_adm, "
         f"τ = V/A_v ≤ τ_adm, contrainte équivalente σ_eq = √(σ² + 3τ²) ≤ σ_adm, flèche ≤ L/{FLECHE_X_DEFAUT:g} (ou une valeur en mm). "
         "A_v du catalogue (EN 1993-1-1 §6.2.6), aire entière pour un plat ou un rond. Soudure : vérification à l'ELS (k = 100 kg/(mm·cm)) ; "
         "chevilles et boulons : résistance de calcul / γ_F = 1,5.".replace("f_y/1.5", "f_y/1,5"),
     ]},
    {"titre": "Eurocode (ELU)",
     "paragraphes": [
         f"Charges × γ_Q = {GAMMA_Q:g} ; σ_Ed ≤ f_y/γ_M0 (élastique, W_el), τ_Ed ≤ f_y/(√3·γ_M0), flèche sous charges de service. "
         "Soudure : EN 1993-1-8 §4.5.3.2 (directionnelle, γ_M2 = 1,25) ; boulons : Tab. 3.4 ; chevilles : valeurs de calcul de l'ETA, "
         "interaction (β_N)^1,5 + (β_V)^1,5 ≤ 1 (EN 1992-4 §7.2.3).",
     ]},
    {"titre": "Ancrage",
     "paragraphes": [
         "Platine rigide, deux rangées de n boulons à ±e_x/2 : z = e_x/2 + h_p/2, Z = (M − N·h_p/2)/z (sous traction N_t : "
         "enveloppe avec M/e_x + N_t/2), F_t = Z/n, F_v = V/(2n). Platine en console du nu du montant à la rangée tendue : "
         "m = e_x/2 − h_p/2 − 0,8·a·√2, b_eff = min(l_p ; Σ min(2πm ; 4m + 1,25e)), σ_p = 6·Z·m/(b_eff·t_p²), épaisseur requise. "
         "Soudure du montant sur la platine : cas du module Soudures (C6 tube, C3 plat, C8 section libre).",
     ]},
    {"titre": "Ancrage latéral",
     "paragraphes": [
         "Platine verticale en applique sur le nez de dalle, le montant soudé le long d'elle par deux cordons longitudinaux (module "
         "Soudures, C8) ; le montant est une console de H + d_h, le moment au centre de la platine vaut M = P·(H + d_h + b_p/2), P arrache "
         "la platine, N_v = q_v·s_trib la cisaille. Deux rangées horizontales de n ancrages à ±e_x/2, la haute tendue : z = e_x, "
         "Z = M/z + P/2, F_t = Z/n, F_v = V/(2n) (un montant d'angle ajoute la torsion M_z·r/Σr²). Platine en console du nu du montant à "
         "la colonne d'ancrages : m = e_y/2 − b/2 − 0,8·a·√2, b_eff = min(b_p ; Σ min(2πm ; 4m + 1,25e)), σ_p = 6·Z·m/(b_eff·t_p²), "
         "épaisseur requise.",
     ]},
    {"titre": "Hors périmètre",
     "paragraphes": [
         "Béton sous la platine et ruines du béton d'ancrage (module Pied de poteau), lisse basse (non calculée), panneaux (fournisseur), "
         "chocs, fatigue, flambement du montant, aluminium et inoxydable, garde-corps de véhicules (EN 1991-1-1 annexe B).",
     ]},
]


def sections(mode: str, fixation: str = "platine", sans_montant: bool = False) -> List[Dict]:
    """Les sections de méthode d'un cas : la vérification choisie (ELS ou Eurocode), et l'ancrage qui
    correspond à la fixation — « Ancrage » (platine sur la dalle, montant soudé sur un plat, consoles au
    mur), « Ancrage latéral » (platine en nez de dalle), aucun sans montant ou pour un montant scellé."""
    els = mode == "els"
    out = []
    for s in SECTIONS:
        t = s["titre"]
        if (t.startswith("Eurocode") and els) or (t.startswith("Vérification à l'ELS") and not els):
            continue
        if t == "Ancrage" and (sans_montant or fixation not in ("platine", "soudee")):
            continue
        if t == "Ancrage latéral" and (sans_montant or fixation != "laterale"):
            continue
        out.append(s)
    return out

REFERENCES = [
    "EN 1991-1-1 §6.4, Tab. 6.12 ; NBN EN 1991-1-1 ANB",
    "NBN B 03-004 : garde-corps de bâtiments (Q_kh,1, Q_kh,2, Q_kv)",
    "EN 1993-1-1 §6.2, §6.2.6 ; EN 1993-1-8 Tab. 3.4, §4.5.3, §6.2.6.5, Tab. 6.6 ; EN 1992-4 §7.2.3",
    "Pratique du bureau : σ_adm = f_y/1,5, τ_adm = σ_adm/√3, σ_eq ≤ σ_adm en service ; k = 100 kg/(mm·cm) ; tables des poutres continues",
]

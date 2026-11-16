# -*- coding: utf-8 -*-
"""Le RAIDISSEUR TRANSVERSAL d'une âme : une paire de plats symétriques, soudés de part et d'autre de l'âme entre les
deux semelles, sous une charge concentrée ou sur un appui (EN 1993-1-5 § 9 ; docs/acier/BARRES_ACIER.md § 5.9 ;
brief § 8.4).

    matériau     f_y,p de chaque plat selon SON épaisseur (table unique acier/materiaux.py : en S235, 20 mm → 225 N/mm²) ;
                 ε_p = √(235/f_y,p) ; ε_w de l'âme selon t_w ; f_y,d = min(f_y,p ; f_y,w) : la f_y de calcul de la
                 section du raidisseur — plats et bande d'âme (brief § 7.1, § 8.4, test 18 du § 11.5)
    géométrie    chanfrein : le congé r du profilé, arrondi à 5 mm vers le haut — le coin du plat dégage le congé ;
                 largeur b_s : (b − t_w)/2 au plus (le plat ne dépasse pas la semelle), arrondie à 5 mm vers le bas, et
                 réduite tant que le plat est trop élancé pour son épaisseur :
                     b_s/t_s ≤ 14·ε_p                  plat en console de classe 3 (EN 1993-1-1 tableau 5.2)
                     I_T/I_p ≥ 5,3·f_y,p/E             torsion du plat (§ 9.2.1(8)) : I_T = b_s·t_s³/3,
                                                       I_p = b_s³·t_s/3 + b_s·t_s³/12 (autour du bord soudé)
                 épaisseur t_s : la première de 8, 10, 12, 15, 20 mm qui porte la force.
    écrasement   au contact de la semelle chargée, plats moins chanfreins (§ 9.4(2), extrémité découpée) :
                     A_c = 2·(b_s − chanfrein)·t_s                        F_Ed ≤ N_c,Rd = A_c·f_y,p/γ_M0
    flambement   hors du plan de l'âme (§ 9.4(2)) — la section en croix : les deux plats et 15·ε_w·t_w d'âme de chaque
                 côté (§ 9.1(2)), dans la limite de l'âme disponible ; longueur 0,75·h_w ; courbe c :
                     A_st = 2·b_s·t_s + l_w·t_w            I_st = [t_s·(2·b_s + t_w)³ + (l_w − t_s)·t_w³]/12
                     N_cr = π²·E·I_st/(0,75·h_w)²          λ̄ = √(A_st·f_y,d/N_cr)        F_Ed ≤ N_b,Rd = χ·A_st·f_y,d/γ_M1
    rigidité     quand le raidisseur tient lieu d'appui rigide d'une âme élancée au cisaillement (§ 9.3.3(3), panneau
                 long) :                                                  I_st ≥ 0,75·h_w·t_w³

Le raidisseur porte TOUTE la force (l'âme seule n'y suffit pas : c'est pour cela qu'on le pose). Il est de la nuance
de la barre ; chaque plat a la f_y de SON épaisseur (sans ``nuance`` : la f_y donnée, pour tous). Ses soudures relèvent
de l'EN 1993-1-8 : hors du module, et la note le dit.

Unités internes : N, mm. Blocs de formules en kN, mm, mm², cm⁴ (gabarits « r_… » de acier/barre/formules.py).
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

from acier.flambement import normes as NO

from . import section as SE
from .resultat import NOK, Journal, fr as _fr, terme as _t

REF = "EN 1993-1-5"
EPAISSEURS = (8.0, 10.0, 12.0, 15.0, 20.0)        # mm : les plats proposés (brief § 8.4)
PAS = 5.0                                          # mm : l'arrondi d'une largeur de plat, d'un chanfrein
AME_PARTICIPANTE = 15.0                            # 15·ε·t_w d'âme de chaque côté (§ 9.1(2))
LONGUEUR = 0.75                                    # longueur de flambement : 0,75·h_w, les deux semelles tenues (§ 9.4(2))
COURBE = "c"                                       # § 9.4(2)
PALIER = 0.2                                       # λ̄ ≤ 0,2 : pas de réduction (EN 1993-1-1 § 6.3.1.2(4))
ELANCEMENT_PLAT = 14.0                             # b_s/t_s ≤ 14·ε
TORSION = 5.3                                      # I_T/I_p ≥ 5,3·f_y/E (9.3)
RIGIDITE = 0.75                                    # I_st ≥ 0,75·h_w·t³ quand a/h_w ≥ √2 (9.6)


def gabarit(s: Dict) -> Optional[Dict]:
    """La place d'un raidisseur dans un I ou un H : {"hw", "bs_max", "chanfrein"} (mm) ; None pour une autre forme, ou
    un profilé dont la demi-semelle ne laisse aucun contact une fois le chanfrein ôté."""
    hw = SE.hauteur_ame(s)
    if hw is None or not s.get("b") or not s.get("tw"):
        return None
    conge = s.get("r") or s["tw"]                              # (un IPN : son congé r₁ vaut t_w — le catalogue ne le range pas sous « r »)
    chanfrein = PAS * math.ceil(conge / PAS - 1e-9)
    bs_max = PAS * math.floor((s["b"] - s["tw"]) / 2.0 / PAS + 1e-9)
    if bs_max - chanfrein < PAS:
        return None
    return {"hw": hw, "bs_max": bs_max, "chanfrein": chanfrein}


def elancement_admis(fy: float, E: float) -> float:
    """Le plus grand b_s/t_s d'un plat : 14·ε, et la torsion — t_s²/(b_s² + t_s²/4) ≥ 5,3·f_y/E."""
    return min(ELANCEMENT_PLAT * NO.epsilon(fy), math.sqrt(E / (TORSION * fy) - 0.25))


def materiau(fy: float, ts: float, nuance: Optional[str] = None, fy_ame: Optional[float] = None) -> Dict:
    """Le matériau d'une paire de plats d'épaisseur ``ts`` (brief § 7.1, § 8.4) : {"fy_plat" (f_y du plat selon SON
    épaisseur — table unique, produit laminé), "fy_ame" (f_y de l'âme selon t_w), "fy_calcul" (le plus petit des deux :
    la section du raidisseur), "eps_plat", "eps_ame"}. Sans ``nuance`` : ``fy`` pour tout (une f_y donnée)."""
    fyp = fy if nuance is None else NO.limite_elastique(nuance, ts, produit="lamine")
    fyw = fy if fy_ame is None else float(fy_ame)
    return {"fy_plat": fyp, "fy_ame": fyw, "fy_calcul": min(fyp, fyw), "eps_plat": NO.epsilon(fyp), "eps_ame": NO.epsilon(fyw)}


def plat(s: Dict, fy: float, P: Dict, ts: float, dispo: Sequence[Optional[float]] = (None, None), nuance: Optional[str] = None,
         fy_ame: Optional[float] = None) -> Optional[Dict]:
    """La paire de plats d'épaisseur ``ts`` (mm) dans le profilé ``s``. ``dispo`` : la longueur d'âme disponible de
    chaque côté de l'axe du raidisseur (mm ; None : sans limite) — le bout de la barre, la moitié de la distance au
    raidisseur voisin. ``nuance`` : celle de la barre — chaque plat prend la f_y de son épaisseur (``materiau``) ;
    ``fy_ame`` : la f_y de l'âme. Rend {"ts", "bs", "chanfrein", "hw", "fy_plat", "fy_ame", "fy_calcul", "eps_plat",
    "eps_ame", "Ac", "NcRd", "cotes" (l'âme prise de chaque côté), "limite" (un côté est borné), "lw", "Ast", "Ist",
    "Ncr", "lambda", "Phi", "chi", "NbRd", "elancement", "elancement_lim", "torsion", "torsion_lim", "I_min"} — N et mm ;
    None si le profilé ne reçoit pas de raidisseur."""
    g = gabarit(s)
    if g is None:
        return None
    E, tw, hw = P["E"], s["tw"], g["hw"]
    m = materiau(fy, ts, nuance, fy_ame)
    fyp, fyd = m["fy_plat"], m["fy_calcul"]
    bs = min(g["bs_max"], PAS * math.floor(ts * elancement_admis(fyp, E) / PAS + 1e-9))
    if bs - g["chanfrein"] < PAS:
        return None
    Ac = 2.0 * (bs - g["chanfrein"]) * ts
    plein = AME_PARTICIPANTE * m["eps_ame"] * tw
    cotes = [plein if d is None else min(plein, max(float(d) - ts / 2.0, 0.0)) for d in dispo]
    lw = ts + cotes[0] + cotes[1]
    Ast = 2.0 * bs * ts + lw * tw
    Ist = (ts * (2.0 * bs + tw) ** 3 + (lw - ts) * tw ** 3) / 12.0
    Ncr = math.pi ** 2 * E * Ist / (LONGUEUR * hw) ** 2
    lam = math.sqrt(Ast * fyd / Ncr)
    Phi = chi = None
    if lam > PALIER:
        Phi = 0.5 * (1.0 + NO.ALPHA[COURBE] * (lam - PALIER) + lam ** 2)
        chi = min(1.0 / (Phi + math.sqrt(Phi ** 2 - lam ** 2)), 1.0)
    return dict(m, ts=ts, bs=bs, chanfrein=g["chanfrein"], hw=hw, Ac=Ac, NcRd=Ac * fyp / P["gamma_M0"], cotes=cotes,
                limite=any(c < plein - 1e-9 for c in cotes), lw=lw, Ast=Ast, Ist=Ist, Ncr=Ncr, Phi=Phi,
                chi=1.0 if chi is None else chi, NbRd=(1.0 if chi is None else chi) * Ast * fyd / P["gamma_M1"],
                elancement=bs / ts, elancement_lim=ELANCEMENT_PLAT * m["eps_plat"],
                torsion=(bs * ts ** 3 / 3.0) / (bs ** 3 * ts / 3.0 + bs * ts ** 3 / 12.0), torsion_lim=TORSION * fyp / E,
                I_min=RIGIDITE * hw * tw ** 3, **{"lambda": lam})


def dimensionner(s: Dict, fy: float, P: Dict, F: float, dispo: Sequence[Optional[float]] = (None, None),
                 appui_rigide: bool = False, nuance: Optional[str] = None, fy_ame: Optional[float] = None) -> Optional[Dict]:
    """La paire de plats qui porte ``F`` (N) : la première épaisseur de EPAISSEURS dont l'écrasement, le flambement —
    et, pour un ``appui_rigide`` d'âme élancée, la rigidité — passent ; à défaut la plus épaisse, avec "ok" faux. Rend
    le plat (``plat``) complété de {"F", "ok", "taux" (le plus fort de F/N_c,Rd et F/N_b,Rd), "rigide" (I_st ≥ I_min,
    None si ce n'est pas demandé)} ; None si le profilé ne reçoit pas de raidisseur."""
    dernier = None
    for ts in EPAISSEURS:
        p = plat(s, fy, P, ts, dispo, nuance, fy_ame)
        if p is None:
            continue
        rigide = (p["Ist"] >= p["I_min"]) if appui_rigide else None
        p.update({"F": F, "taux": max(F / p["NcRd"], F / p["NbRd"]), "rigide": rigide})
        p["ok"] = p["taux"] <= 1.0 and rigide is not False
        dernier = p
        if p["ok"]:
            return p
    return dernier


def designation(p: Dict) -> str:
    """« 2 plats 70 × 8 mm » : ce qu'on commande."""
    return f"2 plats {_fr(p['bs'])} × {_fr(p['ts'])} mm"


def ecrire(J: Journal, base: str, nom: str, s: Dict, fy: float, P: Dict, p: Dict, avant: Sequence[str] = (),
           pourquoi: Sequence[str] = ()) -> Tuple[Dict, Dict]:
    """Le raidisseur ``p`` (``dimensionner``) dans le journal : « {base}-R » (écrasement au contact) et « {base}-Rb »
    (flambement, et les conditions de forme du plat). ``avant`` : les clés des formules qui justifient le raidisseur
    (la résistance de l'âme seule) ; ``pourquoi`` : les phrases qui le disent. Rend les deux vérifications."""
    gM0, gM1, E, tw = P["gamma_M0"], P["gamma_M1"], P["E"], s["tw"]
    fyp, fyd = p.get("fy_plat", fy), p.get("fy_calcul", fy)
    ep, ew = p.get("eps_plat", NO.epsilon(fy)), p.get("eps_ame", NO.epsilon(fy))
    F = p["F"]
    cles = list(avant)
    cles.append(J.f("A_c", "A_c", [_t("b_s", p["bs"], "mm"), _t("chanfrein", p["chanfrein"], "mm"), _t("t_s", p["ts"], "mm")], p["Ac"], "mm²", "r_A_c"))
    cles.append(J.f("N_c_Rd", "N_c,Rd", [_t("A_c", p["Ac"], "mm²"), _t("f_y,p", fyp, "N/mm²"), _t("γ_M0", gM0)], p["NcRd"] / 1e3, "kN", "e_N_Rd"))
    mots = list(pourquoi) + [f"Raidisseur : {designation(p)}, chanfrein {_fr(p['chanfrein'])} mm, sur toute la hauteur de l'âme ; "
                             f"f_y,p = {_fr(fyp)} N/mm² (plats de {_fr(p['ts'])} mm), f_y,d = min(f_y,p ; f_y,w) = {_fr(fyd)} N/mm²."]
    v1 = J.verif(f"{base}-R", f"{nom} : raidisseur, écrasement au contact", f"{REF} § 9.4", "F_Ed ≤ N_c,Rd", F / 1e3, p["NcRd"] / 1e3, "kN", cles,
                 messages=mots)
    cles2: List[str] = []
    mots2: List[str] = []
    if p["limite"]:
        cles2.append(J.f("l_w", "l_w", [_t("t_s", p["ts"], "mm"), _t("l_g", p["cotes"][0], "mm"), _t("l_d", p["cotes"][1], "mm")], p["lw"], "mm",
                         "r_l_w_borne"))
        mots2.append(f"Âme participante bornée par l'âme disponible (15·ε_w·t_w = {_fr(AME_PARTICIPANTE * ew * tw, 1)} mm de chaque côté au plus).")
    else:
        cles2.append(J.f("l_w", "l_w", [_t("t_s", p["ts"], "mm"), _t("ε_w", ew), _t("t_w", tw, "mm")], p["lw"], "mm", "r_l_w"))
    cles2.append(J.f("A_st", "A_st", [_t("b_s", p["bs"], "mm"), _t("t_s", p["ts"], "mm"), _t("l_w", p["lw"], "mm"), _t("t_w", tw, "mm")], p["Ast"],
                     "mm²", "r_A_st"))
    cles2.append(J.f("I_st", "I_st", [_t("t_s", p["ts"], "mm"), _t("b_s", p["bs"], "mm"), _t("t_w", tw, "mm"), _t("l_w", p["lw"], "mm")],
                     p["Ist"] / 1e4, "cm⁴", "r_I_st"))
    cles2.append(J.f("N_cr_st", "N_cr", [_t("E", E, "N/mm²"), _t("I_st", p["Ist"] / 1e4, "cm⁴"), _t("h_w", p["hw"], "mm")], p["Ncr"] / 1e3, "kN",
                     "r_N_cr"))
    cles2.append(J.f("lambda_st", "λ̄", [_t("A_st", p["Ast"], "mm²"), _t("f_y,d", fyd, "N/mm²"), _t("N_cr", p["Ncr"] / 1e3, "kN")], p["lambda"], "",
                     "e_lambda_bar"))
    if p["Phi"] is not None:
        cles2.append(J.f("Phi_st", "Φ", [_t("α", NO.ALPHA[COURBE]), _t("λ̄", p["lambda"])], p["Phi"], "", "e_Phi"))
        if p["chi"] < 1.0:
            cles2.append(J.f("chi_st", "χ", [_t("Φ", p["Phi"]), _t("λ̄", p["lambda"])], p["chi"], "", "e_chi"))
    else:
        mots2.append("λ̄ ≤ 0,2 : χ = 1.")
    cles2.append(J.f("N_b_Rd_st", "N_b,Rd", [_t("χ", p["chi"]), _t("A_st", p["Ast"], "mm²"), _t("f_y,d", fyd, "N/mm²"), _t("γ_M1", gM1)],
                     p["NbRd"] / 1e3, "kN", "e_N_b_Rd"))
    cles2.append(J.f("bs_ts", "b_s/t_s", [_t("b_s", p["bs"], "mm"), _t("t_s", p["ts"], "mm")], p["elancement"], "", "r_elancement"))
    cles2.append(J.f("bs_ts_lim", "14·ε_p", [_t("k", ELANCEMENT_PLAT), _t("ε_p", ep)], p["elancement_lim"], "", "e_lim"))
    cles2.append(J.f("torsion", "I_T/I_p", [_t("b_s", p["bs"], "mm"), _t("t_s", p["ts"], "mm")], p["torsion"], "", "r_torsion"))
    cles2.append(J.f("torsion_lim", "5,3·f_y,p/E", [_t("f_y,p", fyp, "N/mm²"), _t("E", E, "N/mm²")], p["torsion_lim"], "", "r_torsion_lim"))
    mots2.append(f"Longueur de flambement 0,75·h_w, courbe {COURBE} ; plat : b_s/t_s ≤ 14·ε_p et I_T/I_p ≥ 5,3·f_y,p/E.")
    statut = None
    if p["rigide"] is not None:
        cles2.append(J.f("I_min", "I_st,min", [_t("h_w", p["hw"], "mm"), _t("t_w", tw, "mm")], p["I_min"] / 1e4, "cm⁴", "r_I_min"))
        if p["rigide"]:
            mots2.append("Appui rigide de l'âme élancée : I_st ≥ I_st,min.")
        else:
            statut = NOK
            mots2.append(J.alerte("raidisseur_souple", "attention", f"{nom} : le raidisseur n'a pas la rigidité d'un appui de l'âme "
                                                                    "(I_st < I_st,min)."))
    v2 = J.verif(f"{base}-Rb", f"{nom} : raidisseur, flambement", f"{REF} § 9.4", "F_Ed ≤ N_b,Rd", F / 1e3, p["NbRd"] / 1e3, "kN", cles2,
                 statut=statut, taux=None if statut is None else F / p["NbRd"], messages=mots2)
    return v1, v2

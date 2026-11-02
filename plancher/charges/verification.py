# -*- coding: utf-8 -*-
"""Pipeline de vérification (§ 8 de l'étude) : charges → répartiteurs →
charges par élément → efforts → taux ; balayage de position ;
dimensionnement d'une filière par parcours du catalogue.

Le résultat est un dictionnaire sérialisable (JSON), ce qui permet de
figer des cas de référence. Chaque vérification porte sa famille, l'objet
vérifié, l'effet E, la résistance R, le taux et la méthode en toutes
lettres. Les statuts : ``ok``, ``attention`` (≥ seuil_attention),
``nok`` (> seuil_defaut), ``info`` (ligne d'information, P-11),
``non_verifie`` (paramètre à transcrire absent), ``refus`` (donnée
manquante ou situation hors périmètre : rien n'est calculé).
"""
import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import diffusion as DF
from . import repartiteurs as RP
from . import resistances as RS
from .modele import (Hourdis, DalleBA, Predalle, Element, Empreinte, ChargeLineaire, ChargeSurfacique,
                     Engin, Plateau, Filiere, EN_TRAVERS, DANS_LE_SENS, HOURDIS, DALLE, PREDALLE, RD)
from .parametres import Parametres, DEFAUT

OK, ATTENTION, NOK, INFO, NON_VERIFIE, REFUS = "ok", "attention", "nok", "info", "non_verifie", "refus"
ORDRE = {OK: 0, INFO: 0, NON_VERIFIE: 1, ATTENTION: 2, NOK: 3, REFUS: 4}


class Refus(Exception):
    """Situation hors périmètre ou donnée manquante : le calcul s'arrête
    et le message dit quoi faire."""


# ---------------------------------------------------------------- utilitaires
def _statut(taux: Optional[float], p: Parametres) -> str:
    if taux is None:
        return NON_VERIFIE
    if taux > p.seuil_defaut:
        return NOK
    if taux >= p.seuil_attention:
        return ATTENTION
    return OK


def _verif(famille, objet, E, R, taux, methode, p, statut=None, **extra) -> Dict:
    d = {"famille": famille, "objet": objet, "E": E, "R": R, "taux": taux,
         "statut": statut or _statut(taux, p), "methode": methode}
    d.update(extra)
    return d


def _empreintes(charges: Sequence, p: Parametres) -> List[Empreinte]:
    """Toutes les charges ramenées à des empreintes (les gabarits sans
    charge et les angles non droits sont des refus)."""
    out: List[Empreinte] = []
    for c in charges:
        if isinstance(c, Empreinte):
            out.append(c)
        elif isinstance(c, Engin):
            if not c.gabarit.charges_renseignees():
                manquants = [a.nom for a in c.gabarit.appuis if a.P <= 0]
                raise Refus(f"Engin « {c.ref} » : charge par appui non renseignée ({', '.join(manquants)}) — "
                            "à relire sur le tableau de stabilité du constructeur (P-13).")
            out.extend(c.empreintes())
        elif isinstance(c, ChargeLineaire):
            out.extend(c.discretiser(p.pas_lineaire))
        elif isinstance(c, ChargeSurfacique):
            out.extend(c.discretiser(p.pas_surfacique))
        else:
            raise TypeError(f"Charge inconnue : {type(c).__name__}")
    refs = [e.ref for e in out]
    if len(set(refs)) != len(refs):
        raise Refus("Deux charges portent la même référence : " + ", ".join(sorted({r for r in refs if refs.count(r) > 1})))
    return out


def _dans_le_plancher(e: Empreinte, plancher) -> None:
    if not (0.0 <= e.x <= plancher.L) or not (0.0 <= e.y <= plancher.B):
        raise Refus(f"Charge « {e.ref} » hors du plancher (x = {e.x:.2f} m, y = {e.y:.2f} m).")


# ---------------------------------------------------------------- régime 1 : plateaux
def _appliquer_plateau(e: Empreinte, pl: Plateau, p: Parametres) -> Tuple[Dict, List[Dict]]:
    """L'empreinte élargie à l'aire efficace du plateau, et les vérifications
    du plateau. Dimensions en m dans la charge, en mm dans la règle c."""
    verifs = []
    a_mm, b_mm = e.a * 1e3, e.b * 1e3
    if pl.rigide_declare:
        a_eff, b_eff = pl.A, pl.Bp
        pres = e.P_dyn * 1e3 / (a_eff * b_eff)
        detail = {"c": None, "part_efficace": 1.0, "complet": True, "p": pres}
        methode = "plaque déclarée rigide par le fournisseur : aire efficace = plaque entière"
        if pl.p_adm_declare is not None:
            verifs.append(_verif("plateau", f"{pl.nom} sous {e.ref}", pres, pl.p_adm_declare,
                                 pres / pl.p_adm_declare, "pression sous la plaque ≤ pression admissible publiée", p,
                                 unite="N/mm²"))
    else:
        if pl.f is None:
            raise Refus(f"Plateau « {pl.nom} » : résistance de flexion f absente — à renseigner avec sa source "
                        "(fy/γM0 pour l'acier, f_m,d pour le bois ou l'OSB).")
        # la bande plastique est une vérification ELU de l'acier : pression γQ·P sous l'aire efficace (P-2)
        ae = RP.aire_efficace(p.gamma_Q * e.P_dyn * 1e3, a_mm, b_mm, pl.A, pl.Bp, pl.t, pl.f)
        a_eff, b_eff = ae["a_eff"], ae["b_eff"]
        pres = e.P_dyn * 1e3 / (a_eff * b_eff)           # pression de SERVICE sous l'aire efficace (affichage, membrure)
        detail = ae
        methode = ("bande efficace c = t·√(f/(3·p_Ed)) de chaque côté de l'empreinte (EN 1993-1-8 §6.2.5, "
                   f"distribution plastique, p_Ed = γQ·P / aire avec γQ = {p.gamma_Q:g}) ; aire efficace (a + 2c)·(b + 2c)")
        verifs.append(_verif("plateau", f"{pl.nom} sous {e.ref}", ae["c"], ae["c_max"],
                             None, methode, p, statut=INFO, unite="mm", part_efficace=ae["part_efficace"],
                             complet=ae["complet"], a_eff=a_eff, b_eff=b_eff, pression=pres))
    patch = {"ref": e.ref, "x": e.x + pl.dx, "y": e.y + pl.dy, "a": a_eff / 1e3, "b": b_eff / 1e3,
             "P": e.P_dyn, "P_service": e.P, "phi": e.phi, "via": "plateau", "repartiteur": pl.nom,
             "pression": pres, "complet": detail["complet"], "origine": e.origine}
    return patch, verifs


# ---------------------------------------------------------------- régime 2 : filières
def _appliquer_filiere(fil: Filiere, portees: List[Empreinte], plancher, p: Parametres) -> Tuple[List[Dict], List[Dict], Dict]:
    """Les réactions de la filière deviennent des charges sur les éléments ;
    vérifications de la filière (flexion) et équilibre."""
    els = plancher.elements()
    s0, s1 = fil.axe()
    if s0 < -1e-9 or s1 > (plancher.B if fil.orientation == EN_TRAVERS else plancher.L) + 1e-9:
        raise Refus(f"Filière « {fil.nom} » dépasse le plancher.")
    patches, verifs = [], []
    if fil.orientation == EN_TRAVERS:
        appuis, infos = [], []
        for e in els:
            lo, hi = max(s0, e.y0), min(s1, e.y1)
            if hi - lo > 1e-6:
                s = 0.5 * (lo + hi) - s0
                k = plancher.raideur(e.index, fil.x)
                appuis.append((s, k))
                infos.append((e, lo, hi))
        charges = []
        for e in portees:
            if not (s0 - 1e-6 <= e.y <= s1 + 1e-6):
                raise Refus(f"Charge « {e.ref} » hors de la filière « {fil.nom} ».")
            charges.append((min(max(e.y - s0, 0.0), fil.longueur), e.P_dyn))
        res = RP.poutre_sur_ressorts(fil.EI, fil.longueur, appuis, charges)
        for (e, lo, hi), (s, R) in zip(infos, res["reactions"]):
            if R <= 1e-9:
                continue                       # un élément soulevé ne reçoit rien (ressort unilatéral : voir la note)
            patches.append({"ref": f"{fil.nom}→{e.nom}", "x": fil.x, "y": 0.5 * (lo + hi), "a": fil.b_semelle,
                            "b": hi - lo, "P": R, "P_service": R, "phi": 1.0, "via": "filiere",
                            "repartiteur": fil.nom, "pression": R * 1e3 / (fil.b_semelle * 1e3 * (hi - lo) * 1e3),
                            "complet": True, "origine": fil.nom})
        methode = ("poutre d'Euler sur appuis élastiques, un ressort par élément k = 3·E·I·L/(a²(L − a)²), "
                   "méthode des déplacements ; les réactions sont les charges des éléments")
        vf = RP.verifier_filiere(res, fil.W_el, fil.fy, fil.gamma_M0, p.gamma_Q)
        verifs.append(_verif("filiere", fil.nom, vf["M_Ed_service"], vf["M_Rd"], vf["taux"],
                             f"flexion de la filière (ELU) : γQ·M_max ≤ W_el·fy/γM0, γQ = {p.gamma_Q:g} sur les charges de "
                             "chantier ; poids propre de la filière négligé", p, unite="kN·m",
                             reactions=[(round(s + s0, 4), R) for s, R in res["reactions"]], souleve=[
                                 e.nom for (e, _, _), (_, R) in zip(infos, res["reactions"]) if R < -1e-9]))
        if any(R < -1e-9 for _, R in res["reactions"]):
            verifs.append(_verif("filiere", fil.nom, None, None, None,
                                 "un ressort tire (élément soulevé) : le modèle linéaire n'est plus exact, "
                                 "allonger ou raccourcir la filière", p, statut=NON_VERIFIE))
        return patches, verifs, {"nom": fil.nom, "orientation": fil.orientation, "resolution": res, "methode": methode}
    # dans le sens porteur : hypothèse rigide, charge linéaire sur le(s) élément(s) sous la filière
    total = sum(e.P_dyn for e in portees)
    n = max(1, int(math.ceil(fil.longueur / p.pas_lineaire)))
    dl = fil.longueur / n
    for k in range(n):
        x = s0 + (k + 0.5) * dl
        patches.append({"ref": f"{fil.nom}.{k + 1}", "x": x, "y": fil.y, "a": dl, "b": fil.b_semelle,
                        "P": total / n, "P_service": total / n, "phi": 1.0, "via": "filiere",
                        "repartiteur": fil.nom, "pression": total / n * 1e3 / (dl * 1e3 * fil.b_semelle * 1e3),
                        "complet": True, "origine": fil.nom})
    verifs.append(_verif("filiere", fil.nom, None, None, None,
                         "filière dans le sens porteur : aucune répartition entre éléments ; charge linéaire "
                         "uniforme (hypothèse rigide), flexion de la filière non calculée", p, statut=INFO))
    return patches, verifs, {"nom": fil.nom, "orientation": fil.orientation, "resolution": None,
                             "methode": "charge linéaire uniforme (hypothèse rigide)"}


def _sections(L: float, abscisses) -> List[float]:
    """Abscisses où le moment est évalué : celles des charges, la mi-portée et
    un pas ≤ L/60. Le maximum d'une somme de paraboles (charges réparties) et
    de triangles (charges concentrées) tombe ENTRE la charge et la mi-portée :
    ne regarder que ces deux points sous-estimait M_Ed d'environ 1 % pour une
    roue à 2 m d'un appui de 7,20 m — défaut NON CONSERVATIF trouvé le
    02/11/2026 en préparant la validation croisée ; au pas L/60, l'erreur sur
    une parabole est inférieure à w·(L/120)²/2, négligeable."""
    n = 60
    return sorted({L / 2} | set(abscisses) | {L * k / n for k in range(1, n)})


# ---------------------------------------------------------------- efforts : hourdis
def _efforts_hourdis(h: Hourdis, patches: List[Dict], p: Parametres) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    els = h.elements()
    n = len(els)
    charges_el: Dict[int, List[Tuple[float, float, str]]] = {e.index: [] for e in els}     # (x, P, origine) flexion
    tranchant_el: Dict[int, List[Tuple[float, float]]] = {e.index: [] for e in els}
    joints: Dict[int, float] = {}         # joint j (entre j et j+1) : force transmise
    verifs, locaux = [], []
    rep = h.repartition_active()
    for pt in patches:
        parts = DF.parts_levier(pt["y"], pt["b"], els)
        for i, part in parts.items():
            Pi = pt["P"] * part
            if rep:
                shares = DF.parts_voisins(h.parts_annexe_C, i, n)
            else:
                shares = {i: 1.0}
            for j, sh in shares.items():
                charges_el[j].append((pt["x"], Pi * sh, pt["ref"]))
                if j != i:
                    for jj in range(min(i, j), max(i, j)):
                        joints[jj] = joints.get(jj, 0.0) + Pi * sh
            ptr = p.part_tranchant_element
            tranchant_el[i].append((pt["x"], Pi * ptr))
            if ptr < 1.0:
                reste = 1.0 - shares.get(i, 0.0)
                for j, sh in shares.items():
                    if j != i and reste > 0:
                        tranchant_el[j].append((pt["x"], Pi * (1.0 - ptr) * sh / reste))
        # local : pression sur la membrure (P-5), à travers la chape s'il y en a une
        a_mm, b_mm = pt["a"] * 1e3, pt["b"] * 1e3
        if h.chape:
            a_mm += 2 * h.chape.epaisseur * 1e3
            b_mm += 2 * h.chape.epaisseur * 1e3
        pres = pt["P"] * 1e3 / (a_mm * b_mm)
        if h.p_loc_adm is not None:
            locaux.append(_verif("local", f"membrure sous {pt['ref']}", pres, h.p_loc_adm, pres / h.p_loc_adm,
                                 "pression sur la membrure supérieure ≤ pression locale admissible du fabricant "
                                 + ("(diffusion à 45° dans la chape) " if h.chape else "") + "(P-5)", p, unite="N/mm²"))
        elif pt["via"] in ("plateau", "filiere") and pt["complet"]:
            locaux.append(_verif("local", f"membrure sous {pt['ref']}", pres, None, None,
                                 "pression locale admissible non renseignée : non vérifiable, répartiteur présent (P-5)",
                                 p, statut=NON_VERIFIE, unite="N/mm²"))
        else:
            locaux.append(_verif("local", f"membrure sous {pt['ref']}", pres, None, None,
                                 "pression locale admissible non renseignée et pas de répartiteur efficace : "
                                 "répartiteur exigé (P-5)", p, statut=NOK, unite="N/mm²"))
    res_h = RS.resistance_hourdis(h, p.gamma_G, p.gamma_Q)
    gG, gQ = res_h["gamma_G"], res_h["gamma_Q"]
    elements = []
    for e in els:
        # P-2 : les efforts SOLLICITANTS comptent les poids propres (G : hourdis + chape) et les
        # charges de chantier (Q : charges présentes + charges concentrées × φ), chacun avec son
        # coefficient — γG et γQ si la résistance est une valeur de calcul, 1 et 1 si elle est
        # une valeur admissible de service
        w_g = h.g_perm() * e.largeur            # kN/m, permanent
        w_q = h.q_presentes * e.largeur         # kN/m, charges déjà présentes (variable)
        xs = _sections(h.L, [x for x, _, _ in charges_el[e.index]])
        M_max, x_M, M_G, M_Q = 0.0, h.L / 2, 0.0, 0.0
        for xs_ in xs:
            mg = DF.moment_uniforme(w_g, xs_, h.L)
            mq = DF.moment_uniforme(w_q, xs_, h.L) + sum(DF.moment_ponctuel(P, x, xs_, h.L) for x, P, _ in charges_el[e.index])
            m = gG * mg + gQ * mq
            if m > M_max:
                M_max, x_M, M_G, M_Q = m, xs_, mg, mq
        VA = gG * DF.tranchant_uniforme(w_g, h.L) + gQ * (DF.tranchant_uniforme(w_q, h.L)
                                                          + sum(DF.reactions_ponctuelle(P, x, h.L)[0] for x, P in tranchant_el[e.index]))
        VB = gG * DF.tranchant_uniforme(w_g, h.L) + gQ * (DF.tranchant_uniforme(w_q, h.L)
                                                          + sum(DF.reactions_ponctuelle(P, x, h.L)[1] for x, P in tranchant_el[e.index]))
        V_max = max(VA, VB)
        tM = M_max / res_h["M_adm"] if res_h["M_adm"] > 0 else math.inf
        tV = V_max / res_h["V_adm"] if res_h["V_adm"] > 0 else math.inf
        gouverne = "flexion" if tM >= tV else "tranchant"
        elements.append({"nom": e.nom, "index": e.index, "y0": e.y0, "y1": e.y1, "M_max": M_max, "x_M": x_M,
                         "M_G": M_G, "M_Q": M_Q, "gamma_G": gG, "gamma_Q": gQ,
                         "M_adm": res_h["M_adm"], "taux_M": tM, "V_A": VA, "V_B": VB, "V_max": V_max,
                         "V_adm": res_h["V_adm"], "taux_V": tV, "taux": max(tM, tV), "gouverne": gouverne,
                         "charges": [(x, P, o) for x, P, o in charges_el[e.index]], "w": w_g + w_q, "w_g": w_g, "w_q": w_q})
        verifs.append(_verif("flexion", e.nom, M_max, res_h["M_adm"], tM, res_h["methode"], p, unite="kN·m",
                             M_G=M_G, M_Q=M_Q, gamma_G=gG, gamma_Q=gQ))
        verifs.append(_verif("tranchant", e.nom, V_max, res_h["V_adm"], tV,
                             res_h["methode"] + f" ; tranchant : {p.part_tranchant_element:.0%} de la réaction d'appui de chaque "
                             "charge concentrée sur l'élément chargé, sans partage par les joints (P-3)",
                             p, unite="kN", gamma_G=gG, gamma_Q=gQ))
        if res_h["M_cr"] is not None and M_max > 0:
            verifs.append(_verif("fissuration", e.nom, M_G + M_Q, res_h["M_cr"], (M_G + M_Q) / res_h["M_cr"],
                                 "moment de SERVICE (coefficients 1) ≤ moment de fissuration du fabricant (information, P-11)", p,
                                 statut=INFO, unite="kN·m"))
    # joints (P-15)
    for j, F in sorted(joints.items()):
        objet = f"joint {els[j].nom} | {els[j + 1].nom}"
        if p.joint_adm_kN_m is not None and p.joint_longueur_m is not None:
            cap = p.joint_adm_kN_m * p.joint_longueur_m
            verifs.append(_verif("joint", objet, F, cap, p.gamma_Q * F / cap if cap > 0 else math.inf,
                                 f"effort transmis par le joint (parts annexe C) ≤ résistance admissible × longueur mobilisée, γQ = {p.gamma_Q:g} sur l'effort (P-15)",
                                 p, unite="kN"))
        else:
            verifs.append(_verif("joint", objet, F, None, None,
                                 "effort transmis par le joint (parts annexe C) : résistance du joint de l'EN 1168 à transcrire (P-15)",
                                 p, statut=NON_VERIFIE, unite="kN"))
    if res_h["note"]:
        verifs.append(_verif("hypothese", h.nom, None, None, None, res_h["note"], p, statut=INFO))
    return elements, verifs, locaux


# ---------------------------------------------------------------- efforts : dalle et prédalle
def _largeurs(d: DalleBA, pt: Dict, p: Parametres) -> Tuple[float, float, float, List[str]]:
    """(b_m, b_v,A, b_v,B) d'une charge sur une dalle, avec les plafonds."""
    notes = []
    ty = DF.t_y(pt["b"], d.h1, d.h, d.L)
    if not d.armature_transversale_ok:
        notes.append("armature transversale < 20 % : pas de largeur efficace au-delà de t_y")
    bm = DF.b_m(ty, pt["x"], d.L, p.k_flexion_heft240 if d.armature_transversale_ok else None, d.B)
    bvA = DF.b_v(ty, pt["x"], d.B) if d.armature_transversale_ok else ty
    bvB = DF.b_v(ty, d.L - pt["x"], d.B) if d.armature_transversale_ok else ty
    if isinstance(d, Predalle):
        y0, y1 = pt["y"] - pt["b"] / 2, pt["y"] + pt["b"] / 2
        a_cheval = any(y0 < j < y1 for j in d.joints())
        if a_cheval:
            notes.append(f"charge « {pt['ref']} » à cheval sur un joint de prédalle (P-16)")
        if not d.joints_armes or a_cheval:
            bm, bvA, bvB = min(bm, d.b_predalle), min(bvA, d.b_predalle), min(bvB, d.b_predalle)
            notes.append("largeur efficace plafonnée à la prédalle")
    return bm, bvA, bvB, notes


def _efforts_dalle(d: DalleBA, patches: List[Dict], p: Parametres) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    res_d = RS.resistance_dalle(d)
    L = d.portee_etayee if isinstance(d, Predalle) and d.portee_etayee else d.L
    verifs, locaux, notes_all = [], [], []
    infos = []
    for pt in patches:
        bm, bvA, bvB, notes = _largeurs(d, pt, p)
        notes_all.extend(notes)
        infos.append((pt, bm, bvA, bvB))
    w = d.g_total()
    # moment par mètre : sections candidates = abscisses des charges, mi-portée et pas L/60
    xs = _sections(L, [pt["x"] for pt in patches])
    best = (0.0, L / 2, 0.0)
    for xs_ in xs:
        boites = [(pt["y"], bm, DF.moment_ponctuel(pt["P"], pt["x"], xs_, L)) for pt, bm, _, _ in infos]
        m_pt, y_m = DF.max_creneaux(boites, 0.0, d.B)
        m = DF.moment_uniforme(w, xs_, L) + m_pt
        if m > best[0]:
            best = (m, xs_, y_m)
    M_max, x_M, y_M = best
    boites_A = [(pt["y"], bvA, DF.reactions_ponctuelle(pt["P"], pt["x"], L)[0]) for pt, _, bvA, _ in infos]
    boites_B = [(pt["y"], bvB, DF.reactions_ponctuelle(pt["P"], pt["x"], L)[1]) for pt, _, _, bvB in infos]
    vA, yA = DF.max_creneaux(boites_A, 0.0, d.B)
    vB, yB = DF.max_creneaux(boites_B, 0.0, d.B)
    VA, VB = DF.tranchant_uniforme(w, L) + vA, DF.tranchant_uniforme(w, L) + vB
    V_max, y_V = (VA, yA) if VA >= VB else (VB, yB)
    tM = M_max / res_d["M_adm"] if res_d["M_adm"] > 0 else math.inf
    tV = V_max / res_d["V_adm"] if res_d["V_adm"] > 0 else math.inf
    methode = res_d["methode"] + (" ; largeur efficace Heft 240 : b_v = t_y + 0,5·a ; b_m = t_y"
                                  + (" + k·x(1 − x/l)" if p.k_flexion_heft240 is not None else " (k non transcrit, P-4)"))
    verifs.append(_verif("flexion", f"bande gouvernante y = {y_M:.2f} m", M_max, res_d["M_adm"], tM, methode, p, unite="kN·m/m",
                         x=x_M, classe=res_d["classe"], gouverne_M=res_d["gouverne_M"]))
    verifs.append(_verif("tranchant", f"appui {'A' if VA >= VB else 'B'}, y = {y_V:.2f} m", V_max, res_d["V_adm"], tV,
                         methode, p, unite="kN/m", classe=res_d["classe"]))
    # par bande (pour l'affichage) : maximum du profil dans chaque bande
    elements = []
    for e in d.elements():
        # les créneaux restent tronqués au PLANCHER ; seule la recherche du maximum se limite à la bande
        boites = [(pt["y"], bm, DF.moment_ponctuel(pt["P"], pt["x"], x_M, L)) for pt, bm, _, _ in infos]
        mb, _ = DF.max_creneaux(boites, 0.0, d.B, fenetre=(e.y0, e.y1)) if boites else (0.0, e.y0)
        mb += DF.moment_uniforme(w, x_M, L)
        vbA, _ = DF.max_creneaux(boites_A, 0.0, d.B, fenetre=(e.y0, e.y1)) if boites_A else (0.0, e.y0)
        vbB, _ = DF.max_creneaux(boites_B, 0.0, d.B, fenetre=(e.y0, e.y1)) if boites_B else (0.0, e.y0)
        vb = DF.tranchant_uniforme(w, L) + max(vbA, vbB)
        tMb = mb / res_d["M_adm"] if res_d["M_adm"] > 0 else math.inf
        tVb = vb / res_d["V_adm"] if res_d["V_adm"] > 0 else math.inf
        elements.append({"nom": e.nom, "index": e.index, "y0": e.y0, "y1": e.y1, "M_max": mb, "x_M": x_M,
                         "M_adm": res_d["M_adm"], "taux_M": tMb, "V_max": vb, "V_adm": res_d["V_adm"], "taux_V": tVb,
                         "taux": max(tMb, tVb), "gouverne": "flexion" if tMb >= tVb else "tranchant"})
    # effets locaux EN 1992-1-1 : vérifications ELU, γQ sur la charge concentrée (charge de chantier, P-2 et P-9)
    fck = res_d["fck"]
    for pt in patches:
        a_mm, b_mm = pt["a"] * 1e3, pt["b"] * 1e3
        pl = RS.pression_localisee_Rd(a_mm, b_mm, d.h * 1e3, fck, p.gamma_c, p.alpha_cc)
        locaux.append(_verif("local", f"pression localisée sous {pt['ref']}", p.gamma_Q * pt["P"], pl["F_Rdu_kN"],
                             p.gamma_Q * pt["P"] / pl["F_Rdu_kN"], f"EN 1992-1-1 §6.7 (6.63), ELU, γQ = {p.gamma_Q:g} sur la charge (P-9)",
                             p, unite="kN"))
        po = RS.poinconnement(p.gamma_Q * pt["P"], a_mm, b_mm, d.d * 1e3, fck, d.rho_l, p.gamma_c, p.alpha_cc,
                              p.beta_poinconnement, p.coef_vrd_max)
        locaux.append(_verif("local", f"poinçonnement sous {pt['ref']}", po["v_Ed"], po["v_Rd_c"], po["taux"],
                             f"EN 1992-1-1 §6.4, contour u1 à 2d, v_Rd,c (6.47) ≥ v_min, ELU, γQ = {p.gamma_Q:g} sur la charge (P-9)",
                             p, unite="N/mm²"))
        if "taux_u0" in po:
            locaux.append(_verif("local", f"poinçonnement au nu sous {pt['ref']}", po["v_Ed0"], po["v_Rd_max"], po["taux_u0"],
                                 "EN 1992-1-1 §6.4.5(3), v_Rd,max = coef·ν·fcd (coefficient à confirmer sur l'ANB)", p, unite="N/mm²"))
    for n_ in sorted(set(notes_all)):
        verifs.append(_verif("hypothese", d.nom, None, None, None, n_, p, statut=INFO))
    if p.k_flexion_heft240 is None and patches:
        verifs.append(_verif("hypothese", d.nom, None, None, None,
                             "coefficient k de la largeur efficace en flexion non transcrit : b_m = t_y, résultat très "
                             "conservatif sous charge concentrée (P-4)", p, statut=INFO))
    if isinstance(d, Predalle) and d.portee_etayee:
        verifs.append(_verif("hypothese", d.nom, None, None, None,
                             f"étais en place : portée réduite à {d.portee_etayee:.2f} m ; les charges se reportent sur l'étage inférieur, non modélisé (P-8)",
                             p, statut=ATTENTION))
    return elements, verifs, locaux


# ---------------------------------------------------------------- pipeline
def verifier(plancher, charges: Sequence, repartiteurs: Optional[Dict[str, object]] = None,
             params: Parametres = DEFAUT) -> Dict:
    """Vérifie un plancher sous ses charges et répartiteurs. Ne lève jamais :
    un refus devient un résultat de statut ``refus`` avec son message."""
    p = params
    repartiteurs = dict(repartiteurs or {})
    try:
        if plancher.type == HOURDIS and (plancher.M_Rd <= 0.0 or plancher.V_Rd <= 0.0):
            raise Refus("Hourdis : résistances M_Rd et V_Rd du catalogue à renseigner (étape 1, fiche du fabricant) ; "
                        "aucune vérification n'est possible sans elles.")
        if isinstance(plancher, Predalle) and not plancher.chape_durcie:
            if any(not isinstance(c, ChargeSurfacique) for c in charges):
                raise Refus("Prédalle sans chape durcie : seules les charges surfaciques de bétonnage sont admises ; "
                            "pas d'engin ni d'étai (P-7) — voir les tables d'étaiement du fabricant.")
        empreintes = _empreintes(charges, p)
        for e in empreintes:
            _dans_le_plancher(e, plancher)
        patches, verifs, filieres = [], [], []
        par_filiere: Dict[int, Tuple[Filiere, List[Empreinte]]] = {}
        for e in empreintes:
            r = repartiteurs.get(e.ref) or repartiteurs.get(e.origine)
            if isinstance(r, Filiere):
                par_filiere.setdefault(id(r), (r, []))[1].append(e)
            elif isinstance(r, Plateau):
                pt, vs = _appliquer_plateau(e, r, p)
                patches.append(pt)
                verifs.extend(vs)
            else:
                patches.append({"ref": e.ref, "x": e.x, "y": e.y, "a": e.a, "b": e.b, "P": e.P_dyn, "P_service": e.P,
                                "phi": e.phi, "via": "direct", "repartiteur": None,
                                "pression": e.P_dyn * 1e3 / (e.a * 1e3 * e.b * 1e3) if e.a * e.b > 0 else math.inf,
                                "complet": False, "origine": e.origine})
        for fil, portees in par_filiere.values():
            pts, vs, info = _appliquer_filiere(fil, portees, plancher, p)
            patches.extend(pts)
            verifs.extend(vs)
            filieres.append(info)
        for pt in patches:
            if not (0.0 <= pt["x"] <= plancher.L) or not (0.0 <= pt["y"] <= plancher.B):
                raise Refus(f"Charge « {pt['ref']} » hors du plancher après répartition (x = {pt['x']:.2f}, y = {pt['y']:.2f}).")
        if plancher.type == HOURDIS:
            elements, vs, locaux = _efforts_hourdis(plancher, patches, p)
        else:
            elements, vs, locaux = _efforts_dalle(plancher, patches, p)
        verifs.extend(vs)
        verifs.extend(locaux)
    except Refus as r:
        return {"statut": REFUS, "message": str(r), "taux_max": None, "gouverne": None, "elements": [],
                "verifications": [], "patches": [], "filieres": [], "hypotheses": _hypotheses(plancher, p)}
    chiffrees = [v for v in verifs if v["taux"] is not None and v["statut"] in (OK, ATTENTION, NOK)]
    pire = max(chiffrees, key=lambda v: v["taux"]) if chiffrees else None
    statut = max((v["statut"] for v in verifs), key=lambda s: ORDRE[s]) if verifs else OK
    if statut in (INFO, NON_VERIFIE):
        statut = OK
    if any(v["statut"] == NOK for v in verifs):
        statut = NOK
    return {"statut": statut, "message": "", "taux_max": pire["taux"] if pire else 0.0,
            "gouverne": {k: pire[k] for k in ("famille", "objet", "taux")} if pire else None,
            "elements": elements, "verifications": verifs, "patches": patches, "filieres": filieres,
            "hypotheses": _hypotheses(plancher, p), "non_verifies": [v for v in verifs if v["statut"] == NON_VERIFIE]}


def _hypotheses(plancher, p: Parametres) -> List[str]:
    h = []
    if plancher.type == HOURDIS:
        if plancher.nature == RD:
            h.append(f"hourdis, résistances de CALCUL du catalogue (M_Rd, V_Rd) : efforts à l'ELU, M_Ed = γG·M_G + γQ·M_Q avec "
                     f"γG = {p.gamma_G:g} sur les poids propres (hourdis, chape) et γQ = {p.gamma_Q:g} sur les charges de chantier "
                     "(charges présentes, engins × φ, étais, stocks) (P-2)")
        else:
            h.append("hourdis, valeurs ADMISSIBLES de service du catalogue : efforts de service, coefficients 1 sur G et Q (P-2)")
        h.append("flexion : " + ("parts de la figure C.1 de l'EN 1168 fournies par le bureau (joints clavetés ou plancher chapé)"
                                 if plancher.repartition_active() else "aucune répartition entre éléments, l'élément chargé porte tout")
                 + f" ; tranchant : {p.part_tranchant_element:.0%} de la réaction d'appui de chaque charge concentrée reste sur "
                 "l'élément chargé, sans partage par les joints — prudent, l'annexe C répartit la flexion à mi-portée, pas "
                 "l'effort tranchant près de l'appui (P-3)")
        if plancher.chape:
            h.append(f"couche de compression {plancher.chape.epaisseur * 1e3:.0f} mm, fck(t) = {plancher.chape.fck_t:g} MPa"
                     + (" : valeurs catalogue avec chape" if plancher.chape.valeurs_avec_chape_admises() else " : valeurs nues, chape en poids (P-17)"))
    else:
        h.append("béton armé : méthode aux contraintes admissibles de l'application (αb, μ, τadm,I) sur les charges de SERVICE, "
                 f"coefficients 1 sur G et Q ; effets locaux EN 1992-1-1 §6.4 et §6.7 à l'ELU avec γQ = {p.gamma_Q:g} sur la charge (P-9)")
    h.append("charges saisies en valeurs de service ; coefficient dynamique φ par appui (EN 1991-1-1 §6.3.2.3 pour les roues) ; "
             "les poids propres (élément, chape) entrent dans les efforts sollicitants")
    return h


# ---------------------------------------------------------------- balayage
def balayer(plancher, charges: Sequence, repartiteurs, ref_engin: str, params: Parametres = DEFAUT,
            x_min: Optional[float] = None, x_max: Optional[float] = None) -> Dict:
    """Déplace l'engin ``ref_engin`` le long de la portée, par pas
    ``pas_balayage``, et rend l'enveloppe (x, taux_max, statut) et la
    position la plus défavorable. Les répartiteurs suivent l'engin (ils
    sont relatifs à ses empreintes)."""
    p = params
    engin = next(c for c in charges if isinstance(c, Engin) and c.ref == ref_engin)
    k = (engin.angle // 90) % 4
    offs = []
    for a in engin.gabarit.appuis:
        dx, dy = a.dx, a.dy
        for _ in range(k):
            dx, dy = -dy, dx
        offs.append(dx)
    lo = -min(offs) if x_min is None else x_min
    hi = plancher.L - max(offs) if x_max is None else x_max
    points, pire = [], None
    x = lo
    while x <= hi + 1e-9:
        c2 = [copy.copy(c) for c in charges]
        for c in c2:
            if isinstance(c, Engin) and c.ref == ref_engin:
                c.x = x
        r = verifier(plancher, c2, repartiteurs, p)
        t = r["taux_max"] if r["statut"] != REFUS else None
        points.append({"x": round(x, 4), "taux": t, "statut": r["statut"]})
        if t is not None and (pire is None or t > pire["taux"]):
            pire = {"x": round(x, 4), "taux": t, "gouverne": r["gouverne"]}
        x += p.pas_balayage
    admissibles = [pt["x"] for pt in points if pt["taux"] is not None and pt["taux"] <= p.seuil_defaut]
    return {"points": points, "pire": pire, "x_admissibles": admissibles, "x_min": lo, "x_max": hi}


# ---------------------------------------------------------------- dimensionnement d'une filière
def dimensionner_filiere(plancher, charges: Sequence, repartiteurs: Dict, groupes: Sequence[Sequence[str]],
                         candidats: Sequence[Tuple[str, float, float]], fabrique, params: Parametres = DEFAUT,
                         n_elements=(2, 3, 4, 5)) -> Dict:
    """Parcourt des candidats ``(nom, EI, W_el)`` triés par masse pour UNE
    filière en travers par groupe d'empreintes (``groupes`` : listes de
    refs de même x), de longueur croissante (``n_elements`` éléments) ;
    ``fabrique(nom, longueur, x, y, refs)`` construit une filière. Retient
    le premier candidat dont le système entier est ≤ 1 (P-14) et rend le
    résultat complet, le pire taux du système et les taux qui ont monté."""
    p = params
    avant = verifier(plancher, charges, repartiteurs, p)
    emp = {e.ref: e for e in _empreintes(charges, p)}
    larg = plancher.elements()[0].largeur
    essais = []
    for n_el in n_elements:
        longueur = n_el * larg
        for nom, EI, W in candidats:
            rep = dict(repartiteurs)
            fils = []
            for refs in groupes:
                cibles = [emp[r] for r in refs]
                x = sum(e.x for e in cibles) / len(cibles)
                y = sum(e.y for e in cibles) / len(cibles)
                yc = min(max(y, longueur / 2), plancher.B - longueur / 2)
                fil = fabrique(nom, longueur, x, yc, list(refs))
                fils.append(fil)
                for r in refs:
                    rep[r] = fil
            res = verifier(plancher, charges, rep, p)
            essais.append({"profil": nom, "longueur": longueur, "taux_max": res["taux_max"], "statut": res["statut"]})
            if res["statut"] in (OK, ATTENTION):
                return {"trouve": True, "filieres": fils, "repartiteurs": rep, "resultat": res,
                        "taux_max": res["taux_max"], "gouverne": res["gouverne"],
                        "taux_montes": _taux_montes(avant, res), "essais": essais, "avant": avant["taux_max"]}
    return {"trouve": False, "essais": essais, "avant": avant["taux_max"]}


def _taux_montes(avant: Dict, apres: Dict) -> List[Dict]:
    """Les éléments dont le taux a monté entre deux résultats (P-14)."""
    av = {e["nom"]: e["taux"] for e in avant.get("elements", [])}
    out = []
    for e in apres.get("elements", []):
        if e["nom"] in av and e["taux"] > av[e["nom"]] + 1e-9:
            out.append({"nom": e["nom"], "avant": av[e["nom"]], "apres": e["taux"]})
    return out

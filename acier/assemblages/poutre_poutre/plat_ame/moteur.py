# -*- coding: utf-8 -*-
"""Moteur de calcul — assemblage poutre–poutre par plat d'âme soudé (fin
plate). Écrit pour CET assemblage : rotule à la face d'appui, un seul plan
de cisaillement, moment d'excentricité VEd·z repris par le groupe de
boulons, le plat, la soudure et l'âme porteuse.

Modèles et sources, distingués vérification par vérification (champ ``nat``) :
- ``EC``   : formule directe de l'EN 1993-1-8 / EN 1993-1-1 ;
- ``COMP`` : modèle complémentaire d'un guide reconnu — SCI P358 (édition de
             2014), la procédure RETENUE par le bureau le 27/11/2026 pour le
             plat LONG (zp > tp/0,15 : déversement du plat, Check 3, et
             section ABCD de l'âme portée, Check 4), le grugeage court
             (Check 5), le poinçonnement de l'âme porteuse (Check 10) et le
             détail de l'articulation (§5.3) — ``acier/assemblages/sci.py`` ;
             « Steel Buildings in Europe, Part 5: Joint Design » §3 pour ce
             que les deux guides écrivent de la même façon ; ECCS n°126 ;
- ``INT``  : interprétation de l'outil, nommée comme telle.

Coefficients partiels par défaut : ANB belge pour l'EN 1993-1-1 (γM1 = 1,10
au lieu de 1,00 recommandé — il ne sert qu'au déversement du plat long) ;
valeurs recommandées pour l'EN 1993-1-8. Tous modifiables.

Unités : mm, mm², mm³, MPa, kN, kNm, N/mm (cordons).
"""
import math

from acier.bibliotheques import DB, PERSO, find, idx
from acier.assemblages import materiau as MT
from acier.assemblages import ductilite as DU
from acier.assemblages import mecanique as MEC
from acier.assemblages import sci as SCI
from acier.js import AttrDict, INF, N, dv, mn, mx, js_round, ceil5, hyp, js_str
from acier.formats import F, f0, nb
from acier.resultats import Verification, Alerte, Pince, EffortBoulon
from acier import resistances as RS
from .entrees import defaults, NUM, MODE_PREDIM

S3 = math.sqrt(3)
E_BS = 205000.0                 # module d'Young de BS 5950-1 (courbe Table 17)
T34 = "EN 1993-1-8 Tableau 3.4"
MSB3 = "MSB Part 5 §3.2 (fin plate)"

GROUPES_VERIF = ("Boulons", "Plat d'âme", "Poutre secondaire",
                 "Poutre principale", "Soudure")


def _first_ge(liste, v):
    c = sum(1 for x in liste if x < v)
    return min(c, len(liste) - 1)


def pb_bs5950(lam, py):
    """Contrainte de déversement pb (MPa) — BS 5950-1 Annexe B.2.1, la
    formule qui engendre la Table 17 : Perry-Robertson avec αb = 0,007,
    λL0 = 0,4·√(π²E/py), E = 205 000 MPa. pb = py pour λ ≤ λL0."""
    if py <= 0:
        return 0.0
    lam0 = 0.4 * math.sqrt(math.pi * math.pi * E_BS / py)
    if lam <= lam0:
        return py
    eta = 0.007 * (lam - lam0)
    pE = math.pi * math.pi * E_BS / (lam * lam)
    phi = (py + (eta + 1) * pE) / 2
    rac = phi * phi - pE * py
    return dv(pE * py, phi + math.sqrt(rac)) if rac > 0 else py


# =========================================================================
#  compute
# =========================================================================
def compute(u0=None, materiau=None):
    """Calcul complet pour les saisies ``u0`` (clés absentes → défauts).
    ``materiau`` : la fonction (nom, t) → acier qui remplace la table du site
    (``acier/assemblages/materiau.py``) — le matériau d'un exemple publié ou
    d'un calcul à la main (benchmark) ; None : f_y et f_u selon l'épaisseur de
    chaque pièce (NBN EN 1993-1-1 ANB:2018 3.2.1(1), tableau 3.3 ANB).
    Retourne ``R`` : scalaires du modèle, ``u``, ``checks`` / ``ck``,
    ``alerts``, ``dist`` (Tableau 3.3), ``tab`` (efforts par boulon),
    ``pd`` (prédimensionnement), ``gov``, ``eta_max``, ``verified``,
    ``statut``, ``reserve``."""
    u = AttrDict(defaults())
    for k, v in (u0 or {}).items():
        u[k] = v
    for n in NUM:
        u[n] = N(u[n])
    v_inverse = bool(u.V_Ed < 0)                # sens inversé : traité en valeur absolue, chemin à justifier (statut)
    u.V_Ed = abs(u.V_Ed)
    Mabs = abs(u.M_Ed)
    R = AttrDict()
    R.u = u
    R.V_inverse = v_inverse
    R.checks = []
    R.ck = {}
    R.pred = u.mode_calc == MODE_PREDIM
    for X in ("P", "S"):
        if u["prof_" + X] == PERSO:
            p = dict(h=N(u["h" + X + "_u"]), b=N(u["b" + X + "_u"]),
                     tw=N(u["tw" + X + "_u"]), tf=N(u["tf" + X + "_u"]),
                     r=N(u["r" + X + "_u"]))
        else:
            p = find(DB["profils"], u["prof_" + X])
        for d in ("h", "b", "tw", "tf", "r"):
            R[d + "_" + X] = p[d]
    # ---- aciers : f_y et f_u de CHAQUE pièce selon son épaisseur (tableau 3.3 ANB) — profilés : max(tf, tw) ;
    # plat : le prédimensionnement le lit à ses épaisseurs (MT.T_PREDIM), la vérification à l'épaisseur retenue
    perso = dict(fy=N(u.fy_u), fu=N(u.fu_u), bw=N(u.bw_u))
    for X, cle, piece in (("P", "nu_P", "poutre principale"), ("S", "nu_S", "poutre secondaire")):
        MT.poser(R, X, u[cle], MT.t_profil(R["tf_" + X], R["tw_" + X]), perso, materiau, cle, piece)
    MT.poser(R, "L", u.nu_pl, MT.T_PREDIM, perso, materiau, "nu_pl", "plat")
    cl = find(DB["classes"], u.classe)
    R.f_ub = cl["fub"]
    R.a_v = cl["av"] if u.filet == "Oui" else 0.6
    R.k_trou = 1 if u.trou == "Normal" else 0.8
    pd = R.pd = predim(u, R, Mabs)

    # ---- paramètres retenus (◆ en prédimensionnement)
    R.boulon = pd.boulon if R.pred else u.boulon_u

    def ret(a, b):
        return b if R.pred else a

    R.n_1 = ret(u.n1_u, pd.n); R.n_2 = ret(u.n2_u, 1); R.p_1 = ret(u.p1_u, pd.p1)
    R.e_1 = ret(u.e1_u, pd.e1); R.e2b = ret(u.e2b_u, pd.e2)
    R.h_p = ret(u.hp_u, pd.hp); R.t_p = ret(u.tp_u, pd.tp)
    MT.poser(R, "L", u.nu_pl, R.t_p, perso, materiau, "nu_pl", "plat")
    R.b_p = ret(u.bp_u, pd.bp); R.a_w = ret(u.a_w, pd.a)
    R.n_1 = max(1, js_round(R.n_1)); R.n_2 = 2 if N(R.n_2) >= 2 else 1
    bo = find(DB["boulons"], R.boulon)
    R.d_b = bo["d"]; R.d_0 = N(u.d0_u) if N(u.d0_u) > 0 else bo["d0"]
    R.As_b = bo["As"]; R.A_cis = bo["As"] if u.filet == "Oui" else bo["A"]
    R.d_m = bo["dm"]; R.d_w = bo["dw"]
    d0 = R.d_0; d = R.d_b; V = u.V_Ed; NEd = u.N_Ed
    gM0 = u.g_M0; gM1 = u.g_M1; gM2 = u.g_M2; gM2n = u.g_M2n

    # ---- géométrie (z depuis la FACE D'APPUI : rotule à la face du support)
    R.g_B = u.g_h + R.e2b                       # face d'appui → 1re file
    R.z_p = R.g_B + (R.n_2 - 1) * u.p2_u / 2    # face d'appui → centre du groupe
    R.e2_p = R.b_p - R.g_B - (R.n_2 - 1) * u.p2_u   # pince du plat, bord libre
    R.e1bot = R.h_p - R.e_1 - (R.n_1 - 1) * R.p_1
    R.e1b_S = u.z_C + R.e_1 - u.d_nt
    R.h_e = R.h_S - u.d_nb - (u.z_C + R.e_1 + (R.n_1 - 1) * R.p_1)
    R.n_b = R.n_1 * R.n_2
    R.Ip = R.n_b * (R.p_1 * R.p_1 * (R.n_1 * R.n_1 - 1)
                    + u.p2_u * u.p2_u * (R.n_2 * R.n_2 - 1)) / 12
    R.xm = (R.n_2 - 1) * u.p2_u / 2; R.ym = (R.n_1 - 1) * R.p_1 / 2
    R.zeff = R.z_p
    R.M_S = V * R.zeff / 1000 + Mabs
    R.zt_pl = u.d_top + u.z_C                   # dessus porteuse → dessus plat
    R.mod = ("Distribution uniforme" if R.M_S == 0 else
             "Rotule à la face d'appui — groupe de boulons avec moment "
             "VEd·z (répartition élastique, inertie polaire)")
    R.rec_hp = R.h_p >= 0.6 * R.h_S             # maintien en torsion (guides)
    DU.poser(R, d)                              # ductilité : plat OU âme portée ≤ k·d (SCI P358 §5.3 (3))
    # plat LONG : le SCI le juge sur zp, de la face d'appui à la PREMIÈRE file (R.g_B) — règle retenue le 27/11/2026 ;
    # le MSB P5 le juge sur z, au centre du groupe (R.z_p) : à deux files, un plat peut être long pour lui seul
    R.long_p = R.g_B > dv(R.t_p, SCI.SEUIL_PLAT_LONG)
    R.long_msb = R.z_p > dv(R.t_p, SCI.SEUIL_PLAT_LONG)
    # détail ÉPROUVÉ de l'articulation (SCI P358 §5.3 (5)) : toutes les pinces du plat et de l'âme portée ≥ 2d
    pinces = [("e1 du plat", mn(R.e_1, R.e1bot)), ("e2 du plat", R.e2_p), ("e2,b de l'âme portée", R.e2b)]
    if u.d_nt > 0:
        pinces.append(("e1,b de l'âme portée", R.e1b_S))
    if u.d_nb > 0:
        pinces.append(("he de l'âme portée", R.h_e))
    R.pince_2d = min(pinces, key=lambda x: x[1])
    R.rec_2d = R.pince_2d[1] >= SCI.N_2D * d - 1e-9
    R.rec_boul = bool(u.classe == "8.8" and u.cat == "A")      # §5.3 (4) : 8.8 non précontraints

    # ---- Tableau 3.3
    R.dist = []

    def drow(id_, lab, val, kmin, tmax, kind, act, fields, dims):
        if not act:
            return
        mn_ = kmin * d0
        if kind == "e":
            mx_ = 4 * tmax + 40 if u.expo == "Oui" else None
            rule = "e ≥ " + F(kmin, 1) + "·d0 ; e ≤ 4t + 40 mm si exposé"
        else:
            mx_ = mn(14 * tmax, 200)
            rule = "p ≥ " + F(kmin, 1) + "·d0 ; p ≤ min(14t ; 200 mm)"
        R.dist.append(Pince(id=id_, lab=lab, val=val, min=mn_, max=mx_, kmin=kmin,
                            fields=list(fields), dims=list(dims), rule=rule,
                            ok=bool(val >= mn_ - 0.001 and (mx_ is None or val <= mx_))))

    tmin = mn(R.t_p, R.tw_S)
    drow("e1p", "Plat – e1 (haut/bas, valeur mini)", mn(R.e_1, R.e1bot), 1.2, R.t_p, "e", True,
         ["e1_u", "hp_u", "n1_u", "p1_u"], ["e1", "e1bot"])
    drow("e2p", "Plat – e2 (bord libre)", R.e2_p, 1.2, R.t_p, "e", True,
         ["bp_u", "g_h", "e2b_u", "p2_u"], ["e2p"])
    drow("e1b", "Âme poutre secondaire – e1,b (vers grugeage sup.)", R.e1b_S, 1.2, R.tw_S, "e",
         u.d_nt > 0, ["z_C", "e1_u", "d_nt"], ["e1b", "zc", "dnt"])
    drow("he", "Âme poutre secondaire – he (vers grugeage inf.)", R.h_e, 1.2, R.tw_S, "e",
         u.d_nb > 0, ["z_C", "e1_u", "n1_u", "p1_u", "d_nb"], ["he", "dnb"])
    drow("e2b", "Âme poutre secondaire – e2,b (vers l'about)", R.e2b, 1.2, R.tw_S, "e", True,
         ["e2b_u"], ["e2b"])
    drow("p1", "Entraxe p1", R.p_1, 2.2, tmin, "p", R.n_1 > 1, ["p1_u"], ["p1"])
    drow("p2", "Entraxe p2", u.p2_u, 2.4, tmin, "p", R.n_2 > 1, ["p2_u"], ["p2"])
    R.dist_ok = all(x.ok for x in R.dist)

    # ---- alertes
    tfrP = R.tf_P + R.r_P; gap = R.h_P - u.d_top - R.h_S; tfrS = R.tf_S + R.r_S
    m12 = 1.2 * d0
    # gorge de pleine résistance vis-à-vis du plat (le plat plastifie avant le cordon : a ≥ tp·fy·βw·γM2/(√2·fu·γM0),
    # la règle des guides — mecanique.a_pleine_resistance_plat) — recommandation, contrôlée par alerte
    R.fuw, R.bww = RS.fu_bw_cordon(R.fu_L, R.bw_L, R.fu_P, R.bw_P)
    R.a_plein = MEC.a_pleine_resistance_plat(R.t_p, R.fy_L, R.bww, R.fuw, gM2, gM0)
    R.rec_a = R.a_w >= R.a_plein - MEC.TOL_GORGE       # §5.3 (6) et Check 8 : cordons de pleine résistance
    R.Lw1 = SCI.leff_cordon(R.h_p, R.a_w)              # longueur efficace d'UN cordon : hp − 2a (EN 1993-1-8 §4.5.1(1))
    botLim = R.h_S - (u.d_nb if u.d_nb > 0 else tfrS); sumC = u.z_C + R.h_p
    prP = ["prof_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"] if u.prof_P == PERSO else ["prof_P"]
    prS = ["prof_S", "hS_u", "tfS_u", "rS_u"] if u.prof_S == PERSO else ["prof_S"]
    AL = [
        dict(id="e1bot", c=R.e1bot < m12, b=1,
             m="Configuration impossible : entraxe vertical / nombre de rangées incompatible avec la hauteur du plat hp (pince basse < 1,2·d0).",
             f=["hp_u", "n1_u", "p1_u", "e1_u"], d=["hp", "e1", "p1", "e1bot"], e=["plate", "bolts"],
             cv=["e1p"] if R.e_1 >= m12 - 0.001 else [],
             w="Pince basse = hp − e1 − (n1 − 1)·p1 = " + f0(R.h_p) + " − " + f0(R.e_1) + " − " + js_str(R.n_1 - 1) + " × " + f0(R.p_1) + " = " + f0(R.e1bot) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. hp minimal avec ces boulons : " + f0(math.ceil(R.e_1 + (R.n_1 - 1) * R.p_1 + m12)) + " mm."),
        dict(id="e2p", c=R.e2_p < m12, b=1,
             m="Configuration impossible : plat trop court – pince e2 au bord libre non conforme à EN 1993-1-8 Tableau 3.3.",
             f=["bp_u", "g_h", "e2b_u", "n2_u", "p2_u"], d=["e2p", "e2b", "gh", "p2", "bp"], e=["plate", "bolts"], cv=["e2p"],
             w="Pince e2 = bp − gh − e2,b − (n2 − 1)·p2 = " + f0(R.b_p) + " − " + f0(u.g_h) + " − " + f0(R.e2b) + " − " + js_str(R.n_2 - 1) + " × " + f0(u.p2_u) + " = " + f0(R.e2_p) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. Allonger le plat ou réduire e2,b / p2."),
        dict(id="zc_top", c=u.z_C < (u.d_nt if u.d_nt > 0 else tfrS), b=1,
             m="Configuration impossible : le dessus du plat est dans la zone grugée ou dans le congé supérieur de la poutre secondaire.",
             f=["z_C", "d_nt"] + prS, d=["zc", "dnt"], e=["plate", "notchT"],
             w=("zc = " + f0(u.z_C) + " mm est inférieur à la profondeur du grugeage dnt = " + f0(u.d_nt) + " mm : il faut zc ≥ dnt.") if u.d_nt > 0
             else ("zc = " + f0(u.z_C) + " mm est inférieur à tf + r = " + f0(tfrS) + " mm (semelle et congé de la poutre secondaire) : il faut zc ≥ " + f0(math.ceil(tfrS)) + " mm.")),
        dict(id="h_dispo", c=sumC > botLim, b=1,
             m="Configuration impossible : hauteur disponible insuffisante – le plat dépasse la partie droite de l'âme de la poutre secondaire.",
             f=["hp_u", "z_C"] + (["d_nb"] if u.d_nb > 0 else []) + prS, d=["hp", "zc"] + (["dnb"] if u.d_nb > 0 else []), e=["plate"],
             w="zc + hp = " + f0(u.z_C) + " + " + f0(R.h_p) + " = " + f0(sumC) + " mm dépasse la limite basse de la partie droite de l'âme : "
             + (("h − dnb = " + f0(R.h_S) + " − " + f0(u.d_nb)) if u.d_nb > 0 else ("h − (tf + r) = " + f0(R.h_S) + " − " + f0(tfrS)))
             + " = " + f0(botLim) + " mm. Avec zc = " + f0(u.z_C) + " mm, hp maximal = " + f0(math.floor(botLim - u.z_C)) + " mm."),
        dict(id="pl_web", c=R.zt_pl < tfrP or R.zt_pl + R.h_p > R.h_P - tfrP, b=1,
             m="Configuration impossible : le plat (et sa soudure) sort de la partie droite de l'âme de la poutre principale.",
             f=["z_C", "d_top", "hp_u"] + prP, d=["ztp", "hp", "zc", "dtop"], e=["plate", "flPt", "flPb"],
             w="Il faut tf + r = " + f0(tfrP) + " mm entre le plat et chaque semelle de la principale. Haut : décalage + zc = " + F(R.zt_pl, 1) + " mm ; bas : h − (décalage + zc + hp) = " + F(R.h_P - R.zt_pl - R.h_p, 1) + " mm."),
        dict(id="dnt_min", c=u.d_top < tfrP and u.d_nt < tfrP - u.d_top, b=1,
             m="Grugeage supérieur insuffisant : la semelle supérieure de la poutre secondaire heurte la semelle ou le congé de la poutre principale (dnt ≥ tf + r − décalage).",
             f=["d_nt", "d_top"] + prP, d=["dnt", "dtop"], e=["notchT", "flPt"],
             w="dnt = " + f0(u.d_nt) + " mm est inférieur à tf + r − décalage = " + f0(tfrP - u.d_top) + " mm (poutre principale)."),
        dict(id="ln_min", c=u.d_nt > 0 and u.l_n < (R.b_P - R.tw_P) / 2 - u.g_h, b=1,
             m="Longueur de grugeage insuffisante pour dégager la demi-semelle de la poutre principale (ln ≥ (b − tw)/2 − gh, + jeu).",
             f=["l_n", "g_h"] + prP, d=["ln", "gh"], e=["notchT", "flPt"],
             w="ln = " + f0(u.l_n) + " mm est inférieur à (b − tw)/2 − gh = " + f0((R.b_P - R.tw_P) / 2 - u.g_h) + " mm."),
        dict(id="gap_neg", c=gap < 0, b=1,
             m="La poutre secondaire descend sous la poutre principale : configuration hors domaine de l'outil.",
             f=prP + prS + ["d_top"], d=["dtop"], e=["beamS", "flPb"],
             w="h principale − décalage − h secondaire = " + f0(gap) + " mm, valeur négative."),
        dict(id="dnb_min", c=gap >= 0 and gap < tfrP and u.d_nb < tfrP - gap, b=1,
             m="Grugeage inférieur nécessaire ou insuffisant : la semelle inférieure de la poutre secondaire heurte la semelle ou le congé inférieur de la poutre principale.",
             f=["d_nb", "d_top", "l_n"] + prS + prP, d=["dnb", "dtop"], e=["notchB", "flPb"],
             w="Espace sous la poutre secondaire = " + f0(gap) + " mm, inférieur à tf + r = " + f0(tfrP) + " mm : il faut dnb ≥ " + f0(math.ceil(tfrP - gap)) + " mm."),
        dict(id="one_bolt", c=R.n_b == 1, b=1,
             m="Configuration impossible : un seul boulon ne peut pas reprendre le moment d'excentricité VEd·z.",
             f=["n1_u", "n2_u"], d=[], e=["bolts"], w="n1 × n2 = 1 boulon : ajouter au moins une rangée ou une file."),
        dict(id="d0_manq", c=u.trou == "Surdimensionné" and N(u.d0_u) == 0, b=1,
             m="Trou surdimensionné : saisir le diamètre d0.", f=["trou", "d0_u"], d=[], e=[], w=""),
        dict(id="w_court", c=R.Lw1 < mx(30, 6 * R.a_w), b=1,
             m="Cordon trop court pour porter un effort : longueur efficace hp − 2a inférieure à max(30 mm ; 6·a) (EN 1993-1-8 §4.5.1(2)).",
             f=["hp_u", "a_w"], d=["hp", "aw"], e=["weld"],
             w="hp − 2a = " + f0(R.h_p) + " − 2 × " + f0(R.a_w) + " = " + F(R.Lw1, 0) + " mm < max(30 ; 6 × " + f0(R.a_w) + ") = " + F(mx(30, 6 * R.a_w), 0) + " mm."),
        *DU.alertes(R, d),                     # ductilité : recommandation, jamais bloquante (à la place de l'ancienne tp_duct)
        dict(id="pince_2d", c=not R.rec_2d, b=0,
             m="Pince inférieure à 2·d : hors du détail éprouvé des plats d'âme (toutes les pinces du plat et de l'âme portée ≥ 2d, capacité de rotation) — articulation à justifier.",
             f=["e1_u", "e2b_u", "bp_u", "hp_u", "boulon_u"], d=["e1", "e2b", "e2p"], e=["plate", "bolts"],
             w=R.pince_2d[0] + " = " + F(R.pince_2d[1], 0) + " mm < 2·d = " + f0(SCI.N_2D * d) + " mm (SCI P358 §5.3 (5))."),
        dict(id="boul_det", c=not R.rec_boul, b=0,
             m="Boulons hors du détail éprouvé des plats d'âme (classe 8.8, non précontraints) : capacité de rotation à justifier.",
             f=["classe", "cat"], d=[], e=["bolts"],
             w="Classe " + js_str(u.classe) + ", catégorie " + js_str(u.cat) + " (SCI P358 §5.3 (4))."),
        dict(id="hp_rec", c=not R.rec_hp, b=0,
             m="hp < 0,6·h de la poutre portée : recommandation des guides (maintien en torsion de la poutre à l'appui) non respectée.",
             f=["hp_u"] + prS, d=["hp"], e=["plate"],
             w="hp = " + f0(R.h_p) + " mm < 0,6·h = " + F(0.6 * R.h_S, 0) + " mm."),
        dict(id="long_nr", c=R.long_p and u.lt_ok != "Oui", b=0,
             m="Plat LONG (zp > tp/0,15) et poutre non maintenue au déversement : le guide n'admet pas ce détail sans justification par essais (SCI P358 Check 3) — à justifier.",
             f=["lt_ok", "tp_u", "e2b_u", "g_h"], d=["z"], e=["plate"],
             w="zp = gh + e2,b = " + F(R.g_B, 1) + " mm > tp/0,15 = " + F(dv(R.t_p, 0.15), 1) + " mm."),
        dict(id="a_rec", c=not R.rec_a, b=0,
             m="Gorge inférieure à la PLEINE RÉSISTANCE vis-à-vis du plat : la vérification en contrainte du cordon décide de la résistance, mais le plat ne plastifie plus avant le cordon — articulation à justifier.",
             f=["a_w", "tp_u"], d=["aw"], e=["weld"],
             w="Détail éprouvé : a ≥ tp·fy·βw·γM2/(√2·fu·γM0) = " + F(R.a_plein, 1) + " mm (SCI P358 Check 8 et §5.3 (6) ; dérivation EN 1993-1-8 §4.5.3.3) ; actuel : a = " + f0(R.a_w) + " mm."),
        dict(id="MEd", c=Mabs != 0, b=0,
             m="MEd ≠ 0 : l'assemblage reste modélisé comme articulé ; MEd est ajouté au moment d'excentricité VEd·z. Un moment significatif sort du domaine de l'outil.",
             f=["M_Ed"], d=["z"], e=[], w="MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm."),
        dict(id="NEd", c=NEd > 0, b=0,
             m="NEd de traction : repris en cisaillement horizontal des boulons et pressions diamétrales ; la flexion hors plan de l'âme porteuse sous NEd n'est PAS vérifiée (à justifier séparément).",
             f=["N_Ed"], d=[], e=[], w=""),
        dict(id="cat_cl", c=u.cat != "A" and R.f_ub < 800, b=1,
             m="Catégorie B ou C : boulons précontraints de classe 8.8 ou 10.9 requis (EN 1993-1-8 §3.4.1).",
             f=["cat", "classe"], d=[], e=[], w=""),
        dict(id="V_inv", c=R.V_inverse, b=0,
             m="VEd saisi négatif (effort de soulèvement) : calculé en valeur absolue — pinces chargées, bloc et appui changent de côté : chemin de l'effort à justifier.",
             f=["V_Ed"], d=[], e=[], w=""),
        dict(id="blf_off", c=u.opt_blf != "Oui" and (R.n_1 - 1) * R.p_1 > 15 * d, b=0,
             m="Assemblage long (Lj > 15·d) : le coefficient βLf de l'EN 1993-1-8 §3.8 n'est PAS appliqué (option « βLf » sur Non, comme les exemples des guides).",
             f=["opt_blf", "n1_u", "p1_u"], d=["p1"], e=["bolts"],
             w="Lj = (n1 − 1)·p1 = " + f0((R.n_1 - 1) * R.p_1) + " mm > 15·d = " + f0(15 * d) + " mm."),
    ]
    R.alerts = [Alerte(id=a["id"], msg=a["m"], block=bool(a["b"]), fields=list(a["f"]),
                       dims=list(a["d"]), elems=list(a["e"]), why=a["w"],
                       covers=list(a.get("cv") or []))
                for a in AL if a["c"]]
    R.geo_ok = not any(a.block for a in R.alerts)
    cov = {}
    for a in R.alerts:
        for x in a.covers:
            cov[x] = a.id
    for x in R.dist:
        x.alert = cov.get(x.id) or "dist_" + x.id
        if x.ok or cov.get(x.id):
            continue
        if x.val < x.min - 0.001:
            why = "Valeur actuelle " + F(x.val, 1) + " mm, inférieure au minimum " + F(x.kmin, 1) + "·d0 = " + F(x.min, 1) + " mm (d0 = " + f0(d0) + " mm)."
        else:
            why = "Valeur actuelle " + F(x.val, 1) + " mm, supérieure au maximum " + F(x.max, 1) + " mm."
        R.alerts.append(Alerte(id="dist_" + x.id,
                               msg="Pince ou entraxe non conforme à EN 1993-1-8 Tableau 3.3 : " + x.lab + ".",
                               block=True, fields=list(x.fields), dims=list(x.dims), elems=[], covers=[], why=why))

    # ---- boulons (1 plan de cisaillement)
    R.Fv_Rd = RS.Fv_Rd(R.a_v, R.f_ub, R.A_cis, gM2)
    R.Fp_C = RS.Fp_C(R.f_ub, R.As_b)
    R.Lj = (R.n_1 - 1) * R.p_1
    R.bLf = RS.beta_Lf(R.Lj, d, u.opt_blf == "Oui")
    R.Fz_b = dv(V, R.n_b) + (dv(R.M_S * 1000 * R.xm, R.Ip) if R.Ip > 0 else 0)
    R.Fx_b = dv(abs(NEd), R.n_b) + (dv(R.M_S * 1000 * R.ym, R.Ip) if R.Ip > 0 else 0)
    R.F_b = hyp(R.Fz_b, R.Fx_b)
    R.tab = bolt_table(R.n_1, R.n_2, R.p_1, u.p2_u, V, NEd, R.M_S, R.Ip)

    def bearing(key, t, fu, e1, p1, n1, e2, p2, n2):
        abv = mn(INF if e1 is None else e1 / (3 * d0), p1 / (3 * d0) - 0.25 if n1 > 1 else 9, dv(R.f_ub, fu), 1)
        k1v = mn(INF if e2 is None else 2.8 * e2 / d0 - 1.7,
                 1.4 * p2 / d0 - 1.7 if n2 > 1 else 9, 2.5)
        abh = mn(INF if e2 is None else e2 / (3 * d0),
                 p2 / (3 * d0) - 0.25 if n2 > 1 else 9, dv(R.f_ub, fu), 1)
        k1h = mn(INF if e1 is None else 2.8 * e1 / d0 - 1.7, 1.4 * p1 / d0 - 1.7 if n1 > 1 else 9, 2.5)
        R["abv_" + key] = abv; R["k1v_" + key] = k1v; R["abh_" + key] = abh; R["k1h_" + key] = k1h
        R["Fbv_" + key] = dv(k1v * abv * fu * d * t, gM2) / 1000 * R.k_trou
        R["Fbh_" + key] = dv(k1h * abh * fu * d * t, gM2) / 1000 * R.k_trou
        R["ipd_" + key] = hyp(dv(R.Fz_b, R["Fbv_" + key]), dv(R.Fx_b, R["Fbh_" + key]))

    bearing("L", R.t_p, R.fu_L, mn(R.e_1, R.e1bot), R.p_1, R.n_1, R.e2_p, u.p2_u, R.n_2)
    bearing("S", R.tw_S, R.fu_S, mn(R.e1b_S, R.h_e) if u.d_nb > 0 else R.e1b_S,
            R.p_1, R.n_1, R.e2b, u.p2_u, R.n_2)

    def bvals(key):
        return ("αb,ver = " + F(R["abv_" + key], 3) + " ; k1,ver = " + F(R["k1v_" + key], 2)
                + " → Fb,ver,Rd = " + F(R["Fbv_" + key], 1) + " kN ; αb,hor = " + F(R["abh_" + key], 3)
                + " ; k1,hor = " + F(R["k1h_" + key], 2) + " → Fb,hor,Rd = " + F(R["Fbh_" + key], 1) + " kN")

    # ---- plat : aires, flexion, déversement
    R.Av_p = R.h_p * R.t_p
    R.Avn_p = (R.h_p - R.n_1 * d0) * R.t_p
    R.Ant_p = R.t_p * (R.e2_p + (R.n_2 - 1) * u.p2_u - (R.n_2 - 0.5) * d0)
    R.Anv_p = R.t_p * (R.h_p - R.e_1 - (R.n_1 - 0.5) * d0)
    R.Wel_p = R.t_p * R.h_p * R.h_p / 6
    R.court = R.h_p >= 2.73 * R.z_p             # flexion non déterminante
    # déversement du plat LONG — SCI P358 Check 3 (retenu le 27/11/2026) : λ̄LT = 2,8·√(zp·hp/(1,5·tp²))/(π·√(E/fy)),
    # χLT de l'expression (6.56) de l'EN 1993-1-1 avec αLT = 0,76, et le 0,6 du diagramme TRIANGULAIRE du moment
    # VEd·z dans le plat — non appliqué si un moment MEd est imposé (le diagramme n'est plus triangulaire). Avant :
    # la courbe BS 5950-1 des profilés laminés (``pb_bs5950``), sans le 0,6, avec le bras z.
    pl_ = SCI.plat_long(R.t_p, R.h_p, R.z_p, R.g_B, R.fy_L, gM0, gM1)
    R.lam_LT = pl_["lam"]; R.lamb_LT = pl_["lam_bar"]; R.chi_LT = pl_["chi"]
    R.k_tri = 0.6 if Mabs == 0 else 1.0
    R.Mb_LT = dv(R.Wel_p * R.chi_LT * R.fy_L, R.k_tri * gM1) / 1e6
    R.Mb_p = mn(R.Mb_LT, dv(R.Wel_p * R.fy_L, gM0) / 1e6)

    # ---- poutre secondaire (section grugée : mêmes règles que le module
    # ---- doubles cornières, MSB Part 5 §4.2.4 / §4.2.5)
    R.cas_g = 0 if (u.d_nt == 0 and u.d_nb == 0) else (2 if (u.d_nt > 0 and u.d_nb > 0) else 1)
    R.Asec_S = 2 * R.b_S * R.tf_S + (R.h_S - 2 * R.tf_S) * R.tw_S + (4 - math.pi) * R.r_S * R.r_S
    R.h_T = R.h_S - u.d_nt - u.d_nb
    R.A_Tee = R.b_S * R.tf_S + R.tw_S * (R.h_T - R.tf_S)
    if R.cas_g == 0:
        R.Av_S = mx(R.Asec_S - 2 * R.b_S * R.tf_S + (R.tw_S + 2 * R.r_S) * R.tf_S, (R.h_S - 2 * R.tf_S) * R.tw_S)
    elif R.cas_g == 1:
        R.Av_S = R.A_Tee - R.b_S * R.tf_S + (R.tw_S + 2 * R.r_S) * R.tf_S / 2
    else:
        R.Av_S = SCI.K_AV * R.tw_S * R.h_T     # section rectangulaire : 0,9 (SCI P358 Check 4)
    R.VRd_gS = dv(R.Av_S * R.fy_S / S3, gM0) / 1000
    R.VRd_nS = dv((R.Av_S - R.n_1 * d0 * R.tw_S) * R.fu_S / S3, gM2n) / 1000
    AhS = R.e2b + (R.n_2 - 1) * u.p2_u - (R.n_2 - 0.5) * d0
    R.Ant_S = R.tw_S * AhS
    # rupture de bloc de l'âme : (n1 − 0,5)·d0 si l'arrachement part d'un bord
    # libre (poutre grugée), (n1 − 1)·d0 sinon — MSB Part 5 §3.2.5.1
    kv = (R.n_1 - 0.5) if u.d_nt > 0 else (R.n_1 - 1)
    R.Anv_S = R.tw_S * (R.e1b_S + (R.n_1 - 1) * R.p_1 - kv * d0) if u.d_nt > 0 else \
        R.tw_S * ((R.n_1 - 1) * R.p_1 + R.e1b_S - kv * d0)
    R.VRd_bS = (dv(0.5 * R.fu_S * R.Ant_S, gM2n) + dv(R.fy_S * R.Anv_S / S3, gM0)) / 1000
    R.hw_T = R.h_T - R.tf_S
    R.zb_T = dv(R.b_S * R.tf_S * R.tf_S / 2 + R.tw_S * R.hw_T * (R.tf_S + R.hw_T / 2), R.A_Tee)
    R.I_N = (R.b_S * R.tf_S ** 3 / 12 + R.b_S * R.tf_S * (R.zb_T - R.tf_S / 2) ** 2
             + R.tw_S * R.hw_T ** 3 / 12 + R.tw_S * R.hw_T * (R.tf_S + R.hw_T / 2 - R.zb_T) ** 2)
    R.W_N = R.tw_S * R.h_T * R.h_T / 6 if R.cas_g == 2 else dv(R.I_N, mx(R.zb_T, R.h_T - R.zb_T))
    R.rho_N = 1 if V <= 0.5 * R.VRd_gS else mx(0, 1 - (2 * dv(V, R.VRd_gS) - 1) ** 2)
    R.Mv_N = dv(R.W_N * R.fy_S, gM0) / 1e6 * R.rho_N
    Vm = mn(R.VRd_gS, R.VRd_bS)
    R.Mc_2 = dv(R.W_N * R.fy_S, gM0) / 1e6 * (1 if V <= 0.5 * Vm else mx(0, 1 - (2 * dv(V, Vm) - 1) ** 2))
    # GRUGEAGE COURT à deux files (SCI P358 Check 5) : si l'extrémité du grugeage est à moins de 2d de la 2e file
    # (xN < 2d), les trous affaiblissent la section du grugeage — le moment est le plus grand de VEd·(gh + ln) et
    # VEd·(gh + e2,b + p2), la résistance celle du Check 4 (Vpl = min(VRd,g ; VRd,n), section NETTE comprise).
    R.xN = u.l_n - (R.e2b + (R.n_2 - 1) * u.p2_u)
    R.court_N = bool(R.cas_g > 0 and R.n_2 == 2 and R.xN < SCI.N_2D * d)
    Vn_ = mn(R.VRd_gS, R.VRd_nS)
    R.rho_Nn = 1 if V <= 0.5 * Vn_ else mx(0, 1 - (2 * dv(V, Vn_) - 1) ** 2)
    R.Mc_N = dv(R.W_N * R.fy_S, gM0) / 1e6 * R.rho_Nn
    R.bras_N = mx(u.g_h + u.l_n, u.g_h + R.e2b + u.p2_u) if R.court_N else u.g_h + u.l_n
    # plat LONG : section ABCD de l'âme portée (SCI P358 Check 4, « resistance at the connection ») — AB et CD
    # cisaillées, BC fléchie ; hb : hauteur de la section (réduite si la poutre est grugée). Avant : absente.
    R.abcd = AttrDict(SCI.abcd_sci(V, R.tw_S, R.fy_S, R.n_1, R.p_1, R.n_2, u.p2_u, R.e2b, R.g_B, R.h_T, gM0))
    # … et la même selon le MSB P5 (§3.2.5.3), qui la demande dès que z > tp/0,15 : INFORMATION — au-delà de
    # 100 % alors que le SCI ne la demande pas (plat « court » pour lui), le statut le dit
    R.abcd_msb = AttrDict(SCI.abcd_msb(V, R.tw_S, R.fy_S, R.fu_S, R.n_1, R.p_1, R.n_2, u.p2_u, R.e2b, R.g_B, d0,
                                       Vn_, gM0, gM2n))
    R.eta_abcd_msb = dv(R.abcd_msb.MEd + Mabs, R.abcd_msb.MRd) if (R.long_msb and R.n_1 > 1) else 0.0
    # domaine de la règle du guide (limites S275 / S355) : lu sur la NUANCE, pas sur la f_y réduite par l'épaisseur
    fyn_S = MT.nominale(R, "S")
    R.el_N = dv(R.h_S, R.tw_S); R.lim_N = 54.3 if fyn_S <= 275 else 48
    R.ln_max = R.h_S if R.el_N <= R.lim_N else dv((160000 if fyn_S <= 275 else 110000) * R.h_S, R.el_N ** 3)
    if R.cas_g == 0:
        R.val_N = "Sans objet (poutre non grugée)"
    elif u.lt_ok != "Oui":
        R.val_N = "NON VALIDE : poutre non maintenue au déversement – stabilité d'ensemble de la poutre grugée à justifier"
    elif fyn_S > 355:
        R.val_N = "NON VALIDE : nuance > S355 – règle non publiée pour cette nuance"
    elif fyn_S < 275:
        R.val_N = "S235 : limites S275 appliquées (côté sécurité) – interprétation"
    else:
        R.val_N = "Valide"

    # ---- poutre principale (cisaillement local, un seul côté chargé)
    R.Av_P = R.tw_P * R.h_p
    # poinçonnement de l'âme porteuse (SCI P358 Check 10, forme rigoureuse) : l'effort par unité de longueur au pied
    # du plat, 6·MS/hp² (+ |NEd|/hp), ne dépasse pas tw·fu/γM2 — écrit en moment : MS + |NEd|·hp/6 ≤ tw·fu·hp²/(6·γM2)
    R.tp_poi, R.M_poi = SCI.poinconnement(R.tw_P, R.fu_P, R.h_p, R.t_p, R.fy_L, gM2n)
    R.MEd_poi = R.M_S + abs(NEd) * R.h_p / 6000

    # ---- soudure (deux cordons verticaux, gorge a chacun, à pleine gorge sur hp − 2a)
    R.qz_w = dv(V * 1000, 2 * R.Lw1)
    R.qx_w = dv(3 * R.M_S * 1e6, R.Lw1 * R.Lw1) + dv(abs(NEd) * 1000, 2 * R.Lw1)
    R.qw = hyp(R.qz_w, R.qx_w)
    R.fvw = RS.fvw_d(R.fuw, R.bww, gM2)
    R.Fw = R.fvw * R.a_w

    # ---- vérifications
    def ck(key, grp, label, Ed, Rd, unit, ref, nat, act, formula, vals="", ess=False):
        c = Verification(key=key, grp=grp, label=label, Ed=Ed, Rd=Rd, unit=unit, ref=ref,
                         nat=nat, active=bool(act), formula=formula, vals=vals or "", ess=bool(ess))
        if c.active:
            c.eta = (Ed / Rd) if Rd > 0 else (INF if Ed > 0 else 0)
            c.ok = bool(c.eta <= 1 + 1e-9)
        R.checks.append(c)
        R.ck[key] = c

    catB = u.cat == "B"; ks = u.k_ser if catB else 1; g3 = u.g_M3s if catB else u.g_M3
    W445 = "EN 1993-1-8 §4.5.3.3 (méthode simplifiée)"
    ck("bv", "Boulons", "Cisaillement des boulons (1 plan), boulon le plus sollicité", R.F_b, R.Fv_Rd * R.bLf, "kN",
       T34 + " ; groupe excentré : " + MSB3, "EC + COMP", True,
       "Fv,Rd = αv·fub·A/γM2 ; F,Ed = √(Fz² + Fx²) avec Fz = VEd/n + MS·xmax/Ip et Fx = |NEd|/n + MS·ymax/Ip",
       "z = " + F(R.z_p, 1) + " mm ; MS = " + F(R.M_S, 2) + " kNm ; Ip = " + F(R.Ip, 0) + " mm² ; Fz = " + F(R.Fz_b, 2) + " kN ; Fx = " + F(R.Fx_b, 2) + " kN ; Fv,Rd = " + F(R.Fv_Rd, 2) + " kN ; βLf = " + F(R.bLf, 3), 1)
    ck("gl", "Boulons", "Glissement (1 surface de frottement)", R.F_b * ks, dv(u.k_s * u.mu_s * R.Fp_C, g3), "kN",
       "EN 1993-1-8 §3.9.1", "EC", u.cat != "A",
       "Fs,Rd = ks·n·μ·Fp,C/γM3 ; Fp,C = 0,7·fub·As", "Fp,C = " + F(R.Fp_C, 1) + " kN ; catégorie " + u.cat)
    ck("pdL", "Plat d'âme", "Pression diamétrale – plat", R.F_b, dv(R.F_b, R.ipd_L) if R.ipd_L > 0 else R.Fbv_L, "kN",
       T34 + " ; interaction : " + MSB3, "EC + COMP", True,
       "Fb,Rd = k1·αb·fu·d·tp/γM2 ; η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²]", bvals("L"), 1)
    ck("pcg", "Plat d'âme", "Cisaillement section brute", V, dv(R.h_p * R.t_p * R.fy_L, 1.27 * S3 * gM0) / 1000, "kN",
       MSB3 + " (d'après ECCS n°126)", "COMP", True, "VRd,g = hp·tp·fy/(1,27·√3·γM0) — 1,27 : interaction flexion",
       "hp = " + F(R.h_p, 0) + " mm ; tp = " + F(R.t_p, 0) + " mm ; fy = " + F(R.fy_L, 0) + " MPa", 1)
    ck("pcn", "Plat d'âme", "Cisaillement section nette", V, dv(R.Avn_p * R.fu_L, S3 * gM2n) / 1000, "kN",
       MSB3, "COMP", True, "VRd,n = Av,net·fu/(√3·γM2) ; Av,net = tp·(hp − n1·d0)",
       "Av,net = " + F(R.Avn_p, 0) + " mm²", 1)
    ck("pcb", "Plat d'âme", "Rupture de bloc", V, (dv(0.5 * R.fu_L * R.Ant_p, gM2n) + dv(R.fy_L * R.Anv_p / S3, gM0)) / 1000, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : " + MSB3, "EC", True,
       "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0)",
       "Ant = " + F(R.Ant_p, 0) + " mm² ; Anv = " + F(R.Anv_p, 0) + " mm²", 1)
    ck("pfl", "Plat d'âme", "Flexion dans son plan", R.M_S, dv(R.Wel_p * R.fy_L, gM0) / 1e6, "kNm",
       "MSB Part 5 §3.2.3 ; EN 1993-1-1 §6.2.5", "COMP", (not R.court) or Mabs > 0,
       "Sans objet si hp ≥ 2,73·z ET MEd = 0 ; sinon VEd·z + |MEd| ≤ Wel·fy/γM0 avec Wel = tp·hp²/6",
       "hp = " + F(R.h_p, 0) + (" ≥" if R.court else " <") + " 2,73·z = " + F(2.73 * R.z_p, 1) + " mm"
       + (" ; |MEd| = " + F(Mabs, 2) + " kNm" if Mabs > 0 else "") + " ; Wel = " + F(R.Wel_p, 0) + " mm³", 1)
    ck("plt", "Plat d'âme", "Déversement du plat LONG (zp > tp/0,15)", R.M_S, R.Mb_p, "kNm",
       "SCI P358 Check 3 ; χLT : EN 1993-1-1 §6.3.2.3 (6.56), αLT = 0,76 ; γM1 ANB",
       "COMP", R.long_p,
       "VEd·z ≤ min(Wel·χLT·fy/(0,6·γM1) ; Wel·fy/γM0) ; λ̄LT = 2,8·√(zp·hp/(1,5·tp²))/(π·√(E/fy)) ; 0,6 : diagramme "
       "triangulaire du moment (1 si MEd ≠ 0)",
       "zp = " + F(R.g_B, 1) + " mm ; λ̄LT = " + F(R.lamb_LT, 3) + " ; χLT = " + F(R.chi_LT, 3) + " ; Wel = " + F(R.Wel_p, 0)
       + " mm³ ; γM1 = " + F(gM1, 2) + " ; coefficient du diagramme = " + F(R.k_tri, 1), 1)
    ck("ptn", "Plat d'âme", "Traction section nette (NEd > 0)", NEd, dv(0.9 * R.Avn_p * R.fu_L, gM2n) / 1000, "kN",
       "EN 1993-1-1 §6.2.3(2)", "EC", NEd > 0, "Nu,Rd = 0,9·Anet·fu/γM2", "")
    ck("pdS", "Poutre secondaire", "Pression diamétrale – âme de la poutre secondaire", R.F_b,
       dv(R.F_b, R.ipd_S) if R.ipd_S > 0 else R.Fbv_S, "kN",
       T34 + " ; interaction : " + MSB3, "EC + COMP", True,
       "Fb,Rd = k1·αb·fu·d·tw/γM2 ; η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²]", bvals("S"), 1)
    ck("vgS", "Poutre secondaire", "Âme – cisaillement section brute (section réduite si grugée)", V, R.VRd_gS, "kN",
       "EN 1993-1-1 §6.2.6 ; MSB Part 5 §3.2.5.1", "EC", True,
       "VRd,g = Av·fy/(√3·γM0)", "Av = " + F(R.Av_S, 0) + " mm²", 1)
    ck("vnS", "Poutre secondaire", "Âme – cisaillement section nette au droit des boulons", V,
       dv((R.Av_S - R.n_1 * d0 * R.tw_S) * R.fu_S / S3, gM2n) / 1000, "kN",
       "MSB Part 5 §3.2.5.1 (ECCS n°126)", "COMP", True,
       "VRd,n = (Av − n1·d0·tw)·fu/(√3·γM2)", "Av,net = " + F(R.Av_S - R.n_1 * d0 * R.tw_S, 0) + " mm²", 1)
    ck("vbS", "Poutre secondaire", "Âme – rupture de bloc", V, R.VRd_bS, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : MSB Part 5 §3.2.5.1", "EC", True,
       "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0) ; Anv avec (n1 − 1)·d0 si non grugée, (n1 − 0,5)·d0 si grugée",
       "Ant = " + F(R.Ant_S, 0) + " mm² ; Anv = " + F(R.Anv_S, 0) + " mm²", 1)
    ck("mN", "Poutre secondaire", "Poutre grugée – flexion + cisaillement à l'extrémité du grugeage",
       V * R.bras_N / 1000 + Mabs, R.Mc_N if R.court_N else R.Mv_N, "kNm",
       "SCI P358 Check 5 ; MSB Part 5 §4.2.4", "COMP", R.cas_g > 0,
       "MEd = VEd·(gh + ln) ; Mv,N,Rd = fy·Wel,N/γM0·[1 − (2VEd/Vpl,N,Rd − 1)²] si VEd > 0,5·Vpl,N,Rd ; deux files et "
       "xN < 2d : MEd = VEd·max(gh + ln ; gh + e2,b + p2) et Vpl,N,Rd = min(VRd,g ; VRd,n)",
       "Wel,N = " + F(R.W_N, 0) + " mm³ ; " + (("xN = " + F(R.xN, 0) + " mm < 2d = " + f0(SCI.N_2D * d) + " mm : bras = "
                                               + F(R.bras_N, 0) + " mm ; Vpl,N,Rd = min(" + F(R.VRd_gS, 1) + " ; "
                                               + F(R.VRd_nS, 1) + ") kN ; réduction = " + F(R.rho_Nn, 3)) if R.court_N else
                                              ("Vpl,N,Rd = " + F(R.VRd_gS, 1) + " kN ; réduction = " + F(R.rho_N, 3))), 1)
    ck("m2", "Poutre secondaire", "Poutre grugée – interaction à la 2e file de boulons (n2 = 2)",
       V * (u.g_h + R.e2b + u.p2_u) / 1000 + Mabs, R.Mc_2, "kNm", "MSB Part 5 §3.2.5", "COMP",
       R.cas_g > 0 and R.n_2 == 2 and u.l_n > R.e2b + u.p2_u, "MEd = VEd·(gh + e2,b + p2) + |MEd|", "")
    ck("stN", "Poutre secondaire", "Stabilité locale du grugeage – longueur ln", u.l_n, R.ln_max, "mm",
       "MSB Part 5 §4.2.5", "COMP", R.cas_g > 0,
       "ln ≤ h si h/tw ≤ limite ; sinon ln ≤ K·h/(h/tw)³",
       "h/tw = " + F(R.el_N, 1) + " ; limite = " + F(R.lim_N, 1) + " ; domaine : " + R.val_N, 1)
    ck("stD", "Poutre secondaire", "Stabilité locale du grugeage – profondeur", mx(u.d_nt, u.d_nb),
       R.h_S / 2 if R.cas_g == 1 else R.h_S / 5, "mm", "MSB Part 5 §4.2.5", "COMP", R.cas_g > 0,
       "dn ≤ h/2 (1 semelle grugée) ; dn ≤ h/5 (2 semelles)", "")
    ck("tnS", "Poutre secondaire", "Âme – traction section nette (NEd > 0)", NEd,
       dv(0.9 * R.tw_S * (R.h_p - R.n_1 * d0) * R.fu_S, gM2n) / 1000, "kN",
       "EN 1993-1-1 §6.2.3(2) ; hauteur participante = hp : hypothèse de l'outil", "INT", NEd > 0,
       "Nu,Rd = 0,9·Anet·fu/γM2", "")
    ck("abS", "Poutre secondaire", "Âme – section ABCD au droit d'un plat LONG", R.abcd.MEd + Mabs, R.abcd.MRd, "kNm",
       "SCI P358 Check 4", "COMP", R.long_p and R.n_1 > 1,
       "VEd·zp (une file) ou VEd·(zp + p2) (deux files) ≤ Mc,BC,Rd + Vpl,AB,Rd·(n1 − 1)·p1 ; Mc,BC,Rd = "
       "fy·tw·((n1 − 1)·p1)²/(6·γM0), réduit si VBC,Ed = VEd·(n1 − 1)·p1/hb > 0,5·Vpl,BC,Rd",
       "bras = " + F(R.abcd.bras, 0) + " mm ; Mc,BC,Rd = " + F(R.abcd.Mc_BC, 2) + " kNm ; Vpl,AB,Rd = " + F(R.abcd.V_AB, 1)
       + " kN ; VBC,Ed = " + F(R.abcd.VBC, 1) + " kN ; Vpl,BC,Rd = " + F(R.abcd.Vpl_BC, 1) + " kN", 1)
    ck("vlP", "Poutre principale", "Cisaillement local de l'âme porteuse (un seul côté chargé)", V / 2,
       dv(R.Av_P * R.fy_P / S3, gM0) / 1000, "kN",
       "SCI P358 Check 10 (âme support chargée d'un seul côté)", "COMP", True,
       "VEd/2 ≤ Av·fy/(√3·γM0) ; Av = tw·hp", "Av = " + F(R.Av_P, 0) + " mm²", 1)
    ck("poi", "Poutre principale", "Poinçonnement de l'âme porteuse par le plat", R.MEd_poi, R.M_poi, "kNm",
       "SCI P358 Check 10 (forme rigoureuse)", "COMP", True,
       "MS + |NEd|·hp/6 ≤ tw·fu·hp²/(6·γM2) — le plat plastifie avant que l'âme porteuse ne poinçonne ; forme "
       "prudente : tp ≤ tw·fu/(fy,p·γM2)",
       "tw = " + F(R.tw_P, 1) + " mm ; fu = " + F(R.fu_P, 0) + " MPa ; hp = " + F(R.h_p, 0) + " mm ; forme prudente : tp ≤ "
       + F(R.tp_poi, 1) + " mm (tp = " + F(R.t_p, 0) + " mm)", 1)
    ck("w", "Soudure", "Cordons plat / âme porteuse – effort par unité de longueur au point critique", R.qw, R.Fw, "N/mm",
       W445, "EC", True,
       "Fw,Ed = √(qz² + qx²) ≤ Fw,Rd = fvw,d·a ; qz = VEd/(2ℓ) ; qx = 3·MS/ℓ² + |NEd|/(2ℓ) ; ℓ = hp − 2a (§4.5.1(1)) ; fvw,d = fu/(√3·βw·γM2)",
       "a = " + F(R.a_w, 0) + " mm ; ℓ = " + F(R.Lw1, 0) + " mm ; qz = " + F(R.qz_w, 0) + " ; qx = " + F(R.qx_w, 0) + " N/mm ; fvw,d = " + F(R.fvw, 1) + " MPa", 1)
    ck("wa", "Soudure", "Gorge minimale", 3, R.a_w, "mm", "EN 1993-1-8 §4.5.2(2)", "EC", True, "a ≥ 3 mm", "")

    # ---- synthèse
    act = [c for c in R.checks if c.active]
    gov = None
    for c in act:
        if gov is None or c.eta > gov.eta:
            gov = c
    R.gov = gov
    R.eta_max = gov.eta if gov else 0
    R.all_ok = all(c.ok for c in act)
    R.verified = bool(R.all_ok and R.geo_ok and R.dist_ok)
    R.reserve = bool(R.cas_g > 0 and R.val_N.startswith("NON VALIDE"))
    # ---- statut (lot de clôture du 27/11/2026) : ce que le calcul ne PROUVE pas se lit dans le verdict — « À
    # VÉRIFIER (raison) », jamais « VÉRIFIÉ » en vert. L'articulation d'un plat d'âme repose sur le détail éprouvé du
    # guide (SCI P358 §5.3 : boulons 8.8 non précontraints, pinces ≥ 2d, cordons de pleine résistance) ; la règle
    # d'épaisseur (§5.3 (3)) reste un AVIS — décision du bureau du 16/11/2026 (``ductilite.py``).
    SCI.poser_statut(R, [
        (not R.rec_2d) and ("articulation à justifier : " + R.pince_2d[0] + " = " + F(R.pince_2d[1], 0) + " mm < 2d = "
                            + f0(SCI.N_2D * d) + " mm"),
        (not R.rec_a) and ("articulation à justifier : gorge de " + f0(R.a_w) + " mm < " + F(R.a_plein, 1)
                           + " mm (pleine résistance)"),
        SCI.donnee((not R.rec_boul) and "articulation à justifier : boulons hors du détail éprouvé (8.8 non précontraints)"),
        (R.long_p and u.lt_ok != "Oui") and "plat long et poutre non maintenue au déversement",
        (R.eta_abcd_msb > 1 + 1e-9 and not (R.long_p and R.n_1 > 1))
        and ("section ABCD de l'âme portée à " + F(100 * R.eta_abcd_msb, 0) + " % selon le MSB P5 (plat court pour le SCI)"),
        SCI.donnee(R.reserve and "poutre grugée non maintenue au déversement"),
        SCI.donnee(R.V_inverse and "effort tranchant de sens inversé"),
        SCI.donnee((NEd > 0) and "âme porteuse sous la traction non vérifiée"),
    ])
    return R


# =========================================================================
#  bolt_table
# =========================================================================
def bolt_table(n1, n2, p1, p2, V, Hh, M, Ip):
    """Efforts par boulon (répartition élastique, inertie polaire)."""
    t = []
    n = n1 * n2
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            x = (j - (n2 + 1) / 2) * p2; y = ((n1 + 1) / 2 - i) * p1
            fx = dv(abs(Hh), n) - (dv(M * 1000 * y, Ip) if Ip > 0 else 0)
            fz = dv(V, n) + (dv(M * 1000 * x, Ip) if Ip > 0 else 0)
            t.append(EffortBoulon(i=i, j=j, x=x, y=y, fx=fx, fz=fz, f=hyp(fx, fz)))
    return t


# =========================================================================
#  predim
# =========================================================================
def predim(u, R, Mabs):
    """Prédimensionnement : pour chaque boulon (plage choisie) et chaque
    nombre de rangées (2 → 8), pinces et entraxes de détail (multiples de 5),
    taux estimés (boulons, pressions diamétrales plat et âme), puis plat
    nécessaire (tp par les résistances, plafonné par la ductilité, SCI P358 §5.3 (3)) et
    gorge de pleine résistance."""
    rows = []
    hav = R.h_S - (u.d_nb if u.d_nb > 0 else R.tf_S + R.r_S) - u.z_C
    V = u.V_Ed
    i1 = idx(DB["boulons"], u.pd_dmin); i2 = idx(DB["boulons"], u.pd_dmax)
    imin = min(i1, i2); imax = max(i1, i2)
    fyn_pl, fyn_S = MT.nominale(R, "L"), MT.nominale(R, "S")   # nuances (règle de ductilité)
    for i, b in enumerate(DB["boulons"]):
        for n in range(2, 9):
          d = b["d"]; d0 = b["d0"]; A = b["As"] if u.filet == "Oui" else b["A"]
          # deux jeux de pinces : « détail » (k·d0 arrondi au 5 supérieur) et
          # « compacte » (au 5 inférieur, plancher normatif du Tableau 3.3) —
          # en cisaillement simple, la variante compacte débloque souvent la
          # hauteur disponible
          # … et jamais sous 2·d : le détail éprouvé de l'articulation (SCI P358 §5.3 (5)) — sinon chaque
          # proposition sortirait « à vérifier »
          e2d = ceil5(SCI.N_2D * d)
          jd = (mx(ceil5(u.k_e1 * d0), e2d), ceil5(u.k_p1 * d0), mx(ceil5(u.k_e2 * d0), e2d))
          jc = (mx(ceil5(1.2 * d0), 5 * math.floor(u.k_e1 * d0 / 5), e2d),
                mx(ceil5(2.2 * d0), 5 * math.floor(u.k_p1 * d0 / 5)),
                mx(ceil5(1.2 * d0), 5 * math.floor(u.k_e2 * d0 / 5), e2d))
          for var, (e1, p1, e2) in (("détail", jd),) + ((("compacte", jc),) if jc != jd else ()):
            hp = 2 * e1 + (n - 1) * p1
            geom = bool(hp <= hav and imin <= i <= imax)
            Fv = dv(R.a_v * R.f_ub * A, u.g_M2) / 1000
            Lj = (n - 1) * p1
            bLf = mx(0.75, 1 - (Lj - 15 * d) / (200 * d)) if (u.opt_blf == "Oui" and Lj > 15 * d) else 1
            z = u.g_h + e2
            Mz = V * z / 1000 + Mabs; Fz = V / n
            Fx = abs(u.N_Ed) / n + dv(Mz * 1000 * 6, p1 * n * (n + 1)); Fs = hyp(Fz, Fx)
            e1b = u.z_C + e1 - u.d_nt
            Fbv = (mn(2.8 * e2 / d0 - 1.7, 2.5) * mn(e1b / (3 * d0), p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            Fbh = (mn(2.8 * e1b / d0 - 1.7, 1.4 * p1 / d0 - 1.7, 2.5) * mn(e2 / (3 * d0), dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            e = AttrDict(bS=dv(Fs, Fv * bLf), pdS=hyp(dv(Fz, Fbv), dv(Fx, Fbh)))
            eta = mx(e.bS, e.pdS); ok = bool(geom and eta <= u.eta_c)
            rows.append(AttrDict(b=b, i=i, n=n, var=var, e1=e1, p1=p1, e2=e2, hp=hp, z=z,
                                 geom=geom, Fz=Fz, Fx=Fx, Mz=Mz, e=e, eta=eta, ok=ok,
                                 # un boulon qui n'admet aucun plat de la liste sous la ductilité passe après les autres
                                 score=(n * 100 + i + 1 + (0 if DU.admet_un_plat(R.tw_S, d, fyn_pl, fyn_S, DB["epais"][0])
                                                           else DU.RANG_NON_DUCTILE)) if ok else 9999,
                                 etag=eta if geom else 99, prop=None))
    found = any(r.ok for r in rows); key = "score" if found else "etag"; pick = rows[0]
    for r in rows:
        if r[key] < pick[key]:
            pick = r
    gM0 = u.g_M0; gM2n = u.g_M2n

    def plate_for(pk):
        d0 = pk.b["d0"]; d = pk.b["d"]; n = pk.n; hp = pk.hp
        fbv = mn(2.8 * pk.e2 / d0 - 1.7, 2.5) * mn(pk.e1 / (3 * d0), pk.p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_L), 1) * R.fu_L * d
        fbv = dv(fbv, u.g_M2) / 1000 * R.k_trou
        fbh = mn(2.8 * pk.e1 / d0 - 1.7, 1.4 * pk.p1 / d0 - 1.7, 2.5) * mn(pk.e2 / (3 * d0), dv(R.f_ub, R.fu_L), 1) * R.fu_L * d
        fbh = dv(fbh, u.g_M2) / 1000 * R.k_trou
        reqs = [["Pression diamétrale du plat", hyp(dv(pk.Fz, fbv), dv(pk.Fx, fbh))],
                ["Cisaillement section brute", dv(V * 1000 * 1.27 * S3 * gM0, hp * R.fy_L)],
                ["Cisaillement section nette", dv(V * 1000 * S3 * gM2n, (hp - n * d0) * R.fu_L)],
                ["Rupture de bloc", dv(V * 1000, dv(0.5 * R.fu_L * (pk.e2 - 0.5 * d0), gM2n)
                                       + dv(R.fy_L * (hp - pk.e1 - (n - 0.5) * d0) / S3, gM0))],
                ["Flexion (si hp < 2,73·z)", 0 if hp >= 2.73 * pk.z else dv(pk.Mz * 1e6 * 6 * gM0, hp * hp * R.fy_L)]]
        treq = dv(mx(*[r[1] for r in reqs]), u.eta_c)
        # ductilité (SCI P358 §5.3 (3), la règle de la vérification) : sans borne si l'âme portée suffit
        # (tw ≤ k·d), sinon tp ≤ k·d du plat — à défaut, l'épaisseur candidate suivante
        tmax = DU.tp_max(R.tw_S, d, fyn_pl, fyn_S)
        cand = [t for t in DB["epais"] if t <= tmax + 1e-9] or [DB["epais"][0]]
        tp = cand[_first_ge(cand, treq)]
        bp = ceil5(u.g_h + pk.e2 * 2)
        fuw, bww = RS.fu_bw_cordon(R.fu_L, R.bw_L, R.fu_P, R.bw_P)
        a = MEC.gorge_proposee(MEC.a_pleine_resistance_plat(tp, R.fy_L, bww, fuw, u.g_M2, gM0))
        return AttrDict(reqs=reqs, treq=treq, tp=tp, bp=bp, a=a, tmax=tmax)

    for r in rows:
        if r.geom:
            r.prop = plate_for(r)
    if not pick.prop:
        pick.prop = plate_for(pick)
    pp = pick.prop
    if not found:
        msg = ("AUCUNE combinaison ne respecte le taux cible dans les limites fixées : la moins "
               "sollicitée est affichée. Élargir la plage de diamètres, la hauteur disponible ou la classe.")
    elif DU.msg_predim(pp.treq, pp.tp, pp.tmax, pick.b["d"]):
        msg = DU.msg_predim(pp.treq, pp.tp, pp.tmax, pick.b["d"])
    elif pp.treq > pp.tp + 1e-9:
        msg = ("Épaisseur nécessaire > " + f0(pp.tp) + " mm (plus grande épaisseur proposée) : augmenter le "
               "nombre de rangées ou le diamètre.")
    else:
        msg = "Proposition complète"
    return AttrDict(rows=rows, hav=hav, found=found, pick=pick, boulon=pick.b["n"], n=pick.n,
                    e1=pick.e1, p1=pick.p1, e2=pick.e2, hp=pick.hp, eta=pick.eta,
                    reqs=pp.reqs, treq=pp.treq, tp=pp.tp, bp=pp.bp, a=pp.a, msg=msg)


# =========================================================================
#  apply_solution / solution_label
# =========================================================================
def apply_solution(u0, r):
    """Recopie une solution du prédimensionnement dans les saisies et
    repasse en mode VÉRIFICATION."""
    u = dict(u0)
    p = r.prop
    u["mode_calc"] = "VÉRIFICATION"; u["boulon_u"] = r.b["n"]; u["n1_u"] = r.n; u["n2_u"] = 1
    u["p1_u"] = r.p1; u["e1_u"] = r.e1; u["e2b_u"] = r.e2; u["hp_u"] = r.hp
    u["tp_u"] = p.tp; u["bp_u"] = p.bp; u["a_w"] = p.a
    return u


def solution_label(r):
    """« M20, 3 rangées, plat 190 × 10 »."""
    return (r.b["n"] + ", " + js_str(r.n) + " rangées, plat " + nb(r.hp) + " × "
            + nb(r.prop.tp))

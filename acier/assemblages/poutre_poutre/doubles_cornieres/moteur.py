# -*- coding: utf-8 -*-
"""Moteur de calcul — assemblage poutre–poutre par doubles cornières d'âme.

Transcription fonction par fonction du moteur de référence (1er ``<script>``
du HTML, objet ``DC``) : ``compute``, ``bolt_table``, ``predim``,
``apply_solution``, ``solution_label``. Les noms de grandeurs sont ceux du
JavaScript — ce sont les clés du corrigé de parité. Aucune formule n'a été
« modernisée » : là où un choix de modèle paraît discutable, il est consigné
dans ``docs/assemblages/ECARTS_ET_CORRECTIONS.md``, pas corrigé ici.

Conventions de transcription (voir ``acier/js.py``) :
- ``dv(a, b)`` remplace ``a / b`` quand le dénominateur peut être nul : le
  moteur ne lève jamais d'exception arithmétique, il propage Infinity / NaN
  comme le JavaScript ;
- ``mn`` / ``mx`` remplacent ``Math.min`` / ``Math.max`` ;
- ``js_round`` remplace ``Math.round`` ; ``F`` et ``f0`` formatent comme le
  moteur (``toFixed``) ;
- les nombres entiers du modèle (nombres de rangées, cotes du
  prédimensionnement) restent des ``int`` : ils sont concaténés tels quels
  dans des chaînes (« n = 3 boulons »).

Unités : mm, mm², mm³, mm⁴, MPa, kN, kNm, N/mm (cordons).
"""
import math

from acier.bibliotheques import DB, PERSO, ORI_P, find, idx
from acier.js import (AttrDict, INF, N, dv, mn, mx, js_round, ceil5, hyp,
                      js_str)
from acier.formats import F, f0
from acier.resultats import Verification, Alerte, Pince, EffortBoulon
from acier import resistances as RS
from .entrees import defaults, NUM, MODE_PREDIM, BOULONNEE

S3 = math.sqrt(3)
T34 = "EN 1993-1-8 Tableau 3.4"
MSB = "MSB Part 5"

GROUPES_VERIF = ("Boulons", "Cornières", "Poutre secondaire",
                 "Poutre principale", "Soudures")


def _first_ge(liste, v):
    """Indice du premier élément ≥ v (nombre d'éléments < v), borné."""
    c = sum(1 for x in liste if x < v)
    return min(c, len(liste) - 1)


# =========================================================================
#  compute
# =========================================================================
def compute(u0=None):
    """Calcul complet pour les saisies ``u0`` (clés absentes → défauts).

    Retourne ``R`` (``AttrDict``) : les scalaires du modèle, ``u`` (saisies
    normalisées), ``checks`` (liste de ``Verification``), ``ck`` (par clé),
    ``alerts`` (``Alerte``), ``dist`` (``Pince``), ``tabS`` / ``tabP``
    (``EffortBoulon``), ``pd`` (prédimensionnement), ``gov`` (vérification
    dimensionnante), ``eta_max``, ``verified``, ``statut``, ``reserve``."""
    u = AttrDict(defaults())
    for k, v in (u0 or {}).items():
        u[k] = v
    for n in NUM:
        u[n] = N(u[n])
    u.V_Ed = abs(u.V_Ed)
    Mabs = abs(u.M_Ed)
    R = AttrDict()
    R.u = u
    R.checks = []
    R.ck = {}
    R.pred = u.mode_calc == MODE_PREDIM
    R.bolt_P = u.fix_P == BOULONNEE
    R.bolt_S = u.fix_S == BOULONNEE
    for X in ("P", "S"):
        if u["prof_" + X] == PERSO:
            p = dict(h=N(u["h" + X + "_u"]), b=N(u["b" + X + "_u"]),
                     tw=N(u["tw" + X + "_u"]), tf=N(u["tf" + X + "_u"]),
                     r=N(u["r" + X + "_u"]))
        else:
            p = find(DB["profils"], u["prof_" + X])
        for d in ("h", "b", "tw", "tf", "r"):
            R[d + "_" + X] = p[d]
    for X in ("P", "S", "C"):
        if u["nu_" + X] == PERSO:
            a = dict(fy=N(u.fy_u), fu=N(u.fu_u), bw=N(u.bw_u))
        else:
            a = find(DB["aciers"], u["nu_" + X])
        R["fy_" + X] = a["fy"]
        R["fu_" + X] = a["fu"]
        R["bw_" + X] = a["bw"]
    cl = find(DB["classes"], u.classe)
    R.f_ub = cl["fub"]
    R.a_v = cl["av"] if u.filet == "Oui" else 0.6
    R.k_trou = 1 if u.trou == "Normal" else 0.8
    pd = R.pd = predim(u, R, Mabs)

    # ---- paramètres retenus
    R.boulon = pd.boulon if R.pred else u.boulon_u

    def ret(a, b):
        return b if R.pred else a

    R.n1_S = ret(u.n1S_u, pd.n); R.n2_S = ret(u.n2S_u, 1); R.p1_S = ret(u.p1S_u, pd.p1)
    R.e1_S = ret(u.e1S_u, pd.e1); R.e2b_S = ret(u.e2b_u, pd.e2)
    R.n1_P = ret(u.n1P_u, pd.n); R.n2_P = ret(u.n2P_u, 1); R.p1_P = ret(u.p1P_u, pd.p1)
    R.e1_P = ret(u.e1P_u, pd.e1); R.g_A = ret(u.gA_u, pd.gA); R.L_C = ret(u.LC_u, pd.LC)
    R.n1_S = max(1, js_round(R.n1_S)); R.n1_P = max(1, js_round(R.n1_P))
    R.n2_S = 2 if N(R.n2_S) >= 2 else 1; R.n2_P = 2 if N(R.n2_P) >= 2 else 1
    if u.corn_u == PERSO:
        kc = dict(a1=mx(N(u.k1_u), N(u.k2_u)), a2=mn(N(u.k1_u), N(u.k2_u)),
                  t=N(u.kt_u), r=N(u.kr_u))
    else:
        kc = find(DB["cornieres"], u.corn_u)
    R.b_A = pd.leg if R.pred else (kc["a1"] if u.orient == ORI_P else kc["a2"])
    R.b_B = pd.leg if R.pred else (kc["a2"] if u.orient == ORI_P else kc["a1"])
    R.t_C = pd.tC if R.pred else kc["t"]
    R.r_C = pd.rC if R.pred else kc["r"]
    if R.pred:
        R.corn_txt = "L" + js_str(pd.leg) + "x" + js_str(pd.leg) + "x" + js_str(pd.tC) + " (proposée)"
    elif u.corn_u == PERSO:
        R.corn_txt = "L" + js_str(kc["a1"]) + "x" + js_str(kc["a2"]) + "x" + js_str(kc["t"])
    else:
        R.corn_txt = u.corn_u
    bo = find(DB["boulons"], R.boulon)
    R.d_b = bo["d"]; R.d_0 = N(u.d0_u) if N(u.d0_u) > 0 else bo["d0"]
    R.As_b = bo["As"]; R.A_cis = bo["As"] if u.filet == "Oui" else bo["A"]
    R.d_m = bo["dm"]; R.d_w = bo["dw"]
    d0 = R.d_0; d = R.d_b; V = u.V_Ed; NEd = u.N_Ed; H = abs(u.H_Ed)
    gM0 = u.g_M0; gM2 = u.g_M2; gM2n = u.g_M2n

    # ---- géométrie
    R.g_B = u.g_h + R.e2b_S; R.z_S = R.g_B + (R.n2_S - 1) * u.p2_S / 2
    R.e2a_S = R.b_B - R.g_B - (R.n2_S - 1) * u.p2_S
    R.e1bot_S = R.L_C - R.e1_S - (R.n1_S - 1) * R.p1_S; R.e1b_S = u.z_C + R.e1_S - u.d_nt
    R.h_e = R.h_S - u.d_nb - (u.z_C + R.e1_S + (R.n1_S - 1) * R.p1_S)
    R.n_S = R.n1_S * R.n2_S
    R.Ip_S = R.n_S * (R.p1_S * R.p1_S * (R.n1_S * R.n1_S - 1) + u.p2_S * u.p2_S * (R.n2_S * R.n2_S - 1)) / 12
    R.xm_S = (R.n2_S - 1) * u.p2_S / 2; R.ym_S = (R.n1_S - 1) * R.p1_S / 2
    R.p_3 = R.tw_S + 2 * R.g_A; R.e2a_P = R.b_A - R.g_A - (R.n2_P - 1) * u.p2_P
    R.e1bot_P = R.L_C - R.e1_P - (R.n1_P - 1) * R.p1_P
    R.zt_P = u.d_top + u.z_C + R.e1_P; R.zb_P = R.h_P - R.zt_P - (R.n1_P - 1) * R.p1_P
    R.n_P = R.n1_P * R.n2_P
    R.Ip_P = R.n_P * (R.p1_P * R.p1_P * (R.n1_P * R.n1_P - 1) + u.p2_P * u.p2_P * (R.n2_P * R.n2_P - 1)) / 12
    R.xm_P = (R.n2_P - 1) * u.p2_P / 2; R.ym_P = (R.n1_P - 1) * R.p1_P / 2
    R.e_P = R.g_A + (R.n2_P - 1) * u.p2_P / 2
    for X in ("S", "P"):
        lh = u["lh_" + X]; L = R.L_C; Lw = L + 2 * lh; xg = dv(lh * lh, Lw)
        R["Lw_" + X] = Lw; R["xg_" + X] = xg
        R["Iw_" + X] = (L * L * L / 12 + 2 * lh * (L / 2) * (L / 2) + L * xg * xg
                        + 2 * (lh * lh * lh / 12 + lh * (lh / 2 - xg) * (lh / 2 - xg)))
    R.zw_S = R.b_B - R.xg_S; R.ew_P = R.b_A - R.xg_P; R.zeff = R.z_S if R.bolt_S else R.zw_S
    R.M_S = V * R.zeff / 1000 + Mabs
    R.M_P = V / 2 * (R.e_P if R.bolt_P else R.ew_P) / 1000 if u.opt_exc == "Oui" else 0
    if not R.bolt_S:
        R.mod_S = "Soudé – groupe de cordons avec moment"
    elif R.M_S == 0:
        R.mod_S = "Distribution uniforme"
    else:
        R.mod_S = "Groupe de boulons avec moment (répartition élastique – inertie polaire)"
    if not R.bolt_P:
        R.mod_P = "Soudé – cisaillement centré" if R.M_P == 0 else "Soudé – groupe de cordons avec moment"
    elif R.M_P == 0:
        R.mod_P = "Distribution uniforme (cisaillement centré, facteur " + F(u.k_rot, 2) + " sur Fv,Rd)"
    else:
        R.mod_P = "Groupe de boulons avec moment (répartition élastique – inertie polaire)"

    # ---- Tableau 3.3
    R.dist = []
    cornF = ["corn_u", "k1_u", "k2_u", "orient"] if u.corn_u == PERSO else ["corn_u", "orient"]

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

    tS = mn(R.t_C, R.tw_S); tP = mn(R.t_C, R.tw_P)
    drow("S_e1c", "S – e1 cornière (haut/bas, valeur mini)", mn(R.e1_S, R.e1bot_S), 1.2, R.t_C, "e", R.bolt_S,
         ["e1S_u", "LC_u", "n1S_u", "p1S_u"], ["e1S", "e1botS"])
    drow("S_e2c", "S – e2 cornière (bord libre de l'aile B)", R.e2a_S, 1.2, R.t_C, "e", R.bolt_S,
         cornF + ["g_h", "e2b_u", "p2_S"], ["e2B"])
    drow("S_e1b", "S – e1,b âme poutre secondaire (vers grugeage sup.)", R.e1b_S, 1.2, R.tw_S, "e",
         R.bolt_S and u.d_nt > 0, ["z_C", "e1S_u", "d_nt"], ["e1b", "zc", "dnt"])
    drow("S_he", "S – he âme poutre secondaire (vers grugeage inf.)", R.h_e, 1.2, R.tw_S, "e",
         R.bolt_S and u.d_nb > 0, ["z_C", "e1S_u", "n1S_u", "p1S_u", "d_nb"], ["he", "dnb"])
    drow("S_e2b", "S – e2,b âme poutre secondaire (vers l'about)", R.e2b_S, 1.2, R.tw_S, "e", R.bolt_S,
         ["e2b_u"], ["e2b"])
    drow("S_p1", "S – p1", R.p1_S, 2.2, tS, "p", R.bolt_S and R.n1_S > 1, ["p1S_u"], ["p1S"])
    drow("S_p2", "S – p2", u.p2_S, 2.4, tS, "p", R.bolt_S and R.n2_S > 1, ["p2_S"], ["p2S"])
    drow("P_e1c", "P – e1 cornière (haut/bas, valeur mini)", mn(R.e1_P, R.e1bot_P), 1.2, R.t_C, "e", R.bolt_P,
         ["e1P_u", "LC_u", "n1P_u", "p1P_u"], ["e1P", "e1botP"])
    drow("P_e2c", "P – e2 cornière (bord libre de l'aile A)", R.e2a_P, 1.2, R.t_C, "e", R.bolt_P,
         cornF + ["gA_u", "p2_P"], ["e2A"])
    drow("P_p1", "P – p1", R.p1_P, 2.2, tP, "p", R.bolt_P and R.n1_P > 1, ["p1P_u"], ["p1P"])
    drow("P_p2", "P – p2", u.p2_P, 2.4, tP, "p", R.bolt_P and R.n2_P > 1, ["p2_P"], ["p2P"])
    R.dist_ok = all(x.ok for x in R.dist)
    R.rec_L = R.L_C >= 0.6 * R.h_S

    # ---- alertes
    tfrP = R.tf_P + R.r_P; gap = R.h_P - u.d_top - R.h_S; tfrS = R.tf_S + R.r_S; m12 = 1.2 * d0
    botLim = R.h_S - (u.d_nb if u.d_nb > 0 else tfrS); sumC = u.z_C + R.L_C
    prP = ["prof_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"] if u.prof_P == PERSO else ["prof_P"]
    prS = ["prof_S", "hS_u", "tfS_u", "rS_u"] if u.prof_S == PERSO else ["prof_S"]
    cA = R.bolt_P and R.g_A - d0 / 2 < R.t_C + R.r_C
    cB = R.bolt_S and R.g_B - d0 / 2 < R.t_C + R.r_C
    AL = [
        dict(id="S_e1bot", c=R.bolt_S and R.e1bot_S < m12, b=1,
             m="Configuration impossible : entraxe vertical / nombre de rangées du groupe S incompatible avec la longueur de cornière Lc (pince basse < 1,2·d0).",
             f=["LC_u", "n1S_u", "p1S_u", "e1S_u"], d=["Lc", "e1S", "p1S", "e1botS"], e=["cleat", "boltsS"],
             cv=["S_e1c"] if R.e1_S >= m12 - 0.001 else [],
             w="Pince basse du groupe S = Lc − e1 − (n1 − 1)·p1 = " + f0(R.L_C) + " − " + f0(R.e1_S) + " − " + js_str(R.n1_S - 1) + " × " + f0(R.p1_S) + " = " + f0(R.e1bot_S) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. Lc minimal avec ces boulons : " + f0(math.ceil(R.e1_S + (R.n1_S - 1) * R.p1_S + m12)) + " mm."),
        dict(id="P_e1bot", c=R.bolt_P and R.e1bot_P < m12, b=1,
             m="Configuration impossible : entraxe vertical / nombre de rangées du groupe P incompatible avec la longueur de cornière Lc (pince basse < 1,2·d0).",
             f=["LC_u", "n1P_u", "p1P_u", "e1P_u"], d=["Lc", "e1P", "p1P", "e1botP"], e=["cleat", "boltsP"],
             cv=["P_e1c"] if R.e1_P >= m12 - 0.001 else [],
             w="Pince basse du groupe P = Lc − e1 − (n1 − 1)·p1 = " + f0(R.L_C) + " − " + f0(R.e1_P) + " − " + js_str(R.n1_P - 1) + " × " + f0(R.p1_P) + " = " + f0(R.e1bot_P) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. Lc minimal avec ces boulons : " + f0(math.ceil(R.e1_P + (R.n1_P - 1) * R.p1_P + m12)) + " mm."),
        dict(id="S_e2", c=R.bolt_S and R.e2a_S < m12, b=1,
             m="Configuration impossible : aile B trop courte – pince e2 du groupe S non conforme à EN 1993-1-8 Tableau 3.3.",
             f=cornF + ["g_h", "e2b_u", "n2S_u", "p2_S"], d=["e2B", "e2b", "gh", "p2S", "bB"], e=["cleat", "boltsS"], cv=["S_e2c"],
             w="Pince e2 = bB − gh − e2,b − (n2 − 1)·p2 = " + f0(R.b_B) + " − " + f0(u.g_h) + " − " + f0(R.e2b_S) + " − " + js_str(R.n2_S - 1) + " × " + f0(u.p2_S) + " = " + f0(R.e2a_S) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. Réduire e2,b ou p2, ou choisir une aile B plus longue."),
        dict(id="P_e2", c=R.bolt_P and R.e2a_P < m12, b=1,
             m="Configuration impossible : aile A trop courte – pince e2 du groupe P non conforme à EN 1993-1-8 Tableau 3.3.",
             f=cornF + ["gA_u", "n2P_u", "p2_P"], d=["e2A", "gA", "p2P", "bA"], e=["cleat", "boltsP"], cv=["P_e2c"],
             w="Pince e2 = bA − gA − (n2 − 1)·p2 = " + f0(R.b_A) + " − " + f0(R.g_A) + " − " + js_str(R.n2_P - 1) + " × " + f0(u.p2_P) + " = " + f0(R.e2a_P) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. Réduire gA ou p2, ou choisir une aile A plus longue."),
        dict(id="zc_top", c=u.z_C < (u.d_nt if u.d_nt > 0 else tfrS), b=1,
             m="Configuration impossible : le dessus des cornières est dans la zone grugée ou dans le congé supérieur de la poutre secondaire.",
             f=["z_C", "d_nt"] + prS, d=["zc", "dnt"], e=["cleat", "notchT"],
             w=("zc = " + f0(u.z_C) + " mm est inférieur à la profondeur du grugeage dnt = " + f0(u.d_nt) + " mm : il faut zc ≥ dnt.") if u.d_nt > 0
             else ("zc = " + f0(u.z_C) + " mm est inférieur à tf + r = " + f0(R.tf_S) + " + " + f0(R.r_S) + " = " + f0(tfrS) + " mm (semelle et congé de la poutre secondaire) : il faut zc ≥ " + f0(math.ceil(tfrS)) + " mm.")),
        dict(id="h_dispo", c=sumC > botLim, b=1,
             m="Configuration impossible : hauteur disponible insuffisante – les cornières dépassent la partie droite de l'âme de la poutre secondaire.",
             f=["LC_u", "z_C"] + (["d_nb"] if u.d_nb > 0 else []) + prS, d=["Lc", "zc"] + (["dnb"] if u.d_nb > 0 else []), e=["cleat"],
             w="zc + Lc = " + f0(u.z_C) + " + " + f0(R.L_C) + " = " + f0(sumC) + " mm dépasse la limite basse de la partie droite de l'âme : "
             + (("h − dnb = " + f0(R.h_S) + " − " + f0(u.d_nb)) if u.d_nb > 0 else ("h − (tf + r) = " + f0(R.h_S) + " − " + f0(tfrS)))
             + " = " + f0(botLim) + " mm. Dépassement : " + f0(sumC - botLim) + " mm. Avec zc = " + f0(u.z_C) + " mm, Lc maximal = " + f0(math.floor(botLim - u.z_C)) + " mm."),
        dict(id="dnt_min", c=u.d_top < tfrP and u.d_nt < tfrP - u.d_top, b=1,
             m="Grugeage supérieur insuffisant : la semelle supérieure de la poutre secondaire heurte la semelle ou le congé de la poutre principale (dnt ≥ tf + r − décalage).",
             f=["d_nt", "d_top"] + prP, d=["dnt", "dtop"], e=["notchT", "flPt"],
             w="dnt = " + f0(u.d_nt) + " mm est inférieur à tf + r − décalage = " + f0(R.tf_P) + " + " + f0(R.r_P) + " − " + f0(u.d_top) + " = " + f0(tfrP - u.d_top) + " mm (semelle et congé de la poutre principale)."),
        dict(id="ln_min", c=u.d_nt > 0 and u.l_n < (R.b_P - R.tw_P) / 2 - u.g_h, b=1,
             m="Longueur de grugeage insuffisante pour dégager la demi-semelle de la poutre principale (ln ≥ (b − tw)/2 − gh, + jeu).",
             f=["l_n", "g_h"] + prP, d=["ln", "gh"], e=["notchT", "flPt"],
             w="ln = " + f0(u.l_n) + " mm est inférieur à (b − tw)/2 − gh = (" + f0(R.b_P) + " − " + f0(R.tw_P) + ")/2 − " + f0(u.g_h) + " = " + f0((R.b_P - R.tw_P) / 2 - u.g_h) + " mm : la demi-semelle de la poutre principale n'est pas dégagée."),
        dict(id="gap_neg", c=gap < 0, b=1,
             m="La poutre secondaire descend sous la poutre principale : configuration hors domaine de l'outil.",
             f=prP + prS + ["d_top"], d=["dtop"], e=["beamS", "flPb"],
             w="h principale − décalage − h secondaire = " + f0(R.h_P) + " − " + f0(u.d_top) + " − " + f0(R.h_S) + " = " + f0(gap) + " mm, valeur négative."),
        dict(id="dnb_min", c=gap >= 0 and gap < tfrP and u.d_nb < tfrP - gap, b=1,
             m="Grugeage inférieur nécessaire ou insuffisant : la semelle inférieure de la poutre secondaire heurte la semelle ou le congé inférieur de la poutre principale.",
             f=["d_nb", "d_top", "l_n"] + prS + prP, d=["dnb", "dtop"], e=["notchB", "flPb"],
             w="Espace sous la poutre secondaire = h principale − décalage − h secondaire = " + f0(gap) + " mm, inférieur à tf + r = " + f0(tfrP) + " mm de la poutre principale : il faut dnb ≥ " + f0(math.ceil(tfrP - gap)) + " mm (actuel : " + f0(u.d_nb) + " mm)."),
        dict(id="P_web", c=R.bolt_P and (R.zt_P - d0 / 2 < tfrP or R.zb_P - d0 / 2 < tfrP), b=1,
             m="Configuration impossible : boulons du groupe P en dehors de la partie droite de l'âme de la poutre principale.",
             f=["d_top", "z_C", "e1P_u", "n1P_u", "p1P_u"] + prP, d=["ztP", "e1P", "p1P", "zc", "dtop"], e=["boltsP", "flPt", "flPb"],
             w="Il faut au moins tf + r = " + f0(tfrP) + " mm entre le bord du trou et chaque face extérieure de la poutre principale. Rangée haute : zt − d0/2 = " + F(R.zt_P - d0 / 2, 1) + " mm (zt = décalage + zc + e1) ; rangée basse : " + F(R.zb_P - d0 / 2, 1) + " mm."),
        dict(id="fillet", c=cA or cB, b=1,
             m="Configuration impossible : trou de boulon dans le congé de la cornière (trusquinage − d0/2 < tc + rc).",
             f=(["gA_u"] if cA else []) + (["g_h", "e2b_u"] if cB else []) + cornF + ["boulon_u"],
             d=(["gA"] if cA else []) + (["e2b", "gh"] if cB else []),
             e=["cleat"] + (["boltsP"] if cA else []) + (["boltsS"] if cB else []),
             w="Il faut trusquinage − d0/2 ≥ tc + rc = " + f0(R.t_C) + " + " + f0(R.r_C) + " = " + f0(R.t_C + R.r_C) + " mm."
             + ((" Aile A : gA − d0/2 = " + F(R.g_A - d0 / 2, 1) + " mm, donc gA ≥ " + f0(math.ceil(R.t_C + R.r_C + d0 / 2)) + " mm.") if cA else "")
             + ((" Aile B : gh + e2,b − d0/2 = " + F(R.g_B - d0 / 2, 1) + " mm, donc e2,b ≥ " + f0(math.ceil(R.t_C + R.r_C + d0 / 2 - u.g_h)) + " mm.") if cB else "")),
        dict(id="one_bolt", c=R.bolt_S and R.n_S == 1, b=1,
             m="Configuration impossible : un seul boulon côté poutre secondaire ne peut pas reprendre le moment d'excentricité VEd·z.",
             f=["n1S_u", "n2S_u"], d=[], e=["boltsS"], w="n1 × n2 = 1 boulon : ajouter au moins une rangée ou une file."),
        dict(id="d0_manq", c=u.trou == "Surdimensionné" and N(u.d0_u) == 0, b=1,
             m="Trou surdimensionné : saisir le diamètre d0.", f=["trou", "d0_u"], d=[], e=[], w=""),
        dict(id="MEd", c=Mabs != 0, b=0,
             m="MEd ≠ 0 : l'assemblage reste modélisé comme articulé ; MEd est traité comme moment parasite (groupe S : ajouté à VEd·z ; groupe P : traction élastique). Un moment significatif sort du domaine de l'outil.",
             f=["M_Ed"], d=["z"], e=[], w="MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm."),
        dict(id="NEd", c=NEd > 0, b=0,
             m="NEd de traction : la flexion hors plan de l'âme de la poutre principale n'est PAS vérifiée par l'outil (à justifier séparément).",
             f=["N_Ed"], d=[], e=[], w=""),
        dict(id="HEd", c=H > 0.1 * V and H > 0, b=0,
             m="HEd > 10 % de VEd : la flexion hors plan des cornières sous HEd n'est pas couverte par le modèle – étude spécifique à prévoir.",
             f=["H_Ed", "V_Ed"], d=[], e=[], w="HEd = " + F(H, 1) + " kN pour VEd = " + F(V, 1) + " kN."),
        dict(id="soude2", c=not R.bolt_P and not R.bolt_S, b=0,
             m="Cornières soudées des deux côtés : capacité de rotation réduite, à justifier vis-à-vis de l'hypothèse d'articulation (EN 1993-1-8 §5.2).",
             f=["fix_P", "fix_S"], d=[], e=[], w=""),
        dict(id="cat_cl", c=u.cat != "A" and R.f_ub < 800, b=1,
             m="Catégorie B ou C : boulons précontraints de classe 8.8 ou 10.9 requis (EN 1993-1-8 §3.4.1).",
             f=["cat", "classe"], d=[], e=[], w=""),
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

    # ---- boulons
    R.Fv_Rd = RS.Fv_Rd(R.a_v, R.f_ub, R.A_cis, gM2); R.Ft_Rd = RS.Ft_Rd(R.f_ub, R.As_b, gM2)
    R.Bp_Rd = RS.Bp_Rd(R.d_m, mn(R.t_C * R.fu_C, R.tw_P * R.fu_P), gM2); R.Fp_C = RS.Fp_C(R.f_ub, R.As_b)
    for X in ("S", "P"):
        Lj = (R["n1_" + X] - 1) * R["p1_" + X]
        R["Lj_" + X] = Lj
        R["bLf_" + X] = RS.beta_Lf(Lj, d, u.opt_blf == "Oui")
    R.Fz_S = dv(V, R.n_S) + (dv(R.M_S * 1000 * R.xm_S, R.Ip_S) if R.Ip_S > 0 else 0)
    R.Fx_S = dv(abs(NEd), R.n_S) + (dv(R.M_S * 1000 * R.ym_S, R.Ip_S) if R.Ip_S > 0 else 0)
    R.F_S = hyp(R.Fz_S, R.Fx_S)
    R.Fz_P = dv(V / 2, R.n_P) + (dv(R.M_P * 1000 * R.xm_P, R.Ip_P) if R.Ip_P > 0 else 0)
    R.Fx_P = dv(H / 2, R.n_P) + (dv(R.M_P * 1000 * R.ym_P, R.Ip_P) if R.Ip_P > 0 else 0)
    R.F_P = hyp(R.Fz_P, R.Fx_P)
    R.Ft_P = (dv(mx(NEd, 0), 2 * R.n_P)
              + (dv(Mabs * 1000 * R.ym_P, 2 * R.n2_P * R.p1_P * R.p1_P * R.n1_P * (R.n1_P * R.n1_P - 1) / 12) if R.n1_P > 1 else 0)
              + dv(H * R.zeff, R.p_3 * R.n_P))
    R.tabS = bolt_table(R.n1_S, R.n2_S, R.p1_S, u.p2_S, V, NEd, R.M_S, R.Ip_S)
    R.tabP = bolt_table(R.n1_P, R.n2_P, R.p1_P, u.p2_P, V / 2, H / 2, R.M_P, R.Ip_P)

    def bearing(key, t, fu, e1, p1, n1, e2, p2, n2, Fz, Fx, div, exV=None, exH=None):
        abv = mn(INF if e1 is None else e1 / (3 * d0), p1 / (3 * d0) - 0.25 if n1 > 1 else 9, dv(R.f_ub, fu), 1)
        k1v = mn(INF if e2 is None else 2.8 * e2 / d0 - 1.7, INF if exV is None else exV,
                 1.4 * p2 / d0 - 1.7 if n2 > 1 else 9, 2.5)
        abh = mn(INF if e2 is None else e2 / (3 * d0), INF if exH is None else exH,
                 p2 / (3 * d0) - 0.25 if n2 > 1 else 9, dv(R.f_ub, fu), 1)
        k1h = mn(INF if e1 is None else 2.8 * e1 / d0 - 1.7, 1.4 * p1 / d0 - 1.7 if n1 > 1 else 9, 2.5)
        R["abv_" + key] = abv; R["k1v_" + key] = k1v; R["abh_" + key] = abh; R["k1h_" + key] = k1h
        R["Fbv_" + key] = dv(k1v * abv * fu * d * t, gM2) / 1000 * R.k_trou
        R["Fbh_" + key] = dv(k1h * abh * fu * d * t, gM2) / 1000 * R.k_trou
        R["ipd_" + key] = hyp(dv(Fz / div, R["Fbv_" + key]), dv(Fx / div, R["Fbh_" + key]))

    bearing("B", R.t_C, R.fu_C, mn(R.e1_S, R.e1bot_S), R.p1_S, R.n1_S, R.e2a_S, u.p2_S, R.n2_S, R.Fz_S, R.Fx_S, 2)
    bearing("A", R.t_C, R.fu_C, mn(R.e1_P, R.e1bot_P), R.p1_P, R.n1_P, R.e2a_P, u.p2_P, R.n2_P, R.Fz_P, R.Fx_P, 1)
    bearing("S", R.tw_S, R.fu_S, mn(R.e1b_S, R.h_e) if u.d_nb > 0 else R.e1b_S, R.p1_S, R.n1_S, R.e2b_S, u.p2_S, R.n2_S, R.Fz_S, R.Fx_S, 1)
    bearing("P", R.tw_P, R.fu_P, None, R.p1_P, R.n1_P, None, u.p2_P, R.n2_P, R.Fz_P, R.Fx_P, 1,
            1.4 * R.p_3 / d0 - 1.7, R.p_3 / (3 * d0) - 0.25)

    def bvals(key):
        return ("αb,ver = " + F(R["abv_" + key], 3) + " ; k1,ver = " + F(R["k1v_" + key], 2)
                + " → Fb,ver,Rd = " + F(R["Fbv_" + key], 1) + " kN ; αb,hor = " + F(R["abh_" + key], 3)
                + " ; k1,hor = " + F(R["k1h_" + key], 2) + " → Fb,hor,Rd = " + F(R["Fbh_" + key], 1) + " kN")

    # ---- aires et résistances cornières
    R.Ant_B = R.t_C * (R.e2a_S + (R.n2_S - 1) * u.p2_S - (R.n2_S - 0.5) * d0)
    R.Anv_B = R.t_C * (R.L_C - R.e1_S - (R.n1_S - 0.5) * d0)
    R.Ant_A = R.t_C * (R.e2a_P + (R.n2_P - 1) * u.p2_P - (R.n2_P - 0.5) * d0)
    R.Anv_A = R.t_C * (R.L_C - R.e1_P - (R.n1_P - 0.5) * d0)
    R.e1A_T = mn(R.e1_P, R.e1bot_P, 0.5 * (R.p_3 - R.tw_S - 2 * R.r_C) + d0 / 2)
    R.p1A_T = mn(R.p1_P, R.p_3 - R.tw_S - 2 * R.r_C + d0)
    R.leff_T = 2 * R.e1A_T + (R.n1_P - 1) * R.p1A_T
    R.m_T = (R.p_3 - R.tw_S - 2 * R.t_C - 1.6 * R.r_C) / 2
    R.n_T = mn(R.e2a_P, 1.25 * R.m_T); R.ew_T = R.d_w / 4
    R.Mpl_T = dv(0.25 * R.leff_T * R.t_C * R.t_C * R.fy_C, gM0) / 1e6; R.SFt_T = 2 * R.n1_P * R.Ft_Rd
    R.FT_1 = dv((8 * R.n_T - 2 * R.ew_T) * R.Mpl_T * 1000, 2 * R.m_T * R.n_T - R.ew_T * (R.m_T + R.n_T))
    R.FT_2 = dv(2 * R.Mpl_T * 1000 + R.n_T * R.SFt_T, R.m_T + R.n_T); R.FT_3 = R.SFt_T
    eB = mn(R.e1_S, R.e1bot_S); AhB = R.e2a_S + (R.n2_S - 1) * u.p2_S - (R.n2_S - 0.5) * d0
    R.Vt1_B = (dv(R.fu_C * 2 * R.t_C * (R.n1_S - 1) * (R.p1_S - d0), gM2n) + dv(R.fy_C * 4 * R.t_C * AhB / S3, gM0)) / 1000
    R.Vt2_B = (dv(R.fu_C * 2 * R.t_C * (eB + (R.n1_S - 1) * R.p1_S - (R.n1_S - 0.5) * d0), gM2n) + dv(R.fy_C * 2 * R.t_C * AhB / S3, gM0)) / 1000

    # ---- poutre secondaire
    R.cas_g = 0 if (u.d_nt == 0 and u.d_nb == 0) else (2 if (u.d_nt > 0 and u.d_nb > 0) else 1)
    R.Asec_S = 2 * R.b_S * R.tf_S + (R.h_S - 2 * R.tf_S) * R.tw_S + (4 - math.pi) * R.r_S * R.r_S
    R.h_T = R.h_S - u.d_nt - u.d_nb
    R.A_Tee = R.b_S * R.tf_S + R.tw_S * (R.h_T - R.tf_S)
    if R.cas_g == 0:
        R.Av_S = mx(R.Asec_S - 2 * R.b_S * R.tf_S + (R.tw_S + 2 * R.r_S) * R.tf_S, (R.h_S - 2 * R.tf_S) * R.tw_S)
    elif R.cas_g == 1:
        R.Av_S = R.A_Tee - R.b_S * R.tf_S + (R.tw_S + 2 * R.r_S) * R.tf_S / 2
    else:
        R.Av_S = R.tw_S * R.h_T
    R.VRd_gS = dv(R.Av_S * R.fy_S / S3, gM0) / 1000
    AhS = R.e2b_S + (R.n2_S - 1) * u.p2_S - (R.n2_S - 0.5) * d0
    R.Ant_S = R.tw_S * AhS; R.Anv_S = R.tw_S * (R.e1b_S + (R.n1_S - 1) * R.p1_S - (R.n1_S - 0.5) * d0)
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
    R.el_N = dv(R.h_S, R.tw_S); R.lim_N = 54.3 if R.fy_S <= 275 else 48
    R.ln_max = R.h_S if R.el_N <= R.lim_N else dv((160000 if R.fy_S <= 275 else 110000) * R.h_S, R.el_N ** 3)
    if R.cas_g == 0:
        R.val_N = "Sans objet (poutre non grugée)"
    elif u.lt_ok != "Oui":
        R.val_N = "NON VALIDE : poutre non maintenue au déversement – stabilité d'ensemble de la poutre grugée à justifier (SCI P358 Check 7)"
    elif R.fy_S > 355:
        R.val_N = "NON VALIDE : nuance > S355 – règle non publiée pour cette nuance"
    elif R.fy_S < 275:
        R.val_N = "S235 : limites S275 appliquées (côté sécurité) – interprétation"
    else:
        R.val_N = "Valide"
    R.Vt1_S = (dv(R.fu_S * R.tw_S * (R.n1_S - 1) * (R.p1_S - d0), gM2n) + dv(R.fy_S * 2 * R.tw_S * AhS / S3, gM0)) / 1000
    R.Vt2_S = (dv(R.fu_S * R.tw_S * (R.e1b_S + (R.n1_S - 1) * R.p1_S - (R.n1_S - 0.5) * d0), gM2n) + dv(R.fy_S * R.tw_S * AhS / S3, gM0)) / 1000

    # ---- poutre principale
    R.et_P = mn(R.zt_P, 5 * d) if R.bolt_P else 0
    R.eb_P = mn(R.zb_P, R.p_3 / 2, 5 * d) if R.bolt_P else 0
    R.Av_P = R.tw_P * (R.et_P + (R.n1_P - 1) * R.p1_P + R.eb_P) if R.bolt_P else R.tw_P * R.L_C

    # ---- soudures
    for X, w1, w2, w3, w4 in (("S", abs(NEd) / 2, 0, R.M_S / 2, "S"), ("P", H / 2, mx(NEd, 0) / 2, R.M_P, "P")):
        Lw = R["Lw_" + X]; Iw = R["Iw_" + X]; lh = u["lh_" + X]; xg = R["xg_" + X]
        R["qz_" + X] = dv(V / 2 * 1000, Lw) + dv(w3 * 1e6 * (lh - xg), Iw)
        R["qx_" + X] = dv(w1 * 1000, Lw) + dv(w3 * 1e6 * (R.L_C / 2), Iw)
        R["qn_" + X] = dv(w2 * 1000, Lw)
        R["qw_" + X] = hyp(R["qz_" + X], R["qx_" + X], R["qn_" + X])
        fuo = R["fu_" + w4]
        R["fuw_" + X], R["bww_" + X] = RS.fu_bw_cordon(R.fu_C, R.bw_C, fuo, R["bw_" + w4])
        R["fvw_" + X] = RS.fvw_d(R["fuw_" + X], R["bww_" + X], gM2)
        R["Fw_" + X] = R["fvw_" + X] * u["a_" + X]

    # ---- vérifications
    def ck(key, grp, label, Ed, Rd, unit, ref, nat, act, formula, vals="", ess=False):
        c = Verification(key=key, grp=grp, label=label, Ed=Ed, Rd=Rd, unit=unit, ref=ref,
                         nat=nat, active=bool(act), formula=formula, vals=vals or "", ess=bool(ess))
        if c.active:
            c.eta = (Ed / Rd) if Rd > 0 else (INF if Ed > 0 else 0)
            c.ok = bool(c.eta <= 1 + 1e-9)
        R.checks.append(c)
        R.ck[key] = c

    shr = MSB + " §4.2.2 (d'après ECCS n°126)"; catB = u.cat == "B"
    ks = u.k_ser if catB else 1; g3 = u.g_M3s if catB else u.g_M3
    P10 = "SCI P358 Check 10 (un seul côté chargé)"
    W = "EN 1993-1-8 §4.5.3.3 ; répartition élastique du groupe de cordons : modèle complémentaire"
    ck("bv_S", "Boulons", "Cisaillement des boulons – groupe S (2 plans), boulon le plus sollicité", R.F_S, 2 * R.Fv_Rd * R.bLf_S, "kN",
       T34 + " ; groupe excentré : " + MSB + " §4.2.1.1", "EC + COMP", R.bolt_S,
       "Fv,Rd = αv·fub·A/γM2 (× 2 plans) ; F,Ed = √(Fz² + Fx²) avec Fz = VEd/n + MS·xmax/Ip et Fx = |NEd|/n + MS·ymax/Ip",
       "z = " + F(R.z_S, 1) + " mm ; MS = " + F(R.M_S, 2) + " kNm ; Ip = " + F(R.Ip_S, 0) + " mm² ; Fz = " + F(R.Fz_S, 2) + " kN ; Fx = " + F(R.Fx_S, 2) + " kN ; Fv,Rd = " + F(R.Fv_Rd, 2) + " kN/plan ; βLf = " + F(R.bLf_S, 3), 1)
    ck("bv_P", "Boulons", "Cisaillement des boulons – groupe P (1 plan)", R.F_P, u.k_rot * R.Fv_Rd * R.bLf_P, "kN",
       T34 + " ; facteur " + F(u.k_rot, 2) + " : " + MSB + " §4.2.1.2", "EC + COMP", R.bolt_P,
       "F,Ed = (VEd/2)/n par boulon ; FRd = k_rot·Fv,Rd (traction parasite due à la rotation)",
       "n = " + js_str(R.n_P) + " boulons par cornière ; Fz = " + F(R.Fz_P, 2) + " kN ; Fx = " + F(R.Fx_P, 2) + " kN ; Fv,Rd = " + F(R.Fv_Rd, 2) + " kN", 1)
    ck("bt_P", "Boulons", "Traction des boulons – groupe P", R.Ft_P, mn(R.Ft_Rd, R.Bp_Rd), "kN", T34, "EC", R.bolt_P and R.Ft_P > 0,
       "Ft,Rd = 0,9·fub·As/γM2 ; Bp,Rd = 0,6·π·dm·tp·fu/γM2",
       "Ft,Ed = NEd/(2n) + MEd·ymax/Σy² + HEd·z/(p3·n) = " + F(R.Ft_P, 2) + " kN ; Ft,Rd = " + F(R.Ft_Rd, 1) + " kN ; Bp,Rd = " + F(R.Bp_Rd, 1) + " kN")
    ck("bi_P", "Boulons", "Interaction cisaillement + traction – groupe P", dv(R.F_P, R.Fv_Rd * R.bLf_P) + dv(R.Ft_P, 1.4 * R.Ft_Rd), 1, "-", T34, "EC",
       R.bolt_P and R.Ft_P > 0, "Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1", "Fv,Ed = " + F(R.F_P, 2) + " kN ; Ft,Ed = " + F(R.Ft_P, 2) + " kN")
    ck("gl_S", "Boulons", "Glissement – groupe S (2 surfaces de frottement)", R.F_S * ks, dv(u.k_s * 2 * u.mu_s * R.Fp_C, g3), "kN", "EN 1993-1-8 §3.9.1", "EC",
       R.bolt_S and u.cat != "A", "Fs,Rd = ks·n·μ·Fp,C/γM3 ; Fp,C = 0,7·fub·As", "Fp,C = " + F(R.Fp_C, 1) + " kN ; catégorie " + u.cat)
    ck("gl_P", "Boulons", "Glissement – groupe P (1 surface de frottement)", R.F_P * ks, dv(u.k_s * u.mu_s * (R.Fp_C - 0.8 * R.Ft_P * ks), g3), "kN",
       "EN 1993-1-8 §3.9.1 et §3.9.2", "EC", R.bolt_P and u.cat != "A", "Fs,Rd = ks·n·μ·(Fp,C − 0,8·Ft,Ed)/γM3",
       "Fp,C = " + F(R.Fp_C, 1) + " kN ; Ft,Ed = " + F(R.Ft_P, 2) + " kN")
    ck("pdB", "Cornières", "Pression diamétrale – ailes B des cornières (groupe S)", R.F_S / 2, dv(R.F_S / 2, R.ipd_B) if R.ipd_B > 0 else R.Fbv_B, "kN",
       T34 + " ; interaction : " + MSB + " §4.2.1.1", "EC + COMP", R.bolt_S,
       "Fb,Rd = k1·αb·fu·d·t/γM2 ; η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²] ; effort par cornière = F/2", bvals("B"), 1)
    ck("pdA", "Cornières", "Pression diamétrale – ailes A des cornières (groupe P)", R.F_P, dv(R.F_P, R.ipd_A) if R.ipd_A > 0 else R.Fbv_A, "kN", T34,
       "EC + COMP", R.bolt_P, "Fb,Rd = k1·αb·fu·d·t/γM2", bvals("A"), 1)
    VgC = dv(2 * R.L_C * R.t_C * R.fy_C, 1.27 * S3 * gM0) / 1000
    WC = dv(R.t_C * R.L_C * R.L_C / 6 * R.fy_C, gM0) / 1e6
    ck("cgB", "Cornières", "Ailes B – cisaillement section brute", V, VgC, "kN", shr, "COMP", True, "VRd,g = 2·Lc·tc·fy/(1,27·√3·γM0)",
       "Lc = " + F(R.L_C, 0) + " mm ; tc = " + F(R.t_C, 0) + " mm ; fy = " + F(R.fy_C, 0) + " MPa", 1)
    ck("cnB", "Cornières", "Ailes B – cisaillement section nette", V, dv(2 * R.t_C * (R.L_C - R.n1_S * d0) * R.fu_C, S3 * gM2n) / 1000, "kN", shr, "COMP", R.bolt_S,
       "VRd,n = 2·Av,net·fu/(√3·γM2) ; Av,net = tc·(Lc − n1·d0)", "Av,net = " + F(R.t_C * (R.L_C - R.n1_S * d0), 0) + " mm² par cornière", 1)
    ck("cbB", "Cornières", "Ailes B – rupture de bloc", V, 2 * (dv(0.5 * R.fu_C * R.Ant_B, gM2n) + dv(R.fy_C * R.Anv_B / S3, gM0)) / 1000, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : " + MSB + " §4.2.2.1", "EC", R.bolt_S, "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0) (× 2 cornières)",
       "Ant = " + F(R.Ant_B, 0) + " mm² ; Anv = " + F(R.Anv_B, 0) + " mm²", 1)
    ck("flB", "Cornières", "Ailes B – flexion dans leur plan", V / 2 * R.zeff / 1000 + Mabs / 2, WC, "kNm", MSB + " §3.2.3 (plat d'âme) ; EN 1993-1-1 §6.2.5", "COMP", True,
       "MEd = (VEd/2)·z ; MRd = Wel·fy/γM0 avec Wel = tc·Lc²/6", "z = " + F(R.zeff, 1) + " mm ; Wel = " + F(R.t_C * R.L_C * R.L_C / 6, 0) + " mm³", 1)
    ck("cgA", "Cornières", "Ailes A – cisaillement section brute", V, VgC, "kN", MSB + " §4.2.2.2", "COMP", True, "VRd,g = 2·Lc·tc·fy/(1,27·√3·γM0)", "")
    ck("cnA", "Cornières", "Ailes A – cisaillement section nette", V, dv(2 * R.t_C * (R.L_C - R.n1_P * d0) * R.fu_C, S3 * gM2n) / 1000, "kN", MSB + " §4.2.2.2", "COMP", R.bolt_P,
       "VRd,n = 2·Av,net·fu/(√3·γM2)", "")
    ck("cbA", "Cornières", "Ailes A – rupture de bloc", V, 2 * (dv(0.5 * R.fu_C * R.Ant_A, gM2n) + dv(R.fy_C * R.Anv_A / S3, gM0)) / 1000, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : " + MSB + " §4.2.2.2", "EC", R.bolt_P, "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0) (× 2 cornières)",
       "Ant = " + F(R.Ant_A, 0) + " mm² ; Anv = " + F(R.Anv_A, 0) + " mm²")
    ck("flA", "Cornières", "Ailes A – flexion dans leur plan entre talon et file de boulons (ou cordon)", V / 2 * ((R.g_A if R.bolt_P else R.b_A) - R.t_C / 2) / 1000, WC, "kNm",
       "EN 1993-1-1 §6.2.5 ; bras de levier : hypothèse de l'outil", "INT", True, "MEd = (VEd/2)·(gA − tc/2) ; MRd = Wel·fy/γM0", "")
    ck("tsA", "Cornières", "Ailes A en flexion + boulons P en traction (tronçon en T équivalent)", NEd, mn(R.FT_1, R.FT_2, R.FT_3), "kN",
       "EN 1993-1-8 §6.2.4 Tableau 6.2 (méthode 2) ; leff et m : " + MSB + " §4.3.1.1", "EC + COMP", R.bolt_P and NEd > 0,
       "FT,1 = (8n − 2ew)·Mpl/[2mn − ew(m + n)] ; FT,2 = (2Mpl + n·ΣFt,Rd)/(m + n) ; FT,3 = ΣFt,Rd",
       "Σleff = " + F(R.leff_T, 0) + " mm ; m = " + F(R.m_T, 1) + " mm ; n = " + F(R.n_T, 1) + " mm ; Mpl = " + F(R.Mpl_T, 2) + " kNm ; FT,1 = " + F(R.FT_1, 0) + " ; FT,2 = " + F(R.FT_2, 0) + " ; FT,3 = " + F(R.FT_3, 0) + " kN")
    ck("tnB", "Cornières", "Ailes B – traction section nette", NEd, dv(2 * 0.9 * R.t_C * (R.L_C - R.n1_S * d0) * R.fu_C, gM2n) / 1000, "kN", "EN 1993-1-1 §6.2.3(2)", "EC",
       R.bolt_S and NEd > 0, "Nu,Rd = 0,9·Anet·fu/γM2", "")
    ck("tbB", "Cornières", "Ailes B – rupture de bloc en traction", NEd, mn(R.Vt1_B, R.Vt2_B), "kN", "EN 1993-1-8 §3.10.2(2) ; aires : " + MSB + " §4.3.1.4", "EC",
       R.bolt_S and NEd > 0, "Veff,1,Rd = fu·Ant/γM2 + fy·Anv/(√3·γM0)", "cas 1 = " + F(R.Vt1_B, 0) + " kN ; cas 2 = " + F(R.Vt2_B, 0) + " kN")
    ck("pdS", "Poutre secondaire", "Pression diamétrale – âme de la poutre secondaire", R.F_S, dv(R.F_S, R.ipd_S) if R.ipd_S > 0 else R.Fbv_S, "kN",
       T34 + " ; interaction : " + MSB + " §4.2.1.1", "EC + COMP", R.bolt_S, "Fb,Rd = k1·αb·fu·d·tw/γM2 ; η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²]", bvals("S"), 1)
    ck("vgS", "Poutre secondaire", "Âme – cisaillement section brute (section réduite si grugée)", V, R.VRd_gS, "kN", "EN 1993-1-1 §6.2.6 ; " + MSB + " §4.2.3.1", "EC", True,
       "VRd,g = Av·fy/(√3·γM0)", "Av = " + F(R.Av_S, 0) + " mm²", 1)
    ck("vnS", "Poutre secondaire", "Âme – cisaillement section nette au droit des boulons", V, dv((R.Av_S - R.n1_S * d0 * R.tw_S) * R.fu_S / S3, gM2n) / 1000, "kN",
       MSB + " §4.2.3.1 (ECCS n°126)", "COMP", R.bolt_S, "VRd,n = (Av − n1·d0·tw)·fu/(√3·γM2)", "Av,net = " + F(R.Av_S - R.n1_S * d0 * R.tw_S, 0) + " mm²", 1)
    ck("vbS", "Poutre secondaire", "Âme – rupture de bloc", V, R.VRd_bS, "kN", "EN 1993-1-8 §3.10.2(3) ; aires : " + MSB + " §4.2.3.1", "EC", R.bolt_S,
       "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0)", "Ant = " + F(R.Ant_S, 0) + " mm² ; Anv = " + F(R.Anv_S, 0) + " mm²", 1)
    ck("mN", "Poutre secondaire", "Poutre grugée – flexion + cisaillement à l'extrémité du grugeage", V * (u.g_h + u.l_n) / 1000 + Mabs, R.Mv_N, "kNm",
       MSB + " §4.2.4 ; SCI P358 Check 5", "COMP", R.cas_g > 0,
       "MEd = VEd·(gh + ln) ; Mv,N,Rd = fy·Wel,N/γM0·[1 − (2VEd/Vpl,N,Rd − 1)²] si VEd > 0,5·Vpl,N,Rd",
       "Wel,N = " + F(R.W_N, 0) + " mm³ ; Vpl,N,Rd = " + F(R.VRd_gS, 1) + " kN ; réduction = " + F(R.rho_N, 3), 1)
    ck("m2", "Poutre secondaire", "Poutre grugée – interaction à la 2e file de boulons (n2 = 2)", V * (u.g_h + R.e2b_S + u.p2_S) / 1000, R.Mc_2, "kNm", MSB + " §4.2.3.2", "COMP",
       R.cas_g > 0 and R.bolt_S and R.n2_S == 2 and u.l_n > R.e2b_S + u.p2_S, "MEd = VEd·(gh + e2,b + p2)", "")
    ck("stN", "Poutre secondaire", "Stabilité locale du grugeage – longueur ln", u.l_n, R.ln_max, "mm", MSB + " §4.2.5 ; SCI P358 Check 6", "COMP", R.cas_g > 0,
       "ln ≤ h si h/tw ≤ limite ; sinon ln ≤ K·h/(h/tw)³", "h/tw = " + F(R.el_N, 1) + " ; limite = " + F(R.lim_N, 1) + " ; domaine : " + R.val_N, 1)
    ck("stD", "Poutre secondaire", "Stabilité locale du grugeage – profondeur", mx(u.d_nt, u.d_nb), R.h_S / 2 if R.cas_g == 1 else R.h_S / 5, "mm", MSB + " §4.2.5", "COMP", R.cas_g > 0,
       "dn ≤ h/2 (1 semelle grugée) ; dn ≤ h/5 (2 semelles)", "")
    ck("tnS", "Poutre secondaire", "Âme – traction section nette", NEd, dv(0.9 * R.tw_S * (R.L_C - R.n1_S * d0) * R.fu_S, gM2n) / 1000, "kN",
       "EN 1993-1-1 §6.2.3(2) ; hauteur = Lc : " + MSB + " §4.3.2.2", "EC", R.bolt_S and NEd > 0, "Nu,Rd = 0,9·Anet·fu/γM2", "")
    ck("tbS", "Poutre secondaire", "Âme – rupture de bloc en traction", NEd, mn(R.Vt1_S, R.Vt2_S if R.cas_g > 0 else INF), "kN",
       "EN 1993-1-8 §3.10.2(2) ; aires : " + MSB + " §4.3.2.3", "EC", R.bolt_S and NEd > 0, "Veff,1,Rd = fu·Ant/γM2 + fy·Anv/(√3·γM0)", "")
    ck("pdP", "Poutre principale", "Pression diamétrale – âme de la poutre principale", R.F_P, dv(R.F_P, R.ipd_P) if R.ipd_P > 0 else R.Fbv_P, "kN",
       T34 + " ; SCI P358 Check 8", "EC", R.bolt_P, "Fb,Rd = k1·αb·fu·d·tw/γM2 (pas de bord libre : boulons intérieurs)", bvals("P"), 1)
    ck("vlP", "Poutre principale", "Cisaillement local de l'âme porteuse – section brute", V / 2, dv(R.Av_P * R.fy_P / S3, gM0) / 1000, "kN", P10, "COMP", True,
       "VEd/2 ≤ Av·fy/(√3·γM0) ; Av = tw·(et + (n1 − 1)·p1 + eb)", "Av = " + F(R.Av_P, 0) + " mm²", 1)
    ck("vnP", "Poutre principale", "Cisaillement local de l'âme porteuse – section nette", V / 2, dv((R.Av_P - R.n1_P * d0 * R.tw_P) * R.fu_P / S3, gM2n) / 1000, "kN", P10, "COMP",
       R.bolt_P, "VEd/2 ≤ Av,net·fu/(√3·γM2)", "")
    for X, lab, act in (("S", "ailes B / âme poutre secondaire", not R.bolt_S), ("P", "ailes A / âme poutre principale", not R.bolt_P)):
        ck("w" + X, "Soudures", "Cordons " + lab + " – effort par unité de longueur au point critique", R["qw_" + X], R["Fw_" + X], "N/mm", W, "EC + COMP", act,
           "Fw,Ed = √(qz² + qx² + qn²) ≤ Fw,Rd = fvw,d·a ; fvw,d = fu/(√3·βw·γM2)",
           "a = " + F(u["a_" + X], 0) + " mm ; ΣL = " + F(R["Lw_" + X], 0) + " mm ; Ip = " + F(R["Iw_" + X], 0) + " mm³ ; qz = " + F(R["qz_" + X], 0) + " ; qx = " + F(R["qx_" + X], 0) + " ; qn = " + F(R["qn_" + X], 0) + " N/mm ; fvw,d = " + F(R["fvw_" + X], 1) + " MPa", 1)
        ck("wa" + X, "Soudures", "Cordons " + lab + " – gorge minimale", 3, u["a_" + X], "mm", "EN 1993-1-8 §4.5.2(2)", "EC", act, "a ≥ 3 mm", "")
        ck("wl" + X, "Soudures", "Cordons " + lab + " – longueur minimale", mx(30, 6 * u["a_" + X]), R.L_C, "mm", "EN 1993-1-8 §4.5.1(2)", "EC", act, "L ≥ max(30 mm ; 6a)", "")

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
    R.statut = "ASSEMBLAGE VÉRIFIÉ" if R.verified else "ASSEMBLAGE NON VÉRIFIÉ"
    R.reserve = bool(R.cas_g > 0 and R.val_N.startswith("NON VALIDE"))
    return R


# =========================================================================
#  boltTable
# =========================================================================
def bolt_table(n1, n2, p1, p2, V, Hh, M, Ip):
    """Efforts par boulon (répartition élastique, inertie polaire) : rangée
    ``i`` (haut → bas), file ``j``, position, composantes et résultante."""
    t = []
    n = n1 * n2
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            x = (j - (n2 + 1) / 2) * p2; y = ((n1 + 1) / 2 - i) * p1
            fx = dv(Hh, n) - (dv(M * 1000 * y, Ip) if Ip > 0 else 0)
            fz = dv(V, n) + (dv(M * 1000 * x, Ip) if Ip > 0 else 0)
            t.append(EffortBoulon(i=i, j=j, x=x, y=y, fx=fx, fz=fz, f=hyp(fx, fz)))
    return t


# =========================================================================
#  predim
# =========================================================================
def predim(u, R, Mabs):
    """Prédimensionnement : pour chaque boulon (M12 → M36) et chaque nombre
    de rangées (2 → 8), pinces et entraxes de détail (multiples de 5), taux
    estimés (boulons S et P, pressions diamétrales des âmes, interaction
    éventuelle), puis cornière nécessaire pour la solution retenue.

    Règle de choix : parmi les combinaisons géométriquement possibles au
    taux cible, le plus petit ``score = n·100 + i + 1`` (le moins de rangées,
    puis le plus petit diamètre) ; à défaut, la moins sollicitée. Premier
    minimum rencontré, comme le moteur de référence."""
    rows = []
    hav = R.h_S - (u.d_nb if u.d_nb > 0 else R.tf_S + R.r_S) - u.z_C
    V = u.V_Ed; H = abs(u.H_Ed)
    i1 = idx(DB["boulons"], u.pd_dmin); i2 = idx(DB["boulons"], u.pd_dmax)
    imin = min(i1, i2); imax = max(i1, i2)
    for i, b in enumerate(DB["boulons"]):
        for n in range(2, 9):
            d = b["d"]; d0 = b["d0"]; A = b["As"] if u.filet == "Oui" else b["A"]
            e1 = ceil5(u.k_e1 * d0); p1 = ceil5(u.k_p1 * d0); e2 = ceil5(u.k_e2 * d0)
            Lc = 2 * e1 + (n - 1) * p1
            geom = bool(Lc <= hav and imin <= i <= imax)
            Fv = dv(R.a_v * R.f_ub * A, u.g_M2) / 1000; Lj = (n - 1) * p1
            bLf = mx(0.75, 1 - (Lj - 15 * d) / (200 * d)) if (u.opt_blf == "Oui" and Lj > 15 * d) else 1
            Mz = V * (u.g_h + e2) / 1000 + Mabs; Fz = V / n
            Fx = abs(u.N_Ed) / n + dv(Mz * 1000 * 6, p1 * n * (n + 1)); Fs = hyp(Fz, Fx)
            e1b = u.z_C + e1 - u.d_nt
            Fbv = (mn(2.8 * e2 / d0 - 1.7, 2.5) * mn(e1b / (3 * d0), p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            Fbh = (mn(2.8 * e1b / d0 - 1.7, 1.4 * p1 / d0 - 1.7, 2.5) * mn(e2 / (3 * d0), dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            FP = hyp(V / 2 / n, H / 2 / n)
            FbP = (2.5 * mn(p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_P), 1) * R.fu_P * d * R.tw_P / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            e = AttrDict(
                bS=dv(Fs, 2 * Fv * bLf) if R.bolt_S else 0,
                pdS=hyp(dv(Fz, Fbv), dv(Fx, Fbh)) if R.bolt_S else 0,
                bP=dv(FP, u.k_rot * Fv * bLf) if R.bolt_P else 0,
                pdP=dv(FP, FbP) if R.bolt_P else 0,
                vt=(dv(FP, Fv * bLf) + dv(u.N_Ed / (2 * n), dv(1.4 * 0.9 * R.f_ub * b["As"], u.g_M2) / 1000)) if (R.bolt_P and u.N_Ed > 0) else 0)
            eta = mx(e.bS, e.pdS, e.bP, e.pdP, e.vt); ok = bool(geom and eta <= u.eta_c)
            rows.append(AttrDict(b=b, i=i, n=n, e1=e1, p1=p1, e2=e2, Lc=Lc, geom=geom, Fz=Fz, Fx=Fx,
                                 e=e, eta=eta, ok=ok, score=(n * 100 + i + 1) if ok else 9999,
                                 etag=eta if geom else 99, prop=None))
    found = any(r.ok for r in rows); key = "score" if found else "etag"; pick = rows[0]
    for r in rows:
        if r[key] < pick[key]:
            pick = r
    gM0 = u.g_M0; gM2n = u.g_M2n; anyB = R.bolt_S or R.bolt_P
    legs = [a[0] for a in DB["ailes"]]

    def cleat_for(pk):
        d0 = pk.b["d0"]; d = pk.b["d"]; n = pk.n; Lc = pk.Lc
        fbv = mn(2.8 * pk.e2 / d0 - 1.7, 2.5) * mn(pk.e1 / (3 * d0), pk.p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_C), 1) * R.fu_C * d
        fbv = dv(fbv, u.g_M2) / 1000 * R.k_trou
        fbh = mn(2.8 * pk.e1 / d0 - 1.7, 1.4 * pk.p1 / d0 - 1.7, 2.5) * mn(pk.e2 / (3 * d0), dv(R.f_ub, R.fu_C), 1) * R.fu_C * d
        fbh = dv(fbh, u.g_M2) / 1000 * R.k_trou
        reqs = [["Pression diamétrale ailes B (effort par cornière = F/2)", hyp(dv(pk.Fz / 2, fbv), dv(pk.Fx / 2, fbh)) if R.bolt_S else 0],
                ["Pression diamétrale ailes A", hyp(dv(V / 2 / n, fbv), dv(H / 2 / n, fbh)) if R.bolt_P else 0],
                ["Cisaillement section brute", dv(V * 1000 * 1.27 * S3 * gM0, 2 * Lc * R.fy_C)],
                ["Cisaillement section nette", dv(V * 1000 * S3 * gM2n, 2 * (Lc - n * d0) * R.fu_C) if anyB else 0],
                ["Rupture de bloc", dv(V * 1000, 2 * (dv(0.5 * R.fu_C * (pk.e2 - 0.5 * d0), gM2n) + dv(R.fy_C * (Lc - pk.e1 - (n - 0.5) * d0) / S3, gM0))) if anyB else 0],
                ["Flexion des ailes B dans leur plan", dv((V / 2 * (u.g_h + pk.e2) / 1000 + Mabs / 2) * 1e6 * 6 * gM0, Lc * Lc * R.fy_C)],
                ["Traction section nette (NEd > 0)", dv(u.N_Ed * 1000 * gM2n, 2 * 0.9 * (Lc - n * d0) * R.fu_C) if (R.bolt_S and u.N_Ed > 0) else 0]]
        treq = dv(mx(*[r[1] for r in reqs]), u.eta_c); tC = DB["epais"][_first_ge(DB["epais"], treq)]; dw = pk.b["dw"]
        req1 = mx(ceil5(tC + 12 + dw / 2 + 2) + pk.e2, u.g_h + 2 * pk.e2); r1 = DB["ailes"][_first_ge(legs, req1)][1]
        gA = ceil5(tC + r1 + dw / 2 + 2); legreq = mx(gA + pk.e2, u.g_h + 2 * pk.e2); il = _first_ge(legs, legreq)
        return AttrDict(reqs=reqs, treq=treq, tC=tC, gA=gA, legreq=legreq, leg=DB["ailes"][il][0], rC=DB["ailes"][il][1])

    for r in rows:
        if r.geom:
            r.prop = cleat_for(r)
    if not pick.prop:
        pick.prop = cleat_for(pick)
    pp = pick.prop
    if not found:
        msg = "AUCUNE combinaison ne respecte le taux cible dans les limites fixées : la moins sollicitée est affichée. Élargir la plage de diamètres, la hauteur disponible ou la classe de boulons."
    elif pp.treq > 15:
        msg = "Épaisseur nécessaire > 15 mm : hors gamme. Augmenter le nombre de rangées ou la nuance des cornières."
    else:
        msg = "Proposition complète"
    return AttrDict(rows=rows, hav=hav, found=found, pick=pick, boulon=pick.b["n"], n=pick.n, e1=pick.e1,
                    p1=pick.p1, e2=pick.e2, LC=pick.Lc, eta=pick.eta, reqs=pp.reqs, treq=pp.treq, tC=pp.tC,
                    gA=pp.gA, legreq=pp.legreq, leg=pp.leg, rC=pp.rC, msg=msg)


# =========================================================================
#  applySolution / solutionLabel
# =========================================================================
def apply_solution(u0, r):
    """Recopie une solution du prédimensionnement dans les saisies (source
    unique) et repasse en mode VÉRIFICATION. Retourne le nouveau dictionnaire
    des saisies ; ``u0`` n'est pas modifié."""
    u = dict(u0)
    p = r.prop
    c = None
    for x in DB["cornieres"]:
        if c is None and x["a1"] == p.leg and x["a2"] == p.leg and x["t"] == p.tC and x["r"] == p.rC:
            c = x
    u["mode_calc"] = "VÉRIFICATION"; u["boulon_u"] = r.b["n"]; u["n1S_u"] = r.n; u["n2S_u"] = 1
    u["p1S_u"] = r.p1; u["e1S_u"] = r.e1; u["e2b_u"] = r.e2; u["n1P_u"] = r.n; u["n2P_u"] = 1
    u["p1P_u"] = r.p1; u["e1P_u"] = r.e1; u["gA_u"] = p.gA; u["LC_u"] = r.Lc
    if c:
        u["corn_u"] = c["n"]
    else:
        u["corn_u"] = PERSO; u["k1_u"] = p.leg; u["k2_u"] = p.leg; u["kt_u"] = p.tC; u["kr_u"] = p.rC
    return u


def solution_label(r):
    """« M20, 3 rangées, L100x100x10 »."""
    return (r.b["n"] + ", " + js_str(r.n) + " rangées, L" + js_str(r.prop.leg) + "x"
            + js_str(r.prop.leg) + "x" + js_str(r.prop.tC))


def nom_cornière_proposee(r):
    """« L100x100x10 » pour une ligne du prédimensionnement."""
    return "L" + js_str(r.prop.leg) + "x" + js_str(r.prop.leg) + "x" + js_str(r.prop.tC)

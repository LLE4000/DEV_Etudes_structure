# -*- coding: utf-8 -*-
"""Moteur de calcul — assemblage POUTRE–POTEAU par DOUBLES CORNIÈRES d'âme.

Deux cornières identiques, ailes B sur l'âme de la poutre portée (groupe S,
deux plans de cisaillement) et ailes A sur le POTEAU (groupe P, un plan par
cornière) — boulonnées ou soudées de chaque côté.

DEUX cas d'appui, et c'est la clé ``appui`` qui commande toute la mécanique
(`docs/assemblages/POUTRE_POTEAU.md` §1.2) :

- **cas (a), sur l'AILE** : les ailes A portent sur la face extérieure d'une
  aile, SYMÉTRIQUEMENT par rapport à l'âme du poteau (deux files à ±p3/2).
  L'attache vit DANS LE PLAN de cette âme : VEd et VEd·z y cheminent en
  plan, et le côté poteau n'apporte que la PRESSION DIAMÉTRALE de l'aile
  (attache boulonnée) ou les CORDONS (attache soudée). La flexion locale de
  l'aile n'apparaît QUE lorsque les boulons sont tendus — tronçon en T de
  l'aile non raidie (Tableau 6.4) et âme du poteau en traction transversale
  (§6.2.6.3), actifs en tying et à l'ELU si NEd > 0 (§4.1, §7.3, §7.4) ;
- **cas (b), dans l'ÂME** : les ailes A portent sur l'âme, entre les ailes.
  L'âme du poteau joue le rôle de l'âme de la poutre porteuse des modules
  poutre–poutre — cisaillement local d'un élément support chargé d'un seul
  côté (SCI P358), pression diamétrale, et trois conditions géométriques
  (largeur droite entre congés, dégagement de l'écrou, bout de l'aile A).
  La flexion HORS PLAN de l'âme sous la force de chaînage n'est PAS couverte
  par l'EN 1993-1-8 pour une âme non raidie : elle est déclarée en limite
  (§5.4), elle n'est pas transposée.

Modèles et sources, distingués vérification par vérification (champ ``nat``) :
- ``EC``   : formule directe de l'EN 1993-1-8 / EN 1993-1-1 ;
- ``COMP`` : modèle complémentaire d'un guide reconnu — « Steel Buildings in
             Europe, Part 5: Joint Design » §4 (doubles cornières),
             SCI P358, ECCS n°126 ;
- ``INT``  : interprétation de l'outil, nommée comme telle.

Les briques de calcul communes à la famille ne sont PAS recopiées : elles
viennent de ``acier/assemblages/mecanique.py`` (pression diamétrale, groupe
excentré, poutre grugée, tronçon en T, longueurs efficaces de l'aile de
poteau non raidie), que ``tests/test_assemblages_mecanique.py`` lie aux
moteurs poutre–poutre.

Unités : mm, mm², mm³, mm⁴, MPa, kN, kNm, N/mm (cordons).
"""
import math

from acier.bibliotheques import DB, PERSO, ORI_P, find, idx
from acier.js import AttrDict, INF, N, dv, mn, mx, js_round, ceil5, hyp, js_str
from acier.formats import F, f0
from acier.resultats import Verification, Alerte, Pince
from acier import resistances as RS
from acier.assemblages import mecanique as MEC
from .entrees import defaults, NUM, MODE_PREDIM, BOULONNEE, APPUI_AME, NAT_TYING

S3 = math.sqrt(3)
T34 = "EN 1993-1-8 Tableau 3.4"
MSB = "MSB Part 5"
MSB_TY = "MSB Part 5 §2.3 (tying)"
P358_SUP = "SCI P358 (élément support chargé d'un seul côté)"
TAB64 = "EN 1993-1-8 Tableau 6.4 (aile de poteau non raidie)"

GROUPES_VERIF = ("Boulons", "Cornières", "Poutre portée", "Poteau porteur",
                 "Soudures")


def _first_ge(liste, v):
    """Indice du premier élément ≥ v (nombre d'éléments < v), borné."""
    c = sum(1 for x in liste if x < v)
    return min(c, len(liste) - 1)


# =========================================================================
#  compute
# =========================================================================
def compute(u0=None):
    """Calcul complet pour les saisies ``u0`` (clés absentes → défauts).
    Retourne ``R`` : scalaires du modèle, ``u``, ``checks`` / ``ck``,
    ``alerts``, ``dist`` (Tableau 3.3), ``tabS`` / ``tabP`` (efforts par
    boulon), ``pd`` (prédimensionnement), ``gov``, ``eta_max``, ``verified``,
    ``statut``, ``reserve``."""
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
    R.ame = u.appui == APPUI_AME            # cas (b) : attache dans l'âme
    R.aile = not R.ame                      # cas (a) : attache sur l'aile
    R.appui = APPUI_AME if R.ame else u.appui
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
        R["fy_" + X] = a["fy"]; R["fu_" + X] = a["fu"]; R["bw_" + X] = a["bw"]
    cl = find(DB["classes"], u.classe)
    R.f_ub = cl["fub"]
    R.a_v = cl["av"] if u.filet == "Oui" else 0.6
    R.k_trou = 1 if u.trou == "Normal" else 0.8
    pd = R.pd = predim(u, R, Mabs)

    # ---- paramètres retenus (◆ en prédimensionnement)
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
    # ORI_P : la GRANDE aile va du côté du support — ici le POTEAU (aile A)
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
    d0 = R.d_0; d = R.d_b; V = u.V_Ed; H = abs(u.H_Ed)
    gM0 = u.g_M0; gM2 = u.g_M2; gM2n = u.g_M2n; gMu = u.g_Mu

    # ---- effort normal : ELU fondamental (combiné à VEd) ou TYING (seul)
    R.tying = bool(u.nat_N == NAT_TYING and u.N_Ed > 0)
    R.N_elu = 0.0 if R.tying else u.N_Ed      # traction (ou compression) ELU
    R.N_ty = u.N_Ed if R.tying else 0.0       # force de chaînage, seule
    NEd = R.N_elu
    NT = R.N_ty if R.tying else mx(0.0, R.N_elu)   # effort de traction vérifié
    gT = gMu if R.tying else gM2n                  # γ des sections nettes
    gB = gMu if R.tying else gM2                   # γ des boulons et cordons
    R.g_T = gT; R.g_B = gB

    # ---- géométrie : la FACE D'APPUI (rotule) selon le cas
    # cas (a) : face extérieure de l'aile, X = hP/2 depuis l'axe du poteau ;
    # cas (b) : face de l'âme, X = twP/2 (MSB P5 §4.2.1.1 ; étude §6.1).
    R.xf_P = R.tw_P / 2 if R.ame else R.h_P / 2
    R.t_appui = R.tw_P if R.ame else R.tf_P      # épaisseur sollicitée
    R.g_B2 = u.g_h + R.e2b_S                     # face d'appui → 1re file (groupe S)
    R.z_S = R.g_B2 + (R.n2_S - 1) * u.p2_S / 2
    R.e2a_S = R.b_B - R.g_B2 - (R.n2_S - 1) * u.p2_S
    R.e1bot_S = R.L_C - R.e1_S - (R.n1_S - 1) * R.p1_S
    R.e1b_S = u.z_C + R.e1_S - u.d_nt
    R.h_e = R.h_S - u.d_nb - (u.z_C + R.e1_S + (R.n1_S - 1) * R.p1_S)
    R.n_S = R.n1_S * R.n2_S
    grpS = MEC.groupe_excentre(R.n1_S, R.n2_S, R.p1_S, u.p2_S, V, NEd, 0)
    R.Ip_S = grpS["Ip"]; R.xm_S = grpS["xm"]; R.ym_S = grpS["ym"]
    R.p_3 = R.tw_S + 2 * R.g_A                   # entraxe des deux files sur le poteau
    R.e2a_P = R.b_A - R.g_A - (R.n2_P - 1) * u.p2_P
    R.e1bot_P = R.L_C - R.e1_P - (R.n1_P - 1) * R.p1_P
    R.n_P = R.n1_P * R.n2_P
    grpP = MEC.groupe_excentre(R.n1_P, R.n2_P, R.p1_P, u.p2_P, V / 2, H / 2, 0)
    R.Ip_P = grpP["Ip"]; R.xm_P = grpP["xm"]; R.ym_P = grpP["ym"]
    R.e_P = R.g_A + (R.n2_P - 1) * u.p2_P / 2
    for X in ("S", "P"):
        lh = u["lh_" + X]; L = R.L_C; Lw = L + 2 * lh; xg = dv(lh * lh, Lw)
        R["Lw_" + X] = Lw; R["xg_" + X] = xg
        R["Iw_" + X] = (L * L * L / 12 + 2 * lh * (L / 2) * (L / 2) + L * xg * xg
                        + 2 * (lh * lh * lh / 12 + lh * (lh / 2 - xg) * (lh / 2 - xg)))
    R.zw_S = R.b_B - R.xg_S; R.ew_P = R.b_A - R.xg_P
    R.zeff = R.z_S if R.bolt_S else R.zw_S
    R.M_S = V * R.zeff / 1000 + Mabs
    R.M_P = V / 2 * (R.e_P if R.bolt_P else R.ew_P) / 1000 if u.opt_exc == "Oui" else 0
    if not R.bolt_S:
        R.mod_S = "Soudé – groupe de cordons avec moment"
    elif R.M_S == 0:
        R.mod_S = "Distribution uniforme"
    else:
        R.mod_S = ("Rotule à la face d'appui du poteau — groupe de boulons avec moment "
                   "VEd·z (répartition élastique, inertie polaire)")
    if not R.bolt_P:
        R.mod_P = "Soudé – cisaillement centré" if R.M_P == 0 else "Soudé – groupe de cordons avec moment"
    elif R.M_P == 0:
        R.mod_P = "Distribution uniforme (cisaillement centré, facteur " + F(u.k_rot, 2) + " sur Fv,Rd)"
    else:
        R.mod_P = "Groupe de boulons avec moment (répartition élastique – inertie polaire)"
    R.rec_L = R.L_C >= 0.6 * R.h_S              # maintien en torsion (guides)

    # ---- géométrie propre au POTEAU (étude §3.2.2, §4.3, §5.3)
    R.libre_droit = R.h_P - 2 * R.tf_P - 2 * R.r_P        # largeur droite de l'âme
    R.demi_droit = (R.h_P - 2 * R.tf_P) / 2 - R.r_P       # demi-largeur droite
    R.libre_ailes = R.h_P - 2 * R.tf_P - 2 * u.j_lat      # passage de la semelle
    R.p3_congeP = R.tw_P + 2 * R.r_P + R.d_w              # gauge mini sur l'aile
    R.bA_max = R.demi_droit - R.tw_S / 2                  # aile A maxi en cas (b)
    R.p3_max_ame = 2 * (R.demi_droit - R.d_w / 2)         # gauge maxi en cas (b)
    R.gh_b1 = (R.b_P - R.tw_P) / 2 + u.j_lat              # about au-delà des bouts d'aile
    R.rec_min = mx(0.0, R.gh_b1 - u.g_h)                  # recoupe mini si la poutre ne passe pas
    R.passe_S = bool(R.b_S <= R.libre_ailes + 1e-9)       # la semelle passe entre les ailes
    R.b1 = bool(R.ame and u.g_h >= R.gh_b1 - 1e-9)        # about au-delà des bouts d'aile
    R.rec_ok = bool(not R.ame or R.passe_S or R.b1 or u.rec_S >= R.rec_min - 1e-9)
    # excentricité apportée au POTEAU : information, pas une vérification de
    # l'attache — elle appartient au calcul d'élément (EC3-1-1 §6.2 et §6.3)
    R.e_col = (R.h_P / 2 + R.zeff) if R.aile else R.zeff
    R.M_col = V * R.e_col / 1000

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

    tS = mn(R.t_C, R.tw_S); tP = mn(R.t_C, R.t_appui)
    drow("S_e1c", "S – e1 cornière (haut/bas, valeur mini)", mn(R.e1_S, R.e1bot_S), 1.2, R.t_C, "e", R.bolt_S,
         ["e1S_u", "LC_u", "n1S_u", "p1S_u"], ["e1S", "e1botS"])
    drow("S_e2c", "S – e2 cornière (bord libre de l'aile B)", R.e2a_S, 1.2, R.t_C, "e", R.bolt_S,
         cornF + ["g_h", "e2b_u", "p2_S"], ["e2B"])
    drow("S_e1b", "S – e1,b âme de la poutre portée (vers grugeage sup.)", R.e1b_S, 1.2, R.tw_S, "e",
         R.bolt_S and u.d_nt > 0, ["z_C", "e1S_u", "d_nt"], ["e1b", "zc", "dcs"])
    drow("S_he", "S – he âme de la poutre portée (vers grugeage inf.)", R.h_e, 1.2, R.tw_S, "e",
         R.bolt_S and u.d_nb > 0, ["z_C", "e1S_u", "n1S_u", "p1S_u", "d_nb"], ["he", "dci"])
    drow("S_e2b", "S – e2,b âme de la poutre portée (vers l'about)", R.e2b_S, 1.2, R.tw_S, "e", R.bolt_S,
         ["e2b_u"], ["e2b"])
    drow("S_p1", "S – p1", R.p1_S, 2.2, tS, "p", R.bolt_S and R.n1_S > 1, ["p1S_u"], ["p1S"])
    drow("S_p2", "S – p2", u.p2_S, 2.4, tS, "p", R.bolt_S and R.n2_S > 1, ["p2_S"], ["p2S"])
    drow("P_e1c", "P – e1 cornière (haut/bas, valeur mini)", mn(R.e1_P, R.e1bot_P), 1.2, R.t_C, "e", R.bolt_P,
         ["e1P_u", "LC_u", "n1P_u", "p1P_u"], ["e1P", "e1botP"])
    drow("P_e2c", "P – e2 cornière (bord libre de l'aile A)", R.e2a_P, 1.2, R.t_C, "e", R.bolt_P,
         cornF + ["gA_u", "p2_P"], ["e2A"])
    drow("P_p1", "P – p1", R.p1_P, 2.2, tP, "p", R.bolt_P and R.n1_P > 1, ["p1P_u"], ["p1P"])
    drow("P_p2", "P – p2", u.p2_P, 2.4, tP, "p", R.bolt_P and R.n2_P > 1, ["p2_P"], ["p2P"])
    # le GAUGE sur le poteau : entraxe des deux files, p2 du Tableau 3.3
    drow("P_p3", "P – p3 gauge sur le poteau (entraxe des deux files)", R.p_3, 2.4, tP, "p", R.bolt_P,
         ["gA_u"] + cornF, ["p3"])
    # pince au BOUT D'AILE du poteau (cas a) : le seul bord libre du poteau
    drow("P_eP", "P – pince au bout d'aile du poteau e = (bP − p3)/2", (R.b_P - R.p_3) / 2, 1.2,
         R.tf_P, "e", R.bolt_P and R.aile,
         ["gA_u", "prof_P"] + cornF, ["eP"])
    R.dist_ok = all(x.ok for x in R.dist)

    # ---- alertes
    tfrS = R.tf_S + R.r_S; m12 = 1.2 * d0
    botLim = R.h_S - (u.d_nb if u.d_nb > 0 else tfrS); sumC = u.z_C + R.L_C
    prP = ["prof_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"] if u.prof_P == PERSO else ["prof_P"]
    prS = ["prof_S", "hS_u", "bS_u", "twS_u", "tfS_u", "rS_u"] if u.prof_S == PERSO else ["prof_S"]
    cA = R.bolt_P and R.g_A - d0 / 2 < R.t_C + R.r_C
    cB = R.bolt_S and R.g_B2 - d0 / 2 < R.t_C + R.r_C
    AL = [
        dict(id="S_e1bot", c=R.bolt_S and R.e1bot_S < m12, b=1,
             m="Configuration impossible : entraxe vertical / nombre de rangées du groupe S incompatible avec la hauteur de cornière hc (pince basse < 1,2·d0).",
             f=["LC_u", "n1S_u", "p1S_u", "e1S_u"], d=["Lc", "e1S", "p1S", "e1botS"], e=["cleat", "boltsS"],
             cv=["S_e1c"] if R.e1_S >= m12 - 0.001 else [],
             w="Pince basse du groupe S = hc − e1 − (n1 − 1)·p1 = " + f0(R.L_C) + " − " + f0(R.e1_S) + " − " + js_str(R.n1_S - 1) + " × " + f0(R.p1_S) + " = " + f0(R.e1bot_S) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. hc minimal avec ces boulons : " + f0(math.ceil(R.e1_S + (R.n1_S - 1) * R.p1_S + m12)) + " mm."),
        dict(id="P_e1bot", c=R.bolt_P and R.e1bot_P < m12, b=1,
             m="Configuration impossible : entraxe vertical / nombre de rangées du groupe P incompatible avec la hauteur de cornière hc (pince basse < 1,2·d0).",
             f=["LC_u", "n1P_u", "p1P_u", "e1P_u"], d=["Lc", "e1P", "p1P", "e1botP"], e=["cleat", "boltsP"],
             cv=["P_e1c"] if R.e1_P >= m12 - 0.001 else [],
             w="Pince basse du groupe P = hc − e1 − (n1 − 1)·p1 = " + f0(R.L_C) + " − " + f0(R.e1_P) + " − " + js_str(R.n1_P - 1) + " × " + f0(R.p1_P) + " = " + f0(R.e1bot_P) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. hc minimal avec ces boulons : " + f0(math.ceil(R.e1_P + (R.n1_P - 1) * R.p1_P + m12)) + " mm."),
        dict(id="S_e2", c=R.bolt_S and R.e2a_S < m12, b=1,
             m="Configuration impossible : aile B trop courte – pince e2 du groupe S non conforme à EN 1993-1-8 Tableau 3.3.",
             f=cornF + ["g_h", "e2b_u", "n2S_u", "p2_S"], d=["e2B", "e2b", "gh", "p2S", "bB"], e=["cleat", "boltsS"], cv=["S_e2c"],
             w="Pince e2 = bB − gh − e2,b − (n2 − 1)·p2 = " + f0(R.b_B) + " − " + f0(u.g_h) + " − " + f0(R.e2b_S) + " − " + js_str(R.n2_S - 1) + " × " + f0(u.p2_S) + " = " + f0(R.e2a_S) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. Réduire e2,b ou p2, ou choisir une aile B plus longue."),
        dict(id="P_e2", c=R.bolt_P and R.e2a_P < m12, b=1,
             m="Configuration impossible : aile A trop courte – pince e2 du groupe P non conforme à EN 1993-1-8 Tableau 3.3.",
             f=cornF + ["gA_u", "n2P_u", "p2_P"], d=["e2A", "gA", "p2P", "bA"], e=["cleat", "boltsP"], cv=["P_e2c"],
             w="Pince e2 = bA − gA − (n2 − 1)·p2 = " + f0(R.b_A) + " − " + f0(R.g_A) + " − " + js_str(R.n2_P - 1) + " × " + f0(u.p2_P) + " = " + f0(R.e2a_P) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. Réduire gA ou p2, ou choisir une aile A plus longue."),
        dict(id="zc_top", c=u.z_C < (u.d_nt if u.d_nt > 0 else tfrS), b=1,
             m="Configuration impossible : le dessus des cornières est dans la zone grugée ou dans le congé supérieur de la poutre portée.",
             f=["z_C", "d_nt"] + prS, d=["zc", "dcs"], e=["cleat", "notchT"],
             w=("zc = " + f0(u.z_C) + " mm est inférieur à la profondeur du grugeage dc,sup = " + f0(u.d_nt) + " mm : il faut zc ≥ dc,sup.") if u.d_nt > 0
             else ("zc = " + f0(u.z_C) + " mm est inférieur à tf + r = " + f0(tfrS) + " mm (semelle et congé de la poutre portée) : il faut zc ≥ " + f0(math.ceil(tfrS)) + " mm.")),
        dict(id="h_dispo", c=sumC > botLim, b=1,
             m="Configuration impossible : hauteur disponible insuffisante – les cornières dépassent la partie droite de l'âme de la poutre portée.",
             f=["LC_u", "z_C"] + (["d_nb"] if u.d_nb > 0 else []) + prS, d=["Lc", "zc"] + (["dci"] if u.d_nb > 0 else []), e=["cleat"],
             w="zc + hc = " + f0(u.z_C) + " + " + f0(R.L_C) + " = " + f0(sumC) + " mm dépasse la limite basse de la partie droite de l'âme : "
             + (("h − dc,inf = " + f0(R.h_S) + " − " + f0(u.d_nb)) if u.d_nb > 0 else ("h − (tf + r) = " + f0(R.h_S) + " − " + f0(tfrS)))
             + " = " + f0(botLim) + " mm. Avec zc = " + f0(u.z_C) + " mm, hc maximal = " + f0(math.floor(botLim - u.z_C)) + " mm."),
        dict(id="fillet", c=cA or cB, b=1,
             m="Configuration impossible : trou de boulon dans le congé de la cornière (trusquinage − d0/2 < tc + rc).",
             f=(["gA_u"] if cA else []) + (["g_h", "e2b_u"] if cB else []) + cornF + ["boulon_u"],
             d=(["gA"] if cA else []) + (["e2b", "gh"] if cB else []),
             e=["cleat"] + (["boltsP"] if cA else []) + (["boltsS"] if cB else []),
             w="Il faut trusquinage − d0/2 ≥ tc + rc = " + f0(R.t_C) + " + " + f0(R.r_C) + " = " + f0(R.t_C + R.r_C) + " mm."
             + ((" Aile A : gA − d0/2 = " + F(R.g_A - d0 / 2, 1) + " mm, donc gA ≥ " + f0(math.ceil(R.t_C + R.r_C + d0 / 2)) + " mm.") if cA else "")
             + ((" Aile B : gh + e2,b − d0/2 = " + F(R.g_B2 - d0 / 2, 1) + " mm, donc e2,b ≥ " + f0(math.ceil(R.t_C + R.r_C + d0 / 2 - u.g_h)) + " mm.") if cB else "")),
        # --- les trois conditions géométriques NEUVES du poteau (étude §4.3, §5.3)
        dict(id="p3_congeP", c=R.bolt_P and R.aile and R.p_3 < R.p3_congeP - 1e-9, b=1,
             m="Configuration impossible : l'écrou chevauche le congé de raccordement du poteau – le gauge p3 est trop petit pour l'aile de ce poteau.",
             f=["gA_u"] + cornF + ["boulon_u"] + prP, d=["p3"], e=["cleat", "boltsP", "colFl"],
             w="Il faut p3 ≥ twP + 2·rP + dw = " + f0(R.tw_P) + " + 2 × " + f0(R.r_P) + " + " + f0(R.d_w)
             + " = " + F(R.p3_congeP, 1) + " mm ; p3 = tw,S + 2·gA = " + F(R.p_3, 1)
             + " mm. gA minimal : " + f0(math.ceil((R.p3_congeP - R.tw_S) / 2))
             + " mm. Gauges usuels d'aile de poteau : 90 / 100 / 120 / 140 mm."),
        dict(id="p3_libreP", c=R.bolt_P and R.ame and R.p_3 / 2 + R.d_w / 2 > R.demi_droit + 1e-9, b=1,
             m="Configuration impossible : cas (b) – l'écrou dépasse la partie droite de l'âme du poteau (il heurte le congé).",
             f=["gA_u"] + cornF + ["boulon_u"] + prP, d=["p3", "libreP"], e=["cleat", "boltsP", "colWeb"],
             w="Il faut p3/2 + dw/2 ≤ (hP − 2·tfP)/2 − rP = " + F(R.demi_droit, 1) + " mm ; actuel "
             + F(R.p_3 / 2, 1) + " + " + F(R.d_w / 2, 1) + " = " + F(R.p_3 / 2 + R.d_w / 2, 1)
             + " mm. p3 maximal : " + F(R.p3_max_ame, 1) + " mm."),
        dict(id="bA_congeP", c=R.ame and R.tw_S / 2 + R.b_A > R.demi_droit + 1e-9, b=1,
             m="Configuration impossible : cas (b) – le bout de l'aile A de la cornière dépasse la partie droite de l'âme du poteau.",
             f=cornF + prP + prS, d=["bA", "libreP"], e=["cleat", "colWeb"],
             w="Il faut tw,S/2 + bA ≤ (hP − 2·tfP)/2 − rP = " + F(R.demi_droit, 1) + " mm ; actuel "
             + F(R.tw_S / 2, 1) + " + " + f0(R.b_A) + " = " + F(R.tw_S / 2 + R.b_A, 1)
             + " mm. Longueur d'aile A maximale : " + F(R.bA_max, 1) + " mm."),
        dict(id="aile_poutre", c=R.ame and not R.rec_ok, b=1,
             m="Configuration impossible : la semelle de la poutre portée ne passe pas entre les ailes du poteau et la recoupe est insuffisante.",
             f=["rec_S", "j_lat", "g_h"] + prS + prP, d=["rec", "jlat", "gh"], e=["beamS", "recut", "col"],
             w="bS = " + f0(R.b_S) + " mm dépasse la largeur libre entre ailes hP − 2·tfP − 2·j = " + f0(R.libre_ailes)
             + " mm : il faut recouper les semelles sur au moins (bP − twP)/2 + j − gh = " + F(R.rec_min, 1)
             + " mm depuis l'about (recoupe actuelle " + f0(u.rec_S) + " mm)."),
        dict(id="one_bolt", c=R.bolt_S and R.n_S == 1, b=1,
             m="Configuration impossible : un seul boulon côté poutre portée ne peut pas reprendre le moment d'excentricité VEd·z.",
             f=["n1S_u", "n2S_u"], d=[], e=["boltsS"], w="n1 × n2 = 1 boulon : ajouter au moins une rangée ou une file."),
        dict(id="d0_manq", c=u.trou == "Surdimensionné" and N(u.d0_u) == 0, b=1,
             m="Trou surdimensionné : saisir le diamètre d0.", f=["trou", "d0_u"], d=[], e=[], w=""),
        dict(id="ame_hp", c=R.ame, b=0,
             m="Appui dans l'ÂME du poteau : la flexion HORS PLAN de l'âme sous la force de chaînage et sous la traction des boulons n'est PAS couverte par l'EN 1993-1-8 pour une âme non raidie — à justifier séparément, ou raidir (raidisseurs entre ailes).",
             f=["appui"] + prP, d=["libreP"], e=["col", "colWeb"], w=""),
        dict(id="n2P_T", c=R.bolt_P and R.aile and R.n2_P > 1 and NT > 0, b=0,
             m="Deux files par cornière et boulons tendus : le tronçon en T de l'AILE DU POTEAU est calculé sur la file la plus proche de l'âme (bras m tiré de p3) ; la seconde file, plus éloignée, n'est pas modélisée — à justifier séparément.",
             f=["n2P_u", "p2_P"], d=["p3", "p2P"], e=["colFl", "boltsP"], w=""),
        dict(id="MEd", c=Mabs != 0, b=0,
             m="MEd ≠ 0 : l'assemblage reste modélisé comme articulé ; MEd est traité comme moment parasite (groupe S : ajouté à VEd·z ; groupe P : traction élastique). Un moment significatif sort du domaine de l'outil.",
             f=["M_Ed"], d=["z"], e=[], w="MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm."),
        dict(id="NEd", c=u.N_Ed > 0 and not R.tying, b=0,
             m="NEd de traction à l'ELU fondamental : repris en cisaillement horizontal des boulons S, en traction des boulons P, par le tronçon en T des ailes A et — en cas (a) — par celui de l'aile du poteau. En cas (b), la flexion hors plan de l'âme du poteau n'est PAS vérifiée.",
             f=["N_Ed", "nat_N"], d=[], e=[], w=""),
        dict(id="N_tying", c=R.tying, b=0,
             m="NEd en TYING (situation accidentelle) : la force de chaînage est vérifiée SEULE, sans VEd, avec les résistances ultimes et γMu (MSB Part 5 §2.3). En cas (a), le tronçon en T de l'aile du poteau et l'âme du poteau en traction transversale s'ajoutent.",
             f=["N_Ed", "nat_N", "g_Mu"], d=[], e=[],
             w="NEd = " + F(u.N_Ed, 1) + " kN ; γMu = " + F(gMu, 2) + "."),
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
    R.Fv_Rd = RS.Fv_Rd(R.a_v, R.f_ub, R.A_cis, gM2)
    R.Fv_Rd_u = RS.Fv_Rd(R.a_v, R.f_ub, R.A_cis, gMu)       # tying (γMu)
    R.Ft_Rd = RS.Ft_Rd(R.f_ub, R.As_b, gM2)
    R.Ft_Rd_u = RS.Ft_Rd(R.f_ub, R.As_b, gMu)
    R.Ft_Rd_g = RS.Ft_Rd(R.f_ub, R.As_b, gB)                # γ de la situation
    # poinçonnement : la CORNIÈRE d'un côté, le POTEAU de l'autre — deux
    # lignes nommées (bt_P et ppP), au lieu du min() du module poutre–poutre
    R.Bp_Rd = RS.Bp_Rd(R.d_m, R.t_C * R.fu_C, gB)
    R.Bp_P = RS.Bp_Rd(R.d_m, R.t_appui * R.fu_P, gB)
    R.Fp_C = RS.Fp_C(R.f_ub, R.As_b)
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
    # traction des boulons P : à l'ELU, NEd / MEd / HEd ; en tying, NEd seul
    R.Ft_elu = (dv(mx(NEd, 0), 2 * R.n_P)
                + (dv(Mabs * 1000 * R.ym_P, 2 * R.n2_P * R.p1_P * R.p1_P * R.n1_P * (R.n1_P * R.n1_P - 1) / 12) if R.n1_P > 1 else 0)
                + dv(H * R.zeff, R.p_3 * R.n_P))
    R.Ft_ty = dv(NT, 2 * R.n_P)
    R.Ft_P = R.Ft_ty if R.tying else R.Ft_elu
    R.tabS = MEC.bolt_table(R.n1_S, R.n2_S, R.p1_S, u.p2_S, V, NEd, R.M_S, R.Ip_S)
    R.tabP = MEC.bolt_table(R.n1_P, R.n2_P, R.p1_P, u.p2_P, V / 2, H / 2, R.M_P, R.Ip_P)
    R.N_par_S = dv(NT, R.n_S)

    def bearing(key, t, fu, e1, p1, n1, e2, p2, n2, Fz, Fx, div, exV=None, exH=None):
        b = MEC.bearing(t, fu, R.f_ub, d, d0, e1, p1, n1, e2, p2, n2, gM2, R.k_trou,
                        exV=exV, exH=exH, Fz=Fz / div, Fx=Fx / div)
        for nom in ("abv", "k1v", "abh", "k1h", "Fbv", "Fbh", "ipd"):
            R[nom + "_" + key] = b[nom]

    bearing("B", R.t_C, R.fu_C, mn(R.e1_S, R.e1bot_S), R.p1_S, R.n1_S, R.e2a_S, u.p2_S, R.n2_S,
            R.Fz_S, R.Fx_S, 2)
    bearing("A", R.t_C, R.fu_C, mn(R.e1_P, R.e1bot_P), R.p1_P, R.n1_P, R.e2a_P, u.p2_P, R.n2_P,
            R.Fz_P, R.Fx_P, 1)
    bearing("S", R.tw_S, R.fu_S, mn(R.e1b_S, R.h_e) if u.d_nb > 0 else R.e1b_S, R.p1_S, R.n1_S,
            R.e2b_S, u.p2_S, R.n2_S, R.Fz_S, R.Fx_S, 1)
    # ---- POTEAU : pression diamétrale sur l'AILE (cas a) ou l'ÂME (cas b)
    # AUCUNE pince d'extrémité verticale (le poteau continue au-dessus et en
    # dessous : e1 = None) ; en cas (a) le seul bord libre est le BOUT D'AILE
    # (e2 = eP), en cas (b) il n'y a aucun bord libre (e2 = None) — étude
    # §4.2 et §5.2. Les termes du gauge viennent de p3, pas de p2.
    R.e_P_fl = (R.b_P - R.p_3) / 2                     # pince au bout d'aile
    bearing("P", R.t_appui, R.fu_P, None, R.p1_P, R.n1_P,
            R.e_P_fl if R.aile else None, u.p2_P, R.n2_P, R.Fz_P, R.Fx_P, 1,
            1.4 * R.p_3 / d0 - 1.7, R.p_3 / (3 * d0) - 0.25)

    def bvals(key):
        return ("αb,ver = " + F(R["abv_" + key], 3) + " ; k1,ver = " + F(R["k1v_" + key], 2)
                + " → Fb,ver,Rd = " + F(R["Fbv_" + key], 1) + " kN ; αb,hor = " + F(R["abh_" + key], 3)
                + " ; k1,hor = " + F(R["k1h_" + key], 2) + " → Fb,hor,Rd = " + F(R["Fbh_" + key], 1) + " kN")

    # ---- aires et résistances des cornières
    R.Ant_B = R.t_C * (R.e2a_S + (R.n2_S - 1) * u.p2_S - (R.n2_S - 0.5) * d0)
    R.Anv_B = R.t_C * (R.L_C - R.e1_S - (R.n1_S - 0.5) * d0)
    R.Ant_A = R.t_C * (R.e2a_P + (R.n2_P - 1) * u.p2_P - (R.n2_P - 0.5) * d0)
    R.Anv_A = R.t_C * (R.L_C - R.e1_P - (R.n1_P - 0.5) * d0)
    # tronçon en T des AILES A (MSB P5 §4.3.1.1) : géométrie du module
    # poutre–poutre, reprise sans changement de formule
    R.gA_min = R.t_C + R.r_C + d0 / 2                  # trou hors du congé de la cornière
    R.e1A_T = mn(R.e1_P, R.e1bot_P, 0.5 * (R.p_3 - R.tw_S - 2 * R.r_C) + d0 / 2)
    R.p1A_T = mn(R.p1_P, R.p_3 - R.tw_S - 2 * R.r_C + d0)
    R.leff_T = 2 * R.e1A_T + (R.n1_P - 1) * R.p1A_T
    R.m_T = (R.p_3 - R.tw_S - 2 * R.t_C - 1.6 * R.r_C) / 2
    R.n_T = mn(R.e2a_P, 1.25 * R.m_T); R.ew_T = R.d_w / 4
    R.SFt_T = 2 * R.n1_P * R.Ft_Rd_g
    tA = MEC.troncon_T(R.leff_T, R.leff_T, R.t_C, R.m_T, R.n_T, R.ew_T,
                       R.fu_C if R.tying else R.fy_C, gB if R.tying else gM0, R.SFt_T)
    R.Mpl_T = tA["Mpl1"] / 1e6                         # kNm
    R.FT_1 = tA["FT1"]; R.FT_2 = tA["FT2"]; R.FT_3 = tA["FT3"]; R.FT_Rd = tA["FT_Rd"]
    eB = mn(R.e1_S, R.e1bot_S); AhB = R.e2a_S + (R.n2_S - 1) * u.p2_S - (R.n2_S - 0.5) * d0
    R.Vt1_B = (dv(R.fu_C * 2 * R.t_C * (R.n1_S - 1) * (R.p1_S - d0), gT)
               + dv(R.fy_C * 4 * R.t_C * AhB / S3, gM0)) / 1000
    R.Vt2_B = (dv(R.fu_C * 2 * R.t_C * (eB + (R.n1_S - 1) * R.p1_S - (R.n1_S - 0.5) * d0), gT)
               + dv(R.fy_C * 2 * R.t_C * AhB / S3, gM0)) / 1000

    # ---- poutre portée (section grugée : brique partagée, MSB P5 §4.2.4 / §4.2.5)
    sg = MEC.section_grugee(dict(h=R.h_S, b=R.b_S, tw=R.tw_S, tf=R.tf_S, r=R.r_S),
                            R.fy_S, u.d_nt, u.d_nb, gM0, V, u.lt_ok == "Oui",
                            " (SCI P358 Check 7)")
    R.cas_g = sg["cas_g"]; R.Asec_S = sg["Asec"]; R.h_T = sg["h_T"]; R.A_Tee = sg["A_Tee"]
    R.Av_S = sg["Av"]; R.VRd_gS = sg["VRd_g"]; R.hw_T = sg["hw_T"]; R.zb_T = sg["zb_T"]
    R.I_N = sg["I_N"]; R.W_N = sg["W_N"]; R.rho_N = sg["rho_N"]; R.Mv_N = sg["Mv_N"]
    R.el_N = sg["el_N"]; R.lim_N = sg["lim_N"]; R.ln_max = sg["ln_max"]; R.val_N = sg["val_N"]
    AhS = R.e2b_S + (R.n2_S - 1) * u.p2_S - (R.n2_S - 0.5) * d0
    R.Ant_S = R.tw_S * AhS
    R.Anv_S = R.tw_S * (R.e1b_S + (R.n1_S - 1) * R.p1_S - (R.n1_S - 0.5) * d0)
    R.VRd_bS = (dv(0.5 * R.fu_S * R.Ant_S, gM2n) + dv(R.fy_S * R.Anv_S / S3, gM0)) / 1000
    Vm = mn(R.VRd_gS, R.VRd_bS)
    R.Mc_2 = dv(R.W_N * R.fy_S, gM0) / 1e6 * (1 if V <= 0.5 * Vm else mx(0, 1 - (2 * dv(V, Vm) - 1) ** 2))
    R.Vt1_S = (dv(R.fu_S * R.tw_S * (R.n1_S - 1) * (R.p1_S - d0), gT)
               + dv(R.fy_S * 2 * R.tw_S * AhS / S3, gM0)) / 1000
    R.Vt2_S = (dv(R.fu_S * R.tw_S * (R.e1b_S + (R.n1_S - 1) * R.p1_S - (R.n1_S - 0.5) * d0), gT)
               + dv(R.fy_S * R.tw_S * AhS / S3, gM0)) / 1000

    # ---- POTEAU : cisaillement local de l'âme (cas b seulement) — un seul
    # ---- côté chargé (SCI P358). Un poteau n'a pas de face horizontale :
    # ---- les bornes zt / zb du module poutre–poutre se réduisent à 5·d
    # ---- (étude §5.1, point tranché P-8).
    R.et_P = 5 * d if R.bolt_P else 0
    R.eb_P = mn(R.p_3 / 2, 5 * d) if R.bolt_P else 0
    R.Av_P = R.tw_P * (R.et_P + (R.n1_P - 1) * R.p1_P + R.eb_P) if R.bolt_P else R.tw_P * R.L_C
    R.VRd_lP = dv(R.Av_P * R.fy_P / S3, gM0) / 1000
    R.VRd_nP = dv((R.Av_P - R.n1_P * d0 * R.tw_P) * R.fu_P / S3, gM2n) / 1000
    # ---- POTEAU : tronçon en T de l'AILE NON RAIDIE (Tab. 6.4) et âme en
    # ---- traction transversale (§6.2.6.3, ω = 1) — cas (a) seulement
    R.m_P = (R.p_3 - R.tw_P) / 2 - 0.8 * R.r_P
    R.n_Pf = mn(R.e_P_fl, 1.25 * R.m_P)
    R.leff1_P, R.leff2_P = MEC.leff_aile_poteau(R.n1_P, R.p1_P, R.m_P, R.e_P_fl)
    tP_ = MEC.troncon_T(R.leff1_P, R.leff2_P, R.tf_P, R.m_P, R.n_Pf, R.ew_T,
                        R.fu_P if R.tying else R.fy_P, gB if R.tying else gM0, R.SFt_T)
    R.Mpl_P = tP_["Mpl1"] / 1e6                        # kNm
    R.FTP_1 = tP_["FT1"]; R.FTP_2 = tP_["FT2"]; R.FTP_3 = tP_["FT3"]; R.FTP_Rd = tP_["FT_Rd"]
    R.Ft_wc = dv(1.0 * R.leff1_P * R.tw_P * (R.fu_P if R.tying else R.fy_P),
                 gB if R.tying else gM0) / 1000

    # ---- soudures
    for X, w1, w2, w3, w4 in (("S", abs(NEd) / 2, 0, R.M_S / 2, "S"),
                              ("P", H / 2, mx(NEd, 0) / 2, R.M_P, "P")):
        Lw = R["Lw_" + X]; Iw = R["Iw_" + X]; lh = u["lh_" + X]; xg = R["xg_" + X]
        R["qz_" + X] = dv(V / 2 * 1000, Lw) + dv(w3 * 1e6 * (lh - xg), Iw)
        R["qx_" + X] = dv(w1 * 1000, Lw) + dv(w3 * 1e6 * (R.L_C / 2), Iw)
        R["qn_" + X] = dv(w2 * 1000, Lw)
        R["qw_" + X] = hyp(R["qz_" + X], R["qx_" + X], R["qn_" + X])
        fuo = R["fu_" + w4]
        R["fuw_" + X], R["bww_" + X] = RS.fu_bw_cordon(R.fu_C, R.bw_C, fuo, R["bw_" + w4])
        R["fvw_" + X] = RS.fvw_d(R["fuw_" + X], R["bww_" + X], gM2)
        R["Fw_" + X] = R["fvw_" + X] * u["a_" + X]
        # tying : la force de chaînage seule, résistance ultime fu/(√3·βw·γMu)
        R["qt_" + X] = dv(NT * 1000, 2 * Lw)
        R["Fwu_" + X] = RS.fvw_d(R["fuw_" + X], R["bww_" + X], gMu) * u["a_" + X]

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
    W = "EN 1993-1-8 §4.5.3.3 ; répartition élastique du groupe de cordons : modèle complémentaire"
    ty = " (tying : fu/γMu)" if R.tying else ""
    nomP = "aile" if R.aile else "âme"

    # --- Boulons
    ck("bv_S", "Boulons", "Cisaillement des boulons – groupe S (2 plans), boulon le plus sollicité",
       R.F_S, 2 * R.Fv_Rd * R.bLf_S, "kN",
       T34 + " ; groupe excentré : " + MSB + " §4.2.1.1", "EC + COMP", R.bolt_S,
       "Fv,Rd = αv·fub·A/γM2 (× 2 plans) ; F,Ed = √(Fz² + Fx²) avec Fz = VEd/n + MS·xmax/Ip et Fx = |NEd|/n + MS·ymax/Ip",
       "z = " + F(R.z_S, 1) + " mm ; MS = " + F(R.M_S, 2) + " kNm ; Ip = " + F(R.Ip_S, 0) + " mm² ; Fz = "
       + F(R.Fz_S, 2) + " kN ; Fx = " + F(R.Fx_S, 2) + " kN ; Fv,Rd = " + F(R.Fv_Rd, 2)
       + " kN/plan ; βLf = " + F(R.bLf_S, 3), 1)
    ck("bv_P", "Boulons", "Cisaillement des boulons – groupe P sur le poteau (1 plan par cornière)",
       R.F_P, u.k_rot * R.Fv_Rd * R.bLf_P, "kN",
       T34 + " ; facteur " + F(u.k_rot, 2) + " : " + MSB + " §4.2.1.2", "EC + COMP", R.bolt_P,
       "F,Ed = (VEd/2)/n par boulon ; FRd = k_rot·Fv,Rd (traction parasite due à la rotation)",
       "n = " + js_str(R.n_P) + " boulons par cornière ; Fz = " + F(R.Fz_P, 2) + " kN ; Fx = "
       + F(R.Fx_P, 2) + " kN ; Fv,Rd = " + F(R.Fv_Rd, 2) + " kN", 1)
    ck("bt_P", "Boulons", "Traction des boulons – groupe P" + ty, R.Ft_P, mn(R.Ft_Rd_g, R.Bp_Rd), "kN",
       T34 + (" avec γMu ; " + MSB_TY if R.tying else ""), "EC", R.bolt_P and R.Ft_P > 0,
       "Ft,Rd = 0,9·fub·As/γ ; Bp,Rd = 0,6·π·dm·tc·fu,C/γ (poinçonnement de l'aile A ; le poteau a sa ligne ppP)",
       "Ft,Ed = " + F(R.Ft_P, 2) + " kN ; Ft,Rd = " + F(R.Ft_Rd_g, 1) + " kN ; Bp,Rd (cornière) = "
       + F(R.Bp_Rd, 1) + " kN ; γ = " + F(gB, 2))
    ck("bi_P", "Boulons", "Interaction cisaillement + traction – groupe P",
       dv(R.F_P, R.Fv_Rd * R.bLf_P) + dv(R.Ft_elu, 1.4 * R.Ft_Rd), 1, "-", T34, "EC",
       R.bolt_P and R.Ft_elu > 0 and not R.tying, "Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1",
       "Fv,Ed = " + F(R.F_P, 2) + " kN ; Ft,Ed = " + F(R.Ft_elu, 2) + " kN")
    ck("gl_S", "Boulons", "Glissement – groupe S (2 surfaces de frottement)", R.F_S * ks,
       dv(u.k_s * 2 * u.mu_s * R.Fp_C, g3), "kN", "EN 1993-1-8 §3.9.1", "EC",
       R.bolt_S and u.cat != "A", "Fs,Rd = ks·n·μ·Fp,C/γM3 ; Fp,C = 0,7·fub·As",
       "Fp,C = " + F(R.Fp_C, 1) + " kN ; catégorie " + u.cat)
    ck("gl_P", "Boulons", "Glissement – groupe P (1 surface de frottement)", R.F_P * ks,
       dv(u.k_s * u.mu_s * (R.Fp_C - 0.8 * R.Ft_elu * ks), g3), "kN",
       "EN 1993-1-8 §3.9.1 et §3.9.2", "EC", R.bolt_P and u.cat != "A",
       "Fs,Rd = ks·n·μ·(Fp,C − 0,8·Ft,Ed)/γM3",
       "Fp,C = " + F(R.Fp_C, 1) + " kN ; Ft,Ed = " + F(R.Ft_elu, 2) + " kN")
    # La force de chaînage est dans le PLAN de l'âme portée : elle CISAILLE le
    # groupe S (étude §7.2) — ligne ajoutée à la liste du §8.2, qui ne la
    # portait pas (voir docs/assemblages/DOUBLES_CORNIERES_POTEAU.md).
    ck("bvT_S", "Boulons", "Tying – boulons du groupe S en cisaillement (force de chaînage seule)",
       R.N_par_S, 2 * R.Fv_Rd_u, "kN", T34 + " avec γMu ; " + MSB_TY, "EC + COMP", R.tying and R.bolt_S,
       "NEd/n ≤ 2·αv·fub·A/γMu — la force de chaînage est dans le PLAN de l'âme portée : elle cisaille les boulons S",
       "NEd = " + F(R.N_ty, 1) + " kN ; n = " + js_str(R.n_S) + " ; γMu = " + F(gMu, 2)
       + " ; n·2·Fv,Rd,u = " + F(2 * R.n_S * R.Fv_Rd_u, 1) + " kN", 1)
    # --- Cornières
    ck("pdB", "Cornières", "Pression diamétrale – ailes B des cornières (groupe S)", R.F_S / 2,
       dv(R.F_S / 2, R.ipd_B) if R.ipd_B > 0 else R.Fbv_B, "kN",
       T34 + " ; interaction : " + MSB + " §4.2.1.1", "EC + COMP", R.bolt_S,
       "Fb,Rd = k1·αb·fu·d·tc/γM2 ; η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²] ; effort par cornière = F/2",
       bvals("B"), 1)
    ck("pdA", "Cornières", "Pression diamétrale – ailes A des cornières (groupe P)", R.F_P,
       dv(R.F_P, R.ipd_A) if R.ipd_A > 0 else R.Fbv_A, "kN", T34, "EC + COMP", R.bolt_P,
       "Fb,Rd = k1·αb·fu·d·tc/γM2", bvals("A"), 1)
    VgC = dv(2 * R.L_C * R.t_C * R.fy_C, 1.27 * S3 * gM0) / 1000
    WC = dv(R.t_C * R.L_C * R.L_C / 6 * R.fy_C, gM0) / 1e6
    ck("cgB", "Cornières", "Ailes B – cisaillement section brute", V, VgC, "kN", shr, "COMP", True,
       "VRd,g = 2·hc·tc·fy/(1,27·√3·γM0)",
       "hc = " + F(R.L_C, 0) + " mm ; tc = " + F(R.t_C, 0) + " mm ; fy = " + F(R.fy_C, 0) + " MPa", 1)
    ck("cnB", "Cornières", "Ailes B – cisaillement section nette", V,
       dv(2 * R.t_C * (R.L_C - R.n1_S * d0) * R.fu_C, S3 * gM2n) / 1000, "kN", shr, "COMP", R.bolt_S,
       "VRd,n = 2·Av,net·fu/(√3·γM2) ; Av,net = tc·(hc − n1·d0)",
       "Av,net = " + F(R.t_C * (R.L_C - R.n1_S * d0), 0) + " mm² par cornière", 1)
    ck("cbB", "Cornières", "Ailes B – rupture de bloc", V,
       2 * (dv(0.5 * R.fu_C * R.Ant_B, gM2n) + dv(R.fy_C * R.Anv_B / S3, gM0)) / 1000, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : " + MSB + " §4.2.2.1", "EC", R.bolt_S,
       "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0) (× 2 cornières)",
       "Ant = " + F(R.Ant_B, 0) + " mm² ; Anv = " + F(R.Anv_B, 0) + " mm²", 1)
    ck("flB", "Cornières", "Ailes B – flexion dans leur plan", V / 2 * R.zeff / 1000 + Mabs / 2, WC, "kNm",
       MSB + " §3.2.3 (plat d'âme) ; EN 1993-1-1 §6.2.5", "COMP", True,
       "MEd = (VEd/2)·z ; MRd = Wel·fy/γM0 avec Wel = tc·hc²/6",
       "z = " + F(R.zeff, 1) + " mm ; Wel = " + F(R.t_C * R.L_C * R.L_C / 6, 0) + " mm³", 1)
    ck("cgA", "Cornières", "Ailes A – cisaillement section brute", V, VgC, "kN", MSB + " §4.2.2.2",
       "COMP", True, "VRd,g = 2·hc·tc·fy/(1,27·√3·γM0)", "")
    ck("cnA", "Cornières", "Ailes A – cisaillement section nette", V,
       dv(2 * R.t_C * (R.L_C - R.n1_P * d0) * R.fu_C, S3 * gM2n) / 1000, "kN", MSB + " §4.2.2.2",
       "COMP", R.bolt_P, "VRd,n = 2·Av,net·fu/(√3·γM2)", "")
    ck("cbA", "Cornières", "Ailes A – rupture de bloc", V,
       2 * (dv(0.5 * R.fu_C * R.Ant_A, gM2n) + dv(R.fy_C * R.Anv_A / S3, gM0)) / 1000, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : " + MSB + " §4.2.2.2", "EC", R.bolt_P,
       "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0) (× 2 cornières)",
       "Ant = " + F(R.Ant_A, 0) + " mm² ; Anv = " + F(R.Anv_A, 0) + " mm²")
    ck("flA", "Cornières", "Ailes A – flexion dans leur plan entre talon et file de boulons (ou cordon)",
       V / 2 * ((R.g_A if R.bolt_P else R.b_A) - R.t_C / 2) / 1000, WC, "kNm",
       "EN 1993-1-1 §6.2.5 ; bras de levier : hypothèse de l'outil", "INT", True,
       "MEd = (VEd/2)·(gA − tc/2) ; MRd = Wel·fy/γM0", "")
    ck("tsA", "Cornières", "Ailes A en flexion + boulons P en traction – tronçon en T équivalent" + ty,
       NT, R.FT_Rd, "kN",
       "EN 1993-1-8 §6.2.4 Tableau 6.2 (méthode 2) ; leff et m : " + MSB + " §4.3.1.1"
       + (" ; " + MSB_TY if R.tying else ""), "EC + COMP", R.bolt_P and NT > 0,
       "FT,1 = (8n − 2ew)·Mpl/[2mn − ew(m + n)] ; FT,2 = (2Mpl + n·ΣFt,Rd)/(m + n) ; FT,3 = ΣFt,Rd",
       "Σleff = " + F(R.leff_T, 0) + " mm ; m = " + F(R.m_T, 1) + " mm ; n = " + F(R.n_T, 1)
       + " mm ; Mpl = " + F(R.Mpl_T, 2) + " kNm ; FT,1 = " + F(R.FT_1, 0) + " ; FT,2 = "
       + F(R.FT_2, 0) + " ; FT,3 = " + F(R.FT_3, 0) + " kN", 1)
    ck("tnB", "Cornières", "Ailes B – traction section nette" + ty, NT,
       dv(2 * 0.9 * R.t_C * (R.L_C - R.n1_S * d0) * R.fu_C, gT) / 1000, "kN",
       "EN 1993-1-1 §6.2.3(2)" + (" ; " + MSB_TY if R.tying else ""), "EC", R.bolt_S and NT > 0,
       "Nu,Rd = 2 × 0,9·Anet·fu/γ", "Anet = " + F(R.t_C * (R.L_C - R.n1_S * d0), 0)
       + " mm² par cornière ; γ = " + F(gT, 2))
    ck("tbB", "Cornières", "Ailes B – rupture de bloc en traction" + ty, NT, mn(R.Vt1_B, R.Vt2_B), "kN",
       "EN 1993-1-8 §3.10.2(2) ; aires : " + MSB + " §4.3.1.4", "EC", R.bolt_S and NT > 0,
       "Neff,1,Rd = fu·Ant/γ + fy·Anv/(√3·γM0)",
       "cas 1 = " + F(R.Vt1_B, 0) + " kN ; cas 2 = " + F(R.Vt2_B, 0) + " kN")
    # --- Poutre portée
    ck("pdS", "Poutre portée", "Pression diamétrale – âme de la poutre portée", R.F_S,
       dv(R.F_S, R.ipd_S) if R.ipd_S > 0 else R.Fbv_S, "kN",
       T34 + " ; interaction : " + MSB + " §4.2.1.1", "EC + COMP", R.bolt_S,
       "Fb,Rd = k1·αb·fu·d·tw/γM2 ; η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²]", bvals("S"), 1)
    ck("vgS", "Poutre portée", "Âme – cisaillement section brute (section réduite si grugée)", V,
       R.VRd_gS, "kN", "EN 1993-1-1 §6.2.6 ; " + MSB + " §4.2.3.1", "EC", True,
       "VRd,g = Av·fy/(√3·γM0)", "Av = " + F(R.Av_S, 0) + " mm²", 1)
    ck("vnS", "Poutre portée", "Âme – cisaillement section nette au droit des boulons", V,
       dv((R.Av_S - R.n1_S * d0 * R.tw_S) * R.fu_S / S3, gM2n) / 1000, "kN",
       MSB + " §4.2.3.1 (ECCS n°126)", "COMP", R.bolt_S,
       "VRd,n = (Av − n1·d0·tw)·fu/(√3·γM2)",
       "Av,net = " + F(R.Av_S - R.n1_S * d0 * R.tw_S, 0) + " mm²", 1)
    ck("vbS", "Poutre portée", "Âme – rupture de bloc", V, R.VRd_bS, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : " + MSB + " §4.2.3.1", "EC", R.bolt_S,
       "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0)",
       "Ant = " + F(R.Ant_S, 0) + " mm² ; Anv = " + F(R.Anv_S, 0) + " mm²", 1)
    ck("mN", "Poutre portée", "Poutre grugée – flexion + cisaillement à l'extrémité du grugeage",
       V * (u.g_h + u.l_n) / 1000 + Mabs, R.Mv_N, "kNm",
       MSB + " §4.2.4 ; SCI P358 Check 5", "COMP", R.cas_g > 0,
       "MEd = VEd·(gh + c) ; Mv,N,Rd = fy·Wel,N/γM0·[1 − (2VEd/Vpl,N,Rd − 1)²] si VEd > 0,5·Vpl,N,Rd",
       "Wel,N = " + F(R.W_N, 0) + " mm³ ; Vpl,N,Rd = " + F(R.VRd_gS, 1) + " kN ; réduction = "
       + F(R.rho_N, 3), 1)
    ck("m2", "Poutre portée", "Poutre grugée – interaction à la 2e file de boulons (n2 = 2)",
       V * (u.g_h + R.e2b_S + u.p2_S) / 1000, R.Mc_2, "kNm", MSB + " §4.2.3.2", "COMP",
       R.cas_g > 0 and R.bolt_S and R.n2_S == 2 and u.l_n > R.e2b_S + u.p2_S,
       "MEd = VEd·(gh + e2,b + p2)", "")
    ck("stN", "Poutre portée", "Stabilité locale du grugeage – longueur c", u.l_n, R.ln_max, "mm",
       MSB + " §4.2.5 ; SCI P358 Check 6", "COMP", R.cas_g > 0,
       "c ≤ h si h/tw ≤ limite ; sinon c ≤ K·h/(h/tw)³",
       "h/tw = " + F(R.el_N, 1) + " ; limite = " + F(R.lim_N, 1) + " ; domaine : " + R.val_N, 1)
    ck("stD", "Poutre portée", "Stabilité locale du grugeage – profondeur", mx(u.d_nt, u.d_nb),
       R.h_S / 2 if R.cas_g == 1 else R.h_S / 5, "mm", MSB + " §4.2.5", "COMP", R.cas_g > 0,
       "dc ≤ h/2 (1 semelle grugée) ; dc ≤ h/5 (2 semelles)", "")
    ck("tnS", "Poutre portée", "Âme – traction section nette" + ty, NT,
       dv(0.9 * R.tw_S * (R.L_C - R.n1_S * d0) * R.fu_S, gT) / 1000, "kN",
       "EN 1993-1-1 §6.2.3(2) ; hauteur participante = hc : " + MSB + " §4.3.2.2"
       + (" ; " + MSB_TY if R.tying else ""), "EC", R.bolt_S and NT > 0,
       "Nu,Rd = 0,9·Anet·fu/γ ; Anet = tw·(hc − n1·d0)",
       "Anet = " + F(R.tw_S * (R.L_C - R.n1_S * d0), 0) + " mm² ; γ = " + F(gT, 2))
    ck("tbS", "Poutre portée", "Âme – rupture de bloc en traction" + ty, NT,
       mn(R.Vt1_S, R.Vt2_S if R.cas_g > 0 else INF), "kN",
       "EN 1993-1-8 §3.10.2(2) ; aires : " + MSB + " §4.3.2.3", "EC", R.bolt_S and NT > 0,
       "Neff,1,Rd = fu·Ant/γ + fy·Anv/(√3·γM0) ; cas 2 (vers le bord libre) seulement si la poutre est grugée",
       "cas 1 = " + F(R.Vt1_S, 0) + " kN"
       + (" ; cas 2 = " + F(R.Vt2_S, 0) + " kN" if R.cas_g > 0 else " ; cas 2 sans objet (poutre non grugée)"))
    # --- Poteau porteur
    ck("pdP", "Poteau porteur", "Pression diamétrale – " + nomP + " du poteau", R.F_P,
       dv(R.F_P, R.ipd_P) if R.ipd_P > 0 else R.Fbv_P, "kN",
       T34 + " ; " + P358_SUP, "EC", R.bolt_P,
       "Fb,Rd = k1·αb·fu,P·d·t/γM2 avec t = tfP (aile) ou twP (âme) ; AUCUNE pince d'extrémité verticale "
       "(le poteau continue) ; k1 du gauge p3" + (" et de la pince au bout d'aile" if R.aile else ""),
       "t = " + F(R.t_appui, 1) + " mm ; p3 = " + F(R.p_3, 1) + " mm"
       + (" ; e = (bP − p3)/2 = " + F(R.e_P_fl, 1) + " mm" if R.aile else " ; aucun bord libre")
       + " ; " + bvals("P"), 1)
    ck("vlP", "Poteau porteur",
       "Âme du poteau – cisaillement local brut (un seul côté chargé)", V / 2, R.VRd_lP, "kN",
       P358_SUP, "COMP", R.ame,
       "VEd/2 ≤ Av·fy,P/(√3·γM0) ; Av = twP·(5d + (n1 − 1)·p1 + min(p3/2 ; 5d)) — un poteau n'a pas de "
       "face horizontale : les bornes zt / zb du module poutre–poutre se réduisent à 5·d",
       "Av = " + F(R.Av_P, 0) + " mm² ; et = " + F(R.et_P, 1) + " mm ; eb = " + F(R.eb_P, 1)
       + " mm ; twP = " + F(R.tw_P, 1) + " mm ; fy,P = " + F(R.fy_P, 0) + " MPa", 1)
    ck("vnP", "Poteau porteur", "Âme du poteau – cisaillement local net", V / 2, R.VRd_nP, "kN",
       P358_SUP, "COMP", R.ame and R.bolt_P,
       "VEd/2 ≤ (Av − n1·d0·twP)·fu,P/(√3·γM2)",
       "Av,net = " + F(R.Av_P - R.n1_P * d0 * R.tw_P, 0) + " mm²")
    ck("tfP", "Poteau porteur", "Aile du poteau en flexion – tronçon en T équivalent" + ty, NT,
       R.FTP_Rd, "kN",
       "EN 1993-1-8 §6.2.4 Tableau 6.2 (méthode 2) ; Σleff : " + TAB64
       + (" ; " + MSB_TY if R.tying else ""), "EC + COMP", R.aile and R.bolt_P and NT > 0,
       "m = (p3 − twP)/2 − 0,8·rP ; e = (bP − p3)/2 ; n = min(e ; 1,25·m) ; Σleff : Tab. 6.4, aile NON "
       "RAIDIE, toutes les rangées INTÉRIEURES (le poteau continue) ; FT,Rd = min(FT,1 ; FT,2 ; FT,3)",
       "m = " + F(R.m_P, 2) + " mm ; e = " + F(R.e_P_fl, 1) + " mm ; n = " + F(R.n_Pf, 3)
       + " mm ; Σleff,1 = " + F(R.leff1_P, 0) + " mm ; Σleff,2 = " + F(R.leff2_P, 0)
       + " mm ; Mpl,1 = " + F(R.Mpl_P, 4) + " kNm ; FT,1 = " + F(R.FTP_1, 0) + " ; FT,2 = "
       + F(R.FTP_2, 0) + " ; FT,3 = " + F(R.FTP_3, 0) + " kN", 1)
    ck("twtP", "Poteau porteur", "Âme du poteau en traction transversale" + ty, NT, R.Ft_wc, "kN",
       "EN 1993-1-8 §6.2.6.3 ; ω = 1 et beff = Σleff de l'aile : interprétation de l'outil",
       "EC + INT", R.aile and R.bolt_P and NT > 0,
       "Ft,wc,Rd = ω·beff,t,wc·twP·f/γ avec ω = 1 (pas de moment d'étage : §1.3) et beff,t,wc = Σleff,1 de l'aile",
       "beff = " + F(R.leff1_P, 0) + " mm ; twP = " + F(R.tw_P, 1) + " mm ; f/γ = "
       + F(dv(R.fu_P if R.tying else R.fy_P, gB if R.tying else gM0), 1) + " MPa")
    ck("ppP", "Poteau porteur", "Poinçonnement de l'" + nomP + " du poteau" + ty, R.Ft_P, R.Bp_P, "kN",
       T34, "EC", R.bolt_P and R.Ft_P > 0,
       "Bp,Rd = 0,6·π·dm·t·fu,P/γ avec t = tfP (aile) ou twP (âme)",
       "dm = " + F(R.d_m, 1) + " mm ; t = " + F(R.t_appui, 1) + " mm ; fu,P = " + F(R.fu_P, 0)
       + " MPa ; γ = " + F(gB, 2))
    # --- Soudures
    for X, lab, act in (("S", "ailes B / âme de la poutre portée", not R.bolt_S),
                        ("P", "ailes A / " + nomP + " du poteau", not R.bolt_P)):
        ck("w" + X, "Soudures", "Cordons " + lab + " – effort par unité de longueur au point critique",
           R["qw_" + X], R["Fw_" + X], "N/mm", W, "EC + COMP", act,
           "Fw,Ed = √(qz² + qx² + qn²) ≤ Fw,Rd = fvw,d·a ; fvw,d = fu/(√3·βw·γM2) avec fu = min(fu cornière ; fu support)",
           "a = " + F(u["a_" + X], 0) + " mm ; ΣL = " + F(R["Lw_" + X], 0) + " mm ; Ip = "
           + F(R["Iw_" + X], 0) + " mm³ ; qz = " + F(R["qz_" + X], 0) + " ; qx = " + F(R["qx_" + X], 0)
           + " ; qn = " + F(R["qn_" + X], 0) + " N/mm ; fvw,d = " + F(R["fvw_" + X], 1) + " MPa", 1)
        ck("wa" + X, "Soudures", "Cordons " + lab + " – gorge minimale", 3, u["a_" + X], "mm",
           "EN 1993-1-8 §4.5.2(2)", "EC", act, "a ≥ 3 mm", "")
        ck("wl" + X, "Soudures", "Cordons " + lab + " – longueur minimale", mx(30, 6 * u["a_" + X]),
           R.L_C, "mm", "EN 1993-1-8 §4.5.1(2)", "EC", act, "L ≥ max(30 mm ; 6a)", "")
        ck("wt" + X, "Soudures", "Cordons " + lab + " – tying (force de chaînage seule)",
           R["qt_" + X], R["Fwu_" + X], "N/mm",
           "EN 1993-1-8 §4.5.3.3 avec γMu ; " + MSB_TY, "EC + INT", act and R.tying,
           "qt = NEd/(2·ΣL) ≤ fu/(√3·βw·γMu)·a",
           "qt = " + F(R["qt_" + X], 0) + " N/mm ; a = " + F(u["a_" + X], 1) + " mm ; γMu = " + F(gMu, 2))

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
#  bolt_table
# =========================================================================
def bolt_table(n1, n2, p1, p2, V, Hh, M, Ip):
    """Efforts par boulon (répartition élastique, inertie polaire) — la
    brique partagée ``mecanique.bolt_table``."""
    return MEC.bolt_table(n1, n2, p1, p2, V, Hh, M, Ip)


def leff_aile_poteau(n1, p1, m, e):
    """Σleff d'une aile de poteau NON RAIDIE (Tableau 6.4) — la brique
    partagée ``mecanique.leff_aile_poteau``, exposée ici parce que le
    benchmark et les suites la lisent sur le moteur du module."""
    return MEC.leff_aile_poteau(n1, p1, m, e)


# =========================================================================
#  predim
# =========================================================================
def predim(u, R, Mabs):
    """Prédimensionnement : pour chaque boulon (plage choisie) et chaque
    nombre de rangées (2 → 8), pinces et entraxes de détail (multiples de 5),
    taux estimés (boulons S et P, pressions diamétrales de l'âme portée et du
    poteau, interaction éventuelle), puis cornière nécessaire.

    Ce que le POTEAU y apporte, et lui seul (étude §12, lot 4) : la pression
    diamétrale de l'aile ou de l'âme (épaisseur ``t_appui``), le gauge
    minimal ``p3 ≥ twP + 2·rP + dw`` sur l'AILE (d'où un ``gA`` minimal), et
    — en cas (b) — le plafond de la longueur d'aile A et du gauge par la
    partie droite de l'âme entre congés."""
    rows = []
    hav = R.h_S - (u.d_nb if u.d_nb > 0 else R.tf_S + R.r_S) - u.z_C
    V = u.V_Ed; H = abs(u.H_Ed)
    ame = u.appui == APPUI_AME
    t_appui = R.tw_P if ame else R.tf_P
    demi_droit = (R.h_P - 2 * R.tf_P) / 2 - R.r_P
    i1 = idx(DB["boulons"], u.pd_dmin); i2 = idx(DB["boulons"], u.pd_dmax)
    imin = min(i1, i2); imax = max(i1, i2)
    Nx = 0.0 if u.nat_N == NAT_TYING else abs(u.N_Ed)
    for i, b in enumerate(DB["boulons"]):
        for n in range(2, 9):
            d = b["d"]; d0 = b["d0"]; A = b["As"] if u.filet == "Oui" else b["A"]
            e1 = ceil5(u.k_e1 * d0); p1 = ceil5(u.k_p1 * d0); e2 = ceil5(u.k_e2 * d0)
            Lc = 2 * e1 + (n - 1) * p1
            geom = bool(Lc <= hav and imin <= i <= imax)
            Fv = dv(R.a_v * R.f_ub * A, u.g_M2) / 1000; Lj = (n - 1) * p1
            bLf = mx(0.75, 1 - (Lj - 15 * d) / (200 * d)) if (u.opt_blf == "Oui" and Lj > 15 * d) else 1
            Mz = V * (u.g_h + e2) / 1000 + Mabs; Fz = V / n
            Fx = Nx / n + dv(Mz * 1000 * 6, p1 * n * (n + 1)); Fs = hyp(Fz, Fx)
            e1b = u.z_C + e1 - u.d_nt
            Fbv = (mn(2.8 * e2 / d0 - 1.7, 2.5) * mn(e1b / (3 * d0), p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            Fbh = (mn(2.8 * e1b / d0 - 1.7, 1.4 * p1 / d0 - 1.7, 2.5) * mn(e2 / (3 * d0), dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            FP = hyp(V / 2 / n, H / 2 / n)
            # gauge p3 proposé : le plus grand des trois minima (usuel 90,
            # Tab. 3.3, dégagement du congé du poteau sur l'AILE), au 5 supérieur
            p3min = mx(90.0, 2.4 * d0, (R.tw_P + 2 * R.r_P + b["dw"]) if not ame else 0.0)
            p3 = ceil5(p3min)
            if ame:
                p3 = mn(p3, 5 * math.floor(2 * (demi_droit - b["dw"] / 2) / 5))
            FbP = (mn(1.4 * p3 / d0 - 1.7, 2.5) * mn(p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_P), 1)
                   * R.fu_P * d * t_appui / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            e = AttrDict(
                bS=dv(Fs, 2 * Fv * bLf) if R.bolt_S else 0,
                pdS=MEC.interaction_pd(Fz, Fx, Fbv, Fbh) if R.bolt_S else 0,
                bP=dv(FP, u.k_rot * Fv * bLf) if R.bolt_P else 0,
                pdP=dv(FP, FbP) if R.bolt_P else 0,
                vt=(dv(FP, Fv * bLf) + dv(u.N_Ed / (2 * n), dv(1.4 * 0.9 * R.f_ub * b["As"], u.g_M2) / 1000))
                if (R.bolt_P and Nx > 0) else 0)
            eta = mx(e.bS, e.pdS, e.bP, e.pdP, e.vt); ok = bool(geom and eta <= u.eta_c)
            rows.append(AttrDict(b=b, i=i, n=n, e1=e1, p1=p1, e2=e2, Lc=Lc, p3=p3, geom=geom,
                                 Fz=Fz, Fx=Fx, e=e, eta=eta, ok=ok,
                                 score=(n * 100 + i + 1) if ok else 9999,
                                 etag=eta if geom else 99, prop=None))
    found = any(r.ok for r in rows); key = "score" if found else "etag"; pick = rows[0]
    for r in rows:
        if r[key] < pick[key]:
            pick = r
    gM0 = u.g_M0; gM2n = u.g_M2n; anyB = R.bolt_S or R.bolt_P
    legs = [a[0] for a in DB["ailes"]]
    # en cas (b), l'aile A ne peut pas dépasser la partie droite de l'âme
    leg_max = (demi_droit - R.tw_S / 2) if ame else INF

    def cleat_for(pk):
        d0 = pk.b["d0"]; d = pk.b["d"]; n = pk.n; Lc = pk.Lc
        fbv = mn(2.8 * pk.e2 / d0 - 1.7, 2.5) * mn(pk.e1 / (3 * d0), pk.p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_C), 1) * R.fu_C * d
        fbv = dv(fbv, u.g_M2) / 1000 * R.k_trou
        fbh = mn(2.8 * pk.e1 / d0 - 1.7, 1.4 * pk.p1 / d0 - 1.7, 2.5) * mn(pk.e2 / (3 * d0), dv(R.f_ub, R.fu_C), 1) * R.fu_C * d
        fbh = dv(fbh, u.g_M2) / 1000 * R.k_trou
        reqs = [["Pression diamétrale ailes B (effort par cornière = F/2)",
                 hyp(dv(pk.Fz / 2, fbv), dv(pk.Fx / 2, fbh)) if R.bolt_S else 0],
                ["Pression diamétrale ailes A", hyp(dv(V / 2 / n, fbv), dv(H / 2 / n, fbh)) if R.bolt_P else 0],
                ["Cisaillement section brute", dv(V * 1000 * 1.27 * S3 * gM0, 2 * Lc * R.fy_C)],
                ["Cisaillement section nette", dv(V * 1000 * S3 * gM2n, 2 * (Lc - n * d0) * R.fu_C) if anyB else 0],
                ["Rupture de bloc", dv(V * 1000, 2 * (dv(0.5 * R.fu_C * (pk.e2 - 0.5 * d0), gM2n)
                                                      + dv(R.fy_C * (Lc - pk.e1 - (n - 0.5) * d0) / S3, gM0))) if anyB else 0],
                ["Flexion des ailes B dans leur plan",
                 dv((V / 2 * (u.g_h + pk.e2) / 1000 + Mabs / 2) * 1e6 * 6 * gM0, Lc * Lc * R.fy_C)],
                ["Traction section nette (NEd > 0)",
                 dv(u.N_Ed * 1000 * gM2n, 2 * 0.9 * (Lc - n * d0) * R.fu_C) if (R.bolt_S and u.N_Ed > 0) else 0]]
        treq = dv(mx(*[r[1] for r in reqs]), u.eta_c)
        dw = pk.b["dw"]
        # gA : il dégage le congé de la cornière ET porte le gauge p3 retenu.
        # Le gauge est ENCADRÉ : au MINIMUM 90, 2,4·d0 et — sur l'AILE — le
        # dégagement du congé du poteau twP + 2·rP + dw (étude §4.3) ; au
        # MAXIMUM min(14·t ; 200) du Tableau 3.3, où t est la plus mince des
        # pièces extérieures (l'aile A de la cornière), et — en cas (b) — la
        # partie droite de l'âme entre congés. Une cornière MINCE ferme donc
        # la fenêtre : l'épaisseur est relevée jusqu'à ce que le gauge minimal
        # tienne sous le maximum.
        p3min = mx(90.0, 2.4 * pk.b["d0"], 0.0 if ame else R.tw_P + 2 * R.r_P + dw)

        def pour_tC(t):
            req1 = mx(ceil5(t + 12 + dw / 2 + 2) + pk.e2, u.g_h + 2 * pk.e2)
            r1 = DB["ailes"][_first_ge(legs, req1)][1]
            gA_c = ceil5(t + r1 + dw / 2 + 2)
            p3m = mn(14 * mn(t, t_appui), 200.0)
            if ame:
                p3m = mn(p3m, 2 * (demi_droit - dw / 2))
            g = mx(gA_c, ceil5((p3min - R.tw_S) / 2))
            g_max = 5 * math.floor(mx(0.0, (p3m - R.tw_S) / 2) / 5)
            if g > g_max:                          # le plafond du Tableau 3.3 l'emporte
                g = mx(gA_c, g_max)
            return g, R.tw_S + 2 * g, p3m

        cands = [t for t in DB["epais"] if t >= treq] or [DB["epais"][-1]]
        tC = cands[0]
        gA, p3, p3max = pour_tC(tC)
        for t in cands:
            g, pp3, pm = pour_tC(t)
            if p3min - 1e-9 <= pp3 <= pm + 1e-9:
                tC, gA, p3, p3max = t, g, pp3, pm
                break
        legreq = mx(gA + pk.e2, u.g_h + 2 * pk.e2)
        il = _first_ge(legs, legreq)
        if legs[il] > leg_max:                     # cas (b) : plafonné par l'âme du poteau
            dessous = [j for j, x in enumerate(legs) if x <= leg_max]
            il = dessous[-1] if dessous else 0
        return AttrDict(reqs=reqs, treq=treq, tC=tC, gA=gA, p3=p3, p3min=p3min, p3max=p3max,
                        legreq=legreq, leg=DB["ailes"][il][0], rC=DB["ailes"][il][1])

    for r in rows:
        if r.geom:
            r.prop = cleat_for(r)
    if not pick.prop:
        pick.prop = cleat_for(pick)
    pp = pick.prop
    if not found:
        msg = ("AUCUNE combinaison ne respecte le taux cible dans les limites fixées : la moins "
               "sollicitée est affichée. Élargir la plage de diamètres, la hauteur disponible ou la classe.")
    elif pp.treq > 15:
        msg = ("Épaisseur nécessaire > 15 mm : hors gamme. Augmenter le nombre de rangées ou la "
               "nuance des cornières.")
    elif ame and pp.leg + 1e-9 < pp.legreq:
        msg = ("Cas (b) : l'aile A est plafonnée par la partie droite de l'âme du poteau "
               "(longueur maximale " + F(demi_droit - R.tw_S / 2, 1) + " mm) et l'aile proposée, "
               + F(pp.leg, 0) + " mm, ne porte plus gA + e2 = " + F(pp.legreq, 0)
               + " mm : réduire e2,b ou gh, ou choisir un diamètre plus petit.")
    elif pp.p3 > pp.p3max + 1e-9 or pp.p3 < pp.p3min - 1e-9:
        msg = ("Aucun gauge p3 ne tient à la fois son minimum (" + F(pp.p3min, 1) + " mm"
               + ("" if ame else " : dégagement du congé du poteau twP + 2·rP + dw")
               + ") et le maximum min(14·tc ; 200 mm) du Tableau 3.3 (" + F(pp.p3max, 1)
               + " mm) : épaissir les cornières ou changer de diamètre.")
    else:
        msg = "Proposition complète"
    return AttrDict(rows=rows, hav=hav, found=found, pick=pick, boulon=pick.b["n"], n=pick.n,
                    e1=pick.e1, p1=pick.p1, e2=pick.e2, LC=pick.Lc, eta=pick.eta, reqs=pp.reqs,
                    treq=pp.treq, tC=pp.tC, gA=pp.gA, p3=pp.p3, p3min=pp.p3min, p3max=pp.p3max,
                    legreq=pp.legreq, leg=pp.leg, rC=pp.rC, msg=msg)


# =========================================================================
#  apply_solution / solution_label
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
    """« M20, 4 rangées, L100x100x10 »."""
    return (r.b["n"] + ", " + js_str(r.n) + " rangées, L" + js_str(r.prop.leg) + "x"
            + js_str(r.prop.leg) + "x" + js_str(r.prop.tC))


def nom_cornière_proposee(r):
    """« L100x100x10 » pour une ligne du prédimensionnement."""
    return "L" + js_str(r.prop.leg) + "x" + js_str(r.prop.leg) + "x" + js_str(r.prop.tC)

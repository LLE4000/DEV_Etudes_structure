# -*- coding: utf-8 -*-
"""Moteur de calcul — assemblage poutre–poutre par plat d'âme soudé (fin
plate). Écrit pour CET assemblage : rotule à la face d'appui, un seul plan
de cisaillement, moment d'excentricité VEd·z repris par le groupe de
boulons, le plat, la soudure et l'âme porteuse.

Modèles et sources, distingués vérification par vérification (champ ``nat``) :
- ``EC``   : formule directe de l'EN 1993-1-8 / EN 1993-1-1 ;
- ``COMP`` : modèle complémentaire d'un guide reconnu — « Steel Buildings in
             Europe, Part 5: Joint Design » §3 (fin plate), SCI P358,
             ECCS n°126 ; le déversement du plat long suit SCI P358
             (λLT = 2,8·√(z·hp/(1,5·tp²)), courbe de l'Annexe B.2.1 de
             BS 5950-1, celle de sa Table 17, avec E = 205 000 MPa) ;
- ``INT``  : interprétation de l'outil, nommée comme telle.

Coefficients partiels par défaut : ANB belge pour l'EN 1993-1-1 (γM1 = 1,10
au lieu de 1,00 recommandé — il ne sert qu'au déversement du plat long) ;
valeurs recommandées pour l'EN 1993-1-8. Tous modifiables.

Unités : mm, mm², mm³, MPa, kN, kNm, N/mm (cordons).
"""
import math

from acier.bibliotheques import DB, PERSO, find, idx
from acier.js import AttrDict, INF, N, dv, mn, mx, js_round, ceil5, hyp, js_str
from acier.formats import F, f0
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
def compute(u0=None):
    """Calcul complet pour les saisies ``u0`` (clés absentes → défauts).
    Retourne ``R`` : scalaires du modèle, ``u``, ``checks`` / ``ck``,
    ``alerts``, ``dist`` (Tableau 3.3), ``tab`` (efforts par boulon),
    ``pd`` (prédimensionnement), ``gov``, ``eta_max``, ``verified``,
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
    for X in ("P", "S"):
        if u["prof_" + X] == PERSO:
            p = dict(h=N(u["h" + X + "_u"]), b=N(u["b" + X + "_u"]),
                     tw=N(u["tw" + X + "_u"]), tf=N(u["tf" + X + "_u"]),
                     r=N(u["r" + X + "_u"]))
        else:
            p = find(DB["profils"], u["prof_" + X])
        for d in ("h", "b", "tw", "tf", "r"):
            R[d + "_" + X] = p[d]
    for X, cle in (("P", "nu_P"), ("S", "nu_S"), ("L", "nu_pl")):
        if u[cle] == PERSO:
            a = dict(fy=N(u.fy_u), fu=N(u.fu_u), bw=N(u.bw_u))
        else:
            a = find(DB["aciers"], u[cle])
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

    R.n_1 = ret(u.n1_u, pd.n); R.n_2 = ret(u.n2_u, 1); R.p_1 = ret(u.p1_u, pd.p1)
    R.e_1 = ret(u.e1_u, pd.e1); R.e2b = ret(u.e2b_u, pd.e2)
    R.h_p = ret(u.hp_u, pd.hp); R.t_p = ret(u.tp_u, pd.tp)
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
    R.rec_tp = R.t_p <= 0.5 * d + 1e-9          # ductilité (P358 / MSB P5)
    R.long_p = R.z_p > dv(R.t_p, 0.15)          # plat LONG (P358 / MSB P5)

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
    # gorge de pleine résistance vis-à-vis du plat (dérivation §4.5.3.3 :
    # 2·a·fvw,d ≥ tp·fy/(√3·γM0)) — recommandation, contrôlée par alerte
    R.fuw, R.bww = RS.fu_bw_cordon(R.fu_L, R.bw_L, R.fu_P, R.bw_P)
    R.a_plein = dv(R.t_p * R.fy_L * R.bww * gM2, 2 * R.fuw * gM0)
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
             w=("zp = " + f0(u.z_C) + " mm est inférieur à la profondeur du grugeage dnt = " + f0(u.d_nt) + " mm : il faut zp ≥ dnt.") if u.d_nt > 0
             else ("zp = " + f0(u.z_C) + " mm est inférieur à tf + r = " + f0(tfrS) + " mm (semelle et congé de la poutre secondaire) : il faut zp ≥ " + f0(math.ceil(tfrS)) + " mm.")),
        dict(id="h_dispo", c=sumC > botLim, b=1,
             m="Configuration impossible : hauteur disponible insuffisante – le plat dépasse la partie droite de l'âme de la poutre secondaire.",
             f=["hp_u", "z_C"] + (["d_nb"] if u.d_nb > 0 else []) + prS, d=["hp", "zc"] + (["dnb"] if u.d_nb > 0 else []), e=["plate"],
             w="zp + hp = " + f0(u.z_C) + " + " + f0(R.h_p) + " = " + f0(sumC) + " mm dépasse la limite basse de la partie droite de l'âme : "
             + (("h − dnb = " + f0(R.h_S) + " − " + f0(u.d_nb)) if u.d_nb > 0 else ("h − (tf + r) = " + f0(R.h_S) + " − " + f0(tfrS)))
             + " = " + f0(botLim) + " mm. Avec zp = " + f0(u.z_C) + " mm, hp maximal = " + f0(math.floor(botLim - u.z_C)) + " mm."),
        dict(id="pl_web", c=R.zt_pl < tfrP or R.zt_pl + R.h_p > R.h_P - tfrP, b=1,
             m="Configuration impossible : le plat (et sa soudure) sort de la partie droite de l'âme de la poutre principale.",
             f=["z_C", "d_top", "hp_u"] + prP, d=["ztp", "hp", "zc", "dtop"], e=["plate", "flPt", "flPb"],
             w="Il faut tf + r = " + f0(tfrP) + " mm entre le plat et chaque semelle de la principale. Haut : décalage + zp = " + F(R.zt_pl, 1) + " mm ; bas : h − (décalage + zp + hp) = " + F(R.h_P - R.zt_pl - R.h_p, 1) + " mm."),
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
        dict(id="tp_duct", c=not R.rec_tp, b=0,
             m="Épaisseur du plat tp > 0,5·d : la ductilité de l'attache (rotation par ovalisation des trous) n'est plus assurée — recommandation SCI P358 / MSB Part 5 non respectée.",
             f=["tp_u", "boulon_u"], d=["tp"], e=["plate"],
             w="tp = " + f0(R.t_p) + " mm > 0,5·d = " + F(0.5 * d, 1) + " mm."),
        dict(id="hp_rec", c=not R.rec_hp, b=0,
             m="hp < 0,6·h de la poutre portée : recommandation des guides (maintien en torsion de la poutre à l'appui) non respectée.",
             f=["hp_u"] + prS, d=["hp"], e=["plate"],
             w="hp = " + f0(R.h_p) + " mm < 0,6·h = " + F(0.6 * R.h_S, 0) + " mm."),
        dict(id="long_nr", c=R.long_p and u.lt_ok != "Oui", b=0,
             m="Plat LONG (z > tp/0,15) et poutre non maintenue au déversement : le déversement du plat est vérifié, mais la stabilité d'ensemble demande un examen particulier (SCI P358).",
             f=["lt_ok", "tp_u", "e2b_u", "g_h"], d=["z"], e=["plate"],
             w="z = " + F(R.z_p, 1) + " mm > tp/0,15 = " + F(dv(R.t_p, 0.15), 1) + " mm."),
        dict(id="a_rec", c=R.a_w < R.a_plein - 1e-9, b=0,
             m="Gorge inférieure à la PLEINE RÉSISTANCE recommandée vis-à-vis du plat — admissible si la vérification en contrainte du cordon passe, mais le détail courant soude le plat à pleine résistance.",
             f=["a_w", "tp_u"], d=["aw"], e=["weld"],
             w="Recommandé : a ≥ tp·fy·βw·γM2/(2·fu·γM0) = " + F(R.a_plein, 1) + " mm (dérivation EN 1993-1-8 §4.5.3.3) ; actuel : a = " + f0(R.a_w) + " mm."),
        dict(id="MEd", c=Mabs != 0, b=0,
             m="MEd ≠ 0 : l'assemblage reste modélisé comme articulé ; MEd est ajouté au moment d'excentricité VEd·z. Un moment significatif sort du domaine de l'outil.",
             f=["M_Ed"], d=["z"], e=[], w="MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm."),
        dict(id="NEd", c=NEd > 0, b=0,
             m="NEd de traction : repris en cisaillement horizontal des boulons et pressions diamétrales ; la flexion hors plan de l'âme porteuse sous NEd n'est PAS vérifiée (à justifier séparément).",
             f=["N_Ed"], d=[], e=[], w=""),
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
    R.lam_LT = 2.8 * math.sqrt(dv(R.z_p * R.h_p, 1.5 * R.t_p * R.t_p)) if R.t_p > 0 else INF
    R.pb_LT = pb_bs5950(R.lam_LT, R.fy_L)
    R.Mb_p = dv(R.Wel_p * R.pb_LT, gM1) / 1e6

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
        R.Av_S = R.tw_S * R.h_T
    R.VRd_gS = dv(R.Av_S * R.fy_S / S3, gM0) / 1000
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
    R.el_N = dv(R.h_S, R.tw_S); R.lim_N = 54.3 if R.fy_S <= 275 else 48
    R.ln_max = R.h_S if R.el_N <= R.lim_N else dv((160000 if R.fy_S <= 275 else 110000) * R.h_S, R.el_N ** 3)
    if R.cas_g == 0:
        R.val_N = "Sans objet (poutre non grugée)"
    elif u.lt_ok != "Oui":
        R.val_N = "NON VALIDE : poutre non maintenue au déversement – stabilité d'ensemble de la poutre grugée à justifier"
    elif R.fy_S > 355:
        R.val_N = "NON VALIDE : nuance > S355 – règle non publiée pour cette nuance"
    elif R.fy_S < 275:
        R.val_N = "S235 : limites S275 appliquées (côté sécurité) – interprétation"
    else:
        R.val_N = "Valide"

    # ---- poutre principale (cisaillement local, un seul côté chargé)
    R.Av_P = R.tw_P * R.h_p

    # ---- soudure (deux cordons verticaux pleine hauteur, gorge a chacun)
    R.qz_w = dv(V * 1000, 2 * R.h_p)
    R.qx_w = dv(3 * R.M_S * 1e6, R.h_p * R.h_p) + dv(abs(NEd) * 1000, 2 * R.h_p)
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
       "MSB Part 5 §3.2.3 ; EN 1993-1-1 §6.2.5", "COMP", not R.court,
       "Sans objet si hp ≥ 2,73·z ; sinon MEd = VEd·z ≤ Wel·fy/γM0 avec Wel = tp·hp²/6",
       "hp = " + F(R.h_p, 0) + " < 2,73·z = " + F(2.73 * R.z_p, 1) + " mm ; Wel = " + F(R.Wel_p, 0) + " mm³", 1)
    ck("plt", "Plat d'âme", "Déversement du plat LONG (z > tp/0,15)", R.M_S, R.Mb_p, "kNm",
       "SCI P358 (plats longs) ; courbe : BS 5950-1 Annexe B.2.1 (Table 17), E = 205 000 MPa ; γM1 ANB",
       "COMP", R.long_p,
       "Mb,Rd = Wel·pb(λLT)/γM1 ; λLT = 2,8·√(z·hp/(1,5·tp²))",
       "λLT = " + F(R.lam_LT, 1) + " ; pb = " + F(R.pb_LT, 1) + " MPa ; Wel = " + F(R.Wel_p, 0) + " mm³ ; γM1 = " + F(gM1, 2), 1)
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
       V * (u.g_h + u.l_n) / 1000 + Mabs, R.Mv_N, "kNm",
       "MSB Part 5 §4.2.4 (mêmes règles pour le grugeage)", "COMP", R.cas_g > 0,
       "MEd = VEd·(gh + ln) ; Mv,N,Rd = fy·Wel,N/γM0·[1 − (2VEd/Vpl,N,Rd − 1)²] si VEd > 0,5·Vpl,N,Rd",
       "Wel,N = " + F(R.W_N, 0) + " mm³ ; Vpl,N,Rd = " + F(R.VRd_gS, 1) + " kN ; réduction = " + F(R.rho_N, 3), 1)
    ck("m2", "Poutre secondaire", "Poutre grugée – interaction à la 2e file de boulons (n2 = 2)",
       V * (u.g_h + R.e2b + u.p2_u) / 1000, R.Mc_2, "kNm", "MSB Part 5 §3.2.5", "COMP",
       R.cas_g > 0 and R.n_2 == 2 and u.l_n > R.e2b + u.p2_u, "MEd = VEd·(gh + e2,b + p2)", "")
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
    ck("vlP", "Poutre principale", "Cisaillement local de l'âme porteuse (un seul côté chargé)", V / 2,
       dv(R.Av_P * R.fy_P / S3, gM0) / 1000, "kN",
       "SCI P358 (âme support chargée d'un seul côté) — même modèle que le module doubles cornières", "COMP", True,
       "VEd/2 ≤ Av·fy/(√3·γM0) ; Av = tw·hp", "Av = " + F(R.Av_P, 0) + " mm²", 1)
    ck("w", "Soudure", "Cordons plat / âme porteuse – effort par unité de longueur au point critique", R.qw, R.Fw, "N/mm",
       W445, "EC", True,
       "Fw,Ed = √(qz² + qx²) ≤ Fw,Rd = fvw,d·a ; qz = VEd/(2hp) ; qx = 3·MS/hp² + |NEd|/(2hp) ; fvw,d = fu/(√3·βw·γM2)",
       "a = " + F(R.a_w, 0) + " mm ; qz = " + F(R.qz_w, 0) + " ; qx = " + F(R.qx_w, 0) + " N/mm ; fvw,d = " + F(R.fvw, 1) + " MPa", 1)
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
    R.statut = "ASSEMBLAGE VÉRIFIÉ" if R.verified else "ASSEMBLAGE NON VÉRIFIÉ"
    R.reserve = bool(R.cas_g > 0 and R.val_N.startswith("NON VALIDE"))
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
    nécessaire (tp par les résistances, plafonné à 0,5·d — ductilité) et
    gorge de pleine résistance."""
    rows = []
    hav = R.h_S - (u.d_nb if u.d_nb > 0 else R.tf_S + R.r_S) - u.z_C
    V = u.V_Ed
    i1 = idx(DB["boulons"], u.pd_dmin); i2 = idx(DB["boulons"], u.pd_dmax)
    imin = min(i1, i2); imax = max(i1, i2)
    for i, b in enumerate(DB["boulons"]):
        for n in range(2, 9):
          d = b["d"]; d0 = b["d0"]; A = b["As"] if u.filet == "Oui" else b["A"]
          # deux jeux de pinces : « détail » (k·d0 arrondi au 5 supérieur) et
          # « compacte » (au 5 inférieur, plancher normatif du Tableau 3.3) —
          # en cisaillement simple, la variante compacte débloque souvent la
          # hauteur disponible
          jd = (ceil5(u.k_e1 * d0), ceil5(u.k_p1 * d0), ceil5(u.k_e2 * d0))
          jc = (mx(ceil5(1.2 * d0), 5 * math.floor(u.k_e1 * d0 / 5)),
                mx(ceil5(2.2 * d0), 5 * math.floor(u.k_p1 * d0 / 5)),
                mx(ceil5(1.2 * d0), 5 * math.floor(u.k_e2 * d0 / 5)))
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
                                 score=(n * 100 + i + 1) if ok else 9999,
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
        # ductilité : tp ≤ 0,5·d — à défaut, l'épaisseur candidate suivante
        cand = [t for t in DB["epais"] if t <= 0.5 * d + 1e-9] or [DB["epais"][0]]
        tp = cand[_first_ge(cand, treq)]
        bp = ceil5(u.g_h + pk.e2 * 2)
        a = mx(3, math.ceil(dv(tp * R.fy_L * R.bw_L * u.g_M2, 2 * mn(R.fu_L, R.fu_P) * gM0)))
        return AttrDict(reqs=reqs, treq=treq, tp=tp, bp=bp, a=a)

    for r in rows:
        if r.geom:
            r.prop = plate_for(r)
    if not pick.prop:
        pick.prop = plate_for(pick)
    pp = pick.prop
    if not found:
        msg = ("AUCUNE combinaison ne respecte le taux cible dans les limites fixées : la moins "
               "sollicitée est affichée. Élargir la plage de diamètres, la hauteur disponible ou la classe.")
    elif pp.treq > 0.5 * pick.b["d"]:
        msg = ("Épaisseur nécessaire > 0,5·d (ductilité) : augmenter le nombre de rangées, le "
               "diamètre ou la nuance du plat.")
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
    return (r.b["n"] + ", " + js_str(r.n) + " rangées, plat " + js_str(r.hp) + " × "
            + js_str(r.prop.tp))

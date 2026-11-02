# -*- coding: utf-8 -*-
"""Moteur de calcul — assemblage POUTRE–POTEAU par plat d'âme soudé (fin
plate). Écrit pour CET assemblage : rotule à la face d'appui du poteau, un
seul plan de cisaillement, moment d'excentricité VEd·z repris par le groupe
de boulons, le plat et la soudure.

DEUX cas d'appui, et c'est la clé ``appui`` qui commande toute la mécanique
(`docs/assemblages/POUTRE_POTEAU.md` §1.2) :

- **cas (a), sur l'AILE** : le plat est soudé sur la face extérieure d'une
  aile, AU DROIT de l'âme du poteau. L'attache vit DANS LE PLAN de cette
  âme : VEd et VEd·z y cheminent en plan. Aucune vérification locale de
  l'aile n'est demandée par MSB P5 §3.2 ni par SCI P358 au-delà du cordon —
  le groupe « Poteau porteur » est donc VIDE, et l'écran le dit (§4.4) ;
- **cas (b), dans l'ÂME** : le plat est soudé sur l'âme, entre les ailes.
  L'âme du poteau joue le rôle de l'âme de la poutre porteuse des modules
  poutre–poutre — cisaillement local d'un élément support chargé d'un seul
  côté (SCI P358). La flexion HORS PLAN de l'âme sous qx = 3·MS/hp² et sous
  la force de chaînage n'est PAS couverte par l'EN 1993-1-8 pour une âme non
  raidie : elle est déclarée en limite (§5.4), elle n'est pas transposée.

Modèles et sources, distingués vérification par vérification (champ ``nat``) :
- ``EC``   : formule directe de l'EN 1993-1-8 / EN 1993-1-1 ;
- ``COMP`` : modèle complémentaire d'un guide reconnu — « Steel Buildings in
             Europe, Part 5: Joint Design » §3 (fin plate), SCI P358,
             ECCS n°126 ; le déversement du plat long suit SCI P358
             (λLT = 2,8·√(z·hp/(1,5·tp²)), courbe de l'Annexe B.2.1 de
             BS 5950-1, celle de sa Table 17, avec E = 205 000 MPa) ;
- ``INT``  : interprétation de l'outil, nommée comme telle.

Les briques de calcul communes à la famille ne sont PAS recopiées : elles
viennent de ``acier/assemblages/mecanique.py`` (pression diamétrale, groupe
excentré, poutre grugée, déversement du plat long, gorge de pleine
résistance), que ``tests/test_assemblages_mecanique.py`` lie aux moteurs
poutre–poutre.

Coefficients partiels par défaut : ANB belge pour l'EN 1993-1-1 (γM1 = 1,10
au lieu de 1,00 recommandé — il ne sert qu'au déversement du plat long) ;
valeurs recommandées pour l'EN 1993-1-8 ; γMu = 1,1 du guide pour le tying.
Tous modifiables.

Unités : mm, mm², mm³, MPa, kN, kNm, N/mm (cordons).
"""
import math

from acier.bibliotheques import DB, PERSO, find, idx
from acier.js import AttrDict, INF, N, dv, mn, mx, js_round, ceil5, hyp, js_str
from acier.formats import F, f0
from acier.resultats import Verification, Alerte, Pince
from acier import resistances as RS
from acier.assemblages import mecanique as MEC
from .. import textes as TXT
from .entrees import defaults, NUM, MODE_PREDIM, APPUI_AME, NAT_TYING

S3 = math.sqrt(3)
T34 = "EN 1993-1-8 Tableau 3.4"
MSB3 = "MSB Part 5 §3.2 (fin plate)"
MSB_TY = "MSB Part 5 §2.3 (tying)"
P358_SUP = "SCI P358 (élément support chargé d'un seul côté)"

GROUPES_VERIF = ("Boulons", "Plat d'âme", "Poutre portée", "Poteau porteur",
                 "Soudure")


def _profil_P(R):
    """Les cinq dimensions du POTEAU en dictionnaire, comme l'appel de
    ``mecanique.section_grugee`` le fait pour la poutre portée — la
    géométrie du poteau se lit à plusieurs endroits (``compute`` et
    ``predim``), elle s'écrit donc une fois."""
    return dict(h=R.h_P, b=R.b_P, tw=R.tw_P, tf=R.tf_P, r=R.r_P)


def _first_ge(liste, v):
    c = sum(1 for x in liste if x < v)
    return min(c, len(liste) - 1)


def pb_bs5950(lam, py):
    """Contrainte de déversement pb (MPa) — BS 5950-1 Annexe B.2.1 : la
    brique partagée ``mecanique.pb_bs5950``, exposée ici parce que le
    benchmark et les suites la lisent sur le moteur du module."""
    return MEC.pb_bs5950(lam, py)


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
    d0 = R.d_0; d = R.d_b; V = u.V_Ed
    gM0 = u.g_M0; gM1 = u.g_M1; gM2 = u.g_M2; gM2n = u.g_M2n; gMu = u.g_Mu

    # ---- effort normal : ELU fondamental (combiné à VEd) ou TYING (seul)
    R.tying = bool(u.nat_N == NAT_TYING and u.N_Ed > 0)
    R.N_elu = 0.0 if R.tying else u.N_Ed      # traction (ou compression) ELU
    R.N_ty = u.N_Ed if R.tying else 0.0       # force de chaînage, seule
    NEd = R.N_elu
    NT = R.N_ty if R.tying else mx(0.0, R.N_elu)   # effort de traction vérifié
    gT = gMu if R.tying else gM2n                  # γ des résistances ultimes
    R.g_T = gT

    # ---- géométrie : la FACE D'APPUI (rotule) selon le cas
    # cas (a) : face extérieure de l'aile, X = hP/2 depuis l'axe du poteau ;
    # cas (b) : face de l'âme, X = twP/2 (MSB P5 §3.2 ; étude §6.1).
    R.xf_P = R.tw_P / 2 if R.ame else R.h_P / 2
    R.t_appui = R.tw_P if R.ame else R.tf_P      # épaisseur sollicitée
    R.g_B = u.g_h + R.e2b                        # face d'appui → 1re file
    R.z_p = R.g_B + (R.n_2 - 1) * u.p2_u / 2     # face d'appui → centre du groupe
    R.e2_p = R.b_p - R.g_B - (R.n_2 - 1) * u.p2_u    # pince du plat, bord libre
    R.e1bot = R.h_p - R.e_1 - (R.n_1 - 1) * R.p_1
    R.e1b_S = u.z_C + R.e_1 - u.d_nt
    R.h_e = R.h_S - u.d_nb - (u.z_C + R.e_1 + (R.n_1 - 1) * R.p_1)
    grp = MEC.groupe_excentre(R.n_1, R.n_2, R.p_1, u.p2_u, V, NEd,
                              V * (R.g_B + (R.n_2 - 1) * u.p2_u / 2) / 1000 + Mabs)
    R.n_b = grp["n"]; R.Ip = grp["Ip"]; R.xm = grp["xm"]; R.ym = grp["ym"]
    R.zeff = R.z_p
    R.M_S = V * R.zeff / 1000 + Mabs
    R.mod = ("Distribution uniforme" if R.M_S == 0 else
             "Rotule à la face d'appui — groupe de boulons avec moment "
             "VEd·z (répartition élastique, inertie polaire)")
    R.rec_hp = R.h_p >= 0.6 * R.h_S             # maintien en torsion (guides)
    R.rec_tp = R.t_p <= 0.5 * d + 1e-9          # ductilité (P358 / MSB P5)
    R.long_p = R.z_p > dv(R.t_p, 0.15)          # plat LONG (P358 / MSB P5)

    # ---- géométrie propre au POTEAU (étude §3.2.2, §4.3, §5.3)
    # brique partagée : les largeurs utiles du poteau sont les mêmes pour
    # les trois attaches de la famille (``mecanique.geometrie_poteau``) ; pour
    # un plat d'âme, l'about de la poutre est à ``gh`` de la face d'appui.
    geoP = MEC.geometrie_poteau(_profil_P(R), u.j_lat, u.g_h, R.b_S, R.d_w)
    R.libre_droit = geoP["libre_droit"]                   # largeur droite de l'âme
    R.demi_droit = geoP["demi_droit"]                     # demi-largeur droite
    R.libre_ailes = geoP["libre_ailes"]                   # passage de la semelle
    R.p3_congeP = geoP["p3_congeP"]                       # gauge mini d'une file sur l'aile
    R.gh_b1 = geoP["gh_b1"]                               # variante b1 : au-delà des bouts d'aile
    R.rec_min = geoP["rec_min"]                           # recoupe mini si la poutre ne passe pas
    R.passe_S = geoP["passe_S"]                           # la semelle passe entre les ailes
    R.b1 = bool(R.ame and u.g_h >= R.gh_b1 - 1e-9)        # variante b1
    R.rec_ok = bool(not R.ame or R.passe_S or R.b1 or u.rec_S >= R.rec_min - 1e-9)
    # excentricité apportée au POTEAU : information, pas une vérification de
    # l'attache — elle appartient au calcul d'élément (EC3-1-1 §6.2 et §6.3)
    R.e_col = (R.h_P / 2 + R.z_p) if R.aile else R.z_p
    R.M_col = V * R.e_col / 1000

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
    drow("e1b", "Âme de la poutre portée – e1,b (vers grugeage sup.)", R.e1b_S, 1.2, R.tw_S, "e",
         u.d_nt > 0, ["z_C", "e1_u", "d_nt"], ["e1b", "zp", "dcs"])
    drow("he", "Âme de la poutre portée – he (vers grugeage inf.)", R.h_e, 1.2, R.tw_S, "e",
         u.d_nb > 0, ["z_C", "e1_u", "n1_u", "p1_u", "d_nb"], ["he", "dci"])
    drow("e2b", "Âme de la poutre portée – e2,b (vers l'about)", R.e2b, 1.2, R.tw_S, "e", True,
         ["e2b_u"], ["e2b"])
    drow("p1", "Entraxe p1", R.p_1, 2.2, tmin, "p", R.n_1 > 1, ["p1_u"], ["p1"])
    drow("p2", "Entraxe p2", u.p2_u, 2.4, tmin, "p", R.n_2 > 1, ["p2_u"], ["p2"])
    R.dist_ok = all(x.ok for x in R.dist)

    # ---- alertes
    tfrS = R.tf_S + R.r_S
    m12 = 1.2 * d0
    # gorge de pleine résistance vis-à-vis du plat (dérivation §4.5.3.3 :
    # 2·a·fvw,d ≥ tp·fy/(√3·γM0)) — recommandation, contrôlée par alerte
    R.fuw, R.bww = RS.fu_bw_cordon(R.fu_L, R.bw_L, R.fu_P, R.bw_P)
    R.a_plein = MEC.a_pleine_resistance_plat(R.t_p, R.fy_L, R.bww, R.fuw, gM2, gM0)
    botLim = R.h_S - (u.d_nb if u.d_nb > 0 else tfrS); sumC = u.z_C + R.h_p
    prS = ["prof_S", "hS_u", "bS_u", "tfS_u", "rS_u"] if u.prof_S == PERSO else ["prof_S"]
    prP = ["prof_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"] if u.prof_P == PERSO else ["prof_P"]
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
             m="Configuration impossible : le dessus du plat est dans la zone grugée ou dans le congé supérieur de la poutre portée.",
             f=["z_C", "d_nt"] + prS, d=["zp", "dcs"], e=["plate", "notchT"],
             w=("zp = " + f0(u.z_C) + " mm est inférieur à la profondeur du grugeage dc,sup = " + f0(u.d_nt) + " mm : il faut zp ≥ dc,sup.") if u.d_nt > 0
             else ("zp = " + f0(u.z_C) + " mm est inférieur à tf + r = " + f0(tfrS) + " mm (semelle et congé de la poutre portée) : il faut zp ≥ " + f0(math.ceil(tfrS)) + " mm.")),
        dict(id="h_dispo", c=sumC > botLim, b=1,
             m="Configuration impossible : hauteur disponible insuffisante – le plat dépasse la partie droite de l'âme de la poutre portée.",
             f=["hp_u", "z_C"] + (["d_nb"] if u.d_nb > 0 else []) + prS, d=["hp", "zp"] + (["dci"] if u.d_nb > 0 else []), e=["plate"],
             w="zp + hp = " + f0(u.z_C) + " + " + f0(R.h_p) + " = " + f0(sumC) + " mm dépasse la limite basse de la partie droite de l'âme : "
             + (("h − dc,inf = " + f0(R.h_S) + " − " + f0(u.d_nb)) if u.d_nb > 0 else ("h − (tf + r) = " + f0(R.h_S) + " − " + f0(tfrS)))
             + " = " + f0(botLim) + " mm. Avec zp = " + f0(u.z_C) + " mm, hp maximal = " + f0(math.floor(botLim - u.z_C)) + " mm."),
        dict(id="aile_poutre", c=R.ame and not R.rec_ok, b=1,
             m=TXT.M_AILE_POUTRE,
             f=["rec_S", "j_lat", "g_h"] + prS + prP, d=["rec", "jlat", "gh"], e=["beamS", "recut", "col"],
             w=TXT.w_aile_poutre(f0(R.b_S), f0(R.libre_ailes), "gh", F(R.rec_min, 1), f0(u.rec_S))
             + ", ou éloigner la poutre (gh ≥ " + F(R.gh_b1, 1) + " mm)."),
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
        dict(id="b1_long", c=R.b1, b=0,
             m="Cas (b), variante b1 : la poutre reste au-delà des bouts d'aile du poteau — le bras z est multiplié par 3 à 4, le plat devient LONG et la flexion hors plan de l'âme du poteau sous qx n'est pas vérifiée. Préférer la recoupe des semelles (variante b2).",
             f=["g_h", "rec_S"], d=["gh", "z"], e=["plate", "col"],
             w="gh = " + f0(u.g_h) + " mm ≥ (bP − twP)/2 + j = " + F(R.gh_b1, 1) + " mm : z = "
             + F(R.z_p, 1) + " mm au lieu de " + F(R.e2b + mn(u.g_h, 10), 1)
             + " mm avec un about au contact de l'âme."),
        dict(id="ame_hp", c=R.ame, b=0,
             m=TXT.m_ame_hp("qx = 3·MS/hp² et sous la force de chaînage",
                            "plat traversant, raidisseurs entre ailes"),
             f=["appui"] + prP, d=["libreP"], e=["col", "colWeb"], w=""),
        dict(id="a_rec", c=R.a_w < R.a_plein - 1e-9, b=0,
             m="Gorge inférieure à la PLEINE RÉSISTANCE recommandée vis-à-vis du plat — admissible si la vérification en contrainte du cordon passe, mais le détail courant soude le plat à pleine résistance.",
             f=["a_w", "tp_u"], d=["aw"], e=["weld"],
             w="Recommandé : a ≥ tp·fy·βw·γM2/(2·fu·γM0) = " + F(R.a_plein, 1) + " mm (dérivation EN 1993-1-8 §4.5.3.3) ; actuel : a = " + f0(R.a_w) + " mm."),
        dict(id="MEd", c=Mabs != 0, b=0,
             m="MEd ≠ 0 : l'assemblage reste modélisé comme articulé ; MEd est ajouté au moment d'excentricité VEd·z. Un moment significatif sort du domaine de l'outil.",
             f=["M_Ed"], d=["z"], e=[], w="MS = VEd·z + |MEd| = " + F(R.M_S, 2) + " kNm."),
        dict(id="NEd", c=u.N_Ed > 0 and not R.tying, b=0,
             m="NEd de traction à l'ELU fondamental : repris en cisaillement horizontal des boulons et pressions diamétrales. La flexion hors plan de l'aile du poteau sous NEd n'est pas vérifiée pour une attache soudée (le plat est dans le plan de l'âme) ; en cas (b), celle de l'âme ne l'est pas non plus.",
             f=["N_Ed", "nat_N"], d=[], e=[], w=""),
        dict(id="N_tying", c=R.tying, b=0,
             m="NEd en TYING (situation accidentelle) : la force de chaînage est vérifiée SEULE, sans VEd, avec les résistances ultimes et γMu (MSB Part 5 §2.3). Le côté poteau n'apporte aucune vérification de plus : le plat est soudé, la force est dans son plan.",
             f=["N_Ed", "nat_N", "g_Mu"], d=[], e=[],
             w="NEd = " + F(u.N_Ed, 1) + " kN ; γMu = " + F(gMu, 2) + "."),
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
    R.Fv_Rd_u = RS.Fv_Rd(R.a_v, R.f_ub, R.A_cis, gMu)      # tying (γMu)
    R.Fp_C = RS.Fp_C(R.f_ub, R.As_b)
    R.Lj = (R.n_1 - 1) * R.p_1
    R.bLf = RS.beta_Lf(R.Lj, d, u.opt_blf == "Oui")
    R.Fz_b = grp["Fz"]; R.Fx_b = grp["Fx"]; R.F_b = grp["F"]
    R.tab = MEC.bolt_table(R.n_1, R.n_2, R.p_1, u.p2_u, V, NEd, R.M_S, R.Ip)
    R.N_par_boulon = dv(NT, R.n_b)

    def bearing(key, t, fu, e1, p1, n1, e2, p2, n2):
        b = MEC.bearing(t, fu, R.f_ub, d, d0, e1, p1, n1, e2, p2, n2, gM2,
                        R.k_trou, Fz=R.Fz_b, Fx=R.Fx_b)
        for nom in ("abv", "k1v", "abh", "k1h", "Fbv", "Fbh", "ipd"):
            R[nom + "_" + key] = b[nom]

    bearing("L", R.t_p, R.fu_L, mn(R.e_1, R.e1bot), R.p_1, R.n_1, R.e2_p, u.p2_u, R.n_2)
    bearing("S", R.tw_S, R.fu_S, mn(R.e1b_S, R.h_e) if u.d_nb > 0 else R.e1b_S,
            R.p_1, R.n_1, R.e2b, u.p2_u, R.n_2)

    def bvals(key):
        return ("αb,ver = " + F(R["abv_" + key], 3) + " ; k1,ver = " + F(R["k1v_" + key], 2)
                + " → Fb,ver,Rd = " + F(R["Fbv_" + key], 1) + " kN ; αb,hor = " + F(R["abh_" + key], 3)
                + " ; k1,hor = " + F(R["k1h_" + key], 2) + " → Fb,hor,Rd = " + F(R["Fbh_" + key], 1) + " kN")

    # ---- plat : aires, flexion, déversement, traction
    R.Av_p = R.h_p * R.t_p
    R.Avn_p = (R.h_p - R.n_1 * d0) * R.t_p
    R.Ah_p = R.e2_p + (R.n_2 - 1) * u.p2_u - (R.n_2 - 0.5) * d0
    R.Ant_p = R.t_p * R.Ah_p
    R.Anv_p = R.t_p * (R.h_p - R.e_1 - (R.n_1 - 0.5) * d0)
    R.Wel_p = R.t_p * R.h_p * R.h_p / 6
    R.court = R.h_p >= 2.73 * R.z_p             # flexion non déterminante
    R.lam_LT = MEC.lam_LT(R.z_p, R.h_p, R.t_p)
    R.pb_LT = MEC.pb_bs5950(R.lam_LT, R.fy_L)
    R.Mb_p = dv(R.Wel_p * R.pb_LT, gM1) / 1e6
    # rupture de bloc EN TRACTION du plat (§3.10.2(2)) : deux chemins
    # d'arrachement — entre rangées (cas 1) ou vers le bord libre le plus
    # proche (cas 2) ; aires du guide (MSB P5 §4.3.1.4)
    e1t = mn(R.e_1, R.e1bot)
    R.Nt1_p = (dv(R.fu_L * R.t_p * (R.n_1 - 1) * (R.p_1 - d0), gT)
               + dv(R.fy_L * 2 * R.t_p * R.Ah_p / S3, gM0)) / 1000
    R.Nt2_p = (dv(R.fu_L * R.t_p * (e1t + (R.n_1 - 1) * R.p_1 - (R.n_1 - 0.5) * d0), gT)
               + dv(R.fy_L * R.t_p * R.Ah_p / S3, gM0)) / 1000

    # ---- poutre portée (section grugée : mêmes règles que le module
    # ---- doubles cornières, MSB Part 5 §4.2.4 / §4.2.5)
    sg = MEC.section_grugee(dict(h=R.h_S, b=R.b_S, tw=R.tw_S, tf=R.tf_S, r=R.r_S),
                            R.fy_S, u.d_nt, u.d_nb, gM0, V, u.lt_ok == "Oui")
    R.cas_g = sg["cas_g"]; R.Asec_S = sg["Asec"]; R.h_T = sg["h_T"]; R.A_Tee = sg["A_Tee"]
    R.Av_S = sg["Av"]; R.VRd_gS = sg["VRd_g"]; R.hw_T = sg["hw_T"]; R.zb_T = sg["zb_T"]
    R.I_N = sg["I_N"]; R.W_N = sg["W_N"]; R.rho_N = sg["rho_N"]; R.Mv_N = sg["Mv_N"]
    R.el_N = sg["el_N"]; R.lim_N = sg["lim_N"]; R.ln_max = sg["ln_max"]; R.val_N = sg["val_N"]
    R.Ah_S = R.e2b + (R.n_2 - 1) * u.p2_u - (R.n_2 - 0.5) * d0
    R.Ant_S = R.tw_S * R.Ah_S
    # rupture de bloc de l'âme : (n1 − 0,5)·d0 si l'arrachement part d'un bord
    # libre (poutre grugée), (n1 − 1)·d0 sinon — MSB Part 5 §3.2.5.1
    kv = (R.n_1 - 0.5) if u.d_nt > 0 else (R.n_1 - 1)
    R.Anv_S = R.tw_S * (R.e1b_S + (R.n_1 - 1) * R.p_1 - kv * d0)
    R.VRd_bS = (dv(0.5 * R.fu_S * R.Ant_S, gM2n) + dv(R.fy_S * R.Anv_S / S3, gM0)) / 1000
    Vm = mn(R.VRd_gS, R.VRd_bS)
    R.Mc_2 = dv(R.W_N * R.fy_S, gM0) / 1e6 * (1 if V <= 0.5 * Vm else mx(0, 1 - (2 * dv(V, Vm) - 1) ** 2))
    R.Nt1_S = (dv(R.fu_S * R.tw_S * (R.n_1 - 1) * (R.p_1 - d0), gT)
               + dv(R.fy_S * 2 * R.tw_S * R.Ah_S / S3, gM0)) / 1000
    R.Nt2_S = (dv(R.fu_S * R.tw_S * (R.e1b_S + (R.n_1 - 1) * R.p_1 - (R.n_1 - 0.5) * d0), gT)
               + dv(R.fy_S * R.tw_S * R.Ah_S / S3, gM0)) / 1000

    # ---- POTEAU : cas (b) seulement — cisaillement local de l'âme, un seul
    # ---- côté chargé (SCI P358, même composant que l'âme porteuse des
    # ---- modules poutre–poutre) ; cas (a) : groupe VIDE, voir §4.4
    R.Av_P = R.tw_P * R.h_p
    R.VRd_lP = dv(R.Av_P * R.fy_P / S3, gM0) / 1000

    # ---- soudure (deux cordons verticaux pleine hauteur, gorge a chacun)
    R.qz_w = dv(V * 1000, 2 * R.h_p)
    R.qx_w = dv(3 * R.M_S * 1e6, R.h_p * R.h_p) + dv(abs(NEd) * 1000, 2 * R.h_p)
    R.qw = hyp(R.qz_w, R.qx_w)
    R.fvw = RS.fvw_d(R.fuw, R.bww, gM2)
    R.Fw = R.fvw * R.a_w
    R.qx_ty = dv(NT * 1000, 2 * R.h_p)
    R.fvw_u = RS.fvw_d(R.fuw, R.bww, gMu)
    R.Fw_ty = R.fvw_u * R.a_w

    # ---- vérifications
    def ck(key, grp_, label, Ed, Rd, unit, ref, nat, act, formula, vals="", ess=False):
        c = Verification(key=key, grp=grp_, label=label, Ed=Ed, Rd=Rd, unit=unit, ref=ref,
                         nat=nat, active=bool(act), formula=formula, vals=vals or "", ess=bool(ess))
        if c.active:
            c.eta = (Ed / Rd) if Rd > 0 else (INF if Ed > 0 else 0)
            c.ok = bool(c.eta <= 1 + 1e-9)
        R.checks.append(c)
        R.ck[key] = c

    catB = u.cat == "B"; ks = u.k_ser if catB else 1; g3 = u.g_M3s if catB else u.g_M3
    W445 = "EN 1993-1-8 §4.5.3.3 (méthode simplifiée)"
    ty = " (tying : fu/γMu)" if R.tying else ""

    # --- Boulons
    ck("bv", "Boulons", "Cisaillement des boulons (1 plan), boulon le plus sollicité", R.F_b, R.Fv_Rd * R.bLf, "kN",
       T34 + " ; groupe excentré : " + MSB3, "EC + COMP", True,
       "Fv,Rd = αv·fub·A/γM2 ; F,Ed = √(Fz² + Fx²) avec Fz = VEd/n + MS·xmax/Ip et Fx = |NEd|/n + MS·ymax/Ip",
       "z = " + F(R.z_p, 1) + " mm ; MS = " + F(R.M_S, 2) + " kNm ; Ip = " + F(R.Ip, 0) + " mm² ; Fz = " + F(R.Fz_b, 2) + " kN ; Fx = " + F(R.Fx_b, 2) + " kN ; Fv,Rd = " + F(R.Fv_Rd, 2) + " kN ; βLf = " + F(R.bLf, 3), 1)
    ck("gl", "Boulons", "Glissement (1 surface de frottement)", R.F_b * ks, dv(u.k_s * u.mu_s * R.Fp_C, g3), "kN",
       "EN 1993-1-8 §3.9.1", "EC", u.cat != "A",
       "Fs,Rd = ks·n·μ·Fp,C/γM3 ; Fp,C = 0,7·fub·As", "Fp,C = " + F(R.Fp_C, 1) + " kN ; catégorie " + u.cat)
    ck("bvT", "Boulons", "Tying – boulons en cisaillement (force de chaînage seule)", R.N_par_boulon,
       R.Fv_Rd_u, "kN", T34 + " avec γMu ; " + MSB_TY, "EC + COMP", R.tying,
       "NEd/n ≤ αv·fub·A/γMu — la force de chaînage est dans le PLAN du plat : elle cisaille les boulons",
       "NEd = " + F(R.N_ty, 1) + " kN ; n = " + js_str(R.n_b) + " ; γMu = " + F(gMu, 2)
       + " ; n·Fv,Rd,u = " + F(R.n_b * R.Fv_Rd_u, 1) + " kN", 1)
    # --- Plat d'âme
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
    ck("ptn", "Plat d'âme", "Traction section nette" + ty, NT, dv(0.9 * R.Avn_p * R.fu_L, gT) / 1000, "kN",
       "EN 1993-1-1 §6.2.3(2)" + (" ; " + MSB_TY if R.tying else ""), "EC", NT > 0,
       "Nu,Rd = 0,9·Anet·fu/γ ; Anet = tp·(hp − n1·d0)",
       "Anet = " + F(R.Avn_p, 0) + " mm² ; γ = " + F(gT, 2))
    ck("pbt", "Plat d'âme", "Rupture de bloc en traction" + ty, NT, mn(R.Nt1_p, R.Nt2_p), "kN",
       "EN 1993-1-8 §3.10.2(2) ; aires : MSB Part 5 §4.3.1.4", "EC", NT > 0,
       "Neff,1,Rd = fu·Ant/γ + fy·Anv/(√3·γM0)",
       "cas 1 = " + F(R.Nt1_p, 0) + " kN ; cas 2 = " + F(R.Nt2_p, 0) + " kN")
    # --- Poutre portée
    ck("pdS", "Poutre portée", "Pression diamétrale – âme de la poutre portée", R.F_b,
       dv(R.F_b, R.ipd_S) if R.ipd_S > 0 else R.Fbv_S, "kN",
       T34 + " ; interaction : " + MSB3, "EC + COMP", True,
       "Fb,Rd = k1·αb·fu·d·tw/γM2 ; η = √[(Fz/Fb,ver)² + (Fx/Fb,hor)²]", bvals("S"), 1)
    ck("vgS", "Poutre portée", "Âme – cisaillement section brute (section réduite si grugée)", V, R.VRd_gS, "kN",
       "EN 1993-1-1 §6.2.6 ; MSB Part 5 §3.2.5.1", "EC", True,
       "VRd,g = Av·fy/(√3·γM0)", "Av = " + F(R.Av_S, 0) + " mm²", 1)
    ck("vnS", "Poutre portée", "Âme – cisaillement section nette au droit des boulons", V,
       dv((R.Av_S - R.n_1 * d0 * R.tw_S) * R.fu_S / S3, gM2n) / 1000, "kN",
       "MSB Part 5 §3.2.5.1 (ECCS n°126)", "COMP", True,
       "VRd,n = (Av − n1·d0·tw)·fu/(√3·γM2)", "Av,net = " + F(R.Av_S - R.n_1 * d0 * R.tw_S, 0) + " mm²", 1)
    ck("vbS", "Poutre portée", "Âme – rupture de bloc", V, R.VRd_bS, "kN",
       "EN 1993-1-8 §3.10.2(3) ; aires : MSB Part 5 §3.2.5.1", "EC", True,
       "Veff,2,Rd = 0,5·fu·Ant/γM2 + fy·Anv/(√3·γM0) ; Anv avec (n1 − 1)·d0 si non grugée, (n1 − 0,5)·d0 si grugée",
       "Ant = " + F(R.Ant_S, 0) + " mm² ; Anv = " + F(R.Anv_S, 0) + " mm²", 1)
    ck("mN", "Poutre portée", "Poutre grugée – flexion + cisaillement à l'extrémité du grugeage",
       V * (u.g_h + u.l_n) / 1000 + Mabs, R.Mv_N, "kNm",
       "MSB Part 5 §4.2.4 (mêmes règles pour le grugeage)", "COMP", R.cas_g > 0,
       "MEd = VEd·(gh + c) ; Mv,N,Rd = fy·Wel,N/γM0·[1 − (2VEd/Vpl,N,Rd − 1)²] si VEd > 0,5·Vpl,N,Rd",
       "Wel,N = " + F(R.W_N, 0) + " mm³ ; Vpl,N,Rd = " + F(R.VRd_gS, 1) + " kN ; réduction = " + F(R.rho_N, 3), 1)
    ck("m2", "Poutre portée", "Poutre grugée – interaction à la 2e file de boulons (n2 = 2)",
       V * (u.g_h + R.e2b + u.p2_u) / 1000, R.Mc_2, "kNm", "MSB Part 5 §3.2.5", "COMP",
       R.cas_g > 0 and R.n_2 == 2 and u.l_n > R.e2b + u.p2_u, "MEd = VEd·(gh + e2,b + p2)", "")
    ck("stN", "Poutre portée", "Stabilité locale du grugeage – longueur c", u.l_n, R.ln_max, "mm",
       "MSB Part 5 §4.2.5", "COMP", R.cas_g > 0,
       "c ≤ h si h/tw ≤ limite ; sinon c ≤ K·h/(h/tw)³",
       "h/tw = " + F(R.el_N, 1) + " ; limite = " + F(R.lim_N, 1) + " ; domaine : " + R.val_N, 1)
    ck("stD", "Poutre portée", "Stabilité locale du grugeage – profondeur", mx(u.d_nt, u.d_nb),
       R.h_S / 2 if R.cas_g == 1 else R.h_S / 5, "mm", "MSB Part 5 §4.2.5", "COMP", R.cas_g > 0,
       "dc ≤ h/2 (1 semelle grugée) ; dc ≤ h/5 (2 semelles)", "")
    ck("tnS", "Poutre portée", "Âme – traction section nette" + ty, NT,
       dv(0.9 * R.tw_S * (R.h_p - R.n_1 * d0) * R.fu_S, gT) / 1000, "kN",
       "EN 1993-1-1 §6.2.3(2) ; hauteur participante = hp : hypothèse de l'outil"
       + (" ; " + MSB_TY if R.tying else ""), "INT", NT > 0,
       "Nu,Rd = 0,9·Anet·fu/γ ; Anet = tw·(hp − n1·d0)",
       "Anet = " + F(R.tw_S * (R.h_p - R.n_1 * d0), 0) + " mm² ; γ = " + F(gT, 2))
    ck("tbS", "Poutre portée", "Âme – rupture de bloc en traction" + ty, NT,
       mn(R.Nt1_S, R.Nt2_S if u.d_nt > 0 else INF), "kN",
       "EN 1993-1-8 §3.10.2(2) ; aires : MSB Part 5 §4.3.2.3", "EC", NT > 0,
       "Neff,1,Rd = fu·Ant/γ + fy·Anv/(√3·γM0) ; cas 2 (vers le bord libre) seulement si la poutre est grugée",
       "cas 1 = " + F(R.Nt1_S, 0) + " kN"
       + (" ; cas 2 = " + F(R.Nt2_S, 0) + " kN" if u.d_nt > 0 else " ; cas 2 sans objet (poutre non grugée)"))
    # --- Poteau porteur : cas (b) seulement (§4.4 : groupe VIDE en cas (a))
    ck("vlP", "Poteau porteur",
       "Âme du poteau – cisaillement local sur la hauteur du plat (un seul côté chargé)", V / 2,
       R.VRd_lP, "kN", P358_SUP, "COMP", R.ame,
       "VEd/2 ≤ Av·fy,P/(√3·γM0) ; Av = twP·hp", "Av = " + F(R.Av_P, 0) + " mm² ; twP = "
       + F(R.tw_P, 1) + " mm ; fy,P = " + F(R.fy_P, 0) + " MPa", 1)
    # --- Soudure
    ck("w", "Soudure", "Cordons plat / poteau – effort par unité de longueur au point critique", R.qw, R.Fw, "N/mm",
       W445, "EC", True,
       "Fw,Ed = √(qz² + qx²) ≤ Fw,Rd = fvw,d·a ; qz = VEd/(2hp) ; qx = 3·MS/hp² + |NEd|/(2hp) ; fvw,d = fu/(√3·βw·γM2)",
       "a = " + F(R.a_w, 0) + " mm ; qz = " + F(R.qz_w, 0) + " ; qx = " + F(R.qx_w, 0) + " N/mm ; fvw,d = " + F(R.fvw, 1) + " MPa", 1)
    ck("wT", "Soudure", "Cordons – tying (force de chaînage seule)", R.qx_ty, R.Fw_ty, "N/mm",
       W445 + " avec γMu ; " + MSB_TY, "EC + INT", R.tying,
       "qx = NEd/(2hp) ≤ fu/(√3·βw·γMu)·a",
       "qx = " + F(R.qx_ty, 0) + " N/mm ; a = " + F(R.a_w, 1) + " mm ; γMu = " + F(gMu, 2))
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
    """Efforts par boulon (répartition élastique, inertie polaire) — la
    brique partagée ``mecanique.bolt_table``."""
    return MEC.bolt_table(n1, n2, p1, p2, V, Hh, M, Ip)


# =========================================================================
#  predim
# =========================================================================
def predim(u, R, Mabs):
    """Prédimensionnement : pour chaque boulon (plage choisie) et chaque
    nombre de rangées (2 → 8), pinces et entraxes de détail (multiples de 5),
    taux estimés (boulons, pressions diamétrales plat et âme), puis plat
    nécessaire (tp par les résistances, plafonné à 0,5·d — ductilité) et
    gorge de pleine résistance.

    Le POTEAU n'y intervient que par le ``fu`` du cordon (pleine résistance,
    ``min(fu plat ; fu poteau)``) : la hauteur disponible et les taux sont
    commandés par la poutre portée et le plat, dans les deux cas d'appui —
    un poteau n'a ni semelle au-dessus ni semelle en dessous de l'attache."""
    rows = []
    hav = R.h_S - (u.d_nb if u.d_nb > 0 else R.tf_S + R.r_S) - u.z_C
    V = u.V_Ed
    i1 = idx(DB["boulons"], u.pd_dmin); i2 = idx(DB["boulons"], u.pd_dmax)
    imin = min(i1, i2); imax = max(i1, i2)
    Nx = 0.0 if u.nat_N == NAT_TYING else abs(u.N_Ed)
    for i, b in enumerate(DB["boulons"]):
        for n in range(2, 9):
          d = b["d"]; d0 = b["d0"]; A = b["As"] if u.filet == "Oui" else b["A"]
          # deux jeux de pinces : « détail » (k·d0 arrondi au 5 supérieur) et
          # « compacte » (au 5 inférieur, plancher normatif du Tableau 3.3)
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
            Fx = Nx / n + dv(Mz * 1000 * 6, p1 * n * (n + 1)); Fs = hyp(Fz, Fx)
            e1b = u.z_C + e1 - u.d_nt
            Fbv = (mn(2.8 * e2 / d0 - 1.7, 2.5) * mn(e1b / (3 * d0), p1 / (3 * d0) - 0.25, dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            Fbh = (mn(2.8 * e1b / d0 - 1.7, 1.4 * p1 / d0 - 1.7, 2.5) * mn(e2 / (3 * d0), dv(R.f_ub, R.fu_S), 1)
                   * R.fu_S * d * R.tw_S / u.g_M2 / 1000 * R.k_trou) if u.g_M2 != 0 else math.nan
            e = AttrDict(bS=dv(Fs, Fv * bLf), pdS=MEC.interaction_pd(Fz, Fx, Fbv, Fbh))
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
        reqs = [["Pression diamétrale du plat", MEC.interaction_pd(pk.Fz, pk.Fx, fbv, fbh)],
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
    """« M20, 4 rangées, plat 290 × 10 »."""
    return (r.b["n"] + ", " + js_str(r.n) + " rangées, plat " + js_str(r.hp) + " × "
            + js_str(r.prop.tp))

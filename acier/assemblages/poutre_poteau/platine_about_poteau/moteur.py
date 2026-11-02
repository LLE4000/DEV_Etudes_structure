# -*- coding: utf-8 -*-
"""Moteur de calcul — assemblage POUTRE–POTEAU par platine d'about de hauteur
partielle. Écrit pour CET assemblage : rotule à la face d'appui du poteau,
groupe de boulons CENTRÉ (deux files symétriques, aucun moment
d'excentricité), un plan de cisaillement entre la platine et le poteau ;
l'effort tranchant passe de l'âme portée à la platine par deux cordons
d'angle.

DEUX cas d'appui, et c'est la clé ``appui`` qui commande toute la mécanique
(`docs/assemblages/POUTRE_POTEAU.md` §1.2) :

- **cas (a), sur l'AILE** : la platine porte sur la face extérieure d'une
  aile, les deux files symétriques par rapport à l'ÂME du poteau. L'attache
  vit DANS LE PLAN de cette âme : VEd y chemine en plan, et le côté poteau
  n'apporte à l'ELU que la PRESSION DIAMÉTRALE de l'aile (§4.2, sans pince
  d'extrémité verticale : le poteau continue au-dessus et en dessous). Dès
  que les boulons sont tendus s'ajoutent deux composants NEUFS : le TRONÇON
  EN T de l'aile non raidie (Tableau 6.4) et l'ÂME DU POTEAU EN TRACTION
  TRANSVERSALE (§6.2.6.3, ω = 1) ;
- **cas (b), dans l'ÂME** : la platine porte sur l'âme, entre les ailes.
  L'âme du poteau joue le rôle de l'âme de la poutre porteuse des modules
  poutre–poutre — cisaillement local d'un élément support chargé d'un seul
  côté (SCI P358), brut et net. La flexion HORS PLAN de l'âme sous la
  traction des boulons n'est PAS couverte par l'EN 1993-1-8 pour une âme non
  raidie : elle est déclarée en limite (§5.4), elle n'est pas transposée.

Modèles et sources, distingués vérification par vérification (champ ``nat``) :
- ``EC``   : formule directe de l'EN 1993-1-8 / EN 1993-1-1 ;
- ``COMP`` : modèle complémentaire d'un guide reconnu — « Steel Buildings in
             Europe, Part 5: Joint Design » §2 (platine d'about de hauteur
             partielle : §2.1 détails, §2.2 cisaillement, §2.3 tying),
             SCI P358, ECCS n°126 ;
- ``INT``  : interprétation de l'outil, nommée comme telle.

Traction NEd : deux natures. ELU fondamental — fy/γM0 et γM2, combinée avec
VEd. Tying (force de chaînage, situation accidentelle) — vérifiée SEULE avec
fu/γMu, comme le guide (MSB P5 §2.3).

Les briques de calcul communes à la famille ne sont PAS recopiées : elles
viennent de ``acier/assemblages/mecanique.py`` (pression diamétrale, groupe
de fixations, poutre grugée, tronçon en T, longueurs efficaces, ductilité,
gorge de pleine résistance), que ``tests/test_assemblages_mecanique.py`` lie
aux moteurs poutre–poutre.

Coefficients partiels par défaut : ANB belge pour l'EN 1993-1-1 (γM0 = 1,00),
valeurs recommandées pour l'EN 1993-1-8 (γM2 = 1,25) ; γMu = 1,1 du guide.
Tous modifiables.

Unités : mm, mm², mm³, MPa, kN, kNm, N/mm (cordons).
"""
import math

from acier.bibliotheques import DB, PERSO, find, idx
from acier.js import AttrDict, INF, N, dv, mn, mx, js_round, ceil5, js_str
from acier.formats import F, f0
from acier.resultats import Verification, Alerte, Pince
from acier import resistances as RS
from acier.assemblages import mecanique as MEC
from .. import textes as TXT
from .entrees import defaults, NUM, MODE_PREDIM, APPUI_AME, NAT_TYING

S3 = math.sqrt(3)
T34 = "EN 1993-1-8 Tableau 3.4"
MSB2 = "MSB Part 5 §2.2 (platine d'about partielle)"
MSB_GRP = "MSB Part 5 §2.2.4 (groupe de boulons) sur EN 1993-1-8 §3.7"
MSB_TY = "MSB Part 5 §2.3 (tying)"
P358_SUP = "SCI P358 (élément support chargé d'un seul côté)"
TAB64 = "EN 1993-1-8 Tableau 6.4 (aile de poteau non raidie)"
K_CIS = 1.27             # cisaillement brut de la platine : flexion nominale dans son plan (MSB P5 §2.2)

GROUPES_VERIF = ("Boulons", "Platine d'about", "Poutre portée", "Poteau porteur",
                 "Soudure")
DETAIL_SEUL = frozenset({"wa"})          # minimum de détail : jamais dimensionnant s'il passe


def _profil_P(R):
    """Les cinq dimensions du POTEAU en dictionnaire, comme l'appel de
    ``mecanique.section_grugee`` le fait pour la poutre portée — la
    géométrie du poteau se lit à plusieurs endroits (``compute`` et
    ``predim``), elle s'écrit donc une fois."""
    return dict(h=R.h_P, b=R.b_P, tw=R.tw_P, tf=R.tf_P, r=R.r_P)


def _first_ge(liste, v):
    c = sum(1 for x in liste if x < v)
    return min(c, len(liste) - 1)



# =========================================================================
#  compute
# =========================================================================
def compute(u0=None):
    """Calcul complet pour les saisies ``u0`` (clés absentes → défauts).
    Retourne ``R`` : scalaires du modèle, ``u``, ``checks`` / ``ck``,
    ``alerts``, ``dist`` (Tableau 3.3), ``pd`` (prédimensionnement),
    ``gov``, ``eta_max``, ``verified``, ``statut``, ``reserve``."""
    u = AttrDict(defaults())
    for k, v in (u0 or {}).items():
        u[k] = v
    for n in NUM:
        u[n] = N(u[n])
    u.V_Ed = abs(u.V_Ed)
    R = AttrDict()
    R.u = u
    R.checks = []
    R.ck = {}
    R.pred = u.mode_calc == MODE_PREDIM
    R.ame = u.appui == APPUI_AME            # cas (b) : platine dans l'âme
    R.aile = not R.ame                      # cas (a) : platine sur l'aile
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
    pd = R.pd = predim(u, R)

    # ---- paramètres retenus (◆ en prédimensionnement)
    R.boulon = pd.boulon if R.pred else u.boulon_u

    def ret(a, b):
        return b if R.pred else a

    R.n_1 = ret(u.n1_u, pd.n); R.p_1 = ret(u.p1_u, pd.p1); R.e_1 = ret(u.e1_u, pd.e1)
    R.p_3 = ret(u.p3_u, pd.p3)
    R.h_p = ret(u.hp_u, pd.hp); R.t_p = ret(u.tp_u, pd.tp)
    R.b_p = ret(u.bp_u, pd.bp); R.a_w = ret(u.a_w, pd.a)
    R.n_1 = max(1, js_round(R.n_1)); R.n_2 = 2
    bo = find(DB["boulons"], R.boulon)
    R.d_b = bo["d"]; R.d_0 = N(u.d0_u) if N(u.d0_u) > 0 else bo["d0"]
    R.As_b = bo["As"]; R.A_cis = bo["As"] if u.filet == "Oui" else bo["A"]
    R.d_m = bo["dm"]; R.d_w = bo["dw"]
    d0 = R.d_0; d = R.d_b; V = u.V_Ed
    NEd = mx(0, u.N_Ed)                         # compression : par contact
    R.tying = bool(u.nat_N == NAT_TYING and NEd > 0)
    R.N_elu = 0.0 if R.tying else NEd           # traction combinée avec VEd
    R.N_ty = NEd if R.tying else 0.0            # force de chaînage, vérifiée seule
    gM0 = u.g_M0; gM2 = u.g_M2; gM2n = u.g_M2n; gMu = u.g_Mu

    # ---- géométrie : la FACE D'APPUI (rotule) selon le cas
    # cas (a) : face extérieure de l'aile, X = hP/2 depuis l'axe du poteau ;
    # cas (b) : face de l'âme, X = twP/2 (MSB P5 §2.2 ; étude §6.1).
    R.xf_P = R.tw_P / 2 if R.ame else R.h_P / 2
    R.t_appui = R.tw_P if R.ame else R.tf_P      # épaisseur du poteau sollicitée
    R.fu_appui = R.fu_P
    R.n_b = 2 * R.n_1
    R.e1bot = R.h_p - R.e_1 - (R.n_1 - 1) * R.p_1
    R.e_2 = (R.b_p - R.p_3) / 2                 # pince latérale de la platine
    R.e_P = (R.b_P - R.p_3) / 2                 # pince au bout d'aile du poteau (cas a)
    R.e1_min = mn(R.e_1, R.e1bot)               # pince verticale retenue (haut/bas)
    # les largeurs utiles du POTEAU : la brique partagée de la famille
    # (``mecanique.geometrie_poteau``, étude §3.2.2, §4.3 et §5.3). Pour une
    # PLATINE, l'about de la poutre est à ``tp`` de la face d'appui (il n'y a
    # pas de clé ``gh``) : c'est ``tp`` qui se passe en troisième argument, et
    # ``rec_min`` vaut donc (bP − twP)/2 + j − tp (écart assumé, voir la fiche).
    geoP = MEC.geometrie_poteau(_profil_P(R), u.j_lat, R.t_p, R.b_S, R.d_w)
    # la rondelle doit dégager le cordon de l'âme portée : p3/2 − dw/2 ≥ tw/2 + a·√2
    R.p3_min = R.tw_S + 2 * R.a_w * math.sqrt(2) + R.d_w
    # l'écrou doit dégager le CONGÉ DU POTEAU (étude §4.3, point 1) : il est du
    # côté intérieur de l'aile, au droit du congé — c'est la règle qui refuse le
    # p3 = 90 usuel des âmes de poutre sur l'aile d'un HEB 300 (102 mm requis)
    R.p3_congeP = geoP["p3_congeP"]
    R.rec_hp = R.h_p >= 0.6 * R.h_S             # maintien en torsion (MSB P5 §2.1)
    R.tp_max = MEC.tp_ductile(d, R.f_ub, R.fy_L)
    R.rec_tp = R.t_p <= R.tp_max + 1e-9
    R.hp_136 = R.h_p >= 1.36 * R.p_3            # flexion de la platine négligeable
    R.mod = ("Rotule à la face d'appui du poteau — groupe de boulons centré "
             "(deux files symétriques), sans moment d'excentricité")

    # ---- géométrie propre au POTEAU (``geoP``, posé plus haut)
    R.libre_droit = geoP["libre_droit"]                   # largeur droite de l'âme
    R.demi_droit = geoP["demi_droit"]                     # demi-largeur droite
    R.libre_ailes = geoP["libre_ailes"]                   # passage de la semelle
    R.bp_maxP = R.libre_droit - 2 * u.j_lat               # cas (b) : la platine tient
    R.p3_maxP = geoP["p3_maxP"]                           # cas (b) : l'écrou dégage le congé
    # la poutre entre entre les ailes, ou ses semelles sont RECOUPÉES depuis
    # l'about (la recoupe en LARGEUR n'est pas un grugeage — point P-3)
    R.rec_min = geoP["rec_min"]
    R.passe_S = geoP["passe_S"]
    R.rec_ok = bool(not R.ame or R.passe_S or u.rec_S >= R.rec_min - 1e-9)
    # excentricité apportée au POTEAU : information, pas une vérification de
    # l'attache — elle appartient au calcul d'élément (EC3-1-1 §6.2 et §6.3)
    R.e_col = (R.h_P / 2 + R.t_p) if R.aile else R.t_p
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

    tmin = mn(R.t_p, R.t_appui)
    prP = ["prof_P", "hP_u", "bP_u", "twP_u", "tfP_u", "rP_u"] if u.prof_P == PERSO else ["prof_P"]
    prS = ["prof_S", "hS_u", "bS_u", "tfS_u", "rS_u", "twS_u"] if u.prof_S == PERSO else ["prof_S"]
    drow("e1p", "Platine – e1 (haut/bas, valeur mini)", R.e1_min, 1.2, R.t_p, "e", True,
         ["e1_u", "hp_u", "n1_u", "p1_u"], ["e1", "e1bot"])
    drow("e2p", "Platine – e2 (bord latéral)", R.e_2, 1.2, R.t_p, "e", True,
         ["bp_u", "p3_u"], ["e2p"])
    drow("eP", "Poteau – pince au bout d'aile e = (bP − p3)/2", R.e_P, 1.2, R.tf_P, "e",
         R.aile, ["p3_u"] + prP, ["eP", "p3"])
    drow("p1", "Entraxe p1", R.p_1, 2.2, tmin, "p", R.n_1 > 1, ["p1_u"], ["p1"])
    drow("p3", "Entraxe des files p3 (p2 du Tableau 3.3)", R.p_3, 2.4, tmin, "p", True, ["p3_u"], ["p3"])
    R.dist_ok = all(x.ok for x in R.dist)

    # ---- alertes
    tfrS = R.tf_S + R.r_S
    m12 = 1.2 * d0
    # gorge de PLEINE RÉSISTANCE vis-à-vis de l'âme portée (MSB P5 §2.2) —
    # recommandation : elle couvre aussi le tying sans vérification propre
    R.fuw, R.bww = RS.fu_bw_cordon(R.fu_L, R.bw_L, R.fu_S, R.bw_S)
    R.k_plein = MEC.coef_pleine_resistance(R.fy_S, R.fuw, R.bww, gM2, gM0)
    R.a_plein = R.k_plein * R.tw_S
    botLim = R.h_S - (u.d_nb if u.d_nb > 0 else tfrS); sumC = u.z_C + R.h_p
    AL = [
        dict(id="e1bot", c=R.e1bot < m12, b=1,
             m="Configuration impossible : entraxe vertical / nombre de rangées incompatible avec la hauteur de la platine hp (pince basse < 1,2·d0).",
             f=["hp_u", "n1_u", "p1_u", "e1_u"], d=["hp", "e1", "p1", "e1bot"], e=["plate", "bolts"],
             cv=["e1p"] if R.e_1 >= m12 - 0.001 else [],
             w="Pince basse = hp − e1 − (n1 − 1)·p1 = " + f0(R.h_p) + " − " + f0(R.e_1) + " − " + js_str(R.n_1 - 1) + " × " + f0(R.p_1) + " = " + f0(R.e1bot) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. hp minimal avec ces boulons : " + f0(math.ceil(R.e_1 + (R.n_1 - 1) * R.p_1 + m12)) + " mm."),
        dict(id="e2p", c=R.e_2 < m12, b=1,
             m="Configuration impossible : platine trop étroite – pince latérale e2 non conforme à EN 1993-1-8 Tableau 3.3.",
             f=["bp_u", "p3_u"], d=["e2p", "bp", "p3"], e=["plate", "bolts"], cv=["e2p"],
             w="Pince e2 = (bp − p3)/2 = (" + f0(R.b_p) + " − " + f0(R.p_3) + ")/2 = " + f0(R.e_2) + " mm, alors qu'il faut au moins 1,2·d0 = " + F(m12, 1) + " mm. bp minimal avec cet entraxe : " + f0(math.ceil(R.p_3 + 2 * m12)) + " mm."),
        dict(id="p3_cordon", c=R.p_3 < R.p3_min - 1e-9, b=1,
             m="Configuration impossible : entraxe des files p3 trop faible – la rondelle chevauche le cordon de l'âme portée.",
             f=["p3_u", "a_w"] + prS, d=["p3", "aw"], e=["bolts", "weld"],
             w="Il faut p3 ≥ tw + 2·a·√2 + dw = " + F(R.tw_S, 1) + " + 2 × " + f0(R.a_w) + " × 1,414 + " + f0(R.d_w) + " = " + F(R.p3_min, 1) + " mm (dw : diamètre de la rondelle) ; actuel : p3 = " + f0(R.p_3) + " mm."),
        # LE gauge du cas (a) : l'écrou est du côté intérieur de l'aile, au
        # droit du congé de raccordement — c'est pour cela que les gauges
        # usuels d'aile valent 90 / 100 / 120 / 140 selon la série
        dict(id="p3_congeP", c=R.aile and R.p_3 < R.p3_congeP - 1e-9, b=1,
             m="Configuration impossible : l'écrou chevauche le congé de raccordement du poteau – entraxe des "
               "files p3 trop faible pour cette aile.",
             f=["p3_u", "boulon_u"] + prP, d=["p3", "twP"], e=["bolts", "colFl", "colWeb"],
             w="Il faut p3 ≥ twP + 2·rP + dw = " + F(R.tw_P, 1) + " + 2 × " + F(R.r_P, 1) + " + " + f0(R.d_w)
             + " = " + F(R.p3_congeP, 1) + " mm (l'écrou est du côté intérieur de l'aile, au droit du congé) ; "
             + "actuel : p3 = " + f0(R.p_3) + " mm. Prendre p3 = " + f0(ceil5(R.p3_congeP))
             + " mm au moins — gauges usuels d'aile : 90 / 100 / 120 / 140 mm selon la série."),
        dict(id="p3_libreP", c=R.ame and R.p_3 / 2 + R.d_w / 2 > R.demi_droit + 1e-9, b=1,
             m="Configuration impossible : l'écrou chevauche le congé de l'âme du poteau – entraxe des files p3 "
               "trop grand pour la largeur droite de l'âme.",
             f=["p3_u", "boulon_u"] + prP, d=["p3", "libreP"], e=["bolts", "colWeb"],
             w="Il faut p3/2 + dw/2 ≤ (hP − 2·tfP)/2 − rP = " + F(R.demi_droit, 1) + " mm ; actuel : "
             + F(R.p_3 / 2, 1) + " + " + F(R.d_w / 2, 1) + " = " + F(R.p_3 / 2 + R.d_w / 2, 1)
             + " mm. p3 maximal : " + f0(math.floor(R.p3_maxP)) + " mm."),
        dict(id="bp_libreP", c=R.ame and R.b_p > R.bp_maxP + 1e-9, b=1,
             m="Configuration impossible : la platine ne tient pas dans la largeur droite de l'âme du poteau, "
               "entre les congés.",
             f=["bp_u", "p3_u", "j_lat"] + prP, d=["bp", "libreP", "jlat"], e=["plate", "colWeb"],
             w="Il faut bp ≤ hP − 2·tfP − 2·rP − 2·j = " + F(R.libre_droit, 0) + " − 2 × " + f0(u.j_lat)
             + " = " + F(R.bp_maxP, 0) + " mm ; actuel : bp = " + f0(R.b_p) + " mm."),
        dict(id="bp_aile", c=R.aile and R.b_p > R.b_P + 1e-9, b=0,
             m="La platine dépasse le bout d'aile du poteau : détail inhabituel – le guide pose bp ≤ bP "
               "(la platine reste dans la largeur de l'aile).",
             f=["bp_u"] + prP, d=["bp", "bP"], e=["plate", "colFl"],
             w="bp = " + f0(R.b_p) + " mm > bP = " + f0(R.b_P) + " mm."),
        dict(id="zc_top", c=u.z_C < (u.d_nt if u.d_nt > 0 else tfrS), b=1,
             m="Configuration impossible : le dessus de la platine est dans la zone grugée ou dans le congé supérieur de la poutre portée.",
             f=["z_C", "d_nt"] + prS, d=["zp", "dcs"], e=["plate", "notchT"],
             w=("zp = " + f0(u.z_C) + " mm est inférieur à la profondeur du grugeage dc,sup = " + f0(u.d_nt) + " mm : il faut zp ≥ dc,sup.") if u.d_nt > 0
             else ("zp = " + f0(u.z_C) + " mm est inférieur à tf + r = " + f0(tfrS) + " mm (semelle et congé de la poutre portée) : il faut zp ≥ " + f0(math.ceil(tfrS)) + " mm.")),
        dict(id="h_dispo", c=sumC > botLim, b=1,
             m="Configuration impossible : hauteur disponible insuffisante – la platine dépasse la partie droite de l'âme de la poutre portée.",
             f=["hp_u", "z_C"] + (["d_nb"] if u.d_nb > 0 else []) + prS, d=["hp", "zp"] + (["dci"] if u.d_nb > 0 else []), e=["plate"],
             w="zp + hp = " + f0(u.z_C) + " + " + f0(R.h_p) + " = " + f0(sumC) + " mm dépasse la limite basse de la partie droite de l'âme : "
             + (("h − dc,inf = " + f0(R.h_S) + " − " + f0(u.d_nb)) if u.d_nb > 0 else ("h − (tf + r) = " + f0(R.h_S) + " − " + f0(tfrS)))
             + " = " + f0(botLim) + " mm. Avec zp = " + f0(u.z_C) + " mm, hp maximal = " + f0(math.floor(botLim - u.z_C)) + " mm."),
        dict(id="aile_poutre", c=R.ame and not R.rec_ok, b=1,
             m=TXT.M_AILE_POUTRE,
             f=["rec_S", "j_lat"] + prS + prP, d=["rec", "jlat"], e=["beamS", "recut", "col"],
             w=TXT.w_aile_poutre(f0(R.b_S), f0(R.libre_ailes), "tp", F(R.rec_min, 1), f0(u.rec_S)) + "."),
        dict(id="d0_manq", c=u.trou == "Surdimensionné" and N(u.d0_u) == 0, b=1,
             m="Trou surdimensionné : saisir le diamètre d0.", f=["trou", "d0_u"], d=[], e=[], w=""),
        dict(id="tp_duct", c=not R.rec_tp, b=0,
             m="Platine trop épaisse pour la ductilité de l'articulation : la rotation doit venir de la déformation de la platine, pas des boulons.",
             f=["tp_u", "boulon_u", "nu_pl", "classe"], d=["tp"], e=["plate"],
             w="tp = " + f0(R.t_p) + " mm > 0,36·d·√(fub/fy) = 0,36 × " + f0(d) + " × √(" + f0(R.f_ub) + "/" + f0(R.fy_L) + ") = " + F(R.tp_max, 1) + " mm (EN 1993-1-8 §6.4.2(2) ; ECCS n°126)."),
        dict(id="tp_min", c=R.t_p < 8, b=0,
             m="Platine de moins de 8 mm : déconseillée (déformation au soudage, dommages au transport) — MSB P5 §2.1 recommande 10 ou 12 mm.",
             f=["tp_u"], d=["tp"], e=["plate"], w="tp = " + f0(R.t_p) + " mm < 8 mm."),
        dict(id="hp_rec", c=not R.rec_hp, b=0,
             m="hp < 0,6·h de la poutre portée : recommandation du guide (maintien en torsion de la poutre à l'appui) non respectée.",
             f=["hp_u"] + prS, d=["hp"], e=["plate"],
             w="hp = " + f0(R.h_p) + " mm < 0,6·h = " + F(0.6 * R.h_S, 0) + " mm (MSB P5 §2.1)."),
        dict(id="a_rec", c=R.a_w < R.a_plein - 1e-9, b=0,
             m="Gorge sous la PLEINE RÉSISTANCE recommandée par le guide — admissible si la vérification du cordon passe, mais le tying n'est alors plus couvert d'office.",
             f=["a_w"] + prS, d=["aw"], e=["weld"],
             w="Recommandé : a ≥ " + F(R.k_plein, 2) + "·tw = " + F(R.a_plein, 1) + " mm (βw·γM2·fy/(√2·fu) — MSB P5 §2.2) ; actuel : a = " + f0(R.a_w) + " mm."),
        dict(id="ame_hp", c=R.ame, b=0,
             m=TXT.m_ame_hp("la traction des boulons", "raidisseurs entre ailes",
                            "Le tronçon en T de l'aile et l'âme en traction transversale sont "
                            "sans objet dans ce cas d'appui."),
             f=["appui"] + prP, d=["libreP"], e=["col", "colWeb"], w=""),
        dict(id="NEd", c=NEd > 0 and not R.tying, b=0,
             m="NEd de traction à l'ELU fondamental : boulons, platine (tronçon en T), poinçonnement, âme portée "
               "et cordons sont vérifiés ; en cas (a) s'ajoutent le tronçon en T de l'aile du poteau et l'âme du "
               "poteau en traction transversale.",
             f=["N_Ed", "nat_N"], d=[], e=[], w=""),
        dict(id="N_tying", c=R.tying, b=0,
             m="NEd en TYING (situation accidentelle) : la force de chaînage est vérifiée SEULE, sans VEd, avec "
               "les résistances ultimes et γMu (MSB Part 5 §2.3).",
             f=["N_Ed", "nat_N", "g_Mu"], d=[], e=[],
             w="NEd = " + F(NEd, 1) + " kN ; γMu = " + F(gMu, 2) + "."),
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

    # ---- boulons (1 plan de cisaillement, groupe centré)
    R.Fv_Rd = RS.Fv_Rd(R.a_v, R.f_ub, R.A_cis, gM2)
    R.Ft_Rd = RS.Ft_Rd(R.f_ub, R.As_b, gM2)
    R.Ft_Rd_u = RS.Ft_Rd(R.f_ub, R.As_b, gMu)              # tying (γMu)
    R.Fp_C = RS.Fp_C(R.f_ub, R.As_b)
    R.Lj = (R.n_1 - 1) * R.p_1
    R.bLf = RS.beta_Lf(R.Lj, d, u.opt_blf == "Oui")
    R.Fv_Ed = dv(V, R.n_b)                      # effort tranchant par boulon
    R.Ft_Ed = dv(R.N_elu, R.n_b)                # traction ELU directe par boulon

    # ---- pression diamétrale (Tableau 3.4) : PLATINE — boulons d'extrémité
    # ---- (pince e1, mini haut/bas) et intérieurs (p1), k1 de rive des deux files
    pl = MEC.portance_bout_interieur(R.t_p, R.fu_L, R.f_ub, d, d0, R.e1_min, R.e_2,
                                     R.p_1, R.p_3, gM2, R.k_trou)
    R.k1_L = pl["k1"]; R.abb_L = pl["ab_bout"]; R.abi_L = pl["ab_int"]
    R.Fbb_L = pl["Fb_bout"]; R.Fbi_L = pl["Fb_int"]
    n_bout = 2 if R.n_1 > 0 else 0              # rangée d'extrémité, deux files
    n_int = R.n_b - n_bout
    R.FRd_g, R.branche_g = MEC.groupe_boulons(R.Fv_Rd * R.bLf, R.Fbb_L, R.Fbi_L, n_bout, n_int)

    # ---- pression diamétrale du POTEAU : AUCUNE pince d'extrémité verticale
    # ---- (le poteau continue au-dessus et en dessous de l'attache), αb du seul
    # ---- entraxe p1 pour TOUTES les rangées, k1 du gauge et — en cas (a)
    # ---- seulement — de la pince au bout d'aile (étude §4.2 et §5.2)
    bP = MEC.bearing(R.t_appui, R.fu_P, R.f_ub, d, d0, None, R.p_1, R.n_1,
                     R.e_P if R.aile else None, R.p_3, 2, gM2, R.k_trou)
    R.ab_P = bP["abv"]; R.k1_P = bP["k1v"]; R.Fb_P = bP["Fbv"]
    R.VRd_pdP = R.n_b * R.Fb_P

    # ---- platine : aires (deux plans de cisaillement, un par côté de l'âme)
    R.Av_p = R.h_p * R.t_p
    R.Avn_p = (R.h_p - R.n_1 * d0) * R.t_p
    R.Ant_p = R.t_p * (R.e_2 - 0.5 * d0)
    R.Anv_p = R.t_p * (R.h_p - R.e_1 - (R.n_1 - 0.5) * d0)
    # k = 1 (§3.10.2(2), effort centré) si la flexion de la platine est
    # négligeable (hp ≥ 1,36·p3) ; 0,5 (§3.10.2(3)) sinon — MSB P5 §2.2
    R.k_bloc = 0.5 if (not R.hp_136 and R.n_1 > 1) else 1.0
    R.VRd_bp = 2 * (dv(R.k_bloc * R.fu_L * R.Ant_p, gM2n) + dv(R.fy_L * R.Anv_p / S3, gM0)) / 1000

    # ---- poutre portée (section grugée : mêmes règles que les modules
    # ---- existants, MSB Part 5 §4.2.4 / §4.2.5)
    sg = MEC.section_grugee(dict(h=R.h_S, b=R.b_S, tw=R.tw_S, tf=R.tf_S, r=R.r_S),
                            R.fy_S, u.d_nt, u.d_nb, gM0, V, u.lt_ok == "Oui")
    R.cas_g = sg["cas_g"]; R.Asec_S = sg["Asec"]; R.h_T = sg["h_T"]; R.A_Tee = sg["A_Tee"]
    R.Av_S = sg["Av"]; R.VRd_gS = sg["VRd_g"]; R.hw_T = sg["hw_T"]; R.zb_T = sg["zb_T"]
    R.I_N = sg["I_N"]; R.W_N = sg["W_N"]; R.rho_N = sg["rho_N"]; R.Mv_N = sg["Mv_N"]
    R.el_N = sg["el_N"]; R.lim_N = sg["lim_N"]; R.ln_max = sg["ln_max"]; R.val_N = sg["val_N"]
    R.Av_w = R.tw_S * R.h_p                     # âme portée au droit de la platine
    R.e_N = R.t_p + u.l_n                       # face d'appui → extrémité du grugeage

    # ---- POTEAU, cas (b) : cisaillement local de l'âme, un seul côté chargé
    # ---- (SCI P358, même composant que l'âme porteuse des modules
    # ---- poutre–poutre) ; la section NETTE existe : les boulons y percent
    R.Av_P = R.tw_P * R.h_p
    R.Avn_P = R.tw_P * (R.h_p - R.n_1 * d0)

    # ---- soudure (deux cordons verticaux sur toute la hauteur hp, gorge a)
    R.qz_w = dv(V * 1000, 2 * R.h_p)
    R.qx_w = dv(R.N_elu * 1000, 2 * R.h_p)
    R.qw = math.hypot(R.qz_w, R.qx_w)
    R.fvw = RS.fvw_d(R.fuw, R.bww, gM2)
    R.Fw = R.fvw * R.a_w
    R.qx_ty = dv(R.N_ty * 1000, 2 * R.h_p)
    R.fvw_u = RS.fvw_d(R.fuw, R.bww, gMu)
    R.Fw_ty = R.fvw_u * R.a_w

    # ---- traction : les DEUX tronçons en T (platine, puis AILE DU POTEAU)
    NT = R.N_ty if R.tying else R.N_elu
    gT = gMu if R.tying else gM2n               # γ des résistances ultimes
    R.g_T = gT
    fT = R.fu_L if R.tying else R.fy_L          # platine : fu/γMu ou fy/γM0
    gTT = gMu if R.tying else gM0
    FtT = R.Ft_Rd_u if R.tying else R.Ft_Rd
    R.SFt = R.n_b * FtT
    R.update(troncon_platine(R, fT, gTT, FtT))
    R.update(troncon_aile_poteau(R, R.fu_P if R.tying else R.fy_P, gMu if R.tying else gM0, FtT))
    # âme du poteau en traction transversale (§6.2.6.3) : ω = 1 dans le
    # périmètre de l'outil (une attache simple n'apporte pas de moment
    # d'étage, donc aucun cisaillement de panneau) — point P-5
    R.omega_P = 1.0
    R.beff_P = R.leff_fP
    R.Ft_wc = dv(R.omega_P * R.beff_P * R.tw_P * (R.fu_P if R.tying else R.fy_P),
                 gMu if R.tying else gM0) / 1000

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
    ty = " (tying : fu/γMu)" if R.tying else ""
    cas_P = "l'aile" if R.aile else "l'âme"
    # --- Boulons
    ck("bg", "Boulons", "Groupe de boulons – cisaillement et pression diamétrale sur la platine", V, R.FRd_g, "kN",
       T34 + " ; " + MSB_GRP, "EC + COMP", True,
       "FRd = ΣFb,Rd si (Fb,Rd)max ≤ Fv,Rd ; ns·(Fb,Rd)min si (Fb,Rd)min < Fv,Rd < (Fb,Rd)max ; "
       "0,8·ns·Fv,Rd si Fv,Rd ≤ (Fb,Rd)min ; Fb,Rd = k1·αb·fu·d·tp/γM2",
       "Fv,Rd = " + F(R.Fv_Rd, 2) + " kN ; Fb,Rd extrémité = " + F(R.Fbb_L, 1) + " kN (αb = " + F(R.abb_L, 3)
       + ") ; intérieur = " + F(R.Fbi_L, 1) + " kN (αb = " + F(R.abi_L, 3) + ") ; k1 = " + F(R.k1_L, 3)
       + " ; ns = " + js_str(R.n_b) + " → " + R.branche_g, 1)
    ck("gl", "Boulons", "Glissement (1 surface de frottement)", R.Fv_Ed * ks,
       dv(u.k_s * u.mu_s * mx(0, R.Fp_C - 0.8 * R.Ft_Ed), g3), "kN",
       "EN 1993-1-8 §3.9.1 ; §3.9.2 (traction)", "EC", u.cat != "A",
       "Fs,Rd = ks·n·μ·(Fp,C − 0,8·Ft,Ed)/γM3 ; Fp,C = 0,7·fub·As",
       "Fp,C = " + F(R.Fp_C, 1) + " kN ; Ft,Ed = " + F(R.Ft_Ed, 2) + " kN ; catégorie " + u.cat)
    ck("bvt", "Boulons", "Cisaillement + traction ELU (interaction)",
       (R.Fv_Ed / R.Fv_Rd + R.Ft_Ed / (1.4 * R.Ft_Rd)) if R.Fv_Rd > 0 and R.Ft_Rd > 0 else INF,
       1, "-", T34, "EC", R.N_elu > 0,
       "Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1 ; Ft,Ed = NEd/n (effet de levier : couvert par le tronçon en T)",
       "Fv,Ed = " + F(R.Fv_Ed, 2) + " kN ; Ft,Ed = " + F(R.Ft_Ed, 2) + " kN ; Ft,Rd = " + F(R.Ft_Rd, 2) + " kN")
    # --- Platine d'about
    ck("pcg", "Platine d'about", "Cisaillement section brute (2 plans)", V,
       2 * dv(R.Av_p * R.fy_L, K_CIS * S3 * gM0) / 1000, "kN",
       MSB2, "COMP", True, "VRd,g = 2·(hp·tp/1,27)·fy/(√3·γM0) — 1,27 : flexion nominale de la platine dans son plan",
       "hp = " + F(R.h_p, 0) + " mm ; tp = " + F(R.t_p, 0) + " mm ; fy = " + F(R.fy_L, 0) + " MPa", 1)
    ck("pcn", "Platine d'about", "Cisaillement section nette (2 plans)", V,
       2 * dv(R.Avn_p * R.fu_L, S3 * gM2n) / 1000, "kN",
       MSB2, "COMP", True, "VRd,n = 2·tp·(hp − n1·d0)·fu/(√3·γM2)",
       "Av,net = " + F(R.Avn_p, 0) + " mm² par plan", 1)
    ck("pcb", "Platine d'about", "Rupture de bloc (2 blocs)", V, R.VRd_bp, "kN",
       "EN 1993-1-8 §3.10.2 ; k et aires : " + MSB2, "EC + COMP", True,
       "VRd,b = 2·[k·fu·Ant/γM2 + fy·Anv/(√3·γM0)] ; k = 1 si hp ≥ 1,36·p3 (ou n1 = 1), 0,5 sinon ; "
       "Ant = tp·(e2 − d0/2) ; Anv = tp·(hp − e1 − (n1 − 0,5)·d0)",
       "k = " + F(R.k_bloc, 1) + " (hp " + ("≥" if R.hp_136 else "<") + " 1,36·p3 = " + F(1.36 * R.p_3, 1)
       + " mm) ; Ant = " + F(R.Ant_p, 0) + " mm² ; Anv = " + F(R.Anv_p, 0) + " mm²", 1)
    ck("ptT", "Platine d'about", "Flexion de la platine en traction – tronçon en T" + ty, NT, R.FT_Rd, "kN",
       ("EN 1993-1-8 §6.2.4 (Tableau 6.2, méthode 2) ; " + MSB_TY) if R.tying else
       "EN 1993-1-8 §6.2.4 (Tableau 6.2, méthode 2) ; Σleff : Tableau 6.4 appliqué à la platine (interprétation)",
       "EC + COMP" if R.tying else "EC + INT", NT > 0,
       "FT,Rd = min(FT,1 ; FT,2 ; FT,3) ; FT,1 = (8n − 2ew)·Mpl,1/(2mn − ew(m + n)) ; "
       "FT,2 = (2·Mpl,2 + n·ΣFt,Rd)/(m + n) ; FT,3 = ΣFt,Rd ; Mpl = 0,25·Σleff·tp²·f/γ",
       "m = " + F(R.m_T, 1) + " mm ; n = " + F(R.nT, 1) + " mm ; ew = " + F(R.ew_T, 2) + " mm ; Σleff = "
       + F(R.leff_T, 0) + " mm ; FT,1 = " + F(R.FT1, 1) + " ; FT,2 = " + F(R.FT2, 1) + " ; FT,3 = "
       + F(R.FT3, 1) + " kN", 1)
    ck("pp", "Platine d'about", "Poinçonnement sous la tête ou l'écrou (traction)" + ty, dv(NT, R.n_b),
       RS.Bp_Rd(R.d_m, mn(R.t_p * R.fu_L, R.t_appui * R.fu_P), gT), "kN", T34, "EC", NT > 0,
       "Bp,Rd = 0,6·π·dm·t·fu/γ (pièce la plus faible sous la tête ou l'écrou : platine ou poteau)",
       "dm = " + F(R.d_m, 1) + " mm ; platine " + F(R.Bp_pl, 1) + " kN ; poteau (" + cas_P + ", t = "
       + F(R.t_appui, 1) + " mm) " + F(R.Bp_P, 1) + " kN")
    # --- Poutre portée
    ck("vwS", "Poutre portée", "Âme – cisaillement au droit de la platine (Av = hp·tw)", V,
       dv(R.Av_w * R.fy_S / S3, gM0) / 1000, "kN",
       "EN 1993-1-1 §6.2.6(1) ; Av = hp·tw : MSB Part 5 §2.2.1", "EC + COMP", True,
       "VRd = hp·tw·fy/(√3·γM0) — l'âme ne transmet l'effort qu'à la platine",
       "Av = " + F(R.Av_w, 0) + " mm²", 1)
    ck("vgS", "Poutre portée", "Grugeage – cisaillement de la section réduite", V, R.VRd_gS, "kN",
       "EN 1993-1-1 §6.2.6", "EC", R.cas_g > 0,
       "VRd = Av,N·fy/(√3·γM0)", "Av,N = " + F(R.Av_S, 0) + " mm²")
    ck("mN", "Poutre portée", "Poutre grugée – flexion + cisaillement à l'extrémité du grugeage",
       V * R.e_N / 1000, R.Mv_N, "kNm",
       "MSB Part 5 §4.2.4 (mêmes règles pour le grugeage)", "COMP", R.cas_g > 0,
       "MEd = VEd·(tp + c) ; Mv,N,Rd = fy·Wel,N/γM0·[1 − (2VEd/Vpl,N,Rd − 1)²] si VEd > 0,5·Vpl,N,Rd",
       "tp + c = " + F(R.e_N, 0) + " mm ; Wel,N = " + F(R.W_N, 0) + " mm³ ; Vpl,N,Rd = " + F(R.VRd_gS, 1)
       + " kN ; réduction = " + F(R.rho_N, 3), 1)
    ck("stN", "Poutre portée", "Stabilité locale du grugeage – longueur c", u.l_n, R.ln_max, "mm",
       "MSB Part 5 §4.2.5", "COMP", R.cas_g > 0,
       "c ≤ h si h/tw ≤ limite ; sinon c ≤ K·h/(h/tw)³",
       "h/tw = " + F(R.el_N, 1) + " ; limite = " + F(R.lim_N, 1) + " ; domaine : " + R.val_N, 1)
    ck("stD", "Poutre portée", "Stabilité locale du grugeage – profondeur", mx(u.d_nt, u.d_nb),
       R.h_S / 2 if R.cas_g == 1 else R.h_S / 5, "mm", "MSB Part 5 §4.2.5", "COMP", R.cas_g > 0,
       "dc ≤ h/2 (1 semelle grugée) ; dc ≤ h/5 (2 semelles)", "")
    ck("tnS", "Poutre portée", "Âme – traction au droit de la platine" + ty, NT,
       (dv(R.Av_w * R.fu_S, gMu) if R.tying else dv(R.Av_w * R.fy_S, gM0)) / 1000, "kN",
       ("EN 1993-1-1 §6.2.3 ; " + MSB_TY) if R.tying else "EN 1993-1-1 §6.2.3 ; hauteur participante = hp",
       "EC + COMP", NT > 0,
       "Nu,Rd = tw·hp·fu/γMu" if R.tying else "Npl,Rd = tw·hp·fy/γM0", "Av = " + F(R.Av_w, 0) + " mm²")
    # --- Poteau porteur
    ck("pdP", "Poteau porteur", "Pression diamétrale – " + cas_P + " du poteau", V, R.VRd_pdP, "kN",
       T34 + " ; SCI P358 (élément porteur)", "EC", True,
       "VRd = ns·Fb,Rd ; Fb,Rd = k1·αb·fu,P·d·t/γM2 ; AUCUNE pince d'extrémité verticale (le poteau continue) : "
       "αb = min(p1/3d0 − 1/4 ; fub/fu ; 1) pour toutes les rangées"
       + (" ; k1 = min(2,8·e/d0 − 1,7 ; 1,4·p3/d0 − 1,7 ; 2,5) avec e = (bP − p3)/2" if R.aile
          else " ; k1 = min(1,4·p3/d0 − 1,7 ; 2,5) — aucun bord libre"),
       "t = " + F(R.t_appui, 1) + " mm (" + cas_P + ") ; αb = " + F(R.ab_P, 3) + " ; k1 = " + F(R.k1_P, 2)
       + " ; Fb,Rd = " + F(R.Fb_P, 1) + " kN par boulon ; ns = " + js_str(R.n_b), 1)
    ck("vlP", "Poteau porteur", "Âme du poteau – cisaillement local brut (un seul côté chargé)", V / 2,
       dv(R.Av_P * R.fy_P / S3, gM0) / 1000, "kN", P358_SUP, "COMP", R.ame,
       "VEd/2 ≤ Av·fy,P/(√3·γM0) ; Av = twP·hp",
       "Av = " + F(R.Av_P, 0) + " mm² ; twP = " + F(R.tw_P, 1) + " mm ; fy,P = " + F(R.fy_P, 0) + " MPa", 1)
    ck("vnP", "Poteau porteur", "Âme du poteau – cisaillement local net", V / 2,
       dv(R.Avn_P * R.fu_P / S3, gM2n) / 1000, "kN", P358_SUP, "COMP", R.ame,
       "VEd/2 ≤ Av,net·fu,P/(√3·γM2) ; Av,net = twP·(hp − n1·d0)", "Av,net = " + F(R.Avn_P, 0) + " mm²")
    ck("tfP", "Poteau porteur", "Aile du poteau en flexion – tronçon en T" + ty, NT, R.FT_Rd_P, "kN",
       "EN 1993-1-8 §6.2.4 (Tableau 6.2, méthode 2) ; Σleff : " + TAB64
       + (" ; " + MSB_TY if R.tying else ""), "EC + COMP", R.aile and NT > 0,
       "m = (p3 − twP)/2 − 0,8·rP ; e = (bP − p3)/2 ; n = min(e ; 1,25·m) ; AUCUNE rangée d'extrémité "
       "(le poteau continue) : rangées isolées 2πm et 4m + 1,25e, en groupe 2p1 et p1 ; "
       "FT,Rd = min(FT,1 ; FT,2 ; FT,3)",
       "m = " + F(R.m_fP, 1) + " mm ; n = " + F(R.n_fP, 1) + " mm ; Σleff = " + F(R.leff_fP, 0)
       + " mm ; Mpl,1 = " + F(R.Mpl1_fP / 1e6, 2) + " kNm ; FT,1 = " + F(R.FT1_P, 1) + " ; FT,2 = "
       + F(R.FT2_P, 1) + " ; FT,3 = " + F(R.FT3_P, 1) + " kN", 1)
    ck("twtP", "Poteau porteur", "Âme du poteau en traction transversale" + ty, NT, R.Ft_wc, "kN",
       "EN 1993-1-8 §6.2.6.3 ; ω = 1 et beff = Σleff de l'aile : interprétation de l'outil",
       "EC + INT", R.aile and NT > 0,
       "Ft,wc,Rd = ω·beff,t,wc·twP·f/γ ; ω = 1 (une attache simple n'apporte pas de moment d'étage, "
       "donc aucun cisaillement de panneau) ; beff,t,wc = Σleff de l'aile",
       "beff = " + F(R.beff_P, 0) + " mm ; twP = " + F(R.tw_P, 1) + " mm ; ω = " + F(R.omega_P, 2))
    # --- Soudure
    ck("w", "Soudure", "Cordons âme portée / platine – effort par unité de longueur", R.qw, R.Fw, "N/mm",
       W445, "EC", True,
       "Fw,Ed = √(qz² + qx²) ≤ Fw,Rd = fvw,d·a ; qz = VEd/(2hp) ; qx = NEd,ELU/(2hp) ; fvw,d = fu/(√3·βw·γM2)",
       "a = " + F(R.a_w, 1) + " mm ; qz = " + F(R.qz_w, 0) + " ; qx = " + F(R.qx_w, 0) + " N/mm ; fvw,d = "
       + F(R.fvw, 1) + " MPa", 1)
    ck("wT", "Soudure", "Cordons – tying (force de chaînage seule)", R.qx_ty, R.Fw_ty, "N/mm",
       W445 + " avec γMu ; " + MSB_TY, "EC + INT", R.tying,
       "qx = NEd/(2hp) ≤ fu/(√3·βw·γMu)·a — le guide l'admet d'office pour un cordon à pleine résistance",
       "qx = " + F(R.qx_ty, 0) + " N/mm ; a = " + F(R.a_w, 1) + " mm ; γMu = " + F(gMu, 2))
    ck("wa", "Soudure", "Gorge minimale", 3, R.a_w, "mm", "EN 1993-1-8 §4.5.2(2)", "EC", True, "a ≥ 3 mm", "")

    # ---- synthèse : la gorge minimale (3 mm) est un minimum de détail, pas
    # une résistance — elle ne devient « dimensionnante » que si elle échoue
    act = [c for c in R.checks if c.active]
    gov = None
    for c in act:
        if c.key in DETAIL_SEUL and c.ok:
            continue
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
#  tronçon en T de la PLATINE (traction)
# =========================================================================
def troncon_platine(R, f, gamma, Ft):
    """Résistance en traction de la PLATINE : un tronçon en T dont l'âme est
    celle de la poutre portée et dont les deux ailes sont les deux moitiés de
    la platine, une file de boulons de chaque côté (EN 1993-1-8 §6.2.4,
    Tableau 6.2 ; la résistance porte les DEUX côtés).

    - m = (p3 − tw)/2 − 0,8·a·√2 (Figure 6.2, âme soudée) ; e = e2 ;
      n = min(e2 ; 1,25·m) ; ew = dw/4 ;
    - Σleff : platine à bords libres en haut et en bas
      (``mecanique.leff_platine_bords_libres``) — pour les détails courants
      Σleff = hp, la valeur du guide (MSB P5 §2.3) ;
    - ``f`` / ``gamma`` : fy/γM0 à l'ELU, fu/γMu en tying ; ``Ft`` : Ft,Rd
      d'un boulon (γM2 à l'ELU, γMu en tying)."""
    m = mx((R.p_3 - R.tw_S) / 2 - 0.8 * R.a_w * math.sqrt(2), 1e-6)
    e = R.e_2
    n = mx(1e-6, mn(e, 1.25 * m))
    ew = R.d_w / 4
    l1, l2 = MEC.leff_platine_bords_libres(R.n_1, R.p_1, R.e_1, R.e1bot, m, e)
    t = MEC.troncon_T(l1, l2, R.t_p, m, n, ew, f, gamma, R.n_b * Ft)
    return dict(m_T=m, nT=n, ew_T=ew, leff_T=l1, leff2_T=l2,
                Mpl1_T=t["Mpl1"], FT1=t["FT1"], FT2=t["FT2"], FT3=t["FT3"], FT_Rd=t["FT_Rd"],
                Bp_pl=RS.Bp_Rd(R.d_m, R.t_p * R.fu_L, gamma if R.tying else R.u.g_M2n))


# =========================================================================
#  tronçon en T de l'AILE DU POTEAU — le composant NEUF (étude §7.3)
# =========================================================================
def troncon_aile_poteau(R, f, gamma, Ft):
    """Résistance en traction de l'AILE DU POTEAU, cas (a) : tronçon en T
    d'une aile de poteau NON RAIDIE (EN 1993-1-8 §6.2.4, Tableau 6.2,
    méthode 2 ; longueurs efficaces du Tableau 6.4).

    - m = (p3 − twP)/2 − 0,8·rP (Figure 6.8, poteau LAMINÉ : le congé prend
      la place du cordon) ; e = (bP − p3)/2 ; n = min(e ; 1,25·m) ; ew = dw/4 ;
    - Σleff : un poteau continue au-dessus et en dessous de l'attache — il
      n'y a AUCUNE rangée d'extrémité, toutes les rangées sont INTÉRIEURES
      (``mecanique.leff_aile_poteau``) ;
    - ``f`` / ``gamma`` : fy,P/γM0 à l'ELU, fu,P/γMu en tying."""
    m = mx((R.p_3 - R.tw_P) / 2 - 0.8 * R.r_P, 1e-6)
    e = R.e_P
    n = mx(1e-6, mn(e, 1.25 * m))
    ew = R.d_w / 4
    l1, l2 = MEC.leff_aile_poteau(R.n_1, R.p_1, m, e)
    t = MEC.troncon_T(l1, l2, R.tf_P, m, n, ew, f, gamma, R.n_b * Ft)
    return dict(m_fP=m, n_fP=n, e_fP=e, ew_fP=ew, leff_fP=l1, leff2_fP=l2,
                Mpl1_fP=t["Mpl1"], FT1_P=t["FT1"], FT2_P=t["FT2"], FT3_P=t["FT3"],
                FT_Rd_P=t["FT_Rd"],
                Bp_P=RS.Bp_Rd(R.d_m, R.t_appui * R.fu_P, gamma if R.tying else R.u.g_M2n))


# =========================================================================
#  predim
# =========================================================================
def predim(u, R):
    """Prédimensionnement : pour chaque boulon (plage choisie) et chaque
    nombre de rangées (2 → 8), pinces et entraxes de détail (multiples de 5),
    entraxe des files p3 — 90 mm au moins (valeur standard du guide), la
    rondelle dégage le cordon ET, en cas (a), l'écrou dégage le CONGÉ DU
    POTEAU (p3 ≥ twP + 2·rP + dw : c'est ce terme qui refuse le p3 = 90 usuel
    sur un HEB 300) —, hauteur hp ≥ 0,6·h ; taux estimés (groupe de boulons,
    âme portée, pression diamétrale du poteau, cisaillement local de l'âme du
    poteau en cas (b)), puis platine nécessaire (tp par les résistances,
    plafonné par la ductilité) et gorge de pleine résistance.

    Le poteau n'impose AUCUNE hauteur disponible : il n'a ni semelle
    au-dessus ni semelle en dessous de l'attache (l'alerte « pl_web » des
    modules poutre–poutre disparaît)."""
    rows = []
    ame = u.appui == APPUI_AME
    hav = R.h_S - (u.d_nb if u.d_nb > 0 else R.tf_S + R.r_S) - u.z_C
    V = u.V_Ed
    i1 = idx(DB["boulons"], u.pd_dmin); i2 = idx(DB["boulons"], u.pd_dmax)
    imin = min(i1, i2); imax = max(i1, i2)
    gM0 = u.g_M0; gM2 = u.g_M2; gM2n = u.g_M2n
    fuw, bww = RS.fu_bw_cordon(R.fu_L, R.bw_L, R.fu_S, R.bw_S)
    a_s = mx(3, math.ceil(MEC.coef_pleine_resistance(R.fy_S, fuw, bww, gM2, gM0) * R.tw_S - 1e-9))
    # géométrie du poteau : le prédimensionnement tourne AVANT compute(), les
    # scalaires de R ne sont pas encore posés — les deux largeurs se
    # recalculent ici (même expression, étude §5.3)
    libre_droit = MEC.geometrie_poteau(_profil_P(R))["libre_droit"]
    for i, b in enumerate(DB["boulons"]):
        d = b["d"]; d0 = b["d0"]; A = b["As"] if u.filet == "Oui" else b["A"]
        e1 = ceil5(u.k_e1 * d0); p1 = ceil5(u.k_p1 * d0); e2 = ceil5(u.k_e2 * d0)
        # p3 : minima cumulés (guide, Tableau 3.3, cordon, congé du poteau)
        p3 = mx(90, ceil5(R.tw_S + 2 * a_s * math.sqrt(2) + b["dw"]), ceil5(2.4 * d0),
                ceil5(MEC.geometrie_poteau(_profil_P(R), dw=b["dw"])["p3_congeP"]) if not ame else 0)
        # plafond : la platine reste dans l'aile (cas a) ou entre les congés
        # de l'âme (cas b), pince e2 ≥ 1,2·d0 comprise
        cap = (R.b_P - 2 * 1.2 * d0) if not ame else (libre_droit - 2 * u.j_lat - 2 * 1.2 * d0)
        etroit = bool(p3 > cap + 1e-9)
        p3 = mn(p3, mx(5.0, 5 * math.floor(mx(0.0, cap) / 5)))
        for n in range(2, 9):
            # première rangée près du haut (guide) ; la hauteur imposée par
            # hp ≥ 0,6·h (maintien en torsion) allonge la pince BASSE
            hp = mx(2 * e1 + (n - 1) * p1, ceil5(0.6 * R.h_S))
            geom = bool(hp <= hav and imin <= i <= imax and not etroit)
            nb = 2 * n
            Fv = dv(R.a_v * R.f_ub * A, gM2) / 1000
            # poteau : aucune pince d'extrémité verticale (il continue)
            abP = mn((p1 / (3 * d0) - 0.25) if n > 1 else 1, dv(R.f_ub, R.fu_P), 1)
            eP = (R.b_P - p3) / 2
            k1P = mn(INF if ame else 2.8 * eP / d0 - 1.7, 1.4 * p3 / d0 - 1.7, 2.5)
            t_ap = R.tw_P if ame else R.tf_P
            FbP = dv(k1P * abP * R.fu_P * d * t_ap, gM2) / 1000 * R.k_trou
            e = AttrDict(bg=dv(V, MEC.K_ROT * nb * Fv), pdP=dv(V, nb * FbP),
                         vwS=dv(V * 1000 * S3 * gM0, R.tw_S * hp * R.fy_S),
                         vlP=dv(V / 2 * 1000 * S3 * gM0, R.tw_P * hp * R.fy_P) if ame else 0.0)
            eta = mx(e.bg, e.pdP, e.vwS, e.vlP); ok = bool(geom and eta <= u.eta_c)
            rows.append(AttrDict(b=b, i=i, n=n, e1=e1, p1=p1, e2=e2, p3=p3, hp=hp,
                                 geom=geom, e=e, eta=eta, ok=ok,
                                 score=(n * 100 + i + 1) if ok else 9999,
                                 etag=eta if geom else 99, etroit=etroit, prop=None))
    found = any(r.ok for r in rows); key = "score" if found else "etag"; pick = rows[0]
    for r in rows:
        if r[key] < pick[key]:
            pick = r

    def plate_for(pk):
        d0 = pk.b["d0"]; d = pk.b["d"]; n = pk.n; hp = pk.hp; nb = 2 * n
        ab = mn(pk.e1 / (3 * d0), (pk.p1 / (3 * d0) - 0.25) if n > 1 else INF, dv(R.f_ub, R.fu_L), 1)
        k1 = mn(2.8 * pk.e2 / d0 - 1.7, 1.4 * pk.p3 / d0 - 1.7, 2.5)
        fb1 = dv(k1 * ab * R.fu_L * d, gM2) / 1000 * R.k_trou          # Fb,Rd par mm d'épaisseur
        kb = 1.0 if (hp >= 1.36 * pk.p3 or n == 1) else 0.5
        reqs = [["Pression diamétrale de la platine", dv(V, nb * fb1)],
                ["Cisaillement section brute", dv(V * 1000 * K_CIS * S3 * gM0, 2 * hp * R.fy_L)],
                ["Cisaillement section nette", dv(V * 1000 * S3 * gM2n, 2 * (hp - n * d0) * R.fu_L)],
                ["Rupture de bloc", dv(V * 1000, 2 * (dv(kb * R.fu_L * (pk.e2 - 0.5 * d0), gM2n)
                                                      + dv(R.fy_L * (hp - pk.e1 - (n - 0.5) * d0) / S3, gM0)))]]
        treq = dv(mx(*[r[1] for r in reqs]), u.eta_c)
        tmax = MEC.tp_ductile(d, R.f_ub, R.fy_L)
        cand = [t for t in DB["epais"] if 8 <= t <= tmax + 1e-9] or [DB["epais"][0]]
        tp = cand[_first_ge(cand, treq)]
        bp = pk.p3 + 2 * pk.e2
        return AttrDict(reqs=reqs, treq=treq, tp=tp, tmax=tmax, bp=bp, a=a_s)

    for r in rows:
        if r.geom:
            r.prop = plate_for(r)
    if not pick.prop:
        pick.prop = plate_for(pick)
    pp = pick.prop
    if pick.etroit:
        msg = ("AUCUN entraxe des files ne tient : le gauge minimal (rondelle, congé du poteau) dépasse la "
               "largeur disponible " + ("de l'aile" if not ame else "entre les congés de l'âme")
               + ". Changer de poteau, de diamètre, ou passer à l'autre cas d'appui.")
    elif not found:
        msg = ("AUCUNE combinaison ne respecte le taux cible dans les limites fixées : la moins "
               "sollicitée est affichée. Élargir la plage de diamètres, la hauteur disponible ou la classe.")
    elif pp.treq > pp.tmax:
        msg = ("Épaisseur nécessaire au-delà de la limite de ductilité : augmenter le nombre de rangées, "
               "le diamètre ou la nuance de la platine.")
    else:
        msg = "Proposition complète"
    return AttrDict(rows=rows, hav=hav, found=found, pick=pick, boulon=pick.b["n"], n=pick.n,
                    e1=pick.e1, p1=pick.p1, e2=pick.e2, p3=pick.p3, hp=pick.hp, eta=pick.eta,
                    reqs=pp.reqs, treq=pp.treq, tp=pp.tp, bp=pp.bp, a=pp.a, msg=msg)


# =========================================================================
#  apply_solution / solution_label
# =========================================================================
def apply_solution(u0, r):
    """Recopie une solution du prédimensionnement dans les saisies et
    repasse en mode VÉRIFICATION."""
    u = dict(u0)
    p = r.prop
    u["mode_calc"] = "VÉRIFICATION"; u["boulon_u"] = r.b["n"]; u["n1_u"] = r.n
    u["p1_u"] = r.p1; u["e1_u"] = r.e1; u["p3_u"] = r.p3; u["hp_u"] = r.hp
    u["tp_u"] = p.tp; u["bp_u"] = p.bp; u["a_w"] = p.a
    return u


def solution_label(r):
    """« M20, 4 rangées, platine 290 × 220 × 10 »."""
    return (r.b["n"] + ", " + js_str(r.n) + " rangées, platine " + js_str(r.hp) + " × "
            + js_str(r.prop.bp) + " × " + js_str(r.prop.tp))

# -*- coding: utf-8 -*-
"""Moteur de calcul — PIED DE POTEAU sur béton sous N + Mx + My (+ Vx, Vy, T).

Méthode (docs/assemblages/PIED_POTEAU_ETUDE.md, docs/assemblages/PIED_POTEAU.md) :

1. AIRE EFFICACE d'EN 1993-1-8 §6.2.5 : bande c = tp·√(fy/(3·fjd·γM0)) autour
   des parois du profilé, découpée par la platine ; fjd par EN 1992-1-1 §6.7.
2. RÉPARTITION — modèle élastique : platine rigide, lit sans traction sur
   l'aire efficace, tiges tendues seulement, raideurs k13 / k16 d'EN 1993-1-8
   §6.3.4 → pressions, résultante, effort de chaque tige. C'est la dérivation
   des efforts qu'EN 1992-4 (section 6) impose pour les fixations.
3. RÉSISTANCE — niveau 1 (verdict) : capacités uniaxiales au modèle de la
   Table 6.7 (§6.2.8.3), interaction linéaire Mx/Mx,Rd + My/My,Rd ≤ 1,
   conservative parce que le domaine résistant est convexe (théorème
   statique) ; niveau 2 (information) : domaine exact par balayage de l'axe
   neutre — bloc fjd, tiges du côté tendu à leur résistance (champ extrémal,
   frontière convexe).
4. COMPOSANTS : platine comprimée (bande c, contrôle élastique du débord),
   platine tendue (tronçons en T), tiges acier (EC3 et EN 1992-4), tiges
   béton (EN 1992-4 : cône, arrachement ou adhérence, bord, pry-out,
   interaction), aile du poteau, cordons (méthode directionnelle), rigidité.

Natures : EC (Eurocode direct), EC + INT (Eurocode et interprétation nommée),
INT (modèle d'ingénieur — niveau 2, tiges de coin, tiges latérales).

Unités : saisies en mm, kN, kNm ; calcul en N, mm, N·mm ; résultats en kN,
kNm, MPa, kNm/m.
"""
import math

from acier.bibliotheques import DB, PERSO, find
from acier.js import AttrDict, N, dv
from acier.formats import F
from acier.resultats import Verification, Alerte, Pince
from . import composants as CP
from . import equilibre as EQ
from . import geometrie as GE
from .entrees import defaults, NUM, FORME_RHS, TYPE_CROSSE, OUI, fck_de, fy_epaisseur

GROUPES_VERIF = ("Pied", "Béton", "Platine", "Tiges", "Poteau", "Soudure")
DETAIL_SEUL = frozenset({"wa"})          # minimum de détail : jamais dimensionnant s'il passe
INFO_SEUL = frozenset({"fc2", "bet"})    # information : ni dimensionnant, ni dans le verdict
REF_65 = "EN 1993-1-8 §6.2.5 ; EN 1992-1-1 §6.7"
REF_67 = "EN 1993-1-8 §6.2.8.3 (modèle de la Table 6.7) + interaction linéaire (convexité)"
REF_T62 = "EN 1993-1-8 §6.2.4, Tableau 6.2 ; longueurs efficaces dérivées des Tableaux 6.4 / 6.6"
REF_924 = "EN 1992-4"
PI = math.pi
E = CP.E_ACIER


def _sens(v):
    return 1.0 if v >= 0 else -1.0


def compute(u0=None):
    """Calcul complet pour les saisies ``u0`` (clés absentes → défauts).
    Retourne ``R`` : scalaires, ``u``, ``checks`` / ``ck``, ``alerts``,
    ``dist``, ``tiges`` (efforts par tige), ``contact`` (polygones),
    ``dom`` (frontière du domaine), ``gov``, ``eta_max``, ``verified``."""
    u = AttrDict(defaults())
    for k, v in (u0 or {}).items():
        u[k] = v
    for n in NUM:
        u[n] = N(u[n])
    R = AttrDict()
    R.u = u
    R.checks, R.ck, R.alerts, R.dist = [], {}, [], []
    R.pred = False
    gM0, gM2 = u.g_M0 or 1.0, u.g_M2 or 1.25
    gc, gMc = u.g_c or 1.5, u.g_Mc or 1.5

    def al(ident, msg, block, fields=(), why=""):
        R.alerts.append(Alerte(ident, msg, block, list(fields), [], [], why))

    def ck(key, grp, label, Ed, Rd, unit, ref, nat, active, formula, vals="", ess=0):
        if active and Rd is not None and Rd > 0:
            eta = Ed / Rd
        elif active:
            eta = 1e9 if Ed > 0 else 0.0
        else:
            eta = None
        v = Verification(key, grp, label, Ed, Rd, unit, ref, nat, bool(active), formula, vals, bool(ess),
                         eta, (eta <= 1 + 1e-9) if eta is not None else None)
        R.checks.append(v)
        R.ck[key] = v
        return v

    # ------------------------------------------------------------ profilé
    R.forme = "RHS" if u.forme == FORME_RHS else "I"
    if R.forme == "I":
        if u.prof_c == PERSO:
            dims = dict(h=u.h_u or 0, b=u.b_u or 0, tw=u.tw_u or 0, tf=u.tf_u or 0, r=u.r_u or 0)
        else:
            p = find(DB["profils"], u.prof_c)
            dims = dict(h=p["h"], b=p["b"], tw=p["tw"], tf=p["tf"], r=p["r"])
        t_paroi = dims["tf"]
    else:
        dims = dict(h=u.h_u or 0, b=u.b_u or 0, t=u.t_u or 0)
        t_paroi = dims["t"]
    R.dims = dims
    R.h, R.b = dims["h"], dims["b"]
    geo_ok = all(v > 0 for k, v in dims.items() if k != "r")
    if not geo_ok:
        al("profil", "Dimensions du profilé incomplètes.", True, ["prof_c", "h_u", "b_u", "t_u"])

    # ------------------------------------------------------------- aciers
    def acier(nu):
        if nu == PERSO:
            return dict(fy=u.fy_u or 0, fu=u.fu_u or 0, bw=u.bw_u or 0.9)
        return find(DB["aciers"], nu)
    ac, ap = acier(u.nu_c), acier(u.nu_pl)
    R.fy_c, R.fu_c = fy_epaisseur(ac["fy"], t_paroi), ac["fu"]
    R.fy_p, R.fu_p, R.bw_p = fy_epaisseur(ap["fy"], u.tp_u), ap["fu"], ap["bw"]
    R.fu_w, R.bw_w = min(R.fu_c, R.fu_p), max(ac["bw"], ap["bw"])

    # -------------------------------------------------------------- béton
    R.fck = fck_de(u.beton)
    R.fiss = u.fiss == OUI
    bc = CP.beton_comprime(R.fck, u.a_cc, gc, u.bp_u, u.lp_u, u.bc_u, u.lc_u, u.hc_u, u.t_g, u.f_g)
    R.fcd, R.fjd, R.ratio_c, R.Ac0, R.Ac1, R.beta_j = bc["fcd"], bc["fjd"], bc["ratio"], bc["Ac0"], bc["Ac1"], bc["beta_j"]
    R.E_cm = CP.E_cm(R.fck)
    if not bc["mortier_ok"]:
        al("mortier", "Mortier de scellement hors conditions de βj = 2/3 (épaisseur ≤ 0,2·min(bp, lp) et "
           "résistance ≥ 0,2·fck) — EN 1993-1-8 §6.2.5(7).", True, ["t_g", "f_g"],
           f"tg = {F(u.t_g, 0)} mm, fg = {F(u.f_g, 0)} MPa ; limites {F(0.2 * min(u.bp_u, u.lp_u), 0)} mm et "
           f"{F(0.2 * R.fck, 1)} MPa")
    if u.bp_u > u.bc_u or u.lp_u > u.lc_u:
        al("massif", "La platine déborde du massif.", True, ["bp_u", "lp_u", "bc_u", "lc_u"])
        geo_ok = False

    # ------------------------------------------------- aire efficace (EC3)
    R.c = CP.c_bande(u.tp_u, R.fy_p, R.fjd, gM0)
    plaque = (-u.bp_u / 2, -u.lp_u / 2, u.bp_u / 2, u.lp_u / 2)
    parois = GE.parois_I(dims["h"], dims["b"], dims["tw"], dims["tf"]) if R.forme == "I" \
        else GE.parois_RHS(dims["h"], dims["b"], dims["t"])
    R.parois = parois
    R.A_eff_rects = GE.aire_efficace(parois, R.c, plaque) if geo_ok else []
    R.A_eff = GE.aire_rects(R.A_eff_rects)
    R.N_Rd = R.fjd * R.A_eff                                          # N
    R.debord = max((u.lp_u - dims["h"]) / 2, (u.bp_u - dims["b"]) / 2)

    # -------------------------------------------------------------- tiges
    bo = find(DB["boulons"], u.ancr)
    cl = find(DB["classes"], u.classe)
    R.d_a, R.As, R.d0 = bo["d"], bo["As"], bo["d0"]
    R.f_ub, R.f_yb = cl["fub"], cl["fyb"]
    R.n_t = int(u.n_ax) * int(u.n_ay)
    pos = GE.positions_tiges(u.n_ax, u.n_ay, u.w_x, u.w_y)
    R.tiges = [dict(x=x, y=y, i=i, j=j) for (x, y, i, j) in pos]
    xy = [(t["x"], t["y"]) for t in R.tiges]
    if int(u.n_ax) != 2:
        al("files", "Deux files de tiges sont prévues en V1 ; les rangées sont traitées par tiges avec "
           "leurs voisines.", False, ["n_ax"])
    R.Lb = 8 * R.d_a + u.t_g + u.tp_u + 0.6 * R.d_a           # Tableau 6.2 : + rondelle + ½ écrou (0,6·d)
    R.Ft_EC3 = CP.tige_traction_EC3(R.f_ub, R.As, gM2)
    R.N_Rd_s, R.g_Ms_N = CP.tige_traction_EN1992_4(R.As, R.f_ub, R.f_yb)
    R.Ft_anc = min(R.Ft_EC3, R.N_Rd_s)
    if u.typ_a == TYPE_CROSSE:
        ad = CP.adherence_crosse(R.d_a, u.hef, R.fck, gc)
        R.Ft_bond = ad["F"]
        R.fbd = ad["fbd"]
        R.Ft_anc = min(R.Ft_anc, R.Ft_bond)
    else:
        R.Ft_bond = None

    # pinces sur la platine (EN 1993-1-8 Tableau 3.3) et position des tiges
    for t in R.tiges:
        t["e_x"] = u.bp_u / 2 - abs(t["x"])
        t["e_y"] = u.lp_u / 2 - abs(t["y"])
    e_min = min(min(t["e_x"], t["e_y"]) for t in R.tiges)
    R.dist.append(Pince("e_pl", "Pince des tiges au bord de la platine", e_min, 1.2 * R.d0, None, 1.2,
                        ["w_x", "w_y", "bp_u", "lp_u"], [], "e ≥ 1,2·d0", e_min >= 1.2 * R.d0 - 1e-9, "dist_e"))
    if e_min < 1.2 * R.d0 - 1e-9:
        al("dist_e", "Pince d'une tige au bord de la platine inférieure à 1,2·d0 (ou tige hors platine).",
           True, ["w_x", "w_y", "bp_u", "lp_u"], f"e = {F(e_min, 0)} mm < {F(1.2 * R.d0, 1)} mm")
        geo_ok = False
    R.dist_ok = all(p.ok for p in R.dist)
    for t in R.tiges:
        t.update(GE.classer_tige(t["x"], t["y"], R.forme, dims, u.a_w))
    classes = {t["classe"] for t in R.tiges}
    if "dedans" in classes:
        al("dedans", "Une tige tombe dans le vide du profil creux.", True, ["w_x", "w_y"])
        geo_ok = False
    if "coin" in classes:
        al("coin", "Tiges au-delà de l'about d'aile (coin) : pas de longueur efficace codifiée — tronçon en T "
           "contre la face d'aile, longueur plafonnée à la demi-largeur d'aile par tige (modèle d'ingénieur, "
           "à valider).", False, ["w_x", "w_y"])
    if "lat" in classes:
        al("lat", "Tiges au-delà de l'about d'aile entre les niveaux d'aile : appui réduit à l'épaisseur de "
           "l'aile (modèle d'ingénieur, à valider).", False, ["w_x", "w_y"])
    if "int" in classes and R.forme == "I":
        al("int", "Tiges entre les ailes : tronçon en T non raidi contre l'âme (Tableau 6.4, conservatif).",
           False, ["w_x", "w_y"])

    # ------------------------------------------- tronçons en T par rangée
    rangees = {}
    for k, t in enumerate(R.tiges):
        axe, sg = t["ligne"]
        cle = (axe, sg, round(t["y"] if axe == "y" else t["x"], 3), t["classe"] if t["classe"] in ("int", "lat") else "ext")
        rangees.setdefault(cle, []).append(k)
    R.rangees = []
    FtRd = [0.0] * len(R.tiges)
    for cle, idx in rangees.items():
        axe, sg, _coord, cls = cle
        ts = [R.tiges[k] for k in idx]
        along = (lambda t: t["x"]) if axe == "y" else (lambda t: t["y"])
        coords = sorted(along(t) for t in ts)
        lat_demi = u.bp_u / 2 if axe == "y" else u.lp_u / 2
        liste = []
        for t in ts:
            cte = along(t)
            voisins = [abs(cte - c2) for c2 in coords if abs(c2 - cte) > 1e-9]
            liste.append(dict(e_lat=lat_demi - abs(cte), s_min=min(voisins) if voisins else CP.INF))
        m = max(t["m"] for t in ts)
        e_x = (u.lp_u / 2 - abs(ts[0]["y"])) if axe == "y" else (u.bp_u / 2 - abs(ts[0]["x"]))
        cap = None
        classe_r = ts[0]["classe"]
        if classe_r == "coin":
            cap = 0.25 * (dims["b"] if axe == "y" else dims["h"])
        elif classe_r == "lat":
            cap = 0.5 * (dims["tf"] + 4 * m)
        elif classe_r == "int":
            e_x = u.bp_u / 2 - abs(ts[0]["x"])
        ts_r = CP.tstub_rangee(liste, m, e_x, u.tp_u, R.fy_p, gM0, R.Lb, R.As, R.Ft_anc, cap)
        ts_r.update(cle=f"{axe}{'+' if sg > 0 else '-'}:{_coord:.0f}", classe=classe_r, idx=idx,
                    nat="INT" if classe_r in ("coin", "lat") else "EC")
        R.rangees.append(ts_r)
        for k in idx:
            FtRd[k] = ts_r["FT_tige"]
            R.tiges[k]["FT_Rd"] = ts_r["FT_tige"]
            R.tiges[k]["rangee"] = ts_r["cle"]
    R.FtRd = FtRd

    # ---------------------------------------------------- raideurs (§6.3.4)
    band_y = GE.tronçon_comprime(parois, R.c, plaque, (0, 1), R.forme)[1] if geo_ok else 0.0
    R.A_tstub = band_y
    R.k_c = R.E_cm / (1.275 * math.sqrt(band_y)) if band_y > 0 else 0.0     # N/mm³
    levier = any(r["levier"] for r in R.rangees)
    R.k_b = E * (1.6 if levier else 2.0) * R.As / R.Lb                       # N/mm

    # ----------------------------------------------------------- efforts
    N_ = u.N_Ed * 1e3
    Mx, My = u.Mx_Ed * 1e6, u.My_Ed * 1e6
    Vx, Vy, T_ = u.Vx_Ed * 1e3, u.Vy_Ed * 1e3, u.T_Ed * 1e6
    V_res = math.hypot(Vx, Vy)
    M_res = math.hypot(Mx, My)

    # ---------------------------------------------- répartition élastique
    el = EQ.elastique(R.A_eff_rects, xy, R.k_c, R.k_b, N_, Mx, My) if (geo_ok and R.k_c > 0) \
        else dict(ok=False, contact=[], T=[0.0] * len(xy), C=0.0, xC=0.0, yC=0.0, pmax=0.0, A_c=0.0,
                  a=0.0, bx=0.0, by=0.0, iter=0)
    R.eq_ok = bool(el["ok"])
    if geo_ok and not R.eq_ok:
        al("eq", "Équilibre élastique introuvable (soulèvement sans tiges suffisantes ou données incohérentes).",
           True, ["N_Ed", "Mx_Ed", "My_Ed"])
    R.contact, R.A_c = el["contact"], el["A_c"]
    R.C, R.xC, R.yC, R.pmax = el["C"] / 1e3, el["xC"], el["yC"], el["pmax"]
    R.axe = (el["a"], el["bx"], el["by"])
    R.N_c = max(0.0, el["C"]) if R.eq_ok else max(0.0, N_)
    for t, Ti in zip(R.tiges, el["T"]):
        t["T"] = Ti / 1e3
    T_tot = sum(el["T"])
    R.T_tot = T_tot / 1e3
    R.T_max = max(el["T"]) / 1e3 if el["T"] else 0.0
    R.soulevement = T_tot > 1e-6
    if R.soulevement:
        al("soul", "Côté tendu : " + ", ".join(f"tige ({F(t['x'], 0)} ; {F(t['y'], 0)}) {F(t['T'], 1)} kN"
                                               for t in R.tiges if t["T"] > 1e-6) + ".", False, [])

    # cisaillement par tige : EC3 (frottement déduit) et EN 1992-4 (sans)
    R.Ff_Rd = (u.C_f or 0.0) * max(0.0, N_)
    Sr2 = sum(t["x"] ** 2 + t["y"] ** 2 for t in R.tiges) or 1.0
    ux, uy = (Vx / V_res, Vy / V_res) if V_res > 0 else (0.0, 0.0)
    for t in R.tiges:
        tx, ty = -t["y"] * T_ / Sr2, t["x"] * T_ / Sr2
        v_ec3 = max(0.0, V_res - R.Ff_Rd) / R.n_t
        v_en = V_res / R.n_t
        t["V_ec3"] = math.hypot(v_ec3 * ux + tx, v_ec3 * uy + ty) / 1e3
        t["V_en"] = math.hypot(v_en * ux + tx, v_en * uy + ty) / 1e3
    R.V_t_ec3 = max(t["V_ec3"] for t in R.tiges)
    R.V_t_en = max(t["V_en"] for t in R.tiges)

    # ------------------------------------------------- niveau 1 et niveau 2
    R.MRd1, R.MRd2, R.branche1, R.FC_Rd, R.zC = {}, {}, {}, {}, {}
    for nom, sens in (("x+", (0, 1)), ("x-", (0, -1)), ("y+", (1, 0)), ("y-", (-1, 0))):
        if not geo_ok:
            R.MRd1[nom] = R.MRd2[nom] = 0.0
            R.branche1[nom] = "—"
            continue
        _rects, A_t, bras = GE.tronçon_comprime(parois, R.c, plaque, sens, R.forme)
        R.FC_Rd[nom], R.zC[nom] = R.fjd * A_t, bras
        M1, br = EQ.capacite_niveau1(sens, bras, R.fjd * A_t, xy, FtRd, N_)
        R.MRd1[nom], R.branche1[nom] = M1 / 1e6, br
        phi = {"x+": 0.0, "y+": PI / 2, "x-": PI, "y-": 3 * PI / 2}[nom]
        p2 = EQ.capacite_direction(R.A_eff_rects, xy, FtRd, N_, R.fjd, phi)
        R.MRd2[nom] = (math.hypot(p2["Mx"], p2["My"]) / 1e6) if p2 else 0.0
    kx, ky = ("x+" if Mx >= 0 else "x-"), ("y+" if My >= 0 else "y-")
    R.sum_Ft = sum(FtRd) / 1e3
    if M_res > 1e-6:
        eta1 = (abs(u.Mx_Ed) / R.MRd1[kx] if abs(u.Mx_Ed) > 1e-9 else 0.0) if R.MRd1[kx] > 0 or abs(u.Mx_Ed) < 1e-9 else 1e9
        eta1 += (abs(u.My_Ed) / R.MRd1[ky] if abs(u.My_Ed) > 1e-9 else 0.0) if R.MRd1[ky] > 0 or abs(u.My_Ed) < 1e-9 else 1e9
        p2 = EQ.capacite_direction(R.A_eff_rects, xy, FtRd, N_, R.fjd, math.atan2(My, Mx)) if geo_ok else None
        R.M_R2 = (math.hypot(p2["Mx"], p2["My"]) / 1e6) if p2 else 0.0
        eta2 = (M_res / 1e6) / R.M_R2 if R.M_R2 > 0 else 1e9
        R.dom = [(p["Mx"] / 1e6, p["My"] / 1e6) for p in EQ.domaine(R.A_eff_rects, xy, FtRd, N_, R.fjd, 120)] if geo_ok else []
    else:
        R.M_R2, R.dom = 0.0, []
        if N_ >= 0:
            eta1 = eta2 = (N_ / R.N_Rd) if R.N_Rd > 0 else 1e9
        else:
            eta1 = eta2 = (-N_ / (sum(FtRd) or 1e-9))
    if N_ > 2 * min(R.FC_Rd.values() or [0.0]) and M_res > 1e-6:
        al("comp_dom", "Compression supérieure à 2·FC,Rd d'une paroi : le modèle de la Table 6.7 (niveau 1) "
           "ne s'applique plus, le niveau 2 renseigne.", False, ["N_Ed"])
    R.eta1, R.eta2 = eta1, eta2

    # --------------------------------------------------------- vérifications
    M_tot = M_res / 1e6
    ck("fc1", "Pied", "Flexion composée N + Mx + My — niveau 1 : interaction linéaire des capacités "
       "uniaxiales (modèle Table 6.7)", eta1, 1.0, "-", REF_67, "EC + INT", True,
       "|Mx|/Mx,Rd(N) + |My|/My,Rd(N) ≤ 1 ; Mx,Rd, My,Rd : résultante comprimée au centre de la bande de la "
       "paroi comprimée, tiges opposées réparties linéairement, plafonnées ; N seul : N ≤ fjd·Aeff",
       f"Mx,Rd = {F(R.MRd1[kx], 1)} kNm ({R.branche1[kx]}) ; My,Rd = {F(R.MRd1[ky], 1)} kNm "
       f"({R.branche1[ky]}) ; zC = {F(R.zC.get(kx, 0), 0)} / {F(R.zC.get(ky, 0), 0)} mm ; "
       f"FC,Rd = {F(R.FC_Rd.get(kx, 0) / 1e3, 0)} / {F(R.FC_Rd.get(ky, 0) / 1e3, 0)} kN", 1)
    ck("fc2", "Pied", "Flexion composée — niveau 2 : domaine exact (bloc fjd, tiges tendues à leur résistance), information",
       M_tot, R.M_R2 if M_res > 1e-6 else None, "kNm", "Théorème statique (borne inférieure) sur les composants "
       "d'EN 1993-1-8 §6.2.5 et §6.2.4 — modèle d'ingénieur, non codifié", "INT", M_res > 1e-6,
       "|M_Ed| ≤ |M_Rd(φ)| sur la frontière du domaine (Mx, My) à N donné",
       f"Mx,Rd exact = {F(R.MRd2[kx], 1)} kNm ; My,Rd exact = {F(R.MRd2[ky], 1)} kNm ; "
       f"direction φ = {F(math.degrees(math.atan2(My, Mx)), 1)}°")
    ck("ncp", "Béton", "Compression — aire efficace : N ≤ fjd·Aeff", max(0.0, u.N_Ed), R.N_Rd / 1e3, "kN",
       "EN 1993-1-8 §6.2.8.2 et §6.2.5", "EC", N_ > 0,
       "NRd = fjd·Aeff ; Aeff = parois élargies de c, découpées par la platine ; fjd = βj·fcd·√(Ac1/Ac0)",
       f"fcd = {F(R.fcd, 1)} MPa ; √(Ac1/Ac0) = {F(R.ratio_c, 2)} ; fjd = {F(R.fjd, 1)} MPa ; c = {F(R.c, 1)} mm ; "
       f"Aeff = {F(R.A_eff / 100, 0)} cm²", 1)
    ck("bet", "Béton", "Pression maximale de la répartition élastique ≤ fjd (information)", R.pmax, R.fjd, "MPa",
       REF_65 + " — répartition élastique (EN 1992-4 section 6)", "INT", R.eq_ok,
       "pmax = kc·wmax sur la zone de contact ; kc = E·k13/(beff·leff)",
       f"kc = {F(R.k_c, 0)} N/mm³ ; zone de contact {F(R.A_c / 100, 0)} cm² sur {F(R.A_eff / 100, 0)} ; "
       f"C = {F(R.C, 1)} kN en ({F(R.xC, 0)} ; {F(R.yC, 0)}) mm")

    # platine comprimée : moment élastique du débord sous la répartition réelle
    m_el = u.tp_u ** 2 * R.fy_p / (6.0 * gM0)                                          # N·mm/mm
    m_Ed, ligne_m = 0.0, "—"
    if R.eq_ok:
        lignes = [("y = +h/2", 0, 1, dims["h"] / 2), ("y = −h/2", 0, -1, dims["h"] / 2),
                  ("x = +b/2", 1, 0, dims["b"] / 2), ("x = −b/2", -1, 0, dims["b"] / 2)]
        a_, bx_, by_ = R.axe
        for nom, nx, ny, dist in lignes:
            mom, long = 0.0, 0.0
            for poly in R.contact:
                part = GE.clip(poly, nx, ny, dist)
                if len(part) < 3:
                    continue
                A, sx, sy, ixx, ixy, iyy = GE.moments(part)
                # ∫ p·(n·r − dist) dA avec p = kc·(a + bx·x + by·y)
                Fn = a_ * A + bx_ * sx + by_ * sy
                Mn = a_ * (nx * sx + ny * sy) + bx_ * (nx * ixx + ny * ixy) + by_ * (nx * ixy + ny * iyy)
                mom += R.k_c * (Mn - dist * Fn)
                xs = [p[0] for p in part]; ys = [p[1] for p in part]
                # longueur d'appui = somme des étendues des parties (deux bandes d'aile au-delà d'un about)
                long += (max(xs) - min(xs)) if ny else (max(ys) - min(ys))
            if long > 0 and mom / long > m_Ed:
                m_Ed, ligne_m = mom / long, nom
    ck("pc", "Platine", "Platine comprimée — flexion élastique du débord sous la répartition", m_Ed / 1e3, m_el / 1e3,
       "kNm/m", REF_65 + " (la bande c est élastique par construction)", "EC + INT", R.eq_ok,
       "mEd = ∫p·s dA / l ≤ tp²·fy/(6·γM0) sur le débord au-delà de la paroi comprimée",
       f"ligne {ligne_m} ; tp = {F(u.tp_u, 0)} mm ; fy = {F(R.fy_p, 0)} MPa")

    # platine tendue : tronçon en T, tige la plus sollicitée
    pire = None
    for t in R.tiges:
        if t["T"] > 1e-9 and (pire is None or t["T"] / max(t["FT_Rd"], 1e-9) > pire[0]):
            pire = (t["T"] / max(t["FT_Rd"], 1e-9), t)
    rg = next((r for r in R.rangees if pire and r["cle"] == pire[1]["rangee"]), R.rangees[0] if R.rangees else None)
    nat_pt = "EC" if not rg or rg["nat"] == "EC" else "INT"
    ck("pt", "Platine", "Platine tendue — tronçon en T de la rangée de la tige la plus sollicitée",
       pire[1]["T"] if pire else 0.0, (rg["FT_tige"] / 1e3) if rg else None, "kN",
       REF_T62 + (" ; tiges de coin / latérales : longueur plafonnée (modèle d'ingénieur)" if nat_pt == "INT" else ""),
       nat_pt, R.soulevement and rg is not None,
       "FT,Rd = min(mode 1 ; mode 2 ; mode 3)/n par tige ; mode 1 sans effet de levier si Lb > Lb*",
       (f"rangée {rg['cle']} ({rg['classe']}) : m = {F(rg['m'], 1)} mm, Σleff,1 = {F(rg['leff1'], 0)} mm, "
        f"Σleff,2 = {F(rg['leff2'], 0)} mm, Lb = {F(rg['Lb'], 0)} / Lb* = {F(min(rg['Lb_star'], 9999), 0)} mm, "
        f"mode {rg['mode']} ; FT1 = {F(rg['FT1'] / 1e3, 0)}, FT2 = "
        f"{('∞' if rg['FT2'] == CP.INF else F(rg['FT2'] / 1e3, 0))}, FT3 = {F(rg['FT3'] / 1e3, 0)} kN") if rg else "", 1)

    # tiges : acier
    ck("ta", "Tiges", "Traction d'une tige (acier) : min(EN 1993-1-8 Tableau 3.4 ; EN 1992-4 §7.2.1.3"
       + (" ; adhérence de la crosse" if R.Ft_bond is not None else "") + ")", R.T_max,
       R.Ft_anc / 1e3, "kN", "EN 1993-1-8 Tableau 3.4 et §6.2.6.11 ; EN 1992-4 §7.2.1.3", "EC", R.soulevement,
       "Ft,Rd = 0,9·fub·As/γM2 ; NRd,s = As·fuk/γMs,N (γMs,N = 1,2·fuk/fyk ≥ 1,4)"
       + (" ; Ft,bond,Rd = π·d·hef·fbd" if R.Ft_bond is not None else ""),
       f"Ft,Rd = {F(R.Ft_EC3 / 1e3, 1)} kN ; NRd,s = {F(R.N_Rd_s / 1e3, 1)} kN (γMs,N = {F(R.g_Ms_N, 2)})"
       + (f" ; Ft,bond,Rd = {F(R.Ft_bond / 1e3, 1)} kN (fbd = {F(R.fbd, 2)} MPa)" if R.Ft_bond is not None else ""), 1)
    R.F2vb, R.alpha_b = CP.tige_cisaillement_EC3(R.f_yb, R.f_ub, R.As, gM2)
    v_en = CP.tige_cisaillement_EN1992_4(R.As, R.f_ub, R.f_yb, R.d_a, u.t_g, u.f_g, R.T_max * 1e3, R.N_Rd_s)
    R.VRd_s, R.VRd_s0, R.VRd_sM, R.levier_v, R.g_Ms_V = v_en
    eta_tv = max(R.V_t_ec3 / (R.F2vb / 1e3) if R.F2vb > 0 else 1e9, R.V_t_en / (R.VRd_s / 1e3) if R.VRd_s > 0 else 1e9)
    ck("tv", "Tiges", "Cisaillement d'une tige (acier) : EN 1993-1-8 §6.2.2 (frottement déduit) et EN 1992-4 "
       "§7.2.2.3 (sans frottement" + (", avec bras de levier" if R.levier_v else "") + ")",
       eta_tv, 1.0, "-", "EN 1993-1-8 §6.2.2 ; EN 1992-4 §7.2.2.3", "EC", V_res > 0 or abs(T_) > 0,
       "max(Vt,Ed/F2,vb,Rd ; Vt,Ed/VRd,s) ; F2,vb,Rd = αb·fub·As/γM2, αb = 0,44 − 0,0003·fyb ; VRd,s = k6·As·fuk/γMs,V"
       + (" ; VRk,s,M = αM·MRk,s/l, l = d/2 + tg" if R.levier_v else "") + " ; Ff,Rd = Cf,d·Nc,Ed",
       f"Vt,Ed = {F(R.V_t_ec3, 1)} kN (EC3) / {F(R.V_t_en, 1)} kN (EN 1992-4) ; Ff,Rd = {F(R.Ff_Rd / 1e3, 1)} kN ; "
       f"F2,vb,Rd = {F(R.F2vb / 1e3, 1)} kN (αb = {F(R.alpha_b, 3)}) ; VRd,s = {F(R.VRd_s / 1e3, 1)} kN"
       + (f" (sans bras {F(R.VRd_s0 / 1e3, 1)}, avec bras {F(R.VRd_sM / 1e3, 1)})" if R.levier_v else ""), 1)
    eta_int = 0.0
    if R.soulevement and (V_res > 0 or abs(T_) > 0):
        tmax = max(R.tiges, key=lambda t: t["T"] / max(R.Ft_anc, 1) + t["V_ec3"] / max(R.F2vb, 1))
        i_ec3 = tmax["V_ec3"] / (R.F2vb / 1e3) + tmax["T"] / (1.4 * R.Ft_EC3 / 1e3)
        i_en = (tmax["T"] / (R.N_Rd_s / 1e3)) ** 2 + (tmax["V_en"] / (R.VRd_s / 1e3)) ** 2
        eta_int = max(i_ec3, i_en)
    ck("tvn", "Tiges", "Interaction traction / cisaillement d'une tige (acier)", eta_int, 1.0, "-",
       "EN 1993-1-8 Tableau 3.4 ; EN 1992-4 §7.2.3.1", "EC", R.soulevement and (V_res > 0 or abs(T_) > 0),
       "Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1 ; (NEd/NRd,s)² + (VEd/VRd,s)² ≤ 1", "")

    # tiges : béton (EN 1992-4)
    tend = [t for t in R.tiges if t["T"] > 1e-9]
    if tend:
        Tt = sum(t["T"] for t in tend)
        gx = sum(t["x"] for t in tend) / len(tend); gy = sum(t["y"] for t in tend) / len(tend)
        ex = sum(t["T"] * t["x"] for t in tend) / Tt - gx
        ey = sum(t["T"] * t["y"] for t in tend) / Tt - gy
    else:
        ex = ey = 0.0
    cone = CP.cone_beton(u.hef, R.fck, R.fiss, [t["x"] for t in tend], [t["y"] for t in tend], u.bc_u, u.lc_u,
                         ex, ey, gMc)
    R.cone = cone
    ck("cone", "Béton", "Cône de béton du groupe tendu", R.T_tot, cone["NRd"] / 1e3 if cone["actif"] else None, "kN",
       REF_924 + " §7.2.1.4", "EC", bool(tend),
       "NRk,c = N⁰Rk,c·(Ac,N/A⁰c,N)·ψs,N·ψre,N·ψec,N·ψM,N ; N⁰Rk,c = k1·√fck·hef^1,5 ; NRd,c = NRk,c/γMc",
       (f"k1 = {F(cone['k1'], 1)} ({'fissuré' if R.fiss else 'non fissuré'}) ; hef = {F(u.hef, 0)} mm ; "
        f"Ac,N/A⁰c,N = {F(cone['Ac'] / cone['Ac0'], 2)} ; ψs = {F(cone['psi_s'], 2)} ; ψre = {F(cone['psi_re'], 2)} ; "
        f"ψec = {F(cone['psi_ec'], 2)} ; ψM = 1 (conservatif) ; NRk,c = {F(cone['NRk'] / 1e3, 1)} kN") if cone["actif"] else "", 1)
    if u.typ_a == TYPE_CROSSE:
        ck("pull", "Béton", "Adhérence de la tige à crosse", R.T_max, R.Ft_bond / 1e3, "kN",
           "EN 1993-1-8 §6.2.6.12 ; EN 1992-1-1 §8.4.2", "EC", bool(tend),
           "Ft,bond,Rd = π·d·hef·fbd ; fbd = 2,25·η1·η2·fctd (η1 = 0,7)", f"fbd = {F(R.fbd, 2)} MPa")
        R.pull = None
    else:
        pl = CP.arrachement_tete(R.d_a, u.d_h, R.fck, R.fiss, gMc)
        R.pull = pl
        ck("pull", "Béton", "Arrachement de la tête (pull-out)", R.T_max, pl["NRd"] / 1e3, "kN",
           REF_924 + " §7.2.1.5", "EC", bool(tend),
           "NRk,p = k2·Ah·fck ; Ah = π(dh² − d²)/4 ; NRd,p = NRk,p/γMc",
           f"k2 = {F(pl['k2'], 1)} ; Ah = {F(pl['Ah'], 0)} mm² ; NRk,p = {F(pl['NRk'] / 1e3, 1)} kN")
    # bord : le bord le plus défavorable pour la résultante de cisaillement
    xs, ys = sorted({t["x"] for t in R.tiges}), sorted({t["y"] for t in R.tiges})
    cands = []
    for c1, lat_a, lat_b, sp in (
            (u.bc_u / 2 - xs[-1], u.lc_u / 2 + ys[0], u.lc_u / 2 - ys[-1], [b - a for a, b in zip(ys, ys[1:])]),
            (u.bc_u / 2 + xs[0], u.lc_u / 2 + ys[0], u.lc_u / 2 - ys[-1], [b - a for a, b in zip(ys, ys[1:])]),
            (u.lc_u / 2 - ys[-1], u.bc_u / 2 + xs[0], u.bc_u / 2 - xs[-1], [b - a for a, b in zip(xs, xs[1:])]),
            (u.lc_u / 2 + ys[0], u.bc_u / 2 + xs[0], u.bc_u / 2 - xs[-1], [b - a for a, b in zip(xs, xs[1:])])):
        cands.append(CP.bord_beton(R.d_a, u.hef, R.fck, R.fiss, c1, lat_a, lat_b, sp, u.hc_u, gMc))
    bd = min(cands, key=lambda c: c["VRd"])
    R.bord = bd
    ck("edge", "Béton", "Rupture du bord du massif en cisaillement (bord le plus proche, rangée voisine)",
       V_res / 1e3, bd["VRd"] / 1e3, "kN", REF_924 + " §7.2.2.5", "EC", V_res > 0,
       "VRk,c = V⁰Rk,c·(Ac,V/A⁰c,V)·ψs,V·ψh,V ; V⁰Rk,c = k9·d^α·lf^β·√fck·c1^1,5 ; VRd,c = VRk,c/γMc",
       f"c1 = {F(bd.get('c1', 0), 0)} mm ; lf = {F(bd.get('lf', 0), 0)} mm ; Ac,V/A⁰c,V = "
       f"{F(dv(bd.get('Ac', 0), bd.get('Ac0', 1)), 2)} ; ψs = {F(bd.get('psi_s', 1), 2)} ; ψh = {F(bd.get('psi_h', 1), 2)} ; "
       f"VRk,c = {F(bd['VRk'] / 1e3, 1)} kN")
    cone_all = CP.cone_beton(u.hef, R.fck, R.fiss, [t["x"] for t in R.tiges], [t["y"] for t in R.tiges],
                             u.bc_u, u.lc_u, 0.0, 0.0, gMc)
    po = CP.pryout(cone_all["NRk"], u.hef, gMc)
    R.pryout = po
    ck("pry", "Béton", "Pry-out du groupe", V_res / 1e3, po["VRd"] / 1e3, "kN", REF_924 + " §7.2.2.4", "EC",
       V_res > 0, "VRk,cp = k8·NRk,c (groupe entier, sans excentrement) ; VRd,cp = VRk,cp/γMc",
       f"k8 = {F(po['k8'], 0)} ; NRk,c (groupe) = {F(cone_all['NRk'] / 1e3, 1)} kN")
    bN = max((R.ck["cone"].eta or 0) if R.ck["cone"].active else 0, (R.ck["pull"].eta or 0) if R.ck["pull"].active else 0)
    bV = max((R.ck["edge"].eta or 0) if R.ck["edge"].active else 0, (R.ck["pry"].eta or 0) if R.ck["pry"].active else 0)
    ck("cv", "Béton", "Interaction traction / cisaillement du béton", (min(bN, 1e3) ** 1.5 + min(bV, 1e3) ** 1.5),
       1.0, "-", REF_924 + " §7.2.3.2", "EC", bool(tend) and V_res > 0,
       "βN^1,5 + βV^1,5 ≤ 1 ; βN = max(cône ; arrachement), βV = max(bord ; pry-out)",
       f"βN = {F(bN, 2)} ; βV = {F(bV, 2)}")
    if min(cone["cmin"] if cone["actif"] else cone_all["cmin"], cone_all["cmin"]) < 1.5 * u.hef:
        al("fend", "Bord du massif à moins de 1,5·hef : fendage (EN 1992-4 §7.2.1.7) et, pour une tige à tête, "
           "éclatement latéral (§7.2.1.8) à vérifier — non couverts en V1.", False, ["bc_u", "lc_u", "hef"])

    # poteau : aile / paroi comprimée
    Iy, Wpl = CP.inertie_wpl(R.forme, dims)
    R.I_c, R.Wpl = Iy, Wpl
    h_int = dims["h"] - (dims["tf"] if R.forme == "I" else dims["t"])
    R.Fc_fc_Rd = Wpl * R.fy_c / gM0 / h_int
    F_aile, nom_aile = 0.0, "—"
    if R.eq_ok:
        a_, bx_, by_ = R.axe
        for nom, sens in (("+y", (0, 1)), ("−y", (0, -1)), ("+x", (1, 0)), ("−x", (-1, 0))):
            rects, _A, _z = GE.tronçon_comprime(parois, R.c, plaque, sens, R.forme)
            Fp, _sx, _sy, _polys = EQ._pression(rects, a_, bx_, by_)
            if R.k_c * Fp > F_aile:
                F_aile, nom_aile = R.k_c * Fp, nom
    ck("cfc", "Poteau", "Aile / paroi comprimée du poteau (effort de la bande comprimée)", F_aile / 1e3,
       R.Fc_fc_Rd / 1e3, "kN", "EN 1993-1-8 §6.2.6.7 (Fc,fc,Rd = Mc,Rd/(h − tf)), par analogie", "EC", R.eq_ok,
       "Fc,Ed = ∫p dA sur la bande de la paroi ; Fc,fc,Rd = Wpl·fy/(γM0·(h − tf))",
       f"paroi {nom_aile} ; Wpl = {F(Wpl / 1e3, 0)} cm³ ; fy = {F(R.fy_c, 0)} MPa")

    # cordons
    w = CP.cordons(R.forme, dims, u.a_w, R.fu_w, R.bw_w, gM2, N_, Mx, My, Vx, Vy, T_) if u.a_w > 0 and geo_ok \
        else dict(s_eq=0.0, s_perp=0.0, f_eq=1.0, f_perp=1.0, point="—", A=0.0, L=0.0, sig_n=0.0, tau_par=0.0)
    R.cordons = w
    ck("w", "Soudure", "Cordons poteau / platine — méthode directionnelle, point le plus sollicité", w["s_eq"], w["f_eq"],
       "MPa", "EN 1993-1-8 §4.5.3.2", "EC", u.a_w > 0,
       "√(σ⊥² + 3(τ⊥² + τ∥²)) ≤ fu/(βw·γM2) ; groupe de cordons en ligne (N, Mx, My par σn = N/A + M·r/I, "
       "Vx, Vy par τ∥, torsion par T·r/Ip) ; σ⊥ = τ⊥ = σn/√2",
       f"a = {F(u.a_w, 1)} mm ; L = {F(w['L'], 0)} mm ; point : {w['point']} ; σn = {F(w['sig_n'], 1)} MPa ; "
       f"τ∥ = {F(w['tau_par'], 1)} MPa ; fu/(βw·γM2) = {F(w['f_eq'], 0)} MPa", 1)
    ck("wp", "Soudure", "Cordons — contrainte normale σ⊥ ≤ 0,9·fu/γM2", w["s_perp"], w["f_perp"], "MPa",
       "EN 1993-1-8 §4.5.3.2", "EC", u.a_w > 0, "σ⊥ ≤ 0,9·fu/γM2", "")
    ck("wa", "Soudure", "Gorge minimale", 3, u.a_w, "mm", "EN 1993-1-8 §4.5.2(2)", "EC", True, "a ≥ 3 mm", "")

    # rigidité (information)
    rg_rig = next((r for r in R.rangees if r["cle"].startswith("y-")), R.rangees[0] if R.rangees else None)
    if rg_rig and geo_ok:
        z = R.zC.get("x+", 0.0) + max((abs(R.tiges[k]["y"]) for k in rg_rig["idx"]), default=0.0)
        R.rig = CP.rigidite(R.E_cm, R.A_tstub, rg_rig["leff1"], rg_rig["m"], u.tp_u, rg_rig["n"], R.As, R.Lb,
                            rg_rig["levier"], z, Iy, u.L_c, u.contrev == OUI)
        R.Sj = R.rig["Sj"] / 1e6                                   # kNm/rad
        R.Sj_seuil = R.rig["seuil"] / 1e6
        al("rig", f"Rigidité initiale Sj,ini = {F(R.Sj / 1e3, 0)} MNm/rad ; seuil « rigide » (§5.2.2.5) = "
           f"{F(R.Sj_seuil / 1e3, 0)} MNm/rad → pied {'RIGIDE' if R.rig['rigide'] else 'SEMI-RIGIDE'} pour "
           f"Lc = {F(u.L_c / 1e3, 1)} m.", False, ["L_c", "contrev"])
    else:
        R.rig, R.Sj, R.Sj_seuil = None, 0.0, 0.0

    # ------------------------------------------------------------ synthèse
    R.geo_ok = bool(geo_ok and R.eq_ok)
    act = [c for c in R.checks if c.active]
    gov = None
    for c in act:
        if c.key in INFO_SEUL or (c.key in DETAIL_SEUL and c.ok):
            continue
        if gov is None or c.eta > gov.eta:
            gov = c
    R.gov = gov
    R.eta_max = gov.eta if gov else 0.0
    R.all_ok = all(c.ok for c in act if c.key not in INFO_SEUL)
    R.verified = bool(R.all_ok and R.geo_ok and R.dist_ok)
    R.statut = "PIED DE POTEAU VÉRIFIÉ" if R.verified else "PIED DE POTEAU NON VÉRIFIÉ"
    R.reserve = any(a.id in ("coin", "lat") for a in R.alerts)
    return R

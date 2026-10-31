# -*- coding: utf-8 -*-
"""Moteur du CALCUL MANUEL de la platine — l'épaisseur requise comme sur une
note à la main, pour des boulons posés librement, un ou plusieurs éléments
soudés et des raidisseurs (docs/assemblages/PLATINE_MANUELLE.md).

1. EFFORTS DES TIGES — platine rigide (EN 1992-4 section 6) : lit de
   ressorts sans traction sur l'aire d'appui (bande c d'EN 1993-1-8 §6.2.5,
   ou toute la platine), tiges tendues seulement ; N, Mx, My appliqués au
   point (xN ; yN) — un élément excentré crée son moment N·e tout seul.
2. CÔTÉ TENDU, boulon par boulon : l'appui le plus proche (paroi, raidisseur,
   cercle), le bras m jusqu'à lui (cordon déduit — la diagonale jusqu'à
   l'angle pour un boulon d'angle), le moment T·m, la longueur efficace des
   lignes de rupture (Tableau 6.6, convention par boulon) plafonnée à ce que
   la face offre (L/2 + m ; m dans un angle), le tronçon en T du Tableau 6.2
   avec ou sans effet de levier — ou la contrainte de Von Mises de la bande.
3. CÔTÉ TENDU, face par face : la section au nu de chaque paroi sous les
   boulons qui tirent devant elle (dans sa zone de diffusion à 45°), largeur
   L + 2·m bornée par la platine — le calcul classique, et c'est lui qui
   explose quand l'élément est étroit.
4. CÔTÉ COMPRIMÉ, bande par bande : au nu de chaque face, la bande la plus
   sollicitée en console, chaque paroi reprenant la pression qui lui est la
   plus proche (manuel/console.py) — contrôle élastique d'EN 1993-1-8 §6.2.5.
5. TROUS : pression diamétrale des boulons sur la platine (Tableau 3.4) sous
   le cisaillement par boulon (V partagé, torsion autour du centre du
   groupe, frottement déduit).
6. ÉPAISSEUR REQUISE : pour chaque critère, la plus petite épaisseur qui le
   satisfait — la répartition (bande c, raideurs, efforts des tiges,
   pressions) est RECALCULÉE à chaque épaisseur essayée ; la plus grande des
   quatre est l'épaisseur requise. Les tiges (acier) et l'appui (fjd) sont
   donnés en INFORMATION : le verdict porte sur la platine.

Unités : saisies en mm, kN, kNm ; calcul en N, mm, N·mm ; résultats en kN,
kNm, MPa, kNm/m. Signes du module : N > 0 compression ; Mx > 0 comprime
y > 0 ; My > 0 comprime x > 0.
"""
import math

from acier.bibliotheques import DB, PERSO, find
from acier.js import AttrDict, N
from acier.formats import F
from acier.resultats import Verification, Alerte, Pince
from .. import composants as CP
from .. import equilibre as EQ
from .. import geometrie as GE
from ..entrees import (defaults, NUM, LISTES, CRIT_ELAS, APPUI_PLEIN, TYPE_I, TYPE_RHS, TYPE_CHS, fck_de,
                       fy_epaisseur)
from ..moteur import _epaisseurs_requises, TP_MIN, TP_MAX
from . import elements as EL
from . import console as CO
from .conversion import disposition_reguliere

GROUPES_VERIF = ("Platine", "Trous", "Tiges", "Appui")
INFO_SEUL = frozenset({"ta", "tv", "tvn", "bet", "ncp"})      # information : hors verdict
CRITERES_TP = ("T", "S", "c", "b")
REF_T62 = "EN 1993-1-8 §6.2.4, Tableau 6.2 ; longueurs efficaces dérivées du Tableau 6.6, par boulon"
REF_SEC = "Section au nu de la paroi, diffusion à 45° (calcul classique) ; EN 1993-1-8 §6.2.4 (Mpl)"
REF_65 = "EN 1993-1-8 §6.2.5 ; EN 1992-1-1 §6.7"
REF_34 = "EN 1993-1-8 Tableau 3.4 (pression diamétrale)"
E = CP.E_ACIER


def _grille_grossiere():
    """La grille du balayage de l'épaisseur : tous les 2 mm jusqu'à 60, puis
    tous les 10 mm — chaque point recalcule toute la répartition."""
    return [TP_MIN + 2.0 * i for i in range(int((60.0 - TP_MIN) / 2.0) + 1)] + \
        [70.0 + 10.0 * i for i in range(int((TP_MAX - 70.0) / 10.0) + 1)]


def _acier(u, nu):
    if nu == PERSO:
        return dict(fy=u.fy_u or 0, fu=u.fu_u or 0, bw=u.bw_u or 0.9)
    return find(DB["aciers"], nu)


def _sigma_vm(M, V, b, t):
    """Contrainte de Von Mises d'une bande de largeur b et d'épaisseur t sous
    le moment M (N·mm) et l'effort tranchant V (N) : σ = 6M/(b·t²), τ = V/(b·t)."""
    if b <= 0 or t <= 0:
        return float("inf")
    s = 6.0 * M / (b * t * t)
    tau = V / (b * t)
    return math.sqrt(s * s + 3.0 * tau * tau)


def compute(u0=None):
    """Calcul manuel complet pour les saisies ``u0`` (clés absentes → défauts).
    Retourne ``R`` (AttrDict) : efforts, boulons, sections, console, trous,
    épaisseur requise, vérifications, alertes."""
    u = AttrDict(defaults())
    for k, v in (u0 or {}).items():
        u[k] = v
    for n in NUM:
        u[n] = N(u[n])
    for k in LISTES:
        if not isinstance(u[k], list):
            u[k] = []
    R = AttrDict()
    R.u = u
    R.mode = "manuel"
    R.checks, R.ck, R.alerts, R.dist = [], {}, [], []
    R.pred = False
    R.perim_ec3 = True
    R.info_keys = INFO_SEUL
    gM0, gM2, gc = u.g_M0 or 1.0, u.g_M2 or 1.25, u.g_c or 1.5
    elastique = u.crit == CRIT_ELAS
    plein = u.appui == APPUI_PLEIN
    R.elastique, R.plein = elastique, plein

    def al(ident, msg, block, fields=(), why="", dims=(), elems=()):
        R.alerts.append(Alerte(ident, msg, block, list(fields), list(dims), list(elems), why))

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

    # -------------------------------------------------------------- aciers
    ap = _acier(u, u.nu_pl)
    fy_nom = ap["fy"]
    R.fy_p, R.fu_p = fy_epaisseur(fy_nom, u.tp_u), ap["fu"]
    bo = find(DB["boulons"], u.ancr)
    cl = find(DB["classes"], u.classe)
    R.d_a, R.As, R.d0 = bo["d"], bo["As"], bo["d0"]
    R.d_w = bo.get("dw", 2 * bo["d"])
    R.f_ub, R.f_yb = cl["fub"], cl["fyb"]
    R.Ft_EC3 = CP.tige_traction_EC3(R.f_ub, R.As, gM2)
    R.N_Rd_s, R.g_Ms_N = CP.tige_traction_EN1992_4(R.As, R.f_ub, R.f_yb)
    R.Ft_anc = min(R.Ft_EC3, R.N_Rd_s)
    R.F2vb, R.alpha_b = CP.tige_cisaillement_EC3(R.f_yb, R.f_ub, R.As, gM2)
    R.Lb = 8 * R.d_a + u.t_g + u.tp_u + 0.6 * R.d_a
    cord = 0.8 * u.a_w * math.sqrt(2)
    R.cord = cord

    # --------------------------------------------------------------- béton
    R.fck = fck_de(u.beton)
    bc = CP.beton_comprime(R.fck, u.a_cc, gc, u.bp_u, u.lp_u, u.bc_u, u.lc_u, u.hc_u, u.t_g, u.f_g)
    R.fcd, R.fjd, R.ratio_c, R.Ac0, R.Ac1, R.beta_j = bc["fcd"], bc["fjd"], bc["ratio"], bc["Ac0"], bc["Ac1"], bc["beta_j"]
    R.E_cm = CP.E_cm(R.fck)
    if not bc["mortier_ok"]:
        al("mortier", "Mortier de scellement hors conditions de βj = 2/3 (épaisseur ≤ 0,2·min(bp, lp) et "
           "résistance ≥ 0,2·fck) — EN 1993-1-8 §6.2.5(7).", True, ["t_g", "f_g"],
           f"tg = {F(u.t_g, 0)} mm, fg = {F(u.f_g, 0)} MPa", dims=["tg"], elems=["grout"])
    if u.bp_u > u.bc_u or u.lp_u > u.lc_u:
        al("massif", "La platine déborde du massif.", True, ["bp_u", "lp_u", "bc_u", "lc_u"],
           f"platine {F(u.bp_u, 0)} × {F(u.lp_u, 0)} mm, massif {F(u.bc_u, 0)} × {F(u.lc_u, 0)} mm",
           dims=["bp", "lp", "bc", "lc"], elems=["plate", "pedestal"])

    # ------------------------------------------------------------ géométrie
    plaque = (-u.bp_u / 2, -u.lp_u / 2, u.bp_u / 2, u.lp_u / 2)
    R.plaque_rect = plaque
    geo_ok = u.bp_u > 0 and u.lp_u > 0 and u.tp_u > 0
    R.elems = [EL.normaliser(e) for e in u.elems]
    R.raid = [EL.normaliser_raid(r) for r in u.raid]
    R.tiges = [dict(i=i, x=float(N(t.get("x")) or 0.0), y=float(N(t.get("y")) or 0.0)) for i, t in enumerate(u.tiges)
               if isinstance(t, dict)]
    R.n_t = len(R.tiges)
    R.nom_elems = " + ".join(EL.nom(e) for e in R.elems) or "—"
    if not R.elems:
        al("aucun_e", "Aucun élément soudé : la platine n'a pas d'appui.", True, [], "", elems=["col"])
        geo_ok = False
    if not R.tiges:
        al("aucun_b", "Aucun boulon.", True, [], "", elems=["anchors"])
        geo_ok = False
    for i, e in enumerate(R.elems):
        if not EL.geometrie_valide(e):
            al(f"elem_{i}", f"Élément {i + 1} ({EL.nom(e)}) : dimensions incomplètes ou incohérentes.", True, [], "",
               elems=["col"])
            geo_ok = False
            continue
        x0, y0, x1, y1 = EL.enveloppe(e)
        if x0 < plaque[0] - 1e-6 or y0 < plaque[1] - 1e-6 or x1 > plaque[2] + 1e-6 or y1 > plaque[3] + 1e-6:
            al(f"hors_{i}", f"Élément {i + 1} ({EL.nom(e)}) hors de la platine.", True, ["bp_u", "lp_u"],
               f"emprise {F(x0, 0)}…{F(x1, 0)} × {F(y0, 0)}…{F(y1, 0)} mm, platine ±{F(u.bp_u / 2, 0)} × ±{F(u.lp_u / 2, 0)}",
               dims=["bp", "lp"], elems=["col", "plate"])
            geo_ok = False
    for i, r in enumerate(R.raid):
        x0, y0, x1, y1 = EL.rect_raid(r)
        if r["L"] <= 0 or r["t"] <= 0:
            al(f"raid_{i}", f"Raidisseur {i + 1} : longueur ou épaisseur nulle.", True, [], "", elems=["raid"])
            geo_ok = False
        elif x0 < plaque[0] - 1e-6 or y0 < plaque[1] - 1e-6 or x1 > plaque[2] + 1e-6 or y1 > plaque[3] + 1e-6:
            al(f"raidh_{i}", f"Raidisseur {i + 1} hors de la platine.", True, ["bp_u", "lp_u"], "", dims=["bp", "lp"],
               elems=["raid", "plate"])
            geo_ok = False
    # parois (rectangles) de tout ce qui est soudé ; appuis (rectangles et cercles) pour les bandes
    R.parois = []
    for i, e in enumerate(R.elems):
        for (n_, x0, y0, x1, y1) in EL.parois(e):
            R.parois.append((f"élément {i + 1} {n_}", x0, y0, x1, y1))
    for i, r in enumerate(R.raid):
        x0, y0, x1, y1 = EL.rect_raid(r)
        R.parois.append((f"raidisseur {i + 1}", x0, y0, x1, y1))
    appuis_geo = [(x0, y0, x1, y1) for (_n, x0, y0, x1, y1) in R.parois]
    for e in R.elems:
        if EL.est_rond(e) and EL.geometrie_valide(e):
            appuis_geo.append(("c", e["x"], e["y"], EL.rayons(e)[0]))
    R.obstacles = appuis_geo

    # boulons : pinces (Tableau 3.3), entraxes, position par rapport aux éléments
    for t in R.tiges:
        t["e_x"] = u.bp_u / 2 - abs(t["x"])
        t["e_y"] = u.lp_u / 2 - abs(t["y"])
    if R.tiges:
        e_min = min(min(t["e_x"], t["e_y"]) for t in R.tiges)
        R.dist.append(Pince("e_pl", "Pince des boulons au bord de la platine", e_min, 1.2 * R.d0, None, 1.2,
                            ["bp_u", "lp_u"], [], "e ≥ 1,2·d0", e_min >= 1.2 * R.d0 - 1e-9, "dist_e"))
        if e_min < 1.2 * R.d0 - 1e-9:
            al("dist_e", "Pince d'un boulon au bord de la platine inférieure à 1,2·d0 (ou boulon hors platine).",
               True, ["bp_u", "lp_u"], f"e = {F(e_min, 0)} mm < {F(1.2 * R.d0, 1)} mm", dims=["bp", "lp"],
               elems=["anchors"])
            geo_ok = False
        if R.n_t > 1:
            p_min = min(math.hypot(a["x"] - b["x"], a["y"] - b["y"]) for i, a in enumerate(R.tiges) for b in R.tiges[i + 1:])
            R.dist.append(Pince("p_pl", "Entraxe minimal entre boulons", p_min, 2.2 * R.d0, None, 2.2, [], [],
                                "p ≥ 2,2·d0", p_min >= 2.2 * R.d0 - 1e-9, "dist_p"))
            if p_min < 2.2 * R.d0 - 1e-9:
                al("dist_p", "Entraxe de deux boulons inférieur à 2,2·d0 (EN 1993-1-8 Tableau 3.3).", True, [],
                   f"p = {F(p_min, 0)} mm < {F(2.2 * R.d0, 1)} mm", elems=["anchors"])
                geo_ok = False
        for t in R.tiges:
            for i, e in enumerate(R.elems):
                if EL.geometrie_valide(e) and EL.dedans(e, t["x"], t["y"], R.d0 / 2):
                    al(f"bin_{t['i']}", f"Boulon {t['i'] + 1} ({F(t['x'], 0)} ; {F(t['y'], 0)}) dans l'emprise de "
                       f"l'élément {i + 1} ({EL.nom(e)}).", True, [], "", elems=["anchors", "col"])
                    geo_ok = False
                    break
            for i, r in enumerate(R.raid):
                if EL.dedans_rect(EL.rect_raid(r), t["x"], t["y"], R.d0 / 2):
                    al(f"brd_{t['i']}", f"Boulon {t['i'] + 1} dans l'emprise du raidisseur {i + 1}.", True, [], "",
                       elems=["anchors", "raid"])
                    geo_ok = False
                    break
    R.dist_ok = all(p.ok for p in R.dist)
    xy = [(t["x"], t["y"]) for t in R.tiges]

    # ---------------------------------- appui le plus proche de chaque boulon
    for t in R.tiges:
        cands = []
        for i, e in enumerate(R.elems):
            if EL.geometrie_valide(e):
                cands.append(EL.appui_proche(e, i, t["x"], t["y"]))
        for i, r in enumerate(R.raid):
            cands.append(EL.appui_proche_raid(r, i, t["x"], t["y"]))
        a = min(cands, key=lambda c: c["d"]) if cands else dict(d=0.0, px=t["x"], py=t["y"], face="—", L=0.0, s=0.0,
                                                                coin=False, nx=1.0, ny=0.0, dedans=True, faces=[])
        t["appui"] = a
        t["m"] = max(1.0, a["d"] - cord)
        t["coin"] = bool(a["coin"])
        t["cercle"] = bool(a.get("cercle"))
        nx, ny = a["nx"], a["ny"]
        if abs(nx) + abs(ny) < 1e-9:
            nx, ny = 1.0, 0.0
        t["e_perp"] = CO._sortie_rect(t["x"], t["y"], nx, ny, plaque)
        t["e_lat"] = min(CO._sortie_rect(t["x"], t["y"], -ny, nx, plaque), CO._sortie_rect(t["x"], t["y"], ny, -nx, plaque))
        voisins = [math.hypot(t["x"] - o["x"], t["y"] - o["y"]) for o in R.tiges if o is not t]
        t["s_min"] = min(voisins) if voisins else CP.INF
        t["cap"] = t["m"] if a["coin"] else (a["L"] / 2 + t["m"])
        # sections au nu : la face de l'appui le plus proche — les DEUX faces de
        # l'angle pour un boulon d'angle —, chacune avec son bras perpendiculaire
        t["faces"] = []
        for f in a.get("faces") or []:
            (px, py), (fnx, fny) = f["p0"], f["n"]
            h = (t["x"] - px) * fnx + (t["y"] - py) * fny
            if h > 1e-9 and f["L"] > 0:
                t["faces"].append((f, max(1.0, h - cord)))
    if any(t["coin"] for t in R.tiges):
        al("coin", "Boulons dans un angle (au-delà d'une face ET de son about) : bras jusqu'à l'angle, longueur "
           "efficace plafonnée à m par boulon — modèle d'ingénieur, à valider.", False, [],
           ", ".join(f"boulon {t['i'] + 1}" for t in R.tiges if t["coin"]), elems=["anchors"])

    # ------------------------------------------------------------ efforts
    N_ = u.N_Ed * 1e3
    xN, yN = u.xN or 0.0, u.yN or 0.0
    Mx_o = u.Mx_Ed * 1e6 + N_ * yN
    My_o = u.My_Ed * 1e6 + N_ * xN
    Vx, Vy, T_ = u.Vx_Ed * 1e3, u.Vy_Ed * 1e3, u.T_Ed * 1e6
    V_res = math.hypot(Vx, Vy)
    R.xN, R.yN, R.Mx_o, R.My_o = xN, yN, Mx_o / 1e6, My_o / 1e6
    if abs(N_ * yN) > 1e-6 or abs(N_ * xN) > 1e-6:
        al("exc", "Efforts appliqués hors du centre de la platine : N·yN et N·xN s'ajoutent aux moments.", False,
           ["xN", "yN"], f"Mx = {F(u.Mx_Ed, 1)} + {F(N_ * yN / 1e6, 1)} = {F(Mx_o / 1e6, 1)} kNm ; My = {F(u.My_Ed, 1)} + "
           f"{F(N_ * xN / 1e6, 1)} = {F(My_o / 1e6, 1)} kNm", elems=["efforts"])

    # cisaillement par boulon : V partagé, torsion autour du centre du groupe, frottement déduit (EC3)
    R.Ff_Rd = (u.C_f or 0.0) * max(0.0, N_)
    gx = sum(t["x"] for t in R.tiges) / R.n_t if R.tiges else 0.0
    gy = sum(t["y"] for t in R.tiges) / R.n_t if R.tiges else 0.0
    R.gx, R.gy = gx, gy
    T_G = T_ + (xN - gx) * Vy - (yN - gy) * Vx
    R.T_G = T_G / 1e6
    Sr2 = sum((t["x"] - gx) ** 2 + (t["y"] - gy) ** 2 for t in R.tiges) or 1.0
    ux, uy = (Vx / V_res, Vy / V_res) if V_res > 0 else (0.0, 0.0)
    for t in R.tiges:
        tx, ty = -(t["y"] - gy) * T_G / Sr2, (t["x"] - gx) * T_G / Sr2
        v_ec3 = max(0.0, V_res - R.Ff_Rd) / max(R.n_t, 1)
        v_en = V_res / max(R.n_t, 1)
        t["vx"], t["vy"] = v_ec3 * ux + tx, v_ec3 * uy + ty
        t["V_ec3"] = math.hypot(t["vx"], t["vy"]) / 1e3
        t["V_en"] = math.hypot(v_en * ux + tx, v_en * uy + ty) / 1e3
    R.V_t_ec3 = max([t["V_ec3"] for t in R.tiges] or [0.0])
    R.V_t_en = max([t["V_en"] for t in R.tiges] or [0.0])

    # ----------------------------------------------- trous : pression diamétrale
    R.appuis = []
    for t in R.tiges:
        v = math.hypot(t["vx"], t["vy"])
        if v <= 1e-9:
            t["t_b"] = 0.0
            R.appuis.append(dict(i=t["i"], Fv=0.0, e1=0.0, p1=CP.INF, e2=0.0, p2=CP.INF, ab=0.0, k1=0.0, Fb=0.0, t_b=0.0))
            continue
        ux_, uy_ = t["vx"] / v, t["vy"] / v
        e1 = CO._sortie_rect(t["x"], t["y"], ux_, uy_, plaque)
        e2 = min(CO._sortie_rect(t["x"], t["y"], -uy_, ux_, plaque), CO._sortie_rect(t["x"], t["y"], uy_, -ux_, plaque))
        p1 = p2 = CP.INF
        for o in R.tiges:
            if o is t:
                continue
            dx, dy = o["x"] - t["x"], o["y"] - t["y"]
            along, perp = dx * ux_ + dy * uy_, -dx * uy_ + dy * ux_
            if along > 1e-9 and abs(perp) < R.d0:
                p1 = min(p1, along)
            if abs(along) < R.d0 and abs(perp) > 1e-9:
                p2 = min(p2, abs(perp))
        ab = max(0.0, min(e1 / (3 * R.d0), p1 / (3 * R.d0) - 0.25, R.f_ub / R.fu_p, 1.0))
        k1 = max(0.0, min(2.8 * e2 / R.d0 - 1.7, 1.4 * p2 / R.d0 - 1.7, 2.5))
        Fb = k1 * ab * R.fu_p * R.d_a * u.tp_u / gM2
        t_b = v * gM2 / (k1 * ab * R.fu_p * R.d_a) if k1 * ab > 0 else float("inf")
        t["t_b"] = t_b
        R.appuis.append(dict(i=t["i"], Fv=v / 1e3, e1=e1, p1=p1, e2=e2, p2=p2, ab=ab, k1=k1, Fb=Fb / 1e3, t_b=t_b))
    pire_b = max(R.appuis, key=lambda a: a["Fv"] / a["Fb"] if a["Fb"] > 0 else (1e9 if a["Fv"] > 0 else 0.0)) \
        if R.appuis else None
    R.pire_appui = pire_b

    # -------------------------------------- sources des bandes comprimées
    # faces de l'ENVELOPPE de chaque élément (vers l'extérieur), faces
    # intérieures des ailes et des parois d'un profil creux (vers le vide),
    # cercle extérieur et intérieur d'un rond, quatre faces d'un raidisseur ;
    # chacune avec SON appui (exclu des « autres » du partage de Voronoï)
    sources = []
    for i, e in enumerate(R.elems):
        if not EL.geometrie_valide(e):
            continue
        etq = f"élément {i + 1}"
        if EL.est_rond(e):
            prop = ("c", e["x"], e["y"], EL.rayons(e)[0])
            sources.append(dict(kind="cercle", nom=etq + " cercle", cx=e["x"], cy=e["y"], R=EL.rayons(e)[0], sens=1,
                                propre=prop))
            if e["type"] == TYPE_CHS:
                sources.append(dict(kind="cercle", nom=etq + " intérieur du tube", cx=e["x"], cy=e["y"],
                                    R=EL.rayons(e)[1], sens=-1, propre=prop))
            continue
        env = EL.enveloppe(e)
        for f in EL.faces_rect(etq + " enveloppe", env):
            # la paroi propre de la face d'enveloppe : la paroi qui la porte
            prop = None
            for (n_, x0, y0, x1, y1) in EL.parois(e):
                mx_, my_ = 0.5 * (f["p0"][0] + f["p1"][0]), 0.5 * (f["p0"][1] + f["p1"][1])
                if EL.dedans_rect((x0, y0, x1, y1), mx_ - 0.01 * f["n"][0], my_ - 0.01 * f["n"][1], 1e-6):
                    prop = (x0, y0, x1, y1)
            sources.append(dict(kind="face", nom=f["nom"], face=f, propre=prop))
        if e["type"] in (TYPE_I, TYPE_RHS):
            cx, cy = e["x"], e["y"]
            for (n_, x0, y0, x1, y1) in EL.parois(e):
                for f in EL.faces_rect(etq + " " + n_, (x0, y0, x1, y1)):
                    mx_, my_ = 0.5 * (f["p0"][0] + f["p1"][0]), 0.5 * (f["p0"][1] + f["p1"][1])
                    if (cx - mx_) * f["n"][0] + (cy - my_) * f["n"][1] > 1e-9:
                        sources.append(dict(kind="face", nom=f["nom"], face=f, propre=(x0, y0, x1, y1)))
    for i, r in enumerate(R.raid):
        rr = EL.rect_raid(r)
        for f in EL.faces_rect(f"raidisseur {i + 1}", rr):
            sources.append(dict(kind="face", nom=f["nom"], face=f, propre=rr))
    R.sources = sources

    # ----------------------------------------- l'état pour une épaisseur tp
    def tstub(t, tp, fy):
        return CP.tstub_rangee([dict(e_lat=t["e_lat"], s_min=t["s_min"])], t["m"], t["e_perp"], tp, fy, gM0,
                               8 * R.d_a + u.t_g + tp + 0.6 * R.d_a, R.As, R.Ft_anc, t["cap"])

    def etat(tp, final=False):
        """Tout ce qui dépend de l'épaisseur : bande c, aire d'appui,
        raideurs, répartition, tronçons en T, sections, console, et la
        faisabilité des quatre critères."""
        S = dict(tp=tp, fy=fy_epaisseur(fy_nom, tp), ok=False, inf=(True, True, True, True))
        fy = S["fy"]
        lim = fy / gM0
        c = CP.c_bande(tp, fy, R.fjd, gM0)
        S["c"] = c
        bandes = []
        if geo_ok:
            for e in R.elems:
                bandes += EL.bandes(e, c, plaque)
            for r in R.raid:
                b = EL.bande_raid(r, c, plaque)
                if b:
                    bandes.append(b)
        S["A_bandes_rects"] = GE.union_rectangles(bandes) if bandes else []
        S["A_eff_rects"] = [plaque] if (plein and geo_ok) else S["A_bandes_rects"]
        S["A_eff"] = GE.aire_rects(S["A_eff_rects"])
        S["A_bandes"] = GE.aire_rects(S["A_bandes_rects"])
        A_tstub = (u.bp_u * u.lp_u) if (plein and geo_ok) else max([(x1 - x0) * (y1 - y0) for (x0, y0, x1, y1) in bandes] or [0.0])
        S["A_tstub"] = A_tstub
        S["k_c"] = R.E_cm / (1.275 * math.sqrt(A_tstub)) if A_tstub > 0 else 0.0
        ts_list = [tstub(t, tp, fy) for t in R.tiges]
        levier = any(ts["levier"] for ts in ts_list)
        S["levier"] = levier
        Lb = 8 * R.d_a + u.t_g + tp + 0.6 * R.d_a
        S["Lb"] = Lb
        S["k_b"] = E * (1.6 if levier else 2.0) * R.As / Lb
        el = EQ.elastique(S["A_eff_rects"], xy, S["k_c"], S["k_b"], N_, Mx_o, My_o) if (geo_ok and S["k_c"] > 0) \
            else dict(ok=False, contact=[], T=[0.0] * len(xy), C=0.0, xC=0.0, yC=0.0, pmax=0.0, A_c=0.0,
                      a=0.0, bx=0.0, by=0.0, iter=0)
        S["el"] = el
        S["ok"] = bool(el["ok"])
        T = [Ti for Ti in el["T"]]
        S["T"] = T
        # côté tendu, boulon par boulon
        bol = []
        ok_T = True
        for t, ts, T_N in zip(R.tiges, ts_list, T):
            d = dict(ts=ts, T=T_N / 1e3, M=T_N * t["m"] / 1e6, leff1=ts["leff1"], leff2=ts["leff2"],
                     Mpl1=ts["Mpl1"] / 1e6, FT_Rd=ts["FT_tige"] / 1e3, levier=ts["levier"], mode_ts=ts["mode"],
                     Lb_star=ts["Lb_star"])
            # la charnière du tronçon en T d'un boulon mesure 2·leff,1 (convention
            # par boulon du Tableau 6.6) : la bande élastique a cette largeur
            d["t_pl"] = (math.sqrt((1.0 if ts["levier"] else 2.0) * T_N * t["m"] * gM0 / (ts["leff1"] * fy))
                         if ts["leff1"] > 0 and T_N > 0 else 0.0)
            d["b_el"] = 2.0 * ts["leff1"]
            d["sig"] = 6.0 * T_N * t["m"] / (d["b_el"] * tp ** 2) if ts["leff1"] > 0 else 0.0
            d["tau"] = T_N / (d["b_el"] * tp) if ts["leff1"] > 0 else 0.0
            d["svm"] = math.sqrt(d["sig"] ** 2 + 3 * d["tau"] ** 2)
            d["t_el"] = math.sqrt(3.0 * T_N * t["m"] * gM0 / (ts["leff1"] * fy)) if ts["leff1"] > 0 and T_N > 0 else 0.0
            if T_N > 1e-9:
                if elastique:
                    d["eta"] = d["svm"] / lim
                    ok_T = ok_T and d["svm"] <= lim + 1e-9
                    d["t_el_vm"] = d["t_el"]
                else:
                    d["eta"] = (d["T"] / d["FT_Rd"]) if d["FT_Rd"] > 0 else 1e9
                    ok_T = ok_T and T_N <= ts["FT_tige"] + 1e-6
            else:
                d["eta"] = 0.0
            bol.append(d)
        S["boulons"] = bol
        # côté tendu, face par face
        par_face = {}
        for t, d in zip(R.tiges, bol):
            if d["T"] <= 1e-9:
                continue
            for f, mperp in t["faces"]:
                par_face.setdefault(f["nom"], (f, []))[1].append((t, d, mperp))
        secs = []
        ok_S = True
        for nom_, (f, membres) in par_face.items():
            ny = f["n"][1]
            m_max = max(mp for _t, _d, mp in membres)
            L_pl = u.bp_u if abs(ny) > 0.5 else u.lp_u
            b = min(L_pl, f["L"] + 2 * m_max)
            M = sum(d["T"] * 1e3 * mp for _t, d, mp in membres)                   # N·mm
            V = sum(d["T"] * 1e3 for _t, d, _mp in membres)
            Mpl = 0.25 * b * tp ** 2 * fy / gM0
            svm = _sigma_vm(M, V, b, tp)
            s = dict(nom=nom_, idx=[t["i"] for t, _d, _mp in membres], m=[mp for _t, _d, mp in membres], m_max=m_max,
                     L=f["L"], b=b, M=M / 1e6, V=V / 1e3, Mpl=Mpl / 1e6, t_pl=math.sqrt(4 * M * gM0 / (b * fy)),
                     svm=svm, t_el=math.sqrt(6 * M * gM0 / (b * fy)), p0=f["p0"], p1=f["p1"], n=f["n"],
                     eta=(svm / lim) if elastique else (M / Mpl if Mpl > 0 else 1e9))
            ok_S = ok_S and (svm <= lim + 1e-9 if elastique else M <= Mpl + 1e-6)
            secs.append(s)
        # rond : section tangente de chaque boulon qui tire sur un cercle
        for t, d in zip(R.tiges, bol):
            if d["T"] > 1e-9 and t["cercle"]:
                Rr = t["appui"]["L"] / 2
                b = min(u.bp_u, u.lp_u, 2 * Rr + 2 * t["m"])
                M = d["T"] * 1e3 * t["m"]
                Mpl = 0.25 * b * tp ** 2 * fy / gM0
                svm = _sigma_vm(M, d["T"] * 1e3, b, tp)
                secs.append(dict(nom=t["appui"]["face"] + f" (tangente, boulon {t['i'] + 1})", idx=[t["i"]],
                                 m=[t["m"]], m_max=t["m"], L=2 * Rr, b=b, M=M / 1e6, V=d["T"], Mpl=Mpl / 1e6,
                                 t_pl=math.sqrt(4 * M * gM0 / (b * fy)), svm=svm, t_el=math.sqrt(6 * M * gM0 / (b * fy)),
                                 p0=None, p1=None, n=None, eta=(svm / lim) if elastique else (M / Mpl if Mpl > 0 else 1e9)))
                ok_S = ok_S and (svm <= lim + 1e-9 if elastique else M <= Mpl + 1e-6)
        S["sections"] = secs
        # côté comprimé, bande par bande
        a_, bx_, by_ = el["a"], el["bx"], el["by"]
        k_c = S["k_c"]

        def press(x, y):
            return max(0.0, k_c * (a_ + bx_ * x + by_ * y))
        pire_c = None
        if S["ok"] and el["contact"]:
            for src in sources:
                autres = [a for a in appuis_geo if a != src["propre"]]
                if src["kind"] == "face":
                    b = CO.bandes_face(src["face"], plaque, autres, el["contact"], press)
                else:
                    b = CO.bandes_cercle(src["cx"], src["cy"], src["R"], src["sens"], plaque, autres, el["contact"], press)
                if b and (pire_c is None or b["m"] > pire_c["m"]):
                    b = dict(b)
                    b["nom"] = src["nom"]
                    pire_c = b
        m_Ed = pire_c["m"] if pire_c else 0.0                                 # N·mm/mm
        m_el = tp ** 2 * fy / (6.0 * gM0)
        S["pire_c"] = pire_c
        S["m_Ed"], S["m_el"] = m_Ed, m_el
        ok_c = m_Ed <= m_el + 1e-6
        ok_b = all(a["t_b"] <= tp + 1e-9 for a in R.appuis)
        S["inf"] = (not ok_T, not ok_S, not ok_c, not ok_b)
        return S

    S = etat(u.tp_u, True)
    # ----------------------------------------- l'état retenu, dans R
    R.c, R.A_eff_rects, R.A_bandes_rects, R.A_eff = S["c"], S["A_eff_rects"], S["A_bandes_rects"], S["A_eff"]
    R.N_Rd = R.fjd * S["A_bandes"]
    R.A_tstub, R.k_c, R.k_b, R.levier = S["A_tstub"], S["k_c"], S["k_b"], S["levier"]
    el = S["el"]
    R.eq_ok = bool(el["ok"])
    if geo_ok and not R.eq_ok:
        al("eq", "Équilibre introuvable : les boulons tendus et l'appui ne peuvent pas équilibrer ces efforts "
           "(moment autour d'une ligne de boulons sans bras, soulèvement sans boulon…).", True,
           ["N_Ed", "Mx_Ed", "My_Ed"], f"N = {F(u.N_Ed, 0)} kN, Mx = {F(Mx_o / 1e6, 1)}, My = {F(My_o / 1e6, 1)} kNm",
           elems=["efforts", "anchors"])
    R.contact, R.A_c = el["contact"], el["A_c"]
    R.C, R.xC, R.yC, R.pmax = el["C"] / 1e3, el["xC"], el["yC"], el["pmax"]
    R.axe = (el["a"], el["bx"], el["by"])
    for t, d in zip(R.tiges, S["boulons"]):
        t.update(d)
    R.T_tot = sum(S["T"]) / 1e3
    R.T_max = max(S["T"]) / 1e3 if S["T"] else 0.0
    R.soulevement = R.T_tot > 1e-6
    if R.soulevement:
        al("soul", "Boulons tendus : " + ", ".join(f"n° {t['i'] + 1} ({F(t['x'], 0)} ; {F(t['y'], 0)}) {F(t['T'], 1)} kN"
                                                   for t in R.tiges if t["T"] > 1e-6) + ".", False, [],
           f"{sum(1 for t in R.tiges if t['T'] > 1e-6)} boulon(s) tendu(s), T max = {F(R.T_max, 1)} kN, "
           f"ΣT = {F(R.T_tot, 1)} kN", elems=["anchors"])
    tendus = [t for t in R.tiges if t["T"] > 1e-9]
    pire_t = max(tendus, key=lambda t: t["eta"]) if tendus else None
    R.pire_tige = pire_t
    R.sections = S["sections"]
    pire_s = max(R.sections, key=lambda s: s["eta"]) if R.sections else None
    R.pire_section = pire_s
    pire_c = S["pire_c"]
    m_Ed, m_el = S["m_Ed"], S["m_el"]
    R.console = dict(m_Ed=m_Ed / 1e3, m_Rd=m_el / 1e3, nom=pire_c["nom"] if pire_c else "—",
                     L=pire_c["L"] if pire_c else 0.0, p0=pire_c["p0"] if pire_c else 0.0,
                     ox=pire_c["ox"] if pire_c else 0.0, oy=pire_c["oy"] if pire_c else 0.0,
                     nx=pire_c["nx"] if pire_c else 0.0, ny=pire_c["ny"] if pire_c else 0.0,
                     t=math.sqrt(6 * m_Ed * gM0 / R.fy_p) if R.fy_p > 0 else 0.0)
    if plein and any(e["type"] in (TYPE_I, TYPE_RHS, TYPE_CHS) for e in R.elems):
        al("plein", "Appui sur toute la platine : entre les ailes, dans le vide d'un tube et au-delà des abouts, "
           "chaque paroi reprend en console la pression qui lui est la plus proche — approximation du côté de la "
           "sécurité ; l'aire efficace EN 1993-1-8 est le mode de référence.", False, ["appui"], "", elems=["plate"])

    # ------------------------------------------- ÉPAISSEUR REQUISE (balayage)
    if geo_ok and R.eq_ok:
        t_req = _epaisseurs_requises(lambda tp: etat(tp)["inf"], u.tp_u, CRITERES_TP, _grille_grossiere())
    else:
        t_req = {k: float("inf") for k in CRITERES_TP}
    t_tot = max(t_req.values())
    R.plaque = dict(t_req=t_tot, t_T=t_req["T"], t_S=t_req["S"], t_c=t_req["c"], t_b=t_req["b"],
                    critere=("élastique (Von Mises)" if elastique else "EN 1993-1-8 (plastique côté tendu)"),
                    pire=max(CRITERES_TP, key=lambda k: t_req[k]))

    # --------------------------------------------------------- vérifications
    lim = R.fy_p / gM0
    ck("tpq", "Platine", "Épaisseur requise de la platine (boulons, sections, console comprimée, pression "
       "diamétrale)", t_tot, u.tp_u, "mm", "EN 1993-1-8 §6.2.4, §6.2.5, Tableau 3.4 ; calcul manuel", "EC + INT",
       geo_ok and R.eq_ok,
       "tp ≥ max(tT ; tS ; tc ; tb) — tT : tronçon en T de chaque boulon ; tS : section au nu de chaque paroi ; "
       "tc : bande comprimée ; tb : pression diamétrale — répartition recalculée à chaque épaisseur",
       f"tT = {F(t_req['T'], 1)} ; tS = {F(t_req['S'], 1)} ; tc = {F(t_req['c'], 1)} ; tb = {F(t_req['b'], 1)} mm", 1)
    if elastique:
        ck("ptl", "Platine", "Côté tendu — bande du boulon le plus sollicité : σ de Von Mises", pire_t["svm"] if pire_t else 0.0,
           lim, "MPa", REF_T62 + " ; contrainte élastique de la bande", "INT", pire_t is not None,
           "σ = 6·T·m/(2·leff·tp²), τ = T/(2·leff·tp), √(σ² + 3τ²) ≤ fy/γM0 (charnière de 2·leff par boulon)",
           (f"boulon {pire_t['i'] + 1} : T = {F(pire_t['T'], 1)} kN, m = {F(pire_t['m'], 1)} mm, leff = "
            f"{F(pire_t['leff1'], 0)} mm") if pire_t else "", 1)
    else:
        ck("ptl", "Platine", "Côté tendu — tronçon en T du boulon le plus sollicité", pire_t["T"] if pire_t else 0.0,
           pire_t["FT_Rd"] if pire_t else None, "kN", REF_T62, "EC + INT", pire_t is not None,
           "FT,Rd = min(mode 1 ; mode 2 ; mode 3) par boulon ; mode 1 sans effet de levier si Lb > Lb*",
           (f"boulon {pire_t['i'] + 1} : m = {F(pire_t['m'], 1)} mm, leff,1 = {F(pire_t['leff1'], 0)} mm, Mpl,1 = "
            f"{F(pire_t['Mpl1'], 2)} kNm, mode {pire_t['mode_ts']}") if pire_t else "", 1)
    if elastique:
        ck("pts", "Platine", "Côté tendu — section au nu de la paroi la plus sollicitée : σ de Von Mises",
           pire_s["svm"] if pire_s else 0.0, lim, "MPa", REF_SEC, "INT", pire_s is not None,
           "σ = 6·ΣT·m/(b·tp²), b = min(platine ; L + 2·m), τ = ΣT/(b·tp)",
           (f"{pire_s['nom']} : M = {F(pire_s['M'], 2)} kNm, b = {F(pire_s['b'], 0)} mm") if pire_s else "", 1)
    else:
        ck("pts", "Platine", "Côté tendu — section au nu de la paroi la plus sollicitée : M ≤ Mpl",
           pire_s["M"] if pire_s else 0.0, pire_s["Mpl"] if pire_s else None, "kNm", REF_SEC, "INT",
           pire_s is not None, "M = ΣT·m ≤ Mpl = 0,25·b·tp²·fy/γM0, b = min(platine ; L + 2·m)",
           (f"{pire_s['nom']} : boulons {', '.join(str(i + 1) for i in pire_s['idx'])}, b = {F(pire_s['b'], 0)} mm")
           if pire_s else "", 1)
    ck("pc", "Platine", "Côté comprimé — bande la plus sollicitée au nu d'une face : flexion élastique", m_Ed / 1e3,
       m_el / 1e3, "kNm/m", REF_65 + " (contrôle élastique de la bande)", "EC + INT", R.eq_ok and pire_c is not None,
       "mEd = ∫p·t dt sur la bande en console (chaque paroi reprend la pression qui lui est la plus proche) "
       "≤ tp²·fy/(6·γM0)",
       (f"{pire_c['nom']} : p au nu = {F(pire_c['p0'], 2)} MPa, console de {F(pire_c['L'], 0)} mm") if pire_c else "", 1)
    ck("pb", "Trous", "Pression diamétrale du boulon le plus cisaillé sur la platine", pire_b["Fv"] if pire_b else 0.0,
       pire_b["Fb"] if pire_b else None, "kN", REF_34, "EC", bool(pire_b and pire_b["Fv"] > 1e-9),
       "Fb,Rd = k1·αb·fu·d·tp/γM2 ; αb = min(e1/(3d0) ; p1/(3d0) − 1/4 ; fub/fu ; 1) ; k1 = min(2,8·e2/d0 − 1,7 ; "
       "1,4·p2/d0 − 1,7 ; 2,5)",
       (f"boulon {pire_b['i'] + 1} : e1 = {F(pire_b['e1'], 0)}, p1 = {('∞' if pire_b['p1'] == CP.INF else F(pire_b['p1'], 0))}, "
        f"e2 = {F(pire_b['e2'], 0)}, p2 = {('∞' if pire_b['p2'] == CP.INF else F(pire_b['p2'], 0))} mm ; αb = "
        f"{F(pire_b['ab'], 2)}, k1 = {F(pire_b['k1'], 2)}") if pire_b else "")
    # information : tiges (acier) et appui
    ck("ta", "Tiges", "Traction du boulon le plus tendu (acier) — information", R.T_max, R.Ft_anc / 1e3, "kN",
       "EN 1993-1-8 Tableau 3.4 ; EN 1992-4 §7.2.1.3", "EC", R.soulevement,
       "Ft,Rd = 0,9·fub·As/γM2 ; NRd,s = As·fuk/γMs,N",
       f"Ft,Rd = {F(R.Ft_EC3 / 1e3, 1)} kN ; NRd,s = {F(R.N_Rd_s / 1e3, 1)} kN")
    ck("tv", "Tiges", "Cisaillement du boulon le plus cisaillé (acier, frottement déduit) — information", R.V_t_ec3,
       R.F2vb / 1e3, "kN", "EN 1993-1-8 §6.2.2", "EC", V_res > 0 or abs(T_) > 0,
       "F2,vb,Rd = αb·fub·As/γM2, αb = 0,44 − 0,0003·fyb ; Ff,Rd = Cf,d·Nc,Ed",
       f"Ff,Rd = {F(R.Ff_Rd / 1e3, 1)} kN ; αb = {F(R.alpha_b, 3)}")
    eta_int = 0.0
    if R.soulevement and (V_res > 0 or abs(T_) > 0):
        eta_int = max(t["V_ec3"] / (R.F2vb / 1e3) + t["T"] / (1.4 * R.Ft_EC3 / 1e3) for t in R.tiges)
    ck("tvn", "Tiges", "Interaction traction / cisaillement (acier) — information", eta_int, 1.0, "-",
       "EN 1993-1-8 Tableau 3.4", "EC", R.soulevement and (V_res > 0 or abs(T_) > 0),
       "Fv,Ed/Fv,Rd + Ft,Ed/(1,4·Ft,Rd) ≤ 1", "")
    ck("bet", "Appui", "Pression maximale sous la platine ≤ fjd — information", R.pmax, R.fjd, "MPa",
       REF_65 + " — répartition élastique", "INT", R.eq_ok, "pmax = kc·wmax ; fjd = βj·fcd·√(Ac1/Ac0)",
       f"fjd = {F(R.fjd, 1)} MPa ; kc = {F(R.k_c, 0)} N/mm³")
    ck("ncp", "Appui", "Compression N ≤ fjd·Aeff (bande c) — information", max(0.0, u.N_Ed), R.N_Rd / 1e3, "kN",
       "EN 1993-1-8 §6.2.8.2 et §6.2.5", "EC", N_ > 0 and not plein,
       "Aeff = parois élargies de c, découpées par la platine", f"c = {F(R.c, 1)} mm ; Aeff = {F(S['A_bandes'] / 100, 0)} cm²")
    if R.soulevement and R.T_max > R.Ft_anc / 1e3 + 1e-9:
        al("tige_info", "Un boulon dépasse sa résistance en traction (acier) — information : le verdict porte sur "
           "la platine, les ancrages relèvent de leur logiciel.", False, ["ancr", "classe"],
           f"T max = {F(R.T_max, 1)} kN > {F(R.Ft_anc / 1e3, 1)} kN", elems=["anchors"])
    if R.eq_ok and R.pmax > R.fjd + 1e-9:
        al("fjd_info", "La pression sous la platine dépasse fjd — information : le béton d'appui relève de la "
           "vérification complète.", False, ["beton", "bc_u", "lc_u"],
           f"p max = {F(R.pmax, 1)} MPa > fjd = {F(R.fjd, 1)} MPa", elems=["pedestal"])

    # ------------------------------------------------------------ synthèse
    R.regulier = disposition_reguliere(u)
    R.geo_ok = bool(geo_ok and R.eq_ok)
    act = [c for c in R.checks if c.active]
    gov = None
    for c in act:
        if c.key in R.info_keys:
            continue
        if gov is None or c.eta > gov.eta:
            gov = c
    R.gov = gov
    R.eta_max = gov.eta if gov else 0.0
    R.all_ok = all(c.ok for c in act if c.key not in R.info_keys)
    R.verified = bool(R.all_ok and R.geo_ok and R.dist_ok)
    R.statut = ("PLATINE VÉRIFIÉE" if R.verified else "PLATINE NON VÉRIFIÉE") + " (calcul manuel)"
    R.reserve = any(a.id in ("coin", "plein", "tige_info", "fjd_info") for a in R.alerts)
    return R

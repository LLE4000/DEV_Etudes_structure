# -*- coding: utf-8 -*-
"""Dessins de l'assemblage par plat d'âme soudé — élévation, vue en plan,
vue de droite. Le KIT de dessin (primitives, Feuille de cotation, styles,
rendu SVG, lignes de rupture, hachures) est celui du module doubles
cornières, réutilisé tel quel ; seules les SCÈNES sont propres à cet
assemblage.

Cotes : symboles finaux directement (hp, zp, e1, p1, e2,b, gh, Δz, c,
dc,sup…) — pas de couche de notation (aucun moteur de référence à imiter).
"""
import math

from acier.js import N, mn, mx, js_str
from acier.formats import f0
from acier.bibliotheques import PERSO
from ..doubles_cornieres.schemas import (            # kit partagé
    Options, Feuille, PALETTE, INF_MM, SEGMENTS_ARC,
    _rect, _poly, _path, _circle, _text, _group, _arc, _hachures,
    _rect_rompu, _rupture, _renvoi, _section_I, _section_portee)
from .entrees import PILOTEES_PAR_PREDIM

# Cote du dessin → clé d'entrée (édition directe sur le dessin)
CLE_PAR_COTE = {"e2b": "e2b_u", "p2": "p2_u", "gh": "g_h", "c": "l_n",
                "dcs": "d_nt", "dci": "d_nb", "hp": "hp_u", "zp": "z_C",
                "e1": "e1_u", "p1": "p1_u", "bp": "bp_u", "dz": "d_top",
                "aw": "a_w", "tp": "tp_u"}

GROUPE_BOULONS = dict(n1="n1_u", n2="n2_u", titre="Groupe de boulons – plat et âme de la portée")

# Plan de principe : grandeurs calculées admises, et répartition des cotes
FABRICATION_CALC = frozenset({"ztp", "bS", "e2p", "twS"})
EXCLURE_ELEVATION = frozenset({"ztp", "bS", "twS"})
EXCLURE_PLAN = frozenset({"e2b", "p2", "gh", "bp", "dz", "e2p", "ztp", "bS"})
EXCLURE_DROITE = frozenset({"e1", "p1", "zp", "e2b", "gh", "p2", "hp", "e2p",
                            "c", "dcs", "dci", "dz", "aw", "twS"})


def options_ecran(R, lvl=1, hl=None):
    """Options de l'écran : cotes modifiables cliquables, poignées, sans
    cartouche texte (colonne figée courte)."""
    return Options(lvl=lvl, interactive=True, hl=hl, editables=True, poignees=True,
                   cartouche=False, notation={},
                   locked=set(PILOTEES_PAR_PREDIM) if R.pred else None)


def options_rapport():
    return Options(lvl=1, interactive=False, hl=None, locked=None, cartouche=False,
                   notation={})


def options_fabrication(exclure=frozenset(), fs_force=0.0):
    return Options(lvl=1, interactive=False, cartouche=False, fabrication=True,
                   exclure=frozenset(exclure), fs_force=fs_force, notation={},
                   fabrication_calc=FABRICATION_CALC)


# ------------------------------------------------------------------ élévation
def elevation(R, opt):
    """Élévation : porteuse en coupe, portée grugée (glisser gh / Δz), PLAT
    soudé par-devant, groupe de boulons, symbole de soudure double."""
    u = R.u; xf = R.tw_P / 2; gh = N(u.g_h); x0 = xf + gh; yt = N(u.d_top); yb = yt + R.h_S
    zc = N(u.z_C); yc = yt + zc; dnt = N(u.d_nt); dnb = N(u.d_nb); ln = N(u.l_n)
    p2 = N(u.p2_u)
    notch = dnt > 0 or dnb > 0
    xE = mx(x0 + (ln if notch else 0), xf + R.b_p) + 75
    bb = dict(x1=-R.b_P / 2, x2=xE, y1=mn(0, yt), y2=mx(R.h_P, yb))
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 260) / 34, 9)
    xt = xf + R.b_p                                    # bout du plat
    txtG = js_str(R.n_1) + " × " + js_str(R.n_2)
    if opt.poignees:
        S0 = Feuille(bb, fs, opt)
        bb["x2"] = mx(bb["x2"], xt + 0.5 * fs + S0.largeur_poignees(txtG) + 0.3 * fs)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(k):
        return ["hot"] if k in hot else []

    def piece(grp, prims, **attrs):
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    # porteuse en coupe (I hachuré)
    if opt.realiste:
        sec = _section_I(R.b_P, R.h_P, R.tw_P, R.tf_P, R.r_P)
        piece("beamP", [_poly(["pp"], sec), _path(["ht"], _hachures(sec))])
        if "flPt" in hot:
            s.append(_path(["hl"], [[(-R.b_P / 2, R.tf_P), (-R.b_P / 2, 0), (R.b_P / 2, 0), (R.b_P / 2, R.tf_P)]]))
        if "flPb" in hot:
            s.append(_path(["hl"], [[(-R.b_P / 2, R.h_P - R.tf_P), (-R.b_P / 2, R.h_P), (R.b_P / 2, R.h_P), (R.b_P / 2, R.h_P - R.tf_P)]]))
    else:
        piece("beamP", [_rect(["pp"], -R.b_P / 2, 0, R.b_P, R.tf_P),
                        _rect(["pp"], -R.b_P / 2, R.h_P - R.tf_P, R.b_P, R.tf_P),
                        _rect(["pp"], -xf, R.tf_P, R.tw_P, R.h_P - 2 * R.tf_P)])
    # portée grugée, bord droit rompu
    rn = mn(N(u.r_n), ln, dnt if dnt > 0 else INF_MM, dnb if dnb > 0 else INF_MM) if opt.realiste else 0
    if dnt > 0:
        p = [[x0, yt + dnt]]
        if rn > 0:
            p += _arc(x0 + ln - rn, yt + dnt - rn, rn, math.pi / 2, 0)
        else:
            p.append([x0 + ln, yt + dnt])
        p += [[x0 + ln, yt]]
    else:
        p = [[x0, yt]]
    p += _rupture((xE, yt), (xE, yb), 0.55 * fs) if opt.realiste else [[xE, yt], [xE, yb]]
    if dnb > 0:
        p += [[x0 + ln, yb]]
        if rn > 0:
            p += _arc(x0 + ln - rn, yb - dnb + rn, rn, 0, -math.pi / 2)
        else:
            p.append([x0 + ln, yb - dnb])
        p.append([x0, yb - dnb])
    else:
        p.append([x0, yb])
    pS = [_poly(["ps"] + hc("beamS"), p),
          _path(["fl2"], [[(x0 + ln if dnt > 0 else x0, yt + R.tf_S), (xE, yt + R.tf_S)],
                          [(x0 + ln if dnb > 0 else x0, yb - R.tf_S), (xE, yb - R.tf_S)]])]
    if opt.realiste and R.r_S > 0:
        pS.append(_path(["flr"], [[(x0 + ln if dnt > 0 else x0, yt + R.tf_S + R.r_S), (xE, yt + R.tf_S + R.r_S)],
                                  [(x0 + ln if dnb > 0 else x0, yb - R.tf_S - R.r_S), (xE, yb - R.tf_S - R.r_S)]]))
    piece("beamS", pS, drag="g_h", dsym="gh", drag2="d_top", dsym2="Δz")
    if "notchT" in hot:
        s.append(_path(["hl"], [[(x0, yt + dnt), (x0 + ln, yt + dnt), (x0 + ln, yt)]] if dnt > 0
                       else [[(x0, yt + R.tf_S + R.r_S), (x0, yt), (mx(R.b_P / 2, x0 + 40), yt)]]))
    if "notchB" in hot:
        s.append(_path(["hl"], [[(x0, yb - dnb), (x0 + ln, yb - dnb), (x0 + ln, yb)]] if dnb > 0
                       else [[(x0, yb - R.tf_S - R.r_S), (x0, yb), (mx(R.b_P / 2, x0 + 40), yb)]]))
    if opt.fabrication and rn > 0 and dnt > 0:
        cxg, cyg = x0 + ln - rn, yt + dnt - rn
        _renvoi(S, s, cxg + rn * 0.707, cyg + rn * 0.707, cxg - 2.4 * fs, yt + dnt - rn - 0.9 * fs,
                "r " + f0(rn))
    # PLAT par-devant + cordon (bande a·√2 côté visible) + symbole double
    z = N(u.a_w) * math.sqrt(2)
    pl = [_rect(["co"] + hc("plate"), xf, yc, R.b_p, R.h_p)]
    if opt.realiste:
        pl.append(_rect(["wb"], xf, yc, z, R.h_p))
        # symbole EN 22553 : flèche au cordon, coude SOUS le plat (la ligne
        # de référence ne traverse ni le plat ni le groupe de boulons)
        ax_, ay = xf + z, yc + 0.85 * R.h_p
        ex, ey = ax_ + 1.7 * fs, yc + R.h_p + 1.9 * fs
        wt = S.tw("a " + f0(N(u.a_w)))
        tx = ex + wt + 0.9 * fs
        pl.append(_path(["wsy"], [[(ax_, ay), (ex, ey), (tx + 1.6 * fs, ey)],
                                  [(ax_ + 0.55 * fs, ay + 0.1 * fs), (ax_, ay), (ax_ + 0.1 * fs, ay + 0.55 * fs)]]))
        # cordon DOUBLE (une face et l'autre) : triangle au-dessus ET en
        # dessous de la ligne de référence — EN 22553
        pl.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey - 0.95 * fs]]))
        pl.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey + 0.95 * fs]]))
        S.tag(ex - 0.1 * fs, ey - 0.55 * fs, dict(id="aw", sym="a", val=N(u.a_w), key="a_w", lvl=1))
    piece("plate", pl)
    # groupe de boulons (rondelles, trous, axes)
    xc1 = xf + R.g_B; xcl = xc1 + (R.n_2 - 1) * p2
    yb1 = yc + R.e_1; ybl = yb1 + (R.n_1 - 1) * R.p_1; r0 = R.d_0 / 2
    bS = []
    for i in range(R.n_1):
        for j in range(R.n_2):
            cx = xc1 + j * p2; cy = yb1 + i * R.p_1
            if opt.realiste:
                bS.append(_circle(["bw"], cx, cy, R.d_w / 2))
            bS.append(_circle(["bo"] + hc("bolts"), cx, cy, r0))
            if not opt.realiste:
                bS.append(_path(["cm"], [[(cx - r0 - 4, cy), (cx + r0 + 4, cy)], [(cx, cy - r0 - 4), (cx, cy + r0 + 4)]]))
    if opt.realiste:
        dep = R.d_w / 2 + 5
        ax = [[(xc1 - dep, yb1 + i * R.p_1), (xcl + dep, yb1 + i * R.p_1)] for i in range(R.n_1)]
        ax += [[(xc1 + j * p2, yb1 - dep), (xc1 + j * p2, ybl + dep)] for j in range(R.n_2)]
        bS.append(_path(["ax"], ax))
    piece("bolts", bS)
    if opt.fabrication:
        _renvoi(S, s, xcl + R.d_w / 2 * 0.72, yb1 - R.d_w / 2 * 0.72,
                xcl + r0 + 1.9 * fs, yb1 - 1.7 * fs,
                js_str(R.n_1 * R.n_2) + "×Ø" + f0(R.d_0))
    if opt.poignees:
        autres = N(u.N_Ed) != 0 or N(u.M_Ed) != 0
        txtF = "VEd " + f0(N(u.V_Ed)) + (" +" if autres else "")
        wF = S.tw(txtF) + 0.5 * fs
        xF = (mx(x0 + (ln if notch else 0), xt) + xE) / 2
        hitF = _rect(["hit"], -wF / 2, -1.05 * fs, wF, 1.42 * fs); hitF["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitF, _text([], None, None, txtF, fs)],
                        x=xF, y=mx(yt + dnt, yc) + 1.7 * fs, rot=0, grp="efforts", id="efforts"))
    # --- cotes
    S.dim(dict(id="e2b", side="T", a=x0, b=xc1, o1=yt + dnt, o2=yb1, sym="e2,b", val=R.e2b, key="e2b_u", lvl=1))
    if R.n_2 > 1:
        S.dim(dict(id="p2", side="T", a=xc1, b=xcl, o1=yb1, o2=yb1, sym="p2", val=p2, key="p2_u", lvl=1))
    S.dim(dict(id="e2p", side="T", a=xcl, b=xt, o1=yb1, o2=yc, sym="e2", val=R.e2_p, lvl=2, calc=1))
    S.dim(dict(id="gh", side="T", a=xf, b=x0, o1=yc, o2=yt + dnt, sym="gh", val=gh, key="g_h", lvl=2, out="lo"))
    S.dim(dict(id="z", side="T", a=xf, b=xf + R.zeff, o1=yc, o2=yb1, sym="z", val=R.zeff, lvl=1, calc=1))
    if dnt > 0:
        S.dim(dict(id="c", side="T", a=x0, b=x0 + ln, o1=yt + dnt, o2=yt, sym="c", val=ln, key="l_n", lvl=1))
    elif dnb > 0:
        S.dim(dict(id="c", side="B", a=x0, b=x0 + ln, o1=yb - dnb, o2=yb, sym="c", val=ln, key="l_n", lvl=1))
    S.dim(dict(id="bp", side="B", a=xf, b=xt, o1=yc + R.h_p, o2=yc + R.h_p, sym="bp", val=R.b_p, key="bp_u", lvl=1))
    S.dim(dict(id="e1", side="R", a=yc, b=yb1, o1=xt, o2=xcl, sym="e1", val=R.e_1, key="e1_u", lvl=1, out="lo"))
    for i in range(1, R.n_1):
        S.dim(dict(id="p1", side="R", a=yb1 + (i - 1) * R.p_1, b=yb1 + i * R.p_1, o1=xcl, o2=xcl, sym="p1", val=R.p_1, key="p1_u", lvl=1))
    S.dim(dict(id="e1bot", side="R", a=ybl, b=yc + R.h_p, o1=xcl, o2=xt, sym="e1'", val=R.e1bot, lvl=2, calc=1))
    S.dim(dict(id="hp", side="R", a=yc, b=yc + R.h_p, o1=xt, o2=xt, sym="hp", val=R.h_p, key="hp_u", lvl=1))
    S.dim(dict(id="zp", side="R", a=yt, b=yc, o1=xE, o2=xt, sym="zp", val=zc, key="z_C", lvl=1, out="lo"))
    if dnt > 0:
        S.dim(dict(id="dcs", side="R", a=yt, b=yt + dnt, o1=x0 + ln, o2=x0 + ln, sym="dc,sup", val=dnt, key="d_nt", lvl=1, out="lo"))
    if dnb > 0:
        S.dim(dict(id="dci", side="R", a=yb - dnb, b=yb, o1=x0 + ln, o2=x0 + ln, sym="dc,inf", val=dnb, key="d_nb", lvl=1))
    S.dim(dict(id="e1b", side="R", a=yt + dnt, b=yb1, o1=x0 + (ln if dnt > 0 else 0), o2=xcl, sym="e1,b", val=R.e1b_S, lvl=2, calc=1, out="lo"))
    if dnb > 0:
        S.dim(dict(id="he", side="R", a=ybl, b=yb - dnb, o1=xcl, o2=x0 + ln, sym="he", val=R.h_e, lvl=2, calc=1))
    S.dim(dict(id="dz", side="L", a=0, b=yt, o1=-R.b_P / 2, o2=x0, sym="Δz", val=yt, key="d_top", lvl=2, out="lo"))
    if dnt == 0:
        S.tag(xt + 0.4 * fs, yt + R.tf_S + 1.5 * fs, dict(id="dcs", sym="dc,sup", val=0, key="d_nt", lvl=2))
    if dnb == 0:
        S.tag(xt + 0.4 * fs, yb - R.tf_S - 0.7 * fs, dict(id="dci", sym="dc,inf", val=0, key="d_nb", lvl=2))
    if yt == 0:
        S.tag(-R.b_P / 2, -0.55 * fs, dict(id="dz", sym="Δz", val=0, key="d_top", lvl=2))
    S.poignees(xt + 0.5 * fs + S.largeur_poignees(txtG) / 2, (yb1 + ybl) / 2, "G", "n1_u", txtG, "bolts" in hot)
    nomS = "Poutre secondaire" if u.prof_S == PERSO else u.prof_S
    S.name("T", xE - S.tw(nomS) / 2, nomS)
    S.name("B", 0, "Poutre principale" if u.prof_P == PERSO else u.prof_P)
    return S.finish(s, None, "Élévation cotée de l'assemblage par plat d'âme")


# ----------------------------------------------------------------------- plan
def plan(R, opt):
    """Vue en plan (coupe au niveau des boulons) : âme porteuse, âme portée,
    PLAT d'un seul côté de l'âme, files de boulons."""
    u = R.u; xf = R.tw_P / 2; gh = N(u.g_h); x0 = xf + gh; ws = R.tw_S / 2
    xt = xf + R.b_p; xE = xt + 60; p2 = N(u.p2_u)
    Hh = ws + R.t_p + 26
    if opt.realiste:
        Hh = mx(Hh, R.b_S / 2 + 10)
    bb = dict(x1=-xf - 30, x2=xE, y1=-Hh, y2=Hh)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 210) / 32, 8)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(k):
        return ["hot"] if k in hot else []

    def piece(grp, prims, **attrs):
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    if opt.realiste:
        xsem = mn(R.b_P / 2, xE - 4)
        s.append(_path(["hd"], [[(xsem, -Hh + 2), (xsem, Hh - 2)]]))
        s.append(_path(["hd"], [[(x0, -R.b_S / 2), (xE, -R.b_S / 2)], [(x0, R.b_S / 2), (xE, R.b_S / 2)]]))
    if opt.realiste:
        ptsP = _rect_rompu(-xf, -Hh, R.tw_P, 2 * Hh, ("T", "B"), 0.55 * fs)
        ptsS = _rect_rompu(x0, -ws, xE - x0, R.tw_S, ("R",), 0.55 * fs)
        piece("beamP", [_poly(["pp"], ptsP), _path(["ht"], _hachures(ptsP))])
        piece("beamS", [_poly(["ps"] + hc("beamS"), ptsS), _path(["ht"], _hachures(ptsS))],
              drag="g_h", dsym="gh")
    else:
        piece("beamP", [_rect(["pp"], -xf, -Hh, R.tw_P, 2 * Hh)])
        piece("beamS", [_rect(["ps"] + hc("beamS"), x0, -ws, xE - x0, R.tw_S)], drag="g_h", dsym="gh")
    # plat d'un seul côté de l'âme (coupé par le plan : hachures) + cordons
    y1 = -ws - R.t_p
    pts_pl = [(xf, y1), (xt, y1), (xt, -ws), (xf, -ws)]
    pl = [_poly(["co"] + hc("plate"), [list(pt) for pt in pts_pl])]
    if opt.realiste:
        pl.append(_path(["ht", "htc"], _hachures(pts_pl)))
        z = N(u.a_w) * math.sqrt(2)
        pl.append(_poly(["wb"], [[xf, y1], [xf + z, y1], [xf, y1 - z]]))
        pl.append(_poly(["wb"], [[xf, -ws], [xf + z, -ws], [xf, -ws + z]]))
    piece("plate", pl)
    # files de boulons (axes verticaux à travers plat + âme)
    xc1 = xf + R.g_B
    cls_s = ["ax"] if opt.realiste else ["bp"]
    for j in range(R.n_2):
        piece("bolts", [_path(cls_s + hc("bolts"), [[(xc1 + j * p2, y1 - 14), (xc1 + j * p2, ws + 14)]])])
    # cotes
    S.dim(dict(id="e2b", side="T", a=x0, b=xc1, o1=-ws, o2=y1 - 14, sym="e2,b", val=R.e2b, key="e2b_u", lvl=2))
    if R.n_2 > 1:
        S.dim(dict(id="p2", side="T", a=xc1, b=xc1 + (R.n_2 - 1) * p2, o1=y1 - 14, o2=y1 - 14, sym="p2", val=p2, key="p2_u", lvl=2))
    S.dim(dict(id="gh", side="T", a=xf, b=x0, o1=y1, o2=-ws, sym="gh", val=gh, key="g_h", lvl=1, out="lo"))
    S.dim(dict(id="bp", side="T", a=xf, b=xt, o1=y1, o2=y1, sym="bp", val=R.b_p, lvl=1, calc=1))
    S.dim(dict(id="tp", side="R", a=y1, b=-ws, o1=xt, o2=xt, sym="tp", val=R.t_p, key="tp_u", lvl=1))
    S.dim(dict(id="twS", side="R", a=-ws, b=ws, o1=xE, o2=xE, sym="tw", val=R.tw_S, lvl=2, calc=1))
    return S.finish(s, None, "Vue en plan cotée de l'assemblage par plat d'âme")


# ----------------------------------------------------------------- vue droite
def vue_droite(R, opt):
    """Vue de droite (regard le long de la portée) : face de l'âme porteuse
    (rompue des deux côtés), plat PAR LA TRANCHE avec ses deux cordons,
    rangées de boulons vues de côté, coupe de la portée par-devant."""
    u = R.u; ws = R.tw_S / 2; yt = N(u.d_top); zc = N(u.z_C); yc = yt + zc
    aw = N(u.a_w)
    Wp = mx(ws + R.t_p + aw + 24, R.b_S / 2 + 6) + 40
    bb = dict(x1=-Wp, x2=Wp, y1=mn(0, yt), y2=mx(R.h_P, yt + R.h_S))
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 240) / 34, 9)
    S = Feuille(bb, fs, opt); s = []
    s.append(_poly(["pp"], _rect_rompu(-Wp, 0, 2 * Wp, R.h_P, ("L", "R"), 0.55 * fs)))
    s.append(_path(["fl2"], [[(-Wp, R.tf_P), (Wp, R.tf_P)], [(-Wp, R.h_P - R.tf_P), (Wp, R.h_P - R.tf_P)]]))
    if R.r_P > 0:
        s.append(_path(["flr"], [[(-Wp, R.tf_P + R.r_P), (Wp, R.tf_P + R.r_P)],
                                 [(-Wp, R.h_P - R.tf_P - R.r_P), (Wp, R.h_P - R.tf_P - R.r_P)]]))
    # plat par la tranche (un seul côté de l'âme portée) + cordons visibles
    xpl = ws                                        # face du plat côté âme
    s.append(_rect(["co"], xpl, yc, R.t_p, R.h_p))
    s.append(_rect(["wb"], xpl - aw, yc, aw, R.h_p))
    s.append(_rect(["wb"], xpl + R.t_p, yc, aw, R.h_p))
    # coupe de la portée par-devant (profil complet, au-delà du grugeage)
    sec = _section_portee(R, yt)
    s.append(_poly(["ps"], sec))
    s.append(_path(["ht"], _hachures(sec)))
    # rangées de boulons vues de côté — par convention, les traits d'axe
    # traversent tout : ils passent au-dessus de la coupe
    yb1 = yc + R.e_1
    dep = R.d_0 / 2 + 5
    ax = [[(xpl - dep - aw, yb1 + i * R.p_1), (xpl + R.t_p + aw + dep, yb1 + i * R.p_1)]
          for i in range(R.n_1)]
    s.append(_path(["ax"], ax))
    if opt.fabrication:
        _renvoi(S, s, xpl + R.t_p + aw, yb1, xpl + R.t_p + aw + 2.2 * fs, yb1 - 1.6 * fs,
                js_str(R.n_1 * R.n_2) + "×Ø" + f0(R.d_0))
    # cotes de fabrication propres à cette vue
    S.dim(dict(id="ztp", side="L", a=0, b=yc, o1=-Wp, o2=xpl - aw, sym="zt", val=R.zt_pl, lvl=1, calc=1))
    S.dim(dict(id="hp", side="R", a=yc, b=yc + R.h_p, o1=xpl + R.t_p, o2=xpl + R.t_p, sym="hp", val=R.h_p,
               key="hp_u", lvl=1))
    S.dim(dict(id="tp", side="T", a=xpl, b=xpl + R.t_p, o1=yc, o2=yc, sym="tp", val=R.t_p, key="tp_u", lvl=1))
    S.dim(dict(id="bS", side="B", a=-R.b_S / 2, b=R.b_S / 2, o1=yt + R.h_S, o2=yt + R.h_S, sym="bS",
               val=R.b_S, lvl=1, calc=1))
    S.name("T", 0, "Poutre principale" if u.prof_P == PERSO else u.prof_P)
    return S.finish(s, None, "Vue de droite : plat par la tranche et coupe de la portée")


def dessins(R, opt):
    return elevation(R, opt), plan(R, opt)


def cotes_modifiables(dessin):
    """Les cotes cliquables (clé, symbole, id) dans l'ordre de pose."""
    out = []

    def rec(prims):
        for p in prims:
            if p.get("t") == "g":
                if p.get("key"):
                    out.append((p["key"], p.get("sym") or "", p.get("id") or ""))
                rec(p.get("enfants") or [])

    rec(dessin.cotes)
    return out

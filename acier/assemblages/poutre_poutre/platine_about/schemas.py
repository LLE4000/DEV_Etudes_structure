# -*- coding: utf-8 -*-
"""Dessins de l'assemblage par platine d'about de hauteur partielle —
élévation, vue en plan, vue de droite. Le KIT de dessin (primitives, Feuille
de cotation, styles, rendu SVG, lignes de rupture, hachures) est celui du
module doubles cornières, réutilisé tel quel ; seules les SCÈNES sont propres
à cet assemblage.

Repère : x le long de la poutre portée (0 = axe de l'âme porteuse), y vers le
bas (0 = dessus de la porteuse) en élévation et en vue de droite ; en plan, y
transversal (0 = axe de l'âme portée).

La boulonnerie (tête, écrou, rondelles) est dessinée en PROPORTIONS du
diamètre — un dessin de principe, pas une cote : k ≈ 0,63·d, m ≈ 0,9·d,
rondelle ≈ max(2,5 ; 0,15·d) d'épaisseur et dw de diamètre.
"""
import math

from acier.js import N, mn, mx, js_str
from acier.formats import f0
from acier.bibliotheques import PERSO
from ..doubles_cornieres.schemas import (            # kit partagé
    Options, Feuille, INF_MM,
    _rect, _poly, _path, _circle, _text, _group, _arc, _hachures,
    _rect_rompu, _rupture, _renvoi, _section_I, _section_portee)
from .entrees import PILOTEES_PAR_PREDIM

# Cote du dessin → clé d'entrée (édition directe sur le dessin)
CLE_PAR_COTE = {"hp": "hp_u", "zp": "z_C", "e1": "e1_u", "p1": "p1_u",
                "c": "l_n", "dcs": "d_nt", "dci": "d_nb", "dz": "d_top",
                "aw": "a_w", "tp": "tp_u", "p3": "p3_u", "bp": "bp_u"}

GROUPE_BOULONS = dict(n1="n1_u", titre="Groupe de boulons – platine et âme porteuse")

# Plan de principe : grandeurs calculées admises, et répartition des cotes
# (jamais deux fois la même cote sur la planche)
FABRICATION_CALC = frozenset({"ztp", "bS", "e2p", "twS"})
EXCLURE_ELEVATION = frozenset({"ztp", "bS", "twS", "tp", "p3", "bp", "e2p"})
EXCLURE_PLAN = frozenset({"p3", "bp", "e2p", "dz", "ztp", "bS"})
EXCLURE_DROITE = frozenset({"e1", "p1", "zp", "hp", "c", "dcs", "dci", "dz",
                            "aw", "twS", "tp"})


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


def visserie(d):
    """Proportions de dessin de la boulonnerie (mm) : hauteur de tête k,
    hauteur d'écrou m, épaisseur de rondelle tr — dessin de principe."""
    return 0.63 * d, 0.9 * d, mx(2.5, 0.15 * d)


def _boulon_profil(cls_t, x_ap, x_av, y, d, dw, r0):
    """Un boulon vu de profil, axe horizontal à la hauteur ``y`` : rondelle
    et écrou à gauche de ``x_ap`` (face arrière de l'âme porteuse), rondelle
    et tête à droite de ``x_av`` (face de la platine côté portée)."""
    k, m, tr = visserie(d)
    s = 1.5 * d                      # surplat (proportion)
    return [_rect(["bw"], x_ap - tr, y - dw / 2, tr, dw),
            _rect(cls_t, x_ap - tr - m, y - s / 2, m, s),
            _rect(["bw"], x_av, y - dw / 2, tr, dw),
            _rect(cls_t, x_av + tr, y - s / 2, k, s)]


# ------------------------------------------------------------------ élévation
def elevation(R, opt):
    """Élévation : porteuse en coupe, platine PAR LA TRANCHE contre son âme,
    portée soudée sur la platine (grugée le cas échéant, glisser Δz),
    boulons de profil (tête côté platine, écrou côté porteuse), cordons."""
    u = R.u; xf = R.tw_P / 2; tp = R.t_p; x0 = xf + tp
    yt = N(u.d_top); yb = yt + R.h_S
    zc = N(u.z_C); yc = yt + zc; dnt = N(u.d_nt); dnb = N(u.d_nb); ln = N(u.l_n)
    notch = dnt > 0 or dnb > 0
    k, m, tr = visserie(R.d_b)
    xtete = x0 + tr + k                              # bout des têtes
    # longueur de portée montrée : proportionnée à sa hauteur (lisible)
    xE = mx(x0 + (ln if notch else 0), xtete) + mx(85, 0.35 * R.h_S)
    bb = dict(x1=-R.b_P / 2, x2=xE, y1=mn(0, yt), y2=mx(R.h_P, yb))
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 260) / 34, 9)
    txtG = js_str(R.n_1) + " × 2"
    yb1 = yc + R.e_1; ybl = yb1 + (R.n_1 - 1) * R.p_1
    if opt.poignees:
        S0 = Feuille(bb, fs, opt)
        bb["x2"] = mx(bb["x2"], xtete + 1.2 * fs + S0.largeur_poignees(txtG) + 0.3 * fs)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(cle):
        return ["hot"] if cle in hot else []

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
    # portée grugée, soudée sur la platine, bord droit rompu
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
    piece("beamS", pS, drag2="d_top", dsym2="Δz")
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
    # PLATINE par la tranche (contre l'âme porteuse) + cordon vu de côté
    # (bande a·√2 le long de l'âme portée) + symbole de cordon double
    z = N(u.a_w) * math.sqrt(2)
    pl = [_rect(["co"] + hc("plate"), xf, yc, tp, R.h_p)]
    wl = []
    if opt.realiste:
        wl.append(_rect(["wb"] + hc("weld"), x0, yc, z, R.h_p))
        # symbole EN 22553 : flèche au cordon, coude SOUS la platine (la
        # ligne de référence ne traverse ni la platine ni les boulons)
        ax_, ay = x0 + z, yc + 0.88 * R.h_p
        ex, ey = ax_ + 2.6 * fs, yc + R.h_p + 1.9 * fs
        wt = S.tw("a " + f0(N(u.a_w)))
        tx = ex + wt + 0.9 * fs
        wl.append(_path(["wsy"], [[(ax_, ay), (ex, ey), (tx + 1.6 * fs, ey)],
                                  [(ax_ + 0.55 * fs, ay + 0.1 * fs), (ax_, ay), (ax_ + 0.1 * fs, ay + 0.55 * fs)]]))
        # cordon DOUBLE (de part et d'autre de l'âme) : triangle au-dessus ET
        # en dessous de la ligne de référence — EN 22553
        wl.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey - 0.95 * fs]]))
        wl.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey + 0.95 * fs]]))
        S.tag(ex - 0.1 * fs, ey - 0.55 * fs, dict(id="aw", sym="a", val=N(u.a_w), key="a_w", lvl=1))
    piece("plate", pl)
    if wl:
        piece("weld", wl)
    # boulons de profil : tête et rondelle côté platine, écrou et rondelle
    # côté porteuse ; traits d'axe à travers l'âme et la platine
    r0 = R.d_0 / 2
    bS = []
    for i in range(R.n_1):
        y = yb1 + i * R.p_1
        if opt.realiste:
            bS += _boulon_profil(["bo"] + hc("bolts"), -xf, x0, y, R.d_b, R.d_w, r0)
    dep = 6
    ax = [[(-xf - tr - m - dep, yb1 + i * R.p_1), (xtete + dep, yb1 + i * R.p_1)] for i in range(R.n_1)]
    bS.append(_path(["ax"] + hc("bolts"), ax))
    piece("bolts", bS)
    if opt.fabrication:
        _renvoi(S, s, xtete, yb1 - 0.3 * R.d_w, xtete + 1.6 * fs, yb1 - 1.9 * fs,
                js_str(2 * R.n_1) + "×Ø" + f0(R.d_0))
    if opt.poignees:
        autres = N(u.N_Ed) != 0
        txtF = "VEd " + f0(N(u.V_Ed)) + (" +" if autres else "")
        wF = S.tw(txtF) + 0.5 * fs
        xF = (mx(x0 + (ln if notch else 0), xtete) + xE) / 2
        hitF = _rect(["hit"], -wF / 2, -1.05 * fs, wF, 1.42 * fs); hitF["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitF, _text([], None, None, txtF, fs)],
                        x=xF, y=mx(yt + dnt, yc) + 1.7 * fs, rot=0, grp="efforts", id="efforts"))
    # --- cotes
    S.dim(dict(id="tp", side="T", a=xf, b=x0, o1=yc, o2=yc, sym="tp", val=tp, key="tp_u", lvl=2, out="lo"))
    if dnt > 0:
        S.dim(dict(id="c", side="T", a=x0, b=x0 + ln, o1=yt + dnt, o2=yt, sym="c", val=ln, key="l_n", lvl=1))
    elif dnb > 0:
        S.dim(dict(id="c", side="B", a=x0, b=x0 + ln, o1=yb - dnb, o2=yb, sym="c", val=ln, key="l_n", lvl=1))
    S.dim(dict(id="e1", side="R", a=yc, b=yb1, o1=x0 + z, o2=xtete, sym="e1", val=R.e_1, key="e1_u", lvl=1, out="lo"))
    for i in range(1, R.n_1):
        S.dim(dict(id="p1", side="R", a=yb1 + (i - 1) * R.p_1, b=yb1 + i * R.p_1, o1=xtete, o2=xtete,
                   sym="p1", val=R.p_1, key="p1_u", lvl=1))
    S.dim(dict(id="e1bot", side="R", a=ybl, b=yc + R.h_p, o1=xtete, o2=x0 + z, sym="e1'", val=R.e1bot, lvl=2, calc=1))
    S.dim(dict(id="hp", side="R", a=yc, b=yc + R.h_p, o1=x0 + z, o2=x0 + z, sym="hp", val=R.h_p, key="hp_u", lvl=1))
    S.dim(dict(id="zp", side="R", a=yt, b=yc, o1=xE, o2=x0 + z, sym="zp", val=zc, key="z_C", lvl=1, out="lo"))
    if dnt > 0:
        S.dim(dict(id="dcs", side="R", a=yt, b=yt + dnt, o1=x0 + ln, o2=x0 + ln, sym="dc,sup", val=dnt, key="d_nt", lvl=1, out="lo"))
    if dnb > 0:
        S.dim(dict(id="dci", side="R", a=yb - dnb, b=yb, o1=x0 + ln, o2=x0 + ln, sym="dc,inf", val=dnb, key="d_nb", lvl=1))
    S.dim(dict(id="dz", side="L", a=0, b=yt, o1=-R.b_P / 2, o2=x0, sym="Δz", val=yt, key="d_top", lvl=2, out="lo"))
    # étiquettes des cotes NULLES : poignées d'édition à l'écran (créer un
    # grugeage, un décalage) — sans objet sur le plan de fabrication
    if not opt.fabrication:
        if dnt == 0:
            S.tag(xE - 6 * fs, yt + R.tf_S + 1.5 * fs, dict(id="dcs", sym="dc,sup", val=0, key="d_nt", lvl=2))
        if dnb == 0:
            S.tag(xE - 6 * fs, yb - R.tf_S - 0.7 * fs, dict(id="dci", sym="dc,inf", val=0, key="d_nb", lvl=2))
        if yt == 0:
            S.tag(-R.b_P / 2, -0.55 * fs, dict(id="dz", sym="Δz", val=0, key="d_top", lvl=2))
    S.poignees(xtete + 1.2 * fs + S.largeur_poignees(txtG) / 2, (yb1 + ybl) / 2, "G", "n1_u", txtG, "bolts" in hot)
    nomS = "Poutre secondaire" if u.prof_S == PERSO else u.prof_S
    S.name("T", xE - S.tw(nomS) / 2, nomS)
    S.name("B", 0, "Poutre principale" if u.prof_P == PERSO else u.prof_P)
    return S.finish(s, None, "Élévation cotée de l'assemblage par platine d'about")


# ----------------------------------------------------------------------- plan
def plan(R, opt):
    """Vue en plan (coupe au niveau d'une rangée de boulons) : âme porteuse,
    platine coupée, âme portée et ses deux cordons, les deux files de
    boulons (entraxe p3) de part et d'autre de l'âme portée."""
    u = R.u; xf = R.tw_P / 2; tp = R.t_p; x0 = xf + tp; ws = R.tw_S / 2
    k, m, tr = visserie(R.d_b)
    xtete = x0 + tr + k; xecrou = -xf - tr - m
    hb = R.b_p / 2; y3 = R.p_3 / 2
    Hh = hb + 24
    if opt.realiste:
        Hh = mx(Hh, R.b_S / 2 + 10)
    # à l'écran, vue au moins 1,3 fois plus large que haute : affichée en
    # pleine largeur de colonne, une vue étroite deviendrait une bande très
    # haute ; sur la note, la vue reste compacte (échelle du plan de principe)
    xE = mx(xtete + 70, xecrou - 24 + 1.3 * 2 * Hh) if opt.interactive else xtete + 70
    bb = dict(x1=xecrou - 24, x2=xE, y1=-Hh, y2=Hh)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 210) / 32, 8)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(cle):
        return ["hot"] if cle in hot else []

    def piece(grp, prims, **attrs):
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    if opt.realiste:
        # arêtes cachées : bord de la semelle porteuse et bords de la semelle
        # inférieure de la portée — seulement s'ils tombent dans le champ
        if R.b_P / 2 < xE - 4:
            s.append(_path(["hd"], [[(R.b_P / 2, -Hh + 2), (R.b_P / 2, Hh - 2)]]))
        x_sem_S = x0 + (N(u.l_n) if N(u.d_nb) > 0 else 0)
        if x_sem_S < xE - 4:
            s.append(_path(["hd"], [[(x_sem_S, -R.b_S / 2), (xE, -R.b_S / 2)],
                                    [(x_sem_S, R.b_S / 2), (xE, R.b_S / 2)]]))
        ptsP = _rect_rompu(-xf, -Hh, R.tw_P, 2 * Hh, ("T", "B"), 0.55 * fs)
        ptsS = _rect_rompu(x0, -ws, xE - x0, R.tw_S, ("R",), 0.55 * fs)
        piece("beamP", [_poly(["pp"], ptsP), _path(["ht"], _hachures(ptsP))])
        piece("beamS", [_poly(["ps"] + hc("beamS"), ptsS), _path(["ht"], _hachures(ptsS))])
    else:
        piece("beamP", [_rect(["pp"], -xf, -Hh, R.tw_P, 2 * Hh)])
        piece("beamS", [_rect(["ps"] + hc("beamS"), x0, -ws, xE - x0, R.tw_S)])
    # platine coupée par le plan (hachures)
    pts_pl = [(xf, -hb), (x0, -hb), (x0, hb), (xf, hb)]
    pl = [_poly(["co"] + hc("plate"), [list(pt) for pt in pts_pl])]
    if opt.realiste:
        pl.append(_path(["ht", "htc"], _hachures(pts_pl)))
    piece("plate", pl)
    if opt.realiste:
        z = N(u.a_w) * math.sqrt(2)
        piece("weld", [_poly(["wb"] + hc("weld"), [[x0, -ws], [x0 + z, -ws], [x0, -ws - z]]),
                       _poly(["wb"] + hc("weld"), [[x0, ws], [x0 + z, ws], [x0, ws + z]])])
    # deux files de boulons : têtes côté platine, écrous côté porteuse
    bS = []
    for y in (-y3, y3):
        if opt.realiste:
            bS += _boulon_profil(["bo"] + hc("bolts"), -xf, x0, y, R.d_b, R.d_w, R.d_0 / 2)
    bS.append(_path((["ax"] if opt.realiste else ["bp"]) + hc("bolts"),
                    [[(xecrou - 6, y), (xtete + 6, y)] for y in (-y3, y3)]))
    piece("bolts", bS)
    # cotes
    # p3, e2 et bp à gauche (rappels courts depuis les écrous et la platine)
    S.dim(dict(id="p3", side="L", a=-y3, b=y3, o1=xecrou, o2=xecrou, sym="p3", val=R.p_3, key="p3_u", lvl=1))
    S.dim(dict(id="e2p", side="L", a=y3, b=hb, o1=xecrou, o2=xf, sym="e2", val=R.e_2, lvl=2, calc=1))
    S.dim(dict(id="bp", side="L", a=-hb, b=hb, o1=xf, o2=xf, sym="bp", val=R.b_p, key="bp_u", lvl=1))
    S.dim(dict(id="tp", side="T", a=xf, b=x0, o1=-hb, o2=-hb, sym="tp", val=tp, key="tp_u", lvl=1, out="lo"))
    S.dim(dict(id="twS", side="R", a=-ws, b=ws, o1=xE, o2=xE, sym="tw", val=R.tw_S, lvl=2, calc=1))
    return S.finish(s, None, "Vue en plan cotée de l'assemblage par platine d'about")


# ----------------------------------------------------------------- vue droite
def vue_droite(R, opt):
    """Vue de droite (regard le long de la portée, vers l'âme porteuse) :
    face de l'âme porteuse (rompue des deux côtés), platine en VRAIE
    GRANDEUR avec ses deux files de boulons, cordons de part et d'autre de
    l'âme portée, coupe de la portée par-devant (au-delà du grugeage)."""
    u = R.u; yt = N(u.d_top); zc = N(u.z_C); yc = yt + zc
    ws = R.tw_S / 2; hb = R.b_p / 2; y3 = R.p_3 / 2
    Wp = mx(hb, R.b_S / 2 + 6) + 40
    bb = dict(x1=-Wp, x2=Wp, y1=mn(0, yt), y2=mx(R.h_P, yt + R.h_S))
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 240) / 34, 9)
    S = Feuille(bb, fs, opt); s = []
    s.append(_poly(["pp"], _rect_rompu(-Wp, 0, 2 * Wp, R.h_P, ("L", "R"), 0.55 * fs)))
    s.append(_path(["fl2"], [[(-Wp, R.tf_P), (Wp, R.tf_P)], [(-Wp, R.h_P - R.tf_P), (Wp, R.h_P - R.tf_P)]]))
    if R.r_P > 0:
        s.append(_path(["flr"], [[(-Wp, R.tf_P + R.r_P), (Wp, R.tf_P + R.r_P)],
                                 [(-Wp, R.h_P - R.tf_P - R.r_P), (Wp, R.h_P - R.tf_P - R.r_P)]]))
    # platine en vraie grandeur
    s.append(_rect(["co"], -hb, yc, R.b_p, R.h_p))
    # cordons : jambe a·√2 sur la platine, de part et d'autre de l'âme
    z = N(u.a_w) * math.sqrt(2)
    s.append(_rect(["wb"], -ws - z, yc, z, R.h_p))
    s.append(_rect(["wb"], ws, yc, z, R.h_p))
    # boulons vus de face : rondelle, tête (trou) et traits d'axe
    yb1 = yc + R.e_1; ybl = yb1 + (R.n_1 - 1) * R.p_1; r0 = R.d_0 / 2
    for x in (-y3, y3):
        for i in range(R.n_1):
            s.append(_circle(["bw"], x, yb1 + i * R.p_1, R.d_w / 2))
            s.append(_circle(["bo"], x, yb1 + i * R.p_1, r0))
    # coupe de la portée par-devant (profil complet, au-delà du grugeage)
    sec = _section_portee(R, yt)
    s.append(_poly(["ps"], sec))
    s.append(_path(["ht"], _hachures(sec)))
    # traits d'axe : par convention, ils traversent tout
    dep = R.d_w / 2 + 5
    ax = [[(-y3 - dep, yb1 + i * R.p_1), (y3 + dep, yb1 + i * R.p_1)] for i in range(R.n_1)]
    ax += [[(x, yb1 - dep), (x, ybl + dep)] for x in (-y3, y3)]
    s.append(_path(["ax"], ax))
    if opt.fabrication:
        _renvoi(S, s, y3 + r0 * 0.72, ybl + r0 * 0.72, y3 + r0 + 1.6 * fs, ybl + 1.9 * fs,
                js_str(2 * R.n_1) + "×Ø" + f0(R.d_0))
    # cotes de fabrication propres à cette vue : perçage de l'âme porteuse
    # coté depuis SA référence (dessus de la porteuse)
    S.dim(dict(id="ztp", side="L", a=0, b=yc, o1=-Wp, o2=-hb, sym="zt", val=R.zt_pl, lvl=1, calc=1))
    S.dim(dict(id="e1", side="L", a=yc, b=yb1, o1=-hb, o2=-y3, sym="e1", val=R.e_1, key="e1_u", lvl=1, out="lo"))
    for i in range(1, R.n_1):
        S.dim(dict(id="p1", side="L", a=yb1 + (i - 1) * R.p_1, b=yb1 + i * R.p_1, o1=-y3, o2=-y3,
                   sym="p1", val=R.p_1, key="p1_u", lvl=1))
    S.dim(dict(id="hp", side="R", a=yc, b=yc + R.h_p, o1=hb, o2=hb, sym="hp", val=R.h_p, key="hp_u", lvl=1))
    S.dim(dict(id="p3", side="T", a=-y3, b=y3, o1=yb1, o2=yb1, sym="p3", val=R.p_3, key="p3_u", lvl=1))
    S.dim(dict(id="e2p", side="T", a=y3, b=hb, o1=yb1, o2=yc, sym="e2", val=R.e_2, lvl=2, calc=1))
    S.dim(dict(id="bp", side="B", a=-hb, b=hb, o1=yc + R.h_p, o2=yc + R.h_p, sym="bp", val=R.b_p, key="bp_u", lvl=1))
    S.dim(dict(id="bS", side="B", a=-R.b_S / 2, b=R.b_S / 2, o1=yt + R.h_S, o2=yt + R.h_S, sym="bS",
               val=R.b_S, lvl=1, calc=1))
    S.name("T", 0, "Poutre principale" if u.prof_P == PERSO else u.prof_P)
    return S.finish(s, None, "Vue de droite : platine en vraie grandeur et coupe de la portée")


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

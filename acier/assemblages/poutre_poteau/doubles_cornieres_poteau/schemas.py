# -*- coding: utf-8 -*-
"""Dessins de l'assemblage poutre–poteau par doubles cornières d'âme —
élévation, VUE EN PLAN et coupe A-A. Le KIT de dessin (primitives, Feuille de
cotation, styles, rendu SVG, lignes de rupture, hachures, section de
cornière) est celui du module doubles cornières poutre–poutre, réutilisé tel
quel ; seules les SCÈNES sont propres à cet assemblage.

**La VUE EN PLAN est la vue PORTEUSE** (`POUTRE_POTEAU.md` §10.7) : c'est
elle qui montre le poteau EN COUPE, le cas d'appui, les DEUX cornières de
part et d'autre de l'âme portée, le gauge `p3` symétrique par rapport à
l'âme du poteau, la pince au bout d'aile (cas a) ou la largeur droite entre
congés (cas b), le passage de la poutre entre les ailes, la recoupe des
semelles et les jeux latéraux.

Repère des trois vues, pour les deux cas d'appui (étude §1.2) :

===============  =========================  =========================
 vue              cas (a) — sur l'AILE       cas (b) — dans l'ÂME
===============  =========================  =========================
 élévation        regard ∥ aux ailes :       regard ⊥ aux ailes :
                  bande de largeur hP,       bande de largeur bP,
                  ailes en bandes de tfP     âme repérée à ±twP/2
 vue en plan      poteau en coupe, ailes     poteau en coupe, ailes
                  VERTICALES (X = ±hP/2),    HORIZONTALES (y = ±hP/2),
                  cornières sur la face      poutre ENTRE les ailes,
                  extérieure de l'aile       largeur droite de l'âme
 coupe A-A        face extérieure de         intérieur du poteau :
                  l'aile (largeur bP) et     âme en face, ailes par la
                  les deux ailes A en        tranche, et les deux
                  vraie grandeur             ailes A en vraie grandeur
===============  =========================  =========================

Un poteau continue AU-DESSUS et EN DESSOUS de l'attache : il est toujours
tracé en tronçon ROMPU (``_rect_rompu``), et la clé ``d_top`` du grand frère
(décalage des dessus) n'existe plus.

Cotes : symboles finaux directement (hc, zc, tc, e1, p1, e2,b, gA, p3, gh, c,
dc,sup, rc, j…) — pas de couche de notation.
"""
import math

from acier.js import N, mn, mx, js_str
from acier.formats import f0
from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # kit partagé
    Options, Feuille, INF_MM,
    _rect, _poly, _path, _circle, _text, _group, _arc, _hachures,
    _rect_rompu, _rupture, _renvoi, _section_portee, _corniere_plan)
from ..dessin import marge_poteau, nom_portee, nom_poteau, section_poteau
from .entrees import PILOTEES_PAR_PREDIM

# Cote du dessin → clé d'entrée (édition directe sur le dessin) : 20 cotes
CLE_PAR_COTE = {"e2b": "e2b_u", "p2S": "p2_S", "lhS": "lh_S", "gh": "g_h", "c": "l_n",
                "dcs": "d_nt", "dci": "d_nb", "e1S": "e1S_u", "p1S": "p1S_u",
                "Lc": "LC_u", "zc": "z_C", "e1P": "e1P_u", "p1P": "p1P_u",
                "gA": "gA_u", "p2P": "p2_P", "aP": "a_P", "lhP": "lh_P", "aS": "a_S",
                "rec": "rec_S", "jlat": "j_lat"}

# Plan de principe : grandeurs calculées admises, et répartition des cotes —
# chaque cote n'est portée qu'UNE fois sur la planche de la page 2. La
# largeur de la bande du poteau est cotée sur l'ÉLÉVATION (hP en cas a, bP
# en cas b) et l'AUTRE sur la VUE EN PLAN : les deux ids existent donc une
# seule fois, quel que soit le cas d'appui.
FABRICATION_CALC = frozenset({"bA", "bB", "p3", "bS", "tc", "twS", "hP", "bP",
                              "tfP", "twP", "libreP", "ecol", "eP"})
EXCLURE_ELEVATION = frozenset()
EXCLURE_PLAN = frozenset({"e2b", "p2S", "gh"})
EXCLURE_COUPE = frozenset({"Lc", "e1P", "p1P", "e1botP", "gA", "p2P", "p3"})


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
    """Élévation (regard ⊥ à l'axe de la poutre portée) : le poteau vu de
    côté — bande rompue en haut et en bas, ailes (cas a) ou âme (cas b)
    repérées —, la poutre portée vue de côté (grugeage, glissement gh), les
    CORNIÈRES par-devant (aile B de face, aile A par la tranche), le groupe S
    de face, le groupe P en traits d'axe, et le symbole de soudure EN 22553
    quand les ailes B sont soudées."""
    u = R.u
    xf = R.xf_P                                  # face d'appui (axe du poteau → face)
    gh = N(u.g_h); x0 = xf + gh
    yt = 0.0; yb = R.h_S                         # un poteau n'a pas de dessus : Δz = 0
    zc = N(u.z_C); yc = yt + zc
    dnt = N(u.d_nt); dnb = N(u.d_nb); ln = N(u.l_n)
    p2S = N(u.p2_S); lhS = N(u.lh_S)
    larg = R.b_P if R.ame else R.h_P             # largeur de la bande du poteau
    mar = marge_poteau(R)
    notch = dnt > 0 or dnb > 0
    xt = xf + R.b_B                              # bout de l'aile B
    xE = mx(x0 + (ln if notch else 0), xt, larg / 2) + 75
    bb = dict(x1=-larg / 2, x2=xE, y1=yt - mar, y2=yb + mar)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 260) / 34, 9)
    txtS = js_str(R.n1_S) + " × " + js_str(R.n2_S); txtP = js_str(R.n1_P) + " × " + js_str(R.n2_P)
    if opt.poignees:
        S0 = Feuille(bb, fs, opt)
        if R.bolt_S:
            bb["x2"] = mx(bb["x2"], xt + 0.5 * fs + S0.largeur_poignees(txtS) + 0.3 * fs)
        if R.bolt_P:
            bb["x1"] = mn(bb["x1"], -larg / 2 - 0.5 * fs - S0.largeur_poignees(txtP) - 0.3 * fs)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(k):
        return ["hot"] if k in hot else []

    def piece(grp, prims, **attrs):
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    # --- POTEAU : tronçon vu de côté, rompu en haut et en bas
    pts = _rect_rompu(-larg / 2, yt - mar, larg, yb - yt + 2 * mar, ("T", "B"), 0.55 * fs) \
        if opt.realiste else [(-larg / 2, yt - mar), (larg / 2, yt - mar),
                              (larg / 2, yb + mar), (-larg / 2, yb + mar)]
    piece("col", [_poly(["pp"] + hc("col"), pts)])
    if R.ame:
        # l'âme du poteau est DERRIÈRE l'aile et derrière les cornières : ses
        # deux traits sont des lignes CACHÉES (tiretées), posées à la fin, par
        # dessus la poutre et les cornières — sans quoi le repère de l'âme,
        # qui est l'information du cas (b), disparaît sous la pièce d'attache
        web = [[(-R.tw_P / 2, yt - mar), (-R.tw_P / 2, yb + mar)],
               [(R.tw_P / 2, yt - mar), (R.tw_P / 2, yb + mar)]]
    else:                          # les deux ailes, et les congés
        web = None
        xi = larg / 2 - R.tf_P
        s.append(_path(["fl2"] + hc("colFl"), [[(-xi, yt - mar), (-xi, yb + mar)],
                                               [(xi, yt - mar), (xi, yb + mar)]]))
        if opt.realiste and R.r_P > 0:
            xcg = xi - R.r_P
            s.append(_path(["flr"], [[(-xcg, yt - mar), (-xcg, yb + mar)],
                                     [(xcg, yt - mar), (xcg, yb + mar)]]))
    # --- POUTRE PORTÉE grugée, bord droit rompu
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
    piece("beamS", pS, drag="g_h", dsym="gh")
    if "notchT" in hot:
        s.append(_path(["hl"], [[(x0, yt + dnt), (x0 + ln, yt + dnt), (x0 + ln, yt)]] if dnt > 0
                       else [[(x0, yt + R.tf_S + R.r_S), (x0, yt), (mx(larg / 2, x0 + 40), yt)]]))
    if "notchB" in hot:
        s.append(_path(["hl"], [[(x0, yb - dnb), (x0 + ln, yb - dnb), (x0 + ln, yb)]] if dnb > 0
                       else [[(x0, yb - R.tf_S - R.r_S), (x0, yb), (mx(larg / 2, x0 + 40), yb)]]))
    if opt.fabrication and rn > 0 and dnt > 0:
        cxg, cyg = x0 + ln - rn, yt + dnt - rn
        _renvoi(S, s, cxg + rn * 0.707, cyg + rn * 0.707, cxg - 2.4 * fs, yt + dnt - rn - 0.9 * fs,
                "r " + f0(rn))
    # --- CORNIÈRES par-devant : aile B de face, aile A par la tranche
    piece("cleat", [_rect(["co"] + hc("cleat"), xf, yc, R.b_B, R.L_C),
                    _rect(["co2"], xf, yc, R.t_C, R.L_C)])
    xc1 = xf + R.g_B2; xcl = xc1 + (R.n2_S - 1) * p2S; yb1 = yc + R.e1_S
    ybl = yb1 + (R.n1_S - 1) * R.p1_S; r0 = R.d_0 / 2
    if R.bolt_S:
        bS = []
        for i in range(R.n1_S):
            for j in range(R.n2_S):
                cx = xc1 + j * p2S; cy = yb1 + i * R.p1_S
                if opt.realiste:
                    bS.append(_circle(["bw"], cx, cy, R.d_w / 2))
                bS.append(_circle(["bo"] + hc("boltsS"), cx, cy, r0))
                if not opt.realiste:
                    bS.append(_path(["cm"], [[(cx - r0 - 4, cy), (cx + r0 + 4, cy)],
                                             [(cx, cy - r0 - 4), (cx, cy + r0 + 4)]]))
        if opt.realiste:
            dep = R.d_w / 2 + 5
            ax = [[(xc1 - dep, yb1 + i * R.p1_S), (xcl + dep, yb1 + i * R.p1_S)] for i in range(R.n1_S)]
            ax += [[(xc1 + j * p2S, yb1 - dep), (xc1 + j * p2S, ybl + dep)] for j in range(R.n2_S)]
            bS.append(_path(["ax"], ax))
        piece("bolts", bS)
        if opt.fabrication:
            _renvoi(S, s, xcl + R.d_w / 2 * 0.72, yb1 - R.d_w / 2 * 0.72,
                    xcl + r0 + 1.9 * fs, yb1 - 1.7 * fs,
                    js_str(R.n1_S * R.n2_S) + "×Ø" + f0(R.d_0))
    elif opt.realiste:
        # cordons d'angle vus de face : bande de largeur a·√2 le long du bout
        # de l'aile B, retours en haut et en bas ; symbole EN 22553
        z = N(u.a_S) * math.sqrt(2)
        wS = [_rect(["wb"], xt, yc, z, R.L_C)]
        if lhS > 0:
            wS += [_rect(["wb"], xt - lhS, yc - z, lhS + z, z),
                   _rect(["wb"], xt - lhS, yc + R.L_C, lhS + z, z)]
        ax_, ay = xt + z, yc + 0.78 * R.L_C
        ex, ey = ax_ + 1.7 * fs, ay + 1.5 * fs
        wt = S.tw("a " + f0(N(u.a_S)))
        tx = ex + wt + 0.9 * fs
        wS.append(_path(["wsy"], [[(ax_, ay), (ex, ey), (tx + 1.6 * fs, ey)],
                                  [(ax_ + 0.55 * fs, ay + 0.1 * fs), (ax_, ay), (ax_ + 0.1 * fs, ay + 0.55 * fs)]]))
        wS.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey - 0.95 * fs]]))
        piece("weldS", wS)
        S.tag(ex - 0.1 * fs, ey - 0.55 * fs, dict(id="aS", sym="a", val=N(u.a_S), key="a_S", lvl=1))
    else:
        piece("weldS", [_path(["we"], [[(xt - lhS, yc), (xt, yc), (xt, yc + R.L_C), (xt - lhS, yc + R.L_C)]])])
    # --- groupe P : boulons perpendiculaires au plan du dessin (les deux
    # --- files à ±p3/2 se superposent) → traits d'axe horizontaux
    yp1 = yc + R.e1_P; ypl = yp1 + (R.n1_P - 1) * R.p1_P
    xgP = -larg / 2 - 16
    if R.bolt_P:
        cls_p = ["ax"] if opt.realiste else ["bp"]
        piece("bolts", [_path(cls_p + hc("boltsP"),
                              [[(xgP, yp1 + i * R.p1_P), (xf + R.t_C + 16, yp1 + i * R.p1_P)]
                               for i in range(R.n1_P)])])
    if web:                        # l'âme du poteau, en lignes cachées, par-dessus
        s.append(_path(["hd"] + hc("colWeb"), web))
    if opt.poignees:
        autres = any(N(u[k]) != 0 for k in ("N_Ed", "H_Ed", "M_Ed"))
        txtF = "VEd " + f0(N(u.V_Ed)) + (" +" if autres else "")
        wF = S.tw(txtF) + 0.5 * fs
        xF = (mx(x0 + (ln if notch else 0), xt) + xE) / 2
        hitF = _rect(["hit"], -wF / 2, -1.05 * fs, wF, 1.42 * fs); hitF["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitF, _text([], None, None, txtF, fs)],
                        x=xF, y=mx(yt + dnt, yc) + 1.7 * fs, rot=0, grp="efforts", id="efforts"))
    # --- cotes
    if R.bolt_S:
        S.dim(dict(id="e2b", side="T", a=x0, b=xc1, o1=yt + dnt, o2=yb1, sym="e2,b", val=R.e2b_S, key="e2b_u", lvl=1))
        if R.n2_S > 1:
            S.dim(dict(id="p2S", side="T", a=xc1, b=xcl, o1=yb1, o2=yb1, sym="p2", val=p2S, key="p2_S", lvl=1))
        S.dim(dict(id="e2B", side="T", a=xcl, b=xt, o1=yb1, o2=yc, sym="e2", val=R.e2a_S, lvl=2, calc=1))
    elif lhS > 0:
        S.dim(dict(id="lhS", side="T", a=xt - lhS, b=xt, o1=yc, o2=yc, sym="ℓh", val=lhS, key="lh_S", lvl=1))
    S.dim(dict(id="gh", side="T", a=xf, b=x0, o1=yc, o2=yt + dnt, sym="gh", val=gh, key="g_h", lvl=2, out="lo"))
    S.dim(dict(id="z", side="T", a=xf, b=xf + R.zeff, o1=yc, o2=yb1 if R.bolt_S else yc, sym="z",
               val=R.zeff, lvl=1, calc=1))
    if dnt > 0:
        S.dim(dict(id="c", side="T", a=x0, b=x0 + ln, o1=yt + dnt, o2=yt, sym="c", val=ln, key="l_n", lvl=1))
    elif dnb > 0:
        S.dim(dict(id="c", side="B", a=x0, b=x0 + ln, o1=yb - dnb, o2=yb, sym="c", val=ln, key="l_n", lvl=1))
    # largeur de la bande du poteau : hP en cas (a), bP en cas (b) — l'autre
    # est cotée sur la vue en plan (jamais deux fois sur la planche)
    S.dim(dict(id="bP" if R.ame else "hP", side="B", a=-larg / 2, b=larg / 2,
               o1=yb + mar, o2=yb + mar, sym="b" if R.ame else "h", val=larg, lvl=1, calc=1))
    if R.bolt_S:
        S.dim(dict(id="e1S", side="R", a=yc, b=yb1, o1=xt, o2=xcl, sym="e1", val=R.e1_S, key="e1S_u", lvl=1, out="lo"))
        for i in range(1, R.n1_S):
            S.dim(dict(id="p1S", side="R", a=yb1 + (i - 1) * R.p1_S, b=yb1 + i * R.p1_S, o1=xcl, o2=xcl,
                       sym="p1", val=R.p1_S, key="p1S_u", lvl=1))
        S.dim(dict(id="e1botS", side="R", a=ybl, b=yc + R.L_C, o1=xcl, o2=xt, sym="e1'", val=R.e1bot_S, lvl=2, calc=1))
    S.dim(dict(id="Lc", side="R", a=yc, b=yc + R.L_C, o1=xt, o2=xt, sym="hc", val=R.L_C, key="LC_u", lvl=1))
    S.dim(dict(id="zc", side="R", a=yt, b=yc, o1=xE, o2=xt, sym="zc", val=zc, key="z_C", lvl=1, out="lo"))
    if dnt > 0:
        S.dim(dict(id="dcs", side="R", a=yt, b=yt + dnt, o1=x0 + ln, o2=x0 + ln, sym="dc,sup", val=dnt,
                   key="d_nt", lvl=1, out="lo"))
    if dnb > 0:
        S.dim(dict(id="dci", side="R", a=yb - dnb, b=yb, o1=x0 + ln, o2=x0 + ln, sym="dc,inf", val=dnb,
                   key="d_nb", lvl=1))
    if R.bolt_S:
        S.dim(dict(id="e1b", side="R", a=yt + dnt, b=yb1, o1=x0 + (ln if dnt > 0 else 0), o2=xcl,
                   sym="e1,b", val=R.e1b_S, lvl=2, calc=1, out="lo"))
        if dnb > 0:
            S.dim(dict(id="he", side="R", a=ybl, b=yb - dnb, o1=xcl, o2=x0 + ln, sym="he", val=R.h_e, lvl=2, calc=1))
    if R.bolt_P:
        lp = 2 if (R.bolt_S and R.n1_P == R.n1_S and R.p1_P == R.p1_S and R.e1_P == R.e1_S) else 1
        S.dim(dict(id="e1P", side="L", a=yc, b=yp1, o1=xf, o2=xgP, sym="e1", val=R.e1_P, key="e1P_u",
                   lvl=lp, out="lo"))
        for i in range(1, R.n1_P):
            S.dim(dict(id="p1P", side="L", a=yp1 + (i - 1) * R.p1_P, b=yp1 + i * R.p1_P, o1=xgP, o2=xgP,
                       sym="p1", val=R.p1_P, key="p1P_u", lvl=lp))
        S.dim(dict(id="e1botP", side="L", a=ypl, b=yc + R.L_C, o1=xgP, o2=xf, sym="e1'",
                   val=R.e1bot_P, lvl=2, calc=1))
    # paramètres nuls : pas de cote possible, donc une étiquette éditable
    if dnt == 0:
        S.tag(xt + 0.4 * fs, yt + R.tf_S + 1.5 * fs, dict(id="dcs", sym="dc,sup", val=0, key="d_nt", lvl=2))
    if dnb == 0:
        S.tag(xt + 0.4 * fs, yb - R.tf_S - 0.7 * fs, dict(id="dci", sym="dc,inf", val=0, key="d_nb", lvl=2))
    if not R.bolt_S:
        if not opt.realiste:
            S.tag(xt + 0.4 * fs, yc + R.L_C / 2, dict(id="aS", sym="a", val=N(u.a_S), key="a_S", lvl=1))
        if lhS == 0 and opt.editables:
            S.tag(xt + 0.4 * fs, yc + R.L_C / 2 + 1.6 * fs, dict(id="lhS", sym="ℓh", val=0, key="lh_S", lvl=1))
    if R.bolt_S:
        S.poignees(xt + 0.5 * fs + S.largeur_poignees(txtS) / 2, (yb1 + ybl) / 2, "S", "n1S_u", txtS,
                   "boltsS" in hot)
    if R.bolt_P:
        S.poignees(xgP - 0.5 * fs - S.largeur_poignees(txtP) / 2, (yp1 + ypl) / 2, "P", "n1P_u", txtP,
                   "boltsP" in hot)
    S.name("T", xE - S.tw(nom_portee(R)) / 2, nom_portee(R))
    S.name("B", 0, nom_poteau(R))
    return S.finish(s, None, "Élévation cotée de l'assemblage par doubles cornières sur poteau")


# ----------------------------------------------------------------------- plan
def plan(R, opt):
    """VUE EN PLAN (coupe horizontale au niveau des boulons) — la vue
    porteuse : LE POTEAU EN COUPE (hachuré), l'âme de la poutre portée en
    coupe, les DEUX CORNIÈRES de part et d'autre, le gauge p3 symétrique par
    rapport à l'âme du poteau. En cas (a) : la pince au bout d'aile. En
    cas (b) : la poutre arrive ENTRE LES AILES, avec la largeur droite entre
    congés, la recoupe des semelles et les jeux latéraux."""
    u = R.u
    xf = R.xf_P; gh = N(u.g_h); x0 = xf + gh; ws = R.tw_S / 2
    xt = xf + R.b_B; p2S = N(u.p2_S); p2P = N(u.p2_P)
    sec = section_poteau(R)
    xcol = mx(*[abs(px) for px, _ in sec]); ycol = mx(*[abs(py) for _, py in sec])
    xE = mx(xt, xcol) + 60
    rec = N(u.rec_S); brec = mn(R.b_S, R.libre_ailes) if rec > 0 else R.b_S
    Hh = mx(ws + R.b_A + 22, ycol + 10)
    if opt.realiste:
        Hh = mx(Hh, R.b_S / 2 + 10)
    bb = dict(x1=-xcol - 30, x2=xE, y1=-Hh, y2=Hh)
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

    # --- POTEAU EN COUPE (hachuré) : la pièce porteuse de l'information
    piece("col", [_poly(["pp"] + hc("col"), [list(p) for p in sec]),
                  _path(["ht"], _hachures(sec))])
    if "colFl" in hot or "colWeb" in hot:
        s.append(_path(["hl"], [[(p[0], p[1]) for p in sec] + [tuple(sec[0])]]))
    # --- semelles de la poutre portée (vues de dessus, trait fin) et recoupe
    if opt.realiste:
        fl = []
        for sg in (-1, 1):
            if rec > 0:
                fl.append([(x0, sg * brec / 2), (x0 + rec, sg * brec / 2),
                           (x0 + rec, sg * R.b_S / 2), (xE, sg * R.b_S / 2)])
            else:
                fl.append([(x0, sg * R.b_S / 2), (xE, sg * R.b_S / 2)])
        piece("recut" if rec > 0 else "beamS", [_path(["hd"] + hc("recut"), fl)])
    # --- âme de la poutre portée en coupe (hachurée), bord droit rompu
    if opt.realiste:
        ptsS = _rect_rompu(x0, -ws, xE - x0, R.tw_S, ("R",), 0.55 * fs)
        piece("beamS", [_poly(["ps"] + hc("beamS"), ptsS), _path(["ht"], _hachures(ptsS))],
              drag="g_h", dsym="gh")
    else:
        piece("beamS", [_rect(["ps"] + hc("beamS"), x0, -ws, xE - x0, R.tw_S)], drag="g_h", dsym="gh")
    # --- les DEUX cornières, de part et d'autre de l'âme portée
    for g in (-1, 1):
        y1 = g * ws; y2 = g * (ws + R.t_C); y3 = g * (ws + R.b_A)
        if opt.realiste:
            pts = _corniere_plan(xf, xt, R.t_C, y1, y2, y3, g, R.r_C)
            piece("cleat", [_poly(["co"] + hc("cleat"), pts), _path(["ht", "htc"], _hachures(pts))])
        else:
            piece("cleat", [_poly(["co"] + hc("cleat"), [[xf, y1], [xt, y1], [xt, y2],
                                                         [xf + R.t_C, y2], [xf + R.t_C, y3], [xf, y3]])])
        if R.bolt_P:
            cls_p = ["ax"] if opt.realiste else ["bp"]
            piece("bolts", [_path(cls_p + hc("boltsP"),
                                  [[(xf - R.t_appui - 14, g * (ws + R.g_A + i * p2P)),
                                    (xf + R.t_C + 14, g * (ws + R.g_A + i * p2P))]])
                            for i in range(R.n2_P)])
        elif opt.realiste:
            # cordon d'angle en section : triangle de côtés a·√2, dans l'angle
            # entre le bout de l'aile A et la face d'appui du poteau ; symbole
            # EN 22553 sur la cornière du haut
            z = N(u.a_P) * math.sqrt(2)
            wP = [_poly(["wb"] + hc("weldP"), [[xf, y3], [xf + z, y3], [xf, y3 + g * z]])]
            if g == -1:
                ax_, ay = xf + z * 0.55, y3 - z * 0.55
                ex, ey = ax_ + 1.6 * fs, ay - 1.4 * fs
                wt = S.tw("a " + f0(N(u.a_P)))
                tx = ex + wt + 0.9 * fs
                wP.append(_path(["wsy"], [[(ax_, ay), (ex, ey), (tx + 1.6 * fs, ey)],
                                          [(ax_ + 0.55 * fs, ay - 0.1 * fs), (ax_, ay),
                                           (ax_ + 0.1 * fs, ay - 0.55 * fs)]]))
                wP.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey - 0.95 * fs]]))
                S.tag(ex - 0.1 * fs, ey - 0.55 * fs, dict(id="aP", sym="a", val=N(u.a_P), key="a_P", lvl=1))
            piece("weldP", wP)
        else:
            piece("weldP", [_circle(["wd"], xf + R.t_C, y3, mx(N(u.a_P), 4))])
        if not R.bolt_S:
            if opt.realiste:
                z = N(u.a_S) * math.sqrt(2)
                piece("weldS", [_poly(["wb"] + hc("weldS"), [[xt, y1], [xt + z, y1], [xt, y1 + g * z]])])
            else:
                piece("weldS", [_circle(["wd"], xt, y2, mx(N(u.a_S), 4))])
    # --- files du groupe S (axes verticaux à travers cornières + âme)
    xc1 = xf + R.g_B2; xcl = xc1 + (R.n2_S - 1) * p2S; yo = -ws - R.t_C
    if R.bolt_S:
        cls_s = ["ax"] if opt.realiste else ["bp"]
        for i in range(R.n2_S):
            s.append(_path(cls_s + hc("boltsS"), [[(xc1 + i * p2S, yo - 14), (xc1 + i * p2S, -yo + 14)]]))
    # --- cotes
    if R.bolt_S:
        S.dim(dict(id="e2b", side="T", a=x0, b=xc1, o1=-ws, o2=yo - 14, sym="e2,b", val=R.e2b_S,
                   key="e2b_u", lvl=2))
        if R.n2_S > 1:
            S.dim(dict(id="p2S", side="T", a=xc1, b=xcl, o1=yo - 14, o2=yo - 14, sym="p2", val=p2S,
                       key="p2_S", lvl=2))
    S.dim(dict(id="gh", side="T", a=xf, b=x0, o1=-ws - R.b_A, o2=-ws, sym="gh", val=gh, key="g_h",
               lvl=1, out="lo"))
    S.dim(dict(id="bB", side="T", a=xf, b=xt, o1=-ws - R.b_A, o2=yo, sym="bB", val=R.b_B, lvl=1, calc=1))
    if R.bolt_P:
        ya = -(ws + R.g_A); yl = ya - (R.n2_P - 1) * p2P; xo = xf + R.t_C + 14
        S.dim(dict(id="gA", side="R", a=ya, b=-ws, o1=xo, o2=xt, sym="gA", val=R.g_A, key="gA_u",
                   lvl=1, out="lo"))
        if R.n2_P > 1:
            S.dim(dict(id="p2P", side="R", a=yl, b=ya, o1=xo, o2=xo, sym="p2", val=p2P, key="p2_P",
                       lvl=1, out="lo"))
        S.dim(dict(id="e2A", side="R", a=-(ws + R.b_A), b=yl, o1=xf + R.t_C, o2=xo, sym="e2",
                   val=R.e2a_P, lvl=2, calc=1, out="lo"))
        # le GAUGE : entraxe des deux files, symétrique par rapport à l'âme du poteau
        S.dim(dict(id="p3", side="L", a=ya, b=-ya, o1=-xcol, o2=-xcol, sym="p3", val=R.p_3, lvl=1, calc=1))
    else:
        if opt.editables:
            S.tag(xf + R.t_C + 0.6 * fs, (ws + R.b_A) + 1.2 * fs,
                  dict(id="lhP", sym="ℓh", val=N(u.lh_P), key="lh_P", lvl=1))
        if not opt.realiste:
            S.tag(xf + R.t_C + 0.6 * fs, -(ws + R.b_A) - 0.2 * fs,
                  dict(id="aP", sym="a", val=N(u.a_P), key="a_P", lvl=1))
    S.dim(dict(id="bA", side="R", a=ws, b=ws + R.b_A, o1=xt, o2=xf + R.t_C, sym="bA", val=R.b_A, lvl=1, calc=1))
    S.dim(dict(id="tc", side="R", a=ws, b=ws + R.t_C, o1=xt, o2=xt, sym="tc", val=R.t_C, lvl=2, calc=1))
    S.dim(dict(id="twS", side="R", a=-ws, b=ws, o1=xE, o2=xE, sym="tw", val=R.tw_S, lvl=2, calc=1))
    # excentricité apportée au poteau : axe du poteau → point d'application
    S.dim(dict(id="ecol", side="B", a=0, b=xf + R.zeff, o1=ycol, o2=-ws, sym="e_col",
               val=R.e_col, lvl=2, calc=1))
    # la largeur que l'élévation ne porte pas (hP en cas b, bP en cas a)
    if R.ame:
        S.dim(dict(id="hP", side="L", a=-R.h_P / 2, b=R.h_P / 2, o1=-xcol, o2=-xcol,
                   sym="h", val=R.h_P, lvl=1, calc=1))
        S.dim(dict(id="twP", side="T", a=-R.tw_P / 2, b=R.tw_P / 2, o1=-ycol, o2=-ycol,
                   sym="tw", val=R.tw_P, lvl=2, calc=1))
        S.dim(dict(id="tfP", side="L", a=R.h_P / 2 - R.tf_P, b=R.h_P / 2, o1=-xcol, o2=-xcol,
                   sym="tf", val=R.tf_P, lvl=2, calc=1))
        # la LARGEUR DROITE entre congés : la grandeur qui commande le cas (b)
        # (elle plafonne l'aile A, le gauge et le dégagement de l'écrou)
        yi = R.h_P / 2 - R.tf_P - R.r_P
        S.dim(dict(id="libreP", side="R", a=-yi, b=yi, o1=mn(x0 + mx(rec, 20), xE),
                   o2=mn(x0 + mx(rec, 20), xE), sym="ℓ", val=R.libre_droit, lvl=1, calc=1))
        # le JEU LATÉRAL se cote contre la face intérieure de l'aile : c'est la
        # valeur SAISIE (j_lat) qui s'affiche, celle qui commande le passage
        # hP − 2·tfP − 2·j — et non l'écart résiduel, qui se lit sur le dessin
        yf = R.h_P / 2 - R.tf_P
        jl = N(u.j_lat)
        if jl > 0.01:
            S.dim(dict(id="jlat", side="R", a=yf - jl, b=yf, o1=mn(x0 + rec, xE), o2=mn(x0 + rec, xE),
                       sym="j", val=jl, key="j_lat", lvl=1, out="lo"))
        else:
            S.tag(mn(x0 + rec, xE) + 0.4 * fs, yf - 0.6 * fs,
                  dict(id="jlat", sym="j", val=jl, key="j_lat", lvl=1))
        if rec > 0:
            S.dim(dict(id="rec", side="B", a=x0, b=x0 + rec, o1=brec / 2, o2=R.b_S / 2,
                       sym="rc", val=rec, key="rec_S", lvl=1))
        else:
            S.tag(x0 + 0.4 * fs, -R.b_S / 2 - 0.6 * fs,
                  dict(id="rec", sym="rc", val=0, key="rec_S", lvl=1))
    else:
        S.dim(dict(id="bP", side="L", a=-R.b_P / 2, b=R.b_P / 2, o1=-xcol, o2=-xcol,
                   sym="b", val=R.b_P, lvl=1, calc=1))
        S.dim(dict(id="tfP", side="T", a=R.h_P / 2 - R.tf_P, b=R.h_P / 2, o1=-ycol, o2=-ycol,
                   sym="tf", val=R.tf_P, lvl=2, calc=1))
        S.dim(dict(id="twP", side="L", a=-R.tw_P / 2, b=R.tw_P / 2, o1=-xcol, o2=-xcol,
                   sym="tw", val=R.tw_P, lvl=2, calc=1))
        # la PINCE au bout d'aile : le seul bord libre du poteau en cas (a)
        if R.bolt_P:
            S.dim(dict(id="eP", side="R", a=R.b_P / 2 - R.e_P_fl, b=R.b_P / 2, o1=-xcol, o2=-xcol,
                       sym="e", val=R.e_P_fl, lvl=1, calc=1, out="lo"))
    S.name("T", 0, nom_poteau(R))
    return S.finish(s, None, "Vue en plan cotée : poteau en coupe et doubles cornières")


# ------------------------------------------------------------------ coupe A-A
def coupe_aa(R, opt):
    """COUPE A-A (regard le long de la poutre portée) : en cas (a) la face
    EXTÉRIEURE de l'aile du poteau (largeur bP, rompue en haut et en bas) ;
    en cas (b) l'INTÉRIEUR du poteau — l'âme en face, les deux ailes par la
    tranche avec leurs congés. Les deux AILES A sont vues en vraie grandeur,
    avec le perçage du groupe P (e1, p1, gA, p2, p3) ; la poutre portée est
    coupée par-devant (profil complet, au-delà du grugeage)."""
    u = R.u
    ws = R.tw_S / 2; yt = 0.0; zc = N(u.z_C); yc = yt + zc
    p2P = N(u.p2_P)
    larg = R.h_P if R.ame else R.b_P
    mar = marge_poteau(R)
    xg = ws + R.g_A                                   # première file (talon → file)
    xgl = xg + (R.n2_P - 1) * p2P                     # dernière file
    r0 = R.d_0 / 2
    Wp = mx(larg / 2, ws + R.b_A, xgl + r0 + 18, R.b_S / 2 + 6) + 30
    bb = dict(x1=-Wp, x2=Wp, y1=yt - mar, y2=yt + R.h_S + mar)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 240) / 34, 9)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(k):
        return ["hot"] if k in hot else []

    y1c, y2c = yt - mar, yt + R.h_S + mar
    s.append(_poly(["pp"] + hc("col"), _rect_rompu(-larg / 2, y1c, larg, y2c - y1c, ("T", "B"), 0.55 * fs)))
    if R.ame:            # les deux ailes par la tranche, et la largeur droite
        xi = larg / 2 - R.tf_P
        s.append(_path(["fl2"] + hc("colFl"), [[(-xi, y1c), (-xi, y2c)], [(xi, y1c), (xi, y2c)]]))
        if R.r_P > 0:
            xcg = xi - R.r_P
            s.append(_path(["flr"], [[(-xcg, y1c), (-xcg, y2c)], [(xcg, y1c), (xcg, y2c)]]))
    else:                # l'âme du poteau, derrière l'aile : deux traits fins
        xw = R.tw_P / 2
        s.append(_path(["fl2"] + hc("colWeb"), [[(-xw, y1c), (-xw, y2c)], [(xw, y1c), (xw, y2c)]]))
    # --- les deux ailes A en vraie grandeur ; tranche des ailes B
    for g in (-1, 1):
        xa = mn(g * ws, g * (ws + R.b_A)); s.append(_rect(["co"] + hc("cleat"), xa, yc, R.b_A, R.L_C))
        xb = mn(g * ws, g * (ws + R.t_C)); s.append(_rect(["co2"], xb, yc, R.t_C, R.L_C))
    yp1 = yc + R.e1_P; ypl = yp1 + (R.n1_P - 1) * R.p1_P
    if R.bolt_P:
        dep = R.d_0 / 2 + 5
        for g in (-1, 1):
            for j in range(R.n2_P):
                xh = g * (xg + j * p2P)
                for i in range(R.n1_P):
                    s.append(_circle(["bo"] + hc("boltsP"), xh, yp1 + i * R.p1_P, r0))
        ax = [[(-xgl - dep, yp1 + i * R.p1_P), (xgl + dep, yp1 + i * R.p1_P)] for i in range(R.n1_P)]
        for g in (-1, 1):
            for j in range(R.n2_P):
                xh = g * (xg + j * p2P)
                ax.append([(xh, yp1 - dep), (xh, ypl + dep)])
        s.append(_path(["ax"], ax))
        if opt.fabrication:
            _renvoi(S, s, xgl + r0 * 0.72, ypl + r0 * 0.72, xgl + r0 + 1.6 * fs, ypl + 1.9 * fs,
                    js_str(2 * R.n1_P * R.n2_P) + "×Ø" + f0(R.d_0))
    else:
        # cordons A : bande verticale a·√2 au bout de chaque aile A
        z = N(u.a_P) * math.sqrt(2)
        for g in (-1, 1):
            s.append(_rect(["wb"] + hc("weldP"), mn(g * (ws + R.b_A), g * (ws + R.b_A) + g * z),
                           yc, z, R.L_C))
    # --- coupe de la poutre portée, par-devant : profil complet (le plan de
    # --- coupe est au-delà du grugeage et de la recoupe)
    sec = _section_portee(R, yt)
    s.append(_poly(["ps"] + hc("beamS"), sec))
    s.append(_path(["ht"], _hachures(sec)))
    # --- cotes : le groupe P se lit ici (sa vraie face)
    if R.bolt_P:
        S.dim(dict(id="gA", side="T", a=ws, b=xg, o1=yc, o2=yp1, sym="gA", val=R.g_A, key="gA_u",
                   lvl=1, out="lo"))
        if R.n2_P > 1:
            S.dim(dict(id="p2P", side="T", a=xg, b=xgl, o1=yp1, o2=yp1, sym="p2", val=p2P,
                       key="p2_P", lvl=1))
        S.dim(dict(id="p3", side="T", a=-xg, b=xg, o1=yp1, o2=yp1, sym="p3", val=R.p_3, lvl=1, calc=1))
        S.dim(dict(id="e1P", side="L", a=yc, b=yp1, o1=-xgl, o2=-xgl, sym="e1", val=R.e1_P,
                   key="e1P_u", lvl=1, out="lo"))
        for i in range(1, R.n1_P):
            S.dim(dict(id="p1P", side="L", a=yp1 + (i - 1) * R.p1_P, b=yp1 + i * R.p1_P, o1=-xgl, o2=-xgl,
                       sym="p1", val=R.p1_P, key="p1P_u", lvl=1))
        S.dim(dict(id="e1botP", side="L", a=ypl, b=yc + R.L_C, o1=-xgl, o2=-xgl, sym="e1'",
                   val=R.e1bot_P, lvl=2, calc=1))
    S.dim(dict(id="Lc", side="R", a=yc, b=yc + R.L_C, o1=ws + R.b_A, o2=ws + R.b_A, sym="hc",
               val=R.L_C, key="LC_u", lvl=1))
    S.dim(dict(id="bS", side="B", a=-R.b_S / 2, b=R.b_S / 2, o1=yt + R.h_S, o2=yt + R.h_S, sym="bS",
               val=R.b_S, lvl=1, calc=1))
    S.name("T", 0, nom_poteau(R))
    return S.finish(s, None, "Coupe A-A : ailes A en vraie grandeur et coupe de la poutre portée")


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

# -*- coding: utf-8 -*-
"""Dessins du pied de poteau sur béton.

- ``plan``     : la vue en plan — la RÉPARTITION : massif, platine, section du
                 poteau, aire efficace (bande c), zone comprimée de la
                 répartition élastique, axe neutre, résultante C, effort de
                 chaque tige tendue ;
- ``coupe_aa`` : coupe verticale dans le plan x–z (largeur b du poteau, files
                 wx, platine, mortier, tiges et leurs têtes, massif) ;
- ``coupe_bb`` : coupe verticale dans le plan y–z (hauteur h, rangées wy) ;
- ``domaine``  : le diagramme (Mx, My) à N donné — domaine exact (niveau 2)
                 et interaction linéaire de la Table 6.7 (niveau 1, verdict).

Le KIT de dessin (primitives, Feuille de cotation, styles, rendu SVG, peintre
PDF) est celui du module doubles cornières ; seules les SCÈNES sont propres au
pied. Repère des scènes : la vue en plan garde x du moteur (parallèle aux
ailes) et RETOURNE y (le SVG descend) — le côté y > 0 du moteur, comprimé par
Mx > 0, est EN HAUT du dessin ; les coupes ont z vers le bas, origine à la
sous-face de la platine.

Rien n'est recalculé ici : tout vient de ``R`` (moteur).
"""
import math

from acier.js import N, mn, mx, js_str
from acier.formats import F, f0
from ...poutre_poutre.doubles_cornieres.schemas import (            # kit partagé
    Options, Feuille, Dessin,
    _rect, _poly, _path, _circle, _text, _group, _hachures, _rect_rompu, _section_I)
from .entrees import TYPE_CROSSE, TYPE_ETA

# Cote du dessin → clé d'entrée (édition directe sur le dessin)
CLE_PAR_COTE = {"bp": "bp_u", "lp": "lp_u", "tp": "tp_u", "wx": "w_x", "wy": "w_y",
                "hef": "hef", "tg": "t_g", "dh": "d_h", "bc": "bc_u", "lc": "lc_u", "aw": "a_w"}

# Plan de principe : grandeurs calculées admises, et répartition des cotes
# entre les trois vues (jamais deux fois la même cote sur la planche)
FABRICATION_CALC = frozenset({"ex", "ey", "b", "h"})
EXCLURE_PLAN = frozenset({"bc", "lc"})
EXCLURE_AA = frozenset({"wx", "bp", "b", "hef"})
EXCLURE_BB = frozenset({"wy", "lp", "h", "tp", "tg", "dh", "aw"})


def options_ecran(R, lvl=1, hl=None):
    """Options de l'écran : cotes modifiables cliquables, poignées des files et
    des rangées, sans cartouche texte (colonne figée courte)."""
    return Options(lvl=lvl, interactive=True, hl=hl, editables=True, poignees=True,
                   cartouche=False, notation={}, locked=None)


def options_rapport():
    return Options(lvl=1, interactive=False, hl=None, locked=None, cartouche=False, notation={})


def options_fabrication(exclure=frozenset(), fs_force=0.0):
    return Options(lvl=1, interactive=False, cartouche=False, fabrication=True,
                   exclure=frozenset(exclure), fs_force=fs_force, notation={},
                   fabrication_calc=FABRICATION_CALC)


def visserie(d):
    """Proportions de dessin de la boulonnerie (mm) : hauteur de tête k,
    hauteur d'écrou m, épaisseur de rondelle tr — dessin de principe."""
    return 0.63 * d, 0.9 * d, mx(2.5, 0.15 * d)


# ------------------------------------------------------------------ utilitaires
def _etiquette(S, x, y, txt, cls=("dl", "calc"), size=None):
    """Étiquette à halo (texte centré, ligne de base en ``y``) : les grandeurs
    calculées du dessin (efforts des tiges, résultante)."""
    fs = size or S.fs
    w = len(str(txt)) * fs * 0.56
    hit = _rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs)
    hit["rx"] = 0.2 * fs
    return _group(list(cls), [hit, _text([], None, None, txt, fs)], x=x, y=y, rot=0)


def _soustraire(intervalles, lo, hi):
    out = []
    for a, b in intervalles:
        if hi <= a or lo >= b:
            out.append((a, b))
            continue
        if lo > a:
            out.append((a, lo))
        if hi < b:
            out.append((hi, b))
    return out


def _contour_union(rects, eps=1e-6):
    """Le contour d'une union de rectangles DISJOINTS (bandes verticales de
    ``geometrie.union_rectangles``) : tous les bords, moins les portions de
    bords verticaux partagées par deux rectangles voisins."""
    segs = []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        segs.append([(x0, y0), (x1, y0)])
        segs.append([(x0, y1), (x1, y1)])
        for xe, gauche in ((x0, True), (x1, False)):
            iv = [(y0, y1)]
            for j, (a0, b0, a1, b1) in enumerate(rects):
                if j == i:
                    continue
                xo = a1 if gauche else a0
                if abs(xo - xe) > eps:
                    continue
                lo, hi = max(b0, y0), min(b1, y1)
                if hi - lo > eps:
                    iv = _soustraire(iv, lo, hi)
            for ya, yb in iv:
                if yb - ya > eps:
                    segs.append([(xe, ya), (xe, yb)])
    return segs


def _segment_droite(a, bx, by, x0, y0, x1, y1):
    """Les deux points où la droite a + bx·x + by·y = 0 coupe le rectangle
    (x0, y0, x1, y1) — vide si elle ne le traverse pas."""
    pts = []
    for px, py, qx, qy in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        fp = a + bx * px + by * py
        fq = a + bx * qx + by * qy
        if fp == fq:
            continue
        if (fp <= 0) != (fq <= 0) or fp == 0 or fq == 0:
            t = fp / (fp - fq)
            if -1e-9 <= t <= 1 + 1e-9:
                pts.append((px + t * (qx - px), py + t * (qy - py)))
    out = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in out):
            out.append(p)
    return out[:2] if len(out) >= 2 else []


def _pas_rond(x):
    """Un pas de graduation « rond » (1, 2, 2,5, 5 × 10ⁿ) proche de ``x``."""
    if x <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    cands = [1 * e, 2 * e, 2.5 * e, 5 * e, 10 * e]
    return min(cands, key=lambda c: abs(c - x))


# ----------------------------------------------------------------------- plan
def plan(R, opt):
    """Vue en plan : massif, platine, section du poteau (coupée, hachurée),
    aire efficace, zone comprimée, axe neutre, résultante, tiges et leurs
    efforts — la répartition de la page 1 et du dessin principal de l'écran."""
    u = R.u
    bp, lp, bc, lc = u.bp_u, u.lp_u, u.bc_u, u.lc_u
    hb, hl_ = bp / 2, lp / 2
    dims = R.dims
    h, b = dims["h"], dims["b"]
    r0, rw = R.d0 / 2, R.d_w / 2
    xs = sorted({t["x"] for t in R.tiges})
    ys = sorted({t["y"] for t in R.tiges})
    bb = dict(x1=-mx(bc, bp) / 2, x2=mx(bc, bp) / 2, y1=-mx(lc, lp) / 2, y2=mx(lc, lp) / 2)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 300) / 34, 9)
    txtR = js_str(int(u.n_ay)) + " rangées"
    txtF = js_str(int(u.n_ax)) + " files"
    if opt.poignees:
        S0 = Feuille(bb, fs, opt)
        bb["x2"] = mx(bb["x2"], hb + 1.0 * fs + S0.largeur_poignees(txtR) + 0.3 * fs)
        bb["x1"] = mn(bb["x1"], -hb - 1.0 * fs - S0.largeur_poignees(txtF) - 0.3 * fs)
    S = Feuille(bb, fs, opt)
    s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(cle):
        return ["hot"] if cle in hot else []

    def piece(grp, prims, **attrs):
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    # massif et platine (vue de dessus : non coupés)
    piece("pedestal", [_rect(["be"] + hc("pedestal"), -bc / 2, -lc / 2, bc, lc)])
    piece("plate", [_rect(["ps"] + hc("plate"), -hb, -hl_, bp, lp)])
    # aire efficace d'EN 1993-1-8 §6.2.5 : fond translucide et contour de l'union
    rects = [(x0, -y1, x1, -y0) for (x0, y0, x1, y1) in R.A_eff_rects]
    for (x0, y0, x1, y1) in rects:
        s.append(_rect(["ae"], x0, y0, x1 - x0, y1 - y0))
    if rects:
        s.append(_path(["ael"], _contour_union(rects)))
    # zone comprimée de la répartition élastique
    for poly in R.contact:
        if len(poly) >= 3:
            s.append(_poly(["ct"], [(x, -y) for x, y in poly]))
    # section du poteau, coupée au-dessus de la platine
    if R.forme == "I":
        sec = [(x, y - h / 2) for x, y in _section_I(b, h, dims["tw"], dims["tf"], dims.get("r", 0) or 0)]
        piece("col", [_poly(["pp"] + hc("col"), sec), _path(["ht"], _hachures(sec))])
    else:
        t = dims["t"]
        prims = [_poly(["pp"] + hc("col"), [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)])]
        for (_n, x0, y0, x1, y1) in R.parois:
            prims.append(_path(["ht"], _hachures([(x0, -y1), (x1, -y1), (x1, -y0), (x0, -y0)])))
        prims.append(_rect(["bo"], -b / 2 + t, -h / 2 + t, b - 2 * t, h - 2 * t))
        piece("col", prims)
    # axe neutre (quand une partie de l'aire efficace est décollée)
    a_, bx_, by_ = R.axe
    if R.eq_ok and 1e-9 < R.A_c < R.A_eff - 1e-6 and math.hypot(bx_, by_) > 1e-12:
        pts = _segment_droite(a_, bx_, by_, -hb, -hl_, hb, hl_)
        if pts:
            s.append(_path(["an"], [[(x, -y) for x, y in pts]]))
            (xa, ya), (xb, yb) = pts
            xe, ye = (xa, ya) if abs(xa) + abs(ya) >= abs(xb) + abs(yb) else (xb, yb)
            s.append(_etiquette(S, xe * 0.9, -ye * 0.9 + 0.35 * fs, "AN", ("dl",), 0.8 * fs))
    # résultante comprimée
    if R.eq_ok and R.C > 1e-6:
        s.append(_circle(["pt"], R.xC, -R.yC, 0.3 * fs))
        s.append(_etiquette(S, R.xC, -R.yC - 0.75 * fs, "C = " + F(R.C, 0) + " kN"))
    # tiges : rondelle, trou, traits d'axe ; effort des tiges tendues
    an = []
    for t in R.tiges:
        X, Y = t["x"], -t["y"]
        an.append(_circle(["bw"], X, Y, rw))
        an.append(_circle(["bo"] + hc("anchors"), X, Y, r0))
    dep = rw + 0.6 * fs
    ax = [[(xs[0] - dep, -y), (xs[-1] + dep, -y)] for y in ys]
    ax += [[(x, -ys[-1] - dep), (x, -ys[0] + dep)] for x in xs]
    an.append(_path(["ax"] + hc("anchors"), ax))
    piece("anchors", an)
    for t in R.tiges:
        if t["T"] > 1e-6:
            X, Y = t["x"], -t["y"]
            yl = Y - rw - 0.5 * fs if t["y"] >= 0 else Y + rw + 1.25 * fs
            s.append(_etiquette(S, X, yl, "T = " + F(t["T"], 0) + " kN"))
    # étiquette des efforts (panneau au clic)
    if opt.poignees:
        autres = any(N(u[k]) != 0 for k in ("Vx_Ed", "Vy_Ed", "T_Ed"))
        txtE = ("NEd " + f0(N(u.N_Ed)) + " · Mx " + f0(N(u.Mx_Ed)) + " · My " + f0(N(u.My_Ed))
                + (" · V" if autres else ""))
        wE = S.tw(txtE) + 0.5 * fs
        hitE = _rect(["hit"], -wE / 2, -1.05 * fs, wE, 1.42 * fs)
        hitE["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitE, _text([], None, None, txtE, fs)],
                        x=0, y=-hl_ - 0.9 * fs, rot=0, grp="efforts", id="efforts"))
    # --- cotes
    S.dim(dict(id="bp", side="T", a=-hb, b=hb, o1=-hl_, o2=-hl_, sym="bp", val=bp, key="bp_u", lvl=1))
    S.dim(dict(id="lp", side="R", a=-hl_, b=hl_, o1=hb, o2=hb, sym="lp", val=lp, key="lp_u", lvl=1))
    if len(xs) > 1:
        S.dim(dict(id="wx", side="B", a=xs[0], b=xs[-1], o1=-ys[0], o2=-ys[0], sym="wx", val=N(u.w_x), key="w_x", lvl=1))
    if len(ys) > 1:
        S.dim(dict(id="wy", side="L", a=-ys[-1], b=-ys[0], o1=xs[0], o2=xs[0], sym="wy", val=N(u.w_y), key="w_y", lvl=1))
    S.dim(dict(id="bc", side="B", a=-bc / 2, b=bc / 2, o1=lc / 2, o2=lc / 2, sym="bc", val=bc, key="bc_u", lvl=1))
    S.dim(dict(id="lc", side="L", a=-lc / 2, b=lc / 2, o1=-bc / 2, o2=-bc / 2, sym="lc", val=lc, key="lc_u", lvl=1))
    S.dim(dict(id="ex", side="T", a=xs[-1], b=hb, o1=-ys[-1], o2=-hl_, sym="ex", val=hb - xs[-1], lvl=2, calc=1))
    S.dim(dict(id="ey", side="R", a=-hl_, b=-ys[-1], o1=hb, o2=xs[-1], sym="ey", val=hl_ - ys[-1], lvl=2, calc=1, out="lo"))
    S.dim(dict(id="b", side="B", a=-b / 2, b=b / 2, o1=h / 2, o2=h / 2, sym="b", val=b, lvl=2, calc=1))
    S.dim(dict(id="h", side="L", a=-h / 2, b=h / 2, o1=-b / 2, o2=-b / 2, sym="h", val=h, lvl=2, calc=1))
    if R.c > 0 and b / 2 + R.c <= hb - 0.5:
        S.dim(dict(id="c", side="T", a=b / 2, b=b / 2 + R.c, o1=-h / 2, o2=-h / 2, sym="c", val=R.c, lvl=2, calc=1))
    S.poignees(hb + 1.0 * fs + S.largeur_poignees(txtR) / 2, 0, "anchors", "n_ay", txtR, "anchors" in hot)
    S.poignees(-hb - 1.0 * fs - S.largeur_poignees(txtF) / 2, 0, "anchors", "n_ax", txtF, "anchors" in hot)
    S.name("T", 0, R.nom_c)
    return S.finish(s, None, "Vue en plan du pied de poteau : répartition des pressions et efforts des tiges")


# --------------------------------------------------------------------- coupes
def _coupe(R, opt, axe):
    """Coupe verticale : ``axe`` = "x" (A-A : horizontale x — largeur b du
    poteau, files wx, bp, bc) ou "y" (B-B : horizontale y — hauteur h,
    rangées wy, lp, lc)."""
    u = R.u
    dims = R.dims
    tp, tg = u.tp_u, u.t_g
    tete = u.typ_a not in (TYPE_CROSSE, TYPE_ETA)
    scellee = u.typ_a == TYPE_ETA
    d, dh = R.d_a, (N(u.d_h) if tete else 0.0)
    rw = R.d_w / 2
    if axe == "x":
        Wp, Wc, Wcol = u.bp_u, u.bc_u, dims["b"]
        pos = sorted({t["x"] for t in R.tiges})
        id_p, id_c, id_w, key_p, key_c, key_w, val_w = "bp", "bc", "wx", "bp_u", "bc_u", "w_x", N(u.w_x)
        id_col = "b"
    else:
        Wp, Wc, Wcol = u.lp_u, u.lc_u, dims["h"]
        pos = sorted({t["y"] for t in R.tiges})
        id_p, id_c, id_w, key_p, key_c, key_w, val_w = "lp", "lc", "wy", "lp_u", "lc_u", "w_y", N(u.w_y)
        id_col = "h"
    k, m, tr = visserie(d)
    sw = 1.5 * d
    hef = u.hef
    Hc = mx(0.9 * mx(dims["h"], dims["b"]), 0.35 * Wp) + 50          # tronçon de poteau montré
    Hb = mn(u.hc_u, hef + mx(60.0, 0.3 * hef))                         # profondeur de massif montrée
    rompu = Hb < u.hc_u - 1e-9
    bb = dict(x1=-mx(Wc, Wp) / 2, x2=mx(Wc, Wp) / 2, y1=-tp - Hc, y2=tg + Hb)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 300) / 34, 9)
    S = Feuille(bb, fs, opt)
    s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(cle):
        return ["hot"] if cle in hot else []

    def piece(grp, prims, **attrs):
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    # massif (coupé ; continue vers le bas si plus profond que la vue)
    if rompu:
        piece("pedestal", [_poly(["be"] + hc("pedestal"), _rect_rompu(-Wc / 2, tg, Wc, Hb, ("B",), 0.55 * fs))])
    else:
        piece("pedestal", [_rect(["be"] + hc("pedestal"), -Wc / 2, tg, Wc, Hb)])
    # mortier et platine (coupée : hachures)
    if tg > 0:
        piece("grout", [_rect(["gr"] + hc("grout"), -Wp / 2, 0, Wp, tg)])
    ppl = [(-Wp / 2, -tp), (Wp / 2, -tp), (Wp / 2, 0), (-Wp / 2, 0)]
    piece("plate", [_poly(["ps"] + hc("plate"), ppl), _path(["ht"], _hachures(ppl))])
    # poteau : tronçon rompu en tête ; arêtes des parois vues de côté
    col = [_poly(["pp"] + hc("col"), _rect_rompu(-Wcol / 2, -tp - Hc, Wcol, Hc, ("T",), 0.55 * fs))]
    if R.forme == "I":
        tw, tf = dims["tw"], dims["tf"]
        if axe == "x":
            col.append(_path(["hd"], [[(-tw / 2, -tp - Hc + 3), (-tw / 2, -tp)], [(tw / 2, -tp - Hc + 3), (tw / 2, -tp)]]))
        else:
            xi = Wcol / 2 - tf
            col.append(_path(["fl2"], [[(-xi, -tp - Hc + 3), (-xi, -tp)], [(xi, -tp - Hc + 3), (xi, -tp)]]))
    else:
        xi = Wcol / 2 - dims["t"]
        col.append(_path(["fl2"], [[(-xi, -tp - Hc + 3), (-xi, -tp)], [(xi, -tp - Hc + 3), (xi, -tp)]]))
    piece("col", col)
    # cordons : face d'aile vue de front (bande a·√2) en A-A ; en B-B les
    # ailes sont de profil (triangles) et la bande court le long de l'âme
    z = N(u.a_w) * math.sqrt(2)
    wl = []
    if z > 0:
        if R.forme == "I" and axe == "y":
            tf = dims["tf"]
            xo, xi = Wcol / 2, Wcol / 2 - tf
            wl.append(_rect(["wb"] + hc("weld"), -xi, -tp - z, 2 * xi, z))
            for g in (-1, 1):
                wl.append(_poly(["wb"] + hc("weld"), [(g * xo, -tp), (g * (xo + z), -tp), (g * xo, -tp - z)]))
                wl.append(_poly(["wb"] + hc("weld"), [(g * xi, -tp), (g * (xi - z), -tp), (g * xi, -tp - z)]))
        else:
            wl.append(_rect(["wb"] + hc("weld"), -Wcol / 2, -tp - z, Wcol, z))
            for g in (-1, 1):
                wl.append(_poly(["wb"] + hc("weld"), [(g * Wcol / 2, -tp), (g * (Wcol / 2 + z), -tp), (g * Wcol / 2, -tp - z)]))
        if axe == "x":
            # symbole EN 22553 : flèche au cordon, ligne de référence à droite
            # du poteau, triangle du cordon d'angle, désignation « a … »
            ax_, ay = Wcol / 2 + z, -tp - z / 2
            ex, ey = ax_ + 2.2 * fs, -tp - z - 2.6 * fs
            wt = S.tw("a " + f0(N(u.a_w)))
            tx = ex + wt + 0.9 * fs
            wl.append(_path(["wsy"], [[(ax_, ay), (ex, ey), (tx + 1.6 * fs, ey)],
                                      [(ax_ + 0.55 * fs, ay - 0.1 * fs), (ax_, ay), (ax_ + 0.1 * fs, ay - 0.55 * fs)]]))
            wl.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey - 0.95 * fs]]))
            S.tag(ex - 0.1 * fs, ey - 0.55 * fs, dict(id="aw", sym="a", val=N(u.a_w), key="a_w", lvl=1))
    if wl:
        piece("weld", wl)
    # tiges : tige, rondelle et écrou sur la platine, tête (ou crosse) dans le massif
    an = []
    for p in pos:
        an.append(_rect(["tg"] + hc("anchors"), p - d / 2, -tp - tr - m, d, tg + hef + tp + tr + m))
        an.append(_rect(["bo"], p - rw, -tp - tr, 2 * rw, tr))
        an.append(_rect(["bo"], p - sw / 2, -tp - tr - m, sw, m))
        if tete:
            an.append(_rect(["tg"] + hc("anchors"), p - dh / 2, tg + hef - 0.6 * d, dh, 0.6 * d))
        elif scellee:
            # tige scellée (post-installée) : gaine de scellement autour de la tige, sans tête
            an.append(_rect(["gr"], p - 0.9 * d, tg, 1.8 * d, hef))
            an.append(_rect(["tg"] + hc("anchors"), p - d / 2, tg, d, hef))
        else:
            g = -1 if p > 0 else 1                                   # crosse tournée vers l'axe
            an.append(_rect(["tg"] + hc("anchors"), mn(p, p + g * 4 * d), tg + hef - d, 4 * d, d))
    piece("anchors", an)
    # étiquette des efforts, dans le tronçon de poteau (A-A, écran)
    if opt.poignees and axe == "x":
        autres = any(N(u[k]) != 0 for k in ("Vx_Ed", "Vy_Ed", "T_Ed"))
        txtE = ("NEd " + f0(N(u.N_Ed)) + " · Mx " + f0(N(u.Mx_Ed)) + " · My " + f0(N(u.My_Ed))
                + (" · V" if autres else ""))
        wE = S.tw(txtE) + 0.5 * fs
        hitE = _rect(["hit"], -wE / 2, -1.05 * fs, wE, 1.42 * fs)
        hitE["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitE, _text([], None, None, txtE, fs)],
                        x=0, y=-tp - 0.55 * Hc, rot=0, grp="efforts", id="efforts"))
    # --- cotes
    S.dim(dict(id="tp", side="L", a=-tp, b=0, o1=-Wp / 2, o2=-Wp / 2, sym="tp", val=tp, key="tp_u", lvl=1, out="lo"))
    if tg > 0:
        S.dim(dict(id="tg", side="L", a=0, b=tg, o1=-Wp / 2, o2=-Wp / 2, sym="tg", val=tg, key="t_g", lvl=1))
    S.dim(dict(id="hef", side="R", a=tg, b=tg + hef, o1=Wc / 2, o2=pos[-1] + mx(dh / 2, d / 2), sym="hef", val=hef,
               key="hef", lvl=1))
    if tete and dh > 0:
        S.dim(dict(id="dh", side="B", a=pos[-1] - dh / 2, b=pos[-1] + dh / 2, o1=tg + hef, o2=tg + hef, sym="dh",
                   val=dh, key="d_h", lvl=1, out="hi"))
    if len(pos) > 1:
        S.dim(dict(id=id_w, side="B", a=pos[0], b=pos[-1], o1=tg + hef, o2=tg + hef, sym=id_w, val=val_w,
                   key=key_w, lvl=1))
    S.dim(dict(id=id_p, side="T", a=-Wp / 2, b=Wp / 2, o1=-tp, o2=-tp, sym=id_p, val=Wp, key=key_p, lvl=1))
    S.dim(dict(id=id_c, side="T", a=-Wc / 2, b=Wc / 2, o1=tg, o2=tg, sym=id_c, val=Wc, key=key_c, lvl=1))
    S.dim(dict(id=id_col, side="T", a=-Wcol / 2, b=Wcol / 2, o1=-tp - Hc, o2=-tp - Hc, sym=id_col, val=Wcol,
               lvl=2, calc=1))
    S.name("B", 0, R.nom_c)
    return S.finish(s, None, ("Coupe A-A du pied de poteau (plan x–z)" if axe == "x"
                              else "Coupe B-B du pied de poteau (plan y–z)"))


def coupe_aa(R, opt):
    return _coupe(R, opt, "x")


def coupe_bb(R, opt):
    return _coupe(R, opt, "y")


elevation = coupe_aa                 # l'« élévation » de la famille : la coupe A-A


def dessins(R, opt):
    return coupe_aa(R, opt), plan(R, opt)


# -------------------------------------------------------------------- domaine
def domaine(R, opt=None):
    """Le diagramme (Mx, My) à N = NEd : frontière exacte du niveau 2 (bloc
    fjd, tiges tendues à leur résistance), losange du niveau 1 (Table 6.7,
    interaction linéaire — le verdict), point des moments appliqués."""
    pts2 = [(float(x), float(y)) for x, y in (R.dom or [])]
    M1 = R.MRd1 or {}
    m1 = {k: float(M1.get(k, 0.0) or 0.0) for k in ("x+", "x-", "y+", "y-")}
    mx_, my_ = N(R.u.Mx_Ed), N(R.u.My_Ed)
    vals = [abs(v) for p in pts2 for v in p] + list(m1.values()) + [abs(mx_), abs(my_)]
    E = mx(1.0, max(vals)) * 1.15
    fs = (opt.fs_force if opt and opt.fs_force else 0.0) or E / 13
    pas = _pas_rond(E / 4)
    corps = []
    grille = []
    k = pas
    while k < E:
        grille += [[(k, -E), (k, E)], [(-k, -E), (-k, E)], [(-E, k), (E, k)], [(-E, -k), (E, -k)]]
        k += pas
    if grille:
        corps.append(_path(["grd"], grille))
    corps.append(_path(["axe"], [[(-E, 0), (E, 0)], [(0, -E), (0, E)]]))
    k = pas
    while k < E:
        lab = f0(k)
        corps.append(_text([], k, 1.15 * fs, lab, 0.75 * fs))
        corps.append(_text([], -k, 1.15 * fs, "−" + lab, 0.75 * fs))
        corps.append(_text([], -1.2 * fs, -k + 0.28 * fs, lab, 0.75 * fs))
        corps.append(_text([], -1.2 * fs, k + 0.28 * fs, "−" + lab, 0.75 * fs))
        k += pas
    if len(pts2) >= 3:
        corps.append(_poly(["d2"], [(x, -y) for x, y in pts2]))
    if any(v > 0 for v in m1.values()):
        corps.append(_poly(["d1"], [(m1["x+"], 0), (0, -m1["y+"]), (-m1["x-"], 0), (0, m1["y-"])]))
    corps.append(_circle(["pt"], mx_, -my_, 0.32 * fs))
    corps.append(_text(["tx"], mx_ + 1.9 * fs, -my_ - 0.45 * fs, "MEd", 0.85 * fs))
    corps.append(_text(["tx"], E - 2.4 * fs, -0.5 * fs, "Mx (kNm)", 0.85 * fs))
    corps.append(_text(["tx"], 2.6 * fs, -E + 0.9 * fs, "My (kNm)", 0.85 * fs))
    y0 = E + 1.7 * fs
    corps.append(_path(["d2"], [[(-E, y0), (-E + 2.2 * fs, y0)]]))
    corps.append(_text([], -E + 2.6 * fs + 0.56 * 0.75 * fs * 22, y0 + 0.28 * fs,
                       "niveau 2 : domaine exact (information)", 0.75 * fs))
    corps.append(_path(["d1"], [[(-E, y0 + 1.5 * fs), (-E + 2.2 * fs, y0 + 1.5 * fs)]]))
    corps.append(_text([], -E + 2.6 * fs + 0.56 * 0.75 * fs * 27, y0 + 1.78 * fs,
                       "niveau 1 : Table 6.7, interaction linéaire (verdict)", 0.75 * fs))
    vb = (-E - 2.6 * fs, -E - 1.4 * fs, 2 * E + 4.2 * fs, 2 * E + 5.0 * fs)
    return Dessin(viewbox=vb, corps=corps, cotes=[], cartouche=[],
                  aria="Domaine résistant (Mx, My) du pied de poteau à N donné", fs=fs)


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

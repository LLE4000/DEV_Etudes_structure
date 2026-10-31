# -*- coding: utf-8 -*-
"""Dessins du calcul manuel de la platine — le KIT du module doubles
cornières, les utilitaires des dessins du pied (étiquettes, contour d'une
union, droite de l'axe neutre, proportions de la visserie), des SCÈNES
propres à la disposition libre :

- ``plan``     : la vue en plan — massif, platine, aire d'appui, zone
                 comprimée, chaque élément soudé (section coupée, hachurée)
                 et chaque raidisseur, chaque boulon numéroté, et CE QUE LE
                 CALCUL MANUEL REGARDE : le bras de levier de chaque boulon
                 tendu jusqu'à son appui (« m = 65 »), la bande comprimée la
                 plus sollicitée (« mEd »), la résultante C, l'axe neutre ;
                 à l'écran, boulons, éléments et raidisseurs se DÉPLACENT à la
                 souris (clés dérivées positives : distance au bord gauche
                 « xg » et au bord haut « yh » de la platine) ;
- ``coupe_aa`` : coupe verticale x–z (projection : tous les éléments, tous les
                 boulons), ``coupe_bb`` : y–z ;
- en FABRICATION (plan de principe) : chaîne de cotes des perçages depuis
  les bords de la platine (abscisses distinctes en bas, ordonnées à gauche).

Repère : x du moteur, y RETOURNÉ (le SVG descend). Rien n'est recalculé ici.
"""
import math

from acier.js import N, mn, mx, js_str
from acier.formats import F, f0
from ....poutre_poutre.doubles_cornieres.schemas import (           # kit partagé
    Options, Feuille, Dessin,
    _rect, _poly, _path, _circle, _text, _group, _hachures, _rect_rompu)
from ..schemas import _etiquette, _contour_union, _segment_droite, visserie
from ..geometrie import clip as GE_clip
from ..entrees import TYPE_CROSSE, TYPE_ETA, TYPE_I, TYPE_CHS, TYPE_ROND
from . import elements as EL

T_MIN_ETIQ = 0.5                      # kN : en dessous, le boulon est dessiné détendu (ni effort, ni bras)

# Cote du dessin → clé d'entrée (édition directe sur le dessin)
CLE_PAR_COTE = {"bp": "bp_u", "lp": "lp_u", "tp": "tp_u", "tg": "t_g", "bc": "bc_u", "lc": "lc_u", "aw": "a_w",
                "hef": "hef"}
EXCLURE_PLAN = frozenset({"bc", "lc"})
EXCLURE_AA = frozenset({"bp", "hef"})
EXCLURE_BB = frozenset({"lp", "tp", "tg", "aw", "dh"})
SYM_XG, SYM_YH = "xg", "yh"           # clés de glissement : distance au bord gauche / au bord haut


def cle_glisse(prefixe, i):
    """Les deux clés dérivées d'une pièce déplaçable : ``t3_u`` (xg) et ``t3_v`` (yh)."""
    return f"{prefixe}{i}_u", f"{prefixe}{i}_v"


def options_ecran(R, lvl=1, hl=None):
    return Options(lvl=lvl, interactive=True, hl=hl, editables=True, poignees=True,
                   cartouche=False, notation={}, locked=None)


def options_rapport():
    return Options(lvl=1, interactive=False, hl=None, locked=None, cartouche=False, notation={})


def options_fabrication(R, exclure=frozenset(), fs_force=0.0):
    """Plan de principe : toutes les cotes d'entrée et la chaîne des perçages."""
    xs = sorted({round(t["x"], 3) for t in R.tiges})
    ys = sorted({round(t["y"], 3) for t in R.tiges})
    calc = {f"px{i}" for i in range(len(xs) + 1)} | {f"py{j}" for j in range(len(ys) + 1)} | {"ex", "ey"}
    return Options(lvl=1, interactive=False, cartouche=False, fabrication=True,
                   exclure=frozenset(exclure), fs_force=fs_force, notation={}, fabrication_calc=frozenset(calc))


# ----------------------------------------------------------------------- plan
def _prims_element(e, hc):
    """La section coupée d'un élément (repère y retourné) : contour, hachures."""
    c = EL.contour(e)
    if c["t"] == "I":
        pts = [(x, -y) for x, y in c["pts"]]
        return [_poly(["pp"] + hc, pts), _path(["ht"], _hachures(pts))]
    if c["t"] == "disque":
        return [_circle(["pp"] + hc, c["cx"], -c["cy"], c["R"])]
    if c["t"] == "anneau":
        return [_circle(["pp"] + hc, c["cx"], -c["cy"], c["R"]), _circle(["bo"], c["cx"], -c["cy"], c["Ri"])]
    out = []
    for (x0, y0, x1, y1) in c["rects"]:
        pts = [(x0, -y1), (x1, -y1), (x1, -y0), (x0, -y0)]
        out.append(_poly(["pp"] + hc, pts))
        out.append(_path(["ht"], _hachures(pts)))
    return out


def plan(R, opt):
    u = R.u
    bp, lp, bc, lc = u.bp_u, u.lp_u, u.bc_u, u.lc_u
    hb, hl_ = bp / 2, lp / 2
    r0, rw = R.d0 / 2, R.d_w / 2
    bb = dict(x1=-mx(bc, bp) / 2, x2=mx(bc, bp) / 2, y1=-mx(lc, lp) / 2, y2=mx(lc, lp) / 2)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 300) / 34, 9)
    S = Feuille(bb, fs, opt)
    s = []
    hot = (opt.hl or {}).get("elems") or set()
    inter = bool(opt.poignees)

    def hc(cle):
        return ["hot"] if cle in hot else []

    def piece(grp, prims, **attrs):
        if inter and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    piece("pedestal", [_rect(["be"] + hc("pedestal"), -bc / 2, -lc / 2, bc, lc)])
    piece("plate", [_rect(["ps"] + hc("plate"), -hb, -hl_, bp, lp)])
    # aire d'appui (bande c, ou toute la platine) et zone comprimée — pas sur le plan de principe
    ronds = [e for e in R.elems if EL.est_rond(e) and EL.geometrie_valide(e)]
    if not opt.fabrication:
        if ronds and not R.plein:
            # un rond : la bande est un anneau — deux cercles plutôt que ses lamelles
            for e in ronds:
                R_ext, R_int = EL.rayons(e)
                s.append(_circle(["ae"], e["x"], -e["y"], R_ext + R.c))
                s.append(_circle(["ael"], e["x"], -e["y"], R_ext + R.c))
                if R_int - R.c > 0:
                    s.append(_circle(["ps"], e["x"], -e["y"], R_int - R.c))
                    s.append(_circle(["ael"], e["x"], -e["y"], R_int - R.c))
            rects = [(x0, -y1, x1, -y0) for (x0, y0, x1, y1) in R.A_bandes_rects
                     if not any(EL.dedans_rect((e["x"] - EL.rayons(e)[0] - R.c, e["y"] - EL.rayons(e)[0] - R.c,
                                                e["x"] + EL.rayons(e)[0] + R.c, e["y"] + EL.rayons(e)[0] + R.c),
                                               0.5 * (x0 + x1), 0.5 * (y0 + y1), 1e-6) for e in ronds)]
        else:
            rects = [(x0, -y1, x1, -y0) for (x0, y0, x1, y1) in R.A_eff_rects]
        for (x0, y0, x1, y1) in rects:
            s.append(_rect(["ae"], x0, y0, x1 - x0, y1 - y0))
        if rects and not R.plein:
            s.append(_path(["ael"], _contour_union(rects)))
        # zone comprimée : les polygones du moteur, sauf les lamelles d'un rond,
        # remplacées par le disque (R + c) coupé par l'axe neutre, trou du vide
        a_, bx_, by_ = R.axe
        dans_rond = []
        for e in ronds:
            Re = EL.rayons(e)[0] + (0.0 if R.plein else R.c)
            dans_rond.append((e["x"] - Re, e["y"] - Re, e["x"] + Re, e["y"] + Re))
        for poly in R.contact:
            if len(poly) < 3:
                continue
            gx = sum(p[0] for p in poly) / len(poly)
            gy = sum(p[1] for p in poly) / len(poly)
            if any(EL.dedans_rect(r, gx, gy, 1e-6) for r in dans_rond):
                continue
            s.append(_poly(["ct"], [(x, -y) for x, y in poly]))
        if R.eq_ok and ronds and not R.plein:
            for e in ronds:
                R_ext, R_int = EL.rayons(e)
                disque = [(e["x"] + (R_ext + R.c) * math.cos(2 * math.pi * k / 72),
                           e["y"] + (R_ext + R.c) * math.sin(2 * math.pi * k / 72)) for k in range(72)]
                part = disque if (abs(bx_) + abs(by_) < 1e-15 and a_ > 0) else \
                    (GE_clip(disque, bx_, by_, -a_) if abs(bx_) + abs(by_) >= 1e-15 else [])
                if len(part) >= 3:
                    s.append(_poly(["ct"], [(x, -y) for x, y in part]))
                if R_int - R.c > 0:
                    s.append(_circle(["ps"], e["x"], -e["y"], R_int - R.c))
    # éléments soudés et raidisseurs — déplaçables à l'écran
    for i, e in enumerate(R.elems):
        if not EL.geometrie_valide(e):
            continue
        ku, kv = cle_glisse("e", i)
        piece(f"elem{i}", _prims_element(e, hc("col")), drag=ku, dsym=SYM_XG, drag2=kv, dsym2=SYM_YH)
    for i, r in enumerate(R.raid):
        x0, y0, x1, y1 = EL.rect_raid(r)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        pts = [(x0, -y1), (x1, -y1), (x1, -y0), (x0, -y0)]
        ku, kv = cle_glisse("r", i)
        piece(f"raid{i}", [_poly(["rd"] + hc("raid"), pts), _path(["ht"], _hachures(pts))],
              drag=ku, dsym=SYM_XG, drag2=kv, dsym2=SYM_YH)
    # axe neutre et résultante comprimée
    a_, bx_, by_ = R.axe
    if not opt.fabrication and R.eq_ok and 1e-9 < R.A_c < R.A_eff - 1e-6 and math.hypot(bx_, by_) > 1e-12:
        pts = _segment_droite(a_, bx_, by_, -hb, -hl_, hb, hl_)
        if pts:
            s.append(_path(["an"], [[(x, -y) for x, y in pts]]))
            (xa, ya), (xb, yb) = pts
            xe, ye = (xa, ya) if abs(xa) + abs(ya) >= abs(xb) + abs(yb) else (xb, yb)
            # l'étiquette au-delà du bord de la platine, dans le prolongement de l'axe
            ne = math.hypot(xe, ye) or 1.0
            s.append(_etiquette(S, xe + 1.2 * fs * xe / ne, -(ye + 1.2 * fs * ye / ne) + 0.35 * fs, "AN", ("dl",), 0.8 * fs))
    if not opt.fabrication and R.eq_ok and R.C > 1e-6:
        s.append(_circle(["pt"], R.xC, -R.yC, 0.3 * fs))
        s.append(_etiquette(S, R.xC, -R.yC - 0.75 * fs, "C = " + F(R.C, 0) + " kN"))
    # la bande comprimée la plus sollicitée : son repère « mEd » au nu, décalé de côté (la valeur est dans le tableau)
    cs = R.console
    if not opt.fabrication and R.eq_ok and cs.get("L", 0) > 0 and cs.get("m_Ed", 0) > 1e-9:
        ox, oy, nx, ny, L = cs["ox"], cs["oy"], cs["nx"], cs["ny"], cs["L"]
        s.append(_path(["bs"], [[(ox, -oy), (ox + L * nx, -(oy + L * ny))]]))
        s.append(_circle(["bsp"], ox, -oy, 0.28 * fs))
        px_, py_ = -ny, nx
        if px_ * ox + py_ * oy < 0:                 # du côté extérieur (loin du centre de la platine)
            px_, py_ = -px_, -py_
        xm, ym = ox + 0.45 * L * nx + 1.5 * fs * px_, oy + 0.45 * L * ny + 1.5 * fs * py_
        s.append(_etiquette(S, xm, -ym + 0.35 * fs, "mEd", ("dl", "calc"), 0.8 * fs))
    # boulons numérotés, déplaçables ; effort et bras de levier des boulons tendus
    for t in R.tiges:
        X, Y = t["x"], -t["y"]
        ku, kv = cle_glisse("t", t["i"])
        an = [_circle(["bw"], X, Y, rw), _circle(["bo"] + hc("anchors"), X, Y, r0),
              _path(["ax"] + hc("anchors"), [[(X - rw - 0.4 * fs, Y), (X + rw + 0.4 * fs, Y)],
                                             [(X, Y - rw - 0.4 * fs), (X, Y + rw + 0.4 * fs)]])]
        piece(f"bolt{t['i']}", an, drag=ku, dsym=SYM_XG, drag2=kv, dsym2=SYM_YH)
        # le numéro du côté extérieur du boulon (vers le coin de la platine le plus proche)
        sx, sy = (1 if t["x"] >= 0 else -1), (1 if t["y"] >= 0 else -1)
        s.append(_text(["tx"], X + sx * (rw + 0.35 * fs), Y - sy * (rw + 0.15 * fs) + (0.75 * fs if sy < 0 else 0.0),
                       js_str(t["i"] + 1), 0.8 * fs, "start" if sx > 0 else "end"))
        if not opt.fabrication and t.get("T", 0) > T_MIN_ETIQ:
            yl = Y - rw - 0.6 * fs if t["y"] >= 0 else Y + rw + 1.3 * fs
            s.append(_etiquette(S, X, yl, "T = " + (F(t["T"], 1) if t["T"] < 10 else F(t["T"], 0)) + " kN", ("dl", "calc"), 0.9 * fs))
            a = t["appui"]
            if a["d"] > 1e-6:
                s.append(_path(["lv"], [[(X, Y), (a["px"], -a["py"])]]))
                xm, ym = 0.5 * (X + a["px"]), 0.5 * (Y - a["py"])
                dx, dy = a["px"] - X, -a["py"] - Y
                nd = math.hypot(dx, dy) or 1.0
                # l'étiquette décalée perpendiculairement au bras, du côté opposé au centre de la platine
                px_, py_ = -dy / nd, dx / nd
                if px_ * xm + py_ * ym < 0:
                    px_, py_ = -px_, -py_
                s.append(_etiquette(S, xm + 1.1 * fs * px_, ym + 1.1 * fs * py_ + 0.35 * fs, "m = " + F(t["m"], 0),
                                    ("dl", "calc"), 0.8 * fs))
    # étiquette des efforts (panneau au clic), au point d'application
    if inter:
        autres = any(N(u[k]) != 0 for k in ("Vx_Ed", "Vy_Ed", "T_Ed"))
        txtE = ("NEd " + f0(N(u.N_Ed)) + " · Mx " + f0(N(u.Mx_Ed)) + " · My " + f0(N(u.My_Ed))
                + (" · V" if autres else ""))
        wE = S.tw(txtE) + 0.5 * fs
        hitE = _rect(["hit"], -wE / 2, -1.05 * fs, wE, 1.42 * fs)
        hitE["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitE, _text([], None, None, txtE, fs)],
                        x=0, y=-hl_ - 0.9 * fs, rot=0, grp="efforts", id="efforts"))
        if abs(R.xN) + abs(R.yN) > 1e-9:
            s.append(_circle(["pt"], R.xN, -R.yN, 0.25 * fs))
    # --- cotes
    S.dim(dict(id="bp", side="T", a=-hb, b=hb, o1=-hl_, o2=-hl_, sym="bp", val=bp, key="bp_u", lvl=1))
    S.dim(dict(id="lp", side="R", a=-hl_, b=hl_, o1=hb, o2=hb, sym="lp", val=lp, key="lp_u", lvl=1))
    S.dim(dict(id="bc", side="B", a=-bc / 2, b=bc / 2, o1=lc / 2, o2=lc / 2, sym="bc", val=bc, key="bc_u", lvl=1))
    S.dim(dict(id="lc", side="L", a=-lc / 2, b=lc / 2, o1=-bc / 2, o2=-bc / 2, sym="lc", val=lc, key="lc_u", lvl=1))
    if opt.fabrication and R.tiges:
        # chaîne des perçages : abscisses distinctes depuis le bord gauche (en bas), ordonnées depuis le bord bas (à gauche)
        xs = sorted({round(t["x"], 3) for t in R.tiges})
        ys = sorted({round(t["y"], 3) for t in R.tiges})
        prev = -hb
        for i, x in enumerate(xs + [hb]):
            S.dim(dict(id=f"px{i}", side="B", a=prev, b=x, o1=hl_, o2=hl_, sym="", val=x - prev, lvl=1, calc=1))
            prev = x
        prev = -hl_
        for j, y in enumerate(ys + [hl_]):
            S.dim(dict(id=f"py{j}", side="L", a=-y, b=-prev, o1=-hb, o2=-hb, sym="", val=y - prev, lvl=1, calc=1))
            prev = y
    S.name("T", 0, R.nom_elems)
    return S.finish(s, None, "Vue en plan de la platine : boulons, éléments soudés, raidisseurs, répartition, bras de levier")


# --------------------------------------------------------------------- coupes
def _coupe(R, opt, axe):
    """Coupe verticale en projection : ``axe`` = "x" (A-A : horizontale x) ou
    "y" (B-B : horizontale y)."""
    u = R.u
    tp, tg = u.tp_u, u.t_g
    tete = u.typ_a not in (TYPE_CROSSE, TYPE_ETA)
    scellee = u.typ_a == TYPE_ETA
    d, dh = R.d_a, (N(u.d_h) if tete else 0.0)
    rw = R.d_w / 2
    if axe == "x":
        Wp, Wc = u.bp_u, u.bc_u
        pos = sorted({round(t["x"], 3) for t in R.tiges})
        id_p, id_c, key_p, key_c = "bp", "bc", "bp_u", "bc_u"

        def hor(x, y):
            return x
    else:
        Wp, Wc = u.lp_u, u.lc_u
        pos = sorted({round(t["y"], 3) for t in R.tiges})
        id_p, id_c, key_p, key_c = "lp", "lc", "lp_u", "lc_u"

        def hor(x, y):
            return y
    k, m, tr = visserie(d)
    sw = 1.5 * d
    hef = u.hef
    env = [EL.enveloppe(e) for e in R.elems if EL.geometrie_valide(e)]
    tailles = [max(x1 - x0, y1 - y0) for (x0, y0, x1, y1) in env] or [100.0]
    Hc = mx(0.9 * max(tailles), 0.35 * Wp) + 50
    Hb = mn(u.hc_u, hef + mx(60.0, 0.3 * hef))
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

    if rompu:
        piece("pedestal", [_poly(["be"] + hc("pedestal"), _rect_rompu(-Wc / 2, tg, Wc, Hb, ("B",), 0.55 * fs))])
    else:
        piece("pedestal", [_rect(["be"] + hc("pedestal"), -Wc / 2, tg, Wc, Hb)])
    if tg > 0:
        piece("grout", [_rect(["gr"] + hc("grout"), -Wp / 2, 0, Wp, tg)])
    ppl = [(-Wp / 2, -tp), (Wp / 2, -tp), (Wp / 2, 0), (-Wp / 2, 0)]
    piece("plate", [_poly(["ps"] + hc("plate"), ppl), _path(["ht"], _hachures(ppl))])
    # éléments : tronçon rompu en tête, arêtes des parois vues de côté
    for i, e in enumerate(R.elems):
        if not EL.geometrie_valide(e):
            continue
        x0, y0, x1, y1 = EL.enveloppe(e)
        a, b = (x0, x1) if axe == "x" else (y0, y1)
        col = [_poly(["pp"] + hc("col"), _rect_rompu(a, -tp - Hc, b - a, Hc, ("T",), 0.55 * fs))]
        lignes = []
        if e["type"] == TYPE_I:
            le_long = (e["rot"] == 0) == (axe == "x")          # ailes parallèles à l'horizontale de la coupe
            cen = hor(e["x"], e["y"])
            if le_long:
                lignes = [cen - e["tw"] / 2, cen + e["tw"] / 2]
                cls = "hd"
            else:
                lignes = [a + e["tf"], b - e["tf"]]
                cls = "fl2"
        elif e["type"] == TYPE_CHS or (e["type"] not in (TYPE_ROND,) and e.get("t", 0) > 0 and e["type"] != TYPE_I):
            t_ = e["t"] if e["type"] != TYPE_I else 0
            if e["type"] in (TYPE_CHS,) or "creux" in e["type"]:
                lignes = [a + t_, b - t_]
            cls = "fl2"
        for xl in lignes:
            col.append(_path([cls], [[(xl, -tp - Hc + 3), (xl, -tp)]]))
        piece(f"elem{i}", col)
    # raidisseurs : hauteur h au-dessus de la platine
    for i, r in enumerate(R.raid):
        x0, y0, x1, y1 = EL.rect_raid(r)
        a, b = (x0, x1) if axe == "x" else (y0, y1)
        if b - a > 0 and r["h"] > 0:
            piece(f"raid{i}", [_rect(["rd"] + hc("raid"), a, -tp - r["h"], b - a, r["h"])])
    # cordon de l'élément 1 : symbole EN 22553 en A-A, cote a éditable
    if env and N(u.a_w) > 0:
        z = N(u.a_w) * math.sqrt(2)
        x0, y0, x1, y1 = env[0]
        a, b = (x0, x1) if axe == "x" else (y0, y1)
        wl = [_rect(["wb"] + hc("weld"), a, -tp - z, b - a, z)]
        for g, xe in ((-1, a), (1, b)):
            wl.append(_poly(["wb"] + hc("weld"), [(xe, -tp), (xe + g * z, -tp), (xe, -tp - z)]))
        if axe == "x":
            ax_, ay = b + z, -tp - z / 2
            ex, ey = ax_ + 2.2 * fs, -tp - z - 2.6 * fs
            wt = S.tw("a " + f0(N(u.a_w)))
            tx = ex + wt + 0.9 * fs
            wl.append(_path(["wsy"], [[(ax_, ay), (ex, ey), (tx + 1.6 * fs, ey)],
                                      [(ax_ + 0.55 * fs, ay - 0.1 * fs), (ax_, ay), (ax_ + 0.1 * fs, ay - 0.55 * fs)]]))
            wl.append(_poly(["wst"], [[tx, ey], [tx + 1.0 * fs, ey], [tx, ey - 0.95 * fs]]))
            S.tag(ex - 0.1 * fs, ey - 0.55 * fs, dict(id="aw", sym="a", val=N(u.a_w), key="a_w", lvl=1))
        piece("weld", wl)
    # boulons (projection) : tige, rondelle, écrou, tête ou scellement
    an = []
    for p in pos:
        an.append(_rect(["tg"] + hc("anchors"), p - d / 2, -tp - tr - m, d, tg + hef + tp + tr + m))
        an.append(_rect(["bo"], p - rw, -tp - tr, 2 * rw, tr))
        an.append(_rect(["bo"], p - sw / 2, -tp - tr - m, sw, m))
        if tete:
            an.append(_rect(["tg"] + hc("anchors"), p - dh / 2, tg + hef - 0.6 * d, dh, 0.6 * d))
        elif scellee:
            an.append(_rect(["gr"], p - 0.9 * d, tg, 1.8 * d, hef))
            an.append(_rect(["tg"] + hc("anchors"), p - d / 2, tg, d, hef))
        else:
            g = -1 if p > 0 else 1
            an.append(_rect(["tg"] + hc("anchors"), mn(p, p + g * 4 * d), tg + hef - d, 4 * d, d))
    piece("anchors", an)
    # --- cotes
    S.dim(dict(id="tp", side="L", a=-tp, b=0, o1=-Wp / 2, o2=-Wp / 2, sym="tp", val=tp, key="tp_u", lvl=1, out="lo"))
    if tg > 0:
        S.dim(dict(id="tg", side="L", a=0, b=tg, o1=-Wp / 2, o2=-Wp / 2, sym="tg", val=tg, key="t_g", lvl=1))
    if pos:
        S.dim(dict(id="hef", side="R", a=tg, b=tg + hef, o1=Wc / 2, o2=pos[-1] + mx(dh / 2, d / 2), sym="hef", val=hef,
                   key="hef", lvl=1))
        if tete and dh > 0:
            S.dim(dict(id="dh", side="B", a=pos[-1] - dh / 2, b=pos[-1] + dh / 2, o1=tg + hef, o2=tg + hef, sym="dh",
                       val=dh, key="d_h", lvl=1, out="hi"))
    S.dim(dict(id=id_p, side="T", a=-Wp / 2, b=Wp / 2, o1=-tp, o2=-tp, sym=id_p, val=Wp, key=key_p, lvl=1))
    S.dim(dict(id=id_c, side="T", a=-Wc / 2, b=Wc / 2, o1=tg, o2=tg, sym=id_c, val=Wc, key=key_c, lvl=1))
    S.name("B", 0, R.nom_elems)
    return S.finish(s, None, ("Coupe A-A de la platine (plan x–z, projection)" if axe == "x"
                              else "Coupe B-B de la platine (plan y–z, projection)"))


def coupe_aa(R, opt):
    return _coupe(R, opt, "x")


def coupe_bb(R, opt):
    return _coupe(R, opt, "y")


elevation = coupe_aa


def dessins(R, opt):
    return coupe_aa(R, opt), plan(R, opt)


def cotes_modifiables(dessin):
    out = []

    def rec(prims):
        for p in prims:
            if p.get("t") == "g":
                if p.get("key"):
                    out.append((p["key"], p.get("sym") or "", p.get("id") or ""))
                rec(p.get("enfants") or [])

    rec(dessin.cotes)
    return out

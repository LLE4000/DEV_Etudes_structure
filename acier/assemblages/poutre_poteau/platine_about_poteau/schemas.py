# -*- coding: utf-8 -*-
"""Dessins de l'assemblage poutre–poteau par platine d'about de hauteur
partielle — élévation, VUE EN PLAN et coupe A-A. Le KIT de dessin
(primitives, Feuille de cotation, styles, rendu SVG, lignes de rupture,
hachures) est celui du module doubles cornières, réutilisé tel quel ; seules
les SCÈNES sont propres à cet assemblage.

**La VUE EN PLAN est la vue PORTEUSE** (`POUTRE_POTEAU.md` §10.7) : c'est
elle qui montre le poteau EN COUPE, le cas d'appui, les deux files de
boulons de part et d'autre de l'âme du poteau (cas a) ou de l'âme de la
poutre portée (cas b), le passage de la poutre entre les ailes, la recoupe
des semelles et les jeux latéraux. La coupe A-A montre la platine en VRAIE
GRANDEUR et, en cas (b), la LARGEUR DROITE entre congés.

Repère des trois vues, pour les deux cas d'appui (étude §1.2) :

===============  =========================  =========================
 vue              cas (a) — sur l'AILE       cas (b) — dans l'ÂME
===============  =========================  =========================
 élévation        regard ∥ aux ailes :       regard ⊥ aux ailes :
                  bande de largeur hP,       bande de largeur bP,
                  ailes en bandes de tfP     âme repérée à ±twP/2
 vue en plan      poteau en coupe, ailes     poteau en coupe, ailes
                  VERTICALES (X = ±hP/2),    HORIZONTALES, poutre
                  platine sur la face        ENTRE les ailes
                  extérieure de l'aile
 coupe A-A        face extérieure de         intérieur du poteau :
                  l'aile (largeur bP),       âme en face, ailes par la
                  platine en vraie grandeur  tranche, largeur droite
===============  =========================  =========================

Un poteau continue AU-DESSUS et EN DESSOUS de l'attache : il est toujours
tracé en tronçon ROMPU (``_rect_rompu``), et la clé ``d_top`` du grand frère
(décalage des dessus) n'existe plus.

La boulonnerie (tête, écrou, rondelles) est dessinée en PROPORTIONS du
diamètre — un dessin de principe, pas une cote. L'écrou est du côté
INTÉRIEUR du poteau (derrière l'aile en cas a, derrière l'âme en cas b) :
c'est ce qui commande le gauge minimal p3 ≥ twP + 2·rP + dw.

Cotes : symboles finaux directement (hp, tp, bp, zp, p3, e1, p1, c, dc,sup,
rc, j, ℓ…) — pas de couche de notation.
"""
import math

from acier.js import N, mn, mx, js_str
from acier.formats import f0
from acier.bibliotheques import PERSO
from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # kit partagé
    Options, Feuille, PALETTE, INF_MM, SEGMENTS_ARC,
    _rect, _poly, _path, _circle, _text, _group, _arc, _hachures,
    _rect_rompu, _rupture, _renvoi, _section_I, _section_portee)
from .entrees import PILOTEES_PAR_PREDIM

# Cote du dessin → clé d'entrée (édition directe sur le dessin)
CLE_PAR_COTE = {"hp": "hp_u", "zp": "z_C", "e1": "e1_u", "p1": "p1_u",
                "c": "l_n", "dcs": "d_nt", "dci": "d_nb",
                "aw": "a_w", "tp": "tp_u", "p3": "p3_u", "bp": "bp_u",
                "rec": "rec_S", "jlat": "j_lat"}

# Plan de principe : grandeurs calculées admises, et répartition des cotes —
# chaque cote n'est portée qu'UNE fois sur la planche de la page 2. La
# largeur de la bande du poteau est cotée sur l'ÉLÉVATION (hP en cas a, bP
# en cas b) et l'AUTRE sur la VUE EN PLAN : les deux ids existent donc une
# seule fois, quel que soit le cas d'appui.
FABRICATION_CALC = frozenset({"bS", "e2p", "twS", "e1bot", "hP", "bP", "tfP",
                              "twP", "libreP", "ecol", "eP"})
EXCLURE_ELEVATION = frozenset()
EXCLURE_PLAN = frozenset({"p3", "bp", "e2p", "tp"})
EXCLURE_COUPE = frozenset({"hp", "e1", "p1"})


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


# ------------------------------------------------------------------ communs
def visserie(d):
    """Proportions de dessin de la boulonnerie (mm) : hauteur de tête k,
    hauteur d'écrou m, épaisseur de rondelle tr — dessin de principe."""
    return 0.63 * d, 0.9 * d, mx(2.5, 0.15 * d)


def _boulon_profil(cls_t, x_ap, x_av, y, d, dw):
    """Un boulon vu de profil, axe horizontal à la hauteur ``y`` : rondelle
    et écrou du côté INTÉRIEUR du poteau (à gauche de ``x_ap``), rondelle et
    tête du côté de la platine (à droite de ``x_av``)."""
    k, m, tr = visserie(d)
    s = 1.5 * d                      # surplat (proportion)
    return [_rect(["bw"], x_ap - tr, y - dw / 2, tr, dw),
            _rect(cls_t, x_ap - tr - m, y - s / 2, m, s),
            _rect(["bw"], x_av, y - dw / 2, tr, dw),
            _rect(cls_t, x_av + tr, y - s / 2, k, s)]


def _marge_poteau(R):
    """Hauteur du tronçon de poteau dessiné au-dessus et en dessous de la
    poutre portée (il continue : bords rompus)."""
    return mx(60.0, 0.18 * R.h_S)


def _nom_poteau(R):
    return "Poteau" if R.u.prof_P == PERSO else R.u.prof_P


def _nom_portee(R):
    return "Poutre portée" if R.u.prof_S == PERSO else R.u.prof_S


def _section_poteau(R):
    """Contour du poteau en COUPE HORIZONTALE, orienté pour la vue en plan :
    l'axe x du dessin suit l'axe de la poutre portée.

    - cas (a) : x = X (profondeur hP), y = Y (largeur d'aile bP) — les deux
      ailes sont VERTICALES, aux abscisses x = ±hP/2 ;
    - cas (b) : x = Y (largeur d'aile bP), y = X (profondeur hP) — les deux
      ailes sont HORIZONTALES, aux ordonnées y = ±hP/2, et la poutre entre
      ENTRE elles."""
    sec = _section_I(R.b_P, R.h_P, R.tw_P, R.tf_P, R.r_P)
    if R.ame:
        return [(sx, sy - R.h_P / 2) for sx, sy in sec]
    return [(sy - R.h_P / 2, sx) for sx, sy in sec]


# ------------------------------------------------------------------ élévation
def elevation(R, opt):
    """Élévation (regard ⊥ à l'axe de la poutre portée) : le poteau vu de
    côté — bande rompue en haut et en bas, ailes ou âme repérées —, la
    PLATINE par la tranche contre la face d'appui, la poutre portée soudée
    sur la platine (grugeage le cas échéant), les boulons de profil (écrou
    à l'intérieur du poteau, tête côté platine) et le symbole de soudure
    EN 22553."""
    u = R.u
    xf = R.xf_P                                  # face d'appui (axe du poteau → face)
    tp = R.t_p; x0 = xf + tp                     # face avant de la platine = about
    yt = 0.0; yb = R.h_S                         # un poteau n'a pas de dessus : Δz = 0
    zc = N(u.z_C); yc = yt + zc
    dnt = N(u.d_nt); dnb = N(u.d_nb); ln = N(u.l_n)
    larg = R.b_P if R.ame else R.h_P             # largeur de la bande du poteau
    mar = _marge_poteau(R)
    notch = dnt > 0 or dnb > 0
    k, m, tr = visserie(R.d_b)
    xtete = x0 + tr + k                          # bout des têtes de boulon
    xE = mx(x0 + (ln if notch else 0), xtete, larg / 2) + mx(85, 0.35 * R.h_S)
    bb = dict(x1=-larg / 2, x2=xE, y1=yt - mar, y2=yb + mar)
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

    # --- POTEAU : tronçon vu de côté, rompu en haut et en bas
    pts = _rect_rompu(-larg / 2, yt - mar, larg, yb - yt + 2 * mar, ("T", "B"), 0.55 * fs) \
        if opt.realiste else [(-larg / 2, yt - mar), (larg / 2, yt - mar),
                              (larg / 2, yb + mar), (-larg / 2, yb + mar)]
    piece("col", [_poly(["pp"] + hc("col"), pts)])
    if R.ame:
        # l'âme du poteau est DERRIÈRE la platine : ses deux traits sont des
        # lignes CACHÉES (tiretées), posées à la fin, par-dessus la platine —
        # sans quoi le repère de l'âme, qui est l'information du cas (b),
        # disparaît sous la pièce d'attache
        web = [[(-R.tw_P / 2, yt - mar), (-R.tw_P / 2, yb + mar)],
               [(R.tw_P / 2, yt - mar), (R.tw_P / 2, yb + mar)]]
    else:                          # les deux ailes, et les congés
        web = None
        xi = larg / 2 - R.tf_P
        s.append(_path(["fl2"] + hc("colFl"), [[(-xi, yt - mar), (-xi, yb + mar)],
                                               [(xi, yt - mar), (xi, yb + mar)]]))
        if opt.realiste and R.r_P > 0:
            xc = xi - R.r_P
            s.append(_path(["flr"], [[(-xc, yt - mar), (-xc, yb + mar)],
                                     [(xc, yt - mar), (xc, yb + mar)]]))
    # --- POUTRE PORTÉE grugée, soudée sur la platine, bord droit rompu
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
    piece("beamS", pS)
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
    # --- PLATINE par la tranche (contre la face d'appui) + cordon vu de côté
    # --- (bande a·√2 le long de l'âme portée) + symbole de cordon double
    z = N(u.a_w) * math.sqrt(2)
    pl = [_rect(["co"] + hc("plate"), xf, yc, tp, R.h_p)]
    wl = []
    if opt.realiste:
        wl.append(_rect(["wb"] + hc("weld"), x0, yc, z, R.h_p))
        # symbole EN 22553 : flèche au cordon, coude SOUS la platine (la ligne
        # de référence ne traverse ni la platine ni le groupe de boulons) et,
        # si la platine descend bas, SOUS la poutre : l'étiquette « a » y est
        # seule — elle chevauchait l'étiquette « dc,inf » (relecture des
        # captures du 02/11/2026)
        ax_, ay = x0 + z, yc + 0.88 * R.h_p
        ex = ax_ + 2.6 * fs
        ey = mx(yc + R.h_p + 1.9 * fs, yb + 0.9 * fs)
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
    # --- boulons de profil : écrou et rondelle du côté INTÉRIEUR du poteau
    # --- (derrière l'aile en cas a, derrière l'âme en cas b), tête et
    # --- rondelle côté platine ; traits d'axe à travers le poteau et la platine
    x_ap = xf - R.t_appui
    bS = []
    for i in range(R.n_1):
        y = yb1 + i * R.p_1
        if opt.realiste:
            bS += _boulon_profil(["bo"] + hc("bolts"), x_ap, x0, y, R.d_b, R.d_w)
    dep = 6
    ax = [[(x_ap - tr - m - dep, yb1 + i * R.p_1), (xtete + dep, yb1 + i * R.p_1)] for i in range(R.n_1)]
    bS.append(_path(["ax"] + hc("bolts"), ax))
    piece("bolts", bS)
    if web:                        # l'âme du poteau, en lignes cachées, par-dessus
        s.append(_path(["hd"] + hc("colWeb"), web))
    if opt.fabrication:
        _renvoi(S, s, xtete, yb1 - 0.3 * R.d_w, xtete + 1.6 * fs, yb1 - 1.9 * fs,
                js_str(R.n_b) + "×Ø" + f0(R.d_0))
    if opt.poignees:
        autres = N(u.N_Ed) != 0
        txtF = "VEd " + f0(N(u.V_Ed)) + (" +" if autres else "")
        wF = S.tw(txtF) + 0.5 * fs
        xF = (mx(x0 + (ln if notch else 0), xtete) + xE) / 2
        hitF = _rect(["hit"], -wF / 2, -1.05 * fs, wF, 1.42 * fs); hitF["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitF, _text([], None, None, txtF, fs)],
                        x=xF, y=mx(yt + dnt, yc) + 2.6 * fs, rot=0, grp="efforts", id="efforts"))
    # --- cotes
    S.dim(dict(id="tp", side="T", a=xf, b=x0, o1=yc, o2=yc, sym="tp", val=tp, key="tp_u", lvl=2, out="lo"))
    if dnt > 0:
        S.dim(dict(id="c", side="T", a=x0, b=x0 + ln, o1=yt + dnt, o2=yt, sym="c", val=ln, key="l_n", lvl=1))
    elif dnb > 0:
        S.dim(dict(id="c", side="B", a=x0, b=x0 + ln, o1=yb - dnb, o2=yb, sym="c", val=ln, key="l_n", lvl=1))
    # largeur de la bande du poteau : hP en cas (a), bP en cas (b) — l'autre
    # est cotée sur la vue en plan (jamais deux fois sur la planche)
    S.dim(dict(id="bP" if R.ame else "hP", side="B", a=-larg / 2, b=larg / 2,
               o1=yb + mar, o2=yb + mar, sym="b" if R.ame else "h", val=larg, lvl=1, calc=1))
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
    # étiquettes des cotes NULLES : poignées d'édition à l'écran (créer un
    # grugeage) — sans objet sur le plan de fabrication. Elles se posent au
    # BOUT de la poutre (alignées à droite sur xE) : au bord du plat elles
    # couvraient les têtes de boulon.
    if not opt.fabrication:
        if dnt == 0:
            S.tag(xE - S.tw("dc,sup 0") - 0.6 * fs, yt + R.tf_S + 1.2 * fs,
                  dict(id="dcs", sym="dc,sup", val=0, key="d_nt", lvl=2))
        if dnb == 0:
            S.tag(xE - S.tw("dc,inf 0") - 0.6 * fs, yb - R.tf_S - 0.7 * fs,
                  dict(id="dci", sym="dc,inf", val=0, key="d_nb", lvl=2))
    S.poignees(xtete + 1.2 * fs + S.largeur_poignees(txtG) / 2, (yb1 + ybl) / 2, "G", "n1_u", txtG, "bolts" in hot)
    S.name("T", xE - S.tw(_nom_portee(R)) / 2, _nom_portee(R))
    S.name("B", 0, _nom_poteau(R))
    return S.finish(s, None, "Élévation cotée de l'assemblage par platine d'about sur poteau")


# ----------------------------------------------------------------------- plan
def plan(R, opt):
    """VUE EN PLAN (coupe horizontale au niveau d'une rangée de boulons) — la
    vue porteuse : LE POTEAU EN COUPE (hachuré), la PLATINE coupée contre sa
    face d'appui, l'âme de la poutre portée en coupe avec ses deux cordons,
    les deux files de boulons (entraxe p3) de part et d'autre. En cas (b) :
    la poutre arrive ENTRE LES AILES, avec la recoupe des semelles et les
    jeux latéraux."""
    u = R.u
    xf = R.xf_P; tp = R.t_p; x0 = xf + tp; ws = R.tw_S / 2
    k, m, tr = visserie(R.d_b)
    xtete = x0 + tr + k; x_ap = xf - R.t_appui; xecrou = x_ap - tr - m
    hb = R.b_p / 2; y3 = R.p_3 / 2
    sec = _section_poteau(R)
    xcol = mx(*[abs(px) for px, _ in sec]); ycol = mx(*[abs(py) for _, py in sec])
    rec = N(u.rec_S); brec = mn(R.b_S, R.libre_ailes) if rec > 0 else R.b_S
    Hh = mx(hb + 24, ycol + 10)
    if opt.realiste:
        Hh = mx(Hh, R.b_S / 2 + 10)
    # la poutre portée doit se LIRE comme une poutre, pas comme un moignon :
    # au moins 0,30·h de web au-delà de l'about (relecture des captures)
    xE = mx(xtete + 70, xcol + 60, x0 + 0.30 * R.h_S)
    bb = dict(x1=-xcol - 30, x2=xE, y1=-Hh, y2=Hh)
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
        piece("beamS", [_poly(["ps"] + hc("beamS"), ptsS), _path(["ht"], _hachures(ptsS))])
    else:
        piece("beamS", [_rect(["ps"] + hc("beamS"), x0, -ws, xE - x0, R.tw_S)])
    # --- PLATINE coupée par le plan (hachures) contre la face d'appui
    pts_pl = [(xf, -hb), (x0, -hb), (x0, hb), (xf, hb)]
    pl = [_poly(["co"] + hc("plate"), [list(pt) for pt in pts_pl])]
    if opt.realiste:
        pl.append(_path(["ht", "htc"], _hachures(pts_pl)))
    piece("plate", pl)
    if opt.realiste:
        z = N(u.a_w) * math.sqrt(2)
        piece("weld", [_poly(["wb"] + hc("weld"), [[x0, -ws], [x0 + z, -ws], [x0, -ws - z]]),
                       _poly(["wb"] + hc("weld"), [[x0, ws], [x0 + z, ws], [x0, ws + z]])])
    # --- deux files de boulons : têtes côté platine, écrous à l'intérieur
    bS = []
    for y in (-y3, y3):
        if opt.realiste:
            bS += _boulon_profil(["bo"] + hc("bolts"), x_ap, x0, y, R.d_b, R.d_w)
    bS.append(_path((["ax"] if opt.realiste else ["bp"]) + hc("bolts"),
                    [[(xecrou - 6, y), (xtete + 6, y)] for y in (-y3, y3)]))
    piece("bolts", bS)
    # --- cotes : p3, e2 et bp à gauche (rappels courts depuis les écrous)
    S.dim(dict(id="p3", side="L", a=-y3, b=y3, o1=xecrou, o2=xecrou, sym="p3", val=R.p_3, key="p3_u", lvl=1))
    S.dim(dict(id="e2p", side="L", a=y3, b=hb, o1=xecrou, o2=xf, sym="e2", val=R.e_2, lvl=2, calc=1))
    S.dim(dict(id="bp", side="L", a=-hb, b=hb, o1=xf, o2=xf, sym="bp", val=R.b_p, key="bp_u", lvl=1))
    S.dim(dict(id="tp", side="T", a=xf, b=x0, o1=-hb, o2=-hb, sym="tp", val=tp, key="tp_u", lvl=1, out="lo"))
    S.dim(dict(id="twS", side="R", a=-ws, b=ws, o1=xE, o2=xE, sym="tw", val=R.tw_S, lvl=2, calc=1))
    # excentricité apportée au poteau : axe du poteau → point d'application
    S.dim(dict(id="ecol", side="B", a=0, b=x0, o1=ycol, o2=-ws, sym="e_col", val=R.e_col, lvl=2, calc=1))
    # la largeur que l'élévation ne porte pas (hP en cas b, bP en cas a)
    if R.ame:
        S.dim(dict(id="hP", side="R", a=-R.h_P / 2, b=R.h_P / 2, o1=xE, o2=xE,
                   sym="h", val=R.h_P, lvl=1, calc=1))
        S.dim(dict(id="twP", side="T", a=-R.tw_P / 2, b=R.tw_P / 2, o1=-ycol, o2=-ycol,
                   sym="tw", val=R.tw_P, lvl=2, calc=1))
        S.dim(dict(id="tfP", side="L", a=R.h_P / 2 - R.tf_P, b=R.h_P / 2, o1=-xcol, o2=-xcol,
                   sym="tf", val=R.tf_P, lvl=2, calc=1))
        # jeu latéral et recoupe : les deux grandeurs du passage entre ailes.
        # Le JEU LATÉRAL se cote contre la face intérieure de l'aile : c'est la
        # valeur SAISIE (j_lat) qui s'affiche, celle qui commande le passage
        # hP − 2·tfP − 2·j — et non l'écart résiduel, qui se lit sur le dessin
        yi = R.h_P / 2 - R.tf_P
        jl = N(u.j_lat)
        if jl > 0.01:
            S.dim(dict(id="jlat", side="R", a=yi - jl, b=yi, o1=mn(x0 + rec, xE), o2=mn(x0 + rec, xE),
                       sym="j", val=jl, key="j_lat", lvl=1, out="lo"))
        else:
            S.tag(mn(x0 + rec, xE) + 0.4 * fs, yi - 0.6 * fs,
                  dict(id="jlat", sym="j", val=jl, key="j_lat", lvl=1))
        if rec > 0:
            S.dim(dict(id="rec", side="B", a=x0, b=x0 + rec, o1=brec / 2, o2=R.b_S / 2,
                       sym="rc", val=rec, key="rec_S", lvl=1))
        else:
            S.tag(x0 + 0.4 * fs, -R.b_S / 2 - 0.6 * fs,
                  dict(id="rec", sym="rc", val=0, key="rec_S", lvl=1))
    else:
        S.dim(dict(id="bP", side="R", a=-R.b_P / 2, b=R.b_P / 2, o1=xE, o2=xE,
                   sym="b", val=R.b_P, lvl=1, calc=1))
        # la pince au bout d'aile se cote DU CÔTÉ de l'aile qu'elle mesure
        # (à gauche, elle se lisait à l'opposé du bout d'aile)
        S.dim(dict(id="eP", side="R", a=y3, b=R.b_P / 2, o1=xcol, o2=xcol,
                   sym="e", val=R.e_P, lvl=1, calc=1))
        S.dim(dict(id="tfP", side="T", a=R.h_P / 2 - R.tf_P, b=R.h_P / 2, o1=-ycol, o2=-ycol,
                   sym="tf", val=R.tf_P, lvl=2, calc=1))
        S.dim(dict(id="twP", side="L", a=-R.tw_P / 2, b=R.tw_P / 2, o1=-xcol, o2=-xcol,
                   sym="tw", val=R.tw_P, lvl=2, calc=1))
    S.name("T", 0, _nom_poteau(R))
    return S.finish(s, None, "Vue en plan cotée : poteau en coupe et platine d'about")


# ------------------------------------------------------------------ coupe A-A
def coupe_aa(R, opt):
    """COUPE A-A (regard le long de la poutre portée, vers le poteau) : en
    cas (a) la face EXTÉRIEURE de l'aile du poteau (largeur bP, rompue en
    haut et en bas) ; en cas (b) l'INTÉRIEUR du poteau — l'âme en face, les
    deux ailes par la tranche avec leurs congés, c'est la vue qui montre la
    LARGEUR DROITE entre congés. La PLATINE y est en VRAIE GRANDEUR avec ses
    deux files de boulons, les cordons de part et d'autre de l'âme portée, et
    la coupe de la poutre portée par-devant."""
    u = R.u
    yt = 0.0; zc = N(u.z_C); yc = yt + zc
    ws = R.tw_S / 2; hb = R.b_p / 2; y3 = R.p_3 / 2
    larg = R.h_P if R.ame else R.b_P
    mar = _marge_poteau(R)
    Wp = mx(larg / 2, hb, R.b_S / 2 + 6) + 30
    bb = dict(x1=-Wp, x2=Wp, y1=yt - mar, y2=yt + R.h_S + mar)
    fs = opt.fs_force or mx((bb["x2"] - bb["x1"] + 240) / 34, 9)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(cle):
        return ["hot"] if cle in hot else []

    y1c, y2c = yt - mar, yt + R.h_S + mar
    s.append(_poly(["pp"] + hc("col"), _rect_rompu(-larg / 2, y1c, larg, y2c - y1c, ("T", "B"), 0.55 * fs)))
    if R.ame:            # les deux ailes par la tranche, et la largeur droite
        xi = larg / 2 - R.tf_P
        s.append(_path(["fl2"] + hc("colFl"), [[(-xi, y1c), (-xi, y2c)], [(xi, y1c), (xi, y2c)]]))
        if R.r_P > 0:
            xc = xi - R.r_P
            s.append(_path(["flr"], [[(-xc, y1c), (-xc, y2c)], [(xc, y1c), (xc, y2c)]]))
    else:                # l'âme du poteau, derrière l'aile : deux traits fins
        xw = R.tw_P / 2
        s.append(_path(["fl2"] + hc("colWeb"), [[(-xw, y1c), (-xw, y2c)], [(xw, y1c), (xw, y2c)]]))
    # --- PLATINE en vraie grandeur + cordons (jambe a·√2 de part et d'autre
    # --- de l'âme portée)
    s.append(_rect(["co"] + hc("plate"), -hb, yc, R.b_p, R.h_p))
    z = N(u.a_w) * math.sqrt(2)
    s.append(_rect(["wb"] + hc("weld"), -ws - z, yc, z, R.h_p))
    s.append(_rect(["wb"] + hc("weld"), ws, yc, z, R.h_p))
    # --- boulons vus de face : rondelle, trou et traits d'axe
    yb1 = yc + R.e_1; ybl = yb1 + (R.n_1 - 1) * R.p_1; r0 = R.d_0 / 2
    for x in (-y3, y3):
        for i in range(R.n_1):
            s.append(_circle(["bw"], x, yb1 + i * R.p_1, R.d_w / 2))
            s.append(_circle(["bo"] + hc("bolts"), x, yb1 + i * R.p_1, r0))
    # --- coupe de la poutre portée par-devant (profil complet, au-delà du
    # --- grugeage et de la recoupe)
    sec = _section_portee(R, yt)
    s.append(_poly(["ps"] + hc("beamS"), sec))
    s.append(_path(["ht"], _hachures(sec)))
    dep = R.d_w / 2 + 5
    ax = [[(-y3 - dep, yb1 + i * R.p_1), (y3 + dep, yb1 + i * R.p_1)] for i in range(R.n_1)]
    ax += [[(x, yb1 - dep), (x, ybl + dep)] for x in (-y3, y3)]
    s.append(_path(["ax"] + hc("bolts"), ax))
    if opt.fabrication:
        _renvoi(S, s, y3 + r0 * 0.72, ybl + r0 * 0.72, y3 + r0 + 1.6 * fs, ybl + 1.9 * fs,
                js_str(R.n_b) + "×Ø" + f0(R.d_0))
    # --- cotes propres à cette vue
    S.dim(dict(id="p3", side="T", a=-y3, b=y3, o1=yb1, o2=yb1, sym="p3", val=R.p_3, key="p3_u", lvl=1))
    S.dim(dict(id="e2p", side="T", a=y3, b=hb, o1=yb1, o2=yc, sym="e2", val=R.e_2, lvl=2, calc=1))
    S.dim(dict(id="bp", side="B", a=-hb, b=hb, o1=yc + R.h_p, o2=yc + R.h_p, sym="bp", val=R.b_p, key="bp_u", lvl=1))
    S.dim(dict(id="e1", side="L", a=yc, b=yb1, o1=-hb, o2=-y3, sym="e1", val=R.e_1, key="e1_u", lvl=1, out="lo"))
    for i in range(1, R.n_1):
        S.dim(dict(id="p1", side="L", a=yb1 + (i - 1) * R.p_1, b=yb1 + i * R.p_1, o1=-y3, o2=-y3,
                   sym="p1", val=R.p_1, key="p1_u", lvl=1))
    S.dim(dict(id="hp", side="R", a=yc, b=yc + R.h_p, o1=hb, o2=hb, sym="hp", val=R.h_p, key="hp_u", lvl=1))
    S.dim(dict(id="bS", side="B", a=-R.b_S / 2, b=R.b_S / 2, o1=yt + R.h_S, o2=yt + R.h_S, sym="bS",
               val=R.b_S, lvl=1, calc=1))
    if R.ame:
        xc = larg / 2 - R.tf_P - R.r_P
        S.dim(dict(id="libreP", side="B", a=-xc, b=xc, o1=y2c, o2=y2c, sym="ℓ",
                   val=R.libre_droit, lvl=1, calc=1))
    S.name("T", 0, _nom_poteau(R))
    return S.finish(s, None, "Coupe A-A : platine en vraie grandeur et coupe de la poutre portée")


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

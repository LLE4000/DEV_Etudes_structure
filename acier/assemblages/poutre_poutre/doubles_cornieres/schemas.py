# -*- coding: utf-8 -*-
"""Schémas cotés de l'assemblage : élévation et vue en plan.

Un seul générateur, en Python pur, transcrit du 2ᵉ ``<script>`` du HTML de
référence (objet ``DCDraw`` : ``Sheet``, ``elev``, ``plan``) :

- ``Feuille``   : la feuille de cotation — rangées par côté (T haut, B bas,
                  L gauche, R droite), recherche d'un créneau libre sans
                  chevauchement, libellé déporté quand la cote est trop
                  courte, étiquettes pour les valeurs nulles, noms des
                  profilés, cartouche, étendue finale ;
- ``elevation`` / ``plan`` : les deux vues à l'échelle (mm), avec les
                  31 cotes identifiées, les trois niveaux de cotation, la
                  mise en rouge des cotes et éléments cités par une alerte,
                  la distinction cotes modifiables / grandeurs calculées et
                  le verrouillage en prédimensionnement.

Le générateur produit un ``Dessin`` : une scène de primitives (rectangles,
polygones, polylignes, cercles, textes, étiquettes de cote) en coordonnées
millimétriques (y vers le bas, comme le SVG), plus la boîte de vue et le
cartouche. Deux rendus lisent cette scène :

- ``Dessin.svg()`` — SVG autonome (style embarqué) pour l'écran ; les cotes
  modifiables portent ``data-key`` (composant cliquable et panneau des
  cotes) ;
- le peintre ReportLab de ``rapport.py`` — pour la note PDF, vectoriel.

Aucune valeur n'est recalculée ici : tout vient de ``R`` (moteur).
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from acier.js import N, mn, mx, js_str
from acier.formats import F, f0
from acier.bibliotheques import PERSO
from .entrees import PILOTEES_PAR_PREDIM

NIVEAUX = ("Vue simple", "Cotations principales", "Cotations complètes")

# Cotes du dessin → clé d'entrée (les 18 cotes modifiables)
CLE_PAR_COTE = {"e2b": "e2b_u", "p2S": "p2_S", "lhS": "lh_S", "gh": "g_h", "ln": "l_n",
                "e1S": "e1S_u", "p1S": "p1S_u", "Lc": "LC_u", "zc": "z_C", "dnt": "d_nt",
                "dnb": "d_nb", "e1P": "e1P_u", "p1P": "p1P_u", "dtop": "d_top", "aS": "a_S",
                "gA": "gA_u", "p2P": "p2_P", "aP": "a_P"}
COTES_CALCULEES = ("e2B", "z", "bB", "e1botS", "e1b", "he", "e1botP", "ztP", "e2A", "p3",
                   "bA", "tc", "twS")

# Palette des dessins : celle de la note de calcul (styles.py, 01_encre), avec
# les couleurs fonctionnelles du HTML (cote modifiable sur fond jaune, cordon
# orange, grandeur calculée en violet) — une seule charte pour l'écran et le PDF
PALETTE = dict(ink="#15181F", accent="#33415C", muted="#6E7480", ko="#9C3341",
               ext="#93A6B5", calc="#5B4A8A", weld="#D98A00",
               pp="#C3CED7", ps="#E6ECF0", hit="#F6F8FA", inbg="#FFF7CF",
               inink="#0B3D91", inbord="#C9B95A", hover="#FFE37A", hotbg="#FDECEA")


@dataclass
class Options:
    """Options de rendu : niveau de cotation (0, 1, 2), interactivité (cotes
    modifiables cliquables), mise en évidence ``hl`` (``dims`` et ``elems``
    d'une alerte), cotes verrouillées (prédimensionnement)."""
    lvl: int = 1
    interactive: bool = False
    hl: Optional[dict] = None
    locked: Optional[set] = None


def options_ecran(R, lvl=1, hl=None):
    """Options de l'écran : niveau choisi, cotes cliquables, alerte en cours,
    cotes pilotées verrouillées en prédimensionnement."""
    return Options(lvl=lvl, interactive=True, hl=hl,
                   locked=set(PILOTEES_PAR_PREDIM) if R.pred else None)


def options_rapport():
    """Options du rapport : cotations principales, sans interaction."""
    return Options(lvl=1, interactive=False, hl=None, locked=None)


@dataclass
class Dessin:
    """Une vue : boîte de vue ``(x1, y1, w, h)``, corps (primitives), cotes
    (primitives, dans l'ordre de pose), cartouche (textes), description."""
    viewbox: tuple
    corps: list
    cotes: list
    cartouche: list
    aria: str
    fs: float

    def svg(self, palette=None, largeur="100%", identifiant="dc"):
        return vers_svg(self, palette or PALETTE, largeur, identifiant)


# ----------------------------------------------------------------- primitives
def _rect(cls, x, y, w, h):
    return dict(t="rect", cls=cls, x=x, y=y, w=w, h=h)


def _poly(cls, pts):
    return dict(t="poly", cls=cls, pts=[tuple(p) for p in pts])


def _path(cls, lignes, hint=None):
    """Polylignes (liste de listes de points) ; ``hint`` = lettre SVG (H ou V)
    d'un segment de longueur nulle, pour rester identique à la référence."""
    return dict(t="path", cls=cls, lignes=[[tuple(p) for p in l] for l in lignes], hint=hint)


def _circle(cls, cx, cy, r):
    return dict(t="circle", cls=cls, cx=cx, cy=cy, r=r)


def _text(cls, x, y, txt, size, anchor="middle"):
    """``x``/``y`` None = position par défaut (texte d'étiquette transformée) ;
    ``anchor`` None = ancrage par défaut."""
    return dict(t="text", cls=cls, x=x, y=y, txt=txt, size=size, anchor=anchor)


def _group(cls, enfants, **attrs):
    return dict(t="g", cls=cls, enfants=enfants, **attrs)


# -------------------------------------------------------------------- Feuille
class Feuille:
    """Feuille de cotation (transcription de ``Sheet``)."""

    def __init__(self, bb, fs, opt):
        self.bb = bb; self.fs = fs; self.opt = opt; self.gap = 2.35 * fs
        self.rows = {"T": [], "B": [], "L": [], "R": []}; self.g = []
        self.o = {"T": bb["y1"] - 1.5 * fs, "B": bb["y2"] + 2.0 * fs,
                  "L": bb["x1"] - 1.4 * fs, "R": bb["x2"] + 2.0 * fs}
        self.ex = dict(x1=bb["x1"], x2=bb["x2"], y1=bb["y1"], y2=bb["y2"])

    def tw(self, t):
        return len(str(t)) * self.fs * 0.56

    def is_hot(self, id_):
        h = self.opt.hl
        return bool(h and h.get("dims") and id_ in h["dims"])

    def slot(self, side, lo, hi):
        rows = self.rows[side]; pad = 0.4 * self.fs
        for r in range(40):
            if r >= len(rows):
                rows.append([])
            ok = True
            for a, b in rows[r]:
                if lo < b + pad and hi > a - pad:
                    ok = False
                    break
            if ok:
                rows[r].append((lo, hi))
                return r
        return 0

    def pos(self, side, r):
        return self.o[side] + (-1 if side in ("T", "L") else 1) * r * self.gap

    def label(self, x, y, rot, txt, d):
        fs = self.fs; w = self.tw(txt); o = self.opt
        ed = bool(o.interactive and d.get("key") and not (o.locked and d["key"] in o.locked))
        hot = self.is_hot(d["id"])
        cls = ["dl"] + (["ed"] if ed else []) + (["hot"] if hot else []) + (["calc"] if d.get("calc") else [])
        hit = _rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs)
        hit["rx"] = 0.2 * fs
        tx = _text([], None, None, txt, fs)
        return _group(cls, [hit, tx], x=x, y=y, rot=-90 if rot else 0,
                      key=d.get("key") if ed else None, sym=d.get("sym"), id=d["id"])

    def dim(self, d):
        """d : id, side T|B|L|R, a, b, o1, o2, sym, val, key, lvl, calc, out."""
        hot = self.is_hot(d["id"])
        if not (d["lvl"] <= self.opt.lvl or hot):
            return
        if not abs(d["b"] - d["a"]) > 0.01:
            return
        fs = self.fs; lo = mn(d["a"], d["b"]); hi = mx(d["a"], d["b"]); ln = hi - lo
        t1 = d["sym"] + " = " + f0(d["val"]); t2 = d["sym"] + " " + f0(d["val"]); l2 = lo; h2 = hi
        if ln >= self.tw(t1) + 0.9 * fs:
            txt = t1; c = (lo + hi) / 2
        elif ln >= self.tw(t2) + 0.9 * fs:
            txt = t2; c = (lo + hi) / 2
        else:
            txt = t2; w = self.tw(t2)
            if d.get("out") == "lo":
                c = lo - 0.75 * fs - w / 2; l2 = lo - 1.0 * fs - w
            else:
                c = hi + 0.75 * fs + w / 2; h2 = hi + 1.0 * fs + w
        r = self.slot(d["side"], l2, h2); p = self.pos(d["side"], r)
        H = d["side"] in ("T", "B"); t = 0.34 * fs
        calc = bool(d.get("calc"))
        cls_ext = ["ext"]
        cls_dln = ["dln"]
        a, b, o1, o2 = d["a"], d["b"], d["o1"], d["o2"]
        if H:
            e = p + (-0.45 if d["side"] == "T" else 0.45) * fs
            self.ex["x1"] = mn(self.ex["x1"], l2); self.ex["x2"] = mx(self.ex["x2"], h2)
            ext = _path(cls_ext, [[(a, o1), (a, e)], [(b, o2), (b, e)]], hint="V")
            dln = _path(cls_dln, [[(l2, p), (h2, p)], [(a - t, p + t), (a + t, p - t)], [(b - t, p + t), (b + t, p - t)]], hint="H")
            lab = self.label(c, p - 0.42 * fs, False, txt, d)
        else:
            e = p + (-0.45 if d["side"] == "L" else 0.45) * fs
            self.ex["y1"] = mn(self.ex["y1"], l2); self.ex["y2"] = mx(self.ex["y2"], h2)
            ext = _path(cls_ext, [[(o1, a), (e, a)], [(o2, b), (e, b)]], hint="H")
            dln = _path(cls_dln, [[(p, l2), (p, h2)], [(p - t, a + t), (p + t, a - t)], [(p - t, b + t), (p + t, b - t)]], hint="V")
            lab = self.label(p - 0.42 * fs, c, True, txt, d)
        self.g.append(_group(["dm"] + (["hot"] if hot else []) + (["calc"] if calc else []), [ext, dln, lab]))

    def name(self, side, c, txt):
        fs = self.fs; w = self.tw(txt) * 1.08
        r = self.slot(side, c - w / 2, c + w / 2); p = self.pos(side, r)
        self.ex["x1"] = mn(self.ex["x1"], c - w / 2); self.ex["x2"] = mx(self.ex["x2"], c + w / 2)
        self.g.append(_text(["tx"], c, p + (0.3 if side == "B" else -0.1) * fs, txt, fs))

    def tag(self, x, y, d):
        hot = self.is_hot(d["id"])
        if not (d["lvl"] <= self.opt.lvl or hot):
            return
        txt = d["sym"] + " " + f0(d["val"])
        self.g.append(_group(["dm"] + (["hot"] if hot else []),
                             [self.label(x + self.tw(txt) / 2 + 0.3 * self.fs, y, False, txt, d)]))

    def finish(self, body, lines, aria):
        fs = self.fs; n = self.rows; ex = self.ex
        x1 = mn(ex["x1"], self.pos("L", len(n["L"]) - 1) - 1.7 * fs if n["L"] else ex["x1"]) - 0.6 * fs
        x2 = mx(ex["x2"], self.pos("R", len(n["R"]) - 1) + 0.5 * fs if n["R"] else ex["x2"]) + 0.6 * fs
        y1 = mn(ex["y1"], self.pos("T", len(n["T"]) - 1) - 1.7 * fs if n["T"] else ex["y1"]) - 0.6 * fs
        y2 = mx(ex["y2"], self.pos("B", len(n["B"]) - 1) + 0.6 * fs if n["B"] else ex["y2"]) + 0.6 * fs
        cart = []
        if lines:
            fc = fs * 0.95
            maxc = max(24, math.floor((x2 - x1 - 1.2 * fs) / (fc * 0.56)))
            wr = []
            for l in lines:
                cur = ""
                for part in l.split(" – "):
                    t = cur + " – " + part if cur else part
                    if cur and len(t) > maxc:
                        wr.append(cur); cur = "   " + part
                    else:
                        cur = t
                wr.append(cur)
            lines = wr
            wmax = 0
            for l in lines:
                wmax = mx(wmax, len(l) * fc * 0.56)
            x2 = mx(x2, x1 + wmax + 1.2 * fs)
            cart.append(_path(["ext"], [[(x1, y2 + 0.25 * fc), (x2, y2 + 0.25 * fc)]]))
            for i, l in enumerate(lines):
                cart.append(_text([], x1 + 0.6 * fs, y2 + (i + 1) * 1.5 * fc, l, fc, None))
            y2 += (len(lines) + 0.7) * 1.5 * fc
        return Dessin(viewbox=(x1, y1, x2 - x1, y2 - y1), corps=body, cotes=self.g,
                      cartouche=cart, aria=aria, fs=fs)


# ------------------------------------------------------------------ élévation
def elevation(R, opt):
    """Élévation cotée (transcription de ``DCDraw.elev``)."""
    u = R.u; xf = R.tw_P / 2; gh = N(u.g_h); x0 = xf + gh; yt = N(u.d_top); yb = yt + R.h_S
    zc = N(u.z_C); yc = yt + zc; dnt = N(u.d_nt); dnb = N(u.d_nb); ln = N(u.l_n)
    p2S = N(u.p2_S); lhS = N(u.lh_S)
    notch = dnt > 0 or dnb > 0
    xE = mx(x0 + (ln if notch else 0), xf + R.b_B) + 75
    bb = dict(x1=-R.b_P / 2, x2=xE, y1=mn(0, yt), y2=mx(R.h_P, yb))
    fs = mx((bb["x2"] - bb["x1"] + 260) / 34, 9)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(k):
        return ["hot"] if k in hot else []

    s.append(_rect(["pp"] + hc("flPt"), -R.b_P / 2, 0, R.b_P, R.tf_P))
    s.append(_rect(["pp"] + hc("flPb"), -R.b_P / 2, R.h_P - R.tf_P, R.b_P, R.tf_P))
    s.append(_rect(["pp"], -xf, R.tf_P, R.tw_P, R.h_P - 2 * R.tf_P))
    p = [[x0, yt + dnt], [x0 + ln, yt + dnt], [x0 + ln, yt], [xE, yt]] if dnt > 0 else [[x0, yt], [xE, yt]]
    p = p + ([[xE, yb], [x0 + ln, yb], [x0 + ln, yb - dnb], [x0, yb - dnb]] if dnb > 0 else [[xE, yb], [x0, yb]])
    s.append(_poly(["ps"] + hc("beamS"), p))
    s.append(_path(["fl2"], [[(x0 + ln if dnt > 0 else x0, yt + R.tf_S), (xE, yt + R.tf_S)],
                             [(x0 + ln if dnb > 0 else x0, yb - R.tf_S), (xE, yb - R.tf_S)]]))
    if "notchT" in hot:
        if dnt > 0:
            s.append(_path(["hl"], [[(x0, yt + dnt), (x0 + ln, yt + dnt), (x0 + ln, yt)]]))
        else:
            s.append(_path(["hl"], [[(x0, yt + R.tf_S + R.r_S), (x0, yt), (mx(R.b_P / 2, x0 + 40), yt)]]))
    if "notchB" in hot:
        if dnb > 0:
            s.append(_path(["hl"], [[(x0, yb - dnb), (x0 + ln, yb - dnb), (x0 + ln, yb)]]))
        else:
            s.append(_path(["hl"], [[(x0, yb - R.tf_S - R.r_S), (x0, yb), (mx(R.b_P / 2, x0 + 40), yb)]]))
    s.append(_rect(["co"] + hc("cleat"), xf, yc, R.b_B, R.L_C))
    s.append(_rect(["co2"], xf, yc, R.t_C, R.L_C))
    xc1 = xf + R.g_B; xcl = xc1 + (R.n2_S - 1) * p2S; yb1 = yc + R.e1_S
    ybl = yb1 + (R.n1_S - 1) * R.p1_S; xt = xf + R.b_B; r0 = R.d_0 / 2
    if R.bolt_S:
        for i in range(R.n1_S):
            for j in range(R.n2_S):
                cx = xc1 + j * p2S; cy = yb1 + i * R.p1_S
                s.append(_circle(["bo"] + hc("boltsS"), cx, cy, r0))
                s.append(_path(["cm"], [[(cx - r0 - 4, cy), (cx + r0 + 4, cy)], [(cx, cy - r0 - 4), (cx, cy + r0 + 4)]]))
    else:
        s.append(_path(["we"], [[(xt - lhS, yc), (xt, yc), (xt, yc + R.L_C), (xt - lhS, yc + R.L_C)]]))
    yp1 = yc + R.e1_P; ypl = yp1 + (R.n1_P - 1) * R.p1_P
    if R.bolt_P:
        for i in range(R.n1_P):
            s.append(_path(["bp"] + hc("boltsP"), [[(-xf - 16, yp1 + i * R.p1_P), (xf + R.t_C + 16, yp1 + i * R.p1_P)]]))
    # --- cotes : chaînes au plus près, cotes d'ensemble ensuite
    if R.bolt_S:
        S.dim(dict(id="e2b", side="T", a=x0, b=xc1, o1=yt + dnt, o2=yb1, sym="e2,b", val=R.e2b_S, key="e2b_u", lvl=1))
        if R.n2_S > 1:
            S.dim(dict(id="p2S", side="T", a=xc1, b=xcl, o1=yb1, o2=yb1, sym="p2", val=p2S, key="p2_S", lvl=1))
        S.dim(dict(id="e2B", side="T", a=xcl, b=xt, o1=yb1, o2=yc, sym="e2", val=R.e2a_S, lvl=2, calc=1))
    elif lhS > 0:
        S.dim(dict(id="lhS", side="T", a=xt - lhS, b=xt, o1=yc, o2=yc, sym="lh", val=lhS, key="lh_S", lvl=1))
    S.dim(dict(id="gh", side="T", a=xf, b=x0, o1=yc, o2=yt + dnt, sym="gh", val=gh, key="g_h", lvl=2, out="lo"))
    S.dim(dict(id="z", side="T", a=xf, b=xf + R.zeff, o1=yc, o2=yb1 if R.bolt_S else yc, sym="z", val=R.zeff, lvl=1, calc=1))
    if dnt > 0:
        S.dim(dict(id="ln", side="T", a=x0, b=x0 + ln, o1=yt + dnt, o2=yt, sym="ln", val=ln, key="l_n", lvl=1))
    elif dnb > 0:
        S.dim(dict(id="ln", side="B", a=x0, b=x0 + ln, o1=yb - dnb, o2=yb, sym="ln", val=ln, key="l_n", lvl=1))
    S.dim(dict(id="bB", side="B", a=xf, b=xt, o1=yc + R.L_C, o2=yc + R.L_C, sym="bB", val=R.b_B, lvl=2, calc=1))
    if R.bolt_S:
        S.dim(dict(id="e1S", side="R", a=yc, b=yb1, o1=xt, o2=xcl, sym="e1", val=R.e1_S, key="e1S_u", lvl=1, out="lo"))
        for i in range(1, R.n1_S):
            S.dim(dict(id="p1S", side="R", a=yb1 + (i - 1) * R.p1_S, b=yb1 + i * R.p1_S, o1=xcl, o2=xcl, sym="p1", val=R.p1_S, key="p1S_u", lvl=1))
        S.dim(dict(id="e1botS", side="R", a=ybl, b=yc + R.L_C, o1=xcl, o2=xt, sym="e1'", val=R.e1bot_S, lvl=2, calc=1))
    S.dim(dict(id="Lc", side="R", a=yc, b=yc + R.L_C, o1=xt, o2=xt, sym="Lc", val=R.L_C, key="LC_u", lvl=1))
    S.dim(dict(id="zc", side="R", a=yt, b=yc, o1=xE, o2=xt, sym="zc", val=zc, key="z_C", lvl=1, out="lo"))
    if dnt > 0:
        S.dim(dict(id="dnt", side="R", a=yt, b=yt + dnt, o1=x0 + ln, o2=x0 + ln, sym="dnt", val=dnt, key="d_nt", lvl=1, out="lo"))
    if dnb > 0:
        S.dim(dict(id="dnb", side="R", a=yb - dnb, b=yb, o1=x0 + ln, o2=x0 + ln, sym="dnb", val=dnb, key="d_nb", lvl=1))
    if R.bolt_S:
        S.dim(dict(id="e1b", side="R", a=yt + dnt, b=yb1, o1=x0 + (ln if dnt > 0 else 0), o2=xcl, sym="e1,b", val=R.e1b_S, lvl=2, calc=1, out="lo"))
        if dnb > 0:
            S.dim(dict(id="he", side="R", a=ybl, b=yb - dnb, o1=xcl, o2=x0 + ln, sym="he", val=R.h_e, lvl=2, calc=1))
    if R.bolt_P:
        lp = 2 if (R.bolt_S and R.n1_P == R.n1_S and R.p1_P == R.p1_S and R.e1_P == R.e1_S) else 1
        S.dim(dict(id="e1P", side="L", a=yc, b=yp1, o1=xf, o2=-xf, sym="e1", val=R.e1_P, key="e1P_u", lvl=lp, out="lo"))
        for i in range(1, R.n1_P):
            S.dim(dict(id="p1P", side="L", a=yp1 + (i - 1) * R.p1_P, b=yp1 + i * R.p1_P, o1=-xf, o2=-xf, sym="p1", val=R.p1_P, key="p1P_u", lvl=lp))
        S.dim(dict(id="e1botP", side="L", a=ypl, b=yc + R.L_C, o1=-xf, o2=xf, sym="e1'", val=R.e1bot_P, lvl=2, calc=1))
        S.dim(dict(id="ztP", side="L", a=0, b=yp1, o1=-R.b_P / 2, o2=-xf, sym="zt", val=R.zt_P, lvl=2, calc=1))
    S.dim(dict(id="dtop", side="L", a=0, b=yt, o1=-R.b_P / 2, o2=x0, sym="déc.", val=yt, key="d_top", lvl=2, out="lo"))
    # paramètres nuls : pas de cote possible, donc une étiquette éditable
    if dnt == 0:
        S.tag(xt + 0.4 * fs, yt + R.tf_S + 1.5 * fs, dict(id="dnt", sym="dnt", val=0, key="d_nt", lvl=2))
    if dnb == 0:
        S.tag(xt + 0.4 * fs, yb - R.tf_S - 0.7 * fs, dict(id="dnb", sym="dnb", val=0, key="d_nb", lvl=2))
    if yt == 0:
        S.tag(-R.b_P / 2, -0.55 * fs, dict(id="dtop", sym="déc.", val=0, key="d_top", lvl=2))
    if not R.bolt_S:
        S.tag(xt + 0.4 * fs, yc + R.L_C / 2, dict(id="aS", sym="a", val=N(u.a_S), key="a_S", lvl=1))
    nomS = "Poutre secondaire" if u.prof_S == PERSO else u.prof_S
    S.name("T", xE - S.tw(nomS) / 2, nomS)
    S.name("B", 0, "Poutre principale" if u.prof_P == PERSO else u.prof_P)
    L = []
    if opt.lvl >= 1:
        L.append("Cornières : 2 × " + R.corn_txt + " – " + u.nu_C + " – Lc " + f0(R.L_C) + " mm")
        if R.bolt_S or R.bolt_P:
            L.append("Boulons " + R.boulon + " – classe " + u.classe + " – trous d0 " + f0(R.d_0) + " mm"
                     + (" – groupe S : " + js_str(R.n1_S) + " × " + js_str(R.n2_S) if R.bolt_S else "")
                     + (" – groupe P : 2 × (" + js_str(R.n1_P) + " × " + js_str(R.n2_P) + ")" if R.bolt_P else ""))
        if not R.bolt_S:
            L.append("Soudure ailes B : a " + f0(N(u.a_S)) + " mm, cordon vertical Lc + retours " + f0(lhS) + " mm")
        if not R.bolt_P:
            L.append("Soudure ailes A : a " + f0(N(u.a_P)) + " mm, cordon vertical Lc + retours " + f0(N(u.lh_P)) + " mm")
        L.append("Excentricité de calcul z = " + f0(R.zeff) + " mm – MS = " + F(R.M_S, 2) + " kNm"
                 + ((" – eP = " + f0(R.e_P if R.bolt_P else R.ew_P) + " mm, MP = " + F(R.M_P, 2) + " kNm") if R.M_P > 0 else ""))
        if notch:
            L.append("Grugeage : " + ("sup. " + f0(dnt) if dnt > 0 else "") + (" / " if dnt > 0 and dnb > 0 else "")
                     + ("inf. " + f0(dnb) if dnb > 0 else "") + " × " + f0(ln) + " mm – bras de levier gh + ln = " + f0(gh + ln) + " mm")
    return S.finish(s, L, "Élévation cotée de l'assemblage")


# ----------------------------------------------------------------------- plan
def plan(R, opt):
    """Vue en plan cotée (transcription de ``DCDraw.plan``)."""
    u = R.u; xf = R.tw_P / 2; gh = N(u.g_h); x0 = xf + gh; ws = R.tw_S / 2; xt = xf + R.b_B
    xE = xt + 60; Hh = ws + R.b_A + 22; p2S = N(u.p2_S); p2P = N(u.p2_P)
    bb = dict(x1=-xf - 30, x2=xE, y1=-Hh, y2=Hh)
    fs = mx((bb["x2"] - bb["x1"] + 210) / 32, 8)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(k):
        return ["hot"] if k in hot else []

    s.append(_rect(["pp"], -xf, -Hh, R.tw_P, 2 * Hh))
    s.append(_rect(["ps"] + hc("beamS"), x0, -ws, xE - x0, R.tw_S))
    for g in (-1, 1):
        y1 = g * ws; y2 = g * (ws + R.t_C); y3 = g * (ws + R.b_A)
        s.append(_poly(["co"] + hc("cleat"), [[xf, y1], [xt, y1], [xt, y2], [xf + R.t_C, y2], [xf + R.t_C, y3], [xf, y3]]))
        if R.bolt_P:
            for i in range(R.n2_P):
                yy = g * (ws + R.g_A + i * p2P)
                s.append(_path(["bp"] + hc("boltsP"), [[(-xf - 14, yy), (xf + R.t_C + 14, yy)]]))
        else:
            s.append(_circle(["wd"], xf + R.t_C, y3, mx(N(u.a_P), 4)))
        if not R.bolt_S:
            s.append(_circle(["wd"], xt, y2, mx(N(u.a_S), 4)))
    xc1 = xf + R.g_B; xcl = xc1 + (R.n2_S - 1) * p2S; yo = -ws - R.t_C
    if R.bolt_S:
        for i in range(R.n2_S):
            s.append(_path(["bp"] + hc("boltsS"), [[(xc1 + i * p2S, yo - 14), (xc1 + i * p2S, -yo + 14)]]))
    if R.bolt_S:
        S.dim(dict(id="e2b", side="T", a=x0, b=xc1, o1=-ws, o2=yo - 14, sym="e2,b", val=R.e2b_S, key="e2b_u", lvl=2))
        if R.n2_S > 1:
            S.dim(dict(id="p2S", side="T", a=xc1, b=xcl, o1=yo - 14, o2=yo - 14, sym="p2", val=p2S, key="p2_S", lvl=2))
    S.dim(dict(id="gh", side="T", a=xf, b=x0, o1=-ws - R.b_A, o2=-ws, sym="gh", val=gh, key="g_h", lvl=1, out="lo"))
    S.dim(dict(id="bB", side="T", a=xf, b=xt, o1=-ws - R.b_A, o2=yo, sym="bB", val=R.b_B, lvl=1, calc=1))
    if R.bolt_P:
        ya = -(ws + R.g_A); yl = ya - (R.n2_P - 1) * p2P; xo = xf + R.t_C + 14
        S.dim(dict(id="gA", side="R", a=ya, b=-ws, o1=xo, o2=xt, sym="gA", val=R.g_A, key="gA_u", lvl=1, out="lo"))
        if R.n2_P > 1:
            S.dim(dict(id="p2P", side="R", a=yl, b=ya, o1=xo, o2=xo, sym="p2", val=p2P, key="p2_P", lvl=1, out="lo"))
        S.dim(dict(id="e2A", side="R", a=-(ws + R.b_A), b=yl, o1=xf + R.t_C, o2=xo, sym="e2", val=R.e2a_P, lvl=2, calc=1, out="lo"))
        S.dim(dict(id="p3", side="L", a=ya, b=-ya, o1=-xf - 14, o2=-xf - 14, sym="p3", val=R.p_3, lvl=1, calc=1))
    else:
        S.tag(xf + R.t_C + 0.6 * fs, -(ws + R.b_A) - 0.2 * fs, dict(id="aP", sym="a", val=N(u.a_P), key="a_P", lvl=1))
    S.dim(dict(id="bA", side="R", a=ws, b=ws + R.b_A, o1=xt, o2=xf + R.t_C, sym="bA", val=R.b_A, lvl=1, calc=1))
    S.dim(dict(id="tc", side="R", a=ws, b=ws + R.t_C, o1=xt, o2=xt, sym="tc", val=R.t_C, lvl=2, calc=1))
    S.dim(dict(id="twS", side="R", a=-ws, b=ws, o1=xE, o2=xE, sym="tw", val=R.tw_S, lvl=2, calc=1))
    return S.finish(s, None, "Vue en plan cotée de l'assemblage")


def dessins(R, opt):
    """Les deux vues."""
    return elevation(R, opt), plan(R, opt)


def cotes_modifiables(dessin):
    """Les cotes cliquables d'un dessin : ``[(clé, symbole, id), …]`` dans
    l'ordre de pose (pour le panneau des cotes)."""
    out = []

    def rec(prims):
        for p in prims:
            if p["t"] == "g":
                if p.get("key"):
                    out.append((p["key"], p.get("sym"), p.get("id")))
                rec(p["enfants"])
    rec(dessin.cotes)
    return out


# ------------------------------------------------------------------ rendu SVG
def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _n(x):
    return js_str(x)


def _css(p, pre):
    """Feuille de style embarquée (une seule charte : palette de la note)."""
    ink, acc, ko = p["ink"], p["accent"], p["ko"]
    return (
        f"#{pre} *{{vector-effect:non-scaling-stroke}}"
        f"#{pre} .pp{{fill:{p['pp']};stroke:{ink};stroke-width:1.4}}"
        f"#{pre} .ps{{fill:{p['ps']};stroke:{ink};stroke-width:1.4}}"
        f"#{pre} .fl2{{stroke:{ink};stroke-width:.8;fill:none}}"
        f"#{pre} .co{{fill:{acc};fill-opacity:.28;stroke:{acc};stroke-width:1.6}}"
        f"#{pre} .co2{{fill:{acc};opacity:.6}}"
        f"#{pre} .bo{{fill:#fff;stroke:{ink};stroke-width:1.5}}"
        f"#{pre} .bp{{stroke:{ko};stroke-width:2.4;stroke-dasharray:7 3;fill:none}}"
        f"#{pre} .we{{stroke:{p['weld']};stroke-width:5;fill:none}}"
        f"#{pre} .wd{{fill:{p['weld']}}}"
        f"#{pre} .dm path{{stroke:{acc};stroke-width:1;fill:none}}"
        f"#{pre} .ext{{stroke:{p['ext']};stroke-width:.6;fill:none}}"
        f"#{pre} .dm text,#{pre} .tx,#{pre} .cart text{{fill:{acc};font-family:system-ui,Arial,sans-serif}}"
        f"#{pre} .tx{{fill:{ink};font-weight:600}}"
        f"#{pre} .cart text{{fill:{ink}}}"
        f"#{pre} .dl .hit{{fill:{p['hit']};fill-opacity:.9;stroke:none}}"
        f"#{pre} .dl.ed{{cursor:pointer}}"
        f"#{pre} .dl.ed .hit{{fill:{p['inbg']};fill-opacity:1;stroke:{p['inbord']};stroke-width:1}}"
        f"#{pre} .dl.ed text{{fill:{p['inink']};font-weight:600}}"
        f"#{pre} .dl.ed:hover .hit,#{pre} .dl.ed:focus .hit{{fill:{p['hover']};stroke:{acc};stroke-width:2}}"
        f"#{pre} .dl.ed:focus{{outline:none}}"
        f"#{pre} .dm.calc path.dln{{stroke:{p['calc']};stroke-dasharray:5 3}}"
        f"#{pre} .dl.calc text{{fill:{p['calc']};font-style:italic}}"
        f"#{pre} .dm.hot path.dln,#{pre} .dm.hot path.ext{{stroke:{ko};stroke-width:2}}"
        f"#{pre} .dl.hot text{{fill:{ko};font-weight:700}}"
        f"#{pre} .dl.hot .hit{{fill:{p['hotbg']};fill-opacity:1;stroke:{ko};stroke-width:2}}"
        f"#{pre} .pp.hot,#{pre} .ps.hot,#{pre} .co.hot,#{pre} .bo.hot{{stroke:{ko};stroke-width:3}}"
        f"#{pre} .co.hot{{fill:{ko};fill-opacity:.2}}"
        f"#{pre} .bp.hot{{stroke-width:4.5;stroke-dasharray:none}}"
        f"#{pre} .hl{{stroke:{ko};stroke-width:4;fill:none}}"
        f"#{pre} .cm{{stroke:{ink};stroke-width:.6;fill:none}}"
    )


def _path_d(lignes, hint=None):
    parts = []
    for l in lignes:
        (x, y) = l[0]
        parts.append("M" + _n(x) + " " + _n(y))
        px, py = x, y
        for (x, y) in l[1:]:
            if x == px and y == py and hint:
                parts.append(hint + _n(x if hint == "H" else y))
            elif y == py:
                parts.append("H" + _n(x))
            elif x == px:
                parts.append("V" + _n(y))
            else:
                parts.append("L" + _n(x) + " " + _n(y))
            px, py = x, y
    return "".join(parts)


def _prim_svg(p):
    cls = " ".join(p["cls"])
    ca = f' class="{cls}"' if cls else ""
    t = p["t"]
    if t == "rect":
        rx = f' rx="{_n(p["rx"])}"' if p.get("rx") else ""
        return f'<rect class="{cls}" x="{_n(p["x"])}" y="{_n(p["y"])}" width="{_n(p["w"])}" height="{_n(p["h"])}"{rx}/>'
    if t == "poly":
        return f'<polygon class="{cls}" points="{" ".join(_n(x) + "," + _n(y) for x, y in p["pts"])}"/>'
    if t == "path":
        return f'<path{ca} d="{_path_d(p["lignes"], p.get("hint"))}"/>'
    if t == "circle":
        return f'<circle class="{cls}" cx="{_n(p["cx"])}" cy="{_n(p["cy"])}" r="{_n(p["r"])}"/>'
    if t == "text":
        pos = f' x="{_n(p["x"])}" y="{_n(p["y"])}"' if p.get("x") is not None else ""
        anc = f' text-anchor="{p["anchor"]}"' if p.get("anchor") else ""
        return (f'<text{ca}{pos}{anc} font-size="{_n(p["size"])}">'
                f'{_esc(p["txt"])}</text>')
    if t == "g":
        attrs = ""
        if p.get("key"):
            attrs += (f' data-key="{p["key"]}" data-sym="{_esc(p["sym"])}" tabindex="0" role="button"'
                      f' aria-label="Modifier {_esc(p["sym"])}"')
        if "x" in p:
            attrs += f' transform="translate({_n(p["x"])} {_n(p["y"])})' + (" rotate(-90)" if p.get("rot") else "") + '"'
        if p.get("id") and "x" in p:
            attrs += f' data-dim="{p["id"]}"'
        return f'<g{ca}{attrs}>' + "".join(_prim_svg(e) for e in p["enfants"]) + "</g>"
    raise ValueError(t)


def vers_svg(d, palette=PALETTE, largeur="100%", identifiant="dc"):
    """SVG autonome de la vue ``d`` (style embarqué)."""
    x1, y1, w, h = d.viewbox
    corps = "".join(_prim_svg(p) for p in d.corps)
    cotes = "".join(_prim_svg(p) for p in d.cotes)
    cart = ('<g class="cart">' + "".join(_prim_svg(p) for p in d.cartouche) + "</g>") if d.cartouche else ""
    return (f'<svg id="{identifiant}" xmlns="http://www.w3.org/2000/svg" viewBox="{_n(x1)} {_n(y1)} {_n(w)} {_n(h)}"'
            f' width="{largeur}" role="img" aria-label="{_esc(d.aria)}">'
            f"<style>{_css(palette, identifiant)}</style>" + corps + cotes + cart + "</svg>")

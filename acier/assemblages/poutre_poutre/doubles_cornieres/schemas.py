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
INF_MM = 1e9

# Notations affichées : symbole du moteur de référence → notation de
# l'Eurocode (EN 1993-1-8 Tableau 3.3 ; hc, c, dc : conventions des guides
# SCI / MSB pour ce que l'Eurocode ne nomme pas). Les clés de calcul ne
# changent pas ; seule l'écriture change — le test de parité des dessins
# repasse en notation du moteur (``Options(notation={})``).
NOTATION_EC = {"Lc": "hc", "ln": "c", "dnt": "dc,sup", "dnb": "dc,inf", "déc.": "Δz", "lh": "ℓh"}

# Groupes de boulons : clé du nombre de rangées, libellé de la fenêtre
GROUPES_BOULONS = {"S": dict(n1="n1S_u", n2="n2S_u", titre="Groupe S – âme de la poutre secondaire"),
                   "P": dict(n1="n1P_u", n2="n2P_u", titre="Groupe P – âme de la poutre principale (par cornière)")}

# Cotes du dessin → clé d'entrée (les 18 cotes modifiables)
CLE_PAR_COTE = {"e2b": "e2b_u", "p2S": "p2_S", "lhS": "lh_S", "gh": "g_h", "ln": "l_n",
                "e1S": "e1S_u", "p1S": "p1S_u", "Lc": "LC_u", "zc": "z_C", "dnt": "d_nt",
                "dnb": "d_nb", "e1P": "e1P_u", "p1P": "p1P_u", "dtop": "d_top", "aS": "a_S",
                "gA": "gA_u", "p2P": "p2_P", "aP": "a_P",
                "lhP": "lh_P"}       # étiquette du mode édition (retours des cordons A)
COTES_CALCULEES = ("e2B", "z", "bB", "e1botS", "e1b", "he", "e1botP", "ztP", "e2A", "p3",
                   "bA", "tc", "twS")

# Palette des dessins : celle de la note de calcul (styles.py, 01_encre), avec
# les couleurs fonctionnelles du HTML (cote modifiable sur fond jaune, cordon
# orange, grandeur calculée en violet) — une seule charte pour l'écran et le PDF
PALETTE = dict(ink="#15181F", accent="#33415C", muted="#6E7480", ko="#9C3341",
               ext="#93A6B5", calc="#5B4A8A", weld="#D98A00", weld2="#A86B00",
               pp="#C3CED7", ps="#E6ECF0", hit="#F6F8FA", inbg="#FFF7CF",
               inink="#0B3D91", inbord="#C9B95A", hover="#FFE37A", hotbg="#FDECEA",
               hatch="#7F8C9A")

PAS_HACHURES = 5.0          # mm, hachures à 45° des parties coupées
SEGMENTS_ARC = 6            # segments par quart de cercle (congés)


@dataclass
class Options:
    """Options de rendu : niveau de cotation (0, 1, 2), interactivité (cotes
    modifiables cliquables), mise en évidence ``hl`` (``dims`` et ``elems``
    d'une alerte), cotes verrouillées (prédimensionnement), ``editables``
    (toutes les cotes modifiables affichées quel que soit le niveau),
    ``poignees`` (+ / − et étiquette cliquable des groupes de boulons),
    ``notation`` (symboles affichés ; ``{}`` = ceux du moteur de référence)."""
    lvl: int = 1
    interactive: bool = False
    hl: Optional[dict] = None
    locked: Optional[set] = None
    editables: bool = False
    poignees: bool = False
    notation: dict = field(default_factory=lambda: dict(NOTATION_EC))
    # rendu réaliste : congés réels (âme–semelle, racine et bouts des
    # cornières, rayon du grugeage), hachures des parties coupées, cordons
    # à leur taille (a·√2), rondelles ; False = géométrie du HTML (parité)
    realiste: bool = True
    # cartouche (lignes sous l'élévation) : utile à l'écran ; sur la note il
    # répéterait la ligne de données et les hypothèses, donc False au rapport
    cartouche: bool = True


def options_ecran(R, lvl=1, hl=None):
    """Options de l'écran : niveau choisi, toutes les cotes modifiables
    visibles et cliquables, poignées des groupes, alerte en cours, cotes
    pilotées verrouillées en prédimensionnement."""
    return Options(lvl=lvl, interactive=True, hl=hl, editables=True, poignees=True,
                   locked=set(PILOTEES_PAR_PREDIM) if R.pred else None)


def options_rapport():
    """Options du rapport : cotations principales, sans interaction, sans
    cartouche (la note porte déjà les données et les hypothèses)."""
    return Options(lvl=1, interactive=False, hl=None, locked=None, cartouche=False)


def sym_ec(s, notation=None):
    """Symbole affiché pour un symbole du moteur (« Lc » → « hc »)."""
    n = NOTATION_EC if notation is None else notation
    return n.get(s, s)


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


# ------------------------------------------------------- géométrie réaliste
def _arc(cx, cy, r, a0, a1, n=SEGMENTS_ARC):
    """Points d'un arc de cercle (repère y vers le bas), de l'angle ``a0`` à
    ``a1`` inclus, en ``n`` segments par quart de cercle."""
    k = max(1, int(round(n * abs(a1 - a0) / (math.pi / 2))))
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / k), cy + r * math.sin(a0 + (a1 - a0) * i / k))
            for i in range(k + 1)]


def _section_I(b, h, tw, tf, r):
    """Contour d'une section en I (semelles parallèles) avec ses quatre
    congés âme–semelle de rayon ``r`` (EN 10365) ; origine au milieu de la
    face supérieure, y vers le bas."""
    x = tw / 2
    r = mx(0, mn(r, (b - tw) / 2, (h - 2 * tf) / 2))
    pts = [(-b / 2, 0), (b / 2, 0), (b / 2, tf)]
    if r > 0:
        pts += _arc(x + r, tf + r, r, -math.pi / 2, -math.pi)
        pts += _arc(x + r, h - tf - r, r, math.pi, math.pi / 2)
    else:
        pts += [(x, tf), (x, h - tf)]
    pts += [(b / 2, h - tf), (b / 2, h), (-b / 2, h), (-b / 2, h - tf)]
    if r > 0:
        pts += _arc(-x - r, h - tf - r, r, math.pi / 2, 0)
        pts += _arc(-x - r, tf + r, r, 0, -math.pi / 2)
    else:
        pts += [(-x, h - tf), (-x, tf)]
    return pts


def _corniere_plan(xf, xt, tC, y1, y2, y3, g, rC):
    """Section d'une cornière en plan : aile B le long de l'âme secondaire
    (de ``xf`` à ``xt``, entre ``y1`` et ``y2``), aile A le long de l'âme
    principale (épaisseur ``tC``, jusqu'à ``y3``) ; congé de racine ``rC``
    et arrondis de bout ``rC/2`` (convention EN 10056-1 : r2 = r1/2)."""
    xa = xf + tC
    r = mx(0, mn(rC, (xt - xa) / 2, abs(y3 - y2) / 2))
    r2 = r / 2
    pts = [(xf, y1), (xt, y1)]
    if r > 0:
        pts += _arc(xt - r2, y2 - g * r2, r2, 0, g * math.pi / 2)             # bout de l'aile B
        pts += _arc(xa + r, y2 + g * r, r, -g * math.pi / 2, -g * math.pi)     # congé de racine
        pts += _arc(xa - r2, y3 - g * r2, r2, 0, g * math.pi / 2)             # bout de l'aile A
    else:
        pts += [(xt, y2), (xa, y2), (xa, y3)]
    pts.append((xf, y3))
    return pts


def _hachures(pts, pas=PAS_HACHURES):
    """Hachures à 45° d'un polygone (règle pair-impair) : segments
    ``[(x, y), (x, y)]`` sur les droites x − y = c, espacées de ``pas``."""
    cs = [x - y for x, y in pts]
    c0, c1 = min(cs), max(cs)
    d = pas * math.sqrt(2)
    lignes = []
    n = len(pts)
    c = c0 + d / 2
    while c < c1:
        inter = []
        for i in range(n):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
            f1 = x1 - y1 - c; f2 = x2 - y2 - c
            if (f1 < 0) != (f2 < 0):
                t = f1 / (f1 - f2)
                inter.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        inter.sort(key=lambda p: p[0] + p[1])
        for j in range(0, len(inter) - 1, 2):
            a, b = inter[j], inter[j + 1]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1e-6:
                lignes.append([a, b])
        c += d
    return lignes


def _rect_pts(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


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

    def sym(self, s):
        """Symbole affiché (notation de l'option)."""
        return self.opt.notation.get(s, s)

    def is_hot(self, id_):
        h = self.opt.hl
        return bool(h and h.get("dims") and id_ in h["dims"])

    def visible(self, d):
        """Une cote se pose si son niveau est atteint, si une alerte la
        désigne, ou — en mode édition — si elle est modifiable."""
        return bool(d["lvl"] <= self.opt.lvl or self.is_hot(d["id"])
                    or (self.opt.editables and d.get("key")))

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
                      key=d.get("key") if ed else None, sym=self.sym(d.get("sym")), id=d["id"])

    def dim(self, d):
        """d : id, side T|B|L|R, a, b, o1, o2, sym, val, key, lvl, calc, out."""
        hot = self.is_hot(d["id"])
        if not self.visible(d):
            return
        if not abs(d["b"] - d["a"]) > 0.01:
            return
        fs = self.fs; lo = mn(d["a"], d["b"]); hi = mx(d["a"], d["b"]); ln = hi - lo
        sy = self.sym(d["sym"])
        t1 = sy + " = " + f0(d["val"]); t2 = sy + " " + f0(d["val"]); l2 = lo; h2 = hi
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
        if not self.visible(d):
            return
        txt = self.sym(d["sym"]) + " " + f0(d["val"])
        self.g.append(_group(["dm"] + (["hot"] if hot else []),
                             [self.label(x + self.tw(txt) / 2 + 0.3 * self.fs, y, False, txt, d)]))

    def largeur_poignees(self, txt):
        """Largeur occupée par une pile de poignées (étiquette + / −)."""
        return mx(self.tw(txt) + 0.6 * self.fs, 1.3 * self.fs)

    def poignees(self, xc, yc, grp, key, txt, hot=False):
        """Poignées d'un groupe de boulons, empilées et centrées en
        ``(xc, yc)`` : « + » (une rangée de plus), l'étiquette « n1 × n2 »
        (cliquable : fenêtre du groupe), « − » (une rangée de moins)."""
        if not self.opt.poignees:
            return
        fs = self.fs; w = self.largeur_poignees(txt); b = 1.3 * fs
        ed = bool(self.opt.interactive and not (self.opt.locked and key in self.opt.locked))
        h = ["hot"] if hot else []

        def pile(y, cls, enfants, **attrs):
            # comme ``label`` : le rectangle va de −1,05·fs à +0,37·fs autour
            # de la ligne de base, son centre visuel est donc 0,34·fs plus haut
            return _group(cls, enfants, x=xc, y=y + 0.34 * fs, rot=0, **attrs)

        def bouton(y, signe, delta):
            r = _rect(["gbx"], -b / 2, -1.05 * fs + (1.42 * fs - b) / 2, b, b); r["rx"] = 0.25 * fs
            return pile(y, ["gb"] + (["ed"] if ed else []) + h, [r, _text([], None, None, signe, 1.15 * fs)],
                        action=(key + ":" + ("+" if delta > 0 else "") + js_str(delta)) if ed else None,
                        aria=("Une rangée de plus" if delta > 0 else "Une rangée de moins") + " – groupe " + grp)
        hit = _rect(["hit"], -w / 2, -1.05 * fs, w, 1.42 * fs); hit["rx"] = 0.2 * fs
        etiquette = pile(yc, ["gp"] + (["ed"] if ed else []) + h, [hit, _text([], None, None, txt, fs)],
                         grp=grp if ed else None, id="grp" + grp)
        self.g.append(_group(["pg"], [bouton(yc - 1.55 * fs, "+", 1), etiquette, bouton(yc + 1.55 * fs, "−", -1)]))

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
    xt = xf + R.b_B
    txtS = js_str(R.n1_S) + " × " + js_str(R.n2_S); txtP = js_str(R.n1_P) + " × " + js_str(R.n2_P)
    if opt.poignees:
        # la boîte s'élargit pour que les poignées n'empiètent pas sur les cotes
        S0 = Feuille(bb, fs, opt)
        if R.bolt_S:
            bb["x2"] = mx(bb["x2"], xt + 0.5 * fs + S0.largeur_poignees(txtS) + 0.3 * fs)
        if R.bolt_P:
            bb["x1"] = mn(bb["x1"], -xf - 16 - 0.5 * fs - S0.largeur_poignees(txtP) - 0.3 * fs)
    S = Feuille(bb, fs, opt); s = []
    hot = (opt.hl or {}).get("elems") or set()

    def hc(k):
        return ["hot"] if k in hot else []

    def piece(grp, prims, **attrs):
        """Une pièce sélectionnable : groupe cliquable en mode édition
        (``data-group``), primitives à plat sinon (parité, note)."""
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    if opt.realiste:
        # section en I coupée : une seule pièce, congés réels, hachures
        sec = _section_I(R.b_P, R.h_P, R.tw_P, R.tf_P, R.r_P)
        piece("beamP", [_poly(["pp"], sec), _path(["ht"], _hachures(sec))])
        if "flPt" in hot:
            s.append(_path(["hl"], [[(-R.b_P / 2, R.tf_P), (-R.b_P / 2, 0), (R.b_P / 2, 0), (R.b_P / 2, R.tf_P)]]))
        if "flPb" in hot:
            s.append(_path(["hl"], [[(-R.b_P / 2, R.h_P - R.tf_P), (-R.b_P / 2, R.h_P), (R.b_P / 2, R.h_P), (R.b_P / 2, R.h_P - R.tf_P)]]))
    else:
        piece("beamP", [_rect(["pp"] + hc("flPt"), -R.b_P / 2, 0, R.b_P, R.tf_P),
                        _rect(["pp"] + hc("flPb"), -R.b_P / 2, R.h_P - R.tf_P, R.b_P, R.tf_P),
                        _rect(["pp"], -xf, R.tf_P, R.tw_P, R.h_P - 2 * R.tf_P)])
    rn = mn(N(u.r_n), ln, dnt if dnt > 0 else INF_MM, dnb if dnb > 0 else INF_MM) if opt.realiste else 0
    if dnt > 0:
        p = [[x0, yt + dnt]]
        if rn > 0:
            p += _arc(x0 + ln - rn, yt + dnt - rn, rn, math.pi / 2, 0)     # rayon du grugeage
        else:
            p.append([x0 + ln, yt + dnt])
        p += [[x0 + ln, yt], [xE, yt]]
    else:
        p = [[x0, yt], [xE, yt]]
    if dnb > 0:
        p += [[xE, yb], [x0 + ln, yb]]
        if rn > 0:
            p += _arc(x0 + ln - rn, yb - dnb + rn, rn, 0, -math.pi / 2)
        else:
            p.append([x0 + ln, yb - dnb])
        p.append([x0, yb - dnb])
    else:
        p += [[xE, yb], [x0, yb]]
    pS = [_poly(["ps"] + hc("beamS"), p),
          _path(["fl2"], [[(x0 + ln if dnt > 0 else x0, yt + R.tf_S), (xE, yt + R.tf_S)],
                          [(x0 + ln if dnb > 0 else x0, yb - R.tf_S), (xE, yb - R.tf_S)]])]
    if opt.realiste and R.r_S > 0:
        # lignes tangentes des congés âme–semelle vus de côté (tf + r)
        pS.append(_path(["flr"], [[(x0 + ln if dnt > 0 else x0, yt + R.tf_S + R.r_S), (xE, yt + R.tf_S + R.r_S)],
                                  [(x0 + ln if dnb > 0 else x0, yb - R.tf_S - R.r_S), (xE, yb - R.tf_S - R.r_S)]]))
    piece("beamS", pS, drag="g_h", dsym="gh")
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
    piece("cleat", [_rect(["co"] + hc("cleat"), xf, yc, R.b_B, R.L_C),
                    _rect(["co2"], xf, yc, R.t_C, R.L_C)])
    xc1 = xf + R.g_B; xcl = xc1 + (R.n2_S - 1) * p2S; yb1 = yc + R.e1_S
    ybl = yb1 + (R.n1_S - 1) * R.p1_S; r0 = R.d_0 / 2
    if R.bolt_S:
        bS = []
        for i in range(R.n1_S):
            for j in range(R.n2_S):
                cx = xc1 + j * p2S; cy = yb1 + i * R.p1_S
                if opt.realiste:
                    bS.append(_circle(["bw"], cx, cy, R.d_w / 2))          # rondelle (dw)
                bS.append(_circle(["bo"] + hc("boltsS"), cx, cy, r0))
                bS.append(_path(["cm"], [[(cx - r0 - 4, cy), (cx + r0 + 4, cy)], [(cx, cy - r0 - 4), (cx, cy + r0 + 4)]]))
        piece("bolts", bS)
    elif opt.realiste:
        # cordons d'angle vus de face : bande de largeur a·√2 le long du bout
        # de l'aile B, retours en haut et en bas
        z = N(u.a_S) * math.sqrt(2)
        wS = [_rect(["wb"], xt, yc, z, R.L_C)]
        if lhS > 0:
            wS += [_rect(["wb"], xt - lhS, yc - z, lhS + z, z),
                   _rect(["wb"], xt - lhS, yc + R.L_C, lhS + z, z)]
        piece("weldS", wS)
    else:
        piece("weldS", [_path(["we"], [[(xt - lhS, yc), (xt, yc), (xt, yc + R.L_C), (xt - lhS, yc + R.L_C)]])])
    yp1 = yc + R.e1_P; ypl = yp1 + (R.n1_P - 1) * R.p1_P
    if R.bolt_P:
        piece("bolts", [_path(["bp"] + hc("boltsP"), [[(-xf - 16, yp1 + i * R.p1_P), (xf + R.t_C + 16, yp1 + i * R.p1_P)]])
                        for i in range(R.n1_P)])
    if opt.poignees:
        # étiquette des efforts, sur l'âme de la poutre portée (panneau VEd,
        # NEd, HEd, MEd au clic)
        autres = any(N(u[k]) != 0 for k in ("N_Ed", "H_Ed", "M_Ed"))
        txtF = "VEd " + f0(N(u.V_Ed)) + (" +" if autres else "")
        wF = S.tw(txtF) + 0.5 * fs
        xF = (mx(x0 + (ln if notch else 0), xt) + xE) / 2
        hitF = _rect(["hit"], -wF / 2, -1.05 * fs, wF, 1.42 * fs); hitF["rx"] = 0.2 * fs
        s.append(_group(["ef", "ed"], [hitF, _text([], None, None, txtF, fs)],
                        x=xF, y=mx(yt + dnt, yc) + 1.7 * fs, rot=0,
                        grp="efforts", id="efforts"))
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
        if lhS == 0 and opt.editables:
            # retours nuls : pas de cote possible, une étiquette (mode édition seulement)
            S.tag(xt + 0.4 * fs, yc + R.L_C / 2 + 1.6 * fs, dict(id="lhS", sym="lh", val=0, key="lh_S", lvl=1))
    # poignées des groupes de boulons (écran) : dans l'âme de la poutre
    # secondaire à droite des cornières (S), entre les semelles de la
    # principale à gauche de l'âme (P)
    if R.bolt_S:
        S.poignees(xt + 0.5 * fs + S.largeur_poignees(txtS) / 2, (yb1 + ybl) / 2, "S", "n1S_u", txtS, "boltsS" in hot)
    if R.bolt_P:
        S.poignees(-xf - 16 - 0.5 * fs - S.largeur_poignees(txtP) / 2, (yp1 + ypl) / 2, "P", "n1P_u", txtP, "boltsP" in hot)
    nomS = "Poutre secondaire" if u.prof_S == PERSO else u.prof_S
    S.name("T", xE - S.tw(nomS) / 2, nomS)
    S.name("B", 0, "Poutre principale" if u.prof_P == PERSO else u.prof_P)
    L = []
    if opt.lvl >= 1 and opt.cartouche:
        L.append("Cornières : 2 × " + R.corn_txt + " – " + u.nu_C + " – " + S.sym("Lc") + " " + f0(R.L_C) + " mm")
        if R.bolt_S or R.bolt_P:
            L.append("Boulons " + R.boulon + " – classe " + u.classe + " – trous d0 " + f0(R.d_0) + " mm"
                     + (" – groupe S : " + js_str(R.n1_S) + " × " + js_str(R.n2_S) if R.bolt_S else "")
                     + (" – groupe P : 2 × (" + js_str(R.n1_P) + " × " + js_str(R.n2_P) + ")" if R.bolt_P else ""))
        if not R.bolt_S:
            L.append("Soudure ailes B : a " + f0(N(u.a_S)) + " mm, cordon vertical " + S.sym("Lc") + " + retours " + f0(lhS) + " mm")
        if not R.bolt_P:
            L.append("Soudure ailes A : a " + f0(N(u.a_P)) + " mm, cordon vertical " + S.sym("Lc") + " + retours " + f0(N(u.lh_P)) + " mm")
        L.append("Excentricité de calcul z = " + f0(R.zeff) + " mm – MS = " + F(R.M_S, 2) + " kNm"
                 + ((" – eP = " + f0(R.e_P if R.bolt_P else R.ew_P) + " mm, MP = " + F(R.M_P, 2) + " kNm") if R.M_P > 0 else ""))
        if notch:
            L.append("Grugeage : " + ("sup. " + f0(dnt) if dnt > 0 else "") + (" / " if dnt > 0 and dnb > 0 else "")
                     + ("inf. " + f0(dnb) if dnb > 0 else "") + " × " + f0(ln) + " mm – bras de levier gh + " + S.sym("ln") + " = " + f0(gh + ln) + " mm")
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

    def piece(grp, prims, **attrs):
        if opt.poignees and prims:
            s.append(_group(["pc", "ed"], prims, grp=grp, **attrs))
        else:
            s.extend(prims)

    piece("beamP", [_rect(["pp"], -xf, -Hh, R.tw_P, 2 * Hh)]
          + ([_path(["ht"], _hachures(_rect_pts(-xf, -Hh, R.tw_P, 2 * Hh)))] if opt.realiste else []))
    piece("beamS", [_rect(["ps"] + hc("beamS"), x0, -ws, xE - x0, R.tw_S)]
          + ([_path(["ht"], _hachures(_rect_pts(x0, -ws, xE - x0, R.tw_S)))] if opt.realiste else []),
          drag="g_h", dsym="gh")
    for g in (-1, 1):
        y1 = g * ws; y2 = g * (ws + R.t_C); y3 = g * (ws + R.b_A)
        if opt.realiste:
            pts = _corniere_plan(xf, xt, R.t_C, y1, y2, y3, g, R.r_C)
            piece("cleat", [_poly(["co"] + hc("cleat"), pts), _path(["ht", "htc"], _hachures(pts))])
        else:
            piece("cleat", [_poly(["co"] + hc("cleat"), [[xf, y1], [xt, y1], [xt, y2], [xf + R.t_C, y2], [xf + R.t_C, y3], [xf, y3]])])
        if R.bolt_P:
            piece("bolts", [_path(["bp"] + hc("boltsP"), [[(-xf - 14, g * (ws + R.g_A + i * p2P)), (xf + R.t_C + 14, g * (ws + R.g_A + i * p2P))]])
                            for i in range(R.n2_P)])
        elif opt.realiste:
            # cordon d'angle en section : triangle de côtés a·√2, dans l'angle
            # entre le bout de l'aile A et la face de l'âme principale
            z = N(u.a_P) * math.sqrt(2)
            piece("weldP", [_poly(["wb"], [[xf, y3], [xf + z, y3], [xf, y3 + g * z]])])
        else:
            piece("weldP", [_circle(["wd"], xf + R.t_C, y3, mx(N(u.a_P), 4))])
        if not R.bolt_S:
            if opt.realiste:
                z = N(u.a_S) * math.sqrt(2)
                piece("weldS", [_poly(["wb"], [[xt, y1], [xt + z, y1], [xt, y1 + g * z]])])
            else:
                piece("weldS", [_circle(["wd"], xt, y2, mx(N(u.a_S), 4))])
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
        if opt.editables:
            # retours des cordons A : pas de cote dans cette vue, une étiquette (mode édition)
            S.tag(xf + R.t_C + 0.6 * fs, (ws + R.b_A) + 1.2 * fs, dict(id="lhP", sym="lh", val=N(u.lh_P), key="lh_P", lvl=1))
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


# ------------------------------------------------------------ style résolu
def style_de(cls, ctx, p=None):
    """Style d'une primitive d'après ses classes et celles de ses groupes
    parents (``ctx``) — la même règle que la feuille de style, exprimée en
    attributs : ``fill``, ``fo`` (opacité de remplissage), ``stroke``,
    ``sw`` (épaisseur), ``dash``, ``bold``, ``italic``, ``tfill`` (texte).
    Sert au SVG (attributs de présentation, valables sans CSS) et au peintre
    PDF de ``rapport.py``."""
    p = p or PALETTE
    c = set(cls); k = set(ctx)
    hot_self = "hot" in c; hot_ctx = "hot" in k
    st = dict(fill="none", fo=1.0, stroke="none", sw=1.0, dash=None, bold=False,
              italic=False, tfill=p["accent"])
    ink, acc, ko = p["ink"], p["accent"], p["ko"]
    if "pp" in c or "ps" in c:
        st.update(fill=p["pp"] if "pp" in c else p["ps"], stroke=ink, sw=1.4)
        if hot_self:
            st.update(stroke=ko, sw=3)
    elif "fl2" in c:
        st.update(stroke=ink, sw=0.8)
    elif "co" in c:
        st.update(fill=acc, fo=0.28, stroke=acc, sw=1.6)
        if hot_self:
            st.update(fill=ko, fo=0.2, stroke=ko, sw=3)
    elif "co2" in c:
        st.update(fill=acc, fo=0.6)
    elif "bo" in c:
        st.update(fill="#FFFFFF", stroke=ink, sw=1.5)
        if hot_self:
            st.update(stroke=ko, sw=3)
    elif "bp" in c:
        st.update(stroke=ko, sw=2.4, dash=(7, 3))
        if hot_self:
            st.update(sw=4.5, dash=None)
    elif "we" in c:
        st.update(stroke=p["weld"], sw=5)
    elif "wd" in c:
        st.update(fill=p["weld"])
    elif "wb" in c:
        # cordon à sa taille réelle : plein, contour plus soutenu
        st.update(fill=p["weld"], stroke=p["weld2"], sw=0.8)
    elif "bw" in c:
        st.update(stroke=ink, sw=0.5)
    elif "ht" in c:
        st.update(stroke=acc if "htc" in c else p["hatch"], sw=0.5)
    elif "hl" in c:
        st.update(stroke=ko, sw=4)
    elif "cm" in c:
        st.update(stroke=ink, sw=0.6)
    elif "flr" in c:
        # tangente du congé âme–semelle vue de côté : trait fin
        st.update(stroke=p["hatch"], sw=0.45)
    elif "ext" in c:
        st.update(stroke=p["ext"], sw=0.5)
        if hot_ctx:
            st.update(stroke=ko, sw=2)
    elif "dln" in c:
        st.update(stroke=acc, sw=0.8)
        if "calc" in k:
            st.update(stroke=p["calc"], dash=(5, 3))
        if hot_ctx:
            st.update(stroke=ko, sw=2, dash=None)
    elif "hit" in c:
        # halo blanc discret sous chaque étiquette (cotation « propre » ;
        # l'affordance d'édition n'apparaît qu'au survol, à l'écran)
        st.update(fill="#FFFFFF", fo=0.85)
        if hot_ctx:
            st.update(fill=p["hotbg"], fo=1.0, stroke=ko, sw=2)
    elif "gbx" in c:
        # bouton + / − d'un groupe de boulons : plein, couleur d'accent
        st.update(fill=acc if "ed" in k else p["ext"], fo=1.0)
        if hot_ctx:
            st.update(fill=ko)
    # texte
    if "tx" in c:
        st.update(tfill=ink, bold=True)
    elif "cart" in k:
        st.update(tfill=ink)
    elif "gb" in k:
        st.update(tfill="#FFFFFF", bold=True)
    elif "dl" in k or "gp" in k or "ef" in k:
        if "ed" in k:
            st.update(tfill=p["inink"], bold=True)
        if "calc" in k:
            st.update(tfill=p["calc"], italic=True)
        if hot_ctx:
            st.update(tfill=ko, bold=True)
    return st


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
        f"#{pre} .wb{{fill:{p['weld']};stroke:{p['weld2']};stroke-width:.8}}"
        f"#{pre} .bw{{fill:none;stroke:{ink};stroke-width:.5}}"
        f"#{pre} .ht{{stroke:{p['hatch']};stroke-width:.5;fill:none}}"
        f"#{pre} .htc{{stroke:{acc}}}"
        f"#{pre} .dm path{{stroke:{acc};stroke-width:.8;fill:none}}"
        f"#{pre} .ext{{stroke:{p['ext']};stroke-width:.5;fill:none}}"
        f"#{pre} .flr{{stroke:{p['hatch']};stroke-width:.45;fill:none}}"
        f"#{pre} .dm text,#{pre} .tx,#{pre} .cart text{{fill:{acc};font-family:system-ui,Arial,sans-serif}}"
        f"#{pre} .tx{{fill:{ink};font-weight:600}}"
        f"#{pre} .cart text{{fill:{ink}}}"
        f"#{pre} .dl .hit{{fill:#fff;fill-opacity:.85;stroke:none}}"
        f"#{pre} .dl.ed{{cursor:pointer}}"
        f"#{pre} .dl.ed text{{fill:{p['inink']};font-weight:600}}"
        f"#{pre} .dl.ed:hover .hit,#{pre} .dl.ed:focus .hit{{fill:{p['hover']};fill-opacity:1;stroke:{acc};stroke-width:1.6}}"
        f"#{pre} .dl.ed:focus{{outline:none}}"
        f"#{pre} .ef{{cursor:pointer}}"
        f"#{pre} .ef .hit{{fill:#fff;fill-opacity:.88;stroke:{p['inbord']};stroke-width:1}}"
        f"#{pre} .ef text{{fill:{p['inink']};font-weight:700}}"
        f"#{pre} .ef:hover .hit,#{pre} .ef:focus .hit{{fill:{p['hover']};stroke:{acc};stroke-width:1.6}}"
        f"#{pre} .ef:focus{{outline:none}}"
        f"#{pre} .pc.ed{{cursor:pointer}}"
        f"#{pre} g[data-drag]{{cursor:grab}}"
        f"#{pre} g[data-drag].drag{{cursor:grabbing;opacity:.75}}"
        f"#{pre} .pc.ed:hover .pp,#{pre} .pc.ed:hover .ps,#{pre} .pc.sel .pp,#{pre} .pc.sel .ps{{stroke:{acc};stroke-width:2.6}}"
        f"#{pre} .pc.ed:hover .co,#{pre} .pc.sel .co{{stroke-width:2.8}}"
        f"#{pre} .pc.ed:hover .bo,#{pre} .pc.sel .bo{{stroke:{acc};stroke-width:2.4}}"
        f"#{pre} .pc.ed:hover .bp,#{pre} .pc.sel .bp{{stroke-width:3.4}}"
        f"#{pre} .pc.ed:hover .wb,#{pre} .pc.sel .wb{{stroke-width:1.8}}"
        f"#{pre} .pc.ed:hover .we,#{pre} .pc.sel .we{{stroke-width:7}}"
        f"#{pre} .pc:focus{{outline:none}}"
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
        f"#{pre} .gp .hit{{fill:{p['inbg']};fill-opacity:1;stroke:{p['inbord']};stroke-width:1}}"
        f"#{pre} .gp text{{fill:{p['inink']};font-weight:600}}"
        f"#{pre} .gp.ed,#{pre} .gb.ed{{cursor:pointer}}"
        f"#{pre} .gp.ed:hover .hit,#{pre} .gp.ed:focus .hit{{fill:{p['hover']};stroke:{acc};stroke-width:2}}"
        f"#{pre} .gb .gbx{{fill:{p['ext']}}}"
        f"#{pre} .gb.ed .gbx{{fill:{acc}}}"
        f"#{pre} .gb text{{fill:#fff;font-weight:700}}"
        f"#{pre} .gb.ed:hover .gbx,#{pre} .gb.ed:focus .gbx{{fill:{ink}}}"
        f"#{pre} .gp.hot .hit{{fill:{p['hotbg']};stroke:{ko};stroke-width:2}}"
        f"#{pre} .gp.hot text,#{pre} .gb.hot .gbx{{fill:{ko}}}"
        f"#{pre} .gp:focus,#{pre} .gb:focus{{outline:none}}"
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


def _attrs_forme(st):
    a = f' fill="{st["fill"]}"'
    if st["fill"] != "none" and st["fo"] < 1:
        a += f' fill-opacity="{_n(st["fo"])}"'
    a += f' stroke="{st["stroke"]}"'
    if st["stroke"] != "none":
        a += f' stroke-width="{_n(st["sw"])}" vector-effect="non-scaling-stroke"'
        if st["dash"]:
            a += f' stroke-dasharray="{st["dash"][0]} {st["dash"][1]}"'
    return a


def _attrs_texte(st):
    a = f' fill="{st["tfill"]}" font-family="system-ui,Arial,sans-serif"'
    if st["bold"]:
        a += ' font-weight="600"'
    if st["italic"]:
        a += ' font-style="italic"'
    return a


def _prim_svg(p, ctx=(), palette=None):
    """Une primitive en SVG : classes (pour la feuille de style, le survol et
    le composant cliquable) ET attributs de présentation (valables sans CSS)."""
    cls = " ".join(p["cls"])
    ca = f' class="{cls}"' if cls else ""
    t = p["t"]
    if t == "g":
        attrs = ""
        if p.get("key"):
            attrs += (f' data-key="{p["key"]}" data-sym="{_esc(p["sym"])}" tabindex="0" role="button"'
                      f' aria-label="Modifier {_esc(p["sym"])}"')
        if p.get("grp"):
            attrs += (f' data-group="{p["grp"]}" tabindex="0" role="button"'
                      f' aria-label="Modifier le groupe {p["grp"]}"')
        if p.get("drag"):
            attrs += f' data-drag="{p["drag"]}" data-dsym="{_esc(p.get("dsym") or p["drag"])}"'
        if p.get("action"):
            attrs += (f' data-action="{_esc(p["action"])}" tabindex="0" role="button"'
                      f' aria-label="{_esc(p.get("aria") or p["action"])}"')
        if "x" in p:
            attrs += f' transform="translate({_n(p["x"])} {_n(p["y"])})' + (" rotate(-90)" if p.get("rot") else "") + '"'
        if p.get("id") and "x" in p:
            attrs += f' data-dim="{p["id"]}"'
        sous = tuple(ctx) + tuple(p["cls"])
        return f'<g{ca}{attrs}>' + "".join(_prim_svg(e, sous, palette) for e in p["enfants"]) + "</g>"
    st = style_de(p["cls"], ctx, palette)
    if t == "rect":
        rx = f' rx="{_n(p["rx"])}"' if p.get("rx") else ""
        return (f'<rect{ca} x="{_n(p["x"])}" y="{_n(p["y"])}" width="{_n(p["w"])}" height="{_n(p["h"])}"{rx}'
                f'{_attrs_forme(st)}/>')
    if t == "poly":
        return f'<polygon{ca} points="{" ".join(_n(x) + "," + _n(y) for x, y in p["pts"])}"{_attrs_forme(st)}/>'
    if t == "path":
        return f'<path{ca} d="{_path_d(p["lignes"], p.get("hint"))}"{_attrs_forme(st)}/>'
    if t == "circle":
        return f'<circle{ca} cx="{_n(p["cx"])}" cy="{_n(p["cy"])}" r="{_n(p["r"])}"{_attrs_forme(st)}/>'
    if t == "text":
        pos = f' x="{_n(p["x"])}" y="{_n(p["y"])}"' if p.get("x") is not None else ""
        anc = f' text-anchor="{p["anchor"]}"' if p.get("anchor") else ""
        return (f'<text{ca}{pos}{anc} font-size="{_n(p["size"])}"{_attrs_texte(st)}>'
                f'{_esc(p["txt"])}</text>')
    raise ValueError(t)


def vers_svg(d, palette=PALETTE, largeur="100%", identifiant="dc"):
    """SVG autonome de la vue ``d`` (style embarqué)."""
    x1, y1, w, h = d.viewbox
    corps = "".join(_prim_svg(p, (), palette) for p in d.corps)
    cotes = "".join(_prim_svg(p, (), palette) for p in d.cotes)
    cart = ('<g class="cart">' + "".join(_prim_svg(p, ("cart",), palette) for p in d.cartouche) + "</g>") if d.cartouche else ""
    return (f'<svg id="{identifiant}" xmlns="http://www.w3.org/2000/svg" viewBox="{_n(x1)} {_n(y1)} {_n(w)} {_n(h)}"'
            f' width="{largeur}" role="img" aria-label="{_esc(d.aria)}">'
            f"<style>{_css(palette, identifiant)}</style>" + corps + cotes + cart + "</svg>")

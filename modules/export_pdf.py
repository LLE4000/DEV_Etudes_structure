# -*- coding: utf-8 -*-
# ============================================================
#  export_pdf.py — Note de calcul PDF (poutre béton armé)
#  VERSION 2.20 (alignée sur poutre.py 2.20)
#
#  Corrections principales vs version précédente :
#   - LITS MULTIPLES (jusqu'à 4 par face) : lecture de nlits_*,
#     positions réelles (distance axe lit / parement), centre de
#     gravité pondéré par les aires pour la hauteur utile.
#   - DISTANCES axe lit / parement : nouvelles clés (enrob_calc_* pour
#     le lit 1, dist_*_l{i} pour les suivants) + jeux globaux
#     (jeu premier lit, jeu entre lits) et diamètre étrier.
#   - ENROBAGE : affiche l'enrobage BÉTON de la poutre (KB enrobage_beton).
#   - As,min : accolade sur 3 lignes
#       max( 0,26·fctm/fyk·b·h ; 0,0013·b·h ; 0,25·As,req opposé )
#     valeurs numériques exactes = celles retenues par poutre.py.
#   - As,min / As,max : même police (taille) que As,req.
#
#  API : generer_rapport_pdf(beams, values, beton_data, infos=None) -> chemin PDF
# ============================================================

import io
import math
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as _canvas
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Table, TableStyle,
    KeepTogether, PageBreak, Flowable,
)


# ============================================================
#  PALETTE
# ============================================================
INK   = colors.HexColor("#1a1a1a")
MUTE  = colors.HexColor("#737373")
HAIR  = colors.HexColor("#e5e5e5")
SOFT  = colors.HexColor("#f7f7f7")

BEAM_BG = colors.HexColor("#e7ecf2"); BEAM_TX = colors.HexColor("#243b53")
SEC_BG  = colors.HexColor("#f2f5f8"); SEC_TX  = colors.HexColor("#334e68")

OKD = colors.HexColor("#2f7d4f"); WD = colors.HexColor("#9a6a1c"); ND = colors.HexColor("#b3261e")
ECOL  = {"ok": OKD, "warn": WD, "nok": ND}
EPALE = {"ok": colors.HexColor("#eaf6ee"), "warn": colors.HexColor("#fdf4e3"), "nok": colors.HexColor("#fdeceb")}
EDARK = {"ok": colors.HexColor("#1e5b39"), "warn": colors.HexColor("#7a5314"), "nok": colors.HexColor("#8f1d17")}
ELAB  = {"ok": "Vérifié", "warn": "À surveiller", "nok": "Non vérifié"}

# coupe de section : couleurs par lit
LIT_COLORS = [
    (colors.HexColor("#c0392b"), colors.HexColor("#7d2118")),  # lit 1 = rouge
    (colors.HexColor("#2e6fb0"), colors.HexColor("#1c4a78")),  # lit 2 = bleu
    (colors.HexColor("#1f8a70"), colors.HexColor("#125a48")),  # lit 3 = vert
    (colors.HexColor("#8a5a1f"), colors.HexColor("#5a3a12")),  # lit 4 = brun
]
PAL = {
    "conc": colors.HexColor("#f2f2f0"), "conc_bd": INK, "hatch": colors.HexColor("#d9d9d6"),
    "sup":  colors.HexColor("#1f8a70"), "sup_bd": colors.HexColor("#125a48"),
    "stirrup": colors.HexColor("#6b6f76"),
    "dim": MUTE, "txt": INK, "axis": colors.HexColor("#9aa0a6"),
}

MAX_LITS = 4


# ============================================================
#  FORMAT NOMBRES (virgule décimale FR)
# ============================================================
def fn(x, nd=2):
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except Exception:
        return str(x)


def s2():
    return "<super>2</super>"


# ============================================================
#  MOTEUR DE FORMULES VECTORIELLES (zéro image)
# ============================================================
def _w(txt, font, size):
    return stringWidth(txt, font, size)


class _Tok:
    def size_(self, c): raise NotImplementedError
    def draw(self, c, x, yb): raise NotImplementedError


class T(_Tok):
    def __init__(self, s, font="Helvetica", size=10, color=INK, sub=None, sup=None, subsize=None):
        self.s = s; self.font = font; self.size = size; self.color = color
        self.sub = sub; self.sup = sup; self.subsize = subsize or size * 0.72

    def size_(self, c):
        w = _w(self.s, self.font, self.size)
        asc = self.size * 0.72; desc = 0.0; extra = 0
        if self.sub:
            extra = max(extra, _w(self.sub, self.font, self.subsize)); desc = max(desc, self.subsize * 0.55)
        if self.sup:
            extra = max(extra, _w(self.sup, self.font, self.subsize)); asc = max(asc, self.size * 0.72 + self.subsize * 0.5)
        return w + extra, asc, desc

    def draw(self, c, x, yb):
        c.setFont(self.font, self.size); c.setFillColor(self.color)
        c.drawString(x, yb, self.s)
        w = _w(self.s, self.font, self.size)
        if self.sub:
            c.setFont(self.font, self.subsize); c.drawString(x + w + 0.5, yb - self.subsize * 0.45, self.sub)
        if self.sup:
            c.setFont(self.font, self.subsize); c.drawString(x + w + 0.5, yb + self.size * 0.45, self.sup)


class Frac(_Tok):
    def __init__(self, num, den, color=INK, pad=3):
        self.num = num if isinstance(num, Row) else Row(num)
        self.den = den if isinstance(den, Row) else Row(den)
        self.color = color; self.pad = pad

    def size_(self, c):
        nw, na, nd = self.num.size_(c); dw, da, dd = self.den.size_(c)
        w = max(nw, dw) + self.pad * 2; gap = 2.5
        return w, (na + nd) + gap + 1, (da + dd) + gap

    def draw(self, c, x, yb):
        nw, na, nd = self.num.size_(c); dw, da, dd = self.den.size_(c)
        w = max(nw, dw); gap = 2.5; bar_y = yb + 2
        self.num.draw(c, x + self.pad + (w - nw) / 2.0, bar_y + gap + nd)
        c.setStrokeColor(self.color); c.setLineWidth(0.8)
        c.line(x, bar_y, x + w + self.pad * 2, bar_y)
        self.den.draw(c, x + self.pad + (w - dw) / 2.0, bar_y - gap - da)


class Sqrt(_Tok):
    def __init__(self, inner, color=INK):
        self.inner = inner if isinstance(inner, Row) else Row(inner)
        self.color = color

    def size_(self, c):
        iw, ia, idsc = self.inner.size_(c)
        return iw + 10 + 4, ia + 3, idsc

    def draw(self, c, x, yb):
        iw, ia, idsc = self.inner.size_(c)
        top = yb + ia + 2; bot = yb - idsc; h = top - bot
        c.setStrokeColor(self.color); c.setLineWidth(0.9)
        p = c.beginPath()
        p.moveTo(x, bot + h * 0.45); p.lineTo(x + 3, bot)
        p.lineTo(x + 7, top); p.lineTo(x + 10 + iw + 2, top)
        c.drawPath(p, stroke=1, fill=0)
        self.inner.draw(c, x + 10, yb)


class Brace(_Tok):
    """Grande accolade ouvrante '{' verticale, dimensionnée sur son contenu."""
    def __init__(self, height, color=INK, w=6):
        self.h = height; self.color = color; self.wd = w

    def size_(self, c):
        return self.wd + 2, self.h / 2.0, self.h / 2.0

    def draw(self, c, x, yb):
        top = yb + self.h / 2.0; bot = yb - self.h / 2.0; midy = yb
        w = self.wd
        c.setStrokeColor(self.color); c.setLineWidth(0.9)
        p = c.beginPath()
        # branche haute
        p.moveTo(x + w, top)
        p.curveTo(x + w * 0.4, top, x + w * 0.55, midy + (top - midy) * 0.15, x + w * 0.5, midy + 3)
        # pointe centrale
        p.curveTo(x + w * 0.5, midy + 1, x, midy + 1, x, midy)
        p.curveTo(x, midy - 1, x + w * 0.5, midy - 1, x + w * 0.5, midy - 3)
        # branche basse
        p.curveTo(x + w * 0.55, midy - (midy - bot) * 0.15, x + w * 0.4, bot, x + w, bot)
        c.drawPath(p, stroke=1, fill=0)


class Stack(_Tok):
    """Empile plusieurs Row verticalement (aligné à gauche), centré verticalement."""
    def __init__(self, rows, gap=4, valign_center=True):
        self.rows = [r if isinstance(r, Row) else Row(r) for r in rows]
        self.gap = gap; self.valign_center = valign_center

    def size_(self, c):
        w = 0; total_h = 0; heights = []
        for r in self.rows:
            rw, ra, rd = r.size_(c); w = max(w, rw)
            hh = ra + rd; heights.append((ra, rd, hh)); total_h += hh
        total_h += self.gap * (len(self.rows) - 1)
        self._heights = heights; self._total = total_h
        return w, total_h / 2.0, total_h / 2.0

    def draw(self, c, x, yb):
        if not hasattr(self, "_total"):
            self.size_(c)
        cur_top = yb + self._total / 2.0
        for r, (ra, rd, hh) in zip(self.rows, self._heights):
            baseline = cur_top - ra
            r.draw(c, x, baseline)
            cur_top -= (hh + self.gap)

    def n_rows(self):
        return len(self.rows)


class Row(_Tok):
    def __init__(self, items):
        self.items = list(items) if isinstance(items, (list, tuple)) else [items]

    def size_(self, c):
        w = 0; asc = 0; desc = 0
        for it in self.items:
            iw, ia, idsc = it.size_(c)
            w += iw + 1.5; asc = max(asc, ia); desc = max(desc, idsc)
        return w, asc, desc

    def draw(self, c, x, yb):
        for it in self.items:
            iw, ia, idsc = it.size_(c)
            it.draw(c, x, yb); x += iw + 1.5


class Formula(Flowable):
    def __init__(self, row, lpad=0):
        super().__init__()
        self.row = row if isinstance(row, Row) else Row(row)
        self.lpad = lpad; self._w = self._a = self._d = 0

    def wrap(self, aw, ah):
        c = _canvas.Canvas(io.BytesIO())
        self._w, self._a, self._d = self.row.size_(c)
        self.width = self._w + self.lpad
        self.height = self._a + self._d + 2
        return (self.width, self.height)

    def draw(self):
        self.row.draw(self.canv, self.lpad, self._d + 1)


def txt(s, font="Helvetica", size=10, color=INK, sub=None, sup=None):
    return T(s, font, size, color, sub=sub, sup=sup)


def _t(s, **k):
    return txt(s, **k)


def nb(s):
    return txt(s, font="Helvetica-Bold")


# ============================================================
#  NOTATION SCIENTIFIQUE (a·10^n)
# ============================================================
def sci_tokens(value, color=INK, font="Helvetica", size=10):
    v = float(value)
    if v == 0:
        return [txt("0", font=font, size=size, color=color)]
    exp = int(math.floor(math.log10(abs(v))))
    n = 6 if exp >= 6 else (3 if exp >= 3 else 0)
    mant = v / (10 ** n)
    ms = f"{round(mant):d}" if abs(mant - round(mant)) < 1e-9 else f"{mant:.1f}".replace(".", ",")
    if n == 0:
        return [txt(ms, font=font, size=size, color=color)]
    return [txt(f"{ms}·10", font=font, size=size, color=color, sup=str(n))]


# ============================================================
#  ACCÈS AUX VALEURS (mêmes clés que poutre.py 2.20)
# ============================================================
def _g(values, key, default=None):
    return values.get(key, default)


def KB(base, bid):
    return f"b{bid}_{base}"


def KS(base, bid, sid):
    return f"b{bid}_sec{sid}_{base}"


def _bar_area_mm2(d):
    return math.pi * (d / 2.0) ** 2


def _brins_from_type(t):
    if "3 brins" in t:
        return 3
    if "2 brins" in t:
        return 2
    return 1


def _round_up_to_half_cm(x):
    try:
        return math.ceil(float(x) * 2.0) / 2.0
    except Exception:
        return x


def _get_fyk(values, bid):
    try:
        cur = int(float(_g(values, KB("fyk", bid), 500)))
    except Exception:
        cur = 500
    if cur not in (400, 500):
        cur = 500
    return float(cur), str(cur)


def _get_nlits(values, bid, sid, which):
    try:
        nl = int(_g(values, KS(f"nlits_{which}", bid, sid), 1) or 1)
    except Exception:
        nl = 1
    return max(1, min(MAX_LITS, nl))


def _lit_bars(values, bid, sid, which, i):
    if i == 1:
        n = int(_g(values, KS(f"n_as_{which}", bid, sid), 2) or 2)
        d = int(_g(values, KS(f"ø_as_{which}", bid, sid), 16) or 16)
    else:
        n = int(_g(values, KS(f"n_as_{which}_l{i}", bid, sid), 2) or 2)
        d = int(_g(values, KS(f"ø_as_{which}_l{i}", bid, sid), 16) or 16)
    return n, d


def _auto_dist_lit(values, bid, sid, which, i):
    """Réplique _auto_dist_lit de poutre.py 2.20."""
    if i == 1:
        enrob_beton = float(_g(values, KB("enrobage_beton", bid), 3.0) or 3.0)
        d_etrier = float(_g(values, "diam_etrier_mm", 8) or 8)
        jeu1 = float(_g(values, "jeu_enrobage_cm", 1.0) or 0.0)
        _, d1 = _lit_bars(values, bid, sid, which, 1)
        return (enrob_beton
                + _round_up_to_half_cm(d_etrier / 10.0)
                + _round_up_to_half_cm(d1 / 20.0)
                + jeu1)
    prev = _dist_lit(values, bid, sid, which, i - 1)
    _, d_prev = _lit_bars(values, bid, sid, which, i - 1)
    _, d_i = _lit_bars(values, bid, sid, which, i)
    jeuL = float(_g(values, "jeu_entre_lits_cm", 1.0) or 0.0)
    return (prev
            + _round_up_to_half_cm(d_prev / 20.0)
            + jeuL
            + _round_up_to_half_cm(d_i / 20.0))


def _dist_keys(bid, sid, which, i):
    if i == 1:
        return KS(f"enrob_calc_{which}", bid, sid), KS(f"enrob_calc_{which}_override", bid, sid)
    return KS(f"dist_{which}_l{i}", bid, sid), KS(f"dist_{which}_l{i}_override", bid, sid)


def _dist_lit(values, bid, sid, which, i):
    """Distance axe lit i / parement effective (override compris)."""
    key_val, key_ovr = _dist_keys(bid, sid, which, i)
    auto = _auto_dist_lit(values, bid, sid, which, i)
    if bool(_g(values, key_ovr, False)):
        try:
            return float(_g(values, key_val, auto) or auto)
        except Exception:
            return float(auto)
    return float(auto)


def _layers_geometry(values, bid, sid, which):
    """As_total (mm²), e_cdg (cm, parement->c.d.g.), liste de lits, detail."""
    nl = _get_nlits(values, bid, sid, which)
    lits = []
    As_tot = 0.0
    somme = 0.0
    parts = []
    for i in range(1, nl + 1):
        n, d = _lit_bars(values, bid, sid, which, i)
        e = _dist_lit(values, bid, sid, which, i)
        As_i = n * _bar_area_mm2(d)
        As_tot += As_i
        somme += As_i * e
        lits.append({"i": i, "n": n, "d": d, "e": e, "As": As_i})
        parts.append(f"{n}\u00d8{d}")
    e_cdg = (somme / As_tot) if As_tot > 0 else _dist_lit(values, bid, sid, which, 1)
    return {"As": As_tot, "e_cdg": e_cdg, "lits": lits, "detail": " + ".join(parts), "nl": nl}


def _shear_lines(values, bid, sid, reduced):
    if reduced:
        n_lines = int(_g(values, KS("shear_n_lines_r", bid, sid), 1) or 1); prefix = "shear_r_line"
    else:
        n_lines = int(_g(values, KS("shear_n_lines", bid, sid), 1) or 1); prefix = "shear_line"
    n_lines = max(1, n_lines)
    Ast = 0.0; parts = []; groups = []
    for i in range(n_lines):
        typ = str(_g(values, KS(f"{prefix}{i}_type", bid, sid), "Étriers (2 brins)"))
        n_c = int(_g(values, KS(f"{prefix}{i}_n", bid, sid), 1) or 1)
        diam = float(_g(values, KS(f"{prefix}{i}_d", bid, sid), 8) or 8)
        brins = _brins_from_type(typ)
        Ast += n_c * brins * _bar_area_mm2(diam)
        parts.append(f"{n_c}\u00d7 {typ} \u00d8{int(diam)}")
        groups.append({"type": typ, "n": n_c, "d": int(diam), "brins": brins})
    return Ast, " + ".join(parts), groups


def _first_stirrup(values, bid, sid):
    typ = str(_g(values, KS("shear_line0_type", bid, sid), "Étriers (2 brins)"))
    diam = int(float(_g(values, KS("shear_line0_d", bid, sid), 8) or 8))
    return {"type": typ, "d": diam, "brins": _brins_from_type(typ)}


# ============================================================
#  CALCUL SECTION (fidèle à poutre.py 2.20)
# ============================================================
def _compute_section(values, beton_data, bid, sid):
    beton = str(_g(values, KB("beton", bid), "C30/37"))
    if beton not in beton_data:
        beton = list(beton_data.keys())[0]
    bd = beton_data[beton]
    fck_cube = bd["fck_cube"]
    alpha_b = bd["alpha_b"]
    fck_cyl = float(bd.get("fck", 0.8 * fck_cube) or (0.8 * fck_cube))

    fyk, mu_ref = _get_fyk(values, bid)
    fyd = fyk / 1.5

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in bd:
        mu_key = "mu_a500" if "mu_a500" in bd else [k for k in bd if k.startswith("mu_a")][0]
    mu_val = bd[mu_key]

    b = float(_g(values, KB("b", bid), 20))
    h = float(_g(values, KB("h", bid), 40))
    enrob_beton = float(_g(values, KB("enrobage_beton", bid), 3.0) or 3.0)

    geo_inf = _layers_geometry(values, bid, sid, "inf")
    geo_sup = _layers_geometry(values, bid, sid, "sup")

    dist_l1_inf = _dist_lit(values, bid, sid, "inf", 1)
    dist_l1_sup = _dist_lit(values, bid, sid, "sup", 1)

    d_inf = h - geo_inf["e_cdg"]
    d_sup = h - geo_sup["e_cdg"]
    d_shear = h - min(dist_l1_inf, dist_l1_sup)
    d_calc_inf = max(d_inf, 0.1); d_calc_sup = max(d_sup, 0.1); d_calc_shear = max(d_shear, 0.1)
    geom_inf_ok = d_inf > 0; geom_sup_ok = d_sup > 0; geom_shear_ok = d_shear > 0

    tol = float(_g(values, "tau_tolerance_percent", 0.0) or 0.0)

    has_Msup = bool(_g(values, KS("ajouter_moment_sup", bid, sid), False))
    has_Vred = bool(_g(values, KS("ajouter_effort_reduit", bid, sid), False))
    M_inf = float(_g(values, KS("M_inf", bid, sid), 0.0) or 0.0)
    M_sup = float(_g(values, KS("M_sup", bid, sid), 0.0) or 0.0) if has_Msup else 0.0
    V = float(_g(values, KS("V", bid, sid), 0.0) or 0.0)
    V_lim = float(_g(values, KS("V_lim", bid, sid), 0.0) or 0.0) if has_Vred else 0.0
    has_Msup = has_Msup and (M_sup > 0)
    has_Vlim = has_Vred and (V_lim > 0)

    M_max = max(M_inf, M_sup)
    hmin = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10 if M_max > 0 else 0.0
    etat_h = "ok" if (hmin + dist_l1_inf <= h) else "nok"

    # As,min : 3 critères (EC2 / plancher / 0,25·As,req opposé), h partout
    fctm = 0.30 * (fck_cyl ** (2.0 / 3.0)) if fck_cyl > 0 else 0.0
    As_min_ec = 0.26 * fctm / fyk * b * h * 1e2
    As_min_plancher = 0.0013 * b * h * 1e2
    As_min_base = max(As_min_ec, As_min_plancher)
    As_max = 0.04 * b * h * 1e2

    As_req_inf = (M_inf * 1e6) / (fyd * 0.9 * d_calc_inf * 10) if M_inf > 0 else 0.0
    As_req_sup = (M_sup * 1e6) / (fyd * 0.9 * d_calc_sup * 10) if M_sup > 0 else 0.0
    As_min_inf = max(As_min_base, 0.25 * As_req_sup)
    As_min_sup = max(As_min_base, 0.25 * As_req_inf)

    As_inf = geo_inf["As"]; As_sup = geo_sup["As"]
    etat_inf = "ok" if (geom_inf_ok and As_inf >= max(As_req_inf, As_min_inf) and As_inf <= As_max) else "nok"
    etat_sup = "ok" if (geom_sup_ok and As_sup >= max(As_req_sup, As_min_sup) and As_sup <= As_max) else "nok"

    tau_1 = 0.016 * fck_cube / 1.05
    tau_2 = 0.032 * fck_cube / 1.05
    tau_4 = 0.064 * fck_cube / 1.05

    def shear_need(tau):
        if tau <= tau_1:
            return "Pas besoin d'étriers", "ok", "tau_adm,I", tau_1
        if tau <= tau_2:
            return "Besoin d'étriers", "ok", "tau_adm,II", tau_2
        if tau <= tau_4:
            return "Barres inclinées + étriers", "warn", "tau_adm,IV", tau_4
        return "Section insuffisante", "nok", "tau_adm,IV", tau_4

    def status_tol(value, limit):
        if limit <= 0:
            return "nok", ""
        if value <= limit:
            return "ok", ""
        lim2 = limit * (1.0 + max(0.0, tol) / 100.0)
        if value <= lim2:
            return "ok", f"Acceptable (tolérance +{tol:.0f}%)"
        return "nok", ""

    def build_shear(Vx, reduced):
        if Vx <= 0:
            return None
        tau = Vx * 1e3 / (0.75 * b * h * 100)
        besoin, etat_base, nom_lim, tau_lim = shear_need(tau)
        etat_tau, suf = (status_tol(tau, tau_lim) if tau > tau_lim else (etat_base, ""))
        Ast_e, summary, groups = _shear_lines(values, bid, sid, reduced=reduced)
        pas = float(_g(values, KS("shear_pas_r" if reduced else "shear_pas", bid, sid), 30.0) or 30.0)
        pas_th = Ast_e * fyd * (d_calc_shear * 10.0) / (Vx * 1e3) / 10.0 if Ast_e > 0 else 0.0
        s_max = min(0.75 * d_calc_shear, 30.0)
        pas_lim = min(pas_th, s_max) if pas_th > 0 else s_max
        etat_pas, suf_pas = status_tol(pas, pas_lim)
        if not geom_shear_ok:
            etat_tau = "nok"; etat_pas = "nok"
        return {"tau": tau, "besoin": besoin, "etat_tau": etat_tau, "nom_lim": nom_lim,
                "tau_lim": tau_lim, "suf": suf, "Ast": Ast_e, "summary": summary, "groups": groups,
                "pas": pas, "pas_th": pas_th, "s_max": s_max, "pas_lim": pas_lim,
                "etat_pas": etat_pas, "suf_pas": suf_pas, "V": Vx}

    shear = build_shear(V, False)
    shear_r = build_shear(V_lim, True) if has_Vlim else None

    states = [etat_h]
    if M_inf > 0:
        states.append(etat_inf)
    if has_Msup:
        states.append(etat_sup)
    if shear:
        states += [shear["etat_tau"], shear["etat_pas"]]
    if shear_r:
        states += [shear_r["etat_tau"], shear_r["etat_pas"]]
    etat_global = "nok" if any(s == "nok" for s in states) else ("warn" if any(s == "warn" for s in states) else "ok")

    return {
        "beton": beton, "fck": fck_cyl, "fck_cube": fck_cube, "alpha_b": alpha_b, "fctm": fctm,
        "fyk": fyk, "fyd": fyd, "mu_ref": mu_ref, "mu": mu_val,
        "b": b, "h": h, "enrob_beton": enrob_beton,
        "ei": geo_inf["e_cdg"], "es": geo_sup["e_cdg"],
        "dist_l1_inf": dist_l1_inf, "dist_l1_sup": dist_l1_sup,
        "di": d_inf, "ds": d_sup, "dsh": d_shear,
        "M_inf": M_inf, "M_sup": M_sup, "V": V, "V_lim": V_lim,
        "has_Msup": has_Msup, "has_Vlim": has_Vlim,
        "M_max": M_max, "hmin": hmin, "etat_h": etat_h,
        "As_min_ec": As_min_ec, "As_min_plancher": As_min_plancher, "As_max": As_max,
        "As_req_inf": As_req_inf, "As_req_sup": As_req_sup,
        "As_min_inf": As_min_inf, "As_min_sup": As_min_sup,
        "geo_inf": geo_inf, "geo_sup": geo_sup,
        "As_inf": As_inf, "As_sup": As_sup, "etat_inf": etat_inf, "etat_sup": etat_sup,
        "shear": shear, "shear_r": shear_r, "etat_global": etat_global,
    }


# ============================================================
#  COUPE DE SECTION (multi-lits, positions réelles)
# ============================================================
class SectionDrawing(Flowable):
    def __init__(self, R, stirrups, width, height, pal):
        super().__init__()
        self.R = R; self.stirrups = stirrups
        self.width = width; self.height = height; self.pal = pal

    def wrap(self, aw, ah):
        return (self.width, self.height)

    def _dash_axis(self, c, x1, y1, x2, y2):
        c.saveState()
        c.setStrokeColor(self.pal["axis"]); c.setLineWidth(0.4)
        c.setDash([6, 2, 1.5, 2])
        c.line(x1, y1, x2, y2)
        c.restoreState()

    def draw(self):
        c = self.canv; R = self.R; P = self.pal
        b_cm = float(R["b"]); h_cm = float(R["h"])

        pad_l, pad_t, pad_r, pad_b = 34, 16, 150, 22
        aw = self.width - pad_l - pad_r
        ah = self.height - pad_t - pad_b
        b_mm, h_mm = b_cm * 10.0, h_cm * 10.0
        sc = min(aw / b_mm, ah / h_mm)
        sw, sh = b_mm * sc, h_mm * sc
        x0 = pad_l + (aw - sw) / 2.0
        y0 = pad_b + (ah - sh) / 2.0

        c.saveState()
        # béton + hachures
        c.setFillColor(P["conc"]); c.setStrokeColor(P["conc_bd"]); c.setLineWidth(1.5)
        c.rect(x0, y0, sw, sh, stroke=1, fill=1)
        c.saveState()
        p = c.beginPath(); p.rect(x0, y0, sw, sh); c.clipPath(p, stroke=0, fill=0)
        c.setStrokeColor(P["hatch"]); c.setLineWidth(0.35)
        xx = x0 - sh
        while xx < x0 + sw:
            c.line(xx, y0, xx + sh, y0 + sh); xx += 6
        c.restoreState()
        c.setStrokeColor(P["conc_bd"]); c.setLineWidth(1.5)
        c.rect(x0, y0, sw, sh, stroke=1, fill=0)

        # axes
        ext = 7
        self._dash_axis(c, x0 + sw / 2, y0 - ext, x0 + sw / 2, y0 + sh + ext)
        self._dash_axis(c, x0 - ext, y0 + sh / 2, x0 + sw + ext, y0 + sh / 2)

        # enrobage étrier approx pour offset des barres
        enrob_beton = float(R.get("enrob_beton", 3.0))
        st_off = enrob_beton * 10.0 * sc
        st_main = self.stirrups[0] if self.stirrups else {"d": 8}
        stw_main = max(1.0, float(st_main.get("d", 8)) * sc)
        bar_off = st_off + stw_main + 1.0

        # étrier périmétrique
        for stg in self.stirrups:
            st_d = float(stg.get("d", 8))
            stw = max(1.0, st_d * sc)
            c.setStrokeColor(P["stirrup"]); c.setLineWidth(stw)
            rr = max(3.0, 2.0 * st_d * sc)
            c.roundRect(x0 + st_off, y0 + st_off, sw - 2 * st_off, sh - 2 * st_off, rr, stroke=1, fill=0)

        def _xs(n, d_mm, off):
            r = max(1.7, (d_mm * sc) / 2.0)
            inset = off + r + 1.0
            xa, xb = x0 + inset, x0 + sw - inset
            if n <= 1:
                return [(xa + xb) / 2.0], r
            return [xa + (xb - xa) * k / (n - 1) for k in range(n)], r

        def layer(n, d_mm, y_cm_from_bottom, fc, bd):
            if n <= 0:
                return None
            xs, r = _xs(n, d_mm, bar_off)
            yy = y0 + (y_cm_from_bottom / h_cm) * sh
            c.setFillColor(fc); c.setStrokeColor(bd); c.setLineWidth(0.5)
            for xc in xs:
                c.circle(xc, yy, r, stroke=1, fill=1)
            return yy

        geo_inf = R["geo_inf"]; geo_sup = R["geo_sup"]
        y_inf = []
        for lit in geo_inf["lits"]:
            col, bdc = LIT_COLORS[(lit["i"] - 1) % len(LIT_COLORS)]
            yy = layer(lit["n"], lit["d"], lit["e"], col, bdc)  # e = depuis le bas
            y_inf.append((yy, col, lit))
        y_sup = []
        for lit in geo_sup["lits"]:
            col, bdc = LIT_COLORS[(lit["i"] - 1) % len(LIT_COLORS)]
            yy = layer(lit["n"], lit["d"], h_cm - lit["e"], col, bdc)  # e = depuis le haut
            y_sup.append((yy, col, lit))

        # cotes b / h
        c.setStrokeColor(P["dim"]); c.setFillColor(P["dim"]); c.setLineWidth(0.6); c.setFont("Helvetica", 7.5)
        yb = y0 + sh + 10
        c.setDash(); c.line(x0, yb, x0 + sw, yb)
        for xx in (x0, x0 + sw):
            c.line(xx, yb - 2.5, xx, yb + 2.5)
        c.drawCentredString(x0 + sw / 2, yb + 3, f"b = {fn(b_cm,0)} cm")
        xl = x0 - 13
        c.line(xl, y0, xl, y0 + sh)
        for yy in (y0, y0 + sh):
            c.line(xl - 2.5, yy, xl + 2.5, yy)
        c.saveState(); c.translate(xl - 3, y0 + sh / 2); c.rotate(90)
        c.drawCentredString(0, 0, f"h = {fn(h_cm,0)} cm"); c.restoreState()

        # légende
        lx = x0 + sw + 16
        def leg(yy, col, label):
            if yy is None:
                return
            c.setStrokeColor(col); c.setLineWidth(0.5); c.setDash()
            c.line(x0 + sw, yy, lx - 3, yy)
            c.setFillColor(col); c.circle(lx + 2, yy, 2.2, stroke=0, fill=1)
            c.setFillColor(P["txt"]); c.setFont("Helvetica", 7.4)
            c.drawString(lx + 8, yy - 2.6, label)

        for yy, col, lit in reversed(y_sup):
            leg(yy, col, f"Lit {lit['i']} (sup.) : {lit['n']} \u00d8{lit['d']}")
        ymid = y0 + sh / 2.0
        c.setFillColor(P["stirrup"]); c.circle(lx + 2, ymid, 2.2, stroke=0, fill=1)
        c.setFillColor(P["txt"]); c.setFont("Helvetica", 7.4)
        etr = " · ".join(f"\u00d8{int(s.get('d', 8))}" for s in self.stirrups) or "\u00d88"
        c.drawString(lx + 8, ymid - 2.6, f"Étriers : {etr}")
        for yy, col, lit in reversed(y_inf):
            leg(yy, col, f"Lit {lit['i']} (inf.) : {lit['n']} \u00d8{lit['d']}")

        c.restoreState()


def stirrups_for(R, values, bid, sid):
    fs = _first_stirrup(values, bid, sid)
    Sh = R.get("shear")
    groups = (Sh or {}).get("groups") or [{"type": fs["type"], "n": 1, "d": fs["d"], "brins": fs["brins"]}]
    return [{"d": int(groups[0]["d"])}]


# ============================================================
#  STYLES
# ============================================================
def _S(n, sz, **kw):
    d = dict(fontName="Helvetica", fontSize=sz, textColor=INK, leading=sz * 1.35)
    d.update(kw)
    return ParagraphStyle(n, **d)


ST = {
    "h1":   _S("h1", 26, fontName="Helvetica-Bold", leading=30),
    "sub":  _S("sub", 10.5, textColor=MUTE),
    "beam": _S("beam", 15, fontName="Helvetica-Bold", leading=18),
    "sec":  _S("sec", 11.5, fontName="Helvetica-Bold", leading=14),
    "blk":  _S("blk", 11, fontName="Helvetica-Bold"),
    "lab":  _S("lab", 8.5, textColor=MUTE),
    "f":    _S("f", 9.6, leading=14),
    "cell": _S("cell", 8.8, leading=12),
    "cellb": _S("cellb", 8.8, fontName="Helvetica-Bold", leading=12),
    "kv":   _S("kv", 9, leading=12.5),
    "concl": _S("concl", 10, fontName="Helvetica-Bold", leading=13),
    "subt": _S("subt", 8.5, fontName="Helvetica-Bold", textColor=INK),
}

LABEL_FRAC = 0.34


# ============================================================
#  FLOWABLES DE BASE
# ============================================================
class HR(Flowable):
    def __init__(self, w, c=HAIR, t=0.5):
        super().__init__(); self.w = w; self.c = c; self.t = t
    def wrap(self, a, b):
        return (self.w, self.t + 2)
    def draw(self):
        self.canv.setStrokeColor(self.c); self.canv.setLineWidth(self.t); self.canv.line(0, 1, self.w, 1)


class Marker(Flowable):
    def __init__(self, store, key):
        super().__init__(); self.store = store; self.key = key
    def wrap(self, a, b):
        return (0, 0)
    def draw(self):
        self.store[self.key] = self.canv.getPageNumber()


class VerdictIcon(Flowable):
    def __init__(self, ok, color, r=6.5):
        super().__init__(); self.ok = ok; self.color = color; self.r = r
    def wrap(self, aw, ah):
        return (self.r * 2 + 2, self.r * 2 + 2)
    def draw(self):
        c = self.canv; r = self.r; cx = r + 1; cy = r
        c.setStrokeColor(self.color); c.setLineWidth(1.4); c.setFillColor(colors.white)
        c.circle(cx, cy, r, stroke=1, fill=0)
        c.setLineWidth(1.6); c.setLineCap(1); c.setLineJoin(1)
        if self.ok:
            p = c.beginPath(); p.moveTo(cx - r * 0.45, cy - r * 0.02)
            p.lineTo(cx - r * 0.08, cy - r * 0.42); p.lineTo(cx + r * 0.5, cy + r * 0.42)
            c.drawPath(p, stroke=1, fill=0)
        else:
            d = r * 0.42
            c.line(cx - d, cy - d, cx + d, cy + d); c.line(cx - d, cy + d, cx + d, cy - d)


# ============================================================
#  BLOCS / TABLES
# ============================================================
def fline(label, flow, cw):
    t = Table([[Paragraph(label, ST["lab"]), flow]], colWidths=[cw * LABEL_FRAC, cw * (1 - LABEL_FRAC)])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def reslines(rows, cw):
    data = []
    for lab, sym, val in rows:
        cell = f"<b>{sym}</b> = {val}" if sym else f"{val}"
        data.append([Paragraph(lab, ST["lab"]), Paragraph(cell, ST["kv"])])
    t = Table(data, colWidths=[cw * LABEL_FRAC, cw * (1 - LABEL_FRAC)])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
    return t


def conclu(et, cw, left_txt, ok=None):
    lp = Paragraph(f'<font color="{EDARK[et].hexval()}">{left_txt}</font>', ST["concl"])
    if ok is None:
        ok = (et == "ok")
    icon = VerdictIcon(ok, ECOL[et] if ok else ND)
    t = Table([[lp, icon]], colWidths=[cw - 24, 24])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), EPALE[et]),
        ("LEFTPADDING", (0, 0), (0, 0), 10), ("RIGHTPADDING", (0, 0), (0, 0), 6),
        ("LEFTPADDING", (1, 0), (1, 0), 0), ("RIGHTPADDING", (1, 0), (1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("ROUNDEDCORNERS", [4, 4, 4, 4])]))
    return t


def block(num, title, et, body, cw):
    iw = cw - 24
    head = Table([[Paragraph(f'<font color="{INK.hexval()}">{num}</font>&nbsp;&nbsp;{title}', ST["blk"])]], colWidths=[iw])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    inner = [head, HR(iw, ECOL[et], 1.4), Spacer(1, 7)] + body
    outer = Table([[inner]], colWidths=[cw])
    outer.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.white), ("BOX", (0, 0), (-1, -1), 0.8, HAIR),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [6, 6, 6, 6])]))
    return outer


# ============================================================
#  RÉCAP SECTION : caractéristiques (gauche) + coupe (droite)
# ============================================================
def carac(R, cw):
    def sub(t):
        return [Paragraph(t, ST["subt"]), Paragraph("", ST["cell"])]
    def kv(k, vv):
        return [Paragraph(k, ST["cell"]), Paragraph(str(vv), ST["cellb"])]
    rows = [sub("DIMENSIONS"),
            kv("Largeur b", f"{fn(R['b'],0)} cm"), kv("Hauteur h", f"{fn(R['h'],0)} cm"),
            kv("Enrobage béton", f"{fn(R['enrob_beton'],1)} cm"),
            sub("MATÉRIAUX"),
            kv("Béton", f"{R['beton']}"),
            kv("f<sub>ck</sub>", f"{fn(R['fck'],0)} N/mm{s2()}"),
            kv("Acier", f"B{int(R['fyk'])}"),
            kv("f<sub>yd</sub>", f"{fn(R['fyd'],0)} N/mm{s2()}"),
            sub("SOLLICITATIONS"),
            kv("M<sub>inf</sub>", f"{fn(R['M_inf'],1)} kNm")]
    if R["has_Msup"]:
        rows.append(kv("M<sup>sup</sup>", f"{fn(R['M_sup'],1)} kNm"))
    rows.append(kv("V", f"{fn(R['V'],1)} kN"))
    if R["has_Vlim"]:
        rows.append(kv("V<sub>réduit</sub>", f"{fn(R['V_lim'],1)} kN"))
    t = Table(rows, colWidths=[cw * 0.42, cw * 0.58])
    ts = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for i, r in enumerate(rows):
        if r[1].text == "":
            ts += [("SPAN", (0, i), (1, i)), ("LINEBELOW", (0, i), (-1, i), 0.8, INK),
                   ("TOPPADDING", (0, i), (-1, i), 7 if i > 0 else 0), ("BOTTOMPADDING", (0, i), (-1, i), 3)]
    t.setStyle(TableStyle(ts))
    return t


def recap(R, values, bid, sid, cw):
    half = cw * 0.44; gap = 14; rw = cw - half - gap
    left = carac(R, half)
    sts = stirrups_for(R, values, bid, sid)
    draw = SectionDrawing(R, sts, rw, 214, PAL)
    rcell = Table([[Paragraph('<font color="%s">COUPE DE SECTION</font>' % MUTE.hexval(), ST["subt"])], [draw]], colWidths=[rw])
    rcell.setStyle(TableStyle([("LEFTPADDING", (0, 0), (0, 0), 28), ("LEFTPADDING", (0, 1), (0, 1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2), ("TOPPADDING", (0, 1), (0, 1), 0), ("ALIGN", (0, 1), (0, 1), "CENTER")]))
    lay = Table([[left, "", rcell]], colWidths=[half, gap, rw])
    lay.setStyle(TableStyle([("VALIGN", (0, 0), (0, 0), "TOP"), ("VALIGN", (2, 0), (2, 0), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return lay


# ============================================================
#  BLOCS DE VÉRIFICATION
# ============================================================
def b_haut(R, cw):
    iw = cw - 24
    app = Formula(Row([_t("h", sub="min"), _t(" = "),
        Sqrt(Row([Frac(Row(Row(sci_tokens(R['M_max'] * 1e6)).items),
                       Row([_t(f"{fn(R['alpha_b'],2)} · {fn(R['b']*10,0)} · {fn(R['mu'],4)}")]))]), INK),
        _t("  =  "), nb(f"{fn(R['hmin'],1)} cm")]))
    body = [fline("Hauteur minimale", app, iw),
            Spacer(1, 7), HR(iw, HAIR, 0.5), Spacer(1, 7),
            reslines([("h,min + distance axe lit 1", "h<sub>min</sub> + d<sub>1,inf</sub>", f"{fn(R['hmin']+R['dist_l1_inf'],1)} cm"),
                      ("Hauteur de la poutre", "h", f"{fn(R['h'],0)} cm")], iw),
            Spacer(1, 5)]
    ok = R["etat_h"] == "ok"
    left = f"{fn(R['hmin']+R['dist_l1_inf'],1)} cm {'≤' if ok else '&gt;'} {fn(R['h'],0)} cm"
    body.append(conclu(R["etat_h"], iw, left, ok=ok))
    return block("1.", "Vérification de la hauteur", R["etat_h"], body, cw)


def _asmin_formula(R, which):
    """As,min = max{ 3 critères } — accolade + 3 lignes, résultat aligné sur A_s,min.
    Même police (10) que A_s,req."""
    b_mm = R["b"] * 10.0; h_mm = R["h"] * 10.0
    fctm = R["fctm"]; fyk = R["fyk"]
    ec = R["As_min_ec"]; pl = R["As_min_plancher"]
    as_min = R["As_min_inf"] if which == "inf" else R["As_min_sup"]
    as_req_opp = R["As_req_sup"] if which == "inf" else R["As_req_inf"]
    quart = 0.25 * as_req_opp
    face_opp = "sup" if which == "inf" else "inf"

    sz = 10  # même taille que A_s,req

    def _tt(s, **k):
        return txt(s, size=sz, **k)

    # 3 lignes de critères
    line1 = Row([_tt("0,26 · "),
                 Frac(Row([_tt(f"{fn(fctm,1)}")]), Row([_tt(f"{int(fyk)}")]), pad=2),
                 _tt(f" · {fn(b_mm,0)} · {fn(h_mm,0)} = {fn(ec,0)} mm", sup="2")])
    line2 = Row([_tt(f"0,0013 · {fn(b_mm,0)} · {fn(h_mm,0)} = {fn(pl,0)} mm", sup="2")])
    line3 = Row([_tt("0,25 · A", sub="s,req"), _tt(f",{face_opp} = 0,25 · {fn(as_req_opp,0)} = {fn(quart,0)} mm", sup="2")])

    stack = Stack([line1, line2, line3], gap=5)

    # hauteur de l'accolade ~ hauteur du stack
    c0 = _canvas.Canvas(io.BytesIO())
    _, sa, sd = stack.size_(c0)
    brace = Brace(sa + sd, INK, w=6)

    row = Row([
        _tt("A", sub="s,min"), _tt(" = max "),
        brace,
        _tt("  "),
        stack,
        _tt("   =  "),
        txt(f"{fn(as_min,0)} mm", font="Helvetica-Bold", size=sz, sup="2"),
    ])
    return Formula(row)


def b_arm(R, cw, which):
    iw = cw - 24
    if which == "inf":
        title = "Armatures inférieures"; M = R["M_inf"]; Ar = R["As_req_inf"]; geo = R["geo_inf"]; d = R["di"]; et = R["etat_inf"]; nn = "2."; As_min = R["As_min_inf"]
    else:
        title = "Armatures supérieures"; M = R["M_sup"]; Ar = R["As_req_sup"]; geo = R["geo_sup"]; d = R["ds"]; et = R["etat_sup"]; nn = "3."; As_min = R["As_min_sup"]

    nl = geo["nl"]
    # hauteur utile : c.d.g. si plusieurs lits
    if nl > 1:
        dlit = Formula(Row([_t("d", sub="u"), _t(f" = {fn(R['h'],0)} − {fn(geo['e_cdg'],1)} = "), nb(f"{fn(d,1)} cm"),
                            _t(f"   (c.d.g. de {nl} lits)", color=MUTE, size=8.5)]))
    else:
        e1 = geo["lits"][0]["e"]
        dlit = Formula(Row([_t("d", sub="u"), _t(f" = {fn(R['h'],0)} − {fn(e1,1)} = "), nb(f"{fn(d,1)} cm")]))

    app = Formula(Row([_t("A", sub="s,req"), _t(" = "),
        Frac(Row(Row(sci_tokens(M * 1e6)).items), Row([_t(f"{fn(R['fyd'],1)} · 0,9 · {fn(d*10,0)}")])),
        _t("  =  "), txt(f"{fn(Ar,0)} mm", font="Helvetica-Bold", sup="2")]))

    choix = f"{geo['detail']} ({fn(geo['As'],0)} mm{s2()})" + (f" · {nl} lits" if nl > 1 else "")

    asmin_f = _asmin_formula(R, which)
    asmax_f = Formula(Row([
        _t("A", sub="s,max"), _t(f" = 0,04 · {fn(R['b']*10,0)} · {fn(R['h']*10,0)} = "),
        txt(f"{fn(R['As_max'],0)} mm", font="Helvetica-Bold", sup="2")]))

    body = [fline("Moment appliqué",
                  Formula(Row([_t("M", sub=("inf" if which == "inf" else None), sup=(None if which == "inf" else "sup")),
                               _t("  =  "), nb(f"{fn(M,1)} kNm")])), iw),
            Spacer(1, 2),
            fline("Hauteur utile", dlit, iw), Spacer(1, 2),
            fline("Acier requis", app, iw), Spacer(1, 4),
            fline("Section d'acier min", asmin_f, iw), Spacer(1, 3),
            fline("Section d'acier max", asmax_f, iw),
            Spacer(1, 7), HR(iw, HAIR, 0.5), Spacer(1, 7),
            reslines([("Acier requis", "A<sub>s,req</sub>", f"{fn(Ar,0)} mm{s2()}"),
                      ("Acier minimal", "A<sub>s,min</sub>", f"{fn(As_min,0)} mm{s2()}"),
                      ("On prend", "", choix)], iw),
            Spacer(1, 5)]
    ok = et == "ok"
    besoin = max(Ar, As_min)
    left = f"{fn(geo['As'],0)} mm{s2()} {'≥' if ok else '&lt;'} {fn(besoin,0)} mm{s2()}"
    body.append(conclu(et, iw, left, ok=ok))
    return block(nn, title, et, body, cw)


def b_shear(R, cw, reduced=False):
    iw = cw - 24
    Sh = R["shear_r"] if reduced else R["shear"]
    nn = "5." if reduced else "4."; suff = " réduit" if reduced else ""
    app = Formula(Row([_t("τ = "),
        Frac(Row(Row(sci_tokens(Sh['V'] * 1e3)).items), Row([_t(f"0,75 · {fn(R['b']*10,0)} · {fn(R['h']*10,0)}")])),
        _t("  =  "), txt(f"{fn(Sh['tau'],2)} N/mm", font="Helvetica-Bold", sup="2")]))
    if Sh['Ast'] > 0 and Sh['V'] > 0:
        sthapp = Formula(Row([_t("s", sub="th"), _t(" = "),
            Frac(Row([_t(f"{fn(Sh['Ast'],1)} · {fn(R['fyd'],1)} · {fn(R['dsh']*10,0)}")]), Row(Row(sci_tokens(Sh['V'] * 1e3)).items)),
            _t("  =  "), nb(f"{fn(Sh['pas_th'],1)} cm")]))
    else:
        sthapp = Formula(Row([_t("s", sub="th"), _t("  =  "), nb("—")]))
    etr = f"{Sh['summary']} ({fn(Sh['Ast'],1)} mm{s2()})"
    okt = Sh["tau"] <= Sh["tau_lim"]
    okp = Sh["pas"] <= Sh["pas_lim"]
    et_tau = "ok" if okt else ("warn" if Sh["etat_tau"] == "warn" else "nok")
    body = [fline("Contrainte tangentielle", app, iw),
            Spacer(1, 7), HR(iw, HAIR, 0.5), Spacer(1, 7),
            reslines([("Contrainte admissible", "τ<sub>adm</sub>", f"{fn(Sh['tau_lim'],2)} N/mm{s2()}")], iw),
            Spacer(1, 4),
            conclu(et_tau, iw, f"{fn(Sh['tau'],2)} N/mm{s2()} {'≤' if okt else '&gt;'} {fn(Sh['tau_lim'],2)} N/mm{s2()}", ok=okt),
            Spacer(1, 9), Paragraph("<b>Étriers</b>", ST["f"]), Spacer(1, 4),
            reslines([("On prend (étrier)", "", etr)], iw),
            Spacer(1, 2), fline("Pas théorique", sthapp, iw),
            Spacer(1, 2), fline("Pas maximal",
                Formula(Row([_t("s", sub="max"), _t(" = min(0,75 · d ; 30) = "), nb(f"{fn(Sh['s_max'],1)} cm")])), iw),
            Spacer(1, 2), fline("Pas retenu", Formula(Row([_t("s"), _t("  =  "), nb(f"{fn(Sh['pas'],1)} cm")])), iw),
            Spacer(1, 5)]
    et = "nok" if "nok" in (Sh["etat_tau"], Sh["etat_pas"]) else ("warn" if "warn" in (Sh["etat_tau"], Sh["etat_pas"]) else "ok")
    et_pas = "ok" if okp else "nok"
    left = f"pas {fn(Sh['pas'],1)} cm {'≤' if okp else '&gt;'} {fn(Sh['pas_lim'],1)} cm"
    body.append(conclu(et_pas, iw, left, ok=okp))
    return block(nn, f"Effort tranchant{suff} — étriers", et, body, cw)


# ============================================================
#  BANDEAUX POUTRE / SECTION (pastel)
# ============================================================
def beam_banner(txt_, cw):
    st = ParagraphStyle("bb", parent=ST["beam"], textColor=BEAM_TX)
    return Table([[Paragraph(txt_, st)]], colWidths=[cw],
        style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), BEAM_BG), ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("ROUNDEDCORNERS", [5, 5, 5, 5])]))


def sec_banner(txt_, cw):
    st = ParagraphStyle("sb", parent=ST["sec"], textColor=SEC_TX)
    return Table([[Paragraph(txt_, st)]], colWidths=[cw],
        style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), SEC_BG), ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ROUNDEDCORNERS", [5, 5, 5, 5])]))


# ============================================================
#  DOC TEMPLATE (en-tête / pied de page)
# ============================================================
class NoteDoc(BaseDocTemplate):
    def __init__(self, filename, infos, **kw):
        self.infos = infos or {}
        super().__init__(filename, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                         topMargin=24 * mm, bottomMargin=18 * mm, **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="m")
        self.addPageTemplates([PageTemplate(id="all", frames=[fr], onPage=self._decor)])

    def _decor(self, c, doc):
        w, h = A4; c.saveState()
        c.setFillColor(INK); c.setFont("Helvetica-Bold", 10.5)
        c.drawString(18 * mm, h - 12 * mm, "Bureau méthodes et stabilité Valens")
        c.setFillColor(MUTE); c.setFont("Helvetica", 8)
        c.drawString(18 * mm, h - 16.5 * mm, f"Rédigé par : {self.infos.get('initiales','')}")
        c.drawRightString(w - 18 * mm, h - 12 * mm, f"{self.infos.get('nom_projet','')}")
        c.drawRightString(w - 18 * mm, h - 16.5 * mm, f"{self.infos.get('partie','')}")
        c.setStrokeColor(INK); c.setLineWidth(1.6); c.line(18 * mm, h - 18.5 * mm, w - 18 * mm, h - 18.5 * mm)
        c.setStrokeColor(HAIR); c.setLineWidth(0.5); c.line(18 * mm, 14 * mm, w - 18 * mm, 14 * mm)
        c.setFillColor(MUTE); c.setFont("Helvetica", 7.5)
        date = self.infos.get("date") or datetime.today().strftime("%d/%m/%Y")
        c.drawString(18 * mm, 9.5 * mm, f"{date} · indice {self.infos.get('indice','0')}")
        c.drawRightString(w - 18 * mm, 9.5 * mm, f"Page {doc.page}")
        c.restoreState()


# ============================================================
#  PAGE DE GARDE
# ============================================================
def _cover(infos, beams, values, beton_data, cw, pages):
    h1c = ParagraphStyle("h1c", parent=ST["h1"], alignment=TA_CENTER)
    subc = ParagraphStyle("subc", parent=ST["sub"], alignment=TA_CENTER, fontSize=14, leading=18, textColor=INK)
    st = [Spacer(1, 38 * mm),
          Paragraph(str(infos.get("nom_projet", "") or "Projet"), h1c),
          Spacer(1, 4), Paragraph("Note de calcul", subc),
          Spacer(1, 16), HR(cw, INK, 2), Spacer(1, 20)]
    info = [("Projet", infos.get("nom_projet") or "—"), ("Partie", infos.get("partie") or "—")]
    for k, vv in info:
        st.append(Table([[Paragraph(f'<font color="{MUTE.hexval()}">{k.upper()}</font>', ST["lab"]),
                          Paragraph(str(vv), ST["cellb"])]], colWidths=[cw * 0.3, cw * 0.7],
            style=TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.5, HAIR), ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("TOPPADDING", (0, 0), (-1, 0), 5), ("LEFTPADDING", (0, 0), (-1, -1), 0)])))
    st += [Spacer(1, 30), Paragraph("SOMMAIRE", ST["subt"]), Spacer(1, 8)]
    sm = [[Paragraph("<b>POUTRE</b>", ST["lab"]), Paragraph("<b>SECTIONS</b>", ST["lab"]),
           Paragraph("<b>BÉTON / ACIER</b>", ST["lab"]), Paragraph("<b>ÉTAT</b>", ST["lab"]), Paragraph("<b>PAGE</b>", ST["lab"])]]
    for b in beams:
        bid = int(b["id"])
        secs = ", ".join(str(_g(values, f"meta_b{bid}_nom_{int(s['id'])}", s.get("nom", ""))) for s in b.get("sections", []))
        ss = [_compute_section(values, beton_data, bid, int(s["id"]))["etat_global"] for s in b.get("sections", [])]
        eg = "nok" if "nok" in ss else ("warn" if "warn" in ss else "ok")
        pg = pages.get(bid)
        sm.append([Paragraph(str(_g(values, f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}"))), ST["cellb"]),
                   Paragraph(secs, ST["cell"]),
                   Paragraph(f"{_g(values, KB('beton', bid), '—')} / B{_g(values, KB('fyk', bid), '500')}", ST["cell"]),
                   Paragraph(f'<font color="{ECOL[eg].hexval()}"><b>{ELAB[eg]}</b></font>', ST["cell"]),
                   Paragraph(f"p.{pg}" if pg else "—", ST["cellb"])])
    t = Table(sm, colWidths=[cw * 0.24, cw * 0.34, cw * 0.20, cw * 0.13, cw * 0.09])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 1, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.4, HAIR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    st += [t, PageBreak()]
    return st


# ============================================================
#  CONSTRUCTION DU STORY
# ============================================================
def _build_story(beams, values, beton_data, infos, cw, pages, store):
    story = _cover(infos, beams, values, beton_data, cw, pages)
    for bi, b in enumerate(beams):
        bid = int(b["id"])
        if bi > 0:
            story.append(PageBreak())
        story.append(Marker(store, bid))
        story.append(beam_banner(str(_g(values, f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}"))), cw))
        story.append(Spacer(1, 10))
        sections = b.get("sections", [])
        for si, s in enumerate(sections):
            sid = int(s["id"])
            raw = str(_g(values, f"meta_b{bid}_nom_{sid}", s.get("nom", f"Section {sid}")))
            snom = raw if raw.strip().lower().startswith("section") else f"Section {raw}"
            R = _compute_section(values, beton_data, bid, sid)
            blocs = [b_haut(R, cw)]
            if R["M_inf"] > 0:
                blocs.append(b_arm(R, cw, "inf"))
            if R["has_Msup"]:
                blocs.append(b_arm(R, cw, "sup"))
            if R["shear"]:
                blocs.append(b_shear(R, cw, False))
            if R["shear_r"]:
                blocs.append(b_shear(R, cw, True))
            intro = [sec_banner(snom, cw), Spacer(1, 6), recap(R, values, bid, sid, cw), Spacer(1, 12), blocs[0]]
            story.append(KeepTogether(intro))
            for blk in blocs[1:]:
                story.append(Spacer(1, 12)); story.append(KeepTogether([blk]))
            if si < len(sections) - 1:
                story.append(Spacer(1, 16))
    return story


# ============================================================
#  API PRINCIPALE
# ============================================================
def generer_rapport_pdf(beams, values, beton_data, infos=None, output_path=None):
    infos = infos or {}
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".pdf", prefix="note_poutre_")
        os.close(fd)

    # passe 1 : mesure des pages de début de poutre
    pages = {}
    tmp = output_path + ".pass1.tmp"
    d1 = NoteDoc(tmp, infos); cw = d1.width
    d1.build(_build_story(beams, values, beton_data, infos, cw, pages={}, store=pages))
    try:
        os.remove(tmp)
    except OSError:
        pass

    # passe 2 : build final avec numéros de page
    d2 = NoteDoc(output_path, infos); cw = d2.width
    d2.build(_build_story(beams, values, beton_data, infos, cw, pages=dict(pages), store={}))
    return output_path

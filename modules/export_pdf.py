# -*- coding: utf-8 -*-
# ============================================================
#  export_pdf.py — Note de calcul PDF (poutre béton armé)
#  VERSION 3.0 (mise en page ndc_pdf)
#
#  Évolutions vs 2.39 :
#   - GÉNÉRATION : generer_rapport_pdf (même signature, même appelant)
#     produit désormais la note via le paquet `ndc_pdf/` : page de garde
#     A4 portrait (cartouche + sommaire), une planche A4 paysage par
#     section (coupe cotée à gauche sur fond gris, calculs sur deux
#     colonnes en flux continu, conclusions vert / ocre / rouge).
#   - AUCUN CALCUL MODIFIÉ : les valeurs viennent de _compute_section,
#     inchangé et fidèle à poutre.py. La mise en page transcrit.
#   - Si une planche déborde (doc.warnings), la note est régénérée en
#     3 colonnes — remède prescrit par la maquette. Les avertissements
#     restants sont exposés dans DERNIERS_AVERTISSEMENTS.
#   - Toutes les primitives platypus (SectionDrawing, blocs, bandeaux,
#     NoteDoc…) sont CONSERVÉES : export_pdf_dalle.py les importe.
#
#  VERSION 2.39 (alignée sur poutre.py 2.39)
#
#  Évolutions vs 2.38 :
#   - EN-TÊTE : suppression de la ligne "Rédigé par :" — seul
#     "Bureau d'Études Valens" est conservé.
#   - TAUX D'ARMATURE : le TA global de la poutre est affiché à droite
#     du bandeau de poutre ("T.A. = xxx kg/m³") UNIQUEMENT si l'option
#     "Envoyer dans la note de calcul" (taux_arm_pdf) est activée.
#     Pas de TA par section dans le PDF.
#   - VÉRIFICATION DE LA HAUTEUR : formule hᵤ,min inchangée ; la ligne
#     "h_u,min + distance axe lit 1" devient "Hauteur minimale de la
#     poutre = hᵤ,min + CDG armatures = a + b = c cm" avec le CDG RÉEL
#     des armatures (face du moment dimensionnant, inf. ou sup.).
#     Conclusion : "Hauteur de la poutre : X cm ≥ hauteur minimale de
#     la poutre : Y cm".
#   - ARMATURES : "Moment appliqué" -> "Moment inférieur" / "Moment
#     supérieur". La hauteur utile d_u = h − CDG réel (déjà pondéré
#     Σ As·e / Σ As sur tous les lits).
#   - ÉTRIERS : libellé clair "Étrier Ø12" (normalisé), section Asw
#     affichée, calcul du pas admissible s_adm = min(s_th ; s_max)
#     explicité.
#   - COUPE DE SECTION : croix du CDG affichée UNIQUEMENT s'il y a
#     plusieurs lits sur la face (position réelle du CDG pondéré).
#
#  Évolutions vs 2.37 :
#   - yG imposé pris en compte uniquement si le mode auto est désactivé
#     côté application (drapeau ycdg_auto_*). Le TA global de la poutre
#     et le TA par section restent gérés côté application (non exportés).
#
#  Évolutions vs 2.35 :
#   - Tableau matériaux : lignes "Coefficient acier ELS : γs = 1,50"
#     et "Contrainte de calcul acier : fyd = ...".
#   - Le taux d'armature n'est PAS exporté dans le PDF (en attente du
#     calcul global de la poutre avec longueurs de sections).
#
#  Évolutions vs 2.34 :
#   - CONCLUSIONS explicites (hauteur, armatures, tau, pas) sans
#     pourcentages (les %% restent dans l'application uniquement).
#   - LÉGENDES de la coupe simplifiées : "Lit 1 : 3 Ø16" (sans sup./inf.),
#     "Étrier : Ø8 — 15 cm" (sans positions), "Armature de peau : 2×n Ød".
#   - Intitulé matériaux : "coef. acier ELS" (plus de symbole γs).
#   - Style graphique de la coupe inchangé.
#
#  Évolutions vs 2.33 :
#   - En-tête de page : "Bureau d'Études Valens".
#   - COUPE À L'ÉCHELLE EXACTE : suppression des marges fixes ±1-2 px ;
#     étrier tracé sur sa LIGNE MOYENNE (nu extérieur exactement à
#     l'enrobage) ; axe des barres = enrobage + Ø étrier + Ø barre/2,
#     tout à l'échelle. Positions verticales déjà exactes.
#   - ÉTRIERS : tous au même niveau (traverses hautes alignées, basses
#     alignées) — seule la largeur varie ; couleurs alternées gris
#     foncé / gris clair. "Étriers (3 brins)" supprimé (migré côté app).
#   - ÉPINGLES : décalées sur le côté de la barre (l'armature reste
#     visible), crochets orientés vers la barre.
#   - ARMATURES TECHNOLOGIQUES (peau) : réparties automatiquement de
#     chaque côté si l'écart vertical lit1 inf / lit1 sup dépasse
#     l'espacement max (paramètres avancés), à l'intérieur des étriers,
#     dessinées SOUS les étriers, couleur dédiée, en légende.
#     DESSIN UNIQUEMENT — aucun calcul modifié.
#   - LÉGENDE : pas affiché pour les étriers ("Étrier Ø8 @15 cm").
#
#  Évolutions vs 2.32 :
#   - ÉTRIERS / ÉPINGLES POSITIONNÉS : chaque ligne porte une position
#     "de barre X → à barre Y" (barres du lit 1 inférieur). La coupe de
#     section dessine exactement chaque étrier (rectangle limité aux
#     barres choisies, étriers imbriqués distingués par un léger
#     décalage) et chaque épingle (brin vertical sur une barre, ou
#     agrafe horizontale entre deux barres). Par défaut (1 → n), le
#     rendu périmétrique actuel est conservé à l'identique.
#   - "Nbr. cadres" supprimé côté app (1 ligne = 1 étrier) ; lecture
#     conservée ici avec défaut 1 pour compat d'anciens fichiers.
#   - Aucun calcul modifié.
#
#  Corrections principales vs 2.20 :
#   - ARMATURES SUPÉRIEURES : elles n'apparaissaient plus dans le
#     rapport car le bloc dépendait de l'ancienne case à cocher
#     "ajouter_moment_sup" (supprimée de l'application). Les blocs
#     armatures inférieures ET supérieures sont désormais TOUJOURS
#     inclus, comme dans l'application.
#   - fyd = fyk / γs (clé "gamma_s", défaut 1,5) au lieu de fyk/1,5
#     codé en dur.
#   - DISTANCE AUTO LIT 1 : enrobage + (Ø étrier + demi-Ø barre lit 1)
#     arrondi ensemble au 0,5 cm sup. + jeu premier lit. Le Ø étrier
#     est lu dans la configuration des étriers de la section
#     (le paramètre global "diam_etrier_mm" n'existe plus).
#   - M_sup compte dès qu'il est > 0 (plus de case à cocher).
#   - EFFORT TRANCHANT RÉDUIT : concept supprimé (comme dans l'app).
#   - TOLÉRANCE DE DÉPASSEMENT : supprimée (comparaisons strictes).
#   - État global de section : inclut toujours inf. ET sup. (cohérent
#     avec l'application).
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

# étriers : couleurs alternées (1er foncé, 2e clair, puis alternance)
ETRIER_COLORS = [colors.HexColor("#4a4e57"), colors.HexColor("#b3b8c2")]
EPINGLE_COLOR = colors.HexColor("#6b6f76")
PEAU_COLOR = colors.HexColor("#7b4fa6")      # armatures technologiques (peau)
PEAU_BORDER = colors.HexColor("#553675")

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
#  ACCÈS AUX VALEURS (mêmes clés que poutre.py 2.32)
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
    t = str(t)
    if "3 brins" in t:
        return 3
    if "pingle" in t or "1 brin" in t:
        return 1
    return 2


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


def _get_gamma_s(values):
    """Coefficient acier γs (fyd = fyk / γs), défaut 1,5 — identique à poutre.py."""
    try:
        gs = float(_g(values, "gamma_s", 1.5) or 1.5)
    except Exception:
        gs = 1.5
    return gs if gs > 0 else 1.5


def _stirrup_diam_mm(values, bid, sid):
    """Ø étrier (mm) lu dans la config des étriers de la section (Ø max)."""
    n_lines = max(1, int(_g(values, KS("shear_n_lines", bid, sid), 1) or 1))
    diams = []
    for i in range(n_lines):
        try:
            diams.append(float(_g(values, KS(f"shear_line{i}_d", bid, sid), 8) or 8))
        except Exception:
            pass
    return max(diams) if diams else 8.0


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
    """Réplique _auto_dist_lit de poutre.py 2.32."""
    if i == 1:
        enrob_beton = float(_g(values, KB("enrobage_beton", bid), 3.0) or 3.0)
        d_etrier = _stirrup_diam_mm(values, bid, sid)
        jeu1 = float(_g(values, "jeu_enrobage_cm", 1.0) or 0.0)
        _, d1 = _lit_bars(values, bid, sid, which, 1)
        return (enrob_beton
                + _round_up_to_half_cm(d_etrier / 10.0 + d1 / 20.0)
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
    """Distance axe lit i effective (override compris)."""
    key_val, key_ovr = _dist_keys(bid, sid, which, i)
    auto = _auto_dist_lit(values, bid, sid, which, i)
    if bool(_g(values, key_ovr, False)):
        try:
            return float(_g(values, key_val, auto) or auto)
        except Exception:
            return float(auto)
    return float(auto)


def _layers_geometry(values, bid, sid, which):
    """As_total (mm²), e_cdg (cm, parement->c.d.g.), liste de lits, detail.
    Si un yG manuel est saisi dans l'application (clé ycdg_*), il
    remplace la valeur calculée — cohérent avec poutre.py."""
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
    # yG imposé uniquement si l'utilisateur a désactivé le mode auto.
    if not bool(_g(values, KS(f"ycdg_auto_{which}", bid, sid), False)):
        raw = str(_g(values, KS(f"ycdg_{which}", bid, sid), "") or "").strip()
        if raw:
            try:
                v = float(raw.replace(",", "."))
                if v > 0:
                    e_cdg = v
            except Exception:
                pass
    return {"As": As_tot, "e_cdg": e_cdg, "lits": lits, "detail": " + ".join(parts), "nl": nl}


def _shear_lines(values, bid, sid):
    n_lines = max(1, int(_g(values, KS("shear_n_lines", bid, sid), 1) or 1))
    prefix = "shear_line"
    Ast = 0.0; parts = []; groups = []
    for i in range(n_lines):
        typ = str(_g(values, KS(f"{prefix}{i}_type", bid, sid), "Étriers (2 brins)"))
        # 'Nbr. cadres' n'existe plus (1 ligne = 1 étrier) ; lecture avec
        # défaut 1 conservée pour compat d'anciens fichiers non migrés.
        n_c = int(_g(values, KS(f"{prefix}{i}_n", bid, sid), 1) or 1)
        diam = float(_g(values, KS(f"{prefix}{i}_d", bid, sid), 8) or 8)
        try:
            frm = int(float(_g(values, KS(f"{prefix}{i}_from", bid, sid), 0) or 0)) or None
        except Exception:
            frm = None
        try:
            to = int(float(_g(values, KS(f"{prefix}{i}_to", bid, sid), 0) or 0)) or None
        except Exception:
            to = None
        brins = _brins_from_type(typ)
        base = "Épingle" if brins == 1 else "Étrier"   # libellé clair, normalisé
        Ast += n_c * brins * _bar_area_mm2(diam)
        parts.append((f"{n_c}\u00d7 " if n_c > 1 else "") + f"{base} \u00d8{int(diam)}")
        for _ in range(max(1, n_c)):
            groups.append({"type": typ, "d": int(diam), "brins": brins, "from": frm, "to": to})
    return Ast, " + ".join(parts), groups


def _first_stirrup(values, bid, sid):
    typ = str(_g(values, KS("shear_line0_type", bid, sid), "Étriers (2 brins)"))
    diam = int(float(_g(values, KS("shear_line0_d", bid, sid), 8) or 8))
    return {"type": typ, "d": diam, "brins": _brins_from_type(typ)}


# ============================================================
#  TAUX D'ARMATURE GLOBAL DE LA POUTRE (v2.39)
#  Exporté dans le PDF UNIQUEMENT si l'option "Envoyer dans la note
#  de calcul" (taux_arm_pdf) est activée. Mêmes formules que poutre.py
#  (_section_poids_vol / _taux_armature_global). Pas de TA par section.
# ============================================================
RHO_ACIER = 7850.0  # kg/m³


def _masse_lin_kg_m(d_mm):
    """Masse linéique d'une barre (kg/m) : ρ·π·d²/4."""
    return RHO_ACIER * math.pi * (float(d_mm) / 1000.0) ** 2 / 4.0


def _section_poids_vol(values, bid, sid):
    """(poids acier majoré kg/m, volume béton m³/m) d'une section."""
    b = float(_g(values, KB("b", bid), 20))
    h = float(_g(values, KB("h", bid), 40))
    enrob = float(_g(values, KB("enrobage_beton", bid), 3.0) or 3.0)
    maj = float(_g(values, "taux_arm_major_pct", 5.0) or 0.0)
    retour = float(_g(values, "taux_retour_etrier_cm", 10.0) or 0.0)

    poids = 0.0
    # longitudinales : 1 m par barre au mètre courant
    for which in ("inf", "sup"):
        nl = _get_nlits(values, bid, sid, which)
        for i in range(1, nl + 1):
            n, dmm = _lit_bars(values, bid, sid, which, i)
            poids += n * 1.0 * _masse_lin_kg_m(dmm)

    # transversales : n_par_m = 100 / pas
    pas = float(_g(values, KS("shear_pas", bid, sid), 30.0) or 30.0)
    n_par_m = (100.0 / pas) if pas > 0 else 0.0
    n_lines = max(1, int(_g(values, KS("shear_n_lines", bid, sid), 1) or 1))
    n1, d1 = _lit_bars(values, bid, sid, "inf", 1)
    d_et_max = _stirrup_diam_mm(values, bid, sid)
    inset = enrob + d_et_max / 10.0 + d1 / 20.0
    xs = [inset + (b - 2 * inset) * k / (n1 - 1) for k in range(n1)] if n1 > 1 else [b / 2.0]
    for i in range(n_lines):
        typ = str(_g(values, KS(f"shear_line{i}_type", bid, sid), "Étrier"))
        dmm = float(_g(values, KS(f"shear_line{i}_d", bid, sid), 10) or 10)
        try:
            f = int(float(_g(values, KS(f"shear_line{i}_from", bid, sid), 1) or 1))
        except Exception:
            f = 1
        try:
            t = int(float(_g(values, KS(f"shear_line{i}_to", bid, sid), f) or f))
        except Exception:
            t = f
        f = max(1, min(n1, f)); t = max(1, min(n1, t))
        if f > t:
            f, t = t, f
        if _brins_from_type(typ) == 1:
            L_un_cm = ((h - 2 * enrob) if f == t else abs(xs[t - 1] - xs[f - 1])) + 2 * retour
        else:
            w_ext = ((b - 2 * enrob) if (f <= 1 and t >= n1)
                     else (abs(xs[t - 1] - xs[f - 1]) + d1 / 10.0 + 2 * dmm / 10.0))
            L_un_cm = 2 * (w_ext + (h - 2 * enrob)) + 2 * retour
        poids += n_par_m * (L_un_cm / 100.0) * _masse_lin_kg_m(dmm)

    # armatures de peau
    t_d = float(_g(values, "techno_d_mm", 10) or 10)
    t_smax = float(_g(values, "techno_s_max_cm", 30.0) or 30.0)
    d_vert = h - _dist_lit(values, bid, sid, "inf", 1) - _dist_lit(values, bid, sid, "sup", 1)
    if t_smax > 0 and d_vert > t_smax:
        n_side = max(0, int(math.ceil(d_vert / t_smax)) - 1)
        poids += 2 * n_side * 1.0 * _masse_lin_kg_m(t_d)

    poids_maj = poids * (1.0 + maj / 100.0)
    vol = (b / 100.0) * (h / 100.0)
    return poids_maj, vol


def _taux_armature_global(values, beam, bid):
    """TA global de la poutre = Σ(poids_i·L_i) / Σ(vol_i·L_i), arrondi au
    palier supérieur. None si option PDF désactivée ou aucune longueur."""
    if not bool(_g(values, "taux_arm_pdf", False)):
        return None
    arrondi = max(1, int(_g(values, "taux_arrondi_kgm3", 5) or 5))
    num = 0.0
    den = 0.0
    any_len = False
    for s in beam.get("sections", []):
        sid = int(s["id"])
        L = float(_g(values, KS("longueur_m", bid, sid), 0.0) or 0.0)
        if L <= 0:
            continue
        any_len = True
        poids, vol = _section_poids_vol(values, bid, sid)
        num += poids * L
        den += vol * L
    if not any_len or den <= 0:
        return None
    return math.ceil((num / den) / arrondi) * arrondi


# ============================================================
#  CALCUL SECTION (fidèle à poutre.py 2.32)
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
    gamma_s = _get_gamma_s(values)
    fyd = fyk / gamma_s

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

    # M_sup compte dès qu'il est > 0 (plus de case à cocher) ;
    # concept "effort tranchant réduit" supprimé.
    M_inf = float(_g(values, KS("M_inf", bid, sid), 0.0) or 0.0)
    M_sup = float(_g(values, KS("M_sup", bid, sid), 0.0) or 0.0)
    V = float(_g(values, KS("V", bid, sid), 0.0) or 0.0)
    has_Msup = M_sup > 0

    M_max = max(M_inf, M_sup)
    hmin = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10 if M_max > 0 else 0.0
    # Hauteur minimale de la poutre = hᵤ,min + CDG RÉEL des armatures
    # de la face du moment dimensionnant (v2.39 — avant : lit 1 inf.).
    e_cdg_gov = geo_sup["e_cdg"] if M_sup > M_inf else geo_inf["e_cdg"]
    h_min_poutre = hmin + e_cdg_gov
    etat_h = "ok" if (h_min_poutre <= h) else "nok"

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

    def build_shear(Vx):
        if Vx <= 0:
            return None
        tau = Vx * 1e3 / (0.75 * b * h * 100)
        besoin, etat_tau, nom_lim, tau_lim = shear_need(tau)
        Ast_e, summary, groups = _shear_lines(values, bid, sid)
        pas = float(_g(values, KS("shear_pas", bid, sid), 30.0) or 30.0)
        pas_th = Ast_e * fyd * (d_calc_shear * 10.0) / (Vx * 1e3) / 10.0 if Ast_e > 0 else 0.0
        s_max = min(0.75 * d_calc_shear, 30.0)
        pas_lim = min(pas_th, s_max) if pas_th > 0 else s_max
        etat_pas = "ok" if pas <= pas_lim else "nok"
        if not geom_shear_ok:
            etat_tau = "nok"; etat_pas = "nok"
        return {"tau": tau, "besoin": besoin, "etat_tau": etat_tau, "nom_lim": nom_lim,
                "tau_lim": tau_lim, "suf": "", "Ast": Ast_e, "summary": summary, "groups": groups,
                "pas": pas, "pas_th": pas_th, "s_max": s_max, "pas_lim": pas_lim,
                "etat_pas": etat_pas, "suf_pas": "", "V": Vx}

    shear = build_shear(V)

    # État global : inf. et sup. TOUJOURS pris en compte (comme dans l'app)
    states = [etat_h, etat_inf, etat_sup]
    if shear:
        states += [shear["etat_tau"], shear["etat_pas"]]
    etat_global = "nok" if any(s == "nok" for s in states) else ("warn" if any(s == "warn" for s in states) else "ok")

    return {
        "beton": beton, "fck": fck_cyl, "fck_cube": fck_cube, "alpha_b": alpha_b, "fctm": fctm,
        "fyk": fyk, "fyd": fyd, "gamma_s": gamma_s, "mu_ref": mu_ref, "mu": mu_val,
        "b": b, "h": h, "enrob_beton": enrob_beton,
        "ei": geo_inf["e_cdg"], "es": geo_sup["e_cdg"],
        "dist_l1_inf": dist_l1_inf, "dist_l1_sup": dist_l1_sup,
        "di": d_inf, "ds": d_sup, "dsh": d_shear,
        "M_inf": M_inf, "M_sup": M_sup, "V": V,
        "has_Msup": has_Msup,
        "M_max": M_max, "hmin": hmin, "etat_h": etat_h,
        "e_cdg_gov": e_cdg_gov, "h_min_poutre": h_min_poutre,
        "As_min_ec": As_min_ec, "As_min_plancher": As_min_plancher, "As_max": As_max,
        "As_req_inf": As_req_inf, "As_req_sup": As_req_sup,
        "As_min_inf": As_min_inf, "As_min_sup": As_min_sup,
        "geo_inf": geo_inf, "geo_sup": geo_sup,
        "As_inf": As_inf, "As_sup": As_sup, "etat_inf": etat_inf, "etat_sup": etat_sup,
        "shear": shear, "etat_global": etat_global,
        "techno": {
            "d": float(_g(values, "techno_d_mm", 10) or 10),
            "s_max": float(_g(values, "techno_s_max_cm", 30) or 30),
        },
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

        # ---- Offsets à l'échelle EXACTE ----
        # nu extérieur étrier = enrobage béton ; l'étrier est tracé sur sa
        # ligne moyenne (centre du trait) -> centre = enrobage + Ø_et/2.
        # axe barre principale = enrobage + Ø_et + Ø_barre/2.
        enrob_beton = float(R.get("enrob_beton", 3.0))
        st_off = enrob_beton * 10.0 * sc                       # enrobage (px)
        d_et_max = max((float(s.get("d", 8)) for s in self.stirrups), default=8.0)
        stw_main = max(0.8, d_et_max * sc)                      # Ø étrier (px)
        bar_off = st_off + stw_main                             # nu intérieur étrier

        def _xs(n, d_mm, off):
            r = max(1.2, (d_mm * sc) / 2.0)
            inset = off + r                                     # axe barre (exact)
            xa, xb = x0 + inset, x0 + sw - inset
            if n <= 1:
                return [(xa + xb) / 2.0], r
            return [xa + (xb - xa) * k / (n - 1) for k in range(n)], r

        # ---- Barres du lit 1 inférieur : référence des positions ----
        geo_inf = R["geo_inf"]; geo_sup = R["geo_sup"]
        lit1 = geo_inf["lits"][0]
        n1 = max(1, int(lit1["n"]))
        xs1, r1 = _xs(n1, lit1["d"], bar_off)
        y_lit1 = y0 + (lit1["e"] / h_cm) * sh

        def _clamp_bar(v, default):
            try:
                v = int(v)
            except Exception:
                return default
            return max(1, min(n1, v))

        # ---- Armatures technologiques (peau) — dessinées SOUS les étriers ----
        techno = R.get("techno") or {}
        t_d = float(techno.get("d", 10) or 10)
        t_smax = float(techno.get("s_max", 30) or 30)
        e_inf1 = geo_inf["lits"][0]["e"]
        e_sup1 = geo_sup["lits"][0]["e"]
        d_vert = h_cm - e_inf1 - e_sup1
        n_peau = 0
        if t_smax > 0 and d_vert > t_smax:
            n_int = int(math.ceil(d_vert / t_smax))
            n_peau = max(0, n_int - 1)
            if n_peau > 0:
                r_t = max(1.2, t_d * sc / 2.0)
                x_axis_l = x0 + st_off + stw_main + r_t         # axe, à l'intérieur des étriers
                x_axis_r = x0 + sw - st_off - stw_main - r_t
                step = d_vert / n_int
                c.setFillColor(PEAU_COLOR); c.setStrokeColor(PEAU_BORDER); c.setLineWidth(0.5)
                for k in range(1, n_peau + 1):
                    yy = y0 + ((e_inf1 + k * step) / h_cm) * sh
                    c.circle(x_axis_l, yy, r_t, stroke=1, fill=1)
                    c.circle(x_axis_r, yy, r_t, stroke=1, fill=1)

        # ---- Étriers / épingles positionnés ----
        # Tous les étriers partagent le MÊME niveau : traverses hautes
        # alignées, basses alignées (seule la largeur varie).
        off_c = st_off + stw_main / 2.0                          # ligne moyenne
        y_b = y0 + off_c
        y_t = y0 + sh - off_c

        k_et = 0  # index des étriers fermés (alternance des couleurs)
        for stg in self.stirrups:
            st_d = float(stg.get("d", 8))
            stw = max(0.8, st_d * sc)
            brins = int(stg.get("brins", 2))
            f = stg.get("from"); t = stg.get("to")

            if brins == 1:
                # ---- Épingle : décalée sur le côté de la barre ----
                c.setStrokeColor(EPINGLE_COLOR); c.setLineWidth(stw)
                fb = _clamp_bar(f, (n1 + 1) // 2)
                tb = _clamp_bar(t, fb)
                if fb > tb:
                    fb, tb = tb, fb
                if fb == tb:
                    dx = r1 + stw / 2.0 + 0.8                    # au ras de la barre, côté droit
                    xb_ = xs1[fb - 1] + dx
                    c.line(xb_, y_b, xb_, y_t)
                    hk = 5
                    c.line(xb_, y_t, xb_ - hk, y_t - hk)         # crochets vers la barre
                    c.line(xb_, y_b, xb_ - hk, y_b + hk)
                else:
                    # agrafe horizontale reliant les barres fb -> tb au lit 1
                    xa, xb2 = xs1[fb - 1], xs1[tb - 1]
                    c.line(xa, y_lit1, xb2, y_lit1)
                    hk = 6
                    c.line(xa, y_lit1, xa, y_lit1 + hk)
                    c.line(xb2, y_lit1, xb2, y_lit1 + hk)
                continue

            # ---- Étrier fermé : même niveau pour tous, largeur variable ----
            col = ETRIER_COLORS[k_et % len(ETRIER_COLORS)]
            stg["_color"] = col
            c.setStrokeColor(col); c.setLineWidth(stw)
            fb = _clamp_bar(f, 1)
            tb = _clamp_bar(t, n1)
            if fb > tb:
                fb, tb = tb, fb
            full = (fb <= 1 and tb >= n1)
            rr = max(2.5, 1.5 * st_d * sc)
            if full:
                x_l = x0 + off_c
                x_r = x0 + sw - off_c
            else:
                m = r1 + stw / 2.0                               # nu intérieur au contact de la barre
                x_l = xs1[fb - 1] - m
                x_r = xs1[tb - 1] + m
            c.roundRect(x_l, y_b, x_r - x_l, y_t - y_b, rr, stroke=1, fill=0)
            k_et += 1

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

        # centres de gravité des armatures (petites croix noires)
        # v2.39 : croix affichée UNIQUEMENT s'il y a plusieurs lits sur la
        # face — position réelle du CDG pondéré (Σ As·e / Σ As), c.-à-d.
        # ENTRE les lits (fix du bug "croix sur la barre du lit 1").
        def cdg_cross(y_cm_from_bottom):
            yy = y0 + (y_cm_from_bottom / h_cm) * sh
            cx = x0 + sw / 2.0
            r = 3.2
            c.setStrokeColor(INK); c.setLineWidth(0.9)
            c.line(cx - r, yy, cx + r, yy)
            c.line(cx, yy - r, cx, yy + r)
            return yy

        if geo_inf["nl"] > 1 and geo_inf["As"] > 0:
            cdg_cross(geo_inf["e_cdg"])                 # c.d.g. armatures inférieures
        if geo_sup["nl"] > 1 and geo_sup["As"] > 0:
            cdg_cross(h_cm - geo_sup["e_cdg"])          # c.d.g. armatures supérieures

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
            leg(yy, col, f"Lit {lit['i']} : {lit['n']} \u00d8{lit['d']}")

        # légende étriers/épingles : libellés simplifiés, sans positions
        Sh = R.get("shear") or {}
        pas_val = Sh.get("pas")

        def _stg_label(g):
            is_ep = int(g.get("brins", 2)) == 1
            if is_ep:
                return f"Épingle : \u00d8{int(g.get('d', 8))}"
            base = f"Étrier : \u00d8{int(g.get('d', 8))}"
            if pas_val:
                p = float(pas_val)
                base += f" — {p:.0f} cm" if abs(p - round(p)) < 0.05 else (f" — {p:.1f} cm").replace(".", ",")
            return base

        leg_items = [(_stg_label(g),
                      g.get("_color", EPINGLE_COLOR if int(g.get("brins", 2)) == 1 else ETRIER_COLORS[0]))
                     for g in self.stirrups]
        if n_peau > 0:
            leg_items.append((f"Armature de peau : 2\u00d7{n_peau} \u00d8{int(t_d)}", PEAU_COLOR))

        ymid = y0 + sh / 2.0
        y_start = ymid + (len(leg_items) - 1) * 4.5
        c.setFont("Helvetica", 7.4)
        for j, (lab, colr) in enumerate(leg_items):
            yy = y_start - j * 9.0
            c.setFillColor(colr); c.circle(lx + 2, yy, 2.2, stroke=0, fill=1)
            c.setFillColor(P["txt"])
            c.drawString(lx + 8, yy - 2.6, lab)
        for yy, col, lit in reversed(y_inf):
            leg(yy, col, f"Lit {lit['i']} : {lit['n']} \u00d8{lit['d']}")

        c.restoreState()


def stirrups_for(R, values, bid, sid):
    """Toutes les lignes étriers/épingles avec leur position (dessin)."""
    Sh = R.get("shear")
    groups = (Sh or {}).get("groups")
    if not groups:
        fs = _first_stirrup(values, bid, sid)
        groups = [{"type": fs["type"], "d": fs["d"], "brins": fs["brins"], "from": None, "to": None}]
    return groups


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
            kv("Coefficient acier ELS", f"{fn(R['gamma_s'],2)}"),
            kv("Contrainte de calcul acier", f"f<sub>yd</sub> = {fn(R['fyd'],0)} N/mm{s2()}"),
            sub("SOLLICITATIONS"),
            kv("M<sub>inf</sub>", f"{fn(R['M_inf'],1)} kNm")]
    if R["has_Msup"]:
        rows.append(kv("M<sub>sup</sub>", f"{fn(R['M_sup'],1)} kNm"))
    rows.append(kv("V", f"{fn(R['V'],1)} kN"))
    t = Table(rows, colWidths=[cw * 0.55, cw * 0.45])
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
    app = Formula(Row([_t("h", sub="u,min"), _t(" = "),
        Sqrt(Row([Frac(Row(Row(sci_tokens(R['M_max'] * 1e6)).items),
                       Row([_t(f"{fn(R['alpha_b'],2)} · {fn(R['b']*10,0)} · {fn(R['mu'],4)}")]))]), INK),
        _t("  =  "), nb(f"{fn(R['hmin'],1)} cm")]))
    # Hauteur minimale de la poutre = hᵤ,min + CDG réel des armatures
    # (face du moment dimensionnant), valeurs numériques explicites.
    hminp = Formula(Row([
        _t("h", sub="u,min"), _t(" + CDG armatures = "),
        _t(f"{fn(R['hmin'],1)} + {fn(R['e_cdg_gov'],1)}  =  "),
        nb(f"{fn(R['h_min_poutre'],1)} cm")]))
    body = [fline("Hauteur utile minimale", app, iw),
            Spacer(1, 2),
            fline("Hauteur minimale de la poutre", hminp, iw),
            Spacer(1, 7), HR(iw, HAIR, 0.5), Spacer(1, 7),
            reslines([("Hauteur minimale de la poutre", "h<sub>min</sub>", f"{fn(R['h_min_poutre'],1)} cm"),
                      ("Hauteur de la poutre", "h", f"{fn(R['h'],0)} cm")], iw),
            Spacer(1, 5)]
    ok = R["etat_h"] == "ok"
    left = (f"Hauteur de la poutre : {fn(R['h'],0)} cm "
            f"{'≥' if ok else '&lt;'} hauteur minimale de la poutre : {fn(R['h_min_poutre'],1)} cm")
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

    # 3 lignes de critères (unités uniquement sur le résultat final)
    line1 = Row([_tt("0,26 · "),
                 Frac(Row([_tt(f"{fn(fctm,1)}")]), Row([_tt(f"{int(fyk)}")]), pad=2),
                 _tt(f" · {fn(b_mm,0)} · {fn(h_mm,0)} = {fn(ec,0)}")])
    line2 = Row([_tt(f"0,0013 · {fn(b_mm,0)} · {fn(h_mm,0)} = {fn(pl,0)}")])
    line3 = Row([_tt("0,25 · A", sub="s,req"), _tt(f",{face_opp} = 0,25 · {fn(as_req_opp,0)} = {fn(quart,0)}")])

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
    # hauteur utile : toujours à partir du yG effectif (imposé compris)
    suffix_note = f"   (c.d.g. de {nl} lits)" if nl > 1 else ""
    dlit = Formula(Row([_t("d", sub="u"), _t(f" = {fn(R['h'],0)} − {fn(geo['e_cdg'],1)} = "), nb(f"{fn(d,1)} cm"),
                        _t(suffix_note, color=MUTE, size=8.5)]))

    app = Formula(Row([_t("A", sub="s,req"), _t(" = "),
        Frac(Row(Row(sci_tokens(M * 1e6)).items), Row([_t(f"{fn(R['fyd'],1)} · 0,9 · {fn(d*10,0)}")])),
        _t("  =  "), txt(f"{fn(Ar,0)} mm", font="Helvetica-Bold", sup="2")]))

    choix = f"{geo['detail']} ({fn(geo['As'],0)} mm{s2()})" + (f" · {nl} lits" if nl > 1 else "")

    asmin_f = _asmin_formula(R, which)
    asmax_f = Formula(Row([
        _t("A", sub="s,max"), _t(f" = 0,04 · {fn(R['b']*10,0)} · {fn(R['h']*10,0)} = "),
        txt(f"{fn(R['As_max'],0)} mm", font="Helvetica-Bold", sup="2")]))

    moment_label = "Moment inférieur" if which == "inf" else "Moment supérieur"
    body = [fline(moment_label,
                  Formula(Row([_t("M", sub=("inf" if which == "inf" else "sup")),
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
    face_txt = "inférieure" if which == "inf" else "supérieure"
    left = (f"Section d'armature {face_txt} : {fn(geo['As'],0)} mm{s2()} "
            f"{'≥' if ok else '&lt;'} section d'armature requise : {fn(besoin,0)} mm{s2()}")
    body.append(conclu(et, iw, left, ok=ok))
    return block(nn, title, et, body, cw)


def b_shear(R, cw):
    iw = cw - 24
    Sh = R["shear"]
    nn = "4."
    app = Formula(Row([_t("τ = "),
        Frac(Row(Row(sci_tokens(Sh['V'] * 1e3)).items), Row([_t(f"0,75 · {fn(R['b']*10,0)} · {fn(R['h']*10,0)}")])),
        _t("  =  "), txt(f"{fn(Sh['tau'],2)} N/mm", font="Helvetica-Bold", sup="2")]))
    if Sh['Ast'] > 0 and Sh['V'] > 0:
        sthapp = Formula(Row([_t("s", sub="th"), _t(" = "),
            Frac(Row([_t(f"{fn(Sh['Ast'],1)} · {fn(R['fyd'],1)} · {fn(R['dsh']*10,0)}")]), Row(Row(sci_tokens(Sh['V'] * 1e3)).items)),
            _t("  =  "), nb(f"{fn(Sh['pas_th'],1)} cm")]))
    else:
        sthapp = Formula(Row([_t("s", sub="th"), _t("  =  "), nb("—")]))
    etr = f"{Sh['summary']}"
    okt = Sh["tau"] <= Sh["tau_lim"]
    okp = Sh["pas"] <= Sh["pas_lim"]
    et_tau = "ok" if okt else ("warn" if Sh["etat_tau"] == "warn" else "nok")
    # Pas admissible = min( pas théorique ; pas maximal ) — calcul explicite
    sadm = Formula(Row([
        _t("s", sub="adm"),
        _t(f" = min( {fn(Sh['pas_th'],1)} ; {fn(Sh['s_max'],1)} )  =  "),
        nb(f"{fn(Sh['pas_lim'],1)} cm")]))
    body = [fline("Contrainte tangentielle", app, iw),
            Spacer(1, 7), HR(iw, HAIR, 0.5), Spacer(1, 7),
            reslines([("Contrainte admissible", "τ<sub>adm</sub>", f"{fn(Sh['tau_lim'],2)} N/mm{s2()}")], iw),
            Spacer(1, 4),
            conclu(et_tau, iw,
                   f"Contrainte tangentielle : {fn(Sh['tau'],2)} N/mm{s2()} "
                   f"{'≤' if okt else '&gt;'} contrainte tangentielle admissible : {fn(Sh['tau_lim'],2)} N/mm{s2()}",
                   ok=okt),
            Spacer(1, 9), Paragraph("<b>Étriers</b>", ST["f"]), Spacer(1, 4),
            reslines([("On prend", "", etr),
                      ("Section", "A<sub>sw</sub>", f"{fn(Sh['Ast'],1)} mm{s2()}")], iw),
            Spacer(1, 2), fline("Pas théorique", sthapp, iw),
            Spacer(1, 2), fline("Pas maximal",
                Formula(Row([_t("s", sub="max"), _t(" = min(0,75 · d ; 30) = "), nb(f"{fn(Sh['s_max'],1)} cm")])), iw),
            Spacer(1, 2), fline("Pas admissible", sadm, iw),
            Spacer(1, 2), fline("Pas retenu", Formula(Row([_t("s"), _t("  =  "), nb(f"{fn(Sh['pas'],1)} cm")])), iw),
            Spacer(1, 5)]
    et = "nok" if "nok" in (Sh["etat_tau"], Sh["etat_pas"]) else ("warn" if "warn" in (Sh["etat_tau"], Sh["etat_pas"]) else "ok")
    et_pas = "ok" if okp else "nok"
    left = (f"Pas des armatures d'effort tranchant : {fn(Sh['pas'],1)} cm "
            f"{'≤' if okp else '&gt;'} pas maximal : {fn(Sh['pas_lim'],1)} cm")
    body.append(conclu(et_pas, iw, left, ok=okp))
    return block(nn, "Effort tranchant — étriers", et, body, cw)


# ============================================================
#  BANDEAUX POUTRE / SECTION (pastel)
# ============================================================
def beam_banner(txt_, cw, right_txt=None):
    """Bandeau de poutre. right_txt (optionnel) : texte aligné à droite,
    en gras — utilisé pour 'T.A. = xxx kg/m³' (option PDF activée)."""
    st = ParagraphStyle("bb", parent=ST["beam"], textColor=BEAM_TX)
    if right_txt:
        str_ = ParagraphStyle("bbr", parent=ST["beam"], textColor=BEAM_TX,
                              fontSize=11.5, leading=18, alignment=TA_RIGHT)
        t = Table([[Paragraph(txt_, st), Paragraph(f"<b>{right_txt}</b>", str_)]],
                  colWidths=[cw * 0.62, cw * 0.38])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BEAM_BG),
            ("LEFTPADDING", (0, 0), (0, 0), 12), ("RIGHTPADDING", (0, 0), (0, 0), 4),
            ("LEFTPADDING", (1, 0), (1, 0), 4), ("RIGHTPADDING", (1, 0), (1, 0), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROUNDEDCORNERS", [5, 5, 5, 5])]))
        return t
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
        c.drawString(18 * mm, h - 12 * mm, "Bureau d'Études Valens")
        c.setFillColor(MUTE); c.setFont("Helvetica", 8)
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
        # T.A. global de la poutre à droite du bandeau — uniquement si
        # l'option "Envoyer dans la note de calcul" est activée.
        ta_global = _taux_armature_global(values, b, bid)
        ta_txt = f"T.A. = {ta_global:.0f} kg/m³" if ta_global is not None else None
        story.append(beam_banner(str(_g(values, f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}"))), cw,
                                 right_txt=ta_txt))
        story.append(Spacer(1, 10))
        sections = b.get("sections", [])
        for si, s in enumerate(sections):
            sid = int(s["id"])
            raw = str(_g(values, f"meta_b{bid}_nom_{sid}", s.get("nom", f"Section {sid}")))
            snom = raw if raw.strip().lower().startswith("section") else f"Section {raw}"
            R = _compute_section(values, beton_data, bid, sid)
            # Comme dans l'application : blocs inf. ET sup. toujours présents
            blocs = [b_haut(R, cw), b_arm(R, cw, "inf"), b_arm(R, cw, "sup")]
            if R["shear"]:
                blocs.append(b_shear(R, cw))
            intro = [sec_banner(snom, cw), Spacer(1, 6), recap(R, values, bid, sid, cw), Spacer(1, 12), blocs[0]]
            story.append(KeepTogether(intro))
            for blk in blocs[1:]:
                story.append(Spacer(1, 12)); story.append(KeepTogether([blk]))
            if si < len(sections) - 1:
                story.append(Spacer(1, 16))
    return story


# ============================================================
#  API PRINCIPALE — mise en page ndc_pdf (v3.0)
# ============================================================
# Avertissements de débordement de la dernière génération (points hors
# page par planche). Liste vide = rien ne déborde.
DERNIERS_AVERTISSEMENTS = []


def _style_ndc(n_cols=2):
    """Palette 01_encre de la maquette ; n_cols=3 est le remède prescrit
    quand une planche déborde (jamais la réduction du corps de texte)."""
    from ndc_pdf.styles import Encre
    s = Encre()
    if n_cols != 2:
        s.n_cols = n_cols
    return s


def _peau_bars(R):
    """Armatures de peau (paramètres techno) : mêmes règles que le dessin
    de l'application et _section_poids_vol — n = ceil(d_vert/s_max) − 1
    par face latérale, réparties entre les axes des lits 1. Renvoie
    None ou {d (mm), ys (cm depuis le bas), n}."""
    techno = R.get("techno") or {}
    t_d = float(techno.get("d", 10) or 10)
    t_smax = float(techno.get("s_max", 30) or 30)
    e_inf1 = R["geo_inf"]["lits"][0]["e"]
    e_sup1 = R["geo_sup"]["lits"][0]["e"]
    d_vert = R["h"] - e_inf1 - e_sup1
    if t_smax <= 0 or d_vert <= t_smax:
        return None
    n_int = int(math.ceil(d_vert / t_smax))
    n_peau = max(0, n_int - 1)
    if n_peau == 0:
        return None
    step = d_vert / n_int
    return {"d": t_d, "n": n_peau,
            "ys": [e_inf1 + k * step for k in range(1, n_peau + 1)]}


def _collecter_resultats(beams, values, beton_data):
    """Payloads NEUTRES pour ndc_pdf.data : un par section, dans l'ordre
    des planches. Toute la vérité vient de _compute_section (fidèle à
    poutre.py) — la mise en page transcrit, elle ne recalcule rien."""
    out = []
    for b in beams:
        bid = int(b["id"])
        nom_poutre = str(_g(values, f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}")))
        ta = _taux_armature_global(values, b, bid)
        for sec in b.get("sections", []):
            sid = int(sec["id"])
            raw = str(_g(values, f"meta_b{bid}_nom_{sid}", sec.get("nom", f"Section {sid}")))
            snom = raw if raw.strip().lower().startswith("section") else f"Section {raw}"
            R = _compute_section(values, beton_data, bid, sid)
            out.append(dict(
                poutre=nom_poutre, section=snom, R=R,
                stirrups=stirrups_for(R, values, bid, sid),
                peau=_peau_bars(R),
                ta_global=ta,
            ))
    return out


def generer_rapport_pdf(beams, values, beton_data, infos=None, output_path=None):
    """Note de calcul complète : garde portrait + une planche paysage par
    section. Signature et retour inchangés (appelée par poutre.py)."""
    global DERNIERS_AVERTISSEMENTS
    from ndc_pdf import data as ndc_data

    infos = infos or {}
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".pdf", prefix="note_poutre_")
        os.close(fd)

    resultats = _collecter_resultats(beams, values, beton_data)
    doc_meta = ndc_data.construire_doc(
        infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    sections = ndc_data.construire_sections(resultats)

    d = _style_ndc(2).build(output_path, sections=sections, doc=doc_meta)
    if d.warnings:
        d = _style_ndc(3).build(output_path, sections=sections, doc=doc_meta)
    d.save()
    DERNIERS_AVERTISSEMENTS = list(d.warnings)
    return output_path

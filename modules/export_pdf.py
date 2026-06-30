# -*- coding: utf-8 -*-
# ============================================================
#  export_pdf.py
#  Génération de la note de calcul (PDF) — Poutre béton armé
#  Normes "pré-Eurocode" (méthode interne du bureau)
#
#  API principale :
#     generer_rapport_pdf(beams, values, beton_data, infos=None) -> str (chemin PDF)
#
#  - beams      : st.session_state.beams  (liste des poutres + sections)
#  - values     : dict des valeurs (st.session_state, ou copie filtrée)
#  - beton_data : contenu de beton_classes.json
#  - infos      : dict optionnel {nom_projet, partie, date, indice}
#
#  Conçu pour être appelé depuis poutre.py (module multi-poutres / multi-sections).
#  Chaque "paragraphe" (bloc de vérification) tient sur une seule page
#  grâce à KeepTogether + sauts de page entre sections.
# ============================================================

import math
import os
import tempfile
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame,
    Paragraph, Spacer, Table, TableStyle,
    KeepTogether, PageBreak, Flowable,
)


# ============================================================
#  PALETTE / CONSTANTES VISUELLES
# ============================================================
COL_PRIMARY   = colors.HexColor("#1f3a5f")   # bleu profond (titres)
COL_ACCENT    = colors.HexColor("#2f6db5")   # bleu accent
COL_TEXT      = colors.HexColor("#1a1a1a")
COL_MUTED     = colors.HexColor("#6b7280")
COL_LINE      = colors.HexColor("#d9dee5")
COL_BG_SOFT   = colors.HexColor("#f4f6f9")

# Bandeau de poutre : gris-bleu doux, neutre, s'accorde au vert et au rouge des cadres
COL_BANNER_BG = colors.HexColor("#5b6b7d")   # gris ardoise atténué (poutre)
COL_BANNER_TX = colors.HexColor("#ffffff")

# Bandeau de section : encore plus atténué que celui de la poutre
COL_SBANNER_BG = colors.HexColor("#9aa6b2")  # gris clair (section)
COL_SBANNER_TX = colors.HexColor("#ffffff")

COL_OK_BG     = colors.HexColor("#e6f6ec")
COL_OK_BD     = colors.HexColor("#28a745")
COL_OK_TX     = colors.HexColor("#1b6b32")

COL_WARN_BG   = colors.HexColor("#fff7e0")
COL_WARN_BD   = colors.HexColor("#e0a800")
COL_WARN_TX   = colors.HexColor("#8a6400")

COL_NOK_BG    = colors.HexColor("#fdeaea")
COL_NOK_BD    = colors.HexColor("#dc3545")
COL_NOK_TX    = colors.HexColor("#a31621")

ETAT_VIS = {
    "ok":   {"bg": COL_OK_BG,   "bd": COL_OK_BD,   "tx": COL_OK_TX,   "ico": "OK",  "label": "Vérifié"},
    "warn": {"bg": COL_WARN_BG, "bd": COL_WARN_BD, "tx": COL_WARN_TX, "ico": "!",   "label": "À surveiller"},
    "nok":  {"bg": COL_NOK_BG,  "bd": COL_NOK_BD,  "tx": COL_NOK_TX,  "ico": "X",   "label": "Non vérifié"},
}


# ============================================================
#  STYLES
# ============================================================
def _build_styles():
    ss = getSampleStyleSheet()

    styles = {}

    styles["doc_title"] = ParagraphStyle(
        "doc_title", parent=ss["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=COL_PRIMARY, spaceAfter=2, alignment=TA_LEFT,
    )
    styles["doc_sub"] = ParagraphStyle(
        "doc_sub", parent=ss["Normal"],
        fontName="Helvetica", fontSize=10.5, leading=14,
        textColor=COL_MUTED, alignment=TA_LEFT,
    )
    styles["beam_title"] = ParagraphStyle(
        "beam_title", parent=ss["Heading1"],
        fontName="Helvetica-Bold", fontSize=16, leading=20,
        textColor=colors.white, alignment=TA_LEFT,
        spaceBefore=0, spaceAfter=0,
    )
    styles["sec_title"] = ParagraphStyle(
        "sec_title", parent=ss["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=COL_PRIMARY, alignment=TA_LEFT,
        spaceBefore=2, spaceAfter=2,
    )
    styles["block_title"] = ParagraphStyle(
        "block_title", parent=ss["Heading3"],
        fontName="Helvetica-Bold", fontSize=11.5, leading=14,
        textColor=COL_TEXT, alignment=TA_LEFT,
        spaceBefore=0, spaceAfter=0,
    )
    styles["body"] = ParagraphStyle(
        "body", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9.6, leading=13.5,
        textColor=COL_TEXT, alignment=TA_LEFT,
    )
    styles["body_muted"] = ParagraphStyle(
        "body_muted", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9, leading=12.5,
        textColor=COL_MUTED, alignment=TA_LEFT,
    )
    styles["formula"] = ParagraphStyle(
        "formula", parent=ss["Normal"],
        fontName="Helvetica", fontSize=10, leading=16,
        textColor=COL_TEXT, alignment=TA_LEFT,
        leftIndent=8,
    )
    styles["formula_label"] = ParagraphStyle(
        "formula_label", parent=ss["Normal"],
        fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
        textColor=COL_MUTED, alignment=TA_LEFT,
    )
    styles["cell"] = ParagraphStyle(
        "cell", parent=ss["Normal"],
        fontName="Helvetica", fontSize=9, leading=12,
        textColor=COL_TEXT, alignment=TA_LEFT,
    )
    styles["cell_b"] = ParagraphStyle(
        "cell_b", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=12,
        textColor=COL_TEXT, alignment=TA_LEFT,
    )
    styles["cell_head"] = ParagraphStyle(
        "cell_head", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=12,
        textColor=colors.white, alignment=TA_LEFT,
    )
    styles["badge"] = ParagraphStyle(
        "badge", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=9.5, leading=12,
        alignment=TA_RIGHT,
    )
    styles["concl"] = ParagraphStyle(
        "concl", parent=ss["Normal"],
        fontName="Helvetica-Bold", fontSize=10, leading=14,
        alignment=TA_LEFT,
    )
    return styles


# ============================================================
#  HELPERS DE MISE EN FORME "LaTeX-like"
#  -> on utilise les balises markup de reportlab (<sub>, <super>, <i>, <b>)
# ============================================================
def f_num(x, nd=2):
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except Exception:
        return str(x)


def _sub(s):
    return f"<sub>{s}</sub>"


def _sup(s):
    return f"<super>{s}</super>"


# symboles couramment utilisés (markup reportlab)
SYM = {
    "leq": "&#8804;",     # ≤
    "geq": "&#8805;",     # ≥
    "times": "&#215;",    # ×
    "approx": "&#8776;",  # ≈
    "rightarrow": "&#8594;",  # →
    "sqrt": "&#8730;",    # √
    "alpha": "&#945;",
    "tau": "&#964;",
    "phi": "&#216;",      # Ø (diamètre)
    "deg": "&#176;",
}


def frac(num, den):
    """Petite fraction inline lisible : num / den (avec barre)."""
    return f"{num} / {den}"


# ============================================================
#  FLOWABLE : marqueur de page (capture le n° de page de début de poutre)
# ============================================================
class PageMarker(Flowable):
    def __init__(self, store: dict, key):
        super().__init__()
        self.store = store
        self.key = key
        self.width = 0
        self.height = 0

    def wrap(self, availWidth, availHeight):
        return (0, 0)

    def draw(self):
        # self.canv.getPageNumber() renvoie la page sur laquelle ce flowable est posé
        self.store[self.key] = self.canv.getPageNumber()


# ============================================================
#  BANDEAU (poutre / section) — construit comme les cadres pour
#  garantir un bord gauche identique aux blocs verts/rouges.
# ============================================================
def make_banner(text, cw, kind="beam", etat="ok"):
    """kind='beam' (foncé) ou 'section' (plus atténué)."""
    if kind == "section":
        bg = COL_SBANNER_BG
        tx = COL_SBANNER_TX
        fs = 12
    else:
        bg = COL_BANNER_BG
        tx = COL_BANNER_TX
        fs = 13

    style = ParagraphStyle(
        "banner_txt", fontName="Helvetica-Bold", fontSize=fs,
        leading=fs + 3, textColor=tx,
    )
    cell = Paragraph(text, style)
    t = Table([[cell]], colWidths=[cw])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    return t


# ============================================================
#  FLOWABLE : barre de titre poutre (conservé pour compat, non utilisé)
# ============================================================
class BeamBanner(Flowable):
    def __init__(self, text, width, etat="ok", height=24):
        super().__init__()
        self.text = text
        self.width = width
        self.height = height
        self.etat = etat

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(COL_BANNER_BG)
        c.roundRect(0, 0, self.width, self.height, 4, stroke=0, fill=1)
        c.setFillColor(COL_BANNER_TX)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(12, self.height/2 - 4.5, self.text)
        c.restoreState()


# ============================================================
#  FLOWABLE : ligne horizontale fine
# ============================================================
class HLine(Flowable):
    def __init__(self, width, color=COL_LINE, thickness=0.6):
        super().__init__()
        self.width = width
        self.color = color
        self.thickness = thickness

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        return (self.width, self.thickness + 2)

    def draw(self):
        c = self.canv
        c.setStrokeColor(self.color)
        c.setLineWidth(self.thickness)
        c.line(0, 1, self.width, 1)


# ============================================================
#  CALCULS (réplique fidèle de poutre.py)
# ============================================================
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


def _g(values, key, default=None):
    return values.get(key, default)


def KB(base, bid):
    return f"b{bid}_{base}"


def KS(base, bid, sid):
    return f"b{bid}_sec{sid}_{base}"


def _as_layer(values, bid, sid, which):
    if which == "inf":
        n1 = int(_g(values, KS("n_as_inf", bid, sid), 2) or 2)
        d1 = int(_g(values, KS("ø_as_inf", bid, sid), 16) or 16)
        has2 = bool(_g(values, KS("ajouter_second_lit_inf", bid, sid), False))
        n2 = int(_g(values, KS("n_as_inf_2", bid, sid), 2) or 2)
        d2 = int(_g(values, KS("ø_as_inf_2", bid, sid), d1) or d1)
        jeu = float(_g(values, KS("jeu_inf_2", bid, sid), 0.0) or 0.0)
    else:
        n1 = int(_g(values, KS("n_as_sup", bid, sid), 2) or 2)
        d1 = int(_g(values, KS("ø_as_sup", bid, sid), 16) or 16)
        has2 = bool(_g(values, KS("ajouter_second_lit_sup", bid, sid), False))
        n2 = int(_g(values, KS("n_as_sup_2", bid, sid), 2) or 2)
        d2 = int(_g(values, KS("ø_as_sup_2", bid, sid), d1) or d1)
        jeu = float(_g(values, KS("jeu_sup_2", bid, sid), 0.0) or 0.0)

    As1 = n1 * _bar_area_mm2(d1)
    if has2:
        As2 = n2 * _bar_area_mm2(d2)
        return {
            "As": As1 + As2, "n1": n1, "d1": d1, "has2": True,
            "n2": n2, "d2": d2, "jeu": jeu, "As1": As1, "As2": As2,
            "detail": f"{n1}{SYM['phi']}{d1} + {n2}{SYM['phi']}{d2}",
        }
    return {
        "As": As1, "n1": n1, "d1": d1, "has2": False,
        "n2": 0, "d2": 0, "jeu": 0.0, "As1": As1, "As2": 0.0,
        "detail": f"{n1}{SYM['phi']}{d1}",
    }


def _auto_enrob_calc(values, bid, sid, which):
    enrob_beton = float(_g(values, KB("enrobage_beton", bid), 3.0) or 3.0)
    jeu_enrob = float(_g(values, "jeu_enrobage_cm", 1.0) or 1.0)
    diam = float(_g(values, KS("ø_as_inf" if which == "inf" else "ø_as_sup", bid, sid), 16) or 16)
    demi_diam_cm = diam / 20.0
    return enrob_beton + jeu_enrob + _round_up_to_half_cm(demi_diam_cm)


def _enrob_calc(values, bid, sid, which):
    key_val = KS(f"enrob_calc_{which}", bid, sid)
    key_ovr = KS(f"enrob_calc_{which}_override", bid, sid)
    auto_val = _auto_enrob_calc(values, bid, sid, which)
    if bool(_g(values, key_ovr, False)):
        try:
            return float(_g(values, key_val, auto_val) or auto_val)
        except Exception:
            return float(auto_val)
    return float(auto_val)


def _shear_lines(values, bid, sid, reduced):
    if reduced:
        n_lines = int(_g(values, KS("shear_n_lines_r", bid, sid), 1) or 1)
        prefix = "shear_r_line"
    else:
        n_lines = int(_g(values, KS("shear_n_lines", bid, sid), 1) or 1)
        prefix = "shear_line"
    n_lines = max(1, n_lines)
    Ast = 0.0
    parts = []
    for i in range(n_lines):
        typ = str(_g(values, KS(f"{prefix}{i}_type", bid, sid), "Étriers (2 brins)"))
        n_c = int(_g(values, KS(f"{prefix}{i}_n", bid, sid), 1) or 1)
        diam = float(_g(values, KS(f"{prefix}{i}_d", bid, sid), 8) or 8)
        brins = _brins_from_type(typ)
        Ast += n_c * brins * _bar_area_mm2(diam)
        parts.append(f"{n_c}{SYM['times']} {typ} {SYM['phi']}{int(diam)}")
    return Ast, " + ".join(parts)


def _get_fyk(values, bid):
    cur = str(_g(values, KB("fyk", bid), "500"))
    if cur not in ("400", "500"):
        cur = "500"
    return float(cur), cur


def _compute_section(values, beton_data, bid, sid):
    """Recalcule tout pour une section : retourne un dict riche pour le rendu."""
    beton = str(_g(values, KB("beton", bid), "C30/37"))
    if beton not in beton_data:
        beton = list(beton_data.keys())[0]
    bd = beton_data[beton]
    fck = bd.get("fck", 0)
    fck_cube = bd["fck_cube"]
    alpha_b = bd["alpha_b"]

    fyk, mu_ref = _get_fyk(values, bid)
    fyd = fyk / 1.5

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in bd:
        mu_key = "mu_a500" if "mu_a500" in bd else [k for k in bd if k.startswith("mu_a")][0]
    mu_val = bd[mu_key]

    b = float(_g(values, KB("b", bid), 20))
    h = float(_g(values, KB("h", bid), 40))

    enrob_inf = _enrob_calc(values, bid, sid, "inf")
    enrob_sup = _enrob_calc(values, bid, sid, "sup")
    d_inf = h - enrob_inf
    d_sup = h - enrob_sup
    d_shear = h - min(enrob_inf, enrob_sup)

    tol = float(_g(values, "tau_tolerance_percent", 0.0) or 0.0)

    M_inf = float(_g(values, KS("M_inf", bid, sid), 0.0) or 0.0)
    M_sup = float(_g(values, KS("M_sup", bid, sid), 0.0) or 0.0)
    V = float(_g(values, KS("V", bid, sid), 0.0) or 0.0)
    V_lim = float(_g(values, KS("V_lim", bid, sid), 0.0) or 0.0)
    has_Msup = bool(_g(values, KS("ajouter_moment_sup", bid, sid), False)) and (M_sup > 0)
    has_Vlim = bool(_g(values, KS("ajouter_effort_reduit", bid, sid), False)) and (V_lim > 0)

    # Hauteur
    M_max = max(M_inf, M_sup)
    hmin = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10 if M_max > 0 else 0.0
    etat_h = "ok" if (hmin + enrob_inf <= h) else "nok"

    # Aciers
    As_min = 0.0013 * b * h * 1e2
    As_max = 0.04 * b * h * 1e2
    As_req_inf = (M_inf * 1e6) / (fyd * 0.9 * d_inf * 10) if M_inf > 0 else 0.0
    As_req_sup = (M_sup * 1e6) / (fyd * 0.9 * d_sup * 10) if M_sup > 0 else 0.0
    As_min_inf = max(As_min, 0.25 * As_req_sup)
    As_min_sup = max(As_min, 0.25 * As_req_inf)

    inf = _as_layer(values, bid, sid, "inf")
    sup = _as_layer(values, bid, sid, "sup")

    etat_inf = "ok" if (inf["As"] >= As_req_inf and inf["As"] <= As_max) else "nok"
    etat_sup = "ok" if (sup["As"] >= As_req_sup and sup["As"] <= As_max) else "nok"

    # Tranchant
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

    shear = None
    if V > 0:
        tau = V * 1e3 / (0.75 * b * h * 100)
        besoin, etat_tau_base, nom_lim, tau_lim = shear_need(tau)
        if tau > tau_lim:
            etat_tau, suf = status_tol(tau, tau_lim)
        else:
            etat_tau, suf = etat_tau_base, ""
        Ast_e, summary = _shear_lines(values, bid, sid, reduced=False)
        pas = float(_g(values, KS("shear_pas", bid, sid), 30.0) or 30.0)
        pas_th = Ast_e * fyd * d_shear * 10 / (10 * V * 1e3)
        s_max = min(0.75 * d_shear, 30.0)
        pas_lim = min(pas_th, s_max)
        etat_pas, suf_pas = status_tol(pas, pas_lim)
        shear = {
            "tau": tau, "besoin": besoin, "etat_tau": etat_tau, "nom_lim": nom_lim,
            "tau_lim": tau_lim, "suf": suf, "Ast": Ast_e, "summary": summary,
            "pas": pas, "pas_th": pas_th, "s_max": s_max, "pas_lim": pas_lim,
            "etat_pas": etat_pas, "suf_pas": suf_pas, "V": V,
        }

    shear_r = None
    if has_Vlim:
        tau_r = V_lim * 1e3 / (0.75 * b * h * 100)
        besoin_r, etat_r_base, nom_lim_r, tau_lim_r = shear_need(tau_r)
        if tau_r > tau_lim_r:
            etat_r, suf_r = status_tol(tau_r, tau_lim_r)
        else:
            etat_r, suf_r = etat_r_base, ""
        Ast_er, summary_r = _shear_lines(values, bid, sid, reduced=True)
        pas_r = float(_g(values, KS("shear_pas_r", bid, sid), 30.0) or 30.0)
        pas_th_r = Ast_er * fyd * d_shear * 10 / (10 * V_lim * 1e3)
        s_max_r = min(0.75 * d_shear, 30.0)
        pas_lim_r = min(pas_th_r, s_max_r)
        etat_pas_r, suf_pas_r = status_tol(pas_r, pas_lim_r)
        shear_r = {
            "tau": tau_r, "besoin": besoin_r, "etat_tau": etat_r, "nom_lim": nom_lim_r,
            "tau_lim": tau_lim_r, "suf": suf_r, "Ast": Ast_er, "summary": summary_r,
            "pas": pas_r, "pas_th": pas_th_r, "s_max": s_max_r, "pas_lim": pas_lim_r,
            "etat_pas": etat_pas_r, "suf_pas": suf_pas_r, "V": V_lim,
        }

    # état global
    states = [etat_h]
    if M_inf > 0:
        states.append(etat_inf)
    if has_Msup:
        states.append(etat_sup)
    if shear:
        states += [shear["etat_tau"], shear["etat_pas"]]
    if shear_r:
        states += [shear_r["etat_tau"], shear_r["etat_pas"]]
    if any(s == "nok" for s in states):
        etat_global = "nok"
    elif any(s == "warn" for s in states):
        etat_global = "warn"
    else:
        etat_global = "ok"

    return {
        "beton": beton, "fck": fck, "fck_cube": fck_cube, "alpha_b": alpha_b,
        "fyk": fyk, "fyd": fyd, "mu_ref": mu_ref, "mu_val": mu_val,
        "b": b, "h": h, "enrob_inf": enrob_inf, "enrob_sup": enrob_sup,
        "d_inf": d_inf, "d_sup": d_sup, "d_shear": d_shear,
        "M_inf": M_inf, "M_sup": M_sup, "V": V, "V_lim": V_lim,
        "has_Msup": has_Msup, "has_Vlim": has_Vlim,
        "M_max": M_max, "hmin": hmin, "etat_h": etat_h,
        "As_min": As_min, "As_max": As_max,
        "As_req_inf": As_req_inf, "As_req_sup": As_req_sup,
        "As_min_inf": As_min_inf, "As_min_sup": As_min_sup,
        "inf": inf, "sup": sup, "etat_inf": etat_inf, "etat_sup": etat_sup,
        "tau_1": tau_1, "tau_2": tau_2, "tau_4": tau_4,
        "shear": shear, "shear_r": shear_r,
        "etat_global": etat_global,
    }


# ============================================================
#  COUPE DE SECTION (schéma façon AutoCAD)
# ============================================================
def _first_stirrup(values, bid, sid):
    """Diamètre / type / nombre de la 1re ligne d'étriers (pour le dessin)."""
    typ = str(_g(values, KS("shear_line0_type", bid, sid), "Étriers (2 brins)"))
    diam = int(float(_g(values, KS("shear_line0_d", bid, sid), 8) or 8))
    return {"type": typ, "d": diam, "brins": _brins_from_type(typ)}


# couleurs du dessin
COL_CONC      = colors.HexColor("#eceff3")   # remplissage béton
COL_CONC_BD   = colors.HexColor("#2b2f36")   # contour béton
COL_REBAR     = colors.HexColor("#b11d1d")   # acier longitudinal (rouge)
COL_STIRRUP   = colors.HexColor("#1f5fb0")   # étrier (bleu)
COL_DIM       = colors.HexColor("#444a52")   # cotes
COL_HATCH     = colors.HexColor("#c2c8d0")   # hachures béton


class SectionDrawing(Flowable):
    """
    Coupe transversale de la poutre BA, dessinée à l'échelle.
    - béton hachuré, contour net
    - barres inf/sup (1er + 2e lit) au bon diamètre relatif
    - étrier (cadre) avec rayon de pliage
    - cotes b et h
    - annotations nombre + Ø à droite, en regard de chaque nappe
    """
    def __init__(self, R, stirrup, width, height):
        super().__init__()
        self.R = R
        self.stirrup = stirrup
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)

    def draw(self):
        c = self.canv
        R = self.R
        b_cm = float(R["b"]); h_cm = float(R["h"])
        enrob_i = float(R["enrob_inf"]); enrob_s = float(R["enrob_sup"])
        st_d = float(self.stirrup["d"])  # mm

        # ---- zone de dessin (réserver marges pour cotes + annotations) ----
        pad_left = 26      # cote h à gauche
        pad_top = 20       # cote b en haut
        pad_right = 92     # annotations barres à droite
        pad_bottom = 16
        avail_w = self.width - pad_left - pad_right
        avail_h = self.height - pad_top - pad_bottom

        # échelle (mm réels -> points), section homothétique
        b_mm = b_cm * 10.0
        h_mm = h_cm * 10.0
        sc = min(avail_w / b_mm, avail_h / h_mm)

        sec_w = b_mm * sc
        sec_h = h_mm * sc
        x0 = pad_left + (avail_w - sec_w) / 2.0
        y0 = pad_bottom + (avail_h - sec_h) / 2.0

        c.saveState()

        # ---- béton : hachures + contour ----
        c.setFillColor(COL_CONC)
        c.setStrokeColor(COL_CONC_BD)
        c.setLineWidth(1.4)
        c.rect(x0, y0, sec_w, sec_h, stroke=1, fill=1)

        # hachures diagonales légères
        c.saveState()
        p = c.beginPath()
        p.rect(x0, y0, sec_w, sec_h)
        c.clipPath(p, stroke=0, fill=0)
        c.setStrokeColor(COL_HATCH)
        c.setLineWidth(0.4)
        step = 7
        xx = x0 - sec_h
        while xx < x0 + sec_w:
            c.line(xx, y0, xx + sec_h, y0 + sec_h)
            xx += step
        c.restoreState()
        # re-contour net par-dessus les hachures
        c.setStrokeColor(COL_CONC_BD)
        c.setLineWidth(1.4)
        c.rect(x0, y0, sec_w, sec_h, stroke=1, fill=0)

        # ---- étrier (cadre) ----
        cov = min(enrob_i, enrob_s, 3.0)  # enrobage net visuel (cm) ~ jusqu'au cadre
        # on place le cadre à ~ enrobage béton réel s'il est dispo
        cov_mm = (cov) * 10.0 * sc
        st_off = cov_mm
        st_w_pts = max(2.0, st_d * sc)
        c.setStrokeColor(COL_STIRRUP)
        c.setLineWidth(max(1.0, st_w_pts))
        rr = max(3.0, 2.0 * st_d * sc)  # rayon de pliage
        c.roundRect(x0 + st_off, y0 + st_off,
                    sec_w - 2*st_off, sec_h - 2*st_off, rr, stroke=1, fill=0)

        # ---- barres longitudinales ----
        def draw_bars(n, d_mm, y_axis_cm, color=COL_REBAR):
            if n <= 0:
                return
            r_pts = max(1.6, (d_mm * sc) / 2.0)
            inset = st_off + st_w_pts/2.0 + r_pts + 1.0
            xa = x0 + inset
            xb = x0 + sec_w - inset
            yy = y0 + (y_axis_cm / h_cm) * sec_h
            if n == 1:
                xs = [(xa + xb) / 2.0]
            else:
                xs = [xa + (xb - xa) * k / (n - 1) for k in range(n)]
            c.setFillColor(color)
            c.setStrokeColor(colors.HexColor("#7a1010"))
            c.setLineWidth(0.5)
            for xc in xs:
                c.circle(xc, yy, r_pts, stroke=1, fill=1)
            return yy

        inf = R["inf"]; sup = R["sup"]
        # nappe inférieure (axe = enrob_inf depuis le bas)
        y_inf1 = enrob_i
        yy_inf1 = draw_bars(inf["n1"], inf["d1"], y_inf1)
        yy_inf2 = None
        if inf["has2"]:
            y_inf2 = enrob_i + inf["jeu"] + (inf["d1"] + inf["d2"]) / 20.0
            yy_inf2 = draw_bars(inf["n2"], inf["d2"], y_inf2)

        # nappe supérieure (axe = h - enrob_sup)
        y_sup1 = h_cm - enrob_s
        yy_sup1 = draw_bars(sup["n1"], sup["d1"], y_sup1)
        yy_sup2 = None
        if sup["has2"]:
            y_sup2 = h_cm - enrob_s - (sup["jeu"] + (sup["d1"] + sup["d2"]) / 20.0)
            yy_sup2 = draw_bars(sup["n2"], sup["d2"], y_sup2)

        # ---- cotes ----
        c.setStrokeColor(COL_DIM)
        c.setFillColor(COL_DIM)
        c.setLineWidth(0.6)
        c.setFont("Helvetica", 7.5)
        # cote b (en haut)
        yb = y0 + sec_h + 9
        c.line(x0, yb, x0 + sec_w, yb)
        for xx in (x0, x0 + sec_w):
            c.line(xx, yb - 2.5, xx, yb + 2.5)
        c.drawCentredString(x0 + sec_w/2, yb + 3, f"b = {f_num(b_cm,0)} cm")
        # cote h (à gauche, verticale)
        xl = x0 - 10
        c.line(xl, y0, xl, y0 + sec_h)
        for yy in (y0, y0 + sec_h):
            c.line(xl - 2.5, yy, xl + 2.5, yy)
        c.saveState()
        c.translate(xl - 3, y0 + sec_h/2)
        c.rotate(90)
        c.drawCentredString(0, 0, f"h = {f_num(h_cm,0)} cm")
        c.restoreState()

        # ---- annotations barres (à droite, en regard de chaque nappe) ----
        c.setFont("Helvetica-Bold", 8)
        xann = x0 + sec_w + 8
        def annotate(yy, n, d_mm, col):
            if yy is None or n <= 0:
                return
            c.setStrokeColor(col)
            c.setLineWidth(0.5)
            c.line(x0 + sec_w, yy, xann - 2, yy)
            c.setFillColor(col)
            c.drawString(xann, yy - 3, f"{n} {SYM['phi'].replace('&#216;','Ø')}{int(d_mm)}")

        # remplacer entité par caractère Ø direct
        def lbl(n, d):
            return f"{n} \u00d8{int(d)}"

        c.setFillColor(COL_REBAR)
        if yy_sup1 is not None:
            c.setStrokeColor(COL_REBAR); c.setLineWidth(0.5)
            c.line(x0 + sec_w, yy_sup1, xann - 2, yy_sup1)
            c.drawString(xann, yy_sup1 - 3, lbl(sup["n1"], sup["d1"]))
        if yy_sup2 is not None:
            c.line(x0 + sec_w, yy_sup2, xann - 2, yy_sup2)
            c.drawString(xann, yy_sup2 - 3, lbl(sup["n2"], sup["d2"]))
        if yy_inf1 is not None:
            c.line(x0 + sec_w, yy_inf1, xann - 2, yy_inf1)
            c.drawString(xann, yy_inf1 - 3, lbl(inf["n1"], inf["d1"]))
        if yy_inf2 is not None:
            c.line(x0 + sec_w, yy_inf2, xann - 2, yy_inf2)
            c.drawString(xann, yy_inf2 - 3, lbl(inf["n2"], inf["d2"]))

        # étrier : annotation en haut à droite
        c.setFillColor(COL_STIRRUP)
        c.setFont("Helvetica", 7.5)
        c.drawString(xann, y0 + sec_h - 2, f"cadre \u00d8{int(st_d)}")

        c.restoreState()


# ============================================================
#  CONSTRUCTION DES BLOCS (Flowables)
# ============================================================
def _badge_para(etat, styles):
    vis = ETAT_VIS.get(etat, ETAT_VIS["ok"])
    return Paragraph(
        f'<font color="{vis["tx"].hexval()}">[{vis["ico"]}] {vis["label"]}</font>',
        styles["badge"],
    )


def _block(title, etat, body_flowables, styles, content_width):
    """
    Construit un "paragraphe" encadré (titre + badge + corps), prêt à KeepTogether.
    """
    vis = ETAT_VIS.get(etat, ETAT_VIS["ok"])

    head = Table(
        [[Paragraph(title, styles["block_title"]), _badge_para(etat, styles)]],
        colWidths=[content_width * 0.62, content_width * 0.38],
    )
    head.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    inner = [head, HLine(content_width, vis["bd"], 1.0), Spacer(1, 4)]
    inner.extend(body_flowables)
    inner.append(Spacer(1, 4))

    outer = Table([[inner]], colWidths=[content_width])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), vis["bg"]),
        ("BOX", (0, 0), (-1, -1), 0.8, vis["bd"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ROUNDEDCORNERS", [5, 5, 5, 5]),
    ]))
    return outer


def _formula_table(rows, styles, content_width, label_w=0.30, divider=True):
    """
    rows = liste de (label, formule_markup).
    Affiche libellé à gauche, formule à droite (style 'note de calcul').
    divider=False -> pas de lignes horizontales entre les formules.
    """
    data = []
    for lab, form in rows:
        data.append([
            Paragraph(lab, styles["formula_label"]),
            Paragraph(form, styles["formula"]),
        ])
    t = Table(data, colWidths=[content_width * label_w, content_width * (1 - label_w) - 20])
    ts = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if divider:
        ts.append(("LINEBELOW", (0, 0), (-1, -2), 0.4, COL_LINE))
    t.setStyle(TableStyle(ts))
    return t


def _kv_table(rows, styles, content_width, ncols=3, sep=" = "):
    """Petite table clé/valeur en colonnes (pour résultats chiffrés).
    Clé et valeur sont sur la MÊME ligne, séparées par `sep`."""
    cells = []
    line = []
    for (k, v) in rows:
        if k:
            line.append(Paragraph(f"<b>{k}</b>{sep}{v}", styles["cell"]))
        else:
            line.append(Paragraph(f"{v}", styles["cell"]))
        if len(line) == ncols:
            cells.append(line)
            line = []
    if line:
        while len(line) < ncols:
            line.append(Paragraph("", styles["cell"]))
        cells.append(line)
    cw = content_width / ncols
    t = Table(cells, colWidths=[cw] * ncols)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _conclusion(text, etat, styles, content_width):
    vis = ETAT_VIS.get(etat, ETAT_VIS["ok"])
    p = Paragraph(
        f'<font color="{vis["tx"].hexval()}">{SYM["rightarrow"]} {text}</font>',
        styles["concl"],
    )
    t = Table([[p]], colWidths=[content_width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.6, vis["bd"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ---------- Blocs spécifiques ----------
def block_hauteur(R, styles, cw):
    body = []
    body.append(_formula_table([
        ("Moment de calcul",
         f"M<sub>max</sub> = max(M<sub>inf</sub> ; M<sub>sup</sub>) = {f_num(R['M_max'],1)} kNm"),
        ("Hauteur minimale requise",
         f"h<sub>min</sub> = {SYM['sqrt']}( M<sub>max</sub> / ({SYM['alpha']}<sub>b</sub> {SYM['times']} b {SYM['times']} {chr(956)}) )"),
        ("",
         f"h<sub>min</sub> = {SYM['sqrt']}( {f_num(R['M_max']*1e6,0)} / ({f_num(R['alpha_b'],2)} {SYM['times']} {f_num(R['b']*10,0)} {SYM['times']} {f_num(R['mu_val'],4)}) ) = <b>{f_num(R['hmin'],1)} cm</b>"),
    ], styles, cw, divider=False))
    body.append(Spacer(1, 5))
    body.append(HLine(cw - 20, COL_LINE, 0.5))
    body.append(Spacer(1, 5))
    body.append(_kv_table([
        ("h<sub>min</sub>", f"{f_num(R['hmin'],1)} cm"),
        ("h<sub>min</sub> + enrobage", f"{f_num(R['hmin']+R['enrob_inf'],1)} cm"),
        ("hauteur", f"{f_num(R['h'],1)} cm"),
    ], styles, cw, ncols=3))
    body.append(Spacer(1, 4))
    ok = R["etat_h"] == "ok"
    body.append(_conclusion(
        f"h<sub>min</sub> + enrobage = {f_num(R['hmin']+R['enrob_inf'],1)} cm "
        f"{SYM['leq'] if ok else SYM['geq']} h = {f_num(R['h'],1)} cm "
        f"{'— Hauteur suffisante.' if ok else '— Hauteur insuffisante !'}",
        R["etat_h"], styles, cw))
    return _block("1. Vérification de la hauteur", R["etat_h"], body, styles, cw)


def block_armatures(R, styles, cw, which):
    if which == "inf":
        title = "2. Armatures inférieures (flexion positive)"
        M = R["M_inf"]; As_req = R["As_req_inf"]; As_min = R["As_min_inf"]
        lay = R["inf"]; d = R["d_inf"]; etat = R["etat_inf"]; sub = "inf"
    else:
        title = "3. Armatures supérieures (flexion négative)"
        M = R["M_sup"]; As_req = R["As_req_sup"]; As_min = R["As_min_sup"]
        lay = R["sup"]; d = R["d_sup"]; etat = R["etat_sup"]; sub = "sup"

    body = []
    body.append(_formula_table([
        ("Hauteur utile",
         f"d<sub>{sub}</sub> = h &#8722; enrobage = {f_num(R['h'],1)} &#8722; {f_num(R['enrob_inf'] if which=='inf' else R['enrob_sup'],1)} = <b>{f_num(d,1)} cm</b>"),
        ("Section d'acier requise",
         f"A<sub>s,req</sub> = M<sub>{sub}</sub> / (f<sub>yd</sub> {SYM['times']} 0,9 {SYM['times']} d)"),
        ("",
         f"A<sub>s,req</sub> = {f_num(M*1e6,0)} / ({f_num(R['fyd'],1)} {SYM['times']} 0,9 {SYM['times']} {f_num(d*10,0)}) = <b>{f_num(As_req,0)} mm{_sup('2')}</b>"),
    ], styles, cw, divider=False))
    body.append(Spacer(1, 5))
    body.append(HLine(cw - 20, COL_LINE, 0.5))
    body.append(Spacer(1, 5))
    body.append(_kv_table([
        ("A<sub>s,req</sub>", f"{f_num(As_req,0)} mm{_sup('2')}"),
        ("A<sub>s,min</sub>", f"{f_num(As_min,0)} mm{_sup('2')}"),
        ("A<sub>s,max</sub>", f"{f_num(R['As_max'],0)} mm{_sup('2')}"),
    ], styles, cw, ncols=3))
    body.append(Spacer(1, 3))

    # détail choix
    if lay["has2"]:
        choix_txt = (f"{lay['n1']}{SYM['phi']}{lay['d1']} + {lay['n2']}{SYM['phi']}{lay['d2']} "
                     f"(2 lits, jeu {f_num(lay['jeu'],1)} cm)")
    else:
        choix_txt = f"{lay['n1']}{SYM['phi']}{lay['d1']}"
    body.append(_kv_table([
        ("Choix retenu", choix_txt),
        ("A<sub>s,prév</sub>", f"{f_num(lay['As'],1)} mm{_sup('2')}"),
        ("",
         f"{SYM['geq']} A<sub>s,min</sub> : {'OK' if lay['As']>=As_min else 'NON'} &#160;&#160; "
         f"{SYM['leq']} A<sub>s,max</sub> : {'OK' if lay['As']<=R['As_max'] else 'NON'}"),
    ], styles, cw, ncols=3, sep=" : "))
    body.append(Spacer(1, 4))
    ok = etat == "ok"
    body.append(_conclusion(
        f"A<sub>s,prév</sub> = {f_num(lay['As'],1)} mm{_sup('2')} "
        f"{SYM['geq'] if ok else SYM['geq']} A<sub>s,req</sub> = {f_num(As_req,0)} mm{_sup('2')} "
        f"{'&#8212; Section d&#8217;acier vérifiée.' if ok else '&#8212; Section d&#8217;acier insuffisante !'}",
        etat, styles, cw))
    return _block(title, etat, body, styles, cw)


def block_shear(R, styles, cw, reduced=False):
    S = R["shear_r"] if reduced else R["shear"]
    idx = "4" if not reduced else "5"
    suffix = " réduit" if reduced else ""
    title = f"{idx}. Effort tranchant{suffix} — vérification & étriers"

    body = []
    body.append(_formula_table([
        ("Contrainte de cisaillement",
         f"{SYM['tau']} = V / (0,75 {SYM['times']} b {SYM['times']} h)"),
        ("",
         f"{SYM['tau']} = {f_num(S['V']*1e3,0)} / (0,75 {SYM['times']} {f_num(R['b']*10,0)} {SYM['times']} {f_num(R['h']*10,0)}) "
         f"= <b>{f_num(S['tau'],2)} N/mm{_sup('2')}</b>"),
    ], styles, cw, divider=False))
    body.append(Spacer(1, 5))
    body.append(HLine(cw - 20, COL_LINE, 0.5))
    body.append(Spacer(1, 5))
    body.append(_kv_table([
        ("&#964;", f"{f_num(S['tau'],2)} N/mm{_sup('2')}"),
        ("&#964;<sub>adm</sub>", f"{f_num(S['tau_lim'],2)} N/mm{_sup('2')}"),
        ("", S["besoin"]),
    ], styles, cw, ncols=3, sep=" : "))
    body.append(Spacer(1, 6))

    body.append(Paragraph("<b>Détermination des étriers</b>", styles["body"]))
    body.append(Spacer(1, 3))
    body.append(_formula_table([
        ("Section d'armature transversale",
         f"A<sub>st</sub> = {f_num(S['Ast'],1)} mm{_sup('2')} &#8212; ({S['summary']})"),
        ("Pas théorique",
         f"s<sub>th</sub> = A<sub>st</sub> {SYM['times']} f<sub>yd</sub> {SYM['times']} d / V"),
        ("",
         f"s<sub>th</sub> = {f_num(S['Ast'],1)} {SYM['times']} {f_num(R['fyd'],1)} {SYM['times']} {f_num(R['d_shear']*10,0)} / {f_num(S['V']*1e3,0)} "
         f"= <b>{f_num(S['pas_th'],1)} cm</b>"),
        ("Pas maximal",
         f"s<sub>max</sub> = min(0,75 {SYM['times']} d ; 30) = <b>{f_num(S['s_max'],1)} cm</b>"),
    ], styles, cw, divider=False))
    body.append(Spacer(1, 5))
    body.append(HLine(cw - 20, COL_LINE, 0.5))
    body.append(Spacer(1, 5))
    body.append(_kv_table([
        ("Pas théorique", f"{f_num(S['pas_th'],1)} cm"),
        ("Pas maximal", f"{f_num(S['s_max'],1)} cm"),
        ("Pas retenu", f"{f_num(S['pas'],1)} cm"),
    ], styles, cw, ncols=3))
    body.append(Spacer(1, 4))

    etat = "nok" if (S["etat_tau"] == "nok" or S["etat_pas"] == "nok") else \
           ("warn" if (S["etat_tau"] == "warn" or S["etat_pas"] == "warn") else "ok")
    ok_pas = S["pas"] <= S["pas_lim"]
    extra = f" {S['suf_pas']}" if S["suf_pas"] else ""
    body.append(_conclusion(
        f"Pas retenu = {f_num(S['pas'],1)} cm "
        f"{SYM['leq'] if ok_pas else SYM['geq']} pas limite = {f_num(S['pas_lim'],1)} cm.{extra}",
        etat, styles, cw))
    return _block(title, etat, body, styles, cw)


# ============================================================
#  EN-TÊTE SECTION (récap données)
# ============================================================
def _carac_cadre(R, styles, half_w):
    """Cadre arrondi 'Caractéristiques', organisé en Dimensions / Matériaux / Sollicitations.
    Largeur = demi-page. Les lignes M_sup / V_réduit ne s'affichent que si présentes."""
    def sous_titre(txt):
        return Paragraph(f'<font color="{COL_PRIMARY.hexval()}"><b>{txt}</b></font>', styles["cell"])

    def kv(k, v):
        return [Paragraph(k, styles["cell"]), Paragraph(str(v), styles["cell_b"])]

    rows = []
    # --- Dimensions ---
    rows.append([sous_titre("Dimensions"), Paragraph("", styles["cell"])])
    rows.append(kv("Largeur b", f"{f_num(R['b'],0)} cm"))
    rows.append(kv("Hauteur h", f"{f_num(R['h'],0)} cm"))
    rows.append(kv("Enrobage (inf. / sup.)",
                   f"{f_num(R['enrob_inf'],1)} / {f_num(R['enrob_sup'],1)} cm"))
    # --- Matériaux ---
    rows.append([sous_titre("Matériaux"), Paragraph("", styles["cell"])])
    rows.append(kv("Béton", f"{R['beton']}  (f<sub>ck</sub> = {f_num(R['fck'],0)} N/mm{_sup('2')})"))
    rows.append(kv("Acier", f"B{int(R['fyk'])}  (f<sub>yd</sub> = {f_num(R['fyd'],1)} N/mm{_sup('2')})"))
    # --- Sollicitations ---
    rows.append([sous_titre("Sollicitations"), Paragraph("", styles["cell"])])
    rows.append(kv("M<sub>inf</sub>", f"{f_num(R['M_inf'],1)} kNm"))
    if R["has_Msup"]:
        rows.append(kv("M<sub>sup</sub>", f"{f_num(R['M_sup'],1)} kNm"))
    rows.append(kv("V", f"{f_num(R['V'],1)} kN"))
    if R["has_Vlim"]:
        rows.append(kv("V<sub>réduit</sub>", f"{f_num(R['V_lim'],1)} kN"))

    inner = Table(rows, colWidths=[half_w*0.52, half_w*0.48])
    # styles : fond alterné seulement sur les lignes de données, sous-titres sur fond doux
    ts = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("SPAN", (0, 0), (1, 0)),
    ]
    # repérer les lignes de sous-titres pour SPAN + fond
    sub_idx = [i for i, r in enumerate(rows)
               if isinstance(r[0], Paragraph) and "<b>" in r[0].text and r[1].text == ""]
    for i in sub_idx:
        ts.append(("SPAN", (0, i), (1, i)))
        ts.append(("BACKGROUND", (0, i), (-1, i), COL_BG_SOFT))
        ts.append(("LINEBELOW", (0, i), (-1, i), 0.5, COL_LINE))
        ts.append(("TOPPADDING", (0, i), (-1, i), 5))
        ts.append(("BOTTOMPADDING", (0, i), (-1, i), 4))
    inner.setStyle(TableStyle(ts))

    # encadré arrondi (même style visuel que les cadres de résultats, neutre)
    outer = Table([[inner]], colWidths=[half_w])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, COL_LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    return outer


def _section_recap(R, styles, cw, values=None, bid=None, sid=None):
    """Deux colonnes : caractéristiques (gauche) + coupe de section (droite)."""
    half = cw * 0.5
    gap = 10
    left_w = half - gap/2
    right_w = cw - half - gap/2

    carac = _carac_cadre(R, styles, left_w)

    # dessin : hauteur calée sur le contenu (≈ celle du cadre carac)
    if values is not None and bid is not None and sid is not None:
        stirrup = _first_stirrup(values, bid, sid)
    else:
        stirrup = {"type": "Étriers (2 brins)", "d": 8, "brins": 2}
    draw_h = 190
    drawing = SectionDrawing(R, stirrup, right_w, draw_h)

    title_draw = Paragraph('<font color="%s"><b>Coupe de section</b></font>' % COL_PRIMARY.hexval(),
                           styles["cell"])
    right_cell = Table([[title_draw], [drawing]], colWidths=[right_w])
    right_cell.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, 0), 0),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    layout = Table([[carac, "", right_cell]], colWidths=[left_w, gap, right_w])
    layout.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return layout


# ============================================================
#  DOC TEMPLATE (header / footer)
# ============================================================
class NoteDocTemplate(BaseDocTemplate):
    def __init__(self, filename, infos, **kw):
        self.infos = infos or {}
        super().__init__(filename, pagesize=A4,
                         leftMargin=16*mm, rightMargin=16*mm,
                         topMargin=28*mm, bottomMargin=18*mm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin,
                      self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                            onPage=self._decor)])

    def _decor(self, canvas, doc):
        canvas.saveState()
        w, h = A4
        band_h = 20*mm
        # ---- En-tête (bandeau)
        canvas.setFillColor(COL_PRIMARY)
        canvas.rect(0, h - band_h, w, band_h, stroke=0, fill=1)

        # Ligne 1 : nom du bureau / Ligne 2 : rédigé par (initiales)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(16*mm, h - 9*mm, "Bureau méthodes et stabilité Valens")
        canvas.setFont("Helvetica", 9)
        initiales = self.infos.get("initiales") or ""
        canvas.drawString(16*mm, h - 15*mm, f"Rédigé par : {initiales}")

        # Bloc projet / partie à droite
        canvas.setFont("Helvetica", 8.5)
        proj = self.infos.get("nom_projet") or "—"
        canvas.drawRightString(w - 16*mm, h - 9*mm, f"Projet : {proj}")
        partie = self.infos.get("partie") or ""
        if partie:
            canvas.drawRightString(w - 16*mm, h - 15*mm, f"Partie : {partie}")

        # accent line
        canvas.setStrokeColor(COL_ACCENT)
        canvas.setLineWidth(2)
        canvas.line(0, h - band_h, w, h - band_h)

        # ---- Pied de page
        canvas.setStrokeColor(COL_LINE)
        canvas.setLineWidth(0.6)
        canvas.line(16*mm, 14*mm, w - 16*mm, 14*mm)
        canvas.setFillColor(COL_MUTED)
        canvas.setFont("Helvetica", 8)
        date = self.infos.get("date") or datetime.today().strftime("%d/%m/%Y")
        indice = self.infos.get("indice") or "0"
        canvas.drawString(16*mm, 9.5*mm, f"Date : {date}    |    Indice : {indice}")
        canvas.drawRightString(w - 16*mm, 9.5*mm, f"Page {doc.page}")
        canvas.restoreState()


# ============================================================
#  PAGE DE GARDE
# ============================================================
def _cover(infos, beams, values, beton_data, styles, cw, beam_pages=None):
    beam_pages = beam_pages or {}
    story = []
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("Note de calcul", styles["doc_title"]))
    story.append(Paragraph("Dimensionnement de poutres en béton armé", styles["doc_sub"]))
    story.append(Spacer(1, 4))
    story.append(HLine(cw, COL_ACCENT, 2))
    story.append(Spacer(1, 14))

    rows = [
        ("Projet", infos.get("nom_projet") or "—"),
        ("Partie", infos.get("partie") or "—"),
        ("Date", infos.get("date") or datetime.today().strftime("%d/%m/%Y")),
        ("Indice", infos.get("indice") or "0"),
        ("Nombre de poutres", str(len(beams))),
        ("Nombre total de sections", str(sum(len(b.get("sections", [])) for b in beams))),
    ]
    data = [[Paragraph(f"<b>{k}</b>", styles["cell"]), Paragraph(str(v), styles["cell"])] for k, v in rows]
    t = Table(data, colWidths=[cw*0.35, cw*0.65])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [COL_BG_SOFT, colors.white]),
        ("BOX", (0, 0), (-1, -1), 0.6, COL_LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, COL_LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))

    # Sommaire des poutres
    story.append(Paragraph("Sommaire", styles["sec_title"]))
    story.append(Spacer(1, 4))
    summ = [[Paragraph("<b>Poutre</b>", styles["cell_head"]),
             Paragraph("<b>Sections</b>", styles["cell_head"]),
             Paragraph("<b>Béton / Acier</b>", styles["cell_head"]),
             Paragraph("<b>État</b>", styles["cell_head"]),
             Paragraph("<b>Page</b>", styles["cell_head"])]]
    for b in beams:
        bid = int(b["id"])
        sec_states = []
        for s in b.get("sections", []):
            R = _compute_section(values, beton_data, bid, int(s["id"]))
            sec_states.append(R["etat_global"])
        if any(x == "nok" for x in sec_states):
            eg = "nok"
        elif any(x == "warn" for x in sec_states):
            eg = "warn"
        else:
            eg = "ok"
        vis = ETAT_VIS[eg]
        beton = str(_g(values, KB("beton", bid), "—"))
        fyk = str(_g(values, KB("fyk", bid), "500"))
        sec_names = ", ".join(str(_g(values, f"meta_b{bid}_nom_{int(s['id'])}", s.get("nom", "")))
                              for s in b.get("sections", []))
        pg = beam_pages.get(bid)
        pg_txt = f"p.{pg}" if pg else "—"
        summ.append([
            Paragraph(str(_g(values, f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}"))), styles["cell_b"]),
            Paragraph(sec_names, styles["cell"]),
            Paragraph(f"{beton} / B{fyk}", styles["cell"]),
            Paragraph(f'<font color="{vis["tx"].hexval()}"><b>{vis["label"]}</b></font>', styles["cell"]),
            Paragraph(pg_txt, styles["cell_b"]),
        ])
    ts = Table(summ, colWidths=[cw*0.24, cw*0.30, cw*0.20, cw*0.16, cw*0.10])
    ts.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COL_PRIMARY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COL_BG_SOFT]),
        ("GRID", (0, 0), (-1, -1), 0.4, COL_LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(ts)
    story.append(PageBreak())
    return story


# ============================================================
#  API PRINCIPALE
# ============================================================
def _build_story(beams, values, beton_data, infos, styles, cw, beam_pages, page_store):
    """Construit le story complet. `page_store` reçoit les pages de début de poutre
    (via PageMarker) lors du build. `beam_pages` alimente la colonne Page du sommaire."""
    story = []
    story.extend(_cover(infos, beams, values, beton_data, styles, cw, beam_pages=beam_pages))

    for bi, b in enumerate(beams):
        bid = int(b["id"])
        bnom = str(_g(values, f"meta_beam_nom_{bid}", b.get("nom", f"Poutre {bid}")))

        # Chaque poutre commence sur une nouvelle page (la 1re suit la page de garde,
        # qui se termine déjà par un PageBreak).
        if bi > 0:
            story.append(PageBreak())

        # Marqueur de page (capture la page de début de cette poutre)
        story.append(PageMarker(page_store, bid))

        # bandeau poutre (aligné au bord des cadres)
        story.append(make_banner(bnom, cw, kind="beam"))
        story.append(Spacer(1, 10))

        sections = b.get("sections", [])
        for si, s in enumerate(sections):
            sid = int(s["id"])
            # Nom de section : "Section" + valeur (ex. "A") -> "Section A".
            # Si le nom stocké contient déjà "Section", on le garde tel quel.
            raw_nom = str(_g(values, f"meta_b{bid}_nom_{sid}", s.get("nom", f"Section {sid}")))
            if raw_nom.strip().lower().startswith("section"):
                snom = raw_nom
            else:
                snom = f"Section {raw_nom}"
            R = _compute_section(values, beton_data, bid, sid)

            vis = ETAT_VIS[R["etat_global"]]
            # bandeau section (même forme que poutre, couleur plus atténuée) + état à droite
            sec_banner = make_banner(
                f"{snom}"
                f"&#160;&#160;&#160;<font size=9>[{vis['ico']}] {vis['label']}</font>",
                cw, kind="section")

            # Liste ordonnée des blocs à produire pour cette section.
            blocs = [block_hauteur(R, styles, cw)]
            if R["M_inf"] > 0:
                blocs.append(block_armatures(R, styles, cw, "inf"))
            if R["has_Msup"]:
                blocs.append(block_armatures(R, styles, cw, "sup"))
            if R["shear"]:
                blocs.append(block_shear(R, styles, cw, reduced=False))
            if R["shear_r"]:
                blocs.append(block_shear(R, styles, cw, reduced=True))

            # En-tête de section (bandeau + récap+coupe) regroupé avec le 1er bloc.
            intro = [
                sec_banner,
                Spacer(1, 6),
                _section_recap(R, styles, cw, values=values, bid=bid, sid=sid),
                Spacer(1, 10),
                blocs[0],
            ]
            story.append(KeepTogether(intro))

            # Les blocs suivants s'enchaînent et REMPLISSENT la page.
            # Chaque cadre reste insécable (KeepTogether) : s'il ne tient pas
            # dans l'espace restant, il bascule entier sur la page suivante.
            for blk in blocs[1:]:
                story.append(Spacer(1, 8))
                story.append(KeepTogether([blk]))

            # Petit espace entre deux sections d'une même poutre (sans saut forcé).
            if si < len(sections) - 1:
                story.append(Spacer(1, 14))

    return story


def generer_rapport_pdf(beams, values, beton_data, infos=None, output_path=None):
    """
    Génère la note de calcul PDF et renvoie le chemin du fichier.
    Deux passes : la 1re mesure la page de début de chaque poutre,
    la 2e produit le sommaire avec les bons numéros de page.
    """
    infos = infos or {}
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".pdf", prefix="note_poutre_")
        os.close(fd)

    styles = _build_styles()

    # ---- Passe 1 : mesure des pages (sortie jetable) ----
    page_store = {}
    tmp_pdf = output_path + ".pass1.tmp"
    doc1 = NoteDocTemplate(tmp_pdf, infos)
    cw = doc1.width
    story1 = _build_story(beams, values, beton_data, infos, styles, cw, beam_pages={}, page_store=page_store)
    doc1.build(story1)
    try:
        os.remove(tmp_pdf)
    except OSError:
        pass

    # ---- Passe 2 : build final avec les numéros de page connus ----
    doc2 = NoteDocTemplate(output_path, infos)
    cw = doc2.width
    story2 = _build_story(beams, values, beton_data, infos, styles, cw,
                          beam_pages=dict(page_store), page_store={})
    doc2.build(story2)
    return output_path

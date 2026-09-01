# -*- coding: utf-8 -*-
# ============================================================
#  export_pdf_dalle.py — Note de calcul PDF (dalle béton armé)
#  VERSION 2.0 (alignée sur dalle.py 2.0 — dalle BIDIRECTIONNELLE)
#
#  Évolutions v2.0 :
#   - GÉNÉRATION : generer_rapport_pdf (même signature) produit la note
#     via le paquet ndc_pdf/, comme la note Poutre v3.0 : garde A4
#     portrait, une planche A4 paysage par section, conclusions
#     vert/ocre/rouge, coupe de la bande de dalle (draw_dalle).
#   - CALCUL par DIRECTION (X, Y) : mêmes formules qu'avant, exécutées
#     indépendamment pour chaque direction ; sollicitations Mx/My
#     inf/sup + V max ; direction principale = plus grand moment.
#   - Anciennes clés v1 (M_inf, couches inf/sup) lues en repli : un
#     dict de valeurs non migré reste exploitable.
#   - Les aides platypus v1 restent présentes (aucun autre module ne
#     doit casser) mais ne servent plus au rapport.
#
#  VERSION 1.0 (alignée sur dalle.py 1.0)
#
#  Construit sur les MÊMES briques graphiques que export_pdf.py
#  (moteur de formules vectorielles, styles, bandeaux, blocs,
#  en-tête / pied de page) importées telles quelles : la note Dalle a
#  strictement la même identité visuelle que la note Poutre —
#  typographie, tailles, cadres, couleurs, bandeaux, marges, footer,
#  numérotation, présentation des équations et des conclusions.
#  export_pdf.py n'est PAS modifié : le rapport Poutre est inchangé.
#
#  Spécificités Dalle :
#   - armatures par COUCHES (treillis et/ou barres Ø/esp., base +
#     renforts) : sections calculées en mm²/m puis rapportées à la
#     largeur réelle de la bande ;
#   - COUPE DE DALLE : bande b×h avec treillis inf./sup., barres de
#     renfort, cotations réelles ("b = 100 cm", "h = 20 cm") — le
#     dessin exagère l'épaisseur si nécessaire pour rester lisible ;
#   - mêmes formules de calcul que le module Poutre (hᵤ,min, Aₛ,req,
#     Aₛ,min, Aₛ,max, τ, pas des étriers) — aucun coefficient modifié.
#
#  API : generer_rapport_pdf(dalles, values, beton_data, infos=None) -> chemin PDF
# ============================================================

import math
import os
import tempfile

from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle,
    KeepTogether, PageBreak, Flowable,
)

# Briques graphiques partagées avec la note Poutre (import sans effet
# de bord : export_pdf.py n'est pas modifié).
from modules.export_pdf import (
    INK, MUTE, HAIR, ECOL, ELAB, LIT_COLORS, PAL,
    fn, s2, txt, nb, sci_tokens,
    Row, Frac, Sqrt, Formula,
    ST, HR, Marker,
    fline, reslines, conclu, block,
    beam_banner, sec_banner, NoteDoc,
    _bar_area_mm2, _brins_from_type, _round_up_to_half_cm,
    _g, _get_gamma_s, _asmin_formula, b_shear,
)
from modules import treillis as TRD

MAX_COUCHES = 4


def _t(s, **k):
    return txt(s, **k)


# ============================================================
#  ACCÈS AUX VALEURS (mêmes clés que dalle.py 1.0)
# ============================================================
def KD(base, did):
    return f"dal{did}_{base}"


def KS(base, did, sid):
    return f"dal{did}_sec{sid}_{base}"


def _get_fyk(values, did):
    try:
        cur = int(float(_g(values, KD("fyk", did), 500)))
    except Exception:
        cur = 500
    if cur not in (400, 500):
        cur = 500
    return float(cur), str(cur)


def _get_ncouches(values, did, sid, which):
    try:
        nc = int(_g(values, KS(f"ncouches_{which}", did, sid), 1) or 1)
    except Exception:
        nc = 1
    return max(1, min(MAX_COUCHES, nc))


def _couche_data(values, did, sid, which, i):
    """(type, désignation treillis, Ø barres, esp barres, nombre) de la
    couche i — mêmes règles que dalle.py (« n barres » = nombre posé
    dans la bande)."""
    typ = str(_g(values, KS(f"arm_type_{which}_c{i}", did, sid),
                 "Treillis" if i == 1 else "Barres"))
    if typ not in ("Treillis", "Barres", "n barres"):
        typ = "Treillis"
    des = str(_g(values, KS(f"treillis_{which}_c{i}", did, sid), TRD.TREILLIS_DEFAUT))
    try:
        d = int(float(_g(values, KS(f"ø_barres_{which}_c{i}", did, sid), 12) or 12))
    except Exception:
        d = 12
    try:
        esp = float(_g(values, KS(f"esp_barres_{which}_c{i}", did, sid), 150) or 150)
    except Exception:
        esp = 150.0
    try:
        n = int(float(_g(values, KS(f"n_barres_{which}_c{i}", did, sid), 3) or 3))
    except Exception:
        n = 3
    return typ, des, d, esp, max(1, n)


def _couche_as_per_m(values, did, sid, which, i):
    typ, des, d, esp, n = _couche_data(values, did, sid, which, i)
    if typ == "Treillis":
        return TRD.as_treillis_mm2_m(des)
    if typ == "n barres":
        b = float(_g(values, KD("b", did), 100) or 100)
        return n * (math.pi * d * d / 4.0) / max(0.01, b / 100.0)
    return TRD.as_barres_mm2_m(d, esp)


def _couche_diam_mm(values, did, sid, which, i):
    typ, des, d, esp, n = _couche_data(values, did, sid, which, i)
    if typ == "Treillis":
        t = TRD.parse_designation(des)
        return float(t[0]) if t else 10.0
    return float(d)


def _couche_esp_mm(values, did, sid, which, i):
    """Espacement des fils/barres porteurs (mm) — sert au dessin."""
    typ, des, d, esp, n = _couche_data(values, did, sid, which, i)
    if typ == "Treillis":
        t = TRD.parse_designation(des)
        return float(t[2]) if t else 100.0
    return float(esp) if esp > 0 else 100.0


def _couche_n_barres(values, did, sid, which, i):
    """Nombre de barres si la couche est « n barres », sinon 0 (le dessin
    répartit alors à l'espacement)."""
    typ, des, d, esp, n = _couche_data(values, did, sid, which, i)
    return n if typ == "n barres" else 0


def _couche_label(values, did, sid, which, i):
    """'Treillis 10/10/100/100', 'Ø12/150' ou '3 Ø12' (libellé compact)."""
    typ, des, d, esp, n = _couche_data(values, did, sid, which, i)
    if typ == "Treillis":
        return f"Treillis {des}"
    if typ == "n barres":
        return f"{n} Ø{d}"
    esp_txt = f"{esp:.0f}" if abs(esp - round(esp)) < 1e-9 else fn(esp, 1)
    return f"Ø{d}/{esp_txt}"


def _couche_valeur(values, did, sid, which, i):
    """Valeur seule pour la légende de la coupe : '10/10/100/100',
    'Ø12/150' ou '3 Ø12'."""
    typ, des, d, esp, n = _couche_data(values, did, sid, which, i)
    if typ == "Treillis":
        return des
    if typ == "n barres":
        return f"{n} Ø{d}"
    esp_txt = f"{esp:.0f}" if abs(esp - round(esp)) < 1e-9 else fn(esp, 1)
    return f"Ø{d}/{esp_txt}"


def _auto_dist_couche(values, did, sid, which, i):
    """Distance d'axe automatique de la couche i (cm) — même formule que
    dalle.py : enrobage + demi-Ø arrondi au 0,5 cm sup. + jeu premier lit."""
    enrob_beton = float(_g(values, KD("enrobage_beton", did), 3.0) or 3.0)
    jeu1 = float(_g(values, "jeu_enrobage_cm", 0.0) or 0.0)
    d = _couche_diam_mm(values, did, sid, which, i)
    return enrob_beton + _round_up_to_half_cm(d / 20.0) + jeu1


def _dist_couche_eff(values, did, sid, which, i):
    """Distance d'axe EFFECTIVE (cm) : la saisie de la colonne
    « Dist. axe » quand elle est active et valide, sinon l'automatique —
    même règle que dalle.py._dist_couche_eff."""
    auto = _auto_dist_couche(values, did, sid, which, i)
    if bool(_g(values, KS(f"dist_auto_{which}_c{i}", did, sid), True)):
        return auto
    raw = str(_g(values, KS(f"dist_axe_{which}_c{i}", did, sid), "") or "").strip()
    try:
        v = float(raw.replace(",", "."))
        return v if v > 0 else auto
    except Exception:
        return auto


def _layers_geometry(values, did, sid, which):
    """As total (bande), As mm²/m, e_cdg, liste de couches, detail —
    yG manuel pris en compte comme dans dalle.py."""
    b = float(_g(values, KD("b", did), 100))
    nc = _get_ncouches(values, did, sid, which)
    couches = []
    As_pm_tot = 0.0
    somme = 0.0
    parts = []
    for i in range(1, nc + 1):
        typ, des, d, esp, n = _couche_data(values, did, sid, which, i)
        As_pm = _couche_as_per_m(values, did, sid, which, i)
        e = _dist_couche_eff(values, did, sid, which, i)
        As_pm_tot += As_pm
        somme += As_pm * e
        couches.append({
            "i": i, "typ": typ, "des": des, "d": _couche_diam_mm(values, did, sid, which, i),
            "esp": _couche_esp_mm(values, did, sid, which, i),
            "n": _couche_n_barres(values, did, sid, which, i),
            "As_pm": As_pm, "e": e,
            "label": _couche_label(values, did, sid, which, i),
            "valeur": _couche_valeur(values, did, sid, which, i),
        })
        parts.append(_couche_label(values, did, sid, which, i))
    e_cdg = (somme / As_pm_tot) if As_pm_tot > 0 else _dist_couche_eff(values, did, sid, which, 1)
    # yG imposé uniquement si l'utilisateur a désactivé le mode auto.
    if not bool(_g(values, KS(f"ycdg_auto_{which}", did, sid), False)):
        raw = str(_g(values, KS(f"ycdg_{which}", did, sid), "") or "").strip()
        if raw:
            try:
                v = float(raw.replace(",", "."))
                if v > 0:
                    e_cdg = v
            except Exception:
                pass
    As_tot = As_pm_tot * (b / 100.0)
    return {"As": As_tot, "As_pm": As_pm_tot, "e_cdg": e_cdg, "couches": couches,
            "detail": " + ".join(parts), "nc": nc}


def _shear_lines(values, did, sid):
    """Ast (mm²), résumé, groupes — sans positions de barres (dalle)."""
    n_lines = max(1, int(_g(values, KS("shear_n_lines", did, sid), 1) or 1))
    prefix = "shear_line"
    Ast = 0.0; parts = []; groups = []
    for i in range(n_lines):
        typ = str(_g(values, KS(f"{prefix}{i}_type", did, sid), "Étrier"))
        diam = float(_g(values, KS(f"{prefix}{i}_d", did, sid), 10) or 10)
        brins = _brins_from_type(typ)
        base = "Épingle" if brins == 1 else "Étrier"
        Ast += brins * _bar_area_mm2(diam)
        parts.append(f"{base} Ø{int(diam)}")
        groups.append({"type": typ, "d": int(diam), "brins": brins, "from": None, "to": None})
    return Ast, " + ".join(parts), groups


# ============================================================
#  CALCUL SECTION (formules strictement identiques à export_pdf.py —
#  seule la géométrie des armatures provient des couches)
# ============================================================
DIR_KEYS = ("x", "y")
FACES_DIR = ("inf_x", "sup_x", "inf_y", "sup_y")


def _gm(values, key_new, key_old, default):
    """Lecture avec repli v1 : la clé v2 d'abord, sinon l'ancienne."""
    if key_new in values:
        return values[key_new]
    return values.get(key_old, default)


def _layers_geometry_v2(values, did, sid, which):
    """Géométrie d'une face-direction, avec repli v1 : si les clés v2
    (« inf_x ») sont absentes mais que les clés v1 (« inf ») existent,
    la direction X reprend l'ancien contenu — même règle que la
    migration de dalle.py."""
    if which.endswith("_x") and KS(f"ncouches_{which}", did, sid) not in values:
        old = which[:-2]
        if KS(f"ncouches_{old}", did, sid) in values or \
                KS(f"arm_type_{old}_c1", did, sid) in values:
            return _layers_geometry(values, did, sid, old)
    return _layers_geometry(values, did, sid, which)


def _auto_dist_couche_v2(values, did, sid, which, i):
    if which.endswith("_x") and KS(f"arm_type_{which}_c{i}", did, sid) not in values:
        old = which[:-2]
        if KS(f"arm_type_{old}_c{i}", did, sid) in values:
            return _auto_dist_couche(values, did, sid, old, i)
    return _auto_dist_couche(values, did, sid, which, i)


def _compute_section(values, beton_data, did, sid):
    beton = str(_g(values, KD("beton", did), "C30/37"))
    if beton not in beton_data:
        beton = list(beton_data.keys())[0]
    bd = beton_data[beton]
    fck_cube = bd["fck_cube"]
    alpha_b = bd["alpha_b"]
    fck_cyl = float(bd.get("fck", 0.8 * fck_cube) or (0.8 * fck_cube))

    fyk, mu_ref = _get_fyk(values, did)
    gamma_s = _get_gamma_s(values)
    fyd = fyk / gamma_s

    mu_key = f"mu_a{mu_ref}"
    if mu_key not in bd:
        mu_key = "mu_a500" if "mu_a500" in bd else [k for k in bd if k.startswith("mu_a")][0]
    mu_val = bd[mu_key]

    b = float(_g(values, KD("b", did), 100))
    h = float(_g(values, KD("h", did), 20))
    enrob_beton = float(_g(values, KD("enrobage_beton", did), 3.0) or 3.0)

    fctm = 0.30 * (fck_cyl ** (2.0 / 3.0)) if fck_cyl > 0 else 0.0
    As_min_ec = 0.26 * fctm / fyk * b * h * 1e2
    As_min_plancher = 0.0013 * b * h * 1e2
    As_min_base = max(As_min_ec, As_min_plancher)
    As_max = 0.04 * b * h * 1e2

    # --- une passe par direction : mêmes expressions que la v1, le
    #     critère 0,25·Aₛ,req vise la face opposée de la MÊME direction ---
    dirs = {}
    for dk in DIR_KEYS:
        geo_inf = _layers_geometry_v2(values, did, sid, f"inf_{dk}")
        geo_sup = _layers_geometry_v2(values, did, sid, f"sup_{dk}")

        d_inf = h - geo_inf["e_cdg"]
        d_sup = h - geo_sup["e_cdg"]
        d_calc_inf = max(d_inf, 0.1)
        d_calc_sup = max(d_sup, 0.1)
        geom_inf_ok = d_inf > 0
        geom_sup_ok = d_sup > 0

        M_inf = float(_gm(values, KS(f"M{dk}_inf", did, sid),
                          KS("M_inf", did, sid) if dk == "x" else "", 0.0) or 0.0)
        M_sup = float(_gm(values, KS(f"M{dk}_sup", did, sid),
                          KS("M_sup", did, sid) if dk == "x" else "", 0.0) or 0.0)

        As_req_inf = (M_inf * 1e6) / (fyd * 0.9 * d_calc_inf * 10) if M_inf > 0 else 0.0
        As_req_sup = (M_sup * 1e6) / (fyd * 0.9 * d_calc_sup * 10) if M_sup > 0 else 0.0
        As_min_inf = max(As_min_base, 0.25 * As_req_sup)
        As_min_sup = max(As_min_base, 0.25 * As_req_inf)

        As_inf = geo_inf["As"]; As_sup = geo_sup["As"]
        etat_inf = "ok" if (geom_inf_ok and As_inf >= max(As_req_inf, As_min_inf) and As_inf <= As_max) else "nok"
        etat_sup = "ok" if (geom_sup_ok and As_sup >= max(As_req_sup, As_min_sup) and As_sup <= As_max) else "nok"

        dirs[dk] = {
            "M_inf": M_inf, "M_sup": M_sup,
            "geo_inf": geo_inf, "geo_sup": geo_sup,
            "di": d_inf, "ds": d_sup,
            "As_req_inf": As_req_inf, "As_req_sup": As_req_sup,
            "As_min_inf": As_min_inf, "As_min_sup": As_min_sup,
            "As_inf": As_inf, "As_sup": As_sup,
            "etat_inf": etat_inf, "etat_sup": etat_sup,
            "geom_inf_ok": geom_inf_ok, "geom_sup_ok": geom_sup_ok,
        }

    # Direction principale : CHOIX de l'utilisateur (défaut Y — v2.1)
    principale = "x" if str(_g(values, KD("dir_principale", did), "Y")).upper() == "X" else "y"

    V = float(_g(values, KS("V", did, sid), 0.0) or 0.0)

    familles = [(dirs["x"]["M_inf"], dirs["x"]["geo_inf"]["e_cdg"]),
                (dirs["x"]["M_sup"], dirs["x"]["geo_sup"]["e_cdg"]),
                (dirs["y"]["M_inf"], dirs["y"]["geo_inf"]["e_cdg"]),
                (dirs["y"]["M_sup"], dirs["y"]["geo_sup"]["e_cdg"])]
    M_max = max(m for m, _ in familles)
    e_cdg_gov = next(e for m, e in familles if m == M_max)
    hmin = math.sqrt((M_max * 1e6) / (alpha_b * b * 10 * mu_val)) / 10 if M_max > 0 else 0.0
    h_min_dalle = hmin + e_cdg_gov
    etat_h = "ok" if (h_min_dalle <= h) else "nok"

    etat_inf = "nok" if "nok" in (dirs["x"]["etat_inf"], dirs["y"]["etat_inf"]) else "ok"
    etat_sup = "nok" if "nok" in (dirs["x"]["etat_sup"], dirs["y"]["etat_sup"]) else "ok"

    tau_1 = 0.016 * fck_cube / 1.05
    tau_2 = 0.032 * fck_cube / 1.05
    tau_4 = 0.064 * fck_cube / 1.05

    def build_shear(Vx):
        """v2.1 : une dalle ne reçoit pas d'étriers — la vérification se
        réduit à τ ≤ τ_adm,I (seuil « pas besoin d'étriers » existant)."""
        if Vx <= 0:
            return None
        tau = Vx * 1e3 / (0.75 * b * h * 100)
        etat_tau = "ok" if tau <= tau_1 else "nok"
        return {"tau": tau, "tau_adm": tau_1, "etat_tau": etat_tau, "V": Vx}

    shear = build_shear(V)

    states = [etat_h, etat_inf, etat_sup]
    if shear:
        states.append(shear["etat_tau"])
    etat_global = "nok" if any(s == "nok" for s in states) else ("warn" if any(s == "warn" for s in states) else "ok")

    return {
        "beton": beton, "fck": fck_cyl, "fck_cube": fck_cube, "alpha_b": alpha_b, "fctm": fctm,
        "fyk": fyk, "fyd": fyd, "gamma_s": gamma_s, "mu_ref": mu_ref, "mu": mu_val,
        "b": b, "h": h, "enrob_beton": enrob_beton,
        # PRÉDALLE (module predalle.py) : épaisseur de la peau préfabriquée
        # en partie basse (cm). 0 = dalle homogène, rien ne change.
        "h_pre": float(_g(values, KD("h_pre", did), 0) or 0),
        "dirs": dirs, "principale": principale,
        "V": V,
        "M_max": M_max, "hmin": hmin, "etat_h": etat_h,
        "e_cdg_gov": e_cdg_gov, "h_min_dalle": h_min_dalle,
        "As_min_ec": As_min_ec, "As_min_plancher": As_min_plancher, "As_max": As_max,
        "etat_inf": etat_inf, "etat_sup": etat_sup,
        "shear": shear, "etat_global": etat_global,
    }


# ============================================================
#  COUPE DE DALLE (bande b×h, treillis + renforts, cotations réelles)
# ============================================================
class SlabDrawing(Flowable):
    """Coupe d'une bande de dalle : béton hachuré, treillis inférieur /
    supérieur (points = fils porteurs, trait fin = fils de répartition),
    barres de renfort intercalées, cotes b et h réelles, légende.
    L'épaisseur est exagérée si le dessin à l'échelle exacte devient
    illisible (les cotations donnent toujours les vraies dimensions)."""

    LEG_LH = 9.5  # interligne de la légende

    def __init__(self, R, width, height, pal):
        super().__init__()
        self.R = R
        self.width = width; self.height = height; self.pal = pal

    def wrap(self, aw, ah):
        return (self.width, self.height)

    def _dash_axis(self, c, x1, y1, x2, y2):
        c.saveState()
        c.setStrokeColor(self.pal["axis"]); c.setLineWidth(0.4)
        c.setDash([6, 2, 1.5, 2])
        c.line(x1, y1, x2, y2)
        c.restoreState()

    def _legend_rows(self):
        """[(texte, couleur pastille | None)] : inf puis sup, + enrobage."""
        R = self.R
        rows = []
        for which, geo in (("inf", R["geo_inf"]), ("sup", R["geo_sup"])):
            suf = "inf." if which == "inf" else "sup."
            for cch in geo["couches"]:
                col, _bd = LIT_COLORS[(cch["i"] - 1) % len(LIT_COLORS)]
                if cch["i"] == 1:
                    base = "Treillis" if cch["typ"] == "Treillis" else "Barres"
                    rows.append((f"{base} {suf} : {cch['valeur']}", col))
                else:
                    rows.append((f"Renfort {suf} {cch['i'] - 1} : {cch['valeur']}", col))
        rows.append((f"Enrobage : {fn(R['enrob_beton'], 1)} cm", None))
        return rows

    def draw(self):
        c = self.canv; R = self.R; P = self.pal
        b_cm = float(R["b"]); h_cm = float(R["h"])
        enrob = float(R.get("enrob_beton", 3.0))
        geo_inf = R["geo_inf"]; geo_sup = R["geo_sup"]

        rows = self._legend_rows()
        n_lines = (len(rows) + 1) // 2  # légende sur deux colonnes
        leg_h = n_lines * self.LEG_LH + 6

        pad_l, pad_t, pad_r, pad_b = 30, 10, 8, 20
        aw = self.width - pad_l - pad_r
        zone_h = self.height - pad_t - pad_b - leg_h

        b_mm, h_mm = b_cm * 10.0, h_cm * 10.0
        sc_x = aw / b_mm
        sh = h_mm * sc_x
        # Épaisseur exagérée si l'échelle exacte rend la coupe illisible
        # (les cotes portent les vraies dimensions).
        if sh < 42.0:
            sh = min(42.0, zone_h)
        if sh > zone_h:
            sh = zone_h
        sw = aw
        sc_y = sh / h_mm
        x0 = pad_l
        y0 = leg_h + pad_b + max(0.0, (zone_h - sh) / 2.0)

        c.saveState()
        # béton + hachures (mêmes styles que la coupe Poutre)
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

        # ---- Couches d'armatures ----
        def draw_couche(cch, y_cm_from_bottom):
            col, bdc = LIT_COLORS[(cch["i"] - 1) % len(LIT_COLORS)]
            yy = y0 + (y_cm_from_bottom * 10.0) * sc_y
            d_mm = float(cch["d"]); e_sp = float(cch["esp"]) or 100.0
            r = max(1.5, d_mm * sc_x / 2.0)
            inset_mm = enrob * 10.0 + d_mm / 2.0
            span_av = max(0.0, b_mm - 2.0 * inset_mm)
            n = max(1, int(span_av // e_sp) + 1) if e_sp > 0 else 1
            span = (n - 1) * e_sp
            xs_mm = [(b_mm - span) / 2.0 + k * e_sp for k in range(n)]
            # renfort : décalé d'une demi-maille pour s'intercaler entre
            # les fils du treillis de base (pose réelle sur chantier)
            if cch["i"] > 1:
                shift = e_sp / 2.0
                if xs_mm and xs_mm[-1] + shift > b_mm - inset_mm:
                    xs_mm = xs_mm[:-1]
                xs_mm = [x + shift for x in xs_mm]
            if not xs_mm:
                xs_mm = [b_mm / 2.0]
            # treillis : trait fin continu = fils de répartition
            if cch["typ"] == "Treillis" and len(xs_mm) > 1:
                c.setStrokeColor(bdc); c.setLineWidth(max(0.6, cch["d"] * sc_x * 0.5))
                c.line(x0 + xs_mm[0] * sc_x, yy, x0 + xs_mm[-1] * sc_x, yy)
            # fils porteurs / barres : points
            c.setFillColor(col); c.setStrokeColor(bdc); c.setLineWidth(0.5)
            for xm in xs_mm:
                c.circle(x0 + xm * sc_x, yy, r, stroke=1, fill=1)
            return yy

        for cch in geo_inf["couches"]:
            draw_couche(cch, cch["e"])                    # e = depuis le bas
        for cch in geo_sup["couches"]:
            draw_couche(cch, h_cm - cch["e"])             # e = depuis le haut

        # centres de gravité (croix) — uniquement si plusieurs couches
        def cdg_cross(y_cm_from_bottom):
            yy = y0 + (y_cm_from_bottom * 10.0) * sc_y
            cx = x0 + sw / 2.0
            rr = 3.2
            c.setStrokeColor(INK); c.setLineWidth(0.9)
            c.line(cx - rr, yy, cx + rr, yy)
            c.line(cx, yy - rr, cx, yy + rr)

        if geo_inf["nc"] > 1 and geo_inf["As"] > 0:
            cdg_cross(geo_inf["e_cdg"])
        if geo_sup["nc"] > 1 and geo_sup["As"] > 0:
            cdg_cross(h_cm - geo_sup["e_cdg"])

        # ---- Cotes b / h (vraies dimensions) ----
        c.setStrokeColor(P["dim"]); c.setFillColor(P["dim"]); c.setLineWidth(0.6); c.setFont("Helvetica", 7.5)
        yb = y0 - 10
        c.setDash(); c.line(x0, yb, x0 + sw, yb)
        for xx2 in (x0, x0 + sw):
            c.line(xx2, yb - 2.5, xx2, yb + 2.5)
        c.drawCentredString(x0 + sw / 2, yb - 9, f"b = {fn(b_cm,0)} cm")
        xl = x0 - 13
        c.line(xl, y0, xl, y0 + sh)
        for yy2 in (y0, y0 + sh):
            c.line(xl - 2.5, yy2, xl + 2.5, yy2)
        c.saveState(); c.translate(xl - 3, y0 + sh / 2); c.rotate(90)
        c.drawCentredString(0, 0, f"h = {fn(h_cm,0)} cm"); c.restoreState()

        # ---- Légende (sous la coupe, deux colonnes) ----
        c.setFont("Helvetica", 7.4)
        col_w = (self.width - pad_l) / 2.0
        for j, (lab, colr) in enumerate(rows):
            colonne = j // n_lines
            ligne = j % n_lines
            lx = pad_l + colonne * col_w
            yy = leg_h - 8 - ligne * self.LEG_LH
            if colr is not None:
                c.setFillColor(colr); c.circle(lx + 2, yy + 2.4, 2.2, stroke=0, fill=1)
                c.setFillColor(P["txt"])
                c.drawString(lx + 8, yy, lab)
            else:
                c.setFillColor(MUTE)
                c.drawString(lx + 8, yy, lab)

        c.restoreState()


# ============================================================
#  RÉCAP SECTION : caractéristiques (gauche) + coupe (droite)
#  — mêmes styles / mêmes proportions que la note Poutre
# ============================================================
def carac(R, cw):
    def sub(t):
        return [Paragraph(t, ST["subt"]), Paragraph("", ST["cell"])]
    def kv(k, vv):
        return [Paragraph(k, ST["cell"]), Paragraph(str(vv), ST["cellb"])]
    rows = [sub("DIMENSIONS"),
            kv("Largeur de bande b", f"{fn(R['b'],0)} cm"), kv("Épaisseur h", f"{fn(R['h'],0)} cm"),
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


def recap(R, cw):
    half = cw * 0.44; gap = 14; rw = cw - half - gap
    left = carac(R, half)
    draw = SlabDrawing(R, rw, 214, PAL)
    rcell = Table([[Paragraph('<font color="%s">COUPE DE DALLE</font>' % MUTE.hexval(), ST["subt"])], [draw]], colWidths=[rw])
    rcell.setStyle(TableStyle([("LEFTPADDING", (0, 0), (0, 0), 28), ("LEFTPADDING", (0, 1), (0, 1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 2), ("TOPPADDING", (0, 1), (0, 1), 0), ("ALIGN", (0, 1), (0, 1), "CENTER")]))
    lay = Table([[left, "", rcell]], colWidths=[half, gap, rw])
    lay.setStyle(TableStyle([("VALIGN", (0, 0), (0, 0), "TOP"), ("VALIGN", (2, 0), (2, 0), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return lay


# ============================================================
#  BLOCS DE VÉRIFICATION (mêmes gabarits que la note Poutre)
# ============================================================
def b_haut(R, cw):
    iw = cw - 24
    app = Formula(Row([_t("h", sub="u,min"), _t(" = "),
        Sqrt(Row([Frac(Row(Row(sci_tokens(R['M_max'] * 1e6)).items),
                       Row([_t(f"{fn(R['alpha_b'],2)} · {fn(R['b']*10,0)} · {fn(R['mu'],4)}")]))]), INK),
        _t("  =  "), nb(f"{fn(R['hmin'],1)} cm")]))
    # « d₁ » : distance du parement au c.d.g. des aciers de la face
    # dimensionnante (enrobage mécanique) — même vocabulaire que Poutre
    hminp = Formula(Row([
        _t("h", sub="u,min"), _t(" + d", sub="1"), _t(" = "),
        _t(f"{fn(R['hmin'],1)} + {fn(R['e_cdg_gov'],1)}  =  "),
        nb(f"{fn(R['h_min_dalle'],1)} cm")]))
    body = [fline("Hauteur utile minimale", app, iw),
            Spacer(1, 2),
            fline("Hauteur minimale de la dalle", hminp, iw),
            Spacer(1, 7), HR(iw, HAIR, 0.5), Spacer(1, 7),
            reslines([("Hauteur minimale de la dalle", "h<sub>min</sub>", f"{fn(R['h_min_dalle'],1)} cm"),
                      ("Épaisseur de la dalle", "h", f"{fn(R['h'],0)} cm")], iw),
            Spacer(1, 5)]
    ok = R["etat_h"] == "ok"
    left = (f"Épaisseur de la dalle : {fn(R['h'],0)} cm "
            f"{'≥' if ok else '&lt;'} hauteur minimale de la dalle : {fn(R['h_min_dalle'],1)} cm")
    body.append(conclu(R["etat_h"], iw, left, ok=ok))
    return block("1.", "Vérification de la hauteur", R["etat_h"], body, cw)


def b_arm(R, cw, which):
    iw = cw - 24
    if which == "inf":
        title = "Armatures inférieures"; M = R["M_inf"]; Ar = R["As_req_inf"]; geo = R["geo_inf"]; d = R["di"]; et = R["etat_inf"]; nn = "2."; As_min = R["As_min_inf"]
    else:
        title = "Armatures supérieures"; M = R["M_sup"]; Ar = R["As_req_sup"]; geo = R["geo_sup"]; d = R["ds"]; et = R["etat_sup"]; nn = "3."; As_min = R["As_min_sup"]

    nc = geo["nc"]
    suffix_note = f"   (c.d.g. de {nc} couches)" if nc > 1 else ""
    dlit = Formula(Row([_t("d", sub="u"), _t(f" = {fn(R['h'],0)} − {fn(geo['e_cdg'],1)} = "), nb(f"{fn(d,1)} cm"),
                        _t(suffix_note, color=MUTE, size=8.5)]))

    app = Formula(Row([_t("A", sub="s,req"), _t(" = "),
        Frac(Row(Row(sci_tokens(M * 1e6)).items), Row([_t(f"{fn(R['fyd'],1)} · 0,9 · {fn(d*10,0)}")])),
        _t("  =  "), txt(f"{fn(Ar,0)} mm", font="Helvetica-Bold", sup="2")]))

    # 'On prend' : couches + section par mètre + section sur la bande
    choix = f"{geo['detail']} ({fn(geo['As_pm'],0)} mm{s2()}/m)"

    # _asmin_formula (note Poutre) lit As_req_inf/sup, As_min_inf/sup,
    # fctm, fyk, b, h — mêmes clés dans R : réutilisation directe.
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
                      ("On prend", "", choix),
                      ("Acier fourni (bande)", "A<sub>s</sub>", f"{fn(geo['As'],0)} mm{s2()}")], iw),
            Spacer(1, 5)]
    ok = et == "ok"
    besoin = max(Ar, As_min)
    face_txt = "inférieure" if which == "inf" else "supérieure"
    left = (f"Section d'armature {face_txt} : {fn(geo['As'],0)} mm{s2()} "
            f"{'≥' if ok else '&lt;'} section d'armature requise : {fn(besoin,0)} mm{s2()}")
    body.append(conclu(et, iw, left, ok=ok))
    return block(nn, title, et, body, cw)


# ============================================================
#  PAGE DE GARDE
# ============================================================
def _cover(infos, dalles, values, beton_data, cw, pages):
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
    sm = [[Paragraph("<b>DALLE</b>", ST["lab"]), Paragraph("<b>SECTIONS</b>", ST["lab"]),
           Paragraph("<b>BÉTON / ACIER</b>", ST["lab"]), Paragraph("<b>ÉTAT</b>", ST["lab"]), Paragraph("<b>PAGE</b>", ST["lab"])]]
    for d in dalles:
        did = int(d["id"])
        secs = ", ".join(str(_g(values, f"meta_dal{did}_nom_{int(s['id'])}", s.get("nom", ""))) for s in d.get("sections", []))
        ss = [_compute_section(values, beton_data, did, int(s["id"]))["etat_global"] for s in d.get("sections", [])]
        eg = "nok" if "nok" in ss else ("warn" if "warn" in ss else "ok")
        pg = pages.get(did)
        sm.append([Paragraph(str(_g(values, f"meta_dalle_nom_{did}", d.get("nom", f"Dalle {did}"))), ST["cellb"]),
                   Paragraph(secs, ST["cell"]),
                   Paragraph(f"{_g(values, KD('beton', did), '—')} / B{_g(values, KD('fyk', did), '500')}", ST["cell"]),
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
def _build_story(dalles, values, beton_data, infos, cw, pages, store):
    story = _cover(infos, dalles, values, beton_data, cw, pages)
    for di, d in enumerate(dalles):
        did = int(d["id"])
        if di > 0:
            story.append(PageBreak())
        story.append(Marker(store, did))
        story.append(beam_banner(str(_g(values, f"meta_dalle_nom_{did}", d.get("nom", f"Dalle {did}"))), cw))
        story.append(Spacer(1, 10))
        sections = d.get("sections", [])
        for si, s in enumerate(sections):
            sid = int(s["id"])
            raw = str(_g(values, f"meta_dal{did}_nom_{sid}", s.get("nom", f"Section {sid}")))
            snom = raw if raw.strip().lower().startswith("section") else f"Section {raw}"
            R = _compute_section(values, beton_data, did, sid)
            # Comme dans l'application : blocs inf. ET sup. toujours présents
            blocs = [b_haut(R, cw), b_arm(R, cw, "inf"), b_arm(R, cw, "sup")]
            if R["shear"]:
                blocs.append(b_shear(R, cw))
            intro = [sec_banner(snom, cw), Spacer(1, 6), recap(R, cw), Spacer(1, 12), blocs[0]]
            story.append(KeepTogether(intro))
            for blk in blocs[1:]:
                story.append(Spacer(1, 12)); story.append(KeepTogether([blk]))
            if si < len(sections) - 1:
                story.append(Spacer(1, 16))
    return story


# ============================================================
#  API PRINCIPALE
# ============================================================
def generer_rapport_pdf(dalles, values, beton_data, infos=None, output_path=None):
    """Note de calcul Dalle : garde portrait + une planche paysage par
    section — mêmes briques ndc_pdf que la note Poutre v3.0. Signature
    et retour inchangés (appelée par dalle.py)."""
    global DERNIERS_AVERTISSEMENTS
    from datetime import datetime
    from modules.export_pdf import _style_ndc
    from ndc_pdf import data as ndc_data

    infos = infos or {}
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".pdf", prefix="note_dalle_")
        os.close(fd)

    resultats = _collecter_resultats(dalles, values, beton_data)
    doc_meta = ndc_data.construire_doc(
        infos, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    sections = ndc_data.construire_sections_dalle(resultats)

    d = _style_ndc(2).build(output_path, sections=sections, doc=doc_meta)
    if d.warnings:
        # remède prescrit par la maquette : planche à 3 colonnes
        d = _style_ndc(3).build(output_path, sections=sections, doc=doc_meta)
    d.save()
    DERNIERS_AVERTISSEMENTS = list(d.warnings)
    return output_path


# Avertissements de débordement de la dernière génération.
DERNIERS_AVERTISSEMENTS = []


def _collecter_resultats(dalles, values, beton_data):
    """Payloads NEUTRES pour ndc_pdf.data : un par section, dans l'ordre
    des planches — même motif que la note Poutre."""
    out = []
    for d in dalles:
        did = int(d["id"])
        nom_dalle = str(_g(values, f"meta_dalle_nom_{did}", d.get("nom", f"Dalle {did}")))
        for sec in d.get("sections", []):
            sid = int(sec["id"])
            raw = str(_g(values, f"meta_dal{did}_nom_{sid}", sec.get("nom", f"Section {sid}")))
            snom = raw if raw.strip().lower().startswith("section") else f"Section {raw}"
            R = _compute_section(values, beton_data, did, sid)
            out.append(dict(dalle=nom_dalle, section=snom, R=R))
    return out

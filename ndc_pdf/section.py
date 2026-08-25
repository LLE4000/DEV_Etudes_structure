"""
ndc_section.py — dessin de la coupe de section béton armé.

Dessin vectoriel complet : béton (hachures ou aplat), cadre d'effort tranchant
avec crochets, barres longitudinales à l'échelle, chaîne de cotation (b, h,
enrobage, hauteur utile d), lignes de rappel annotées, échelle et repère.

Quatre habillages : "technique", "blueprint", "plein", "minimal".
"""
import math
from reportlab.lib.colors import HexColor, Color
from .fonts import draw_text, string_width


class SectionStyle:
    def __init__(self, mode="technique", ink="#141414", accent="#B3341F",
                 muted="#8A8A8A", panel=None, hatch=None, font="Heros",
                 font_bold="Heros-Bold", mono="Mono", arrows="arrow",
                 show_hatch=True, show_d=True, label_size=6.6, dim_size=6.4,
                 concrete=None, bar="#141414", rule_w=0.9, title=None,
                 leader_side="right", concrete_texture=False, steel=False):
        self.mode = mode
        self.ink = HexColor(ink) if isinstance(ink, str) else ink
        self.accent = HexColor(accent) if isinstance(accent, str) else accent
        self.muted = HexColor(muted) if isinstance(muted, str) else muted
        self.panel = HexColor(panel) if isinstance(panel, str) else panel
        self.hatch = HexColor(hatch) if isinstance(hatch, str) else (hatch or self.muted)
        self.concrete = HexColor(concrete) if isinstance(concrete, str) else concrete
        self.bar = HexColor(bar) if isinstance(bar, str) else bar
        self.font, self.font_bold, self.mono = font, font_bold, mono
        self.arrows = arrows          # arrow | slash | dot
        self.show_hatch = show_hatch
        self.show_d = show_d
        self.label_size, self.dim_size = label_size, dim_size
        self.rule_w = rule_w
        self.title = title
        self.leader_side = leader_side
        # représentation conventionnelle du béton coupé (fond blanc +
        # hachure discrète + petits triangles et points) au lieu de l'aplat
        self.concrete_texture = concrete_texture
        # rendu « acier » des armatures : dégradé métallique sur les barres,
        # reflet cylindrique sur les étriers (au lieu de l'aplat accent)
        self.steel = steel


def _tick(c, x, y, kind, ang=0, size=2.6, color=None):
    c.saveState()
    c.translate(x, y)
    c.rotate(ang)
    if kind == "arrow":
        p = c.beginPath()
        p.moveTo(0, 0)
        p.lineTo(-size * 1.6, size * 0.55)
        p.lineTo(-size * 1.6, -size * 0.55)
        p.close()
        c.setFillColor(color)
        c.drawPath(p, stroke=0, fill=1)
    elif kind == "slash":
        c.setStrokeColor(color)
        c.setLineWidth(0.7)
        c.line(-size * 0.8, -size * 0.8, size * 0.8, size * 0.8)
    else:
        c.setFillColor(color)
        c.circle(0, 0, size * 0.42, stroke=0, fill=1)
    c.restoreState()


def _dim_h(c, x1, x2, y, label, st, color=None, ext_from=None, above=True):
    """Cotation horizontale entre x1 et x2 à l'ordonnée y."""
    col = color or st.muted
    c.setStrokeColor(col)
    c.setLineWidth(0.45)
    c.line(x1, y, x2, y)
    if ext_from is not None:
        c.setDash(1, 2)
        c.line(x1, y, x1, ext_from)
        c.line(x2, y, x2, ext_from)
        c.setDash()
    _tick(c, x1, y, st.arrows, 180 if st.arrows == "arrow" else 0, color=col)
    _tick(c, x2, y, st.arrows, 0, color=col)
    w = string_width(label, st.font, st.dim_size)
    c.setFillColor(col)
    if x2 - x1 > w + 10:
        c.setFillColor(st.panel if st.panel else HexColor("#FFFFFF"))
        c.rect((x1 + x2) / 2 - w / 2 - 2, y - 2.2, w + 4, st.dim_size + 0.6,
               stroke=0, fill=1 if st.panel is not None or True else 0)
        draw_text(c, (x1 + x2) / 2, y - 1.2, label, st.font, st.dim_size, col, "center")
    else:
        draw_text(c, x2 + 4, y - st.dim_size * 0.35, label, st.font, st.dim_size, col)


def _dim_v(c, y1, y2, x, label, st, color=None, ext_from=None):
    col = color or st.muted
    c.setStrokeColor(col)
    c.setLineWidth(0.45)
    c.line(x, y1, x, y2)
    if ext_from is not None:
        c.setDash(1, 2)
        c.line(x, y1, ext_from, y1)
        c.line(x, y2, ext_from, y2)
        c.setDash()
    _tick(c, x, y1, st.arrows, 270 if st.arrows == "arrow" else 0, color=col)
    _tick(c, x, y2, st.arrows, 90 if st.arrows == "arrow" else 0, color=col)
    c.saveState()
    c.translate(x - 2.4, (y1 + y2) / 2)
    c.rotate(90)
    w = string_width(label, st.font, st.dim_size)
    if st.panel is not None:
        c.setFillColor(st.panel)
        c.rect(-w / 2 - 2, -0.5, w + 4, st.dim_size + 0.4, stroke=0, fill=1)
    draw_text(c, 0, 0.6, label, st.font, st.dim_size, col, "center")
    c.restoreState()


def _blanc_mix(col, t):
    """Éclaircit une couleur vers le blanc (0 = inchangée, 1 = blanc)."""
    return Color(col.red + (1 - col.red) * t, col.green + (1 - col.green) * t,
                 col.blue + (1 - col.blue) * t)


def _lerp_col(a, b, t):
    """Interpolation linéaire entre deux couleurs."""
    return Color(a.red + (b.red - a.red) * t, a.green + (b.green - a.green) * t,
                 a.blue + (b.blue - a.blue) * t)


# ---- rendu « acier » des armatures : nuances d'un acier au carbone
_ACIER = dict(
    jante=HexColor("#22262D"),      # jante sombre de la barre
    fonce=HexColor("#3A404A"),      # métal à l'ombre
    moyen=HexColor("#8A919D"),      # métal courant
    clair=HexColor("#DADDE3"),      # reflet
    trait=HexColor("#343A43"),      # corps des étriers
    reflet=HexColor("#9AA2AD"),     # reflet cylindrique des étriers
)


def _metal(t):
    """Dégradé sombre -> moyen -> clair pour t dans [0 ; 1]."""
    if t < 0.5:
        return _lerp_col(_ACIER["fonce"], _ACIER["moyen"], t * 2.0)
    return _lerp_col(_ACIER["moyen"], _ACIER["clair"], (t - 0.5) * 2.0)


def _barre_acier(c, cx, cy, r):
    """Coupe d'une barre : jante sombre puis dégradé métallique décalé
    vers le reflet haut-gauche — disques concentriques, vectoriel pur."""
    c.setFillColor(_ACIER["jante"])
    c.circle(cx, cy, r, stroke=0, fill=1)
    steps = 9
    for k in range(steps):
        t = k / (steps - 1.0)
        rr = r * (0.88 - 0.68 * t)
        if rr <= 0.2:
            break
        c.setFillColor(_metal(t))
        c.circle(cx - r * 0.24 * t, cy + r * 0.24 * t, rr, stroke=0, fill=1)


def _h01(i, j, k=0.0):
    """Hachage déterministe -> [0 ; 1[ (aucun aléa : rendu reproductible)."""
    v = math.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453
    return v - math.floor(v)


def _texture_beton(c, x, y, w, h, col_grain, col_hachure, step=12.0):
    """Représentation conventionnelle du béton coupé : fond blanc, hachure
    à 45° discrète, semis irrégulier de petits granulats (triangles
    scalènes) et de points. Semis DÉTERMINISTE, sans motif de grille."""
    c.saveState()
    p = c.beginPath()
    p.rect(x, y, w, h)
    c.clipPath(p, stroke=0, fill=0)
    # hachure à 45° (convention), montante vers la droite
    c.setStrokeColor(col_hachure)
    c.setLineWidth(0.25)
    pas_h = 11.0
    n = int((w + h) / pas_h) + 2
    for i in range(-1, n):
        px = x - h + i * pas_h
        c.line(px, y, px + h, y + h)
    # granulats : positions désordonnées par hachage (pas d'alignement)
    c.setStrokeColor(col_grain)
    c.setFillColor(col_grain)
    for i in range(int(w / step) + 2):
        for j in range(int(h / step) + 2):
            cx = x + (i + 0.10 + 0.80 * _h01(i, j, 1)) * step
            cy = y + (j + 0.10 + 0.80 * _h01(i, j, 2)) * step
            u = _h01(i, j, 3)
            if u < 0.42:                    # point (sable)
                c.circle(cx, cy, 0.40 + 0.25 * _h01(i, j, 4), stroke=0, fill=1)
            elif u < 0.72:                  # petit triangle scalène (granulat)
                r = 1.1 + 1.0 * _h01(i, j, 5)
                a0 = 6.2832 * _h01(i, j, 6)
                tri = c.beginPath()
                pts = []
                for t in range(3):
                    at = a0 + t * 2.0944 + (_h01(i, j, 7 + t) - 0.5) * 0.9
                    rt = r * (0.70 + 0.45 * _h01(i, j, 10 + t))
                    pts.append((cx + rt * math.cos(at), cy + rt * math.sin(at)))
                tri.moveTo(*pts[0])
                tri.lineTo(*pts[1])
                tri.lineTo(*pts[2])
                tri.close()
                c.setLineWidth(0.32)
                c.drawPath(tri, stroke=1, fill=0)
    c.restoreState()


def _hatch(c, x, y, w, h, color, step=4.2, lw=0.28):
    c.saveState()
    p = c.beginPath()
    p.rect(x, y, w, h)
    c.clipPath(p, stroke=0, fill=0)
    c.setStrokeColor(color)
    c.setLineWidth(lw)
    n = int((w + h) / step) + 2
    for i in range(-2, n):
        c.line(x + i * step, y, x + i * step - h, y + h)
    c.restoreState()


def _leader(c, x0, y0, x1, y1, x2, st, color=None):
    """Ligne de rappel : point -> coude -> palier horizontal."""
    col = color or st.ink
    c.setStrokeColor(col)
    c.setLineWidth(0.5)
    p = c.beginPath()
    p.moveTo(x0, y0)
    p.lineTo(x1, y1)
    p.lineTo(x2, y1)
    c.drawPath(p)
    c.setFillColor(col)
    c.circle(x0, y0, 1.0, stroke=0, fill=1)


def draw_section(c, x, y, w, h, sec, st, label_w=104):
    """
    Dessine la coupe dans le rectangle (x, y, w, h).
    `sec` : dict avec b, h, enrobage (mm), barres et cadre.
    Renvoie la hauteur réellement utilisée.
    """
    b_mm, h_mm = sec["b"], sec["h"]
    cov = sec["enrobage"]
    dst = sec.get("cadre_dia", 10)
    inf, sup = sec["lit_inf"], sec["lit_sup"]

    if st.panel is not None:
        c.setFillColor(st.panel)
        c.rect(x, y, w, h, stroke=0, fill=1)

    pad_l, pad_b, pad_t = 44, 30, 30
    lab = label_w if st.leader_side == "right" else 0
    core_w = w - pad_l - lab - 12
    core_h = h - pad_b - pad_t
    s = min(core_w / b_mm, core_h / h_mm)
    bw, bh = b_mm * s, h_mm * s
    # l'ensemble coupe + bloc d'annotations est centré dans le cadre
    total = pad_l + bw + (46 + lab if st.leader_side == "right" else 0)
    ox = x + max(0, (w - total) / 2) + pad_l
    oy = y + pad_b + max(0, (core_h - bh) / 2)

    # ---- béton
    if st.concrete_texture:
        # convention de coupe : fond blanc + hachure discrète + granulats
        c.setFillColor(HexColor("#FFFFFF"))
        c.rect(ox, oy, bw, bh, stroke=0, fill=1)
        _texture_beton(c, ox, oy, bw, bh,
                       _blanc_mix(st.muted, 0.38), _blanc_mix(st.muted, 0.62))
    elif st.concrete is not None:
        c.setFillColor(st.concrete)
        c.rect(ox, oy, bw, bh, stroke=0, fill=1)
    if st.show_hatch:
        _hatch(c, ox, oy, bw, bh, st.hatch)
    c.setStrokeColor(st.ink)
    c.setLineWidth(st.rule_w)
    c.rect(ox, oy, bw, bh, stroke=1, fill=0)

    # ---- cadre(s) (étriers) avec crochets
    ci = cov * s
    sx0, sy0 = ox + ci, oy + ci
    sw, sh = bw - 2 * ci, bh - 2 * ci
    r = max(1.6, dst * s * 2.0)

    def _cadre_plein(dia, color, hooks):
        """Rectangle périmétrique d'un étrier fermé ; en mode acier, un
        second passage plus fin trace le reflet cylindrique."""
        w_tr = max(0.9, dia * s * 0.85)
        w_hk = max(0.8, dia * s * 0.8)
        passes = [(color, w_tr, w_hk)]
        if st.steel:
            passes.append((_ACIER["reflet"], w_tr * 0.34, w_hk * 0.34))
        hk = min(sw * 0.34, max(5.0, dia * s * 6.0))
        e = max(1.8, dia * s * 1.7)
        u = 0.7071
        for col, wt, wh in passes:
            c.setStrokeColor(col)
            c.setLineWidth(wt)
            c.roundRect(sx0, sy0, sw, sh, r, stroke=1, fill=0)
            if hooks:
                c.saveState()
                c.setStrokeColor(col)
                c.setLineWidth(wh)
                for k in (0, 1):
                    px = sx0 + r * 0.75 - k * e * u
                    py = sy0 + sh - r * 0.25 - k * e * u
                    p = c.beginPath()
                    p.moveTo(px, py)
                    p.lineTo(px + hk * u, py - hk * u)
                    c.drawPath(p)
                c.restoreState()

    def _teinte(i):
        """Alternance des teintes entre groupes (1er = accent)."""
        if i % 2 == 0:
            return st.accent
        a = st.accent
        return Color(a.red + (1 - a.red) * 0.45, a.green + (1 - a.green) * 0.45,
                     a.blue + (1 - a.blue) * 0.45)

    def _coul_groupe(i):
        """Couleur de base du i-e groupe fermé (acier ou palette)."""
        if st.steel:
            return _ACIER["trait"] if i % 2 == 0 else _blanc_mix(_ACIER["trait"], 0.30)
        return _teinte(i)

    # groupes positionnés (extension) : `cadres` = [{dia, brins, de, a}],
    # de/a = indices de barres du lit 1 inférieur (None = toute la largeur).
    # Sans cette clé, le rendu historique à un cadre est inchangé.
    cadres = sec.get("cadres")
    n1_ref = int((sec.get("lits_inf") or [sec["lit_inf"]])[0].get("n", 1) or 1)

    def _clampb(v, dflt):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return dflt
        return max(1, min(n1_ref, v))

    attente = []          # groupes dessinés APRÈS les barres (positions requises)
    if not cadres:
        _cadre_plein(dst, _coul_groupe(0), True)
    else:
        k_ferme = 0
        for g in cadres:
            dia = float(g.get("dia", 8) or 8)
            if int(g.get("brins", 2)) == 1:
                attente.append(("epingle", dia,
                                _clampb(g.get("de"), (n1_ref + 1) // 2),
                                _clampb(g.get("a"), _clampb(g.get("de"), (n1_ref + 1) // 2)),
                                None))
                continue
            fb = _clampb(g.get("de"), 1)
            tb = _clampb(g.get("a"), n1_ref)
            if fb > tb:
                fb, tb = tb, fb
            if fb <= 1 and tb >= n1_ref:
                _cadre_plein(dia, _coul_groupe(k_ferme), k_ferme == 0)
            else:
                attente.append(("partiel", dia, fb, tb, _coul_groupe(k_ferme)))
            k_ferme += 1

    # ---- barres longitudinales
    def bars(layer, bottom=True):
        n, dia = layer["n"], layer["dia"]
        rr = dia * s / 2
        y0 = (oy + ci + dst * s + rr) if bottom else (oy + bh - ci - dst * s - rr)
        x0 = sx0 + dst * s + rr
        x1 = sx0 + sw - dst * s - rr
        xs = [x0] if n == 1 else [x0 + i * (x1 - x0) / (n - 1) for i in range(n)]
        for bx in xs:
            if st.steel:
                _barre_acier(c, bx, y0, max(1.4, rr))
            else:
                c.setFillColor(st.bar)
                c.setStrokeColor(st.ink if st.mode != "blueprint" else st.bar)
                c.setLineWidth(0.4)
                c.circle(bx, y0, max(1.4, rr), stroke=1, fill=1)
        return xs, y0

    # ---- extension multi-lits : listes `lits_inf` / `lits_sup`, chaque lit
    # avec sa position d'axe réelle e (mm depuis le parement). Sans ces
    # clés, le rendu historique à un lit par face est strictement inchangé.
    lits_inf = sec.get("lits_inf")
    lits_sup = sec.get("lits_sup")
    multi = bool(lits_inf or lits_sup)

    def bars_face(lits, bottom=True):
        """Dessine chaque lit à sa position réelle ; à défaut de e fourni,
        empile vers l'intérieur avec 25 mm d'espace libre."""
        out = []
        prev_e = prev_d = None
        for lyr in lits:
            n, dia = max(1, int(lyr["n"])), float(lyr["dia"])
            rr = dia * s / 2.0
            e_mm = lyr.get("e")
            if e_mm is None:
                e_mm = (cov + dst + dia / 2.0) if prev_e is None else \
                       (prev_e + prev_d / 2.0 + 25.0 + dia / 2.0)
            y0 = (oy + e_mm * s) if bottom else (oy + bh - e_mm * s)
            x0 = sx0 + dst * s + rr
            x1 = sx0 + sw - dst * s - rr
            xs = [(x0 + x1) / 2.0] if n == 1 else \
                 [x0 + i * (x1 - x0) / (n - 1) for i in range(n)]
            for bx in xs:
                c.setFillColor(st.bar)
                c.setStrokeColor(st.ink if st.mode != "blueprint" else st.bar)
                c.setLineWidth(0.4)
                c.circle(bx, y0, max(1.4, rr), stroke=1, fill=1)
            out.append((xs, y0))
            prev_e, prev_d = e_mm, dia
        return out

    if multi:
        f_inf = bars_face(lits_inf or [dict(n=inf["n"], dia=inf["dia"])], True)
        f_sup = bars_face(lits_sup or [dict(n=sup["n"], dia=sup["dia"])], False)
        xs_inf, y_inf = f_inf[0]
        xs_sup, y_sup = f_sup[0]
    else:
        xs_inf, y_inf = bars(inf, True)
        xs_sup, y_sup = bars(sup, False)

    # ---- groupes positionnés en attente : étriers partiels, épingles,
    # agrafes — tracés au contact des barres du lit 1 inférieur
    r1 = float((sec.get("lits_inf") or [sec["lit_inf"]])[0]["dia"]) * s / 2.0
    for kind, dia, fb, tb, col in attente:
        w_st = max(0.8, dia * s * 0.85)
        if fb > tb:
            fb, tb = tb, fb
        if kind == "partiel":
            m = r1 + w_st / 2.0
            x_l = xs_inf[fb - 1] - m
            x_r = xs_inf[tb - 1] + m
            passes = [(col, w_st)]
            if st.steel:
                passes.append((_ACIER["reflet"], w_st * 0.34))
            for cc, wt in passes:
                c.setStrokeColor(cc)
                c.setLineWidth(wt)
                c.roundRect(x_l, sy0, x_r - x_l, sh, max(2.0, 1.5 * dia * s),
                            stroke=1, fill=0)
        else:                       # épingle (1 brin) ou agrafe entre barres
            # teinte claire : l'épingle au ras d'une barre reste lisible
            # même quand elle longe le montant du cadre principal
            base = _blanc_mix(_ACIER["trait"], 0.18) if st.steel else _teinte(1)
            passes = [(base, w_st)]
            if st.steel:
                passes.append((_ACIER["reflet"], w_st * 0.34))
            hk = max(3.0, dia * s * 3.0)
            for cc, wt in passes:
                c.setStrokeColor(cc)
                c.setLineWidth(wt)
                if fb == tb:
                    xb = xs_inf[fb - 1] + r1 + w_st / 2.0 + 0.8
                    c.line(xb, sy0, xb, sy0 + sh)
                    c.line(xb, sy0 + sh, xb - hk, sy0 + sh - hk)
                    c.line(xb, sy0, xb - hk, sy0 + hk)
                else:
                    xa, xb2 = xs_inf[fb - 1], xs_inf[tb - 1]
                    c.line(xa, y_inf, xb2, y_inf)
                    c.line(xa, y_inf, xa, y_inf + hk)
                    c.line(xb2, y_inf, xb2, y_inf + hk)

    # ---- armatures de peau : sur les deux faces latérales, à l'intérieur
    # des étriers, aux positions calculées par le moteur
    peau = sec.get("peau")
    peau_anchor = None
    if peau and peau.get("ys"):
        r_t = max(1.2, float(peau["dia"]) * s / 2.0)
        w_main = max(0.9, dst * s * 0.85)
        x_pl = ox + ci + w_main + r_t
        x_pr = ox + bw - ci - w_main - r_t
        for y_mm in peau["ys"]:
            yy = oy + y_mm * s
            for xc in (x_pl, x_pr):
                if st.steel:
                    _barre_acier(c, xc, yy, r_t)
                else:
                    c.setFillColor(st.bar)
                    c.setStrokeColor(st.ink if st.mode != "blueprint" else st.bar)
                    c.setLineWidth(0.4)
                    c.circle(xc, yy, r_t, stroke=1, fill=1)
        peau_anchor = (x_pr, oy + max(peau["ys"]) * s)

    # ---- hauteur utile d
    # ---- cotations principales (chaîne empilée à gauche)
    _dim_h(c, ox, ox + bw, oy - 15, sec["b_label"], st, ext_from=oy - 3)
    if st.show_d and sec.get("d"):
        yd = oy + bh - sec["d"] * s
        c.setStrokeColor(st.muted)
        c.setLineWidth(0.4)
        c.setDash(2, 2)
        c.line(ox - 14, yd, ox + bw - 2, yd)
        c.setDash()
        _dim_v(c, yd, oy + bh, ox - 13, sec["d_label"], st, st.ink)
        if sec.get("d1_label"):
            # segment complémentaire de la chaîne : d₁ = h − d (enrobage
            # mécanique de la face inférieure)
            _dim_v(c, oy, yd, ox - 13, sec["d1_label"], st, st.ink)
        _dim_v(c, oy, oy + bh, ox - 28, sec["h_label"], st, ext_from=ox - 3)
    else:
        _dim_v(c, oy, oy + bh, ox - 15, sec["h_label"], st, ext_from=ox - 3)

    # ---- enrobage
    c.setStrokeColor(st.muted)
    c.setLineWidth(0.4)
    c.line(ox, oy + bh + 6, ox, oy + bh + 12)
    c.line(ox + ci, oy + bh + 6, ox + ci, oy + bh + 12)
    _dim_h(c, ox, ox + ci, oy + bh + 9, sec["c_label"], st, st.muted)

    # ---- lignes de rappel annotées
    if st.leader_side == "right":
        lx = ox + bw + 46
        tx = lx + 4

        def _annot(xsl, yl, yy, txt, sub):
            _leader(c, xsl[-1], yl, lx - 14, yy, lx, st, st.ink)
            draw_text(c, tx, yy + 1.6, txt, st.font_bold, st.label_size, st.ink)
            if sub:
                draw_text(c, tx, yy - st.label_size + 0.4, sub, st.font,
                          st.label_size - 0.6, st.muted)

        if multi:
            # un libellé par lit ; les étiquettes s'écartent du parement
            # d'au moins 15 pt pour ne jamais se chevaucher
            labs_sup = sec.get("labs_sup") or [(sec["lab_sup"], sec.get("lab_sup2"))]
            labs_inf = sec.get("labs_inf") or [(sec["lab_inf"], sec.get("lab_inf2"))]
            slot = None
            for (xsl, yl), (txt, sub) in zip(f_sup, labs_sup):
                yy = yl + 16 if slot is None else min(yl + 16, slot - 15)
                _annot(xsl, yl, yy, txt, sub)
                slot = yy
            slot = None
            for (xsl, yl), (txt, sub) in zip(f_inf, labs_inf):
                yy = yl - 16 if slot is None else max(yl - 16, slot + 15)
                _annot(xsl, yl, yy, txt, sub)
                slot = yy
        else:
            for (xsl, yl, txt, sub) in (
                (xs_sup, y_sup, sec["lab_sup"], sec.get("lab_sup2")),
                (xs_inf, y_inf, sec["lab_inf"], sec.get("lab_inf2")),
            ):
                yy = yl + (16 if yl > oy + bh / 2 else -16)
                _annot(xsl, yl, yy, txt, sub)
        # cadre(s)
        yy = oy + bh * 0.52
        _leader(c, sx0 + sw, yy, lx - 14, yy, lx, st, st.accent)
        labs_c = sec.get("labs_cadre")
        if labs_c:
            # une ligne par groupe (étriers, épingles), empilées sous la
            # ligne de rappel — 1er groupe en accent, suivants en encre
            yline = yy
            for i, lab in enumerate(labs_c):
                draw_text(c, tx, yline + 1.6, lab,
                          st.font_bold if i == 0 else st.font,
                          st.label_size - (0 if i == 0 else 0.4),
                          st.accent if i == 0 else st.ink)
                yline -= st.label_size + 2.2
            y_bas_cadre = yline + 1.6
        else:
            draw_text(c, tx, yy + 1.6, sec["lab_cadre"], st.font_bold, st.label_size, st.accent)
            if sec.get("lab_cadre2"):
                draw_text(c, tx, yy - st.label_size + 0.4, sec["lab_cadre2"], st.font,
                          st.label_size - 0.6, st.muted)
            y_bas_cadre = yy - st.label_size + 0.4

        # armature de peau : ligne de rappel dédiée, sous le bloc du cadre
        if peau_anchor is not None:
            yp = y_bas_cadre - 14
            _leader(c, peau_anchor[0], peau_anchor[1], lx - 14, yp, lx, st, st.muted)
            draw_text(c, tx, yp + 1.6, sec["peau"].get("label", "Armature de peau"),
                      st.font_bold, st.label_size, st.ink)

    # ---- repère / échelle
    if st.title:
        draw_text(c, x + 2, y + h - 9, st.title, st.font_bold, st.label_size + 0.6, st.ink)
    ech = f"éch. 1:{max(1, round(72.0 / (s * 25.4))):d}"
    draw_text(c, x + w - 2, y + 3, ech, st.font, st.dim_size - 0.4, st.muted, "right")
    return h

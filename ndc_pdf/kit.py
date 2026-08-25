r"""
kit.py — briques de mise en page.

Une Frame est une colonne rectangulaire avec son propre curseur vertical :
c'est ce qui permet des architectures très différentes (une colonne, deux
colonnes, rail latéral, paysage) sans dupliquer la logique de contenu.
"""
from reportlab.pdfgen import canvas as _canvas
from reportlab.lib.colors import HexColor, Color

from .fonts import register_all, draw_text, string_width
from . import mathx as M
from .section import draw_section


def hx(v):
    return HexColor(v) if isinstance(v, str) else v


def mix(a, b, t):
    a, b = hx(a), hx(b)
    return Color(a.red + (b.red - a.red) * t,
                 a.green + (b.green - a.green) * t,
                 a.blue + (b.blue - a.blue) * t)


class Frame:
    """Colonne de composition : x, largeur, haut et bas, curseur y."""

    def __init__(self, x, top, w, bottom):
        self.x, self.top, self.w, self.bottom = x, top, w, bottom
        self.y = top
        self.min_y = top

    @property
    def x1(self):
        return self.x + self.w

    def room(self):
        return self.y - self.bottom

    def fits(self, h):
        return self.y - h >= self.bottom

    def down(self, h):
        self.y -= h
        if self.y < self.min_y:
            self.min_y = self.y
        return self.y

    def reset(self, top=None):
        self.y = self.top if top is None else top
        return self

    def overflow(self):
        return max(0.0, self.bottom - self.min_y)


class Doc:
    def __init__(self, path, page, title="Note de calcul", author=""):
        register_all()
        self.c = _canvas.Canvas(path, pagesize=page)
        self.c.setTitle(title)
        if author:
            self.c.setAuthor(author)
        self.W, self.H = page
        self.n = 0

    # --------------------------------------------------------- primitives
    def t(self, x, y, s, f, size, col, align="left", track=0.0):
        return draw_text(self.c, x, y, s, f, size, hx(col), align, charspace=track)

    def w(self, s, f, size):
        return string_width(s, f, size)

    def line(self, x1, y1, x2, y2, col, lw=0.5, dash=None):
        self.c.setStrokeColor(hx(col))
        self.c.setLineWidth(lw)
        if dash:
            self.c.setDash(*dash)
        self.c.line(x1, y1, x2, y2)
        if dash:
            self.c.setDash()

    def box(self, x, y, w, h, fill=None, stroke=None, lw=0.5, r=0):
        c = self.c
        if fill is not None:
            c.setFillColor(hx(fill))
        if stroke is not None:
            c.setStrokeColor(hx(stroke))
            c.setLineWidth(lw)
        if r:
            c.roundRect(x, y, w, h, r, stroke=1 if stroke else 0,
                        fill=1 if fill else 0)
        else:
            c.rect(x, y, w, h, stroke=1 if stroke else 0, fill=1 if fill else 0)

    def wrap(self, s, f, size, width):
        words, lines, cur = s.split(), [], ""
        for wd in words:
            t = (cur + " " + wd).strip()
            if string_width(t, f, size) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return lines

    def para(self, fr, s, f, size, col, lead=1.32, indent=0):
        for ln in self.wrap(s, f, size, fr.w - indent):
            fr.down(size * lead)
            self.t(fr.x + indent, fr.y, ln, f, size, col)
        return fr.y

    def fit(self, s, f, size, wmax):
        if string_width(s, f, size) <= wmax:
            return s
        while s and string_width(s + "…", f, size) > wmax:
            s = s[:-1]
        return s + "…"

    def new_page(self, size=None, chrome=None):
        """Nouvelle page. `size` permet de mélanger portrait et paysage."""
        if self.n:
            self.c.showPage()
        self.n += 1
        if size is not None:
            self.c.setPageSize(size)
            self.W, self.H = size
        if chrome:
            chrome(self)

    def save(self):
        self.c.save()
        return self.n


# ------------------------------------------------------------------ blocs

def sym(d, x, y, s, ms, size, align="left"):
    """Dessine un symbole mathématique isolé."""
    if not s:
        return 0
    n = M.parse(s)
    w = M.width(n, ms, size)
    if align == "right":
        x -= w
    M.draw(d.c, n, x, y, ms, size)
    return w


def formula(d, fr, label, expr, ms, size, col_lab, f_lab, s_lab,
            lead=1.0, indent=0, label_mode="above", label_w=0.34, tag=None,
            min_size=6.0):
    """
    Compose une formule dans la colonne. label_mode :
    above (libellé au-dessus), left (libellé en colonne), none.
    Réduit la taille si la formule dépasse la largeur disponible.
    """
    node = M.parse(expr)
    xm = fr.x + indent
    avail = fr.w - indent
    if label_mode == "left":
        xm = fr.x + fr.w * label_w
        avail = fr.x1 - xm
    b = M.measure(node, ms, size)
    if b.w > avail:
        size = max(min_size, size * avail / b.w)
        b = M.measure(node, ms, size)
    gap_lab = 0.0
    if label and label_mode == "above":
        fr.down(s_lab * 1.25)
        d.t(fr.x + indent, fr.y, label, f_lab, s_lab, col_lab)
        gap_lab = s_lab * 0.45          # descendantes du libellé + respiration
    fr.down(b.a + 2 * lead + gap_lab)
    if label and label_mode == "left":
        d.t(fr.x, fr.y, d.fit(label, f_lab, s_lab, fr.w * label_w - 8),
            f_lab, s_lab, col_lab)
    M.draw(d.c, node, xm, fr.y, ms, size)
    if tag:
        d.t(fr.x1, fr.y, tag, f_lab, s_lab, col_lab, "right")
    fr.down(b.d + 3 * lead)
    return fr.y


def kv_rows(d, fr, rows, ms, f_lab, f_val, size, col_ink, col_mut, col_rule,
            mode="dots", sym_col=0.52, lead=1.55):
    """mode : dots (points de conduite) | flat | band"""
    for i, (lab, sy, val, unit) in enumerate(rows):
        fr.down(size * lead)
        y = fr.y
        if mode == "band" and i % 2 == 0:
            d.box(fr.x - 3, y - size * 0.42, fr.w + 6, size * lead * 0.98,
                  fill=col_rule)
        v = f"{val} {unit}".strip()
        wv = d.w(v, f_val, size)
        sw = 0.0
        if sy:
            sw = M.width(M.parse(sy), ms, size)
        # le symbole recule si la valeur, calée à droite, viendrait le toucher
        xs = min(fr.x + fr.w * sym_col, fr.x1 - wv - sw - 8)
        xs = max(xs, fr.x)
        if lab:
            d.t(fr.x, y, d.fit(lab, f_lab, size, xs - fr.x - 6),
                f_lab, size, col_ink)
        sym(d, xs, y, sy, ms, size)
        d.t(fr.x1, y, v, f_val, size, col_ink, "right")
        if mode == "dots":
            x0 = fr.x + (d.w(lab, f_lab, size) + 5 if lab else 0)
            x1 = xs - 5
            if x1 > x0 + 6:
                d.line(x0, y + 1.6, x1, y + 1.6, col_rule, 0.35, (0.5, 2.0))
            x0b = xs + (sw + 5 if sy else 0)
            x1b = fr.x1 - wv - 5
            if x1b > x0b + 6:
                d.line(x0b, y + 1.6, x1b, y + 1.6, col_rule, 0.35, (0.5, 2.0))
    return fr.y


def gauge(d, x, y, w, ratio, col_ok, col_ko, col_bg, h=4.0, r=None):
    """Jauge de rapport ; le repère marque la valeur 1,0."""
    ok = ratio <= 1.0
    d.box(x, y, w, h, fill=col_bg, r=h / 2)
    span = 4.0                      # échelle de la jauge : 0 à 4×
    f = min(ratio, span) / span
    d.box(x, y, max(3.0, w * f), h, fill=col_ok if ok else col_ko, r=h / 2)
    xm = x + w / span
    d.line(xm, y - 1.6, xm, y + h + 1.6, mix(col_bg, "#000000", 0.45), 0.6)


def chip(d, x, y, txt, f, size, fg, bg, pad=7, h=None, r=None, align="left"):
    w = d.w(txt, f, size) + 2 * pad
    h = h or size + 6.5
    if align == "right":
        x -= w
    d.box(x, y, w, h, fill=bg, r=(h / 2 if r is None else r))
    d.t(x + w / 2, y + h * 0.30, txt, f, size, fg, "center")
    return w


def coupe(d, x, y, w, h, style, label_w=100):
    draw_section(d.c, x, y, w, h, __import__(
        "ndc_pdf.data", fromlist=["COUPE"]).COUPE, style, label_w=label_w)

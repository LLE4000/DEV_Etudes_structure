r"""
ndc_math.py — composition typographique des formules d'ingénierie.

Mini-syntaxe (proche LaTeX) :
    \frac{a}{b}          fraction avec barre
    \sqrt{a}             racine
    \max{a ; b ; c}      accolade avec lignes empilées, alignées sur le "="
    \min{a ; b}          idem
    \u{N/mm^2}           unité (romain, jamais italique)
    \t{texte}            texte courant (romain)
    x^2  x_ck  A_s,req   exposants / indices
    (...)               parenthèses à hauteur variable
    \cdot \le \ge \times \pm \phi \approx \sum \deg
Conventions : une lettre seule = variable (italique), plusieurs lettres = romain
(min, max, cos, cm...), chiffres = romain.
"""
from reportlab.pdfbase import pdfmetrics
from .fonts import split_runs

# ---------------------------------------------------------------- noeuds


class Node:
    pass


class Run(Node):
    __slots__ = ("text", "role")

    def __init__(self, text, role="num"):
        self.text, self.role = text, role


class Space(Node):
    __slots__ = ("em",)

    def __init__(self, em=0.28):
        self.em = em


class Seq(Node):
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items


class Script(Node):
    __slots__ = ("base", "sup", "sub")

    def __init__(self, base, sup=None, sub=None):
        self.base, self.sup, self.sub = base, sup, sub


class Frac(Node):
    __slots__ = ("num", "den", "_inl")

    def __init__(self, num, den):
        self.num, self.den = num, den
        self._inl = None


class Sqrt(Node):
    __slots__ = ("body",)

    def __init__(self, body):
        self.body = body


class Paren(Node):
    __slots__ = ("body", "left", "right")

    def __init__(self, body, left="(", right=")"):
        self.body, self.left, self.right = body, left, right


class Cases(Node):
    """Accolade + lignes empilées, alignées sur le premier '=' de chaque ligne."""
    __slots__ = ("rows", "label")

    def __init__(self, rows, label="max"):
        self.rows, self.label = rows, label


# ---------------------------------------------------------------- parser

SYMBOLS = {
    "cdot": "·", "times": "×", "le": "≤", "ge": "≥", "ne": "≠", "pm": "±",
    "phi": "Ø", "approx": "≈", "sum": "Σ", "deg": "°", "to": "→", "minus": "−",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
    "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phiG": "φ", "eps": "ε", "inf": "∞",
}
OPS = set("=+−-<>≤≥≠±·×/;→≈")
DIGITS = set("0123456789")


class _P:
    def __init__(self, s):
        self.s, self.i = s, 0

    def eof(self):
        return self.i >= len(self.s)

    def peek(self):
        return self.s[self.i] if not self.eof() else ""

    def group(self):
        """Lit un {...} (accolades équilibrées) et renvoie la chaîne interne."""
        while not self.eof() and self.s[self.i] == " ":
            self.i += 1
        if self.peek() != "{":
            # groupe implicite : un seul caractère
            ch = self.peek()
            self.i += 1
            return ch
        depth, self.i, start = 1, self.i + 1, self.i + 1
        while not self.eof() and depth:
            if self.s[self.i] == "{":
                depth += 1
            elif self.s[self.i] == "}":
                depth -= 1
                if depth == 0:
                    break
            self.i += 1
        out = self.s[start:self.i]
        self.i += 1
        return out

    def parse(self, stop=""):
        items = []
        while not self.eof():
            ch = self.s[self.i]
            if ch in stop:
                break
            if ch == "\\":
                self.i += 1
                name = ""
                while not self.eof() and (self.s[self.i].isalpha()):
                    name += self.s[self.i]
                    self.i += 1
                items.append(self._command(name))
            elif ch == "^" or ch == "_":
                self.i += 1
                g = self.group()
                sub_node = parse(g, script=True)
                if items and isinstance(items[-1], (Run, Script, Paren, Seq)):
                    base = items.pop()
                    if isinstance(base, Script):
                        if ch == "^":
                            base.sup = sub_node
                        else:
                            base.sub = sub_node
                        items.append(base)
                    else:
                        items.append(Script(base, sup=sub_node if ch == "^" else None,
                                            sub=sub_node if ch == "_" else None))
                else:
                    items.append(sub_node)
            elif ch == "(":
                self.i += 1
                inner = self.parse(stop=")")
                if self.peek() == ")":
                    self.i += 1
                items.append(Paren(inner))
            elif ch == " ":
                self.i += 1
                items.append(Space(0.30))
            elif ch in OPS:
                self.i += 1
                items.append(Run(ch, "op"))
            elif ch in DIGITS or ch == ",":
                t = ""
                while not self.eof() and (self.s[self.i] in DIGITS or self.s[self.i] in ",."):
                    t += self.s[self.i]
                    self.i += 1
                items.append(Run(t, "num"))
            elif ch.isalpha():
                t = ""
                while not self.eof() and self.s[self.i].isalpha():
                    t += self.s[self.i]
                    self.i += 1
                items.append(Run(t, "var" if len(t) == 1 else "rom"))
            else:
                self.i += 1
                items.append(Run(ch, "num"))
        return Seq(items)

    def _command(self, name):
        if name == "frac":
            return Frac(parse(self.group()), parse(self.group()))
        if name == "sqrt":
            return Sqrt(parse(self.group()))
        if name in ("max", "min", "cases"):
            body = self.group()
            rows = [parse(r) for r in _split_top(body, ";")]
            return Cases(rows, "" if name == "cases" else name)
        if name in ("u", "unit"):
            return Seq([Space(0.16), parse(self.group(), roman=True)])
        if name in ("res", "r"):
            n = parse(self.group())
            _mark_res(n)
            return n
        if name in ("t", "text", "rm"):
            return Run(self.group(), "rom")
        if name == "it":
            return Run(self.group(), "var")
        if name == "q":  # espace fine
            return Space(0.18)
        if name in SYMBOLS:
            if name in ("cdot", "times", "le", "ge", "ne", "pm", "approx", "to", "minus"):
                role = "op"
            elif name in ("phi", "deg", "sum", "inf", "Delta"):
                role = "sym"
            else:
                role = "var"          # lettres grecques minuscules : italique
            return Run(SYMBOLS[name], role)
        return Run(name, "rom")


def _split_top(s, sep):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def parse(s, roman=False, script=False):
    node = _P(s).parse()
    if roman:
        _force_roman(node)
    return node


def _mark_res(n):
    if isinstance(n, Run):
        n.role = "res" if n.role in ("num", "rom", "var") else n.role
    elif isinstance(n, Seq):
        for it in n.items:
            _mark_res(it)
    elif isinstance(n, Script):
        for x in (n.base, n.sup, n.sub):
            if x is not None:
                _mark_res(x)
    elif isinstance(n, (Frac, Paren, Sqrt)):
        for a in ("num", "den", "body"):
            x = getattr(n, a, None)
            if x is not None:
                _mark_res(x)


def _force_roman(n):
    if isinstance(n, Run):
        if n.role == "var":
            n.role = "rom"
    elif isinstance(n, Seq):
        for it in n.items:
            _force_roman(it)
    elif isinstance(n, Script):
        for x in (n.base, n.sup, n.sub):
            if x is not None:
                _force_roman(x)
    elif isinstance(n, (Frac, Paren, Sqrt)):
        for a in ("num", "den", "body"):
            x = getattr(n, a, None)
            if x is not None:
                _force_roman(x)


M = parse  # alias court


def _has_top_op(n, chars):
    if isinstance(n, Seq):
        return any(isinstance(i, Run) and i.role == "op" and i.text in chars
                   for i in n.items)
    return False


def _inline_frac(n, st):
    """Transforme a/b en écriture en ligne, avec parenthèses si nécessaire."""
    cached = getattr(n, "_inl", None)
    if cached is not None:
        return cached
    num = Paren(n.num) if _has_top_op(n.num, "+−-") else n.num
    den = Paren(n.den) if _has_top_op(n.den, "+−-·×/") else n.den
    out = Seq([num, Space(0.18), Run("/", "op"), Space(0.18), den])
    try:
        n._inl = out
    except AttributeError:
        pass
    return out


# ---------------------------------------------------------------- rendu

class MathStyle:
    """Polices et couleurs employées pour composer les formules."""

    def __init__(self, var="Heros-Italic", rom="Heros", num="Heros",
                 op="Heros", sym="Sym", ink=None, accent=None, rule=None,
                 fallback="Sym", frac="inline", cases="rule", res=None,
                 res_color=None, op_color=None, tight=1.0, cases_align=True):
        self.f = {"var": var, "rom": rom, "num": num, "op": op, "sym": sym,
                  "res": res or rom}
        self.ink = ink
        self.accent = accent or ink
        self.rule = rule or ink
        self.fallback = fallback
        self.frac = frac        # inline (a / b) ou stacked (barre)
        self.cases = cases      # rule | brace
        self.cases_align = cases_align   # aligner les lignes sur le « = »
        self.res_color = res_color or accent or ink
        self.op_color = op_color
        self.tight = tight      # facteur d'espacement horizontal

    def font(self, role):
        return self.f.get(role, self.f["num"])


class Box:
    __slots__ = ("w", "a", "d")

    def __init__(self, w, a, d):
        self.w, self.a, self.d = w, a, d   # largeur, au-dessus, en-dessous de la ligne de base


def _sw(text, font, size, fb):
    return sum(pdfmetrics.stringWidth(t, f, size)
               for t, f in split_runs(text, font, fb))


def measure(n, st, size):
    if isinstance(n, Run):
        f = st.font(n.role)
        w = _sw(n.text, f, size, st.fallback)
        if n.role == "op" and n.text in ("·", "×"):
            w = max(w, size * 0.34)
        return Box(w, size * 0.72, size * 0.22)
    if isinstance(n, Space):
        return Box(size * n.em * st.tight, 0, 0)
    if isinstance(n, Seq):
        w = a = d = 0.0
        for it in n.items:
            b = measure(it, st, size)
            w += b.w
            a, d = max(a, b.a), max(d, b.d)
        return Box(w, a or size * 0.72, d or size * 0.22)
    if isinstance(n, Script):
        b = measure(n.base, st, size)
        ss = max(size * 0.70, 5.2)
        w = b.w
        a, d = b.a, b.d
        wsup = wsub = 0.0
        if n.sup is not None:
            bs = measure(n.sup, st, ss)
            wsup = bs.w
            a = max(a, size * 0.44 + bs.a)
        if n.sub is not None:
            bs = measure(n.sub, st, ss)
            wsub = bs.w
            d = max(d, size * 0.20 + bs.d)
        return Box(w + max(wsup, wsub) + size * 0.04, a, d)
    if isinstance(n, Frac):
        if st.frac == "inline":
            return measure(_inline_frac(n, st), st, size)
        s2 = size * 0.98
        bn, bd = measure(n.num, st, s2), measure(n.den, st, s2)
        w = max(bn.w, bd.w) + size * 0.5
        axis = size * 0.30
        a = axis + size * 0.26 + bn.a + bn.d * 0.15
        d = -axis + size * 0.30 + bd.a + bd.d
        return Box(w, a, d)
    if isinstance(n, Sqrt):
        b = measure(n.body, st, size)
        return Box(b.w + size * 0.85, b.a + size * 0.22, b.d)
    if isinstance(n, Paren):
        b = measure(n.body, st, size)
        return Box(b.w + size * 0.62, max(b.a, size * 0.75), max(b.d, size * 0.24))
    if isinstance(n, Cases):
        rows = [_case_parts(r, st, size) for r in n.rows]
        lw = max(r[2] for r in rows)
        rw = max(r[3] for r in rows)
        lh = size * (1.62 if st.cases == "brace" else 1.30)
        total = lh * len(rows)
        lab = _sw(n.label, st.font("rom"), size, st.fallback) if n.label else 0.0
        w = lab + size * 0.28 + size * (0.55 if st.cases == "brace" else 0.40) + lw + rw
        return Box(w, total / 2 + size * 0.28, total / 2 - size * 0.05)
    return Box(0, 0, 0)


def _case_parts(row, st, size):
    """Coupe une ligne d'accolade au premier '=' pour l'alignement."""
    items = row.items if isinstance(row, Seq) else [row]
    if not st.cases_align:
        b = measure(Seq(items), st, size)
        return Seq(items), Seq([]), b.w, 0.0
    idx = None
    for k, it in enumerate(items):
        if isinstance(it, Run) and it.text == "=":
            idx = k
            break
    if idx is None:
        left, right = Seq(items), Seq([])
    else:
        left, right = Seq(items[:idx]), Seq(items[idx:])
    bl, br = measure(left, st, size), measure(right, st, size)
    return left, right, bl.w, br.w


def draw(c, n, x, y, st, size):
    """Dessine le noeud, x = bord gauche, y = ligne de base. Renvoie la largeur."""
    if isinstance(n, Run):
        f = st.font(n.role)
        w = _sw(n.text, f, size, st.fallback)
        adv = w
        if n.role == "op" and n.text in ("·", "×"):
            adv = max(w, size * 0.34)
            x += (adv - w) / 2.0          # glyphe centré dans son avance
        col = st.ink
        if n.role == "res":
            col = st.res_color
        elif n.role == "op" and st.op_color is not None:
            col = st.op_color
        c.setFillColor(col)
        to = c.beginText(x, y)
        to.setCharSpace(0)
        for t, ff in split_runs(n.text, f, st.fallback):
            to.setFont(ff, size)
            to.textOut(t)
        c.drawText(to)
        return adv
    if isinstance(n, Space):
        return size * n.em * st.tight
    if isinstance(n, Seq):
        dx = 0.0
        for it in n.items:
            dx += draw(c, it, x + dx, y, st, size)
        return dx
    if isinstance(n, Script):
        b = measure(n.base, st, size)
        draw(c, n.base, x, y, st, size)
        ss = max(size * 0.70, 5.2)
        w2 = 0.0
        if n.sup is not None:
            w2 = max(w2, draw(c, n.sup, x + b.w + size * 0.04, y + size * 0.44, st, ss))
        if n.sub is not None:
            w2 = max(w2, draw(c, n.sub, x + b.w + size * 0.04, y - size * 0.20, st, ss))
        return b.w + w2 + size * 0.04
    if isinstance(n, Frac):
        if st.frac == "inline":
            return draw(c, _inline_frac(n, st), x, y, st, size)
        s2 = size * 0.98
        bn, bd = measure(n.num, st, s2), measure(n.den, st, s2)
        w = max(bn.w, bd.w) + size * 0.5
        axis = y + size * 0.30
        draw(c, n.num, x + (w - bn.w) / 2, axis + size * 0.26 + bn.d * 0.15, st, s2)
        draw(c, n.den, x + (w - bd.w) / 2, axis - size * 0.30 - bd.a, st, s2)
        c.setStrokeColor(st.rule)
        c.setLineWidth(max(0.45, size * 0.052))
        c.line(x + size * 0.06, axis, x + w - size * 0.06, axis)
        return w
    if isinstance(n, Sqrt):
        b = measure(n.body, st, size)
        h = b.a + b.d + size * 0.18
        x0, y0 = x, y - b.d - size * 0.10
        c.setStrokeColor(st.rule)
        c.setLineWidth(max(0.5, size * 0.055))
        p = c.beginPath()
        p.moveTo(x0, y0 + h * 0.45)
        p.lineTo(x0 + size * 0.20, y0 + h * 0.30)
        p.lineTo(x0 + size * 0.42, y0 + h + size * 0.04)
        p.lineTo(x0 + b.w + size * 0.80, y0 + h + size * 0.04)
        c.drawPath(p)
        draw(c, n.body, x + size * 0.62, y, st, size)
        return b.w + size * 0.85
    if isinstance(n, Paren):
        b = measure(n.body, st, size)
        top = max(b.a, size * 0.75)
        bot = max(b.d, size * 0.24)
        c.setStrokeColor(st.ink)
        c.setLineWidth(max(0.5, size * 0.055))
        for side in (0, 1):
            xx = x + size * 0.24 if side == 0 else x + b.w + size * 0.40
            k = 1 if side == 0 else -1
            p = c.beginPath()
            p.moveTo(xx + k * size * 0.12, y + top)
            p.curveTo(xx - k * size * 0.10, y + top * 0.45,
                      xx - k * size * 0.10, y - bot * 0.45,
                      xx + k * size * 0.12, y - bot)
            c.drawPath(p)
        draw(c, n.body, x + size * 0.34, y, st, size)
        return b.w + size * 0.62
    if isinstance(n, Cases):
        rows = [_case_parts(r, st, size) for r in n.rows]
        lw = max(r[2] for r in rows)
        lh = size * 1.62
        total = lh * len(rows)
        dx = 0.0
        if n.label:
            c.setFillColor(st.ink)
            c.setFont(st.font("rom"), size)
            c.drawString(x, y, n.label)
            dx = _sw(n.label, st.font("rom"), size, st.fallback) + size * 0.24
        # accolade
        bx = x + dx
        top, bot = y + total / 2 + size * 0.22, y - total / 2 + size * 0.30
        if st.cases != "brace":
            c.setFillColor(st.accent)
            c.rect(bx + size * 0.06, bot - size * 0.10, max(0.7, size * 0.075),
                   top - bot + size * 0.10, stroke=0, fill=1)
            x0 = bx + size * 0.40
            for k, (left, right, wl, wr) in enumerate(rows):
                yy = y + total / 2 - lh * (k + 0.5) + size * 0.30
                draw(c, left, x0, yy, st, size)
                draw(c, right, x0 + lw + size * 0.12, yy, st, size)
            return dx + size * 0.40 + lw + max(r[3] for r in rows) + size * 0.12
        c.setStrokeColor(st.rule)
        c.setLineWidth(max(0.5, size * 0.06))
        mid = (top + bot) / 2
        r = min(size * 0.34, (top - bot) / 6)
        p = c.beginPath()
        p.moveTo(bx + size * 0.34, top)
        p.curveTo(bx + size * 0.10, top, bx + size * 0.18, top - r, bx + size * 0.18, mid + r)
        p.curveTo(bx + size * 0.18, mid + r * 0.2, bx + size * 0.10, mid, bx + size * 0.02, mid)
        p.curveTo(bx + size * 0.10, mid, bx + size * 0.18, mid - r * 0.2, bx + size * 0.18, mid - r)
        p.curveTo(bx + size * 0.18, bot + r, bx + size * 0.10, bot, bx + size * 0.34, bot)
        c.drawPath(p)
        x0 = bx + size * 0.55
        for k, (left, right, wl, wr) in enumerate(rows):
            yy = y + total / 2 - lh * (k + 0.5) + size * 0.30
            draw(c, left, x0, yy, st, size)
            draw(c, right, x0 + lw + size * 0.12, yy, st, size)
        return dx + size * 0.55 + lw + max(r[3] for r in rows) + size * 0.12
    return 0.0


def width(n, st, size):
    return measure(n, st, size).w


def height(n, st, size):
    b = measure(n, st, size)
    return b.a + b.d

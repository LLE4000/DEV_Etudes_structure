"""
ndc_fonts.py — enregistrement des polices et repli automatique sur glyphes manquants.

Toutes les polices sont libres de redistribution (OFL / GUST / Apache).
Elles sont attendues dans le dossier `fonts/` situé à côté de ce module,
ou dans le dossier pointé par la variable d'environnement NDC_FONT_DIR.
"""
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

FONT_DIR = os.environ.get(
    "NDC_FONT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"),
)

# nom logique -> fichier ttf
FONTS = {
    "Pagella": "Pagella-Regular.ttf",
    "Pagella-Bold": "Pagella-Bold.ttf",
    "Pagella-Italic": "Pagella-Italic.ttf",
    "Pagella-BoldItalic": "Pagella-BoldItalic.ttf",
    "Heros": "Heros-Regular.ttf",
    "Heros-Bold": "Heros-Bold.ttf",
    "Heros-Italic": "Heros-Italic.ttf",
    "HerosCn": "HerosCn-Regular.ttf",
    "HerosCn-Bold": "HerosCn-Bold.ttf",
    "Adventor": "Adventor-Regular.ttf",
    "Adventor-Bold": "Adventor-Bold.ttf",
    "Adventor-Italic": "Adventor-Italic.ttf",
    "Carlito": "Carlito-Regular.ttf",
    "Carlito-Bold": "Carlito-Bold.ttf",
    "Carlito-Italic": "Carlito-Italic.ttf",
    "Caladea": "Caladea-Regular.ttf",
    "Caladea-Bold": "Caladea-Bold.ttf",
    "Caladea-Italic": "Caladea-Italic.ttf",
    "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
    "Cond": "DejaVuSansCond.ttf",
    "Cond-Bold": "DejaVuSansCond-Bold.ttf",
    "Cond-Oblique": "Cond-Oblique.ttf",
    "Poppins-Light": "Poppins-Light.ttf",
    "Poppins": "Poppins-Regular.ttf",
    "Poppins-Medium": "Poppins-Medium.ttf",
    "Poppins-Bold": "Poppins-Bold.ttf",
    "Poppins-Italic": "Poppins-Italic.ttf",
    "Lora": "Lora-Regular.ttf",
    "Lora-SemiBold": "Lora-SemiBold.ttf",
    "Lora-Bold": "Lora-Bold.ttf",
    "Lora-Italic": "Lora-Italic.ttf",
    "LMRoman": "LMRoman-Regular.ttf",
    "LMRoman-Bold": "LMRoman-Bold.ttf",
    "LMRoman-Italic": "LMRoman-Italic.ttf",
    "LMRoman-BoldItalic": "LMRoman-BoldItalic.ttf",
    "LMSans": "LMSans-Regular.ttf",
    "LMSans-Bold": "LMSans-Bold.ttf",
    "LMMono": "LMMono-Regular.ttf",
    "LMMono-Bold": "LMMono-Bold.ttf",
    "Cursor": "Cursor-Regular.ttf",
    "Cursor-Bold": "Cursor-Bold.ttf",
    "Mono": "DejaVuMono.ttf",
    "Mono-Bold": "DejaVuMono-Bold.ttf",
    "Sym": "DejaVuSans.ttf",           # police de repli universelle
    "Sym-Bold": "DejaVuSans-Bold.ttf",
    "SymSerif": "DejaVuSerif.ttf",
}

FAMILIES = [
    ("Pagella", "Pagella", "Pagella-Bold", "Pagella-Italic", "Pagella-BoldItalic"),
    ("Heros", "Heros", "Heros-Bold", "Heros-Italic", "Heros-Bold"),
    ("HerosCn", "HerosCn", "HerosCn-Bold", "HerosCn", "HerosCn-Bold"),
    ("Adventor", "Adventor", "Adventor-Bold", "Adventor-Italic", "Adventor-Bold"),
    ("Carlito", "Carlito", "Carlito-Bold", "Carlito-Italic", "Carlito-Bold"),
    ("Caladea", "Caladea", "Caladea-Bold", "Caladea-Italic", "Caladea-Bold"),
    ("Poppins", "Poppins", "Poppins-Bold", "Poppins-Italic", "Poppins-Bold"),
    ("Lora", "Lora", "Lora-Bold", "Lora-Italic", "Lora-Bold"),
    ("LMRoman", "LMRoman", "LMRoman-Bold", "LMRoman-Italic", "LMRoman-BoldItalic"),
    ("LMSans", "LMSans", "LMSans-Bold", "LMSans", "LMSans-Bold"),
    ("Cursor", "Cursor", "Cursor-Bold", "Cursor", "Cursor-Bold"),
    ("Mono", "Mono", "Mono-Bold", "Mono", "Mono-Bold"),
    ("Sym", "Sym", "Sym-Bold", "Sym", "Sym-Bold"),
]

_cmaps = {}
_registered = False

# Les polices issues de TeX dessinent « < » et « > » comme des chevrons :
# ces deux caractères sont pris dans la police de repli.
_TEX_FONTS = {
    "Pagella", "Pagella-Bold", "Pagella-Italic", "Pagella-BoldItalic",
    "Heros", "Heros-Bold", "Heros-Italic", "HerosCn", "HerosCn-Bold",
    "Adventor", "Adventor-Bold", "Adventor-Italic",
    "Cursor", "Cursor-Bold", "LMRoman", "LMRoman-Bold", "LMRoman-Italic",
    "LMRoman-BoldItalic", "LMSans", "LMSans-Bold", "LMMono", "LMMono-Bold",
}
_FORCE_FALLBACK = "<>"


def register_all():
    """Enregistre toutes les polices disponibles. Idempotent."""
    global _registered
    if _registered:
        return
    from fontTools.ttLib import TTFont as _FT
    for name, fn in FONTS.items():
        path = os.path.join(FONT_DIR, fn)
        if not os.path.exists(path):
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            _cmaps[name] = set(_FT(path).getBestCmap().keys())
        except Exception:
            pass
    for fam, r, b, i, bi in FAMILIES:
        if r in _cmaps:
            pdfmetrics.registerFontFamily(fam, normal=r, bold=b, italic=i, boldItalic=bi)
            addMapping(fam, 0, 0, r)
            addMapping(fam, 1, 0, b)
            addMapping(fam, 0, 1, i)
            addMapping(fam, 1, 1, bi)
    _registered = True


def has_glyph(font, ch):
    cm = _cmaps.get(font)
    return True if cm is None else (ord(ch) in cm)


def split_runs(text, font, fallback="Sym"):
    """Découpe `text` en (fragment, police) pour gérer les glyphes absents."""
    if not text:
        return []
    out, cur, cur_font = [], "", None
    tex = font in _TEX_FONTS
    for ch in text:
        if tex and ch in _FORCE_FALLBACK:
            f = fallback
        else:
            f = font if has_glyph(font, ch) else fallback
        if f != cur_font:
            if cur:
                out.append((cur, cur_font))
            cur, cur_font = ch, f
        else:
            cur += ch
    if cur:
        out.append((cur, cur_font))
    return out


def string_width(text, font, size, fallback="Sym"):
    return sum(pdfmetrics.stringWidth(t, f, size) for t, f in split_runs(text, font, fallback))


def draw_text(c, x, y, text, font, size, color=None, align="left", fallback="Sym",
              charspace=0.0):
    """Écrit du texte en gérant le repli de glyphes. align: left|center|right."""
    runs = split_runs(text, font, fallback)
    w = sum(pdfmetrics.stringWidth(t, f, size) for t, f in runs)
    if charspace:
        w += charspace * max(0, len(text) - 1)
    if align == "center":
        x -= w / 2.0
    elif align == "right":
        x -= w
    if color is not None:
        c.setFillColor(color)
    to = c.beginText(x, y)
    # Tc fait partie de l'état de texte PDF et persiste hors BT/ET :
    # on le réinitialise systématiquement pour ne pas le laisser fuir.
    to.setCharSpace(charspace or 0)
    for t, f in runs:
        to.setFont(f, size)
        to.textOut(t)
    c.drawText(to)
    return w

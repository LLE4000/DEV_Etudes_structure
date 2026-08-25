r"""
build.py — génère les dix notes puis le catalogue (index + notes à la suite).

    python -m ndc_pdf.build sortie
"""
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from .fonts import register_all, draw_text, string_width
from .styles import STYLES
from . import data as D


def build_all(outdir="."):
    os.makedirs(outdir, exist_ok=True)
    out = []
    for s in STYLES:
        path = os.path.join(outdir, f"NDC_{s.key}.pdf")
        s.build(path, garde=False).save()
        out.append((s, path))
    return out


def _index(path, entries):
    register_all()
    W, H = A4
    c = canvas.Canvas(path, pagesize=A4)
    ink, mut, acc, rule = "#101418", "#6E7781", "#C2410C", "#E3E6E9"
    draw_text(c, 52, H - 62, D.DOC["bureau"].upper(), "Heros-Bold", 8, mut,
              charspace=2.0)
    draw_text(c, 52, H - 92, "Note de calcul — dix palettes", "Poppins-Bold", 22, ink)
    draw_text(c, 52, H - 112,
              "Mise en page unique. Seule la palette change : panneau, "
              "accent, teinte du béton.", "Carlito", 10.5, mut)
    from reportlab.lib.colors import HexColor
    c.setStrokeColor(HexColor(ink))
    c.setLineWidth(1.2)
    c.line(52, H - 126, W - 52, H - 126)
    y = H - 158
    for s, pg, npages in entries:
        c.setFillColor(HexColor(acc))
        c.rect(52, y - 4, 2.6, 30, stroke=0, fill=1)
        num = s.key.split("_")[0]
        draw_text(c, 64, y + 12, num, "Poppins-Bold", 15, "#D5D9DE")
        draw_text(c, 94, y + 13, s.name, "Poppins-Medium", 11.5, ink)
        draw_text(c, 94, y, s.pitch, "Carlito", 8.6, mut)
        draw_text(c, W - 52 - 60, y + 6, f"page {pg}", "Carlito", 8.6, ink, "right")
        d = 20
        c.setFillColor(HexColor(s.panel))
        c.setStrokeColor(HexColor("#D8DBE0"))
        c.setLineWidth(0.5)
        c.rect(W - 52 - 2 * d - 5, y + 1, d, d, stroke=1, fill=1)
        c.setFillColor(HexColor(s.acc))
        c.rect(W - 52 - d, y + 1, d, d, stroke=0, fill=1)
        c.setStrokeColor(HexColor(rule))
        c.setLineWidth(0.5)
        c.line(52, y - 12, W - 52, y - 12)
        y -= 46
    draw_text(c, 52, 52, "Contenu strictement identique à la note exportée : "
              "mêmes valeurs, mêmes libellés, mêmes verdicts. Fonds clairs, "
              "prêts à imprimer.", "Carlito", 8.2, mut)
    c.save()


def catalogue(outdir=".", dst=None):
    from pypdf import PdfReader, PdfWriter
    files = build_all(outdir)
    entries, n = [], 2
    for s, path in files:
        k = len(PdfReader(path).pages)
        entries.append((s, n, k))
        n += k
    idx = os.path.join(outdir, "_index.pdf")
    _index(idx, entries)
    w = PdfWriter()
    for p in PdfReader(idx).pages:
        w.add_page(p)
    for _, path in files:
        for p in PdfReader(path).pages:
            w.add_page(p)
    dst = dst or os.path.join(outdir, "CATALOGUE.pdf")
    with open(dst, "wb") as fh:
        w.write(fh)
    os.remove(idx)
    return dst, [p for _, p in files]


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    cat, files = catalogue(out)
    print(cat)
    for f in files:
        print(f)

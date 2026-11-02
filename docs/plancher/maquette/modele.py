# -*- coding: utf-8 -*-
"""Schéma du modèle de calcul « Charges sur plancher », en quatre planches
dessinées au trait (SVG), rendues par Chromium. Valeurs d'illustration."""
import sys, pathlib, math

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

BLEU, NOIR, GRIS, ROUGE = "#1d4ed8", "#111827", "#6b7280", "#b91c1c"
DEFS = '''<defs>
<pattern id="hA" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" stroke="#555" stroke-width="1"/></pattern>
<pattern id="hC" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#1d4ed8" stroke-width="0.9"/></pattern>
<pattern id="hB" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="#9ca3af" stroke-width="0.7"/></pattern>
<marker id="tk" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 2 8 L 8 2" stroke="#1d4ed8" stroke-width="1.4"/></marker>
<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b91c1c"/></marker>
<marker id="arN" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker>
<marker id="arG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker>
</defs>'''

def svg_open(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" style="font-family:Inter,sans-serif;font-size:11px">{DEFS}']
def cote_h(A, x1, x2, y, txt, dy=-5, fs=11):
    A(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{BLEU}" stroke-width="1" marker-start="url(#tk)" marker-end="url(#tk)"/>')
    A(f'<text x="{(x1+x2)/2}" y="{y+dy}" font-size="{fs}" font-weight="700" fill="{BLEU}" text-anchor="middle">{txt}</text>')
def cote_v(A, y1, y2, x, txt, dx=-6, fs=11):
    A(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{BLEU}" stroke-width="1" marker-start="url(#tk)" marker-end="url(#tk)"/>')
    A(f'<text transform="translate({x+dx} {(y1+y2)/2}) rotate(-90)" font-size="{fs}" font-weight="700" fill="{BLEU}" text-anchor="middle">{txt}</text>')
def ext(A, x1, y1, x2, y2):
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{BLEU}" stroke-width="0.6"/>')
def txt(A, x, y, t, fs=11, anchor="start", fill=NOIR, w=None, it=False):
    A(f'<text x="{x}" y="{y}" font-size="{fs}" text-anchor="{anchor}" fill="{fill}" {"font-weight=700" if w else ""} {"font-style=italic" if it else ""}>{t}</text>')
def hourdis_coupe(A, x0, y0, b=300, h=50, nv=5, fill="#e5e7eb"):
    """Coupe d'un hourdis (b × h px), alvéoles ovales."""
    A(f'<rect x="{x0}" y="{y0}" width="{b}" height="{h}" fill="{fill}" stroke="{NOIR}" stroke-width="1.2"/>')
    for i in range(nv):
        A(f'<ellipse cx="{x0 + b*(i+0.5)/nv}" cy="{y0+h/2}" rx="{b/nv*0.36}" ry="{h*0.33}" fill="white" stroke="{NOIR}" stroke-width="1"/>')

# ================================================================ (a) coupe par un pied
def planche_a():
    A = svg_open(640, 400).append
    sc = 0.25                      # px / mm
    x0, yt = 90, 246               # coin haut-gauche du hourdis chargé (b = 1200 mm → 300 px)
    hourdis_coupe(lambda s: A(s), x0, yt, 300, 50)
    hourdis_coupe(lambda s: A(s), x0 + 300 + 6, yt, 120, 50, nv=2)       # voisin (partiel)
    A(f'<rect x="{x0+300}" y="{yt}" width="6" height="50" fill="url(#hB)" stroke="{NOIR}" stroke-width="0.8"/>')
    txt(A, x0 + 303, yt + 64, "joint", 9, "middle", GRIS)
    # plateau 600 × 20 (t), pied 300 × 60, centré sur x = 600 mm
    xc = x0 + 150
    pw, pt = 600 * sc, 20 * sc + 1
    A(f'<rect x="{xc-pw/2}" y="{yt-pt}" width="{pw}" height="{pt}" fill="#9ca3af" stroke="{NOIR}" stroke-width="1.2"/>')
    # bandes c de part et d'autre du pied (efficaces) et zone inefficace au-delà
    aw, cw = 300 * sc, 120 * sc
    for sgn in (-1, 1):
        xa = xc + sgn * aw / 2 if sgn > 0 else xc - aw / 2 - cw
        A(f'<rect x="{xa}" y="{yt-pt}" width="{cw}" height="{pt}" fill="url(#hC)" stroke="{BLEU}" stroke-width="0.8"/>')
    A(f'<rect x="{xc-aw/2}" y="{yt-pt-60*sc}" width="{aw}" height="{60*sc}" fill="{NOIR}"/>')
    txt(A, xc, yt - pt - 60 * sc - 8, "pied 300 × 300", 10, "middle")
    # charge
    A(f'<line x1="{xc}" y1="{yt-pt-60*sc-100}" x2="{xc}" y2="{yt-pt-60*sc-52}" stroke="{ROUGE}" stroke-width="2.2" marker-end="url(#ar)"/>')
    txt(A, xc + 8, yt - pt - 60 * sc - 80, "P4 = 95 kN", 12, "start", ROUGE, w=True)
    # pression sous l'aire efficace (bloc)
    ew = aw + 2 * cw
    A(f'<rect x="{xc-ew/2}" y="{yt}" width="{ew}" height="14" fill="#fecaca" fill-opacity="0.9" stroke="{ROUGE}" stroke-width="0.8"/>')
    for i in range(9):
        xx = xc - ew / 2 + ew * (i + 0.5) / 9
        A(f'<line x1="{xx}" y1="{yt+2}" x2="{xx}" y2="{yt+12}" stroke="{ROUGE}" stroke-width="0.8"/>')
    txt(A, xc + pw / 2 + 10, yt - 6, "p = 0,33 N/mm² sous l'aire efficace", 10, "start", ROUGE)
    # cotes
    ext(A, xc - ew/2, yt + 16, xc - ew/2, yt + 100); ext(A, xc + ew/2, yt + 16, xc + ew/2, yt + 100)
    ext(A, xc - aw/2, yt + 16, xc - aw/2, yt + 86); ext(A, xc + aw/2, yt + 16, xc + aw/2, yt + 86)
    cote_h(A, xc - aw/2, xc + aw/2, yt + 82, "a = 300", dy=-4, fs=10)
    cote_h(A, xc - ew/2, xc - aw/2, yt + 82, "c", dy=-4, fs=10); cote_h(A, xc + aw/2, xc + ew/2, yt + 82, "c", dy=-4, fs=10)
    cote_h(A, xc - ew/2, xc + ew/2, yt + 96, "aire efficace a + 2c = 540", dy=12, fs=10)
    ext(A, xc - pw/2, yt - pt - 4, xc - pw/2, yt - pt - 46); ext(A, xc + pw/2, yt - pt - 4, xc + pw/2, yt - pt - 46)
    cote_h(A, xc - pw/2, xc + pw/2, yt - pt - 42, "plateau 600 · t = 20", dy=-4, fs=10)
    ext(A, x0 - 4, yt, x0 - 40, yt); ext(A, x0 - 4, yt + 50, x0 - 40, yt + 50)
    cote_v(A, yt, yt + 50, x0 - 34, "h = 200", fs=10)
    txt(A, x0 + 303, yt + 76, "membrure sup. 30 mm", 9, "middle", GRIS)
    # texte
    txt(A, 20, 24, "a · Plateau posé sur le plancher : bande efficace c (régime de résistance)", 13, w=True)
    txt(A, 20, 44, "Le plateau ne porte en console que sur c de chaque côté de l'empreinte ; au-delà il ne transmet rien.", 10.5, fill=GRIS)
    txt(A, 20, 60, "c = t·√( f / (3·p) )  ·  p = P / ((a + 2c)·(b + 2c))  ·  2 à 3 itérations", 11)
    txt(A, 20, 76, "acier : f = fy / γM0 (EN 1993-1-8 §6.2.5) · OSB, bois : f de la fiche, saisi avec sa source", 10, fill=GRIS)
    txt(A, 20, 372, "→ aire efficace : pression localisée (EN 1992-1-1 §6.7), membrure supérieure (donnée fabricant), poinçonnement (§6.4)", 10.5)
    txt(A, 20, 388, "→ flexion globale du plancher : seule la résultante P compte", 10.5)
    A('</svg>')
    return "\n".join(A.__self__)

# ================================================================ (b) filière sur ressorts
def planche_b():
    A = svg_open(640, 400).append
    txt(A, 20, 24, "b · Filière en travers des hourdis : poutre sur appuis élastiques (régime de rigidité)", 13, w=True)
    txt(A, 20, 44, "Chaque hourdis est un ressort ; la raideur relative filière / hourdis décide du partage.", 10.5, fill=GRIS)
    sc = 0.11                    # px / mm (3 hourdis de 1200 = 396 px)
    x0, yt = 110, 190
    n = 3
    for i in range(n):
        hourdis_coupe(lambda s: A(s), x0 + i * (1200 * sc + 4), yt, 1200 * sc, 200 * sc, nv=5)
    # filière HEB 140 (3600) posée sur les trois
    fw, fh = 3600 * sc, 14
    fx = x0 + (3 * 1200 * sc + 8 - fw) / 2
    A(f'<rect x="{fx}" y="{yt-fh}" width="{fw}" height="{fh}" fill="#9ca3af" stroke="{NOIR}" stroke-width="1.2"/>')
    txt(A, fx + fw + 6, yt - 3, "HEB 140 · 3,60 m", 10)
    # déformée (exagérée)
    pts = []
    for k in range(41):
        t = k / 40
        xx = fx + fw * t
        yy = yt - fh - 10 - 16 * math.sin(math.pi * t) ** 1.2
        pts.append(f"{xx:.1f},{yy:.1f}")
    A(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{GRIS}" stroke-width="1" stroke-dasharray="4 3"/>')
    txt(A, fx + fw / 2, yt - fh - 32, "déformée w(x)", 9, "middle", GRIS, it=True)
    # charge au centre
    xc = x0 + 1.5 * 1200 * sc + 4
    A(f'<line x1="{xc}" y1="{yt-fh-74}" x2="{xc}" y2="{yt-fh-14}" stroke="{ROUGE}" stroke-width="2.2" marker-end="url(#ar)"/>')
    txt(A, xc + 8, yt - fh - 56, "P4 = 95 kN", 12, "start", ROUGE, w=True)
    A(f'<rect x="{xc-300*sc/2}" y="{yt-fh-8}" width="{300*sc}" height="8" fill="{NOIR}"/>')
    # ressorts + réactions
    R = (24, 47, 24)
    for i in range(n):
        cx = x0 + i * (1200 * sc + 4) + 600 * sc
        yb = yt + 200 * sc
        zz = "".join(f" L {cx + (6 if k % 2 else -6)} {yb + 6 + k*6}" for k in range(1, 9))
        A(f'<path d="M {cx} {yb}{zz} L {cx} {yb+58}" fill="none" stroke="{GRIS}" stroke-width="1.3"/>')
        A(f'<line x1="{cx-14}" y1="{yb+58}" x2="{cx+14}" y2="{yb+58}" stroke="{NOIR}" stroke-width="2"/>')
        A(f'<rect x="{cx-14}" y="{yb+58}" width="28" height="6" fill="url(#hA)"/>')
        txt(A, cx, yb + 80, f"k{i+1}", 11, "middle", w=True)
        txt(A, cx, yb + 94, f"R{i+1} = {R[i]} kN", 10, "middle", ROUGE, w=True)
        txt(A, cx, yt + 200 * sc / 2 + 4, f"H{i+3}", 9, "middle", GRIS)
    txt(A, x0 - 6, yt + 200 * sc / 2 + 4, "3 × 1 200", 9, "end", GRIS)
    # formules
    y = 330
    txt(A, 20, y, "k_i = 3·E·I·L / (a²·(L − a)²) : flèche d'un hourdis sur deux appuis, charge à l'abscisse a de la filière", 10.5)
    txt(A, 20, y + 16, "filière : poutre d'Euler EI_f libre aux extrémités, 2 ddl par nœud, k_i sur la diagonale, numpy.linalg.solve", 10.5)
    txt(A, 20, y + 32, "sorties : R_i (Σ R_i = Σ P), M et V dans la filière (1,5·M ≤ W·fy/γM0), pression de semelle sur chaque hourdis", 10.5)
    txt(A, 20, y + 50, "Hourdis indépendants : exact non clavetés, conservatif pour le plus chargé sinon.", 10, fill=GRIS)
    txt(A, 20, y + 62, "Plateau à cheval sur un joint : même modèle avec EI = E·(b + 2c)·t³/12.", 10, fill=GRIS)
    A('</svg>')
    return "\n".join(A.__self__)

# ================================================================ (c) largeur efficace (plan)
def planche_c():
    A = svg_open(640, 400).append
    txt(A, 20, 24, "c · Dalle unidirectionnelle : largeur efficace sous charge concentrée (DAfStb Heft 240)", 13, w=True)
    txt(A, 20, 44, "La charge se répartit sur une largeur qui croît avec la distance a à l'appui ; les efforts par mètre se somment en créneaux.", 10.5, fill=GRIS)
    x0, y0, w, h = 120, 86, 400, 206       # dalle vue en plan, appuis à gauche et à droite
    A(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#f8fafc" stroke="{NOIR}" stroke-width="1"/>')
    for xa in (x0 - 14, x0 + w):
        A(f'<rect x="{xa}" y="{y0}" width="14" height="{h}" fill="url(#hA)" stroke="{NOIR}" stroke-width="1"/>')
    txt(A, x0 - 7, y0 - 6, "appui", 9, "middle", GRIS); txt(A, x0 + w + 7, y0 - 6, "appui", 9, "middle", GRIS)
    # charge (empreinte) à a = 0,3 l, centrée en y
    lx, ly = x0 + 0.32 * w, y0 + h / 2
    bw, bh = 34, 26
    A(f'<rect x="{lx-bw/2}" y="{ly-bh/2}" width="{bw}" height="{bh}" fill="{NOIR}"/>')
    txt(A, lx, ly - bh / 2 - 8, "P · empreinte b₀", 10, "middle")
    # t_y (diffusion à 45° jusqu'au plan moyen) et trapèzes de largeur efficace vers les appuis
    ty = 70
    A(f'<rect x="{lx-bw/2-10}" y="{ly-ty/2}" width="{bw+20}" height="{ty}" fill="none" stroke="{BLEU}" stroke-width="0.8" stroke-dasharray="3 2"/>')
    bL, bR = 150, 210      # b_eff aux appuis (illustration)
    A(f'<polygon points="{x0},{ly-bL/2} {lx-bw/2-10},{ly-ty/2} {lx+bw/2+10},{ly-ty/2} {x0+w},{ly-bR/2} {x0+w},{ly+bR/2} {lx+bw/2+10},{ly+ty/2} {lx-bw/2-10},{ly+ty/2} {x0},{ly+bL/2}" fill="#dbeafe" fill-opacity="0.5" stroke="{BLEU}" stroke-width="1.2" stroke-dasharray="6 3"/>')
    # cotes
    ext(A, lx - bw/2 - 10, ly - ty/2, lx - bw/2 - 50, ly - ty/2); ext(A, lx - bw/2 - 10, ly + ty/2, lx - bw/2 - 50, ly + ty/2)
    cote_v(A, ly - ty/2, ly + ty/2, lx - bw/2 - 44, "t_y", fs=10)
    ext(A, x0 + w, ly - bR/2, x0 + w + 40, ly - bR/2); ext(A, x0 + w, ly + bR/2, x0 + w + 40, ly + bR/2)
    cote_v(A, ly - bR/2, ly + bR/2, x0 + w + 34, "b_v à l'appui", dx=12, fs=10)
    ext(A, x0, y0 + h + 2, x0, y0 + h + 40); ext(A, lx, ly + bh / 2 + 2, lx, y0 + h + 24); ext(A, x0 + w, y0 + h + 2, x0 + w, y0 + h + 40)
    cote_h(A, x0, lx, y0 + h + 18, "a", dy=-4, fs=10)
    cote_h(A, x0, x0 + w, y0 + h + 36, "l", dy=-4, fs=10)
    # créneaux : profil m(y) à droite
    txt(A, x0 + w / 2, y0 + h - 8, "sens porteur →", 9, "middle", GRIS)
    y = 345
    txt(A, 20, y, "t_y = b₀ + 2·h₁ + h (diffusion à 45° dans le revêtement h₁, puis jusqu'au plan moyen), t_y ≤ 0,8·l", 10.5)
    txt(A, 20, y + 16, "tranchant : b_v = t_y + 0,5·a ≤ b (0,3·a en porte-à-faux) · flexion : b_m = t_y + k·x·(1 − x/l), k [à transcrire]", 10.5)
    txt(A, 20, y + 32, "plusieurs charges : m(y) = Σ M_j / b_eff,j en créneaux, maximum sur la largeur · rive libre : demi-largeur · As,transv ≥ 20 %", 10.5)
    A('</svg>')
    return "\n".join(A.__self__)

# ================================================================ (d) répartition entre hourdis
def planche_d():
    A = svg_open(640, 400).append
    txt(A, 20, 24, "d · Hourdis : répartition transversale selon l'état des joints", 13, w=True)
    txt(A, 20, 44, "Clavetés : parts de la figure C.1 de l'EN 1168 (flexion). Non clavetés : rien, ou règle du levier sur le joint.", 10.5, fill=GRIS)
    sc = 0.1
    x0, yt = 60, 196
    n = 5
    parts = (6, 24, 52, 24, 6)      # illustration, à remplacer par la figure C.1 transcrite
    for i in range(n):
        xx = x0 + i * (1200 * sc + 3)
        hourdis_coupe(lambda s: A(s), xx, yt, 1200 * sc, 200 * sc, nv=4)
        cx = xx + 600 * sc
        # barre de part
        hb = parts[i] * 1.6
        A(f'<rect x="{cx-22}" y="{yt-14-hb}" width="44" height="{hb}" fill="#dbeafe" stroke="{BLEU}" stroke-width="1"/>')
        txt(A, cx, yt - 18 - hb, f"{parts[i]} %", 10, "middle", BLEU, w=True)
        txt(A, cx, yt + 200 * sc + 14, f"H{i+2}", 9, "middle", GRIS)
    xc = x0 + 2 * (1200 * sc + 3) + 600 * sc
    A(f'<line x1="{xc}" y1="{yt-14-52*1.6-44}" x2="{xc}" y2="{yt-14-52*1.6-20}" stroke="{ROUGE}" stroke-width="2.2" marker-end="url(#ar)"/>')
    txt(A, xc + 8, yt - 14 - 52 * 1.6 - 30, "P", 12, "start", ROUGE, w=True)
    txt(A, x0, yt + 200 * sc + 34, "joints clavetés et durcis : parts illustratives ; les valeurs réelles sont celles de la figure C.1 [à transcrire],", 9.5, fill=GRIS)
    txt(A, x0, yt + 200 * sc + 46, "fonction de la portée et de la position de la charge (centre / rive)", 9.5, fill=GRIS)
    # levier (non clavetés) : petite vignette à droite
    xv, yv = 430, 296
    hourdis_coupe(lambda s: A(s), xv, yv, 90, 18, nv=3); hourdis_coupe(lambda s: A(s), xv + 93, yv, 90, 18, nv=3)
    A(f'<rect x="{xv+70}" y="{yv-10}" width="44" height="10" fill="{NOIR}"/>')
    A(f'<line x1="{xv+92}" y1="{yv-40}" x2="{xv+92}" y2="{yv-14}" stroke="{ROUGE}" stroke-width="2" marker-end="url(#ar)"/>')
    txt(A, xv + 92, yv + 34, "non clavetés : règle du levier", 9.5, "middle")
    txt(A, xv + 92, yv + 46, "55 % | 45 % selon la position de l'empreinte", 9, "middle", GRIS)
    txt(A, 20, 300, "Tranchant : 100 % à l'élément chargé (V1).", 10.5, w=True)
    txt(A, 20, 316, "Filière et annexe C ne se cumulent pas :", 10.5)
    txt(A, 20, 330, "chaque hourdis reçoit sa réaction R_i.", 10.5)
    txt(A, 20, 372, "Données indispensables : état des joints et leur âge (fck(t)), M_Rd et V_Rd avec leur nature, I de la section.", 9.5, fill=GRIS)
    txt(A, 20, 384, "Facultatives : M_cr, pression locale admissible sur la membrure.", 9.5, fill=GRIS)
    A('</svg>')
    return "\n".join(A.__self__)

CSS = """
*{box-sizing:border-box} body{margin:0;background:#fff;color:#111827;font-family:Inter,sans-serif}
.page{max-width:1400px;margin:0 auto;padding:18px 28px 26px}
h1{font-size:22px;margin:0 0 4px} .sub{color:#6b7280;font-size:12px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{border:1px solid #e5e7eb;border-radius:10px;padding:8px 10px;background:#fff}
.foot{font-size:11px;color:#6b7280;margin-top:10px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
"""
HTML = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Modèle de calcul</title><style>{CSS}</style></head><body><div class="page">
<h1>Charges sur plancher — modèle de calcul sans maillage</h1>
<div class="sub">Deux régimes de répartition (résistance du plateau, rigidité de la filière), diffusion par formules fermées, répartition entre hourdis selon l'état des joints · valeurs d'illustration</div>
<div class="grid">
<div class="card">{planche_a()}</div>
<div class="card">{planche_b()}</div>
<div class="card">{planche_c()}</div>
<div class="card">{planche_d()}</div>
</div>
<div class="foot">Appui sur quatre côtés (V2) : la série de Navier de la plaque remplace les ressorts par une matrice de souplesse ; le solveur de la planche b ne change pas. Aucun coefficient de norme n'est reconstitué de mémoire : les valeurs marquées [à transcrire] se relisent sur le texte imprimé au lot 1.</div>
</div></body></html>"""
(OUT / "modele.html").write_text(HTML, encoding="utf-8")
print("ok")

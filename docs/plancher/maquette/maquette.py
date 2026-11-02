# -*- coding: utf-8 -*-
"""Maquette « digeste » de l'écran Charges sur plancher, troisième itération :
symboles d'engins au trait, à l'échelle des dimensions publiées (sources
secondaires relevées en ligne le 02/11/2026, à confirmer sur les fiches
officielles), couche de compression du hourdis. HTML + SVG rendus par
Chromium. Valeurs de calcul d'illustration."""
import sys, pathlib

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

ROUGE, ROUGE_F, ORANGE, ORANGE_F = "#b91c1c", "#fee2e2", "#c2410c", "#ffedd5"
BLEU, NOIR, GRIS, LIGNE, ROUE, CAPOT = "#2563eb", "#111827", "#6b7280", "#d1d5db", "#d1d5db", "#f3f4f6"
NS = 'vector-effect="non-scaling-stroke"'

# ------------------------------------------------------------------ symboles d'engins (mètres, x vers l'avant, y vers la droite)
def _rect(x, y, w, h, fill="white", rx=0.0, sw=1.1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{NOIR}" stroke-width="{sw}" {NS}{d}/>'
def _poly(pts, fill="white", sw=1.1):
    return f'<polygon points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="{fill}" stroke="{NOIR}" stroke-width="{sw}" {NS}/>'
def _circle(cx, cy, r, fill="white", sw=1.1):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{NOIR}" stroke-width="{sw}" {NS}/>'
def _poutre(x1, y1, x2, y2, w):
    """Bras rectangulaire de largeur w entre deux points (polygone)."""
    import math
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy); ux, uy = -dy / n * w / 2, dx / n * w / 2
    return _poly([(x1 + ux, y1 + uy), (x2 + ux, y2 + uy), (x2 - ux, y2 - uy), (x1 - ux, y1 - uy)], fill="white")

def mrt_2150():
    """Télescopique rotatif sur 4 stabilisateurs. Dimensions : largeur 2,49,
    châssis 5,35, empattement 3,05, emprise stabilisateurs 5,55 × 5,19
    (brochure Manitou relevée en ligne, à confirmer). Vue de dessus : châssis
    et roues, stabilisateurs en ciseaux, tourelle (cabine à gauche, moteur à
    l'arrière), flèche sur le côté droit de la tourelle."""
    s = []
    for sx in (-1, 1):                                   # pneus
        for sy in (-1, 1):
            s.append(_rect(sx * 1.525 - 0.65, sy * 1.05 - 0.23, 1.3, 0.46, fill=ROUE, rx=0.08))
    s.append(_rect(-2.675, -1.245, 5.35, 2.49, rx=0.18))  # châssis
    for sx in (-1, 1):                                   # stabilisateurs en ciseaux
        for sy in (-1, 1):
            s.append(_poutre(sx * 2.0, sy * 1.1, sx * 2.35, sy * 2.52, 0.24))
    s.append(_rect(-2.1, -1.15, 3.4, 2.3, rx=0.55))       # tourelle (arrière arrondi)
    s.append(_circle(-0.45, 0, 0.95, fill="none", sw=0.8, ) .replace('stroke-width="0.8"', 'stroke-width="0.8" stroke-dasharray="3 2"'))  # couronne
    s.append(_rect(-2.0, -0.95, 0.9, 1.9, fill=CAPOT, rx=0.08))        # moteur
    s.append(_rect(-0.25, -1.1, 1.1, 0.95, rx=0.1))                    # cabine (gauche)
    s.append(_poly([(-1.9, 0.22), (3.4, 0.3), (3.4, 0.82), (-1.9, 0.9)]))      # flèche (côté droit, projection levée)
    s.append(f'<line x1="-1.9" y1="0.56" x2="3.4" y2="0.56" stroke="{NOIR}" stroke-width="0.6" {NS}/>')
    s.append(_rect(3.4, 0.0, 0.14, 1.12))                                 # tablier
    s.append(_circle(-0.45, 0, 0.09, fill=NOIR))
    pads = [(2.35, -2.52), (2.35, 2.52), (-2.35, -2.52), (-2.35, 2.52)]
    return "".join(s), pads, (5.6, 5.6)

def mt_1840():
    """Télescopique à roues. Dimensions : longueur au tablier 6,27, largeur
    2,42, empattement 3,07, pneus 400/80-24 (manitou.com, à confirmer)."""
    s = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.append(_rect(sx * 1.535 - 0.63, sy * 1.0 - 0.22, 1.26, 0.44, fill=ROUE, rx=0.08))
    s.append(_rect(-2.35, -1.21, 4.7, 2.42, rx=0.18))
    s.append(_rect(-1.4, -1.15, 1.25, 0.8, fill=CAPOT, rx=0.06))        # capot moteur (droite)
    s.append(_rect(-1.45, 0.3, 1.2, 0.88, rx=0.1))                      # cabine (gauche)
    s.append(_poly([(-2.2, -0.52), (2.5, -0.46), (2.5, 0.02), (-2.2, 0.08)]))   # flèche
    s.append(f'<line x1="-2.2" y1="-0.22" x2="2.5" y2="-0.22" stroke="{NOIR}" stroke-width="0.6" {NS}/>')
    s.append(_rect(2.5, -0.75, 0.14, 1.3))                              # tablier
    s.append(_rect(2.64, -0.58, 1.2, 0.12, fill=NOIR))                  # fourches
    s.append(_rect(2.64, 0.46, 1.2, 0.12, fill=NOIR))
    roues = [(1.535, -1.0), (1.535, 1.0), (-1.535, -1.0), (-1.535, 1.0)]
    return "".join(s), roues, (6.3, 2.5)

def ltm_1060():
    """Grue mobile 3 essieux. Dimensions : porteur 11,98 × 2,69, base d'appui
    7,36 × 6,30 (annonces de revendeurs, à confirmer sur la fiche Liebherr)."""
    s = []
    for x in (3.9, -1.9, -3.4):
        for sy in (-1, 1):
            s.append(_rect(x - 0.55, sy * 1.15 - 0.2, 1.1, 0.4, fill=ROUE, rx=0.06))
    s.append(_rect(-5.99, -1.345, 11.98, 2.69, rx=0.12))
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.append(_poutre(sx * 3.68, sy * 0.9, sx * 3.68, sy * 3.15, 0.34))
    s.append(_rect(4.3, -1.2, 1.6, 2.4, rx=0.12))                       # cabine porteur
    s.append(_rect(-3.7, -1.3, 1.7, 2.6, fill=CAPOT, rx=0.06))          # contrepoids
    s.append(_circle(-0.9, 0, 1.5))                                      # tourelle
    s.append(_rect(0.2, 0.85, 1.5, 0.95, rx=0.1))                       # cabine grue
    s.append(_poly([(-0.9, -0.5), (8.6, -0.4), (8.6, 0.4), (-0.9, 0.5)]))      # flèche
    s.append(f'<line x1="-0.9" y1="0" x2="8.6" y2="0" stroke="{NOIR}" stroke-width="0.6" {NS}/>')
    s.append(_circle(-0.9, 0, 0.1, fill=NOIR))
    pads = [(3.68, -3.15), (3.68, 3.15), (-3.68, -3.15), (-3.68, 3.15)]
    return "".join(s), pads, (14.6, 6.9)

def nacelle():
    s = [_rect(-1.2, -0.6, 2.4, 1.2, rx=0.08)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.append(_rect(sx * 0.85 - 0.2, sy * 0.64 - 0.08, 0.4, 0.16, fill=NOIR, rx=0.04))
    s.append(_rect(-1.1, -0.5, 2.2, 1.0, dash="3 2"))
    s.append(_rect(1.2, -0.52, 0.9, 1.04, dash="3 2"))
    return "".join(s), [(0.85, -0.64), (0.85, 0.64), (-0.85, -0.64), (-0.85, 0.64)], (3.2, 1.6)

def grue_araignee():
    """Mini-grue sur chenilles, 4 stabilisateurs en diagonale (gabarit générique,
    proportions d'une mini-grue de 3 à 6 t : chenilles 2,9 × 1,2, emprise 4,0 × 4,0)."""
    s = []
    for sy in (-1, 1):
        s.append(_rect(-1.45, sy * 0.5 - 0.17, 2.9, 0.34, fill=ROUE, rx=0.06))
    s.append(_rect(-1.1, -0.42, 2.2, 0.84, rx=0.08))
    for sx in (-1, 1):
        for sy in (-1, 1):
            s.append(_poutre(sx * 0.9, sy * 0.35, sx * 1.75, sy * 1.75, 0.16))
    s.append(_circle(-0.2, 0, 0.38))
    s.append(_poly([(-0.2, -0.16), (2.6, -0.1), (2.6, 0.1), (-0.2, 0.16)]))
    s.append(_circle(-0.2, 0, 0.06, fill=NOIR))
    pads = [(1.75, -1.75), (1.75, 1.75), (-1.75, -1.75), (-1.75, 1.75)]
    return "".join(s), pads, (4.2, 4.2)

ENGINS = {"mrt": mrt_2150, "mt": mt_1840, "ltm": ltm_1060, "nac": nacelle, "ara": grue_araignee}

def place(fn, px, py, scale, angle=0):
    body, appuis, _ = fn()
    return f'<g transform="translate({px} {py}) rotate({angle}) scale({scale})">{body}</g>'

def picto(kind, w=112, h=72):
    if kind in ENGINS:
        body, appuis, (lw, lh) = ENGINS[kind]()
        sc = min((w - 10) / lw, (h - 8) / lh)
        pads = ""
        if kind in ("mrt", "ltm", "ara"):
            pads = "".join(f'<rect x="{x-0.3}" y="{y-0.3}" width="0.6" height="0.6" fill="white" stroke="{NOIR}" stroke-width="1" {NS}/><rect x="{x-0.15}" y="{y-0.15}" width="0.3" height="0.3" fill="{NOIR}"/>' for x, y in appuis)
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"><g transform="translate({w/2} {h/2}) scale({sc:.3f})">{body}{pads}</g></svg>'
    if kind == "etancon":
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"><circle cx="{w/2}" cy="{h/2}" r="11" fill="white" stroke="{NOIR}" stroke-width="1.4"/><line x1="{w/2-7}" y1="{h/2-7}" x2="{w/2+7}" y2="{h/2+7}" stroke="{NOIR}"/><line x1="{w/2-7}" y1="{h/2+7}" x2="{w/2+7}" y2="{h/2-7}" stroke="{NOIR}"/></svg>'
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}"><defs><pattern id="hb" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#7c3aed" stroke-width=".7"/></pattern></defs><rect x="{w/2-30}" y="{h/2-16}" width="60" height="32" fill="url(#hb)" stroke="#7c3aed"/></svg>'

# ------------------------------------------------------------------ plan
S = 60.0
L, B, NU, BU = 7.20, 9.60, 8, 1.20
OX, OY = 118.0, 58.0
def X(x): return OX + x * S
def Yt(yt): return OY + yt * S

TAUX = {1: 24, 2: 46, 3: 118, 4: 62, 5: 55, 6: 60, 7: 93, 8: 31}

def plan_svg():
    A = []; a = A.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 736" width="100%" style="font-family:Inter,sans-serif;font-size:12px">')
    a('''<defs>
<pattern id="hc" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#9ca3af" stroke-width="0.8"/></pattern>
<marker id="tk" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 5 1 L 5 9" stroke="#374151" stroke-width="1.4"/></marker>
</defs>''')
    for i in range(NU):
        n = i + 1
        y0 = Yt(i * BU)
        fill = ROUGE_F if TAUX[n] > 100 else (ORANGE_F if TAUX[n] >= 85 else "white")
        a(f'<rect x="{X(0)}" y="{y0}" width="{L*S}" height="{BU*S}" fill="{fill}"/>')
        col = ROUGE if TAUX[n] > 100 else (ORANGE if TAUX[n] >= 85 else GRIS)
        a(f'<text x="{X(0)-9}" y="{y0+BU*S/2+4}" font-size="11.5" fill="{col}" font-weight="{600 if TAUX[n] >= 85 else 400}" text-anchor="end">H{n}</text>')
        tx, ty = X(L) + 26, y0 + BU * S / 2 + 4
        if TAUX[n] > 100:
            a(f'<rect x="{tx-4}" y="{ty-13}" width="58" height="19" rx="9.5" fill="{ROUGE}"/><text x="{tx+25}" y="{ty+1}" font-size="12.5" font-weight="700" fill="white" text-anchor="middle">{TAUX[n]} %</text>')
        elif TAUX[n] >= 85:
            a(f'<rect x="{tx-4}" y="{ty-13}" width="54" height="19" rx="9.5" fill="white" stroke="{ORANGE}" stroke-width="1.2"/><text x="{tx+23}" y="{ty+1}" font-size="12.5" font-weight="700" fill="{ORANGE}" text-anchor="middle">{TAUX[n]} %</text>')
        else:
            a(f'<text x="{tx}" y="{ty}" font-size="12.5" fill="{GRIS}">{TAUX[n]} %</text>')
    for i in range(1, NU):
        y = Yt(i * BU)
        a(f'<line x1="{X(0)}" y1="{y}" x2="{X(L)}" y2="{y}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="5 4"/>')
    a(f'<line x1="{X(0)}" y1="{Yt(0)}" x2="{X(L)}" y2="{Yt(0)}" stroke="{LIGNE}" stroke-width="1.2"/>')
    a(f'<line x1="{X(0)}" y1="{Yt(B)}" x2="{X(L)}" y2="{Yt(B)}" stroke="{LIGNE}" stroke-width="1.2"/>')
    a(f'<line x1="{X(0)}" y1="{Yt(0)-2}" x2="{X(0)}" y2="{Yt(B)+2}" stroke="{NOIR}" stroke-width="5"/>')
    a(f'<line x1="{X(L)}" y1="{Yt(0)-2}" x2="{X(L)}" y2="{Yt(B)+2}" stroke="{NOIR}" stroke-width="5"/>')
    a(f'<text x="{X(0)}" y="{Yt(0)-10}" font-size="12" font-weight="700" fill="{NOIR}" text-anchor="middle">A</text>')
    a(f'<text x="{X(L)}" y="{Yt(0)-10}" font-size="12" font-weight="700" fill="{NOIR}" text-anchor="middle">B</text>')
    for (x, y, on) in ((X(0), Yt(B/2), True), (X(L), Yt(B/2), True), (X(L/2), Yt(0), False), (X(L/2), Yt(B), False)):
        a(f'<circle cx="{x}" cy="{y}" r="6" fill="{NOIR if on else "white"}" stroke="{NOIR}" stroke-width="1.4"/>')
    a(f'<text x="{X(L/2)+12}" y="{Yt(0)-6}" font-size="10.5" fill="{GRIS}">bord libre · clic = appui</text>')
    a(f'<line x1="{X(0)}" y1="{Yt(0)-34}" x2="{X(L)}" y2="{Yt(0)-34}" stroke="#374151" stroke-width="1" marker-start="url(#tk)" marker-end="url(#tk)"/>')
    a(f'<text x="{X(L/2)}" y="{Yt(0)-39}" font-size="12" fill="#374151" text-anchor="middle">7,20 m</text>')
    a(f'<line x1="{X(0)-40}" y1="{Yt(0)}" x2="{X(0)-40}" y2="{Yt(B)}" stroke="#374151" stroke-width="1" marker-start="url(#tk)" marker-end="url(#tk)"/>')
    a(f'<text transform="translate({X(0)-46} {Yt(B/2)}) rotate(-90)" font-size="12" fill="#374151" text-anchor="middle">8 × 1,20 = 9,60 m</text>')

    # ---- grue araignée MRT 2150 à l'échelle : centre (3,00 ; 5,60), flèche vers B
    cx, cy = 3.00, 5.60
    body, pads, _ = mrt_2150()
    def R(x, y, w, h, **kw):
        at = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        a(f'<rect x="{X(x)}" y="{Yt(y)}" width="{w*S}" height="{h*S}" {at}/>')
    a(place(mrt_2150, X(cx), Yt(cy), S))
    a(f'<text x="{X(cx)}" y="{Yt(cy+1.245)+15}" font-size="11" fill="{GRIS}" text-anchor="middle">MRT 2150 · levage sur stabilisateurs</text>')
    NOMS = {(2.35, -2.52): ("P2", 95, True), (-2.35, -2.52): ("P1", 95, False), (2.35, 2.52): ("P4", 60, False), (-2.35, 2.52): ("P3", 60, False)}
    for (dx, dy) in pads:
        nom, P, sel = NOMS[(dx, dy)]
        px, py = cx + dx, cy + dy
        R(px - 0.27, py - 0.27, 0.54, 0.54, fill="none", stroke=BLEU, stroke_width=1, stroke_dasharray="3 2")
        R(px - 0.30, py - 0.30, 0.60, 0.60, fill="url(#hc)", stroke=NOIR, stroke_width=1.1)
        R(px - 0.15, py - 0.15, 0.30, 0.30, fill=NOIR)
        if sel:
            R(px - 0.36, py - 0.36, 0.72, 0.72, fill="none", stroke=BLEU, stroke_width=1.8, rx=3)
        lx, ly = X(px) - 0.42 * S, Yt(py) + (-0.42 * S - 20 if dy < 0 else 0.42 * S + 2)
        if sel:
            a(f'<rect x="{lx}" y="{ly}" width="82" height="20" rx="3" fill="{BLEU}"/><text x="{lx+8}" y="{ly+14}" font-size="11.5" font-weight="600" fill="white">{nom} · {P} kN</text>')
        else:
            a(f'<rect x="{lx}" y="{ly}" width="82" height="20" rx="3" fill="white" stroke="{LIGNE}"/><text x="{lx+8}" y="{ly+14}" font-size="11.5" fill="{NOIR}"><tspan font-weight="600">{nom}</tspan> · {P} kN</text>')
    # filière proposée sous P2, en travers de H2–H4
    fx = cx + 2.35
    R(fx - 0.07, 1.2 + 0.05, 0.14, 3.6, fill="none", stroke=BLEU, stroke_width=1.2, stroke_dasharray="6 3")
    a(f'<text x="{X(fx)-10}" y="{Yt(1.2)+16}" font-size="11" fill="{BLEU}" text-anchor="end">filière HEB 140 · 3,60 m (proposée)</text>')
    # fenêtre du répartiteur (clic sur le plateau P2) : à gauche sous le pied
    ppx, ppy = 655, Yt(cy - 2.52) + 16
    a(f'<line x1="{X(cx+2.35)+0.36*S}" y1="{Yt(cy-2.52)}" x2="{ppx}" y2="{ppy+35}" stroke="{BLEU}" stroke-width="1" stroke-dasharray="3 3"/>')
    a(f'<rect x="{ppx}" y="{ppy}" width="238" height="70" rx="8" fill="white" stroke="{BLEU}" filter="drop-shadow(0 2px 4px rgba(0,0,0,.12))"/>')
    a(f'<text x="{ppx+12}" y="{ppy+20}" font-size="11.5" font-weight="700" fill="{NOIR}">Répartiteur sous P2</text>')
    a(f'<rect x="{ppx+12}" y="{ppy+30}" width="214" height="24" rx="5" fill="#f3f4f6" stroke="{LIGNE}"/>')
    a(f'<text x="{ppx+20}" y="{ppy+46}" font-size="11.5" fill="{NOIR}">Plateau acier 600 × 600 × 20</text><text x="{ppx+216}" y="{ppy+46}" font-size="11" fill="{GRIS}" text-anchor="end">▾</text>')
    a(f'<text x="{ppx+12}" y="{ppy+64}" font-size="10.5" fill="{GRIS}">centré sur le pied · glisser · ✕ retirer</text>')
    # étançon E1
    ex, ey = 6.30, 0.60
    a(f'<circle cx="{X(ex)}" cy="{Yt(ey)}" r="11" fill="white" stroke="{NOIR}" stroke-width="1.4"/>')
    a(f'<line x1="{X(ex)-7}" y1="{Yt(ey)-7}" x2="{X(ex)+7}" y2="{Yt(ey)+7}" stroke="{NOIR}" stroke-width="1.2"/><line x1="{X(ex)-7}" y1="{Yt(ey)+7}" x2="{X(ex)+7}" y2="{Yt(ey)-7}" stroke="{NOIR}" stroke-width="1.2"/>')
    a(f'<rect x="{X(ex)-74}" y="{Yt(ey)+16}" width="86" height="20" rx="3" fill="white" stroke="{LIGNE}"/><text x="{X(ex)-66}" y="{Yt(ey)+30}" font-size="11.5" fill="{NOIR}"><tspan font-weight="600">E1</tspan> · 35 kN</text>')
    sx, sy = X(0), Yt(B) + 30
    a(f'<rect x="{sx}" y="{sy}" width="{S}" height="5" fill="{NOIR}"/><rect x="{sx+S}" y="{sy}" width="{S}" height="5" fill="white" stroke="{NOIR}"/>')
    for i, t in enumerate(("0", "1", "2 m")):
        a(f'<text x="{sx+i*S}" y="{sy+18}" font-size="10.5" fill="{GRIS}" text-anchor="middle">{t}</text>')
    a('</svg>')
    return "\n".join(A)

def coupe_svg():
    A = []; a = A.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 236" width="100%" style="font-family:Inter,sans-serif;font-size:12px">')
    a('<defs><pattern id="hc2" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#9ca3af" stroke-width="0.8"/></pattern>'
      '<pattern id="hb2" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="#9ca3af" stroke-width="0.6"/></pattern>'
      '<marker id="tk2" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 5 1 L 5 9" stroke="#374151" stroke-width="1.4"/></marker>'
      '<marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker></defs>')
    sc = 0.22
    yt, h, tc = 136, 200 * sc, 50 * sc          # hourdis 200, couche de compression 50
    bu = 1200 * sc
    x3 = 450 - bu / 2
    xs = [x3 - bu - 4, x3, x3 + bu + 4]
    # couche de compression, continue sur les trois hourdis
    a(f'<rect x="{xs[0]}" y="{yt-tc}" width="{3*bu+8}" height="{tc}" fill="url(#hb2)" stroke="{NOIR}" stroke-width="1"/>')
    for x0, nom, fill in zip(xs, ("H2", "H3", "H4"), ("white", ROUGE_F, "white")):
        a(f'<rect x="{x0}" y="{yt}" width="{bu}" height="{h}" fill="{fill}" stroke="{NOIR}" stroke-width="1.2"/>')
        for i in range(5):
            a(f'<circle cx="{x0+bu*(i+0.5)/5}" cy="{yt+h/2}" r="14" fill="white" stroke="{NOIR}" stroke-width="1"/>')
        a(f'<text x="{x0+bu/2}" y="{yt+h+16}" font-size="12" fill="{ROUGE if nom == "H3" else GRIS}" font-weight="{600 if nom == "H3" else 400}" text-anchor="middle">{nom}</text>')
    xc = 450
    top = yt - tc
    a(f'<rect x="{xc-66}" y="{top-6}" width="132" height="6" fill="url(#hc2)" stroke="{NOIR}" stroke-width="1"/>')
    a(f'<rect x="{xc-33}" y="{top-6-14}" width="66" height="14" fill="{NOIR}"/>')
    a(f'<line x1="{xc}" y1="{top-82}" x2="{xc}" y2="{top-26}" stroke="{NOIR}" stroke-width="2" marker-end="url(#ar2)"/>')
    a(f'<text x="{xc+10}" y="{top-56}" font-size="13" font-weight="600" fill="{NOIR}">95 kN</text>')
    a(f'<text x="{xc+76}" y="{top-10}" font-size="11.5" fill="{GRIS}">plateau 600 × 20</text>')
    a(f'<text x="{xs[0]+4}" y="{top-6}" font-size="11" fill="{GRIS}">couche de compression 50 · C25/30 · 14 j</text>')
    e = 540 * sc / 2
    for sgn in (-1, 1):
        a(f'<line x1="{xc+sgn*e}" y1="{top}" x2="{xc+sgn*(e+tc+h/2)}" y2="{yt+h/2}" stroke="{BLEU}" stroke-width="1" stroke-dasharray="4 3"/>')
    a(f'<line x1="{xc-e}" y1="{yt+h+30}" x2="{xc+e}" y2="{yt+h+30}" stroke="{BLEU}" stroke-width="1" marker-start="url(#tk2)" marker-end="url(#tk2)"/>')
    a(f'<text x="{xc}" y="{yt+h+48}" font-size="12.5" fill="{BLEU}" text-anchor="middle">aire efficace 540 mm · diffusion à 45° jusqu\'au plan moyen</text>')
    xr = xs[2] + bu + 14
    a(f'<line x1="{xr}" y1="{yt}" x2="{xr}" y2="{yt+h}" stroke="#374151" stroke-width="1" marker-start="url(#tk2)" marker-end="url(#tk2)"/>')
    a(f'<text x="{xr+8}" y="{yt+h/2+4}" font-size="11.5" fill="#374151">200</text>')
    a(f'<line x1="{xr}" y1="{top}" x2="{xr}" y2="{yt}" stroke="#374151" stroke-width="1" marker-start="url(#tk2)" marker-end="url(#tk2)"/>')
    a(f'<text x="{xr+8}" y="{top+tc/2+4}" font-size="11.5" fill="#374151">50</text>')
    a('</svg>')
    return "\n".join(A)

CSS = """
*{box-sizing:border-box} body{margin:0;background:#f5f6f8;color:#111827;font-family:Inter,'Source Sans Pro',sans-serif;font-size:13.5px}
.top{display:flex;align-items:center;gap:14px;background:#fff;border-bottom:1px solid #e5e7eb;padding:10px 24px}
.top h1{font-size:17px;margin:0;font-weight:600} .top .cas{display:flex;align-items:center;gap:8px;margin-left:12px;font-size:12px;color:#6b7280;letter-spacing:.06em}
.sel{border:1px solid #d1d5db;border-radius:7px;padding:6px 10px;background:#fff;font-size:13px;color:#111827;letter-spacing:0}
.btn{border:1px solid #d1d5db;border-radius:7px;padding:6px 12px;background:#fff;font-size:13px;white-space:nowrap}
.btn.dark{background:#111827;color:#fff;border-color:#111827} .right{margin-left:auto;display:flex;gap:8px}
.page{display:grid;grid-template-columns:1fr 400px;gap:18px;padding:18px 24px 28px;align-items:start}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px}
.card+.card{margin-top:12px}
.bar{display:flex;align-items:center;gap:16px;font-size:12.5px;color:#374151;padding-bottom:8px;border-bottom:1px solid #f3f4f6;margin-bottom:6px}
.seg{display:inline-flex;border:1px solid #d1d5db;border-radius:7px;overflow:hidden;font-size:12.5px}
.seg span{padding:5px 12px;color:#6b7280} .seg span.on{background:#111827;color:#fff}
.chk{display:inline-flex;align-items:center;gap:6px} .chk i{display:inline-block;width:14px;height:14px;border-radius:3px;background:#2563eb;color:#fff;font-style:normal;font-size:11px;line-height:14px;text-align:center}
.legend{display:flex;gap:18px;font-size:11.5px;color:#6b7280;padding-top:8px;border-top:1px solid #f3f4f6;margin-top:4px;flex-wrap:wrap}
.legend i{display:inline-block;width:14px;height:11px;vertical-align:-1px;margin-right:6px;border:1px solid #9ca3af}
.h{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.h .t{font-size:11px;letter-spacing:.08em;color:#6b7280;font-weight:600}
.pill{font-size:12px;border-radius:999px;padding:3px 10px;font-weight:600} .pill.nok{background:#fee2e2;color:#b91c1c}
.row{display:grid;grid-template-columns:120px 1fr 56px;align-items:center;gap:10px;font-size:13px;margin:7px 0}
.trk{height:6px;background:#f1f3f5;border-radius:3px;overflow:hidden} .trk b{display:block;height:100%}
.v{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.tip{display:flex;align-items:center;gap:10px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:8px 10px;margin-top:10px;font-size:12.5px}
.tip span{white-space:nowrap} .tip .btn{margin-left:auto;border-color:#2563eb;color:#2563eb;padding:4px 10px}
.step{display:flex;align-items:center;gap:12px} .num{width:22px;height:22px;border-radius:50%;background:#111827;color:#fff;font-size:12px;display:flex;align-items:center;justify-content:center;font-weight:600;flex:none}
.num.g{background:#e5e7eb;color:#374151}
.step .ttl{font-weight:600;font-size:14px} .step .sub{font-size:12px;color:#6b7280;margin-top:2px}
.chev{margin-left:auto;color:#9ca3af;font-size:14px}
.list{margin-top:10px;border-top:1px solid #f3f4f6}
.it{display:flex;align-items:center;padding:9px 8px;border-bottom:1px solid #f3f4f6;font-size:13px}
.it.on{background:#f8fafc;border-left:3px solid #2563eb;padding-left:5px}
.it .sub{display:block;font-size:12px;color:#6b7280;margin-top:2px} .it .kn{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap;padding-left:10px}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px}
th{text-align:left;font-size:10.5px;letter-spacing:.06em;color:#6b7280;font-weight:600;padding:4px 6px;border-bottom:1px solid #e5e7eb}
td{padding:7px 6px;border-bottom:1px solid #f3f4f6;white-space:nowrap} td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.r{color:#b91c1c;font-weight:700} .o{color:#c2410c;font-weight:700}
.sub2{font-size:10.5px;letter-spacing:.06em;color:#6b7280;font-weight:600;margin:12px 0 2px}
.segw{display:flex;border:1px solid #d1d5db;border-radius:7px;overflow:hidden;font-size:12.5px;margin:8px 0}
.segw span{flex:1;text-align:center;padding:6px 4px;color:#6b7280} .segw span.on{background:#f3f4f6;color:#111827;font-weight:600}
.f3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px} .f3 label{display:block;font-size:11px;color:#6b7280;margin-bottom:3px}
.f3 .in{border:1px solid #d1d5db;border-radius:7px;padding:6px 8px;font-size:13px;background:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.meta{font-size:12px;color:#6b7280;margin-top:8px} .meta b{color:#111827;font-weight:600}
.link{color:#2563eb;font-size:13px;margin-top:12px;display:block}
.note{font-size:11.5px;color:#9ca3af;text-align:right;padding:6px 24px 0}
@media (max-width:760px){.page{grid-template-columns:1fr;padding:12px}.top{flex-wrap:wrap}.right{margin-left:0}.row{grid-template-columns:96px 1fr 50px}}
"""

def bar(label, pct, col):
    return f'<div class="row"><span>{label}</span><div class="trk"><b style="width:{min(pct,100)}%;background:{col}"></b></div><span class="v" style="color:{col}">{pct} %</span></div>'

RIGHT = f"""
<div class="card">
  <div class="h"><span class="t">RÉSULTAT</span><span class="pill nok">● Non conforme</span></div>
  <div style="font-size:13.5px">Flexion de <b>H3</b> sous le pied <b>P2</b> — ELU, γF 1,5 sur G + Q.</div>
  {bar("Flexion", 118, "#b91c1c")}{bar("Tranchant", 93, "#c2410c")}{bar("Pression locale", 48, "#2563eb")}{bar("Plateaux", 80, "#2563eb")}
  <div class="tip">💡 <span>Filière HEB 140 sous P2 → <b>79 %</b></span><span class="btn">Appliquer</span></div>
  <div class="tip" style="background:#fff;border-color:#e5e7eb">↔ <span>ou avancer l'engin de 0,40 m → <b>98 %</b></span><span class="btn" style="border-color:#d1d5db;color:#374151">Voir</span></div>
</div>
<div class="card"><div class="step"><span class="num g">1</span><div><div class="ttl">Plancher</div><div class="sub">Hourdis ERGON SP 200 + couche de compression 50 mm · L 7,20 m · 8 × 1,20 m · appuis A, B · 14 j</div></div><span class="chev">⌄</span></div></div>
<div class="card">
  <div class="step"><span class="num">2</span><div><div class="ttl">Charges</div><div class="sub">2 objets · 345 kN</div></div><span class="btn" style="margin-left:auto">＋ Ajouter</span></div>
  <div class="list">
    <div class="it on"><div><b>Télescopique rotatif MRT 2150</b><span class="sub">levage · 4 stabilisateurs · x 3,00 · y 5,60 m</span></div><span class="kn">310 kN</span></div>
    <div class="it"><div><b>Étançon E1</b><span class="sub">sous poutre de l'étage · x 6,30 · y 0,60 m</span></div><span class="kn">35 kN</span></div>
  </div>
  <div class="sub2">PIEDS DE L'ENGIN</div>
  <table><tr><th>Pied</th><th class="n">Charge</th><th>Répartiteur</th><th>Élém.</th><th class="n">Taux</th></tr>
  <tr style="background:#f8fafc"><td><b>P2</b></td><td class="n">95 kN</td><td>Plateau 600×600</td><td>H3</td><td class="n r">118 %</td></tr>
  <tr><td><b>P1</b></td><td class="n">95 kN</td><td>Plateau 600×600</td><td>H3</td><td class="n r">118 %</td></tr>
  <tr><td><b>P4</b></td><td class="n">60 kN</td><td>Plateau 600×600</td><td>H7</td><td class="n o">93 %</td></tr>
  <tr><td><b>P3</b></td><td class="n">60 kN</td><td>Plateau 600×600</td><td>H7</td><td class="n o">93 %</td></tr></table>
</div>
<div class="card">
  <div class="step"><span class="num">3</span><div><div class="ttl">Répartiteur sous P2</div><div class="sub">centré sur le pied · déplaçable</div></div><span class="chk" style="margin-left:auto;font-size:12.5px;white-space:nowrap"><i>✓</i> idem 4 pieds</span></div>
  <div class="segw"><span>Aucun</span><span class="on">Plateau</span><span>Filière</span><span>Plateau + filière</span></div>
  <div class="f3"><div><label>Matériau</label><div class="in">Acier S235 ▾</div></div><div><label>Dimensions (mm)</label><div class="in">600 × 600</div></div><div><label>Épaisseur (mm)</label><div class="in">20</div></div></div>
  <div class="meta">Aire efficace <b>540 × 540</b> · Pression <b>0,33 N/mm²</b> · Éléments touchés <b>1</b> · bibliothèque : OSB 18, bois 75, acier 20/30, plaque de grue</div>
</div>
<div class="card"><div class="step"><span class="num g">4</span><div><div class="ttl">Hypothèses de calcul</div><div class="sub">ELU · γF 1,5 · φ roues 1,40 · M_Rd / V_Rd catalogue avec couche de compression (Rd) · annexe C EN 1168</div></div><span class="chev">⌄</span></div></div>
<a class="link">Voir le détail des vérifications par élément →</a>
"""

PAGE = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Charges sur plancher</title><style>{CSS}</style></head><body>
<div class="top"><span style="font-size:18px">⛶</span><h1>Charges sur plancher</h1><span class="cas">CAS <span class="sel">C1 · Niveau +2 — zone grue ▾</span><span class="btn">＋</span></span>
<div class="right"><span class="btn">Ouvrir</span><span class="btn">Enregistrer</span><span class="btn dark">⬇ Note PDF</span></div></div>
<div class="page">
<div>
  <div class="card">
    <div class="bar"><span class="seg"><span class="on">Plan</span><span>Coupe</span><span>Balayage</span></span><span class="chk"><i>✓</i> Cotes</span><span class="chk"><i>✓</i> Aires efficaces</span><span class="chk"><i>✓</i> Taux par élément</span><span style="margin-left:auto;color:#9ca3af">éch. 1:60</span></div>
    {plan_svg()}
    <div class="legend"><span><i style="background:#111827"></i>semelle</span><span><i style="background:repeating-linear-gradient(45deg,#fff 0 2px,#9ca3af 2px 3px)"></i>plateau acier</span><span><i style="border:1px dashed #2563eb"></i>aire efficace</span><span><i style="background:#ffedd5;border-color:#fdba74"></i>85–100 %</span><span><i style="background:#fee2e2;border-color:#fca5a5"></i>&gt; 100 %</span><span style="margin-left:auto">● appui &nbsp; ○ bord libre &nbsp;·&nbsp; engin à l'échelle de sa fiche</span></div>
  </div>
  <div class="card">
    <div class="h"><span style="font-weight:600">Coupe transversale sous P2</span><span style="font-size:12px;color:#6b7280">élément H3 · hourdis 200 + couche de compression 50</span></div>
    {coupe_svg()}
  </div>
</div>
<div>{RIGHT}</div>
</div>
<div class="note">maquette d'étude · valeurs de calcul d'illustration · dimensions des engins : sources en ligne du 02/11/2026, à confirmer sur les fiches officielles</div>
</body></html>"""
(OUT / "ecran.html").write_text(PAGE, encoding="utf-8")

# ------------------------------------------------------------------ bibliothèque
CSS2 = CSS + """
.lib{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.eng{display:flex;gap:12px;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;align-items:center;background:#fff}
.eng.on{border-color:#2563eb;box-shadow:0 0 0 2px #dbeafe}
.eng .nm{font-weight:600;font-size:13.5px} .eng .d{font-size:12px;color:#6b7280;margin-top:2px;line-height:1.5}
.src{font-size:10.5px;border-radius:999px;padding:2px 8px;vertical-align:1px} .src.ok{background:#dcfce7;color:#15803d} .src.gen{background:#f3f4f6;color:#6b7280} .src.todo{background:#fef3c7;color:#92400e}
.form{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.form label{display:block;font-size:11px;color:#6b7280;margin-bottom:3px} .form .in{border:1px solid #d1d5db;border-radius:7px;padding:6px 8px;font-size:13px;background:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.form .in.todo{border-color:#f59e0b;background:#fffbeb}
.tabs{display:flex;gap:6px;margin-top:4px} .tabs span{border:1px solid #d1d5db;border-radius:999px;padding:4px 12px;font-size:12.5px;color:#6b7280} .tabs span.on{background:#111827;color:#fff;border-color:#111827}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px 24px 28px;align-items:start}
"""
def fiche(kind, nom, d, badge_cls, badge, on=False):
    return f'<div class="eng {"on" if on else ""}">{picto(kind)}<div><div class="nm">{nom}</div><div class="d">{d}<br><span class="src {badge_cls}">{badge}</span></div></div></div>'

LIB = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Bibliothèque</title><style>{CSS2}</style></head><body>
<div class="top"><span style="font-size:18px">⛶</span><h1>Charges sur plancher</h1><span class="cas">CAS <span class="sel">C1 · Niveau +2 — zone grue ▾</span></span><div class="right"><span class="btn">Fermer</span></div></div>
<div class="two">
<div class="card">
  <div class="h"><span style="font-weight:600;font-size:15px">Bibliothèque — ajouter une charge</span><span class="btn">＋ Nouvelle fiche</span></div>
  <div class="tabs"><span class="on">Engins</span><span>Grues</span><span>Étais et filières</span><span>Stocks</span><span>Plateaux</span></div>
  <div class="lib">
    {fiche("mrt", "Télescopique rotatif MRT 2150", "4 stabilisateurs · emprise 5,55 × 5,19 m · largeur 2,49 m · semelle 300 × 300", "ok", "dimensions : brochure Manitou (en ligne) · charges par pied : à relire", True)}
    {fiche("mt", "Télescopique MT 1840", "4 roues 400/80-24 · 6,27 × 2,42 m · empattement 3,07 m · 11,8 t · φ 1,40", "ok", "dimensions : manitou.com · charges par roue : à relire")}
    {fiche("ara", "Grue araignée (mini-grue)", "4 stabilisateurs · chenilles · emprise et charges par pied selon la fiche (Unic, Maeda, Jekko…)", "gen", "gabarit générique · à compléter d'une fiche")}
    {fiche("nac", "Nacelle ciseaux 12 m", "4 roues pleines · 2,4 × 1,2 m · φ 2,00 · gabarit", "gen", "générique · à compléter d'une fiche")}
    {fiche("etancon", "Étançon", "charge ponctuelle · semelle 150 × 150 · charge saisie", "gen", "générique")}
    {fiche("stock", "Stock de palettes", "charge surfacique · 15 kN/m² · emprise saisie", "gen", "générique")}
  </div>
  <div class="meta" style="margin-top:12px">Chaque fiche porte ses dimensions, ses charges par appui et sa <b>source</b> (document, date, page). Les dimensions ci-dessus ont été relevées en ligne le 02/11/2026 sur des sources secondaires : elles servent au dessin à l'échelle et se confirment sur la fiche officielle avant usage. Les fiches du bureau sont versionnées avec l'application ; une fiche ajoutée dans un cas reste dans son fichier. La bibliothèque livrée ne contient que des engins de plancher ; une grue mobile se crée comme fiche utilisateur, pour une dalle conçue pour la recevoir.</div>
</div>
<div>
  <div class="card">
    <div class="h"><span style="font-weight:600;font-size:15px">Fiche — Télescopique rotatif MRT 2150</span><span class="src ok" style="font-size:11.5px">brochure Manitou MRT 2150 Privilege Plus · relevé en ligne 02/11/2026</span></div>
    <div class="form">
      <div><label>Nom</label><div class="in">Télescopique rotatif MRT 2150</div></div><div><label>Type d'appuis</label><div class="in">Stabilisateurs ▾</div></div><div><label>Nombre</label><div class="in">4</div></div><div><label>φ dynamique</label><div class="in">1,00</div></div>
      <div><label>Largeur hors tout (m)</label><div class="in">2,49</div></div><div><label>Longueur châssis (m)</label><div class="in">5,35</div></div><div><label>Empattement (m)</label><div class="in">3,05</div></div><div><label>Emprise stabilisateurs (m)</label><div class="in">5,55 × 5,19</div></div>
      <div><label>Semelle (mm)</label><div class="in">300 × 300</div></div><div><label>Charge max / pied AV (kN)</label><div class="in todo">à relire</div></div><div><label>Charge max / pied AR (kN)</label><div class="in todo">à relire</div></div><div><label>Source</label><div class="in">brochure Manitou, p. — ▾</div></div>
    </div>
    <div class="meta">Les charges par pied dépendent de l'opération (charge levée, portée, orientation) : elles se relisent sur le <b>tableau de stabilité</b> du constructeur et se saisissent par cas ; la fiche mémorise la dernière valeur et sa page. Un champ orange bloque le calcul tant qu'il n'est pas renseigné.</div>
  </div>
  <div class="card">
    <div class="step"><span class="num">1</span><div><div class="ttl">Plancher</div><div class="sub">déplié</div></div><span class="chev">⌃</span></div>
    <div class="segw"><span class="on">Hourdis</span><span>Dalle BA</span><span>Prédalle</span></div>
    <div class="form" style="grid-template-columns:repeat(3,1fr)">
      <div><label>Type (catalogue)</label><div class="in">ERGON SP 200 ▾</div></div><div><label>Portée L (m)</label><div class="in">7,20</div></div><div><label>Éléments</label><div class="in">8 × 1,20 m</div></div>
      <div><label>Couche de compression</label><div class="in">Oui · 50 mm ▾</div></div><div><label>Béton de la couche</label><div class="in">C25/30 · 14 j → C20/25 (t)</div></div><div><label>Charges présentes</label><div class="in">1,0 kN/m²</div></div>
    </div>
    <div class="meta">Catalogue avec couche de compression 50 mm : M_Rd <b>112 kN·m / él.</b> · V_Rd <b>104 kN / él.</b> · g <b>2,9 + 1,25 kN/m²</b> — nature <b>Rd</b> (fiche SP, édition à renseigner) · membrure p_adm 0,70 N/mm² · M_cr non renseigné. Sans couche : M_Rd 90, V_Rd 95. Appuis : bords A et B (clic sur un bord du plan pour changer).</div>
  </div>
</div>
</div>
<div class="note">maquette d'étude · valeurs de calcul d'illustration</div>
</body></html>"""
(OUT / "bibliotheque.html").write_text(LIB, encoding="utf-8")
print("ok")

# -*- coding: utf-8 -*-
"""Maquette haute fidélité de l'écran « Charges sur plancher » (HTML + SVG à
l'échelle), rendue par Chromium. Valeurs d'illustration : aucune n'est un
résultat de calcul."""
import sys, pathlib, math

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- géométrie
S = 78.0            # px par mètre
L, B = 7.20, 9.60   # portée (x, horizontale) et largeur (y, verticale) en m
NU, BU = 8, 1.20    # hourdis : H1 en haut, H8 en bas
OX, OY = 104.0, 54.0
W_SVG, H_SVG = 860, 905

def X(x): return OX + x * S
def Y(y): return OY + (B - y) * S

TAUX = {1: 39, 2: 58, 3: 71, 4: 118, 5: 88, 6: 64, 7: 83, 8: 93}
GOUV = {4: "flexion · P4", 8: "tranchant · P1", 5: "flexion · R1", 7: "flexion · P1"}
def coul(t):   return "#e6ffe6" if t <= 85 else ("#fff3cd" if t <= 100 else "#ffe6e6")
def coul_t(t): return "#1a7f37" if t <= 85 else ("#9a6700" if t <= 100 else "#b42318")

svg = []
A = svg.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_SVG} {H_SVG}" width="100%" '
  f'style="font-family:Inter,\'Source Sans Pro\',sans-serif;font-size:11px;">')
A('''<defs>
<pattern id="hachA" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
  <line x1="0" y1="0" x2="0" y2="8" stroke="#555" stroke-width="1"/></pattern>
<pattern id="hachEff" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
  <line x1="0" y1="0" x2="0" y2="6" stroke="#1d4ed8" stroke-width="0.9"/></pattern>
<pattern id="hachStock" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
  <line x1="0" y1="0" x2="0" y2="7" stroke="#7c3aed" stroke-width="0.8"/></pattern>
<pattern id="grille" width="78" height="78" patternUnits="userSpaceOnUse">
  <path d="M 78 0 L 0 0 0 78" fill="none" stroke="#9ca3af" stroke-width="0.5" stroke-dasharray="2 3"/></pattern>
<marker id="tick" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto">
  <path d="M 2 8 L 8 2" stroke="#1d4ed8" stroke-width="1.4"/></marker>
<marker id="fl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker>
</defs>''')

# fond + grille + hourdis
A(f'<rect x="{X(0)}" y="{Y(B)}" width="{L*S}" height="{B*S}" fill="url(#grille)"/>')
for i in range(NU):
    n = i + 1
    y_top = Y(B) + i * BU * S
    A(f'<rect x="{X(0)}" y="{y_top}" width="{L*S}" height="{BU*S}" fill="{coul(TAUX[n])}" fill-opacity="0.8" stroke="#6b7280" stroke-width="0.8"/>')
    A(f'<text x="{X(0)+6}" y="{y_top+12}" font-size="9.5" fill="#6b7280">H{n}</text>')
    # colonne des taux, hors dessin, à droite de l'appui B
    cx, cy = X(L) + 30, y_top + BU * S / 2
    A(f'<rect x="{cx}" y="{cy-10}" width="92" height="20" rx="10" fill="white" stroke="{coul_t(TAUX[n])}" stroke-width="1.1"/>')
    A(f'<text x="{cx+9}" y="{cy+4}" font-size="11.5" font-weight="600" fill="{coul_t(TAUX[n])}">H{n}</text>')
    A(f'<text x="{cx+84}" y="{cy+4}" font-size="11.5" font-weight="700" text-anchor="end" fill="{coul_t(TAUX[n])}">{TAUX[n]} %</text>')
    if n in GOUV:
        A(f'<text x="{cx+4}" y="{cy+24}" font-size="9" fill="#6b7280">{GOUV[n]}</text>')
for i in range(1, NU):   # joints
    yj = Y(B) + i * BU * S
    A(f'<line x1="{X(0)}" y1="{yj}" x2="{X(L)}" y2="{yj}" stroke="#374151" stroke-width="1.2" stroke-dasharray="6 3"/>')
for xa in (X(0) - 16, X(L)):   # appuis
    A(f'<rect x="{xa}" y="{Y(B)}" width="16" height="{B*S}" fill="url(#hachA)" stroke="#374151" stroke-width="1"/>')
A(f'<text x="{X(0)-8}" y="{Y(B)-6}" font-size="10.5" text-anchor="middle" fill="#374151">A · mur</text>')
A(f'<text x="{X(L)+8}" y="{Y(B)-6}" font-size="10.5" text-anchor="middle" fill="#374151">B · poutre</text>')

# cotes bleues
def cote_h(x1, x2, y, txt, dy=-5):
    A(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#1d4ed8" stroke-width="1" marker-start="url(#tick)" marker-end="url(#tick)"/>')
    A(f'<text x="{(x1+x2)/2}" y="{y+dy}" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">{txt}</text>')
def cote_v(y1, y2, x, txt, dx=-6):
    A(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#1d4ed8" stroke-width="1" marker-start="url(#tick)" marker-end="url(#tick)"/>')
    A(f'<text transform="translate({x+dx} {(y1+y2)/2}) rotate(-90)" font-size="12" font-weight="700" fill="#1d4ed8" text-anchor="middle">{txt}</text>')
def ext(x1, y1, x2, y2, dash=""):
    A(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#1d4ed8" stroke-width="0.6" {("stroke-dasharray=" + chr(34) + dash + chr(34)) if dash else ""}/>')
ext(X(0), Y(B) - 8, X(0), Y(B) - 36); ext(X(L), Y(B) - 8, X(L), Y(B) - 36)
cote_h(X(0), X(L), Y(B) - 30, "L = 7,20 m")
ext(X(0) - 20, Y(B), X(0) - 72, Y(B)); ext(X(0) - 20, Y(0), X(0) - 72, Y(0))
cote_v(Y(B), Y(0), X(0) - 66, "8 × 1,20 = 9,60 m")
for i in range(0, 8):   # règle x
    A(f'<line x1="{X(i)}" y1="{Y(0)}" x2="{X(i)}" y2="{Y(0)+6}" stroke="#6b7280" stroke-width="0.8"/>')
    A(f'<text x="{X(i)}" y="{Y(0)+17}" font-size="9" text-anchor="middle" fill="#6b7280">{i}</text>')
A(f'<text x="{X(L)+6}" y="{Y(0)+17}" font-size="9" fill="#6b7280">x (m)</text>')

def rect_m(x, y, w, h, **kw):
    attrs = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
    A(f'<rect x="{X(x)}" y="{Y(y+h)}" width="{w*S}" height="{h*S}" {attrs}/>')
def bulle(x_px, y_px, l1, l2=None, w=104):
    h = 30 if l2 else 18
    A(f'<rect x="{x_px}" y="{y_px}" width="{w}" height="{h}" rx="4" fill="white" stroke="#111827" stroke-width="0.8"/>')
    A(f'<text x="{x_px+6}" y="{y_px+12}" font-size="11" font-weight="700" fill="#111827">{l1}</text>')
    if l2:
        A(f'<text x="{x_px+6}" y="{y_px+24}" font-size="8.5" fill="#374151">{l2}</text>')

# ---------------------------------------------------------------- ③ stock (haut-gauche)
sx0, sy0, sw, sh = 0.30, 7.00, 1.80, 1.20
rect_m(sx0, sy0, sw, sh, fill="url(#hachStock)", stroke="#7c3aed", stroke_width=1.2)
A(f'<text x="{X(sx0)+6}" y="{Y(sy0+sh)+14}" font-size="10" font-weight="700" fill="#5b21b6">③ stock blocs · 15 kN/m²</text>')
A(f'<text x="{X(sx0)+6}" y="{Y(sy0+sh)+27}" font-size="9" fill="#5b21b6">1,80 × 1,20 m · 32 kN</text>')

# ---------------------------------------------------------------- ② MT 1840 en déplacement (le long de x)
mx, my = 4.35, 7.15
for yy in (my - 1.15 - 0.5, my + 1.15 - 0.5):      # plaques de roulage 3800 × 1000
    rect_m(mx - 1.9, yy, 3.8, 1.0, fill="#f1f5f9", stroke="#111827", stroke_width=1.3)
A(f'<text x="{X(0.35)}" y="{Y(my+1.15+0.5)-2}" font-size="9" fill="#374151">plaques de roulage acier</text>')
A(f'<text x="{X(0.35)}" y="{Y(my+1.15+0.5)+10}" font-size="9" fill="#374151">3800×1000×20, une par file →</text>')
rect_m(mx - 2.1, my - 1.15, 4.2, 2.3, rx=6, fill="#f8fafc", stroke="#111827", stroke_width=1.3)
rect_m(mx + 0.3, my + 0.35, 1.0, 0.75, fill="#cbd5e1", stroke="#111827", stroke_width=1)
A(f'<polygon points="{X(mx-2.0)},{Y(my+0.18)} {X(mx+1.4)},{Y(my+0.12)} {X(mx+1.4)},{Y(my-0.12)} {X(mx-2.0)},{Y(my-0.18)}" fill="#d1d5db" stroke="#111827" stroke-width="1"/>')
A(f'<text x="{X(mx-0.4)}" y="{Y(my)+4}" font-size="9" fill="#374151" text-anchor="middle">② MT 1840 · déplacement · φ 1,40</text>')
A(f'<line x1="{X(mx+2.2)}" y1="{Y(my+0.55)}" x2="{X(mx+2.75)}" y2="{Y(my+0.55)}" stroke="#111827" stroke-width="1.2" marker-end="url(#fl)"/>')
A(f'<text x="{X(mx+2.2)}" y="{Y(my+0.55)-6}" font-size="8.5" fill="#374151">sens de</text>')
A(f'<text x="{X(mx+2.2)}" y="{Y(my+0.55)+14}" font-size="8.5" fill="#374151">marche</text>')
ROUES = [("R1", 1.35, -1.15, 45), ("R2", 1.35, 1.15, 45), ("R3", -1.35, -1.15, 25), ("R4", -1.35, 1.15, 25)]
for nom, dx, dy, R in ROUES:
    rect_m(mx + dx - 0.30, my + dy - 0.19, 0.60, 0.38, fill="url(#hachEff)", stroke="#1d4ed8", stroke_width=0.8)
    rect_m(mx + dx - 0.26, my + dy - 0.15, 0.52, 0.30, fill="#111827")
    ly = Y(my + dy) - (49 if dy < 0 else 44)       # bulles hors des empreintes : dans le châssis (bas) / au-dessus (haut)
    bulle(X(mx + dx) - 52, ly, f"{nom} · {R} kN", "× φ 1,40 en mouvement")
A(f'<text x="{X(0.35)}" y="{Y(my-1.15)-8}" font-size="8.5" fill="#1d4ed8">file de roues sur le joint</text>')
A(f'<text x="{X(0.35)}" y="{Y(my-1.15)+3}" font-size="8.5" fill="#1d4ed8">H3 | H4 → règle du levier →</text>')

# ---------------------------------------------------------------- ① MRT en levage sur 4 pieds
cx, cy = 2.95, 3.10
rect_m(cx - 2.45, cy - 1.20, 4.90, 2.40, rx=6, fill="#f8fafc", stroke="#111827", stroke_width=1.4)
for dx in (-1.55, 1.55):          # roues délestées
    for dy in (-1.15, 1.15):
        rect_m(cx + dx - 0.16, cy + dy - 0.26, 0.32, 0.52, fill="#e5e7eb", stroke="#9ca3af", stroke_width=1, stroke_dasharray="3 2")
A(f'<circle cx="{X(cx-0.4)}" cy="{Y(cy)}" r="{0.78*S}" fill="#e5e7eb" stroke="#111827" stroke-width="1"/>')
A(f'<polygon points="{X(cx-0.4)},{Y(cy+0.3)} {X(cx+3.4)},{Y(cy+0.17)} {X(cx+3.4)},{Y(cy-0.17)} {X(cx-0.4)},{Y(cy-0.3)}" fill="#d1d5db" stroke="#111827" stroke-width="1"/>')
A(f'<text x="{X(cx+1.9)}" y="{Y(cy)+4}" font-size="9" fill="#374151" text-anchor="middle">flèche</text>')
rect_m(cx - 2.1, cy + 0.3, 0.95, 0.75, fill="#cbd5e1", stroke="#111827", stroke_width=1)
A(f'<text x="{X(cx)}" y="{Y(cy-1.20)+12}" font-size="9" fill="#374151" text-anchor="middle">① MRT 2150 · levage · roues délestées</text>')
PIEDS = [("P1 · AR-D", -2.35, -1.95, 60, "H7 | H8", (0.36, -0.52)), ("P2 · AV-D", 2.35, -1.95, 95, "H8", (0.36, -0.52)),
         ("P3 · AR-G", -2.35, 1.95, 60, "H4", (0.36, 0.19)), ("P4 · AV-G", 2.35, 1.95, 95, "H4", (0.36, -0.30))]
for nom, dx, dy, P, el, (bdx, bdy) in PIEDS:
    px, py = cx + dx, cy + dy
    A(f'<line x1="{X(cx+dx*0.62)}" y1="{Y(cy+dy*0.5)}" x2="{X(px)}" y2="{Y(py)}" stroke="#111827" stroke-width="5" stroke-linecap="round"/>')
    rect_m(px - 0.30, py - 0.30, 0.60, 0.60, fill="white", stroke="#111827", stroke_width=1.6)
    rect_m(px - 0.27, py - 0.27, 0.54, 0.54, fill="url(#hachEff)", stroke="#1d4ed8", stroke_width=0.8)
    rect_m(px - 0.15, py - 0.15, 0.30, 0.30, fill="#111827")
    bulle(X(px + bdx), Y(py + bdy), f"{nom} · {P} kN", f"plateau acier 600×600×20 · {el}", w=112)
# sélection
sx_, sy_, sw_, sh_ = cx - 2.75, cy - 2.35, 5.5, 4.7
rect_m(sx_, sy_, sw_, sh_, fill="none", stroke="#dc2626", stroke_width=1.2, stroke_dasharray="6 4")
for hx in (sx_, sx_ + sw_):
    for hy in (sy_, sy_ + sh_):
        A(f'<rect x="{X(hx)-4}" y="{Y(hy)-4}" width="8" height="8" fill="white" stroke="#dc2626" stroke-width="1.2"/>')
tbx, tby = X(sx_ + sw_) - 184, Y(sy_) + 4
A(f'<rect x="{tbx}" y="{tby}" width="168" height="24" rx="6" fill="white" stroke="#d1d5db"/>')
A(f'<text x="{tbx+8}" y="{tby+16}" font-size="11" fill="#374151">① sélectionné  ↻ 90°   ⧉ copier   ✕</text>')
# cotes de position de ①
ext(X(cx), Y(cy), X(cx), Y(0) + 32, "3 3")
cote_h(X(0), X(cx), Y(0) + 30, "x = 2,95", dy=14)
ext(X(cx), Y(cy), X(0) - 30, Y(cy), "3 3")
cote_v(Y(cy), Y(0), X(0) - 28, "y = 3,10", dx=-6)

# ---------------------------------------------------------------- légende + échelle
lx0, ly0 = X(0) - 16, Y(0) + 54
items = [("#e6ffe6", "#1a7f37", "≤ 85 %"), ("#fff3cd", "#9a6700", "85–100 %"), ("#ffe6e6", "#b42318", "> 100 %")]
xx = lx0
for fc, ec, t in items:
    A(f'<rect x="{xx}" y="{ly0}" width="14" height="12" fill="{fc}" stroke="{ec}"/><text x="{xx+18}" y="{ly0+10}" font-size="10" fill="#374151">{t}</text>')
    xx += 74
A(f'<rect x="{xx}" y="{ly0}" width="14" height="12" fill="url(#hachA)" stroke="#374151"/><text x="{xx+18}" y="{ly0+10}" font-size="10" fill="#374151">appui</text>'); xx += 58
A(f'<line x1="{xx}" y1="{ly0+6}" x2="{xx+14}" y2="{ly0+6}" stroke="#374151" stroke-width="1.2" stroke-dasharray="6 3"/><text x="{xx+18}" y="{ly0+10}" font-size="10" fill="#374151">joint clavetté</text>'); xx += 92
A(f'<rect x="{xx}" y="{ly0}" width="14" height="12" fill="#111827"/><text x="{xx+18}" y="{ly0+10}" font-size="10" fill="#374151">empreinte</text>'); xx += 76
A(f'<rect x="{xx}" y="{ly0}" width="14" height="12" fill="url(#hachEff)" stroke="#1d4ed8"/><text x="{xx+18}" y="{ly0+10}" font-size="10" fill="#374151">aire efficace (bande c)</text>'); xx += 140
A(f'<rect x="{xx}" y="{ly0}" width="14" height="12" fill="url(#hachStock)" stroke="#7c3aed"/><text x="{xx+18}" y="{ly0+10}" font-size="10" fill="#374151">surfacique</text>')
ex0 = X(L) - 2 * S
A(f'<rect x="{ex0}" y="{ly0+22}" width="{S}" height="6" fill="#111827"/><rect x="{ex0+S}" y="{ly0+22}" width="{S}" height="6" fill="white" stroke="#111827"/>')
A(f'<text x="{ex0}" y="{ly0+40}" font-size="9" fill="#374151">0</text><text x="{ex0+S}" y="{ly0+40}" font-size="9" text-anchor="middle" fill="#374151">1</text><text x="{ex0+2*S}" y="{ly0+40}" font-size="9" text-anchor="middle" fill="#374151">2 m</text>')
A(f'<text x="{ex0-10}" y="{ly0+30}" font-size="9" text-anchor="end" fill="#6b7280">échelle 1:75 à l\'impression · grille 1 m · clic = pose sur la grille 10 cm</text>')
A('</svg>')
PLAN = "\n".join(svg)

# ---------------------------------------------------------------- bandeau de balayage
bal = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 58" width="100%" style="font-family:Inter,sans-serif">']
for i in range(72):
    x0 = i * 0.1
    t = 60 + 62 * math.sin(math.pi * (x0 + 0.05) / L) ** 2
    bal.append(f'<rect x="{X(x0)}" y="16" width="{0.1*S+0.3}" height="14" fill="{"#86efac" if t <= 100 else "#fca5a5"}"/>')
bal.append(f'<rect x="{X(0)}" y="16" width="{L*S}" height="14" fill="none" stroke="#6b7280" stroke-width="0.8"/>')
bal.append(f'<line x1="{X(2.95)}" y1="10" x2="{X(2.95)}" y2="36" stroke="#dc2626" stroke-width="1.5"/>')
bal.append(f'<text x="{X(0)}" y="11" font-size="10" fill="#374151">Balayage de ① le long de la portée (pas 10 cm) — vert : taux ≤ 100 %, rouge : au-delà · zones admissibles : x ≤ 1,2 m et x ≥ 4,9 m</text>')
bal.append(f'<text x="{X(2.95)+5}" y="48" font-size="10" fill="#b42318" font-weight="700">position actuelle x = 2,95 · pire position x = 2,80 m (121 %)</text>')
bal.append('</svg>')
BALAYAGE = "\n".join(bal)

# ---------------------------------------------------------------- pictogramme de coupe
def coupe_hourdis():
    s = ['<svg viewBox="0 0 178 74" width="150" height="62" style="font-family:Inter,sans-serif"><defs><marker id="tk2" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 2 8 L 8 2" stroke="#1d4ed8" stroke-width="1.4"/></marker></defs>']
    s.append('<rect x="10" y="16" width="150" height="40" fill="#e5e7eb" stroke="#111827" stroke-width="1.2"/>')
    for i in range(5):
        s.append(f'<ellipse cx="{25+i*30}" cy="36" rx="11" ry="13" fill="white" stroke="#111827" stroke-width="1"/>')
    s.append('<line x1="10" y1="9" x2="160" y2="9" stroke="#1d4ed8" stroke-width="1" marker-start="url(#tk2)" marker-end="url(#tk2)"/>')
    s.append('<text x="85" y="7" font-size="8" fill="#1d4ed8" text-anchor="middle" font-weight="700">b = 1200</text>')
    s.append('<line x1="168" y1="16" x2="168" y2="56" stroke="#1d4ed8" stroke-width="1" marker-start="url(#tk2)" marker-end="url(#tk2)"/>')
    s.append('<text x="176" y="38" font-size="8" fill="#1d4ed8" font-weight="700" transform="rotate(-90 176 38)" text-anchor="middle">200</text>')
    s.append('<text x="85" y="69" font-size="8" fill="#374151" text-anchor="middle">membrure sup. 30 mm · 5 alvéoles · joint clavetté</text>')
    s.append('</svg>')
    return "".join(s)

CSS = """
*{box-sizing:border-box} body{margin:0;background:#fff;color:#111827;font-family:Inter,'Source Sans Pro',sans-serif;font-size:14px}
.page{max-width:1400px;margin:0 auto;padding:18px 28px 40px}
h1{font-size:28px;margin:0 0 2px} .sub{color:#6b7280;font-size:12px;margin-bottom:10px}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 12px}
.btn{border:1px solid #d1d5db;border-radius:8px;padding:6px 10px;background:#fff;font-size:13px}
.sel-wide{min-width:220px}
.status{border-left:5px solid #16a34a;background:#e6ffe6;padding:10px 14px;border-radius:8px;margin:6px 0 10px}
.status.nok{border-left-color:#dc2626;background:#ffe6e6}
.status b{font-size:15px} .chips{margin-top:4px;font-size:12px;color:#374151}
.chips span{display:inline-block;margin-right:10px} .chips .r{color:#b42318;font-weight:700} .chips .o{color:#9a6700;font-weight:700} .chips .g{color:#1a7f37;font-weight:700}
.cols{display:grid;grid-template-columns:1.15fr 1fr;gap:22px;align-items:start}
.opts{font-size:12px;color:#374151;margin-bottom:6px} .opts span{margin-right:14px;white-space:nowrap}
.bloc{display:grid;grid-template-columns:76px 1fr;gap:8px;align-items:end;margin:8px 0}
.bloc .t{font-size:10.5px;font-weight:700;letter-spacing:.04em;color:#374151;padding-bottom:9px;text-transform:uppercase}
.bloc .t.s{font-weight:400;color:#9ca3af;text-transform:none}
.row{display:flex;gap:8px;flex-wrap:wrap}
.w{flex:1 1 120px;min-width:120px}
.w label{display:block;font-size:11px;color:#374151;margin-bottom:3px}
.w .v{border:1px solid #d1d5db;border-radius:8px;padding:7px 9px;background:#f9fafb;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.w .v.sel{border-color:#dc2626;color:#dc2626;background:#fff;text-align:center}
.w .v.num{display:flex;justify-content:space-between} .w .v.num i{font-style:normal;color:#6b7280}
.seg{display:flex;border:1px solid #d1d5db;border-radius:8px;overflow:hidden;font-size:12.5px}
.seg span{flex:1;text-align:center;padding:6px 4px;border-right:1px solid #e5e7eb;color:#374151;white-space:nowrap}
.seg span:last-child{border-right:0} .seg span.on{color:#dc2626;font-weight:700;background:#fff5f5}
.badge{display:inline-block;font-size:9.5px;border:1px solid #9ca3af;border-radius:4px;padding:0 4px;margin-left:4px;color:#374151;vertical-align:middle}
.exp{border:1px solid #e5e7eb;border-radius:8px;padding:8px 12px;margin:6px 0;font-size:13px;color:#374151}
table{border-collapse:collapse;width:100%;font-size:12px;margin:4px 0 8px}
th{font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:.04em;text-align:left;padding:5px 5px;border-bottom:1px solid #d1d5db;font-weight:600}
td{padding:5px 5px;border-bottom:1px solid #eee;vertical-align:middle;white-space:nowrap}
td.n{text-align:right;font-variant-numeric:tabular-nums} th.n{text-align:right}
.tx{font-weight:700} .tx.r{color:#b42318} .tx.o{color:#9a6700} .tx.g{color:#1a7f37}
tr.selrow td{background:#fff5f5}
.tabs{display:flex;gap:6px;margin:10px 0 6px;flex-wrap:wrap} .tabs span{border:1px solid #d1d5db;border-radius:8px;padding:5px 10px;font-size:13px}
.tabs span.on{border-color:#dc2626;color:#dc2626}
h3{font-size:16px;margin:10px 0 6px}
.prop{background:#fffbe6;border:1px solid #f59e0b;border-radius:8px;padding:8px 12px;font-size:12.5px;margin:6px 0}
.note{font-size:11.5px;color:#6b7280}
.scroll{overflow-x:auto}
@media (max-width:700px){.cols{grid-template-columns:1fr}.page{padding:12px 12px 30px}h1{font-size:22px}.w{flex:1 1 150px}.bloc{grid-template-columns:1fr}.bloc .t{padding-bottom:2px}.sel-wide{min-width:0}}
"""

def w(label, val, cls="", badge=""):
    return f'<div class="w"><label>{label}</label><div class="v {cls}">{val}{("<span class=badge>"+badge+"</span>") if badge else ""}</div></div>'
def num(label, val, unit=""):
    return f'<div class="w"><label>{label}</label><div class="v num"><span>{val}</span><i>{unit}&nbsp; − &nbsp;+</i></div></div>'
def seg(opts, on):
    return '<div class="w" style="flex:2 1 260px"><label>&nbsp;</label><div class="seg">' + "".join(
        f'<span class="{"on" if o == on else ""}">{o}</span>' for o in opts) + '</div></div>'
def tx(v):
    c = "g" if v <= 85 else ("o" if v <= 100 else "r")
    return f'<span class="tx {c}">{v} %</span>'

RIGHT = f"""
<div class="bloc"><div class="t">Plancher</div><div class="row">
  {seg(["Hourdis", "Dalle BA", "Prédalle"], "Hourdis")}
  {seg(["Posés", "Clavetés", "Chapés"], "Clavetés")}
</div></div>
<div class="bloc"><div class="t s">↳ état</div><div class="row">
  {w("Clavetage", "14 j · C25/30 ▾")}{w("fck(t) retenu", "C20/25", badge="arrondi ↓")}{w("Appuis", "2 côtés (A, B)")}{w("Charges présentes", "1,0 kN/m²")}
</div></div>
<div class="bloc"><div class="t s">↳ élément</div><div class="row">
  {w("Type", "ERGON SP 200 ▾")}{num("Portée L", "7,20", "m")}{num("Éléments", "8", "")}{num("Largeur", "1,20", "m")}
</div></div>
<div class="bloc"><div class="t s">↳ catalogue</div><div class="row">
  <div class="w" style="flex:0 0 150px;min-width:150px"><label>Coupe (catalogue)</label>{coupe_hourdis()}</div>
  {w("M_Rd", "90 kN·m / él.", badge="Rd")}{w("V_Rd", "95 kN / él.", badge="Rd")}
</div></div>
<div class="bloc"><div class="t s">↳ local</div><div class="row">
  {w("g", "2,9 kN/m²")}{w("I", "1,05·10⁹ mm⁴", badge="fab.")}{w("Membrure p_adm", "0,70 N/mm²", badge="fab.")}{w("M_cr", "— non renseigné")}
</div></div>

<div class="bloc"><div class="t">Charges</div><div class="row">
  {w("Ajouter depuis la bibliothèque", "MRT 2150 (générique) ▾")}{w("Configuration", "levage · 4 pieds ▾")}{w("&nbsp;", "＋ Poser sur le plan", "sel")}
</div></div>
<table>
<tr><th>n°</th><th>objet</th><th>configuration</th><th class="n">x (m)</th><th class="n">y (m)</th><th class="n">angle</th><th class="n">φ</th><th class="n">total</th><th></th></tr>
<tr class="selrow"><td>①</td><td>MRT 2150 (générique)</td><td>levage · 4 pieds</td><td class="n">2,95</td><td class="n">3,10</td><td class="n">0°</td><td class="n">1,00</td><td class="n">310 kN</td><td>↻ ⧉ 🗑</td></tr>
<tr><td>②</td><td>MT 1840 (générique)</td><td>déplacement · 4 roues</td><td class="n">4,35</td><td class="n">7,15</td><td class="n">0°</td><td class="n">1,40</td><td class="n">140 kN</td><td>↻ ⧉ 🗑</td></tr>
<tr><td>③</td><td>Stock de blocs</td><td>surfacique 15 kN/m²</td><td class="n">1,20</td><td class="n">7,60</td><td class="n">0°</td><td class="n">1,00</td><td class="n">32 kN</td><td>↻ ⧉ 🗑</td></tr>
</table>

<div class="bloc"><div class="t">Appuis de ①</div><div class="row"><div class="note" style="padding-bottom:6px">Un répartiteur par pied, ou une filière sous plusieurs pieds. La charge par pied vient du tableau de stabilité de l'engin et se saisit par opération. Clic sur une ligne : le pied clignote sur le plan.</div></div></div>
<div class="scroll"><table>
<tr><th>pied</th><th class="n">charge</th><th>contact</th><th>répartiteur</th><th>aire efficace</th><th class="n">pression</th><th>élément</th><th class="n">taux</th></tr>
<tr><td>P4 · AV-G</td><td class="n">95 kN</td><td>300×300</td><td>acier 600×600×20 ▾</td><td>540×540 · c 120</td><td class="n">0,33 N/mm²</td><td>H4</td><td class="n">{tx(118)}</td></tr>
<tr><td>P2 · AV-D</td><td class="n">95 kN</td><td>300×300</td><td>acier 600×600×20 ▾</td><td>540×540 · c 120</td><td class="n">0,33 N/mm²</td><td>H8</td><td class="n">{tx(93)}</td></tr>
<tr><td>P3 · AR-G</td><td class="n">60 kN</td><td>300×300</td><td>acier 600×600×20 ▾</td><td>540×540 · c 120</td><td class="n">0,21 N/mm²</td><td>H4</td><td class="n">{tx(118)}</td></tr>
<tr><td>P1 · AR-D</td><td class="n">60 kN</td><td>300×300</td><td>acier 600×600×20 ▾</td><td>540×540 · c 120</td><td class="n">0,21 N/mm²</td><td>H7 | H8 (levier)</td><td class="n">{tx(93)}</td></tr>
</table></div>
<div class="prop">⚙ <b>Dimensionner</b> — les plateaux règlent le local (membrure 48 %), pas la flexion de H4 : proposition <b>filière HEB 140 × 3,60 m sous P4</b>, en travers de H3–H5 → H4 118 % → 79 % · filière 61 % · semelle 0,28 N/mm². <span class="note">Alternative sans filière : avancer ① à x ≥ 4,9 m (balayage) → 98 %.</span></div>

<div class="bloc"><div class="t">Efforts</div><div class="row">
  {w("Méthode hourdis", "ELU · γF 1,5 sur G + Q ▾")}{w("φ roues en mouvement", "1,40 · EN 1991-1-1")}{w("Tranchant", "100 % élément chargé ▾")}{w("Répartition flexion", "annexe C EN 1168 ▾")}
</div></div>
<div class="exp">› Paramètres avancés — source du catalogue (fiche SP, édition à renseigner), nature des valeurs, largeur efficace plafonnée, parts annexe C</div>
<div class="exp">› Cartouche (note et plan)</div>
"""

RESULTS = f"""
<div class="tabs"><span class="on">Vérifications</span><span>Appuis de l'engin</span><span>Balayage</span><span>Note</span><span>Méthode</span></div>
<h3>Éléments — flexion et tranchant (ELU, γF 1,5 sur G + Q, M_Rd / V_Rd du catalogue)</h3>
<table>
<tr><th>élément</th><th>charges reçues</th><th class="n">M_Ed</th><th class="n">M_Rd</th><th class="n">η_M</th><th class="n">V_Ed</th><th class="n">V_Rd</th><th class="n">η_V</th><th>gouverne</th></tr>
<tr><td>H4</td><td>P4 (part C.1 52 %) + P3 (52 %) + R1, R3 ×1,40 (levier 50 %) + g + 1,0</td><td class="n">106,2</td><td class="n">90,0</td><td class="n">{tx(118)}</td><td class="n">58,1</td><td class="n">95,0</td><td class="n">{tx(61)}</td><td>flexion sous P4</td></tr>
<tr><td>H8</td><td>P2 (52 %) + P1 (levier 50 %) + g + 1,0</td><td class="n">72,1</td><td class="n">90,0</td><td class="n">{tx(80)}</td><td class="n">88,4</td><td class="n">95,0</td><td class="n">{tx(93)}</td><td>tranchant · P1 à 0,60 m de A</td></tr>
<tr><td>H5</td><td>P4 (24 %) + P3 (24 %) + R1, R3 ×1,40 (levier 50 %) + g</td><td class="n">79,2</td><td class="n">90,0</td><td class="n">{tx(88)}</td><td class="n">47,3</td><td class="n">95,0</td><td class="n">{tx(50)}</td><td>flexion sous R1</td></tr>
</table>
<div class="exp">› 5 autres éléments (H1, H2, H3, H6, H7) — tous ≤ 83 %</div>
<h3>Répartiteurs et effets locaux</h3>
<table>
<tr><th>vérification</th><th>objet</th><th class="n">E</th><th class="n">R</th><th class="n">η</th><th>réf.</th></tr>
<tr><td>Plateau — bande efficace c <span class="badge">EC3</span></td><td>acier 20 mm sous P4</td><td class="n">c = 120 mm</td><td class="n">(600−300)/2 = 150</td><td class="n">{tx(80)}</td><td>EN 1993-1-8 §6.2.5 : plateau efficace à 80 %, aire 540 × 540</td></tr>
<tr><td>Membrure supérieure <span class="badge">fab.</span></td><td>H4 sous P4</td><td class="n">0,33 N/mm²</td><td class="n">0,70 N/mm²</td><td class="n">{tx(48)}</td><td>valeur locale admissible du catalogue</td></tr>
<tr><td>Plaque de roulage — bande c <span class="badge">EC3</span></td><td>acier 20 mm sous R1</td><td class="n">c = 41 mm</td><td class="n">—</td><td class="n">—</td><td>aire efficace 602 × 382 : la plaque ne répartit pas la roue, elle protège la membrure</td></tr>
<tr><td>Joint longitudinal <span class="badge">V1.1</span></td><td>H3 | H4</td><td class="n">—</td><td class="n">—</td><td class="n">—</td><td>EN 1168, contrainte de cisaillement du joint : non vérifié en V1</td></tr>
</table>
"""

HTML = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Charges sur plancher — maquette</title><style>{CSS}</style></head>
<body><div class="page">
<h1>Charges sur plancher</h1>
<div class="sub">Charges temporaires de chantier sur hourdis, dalle ou prédalle – maquette d'étude, valeurs d'illustration</div>
<div class="toolbar"><span class="btn">‹ Projet</span><span class="btn">🏠 Accueil</span><span class="btn">💾 Enregistrer</span><span class="btn">📂 Ouvrir</span><span class="btn">📄 PDF</span><span class="btn sel-wide">C1 · Niveau +2 — hourdis SP 200, zone grue ▾</span><span class="btn">＋ Cas</span><span class="btn">⧉</span><span class="btn">🔒</span></div>
<div class="status nok"><b>❌ PLANCHER NON CONFORME — H4 flexion 118 % sous le pied P4 de ①</b> — ELU, γF 1,5 sur G + Q, M_Rd du catalogue (nature Rd) · proposition : filière HEB 140 × 3,60 m sous P4 → 79 %
<div class="chips"><span>Flexion <span class="r">118 %</span></span><span>Tranchant <span class="o">93 %</span></span><span>Local membrure <span class="g">48 %</span></span><span>Plateaux <span class="g">80 %</span></span><span>Balayage de ① : pire 121 % à x = 2,80 m · admissible x ≤ 1,2 ou x ≥ 4,9 m</span></div></div>
<div class="cols">
<div>
<div class="opts"><span>◯ Vue simple</span><span>◉ Édition</span><span>☑ Cotations</span><span>☑ Charges sur le plan</span><span>☑ Aires efficaces</span><span>☑ Taux par élément</span><span>Poser : <b>① MRT</b> ▾</span></div>
{PLAN}
{BALAYAGE}
<div class="note">Vue en plan à l'échelle. Chaque pied ou roue porte sa charge et son répartiteur ; l'aire efficace (bande c) est ce que le plateau transmet réellement. Clic sur un objet : sélection, poignées, ↻ 90°, copier, supprimer. Clic sur une ligne des tableaux : l'objet clignote.</div>
</div>
<div>{RIGHT}</div>
</div>
{RESULTS}
</div></body></html>"""
(OUT / "maquette.html").write_text(HTML, encoding="utf-8")
print("ok", OUT / "maquette.html")

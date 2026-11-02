# -*- coding: utf-8 -*-
"""Maquette « digeste » de l'écran Charges sur plancher : plan épuré, étapes
repliées à droite, plateau posé sous le pied, bibliothèque d'engins.
HTML + SVG rendus par Chromium. Valeurs d'illustration."""
import sys, pathlib

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ plan
S = 60.0                          # px / m
L, B, NU, BU = 7.20, 9.60, 8, 1.20
OX, OY = 118.0, 58.0              # coin haut-gauche du plancher
def X(x): return OX + x * S
def Yt(yt): return OY + yt * S    # yt mesuré depuis le haut du plan (m)

TAUX = {1: 22, 2: 41, 3: 118, 4: 64, 5: 58, 6: 93, 7: 71, 8: 30}
ROUGE, ROUGE_F, ORANGE, ORANGE_F, BLEU, NOIR, GRIS, LIGNE = "#b91c1c", "#fee2e2", "#c2410c", "#ffedd5", "#2563eb", "#111827", "#6b7280", "#d1d5db"

def plan_svg():
    A = []
    a = A.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 736" width="100%" style="font-family:Inter,sans-serif;font-size:12px">')
    a('''<defs>
<pattern id="hp" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#4b5563" stroke-width="1"/></pattern>
<pattern id="hc" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#9ca3af" stroke-width="0.8"/></pattern>
<marker id="tk" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 5 1 L 5 9" stroke="#374151" stroke-width="1.4"/></marker>
</defs>''')
    # bandes
    for i in range(NU):
        n = i + 1
        y0 = Yt(i * BU)
        fill = ROUGE_F if TAUX[n] > 100 else (ORANGE_F if TAUX[n] >= 85 else "white")
        a(f'<rect x="{X(0)}" y="{y0}" width="{L*S}" height="{BU*S}" fill="{fill}"/>')
        col = ROUGE if TAUX[n] > 100 else (ORANGE if TAUX[n] >= 85 else GRIS)
        a(f'<text x="{X(0)+10}" y="{y0+16}" font-size="12" fill="{col}" font-weight="{600 if TAUX[n] >= 85 else 400}">H{n}</text>')
        # taux à droite
        tx, ty = X(L) + 26, y0 + BU * S / 2 + 4
        if TAUX[n] > 100:
            a(f'<rect x="{tx-4}" y="{ty-13}" width="58" height="19" rx="9.5" fill="{ROUGE}"/><text x="{tx+25}" y="{ty+1}" font-size="12.5" font-weight="700" fill="white" text-anchor="middle">{TAUX[n]} %</text>')
        elif TAUX[n] >= 85:
            a(f'<rect x="{tx-4}" y="{ty-13}" width="54" height="19" rx="9.5" fill="white" stroke="{ORANGE}" stroke-width="1.2"/><text x="{tx+23}" y="{ty+1}" font-size="12.5" font-weight="700" fill="{ORANGE}" text-anchor="middle">{TAUX[n]} %</text>')
        else:
            a(f'<text x="{tx}" y="{ty}" font-size="12.5" fill="{GRIS}">{TAUX[n]} %</text>')
    for i in range(1, NU):   # joints
        y = Yt(i * BU)
        a(f'<line x1="{X(0)}" y1="{y}" x2="{X(L)}" y2="{y}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="5 4"/>')
    # contour : appuis = trait noir épais (A, B) ; bords libres = trait fin gris
    a(f'<line x1="{X(0)}" y1="{Yt(0)}" x2="{X(L)}" y2="{Yt(0)}" stroke="{LIGNE}" stroke-width="1.2"/>')
    a(f'<line x1="{X(0)}" y1="{Yt(B)}" x2="{X(L)}" y2="{Yt(B)}" stroke="{LIGNE}" stroke-width="1.2"/>')
    a(f'<line x1="{X(0)}" y1="{Yt(0)-2}" x2="{X(0)}" y2="{Yt(B)+2}" stroke="{NOIR}" stroke-width="5"/>')
    a(f'<line x1="{X(L)}" y1="{Yt(0)-2}" x2="{X(L)}" y2="{Yt(B)+2}" stroke="{NOIR}" stroke-width="5"/>')
    a(f'<text x="{X(0)}" y="{Yt(0)-10}" font-size="12" font-weight="700" fill="{NOIR}" text-anchor="middle">A</text>')
    a(f'<text x="{X(L)}" y="{Yt(0)-10}" font-size="12" font-weight="700" fill="{NOIR}" text-anchor="middle">B</text>')
    # poignées de bord (clic : appui / libre)
    for (x, y, on) in ((X(0), Yt(B/2), True), (X(L), Yt(B/2), True), (X(L/2), Yt(0), False), (X(L/2), Yt(B), False)):
        a(f'<circle cx="{x}" cy="{y}" r="6" fill="{NOIR if on else "white"}" stroke="{NOIR}" stroke-width="1.4"/>')
    a(f'<text x="{X(L/2)+12}" y="{Yt(0)-6}" font-size="10.5" fill="{GRIS}">bord libre · clic = appui</text>')
    # cotes fines
    a(f'<line x1="{X(0)}" y1="{Yt(0)-34}" x2="{X(L)}" y2="{Yt(0)-34}" stroke="#374151" stroke-width="1" marker-start="url(#tk)" marker-end="url(#tk)"/>')
    a(f'<text x="{X(L/2)}" y="{Yt(0)-39}" font-size="12" fill="#374151" text-anchor="middle">7,20 m</text>')
    a(f'<line x1="{X(0)-40}" y1="{Yt(0)}" x2="{X(0)-40}" y2="{Yt(B)}" stroke="#374151" stroke-width="1" marker-start="url(#tk)" marker-end="url(#tk)"/>')
    a(f'<text transform="translate({X(0)-46} {Yt(B/2)}) rotate(-90)" font-size="12" fill="#374151" text-anchor="middle">8 × 1,20 = 9,60 m</text>')
    # ---- grue araignée MRT : centre (3,00 ; 4,80 depuis le haut), pieds à ±1,6 / ±1,8
    cx, cy = 3.00, 4.80
    def R(x, y, w, h, **kw):
        at = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
        a(f'<rect x="{X(x)}" y="{Yt(y)}" width="{w*S}" height="{h*S}" {at}/>')
    # chenilles / châssis
    R(cx - 1.15, cy - 0.55, 2.3, 0.18, fill="url(#hp)", stroke=NOIR, stroke_width=1)
    R(cx - 1.15, cy + 0.37, 2.3, 0.18, fill="url(#hp)", stroke=NOIR, stroke_width=1)
    R(cx - 1.0, cy - 0.37, 2.0, 0.74, fill="white", stroke=NOIR, stroke_width=1.1)
    a(f'<circle cx="{X(cx-0.3)}" cy="{Yt(cy)}" r="{0.42*S}" fill="white" stroke="{NOIR}" stroke-width="1.1"/>')
    a(f'<circle cx="{X(cx-0.3)}" cy="{Yt(cy)}" r="2.5" fill="{NOIR}"/>')
    a(f'<rect x="{X(cx+0.05)}" y="{Yt(cy)-5}" width="{3.2*S}" height="10" rx="3" fill="white" stroke="{NOIR}" stroke-width="1.1"/>')
    a(f'<circle cx="{X(cx+3.3)}" cy="{Yt(cy)}" r="5" fill="white" stroke="{NOIR}" stroke-width="1.1"/>')
    a(f'<text x="{X(cx)}" y="{Yt(cy+0.78)+14}" font-size="11" fill="{GRIS}" text-anchor="middle">MRT 2150 · levage</text>')
    PIEDS = [("P1", -1.6, -1.8, 95, False), ("P2", 1.6, -1.8, 95, True), ("P3", -1.6, 1.8, 60, False), ("P4", 1.6, 1.8, 60, False)]
    for nom, dx, dy, P, sel in PIEDS:
        px, py = cx + dx, cy + dy
        a(f'<line x1="{X(cx+dx*0.62)}" y1="{Yt(cy+dy*0.3)}" x2="{X(px)}" y2="{Yt(py)}" stroke="{NOIR}" stroke-width="2.2"/>')
        a(f'<line x1="{X(cx+dx*0.62)}" y1="{Yt(cy+dy*0.3)}" x2="{X(px)}" y2="{Yt(py)}" stroke="white" stroke-width="0.8"/>')
        # aire efficace (pointillé bleu), plateau (hachure), semelle (noir)
        R(px - 0.27, py - 0.27, 0.54, 0.54, fill="none", stroke=BLEU, stroke_width=1, stroke_dasharray="3 2")
        R(px - 0.30, py - 0.30, 0.60, 0.60, fill="url(#hc)", stroke=NOIR, stroke_width=1.1)
        R(px - 0.15, py - 0.15, 0.30, 0.30, fill=NOIR)
        if sel:
            R(px - 0.36, py - 0.36, 0.72, 0.72, fill="none", stroke=BLEU, stroke_width=1.8, rx=3)
        # étiquette
        lx, ly = X(px) - 0.4 * S, Yt(py) + (-0.42 * S - 20 if dy < 0 else 0.42 * S + 2)
        if sel:
            a(f'<rect x="{lx}" y="{ly}" width="82" height="20" rx="3" fill="{BLEU}"/><text x="{lx+8}" y="{ly+14}" font-size="11.5" font-weight="600" fill="white">{nom} · {P} kN</text>')
        else:
            a(f'<rect x="{lx}" y="{ly}" width="82" height="20" rx="3" fill="white" stroke="{LIGNE}"/><text x="{lx+8}" y="{ly+14}" font-size="11.5" fill="{NOIR}"><tspan font-weight="600">{nom}</tspan> · {P} kN</text>')
    # filière proposée sous P2 (en travers de H2–H4)
    fx = cx + 1.6
    R(fx - 0.07, 1.2 + 0.05, 0.14, 3.6, fill="none", stroke=BLEU, stroke_width=1.2, stroke_dasharray="6 3")
    a(f'<text x="{X(fx)-10}" y="{Yt(1.2)+16}" font-size="11" fill="{BLEU}" text-anchor="end">filière HEB 140 · 3,60 m (proposée)</text>')
    # popover sur le plateau sélectionné (clic sur le plateau)
    ppx, ppy = X(cx + 1.6) + 0.46 * S, Yt(cy - 1.8) + 0.40 * S
    a(f'<rect x="{ppx}" y="{ppy}" width="238" height="70" rx="8" fill="white" stroke="{LIGNE}" filter="drop-shadow(0 2px 4px rgba(0,0,0,.12))"/>')
    a(f'<text x="{ppx+12}" y="{ppy+20}" font-size="11.5" font-weight="700" fill="{NOIR}">Répartiteur sous P2</text>')
    a(f'<rect x="{ppx+12}" y="{ppy+30}" width="214" height="24" rx="5" fill="#f3f4f6" stroke="{LIGNE}"/>')
    a(f'<text x="{ppx+20}" y="{ppy+46}" font-size="11.5" fill="{NOIR}">Plateau acier 600 × 600 × 20</text><text x="{ppx+216}" y="{ppy+46}" font-size="11" fill="{GRIS}" text-anchor="end">▾</text>')
    a(f'<text x="{ppx+12}" y="{ppy+64}" font-size="10.5" fill="{GRIS}">centré sur le pied · glisser pour décaler · ✕ retirer</text>')
    # étançon E1
    ex, ey = 6.20, 7.80
    a(f'<circle cx="{X(ex)}" cy="{Yt(ey)}" r="11" fill="white" stroke="{NOIR}" stroke-width="1.4"/>')
    a(f'<line x1="{X(ex)-7}" y1="{Yt(ey)-7}" x2="{X(ex)+7}" y2="{Yt(ey)+7}" stroke="{NOIR}" stroke-width="1.2"/><line x1="{X(ex)-7}" y1="{Yt(ey)+7}" x2="{X(ex)+7}" y2="{Yt(ey)-7}" stroke="{NOIR}" stroke-width="1.2"/>')
    a(f'<rect x="{X(ex)-74}" y="{Yt(ey)+16}" width="86" height="20" rx="3" fill="white" stroke="{LIGNE}"/><text x="{X(ex)-66}" y="{Yt(ey)+30}" font-size="11.5" fill="{NOIR}"><tspan font-weight="600">E1</tspan> · 35 kN</text>')
    # échelle
    sx, sy = X(0), Yt(B) + 30
    a(f'<rect x="{sx}" y="{sy}" width="{S}" height="5" fill="{NOIR}"/><rect x="{sx+S}" y="{sy}" width="{S}" height="5" fill="white" stroke="{NOIR}"/>')
    for i, t in enumerate(("0", "1", "2 m")):
        a(f'<text x="{sx+i*S}" y="{sy+18}" font-size="10.5" fill="{GRIS}" text-anchor="middle">{t}</text>')
    a('</svg>')
    return "\n".join(A)

def coupe_svg():
    A = []; a = A.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 232" width="100%" style="font-family:Inter,sans-serif;font-size:12px">')
    a('<defs><pattern id="hc2" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#9ca3af" stroke-width="0.8"/></pattern>'
      '<marker id="tk2" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 5 1 L 5 9" stroke="#374151" stroke-width="1.4"/></marker>'
      '<marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/></marker></defs>')
    sc = 0.22                     # px / mm
    yt, h = 128, 200 * sc
    bu = 1200 * sc                # 264 px
    x3 = 450 - bu / 2             # H3 (chargé) centré
    for k, (x0, nom, fill) in enumerate(((x3 - bu - 4, "H2", "white"), (x3, "H3", ROUGE_F), (x3 + bu + 4, "H4", "white"))):
        a(f'<rect x="{x0}" y="{yt}" width="{bu}" height="{h}" fill="{fill}" stroke="{NOIR}" stroke-width="1.2"/>')
        for i in range(5):
            a(f'<circle cx="{x0+bu*(i+0.5)/5}" cy="{yt+h/2}" r="14" fill="white" stroke="{NOIR}" stroke-width="1"/>')
        a(f'<text x="{x0+bu/2}" y="{yt-8}" font-size="12" fill="{ROUGE if nom == "H3" else GRIS}" font-weight="{600 if nom == "H3" else 400}" text-anchor="middle">{nom}</text>')
    xc = 450
    a(f'<rect x="{xc-66}" y="{yt-6}" width="132" height="6" fill="url(#hc2)" stroke="{NOIR}" stroke-width="1"/>')
    a(f'<rect x="{xc-33}" y="{yt-6-14}" width="66" height="14" fill="{NOIR}"/>')
    a(f'<line x1="{xc}" y1="{yt-82}" x2="{xc}" y2="{yt-26}" stroke="{NOIR}" stroke-width="2" marker-end="url(#ar2)"/>')
    a(f'<text x="{xc+10}" y="{yt-56}" font-size="13" font-weight="600" fill="{NOIR}">95 kN</text>')
    a(f'<text x="{xc+76}" y="{yt-10}" font-size="11.5" fill="{GRIS}">plateau 600 × 20</text>')
    # diffusion à 45° jusqu'au plan moyen depuis l'aire efficace (540)
    e = 540 * sc / 2
    for sgn in (-1, 1):
        a(f'<line x1="{xc+sgn*e}" y1="{yt}" x2="{xc+sgn*(e+h/2)}" y2="{yt+h/2}" stroke="{BLEU}" stroke-width="1" stroke-dasharray="4 3"/>')
    a(f'<line x1="{xc-e}" y1="{yt+h+22}" x2="{xc+e}" y2="{yt+h+22}" stroke="{BLEU}" stroke-width="1" marker-start="url(#tk2)" marker-end="url(#tk2)"/>')
    a(f'<text x="{xc}" y="{yt+h+40}" font-size="12.5" fill="{BLEU}" text-anchor="middle">aire efficace 540 mm</text>')
    a(f'<line x1="{x3+2*bu+4+14}" y1="{yt}" x2="{x3+2*bu+4+14}" y2="{yt+h}" stroke="#374151" stroke-width="1" marker-start="url(#tk2)" marker-end="url(#tk2)"/>')
    a(f'<text x="{x3+2*bu+4+22}" y="{yt+h/2+4}" font-size="11.5" fill="#374151">200</text>')
    a('</svg>')
    return "\n".join(A)

CSS = """
*{box-sizing:border-box} body{margin:0;background:#f5f6f8;color:#111827;font-family:Inter,'Source Sans Pro',sans-serif;font-size:13.5px}
.top{display:flex;align-items:center;gap:14px;background:#fff;border-bottom:1px solid #e5e7eb;padding:10px 24px}
.top h1{font-size:17px;margin:0;font-weight:600} .top .cas{display:flex;align-items:center;gap:8px;margin-left:12px;font-size:12px;color:#6b7280;letter-spacing:.06em}
.sel{border:1px solid #d1d5db;border-radius:7px;padding:6px 10px;background:#fff;font-size:13px;color:#111827;letter-spacing:0}
.btn{border:1px solid #d1d5db;border-radius:7px;padding:6px 12px;background:#fff;font-size:13px}
.btn.dark{background:#111827;color:#fff;border-color:#111827} .right{margin-left:auto;display:flex;gap:8px}
.page{display:grid;grid-template-columns:1fr 400px;gap:18px;padding:18px 24px 28px;align-items:start}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px 14px}
.card+.card{margin-top:12px}
.bar{display:flex;align-items:center;gap:16px;font-size:12.5px;color:#374151;padding-bottom:8px;border-bottom:1px solid #f3f4f6;margin-bottom:6px}
.seg{display:inline-flex;border:1px solid #d1d5db;border-radius:7px;overflow:hidden;font-size:12.5px}
.seg span{padding:5px 12px;color:#6b7280} .seg span.on{background:#111827;color:#fff}
.chk{display:inline-flex;align-items:center;gap:6px} .chk i{display:inline-block;width:14px;height:14px;border-radius:3px;background:#2563eb;color:#fff;font-style:normal;font-size:11px;line-height:14px;text-align:center}
.legend{display:flex;gap:18px;font-size:11.5px;color:#6b7280;padding-top:8px;border-top:1px solid #f3f4f6;margin-top:4px}
.legend i{display:inline-block;width:14px;height:11px;vertical-align:-1px;margin-right:6px;border:1px solid #9ca3af}
.h{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.h .t{font-size:11px;letter-spacing:.08em;color:#6b7280;font-weight:600}
.pill{font-size:12px;border-radius:999px;padding:3px 10px;font-weight:600} .pill.nok{background:#fee2e2;color:#b91c1c} .pill.ok{background:#dcfce7;color:#15803d}
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
.r{color:#b91c1c;font-weight:700} .o{color:#c2410c;font-weight:700} .g{color:#15803d;font-weight:700}
.sub2{font-size:10.5px;letter-spacing:.06em;color:#6b7280;font-weight:600;margin:12px 0 2px}
.segw{display:flex;border:1px solid #d1d5db;border-radius:7px;overflow:hidden;font-size:12.5px;margin:8px 0}
.segw span{flex:1;text-align:center;padding:6px 4px;color:#6b7280} .segw span.on{background:#f3f4f6;color:#111827;font-weight:600}
.f3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px} .f3 label{display:block;font-size:11px;color:#6b7280;margin-bottom:3px}
.f3 .in{border:1px solid #d1d5db;border-radius:7px;padding:6px 8px;font-size:13px;background:#fff}
.meta{font-size:12px;color:#6b7280;margin-top:8px} .meta b{color:#111827;font-weight:600}
.link{color:#2563eb;font-size:13px;margin-top:12px;display:block}
.note{font-size:11.5px;color:#9ca3af;text-align:right;padding:6px 24px 0}
@media (max-width:760px){.page{grid-template-columns:1fr;padding:12px}.top{flex-wrap:wrap}.right{margin-left:0}.row{grid-template-columns:96px 1fr 50px}}
"""

def bar(label, pct, col):
    w = min(pct, 100)
    return f'<div class="row"><span>{label}</span><div class="trk"><b style="width:{w}%;background:{col}"></b></div><span class="v" style="color:{col}">{pct} %</span></div>'

RIGHT = f"""
<div class="card">
  <div class="h"><span class="t">RÉSULTAT</span><span class="pill nok">● Non conforme</span></div>
  <div style="font-size:13.5px">Flexion de <b>H3</b> sous le pied <b>P2</b> — ELU, γF 1,5 sur G + Q.</div>
  {bar("Flexion", 118, "#b91c1c")}{bar("Tranchant", 93, "#c2410c")}{bar("Pression locale", 48, "#2563eb")}{bar("Plateaux", 80, "#2563eb")}
  <div class="tip">💡 <span>Filière HEB 140 sous P2 → <b>79 %</b></span><span class="btn">Appliquer</span></div>
  <div class="tip" style="background:#fff;border-color:#e5e7eb">↔ <span>ou reculer la grue de 0,40 m → <b>98 %</b></span><span class="btn" style="border-color:#d1d5db;color:#374151">Voir</span></div>
</div>

<div class="card"><div class="step"><span class="num g">1</span><div><div class="ttl">Plancher</div><div class="sub">Hourdis ERGON SP 200 · L 7,20 m · 8 × 1,20 m · appuis A, B · claveté 14 j</div></div><span class="chev">⌄</span></div></div>

<div class="card">
  <div class="step"><span class="num">2</span><div><div class="ttl">Charges</div><div class="sub">2 objets · 345 kN</div></div><span class="btn" style="margin-left:auto">＋ Ajouter</span></div>
  <div class="list">
    <div class="it on"><div><b>Grue araignée MRT 2150</b><span class="sub">levage · 4 pieds · x 3,00 · y 4,80 m</span></div><span class="kn">310 kN</span></div>
    <div class="it"><div><b>Étançon E1</b><span class="sub">sous poutre de l'étage · x 6,20 · y 7,80 m</span></div><span class="kn">35 kN</span></div>
  </div>
  <div class="sub2">PIEDS DE LA GRUE</div>
  <table><tr><th>Pied</th><th class="n">Charge</th><th>Répartiteur</th><th>Élém.</th><th class="n">Taux</th></tr>
  <tr style="background:#f8fafc"><td><b>P2</b></td><td class="n">95 kN</td><td>Plateau 600×600</td><td>H3</td><td class="n r">118 %</td></tr>
  <tr><td><b>P1</b></td><td class="n">95 kN</td><td>Plateau 600×600</td><td>H3</td><td class="n r">118 %</td></tr>
  <tr><td><b>P4</b></td><td class="n">60 kN</td><td>Plateau 600×600</td><td>H6</td><td class="n o">93 %</td></tr>
  <tr><td><b>P3</b></td><td class="n">60 kN</td><td>Plateau 600×600</td><td>H6</td><td class="n o">93 %</td></tr></table>
</div>

<div class="card">
  <div class="step"><span class="num">3</span><div><div class="ttl">Répartiteur sous P2</div><div class="sub">centré sur le pied · déplaçable</div></div><span class="chk" style="margin-left:auto;font-size:12.5px;white-space:nowrap"><i>✓</i> idem 4 pieds</span></div>
  <div class="segw"><span>Aucun</span><span class="on">Plateau</span><span>Filière</span><span>Plateau + filière</span></div>
  <div class="f3"><div><label>Matériau</label><div class="in">Acier S235 ▾</div></div><div><label>Dimensions (mm)</label><div class="in">600 × 600</div></div><div><label>Épaisseur (mm)</label><div class="in">20</div></div></div>
  <div class="meta">Aire efficace <b>540 × 540</b> · Pression <b>0,33 N/mm²</b> · Éléments touchés <b>1</b> · bibliothèque : OSB 18, bois 75, acier 20/30, plaque de grue</div>
</div>

<div class="card"><div class="step"><span class="num g">4</span><div><div class="ttl">Hypothèses de calcul</div><div class="sub">ELU · γF 1,5 · φ roues 1,40 · M_Rd / V_Rd catalogue (Rd) · annexe C EN 1168</div></div><span class="chev">⌄</span></div></div>
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
    <div class="legend"><span><i style="background:#111827"></i>semelle</span><span><i style="background:repeating-linear-gradient(45deg,#fff 0 2px,#9ca3af 2px 3px)"></i>plateau acier</span><span><i style="border:1px dashed #2563eb"></i>aire efficace</span><span><i style="background:#ffedd5;border-color:#fdba74"></i>85–100 %</span><span><i style="background:#fee2e2;border-color:#fca5a5"></i>&gt; 100 %</span><span style="margin-left:auto">● appui &nbsp; ○ bord libre</span></div>
  </div>
  <div class="card">
    <div class="h"><span style="font-weight:600">Coupe transversale sous P2</span><span style="font-size:12px;color:#6b7280">élément H3 · diffusion au plan moyen</span></div>
    {coupe_svg()}
  </div>
</div>
<div>{RIGHT}</div>
</div>
<div class="note">maquette d'étude · valeurs d'illustration</div>
</body></html>"""
(OUT / "ecran.html").write_text(PAGE, encoding="utf-8")

# ------------------------------------------------------------------ bibliothèque
def picto(kind):
    if kind == "pieds":
        return '<svg viewBox="0 0 60 44" width="60" height="44"><rect x="18" y="14" width="24" height="16" fill="#fff" stroke="#111827"/><circle cx="27" cy="22" r="5" fill="#fff" stroke="#111827"/><line x1="30" y1="22" x2="54" y2="22" stroke="#111827" stroke-width="2"/>' + "".join(f'<line x1="{a}" y1="{b}" x2="{c}" y2="{d}" stroke="#111827" stroke-width="1.5"/><rect x="{c-4}" y="{d-4}" width="8" height="8" fill="#111827"/>' for a,b,c,d in ((20,16,8,6),(40,16,52,6),(20,28,8,38),(40,28,52,38))) + '</svg>'
    if kind == "roues":
        return '<svg viewBox="0 0 60 44" width="60" height="44"><rect x="12" y="12" width="36" height="20" rx="3" fill="#fff" stroke="#111827"/><rect x="36" y="14" width="9" height="8" fill="#d1d5db" stroke="#111827"/>' + "".join(f'<rect x="{x}" y="{y}" width="6" height="9" fill="#111827"/>' for x,y in ((14,8),(40,8),(14,27),(40,27))) + '</svg>'
    if kind == "etancon":
        return '<svg viewBox="0 0 60 44" width="60" height="44"><circle cx="30" cy="22" r="9" fill="#fff" stroke="#111827" stroke-width="1.4"/><line x1="24" y1="16" x2="36" y2="28" stroke="#111827"/><line x1="24" y1="28" x2="36" y2="16" stroke="#111827"/></svg>'
    return '<svg viewBox="0 0 60 44" width="60" height="44"><rect x="10" y="10" width="40" height="24" fill="url(#hb)" stroke="#7c3aed"/><defs><pattern id="hb" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#7c3aed" stroke-width=".7"/></pattern></defs></svg>'

CSS2 = CSS + """
.lib{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}
.eng{display:flex;gap:12px;border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;align-items:center;background:#fff}
.eng.on{border-color:#2563eb;box-shadow:0 0 0 2px #dbeafe}
.eng .nm{font-weight:600;font-size:13.5px} .eng .d{font-size:12px;color:#6b7280;margin-top:2px}
.src{font-size:10.5px;border-radius:999px;padding:2px 8px;margin-left:6px;vertical-align:1px} .src.ok{background:#dcfce7;color:#15803d} .src.gen{background:#f3f4f6;color:#6b7280} .src.todo{background:#fef3c7;color:#92400e}
.form{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.form label{display:block;font-size:11px;color:#6b7280;margin-bottom:3px} .form .in{border:1px solid #d1d5db;border-radius:7px;padding:6px 8px;font-size:13px;background:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tabs{display:flex;gap:6px;margin-top:4px} .tabs span{border:1px solid #d1d5db;border-radius:999px;padding:4px 12px;font-size:12.5px;color:#6b7280} .tabs span.on{background:#111827;color:#fff;border-color:#111827}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px 24px 28px;align-items:start}
"""
LIB = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Bibliothèque</title><style>{CSS2}</style></head><body>
<div class="top"><span style="font-size:18px">⛶</span><h1>Charges sur plancher</h1><span class="cas">CAS <span class="sel">C1 · Niveau +2 — zone grue ▾</span></span><div class="right"><span class="btn">Fermer</span></div></div>
<div class="two">
<div class="card">
  <div class="h"><span style="font-weight:600;font-size:15px">Bibliothèque — ajouter une charge</span><span class="btn">＋ Nouvelle fiche</span></div>
  <div class="tabs"><span class="on">Engins</span><span>Grues</span><span>Étais et filières</span><span>Stocks</span><span>Plateaux</span></div>
  <div class="lib">
    <div class="eng on">{picto("pieds")}<div><div class="nm">Grue araignée MRT 2150</div><div class="d">4 pieds · 4,2 × 4,2 m · 95 kN max par pied<br><span class="src ok" style="margin:0">fiche Manitou 2024</span></div></div></div>
    <div class="eng">{picto("roues")}<div><div class="nm">Télescopique MT 1840</div><div class="d">4 roues · 2,3 × 4,2 m · essieu AV 90 kN · φ 1,40<br><span class="src gen" style="margin:0">générique</span></div></div></div>
    <div class="eng">{picto("pieds")}<div><div class="nm">Grue mobile LTM 1060</div><div class="d">4 stabilisateurs · 7,0 × 6,5 m · réaction : tableau de levage<br><span class="src todo" style="margin:0">fiche à compléter</span></div></div></div>
    <div class="eng">{picto("roues")}<div><div class="nm">Nacelle ciseaux 12 m</div><div class="d">4 roues pleines · 1,2 × 2,4 m · 18 kN par roue · φ 2,00<br><span class="src ok" style="margin:0">fiche Haulotte 2023</span></div></div></div>
    <div class="eng">{picto("etancon")}<div><div class="nm">Étançon</div><div class="d">charge ponctuelle · semelle 150 × 150 · charge saisie<br><span class="src gen" style="margin:0">générique</span></div></div></div>
    <div class="eng">{picto("stock")}<div><div class="nm">Stock de palettes</div><div class="d">charge surfacique · 15 kN/m² · emprise saisie<br><span class="src gen" style="margin:0">générique</span></div></div></div>
  </div>
  <div class="meta" style="margin-top:12px">Les fiches du bureau sont versionnées avec l'application ; une fiche ajoutée dans un cas reste dans son fichier. Chaque fiche porte sa source (document, date) et son coefficient dynamique.</div>
</div>
<div>
  <div class="card">
    <div class="h"><span style="font-weight:600;font-size:15px">Fiche — Grue araignée MRT 2150</span><span class="src ok" style="font-size:11.5px">fiche Manitou 2024 · p. 12</span></div>
    <div class="form">
      <div><label>Nom</label><div class="in">Grue araignée MRT 2150</div></div><div><label>Type d'appuis</label><div class="in">Pieds ▾</div></div><div><label>Nombre</label><div class="in">4</div></div><div><label>φ dynamique</label><div class="in">1,00</div></div>
      <div><label>Entraxe long. (m)</label><div class="in">4,20</div></div><div><label>Entraxe transv. (m)</label><div class="in">4,20</div></div><div><label>Semelle (mm)</label><div class="in">300 × 300</div></div><div><label>Masse (t)</label><div class="in">21,5</div></div>
      <div><label>Charge max / pied AV (kN)</label><div class="in">95</div></div><div><label>Charge max / pied AR (kN)</label><div class="in">60</div></div><div><label>Configurations</label><div class="in">levage · déplacement ▾</div></div><div><label>Source</label><div class="in">Manitou, MRT 2150, 2024</div></div>
    </div>
    <div class="meta">Les charges par pied se relisent sur le tableau de stabilité du constructeur pour l'opération prévue ; la fiche mémorise la dernière valeur saisie et sa source.</div>
  </div>
  <div class="card">
    <div class="step"><span class="num">1</span><div><div class="ttl">Plancher</div><div class="sub">déplié</div></div><span class="chev">⌃</span></div>
    <div class="segw"><span class="on">Hourdis</span><span>Dalle BA</span><span>Prédalle</span></div>
    <div class="form" style="grid-template-columns:repeat(3,1fr)">
      <div><label>Type (catalogue)</label><div class="in">ERGON SP 200 ▾</div></div><div><label>Portée L (m)</label><div class="in">7,20</div></div><div><label>Éléments</label><div class="in">8 × 1,20 m</div></div>
      <div><label>État</label><div class="in">Claveté · 14 j ▾</div></div><div><label>Béton clavetage</label><div class="in">C25/30 → C20/25 (t)</div></div><div><label>Charges présentes</label><div class="in">1,0 kN/m²</div></div>
    </div>
    <div class="meta">Catalogue : M_Rd <b>90 kN·m / él.</b> · V_Rd <b>95 kN / él.</b> · g <b>2,9 kN/m²</b> — nature <b>Rd</b> (fiche SP, édition à renseigner) · membrure p_adm 0,70 N/mm² · M_cr non renseigné. Appuis : bords A et B (clic sur un bord du plan pour changer).</div>
  </div>
</div>
</div>
<div class="note">maquette d'étude · valeurs d'illustration</div>
</body></html>"""
(OUT / "bibliotheque.html").write_text(LIB, encoding="utf-8")
print("ok")

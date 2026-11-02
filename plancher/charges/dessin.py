# -*- coding: utf-8 -*-
"""Dessins du module « Charges sur plancher » : le plan cliquable et
déplaçable, la coupe sous un pied, le bandeau de balayage. SVG en
MILLIMÈTRES (le composant ``svg_cliquable`` renvoie les glissements en
unités du dessin, entières : ici des mm).

Repère ÉCRAN et repère MOTEUR. Le moteur a toujours sa portée le long de
son axe x (appuis A et B en x = 0 et x = L). À l'écran, ``sens = 1`` montre
les appuis à gauche et à droite (écran = moteur), ``sens = 2`` les montre
en haut et en bas (écran = moteur transposé). Toutes les positions
stockées dans le cas sont dans le repère MOTEUR ; ce module convertit.

Grammaire des clés renvoyées par le composant (``valeurs`` / ``changes``) :
  ``L``                      portée (mm, cote cliquable)
  ``n_elements`` / ``largeur``   poignées ± de la largeur (hourdis) / cote (dalle)
  ``sens``                   poignée d'un bord libre (1 ↔ 2)
  ``engin:<ref>.sx`` ``.sy`` glissement d'un engin (mm écran)
  ``charge:<ref>.sx`` ``.sy`` glissement d'une charge seule
  ``rep:<ref>.sx`` ``.sy``   glissement d'un plateau (centre, mm écran)
  ``fil:<id>.sx`` ``.sy``    glissement d'une filière
  ``<obj>:<ref>.<champ>``    champs des fenêtres (angle, P, type, element, …)
"""
from typing import Dict, List, Optional, Tuple

from . import symboles as SY

NOIR, GRIS, LIGNE, BLEU, ROUGE, ROUGE_F, ORANGE, ORANGE_F = (
    "#111827", "#6b7280", "#d1d5db", "#2563eb", "#b91c1c", "#fee2e2", "#c2410c", "#ffedd5")
VIOLET = "#7c3aed"
NS = 'vector-effect="non-scaling-stroke"'
FS, FS_S, FS_L = 230, 200, 260            # tailles de texte en unités du dessin (mm), pour un plan de référence
#   → ≈ 13 px dans une colonne de 650 px (PC), 9 px à 450 px (tablette) et à la largeur minimale (téléphone)
LARGEUR_REF = 11400                        # largeur SVG (mm) du plan de référence : au-delà, textes et étiquettes grossissent (k)
OX, OY = 1500, 800                         # origine du plancher dans le SVG
MARGE_D, MARGE_B = 2700, 1000             # marge basse : échelle graphique, puis la légende des appuis sur une 2e ligne
STYLE = ("<style>.ed{cursor:pointer}.ed.sel>rect,.ed.sel>g>rect:first-child{stroke:#2563eb !important;stroke-width:2.2px !important}"
         ".ed.drag{opacity:.75}text{font-family:Inter,system-ui,sans-serif}</style>")
DEFS = f'''<defs>
<pattern id="chp_hc" width="60" height="60" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="60" stroke="#9ca3af" stroke-width="0.8" {NS}/></pattern>
<pattern id="chp_hs" width="70" height="70" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="70" stroke="{VIOLET}" stroke-width="0.8" {NS}/></pattern>
<marker id="chp_tk" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 5 1 L 5 9" stroke="#374151" stroke-width="1.4"/></marker>
</defs>'''


def _fmt(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


class Vue:
    """Conversions moteur (m) ↔ écran (mm) selon le sens d'affichage."""

    def __init__(self, L: float, B: float, sens: int):
        self.L, self.B, self.sens = L, B, 1 if sens != 2 else 2
        self.W, self.H = (L * 1e3, B * 1e3) if self.sens == 1 else (B * 1e3, L * 1e3)

    def s(self, x: float, y: float) -> Tuple[float, float]:
        """Moteur (m) → écran (mm, absolu dans le SVG)."""
        if self.sens == 1:
            return OX + x * 1e3, OY + y * 1e3
        return OX + y * 1e3, OY + x * 1e3

    def m(self, sx: float, sy: float) -> Tuple[float, float]:
        """Écran (mm absolu) → moteur (m)."""
        if self.sens == 1:
            return (sx - OX) / 1e3, (sy - OY) / 1e3
        return (sy - OY) / 1e3, (sx - OX) / 1e3

    def dims(self, a: float, b: float) -> Tuple[float, float]:
        """Dimensions moteur (a le long de la portée, b en travers ; m) → écran (mm)."""
        return (a * 1e3, b * 1e3) if self.sens == 1 else (b * 1e3, a * 1e3)

    def bande(self, y0: float, y1: float) -> Tuple[float, float, float, float]:
        """Rectangle écran (x, y, w, h) d'une bande moteur [y0, y1]."""
        if self.sens == 1:
            return OX, OY + y0 * 1e3, self.W, (y1 - y0) * 1e3
        return OX + y0 * 1e3, OY, (y1 - y0) * 1e3, self.H

    @property
    def largeur_svg(self):
        return OX + self.W + MARGE_D

    @property
    def hauteur_svg(self):
        return OY + self.H + MARGE_B + (900 if self.sens == 2 else 0)


def _couleurs(taux: Optional[float], att: float, nok: float):
    if taux is None:
        return "white", GRIS, 400
    if taux > nok:
        return ROUGE_F, ROUGE, 700
    if taux >= att:
        return ORANGE_F, ORANGE, 700
    return "white", GRIS, 400


def _texte(x, y, t, fs=FS, fill=NOIR, anchor="start", w=400, extra=""):
    return f'<text x="{x:.0f}" y="{y:.0f}" font-size="{fs}" fill="{fill}" text-anchor="{anchor}" font-weight="{w}" {extra}>{t}</text>'


def _etiquette(k, x, y, t, sel=False, w=1400):
    """Étiquette courte (« P2 · 95 kN »), bleue si sélectionnée ; (x, y) = coin haut-gauche ;
    ``k`` = facteur d'échelle des textes du plan."""
    w, h = w * k, 340 * k
    if sel:
        return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{45 * k:.0f}" fill="{BLEU}"/>'
                + _texte(x + 120 * k, y + 240 * k, t, FS_S * k, "white", w=600))
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{45 * k:.0f}" fill="white" stroke="{LIGNE}" {NS}/>'
            + _texte(x + 120 * k, y + 240 * k, t, FS_S * k, NOIR))


# ---------------------------------------------------------------- plan
def plan(ctx: Dict) -> Tuple[str, Dict, Dict]:
    """Le plan. ``ctx`` (préparé par ``etat.contexte_dessin``) :
      L, B, sens, type ('hourdis' | 'dalle' | 'predalle'), elements [{nom, y0, y1, taux, gouverne}],
      joints [y…] (prédalle), engins [{ref, gabarit(dict), x, y, angle, empreintes [{nom, x, y, a, b, P, phi}]}],
      charges [{ref, type 'ponctuel'|'surfacique'|'lineaire', x, y, a, b, P, …}],
      plateaux [{ref, nom, x, y, A, Bp, a_eff, b_eff, complet, pression}] (centre du plateau, moteur),
      filieres [{id, nom, x, y, longueur, orientation, b, refs}],
      selection (clé), options {cotes, aires, taux}, seuils (att, nok), libelles {engin ref: texte},
      groupes (définitions des fenêtres), valeurs (dict des clés)."""
    v = Vue(ctx["L"], ctx["B"], ctx.get("sens", 1))
    att, nok = ctx.get("seuils", (0.85, 1.0))
    opt = ctx.get("options", {})
    sel = ctx.get("selection")
    # Textes : constants À L'ÉCRAN quelle que soit la taille du plan. k grossit les textes des grands
    # plans ; la largeur minimale (px) garde une échelle lisible sur téléphone — le conteneur du
    # composant défile alors horizontalement (#fig : overflow-x auto).
    k = max(1.0, v.largeur_svg / LARGEUR_REF)
    fs, fs_s, fs_l = FS * k, FS_S * k, FS_L * k
    px_min = int(min(440, v.largeur_svg * 0.04))      # 440 px tient dans la colonne d'une tablette (1024)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {v.largeur_svg:.0f} {v.hauteur_svg:.0f}" width="100%" '
         f'style="max-width:{int(v.largeur_svg / 11)}px;min-width:{px_min}px;display:block;margin:0 auto">', STYLE, DEFS]
    a = s.append
    # bandes
    for e in ctx["elements"]:
        fill, col, w = _couleurs(e.get("taux"), att, nok)
        x, y, wd, ht = v.bande(e["y0"], e["y1"])
        a(f'<rect x="{x:.0f}" y="{y:.0f}" width="{wd:.0f}" height="{ht:.0f}" fill="{fill}" stroke="#9ca3af" stroke-width="0.8" {NS}/>')
        # nom de la bande, hors du plancher
        if v.sens == 1:
            a(_texte(OX - 120, y + ht / 2 + 55, e["nom"], fs_s, col, "end", w))
        else:
            a(_texte(x + wd / 2, OY - 90, e["nom"], fs_s, col, "middle", w))
        # taux, hors du plancher
        if opt.get("taux", True) and e.get("taux") is not None:
            t = e["taux"]
            txt = f"{t * 100:.0f} %"
            if v.sens == 1:
                cx, cy = OX + v.W + 330, y + ht / 2
            else:
                cx, cy = x + wd / 2, OY + v.H + 420
            if t >= att:
                pw = 900
                a(f'<rect x="{cx - (0 if v.sens == 1 else pw / 2):.0f}" y="{cy - 160:.0f}" width="{pw}" height="320" rx="160" '
                  f'fill="{col if t > nok else "white"}" stroke="{col}" stroke-width="1.2" {NS}/>')
                a(_texte(cx + (pw / 2 if v.sens == 1 else 0), cy + 60, txt, fs, "white" if t > nok else col, "middle", 700))
            else:
                a(_texte(cx + (0 if v.sens == 1 else 0), cy + 60, txt, fs, GRIS, "start" if v.sens == 1 else "middle"))
            if e.get("gouverne") and t >= att:
                gx = cx + (pw + 120 if v.sens == 1 else 0)
                gy = cy + 60 if v.sens == 1 else cy + 400
                a(_texte(gx, gy, e["gouverne"], fs_s, GRIS, "start" if v.sens == 1 else "middle"))
    # joints
    els = ctx["elements"]
    for i in range(1, len(els)):
        yj = els[i]["y0"]
        if ctx["type"] == "hourdis" or yj in ctx.get("joints", []):
            if v.sens == 1:
                a(f'<line x1="{OX}" y1="{OY + yj * 1e3:.0f}" x2="{OX + v.W:.0f}" y2="{OY + yj * 1e3:.0f}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="80 60" {NS}/>')
            else:
                a(f'<line x1="{OX + yj * 1e3:.0f}" y1="{OY}" x2="{OX + yj * 1e3:.0f}" y2="{OY + v.H:.0f}" stroke="#9ca3af" stroke-width="1" stroke-dasharray="80 60" {NS}/>')
    # contour : appuis en trait épais, bords libres fins, poignées de bord
    if v.sens == 1:
        appuis = [((OX, OY), (OX, OY + v.H)), ((OX + v.W, OY), (OX + v.W, OY + v.H))]
        libres = [((OX, OY), (OX + v.W, OY)), ((OX, OY + v.H), (OX + v.W, OY + v.H))]
        a(_texte(OX, OY - 150, "A", fs, NOIR, "middle", 700)); a(_texte(OX + v.W, OY - 150, "B", fs, NOIR, "middle", 700))
    else:
        appuis = [((OX, OY), (OX + v.W, OY)), ((OX, OY + v.H), (OX + v.W, OY + v.H))]
        libres = [((OX, OY), (OX, OY + v.H)), ((OX + v.W, OY), (OX + v.W, OY + v.H))]
        a(_texte(OX - 150, OY + 60, "A", fs, NOIR, "end", 700)); a(_texte(OX - 150, OY + v.H + 60, "B", fs, NOIR, "end", 700))
    for (x1, y1), (x2, y2) in libres:
        a(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{LIGNE}" stroke-width="1.2" {NS}/>')
    for (x1, y1), (x2, y2) in appuis:
        a(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{NOIR}" stroke-width="5" {NS}/>')
    delta = "+1" if v.sens == 1 else "-1"
    for (x1, y1), (x2, y2) in appuis:
        a(f'<circle cx="{(x1 + x2) / 2:.0f}" cy="{(y1 + y2) / 2:.0f}" r="75" fill="{NOIR}" stroke="{NOIR}" stroke-width="1.4" {NS}/>')
    for (x1, y1), (x2, y2) in libres:
        a(f'<g class="ed" data-action="sens:{delta}"><title>Bord libre : clic = appui (change le sens porteur)</title>'
          f'<circle cx="{(x1 + x2) / 2:.0f}" cy="{(y1 + y2) / 2:.0f}" r="75" fill="white" stroke="{NOIR}" stroke-width="1.4" {NS}/></g>')
    # cotes
    if opt.get("cotes", True):
        if v.sens == 1:
            a(f'<line x1="{OX}" y1="{OY - 420}" x2="{OX + v.W:.0f}" y2="{OY - 420}" stroke="#374151" stroke-width="1" marker-start="url(#chp_tk)" marker-end="url(#chp_tk)" {NS}/>')
            a(f'<g class="ed" data-key="L" data-sym="L"><title>Portée : clic pour saisir (mm)</title>'
              f'<rect x="{OX + v.W / 2 - 700:.0f}" y="{OY - 680}" width="1400" height="300" fill="white" opacity="0"/>'
              + _texte(OX + v.W / 2, OY - 470, f"L = {_fmt(v.L)} m", fs, "#374151", "middle") + '</g>')
            a(f'<line x1="{OX - 520}" y1="{OY}" x2="{OX - 520}" y2="{OY + v.H:.0f}" stroke="#374151" stroke-width="1" marker-start="url(#chp_tk)" marker-end="url(#chp_tk)" {NS}/>')
            a(f'<text transform="translate({OX - 600} {OY + v.H / 2:.0f}) rotate(-90)" font-size="{fs}" fill="#374151" text-anchor="middle">{ctx["cote_largeur"]}</text>')
            if ctx["type"] == "hourdis":
                for d, xx in (("-1", OX - 1000), ("+1", OX - 620)):
                    yy = OY + v.H + 120
                    a(f'<g class="ed" data-action="n_elements:{d}"><title>{"Un hourdis de plus" if d == "+1" else "Un hourdis de moins"}</title>'
                      f'<rect x="{xx}" y="{yy:.0f}" width="300" height="300" rx="60" fill="white" stroke="{LIGNE}" {NS}/>'
                      + _texte(xx + 150, yy + 215, "+" if d == "+1" else "−", fs, NOIR, "middle", 700) + "</g>")
        else:
            a(f'<line x1="{OX - 520}" y1="{OY}" x2="{OX - 520}" y2="{OY + v.H:.0f}" stroke="#374151" stroke-width="1" marker-start="url(#chp_tk)" marker-end="url(#chp_tk)" {NS}/>')
            a(f'<g class="ed" data-key="L" data-sym="L"><title>Portée : clic pour saisir (mm)</title>'
              f'<text transform="translate({OX - 600} {OY + v.H / 2:.0f}) rotate(-90)" font-size="{fs}" fill="#374151" text-anchor="middle">L = {_fmt(v.L)} m</text></g>')
            a(f'<line x1="{OX}" y1="{OY - 420}" x2="{OX + v.W:.0f}" y2="{OY - 420}" stroke="#374151" stroke-width="1" marker-start="url(#chp_tk)" marker-end="url(#chp_tk)" {NS}/>')
            a(_texte(OX + v.W / 2, OY - 470, ctx["cote_largeur"], fs, "#374151", "middle"))
            if ctx["type"] == "hourdis":
                for d, xx in (("-1", OX + v.W + 150), ("+1", OX + v.W + 530)):
                    a(f'<g class="ed" data-action="n_elements:{d}"><title>{"Un hourdis de plus" if d == "+1" else "Un hourdis de moins"}</title>'
                      f'<rect x="{xx:.0f}" y="{OY - 570}" width="300" height="300" rx="60" fill="white" stroke="{LIGNE}" {NS}/>'
                      + _texte(xx + 150, OY - 355, "+" if d == "+1" else "−", fs, NOIR, "middle", 700) + "</g>")
    # charges surfaciques et linéaires
    for c in ctx.get("charges", []):
        if c["type"] == "surfacique":
            x, y = v.s(c["x0"], c["y0"])
            w, h = v.dims(c["lx"], c["ly"])
            if v.sens == 2:
                x, y = v.s(c["x0"], c["y0"])
            a(f'<g class="ed" data-group="charge:{c["ref"]}" data-drag="charge:{c["ref"]}.sx" data-drag2="charge:{c["ref"]}.sy" data-dsym="x" data-dsym2="y">'
              f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" fill="url(#chp_hs)" stroke="{VIOLET}" stroke-width="1.2" {NS}/></g>')
            a(_texte(x + 100, y + 230, f"{c['ref']} · {_fmt(c['q'], 1)} kN/m²", fs_s, "#5b21b6", w=600))
        elif c["type"] == "lineaire":
            x, y = v.s(c["x"], c["y"])
            lw, lh = (c["longueur"], c["largeur"]) if c["orientation"] == "x" else (c["largeur"], c["longueur"])
            w, h = v.dims(lw, lh)
            a(f'<g class="ed" data-group="charge:{c["ref"]}" data-drag="charge:{c["ref"]}.sx" data-drag2="charge:{c["ref"]}.sy" data-dsym="x" data-dsym2="y">'
              f'<rect x="{x - w / 2:.0f}" y="{y - h / 2:.0f}" width="{w:.0f}" height="{h:.0f}" fill="#e5e7eb" stroke="{NOIR}" stroke-width="1.2" {NS}/></g>')
            a(_texte(x + w / 2 + 100, y + 60, f"{c['ref']} · {_fmt(c['p'], 1)} kN/m", fs_s, NOIR, w=600))
    # filières (sous les engins, au-dessus des bandes)
    for f in ctx.get("filieres", []):
        x, y = v.s(f["x"], f["y"])
        lw, lh = (f["longueur"], f["b"]) if f["orientation"] == "x" else (f["b"], f["longueur"])
        w, h = v.dims(lw, lh)
        selc = sel == f"fil:{f['id']}"
        if selc:
            a(f'<rect x="{x - w / 2 - 120:.0f}" y="{y - h / 2 - 120:.0f}" width="{w + 240:.0f}" height="{h + 240:.0f}" rx="80" fill="none" stroke="{BLEU}" stroke-width="1.6" stroke-dasharray="90 60" {NS}/>')
        a(f'<g class="ed" data-group="fil:{f["id"]}" data-drag="fil:{f["id"]}.sx" data-drag2="fil:{f["id"]}.sy" data-dsym="x" data-dsym2="y">'
          f'<title>Filière {f["nom"]} : clic = réglages, glisser = déplacer</title>'
          f'<rect x="{x - w / 2:.0f}" y="{y - h / 2:.0f}" width="{w:.0f}" height="{h:.0f}" fill="#cbd5e1" stroke="{BLEU if selc else NOIR}" stroke-width="1.6" {NS}/></g>')
        if h > w:      # filière verticale à l'écran : étiquette sous son extrémité basse
            a(_texte(x, y + h / 2 + 230, f["nom"], fs_s, BLEU, "middle", 600))
        else:          # horizontale : étiquette au bout droit
            a(_texte(x + w / 2 + 120, y + 60, f["nom"], fs_s, BLEU, "start", 600))
    # engins (symbole + semelles), charges ponctuelles
    for en in ctx.get("engins", []):
        x, y = v.s(en["x"], en["y"])
        tr = f'translate({x:.0f} {y:.0f})' + (' matrix(0 1 1 0 0 0)' if v.sens == 2 else '') + f' rotate({en["angle"]})'
        selc = sel == f"engin:{en['ref']}"
        corps = SY.symbole(en["gabarit"])
        pieds = "".join(f'<rect x="{ap["dx"] * 1e3 - ap["a"] * 500:.0f}" y="{ap["dy"] * 1e3 - ap["b"] * 500:.0f}" width="{ap["a"] * 1e3:.0f}" height="{ap["b"] * 1e3:.0f}" fill="{NOIR}"/>'
                        for ap in en["gabarit"].get("appuis", []))
        if selc:
            r = en.get("rayon_etiquette", 1600) + 200
            a(f'<rect x="{x - r:.0f}" y="{y - r:.0f}" width="{2 * r:.0f}" height="{2 * r:.0f}" rx="120" fill="none" stroke="{BLEU}" stroke-width="1.6" stroke-dasharray="90 60" {NS}/>')
        a(f'<g class="ed" data-group="engin:{en["ref"]}" data-drag="engin:{en["ref"]}.sx" data-drag2="engin:{en["ref"]}.sy" data-dsym="x" data-dsym2="y">'
          f'<title>{en["ref"]} · {en["gabarit"]["nom"]} : clic = réglages, glisser = déplacer</title><g transform="{tr}">{corps}{pieds}</g></g>')
        a(_texte(x, y + (en.get("rayon_etiquette", 1600)), ctx.get("libelles", {}).get(en["ref"], en["ref"]), fs_s, GRIS, "middle"))
    for c in ctx.get("charges", []):
        if c["type"] == "ponctuel":
            x, y = v.s(c["x"], c["y"])
            selc = sel == f"charge:{c['ref']}"
            if selc:
                a(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="300" fill="none" stroke="{BLEU}" stroke-width="1.6" stroke-dasharray="90 60" {NS}/>')
            w, h = v.dims(c["a"], c["b"])
            a(f'<g class="ed" data-group="charge:{c["ref"]}" data-drag="charge:{c["ref"]}.sx" data-drag2="charge:{c["ref"]}.sy" data-dsym="x" data-dsym2="y">'
              f'<title>{c["ref"]} : clic = réglages (charge, semelle, répartiteur), glisser = déplacer</title><g transform="translate({x:.0f} {y:.0f})">{SY._ponctuel()}</g>'
              f'<rect x="{x - w / 2:.0f}" y="{y - h / 2:.0f}" width="{w:.0f}" height="{h:.0f}" fill="{NOIR}"/></g>')
            a(_etiquette(k, x - 700 * k, y + 260, f"{c['ref']} · {_fmt(c['P'], 0)} kN", selc))
    # plateaux (objets séparés : glissement propre), aires efficaces, étiquettes des appuis
    for p in ctx.get("plateaux", []):
        x, y = v.s(p["x"], p["y"])
        w, h = v.dims(p["A"] / 1e3, p["Bp"] / 1e3)
        selc = sel == f"rep:{p['ref']}"
        if selc:
            a(f'<rect x="{x - w / 2 - 110:.0f}" y="{y - h / 2 - 110:.0f}" width="{w + 220:.0f}" height="{h + 220:.0f}" rx="60" fill="none" stroke="{BLEU}" stroke-width="1.8" {NS}/>')
        a(f'<g class="ed" data-group="rep:{p["ref"]}" data-drag="rep:{p["ref"]}.sx" data-drag2="rep:{p["ref"]}.sy" data-dsym="x" data-dsym2="y">'
          f'<title>Plateau {p["nom"]} sous {p["ref"]} : clic = réglages, glisser = décaler</title>'
          f'<rect x="{x - w / 2:.0f}" y="{y - h / 2:.0f}" width="{w:.0f}" height="{h:.0f}" fill="url(#chp_hc)" stroke="{NOIR}" stroke-width="1.3" {NS}/></g>')
        if opt.get("aires", True) and p.get("a_eff"):
            we, he = v.dims(p["a_eff"] / 1e3, p["b_eff"] / 1e3)
            a(f'<rect x="{x - we / 2:.0f}" y="{y - he / 2:.0f}" width="{we:.0f}" height="{he:.0f}" fill="none" stroke="{BLEU}" stroke-width="1" stroke-dasharray="50 35" {NS}/>')
    for en in ctx.get("engins", []):
        for ap in en["empreintes"]:
            x, y = v.s(ap["x"], ap["y"])
            w, h = v.dims(ap["a"], ap["b"])
            selc = sel == f"rep:{ap['ref']}"
            # semelle (répétée au-dessus du plateau) + étiquette : CLIQUABLES — la fenêtre « rep:<ref> »
            # règle le répartiteur de cet appui (aucun, plateau, filière, idem) ; la semelle suit
            # l'engin, elle ne se glisse pas (le plateau, lui, se décale)
            a(f'<g class="ed" data-group="rep:{ap["ref"]}"><title>Appui {ap["ref"]} : clic = répartiteur (aucun, plateau, filière)</title>'
              f'<rect x="{x - w / 2:.0f}" y="{y - h / 2:.0f}" width="{w:.0f}" height="{h:.0f}" fill="{NOIR}"/>'
              + _etiquette(k, x - 700 * k, y + max(h / 2, 300) + 120, f"{ap['nom']} · {_fmt(ap['P'] * ap.get('phi', 1.0), 0)} kN", selc) + "</g>")
    # échelle graphique
    sx0, sy0 = OX, OY + v.H + (280 if v.sens == 1 else 1000)
    a(f'<rect x="{sx0}" y="{sy0}" width="1000" height="70" fill="{NOIR}"/><rect x="{sx0 + 1000}" y="{sy0}" width="1000" height="70" fill="white" stroke="{NOIR}" {NS}/>')
    for i, t in enumerate(("0", "1", "2 m")):
        a(_texte(sx0 + i * 1000, sy0 + 290, t, fs_s, GRIS, "middle"))
    # légende des appuis sur une DEUXIÈME ligne : sur la ligne de l'échelle elle recouvrait « 2 m »
    # (capture du 02/11/2026, après l'agrandissement des textes)
    a(_texte(OX + v.W, sy0 + 290 + 330 * k, "● appui  ○ bord libre : clic pour changer le sens porteur", fs_s, GRIS, "end"))
    a("</svg>")
    return "\n".join(s), ctx.get("valeurs", {}), ctx.get("groupes", {})


# ---------------------------------------------------------------- coupe sous un appui
def coupe(ctx: Dict) -> str:
    """Coupe transversale sous l'appui sélectionné (ou gouvernant). ``ctx`` :
      type, h (m), chape (m ou 0), b_element (m), nom_element, voisins (noms),
      plateau {A, t} ou None, semelle (a, m), P (kN), a_eff (m), diffusion (bool),
      joint (bool, prédalle : joint au bord), titre."""
    sc = 0.22
    h = ctx["h"] * 1e3 * sc
    tc = ctx.get("chape", 0.0) * 1e3 * sc
    bu = ctx.get("b_element", 1.2) * 1e3 * sc
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {int(190 + h + tc)}" width="100%" style="display:block">', DEFS.replace("chp_", "cpc_")]
    a = s.append
    a('<pattern id="cpc_hb" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" stroke="#9ca3af" stroke-width="0.6"/></pattern>')
    yt = 120
    xc = 450
    x3 = xc - bu / 2
    noms = ctx.get("voisins", ("", ctx.get("nom_element", ""), ""))
    xs = [x3 - bu - 4, x3, x3 + bu + 4]
    if tc > 0:
        a(f'<rect x="{xs[0]:.0f}" y="{yt - tc:.0f}" width="{3 * bu + 8:.0f}" height="{tc:.0f}" fill="url(#cpc_hb)" stroke="{NOIR}" stroke-width="1"/>')
        a(f'<text x="{xs[0] + 4:.0f}" y="{yt - tc - 6:.0f}" font-size="15" fill="{GRIS}">couche de compression {ctx["chape"] * 1e3:.0f}</text>')
    for k, (x0, nom) in enumerate(zip(xs, noms)):
        fill = ROUGE_F if k == 1 and ctx.get("taux", 0) > 1 else (ORANGE_F if k == 1 and ctx.get("taux", 0) >= 0.85 else "white")
        if ctx["type"] == "hourdis":
            a(f'<rect x="{x0:.0f}" y="{yt}" width="{bu:.0f}" height="{h:.0f}" fill="{fill}" stroke="{NOIR}" stroke-width="1.2"/>')
            for i in range(5):
                a(f'<circle cx="{x0 + bu * (i + 0.5) / 5:.0f}" cy="{yt + h / 2:.0f}" r="{min(14, h * 0.32):.0f}" fill="white" stroke="{NOIR}" stroke-width="1"/>')
        else:
            a(f'<rect x="{x0:.0f}" y="{yt}" width="{bu:.0f}" height="{h:.0f}" fill="{fill}" stroke="{NOIR}" stroke-width="1.2"/>')
            for i in range(6):
                a(f'<circle cx="{x0 + bu * (i + 0.5) / 6:.0f}" cy="{yt + h - 10:.0f}" r="2.2" fill="{NOIR}"/>')
        if nom:
            a(f'<text x="{x0 + bu / 2:.0f}" y="{yt + h + 16:.0f}" font-size="16" fill="{ROUGE if k == 1 else GRIS}" font-weight="{600 if k == 1 else 400}" text-anchor="middle">{nom}</text>')
    if ctx.get("joints_predalle"):
        for xj in (x3, x3 + bu + 4):
            a(f'<line x1="{xj - 2:.0f}" y1="{yt - tc:.0f}" x2="{xj - 2:.0f}" y2="{yt + h:.0f}" stroke="{NOIR}" stroke-width="2"/>')
    top = yt - tc
    if ctx.get("plateau"):
        A, t = ctx["plateau"]["A"] * sc, max(ctx["plateau"]["t"] * sc, 4)
        a(f'<rect x="{xc - A / 2:.0f}" y="{top - t:.0f}" width="{A:.0f}" height="{t:.0f}" fill="url(#cpc_hc)" stroke="{NOIR}" stroke-width="1"/>')
        a(f'<text x="{xc + A / 2 + 10:.0f}" y="{top - 4:.0f}" font-size="15" fill="{GRIS}">plateau {ctx["plateau"]["A"]:.0f} × {ctx["plateau"]["t"]:.0f}</text>')
        top -= t
    sa = ctx["semelle"] * 1e3 * sc
    a(f'<rect x="{xc - sa / 2:.0f}" y="{top - 14:.0f}" width="{sa:.0f}" height="14" fill="{NOIR}"/>')
    a(f'<line x1="{xc}" y1="{top - 86:.0f}" x2="{xc}" y2="{top - 26:.0f}" stroke="{NOIR}" stroke-width="2" marker-end="url(#cpc_tk)"/>')
    a(f'<text x="{xc + 10:.0f}" y="{top - 58:.0f}" font-size="17" font-weight="600" fill="{NOIR}">{_fmt(ctx["P"], 0)} kN</text>')
    if ctx.get("a_eff"):
        e = ctx["a_eff"] * 1e3 * sc / 2
        for sg in (-1, 1):
            a(f'<line x1="{xc + sg * e:.0f}" y1="{yt - tc:.0f}" x2="{xc + sg * (e + tc + h / 2):.0f}" y2="{yt + h / 2:.0f}" stroke="{BLEU}" stroke-width="1" stroke-dasharray="4 3"/>')
        a(f'<line x1="{xc - e:.0f}" y1="{yt + h + 30:.0f}" x2="{xc + e:.0f}" y2="{yt + h + 30:.0f}" stroke="{BLEU}" stroke-width="1" marker-start="url(#cpc_tk)" marker-end="url(#cpc_tk)"/>')
        a(f'<text x="{xc}" y="{yt + h + 48:.0f}" font-size="16" fill="{BLEU}" text-anchor="middle">aire efficace {ctx["a_eff"] * 1e3:.0f} mm · diffusion à 45° jusqu\'au plan moyen</text>')
    xr = xs[2] + bu + 14
    a(f'<line x1="{xr:.0f}" y1="{yt}" x2="{xr:.0f}" y2="{yt + h:.0f}" stroke="#374151" stroke-width="1" marker-start="url(#cpc_tk)" marker-end="url(#cpc_tk)"/>')
    a(f'<text x="{xr + 8:.0f}" y="{yt + h / 2 + 4:.0f}" font-size="15" fill="#374151">{ctx["h"] * 1e3:.0f}</text>')
    if tc > 0:
        a(f'<line x1="{xr:.0f}" y1="{yt - tc:.0f}" x2="{xr:.0f}" y2="{yt}" stroke="#374151" stroke-width="1" marker-start="url(#cpc_tk)" marker-end="url(#cpc_tk)"/>')
        a(f'<text x="{xr + 8:.0f}" y="{yt - tc / 2 + 4:.0f}" font-size="15" fill="#374151">{ctx["chape"] * 1e3:.0f}</text>')
    a("</svg>")
    return "\n".join(s)


# ---------------------------------------------------------------- balayage
def balayage(points: List[Dict], pire: Optional[Dict], x_actuel: float, L: float, seuil: float = 1.0) -> str:
    """Bandeau : taux du système selon la position de l'engin (vert ≤ seuil, rouge au-delà), position actuelle, pire."""
    W, H = 900, 150
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" style="display:block">']
    a = s.append
    x0, x1, y0, y1 = 60, W - 40, 30, 100
    def X(x): return x0 + (x1 - x0) * x / L
    tmax = max([p["taux"] for p in points if p["taux"] is not None] + [1.2])
    def Y(t): return y1 - (y1 - y0) * min(t, tmax) / tmax
    for p0, p1 in zip(points, points[1:]):
        if p0["taux"] is None:
            continue
        col = "#86efac" if p0["taux"] <= seuil else "#fca5a5"
        a(f'<rect x="{X(p0["x"]):.1f}" y="{Y(p0["taux"]):.1f}" width="{X(p1["x"]) - X(p0["x"]) + 0.3:.1f}" height="{y1 - Y(p0["taux"]):.1f}" fill="{col}"/>')
    pts = " ".join(f"{X(p['x']):.1f},{Y(p['taux']):.1f}" for p in points if p["taux"] is not None)
    if pts:
        a(f'<polyline points="{pts}" fill="none" stroke="{NOIR}" stroke-width="1.5"/>')
    a(f'<line x1="{x0}" y1="{Y(seuil):.1f}" x2="{x1}" y2="{Y(seuil):.1f}" stroke="{ROUGE}" stroke-width="1" stroke-dasharray="5 4"/>')
    a(f'<line x1="{X(x_actuel):.1f}" y1="{y0 - 8}" x2="{X(x_actuel):.1f}" y2="{y1 + 6}" stroke="{BLEU}" stroke-width="2"/>')
    a(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#374151" stroke-width="1"/>')
    for xm in range(0, int(L) + 1):
        a(f'<text x="{X(xm):.1f}" y="{y1 + 16}" font-size="10" fill="{GRIS}" text-anchor="middle">{xm}</text>')
    a(f'<text x="{x0}" y="{y0 - 12}" font-size="11" fill="#374151">Taux du système selon la position x de l\'engin (m) — vert ≤ 100 %, rouge au-delà</text>')
    if pire:
        a(f'<text x="{x1}" y="{y1 + 34}" font-size="11" fill="{ROUGE}" font-weight="700" text-anchor="end">pire : x = {_fmt(pire["x"])} m · {pire["taux"] * 100:.0f} %</text>')
    a(f'<text x="{x0}" y="{y1 + 34}" font-size="11" fill="{BLEU}" font-weight="700">position actuelle x = {_fmt(x_actuel)} m</text>')
    a("</svg>")
    return "\n".join(s)

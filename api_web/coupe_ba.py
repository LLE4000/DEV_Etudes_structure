# -*- coding: utf-8 -*-
"""La COUPE COTÉE d'une section en béton armé, dessinée à l'échelle (mm), INTERACTIVE : le composant de dessin du
ferraillage du site (« Poutres béton armé », réutilisable par les autres modules béton).

Tout vient de la géométrie que le moteur a calculée (``beton.poutre.moteur``, clé « geometrie » d'une section) : les
lits (position de l'axe, abscisses des barres), les lignes d'effort tranchant (emprise de chaque cadre), les armatures
latérales. Rien n'est recalculé ici : le dessin TRANSCRIT — la note PDF pose les mêmes barres et les mêmes cadres
(tests/test_note_poutre.py, tests/test_api_web_poutre.py le mesurent).

Un plan de ferraillage, sobre :
- le béton : contour net, fond clair semé d'un pointillé de granulats très discret (la convention de coupe du béton) ;
- chaque CADRE à sa place RÉELLE : ses nus extérieurs à l'emprise du moteur (x0, x1), ses branches haute et basse au nu de
  l'enrobage — deux cadres qui se recouvrent partagent leurs branches, aucun n'est décalé pour « se voir » ; dessiné en
  double trait (le contour de l'acier et son âme claire), angles cintrés autour de la barre d'angle (rayon Ø/2 + Ø étrier
  /2, la barre posée dans le pli) ; FERMETURE en haut à gauche : les deux extrémités s'enroulent de 135° autour de la
  barre d'angle et rentrent dans le noyau par un retour droit de max(5 Ø ; 50 mm) (EC2 8.5, fig. 8.5) ;
- une épingle longe sa barre et s'enroule autour des barres haute et basse ;
- chaque LIT annoté à l'extérieur, à droite, « L1 : 3Ø20 », par une fine ligne de rappel ; les cadres étiquetés
  « C1 », « C2 »… au-dessus ; les cotes b, h, d et l'enrobage c.

Les efforts et la désignation des aciers ne sont PAS dans le dessin : l'en-tête de la vue les porte (api_web/poutre.py),
une seule fois. La vue « Toutes » titre chaque coupe (nom de la section, efforts en clair).

Chaque pièce est un groupe ``<g data-sel="s<i>.<pièce>">`` (pièces : beton, lit-inf-<j>, lit-sup-<j>, cadre-<k>, lat) :
un clic sur la pièce ou sur son annotation la choisit (js/compact.js). Un cadre choisi montre deux POIGNÉES au pied de
ses montants (``.poignee[data-bout="de"|"a"]``, cachées sinon) : on les fait glisser, elles s'aimantent aux barres du lit
1 inférieur (``data-barres`` de la coupe, mm) — impossible de poser un cadre ailleurs que sur une barre ; le groupe du
cadre dit ses barres entourées (``data-de``, ``data-a``, numérotées depuis 1) et son type. Les tailles des textes et
des traits suivent la taille de la section : une poutre 20 × 40 et une poutre 80 × 120 se lisent de la même façon.
"""
import math
from html import escape
from typing import Dict, List

ENCRE = "#1B2230"
BETON = "#F6F7F9"
BETON_BORD = "#2A3240"
GRANULAT = "#9AA3B0"
ACIER = "#222A36"
CADRE_BORD = "#3F4A5C"
CADRE_AME = "#D3D9E2"
COTE = "#8A94A3"
DOUX = "#6B7480"
POLICE = "Inter,system-ui,Arial,sans-serif"
C45 = math.sqrt(0.5)
MOTIF = "__MOTIF_BETON__"     # remplacé par l'identifiant du motif de chaque <svg> (svg())


def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _esc(t) -> str:
    return escape(str(t), quote=True)


class _Plume:
    """Les tailles du dessin, proportionnelles à la section (unité u : 1 pour une section de 400 mm)."""

    def __init__(self, b: float, h: float):
        self.u = max(b, h, 250.0) / 400.0
        self.F = 14.0 * self.u          # texte courant
        self.f = 11.5 * self.u          # texte secondaire
        self.trait = 0.85 * self.u      # cotes
        self.contour = 1.6 * self.u     # béton


def _txt(x, y, t, taille, ancre="start", couleur=ENCRE, poids=500, extra="", classe="") -> str:
    cl = f' class="{classe}"' if classe else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}" font-weight="{poids}"{cl}{extra}>{t}</text>')


def _ligne(x1, y1, x2, y2, couleur, ep, extra="") -> str:
    return (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"'
            f' stroke-linecap="round"{extra}/>')


def _cote_h(pl: _Plume, x1, x2, y, texte, y_attache=None) -> str:
    t = 3.6 * pl.u
    out = ""
    if y_attache is not None:
        for x in (x1, x2):
            out += _ligne(x, y_attache, x, y + t * 1.4, COTE, pl.trait * 0.7)
    out += (_ligne(x1 - t * 1.2, y, x2 + t * 1.2, y, COTE, pl.trait)
            + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, pl.trait * 1.15) + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, pl.trait * 1.15)
            + _txt((x1 + x2) / 2, y - 4.5 * pl.u, texte, pl.f, "middle", DOUX, 500))
    return out


def _cote_v(pl: _Plume, x, y1, y2, texte, x_attache=None) -> str:
    t = 3.6 * pl.u
    out = ""
    if x_attache is not None:
        for y in (y1, y2):
            out += _ligne(x_attache, y, x + (t * 1.4 if x > x_attache else -t * 1.4), y, COTE, pl.trait * 0.7)
    xt = x - 4.5 * pl.u
    ym = (y1 + y2) / 2
    out += (_ligne(x, y1 - t * 1.2, x, y2 + t * 1.2, COTE, pl.trait)
            + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, pl.trait * 1.15) + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, pl.trait * 1.15)
            + _txt(xt, ym, texte, pl.f, "middle", DOUX, 500, extra=f' transform="rotate(-90 {_f(xt)} {_f(ym)})"'))
    return out


def _nb(x: float, nd: int = 0) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


# ------------------------------------------------------------------ les cadres
def _arc(cx, cy, r, a_deg) -> tuple:
    a = math.radians(a_deg)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def chemin_cadre(x0: float, x1: float, y0: float, y1: float, de: float, r_haut: float, r_bas: float) -> str:
    """Le chemin d'un étrier FERMÉ sur sa ligne moyenne (repère du dessin : y vers le bas) : x0..x1, y0..y1 = ses nus
    EXTÉRIEURS ; ``r_haut`` / ``r_bas`` : le rayon des barres d'angle haut / bas (le pli les enveloppe : rayon de la
    ligne moyenne = r + Ø étrier / 2). Fermeture en haut à gauche : chaque extrémité s'enroule de 135° autour de la barre
    d'angle, puis rentre dans le noyau à 45° sur max(5 Ø ; 50 mm) — les deux retours encadrent la barre (EC2 fig. 8.5).
    Le chemin part du retour du montant gauche, descend, fait le tour et finit par le retour de la branche haute."""
    xa, xb, ya, yb = x0 + de / 2, x1 - de / 2, y0 + de / 2, y1 - de / 2
    w, h = xb - xa, yb - ya
    rt = max(de * 0.6, min(r_haut + de / 2, w / 2.05, h / 2.05))
    rb = max(de * 0.6, min(r_bas + de / 2, w / 2.05, h / 2.05))
    lg = min(max(5.0 * de, 50.0), 0.42 * min(w, h))
    ctx, cty = xa + rt, ya + rt                       # le centre du pli d'angle haut gauche (l'axe de la barre d'angle)
    t2 = _arc(ctx, cty, rt, 315.0)                    # le retour du montant gauche part du haut droit de la barre…
    t1 = _arc(ctx, cty, rt, 135.0)                    # … celui de la branche haute, du bas gauche
    d = (f"M{_f(t2[0] + lg * C45)},{_f(t2[1] + lg * C45)} L{_f(t2[0])},{_f(t2[1])}"
         f" A{_f(rt)},{_f(rt)} 0 0 0 {_f(xa)},{_f(cty)}"                 # 315° → 180° autour de la barre d'angle
         f" V{_f(yb - rb)} A{_f(rb)},{_f(rb)} 0 0 0 {_f(xa + rb)},{_f(yb)}"
         f" H{_f(xb - rb)} A{_f(rb)},{_f(rb)} 0 0 0 {_f(xb)},{_f(yb - rb)}"
         f" V{_f(ya + rt)} A{_f(rt)},{_f(rt)} 0 0 0 {_f(xb - rt)},{_f(ya)}"
         f" H{_f(ctx)} A{_f(rt)},{_f(rt)} 0 0 0 {_f(t1[0])},{_f(t1[1])}"  # 270° → 135° : le second pli, 135°
         f" L{_f(t1[0] + lg * C45)},{_f(t1[1] + lg * C45)}")
    return d


def _trace_acier(d: str, de: float, u: float) -> str:
    """Un acier de cadre en double trait : le contour (le diamètre) et l'âme claire ; joints et bouts arrondis. Dessous,
    une ZONE de saisie invisible, plus large que l'acier : un cadre se choisit sans viser son trait au pixel près."""
    bord = max(0.35 * u, min(0.9 * u, de * 0.16))
    return (f'<path class="zone" d="{d}" fill="none" stroke="#000" stroke-opacity="0" stroke-width="{_f(max(2.4 * de, 14 * u))}" '
            f'stroke-linejoin="round" pointer-events="stroke"/>'
            f'<path class="st b" d="{d}" fill="none" stroke="{CADRE_BORD}" stroke-width="{_f(de)}" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<path class="st a" d="{d}" fill="none" stroke="{CADRE_AME}" stroke-width="{_f(max(de - 2 * bord, de * 0.3))}" '
            f'stroke-linejoin="round" stroke-linecap="round"/>')


def chemin_epingle(xc: float, yh: float, yl: float, r: float, de: float) -> str:
    """Une épingle (1 brin) sur la barre d'abscisse ``xc`` : un brin vertical au contact de la barre (à droite), qui
    s'enroule de 180° autour de la barre haute (axe yh) et de la barre basse (axe yl) et revient par un retour droit de
    max(4 Ø ; 30 mm) le long de la barre. ``r`` : le plus grand des rayons des deux barres (le pli les enveloppe)."""
    R = r + de / 2
    lg = min(max(4.0 * de, 30.0), max(de, (yl - yh) / 2.5))
    return (f"M{_f(xc - R)},{_f(yh + lg)} L{_f(xc - R)},{_f(yh)} A{_f(R)},{_f(R)} 0 0 1 {_f(xc + R)},{_f(yh)}"
            f" V{_f(yl)} A{_f(R)},{_f(R)} 0 0 1 {_f(xc - R)},{_f(yl)} L{_f(xc - R)},{_f(yl - lg)}")


def _poignees(bouts, y: float, u: float) -> str:
    """Les POIGNÉES d'un cadre (une par montant ; une seule pour une épingle), au pied des montants, dans l'enrobage :
    cachées tant que le cadre n'est pas choisi (la feuille de l'établi les montre sur la pièce choisie), chacune avec une
    zone de saisie plus large que son dessin (le doigt)."""
    out = "".join(f'<g class="poignee" data-bout="{bout}" transform="translate({_f(x)} {_f(y)})">'
                  f'<circle class="z" r="{_f(13 * u)}" fill="#fff" fill-opacity="0"/>'
                  f'<circle class="o" r="{_f(5.2 * u)}" fill="#fff" stroke="#2B4BCF" stroke-width="{_f(1.6 * u)}"/></g>'
                  for bout, x in bouts)
    return f'<g class="poignees" visibility="hidden">{out}</g>'


# ------------------------------------------------------------------ les annotations empilées
def _empiler(cibles: List[float], ecart: float, sens: int) -> List[float]:
    """Des ordonnées de textes au plus près de leurs cibles, jamais à moins de ``ecart`` l'une de l'autre ; ``sens`` :
    +1 empile vers le bas (face inférieure : on s'écarte vers l'extérieur, vers le bas), −1 vers le haut."""
    out: List[float] = []
    for y in cibles:
        if out and (y - out[-1]) * sens < ecart:
            y = out[-1] + sens * ecart
        out.append(y)
    return out


def coupe(geo: Dict, i: int, d_utile=None, avec_cotes: bool = True, titre: str = "", sous_titre: str = "") -> Dict:
    """Le dessin d'UNE section : {"svg" (un groupe <g class="coupe"> en mm, origine au coin haut gauche du béton), "x0",
    "x1", "y0", "y1" (son emprise, annotations comprises), "u"}. ``i`` : l'indice de la section (préfixe des pièces,
    « s<i>. ») ; ``titre`` / ``sous_titre`` : la vue « Toutes » nomme chaque coupe (le nom, les efforts en clair)."""
    b, h = geo["b"] * 10.0, geo["h"] * 10.0
    pl = _Plume(b, h)
    c = geo["enrobage_calcul"] * 10.0
    p = f"s{i}."
    out = []
    # ---- le béton et ses cotes
    beton = (f'<rect x="0" y="0" width="{_f(b)}" height="{_f(h)}" fill="{BETON}"/>'
             f'<rect x="0" y="0" width="{_f(b)}" height="{_f(h)}" fill="url(#{MOTIF})"/>'
             f'<rect class="bord" x="0" y="0" width="{_f(b)}" height="{_f(h)}" fill="none" stroke="{BETON_BORD}" '
             f'stroke-width="{_f(pl.contour)}"/>')
    yb = h + 2.4 * pl.F
    cotes = ""
    if avec_cotes:
        cotes += _cote_h(pl, 0, b, yb, f"b = {_nb(b)}", y_attache=h + 3 * pl.u)
        cotes += _cote_v(pl, -3.2 * pl.F, 0, h, f"h = {_nb(h)}", x_attache=-3 * pl.u)
        cotes += _txt(0, -0.8 * pl.F, f"c = {_nb(c)}", pl.f * 0.92, "start", DOUX, 450)
    out.append(f'<g class="pc beton" data-sel="{p}beton">{beton}{cotes}</g>')

    def y_lit(face, lit):
        return h - lit["distance"] * 10.0 if face == "inf" else lit["distance"] * 10.0

    lits_inf, lits_sup = geo["inf"]["lits"], geo["sup"]["lits"]
    r_inf1, r_sup1 = lits_inf[0]["d"] / 2.0, lits_sup[0]["d"] / 2.0
    xs_inf1 = [x * 10.0 for x in lits_inf[0]["x"]]
    y_haut, y_bas = y_lit("sup", lits_sup[0]), y_lit("inf", lits_inf[0])
    # ---- les cadres (sous les barres), puis les épingles
    lignes = geo["etriers"].get("cadres") or []
    de_max = geo["etriers"]["d_max"]
    etiquettes = []
    couche = []
    for k, cd in enumerate(lignes):
        de = float(cd["d"])
        if cd["type"] == "epingle":
            xc = cd["x0"] * 10.0
            g = _trace_acier(chemin_epingle(xc, y_haut, y_bas, max(r_sup1, r_inf1), de), de, pl.u)
            poignees = _poignees((("de", xc),), h - c / 2, pl.u)
            etiquettes.append((k, xc, y_haut - max(r_sup1, r_inf1) - de))
        else:
            x0, x1 = cd["x0"] * 10.0, cd["x1"] * 10.0
            g = _trace_acier(chemin_cadre(x0, x1, c, h - c, de, r_sup1, r_inf1), de, pl.u)
            poignees = _poignees((("de", x0 + de / 2), ("a", x1 - de / 2)), h - c / 2, pl.u)
            etiquettes.append((k, (x0 + x1) / 2, c))
        couche.append(f'<g class="pc cadre" data-sel="{p}cadre-{k}" data-cadre="{k}" data-type="{cd["type"]}" data-brins="{cd["brins"]}" '
                      f'data-de="{cd["de"]}" data-a="{cd["a"]}">{g}{poignees}</g>')
    out.append(f'<g class="couche-cadres">{"".join(couche)}</g>')
    # ---- les armatures latérales
    lat = geo.get("laterales") or {}
    if lat.get("n"):
        dl = float(lat["d"])
        xg = c + de_max + dl / 2
        cercles = "".join(f'<circle cx="{_f(x)}" cy="{_f(h - y * 10.0)}" r="{_f(dl / 2)}" fill="{ACIER}"/>'
                          for x in (xg, b - xg) for y in lat["ys"])
        out.append(f'<g class="pc lat" data-sel="{p}lat">{cercles}</g>')
    # ---- les lits
    annot: Dict[str, List] = {"inf": [], "sup": []}
    for face in ("inf", "sup"):
        for j, lit in enumerate(geo[face]["lits"]):
            y = y_lit(face, lit)
            r = lit["d"] / 2.0
            cercles = "".join(f'<circle cx="{_f(x * 10.0)}" cy="{_f(y)}" r="{_f(r)}" fill="{ACIER}"/>' for x in lit["x"])
            annot[face].append((j, y, max(lit["x"]) * 10.0 + r, lit))
            out.append(f'<g class="pc lit" data-sel="{p}lit-{face}-{j}">{cercles}</g>')

    # ---- la cote d (hauteur utile de la face tendue), à gauche entre la section et la cote h
    if avec_cotes and d_utile is not None and d_utile > 0:
        tendue = "inf" if geo.get("tendu", "inf") == "inf" else "sup"
        yG = geo[tendue]["yG"] * 10.0
        ya, yz = (0.0, h - yG) if tendue == "inf" else (yG, h)
        out.append(f'<g class="pc beton" data-sel="{p}beton">{_cote_v(pl, -1.45 * pl.F, ya, yz, f"d = {_nb(d_utile * 10.0)}", x_attache=-3 * pl.u)}</g>')
    # ---- les annotations des lits, à droite, sans rien à traverser
    droite = b + 0.6 * pl.F
    x_txt = droite + 2.2 * pl.F
    ecart = 1.4 * pl.F
    largeur_txt = 0.0
    for face, sens in (("sup", -1), ("inf", 1)):
        items = annot[face]
        if not items:
            continue
        cibles = [y for _, y, _, _ in items]
        ordre = sorted(range(len(items)), key=lambda k: cibles[k], reverse=(face == "sup"))
        ys_ord = _empiler([cibles[k] + 0.35 * pl.F for k in ordre], ecart, sens)
        ys_txt = [0.0] * len(items)
        for pos, k in enumerate(ordre):
            ys_txt[k] = ys_ord[pos]
        for (j, y, xr, lit), yt in zip(items, ys_txt):
            libelle = f"L{j + 1} : {lit['n']}Ø{lit['d']}"
            largeur_txt = max(largeur_txt, len(libelle) * 0.58 * pl.F)
            coude = droite + 0.9 * pl.F
            ligne = (f'<polyline points="{_f(xr + 1.5 * pl.u)},{_f(y)} {_f(coude)},{_f(y)} {_f(x_txt - 0.5 * pl.F)},{_f(yt - 0.35 * pl.F)}"'
                     f' fill="none" stroke="{COTE}" stroke-width="{_f(pl.trait * 0.75)}"/>'
                     f'<circle cx="{_f(xr + 1.5 * pl.u)}" cy="{_f(y)}" r="{_f(1.3 * pl.u)}" fill="{COTE}"/>')
            out.append(f'<g class="pc lit" data-sel="{p}lit-{face}-{j}">{ligne}'
                       f'{_txt(x_txt, yt, _esc(libelle), pl.F, "start", ENCRE, 600, classe="an")}</g>')
    x_fin = x_txt + max(largeur_txt, 5 * pl.F)

    # ---- les étiquettes des cadres au-dessus
    y_top = -1.4 * pl.F
    if etiquettes:
        y_et = -1.9 * pl.F
        tri = sorted(etiquettes, key=lambda e: e[1])
        xs_et: List[float] = []
        for _, xm, _ in tri:
            xs_et.append(max(xm, xs_et[-1] + 2.4 * pl.f) if xs_et else xm)
        glisse = (xs_et[-1] - tri[-1][1]) / 2 if len(tri) > 1 else 0.0
        for (k, xm, yc), xe in zip(tri, xs_et):
            xe -= glisse
            out.append(f'<g class="pc cadre" data-sel="{p}cadre-{k}">'
                       f'{_ligne(xm, yc - 1.5 * pl.u, xe, y_et + 0.35 * pl.F, COTE, pl.trait * 0.7)}'
                       f'{_txt(xe, y_et, f"C{k + 1}", pl.f, "middle", CADRE_BORD, 650, classe="an")}</g>')
        y_top = y_et - pl.F
    if titre:
        if sous_titre:
            out.append(_txt(b / 2, y_top - 0.1 * pl.F, _esc(sous_titre), pl.f, "middle", DOUX, 500))
            y_top -= 1.3 * pl.F
        out.append(_txt(b / 2, y_top - 0.1 * pl.F, _esc(titre), pl.F * 1.1, "middle", ENCRE, 700))
        y_top -= 1.4 * pl.F
    x_debut = (-3.2 * pl.F - 1.6 * pl.F) if avec_cotes else -pl.F
    y_fin = (yb + 0.9 * pl.F) if avec_cotes else (h + pl.F)
    barres = ",".join(_f(x) for x in xs_inf1)
    corps = f'<g class="coupe" data-section="{i}" data-barres="{barres}" data-b="{_f(b)}">{"".join(out)}</g>'
    return {"svg": corps, "x0": x_debut, "x1": x_fin, "y0": y_top - 0.4 * pl.F, "y1": y_fin, "u": pl.u}


def _motif(ident: str, u: float) -> str:
    """Le béton en coupe : un semis discret de granulats (points et petits triangles), tuile de 64·u mm, sans
    régularité visible — la convention de dessin, à peine marquée pour ne rien disputer aux aciers."""
    t = 64.0 * u
    pts = ((7, 9, .9), (23, 4, .6), (41, 13, .8), (55, 6, .55), (12, 29, .6), (33, 25, .95), (49, 34, .6), (4, 46, .8),
           (22, 51, .55), (38, 44, .7), (58, 52, .9), (29, 60, .6), (61, 22, .5), (16, 39, .5))
    tri = ((45, 22, 2.6, 15), (14, 15, 2.2, 200), (52, 45, 2.4, 95), (27, 37, 2.0, 300), (8, 58, 2.3, 40))
    corps = "".join(f'<circle cx="{_f(x * u)}" cy="{_f(y * u)}" r="{_f(r * u)}"/>' for x, y, r in pts)
    for x, y, s_, a in tri:
        pts3 = " ".join(f"{_f(x * u + s_ * u * math.cos(math.radians(a + k * 120)))},{_f(y * u + s_ * u * math.sin(math.radians(a + k * 120)))}"
                        for k in range(3))
        corps += f'<polygon points="{pts3}"/>'
    return (f'<pattern id="{_esc(ident)}" patternUnits="userSpaceOnUse" width="{_f(t)}" height="{_f(t)}">'
            f'<g fill="{GRANULAT}" fill-opacity=".55">{corps}</g></pattern>')


def svg(ident: str, coupes: List[Dict]) -> str:
    """Une ou plusieurs coupes côte à côte (même échelle), dans un <svg> en mm. ``coupes`` : les dicts de ``coupe``."""
    morceaux, x = [], 0.0
    y0 = min(cp["y0"] for cp in coupes)
    y1 = max(cp["y1"] for cp in coupes)
    motif = f"bet-{ident}"
    u = max(cp["u"] for cp in coupes)
    for cp in coupes:
        dx = x - cp["x0"]
        morceaux.append(f'<g transform="translate({_f(dx)} 0)">{cp["svg"].replace(MOTIF, motif)}</g>')
        x = dx + cp["x1"] + 3.0 * 14.0 * cp["u"]
    largeur = x - 3.0 * 14.0 * coupes[-1]["u"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" id="{_esc(ident)}" class="coupe-ba" viewBox="{_f(0)} {_f(y0)} {_f(largeur)} {_f(y1 - y0)}"'
            f' preserveAspectRatio="xMidYMid meet" font-family="{POLICE}"><defs>{_motif(motif, u)}</defs>{"".join(morceaux)}</svg>')

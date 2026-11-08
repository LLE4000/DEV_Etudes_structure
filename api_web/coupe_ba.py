# -*- coding: utf-8 -*-
"""La COUPE COTÉE d'une section en béton armé, dessinée à l'échelle (mm), INTERACTIVE : le composant de dessin du
ferraillage du site (pilote « Poutres béton armé », réutilisable par les autres modules béton).

Tout vient de la géométrie que le moteur a calculée (``beton.poutre.moteur``, clé « geometrie » d'une section) : les
lits (position de l'axe, abscisses des barres), les lignes d'effort tranchant (emprise de chaque cadre), les armatures
latérales. Rien n'est recalculé ici : le dessin TRANSCRIT.

Le dessin d'un ingénieur, sobre et lisible :
- le béton en gris très clair, contour net ; les cotes fines (b dessous, h à gauche, d à droite) ;
- chaque CADRE dessiné sur sa ligne moyenne, angles cintrés (rayon 2,5 Ø), fermeture en haut à gauche par deux
  crochets à 135° ; plusieurs cadres côte à côte ou imbriqués restent distincts ; une épingle longe sa barre et
  s'enroule autour des barres haute et basse ;
- chaque LIT annoté à l'extérieur, à droite, « L1 : 3Ø20 », par une fine ligne de rappel ; les annotations d'une face
  s'empilent sans se toucher et ne croisent jamais les cotes ;
- les cadres étiquetés « C1 », « C2 »… au-dessus de la section ; la légende des cadres et des latérales dessous.

Chaque pièce est un groupe ``<g data-sel="s<i>.<pièce>">`` (pièces : beton, efforts, lit-inf-<j>, lit-sup-<j>,
cadre-<k>, lat) : un clic sur la pièce ou sur son annotation la choisit (js/compact.js), et le panneau de gauche ouvre
ses propriétés. Les tailles des textes et des marges suivent la taille de la section : une poutre 20 × 40 et une poutre
80 × 120 se lisent de la même façon.
"""
import math
from html import escape
from typing import Dict, List, Optional

ENCRE = "#1B2230"
BETON = "#EEF1F5"
BETON_BORD = "#1B2230"
ACIER = "#1F2733"
CADRE = "#4A5568"
COTE = "#8A94A3"
EFFORT = "#9C3341"
DOUX = "#6B7480"
POLICE = "Inter,system-ui,Arial,sans-serif"
C45 = math.sqrt(0.5)


def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _esc(t) -> str:
    return escape(str(t), quote=True)


class _Plume:
    """Les tailles du dessin, proportionnelles à la section (unité u : 1 pour une section de 400 mm)."""

    def __init__(self, b: float, h: float):
        self.u = max(b, h, 250.0) / 400.0
        self.F = 15.0 * self.u          # texte courant
        self.f = 12.5 * self.u          # texte secondaire
        self.trait = 1.1 * self.u       # cotes
        self.contour = 2.0 * self.u     # béton


def _txt(x, y, t, taille, ancre="start", couleur=ENCRE, poids=500, extra="", classe="") -> str:
    cl = f' class="{classe}"' if classe else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}" font-weight="{poids}"{cl}{extra}>{t}</text>')


def _ligne(x1, y1, x2, y2, couleur, ep, extra="") -> str:
    return (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"'
            f' stroke-linecap="round"{extra}/>')


def _cote_h(pl: _Plume, x1, x2, y, texte, y_attache=None) -> str:
    t = 4.5 * pl.u
    out = ""
    if y_attache is not None:
        for x in (x1, x2):
            out += _ligne(x, y_attache, x, y + t * 1.4, COTE, pl.trait * 0.7, ' stroke-dasharray="3 3"')
    out += (_ligne(x1 - t * 1.4, y, x2 + t * 1.4, y, COTE, pl.trait)
            + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, pl.trait * 1.1) + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, pl.trait * 1.1)
            + _txt((x1 + x2) / 2, y - 5 * pl.u, texte, pl.F, "middle"))
    return out


def _cote_v(pl: _Plume, x, y1, y2, texte, droite=False, x_attache=None) -> str:
    t = 4.5 * pl.u
    out = ""
    if x_attache is not None:
        for y in (y1, y2):
            out += _ligne(x_attache, y, x + (t * 1.4 if x > x_attache else -t * 1.4), y, COTE, pl.trait * 0.7, ' stroke-dasharray="3 3"')
    xt = x + (pl.F * 1.05 if droite else -6 * pl.u)
    ym = (y1 + y2) / 2
    out += (_ligne(x, y1 - t * 1.4, x, y2 + t * 1.4, COTE, pl.trait)
            + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, pl.trait * 1.1) + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, pl.trait * 1.1)
            + _txt(xt, ym, texte, pl.F, "middle", extra=f' transform="rotate(-90 {_f(xt)} {_f(ym)})"'))
    return out


def _nb(x: float, nd: int = 0) -> str:
    s = f"{x:.{nd}f}".replace(".", ",")
    return s


# ------------------------------------------------------------------ les cadres
def _cadre(x0, x1, y0, y1, de: float, couleur: str, ep: float, crochets: bool, pl: _Plume) -> str:
    """Un étrier fermé : rectangle sur la ligne moyenne (x0..x1, y0..y1 = nus EXTÉRIEURS), angles cintrés, et sa
    fermeture en haut à gauche — deux crochets à 135° qui rentrent vers le noyau."""
    xa, xb, ya, yb = x0 + de / 2, x1 - de / 2, y0 + de / 2, y1 - de / 2
    w, h = xb - xa, yb - ya
    r = min(2.5 * de, w / 2.2, h / 2.2)
    d = (f"M{_f(xa + r)},{_f(ya)} H{_f(xb - r)} A{_f(r)},{_f(r)} 0 0 1 {_f(xb)},{_f(ya + r)} V{_f(yb - r)}"
         f" A{_f(r)},{_f(r)} 0 0 1 {_f(xb - r)},{_f(yb)} H{_f(xa + r)} A{_f(r)},{_f(r)} 0 0 1 {_f(xa)},{_f(yb - r)}"
         f" V{_f(ya + r)} A{_f(r)},{_f(r)} 0 0 1 {_f(xa + r)},{_f(ya)} Z")
    out = (f'<path class="st" d="{d}" fill="none" stroke="{couleur}" stroke-width="{_f(ep)}" stroke-linejoin="round"/>')
    if crochets:
        # le milieu de l'arc d'angle, puis deux retours parallèles à 45° vers l'intérieur
        cx, cy = xa + r - r * C45, ya + r - r * C45
        lg = min(max(10.0 * de, 70.0) * 0.55, 0.36 * min(w, h))
        nx, ny = -C45, C45          # la normale au retour
        for k in (-0.62, 0.62):
            bx, by = cx + nx * k * de, cy + ny * k * de
            out += (f'<line class="st" x1="{_f(bx)}" y1="{_f(by)}" x2="{_f(bx + lg * C45)}" y2="{_f(by + lg * C45)}"'
                    f' stroke="{couleur}" stroke-width="{_f(ep * 0.82)}" stroke-linecap="round"/>')
    return out


def _epingle(xb, yb_haut, yb_bas, r_haut, r_bas, de: float, couleur: str, ep: float) -> str:
    """Une épingle : un brin vertical au contact de la barre (à sa droite), enroulé autour des barres haute et basse."""
    rh, rb = r_haut + de / 2, r_bas + de / 2
    lg = max(4.0 * de, 6.0)
    d = (f"M{_f(xb - rh - lg * C45)},{_f(yb_haut + lg * C45)} L{_f(xb - rh)},{_f(yb_haut)}"
         f" A{_f(rh)},{_f(rh)} 0 0 1 {_f(xb + rh)},{_f(yb_haut)} V{_f(yb_bas)}"
         f" A{_f(rb)},{_f(rb)} 0 0 1 {_f(xb - rb)},{_f(yb_bas)} L{_f(xb - rb - lg * C45)},{_f(yb_bas - lg * C45)}")
    return f'<path class="st" d="{d}" fill="none" stroke="{couleur}" stroke-width="{_f(ep)}" stroke-linecap="round" stroke-linejoin="round"/>'


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


def coupe(geo: Dict, i: int, efforts: str = "", d_utile: Optional[float] = None, avec_cotes: bool = True,
          titre: str = "") -> Dict:
    """Le dessin d'UNE section : {"svg" (groupe <g> en mm, origine au coin haut gauche du béton), "x0", "x1", "y0",
    "y1" (son emprise, annotations comprises)}. ``i`` : l'indice de la section (préfixe des pièces, « s<i>. »)."""
    b, h = geo["b"] * 10.0, geo["h"] * 10.0
    pl = _Plume(b, h)
    c = geo["enrobage_calcul"] * 10.0
    p = f"s{i}."
    out = []
    # ---- le béton et ses cotes
    beton = (f'<rect x="0" y="0" width="{_f(b)}" height="{_f(h)}" fill="{BETON}" stroke="{BETON_BORD}" '
             f'stroke-width="{_f(pl.contour)}"/>')
    yb = h + 2.6 * pl.F
    cotes = ""
    if avec_cotes:
        cotes += _cote_h(pl, 0, b, yb, f"b = {_nb(b)}", y_attache=h + 4 * pl.u)
        cotes += _cote_v(pl, -3.4 * pl.F, 0, h, f"h = {_nb(h)}", x_attache=-4 * pl.u)
        cotes += _txt(c / 2, -0.7 * pl.F, f"c = {_nb(c)}", pl.f, "start", DOUX, 400)
    out.append(f'<g class="pc beton" data-sel="{p}beton">{beton}{cotes}</g>')

    # ---- les barres : positions des lits (y depuis le haut)
    def y_lit(face, lit):
        return h - lit["distance"] * 10.0 if face == "inf" else lit["distance"] * 10.0

    # ---- les cadres (sous les barres), puis les épingles (dessus)
    lignes = geo["etriers"].get("cadres") or []
    de_max = geo["etriers"]["d_max"]
    y0c, y1c = c, h - c
    etiquettes_cadres = []
    fermes = [(k, cd) for k, cd in enumerate(lignes) if cd["type"] != "epingle"]

    def niveau(k, cd):
        """Le rang d'imbrication : combien d'autres cadres l'entourent (un cadre complet répété compte après le premier)."""
        n = 0
        for k2, c2 in fermes:
            if k2 == k:
                continue
            dedans = c2["x0"] <= cd["x0"] + 1e-9 and c2["x1"] >= cd["x1"] - 1e-9
            egal = abs(c2["x0"] - cd["x0"]) < 1e-9 and abs(c2["x1"] - cd["x1"]) < 1e-9
            if dedans and (not egal or k2 < k):
                n += 1
        return n

    for k, cd in fermes:
        de = float(cd["d"])
        ep = de * 0.9
        lv = niveau(k, cd)
        dec = lv * de * 1.1
        if cd["complet"]:
            x0, x1 = c + dec, b - c - dec
        else:
            x0, x1 = cd["x0"] * 10.0, cd["x1"] * 10.0
        y0, y1 = y0c + dec, y1c - dec
        g = _cadre(x0, x1, y0, y1, de, CADRE, ep, True, pl)
        # l'ancre de l'étiquette : sur la branche haute, décalée selon le rang (deux cadres centrés ne la partagent pas)
        etiquettes_cadres.append((k, x0 + (x1 - x0) * min(0.82, 0.5 + 0.16 * lv), y0))
        out.append(f'<g class="pc cadre" data-sel="{p}cadre-{k}">{g}</g>')
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
    # ---- les épingles (au-dessus des barres, au contact de leur barre)
    xs1 = geo["inf"]["lits"][0]["x"]
    r1i = geo["inf"]["lits"][0]["d"] / 2.0
    r1s = geo["sup"]["lits"][0]["d"] / 2.0
    y_haut = y_lit("sup", geo["sup"]["lits"][0])
    y_bas = y_lit("inf", geo["inf"]["lits"][0])
    for k, cd in enumerate(lignes):
        if cd["type"] != "epingle":
            continue
        de = float(cd["d"])
        xb = cd["x0"] * 10.0
        g = _epingle(xb, y_haut, y_bas, r1s, r1i, de, CADRE, de * 0.9)
        etiquettes_cadres.append((k, xb, y_haut))
        out.append(f'<g class="pc cadre" data-sel="{p}cadre-{k}">{g}</g>')

    # ---- la cote d (hauteur utile de la face tendue), à gauche entre la section et la cote h ; les annotations des
    #      lits à droite, sans rien à traverser
    if avec_cotes and d_utile is not None and d_utile > 0:
        tendue = "inf" if geo.get("tendu", "inf") == "inf" else "sup"
        yG = geo[tendue]["yG"] * 10.0
        ya, yz = (0.0, h - yG) if tendue == "inf" else (yG, h)
        x_d = -1.5 * pl.F
        out.append(f'<g class="pc beton" data-sel="{p}beton">{_cote_v(pl, x_d, ya, yz, f"d = {_nb(d_utile * 10.0)}", False, x_attache=-4 * pl.u)}</g>')
    droite = b + 0.6 * pl.F
    x_txt = droite + 2.4 * pl.F
    ecart = 1.45 * pl.F
    largeur_txt = 0.0
    for face, sens in (("sup", -1), ("inf", 1)):
        items = annot[face]
        if not items:
            continue
        # du parement vers le cœur : on empile vers l'extérieur de la face, puis on remet dans l'ordre des lits
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
                     f' fill="none" stroke="{COTE}" stroke-width="{_f(pl.trait * 0.8)}"/>'
                     f'<circle cx="{_f(xr + 1.5 * pl.u)}" cy="{_f(y)}" r="{_f(1.6 * pl.u)}" fill="{COTE}"/>')
            out.append(f'<g class="pc lit" data-sel="{p}lit-{face}-{j}">{ligne}'
                       f'{_txt(x_txt, yt, _esc(libelle), pl.F, "start", ENCRE, 600, classe="an")}</g>')
    x_fin = x_txt + max(largeur_txt, 5 * pl.F)

    # ---- les étiquettes des cadres au-dessus, la légende dessous
    y_top = -0.7 * pl.F
    if etiquettes_cadres:
        y_et = -1.9 * pl.F
        tri = sorted(etiquettes_cadres, key=lambda e: e[1])
        xs_et = []
        for _, xm, _ in tri:
            xs_et.append(max(xm, xs_et[-1] + 2.3 * pl.f) if xs_et else xm)
        # recentrer le groupe s'il a glissé vers la droite
        glisse = (xs_et[-1] - tri[-1][1]) / 2 if len(tri) > 1 else 0.0
        for (k, xm, yc), xe in zip(tri, xs_et):
            xe -= glisse
            out.append(f'<g class="pc cadre" data-sel="{p}cadre-{k}">'
                       f'{_ligne(xm, yc - 1 * pl.u, xe, y_et + 0.35 * pl.F, COTE, pl.trait * 0.7)}'
                       f'{_txt(xe, y_et, f"C{k + 1}", pl.f, "middle", CADRE, 600, classe="an")}</g>')
        y_top = y_et - pl.F
    if efforts:
        y_ef = y_top - 0.6 * pl.F
        out.append(f'<g class="pc efforts" data-sel="{p}efforts">{_txt(b / 2, y_ef, efforts, pl.F, "middle", EFFORT, 600)}</g>')
        y_top = y_ef - pl.F
    leg = []
    if lignes:
        groupes: Dict[str, List[int]] = {}
        for k, cd in enumerate(lignes):
            nom = ("Épingle" if cd["type"] == "epingle" else "Étrier") + f" Ø{int(cd['d'])}"
            groupes.setdefault(nom, []).append(k + 1)
        morceaux = [", ".join(f"C{n}" for n in ns) + f" : {nom}" for nom, ns in groupes.items()]
        pas = geo["etriers"].get("pas")
        leg.append((f"{' · '.join(morceaux)}" + (f" — e = {_nb(pas * 10.0)}" if pas else ""), "cadre", ""))
    if lat.get("n"):
        leg.append((f"Latérales : 2 × {lat['n']}Ø{int(lat['d'])}{' (auto)' if lat.get('auto') else ''}", "lat", "lat"))
    y_leg = (yb + 1.9 * pl.F) if avec_cotes else (h + 1.9 * pl.F)
    for t, cle, sel in leg:
        attr = f' data-sel="{p}{sel}"' if sel else ""
        out.append(f'<g class="pc leg {cle}"{attr}>{_txt(b / 2, y_leg, _esc(t), pl.f, "middle", DOUX, 500)}</g>')
        y_leg += 1.45 * pl.f
    x_debut = (-3.4 * pl.F - 1.6 * pl.F) if avec_cotes else -pl.F
    if titre:
        out.append(_txt(b / 2, y_top - 0.2 * pl.F, _esc(titre), pl.F * 1.1, "middle", ENCRE, 650))
        y_top -= 1.4 * pl.F
    return {"svg": "".join(out), "x0": x_debut, "x1": x_fin, "y0": y_top - 0.4 * pl.F, "y1": y_leg, "u": pl.u}


def svg(ident: str, coupes: List[Dict]) -> str:
    """Une ou plusieurs coupes côte à côte (même échelle), dans un <svg> en mm. ``coupes`` : les dicts de ``coupe``."""
    morceaux, x = [], 0.0
    y0 = min(cp["y0"] for cp in coupes)
    y1 = max(cp["y1"] for cp in coupes)
    for cp in coupes:
        dx = x - cp["x0"]
        morceaux.append(f'<g transform="translate({_f(dx)} 0)">{cp["svg"]}</g>')
        x = dx + cp["x1"] + 3.0 * 15.0 * cp["u"]
    largeur = x - 3.0 * 15.0 * coupes[-1]["u"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" id="{_esc(ident)}" class="coupe-ba" viewBox="{_f(0)} {_f(y0)} {_f(largeur)} {_f(y1 - y0)}"'
            f' preserveAspectRatio="xMidYMid meet" font-family="{POLICE}">{"".join(morceaux)}</svg>')

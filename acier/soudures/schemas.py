# -*- coding: utf-8 -*-
"""Les SCHÉMAS TECHNIQUES des configurations de soudures, sur le kit de dessin
des assemblages (``acier.assemblages.poutre_poutre.doubles_cornieres.schemas`` :
feuille de cotation, primitives, sections réelles du catalogue, hachures,
lignes de rupture, bandes de cordon à leur taille a·√2, symboles de soudure
EN 22553, rendu SVG) — le même trait que les dessins d'assemblage, et le même
peintre vectoriel pour la note (``rapport.peindre``).

Chaque configuration a deux ou trois vues, chacune un ``Dessin`` (scène en mm,
y vers le bas) :

- la VUE DU PLAN DE JOINT (regard selon z, la normale au plan de joint du
  moteur) : les pièces, les cordons TELS QUE CALCULÉS (``res["cordons"]`` :
  bandes de largeur a·√2 hors de la pièce rapportée, chaîne par chaîne, cordons
  exclus en interrompu), les efforts au point de référence R (V_x, V_y en
  flèches, N en ⊙/⊗, M_x, M_y en vecteurs ⇒, M_z en arc ; les valeurs dans un
  bloc sous le dessin), le centre des gorges G et le point gouvernant ◉ avec
  son taux ;
- une COUPE ou une ÉLÉVATION : la jonction en section — cordons d'angle
  (triangles), pénétration partielle (chanfrein rempli, renfort), pleine
  pénétration (chanfrein K sur toute l'épaisseur) — avec les efforts qui
  travaillent dans ce plan, et le symbole de soudure.

Construction d'une vue : le corps (pièces, cordons, efforts, repères) d'abord,
puis la feuille de cotation sur l'ÉTENDUE RÉELLE du corps — les cotes et les
noms se posent hors de tout, jamais sur une flèche d'effort.

Rien n'est recalculé ici : géométrie, cordons, efforts, point gouvernant et
taux viennent de ``etat.verifier``. Pureté : aucun Streamlit, aucun moteur.
"""
import math
from typing import Dict, List, Optional, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # kit partagé
    Dessin, Feuille, Options, PALETTE, _arc, _circle, _group, _hachures, _path, _poly, _rect, _rect_rompu,
    _section_I, _text, vers_svg)
from acier.formats import F, f0

from . import configurations as CF

S2 = math.sqrt(2.0)


def options_schemas(lvl: int = 2) -> Options:
    """Options des schémas : toutes les cotes (niveau 2) ou les principales (1), sans interaction ni cartouche."""
    return Options(lvl=lvl, interactive=False, cartouche=False, realiste=True)


# ====================================================================== briques
def _fs(w: float) -> float:
    """Taille de police de la scène, comme le kit : lisible quelle que soit l'étendue du dessin."""
    return max((w + 260.0) / 34.0, 9.0)


def _kn(v: float) -> str:
    return F(abs(v) / 1e3, 2 if abs(v) < 10e3 else 1) + " kN"      # deux décimales sous 10 kN (0,45 kN, pas 0,5)


def _knm(v: float) -> str:
    return F(abs(v) / 1e6, 2) + " kN·m"


def _tw(txt, fs) -> float:
    return len(str(txt)) * fs * 0.56


def _label(s, x, y, txt, fs, cls=("fzl",), anchor="middle"):
    """Étiquette à halo blanc (groupe transformé, comme les étiquettes de cote du kit), centrée en (x, y)."""
    w = _tw(txt, fs)
    hit = _rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs)
    hit["rx"] = 0.2 * fs
    s.append(_group(list(cls), [hit, _text([], None, None, txt, fs, anchor)], x=x, y=y, rot=0))


def _extent(prims, fs) -> Optional[Tuple[float, float, float, float]]:
    """L'étendue (x1, y1, x2, y2) d'une liste de primitives (textes estimés)."""
    xs, ys = [], []
    for p in prims:
        t = p["t"]
        if t == "rect":
            xs += [p["x"], p["x"] + p["w"]]; ys += [p["y"], p["y"] + p["h"]]
        elif t == "poly":
            xs += [q[0] for q in p["pts"]]; ys += [q[1] for q in p["pts"]]
        elif t == "path":
            for l in p["lignes"]:
                xs += [q[0] for q in l]; ys += [q[1] for q in l]
        elif t == "circle":
            xs += [p["cx"] - p["r"], p["cx"] + p["r"]]; ys += [p["cy"] - p["r"], p["cy"] + p["r"]]
        elif t == "text" and p.get("x") is not None:
            w = _tw(p["txt"], p["size"])
            a = p.get("anchor") or "middle"
            x0, x1 = (p["x"] - w / 2, p["x"] + w / 2) if a == "middle" else ((p["x"], p["x"] + w) if a == "start" else (p["x"] - w, p["x"]))
            xs += [x0, x1]; ys += [p["y"] - p["size"], p["y"] + 0.3 * p["size"]]
        elif t == "g":
            sub = _extent(p["enfants"], fs)
            if sub is None:
                continue
            if "x" in p:
                if p.get("rot"):
                    xs += [p["x"] + sub[1], p["x"] + sub[3]]; ys += [p["y"] - sub[2], p["y"] - sub[0]]
                else:
                    xs += [p["x"] + sub[0], p["x"] + sub[2]]; ys += [p["y"] + sub[1], p["y"] + sub[3]]
            else:
                xs += [sub[0], sub[2]]; ys += [sub[1], sub[3]]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _bb(bb0: Dict, prims, fs, marge=0.4) -> Dict:
    """La boîte de vue : celle des pièces élargie à l'étendue du corps (efforts, repères, étiquettes)."""
    ex = _extent(prims, fs)
    if ex is None:
        return dict(bb0)
    m = marge * fs
    return dict(x1=min(bb0["x1"], ex[0] - m), x2=max(bb0["x2"], ex[2] + m), y1=min(bb0["y1"], ex[1] - m), y2=max(bb0["y2"], ex[3] + m))


def _etendre(S, prims, fs):
    """Après la pose des symboles : la feuille englobe aussi ce qu'ils ont ajouté au corps."""
    ex = _extent(prims, fs)
    if ex is None:
        return
    S.ex["x1"] = min(S.ex["x1"], ex[0]); S.ex["x2"] = max(S.ex["x2"], ex[2])
    S.ex["y1"] = min(S.ex["y1"], ex[1]); S.ex["y2"] = max(S.ex["y2"], ex[3])


def _fleche(s, p, q, txt, fs, etiq=None, cote=+1):
    """Flèche d'effort de p vers q (pointe en q) ; étiquette : au-dessus/au-dessous du fût
    (horizontale, ``cote`` −1 = dessous), à côté du fût (verticale, ``cote`` +1 = à droite), ou en ``etiq``."""
    (x0, y0), (x1, y1) = p, q
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    hl, hw = 1.0 * fs, 0.38 * fs
    s.append(_path(["fz"], [[(x0, y0), (x1 - ux * hl * 0.6, y1 - uy * hl * 0.6)]]))
    s.append(_poly(["fzh"], [(x1, y1), (x1 - ux * hl + nx * hw, y1 - uy * hl + ny * hw), (x1 - ux * hl - nx * hw, y1 - uy * hl - ny * hw)]))
    if txt:
        if etiq is None:
            if abs(ux) > 0.7:
                etiq = ((x0 + x1) / 2, (y0 + y1) / 2 + (0.95 * fs + 0.34 * fs if cote < 0 else -0.75 * fs))
            else:
                etiq = ((x0 + x1) / 2 + cote * (_tw(txt, fs) / 2 + 0.7 * fs), (y0 + y1) / 2 + 0.34 * fs)
        _label(s, etiq[0], etiq[1], txt, fs)


def _fleche_axe(s, p, d, tire, txt, fs, cote=-1, long=4.5, bout=False):
    """Effort au bout d'une pièce, dans l'axe ``d`` (unitaire, vers l'extérieur) : ``tire`` → la flèche
    part du bout vers l'extérieur (traction), sinon elle vient de l'extérieur sur le bout (compression).
    ``bout`` : étiquette au-delà de l'extrémité libre (flèches verticales)."""
    (x, y), (dx, dy) = p, d
    a = (x + dx * 0.2 * fs, y + dy * 0.2 * fs)
    b = (x + dx * long * fs, y + dy * long * fs)
    etiq = (b[0], b[1] + dy * 1.2 * fs + 0.34 * fs) if bout else None
    if tire:
        _fleche(s, a, b, txt, fs, etiq=etiq, cote=cote)
    else:
        _fleche(s, b, a, txt, fs, etiq=etiq, cote=cote)


def _arc_moment(s, c, r, sens, txt, fs, etiq=None):
    """Moment dans le plan de la vue : arc de 250° autour de c, pointe au bout ; ``sens`` +1 =
    trigonométrique vu à l'écran. Étiquette à droite de l'arc par défaut."""
    cx, cy = c
    a0, a1 = math.radians(200), math.radians(-50)
    if sens < 0:
        a0, a1 = -a0, -a1
    pts = _arc(cx, cy, r, a0, a1, n=8)
    s.append(_path(["fz"], [pts]))
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    L = math.hypot(xb - xa, yb - ya) or 1.0
    ux, uy = (xb - xa) / L, (yb - ya) / L
    nx, ny = -uy, ux
    hl, hw = 0.9 * fs, 0.36 * fs
    s.append(_poly(["fzh"], [(xb + ux * hl * 0.5, yb + uy * hl * 0.5), (xb - ux * hl * 0.5 + nx * hw, yb - uy * hl * 0.5 + ny * hw),
                             (xb - ux * hl * 0.5 - nx * hw, yb - uy * hl * 0.5 - ny * hw)]))
    if txt:
        _label(s, *(etiq or (cx + r + _tw(txt, fs) / 2 + 0.9 * fs, cy + 0.34 * fs)), txt, fs)


def _vecteur_moment(s, p, d, txt, fs, long, etiq=None):
    """Vecteur moment (règle de la main droite) : flèche à DOUBLE pointe ⇒ depuis p dans la direction d."""
    (x0, y0), (dx, dy) = p, d
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    x1, y1 = x0 + ux * long, y0 + uy * long
    hl, hw = 1.0 * fs, 0.38 * fs
    s.append(_path(["fz"], [[(x0, y0), (x1 - ux * hl * 0.6, y1 - uy * hl * 0.6)]]))
    for k in (0.0, 0.7):
        xt, yt = x1 - ux * hl * k, y1 - uy * hl * k
        s.append(_poly(["fzh"], [(xt, yt), (xt - ux * hl + nx * hw, yt - uy * hl + ny * hw), (xt - ux * hl - nx * hw, yt - uy * hl - ny * hw)]))
    if txt:
        if etiq is None:
            etiq = (x1 + ux * (_tw(txt, fs) / 2 + 0.9 * fs), y1 + 0.34 * fs) if abs(ux) > 0.7 else (x1, y1 + uy * 1.2 * fs + 0.34 * fs)
        _label(s, etiq[0], etiq[1], txt, fs)


def _normal(s, p, vers_soi, txt, fs, etiq=None):
    """Effort normal au plan de la vue : ⊙ (vers soi) ou ⊗ (vers le fond)."""
    x, y = p
    r = 0.55 * fs
    s.append(_circle(["gv"], x, y, r))
    if vers_soi:
        s.append(_circle(["gvd"], x, y, 0.18 * fs))
    else:
        k = r * 0.62
        s.append(_path(["fz"], [[(x - k, y - k), (x + k, y + k)], [(x - k, y + k), (x + k, y - k)]]))
    if txt:
        _label(s, *(etiq or (x - r - _tw(txt, fs) / 2 - 0.7 * fs, y - 0.9 * fs)), txt, fs)


def _gouvernant(s, res, X, Y, fs, loin=None):
    """Le point gouvernant ◉ et son taux : l'étiquette s'écarte du centre des gorges G (ou de ``loin``),
    vers l'extérieur du groupe, pour ne pas couvrir les cordons."""
    g = res.get("gouvernant")
    if not g or "point" not in g:
        return
    x, y = X(g["point"]), Y(g["point"])
    s.append(_circle(["gv"], x, y, 0.5 * fs))
    s.append(_circle(["gvd"], x, y, 0.2 * fs))
    pr = res.get("proprietes") or {}
    gx, gy = loin if loin else ((X((pr["xG"], pr["yG"], 0.0)), Y((pr["xG"], pr["yG"], 0.0))) if "xG" in pr else (x - 1.0, y + 1.0))
    dx, dy = x - gx, y - gy
    L = math.hypot(dx, dy) or 1.0
    dx, dy = dx / L, dy / L
    txt = f"◉ {g['taux'] * 100:.0f} %"
    w = _tw(txt, fs)
    e = res.get("efforts")
    r_ref = (X(e.reference), Y(e.reference)) if e is not None else None

    def pos(dx, dy):
        if abs(dx) >= abs(dy):
            return x + dx * (0.6 * fs + w / 2 + 0.5 * fs), y + dy * 0.6 * fs + 0.34 * fs
        return x + dx * 0.4 * fs, y + dy * (1.3 * fs) + 0.34 * fs + (0.5 * fs if dy > 0 else 0.0)
    lx, ly = pos(dx, dy)
    if r_ref is not None and math.hypot(lx - r_ref[0], ly - r_ref[1]) < 4.5 * fs:     # trop près des flèches d'effort : de l'autre côté
        lx, ly = pos(-dx, -dy)
    _label(s, lx, ly, txt, fs)


def _trait_coupe(s, p, q, fs, nom="A"):
    """Trait de coupe : mixte fin avec la lettre aux deux bouts."""
    (x0, y0), (x1, y1) = p, q
    s.append(_path(["ax"], [[p, q]]))
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    for (x, y), sg in ((p, -1), (q, +1)):
        s.append(_path(["fl2"], [[(x + sg * ux * 0.1 * fs, y + sg * uy * 0.1 * fs), (x + sg * ux * 1.2 * fs, y + sg * uy * 1.2 * fs)]]))
        s.append(_text(["tx"], x + sg * ux * 1.9 * fs, y + sg * uy * 1.9 * fs + 0.34 * fs, nom, fs))


def _hach(cls, pts, pas=None):
    return _path(cls, _hachures(pts, pas) if pas else _hachures(pts))


def _nom(S, side, c, txt, cls=None):
    """Un texte dans une rangée libre du côté ``side`` (comme ``Feuille.name``), de la classe voulue
    (``tx`` ink gras par défaut ; ``fzl`` = ligne d'efforts, rouge à halo)."""
    fs = S.fs
    w = _tw(txt, fs) * 1.08
    r = S.slot(side, c - w / 2, c + w / 2)
    p = S.pos(side, r)
    S.ex["x1"] = min(S.ex["x1"], c - w / 2); S.ex["x2"] = max(S.ex["x2"], c + w / 2)
    if cls == "fzl":
        hit = _rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs)
        hit["rx"] = 0.2 * fs
        S.g.append(_group(["fzl"], [hit, _text([], None, None, txt, fs)], x=c, y=p + (0.3 if side == "B" else -0.1) * fs, rot=0))
    else:
        S.g.append(_text([cls or "tx"], c, p + (0.3 if side == "B" else -0.1) * fs, txt, fs))


def _bloc_efforts(S, items: List[str], c: float, side: str = "B", par_ligne: int = 3):
    """Les valeurs des efforts, en lignes de ``par_ligne`` items, dans les rangées libres du côté ``side``."""
    for i in range(0, len(items), par_ligne):
        _nom(S, side, c, " · ".join(items[i:i + par_ligne]), cls="fzl")


# ---------------------------------------------------------------- symboles de soudure (EN 22553)
def _symbole(S, s, ax, ay, fs, jon: Dict, vers=(+1, +1)):
    """Symbole de soudure accroché en (ax, ay) : flèche, ligne de référence coudée du côté ``vers``,
    puis le symbole élémentaire — triangle (cordon d'angle, « a … »), demi-V (pénétration partielle,
    « s p »), V (pleine pénétration) — et sa désignation."""
    sx, sy = vers
    ex, ey = ax + sx * 1.7 * fs, ay + sy * 1.5 * fs
    typ = jon.get("type", "angle")
    if typ == "angle":
        txt = "a " + f0(jon["a"])
    elif typ == "partielle":
        txt = "s " + f0(jon["p"]) + (" + a " + f0(jon["renfort"]) if jon.get("renfort", 0) > 0 else "")
    else:
        txt = "pleine pénétration"
    wt = _tw(txt, fs)
    if sx > 0:
        tx = ex + wt + 0.9 * fs
        fin = tx + 1.6 * fs
        s.append(_path(["wsy"], [[(ax, ay), (ex, ey), (fin, ey)],
                                 [(ax + 0.55 * fs, ay + sy * 0.1 * fs), (ax, ay), (ax + 0.1 * fs, ay + sy * 0.55 * fs)]]))
        x_txt = ex + wt / 2 + 0.2 * fs
    else:
        tx = ex - wt - 1.9 * fs
        fin = tx - 0.6 * fs
        s.append(_path(["wsy"], [[(ax, ay), (ex, ey), (fin, ey)],
                                 [(ax - 0.55 * fs, ay + sy * 0.1 * fs), (ax, ay), (ax - 0.1 * fs, ay + sy * 0.55 * fs)]]))
        x_txt = tx + 1.0 * fs + 0.8 * fs + wt / 2
    if typ == "angle":
        s.append(_poly(["wst"], [(tx, ey), (tx + 1.0 * fs, ey), (tx, ey - 0.95 * fs)]))
    elif typ == "partielle":
        s.append(_path(["wsy"], [[(tx, ey - 0.95 * fs), (tx, ey), (tx + 1.0 * fs, ey)], [(tx, ey), (tx + 1.0 * fs, ey - 0.95 * fs)]]))
    else:
        s.append(_path(["wsy"], [[(tx, ey - 0.95 * fs), (tx + 0.5 * fs, ey), (tx + 1.0 * fs, ey - 0.95 * fs)]]))
    S.g.append(_group(["dm"], [S.label(x_txt, ey - 0.55 * fs, False, txt, dict(id="w", sym="a", lvl=1))]))
    S.ex["x1"] = min(S.ex["x1"], min(ax, fin, x_txt - wt / 2) - 0.6 * fs); S.ex["x2"] = max(S.ex["x2"], max(ax, fin, x_txt + wt / 2) + 0.6 * fs)
    S.ex["y1"] = min(S.ex["y1"], ey - 1.7 * fs, ay); S.ex["y2"] = max(S.ex["y2"], ey + 0.6 * fs, ay)


# ---------------------------------------------------------------- bandes de cordon dans le plan de joint
def _chaines_ordonnees(cordons: List[Dict]) -> List[List[Dict]]:
    """Les cordons groupés par chaîne, dans l'ordre du calcul (les segments d'une chaîne se suivent)."""
    out: List[List[Dict]] = []
    index: Dict[str, List[Dict]] = {}
    for c in cordons:
        nom = c.get("chaine") or c["nom"]
        if nom not in index:
            index[nom] = []
            out.append(index[nom])
        index[nom].append(c)
    return out


def _normale_vue(c, X, Y):
    ua = c["u_A"]
    nx, ny = X((ua[0], ua[1], 0.0)) - X((0, 0, 0)), Y((ua[0], ua[1], 0.0)) - Y((0, 0, 0))
    L = math.hypot(nx, ny) or 1.0
    return nx / L, ny / L


def _bande_chaine(segs: List[Dict], X, Y, k=1.0) -> List[Tuple[float, float]]:
    """Polygone de la bande d'une chaîne : la ligne de racine (p1 → … → p2) et sa parallèle à
    z = a·√2 (× k) du côté u_A (hors de la pièce) ; onglets aux sommets."""
    inner = [(X(segs[0]["p1"]), Y(segs[0]["p1"]))] + [(X(c["p2"]), Y(c["p2"])) for c in segs]
    normales = [_normale_vue(c, X, Y) for c in segs]
    ferme = bool(segs[0].get("ferme")) and len(segs) > 2
    outer = []
    for i in range(len(inner)):
        if ferme:
            n1, n2 = normales[(i - 1) % len(segs)], normales[i % len(segs)]
        else:
            n1, n2 = normales[max(0, i - 1)], normales[min(i, len(segs) - 1)]
        bx, by = n1[0] + n2[0], n1[1] + n2[1]
        Lb = math.hypot(bx, by)
        if Lb < 1e-9:
            bx, by, Lb = n2[0], n2[1], 1.0
        cos2 = max(0.4, (bx * n2[0] + by * n2[1]) / Lb)         # onglet : z / cos(θ/2), borné
        z = segs[i % len(segs)]["a"] * S2 * k / cos2
        outer.append((inner[i][0] + bx / Lb * z, inner[i][1] + by / Lb * z))
    return inner + outer[::-1]


def _bandes(s, res, X, Y):
    """Les bandes de tous les cordons dans une vue du plan de joint : cordons d'angle à a·√2 hors de la
    pièce, soudures bout à bout centrées sur la ligne de joint (largeur = pénétration), pleine
    pénétration en T : liseré de chaque côté (la pièce est soudée sur toute son épaisseur) ; exclus en interrompu."""
    for segs in _chaines_ordonnees(res.get("cordons", [])):
        c0 = segs[0]
        if c0.get("pleine") and "bord" in c0["nom"]:
            for c in segs:
                if "bord" not in c["nom"]:
                    continue
                x1, y1, x2, y2 = X(c["p1"]), Y(c["p1"]), X(c["p2"]), Y(c["p2"])
                nx, ny = _normale_vue(c, X, Y)
                z = max(2.0, 0.4 * (c.get("t_piece") or c["a"]))
                s.append(_poly(["wb"], [(x1, y1), (x2, y2), (x2 + nx * z, y2 + ny * z), (x1 + nx * z, y1 + ny * z)]))
            continue
        if c0.get("bout_a_bout") and not c0.get("continu") and not (c0.get("pleine") and res.get("config") != "C7"):
            w = c0.get("t_piece") or c0["a"]
            for c in segs:
                x1, y1, x2, y2 = X(c["p1"]), Y(c["p1"]), X(c["p2"]), Y(c["p2"])
                L = math.hypot(x2 - x1, y2 - y1) or 1.0
                nx, ny = -(y2 - y1) / L * w / 2, (x2 - x1) / L * w / 2
                s.append(_poly(["wb"], [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)]))
            continue
        k = 1.0 / S2 if (c0.get("pleine") or c0.get("bout_a_bout")) else 1.0
        s.append(_poly(["wb"], _bande_chaine(segs, X, Y, k=k)))
    for c in res.get("cordons_exclus", []):
        s.append(_path(["wbx"], [[(X(c["p1"]), Y(c["p1"])), (X(c["p2"]), Y(c["p2"]))]]))


def _efforts_joint(s, e, X, Y, fs, long) -> List[str]:
    """Les six composantes au point de référence R dans la vue du plan de joint : flèches à
    étiquette courte (le symbole), et la liste des valeurs pour le bloc sous le dessin."""
    rx, ry = X(e.reference), Y(e.reference)
    s.append(_path(["mkg"], [[(rx - 0.6 * fs, ry), (rx + 0.6 * fs, ry)], [(rx, ry - 0.6 * fs), (rx, ry + 0.6 * fs)]]))
    items = []
    sx = (1 if e.Vx > 0 else -1) if abs(e.Vx) > 0 else 0
    sy = (-1 if e.Vy > 0 else 1) if abs(e.Vy) > 0 else 0          # y du moteur vers le haut, de la scène vers le bas
    if sx:
        _fleche(s, (rx + sx * 0.3 * fs, ry), (rx + sx * long, ry), "V_x", fs, etiq=(rx + sx * (long + 1.6 * fs), ry + 0.34 * fs))
        items.append("V_x = " + _kn(e.Vx))
    if sy:
        _fleche(s, (rx, ry + sy * 0.3 * fs), (rx, ry + sy * long), "V_y", fs, etiq=(rx, ry + sy * (long + 1.2 * fs) + 0.34 * fs))
        items.append("V_y = " + _kn(e.Vy))
    if abs(e.N) > 0:
        _normal(s, (rx, ry), e.N > 0, "N", fs, etiq=(rx - 1.4 * fs, ry - 1.3 * fs))
        items.append("N = " + _kn(e.N) + (" ⊙ (arrachement)" if e.N > 0 else " ⊗ (compression)"))
    if abs(e.Mx) > 0:                                               # vecteur M_x à l'opposé de la flèche V_y
        y0 = ry - 2.2 * fs if sy > 0 else ry + 2.2 * fs
        _vecteur_moment(s, (rx, y0), (1 if e.Mx > 0 else -1, 0), "M_x", fs, long * 0.7)
        items.append("M_x = " + _knm(e.Mx))
    if abs(e.My) > 0:                                               # vecteur M_y à l'opposé de la flèche V_x
        x0 = rx - 2.2 * fs if sx > 0 else rx + 2.2 * fs
        _vecteur_moment(s, (x0, ry), (0, -1 if e.My > 0 else 1), "M_y", fs, long * 0.7)
        items.append("M_y = " + _knm(e.My))
    if abs(e.Mz) > 0:
        r = 1.5 * fs
        _arc_moment(s, (rx, ry), r, 1 if e.Mz > 0 else -1, "M_z", fs, etiq=(rx - r - 1.4 * fs, ry + 0.34 * fs))
        items.append("M_z = " + _knm(e.Mz))
    return items


def _marque_G(s, res, X, Y, fs):
    pr = res.get("proprietes") or {}
    if "xG" not in pr:
        return
    x, y = X((pr["xG"], pr["yG"], 0.0)), Y((pr["xG"], pr["yG"], 0.0))
    s.append(_circle(["mkg"], x, y, 0.45 * fs))
    s.append(_path(["mkg"], [[(x - 0.8 * fs, y), (x + 0.8 * fs, y)], [(x, y - 0.8 * fs), (x, y + 0.8 * fs)]]))
    s.append(_text(["tx2"], x + 0.7 * fs, y - 0.6 * fs, "G", 0.9 * fs, "start"))


def _proj_plan():
    """Projection du plan de joint : x → X, y (vers le haut) → −Y."""
    return (lambda p: float(p[0])), (lambda p: -float(p[1]))


def _nom_profil(rec: Dict) -> str:
    return rec.get("nom") or ""


def _jon(pieces: Dict, cle: str = "jonction") -> Dict:
    return pieces.get(cle) or {"type": "angle", "a": pieces.get("a", 5.0), "p": 0.0, "renfort": 0.0}


# ====================================================================== sections de jonction (coupes)
def _jonction_coupe(s, S, fs, racine, t, jon: Dict, up, out, deux=True, symbole=True, haut=True):
    """La jonction d'une pièce d'épaisseur t debout sur une face, en coupe : ``racine`` = milieu de la
    racine (sur la face d'appui), ``up`` = unitaire le long de la pièce (vers son corps), ``out`` =
    unitaire dans la face d'appui, vers la face 1. Cordons d'angle : deux triangles de côtés a·√2 ;
    pénétration partielle : chanfreins remplis de profondeur p (+ renfort a_r) ; pleine : chanfrein K
    sur toute l'épaisseur (les deux faces)."""
    (rx, ry), (ux, uy), (ox, oy) = racine, up, out
    typ = jon.get("type", "angle")

    def pt(d_out, d_up):
        return (rx + ox * d_out + ux * d_up, ry + oy * d_out + uy * d_up)
    faces = [(+1, True), (-1, deux)] if typ != "pleine" else [(+1, True), (-1, True)]
    for sg, actif in faces:
        if not actif:
            continue
        if typ == "angle":
            z = jon["a"] * S2
            tri = [pt(sg * t / 2, 0), pt(sg * (t / 2 + z), 0), pt(sg * t / 2, z)]
            s.append(_poly(["wb"], tri)); s.append(_hach(["ht", "htw"], tri, 2.5))
        elif typ == "partielle":
            p = jon["p"]
            ch = [pt(sg * t / 2, 0), pt(sg * (t / 2 - p), 0), pt(sg * t / 2, p * 1.15)]
            s.append(_poly(["wb"], ch)); s.append(_hach(["ht", "htw"], ch, 2.5))
            if jon.get("renfort", 0) > 0:
                zr = jon["renfort"] * S2
                tri = [pt(sg * t / 2, 0), pt(sg * (t / 2 + zr), 0), pt(sg * t / 2, zr)]
                s.append(_poly(["wb"], tri)); s.append(_hach(["ht", "htw"], tri, 2.5))
        else:
            ch = [pt(sg * t / 2, 0), pt(0, 0), pt(sg * t / 2, t / 2 * 1.1)]
            s.append(_poly(["wb"], ch)); s.append(_hach(["ht", "htw"], ch, 2.5))
            zr = max(2.0, 0.15 * t)
            s.append(_poly(["wb"], [pt(sg * t / 2, 0), pt(sg * (t / 2 + zr), 0), pt(sg * t / 2, zr)]))
    if symbole:
        z = jon["a"] * S2 if typ == "angle" else (jon["p"] if typ == "partielle" else t / 2)
        ax, ay = pt(t / 2 + z * 0.7, z * 0.3)
        _symbole(S, s, ax, ay, fs, jon, vers=(+1, -1 if haut else +1))


# ====================================================================== C1 recouvrement
def c1_joint(res, opt):
    pc = res["pieces"]
    L, b, tA, tB = pc["L"], pc["b"], pc["tA"], pc["tB"]
    X, Y = _proj_plan()
    xa0, xb1 = -0.6 * L, 1.6 * L
    bb0 = dict(x1=xa0, x2=xb1, y1=-0.75 * b, y2=0.75 * b)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    s.append(_poly(["pp"], _rect_rompu(xa0, -0.75 * b, L - xa0, 1.5 * b, {"L"}, 0.55 * fs)))      # plat A, continue à gauche
    s.append(_path(["hd"], [[(L, -b / 2), (L, b / 2)]]))                                            # son bout, caché sous B
    s.append(_poly(["ps"], _rect_rompu(0, -b / 2, xb1, b, {"R"}, 0.55 * fs)))                     # plat B, continue à droite
    _bandes(s, res, X, Y)
    _trait_coupe(s, (L / 2, -0.75 * b - 1.2 * fs), (L / 2, 0.75 * b + 1.2 * fs), fs)
    e = res["efforts"]
    if abs(e.Vx) > 0:
        _fleche_axe(s, (xb1, 0), (1, 0), e.Vx > 0, "N = " + _kn(e.Vx), fs)
        _fleche_axe(s, (xa0, 0), (-1, 0), e.Vx > 0, "", fs)
    if abs(e.Vy) > 0:
        sg = -1 if e.Vy > 0 else 1
        _fleche(s, (xb1 - 2.0 * fs, sg * (b / 2 + 0.3 * fs)), (xb1 - 2.0 * fs, sg * (b / 2 + 4.0 * fs)), "V = " + _kn(e.Vy), fs)
    _gouvernant(s, res, X, Y, fs)
    _marque_G(s, res, X, Y, fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    c = next((c for c in res.get("cordons", []) if "latéral" in c["nom"]), None) or (res.get("cordons") or [None])[0]
    if c:
        y_c = (Y(c["p1"]) + Y(c["p2"])) / 2
        _symbole(S, s, (X(c["p1"]) + X(c["p2"])) / 2 + 0.3 * L, y_c + (0.0 if y_c > 0 else 0.0), fs, {"type": "angle", "a": pc["a"]}, vers=(+1, +1 if y_c > 0 else -1))
    S.dim(dict(id="L", side="T", a=0, b=L, o1=-b / 2, o2=-0.75 * b, sym="L_rec", val=L, lvl=1))
    S.dim(dict(id="b", side="R", a=-b / 2, b=b / 2, o1=xb1, o2=xb1, sym="b", val=b, lvl=1))
    S.dim(dict(id="bA", side="L", a=-0.75 * b, b=0.75 * b, o1=xa0, o2=xa0, sym="b_A", val=1.5 * b, lvl=2))
    _nom(S, "T", xa0 + 0.3 * L, "Plat A · t " + f0(tA))
    _nom(S, "B", xb1 - 0.3 * L, "Plat B · t " + f0(tB))
    return S.finish(s, [], "Recouvrement : vue en plan des deux plats et des cordons")


def c1_coupe(res, opt):
    pc = res["pieces"]
    b, tA, tB, a = pc["b"], pc["tA"], pc["tB"], pc["a"]
    bb0 = dict(x1=-0.75 * b, x2=0.75 * b, y1=-tB, y2=tA)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    A = [(-0.75 * b, 0), (0.75 * b, 0), (0.75 * b, tA), (-0.75 * b, tA)]
    B = [(-b / 2, -tB), (b / 2, -tB), (b / 2, 0), (-b / 2, 0)]
    s.append(_poly(["pp"], A)); s.append(_hach(["ht"], A))
    s.append(_poly(["ps"], B)); s.append(_hach(["ht"], B))
    z = a * S2
    if pc.get("lateraux", True):
        for sg in (+1, -1):
            tri = [(sg * b / 2, 0), (sg * (b / 2 + z), 0), (sg * b / 2, -z)]
            s.append(_poly(["wb"], tri)); s.append(_hach(["ht", "htw"], tri, 2.5))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    if pc.get("lateraux", True):
        _symbole(S, s, b / 2 + z * 0.7, -z * 0.3, fs, {"type": "angle", "a": a}, vers=(+1, -1))
    S.dim(dict(id="b", side="T", a=-b / 2, b=b / 2, o1=-tB, o2=-tB, sym="b", val=b, lvl=1))
    S.dim(dict(id="tB", side="L", a=-tB, b=0, o1=-b / 2, o2=-b / 2, sym="t_B", val=tB, lvl=1))
    S.dim(dict(id="tA", side="L", a=0, b=tA, o1=-0.75 * b, o2=-0.75 * b, sym="t_A", val=tA, lvl=1))
    _nom(S, "B", 0, "Coupe A-A")
    return S.finish(s, [], "Recouvrement : coupe des deux plats et des cordons latéraux")


# ====================================================================== C2 jonction en T
def c2_coupe(res, opt):
    pc = res["pieces"]
    tA, tB, hB = pc["tA"], pc["tB"], pc["hB"]
    jon = _jon(pc)
    w = max(60.0, 3.0 * tB, 2.5 * (jon["a"] * S2 if jon["type"] == "angle" else jon["p"]) + tB) + 20
    bb0 = dict(x1=-w, x2=w, y1=-hB, y2=tA)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    A = _rect_rompu(-w, 0, 2 * w, tA, {"L", "R"}, 0.55 * fs)
    s.append(_poly(["pp"], A)); s.append(_hach(["ht"], A))
    B = _rect_rompu(-tB / 2, -hB, tB, hB, {"T"}, 0.55 * fs)
    s.append(_poly(["ps"], B)); s.append(_hach(["ht"], B))
    e = res["efforts"]
    if abs(e.N) > 0:
        _fleche_axe(s, (0, -hB), (0, -1), e.N > 0, "N = " + _kn(e.N), fs, bout=True)
    if abs(e.Vy) > 0:                                                # V⊥ selon y du moteur = X de la coupe (face 1 à droite)
        sg = 1 if e.Vy > 0 else -1
        _fleche(s, (0, -hB * 0.62), (sg * 4.5 * fs, -hB * 0.62), "V⊥ = " + _kn(e.Vy), fs, cote=-1)
    if abs(e.Mx) > 0:
        _arc_moment(s, (0, -hB * 0.38), 1.7 * fs, 1 if e.Mx > 0 else -1, "M = " + _knm(e.Mx), fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    _jonction_coupe(s, S, fs, (0, 0), tB, jon, (0, -1), (1, 0), deux=jon.get("deux_faces", True))
    S.dim(dict(id="tB", side="T", a=-tB / 2, b=tB / 2, o1=-hB, o2=-hB, sym="t_B", val=tB, lvl=1, out="hi"))
    S.dim(dict(id="tA", side="R", a=0, b=tA, o1=w, o2=w, sym="t_A", val=tA, lvl=1))
    _nom(S, "B", -w * 0.5, "Pièce A (support)")
    _nom(S, "T", -w * 0.55, "Pièce B")
    return S.finish(s, [], "Jonction en T : coupe A-A, la pièce debout et sa jonction")


def c2_joint(res, opt):
    pc = res["pieces"]
    l, tB = pc["l"], pc["tB"]
    X, Y = _proj_plan()
    m = max(40.0, 0.25 * l)
    wy = max(50.0, 3 * tB, 0.3 * l)
    bb0 = dict(x1=-m, x2=l + m, y1=-wy, y2=wy)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    s.append(_poly(["pp"], _rect_rompu(-m, -wy, l + 2 * m, 2 * wy, {"L", "R"}, 0.55 * fs)))
    s.append(_rect(["ps"], 0, -tB / 2, l, tB))
    _bandes(s, res, X, Y)
    _trait_coupe(s, (l / 2, -wy - 2.2 * fs), (l / 2, wy + 2.2 * fs), fs)
    e = res["efforts"]
    rx = X(e.reference)
    if abs(e.Vx) > 0:
        sg = 1 if e.Vx > 0 else -1
        _fleche(s, (rx, -wy * 0.55), (rx + sg * 4.5 * fs, -wy * 0.55), "V∥ = " + _kn(e.Vx), fs, cote=+1)
    _gouvernant(s, res, X, Y, fs)
    _marque_G(s, res, X, Y, fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="l", side="B", a=0, b=l, o1=tB / 2, o2=tB / 2, sym="l", val=l, lvl=1))
    S.dim(dict(id="tB", side="R", a=-tB / 2, b=tB / 2, o1=l, o2=l, sym="t_B", val=tB, lvl=2, out="hi"))
    _nom(S, "T", l / 2, "Vue en plan · pièce B vue par la tranche")
    return S.finish(s, [], "Jonction en T : vue en plan, les cordons le long de la pièce debout")


# ====================================================================== C3 console
def _support_cote(s, fs, pot: Optional[Dict], y1, y2, t_plat=15.0):
    """Le support vu de côté (regard selon x) : le poteau par sa profondeur h (ailes aux deux bords,
    âme entre), ou un plat ; il continue en haut et en bas. La face d'appui est à X = 0, le support à X < 0."""
    if pot:
        hp, tf = pot["h"], pot["tf"]
        s.append(_poly(["pp"], _rect_rompu(-hp, y1, hp, y2 - y1, {"T", "B"}, 0.55 * fs)))
        s.append(_path(["fl2"], [[(-tf, y1 + 2), (-tf, y2 - 2)], [(-hp + tf, y1 + 2), (-hp + tf, y2 - 2)]]))
        return hp
    s.append(_poly(["pp"], _rect_rompu(-t_plat, y1, t_plat, y2 - y1, {"T", "B"}, 0.55 * fs)))
    return t_plat


def c3_elevation(res, opt):
    pc = res["pieces"]
    h, t, ex, Lc = pc["h"], pc["t"], pc["e"], pc["longueur"]
    pot = pc.get("poteau")
    jon = _jon(pc)
    prof = pot["h"] if pot else 15.0
    marge = max(40.0, 0.35 * h)
    bb0 = dict(x1=-prof, x2=Lc, y1=-h / 2 - marge, y2=h / 2 + marge)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    _support_cote(s, fs, pot, bb0["y1"], bb0["y2"])
    s.append(_poly(["ps"], _rect_rompu(0, -h / 2, Lc, h, {"R"}, 0.55 * fs)))                  # le plat, continue à droite
    # le cordon de la face vue : bande de largeur a·√2 le long de la jonction (chaîne par chaîne)
    z = jon["a"] * S2 if jon["type"] == "angle" else ((jon["p"] + jon.get("renfort", 0.0)) * 0.9 if jon["type"] == "partielle" else t * 0.5)
    for c in res.get("cordons", []):
        if c["nom"].startswith("cordon droit") or (c.get("pleine") and "bord 1" in c["nom"]):
            y1, y2 = -max(c["p1"][1], c["p2"][1]), -min(c["p1"][1], c["p2"][1])
            s.append(_rect(["wb"], 0, y1, z, y2 - y1))
    for c in res.get("cordons_exclus", []):
        if c["chaine"].startswith("cordon droit"):
            s.append(_path(["wbx"], [[(0, -c["p1"][1]), (0, -c["p2"][1])]]))
    e = res["efforts"]
    if abs(e.Vy) > 0:
        sg = -1 if e.Vy < 0 else 1                                   # V saisi vers le bas → Vy < 0 : flèche qui descend sur le plat
        y0 = -h / 2 - 0.3 * fs if sg < 0 else h / 2 + 0.3 * fs
        _fleche(s, (ex, y0 + sg * 4.5 * fs), (ex, y0), "V = " + _kn(e.Vy), fs, cote=+1)
        s.append(_path(["ax"], [[(ex, -h / 2 - 5.5 * fs), (ex, h / 2 + 1.0 * fs)]]))
    if abs(e.N) > 0:
        _fleche_axe(s, (Lc, 0), (1, 0), e.N > 0, "N = " + _kn(e.N), fs)
    g = res.get("gouvernant")
    if g and "point" in g:
        s.append(_circle(["gv"], z / 2, -g["point"][1], 0.5 * fs)); s.append(_circle(["gvd"], z / 2, -g["point"][1], 0.2 * fs))
        _label(s, z / 2 + 3.4 * fs, -g["point"][1] - 0.9 * fs, f"◉ {g['taux'] * 100:.0f} %", fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    _symbole(S, s, z, h / 2 * 0.75, fs, jon, vers=(+1, +1))
    S.dim(dict(id="h", side="L", a=-h / 2, b=h / 2, o1=0, o2=0, sym="h", val=h, lvl=1))
    S.dim(dict(id="e", side="T", a=0, b=ex, o1=-h / 2, o2=-h / 2 - 4.0 * fs, sym="e", val=ex, lvl=1))
    if pot:
        S.dim(dict(id="hp", side="B", a=-prof, b=0, o1=bb0["y2"], o2=bb0["y2"], sym="h_p", val=prof, lvl=2))
    _nom(S, "T", Lc * 0.75, "Console · plat " + f0(h) + " × " + f0(t))
    _nom(S, "B", -prof / 2, _nom_profil(pot) if pot else "Appui")
    return S.finish(s, [], "Console : élévation, le plat soudé sur la face du poteau, V à l'excentricité e")


def c3_plan(res, opt):
    """Coupe horizontale : la section du poteau (ou le plat d'appui), le plat de la console par la tranche, la jonction."""
    pc = res["pieces"]
    h, t, Lc = pc["h"], pc["t"], pc["longueur"]
    pot = pc.get("poteau")
    jon = _jon(pc)
    Lv = min(Lc, max(90.0, 4 * t, 60.0 + 2 * (jon["a"] * S2 if jon["type"] == "angle" else t)))
    if pot:
        bp, hp = pot["b"], pot["h"]
        sec = _section_I(bp, hp, pot["tw"], pot["tf"], pot.get("r", 0.0))
    else:
        bp, hp = max(120.0, 4 * t + 60), 15.0
        sec = _rect_rompu(-bp / 2, 0, bp, hp, {"L", "R"}, 4.0)
    bb0 = dict(x1=-bp / 2, x2=bp / 2, y1=-Lv, y2=hp)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    s.append(_poly(["pp"], sec)); s.append(_hach(["ht"], sec))
    B = _rect_rompu(-t / 2, -Lv, t, Lv, {"T"}, 0.45 * fs)
    s.append(_poly(["ps"], B)); s.append(_hach(["ht"], B))
    e = res["efforts"]
    if abs(e.N) > 0:
        _fleche_axe(s, (0, -Lv), (0, -1), e.N > 0, "N = " + _kn(e.N), fs, long=4.0, bout=True)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    _jonction_coupe(s, S, fs, (0, 0), t, jon, (0, -1), (1, 0), deux=jon.get("deux_faces", True))
    S.dim(dict(id="t", side="T", a=-t / 2, b=t / 2, o1=-Lv, o2=-Lv, sym="t", val=t, lvl=1, out="hi"))
    if pot:
        S.dim(dict(id="bp", side="B", a=-bp / 2, b=bp / 2, o1=hp, o2=hp, sym="b_p", val=bp, lvl=2))
    _nom(S, "B", 0, "Coupe horizontale · " + (_nom_profil(pot) if pot else "appui"))
    return S.finish(s, [], "Console : coupe horizontale, la jonction du plat sur l'aile")


def c3_face(res, opt):
    """Vue de face (plan de joint) : l'aile du poteau, le plat par la tranche, les deux cordons."""
    pc = res["pieces"]
    h, t = pc["h"], pc["t"]
    pot = pc.get("poteau")
    X, Y = _proj_plan()
    bp = pot["b"] if pot else max(120.0, 4 * t + 60)
    marge = max(40.0, 0.3 * h)
    bb0 = dict(x1=-bp / 2, x2=bp / 2, y1=-h / 2 - marge, y2=h / 2 + marge)
    fs = _fs(bb0["x2"] - bb0["x1"] + 120)
    s = []
    s.append(_poly(["pp"], _rect_rompu(-bp / 2, bb0["y1"], bp, bb0["y2"] - bb0["y1"], {"T", "B"}, 0.55 * fs)))
    if pot:
        s.append(_path(["hd"], [[(-pot["tw"] / 2, bb0["y1"] + 2), (-pot["tw"] / 2, bb0["y2"] - 2)], [(pot["tw"] / 2, bb0["y1"] + 2), (pot["tw"] / 2, bb0["y2"] - 2)]]))
    s.append(_rect(["ps"], -t / 2, -h / 2, t, h))
    _bandes(s, res, X, Y)
    _gouvernant(s, res, X, Y, fs)
    _marque_G(s, res, X, Y, fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="h", side="R", a=-h / 2, b=h / 2, o1=t / 2, o2=t / 2, sym="h", val=h, lvl=1))
    S.dim(dict(id="bp", side="B", a=-bp / 2, b=bp / 2, o1=bb0["y2"], o2=bb0["y2"], sym="b_p", val=bp, lvl=2))
    _nom(S, "T", 0, "Vue de face · " + (_nom_profil(pot) if pot else "appui"))
    return S.finish(s, [], "Console : vue de face, les cordons verticaux sur l'aile")


# ====================================================================== C4 gousset
def c4_joint(res, opt):
    pc = res["pieces"]
    bx, by, t = pc["bx"], pc["by"], pc["t"]
    pot = pc.get("poteau")
    X, Y = _proj_plan()
    Bp = pc["B"]
    xb1 = max(x for x, _ in Bp)
    bp = pot["b"] if pot else bx + 60
    marge = max(40.0, 0.25 * by)
    xs0 = bx / 2 - bp / 2
    bb0 = dict(x1=min(xs0, -25.0), x2=max(xb1, bx + 25), y1=-by - marge, y2=marge)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    s.append(_poly(["pp"], _rect_rompu(xs0, bb0["y1"], bp, bb0["y2"] - bb0["y1"], {"T", "B"}, 0.55 * fs)))    # l'aile (ou l'appui)
    if pot:
        s.append(_path(["hd"], [[(bx / 2 - pot["tw"] / 2, bb0["y1"] + 2), (bx / 2 - pot["tw"] / 2, bb0["y2"] - 2)],
                                [(bx / 2 + pot["tw"] / 2, bb0["y1"] + 2), (bx / 2 + pot["tw"] / 2, bb0["y2"] - 2)]]))
    s.append(_poly(["ps"], [(X((x, y)), Y((x, y))) for x, y in Bp]))
    _bandes(s, res, X, Y)
    _trait_coupe(s, (xs0 - 1.0 * fs, -by / 2), (max(xb1, xs0 + bp) + 1.0 * fs, -by / 2), fs)
    items = _efforts_joint(s, res["efforts"], X, Y, fs, 4.0 * fs)
    _gouvernant(s, res, X, Y, fs)
    _marque_G(s, res, X, Y, fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    c = (res.get("cordons") or [None])[0]
    if c:
        _symbole(S, s, (X(c["p1"]) + X(c["p2"])) / 2, (Y(c["p1"]) + Y(c["p2"])) / 2, fs, {"type": "angle", "a": pc["a"]}, vers=(-1, -1))
    S.dim(dict(id="bx", side="B", a=0, b=bx, o1=0, o2=0, sym="b_x", val=bx, lvl=1))
    S.dim(dict(id="by", side="R", a=-by, b=0, o1=xb1, o2=xb1, sym="b_y", val=by, lvl=1))
    S.dim(dict(id="xV", side="T", a=0, b=pc["xV"], o1=-by, o2=-pc["yV"], sym="x_V", val=pc["xV"], lvl=2))
    S.dim(dict(id="yV", side="L", a=-pc["yV"], b=0, o1=pc["xV"], o2=0, sym="y_V", val=pc["yV"], lvl=2))
    _nom(S, "T", bx / 2, "Gousset · plat " + f0(bx) + " × " + f0(by) + " × " + f0(t))
    _nom(S, "B", xs0 + bp / 2, _nom_profil(pot) if pot else "Appui")
    _bloc_efforts(S, items, (bb0["x1"] + bb0["x2"]) / 2)
    return S.finish(s, [], "Gousset : vue en plan, le plat à plat sur l'aile et ses cordons")


def c4_coupe(res, opt):
    """Coupe A-A (regard selon x) : le support vu de côté, le plat posé à plat, les cordons des côtés haut et bas."""
    pc = res["pieces"]
    by, t, a = pc["by"], pc["t"], pc["a"]
    pot = pc.get("poteau")
    prof = pot["h"] if pot else 15.0
    marge = max(40.0, 0.3 * by)
    bb0 = dict(x1=-prof, x2=t + 30, y1=-by - marge, y2=marge)
    fs = _fs(bb0["x2"] - bb0["x1"] + 200)
    s = []
    _support_cote(s, fs, pot, bb0["y1"], bb0["y2"])
    B = [(0, -by), (t, -by), (t, 0), (0, 0)]
    s.append(_poly(["ps"], B)); s.append(_hach(["ht"], B))
    z = a * S2
    cotes = pc.get("cotes", [])
    for nom, yb, sg in (("haut", -by, -1), ("bas", 0, +1)):
        if nom in cotes:
            tri = [(0, yb), (0, yb + sg * z), (z, yb)]
            s.append(_poly(["wb"], tri)); s.append(_hach(["ht", "htw"], tri, 2.5))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    _symbole(S, s, z * 0.6, -by - z * 0.4, fs, {"type": "angle", "a": a}, vers=(+1, -1))
    S.dim(dict(id="t", side="T", a=0, b=t, o1=-by, o2=-by, sym="t", val=t, lvl=1, out="hi"))
    S.dim(dict(id="by", side="R", a=-by, b=0, o1=t, o2=t, sym="b_y", val=by, lvl=2))
    _nom(S, "B", 0, "Coupe A-A")
    return S.finish(s, [], "Gousset : coupe, le plat à plat et les cordons de ses bords")


# ====================================================================== C5 platine d'about
def c5_face(res, opt):
    pc = res["pieces"]
    rec = pc["profil"]
    h, b, tw, tf, r = rec["h"], rec["b"], rec["tw"], rec["tf"], rec.get("r", rec.get("r1", 0.0))
    X, Y = _proj_plan()
    pl = pc["platine"]
    bb0 = dict(x1=min(x for x, _ in pl), x2=max(x for x, _ in pl), y1=-max(y for _, y in pl), y2=-min(y for _, y in pl))
    fs = _fs(bb0["x2"] - bb0["x1"] + 100)
    s = []
    s.append(_rect(["pp"], bb0["x1"], bb0["y1"], bb0["x2"] - bb0["x1"], bb0["y2"] - bb0["y1"]))
    sec = [(x, y - h / 2) for x, y in _section_I(b, h, tw, tf, r)]       # section coupée, centrée
    s.append(_poly(["ps"], sec)); s.append(_hach(["ht"], sec))
    _bandes(s, res, X, Y)
    _trait_coupe(s, (bb0["x1"] - 1.2 * fs, 0), (bb0["x2"] + 1.2 * fs, 0), fs)
    items = _efforts_joint(s, res["efforts"], X, Y, fs, 4.0 * fs)
    _gouvernant(s, res, X, Y, fs)
    _marque_G(s, res, X, Y, fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    jf, jw = _jon(pc, "jonction_semelles"), _jon(pc, "jonction_ame")
    _symbole(S, s, b / 2 * 0.6, -h / 2 - (jf["a"] * S2 if jf["type"] == "angle" else 2.0), fs, jf, vers=(+1, -1))
    if any(c["nom"].startswith("âme") for c in res.get("cordons", [])):
        _symbole(S, s, tw / 2 + (jw["a"] * S2 if jw["type"] == "angle" else 1.0), h * 0.33, fs, jw, vers=(+1, +1))
    S.dim(dict(id="h", side="L", a=-h / 2, b=h / 2, o1=-b / 2, o2=-b / 2, sym="h", val=h, lvl=1))
    S.dim(dict(id="b", side="T", a=-b / 2, b=b / 2, o1=-h / 2, o2=-h / 2, sym="b", val=b, lvl=1))
    S.dim(dict(id="bpl", side="B", a=bb0["x1"], b=bb0["x2"], o1=bb0["y2"], o2=bb0["y2"], sym="b_pl", val=bb0["x2"] - bb0["x1"], lvl=2))
    _nom(S, "T", 0, _nom_profil(rec) + " sur platine t " + f0(pc.get("tp", 15)))
    _bloc_efforts(S, items, 0.0)
    return S.finish(s, [], "Platine d'about : vue de face, la section du profilé et ses cordons")


def c5_elevation(res, opt):
    """Élévation (regard selon y, l'âme vue de face) : la platine par la tranche, le profilé, les cordons des semelles."""
    pc = res["pieces"]
    rec = pc["profil"]
    h, b, tw, tf, r = rec["h"], rec["b"], rec["tw"], rec["tf"], rec.get("r", rec.get("r1", 0.0))
    tp = pc.get("tp", 15.0)
    jf, jw = _jon(pc, "jonction_semelles"), _jon(pc, "jonction_ame")
    Lv = max(1.2 * h, 160.0)
    pl_h = max(y for _, y in pc["platine"]) - min(y for _, y in pc["platine"])
    bb0 = dict(x1=-tp, x2=Lv, y1=-pl_h / 2, y2=pl_h / 2)
    fs = _fs(bb0["x2"] - bb0["x1"] + 60)
    s = []
    P = [(-tp, -pl_h / 2), (0, -pl_h / 2), (0, pl_h / 2), (-tp, pl_h / 2)]
    s.append(_poly(["pp"], P)); s.append(_hach(["ht"], P))
    s.append(_poly(["ps"], _rect_rompu(0, -h / 2, Lv, h, {"R"}, 0.55 * fs)))                   # le profilé, continue à droite
    s.append(_path(["fl2"], [[(0, -h / 2 + tf), (Lv, -h / 2 + tf)], [(0, h / 2 - tf), (Lv, h / 2 - tf)]]))
    if r > 0:
        s.append(_path(["flr"], [[(0, -h / 2 + tf + r), (Lv, -h / 2 + tf + r)], [(0, h / 2 - tf - r), (Lv, h / 2 - tf - r)]]))
    if any(c["nom"].startswith("âme") for c in res.get("cordons", [])):
        za = jw["a"] * S2 if jw["type"] == "angle" else tw * 0.5
        d = h / 2 - tf - r
        s.append(_rect(["wb"], 0, -d, za, 2 * d))
    e = res["efforts"]
    if abs(e.N) > 0:
        _fleche_axe(s, (Lv, 0), (1, 0), e.N > 0, "N = " + _kn(e.N), fs)
    if abs(e.Vy) > 0:                                                  # V_z saisi = Vy du moteur (vers le haut si > 0)
        sg = -1 if e.Vy > 0 else 1
        x = Lv * 0.6
        _fleche(s, (x, -sg * (h / 2 + 4.8 * fs)), (x, -sg * (h / 2 + 0.3 * fs)), "V_z = " + _kn(e.Vy), fs, cote=+1)
    if abs(e.Mx) > 0:
        _arc_moment(s, (Lv * 0.35, 0), 1.7 * fs, 1 if e.Mx > 0 else -1, "M_y = " + _knm(e.Mx), fs, etiq=(Lv * 0.35, 1.7 * fs + 1.4 * fs + 0.34 * fs))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    deux = jf.get("deux_faces", True)
    for sg in (-1, +1):          # semelle sup. (sg = −1, en haut de la scène) et inf.
        _jonction_coupe(s, S, fs, (0, sg * (h / 2 - tf / 2)), tf, jf, (1, 0), (0, sg), deux=deux, symbole=(sg < 0), haut=(sg < 0))
    S.dim(dict(id="h", side="L", a=-h / 2, b=h / 2, o1=-tp, o2=-tp, sym="h", val=h, lvl=1))
    S.dim(dict(id="tp", side="T", a=-tp, b=0, o1=-pl_h / 2, o2=-h / 2, sym="t_p", val=tp, lvl=1, out="lo"))
    _nom(S, "T", Lv / 2, _nom_profil(rec))
    _nom(S, "B", -tp / 2, "Platine")
    return S.finish(s, [], "Platine d'about : élévation, les cordons des semelles et de l'âme")


# ====================================================================== C6 tube sur platine
def _dans(poly, p) -> bool:
    """Point dans un polygone (règle pair-impair)."""
    x, y = p
    n = len(poly)
    inside = False
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def _est_rond(rec: Dict) -> bool:
    return rec["famille"] in ("CHS", "CHS-F", "TUBE")


def c6_face(res, opt):
    pc = res["pieces"]
    rec = pc["profil"]
    X, Y = _proj_plan()
    pl = pc["platine"]
    bb0 = dict(x1=min(x for x, _ in pl), x2=max(x for x, _ in pl), y1=-max(y for _, y in pl), y2=-min(y for _, y in pl))
    fs = _fs(bb0["x2"] - bb0["x1"] + 100)
    s = []
    s.append(_rect(["pp"], bb0["x1"], bb0["y1"], bb0["x2"] - bb0["x1"], bb0["y2"] - bb0["y1"]))
    t = float(rec.get("t", 0) or 0)
    ext = [(x, -y) for x, y in pc["contour"]]
    s.append(_poly(["ps"], ext))
    if t > 0:
        inn = [(x, -y) for x, y in CF.contour_tube(rec, t)]
        s.append(_poly(["pp"], inn))
        s.append(_path(["ht"], [l for l in _hachures(ext) if not _dans(inn, ((l[0][0] + l[1][0]) / 2, (l[0][1] + l[1][1]) / 2))]))
    _bandes(s, res, X, Y)
    _trait_coupe(s, (bb0["x1"] - 1.2 * fs, 0), (bb0["x2"] + 1.2 * fs, 0), fs)
    items = _efforts_joint(s, res["efforts"], X, Y, fs, 4.0 * fs)
    _gouvernant(s, res, X, Y, fs)
    _marque_G(s, res, X, Y, fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    jon = _jon(pc)
    px, py = max(ext, key=lambda q: q[0] + q[1])                       # le point bas-droit du pourtour
    z = (jon["a"] * S2 if jon["type"] == "angle" else 2.0) * 0.7
    _symbole(S, s, px + z, py + z, fs, jon, vers=(+1, +1))
    if _est_rond(rec):
        S.dim(dict(id="d", side="T", a=-rec["d"] / 2, b=rec["d"] / 2, o1=0, o2=0, sym="d", val=rec["d"], lvl=1))
    else:
        S.dim(dict(id="b", side="T", a=-rec["b"] / 2, b=rec["b"] / 2, o1=-rec["h"] / 2, o2=-rec["h"] / 2, sym="b", val=rec["b"], lvl=1))
        S.dim(dict(id="h", side="L", a=-rec["h"] / 2, b=rec["h"] / 2, o1=-rec["b"] / 2, o2=-rec["b"] / 2, sym="h", val=rec["h"], lvl=1))
    S.dim(dict(id="bpl", side="B", a=bb0["x1"], b=bb0["x2"], o1=bb0["y2"], o2=bb0["y2"], sym="b_pl", val=bb0["x2"] - bb0["x1"], lvl=2))
    _nom(S, "T", 0, _nom_profil(rec) + " sur platine t " + f0(pc.get("tp", 15)))
    _bloc_efforts(S, items, 0.0)
    return S.finish(s, [], "Tube sur platine : vue de face, la paroi du tube et le cordon de pourtour")


def c6_elevation(res, opt):
    pc = res["pieces"]
    rec = pc["profil"]
    t = float(rec.get("t", 0) or 0)
    tp = pc.get("tp", 15.0)
    jon = _jon(pc)
    w = rec["d"] if _est_rond(rec) else rec["b"]
    Lv = max(1.2 * w, 140.0)
    pl_h = max(y for _, y in pc["platine"]) - min(y for _, y in pc["platine"])
    bb0 = dict(x1=-tp, x2=Lv, y1=-pl_h / 2, y2=pl_h / 2)
    fs = _fs(bb0["x2"] - bb0["x1"] + 60)
    s = []
    P = [(-tp, -pl_h / 2), (0, -pl_h / 2), (0, pl_h / 2), (-tp, pl_h / 2)]
    s.append(_poly(["pp"], P)); s.append(_hach(["ht"], P))
    s.append(_poly(["ps"], _rect_rompu(0, -w / 2, Lv, w, {"R"}, 0.55 * fs)))
    if t > 0:
        s.append(_path(["hd"], [[(0, -w / 2 + t), (Lv, -w / 2 + t)], [(0, w / 2 - t), (Lv, w / 2 - t)]]))
    e = res["efforts"]
    if abs(e.N) > 0:
        _fleche_axe(s, (Lv, 0), (1, 0), e.N > 0, "N = " + _kn(e.N), fs)
    if abs(e.Vy) > 0:
        sg = -1 if e.Vy > 0 else 1
        x = Lv * 0.6
        _fleche(s, (x, -sg * (w / 2 + 4.8 * fs)), (x, -sg * (w / 2 + 0.3 * fs)), "V_z = " + _kn(e.Vy), fs, cote=+1)
    if abs(e.Mx) > 0:
        _arc_moment(s, (Lv * 0.35, 0), 1.7 * fs, 1 if e.Mx > 0 else -1, "M_y = " + _knm(e.Mx), fs, etiq=(Lv * 0.35, 1.7 * fs + 1.4 * fs + 0.34 * fs))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    jon1 = dict(jon, deux_faces=False)
    for sg in (-1, +1):          # la paroi d'épaisseur t, en haut et en bas, face extérieure seule
        _jonction_coupe(s, S, fs, (0, sg * (w / 2 - t / 2)), t, jon1, (1, 0), (0, sg), deux=False, symbole=(sg < 0), haut=(sg < 0))
    S.dim(dict(id="w", side="L", a=-w / 2, b=w / 2, o1=-tp, o2=-tp, sym="d" if _est_rond(rec) else "b", val=w, lvl=1))
    S.dim(dict(id="tp", side="T", a=-tp, b=0, o1=-pl_h / 2, o2=-w / 2, sym="t_p", val=tp, lvl=1, out="lo"))
    _nom(S, "T", Lv / 2, _nom_profil(rec))
    _nom(S, "B", -tp / 2, "Platine")
    return S.finish(s, [], "Tube sur platine : élévation, la paroi soudée sur la platine")


# ====================================================================== C7 bout à bout
def c7_joint(res, opt):
    pc = res["pieces"]
    b, t = pc["b"], pc["t"]
    X, Y = _proj_plan()
    L = max(1.2 * b, 120.0)
    bb0 = dict(x1=-L, x2=L, y1=-b / 2, y2=b / 2)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    s.append(_poly(["pp"], _rect_rompu(-L, -b / 2, L, b, {"L"}, 0.55 * fs)))
    s.append(_poly(["ps"], _rect_rompu(0, -b / 2, L, b, {"R"}, 0.55 * fs)))
    _bandes(s, res, X, Y)
    _trait_coupe(s, (-L * 0.45, -b / 2 - 1.2 * fs), (-L * 0.45, b / 2 + 1.2 * fs), fs)
    e = res["efforts"]
    if abs(e.Vx) > 0:
        _fleche_axe(s, (L, 0), (1, 0), e.Vx > 0, "N = " + _kn(e.Vx), fs)
        _fleche_axe(s, (-L, 0), (-1, 0), e.Vx > 0, "", fs)
    if abs(e.Vy) > 0:
        sg = -1 if e.Vy > 0 else 1
        _fleche(s, (L * 0.35, sg * (b / 2 + 0.3 * fs)), (L * 0.35, sg * (b / 2 + 4.5 * fs)), "V = " + _kn(e.Vy), fs)
    if abs(e.Mz) > 0:
        _arc_moment(s, (L * 0.5, 0), 1.6 * fs, 1 if e.Mz > 0 else -1, "M = " + _knm(e.Mz), fs, etiq=(L * 0.5, 1.6 * fs + 1.4 * fs + 0.34 * fs))
    _gouvernant(s, res, X, Y, fs, loin=(0.0, 0.0))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    jon = {"type": "pleine" if pc.get("pleine") else "partielle", "p": pc.get("p", t / 2), "renfort": 0.0, "a": t}
    _symbole(S, s, 0, -b / 2 * 0.55, fs, jon, vers=(-1, -1))
    S.dim(dict(id="b", side="L", a=-b / 2, b=b / 2, o1=-L, o2=-L, sym="b", val=b, lvl=1))
    _nom(S, "T", -L * 0.6, "Plat A · t " + f0(t))
    _nom(S, "T", L * 0.6, "Plat B · t " + f0(t))
    return S.finish(s, [], "Bout à bout : vue en plan des deux plats et de la soudure")


def c7_coupe(res, opt):
    pc = res["pieces"]
    t, p = pc["t"], pc.get("p", pc["t"] / 2)
    pleine = bool(pc.get("pleine"))
    deux = bool(pc.get("deux_faces", True))
    w = max(4.0 * t, 50.0)
    fs = max(0.9 * t, 6.0)                       # la coupe se lit à l'échelle de l'épaisseur
    bb0 = dict(x1=-w, x2=w, y1=-t / 2, y2=t / 2)
    s = []
    k = 0.6 * t
    if pleine:
        A = [(-w, -t / 2), (-k, -t / 2), (0, 0), (-k, t / 2), (-w, t / 2)]
        B = [(w, -t / 2), (k, -t / 2), (0, 0), (k, t / 2), (w, t / 2)]
        W = [(-k, -t / 2), (k, -t / 2), (0, 0), (k, t / 2), (-k, t / 2), (0, 0)]
        W2 = None
    else:
        kk = k * p / t * 2
        A = [(-w, -t / 2), (-kk, -t / 2), (0, -t / 2 + p), (0, t / 2 - (p if deux else 0)), (-kk if deux else 0, t / 2), (-w, t / 2)]
        B = [(w, -t / 2), (kk, -t / 2), (0, -t / 2 + p), (0, t / 2 - (p if deux else 0)), (kk if deux else 0, t / 2), (w, t / 2)]
        W = [(-kk, -t / 2), (kk, -t / 2), (0, -t / 2 + p)]
        W2 = [(-kk, t / 2), (kk, t / 2), (0, t / 2 - p)] if deux else None
    s.append(_poly(["pp"], A)); s.append(_hach(["ht"], A, 2.5))
    s.append(_poly(["ps"], B)); s.append(_hach(["ht"], B, 2.5))
    s.append(_poly(["wb"], W)); s.append(_hach(["ht", "htw"], W, 1.5))
    if W2:
        s.append(_poly(["wb"], W2)); s.append(_hach(["ht", "htw"], W2, 1.5))
    e = res["efforts"]
    if abs(e.Vx) > 0:
        _fleche_axe(s, (w, 0), (1, 0), e.Vx > 0, "N", fs, long=3.5)
        _fleche_axe(s, (-w, 0), (-1, 0), e.Vx > 0, "", fs, long=3.5)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="t", side="R", a=-t / 2, b=t / 2, o1=w, o2=w, sym="t", val=t, lvl=1, out="hi"))
    if not pleine:
        S.dim(dict(id="p", side="L", a=-t / 2, b=-t / 2 + p, o1=-k * p / t * 2, o2=0, sym="p", val=p, lvl=1, out="lo"))
    _nom(S, "B", 0, "Coupe A-A · " + ("pleine pénétration, chanfrein en X" if pleine else "pénétration partielle p" + (" des deux faces" if deux else " d'une face")))
    return S.finish(s, [], "Bout à bout : coupe de la soudure dans l'épaisseur")


# ====================================================================== C8 cordons libres
def c8_joint(res, opt):
    X, Y = _proj_plan()
    cs = res.get("cordons", []) + res.get("cordons_exclus", [])
    e = res["efforts"]
    xs = [X(c["p1"]) for c in cs] + [X(c["p2"]) for c in cs] + [X(e.reference)]
    ys = [Y(c["p1"]) for c in cs] + [Y(c["p2"]) for c in cs] + [Y(e.reference)]
    if not cs:
        xs, ys = [0.0, 100.0], [0.0, 60.0]
    m = max(40.0, 0.25 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    bb0 = dict(x1=min(xs) - m, x2=max(xs) + m, y1=min(ys) - m, y2=max(ys) + m)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s = []
    s.append(_rect(["bg"], bb0["x1"], bb0["y1"], bb0["x2"] - bb0["x1"], bb0["y2"] - bb0["y1"]))
    _bandes(s, res, X, Y)
    for segs in _chaines_ordonnees(res.get("cordons", [])):         # le nom de chaque chaîne, hors de la pièce (côté u_A)
        c = segs[len(segs) // 2]
        mx_, my_ = (X(c["p1"]) + X(c["p2"])) / 2, (Y(c["p1"]) + Y(c["p2"])) / 2
        nx, ny = _normale_vue(c, X, Y)
        txt = (c.get("chaine") or c["nom"]) + " · a " + f0(c["a"])
        d = c["a"] * S2 + 0.9 * fs
        s.append(_text(["txw"], mx_ + nx * d + nx * _tw(txt, 0.9 * fs) / 2, my_ + ny * d + ny * 0.5 * fs + 0.34 * fs, txt, 0.9 * fs))
    items = _efforts_joint(s, e, X, Y, fs, 4.0 * fs)
    _label(s, X(e.reference) - 1.3 * fs, Y(e.reference) + 1.6 * fs, "R", fs, cls=("dl",))
    _gouvernant(s, res, X, Y, fs)
    _marque_G(s, res, X, Y, fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    _nom(S, "T", (bb0["x1"] + bb0["x2"]) / 2, "Plan de joint · cordons libres")
    _bloc_efforts(S, items, (bb0["x1"] + bb0["x2"]) / 2)
    return S.finish(s, [], "Cordons libres : vue du plan de joint")


# ====================================================================== l'entrée
VUES = {"C1": (("joint", "Vue en plan", c1_joint), ("coupe", "Coupe A-A", c1_coupe)),
        "C2": (("coupe", "Coupe A-A", c2_coupe), ("joint", "Vue en plan", c2_joint)),
        "C3": (("elevation", "Élévation", c3_elevation), ("plan", "Coupe horizontale", c3_plan), ("face", "Vue de face", c3_face)),
        "C4": (("joint", "Vue en plan", c4_joint), ("coupe", "Coupe A-A", c4_coupe)),
        "C5": (("joint", "Vue de face", c5_face), ("elevation", "Élévation", c5_elevation)),
        "C6": (("joint", "Vue de face", c6_face), ("elevation", "Élévation", c6_elevation)),
        "C7": (("joint", "Vue en plan", c7_joint), ("coupe", "Coupe A-A", c7_coupe)),
        "C8": (("joint", "Vue du plan de joint", c8_joint),)}


def vues(res: Dict, opt: Optional[Options] = None) -> List[Tuple[str, str, Dessin]]:
    """Les vues d'un résultat : [(clé, titre, Dessin)] — la première est la vue principale.
    Un refus avant la construction (pas d'efforts ni de pièces) ne donne aucune vue."""
    if "efforts" not in res or "pieces" not in res:
        return []
    opt = opt or options_schemas()
    return [(cle, titre, fn(res, opt)) for cle, titre, fn in VUES[res["config"]]]


def svg(dessin: Dessin, identifiant: str = "sdr") -> str:
    return vers_svg(dessin, PALETTE, "100%", identifiant)

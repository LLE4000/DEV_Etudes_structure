# -*- coding: utf-8 -*-
"""Les SCHÉMAS du garde-corps, en DESSIN TECHNIQUE, sur le kit de dessin des assemblages
(feuille de cotation, primitives, hachures, lignes de rupture, rendu SVG —
``acier.assemblages….schemas``) et les briques des schémas de soudure (flèches d'effort,
étiquettes à halo, ⊙, repère de coupe, étendue réelle du corps). Le même peintre vectoriel
sert la note (``rapport.peindre``).

Six vues (``vues(res)`` : [(clé, titre, Dessin)]), scènes en mm À L'ÉCHELLE, y vers le bas :

- ÉLÉVATION (développée pour un L ou un U) : la dalle hachurée, les montants à leurs entraxes
  réels, la main courante, la lisse basse, le remplissage du type (barreaux à leur entraxe,
  centrés dans chaque travée ; lisses ; panneaux), les platines et leurs chevilles, les murs
  d'extrémité, les charges nommées (q_h ⊙, q_v ↓, Q_r ⊙), le montant gouvernant ◉ avec son
  taux, la chaîne des entraxes, H, h_bas, e_b, le repère de la coupe A-A ;
- COUPE A-A : le garde-corps vu de bout dans une travée — la dalle, la platine et ses deux
  rangées de chevilles, le montant vu de côté, la main courante et la lisse basse coupées
  (sections du catalogue hachurées), les charges q_h, q_v, Q_r, les cotes H, h_bas, b_p, e_x ;
  au mur : le mur, la platine, la console, la main courante, e ;
- PLAN : la ligne en plan (droit, L, U), la main courante, les montants orientés, les charges
  perpendiculaires à chaque tronçon, les longueurs, le repère A-A ;
- PLATINE (en plan) : la platine, la section du montant à l'échelle, la bande du cordon, les
  trous des chevilles avec leurs axes, e_x, e_y, b_p, l_p, les distances aux bords ;
- PIED (coupe sur la fixation, à grande échelle) : montant coupé, cordons a·√2, platine,
  chevilles dans le support, bras de levier m et z, efforts M, V, N du cas gouvernant ;
- SECTION de l'élément principal, axes y-y et z-z, direction de la charge.

Pour la main courante murale, les mêmes vues avec la console à la place du montant. Pour la
PLATINE LATÉRALE (nez de dalle) : l'élévation montre le montant qui descend le long de la dalle
jusqu'à sa platine, la coupe A-A la platine contre le nez de dalle et ses chevilles horizontales,
la vue de face de la platine (montant, cordons longitudinaux, trous, e_x, e_y, z, m) et la COUPE
HORIZONTALE sur la fixation (dalle, platine, section du montant, cordons coupés, chevilles).
Un garde-corps SANS MONTANT (main courante portante entre deux murs) a son élévation, sa coupe et
son plan sans montant ni fixation. Un garde-corps RAMPANT (escalier) : l'élévation est inclinée
de la pente, montants verticaux, entraxes mesurés suivant la pente.
Rien n'est recalculé : tout vient de ``etat.verifier``. Pureté : aucun Streamlit.
"""
import math
from typing import Dict, List, Optional, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # kit partagé
    Dessin, Feuille, Options, PALETTE, _circle, _group, _hachures, _path, _poly, _rect, _rect_pts, _rect_rompu, _text, vers_svg)
from acier.formats import F, f0
from acier.soudures.schemas import (_arc_moment, _bb, _bloc_efforts, _etendre, _fleche, _fs, _label, _nom, _normal,   # briques partagées
                                    _trait_coupe, _tw)

from . import bibliotheque as BI
from . import charges as CH

S2 = math.sqrt(2.0)
MIN_VIS = 0.22          # largeur visuelle minimale d'une pièce mince (fraction de la police) : un plat de 8 mm reste lisible
EP_DALLE = 180.0        # épaisseur dessinée de la dalle (mm) ; elle continue en dessous (ligne de rupture)
D_COUPE = 200.0         # distance (mm) du montant gouvernant au plan de la coupe A-A


def options_schemas(lvl: int = 2) -> Options:
    return Options(lvl=lvl, interactive=False, cartouche=False, realiste=True)


# ====================================================================== briques
def _hach(cls, pts, pas=None):
    return _path(cls, _hachures(pts, pas) if pas else _hachures(pts))


def _hachures_anneaux(rings, pas: float):
    """Hachures à 45° de la MATIÈRE entre plusieurs contours (règle pair-impair sur tous les
    anneaux) : la paroi d'un tube, et non son trou."""
    polys = [[(float(p[0]), float(p[1])) for p in r] for r in rings]
    cs = [x - y for poly in polys for x, y in poly]
    c0, c1 = min(cs), max(cs)
    d = pas * math.sqrt(2)
    lignes = []
    c = c0 + d / 2
    while c < c1:
        inter = []
        for poly in polys:
            n = len(poly)
            for i in range(n):
                (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
                f1 = x1 - y1 - c
                f2 = x2 - y2 - c
                if (f1 < 0) != (f2 < 0):
                    t = f1 / (f1 - f2)
                    inter.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        inter.sort(key=lambda p: p[0] + p[1])
        for j in range(0, len(inter) - 1, 2):
            a, b = inter[j], inter[j + 1]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1e-6:
                lignes.append([a, b])
        c += d
    return lignes


def _pas(x0: float, x1: float, pas: float) -> List[float]:
    out = []
    x = x0
    n = 0
    while x <= x1 + 1e-9 and n < 2000:
        out.append(x)
        x += pas
        n += 1
    return out


def _kn(v_N: float) -> str:
    return F(abs(v_N) / 1e3, 2) + " kN"


def _dim_sec(sec: Optional[Dict]) -> Tuple[float, float]:
    """(h, b) d'une section : h dans le sens de la charge horizontale, b dans le plan du
    garde-corps (un rond : d, d) ; (0, 0) sans section."""
    sec = sec or {}
    return float(sec.get("h") or sec.get("d") or 0.0), float(sec.get("b") or sec.get("d") or 0.0)


def _chaine(S, side: str, vals: List[float], o: float, lvl: int = 1):
    """Une CHAÎNE de cotes sur une seule rangée du côté ``side`` (T, B, L ou R) : lignes
    d'attache à chaque position, une ligne de cote continue, un repère à chaque position et la
    valeur de chaque intervalle centrée dedans (déportée hors de la rangée s'il est trop court)."""
    vals = sorted(float(v) for v in vals)
    if len(vals) < 2 or lvl > S.opt.lvl:
        return
    fs = S.fs
    t = 0.34 * fs
    H = side in ("T", "B")
    r = S.slot(side, vals[0], vals[-1])
    p = S.pos(side, r)
    e = p + (-0.45 if side in ("T", "L") else 0.45) * fs
    if H:
        S.ex["x1"] = min(S.ex["x1"], vals[0]); S.ex["x2"] = max(S.ex["x2"], vals[-1])
        ext = _path(["ext"], [[(v, o), (v, e)] for v in vals], hint="V")
        dln = _path(["dln"], [[(vals[0], p), (vals[-1], p)]] + [[(v - t, p + t), (v + t, p - t)] for v in vals], hint="H")
    else:
        S.ex["y1"] = min(S.ex["y1"], vals[0]); S.ex["y2"] = max(S.ex["y2"], vals[-1])
        ext = _path(["ext"], [[(o, v), (e, v)] for v in vals], hint="H")
        dln = _path(["dln"], [[(p, vals[0]), (p, vals[-1])]] + [[(p - t, v + t), (p + t, v - t)] for v in vals], hint="V")
    labs = []
    for v0, v1 in zip(vals[:-1], vals[1:]):
        txt = f0(v1 - v0)
        court = v1 - v0 < S.tw(txt) + 0.9 * fs
        c = (v0 + v1) / 2
        d = dict(id=f"ch{len(labs)}", sym="", lvl=lvl)
        if H:
            y = (p - 1.5 * fs if side == "T" else p + 1.3 * fs) if court else p - 0.42 * fs
            labs.append(S.label(c, y, False, txt, d))
        else:
            x = (p - 1.5 * fs if side == "L" else p + 1.3 * fs) if court else p - 0.42 * fs
            labs.append(S.label(x, c, True, txt, d))
    S.g.append(_group(["dm"], [ext, dln] + labs))


def _decale(pts: List[Tuple[float, float]], z: float) -> List[Tuple[float, float]]:
    """Le contour décalé de ``z`` vers l'EXTÉRIEUR (deux points par arête : le coin est coupé,
    pas arrondi — invisible sur un contour à congés)."""
    n = len(pts)
    aire = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)) / 2
    sg = 1.0 if aire > 0 else -1.0
    out = []
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-9:
            continue
        nx, ny = (y2 - y1) / L * sg, -(x2 - x1) / L * sg
        out += [(x1 + nx * z, y1 + ny * z), (x2 + nx * z, y2 + ny * z)]
    return out


def _coupe_section(s, sec: Dict, cx: float, cy: float, pas: float, h_vertical: bool = True, cls: str = "pp"):
    """La section d'une pièce COUPÉE, à l'échelle, centrée en (cx, cy) : contour du catalogue,
    trou d'un creux (blanc), hachures de la matière seule. ``h_vertical`` : h vertical sur le
    dessin (main courante, lisse) ; sinon h horizontal."""
    polys = []
    for r in BI.contours(sec):
        pts = []
        for p in r:
            x, y = float(p[0]), float(p[1])            # x selon b, y selon h, y vers le haut
            pts.append((cx + x, cy - y) if h_vertical else (cx + y, cy + x))
        polys.append(pts)
    s.append(_poly([cls], polys[0]))
    for inn in polys[1:]:
        s.append(_poly(["bo"], inn))
    s.append(_path(["ht"], _hachures_anneaux(polys, pas)))
    return polys


def _dalle(s, x1: float, x2: float, y: float, fs: float, ep: float = EP_DALLE, bords=("L", "R", "B")):
    """La dalle (ou le support) coupée : béton hachuré, rompu sur les bords ``bords`` (elle continue)."""
    pts = _rect_rompu(x1, y, x2 - x1, ep, set(bords), 0.5 * fs)
    s.append(_poly(["be"], pts)); s.append(_hach(["ht"], pts, 0.6 * fs))
    return pts


def _cheville(s, x: float, y_sol: float, d: float, prof: float, y_ecrou: float):
    """Une cheville vue de côté, verticale : la tige cachée dans le support (traits
    interrompus), l'écrou posé sur la platine."""
    s.append(_path(["hd"], [[(x - d / 2, y_sol), (x - d / 2, y_sol + prof)], [(x + d / 2, y_sol), (x + d / 2, y_sol + prof)],
                            [(x - d / 2, y_sol + prof), (x + d / 2, y_sol + prof)]]))
    s.append(_rect(["bo"], x - 0.8 * d, y_ecrou - 0.8 * d, 1.6 * d, 0.8 * d))


def _cheville_h(s, x_face: float, y: float, d: float, prof: float, tp: float):
    """Une cheville horizontale dans un mur (face en ``x_face``, mur à x < x_face) : tige cachée, écrou dehors."""
    s.append(_path(["hd"], [[(x_face, y - d / 2), (x_face - prof, y - d / 2)], [(x_face, y + d / 2), (x_face - prof, y + d / 2)],
                            [(x_face - prof, y - d / 2), (x_face - prof, y + d / 2)]]))
    s.append(_rect(["bo"], x_face + tp, y - 0.8 * d, 0.8 * d, 1.6 * d))


def _cordons(s, x0: float, y0: float, z: float, sens_x: int, sens_y: int, hach: float):
    """Un cordon d'angle coupé : triangle a·√2 au coin (x0, y0), ouvert vers ``sens_x``
    (le long de la platine) et ``sens_y`` (le long de la pièce)."""
    tri = [(x0, y0), (x0 + sens_x * z, y0), (x0, y0 + sens_y * z)]
    s.append(_poly(["wb"], tri)); s.append(_hach(["ht", "htw"], tri, hach))


def _gouvernant_nom(res: Dict) -> Optional[str]:
    return ((res.get("montant") or res.get("console")) or {}).get("nom")


def _montant_gouvernant(res: Dict) -> Optional[Dict]:
    """Le montant (ou la console) gouvernant ; None sans montant (main courante entre deux murs)."""
    ms = res["geometrie"].get("montants") or []
    if not ms:
        return None
    nom = (_gouvernant_nom(res) or "").replace("C", "M")
    return next((m for m in ms if m["nom"] == nom), ms[0])


def _x_coupe(res: Dict) -> float:
    """L'abscisse développée du plan de la coupe A-A : dans la travée à côté du montant gouvernant
    (sans montant : au tiers de la portée)."""
    m = _montant_gouvernant(res)
    if m is None:
        return 0.33 * float(res["geometrie"].get("L_total") or 1000.0)
    if m["s_d"] > 0:
        return m["x"] + min(D_COUPE, 0.3 * m["s_d"])
    return m["x"] - min(D_COUPE, 0.3 * m["s_g"])


def _incliner(prims: List[Dict], alpha_deg: float) -> List[Dict]:
    """Le garde-corps RAMPANT : la scène droite (x le long de la main courante) est inclinée de la
    pente — (x, y) → (x·cos α, y − x·sin α) : les montants restent verticaux, la main courante, la
    lisse basse et la dalle suivent la pente, les longueurs le long de la main courante sont gardées.
    Les rectangles deviennent des polygones."""
    ca, sa = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))

    def T(pt):
        x, y = float(pt[0]), float(pt[1])
        return (x * ca, y - x * sa)
    out = []
    for p in prims:
        t = p["t"]
        if t == "rect":
            q = dict(p); q["t"] = "poly"; q["pts"] = [T(pt) for pt in _rect_pts(p["x"], p["y"], p["w"], p["h"])]
            for k in ("x", "y", "w", "h"):
                q.pop(k, None)
            out.append(q)
        elif t == "poly":
            out.append({**p, "pts": [T(pt) for pt in p["pts"]]})
        elif t == "path":
            out.append({**p, "lignes": [[T(pt) for pt in l] for l in p["lignes"]]})
        elif t == "circle":
            x, y = T((p["cx"], p["cy"]))
            out.append({**p, "cx": x, "cy": y})
        elif t == "text" and p.get("x") is not None:
            x, y = T((p["x"], p["y"]))
            out.append({**p, "x": x, "y": y})
        elif t == "g" and "x" in p:
            x, y = T((p["x"], p["y"]))
            out.append({**p, "x": x, "y": y})
        else:
            out.append(p)
    return out


def _chaine_prims(s, xs: List[float], y: float, fs: float):
    """Une chaîne de cotes posée dans le CORPS (pour être inclinée avec lui) : ligne de cote à ``y``,
    repères, valeurs des intervalles (mesurées le long de la ligne)."""
    xs = sorted(float(v) for v in xs)
    if len(xs) < 2:
        return
    t = 0.34 * fs
    s.append(_path(["ext"], [[(x, y - 1.0 * fs), (x, y + 0.45 * fs)] for x in xs], hint="V"))
    s.append(_path(["dln"], [[(xs[0], y), (xs[-1], y)]] + [[(x - t, y + t), (x + t, y - t)] for x in xs], hint="H"))
    for x0, x1 in zip(xs[:-1], xs[1:]):
        s.append(_text(["tx2"], (x0 + x1) / 2, y - 0.42 * fs, f0(x1 - x0), fs))


def _efforts_fixation(res: Dict) -> Tuple[str, float, float, float]:
    """(libellé du cas, V, N (> 0 compression), M) à la fixation, en N et N·mm, pour le cas
    gouvernant : ceux de la platine si elle est vérifiée, sinon ceux du montant (ou de la
    console) sous le cas gouvernant de la soudure."""
    fx = res.get("fixation") or {}
    pl = fx.get("platine") or {}
    if pl.get("taux_max") is not None:
        return CH.CAS_LIBELLES.get(pl.get("cas_charge"), ""), float(pl.get("V") or 0.0), float(pl.get("N") or 0.0), float(pl.get("M") or 0.0)
    if fx.get("type") == "laterale" and fx.get("soudure") and res.get("montant"):
        v = res["montant"]["verif"]
        g = float((res.get("limites") or {}).get("gamma") or 1.0)
        cas = fx["soudure"].get("cas_charge") or "1"
        y_P = v["H"] + float(((res.get("fixation") or {}).get("platine") or {}).get("bp") or 150.0) / 2
        P1, Nv = v.get("P_h", 0.0), v.get("N_v", 0.0)
        if cas == "2":
            return CH.CAS_LIBELLES.get(cas, ""), Nv * g, 0.0, 0.0
        return CH.CAS_LIBELLES.get(cas, ""), (Nv if cas != "1" else 0.0) * g, -P1 * g, P1 * y_P * g
    g = float((res.get("limites") or {}).get("gamma") or 1.0)
    cas = ((fx.get("soudure") or {}).get("cas_charge")) or "1"
    if res["geometrie"].get("systeme") == "mur" and res.get("console"):
        v = res["console"]["verif"]
        Rv, Rh, Rh2, e = v["Rv"], v["Rh"], v["Rh2"], v["e"]
        if cas == "1":
            V, N, M = Rh2, -Rh, Rh2 * e
        elif cas == "2":
            V, N, M = Rv, 0.0, Rv * e
        else:
            V, N, M = math.hypot(Rv, Rh2), -Rh, math.hypot(Rv * e, Rh2 * e)
    else:
        v = (res.get("montant") or {}).get("verif") or {}
        P1, P2, Nv, H = v.get("P_h", 0.0), v.get("P_h2", 0.0), v.get("N_v", 0.0), v.get("H", 0.0)
        if cas == "2":
            V, N, M = 0.0, Nv, 0.0
        elif cas == "1":
            V, N, M = math.hypot(P1, P2), 0.0, math.hypot(P1, P2) * H
        else:
            V, N, M = math.hypot(P1, P2), Nv, math.hypot(P1, P2) * H
    return CH.CAS_LIBELLES.get(cas, ""), V * g, N * g, M * g


def _texte_efforts(cas: str, V: float, N: float, M: float) -> str:
    return ((cas + " : ") if cas else "") + "V = " + _kn(V) + " · N = " + _kn(N) + (" (compression)" if N > 0 else " (traction)" if N < 0 else "") \
        + " · M = " + F(M / 1e6, 3) + " kN·m"


# ====================================================================== élévation
def _mur_bout(s, x, y1, y2, fs, cote):
    """Un mur d'extrémité : bande hachurée de 2,5·fs de large, du côté ``cote`` (−1 à gauche)."""
    w = 2.5 * fs
    pts = [(x, y1), (x + cote * w, y1), (x + cote * w, y2), (x, y2)]
    s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.7 * fs))


def _xs_barreaux(travees: List[Dict], eb: float, w_post: float, fs: float) -> List[float]:
    """Les barreaux de chaque travée : n = ⌊(L − montant) / e_b⌋, centrés dans la travée."""
    xs = []
    for t in travees:
        n = int((t["L"] - w_post - 0.2 * fs) // eb) if eb > 0 else 0
        if n >= 1:
            xs += [t["x0"] + (t["L"] - (n - 1) * eb) / 2 + k * eb for k in range(n)]
    return xs


def elevation(res: Dict, opt: Options) -> Dessin:
    geo = res["geometrie"]
    H = float(geo["H"])
    systeme = geo.get("systeme", "montants")
    Lt = geo["L_total"]
    montants, travees = geo["montants"], geo["travees"]
    sec_p = res.get("section_montant") or {}
    sec_l = (res.get("lisse") or {}).get("section") or {}
    rp = res.get("remplissage") or {}
    fx = res.get("fixation") or {}
    pl = fx.get("platine") or {}
    typ_fix = fx.get("type", "platine")
    cons = res.get("console") or {}
    sec_c = cons.get("section") or {}
    fs = opt.fs_force or _fs(Lt)
    hp, b_post = _dim_sec(sec_p)
    h_mc, b_mc = _dim_sec(sec_l)
    h_mc = max(h_mc, 0.25 * fs)
    h_c, b_c = _dim_sec(sec_c)
    w_post = max(b_post if systeme == "montants" else b_c, MIN_VIS * fs)
    lat = typ_fix == "laterale" and systeme == "montants"           # platine latérale : H se mesure du dessus de dalle, le montant descend le long du nez
    d_h = float(geo.get("d_h") or 0.0) if lat else 0.0
    tp = float(pl.get("tp") or 10.0) if typ_fix in ("platine", "soudee") else 0.0
    lp, bp = float(pl.get("lp") or 150.0), float(pl.get("bp") or 150.0)
    ex, ey = float(pl.get("ex") or 100.0), float(pl.get("ey") or 100.0)
    n_r, d_a = int(pl.get("n_tendus") or 2), float(pl.get("d") or 12.0)
    pente = float(geo.get("pente") or 0.0)
    y_sol = 0.0                                    # le dessus de la dalle
    y_pl = -tp if systeme == "montants" else 0.0   # le dessus de la platine : H se mesure de là
    y_mc = y_pl - H                                # l'axe de la main courante
    ep = max(EP_DALLE, 1.4 * fs)
    if lat:
        ep = max(ep, d_h + bp + 1.0 * fs)
    m_d = 2.2 * fs
    bb0 = dict(x1=0.0, x2=Lt, y1=y_mc - h_mc / 2, y2=y_sol + ep)
    s: List[Dict] = []
    if systeme == "mur":                           # la face du mur, en fond
        s.append(_rect(["bg"], -m_d, y_mc - 3.5 * fs, Lt + 2 * m_d, y_sol - (y_mc - 3.5 * fs)))
    _dalle(s, -m_d, Lt + m_d, y_sol, fs, ep)
    # la lisse basse (axe à h_bas au-dessus de la platine) et le remplissage du type
    typ = rp.get("type")
    y_lb = h_lb = None
    h_bas = float(rp.get("h_bas") or 100.0)
    if typ in ("barreaux", "panneau") and rp.get("section_basse"):
        h_lb = max(_dim_sec(rp["section_basse"])[0], 0.2 * fs)
        y_lb = y_pl - h_bas
        s.append(_rect(["ps"], 0.0, y_lb - h_lb / 2, Lt, h_lb))
    y_haut = y_mc + h_mc / 2                                   # le dessous de la main courante
    y_bas = (y_lb - h_lb / 2) if y_lb is not None else y_pl - 0.4 * fs
    xs_b: List[float] = []
    eb = el = 0.0
    if typ == "barreaux" and rp.get("verif"):
        eb = float(rp["entraxe"])
        wb = max(_dim_sec(rp.get("section"))[1], MIN_VIS * fs)
        xs_b = _xs_barreaux(travees, eb, w_post, fs)
        pas = max(1, int(math.ceil(len(xs_b) / 160.0)))
        for x in xs_b[::pas]:
            s.append(_rect(["ps"], x - wb / 2, y_haut, wb, y_bas - y_haut))
    elif typ == "lisses" and rp.get("verif"):
        el, nl = float(rp["entraxe"]), int(rp.get("n") or 1)
        hl = max(_dim_sec(rp.get("section"))[0], 0.2 * fs)
        for k in range(1, nl + 1):
            y = y_mc + k * el
            if y < y_pl - 0.6 * fs:
                s.append(_rect(["ps"], 0.0, y - hl / 2, Lt, hl))
    elif typ == "panneau":
        y0, y1 = y_haut + 0.15 * fs, y_bas - 0.15 * fs
        for t in travees:
            x0, x1 = t["x0"] + w_post / 2 + 0.2 * fs, t["x0"] + t["L"] - w_post / 2 - 0.2 * fs
            if x1 - x0 < fs or y1 - y0 < fs:
                continue
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            s.append(_poly(["bg"], pts)); s.append(_path(["fl2"], [pts + [pts[0]]]))
            d = min(0.9 * fs, (x1 - x0) / 4, (y1 - y0) / 4)              # le symbole du verre : deux traits obliques
            s.append(_path(["hd"], [[(x0 + 0.3 * d, y0 + 1.6 * d), (x0 + 1.6 * d, y0 + 0.3 * d)], [(x0 + 0.8 * d, y0 + 2.1 * d), (x0 + 2.1 * d, y0 + 0.8 * d)]]))
    # les montants et leurs platines (ou, au mur, les platines et les consoles vues de bout)
    gouv = _gouvernant_nom(res)
    ancr_apres = []
    for m in montants:
        x = m["x"]
        if systeme == "montants" and lat:                              # la platine sur le nez de dalle, le montant qui descend jusqu'à son bas, les chevilles
            s.append(_rect(["ps"], x - lp / 2, y_sol + d_h, lp, bp))
            s.append(_rect(["pp"], x - w_post / 2, y_haut, w_post, y_sol + d_h + bp - y_haut))
            for i in (-1, 1):
                for k in range(n_r):
                    ancr_apres.append(_circle(["bo"], x + (k - (n_r - 1) / 2) * ey, y_sol + d_h + bp / 2 + i * ex / 2, d_a / 2))
        elif systeme == "montants":
            s.append(_rect(["pp"], x - w_post / 2, y_haut, w_post, y_pl - y_haut))
            if typ_fix == "platine":
                s.append(_rect(["pp"], x - lp / 2, y_pl, lp, tp))
                for k in range(n_r):
                    _cheville(s, x + (k - (n_r - 1) / 2) * ey, y_sol, d_a, min(8 * d_a, 0.8 * ep), y_pl)
            elif typ_fix == "soudee":
                s.append(_rect(["ps"], x - lp / 2, y_pl, lp, tp))
            else:                                                  # scellé : le montant entre dans la dalle, scellement au mortier
                prof = 0.6 * ep
                s.append(_rect(["gr"], x - w_post / 2 - 0.3 * fs, y_sol, w_post + 0.6 * fs, prof))
                s.append(_rect(["pp"], x - w_post / 2, y_sol, w_post, prof))
        else:
            y_c = y_haut + h_c / 2                                  # l'axe de la console, sous la main courante
            if typ_fix in ("platine", "soudee"):
                s.append(_rect(["pp" if typ_fix == "platine" else "ps"], x - lp / 2, y_c - bp / 2, lp, bp))
            s.append(_rect(["pp"], x - w_post / 2, y_haut, w_post, h_c))
            if typ_fix == "platine":
                for i in (-1, 1):
                    for k in range(n_r):
                        xa, ya = x + (k - (n_r - 1) / 2) * ey, y_c + i * ex / 2
                        cache = abs(ya - y_mc) < h_mc / 2 + d_a / 2
                        ancr_apres.append(_circle(["hd" if cache else "bo"], xa, ya, d_a / 2))
        if m["coin"]:
            s.append(_path(["ax"], [[(x, y_mc - h_mc / 2 - 2.6 * fs), (x, y_sol + 0.6 * fs)]]))
    # la main courante sur toute la longueur, puis ce qui se voit devant ou à travers elle
    s.append(_rect(["ps"], 0.0, y_mc - h_mc / 2, Lt, h_mc))
    s.extend(ancr_apres)
    for m in montants:
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if nom == gouv:
            yg = y_pl - 0.7 * fs if systeme == "montants" else y_mc
            s.append(_circle(["gv"], m["x"], yg, 0.5 * fs)); s.append(_circle(["gvd"], m["x"], yg, 0.2 * fs))
    for r in geo.get("reprises_mur", []):
        _mur_bout(s, 0.0 if r["ou"] == "début" else Lt, y_mc - h_mc / 2 - 1.5 * fs, y_sol, fs, -1 if r["ou"] == "début" else +1)
    # les charges nommées : q_h ⊙ (vers soi) et q_v ↓ sur la main courante, Q_r ⊙ sur le remplissage
    ch = res["charges"]
    items = []
    n_q = max(3, min(8, int(round(Lt / 700.0))))
    xs_p = [m["x"] for m in montants]
    xs_q = [Lt * (i + 0.5) / n_q for i in range(n_q)]
    xs_q = [x + (0.9 * fs if any(abs(x - xp) < 0.8 * fs for xp in xs_p) else 0.0) for x in xs_q]
    if ch["mode"] == "P":
        for i, m in enumerate(montants):
            _normal(s, (m["x"], y_mc), True, "P" if i == 0 else "", fs)
        items.append(f"P = {F(ch['P'], 2)} kN par montant ⊙ (vers soi)")
    else:
        for i, x in enumerate(xs_q):
            _normal(s, (x, y_mc), True, "q_h" if i == 0 else "", fs)
        items.append(f"q_h = {F(ch['qh'], 2)} kN/m ⊙ (vers soi)")
    if ch["qv"] > 0:
        for i, x in enumerate(xs_q):
            _fleche(s, (x, y_mc - h_mc / 2 - 3.2 * fs), (x, y_mc - h_mc / 2 - 0.2 * fs), "q_v" if i == 0 else "", fs, cote=+1)
        items.append(f"q_v = {F(ch['qv'], 2)} kN/m ↓")
    if ch.get("Q", 0) > 0:
        items.append(f"Q = {F(ch['Q'], 2)} kN sur la main courante")
    if typ in ("barreaux", "lisses", "panneau") and y_bas - y_haut > 2 * fs:
        mg = _montant_gouvernant(res) or {"x": -1.0}
        t = next((t for t in travees if abs(t["x0"] - mg["x"]) < 1e-6), None) or next((t for t in travees if abs(t["x0"] + t["L"] - mg["x"]) < 1e-6), travees[0])
        xq, yq = t["x0"] + t["L"] / 2, (y_haut + y_bas) / 2
        _normal(s, (xq, yq), True, "Q_r", fs, etiq=(xq + 0.55 * fs + _tw("Q_r", fs) / 2 + 0.7 * fs, yq + 0.34 * fs))
        items.append(f"Q_r = {F(ch['Qr'], 2)} kN ⊙")
        items.append(f"q_r = {F(ch['qr'], 2)} kN/m²")
    # le repère de la coupe A-A, dans la travée à côté du montant gouvernant
    xc = _x_coupe(res)
    if any(abs(xc - xq) < 1.2 * fs for xq in xs_q):
        xc += 1.2 * fs
    _trait_coupe(s, (xc, y_mc - h_mc / 2 - 1.4 * fs), (xc, y_sol + ep + 0.4 * fs), fs, "A")
    # le garde-corps rampant : la chaîne des entraxes dans le corps (le long de la pente), puis toute la scène inclinée
    ca = math.cos(math.radians(pente)) if pente else 1.0
    if pente:
        _chaine_prims(s, [0.0] + [t["x0"] + t["L"] for t in travees], y_sol + ep + 1.6 * fs, fs)
        s = _incliner(s, pente)
        bb0 = dict(x1=0.0, x2=Lt * ca, y1=y_mc - h_mc / 2 - Lt * math.sin(math.radians(pente)), y2=y_sol + ep + 2.6 * fs)

    def X(x):
        return x * ca
    # la feuille : chaîne des entraxes sur une rangée, L, H, h_bas, e_b ou e_l, noms, gouvernant, charges
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    if not pente:
        _chaine(S, "B", [0.0] + [t["x0"] + t["L"] for t in travees], y_sol + ep)
        if len(travees) > 1:
            S.dim(dict(id="L", side="B", a=0.0, b=Lt, o1=y_sol + ep, o2=y_sol + ep, sym="L", val=Lt, lvl=1))
    S.dim(dict(id="H", side="L", a=y_mc, b=y_pl, o1=0.0, o2=0.0, sym="H", val=H, lvl=1))
    if lat and d_h > 0:
        S.dim(dict(id="dh", side="L", a=y_sol, b=y_sol + d_h, o1=0.0, o2=0.0, sym="d_h", val=d_h, lvl=2))
    if y_lb is not None:
        S.dim(dict(id="hb", side="L", a=y_lb, b=y_pl, o1=0.0, o2=0.0, sym="h_bas", val=h_bas, lvl=2))
    if typ == "barreaux" and len(xs_b) >= 2:
        S.dim(dict(id="eb", side="T", a=X(xs_b[0]), b=X(xs_b[1]), o1=y_haut, o2=y_haut, sym="e_b", val=eb, lvl=2, out="lo"))
    elif typ == "lisses" and el > 0:
        S.dim(dict(id="el", side="R", a=y_mc, b=y_mc + el, o1=X(Lt), o2=X(Lt), sym="e_l", val=el, lvl=2))
    t_g = ((res.get("montant") or res.get("console")) or {}).get("verif", {}).get("taux_max")
    for m in montants:
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if nom == gouv and t_g is not None:
            _nom(S, "T", X(m["x"]), f"◉ {nom} · {t_g * 100:.0f} %", cls="fzl")
        else:
            S.name("T", X(m["x"]), nom)
    for i, ent in enumerate(geo["entraxes"]):
        if i and ent:
            _nom(S, "T", X(sum(sum(e) for e in geo["entraxes"][:i])), "angle")
    _bloc_efforts(S, items, X(Lt) / 2, "B", 4)
    _nom(S, "B", X(Lt) / 2, CH.phrase_cumul(ch) + (f" · rampant, pente {F(pente, 0)}°, entraxes suivant la pente" if pente else ""), cls="tx2")
    _etendre(S, s, fs)
    n_tr = sum(1 for e in geo["entraxes"] if e)
    _nom(S, "T", X(Lt) / 2, ("Élévation développée" if n_tr > 1 else "Élévation") + (" · rampant" if pente else "") + (" · sans montant" if not montants else ""))
    return S.finish(s, [], "Élévation du garde-corps : dalle, montants, main courante, lisse basse, remplissage, platines, charges, entraxes")


# ====================================================================== coupe A-A
def coupe(res: Dict, opt: Options) -> Dessin:
    """Le garde-corps vu de bout, dans la travée à côté du montant gouvernant : la dalle, la
    platine et ses chevilles, le montant vu de côté, la main courante et la lisse basse coupées."""
    geo = res["geometrie"]
    if geo.get("systeme") == "mur":
        return _coupe_mur(res, opt)
    H = float(geo["H"])
    sans_montant = not geo.get("montants")
    sec_p, sec_l = res["section_montant"], res["lisse"]["section"]
    rp = res.get("remplissage") or {}
    fx = res.get("fixation") or {}
    pl = fx.get("platine") or {}
    typ_fix = fx.get("type", "platine") if not sans_montant else "aucune"
    lat = typ_fix == "laterale"
    ch = res["charges"]
    hp, _b = _dim_sec(sec_p) if not sans_montant else (0.0, 0.0)
    h_mc, b_mc = _dim_sec(sec_l)
    tp = float(pl.get("tp") or 10.0) if typ_fix in ("platine", "soudee", "laterale") else 0.0
    bp, ex, d_a = float(pl.get("bp") or 150.0), float(pl.get("ex") or 100.0), float(pl.get("d") or 12.0)
    d_h = float(geo.get("d_h") or 0.0) if lat else 0.0
    ep = EP_DALLE if not lat else max(EP_DALLE, d_h + bp + 40.0)
    fs = opt.fs_force or _fs(max(0.62 * (H + ep + 60.0), bp + 300.0))
    y_sol, y_pl = 0.0, (-tp if typ_fix in ("platine", "soudee") else 0.0)
    y_mc = y_pl - H
    y_haut = y_mc + h_mc / 2
    s: List[Dict] = []
    x_d = max(bp, hp, b_mc + 20.0) / 2 + 2.4 * fs
    bb0 = dict(x1=-x_d, x2=x_d, y1=y_mc - h_mc / 2, y2=y_sol + ep)
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    a = float(soud.get("a") or 4.0)
    jon = soud.get("jonction", "angle")
    if lat:
        # la dalle à gauche, son nez en x_face ; la platine contre le nez, le montant devant elle, les chevilles horizontales
        x_face = -hp / 2 - tp
        sup = _rect_rompu(-x_d - 0.8 * fs, y_sol, x_face + x_d + 0.8 * fs, ep, {"L", "B"}, 0.5 * fs)
        s.append(_poly(["be"], sup)); s.append(_hach(["ht"], sup, 0.6 * fs))
        s.append(_rect(["pp"], x_face, y_sol + d_h, tp, bp))
        for sg in (-1, 1):
            _cheville_h(s, x_face, y_sol + d_h + bp / 2 + sg * ex / 2, d_a, min(8 * d_a, 0.8 * x_d), tp)
        s.append(_rect(["pp"], -hp / 2, y_haut, hp, y_sol + d_h + bp - y_haut))                   # le montant descend le long du nez de dalle
        z = a * S2 if jon == "angle" else 0.6 * float(sec_p.get("t") or sec_p.get("b") or 4.0)
        s.append(_rect(["wb"], x_face + tp - z, y_sol + d_h, z, bp))                                # les cordons longitudinaux, vus de côté
    else:
        _dalle(s, -x_d - 0.8 * fs, x_d + 0.8 * fs, y_sol, fs, ep)
    if typ_fix == "platine":
        s.append(_rect(["pp"], -bp / 2, y_pl, bp, tp))
        for sg in (-1, 1):
            _cheville(s, sg * ex / 2, y_sol, d_a, min(8 * d_a, 0.8 * ep), y_pl)
    elif typ_fix == "soudee":
        s.append(_rect(["ps"], -bp / 2, y_pl, bp, tp))
    elif typ_fix == "scellee":
        prof = 0.6 * ep
        s.append(_rect(["gr"], -hp / 2 - 0.3 * fs, y_sol, hp + 0.6 * fs, prof))
        s.append(_rect(["pp"], -hp / 2, y_sol, hp, prof))
    if not sans_montant and not lat:
        s.append(_rect(["pp"], -hp / 2, y_haut, hp, y_pl - y_haut))              # le montant vu de côté (profondeur h_p)
    if typ_fix in ("platine", "soudee"):
        z = a * S2 if jon == "angle" else 0.6 * float(sec_p.get("t") or sec_p.get("b") or 4.0)
        for sg in (-1, 1):
            s.append(_poly(["wb"], [(sg * hp / 2, y_pl), (sg * hp / 2 + sg * z, y_pl), (sg * hp / 2, y_pl - z)]))
    # la main courante, la lisse basse et les lisses intermédiaires coupées
    pas_h = max(2.5, 0.12 * fs)
    _coupe_section(s, sec_l, 0.0, y_mc, pas_h)
    typ = rp.get("type")
    y_lb = None
    h_bas = float(rp.get("h_bas") or 100.0)
    if typ in ("barreaux", "panneau") and rp.get("section_basse"):
        y_lb = y_pl - h_bas
        _coupe_section(s, rp["section_basse"], 0.0, y_lb, pas_h)
    elif typ == "lisses" and rp.get("verif"):
        el, nl = float(rp["entraxe"]), int(rp.get("n") or 1)
        for k in range(1, nl + 1):
            y = y_mc + k * el
            if y < y_pl - 0.6 * fs:
                _coupe_section(s, rp["section"], 0.0, y, pas_h)
    # les charges : q_h → (de l'intérieur vers le vide) à l'axe de la main courante, q_v ↓ dessus, Q_r → sur le remplissage
    items = []
    x_l = -b_mc / 2 - 0.3 * fs
    if ch["mode"] == "P":
        _fleche(s, (x_l - 4.3 * fs, y_mc), (x_l, y_mc), "P", fs, cote=+1)
        items.append(f"P = {F(ch['P'], 2)} kN par montant")
    else:
        _fleche(s, (x_l - 4.3 * fs, y_mc), (x_l, y_mc), "q_h", fs, cote=+1)
        items.append(f"q_h = {F(ch['qh'], 2)} kN/m")
    if ch["qv"] > 0:
        _fleche(s, (0.0, y_mc - h_mc / 2 - 4.3 * fs), (0.0, y_mc - h_mc / 2 - 0.3 * fs), "q_v", fs, cote=+1)
        items.append(f"q_v = {F(ch['qv'], 2)} kN/m")
    if typ in ("barreaux", "lisses", "panneau"):
        y_r = (y_haut + (y_lb if y_lb is not None else y_pl)) / 2
        _fleche(s, (-hp / 2 - 4.6 * fs, y_r), (-hp / 2 - 0.3 * fs, y_r), "Q_r", fs, cote=+1)
        items.append(f"Q_r = {F(ch['Qr'], 2)} kN")
        items.append(f"q_r = {F(ch['qr'], 2)} kN/m²")
    # les noms des pièces, du côté du vide
    s.append(_text(["tx"], b_mc / 2 + 0.5 * fs, y_mc + 0.34 * fs, sec_l["nom"] + " · main courante", fs, "start"))
    if not sans_montant:
        s.append(_text(["tx"], hp / 2 + 0.5 * fs, y_pl - 0.28 * H + 0.34 * fs, sec_p["nom"] + " · montant", fs, "start"))
    if y_lb is not None:
        s.append(_text(["tx"], _dim_sec(rp["section_basse"])[1] / 2 + 0.5 * fs, y_lb + 0.34 * fs, rp["section_basse"]["nom"] + " · lisse basse", fs, "start"))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="H", side="L", a=y_mc, b=y_pl, o1=x_l - 4.6 * fs, o2=-bp / 2, sym="H", val=H, lvl=1))
    if y_lb is not None:
        S.dim(dict(id="hb", side="L", a=y_lb, b=y_pl, o1=-_dim_sec(rp["section_basse"])[1] / 2, o2=-bp / 2, sym="h_bas", val=h_bas, lvl=2))
    if typ_fix in ("platine", "soudee"):
        S.dim(dict(id="bp", side="B", a=-bp / 2, b=bp / 2, o1=y_sol + ep, o2=y_sol + ep, sym="b_p", val=bp, lvl=1))
    if typ_fix == "platine":
        S.dim(dict(id="ex", side="B", a=-ex / 2, b=ex / 2, o1=y_sol + ep, o2=y_sol + ep, sym="e_x", val=ex, lvl=2))
    if lat:
        S.dim(dict(id="dh", side="R", a=y_sol, b=y_sol + d_h, o1=hp / 2, o2=hp / 2, sym="d_h", val=d_h, lvl=1))
        S.dim(dict(id="bp", side="R", a=y_sol + d_h, b=y_sol + d_h + bp, o1=hp / 2, o2=hp / 2, sym="b_p", val=bp, lvl=1))
        S.dim(dict(id="ex", side="R", a=y_sol + d_h + bp / 2 - ex / 2, b=y_sol + d_h + bp / 2 + ex / 2, o1=hp / 2, o2=hp / 2, sym="e_x", val=ex, lvl=2))
        S.dim(dict(id="tp", side="B", a=-hp / 2 - tp, b=-hp / 2, o1=y_sol + d_h + bp, o2=y_sol + d_h + bp, sym="t_p", val=tp, lvl=2, out="hi"))
    if not sans_montant:
        S.tag(hp / 2 + 0.4 * fs, y_pl - 0.5 * H, dict(id="hp", sym="h_p", val=hp, lvl=2))
    _bloc_efforts(S, items, 0.0, "B", 2)
    _etendre(S, s, fs)
    _nom(S, "T", 0.0, "Coupe A-A" + (" · platine latérale" if lat else "") + (" · sans montant" if sans_montant else ""))
    return S.finish(s, [], "Coupe A-A : dalle, platine et chevilles, montant vu de côté, main courante et lisse basse coupées, charges")


def _coupe_mur(res: Dict, opt: Options) -> Dessin:
    """Au mur : le mur coupé, la platine et ses chevilles, la console vue de côté, la main
    courante coupée à e du mur, les charges q_v et q_h (qui tire la console hors du mur)."""
    cons = res["console"]
    sec_c, sec_l = cons["section"], res["lisse"]["section"]
    e = float(cons["e"])
    fx = res.get("fixation") or {}
    pl = fx.get("platine") or {}
    typ_fix = fx.get("type", "platine")
    ch = res["charges"]
    h_c, _b = _dim_sec(sec_c)
    h_mc, b_mc = _dim_sec(sec_l)
    tp = float(pl.get("tp") or 10.0) if typ_fix in ("platine", "soudee") else 0.0
    bp, ex, d_a = float(pl.get("bp") or 150.0), float(pl.get("ex") or 100.0), float(pl.get("d") or 12.0)
    fs = opt.fs_force or _fs(e + 340.0)
    t_mur = max(3.0 * fs, 120.0)
    y_mc = 0.0
    y_haut = h_mc / 2
    y_c = y_haut + h_c / 2                                  # l'axe de la console, sous la main courante qu'elle porte
    s: List[Dict] = []
    y1, y2 = y_mc - h_mc / 2 - 5.0 * fs, y_c + bp / 2 + 3.0 * fs
    bb0 = dict(x1=-t_mur, x2=e + b_mc / 2, y1=y1, y2=y2)
    mur = _rect_rompu(-t_mur, y1, t_mur, y2 - y1, {"T", "B"}, 0.5 * fs)
    s.append(_poly(["be"], mur)); s.append(_hach(["ht"], mur, 0.6 * fs))
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    a = float(soud.get("a") or 4.0)
    jon = soud.get("jonction", "angle")
    if typ_fix == "platine":
        s.append(_rect(["pp"], 0.0, y_c - bp / 2, tp, bp))
        for sg in (-1, 1):
            _cheville_h(s, 0.0, y_c + sg * ex / 2, d_a, min(8 * d_a, 0.8 * t_mur), tp)
    elif typ_fix == "soudee":
        s.append(_rect(["ps"], 0.0, y_c - bp / 2, tp, bp))
    x0 = tp
    s.append(_rect(["pp"], x0, y_c - h_c / 2, e + b_mc / 2 - x0, h_c))        # la console vue de côté
    if typ_fix != "scellee":
        z = a * S2 if jon == "angle" else 0.6 * float(sec_c.get("t") or sec_c.get("b") or 4.0)
        for sg in (-1, 1):
            s.append(_poly(["wb"], [(x0, y_c + sg * h_c / 2), (x0 + z, y_c + sg * h_c / 2), (x0, y_c + sg * h_c / 2 + sg * z)]))
    _coupe_section(s, sec_l, e, y_mc, max(2.5, 0.12 * fs))
    items = []
    _fleche(s, (e, y_mc - h_mc / 2 - 4.3 * fs), (e, y_mc - h_mc / 2 - 0.3 * fs), "q_v", fs, cote=+1)
    items.append(f"q_v = {F(ch['qv'], 2)} kN/m")
    if ch["mode"] == "P":
        _fleche(s, (e + b_mc / 2 + 0.3 * fs, y_mc), (e + b_mc / 2 + 4.3 * fs, y_mc), "P", fs, cote=+1)
        items.append(f"P = {F(ch['P'], 2)} kN par console (tire la console hors du mur)")
    else:
        _fleche(s, (e + b_mc / 2 + 0.3 * fs, y_mc), (e + b_mc / 2 + 4.3 * fs, y_mc), "q_h", fs, cote=+1)
        items.append(f"q_h = {F(ch['qh'], 2)} kN/m (tire la console hors du mur)")
    s.append(_text(["tx"], x0 + 0.4 * fs, y_c + h_c / 2 + 1.15 * fs, sec_c["nom"] + " · console", fs, "start"))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    _nom(S, "T", e, sec_l["nom"] + " · main courante")
    S.dim(dict(id="e", side="B", a=0.0, b=e, o1=y_c + bp / 2 if typ_fix != "scellee" else y_c + h_c / 2, o2=y_c + h_c / 2, sym="e", val=e, lvl=1))
    if typ_fix in ("platine", "soudee"):
        S.dim(dict(id="bp", side="L", a=y_c - bp / 2, b=y_c + bp / 2, o1=0.0, o2=0.0, sym="b_p", val=bp, lvl=1))
        S.dim(dict(id="tp", side="T", a=0.0, b=tp, o1=y_c - bp / 2, o2=y_c - bp / 2, sym="t_p", val=tp, lvl=2, out="hi"))
    if typ_fix == "platine":
        S.dim(dict(id="ex", side="L", a=y_c - ex / 2, b=y_c + ex / 2, o1=0.0, o2=0.0, sym="e_x", val=ex, lvl=2))
    S.tag(x0 + 0.4 * fs, y_c - h_c / 2 - 0.5 * fs, dict(id="hc", sym="h", val=h_c, lvl=2))
    _bloc_efforts(S, items, e / 2, "B", 1)
    _etendre(S, s, fs)
    _nom(S, "T", e / 2, "Coupe A-A")
    return S.finish(s, [], "Coupe A-A au mur : mur, platine et chevilles, console vue de côté, main courante coupée, charges")


# ====================================================================== plan
def _points_plan(geo: Dict) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Les tronçons en plan [(P0, P1)] : droit le long de x ; en L, le second tronçon part vers
    le haut du dessin (y < 0) ; en U, le troisième revient parallèle au premier. Le sol est à
    l'intérieur de la ligne brisée, le vide à l'extérieur."""
    Ls = [sum(e) for e in geo["entraxes"]]
    dirs = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    out = []
    p = (0.0, 0.0)
    for L, d in zip(Ls, dirs):
        q = (p[0] + d[0] * L, p[1] + d[1] * L)
        out.append((p, q))
        p = q
    return out


def _repere(p0, p1):
    """(u, n) d'un tronçon : u le long, n vers l'extérieur (à droite du parcours)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    return (ux, uy), (-uy, ux)


def _sur_ligne(segs, x: float):
    """Le point de la ligne en plan à l'abscisse développée x, et le repère de son tronçon."""
    reste = x
    for (p0, p1) in segs:
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if reste <= L + 1e-6:
            t = reste / L if L else 0.0
            return (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t), _repere(p0, p1)
        reste -= L
    p0, p1 = segs[-1]
    return p1, _repere(p0, p1)


def plan(res: Dict, opt: Options) -> Dessin:
    geo = res["geometrie"]
    systeme = geo.get("systeme", "montants")
    segs = _points_plan(geo)
    xs = [p[0] for sg in segs for p in sg]
    ys = [p[1] for sg in segs for p in sg]
    w = max(max(xs) - min(xs), 1.0)
    h = max(max(ys) - min(ys), 1.0)
    marge = max(0.12 * max(w, h), 120.0)
    bb0 = dict(x1=min(xs) - marge, x2=max(xs) + marge, y1=min(ys) - marge, y2=max(ys) + marge)
    fs = opt.fs_force or _fs(bb0["x2"] - bb0["x1"]) * 1.1
    s: List[Dict] = []
    sec_p = res.get("section_montant") or {}
    sec_l = (res.get("lisse") or {}).get("section") or {}
    hp, b_post = _dim_sec(sec_p)
    h_mc, b_mc = _dim_sec(sec_l)
    rond = sec_p.get("forme") in ("creux_rond", "plein_rond")
    wp, dp = max(b_post, 0.5 * fs), max(hp, 0.5 * fs)
    bm = max(b_mc, 0.22 * fs)
    e_c = float((res.get("console") or {}).get("e") or 100.0)
    sec_c = (res.get("console") or {}).get("section") or {}
    if systeme == "mur":                 # le mur hachuré du côté extérieur, à e de l'axe de la main courante
        for (p0, p1) in segs:
            _u, (nx, ny) = _repere(p0, p1)
            pts = [(p0[0] + nx * e_c, p0[1] + ny * e_c), (p1[0] + nx * e_c, p1[1] + ny * e_c),
                   (p1[0] + nx * (e_c + 2.5 * fs), p1[1] + ny * (e_c + 2.5 * fs)), (p0[0] + nx * (e_c + 2.5 * fs), p0[1] + ny * (e_c + 2.5 * fs))]
            s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.7 * fs))
    # les montants (orientés selon leur tronçon) ou les consoles (du mur à l'axe de la main courante, qui les recouvre)
    pos = []
    for m in geo["montants"]:
        pt, ((ux, uy), (nx, ny)) = _sur_ligne(segs, m["x"])
        if m["troncon"] < len(segs):
            (ux, uy), (nx, ny) = _repere(*segs[m["troncon"]])
        pos.append((m, pt))
        if systeme == "mur":
            bc = max(_dim_sec(sec_c)[1], 0.3 * fs)
            s.append(_poly(["pp"], [(pt[0] + ux * bc / 2, pt[1] + uy * bc / 2), (pt[0] + ux * bc / 2 + nx * e_c, pt[1] + uy * bc / 2 + ny * e_c),
                                    (pt[0] - ux * bc / 2 + nx * e_c, pt[1] - uy * bc / 2 + ny * e_c), (pt[0] - ux * bc / 2, pt[1] - uy * bc / 2)]))
    # la main courante : une bande de largeur b le long de chaque tronçon
    for (p0, p1) in segs:
        (ux, uy), (nx, ny) = _repere(p0, p1)
        s.append(_poly(["ps"], [(p0[0] + nx * bm / 2, p0[1] + ny * bm / 2), (p1[0] + nx * bm / 2, p1[1] + ny * bm / 2),
                                (p1[0] - nx * bm / 2, p1[1] - ny * bm / 2), (p0[0] - nx * bm / 2, p0[1] - ny * bm / 2)]))
    for m, pt in pos:
        (ux, uy), (nx, ny) = _repere(*segs[min(m["troncon"], len(segs) - 1)])
        if systeme == "mur":
            continue
        if rond:
            s.append(_circle(["pp"], pt[0], pt[1], dp / 2))
        else:
            s.append(_poly(["pp"], [(pt[0] + ux * wp / 2 + nx * dp / 2, pt[1] + uy * wp / 2 + ny * dp / 2),
                                    (pt[0] - ux * wp / 2 + nx * dp / 2, pt[1] - uy * wp / 2 + ny * dp / 2),
                                    (pt[0] - ux * wp / 2 - nx * dp / 2, pt[1] - uy * wp / 2 - ny * dp / 2),
                                    (pt[0] + ux * wp / 2 - nx * dp / 2, pt[1] + uy * wp / 2 - ny * dp / 2)]))
    gouv = _gouvernant_nom(res)
    for m, pt in pos:
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if nom == gouv:
            s.append(_circle(["gv"], pt[0], pt[1], 0.55 * fs)); s.append(_circle(["gvd"], pt[0], pt[1], 0.2 * fs))
    # les murs d'extrémité
    for r in geo.get("reprises_mur", []):
        pt = segs[0][0] if r["ou"] == "début" else segs[-1][1]
        p0, p1 = segs[0] if r["ou"] == "début" else segs[-1]
        (ux, uy), (nx, ny) = _repere(p0, p1)
        sg = -1 if r["ou"] == "début" else 1
        pts = [(pt[0] + nx * 1.6 * fs, pt[1] + ny * 1.6 * fs), (pt[0] - nx * 1.6 * fs, pt[1] - ny * 1.6 * fs),
               (pt[0] - nx * 1.6 * fs + sg * ux * 2.0 * fs, pt[1] - ny * 1.6 * fs + sg * uy * 2.0 * fs),
               (pt[0] + nx * 1.6 * fs + sg * ux * 2.0 * fs, pt[1] + ny * 1.6 * fs + sg * uy * 2.0 * fs)]
        s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.6 * fs))
    # les charges : perpendiculaires à chaque tronçon, du sol vers le vide (au mur : elles tirent la main courante hors du mur)
    ch = res["charges"]
    nom_q = "P" if ch["mode"] == "P" else "q_h"
    premier = True
    for (p0, p1) in segs:
        (ux, uy), (nx, ny) = _repere(p0, p1)
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        for k in _pas(0.18 * L, 0.84 * L, max(L / 4.0, 2.5 * fs)):
            c = (p0[0] + ux * k, p0[1] + uy * k)
            if systeme == "mur":
                _fleche(s, (c[0] - nx * 0.9 * fs, c[1] - ny * 0.9 * fs), (c[0] - nx * 3.6 * fs, c[1] - ny * 3.6 * fs), nom_q if premier else "", fs)
            else:
                _fleche(s, (c[0] - nx * 3.4 * fs, c[1] - ny * 3.4 * fs), (c[0] - nx * 0.9 * fs, c[1] - ny * 0.9 * fs), nom_q if premier else "", fs)
            premier = False
    # le repère de la coupe A-A, au même endroit que sur l'élévation
    pt, ((ux, uy), (nx, ny)) = _sur_ligne(segs, _x_coupe(res))
    d_c = 2.0 * fs + (e_c if systeme == "mur" else 0.0)
    _trait_coupe(s, (pt[0] - nx * 2.0 * fs, pt[1] - ny * 2.0 * fs), (pt[0] + nx * d_c, pt[1] + ny * d_c), fs, "A")
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    for i, ((p0, p1), ent) in enumerate(zip(segs, geo["entraxes"])):
        L = sum(ent)
        if i == 0:
            S.dim(dict(id="L1", side="B", a=p0[0], b=p1[0], o1=p0[1], o2=p1[1], sym="L₁", val=L, lvl=1))
        elif i == 1:
            S.dim(dict(id="L2", side="R", a=min(p0[1], p1[1]), b=max(p0[1], p1[1]), o1=p0[0], o2=p1[0], sym="L₂", val=L, lvl=1))
        else:
            S.dim(dict(id="L3", side="T", a=min(p0[0], p1[0]), b=max(p0[0], p1[0]), o1=p0[1], o2=p1[1], sym="L₃", val=L, lvl=1))
    cote_0 = "B" if len(segs) == 3 else "T"
    for m, pt in pos:
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if m["troncon"] == 0 and not m["coin"]:
            S.name(cote_0, pt[0], nom)
        elif m["troncon"] <= 1:
            S.name("R", pt[1], nom)
        else:
            S.name("T", pt[0], nom)
    forme = "en L" if len(segs) == 2 else "en U" if len(segs) == 3 else "droit"
    _bloc_efforts(S, [nom_q + " perpendiculaire à chaque tronçon · " + forme], (bb0["x1"] + bb0["x2"]) / 2, cote_0, 1)
    _etendre(S, s, fs)
    _nom(S, cote_0, (bb0["x1"] + bb0["x2"]) / 2, "Vue en plan")
    return S.finish(s, [], "Vue en plan du garde-corps : la ligne, la main courante, les montants, les charges, le repère de coupe")


# ====================================================================== platine en plan
def platine(res: Dict, opt: Options) -> Dessin:
    """La platine vue en plan (au mur : vue sur la face du mur) : la section du montant à
    l'échelle, la bande du cordon, les trous des chevilles et leurs axes, e_x, e_y, b_p, l_p,
    les distances aux bords, la direction de la charge qui fléchit la platine."""
    fx = res["fixation"]
    pl = fx.get("platine") or {}
    if pl.get("mode") == "laterale":
        return _platine_laterale(res, opt)
    mur = res["geometrie"].get("systeme") == "mur"
    sec = (res["console"] if mur else res["montant"])["section"]
    bp, lp, tp = float(pl.get("bp") or 150.0), float(pl.get("lp") or 150.0), float(pl.get("tp") or 10.0)
    ex, ey, n = float(pl.get("ex") or 100.0), float(pl.get("ey") or 100.0), int(pl.get("n_tendus") or 2)
    d0 = float(pl.get("d0") or 14.0)
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    a = float(soud.get("a") or 4.0)
    jon = soud.get("jonction", "angle")

    def P(u, v):
        """u : le sens de e_x (la charge qui fléchit la platine) ; v : le sens de e_y (le long du garde-corps)."""
        return (v, u) if mur else (u, v)
    W, Hh = (lp, bp) if mur else (bp, lp)
    fs = opt.fs_force or _fs(max(W, Hh) + 70.0)
    bb0 = dict(x1=-W / 2, x2=W / 2, y1=-Hh / 2, y2=Hh / 2)
    s: List[Dict] = [_rect(["pp"], -W / 2, -Hh / 2, W, Hh)]
    polys = [[P(float(p[1]), float(p[0])) for p in r] for r in BI.contours(sec)]     # u = h (y du contour), v = b (x du contour)
    z = a * S2 if jon == "angle" else max(2.0, float(soud.get("p") or 0.0) + float(soud.get("renfort") or 0.0))
    s.append(_poly(["wb"], _decale(polys[0], z)))
    s.append(_poly(["ps"], polys[0]))
    for inn in polys[1:]:
        s.append(_poly(["bo"], inn))
    s.append(_path(["ht"], _hachures_anneaux(polys, max(1.5, 0.22 * fs))))
    vs = [(k - (n - 1) / 2) * ey for k in range(n)]
    for i in (-1, 1):
        for v in vs:
            x, y = P(i * ex / 2, v)
            s.append(_circle(["bo"], x, y, d0 / 2))
            r = d0 / 2 + 0.45 * fs
            s.append(_path(["ax"], [[(x - r, y), (x + r, y)], [(x, y - r), (x, y + r)]]))
    # la charge qui fléchit la platine, dans le sens de e_x, hors de la platine
    if mur:
        _fleche(s, (W / 2 + 1.4 * fs, -1.8 * fs), (W / 2 + 1.4 * fs, 1.8 * fs), "q_v", fs, cote=+1)
    else:
        _fleche(s, (-1.8 * fs, -Hh / 2 - 1.3 * fs), (1.8 * fs, -Hh / 2 - 1.3 * fs), "q_h", fs, cote=+1)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    if mur:
        _chaine(S, "R", [-bp / 2, -ex / 2, ex / 2, bp / 2], W / 2, lvl=2)
        S.dim(dict(id="ex", side="R", a=-ex / 2, b=ex / 2, o1=W / 2, o2=W / 2, sym="e_x", val=ex, lvl=1))
        S.dim(dict(id="bp", side="R", a=-bp / 2, b=bp / 2, o1=W / 2, o2=W / 2, sym="b_p", val=bp, lvl=1))
        if n > 1:
            _chaine(S, "B", [-lp / 2] + vs + [lp / 2], Hh / 2, lvl=2)
            S.dim(dict(id="ey", side="B", a=vs[0], b=vs[1], o1=Hh / 2, o2=Hh / 2, sym="e_y", val=ey, lvl=1))
        S.dim(dict(id="lp", side="B", a=-lp / 2, b=lp / 2, o1=Hh / 2, o2=Hh / 2, sym="l_p", val=lp, lvl=1))
    else:
        _chaine(S, "B", [-bp / 2, -ex / 2, ex / 2, bp / 2], Hh / 2, lvl=2)
        S.dim(dict(id="ex", side="B", a=-ex / 2, b=ex / 2, o1=Hh / 2, o2=Hh / 2, sym="e_x", val=ex, lvl=1))
        S.dim(dict(id="bp", side="B", a=-bp / 2, b=bp / 2, o1=Hh / 2, o2=Hh / 2, sym="b_p", val=bp, lvl=1))
        if n > 1:
            _chaine(S, "R", [-lp / 2] + vs + [lp / 2], W / 2, lvl=2)
            S.dim(dict(id="ey", side="R", a=vs[0], b=vs[1], o1=W / 2, o2=W / 2, sym="e_y", val=ey, lvl=1))
        S.dim(dict(id="lp", side="R", a=-lp / 2, b=lp / 2, o1=W / 2, o2=W / 2, sym="l_p", val=lp, lvl=1))
    _etendre(S, s, fs)
    _nom(S, "T", 0.0, f"Platine {bp:g} × {lp:g} × {tp:g} · {2 * n} {pl.get('nom_ancrage') or 'chevilles'}" + (" · au mur" if mur else ""))
    return S.finish(s, [], "Platine en plan : section du montant, bande du cordon, trous des chevilles, e_x, e_y, b_p, l_p")


def _platine_laterale(res: Dict, opt: Options) -> Dessin:
    """La platine latérale vue de FACE (depuis le vide) : la platine l_p × b_p, le montant qui la
    traverse de haut en bas (largeur b), les deux bandes des cordons longitudinaux, les trous des
    chevilles (rangée haute tendue, rangée basse en appui), e_x, e_y, b_p, l_p, le bras z = e_x
    du couple et le bras m de la platine en console, la charge P ⊙ (vers soi) en tête."""
    fx = res["fixation"]
    pl = fx["platine"]
    sec = res["montant"]["section"]
    bp, lp, tp = float(pl["bp"]), float(pl["lp"]), float(pl["tp"])
    ex, ey, n = float(pl["ex"]), float(pl["ey"]), int(pl["n_tendus"])
    b = float(pl.get("b_post") or _dim_sec(sec)[1])
    d0 = float(pl.get("d0") or 14.0)
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    els = soud.get("elements") or [{}]
    a = float(els[0].get("a") or 4.0)
    fs = opt.fs_force or _fs(max(lp, bp) + 90.0)
    z = a * S2
    bb0 = dict(x1=-lp / 2, x2=lp / 2, y1=-bp / 2 - 1.6 * fs, y2=bp / 2)
    s: List[Dict] = [_rect(["pp"], -lp / 2, -bp / 2, lp, bp)]
    wb = max(b, MIN_VIS * fs)
    s.append(_rect(["wb"], -wb / 2 - z, -bp / 2, wb + 2 * z, bp))                      # les deux bandes des cordons, le long du montant
    pts = _rect_rompu(-wb / 2, -bp / 2 - 1.6 * fs, wb, bp + 1.6 * fs, {"T"}, 0.45 * fs)
    s.append(_poly(["ps"], pts))
    vs = [(k - (n - 1) / 2) * ey for k in range(n)]
    for i in (-1, 1):
        for v in vs:
            x, y = v, i * ex / 2
            s.append(_circle(["bo"], x, y, d0 / 2))
            r = d0 / 2 + 0.45 * fs
            s.append(_path(["ax"], [[(x - r, y), (x + r, y)], [(x, y - r), (x, y + r)]]))
    _normal(s, (0.0, -bp / 2 - 1.1 * fs), True, "P", fs, etiq=(wb / 2 + 1.6 * fs, -bp / 2 - 0.8 * fs))
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    _chaine(S, "T", [-lp / 2] + vs + [lp / 2], -bp / 2, lvl=2)
    if n > 1:
        S.dim(dict(id="ey", side="T", a=vs[-2], b=vs[-1], o1=-bp / 2, o2=-bp / 2, sym="e_y", val=ey, lvl=1))
    S.dim(dict(id="lp", side="T", a=-lp / 2, b=lp / 2, o1=-bp / 2, o2=-bp / 2, sym="l_p", val=lp, lvl=1))
    _chaine(S, "R", [-bp / 2, -ex / 2, ex / 2, bp / 2], lp / 2, lvl=2)
    S.dim(dict(id="ex", side="R", a=-ex / 2, b=ex / 2, o1=lp / 2, o2=lp / 2, sym="e_x", val=ex, lvl=1))
    S.dim(dict(id="bp", side="R", a=-bp / 2, b=bp / 2, o1=lp / 2, o2=lp / 2, sym="b_p", val=bp, lvl=1))
    if pl.get("z", 0) > 0:
        S.dim(dict(id="z", side="L", a=-ex / 2, b=ex / 2, o1=vs[0] if vs else 0.0, o2=vs[0] if vs else 0.0, sym="z", val=pl["z"], lvl=2, calc=True))
    if pl.get("m", 0) > 0 and vs:
        S.dim(dict(id="m", side="B", a=wb / 2 + 0.8 * z, b=vs[-1], o1=-ex / 2, o2=-ex / 2, sym="m", val=pl["m"], lvl=2, calc=True))
    S.tag(wb / 2 + z + 0.4 * fs, bp / 2 - 0.4 * fs, dict(id="a", sym="a", val=a, lvl=1))
    _etendre(S, s, fs)
    _nom(S, "T", 0.0, f"Platine latérale {lp:g} × {bp:g} × {tp:g} · {2 * n} {pl.get('nom_ancrage') or 'chevilles'} · vue de face")
    return S.finish(s, [], "Platine latérale vue de face : montant, cordons longitudinaux, trous des chevilles, e_x, e_y, b_p, l_p, z, m")


# ====================================================================== coupe sur la fixation (pied)
def pied(res: Dict, opt: Options) -> Dessin:
    """Le pied du montant en coupe, à grande échelle : montant coupé, cordons, platine,
    chevilles dans le support, bras de levier m et z, efforts du cas gouvernant."""
    if res["geometrie"].get("systeme") == "mur":
        return _pied_console(res, opt)
    if ((res.get("fixation") or {}).get("platine") or {}).get("mode") == "laterale":
        return _pied_lateral(res, opt)
    sec = res["montant"]["section"]
    fx = res.get("fixation") or {}
    pl = fx.get("platine") or {}
    typ = fx.get("type", "platine")
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    a = float(soud.get("a") or 4.0)
    jon = soud.get("jonction", "angle")
    hp, _b = _dim_sec(sec)
    t = float(sec.get("t") or 0.0)
    plein = sec.get("forme") in ("plein_rect", "plein_rond")
    bp = float(pl.get("bp") or 150.0) if typ == "platine" else max(hp + 80.0, 140.0)
    tp = float(pl.get("tp") or 10.0) if typ == "platine" else 10.0
    ex, d = float(pl.get("ex") or 100.0), float(pl.get("d") or 12.0)
    Lv = max(2.2 * hp, 130.0)
    fs = opt.fs_force or _fs(max(bp, hp) + 190.0)
    ep = max(2.6 * fs, 90.0)
    z = a * S2 if jon == "angle" else (float(soud.get("p") or 3.0) + float(soud.get("renfort") or 0.0) if jon == "partielle" else max(0.6 * (t or hp), 2.0))
    s: List[Dict] = []
    bb0 = dict(x1=-bp / 2, x2=bp / 2, y1=-Lv, y2=tp + ep)
    beton = typ == "platine" and pl.get("As") is None          # chevilles : béton ; boulons ou plat existant : acier
    sup = _rect_rompu(-bp / 2 - 2.2 * fs, tp, bp + 4.4 * fs, ep, {"B", "L", "R"}, 0.5 * fs)
    s.append(_poly(["be" if beton else "pp"], sup)); s.append(_hach(["ht"], sup, 0.55 * fs))
    P = [(-bp / 2, 0.0), (bp / 2, 0.0), (bp / 2, tp), (-bp / 2, tp)]
    s.append(_poly(["pp" if typ == "platine" else "ps"], P)); s.append(_hach(["ht"], P, 0.3 * fs))
    if typ == "platine":
        prof = min(8 * d, 0.85 * ep)
        for sg in (-1, 1):
            x = sg * ex / 2
            s.append(_rect(["tg"], x - d / 2, -0.8 * d, d, tp + 0.8 * d + prof))
            s.append(_rect(["bo"], x - 0.8 * d, -0.8 * d, 1.6 * d, 0.8 * d))
    if plein:
        pts = _rect_rompu(-hp / 2, -Lv, hp, Lv, {"T"}, 0.45 * fs)
        s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.3 * fs))
    else:
        for sg in (-1, 1):
            x0 = sg * hp / 2 - (t if sg > 0 else 0.0)
            pts = _rect_rompu(x0, -Lv, t, Lv, {"T"}, 0.45 * fs)
            s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.25 * fs))
    s.append(_path(["ax"], [[(0.0, -Lv - 0.6 * fs), (0.0, tp + 0.5 * fs)]]))
    for sg in (-1, 1):
        _cordons(s, sg * hp / 2, 0.0, z, sg, -1, 0.2 * fs)
    cas, V, N, M = _efforts_fixation(res)
    _fleche(s, (-hp / 2 - 4.4 * fs, -Lv + 0.9 * fs), (-hp / 2 - 0.3 * fs, -Lv + 0.9 * fs), "V", fs, cote=+1)
    if N > 0:
        _fleche(s, (0.0, -Lv - 3.4 * fs), (0.0, -Lv - 0.4 * fs), "N", fs, cote=+1)
    elif N < 0:
        _fleche(s, (0.0, -Lv - 0.4 * fs), (0.0, -Lv - 3.4 * fs), "N", fs, cote=+1)
    if M > 0:
        _arc_moment(s, (hp / 2 + 2.4 * fs, -Lv + 1.2 * fs), 1.4 * fs, -1, "M", fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="hp", side="T", a=-hp / 2, b=hp / 2, o1=-Lv, o2=-Lv, sym="h_p", val=hp, lvl=1))
    if typ == "platine":
        S.dim(dict(id="bp", side="B", a=-bp / 2, b=bp / 2, o1=tp + ep, o2=tp + ep, sym="b_p", val=bp, lvl=1))
        S.dim(dict(id="ex", side="B", a=-ex / 2, b=ex / 2, o1=tp + ep, o2=tp + ep, sym="e_x", val=ex, lvl=2))
        S.dim(dict(id="tp", side="R", a=0.0, b=tp, o1=bp / 2, o2=bp / 2, sym="t_p", val=tp, lvl=1, out="hi"))
        if pl.get("m", 0) > 0:
            S.dim(dict(id="m", side="T", a=-ex / 2, b=-hp / 2 - 0.8 * z, o1=-0.8 * d, o2=-z, sym="m", val=pl["m"], lvl=2, calc=True))
        if pl.get("z", 0) > 0:
            S.dim(dict(id="z", side="T", a=-ex / 2, b=hp / 2, o1=-0.8 * d, o2=-Lv, sym="z", val=pl["z"], lvl=2, calc=True))
    S.tag(hp / 2 + z + 0.5 * fs, -z - 0.2 * fs, dict(id="a", sym="a" if jon == "angle" else "s", val=a if jon == "angle" else z, lvl=1))
    _bloc_efforts(S, [_texte_efforts(cas, V, N, M)], 0.0, "B", 1)
    _etendre(S, s, fs)
    _nom(S, "T", 0.0, "Coupe sur la fixation · " + sec["nom"] + (" sur platine" if typ == "platine" else " soudé sur plat"))
    return S.finish(s, [], "Coupe sur la fixation : montant coupé, cordons, platine, chevilles, bras de levier, efforts")


def _pied_console(res: Dict, opt: Options) -> Dessin:
    """La console au mur en coupe, à grande échelle : mur, platine verticale, chevilles
    horizontales, console coupée, cordons, bras de levier, efforts du cas gouvernant."""
    sec = res["console"]["section"]
    fx = res.get("fixation") or {}
    pl = fx.get("platine") or {}
    typ = fx.get("type", "platine")
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    a = float(soud.get("a") or 4.0)
    jon = soud.get("jonction", "angle")
    h_c, _b = _dim_sec(sec)
    t = float(sec.get("t") or 0.0)
    plein = sec.get("forme") in ("plein_rect", "plein_rond")
    bp = float(pl.get("bp") or 150.0) if typ == "platine" else max(h_c + 80.0, 140.0)
    tp = float(pl.get("tp") or 10.0) if typ == "platine" else 10.0
    ex, d = float(pl.get("ex") or 100.0), float(pl.get("d") or 12.0)
    Lv = max(2.2 * h_c, 140.0)
    fs = opt.fs_force or _fs(max(bp, 2 * Lv) + 280.0)
    ep = max(2.6 * fs, 90.0)
    z = a * S2 if jon == "angle" else (float(soud.get("p") or 3.0) + float(soud.get("renfort") or 0.0) if jon == "partielle" else max(0.6 * (t or h_c), 2.0))
    s: List[Dict] = []
    bb0 = dict(x1=-ep, x2=tp + Lv, y1=-bp / 2, y2=bp / 2)
    sup = _rect_rompu(-ep, -bp / 2 - 2.2 * fs, ep, bp + 4.4 * fs, {"T", "B", "L"}, 0.5 * fs)
    s.append(_poly(["be" if typ == "platine" and pl.get("As") is None else "pp"], sup)); s.append(_hach(["ht"], sup, 0.55 * fs))
    P = [(0.0, -bp / 2), (tp, -bp / 2), (tp, bp / 2), (0.0, bp / 2)]
    s.append(_poly(["pp" if typ == "platine" else "ps"], P)); s.append(_hach(["ht"], P, 0.3 * fs))
    if typ == "platine":
        prof = min(8 * d, 0.85 * ep)
        for sg in (-1, 1):
            y = sg * ex / 2
            s.append(_rect(["tg"], -prof, y - d / 2, prof + tp + 0.8 * d, d))
            s.append(_rect(["bo"], tp, y - 0.8 * d, 0.8 * d, 1.6 * d))
    if plein:
        pts = _rect_rompu(tp, -h_c / 2, Lv, h_c, {"R"}, 0.45 * fs)
        s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.3 * fs))
    else:
        for sg in (-1, 1):
            y0 = sg * h_c / 2 - (t if sg > 0 else 0.0)
            pts = _rect_rompu(tp, y0, Lv, t, {"R"}, 0.45 * fs)
            s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.25 * fs))
    s.append(_path(["ax"], [[(-0.5 * fs, 0.0), (tp + Lv + 0.6 * fs, 0.0)]]))
    for sg in (-1, 1):
        _cordons(s, tp, sg * h_c / 2, z, +1, sg, 0.2 * fs)
    cas, V, N, M = _efforts_fixation(res)
    xe = tp + Lv - 0.9 * fs
    _fleche(s, (xe, -h_c / 2 - 4.4 * fs), (xe, -h_c / 2 - 0.3 * fs), "V", fs, cote=+1)
    if N < 0:
        _fleche(s, (tp + Lv + 0.4 * fs, 0.0), (tp + Lv + 3.4 * fs, 0.0), "N", fs, cote=+1)
    elif N > 0:
        _fleche(s, (tp + Lv + 3.4 * fs, 0.0), (tp + Lv + 0.4 * fs, 0.0), "N", fs, cote=+1)
    if M > 0:
        _arc_moment(s, (tp + Lv - 1.2 * fs, h_c / 2 + 2.4 * fs), 1.4 * fs, +1, "M", fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="bp", side="L", a=-bp / 2, b=bp / 2, o1=0.0, o2=0.0, sym="b_p", val=bp, lvl=1))
    if typ == "platine":
        S.dim(dict(id="ex", side="L", a=-ex / 2, b=ex / 2, o1=0.0, o2=0.0, sym="e_x", val=ex, lvl=2))
        if pl.get("m", 0) > 0:
            S.dim(dict(id="m", side="L", a=-ex / 2, b=-h_c / 2 - 0.8 * z, o1=-0.8 * d, o2=-z, sym="m", val=pl["m"], lvl=2, calc=True))
        if pl.get("z", 0) > 0:
            S.dim(dict(id="z", side="L", a=-ex / 2, b=h_c / 2, o1=-0.8 * d, o2=tp + Lv - 0.6 * fs, sym="z", val=pl["z"], lvl=2, calc=True))
    S.dim(dict(id="tp", side="T", a=0.0, b=tp, o1=-bp / 2, o2=-bp / 2, sym="t_p", val=tp, lvl=1, out="hi"))
    S.tag(tp + 0.4 * fs, h_c / 2 + z + 1.3 * fs, dict(id="hc", sym="h", val=h_c, lvl=2))
    S.tag(tp + z + 0.5 * fs, -h_c / 2 - z - 0.2 * fs, dict(id="a", sym="a" if jon == "angle" else "s", val=a if jon == "angle" else z, lvl=1))
    _bloc_efforts(S, [_texte_efforts(cas, V, N, M)], (tp + Lv) / 2, "B", 1)
    _etendre(S, s, fs)
    _nom(S, "T", (tp + Lv) / 2, "Coupe sur la fixation · console " + sec["nom"] + " au mur")
    return S.finish(s, [], "Coupe sur la fixation au mur : mur, platine, chevilles, console coupée, cordons, bras de levier, efforts")


def _pied_lateral(res: Dict, opt: Options) -> Dessin:
    """La fixation latérale en COUPE HORIZONTALE, à grande échelle, au niveau de la rangée tendue : la
    dalle (à gauche, son nez en x = 0), la platine contre le nez, la section du montant (h vers le
    vide, b le long de la platine), les deux cordons longitudinaux coupés, les chevilles horizontales
    des colonnes, l_p, e_y, t_p, h, b, le bras m, les efforts du cas gouvernant (V en ⊗)."""
    sec = res["montant"]["section"]
    fx = res.get("fixation") or {}
    pl = fx.get("platine") or {}
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    els = soud.get("elements") or [{}]
    a = float(els[0].get("a") or 4.0)
    hp, b = _dim_sec(sec)
    lp, tp = float(pl.get("lp") or 150.0), float(pl.get("tp") or 10.0)
    ey, n, d = float(pl.get("ey") or 100.0), int(pl.get("n_tendus") or 2), float(pl.get("d") or 12.0)
    fs = opt.fs_force or _fs(lp + hp + 240.0)
    ep = max(3.0 * fs, 110.0)
    z = a * S2
    s: List[Dict] = []
    bb0 = dict(x1=-ep, x2=tp + hp + 1.5 * fs, y1=-lp / 2, y2=lp / 2)
    sup = _rect_rompu(-ep, -lp / 2 - 2.2 * fs, ep, lp + 4.4 * fs, {"T", "B", "L"}, 0.5 * fs)
    beton = pl.get("As") is None
    s.append(_poly(["be" if beton else "pp"], sup)); s.append(_hach(["ht"], sup, 0.55 * fs))
    P = [(0.0, -lp / 2), (tp, -lp / 2), (tp, lp / 2), (0.0, lp / 2)]
    s.append(_poly(["pp"], P)); s.append(_hach(["ht"], P, 0.3 * fs))
    xs = [(k - (n - 1) / 2) * ey for k in range(n)]
    prof = min(8 * d, 0.85 * ep)
    for y in xs:
        s.append(_rect(["tg"], -prof, y - d / 2, prof + tp + 0.8 * d, d))
        s.append(_rect(["bo"], tp, y - 0.8 * d, 0.8 * d, 1.6 * d))
    _coupe_section(s, sec, tp + hp / 2, 0.0, max(1.5, 0.22 * fs), h_vertical=False)
    for sg in (-1, 1):
        _cordons(s, tp, sg * b / 2, z, +1, sg, 0.2 * fs)
    s.append(_path(["ax"], [[(-0.5 * fs, 0.0), (tp + hp + 1.0 * fs, 0.0)]]))
    cas, V, N, M = _efforts_fixation(res)
    xe = tp + hp + 0.5 * fs
    if N < 0:
        _fleche(s, (xe + 0.2 * fs, 0.0), (xe + 3.4 * fs, 0.0), "P", fs, cote=+1)
    if V > 0:                                                          # le cisaillement vertical traverse la coupe horizontale : ⊗ (vers le fond)
        _normal(s, (tp + hp / 2, -b / 2 - 2.2 * fs), False, "V", fs, etiq=(tp + hp / 2 + 1.5 * fs, -b / 2 - 2.0 * fs))
    if M > 0:
        _arc_moment(s, (xe + 1.0 * fs, b / 2 + 2.6 * fs), 1.4 * fs, +1, "M", fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="lp", side="L", a=-lp / 2, b=lp / 2, o1=0.0, o2=0.0, sym="l_p", val=lp, lvl=1))
    if n > 1:
        S.dim(dict(id="ey", side="L", a=xs[-2], b=xs[-1], o1=0.0, o2=0.0, sym="e_y", val=ey, lvl=2))
    if pl.get("m", 0) > 0:
        S.dim(dict(id="m", side="R", a=b / 2 + 0.8 * z, b=xs[-1], o1=tp + 0.8 * d, o2=tp, sym="m", val=pl["m"], lvl=2, calc=True))
    S.dim(dict(id="tp", side="T", a=0.0, b=tp, o1=-lp / 2, o2=-lp / 2, sym="t_p", val=tp, lvl=1, out="hi"))
    S.dim(dict(id="hp", side="T", a=tp, b=tp + hp, o1=-b / 2, o2=-b / 2, sym="h", val=hp, lvl=2))
    S.tag(tp + hp + 0.4 * fs, -b / 2 - 0.2 * fs, dict(id="b", sym="b", val=b, lvl=2))
    S.tag(tp + z + 0.5 * fs, -b / 2 - z - 0.2 * fs, dict(id="a", sym="a", val=a, lvl=1))
    _bloc_efforts(S, [_texte_efforts(cas, V, N, M).replace("(traction)", "(arrachement P)")], (tp + hp) / 2, "B", 1)
    _etendre(S, s, fs)
    _nom(S, "T", (tp + hp) / 2, "Coupe horizontale sur la fixation · " + sec["nom"] + " sur platine latérale")
    return S.finish(s, [], "Coupe horizontale sur la fixation latérale : dalle, platine, section du montant, cordons coupés, chevilles, bras m, efforts")


# ====================================================================== section
def section(res: Dict, opt: Options) -> Dessin:
    """La section de l'élément principal, à l'échelle : contour du catalogue, hachures de la
    matière, axes y-y et z-z, la direction de la charge horizontale (selon h), les cotes."""
    sec = res.get("section_montant") or {}
    if not sec:
        return Dessin((0, 0, 100, 60), [], [], [], "section", 6.0)
    rings = BI.contours(sec)
    pts_all = [(float(p[0]), float(p[1])) for ring in rings for p in ring]
    xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    bb0 = dict(x1=min(xs) - 0.4 * w - 10, x2=max(xs) + 0.4 * w + 10, y1=-max(ys) - 0.4 * h - 10, y2=-min(ys) + 0.4 * h + 10)
    fs = opt.fs_force or _fs(bb0["x2"] - bb0["x1"])
    s: List[Dict] = []
    polys = [[(float(p[0]), -float(p[1])) for p in r] for r in rings]
    s.append(_poly(["pp"], polys[0]))
    for inn in polys[1:]:
        s.append(_poly(["bo"], inn))
    s.append(_path(["ht"], _hachures_anneaux(polys, 0.45 * fs)))
    xa0, xa1 = min(xs) - 0.3 * w, max(xs) + 0.3 * w
    ya0, ya1 = -max(ys) - 0.3 * h, -min(ys) + 0.3 * h
    s.append(_path(["ax"], [[(xa0, 0), (xa1, 0)], [(0, ya0), (0, ya1)]]))
    s.append(_text(["tx2"], xa1 + 0.55 * fs, 0.34 * fs, "y", fs))
    s.append(_text(["tx2"], 0.0, ya0 - 0.35 * fs, "z", fs))
    mur = res["geometrie"].get("systeme") == "mur"
    _fleche(s, (0.0, ya1 + 2.6 * fs), (0.0, ya1 + 0.3 * fs), "q_v" if mur else ("q_h" if res["geometrie"].get("sans_montant") else "P₁"), fs, cote=+1)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    d = sec["dims"]
    if sec["forme"] in ("creux_rond", "plein_rond"):
        S.dim(dict(id="d", side="T", a=min(xs), b=max(xs), o1=-max(ys), o2=-max(ys), sym="Ø", val=d["d"], lvl=1))
    else:
        S.dim(dict(id="b", side="T", a=min(xs), b=max(xs), o1=-max(ys), o2=-max(ys), sym="b", val=d["b"], lvl=1))
        S.dim(dict(id="h", side="L", a=-max(ys), b=-min(ys), o1=min(xs), o2=min(xs), sym="h", val=d["h"], lvl=1))
    if "t" in d:
        S.tag(max(xs) + 0.3 * fs, -min(ys) * 0.5, dict(id="t", sym="t", val=d["t"], lvl=1))
    _etendre(S, s, fs)
    _nom(S, "B", 0.0, sec["nom"] + " · " + F(sec["masse"], 2) + " kg/m")
    return S.finish(s, [], "Section de l'élément principal : contour, axes y-y et z-z, direction de la charge")


# ====================================================================== l'entrée
VUES = (("elevation", "Élévation", elevation), ("coupe", "Coupe A-A", coupe), ("plan", "Vue en plan", plan),
        ("platine", "Platine en plan", platine), ("pied", "Coupe sur la fixation", pied), ("section", "Section", section))


def vues(res: Dict, opt: Optional[Options] = None) -> List[Tuple[str, str, Dessin]]:
    """Les vues d'un résultat : [(clé, titre, Dessin)] — aucune sur un refus (géométrie, section
    de la main courante ou de l'élément principal) ; sans platine en plan si le montant n'est
    pas sur platine boulonnée, sans coupe sur la fixation s'il est scellé."""
    geo = res.get("geometrie") or {}
    sans_montant = not geo.get("montants") and bool(geo.get("travees")) and res.get("lisse") is not None
    if not res.get("lisse") or not res.get("section_montant") or not (res.get("montant") or res.get("console") or sans_montant):
        return []
    opt = opt or options_schemas()
    typ_fix = (res.get("fixation") or {}).get("type")
    n_tr = sum(1 for e in geo.get("entraxes", []) if e)
    out = []
    for cle, titre, fn in VUES:
        if cle in ("platine", "pied") and sans_montant:
            continue
        if cle == "platine" and typ_fix not in ("platine", "laterale"):
            continue
        if cle == "pied" and typ_fix == "scellee":
            continue
        if cle == "elevation" and n_tr > 1:
            titre = "Élévation développée"
        if typ_fix == "laterale" and geo.get("systeme") == "montants":
            titre = {"platine": "Platine latérale (vue de face)", "pied": "Coupe horizontale sur la fixation"}.get(cle, titre)
        out.append((cle, titre, fn(res, opt)))
    return out


def svg(dessin: Dessin, identifiant: str = "gc") -> str:
    return vers_svg(dessin, PALETTE, "100%", identifiant)

# -*- coding: utf-8 -*-
"""Les SCHÉMAS du garde-corps, sur le kit de dessin des assemblages (feuille de cotation,
primitives, hachures, lignes de rupture, rendu SVG — ``acier.assemblages….schemas``) et
avec les briques des schémas de soudure (flèches d'effort, étiquettes, ⊙, étendue réelle du
corps). Le même peintre vectoriel sert la note (``rapport.peindre``).

Trois vues (``vues(res)`` : [(clé, titre, Dessin)]), scènes en mm, y vers le bas :

- ÉLÉVATION développée : les montants (ou les consoles) à leurs entraxes réels, la lisse
  haute, le remplissage (barreaux ou lisses intermédiaires), les platines, les angles d'un L ou
  d'un U marqués, les murs d'extrémité hachurés, les charges (q_h ⊙ vers soi, q_v ↓), le montant
  gouvernant ◉ avec son taux, la chaîne des entraxes, H et la longueur développée ;
- PLAN : la ligne en plan (droit, L, U) avec les montants, les charges perpendiculaires à
  chaque tronçon, les longueurs ;
- DÉTAIL de la fixation : le pied du montant (ou la console au mur) en coupe — montant coupé
  (ligne de rupture), platine, cordons à leur taille a·√2, chevilles ou boulons dans le support
  hachuré, bras de levier, efforts M, V, N, cotes b_p, t_p, e_x, h_p, a.

Rien n'est recalculé : tout vient de ``etat.verifier``. Pureté : aucun Streamlit.
"""
import math
from typing import Dict, List, Optional, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # kit partagé
    Dessin, Feuille, Options, PALETTE, _circle, _group, _hachures, _path, _poly, _rect, _rect_rompu, _text, vers_svg)
from acier.formats import F, f0
from acier.soudures.schemas import _bb, _bloc_efforts, _etendre, _fleche, _fs, _label, _nom, _normal, _tw  # briques partagées

from . import bibliotheque as BI

S2 = math.sqrt(2.0)


def options_schemas(lvl: int = 2) -> Options:
    return Options(lvl=lvl, interactive=False, cartouche=False, realiste=True)


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


def _chaine(S, side: str, xs: List[float], o: float, lvl: int = 1):
    """Une CHAÎNE de cotes sur une seule rangée du côté ``side`` (T ou B) : lignes d'attache à
    chaque abscisse, une ligne de cote continue, un repère à chaque abscisse et la valeur de
    chaque intervalle centrée dedans (déportée si l'intervalle est trop court)."""
    if len(xs) < 2 or lvl > S.opt.lvl:
        return
    fs = S.fs
    t = 0.34 * fs
    r = S.slot(side, min(xs), max(xs))
    p = S.pos(side, r)
    e = p + (-0.45 if side == "T" else 0.45) * fs
    S.ex["x1"] = min(S.ex["x1"], min(xs))
    S.ex["x2"] = max(S.ex["x2"], max(xs))
    ext = _path(["ext"], [[(x, o), (x, e)] for x in xs], hint="V")
    marques = [[(min(xs), p), (max(xs), p)]] + [[(x - t, p + t), (x + t, p - t)] for x in xs]
    dln = _path(["dln"], marques, hint="H")
    labs = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        txt = f0(x1 - x0)
        w = S.tw(txt)
        c = (x0 + x1) / 2
        if x1 - x0 < w + 0.9 * fs:                   # trop court : la valeur au-dessus de la rangée, décalée
            labs.append(S.label(c, p - 1.5 * fs, False, txt, dict(id=f"ch{len(labs)}", sym="", lvl=lvl)))
        else:
            labs.append(S.label(c, p - 0.42 * fs, False, txt, dict(id=f"ch{len(labs)}", sym="", lvl=lvl)))
    S.g.append(_group(["dm"], [ext, dln] + labs))


def _kn(v_N: float) -> str:
    return F(abs(v_N) / 1e3, 2) + " kN"


def _m(v_mm: float) -> str:
    return F(v_mm / 1000.0, 2) + " m"


# ====================================================================== élévation développée
def _mur(s, x, y1, y2, fs, cote):
    """Un mur d'extrémité : bande hachurée de 2,5·fs de large, du côté ``cote`` (−1 à gauche)."""
    w = 2.5 * fs
    pts = [(x, y1), (x + cote * w, y1), (x + cote * w, y2), (x, y2)]
    s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.7 * fs))


def elevation(res: Dict, opt: Options) -> Dessin:
    geo = res["geometrie"]
    H = geo["H"]
    systeme = geo.get("systeme", "montants")
    Lt = geo["L_total"]
    montants = geo["montants"]
    sec_p = res.get("section_montant") or {}
    sec_l = (res.get("lisse") or {}).get("section") or {}
    rp = res.get("remplissage") or {}
    fx = res.get("fixation") or {}
    bb0 = dict(x1=0.0, x2=Lt, y1=-H - 60.0, y2=60.0)
    fs = _fs(bb0["x2"] - bb0["x1"]) * 1.15
    s: List[Dict] = []
    w_post = max(float(sec_p.get("b") or sec_p.get("d") or 40.0), 0.9 * fs) if systeme == "montants" else max(float(sec_p.get("h") or 40.0), 0.9 * fs)
    h_lisse = max(float(sec_l.get("h") or sec_l.get("d") or 40.0), 0.8 * fs)
    sol = 0.0
    # le sol (ou le mur, système mur) : trait fort et hachures sous la ligne
    s.append(_path(["fl2"], [[(bb0["x1"] - 2 * fs, sol), (bb0["x2"] + 2 * fs, sol)]]))
    s.append(_path(["ht"], [[(x, sol), (x - 0.8 * fs, sol + 0.8 * fs)] for x in _pas(bb0["x1"] - fs, bb0["x2"] + 2 * fs, 1.0 * fs)]))
    # la lisse haute sur toute la longueur
    y_l = -H
    s.append(_rect(["ps"], 0.0, y_l - h_lisse / 2, Lt, h_lisse))
    # le remplissage
    typ = rp.get("type")
    if typ == "barreaux" and rp.get("verif"):
        eb, Lb = rp["entraxe"], rp["L"]
        wb = max(float((rp.get("section") or {}).get("b") or (rp.get("section") or {}).get("d") or 8.0), 0.35 * fs)
        y0 = y_l + h_lisse / 2
        y1 = min(y0 + Lb, sol - 0.6 * fs)
        xs = _pas(eb, Lt - eb / 2, eb)
        if len(xs) > 90:
            xs = xs[:: int(math.ceil(len(xs) / 90.0))]
        for x in xs:
            s.append(_rect(["ps"], x - wb / 2, y0, wb, y1 - y0))
        s.append(_rect(["ps"], 0.0, y1, Lt, max(0.5 * h_lisse, 0.4 * fs)))      # la lisse basse
    elif typ == "lisses" and rp.get("verif"):
        el, nl = rp["entraxe"], int(rp.get("n") or 1)
        hl = max(float((rp.get("section") or {}).get("h") or (rp.get("section") or {}).get("d") or 20.0), 0.4 * fs)
        for k in range(1, nl + 1):
            y = y_l + k * el
            if y < sol - 0.6 * fs:
                s.append(_rect(["ps"], 0.0, y - hl / 2, Lt, hl))
    elif typ == "panneau":
        pts = [(0.4 * fs, y_l + h_lisse / 2 + 0.3 * fs), (Lt - 0.4 * fs, y_l + h_lisse / 2 + 0.3 * fs), (Lt - 0.4 * fs, sol - 0.8 * fs), (0.4 * fs, sol - 0.8 * fs)]
        s.append(_poly(["bg"], pts)); s.append(_path(["hd"], [[pts[0], pts[2]], [pts[1], pts[3]]]))
    # les montants (ou les consoles vues de face : une plaque au mur) et leurs platines
    pl = (fx.get("platine") or {}) if fx else {}
    bp = float(pl.get("bp") or 150.0)
    tp = float(pl.get("tp") or 10.0)
    gouv = ((res.get("montant") or res.get("console")) or {}).get("nom")
    for m in montants:
        x = m["x"]
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if systeme == "montants":
            s.append(_rect(["pp"], x - w_post / 2, y_l + h_lisse / 2, w_post, sol - (y_l + h_lisse / 2)))
            if fx.get("type") == "platine":
                s.append(_rect(["pp"], x - bp / 2, sol - tp, bp, tp))
                for sg in (-1, 1):
                    s.append(_rect(["tg"], x + sg * bp / 3 - 0.12 * fs, sol - tp, 0.24 * fs, tp + 1.6 * fs))
        else:
            s.append(_rect(["pp"], x - w_post / 2, y_l - h_lisse / 2 - 0.8 * fs, w_post, h_lisse + 1.6 * fs))
        if m["coin"]:
            s.append(_path(["ax"], [[(x, y_l - h_lisse / 2 - 2.6 * fs), (x, sol + 0.6 * fs)]]))
        if nom == gouv:
            yg = sol - 0.6 * fs if systeme == "montants" else y_l
            s.append(_circle(["gv"], x, yg, 0.5 * fs)); s.append(_circle(["gvd"], x, yg, 0.2 * fs))
    # les murs d'extrémité
    if geo.get("reprises_mur"):
        for r in geo["reprises_mur"]:
            _mur(s, 0.0 if r["ou"] == "début" else Lt, y_l - h_lisse / 2 - 1.5 * fs, sol, fs, -1 if r["ou"] == "début" else +1)
    # les charges : q_v en flèches vers le bas, q_h en ⊙ (vers soi)
    ch = res["charges"]
    items = []
    xs_q = _pas(0.5 * fs + Lt * 0.06, Lt - 0.5 * fs, max(Lt / 8.0, 2.2 * fs))
    if ch["mode"] == "P":
        for m in montants:
            _normal(s, (m["x"], y_l), True, "", fs)
        items.append(f"P = {F(ch['P'], 2)} kN par montant ⊙ (vers soi)")
    else:
        for x in xs_q:
            _normal(s, (x, y_l), True, "", fs)
        items.append(f"q_h = {F(ch['qh'], 2)} kN/m ⊙ (vers soi), à la main courante")
    if ch["qv"] > 0:
        for x in xs_q:
            _fleche(s, (x, y_l - h_lisse / 2 - 3.2 * fs), (x, y_l - h_lisse / 2 - 0.2 * fs), "", fs)
        items.append(f"q_v = {F(ch['qv'], 2)} kN/m ↓")
    if ch.get("Q", 0) > 0:
        items.append(f"Q = {F(ch['Q'], 2)} kN sur la lisse")
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    # cotes : la chaîne des entraxes sur une rangée, la longueur, H
    _chaine(S, "B", [0.0] + [t["x0"] + t["L"] for t in geo["travees"]], sol)
    if len(geo["travees"]) > 1:
        S.dim(dict(id="L", side="B", a=0.0, b=Lt, o1=sol, o2=sol, sym="L", val=Lt, lvl=1))
    S.dim(dict(id="H", side="L", a=y_l, b=sol, o1=0.0, o2=0.0, sym="H", val=H, lvl=1))
    t_g = ((res.get("montant") or res.get("console")) or {}).get("verif", {}).get("taux_max")
    for m in montants:
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if nom == gouv and t_g is not None:
            _nom(S, "T", m["x"], f"◉ {nom} · {t_g * 100:.0f} %", cls="fzl")
        else:
            S.name("T", m["x"], nom)
    for i, ent in enumerate(geo["entraxes"]):
        if i and ent:
            x = sum(sum(e) for e in geo["entraxes"][:i])
            _nom(S, "T", x, "angle")
    _bloc_efforts(S, items, Lt / 2, "B", 2)
    _etendre(S, s, fs)
    titre = ("Élévation développée · " + ("montants " + (sec_p.get("nom") or "") if systeme == "montants" else "lisse " + (sec_l.get("nom") or "") + " sur consoles"))
    _nom(S, "T", Lt / 2, titre)
    return S.finish(s, [], "Élévation développée du garde-corps : montants, lisse, remplissage, charges, entraxes")


def _pas(x0: float, x1: float, pas: float) -> List[float]:
    out = []
    x = x0
    n = 0
    while x <= x1 + 1e-9 and n < 2000:
        out.append(x)
        x += pas
        n += 1
    return out


# ====================================================================== plan
def _points_plan(geo: Dict) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Les tronçons en plan [(P0, P1)] : droit le long de x ; en L, le second tronçon part vers
    le haut du dessin (y < 0) ; en U, le troisième revient parallèle au premier."""
    Ls = [sum(e) for e in geo["entraxes"]]
    dirs = [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    out = []
    p = (0.0, 0.0)
    for L, d in zip(Ls, dirs):
        q = (p[0] + d[0] * L, p[1] + d[1] * L)
        out.append((p, q))
        p = q
    return out


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
    fs = _fs(bb0["x2"] - bb0["x1"]) * 1.1
    s: List[Dict] = []
    sec_p = res.get("section_montant") or {}
    r_post = max(float(sec_p.get("b") or sec_p.get("d") or 40.0), 0.7 * fs) / 2
    # la lisse en plan : un trait épais le long des tronçons, le mur hachuré du côté intérieur (système mur)
    for (p0, p1) in segs:
        s.append(_path(["we" if False else "fl2"], [[p0, p1]]))
        s.append(_rect(["ps"], min(p0[0], p1[0]) - (0.25 * fs if p0[0] == p1[0] else 0), min(p0[1], p1[1]) - (0.25 * fs if p0[1] == p1[1] else 0),
                       abs(p1[0] - p0[0]) + (0.5 * fs if p0[0] == p1[0] else 0), abs(p1[1] - p0[1]) + (0.5 * fs if p0[1] == p1[1] else 0)))
    if systeme == "mur":
        for (p0, p1) in segs:
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L                      # côté extérieur (à droite du parcours) : le mur
            e = float((res.get("console") or {}).get("e") or 100.0)
            pts = [(p0[0] + nx * e, p0[1] + ny * e), (p1[0] + nx * e, p1[1] + ny * e), (p1[0] + nx * (e + 2.5 * fs), p1[1] + ny * (e + 2.5 * fs)),
                   (p0[0] + nx * (e + 2.5 * fs), p0[1] + ny * (e + 2.5 * fs))]
            s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.7 * fs))
    # les montants à leur abscisse développée
    pos = []
    for m in geo["montants"]:
        x = m["x"]
        reste = x
        pt = None
        for (p0, p1) in segs:
            L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if reste <= L + 1e-6:
                t = reste / L if L else 0.0
                pt = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
                break
            reste -= L
        if pt is None:
            pt = segs[-1][1]
        pos.append((m, pt))
        if sec_p.get("forme") in ("creux_rond", "plein_rond") or systeme == "mur":
            s.append(_circle(["pp"], pt[0], pt[1], r_post))
        else:
            s.append(_rect(["pp"], pt[0] - r_post, pt[1] - r_post, 2 * r_post, 2 * r_post))
    gouv = ((res.get("montant") or res.get("console")) or {}).get("nom")
    for m, pt in pos:
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if nom == gouv:
            s.append(_circle(["gv"], pt[0], pt[1], 0.55 * fs)); s.append(_circle(["gvd"], pt[0], pt[1], 0.2 * fs))
    # les murs d'extrémité
    for r in geo.get("reprises_mur", []):
        pt = segs[0][0] if r["ou"] == "début" else segs[-1][1]
        p0, p1 = segs[0] if r["ou"] == "début" else segs[-1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        sg = -1 if r["ou"] == "début" else 1
        nx, ny = -uy, ux
        pts = [(pt[0] + nx * 1.6 * fs, pt[1] + ny * 1.6 * fs), (pt[0] - nx * 1.6 * fs, pt[1] - ny * 1.6 * fs),
               (pt[0] - nx * 1.6 * fs + sg * ux * 2.0 * fs, pt[1] - ny * 1.6 * fs + sg * uy * 2.0 * fs),
               (pt[0] + nx * 1.6 * fs + sg * ux * 2.0 * fs, pt[1] + ny * 1.6 * fs + sg * uy * 2.0 * fs)]
        s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.6 * fs))
    # les charges : flèches perpendiculaires à chaque tronçon, de l'intérieur vers l'extérieur (on pousse le garde-corps)
    ch = res["charges"]
    for (p0, p1) in segs:
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        nx, ny = -uy, ux
        for k in _pas(0.18 * L, 0.84 * L, max(L / 4.0, 2.5 * fs)):
            c = (p0[0] + ux * k, p0[1] + uy * k)
            _fleche(s, (c[0] - nx * 3.2 * fs, c[1] - ny * 3.2 * fs), (c[0] - nx * 0.9 * fs, c[1] - ny * 0.9 * fs), "", fs)
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    sides = ["B", "R", "T"]
    for i, ((p0, p1), ent) in enumerate(zip(segs, geo["entraxes"])):
        L = sum(ent)
        if i == 0:
            S.dim(dict(id="L1", side="B", a=p0[0], b=p1[0], o1=p0[1], o2=p1[1], sym="L₁", val=L, lvl=1))
        elif i == 1:
            S.dim(dict(id="L2", side="R", a=min(p0[1], p1[1]), b=max(p0[1], p1[1]), o1=p0[0], o2=p1[0], sym="L₂", val=L, lvl=1))
        else:
            S.dim(dict(id="L3", side="T", a=min(p0[0], p1[0]), b=max(p0[0], p1[0]), o1=p0[1], o2=p1[1], sym="L₃", val=L, lvl=1))
    # les noms : au-dessus du premier tronçon (en dessous pour un U, dont le troisième tronçon est en haut), à droite du deuxième
    cote_0 = "B" if len(segs) == 3 else "T"
    for m, pt in pos:
        nom = m["nom"] if systeme == "montants" else m["nom"].replace("M", "C")
        if m["troncon"] == 0 and not m["coin"]:
            S.name(cote_0, pt[0], nom)
        elif m["troncon"] == 0 or (m["troncon"] == 1 and len(segs) == 3):
            S.name("R", pt[1], nom)
        elif m["troncon"] == 1:
            S.name("R", pt[1], nom)
        else:
            S.name("T", pt[0], nom)
    _bloc_efforts(S, [("P" if ch["mode"] == "P" else "q_h") + " perpendiculaire à chaque tronçon · " + ("en L" if len(segs) == 2 else "en U" if len(segs) == 3 else "droit")],
                  (bb0["x1"] + bb0["x2"]) / 2, cote_0, 1)
    _etendre(S, s, fs)
    _nom(S, cote_0, (bb0["x1"] + bb0["x2"]) / 2, "Vue en plan")
    return S.finish(s, [], "Vue en plan du garde-corps : la ligne, les montants, les charges")


# ====================================================================== détail de la fixation
def detail(res: Dict, opt: Options) -> Dessin:
    """Le pied du montant (ou la console au mur) en coupe, à grande échelle."""
    geo = res["geometrie"]
    systeme = geo.get("systeme", "montants")
    fx = res.get("fixation") or {}
    princ = res.get("montant") if systeme == "montants" else res.get("console")
    sec = (princ or {}).get("section") or res.get("section_montant") or {}
    v = (princ or {}).get("verif") or {}
    typ = fx.get("type", "platine")
    pl = fx.get("platine") or {}
    soud = ((fx.get("soudure") or {}).get("cas") or {}).get("params") or {}
    a = float(soud.get("a") or 4.0)
    jon = soud.get("jonction", "angle")
    hp = float(sec.get("h") or sec.get("d") or 48.0)
    tw = float(sec.get("t") or sec.get("b") or 4.0) if sec.get("forme") != "plein_rect" else hp
    bp = float(pl.get("bp") or 150.0) if typ == "platine" else max(hp + 60.0, 120.0)
    tp = float(pl.get("tp") or 10.0) if typ == "platine" else 12.0
    ex = float(pl.get("ex") or 100.0)
    Lv = max(2.2 * hp, 120.0)              # longueur visible du montant
    bb0 = dict(x1=-bp / 2 - 10, x2=bp / 2 + 10, y1=-Lv - 20, y2=tp + 60)
    fs = _fs(bb0["x2"] - bb0["x1"]) * 0.95
    s: List[Dict] = []
    z = a * S2 if jon == "angle" else (float(soud.get("p") or 3.0) + float(soud.get("renfort") or 0.0))
    # le support : béton (chevilles) ou plat / mur (acier) hachuré sous la platine
    yb = tp if typ == "platine" else 0.0
    H_sup = 2.6 * fs + 20.0
    sup = _rect_rompu(-bp / 2 - 3 * fs, yb, bp + 6 * fs, H_sup, {"B", "L", "R"}, 0.5 * fs)
    s.append(_poly(["be" if (typ == "platine" and (fx.get("platine") or {}).get("As") is None) else "pp"], sup))
    s.append(_hach(["ht"], sup, 0.6 * fs))
    if typ == "platine":
        s.append(_rect(["pp"], -bp / 2, 0.0, bp, tp))
        # les ancrages : deux rangées à ±e_x/2, à travers la platine, dans le support
        d_anc = float(pl.get("d") or 12.0)
        for sg in (-1, 1):
            x = sg * ex / 2
            s.append(_rect(["tg"], x - d_anc / 2, -0.9 * fs, d_anc, tp + 0.9 * fs + H_sup * 0.8))
            s.append(_rect(["bo"], x - 0.9 * d_anc, -0.9 * fs, 1.8 * d_anc, 0.9 * fs))       # tête / écrou
    # le montant (coupé en haut) : deux parois pour un creux, pleine pour un plat
    if sec.get("forme") in ("creux_rond", "creux_rect"):
        for sg in (-1, 1):
            x0 = sg * hp / 2 - (tw if sg > 0 else 0)
            s.append(_poly(["pp"], _rect_rompu(x0, -Lv, tw, Lv, {"T"}, 0.45 * fs)))
            s.append(_hach(["ht"], [(x0, -Lv), (x0 + tw, -Lv), (x0 + tw, 0), (x0, 0)], 0.4 * fs))
        s.append(_path(["hd"], [[(-hp / 2 + tw, -Lv), (-hp / 2 + tw, 0)], [(hp / 2 - tw, -Lv), (hp / 2 - tw, 0)]]) if False else _path(["ax"], [[(0, -Lv - 0.5 * fs), (0, tp + 0.5 * fs)]]))
    else:
        pts = _rect_rompu(-hp / 2, -Lv, hp, Lv, {"T"}, 0.45 * fs)
        s.append(_poly(["pp"], pts)); s.append(_hach(["ht"], pts, 0.5 * fs))
    # les cordons : triangles a·√2 de chaque côté (ou chanfrein)
    for sg in (-1, 1):
        x0 = sg * hp / 2
        if jon == "pleine":
            tri = [(x0, 0.0), (x0 + sg * 0.5 * tw, 0.0), (x0, -0.6 * tw)]
        else:
            tri = [(x0, 0.0), (x0 + sg * z, 0.0), (x0, -z)]
        s.append(_poly(["wb"], tri)); s.append(_hach(["ht", "htw"], tri, 0.25 * fs))
    # les efforts en tête de la partie visible : V (horizontal), N (vertical), M (arc)
    items = []
    if v:
        g = v.get("gamma", 1.0)
        P = float(v.get("P1", 0.0)) * g
        N = float(v.get("N", 0.0)) * g if systeme == "montants" else -float(v.get("Rh", 0.0)) * g
        M = math.hypot(float(v.get("M1", 0.0)), float(v.get("M2", 0.0))) * g
        if systeme == "montants":
            _fleche(s, (-hp / 2 - 4.5 * fs, -Lv + 0.9 * fs), (-hp / 2 - 0.3 * fs, -Lv + 0.9 * fs), "V", fs, cote=-1)
            if N > 0:
                _fleche(s, (0.0, -Lv - 3.2 * fs), (0.0, -Lv - 0.4 * fs), "N", fs, etiq=(1.6 * fs, -Lv - 3.0 * fs))
        else:
            _fleche(s, (0.0, -Lv - 3.2 * fs), (0.0, -Lv - 0.4 * fs), "V", fs, etiq=(1.6 * fs, -Lv - 3.0 * fs))
            _fleche(s, (-hp / 2 - 0.3 * fs, -Lv + 0.9 * fs), (-hp / 2 - 4.5 * fs, -Lv + 0.9 * fs), "N", fs, cote=-1)
        items.append("V = " + _kn(P) + " · N = " + _kn(abs(N)) + (" (compression)" if N > 0 else " (traction)" if N < 0 else "") + " · M = " + F(M / 1e6, 3) + " kN·m")
    S = Feuille(_bb(bb0, s, fs), fs, opt)
    S.dim(dict(id="hp", side="T", a=-hp / 2, b=hp / 2, o1=-Lv, o2=-Lv, sym="h_p", val=hp, lvl=1))
    if typ == "platine":
        S.dim(dict(id="bp", side="B", a=-bp / 2, b=bp / 2, o1=yb + H_sup, o2=yb + H_sup, sym="b_p", val=bp, lvl=1))
        S.dim(dict(id="ex", side="B", a=-ex / 2, b=ex / 2, o1=yb + H_sup, o2=yb + H_sup, sym="e_x", val=ex, lvl=1))
        S.dim(dict(id="tp", side="R", a=0.0, b=tp, o1=bp / 2, o2=bp / 2, sym="t_p", val=tp, lvl=1, out="hi"))
        if pl.get("m", 0) > 0:
            S.dim(dict(id="m", side="T", a=hp / 2 + 0.8 * a * S2, b=ex / 2, o1=-z, o2=-0.9 * fs, sym="m", val=pl["m"], lvl=2, calc=True))
    S.tag(hp / 2 + z + 0.6 * fs, -z - 0.3 * fs, dict(id="a", sym="a" if jon == "angle" else "s", val=a if jon == "angle" else z, lvl=1))
    _bloc_efforts(S, items, 0.0, "B", 1)
    _etendre(S, s, fs)
    nom = ("Pied du montant" if systeme == "montants" else "Console au mur") + " · " + (sec.get("nom") or "") + (" sur platine" if typ == "platine" else " soudé sur plat")
    _nom(S, "T", 0.0, nom)
    return S.finish(s, [], "Détail de la fixation : montant, cordons, platine, ancrages")


# ====================================================================== section
def section(res: Dict, opt: Options) -> Dessin:
    """La section de l'élément principal, à l'échelle, avec la direction de la charge et ses cotes."""
    sec = res.get("section_montant") or {}
    if not sec:
        return Dessin((0, 0, 100, 60), [], [], [], "section", 6.0)
    rings = BI.contours(sec)
    pts_all = [(float(p[0]), float(p[1])) for ring in rings for p in ring]
    xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    bb0 = dict(x1=min(xs) - 0.4 * w - 10, x2=max(xs) + 0.4 * w + 10, y1=-max(ys) - 0.4 * h - 10, y2=-min(ys) + 0.4 * h + 10)
    fs = _fs(bb0["x2"] - bb0["x1"])
    s: List[Dict] = []
    ext = [(float(p[0]), -float(p[1])) for p in rings[0]]
    s.append(_poly(["pp"], ext))
    if len(rings) > 1:
        for ring in rings[1:]:
            s.append(_poly(["bo"], [(float(p[0]), -float(p[1])) for p in ring]))
        s.append(_path(["ht"], _hachures_anneaux([[(float(p[0]), -float(p[1])) for p in r] for r in rings], 0.45 * fs)))
    else:
        s.append(_hach(["ht"], ext, 0.45 * fs))
    s.append(_path(["ax"], [[(min(xs) - 0.3 * w, 0), (max(xs) + 0.3 * w, 0)], [(0, -max(ys) - 0.3 * h), (0, -min(ys) + 0.3 * h)]]))
    # la charge principale : dans le sens de h (vertical sur le dessin), vers le haut
    _fleche(s, (0.0, -min(ys) + 0.3 * h + 2.6 * fs), (0.0, -min(ys) + 0.3 * h + 0.3 * fs), "P₁", fs, cote=+1)
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
    return S.finish(s, [], "Section de l'élément principal")


# ====================================================================== l'entrée
VUES = (("elevation", "Élévation développée", elevation), ("plan", "Vue en plan", plan), ("detail", "Détail de la fixation", detail),
        ("section", "Section", section))


def vues(res: Dict, opt: Optional[Options] = None) -> List[Tuple[str, str, Dessin]]:
    """Les vues d'un résultat : [(clé, titre, Dessin)] — aucune sur un refus de géométrie."""
    if not res.get("geometrie") or not res["geometrie"].get("montants"):
        return []
    opt = opt or options_schemas()
    out = []
    for cle, titre, fn in VUES:
        if cle == "detail" and (res.get("fixation") or {}).get("type") == "scellee":
            continue
        out.append((cle, titre, fn(res, opt)))
    return out


def svg(dessin: Dessin, identifiant: str = "gc") -> str:
    return vers_svg(dessin, PALETTE, "100%", identifiant)

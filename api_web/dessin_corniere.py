# -*- coding: utf-8 -*-
"""Les DESSINS de la cornière ancrée, en dessin technique, sur le kit de dessin des assemblages (feuille de cotation,
primitives, hachures, lignes de rupture, rendu SVG — le même que le garde-corps) et les briques des schémas de soudure
(flèches d'effort, étiquettes à halo, rangées de texte).

Quatre vues, scènes en mm À L'ÉCHELLE (y vers le bas) :

- COUPE (le schéma statique) : le mur coupé, la cornière coupée à l'échelle (congés du catalogue), sa fixation, la
  charge V à l'excentricité e, les cotes A, B, t, e ;
- ÉLÉVATION (entre fixations) : la cornière vue de face sur trois entraxes, les fixations à l'entraxe s, la charge
  répartie, le panneau de largeur s qui sert à la vérification longitudinale, la chaîne des entraxes ;
- EFFORTS (le modèle lame) : la cornière coupée, les diagrammes de l'effort tranchant et du moment sur le bras de
  levier e, les valeurs ELS et ELU par mètre courant ;
- SECTION : la cornière seule, grande, avec ses cotes (A, B, t, r₁, r₂), ses axes par le centre de gravité, sa masse.

Chaque vue est construite pour une taille de texte, puis RECONSTRUITE (``_ajuster``) jusqu'à ce que ce texte se lise à
``TEXTE_PX`` pixels dans la toile de l'établi (``TOILE``, mesurée par site/web/verifier_site.js) : une petite cornière
(L 20×20×3) ne disparaît pas sous ses cotes, une grande n'a pas un texte minuscule.

Rien n'est recalculé : les grandeurs viennent du moteur (``acier.corniere.moteur.verifier``) ; ce fichier les DESSINE.
La position de la fixation sur l'aile et son diamètre sont une convention de dessin : l'écran Streamlit ne les définit
pas et ne vérifie pas les ancrages (le dessin le dit).
"""
from typing import Callable, Dict, List, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # le kit de dessin partagé
    Feuille, PALETTE, _circle, _group, _hachures, _path, _poly, _rect, _rect_rompu, _text, vers_svg)
from acier.profiles import geometrie as GEO
from acier.soudures.schemas import _bb, _bloc_efforts, _etendre, _fleche, _label, _nom, _tw, options_schemas

from . import commun as C

TOILE = (626.0, 682.0)          # la toile de l'établi au format PC (px), mesurée par le contrôle navigateur
TEXTE_PX = 13.0                 # la taille du texte des dessins à l'écran (px)


def _n(x: float, nd: int) -> str:
    """Un nombre du dessin, à la française (virgule, espace fine des milliers)."""
    return C.fr(x, nd)


def _g(x: float) -> str:
    """Une cote : sans décimale inutile (100, 6,5, 22,2)."""
    return C.nb(x, 1)


def _contour(sec: Dict) -> List[Tuple[float, float]]:
    """Le contour de la cornière (congés r₁, r₂ du catalogue ; vifs en saisie libre), talon à l'origine, aile A
    verticale vers le haut (y négatif), aile B horizontale vers la droite."""
    pts = GEO.contour_L(float(sec["A"]), float(sec["B"]), float(sec["t"]), float(sec["r1"] or 0.0), float(sec["r2"] or 0.0))
    return [(float(p[0]), -float(p[1])) for p in pts]


def _hach(cls, pts, pas):
    return _path(cls, _hachures(pts, pas))


def _cdg(sec: Dict) -> Tuple[float, float]:
    """Le centre de gravité (depuis le talon) : celui du catalogue, ou celui des deux rectangles d'une saisie libre
    (géométrie du dessin, sans effet sur le calcul)."""
    if sec.get("yc") is not None:
        return float(sec["yc"]), float(sec["zc"])
    A, B, t = float(sec["A"]), float(sec["B"]), float(sec["t"])
    a1, a2 = B * t, (A - t) * t
    return (a1 * B / 2 + a2 * t / 2) / (a1 + a2), (a1 * t / 2 + a2 * (t + (A - t) / 2)) / (a1 + a2)


def _diametre_fixation(sec: Dict) -> float:
    """Le diamètre DESSINÉ de la fixation (convention de dessin : l'écran ne le définit pas) : M6 à M20 selon l'aile,
    sans que la rondelle (2 d) déborde la partie droite de l'aile A."""
    A, t, r1 = float(sec["A"]), float(sec["t"]), float(sec["r1"] or 0.0)
    d = min(max(round(min(A, 160.0) / 9.0 / 2.0) * 2.0, 6.0), 20.0)
    return float(min(d, 0.4 * max(A - t - r1, 2.0)))


def _y_fixation(sec: Dict) -> float:
    """La hauteur DESSINÉE de la fixation : au milieu de la partie droite de l'aile A (convention de dessin)."""
    return -(float(sec["t"]) + float(sec["r1"] or 0.0) + float(sec["A"])) / 2


def _fixation(s, x_face: float, y: float, d: float, prof: float, t: float, fs: float):
    """Une fixation dans le mur (face en ``x_face``, mur à gauche) à travers l'aile A : la tige cachée dans le béton
    (traits interrompus), la tige dans l'aile, la rondelle et l'écrou côté libre."""
    s.append(_path(["hd"], [[(x_face, y - d / 2), (x_face - prof, y - d / 2)], [(x_face, y + d / 2), (x_face - prof, y + d / 2)],
                            [(x_face - prof, y - d / 2), (x_face - prof, y + d / 2)]]))
    s.append(_rect(["tg"], x_face, y - d / 2, t + 1.25 * d, d))
    s.append(_rect(["bo"], x_face + t, y - 1.0 * d, 0.18 * d, 2.0 * d))
    s.append(_rect(["bo"], x_face + t + 0.18 * d, y - 0.85 * d, 0.8 * d, 1.7 * d))
    s.append(_path(["ax"], [[(x_face - prof - 0.6 * fs, y), (x_face + t + 1.25 * d + 0.6 * fs, y)]]))


def _mur(s, x_face: float, y1: float, y2: float, ep: float, fs: float):
    """Le mur coupé (béton hachuré), rompu en haut, en bas et à l'arrière : il continue."""
    pts = _rect_rompu(x_face - ep, y1, ep, y2 - y1, {"T", "B", "L"}, 0.5 * fs)
    s.append(_poly(["be"], pts))
    s.append(_hach(["ht"], pts, 0.6 * fs))


def _renvoi(s, x0: float, y0: float, x1: float, y1: float, txt: str, fs: float):
    """Un renvoi : ligne de rappel depuis la pièce, étiquette à halo au bout (à droite ou à gauche du point)."""
    s.append(_path(["ext"], [[(x0, y0), (x1, y1)]]))
    w = _tw(txt, fs)
    _label(s, x1 + (w / 2 + 0.45 * fs) * (1 if x1 >= x0 else -1), y1 + 0.34 * fs, txt, fs, cls=("dl",))


def _chaine(S, xs: List[float], o: float):
    """Une chaîne de cotes horizontale sous la vue, sur UNE rangée : attaches, ligne continue, repères, valeurs."""
    fs = S.fs
    t = 0.34 * fs
    r = S.slot("B", xs[0], xs[-1])
    p = S.pos("B", r)
    e = p + 0.45 * fs
    S.ex["x1"] = min(S.ex["x1"], xs[0]); S.ex["x2"] = max(S.ex["x2"], xs[-1])
    ext = _path(["ext"], [[(x, o), (x, e)] for x in xs], hint="V")
    dln = _path(["dln"], [[(xs[0], p), (xs[-1], p)]] + [[(x - t, p + t), (x + t, p - t)] for x in xs], hint="H")
    labs = [S.label((a + b) / 2, p - 0.42 * fs, False, f"s = {_g(b - a)}", dict(id=f"s{k}", sym="", lvl=1))
            for k, (a, b) in enumerate(zip(xs[:-1], xs[1:]))]
    S.g.append(_group(["dm"], [ext, dln] + labs))


def _charge(c: Dict) -> str:
    return f"V = {C.nb(c['V'])} kN/m" if c["charge"] == "lineaire" else f"V = {C.nb(c['V'])} kN sur L = {_g(c['L'])} m"


def _ajuster(construire: Callable[[float], object], fs0: float) -> object:
    """La vue reconstruite jusqu'à ce que son texte se lise à TEXTE_PX dans la TOILE (trois passes suffisent : la boîte
    de vue ne dépend qu'en partie de la taille du texte)."""
    fs, d = fs0, None
    for _ in range(4):
        d = construire(fs)
        _, _, w, h = d.viewbox
        px = fs * min(TOILE[0] / w, TOILE[1] / h)
        if abs(px - TEXTE_PX) < 0.25:
            break
        fs *= TEXTE_PX / px
    return d


# ====================================================================== la coupe : le schéma statique
def coupe(res: Dict) -> object:
    sec, c, g = res["section"], res["corniere"], res["grandeurs"]
    A, B, t = float(sec["A"]), float(sec["B"]), float(sec["t"])
    e_mm = c["e"] * 10.0
    larg = max(B, e_mm)
    ep_mur = max(0.75 * A, 25.0 + 0.3 * A)

    def construire(fs):
        s: List[Dict] = []
        marge = max(0.35 * A, 2.2 * fs)
        _mur(s, 0.0, -A - marge, marge, ep_mur, fs)
        pts = _contour(sec)
        s.append(_poly(["pp"], pts))
        s.append(_hach(["ht"], pts, max(1.2, 0.22 * fs)))
        d = _diametre_fixation(sec)
        y_fix = _y_fixation(sec)
        _fixation(s, 0.0, y_fix, d, min(0.8 * ep_mur, 8 * d), t, fs)
        # la charge V à l'excentricité e (comptée depuis le dos de la cornière, contre le mur), sur l'aile B
        haut_v = -t - 4.2 * fs
        if c["V"] > 0:
            _fleche(s, (e_mm, haut_v), (e_mm, -t - 0.15 * fs), "V", fs, cote=+1)
        s.append(_path(["ax"], [[(e_mm, -t - 0.1 * fs), (e_mm, 1.3 * fs)]]))
        # le renvoi de la fixation : dans le creux de la cornière s'il y a la place, sinon au-dessus d'elle
        y_lab = -A + 1.2 * fs if -A + 1.2 * fs < haut_v - 1.6 * fs else -A - 1.6 * fs
        _renvoi(s, t + 1.0 * d, y_fix - 0.85 * d, t + 1.0 * d + 1.6 * fs, y_lab, "fixation non vérifiée", 0.9 * fs)
        bb0 = dict(x1=-ep_mur, x2=larg + 0.5 * fs, y1=-A - marge, y2=marge)
        S = Feuille(_bb(bb0, s, fs), fs, options_schemas(2))
        S.dim(dict(id="A", side="R", a=-A, b=0.0, o1=t, o2=B, sym="A", val=A, lvl=1))
        S.dim(dict(id="B", side="B", a=0.0, b=B, o1=0.0, o2=0.0, sym="B", val=B, lvl=1))
        if e_mm > 0:
            S.dim(dict(id="e", side="B", a=0.0, b=e_mm, o1=0.0, o2=1.3 * fs, sym="e", val=e_mm, lvl=1, out="hi"))
        S.dim(dict(id="t", side="T", a=0.0, b=t, o1=-A, o2=-A, sym="t", val=t, lvl=1, out="hi"))
        _bloc_efforts(S, [_charge(c), f"e = {C.nb(c['e'])} cm", f"M = V·e = {_n(g['M_ELS'], 3)} kN·m/m (ELS)"], 0.5 * (larg - ep_mur), "B", 3)
        _etendre(S, s, fs)
        _nom(S, "T", 0.5 * (larg - ep_mur), f"{sec['designation']} ancrée au mur")
        return S.finish(s, [], f"Coupe : mur, cornière {sec['designation']} et sa fixation, charge V à l'excentricité e")
    return _ajuster(construire, (ep_mur + larg + A) / 40.0)


# ====================================================================== l'élévation : entre fixations
def elevation(res: Dict) -> object:
    sec, c, g = res["section"], res["corniere"], res["grandeurs"]
    A, t = float(sec["A"]), float(sec["t"])
    s_mm = c["s"] * 10.0
    x0, x1 = -0.5 * s_mm, 2.5 * s_mm                    # trois entraxes ; la cornière continue au-delà (rupture)

    def construire(fs):
        s: List[Dict] = []
        marge = max(0.45 * A, 1.6 * fs)
        # le mur, vu de face : un fond clair, rompu
        fond = _rect_rompu(x0 - 1.2 * fs, -A - marge, x1 - x0 + 2.4 * fs, A + 2 * marge, {"L", "R", "T", "B"}, 0.5 * fs)
        s.append(_poly(["be"], fond))
        # la cornière vue de face : l'aile A contre le mur, la tranche de l'aile B en bas (devant)
        s.append(_poly(["ps"], _rect_rompu(x0, -A, x1 - x0, A, {"L", "R"}, min(0.35 * fs, 0.3 * A))))
        s.append(_poly(["pp"], _rect_rompu(x0, -t, x1 - x0, t, {"L", "R"}, min(0.35 * fs, 0.45 * t))))
        # le panneau de largeur s, centré sur la fixation du milieu (celui de la vérification longitudinale)
        dep = 0.6 * fs
        s.append(_rect(["ae"], 0.5 * s_mm, -A - dep, s_mm, A + 2 * dep))
        s.append(_path(["ael"], [[(0.5 * s_mm, -A - dep), (0.5 * s_mm, dep)], [(1.5 * s_mm, -A - dep), (1.5 * s_mm, dep)]]))
        d = _diametre_fixation(sec)
        y_fix = _y_fixation(sec)
        for k in range(3):
            x = k * s_mm
            s.append(_path(["ax"], [[(x, -A - 0.9 * fs), (x, dep + 0.4 * fs)]]))
            s.append(_circle(["bo"], x, y_fix, 1.0 * d))
            s.append(_circle(["tg"], x, y_fix, 0.5 * d))
        # la charge répartie, au-dessus de la cornière
        xa, xb = x0 + 0.04 * (x1 - x0), x1 - 0.04 * (x1 - x0)
        n = max(6, min(14, int((xb - xa) / (2.2 * fs))))
        if c["V"] > 0:
            for k in range(n + 1):
                x = xa + (xb - xa) * k / n
                _fleche(s, (x, -A - dep - 2.6 * fs), (x, -A - dep - 0.1 * fs), "", fs)
            s.append(_path(["fz"], [[(xa, -A - dep - 2.6 * fs), (xb, -A - dep - 2.6 * fs)]]))
        _label(s, 0.5 * (x0 + x1), -A - dep - 3.5 * fs, _charge(c), fs)
        bb0 = dict(x1=x0 - 1.2 * fs, x2=x1 + 1.2 * fs, y1=-A - marge, y2=marge)
        S = Feuille(_bb(bb0, s, fs), fs, options_schemas(2))
        _chaine(S, [0.0, s_mm, 2 * s_mm], dep + 0.4 * fs)
        S.dim(dict(id="A", side="L", a=-A, b=0.0, o1=x0, o2=x0, sym="A", val=A, lvl=1))
        _bloc_efforts(S, [f"V loc = V·s = {_n(g['Vloc_ELS'], 2)} kN (ELS) · {_n(g['Vloc_ELU'], 2)} kN (ELU)",
                          f"M long = V loc·e = {_n(g['Mlong_ELU'], 3)} kN·m (ELU)"], s_mm, "B", 2)
        _etendre(S, s, fs)
        _nom(S, "T", s_mm, f"{sec['designation']} vue de face · panneau de largeur s autour d'une fixation")
        return S.finish(s, [], f"Élévation : cornière {sec['designation']} vue de face, fixations à l'entraxe s, panneau de largeur s")
    return _ajuster(construire, (x1 - x0) / 45.0)


# ====================================================================== les efforts : le modèle lame
def efforts(res: Dict) -> object:
    sec, c, g = res["section"], res["corniere"], res["grandeurs"]
    A, B, t = float(sec["A"]), float(sec["B"]), float(sec["t"])
    e_mm = c["e"] * 10.0
    larg = max(B, e_mm)
    ep_mur = max(0.35 * A, 12.0 + 0.2 * A)

    def construire(fs):
        s: List[Dict] = []
        _mur(s, 0.0, -A - 1.0 * fs, 1.0 * fs, ep_mur, fs)
        pts = _contour(sec)
        s.append(_poly(["pp"], pts))
        s.append(_hach(["ht"], pts, max(1.2, 0.22 * fs)))
        if g["V_ELU"] > 0:
            _fleche(s, (e_mm, -t - 3.6 * fs), (e_mm, -t - 0.15 * fs), "V", fs, cote=+1)
        # les diagrammes, sous la cornière, sur le bras de levier e (compté depuis la face du mur), à l'échelle des
        # longueurs ; leur hauteur est conventionnelle
        hV = max(0.22 * A, 2.4 * fs)
        hM = max(0.4 * A, 3.6 * fs)
        yV = 2.6 * fs
        yM = yV + hV + 2.4 * fs
        x_lab = max(e_mm, 0.0) + 1.2 * fs
        for y in (yV, yM):
            s.append(_path(["axe"], [[(-0.4 * fs, y), (max(e_mm, 0.0) + 0.4 * fs, y)]]))
        s.append(_path(["ext"], [[(0.0, yV - 0.6 * fs), (0.0, yM + hM + 0.4 * fs)]]))
        if e_mm > 0:
            s.append(_path(["ext"], [[(e_mm, -t), (e_mm, yM + 0.4 * fs)]]))
        if e_mm > 0 and g["V_ELU"] > 0:
            s.append(_rect(["d2"], 0.0, yV, e_mm, hV))
            s.append(_poly(["ct"], [(0.0, yM), (e_mm, yM), (0.0, yM + hM)]))
        s.append(_text(["tx"], -1.0 * fs, yV + 0.5 * hV + 0.34 * fs, "V", fs, "end"))
        s.append(_text(["tx"], -1.0 * fs, yM + 0.5 * hM + 0.34 * fs, "M", fs, "end"))
        for y, txt in ((yV + 0.5 * hV - 0.35 * fs, f"{_n(g['V_ELS'], 2)} kN/m (ELS)"), (yV + 0.5 * hV + 1.05 * fs, f"{_n(g['V_ELU'], 2)} kN/m (ELU)"),
                       (yM + 0.6 * fs, f"{_n(g['M_ELS'], 3)} kN·m/m (ELS)"), (yM + 2.0 * fs, f"{_n(g['M_ELU'], 3)} kN·m/m (ELU)")):
            s.append(_text(["tx2"], x_lab, y, txt, 0.92 * fs, "start"))
        bb0 = dict(x1=-ep_mur, x2=larg + 0.5 * fs, y1=-A - 1.0 * fs, y2=yM + hM)
        S = Feuille(_bb(bb0, s, fs), fs, options_schemas(2))
        if e_mm > 0:
            S.dim(dict(id="e", side="T", a=0.0, b=e_mm, o1=-A, o2=-t - 3.6 * fs, sym="e", val=e_mm, lvl=1, out="hi"))
        _bloc_efforts(S, [f"σ = M/W = {_n(g['sigma_ELU'], 2)} N/mm²", f"τ = V/Az = {_n(g['tau_ELU'], 2)} N/mm²",
                          f"σ eq = {_n(g['sigma_eq_ELU'], 2)} N/mm² (ELU)"], 0.5 * (larg - ep_mur), "B", 3)
        _nom(S, "B", 0.5 * (larg - ep_mur), f"modèle lame : b = 1 000 mm × t = {_g(t)} mm · W = b·t²/6 = {_n(g['W_trans'], 0)} mm³", cls="tx2")
        _etendre(S, s, fs)
        _nom(S, "T", 0.5 * (larg - ep_mur), "Effort tranchant et moment " + (
            "par mètre de cornière" if c["charge"] == "lineaire" else f"par mètre (V réparti sur L = {_g(c['L'])} m)"))
        return S.finish(s, [], "Efforts : effort tranchant et moment sur le bras de levier e, par mètre courant (modèle lame)")
    return _ajuster(construire, (ep_mur + larg + 1.2 * A) / 40.0)


# ====================================================================== la section
def section(res: Dict) -> object:
    sec = res["section"]
    A, B, t = float(sec["A"]), float(sec["B"]), float(sec["t"])
    r1, r2 = float(sec["r1"] or 0.0), float(sec["r2"] or 0.0)
    yc, zc = _cdg(sec)

    def construire(fs):
        pts = _contour(sec)
        s: List[Dict] = [_poly(["pp"], pts), _hach(["ht"], pts, max(1.2, 0.22 * fs))]
        s.append(_path(["ax"], [[(-0.15 * B - 0.5 * fs, -zc), (1.12 * B + 0.5 * fs, -zc)], [(yc, -1.12 * A - 0.5 * fs), (yc, 0.15 * A + 0.5 * fs)]]))
        s.append(_text(["tx2"], 1.12 * B + 0.9 * fs, -zc + 0.34 * fs, "y", fs, "start"))
        s.append(_text(["tx2"], yc + 0.5 * fs, -1.12 * A - 0.4 * fs, "z", fs, "start"))
        s.append(_circle(["gvd"], yc, -zc, 0.22 * fs))
        lab = 0.9 * fs
        if r1:
            k = 1 - 0.7071
            _renvoi(s, t + r1 * k, -t - r1 * k, max(0.55 * B, t + r1 + 2.5 * fs), min(-0.45 * A, -zc - 2.4 * fs), f"r₁ = {_g(r1)}", lab)
        if r2:
            _renvoi(s, B - 0.3 * r2, -t + 0.3 * r2, B + 1.0 * fs, 1.6 * fs, f"r₂ = {_g(r2)}", lab)
        bb0 = dict(x1=-0.15 * B, x2=1.12 * B, y1=-1.12 * A, y2=0.15 * A)
        S = Feuille(_bb(bb0, s, fs), fs, options_schemas(2))
        S.dim(dict(id="A", side="L", a=-A, b=0.0, o1=0.0, o2=0.0, sym="A", val=A, lvl=1))
        S.dim(dict(id="B", side="B", a=0.0, b=B, o1=0.0, o2=0.0, sym="B", val=B, lvl=1))
        S.dim(dict(id="t", side="T", a=0.0, b=t, o1=-A, o2=-A, sym="t", val=t, lvl=1, out="hi"))
        S.dim(dict(id="yc", side="B", a=0.0, b=yc, o1=0.0, o2=-zc, sym="yG", val=yc, lvl=2, out="hi"))
        S.dim(dict(id="zc", side="R", a=-zc, b=0.0, o1=yc, o2=B, sym="zG", val=zc, lvl=2, out="lo"))
        _etendre(S, s, fs)
        masse = f" · {_n(sec['masse'], 2)} kg/m" if sec.get("masse") else " · saisie libre"
        _nom(S, "B", 0.5 * B, sec["designation"] + masse)
        if sec.get("aire"):
            _nom(S, "B", 0.5 * B, f"aire {_n(sec['aire'], 2)} cm² · {sec['norme']}", cls="tx2")
        return S.finish(s, [], f"Section de la cornière {sec['designation']} à l'échelle : cotes, axes par le centre de gravité")
    return _ajuster(construire, 1.6 * max(A, B) / 30.0)


# ====================================================================== l'entrée
VUES = (("coupe", "Coupe · schéma statique", "Coupe", coupe), ("elevation", "Élévation · entre fixations", "Élévation", elevation),
        ("efforts", "Efforts · modèle lame", "Efforts", efforts), ("section", "Section de la cornière", "Section", section))


TEXTE_MIN_PX = 6.0              # sous cette taille, le dessin n'est plus lisible : on le dit au lieu de le montrer


def _secours(ident: str, titre: str) -> str:
    """Le dessin qu'on ne sait pas faire (proportions extrêmes : e ou s de plusieurs dizaines de mètres face à une
    aile de quelques centimètres) : un cadre qui le DIT, plutôt qu'un dessin illisible ou une erreur. Le calcul, lui,
    est rendu normalement."""
    return (f'<svg id="{ident}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 220" width="100%" role="img" '
            f'aria-label="{C.esc(titre)} : dessin impossible à cette échelle">'
            '<rect x="20" y="20" width="560" height="180" rx="12" fill="none" stroke="#7F8C9A" stroke-width="1.2" stroke-dasharray="6 4"/>'
            '<text x="300" y="104" text-anchor="middle" font-size="20" font-weight="600" fill="#33415C">Dessin impossible à cette échelle</text>'
            '<text x="300" y="138" text-anchor="middle" font-size="15" fill="#5B6878">les dimensions saisies sont trop disproportionnées '
            '— le calcul reste valable</text></svg>')


def vues(res: Dict, ident: str = "co") -> List[Dict]:
    """Les vues d'un résultat du moteur : [{cle, titre, court, svg}] (le SVG autonome, style embarqué). Une vue qu'on
    ne sait pas dessiner (proportions absurdes) est remplacée par un cadre qui le dit : jamais d'erreur, jamais de
    texte illisible."""
    out = []
    for cle, titre, court, fn in VUES:
        try:
            svg = vers_svg(fn(res), PALETTE, "100%", f"{ident}_{cle}")
            if not TEXTE_MIN_PX <= taille_texte(svg) < 1e6:
                raise ValueError("texte illisible")
        except (ArithmeticError, ValueError, ZeroDivisionError):
            svg = _secours(f"{ident}_{cle}", titre)
        out.append({"cle": cle, "titre": titre, "court": court, "svg": svg})
    return out


def taille_texte(svg: str, fs: float = None) -> float:
    """La taille à l'écran (px) d'un texte de ``fs`` unités (par défaut : la plus fréquente du dessin) dans la TOILE —
    pour les tests de lisibilité."""
    import re
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0].split()
    w, h = float(vb[2]), float(vb[3])
    if fs is None:
        tailles = re.findall(r'font-size="([0-9.]+)"', svg) or re.findall(r"font-size:([0-9.]+)", svg)
        fs = max(set(tailles), key=tailles.count) if tailles else "0"
        fs = float(fs)
    return fs * min(TOILE[0] / w, TOILE[1] / h)


def aire_dessin(svg: str) -> Tuple[float, float]:
    """La boîte de vue d'un SVG (largeur, hauteur) — pour les tests de proportion."""
    vb = svg.split('viewBox="', 1)[1].split('"', 1)[0].split()
    return float(vb[2]), float(vb[3])

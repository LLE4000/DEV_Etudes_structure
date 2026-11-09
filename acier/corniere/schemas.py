# -*- coding: utf-8 -*-
"""Cornière d'appui — les QUATRE VUES partagées par l'écran et la note (``docs/acier/CORNIERE_DECISIONS.md`` § 4 ;
modèle visuel : figures a, b, c de ``docs/acier/corniere_excel/ANALYSE_CORNIERE_EXCEL.md``), construites sur le kit de
dessin des assemblages (``acier.assemblages.poutre_poutre.doubles_cornieres.schemas`` : feuille de cotation, primitives,
hachures, lignes de rupture, rendu SVG) et les briques des schémas de soudure (flèches d'effort, étiquettes à halo,
étendue réelle du corps).

Chaque vue est une fonction ``(res, fs=None, interactif=True) -> Dessin`` qui ne lit que la SORTIE de
``acier.corniere.moteur.verifier`` : rien n'est recalculé ici — les bras de levier sont des cotes mesurées sur le
dessin (leur valeur est leur longueur), les efforts et les taux sont ceux du moteur. ``fs`` est la taille du texte
dans l'unité de la scène (mm) ; ``interactif`` pose ``data-sel`` sur chaque pièce qu'une donnée pilote.

- ``coupe`` — LE schéma statique coté, en mm à l'échelle (y vers le bas) : le mur hachuré, la cornière (congé r₁,
  bouts r₂ du catalogue), la charge R à X en trait plein et ses deux positions X ± dx en tirets, l'ancrage (axe, Ø,
  écrou) et ses positions Y ± dy en tirets fins, la résultante de compression C (pivot y_c en A, bout libre en B), les
  bras de levier DU MODE (historique : Xe, Xe+r, Ye, Ye+r, YH ; équilibre : X − uV, Y − yc ou ya − c), les sections
  1 à 4 en orange avec leur pastille, les cotes larg, haut, e, r, X, Y, les valeurs R, H, C. Configuration A (aile
  horizontale en bas) et B (en haut : l'aile verticale descend, le contact est au bout libre).
- ``moments`` — HORS ÉCHELLE : les deux ailes développées l'une au-dessus de l'autre (abscisse s depuis le dos de
  l'autre aile) ; les trois courbes de moment (une par position de la charge : extrêmes en trait fin, nominale en
  trait moyen) et l'ENVELOPPE en gras ; la tolérance ± dy de l'ancrage en bande grise ; N et V par tronçon ; les
  sections 1 à 4 avec la valeur de l'enveloppe ; congé et bloc d'angle grisés. En mode historique, les droites du
  classeur telles que le moteur les rend.
- ``contraintes`` — HORS ÉCHELLE : le long des mêmes ailes, la bande d'épaisseur t(s) (le congé), σéq de l'enveloppe
  et le taux avec la ligne des 100 %, le maximum ◉ et sa position (il n'est pas à la section : c'est ce que cette vue
  montre), les sections repérées.
- ``elevation`` — à l'échelle : la tranche L vue de face (les n ancrages à L/n, le premier à L/(2n), N et V ELS de
  l'enveloppe par ancrage) et l'aile horizontale RABATTUE (vue de dessus) : la charge de largeur a (ou L) centrée et sa
  diffusion à 45° jusqu'à la section 1 — le trapèze qui donne L_eff, coté — la cote L et la chaîne des entraxes.

Contrat de sélection (établi compact) : chaque pièce qu'une donnée pilote est un groupe ``<g class="pc"
data-sel="<clé>">`` sous la clé de sa ligne dans les données — X, dx, Y, dy, R, L, a, n, d, profil, yc, uV,
jeu_ancrage (``CLES_SEL``). Une vue qu'on ne sait pas dessiner (proportions absurdes) lève ValueError ; ``secours``
rend le cadre « Dessin impossible à cette échelle ». ``ajuster`` reconstruit une vue jusqu'à ce que son texte se lise
à la taille voulue dans une toile donnée (l'écran : 13 px dans 626 × 682 ; la note choisit la sienne).

Bibliothèque standard ; aucun Streamlit, aucun moteur (le résultat est reçu).
"""
import math
from typing import Callable, Dict, List, Optional, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # le kit de dessin partagé
    Dessin, Feuille, PALETTE, _arc, _circle, _group, _hachures, _path, _poly, _rect, _rect_rompu, _text, vers_svg)
from acier.formats import F, f0
from acier.soudures.schemas import _bb, _etendre, _extent, _fleche, _label, _tw, options_schemas   # briques partagées

from .entree import LIBELLES_CONFIGURATION, LIBELLES_MODE
from .statique import EQUILIBRE, HISTORIQUE, HORIZONTALE, VERTICALE

# les vues : (clé, titre, titre court, à l'échelle)
VUES: Tuple[Tuple[str, str, str, bool], ...] = (
    ("coupe", "Coupe · schéma statique", "Coupe", True),
    ("moments", "Moments · les deux ailes développées", "Moments", False),
    ("contraintes", "Contraintes · σéq et taux le long des ailes", "Contraintes", False),
    ("elevation", "Élévation · la tranche L et l'aile rabattue", "Élévation", True))
# les clés de données qu'une pièce du dessin peut désigner (data-sel)
CLES_SEL = ("X", "dx", "Y", "dy", "R", "L", "a", "n", "d", "profil", "yc", "uV", "jeu_ancrage")
# la lisibilité : sous ce rapport entre la hauteur de la cornière et la taille du texte, le dessin ne se lit plus
LISIBLE_MIN = 1.6


# ====================================================================== nombres, briques
def _num(x: float, nd: int) -> str:
    """Un nombre signé (signe « − »), ``nd`` décimales, virgule."""
    return F(x, nd).replace("-", "−", 1)


def _kn(x: float) -> str:
    """Un effort en kN : deux décimales, sans zéro inutile (35 kN, 14,25 kN, 42,76 kN)."""
    t = _num(x, 2)
    if "," in t:
        t = t.rstrip("0").rstrip(",")
    return t + " kN"


def _knm(x: float) -> str:
    """Un moment en kN·m à trois décimales, comme le classeur (2,010)."""
    return _num(x, 3)


def _pct(t: float) -> str:
    return _num(100.0 * t, 1) + " %"


def _pas(vmax: float, n: int = 4) -> Tuple[float, int]:
    """Le pas « rond » d'un axe (1, 2, 2,5 ou 5 × 10ᵏ) pour environ ``n`` intervalles, et ses décimales."""
    if vmax <= 0:
        return 1.0, 0
    brut = vmax / n
    mag = 10.0 ** math.floor(math.log10(brut))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= brut - 1e-12:
            pas = m * mag
            break
    nd = max(0, -math.floor(math.log10(pas) + 1e-9))
    if abs(pas / 10.0 ** math.floor(math.log10(pas) + 1e-9) - 2.5) < 1e-9:
        nd += 1
    return pas, nd


def _hach(cls, pts, pas):
    return _path(cls, _hachures(pts, pas))


def _fleche_cls(s, p, q, fut: str, pointe: str, fs: float):
    """Une flèche de p vers q (pointe en q) dans les classes voulues (fût en tirets, résultante verte…)."""
    (x0, y0), (x1, y1) = p, q
    L = math.hypot(x1 - x0, y1 - y0) or 1.0
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    hl, hw = 1.0 * fs, 0.38 * fs
    s.append(_path([fut], [[(x0, y0), (x1 - ux * hl * 0.6, y1 - uy * hl * 0.6)]]))
    s.append(_poly([pointe], [(x1, y1), (x1 - ux * hl + nx * hw, y1 - uy * hl + ny * hw), (x1 - ux * hl - nx * hw, y1 - uy * hl - ny * hw)]))


def _pastille(x: float, y: float, num: int, fs: float) -> Dict:
    """La pastille orange numérotée d'une section, centrée en (x, y)."""
    return _group(["scg"], [_circle(["scb"], 0.0, -0.34 * fs, 0.78 * fs), _text([], None, None, str(num), 0.95 * fs)],
                  x=x, y=y + 0.34 * fs, rot=0)


def _piece(s: List, prims: List, sel: Optional[str]):
    """Une pièce du dessin : groupée sous ``data-sel`` quand une donnée la pilote, à plat sinon."""
    if sel:
        s.append(_group(["pc"], list(prims), sel=sel))
    else:
        s.extend(prims)


def _renvoi(s, x0: float, y0: float, x1: float, y1: float, txt: str, fs: float):
    """Un renvoi : ligne de rappel depuis la pièce, étiquette à halo au bout (à droite ou à gauche du point)."""
    s.append(_path(["ext"], [[(x0, y0), (x1, y1)]]))
    w = _tw(txt, fs)
    _label(s, x1 + (w / 2 + 0.45 * fs) * (1 if x1 >= x0 else -1), y1 + 0.34 * fs, txt, fs, cls=("dl",))


def _cote_libre(s, fs: float, a: Tuple[float, float], b: Tuple[float, float], pos: float, txt: str, horizontale: bool = True,
                deport: int = +1):
    """Une cote posée dans le corps (hors des rangées de la feuille) : lignes d'attache depuis ``a`` et ``b``, ligne de
    cote en ``pos`` (y d'une cote horizontale, x d'une verticale), repères obliques, étiquette au milieu — déportée au
    bout ``deport`` (+1 : droite ou bas ; −1 : gauche ou haut) quand la cote est trop courte pour la porter."""
    t = 0.34 * fs
    w = _tw(txt, fs)
    if horizontale:
        (x0, y0), (x1, y1) = sorted([a, b])
        e0 = pos + (0.45 * fs if pos >= y0 else -0.45 * fs)
        ext = _path(["ext"], [[(x0, y0), (x0, e0)], [(x1, y1), (x1, e0)]], hint="V")
        dln = _path(["dln"], [[(x0, pos), (x1, pos)], [(x0 - t, pos + t), (x0 + t, pos - t)], [(x1 - t, pos + t), (x1 + t, pos - t)]], hint="H")
        if x1 - x0 >= w + 0.9 * fs:
            c = (x0 + x1) / 2
        else:
            c = (x1 + 0.75 * fs + w / 2) if deport > 0 else (x0 - 0.75 * fs - w / 2)
        lab = _group(["dl"], [_rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs), _text([], None, None, txt, fs)],
                     x=c, y=pos - 0.42 * fs, rot=0)
    else:
        (x0, y0), (x1, y1) = sorted([a, b], key=lambda p: p[1])
        e0 = pos + (0.45 * fs if pos >= x0 else -0.45 * fs)
        ext = _path(["ext"], [[(x0, y0), (e0, y0)], [(x1, y1), (e0, y1)]], hint="H")
        dln = _path(["dln"], [[(pos, y0), (pos, y1)], [(pos - t, y0 + t), (pos + t, y0 - t)], [(pos - t, y1 + t), (pos + t, y1 - t)]], hint="V")
        if y1 - y0 >= w + 0.9 * fs:
            c = (y0 + y1) / 2
        else:
            c = (y1 + 0.75 * fs + w / 2) if deport > 0 else (y0 - 0.75 * fs - w / 2)
        lab = _group(["dl"], [_rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs), _text([], None, None, txt, fs)],
                     x=pos - 0.42 * fs, y=c, rot=-90)
    s.append(_group(["dm"], [ext, dln, lab]))


def _dim_sel(S: Feuille, d: Dict, sel: Optional[str]):
    """Une cote de la feuille, groupée sous ``data-sel`` quand une donnée la pilote."""
    n = len(S.g)
    S.dim(d)
    if sel and len(S.g) > n:
        S.g[-1] = _group(["pc"], [S.g[-1]], sel=sel)


def _contour(larg: float, haut: float, e: float, r1: float, r2: float, yv: Callable[[float], float]) -> List[Tuple[float, float]]:
    """Le contour de la cornière dans la scène : talon à l'origine (dos de l'aile verticale au nu du mur, face
    extérieure de l'aile horizontale en y = 0), aile horizontale vers la droite, aile verticale dans le sens ``yv``
    (vers le haut en A, vers le bas en B) ; congé r₁ au talon intérieur, arrondis r₂ aux bouts des ailes."""
    r1 = max(0.0, min(r1, larg - e, haut - e))
    r2 = max(0.0, min(r2, e / 2.0, (larg - e) / 2.0, (haut - e) / 2.0))
    pts: List[Tuple[float, float]] = [(0.0, 0.0), (larg, 0.0)]
    if r2 > 0:
        pts += _arc(larg - r2, e - r2, r2, 0.0, math.pi / 2)                 # bout de l'aile horizontale
    else:
        pts.append((larg, e))
    if r1 > 0:
        pts += _arc(e + r1, e + r1, r1, -math.pi / 2, -math.pi)             # le congé
    else:
        pts.append((e, e))
    if r2 > 0:
        pts += _arc(e - r2, haut - r2, r2, 0.0, math.pi / 2)               # bout de l'aile verticale
    else:
        pts.append((e, haut))
    pts.append((0.0, haut))
    return [(x, yv(y)) for x, y in pts]


def _lisible(dimension: float, fs: float, quoi: str):
    """Le dessin doit rester lisible : une pièce plus petite que ~1,6 ligne de texte n'est plus un dessin."""
    if dimension < LISIBLE_MIN * fs:
        raise ValueError(f"dessin impossible à cette échelle : {quoi} ({f0(dimension)} mm) trop petit face au texte ({f0(fs)} mm)")


def _dessin(prims: List, fs: float, aria: str) -> Dessin:
    """Un dessin sans feuille de cotation (les diagrammes) : la boîte de vue est l'étendue du corps plus une marge."""
    ex = _extent(prims, fs) or (0.0, 0.0, 1.0, 1.0)
    m = 0.6 * fs
    return Dessin(viewbox=(ex[0] - m, ex[1] - m, ex[2] - ex[0] + 2 * m, ex[3] - ex[1] + 2 * m), corps=prims, cotes=[], cartouche=[],
                  aria=aria, fs=fs)


def _bras(res: Dict) -> Dict:
    """La résultante de compression et le bras de la traction, en abscisse y (depuis la face extérieure de l'aile
    horizontale, vers l'aile verticale) : le pivot en A (Y′ − bras), le contact en B (Y′ + bras) — Y′ celui de la
    combinaison qui gouverne la traction."""
    anc = res["ancrages"]
    Yp = anc["combinaison_N"]["Y"]
    if res["configuration"] == "A":
        return {"y_C": Yp - anc["bras"], "Yp": Yp}
    return {"y_C": Yp + anc["bras"], "Yp": Yp}


# ====================================================================== la coupe : le schéma statique
def coupe(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """Le schéma statique coté (voir l'en-tête du module)."""
    g, sec, cfg, mode = res["geometrie"], res["section"], res["configuration"], res["mode"]
    e, r, larg, haut = g["e"], g["r"], g["larg"], g["haut"]
    r2 = float(sec.get("r2") or 0.0)
    X, dx, Y, dy, d, d0 = g["X"], g["dx"], g["Y"], g["dy"], g["d"], g["d0"]
    eff, anc = res["efforts"], res["ancrages"]
    A = cfg == "A"
    k = -1.0 if A else 1.0

    def yv(y: float) -> float:
        """y de la scène d'une abscisse comptée depuis la face extérieure de l'aile horizontale vers l'aile verticale."""
        return k * y

    fs = float(fs) if fs else (larg + haut) / 34.0
    sel = (lambda cle: cle) if interactif else (lambda cle: None)
    hist = mode == HISTORIQUE
    br = _bras(res)
    y_face = yv(e) if A else yv(0.0)                 # la face qui reçoit la charge (le dessus de l'aile horizontale)
    s: List[Dict] = []
    # --- la cornière coupée (congé r₁, bouts r₂), hachurée
    pts = _contour(larg, haut, e, r, r2, yv)
    _piece(s, [_poly(["pp"], pts), _hach(["ht"], pts, max(1.2, 0.22 * fs))], sel("profil"))
    # --- l'ancrage : le trou dans l'aile, la tige (cachée dans le mur), la rondelle, l'écrou, l'axe
    ya = yv(Y)
    prof = max(2.2 * fs, 3.0 * d)
    x_n = e + 0.18 * d + 0.8 * d                     # le nu de l'écrou
    s.append(_rect(["bg"], 0.0, ya - d0 / 2, e, d0))
    s.append(_path(["fl2"], [[(0.0, ya - d0 / 2), (e, ya - d0 / 2)], [(0.0, ya + d0 / 2), (e, ya + d0 / 2)]]))
    tige = [_path(["hd"], [[(0.0, ya - d / 2), (-prof, ya - d / 2)], [(0.0, ya + d / 2), (-prof, ya + d / 2)], [(-prof, ya - d / 2), (-prof, ya + d / 2)]]),
            _rect(["tg"], 0.0, ya - d / 2, x_n + 0.3 * d, d),
            _rect(["bo"], e, ya - 1.0 * d, 0.18 * d, 2.0 * d),
            _rect(["bo"], e + 0.18 * d, ya - 0.85 * d, 0.8 * d, 1.7 * d)]
    _piece(s, tige, sel("d"))
    _piece(s, [_path(["ax"], [[(-prof - 0.6 * fs, ya), (x_n + 0.3 * d + 0.9 * fs, ya)]])], sel("Y"))
    if dy > 0:
        pos = [_path(["hd"], [[(-1.6 * fs, yv(Y + j)), (x_n + 0.3 * d + 0.4 * fs, yv(Y + j))]]) for j in (-dy, dy)]
        _cote_libre(pos, fs, (x_n + 0.3 * d + 0.6 * fs, ya), (x_n + 0.3 * d + 0.6 * fs, yv(Y + dy)), x_n + 0.3 * d + 1.3 * fs,
                    f"dy = {f0(dy)}", horizontale=False, deport=-1 if A else +1)
        _piece(s, pos, sel("dy"))
    # --- la charge R à X (plein) et ses positions X ± dx (tirets) ; en A, quand l'ancrage est bas, la flèche s'allonge
    #     pour que l'étiquette de R passe au-dessus de celles de l'ancrage (H, Ø), posées ensuite par-dessus son fût
    xa0 = x_n + 0.3 * d
    x_lab_h = xa0 + 3.9 * fs
    txt_R = f"R = {_kn(eff['R'])}"
    w_R = _tw(txt_R, fs)
    y_R = e + 5.56 * fs                              # l'étiquette de R, en abscisse y (depuis la face extérieure)
    if A and X + w_R / 2 + 0.3 * fs > x_lab_h and Y - 2.0 * fs < y_R + 0.7 * fs and Y + 1.3 * fs > e:
        y_R = Y + 2.4 * fs
    y_t = yv(y_R - 0.56 * fs) if A else y_face - (y_R - e - 0.56 * fs)
    _piece(s, [_path(["fz"], [[(X, y_t), (X, y_face - 0.6 * fs)]]),
               _poly(["fzh"], [(X, y_face), (X - 0.38 * fs, y_face - 1.0 * fs), (X + 0.38 * fs, y_face - 1.0 * fs)])], sel("X"))
    rlab: List[Dict] = []
    _label(rlab, X, y_t - 0.9 * fs + 0.34 * fs, txt_R, fs)
    _piece(s, rlab, sel("R"))
    if dx > 0:
        pos = []
        for xj in (X - dx, X + dx):
            _fleche_cls(pos, (xj, y_t + 0.9 * fs), (xj, y_face), "fzd", "fzh", fs)
        _cote_libre(pos, fs, (X, y_face - 2.6 * fs), (X + dx, y_face - 2.6 * fs), y_face - 2.6 * fs, f"dx = {f0(dx)}", deport=+1)
        _piece(s, pos, sel("dx"))
    # --- la traction de l'ancrage sur la cornière (vers le mur), sa valeur, le diamètre et le nombre
    _fleche(s, (x_lab_h - 0.3 * fs, ya), (xa0 + 0.3 * fs, ya), "", fs)
    txt_h = f"H = {_kn(eff['H'])}"
    _label(s, x_lab_h + _tw(txt_h, fs) / 2, ya - 0.5 * fs, txt_h, fs)
    dlab: List[Dict] = []
    txt_d = f"Ø{f0(d)} · n = {g['n']}"
    _label(dlab, x_lab_h + _tw(txt_d, 0.9 * fs) / 2, ya + 1.25 * fs + 0.34 * fs, txt_d, 0.9 * fs, cls=("dl",))
    _piece(s, dlab, sel("d"))
    # --- la résultante de compression contre le mur : au pivot (A) ou au bout libre (B)
    yc_s = yv(br["y_C"])
    comp: List[Dict] = []
    _fleche_cls(comp, (-3.4 * fs, yc_s), (-0.15 * fs, yc_s), "fc", "fch", fs)
    comp.append(_circle(["fcp"], 0.0, yc_s, 0.3 * fs))
    txt_c = f"C = {_kn(eff['C'])}" if not hist else f"C = H = {_kn(eff['C'])}"
    _label(comp, -3.6 * fs - _tw(txt_c, fs) / 2, yc_s - 0.55 * fs, txt_c, fs, cls=("fcl",))
    _piece(s, comp, sel("yc" if not hist else "jeu_ancrage"))
    # --- la ligne d'action du cisaillement d'ancrage (équilibre) ; son nom au bout, côté mur (les pastilles 1 et 2 sont
    #     de l'autre côté)
    if not hist:
        uV = g["uV"]
        _piece(s, [_path(["ael"], [[(uV, ya), (uV, yv(-1.3 * fs))]]),
                   _text(["tx2"], uV - 0.25 * fs, yv(-1.3 * fs) + (0.9 * fs if A else -0.3 * fs), "uV", 0.9 * fs, "end")], sel("uV"))
    # --- les sections 1 à 4 : traits orange à travers l'aile, prolongés au-delà de la face extérieure (1, 2) ou dans
    #     l'angle (3, 4) jusqu'à leur pastille ; congé étroit : la pastille de la section au nu est décalée d'un cran
    etag = r < 1.9 * fs
    y1p, y2p = yv(-2.2 * fs), yv(-3.9 * fs if etag else -2.2 * fs)
    secs = [_path(["sc"], [[(e + r, yv(e)), (e + r, yv(-1.4 * fs))]]),
            _path(["sc"], [[(e, yv(e + r)), (e, yv(-3.1 * fs if etag else -1.4 * fs))]]),
            _path(["sc"], [[(-0.6 * fs, yv(e + r)), (e + 1.0 * fs, yv(e + r))]]),
            _path(["sc"], [[(-0.6 * fs, yv(e)), (e + r + (2.6 * fs if etag else 1.0 * fs), yv(e))]])]
    past = [_pastille(e + r, y1p, 1, fs), _pastille(e, y2p, 2, fs), _pastille(e + 1.8 * fs, yv(e + r), 3, fs),
            _pastille(e + r + (3.4 * fs if etag else 1.8 * fs), yv(e), 4, fs)]
    # --- la feuille de cotation : la boîte des pièces, élargie à l'étendue du corps (flèches, étiquettes, pastilles)
    bb0 = dict(x1=0.0, x2=larg, y1=min(yv(0.0), yv(haut)), y2=max(yv(0.0), yv(haut)))
    S = Feuille(_bb(bb0, s + past, fs), fs, options_schemas(2))
    cote_x = "B" if A else "T"                     # les cotes de l'aile horizontale, du côté de sa face extérieure
    # les bras de levier du mode (les plus près de la pièce), puis Y et haut
    if hist:
        _dim_sel(S, dict(id="Ye", side="L", a=yv(e + r), b=yv(Y), o1=0.0, o2=0.0, sym="Ye", val=Y - e - r, lvl=1), None)
        _dim_sel(S, dict(id="Yer", side="L", a=yv(e), b=yv(Y), o1=0.0, o2=0.0, sym="Ye+r", val=Y - e, lvl=1), None)
        _dim_sel(S, dict(id="YH", side="L", a=yv(e / 2.0), b=yv(Y), o1=0.0, o2=0.0, sym="YH", val=Y - e / 2.0, lvl=1), None)
    elif A:
        _dim_sel(S, dict(id="bras", side="L", a=yv(g["yc"]), b=yv(Y), o1=0.0, o2=0.0, sym="Y − yc", val=Y - g["yc"], lvl=1), sel("yc"))
    else:
        _dim_sel(S, dict(id="bras", side="L", a=yv(Y), b=yv(haut - g["yc"]), o1=0.0, o2=0.0, sym="ya − c", val=haut - g["yc"] - Y, lvl=1), sel("yc"))
    _dim_sel(S, dict(id="Y", side="L", a=yv(0.0), b=yv(Y), o1=0.0, o2=0.0, sym="Y", val=Y, lvl=1), sel("Y"))
    _dim_sel(S, dict(id="haut", side="L", a=yv(0.0), b=yv(haut), o1=0.0, o2=0.0, sym="haut", val=haut, lvl=1), sel("profil"))
    if hist:
        _dim_sel(S, dict(id="Xe", side=cote_x, a=e + r, b=X, o1=yv(0.0), o2=yv(0.0), sym="Xe", val=X - r - e, lvl=1), None)
        _dim_sel(S, dict(id="Xer", side=cote_x, a=e, b=X, o1=yv(0.0), o2=yv(0.0), sym="Xe+r", val=X - e, lvl=1), None)
    else:
        _dim_sel(S, dict(id="bras_x", side=cote_x, a=g["uV"], b=X, o1=yv(0.0), o2=yv(0.0), sym="X − uV", val=X - g["uV"], lvl=1), sel("uV"))
    _dim_sel(S, dict(id="X", side=cote_x, a=0.0, b=X, o1=yv(0.0), o2=yv(0.0), sym="X", val=X, lvl=1), sel("X"))
    _dim_sel(S, dict(id="larg", side=cote_x, a=0.0, b=larg, o1=yv(0.0), o2=yv(0.0), sym="larg", val=larg, lvl=1), sel("profil"))
    _dim_sel(S, dict(id="e", side="R", a=yv(0.0), b=yv(e), o1=larg, o2=larg, sym="e", val=e, lvl=1, out="lo" if A else "hi"), sel("profil"))
    # le congé : un renvoi, placé là où il ne gêne pas — le long de l'aile verticale entre les pastilles 3, 4 et
    # l'écrou si la charge la plus proche et l'étiquette de R laissent la place, sinon à droite des charges, sinon
    # au-dessus de l'écrou, sinon au-dessus de l'étiquette de R
    if r > 0:
        txt_r = f"r = {f0(r)}"
        w = _tw(txt_r, fs) + 0.9 * fs
        c45 = math.cos(math.pi / 4)
        pm = (e + r - r * c45, yv(e + r - r * c45))
        y0_r = e + r + 1.6 * fs                      # juste au-dessus de la pastille 3
        bord = X - dx - 0.5 * fs                     # la flèche de charge la plus proche…
        if y0_r + 0.7 * fs > y_R - 0.7 * fs:         # … et l'étiquette de R si l'on monte jusqu'à elle
            bord = min(bord, X - w_R / 2 - 0.5 * fs)
        if not A and e + r + 3.2 * fs + w <= larg - 1.0 * fs:
            lab = (e + r + 3.2 * fs, yv(e + 2.2 * fs))     # B : l'intérieur de l'angle est libre, à droite de la pastille 4
        elif A and (Y - 0.85 * d - 0.3 * fs) >= y0_r + 0.7 * fs and e + 0.8 * fs + w <= bord:
            lab = (e + 0.8 * fs, yv(y0_r))
        elif larg - (X + dx) >= w + 1.2 * fs:
            lab = (X + dx + 0.6 * fs, yv(e + 1.0 * fs))
        elif Y + 0.85 * d + 2.4 * fs <= haut:
            lab = ((x_n + 0.3 * d + 2.3 * fs) if dy > 0 else (e + 1.0 * fs), yv(Y + 0.85 * d + 1.8 * fs))
        else:
            lab = (e + r + 1.0 * fs, yv(e + 8.8 * fs))
        rv: List[Dict] = []
        _renvoi(rv, pm[0], pm[1], lab[0], lab[1], txt_r, fs)
        _piece(S.g, rv, sel("profil"))
    S.g += secs + past
    _etendre(S, past, fs)
    # --- le mur : derrière les cotes de l'aile verticale, rompu (il continue), hachuré clair
    n_L = len(S.rows["L"])
    ep_mur = 1.4 * fs + n_L * 2.35 * fs + 1.6 * fs
    rows_T, rows_B = S.rows["T"], S.rows["B"]
    y_haut = min(S.ex["y1"], S.pos("T", len(rows_T) - 1) - 1.7 * fs if rows_T else S.ex["y1"]) - 3.5 * fs
    y_bas = max(S.ex["y2"], S.pos("B", len(rows_B) - 1) + 0.6 * fs if rows_B else S.ex["y2"]) + 2.0 * fs
    pm_pts = _rect_rompu(-ep_mur, y_haut, ep_mur, y_bas - y_haut, {"T", "B", "L"}, 0.5 * fs)
    mur = [_poly(["be"], pm_pts), _hach(["ht"], pm_pts, 0.9 * fs), _text(["tx2"], -ep_mur + 0.6 * fs, y_haut + 1.5 * fs, "mur", 0.9 * fs, "start")]
    _etendre(S, mur, fs)
    aria = (f"Coupe : schéma statique de la cornière {sec['designation']} en configuration {cfg} ({LIBELLES_CONFIGURATION[cfg]}), "
            f"{LIBELLES_MODE[mode]} — charge R à X, ancrage à Y, sections 1 à 4, bras de levier")
    return S.finish(mur + s, [], aria)


# ====================================================================== l'élévation : la tranche L et l'aile rabattue
def elevation(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """La tranche L vue de face et l'aile horizontale rabattue (voir l'en-tête du module)."""
    g, sec, cfg, anc, eff = res["geometrie"], res["section"], res["configuration"], res["ancrages"], res["efforts"]
    e, r, larg, haut, L = g["e"], g["r"], g["larg"], g["haut"], g["L"]
    n, d, d0, Y, X = g["n"], g["d"], g["d0"], g["Y"], g["X"]
    zs = anc["positions_z"]
    a = g["a"] if g["a"] is not None else L
    cb = g["combinaison_L_eff"]
    Xp = cb["X"]
    Xe = Xp - r - e
    L_eff = g["L_eff"]
    A = cfg == "A"
    k = -1.0 if A else 1.0

    def yv(y: float) -> float:
        return k * y

    fs = float(fs) if fs else (L + 1.5 * (haut + larg)) / 44.0
    _lisible(haut, fs, "la hauteur de la cornière")
    sel = (lambda cle: cle) if interactif else (lambda cle: None)
    gap = 3.2 * fs                                 # entre la face de la cornière et l'aile rabattue

    def yu(u: float) -> float:
        """y de la scène d'un point de l'aile horizontale rabattue (u depuis le nu du mur) : le bout libre reste contre
        l'élévation, le côté du mur s'en éloigne."""
        return -k * (gap + larg - u)

    s: List[Dict] = []
    # --- le mur derrière l'aile verticale (il continue au-delà : rompu), la face de l'aile verticale, la tranche de
    #     l'aile horizontale devant
    y_lo, y_hi = sorted([yv(0.0), yv(haut)])
    marge_z, marge_h = 3.6 * fs, 2.6 * fs
    y_mur = (y_lo - marge_h) if A else (y_lo - 0.5 * fs)
    h_mur = haut + marge_h + 0.5 * fs
    fond = _rect_rompu(-marge_z, y_mur, L + 2 * marge_z, h_mur, {"L", "R", "T", "B"}, 0.5 * fs)
    s.append(_poly(["be"], fond))
    piece_prims = [_rect(["ps"], 0.0, y_lo, L, haut), _rect(["pp"], 0.0, min(yv(0.0), yv(e)), L, e)]
    # --- l'aile horizontale rabattue : la tranche de l'aile verticale vue de dessus (u ∈ [0, e]), le dessus de l'aile,
    #     la tangente du congé
    u_lo, u_hi = sorted([yu(0.0), yu(e)])
    piece_prims += [_rect(["ps"], 0.0, min(yu(e), yu(larg)), L, larg - e), _rect(["pp"], 0.0, u_lo, L, e),
                    _path(["flr"], [[(0.0, yu(e + r)), (L, yu(e + r))]])]
    _piece(s, piece_prims, sel("profil"))
    titre_plan = "aile horizontale vue de dessus (rabattue)"
    if L >= _tw(titre_plan, 0.9 * fs) + 1.0 * fs:
        s.append(_text(["tx2"], L / 2, (yv(0.0) + yu(larg)) / 2 + 0.3 * fs, titre_plan, 0.9 * fs, "middle"))
    # --- la section 1 (pied du congé) sur toute la longueur, sa pastille
    s.append(_path(["sc"], [[(-0.6 * fs, yu(e + r)), (L + 0.6 * fs, yu(e + r))]]))
    s.append(_pastille(-1.6 * fs, yu(e + r), 1, fs))
    # --- la charge : la bande de largeur a centrée sur L à u = X ; la position X′ qui gouverne L_eff en tirets si
    #     elle diffère ; la diffusion à 45° jusqu'à la section 1 (le trapèze L_eff)
    z0, z1 = max(0.0, L / 2 - a / 2), min(L, L / 2 + a / 2)
    charge: List[Dict] = [_rect(["ct"], z0, min(yu(X - 0.45 * fs), yu(X + 0.45 * fs)), z1 - z0, 0.9 * fs)]
    if abs(Xp - X) > 1e-9:
        charge.append(_path(["fzd"], [[(z0, yu(Xp)), (z1, yu(Xp))]]))
    if a < L - 1e-9 and Xe > 1e-9:
        # le trapèze, coupé aux bords de la tranche : les côtés à 45° (du bord de la charge vers le pied du congé)
        za, zb = z0 - Xe, z1 + Xe
        poly = [(z0, Xp), (z1, Xp)]
        if zb <= L:
            poly.append((zb, e + r))
        else:
            poly += [(L, Xp - (L - z1)), (L, e + r)]
        if za >= 0:
            poly.append((za, e + r))
        else:
            poly += [(0.0, e + r), (0.0, Xp - z0)]
        charge.append(_poly(["ae"], [(z, yu(u)) for z, u in poly]))
        charge.append(_path(["ael"], [[(z0, yu(Xp)), (max(za, 0.0), yu(Xp - (z0 - max(za, 0.0))))],
                                      [(z1, yu(Xp)), (min(zb, L), yu(Xp - (min(zb, L) - z1)))]]))
        _cote_libre(charge, fs, (z0, yu(X)), (z1, yu(X)), yu(X + 1.4 * fs), f"a = {f0(a)}", deport=+1)
        txt_r = f"R = {_kn(eff['R'])}"
        _label(charge, L / 2, yu(X + 3.7 * fs) + 0.34 * fs, txt_r, fs)
    else:
        _label(charge, L / 2, yu(X + 1.5 * fs) + 0.34 * fs, f"R = {_kn(eff['R'])} sur a = L", fs)
    _piece(s, charge, sel("a"))
    # --- les ancrages : rondelle, tige, axes ; N et V ELS de l'enveloppe par ancrage quand la place le permet
    ancr: List[Dict] = []
    ya = yv(Y)
    for z in zs:
        ancr += [_circle(["bw"], z, ya, 0.95 * d), _circle(["bo"], z, ya, d / 2)]
    dep = 0.95 * d + 0.5 * fs
    ancr.append(_path(["ax"], [[(zs[0] - dep, ya), (zs[-1] + dep, ya)]] + [[(z, ya - dep), (z, ya + dep)] for z in zs]))
    _piece(s, ancr, sel("n"))
    txt_n, txt_v = f"N = {_kn(anc['N_ELS'])}", f"V = {_kn(anc['V_ELS'])}"
    w_nv = max(_tw(txt_n, 0.9 * fs), _tw(txt_v, 0.9 * fs)) + 0.8 * fs
    y_et = ya - k * (0.95 * d + 1.4 * fs)          # vers le bout libre de l'aile verticale
    place = haut - Y - 0.95 * d >= 3.2 * fs and (L / n >= w_nv)
    if place:
        for z in zs:
            _label(s, z, y_et - (0.7 * fs if A else -0.7 * fs) + 0.34 * fs, txt_n, 0.9 * fs)
            _label(s, z, y_et + (0.7 * fs if A else -0.7 * fs) + 0.34 * fs, txt_v, 0.9 * fs)
    # le diamètre et le nombre : un renvoi depuis le premier ancrage
    dlab: List[Dict] = []
    _renvoi(dlab, zs[0] + 0.7 * d, ya + k * 0.7 * d, zs[0] + 0.95 * d + 1.6 * fs, ya + k * (0.95 * d + 1.3 * fs),
            f"{n} Ø{f0(d)} · d0 = {f0(d0)}", 0.9 * fs)
    _piece(s, dlab, sel("d"))
    # --- la feuille : la chaîne des entraxes du côté de l'aile verticale, L et L_eff du côté de l'aile rabattue
    bb0 = dict(x1=0.0, x2=L, y1=min(y_lo, yu(0.0), yu(larg)), y2=max(y_hi, yu(0.0), yu(larg)))
    S = Feuille(_bb(bb0, s, fs), fs, options_schemas(2))
    cote_chaine = "T" if A else "B"
    cote_L = "B" if A else "T"
    xs = [0.0] + list(zs) + [L]
    o_ch = yv(haut)
    t = 0.34 * fs
    rang = S.slot(cote_chaine, 0.0, L)
    p = S.pos(cote_chaine, rang)
    e_ch = p + (-0.45 if cote_chaine == "T" else 0.45) * fs
    ext = _path(["ext"], [[(x, o_ch), (x, e_ch)] for x in xs], hint="V")
    dln = _path(["dln"], [[(xs[0], p), (xs[-1], p)]] + [[(x - t, p + t), (x + t, p - t)] for x in xs], hint="H")
    labs = []
    for z_a, z_b in zip(xs[:-1], xs[1:]):
        txt = f0(z_b - z_a)
        if z_b - z_a >= _tw(txt, fs) + 0.6 * fs:
            labs.append(S.label((z_a + z_b) / 2, p - 0.42 * fs, False, txt, dict(id="entraxe", sym="", lvl=1)))
    _piece(S.g, [_group(["dm"], [ext, dln] + labs)], sel("n"))
    if g["a"] is not None:
        _dim_sel(S, dict(id="Leff", side=cote_L, a=max(0.0, L / 2 - L_eff / 2), b=min(L, L / 2 + L_eff / 2), o1=yu(e + r), o2=yu(e + r),
                         sym="Leff", val=L_eff, lvl=1), sel("a"))
    _dim_sel(S, dict(id="L", side=cote_L, a=0.0, b=L, o1=yu(0.0), o2=yu(0.0), sym="L", val=L, lvl=1), sel("L"))
    _dim_sel(S, dict(id="haut", side="L", a=yv(0.0), b=yv(haut), o1=0.0, o2=0.0, sym="haut", val=haut, lvl=1), sel("profil"))
    _dim_sel(S, dict(id="larg", side="L", a=yu(0.0), b=yu(larg), o1=0.0, o2=0.0, sym="larg", val=larg, lvl=1), sel("profil"))
    _dim_sel(S, dict(id="Y", side="R", a=yv(0.0), b=yv(Y), o1=L, o2=L, sym="Y", val=Y, lvl=1), sel("Y"))
    _dim_sel(S, dict(id="X", side="R", a=yu(0.0), b=yu(X), o1=L, o2=L, sym="X", val=X, lvl=1), sel("X"))
    if not place:
        # pas la place sous chaque ancrage : une ligne pour tous, dans une rangée libre du côté de l'aile rabattue
        txt = f"par ancrage (ELS, enveloppe) : {txt_n} · {txt_v}"
        w = _tw(txt, fs) * 1.08
        rg = S.slot(cote_L, L / 2 - w / 2, L / 2 + w / 2)
        pp = S.pos(cote_L, rg)
        hit = _rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs)
        S.g.append(_group(["fzl"], [hit, _text([], None, None, txt, fs)], x=L / 2, y=pp + (0.3 if cote_L == "B" else -0.1) * fs, rot=0))
    _etendre(S, s, fs)
    aria = (f"Élévation : la tranche L = {f0(L)} mm de la cornière {sec['designation']} vue de face, {n} ancrage(s) Ø{f0(d)} à "
            f"l'entraxe {f0(L / n)} mm, et l'aile horizontale rabattue avec la charge de largeur {f0(a)} mm et sa diffusion à 45°")
    return S.finish(s, [], aria)


# ====================================================================== les diagrammes : un panneau par aile
class _Panneau:
    """Un graphe « s → valeur » d'une aile : abscisse s de 0 à ``fin`` (mm) sur la largeur ``w``, ordonnée de ``vmin`` à
    ``vmax`` sur la hauteur ``h`` ; coin haut gauche en (x0, y0). Tout est en unités de la scène ; les épaisseurs de
    trait sont des pixels d'écran (vector-effect)."""

    def __init__(self, s: List, fs: float, x0: float, y0: float, w: float, h: float, fin: float, vmin: float, vmax: float):
        self.s, self.fs, self.x0, self.y0, self.w, self.h, self.fin = s, fs, x0, y0, w, h, fin
        self.vmin, self.vmax = vmin, (vmax if vmax > vmin else vmin + 1.0)

    def X(self, sv: float) -> float:
        return self.x0 + self.w * sv / self.fin

    def Y(self, v: float) -> float:
        return self.y0 + self.h * (self.vmax - v) / (self.vmax - self.vmin)

    def zones(self, e: float, r: float, etiquettes: bool = True, en_haut: bool = False):
        """Le bloc d'angle (s < e) hachuré, le congé (e ≤ s ≤ e + r) en orange clair, leurs noms quand la place le
        permet — en bas du panneau, ou en haut (``en_haut``) quand les courbes occupent le bas (moments négatifs)."""
        fs = self.fs
        x0, x1, x2 = self.X(0.0), self.X(e), self.X(e + r)
        bloc = [(x0, self.y0), (x1, self.y0), (x1, self.y0 + self.h), (x0, self.y0 + self.h)]
        self.s.append(_poly(["bg"], bloc))
        self.s.append(_hach(["ht"], bloc, 0.45 * fs))
        if r > 0:
            self.s.append(_rect(["cong"], x1, self.y0, x2 - x1, self.h))
        if etiquettes:
            yb = self.y0 + 1.1 * fs if en_haut else self.y0 + self.h - 0.45 * fs
            if x1 - x0 >= _tw("angle", 0.8 * fs) + 0.4 * fs:
                self.s.append(_text(["tx2"], (x0 + x1) / 2, yb, "angle", 0.8 * fs))
            if x2 - x1 >= _tw("congé", 0.8 * fs) + 0.4 * fs:
                self.s.append(_text(["txw"], (x1 + x2) / 2, yb, "congé", 0.8 * fs))

    def cadre(self, pas_s: float, pas_v: float, nd_v: int, titre_v: str, axe_s: bool = True, zero: bool = True):
        """Le quadrillage, les deux axes, les graduations (s en bas, valeurs à gauche), le titre de l'ordonnée."""
        fs = self.fs
        x0, x1, y0, y1 = self.x0, self.x0 + self.w, self.y0, self.y0 + self.h
        v = math.ceil(self.vmin / pas_v - 1e-9) * pas_v
        while v <= self.vmax + 1e-9:
            y = self.Y(v)
            if abs(v) > 1e-12 or not zero:
                self.s.append(_path(["grd"], [[(x0, y), (x1, y)]]))
            self.s.append(_text(["tx2"], x0 - 0.45 * fs, y + 0.3 * fs, _num(v, nd_v) if abs(v) > 1e-12 else "0", 0.8 * fs, "end"))
            v += pas_v
        sv = 0.0
        while sv <= self.fin + 1e-9:
            x = self.X(sv)
            self.s.append(_path(["grd"], [[(x, y0), (x, y1)]]))
            if axe_s:
                self.s.append(_text(["tx2"], x, y1 + 1.15 * fs, f0(sv), 0.8 * fs))
            sv += pas_s
        self.s.append(_path(["axe"], [[(x0, y0), (x0, y1)], [(x0, y1), (x1, y1)]]))
        if zero and self.vmin < -1e-12 < self.vmax:
            self.s.append(_path(["axe"], [[(x0, self.Y(0.0)), (x1, self.Y(0.0))]]))
        if titre_v:
            self.s.append(_titre_vertical(titre_v, x0 - 3.3 * fs, (y0 + y1) / 2, 0.85 * fs))

    def courbe(self, pts: List[Tuple[float, float]], cls: str):
        self.s.append(_path([cls], [[(self.X(a), self.Y(b)) for a, b in pts]]))

    def sections(self, secs: List[Dict], ordre: Dict[str, int], y_pastille: Optional[float] = None):
        """Les traits des sections (tirets orange) et leurs pastilles au-dessus du panneau (en ``y_pastille``) ; la
        pastille de la section au nu (s = e) monte d'un cran quand le congé est trop étroit pour les deux."""
        fs = self.fs
        yp = self.y0 - 1.0 * fs if y_pastille is None else y_pastille
        xs = {v["cle"]: self.X(v["s"]) for v in secs}
        serre = len(xs) == 2 and abs(max(xs.values()) - min(xs.values())) < 1.8 * fs
        for v in secs:
            x = xs[v["cle"]]
            self.s.append(_path(["d1"], [[(x, self.y0), (x, self.y0 + self.h)]]))
            haut = serre and v["s"] < max(w["s"] for w in secs)
            self.s.append(_pastille(x, yp - (1.65 * fs if haut else 0.0), ordre[v["cle"]], fs))

    def point(self, sv: float, v: float, txt: str, droite: bool = True, dessus: bool = True, cls=("fzl",), taille: float = 1.0,
              x_min: Optional[float] = None, y_fixe: Optional[float] = None) -> Tuple[float, float, float, float]:
        """Une valeur écrite près d'un point de la courbe (au-dessus à droite par défaut ; ``x_min`` : pas plus à
        gauche que ce bord, ``y_fixe`` : sur cette ligne). Rend la boîte (x1, y1, x2, y2) de l'étiquette."""
        fs = self.fs * taille
        x, y = self.X(sv), self.Y(v)
        w = _tw(txt, fs)
        xg = x + 0.55 * fs if droite else x - 0.55 * fs - w
        if x_min is not None:
            xg = max(xg, x_min)
        yl = (y - 0.55 * fs if dessus else y + 1.25 * fs) if y_fixe is None else y_fixe
        _label(self.s, xg + w / 2, yl + 0.34 * fs, txt, fs, cls=cls)
        return xg - 0.3 * fs, yl - 1.05 * fs, xg + w + 0.3 * fs, yl + 0.37 * fs


def _placer_bloc(P: _Panneau, courbe: List[Tuple[float, float]], w: float, h: float, points: List[Tuple[float, float]],
                 fs: float) -> Tuple[float, float]:
    """Le coin haut gauche d'un bloc de texte w × h dans le panneau, au premier emplacement candidat (en haut à droite,
    à mi-hauteur à droite, en bas à droite, en bas au centre, en haut au centre) que la courbe et les points marqués
    ne traversent pas ; à défaut, en haut à droite."""
    m = 0.35 * fs
    xr = P.x0 + P.w - w - 0.4 * fs
    cands = [(xr, P.y0 + 0.4 * fs), (xr, P.y0 + (P.h - h) / 2), (xr, P.y0 + P.h - h - 1.0 * fs),
             (P.x0 + (P.w - w) / 2, P.y0 + P.h - h - 1.0 * fs), (P.x0 + (P.w - w) / 2, P.y0 + 0.4 * fs)]
    for x, y in cands:
        libre = all(not (x - m <= px <= x + w + m and y - m <= py <= y + h + m) for px, py in courbe + points)
        if libre:
            return x, y
    return cands[0]


def _etiquettes_troncons(P: _Panneau, troncons: List[Dict], y_rang: float, fs: float):
    """N et V par tronçon, chacun sur sa propre rangée au-dessus du panneau (les tronçons de B se chevauchent)."""
    for i, tr in enumerate(troncons):
        xa, xb = P.X(max(0.0, tr["de"])), P.X(min(P.fin, tr["a"]))
        txt = f"N = {_kn(tr['N'])} · V = {_kn(tr['V'])}"
        w = _tw(txt, 0.9 * fs)
        c = (xa + xb) / 2
        c = min(max(c, P.x0 + w / 2), P.x0 + P.w - w / 2)
        y = y_rang - i * 1.45 * fs
        P.s.append(_path(["ext"], [[(xa, y + 0.55 * fs), (xb, y + 0.55 * fs)], [(xa, y + 0.25 * fs), (xa, y + 0.85 * fs)],
                                   [(xb, y + 0.25 * fs), (xb, y + 0.85 * fs)]]))
        _label(P.s, c, y - 0.1 * fs + 0.34 * fs, txt, 0.9 * fs, cls=("dl",))


def _marques_charge(P: _Panneau, rep: Dict, fs: float):
    """La charge R : une flèche par position (nominale en plein, extrêmes en tirets) qui tombe sur l'axe du moment nul."""
    y0 = P.Y(0.0)
    for xp in rep["positions"]:
        nominale = abs(xp - rep["X"]) < 1e-9
        _fleche_cls(P.s, (P.X(xp), y0 - 2.6 * fs), (P.X(xp), y0), "fz" if nominale else "fzd", "fzh", fs)
        if nominale:
            P.s.append(_text(["tx"], P.X(xp) + 0.35 * fs, y0 - 2.2 * fs, "R", 0.9 * fs, "start"))


def _marques_ancrage(P: _Panneau, rep: Dict, contact: Optional[float], fs: float):
    """L'ancrage : la bande de tolérance Y ± dy, les positions extrêmes en tirets fins, l'axe nominal ; le contact (B)."""
    b0, b1 = rep["bande"]
    if b1 - b0 > 1e-9:
        P.s.append(_rect(["tol"], P.X(b0), P.y0, P.X(b1) - P.X(b0), P.h))
        for xp in rep["positions"]:
            if abs(xp - rep["Y"]) > 1e-9:
                P.s.append(_path(["hd"], [[(P.X(xp), P.y0), (P.X(xp), P.y0 + P.h)]]))
    P.s.append(_path(["ax"], [[(P.X(rep["Y"]), P.y0 - 0.3 * fs), (P.X(rep["Y"]), P.y0 + P.h)]]))
    P.s.append(_text(["tx"], P.X(rep["Y"]) + 0.35 * fs, P.y0 + 0.5 * P.h + 0.3 * fs, "ancrage", 0.85 * fs, "start"))
    if contact is not None:
        xc = P.X(contact)
        yb = P.y0 + P.h
        P.s.append(_poly(["fch"], [(xc, yb), (xc - 0.45 * fs, yb + 0.8 * fs), (xc + 0.45 * fs, yb + 0.8 * fs)]))
        P.s.append(_text(["tx2"], xc - 0.35 * fs, yb - 0.45 * fs, "contact", 0.8 * fs, "end"))


def _titre_aile(s: List, x: float, y: float, aile: str, res: Dict, fs: float):
    g = res["geometrie"]
    if aile == HORIZONTALE:
        txt = f"Aile horizontale · s depuis le nu du mur · charge à X = {f0(g['X'])}" + (f" ± {f0(g['dx'])}" if g["dx"] > 0 else "")
    else:
        face = "le dessous" if res["configuration"] == "A" else "le dessus"
        txt = f"Aile verticale · s depuis {face} de l'aile horizontale · ancrage à Y = {f0(g['Y'])}" + (f" ± {f0(g['dy'])}" if g["dy"] > 0 else "")
    s.append(_text(["tx"], x, y, txt, fs, "start"))


def _axe_s(s: List, P: _Panneau, fs: float):
    s.append(_text(["tx2"], P.x0 + P.w / 2, P.y0 + P.h + 2.25 * fs, "s (mm)", 0.8 * fs))


def _titre_vertical(txt: str, x: float, y: float, taille: float) -> Dict:
    """Un titre d'axe tourné de 90° (texte à l'origine du groupe : son étendue est connue)."""
    return _group(["tx2"], [_text(["tx2"], 0.0, 0.0, txt, taille)], x=x, y=y, rot=-90)


ORDRE_SECTIONS = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
LARGEUR_PANNEAU = 40.0          # en tailles de texte : avec les marges, la vue fait ≈ 48 fs de large → 13 px dans 626 px


def moments(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """Les moments le long des deux ailes développées, hors échelle (voir l'en-tête du module)."""
    if res.get("diagrammes") is None:
        raise ValueError("pas de diagrammes dans ce résultat (calculer(…, diagrammes=False))")
    fs = float(fs) if fs else 10.0
    g = res["geometrie"]
    e, r = g["e"], g["r"]
    verifs = {v["cle"]: v for v in res["verifications"]}
    s: List[Dict] = []
    x0, W, H = 6.2 * fs, LARGEUR_PANNEAU * fs, 9.0 * fs
    y = 0.0
    for aile in (HORIZONTALE, VERTICALE):
        dg = res["diagrammes"][aile]
        fin = dg["fin"]
        vals = [p[1] for c in dg["courbes"] for p in c["points"]] + [p["M"] for p in dg["points"]] + [0.0]
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        marge_v = 0.06 * (vmax - vmin)
        vmin_t = vmin - marge_v if vmin < -1e-12 else 0.0
        vmax_t = vmax + marge_v if vmax > 1e-12 else 0.0
        _titre_aile(s, x0, y + 0.9 * fs, aile, res, fs)
        n_tr = len(dg["troncons"])
        # en B le moment est négatif : l'axe du moment nul est en haut du panneau et les flèches de la charge, qui y
        # tombent, prennent leur place au-dessus des rangées d'étiquettes
        h_fl = 2.9 * fs if (aile == HORIZONTALE and vmax_t <= 1e-12) else 0.0
        y_plot = y + 1.6 * fs + 1.9 * fs + n_tr * 1.45 * fs + 0.3 * fs + h_fl
        negatif = vmax_t <= 1e-12                   # B : la face du congé est comprimée, M < 0, les courbes occupent le bas
        P = _Panneau(s, fs, x0, y_plot, W, H, fin, vmin_t, vmax_t)
        P.zones(e, r, en_haut=negatif)
        pas_s, _ = _pas(fin, 6)
        pas_v, nd_v = _pas(max(abs(vmin_t), abs(vmax_t)), 4)
        P.cadre(pas_s, pas_v, nd_v, "M (kN·m)")
        _etiquettes_troncons(P, dg["troncons"], y_plot - 0.9 * fs - h_fl, fs)
        if aile == HORIZONTALE:
            _marques_charge(P, dg["reperes"]["charge"], fs)
        else:
            _marques_ancrage(P, dg["reperes"]["ancrage"], dg["reperes"].get("contact"), fs)
        P.sections(dg["reperes"]["sections"], ORDRE_SECTIONS, y_plot - 0.9 * fs - h_fl - n_tr * 1.45 * fs - 0.6 * fs)
        # les courbes : extrêmes en fin, nominale en moyen, l'enveloppe en gras
        for c in sorted(dg["courbes"], key=lambda c: abs(c["jeu_charge"]), reverse=True):
            P.courbe([(p[0], p[1]) for p in c["points"]], "cvm" if abs(c["jeu_charge"]) < 1e-9 else "cvf")
        P.courbe([(p["s"], p["M"]) for p in dg["points"]], "env")
        # les valeurs de l'enveloppe aux sections : au-dessus à droite du point (M > 0 : la courbe descend vers la
        # droite, l'espace y est libre) ; en B (M < 0, courbes en bas) dans la bande libre du haut, sur le trait de
        # la section ; la deuxième étiquette est poussée à droite de la première quand leurs boîtes se croisent
        secs = sorted(dg["reperes"]["sections"], key=lambda v: v["s"])
        boite = None
        for v in secs:
            vf = verifs[v["cle"]]
            txt = f"M{ORDRE_SECTIONS[v['cle']]} = {_knm(vf['M'])}"
            x_min = None
            y_fixe = P.y0 + 2.4 * fs if negatif else None
            if boite is not None:
                y_lab = y_fixe if y_fixe is not None else P.Y(vf["M"]) - 0.55 * fs
                if abs(y_lab - (boite[3] - 0.37 * fs)) < 1.45 * fs:
                    x_min = boite[2] + 0.2 * fs
            boite = P.point(v["s"], vf["M"], txt, droite=True, dessus=True, cls=("fzl",), taille=0.9, x_min=x_min, y_fixe=y_fixe)
        _axe_s(s, P, fs)
        y = y_plot + H + 3.2 * fs + 1.2 * fs
    aria = ("Moments le long des deux ailes développées : les trois courbes (une par position de la charge), l'enveloppe en gras, "
            "la tolérance de l'ancrage, N et V par tronçon, les sections 1 à 4 — hors échelle")
    return _dessin(s, fs, aria)


def contraintes(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """L'épaisseur t(s), σéq de l'enveloppe et le taux le long des deux ailes, hors échelle (voir l'en-tête du module)."""
    if res.get("diagrammes") is None:
        raise ValueError("pas de diagrammes dans ce résultat (calculer(…, diagrammes=False))")
    fs = float(fs) if fs else 10.0
    g, mat = res["geometrie"], res["materiau"]
    e, r = g["e"], g["r"]
    sadm = mat["sigma_adm"]
    verifs = {v["cle"]: v for v in res["verifications"]}
    s: List[Dict] = []
    x0, W, H, Ht = 6.2 * fs, LARGEUR_PANNEAU * fs, 8.6 * fs, 2.4 * fs
    y = 0.0
    for aile in (HORIZONTALE, VERTICALE):
        dg = res["diagrammes"][aile]
        fin = dg["fin"]
        pts = dg["points"]
        _titre_aile(s, x0, y + 0.9 * fs, aile, res, fs)
        # la bande d'épaisseur t(s) : le congé puis l'aile ; le bloc d'angle hachuré
        y_t = y + 1.6 * fs + 1.9 * fs
        Pt = _Panneau(s, fs, x0, y_t, W, Ht, fin, 0.0, (e + r) * 1.08)
        Pt.zones(e, r, etiquettes=False)
        bande = [(Pt.X(p["s"]), Pt.Y(p["t"])) for p in pts if p["t"] is not None]
        if bande:
            s.append(_poly(["pp"], [(bande[0][0], Pt.Y(0.0))] + bande + [(bande[-1][0], Pt.Y(0.0))]))
        s.append(_path(["axe"], [[(Pt.x0, Pt.y0), (Pt.x0, Pt.y0 + Ht)], [(Pt.x0, Pt.y0 + Ht), (Pt.x0 + W, Pt.y0 + Ht)]]))
        s.append(_titre_vertical("t (mm)", x0 - 3.3 * fs, y_t + Ht / 2, 0.85 * fs))
        s.append(_text(["tx2"], x0 - 0.45 * fs, Pt.Y(e) + 0.3 * fs, f0(e), 0.8 * fs, "end"))
        if r > 0:
            s.append(_text(["tx2"], x0 - 0.45 * fs, Pt.Y(e + r) + 0.3 * fs, f0(e + r), 0.8 * fs, "end"))
        Pt.sections(dg["reperes"]["sections"], ORDRE_SECTIONS)
        # σéq et le taux
        y_p = y_t + Ht + 1.0 * fs
        smax = max([p["sigma_eq"] for p in pts] + [sadm])
        P = _Panneau(s, fs, x0, y_p, W, H, fin, 0.0, smax * 1.1)
        P.zones(e, r)
        pas_s, _ = _pas(fin, 6)
        pas_v, nd_v = _pas(P.vmax, 4)
        P.cadre(pas_s, pas_v, nd_v, "σéq (N/mm²)", zero=False)
        for v in dg["reperes"]["sections"]:
            s.append(_path(["d1"], [[(P.X(v["s"]), P.y0), (P.X(v["s"]), P.y0 + P.h)]]))
        # la ligne des 100 % et l'échelle des taux, à droite : « 100 % » porte la valeur de σadm sous lui
        s.append(_path(["lim"], [[(P.X(0.0), P.Y(sadm)), (P.X(fin), P.Y(sadm))]]))
        for q in (0.5, 1.0):
            s.append(_text(["tx2"], P.x0 + W + 0.4 * fs, P.Y(q * sadm) + 0.3 * fs, f"{f0(100 * q)} %", 0.8 * fs, "start"))
        s.append(_text(["fzl"], P.x0 + W + 0.4 * fs, P.Y(sadm) + 1.35 * fs, f"σadm {_num(sadm, 1)}", 0.75 * fs, "start"))
        s.append(_titre_vertical("taux", P.x0 + W + 5.4 * fs, y_p + H / 2, 0.85 * fs))
        if aile == HORIZONTALE:
            # les positions de la charge, rappelées sur l'axe
            for xp in dg["reperes"]["charge"]["positions"]:
                nominale = abs(xp - dg["reperes"]["charge"]["X"]) < 1e-9
                s.append(_path(["fz" if nominale else "fzd"], [[(P.X(xp), P.y0 + P.h), (P.X(xp), P.y0 + P.h - 0.9 * fs)]]))
        else:
            _marques_ancrage(P, dg["reperes"]["ancrage"], dg["reperes"].get("contact"), fs)
        P.courbe([(p["s"], p["sigma_eq"]) for p in pts], "sig")
        # les sections : le taux de l'enveloppe ; le maximum et sa position
        secs = sorted(dg["reperes"]["sections"], key=lambda v: v["s"])
        for v in secs:
            vf = verifs[v["cle"]]
            au_nu = v["s"] < max(w["s"] for w in secs)        # la section au nu (basse) s'écrit au-dessus, celle au pied du congé au-dessous
            P.point(v["s"], vf["sigma_eq"], f"S{ORDRE_SECTIONS[v['cle']]} : {_pct(vf['taux'])}", droite=True, dessus=au_nu,
                    cls=("dl",), taille=0.9)
        # le maximum : marqué sur la courbe, décrit dans un bloc de deux lignes posé là où la courbe ne passe pas
        mx = res["maxima"][aile]
        xm, ym = P.X(mx["s"]), P.Y(mx["sigma_eq"])
        s.append(_circle(["gv"], xm, ym, 0.5 * fs))
        s.append(_circle(["gvd"], xm, ym, 0.2 * fs))
        ou = (f"{_num(mx['recul'], 2)} mm avant le pied du congé" if mx["dans_conge"] else
              ("au pied du congé" if abs(mx["recul"]) < 1e-6 else "dans la partie droite"))
        txt1 = f"maximum {_pct(mx['taux'])} à s = {_num(mx['s'], 1)} mm"
        txt2 = f"σéq = {_num(mx['sigma_eq'], 1)} N/mm² · {ou}"
        wb, hb = max(_tw(txt1, 0.9 * fs), _tw(txt2, 0.9 * fs)) + 0.6 * fs, 2.9 * fs
        courbe_xy = [(P.X(p["s"]), P.Y(p["sigma_eq"])) for p in pts]
        xl, yl = _placer_bloc(P, courbe_xy, wb, hb, [(xm, ym)], fs)
        # la ligne de rappel rejoint le bord du bloc le plus proche, à la hauteur du point quand elle le peut
        x_att = xl if xl > xm else (xl + wb if xl + wb < xm else xm)
        y_att = min(max(ym, yl + 0.6 * fs), yl + hb - 0.6 * fs) if x_att != xm else (yl if yl > ym else yl + hb)
        s.append(_path(["ext"], [[(xm + 0.5 * fs, ym - 0.3 * fs), (x_att, y_att)]]))
        _label(s, xl + wb / 2, yl + 0.9 * fs + 0.34 * fs, txt1, 0.9 * fs)
        _label(s, xl + wb / 2, yl + 2.35 * fs + 0.34 * fs, txt2, 0.9 * fs, cls=("dl",))
        _axe_s(s, P, fs)
        y = y_p + H + 3.2 * fs + 1.2 * fs
    aria = ("Contraintes le long des deux ailes développées : l'épaisseur t(s) du congé, σéq de l'enveloppe, la ligne des 100 %, "
            "le maximum et sa position, les sections 1 à 4 — hors échelle")
    return _dessin(s, fs, aria)


# ====================================================================== l'entrée : vues, légendes, ajustement, secours
FONCTIONS: Dict[str, Callable[..., Dessin]] = {"coupe": coupe, "moments": moments, "contraintes": contraintes, "elevation": elevation}


def legende(res: Dict, cle: str) -> str:
    """Ce que montre une vue, et le mode — le TEXTE sous le titre, partagé par l'écran et la note."""
    g, cfg, mode = res["geometrie"], res["configuration"], res["mode"]
    conf = f"configuration {cfg} ({LIBELLES_CONFIGURATION[cfg]})"
    if cle == "coupe":
        bras = ("bras du classeur Xe, Xe+r, Ye, Ye+r, YH" if mode == HISTORIQUE else
                ("bras X − uV et Y − yc" if cfg == "A" else "bras X − uV et ya − c"))
        jeux = (f"charge à X ± {f0(g['dx'])} (tirets)" if g["dx"] > 0 else "charge à X") + \
               (f", ancrage à Y ± {f0(g['dy'])} (tirets fins)" if g["dy"] > 0 else ", ancrage à Y")
        return f"{conf} · {LIBELLES_MODE[mode]} : {bras} · {jeux} · sections 1 à 4 en orange · C la résultante de compression"
    if cle == "moments":
        return (f"{conf} · {LIBELLES_MODE[mode]} · M le long de chaque aile (s depuis le dos de l'autre aile) : une courbe par position "
                f"de la charge (extrêmes en trait fin), l'enveloppe en gras, la tolérance ± {f0(g['dy'])} de l'ancrage en bande, N et V par "
                f"tronçon · M > 0 : face côté congé tendue · hors échelle")
    if cle == "contraintes":
        return (f"{conf} · {LIBELLES_MODE[mode]} · l'épaisseur t(s) du congé, σéq de l'enveloppe et le taux (ligne des 100 % : "
                f"σadm = {_num(res['materiau']['sigma_adm'], 1)} N/mm²), le maximum ◉ et sa position, les sections 1 à 4 · hors échelle")
    a = f"a = {f0(g['a'])}" if g["a"] is not None else "a = L"
    return (f"{conf} · la tranche L = {f0(g['L'])} vue de face : {g['n']} ancrage(s) à l'entraxe {f0(g['L'] / g['n'])} (le premier à "
            f"{f0(g['L'] / (2 * g['n']))}), N et V ELS par ancrage (enveloppe) · l'aile horizontale rabattue : la charge ({a}) et sa "
            f"diffusion à 45° jusqu'à la section 1, L eff = {f0(g['L_eff'])}")


def ajuster(construire: Callable[[float], Dessin], toile: Tuple[float, float], texte_px: float, fs0: float, passes: int = 6) -> Dessin:
    """Une vue reconstruite jusqu'à ce que son texte se lise à ``texte_px`` pixels dans la toile ``(largeur, hauteur)``
    en pixels (quelques passes suffisent : la boîte de vue ne dépend qu'en partie de la taille du texte)."""
    fs, d, px_prec = fs0, None, None
    for _ in range(passes):
        d = construire(fs)
        _, _, w, h = d.viewbox
        px = fs * min(toile[0] / w, toile[1] / h)
        if abs(px - texte_px) < 0.25 or (px_prec is not None and abs(px - px_prec) < 0.02 * texte_px):
            break                                  # au but — ou la taille ne bouge plus (mise en page en tailles de texte, toile trop petite)
        fs *= texte_px / px
        px_prec = px
    return d


def taille_texte(dessin: Dessin, toile: Tuple[float, float]) -> float:
    """La taille à l'écran (px) du texte courant d'un dessin posé dans la toile."""
    _, _, w, h = dessin.viewbox
    return dessin.fs * min(toile[0] / w, toile[1] / h)


def svg(dessin: Dessin, identifiant: str = "co") -> str:
    """Le SVG autonome d'une vue (style embarqué, attributs de présentation : il se peint aussi sans CSS)."""
    return vers_svg(dessin, PALETTE, "100%", identifiant)


def secours(identifiant: str, titre: str) -> str:
    """Le dessin qu'on ne sait pas faire (proportions extrêmes : une tranche de plusieurs mètres face à une aile de
    quelques centimètres) : un cadre qui le DIT, plutôt qu'un dessin illisible ou une erreur. Le calcul, lui, est rendu."""
    t = str(titre).replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    return (f'<svg id="{identifiant}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 220" width="100%" role="img" '
            f'aria-label="{t} : dessin impossible à cette échelle">'
            '<rect x="20" y="20" width="560" height="180" rx="12" fill="none" stroke="#7F8C9A" stroke-width="1.2" stroke-dasharray="6 4"/>'
            '<text x="300" y="104" text-anchor="middle" font-size="20" font-weight="600" fill="#33415C" font-family="system-ui,Arial,sans-serif">'
            'Dessin impossible à cette échelle</text>'
            '<text x="300" y="138" text-anchor="middle" font-size="15" fill="#5B6878" font-family="system-ui,Arial,sans-serif">'
            'les dimensions saisies sont trop disproportionnées — le calcul reste valable</text></svg>')

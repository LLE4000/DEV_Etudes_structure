# -*- coding: utf-8 -*-
"""Le DESSIN du formulaire de poutres (api_web/biblio_poutres.py) : le schéma statique (appuis, charges, réactions,
cotes), puis V, M et la flèche alignés sous lui, et le curseur de la section lue. Du SVG en unités d'écran (largeur 640),
coloré par les classes du thème (css/bibliotheque-autres.css). Rien n'est calculé ici : les points viennent du moteur
(``Solution.courbes``, ``extremes``, ``appuis``) ; ce fichier les POSE.

Une charge se GLISSE le long de la poutre : son groupe porte ``data-glisser`` — les paramètres que le geste règle, leur
valeur de départ, l'échelle (unités du dessin par mètre), le pas et les bornes ; js/biblio.js fait le reste.
"""
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from api_web import commun as C

W = 640.0
ML, MR = 56.0, 28.0
PW = W - ML - MR
H_DIAG = 86.0
MOINS = "−"
_IND = re.compile(r"([A-Za-zθωσ])_([A-Za-z0-9,]+)")


def _fr(x: float, nd: int = 2) -> str:
    return C.fr(x, nd).replace("-", MOINS)


def texte(x: float, y: float, s: str, cls: str = "", ancre: str = "middle") -> str:
    """Un texte du dessin ; « R_A » y devient R avec A en indice."""
    parts, pos = "", 0
    for mm in _IND.finditer(s):
        parts += C.esc(s[pos:mm.start()]) + C.esc(mm.group(1)) + f'<tspan class="ind" dy="3">{C.esc(mm.group(2))}</tspan><tspan dy="-3">​</tspan>'
        pos = mm.end()
    parts += C.esc(s[pos:])
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{ancre}"{(" class=" + chr(34) + cls + chr(34)) if cls else ""}>{parts}</text>'


def _fleche(x: float, y1: float, y2: float, cls: str) -> str:
    """Une flèche verticale de y1 vers y2 (la pointe en y2)."""
    s = 1.0 if y2 > y1 else -1.0
    return (f'<path class="{cls}" d="M{x:.1f} {y1:.1f}V{y2 - s * 1.0:.1f}"/>'
            f'<path class="{cls} pte" d="M{x:.1f} {y2:.1f}l-3.6 {-s * 7.5:.1f}h7.2z"/>')


def _pas(Lt: float) -> float:
    return 0.05 if Lt <= 12 else (0.1 if Lt <= 40 else 0.25)


def _glisser(params: Dict, ech: float, Lt: float) -> str:
    d = {"x": params, "ech": round(ech, 4), "pas": _pas(Lt), "min": 0, "max": round(Lt, 6), "unite": "m"}
    return f' data-glisser="{C.esc(json.dumps(d, ensure_ascii=False))}"'


class _Etiquettes:
    """Le placement des étiquettes des charges : chacune prend le premier NIVEAU (au-dessus de la poutre) où elle ne
    recouvre aucune autre."""

    def __init__(self):
        self.niveaux: List[List[Tuple[float, float]]] = []

    def placer(self, xc: float, largeur: float, mini: int = 0) -> int:
        a, b = xc - largeur / 2 - 4, xc + largeur / 2 + 4
        k = mini
        while True:
            while len(self.niveaux) <= k:
                self.niveaux.append([])
            if all(b <= u or a >= v for u, v in self.niveaux[k]):
                self.niveaux[k].append((a, b))
                return k
            k += 1


def schema(sol, charges: Sequence[Dict], noms_charges: Sequence[str]) -> Tuple[str, float, float]:
    """Le schéma statique : (SVG intérieur, ordonnée du bas, ordonnée de la poutre). ``charges`` : les charges
    normalisées du moteur (abscisses globales) ; ``noms_charges`` : « q_1 », « P_2 »… dans le même ordre.

    Les ÉTIQUETTES des charges vivent toutes dans une même zone, au-dessus de la plus haute charge : aucune n'est écrite
    sur une autre charge. La poutre descend d'autant de niveaux qu'il en faut."""
    Lt = sol.L
    ech = PW / Lt
    X = lambda x: ML + x * ech                                                       # noqa: E731
    out = []
    vmax = max((abs(c["v"]) for c in charges if c["type"] in ("q", "tri")), default=1.0) or 1.0
    et = _Etiquettes()
    base = max([{"q": 42.0, "tri": 42.0, "P": 40.0, "C": 24.0}[c["type"]] for c in charges], default=0.0)
    labs = []
    for c, nom in zip(charges, noms_charges):
        lab = f"{nom} = {_fr(c['v'])} " + {"q": "kN/m", "tri": "kN/m", "P": "kN", "C": "kN·m"}[c["type"]]
        larg = 7.0 * len(lab)
        xc = (X(c["a"]) + X(c["b"])) / 2 if c["type"] in ("q", "tri") else X(c["a"])
        xc = min(W - 4 - larg / 2, max(4 + larg / 2, xc))
        labs.append((lab, xc, et.placer(xc, larg)))
    niveaux = max((x[2] for x in labs), default=-1) + 1
    yb = (18.0 + 15.0 * niveaux + base + 8.0) if charges else 30.0
    y_lab = lambda niv: yb - base - 8.0 - 15.0 * niv                               # noqa: E731
    # ---- les charges
    for i, (c, nom) in enumerate(zip(charges, noms_charges)):
        t, v = c["type"], c["v"]
        k = i + 1
        bas = v >= 0
        lab, xl, niv = labs[i]
        ty = y_lab(niv)
        titre = f'<title>{C.esc(nom.replace("_", ""))} : glisser pour déplacer</title>'
        if t in ("q", "tri"):
            xa, xb = X(c["a"]), X(c["b"])
            h = 15.0 + 24.0 * abs(v) / vmax
            ha, hb = (h, h) if t == "q" else ((0.0, h) if c["sens"] == "croissante" else (h, 0.0))
            fl = ""
            nb_ = max(2, int((xb - xa) / 26.0) + 1)
            for j in range(nb_):
                xj = xa + (xb - xa) * j / (nb_ - 1)
                hj = ha + (hb - ha) * j / (nb_ - 1)
                if hj > 9:
                    fl += _fleche(xj, yb - 3 - hj, yb - 3, "ch") if bas else _fleche(xj, yb - 3, yb - 3 - hj, "ch")
            poly = f'<path class="ch-b" d="M{xa:.1f} {yb - 3:.1f}L{xa:.1f} {yb - 3 - ha:.1f}L{xb:.1f} {yb - 3 - hb:.1f}L{xb:.1f} {yb - 3:.1f}Z"/>'
            out.append(f'<g class="charge glis" data-charge="{k}"{_glisser({f"a{k}": c["a"], f"b{k}": c["b"]}, ech, Lt)}>{titre}'
                       f'{poly}{fl}{texte(xl, ty, lab, "ch-t")}</g>')
        elif t == "P":
            x = X(c["a"])
            top = ty + 5
            f_ = _fleche(x, top, yb - 3, "ch fo") if bas else _fleche(x, yb - 3, top, "ch fo")
            out.append(f'<g class="charge glis" data-charge="{k}"{_glisser({f"a{k}": c["a"]}, ech, Lt)}>{titre}'
                       f'<rect class="prise" x="{x - 11:.1f}" y="{top - 16:.1f}" width="22" height="{yb - top + 14:.1f}"/>{f_}{texte(xl, ty, lab, "ch-t")}</g>')
        else:
            x = X(c["a"])
            ry = 15.0
            # un arc de 270° autour du point d'application, la pointe dans le sens du couple (horaire si positif)
            a0, a1 = (math.radians(200), math.radians(-70)) if bas else (math.radians(-20), math.radians(250))
            p0 = (x + ry * math.cos(a0), yb - ry * math.sin(a0))
            p1 = (x + ry * math.cos(a1), yb - ry * math.sin(a1))
            arc = f'<path class="ch fo" fill="none" d="M{p0[0]:.1f} {p0[1]:.1f}A{ry} {ry} 0 1 {1 if bas else 0} {p1[0]:.1f} {p1[1]:.1f}"/>'
            tang = a1 + (-math.pi / 2 if bas else math.pi / 2)
            q0 = (p1[0] - 7.5 * math.cos(tang), p1[1] + 7.5 * math.sin(tang))
            nx, ny = -math.sin(tang), -math.cos(tang)
            pte = (f'<path class="ch pte" d="M{p1[0]:.1f} {p1[1]:.1f}L{q0[0] + 3.6 * nx:.1f} {q0[1] + 3.6 * ny:.1f}L{q0[0] - 3.6 * nx:.1f} {q0[1] - 3.6 * ny:.1f}Z"/>')
            rappel = f'<path class="ch-r" d="M{x:.1f} {yb - ry - 3:.1f}V{ty + 5:.1f}"/>' if ty + 5 < yb - ry - 6 else ""
            out.append(f'<g class="charge glis" data-charge="{k}"{_glisser({f"a{k}": c["a"]}, ech, Lt)}>{titre}'
                       f'<rect class="prise" x="{x - 20:.1f}" y="{yb - 22:.1f}" width="40" height="44"/>{rappel}{arc}{pte}'
                       f'<circle class="ch-p" cx="{x:.1f}" cy="{yb:.1f}" r="2.6"/>{texte(xl, ty, lab, "ch-t")}</g>')
    # ---- la poutre et ses appuis
    out.append(f'<path class="poutre" d="M{X(0):.1f} {yb:.1f}H{X(Lt):.1f}"/>')
    for a in sol.appuis:
        x = X(a["x"])
        if a["type"] == "encastrement":
            g = a["x"] <= 0
            x0 = x - 9 if g else x
            hach = "".join(f'<path class="sol" d="M{x0 + (0 if g else 9):.1f} {yb - 20 + 8 * j:.1f}l{-9 if g else 9} 7"/>' for j in range(5))
            out.append(f'<g class="appui"><path class="mur" d="M{x:.1f} {yb - 22:.1f}V{yb + 22:.1f}"/>{hach}</g>')
        else:
            out.append(f'<g class="appui"><path class="tri" d="M{x:.1f} {yb + 2:.1f}l-9 15h18z"/>'
                       f'<path class="sol" d="M{x - 13:.1f} {yb + 20:.1f}h26"/></g>')
        out.append(texte(x, yb + 36, a["nom"], "nom"))
    # ---- les réactions (et les moments d'encastrement)
    n_app = len(sol.appuis)
    for j, a in enumerate(sol.appuis):
        x = X(a["x"])
        ancre = "start" if j == 0 else ("end" if j == n_app - 1 and n_app > 1 else "middle")
        xt = x - 14 if ancre == "start" else (x + 14 if ancre == "end" else x)
        haut = a["R"] >= 0
        out.append(_fleche(x, yb + 68, yb + 44, "re") if haut else _fleche(x, yb + 44, yb + 68, "re"))
        out.append(texte(xt, yb + 82, f'R_{a["nom"]} = {_fr(a["R"])} kN', "re-t", ancre))
        if a["type"] == "encastrement":
            out.append(texte(xt, yb + 96, f'M_{a["nom"]} = {_fr(a["M"])} kN·m', "re-t", ancre))
    # ---- les cotes
    yc = yb + 116
    for t in sol.travees:
        xa, xb = X(t.x0), X(t.x0 + t.L)
        out.append(f'<path class="cote" d="M{xa:.1f} {yc:.1f}H{xb:.1f}M{xa:.1f} {yc - 5:.1f}v10M{xb:.1f} {yc - 5:.1f}v10"/>')
        nomL = "L" if len(sol.travees) == 1 else f"L_{sol.travees.index(t) + 1}"
        out.append(texte((xa + xb) / 2, yc - 5, f"{nomL} = {_fr(t.L)} m", "cote-t"))
    return "".join(out), yc + 10, yb


def _diagramme(titre: str, unite: str, pts: Sequence[Tuple[float, float]], y_haut: float, positif_bas: bool, cls: str,
               marques: Sequence[Tuple[float, float]], Lt: float, nd: int = 2, sauts: Sequence[float] = ()) -> str:
    """Un diagramme : l'axe, l'aire, la courbe, les valeurs marquées (extremums). ``pts`` : (x global, valeur)."""
    ech = PW / Lt
    X = lambda x: ML + x * ech                                                       # noqa: E731
    vals = [v for _, v in pts] + [0.0]
    lo, hi = min(vals), max(vals)
    etendue = (hi - lo) or 1.0
    k = H_DIAG / etendue
    Y = (lambda v: y_haut + (v - lo) * k) if positif_bas else (lambda v: y_haut + (hi - v) * k)       # noqa: E731
    y0 = Y(0.0)
    chemin = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f} {Y(v):.1f}" for i, (x, v) in enumerate(pts))
    aire = f"M{X(pts[0][0]):.1f} {y0:.1f} " + " ".join(f"L{X(x):.1f} {Y(v):.1f}" for x, v in pts) + f" L{X(pts[-1][0]):.1f} {y0:.1f}Z"
    out = [f'<g class="diag {cls}">', texte(8, y_haut + H_DIAG / 2 - 3, titre, "dt", "start"), texte(8, y_haut + H_DIAG / 2 + 11, unite, "du", "start"),
           f'<path class="aire" d="{aire}"/>', f'<path class="axe" d="M{X(0):.1f} {y0:.1f}H{X(Lt):.1f}"/>', f'<path class="courbe" d="{chemin}"/>']
    vus = []
    for xm, vm in marques:
        if abs(vm) < 1e-9 * max(abs(lo), abs(hi), 1e-300) or any(abs(xm - u) < 1e-6 * Lt and abs(vm - w) < 1e-9 * etendue for u, w in vus):
            continue
        vus.append((xm, vm))
        x, y = X(xm), Y(vm)
        dessus = (y < y0) or (y == y0 and not positif_bas)
        ancre = "start" if x < ML + 60 else ("end" if x > W - MR - 60 else "middle")
        dx = 5 if ancre == "start" else (-5 if ancre == "end" else 0)
        out.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="2.8"/>')
        out.append(texte(x + dx, y - 6 if dessus else y + 14, _fr(vm, nd), "val", ancre))
    out.append("</g>")
    return "".join(out)


def dessin(sol, charges: Sequence[Dict], noms_charges: Sequence[str], x_lu: Optional[float], unite_f: str, k_f: float, nd_f: int) -> str:
    """Le dessin complet. ``k_f`` : le facteur qui change E·I·f en la flèche affichée (1000/E·I pour des mm, 1 sans
    rigidité) ; ``unite_f`` : son unité."""
    Lt = sol.L
    corps, y, yb = schema(sol, charges, noms_charges)
    co = sol.courbes(48)
    pV = [(p[0], p[1]) for t in co for p in t["points"]]
    pM = [(p[0], p[2]) for t in co for p in t["points"]]
    pf = [(p[0], p[3] * k_f) for t in co for p in t["points"]]
    e = sol.extremes
    yV = y + 30
    yM = yV + H_DIAG + 46
    yf = yM + H_DIAG + 46
    mV = [(e["V_max"][1], e["V_max"][0]), (e["V_min"][1], e["V_min"][0])]
    mM = [(tt[k][1], tt[k][0]) for tt in e["travees"] for k in ("M_max", "M_min")]
    mf = [(tt[k][1], tt[k][0] * k_f) for tt in e["travees"] for k in ("f_max", "f_min")]
    corps += _diagramme("V", "kN", pV, yV, False, "dV", mV, Lt)
    corps += _diagramme("M", "kN·m", pM, yM, True, "dM", mM, Lt)
    corps += _diagramme("f" if unite_f == "mm" else "E·I·f", unite_f, pf, yf, True, "df", mf, Lt, nd_f)
    H = yf + H_DIAG + 36
    if x_lu is not None:
        xs = ML + x_lu * PW / Lt
        corps += (f'<g class="lu"><path d="M{xs:.1f} {yb - 6:.1f}V{H - 16:.1f}"/>'
                  f'{texte(xs, H - 2, f"x = {_fr(x_lu)} m", "lu-t", "start" if xs < ML + 50 else ("end" if xs > W - MR - 50 else "middle"))}</g>')
    return (f'<svg class="rdm-svg rdm-svg-poutre" viewBox="0 0 {W:.0f} {H + 6:.0f}" role="img" aria-label="Schéma de la poutre, effort tranchant, '
            f'moment fléchissant et flèche">{corps}</svg>')

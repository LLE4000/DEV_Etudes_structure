# -*- coding: utf-8 -*-
"""Le PLAN FERRAILLÉ et les COUPES A-A et B-B d'une fondation de grue (lot 4d, § 6 du cahier des charges du
15/11/2026) — l'onglet « Ferraillage », l'onglet « Coupes » et la note PDF.

Ce fichier ne calcule rien : il dessine le ferraillage (``saf.fondation_grue.ferraillage.calculer``) avec les repères du
bordereau (T1 à T4, R, L, U, C, V). Les couleurs sont celles du plan (api_web/dessin_saf.py) ; la face inférieure en
bleu, la face supérieure en rouge.

    plan(el, m, fer, ef, fond=None, ident="fer")   -> le plan ferraillé (SVG), à l'échelle, coté ; ``fond`` : une
                                                       face-direction (« inf_x »…) dont le taux de travail du treillis
                                                       seul est peint, avec la ligne noire à 100 %
    coupes(el, m, fer, ef)                          -> les coupes A-A (le long de BX1) et B-B (le long de BY1), à l'échelle
    reperes_plan(svg) / reperes_coupes(svg)         -> les repères écrits (tests D1, D2)
"""
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from saf.fondation_grue import ferraillage as FE

from . import commun as C
from . import dessin_saf as DS

ECH = DS.ECH
INF, SUP = "#2c6cb0", "#b4384e"                  # les deux faces
COULEUR = {"inf": INF, "sup": SUP}
N = C.NBSP
FONDS = (("", "Sans fond"), ("inf_x", "Taux treillis inf. x"), ("inf_y", "Taux treillis inf. y"), ("sup_x", "Taux treillis sup. x"),
         ("sup_y", "Taux treillis sup. y"))


def _bandes(fer: Dict) -> Dict[str, Dict]:
    return {ln["rep"]: ln for ln in fer["bordereau"]["lignes"]}


def _rep(fer: Dict, famille: str, **cle) -> List[str]:
    """Les repères du bordereau d'une famille, filtrés par des clés (cote, sens, bande…)."""
    out = []
    for ln in fer["bordereau"]["lignes"]:
        if ln["famille"] == famille and all(ln.get(k) == v for k, v in cle.items()) and ln["rep"] not in out:
            out.append(ln["rep"])
    return out


def _quad(b: Dict, t0: float, t1: float, s0: float = None, s1: float = None) -> List[Tuple[float, float]]:
    """Le quadrilatère d'une bande entre les distances t0, t1 de son axe (et les abscisses s0, s1)."""
    p0, u = np.asarray(b["p0"], float), np.asarray(b["u"], float)
    n = np.array([-u[1], u[0]])
    s0 = 0.0 if s0 is None else s0
    s1 = b["L"] if s1 is None else s1
    return [tuple(p0 + s0 * u + t0 * n), tuple(p0 + s1 * u + t0 * n), tuple(p0 + s1 * u + t1 * n), tuple(p0 + s0 * u + t1 * n)]


def _hachures(ident: str, t: float) -> str:
    """Les deux hachures : « \\\\ » pour le bas, « // » pour le haut."""
    pas = t * 0.9
    return (f'<pattern id="{ident}_hinf" width="{pas:.1f}" height="{pas:.1f}" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="{pas:.1f}" stroke="{INF}" stroke-width="{t * 0.07:.2f}" stroke-opacity="0.65"/></pattern>'
            f'<pattern id="{ident}_hsup" width="{pas:.1f}" height="{pas:.1f}" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">'
            f'<line x1="0" y1="0" x2="0" y2="{pas:.1f}" stroke="{SUP}" stroke-width="{t * 0.07:.2f}" stroke-opacity="0.65"/></pattern>')


def _taux_fond(fer: Dict, ef: Dict, fond: str) -> Optional[Dict]:
    """Le champ du taux de travail du TREILLIS SEUL (besoin ÷ treillis) d'une face-direction, au centre des éléments
    (zones de renfort et plots coulés : 0) ; la ligne noire à 100 % montre où il faut des renforts (§ 2.3)."""
    if not fond:
        return None
    cote, sens = fond.split("_")
    cv = fer["faces"][cote]["couverture"][sens]
    val = np.where(cv["controle"], np.asarray(cv["taux_treillis"], float), np.nan)
    return {"mode": "taux", "par_element": True, "valeurs": val, "decolle": None, "limite": 1.0,
            "unite": ("%", 0.01, 0), "symbole": f"taux treillis {cote}. {sens}", "maxi": float(np.nanmax(val)) if np.isfinite(val).any() else 0.0}


def _ligne_100(v, ef: Dict, valeurs: np.ndarray, ident: str) -> str:
    """La ligne noire à 100 % d'un champ par élément : les côtés communs à un élément au-dessus et à un élément en dessous
    (ou au bord de la dalle)."""
    ch = ef["champ"]
    I, J = ch["I"], ch["J"]
    rouge = {(int(i), int(j)) for i, j, x in zip(I, J, valeurs) if np.isfinite(x) and x > 1.0 + 1e-9}
    if not rouge:
        return ""
    a, b, x0, y0 = ch["a"], ch["b"], ch["x0"], ch["y0"]
    d = ""
    for i, j in rouge:
        for di, dj, seg in ((1, 0, ((1, 0), (1, 1))), (-1, 0, ((0, 0), (0, 1))), (0, 1, ((0, 1), (1, 1))), (0, -1, ((0, 0), (1, 0)))):
            if (i + di, j + dj) not in rouge:
                (u0, w0), (u1, w1) = seg
                d += (f"M{v.X(x0 + (i + u0) * a):.1f} {v.Y(y0 + (j + w0) * b):.1f}L{v.X(x0 + (i + u1) * a):.1f} {v.Y(y0 + (j + w1) * b):.1f}")
    return (f'<path class="df-100" d="{d}" fill="none" stroke="{DS.LIMITE}" stroke-width="{v.t * 0.09:.2f}" clip-path="url(#{ident}_dalle)" '
            f'pointer-events="none"/>')


def plan(el: Dict, m: Dict, fer: Dict, ef: Dict, fond: Optional[str] = None, ident: str = "fer", carte: str = "image") -> str:
    """Le plan ferraillé (§ 6.1) : le contour, les plaques ou les plots, les axes du châssis ; les treillis (texte et un
    carré de grille) ; les renforts de bandes (zone hachurée, quelques barres, étiquette) ; les renforts locaux (le
    rectangle, le repère) ; les U le long du bord ; la cage des plots coulés ; les repères des coupes A-A et B-B ; les
    cotes. Toutes les barres du bordereau y portent leur repère (test D1)."""
    c = m["contour"]
    pl = m.get("plots")
    v = DS._Vue(c, list(m["plaques"]) + (pl["polys"] if pl else []))
    t = v.t
    out = [f'<defs><clipPath id="{ident}_dalle"><polygon points="{v.pts(c)}"/></clipPath>{_hachures(ident, t)}</defs>',
           f'<polygon points="{v.pts(c)}" fill="{DS.BETON}" stroke="none"/>']
    champ = _taux_fond(fer, ef, fond)
    if champ is not None:
        champ["dalle"] = _dalle_de(ef)
        out.append(f'<g class="sf-carte" pointer-events="none">{DS._carte_image(v, champ, ident) if carte == "image" else DS._carte_vecteur(v, champ)}</g>')
        out.append(_ligne_100(v, ef, champ["valeurs"], ident))
    out.append(f'<polygon points="{v.pts(c)}" fill="none" stroke="{DS.BETON_BORD}" stroke-width="{t * 0.11:.2f}" stroke-linejoin="round"/>')
    # ---- les axes du châssis, les plaques, les plots
    pieds = m["pieds"]
    for i in range(4):
        a, b = pieds[i], pieds[(i + 1) % 4]
        out.append(f'<line x1="{v.X(a[0]):.1f}" y1="{v.Y(a[1]):.1f}" x2="{v.X(b[0]):.1f}" y2="{v.Y(b[1]):.1f}" stroke="{DS.AXE}" '
                   f'stroke-width="{t * 0.04:.2f}" stroke-dasharray="{t * 0.9:.1f} {t * 0.3:.1f} {t * 0.12:.1f} {t * 0.3:.1f}"/>')
    if pl:
        for q in pl["polys"]:
            out.append(f'<polygon points="{v.pts(q)}" fill="{DS.PLOT_FOND if pl["type"] == "coule" else "none"}" fill-opacity="0.55" '
                       f'stroke="{DS.PLOT}" stroke-width="{t * 0.07:.2f}"/>')
    for q in m["plaques"]:
        out.append(f'<polygon points="{v.pts(q)}" fill="{DS.APPUI_FOND}" stroke="{DS.APPUI}" stroke-width="{t * 0.07:.2f}"/>')
    # ---- les renforts de bandes : la zone hachurée, quelques barres, l'étiquette
    noms = {b["nom"]: b for b in ef["bandes"]}
    for cote in ("inf", "sup"):
        for nom, bf in fer["faces"][cote]["bandes"].items():
            r = bf["renfort"]
            if not r:
                continue
            b = noms[nom]
            z0, z1 = b["zone"]
            cc = el["armatures"]["enrobage"]
            out.append(f'<polygon class="df-r" points="{v.pts(_quad(b, z0, z1, cc, b["L"] - cc))}" fill="url(#{ident}_h{cote})" stroke="{COULEUR[cote]}" '
                       f'stroke-width="{t * 0.05:.2f}" clip-path="url(#{ident}_dalle)"/>')
            for k in range(min(r["n"], 3)):                          # quelques barres dessinées
                tt = z0 + (k + 0.5) * (z1 - z0) / min(r["n"], 3)
                q = _quad(b, tt, tt, cc, b["L"] - cc)
                out.append(f'<line x1="{v.X(q[0][0]):.1f}" y1="{v.Y(q[0][1]):.1f}" x2="{v.X(q[1][0]):.1f}" y2="{v.Y(q[1][1]):.1f}" '
                           f'stroke="{COULEUR[cote]}" stroke-width="{max(r["d"] / 1000 * ECH, t * 0.08):.2f}" stroke-linecap="round"/>')
            rep = ", ".join(_rep(fer, "R", bande=nom, cote=cote))
            txt = f"{rep} : {r['n']}{N}Ø{r['d']} {cote}. – L{N}={N}{C.fr(r['L'], 2)}{N}m"
            sens = 1 if cote == "inf" else -1
            tt = (z1 if sens > 0 else z0) + sens * t * 0.55 / ECH
            p = _quad(b, tt, tt, b["L"] * (0.30 if cote == "inf" else 0.70), b["L"] * (0.30 if cote == "inf" else 0.70))[0]
            ang = DS._lisible(math.degrees(math.atan2(-b["u"][1], b["u"][0])))
            out.append(DS._texte(v.X(p[0]), v.Y(p[1]), txt, t * 0.72, COULEUR[cote], "middle", gras=True, angle=ang, attrs=f' data-rep="{rep}"'))
    # ---- les renforts locaux : le rectangle, le repère
    locaux = [ln for ln in fer["bordereau"]["lignes"] if ln["famille"] == "L"]
    vus = set()
    for ln in locaux:
        if ln["rep"] in vus:
            continue
        vus.add(ln["rep"])
        x0, y0, x1, y1 = ln["rect"]
        col = COULEUR[ln["cote"]]
        out.append(f'<rect class="df-l" x="{v.X(x0):.1f}" y="{v.Y(y1):.1f}" width="{(x1 - x0) * ECH:.1f}" height="{(y1 - y0) * ECH:.1f}" fill="{col}" '
                   f'fill-opacity="0.06" stroke="{col}" stroke-width="{t * 0.05:.2f}" stroke-dasharray="{t * 0.3:.1f} {t * 0.2:.1f}"/>')
        out.append(DS._texte(v.X((x0 + x1) / 2), v.Y((y0 + y1) / 2) + t * 0.25, ln["rep"], t * 0.62, col, "middle", gras=True,
                             attrs=f' data-rep="{ln["rep"]}"'))
    # ---- les treillis : le texte, un carré de grille ; les U ; les cages ; les C d'effort tranchant
    x_txt, y_txt = v.X(v.x0) + t * 0.6, v.Y(v.y1) - 0.6 * ECH - t * 2.4      # (au-dessus des lettres des coupes)
    lignes = []
    for cote, (r1, r2) in (("inf", ("T1", "T2")), ("sup", ("T3", "T4"))):
        f = fer["faces"][cote]
        lignes.append((f"Treillis {cote}. {FE.libelle_treillis(f['code'])} (x et y) – {r1}, {r2}", COULEUR[cote], f"{r1},{r2}"))
    u = fer["U"]
    par = sorted({b_["n"] for b_ in u["bords"]})
    lignes.append((f"U1 : Ø{u['d']} e{N}={N}{C.fr(u['e'], 0)}{N}cm – " + (f"{par[0]} U par bord" if len(par) == 1 else f"{u['n']} U en tout"), DS.ENCRE, "U1"))
    reps_c = _rep(fer, "C")
    if reps_c:
        cg = fer["cages"][0]
        lignes.append((f"{', '.join(reps_c)} : cage des plots Ø{cg['d']} e{N}={N}{C.fr(cg['e'], 0)}{N}cm (C sur 4 faces, 2 U en tête)", DS.PLOT, ",".join(reps_c)))
    reps_v = _rep(fer, "V")
    for rv in reps_v:
        ln = next(x for x in fer["bordereau"]["lignes"] if x["rep"] == rv)
        lignes.append((f"{rv} : {ln['n']} C Ø{ln['d']} e{N}={N}{C.fr(ln['s'], 1)}{N}cm (effort tranchant {ln['bande']})", DS.ENCRE, rv))
    for k, (txt, col, rep) in enumerate(lignes):
        yy = y_txt - (len(lignes) - 1 - k) * t * 1.05
        if k < 2:
            g = t * 0.75
            out.append(f'<path d="M{x_txt:.1f} {yy - g:.1f}h{g:.1f}M{x_txt:.1f} {yy - g / 2:.1f}h{g:.1f}M{x_txt:.1f} {yy:.1f}h{g:.1f}'
                       f'M{x_txt:.1f} {yy - g:.1f}v{g:.1f}M{x_txt + g / 2:.1f} {yy - g:.1f}v{g:.1f}M{x_txt + g:.1f} {yy - g:.1f}v{g:.1f}" '
                       f'stroke="{col}" stroke-width="{t * 0.06:.2f}" fill="none"/>')
        out.append(DS._texte(x_txt + t * 1.2, yy, txt, t * 0.72, col, "start", gras=True, attrs=f' data-rep="{rep}"'))
    # (les U dessinés le long du bord : un trait court par U, au pas, sur le bord bas)
    for b_ in u["bords"][:1]:
        p, q = b_["de"], b_["a"]
        L = b_["L"]
        for k in range(0, b_["n"], max(1, b_["n"] // 12)):
            s_ = el["armatures"]["enrobage"] + k * u["e"] / 100.0
            x, y = p[0] + (q[0] - p[0]) * s_ / L, p[1] + (q[1] - p[1]) * s_ / L
            out.append(f'<circle cx="{v.X(x):.1f}" cy="{v.Y(y):.1f}" r="{t * 0.12:.1f}" fill="{DS.ENCRE}"/>')
    # ---- les repères des coupes A-A (le long de BX1) et B-B (le long de BY1)
    for nom, lettre in (("BX1", "A"), ("BY1", "B")):
        if nom not in noms:
            continue
        b = noms[nom]
        p0 = np.asarray(b["p0"], float) - np.asarray(b["u"], float) * 0.6
        p1 = np.asarray(b["p0"], float) + np.asarray(b["u"], float) * (b["L"] + 0.6)
        out.append(f'<line class="df-coupe" x1="{v.X(p0[0]):.1f}" y1="{v.Y(p0[1]):.1f}" x2="{v.X(p1[0]):.1f}" y2="{v.Y(p1[1]):.1f}" stroke="{DS.ENCRE}" '
                   f'stroke-width="{t * 0.06:.2f}" stroke-dasharray="{t * 1.2:.1f} {t * 0.35:.1f} {t * 0.2:.1f} {t * 0.35:.1f}"/>')
        for p in (p0, p1):
            out.append(DS._texte(v.X(p[0]), v.Y(p[1]) - t * 0.3, lettre, t * 1.05, DS.ENCRE, "middle", gras=True, attrs=f' data-coupe="{lettre}"'))
    # ---- les cotes, la feuille
    out.append(DS._cote(v, (v.x0, v.y0), (v.x1, v.y0), -t * 2.6, f"{C.fr(v.x1 - v.x0, 2)}"))
    out.append(DS._cote(v, (v.x0, v.y1), (v.x0, v.y0), -t * 2.6, f"{C.fr(v.y1 - v.y0, 2)}"))
    if champ is not None:
        out.append(f'<g pointer-events="none">{DS._legende(v, champ, v.X(v.x1) + 0.6 * ECH + t * 1.8, v.Y(v.y1) + t * 1.2, min(v.Y(v.y0) - v.Y(v.y1) - t * 4.0, t * 16))}</g>')
    x_min, y_min = v.X(v.gx0) - t * 4.6, y_txt - t * (1.6 + 1.05 * (len(lignes) - 1))
    x_max = v.X(v.gx1) + 0.6 * ECH + t * (10.5 if champ is not None else 2.0)
    y_max = v.Y(v.gy0) + t * 4.4
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x_min:.1f} {y_min:.1f} {x_max - x_min:.1f} {y_max - y_min:.1f}" role="img" '
            f'aria-label="Plan ferraillé de la fondation" class="sf-plan df-plan">{"".join(out)}</svg>')


def _dalle_de(ef: Dict):
    """La dalle maillée d'une enveloppe : un objet à la mode de ``plaque.Dalle`` pour la carte par élément (I, J, a, b,
    x0, y0, nx, ny)."""
    ch = ef["champ"]

    class _D:
        pass
    d = _D()
    d.I, d.J, d.a, d.b, d.x0, d.y0 = ch["I"], ch["J"], ch["a"], ch["b"], ch["x0"], ch["y0"]
    d.nx, d.ny = int(ch["I"].max()) + 1, int(ch["J"].max()) + 1
    d.on0 = None
    return d


# ------------------------------------------------------------------ les coupes
def _coupe(el: Dict, m: Dict, fer: Dict, ef: Dict, nom_bande: str, lettre: str, y_haut: float, largeur: float) -> Tuple[str, float]:
    """Une coupe le long d'une bande (§ 6.2), à l'échelle : le béton (dalle, plots coulés), les treillis (traits pour les
    barres dans le sens de la coupe, points pour l'autre sens), les renforts, les U aux deux bouts, les C des plots
    coulés, les C d'effort tranchant, les repères ; cotées : h, hauteur des plots, longueur, enrobage. Rend (svg, hauteur)."""
    b = next(x for x in ef["bandes"] if x["nom"] == nom_bande)
    sens = "x" if nom_bande.startswith("BX") else "y"
    autre = "y" if sens == "x" else "x"
    p0, u = np.asarray(b["p0"], float), np.asarray(b["u"], float)
    L = b["L"]
    h, c = fer["h"], fer["enrobage"]
    hp = fer["hp"]
    ech = largeur / max(L, 1e-9)
    t = max(L * ECH, 300.0) * 0.024 * (largeur / (L * ECH))
    X = lambda s: s * ech                                        # noqa: E731
    Z = lambda z: y_haut + (h + hp + 0.4) * ech - z * ech        # noqa: E731
    out = []
    out.append(DS._texte(X(0), y_haut - t * 0.6, f"Coupe {lettre}-{lettre} (le long de {nom_bande})", t * 0.95, DS.ENCRE, "start", gras=True))
    out.append(f'<rect x="{X(0):.1f}" y="{Z(h):.1f}" width="{X(L):.1f}" height="{h * ech:.1f}" fill="{DS.BETON}" stroke="{DS.BETON_BORD}" stroke-width="{t * 0.08:.2f}"/>')
    plots_s = []
    if fer["coule"]:
        for s1, s2 in b["charge"]:
            plots_s.append((s1, s2))
            out.append(f'<rect x="{X(s1):.1f}" y="{Z(h + hp):.1f}" width="{X(s2 - s1):.1f}" height="{hp * ech:.1f}" fill="{DS.PLOT_FOND}" '
                       f'stroke="{DS.BETON_BORD}" stroke-width="{t * 0.08:.2f}"/>')
    reps = []

    def barre(s0, s1, z, d, col, rep):
        reps.append(rep)
        return (f'<line x1="{X(s0):.1f}" y1="{Z(z):.1f}" x2="{X(s1):.1f}" y2="{Z(z):.1f}" stroke="{col}" stroke-width="{max(d / 1000 * ech, t * 0.07):.2f}" '
                f'stroke-linecap="round" data-rep="{rep}"/>')

    def z_lit(bf, cote):
        """La cote (m, depuis le bas) du lit retenu d'une bande : celui que le moteur poutre a calculé."""
        pos = bf["retenu"]["lit"]["pos"] / 100.0
        return pos if cote == "inf" else h - pos

    def point(s, z, d, col, rep):
        reps.append(rep)
        return f'<circle cx="{X(s):.1f}" cy="{Z(z):.1f}" r="{max(d / 2000 * ech, t * 0.09):.2f}" fill="{col}" data-rep="{rep}"/>'

    for cote, (rx, ry) in (("inf", ("T1", "T2")), ("sup", ("T3", "T4"))):
        f = fer["faces"][cote]
        d, e = f["d"], f["e"] / 100.0
        col = COULEUR[cote]
        z_x = c + d / 2000.0                                      # les barres en x à l'extérieur
        z_y = c + f["dessous_y"] / 1000.0 + d / 2000.0
        zz = {"x": z_x, "y": z_y}
        if cote == "sup":
            zz = {k: h - v_ for k, v_ in zz.items()}
        rep_l, rep_p = (rx, ry) if sens == "x" else (ry, rx)
        out.append(barre(c, L - c, zz[sens], d, col, rep_l))
        n = int(math.floor((L - 2 * c) / e + 1e-9)) + 1
        for k in range(n):
            out.append(point(c + k * e, zz[autre], d, col, rep_p))
        # les renforts de la bande coupée (dans le sens de la coupe) et ceux des bandes qui la croisent (des points)
        bf = f["bandes"][nom_bande]
        if bf["renfort"]:
            r = bf["renfort"]
            # (au niveau du lit CALCULÉ : dans le plan des barres de sa direction, centré sur la plus grosse barre — pas
            # empilé sur le treillis, qui ferait perdre au chantier le d du calcul)
            out.append(barre(c, L - c, z_lit(bf, cote), r["d"], col, _rep(fer, "R", bande=nom_bande, cote=cote)[0]))
        for nb, bx in f["bandes"].items():
            if (nb.startswith("BX") == (sens == "x")) or not bx["renfort"]:
                continue
            bb = next(x for x in ef["bandes"] if x["nom"] == nb)
            # où la bande croisée coupe l'axe de la coupe : la zone de renfort de la bande croisée, le long de cette coupe
            q0 = np.asarray(bb["p0"], float)
            nrm = np.array([-bb["u"][1], bb["u"][0]])
            s_c = float((q0 - p0) @ u)
            z0_, z1_ = bb["zone"]
            ext = sorted([s_c + z0_ * float(nrm @ u), s_c + z1_ * float(nrm @ u)])
            r = bx["renfort"]
            for k in range(r["n"]):
                s_ = ext[0] + (k + 0.5) * (ext[1] - ext[0]) / r["n"]
                out.append(point(s_, z_lit(bx, cote), r["d"], col, _rep(fer, "R", bande=nb, cote=cote)[0]))
        # les renforts locaux que la coupe traverse
        for ln in fer["bordereau"]["lignes"]:
            if ln["famille"] != "L" or ln["cote"] != cote:
                continue
            x0, y0, x1, y1 = ln["rect"]
            if sens == "x":
                y_c = p0[1]
                if not (y0 - 1e-9 <= y_c <= y1 + 1e-9):
                    continue
                a0, a1 = x0 - p0[0], x1 - p0[0]
            else:
                x_c = p0[0]
                if not (x0 - 1e-9 <= x_c <= x1 + 1e-9):
                    continue
                a0, a1 = y0 - p0[1], y1 - p0[1]
            if ln["sens"] == sens:
                out.append(barre(max(a0, c), min(a1, L - c), zz[sens], ln["d"], col, ln["rep"]))
            else:
                nb_ = max(1, round((a1 - a0) / (ln["s"] / 100.0)))
                for k in range(nb_):
                    out.append(point(a0 + (k + 0.5) * (a1 - a0) / nb_, zz[autre], ln["d"], col, ln["rep"]))
    # ---- les U aux deux bouts
    u_ = fer["U"]
    ue = c + u_["d"] / 2000.0
    for s_, sg in ((ue, 1), (L - ue, -1)):
        br = min(u_["branche"], L / 2)
        d_ = (f"M{X(s_ + sg * br):.1f} {Z(h - ue):.1f}L{X(s_):.1f} {Z(h - ue):.1f}L{X(s_):.1f} {Z(ue):.1f}L{X(s_ + sg * br):.1f} {Z(ue):.1f}")
        out.append(f'<path d="{d_}" fill="none" stroke="{DS.ENCRE}" stroke-width="{max(u_["d"] / 1000 * ech, t * 0.07):.2f}" data-rep="U1"/>')
    reps.append("U1")
    # ---- les C des plots coulés
    if fer["cages"]:
        cg = fer["cages"][0]
        rc = _rep(fer, "C")
        for s1, s2 in plots_s:
            for s_, sg in ((s1 + c, 1), (s2 - c, -1)):
                d_ = (f"M{X(s_ + sg * min(cg['C'][0]['basse'], (s2 - s1) / 2)):.1f} {Z(c):.1f}L{X(s_):.1f} {Z(c):.1f}L{X(s_):.1f} {Z(h + hp - c):.1f}"
                      f"L{X(s_ + sg * (s2 - s1 - 2 * c) / 2):.1f} {Z(h + hp - c):.1f}")
                out.append(f'<path d="{d_}" fill="none" stroke="{DS.PLOT}" stroke-width="{max(cg["d"] / 1000 * ech, t * 0.07):.2f}" data-rep="{rc[0]}"/>')
            out.append(f'<line x1="{X(s1 + c):.1f}" y1="{Z(h + hp - c - 0.03):.1f}" x2="{X(s2 - c):.1f}" y2="{Z(h + hp - c - 0.03):.1f}" stroke="{DS.PLOT}" '
                       f'stroke-width="{max(cg["d"] / 1000 * ech, t * 0.07):.2f}" data-rep="{rc[-1]}"/>')
        reps += rc
    # ---- les C d'effort tranchant
    tr = fer["tranchant"].get(nom_bande)
    if tr and tr["C"]:
        rv = _rep(fer, "V", bande=nom_bande)
        for z0, z1 in tr["C"]["zones"]:
            n = int(math.floor((z1 - z0) * 100.0 / tr["C"]["s"] + 1e-9)) + 1
            for k in range(n):
                s_ = z0 + k * tr["C"]["s"] / 100.0
                out.append(f'<line x1="{X(s_):.1f}" y1="{Z(c):.1f}" x2="{X(s_):.1f}" y2="{Z(h - c):.1f}" stroke="{DS.ENCRE}" '
                           f'stroke-width="{max(tr["C"]["d"] / 1000 * ech, t * 0.06):.2f}" data-rep="{rv[0]}"/>')
        reps += rv
    # ---- les repères écrits, sous la coupe ; les cotes
    vus = []
    for r_ in reps:
        if r_ not in vus:
            vus.append(r_)
    out.append(DS._texte(X(0), Z(0) + t * 3.4, "Repères : " + ", ".join(vus), t * 0.72, DS.ENCRE, "start", attrs=f' data-reperes="{",".join(vus)}"'))
    yc = Z(0) + t * 1.6
    out.append(f'<line x1="{X(0):.1f}" y1="{yc:.1f}" x2="{X(L):.1f}" y2="{yc:.1f}" stroke="{DS.COTE}" stroke-width="{t * 0.06:.2f}"/>'
               + DS._texte(X(L / 2), yc - t * 0.25, f"{C.fr(L, 2)}{N}m", t * 0.72, DS.COTE, "middle"))
    xc = X(L) + t * 1.2
    out.append(f'<line x1="{xc:.1f}" y1="{Z(0):.1f}" x2="{xc:.1f}" y2="{Z(h):.1f}" stroke="{DS.COTE}" stroke-width="{t * 0.06:.2f}"/>'
               + DS._texte(xc + t * 0.3, Z(h / 2) + t * 0.25, f"h{N}={N}{C.fr(h, 2)}{N}m · c{N}={N}{C.fr(c * 100, 1)}{N}cm", t * 0.72, DS.COTE, "start"))
    if hp > 0:
        out.append(DS._texte(xc + t * 0.3, Z(h + hp / 2) + t * 0.25, f"plot {C.fr(hp, 2)}{N}m", t * 0.72, DS.COTE, "start"))
    return "".join(out), (h + hp + 0.4) * ech + t * 4.6


def coupes(el: Dict, m: Dict, fer: Dict, ef: Dict, largeur: float = 1000.0) -> str:
    """Les deux coupes, A-A le long de BX1 et B-B le long de BY1, l'une sous l'autre, à la même échelle."""
    noms = [b["nom"] for b in ef["bandes"]]
    out, y = [], 0.0
    t = 1000.0 * 0.024
    for nom, lettre in (("BX1", "A"), ("BY1", "B")):
        if nom not in noms:
            continue
        svg, h_ = _coupe(el, m, fer, ef, nom, lettre, y + t * 1.6, largeur)
        out.append(f'<g class="df-coupe-{lettre}">{svg}</g>')
        y += h_ + t * 3.0
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-t * 1.5:.1f} {-t * 0.5:.1f} {largeur + t * 18:.1f} {y + t:.1f}" role="img" '
            f'aria-label="Coupes A-A et B-B" class="sf-plan df-coupes">{"".join(out)}</svg>')


def reperes(svg: str) -> List[str]:
    """Les repères écrits dans un dessin (data-rep), sans doublon, dans l'ordre."""
    out = []
    for r in re.findall(r'data-rep="([^"]+)"', svg):
        for x in r.split(","):
            x = x.strip()
            if x and x not in out:
                out.append(x)
    return out

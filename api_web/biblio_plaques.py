# -*- coding: utf-8 -*-
"""La rubrique « PLAQUES » de la bibliothèque « Autres » : plaques rectangulaires minces sous charge uniforme
(docs/site/BIBLIOTHEQUE_AUTRES.md § 4). Le moteur est rdm/plaques.py ; ce fichier LIT les paramètres, COMPOSE l'écran
(le plan de la plaque — un clic sur un bord change sa condition —, la carte des flèches, les résultats, la feuille de
calcul) et FORMATE.

Paramètres : a, b (m), t (mm), q (kN/m²), les quatre bords bx0, bxa, by0, byb (A appui simple, E encastrement, L
libre), mat (acier, beton, alu, autre) et ses données, g (la grandeur de la bande).

Le COEFFICIENT DE POISSON n'est pas demandé : il vient du matériau choisi (acier 0,3 ; béton non fissuré 0,2 ; aluminium
0,3) et s'écrit comme hypothèse ; seul « autre matériau » le fait saisir, avec E. Un cas hors du périmètre est REFUSÉ avec
sa raison (``NonPrisEnCharge`` du moteur) : aucune valeur approchée n'est affichée à sa place.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

from api_web import biblio_autres as A
from api_web import biblio_beton as B
from api_web import commun as C
from beton.donnees import ec2 as EC
from rdm import plaques as PL

n, r, m = A.n, A.r, A.m
BORDS = (("A", "Appui simple"), ("E", "Encastré"), ("L", "Libre"))
NOM_BORD = {"A": "appui simple", "E": "encastré", "L": "libre"}
SUIVANT = {"A": "E", "E": "L", "L": "A"}
COTES = (("x0", "bx0", "Gauche", "x = 0"), ("xa", "bxa", "Droit", "x = a"), ("y0", "by0", "Bas", "y = 0"), ("yb", "byb", "Haut", "y = b"))
NOM_COTE = {"x0": "gauche", "xa": "droit", "y0": "bas", "yb": "haut"}
MATERIAUX = (("acier", "Acier"), ("beton", "Béton"), ("alu", "Aluminium"), ("autre", "Autre"))
# (E en MPa, ν, référence) — le béton lit E_cm de sa classe
MAT = {"acier": (210000.0, 0.3, "EN 1993-1-1 §3.2.6"), "alu": (70000.0, 0.3, "EN 1999-1-1 §3.2.5")}
NU_BETON, REF_BETON = 0.2, "EN 1992-1-1 §3.1.3 (E_cm, tableau 3.1 ; ν = 0,2 béton non fissuré)"
CLASSES_BETON = [f"C{a_}/{b_}" for a_, b_ in EC.CLASSES_3_1]
GRANDEURS = ("w", "M", "sigma")
PREREGLAGES = (("AAAA", "4 appuis"), ("EEEE", "4 encastrés"), ("AAEE", "2 appuis + 2 encastrés"), ("AAAL", "3 appuis + 1 libre"), ("EAEA", "2 encastrés adjacents"))
METHODES = {"levy": ("série de Lévy", "Timoshenko & Woinowsky-Krieger §§ 30, 41-48"),
            "superposition": ("superposition de Timoshenko", "Timoshenko & Woinowsky-Krieger §§ 42-45")}
SOURCES = ("S. Timoshenko, S. Woinowsky-Krieger, <i>Theory of Plates and Shells</i>, 2ᵉ éd., McGraw-Hill, 1959 : § 28 (Navier), § 30 (Lévy), "
           "§§ 41-48 (bords encastrés et libres), tableaux 8, 29, 30 et 35.",
           "Valeurs de contrôle récentes de la plaque encastrée : R. L. Taylor, S. Govindjee, « Solution of clamped rectangular plate problems », "
           "<i>Commun. Numer. Meth. Engng</i>, 2004.",
           "Matériaux : EN 1993-1-1 §3.2.6 (acier), EN 1992-1-1 §3.1.3 (béton), EN 1999-1-1 §3.2.5 (aluminium).")
VOLET = [
    ("Hypothèses", ["Plaque mince d'épaisseur constante, matériau élastique linéaire isotrope (théorie de Kirchhoff-Love).",
                    "Petites flèches : la flèche reste faible devant l'épaisseur (au-delà de t/2 environ, l'effet de membrane raidit la plaque et "
                    "la flèche affichée est majorée).",
                    "Charge uniforme sur toute la plaque, perpendiculaire à son plan.",
                    "Appui simple : flèche nulle, rotation libre, coins MAINTENUS (une plaque appuyée tend à se soulever aux coins). "
                    "Encastrement : flèche et rotation nulles. Bord libre : ni moment ni réaction."]),
    ("Conventions", ["x le long du côté a, y le long du côté b ; l'origine est le coin bas gauche.",
                     "M<sub>x</sub> est le moment qui fléchit les fibres parallèles à x (contraintes σ<sub>x</sub>) ; M<sub>y</sub> de même pour y. "
                     "Positif quand la face opposée à la charge est tendue.",
                     "w = α·q·L⁴/D et M = β·q·L², avec L le PETIT côté : α et β se comparent aux tableaux publiés (qui valent pour ν = 0,3)."]),
    ("Méthode", ["Deux bords opposés appuyés : série de Lévy — exacte pour chaque harmonique, les deux autres bords appuyés, encastrés ou libres.",
                 "Bords appuyés et encastrés sans paire opposée appuyée (dont la plaque encastrée sur quatre bords) : superposition de Timoshenko — "
                 "plaque appuyée + moments de bord en séries, rotations annulées.",
                 "Quatre bords appuyés : la double série de Navier est calculée en plus et comparée.",
                 "Le nombre de termes est fixé ; le moteur recalcule avec la moitié des termes et affiche l'écart."]),
    ("Non pris en charge", ["Un bord libre dont les deux bords voisins ne sont pas tous deux simplement appuyés (console, coin libre, trois bords "
                            "encastrés et un libre…) : pas de solution classique en série simple.",
                            "Charges partielles, ponctuelles ou hydrostatiques ; appuis ponctuels ; plaques épaisses ; grandes flèches.",
                            "Aucune vérification de résistance au sens d'un Eurocode : σ = 6·M/t² est la contrainte élastique de flexion."]),
]


# ================================================================================================ lecture
def lire(params: Optional[Dict]) -> Tuple[B.Lecture, Dict]:
    L = B.Lecture(params)
    o: Dict = {"a": L.nombre("a", 2.0, 0.05, 200.0), "b": L.nombre("b", 1.5, 0.05, 200.0), "t": L.nombre("t", 10.0, 0.1, 2000.0),
               "q": L.nombre("q", 5.0, -1e5, 1e5)}
    o["bords"] = {cote: L.choix(cle, [k for k, _ in BORDS], "A") for cote, cle, _, _ in COTES}
    o["mat"] = L.choix("mat", [k for k, _ in MATERIAUX], "acier")
    if o["mat"] == "beton":
        classe = L.choix("classe", CLASSES_BETON, "C30/37")
        fck = EC.CLASSES_3_1[CLASSES_BETON.index(classe)][0]
        o.update(classe=classe, E=EC.Ecm(fck) * 1000.0, nu=NU_BETON, ref=REF_BETON)        # E_cm est en GPa au tableau 3.1
    elif o["mat"] == "autre":
        o.update(E=L.nombre("E", 210000.0, 1.0, 1e7), nu=L.nombre("nu", 0.3, 0.0, 0.49), ref="valeurs saisies")
    else:
        E, nu, ref = MAT[o["mat"]]
        o.update(E=E, nu=nu, ref=ref)
    o["g"] = L.choix("g", GRANDEURS, "w")
    return L, o


def _barre(o: Dict) -> str:
    plaque = (B._champ("a", "a (selon x)", B._num("a", o["a"], "m")) + B._champ("b", "b (selon y)", B._num("b", o["b"], "m"))
              + B._champ("t", "Épaisseur t", B._num("t", o["t"], "mm")))
    charge = B._champ("q", "Charge q", B._num("q", o["q"], "kN/m²"), "Charge uniforme sur toute la plaque, positive vers le bas.")
    mat = B._champ("mat", "Matériau", B._segments("mat", o["mat"], MATERIAUX),
                   "Le matériau donne E et le coefficient de Poisson ν : D = E·t³/(12·(1 − ν²)).")
    if o["mat"] == "beton":
        mat += B._champ("classe", "Classe", B._select("classe", o["classe"], [(k, k) for k in CLASSES_BETON]))
    elif o["mat"] == "autre":
        mat += B._champ("E", "E", B._num("E", o["E"], "MPa")) + B._champ("nu", "ν", B._num("nu", o["nu"]))
    return B._barre(B._groupe("Plaque", plaque) + B._groupe("Charge", charge), B._groupe("Matériau", mat))


# ================================================================================================ le dessin
W = 640.0
MARGE = 78.0
H_MAX = 400.0


def _niveaux(grille: Sequence[Sequence[float]], X, Y, a: float, b: float, nb: int = 7) -> str:
    """Les lignes de niveau de la flèche (carrés marchants sur la grille du moteur) : un chemin par niveau."""
    ny, nx = len(grille) - 1, len(grille[0]) - 1
    vals = [v for li in grille for v in li]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-300:
        return ""
    ext = hi if abs(hi) >= abs(lo) else lo
    out = ""
    for k in range(1, nb + 1):
        niv = ext * k / (nb + 0.6)
        d = []
        for j in range(ny):
            for i in range(nx):
                c = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                v = [grille[q_][p_] - niv for p_, q_ in c]
                pts = []
                for e in range(4):
                    v0, v1 = v[e], v[(e + 1) % 4]
                    if (v0 > 0) != (v1 > 0):
                        s = v0 / (v0 - v1)
                        (p0, q0), (p1, q1) = c[e], c[(e + 1) % 4]
                        pts.append((a * (p0 + s * (p1 - p0)) / nx, b * (q0 + s * (q1 - q0)) / ny))
                for u in range(0, len(pts) - 1, 2):
                    d.append(f"M{X(pts[u][0]):.1f} {Y(pts[u][1]):.1f}L{X(pts[u + 1][0]):.1f} {Y(pts[u + 1][1]):.1f}")
        if d:
            out += f'<path class="niv" style="opacity:{0.35 + 0.6 * k / nb:.2f}" d="{"".join(d)}"/>'
    return out


def _carte(grille: Sequence[Sequence[float]], X, Y, a: float, b: float) -> str:
    """La carte des flèches : une case par maille, d'autant plus teintée que la flèche est grande."""
    ny, nx = len(grille) - 1, len(grille[0]) - 1
    ext = max(abs(v) for li in grille for v in li) or 1.0
    out = ""
    for j in range(ny):
        for i in range(nx):
            v = (grille[j][i] + grille[j][i + 1] + grille[j + 1][i] + grille[j + 1][i + 1]) / 4.0
            op = 0.62 * (abs(v) / ext) ** 0.85
            if op > 0.02:
                out += (f'<rect x="{X(a * i / nx):.1f}" y="{Y(b * (j + 1) / ny):.1f}" width="{X(a * (i + 1) / nx) - X(a * i / nx) + 0.4:.1f}" '
                        f'height="{Y(b * j / ny) - Y(b * (j + 1) / ny) + 0.4:.1f}" style="opacity:{op:.3f}"/>')
    return f'<g class="carte">{out}</g>'


def _bord_svg(cote: str, typ: str, x0: float, y0: float, x1: float, y1: float, cle: str) -> str:
    """Un bord de la plaque : son symbole (appui simple : trait + tirets ; encastré : trait fort + hachures ; libre : trait
    fin), son nom, et la zone qui se clique (le bord passe à la condition suivante)."""
    horiz = cote in ("y0", "yb")
    ext = {"x0": (-1, 0), "xa": (1, 0), "y0": (0, 1), "yb": (0, -1)}[cote]              # vers l'extérieur (repère écran)
    s = ""
    if typ == "E":
        s += f'<path class="b-enc" d="M{x0:.1f} {y0:.1f}L{x1:.1f} {y1:.1f}"/>'
        nb_ = int((abs(x1 - x0) + abs(y1 - y0)) / 9)
        for k in range(nb_ + 1):
            px = x0 + (x1 - x0) * k / max(1, nb_)
            py = y0 + (y1 - y0) * k / max(1, nb_)
            s += f'<path class="b-hach" d="M{px:.1f} {py:.1f}l{8 * ext[0] + (5 if horiz else 0):.1f} {8 * ext[1] + (0 if horiz else 5):.1f}"/>'
    elif typ == "A":
        s += (f'<path class="b-app" d="M{x0:.1f} {y0:.1f}L{x1:.1f} {y1:.1f}"/>'
              f'<path class="b-app2" d="M{x0 + 5 * ext[0]:.1f} {y0 + 5 * ext[1]:.1f}L{x1 + 5 * ext[0]:.1f} {y1 + 5 * ext[1]:.1f}"/>')
    else:
        s += f'<path class="b-lib" d="M{x0:.1f} {y0:.1f}L{x1:.1f} {y1:.1f}"/>'
    xm, ym = (x0 + x1) / 2 + 24 * ext[0], (y0 + y1) / 2 + 24 * ext[1]
    if horiz:
        etiq = f'<text class="b-t" x="{xm:.1f}" y="{ym + 4:.1f}" text-anchor="middle">{NOM_BORD[typ]}</text>'
    else:
        etiq = f'<text class="b-t" x="{xm:.1f}" y="{ym:.1f}" text-anchor="middle" transform="rotate({-90 if cote == "x0" else 90} {xm:.1f} {ym:.1f})">{NOM_BORD[typ]}</text>'
    zx, zy = min(x0, x1) - (16 if not horiz else 0), min(y0, y1) - (16 if horiz else 0)
    zw, zh = abs(x1 - x0) + (32 if not horiz else 0), abs(y1 - y0) + (32 if horiz else 0)
    return (f'<g class="bord" data-bord="{cote}" data-type="{typ}" role="button" tabindex="0" aria-label="Bord {NOM_COTE[cote]} : {NOM_BORD[typ]} — changer"'
            f'{A.sel({cle: SUIVANT[typ]})}><title>Bord {NOM_COTE[cote]} : {NOM_BORD[typ]} — cliquer pour changer</title>'
            f'<rect class="b-zone" x="{zx:.1f}" y="{zy:.1f}" width="{zw:.1f}" height="{zh:.1f}"/>{s}{etiq}</g>')


def dessin(o: Dict, res: Optional[Dict]) -> str:
    a, b = o["a"], o["b"]
    ech = min((W - 2 * MARGE) / a, H_MAX / b)
    lw, lh = a * ech, b * ech
    ox, oy = (W - lw) / 2, MARGE - 26
    H = oy + lh + MARGE - 14
    X = lambda x: ox + x * ech                                        # noqa: E731
    Y = lambda y: oy + lh - y * ech                                   # noqa: E731
    s = f'<rect class="plaque" x="{ox:.1f}" y="{oy:.1f}" width="{lw:.1f}" height="{lh:.1f}"/>'
    if res:
        s += _carte(res["grille"], X, Y, a, b) + _niveaux(res["grille"], X, Y, a, b)
    cles = {c: k for c, k, _, _ in COTES}
    bd = o["bords"]
    s += _bord_svg("y0", bd["y0"], X(0), Y(0), X(a), Y(0), cles["y0"]) + _bord_svg("yb", bd["yb"], X(0), Y(b), X(a), Y(b), cles["yb"])
    s += _bord_svg("x0", bd["x0"], X(0), Y(0), X(0), Y(b), cles["x0"]) + _bord_svg("xa", bd["xa"], X(a), Y(0), X(a), Y(b), cles["xa"])
    # les cotes et les axes
    s += (f'<text class="cote-t" x="{X(a / 2):.1f}" y="{Y(0) + 54:.1f}" text-anchor="middle">a = {r(a)} m</text>'
          f'<text class="cote-t" x="{X(0) - 52:.1f}" y="{Y(b / 2):.1f}" text-anchor="middle" transform="rotate(-90 {X(0) - 52:.1f} {Y(b / 2):.1f})">b = {r(b)} m</text>'
          f'<g class="axes"><path d="M{X(0):.1f} {Y(0):.1f}h26M{X(0):.1f} {Y(0):.1f}v-26"/><text x="{X(0) + 30:.1f}" y="{Y(0) - 4:.1f}">x</text>'
          f'<text x="{X(0) + 5:.1f}" y="{Y(0) - 30:.1f}">y</text></g>')
    if res:
        def marque(x, y, txt, cls, dx=0, dy=-9, ancre="middle"):
            return (f'<g class="mq {cls}"><circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.4"/>'
                    f'<text x="{X(x) + dx:.1f}" y="{Y(y) + dy:.1f}" text-anchor="{ancre}">{txt}</text></g>')
        wm = res["w_max"]
        sur_bord_haut = wm["y"] >= b * 0.97
        s += marque(wm["x"], wm["y"], f"w = {r(wm['valeur'] * 1000)} mm", "mq-w", 0, 18 if sur_bord_haut else -9)
        if abs(wm["x"] - a / 2) > 0.03 * a or abs(wm["y"] - b / 2) > 0.03 * b:
            s += f'<path class="centre" d="M{X(a / 2) - 6:.1f} {Y(b / 2):.1f}h12M{X(a / 2):.1f} {Y(b / 2) - 6:.1f}v12"/>'
        for e in res["encastrements"]:
            dedans = {"x0": (8, 4, "start"), "xa": (-8, 4, "end"), "y0": (0, -9, "middle"), "yb": (0, 18, "middle")}[e["cote"]]
            s += marque(e["x"], e["y"], f"{r(e['valeur'])}", "mq-m", *dedans)
    return (f'<svg class="rdm-svg rdm-svg-plaque" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="Plan de la plaque : bords, flèches et moments">'
            f'{s}</svg>')


# ================================================================================================ résultats, feuille
def _bords_bloc(o: Dict) -> str:
    cases = "".join(A.menu_champ(cle, o["bords"][cote], BORDS, f"{nom} <small>({ou})</small>") for cote, cle, nom, ou in COTES)
    pre = "".join(A.bouton(nom, dict(zip([c[1] for c in COTES], code)), "", "btn sm") for code, nom in PREREGLAGES)
    return A.bloc("Bords", f'<div class="rdm-bords">{cases}</div><div class="rdm-plus">{pre}</div>'
                  '<p class="rdm-aide">Un clic sur un bord du dessin change aussi sa condition : appui simple → encastré → libre.</p>', "rdm-bords-b")


def _resultats(res: Dict, o: Dict) -> str:
    def lig(g, sym, nom, val, unite, pos="", coef=""):
        on = o["g"] == g
        sel = A.sel({"g": g}) if g else ""
        return (f'<tr{sel} class="rdm-r{" on" if on else ""}{"" if g else " fixe"}"{" tabindex=" + chr(34) + "0" + chr(34) if g else ""}><th>{m(sym)}</th>'
                f'<td class="nm">{nom}{("<br>" + pos) if pos else ""}</td><td class="v"><b>{val}</b> {unite}</td><td class="p">{coef}</td></tr>')
    wm, ce = res["w_max"], res["centre"]
    corps = lig("w", "w_max", "flèche maximale", r(wm["valeur"] * 1000), "mm", f"x = {r(wm['x'])} m, y = {r(wm['y'])} m", f"α = {r(wm['alpha'], 5)}")
    corps += lig("M", "M_x", "moment au centre, sens x", r(ce["Mx"]), "kN·m/m", "", f"β = {r(ce['beta_x'], 4)}")
    corps += lig("M", "M_y", "moment au centre, sens y", r(ce["My"]), "kN·m/m", "", f"β = {r(ce['beta_y'], 4)}")
    for cle, sym, sens in (("Mx_max", "M_x,max", "x"), ("My_max", "M_y,max", "y")):
        e = res[cle]
        if abs(e["valeur"] - ce["M" + sens]) > 1e-3 * (res["M_abs_max"] or 1.0):
            corps += lig("M", sym, f"moment maximal en travée, sens {sens}", r(e["valeur"]), "kN·m/m", f"x = {r(e['x'])} m, y = {r(e['y'])} m", f"β = {r(e['beta'], 4)}")
    for e in res["encastrements"]:
        corps += lig("M", f"M_{e['composante'][1]}", f"encastrement, bord {NOM_COTE[e['cote']]}", r(e["valeur"]), "kN·m/m",
                     f"x = {r(e['x'])} m, y = {r(e['y'])} m", f"β = {r(e['beta'], 4)}")
    for e in res["libres"]:
        corps += lig("w", "w", f"bord libre {NOM_COTE[e['cote']]}", r(e["w"] * 1000), "mm", f"x = {r(e['x'])} m, y = {r(e['y'])} m", f"α = {r(e['alpha'], 5)}")
        corps += lig("M", f"M_{e['composante'][1]}", f"le long du bord libre {NOM_COTE[e['cote']]}", r(e["M"]), "kN·m/m", "", f"β = {r(e['beta'], 4)}")
    corps += lig("sigma", "σ_max", "contrainte de flexion", r(res["sigma_max"], 1), "MPa", "", "6·M/t²")
    return A.bloc("Résultats", f'<table class="rdm-res"><tbody>{corps}</tbody></table>'
                  f'<p class="rdm-aide">w = α·q·L⁴/D et M = β·q·L², L = {r(res["L"])} m (le petit côté) — α et β se comparent aux tableaux publiés pour ν = 0,3.</p>',
                  "rdm-resultats")


def _feuille(res: Dict, o: Dict) -> List:
    q, L, D, t = res["q"], res["L"], res["D"], res["t"]
    nom_m, ref_m = METHODES[res["methode"]]
    g = o["g"]
    wm, ce = res["w_max"], res["centre"]
    Mg = max([("M_x,max", res["Mx_max"]["valeur"], res["Mx_max"]["beta"]), ("M_y,max", res["My_max"]["valeur"], res["My_max"]["beta"])]
             + [(f"M_{e['composante'][1]} (bord {NOM_COTE[e['cote']]})", e["valeur"], e["beta"]) for e in res["encastrements"]]
             + [(f"M_{e['composante'][1]} (bord libre {NOM_COTE[e['cote']]})", e["M"], e["beta"]) for e in res["libres"]], key=lambda x: abs(x[1]))
    donnees = [A.ligne("D", "rigidité de flexion de la plaque", "E·t³/(12·(1 − ν²))", f"{n(res['E'], 0)} × {n(t)}³ / (12 × (1 − {n(res['nu'])}²)) / 1000000",
                       r(D, 3), "kN·m", "Kirchhoff", False, "D"),
               A.ligne("L", "petit côté", "min(a ; b)", f"min({n(res['a'])} ; {n(res['b'])})", r(L), "m", "", False, "L")]
    fle = [A.ligne("α", f"coefficient de flèche ({nom_m})", "", "", r(wm["alpha"], 5), "", ref_m, False, "alpha"),
           A.ligne("w_max", f"flèche maximale · x = {r(wm['x'])} m, y = {r(wm['y'])} m", "α·q·L⁴/D", f"{n(wm['alpha'], 6)} × {n(q)} × {n(L)}⁴ / {n(D, 4)} × 1000",
                   r(wm["valeur"] * 1000), "mm", "", g == "w", "w_max")]
    for e in res["libres"]:
        fle.append(A.ligne("w", f"flèche du bord libre {NOM_COTE[e['cote']]}", "α·q·L⁴/D", f"{n(e['alpha'], 6)} × {n(q)} × {n(L)}⁴ / {n(D, 4)} × 1000",
                           r(e["w"] * 1000), "mm", "", False, f"w_{e['cote']}"))
    mom = [A.ligne("M_x", "moment au centre, sens x", "β_x·q·L²", f"{n(ce['beta_x'], 5)} × {n(q)} × {n(L)}²", r(ce["Mx"]), "kN·m/m", ref_m, False, "Mx"),
           A.ligne("M_y", "moment au centre, sens y", "β_y·q·L²", f"{n(ce['beta_y'], 5)} × {n(q)} × {n(L)}²", r(ce["My"]), "kN·m/m", ref_m, False, "My")]
    for e in res["encastrements"]:
        mom.append(A.ligne(f"M_{e['composante'][1]}", f"encastrement du bord {NOM_COTE[e['cote']]} · x = {r(e['x'])} m, y = {r(e['y'])} m", "β·q·L²",
                           f"{n(e['beta'], 5)} × {n(q)} × {n(L)}²", r(e["valeur"]), "kN·m/m", ref_m, False, f"Menc_{e['cote']}"))
    for e in res["libres"]:
        mom.append(A.ligne(f"M_{e['composante'][1]}", f"le long du bord libre {NOM_COTE[e['cote']]}", "β·q·L²", f"{n(e['beta'], 5)} × {n(q)} × {n(L)}²",
                           r(e["M"]), "kN·m/m", ref_m, False, f"Mlib_{e['cote']}"))
    mom.append(A.ligne("M_max", f"le plus grand moment en valeur absolue : {Mg[0]}", "", "", r(Mg[1]), "kN·m/m", "", g == "M", "M_max"))
    con = [A.ligne("σ_max", "contrainte de flexion (fibres extrêmes)", "6·M_max/t²", f"6 × {n(res['M_abs_max'], 4)} × 1000 / {n(t)}²", r(res["sigma_max"], 1), "MPa",
                   "Navier", g == "sigma", "sigma")]
    k = res["controles"]
    ctl = [A.ligne("convergence", "écart en recalculant avec la moitié des termes", "", "", f"{max(k['convergence_w'], k['convergence_M']):.1e}".replace(".", ",").replace("e", "·10^"), "",
                   "flèche et moments", False, "conv")]
    if "navier" in k:
        ctl.append(A.ligne("Navier", "double série de sinus, au centre : écart avec Lévy", "", "", f"{max(k['navier']['ecart_w'], k['navier']['ecart_M']):.1e}".replace(".", ",").replace("e", "·10^"),
                           "", "T. & W.-K. § 28", False, "navier"))
    return [("Données", "", donnees), ("Flèche", "w = α·q·L⁴/D", fle), ("Moments de flexion", "M = β·q·L² · par mètre de largeur", mom), ("Contrainte", "", con),
            ("Contrôles du calcul", "", ctl)]


def _hypotheses(o: Dict, res: Optional[Dict]) -> List[str]:
    bd = o["bords"]
    h = [f"Plaque {r(o['a'])} × {r(o['b'])} m, épaisseur {r(o['t'], 1)} mm, charge uniforme q = {r(o['q'])} kN/m².",
         "Bords : " + ", ".join(f"{NOM_COTE[c]} {NOM_BORD[bd[c]]}" for c in ("x0", "xa", "y0", "yb")) + "."]
    nom = {"acier": "Acier", "beton": f"Béton {o.get('classe', '')}", "alu": "Aluminium", "autre": "Matériau saisi"}[o["mat"]]
    h.append(f"{nom} : E = {r(o['E'], 0)} MPa, ν = {r(o['nu'], 2)} ({m(o['ref'])}).")
    h.append("Plaque mince élastique isotrope (Kirchhoff-Love), petites flèches, coins maintenus.")
    if res:
        h.append(f"Méthode : {METHODES[res['methode']][0]} ({METHODES[res['methode']][1]}).")
    return h


def rendre(params: Optional[Dict], hote: str = "page") -> Dict:
    L, o = lire(params)
    titre = "Plaques rectangulaires"
    barre = _barre(o)
    base = {"titre": titre, "barre": barre, "pourquoi": A.volet(titre, "charge uniforme, théorie de Kirchhoff", VOLET, SOURCES),
            "sources": A.sources(*SOURCES), "params": L.out, "message": L.message()}
    try:
        res = PL.calculer(o["a"], o["b"], o["bords"], o["q"], o["E"], o["nu"], o["t"])
    except PL.NonPrisEnCharge as ex:
        refus = A.refus("Cette combinaison de bords n'est pas prise en charge", C.esc(str(ex)[0].upper() + str(ex)[1:]) + ".",
                        ["Sont calculées : les plaques dont deux bords OPPOSÉS sont simplement appuyés (les deux autres appuyés, encastrés ou libres), "
                         "et toutes les plaques à bords appuyés ou encastrés.",
                         "Aucune valeur approchée n'est affichée à la place : pour ce cas, un calcul aux éléments finis ou un tableau spécialisé est nécessaire."])
        hyp = _hypotheses(o, None)
        tableau = (f'<div class="rdm rdm-plaques">{A.entete_impression(titre, "cas non pris en charge", hyp)}<div class="rdm-g"><figure class="rdm-dessin">'
                   f'{dessin(o, None)}</figure></div><div class="rdm-d">{_bords_bloc(o)}{refus}</div></div>')
        return dict(base, tableau=tableau, bande=A.bande("—", "", "w_max", "cas non pris en charge", "changer la condition d'un bord"))
    except PL.ErreurDonnees as ex:
        tableau = f'<div class="rdm rdm-plaques">{A.refus("Données hors du domaine", C.esc(str(ex)) + ".")}<div class="rdm-d">{_bords_bloc(o)}</div></div>'
        return dict(base, tableau=tableau, bande="")
    hyp = _hypotheses(o, res)
    entete = A.entete_impression(titre, f"{r(o['a'])} × {r(o['b'])} m", hyp)
    feuille = (A.bloc("Feuille de calcul", f'<ul class="rdm-hyp">{"".join(f"<li>{h}</li>" for h in hyp)}</ul>{A.feuille(_feuille(res, o))}', "rdm-feuille"))
    tableau = (f'<div class="rdm rdm-plaques">{entete}<div class="rdm-g"><figure class="rdm-dessin">{dessin(o, res)}</figure>'
               '<p class="rdm-aide">Carte et lignes de niveau de la flèche. Points : flèche maximale ; moment le plus fort de chaque bord encastré (kN·m/m).</p></div>'
               f'<div class="rdm-d">{_bords_bloc(o)}{_resultats(res, o)}</div><div class="rdm-f">{feuille}</div></div>')
    av = []
    if "grande_fleche" in res["avertissements"]:
        av.append(f"La flèche dépasse t/2 = {r(o['t'] / 2, 1)} mm : hors du domaine des petites flèches, l'effet de membrane raidit la plaque — la flèche affichée est majorée.")
    if "epaisse" in res["avertissements"]:
        av.append(f"Plaque épaisse (t > L/10 = {r(res['L'] * 100, 0)} mm) : la théorie des plaques minces néglige la déformation d'effort tranchant.")
    wm = res["w_max"]
    if o["g"] == "M":
        Mg = max([res["Mx_max"]["valeur"], res["My_max"]["valeur"]] + [e["valeur"] for e in res["encastrements"]] + [e["M"] for e in res["libres"]], key=abs)
        bande = A.bande(r(Mg), "kN·m/m", "M_max", "plus grand moment de flexion, en valeur absolue", f"σ = 6·M/t² = {r(res['sigma_max'], 1)} MPa", av)
    elif o["g"] == "sigma":
        bande = A.bande(r(res["sigma_max"], 1), "MPa", "σ_max", "contrainte de flexion maximale", f"6·M/t² · M = {r(res['M_abs_max'])} kN·m/m, t = {r(o['t'], 1)} mm", av)
    else:
        rapport = f" · L/{C.fr(res['L'] * 1000 / abs(wm['valeur'] * 1000), 0)}" if abs(wm["valeur"]) > 1e-12 else ""
        bande = A.bande(r(wm["valeur"] * 1000), "mm", "w_max", "flèche maximale", f"en x = {r(wm['x'])} m, y = {r(wm['y'])} m{rapport} · α = {r(wm['alpha'], 5)}", av)
    return dict(base, tableau=tableau, bande=bande)

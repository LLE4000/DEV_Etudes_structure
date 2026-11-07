# -*- coding: utf-8 -*-
"""Les DESSINS du module « Maturité du béton » : la courbe f_ck(t) et le relevé des températures, en SVG.

La courbe est celle que l'écran Streamlit trace avec matplotlib (``modules/age_beton.py``, ``build_figure``) : les MÊMES
données — les 500 abscisses de 1 à 40 j, la courbe de référence, la variante défavorable, les classes du comparateur,
la classe de l'optimisation et son repère, le point de l'âge choisi, le seuil (le décoffrage), l'essai, le repère
t_e = 28 j — que le moteur ``beton/maturite`` rend dans ``courbes`` ; ce fichier ne calcule rien, il place. Il ajoute
ce que la figure de Streamlit laissait deviner : la bande grise du domaine t_e ≤ 3 j (hors du domaine de l'EC2), le nom
de chaque classe au bout de sa courbe, des graduations lisibles en français. La légende est posée AU-DESSUS du tracé
(celle de matplotlib, en bas à droite, cache les courbes des petites classes) ; l'étiquette du point ① se pose là où
elle ne couvre ni un autre repère ni une courbe.

Le relevé (vue « Températures ») montre jour par jour la température saisie (mesurée, prévue, prolongée, sans donnée),
la variante défavorable (jours prévus − 3 °C), la moyenne qui sert à la courbe, et l'âge équivalent cumulé t_e.

Les deux graphiques ont 580 unités de large : la feuille les met à l'échelle de l'écran (≈ 1 sur PC, ≈ 0,6 sur
téléphone), et les textes de 15 à 17 unités y restent lisibles. Les couleurs des courbes sont celles du module
(``moteur.COULEURS``) ; le dessin est posé sur le papier blanc de la feuille, en clair comme en sombre. Tout texte venu
de l'état (un lieu, un nom) est échappé ici.
"""
import math
import re
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

ENCRE, GRIS, GRIS_CLAIR, GRILLE, AXE = "#15181F", "#6B7686", "#98A2B3", "#E8EBF0", "#B8C0CC"
FOND_DOMAINE = "#F1F2F5"
POLICE = "system-ui,Arial,sans-serif"
SOURCES = {"mesure": ("#2e6fb0", "mesuré"), "prevision": ("#d97706", "prévision"), "prolonge": ("#9AA3AF", "prolongé")}
COL_TE = "#1f8a70"
COL_DEFAV = "#b91c1c"
# les tailles de texte (unités du dessin) : graduations, noms au bout des courbes, titres des axes, légende, étiquettes
T_GRAD, T_NOM, T_TITRE, T_LEG, T_BOITE = 16.0, 15.0, 17.0, 15.0, 16.0
CAR = 0.56                     # largeur moyenne d'un caractère, en fraction de la taille (system-ui, Arial)


# ------------------------------------------------------------------ petits outils
def _f(x: float) -> str:
    """Une coordonnée SVG (au dixième, sans zéro inutile)."""
    s = f"{x:.1f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fr(x: float, nd: int = 1) -> str:
    """Un nombre à la française (virgule, vrai signe moins) pour les étiquettes du dessin."""
    s = f"{x:.{nd}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−").replace(".", ",")


def _txt(x, y, t, taille=13.0, ancre="middle", couleur=ENCRE, gras=False, extra="") -> str:
    poids = ' font-weight="650"' if gras else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-family="{POLICE}"{poids}{extra}>{t}</text>')


def _sub(base: str, indice: str, suite: str = "") -> str:
    """« f<sub>ck</sub>(t) » en SVG : l'indice descendu, la suite remontée."""
    return f"{base}<tspan dy=\"4\" font-size=\"75%\">{indice}</tspan><tspan dy=\"-4\">{suite}</tspan>"


def _te(suite: str = "") -> str:
    """« t_e » en SVG (t, indice e), suivi d'un texte déjà échappé."""
    return _sub("t", "e", suite)


def _largeur(plain: str, taille: float, gras: bool = False) -> float:
    """La largeur approchée d'un texte (sans balises), en unités du dessin."""
    return len(plain) * taille * (CAR + (0.03 if gras else 0.0))


def graduations(a: float, b: float, cible: int = 8) -> Tuple[float, float, float]:
    """Des graduations « rondes » couvrant [a, b] : (début, fin, pas) avec un pas de 1, 2, 2,5 ou 5 × 10ⁿ."""
    if b <= a:
        b = a + 1.0
    brut = (b - a) / cible
    p = 10 ** math.floor(math.log10(brut))
    pas = next(k * p for k in (1, 2, 2.5, 5, 10) if k * p >= brut)
    return math.floor(a / pas + 1e-9) * pas, math.ceil(b / pas - 1e-9) * pas, pas


def _ticks(a: float, b: float, pas: float) -> List[float]:
    """Les multiples de ``pas`` compris dans [a, b]."""
    k0, k1 = math.ceil(a / pas - 1e-9), math.floor(b / pas + 1e-9)
    return [k * pas for k in range(k0, k1 + 1)]


def _etiquette_pas(v: float, pas: float) -> str:
    nd = 0 if float(pas).is_integer() else (1 if (pas * 10).is_integer() else 2)
    return fr(v, nd)


def axe_resistances(ymin: float, ymax: float) -> Tuple[float, float, float]:
    """L'axe des résistances : de 0 (ou un peu sous le minimum quand une résistance est négative — au très jeune âge, par
    grand froid) à une graduation ronde au-dessus du maximum. Sous zéro, l'axe ne descend pas jusqu'à la graduation
    ronde suivante (−10 pour −1) : il s'arrête juste sous la valeur, les graduations restant des multiples du pas."""
    _, by, py = graduations(0.0, ymax * 1.06 + 1.0, 7)
    ay = 0.0 if ymin >= 0 else ymin - 0.04 * (by - ymin)
    return ay, by, py


class Repere:
    """Le repère d'un graphique : la zone de tracé en unités SVG et les échelles."""

    def __init__(self, x0, x1, y0, y1, ax, bx, ay, by):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.ax, self.bx, self.ay, self.by = ax, bx, ay, by

    def X(self, t: float) -> float:
        return self.x0 + (t - self.ax) / (self.bx - self.ax) * (self.x1 - self.x0)

    def Y(self, v: float) -> float:
        return self.y1 - (v - self.ay) / (self.by - self.ay) * (self.y1 - self.y0)

    def dedans_x(self, t: float) -> bool:
        return self.ax - 1e-9 <= t <= self.bx + 1e-9


def _chemin(rp: Repere, xs: Sequence[float], ys: Sequence[float]) -> str:
    pts = [f"{rp.X(x):.1f} {rp.Y(y):.1f}" for x, y in zip(xs, ys) if rp.dedans_x(x)]
    return ("M" + " L".join(pts)) if pts else ""


def _grille(rp: Repere, xt: List[float], yt: List[float], pas_x: float, pas_y: float) -> str:
    """Le fond, la grille, les axes gradués (comme la figure du module : grille fine, axes gris, pas de cadre haut/droit)."""
    g = [f'<rect x="{_f(rp.x0)}" y="{_f(rp.y0)}" width="{_f(rp.x1 - rp.x0)}" height="{_f(rp.y1 - rp.y0)}" fill="#FBFCFE"/>']
    for v in yt:
        y = rp.Y(v)
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(y)}" x2="{_f(rp.x1)}" y2="{_f(y)}" stroke="{GRILLE}" stroke-width="1"/>')
        g.append(_txt(rp.x0 - 8, y + 5.5, _etiquette_pas(v, pas_y), T_GRAD, "end", GRIS))
    for t in xt:
        x = rp.X(t)
        g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y0)}" x2="{_f(x)}" y2="{_f(rp.y1)}" stroke="{GRILLE}" stroke-width="1"/>')
        g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y1)}" x2="{_f(x)}" y2="{_f(rp.y1 + 5)}" stroke="{AXE}" stroke-width="1"/>')
        g.append(_txt(x, rp.y1 + 22, _etiquette_pas(t, pas_x), T_GRAD, "middle", GRIS))
    if rp.ay < 0 < rp.by:                                        # le zéro des résistances, appuyé
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.Y(0))}" x2="{_f(rp.x1)}" y2="{_f(rp.Y(0))}" stroke="{AXE}" stroke-width="1.2"/>')
    g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y1)}" x2="{_f(rp.x1)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>')
    g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y0)}" x2="{_f(rp.x0)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>')
    return "".join(g)


def _boite(x, y_haut, texte: str, couleur: str, taille: float = T_BOITE, balise: str = "") -> str:
    """Une étiquette encadrée posée par son coin haut-gauche (le point de l'âge choisi, comme l'annotation du module) ;
    ``balise`` : le même texte déjà mis en forme (un indice), la largeur se mesure sur ``texte``."""
    w, h = _dims_boite(texte, taille)
    return (f'<rect x="{_f(x)}" y="{_f(y_haut)}" width="{_f(w)}" height="{_f(h)}" rx="6" fill="#FFFFFF" fill-opacity=".95" '
            f'stroke="{couleur}" stroke-width="1.1"/>' + _txt(x + w / 2, y_haut + h - 7.5, balise or escape(texte), taille, "middle", couleur, True))


def _dims_boite(texte: str, taille: float = T_BOITE) -> Tuple[float, float]:
    return _largeur(texte, taille, True) + 16, taille + 12


def _marque(forme: str, x: float, y: float, couleur: str, r: float = 6.5) -> str:
    """Les marqueurs de la figure du module : rond (le point), triangle (l'optimisation), carré (le seuil), croix
    pleine (l'essai, le « P » de matplotlib)."""
    trait = 'stroke="#FFFFFF" stroke-width="1.6"'
    if forme == "rond":
        return f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="{couleur}" {trait}/>'
    if forme == "carre":
        return f'<rect x="{_f(x - r)}" y="{_f(y - r)}" width="{_f(2 * r)}" height="{_f(2 * r)}" fill="{couleur}" {trait}/>'
    if forme == "triangle":
        return (f'<path d="M{_f(x)} {_f(y - r * 1.15)} L{_f(x + r)} {_f(y + r * .75)} L{_f(x - r)} {_f(y + r * .75)} Z" '
                f'fill="{couleur}" {trait}/>')
    a, b = r * 1.05, r * 0.38                                 # la croix pleine
    d = (f"M{_f(x - b)} {_f(y - a)} H{_f(x + b)} V{_f(y - b)} H{_f(x + a)} V{_f(y + b)} H{_f(x + b)} V{_f(y + a)} "
         f"H{_f(x - b)} V{_f(y + b)} H{_f(x - a)} V{_f(y - b)} H{_f(x - b)} Z")
    return f'<path d="{d}" fill="{couleur}" {trait}/>'


def _ecarter(positions: List[Tuple[float, int]], ecart: float, haut: float, bas: float) -> Dict[int, float]:
    """Des étiquettes posées à droite des courbes, écartées d'au moins ``ecart`` sans quitter [haut, bas]."""
    ordre = sorted(positions)
    ys = [p for p, _ in ordre]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + ecart)
    if ys and ys[-1] > bas:
        ys[-1] = bas
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - ecart)
    if ys and ys[0] < haut:
        d = haut - ys[0]
        ys = [y + d for y in ys]
    return {k: y for (_, k), y in zip(ordre, ys)}


# ------------------------------------------------------------------ la légende, au-dessus du tracé
def _symbole(sorte: str, x: float, y: float, col: str) -> str:
    """Le petit dessin d'une entrée de légende (28 unités de large), à la manière du tracé qu'il nomme."""
    from beton.maturite.moteur import COULEURS_CMP
    ligne = lambda ep, extra="": (f'<line x1="{_f(x)}" y1="{_f(y)}" x2="{_f(x + 28)}" y2="{_f(y)}" stroke="{col}" '  # noqa: E731
                                  f'stroke-width="{ep}"{extra}/>')
    if sorte == "ref":
        return ligne(2.6) + _marque("rond", x + 14, y, col, 4.5)
    if sorte == "defav":
        return ligne(1.8, ' stroke-dasharray="7 5"')
    if sorte == "cmp":
        return "".join(f'<line x1="{_f(x + j * 10)}" y1="{_f(y)}" x2="{_f(x + j * 10 + 8)}" y2="{_f(y)}" stroke="{COULEURS_CMP[j]}" '
                       f'stroke-width="2.4"/>' for j in range(3))
    if sorte == "opt":
        return ligne(2.1) + _marque("triangle", x + 14, y, col, 5)
    if sorte == "seuil":
        return _marque("carre", x + 14, y, col, 5)
    if sorte == "lab":
        return _marque("croix", x + 14, y, col, 6)
    if sorte == "domaine":
        return f'<rect x="{_f(x + 3)}" y="{_f(y - 6)}" width="22" height="12" fill="{FOND_DOMAINE}" stroke="{GRIS_CLAIR}" stroke-dasharray="2 2"/>'
    if sorte == "barre":
        return f'<rect x="{_f(x + 6)}" y="{_f(y - 7)}" width="16" height="14" rx="2" fill="{col}" fill-opacity=".8"/>'
    if sorte == "creux":
        return f'<rect x="{_f(x + 6)}" y="{_f(y - 7)}" width="16" height="14" rx="2" fill="none" stroke="{col}" stroke-dasharray="3 2"/>'
    if sorte == "trait":
        return ligne(2.4)
    if sorte == "tirets":
        return ligne(1.4, ' stroke-dasharray="5 3"')
    return ligne(2.4) + f'<circle cx="{_f(x + 14)}" cy="{_f(y)}" r="3.5" fill="{col}"/>'      # « ligne » : t_e cumulé


def _legende(items: List[Tuple[str, str, str, str]], x0: float, x1: float, y: float) -> Tuple[str, float]:
    """La légende AU-DESSUS du tracé, en lignes qui se remplissent de gauche à droite ; ``items`` : (sorte, couleur,
    texte SVG, texte brut — pour la largeur). Rend (svg, hauteur occupée)."""
    g, x, yy = [], x0, y
    for sorte, col, t, brut in items:
        w = 36 + _largeur(brut, T_LEG) + 18
        if x + w - 18 > x1 and x > x0:
            x, yy = x0, yy + 24
        g.append(_symbole(sorte, x, yy, col))
        g.append(_txt(x + 36, yy + 5, t, T_LEG, "start", ENCRE))
        x += w
    return "".join(g), (yy - y + 24) if items else 0.0


# ------------------------------------------------------------------ la courbe f_ck(t)
LARGEUR = 580.0
H_TRACE = 430.0                # la hauteur de la zone de tracé (la légende s'ajoute au-dessus)
X_MAX_LISIBLE = 120.0          # au-delà, un repère (délai, essai) est signalé au bord plutôt que d'écraser la courbe
GAUCHE, DROITE, BAS = 58.0, 100.0, 80.0


def _place_etiquette(rp: Repere, x: float, y: float, texte: str, obstacles: List[Tuple[float, float, float, float]],
                     courbes: List[Tuple[Sequence[float], Sequence[float]]]) -> Tuple[float, float]:
    """Le coin haut-gauche de l'étiquette du point ① : la première des quatre places (au-dessus à droite, au-dessus à
    gauche, au-dessous à droite, au-dessous à gauche) qui tient dans le tracé sans couvrir un autre repère ; à égalité, celle
    que le moins de courbes traversent."""
    w, h = _dims_boite(texte)
    places = [(x + 10, y - 12 - h), (x - 10 - w, y - 12 - h), (x + 10, y + 12), (x - 10 - w, y + 12)]
    meilleur, score_min = None, None
    for rang, (bx, by) in enumerate(places):
        if bx < rp.x0 + 2 or bx + w > rp.x1 - 2 or by < rp.y0 + 2 or by + h > rp.y1 - 2:
            continue
        reperes = sum(1 for (ox0, oy0, ox1, oy1) in obstacles if ox0 < bx + w + 6 and ox1 > bx - 6 and oy0 < by + h + 6 and oy1 > by - 6)
        traverses = 0
        for xs, ys in courbes:
            if any(bx <= px <= bx + w and by - 2 <= py <= by + h + 2 for px, py in zip(xs, ys)):
                traverses += 1
        score = (reperes, traverses, rang)
        if score_min is None or score < score_min:
            meilleur, score_min = (bx, by), score
    if meilleur is None:                                     # rien ne tient : au-dessus, ramenée dans le tracé
        bx = min(max(x + 10, rp.x0 + 2), rp.x1 - 2 - w)
        meilleur = (bx, max(rp.y0 + 2, y - 12 - h))
    return meilleur


def courbe(ident: str, r: Dict) -> str:
    """La figure f_ck(t) d'un coulage, depuis la sortie de ``beton.maturite.moteur.calculer``."""
    from beton.maturite.moteur import COULEURS, COULEURS_CMP
    c = r["courbes"]
    t = c["t"]
    series = [c["ref"]["y"]] + ([c["defav"]["y"]] if c["defav"] else []) + [x["y"] for x in (c["comparateur"] or [])] \
        + ([c["opt"]["y"]] if c["opt"] else [])
    # les repères ponctuels (abscisse, ordonnée) : ils élargissent les axes comme matplotlib le fait
    reperes = [tuple(c["point"])]
    if c["seuil"]:
        reperes.append((c["seuil"]["t"], c["seuil"]["fck"]))
    if c["opt"]:
        reperes.append((c["opt"]["t"], c["opt"]["fck"]))
    if c["lab"]:
        reperes.append((c["lab"]["age"], c["lab"]["fck"]))
    xs_max = max([t[-1]] + [x for x, _ in reperes if x <= X_MAX_LISIBLE])
    ax, bx, px = graduations(0.0, xs_max, 8)
    ymin = min(min(min(s) for s in series), min(y for _, y in reperes), 0.0)
    ymax = max(max(max(s) for s in series), max(y for _, y in reperes))
    ay, by, py = axe_resistances(ymin, ymax)

    # ---- la légende d'abord : sa hauteur fixe le haut du tracé
    leg: List[Tuple[str, str, str, str]] = []
    nom_ref = c["ref"]["label"].replace(" — Classe ", " · ciment ")
    leg.append(("ref", COULEURS["ref"], escape(nom_ref), nom_ref))
    t3 = r["temperature"].get("t3_reel")
    if t3 and t3 > ax:
        leg.append(("domaine", GRIS_CLAIR, _te(" ≤ 3 j : hors du domaine de l'EC2"), "te ≤ 3 j : hors du domaine de l'EC2"))
    if c["defav"]:
        leg.append(("defav", COULEURS["defav"], escape(c["defav"]["label"]), c["defav"]["label"]))
    if c["comparateur"]:
        n = len(c["comparateur"])
        txt = f"comparateur · {n} classe{'s' if n > 1 else ''} · ciment {r['comparateur']['ciment']}"
        leg.append(("cmp", "", escape(txt), txt))
    if c["opt"]:
        txt = c["opt"]["label"].replace(" (optim.)", " · classe minimale ③")
        leg.append(("opt", COULEURS["opt"], escape(txt), txt))
    if c["seuil"]:
        txt = f"seuil ② : {fr(c['seuil']['fck'], 1)} MPa à {fr(c['seuil']['t'], 1)} j"
        leg.append(("seuil", COULEURS["seuil"], txt, txt))
    if c["lab"]:
        txt = f"essai ⑤ : {fr(c['lab']['fck'], 1)} MPa à {fr(c['lab']['age'], 1)} j"
        leg.append(("lab", COULEURS["mes"], txt, txt))
    svg_leg, h_leg = _legende(leg, GAUCHE - 30, LARGEUR - 8, 16.0)
    y0 = 16.0 + h_leg + 8.0
    hauteur = y0 + H_TRACE + BAS
    rp = Repere(GAUCHE, LARGEUR - DROITE, y0, y0 + H_TRACE, ax, bx, ay, by)
    g = [svg_leg, _grille(rp, _ticks(ax, bx, px), _ticks(ay, by, py), px, py)]

    # ---- le domaine t_e ≤ 3 j (hors du domaine de l'EC2, §3.1.2(5)), à la température de la courbe
    if t3 and t3 > ax:
        x3 = rp.X(min(t3, bx))
        g.append(f'<rect x="{_f(rp.x0)}" y="{_f(rp.y0)}" width="{_f(x3 - rp.x0)}" height="{_f(rp.y1 - rp.y0)}" fill="{FOND_DOMAINE}"/>'
                 f'<line x1="{_f(x3)}" y1="{_f(rp.y0)}" x2="{_f(x3)}" y2="{_f(rp.y1)}" stroke="{GRIS_CLAIR}" stroke-width="1" stroke-dasharray="2 3"/>')
    # ---- le repère t_e = 28 j
    t28 = c["t28"]
    if t28 is not None and t[0] < t28 < t[-1] and t28 <= bx:
        x = rp.X(t28)
        g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y0)}" x2="{_f(x)}" y2="{_f(rp.y1)}" stroke="#9CA3AF" stroke-width="1.2" stroke-dasharray="1.5 4" '
                 f'stroke-linecap="round"/>')
        g.append(_txt(x + 5, rp.y0 + 17, _te(" = 28 j"), T_NOM, "start", GRIS))
    # ---- la courbe de référence, sa surface
    d_ref = _chemin(rp, t, c["ref"]["y"])
    if d_ref:
        y_bas = rp.Y(max(ay, min(0.0, by)))
        g.append(f'<path d="{d_ref} L{_f(rp.X(t[-1]))} {_f(y_bas)} L{_f(rp.X(t[0]))} {_f(y_bas)} Z" fill="{COULEURS["ref"]}" fill-opacity=".07"/>')
    fin: List[Tuple[float, int, str, str]] = []                 # les étiquettes au bout des courbes
    if c["defav"]:
        g.append(f'<path d="{_chemin(rp, t, c["defav"]["y"])}" fill="none" stroke="{COULEURS["defav"]}" stroke-width="1.8" '
                 f'stroke-dasharray="7 5" stroke-linejoin="round"/>')
    for i, cmp_ in enumerate(c["comparateur"] or []):
        col = COULEURS_CMP[i % len(COULEURS_CMP)]
        g.append(f'<path d="{_chemin(rp, t, cmp_["y"])}" fill="none" stroke="{col}" stroke-width="1.7" stroke-opacity=".85" stroke-linejoin="round"/>')
        fin.append((rp.Y(cmp_["y"][-1]), len(fin), escape(cmp_["label"]), col))
    if c["opt"]:
        o = c["opt"]
        g.append(f'<path d="{_chemin(rp, t, o["y"])}" fill="none" stroke="{COULEURS["opt"]}" stroke-width="2.1" stroke-opacity=".9" stroke-linejoin="round"/>')
        nom_o = escape(o["label"].replace(" (optim.)", ""))
        y_o = rp.Y(o["y"][-1])
        double = next((k for k, (y, _, nom, _) in enumerate(fin) if nom == nom_o and abs(y - y_o) < 1.5), None)
        if double is not None:                                  # la même courbe au comparateur : une seule étiquette
            y, kk, _, _ = fin[double]
            fin[double] = (y, kk, nom_o + " ③", COULEURS["opt"])
        else:
            fin.append((y_o, len(fin), nom_o + " ③", COULEURS["opt"]))
    g.append(f'<path d="{d_ref}" fill="none" stroke="{COULEURS["ref"]}" stroke-width="2.6" stroke-linejoin="round"/>')
    nom_ref_fin = escape(r["materiau"]["classe"])
    meme = next((k for k, (_, _, nom, _) in enumerate(fin) if nom == nom_ref_fin + " ③"), None)
    if meme is not None:                                        # la classe minimale est celle du coulage : un seul nom
        del fin[meme]
        nom_ref_fin += " ③"
    fin = [(y, k, nom, col) for k, (y, _, nom, col) in enumerate(fin)]
    fin.append((rp.Y(c["ref"]["y"][-1]), len(fin), nom_ref_fin, COULEURS["ref"]))
    # ---- les noms au bout des courbes, écartés
    pos = _ecarter([(y, k) for y, k, _, _ in fin], T_NOM + 3, rp.y0 + 6, rp.y1 - 4)
    marques, obstacles = [], []
    for y, k, nom, col in fin:
        yy = pos[k]
        if abs(yy - y) > 2:
            g.append(f'<line x1="{_f(rp.X(t[-1]) + 3)}" y1="{_f(y)}" x2="{_f(rp.X(t[-1]) + 10)}" y2="{_f(yy)}" stroke="{col}" stroke-width="1"/>')
        g.append(_txt(rp.X(t[-1]) + 12, yy + 5, nom, T_NOM, "start", col, True))
        obstacles.append((rp.X(t[-1]) + 12, yy - 10, rp.X(t[-1]) + 12 + _largeur(nom.replace("&amp;", "&"), T_NOM, True), yy + 6))
    # ---- le seuil (la date d'obtention : le décoffrage)
    if c["seuil"]:
        s = c["seuil"]
        col = COULEURS["seuil"]
        y = rp.Y(s["fck"])
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(y)}" x2="{_f(rp.x1)}" y2="{_f(y)}" stroke="{col}" stroke-width="1.2" stroke-dasharray="6 4" stroke-opacity=".75"/>')
        if s["t"] <= bx:
            x = rp.X(s["t"])
            g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y1)}" x2="{_f(x)}" y2="{_f(y)}" stroke="{col}" stroke-width="1.2" stroke-dasharray="6 4" stroke-opacity=".75"/>')
            marques.append(_marque("carre", x, y, col, 6))
            obstacles.append((x - 6, y - 6, x + 6, y + 6))
            marques.append(_txt(x, rp.y1 + 42, f"{fr(s['t'], 1)} j", T_NOM, "middle", col, True))
    # ---- l'optimisation : le repère (délai, f_ck atteint)
    if c["opt"] and c["opt"].get("t") is not None:
        o = c["opt"]
        col = COULEURS["opt"]
        if o["t"] <= bx:
            x, y = rp.X(o["t"]), rp.Y(o["fck"])
            g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y1)}" x2="{_f(x)}" y2="{_f(y)}" stroke="{col}" stroke-width="1" stroke-dasharray="5 4" stroke-opacity=".55"/>')
            marques.append(_marque("triangle", x, y, col, 7))
            obstacles.append((x - 7, y - 8, x + 7, y + 6))
        else:
            marques.append(_hors_axe(rp, o["fck"], col, f"{fr(o['t'], 1)} j →", obstacles))
    # ---- l'essai de laboratoire
    if c["lab"]:
        lb = c["lab"]
        col = COULEURS["mes"]
        if lb["age"] <= bx:
            marques.append(_marque("croix", rp.X(lb["age"]), rp.Y(lb["fck"]), col, 7.5))
            obstacles.append((rp.X(lb["age"]) - 8, rp.Y(lb["fck"]) - 8, rp.X(lb["age"]) + 8, rp.Y(lb["fck"]) + 8))
        else:
            marques.append(_hors_axe(rp, lb["fck"], col, f"{fr(lb['age'], 1)} j →", obstacles))
    # ---- le point de l'âge choisi ①, et son étiquette là où elle ne cache rien
    ts, fs = c["point"]
    col = COULEURS["ref"]
    if ts <= bx:
        x, y = rp.X(ts), rp.Y(fs)
        g.append(f'<path d="M{_f(x)} {_f(rp.y1)} V{_f(y)} H{_f(rp.x0)}" fill="none" stroke="{col}" stroke-width="1.1" stroke-dasharray="5 4" stroke-opacity=".6"/>')
        marques.append(_marque("rond", x, y, col, 6.5))
        texte = f"{fr(fs, 1)} MPa à {ts:.0f} j"
        pas = max(1, len(t) // 120)
        traces = [([rp.X(v) for v in t[::pas]], [rp.Y(v) for v in s_[::pas]]) for s_ in series]
        bx_, by_ = _place_etiquette(rp, x, y, texte, obstacles, traces)
        marques.append(_boite(bx_, by_, texte, col))
    else:
        marques.append(_hors_axe(rp, fs, col, f"{ts:.0f} j →", obstacles))
    g += marques
    # ---- les titres des axes
    g.append(_txt((rp.x0 + rp.x1) / 2, hauteur - 10, "Âge du béton (jours réels)", T_TITRE, "middle", ENCRE, True))
    yt = (rp.y0 + rp.y1) / 2
    g.append(_txt(17, yt, _sub("f", "ck", "(t) [MPa]"), T_TITRE, "middle", ENCRE, True, extra=f' transform="rotate(-90 17 {_f(yt)})"'))
    return (f'<svg id="{escape(ident)}" class="age-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(LARGEUR)} {_f(hauteur)}" '
            f'width="100%" role="img" aria-label="Résistance du béton au jeune âge f_ck(t)">{"".join(g)}</svg>')


def _hors_axe(rp: Repere, y_val: float, col: str, texte: str, obstacles: List[Tuple[float, float, float, float]]) -> str:
    """Un repère au-delà de l'axe des âges : une flèche au bord droit, à sa hauteur, avec son âge — écrit sous la flèche
    (au-dessus si un autre repère y est déjà) ; la place prise s'ajoute aux ``obstacles``."""
    y = min(max(rp.Y(y_val), rp.y0 + 8), rp.y1 - 26)
    x = rp.x1 - 4
    w = _largeur(texte, T_NOM, True)
    yt = y + 22
    if any(ox0 < x and ox1 > x - 16 - w and oy0 < yt + 2 and oy1 > yt - 14 for ox0, oy0, ox1, oy1 in obstacles):
        yt = y - 10
    obstacles += [(x - 14, y - 7, x, y + 7), (x - 4 - w, yt - 14, x, yt + 3)]
    return (f'<path d="M{_f(x - 12)} {_f(y - 6)} L{_f(x)} {_f(y)} L{_f(x - 12)} {_f(y + 6)}" fill="none" stroke="{col}" stroke-width="2"/>'
            + _txt(x - 2, yt, escape(texte), T_NOM, "end", col, True))


# ------------------------------------------------------------------ le relevé des températures
H_RELEVE = 400.0


def releve(ident: str, r: Dict) -> str:
    """Le relevé jour par jour : une barre par jour (couleur de sa source), la variante défavorable des jours prévus,
    la température moyenne de la courbe, l'âge équivalent cumulé (échelle de droite) et l'âge choisi ①."""
    tp = r["temperature"]
    lignes = r["tableau"]["lignes"]
    n = len(lignes)
    temps = [li["T"] for li in lignes]
    t_def = [li.get("T_defav") for li in lignes] if r["tableau"]["defav"] else []
    vals = temps + [x for x in t_def if x is not None] + [tp["T"]]
    a_t, b_t, p_t = graduations(min(0.0, min(vals) - 1.0), max(vals) + 2.0, 6)
    te = [li["t_e"] for li in lignes]
    a_e, b_e, p_e = graduations(0.0, max(te) * 1.05 + 0.1, 6)
    larg = LARGEUR
    # la légende d'abord : sa hauteur fixe le haut du graphique
    rel = tp["releve"] or {}
    items = [("barre", SOURCES["mesure"][0], f"mesuré · {rel.get('n_mes', 0)} j")]
    if rel.get("n_prev"):
        items.append(("barre", SOURCES["prevision"][0], f"prévision · {rel['n_prev']} j"))
    if rel.get("n_extra"):
        items.append(("barre", SOURCES["prolonge"][0], f"prolongé · {rel['n_extra']} j"))
    if rel.get("n_nan"):
        items.append(("creux", GRIS, f"sans donnée · {rel['n_nan']} j"))
    if t_def:
        items.append(("trait", COL_DEFAV, "défavorable (− 3 °C)"))
    items.append(("tirets", ENCRE, f"T moyenne {fr(tp['T'], 1)} °C"))
    leg = [(k, col, escape(t), t) for k, col, t in items] + [("ligne", COL_TE, _te(" cumulé"), "te cumulé")]
    svg_leg, h_leg = _legende(leg, 26.0, larg - 8, 16.0)
    y0 = 16.0 + h_leg + 12.0
    hauteur = y0 + H_RELEVE + 84.0
    rp = Repere(56.0, larg - 50.0, y0, y0 + H_RELEVE, 0.5, n + 0.5, a_t, b_t)
    re_ = Repere(rp.x0, rp.x1, rp.y0, rp.y1, 0.5, n + 0.5, a_e, b_e)
    # les jours gradués : tous jusqu'à 12, sinon un pas rond ; les dates (JJ/MM) sous les jours quand elles tiennent
    pas_j = max(1, int(graduations(0, n, 8)[2])) if n > 12 else 1
    jours = [j for j in range(1, n + 1) if j == 1 or j % pas_j == 0]
    # une date n'est écrite que si elle a sa place après la précédente (le jour 1 et le jour 2 sont voisins quand les
    # autres jours vont de deux en deux : leurs dates se chevauchaient, « 24/1025/10 » au téléphone)
    l_date, x_date = _largeur("00/00", 13.0) + 6, float("-inf")
    g = [svg_leg, f'<rect x="{_f(rp.x0)}" y="{_f(rp.y0)}" width="{_f(rp.x1 - rp.x0)}" height="{_f(rp.y1 - rp.y0)}" fill="#FBFCFE"/>']
    for v in _ticks(a_t, b_t, p_t):
        y = rp.Y(v)
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(y)}" x2="{_f(rp.x1)}" y2="{_f(y)}" stroke="{GRILLE}"/>'
                 + _txt(rp.x0 - 8, y + 5.5, _etiquette_pas(v, p_t), T_GRAD, "end", GRIS))
    for v in _ticks(a_e, b_e, p_e):
        g.append(_txt(rp.x1 + 8, re_.Y(v) + 5.5, _etiquette_pas(v, p_e), T_GRAD, "start", COL_TE))
    dates = [li.get("date") for li in lignes]
    for j in jours:
        x = rp.X(j)
        g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y1)}" x2="{_f(x)}" y2="{_f(rp.y1 + 5)}" stroke="{AXE}"/>' + _txt(x, rp.y1 + 21, str(j), T_NOM, "middle", GRIS))
        if dates[j - 1] and x - x_date >= l_date:
            x_date = x
            _, m, jj = dates[j - 1].split("-")
            g.append(_txt(x, rp.y1 + 39, f"{jj}/{m}", 13.0, "middle", GRIS_CLAIR))
    # les barres
    w = min(30.0, (rp.x1 - rp.x0) / n * 0.72)
    y0b = rp.Y(max(a_t, min(0.0, b_t)))
    age = r["age"]["t"]
    for i, li in enumerate(lignes):
        x = rp.X(i + 1)
        y = rp.Y(li["T"])
        col = SOURCES[li["source"]][0]
        haut, bas = min(y, y0b), max(y, y0b)
        if li["T_brut"] is None:                         # un jour sans donnée : la moyenne, en creux
            g.append(f'<rect x="{_f(x - w / 2)}" y="{_f(haut)}" width="{_f(w)}" height="{_f(max(bas - haut, 1))}" fill="none" '
                     f'stroke="{col}" stroke-width="1.4" stroke-dasharray="3 2"/>')
        else:
            opa = ".8" if li["source"] == "mesure" else ".72"
            g.append(f'<rect x="{_f(x - w / 2)}" y="{_f(haut)}" width="{_f(w)}" height="{_f(max(bas - haut, 1))}" fill="{col}" fill-opacity="{opa}"'
                     + (' stroke="#15181F" stroke-width="1.8"' if i + 1 == age else "") + "/>")
    # la variante défavorable : les jours prévus abaissés de la marge
    if t_def:
        for i, (li, td) in enumerate(zip(lignes, t_def)):
            if li["source"] != "mesure" and td is not None:
                x = rp.X(i + 1)
                g.append(f'<line x1="{_f(x - w / 2 - 2)}" y1="{_f(rp.Y(td))}" x2="{_f(x + w / 2 + 2)}" y2="{_f(rp.Y(td))}" stroke="{COL_DEFAV}" stroke-width="2.4"/>')
    # la moyenne (la courbe, le seuil et la classe minimale sont calculés à cette température)
    ym = rp.Y(tp["T"])
    g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(ym)}" x2="{_f(rp.x1)}" y2="{_f(ym)}" stroke="{ENCRE}" stroke-width="1.3" stroke-dasharray="7 5"/>')
    # l'âge équivalent cumulé
    pts = " L".join(f"{_f(rp.X(i + 1))} {_f(re_.Y(v))}" for i, v in enumerate(te))
    g.append(f'<path d="M{pts}" fill="none" stroke="{COL_TE}" stroke-width="2.4" stroke-linejoin="round"/>')
    if n <= 60:
        g += [f'<circle cx="{_f(rp.X(i + 1))}" cy="{_f(re_.Y(v))}" r="3.5" fill="{COL_TE}"/>' for i, v in enumerate(te)]
    # l'âge choisi ① : son trait, son t_e
    if 1 <= age <= n:
        xa = rp.X(age)
        g.append(f'<line x1="{_f(xa)}" y1="{_f(rp.y0)}" x2="{_f(xa)}" y2="{_f(rp.y1)}" stroke="#2e6fb0" stroke-width="1.3" stroke-dasharray="3 3"/>')
        texte = f"{age} j : te = {fr(r['age']['t_e'], 1)} j"
        balise = f"{age} j : {_te(' = ' + fr(r['age']['t_e'], 1) + ' j')}"
        wtx, _ = _dims_boite(texte)
        bx_ = xa + 10 if xa + 10 + wtx < rp.x1 - 2 else xa - 10 - wtx
        g.append(_boite(max(rp.x0 + 2, bx_), rp.y0 + 8, texte, "#2e6fb0", T_BOITE, balise))
    # les axes
    g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y1)}" x2="{_f(rp.x1)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>'
             f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y0)}" x2="{_f(rp.x0)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>'
             f'<line x1="{_f(rp.x1)}" y1="{_f(rp.y0)}" x2="{_f(rp.x1)}" y2="{_f(rp.y1)}" stroke="{COL_TE}" stroke-width="1.3" stroke-opacity=".6"/>')
    if a_t < 0 < b_t:
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.Y(0))}" x2="{_f(rp.x1)}" y2="{_f(rp.Y(0))}" stroke="{AXE}" stroke-width="1.2"/>')
    # les titres des axes
    g.append(_txt((rp.x0 + rp.x1) / 2, hauteur - 10, "Jour depuis le coulage" + (" (et sa date)" if any(dates) else ""), T_TITRE, "middle", ENCRE, True))
    yt = (rp.y0 + rp.y1) / 2
    g.append(_txt(16, yt, "T moyenne du jour [°C]", T_TITRE, "middle", ENCRE, True, extra=f' transform="rotate(-90 16 {_f(yt)})"'))
    g.append(_txt(larg - 12, yt, _te(" cumulé [j]"), T_TITRE, "middle", COL_TE, True, extra=f' transform="rotate(90 {_f(larg - 12)} {_f(yt)})"'))
    return (f'<svg id="{escape(ident)}" class="age-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(larg)} {_f(hauteur)}" '
            f'width="100%" role="img" aria-label="Relevé des températures de cure">{"".join(g)}</svg>')


__all__ = ["courbe", "releve", "graduations", "axe_resistances", "fr"]

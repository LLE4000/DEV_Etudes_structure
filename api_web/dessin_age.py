# -*- coding: utf-8 -*-
"""Les DESSINS du module « Maturité du béton » : la courbe des résistances et le graphique des températures, en SVG.

Ce fichier ne calcule rien : il PLACE ce que le moteur ``beton/maturite`` rend (``courbes``, ``profil``, ``objectifs``,
``gel``, ``essai``).

- « Courbe » : f_cm(t) et f_ck(t) de 0 à l'horizon, en âge RÉEL (jours depuis la fin du bétonnage, et les dates sous
  l'axe quand la date est connue) ; chaque objectif : un trait fin à sa cible, un point là où il est atteint — sur LA
  courbe à laquelle il se compare —, son étiquette (« Décoffrage 21 MPa · J+10,7 ») posée là où elle ne couvre ni une
  autre étiquette ni un point ; la variante prudente en tirets ; le domaine t_T ≤ 3 j grisé (valeurs indicatives) ; le
  point de l'essai.
- « Températures » : une barre par jour civil, colorée par sa source (mesuré, prévision, prolongé, constante ; un jour
  sans donnée en creux), les jours de gel marqués, la température de la variante prudente, et l'âge équivalent cumulé
  sur l'axe de droite.

Les graphiques ont 580 unités de large : la feuille les met à l'échelle de l'écran (≈ 1 sur PC, ≈ 0,6 sur téléphone),
et les textes de 14 à 16 unités y restent lisibles. Ils sont posés sur le papier blanc de la feuille, en clair comme
en sombre : leurs couleurs sont fixes. La légende est AU-DESSUS du tracé. Tout texte venu de l'état (le nom d'un
objectif) est échappé ici.
"""
import datetime as _dt
import math
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

ENCRE, GRIS, GRIS_CLAIR, GRILLE, AXE = "#15181F", "#6B7686", "#98A2B3", "#E8EBF0", "#B8C0CC"
FOND_DOMAINE = "#F1F2F5"
POLICE = "system-ui,Arial,sans-serif"
COL_FCK, COL_FCM, COL_OBJ, COL_ESSAI, COL_TE, COL_GEL, COL_PRUDENT = "#2e6fb0", "#1f8a70", "#b45309", "#7c3aed", "#1f8a70", "#0284c7", "#b91c1c"
SOURCES = {"mesure": ("#2e6fb0", "mesuré"), "prevision": ("#d97706", "prévision"), "prolonge": ("#9AA3AF", "prolongé"),
           "constante": ("#5B7BA6", "constante"), "sans_donnee": ("#6B7686", "sans donnée")}
# les tailles de texte (unités du dessin) : graduations, noms au bout des courbes, titres des axes, légende, étiquettes
T_GRAD, T_NOM, T_TITRE, T_LEG, T_BOITE, T_DATE = 15.0, 14.5, 15.5, 14.0, 13.5, 12.5
CAR = 0.56                     # largeur moyenne d'un caractère, en fraction de la taille (system-ui, Arial)
LARGEUR = 580.0
H_COURBE, H_TEMP = 400.0, 380.0
NOM_MAX = 22                   # un nom d'objectif plus long est coupé dans son étiquette (« … »)


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


def _largeur(plain: str, taille: float, gras: bool = False) -> float:
    """La largeur approchée d'un texte (sans balises), en unités du dessin."""
    return len(plain) * taille * (CAR + (0.03 if gras else 0.0))


def graduations(a: float, b: float, cible: int = 8) -> Tuple[float, float, float]:
    """Des graduations « rondes » couvrant [a, b] : (début, fin, pas) avec un pas de 1, 2, 2,5 ou 5 × 10ⁿ."""
    if b <= a:
        b = a + 1.0
    brut = (b - a) / cible
    p = 10 ** math.floor(math.log10(brut))
    pas = next(k * p for k in (1, 2, 2.5, 5, 10) if k * p >= brut - 1e-12)
    return math.floor(a / pas + 1e-9) * pas, math.ceil(b / pas - 1e-9) * pas, pas


def pas_jours(horizon: float, cible: int = 8) -> float:
    """Le pas des graduations d'un axe en JOURS : 1, 2, 5, 7, 10, 14, 20… (une semaine se lit mieux que 2,5 jours)."""
    brut = horizon / cible
    return float(next((p for p in (1, 2, 5, 7, 10, 14, 20, 28, 50) if p >= brut - 1e-9), 100))


def _ticks(a: float, b: float, pas: float) -> List[float]:
    """Les multiples de ``pas`` compris dans [a, b]."""
    k0, k1 = math.ceil(a / pas - 1e-9), math.floor(b / pas + 1e-9)
    return [k * pas for k in range(k0, k1 + 1)]


def _etiquette_pas(v: float, pas: float) -> str:
    nd = 0 if float(pas).is_integer() else (1 if (pas * 10).is_integer() else 2)
    return fr(v, nd)


class Repere:
    """Le repère d'un graphique : la zone de tracé en unités SVG et les échelles."""

    def __init__(self, x0, x1, y0, y1, ax, bx, ay, by):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.ax, self.bx, self.ay, self.by = ax, bx, ay, by

    def X(self, t: float) -> float:
        return self.x0 + (t - self.ax) / (self.bx - self.ax) * (self.x1 - self.x0)

    def Y(self, v: float) -> float:
        return self.y1 - (v - self.ay) / (self.by - self.ay) * (self.y1 - self.y0)


def _chemin(rp: Repere, xs: Sequence[float], ys: Sequence[float]) -> str:
    pts = [f"{rp.X(x):.1f} {rp.Y(y):.1f}" for x, y in zip(xs, ys)]
    return ("M" + " L".join(pts)) if pts else ""


def _date_fr(iso: Optional[str], j: float = 0.0) -> str:
    """« JJ/MM » du jour ``iso`` + j jours ; « » sans date."""
    if not iso:
        return ""
    try:
        return (_dt.date.fromisoformat(iso) + _dt.timedelta(days=int(round(j)))).strftime("%d/%m")
    except (OverflowError, ValueError):
        return ""


def _axe_x(rp: Repere, pas: float, date: Optional[str], decalage: float = 0.0) -> str:
    """L'axe des âges : les graduations en jours et, quand la date est connue, la DATE sous chaque graduation — écrite
    seulement si elle a sa place après la précédente (deux dates ne se touchent jamais)."""
    g = []
    l_date, x_date = _largeur("00/00", T_DATE) + 8, float("-inf")
    for t in _ticks(rp.ax, rp.bx, pas):
        x = rp.X(t)
        g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y0)}" x2="{_f(x)}" y2="{_f(rp.y1)}" stroke="{GRILLE}" stroke-width="1"/>'
                 f'<line x1="{_f(x)}" y1="{_f(rp.y1)}" x2="{_f(x)}" y2="{_f(rp.y1 + 5)}" stroke="{AXE}" stroke-width="1"/>')
        g.append(_txt(x, rp.y1 + 21, _etiquette_pas(t, pas), T_GRAD, "middle", GRIS))
        if date and x - x_date >= l_date:
            x_date = x
            g.append(_txt(x, rp.y1 + 38, _date_fr(date, t + decalage), T_DATE, "middle", GRIS_CLAIR))
    return "".join(g)


def _marque(forme: str, x: float, y: float, couleur: str, r: float = 6.0, creux: bool = False) -> str:
    """Un marqueur : rond (un objectif atteint), losange (l'âge prévu d'une opération), croix pleine (l'essai)."""
    trait = f'stroke="{couleur}" stroke-width="1.8" fill="#FFFFFF"' if creux else f'fill="{couleur}" stroke="#FFFFFF" stroke-width="1.6"'
    if forme == "rond":
        return f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" {trait}/>'
    if forme == "losange":
        return f'<path d="M{_f(x)} {_f(y - r * 1.2)} L{_f(x + r)} {_f(y)} L{_f(x)} {_f(y + r * 1.2)} L{_f(x - r)} {_f(y)} Z" {trait}/>'
    a, b = r * 1.05, r * 0.38                                 # la croix pleine
    d = (f"M{_f(x - b)} {_f(y - a)} H{_f(x + b)} V{_f(y - b)} H{_f(x + a)} V{_f(y + b)} H{_f(x + b)} V{_f(y + a)} "
         f"H{_f(x - b)} V{_f(y + b)} H{_f(x - a)} V{_f(y - b)} H{_f(x - b)} Z")
    return f'<path d="{d}" {trait}/>'


# ------------------------------------------------------------------ la légende, au-dessus du tracé
def _symbole(sorte: str, x: float, y: float, col: str) -> str:
    """Le petit dessin d'une entrée de légende (28 unités de large), à la manière du tracé qu'il nomme."""
    def ligne(ep, extra=""):
        return f'<line x1="{_f(x)}" y1="{_f(y)}" x2="{_f(x + 28)}" y2="{_f(y)}" stroke="{col}" stroke-width="{ep}"{extra}/>'
    if sorte == "trait":
        return ligne(2.6)
    if sorte == "fin":
        return ligne(2.0)
    if sorte == "tirets":
        return ligne(1.6, ' stroke-dasharray="6 4"')
    if sorte == "point":
        return ligne(1.0, ' stroke-dasharray="2 3"') + _marque("rond", x + 14, y, col, 5)
    if sorte == "losange":
        return _marque("losange", x + 14, y, col, 5, creux=True)
    if sorte == "croix":
        return _marque("croix", x + 14, y, col, 6)
    if sorte == "domaine":
        return f'<rect x="{_f(x + 3)}" y="{_f(y - 6)}" width="22" height="12" fill="{FOND_DOMAINE}" stroke="{GRIS_CLAIR}" stroke-dasharray="2 2"/>'
    if sorte == "barre":
        return f'<rect x="{_f(x + 6)}" y="{_f(y - 7)}" width="16" height="14" rx="2" fill="{col}" fill-opacity=".8"/>'
    if sorte == "creux":
        return f'<rect x="{_f(x + 6)}" y="{_f(y - 7)}" width="16" height="14" rx="2" fill="none" stroke="{col}" stroke-dasharray="3 2"/>'
    if sorte == "gel":
        return f'<rect x="{_f(x + 4)}" y="{_f(y - 7)}" width="20" height="14" rx="2" fill="{col}" fill-opacity=".16"/>' + _flocon(x + 14, y, 5.5, col)
    if sorte == "cran":
        return f'<line x1="{_f(x + 4)}" y1="{_f(y)}" x2="{_f(x + 24)}" y2="{_f(y)}" stroke="{col}" stroke-width="2.4"/>'
    return ligne(2.4) + f'<circle cx="{_f(x + 14)}" cy="{_f(y)}" r="3.2" fill="{col}"/>'      # « ligne » : t_T cumulé


def _legende(items: List[Tuple[str, str, str, str]], x0: float, x1: float, y: float) -> Tuple[str, float]:
    """La légende AU-DESSUS du tracé, en lignes qui se remplissent de gauche à droite ; ``items`` : (sorte, couleur,
    texte SVG, texte brut — pour la largeur). Rend (svg, hauteur occupée)."""
    g, x, yy = [], x0, y
    for sorte, col, t, brut in items:
        w = 36 + _largeur(brut, T_LEG) + 16
        if x + w - 16 > x1 and x > x0:
            x, yy = x0, yy + 23
        g.append(_symbole(sorte, x, yy, col))
        g.append(_txt(x + 36, yy + 5, t, T_LEG, "start", ENCRE))
        x += w
    return "".join(g), (yy - y + 23) if items else 0.0


def _flocon(x: float, y: float, r: float, col: str) -> str:
    """Un petit flocon (trois traits croisés) : le jour de gel."""
    out = ""
    for a in (0, 60, 120):
        dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        out += f'<line x1="{_f(x - dx)}" y1="{_f(y - dy)}" x2="{_f(x + dx)}" y2="{_f(y + dy)}" stroke="{col}" stroke-width="1.5" stroke-linecap="round"/>'
    return out


# ------------------------------------------------------------------ les étiquettes qui ne se chevauchent pas
Boite = Tuple[float, float, float, float]                     # (gauche, haut, droite, bas)


def _libre(b: Boite, obstacles: List[Boite], marge: float = 3.0) -> bool:
    return not any(o[0] < b[2] + marge and o[2] > b[0] - marge and o[1] < b[3] + marge and o[3] > b[1] - marge for o in obstacles)


def poser_etiquette(rp: Repere, x: float, y: float, w: float, h: float, obstacles: List[Boite]) -> Boite:
    """La boîte d'une étiquette de ``w`` × ``h`` près du point (x, y) : la première place — sous la courbe à droite,
    au-dessus à gauche, puis de plus en plus loin, de part et d'autre — qui tient dans le tracé sans toucher un
    obstacle (une autre étiquette, un point). À défaut, la place la moins mauvaise, ramenée dans le tracé."""
    places = []
    for n in range(0, 9):
        dy = n * (h + 4)
        places += [(x + 9, y + 7 + dy), (x - 9 - w, y - h - 7 - dy), (x + 9, y - h - 7 - dy), (x - 9 - w, y + 7 + dy)]
    for bx, by in places:
        b = (bx, by, bx + w, by + h)
        if bx >= rp.x0 + 2 and bx + w <= rp.x1 - 2 and by >= rp.y0 + 2 and by + h <= rp.y1 - 2 and _libre(b, obstacles):
            return b
    bx = min(max(x + 9, rp.x0 + 2), rp.x1 - 2 - w)
    by = min(max(y + 7, rp.y0 + 2), rp.y1 - 2 - h)
    return (bx, by, bx + w, by + h)


def _etiquette(b: Boite, texte: str, couleur: str, ancre_pt: Optional[Tuple[float, float]] = None) -> str:
    """Une étiquette encadrée (fond blanc) ; ``ancre_pt`` : le point qu'elle nomme — un filet l'y relie quand elle en
    est éloignée."""
    x0, y0, x1, y1 = b
    g = ""
    if ancre_pt is not None:
        px, py = ancre_pt
        cx, cy = min(max(px, x0), x1), min(max(py, y0), y1)
        if math.hypot(cx - px, cy - py) > 14:
            g += f'<line x1="{_f(px)}" y1="{_f(py)}" x2="{_f(cx)}" y2="{_f(cy)}" stroke="{couleur}" stroke-width="1" stroke-opacity=".7"/>'
    return (g + f'<rect x="{_f(x0)}" y="{_f(y0)}" width="{_f(x1 - x0)}" height="{_f(y1 - y0)}" rx="5" fill="#FFFFFF" fill-opacity=".94" '
            f'stroke="{couleur}" stroke-width="1"/>' + _txt((x0 + x1) / 2, y1 - 5.5, escape(texte), T_BOITE, "middle", ENCRE, True))


def _nom_court(nom: str) -> str:
    return nom if len(nom) <= NOM_MAX else nom[:NOM_MAX - 1].rstrip() + "…"


def etiquette_objectif(o: Dict) -> str:
    """Le texte de l'étiquette d'un objectif : « Décoffrage 21 MPa · J+10,7 » (ou « · non atteignable », « · jamais »)."""
    quand = f"J+{fr(o['t'], 1)}" if o["t"] is not None else ("non atteignable" if o["etat"] in ("non_atteignable",) or o.get("non_atteignable") else "jamais")
    return f"{_nom_court(o['nom'])} {fr(o['cible'], 1 if abs(o['cible'] - round(o['cible'])) > 1e-9 else 0)} MPa · {quand}"


# ------------------------------------------------------------------ la courbe des résistances
def courbe(ident: str, r: Dict) -> str:
    """La figure f_cm(t), f_ck(t) d'un coulage, depuis la sortie de ``beton.maturite.moteur.calculer``."""
    c, m = r["courbes"], r["materiau"]
    t, H = c["t"], c["horizon"]
    pas_x = pas_jours(H)
    ymax = max(m["fcm"], max((o["cible"] for o in c["objectifs"]), default=0.0), c["essai"]["f"] if c["essai"] else 0.0)
    _, by, py = graduations(0.0, ymax * 1.04, 7)
    # ---- la légende d'abord : sa hauteur fixe le haut du tracé
    leg: List[Tuple[str, str, str, str]] = [("trait", COL_FCK, _sub("f", "ck", "(t) · caractéristique"), "fck(t) · caractéristique"),
                                            ("fin", COL_FCM, _sub("f", "cm", "(t) · moyenne"), "fcm(t) · moyenne")]
    if c["fck_prudent"]:
        txt = f"variante prudente (− {fr(r['temperature']['marge'], 1).replace(',0', '')} °C)"
        leg.append(("tirets", COL_FCK, escape(txt), txt))
    if c["t_3"] is not None and c["t_3"] > 0:
        leg.append(("domaine", GRIS_CLAIR, _sub("t", "T", " ≤ 3 j : valeurs indicatives"), "tT ≤ 3 j : valeurs indicatives"))
    leg.append(("point", COL_OBJ, "objectif atteint", "objectif atteint"))
    if any(o["age_prevu"] is not None for o in c["objectifs"]):
        leg.append(("losange", COL_OBJ, "âge prévu", "âge prévu"))
    if c["essai"]:
        leg.append(("croix", COL_ESSAI, "essai sur éprouvettes", "essai sur éprouvettes"))
    svg_leg, h_leg = _legende(leg, 26.0, LARGEUR - 8, 16.0)
    y0 = 16.0 + h_leg + 6.0
    bas = 66.0 if c["date"] else 50.0
    hauteur = y0 + H_COURBE + bas
    rp = Repere(52.0, LARGEUR - 46.0, y0, y0 + H_COURBE, 0.0, H, 0.0, by)
    g = [svg_leg, f'<rect x="{_f(rp.x0)}" y="{_f(rp.y0)}" width="{_f(rp.x1 - rp.x0)}" height="{_f(rp.y1 - rp.y0)}" fill="#FBFCFE"/>']
    # ---- le domaine t_T ≤ 3 j (valeurs indicatives, EN 1992-1-1 §3.1.2(5)), sous la grille
    t3 = c["t_3"]
    x3 = rp.X(min(t3, H)) if t3 is not None else rp.x1
    if x3 > rp.x0 + 0.5:
        g.append(f'<rect x="{_f(rp.x0)}" y="{_f(rp.y0)}" width="{_f(x3 - rp.x0)}" height="{_f(rp.y1 - rp.y0)}" fill="{FOND_DOMAINE}"/>')
    # ---- la grille et les axes
    for v in _ticks(0.0, by, py):
        y = rp.Y(v)
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(y)}" x2="{_f(rp.x1)}" y2="{_f(y)}" stroke="{GRILLE}" stroke-width="1"/>')
        g.append(_txt(rp.x0 - 8, y + 5, _etiquette_pas(v, py), T_GRAD, "end", GRIS))
    g.append(_axe_x(rp, pas_x, c["date"]))
    if x3 > rp.x0 + 0.5 and t3 is not None and t3 < H:
        g.append(f'<line x1="{_f(x3)}" y1="{_f(rp.y0)}" x2="{_f(x3)}" y2="{_f(rp.y1)}" stroke="{GRIS_CLAIR}" stroke-width="1" stroke-dasharray="2 3"/>')
    g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y1)}" x2="{_f(rp.x1)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>'
             f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y0)}" x2="{_f(rp.x0)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>')
    # ---- les courbes : la variante prudente (tirets) dessous, puis f_cm (fine) et f_ck (appuyée, sa surface)
    d_ck = _chemin(rp, t, c["fck"])
    g.append(f'<path d="{d_ck} L{_f(rp.X(t[-1]))} {_f(rp.y1)} L{_f(rp.X(t[0]))} {_f(rp.y1)} Z" fill="{COL_FCK}" fill-opacity=".06"/>')
    if c["fck_prudent"]:
        g.append(f'<path d="{_chemin(rp, t, c["fcm_prudent"])}" fill="none" stroke="{COL_FCM}" stroke-width="1.3" stroke-dasharray="6 4" stroke-opacity=".8"/>')
        g.append(f'<path d="{_chemin(rp, t, c["fck_prudent"])}" fill="none" stroke="{COL_FCK}" stroke-width="1.6" stroke-dasharray="6 4" stroke-opacity=".85"/>')
    g.append(f'<path d="{_chemin(rp, t, c["fcm"])}" fill="none" stroke="{COL_FCM}" stroke-width="2" stroke-linejoin="round"/>')
    g.append(f'<path d="{d_ck}" fill="none" stroke="{COL_FCK}" stroke-width="2.6" stroke-linejoin="round"/>')
    # ---- les noms au bout des courbes (à droite du tracé)
    y_cm, y_ck = rp.Y(c["fcm"][-1]), rp.Y(c["fck"][-1])
    if y_ck - y_cm < T_NOM + 2:
        y_ck = y_cm + T_NOM + 2
    g.append(_txt(rp.x1 + 6, y_cm + 5, _sub("f", "cm"), T_NOM, "start", COL_FCM, True))
    g.append(_txt(rp.x1 + 6, y_ck + 5, _sub("f", "ck"), T_NOM, "start", COL_FCK, True))
    obstacles: List[Boite] = []
    marques, etiquettes = [], []
    # ---- la mention du domaine indicatif (en haut du domaine grisé, si elle y tient)
    if x3 - rp.x0 > 30:
        mention = "indicatif"
        wm = _largeur(mention, 12.0)
        if x3 - rp.x0 >= wm + 8:
            g.append(_txt(rp.x0 + 5, rp.y0 + 15, mention, 12.0, "start", GRIS))
            obstacles.append((rp.x0 + 3, rp.y0 + 3, rp.x0 + 7 + wm, rp.y0 + 19))
        else:
            xm = (rp.x0 + x3) / 2 + 4
            g.append(_txt(xm, rp.y0 + 6, mention, 12.0, "end", GRIS, extra=f' transform="rotate(-90 {_f(xm)} {_f(rp.y0 + 6)})"'))
            obstacles.append((xm - 13, rp.y0 + 3, xm + 3, rp.y0 + 8 + wm))
    # ---- les objectifs : le trait fin à la cible, le point où il est atteint, l'âge prévu
    points = []
    for o in c["objectifs"]:
        yc = rp.Y(min(o["cible"], by))
        dans = o["t"] is not None and o["t"] <= H + 1e-9
        xf = rp.X(o["t"]) if dans else rp.x1
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(yc)}" x2="{_f(xf)}" y2="{_f(yc)}" stroke="{COL_OBJ}" stroke-width="1" stroke-dasharray="2 3" stroke-opacity=".9"/>')
        if dans:
            g.append(f'<line x1="{_f(xf)}" y1="{_f(yc)}" x2="{_f(xf)}" y2="{_f(rp.y1)}" stroke="{COL_OBJ}" stroke-width="1" stroke-dasharray="2 3" stroke-opacity=".9"/>')
            marques.append(_marque("rond", xf, yc, COL_OBJ, 5.5))
            obstacles.append((xf - 7, yc - 7, xf + 7, yc + 7))
            if o["t_prudent"] is not None and o["t_prudent"] <= H and c["fck_prudent"] and abs(o["t_prudent"] - o["t"]) > 1e-9:
                xp = rp.X(o["t_prudent"])
                marques.append(_marque("rond", xp, yc, COL_OBJ, 4.2, creux=True))
                obstacles.append((xp - 5, yc - 5, xp + 5, yc + 5))
            points.append((o, xf, yc))
        else:
            points.append((o, rp.x1 - 4, yc))
        if o["age_prevu"] is not None and o["age_prevu"] <= H and o["f_prevu"] is not None:
            xa, ya = rp.X(o["age_prevu"]), rp.Y(min(o["f_prevu"], by))
            col = COL_PRUDENT if o["etat"] == "nok" else COL_OBJ
            g.append(f'<line x1="{_f(xa)}" y1="{_f(rp.y1)}" x2="{_f(xa)}" y2="{_f(min(ya, yc))}" stroke="{col}" stroke-width="1.1" stroke-opacity=".8"/>')
            marques.append(_marque("losange", xa, ya, col, 5.5, creux=True))
            obstacles.append((xa - 7, ya - 8, xa + 7, ya + 8))
    # ---- l'essai
    if c["essai"] and c["essai"]["t"] <= H:
        xe, ye = rp.X(c["essai"]["t"]), rp.Y(min(c["essai"]["f"], by))
        marques.append(_marque("croix", xe, ye, COL_ESSAI, 7))
        obstacles.append((xe - 8, ye - 8, xe + 8, ye + 8))
        txt = f"essai {fr(c['essai']['f'], 1)} MPa"
        b = poser_etiquette(rp, xe, ye, _largeur(txt, T_BOITE, True) + 12, T_BOITE + 9, obstacles)
        obstacles.append(b)
        etiquettes.append(_etiquette(b, txt, COL_ESSAI, (xe, ye)))
    # ---- les étiquettes des objectifs, posées une à une sans chevauchement (les plus précoces d'abord)
    for o, x, y in sorted(points, key=lambda p: (p[1], p[2])):
        txt = etiquette_objectif(o)
        b = poser_etiquette(rp, x, y, _largeur(txt, T_BOITE, True) + 12, T_BOITE + 9, obstacles)
        obstacles.append(b)
        etiquettes.append(_etiquette(b, txt, COL_OBJ, (x, y)))
    g += marques + etiquettes
    # ---- les titres des axes
    g.append(_txt((rp.x0 + rp.x1) / 2, hauteur - 9, "Âge réel depuis la fin du bétonnage (jours)" + (" · dates" if c["date"] else ""), T_TITRE, "middle", ENCRE, True))
    yt = (rp.y0 + rp.y1) / 2
    g.append(_txt(15, yt, "Résistance (MPa)", T_TITRE, "middle", ENCRE, True, extra=f' transform="rotate(-90 15 {_f(yt)})"'))
    return (f'<svg id="{escape(ident)}" class="age-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(LARGEUR)} {_f(hauteur)}" '
            f'width="100%" role="img" aria-label="Résistance du béton au jeune âge : moyenne et caractéristique">{"".join(g)}</svg>')


# ------------------------------------------------------------------ les températures jour par jour
def temperatures(ident: str, r: Dict) -> str:
    """Le profil jour par jour : une barre par jour civil (la couleur de sa source), les jours de gel, la température de
    la variante prudente, l'âge équivalent cumulé (échelle de droite), et le jour où chaque objectif est atteint."""
    jours = r["profil"]
    n = len(jours)
    marge = r["temperature"]["marge"]
    prudent = r["synthese"]["prudent_distinct"]
    vals = [j["T"] for j in jours if j["T"] is not None] + [j["T_gel"] for j in jours if j["T_gel"] is not None]
    if prudent:
        vals += [j["T_prudent"] for j in jours if j["T_prudent"] is not None]
    a_t, b_t, p_t = graduations(min(0.0, min(vals, default=0.0) - 1.0), max(vals, default=20.0) + 2.0, 6)
    te = [j["t_T"] for j in jours]
    _, b_e, p_e = graduations(0.0, max(te) * 1.05 + 0.1, 6)
    # ---- la légende
    comptes = {}
    for j in jours:
        comptes[j["source"]] = comptes.get(j["source"], 0) + 1
    items = [("creux" if k == "sans_donnee" else "barre", SOURCES[k][0], SOURCES[k][1]) for k in ("mesure", "prevision", "prolonge", "constante", "sans_donnee")
             if comptes.get(k)]
    if prudent:
        items.append(("cran", COL_PRUDENT, f"prudent (− {fr(marge, 1).replace(',0', '')} °C)"))
    if any(j["gel"] for j in jours):
        items.append(("gel", COL_GEL, "jour de gel"))
    leg = [(k, col, escape(t), t) for k, col, t in items] + [("ligne", COL_TE, _sub("t", "T", " cumulé"), "tT cumulé")]
    svg_leg, h_leg = _legende(leg, 26.0, LARGEUR - 8, 16.0)
    y0 = 16.0 + h_leg + 8.0
    date = jours[0]["date"]
    hauteur = y0 + H_TEMP + (66.0 if date else 50.0)
    rp = Repere(52.0, LARGEUR - 50.0, y0, y0 + H_TEMP, -0.5, n - 0.5, a_t, b_t)
    re_ = Repere(rp.x0, rp.x1, rp.y0, rp.y1, -0.5, n - 0.5, 0.0, b_e)
    g = [svg_leg, f'<rect x="{_f(rp.x0)}" y="{_f(rp.y0)}" width="{_f(rp.x1 - rp.x0)}" height="{_f(rp.y1 - rp.y0)}" fill="#FBFCFE"/>']
    w = min(30.0, (rp.x1 - rp.x0) / n * 0.74)
    # ---- les jours de gel : une bande sur toute la hauteur, un flocon en tête
    for j in jours:
        if j["gel"]:
            x = rp.X(j["jour"])
            pas = (rp.x1 - rp.x0) / n
            g.append(f'<rect x="{_f(x - pas / 2)}" y="{_f(rp.y0)}" width="{_f(pas)}" height="{_f(rp.y1 - rp.y0)}" fill="{COL_GEL}" fill-opacity=".12"/>')
    for v in _ticks(a_t, b_t, p_t):
        y = rp.Y(v)
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(y)}" x2="{_f(rp.x1)}" y2="{_f(y)}" stroke="{GRILLE}"/>'
                 + _txt(rp.x0 - 8, y + 5, _etiquette_pas(v, p_t), T_GRAD, "end", GRIS))
    for v in _ticks(0.0, b_e, p_e):
        g.append(_txt(rp.x1 + 8, re_.Y(v) + 5, _etiquette_pas(v, p_e), T_GRAD, "start", COL_TE))
    # ---- l'axe des jours : le rang du jour (0 : le bétonnage) et sa date
    pas_j = pas_jours(n, 9)
    l_date, x_date = _largeur("00/00", T_DATE) + 8, float("-inf")
    for k in range(0, n, int(pas_j)):
        x = rp.X(k)
        g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y1)}" x2="{_f(x)}" y2="{_f(rp.y1 + 5)}" stroke="{AXE}"/>' + _txt(x, rp.y1 + 21, str(k), T_GRAD, "middle", GRIS))
        if date and x - x_date >= l_date:
            x_date = x
            g.append(_txt(x, rp.y1 + 38, _date_fr(date, k), T_DATE, "middle", GRIS_CLAIR))
    # ---- les barres
    y0b = rp.Y(max(a_t, min(0.0, b_t)))
    for j in jours:
        x = rp.X(j["jour"])
        col = SOURCES[j["source"]][0]
        if j["T"] is None:                               # un jour sans donnée : en creux, sur toute la hauteur utile
            g.append(f'<rect x="{_f(x - w / 2)}" y="{_f(rp.y0 + 14)}" width="{_f(w)}" height="{_f(rp.y1 - rp.y0 - 14)}" fill="none" '
                     f'stroke="{col}" stroke-width="1.2" stroke-dasharray="3 3" stroke-opacity=".7"/>')
            continue
        y = rp.Y(j["T"])
        haut, bas = min(y, y0b), max(y, y0b)
        g.append(f'<rect x="{_f(x - w / 2)}" y="{_f(haut)}" width="{_f(w)}" height="{_f(max(bas - haut, 1.5))}" fill="{col}" fill-opacity=".78"/>')
        if prudent and j["T_prudent"] is not None and abs(j["T_prudent"] - j["T"]) > 1e-9:
            yp = rp.Y(j["T_prudent"])
            g.append(f'<line x1="{_f(x - w / 2 - 1.5)}" y1="{_f(yp)}" x2="{_f(x + w / 2 + 1.5)}" y2="{_f(yp)}" stroke="{COL_PRUDENT}" stroke-width="2"/>')
    for j in jours:
        if j["gel"]:
            g.append(_flocon(rp.X(j["jour"]), rp.y0 + 9, min(5.5, max(3.0, w / 2)), COL_GEL))
    # ---- l'âge équivalent cumulé, en fin de jour (le bord droit de chaque barre)
    pas = (rp.x1 - rp.x0) / n
    pts = [f"{_f(rp.x0)} {_f(re_.Y(0.0))}"] + [f"{_f(rp.X(j['jour']) + pas / 2)} {_f(re_.Y(j['t_T']))}" for j in jours]
    g.append(f'<path d="M{" L".join(pts)}" fill="none" stroke="{COL_TE}" stroke-width="2.2" stroke-linejoin="round"/>')
    if n <= 45:
        g += [f'<circle cx="{_f(rp.X(j["jour"]) + pas / 2)}" cy="{_f(re_.Y(j["t_T"]))}" r="2.8" fill="{COL_TE}"/>' for j in jours]
    # ---- les axes
    g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y1)}" x2="{_f(rp.x1)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>'
             f'<line x1="{_f(rp.x0)}" y1="{_f(rp.y0)}" x2="{_f(rp.x0)}" y2="{_f(rp.y1)}" stroke="{AXE}" stroke-width="1.3"/>'
             f'<line x1="{_f(rp.x1)}" y1="{_f(rp.y0)}" x2="{_f(rp.x1)}" y2="{_f(rp.y1)}" stroke="{COL_TE}" stroke-width="1.3" stroke-opacity=".6"/>')
    if a_t < 0 < b_t:
        g.append(f'<line x1="{_f(rp.x0)}" y1="{_f(rp.Y(0))}" x2="{_f(rp.x1)}" y2="{_f(rp.Y(0))}" stroke="{AXE}" stroke-width="1.2"/>')
    # ---- le jour où chaque objectif est atteint : un trait fin et son nom, en haut (écartés pour ne pas se toucher)
    obstacles: List[Boite] = []
    for o in sorted((o for o in r["objectifs"] if o["t"] is not None and o["t"] <= jours[-1]["fin"]), key=lambda o: o["t"]):
        jr = next(j for j in jours if o["t"] <= j["fin"] + 1e-12)
        x = rp.X(jr["jour"]) - pas / 2 + pas * (o["t"] - jr["debut"]) / max(jr["fin"] - jr["debut"], 1e-9)
        g.append(f'<line x1="{_f(x)}" y1="{_f(rp.y0)}" x2="{_f(x)}" y2="{_f(rp.y1)}" stroke="{COL_OBJ}" stroke-width="1.1" stroke-dasharray="4 3"/>')
        txt = f"{_nom_court(o['nom'])} · J+{fr(o['t'], 1)}"
        wb, hb = _largeur(txt, T_BOITE, True) + 12, T_BOITE + 9
        places = [(bx, rp.y0 + 20 + k * (hb + 4)) for k in range(0, 10) for bx in (x + 5, x - 5 - wb)]
        bx, byy = next(((bx, by_) for bx, by_ in places if bx >= rp.x0 + 2 and bx + wb <= rp.x1 - 2 and _libre((bx, by_, bx + wb, by_ + hb), obstacles)),
                       (min(max(x + 5, rp.x0 + 2), rp.x1 - 2 - wb), rp.y0 + 20))
        b = (bx, byy, bx + wb, byy + hb)
        obstacles.append(b)
        g.append(_etiquette(b, txt, COL_OBJ))
    # ---- les titres des axes
    g.append(_txt((rp.x0 + rp.x1) / 2, hauteur - 9, "Jour depuis le bétonnage (0 : le jour du bétonnage)" + (" · dates" if date else ""), T_TITRE, "middle", ENCRE, True))
    yt = (rp.y0 + rp.y1) / 2
    g.append(_txt(15, yt, "Température du jour (°C)", T_TITRE, "middle", ENCRE, True, extra=f' transform="rotate(-90 15 {_f(yt)})"'))
    g.append(_txt(LARGEUR - 11, yt, _sub("t", "T", " cumulé (j)"), T_TITRE, "middle", COL_TE, True, extra=f' transform="rotate(90 {_f(LARGEUR - 11)} {_f(yt)})"'))
    return (f'<svg id="{escape(ident)}" class="age-graphe" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(LARGEUR)} {_f(hauteur)}" '
            f'width="100%" role="img" aria-label="Températures jour par jour et âge équivalent cumulé">{"".join(g)}</svg>')


__all__ = ["courbe", "temperatures", "graduations", "pas_jours", "poser_etiquette", "etiquette_objectif", "fr"]

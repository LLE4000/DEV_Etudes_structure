# -*- coding: utf-8 -*-
"""Les dessins du module « Taux d'armature » (docs/site/LOT_Q_QUATRE_MODULES.md § 3.6), à partir de ce que le moteur
``beton/taux`` a calculé — rien n'est recalculé ici : les abscisses, les niveaux, les nombres d'étriers, les longueurs
et les repères sont les siens.

- ``elevation`` : la poutre vue de côté. Les LONGUEURS sont à l'échelle ; les hauteurs sont amplifiées quand la poutre
  est élancée (``amplification`` : le facteur est écrit sur le dessin). Chaque ligne de barres court de x₁ à x₂ à son
  niveau (les retours relevés à leurs bouts, les bouts d'un renfort marqués), repérée par le numéro du bordereau ;
  chaque étrier de chaque zone est tracé — autant que le moteur en compte (trop serrés pour se distinguer, un sur m :
  le dessin le dit) ; sous la poutre, la chaîne des zones (« 10 Ø 8 · e = 15 », les abscisses en m, les portions sans
  étriers en rouge), la longueur, puis les barres DÉVELOPPÉES, une par ligne, avec leur longueur unitaire ; les
  repères des coupes (A, B…) au-dessus.
- ``coupe`` : une coupe par COMPOSITION de la poutre (``res["coupes"]`` : mêmes barres, même type d'étriers), à
  l'échelle, en mm : le béton (l'âme et ses talons), les cadres, les cadres intérieurs, les épingles et le cadre de
  talon à leur place, les barres, chaque acier repéré par son numéro du bordereau ; les cotes en cm.
- ``bordereau`` : le bordereau de la poutre (un bloc HTML : un tableau), un repère par ligne, la synthèse par
  diamètre, le total, la majoration, le total majoré.

Chaque dessin existe en DEUX variantes dans un même ``<svg class="tx-dessin">`` : ``.tx-large`` (PC, tablette) et
``.tx-etroit`` (une feuille étroite : le téléphone), cette dernière cachée par un style en ligne que la feuille du
module lève sous 520 px de feuille (css/modules/taux.css, une requête de conteneur). Le téléphone ne reçoit donc pas
une élévation de 6 m réduite à 120 px de haut : il a la sienne, plus haute, aux textes plus gros, les coupes l'une sous
l'autre. Les deux variantes portent les mêmes pièces.

Le CONTRAT DE SÉLECTION (js/compact.js) : chaque pièce est un groupe ``<g class="pc …" data-sel="…">`` — « beton »,
« b<i> » (la ligne de barres i), « z<i> » (la zone d'étriers i), « t<i> » (le type d'étriers i) — et les données
portent la même clé (``data-cible``, api_web/taux.py).

Le style est celui de la coupe des poutres (api_web/coupe_ba.py) : ses couleurs, le semis de granulats du béton coupé
et le double trait des cadres (``chemin_cadre``, ``chemin_epingle``) sont APPELÉS, pas recopiés. Tout texte venu de la
note (le nom d'un type d'étriers) est échappé.
"""
import math
from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

from . import commun as C
from . import coupe_ba as CB

ENCRE, BETON, BETON_BORD, ACIER, CADRE_BORD, COTE, DOUX = CB.ENCRE, CB.BETON, CB.BETON_BORD, CB.ACIER, CB.CADRE_BORD, CB.COTE, CB.DOUX
ALERTE = "#B42318"
POLICE = CB.POLICE

# ---- l'élévation : l'amplification des hauteurs
_AMPLIS = (1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50)
# 6 (et non 10) : une poutre courante de 6 m × 60 cm dessinée à l'échelle n'était qu'une bande de 65 px au PC, ses
# étriers à 11 px les uns des autres ; dessinée deux fois plus haute, barres et étriers se lisent (relecture du 07/11/2026)
ELANCEMENT = 6
# la feuille étroite (le téléphone) : la largeur manque, la hauteur non (la page défile) — la poutre y est dessinée plus
# haute encore, deux lits d'une face s'y distinguent (4 px les séparaient à l'élancement 6, mesuré le 10/11/2026)
ELANCEMENT_ETROIT = 3
# la taille du texte : la longueur de la poutre en vaut tant de fois (50 : 12,5 px sur une toile de 700 px ; 26 : 11 px
# sur les 345 px d'un téléphone)
DIVISEUR, DIVISEUR_ETROIT = 50.0, 26.0
# deux étriers à moins de 0,2 hauteur de texte l'un de l'autre ne se distinguent plus : on en trace un sur m, et on le dit
PAS_MIN = 0.2
# au-delà, les barres développées allongeraient trop le dessin : le bordereau les porte
MAX_DEVELOPPEES = 10
# où poser le repère d'une ligne de barres le long d'elle (en fraction de sa longueur), dans l'ordre des essais
FRACTIONS = (0.5, 0.32, 0.68, 0.18, 0.82, 0.08, 0.92)
LARGEUR_CAR = 0.58          # la largeur d'un caractère, en hauteurs de texte (chiffres tabulaires, capitales comprises)


def amplification(L_m: float, h_cm: float, elancement: float = ELANCEMENT) -> float:
    """Le facteur des hauteurs de l'élévation : 1 tant que la poutre n'est pas plus de ``elancement`` fois plus longue
    que haute ; au-delà, la plus petite valeur ronde qui ramène l'élancement dessiné à ``elancement`` (écrit sur le
    dessin)."""
    r = L_m * 1000 / (h_cm * 10)
    if r <= elancement:
        return 1.0
    return next((k for k in _AMPLIS if r / k <= elancement), _AMPLIS[-1])


# ============================================================ les petites briques
def _f(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _esc(t) -> str:
    return escape(str(t), quote=True)


def _m(x: float) -> str:
    """Une abscisse ou une longueur en mètres : « 1,50 » — au millimètre quand il le faut (« 1,275 »)."""
    return C.fr(x, 2) if abs(x * 100 - round(x * 100)) < 1e-6 else C.fr(x, 3)


def _cm(x: float) -> str:
    """Une cote en cm, écrite telle quelle (30 ; 12,5)."""
    return C.nb(round(x, 2), 2)


def _large(texte: str, taille: float) -> float:
    """La largeur d'un texte du dessin, estimée par excès (le rangement des étiquettes s'y fie)."""
    return LARGEUR_CAR * taille * len(texte)


class _Etendue:
    """L'emprise de ce qui est dessiné : le cadre de vue (viewBox) en sort, rien n'est rogné."""

    def __init__(self):
        self.x0 = self.y0 = math.inf
        self.x1 = self.y1 = -math.inf

    def boite(self, x0, y0, x1, y1) -> None:
        self.x0, self.y0 = min(self.x0, x0, x1), min(self.y0, y0, y1)
        self.x1, self.y1 = max(self.x1, x0, x1), max(self.y1, y0, y1)

    def vue(self, marge: float) -> List[float]:
        return [self.x0 - marge, self.y0 - marge, self.x1 - self.x0 + 2 * marge, self.y1 - self.y0 + 2 * marge]


def _txt(ext: Optional[_Etendue], x, y, t, taille, ancre="start", couleur=ENCRE, poids=500, classe="", extra="") -> str:
    """Un texte (``t`` : déjà échappé) ; son emprise estimée entre dans ``ext``."""
    if ext is not None:
        w = _large(t, taille)
        gx = x if ancre == "start" else (x - w if ancre == "end" else x - w / 2)
        ext.boite(gx, y - taille, gx + w, y + 0.3 * taille)
    cl = f' class="{classe}"' if classe else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" text-anchor="{ancre}" font-size="{_f(taille)}" fill="{couleur}" '
            f'font-weight="{poids}"{cl}{extra}>{t}</text>')


def _ligne(x1, y1, x2, y2, couleur, ep, extra="") -> str:
    return (f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{couleur}" stroke-width="{_f(ep)}"'
            f' stroke-linecap="round"{extra}/>')


def _bulle(ext: Optional[_Etendue], x, y, numero, taille) -> str:
    """Le REPÈRE d'un acier : son numéro du bordereau dans un cercle (rayon 0,68 hauteur de texte)."""
    r = 0.68 * taille
    if ext is not None:
        ext.boite(x - r, y - r, x + r, y + r)
    t = taille * (0.86 if numero < 10 else 0.7)
    return (f'<circle class="bul" cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="#fff" stroke="{ENCRE}" stroke-width="{_f(0.075 * taille)}"/>'
            f'<text class="bul" x="{_f(x)}" y="{_f(y + 0.31 * t)}" text-anchor="middle" font-size="{_f(t)}" fill="{ENCRE}" font-weight="700">{numero}</text>')


def _cote_h(ext: _Etendue, x1, x2, y, texte, taille, attache=None) -> str:
    """Une cote horizontale : la ligne, deux traits obliques, le nombre au-dessus."""
    t, ep = 0.3 * taille, 0.065 * taille
    out = ""
    if attache is not None:
        for x in (x1, x2):
            out += _ligne(x, attache, x, y + 1.4 * t, COTE, 0.7 * ep)
    out += (_ligne(x1 - 1.2 * t, y, x2 + 1.2 * t, y, COTE, ep) + _ligne(x1 - t, y + t, x1 + t, y - t, ENCRE, 1.15 * ep)
            + _ligne(x2 - t, y + t, x2 + t, y - t, ENCRE, 1.15 * ep) + _txt(ext, (x1 + x2) / 2, y - 0.42 * taille, texte, taille, "middle", DOUX, 500))
    ext.boite(x1 - 1.2 * t, y - t, x2 + 1.2 * t, y + 1.4 * t)
    return out


def _cote_v(ext: _Etendue, x, y1, y2, texte, taille, attache=None) -> str:
    """Une cote verticale : le nombre tourné le long de la ligne, à sa gauche."""
    t, ep = 0.3 * taille, 0.065 * taille
    out = ""
    if attache is not None:
        for y in (y1, y2):
            out += _ligne(attache, y, x + (1.4 * t if x > attache else -1.4 * t), y, COTE, 0.7 * ep)
    xt, ym = x - 0.42 * taille, (y1 + y2) / 2
    w = _large(texte, taille)
    ext.boite(xt - taille, min(y1 - 1.2 * t, ym - w / 2), x + t, max(y2 + 1.2 * t, ym + w / 2))
    out += (_ligne(x, y1 - 1.2 * t, x, y2 + 1.2 * t, COTE, ep) + _ligne(x - t, y1 + t, x + t, y1 - t, ENCRE, 1.15 * ep)
            + _ligne(x - t, y2 + t, x + t, y2 - t, ENCRE, 1.15 * ep)
            + _txt(None, xt, ym, texte, taille, "middle", DOUX, 500, extra=f' transform="rotate(-90 {_f(xt)} {_f(ym)})"'))
    return out


# ============================================================ le rangement des repères d'un côté de la poutre
class _Rangees:
    """Les repères d'un côté de la poutre, rangés en RANGÉES (la première contre la poutre) : chaque repère est une
    boîte (la bulle et son texte) et un trait de rappel vertical qui descend de la bulle à sa barre. Une boîte ne
    recouvre aucune autre ; un rappel ne traverse aucune boîte ni aucune colonne interdite (le repère d'une coupe), et
    deux rappels ne se confondent pas."""

    def __init__(self, x_max: float, taille: float, interdits: Sequence[Tuple[float, float]] = ()):
        self.x_max, self.t, self.interdits = x_max, taille, list(interdits)
        self.boites: List[Tuple[int, float, float]] = []
        self.rappels: List[Tuple[int, float]] = []

    def _libre(self, rang: int, x0: float, x1: float, xr: float) -> bool:
        if x0 < -1.2 * self.t or x1 > self.x_max + 1.2 * self.t:
            return False
        if any(a <= xr <= b for a, b in self.interdits):
            return False
        for r, a, b in self.boites:
            if r == rang and x0 < b and a < x1:
                return False
            if r < rang and a <= xr <= b:
                return False
        for r, x in self.rappels:
            if abs(x - xr) < 0.9 * self.t or (r > rang and x0 <= x <= x1):
                return False
        return True

    def placer(self, x1: float, x2: float, largeur: float) -> Tuple[int, float, int]:
        """La place du repère d'une barre qui court de ``x1`` à ``x2`` : (rangée, abscisse de la bulle, sens du texte :
        +1 à droite de la bulle, −1 à sa gauche)."""
        rb, jeu = 0.68 * self.t, 0.35 * self.t
        for rang in range(1, 9):
            for sens in (1, -1):
                for f in FRACTIONS:
                    xr = x1 + f * (x2 - x1)
                    a, b = (xr - rb - jeu, xr - rb + largeur + jeu) if sens > 0 else (xr + rb - largeur - jeu, xr + rb + jeu)
                    if self._libre(rang, a, b, xr):
                        self.boites.append((rang, a, b))
                        self.rappels.append((rang, xr))
                        return rang, xr, sens
        rang = max((r for r, _, _ in self.boites), default=0) + 1
        xr = (x1 + x2) / 2
        self.boites.append((rang, xr - rb - jeu, xr - rb + largeur + jeu))
        self.rappels.append((rang, xr))
        return rang, xr, 1

    @property
    def n(self) -> int:
        return max((r for r, _, _ in self.boites), default=0)


# ============================================================ l'élévation
def positions_etriers(res: Dict) -> List[List[float]]:
    """Où se trace chaque étrier de chaque zone (m, depuis l'about gauche) : la pose que le moteur compte — les cours
    d'une zone à pas égaux (l_z / n_c ≤ s) depuis son début, le cours de fermeture au bout de la suite —, les étriers
    d'about ramenés à l'enrobage. Autant d'abscisses par zone que le moteur compte d'étriers (``zones[i]["n"]``)."""
    p = res["poutre"]
    c, L = p["enrobage"] / 100.0, p["L"]
    out = []
    for z in res["zones"]:
        pas = z["longueur"] / z["cours"]
        xs = [z["de"] + j * pas for j in range(z["cours"])] + ([z["a"]] if z["fermeture"] else [])
        out.append([min(max(x, c), L - c) for x in xs])
    return out


def _texte_zone(z: Dict, type_: Optional[Dict]) -> Tuple[str, str]:
    """Ce que la chaîne dit d'une zone : (« 10 Ø 8 », « e = 15 ») — le Ø quand le type n'a qu'une ligne, son nom sinon."""
    seul = type_ is not None and len(type_["lignes"]) == 1 and type_["talon"] is None
    quoi = f"{z['n']} Ø {type_['lignes'][0]['d']}" if seul else f"{z['n']} × {_esc(z['type'])}"
    return quoi, f"e = {_cm(z['s'])}"


def _elevation(res: Dict, diviseur: float, elancement: float) -> Tuple[str, List[float]]:
    """Le corps de l'élévation et son cadre de vue ; ``diviseur`` : la longueur de la poutre en hauteurs de texte."""
    p, sec = res["poutre"], res["section"]
    L, h, c = p["L"], p["h"], p["enrobage"]
    Lmm = L * 1000.0
    fs = Lmm / diviseur
    k = amplification(L, h, elancement)
    H = h * 10.0 * k
    ext = _Etendue()
    ext.boite(0, 0, Lmm, H)

    def Y(y_cm: float) -> float:
        return (h - y_cm) * 10.0 * k

    # ---- le béton : la poutre de côté, le dessus des talons, la cote h
    beton = (f'<rect x="0" y="0" width="{_f(Lmm)}" height="{_f(H)}" fill="{BETON}"/>'
             f'<rect class="bord" x="0" y="0" width="{_f(Lmm)}" height="{_f(H)}" fill="none" stroke="{BETON_BORD}" stroke-width="{_f(0.1 * fs)}"/>')
    if sec["h_talon"]:
        beton += _ligne(0, Y(sec["h_talon"]), Lmm, Y(sec["h_talon"]), COTE, 0.06 * fs, f' stroke-dasharray="{_f(0.5 * fs)} {_f(0.3 * fs)}"')
    beton += _cote_v(ext, -1.5 * fs, 0, H, f"h = {_cm(h)}", 0.9 * fs, attache=-0.25 * fs)
    morceaux = [f'<g class="pc beton" data-sel="beton">{beton}</g>']

    # ---- les étriers, zone par zone (sous les barres)
    types = {t["nom"]: t for t in res["types"]}
    y_haut, y_bas = Y(h - c), Y(c)
    eclaircis = []
    for z, xs in zip(res["zones"], positions_etriers(res)):
        pas = z["longueur"] * 1000.0 / z["cours"]
        m = 1 if pas >= PAS_MIN * fs else int(math.ceil(PAS_MIN * fs / pas))
        if m > 1:
            eclaircis.append((z["nom"], m))
        cours = [x for j, x in enumerate(xs[:z["cours"]]) if j % m == 0] + xs[z["cours"]:]
        traits = "".join(f'<line class="et" x1="{_f(x * 1000)}" y1="{_f(y_haut)}" x2="{_f(x * 1000)}" y2="{_f(y_bas)}" stroke="{CADRE_BORD}" '
                         f'stroke-width="{_f(0.075 * fs)}"/>' for x in cours)
        morceaux.append(f'<g class="pc zone" data-sel="z{z["index"]}" data-etriers="{z["n"]}" data-traces="{len(cours)}">'
                        f'<rect class="zn" x="{_f(z["de"] * 1000)}" y="0" width="{_f(z["longueur"] * 1000)}" height="{_f(H)}" fill="#2B4BCF" fill-opacity="0"/>'
                        f'{traits}</g>')

    for x1, x2 in res["trous"]:
        morceaux.append(f'<rect class="trou" x="{_f(x1 * 1000)}" y="0" width="{_f((x2 - x1) * 1000)}" height="{_f(H)}" fill="{ALERTE}" fill-opacity=".09"/>')

    # ---- les repères des coupes : un trait fort de part et d'autre de la poutre, la lettre en haut
    coupes = res["coupes"] if len(res["coupes"]) > 1 else []
    interdits = [(q["x"] * 1000 - 0.5 * fs, q["x"] * 1000 + 1.5 * fs) for q in coupes]
    for q in coupes:
        x = q["x"] * 1000
        morceaux.append('<g class="coupe-rep">' + _ligne(x, -1.05 * fs, x, -0.2 * fs, ENCRE, 0.15 * fs) + _ligne(x, H + 0.2 * fs, x, H + 1.05 * fs, ENCRE, 0.15 * fs)
                        + _txt(ext, x + 0.35 * fs, -0.3 * fs, q["cle"], fs, "start", ENCRE, 700) + "</g>")
    marge_haut = marge_bas = (1.35 if coupes else 0.45) * fs

    # ---- les barres, et leurs repères rangés au-dessus (supérieures, peau) ou au-dessous (inférieures)
    niveaux = res["elevation"]["niveaux"]
    rangees = {"haut": _Rangees(Lmm, fs, interdits), "bas": _Rangees(Lmm, fs, interdits)}
    pas_rang = 1.75 * fs
    barres = []
    poses: List[Tuple[float, float, float]] = []          # (niveau, x1, x2) des lignes déjà tracées
    for r in res["reperes"]:
        if r["sorte"] != "barre":
            continue
        d = r["detail"]
        i = d["index"]
        x1, x2 = d["x1"] * 1000.0, d["x2"] * 1000.0
        ep = min(max(r["d"] * k, 0.14 * fs), 0.36 * fs)
        peau = d["famille"] == "peau"
        ys = [Y(v) for v in niveaux[i]]
        if not peau:
            # deux lignes d'un même lit se cacheraient l'une l'autre (elles sont côte à côte dans la section) : la
            # suivante est tracée une épaisseur plus à l'intérieur — une convention de dessin, pas une position
            deja = sum(1 for yv, a, b in poses if abs(yv - ys[0]) < 1e-6 and a < x2 and x1 < b)
            poses.append((ys[0], x1, x2))
            ys = [ys[0] + deja * 1.35 * ep * (-1 if d["famille"] == "inf" else 1)]
        g = "".join(f'<line class="zn" x1="{_f(x1)}" y1="{_f(y)}" x2="{_f(x2)}" y2="{_f(y)}" stroke="#000" stroke-opacity="0" stroke-width="{_f(max(ep, 0.9 * fs))}"/>'
                    f'<line class="ba" x1="{_f(x1)}" y1="{_f(y)}" x2="{_f(x2)}" y2="{_f(y)}" stroke="{ACIER}" stroke-width="{_f(ep)}"/>' for y in ys)
        # les bouts : un retour relevé (un seul : au bout le plus près d'un about), sinon le trait d'arrêt d'un renfort
        bouts = {x1: d["de"] is not None, x2: d["a"] is not None}
        releves = []
        if d["retours"] == 2:
            releves = [x1, x2]
        elif d["retours"] == 1:
            releves = [x1 if x1 <= Lmm - x2 else x2]
        for x, arrete in bouts.items():
            for y in ys:
                if x in releves:
                    if peau:
                        g += f'<circle class="ba" cx="{_f(x)}" cy="{_f(y)}" r="{_f(0.75 * ep)}" fill="{ACIER}"/>'
                    else:
                        lr = min(d["l_retour"], h - 2.0 * c) * 10.0 * k
                        g += _ligne(x, y, x, y - lr if d["famille"] == "inf" else y + lr, ACIER, ep, ' class="ba"')
                elif arrete:
                    g += _ligne(x, y - 0.32 * fs, x, y + 0.32 * fs, ACIER, 0.09 * fs, ' class="ba"')
        cote = "bas" if d["famille"] == "inf" else "haut"
        largeur = 2 * 0.68 * fs + 0.3 * fs + _large(r["libelle"], fs)
        rang, xr, sens = rangees[cote].placer(x1, x2, largeur)
        barres.append((r, g, cote, rang, xr, sens, ys))
    for r, g, cote, rang, xr, sens, ys in barres:
        if cote == "haut":
            yc = -marge_haut - 0.85 * fs - (rang - 1) * pas_rang
            y_barre = min(ys)
            y_bulle = yc + 0.68 * fs
        else:
            yc = H + marge_bas + 0.85 * fs + (rang - 1) * pas_rang
            y_barre = max(ys)
            y_bulle = yc - 0.68 * fs
        g += (_ligne(xr, y_bulle, xr, y_barre, COTE, 0.06 * fs, ' class="rp"')
              + f'<circle class="rp" cx="{_f(xr)}" cy="{_f(y_barre)}" r="{_f(0.17 * fs)}" fill="{COTE}"/>'
              + _bulle(ext, xr, yc, r["repere"], fs)
              + _txt(ext, xr + sens * 0.98 * fs, yc + 0.35 * fs, _esc(r["libelle"]), fs, "start" if sens > 0 else "end", ENCRE, 600, "an"))
        morceaux.append(f'<g class="pc barre" data-sel="{r["cle"]}">{g}</g>')

    # ---- la chaîne des zones d'étriers : ce que chaque zone porte, ses abscisses ; les portions sans étriers
    y = H + marge_bas + rangees["bas"].n * pas_rang + 0.5 * fs
    tz, ta = 0.92 * fs, 0.82 * fs
    textes = []
    for z in res["zones"]:
        quoi, pas = _texte_zone(z, types.get(z["type"]))
        place = z["longueur"] * 1000.0 - 0.5 * fs
        if _large(f"{quoi} · {pas}", tz) <= place:
            textes.append((z, [f"{quoi} · {pas}"]))
        elif max(_large(quoi, tz), _large(pas, tz)) <= place:
            textes.append((z, [quoi, pas]))
        else:
            textes.append((z, [z["nom"]] if _large(z["nom"], tz) <= place else []))
    n_lignes = max([len(t) for _, t in textes] + [1 if res["trous"] else 0], default=0)
    y_chaine = y + n_lignes * 1.2 * tz + 0.55 * fs
    if res["zones"] or res["trous"]:
        chaine = _ligne(0, y_chaine, Lmm, y_chaine, COTE, 0.06 * fs)
        bornes = sorted({0.0, L} | {x for z in res["zones"] for x in (z["de"], z["a"])})
        t = 0.3 * fs
        ecrites = []
        for x in [bornes[0], bornes[-1]] + bornes[1:-1]:           # les deux abouts d'abord : ils s'écrivent toujours
            X = x * 1000
            chaine += _ligne(X - t, y_chaine + t, X + t, y_chaine - t, ENCRE, 0.075 * fs)
            w = _large(_m(x), ta) / 2 + 0.2 * fs
            if all(abs(X - xe) >= w + we for xe, we in ecrites):
                ecrites.append((X, w))
                chaine += _txt(ext, X, y_chaine + 1.25 * ta, _m(x), ta, "middle", DOUX, 500)
        morceaux.append(f'<g class="chaine">{chaine}</g>')
        for z, lignes in textes:
            xm = (z["de"] + z["a"]) / 2 * 1000
            tx = "".join(_txt(ext, xm, y_chaine - 0.5 * fs - (len(lignes) - 1 - j) * 1.2 * tz, li, tz, "middle", ENCRE if j == 0 else DOUX, 600 if j == 0 else 500, "an")
                         for j, li in enumerate(lignes))
            morceaux.append(f'<g class="pc zone" data-sel="z{z["index"]}">'
                            f'<rect class="zn" x="{_f(z["de"] * 1000)}" y="{_f(y)}" width="{_f(z["longueur"] * 1000)}" height="{_f(y_chaine - y)}" fill="#2B4BCF" fill-opacity="0"/>{tx}</g>')
        for x1, x2 in res["trous"]:
            X1, X2 = x1 * 1000, x2 * 1000
            mot = next((m_ for m_ in ("sans étriers", "sans") if _large(m_, tz) <= X2 - X1 - 0.4 * fs), "")
            morceaux.append('<g class="trou">' + _ligne(X1, y_chaine, X2, y_chaine, ALERTE, 0.12 * fs, f' stroke-dasharray="{_f(0.4 * fs)} {_f(0.3 * fs)}"')
                            + (_txt(ext, (X1 + X2) / 2, y_chaine - 0.5 * fs, mot, tz, "middle", ALERTE, 600) if mot else "") + "</g>")
        y = y_chaine + 1.25 * ta + 0.6 * fs
    # ---- la longueur ; ce que le dessin amplifie ou éclaircit
    y += 1.5 * fs
    morceaux.append(f'<g class="pc beton" data-sel="beton">{_cote_h(ext, 0, Lmm, y, f"L = {_m(L)} m", 0.95 * fs)}</g>')
    notes = [f"hauteurs × {C.nb(k)}"] if k > 1 else []
    for m in sorted({m for _, m in eclaircis}):
        noms = [nom for nom, m_ in eclaircis if m_ == m]
        notes.append(f"1 étrier sur {m} tracé ({', '.join(noms[:4])}{'…' if len(noms) > 4 else ''})")
    if notes:
        y += 1.35 * fs
        morceaux.append(_txt(ext, 0, y, " · ".join(notes), 0.8 * fs, "start", DOUX, 500, "note"))
    # ---- les barres développées : une par ligne, à sa place en long, sa longueur unitaire
    lignes_barres = [r for r in res["reperes"] if r["sorte"] == "barre"]
    if lignes_barres and len(lignes_barres) <= MAX_DEVELOPPEES:
        y += 0.9 * fs
        for r in lignes_barres:
            d = r["detail"]
            x1, x2 = d["x1"] * 1000.0, d["x2"] * 1000.0
            texte = f"{r['libelle']} · l = {_m(r['l_unitaire'])} m"
            if d["n_rec"]:
                texte += f" · {d['n_rec']} recouvrement{'s' if d['n_rec'] > 1 else ''}"
            w = 2 * 0.68 * fs + 0.3 * fs + _large(texte, 0.92 * fs)
            xl = max(0.0, min(x1, Lmm - w))
            y += 1.25 * fs
            g = _bulle(ext, xl + 0.68 * fs, y, r["repere"], 0.92 * fs) + _txt(ext, xl + 1.66 * fs, y + 0.33 * fs, _esc(texte), 0.92 * fs, "start", ENCRE, 500, "an")
            y += 1.3 * fs
            ep = 0.2 * fs
            g += (f'<line class="zn" x1="{_f(x1)}" y1="{_f(y)}" x2="{_f(x2)}" y2="{_f(y)}" stroke="#000" stroke-opacity="0" stroke-width="{_f(1.2 * fs)}"/>'
                  + _ligne(x1, y, x2, y, ACIER, ep, ' class="ba"'))
            releves = [x1, x2] if d["retours"] == 2 else ([x1 if x1 <= Lmm - x2 else x2] if d["retours"] == 1 else [])
            for x in releves:
                g += _ligne(x, y, x, y - 0.75 * fs, ACIER, ep, ' class="ba"')
            ext.boite(x1, y - 0.8 * fs, x2, y + 0.2 * fs)
            y += 0.35 * fs
            morceaux.append(f'<g class="pc barre" data-sel="{r["cle"]}">{g}</g>')
    elif lignes_barres:
        y += 1.6 * fs
        morceaux.append(_txt(ext, 0, y, f"{len(lignes_barres)} lignes de barres : leurs longueurs sont au bordereau", 0.85 * fs, "start", DOUX, 500, "note"))
    return "".join(morceaux), ext.vue(0.5 * fs)


# ============================================================ la coupe
LIBELLES_ACIER = {"cadre": "cadre", "cadre_interieur": "cadre int.", "epingle": "épingle", "talon": "cadre de talon"}


def _plages(plages: Sequence[Sequence[float]]) -> str:
    """Où vaut une coupe : « de 0,00 à 1,20 m et de 4,80 à 6,00 m » ; au-delà de deux tronçons, leur nombre."""
    if len(plages) > 2:
        return f"{len(plages)} tronçons · {_m(sum(b - a for a, b in plages))} m en tout"
    return " et ".join(f"de {_m(a)} à {_m(b)} m" for a, b in plages)


def _une_coupe(res: Dict, q: Dict, motif: str, grossi: float = 1.0) -> Dict:
    """UNE coupe (mm, l'origine au coin haut gauche du rectangle qui englobe la section) : {"svg", "x0", "x1", "y0",
    "y1" (son emprise), "u"}. ``grossi`` : le facteur des textes et des cotes (les aciers restent à l'échelle) — trois
    ou quatre coupes côte à côte se partagent la toile, leurs textes doivent rester lisibles."""
    geo = q["geometrie"]
    sec = geo["section"]
    b_tot, h, c = sec["b_tot"] * 10.0, sec["h"] * 10.0, sec["c"] * 10.0
    u0 = max(b_tot, h, 250.0) / 400.0            # l'unité des traits d'acier (celle de la coupe des poutres)
    u = u0 * grossi                              # … et celle des textes et des cotes
    F, f = 14.0 * u, 11.5 * u
    ext = _Etendue()
    ext.boite(0, 0, b_tot, h)

    def X(x_cm):
        return x_cm * 10.0

    def Y(y_cm):
        return h - y_cm * 10.0

    out = []
    # ---- le béton et ses cotes (cm)
    pts = " ".join(f"{_f(X(x))},{_f(Y(y))}" for x, y in sec["polygone"])
    beton = (f'<polygon points="{pts}" fill="{BETON}"/><polygon points="{pts}" fill="url(#{motif})"/>'
             f'<polygon class="bord" points="{pts}" fill="none" stroke="{BETON_BORD}" stroke-width="{_f(1.6 * u0)}" stroke-linejoin="miter"/>')
    xg, xd = X(sec["bg"]), X(sec["bg"] + sec["b"])
    yb = h + 2.3 * F
    cotes = _cote_h(ext, xg, xd, yb, f"b = {_cm(sec['b'])}", f, attache=h + 3 * u)
    if sec["bg"]:
        cotes += _cote_h(ext, 0, xg, yb, _cm(sec["bg"]), f, attache=h + 3 * u)
        cotes += _cote_v(ext, -1.25 * F, Y(sec["hg"]), h, _cm(sec["hg"]), f, attache=-3 * u)
    if sec["bd"]:
        cotes += _cote_h(ext, xd, b_tot, yb, _cm(sec["bd"]), f, attache=h + 3 * u)
        cotes += _cote_v(ext, b_tot + 1.9 * F, Y(sec["hd"]), h, _cm(sec["hd"]), f, attache=b_tot + 3 * u)
    cotes += _cote_v(ext, -(3.0 * F if sec["bg"] else 1.6 * F), 0, h, f"h = {_cm(sec['h'])}", f, attache=-3 * u)
    cotes += _txt(ext, xg, -0.75 * F, f"c = {_cm(sec['c'])}", 0.92 * f, "start", DOUX, 450)
    out.append(f'<g class="pc beton" data-sel="beton">{beton}{cotes}</g>')

    # ---- les aciers transversaux du type : à leur place, sous les barres
    haut = [bb for bb in geo["barres"] if bb["famille"] == "sup"]
    bas = [bb for bb in geo["barres"] if bb["famille"] == "inf"]
    r_haut = max((bb["d"] for bb in haut if bb["lit"] == min(x["lit"] for x in haut)), default=0) / 2.0
    r_bas = max((bb["d"] for bb in bas if bb["lit"] == min(x["lit"] for x in bas)), default=0) / 2.0
    annot = []                                    # (ordonnée visée, abscisse d'attache, clé, repère, texte)
    type_ = next((t for t in res["types"] if t["nom"] == q["type"]), None)
    if type_ is not None:
        ti = type_["index"]
        reperes = {r["cle"]: r for r in res["reperes"] if r["sorte"] == "etrier"}
        cadres = [cd for cd in geo["cadres"] if cd["sorte"] == "cadre"]
        interieurs = [cd for cd in geo["cadres"] if cd["sorte"] == "cadre_interieur"]
        talon = next((cd for cd in geo["cadres"] if cd["sorte"] == "talon"), None)
        pinces = list(geo["epingles"])
        aciers = ""
        if talon is not None:
            de = float(talon["d"])
            aciers += CB._trace_acier(CB.chemin_cadre(X(talon["x0"]), X(talon["x1"]), Y(talon["y1"]), Y(talon["y0"]), de, 0.0, r_bas), de, u0)
            if f"t{ti}.talon" in reperes:
                annot.append(((Y(talon["y0"]) + Y(talon["y1"])) / 2, X(talon["x1"]), f"t{ti}", reperes[f"t{ti}.talon"]["repere"],
                              f"{LIBELLES_ACIER['talon']} Ø {talon['d']:g}"))
        k_cadre = k_int = k_ep = 0
        for j, li in enumerate(type_["lignes"]):
            de = float(li["d"])
            if li["forme"] == "epingle":
                ep = pinces[k_ep]
                k_ep += 1
                yh = Y(max((y for bb in haut for _, y in bb["points"]), default=ep["y1"] - (de + 6) / 10.0))
                yl = Y(min((y for bb in bas for _, y in bb["points"]), default=ep["y0"] + (de + 6) / 10.0))
                rayon = max(r_haut, r_bas, 4.0)
                aciers += CB._trace_acier(CB.chemin_epingle(X(ep["x"]), yh, yl, rayon, de), de, u0)
                attache, cible = X(ep["x"]) + rayon + de, h * (0.5 + 0.07 * k_ep)
            else:
                cd = (cadres[k_cadre] if li["forme"] == "cadre" else interieurs[k_int])
                if li["forme"] == "cadre":
                    k_cadre += 1
                else:
                    k_int += 1
                aciers += CB._trace_acier(CB.chemin_cadre(X(cd["x0"]), X(cd["x1"]), Y(cd["y1"]), Y(cd["y0"]), de, r_haut, r_bas), de, u0)
                attache, cible = X(cd["x1"]), h * ((0.3 + 0.07 * (k_cadre - 1)) if li["forme"] == "cadre" else (0.4 + 0.07 * k_int))
            if f"t{ti}.{j}" in reperes:
                annot.append((cible, attache, f"t{ti}", reperes[f"t{ti}.{j}"]["repere"], f"{LIBELLES_ACIER[li['forme']]} Ø {li['d']}"))
        out.append(f'<g class="pc cadre" data-sel="t{ti}">{aciers}</g>')

    # ---- les barres
    par_index = {r["detail"]["index"]: r for r in res["reperes"] if r["sorte"] == "barre"}
    for bb in geo["barres"]:
        r = bb["d"] / 2.0
        cercles = "".join(f'<circle cx="{_f(X(x))}" cy="{_f(Y(y))}" r="{_f(r)}" fill="{ACIER}"/>' for x, y in bb["points"])
        out.append(f'<g class="pc barre" data-sel="b{bb["index"]}">{cercles}</g>')
        x_max = max(x for x, _ in bb["points"])
        y_vise = max(y for x, y in bb["points"] if x == x_max) if bb["famille"] != "inf" else min(y for x, y in bb["points"] if x == x_max)
        rep = par_index[bb["index"]]
        annot.append((Y(y_vise), X(x_max) + r, f"b{bb['index']}", rep["repere"], rep["libelle"]))

    # ---- les repères, à droite : rangés de haut en bas, jamais à moins d'une ligne et demie l'un de l'autre
    droite = b_tot + (3.1 * F if sec["bd"] else 0.6 * F)
    x_bulle = droite + 1.9 * F
    annot.sort(key=lambda a: a[0])
    ys, ecart = [], 1.55 * F
    for cible, *_ in annot:
        ys.append(cible if not ys else max(cible, ys[-1] + ecart))
    if ys:
        glisse = sum(y - a[0] for y, a in zip(ys, annot)) / len(ys)
        ys = [y - glisse for y in ys]
    for (cible, xa, cle, numero, texte), yt in zip(annot, ys):
        coude = droite + 0.5 * F
        g = (f'<polyline class="rp" points="{_f(xa + 1.2 * u)},{_f(cible)} {_f(coude)},{_f(cible)} {_f(x_bulle - 0.68 * F)},{_f(yt)}" fill="none" stroke="{COTE}" '
             f'stroke-width="{_f(0.65 * u)}"/><circle class="rp" cx="{_f(xa + 1.2 * u)}" cy="{_f(cible)}" r="{_f(1.3 * u)}" fill="{COTE}"/>'
             + _bulle(ext, x_bulle, yt, numero, F) + _txt(ext, x_bulle + 0.98 * F, yt + 0.35 * F, _esc(texte), F, "start", ENCRE, 600, "an"))
        out.append(f'<g class="pc {"barre" if cle.startswith("b") else "cadre"}" data-sel="{cle}">{g}</g>')

    # ---- le titre de la coupe, où elle vaut, ses étriers
    titre = (f'<text x="{_f(xg)}" y="{_f(-2.3 * F)}" font-size="{_f(1.1 * F)}" fill="{ENCRE}" font-weight="700">{q["cle"]}–{q["cle"]}'
             f'<tspan font-size="{_f(f)}" fill="{DOUX}" font-weight="500" dx="{_f(0.5 * F)}">x = {_m(q["x"])} m</tspan></text>')
    ext.boite(xg, -3.4 * F, xg + 11 * F, -2.0 * F)
    y_l = yb + 1.9 * f
    etr = "sans étriers" if type_ is None else (f"étriers {_esc(type_['nom'])} · e = " + " et ".join(_cm(s) for s in q["pas"]) + " cm")
    legende = (_txt(ext, 0, y_l, _plages(q["plages"]), f, "start", DOUX, 500)
               + _txt(ext, 0, y_l + 1.4 * f, etr, f, "start", ALERTE if type_ is None else DOUX, 600 if type_ is None else 500))
    out.append(f'<g class="titre">{titre}{legende}</g>')
    return {"svg": f'<g class="coupe">{"".join(out)}</g>', "x0": ext.x0 - 0.4 * F, "x1": ext.x1 + 0.4 * F, "y0": ext.y0 - 0.3 * F, "y1": ext.y1 + 0.5 * F,
            "u": u, "u0": u0}


# la toile où se rangent les coupes côte à côte : sa largeur vaut tant de fois sa hauteur (le PC, la tablette en paysage)
TOILE = 1.5
GROSSI = 1.5        # trois ou quatre coupes : leurs textes grossis d'autant (les sections y perdent, les repères se lisent)


def _coupes(res: Dict, motif: str, en_colonne: bool) -> Tuple[str, List[float]]:
    """Les coupes de la poutre, à la même échelle : l'une sous l'autre (feuille étroite), ou côte à côte — sur une
    rangée ou sur deux, selon ce qui les montre le plus grandes sur une toile de proportion ``TOILE``."""
    n = len(res["coupes"])
    cps = [_une_coupe(res, q, motif, 1.0 if (en_colonne or n <= 2) else GROSSI) for q in res["coupes"]]
    u = max(cp["u"] for cp in cps)
    lc, hc = max(cp["x1"] - cp["x0"] for cp in cps), max(cp["y1"] - cp["y0"] for cp in cps)
    jx, jy = 24.0 * u, 16.0 * u

    def emprise(colonnes: int) -> Tuple[float, float]:
        rangs = -(-n // colonnes)
        return colonnes * lc + (colonnes - 1) * jx, rangs * hc + (rangs - 1) * jy

    if en_colonne:
        colonnes = 1
    else:
        # la disposition qui donne la plus grande échelle sur la toile : min(TOILE / largeur, 1 / hauteur)
        colonnes = max((k for k in (n, 2) if k <= n), key=lambda k: (min(TOILE / emprise(k)[0], 1.0 / emprise(k)[1]), k))
    largeur, hauteur = emprise(colonnes)
    morceaux = []
    for j, cp in enumerate(cps):
        x, y = (j % colonnes) * (lc + jx), (j // colonnes) * (hc + jy)
        morceaux.append(f'<g transform="translate({_f(x - cp["x0"])} {_f(y - cp["y0"])})">{cp["svg"]}</g>')
    if res["coupes_omises"]:
        k = res["coupes_omises"]
        hauteur += 22.0 * u
        s_ = "s" if k > 1 else ""
        morceaux.append(_txt(None, 0, hauteur - 6.0 * u, f"+ {k} autre{s_} composition{s_} plus courte{s_}, non dessinée{s_}", 11.5 * u, "start", DOUX, 500, "note"))
    return f'<defs>{CB._motif(motif, max(cp["u0"] for cp in cps))}</defs>{"".join(morceaux)}', [0.0, 0.0, largeur, hauteur]


# ============================================================ les deux variantes d'un dessin, dans un seul <svg>
def _deux(ident: str, titre: str, large: Tuple[str, List[float]], etroit: Tuple[str, List[float]]) -> str:
    """Le dessin et ses deux variantes : ``.tx-large`` et ``.tx-etroit`` (cachée ici, montrée par la feuille du module
    sur une feuille étroite). ``--tx-l`` et ``--tx-e`` : leurs proportions (largeur / hauteur), que la feuille donne au
    dessin là où sa hauteur est libre (le téléphone)."""
    def vb(v):
        return " ".join(_f(x) for x in v)
    (cl, vl), (ce, ve) = large, etroit
    return (f'<svg xmlns="http://www.w3.org/2000/svg" id="{_esc(ident)}" class="tx-dessin" role="img" aria-label="{_esc(titre)}" '
            f'font-family="{POLICE}" style="--tx-l:{vl[2] / vl[3]:.4f};--tx-e:{ve[2] / ve[3]:.4f}">'
            f'<svg class="tx-large" viewBox="{vb(vl)}" preserveAspectRatio="xMidYMid meet">{cl}</svg>'
            f'<svg class="tx-etroit" viewBox="{vb(ve)}" preserveAspectRatio="xMidYMid meet" style="display:none">{ce}</svg></svg>')


def elevation(ident: str, res: Dict) -> str:
    """L'élévation de la poutre calculée ``res`` (ses deux variantes)."""
    return _deux(ident, "Élévation de la poutre : barres, étriers par zones, barres développées",
                 _elevation(res, DIVISEUR, ELANCEMENT), _elevation(res, DIVISEUR_ETROIT, ELANCEMENT_ETROIT))


def coupe(ident: str, res: Dict) -> str:
    """Les coupes de la poutre calculée ``res`` : une par composition (ses deux variantes)."""
    return _deux(ident, "Coupes de la poutre : béton, étriers, barres repérées",
                 _coupes(res, f"bet-{ident}-l", False), _coupes(res, f"bet-{ident}-e", True))


# ============================================================ le bordereau (un tableau)
def bordereau(res: Dict) -> str:
    """Le bordereau de la poutre, en HTML : un repère par ligne (les barres, puis les étriers de chaque type employé),
    la synthèse par diamètre, le total, la majoration, le total majoré. Longueurs unitaires au millimètre, longueurs
    totales au centimètre, masses linéiques au gramme par mètre, masses au centième de kg : les nombres du moteur,
    écrits par ``commun.fr``. Chaque ligne est une PIÈCE (``data-sel``) : un clic la montre dans les données."""
    t = res["totaux"]
    if not res["reperes"]:
        return ('<div class="tx-bord"><div class="tx-vide"><b>Aucun acier</b>Ajouter un lit, un renfort, des barres de peau ou une zone '
                'd\'étriers : le bordereau apparaît ici.</div></div>')
    rows = ""
    for r in res["reperes"]:
        cle = r["cle"] if r["sorte"] == "barre" else r["cle"].split(".")[0]
        rows += (f'<tr class="pc" data-sel="{_esc(cle)}"><td class="rep"><span class="tx-bul">{r["repere"]}</span></td>'
                 f'<td class="de"><b>{_esc(r["designation"])}</b><span>{_esc(r["position"])}</span></td><td>{r["d"]}</td><td>{r["n"]}</td>'
                 f'<td>{C.fr(r["l_unitaire"], 3)}</td><td>{C.fr(r["l_totale"], 2)}</td><td>{C.fr(r["masse_lineique"], 3)}</td>'
                 f'<td class="kg">{C.fr(r["masse"], 2)}</td></tr>')
    synthese = "".join(
        f'<tr><td class="rep"></td><td class="de"><b>Ø{C.NBSP}{d["d"]}</b><span>repère{"s" if len(d["reperes"]) > 1 else ""} '
        f'{", ".join(str(x) for x in d["reperes"])}</span></td><td>{d["d"]}</td><td></td><td></td><td>{C.fr(d["longueur"], 2)}</td>'
        f'<td>{C.fr(d["masse_lineique"], 3)}</td><td class="kg">{C.fr(d["masse"], 2)}</td></tr>' for d in res["diametres"])
    pied = (f'<tr><td class="rep"></td><td class="de"><b>Total</b><span>acier, sans majoration</span></td><td></td><td></td><td></td>'
            f'<td>{C.fr(t["longueur"], 2)}</td><td></td><td class="kg">{C.fr(t["masse"], 2)}</td></tr>'
            f'<tr><td class="rep"></td><td class="de"><b>Majoration</b><span>chutes et ligatures, {C.nb(t["majoration"])}{C.NBSP}%</span></td><td></td><td></td>'
            f'<td></td><td></td><td></td><td class="kg">{C.fr(t["supplement"], 2)}</td></tr>'
            f'<tr class="tot"><td class="rep"></td><td class="de"><b>Total majoré</b><span>acier à commander</span></td><td></td><td></td><td></td><td></td>'
            f'<td></td><td class="kg">{C.fr(t["masse_majoree"], 2)}</td></tr>')
    return ('<div class="tx-bord"><div class="tx-defile"><table class="tx-tab"><thead><tr><th class="rep">Rep.</th><th class="de">Désignation</th>'
            '<th>Ø<small>mm</small></th><th>Nombre</th><th>Long. unitaire<small>m</small></th><th>Long. totale<small>m</small></th>'
            '<th>Masse linéique<small>kg/m</small></th><th>Masse<small>kg</small></th></tr></thead>'
            f'<tbody>{rows}</tbody><tbody class="syn"><tr class="st"><td class="rep"></td><td class="de" colspan="7">Par diamètre</td></tr>{synthese}</tbody>'
            f'<tfoot>{pied}</tfoot></table></div></div>')


# ============================================================ les vues
def vues(res: Dict, ident: str) -> List[Dict]:
    """Les trois vues de la poutre calculée : l'élévation, les coupes, le bordereau."""
    n = len(res["coupes"])
    return [
        {"cle": "elevation", "titre": "Élévation", "court": "Élévation", "svg": elevation(f"{ident}_elevation", res),
         "legende": "longueurs à l'échelle, hauteurs amplifiées quand la poutre est élancée (le facteur est écrit sur le dessin) · "
                    "abscisses et longueurs en m, pas des étriers en cm · les numéros sont les repères du bordereau"},
        {"cle": "coupe", "titre": "Coupe" if n == 1 else "Coupes", "court": "Coupe", "svg": coupe(f"{ident}_coupe", res),
         "legende": "à l'échelle, cotes en cm · une coupe par composition de la poutre (mêmes barres, même type d'étriers) · "
                    "les numéros sont les repères du bordereau"},
        {"cle": "bordereau", "titre": "Bordereau", "court": "Bordereau", "svg": bordereau(res),
         "legende": "un repère par ligne · longueurs en m, masses en kg"},
    ]

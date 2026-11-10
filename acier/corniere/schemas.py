# -*- coding: utf-8 -*-
"""Cornière d'appui — les QUATRE VUES partagées par l'écran et la note (``docs/acier/CORNIERE_DECISIONS.md`` § 4 et
§ 9.1 à 9.3 ; modèles visuels : figures a, b, c de ``docs/acier/corniere_excel/ANALYSE_CORNIERE_EXCEL.md``), construites
sur le kit de dessin des assemblages (``acier.assemblages.poutre_poutre.doubles_cornieres.schemas`` : feuille de
cotation, primitives, hachures, lignes de rupture, rendu SVG) et les briques des schémas de soudure (flèches d'effort,
étiquettes à halo, étendue réelle du corps).

Chaque vue est une fonction ``(res, fs=None, interactif=True) -> Dessin`` qui ne lit que la SORTIE de
``acier.corniere.moteur.verifier`` : rien n'est recalculé ici — les bras de levier sont des cotes mesurées sur le
dessin (leur valeur est leur longueur), les efforts, les moments et les taux sont ceux du moteur, maille par maille.
``fs`` est la taille du texte dans l'unité de la scène (mm) ; ``interactif`` pose ``data-sel`` sur chaque pièce qu'une
donnée pilote. Toutes les scènes partagent le même repère : le talon à l'origine (dos de l'aile verticale au nu du
support, face extérieure de l'aile horizontale en y = 0), l'aile horizontale vers la droite, l'aile verticale vers le
haut (configuration A) ou vers le bas (B) ; y vers le bas, comme le SVG.

- ``coupe`` — LE plan de principe coté, en mm à l'échelle : le support (« voile » : le mur sans limite, rompu ;
  « poutre » : ses deux arêtes, cotées depuis l'axe d'ancrage le plus proche — ``bords_poutre``, la distance que Hilti
  demande — et COUPÉE par des lignes de rupture quand elle dépasse 1,5 fois la cornière), la cornière (congé r₁, bouts
  r₂ du catalogue), la charge R à X en trait plein et ses deux positions X ± dx en tirets, l'ancrage : trou, tige, les
  rondelles selon ``rondelles`` (ISO 7089 sous l'écrou ; ISO 7093-1 large contre l'aile), l'écrou, la profondeur
  ``profondeur`` cotée dans le support et rendue selon ``type_ancrage`` (mécanique : douille d'expansion au fond ;
  chimique : tige dans son scellement hachuré fin ; traversant : la tige traverse l'épaisseur du support, écrou et
  rondelle sur la face arrière), l'axe et les positions Y ± dy en tirets fins, la résultante de compression C (pivot
  y_c en A, bout libre en B), les bras de levier DU MODE (historique : Xe, Xe+r, Ye, Ye+r, YH ; équilibre : X − uV,
  Y − yc ou ya − c), les sections 1 à 4 en orange avec leur pastille, les cotes larg, haut, e, r, X, Y, les valeurs R,
  H, C. L'état limite (ELS, ELU) est dit par la légende, pas par les étiquettes.
- ``moments`` — « les moments sur la cornière », le principe du classeur (figure c) : la cornière à l'échelle, et le
  diagramme de moment de CHAQUE aile porté PERPENDICULAIREMENT à l'aile depuis son axe (la ligne M = 0, tracée et
  nommée), hors de l'angle — l'aile horizontale vers le bas en A et vers le haut en B, l'aile verticale côté support ;
  une ÉCHELLE commune (``PART_MOMENT`` × haut pour le plus grand moment) ; les trois courbes (une par position de la
  charge : extrêmes en trait fin, nominale en trait moyen), l'ENVELOPPE en gras, remplie légèrement ; le
  prolongement en pointillé dans le bloc d'angle jusqu'à Mmax (et M angle en B) ; les sections 1 à 4 avec la valeur de
  l'enveloppe (M1 à M4, trois décimales), l'axe de l'ancrage et sa bande ± dy, la charge R en petite flèche. Les
  valeurs écrites sont des grandeurs (|M|) : le signe est une affaire de face, dite par la légende (M > 0 : face du
  congé tendue ; tout négatif en B).
- ``contraintes`` — « le maillage de von Mises » : la cornière à l'échelle, découpée en mailles (s_i, s_{i+1}) ×
  (ζ_j, ζ_{j+1}) posées sur la vraie géométrie (ζ = −1 face extérieure, +1 face du congé, qui suit l'arc t(s)), chaque
  maille colorée par le plus grand des taux de ses quatre nœuds (``res["maillage"]``) avec le dégradé IMPOSÉ
  (``couleur_taux`` : 0 → 100 % vert progressif, 100 → 105 % orange progressif, au-delà rouge) ; le bloc d'angle
  (s < e) hachuré gris « hors modèle de poutre » ; le contour par-dessus ; le maximum ◉ de chaque aile à
  ``maxima[aile]["s"]`` sur la face ζ du maillage, étiqueté avec le taux de la VÉRIFICATION ; les sections 1 à 4 ; l'axe
  de l'ancrage ; une barre de couleur (0, 50, 100, 105 %) ; deux lignes qui disent ce que c'est — des contraintes de
  poutre, l'enveloppe des positions, PAS un calcul par éléments finis. Pas de déformée.
- ``elevation`` — à l'échelle : la tranche L vue de face (les n ancrages à ``ancrages["positions_z"]``, la chaîne de
  cotes e₁ · s · … · e₁, N et V par ancrage dans l'état limite saisi) et l'aile horizontale RABATTUE : la charge de
  largeur a (ou L) et sa diffusion à 45° jusqu'à la section 1 — le trapèze qui donne L_eff, coté ; des COUPURES au
  milieu de la tranche quand L dépasse 2,5 fois la hauteur de la cornière (la cote L garde sa vraie valeur).

Les coupures (``_Coupures``) : des tranches de l'axe retirées du dessin et remplacées par un intervalle entre deux
lignes de rupture ; une cote qui traverse une tranche est plus courte que sa valeur — c'est la convention du dessin
de bureau, et la seule entorse à « à l'échelle ». Elles ne s'appliquent qu'à ce qui est long et vide (le support, la
tranche entre deux ancrages, la tige d'une cheville profonde), jamais à la cornière en coupe.

Contrat de sélection (établi compact) : chaque pièce qu'une donnée pilote est un groupe ``<g class="pc"
data-sel="<clé>">`` sous la clé de sa ligne dans les données — X, dx, Y, dy, R, L, a, n, d, profil, yc, uV,
jeu_ancrage (``CLES_SEL``). Une vue qu'on ne sait pas dessiner (proportions absurdes) lève ValueError ; ``secours``
rend le cadre « Dessin impossible à cette échelle ». ``ajuster`` reconstruit une vue jusqu'à ce que son texte se lise
à la taille voulue dans une toile donnée (l'écran : 13 px dans 626 × 682 ; la note choisit la sienne).

Bibliothèque standard ; aucun Streamlit, aucun moteur (le résultat est reçu).
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from acier.assemblages.poutre_poutre.doubles_cornieres.schemas import (   # le kit de dessin partagé
    Dessin, Feuille, PALETTE, _arc, _circle, _group, _hachures, _path, _poly, _rect, _rect_rompu, _text, vers_svg)
from acier.formats import F, f0
from acier.soudures.schemas import _bb, _etendre, _extent, _fleche, _label, _tw, options_schemas   # briques partagées

from .entree import LIBELLES_CONFIGURATION, LIBELLES_MODE, LIBELLES_SUPPORT, LIBELLES_TYPE_ANCRAGE, POUTRE
from .statique import HISTORIQUE, HORIZONTALE, VERTICALE

# les vues : (clé, titre, titre court, à l'échelle — la feuille peut écrire « mm · à l'échelle »)
VUES: Tuple[Tuple[str, str, str, bool], ...] = (
    ("coupe", "Coupe · plan de principe coté", "Coupe", True),
    ("moments", "Moments · sur la cornière", "Moments", True),
    ("contraintes", "Contraintes · von Mises, maillage", "Contraintes", True),
    ("elevation", "Élévation · la tranche L et l'aile rabattue", "Élévation", True))
# les clés de données qu'une pièce du dessin peut désigner (data-sel)
CLES_SEL = ("X", "dx", "Y", "dy", "R", "L", "a", "n", "d", "profil", "yc", "uV", "jeu_ancrage")
# la lisibilité : sous ce rapport entre la hauteur de la cornière et la taille du texte, le dessin ne se lit plus
LISIBLE_MIN = 1.6
# les moments : le plus grand moment dessiné occupe cette part de la hauteur de la cornière (§ 9.1 : 0,6 à 0,8)
PART_MOMENT = 0.7
# les coupures : le support au-delà de ce multiple de la hauteur de la cornière, la tranche au-delà de celui-ci de L
COUPURE_SUPPORT, COUPURE_TRANCHE, COUPURE_TIGE = 1.5, 2.5, 1.2
# le dégradé imposé des taux (§ 9.1) : 0 → 100 % vert progressif, 100 → 105 % orange progressif, au-delà rouge
DEGRADE_VERT = ("#E8F5E9", "#1B5E20")
DEGRADE_ORANGE = ("#FFB74D", "#E65100")
ROUGE_TAUX = "#C62828"
SEUIL_ORANGE = 1.05


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
    """Un moment en kN·m à trois décimales, comme le classeur (2,010) — la grandeur, sans signe."""
    return _num(abs(x), 3)


def _pct(t: float) -> str:
    return _num(100.0 * t, 1) + " %"


def _symbole_sigma(mat: Dict) -> str:
    """Le nom de la limite en contrainte, sans indice brut : σadm (ELS) ou fy/γM0 (ELU)."""
    return "σadm" if mat.get("etat_limite", "ELS") == "ELS" else "fy/γM0"


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
                deport: int = +1, forcer_deport: bool = False):
    """Une cote posée dans le corps (hors des rangées de la feuille) : lignes d'attache depuis ``a`` et ``b``, ligne de
    cote en ``pos`` (y d'une cote horizontale, x d'une verticale), repères obliques, étiquette au milieu — déportée au
    bout ``deport`` (+1 : droite ou bas ; −1 : gauche ou haut) quand la cote est trop courte pour la porter, ou
    toujours (``forcer_deport``)."""
    t = 0.34 * fs
    w = _tw(txt, fs)
    if horizontale:
        (x0, y0), (x1, y1) = sorted([a, b])
        e0 = pos + (0.45 * fs if pos >= y0 else -0.45 * fs)
        ext = _path(["ext"], [[(x0, y0), (x0, e0)], [(x1, y1), (x1, e0)]], hint="V")
        dln = _path(["dln"], [[(x0, pos), (x1, pos)], [(x0 - t, pos + t), (x0 + t, pos - t)], [(x1 - t, pos + t), (x1 + t, pos - t)]], hint="H")
        if x1 - x0 >= w + 0.9 * fs and not forcer_deport:
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
        if y1 - y0 >= w + 0.9 * fs and not forcer_deport:
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
    """Un dessin sans feuille de cotation : la boîte de vue est l'étendue du corps plus une marge."""
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


def _boites(prims: List, fs: float) -> List[Tuple[float, float, float, float]]:
    """Les boîtes (x1, y1, x2, y2) des étiquettes à halo d'une liste de primitives (groupes qui portent un ``hit``),
    pour poser une pièce sans la recouvrir."""
    out = []
    for p in prims:
        if p["t"] != "g":
            continue
        if any(e["t"] == "rect" and "hit" in e["cls"] for e in p["enfants"]):
            ex = _extent([p], fs)
            if ex:
                out.append(ex)
        else:
            out += _boites(p["enfants"], fs)
    return out


def _libre(boite: Tuple[float, float, float, float], boites: Sequence[Tuple[float, float, float, float]], marge: float = 0.0) -> bool:
    """Vrai quand la boîte ne croise aucune des autres (à ``marge`` près)."""
    x1, y1, x2, y2 = boite
    return all(x2 + marge <= a or x1 - marge >= c or y2 + marge <= b or y1 - marge >= d for a, b, c, d in boites)


# ====================================================================== les coupures : une pièce trop longue pour la feuille
class _Coupures:
    """Des TRANCHES d'un axe retirées du dessin : chaque tranche [a, b] est remplacée par un intervalle ``gap`` entre
    deux lignes de rupture, et ce qui est au-delà se rapproche ; ``fixe`` est la coordonnée qui ne bouge pas (le
    talon, le bout gauche). Une cote qui traverse une tranche est dessinée plus courte que sa valeur — la convention
    du dessin de bureau. ``f`` donne la coordonnée dessinée d'une coordonnée réelle ; ``segments`` découpe un
    intervalle réel en morceaux hors des tranches."""

    def __init__(self, tranches: Sequence[Tuple[float, float]], gap: float, fixe: float = 0.0):
        self.gap, self.fixe = gap, fixe
        self.t = sorted((min(a, b), max(a, b)) for a, b in tranches if abs(b - a) > 1e-9)
        self.dessus = [x for x in self.t if x[0] >= fixe]                 # au-delà du point fixe, de la plus proche à la plus loin
        self.dessous = [x for x in reversed(self.t) if x[1] <= fixe]      # en deçà, de la plus proche à la plus loin

    def __bool__(self) -> bool:
        return bool(self.t)

    def f(self, v: float) -> float:
        """La coordonnée dessinée (dans une tranche : au prorata, dans l'intervalle de rupture)."""
        d = 0.0
        if v >= self.fixe:
            for a, b in self.dessus:
                if v >= b:
                    d += (b - a) - self.gap
                elif v > a:
                    return a - d + (v - a) / (b - a) * self.gap
            return v - d
        for a, b in self.dessous:
            if v <= a:
                d += (b - a) - self.gap
            elif v < b:
                return b + d - (b - v) / (b - a) * self.gap
        return v + d

    def segments(self, lo: float, hi: float) -> List[Tuple[float, float]]:
        """Les morceaux de [lo, hi] hors des tranches, en coordonnées RÉELLES."""
        out, cur = [], lo
        for a, b in self.t:
            if b <= lo or a >= hi:
                continue
            if a > cur:
                out.append((cur, a))
            cur = max(cur, b)
        if cur < hi:
            out.append((cur, hi))
        return out

    def retire(self) -> float:
        """La longueur retirée du dessin (tranches moins intervalles de rupture)."""
        return sum((b - a) - self.gap for a, b in self.t)


def _rect_coupe(x: float, y: float, w: float, h: float, bords, C: _Coupures, axe: str, amp: float) -> List[List[Tuple[float, float]]]:
    """Un rectangle (coordonnées réelles) coupé par les tranches de ``C`` le long de l'axe ``axe`` (« x » ou « y ») :
    un polygone par morceau, en coordonnées dessinées, les bords coupés en lignes de rupture ; ``bords`` ⊆ {T, R, B, L}
    sont les bords du rectangle entier qui sont eux-mêmes des ruptures (la pièce continue au-delà)."""
    bords = set(bords)
    if axe == "x":
        morceaux = C.segments(x, x + w)
        lo_hi, autres = ("L", "R"), bords & {"T", "B"}
    else:
        morceaux = C.segments(y, y + h)
        lo_hi, autres = ("T", "B"), bords & {"L", "R"}
    out = []
    for i, (a, b) in enumerate(morceaux):
        bd = set(autres)
        bd |= ({lo_hi[0]} & bords) if i == 0 else {lo_hi[0]}
        bd |= ({lo_hi[1]} & bords) if i == len(morceaux) - 1 else {lo_hi[1]}
        fa, fb = C.f(a), C.f(b)
        if axe == "x":
            out.append(_rect_rompu(fa, y, fb - fa, h, bd, amp))
        else:
            out.append(_rect_rompu(x, fa, w, fb - fa, bd, amp))
    return out


def _segments_x(C: _Coupures, x0: float, x1: float, y: float) -> List[List[Tuple[float, float]]]:
    """Une ligne horizontale réelle [x0, x1] en morceaux dessinés hors des tranches."""
    return [[(C.f(a), y), (C.f(b), y)] for a, b in C.segments(min(x0, x1), max(x0, x1))]


def coupures_elevation(L: float, haut: float, protegees: Sequence[Tuple[float, float]]) -> _Coupures:
    """Les tranches de la longueur L retirées de l'élévation : rien tant que L ≤ 2,5 × haut ; sinon chaque intervalle
    LIBRE (hors des zones protégées : les ancrages, la charge et sa diffusion) assez long pour qu'une coupure y fasse
    gagner de la place est réduit à deux gardes et un intervalle de rupture. Tout est en proportion de la pièce, jamais
    du texte : la décision ne bouge pas quand ``ajuster`` fait varier la taille du texte."""
    gap, garde = 0.15 * haut, 0.2 * haut
    if L <= COUPURE_TRANCHE * haut:
        return _Coupures([], gap)
    lmin = 2.0 * garde + gap + 0.5 * haut              # une coupure qui ne retire pas une demi-hauteur ne vaut pas sa rupture
    zones = sorted((max(0.0, a), min(L, b)) for a, b in protegees if b > 0 and a < L)
    libres, cur = [], 0.0
    for a, b in zones:
        if a > cur:
            libres.append((cur, a))
        cur = max(cur, b)
    if cur < L:
        libres.append((cur, L))
    # les intervalles entre ancrages se coupent dès que la coupure fait gagner de la place ; les bouts (e₁) seulement
    # quand ils sont longs face à la cornière — « des coupures au milieu de la tranche »
    tranches = [(a + garde, b - garde) for a, b in libres
                if b - a > lmin and (a > 1e-9 and b < L - 1e-9 or b - a > 1.2 * haut)]
    return _Coupures(tranches, gap)


# ====================================================================== le dégradé des taux (§ 9.1)
def _hex(c: str) -> Tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _lerp(c0: str, c1: str, t: float) -> str:
    t = min(1.0, max(0.0, t))
    a, b = _hex(c0), _hex(c1)
    return "#%02X%02X%02X" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def couleur_taux(t: float) -> str:
    """La couleur IMPOSÉE d'un taux : de 0 à 100 % un vert qui fonce, de 100 à 105 % un orange qui fonce, rouge au-delà."""
    if t <= 1.0:
        return _lerp(DEGRADE_VERT[0], DEGRADE_VERT[1], t)
    if t <= SEUIL_ORANGE:
        return _lerp(DEGRADE_ORANGE[0], DEGRADE_ORANGE[1], (t - 1.0) / (SEUIL_ORANGE - 1.0))
    return ROUGE_TAUX


# ====================================================================== la coupe : le plan de principe
def coupe(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """Le plan de principe coté (voir l'en-tête du module)."""
    g, sec, cfg, mode = res["geometrie"], res["section"], res["configuration"], res["mode"]
    e, r, larg, haut = g["e"], g["r"], g["larg"], g["haut"]
    r2 = float(sec.get("r2") or 0.0)
    X, dx, Y, dy, d, d0 = g["X"], g["dx"], g["Y"], g["dy"], g["d"], g["d0"]
    prof, typ, nr, rond = g["profondeur"], g["type_ancrage"], g["rondelles"], g["rondelle"]
    poutre = g["support"] == POUTRE
    traversant = typ == "traversant"
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
    # --- la profondeur d'ancrage dans le support : à l'échelle, coupée (lignes de rupture) quand la cheville est
    #     profonde face à la cornière (COUPURE_TIGE × haut) — la cote garde sa vraie valeur ; en proportion de la
    #     pièce, pas du texte (la décision ne bouge pas sous ``ajuster``)
    garde_fond = max(2.6 * d, 0.3 * haut) if not traversant else 0.2 * haut
    Cx = _Coupures([(-(prof - garde_fond), -0.45 * haut)] if prof > COUPURE_TIGE * haut and prof - garde_fond > 0.45 * haut + 0.3 * haut else [],
                   0.15 * haut)
    x_fond = Cx.f(-prof)                             # le fond du forage (ou la face arrière du support), dessiné
    # --- l'ancrage : le trou dans l'aile, la pile (rondelles, écrou), la tige, l'axe ; dans le support, le rendu du type
    ya = yv(Y)
    pile: List[Dict] = []
    x = e
    if nr == 2:
        D_, ep_ = rond["large"]
        pile.append(_rect(["bo"], x, ya - D_ / 2, ep_, D_))        # rondelle large ISO 7093-1 contre l'aile
        x += ep_
    if nr >= 1:
        D_, ep_ = rond["normale"]
        pile.append(_rect(["bo"], x, ya - D_ / 2, ep_, D_))        # rondelle ISO 7089 sous l'écrou
        x += ep_
    x_n = x                                          # le nu de l'écrou
    pile.append(_rect(["bo"], x_n, ya - 0.85 * d, 0.8 * d, 1.7 * d))
    xa0 = x_n + 0.8 * d + 0.3 * d                    # le bout de la tige
    D_n, ep_n = rond["normale"]
    x_tige = -prof if not traversant else -prof - ep_n - 1.1 * d
    s.append(_rect(["bg"], 0.0, ya - d0 / 2, e, d0))
    s.append(_path(["fl2"], [[(0.0, ya - d0 / 2), (e, ya - d0 / 2)], [(0.0, ya + d0 / 2), (e, ya + d0 / 2)]]))
    tige = [_poly(["tg"], p) for p in _rect_coupe(x_tige, ya - d / 2, xa0 - x_tige, d, set(), Cx, "x", 0.3 * fs)] + pile
    _piece(s, tige, sel("d"))
    _piece(s, [_path(["ax"], [[(Cx.f(x_tige) - 0.6 * fs, ya), (xa0 + 0.9 * fs, ya)]])], sel("Y"))
    if dy > 0:
        pos = [_path(["hd"], [[(-1.6 * fs, yv(Y + j)), (xa0 + 0.4 * fs, yv(Y + j))]]) for j in (-dy, dy)]
        _cote_libre(pos, fs, (xa0 + 0.6 * fs, ya), (xa0 + 0.6 * fs, yv(Y + dy)), xa0 + 1.3 * fs,
                    f"dy = {f0(dy)}", horizontale=False, deport=-1 if A else +1)
        _piece(s, pos, sel("dy"))
    fond: List[Dict] = []                            # sous la tige (le scellement)
    dessus: List[Dict] = []                          # sur la tige (la douille, la pile arrière)
    if typ == "mecanique":
        # la douille d'expansion au fond : un manchon plus large que la tige, son cône
        douille = [_rect(["bo"], x_fond + 0.5 * d, ya - 0.66 * d, 1.5 * d, 1.32 * d),
                   _poly(["tg"], [(x_fond + 0.5 * d, ya - 0.66 * d), (x_fond, ya - 0.42 * d), (x_fond, ya + 0.42 * d), (x_fond + 0.5 * d, ya + 0.66 * d)])]
        dessus += douille
    elif typ == "chimique":
        # le forage rempli de résine : une bande hachurée fin autour de la tige, du nu du support au fond
        for p in _rect_coupe(-prof, ya - 0.62 * d, prof, 1.24 * d, set(), Cx, "x", 0.3 * fs):
            fond += [_poly(["scel"], p), _hach(["ht"], p, 0.32 * fs)]
    else:
        # le boulon traversant : rondelle et écrou sur la face arrière du support, à x = −profondeur
        dessus += [_rect(["bo"], x_fond - ep_n, ya - D_n / 2, ep_n, D_n), _rect(["bo"], x_fond - ep_n - 0.8 * d, ya - 0.85 * d, 0.8 * d, 1.7 * d)]
    # --- la charge R à X (plein) et ses positions X ± dx (tirets) ; en A, quand l'ancrage est bas, la flèche s'allonge
    #     pour que l'étiquette de R passe au-dessus de celles de l'ancrage (H, Ø), posées ensuite par-dessus son fût
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
    # --- la feuille de cotation : la boîte des pièces, élargie à l'étendue du corps (flèches, étiquettes, pastilles,
    #     et l'ancrage dans le support : les rangées de cotes de gauche commencent au-delà de son fond)
    bb0 = dict(x1=0.0, x2=larg, y1=min(yv(0.0), yv(haut)), y2=max(yv(0.0), yv(haut)))
    S = Feuille(_bb(bb0, s + past + fond + dessus, fs), fs, options_schemas(2))
    cote_x = "B" if A else "T"                     # les cotes de l'aile horizontale, du côté de sa face extérieure
    # --- le support : les tranches coupées (poutre au-delà de 1,5 × haut) — avant les cotes, qui lisent ces positions
    hs, bs = g["support_haut"], g["support_bas"]
    Cy = _Coupures([], 0.15 * haut)
    if poutre:
        occupe_h, occupe_b = (haut, 0.0) if A else (0.0, haut)        # ce que la cornière occupe au-dessus / au-dessous de y = 0
        garde_c, garde_b = 0.4 * haut, 0.2 * haut                     # en proportion de la pièce : stable sous ``ajuster``
        tranches = []
        if hs > COUPURE_SUPPORT * haut and hs - garde_b - (occupe_h + garde_c) > 0.3 * haut:
            tranches.append((-(hs - garde_b), -(occupe_h + garde_c)))
        if bs > COUPURE_SUPPORT * haut and bs - garde_b - (occupe_b + garde_c) > 0.3 * haut:
            tranches.append((occupe_b + garde_c, bs - garde_b))
        Cy = _Coupures(tranches, 0.15 * haut)
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
    if poutre:
        # les arêtes de la poutre, cotées depuis l'axe d'ancrage le plus proche (la distance au bord pour Hilti) : la
        # cote traverse la coupure et garde sa vraie valeur
        bords = g["bords_poutre"]
        _dim_sel(S, dict(id="bord_haut", side="L", a=yv(Y - k * dy), b=Cy.f(-hs), o1=0.0, o2=0.0, sym="bord haut", val=bords["haut"], lvl=1, out="lo"), None)
        _dim_sel(S, dict(id="bord_bas", side="L", a=yv(Y + k * dy), b=Cy.f(bs), o1=0.0, o2=0.0, sym="bord bas", val=bords["bas"], lvl=1), None)
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
    # au-dessus de l'écrou, sinon au-dessus de l'étiquette de R ; le premier emplacement dont l'étiquette ne recouvre
    # aucune autre (R, H, Ø, dy, C, les pastilles) est retenu
    if r > 0:
        txt_r = f"r = {f0(r)}"
        w = _tw(txt_r, fs) + 0.9 * fs
        c45 = math.cos(math.pi / 4)
        pm = (e + r - r * c45, yv(e + r - r * c45))
        y0_r = e + r + 1.6 * fs                      # juste au-dessus de la pastille 3
        bord = X - dx - 0.5 * fs                     # la flèche de charge la plus proche…
        if y0_r + 0.7 * fs > y_R - 0.7 * fs:         # … et l'étiquette de R si l'on monte jusqu'à elle
            bord = min(bord, X - w_R / 2 - 0.5 * fs)
        cands = []
        if not A and e + r + 3.2 * fs + w <= larg - 1.0 * fs:
            cands.append((e + r + 3.2 * fs, yv(e + 2.2 * fs)))     # B : l'intérieur de l'angle est libre, à droite de la pastille 4
        if A and (Y - 0.85 * d - 0.3 * fs) >= y0_r + 0.7 * fs and e + 0.8 * fs + w <= bord:
            cands.append((e + 0.8 * fs, yv(y0_r)))
        if larg - (X + dx) >= w + 1.2 * fs:
            cands.append((X + dx + 0.6 * fs, yv(e + 1.0 * fs)))
        if Y + 0.85 * d + 2.4 * fs <= haut:
            cands.append(((xa0 + 2.3 * fs) if dy > 0 else (e + 1.0 * fs), yv(Y + 0.85 * d + 1.8 * fs)))
        cands.append((e + r + 1.0 * fs, yv(e + 8.8 * fs)))
        occupe = _boites(s + past, fs) + [(_x - 0.78 * fs, _y - 0.78 * fs, _x + 0.78 * fs, _y + 0.78 * fs)
                                            for _x, _y in ((e + r, y1p), (e, y2p), (e + 1.8 * fs, yv(e + r)), (e + r + (3.4 * fs if etag else 1.8 * fs), yv(e)))]
        lab = cands[-1]
        for cx_, cy_ in cands:
            boite = (cx_, cy_ - 0.75 * fs, cx_ + w, cy_ + 0.75 * fs)
            if _libre(boite, occupe, 0.45 * fs):
                lab = (cx_, cy_)
                break
        rv: List[Dict] = []
        _renvoi(rv, pm[0], pm[1], lab[0], lab[1], txt_r, fs)
        _piece(S.g, rv, sel("profil"))
    # --- la profondeur d'ancrage, cotée dans le support : du côté de la tige où aucune étiquette ne gêne (la
    #     résultante C, les cotes des rangées), en s'éloignant de la tige au besoin
    txt_p = f"hef = {f0(prof)}" if not traversant else f"ép. = {f0(prof)}"
    w_p = _tw(txt_p, fs)
    boites = _boites(s + S.g, fs)
    cand = []
    for i in range(4):
        for signe in ((-1, 1) if A else (1, -1)):             # d'abord vers le bout libre de l'aile verticale
            cand.append(ya + signe * (d / 2 + (1.6 + 1.5 * i) * fs))
    pos_p = cand[0]
    cx_p = (x_fond + 0.0) / 2.0
    for pos in cand:
        boite = (cx_p - w_p / 2 - 0.3 * fs, pos - 1.47 * fs, cx_p + w_p / 2 + 0.3 * fs, pos - 0.05 * fs)
        if _libre(boite, boites, 0.2 * fs):
            pos_p = pos
            break
    _cote_libre(S.g, fs, (x_fond, ya), (0.0, ya), pos_p, txt_p)
    S.g += secs + past
    _etendre(S, past, fs)
    # --- le support, derrière tout : le voile sans limite (rompu en haut et en bas), ou la poutre entre ses deux
    #     arêtes, coupée quand elle est haute ; l'épaisseur englobe la cheville (ou vaut la profondeur, traversant)
    n_L = len(S.rows["L"])
    ep = -x_fond if traversant else max(-x_fond + 2.4 * fs, 1.4 * fs + n_L * 2.35 * fs + 1.6 * fs)
    rows_T, rows_B = S.rows["T"], S.rows["B"]
    if poutre:
        y_top, y_bot = Cy.f(-hs), Cy.f(bs)
        morceaux = _rect_coupe(-ep, -hs, ep, hs + bs, set() if traversant else {"L"}, Cy, "y", 0.5 * fs)
        nom = "poutre"
    else:
        y_top = min(S.ex["y1"], S.pos("T", len(rows_T) - 1) - 1.7 * fs if rows_T else S.ex["y1"]) - 3.5 * fs
        y_bot = max(S.ex["y2"], S.pos("B", len(rows_B) - 1) + 0.6 * fs if rows_B else S.ex["y2"]) + 2.0 * fs
        morceaux = [_rect_rompu(-ep, y_top, ep, y_bot - y_top, {"T", "B"} | (set() if traversant else {"L"}), 0.5 * fs)]
        nom = "voile"
    mur: List[Dict] = []
    for p in morceaux:
        mur += [_poly(["be"], p), _hach(["ht"], p, 0.9 * fs)]
    # son nom, dans son coin haut contre la cornière (les rangées de cotes occupent l'autre côté)
    mur.append(_text(["tx2"], -0.6 * fs, y_top + 1.5 * fs, nom, 0.9 * fs, "end"))
    _etendre(S, mur + fond + dessus, fs)
    aria = (f"Coupe : plan de principe de la cornière {sec['designation']} en configuration {cfg} ({LIBELLES_CONFIGURATION[cfg]}), "
            f"{LIBELLES_MODE[mode]} — charge R à X, ancrage à Y ({LIBELLES_TYPE_ANCRAGE[typ]}, profondeur {f0(prof)} mm) dans "
            f"{LIBELLES_SUPPORT[g['support']]}, sections 1 à 4, bras de levier")
    return S.finish(mur + fond + s + dessus, [], aria)


# ====================================================================== l'élévation : la tranche L et l'aile rabattue
def elevation(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """La tranche L vue de face et l'aile horizontale rabattue (voir l'en-tête du module)."""
    g, sec, cfg, anc, eff = res["geometrie"], res["section"], res["configuration"], res["ancrages"], res["efforts"]
    e, r, larg, haut, L = g["e"], g["r"], g["larg"], g["haut"], g["L"]
    n, d, d0, Y, X = g["n"], g["d"], g["d0"], g["Y"], g["X"]
    zs = list(anc["positions_z"])
    etat = anc["etat_limite"]
    a = g["a"] if g["a"] is not None else L
    cb = g["combinaison_L_eff"]
    Xp = cb["X"]
    Xe = Xp - r - e
    L_eff = g["L_eff"]
    A = cfg == "A"
    k = -1.0 if A else 1.0

    def yv(y: float) -> float:
        return k * y

    fs = float(fs) if fs else (min(L, COUPURE_TRANCHE * haut) + 1.5 * (haut + larg)) / 44.0
    _lisible(haut, fs, "la hauteur de la cornière")
    sel = (lambda cle: cle) if interactif else (lambda cle: None)
    gap = 3.2 * fs                                 # entre la face de la cornière et l'aile rabattue

    def yu(u: float) -> float:
        """y de la scène d'un point de l'aile horizontale rabattue (u depuis le nu du mur) : le bout libre reste contre
        l'élévation, le côté du mur s'en éloigne."""
        return -k * (gap + larg - u)

    # --- les coupures de la tranche : hors des ancrages, hors de la charge et de sa diffusion (en proportion de la pièce)
    txt_n, txt_v = f"N {etat} = {_kn(anc['N'])}", f"V {etat} = {_kn(anc['V'])}"
    w_nv = max(_tw(txt_n, 0.9 * fs), _tw(txt_v, 0.9 * fs)) + 0.8 * fs
    dep = 0.95 * d + 0.5 * fs
    z0, z1 = max(0.0, L / 2 - a / 2), min(L, L / 2 + a / 2)
    zone = 0.95 * d + 0.2 * haut
    protegees = [(z - zone, z + zone) for z in zs]
    if a < L - 1e-9:
        protegees.append((z0 - max(Xe, 0.0) - 0.1 * haut, z1 + max(Xe, 0.0) + 0.1 * haut))
    C = coupures_elevation(L, haut, protegees)
    fz = C.f
    Ld = fz(L)                                     # la longueur dessinée
    s: List[Dict] = []
    # --- le mur derrière l'aile verticale (il continue au-delà : rompu), la face de l'aile verticale, la tranche de
    #     l'aile horizontale devant
    y_lo, y_hi = sorted([yv(0.0), yv(haut)])
    marge_z, marge_h = 3.6 * fs, 2.6 * fs
    y_mur = (y_lo - marge_h) if A else (y_lo - 0.5 * fs)
    h_mur = haut + marge_h + 0.5 * fs
    for p in _rect_coupe(-marge_z, y_mur, L + 2 * marge_z, h_mur, {"L", "R", "T", "B"}, C, "x", 0.5 * fs):
        s.append(_poly(["be"], p))
    piece_prims = [_poly(["ps"], p) for p in _rect_coupe(0.0, y_lo, L, haut, set(), C, "x", 0.5 * fs)]
    piece_prims += [_poly(["pp"], p) for p in _rect_coupe(0.0, min(yv(0.0), yv(e)), L, e, set(), C, "x", 0.5 * fs)]
    # --- l'aile horizontale rabattue : la tranche de l'aile verticale vue de dessus (u ∈ [0, e]), le dessus de l'aile,
    #     la tangente du congé
    u_lo = min(yu(0.0), yu(e))
    piece_prims += [_poly(["ps"], p) for p in _rect_coupe(0.0, min(yu(e), yu(larg)), L, larg - e, set(), C, "x", 0.5 * fs)]
    piece_prims += [_poly(["pp"], p) for p in _rect_coupe(0.0, u_lo, L, e, set(), C, "x", 0.5 * fs)]
    piece_prims.append(_path(["flr"], _segments_x(C, 0.0, L, yu(e + r))))
    _piece(s, piece_prims, sel("profil"))
    titre_plan = "aile horizontale rabattue"
    if Ld >= _tw(titre_plan, 0.9 * fs) + 1.0 * fs:
        s.append(_text(["tx2"], Ld / 2, (yv(0.0) + yu(larg)) / 2 + 0.3 * fs, titre_plan, 0.9 * fs, "middle"))
    # --- la section 1 (pied du congé) sur toute la longueur, sa pastille
    seg1 = _segments_x(C, 0.0, L, yu(e + r))
    seg1[0][0] = (-0.6 * fs, yu(e + r))
    seg1[-1][-1] = (Ld + 0.6 * fs, yu(e + r))
    s.append(_path(["sc"], seg1))
    s.append(_pastille(-1.6 * fs, yu(e + r), 1, fs))
    # --- la charge : la bande de largeur a centrée sur L à u = X ; la position X′ qui gouverne L_eff en tirets si
    #     elle diffère ; la diffusion à 45° jusqu'à la section 1 (le trapèze L_eff)
    charge: List[Dict] = [_poly(["ct"], p) for p in _rect_coupe(z0, min(yu(X - 0.45 * fs), yu(X + 0.45 * fs)), z1 - z0, 0.9 * fs, set(), C, "x", 0.3 * fs)]
    if abs(Xp - X) > 1e-9:
        charge.append(_path(["fzd"], _segments_x(C, z0, z1, yu(Xp))))
    txt_r = f"R = {_kn(eff['R'])}"
    if a < L - 1e-9 and Xe > 1e-9:
        # le trapèze, coupé aux bords de la tranche : les côtés à 45° (du bord de la charge vers le pied du congé) —
        # dans une zone protégée des coupures : ses points se dessinent tels quels
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
        charge.append(_poly(["ae"], [(fz(z), yu(u)) for z, u in poly]))
        charge.append(_path(["ael"], [[(fz(z0), yu(Xp)), (fz(max(za, 0.0)), yu(Xp - (z0 - max(za, 0.0))))],
                                      [(fz(z1), yu(Xp)), (fz(min(zb, L)), yu(Xp - (min(zb, L) - z1)))]]))
        _cote_libre(charge, fs, (fz(z0), yu(X)), (fz(z1), yu(X)), yu(X + 1.4 * fs), f"a = {f0(a)}", deport=+1)
        _label(charge, fz(L / 2), yu(X + 3.7 * fs) + 0.34 * fs, txt_r, fs)
    else:
        _label(charge, Ld / 2, yu(X + 1.5 * fs) + 0.34 * fs, f"{txt_r} sur L", fs)
    _piece(s, charge, sel("a"))
    # --- les ancrages à leurs positions z_i : rondelle, tige, axes ; N et V par ancrage (état limite saisi) quand la
    #     place le permet
    ancr: List[Dict] = []
    ya = yv(Y)
    for z in zs:
        ancr += [_circle(["bw"], fz(z), ya, 0.95 * d), _circle(["bo"], fz(z), ya, d / 2)]
    ancr.append(_path(["ax"], _segments_x(C, zs[0] - dep, zs[-1] + dep, ya) + [[(fz(z), ya - dep), (fz(z), ya + dep)] for z in zs]))
    _piece(s, ancr, sel("n"))
    y_et = ya - k * (0.95 * d + 1.4 * fs)          # vers le bout libre de l'aile verticale
    ecarts = [fz(b) - fz(a_) for a_, b in zip(zs[:-1], zs[1:])] + [2 * fz(zs[0]), 2 * (Ld - fz(zs[-1]))]
    place = haut - Y - 0.95 * d >= 3.2 * fs and min(ecarts) >= w_nv
    if place:
        for z in zs:
            _label(s, fz(z), y_et - (0.7 * fs if A else -0.7 * fs) + 0.34 * fs, txt_n, 0.9 * fs)
            _label(s, fz(z), y_et + (0.7 * fs if A else -0.7 * fs) + 0.34 * fs, txt_v, 0.9 * fs)
    # le diamètre et le nombre : un renvoi depuis le premier ancrage
    dlab: List[Dict] = []
    _renvoi(dlab, fz(zs[0]) + 0.7 * d, ya + k * 0.7 * d, fz(zs[0]) + 0.95 * d + 1.6 * fs, ya + k * (0.95 * d + 1.3 * fs), f"{n} Ø{f0(d)}", 0.9 * fs)
    _piece(s, dlab, sel("d"))
    # --- la feuille : la chaîne des cotes e₁ · s · … · e₁ du côté de l'aile verticale, L et L_eff du côté de l'aile rabattue
    bb0 = dict(x1=0.0, x2=Ld, y1=min(y_lo, yu(0.0), yu(larg)), y2=max(y_hi, yu(0.0), yu(larg)))
    S = Feuille(_bb(bb0, s, fs), fs, options_schemas(2))
    cote_chaine = "T" if A else "B"
    cote_L = "B" if A else "T"
    xs = [0.0] + zs + [L]
    o_ch = yv(haut)
    t = 0.34 * fs
    rang = S.slot(cote_chaine, 0.0, Ld)
    p = S.pos(cote_chaine, rang)
    e_ch = p + (-0.45 if cote_chaine == "T" else 0.45) * fs
    ext = _path(["ext"], [[(fz(x), o_ch), (fz(x), e_ch)] for x in xs], hint="V")
    dln = _path(["dln"], [[(fz(xs[0]), p), (fz(xs[-1]), p)]] + [[(fz(x) - t, p + t), (fz(x) + t, p - t)] for x in xs], hint="H")
    labs = []
    for z_a, z_b in zip(xs[:-1], xs[1:]):
        txt = f0(z_b - z_a)
        if fz(z_b) - fz(z_a) >= _tw(txt, fs) + 0.6 * fs:
            labs.append(S.label((fz(z_a) + fz(z_b)) / 2, p - 0.42 * fs, False, txt, dict(id="entraxe", sym="", lvl=1)))
    _piece(S.g, [_group(["dm"], [ext, dln] + labs)], sel("n"))
    if g["a"] is not None:
        _dim_sel(S, dict(id="Leff", side=cote_L, a=fz(max(0.0, L / 2 - L_eff / 2)), b=fz(min(L, L / 2 + L_eff / 2)), o1=yu(e + r), o2=yu(e + r),
                         sym="Leff", val=L_eff, lvl=1), sel("a"))
    _dim_sel(S, dict(id="L", side=cote_L, a=0.0, b=Ld, o1=yu(0.0), o2=yu(0.0), sym="L", val=L, lvl=1), sel("L"))
    _dim_sel(S, dict(id="haut", side="L", a=yv(0.0), b=yv(haut), o1=0.0, o2=0.0, sym="haut", val=haut, lvl=1), sel("profil"))
    _dim_sel(S, dict(id="larg", side="L", a=yu(0.0), b=yu(larg), o1=0.0, o2=0.0, sym="larg", val=larg, lvl=1), sel("profil"))
    _dim_sel(S, dict(id="Y", side="R", a=yv(0.0), b=yv(Y), o1=Ld, o2=Ld, sym="Y", val=Y, lvl=1), sel("Y"))
    _dim_sel(S, dict(id="X", side="R", a=yu(0.0), b=yu(X), o1=Ld, o2=Ld, sym="X", val=X, lvl=1), sel("X"))
    if not place:
        # pas la place sous chaque ancrage : une ligne pour tous, dans une rangée libre du côté de l'aile rabattue
        txt = f"par ancrage : {txt_n} · {txt_v}"
        w = _tw(txt, fs) * 1.08
        rg = S.slot(cote_L, Ld / 2 - w / 2, Ld / 2 + w / 2)
        pp = S.pos(cote_L, rg)
        hit = _rect(["hit"], -w / 2 - 0.3 * fs, -1.05 * fs, w + 0.6 * fs, 1.42 * fs)
        S.g.append(_group(["fzl"], [hit, _text([], None, None, txt, fs)], x=Ld / 2, y=pp + (0.3 if cote_L == "B" else -0.1) * fs, rot=0))
    _etendre(S, s, fs)
    aria = (f"Élévation : la tranche L = {f0(L)} mm de la cornière {sec['designation']} vue de face, {n} ancrage(s) Ø{f0(d)} à "
            f"l'entraxe {f0(anc['entraxe'])} mm (e₁ = {f0(anc['distance_bout'])} mm), N et V par ancrage ({etat}), et l'aile horizontale "
            f"rabattue avec la charge de largeur {f0(a)} mm et sa diffusion à 45°" + (" — tranche coupée au dessin" if C else ""))
    return S.finish(s, [], aria)


# ====================================================================== les moments sur la cornière
def _etiquette_pastille(s: List, x_p: float, y_p: float, num: int, txt: str, fs: float, cote: int):
    """Une pastille de section et, à côté (``cote`` +1 : à droite, −1 : à gauche), la valeur de l'enveloppe."""
    s.append(_pastille(x_p, y_p, num, fs))
    w = _tw(txt, fs)
    _label(s, x_p + cote * (1.1 * fs + w / 2 + 0.3 * fs), y_p + 0.34 * fs, txt, fs)


def moments(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """Les moments sur la cornière, le principe du classeur (voir l'en-tête du module)."""
    if res.get("diagrammes") is None:
        raise ValueError("pas de diagrammes dans ce résultat (calculer(…, diagrammes=False))")
    g, sec, cfg, mode = res["geometrie"], res["section"], res["configuration"], res["mode"]
    e, r, larg, haut = g["e"], g["r"], g["larg"], g["haut"]
    r2 = float(sec.get("r2") or 0.0)
    X, dx, Y, dy, d0 = g["X"], g["dx"], g["Y"], g["dy"], g["d0"]
    A = cfg == "A"
    k = -1.0 if A else 1.0

    def yv(y: float) -> float:
        return k * y

    fs = float(fs) if fs else (larg + haut) / 34.0
    _lisible(haut, fs, "la hauteur de la cornière")
    dgs, eff = res["diagrammes"], res["efforts"]
    verifs = {v["cle"]: v for v in res["verifications"]}
    Mmax = abs(eff["Mmax"])
    M_ref = max([Mmax] + [abs(p["M"]) for dg in dgs.values() for p in dg["points"]]
                + [abs(p[1]) for dg in dgs.values() for c in dg["courbes"] for p in c["points"]])
    ech = PART_MOMENT * haut / M_ref if M_ref > 1e-12 else 0.0      # mm de dessin par kN·m, commune aux deux ailes
    s: List[Dict] = []
    # --- la cornière, claire, et les axes des ailes : la ligne M = 0 de chaque diagramme
    s.append(_poly(["ps"], _contour(larg, haut, e, r, r2, yv)))
    y_ax, x_ax = yv(e / 2.0), e / 2.0
    s.append(_path(["m0"], [[(0.0, y_ax), (larg, y_ax)], [(x_ax, yv(0.0)), (x_ax, yv(haut))]]))
    s.append(_text(["tx2"], larg + 0.5 * fs, y_ax + 0.3 * fs, "M = 0", 0.85 * fs, "start"))
    s.append(_text(["tx2"], x_ax, yv(haut) + (-0.6 * fs if A else 1.3 * fs), "M = 0", 0.85 * fs))

    def pt_h(sv: float, M: float) -> Tuple[float, float]:
        """Un point du diagramme de l'aile horizontale : porté hors de l'angle, depuis l'axe de l'aile."""
        return (sv, y_ax - k * abs(M) * ech)

    def pt_v(sv: float, M: float) -> Tuple[float, float]:
        """Un point du diagramme de l'aile verticale : porté côté support, depuis l'axe de l'aile."""
        return (x_ax - abs(M) * ech, yv(sv))

    for aile, pt in ((HORIZONTALE, pt_h), (VERTICALE, pt_v)):
        dg = dgs[aile]
        pts = dg["points"]
        # la zone entre l'axe et l'enveloppe, remplie léger ; les courbes (extrêmes fines, nominale moyenne) ; l'enveloppe
        s.append(_poly(["envf"], [pt(pts[0]["s"], 0.0)] + [pt(p["s"], p["M"]) for p in pts] + [pt(pts[-1]["s"], 0.0)]))
        for c in sorted(dg["courbes"], key=lambda c: abs(c["jeu_charge"]), reverse=True):
            s.append(_path(["cvm" if abs(c["jeu_charge"]) < 1e-9 else "cvf"], [[pt(a, b) for a, b in c["points"]]]))
        s.append(_path(["env"], [[pt(p["s"], p["M"]) for p in pts]]))
    # --- le prolongement dans le bloc d'angle (hors modèle), en pointillé : l'aile horizontale jusqu'à Mmax au nu du
    #     support (M angle à l'axe de l'aile verticale en B) ; l'aile verticale jusqu'au pivot du classeur (historique :
    #     Mmax à e/2) ou le moment constant de B
    M_e_h = dgs[HORIZONTALE]["points"][0]["M"]
    M_e_v = dgs[VERTICALE]["points"][0]["M"]
    if ech > 0:
        s.append(_path(["envp"], [[pt_h(e, M_e_h), pt_h(0.0, Mmax)]]))
        if mode == HISTORIQUE:
            s.append(_path(["envp"], [[pt_v(e, M_e_v), pt_v(e / 2.0, Mmax)]]))
        elif not A:
            s.append(_path(["envp"], [[pt_v(e, M_e_v), pt_v(e / 2.0, eff["M_angle"])]]))
    # --- la charge R en petite flèche à X (ses positions X ± dx en tirets) ; l'axe de l'ancrage, sa bande ± dy, le contact (B)
    y_face = yv(e) if A else yv(0.0)
    _fleche_cls(s, (X, y_face - 2.6 * fs), (X, y_face), "fz", "fzh", fs)
    for xj in ((X - dx, X + dx) if dx > 0 else ()):
        _fleche_cls(s, (xj, y_face - 1.7 * fs), (xj, y_face), "fzd", "fzh", fs)
    _label(s, X + 0.9 * fs + 0.3 * fs, y_face - 2.6 * fs + 0.34 * fs, "R", fs)
    ya = yv(Y)
    if dy > 0:
        s.append(_rect(["tol"], 0.0, min(yv(Y - dy), yv(Y + dy)), e, 2 * dy))
        s.append(_path(["hd"], [[(-1.2 * fs, yv(Y - dy)), (e + 1.2 * fs, yv(Y - dy))], [(-1.2 * fs, yv(Y + dy)), (e + 1.2 * fs, yv(Y + dy))]]))
    s.append(_path(["ax"], [[(-2.0 * fs, ya), (e + 2.0 * fs, ya)]]))
    s.append(_path(["fl2"], [[(0.0, yv(Y - d0 / 2)), (e, yv(Y - d0 / 2))], [(0.0, yv(Y + d0 / 2)), (e, yv(Y + d0 / 2))]]))
    s.append(_text(["tx2"], e + 2.5 * fs, ya + 0.3 * fs, f"Y = {f0(Y)}" + (f" ± {f0(dy)}" if dy > 0 else ""), 0.85 * fs, "start"))
    if not A:
        s.append(_circle(["fcp"], 0.0, yv(haut - g["yc"]), 0.3 * fs))
    # --- les sections 1 à 4 : le trait à travers l'aile, prolongé en tirets jusqu'à l'enveloppe, la pastille au-delà et
    #     la valeur de l'enveloppe (M1 à M4) ; deux étiquettes qui se croiseraient s'écartent
    secs = {v["cle"]: v for v in g["sections"]}
    prec = None
    for cle, num in (("S1", 1), ("S2", 2)):                       # du plus petit moment au plus grand : vers l'extérieur
        sv, M = secs[cle]["s"], verifs[cle]["M"]
        x_e, y_e = pt_h(sv, M)
        y_p = y_e - k * 1.5 * fs
        if prec is not None and (y_p - prec) * (-k) < 1.6 * fs:
            y_p = prec - k * 1.6 * fs
        prec = y_p
        s.append(_path(["sc"], [[(sv, yv(0.0)), (sv, yv(e))]]))
        s.append(_path(["d1"], [[(sv, yv(0.0)), (sv, y_p + k * 0.8 * fs)]]))
        _etiquette_pastille(s, sv, y_p, num, f"M{num} = {_knm(M)}", fs, +1)
    x_prec, w_prec = None, 0.0
    for cle, num in (("S3", 3), ("S4", 4)):
        sv, M = secs[cle]["s"], verifs[cle]["M"]
        x_e, y_e = pt_v(sv, M)
        x_p = x_e - 1.5 * fs
        txt = f"M{num} = {_knm(M)}"
        if not A and num == 4:
            txt = f"M4 = M angle = {_knm(eff['M_angle'])}" if abs(abs(M) - abs(eff["M_angle"])) < 1e-9 else txt
        if x_prec is not None and abs(y_e - yv(secs["S3"]["s"])) < 1.45 * fs:
            x_p = min(x_p, x_prec - 1.5 * fs - w_prec - 0.9 * fs)
        x_prec, w_prec = x_p, _tw(txt, fs)
        s.append(_path(["sc"], [[(0.0, yv(sv)), (secs[cle]["t"], yv(sv))]]))
        s.append(_path(["d1"], [[(0.0, yv(sv)), (x_p + 0.8 * fs, yv(sv))]]))
        _etiquette_pastille(s, x_p, yv(sv), num, txt, fs, -1)
    # Mmax au nu du support (le bout du prolongement de l'aile horizontale)
    if ech > 0:
        xm, ym = pt_h(0.0, Mmax)
        s.append(_circle(["gvd"], xm, ym, 0.22 * fs))
        txt = f"Mmax = {_knm(Mmax)}"
        _label(s, xm - 0.6 * fs - _tw(txt, fs) / 2, ym - k * 0.2 * fs + 0.34 * fs, txt, fs)
    aria = (f"Moments sur la cornière {sec['designation']} (configuration {cfg}) : le diagramme de chaque aile porté perpendiculairement "
            f"à l'aile depuis son axe, une courbe par position de la charge, l'enveloppe en gras, M1 à M4 et Mmax = {_knm(Mmax)} kN·m")
    return _dessin(s, fs, aria)


# ====================================================================== les contraintes : le maillage de von Mises
def contraintes(res: Dict, fs: Optional[float] = None, interactif: bool = True) -> Dessin:
    """Le champ de von Mises de l'enveloppe, maille par maille (voir l'en-tête du module)."""
    if res.get("maillage") is None or res.get("diagrammes") is None:
        raise ValueError("pas de maillage dans ce résultat (calculer(…, diagrammes=False))")
    g, sec, cfg, mat = res["geometrie"], res["section"], res["configuration"], res["materiau"]
    e, r, larg, haut = g["e"], g["r"], g["larg"], g["haut"]
    Y, dy = g["Y"], g["dy"]
    A = cfg == "A"
    k = -1.0 if A else 1.0

    def yv(y: float) -> float:
        return k * y

    fs = float(fs) if fs else (larg + haut) / 34.0
    _lisible(haut, fs, "la hauteur de la cornière")
    mz = res["maillage"]
    s: List[Dict] = []
    # --- le bloc d'angle (s < e) : hors modèle de poutre, hachuré gris clair
    bloc = [(0.0, yv(0.0)), (e, yv(0.0)), (e, yv(e)), (0.0, yv(e))]
    s += [_poly(["bg"], bloc), _hach(["ht"], bloc, 0.3 * fs)]
    # --- les mailles : (s_i, s_{i+1}) × (ζ_j, ζ_{j+1}) sur la vraie géométrie, colorées par le plus grand taux de leurs
    #     quatre nœuds (le durable garde le défavorable). Dans la matière du congé, les deux modèles de poutre se
    #     recouvrent : l'aile la plus sollicitée est dessinée en dernier, c'est elle qu'on voit
    ordre = sorted((HORIZONTALE, VERTICALE), key=lambda a: res["maxima"][a]["taux"])
    for aile in ordre:
        ch = mz[aile]
        S_, T_, Z_, TX = ch["s"], ch["t"], ch["zeta"], ch["taux"]

        def noeud(i: int, j: int, aile=aile, S_=S_, T_=T_, Z_=Z_) -> Tuple[float, float]:
            u = (Z_[j] + 1.0) / 2.0 * T_[i]                       # la distance à la face extérieure
            return (S_[i], yv(u)) if aile == HORIZONTALE else (u, yv(S_[i]))

        for i in range(len(S_) - 1):
            for j in range(len(Z_) - 1):
                t = max(TX[i][j], TX[i + 1][j], TX[i][j + 1], TX[i + 1][j + 1])
                p = _poly(["mz"], [noeud(i, j), noeud(i + 1, j), noeud(i + 1, j + 1), noeud(i, j + 1)])
                p["fill"] = couleur_taux(t)
                s.append(p)
    # --- le contour par-dessus (bouts droits : la géométrie du modèle)
    s.append(_poly(["ctr"], _contour(larg, haut, e, r, 0.0, yv)))
    # --- les sections 1 à 4, pastilles hors de la pièce (1, 2 côté face extérieure de l'aile horizontale ; 3, 4 côté
    #     support) ; congé étroit : la pastille de la section au nu est décalée d'un cran
    etag = r < 1.9 * fs
    y1p, y2p = yv(-2.2 * fs), yv(-3.9 * fs if etag else -2.2 * fs)
    x3p, x4p = -2.2 * fs, -(3.9 * fs if etag else 2.2 * fs)
    s += [_path(["sc"], [[(e + r, yv(e)), (e + r, yv(-1.4 * fs))]]),
          _path(["sc"], [[(e, yv(e + r)), (e, yv(-3.1 * fs if etag else -1.4 * fs))]]),
          _path(["sc"], [[(e, yv(e + r)), (-1.4 * fs, yv(e + r))]]),
          _path(["sc"], [[(e + r, yv(e)), (-(3.1 * fs if etag else 1.4 * fs), yv(e))]]),
          _pastille(e + r, y1p, 1, fs), _pastille(e, y2p, 2, fs), _pastille(x3p, yv(e + r), 3, fs), _pastille(x4p, yv(e), 4, fs)]
    # --- l'axe de l'ancrage et ses positions ± dy
    ya = yv(Y)
    s.append(_path(["ax"], [[(-1.5 * fs, ya), (e + 1.5 * fs, ya)]]))
    if dy > 0:
        s.append(_path(["hd"], [[(-0.6 * fs, yv(Y - dy)), (e + 0.6 * fs, yv(Y - dy))], [(-0.6 * fs, yv(Y + dy)), (e + 0.6 * fs, yv(Y + dy))]]))
    # --- le maximum ◉ de chaque aile : à la position de la VÉRIFICATION, sur la face ζ du maillage ; l'étiquette hors de
    #     la pièce, reliée par un renvoi
    x_gauche = x4p - 1.2 * fs                       # le bord gauche des pastilles 3 et 4
    for aile in (HORIZONTALE, VERTICALE):
        mx = res["maxima"][aile]
        z = mz[aile]["max"]["zeta"]
        u = (z + 1.0) / 2.0 * mx["t"]
        xm, ym = (mx["s"], yv(u)) if aile == HORIZONTALE else (u, yv(mx["s"]))
        txt = f"{_pct(mx['taux'])} · s = {_num(mx['s'], 1)}"
        w = _tw(txt, fs)
        if aile == HORIZONTALE:
            xl, yl = mx["s"] + 1.2 * fs + w / 2, yv(-4.9 * fs if etag else -3.9 * fs) - k * 1.5 * fs
            att = (xl - w / 2 - 0.2 * fs, yl - k * 0.0)
        else:
            xl, yl = x_gauche - 0.4 * fs - w / 2, yv(mx["s"] + 2.6 * fs)
            att = (xl + w / 2 + 0.2 * fs, yl)
        s.append(_path(["ext"], [[(xm, ym), att]]))
        s += [_circle(["gv"], xm, ym, 0.5 * fs), _circle(["gvd"], xm, ym, 0.2 * fs)]
        _label(s, xl, yl + 0.34 * fs, txt, fs)
    # --- la barre de couleur : 0 → 100 % (vert) sur 70 % de sa hauteur, 100 → 105 % (orange) sur 15 %, au-delà (rouge)
    xb, wb = larg + 2.2 * fs, 1.0 * fs
    y_b0, y_b1 = max(yv(0.0), yv(0.85 * haut)), min(yv(0.0), yv(0.85 * haut))
    hb = y_b0 - y_b1
    degres = [(i / 20.0, (i + 1) / 20.0, 0.70) for i in range(20)]            # vert : 20 degrés
    for i, (t0, t1, part) in enumerate(degres):
        p = _rect(["mz"], xb, y_b0 - t1 * part * hb, wb, (t1 - t0) * part * hb + 0.02 * fs)
        p["fill"] = couleur_taux((t0 + t1) / 2.0)
        s.append(p)
    for i in range(5):                                                         # orange : 100 → 105 %, cinq degrés
        t0 = 1.0 + 0.01 * i
        p = _rect(["mz"], xb, y_b0 - (0.70 + 0.03 * (i + 1)) * hb, wb, 0.03 * hb + 0.02 * fs)
        p["fill"] = couleur_taux(t0 + 0.005)
        s.append(p)
    p = _rect(["mz"], xb, y_b1, wb, 0.15 * hb)                                 # rouge : au-delà de 105 %
    p["fill"] = couleur_taux(2.0)
    s.append(p)
    s.append(_rect(["ctr"], xb, y_b1, wb, hb))
    for part, txt in ((0.0, "0"), (0.35, "50"), (0.70, "100 %"), (0.85, "105 %")):
        y = y_b0 - part * hb
        s.append(_path(["axe"], [[(xb + wb, y), (xb + wb + 0.35 * fs, y)]]))
        s.append(_text(["tx2"], xb + wb + 0.55 * fs, y + 0.3 * fs, txt, 0.85 * fs, "start"))
    s.append(_text(["tx2"], xb + wb / 2, y_b1 - 0.9 * fs, f"taux σéq / {_symbole_sigma(mat)}", 0.85 * fs))
    # --- ce que c'est, en deux lignes sous le dessin : des contraintes de poutre, l'enveloppe des positions, pas un
    #     calcul par éléments finis
    ex = _extent(s, fs) or (0.0, 0.0, 1.0, 1.0)
    x0, y_bas = ex[0], ex[3]
    s.append(_text(["tx2"], x0, y_bas + 1.9 * fs, "σ = M·ζ/W + N/A, τ = V/A uniforme (poutre, classeur) · enveloppe des positions", 0.85 * fs, "start"))
    s.append(_text(["tx2"], x0, y_bas + 3.2 * fs, "pas un calcul par éléments finis", 0.85 * fs, "start"))
    aria = (f"Contraintes de von Mises sur la cornière {sec['designation']} (configuration {cfg}) : le maillage de l'enveloppe, chaque "
            f"maille colorée par son taux (vert sous 100 %, orange de 100 à 105 %, rouge au-delà), le maximum de chaque aile, les "
            f"sections 1 à 4, la barre de couleur — contraintes de poutre, pas un calcul par éléments finis")
    return _dessin(s, fs, aria)


# ====================================================================== l'entrée : vues, légendes, ajustement, secours
FONCTIONS: Dict[str, Callable[..., Dessin]] = {"coupe": coupe, "moments": moments, "contraintes": contraintes, "elevation": elevation}


def legende(res: Dict, cle: str) -> str:
    """Ce que montre une vue, l'état limite et le mode — UNE ligne sous le titre, partagée par l'écran et la note."""
    g, cfg, mode, mat, anc = res["geometrie"], res["configuration"], res["mode"], res["materiau"], res["ancrages"]
    etat = mat["etat_limite"]
    conf = f"configuration {cfg} ({LIBELLES_CONFIGURATION[cfg]})"
    if cle == "coupe":
        bras = ("bras du classeur Xe, Xe+r, Ye, Ye+r, YH" if mode == HISTORIQUE else
                ("bras X − uV et Y − yc" if cfg == "A" else "bras X − uV et ya − c"))
        jeux = (f"charge à X ± {f0(g['dx'])} (tirets)" if g["dx"] > 0 else "charge à X") + \
               (f", ancrage à Y ± {f0(g['dy'])} (tirets fins)" if g["dy"] > 0 else ", ancrage à Y")
        anc_txt = f"{LIBELLES_TYPE_ANCRAGE[g['type_ancrage']]} Ø{f0(g['d'])}, profondeur {f0(g['profondeur'])}"
        return (f"{conf} · {LIBELLES_MODE[mode]} · {etat} : R = {_kn(res['efforts']['R'])} · {bras} · {jeux} · "
                f"{LIBELLES_SUPPORT[g['support']]}, {anc_txt} · sections 1 à 4 en orange · C la résultante de compression")
    if cle == "moments":
        signe = ("M > 0 : face du congé tendue" if cfg == "A" else
                 "M < 0 (face du congé comprimée), constant de l'ancrage à l'angle (M angle)")
        return (f"{conf} · {LIBELLES_MODE[mode]} · {etat} · le diagramme de chaque aile porté hors de l'angle depuis son axe (M = 0) : "
                f"une courbe par position de la charge (extrêmes en trait fin), l'enveloppe en gras et remplie, M1 à M4 et Mmax · {signe}")
    if cle == "contraintes":
        return (f"{conf} · {LIBELLES_MODE[mode]} · {etat} · taux σéq / {_symbole_sigma(mat)} = {_num(mat['sigma_adm'], 1)} N/mm² par "
                f"maille, enveloppe des positions : contraintes de poutre σ = M·ζ/W + N/A, τ = V/A uniforme (comme le classeur) — "
                f"pas un calcul par éléments finis · ◉ le maximum de chaque aile, vert sous 100 %, orange de 100 à 105 %, rouge au-delà")
    a = f"a = {f0(g['a'])}" if g["a"] is not None else "a = L"
    coupe_txt = " · tranche coupée au dessin, cotes vraies" if g["L"] > COUPURE_TRANCHE * g["haut"] else ""
    return (f"{conf} · {etat} : N et V par ancrage (enveloppe) · la tranche L = {f0(g['L'])} vue de face : {g['n']} ancrage(s) à "
            f"l'entraxe {f0(anc['entraxe'])}, e₁ = {f0(anc['distance_bout'])} au bout · l'aile horizontale rabattue : la charge ({a}) et "
            f"sa diffusion à 45° jusqu'à la section 1, L eff = {f0(g['L_eff'])}{coupe_txt}")


def ajuster(construire: Callable[[float], Dessin], toile: Tuple[float, float], texte_px: float, fs0: float, passes: int = 12) -> Dessin:
    """Une vue reconstruite jusqu'à ce que son texte se lise à ``texte_px`` pixels dans la toile ``(largeur, hauteur)``
    en pixels. La taille du texte à l'écran croît avec ``fs`` mais de moins en moins vite (une partie de la boîte de vue
    est en tailles de texte) : on corrige du rapport manquant, en accélérant quand le texte est trop grand et que la
    boîte suit à peine (le maillage et ses mentions), puis on interpole entre deux passes qui encadrent la cible ; on
    s'arrête quand le texte est trop petit et ne grandit plus (toile trop petite pour la cible : la mise en page est
    en tailles de texte). La meilleure passe est gardée."""
    lo = hi = None                                 # (fs, px) de part et d'autre de la cible
    fs, expo, px_prec = fs0, 1.0, None
    meilleur: Optional[Tuple[float, Dessin]] = None
    for _ in range(passes):
        d = construire(fs)
        _, _, w, h = d.viewbox
        px = fs * min(toile[0] / w, toile[1] / h)
        if meilleur is None or abs(px - texte_px) < meilleur[0]:
            meilleur = (abs(px - texte_px), d)
        if abs(px - texte_px) < 0.25:
            break
        if px < texte_px:
            lo = (fs, px)
            if px_prec is not None and abs(px - px_prec) < 0.02 * texte_px:
                break                              # le texte ne grandit plus : rien à gagner
        else:
            hi = (fs, px)
            if px_prec is not None and abs(px - px_prec) < 0.5 * abs(texte_px - px_prec):
                expo = min(expo * 2.0, 8.0)
        if lo and hi:
            t = min(0.9, max(0.1, (texte_px - lo[1]) / (hi[1] - lo[1])))
            fs = math.exp(math.log(lo[0]) + (math.log(hi[0]) - math.log(lo[0])) * t)
        else:
            fs *= (texte_px / px) ** expo
        px_prec = px
    return meilleur[1]


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

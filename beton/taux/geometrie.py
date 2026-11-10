# -*- coding: utf-8 -*-
"""Taux d'armature — la GÉOMÉTRIE : la section (l'âme et ses talons BAS, à l'extérieur de l'âme), son aire, son
polygone ; les contours des cadres ; et, pour la coupe, la position de chaque barre, de chaque cadre et de chaque
épingle.

Repère d'une coupe (cm) : l'origine au coin bas gauche du rectangle qui ENGLOBE la section (le bord extérieur du talon
gauche, la fibre inférieure) ; x vers la droite, y vers le haut. L'âme occupe donc x de ``bg`` à ``bg + b``.

La convention des positions est celle de la coupe des poutres (beton/poutre/geometrie.py, dont les fonctions sont
REPRISES, pas recopiées) : le nu extérieur du cadre à l'enrobage, les barres d'un lit réparties entre les nus
intérieurs du cadre, la barre d'angle au contact du cadre — axe à c + Ø étrier + Ø/2 du parement. Deux lits d'une face
sont séparés de l'espacement libre minimal de l'EN 1992-1-1 §8.2(2), ``beton.poutre.dispositions.s_min`` (une seule
source pour la règle). Ces positions servent au DESSIN : aucune quantité du bordereau n'en dépend.
"""
from beton.poutre import dispositions as DP
from beton.poutre import entree as EP
from beton.poutre import geometrie as GP

from . import entree as EN

# le plus gros granulat pris pour l'espacement entre deux lits : le défaut des poutres (20 mm) — une seule source
D_GRANULAT = EP.DEFAUTS_DETAIL["dg"]


# ============================================================ la section
def section(p: dict) -> dict:
    """La section d'une poutre NORMALISÉE : ``b``, ``h``, ``c`` (cm), les talons que la forme porte (``bg``, ``hg``,
    ``bd``, ``hd`` : 0 sans talon), la largeur hors tout au niveau des talons ``b_tot``, la hauteur du plus haut talon
    ``h_talon``, l'aire ``aire_cm2`` et le contour ``polygone`` ([[x, y], …], cm)."""
    b, h = p["b"], p["h"]
    portes = EN.talons(p["forme"])
    bg, hg = (p["talon_gauche"]["b"], p["talon_gauche"]["h"]) if "talon_gauche" in portes else (0.0, 0.0)
    bd, hd = (p["talon_droit"]["b"], p["talon_droit"]["h"]) if "talon_droit" in portes else (0.0, 0.0)
    b_tot = bg + b + bd
    pts = [[0.0, 0.0], [b_tot, 0.0]]
    if bd:
        pts += [[b_tot, hd], [bg + b, hd]]
    pts += [[bg + b, h], [bg, h]]
    if bg:
        pts += [[bg, hg], [0.0, hg]]
    return {"forme": p["forme"], "b": b, "h": h, "c": p["enrobage"], "bg": bg, "hg": hg, "bd": bd, "hd": hd, "b_tot": b_tot,
            "h_talon": max(hg, hd), "aire_cm2": b * h + bg * hg + bd * hd, "polygone": pts}


def aire_polygone(pts: list) -> float:
    """L'aire d'un polygone (formule du lacet) : le contrôle de ``section`` par un autre chemin."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


# ============================================================ les contours comptés (hors tout, à l'enrobage)
def contour_cadre(sec: dict) -> tuple:
    """(largeur, hauteur) hors tout du cadre de l'âme, à l'enrobage (cm) : b − 2c, h − 2c."""
    return sec["b"] - 2.0 * sec["c"], sec["h"] - 2.0 * sec["c"]


def contour_talon(sec: dict) -> tuple:
    """(largeur, hauteur) hors tout du cadre de talon (cm) : b_tot − 2c, h_t − 2c — b_tot la largeur hors tout au niveau
    des talons, h_t la hauteur du plus HAUT talon (deux talons inégaux : le côté qui commande le plus d'acier)."""
    return sec["b_tot"] - 2.0 * sec["c"], sec["h_talon"] - 2.0 * sec["c"]


# ============================================================ les positions, pour la coupe et l'élévation
def diametre_cadre(type_: dict = None) -> float:
    """Le Ø (mm) du cadre qui place les barres d'une coupe : le plus gros des lignes du type (0 sans type)."""
    return float(max((li["d"] for li in type_["lignes"]), default=0)) if type_ else 0.0


def diametre_cadres(p: dict) -> float:
    """Le Ø (mm) qui place les barres de l'ÉLÉVATION : le plus gros de tous les types de la poutre."""
    return max((diametre_cadre(t) for t in p["types_etriers"]), default=0.0)


def diametre_talon(type_: dict = None) -> int:
    """Le Ø (mm) du cadre de talon d'un type : celui de son premier cadre, à défaut de sa première ligne (0 sans type)."""
    if not type_:
        return 0
    return int(next((li["d"] for li in type_["lignes"] if li["forme"] == "cadre"), type_["lignes"][0]["d"]))


def niveaux(p: dict, d_cadre: float) -> list:
    """L'ordonnée de l'axe de chaque ligne de barres (cm, depuis la fibre inférieure), dans l'ordre de ``p["barres"]`` ;
    None pour la peau. Sur une face, les lits sont pris dans l'ordre de leur rang : le premier repose sur le cadre (son
    nu inférieur à c + Ø cadre), chaque lit suivant est posé au-dessus du plus gros Ø du précédent, à l'espacement libre
    minimal de l'EN 1992-1-1 §8.2(2) ; l'axe d'une barre est à Ø/2 du nu de son lit."""
    c, h = p["enrobage"], p["h"]
    out = [None] * len(p["barres"])
    for face in ("inf", "sup"):
        lignes = [(i, b) for i, b in enumerate(p["barres"]) if b["famille"] == face]
        rangs = sorted({b["lit"] for _, b in lignes})
        nu = c + d_cadre / 10.0                       # le nu du lit, côté parement (cm depuis la face)
        for k, rang in enumerate(rangs):
            du_lit = [(i, b) for i, b in lignes if b["lit"] == rang]
            d_max = max(b["d"] for _, b in du_lit)
            for i, b in du_lit:
                e = nu + b["d"] / 20.0
                out[i] = e if face == "inf" else h - e
            if k + 1 < len(rangs):
                d_suiv = max(b["d"] for _, b in lignes if b["lit"] == rangs[k + 1])
                nu += (d_max + DP.s_min(d_max, d_suiv, D_GRANULAT)) / 10.0
    return out


def _creneaux(n: int, k: int) -> list:
    """Les rangs que prennent ``k`` barres parmi ``n`` places alignées : les angles d'abord, puis réparties (une seule :
    au milieu)."""
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return [round(j * (n - 1) / (k - 1)) for j in range(k)]


def _repartir(lignes: list, xs: list) -> dict:
    """Les abscisses de chaque ligne d'un lit : ``lignes`` = [(index, n)], ``xs`` = les places du lit (cm). La première
    ligne prend les angles et se répartit ; les suivantes remplissent les places libres, de gauche à droite."""
    libres = list(range(len(xs)))
    out = {}
    for rang, (i, n) in enumerate(lignes):
        pris = [libres[j] for j in _creneaux(len(libres), n)] if rang == 0 else libres[:n]
        out[i] = [xs[j] for j in pris]
        libres = [j for j in libres if j not in pris]
    return out


def coupe(p: dict, presentes: list, type_: dict = None) -> dict:
    """Ce qu'on voit dans une coupe : les barres des lignes ``presentes`` (indices de ``p["barres"]``) et les aciers du
    type d'étriers ``type_`` (None : une coupe sans étriers).

    Rend {"section", "d_cadre" (mm),
          "barres": [{"index", "famille", "lit", "d", "points": [[x, y], …]}],
          "cadres": [{"sorte": "cadre" | "cadre_interieur" | "talon", "d", "x0", "x1", "y0", "y1"}]  (nus extérieurs),
          "epingles": [{"d", "x", "y0", "y1"}]}."""
    sec = section(p)
    c, b, h, bg = sec["c"], sec["b"], sec["h"], sec["bg"]
    d_cadre = diametre_cadre(type_)
    a_talon = bool(type_ and type_["cadre_talon"] and EN.a_talon(p["forme"]))
    d_talon = float(diametre_talon(type_)) if a_talon else 0.0
    ys = niveaux(p, max(d_cadre, d_talon))
    barres = []
    for face in ("inf", "sup"):
        rangs = sorted({p["barres"][i]["lit"] for i in presentes if p["barres"][i]["famille"] == face})
        tous = sorted({x["lit"] for x in p["barres"] if x["famille"] == face})
        for rang in rangs:
            du_lit = [i for i in presentes if p["barres"][i]["famille"] == face and p["barres"][i]["lit"] == rang]
            # les filantes d'abord : elles tiennent les angles
            du_lit.sort(key=lambda i: (not EN.filante(p["barres"][i]), i))
            n_tot = sum(p["barres"][i]["n"] for i in du_lit)
            d_max = max(p["barres"][i]["d"] for i in du_lit)
            if face == "inf" and a_talon and rang == tous[0]:
                # le premier lit inférieur s'étale dans le cadre de talon, sur toute la largeur
                xs = GP.abscisses_lit({"n": n_tot, "d": d_max}, sec["b_tot"], c, d_talon)
            else:
                xs = [bg + x for x in GP.abscisses_lit({"n": n_tot, "d": d_max}, b, c, d_cadre)]
            places = _repartir([(i, p["barres"][i]["n"]) for i in du_lit], xs)
            for i in du_lit:
                x_ = p["barres"][i]
                barres.append({"index": i, "famille": face, "lit": rang, "d": x_["d"], "points": [[x, ys[i]] for x in places[i]]})
    # la peau : n barres par face, réparties entre les premiers lits des deux faces, au contact des montants du cadre
    peau = [i for i in presentes if p["barres"][i]["famille"] == "peau"]
    if peau:
        bas = min((y for i, y in enumerate(ys) if y is not None and p["barres"][i]["famille"] == "inf"), default=c + d_cadre / 10.0)
        haut = max((y for i, y in enumerate(ys) if y is not None and p["barres"][i]["famille"] == "sup"), default=h - c - d_cadre / 10.0)
        n_tot = sum(p["barres"][i]["n"] for i in peau)
        niv = [bas + k * (haut - bas) / (n_tot + 1) for k in range(1, n_tot + 1)]
        k0 = 0
        for i in peau:
            x_ = p["barres"][i]
            xg = bg + c + d_cadre / 10.0 + x_["d"] / 20.0
            mes = niv[k0:k0 + x_["n"]]
            k0 += x_["n"]
            barres.append({"index": i, "famille": "peau", "lit": None, "d": x_["d"],
                           "points": [[x, y] for y in mes for x in (xg, bg + b - (xg - bg))]})
    barres.sort(key=lambda x: x["index"])
    cadres, epingles = [], []
    if type_:
        x0, x1 = bg + c, bg + b - c
        interieurs = [li for li in type_["lignes"] if li["forme"] == "cadre_interieur"]
        pinces = [li for li in type_["lignes"] if li["forme"] == "epingle"]
        for li in type_["lignes"]:
            if li["forme"] == "cadre":
                cadres.append({"sorte": "cadre", "d": li["d"], "x0": x0, "x1": x1, "y0": c, "y1": h - c})
        for k, li in enumerate(interieurs):
            w = li["largeur"]
            xc = x0 + (x1 - x0) * (k + 1) / (len(interieurs) + 1)
            xc = min(max(xc, x0 + w / 2.0), x1 - w / 2.0)
            cadres.append({"sorte": "cadre_interieur", "d": li["d"], "x0": xc - w / 2.0, "x1": xc + w / 2.0, "y0": c, "y1": h - c})
        # une épingle tient une barre : elle se pose sur une barre intérieure du premier lit inférieur de l'âme quand
        # il y en a une près de sa place, sinon elle partage l'intervalle entre les montants
        rang1 = min((bb["lit"] for bb in barres if bb["famille"] == "inf"), default=None)
        inf1 = sorted(x for bb in barres if bb["famille"] == "inf" and bb["lit"] == rang1 for x, _ in bb["points"] if x0 < x < x1)
        interieures = inf1[1:-1] if len(inf1) >= 3 else []
        prises = set()
        for k, li in enumerate(pinces):
            x = x0 + (x1 - x0) * (k + 1) / (len(pinces) + 1)
            libres = [v for v in interieures if v not in prises]
            if libres:
                x = min(libres, key=lambda v: abs(v - x))
                prises.add(x)
            epingles.append({"d": li["d"], "x": x, "y0": c, "y1": h - c})
        if a_talon:
            w_t, h_t = contour_talon(sec)
            cadres.append({"sorte": "talon", "d": d_talon, "x0": c, "x1": c + w_t, "y0": c, "y1": c + h_t})
    return {"section": sec, "d_cadre": d_cadre, "barres": barres, "cadres": cadres, "epingles": epingles}

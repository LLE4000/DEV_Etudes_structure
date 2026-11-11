# -*- coding: utf-8 -*-
"""Fondations de grues — la GÉOMÉTRIE EN PLAN d'une fondation, une seule fois pour le dessin de l'écran, le plan de
principe de la note, le fichier DXF et l'export vers RBAR. Bibliothèque standard seulement.

Repère : origine au centre du châssis de la grue, x vers la droite, y vers le haut, en cm. Les pieds de la grue sont
aux quatre coins du châssis, nommés comme la fiche BASE les nomme (A, B, C, D en tournant) : A en haut à gauche, B en
haut à droite, C en bas à droite, D en bas à gauche — la poutre support du haut porte A (côté semelle) et B (côté b),
celle du bas D et C : ce sont les orientations « A/B » et « D/C » des grands tableaux, l'une « respective » de l'autre.

    plan(f) -> {"type", "chassis", "pieds": [{nom, x, y}], "pieux": [{nom, x, y, D}],
                "pieces": [{cle, nom, contour: [(x, y)…], axe: [(x, y), (x, y)], b, longueur}],
                "paroi": None | {x, y0, y1, D}, "cotes": [{a: (x, y), b: (x, y), texte, decalage}], "etendue"}

Ce que les classeurs ne fixent PAS et que ce plan de principe choisit, en le disant (``conventions``) : la longueur
dessinée d'une poutre au-delà de ses appuis, le côté du pied de la grue, l'étendue de la paroi de pieux sécants.
"""
import math

from . import entree as E

COTE_PIED = 40.0          # cm : côté de la platine d'un pied sur le plan de principe (non donné par les classeurs)
DEBORD_PAROI = 150.0      # cm : la paroi de pieux sécants est dessinée 1,50 m au-delà des poutres


def _rect(cx, cy, lx, ly):
    """Un rectangle d'axes parallèles au repère, centré en (cx, cy), de côtés lx (selon x) et ly (selon y)."""
    return [(cx - lx / 2, cy - ly / 2), (cx + lx / 2, cy - ly / 2), (cx + lx / 2, cy + ly / 2), (cx - lx / 2, cy + ly / 2)]


def _rect_oriente(cx, cy, longueur, largeur, ux, uy):
    """Un rectangle centré en (cx, cy) dont l'axe long suit le vecteur unitaire (ux, uy)."""
    vx, vy = -uy, ux
    L, B = longueur / 2, largeur / 2
    return [(cx - ux * L - vx * B, cy - uy * L - vy * B), (cx + ux * L - vx * B, cy + uy * L - vy * B),
            (cx + ux * L + vx * B, cy + uy * L + vy * B), (cx - ux * L + vx * B, cy - uy * L + vy * B)]


def _pieds(Lc):
    h = Lc / 2
    return [{"nom": "A", "x": -h, "y": h}, {"nom": "B", "x": h, "y": h}, {"nom": "C", "x": h, "y": -h},
            {"nom": "D", "x": -h, "y": -h}]


def _etendue(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _cadre4(f):
    L, po, D = f["chassis"], f["poutre"], f["pieux"]["D"]
    h = L / 2
    pieux = [{"nom": f"P{i + 1}", "x": p["x"], "y": p["y"], "D": D} for i, p in enumerate(_pieds(L))]
    pieces = []
    for i, (cx, cy, ux, uy) in enumerate(((0, h, 1, 0), (h, 0, 0, 1), (0, -h, 1, 0), (-h, 0, 0, 1))):
        pieces.append({"cle": "poutre", "nom": f"Poutre {i + 1}", "b": po["b"], "longueur": L,
                       "contour": _rect_oriente(cx, cy, L, po["b"], ux, uy),
                       "axe": [(cx - ux * h, cy - uy * h), (cx + ux * h, cy + uy * h)]})
    cotes = [{"a": (-h, h), "b": (h, h), "texte": f"châssis {L:g}", "decalage": po["b"] / 2 + D / 2 + 45},
             {"a": (h, -h), "b": (h, h), "texte": f"{L:g}", "decalage": -(po["b"] / 2 + D / 2 + 45)}]
    return {"pieds": _pieds(L), "pieux": pieux, "pieces": pieces, "paroi": None, "cotes": cotes,
            "conventions": ["les poutres sont dessinées d'axe de pieu à axe de pieu (longueur = largeur du châssis)"]}


def _diagonales8(f):
    L, se, pr, pi_ = f["chassis"], f["secondaire"], f["primaire"], f["pieux"]
    h, s, D = L / 2, pi_["entraxe"], pi_["D"]
    r2 = math.sqrt(2) / 2
    pieux, pieces = [], []
    for i, (cx, cy, ux, uy) in enumerate(((0, h, 1, 0), (h, 0, 0, 1), (0, -h, 1, 0), (-h, 0, 0, 1))):
        pieces.append({"cle": "secondaire", "nom": f"Poutre secondaire {i + 1}", "b": se["b"], "longueur": L,
                       "contour": _rect_oriente(cx, cy, L, se["b"], ux, uy),
                       "axe": [(cx - ux * h, cy - uy * h), (cx + ux * h, cy + uy * h)]})
    n = 0
    for i, p in enumerate(_pieds(L)):
        sx, sy = (1 if p["x"] > 0 else -1), (1 if p["y"] > 0 else -1)
        ux, uy = sx * r2, -sy * r2                       # l'axe de la primaire : perpendiculaire à la diagonale
        pieces.append({"cle": "primaire", "nom": f"Poutre primaire {i + 1}", "b": pr["b"], "longueur": pr["L"],
                       "contour": _rect_oriente(p["x"], p["y"], pr["L"], pr["b"], ux, uy),
                       "axe": [(p["x"] - ux * pr["L"] / 2, p["y"] - uy * pr["L"] / 2),
                               (p["x"] + ux * pr["L"] / 2, p["y"] + uy * pr["L"] / 2)]})
        for k in (-1, 1):
            n += 1
            pieux.append({"nom": f"P{n}", "x": p["x"] + k * ux * s / 2, "y": p["y"] + k * uy * s / 2, "D": D})
    cotes = [{"a": (-h, h), "b": (h, h), "texte": f"châssis {L:g}", "decalage": pr["L"] / 2 * r2 + pr["b"] + 40},
             {"a": (pieux[2]["x"], pieux[2]["y"]), "b": (pieux[3]["x"], pieux[3]["y"]), "texte": f"{s:g}",
              "decalage": pr["b"] / 2 + 45}]
    return {"pieds": _pieds(L), "pieux": pieux, "pieces": pieces, "paroi": None, "cotes": cotes,
            "conventions": ["les poutres primaires sont à 45°, centrées sous les angles du châssis",
                            "les poutres secondaires sont dessinées d'angle à angle (longueur = largeur du châssis)"]}


def _semelles8(f):
    Lc, a, b = f["chassis"], f["dist_a"], f["dist_b"]
    su, sm, pi_ = f["support"], f["semelle"], f["pieux"]
    h, s, D = Lc / 2, pi_["entraxe"], pi_["D"]
    xa, xb = -h - a, h + b
    paroi = f["cote_b"] == E.COTE_PAROI
    pieux, pieces, n = [], [], 0
    cotes_x = [(xa, True)] + ([] if paroi else [(xb, False)])
    for i, y in enumerate((h, -h)):
        x0, x1 = xa - sm["b"] / 2, (xb + (D / 2 if paroi else sm["b"] / 2))
        pieces.append({"cle": "support", "nom": f"Poutre support {i + 1}", "b": su["b"], "longueur": x1 - x0,
                       "portee": xb - xa, "contour": _rect((x0 + x1) / 2, y, x1 - x0, su["b"]), "axe": [(xa, y), (xb, y)]})
    k = 0
    for x, _ in cotes_x:
        for y in (h, -h):
            k += 1
            pieces.append({"cle": "semelle", "nom": f"Semelle {k}", "b": sm["b"], "longueur": sm["L"], "portee": s,
                           "contour": _rect(x, y, sm["b"], sm["L"]), "axe": [(x, y - sm["L"] / 2), (x, y + sm["L"] / 2)]})
            for sg in (1, -1):
                n += 1
                pieux.append({"nom": f"P{n}", "x": x, "y": y + sg * s / 2, "D": D})
    ybas = -h - sm["L"] / 2 - 60
    cotes = [{"a": (xa, ybas), "b": (-h, ybas), "texte": f"a = {a:g}", "decalage": 0},
             {"a": (-h, ybas), "b": (h, ybas), "texte": f"châssis {Lc:g}", "decalage": 0},
             {"a": (h, ybas), "b": (xb, ybas), "texte": f"b = {b:g}", "decalage": 0},
             {"a": (xa, ybas - 55), "b": (xb, ybas - 55), "texte": f"portée {xb - xa:g}", "decalage": 0},
             {"a": (xa - sm["b"] / 2 - 60, h - s / 2), "b": (xa - sm["b"] / 2 - 60, h + s / 2), "texte": f"{s:g}", "decalage": 0},
             {"a": (xa - sm["b"] / 2 - 115, h - sm["L"] / 2), "b": (xa - sm["b"] / 2 - 115, h + sm["L"] / 2),
              "texte": f"{sm['L']:g}", "decalage": 0},
             {"a": (xa - sm["b"] / 2 - 60, -h), "b": (xa - sm["b"] / 2 - 60, h), "texte": f"{Lc:g}", "decalage": 0}]
    conv = ["les poutres support sont dessinées jusqu'à la face extérieure des semelles (le calcul s'arrête aux axes)"]
    if paroi:
        conv.append("la paroi de pieux sécants est figurée sur 1,50 m de part et d'autre des poutres : seule sa position "
                    "(distance b) entre dans le calcul")
    return {"pieds": _pieds(Lc), "pieux": pieux, "pieces": pieces, "cotes": cotes, "conventions": conv,
            "paroi": {"x": xb, "y0": -h - DEBORD_PAROI, "y1": h + DEBORD_PAROI, "D": D} if paroi else None}


_PAR_TYPE = {E.CADRE4: _cadre4, E.DIAGONALES8: _diagonales8, E.SEMELLES8: _semelles8}


def _decaler(cote: dict) -> dict:
    """Une cote posée à ``decalage`` de la ligne qu'elle mesure (vers la gauche du sens a → b) ; le décalage est
    appliqué ICI, pour que l'étendue du plan contienne la cote telle qu'elle se dessine."""
    (ax, ay), (bx, by), d = cote["a"], cote["b"], cote.get("decalage", 0)
    L = math.hypot(bx - ax, by - ay) or 1.0
    nx, ny = -(by - ay) / L, (bx - ax) / L
    return {"a": (ax + nx * d, ay + ny * d), "b": (bx + nx * d, by + ny * d), "texte": cote["texte"]}


def plan(f: dict) -> dict:
    """Le plan de la fondation NORMALISÉE ``f`` (cm)."""
    g = _PAR_TYPE[f["type"]](f)
    g["cotes"] = [_decaler(c) for c in g["cotes"]]
    pts = [p for piece in g["pieces"] for p in piece["contour"]]
    pts += [(p["x"] + sx * p["D"] / 2, p["y"] + sy * p["D"] / 2) for p in g["pieux"] for sx in (-1, 1) for sy in (-1, 1)]
    pts += [c[k] for c in g["cotes"] for k in ("a", "b")]
    if g["paroi"]:
        pts += [(g["paroi"]["x"] + g["paroi"]["D"] / 2, g["paroi"]["y0"]), (g["paroi"]["x"], g["paroi"]["y1"])]
    g.update({"type": f["type"], "chassis": f["chassis"], "cote_pied": COTE_PIED, "etendue": _etendue(pts)})
    return g

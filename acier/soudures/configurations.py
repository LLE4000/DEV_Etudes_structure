# -*- coding: utf-8 -*-
"""Les configurations courantes : des dimensions → un groupe de cordons, les
efforts au bon point et dans le bon sens, et les pièces du dessin.

Repère de chaque cas (plan de joint z = 0) :
- C1 recouvrement : plan de joint = interface des deux plats ; x le long de l'effort N, pièce
  rapportée B sur 0 ≤ x ≤ L_rec, |y| ≤ b/2 ;
- C2 jonction en T : plan de joint = face de A ; la pièce B debout, d'épaisseur t_B, le long de x ;
- C3 console : plan de joint = face du poteau ; x horizontal dans la face, y vertical (haut > 0),
  z vers l'extérieur ; le plat de la console est perpendiculaire à la face, d'épaisseur t le long de x ;
- C4 gousset : plan de joint = face d'appui ; le plat b_x × b_y posé à plat, origine à son coin bas-gauche ;
- C5 platine d'about : plan de joint = face de la platine ; profilé centré, semelles le long de x
  (axe fort de l'EN 1993 = x), âme le long de y ;
- C6 tube sur platine : idem, tube centré ;
- C7 bout à bout : plan de joint = plan du plat ; cordon le long de y (largeur b), effort N le long de x ;
- C8 cordons libres : tel que saisi.

Les efforts se saisissent en kN et kN·m dans le sens du dessin ; ``efforts`` les convertit en N et
N·mm au point de référence. ``pieces`` décrit les pièces pour le dessin (polygones dans le plan de
joint, mm) et la section de gorge.
"""
import math
from typing import Dict, List, Tuple

import numpy as np

from .cordon import Cordon, beta_Lw, cordon_angle, cordon_bout_a_bout
from .groupe import Efforts

CONFIGS: Dict[str, Dict] = {
    "C1": {"libelle": "Recouvrement", "vue": "plan",
           "description": "Deux plats à recouvrement : cordons latéraux (cisaillés le long) et frontaux (sollicités en travers)."},
    "C2": {"libelle": "Jonction en T", "vue": "elevation",
           "description": "Pièce debout soudée sur une face par un ou deux cordons : arrachement, cisaillement, flexion du T."},
    "C3": {"libelle": "Console fléchie", "vue": "elevation",
           "description": "Plat en console soudé sur un poteau par ses cordons verticaux : V à l'excentricité e → flexion + cisaillement."},
    "C4": {"libelle": "Gousset en torsion", "vue": "plan",
           "description": "Plat posé à plat, soudé sur un à quatre côtés, chargé dans son plan hors du centre : torsion + cisaillement."},
    "C5": {"libelle": "Platine d'about de profilé", "vue": "face",
           "description": "Profilé du catalogue soudé sur une platine : semelles et âme sous N, V, M, T."},
    "C6": {"libelle": "Tube sur platine", "vue": "face",
           "description": "Tube rond, carré ou rectangulaire soudé sur son pourtour : N, V, M, T."},
    "C7": {"libelle": "Bout à bout", "vue": "plan",
           "description": "Pleine pénétration : la pièce gouverne ; pénétration partielle : cordon de gorge = pénétration."},
    "C8": {"libelle": "Cordons libres", "vue": "plan",
           "description": "N'importe quelle géométrie : liste de cordons et six composantes d'effort au point de référence."},
}

# efforts de chaque cas : (clé, libellé, unité, aide — le sens est celui du dessin)
EFFORTS: Dict[str, List[Tuple[str, str, str, str]]] = {
    "C1": [("N", "N (kN)", "kN", "Traction dans le plat rapporté, le long des cordons latéraux (> 0 : tire la pièce B)."),
           ("V", "V (kN)", "kN", "Effort transversal dans le plan des plats, en travers des cordons latéraux.")],
    "C2": [("N", "N (kN)", "kN", "Arrachement de la pièce debout (> 0 : traction perpendiculaire à la face)."),
           ("Va", "V∥ (kN)", "kN", "Cisaillement le long de la jonction."),
           ("Vp", "V⊥ (kN)", "kN", "Cisaillement dans la face, en travers de la jonction."),
           ("M", "M (kN·m)", "kN·m", "Flexion de la pièce debout autour de la jonction (repris par le couple des deux cordons).")],
    "C3": [("V", "V (kN)", "kN", "Charge verticale vers le bas à l'extrémité, à l'excentricité e de la face du poteau."),
           ("N", "N (kN)", "kN", "Traction horizontale dans la console (> 0 : tire la console hors du poteau), à mi-hauteur.")],
    "C4": [("V", "V (kN)", "kN", "Effort dans le plan du plat, vertical sur le dessin (> 0 vers le bas), au point (x_V ; y_V)."),
           ("H", "H (kN)", "kN", "Effort dans le plan du plat, horizontal (> 0 vers la droite), au même point."),
           ("N", "N (kN)", "kN", "Arrachement du plat (> 0 : tire le plat hors de l'appui), au même point.")],
    "C5": [("N", "N (kN)", "kN", "Effort normal (> 0 : traction, tire le profilé hors de la platine)."),
           ("Vz", "V_z (kN)", "kN", "Cisaillement selon l'âme (axe fort)."),
           ("Vy", "V_y (kN)", "kN", "Cisaillement selon les semelles (axe faible)."),
           ("My", "M_y (kN·m)", "kN·m", "Flexion d'axe fort (> 0 : tend la semelle supérieure)."),
           ("Mz", "M_z (kN·m)", "kN·m", "Flexion d'axe faible (> 0 : tend le bout de semelle de gauche)."),
           ("T", "T (kN·m)", "kN·m", "Torsion autour de l'axe du profilé.")],
    "C6": [("N", "N (kN)", "kN", "Effort normal (> 0 : traction)."),
           ("Vz", "V_z (kN)", "kN", "Cisaillement vertical sur le dessin."),
           ("Vy", "V_y (kN)", "kN", "Cisaillement horizontal sur le dessin."),
           ("My", "M_y (kN·m)", "kN·m", "Flexion autour de l'axe horizontal (> 0 : tend le haut)."),
           ("Mz", "M_z (kN·m)", "kN·m", "Flexion autour de l'axe vertical (> 0 : tend la gauche)."),
           ("T", "T (kN·m)", "kN·m", "Torsion autour de l'axe du tube.")],
    "C7": [("N", "N (kN)", "kN", "Traction perpendiculaire à la soudure, dans le plan du plat."),
           ("V", "V (kN)", "kN", "Cisaillement le long de la soudure, dans le plan du plat."),
           ("M", "M (kN·m)", "kN·m", "Flexion dans le plan du plat (> 0 : tend le bord y > 0).")],
    "C8": [("N", "N (kN)", "kN", "Normal au plan de joint (> 0 : arrachement)."),
           ("Vx", "V_x (kN)", "kN", "Dans le plan, selon x."), ("Vy", "V_y (kN)", "kN", "Dans le plan, selon y."),
           ("Mx", "M_x (kN·m)", "kN·m", "Flexion, vecteur selon x (> 0 : tend y > 0)."),
           ("My", "M_y (kN·m)", "kN·m", "Flexion, vecteur selon y (> 0 : comprime x > 0)."),
           ("Mz", "M_z (kN·m)", "kN·m", "Torsion autour de la normale.")],
}

PARAMS_DEFAUT: Dict[str, Dict] = {
    "C1": {"b": 100.0, "tB": 10.0, "tA": 10.0, "L_rec": 120.0, "lateraux": True, "frontaux": 1, "a": 5.0, "retours": False,
           "nuance_A": "S235", "nuance_B": "S235"},
    "C2": {"l": 200.0, "tB": 10.0, "tA": 12.0, "deux_cordons": True, "a": 5.0, "retours": False, "nuance_A": "S235", "nuance_B": "S235"},
    "C3": {"h": 200.0, "t": 10.0, "e": 150.0, "deux_cordons": True, "a": 5.0, "retours": False, "nuance_A": "S235", "nuance_B": "S235"},
    "C4": {"bx": 200.0, "by": 150.0, "t": 10.0, "cotes": ["gauche", "haut", "bas"], "a": 5.0, "retours": False,
           "xV": 250.0, "yV": 75.0, "nuance_A": "S235", "nuance_B": "S235"},
    "C5": {"profil": "IPE 200", "a_f": 5.0, "a_w": 4.0, "semelles": "deux faces", "ame": True, "retours": True,
           "nuance_A": "S235", "nuance_B": "S235"},
    "C6": {"profil": "SHS 100x100x5", "a": 4.0, "nuance_A": "S235", "nuance_B": "S235"},
    "C7": {"b": 150.0, "t": 12.0, "penetration": "pleine", "p": 6.0, "deux_faces": True, "nuance_A": "S235", "nuance_B": "S235"},
    "C8": {"cordons": [{"x1": -50.0, "y1": -100.0, "x2": -50.0, "y2": 100.0, "a": 5.0, "cote": -1, "retours": False},
                       {"x1": 50.0, "y1": -100.0, "x2": 50.0, "y2": 100.0, "a": 5.0, "cote": 1, "retours": False}],
           "xR": 0.0, "yR": 0.0, "zR": 0.0, "nuance_A": "S235", "nuance_B": "S235"},
}
EFFORTS_DEFAUT: Dict[str, Dict[str, float]] = {
    "C1": {"N": 80.0, "V": 0.0}, "C2": {"N": 40.0, "Va": 0.0, "Vp": 0.0, "M": 1.0}, "C3": {"V": 60.0, "N": 0.0},
    "C4": {"V": 40.0, "H": 0.0, "N": 0.0}, "C5": {"N": 0.0, "Vz": 60.0, "Vy": 0.0, "My": 30.0, "Mz": 0.0, "T": 0.0},
    "C6": {"N": 50.0, "Vz": 20.0, "Vy": 0.0, "My": 5.0, "Mz": 0.0, "T": 0.0}, "C7": {"N": 200.0, "V": 0.0, "M": 0.0},
    "C8": {"N": 0.0, "Vx": 0.0, "Vy": -60.0, "Mx": 9.0, "My": 0.0, "Mz": 0.0},
}


def _kn(v):
    return float(v) * 1e3


def _knm(v):
    return float(v) * 1e6


# ------------------------------------------------------------------ C1 recouvrement
def c1(p: Dict, e: Dict) -> Dict:
    b, L, a = float(p["b"]), float(p["L_rec"]), float(p["a"])
    ret = bool(p.get("retours", False))
    cs: List[Cordon] = []
    if p.get("lateraux", True):
        cs.append(cordon_angle("latéral haut", (0, b / 2), (L, b / 2), a, cote=-1, retours=ret))
        cs.append(cordon_angle("latéral bas", (0, -b / 2), (L, -b / 2), a, cote=+1, retours=ret))
    nf = int(p.get("frontaux", 1))
    if nf >= 1:
        cs.append(cordon_angle("frontal (bout de B)", (L, -b / 2), (L, b / 2), a, cote=+1, retours=ret))
    if nf >= 2:
        cs.append(cordon_angle("frontal (bout de A)", (0, -b / 2), (0, b / 2), a, cote=-1, retours=ret))
    blw = beta_Lw(L, a) if p.get("lateraux", True) else 1.0
    eff = Efforts(Vx=_kn(e.get("N", 0)), Vy=_kn(e.get("V", 0)), reference=(L / 2, 0.0, 0.0))
    pieces = {"vue": "plan", "A": [(-L * 0.6, -b * 0.75), (L, -b * 0.75), (L, b * 0.75), (-L * 0.6, b * 0.75)],
              "B": [(0, -b / 2), (L * 1.6, -b / 2), (L * 1.6, b / 2), (0, b / 2)], "tA": float(p["tA"]), "tB": float(p["tB"])}
    return {"cordons": cs, "efforts": eff, "beta_Lw": blw, "pieces": pieces,
            "fleches": [("N", (L * 1.5, 0.0), (1.0, 0.0)), ("V", (L * 1.3, 0.0), (0.0, -1.0))]}


# ------------------------------------------------------------------ C2 jonction en T
def c2(p: Dict, e: Dict) -> Dict:
    l, tB, a = float(p["l"]), float(p["tB"]), float(p["a"])
    ret = bool(p.get("retours", False))
    cs = [cordon_angle("cordon 1", (0, tB / 2), (l, tB / 2), a, cote=-1, retours=ret)]
    if p.get("deux_cordons", True):
        cs.append(cordon_angle("cordon 2", (0, -tB / 2), (l, -tB / 2), a, cote=+1, retours=ret))
    eff = Efforts(N=_kn(e.get("N", 0)), Vx=_kn(e.get("Va", 0)), Vy=_kn(e.get("Vp", 0)), Mx=_knm(e.get("M", 0)),
                  reference=(l / 2, 0.0, 0.0))
    pieces = {"vue": "elevation", "tA": float(p["tA"]), "tB": tB, "l": l, "hB": max(120.0, 6 * tB)}
    return {"cordons": cs, "efforts": eff, "beta_Lw": 1.0, "pieces": pieces, "fleches": []}


# ------------------------------------------------------------------ C3 console
def c3(p: Dict, e: Dict) -> Dict:
    h, t, ex, a = float(p["h"]), float(p["t"]), float(p["e"]), float(p["a"])
    ret = bool(p.get("retours", False))
    cs = [cordon_angle("cordon droit", (t / 2, -h / 2), (t / 2, h / 2), a, cote=+1, retours=ret)]
    if p.get("deux_cordons", True):
        cs.append(cordon_angle("cordon gauche", (-t / 2, -h / 2), (-t / 2, h / 2), a, cote=-1, retours=ret))
    eff = Efforts(Vy=-_kn(e.get("V", 0)), N=_kn(e.get("N", 0)), reference=(0.0, 0.0, ex))
    pieces = {"vue": "elevation", "h": h, "t": t, "e": ex, "longueur": ex * 1.35}
    return {"cordons": cs, "efforts": eff, "beta_Lw": 1.0, "pieces": pieces, "fleches": []}


# ------------------------------------------------------------------ C4 gousset
def c4(p: Dict, e: Dict) -> Dict:
    bx, by, a = float(p["bx"]), float(p["by"]), float(p["a"])
    ret = bool(p.get("retours", False))
    cotes = p.get("cotes", ["gauche"])
    cs = []
    if "gauche" in cotes:
        cs.append(cordon_angle("gauche", (0, 0), (0, by), a, cote=-1, retours=ret))
    if "droite" in cotes:
        cs.append(cordon_angle("droite", (bx, 0), (bx, by), a, cote=+1, retours=ret))
    if "bas" in cotes:
        cs.append(cordon_angle("bas", (0, 0), (bx, 0), a, cote=+1, retours=ret))
    if "haut" in cotes:
        cs.append(cordon_angle("haut", (0, by), (bx, by), a, cote=-1, retours=ret))
    xV, yV = float(p.get("xV", bx)), float(p.get("yV", by / 2))
    eff = Efforts(Vy=-_kn(e.get("V", 0)), Vx=_kn(e.get("H", 0)), N=_kn(e.get("N", 0)), reference=(xV, yV, 0.0))
    pieces = {"vue": "plan", "B": [(0, 0), (max(bx, xV + 20), 0), (max(bx, xV + 20), by), (0, by)], "bx": bx, "by": by,
              "t": float(p["t"]), "xV": xV, "yV": yV}
    return {"cordons": cs, "efforts": eff, "beta_Lw": 1.0, "pieces": pieces, "fleches": []}


# ------------------------------------------------------------------ C5 platine d'about
def _profil(nom: str) -> Dict:
    from acier.profiles import catalogue as CAT
    rec = CAT.par_nom().get(nom)
    if rec is None:
        raise ValueError(f"profilé inconnu au catalogue : {nom}")
    return rec


def c5(p: Dict, e: Dict) -> Dict:
    rec = _profil(p["profil"])
    h, b, tw, tf, r = rec["h"], rec["b"], rec["tw"], rec["tf"], rec.get("r", rec.get("r1", 0.0))
    af, aw = float(p["a_f"]), float(p["a_w"])
    ret = bool(p.get("retours", True))
    cs = [cordon_angle("semelle sup. ext.", (-b / 2, h / 2), (b / 2, h / 2), af, cote=-1, retours=ret),
          cordon_angle("semelle inf. ext.", (-b / 2, -h / 2), (b / 2, -h / 2), af, cote=+1, retours=ret)]
    if p.get("semelles", "deux faces") == "deux faces":
        x0 = tw / 2 + r
        cs += [cordon_angle("semelle sup. int. droite", (x0, h / 2 - tf), (b / 2, h / 2 - tf), af, cote=+1, retours=ret),
               cordon_angle("semelle sup. int. gauche", (-b / 2, h / 2 - tf), (-x0, h / 2 - tf), af, cote=+1, retours=ret),
               cordon_angle("semelle inf. int. droite", (x0, -h / 2 + tf), (b / 2, -h / 2 + tf), af, cote=-1, retours=ret),
               cordon_angle("semelle inf. int. gauche", (-b / 2, -h / 2 + tf), (-x0, -h / 2 + tf), af, cote=-1, retours=ret)]
    if p.get("ame", True):
        d = h / 2 - tf - r
        cs += [cordon_angle("âme droite", (tw / 2, -d), (tw / 2, d), aw, cote=+1, retours=ret),
               cordon_angle("âme gauche", (-tw / 2, -d), (-tw / 2, d), aw, cote=-1, retours=ret)]
    eff = Efforts(N=_kn(e.get("N", 0)), Vy=_kn(e.get("Vz", 0)), Vx=_kn(e.get("Vy", 0)),
                  Mx=_knm(e.get("My", 0)), My=_knm(e.get("Mz", 0)), Mz=_knm(e.get("T", 0)), reference=(0.0, 0.0, 0.0))
    pieces = {"vue": "face", "profil": rec, "platine": [(-b / 2 - 40, -h / 2 - 40), (b / 2 + 40, -h / 2 - 40), (b / 2 + 40, h / 2 + 40), (-b / 2 - 40, h / 2 + 40)]}
    return {"cordons": cs, "efforts": eff, "beta_Lw": 1.0, "pieces": pieces, "fleches": []}


# ------------------------------------------------------------------ C6 tube
def c6(p: Dict, e: Dict) -> Dict:
    rec = _profil(p["profil"])
    a = float(p["a"])
    cs = []
    if rec["famille"] in ("CHS", "CHS-F", "TUBE"):
        d = rec["d"]
        n = 24
        pts = [(d / 2 * math.cos(2 * math.pi * k / n), d / 2 * math.sin(2 * math.pi * k / n)) for k in range(n)]
    else:
        from acier.profiles import familles as F
        from acier.profiles import geometrie as G
        h, b, t = rec["h"], rec["b"], rec["t"]
        re, _ = F.rayons(rec["famille"], t)
        coins = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
        ring = G.arrondir(coins, [re] * 4, n=4)               # côtés droits + 4 tronçons par coin arrondi
        pts = [(float(x), float(y)) for x, y in ring]
    # un cordon continu sur le pourtour, découpé en tronçons pour le calcul (pas de longueur minimale propre)
    for k in range(len(pts)):
        cs.append(cordon_angle(f"pourtour {k + 1}", pts[k], pts[(k + 1) % len(pts)], a, cote=+1, retours=True, continu=True))
    contour = pts
    eff = Efforts(N=_kn(e.get("N", 0)), Vy=_kn(e.get("Vz", 0)), Vx=_kn(e.get("Vy", 0)),
                  Mx=_knm(e.get("My", 0)), My=_knm(e.get("Mz", 0)), Mz=_knm(e.get("T", 0)), reference=(0.0, 0.0, 0.0))
    ext = max(max(abs(x) for x, _ in contour), max(abs(y) for _, y in contour)) + 40
    pieces = {"vue": "face", "profil": rec, "contour": contour, "platine": [(-ext, -ext), (ext, -ext), (ext, ext), (-ext, ext)]}
    return {"cordons": cs, "efforts": eff, "beta_Lw": 1.0, "pieces": pieces, "fleches": []}


# ------------------------------------------------------------------ C7 bout à bout
def c7(p: Dict, e: Dict) -> Dict:
    b, t = float(p["b"]), float(p["t"])
    pleine = p.get("penetration", "pleine") == "pleine"
    cs = []
    if not pleine:
        pen = float(p["p"])
        cs.append(cordon_bout_a_bout("pénétration face 1", (0, -b / 2), (0, b / 2), pen, cote=+1, retours=True))
        if p.get("deux_faces", True):
            cs.append(cordon_bout_a_bout("pénétration face 2", (0, -b / 2), (0, b / 2), pen, cote=+1, retours=True))
    eff = Efforts(Vx=_kn(e.get("N", 0)), Vy=_kn(e.get("V", 0)), Mz=_knm(e.get("M", 0)), reference=(0.0, 0.0, 0.0))
    pieces = {"vue": "plan", "b": b, "t": t, "pleine": pleine, "A": [(-b, -b / 2), (0, -b / 2), (0, b / 2), (-b, b / 2)],
              "B": [(0, -b / 2), (b, -b / 2), (b, b / 2), (0, b / 2)]}
    return {"cordons": cs, "efforts": eff, "beta_Lw": 1.0, "pieces": pieces, "fleches": [], "pleine": pleine}


# ------------------------------------------------------------------ C8 libre
def c8(p: Dict, e: Dict) -> Dict:
    cs = []
    for i, c in enumerate(p.get("cordons", []), 1):
        cs.append(cordon_angle(c.get("nom") or f"cordon {i}", (c["x1"], c["y1"]), (c["x2"], c["y2"]), float(c["a"]),
                               cote=int(c.get("cote", 1)), retours=bool(c.get("retours", False))))
    eff = Efforts(N=_kn(e.get("N", 0)), Vx=_kn(e.get("Vx", 0)), Vy=_kn(e.get("Vy", 0)), Mx=_knm(e.get("Mx", 0)),
                  My=_knm(e.get("My", 0)), Mz=_knm(e.get("Mz", 0)),
                  reference=(float(p.get("xR", 0)), float(p.get("yR", 0)), float(p.get("zR", 0))))
    pieces = {"vue": "plan"}
    return {"cordons": cs, "efforts": eff, "beta_Lw": 1.0, "pieces": pieces, "fleches": []}


CONSTRUCTEURS = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6, "C7": c7, "C8": c8}


def construire(config: str, params: Dict, efforts: Dict) -> Dict:
    """Groupe de cordons, efforts et pièces d'une configuration."""
    if config not in CONSTRUCTEURS:
        raise ValueError(f"configuration inconnue : {config}")
    return CONSTRUCTEURS[config](params, efforts)

# -*- coding: utf-8 -*-
"""CARACTÉRISTIQUES D'UNE SECTION QUELCONQUE (docs/site/BIBLIOTHEQUE_AUTRES.md § 5).

Une section est une liste d'ÉLÉMENTS : un profilé du catalogue, un plat, un rond, un polygone. Chaque élément est un jeu
de CONTOURS polygonaux (extérieur dans le sens trigonométrique, trous dans le sens horaire ; les arcs — congés, cercles —
sont discrétisés finement, comme dans le catalogue des profilés). Un élément se COUPE par une droite (on garde un côté),
se retourne, se tourne, se déplace ; déclaré ÉVIDEMENT, sa matière est retirée.

MÉTHODE — celle d'un calcul à la main, sans approximation autre que la discrétisation des arcs :
  1. sur chaque contour, les INTÉGRALES DE GREEN donnent exactement pour le polygone : A, ∫y, ∫z, ∫y², ∫z², ∫yz ;
  2. centre de gravité : y_G = Σ A_i·y_i / Σ A_i, z_G = Σ A_i·z_i / Σ A_i ;
  3. HUYGENS-STEINER : I_y = Σ (I_y,i + A_i·(z_i − z_G)²), I_z = Σ (I_z,i + A_i·(y_i − y_G)²),
     I_yz = Σ (I_yz,i + A_i·(y_i − y_G)·(z_i − z_G)) ;
  4. axes principaux : tan 2α = −2·I_yz/(I_y − I_z), I_1,2 = (I_y + I_z)/2 ± √(((I_y − I_z)/2)² + I_yz²) ;
  5. modules élastiques W = I/v à chaque fibre extrême ; modules plastiques par l'axe d'égales aires (un seul matériau).

Repère : y horizontal (vers la droite), z vertical (vers le haut) ; I_y = ∫z² dA (flexion autour de l'axe horizontal).
Unités : mm, mm², mm⁴, mm³. Le cœur est en bibliothèque standard ; seul le CONTOUR d'un profilé du catalogue vient de
acier/profiles (numpy), le même qui a calculé le catalogue.

NON CALCULÉS, et dits : centre de cisaillement, inertie de torsion, inertie de gauchissement (ils demandent la fonction de
gauchissement de la section entière — voir ``NON_CALCULES``). La matière comptée deux fois (éléments qui se recouvrent) et
un évidement qui sort de la matière sont DÉTECTÉS (``recouvrements``).
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Anneau = List[Point]
N_CERCLE = 180
COTES_COUPE = ("haut", "bas", "gauche", "droite")
NON_CALCULES = {
    "centre_cisaillement": "Le centre de cisaillement demande la fonction de gauchissement de la section entière (ou la théorie des parois minces "
                           "sur sa ligne moyenne) : elle n'est pas définie pour un assemblage quelconque d'éléments pleins, de tubes et de pièces "
                           "coupées. Pour une section à deux axes de symétrie, il est au centre de gravité.",
    "torsion": "L'inertie de torsion I_t n'est pas la somme de celles des éléments dès qu'ils forment une cellule fermée ou se touchent : elle demande "
               "la résolution du problème de Saint-Venant sur la section entière.",
    "gauchissement": "L'inertie de gauchissement I_w dépend du centre de cisaillement et de la fonction de gauchissement : non calculée.",
}


class ErreurDonnees(ValueError):
    """Une donnée hors du domaine : le message dit laquelle."""


# ================================================================================================ contours élémentaires
def rectangle(b: float, h: float) -> List[Anneau]:
    """Un rectangle b × h centré à l'origine."""
    return [[(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]]


def rond(d: float, n: int = N_CERCLE) -> List[Anneau]:
    """Un disque de diamètre d, centré : polygone INSCRIT de n côtés (aire à 2·10⁻⁴ près pour n = 180)."""
    return [[(d / 2 * math.cos(2 * math.pi * k / n), d / 2 * math.sin(2 * math.pi * k / n)) for k in range(n)]]


def polygone(points: Sequence[Point]) -> List[Anneau]:
    """Un polygone libre (au moins trois sommets), remis dans le sens trigonométrique."""
    pts = [(float(y), float(z)) for y, z in points]
    if len(pts) < 3:
        raise ErreurDonnees("un polygone a au moins trois sommets")
    if integrales(pts)["A"] < 0:
        pts.reverse()
    return [pts]


def profil(nom: str) -> Tuple[List[Anneau], Dict]:
    """Les contours d'un profilé du catalogue (recentrés sur le centre de son rectangle enveloppe) et sa fiche."""
    from acier.profiles import catalogue as KA
    from acier.profiles import familles as FA
    p = KA.par_nom().get(nom)
    if p is None:
        raise ErreurDonnees(f"profilé inconnu : {nom!r}")
    rings = [[(float(q[0]), float(q[1])) for q in r] for r in FA.contours(p["famille"], p["dims"])]
    y0, y1, z0, z1 = boite(rings)
    cy, cz = (y0 + y1) / 2, (z0 + z1) / 2
    return [[(y - cy, z - cz) for y, z in r] for r in rings], p


# ================================================================================================ géométrie
def boite(rings: Sequence[Anneau]) -> Tuple[float, float, float, float]:
    """(y_min, y_max, z_min, z_max) des contours."""
    ys = [p[0] for r in rings for p in r]
    zs = [p[1] for r in rings for p in r]
    if not ys:
        return 0.0, 0.0, 0.0, 0.0
    return min(ys), max(ys), min(zs), max(zs)


def integrales(ring: Anneau) -> Dict[str, float]:
    """Les intégrales de Green d'un contour fermé, signées par son orientation : A, Qy = ∫y dA, Qz = ∫z dA,
    Jyy = ∫y² dA, Jzz = ∫z² dA, Jyz = ∫yz dA (par rapport à l'origine)."""
    A = Qy = Qz = Jyy = Jzz = Jyz = 0.0
    n = len(ring)
    for i in range(n):
        y1, z1 = ring[i]
        y2, z2 = ring[(i + 1) % n]
        c = y1 * z2 - y2 * z1
        A += c
        Qy += (y1 + y2) * c
        Qz += (z1 + z2) * c
        Jyy += (y1 * y1 + y1 * y2 + y2 * y2) * c
        Jzz += (z1 * z1 + z1 * z2 + z2 * z2) * c
        Jyz += (y1 * z2 + 2 * y1 * z1 + 2 * y2 * z2 + y2 * z1) * c
    return {"A": A / 2, "Qy": Qy / 6, "Qz": Qz / 6, "Jyy": Jyy / 12, "Jzz": Jzz / 12, "Jyz": Jyz / 24}


def somme(rings: Sequence[Anneau]) -> Dict[str, float]:
    tot = dict.fromkeys(("A", "Qy", "Qz", "Jyy", "Jzz", "Jyz"), 0.0)
    for r in rings:
        if len(r) >= 3:
            for k, v in integrales(r).items():
                tot[k] += v
    return tot


def couper_demi_plan(ring: Anneau, axe: int, c: float, garder_sup: bool) -> Anneau:
    """La partie du contour du côté coord[axe] ≥ c (ou ≤ c) : découpage de Sutherland-Hodgman par un demi-plan,
    orientation conservée. Un contour concave coupé en plusieurs morceaux reste UN contour (les morceaux sont reliés par
    des allers-retours d'aire nulle le long de la coupe) : ses intégrales sont exactes."""
    out: Anneau = []
    n = len(ring)
    sg = 1.0 if garder_sup else -1.0
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        fp, fq = sg * (p[axe] - c), sg * (q[axe] - c)
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out if len(out) >= 3 else []


def couper(rings: Sequence[Anneau], cote: str, c: float) -> List[Anneau]:
    """On GARDE le côté nommé : « haut » (z ≥ c), « bas » (z ≤ c), « droite » (y ≥ c), « gauche » (y ≤ c)."""
    if cote not in COTES_COUPE:
        raise ErreurDonnees(f"côté de coupe inconnu : {cote!r}")
    axe = 1 if cote in ("haut", "bas") else 0
    sup = cote in ("haut", "droite")
    return [r for r in (couper_demi_plan(x, axe, c, sup) for x in rings) if r]


def transformer(rings: Sequence[Anneau], angle_deg: float = 0.0, miroir: bool = False, dy: float = 0.0, dz: float = 0.0) -> List[Anneau]:
    """Miroir (y → −y), puis rotation autour de l'origine (sens trigonométrique), puis translation. Le miroir inverse
    l'orientation : le contour est retourné pour la garder."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for r in rings:
        pts = [((-y if miroir else y), z) for y, z in r]
        if miroir:
            pts.reverse()
        out.append([(y * ca - z * sa + dy, y * sa + z * ca + dz) for y, z in pts])
    return out


# ================================================================================================ une section
def element(contours: Sequence[Anneau], nom: str = "", evidement: bool = False) -> Dict:
    """Un élément de la section : ses contours (déjà placés), ses intégrales, son aire, son centre de gravité et ses
    inerties PROPRES (par rapport à son centre de gravité)."""
    t = somme(contours)
    A = t["A"]
    if abs(A) < 1e-9:
        raise ErreurDonnees(f"élément {nom or ''} d'aire nulle (entièrement coupé ?)".replace("  ", " "))
    yg, zg = t["Qy"] / A, t["Qz"] / A
    return {"nom": nom, "evidement": bool(evidement), "contours": [list(r) for r in contours], "A": A, "y": yg, "z": zg,
            "Iy": t["Jzz"] - A * zg * zg, "Iz": t["Jyy"] - A * yg * yg, "Iyz": t["Jyz"] - A * yg * zg, "boite": boite(contours)}


def _module_plastique(rings_signes: Sequence[Tuple[Anneau, float]], axe: int, lo: float, hi: float, A: float) -> Tuple[float, float]:
    """(W_pl, position de l'axe d'égales aires) : la droite coord[axe] = c0 partage l'aire en deux moitiés ;
    W_pl = ∫ |coord − c0| dA. ``rings_signes`` : (contour, +1 ou −1 pour un évidement)."""
    def aire_sup(c):
        return sum(sg * integrales(p)["A"] for r, sg in rings_signes for p in [couper_demi_plan(r, axe, c, True)] if p)
    for _ in range(70):
        mid = (lo + hi) / 2
        if aire_sup(mid) > A / 2:
            lo = mid
        else:
            hi = mid
    c0 = (lo + hi) / 2
    S = 0.0
    cle = "Qz" if axe == 1 else "Qy"
    for r, sg in rings_signes:
        for part, s2 in ((couper_demi_plan(r, axe, c0, True), 1.0), (couper_demi_plan(r, axe, c0, False), -1.0)):
            if part:
                m = integrales(part)
                S += sg * s2 * (m[cle] - c0 * m["A"])
    return abs(S), c0


def caracteristiques(elements: Sequence[Dict], plastique: bool = True) -> Dict:
    """Les caractéristiques de la section : aire, centre de gravité, inerties (Huygens-Steiner), axes principaux,
    rayons de giration, fibres extrêmes, modules élastiques et plastiques ; et le DÉTAIL par élément (ce que la feuille de
    calcul écrit)."""
    if not elements:
        raise ErreurDonnees("la section n'a aucun élément")
    lignes = []
    A = Sy = Sz = 0.0
    for e in elements:
        sg = -1.0 if e["evidement"] else 1.0
        lignes.append({"nom": e["nom"], "signe": sg, "A": sg * e["A"], "y": e["y"], "z": e["z"], "Iy": sg * e["Iy"], "Iz": sg * e["Iz"], "Iyz": sg * e["Iyz"]})
        A += sg * e["A"]
        Sy += sg * e["A"] * e["y"]
        Sz += sg * e["A"] * e["z"]
    if A <= 1e-9:
        raise ErreurDonnees("l'aire de la section est nulle ou négative (les évidements retirent plus que la matière)")
    yG, zG = Sy / A, Sz / A
    Iy = Iz = Iyz = 0.0
    for li in lignes:
        li["dy"], li["dz"] = li["y"] - yG, li["z"] - zG
        li["Iy_G"] = li["Iy"] + li["A"] * li["dz"] ** 2
        li["Iz_G"] = li["Iz"] + li["A"] * li["dy"] ** 2
        li["Iyz_G"] = li["Iyz"] + li["A"] * li["dy"] * li["dz"]
        Iy += li["Iy_G"]
        Iz += li["Iz_G"]
        Iyz += li["Iyz_G"]
    # axes principaux
    R = math.hypot((Iy - Iz) / 2.0, Iyz)
    I1, I2 = (Iy + Iz) / 2.0 + R, (Iy + Iz) / 2.0 - R
    isotrope = R < 1e-9 * max(Iy, Iz)
    alpha = 0.0 if isotrope else 0.5 * math.atan2(-2.0 * Iyz, Iy - Iz)
    # fibres extrêmes (sommets de la matière : les évidements n'en ajoutent pas)
    pts = [p for e in elements if not e["evidement"] for r in e["contours"] for p in r]
    ys, zs = [p[0] for p in pts], [p[1] for p in pts]
    v = {"z_sup": max(zs) - zG, "z_inf": zG - min(zs), "y_droite": max(ys) - yG, "y_gauche": yG - min(ys)}
    ca, sa = math.cos(alpha), math.sin(alpha)
    us = [(p[0] - yG) * ca + (p[1] - zG) * sa for p in pts]
    vs = [-(p[0] - yG) * sa + (p[1] - zG) * ca for p in pts]
    out = {"A": A, "yG": yG, "zG": zG, "Iy": Iy, "Iz": Iz, "Iyz": Iyz, "I1": I1, "I2": I2, "alpha": alpha, "alpha_deg": math.degrees(alpha),
           "isotrope": isotrope, "iy": math.sqrt(Iy / A), "iz": math.sqrt(Iz / A), "i1": math.sqrt(I1 / A), "i2": math.sqrt(max(I2, 0.0) / A),
           "fibres": v, "boite": (min(ys), max(ys), min(zs), max(zs)),
           "Wel_y_sup": Iy / v["z_sup"], "Wel_y_inf": Iy / v["z_inf"], "Wel_z_droite": Iz / v["y_droite"], "Wel_z_gauche": Iz / v["y_gauche"],
           "Wel_1": I1 / max(abs(x) for x in vs), "Wel_2": I2 / max(abs(x) for x in us), "lignes": lignes, "Sy": Sy, "Sz": Sz}
    out["Wel_y"], out["Wel_z"] = min(out["Wel_y_sup"], out["Wel_y_inf"]), min(out["Wel_z_droite"], out["Wel_z_gauche"])
    if plastique:
        rs = [(r, -1.0 if e["evidement"] else 1.0) for e in elements for r in e["contours"]]
        out["Wpl_y"], out["z_pl"] = _module_plastique(rs, 1, min(zs), max(zs), A)
        out["Wpl_z"], out["y_pl"] = _module_plastique(rs, 0, min(ys), max(ys), A)
    return out


# ================================================================================================ recouvrements
def _intervalles(rings: Sequence[Anneau], z: float) -> List[Tuple[float, float]]:
    """Les intervalles en y occupés par la matière des contours sur la droite horizontale d'ordonnée z (règle pair-impair :
    les trous d'un tube sont respectés)."""
    xs = []
    for r in rings:
        n = len(r)
        for i in range(n):
            (y1, z1), (y2, z2) = r[i], r[(i + 1) % n]
            if (z1 > z) != (z2 > z):
                xs.append(y1 + (z - z1) * (y2 - y1) / (z2 - z1))
    xs.sort()
    return [(xs[k], xs[k + 1]) for k in range(0, len(xs) - 1, 2)]


def _commun(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> float:
    return sum(max(0.0, min(u1, v1) - max(u0, v0)) for u0, u1 in a for v0, v1 in b)


def _union(ints: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for u, v in sorted(ints):
        if out and u <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], v))
        else:
            out.append((u, v))
    return out


def recouvrements(elements: Sequence[Dict], lignes: int = 400, seuil: float = 2e-3) -> List[Dict]:
    """La matière comptée DEUX FOIS : pour chaque paire d'éléments pleins, l'aire commune (balayage par ``lignes`` droites
    horizontales : une DÉTECTION, à quelques millièmes près), signalée au-delà de ``seuil`` × la plus petite des deux
    aires. Et pour chaque évidement, sa partie HORS de la matière (elle serait retirée à tort)."""
    out = []
    pleins = [(i, e) for i, e in enumerate(elements) if not e["evidement"]]
    for a in range(len(pleins)):
        for b in range(a + 1, len(pleins)):
            (i, ea), (j, eb) = pleins[a], pleins[b]
            z0, z1 = max(ea["boite"][2], eb["boite"][2]), min(ea["boite"][3], eb["boite"][3])
            if z1 <= z0 or min(ea["boite"][1], eb["boite"][1]) <= max(ea["boite"][0], eb["boite"][0]):
                continue
            h = (z1 - z0) / lignes
            aire = sum(_commun(_intervalles(ea["contours"], z0 + (k + 0.5) * h), _intervalles(eb["contours"], z0 + (k + 0.5) * h)) for k in range(lignes)) * h
            if aire > seuil * min(abs(ea["A"]), abs(eb["A"])):
                out.append({"sorte": "double", "i": i, "j": j, "aire": aire})
    for i, ev in enumerate(elements):
        if not ev["evidement"]:
            continue
        z0, z1 = ev["boite"][2], ev["boite"][3]
        h = (z1 - z0) / lignes
        hors = 0.0
        for k in range(lignes):
            z = z0 + (k + 0.5) * h
            trou = _intervalles(ev["contours"], z)
            matiere = _union([iv for _, e in pleins for iv in _intervalles(e["contours"], z)])
            hors += sum(v - u for u, v in trou) - _commun(trou, matiere)
        hors *= h
        if hors > seuil * abs(ev["A"]):
            out.append({"sorte": "evidement_hors", "i": i, "aire": hors})
    return out

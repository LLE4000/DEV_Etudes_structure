# -*- coding: utf-8 -*-
"""Géométrie et propriétés de sections d'acier : profilés laminés (I, H, U,
cornières), creux (ronds, carrés, rectangulaires) et reconstitués (PRS,
plats, ronds pleins). Pur numpy, aucune dépendance de calcul : tout ce que
le catalogue affiche est CALCULÉ ici depuis les dimensions, par intégration
exacte du contour (formule de Green sur le polygone, congés discrétisés),
sauf ce qui est dit approché dans le nom de la grandeur.

Conventions (EN 1993-1-1 §1.7) : l'axe y-y est l'axe FORT, horizontal,
parallèle aux semelles ; l'axe z-z est vertical, dans l'âme. Un point de la
section a pour coordonnées (y, z) ; I_y = ∫ z² dA, I_z = ∫ y² dA.
Unités internes : mm, mm², mm³, mm⁴, mm⁶ ; ``en_catalogue`` convertit en
cm², cm⁴, cm³, cm, kg/m comme les tables des fabricants.

Grandeurs et méthodes :
- A, centre de gravité, I_y, I_z, I_yz, axes principaux (α, I_u, I_v) :
  formule de Green sur le contour fermé (exacte sur le polygone, les arcs
  de congé étant discrétisés à ``N_ARC`` segments par arc : erreur relative
  sur l'aire d'un congé < 10⁻⁴) ;
- W_el : I / distance de la fibre extrême (module MINIMAL des deux côtés) ;
- W_pl : axe neutre plastique par bissection, moments statiques des deux
  moitiés découpées (découpage de polygone par un demi-plan) ;
- A_v,z et A_v,y : EN 1993-1-1 §6.2.6(3), selon la famille ;
- I_t : résolution numérique du problème de Saint-Venant (fonction de
  Prandtl, différences finies, gradient conjugué) sur le contour EXACT —
  congés compris — pour les sections ouvertes ; formules de Bredt de
  l'EN 10210-2 (annexe B) pour les creux ; Σ b·t³/3 (« I_t mince ») pour
  les sections reconstituées sans congé ;
- I_w et centre de cisaillement : théorie de Vlasov sur la ligne moyenne
  (coordonnée sectorielle, pôle ramené au centre de cisaillement) ; pour
  les I et H doublement symétriques, la forme des catalogues
  I_w = I_z·(h − t_f)²/4 est retenue (même valeur à 1 % près) ;
- masse : A × 7 850 kg/m³ (EN 1991-1-1 Tab. A.4) ; surface extérieure par
  mètre : périmètre extérieur.

Vérifications : ``tests/test_profiles_geometrie.py`` (rectangle, cercle,
tube, cornière, section en I contre les formules fermées ; Saint-Venant
contre la série exacte du rectangle et le cercle ; centre de cisaillement
du U contre e = 3·b'²·t_f/(6·b'·t_f + h'·t_w)).
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

RHO_ACIER = 7850.0          # kg/m³ — EN 1991-1-1 Tab. A.4 (acier)
N_ARC = 36                  # segments par arc de congé (erreur d'aire d'un quart de cercle ≈ 2,5·10⁻⁴·r²)
PENTE_IPN = 0.14            # pente intérieure des semelles IPN (DIN 1025-1)
PENTE_UPN = 0.08            # pente intérieure des semelles UPN (DIN 1026-1), h ≤ 300
PENTE_UPN_GRAND = 0.05      # UPN 320 à 400

Point = Tuple[float, float]


# ======================================================================= contours
def arrondir(points: Sequence[Point], rayons: Sequence[float], n: int = N_ARC) -> np.ndarray:
    """Contour fermé dont chaque sommet de rayon > 0 est remplacé par un arc
    tangent aux deux côtés qui s'y rencontrent (congé rentrant comme
    arrondi sortant : même construction). Les sommets sont donnés dans
    l'ordre du contour ; le résultat est un tableau (N, 2) fermé
    implicitement (le dernier point ne répète pas le premier)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    m = len(pts)
    out: List[np.ndarray] = []
    for i, p in enumerate(pts):
        r = float(rayons[i]) if i < len(rayons) else 0.0
        if r <= 0.0:
            out.append(p)
            continue
        u = p - pts[i - 1]
        v = pts[(i + 1) % m] - p
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        cos_phi = float(np.clip(np.dot(-u, v), -1.0, 1.0))
        phi = math.acos(cos_phi)                      # angle intérieur entre les deux côtés
        if phi < 1e-9 or abs(phi - math.pi) < 1e-9:
            out.append(p)
            continue
        d = r / math.tan(phi / 2.0)                   # distance des points de tangence au sommet
        bis = -u + v
        bis /= np.linalg.norm(bis)
        centre = p + bis * (r / math.sin(phi / 2.0))
        t1 = p - u * d
        t2 = p + v * d
        a1 = math.atan2(t1[1] - centre[1], t1[0] - centre[0])
        a2 = math.atan2(t2[1] - centre[1], t2[0] - centre[0])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        for k in range(n + 1):
            a = a1 + da * k / n
            out.append(centre + r * np.array([math.cos(a), math.sin(a)]))
    return np.array(out)


def cercle(rayon: float, centre: Point = (0.0, 0.0), n: int = 180, sens: int = 1) -> np.ndarray:
    """Contour circulaire discrétisé (n segments), sens +1 trigonométrique,
    −1 horaire (trou). Le rayon est corrigé pour que l'aire du polygone
    soit celle du disque (r_p = r·√(2π/(n·sin(2π/n))))."""
    rp = rayon * math.sqrt(2 * math.pi / (n * math.sin(2 * math.pi / n)))
    a = np.arange(n) * (2 * math.pi / n) * sens
    return np.column_stack([centre[0] + rp * np.cos(a), centre[1] + rp * np.sin(a)])


def contour_I(h: float, b: float, tw: float, tf: float, r: float = 0.0) -> np.ndarray:
    """I ou H à ailes parallèles (IPE, HE, HD, HL, HP, PRS avec r = 0) :
    quatre congés de rayon r à la jonction âme-semelle."""
    pts = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, -h / 2 + tf), (tw / 2, -h / 2 + tf),
           (tw / 2, h / 2 - tf), (b / 2, h / 2 - tf), (b / 2, h / 2), (-b / 2, h / 2),
           (-b / 2, h / 2 - tf), (-tw / 2, h / 2 - tf), (-tw / 2, -h / 2 + tf), (-b / 2, -h / 2 + tf)]
    ray = [0, 0, 0, r, r, 0, 0, 0, 0, r, r, 0]
    return arrondir(pts, ray)


def contour_I_conique(h: float, b: float, tw: float, tf: float, r1: float, r2: float,
                      pente: float = PENTE_IPN) -> np.ndarray:
    """I à semelles coniques (IPN) : faces intérieures des semelles inclinées
    de ``pente``, épaisseur t_f mesurée à b/4 de l'axe de l'âme (DIN 1025-1),
    congé r1 à l'âme, arrondi r2 au bout de la semelle."""
    e_bout = tf - pente * (b / 4)                 # épaisseur au bout de la semelle
    e_ame = tf + pente * (b / 4 - tw / 2)         # épaisseur contre l'âme
    pts = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, -h / 2 + e_bout), (tw / 2, -h / 2 + e_ame),
           (tw / 2, h / 2 - e_ame), (b / 2, h / 2 - e_bout), (b / 2, h / 2), (-b / 2, h / 2),
           (-b / 2, h / 2 - e_bout), (-tw / 2, h / 2 - e_ame), (-tw / 2, -h / 2 + e_ame), (-b / 2, -h / 2 + e_bout)]
    ray = [0, 0, r2, r1, r1, r2, 0, 0, r2, r1, r1, r2]
    return arrondir(pts, ray)


def contour_U(h: float, b: float, tw: float, tf: float, r: float = 0.0) -> np.ndarray:
    """U à ailes parallèles (UPE) : dos de l'âme en y = 0, ailes vers y > 0 ;
    deux congés r à la jonction âme-semelle. Le contour est rendu tel quel
    (le centre de gravité est calculé ensuite)."""
    pts = [(0.0, -h / 2), (b, -h / 2), (b, -h / 2 + tf), (tw, -h / 2 + tf),
           (tw, h / 2 - tf), (b, h / 2 - tf), (b, h / 2), (0.0, h / 2)]
    ray = [0, 0, 0, r, r, 0, 0, 0]
    return arrondir(pts, ray)


def contour_U_conique(h: float, b: float, tw: float, tf: float, r1: float, r2: float,
                      pente: float, u: Optional[float] = None) -> np.ndarray:
    """U à semelles coniques (UPN) : t_f mesurée à la distance ``u`` du bout
    de la semelle — b/2 par défaut (DIN 1026-1, UPN 50 à 300 à 8 % : c'est
    cette convention qui retrouve les aires publiées, UPN 200 → 32,2 cm²
    contre 32,6 si t_f était prise au milieu de la partie inclinée) ;
    (b − t_w)/2 pour les UPN 320 à 400 à 5 % (UPN 400 → 91,6 cm² publié
    91,5). Congé r1 à l'âme, arrondi r2 au bout de la semelle."""
    u = b / 2.0 if u is None else u
    e_bout = tf - pente * u
    e_ame = tf + pente * (b - tw - u)
    pts = [(0.0, -h / 2), (b, -h / 2), (b, -h / 2 + e_bout), (tw, -h / 2 + e_ame),
           (tw, h / 2 - e_ame), (b, h / 2 - e_bout), (b, h / 2), (0.0, h / 2)]
    ray = [0, 0, r2, r1, r1, r2, 0, 0]
    return arrondir(pts, ray)


def contour_L(h: float, b: float, t: float, r1: float = 0.0, r2: float = 0.0) -> np.ndarray:
    """Cornière : aile verticale de hauteur h (le long de z), aile horizontale
    de largeur b (le long de y), épaisseur t ; congé r1 au talon, arrondis r2
    aux bouts des ailes (EN 10056-1). Talon extérieur à l'origine."""
    pts = [(0.0, 0.0), (b, 0.0), (b, t), (t, t), (t, h), (0.0, h)]
    ray = [0, 0, r2, r1, r2, 0]
    return arrondir(pts, ray)


def contour_rectangle(b: float, h: float, r: float = 0.0, sens: int = 1) -> np.ndarray:
    """Rectangle b × h centré, coins arrondis de rayon r ; sens −1 pour un
    trou (contour horaire)."""
    pts = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
    c = arrondir(pts, [r, r, r, r])
    return c if sens > 0 else c[::-1].copy()


def rayons_creux(t: float, norme: str) -> Tuple[float, float]:
    """Rayons extérieur et intérieur des coins d'un creux carré ou
    rectangulaire : EN 10210-2 (fini à chaud) r_ext = 1,5·t, r_int = 1,0·t ;
    EN 10219-2 (formé à froid) r_ext = 2·t (t ≤ 6), 2,5·t (6 < t ≤ 10),
    3·t (t > 10), r_int = r_ext − t."""
    if norme == "EN 10210":
        return 1.5 * t, 1.0 * t
    if norme == "EN 10219":
        re = 2.0 * t if t <= 6.0 else (2.5 * t if t <= 10.0 else 3.0 * t)
        return re, re - t
    raise ValueError(f"norme de creux inconnue : {norme!r}")


def contours_creux_rect(b: float, h: float, t: float, r_ext: float, r_int: float) -> List[np.ndarray]:
    """Creux rectangulaire : contour extérieur (trigonométrique) et trou (horaire)."""
    return [contour_rectangle(b, h, r_ext, +1), contour_rectangle(b - 2 * t, h - 2 * t, r_int, -1)]


def contours_creux_rond(d: float, t: float) -> List[np.ndarray]:
    """Creux circulaire : cercle extérieur et trou."""
    return [cercle(d / 2, sens=+1), cercle(d / 2 - t, sens=-1)]


# ======================================================================= intégration
def _moments(ring: np.ndarray) -> Dict[str, float]:
    """Intégrales de Green sur un contour fermé (signées par l'orientation) :
    A, ∫y, ∫z, ∫y², ∫z², ∫yz."""
    y, z = ring[:, 0], ring[:, 1]
    y2, z2 = np.roll(y, -1), np.roll(z, -1)
    c = y * z2 - y2 * z
    return {"A": float(np.sum(c) / 2.0),
            "Sy": float(np.sum((y + y2) * c) / 6.0),          # ∫ y dA
            "Sz": float(np.sum((z + z2) * c) / 6.0),          # ∫ z dA
            "Iyy": float(np.sum((y * y + y * y2 + y2 * y2) * c) / 12.0),   # ∫ y² dA
            "Izz": float(np.sum((z * z + z * z2 + z2 * z2) * c) / 12.0),   # ∫ z² dA
            "Iyz": float(np.sum((y * z2 + 2 * y * z + 2 * y2 * z2 + y2 * z) * c) / 24.0)}


def _somme(rings: Sequence[np.ndarray]) -> Dict[str, float]:
    tot = {"A": 0.0, "Sy": 0.0, "Sz": 0.0, "Iyy": 0.0, "Izz": 0.0, "Iyz": 0.0}
    for r in rings:
        m = _moments(r)
        for k in tot:
            tot[k] += m[k]
    return tot


def _decouper(ring: np.ndarray, axe: int, c: float, garder_sup: bool) -> np.ndarray:
    """Partie du contour du côté ``coord[axe] ≥ c`` (ou ≤ c) — découpage de
    Sutherland-Hodgman par un demi-plan ; l'orientation est conservée."""
    out = []
    n = len(ring)
    sgn = 1.0 if garder_sup else -1.0
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        fp, fq = sgn * (p[axe] - c), sgn * (q[axe] - c)
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            out.append(p + t * (q - p))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def module_plastique(rings: Sequence[np.ndarray], axe_flexion: str) -> Tuple[float, float]:
    """W_pl autour de y (axe_flexion = 'y', ligne neutre horizontale z = z0)
    ou de z : (W_pl en mm³, position de l'axe neutre plastique en mm)."""
    coord = 1 if axe_flexion == "y" else 0          # coordonnée perpendiculaire à l'axe
    tot = _somme(rings)
    A = tot["A"]
    lo = min(float(r[:, coord].min()) for r in rings if len(r))
    hi = max(float(r[:, coord].max()) for r in rings if len(r))

    def aire_sup(c):
        return sum(_moments(_decouper(r, coord, c, True))["A"] for r in rings
                   if len(_decouper(r, coord, c, True)))

    for _ in range(80):                                  # bissection : aire au-dessus = A/2
        mid = (lo + hi) / 2
        if aire_sup(mid) > A / 2:
            lo = mid
        else:
            hi = mid
    c0 = (lo + hi) / 2
    S = 0.0
    for r in rings:
        for part, sgn in ((_decouper(r, coord, c0, True), 1.0), (_decouper(r, coord, c0, False), -1.0)):
            if len(part):
                m = _moments(part)
                S += sgn * ((m["Sz"] if coord == 1 else m["Sy"]) - c0 * m["A"])
    return abs(S), c0


def proprietes(rings: Sequence[np.ndarray], plastique: bool = True) -> Dict[str, float]:
    """Propriétés géométriques d'une section (mm) : aire, centre de gravité
    (yc, zc dans le repère du contour), I_y, I_z, I_yz, axes principaux,
    modules élastiques minimaux, modules plastiques, rayons de giration,
    distances aux fibres extrêmes, périmètre extérieur."""
    tot = _somme(rings)
    A = tot["A"]
    yc, zc = tot["Sy"] / A, tot["Sz"] / A
    Iy = tot["Izz"] - A * zc * zc        # ∫ (z − zc)² dA
    Iz = tot["Iyy"] - A * yc * yc
    Iyz = tot["Iyz"] - A * yc * zc
    pts = np.vstack([r for r in rings if len(r)])
    ys, zs = pts[:, 0] - yc, pts[:, 1] - zc
    z_sup, z_inf = float(zs.max()), float(-zs.min())
    y_dro, y_gau = float(ys.max()), float(-ys.min())
    # axes principaux : tan 2α = −2·I_yz / (I_y − I_z) ; u = axe de I maximal
    alpha = 0.5 * math.atan2(-2.0 * Iyz, Iy - Iz) if abs(Iyz) > 1e-9 * max(Iy, Iz) else 0.0
    ca, sa = math.cos(alpha), math.sin(alpha)
    Iu = Iy * ca * ca + Iz * sa * sa - 2 * Iyz * sa * ca
    Iv = Iy + Iz - Iu
    if Iv > Iu:                                          # garder u = axe fort
        alpha += math.pi / 2
        ca, sa = math.cos(alpha), math.sin(alpha)
        Iu, Iv = Iv, Iu
    u = ys * ca + zs * sa                                # coordonnées dans le repère principal
    v = -ys * sa + zs * ca
    out = {"A": A, "yc": yc, "zc": zc, "Iy": Iy, "Iz": Iz, "Iyz": Iyz,
           "z_sup": z_sup, "z_inf": z_inf, "y_droite": y_dro, "y_gauche": y_gau,
           "Wel_y": Iy / max(z_sup, z_inf), "Wel_z": Iz / max(y_dro, y_gau),
           "iy": math.sqrt(Iy / A), "iz": math.sqrt(Iz / A),
           "alpha_deg": math.degrees(alpha), "Iu": Iu, "Iv": Iv,
           "Wel_u": Iu / float(np.max(np.abs(v))), "Wel_v": Iv / float(np.max(np.abs(u))),
           "iu": math.sqrt(Iu / A), "iv": math.sqrt(Iv / A),
           "perimetre": float(sum(np.sum(np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1))
                                  for r in rings if len(r) and _moments(r)["A"] > 0))}
    if plastique:
        out["Wpl_y"], out["z_pl"] = module_plastique(rings, "y")
        out["Wpl_z"], out["y_pl"] = module_plastique(rings, "z")
    return out


# ======================================================================= aires de cisaillement
def aires_cisaillement(famille: str, h: float, b: float, tw: float, tf: float, r: float, A: float) -> Dict[str, Optional[float]]:
    """A_v,z (effort parallèle à l'âme) et A_v,y (parallèle aux semelles),
    EN 1993-1-1 §6.2.6(3) : I/H laminés a) A − 2·b·t_f + (t_w + 2·r)·t_f ;
    U laminés c) A − 2·b·t_f + (t_w + r)·t_f ; parallèle aux semelles
    A − Σ(h_w·t_w) ; creux rectangulaires f) A·h/(b + h) et A·b/(b + h) ;
    creux circulaires g) 2·A/π. Le plancher η·h_w·t_w de a) n'est pas
    appliqué (η de l'EN 1993-1-5, annexe nationale) : dit dans la doc."""
    if famille in ("I", "I_conique"):
        return {"Avz": A - 2 * b * tf + (tw + 2 * r) * tf, "Avy": A - (h - 2 * tf) * tw}
    if famille in ("U", "U_conique"):
        return {"Avz": A - 2 * b * tf + (tw + r) * tf, "Avy": A - (h - 2 * tf) * tw}
    if famille == "creux_rect":
        return {"Avz": A * h / (b + h), "Avy": A * b / (b + h)}
    if famille == "creux_rond":
        return {"Avz": 2 * A / math.pi, "Avy": 2 * A / math.pi}
    return {"Avz": None, "Avy": None}


# ======================================================================= torsion : Saint-Venant
def _interieur(rings: Sequence[np.ndarray], Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Parité des croisements (even-odd) : points (Y, Z) intérieurs à l'union
    des contours (un trou, de parité opposée, est exclu)."""
    inside = np.zeros(Y.shape, dtype=bool)
    for ring in rings:
        n = len(ring)
        for i in range(n):
            y1, z1 = ring[i]
            y2, z2 = ring[(i + 1) % n]
            if z1 == z2:
                continue
            cond = (z1 > Z) != (z2 > Z)
            yx = (y2 - y1) * (Z - z1) / (z2 - z1) + y1
            inside ^= cond & (Y < yx)
    return inside


def _distances_bord(rings: Sequence[np.ndarray], Y: np.ndarray, Z: np.ndarray, masque: np.ndarray,
                    pas: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pour chaque cellule intérieure et chaque direction (+y, −y, +z, −z) :
    la distance au bord VRAI de la section le long de la ligne de grille,
    bornée au pas (pas entier quand le voisin est intérieur). Lancer de
    rayon contre toutes les arêtes des contours, vectorisé par direction."""
    ny, nz = masque.shape
    h = pas
    dE, dW, dN, dS = (np.full(masque.shape, h) for _ in range(4))
    voisin_ext = {
        "E": np.zeros_like(masque), "W": np.zeros_like(masque), "N": np.zeros_like(masque), "S": np.zeros_like(masque)}
    voisin_ext["E"][:-1, :] = ~masque[1:, :]
    voisin_ext["E"][-1, :] = True
    voisin_ext["W"][1:, :] = ~masque[:-1, :]
    voisin_ext["W"][0, :] = True
    voisin_ext["N"][:, :-1] = ~masque[:, 1:]
    voisin_ext["N"][:, -1] = True
    voisin_ext["S"][:, 1:] = ~masque[:, :-1]
    voisin_ext["S"][:, 0] = True
    aretes = []
    for ring in rings:
        aretes.append(np.column_stack([ring, np.roll(ring, -1, axis=0)]))      # y1 z1 y2 z2
    A = np.vstack(aretes)
    y1, z1, y2, z2 = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    for sens, dist, horiz, signe in (("E", dE, True, 1.0), ("W", dW, True, -1.0), ("N", dN, False, 1.0), ("S", dS, False, -1.0)):
        cellules = masque & voisin_ext[sens]
        if not cellules.any():
            continue
        yc, zc = Y[cellules], Z[cellules]
        meilleur = np.full(yc.shape, h)
        if horiz:                                  # rayon horizontal z = zc, vers +y ou −y
            croise = (z1[None, :] > zc[:, None]) != (z2[None, :] > zc[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                y_int = y1[None, :] + (zc[:, None] - z1[None, :]) * (y2 - y1)[None, :] / (z2 - z1)[None, :]
            s = signe * (y_int - yc[:, None])
        else:                                      # rayon vertical y = yc, vers +z ou −z
            croise = (y1[None, :] > yc[:, None]) != (y2[None, :] > yc[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                z_int = z1[None, :] + (yc[:, None] - y1[None, :]) * (z2 - z1)[None, :] / (y2 - y1)[None, :]
            s = signe * (z_int - zc[:, None])
        s = np.where(croise & (s > 0.0) & (s <= h), s, np.inf)
        meilleur = np.minimum(meilleur, s.min(axis=1))
        meilleur = np.where(np.isfinite(meilleur), meilleur, h)
        dist[cellules] = np.maximum(meilleur, 0.1 * h)        # garde-fou de conditionnement
    return dE, dW, dN, dS


def torsion_saint_venant(rings: Sequence[np.ndarray], pas: float, tol: float = 1e-9,
                         max_iter: Optional[int] = None, solveur: Optional[Callable] = None) -> float:
    """Constante de torsion I_t (mm⁴) d'une section OUVERTE (simplement
    connexe) : fonction de Prandtl φ, ∇²φ = −2 à l'intérieur, φ = 0 sur le
    bord ; I_t = 2·∫φ dA. Différences finies sur une grille carrée de pas
    ``pas`` (cellules dont le centre est dans la section) avec le schéma de
    Shortley-Weller : près du bord, le maillon du stencil est raccourci à la
    distance VRAIE au contour (congés compris), ce qui rend l'erreur
    O(h²) quelle que soit la position du bord — une grille « en escalier »
    donnait O(h) et jusqu'à −5 % sur les semelles minces à grands congés
    (HE 320 AA, mesuré le 04/11/2026). Système non symétrique résolu par
    BiCGSTAB sans matrice (numpy seul) ; ``solveur`` optionnel :
    f(masque, diag, cE, cW, cN, cS) → φ (grille complète), pour un solveur
    direct hors chemin de production (scipy dans l'outil de transcription)."""
    pts = np.vstack(rings)
    y0, y1 = pts[:, 0].min() - pas, pts[:, 0].max() + pas
    z0, z1 = pts[:, 1].min() - pas, pts[:, 1].max() + pas
    ny, nz = int(math.ceil((y1 - y0) / pas)), int(math.ceil((z1 - z0) / pas))
    yc = y0 + (np.arange(ny) + 0.5) * pas
    zc = z0 + (np.arange(nz) + 0.5) * pas
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    masque = _interieur(rings, Y, Z)
    dE, dW, dN, dS = _distances_bord(rings, Y, Z, masque, pas)
    # stencil de Shortley-Weller : u'' ≈ 2·[u_E/(hE·(hE+hW)) + u_W/(hW·(hE+hW)) − u_P/(hE·hW)]
    diag = 2.0 / (dE * dW) + 2.0 / (dN * dS)
    cE = 2.0 / (dE * (dE + dW))
    cW = 2.0 / (dW * (dE + dW))
    cN = 2.0 / (dN * (dN + dS))
    cS = 2.0 / (dS * (dN + dS))
    if solveur is not None:
        phi = solveur(masque, diag, cE, cW, cN, cS)
        return float(2.0 * pas * pas * np.sum(phi[masque]))

    def op(x):                                          # −∇² discret, φ nul hors de la section
        x = np.where(masque, x, 0.0)
        out = diag * x
        out[:-1, :] -= cE[:-1, :] * x[1:, :]
        out[1:, :] -= cW[1:, :] * x[:-1, :]
        out[:, :-1] -= cN[:, :-1] * x[:, 1:]
        out[:, 1:] -= cS[:, 1:] * x[:, :-1]
        return np.where(masque, out, 0.0)

    x = _bicgstab(op, np.where(masque, 2.0, 0.0), tol, max_iter or 60 * (ny + nz))
    return float(2.0 * pas * pas * np.sum(x[masque]))


def _bicgstab(op: Callable, b: np.ndarray, tol: float, n_max: int) -> np.ndarray:
    """BiCGSTAB (van der Vorst) sans matrice, pour le système non symétrique
    du schéma de Shortley-Weller."""
    x = np.zeros_like(b)
    r = b - op(x)
    r0 = r.copy()
    rho = alpha = omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    bb = math.sqrt(float(np.sum(b * b)))
    for _ in range(n_max):
        rho_new = float(np.sum(r0 * r))
        if rho_new == 0.0:
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = op(p)
        alpha = rho_new / float(np.sum(r0 * v))
        s = r - alpha * v
        if math.sqrt(float(np.sum(s * s))) <= tol * bb:
            x += alpha * p
            break
        t = op(s)
        omega = float(np.sum(t * s)) / float(np.sum(t * t))
        x += alpha * p + omega * s
        r = s - omega * t
        rho = rho_new
        if math.sqrt(float(np.sum(r * r))) <= tol * bb:
            break
    return x


def torsion_mince(segments: Sequence[Tuple[Point, Point, float]]) -> float:
    """I_t ≈ Σ L·t³/3 d'une section ouverte à parois minces (sans congés)."""
    return sum(math.dist(a, b_) * t ** 3 / 3.0 for a, b_, t in segments)


def torsion_creux_rond(d: float, t: float) -> Tuple[float, float]:
    """Tube rond : I_t = π·(d⁴ − (d − 2t)⁴)/32 (exact), W_t = 2·I_t/d."""
    It = math.pi * (d ** 4 - (d - 2 * t) ** 4) / 32.0
    return It, 2 * It / d


def torsion_creux_rect(b: float, h: float, t: float, r_ext: float, r_int: float) -> Tuple[float, float]:
    """Tube rectangulaire, EN 10210-2 annexe B : r_c = (r_ext + r_int)/2,
    h_m = 2·[(b − t) + (h − t)] − 2·r_c·(4 − π), A_h = (b − t)·(h − t) −
    r_c²·(4 − π), k = 2·A_h·t/h_m, I_t = t³·h_m/3 + 2·k·A_h,
    W_t = I_t/(t + k/t)."""
    rc = (r_ext + r_int) / 2.0
    hm = 2.0 * ((b - t) + (h - t)) - 2.0 * rc * (4.0 - math.pi)
    Ah = (b - t) * (h - t) - rc * rc * (4.0 - math.pi)
    k = 2.0 * Ah * t / hm
    It = t ** 3 * hm / 3.0 + 2.0 * k * Ah
    return It, It / (t + k / t)


# ======================================================================= gauchissement : ligne moyenne
def gauchissement(segments: Sequence[Tuple[Point, Point, float]]) -> Dict[str, float]:
    """Théorie de Vlasov sur une ligne moyenne CONNEXE et ouverte (arbre de
    segments droits (P1, P2, épaisseur)) : centre de cisaillement (y_s, z_s)
    dans le repère des segments, constante de gauchissement I_w = ∫ ω_n² t ds
    (mm⁶) avec ω_n la coordonnée sectorielle normalisée par rapport au
    centre de cisaillement. Les propriétés de la ligne moyenne (A, I)
    servent au système ; les segments sont parcourus depuis le premier
    point rencontré, chaque segment prolongeant un nœud déjà atteint."""
    segs = [(np.asarray(a, float), np.asarray(b_, float), float(t)) for a, b_, t in segments]
    A = sum(np.linalg.norm(b_ - a) * t for a, b_, t in segs)
    Sy = sum(np.linalg.norm(b_ - a) * t * (a[0] + b_[0]) / 2 for a, b_, t in segs)
    Sz = sum(np.linalg.norm(b_ - a) * t * (a[1] + b_[1]) / 2 for a, b_, t in segs)
    yc, zc = Sy / A, Sz / A
    segs = [(a - (yc, zc), b_ - (yc, zc), t) for a, b_, t in segs]     # repère centré

    def integ(f1, f2, g1, g2, L, t):
        """∫ f·g t ds pour f, g linéaires le long du segment."""
        df, dg = f2 - f1, g2 - g1
        return t * L * (f1 * g1 + (f1 * dg + df * g1) / 2.0 + df * dg / 3.0)

    Iy = Iz = Iyz = 0.0
    for a, b_, t in segs:
        L = float(np.linalg.norm(b_ - a))
        Iy += integ(a[1], b_[1], a[1], b_[1], L, t)
        Iz += integ(a[0], b_[0], a[0], b_[0], L, t)
        Iyz += integ(a[0], b_[0], a[1], b_[1], L, t)
    # coordonnée sectorielle, pôle au centre de gravité, parcours en arbre
    omega: Dict[Tuple[float, float], float] = {}
    restants = list(range(len(segs)))
    cle = lambda p: (round(float(p[0]), 6), round(float(p[1]), 6))   # noqa: E731
    omega[cle(segs[0][0])] = 0.0
    while restants:
        avance = False
        for i in list(restants):
            a, b_, t = segs[i]
            ka, kb = cle(a), cle(b_)
            if ka in omega and kb not in omega:
                omega[kb] = omega[ka] + (a[0] * (b_[1] - a[1]) - a[1] * (b_[0] - a[0]))
            elif kb in omega and ka not in omega:
                omega[ka] = omega[kb] + (b_[0] * (a[1] - b_[1]) - b_[1] * (a[0] - b_[0]))
            elif ka not in omega and kb not in omega:
                continue
            restants.remove(i)
            avance = True
        if not avance:
            raise ValueError("ligne moyenne non connexe")
    # centre de cisaillement : ∫ω_S·y = ∫ω_S·z = 0 avec ω_S = ω − Δy·z + Δz·y
    Iwy = Iwz = 0.0
    for a, b_, t in segs:
        L = float(np.linalg.norm(b_ - a))
        wa, wb = omega[cle(a)], omega[cle(b_)]
        Iwy += integ(wa, wb, a[0], b_[0], L, t)
        Iwz += integ(wa, wb, a[1], b_[1], L, t)
    M = np.array([[-Iyz, Iz], [-Iy, Iyz]])
    dy, dz = np.linalg.solve(M, np.array([-Iwy, -Iwz]))
    ws = {k: omega[k] - dy * k[1] + dz * k[0] for k in omega}
    moy = sum(np.linalg.norm(b_ - a) * t * (ws[cle(a)] + ws[cle(b_)]) / 2 for a, b_, t in segs) / A
    Iw = 0.0
    for a, b_, t in segs:
        L = float(np.linalg.norm(b_ - a))
        wa, wb = ws[cle(a)] - moy, ws[cle(b_)] - moy
        Iw += integ(wa, wb, wa, wb, L, t)
    return {"Iw": Iw, "y_s": yc + dy, "z_s": zc + dz, "yc_m": yc, "zc_m": zc, "A_m": A, "Iy_m": Iy, "Iz_m": Iz}


# ======================================================================= lignes moyennes
def ligne_moyenne_I(h: float, b: float, tw: float, tf: float) -> List[Tuple[Point, Point, float]]:
    """Semelles coupées en deux à l'âme : les segments partagent leurs nœuds."""
    hm = h - tf
    return [((-b / 2, -hm / 2), (0.0, -hm / 2), tf), ((0.0, -hm / 2), (b / 2, -hm / 2), tf),
            ((0.0, -hm / 2), (0.0, hm / 2), tw),
            ((-b / 2, hm / 2), (0.0, hm / 2), tf), ((0.0, hm / 2), (b / 2, hm / 2), tf)]


def ligne_moyenne_U(h: float, b: float, tw: float, tf: float) -> List[Tuple[Point, Point, float]]:
    """Dos de l'âme en y = 0 : âme en y = t_w/2, ailes de t_w/2 au bout b
    (ligne moyenne de longueur b − t_w/2)."""
    hm = h - tf
    return [((tw / 2, -hm / 2), (b, -hm / 2), tf), ((tw / 2, -hm / 2), (tw / 2, hm / 2), tw),
            ((tw / 2, hm / 2), (b, hm / 2), tf)]


def ligne_moyenne_U_conique(h: float, b: float, tw: float, tf: float, pente: float, u: float,
                            n: int = 8) -> List[Tuple[Point, Point, float]]:
    """UPN : chaque aile découpée en ``n`` tronçons d'épaisseur locale
    e(y) = t_f + pente·(b − u − y) (t_f mesurée à u du bout), l'épaisseur
    moyenne du tronçon servant à Vlasov ; l'axe de l'aile reste à
    ±(h − t_f)/2. Sans ce découpage, I_w est sous-estimé d'environ 15 %
    (mesuré le 04/11/2026 contre les valeurs publiées)."""
    hm = h - tf
    segs: List[Tuple[Point, Point, float]] = [((tw / 2, -hm / 2), (tw / 2, hm / 2), tw)]
    ys = np.linspace(tw / 2, b, n + 1)
    for signe in (-1.0, 1.0):
        for ya, yb in zip(ys[:-1], ys[1:]):
            ym = (ya + yb) / 2
            e = tf + pente * (b - u - ym)
            segs.append(((float(ya), signe * hm / 2), (float(yb), signe * hm / 2), float(e)))
    return segs


def ligne_moyenne_L(h: float, b: float, t: float) -> List[Tuple[Point, Point, float]]:
    return [((t / 2, t / 2), (b, t / 2), t), ((t / 2, t / 2), (t / 2, h), t)]


# ======================================================================= conversions
def en_catalogue(p: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """mm → unités des catalogues : A, A_v en cm² ; I en cm⁴ ; W en cm³ ;
    i, distances en cm ; I_w en cm⁶ ; masse en kg/m ; périmètre en m²/m."""
    conv = {"A": 1e-2, "Avz": 1e-2, "Avy": 1e-2, "Iy": 1e-4, "Iz": 1e-4, "Iyz": 1e-4, "Iu": 1e-4, "Iv": 1e-4,
            "It": 1e-4, "Wel_y": 1e-3, "Wel_z": 1e-3, "Wpl_y": 1e-3, "Wpl_z": 1e-3, "Wel_u": 1e-3, "Wel_v": 1e-3,
            "Wt": 1e-3, "iy": 1e-1, "iz": 1e-1, "iu": 1e-1, "iv": 1e-1, "Iw": 1e-6, "perimetre": 1e-3}
    out = {}
    for k, v in p.items():
        if v is None:
            out[k] = None
        elif k in conv:
            out[k] = v * conv[k]
        else:
            out[k] = v
    if p.get("A") is not None:
        out["masse"] = p["A"] * 1e-6 * RHO_ACIER                       # mm² → m² × kg/m³ = kg/m
    return out

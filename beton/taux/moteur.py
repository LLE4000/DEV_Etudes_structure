# -*- coding: utf-8 -*-
"""Taux d'armature d'une poutre en béton armé — le MOTEUR : un QUANTITATIF (aucune vérification). Bibliothèque standard
seule ; la méthode est celle de la conception du 10/11/2026 (docs/site/LOT_Q_QUATRE_MODULES.md § 3.4), écrite et
prouvée dans docs/beton/TAUX_ARMATURE_MOTEUR.md. Un quantitatif ne doit JAMAIS sous-estimer : dans le doute sur une
longueur, c'est le côté qui commande le plus d'acier qui est pris, et la règle est dite.

LA MÉTHODE
- Béton : A_c = b·h + Σ talons ; V_c = A_c·L ; M_c = ρ_c·V_c.
- Masse linéique d'un acier : m = π·Ø²/4·ρ_s (ρ_s = 7 850 kg/m³ : les valeurs nominales de l'EN 10080 à moins de 0,5 %).
- Une ligne de BARRES (n barres, de x₁ à x₂, bornés aux abouts à l'enrobage : x₁ = max(de ; c), x₂ = min(à ; L − c)) :
      l = (x₂ − x₁) + r·l_r + n_rec·l_rec        r retours de l_r (vide : h − 2c, b − 2c pour la peau)
      n_rec = ⌈(l₀ − l_com) / (l_com − l_rec)⌉   si l₀ = (x₂ − x₁) + r·l_r dépasse la longueur commerciale l_com,
      l_rec = k·Ø                                (recouvrements comptés : l_com = 12 m, k = 50 par défaut)
  n_rec est le nombre de recouvrements qu'il FAUT pour couvrir l₀ avec des barres de l_com au plus, la longueur perdue
  dans chaque recouvrement comprise ; la conception écrivait ⌈l₀/l_com⌉ − 1, qui en oublie un quand l₀ approche d'un
  multiple de l_com (l₀ = 23,9 m, Ø 20 : deux barres de 12 m se recouvrant de 1 m ne couvrent que 23 m). Écart dit.
  Barres de peau : n par face, deux faces.
- Un cours d'ÉTRIERS est décrit par son TYPE (une liste de lignes) ; longueur développée d'une ligne, sans déduction
  des cintrages (côté sûr de quelques centimètres) :
      cadre             2·[(b − 2c) + (h − 2c)] + 2·l_r         le contour hors tout à l'enrobage
      cadre intérieur   2·[w + (h − 2c)] + 2·l_r                de largeur hors tout w
      épingle           (h − 2c) + 2·l_r
      cadre de talon    2·[(b_tot − 2c) + (h_t − 2c)] + 2·l_r   b_tot : largeur hors tout au niveau des talons
      l_r = max(10·Ø ; 70 mm)                                   EN 1992-1-1:2004 §8.5(2), figure 8.5 (coude à 90°)
- Nombre de cours d'une ZONE de longueur l_z au pas s : ⌈l_z/s − 10⁻⁹⌉, plus UN cours de fermeture par suite de zones
  jointives (``compter_etriers`` : la règle est écrite là, une fois).
- Quantités : masse par repère, par famille, par diamètre ; M_s ; M_s,maj = M_s·(1 + majoration) ;
  taux = M_s,maj / V_c (kg/m³) ; kg par mètre de poutre = M_s,maj / L.

SORTIE de ``calculer`` (JSON strict, nombres NON arrondis — l'écran formate) :
- ``reference``, ``poutre`` (normalisée) ;
- ``section`` (geometrie.section) ; ``beton`` : ``aire_cm2``, ``volume`` (m³), ``masse`` (kg), ``rho``, ``formules`` ;
- ``reperes`` : le BORDEREAU, un repère par ligne de barres (dans leur ordre) puis un par ligne de chaque type d'étriers
  EMPLOYÉ (cadre, cadre intérieur, épingle, cadre de talon) :
      {"repere" (1, 2, …), "cle" (« b0 », « t0.0 », « t0.talon » : la pièce des données et du dessin), "sorte"
       (« barre » | « etrier »), "famille" (inf, sup, renfort, peau, etriers), "designation", "position", "libelle"
       (« 3 Ø 20 »), "d", "n", "l_unitaire", "l_totale" (m), "masse_lineique" (kg/m), "masse" (kg), "part" (de M_s),
       "detail" {...}, "formules" [...]}
- ``zones`` : par zone (de gauche à droite) : ``index``, ``nom`` (« Z1 »), ``de``, ``a``, ``s``, ``type``, ``longueur``,
  ``quotient``, ``cours``, ``fermeture`` (0 | 1), ``n``, ``suite``, ``formules`` ; ``suites`` : les suites de zones
  jointives ; ``trous`` : les portions SANS étriers [[x₁, x₂], …] ;
- ``types`` : par type : ``index``, ``nom``, ``employe``, ``n`` (cours), ``zones``, ``resume`` (« cadre Ø 8 +
  épingle Ø 8 »), ``lignes``, ``talon``, ``l_cours`` (m d'acier par cours), ``masse_cours`` (kg) ;
- ``familles`` (les cinq, dans l'ordre : inférieures, supérieures, renforts, peau, étriers) et ``diametres`` : la
  synthèse (``longueur``, ``masse``, ``part``) ;
- ``totaux`` : ``longueur``, ``masse``, ``majoration`` (%), ``supplement``, ``masse_majoree``, ``taux``,
  ``taux_majore`` (kg/m³), ``kg_m``, ``kg_m_majore`` (kg/m), ``formules`` ;
- ``coupes`` : les COMPOSITIONS de la poutre (mêmes barres, même type d'étriers), de gauche à droite : ``cle`` (« A »),
  ``x`` (l'abscisse de la coupe), ``plages``, ``longueur``, ``barres`` (indices), ``reperes``, ``type``, ``pas``,
  ``geometrie`` (geometrie.coupe) ; ``coupes_omises`` ; ``elevation`` : ``niveaux`` (cm, par ligne de barres) ;
- ``avertissements`` : [{``code``, ``message``, ``grave``}] — ils ne changent aucun chiffre.

Chaque formule : {"cle", "symbole", "termes": [{"symbole", "valeur", "unite"}], "resultat", "unite"} — recalculable à
partir de ses seuls termes (tests/test_taux_moteur.py le fait).
"""
import csv
import io
import math
from typing import Dict, List, Tuple

from . import entree as EN
from . import geometrie as GE
from .entree import normaliser, nouvelle_poutre  # noqa: F401 — l'API

REFERENCE = "docs/site/LOT_Q_QUATRE_MODULES.md § 3.4 — conception du 10/11/2026"
# le retour d'un cadre ou d'une épingle : 10 Ø, 70 mm au moins — EN 1992-1-1:2004 §8.5(2), figure 8.5 (ancrage d'une
# armature transversale par un coude à 90° ; le crochet, lui, ne demande que 5 Ø ≥ 50 mm : 10 Ø est le côté sûr)
RETOUR_K = 10.0        # × Ø
RETOUR_MIN = 70.0      # mm
# le quotient l_z / s est arrondi à l'entier SUPÉRIEUR à 10⁻⁹ près : 0,9 / 0,1 = 9,000000000000002 en flottants compte
# 9 intervalles (pas 10), et 0,3 / 0,1 = 2,9999999999999996 en compte 3
ARRONDI = 1e-9
MAX_COUPES = 4
# sous cette largeur entre les nus du cadre de l'âme (cm), une épingle ne tient plus aucune barre intermédiaire : c'est
# un avertissement (aucun chiffre ne change), pas une règle de norme
LARGEUR_EPINGLE_UTILE = 20.0
FAMILLES = (("inf", "Inférieures"), ("sup", "Supérieures"), ("renfort", "Renforts"), ("peau", "Peau"), ("etriers", "Étriers"))
DESIGNATIONS_LIGNE = {"cadre": "Cadres", "cadre_interieur": "Cadres intérieurs", "epingle": "Épingles", "talon": "Cadres de talon"}
_UN = {"cadre": "cadre", "cadre_interieur": "cadre intérieur", "epingle": "épingle"}
_PLUSIEURS = {"cadre": "cadres", "cadre_interieur": "cadres intérieurs", "epingle": "épingles"}
EN_TETE_CSV = ("Repère", "Désignation", "Ø (mm)", "Nombre", "Longueur unitaire (m)", "Longueur totale (m)",
               "Masse linéique (kg/m)", "Masse (kg)")


# ============================================================ les règles, une fois chacune
def masse_lineique(d_mm: float, rho: float = EN.DEFAUTS["rho_acier"]) -> float:
    """La masse d'un mètre de barre Ø d (kg/m) : π·Ø²/4·ρ_s."""
    return math.pi * (d_mm / 1000.0) ** 2 / 4.0 * rho


def retour_etrier(d_mm: float) -> float:
    """Le retour d'un cadre ou d'une épingle (mm) : max(10·Ø ; 70 mm) — EN 1992-1-1 §8.5(2)."""
    return max(RETOUR_K * d_mm, RETOUR_MIN)


def cours(l_z: float, s_cm: float) -> int:
    """Le nombre de COURS d'une zone de longueur ``l_z`` (m) au pas ``s_cm`` : ⌈l_z/s − 10⁻⁹⌉ — autant d'intervalles
    de s au plus qu'il en faut pour couvrir la zone."""
    return int(math.ceil(l_z / (s_cm / 100.0) - ARRONDI))


def suites_de_zones(zones: List[dict]) -> List[List[int]]:
    """Les suites de zones JOINTIVES (la fin de l'une est le début de la suivante, à ``EPS_X`` près) : des listes
    d'indices de ``zones``, de gauche à droite."""
    suites: List[List[int]] = []
    for i in sorted(range(len(zones)), key=lambda k: (zones[k]["de"], zones[k]["a"])):
        if suites and zones[i]["de"] <= zones[suites[-1][-1]]["a"] + EN.EPS_X:
            suites[-1].append(i)
        else:
            suites.append([i])
    return suites


def compter_etriers(zones: List[dict]) -> List[dict]:
    """LA règle du nombre d'étriers, écrite une fois. ``zones`` : [{"de", "a" (m), "s" (cm)}], sans recouvrement.

    Chaque zone compte ⌈l_z/s − 10⁻⁹⌉ cours (ses intervalles) ; chaque SUITE de zones jointives en compte UN de plus,
    le cours de fermeture, porté par sa dernière zone : une suite de k intervalles a k + 1 étriers, et deux zones qui se
    touchent ne comptent pas deux fois l'étrier de leur jonction. C'est la pose réelle « un étrier à chaque bout de la
    suite et un tous les s au plus », un étrier posé à chaque jonction — jamais moins.

    Rend, dans l'ordre de ``zones`` : {"longueur", "quotient", "cours", "fermeture" (0 | 1), "n", "suite"}."""
    out: List[dict] = [{} for _ in zones]
    for k, suite in enumerate(suites_de_zones(zones)):
        for i in suite:
            z = zones[i]
            l_z = z["a"] - z["de"]
            n_c = cours(l_z, z["s"])
            f = 1 if i == suite[-1] else 0
            out[i] = {"longueur": l_z, "quotient": l_z / (z["s"] / 100.0), "cours": n_c, "fermeture": f, "n": n_c + f, "suite": k}
    return out


def recouvrements(l_0: float, d_mm: float, rec: dict) -> Tuple[int, float]:
    """(n_rec, l_rec) : le nombre de recouvrements d'une barre de longueur développée ``l_0`` (m) et la longueur de
    chacun, k·Ø (m). Sans l'option, ou si la barre tient dans une longueur commerciale : aucun.

    n_rec = ⌈(l₀ − l_com) / (l_com − l_rec)⌉ : avec n_rec + 1 barres de l_com au plus qui se recouvrent de l_rec, on
    couvre (n_rec + 1)·l_com − n_rec·l_rec ≥ l₀. (⌈l₀/l_com⌉ − 1 en compte un de moins quand l₀ approche d'un multiple
    de l_com : la longueur mangée par les recouvrements y est oubliée.)"""
    l_rec = rec["k"] * d_mm / 1000.0
    if not rec["actifs"] or l_0 <= rec["l_com"] + EN.EPS_X:
        return 0, l_rec
    return int(math.ceil((l_0 - rec["l_com"]) / (rec["l_com"] - l_rec) - ARRONDI)), l_rec


def contour_ligne(forme: str, sec: dict, largeur=None) -> float:
    """Le contour hors tout d'une ligne d'étriers, à l'enrobage, SANS ses retours (m) ; ``forme`` : cadre,
    cadre_interieur (de ``largeur`` cm), epingle, talon."""
    w, h = GE.contour_cadre(sec)
    if forme == "cadre":
        return 2.0 * (w + h) / 100.0
    if forme == "cadre_interieur":
        return 2.0 * (largeur + h) / 100.0
    if forme == "epingle":
        return h / 100.0
    if forme == "talon":
        wt, ht = GE.contour_talon(sec)
        return 2.0 * (wt + ht) / 100.0
    raise ValueError(f"forme de ligne inconnue : {forme!r}")


def longueur_ligne(forme: str, d_mm: float, sec: dict, largeur=None) -> float:
    """La longueur développée d'une ligne d'étriers (m) : son contour et deux retours de max(10·Ø ; 70 mm)."""
    return contour_ligne(forme, sec, largeur) + 2.0 * retour_etrier(d_mm) / 1000.0


# ============================================================ les formules, telles que le moteur les a calculées
def _f(cle: str, symbole: str, termes: list, resultat, unite: str) -> dict:
    return {"cle": cle, "symbole": symbole, "termes": [{"symbole": s, "valeur": v, "unite": u} for s, v, u in termes],
            "resultat": resultat, "unite": unite}


def _metres(x: float) -> str:
    """Une abscisse ou une longueur dans un message : « 1,50 m » (au centimètre, au millimètre s'il le faut)."""
    return (f"{x:.2f}" if abs(x * 100 - round(x * 100)) < 1e-6 else f"{x:.3f}").replace(".", ",") + " m"


# ============================================================ une ligne de barres
def _repere_barre(p: dict, i: int, b: dict, repere: int) -> dict:
    L, c, rho = p["L"], p["enrobage"], p["rho_acier"]
    c_m = c / 100.0
    x1, x2 = EN.etendue(b, L, c)
    l_d = x2 - x1
    peau, fil = b["famille"] == "peau", EN.filante(b)
    auto = b["l_retour"] is None
    cote, sym_cote = (p["b"], "b") if peau else (p["h"], "h")
    l_r = (cote - 2.0 * c) if auto else b["l_retour"]                 # cm
    r = b["retours"]
    l_0 = l_d + r * l_r / 100.0
    n_rec, l_rec = recouvrements(l_0, b["d"], p["recouvrements"])
    l_u = l_0 + n_rec * l_rec
    n = 2 * b["n"] if peau else b["n"]
    l_tot = n * l_u
    m = masse_lineique(b["d"], rho)
    masse = l_tot * m
    fx = [_f("x1", "x_1", [("de", 0.0 if b["de"] is None else b["de"], "m"), ("c", c_m, "m")], x1, "m"),
          _f("x2", "x_2", [("à", L if b["a"] is None else b["a"], "m"), ("L", L, "m"), ("c", c_m, "m")], x2, "m"),
          _f("l_droite", "l_d", [("x_2", x2, "m"), ("x_1", x1, "m")], l_d, "m")]
    if r and auto:
        fx.append(_f("l_retour_auto", "l_r", [(sym_cote, cote, "cm"), ("c", c, "cm")], l_r, "cm"))
    termes = [("l_d", l_d, "m")]
    if r:
        termes += [("r", r, ""), ("l_r", l_r, "cm")]
        fx.append(_f("l_dev", "l_0", list(termes), l_0, "m"))
    if n_rec:
        fx.append(_f("l_rec", "l_rec", [("k", p["recouvrements"]["k"], ""), ("Ø", b["d"], "mm")], l_rec, "m"))
        fx.append(_f("n_rec", "n_rec", [("l_0", l_0, "m"), ("l_com", p["recouvrements"]["l_com"], "m"), ("l_rec", l_rec, "m")], n_rec, ""))
        termes += [("n_rec", n_rec, ""), ("l_rec", l_rec, "m")]
    fx.append(_f("l_unitaire", "l", termes, l_u, "m"))
    if peau:
        fx.append(_f("n_peau", "n", [("n_face", b["n"], ""), ("faces", 2, "")], n, ""))
    fx += [_f("l_totale", "L", [("n", n, ""), ("l", l_u, "m")], l_tot, "m"),
           _f("m_lin", "m", [("Ø", b["d"], "mm"), ("ρ_s", rho, "kg/m³")], m, "kg/m"),
           _f("masse", "M", [("L", l_tot, "m"), ("m", m, "kg/m")], masse, "kg")]
    if peau:
        famille, designation = "peau", "Barres de peau"
    else:
        face = "inférieur" if b["famille"] == "inf" else "supérieur"
        famille = b["famille"] if fil else "renfort"
        designation = f"Lit {face} {b['lit']}" if fil else f"Renfort {face} (lit {b['lit']})"
    if fil:
        position = "filant"
    elif b["de"] is None:
        position = f"de l'about gauche à {_metres(b['a'])}"
    elif b["a"] is None:
        position = f"de {_metres(b['de'])} à l'about droit"
    else:
        position = f"de {_metres(b['de'])[:-2]} à {_metres(b['a'])}"
    return {"repere": repere, "cle": f"b{i}", "sorte": "barre", "famille": famille, "designation": designation, "position": position,
            "libelle": (f"2 × {EN.libelle_barres(b['n'], b['d'])}" if peau else EN.libelle_barres(b["n"], b["d"])),
            "d": b["d"], "n": n, "l_unitaire": l_u, "l_totale": l_tot, "masse_lineique": m, "masse": masse, "part": 0.0,
            "detail": {"index": i, "famille": b["famille"], "lit": b["lit"], "filant": fil, "de": b["de"], "a": b["a"], "x1": x1, "x2": x2,
                       "l_droite": l_d, "retours": r, "l_retour": l_r, "l_retour_auto": auto, "l_retours": r * l_r / 100.0, "l_dev": l_0,
                       "n_rec": n_rec, "l_rec": l_rec, "l_recouvrements": n_rec * l_rec},
            "formules": fx}


# ============================================================ une ligne d'un type d'étriers
def _ligne_etrier(p: dict, sec: dict, forme: str, d: float, largeur) -> dict:
    """Ce qu'une ligne vaut à CHAQUE cours : son contour, son retour, sa longueur développée, et les formules."""
    c = p["enrobage"]
    l_r = retour_etrier(d)
    contour = contour_ligne(forme, sec, largeur)
    l_u = contour + 2.0 * l_r / 1000.0
    if forme == "cadre":
        fc = _f("contour_cadre", "p", [("b", sec["b"], "cm"), ("h", sec["h"], "cm"), ("c", c, "cm")], contour, "m")
    elif forme == "cadre_interieur":
        fc = _f("contour_interieur", "p", [("w", largeur, "cm"), ("h", sec["h"], "cm"), ("c", c, "cm")], contour, "m")
    elif forme == "epingle":
        fc = _f("contour_epingle", "p", [("h", sec["h"], "cm"), ("c", c, "cm")], contour, "m")
    else:
        fc = _f("contour_talon", "p", [("b_tot", sec["b_tot"], "cm"), ("h_t", sec["h_talon"], "cm"), ("c", c, "cm")], contour, "m")
    return {"forme": forme, "d": d, "largeur": largeur, "contour": contour, "l_retour": l_r, "l_unitaire": l_u,
            "formules": [_f("l_r_etrier", "l_r", [("Ø", d, "mm")], l_r, "mm"), fc,
                         _f("l_etrier", "l", [("p", contour, "m"), ("l_r", l_r, "mm")], l_u, "m")]}


def _resume_type(t: dict, talon) -> str:
    """« cadre Ø 8 + 2 épingles Ø 8 + cadre de talon Ø 8 » : la composition d'un cours."""
    ordre, compte = [], {}
    for li in t["lignes"]:
        cle = (li["forme"], li["d"])
        if cle not in compte:
            compte[cle] = 0
            ordre.append(cle)
        compte[cle] += 1
    mots = [f"{_UN[f]} Ø {d}" if compte[(f, d)] == 1 else f"{compte[(f, d)]} {_PLUSIEURS[f]} Ø {d}" for f, d in ordre]
    if talon:
        mots.append(f"cadre de talon Ø {talon['d']}")
    return " + ".join(mots)


# ============================================================ les compositions de la poutre (les coupes)
def _compositions(p: dict) -> Tuple[List[dict], int]:
    """Les tronçons de même composition (mêmes lignes de barres, même type d'étriers), de gauche à droite : une coupe
    par composition, prise au milieu de sa plus longue plage. Au plus ``MAX_COUPES`` : les plus longues ; le nombre de
    celles qu'on ne montre pas est rendu."""
    L = p["L"]
    bornes = {0.0, L}
    for b in p["barres"]:
        bornes |= {x for x in (b["de"], b["a"]) if x is not None}
    for z in p["zones"]:
        bornes |= {z["de"], z["a"]}
    xs = sorted(bornes)
    par_cle: Dict[tuple, dict] = {}
    ordre = []
    for x1, x2 in zip(xs, xs[1:]):
        if x2 - x1 <= EN.EPS_X:
            continue
        xm = (x1 + x2) / 2.0
        presentes = tuple(i for i, b in enumerate(p["barres"])
                          if (b["de"] is None or b["de"] <= xm) and (b["a"] is None or xm <= b["a"]))
        zone = next((k for k, z in enumerate(p["zones"]) if z["de"] <= xm <= z["a"]), None)
        cle = (presentes, None if zone is None else p["zones"][zone]["type"])
        if cle not in par_cle:
            par_cle[cle] = {"barres": list(presentes), "type": cle[1], "plages": [], "pas": []}
            ordre.append(cle)
        q = par_cle[cle]
        if q["plages"] and abs(q["plages"][-1][1] - x1) <= EN.EPS_X:
            q["plages"][-1][1] = x2
        else:
            q["plages"].append([x1, x2])
        s = None if zone is None else p["zones"][zone]["s"]
        if s is not None and s not in q["pas"]:
            q["pas"].append(s)
    toutes = [par_cle[c] for c in ordre]
    for q in toutes:
        q["longueur"] = sum(b - a for a, b in q["plages"])
        # (la plus longue plage ; à longueurs égales — au micromètre —, la première : la coupe d'about est à gauche)
        a, b = max(q["plages"], key=lambda pl: (round(pl[1] - pl[0], 6), -pl[0]))
        q["x"] = (a + b) / 2.0
    gardees = sorted(sorted(toutes, key=lambda q: -q["longueur"])[:MAX_COUPES], key=lambda q: q["plages"][0][0])
    types = {t["nom"]: t for t in p["types_etriers"]}
    for k, q in enumerate(gardees):
        q["cle"] = "ABCDEFGH"[k]
        q["reperes"] = [i + 1 for i in q["barres"]]
        q["geometrie"] = GE.coupe(p, q["barres"], types.get(q["type"]))
    return gardees, len(toutes) - len(gardees)


def _niveaux_elevation(p: dict) -> List[list]:
    """Les ordonnées (cm depuis la fibre inférieure) où chaque ligne de barres se dessine en élévation : une pour un
    lit, une par barre d'une face pour la peau."""
    plus_gros = max(p["types_etriers"], key=GE.diametre_cadre, default=None)
    geo = GE.coupe(p, list(range(len(p["barres"]))), plus_gros)
    out = [[] for _ in p["barres"]]
    for bb in geo["barres"]:
        out[bb["index"]] = sorted({round(y, 6) for _, y in bb["points"]})
    return out


# ============================================================ le calcul
def calculer(poutre: dict) -> dict:
    """Le quantitatif d'une poutre : la sortie décrite en tête de ce fichier. ``poutre`` est normalisée d'abord
    (``ValueError`` hors du domaine)."""
    p = EN.normaliser(poutre)
    sec = GE.section(p)
    L, c, rho = p["L"], p["enrobage"], p["rho_acier"]
    # ---- le béton
    aire = sec["aire_cm2"]
    volume = aire / 1.0e4 * L
    masse_beton = volume * p["rho_beton"]
    termes_aire = [("b", sec["b"], "cm"), ("h", sec["h"], "cm")]
    if sec["bg"]:
        termes_aire += [("b_g", sec["bg"], "cm"), ("h_g", sec["hg"], "cm")]
    if sec["bd"]:
        termes_aire += [("b_d", sec["bd"], "cm"), ("h_d", sec["hd"], "cm")]
    beton = {"aire_cm2": aire, "volume": volume, "masse": masse_beton, "rho": p["rho_beton"],
             "formules": [_f("A_c", "A_c", termes_aire, aire, "cm²"),
                          _f("V_c", "V_c", [("A_c", aire, "cm²"), ("L", L, "m")], volume, "m³"),
                          _f("M_c", "M_c", [("V_c", volume, "m³"), ("ρ_c", p["rho_beton"], "kg/m³")], masse_beton, "kg")]}
    # ---- les barres
    reperes = [_repere_barre(p, i, b, i + 1) for i, b in enumerate(p["barres"])]
    # ---- les zones d'étriers, puis un repère par ligne de chaque type employé
    comptes = compter_etriers(p["zones"])
    zones = []
    for i, (z, k) in enumerate(zip(p["zones"], comptes)):
        zones.append({"index": i, "nom": EN.nom_zone(i), "de": z["de"], "a": z["a"], "s": z["s"], "type": z["type"], **k,
                      "formules": [_f("l_zone", "l_z", [("à", z["a"], "m"), ("de", z["de"], "m")], k["longueur"], "m"),
                                   _f("quotient", "q", [("l_z", k["longueur"], "m"), ("s", z["s"], "cm")], k["quotient"], ""),
                                   _f("cours", "n_c", [("q", k["quotient"], "")], k["cours"], ""),
                                   _f("n_zone", "n", [("n_c", k["cours"], ""), ("fermeture", k["fermeture"], "")], k["n"], "")]})
    suites = []
    for s in suites_de_zones(p["zones"]):
        suites.append({"de": p["zones"][s[0]]["de"], "a": p["zones"][s[-1]]["a"], "zones": list(s), "n": sum(zones[i]["n"] for i in s)})
    trous, x = [], 0.0
    for s in suites:
        if s["de"] > x + EN.EPS_X:
            trous.append([x, s["de"]])
        x = s["a"]
    if x < L - EN.EPS_X:
        trous.append([x, L])
    types = []
    for ti, t in enumerate(p["types_etriers"]):
        ses_zones = [z for z in zones if z["type"] == t["nom"]]
        n_t = sum(z["n"] for z in ses_zones)
        lignes = [_ligne_etrier(p, sec, li["forme"], li["d"], li["largeur"]) for li in t["lignes"]]
        talon = _ligne_etrier(p, sec, "talon", GE.diametre_talon(t), None) if (t["cadre_talon"] and EN.a_talon(p["forme"])) else None
        toutes = lignes + ([talon] if talon else [])
        types.append({"index": ti, "nom": t["nom"], "employe": bool(ses_zones), "n": n_t, "zones": [z["index"] for z in ses_zones],
                      "resume": _resume_type(t, talon), "lignes": lignes, "talon": talon, "cadre_talon": t["cadre_talon"],
                      "l_cours": sum(x_["l_unitaire"] for x_ in toutes),
                      "masse_cours": sum(x_["l_unitaire"] * masse_lineique(x_["d"], rho) for x_ in toutes)})
        if not ses_zones:
            continue
        f_n = _f("n_cours", "n", [(z["nom"], z["n"], "") for z in ses_zones], n_t, "")
        for j, li in enumerate(toutes):
            est_talon = li is talon
            m = masse_lineique(li["d"], rho)
            l_tot = n_t * li["l_unitaire"]
            reperes.append({
                "repere": len(reperes) + 1, "cle": f"t{ti}.talon" if est_talon else f"t{ti}.{j}", "sorte": "etrier", "famille": "etriers",
                "designation": f"{DESIGNATIONS_LIGNE[li['forme']]} {t['nom']}", "position": " + ".join(z["nom"] for z in ses_zones),
                "libelle": EN.libelle_barres(n_t, li["d"]), "d": li["d"], "n": n_t, "l_unitaire": li["l_unitaire"], "l_totale": l_tot,
                "masse_lineique": m, "masse": l_tot * m, "part": 0.0,
                "detail": {"type": t["nom"], "index_type": ti, "ligne": "talon" if est_talon else j, "forme": li["forme"],
                           "largeur": li["largeur"], "contour": li["contour"], "l_retour": li["l_retour"], "cours": n_t,
                           "zones": [z["index"] for z in ses_zones]},
                "formules": li["formules"] + [f_n, _f("l_totale", "L", [("n", n_t, ""), ("l", li["l_unitaire"], "m")], l_tot, "m"),
                                              _f("m_lin", "m", [("Ø", li["d"], "mm"), ("ρ_s", rho, "kg/m³")], m, "kg/m"),
                                              _f("masse", "M", [("L", l_tot, "m"), ("m", m, "kg/m")], l_tot * m, "kg")]})
    # ---- les quantités : par repère, par famille, par diamètre ; les totaux
    masse = sum(r["masse"] for r in reperes)
    longueur = sum(r["l_totale"] for r in reperes)
    for r in reperes:
        r["part"] = r["masse"] / masse if masse > 0 else 0.0
    familles = []
    for cle, lib in FAMILLES:
        rs = [r for r in reperes if r["famille"] == cle]
        m_f = sum(r["masse"] for r in rs)
        familles.append({"cle": cle, "libelle": lib, "reperes": [r["repere"] for r in rs], "libelles": [r["libelle"] for r in rs],
                         "longueur": sum(r["l_totale"] for r in rs), "masse": m_f, "part": m_f / masse if masse > 0 else 0.0})
    diametres = []
    for d in sorted({r["d"] for r in reperes}):
        rs = [r for r in reperes if r["d"] == d]
        l_d = sum(r["l_totale"] for r in rs)
        m = masse_lineique(d, rho)
        diametres.append({"d": d, "reperes": [r["repere"] for r in rs], "longueur": l_d, "masse_lineique": m, "masse": l_d * m,
                          "part": l_d * m / masse if masse > 0 else 0.0,
                          "formules": [_f("L_d", "L_Ø", [(f"L_{r['repere']}", r["l_totale"], "m") for r in rs], l_d, "m"),
                                       _f("m_lin", "m", [("Ø", d, "mm"), ("ρ_s", rho, "kg/m³")], m, "kg/m"),
                                       _f("masse", "M", [("L", l_d, "m"), ("m", m, "kg/m")], l_d * m, "kg")]})
    maj = p["majoration"]
    masse_maj = masse * (1.0 + maj / 100.0)
    totaux = {"longueur": longueur, "masse": masse, "majoration": maj, "supplement": masse_maj - masse, "masse_majoree": masse_maj,
              "taux": masse / volume, "taux_majore": masse_maj / volume, "kg_m": masse / L, "kg_m_majore": masse_maj / L,
              "formules": [_f("M_s", "M_s", [(f"M_{r['repere']}", r["masse"], "kg") for r in reperes], masse, "kg"),
                           _f("M_maj", "M_s,maj", [("M_s", masse, "kg"), ("maj", maj, "%")], masse_maj, "kg"),
                           _f("taux", "T", [("M_s,maj", masse_maj, "kg"), ("V_c", volume, "m³")], masse_maj / volume, "kg/m³"),
                           _f("taux_net", "T_0", [("M_s", masse, "kg"), ("V_c", volume, "m³")], masse / volume, "kg/m³"),
                           _f("kg_m", "m_L", [("M_s,maj", masse_maj, "kg"), ("L", L, "m")], masse_maj / L, "kg/m")]}
    coupes, omises = _compositions(p)
    return {"reference": REFERENCE, "poutre": p, "section": sec, "beton": beton, "reperes": reperes, "zones": zones, "suites": suites,
            "trous": trous, "types": types, "familles": familles, "diametres": diametres, "totaux": totaux, "coupes": coupes,
            "coupes_omises": omises, "elevation": {"niveaux": _niveaux_elevation(p)},
            "avertissements": _avertissements(p, sec, reperes, zones, types, trous)}


# ============================================================ ce qu'il faut dire (aucun chiffre ne change)
def _avertissements(p: dict, sec: dict, reperes: list, zones: list, types: list, trous: list) -> list:
    out = []

    def av(code, message, grave=False):
        out.append({"code": code, "message": message, "grave": grave})
    L, rec = p["L"], p["recouvrements"]
    if not p["barres"]:
        av("sans_barres", "Aucune barre longitudinale : le bordereau ne compte que les étriers.")
    if not zones:
        av("sans_etriers", f"Sans étriers sur toute la longueur (de 0 à {_metres(L)}) : aucune zone n'est décrite.", True)
    else:
        for x1, x2 in trous:
            av("sans_etriers", f"Sans étriers de {_metres(x1)[:-2]} à {_metres(x2)} ({_metres(x2 - x1)}) : aucune zone n'y est décrite.", True)
    for r in reperes:
        if r["sorte"] != "barre":
            continue
        d = r["detail"]
        if d["retours"] and d["l_retour"] == 0.0:
            av("retour_nul", f"Repère {r['repere']} ({r['libelle']}) : {d['retours']} retour{'s' if d['retours'] > 1 else ''} de 0 cm — "
               "saisir leur longueur (vide : toute la hauteur entre enrobages).", True)
        if not rec["actifs"] and d["l_dev"] > rec["l_com"] + EN.EPS_X:
            av("barre_longue", f"Repère {r['repere']} ({r['libelle']}) : {_metres(d['l_dev'])} par barre, plus que la longueur commerciale "
               f"({_metres(rec['l_com'])}) — les recouvrements ne sont pas comptés (option désactivée).", True)
    libre = sec["b"] - 2.0 * sec["c"]
    for t in types:
        if not t["employe"]:
            av("type_inemploye", f"Type {t['nom']} : aucune zone ne l'emploie — il ne compte pas.")
        if libre < LARGEUR_EPINGLE_UTILE and any(li["forme"] == "epingle" for li in t["lignes"]):
            av("epingle_inutile", f"Type {t['nom']} : une épingle dans une âme de {EN.nombre_fr(round(libre, 3))} cm entre les nus du cadre "
               f"(moins de {EN.nombre_fr(LARGEUR_EPINGLE_UTILE)} cm) — sans utilité, mais comptée.")
        if t["employe"] and EN.a_talon(p["forme"]) and not t["cadre_talon"]:
            av("talon_sans_cadre", f"Type {t['nom']} : pas de cadre de talon — les aciers transversaux des talons ne sont pas comptés "
               "(cocher « cadre de talon » s'ils sont ferraillés).")
        for li in t["lignes"] + ([t["talon"]] if t["talon"] else []):
            if t["employe"] and li["l_unitaire"] > rec["l_com"] + EN.EPS_X:
                av("cadre_long", f"Type {t['nom']} : un {('cadre de talon' if li['forme'] == 'talon' else _UN[li['forme']])} de "
                   f"{_metres(li['l_unitaire'])}, plus long que la longueur commerciale ({_metres(rec['l_com'])}) — à façonner en "
                   "plusieurs pièces, dont les recouvrements ne sont pas comptés.", True)
    return out


# ============================================================ la note entière, le bordereau en CSV
def synthese_note(resultats: List[dict]) -> dict:
    """La NOTE entière : la somme des volumes de béton et des masses d'acier des poutres calculées, et leur rapport."""
    volume = sum(r["beton"]["volume"] for r in resultats)
    masse = sum(r["totaux"]["masse"] for r in resultats)
    masse_maj = sum(r["totaux"]["masse_majoree"] for r in resultats)
    longueur = sum(r["poutre"]["L"] for r in resultats)
    return {"n": len(resultats), "volume": volume, "masse": masse, "masse_majoree": masse_maj, "longueur": longueur,
            "taux": masse / volume if volume > 0 else 0.0, "taux_majore": masse_maj / volume if volume > 0 else 0.0}


def _c(x: float, nd: int = 3) -> str:
    """Un nombre du CSV : ``nd`` décimales, la virgule."""
    return f"{x:.{nd}f}".replace(".", ",")


def bordereau_csv(res: dict) -> str:
    """Le bordereau d'une poutre calculée, en CSV pour un tableur français : « ; » entre les cases, la virgule décimale,
    une ligne d'en-tête (``EN_TETE_CSV``), un repère par ligne, puis la synthèse par diamètre (« Ø 8 »…), puis le total,
    la majoration et le total majoré. Longueurs au millimètre, masses au gramme ; une case de texte qui porte un « ; »
    ou un guillemet est mise entre guillemets (module ``csv``)."""
    sortie = io.StringIO()
    w = csv.writer(sortie, delimiter=";", lineterminator="\r\n")
    w.writerow(EN_TETE_CSV)
    for r in res["reperes"]:
        w.writerow([r["repere"], f"{r['designation']} · {r['position']}", r["d"], r["n"], _c(r["l_unitaire"]), _c(r["l_totale"]),
                    _c(r["masse_lineique"]), _c(r["masse"])])
    for d in res["diametres"]:
        w.writerow([f"Ø {d['d']}", "Synthèse par diamètre", d["d"], "", "", _c(d["longueur"]), _c(d["masse_lineique"]), _c(d["masse"])])
    t = res["totaux"]
    w.writerow(["Total", "Acier, sans majoration", "", "", "", _c(t["longueur"]), "", _c(t["masse"])])
    w.writerow(["Majoration", f"Chutes et ligatures, {EN.nombre_fr(t['majoration'])} %", "", "", "", "", "", _c(t["supplement"])])
    w.writerow(["Total majoré", "Acier à commander", "", "", "", "", "", _c(t["masse_majoree"])])
    return sortie.getvalue()

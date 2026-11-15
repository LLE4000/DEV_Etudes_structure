# -*- coding: utf-8 -*-
"""Fondation de grue — le PROJET : ce qu'on saisit (ses valeurs par défaut, sa relecture stricte) et le MODÈLE qu'on en
tire (le contour, le châssis posé, les surfaces d'appui, les plots, les bandes, les cas de charge, les contrôles).

    nouveau(nom)   -> un projet neuf, aux valeurs du bureau ; les réactions et les textes du projet sont VIDES
    lire(el)       -> le projet relu : chaque valeur contrôlée (ValueError avec un message court sinon)
    modele(el)     -> tout ce qui se déduit du projet, pour le calcul, le dessin, le SAF et la note

Unités : m, kN, MN/m³ ; la limite de contrainte de contact en kg/cm² (1 kg/cm² = 98,0665 kPa).
Repère : l'origine est au coin bas-gauche de la boîte de la dalle ; la position de la grue (ex, ey) est celle du centre du
châssis par rapport au CENTRE de cette boîte — la grue garde ses cotes quand la dalle change de taille.
"""
import copy
import math
from typing import Dict, List, Optional, Tuple

from beton import position as POS_BETON
from beton.poutre.entree import DEFAUTS_POUTRE

from . import chassis as CH
from . import geometrie as G
from . import positions as PO

BETONS = ("C20/25", "C25/30", "C30/37", "C35/45", "C40/50", "C45/55", "C50/60")
NU = 0.2
GAMMA_BETON = 25.0               # kN/m³
K_DEFAUT = ((2.0, True), (5.0, True), (10.0, True), (20.0, False), (30.0, False))   # (k en MN/m³, coché « SCIA »)
K_MAX_VALEURS, JEUX_MAX, SOMMETS_MAX = 8, 6, 40
# le nom d'une fondation : 60 caractères au plus, la borne des autres modules du site (il titre la note et le fichier)
NOM_MAX = 60
ANNEAUX = (0.01, 0.02, 0.03)     # la largeur de l'anneau D2 du SAF (m)
LARGEURS_BANDE = (1.0, 0.5)
MAILLES = (0.10, 0.50)
PLOTS = ("aucun", "prefabrique", "coule")
PLOT_DEFAUT = (2.0, 2.0, 0.35)   # a × b × hauteur (m) : les plots COULÉS EN PLACE (cahier du lot 4, § 8.2)
# les dimensions par défaut de chaque sorte de plot (le propriétaire, 15/11/2026 : « les plots préfabriqués font 2,50 m par
# 2,50 m fois 35 cm de haut ») ; choisir une sorte pose ses dimensions tant que le plot garde celles d'une autre sorte
PLOTS_DEFAUT = {"prefabrique": (2.5, 2.5, 0.35), "coule": PLOT_DEFAUT}
FORMES = ("rectangle", "contour")
# la vue du centre (lot 4, § 2.3) : l'onglet, la carte (« Enveloppe » quand ni le cas ni le k ne sont choisis)
ONGLETS = ("plan", "ferraillage", "coupes")
CARTES = ("sigma", "w", "m_inf_x", "m_inf_y", "m_sup_x", "m_sup_y", "aucune")
# lot 4 (cahier des charges du 15/11/2026) : la maille et les bandes par défaut, l'unité des contraintes de sol, les armatures
MAILLE_DEFAUT = 0.20             # § 3.1 (0,25 m avant le lot 4 ; une note enregistrée garde la sienne)
LARGEUR_BANDE_DEFAUT = 0.5       # § 4.3 (1,00 m avant le lot 4)
UNITES = ("kgcm2", "kpa", "mpa")  # § 2.4 : kg/cm² par défaut ; le calcul reste en kPa, seul l'affichage change
TREILLIS = ("auto", "10-15", "12-15", "10-10", "12-10")   # § 5.2 : Ø (mm) - espacement (cm), les mêmes dans les deux sens
ENROBAGE_DEFAUT = 0.05           # § 5.1 (m)
RECOUVREMENT_DEFAUT = 50         # § 5.7 : longueur de recouvrement = 50 Ø, quand le logiciel n'a pas de règle
L_BARRE_MAX = 12.0               # § 5.7 : longueur maxi des barres (m)
# les options du MOTEUR POUTRE (§ 2.1, bloc 4), à leurs valeurs par défaut du module « Poutres béton armé » — lues chez lui
GAMMA_S_DEFAUT = float(DEFAUTS_POUTRE["gamma_s"])             # « coefficient acier ELS » : fyd = fyk / γs
JEU_DEFAUT, ARRONDI_DEFAUT = POS_BETON.JEU_POSE, POS_BETON.ARRONDI   # cm : le jeu de pose, le pas d'arrondi de l'axe


def module_E(classe: str) -> float:
    """E d'une classe de béton (MPa) : 22·((f_ck + 8)/10)^0,3 GPa, arrondi à 100 MPa (C30/37 : 32 800, la valeur du
    fichier SAF du bureau)."""
    fck = float(classe[1:].split("/")[0])
    return round(22000.0 * ((fck + 8.0) / 10.0) ** 0.3 / 100.0) * 100.0


def nouveau(nom: str = "F1") -> Dict:
    """Un projet neuf : les valeurs par défaut du bureau ; les textes du projet et les réactions restent vides."""
    return {
        "nom": nom,
        "projet": {"projet": "", "grue": "", "creation": "", "mise_a_jour": "", "indice": ""},
        "dalle": {"forme": "rectangle", "Lx": 8.0, "Ly": 8.0, "sommets": [], "h": 0.50, "beton": "C30/37"},
        "sol": {"k": [{"k": k, "scia": s} for k, s in K_DEFAUT], "reprise": None},
        "grue": {"chassis": CH.SAISIE, "a": CH.ENTRAXE_DEFAUT[0], "b": CH.ENTRAXE_DEFAUT[1], "appuis": "rectangle",
                 "rect": {"a": CH.RECT_DEFAUT[0], "b": CH.RECT_DEFAUT[1], "angle": 0},
                 "plots": {"type": "aucun", "a": PLOT_DEFAUT[0], "b": PLOT_DEFAUT[1], "h": PLOT_DEFAUT[2]},
                 "ex": 0.0, "ey": 0.0, "beta": 0.0, "dmin": 0.50, "repere": None},
        "reactions": [{"nom": "", "P": [None, None, None, None]}],
        "bandes": {"largeur": LARGEUR_BANDE_DEFAUT},
        "limite": 0.5,
        "armatures": {"enrobage": ENROBAGE_DEFAUT, "treillis_inf": "auto", "treillis_sup": "auto"},
        "options": {"maille": MAILLE_DEFAUT, "anneau": 0.02, "c1x": 0.1, "c1y": 0.1, "c2x": 0.1, "c2y": 0.1, "rond": "carre",
                    "aretes": False, "nonlinear": False, "h_min": 0.30, "h_max": 1.20, "h_pas": 0.10,
                    "unite": "kgcm2", "torsion": True, "recouvrement": RECOUVREMENT_DEFAUT, "l_max": L_BARRE_MAX,
                    "gamma_s": GAMMA_S_DEFAUT, "jeu": JEU_DEFAUT, "arrondi": ARRONDI_DEFAUT},
        "vue": {"onglet": "plan", "carte": "sigma", "cas": None, "k": None},
        "proposition": None,
    }


# ------------------------------------------------------------------ la relecture
def _nombre(v, nom: str, mini: float, maxi: float, strict: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{nom} : un nombre est attendu")
    x = float(v)
    if x < mini or x > maxi or (strict and x <= mini):
        raise ValueError(f"{nom} doit être entre {_fr(mini)} et {_fr(maxi)}")
    return x


def _fr(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def _texte(v, maxi: int = 120) -> str:
    return " ".join(str(v or "").split())[:maxi]


def lire(el: Dict) -> Dict:
    """Le projet relu, complété de ses valeurs par défaut ; chaque valeur est contrôlée."""
    b = nouveau(_texte((el or {}).get("nom"), NOM_MAX) or "F1")
    el = el or {}
    for k in b["projet"]:
        b["projet"][k] = _texte((el.get("projet") or {}).get(k))
    d = el.get("dalle") or {}
    if d.get("forme", "rectangle") not in FORMES:
        raise ValueError("forme de dalle inconnue")
    b["dalle"]["forme"] = d.get("forme", "rectangle")
    b["dalle"]["Lx"] = _nombre(d.get("Lx", 8.0), "L_x", 1.0, 60.0)
    b["dalle"]["Ly"] = _nombre(d.get("Ly", 8.0), "L_y", 1.0, 60.0)
    b["dalle"]["h"] = _nombre(d.get("h", 0.5), "l'épaisseur", 0.10, 3.0)
    if d.get("beton", "C30/37") not in BETONS:
        raise ValueError("classe de béton inconnue")
    b["dalle"]["beton"] = d.get("beton", "C30/37")
    sommets = d.get("sommets") or []
    if len(sommets) > SOMMETS_MAX:
        raise ValueError(f"le contour a {SOMMETS_MAX} sommets au plus")
    b["dalle"]["sommets"] = [[_nombre(p[0], "un sommet", -200.0, 200.0), _nombre(p[1], "un sommet", -200.0, 200.0)] for p in sommets]
    if b["dalle"]["forme"] == "contour":
        _controler_contour(b["dalle"]["sommets"])
    s = el.get("sol") or {}
    ks = s.get("k") if s.get("k") is not None else b["sol"]["k"]
    if not ks or len(ks) > K_MAX_VALEURS:
        raise ValueError(f"de 1 à {K_MAX_VALEURS} valeurs de k")
    b["sol"]["k"] = [{"k": _nombre(x.get("k"), "k", 0.0, 500.0, strict=True), "scia": bool(x.get("scia"))} for x in ks]
    if len({x["k"] for x in b["sol"]["k"]}) != len(b["sol"]["k"]):
        raise ValueError("deux valeurs de k sont égales")
    b["sol"]["reprise"] = copy.deepcopy(s.get("reprise")) if isinstance(s.get("reprise"), dict) else None
    g = el.get("grue") or {}
    bg = b["grue"]
    ident = g.get("chassis", CH.SAISIE)
    if ident != CH.SAISIE and not CH.existe(ident):
        raise ValueError("ce châssis n'est pas dans la bibliothèque")
    bg["chassis"] = ident
    bg["a"] = _nombre(g.get("a", bg["a"]), "l'entraxe a", 1.0, 20.0)
    bg["b"] = _nombre(g.get("b", bg["b"]), "l'entraxe b", 1.0, 20.0)
    if g.get("appuis", "reels") not in ("reels", "rectangle"):
        raise ValueError("appuis : réels ou rectangle")
    bg["appuis"] = g.get("appuis", "reels")
    r = g.get("rect") or {}
    bg["rect"] = {"a": _nombre(r.get("a", CH.RECT_DEFAUT[0]), "la plaque a", 0.05, 3.0),
                  "b": _nombre(r.get("b", CH.RECT_DEFAUT[1]), "la plaque b", 0.05, 3.0),
                  "angle": r.get("angle") if r.get("angle") in (0, 45) else None}
    p = g.get("plots") or {}
    if p.get("type", "aucun") not in PLOTS:
        raise ValueError("plots : aucun, préfabriqués ou coulés en place")
    bg["plots"] = {"type": p.get("type", "aucun"), "a": _nombre(p.get("a", PLOT_DEFAUT[0]), "le plot a", 0.20, 10.0),
                   "b": _nombre(p.get("b", PLOT_DEFAUT[1]), "le plot b", 0.20, 10.0),
                   "h": _nombre(p.get("h", PLOT_DEFAUT[2]), "la hauteur du plot", 0.05, 2.0)}
    # le projet garde ce qui VAUT : un châssis saisi, ou sans platine dans la bibliothèque, n'a que des appuis
    # « rectangle » ; un rectangle sans orientation prend celle de la platine réelle (0° pour un châssis saisi). Ce que
    # l'écran montre est ainsi ce que la note porte.
    ch = CH.lire(bg)
    bg["appuis"] = ch["appuis"]
    if ch["appuis"] == "rectangle":
        bg["rect"]["angle"] = ch["angle"]
    bg["ex"] = _nombre(g.get("ex", 0.0), "e_x", -100.0, 100.0)
    bg["ey"] = _nombre(g.get("ey", 0.0), "e_y", -100.0, 100.0)
    bg["beta"] = _nombre(g.get("beta", 0.0), "l'angle β", -360.0, 360.0)
    bg["dmin"] = _nombre(g.get("dmin", 0.5), "la distance mini", 0.0, 5.0)
    # le repère des cotes du châssis : les axes de la dalle (None), ou un bord du contour (son rang)
    n_bords = 4 if b["dalle"]["forme"] == "rectangle" else len(b["dalle"]["sommets"])
    r = g.get("repere")
    bg["repere"] = r if isinstance(r, int) and not isinstance(r, bool) and 0 <= r < n_bords else None
    jeux = el.get("reactions") if el.get("reactions") else b["reactions"]
    if len(jeux) > JEUX_MAX:
        raise ValueError(f"{JEUX_MAX} jeux de réactions au plus")
    b["reactions"] = []
    for j in jeux:
        P = list((j or {}).get("P") or [None] * 4)[:4] + [None] * 4
        b["reactions"].append({"nom": _texte((j or {}).get("nom"), 40),
                               "P": [None if v is None or v == "" else _nombre(v, "une réaction", 0.0, 1e5) for v in P[:4]]})
    lb = (el.get("bandes") or {}).get("largeur", LARGEUR_BANDE_DEFAUT)
    if lb not in LARGEURS_BANDE:
        raise ValueError("largeur des bandes : 1,00 ou 0,50 m")
    b["bandes"]["largeur"] = float(lb)
    b["limite"] = _nombre(el.get("limite", 0.5), "la limite", 0.0, 5.0, strict=True)
    o = el.get("options") or {}
    bo = b["options"]
    bo["maille"] = _nombre(o.get("maille", MAILLE_DEFAUT), "la maille", MAILLES[0], MAILLES[1])
    if round(float(o.get("anneau", 0.02)), 3) not in ANNEAUX:
        raise ValueError("anneau : 1, 2 ou 3 cm")
    bo["anneau"] = round(float(o.get("anneau", 0.02)), 3)
    for c in ("c1x", "c1y", "c2x", "c2y"):
        bo[c] = _nombre(o.get(c, 0.1), c.upper(), 0.0, 1.0)
    if o.get("rond", "carre") not in CH.RONDS:
        raise ValueError("platine ronde : carré de même aire ou 48 côtés")
    bo["rond"] = o.get("rond", "carre")
    bo["aretes"], bo["nonlinear"] = bool(o.get("aretes")), bool(o.get("nonlinear"))
    bo["h_min"] = _nombre(o.get("h_min", 0.30), "l'épaisseur mini testée", 0.10, 3.0)
    bo["h_max"] = _nombre(o.get("h_max", 1.20), "l'épaisseur maxi testée", bo["h_min"], 3.0)
    bo["h_pas"] = _nombre(o.get("h_pas", 0.10), "le pas des épaisseurs", 0.05, 1.0)
    if o.get("unite", "kgcm2") not in UNITES:
        raise ValueError("unité des contraintes de sol : kg/cm², kPa ou MPa")
    bo["unite"] = o.get("unite", "kgcm2")
    bo["torsion"] = bool(o.get("torsion", True))
    bo["recouvrement"] = _nombre(o.get("recouvrement", RECOUVREMENT_DEFAUT), "la longueur de recouvrement", 20.0, 120.0)
    bo["l_max"] = _nombre(o.get("l_max", L_BARRE_MAX), "la longueur maxi des barres", 3.0, 24.0)
    bo["gamma_s"] = _nombre(o.get("gamma_s", GAMMA_S_DEFAUT), "γs", 1.0, 3.0)
    bo["jeu"] = _nombre(o.get("jeu", JEU_DEFAUT), "le jeu de pose", 0.0, 5.0)
    if float(o.get("arrondi", ARRONDI_DEFAUT)) not in POS_BETON.ARRONDIS:
        raise ValueError("arrondi de l'axe : 0, 0,5 ou 1 cm")
    bo["arrondi"] = float(o.get("arrondi", ARRONDI_DEFAUT))
    a = el.get("armatures") or {}
    ba = b["armatures"]
    ba["enrobage"] = _nombre(a.get("enrobage", ENROBAGE_DEFAUT), "l'enrobage", 0.01, 0.15)
    for face in ("treillis_inf", "treillis_sup"):
        if a.get(face, "auto") not in TREILLIS:
            raise ValueError("treillis : automatique, Ø10 ou Ø12, tous les 10 ou 15 cm")
        ba[face] = a.get(face, "auto")
    v = el.get("vue") or {}
    b["vue"] = {"onglet": v.get("onglet") if v.get("onglet") in ONGLETS else "plan",
                "carte": v.get("carte") if v.get("carte") in CARTES else "sigma",
                "cas": v.get("cas") if isinstance(v.get("cas"), str) else None,
                "k": float(v["k"]) if isinstance(v.get("k"), (int, float)) and not isinstance(v.get("k"), bool) else None}
    b["proposition"] = copy.deepcopy(el.get("proposition")) if isinstance(el.get("proposition"), dict) else None
    return b


def _controler_contour(sommets: List) -> None:
    if len(sommets) < 3:
        raise ValueError("le contour a au moins 3 sommets")
    pts = [tuple(p) for p in sommets]
    if G.aire(pts) < 1.0:
        raise ValueError("le contour est trop petit (moins de 1 m²)")
    if G.se_recoupe(pts):
        raise ValueError("le contour se recoupe")
    xmin, ymin, xmax, ymax = G.boite(pts)
    if xmax - xmin > 60.0 or ymax - ymin > 60.0:
        raise ValueError("le contour dépasse 60 m")


# ------------------------------------------------------------------ le modèle
def contour(el: Dict) -> List:
    """Le contour de la dalle, sens trigonométrique, origine au coin bas-gauche de sa boîte ; un rectangle commence à ce
    coin, un contour dessiné garde son premier sommet."""
    d = el["dalle"]
    if d["forme"] == "rectangle":
        return [(0.0, 0.0), (d["Lx"], 0.0), (d["Lx"], d["Ly"]), (0.0, d["Ly"])]
    pts = G.sens_trigo([tuple(p) for p in d["sommets"]])
    xmin, ymin, _, _ = G.boite(pts)
    return [(x - xmin, y - ymin) for x, y in pts]


def jeux_complets(el: Dict) -> List[Dict]:
    """Les jeux de réactions COMPLETS (leurs quatre valeurs saisies), avec leur numéro dans la liste (1…)."""
    return [{"numero": i, "nom": j["nom"], "P": tuple(float(v) for v in j["P"])}
            for i, j in enumerate(el["reactions"], 1) if all(v is not None for v in j["P"])]


def k_scia(el: Dict) -> List[float]:
    return [x["k"] for x in el["sol"]["k"] if x["scia"]]


def epaisseurs(el: Dict) -> List[float]:
    """Les épaisseurs que « Proposer » essaie."""
    o = el["options"]
    n = int(math.floor((o["h_max"] - o["h_min"]) / o["h_pas"] + 1e-9))
    return [round(o["h_min"] + i * o["h_pas"], 3) for i in range(n + 1)]


def modele(el: Dict) -> Dict:
    """Tout ce qui se déduit du projet (voir l'en-tête du fichier) :
    {"contour", "h", "E", "nu", "maille", "gamma", "beton", "centre" (du châssis), "beta", "chassis" (chassis.lire),
     "pieds" [P1…P4] (les axes d'appui), "plaques" [P1…P4] (les surfaces d'appui posées), "centres" [P1…P4],
     "plots": None | {"type", "polys", "hauteur", "a", "b", "diffusion" (préfabriqué : [P1…P4] les carrés de diffusion
     à 45°, où arrive chaque réaction)}, "bandes", "jeux", "cas", "carre",
     "controles": {"plaques_dedans" [bool × 4], "distance" (mini plaque–bord), "distances" [× 4], "distance_ok",
                   "plots_dedans" [bool × 4], "plots_separes", "plots_couvrent" [bool × 4], "plots_ok"}}"""
    c = contour(el)
    g = el["grue"]
    xmin, ymin, xmax, ymax = G.boite(c)
    cx, cy = (xmin + xmax) / 2 + g["ex"], (ymin + ymax) / 2 + g["ey"]
    ch = CH.lire(g, el["options"]["rond"])
    plaques = CH.poser(ch["surfaces"], cx, cy, g["beta"])
    centres = G.tourner(cx, cy, ch["centres"], g["beta"])
    pieds = G.pieds(cx, cy, ch["a"], ch["b"], g["beta"])
    plots = None
    if g["plots"]["type"] != "aucun":
        p = g["plots"]
        # (le plot est centré sur le centre de GRAVITÉ de la surface d'appui : la règle de la référence du lot 4)
        polys = [G.rectangle(*G.centre_gravite(s), p["a"], p["b"], g["beta"]) for s in plaques]
        plots = {"type": p["type"], "polys": polys, "hauteur": p["h"], "a": p["a"], "b": p["b"]}
        # préfabriqué : la réaction arrive sur le carré de diffusion à 45° (lot 4, § 3.3) — le calcul ET le SAF
        if p["type"] == "prefabrique":
            plots["diffusion"] = [G.diffusion_45(s, p["h"], p["a"], p["b"], g["beta"]) for s in plaques]
    jeux = jeux_complets(el)
    cas = PO.cas_de_charge([j["P"] for j in jeux], ch["carre"], [j["numero"] for j in jeux])
    for x in cas:                                                # le jeu d'un cas : son numéro dans la liste saisie
        x["jeu_saisi"] = jeux[x["jeu"] - 1]["numero"]
    distances = [G.distance_au_bord(c, pl) for pl in plaques]
    dedans = [G.dans_dalle(c, pl) for pl in plaques]
    ctl = {"plaques_dedans": dedans, "distances": distances, "distance": min(distances),
           "distance_ok": all(dedans) and min(distances) >= g["dmin"] - G.TOLERANCE}
    if plots:
        ctl["plots_dedans"] = [G.dans_dalle(c, q) for q in plots["polys"]]
        ctl["plots_separes"] = not any(G.se_chevauchent(plots["polys"][i], plots["polys"][j]) for i in range(4) for j in range(i + 1, 4))
        ctl["plots_couvrent"] = [G.dans_dalle(q, pl) for q, pl in zip(plots["polys"], plaques)]
        ctl["plots_ok"] = all(ctl["plots_dedans"]) and ctl["plots_separes"] and all(ctl["plots_couvrent"])
    return {"contour": c, "h": el["dalle"]["h"], "E": module_E(el["dalle"]["beton"]), "nu": NU, "maille": el["options"]["maille"],
            "gamma": GAMMA_BETON, "beton": el["dalle"]["beton"], "centre": (cx, cy), "beta": g["beta"], "chassis": ch,
            "pieds": pieds, "plaques": plaques, "centres": centres, "plots": plots,
            "bandes": G.bandes_integration(c, centres, g["beta"], el["bandes"]["largeur"]),
            "jeux": jeux, "cas": cas, "carre": ch["carre"], "controles": ctl}


def cotes_au_bord(el: Dict, m: Dict) -> Optional[Dict]:
    """Les deux cotes du centre du châssis par rapport au bord pris pour repère : {"bord", "a", "b" (ses extrémités),
    "pied" (le pied de la perpendiculaire abaissée du centre), "perp" (la distance au bord, comptée vers l'intérieur de
    la dalle), "long" (du début du bord au pied)} ; None quand le repère est les axes de la dalle."""
    i = el["grue"].get("repere")
    if i is None:
        return None
    c = m["contour"]
    a, b = c[i], c[(i + 1) % len(c)]
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    if L < 1e-9:
        return None
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    cx, cy = m["centre"]
    long_ = (cx - a[0]) * ux + (cy - a[1]) * uy
    perp = -(cx - a[0]) * uy + (cy - a[1]) * ux               # contour dans le sens trigonométrique : l'intérieur est à gauche
    return {"bord": i, "a": a, "b": b, "pied": (a[0] + long_ * ux, a[1] + long_ * uy), "perp": perp, "long": long_, "u": (ux, uy)}


def position_au_bord(el: Dict, m: Dict, perp: Optional[float] = None, long_: Optional[float] = None) -> Tuple[float, float]:
    """(e_x, e_y) du châssis posé à la distance ``perp`` du bord pris pour repère et à ``long_`` de son début (une
    valeur absente garde celle d'aujourd'hui)."""
    r = cotes_au_bord(el, m)
    if r is None:
        raise ValueError("aucun bord n'est pris pour repère")
    perp = r["perp"] if perp is None else perp
    long_ = r["long"] if long_ is None else long_
    ux, uy = r["u"]
    x, y = r["a"][0] + long_ * ux - perp * uy, r["a"][1] + long_ * uy + perp * ux
    x0, y0, x1, y1 = G.boite(m["contour"])
    return x - (x0 + x1) / 2, y - (y0 + y1) / 2


def agrandi(el: Dict, d: float) -> Optional[Dict]:
    """Le projet dont la dalle est agrandie de ``d`` de chaque côté (« Proposer ») : un rectangle gagne 2·d sur L_x et
    L_y, un contour est décalé de d vers l'extérieur ; la grue garde ses cotes par rapport au centre. None si le contour
    décalé n'est plus valable."""
    e = copy.deepcopy(el)
    if e["dalle"]["forme"] == "rectangle":
        e["dalle"]["Lx"] += 2 * d
        e["dalle"]["Ly"] += 2 * d
        return e
    pts = G.decaler(G.sens_trigo([tuple(p) for p in e["dalle"]["sommets"]]), d)
    if G.se_recoupe(pts):
        return None
    e["dalle"]["sommets"] = [[x, y] for x, y in pts]
    return e

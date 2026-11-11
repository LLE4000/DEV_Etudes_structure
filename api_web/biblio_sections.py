# -*- coding: utf-8 -*-
"""La rubrique « SECTIONS » de la bibliothèque « Autres » : les caractéristiques d'une SECTION QUELCONQUE, composée à
l'écran (docs/site/BIBLIOTHEQUE_AUTRES.md § 5). Le moteur est rdm/sections.py ; ce fichier LIT les paramètres, COMPOSE
l'éditeur — le dessin à l'échelle, où un élément se clique et se GLISSE ; la liste des éléments et la fiche de celui qui
est choisi ; les résultats ; la feuille de calcul (le tableau de Huygens-Steiner) — et FORMATE.

Paramètres plats (une adresse se partage) : n, sel, puis pour chaque élément i : k{i} (profil, plat, rond, poly), ses
dimensions (f{i} p{i} | b{i} h{i} | d{i} | pn{i} py{i}_{j} pz{i}_{j}), sa position y{i} z{i} (mm), sa rotation r{i} (°),
m{i} (miroir), c{i} cc{i} (coupe : le côté gardé et la position de la coupe), v{i} (évidement). g : la grandeur de la
bande.

La POSITION d'un élément est celle du centre de son rectangle enveloppe, avant coupe (un polygone : son origine locale).
Un élément ajouté sans position se pose de lui-même SUR la section déjà dessinée.
"""
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from acier.profiles import catalogue as KA
from api_web import biblio_autres as A
from api_web import biblio_beton as B
from api_web import commun as C
from rdm import sections as S

n, r, m = A.n, A.r, A.m
N_MAX = 10
SOMMETS_MAX = 10
SORTES = (("profil", "Profilé"), ("plat", "Plat"), ("rond", "Rond"), ("poly", "Polygone"))
COUPES = (("non", "entier"), ("haut", "garder le haut"), ("bas", "garder le bas"), ("gauche", "garder la gauche"), ("droite", "garder la droite"))
ROTATIONS = ((0, "0°"), (90, "90°"), (180, "180°"), (270, "270°"))
GRANDEURS = ("Iy", "Iz", "A", "W")
FAMILLES = [f for f, _ in KA.familles_presentes()]
RHO_ACIER = 7850.0
CLES_ELEMENT = ("k", "f", "p", "b", "h", "d", "y", "z", "r", "m", "c", "cc", "v", "pn")
POLY_DEFAUT = ((-50.0, -40.0), (50.0, -40.0), (50.0, 40.0), (-50.0, 40.0))
SOURCES = ("Intégrales de Green sur les contours polygonaux, théorème de Huygens-Steiner, axes principaux d'inertie : tout cours de résistance des matériaux "
           "(Timoshenko, <i>Résistance des matériaux</i>, t. 1, annexe ; <i>Techniques de l'ingénieur</i> C 2 060).",
           "Profilés : contours et valeurs du catalogue de la Bibliothèque Acier (congés et pentes compris ; arcs discrétisés, 36 segments par congé, "
           "180 par cercle).",
           "Module plastique : axe d'égales aires, section d'un seul matériau (EN 1993-1-1 §6.2.5 pour son emploi).")
VOLET = [
    ("Méthode", ["Chaque élément est un contour polygonal ; les intégrales de Green donnent exactement son aire, son centre de gravité et ses inerties propres.",
                 "Centre de gravité : y<sub>G</sub> = Σ A<sub>i</sub>·y<sub>i</sub> / Σ A<sub>i</sub> (de même pour z).",
                 "Huygens-Steiner : I<sub>y</sub> = Σ (I<sub>y,i</sub> + A<sub>i</sub>·(z<sub>i</sub> − z<sub>G</sub>)²), et de même I<sub>z</sub> et I<sub>yz</sub>.",
                 "Axes principaux : tan 2α = −2·I<sub>yz</sub>/(I<sub>y</sub> − I<sub>z</sub>) ; I<sub>1,2</sub> = (I<sub>y</sub> + I<sub>z</sub>)/2 ± √(((I<sub>y</sub> − I<sub>z</sub>)/2)² + I<sub>yz</sub>²).",
                 "Un profilé du catalogue non coupé retrouve ses valeurs du catalogue : c'est la méthode même qui l'a calculé."]),
    ("Conventions", ["y horizontal (vers la droite), z vertical (vers le haut). I<sub>y</sub> = ∫z² dA : la flexion autour de l'axe horizontal.",
                     "La position d'un élément est celle du centre de son rectangle enveloppe, avant coupe ; la rotation tourne autour de ce point, "
                     "dans le sens trigonométrique.",
                     "α est l'angle de l'axe principal 1 (inertie maximale) par rapport à y, positif dans le sens trigonométrique."]),
    ("Ce qui n'est pas calculé", list(S.NON_CALCULES.values())),
    ("Limites", ["Un seul matériau : pas d'homogénéisation (acier + bois, béton armé).",
                 "Deux éléments qui se recouvrent comptent leur partie commune deux fois : le recouvrement est détecté et signalé, pas corrigé.",
                 "Les arcs sont discrétisés : un rond est un polygone inscrit de 180 côtés (aire à 2·10⁻⁴ près).",
                 "Aucune classification de section, aucun voilement local : ce sont des caractéristiques géométriques."]),
]
EXEMPLES = (
    ("HEA + plat", [{"k": "profil", "f": "HEA", "p": "HEA 200", "y": 0, "z": 0}, {"k": "plat", "b": 220, "h": 12, "y": 0, "z": 101}]),
    ("Demi-HEA", [{"k": "profil", "f": "HEA", "p": "HEA 300", "y": 0, "z": 0, "c": "haut", "cc": 0}]),
    ("Caisson de plats", [{"k": "plat", "b": 300, "h": 10, "y": 0, "z": 95}, {"k": "plat", "b": 300, "h": 10, "y": 0, "z": -95},
                          {"k": "plat", "b": 10, "h": 180, "y": 145, "z": 0}, {"k": "plat", "b": 10, "h": 180, "y": -145, "z": 0}]),
    ("PRS en I", [{"k": "plat", "b": 200, "h": 15, "y": 0, "z": 157.5}, {"k": "plat", "b": 10, "h": 300, "y": 0, "z": 0},
                  {"k": "plat", "b": 200, "h": 15, "y": 0, "z": -157.5}]),
    ("2 UPN dos à dos", [{"k": "profil", "f": "UPN", "p": "UPN 200", "y": 42.5, "z": 0}, {"k": "profil", "f": "UPN", "p": "UPN 200", "y": -42.5, "z": 0, "m": 1}]),
    # (un trou de boulon traverse l'épaisseur : dans la COUPE il retire un rectangle d0 × t, pas un disque — l'exemple portait
    #  deux ronds Ø 22 dans un plat de 20, qui sortaient de la matière et s'ouvraient sur deux avertissements : 11/11/2026)
    ("Plat percé", [{"k": "plat", "b": 200, "h": 20, "y": 0, "z": 0}, {"k": "plat", "b": 22, "h": 20, "y": -50, "z": 0, "v": 1},
                    {"k": "plat", "b": 22, "h": 20, "y": 50, "z": 0, "v": 1}]),
)


# ================================================================================================ lecture
def _cles_element(i: int, pn: int = SOMMETS_MAX) -> List[str]:
    return [f"{c}{i}" for c in CLES_ELEMENT] + [f"p{a}{i}_{j}" for j in range(1, pn + 1) for a in ("y", "z")]


def _jeu(elements: Sequence[Dict], sel: int = 1) -> Dict:
    """Les paramètres COMPLETS d'une liste d'éléments (les clés absentes valent None : rien ne traîne d'un état précédent)."""
    d: Dict = {"n": len(elements), "sel": sel}
    for i, e in enumerate(elements, 1):
        for k in _cles_element(i):
            d[k] = None
        for k, v in e.items():
            d[f"{k}{i}"] = v
    return d


def _local(e: Dict) -> List:
    """Les contours de l'élément dans son repère (centre du rectangle enveloppe ; origine locale pour un polygone)."""
    if e["k"] == "profil":
        return S.profil(e["p"])[0]
    if e["k"] == "plat":
        return S.rectangle(e["b"], e["h"])
    if e["k"] == "rond":
        return S.rond(e["d"])
    return S.polygone(e["pts"])


def lire(params: Optional[Dict]) -> Tuple[B.Lecture, Dict]:
    L = B.Lecture(params)
    vide = not L.brut or not any(k in L.brut for k in ("n", "k1"))
    nb = L.nombre("n", 2 if vide else 1, 1, N_MAX, entier=True)
    els: List[Dict] = []
    boite_tot = None
    for i in range(1, nb + 1):
        e: Dict = {"i": i}
        e["k"] = L.choix(f"k{i}", [k for k, _ in SORTES], "profil" if i == 1 else "plat")
        if e["k"] == "profil":
            e["f"] = L.choix(f"f{i}", FAMILLES, "HEA")
            noms = [p["nom"] for p in KA.profils() if p["famille"] == e["f"]]
            if L.brut.get(f"p{i}") not in noms:
                L.brut.pop(f"p{i}", None)
            e["p"] = L.choix(f"p{i}", noms, "HEA 200" if "HEA 200" in noms else noms[len(noms) // 2])
            e["noms"] = noms
        elif e["k"] == "plat":
            e["b"] = L.nombre(f"b{i}", 220.0 if i == 2 and vide else 100.0, 0.1, 5000.0)
            e["h"] = L.nombre(f"h{i}", 12.0 if i == 2 and vide else 10.0, 0.1, 5000.0)
        elif e["k"] == "rond":
            e["d"] = L.nombre(f"d{i}", 30.0, 0.1, 5000.0)
        else:
            pn = L.nombre(f"pn{i}", 4, 3, SOMMETS_MAX, entier=True)
            e["pts"] = []
            for j in range(1, pn + 1):
                dy, dz = POLY_DEFAUT[(j - 1) % 4] if j <= 4 else (0.0, 60.0 + 10.0 * j)
                e["pts"].append((L.nombre(f"py{i}_{j}", dy, -5000.0, 5000.0), L.nombre(f"pz{i}_{j}", dz, -5000.0, 5000.0)))
        e["r"] = L.nombre(f"r{i}", 0.0, -360.0, 360.0)
        e["m"] = L.case(f"m{i}")
        e["v"] = L.case(f"v{i}")
        e["c"] = L.choix(f"c{i}", [k for k, _ in COUPES], "non")
        e["erreur"] = None
        try:
            loc = _local(e)
            y0, y1, z0, z1 = S.boite(loc)
            if e["c"] != "non":
                lo, hi = (z0, z1) if e["c"] in ("haut", "bas") else (y0, y1)
                marge = min(0.5, (hi - lo) / 100.0)
                e["cc"] = L.nombre(f"cc{i}", (lo + hi) / 2.0, lo + marge, hi - marge)
                loc = S.couper(loc, e["c"], e["cc"])
            # un élément sans position se pose sur ce qui est déjà dessiné (centré, bord à bord)
            tourne = S.transformer(loc, e["r"], e["m"], 0.0, 0.0)
            ty0, ty1, tz0, tz1 = S.boite(tourne)
            if boite_tot is None:
                auto_y, auto_z = 0.0, 0.0
            else:
                auto_y = (boite_tot[0] + boite_tot[1]) / 2.0 - (ty0 + ty1) / 2.0
                auto_z = boite_tot[3] - tz0
                if e["v"]:                                      # un évidement se pose AU CENTRE de la matière
                    auto_z = (boite_tot[2] + boite_tot[3]) / 2.0 - (tz0 + tz1) / 2.0
            e["y"] = L.nombre(f"y{i}", round(auto_y, 3), -20000.0, 20000.0)
            e["z"] = L.nombre(f"z{i}", round(auto_z, 3), -20000.0, 20000.0)
            contours = S.transformer(loc, e["r"], e["m"], e["y"], e["z"])
            e["el"] = S.element(contours, _nom(e), e["v"])
            bt = e["el"]["boite"]
            if not e["v"]:
                boite_tot = bt if boite_tot is None else (min(boite_tot[0], bt[0]), max(boite_tot[1], bt[1]), min(boite_tot[2], bt[2]), max(boite_tot[3], bt[3]))
        except S.ErreurDonnees as ex:
            e["erreur"] = str(ex)
            e.setdefault("y", L.nombre(f"y{i}", 0.0, -20000.0, 20000.0))
            e.setdefault("z", L.nombre(f"z{i}", 0.0, -20000.0, 20000.0))
        els.append(e)
    sel = L.nombre("sel", 1, 1, nb, entier=True)
    g = L.choix("g", GRANDEURS, "Iy")
    return L, {"els": els, "sel": sel, "g": g}


def _nom(e: Dict) -> str:
    if e["k"] == "profil":
        base = e["p"]
    elif e["k"] == "plat":
        base = f"Plat {C.nb(e['b'], 1)} × {C.nb(e['h'], 1)}"
    elif e["k"] == "rond":
        base = f"Rond Ø {C.nb(e['d'], 1)}"
    else:
        base = f"Polygone ({len(e['pts'])} sommets)"
    if e.get("c", "non") != "non":
        base += " coupé"
    return ("Évidement · " if e.get("v") else "") + base


# ================================================================================================ le dessin
W, H_MAX, MARGE = 640.0, 470.0, 46.0


def _chemin(contours: Sequence, X, Y) -> str:
    return "".join("M" + "L".join(f"{X(y):.1f} {Y(z):.1f}" for y, z in ring) + "Z" for ring in contours if ring)


def dessin(o: Dict, car: Optional[Dict], recouv: Sequence[Dict]) -> str:
    els = [e for e in o["els"] if e.get("el")]
    if not els:
        return f'<svg class="rdm-svg rdm-svg-section" viewBox="0 0 {W:.0f} 200"></svg>'
    bs = [e["el"]["boite"] for e in els]
    y0, y1 = min(b[0] for b in bs + [(0, 0, 0, 0)]), max(b[1] for b in bs + [(0, 0, 0, 0)])
    z0, z1 = min(b[2] for b in bs + [(0, 0, 0, 0)]), max(b[3] for b in bs + [(0, 0, 0, 0)])
    lw, lh = max(y1 - y0, 1.0), max(z1 - z0, 1.0)
    ech = min((W - 2 * MARGE - 60) / lw, (H_MAX - 2 * MARGE) / lh)
    H = lh * ech + 2 * MARGE + 26
    ox = (W - lw * ech) / 2 + 16
    X = lambda y: ox + (y - y0) * ech                                  # noqa: E731
    Y = lambda z: MARGE + (z1 - z) * ech                               # noqa: E731
    s = ""
    # les axes du repère (par l'origine)
    s += (f'<g class="repere"><path d="M{X(y0) - 22:.1f} {Y(0):.1f}H{X(y1) + 22:.1f}M{X(0):.1f} {Y(z1) - 22:.1f}V{Y(z0) + 22:.1f}"/>'
          f'<text x="{X(y1) + 14:.1f}" y="{Y(0) + 15:.1f}">y</text><text x="{X(0) - 10:.1f}" y="{Y(z1) - 14:.1f}" text-anchor="middle">z</text></g>')
    doubles = {x for rc in recouv for x in (rc["i"], rc.get("j", rc["i"]))}
    pas = 1.0 if lw + lh < 4000 else 5.0
    for idx, e in enumerate(o["els"]):
        if not e.get("el"):
            continue
        i = e["i"]
        el = e["el"]
        cls = "elt" + (" ev" if e["v"] else "") + (" on" if i == o["sel"] else "") + (" double" if idx in doubles else "")
        gl = {"x": {f"y{i}": e["y"]}, "y": {f"z{i}": e["z"]}, "ech": round(ech, 5), "pas": pas, "sy": -1, "unite": "mm", "sel": {"sel": i}}
        s += (f'<g class="{cls}" data-element="{i}"{A.sel({"sel": i})} data-glisser="{C.esc(json.dumps(gl, ensure_ascii=False))}">'
              f'<title>{C.esc(el["nom"])} — cliquer pour choisir, glisser pour déplacer</title>'
              f'<path fill-rule="evenodd" d="{_chemin(el["contours"], X, Y)}"/>'
              f'<g class="num"><circle cx="{X(el["y"]):.1f}" cy="{Y(el["z"]):.1f}" r="8.5"/><text x="{X(el["y"]):.1f}" y="{Y(el["z"]) + 4:.1f}" text-anchor="middle">{i}</text></g></g>')
    # les sommets du polygone choisi se glissent un à un (s'il n'est ni tourné ni retourné)
    cho = next((e for e in o["els"] if e["i"] == o["sel"]), None)
    if cho and cho["k"] == "poly" and cho.get("el") and not cho["m"] and abs(cho["r"]) < 1e-9 and cho["c"] == "non":
        for j, (py, pz) in enumerate(cho["pts"], 1):
            gl = {"x": {f"py{cho['i']}_{j}": py}, "y": {f"pz{cho['i']}_{j}": pz}, "ech": round(ech, 5), "pas": pas, "sy": -1, "unite": "mm"}
            s += (f'<circle class="sommet glis" data-sommet="{j}" cx="{X(cho["y"] + py):.1f}" cy="{Y(cho["z"] + pz):.1f}" r="6.5" '
                  f'data-glisser="{C.esc(json.dumps(gl, ensure_ascii=False))}"><title>Sommet {j} — glisser</title></circle>')
    if car:
        gx, gy = X(car["yG"]), Y(car["zG"])
        # les axes principaux par G
        if not car["isotrope"]:
            ca, sa = math.cos(car["alpha"]), math.sin(car["alpha"])

            def portee(dx, dy):
                """La demi-longueur d'un axe depuis G, sans sortir du dessin (ni dépasser la section de plus de 26)."""
                lim = 0.5 * math.hypot(lw, lh) * ech + 26
                for d_, lo_, hi_, g_ in ((dx, 16.0, W - 16.0, gx), (dy, 16.0, H - 16.0, gy)):
                    if abs(d_) > 1e-9:
                        lim = min(lim, (hi_ - g_) / abs(d_), (g_ - lo_) / abs(d_))
                return max(lim, 12.0)
            l1, l2 = portee(ca, sa), portee(sa, ca)
            # le nom d'un axe se pose à son bout du HAUT quand l'axe est plutôt vertical, jamais à celui du bas : il y tombait
            # sur la cote de largeur (un plat à plat : l'axe 1 est vertical, α = −90°)
            b1_, b2_ = (-1 if abs(sa) > abs(ca) and sa < 0 else 1), (-1 if abs(sa) > abs(ca) and sa > 0 else 1)
            v1x, v1y = (l1 - 2) * ca + 9 * sa, -(l1 - 2) * sa + 9 * ca
            v2x, v2y = -(l2 - 2) * sa + 9 * ca, -(l2 - 2) * ca - 9 * sa
            s += (f'<g class="principaux"><path d="M{gx - l1 * ca:.1f} {gy + l1 * sa:.1f}L{gx + l1 * ca:.1f} {gy - l1 * sa:.1f}"/>'
                  f'<path d="M{gx + l2 * sa:.1f} {gy + l2 * ca:.1f}L{gx - l2 * sa:.1f} {gy - l2 * ca:.1f}"/>'
                  f'<text class="ax1" x="{gx + b1_ * v1x:.1f}" y="{gy + b1_ * v1y + 4:.1f}" text-anchor="middle">1</text>'
                  f'<text class="ax2" x="{gx + b2_ * v2x:.1f}" y="{gy + b2_ * v2y + 4:.1f}" text-anchor="middle">2</text></g>')
        s += (f'<g class="cdg"><path d="M{gx - 12:.1f} {gy:.1f}h24M{gx:.1f} {gy - 12:.1f}v24"/><circle cx="{gx:.1f}" cy="{gy:.1f}" r="4.2"/>'
              f'<text x="{gx + 9:.1f}" y="{gy - 8:.1f}">G</text></g>')
        # l'encombrement
        b0, b1, b2, b3 = car["boite"]
        yc_ = Y(b2) + 30
        s += (f'<g class="cotes"><path d="M{X(b0):.1f} {yc_:.1f}H{X(b1):.1f}M{X(b0):.1f} {yc_ - 5:.1f}v10M{X(b1):.1f} {yc_ - 5:.1f}v10"/>'
              f'<text x="{(X(b0) + X(b1)) / 2:.1f}" y="{yc_ - 5:.1f}" text-anchor="middle">{r(b1 - b0, 1)} mm</text>'
              f'<path d="M{X(b0) - 30:.1f} {Y(b3):.1f}V{Y(b2):.1f}M{X(b0) - 35:.1f} {Y(b3):.1f}h10M{X(b0) - 35:.1f} {Y(b2):.1f}h10"/>'
              f'<text x="{X(b0) - 35:.1f}" y="{(Y(b2) + Y(b3)) / 2:.1f}" text-anchor="middle" transform="rotate(-90 {X(b0) - 35:.1f} {(Y(b2) + Y(b3)) / 2:.1f})">{r(b3 - b2, 1)} mm</text></g>')
    return (f'<svg class="rdm-svg rdm-svg-section" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="Dessin de la section à l\'échelle : éléments, '
            f'centre de gravité, axes principaux">{s}</svg>')


# ================================================================================================ l'éditeur
def _params_sans(p: Dict, i: int, nb: int) -> Dict:
    """Les paramètres après le RETRAIT de l'élément i : les suivants remontent d'un rang (toutes leurs clés)."""
    d: Dict = {"n": nb - 1, "sel": max(1, min(i, nb - 1))}
    for j in range(i, nb + 1):
        for k in _cles_element(j):
            src = re.sub(r"^([a-z]+)(\d+)", lambda mm: f"{mm.group(1)}{j + 1}", k)
            d[k] = p.get(src) if j < nb else None
    return d


def _params_copie(p: Dict, i: int, nb: int, boite: Tuple) -> Dict:
    """Les paramètres d'une COPIE de l'élément i, posée à côté (à droite de son rectangle enveloppe)."""
    k = nb + 1
    d: Dict = {"n": k, "sel": k}
    for cle in _cles_element(i):
        d[re.sub(r"^([a-z]+)(\d+)", lambda mm: f"{mm.group(1)}{k}", cle)] = p.get(cle)
    d[f"y{k}"] = round((p.get(f"y{i}") or 0.0) + (boite[1] - boite[0]) + 10.0, 3)
    return d


def _accoler(e: Dict, ref: Dict) -> str:
    """Quatre boutons : poser l'élément choisi contre le rectangle enveloppe d'un autre (centré sur l'autre axe)."""
    b, a = e["el"]["boite"], ref["el"]["boite"]
    i = e["i"]
    cy = (a[0] + a[1]) / 2 - (b[0] + b[1]) / 2
    cz = (a[2] + a[3]) / 2 - (b[2] + b[3]) / 2
    cas = (("dessus", {f"y{i}": e["y"] + cy, f"z{i}": e["z"] + a[3] - b[2]}, "arrow-up"), ("dessous", {f"y{i}": e["y"] + cy, f"z{i}": e["z"] + a[2] - b[3]}, "arrow-down"),
           ("à gauche", {f"y{i}": e["y"] + a[0] - b[1], f"z{i}": e["z"] + cz}, "arrow-left"), ("à droite", {f"y{i}": e["y"] + a[1] - b[0], f"z{i}": e["z"] + cz}, "chevron-right"),
           ("centré", {f"y{i}": e["y"] + cy, f"z{i}": e["z"] + cz}, "scan"))
    return "".join(A.bouton(nom, {k: round(v, 3) for k, v in d.items()}, ic, "btn sm") for nom, d, ic in cas)


def _fiche_element(e: Dict, o: Dict, params: Dict) -> str:
    i = e["i"]
    lignes = [A.menu_champ(f"k{i}", e["k"], SORTES, "Sorte")]
    if e["k"] == "profil":
        lignes[0] += A.menu_champ(f"f{i}", e["f"], [(f, f) for f in FAMILLES], "Famille") + A.menu_champ(f"p{i}", e["p"], [(x, x) for x in e["noms"]], "Profilé")
    elif e["k"] == "plat":
        lignes[0] += A.champ(f"b{i}", e["b"], "mm", "b (selon y)") + A.champ(f"h{i}", e["h"], "mm", "h (selon z)")
    elif e["k"] == "rond":
        lignes[0] += A.champ(f"d{i}", e["d"], "mm", "Ø")
    else:
        som = "".join(f'<div class="rdm-som"><span>{j}</span>{A.champ(f"py{i}_{j}", py, "", "y")}{A.champ(f"pz{i}_{j}", pz, "mm", "z")}</div>' for j, (py, pz) in enumerate(e["pts"], 1))
        nbp = len(e["pts"])
        plus = A.bouton("Sommet", {f"pn{i}": nbp + 1, f"py{i}_{nbp + 1}": round((e["pts"][-1][0] + e["pts"][0][0]) / 2, 1),
                                   f"pz{i}_{nbp + 1}": round((e["pts"][-1][1] + e["pts"][0][1]) / 2 + 20.0, 1)}, "plus", "btn sm") if nbp < SOMMETS_MAX else ""
        moins = A.bouton("Dernier sommet", {f"pn{i}": nbp - 1}, "minus", "btn sm") if nbp > 3 else ""
        lignes.append(f'<div class="rdm-soms">{som}</div><div class="rdm-plus">{plus}{moins}</div>'
                      '<p class="rdm-aide">Sommets dans le repère de l\'élément (mm) ; ils se glissent aussi sur le dessin.</p>')
    lignes.append(A.champ(f"y{i}", e["y"], "mm", "Position y") + A.champ(f"z{i}", e["z"], "mm", "z")
                  + A.champ(f"r{i}", e["r"], "°", "Rotation"))
    rot = "".join(A.bouton(t, {f"r{i}": v}, "", "btn sm" + (" on" if abs((e["r"] % 360) - v) < 1e-9 else "")) for v, t in ROTATIONS)
    lignes.append(f'<div class="rdm-plus">{rot}{A.bouton("Miroir", {f"m{i}": "0" if e["m"] else "1"}, "", "btn sm" + (" on" if e["m"] else ""))}'
                  f'{A.bouton("Évidement", {f"v{i}": "0" if e["v"] else "1"}, "", "btn sm" + (" on" if e["v"] else ""), "La matière de cet élément est retirée")}</div>')
    coupe = A.menu_champ(f"c{i}", e["c"], COUPES, "Coupe")
    if e["c"] != "non":
        coupe += A.champ(f"cc{i}", e["cc"], "mm", "à " + ("z =" if e["c"] in ("haut", "bas") else "y ="))
    lignes.append(coupe)
    autres = [x for x in o["els"] if x["i"] != i and x.get("el")]
    if autres and e.get("el"):
        ref = next((x for x in autres if x["i"] == i - 1), autres[0])
        lignes.append(f'<div class="rdm-acc"><span class="rdm-cl">Poser contre l\'élément {ref["i"]} :</span><div class="rdm-plus">{_accoler(e, ref)}</div></div>')
    if e["erreur"]:
        lignes.append(A.refus("Élément refusé", C.esc(e["erreur"])))
    return "".join(f'<div class="rdm-fl">{x}</div>' for x in lignes)


def _elements(o: Dict, params: Dict) -> str:
    nb = len(o["els"])
    corps = ""
    for e in o["els"]:
        i = e["i"]
        on = i == o["sel"]
        aire = f'{r(abs(e["el"]["A"]) / 100.0)} cm²' if e.get("el") else "—"
        outils = ""
        if on:
            if nb < N_MAX and e.get("el"):
                outils += A.bouton("", _params_copie(params, i, nb, e["el"]["boite"]), "copy", "btn icon ghost sm", f"Dupliquer l'élément {i}")
            if nb > 1:
                outils += A.bouton("", _params_sans(params, i, nb), "trash-2", "btn icon ghost sm", f"Retirer l'élément {i}")
        corps += (f'<div class="rdm-el{" on" if on else ""}{" ko" if e["erreur"] else ""}" data-element="{i}">'
                  f'<button type="button" class="rdm-el-t"{A.sel({"sel": i})} aria-pressed="{"true" if on else "false"}"><span class="rdm-el-n">{i}</span>'
                  f'<span class="rdm-el-d">{C.esc(_nom(e))}</span><span class="rdm-el-a">{aire}</span></button>{outils}'
                  f'{("<div class=" + chr(34) + "rdm-el-f" + chr(34) + ">" + _fiche_element(e, o, params) + "</div>") if on else ""}</div>')
    ajout = ""
    if nb < N_MAX:
        k = nb + 1
        ajout = "".join(A.bouton(t, dict({c: None for c in _cles_element(k)}, **{"n": k, "sel": k, f"k{k}": s_}), "plus", "btn sm") for s_, t in SORTES)
    ex = "".join(A.bouton(nom, _jeu(els), "", "btn sm ghost") for nom, els in EXEMPLES)
    return A.bloc("Éléments", f'{corps}<div class="rdm-plus">{ajout}</div><div class="rdm-ex"><span class="rdm-cl">Exemples :</span>{ex}</div>'
                  '<p class="rdm-aide">Sur le dessin : un clic choisit un élément, un glisser le déplace (pas de 1 mm). Un élément ajouté se pose sur la section.</p>',
                  "rdm-elements")


# ================================================================================================ résultats, feuille
def _resultats(car: Dict, o: Dict) -> str:
    def lig(g, sym, nom, val, unite, extra=""):
        on = g and o["g"] == g
        sel = A.sel({"g": g}) if g else ""
        return (f'<tr{sel} class="rdm-r{" on" if on else ""}{"" if g else " fixe"}"{" tabindex=" + chr(34) + "0" + chr(34) if g else ""}><th>{m(sym)}</th>'
                f'<td class="nm">{nom}</td><td class="v"><b>{val}</b> {unite}</td><td class="p">{extra}</td></tr>')
    f = car["fibres"]
    corps = lig("A", "A", "aire", r(car["A"] / 100), "cm²", f"{r(car['A'] * RHO_ACIER / 1e6, 1)} kg/m en acier")
    corps += lig(None, "y_G ; z_G", "centre de gravité", f"{r(car['yG'], 1)} ; {r(car['zG'], 1)}", "mm")
    corps += lig("Iy", "I_y", "inertie, axe horizontal", r(car["Iy"] / 1e4), "cm⁴", f"i = {r(car['iy'], 1)} mm")
    corps += lig("Iz", "I_z", "inertie, axe vertical", r(car["Iz"] / 1e4), "cm⁴", f"i = {r(car['iz'], 1)} mm")
    corps += lig(None, "I_yz", "produit d'inertie", r(car["Iyz"] / 1e4), "cm⁴")
    if car["isotrope"]:
        corps += lig(None, "I_1 = I_2", "axes principaux", r(car["I1"] / 1e4), "cm⁴", "toute direction")
    else:
        corps += lig(None, "I_1 ; I_2", "inerties principales", f"{r(car['I1'] / 1e4)} ; {r(car['I2'] / 1e4)}", "cm⁴", f"α = {r(car['alpha_deg'], 2)}°")
    corps += lig("W", "W_el,y", "module élastique (sup. ; inf.)", f"{r(car['Wel_y_sup'] / 1e3, 1)} ; {r(car['Wel_y_inf'] / 1e3, 1)}", "cm³",
                 f"v = {r(f['z_sup'], 1)} ; {r(f['z_inf'], 1)} mm")
    corps += lig(None, "W_el,z", "module élastique (droite ; gauche)", f"{r(car['Wel_z_droite'] / 1e3, 1)} ; {r(car['Wel_z_gauche'] / 1e3, 1)}", "cm³",
                 f"v = {r(f['y_droite'], 1)} ; {r(f['y_gauche'], 1)} mm")
    corps += lig(None, "W_pl,y ; W_pl,z", "modules plastiques", f"{r(car['Wpl_y'] / 1e3, 1)} ; {r(car['Wpl_z'] / 1e3, 1)}", "cm³")
    corps += ('<tr class="rdm-r fixe nc"><th>y<sub>C</sub>, I<sub>t</sub>, I<sub>w</sub></th><td class="nm" colspan="3">centre de cisaillement, torsion, gauchissement : '
              '<b>non calculés</b> — voir « Hypothèses et conventions »</td></tr>')
    return A.bloc("Résultats", f'<table class="rdm-res"><tbody>{corps}</tbody></table>', "rdm-resultats")


def _tableau_steiner(car: Dict) -> str:
    """Le tableau de Huygens-Steiner : une ligne par élément (centimètres)."""
    tete = ("<tr><th>n°</th><th>élément</th><th>A<sub>i</sub><span>cm²</span></th><th>y<sub>i</sub><span>cm</span></th><th>z<sub>i</sub><span>cm</span></th>"
            "<th>I<sub>y,i</sub><span>cm⁴</span></th><th>I<sub>z,i</sub><span>cm⁴</span></th><th>I<sub>yz,i</sub><span>cm⁴</span></th>"
            "<th>A<sub>i</sub>·(z<sub>i</sub> − z<sub>G</sub>)²<span>cm⁴</span></th><th>A<sub>i</sub>·(y<sub>i</sub> − y<sub>G</sub>)²<span>cm⁴</span></th></tr>")
    corps = ""
    for k, li in enumerate(car["lignes"], 1):
        corps += (f'<tr{" class=" + chr(34) + "ev" + chr(34) if li["signe"] < 0 else ""}><td>{k}</td><td class="nm">{C.esc(li["nom"])}</td><td>{r(li["A"] / 100)}</td>'
                  f'<td>{r(li["y"] / 10, 3)}</td><td>{r(li["z"] / 10, 3)}</td><td>{r(li["Iy"] / 1e4)}</td><td>{r(li["Iz"] / 1e4)}</td><td>{r(li["Iyz"] / 1e4)}</td>'
                  f'<td>{r(li["A"] * li["dz"] ** 2 / 1e4)}</td><td>{r(li["A"] * li["dy"] ** 2 / 1e4)}</td></tr>')
    pied = (f'<tr class="tot"><td></td><td class="nm">Section</td><td>{r(car["A"] / 100)}</td><td>{r(car["yG"] / 10, 3)}</td><td>{r(car["zG"] / 10, 3)}</td>'
            f'<td colspan="2">I<sub>y</sub> = {r(car["Iy"] / 1e4)}</td><td></td><td colspan="2">I<sub>z</sub> = {r(car["Iz"] / 1e4)}</td></tr>')
    return f'<div class="rdm-st-c"><table class="rdm-st"><thead>{tete}</thead><tbody>{corps}{pied}</tbody></table></div>'


def _feuille(car: Dict, o: Dict) -> List:
    li = car["lignes"]
    A_ = car["A"] / 100.0
    g = o["g"]
    cm = lambda x, d=4: n(x, d)                                                                          # noqa: E731
    centre = [A.ligne("A", "aire", "Σ A_i", " + ".join(cm(x["A"] / 100) for x in li), r(A_), "cm²", "", g == "A", "A"),
              A.ligne("y_G", "centre de gravité", "Σ A_i·y_i / Σ A_i", "(" + " + ".join(f"{cm(x['A'] / 100)} × {cm(x['y'] / 10)}" for x in li) + f") / {cm(A_)}",
                      r(car["yG"] / 10, 3), "cm", "", False, "yG"),
              A.ligne("z_G", "", "Σ A_i·z_i / Σ A_i", "(" + " + ".join(f"{cm(x['A'] / 100)} × {cm(x['z'] / 10)}" for x in li) + f") / {cm(A_)}",
                      r(car["zG"] / 10, 3), "cm", "", False, "zG")]
    yG, zG = car["yG"] / 10.0, car["zG"] / 10.0
    inert = [A.ligne("I_y", "Huygens-Steiner", "Σ (I_y,i + A_i·(z_i − z_G)²)",
                     " + ".join(f"{cm(x['Iy'] / 1e4, 3)} + {cm(x['A'] / 100)} × ({cm(x['z'] / 10)} − {cm(zG)})²" for x in li), r(car["Iy"] / 1e4), "cm⁴", "Steiner", g == "Iy", "Iy"),
             A.ligne("I_z", "", "Σ (I_z,i + A_i·(y_i − y_G)²)",
                     " + ".join(f"{cm(x['Iz'] / 1e4, 3)} + {cm(x['A'] / 100)} × ({cm(x['y'] / 10)} − {cm(yG)})²" for x in li), r(car["Iz"] / 1e4), "cm⁴", "Steiner", g == "Iz", "Iz"),
             A.ligne("I_yz", "produit d'inertie", "Σ (I_yz,i + A_i·(y_i − y_G)·(z_i − z_G))",
                     " + ".join(f"{cm(x['Iyz'] / 1e4, 3)} + {cm(x['A'] / 100)} × ({cm(x['y'] / 10)} − {cm(yG)}) × ({cm(x['z'] / 10)} − {cm(zG)})" for x in li),
                     r(car["Iyz"] / 1e4), "cm⁴", "Steiner", False, "Iyz")]
    Iy, Iz, Iyz = car["Iy"] / 1e4, car["Iz"] / 1e4, car["Iyz"] / 1e4
    if car["isotrope"]:
        princ = [A.ligne("I_1 = I_2", "I_y = I_z et I_yz = 0 : toutes les directions sont principales", "", "", r(car["I1"] / 1e4), "cm⁴", "", False, "I1")]
    else:
        rac = f"√((({cm(Iy, 3)} − {cm(Iz, 3)}) / 2)² + {cm(Iyz, 3)}²)"
        princ = [A.ligne("I_1", "inertie principale maximale", "(I_y + I_z)/2 + √(((I_y − I_z)/2)² + I_yz²)", f"({cm(Iy, 3)} + {cm(Iz, 3)}) / 2 + {rac}",
                         r(car["I1"] / 1e4), "cm⁴", "", False, "I1"),
                 A.ligne("I_2", "inertie principale minimale", "(I_y + I_z)/2 − √(((I_y − I_z)/2)² + I_yz²)", f"({cm(Iy, 3)} + {cm(Iz, 3)}) / 2 − {rac}",
                         r(car["I2"] / 1e4), "cm⁴", "", False, "I2"),
                 A.ligne("α", "angle de l'axe 1 avec y (sens trigonométrique)", "½·atan2(−2·I_yz ; I_y − I_z)", "", r(car["alpha_deg"], 2), "°",
                         "tan 2α = −2·I_yz/(I_y − I_z)", False, "alpha")]
    f = car["fibres"]
    mod = [A.ligne("W_el,y,sup", "fibre supérieure", "I_y / v_sup", f"{cm(Iy, 3)} / {cm(f['z_sup'] / 10)}", r(car["Wel_y_sup"] / 1e3, 1), "cm³", "", g == "W" and car["Wel_y_sup"] <= car["Wel_y_inf"], "Wys"),
           A.ligne("W_el,y,inf", "fibre inférieure", "I_y / v_inf", f"{cm(Iy, 3)} / {cm(f['z_inf'] / 10)}", r(car["Wel_y_inf"] / 1e3, 1), "cm³", "", g == "W" and car["Wel_y_inf"] < car["Wel_y_sup"], "Wyi"),
           A.ligne("W_el,z,droite", "fibre de droite", "I_z / v_droite", f"{cm(Iz, 3)} / {cm(f['y_droite'] / 10)}", r(car["Wel_z_droite"] / 1e3, 1), "cm³", "", False, "Wzd"),
           A.ligne("W_el,z,gauche", "fibre de gauche", "I_z / v_gauche", f"{cm(Iz, 3)} / {cm(f['y_gauche'] / 10)}", r(car["Wel_z_gauche"] / 1e3, 1), "cm³", "", False, "Wzg"),
           A.ligne("i_y", "rayon de giration", "√(I_y / A)", f"√({cm(Iy, 3)} / {cm(A_)})", r(car["iy"] / 10, 3), "cm", "", False, "iy"),
           A.ligne("i_z", "rayon de giration", "√(I_z / A)", f"√({cm(Iz, 3)} / {cm(A_)})", r(car["iz"] / 10, 3), "cm", "", False, "iz"),
           A.ligne("W_pl,y", f"axe d'égales aires à z = {r(car['z_pl'], 1)} mm", "∫ |z − z_pl| dA", "", r(car["Wpl_y"] / 1e3, 1), "cm³", "un seul matériau", False, "Wply"),
           A.ligne("W_pl,z", f"axe d'égales aires à y = {r(car['y_pl'], 1)} mm", "∫ |y − y_pl| dA", "", r(car["Wpl_z"] / 1e3, 1), "cm³", "un seul matériau", False, "Wplz")]
    nc = [A.ligne("y_C ; z_C", "centre de cisaillement", "", "", "non calculé", "", "", False, "nc_c"),
          A.ligne("I_t", "inertie de torsion", "", "", "non calculé", "", "", False, "nc_t"),
          A.ligne("I_w", "inertie de gauchissement", "", "", "non calculé", "", "", False, "nc_w")]
    return [("Aire et centre de gravité", "unités de la feuille : cm, cm², cm⁴ (positions du tableau en cm)", centre), ("Inerties par rapport au centre de gravité", "", inert),
            ("Axes principaux", "", princ), ("Modules et rayons de giration", "", mod),
            ("Non calculé", "ils demandent la fonction de gauchissement de la section entière — détail dans « Hypothèses et conventions »", nc)]


def _controles_catalogue(o: Dict) -> List[str]:
    """Pour chaque profilé du catalogue NON coupé : son inertie propre recalculée face à la valeur du catalogue."""
    out = []
    for e in o["els"]:
        if e["k"] == "profil" and e["c"] == "non" and e.get("el"):
            p = KA.par_nom()[e["p"]]
            a0 = e["r"] % 180
            if abs(a0) < 1e-9 or abs(a0 - 90) < 1e-9:
                Iy_cat = p["Iy"] if abs(a0) < 1e-9 else p["Iz"]
                ecart = abs(abs(e["el"]["Iy"]) / 1e4 / Iy_cat - 1)
                out.append(f"Élément {e['i']} — {C.esc(e['p'])} : I<sub>y</sub> propre recalculé {r(abs(e['el']['Iy']) / 1e4)} cm⁴, catalogue {r(Iy_cat)} cm⁴ "
                           f"(écart {r(ecart * 100, 4)} %).")
    return out


def rendre(params: Optional[Dict], hote: str = "page") -> Dict:
    L, o = lire(params)
    titre = "Section quelconque"
    base = {"titre": titre, "barre": "", "pourquoi": A.volet(titre, "caractéristiques géométriques", VOLET, SOURCES), "sources": A.sources(*SOURCES),
            "params": L.out, "message": L.message()}
    bons = [e["el"] for e in o["els"] if e.get("el")]
    try:
        car = S.caracteristiques(bons)
        recouv = S.recouvrements(bons)
    except S.ErreurDonnees as ex:
        tableau = (f'<div class="rdm rdm-sections"><div class="rdm-g"><figure class="rdm-dessin">{dessin(o, None, [])}</figure></div>'
                   f'<div class="rdm-d">{_elements(o, L.out)}{A.refus("Section refusée", C.esc(str(ex)[0].upper() + str(ex)[1:]) + ".")}</div></div>')
        return dict(base, tableau=tableau, bande=A.bande("—", "", "I_y", "section refusée", C.esc(str(ex))))
    av = []
    idx = [e["i"] for e in o["els"] if e.get("el")]
    for rc in recouv:
        if rc["sorte"] == "double":
            av.append(f"Les éléments {idx[rc['i']]} et {idx[rc['j']]} se recouvrent sur environ {r(rc['aire'] / 100, 2)} cm² : cette matière est comptée deux fois.")
        else:
            av.append(f"L'évidement {idx[rc['i']]} sort de la matière sur environ {r(rc['aire'] / 100, 2)} cm² : cette partie est retirée à tort.")
    for e in o["els"]:
        if e["erreur"]:
            av.append(f"Élément {e['i']} écarté du calcul : {C.esc(e['erreur'])}.")
    hyp = [f"Section de {len(bons)} élément{'s' if len(bons) > 1 else ''} : " + " ; ".join(C.esc(e["nom"]) for e in bons) + ".",
           "Repère : y horizontal (vers la droite), z vertical (vers le haut) ; I<sub>y</sub> = ∫z² dA. Un seul matériau.",
           "Intégrales de Green sur les contours (arcs discrétisés), Huygens-Steiner, axes principaux."] + _controles_catalogue(o)
    entete = A.entete_impression(titre, f"{len(bons)} élément(s)", hyp)
    feuille = (A.bloc("Feuille de calcul", f'<ul class="rdm-hyp">{"".join(f"<li>{h}</li>" for h in hyp)}</ul>{_tableau_steiner(car)}{A.feuille(_feuille(car, o))}', "rdm-feuille"))
    tableau = (f'<div class="rdm rdm-sections">{entete}<div class="rdm-g"><figure class="rdm-dessin">{dessin(o, car, recouv)}</figure>'
               '<p class="rdm-aide">G : centre de gravité · 1, 2 : axes principaux d\'inertie · cotes : encombrement de la section.</p></div>'
               f'<div class="rdm-d">{_elements(o, L.out)}{_resultats(car, o)}</div><div class="rdm-f">{feuille}</div></div>')
    g = o["g"]
    if g == "A":
        bande = A.bande(r(car["A"] / 100), "cm²", "A", "aire de la section", f"{r(car['A'] * RHO_ACIER / 1e6, 1)} kg/m si elle est en acier", av)
    elif g == "Iz":
        bande = A.bande(r(car["Iz"] / 1e4), "cm⁴", "I_z", "inertie par rapport à l'axe vertical passant par G", f"i_z = {r(car['iz'], 1)} mm", av)
    elif g == "W":
        bande = A.bande(r(car["Wel_y"] / 1e3, 1), "cm³", "W_el,y", "module élastique minimal, flexion autour de y",
                        f"fibre {'supérieure' if car['Wel_y_sup'] <= car['Wel_y_inf'] else 'inférieure'} · W<sub>pl,y</sub> = {r(car['Wpl_y'] / 1e3, 1)} cm³", av)
    else:
        bande = A.bande(r(car["Iy"] / 1e4), "cm⁴", "I_y", "inertie par rapport à l'axe horizontal passant par G",
                        f"G à y = {r(car['yG'], 1)} mm, z = {r(car['zG'], 1)} mm · A = {r(car['A'] / 100)} cm²", av)
    return dict(base, tableau=tableau, bande=bande)

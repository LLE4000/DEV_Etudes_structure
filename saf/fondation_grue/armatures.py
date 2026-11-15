# -*- coding: utf-8 -*-
"""Fondation de grue — le PRINCIPE D'ARMATURES (lot 5, cahier des charges du 15/11/2026, § 7 ; annexe A,
``principe_armatures``). numpy seul (pas de scipy : les zones rouges se regroupent ici en 4-connexité, l'étiquetage de
``scipy.ndimage.label``, dans le même ordre).

L'ingénieur CHOISIT les treillis (Ø et espacement, en bas et en haut, en x et en y) et la largeur des bandes. Le module :

  1. demande au MOTEUR DU MODULE DALLES (``beton/dalle/moteur.py``, méthode du bureau, ELS) ce que reprend chaque treillis :
     A_s,req y est linéaire en M, donc mRd = A_s du treillis ÷ A_s,req pour 1 kNm/m — aucune formule réécrite ;
  2. pose les renforts des bandes et les renforts locaux des zones rouges (``principe`` : l'annexe A, à l'identique) ;
  3. compte les barres de chaque renfort avec A_s() du moteur Dalles : n = ⌈(A_s(m_dim) − A_s du treillis) × largeur ÷
     aire d'une barre⌉ (§ 7.4) ; Ø automatique (12, 16 ou 20 : le plus léger, à poids égal le plus gros) ou imposé ;
  4. fait contrôler par le moteur Dalles la hauteur de béton (h_min ≤ h pour m_dim) et l'effort tranchant (V à d ÷ largeur :
     τ ≤ τ_adm,I, la dalle ne reçoit pas d'étriers) — les deux seuls cas, avec le sol, où le module dit d'épaissir.

Rien d'autre n'est dimensionné (§ 2) : pas de torsion, pas d'armatures d'effort tranchant, pas de cage de plot, pas de
recouvrements ni de longueur maxi des barres, pas de bordereau. « Principe d'armatures, pas un plan d'exécution. »

Les lits : en x la couche contre l'enrobage, en y la couche posée dessus (beton/position.py : c + Ø_x + Ø_y/2), sur les
deux faces ; un renfort est dans le plan des barres de sa direction (son axe est celui du treillis de cette direction).
Le moteur Dalles n'a pas de règle d'ancrage : l'ancrage vaut n × Ø (50 Ø par défaut, les Options) ; il n'a pas de règle
d'espacement : l'espacement libre minimal est celui du moteur Poutres (EC2 8.2(2), ``beton/poutre/dispositions.s_min``).

    resistances(el)                              -> mRd, d, A_s du treillis de chaque face (moteur Dalles)
    principe(ef, contour, mRd, ancrage, echelle) -> les renforts des bandes et les zones rouges (l'annexe A)
    calculer(el, modele, ef, scia=None)          -> le principe complet : renforts, barres, contrôles, lignes du principe
"""
import json
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from beton import position as POS
from beton.dalle import entree as DE
from beton.dalle import moteur as DM
from beton.poutre import dispositions as DI
from beton.poutre.entree import DEFAUTS_DETAIL

from . import geometrie as G

CLES = ("inf_x", "inf_y", "sup_x", "sup_y")
# les listes du module Dalles (le sélecteur de sa nappe de base) : Ø (mm) et espacements (mm) des treillis
DIAMETRES = DM.CANDIDATS_D
ESPACEMENTS = DM.CANDIDATS_ESP
DIAMETRES_RENFORT = (12, 16, 20)       # § 7.4 : le Ø automatique des renforts
ANCRAGE_DEFAUT = 50.0                  # × Ø : le moteur Dalles n'a pas de règle d'ancrage (§ 7.4, Options)
ACIER = 7850.0                         # kg/m³ : le poids des renforts, pour choisir le Ø le plus léger
H_MAX_MOTEUR = DE.H_MAX                # cm : l'épaisseur la plus forte que le moteur Dalles accepte
GRANULAT = float(DEFAUTS_DETAIL["dg"])  # mm : le plus gros granulat du moteur Poutres, pour l'espacement libre minimal


# ------------------------------------------------------------------ le moteur du module Dalles
_CACHE: "OrderedDict[str, Dict]" = OrderedDict()


def _moteur(entree: Dict) -> Dict:
    """``beton.dalle.moteur.verifier`` (gardé : une même section n'est calculée qu'une fois)."""
    cle = json.dumps(entree, sort_keys=True)
    r = _CACHE.get(cle)
    if r is None:
        r = DM.verifier(entree)
        _CACHE[cle] = r
        while len(_CACHE) > 400:
            _CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(cle)
    return r


def h_moteur(el: Dict) -> int:
    """L'épaisseur (cm, entière) que le moteur Dalles reçoit : h arrondi au cm INFÉRIEUR (d diminue : côté sécurité)."""
    return int(math.floor(el["dalle"]["h"] * 100 + 1e-9))


def hors_domaine(el: Dict) -> Optional[str]:
    """Pourquoi le moteur Dalles ne peut pas calculer cette dalle (texte), ou None."""
    if el["dalle"]["beton"] not in DE.classes_beton():
        return f"le béton {el['dalle']['beton']} n'est pas dans le moteur Dalles"
    if h_moteur(el) > H_MAX_MOTEUR:
        return f"h > {H_MAX_MOTEUR / 100:.2f} m : hors du moteur Dalles".replace(".", ",")
    return None


def _couches(el: Dict) -> Dict:
    """Les quatre faces de la section du moteur Dalles : en x la couche contre l'enrobage, en y la couche posée dessus."""
    a = el["armatures"]
    c = a["enrobage"] * 100.0
    faces = {}
    for cote in ("inf", "sup"):
        tx, ty = a[cote]["x"], a[cote]["y"]
        dessus_x = POS.dessus(POS.sur_parement(c, tx["d"]), tx["d"])
        faces[f"{cote}_x"] = {"couches": [{"type": "barres", "d": tx["d"], "esp": tx["e"], "dist": None}], "yG": None}
        faces[f"{cote}_y"] = {"couches": [{"type": "barres", "d": ty["d"], "esp": ty["e"],
                                           "dist": round(POS.sur_couche(dessus_x, ty["d"]), 6)}], "yG": None}
    return faces


def section(el: Dict, moments: Dict[str, float], V: float = 0.0) -> Dict:
    """Le calcul d'UN MÈTRE de dalle (b = 100 cm) par le moteur Dalles, avec les treillis choisis : ``moments``
    {"inf_x": m…} en kNm/m (≥ 0), V en kN/m. Rend la section calculée du moteur (grandeurs, vérifications)."""
    cle = {"inf_x": "Mx_inf", "sup_x": "Mx_sup", "inf_y": "My_inf", "sup_y": "My_sup"}
    s = {"nom": "A", "Mx_inf": 0.0, "Mx_sup": 0.0, "My_inf": 0.0, "My_sup": 0.0, "V": max(0.0, float(V)), "faces": _couches(el)}
    for f, m in moments.items():
        s[cle[f]] = max(0.0, float(m))
    entree = {"nom": "Fondation de grue", "b": 100, "h": h_moteur(el), "beton": el["dalle"]["beton"],
              "enrobage": round(el["armatures"]["enrobage"] * 100.0, 6), "sections": [s]}
    return _moteur(entree)["sections"][0]


def resistances(el: Dict) -> Dict:
    """Ce que reprend chaque treillis, par le moteur Dalles : {"mRd": {cle: kNm/m}, "d": {cle: cm}, "As": {cle: mm²/m},
    "fyd", "gamma_s", "jeu", "arrondi"} — mRd = A_s ÷ A_s,req(1 kNm/m)."""
    sec = section(el, {f: 1.0 for f in CLES})
    g = sec["grandeurs"]
    return {"mRd": {f: g[f"As_{f}"] / g[f"As_req_{f}"] for f in CLES}, "d": {f: g[f"d_{f}"] for f in CLES},
            "As": {f: g[f"As_pm_{f}"] for f in CLES}, "fyd": g["fyd"], "gamma_s": g["gamma_s"],
            "jeu": g["jeu_premier_lit"], "arrondi": g["arrondi"], "h": g["h"]}


# ------------------------------------------------------------------ le principe (annexe A, principe_armatures)
def _troncons(masque: np.ndarray) -> List[Tuple[int, int]]:
    """Les tronçons [i0, i1] (indices) où le masque est vrai."""
    out, i, n = [], 0, len(masque)
    while i < n:
        if masque[i]:
            j = i
            while j + 1 < n and masque[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _fusion(rects: List[Dict]) -> List[Dict]:
    """Fusionne les rectangles qui se chevauchent (l'aire cumulée, le m_dim le plus fort), dans l'ordre de l'annexe A."""
    rects = [dict(r) for r in rects]
    fusion = True
    while fusion:
        fusion = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                A, B = rects[i], rects[j]
                if (A["x0"] < B["x1"] - 1e-9 and B["x0"] < A["x1"] - 1e-9
                        and A["y0"] < B["y1"] - 1e-9 and B["y0"] < A["y1"] - 1e-9):
                    rects[i] = {"x0": min(A["x0"], B["x0"]), "x1": max(A["x1"], B["x1"]),
                                "y0": min(A["y0"], B["y0"]), "y1": max(A["y1"], B["y1"]),
                                "aire": A["aire"] + B["aire"], "m_dim": max(A["m_dim"], B["m_dim"])}
                    del rects[j]
                    fusion = True
                    break
            if fusion:
                break
    return rects


def etiqueter(masque: np.ndarray) -> Tuple[np.ndarray, int]:
    """Les groupes de nœuds voisins (4-connexité : gauche, droite, haut, bas) d'une grille booléenne, numérotés 1… dans
    l'ordre où le balayage ligne par ligne les rencontre — ``scipy.ndimage.label`` et sa structure par défaut."""
    ny, nx = masque.shape
    lab = np.zeros((ny, nx), int)
    n = 0
    for j0 in range(ny):
        for i0 in range(nx):
            if masque[j0, i0] and not lab[j0, i0]:
                n += 1
                lab[j0, i0] = n
                pile = [(j0, i0)]
                while pile:
                    j, i = pile.pop()
                    for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
                        if 0 <= jj < ny and 0 <= ii < nx and masque[jj, ii] and not lab[jj, ii]:
                            lab[jj, ii] = n
                            pile.append((jj, ii))
    return lab, n


def principe(ef: Dict, contour: Sequence, mRd: Dict[str, float], ancrage: float = 0.80, echelle: Optional[Dict] = None,
             mRd_bandes: Optional[Dict[str, float]] = None) -> Dict:
    """Le principe d'armatures de l'annexe A (``principe_armatures``), à l'identique (test T19, A1).
    ``mRd`` : ce que reprend le treillis (kNm/m) par face et direction ; ``mRd_bandes`` : celui des bandes s'il diffère (une
    bande d'un châssis tourné prend le plus faible des deux directions, § 7.2) ; ``echelle`` : le rapport SCIA ÷ module par
    face et direction (1 par défaut, § 8.2) ; ``ancrage`` (m).
    1. Bandes : là où M ÷ largeur dépasse mRd (hors des plots coulés), renfort de bande sur toute la largeur, de la limite +
       ancrage de chaque côté, sans sortir de la bande ; m_dim = le plus grand M ÷ largeur du tronçon.
    2. Hors des bandes : là où le moment au point dépasse mRd (mx pour les barres en x, my en y), zone rouge ; chaque
       groupe de nœuds voisins donne un renfort local (une demi-maille autour + l'ancrage dans le sens des barres, sans
       sortir de la dalle ; fusionnés s'ils se chevauchent ; m_dim = le plus grand moment du groupe). Une bande ne couvre
       que ses barres ; les nœuds des plots coulés sont exclus.
    Rend {"bandes": [{"bande", "cle", "s0", "s1", "x0", "x1", "y0", "y1", "m_dim", "coins"}], "locaux": {cle: [{"x0", "x1",
    "y0", "y1", "aire" (m² de nœuds rouges), "m_dim"}]}, "rouge": {cle: masque nny × nnx}}."""
    echelle = echelle or {}
    mRd_b = mRd_bandes or mRd
    xs, ys = [p[0] for p in contour], [p[1] for p in contour]
    X0, X1, Y0, Y1 = min(xs), max(xs), min(ys), max(ys)
    g = ef["grille"]
    nnx, nny = g["nx"], g["ny"]
    X = g["x0"] + np.arange(nnx)[None, :] * g["a"] + 0 * np.arange(nny)[:, None]
    Y = g["y0"] + np.arange(nny)[:, None] * g["b"] + 0 * np.arange(nnx)[None, :]
    out = {"bandes": [], "locaux": {}, "rouge": {}}
    dans_bande = {"x": np.zeros((nny, nnx), bool), "y": np.zeros((nny, nnx), bool)}
    for bd in ef["bandes"]:
        sens = "x" if bd["nom"].startswith("BX") else "y"
        p0, p1 = np.asarray(bd["debut"], float), np.asarray(bd["fin"], float)
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / L
        n = np.array([-u[1], u[0]])
        b = bd["largeur"]
        T = (X - p0[0]) * n[0] + (Y - p0[1]) * n[1]
        S = (X - p0[0]) * u[0] + (Y - p0[1]) * u[1]
        dans_bande[sens] |= (np.abs(T) <= b / 2 + 1e-9) & (S >= -1e-9) & (S <= L + 1e-9)
        s = bd["s"]
        for face in ("inf", "sup"):
            cle = f"{face}_{sens}"
            m = (bd["Mmax"] if face == "inf" else -bd["Mmin"]) / b * echelle.get(cle, 1.0)
            dep = (m > mRd_b[cle] + 1e-9) & ~bd["dans_plot"]
            for i0, i1 in _troncons(dep):
                s0, s1 = max(0.0, float(s[i0]) - ancrage), min(L, float(s[i1]) + ancrage)
                c = [p0 + s0 * u + n * b / 2, p0 + s1 * u + n * b / 2, p0 + s1 * u - n * b / 2, p0 + s0 * u - n * b / 2]
                out["bandes"].append({"bande": bd["nom"], "cle": cle, "s0": s0, "s1": s1,
                                      "x0": min(v[0] for v in c), "x1": max(v[0] for v in c),
                                      "y0": min(v[1] for v in c), "y1": max(v[1] for v in c),
                                      "m_dim": float(m[i0:i1 + 1].max()), "coins": [tuple(map(float, v)) for v in c]})
    # les tronçons d'une même bande qui se chevauchent : fusionnés
    fus = []
    for r in out["bandes"]:
        if fus and fus[-1]["bande"] == r["bande"] and fus[-1]["cle"] == r["cle"] and r["s0"] <= fus[-1]["s1"] + 1e-9:
            f = fus[-1]
            f["s1"] = max(f["s1"], r["s1"])
            f["m_dim"] = max(f["m_dim"], r["m_dim"])
            for k_, fn in (("x0", min), ("y0", min), ("x1", max), ("y1", max)):
                f[k_] = fn(f[k_], r[k_])
            f["coins"] = [f["coins"][0], r["coins"][1], r["coins"][2], f["coins"][3]]
        else:
            fus.append(dict(r))
    out["bandes"] = fus
    for cle in CLES:
        v = ef["noeuds"][cle]
        m = (v if cle.startswith("inf") else -v) * echelle.get(cle, 1.0)
        rouge = (m > mRd[cle] + 1e-9) & g["actif"] & ~dans_bande[cle[-1]]
        out["rouge"][cle] = rouge
        lab, nb = etiqueter(rouge)
        rects = []
        for k in range(1, nb + 1):
            jj, ii = np.nonzero(lab == k)
            r = {"x0": max(X0, g["x0"] + (ii.min() - 0.5) * g["a"]), "x1": min(X1, g["x0"] + (ii.max() + 0.5) * g["a"]),
                 "y0": max(Y0, g["y0"] + (jj.min() - 0.5) * g["b"]), "y1": min(Y1, g["y0"] + (jj.max() + 0.5) * g["b"]),
                 "aire": len(ii) * g["a"] * g["b"], "m_dim": float(m[jj, ii].max())}
            lo, hi, LO, HI = ("x0", "x1", X0, X1) if cle.endswith("x") else ("y0", "y1", Y0, Y1)
            r[lo] = max(LO, r[lo] - ancrage)
            r[hi] = min(HI, r[hi] + ancrage)
            rects.append(r)
        out["locaux"][cle] = sorted(_fusion(rects), key=lambda r: (round(r["y0"], 6), round(r["x0"], 6)))
    return out


# ------------------------------------------------------------------ les barres et les contrôles du moteur Dalles
def aire_barre(d_mm: float) -> float:
    return math.pi * d_mm * d_mm / 4.0


def barres(renfort_pm: float, largeur: float, d_mm: float) -> int:
    """Le nombre de barres d'un renfort (§ 7.4) : ⌈renfort par mètre (mm²/m) × largeur (m) ÷ aire d'une barre⌉."""
    if renfort_pm <= 1e-9:
        return 0
    return int(math.ceil(renfort_pm * largeur / aire_barre(d_mm) - 1e-9))


def espacement_ok(largeur: float, n: int, d_mm: float) -> bool:
    """Les n barres réparties sur la largeur : l'espacement libre (largeur ÷ n − Ø) tient le minimum du moteur Poutres."""
    if n <= 1:
        return True
    return largeur * 1000.0 / n - d_mm >= DI.s_min(d_mm, d_mm, GRANULAT) - 1e-9


def _sens_bande(nom: str) -> str:
    return "x" if nom.startswith("BX") else "y"


def tourne(el: Dict) -> bool:
    """Le châssis est tourné : ses axes ne sont plus ceux de la dalle (β ≠ 0° et ≠ 180°)."""
    return abs(math.sin(math.radians(el["grue"]["beta"]))) > 1e-9


def _mrd_bandes(el: Dict, mRd: Dict[str, float]) -> Dict[str, float]:
    """Ce que reprend le treillis dans une bande : sa direction ; châssis tourné (β ≠ 0) : le plus faible des deux
    directions de la face (§ 7.2)."""
    if not tourne(el):
        return dict(mRd)
    out = {}
    for face in ("inf", "sup"):
        m = min(mRd[f"{face}_x"], mRd[f"{face}_y"])
        out[f"{face}_x"] = out[f"{face}_y"] = m
    return out


def _cle_section(el: Dict, cle: str, beta_tourne: bool, mRd: Dict) -> str:
    """La face et la direction dont la section sert à A_s() d'un renfort de bande : la sienne, ou (châssis tourné) la plus
    faible des deux directions de la face."""
    if not beta_tourne:
        return cle
    face = cle.split("_")[0]
    return min((f"{face}_x", f"{face}_y"), key=lambda f: (mRd[f], f))


def _evaluer(el: Dict, cle: str, cle_sec: str, m_dim: float, largeur: float, longueur: float, d_mm: float,
             As_tr: float) -> Dict:
    """Un renfort pour un Ø : A_s(m_dim) du moteur Dalles (le besoin : max(A_s,req ; A_s,min)), le renfort au mètre, les
    barres, l'espacement, le poids ; et la hauteur de béton (h_min du moteur pour m_dim)."""
    sec = section(el, {cle_sec: m_dim})
    g = sec["grandeurs"]
    besoin = g[f"besoin_{cle_sec}"]
    renfort = max(0.0, besoin - As_tr)
    n = barres(renfort, largeur, d_mm)
    return {"d": d_mm, "n": n, "besoin": besoin, "As_req": g[f"As_req_{cle_sec}"], "As_min": g[f"As_min_{cle_sec}"],
            "renfort_pm": renfort, "esp": largeur * 1000.0 / n if n else None, "serre": not espacement_ok(largeur, n, d_mm),
            "poids": n * aire_barre(d_mm) * 1e-6 * longueur * ACIER, "h_min": g["h_min"], "hu_min": g["hu_min"],
            "yG": g["yG_hauteur"], "etat_hauteur": g["etat_hauteur"], "d_utile": g[f"d_{cle_sec}"]}


def _ancrage(el: Dict, d_mm: float) -> float:
    return el["armatures"]["ancrage"] * d_mm / 1000.0


def calculer(el: Dict, modele: Dict, ef: Dict, scia: Optional[Dict] = None) -> Dict:
    """Le principe d'armatures complet de la fondation (§ 7), sur l'enveloppe des efforts ``ef`` (``efforts.enveloppe``).
    ``scia`` : les efforts SCIA recopiés (``scia.rapports`` : l'échelle des moments, V SCIA), ou None (§ 8.2).

    Rend {"res" (resistances), "mRd_bandes", "renforts": [{"rep" (R1…), "type" ("bande" | "local"), "bande" (ou None),
          "cle", "face", "sens", "x0", "x1", "y0", "y1", "coins", "longueur", "largeur" (m), "m_dim" (kNm/m), "d" (Ø),
          "n", "esp" (mm), "serre", "besoin", "As_tr", "renfort_pm" (mm²/m), "h_min", "etat_hauteur", "aire" (local),
          "ancrage" (m)}], "zones": {cle: {"n", "aire", "max"}}, "rouge": {cle: masque}, "tranchant": {bande: {…}},
          "hauteur": {"h_min", "h", "taux", "etat", "rep"}, "treillis": la section du treillis seul (A_s,min, hauteur),
          "etat" ("ok" | "nok" | "v"), "taux", "alertes": [(texte, niveau)], "non_verifies": [...], "u": texte,
          "echelle", "hors_domaine"} — ou {"hors_domaine": texte} quand le moteur Dalles ne peut pas calculer."""
    echelle = scia["echelle"] if scia else None
    V_scia = (scia or {}).get("V") or {}
    hd = hors_domaine(el)
    if hd:
        return {"hors_domaine": hd, "etat": "v", "taux": None, "renforts": [], "alertes": [(f"Armatures : non vérifiées — {hd}", "v")],
                "non_verifies": ["les armatures (" + hd + ")"]}
    res = resistances(el)
    mRd = res["mRd"]
    tourne_ = tourne(el)
    mRd_b = _mrd_bandes(el, mRd)
    a = el["armatures"]
    contour = modele["contour"]
    choix_d = DIAMETRES_RENFORT if a["renfort"] == "auto" else (int(a["renfort"]),)
    # un principe par Ø candidat (l'ancrage = n × Ø dépend du Ø), puis, face par face, le Ø le plus léger
    essais = {d: principe(ef, contour, mRd, _ancrage(el, d), echelle, mRd_b) for d in choix_d}
    retenus: Dict[str, Tuple[int, List[Dict]]] = {}
    for cle in CLES:
        meilleur = None
        for d in choix_d:
            pa = essais[d]
            liste = []
            for r in pa["bandes"]:
                if r["cle"] != cle:
                    continue
                largeur, longueur = ef["largeur"], r["s1"] - r["s0"]
                cs = _cle_section(el, cle, tourne_, mRd)
                ev = _evaluer(el, cle, cs, r["m_dim"], largeur, longueur, d, res["As"][cs])
                liste.append(dict(r, type="bande", largeur=largeur, longueur=longueur, As_tr=res["As"][cs], cle_section=cs, **ev))
            for r in pa["locaux"][cle]:
                if cle.endswith("x"):
                    longueur, largeur = r["x1"] - r["x0"], r["y1"] - r["y0"]
                else:
                    longueur, largeur = r["y1"] - r["y0"], r["x1"] - r["x0"]
                ev = _evaluer(el, cle, cle, r["m_dim"], largeur, longueur, d, res["As"][cle])
                liste.append(dict(r, type="local", bande=None, cle=cle, largeur=largeur, longueur=longueur, As_tr=res["As"][cle],
                                  cle_section=cle, **ev))
            poids = sum(x["poids"] for x in liste)
            serre = any(x["serre"] for x in liste)
            # le plus léger ; à poids égal (au gramme), le plus gros Ø ; un espacement trop serré ne passe qu'en dernier
            note = (serre, round(poids, 3), -d)
            if meilleur is None or note < meilleur[0]:
                meilleur = (note, d, liste)
        retenus[cle] = (meilleur[1], meilleur[2])
    # les repères : en bas puis en haut, les renforts des bandes puis les renforts locaux
    renforts = []
    for face in ("inf", "sup"):
        for typ in ("bande", "local"):
            for cle in (f"{face}_x", f"{face}_y"):
                d, liste = retenus[cle]
                for r in liste:
                    if r["type"] == typ:
                        renforts.append(dict(r, face=face, sens=cle[-1], ancrage=_ancrage(el, d)))
    for i, r in enumerate(renforts, 1):
        r["rep"] = f"R{i}"
    rouge = essais[choix_d[0]]["rouge"]
    g = ef["grille"]
    zones = {cle: {"n": sum(1 for r in retenus[cle][1] if r["type"] == "local"),
                   "aire": float(rouge[cle].sum() * g["a"] * g["b"]),
                   "max": max([r["m_dim"] for r in retenus[cle][1] if r["type"] == "local"], default=None)} for cle in CLES}
    # l'effort tranchant : V à d ÷ largeur, contrôlé par le moteur Dalles (τ ≤ τ_adm,I) ; V à d illisible : non vérifié
    tranchant = {}
    for bd in ef["bandes"]:
        V = bd["V_d"]
        V_s = V_scia.get(_sens_bande(bd["nom"]))
        if V_s is not None:                                       # (§ 8.2 : la valeur SCIA va directement au moteur)
            V = abs(V_s)
        if V is None:
            tranchant[bd["nom"]] = {"V": None, "v": None, "tau": None, "tau_adm": None, "taux": None, "etat": "non_verifie"}
            continue
        v = V / bd["largeur"]
        sec = section(el, {}, v)
        gg = sec["grandeurs"]
        tranchant[bd["nom"]] = {"V": V, "v": v, "tau": gg["tau"], "tau_adm": gg["tau_adm"], "taux": gg["tau"] / gg["tau_adm"],
                                "etat": gg["etat_tau"], "cas": bd["cas_V_d"], "scia": V_s is not None}
    # le treillis seul : A_s,min et la hauteur pour le plus fort moment qu'il porte (au plus mRd)
    ex = ef["extremes"]
    m_tr = {cle: min(mRd[cle], max(0.0, (ex[cle] if cle.startswith("inf") else -ex[cle]) * (echelle or {}).get(cle, 1.0))) for cle in CLES}
    tr = section(el, m_tr)
    gt = tr["grandeurs"]
    treillis = {"As_min": {cle: gt[f"As_min_{cle}"] for cle in CLES}, "As": res["As"],
                "ok_min": {cle: gt[f"As_{cle}"] >= gt[f"As_min_{cle}"] - 1e-9 for cle in CLES}, "h_min": gt["h_min"],
                "etat_hauteur": gt["etat_hauteur"], "m": m_tr}
    # la hauteur de béton (§ 7.2) : le plus exigeant des renforts (h_min du moteur pour m_dim) et du treillis seul ;
    # h_min = h_u,min + y_G ≤ h  ⟺  d_min = h_u,min ≤ d = h − y_G : le taux est d_min ÷ d (le verdict du moteur)
    h_cm = float(h_moteur(el))
    cands = [(r["h_min"], r["hu_min"], r["yG"], r["rep"]) for r in renforts] + [(gt["h_min"], gt["hu_min"], gt["yG_hauteur"], None)]
    h_min, hu_min, yG, rep_h = max(cands, key=lambda x: x[0])
    hauteur = {"h_min": h_min, "hu_min": hu_min, "yG": yG, "d": h_cm - yG, "h": h_cm, "taux": hu_min / max(h_cm - yG, 1e-9),
               "etat": "ok" if h_min <= h_cm + 1e-9 else "nok", "rep": rep_h}
    alertes = []
    if hauteur["etat"] != "ok":
        alertes.append(("Hauteur de béton insuffisante : épaissir la dalle", "r"))
    if any(t["etat"] == "nok" for t in tranchant.values()):
        alertes.append(("Effort tranchant : τ > τ_adm — épaissir la dalle", "r"))
    if not all(treillis["ok_min"].values()):
        alertes.append(("Treillis sous le minimum du moteur (A_s,min) : choisir un treillis plus fort", "r"))
    if any(r["serre"] for r in renforts):
        alertes.append(("Renforts serrés : l'espacement libre est sous le minimum — à revoir", "v"))
    if any(t["etat"] == "non_verifie" for t in tranchant.values()):
        alertes.append(("V à d ne se lit pas (zones chargées trop proches) : effort tranchant non vérifié", "v"))
    taux = max([hauteur["taux"]] + [t["taux"] for t in tranchant.values() if t["taux"] is not None])
    if any(n == "r" for _, n in alertes):
        etat = "nok"
    elif alertes:
        etat = "v"
    else:
        etat = "ok"
    pl = modele.get("plots")
    nv = ["le poinçonnement"]
    if pl and pl["type"] == "coule":
        nv.append("le ferraillage des plots coulés en place")
    # le U de fermeture : le treillis le plus fort des deux faces, sur tout le pourtour
    forts = max(((a[c][s]["d"], -a[c][s]["e"]) for c in ("inf", "sup") for s in ("x", "y")))
    u = f"U de fermeture : Ø{forts[0]} e {_fr_cm(-forts[1])} cm sur tout le pourtour ({_fr_m(G.perimetre(contour))} m)"
    return {"res": res, "mRd_bandes": mRd_b, "tourne": tourne_, "renforts": renforts, "zones": zones, "rouge": rouge,
            "tranchant": tranchant, "hauteur": hauteur, "treillis": treillis, "etat": etat, "taux": taux, "alertes": alertes,
            "non_verifies": nv, "u": u, "echelle": dict(echelle or {}), "hors_domaine": None,
            "plot_coule": bool(pl and pl["type"] == "coule"), "diametres": {cle: retenus[cle][0] for cle in CLES}}


def _fr_cm(mm: float) -> str:
    return f"{mm / 10:g}".replace(".", ",")


def _fr_m(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")

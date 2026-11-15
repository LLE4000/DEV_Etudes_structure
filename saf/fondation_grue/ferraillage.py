# -*- coding: utf-8 -*-
"""Fondation de grue — le FERRAILLAGE (lot 4, § 5 du cahier des charges du 15/11/2026), méthode du bureau, ELS.

Le calcul des aciers APPELLE le moteur du module « Poutres béton armé » (``beton.poutre.moteur.verifier``), sans le
modifier : d (jeu de pose, arrondi), l'acier demandé (« besoin » : max(A_s,req ; A_s,min)), la hauteur mini, l'effort
tranchant. Une dalle n'a pas d'étrier : le moteur en exige une ligne, on lui donne une épingle Ø6 FICTIVE et chaque lit
avec sa position réelle imposée (« pos ») — l'épingle ne déplace donc aucune barre — ; la vérification « pas » des
étriers, propre à une poutre, est ignorée, comme les dispositions constructives d'une poutre (``statut_complet``).

    calculer(el, modele, ef)   -> le ferraillage complet de la fondation (voir ``calculer``)

LES NAPPES (§ 5.1) : les barres en x à l'EXTÉRIEUR, les barres en y à l'INTÉRIEUR, en bas comme en haut. Un renfort est
posé dans le plan des barres de sa direction, entre les barres du treillis (« comptés comme des barres filantes ») ; le
lit d'une direction est pris, pour d, au centre de sa plus GROSSE barre (côté sécurité : ≤ 4 mm). Les barres en y
reposent sur la plus grosse barre en x de leur face.

CHOIX FAITS ICI, À REVOIR AVEC LE BUREAU (le cahier n'en dit rien) :
- l'acier minimal « 0,25·A_s,req de la face opposée » est calculé avec la face opposée la plus défavorable (un lit Ø20,
  le plus gros renfort) : les deux faces se dimensionnent alors indépendamment ;
- un renfort se pose dans les intervalles du treillis : un tous les m intervalles (m = 1 à 4) ou deux par intervalle,
  si l'espacement libre du moteur poutre (``dispositions.s_min``) le permet ;
- les C d'effort tranchant ont des retours de 10 Ø ; une rangée de C compte une branche par barre de treillis de la
  largeur de la bande.

Les éléments hors des zones de renfort sont contrôlés UN PAR UN (R1) : l'acier demandé y est celui du moteur, obtenu par
sa linéarité (A_s,req ∝ M à d donné, A_s,min indépendant de M) à partir de deux appels par configuration ; les éléments
les plus sollicités sont RECONTRÔLÉS par un appel direct du moteur à chaque calcul (``NB_CONTROLES``) — un écart lève
une erreur. tests/test_saf_grue_ferraillage.py rejoue le moteur sur tous les éléments de la dalle type.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from beton.poutre import dispositions as DI
from beton.poutre import geometrie as GP
from beton.poutre import moteur as MO
from beton.poutre import taux as TX
from beton.poutre.entree import DEFAUTS_DETAIL

from . import efforts as EF
from . import geometrie as G
from . import projet as PR

TREILLIS_CHOIX = PR.TREILLIS[1:]          # § 5.2 : « 10-15 », « 12-15 », « 10-10 », « 12-10 » (Ø mm - espacement cm)
D_RENFORTS = (12, 16, 20)                 # § 5.3 : renforts de bandes et locaux
D_TRANCHANT = (10, 12)                    # § 5.5 : C verticaux d'effort tranchant
D_OPPOSEE = 20                            # la face opposée la plus défavorable (le plus gros renfort), voir l'en-tête
MULTIPLES = (1, 2, 3, 4)                  # un renfort tous les m intervalles du treillis (et 2 par intervalle si possible)
RETOUR_C = 10                             # les retours d'un C d'effort tranchant : 10 Ø
PAS_LONGUEUR = 0.05                       # § 6.3 : les barres regroupées par longueur, au 5 cm près
EPINGLE_FICTIVE = [{"type": "epingle", "d": 6}]
NB_CONTROLES = 12                         # les éléments les plus sollicités, recontrôlés par un appel direct du moteur
DG = DEFAUTS_DETAIL["dg"]                 # le plus gros granulat du moteur poutre (mm)
EPS = 1e-9


# ------------------------------------------------------------------ petites règles
def treillis(code: str) -> Tuple[int, float]:
    """(Ø en mm, espacement en cm) d'un treillis : « 12-15 » → (12, 15.0)."""
    d, e = code.split("-")
    return int(d), float(e)


def libelle_treillis(code: str) -> str:
    d, e = treillis(code)
    return f"Ø{d} e = {e:g} cm"


def as_pm(d: float, e_cm: float) -> float:
    """La section au mètre (mm²/m) de barres de Ø d (mm) tous les e (cm)."""
    return GP.aire_barre(d) * 100.0 / e_cm


def masse(d: float) -> float:
    """La masse d'un mètre de barre (kg/m) : celle du moteur poutre (ρ·π·Ø²/4, ρ = 7 850 kg/m³)."""
    return TX.masse_lineique(d)


def recouvrement(el: Dict, d: float) -> float:
    """La longueur de recouvrement (m) : la règle du logiciel n'existe pas (le moteur poutre ne compte ni ancrage ni
    recouvrement) → le champ « Longueur de recouvrement » des Options, 50 Ø par défaut (§ 5.7)."""
    return el["options"]["recouvrement"] * d / 1000.0


def troncons(el: Dict, L: float, d: float) -> Tuple[int, float]:
    """(nombre de morceaux, longueur d'un morceau) d'une barre de longueur L (m) : au-delà de la longueur maxi des barres
    (12 m), des recouvrements (§ 5.7)."""
    lmax, lr = el["options"]["l_max"], recouvrement(el, d)
    if L <= lmax + EPS:
        return 1, L
    n = math.ceil((L - lr) / (lmax - lr) - EPS)
    return n, (L + (n - 1) * lr) / n


def longueur_bordereau(L: float) -> float:
    """Une longueur arrondie au 5 cm SUPÉRIEUR (le regroupement du bordereau)."""
    return math.ceil(L / PAS_LONGUEUR - 1e-6) * PAS_LONGUEUR


# ------------------------------------------------------------------ l'appel du moteur poutre
def _lit(c_cm: float, d: float, dessous: float = 0.0) -> Dict:
    """Un lit pour le moteur poutre : sa position RÉELLE imposée (cm) = enrobage + Ø des barres de dessous + Ø/2."""
    return {"n": 1, "d": int(d), "pos": round(c_cm + dessous / 10.0 + d / 20.0, 6)}


def _couche(c_cm: float, sens: str, d: float, dessous_y: float) -> Dict:
    """Le lit d'une direction : en x à l'extérieur, en y posé sur la plus grosse barre en x (``dessous_y``)."""
    return _lit(c_cm, d, dessous_y if sens == "y" else 0.0)


def _opposee(c_cm: float, sens: str) -> Dict:
    """La face opposée la plus défavorable (voir l'en-tête) : un lit Ø20, en y posé sur un Ø20."""
    return _couche(c_cm, sens, D_OPPOSEE, D_OPPOSEE)


def poutre(el: Dict, b_cm: float, h_cm: float, sections: List[Dict]) -> List[Dict]:
    """Les sections rendues par le moteur poutre (``verifier``) pour une « poutre » b × h (cm) de la fondation."""
    o = el["options"]
    p = {"b": float(b_cm), "h": float(h_cm), "beton": el["dalle"]["beton"], "gamma_s": o["gamma_s"],
         "enrobage": el["armatures"]["enrobage"] * 100.0, "jeu_premier_lit": o["jeu"], "arrondi": o["arrondi"],
         "jeu_entre_lits": 0.0, "sections": sections}
    return MO.verifier(p)["sections"]


def section(M_inf: float, M_sup: float, V: float, lit_inf: Dict, lit_sup: Dict, etriers=None) -> Dict:
    """Une section pour le moteur : M⁺ (fibre inférieure tendue), M⁻ (fibre supérieure tendue, en valeur absolue), V."""
    return {"M_inf": max(float(M_inf), 0.0), "M_sup": max(float(M_sup), 0.0), "V": max(float(V), 0.0),
            "inf": [dict(lit_inf)], "sup": [dict(lit_sup)], "etriers": etriers or EPINGLE_FICTIVE}


def _lits(face: str, lit_face: Dict, lit_opp: Dict) -> Tuple[Dict, Dict]:
    return (lit_face, lit_opp) if face == "inf" else (lit_opp, lit_face)


# ------------------------------------------------------------------ le contexte d'un calcul
class _Contexte:
    """Ce qui ne change pas d'une face à l'autre : la dalle, l'enveloppe des efforts, les masques des éléments."""

    def __init__(self, el: Dict, modele: Dict, ef: Dict):
        self.el, self.m, self.ef = el, modele, ef
        self.c = el["armatures"]["enrobage"]
        self.c_cm = self.c * 100.0
        self.h = modele["h"]
        self.h_cm = self.h * 100.0
        pl = modele.get("plots")
        self.coule = bool(pl and pl["type"] == "coule")
        self.hp = pl["hauteur"] if self.coule else 0.0
        ch = ef["champ"]
        self.ch = ch
        self.controle = ~ch["zone_renfort"] & ~ch["plot"]          # R1 : hors des zones de renfort et des plots coulés
        xs = [p[0] for p in modele["contour"]]
        ys = [p[1] for p in modele["contour"]]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self.bandes = {b["nom"]: b for b in ef["bandes"]}

    def moments(self, sens: str) -> Tuple[np.ndarray, np.ndarray]:
        """(M⁺, |M⁻|) de calcul par mètre au centre des éléments, dans une direction (§ 4.4, torsion selon l'option)."""
        return np.maximum(self.ch[f"inf_{sens}"], 0.0), np.maximum(-self.ch[f"sup_{sens}"], 0.0)


def _coefficients(ctx: _Contexte, face: str, lit_face: Dict, lit_opp: Dict) -> Tuple[float, float, float]:
    """(k de la face, k de la face opposée, A_s,min de base) du moteur pour b = 1 m et la hauteur de la dalle :
    A_s,req = k·M (mm² pour M en kNm/m), lus sur un appel du moteur à M = 100 kNm/m."""
    opp = "sup" if face == "inf" else "inf"
    li, ls = _lits(face, lit_face, lit_opp)
    g = poutre(ctx.el, 100.0, ctx.h_cm, [section(100.0, 100.0, 0.0, li, ls)])[0]["grandeurs"]
    return g[f"As_req_{face}"] / 100.0, g[f"As_req_{opp}"] / 100.0, g["As_min_base"]


def besoin_elements(ctx: _Contexte, face: str, sens: str, lit_face: Dict, lit_opp: Dict, M_inf, M_sup) -> np.ndarray:
    """L'acier demandé par le moteur (mm²/m) en chaque élément, pour une configuration de lits : max(A_s,req ;
    A_s,min de base ; 0,25·A_s,req opposé) — la règle du moteur (``K_AS_MIN_OPPOSEE``), avec ses coefficients."""
    kf, ko, base = _coefficients(ctx, face, lit_face, lit_opp)
    Mf, Mo = (M_inf, M_sup) if face == "inf" else (M_sup, M_inf)
    return np.maximum(np.maximum(kf * Mf, base), MO.K_AS_MIN_OPPOSEE * ko * Mo)


def _recontroler(ctx: _Contexte, face: str, besoin: np.ndarray, lits_face: List[Dict], lit_opp: Dict, M_inf, M_sup,
                 idx: np.ndarray) -> None:
    """Les éléments ``idx`` recontrôlés par un appel DIRECT du moteur poutre : un écart lève une erreur."""
    if not len(idx):
        return
    secs = [section(M_inf[i], M_sup[i], 0.0, *_lits(face, lits_face[i], lit_opp)) for i in idx]
    rs = poutre(ctx.el, 100.0, ctx.h_cm, secs)
    for i, r in zip(idx, rs):
        v = r["grandeurs"][f"besoin_{face}"]
        if abs(v - besoin[i]) > 1e-6 * max(1.0, v):
            raise RuntimeError(f"ferraillage : l'acier demandé par élément s'écarte du moteur poutre ({besoin[i]} ≠ {v})")


# ------------------------------------------------------------------ le choix des barres de renfort
def options_espacement(d_r: float, d_t: float, e_t: float) -> List[Tuple[float, int]]:
    """Les espacements (cm) possibles d'un renfort Ø d_r posé dans les intervalles d'un treillis Ø d_t tous les e_t cm :
    un tous les m intervalles, ou deux par intervalle si l'espacement libre du moteur poutre (s_min) le permet."""
    out = [(e_t * m, m) for m in MULTIPLES]
    libre_t = e_t * 10.0 / 3.0 - (d_r + d_t) / 2.0            # (mm) deux renforts par intervalle, au tiers
    libre_r = e_t * 10.0 / 3.0 - d_r
    if libre_t >= DI.s_min(d_r, d_t, DG) - EPS and libre_r >= DI.s_min(d_r, d_r, DG) - EPS:
        out.append((e_t / 2.0, 0))
    return out


def choisir_renfort(manque: Dict[int, float], d_t: float, e_t: float) -> Optional[Tuple[int, float]]:
    """(Ø, espacement en cm) du renfort le plus LÉGER qui couvre le manque (mm²/m, par Ø : il dépend de d) ; à poids
    égal, le plus gros Ø (§ 5.3). None si aucun ne suffit."""
    meilleur = None
    for d in D_RENFORTS:
        if manque[d] <= EPS:
            continue
        for s, _m in options_espacement(d, d_t, e_t):
            if as_pm(d, s) + EPS < manque[d]:
                continue
            cle = (round(masse(d) * 100.0 / s, 9), -d)
            if meilleur is None or cle < meilleur[0]:
                meilleur = (cle, (d, s))
    return None if meilleur is None else meilleur[1]


def nombre_barres(largeur_m: float, s_cm: float) -> int:
    """Le nombre de barres réparties sur une largeur (m) pour un espacement d'au plus s (cm)."""
    return max(1, math.ceil(largeur_m * 100.0 / s_cm - 1e-6))


# ------------------------------------------------------------------ les barres d'un contour
def segments(contour: Sequence, sens: str, v: float) -> List[Tuple[float, float]]:
    """Les morceaux, dans la dalle, de la droite y = v (barres en x) ou x = v (barres en y)."""
    k, j = (1, 0) if sens == "x" else (0, 1)
    xs = []
    n = len(contour)
    for i in range(n):
        p, q = contour[i], contour[(i + 1) % n]
        if (p[k] <= v < q[k]) or (q[k] <= v < p[k]):
            xs.append(p[j] + (v - p[k]) * (q[j] - p[j]) / (q[k] - p[k]))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def barres_filantes(contour: Sequence, sens: str, c: float, e: float, borne: Optional[Tuple[float, float, float, float]] = None
                    ) -> List[Tuple[float, float, float]]:
    """Les barres en ``sens`` tous les e (m), à c des bords : [(position transversale, début, fin)] ; ``borne`` (x0, y0,
    x1, y1) limite la nappe à un rectangle (un renfort local)."""
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    lo, hi = (min(ys), max(ys)) if sens == "x" else (min(xs), max(xs))
    a0, a1 = (min(xs), max(xs)) if sens == "x" else (min(ys), max(ys))
    if borne is not None:
        lo, hi = (borne[1], borne[3]) if sens == "x" else (borne[0], borne[2])
        a0, a1 = (borne[0], borne[2]) if sens == "x" else (borne[1], borne[3])
        n = nombre_barres(hi - lo, e * 100.0)
        pas = (hi - lo) / n
        positions = [lo + (i + 0.5) * pas for i in range(n)]
    else:
        n = int(math.floor((hi - lo - 2 * c) / e + 1e-9)) + 1
        positions = [lo + c + i * e for i in range(n)]
    out = []
    for v in positions:
        for u0, u1 in segments(contour, sens, v):
            u0, u1 = max(u0 + c, a0 if borne is not None else -1e18), min(u1 - c, a1 if borne is not None else 1e18)
            if u1 - u0 > EPS:
                out.append((v, u0, u1))
    return out


# ------------------------------------------------------------------ une face
def _bande_face(ctx: _Contexte, face: str, sens: str, rb: Dict, d_t: int, e_t: float, dessous_y: float) -> Dict:
    """Le renfort d'une bande, pour une face (§ 5.3) : A_s nécessaire par mètre = le plus grand de l'acier du moteur
    pour la bande (M de la bande, largeur b ; section h + plot au droit d'un plot coulé, en face inférieure), ramené au
    mètre, et de l'acier par mètre pour le plus grand moment de calcul du prolongement jusqu'au bord (§ 4.5)."""
    el = ctx.el
    b = rb["largeur"]
    b_cm = b * 100.0
    W = rb["zone"][1] - rb["zone"][0]
    opp = _opposee(ctx.c_cm, sens)
    M_plus, M_moins = max(rb["M_plus"], 0.0), max(-rb["M_moins"], 0.0)
    plot_s = None
    if ctx.coule and face == "inf":
        dans = np.zeros(len(rb["s"]), bool)
        for s1, s2 in rb["charge"]:
            dans |= (rb["s"] >= s1 - EPS) & (rb["s"] <= s2 + EPS)
        plot_s = dans
    prol = (max(rb["prol_inf"], 0.0), max(-rb["prol_sup"], 0.0))
    As_t = as_pm(d_t, e_t)

    def demande(d_couche: int) -> Dict:
        lit = _couche(ctx.c_cm, sens, d_couche, dessous_y)
        li, ls = _lits(face, lit, opp)
        out = {"d_couche": d_couche, "lit": lit}
        if plot_s is not None:
            M_d = float(np.max(np.maximum(rb["Mmax"][~plot_s], 0.0))) if (~plot_s).any() else 0.0
            M_p = float(np.max(np.maximum(rb["Mmax"][plot_s], 0.0))) if plot_s.any() else 0.0
            r = poutre(el, b_cm, ctx.h_cm, [section(M_d, M_moins, 0.0, li, ls)])[0]
            rp = poutre(el, b_cm, (ctx.h + ctx.hp) * 100.0, [section(M_p, M_moins, 0.0, li, ls)])[0]
            out.update(M=M_d, M_plot=M_p, moteur=r, moteur_plot=rp,
                       besoin_pm=max(r["grandeurs"]["besoin_inf"], rp["grandeurs"]["besoin_inf"]) * 100.0 / b_cm)
        else:
            r = poutre(el, b_cm, ctx.h_cm, [section(M_plus, M_moins, 0.0, li, ls)])[0]
            out.update(M=M_plus if face == "inf" else M_moins, M_plot=None, moteur=r, moteur_plot=None,
                       besoin_pm=r["grandeurs"][f"besoin_{face}"] * 100.0 / b_cm)
        rpr = poutre(el, 100.0, ctx.h_cm, [section(prol[0], prol[1], 0.0, li, ls)])[0]
        out["prol_pm"] = rpr["grandeurs"][f"besoin_{face}"]
        out["necessaire_pm"] = max(out["besoin_pm"], out["prol_pm"])
        return out

    base = demande(d_t)
    res = {"nom": rb["nom"], "face": face, "sens": sens, "b": b, "W": W, "As_t_pm": As_t, "M_prol": prol[0 if face == "inf" else 1],
           "base": base, "renfort": None, "impossible": False}
    if base["necessaire_pm"] <= As_t + EPS:
        res["retenu"] = base
    else:
        essais = {d: demande(max(d_t, d)) for d in D_RENFORTS}
        choix = choisir_renfort({d: essais[d]["necessaire_pm"] - As_t for d in D_RENFORTS}, d_t, e_t)
        if choix is None:
            res["impossible"] = True
            res["retenu"] = essais[D_RENFORTS[-1]]
        else:
            d_r, s_r = choix
            n = nombre_barres(W, s_r)
            L = rb["L"] - 2 * ctx.c
            res["renfort"] = {"d": d_r, "s": s_r, "n": n, "L": L, "As_pm": n * GP.aire_barre(d_r) / W}
            res["retenu"] = essais[d_r]
    place = As_t + (res["renfort"]["As_pm"] if res["renfort"] else 0.0)
    res["As_place_pm"] = place
    res["taux"] = res["retenu"]["necessaire_pm"] / place
    res["ok"] = not res["impossible"] and res["retenu"]["necessaire_pm"] <= place + EPS
    return res


def _groupes(I: np.ndarray, J: np.ndarray) -> List[np.ndarray]:
    """Les éléments qui se touchent (par un côté), regroupés : des listes d'indices dans (I, J)."""
    cle = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(I, J))}
    vu = np.zeros(len(I), bool)
    out = []
    for k0 in range(len(I)):
        if vu[k0]:
            continue
        pile, g = [k0], []
        vu[k0] = True
        while pile:
            k = pile.pop()
            g.append(k)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = cle.get((int(I[k]) + di, int(J[k]) + dj))
                if v is not None and not vu[v]:
                    vu[v] = True
                    pile.append(v)
        out.append(np.array(sorted(g)))
    return out


def _locaux(ctx: _Contexte, face: str, sens: str, d_t: int, e_t: float, dessous_y: float, M_inf, M_sup,
            besoins: Dict[int, np.ndarray]) -> List[Dict]:
    """Les renforts locaux d'une face et d'une direction (§ 5.4) : les éléments où le treillis ne suffit pas, hors des
    zones de renfort, regroupés ; chaque groupe donne un rectangle parallèle aux axes, allongé d'un recouvrement de
    chaque côté dans le sens des barres, sans sortir de la dalle ; les rectangles qui se chevauchent sont fusionnés."""
    ch = ctx.ch
    As_t = as_pm(d_t, e_t)
    rouge = np.flatnonzero(ctx.controle & (besoins[d_t] > As_t + EPS))
    if not len(rouge):
        return []
    a, b_ = ch["a"], ch["b"]
    x0b, y0b, x1b, y1b = ctx.bbox
    c = ctx.c
    paquets = [rouge[g] for g in _groupes(ch["I"][rouge], ch["J"][rouge])]
    while True:
        rects = []
        for p in paquets:
            manque = {d: float((besoins[max(d_t, d)][p] - As_t).max()) for d in D_RENFORTS}
            choix = choisir_renfort(manque, d_t, e_t)
            d_r = choix[0] if choix else D_RENFORTS[-1]
            lr = recouvrement(ctx.el, d_r)
            X0, X1 = ch["x0"] + ch["I"][p].min() * a, ch["x0"] + (ch["I"][p].max() + 1) * a
            Y0, Y1 = ch["y0"] + ch["J"][p].min() * b_, ch["y0"] + (ch["J"][p].max() + 1) * b_
            if sens == "x":
                X0, X1 = X0 - lr, X1 + lr
            else:
                Y0, Y1 = Y0 - lr, Y1 + lr
            rects.append({"elements": p, "choix": choix, "manque": manque,
                          "rect": (max(X0, x0b + c), max(Y0, y0b + c), min(X1, x1b - c), min(Y1, y1b - c))})
        fusion = None
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                r1, r2 = rects[i]["rect"], rects[j]["rect"]
                if r1[0] < r2[2] - EPS and r2[0] < r1[2] - EPS and r1[1] < r2[3] - EPS and r2[1] < r1[3] - EPS:
                    fusion = (i, j)
                    break
            if fusion:
                break
        if fusion is None:
            break
        i, j = fusion
        paquets = [p for k, p in enumerate(paquets) if k not in fusion] + [np.union1d(paquets[i], paquets[j])]
    out = []
    for r in rects:
        X0, Y0, X1, Y1 = r["rect"]
        large = (Y1 - Y0) if sens == "x" else (X1 - X0)
        long_ = (X1 - X0) if sens == "x" else (Y1 - Y0)
        if r["choix"] is None:
            out.append({"face": face, "sens": sens, "rect": r["rect"], "impossible": True, "elements": r["elements"],
                        "d": None, "s": None, "n": 0, "L": long_, "As_pm": 0.0, "manque": r["manque"]})
            continue
        d_r, s_r = r["choix"]
        n = nombre_barres(large, s_r)
        out.append({"face": face, "sens": sens, "rect": r["rect"], "impossible": False, "elements": r["elements"], "d": d_r,
                    "s": s_r, "n": n, "L": long_, "As_pm": n * GP.aire_barre(d_r) / large, "manque": r["manque"]})
    return out


def _dans_rect(ch: Dict, rect) -> np.ndarray:
    return (ch["x"] >= rect[0] - EPS) & (ch["x"] <= rect[2] + EPS) & (ch["y"] >= rect[1] - EPS) & (ch["y"] <= rect[3] + EPS)


def face(ctx: _Contexte, cote: str, code: str) -> Dict:
    """Le ferraillage d'une face (« inf » ou « sup ») avec un treillis donné : renforts de bandes (§ 5.3) et renforts
    locaux (§ 5.4), en x puis en y (les barres en y reposent sur la plus grosse barre en x), la couverture élément par
    élément (R1) et le poids (treillis + renforts locaux + renforts de bandes, § 5.2)."""
    d_t, e_t = treillis(code)
    ch = ctx.ch
    out = {"cote": cote, "code": code, "d": d_t, "e": e_t, "As_pm": as_pm(d_t, e_t), "bandes": {}, "locaux": [],
           "couverture": {}, "ok": True}
    dessous_y = d_t
    for sens in ("x", "y"):
        M_inf, M_sup = ctx.moments(sens)
        opp = _opposee(ctx.c_cm, sens)
        noms = [n for n in ("BX1", "BX2") if n in ctx.bandes] if sens == "x" else [n for n in ("BY1", "BY2") if n in ctx.bandes]
        for nom in noms:
            out["bandes"][nom] = _bande_face(ctx, cote, sens, ctx.bandes[nom], d_t, e_t, dessous_y)
        besoins = {}
        for d in sorted({d_t} | {max(d_t, d) for d in D_RENFORTS}):
            besoins[d] = besoin_elements(ctx, cote, sens, _couche(ctx.c_cm, sens, d, dessous_y), opp, M_inf, M_sup)
        loc = _locaux(ctx, cote, sens, d_t, e_t, dessous_y, M_inf, M_sup, besoins)
        out["locaux"] += loc
        # la couverture, élément par élément : la couche de chaque élément (la plus grosse barre qui le couvre)
        d_el = np.full(len(ch["x"]), d_t, int)
        As_l = np.zeros(len(ch["x"]))
        for r in loc:
            if r["impossible"]:
                continue
            dans = _dans_rect(ch, r["rect"])
            d_el = np.where(dans, np.maximum(d_el, r["d"]), d_el)
            As_l = As_l + np.where(dans, r["As_pm"], 0.0)
        besoin = np.select([d_el == d for d in besoins], [besoins[d] for d in besoins])
        place = out["As_pm"] + As_l
        out["couverture"][sens] = {"besoin": besoin, "besoin_treillis": besoins[d_t], "As_place": place, "d_couche": d_el,
                                   "dessous": dessous_y if sens == "y" else 0.0, "controle": ctx.controle.copy(),
                                   "taux_treillis": np.where(ctx.controle, besoins[d_t] / out["As_pm"], 0.0)}
        if sens == "x":
            gros = [d_t] + [r["d"] for r in loc if r["d"]] + [b["renfort"]["d"] for b in out["bandes"].values() if b["renfort"]]
            dessous_y = max(gros)
        out["ok"] = out["ok"] and bool(np.all(besoin[ctx.controle] <= place[ctx.controle] + 1e-6))
    out["dessous_y"] = dessous_y
    out["ok"] = out["ok"] and all(b["ok"] for b in out["bandes"].values()) and not any(r["impossible"] for r in out["locaux"])
    out["poids_T"] = poids_treillis(ctx, d_t, e_t)
    out["poids_R"] = sum(b["renfort"]["n"] * _longueur_totale(ctx.el, b["renfort"]["L"], b["renfort"]["d"]) * masse(b["renfort"]["d"])
                         for b in out["bandes"].values() if b["renfort"])
    out["poids_L"] = sum(sum(_longueur_totale(ctx.el, u1 - u0, r["d"]) for _v, u0, u1 in
                             barres_filantes(ctx.m["contour"], r["sens"], ctx.c, r["s"] / 100.0, r["rect"])) * masse(r["d"])
                         for r in out["locaux"] if not r["impossible"])
    out["poids"] = out["poids_T"] + out["poids_R"] + out["poids_L"]
    return out


def recontroler_face(ctx: _Contexte, f: Dict) -> None:
    """Les éléments les plus sollicités d'une face RETENUE, recontrôlés par un appel direct du moteur poutre (un écart lève
    une erreur) : la linéarité qu'emploie ``besoin_elements`` est vérifiée à chaque calcul."""
    for sens in ("x", "y"):
        cv = f["couverture"][sens]
        ctl = np.flatnonzero(cv["controle"])
        if not len(ctl):
            continue
        M_inf, M_sup = ctx.moments(sens)
        pire = ctl[np.argsort(-(cv["besoin"][ctl] / cv["As_place"][ctl]))[:NB_CONTROLES]]
        lits = {i: _couche(ctx.c_cm, sens, int(cv["d_couche"][i]), cv["dessous"]) for i in pire}
        _recontroler(ctx, f["cote"], cv["besoin"], lits, _opposee(ctx.c_cm, sens), M_inf, M_sup, pire)


def _longueur_totale(el: Dict, L: float, d: float) -> float:
    n, l1 = troncons(el, L, d)
    return n * l1


def poids_treillis(ctx: _Contexte, d: int, e_cm: float) -> float:
    """Le poids (kg) d'un treillis d'une face : ses barres filantes en x et en y (§ 5.2 : pas de panneaux)."""
    tot = 0.0
    for sens in ("x", "y"):
        for _v, u0, u1 in barres_filantes(ctx.m["contour"], sens, ctx.c, e_cm / 100.0):
            tot += _longueur_totale(ctx.el, u1 - u0, d)
    return tot * masse(d)


# ------------------------------------------------------------------ effort tranchant, U, cages
def tranchant(ctx: _Contexte, rb: Dict, lit_inf: Dict, lit_sup: Dict, d_t_inf: int, e_t_inf: float) -> Dict:
    """Le contrôle du moteur poutre sur une bande avec V à d (§ 5.5) — d étant celui du moteur pour la section de la
    bande — et, s'il demande des armatures, des C verticaux là où |V| dépasse ce que la dalle reprend sans armatures."""
    el = ctx.el
    b_cm = rb["largeur"] * 100.0
    r0 = poutre(el, b_cm, ctx.h_cm, [section(0.0, 0.0, 0.0, lit_inf, lit_sup)])[0]
    d_v = r0["grandeurs"]["d_cisaillement"] / 100.0
    V = EF.v_a_d(rb, d_v)
    r = poutre(el, b_cm, ctx.h_cm, [section(0.0, 0.0, V, lit_inf, lit_sup)])[0]
    g = r["grandeurs"]
    etat = next((v["etat"] for v in r["verifications"] if v["cle"] == "tau"), "ok")
    out = {"nom": rb["nom"], "V": V, "d_v": d_v, "tau": g["tau"], "tau_lim": g["tau_lim"], "bande": g["bande_tau"],
           "besoin_tau": g["besoin_tau"], "taux": (g["tau"] / g["tau_lim"]) if g["tau"] else 0.0, "etat": etat, "C": None, "moteur": r}
    if V <= 0 or g["besoin_tau"] in (None, "pas_d_etriers", "inacceptable"):
        return out
    branches = int(math.floor(b_cm / e_t_inf + 1e-9)) + 1
    ok_s = EF.lecture_v(rb["s"], rb["charge"], d_v)
    secs = [section(0.0, 0.0, float(v), lit_inf, lit_sup) for v in rb["Vabs"]]
    besoins = [x["grandeurs"]["besoin_tau"] for x in poutre(el, b_cm, ctx.h_cm, secs)]
    arm = np.array([bt not in (None, "pas_d_etriers") for bt in besoins]) & ok_s
    for d in D_TRANCHANT:
        rr = poutre(el, b_cm, ctx.h_cm, [section(0.0, 0.0, V, lit_inf, lit_sup, [{"type": "epingle", "d": d}] * branches)])[0]
        s_lim = rr["grandeurs"]["s_lim_tronque"]
        if s_lim and s_lim >= 5.0 - EPS:
            break
    zones = _intervalles(rb["s"], arm)
    rangs = sum(int(math.floor((z1 - z0) * 100.0 / s_lim + 1e-9)) + 1 for z0, z1 in zones)
    out["C"] = {"d": d, "s": s_lim, "branches": branches, "rangs": rangs, "n": rangs * branches,
                "longueur": sum(z1 - z0 for z0, z1 in zones), "zones": zones, "a": RETOUR_C * d / 1000.0, "ame": ctx.h - 2 * ctx.c}
    return out


def _intervalles(s: np.ndarray, masque: np.ndarray) -> List[Tuple[float, float]]:
    out, debut = [], None
    for x, m in zip(s, masque):
        if m and debut is None:
            debut = float(x)
        if not m and debut is not None:
            out.append((debut, float(x)))
            debut = None
    if debut is not None:
        out.append((debut, float(s[-1])))
    return out


def u_fermeture(ctx: _Contexte, fi: Dict, fs: Dict) -> Dict:
    """Les U de fermeture sur tout le pourtour (§ 5.6) : Ø et espacement du treillis le plus fort des deux faces ; par
    bord, partie entière de (longueur − 2c) ÷ e, + 1 ; âme verticale h − 2c, branches = un recouvrement."""
    fort = max((fi, fs), key=lambda f: (f["As_pm"], f["d"]))
    d, e = fort["d"], fort["e"] / 100.0
    lr = recouvrement(ctx.el, d)
    bords = []
    cont = ctx.m["contour"]
    for i in range(len(cont)):
        p, q = cont[i], cont[(i + 1) % len(cont)]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        bords.append({"de": tuple(p), "a": tuple(q), "L": L, "n": int(math.floor((L - 2 * ctx.c) / e + 1e-9)) + 1})
    ame = ctx.h - 2 * ctx.c
    return {"d": d, "e": fort["e"], "code": fort["code"], "bords": bords, "n": sum(b["n"] for b in bords), "ame": ame,
            "branche": lr, "L": ame + 2 * lr}


def cages(ctx: _Contexte, fi: Dict) -> List[Dict]:
    """La cage de chaque plot coulé en place (§ 5.8) : des C verticaux sur les 4 faces (Ø et espacement du treillis
    inférieur ; branche basse dans la dalle = un recouvrement, montant h + hauteur du plot − 2c, branche haute = côté du
    plot − 2c), 2 U horizontaux en tête (âme et branches = côté − 2c). Les brins verticaux sous le pied : le logiciel n'a
    pas de vérification du poinçonnement → « à dimensionner »."""
    pl = ctx.m.get("plots")
    if not ctx.coule:
        return []
    d, e = fi["d"], fi["e"] / 100.0
    c, lr = ctx.c, recouvrement(ctx.el, fi["d"])
    a, b = pl["a"], pl["b"]
    montant = ctx.h + ctx.hp - 2 * c
    out = []
    for i, q in enumerate(pl["polys"]):
        n_a = int(math.floor((a - 2 * c) / e + 1e-9)) + 1         # le long d'un côté a (la branche haute traverse b)
        n_b = int(math.floor((b - 2 * c) / e + 1e-9)) + 1
        out.append({"plot": i, "poly": q, "d": d, "e": fi["e"],
                    "C": [{"cote": "a", "n": 2 * n_a, "basse": lr, "montant": montant, "haute": b - 2 * c},
                          {"cote": "b", "n": 2 * n_b, "basse": lr, "montant": montant, "haute": a - 2 * c}],
                    "U": [{"n": 1, "ame": a - 2 * c, "branche": b - 2 * c}, {"n": 1, "ame": b - 2 * c, "branche": a - 2 * c}],
                    "brins": "à dimensionner (poinçonnement non vérifié)"})
    return out


# ------------------------------------------------------------------ le bordereau (§ 6.3)
def _ligne(rep: str, famille: str, d: float, forme: str, cotes: Tuple, n: int, L: float, libelle: str, **plus) -> Dict:
    """Une ligne du bordereau : n barres de Ø d, de longueur unitaire L (m) = somme des cotes, et leur poids."""
    return {"rep": rep, "famille": famille, "d": int(d), "forme": forme, "cotes": tuple(cotes), "n": int(n), "L": L,
            "Lt": n * L, "poids": n * L * masse(d), "libelle": libelle, **plus}


def _droites(el: Dict, rep: str, famille: str, d: float, barres: List[Tuple[float, float, float]], libelle: str, **plus) -> List[Dict]:
    """Des barres droites regroupées par longueur (au 5 cm supérieur) ; au-delà de la longueur maxi, en morceaux."""
    par = {}
    for _v, u0, u1 in barres:
        n, l1 = troncons(el, u1 - u0, d)
        L = longueur_bordereau(l1)
        par[L] = par.get(L, 0) + n
    return [_ligne(rep, famille, d, "droite", (L,), n, L, libelle, **plus) for L, n in sorted(par.items(), reverse=True)]


def bordereau(el: Dict, modele: Dict, fer: Dict) -> Dict:
    """Le bordereau des aciers (§ 6.3) — T1 à T4 : treillis inférieur x, inférieur y, supérieur x, supérieur y ; R :
    renforts de bandes ; L : renforts locaux ; U : U de fermeture ; C : cages des plots ; V : C d'effort tranchant —, les
    totaux par Ø, le total, le béton (dalle + plots coulés ; les préfabriqués ne comptent pas) et le ratio acier / béton."""
    c = el["armatures"]["enrobage"]
    cont = modele["contour"]
    lignes = []
    for k, (cote, sens) in enumerate((("inf", "x"), ("inf", "y"), ("sup", "x"), ("sup", "y")), 1):
        f = fer["faces"][cote]
        lignes += _droites(el, f"T{k}", "T", f["d"], barres_filantes(cont, sens, c, f["e"] / 100.0),
                           f"treillis {cote}. {sens}", cote=cote, sens=sens, e=f["e"])
    i = 0
    for cote in ("inf", "sup"):
        for nom, b in fer["faces"][cote]["bandes"].items():
            r = b["renfort"]
            if r:
                i += 1
                n_t, l1 = troncons(el, r["L"], r["d"])
                L = longueur_bordereau(l1)
                lignes.append(_ligne(f"R{i}", "R", r["d"], "droite", (L,), r["n"] * n_t, L, f"renfort {nom} {cote}.",
                                     cote=cote, sens=b["sens"], bande=nom, s=r["s"]))
    i = 0
    for cote in ("inf", "sup"):
        for r in fer["faces"][cote]["locaux"]:
            if r["impossible"]:
                continue
            i += 1
            lignes += _droites(el, f"L{i}", "L", r["d"], barres_filantes(cont, r["sens"], c, r["s"] / 100.0, r["rect"]),
                               f"renfort local {cote}. {r['sens']}", cote=cote, sens=r["sens"], rect=r["rect"], s=r["s"])
    u = fer["U"]
    lignes.append(_ligne("U1", "U", u["d"], "U", (u["branche"], u["ame"], u["branche"]), u["n"], u["L"], "U de fermeture", e=u["e"]))
    if fer["cages"]:
        cg = fer["cages"]
        d = cg[0]["d"]
        formes = {}
        for x in cg:
            for cc in x["C"]:
                cle = (round(cc["basse"], 4), round(cc["montant"], 4), round(cc["haute"], 4))
                formes[cle] = formes.get(cle, 0) + cc["n"]
        for j, (cle, n) in enumerate(sorted(formes.items(), reverse=True), 1):
            lignes.append(_ligne(f"C{j}", "C", d, "C", cle, n, sum(cle), "C verticaux des plots"))
        us = {}
        for x in cg:
            for uu in x["U"]:
                cle = (round(uu["branche"], 4), round(uu["ame"], 4), round(uu["branche"], 4))
                us[cle] = us.get(cle, 0) + uu["n"]
        for cle, n in sorted(us.items(), reverse=True):
            j = len([ln for ln in lignes if ln["famille"] == "C"]) + 1
            lignes.append(_ligne(f"C{j}", "C", d, "U", cle, n, sum(cle), "U horizontaux en tête des plots"))
    i = 0
    for nom, t in fer["tranchant"].items():
        if t["C"]:
            i += 1
            cc = t["C"]
            lignes.append(_ligne(f"V{i}", "V", cc["d"], "C", (cc["a"], cc["ame"], cc["a"]), cc["n"], cc["ame"] + 2 * cc["a"],
                                 f"C d'effort tranchant {nom}", bande=nom, s=cc["s"]))
    par_d = {}
    for ln in lignes:
        par_d[ln["d"]] = par_d.get(ln["d"], 0.0) + ln["poids"]
    total = sum(ln["poids"] for ln in lignes)
    pl = modele.get("plots")
    beton = G.aire(cont) * modele["h"] + (sum(G.aire(q) for q in pl["polys"]) * pl["hauteur"] if pl and pl["type"] == "coule" else 0.0)
    return {"lignes": lignes, "par_d": dict(sorted(par_d.items())), "total": total, "beton": beton,
            "ratio": total / beton if beton > 0 else 0.0}


# ------------------------------------------------------------------ la fondation
def calculer(el: Dict, modele: Dict, ef: Dict) -> Dict:
    """Le ferraillage de la fondation, à partir de l'enveloppe des efforts ``ef`` (``efforts.enveloppe`` : tous les cas,
    les k cochés « SCIA »). Pour chaque face, le treillis imposé, ou — « automatique » — celui des quatre qui donne le
    poids total le plus faible (treillis + renforts locaux + renforts de bandes, § 5.2 ; à égalité, le premier de la
    liste). Rend {"faces": {"inf", "sup"} (chacune : le treillis, ses bandes, ses renforts locaux, sa couverture, les
    quatre candidats et leur poids), "tranchant" {bande: …}, "hauteur" {…}, "U", "cages", "arrachement", "lignes"
    (ce que le module ne vérifie pas : [(titre, texte, niveau)]), "non_verifies", "ok"}."""
    ctx = _Contexte(el, modele, ef)
    faces = {}
    for cote in ("inf", "sup"):
        impose = el["armatures"][f"treillis_{cote}"]
        if impose != "auto":
            f = face(ctx, cote, impose)
            f["auto"], f["candidats"] = False, []
        else:
            cands = [face(ctx, cote, code) for code in TREILLIS_CHOIX]
            f = min(cands, key=lambda x: (round(x["poids"], 6), TREILLIS_CHOIX.index(x["code"])))
            f["auto"] = True
            f["candidats"] = [{"code": x["code"], "poids": x["poids"], "poids_T": x["poids_T"], "poids_R": x["poids_R"],
                               "poids_L": x["poids_L"], "ok": x["ok"]} for x in cands]
        recontroler_face(ctx, f)
        faces[cote] = f
    fi, fs = faces["inf"], faces["sup"]
    # l'effort tranchant et la hauteur des bandes : la section avec les lits RETENUS des deux faces
    tr, haut = {}, {}
    for nom, rb in ctx.bandes.items():
        sens = "x" if nom.startswith("BX") else "y"
        li = fi["bandes"][nom]["retenu"]["lit"]
        ls = fs["bandes"][nom]["retenu"]["lit"]
        tr[nom] = tranchant(ctx, rb, li, ls, fi["d"], fi["e"])
        r = poutre(el, rb["largeur"] * 100.0, ctx.h_cm, [section(max(rb["M_plus"], 0.0), max(-rb["M_moins"], 0.0), 0.0, li, ls)])[0]
        g = r["grandeurs"]
        haut[nom] = {"h_min": g["h_min"], "h": ctx.h_cm, "taux": g["h_min"] / ctx.h_cm, "ok": g["h_min"] <= ctx.h_cm + EPS, "sens": sens}
    # la hauteur dans la zone courante : les plus grands moments de calcul par mètre (b = 1 m)
    c_ = ef["courante"]
    r = poutre(el, 100.0, ctx.h_cm, [section(max(c_["inf_x"], c_["inf_y"], 0.0), max(-c_["sup_x"], -c_["sup_y"], 0.0), 0.0,
                                             _couche(ctx.c_cm, "y", fi["d"], fi["dessous_y"]), _couche(ctx.c_cm, "y", fs["d"], fs["dessous_y"]))])[0]
    hc = r["grandeurs"]["h_min"]
    haut["courante"] = {"h_min": hc, "h": ctx.h_cm, "taux": hc / ctx.h_cm, "ok": hc <= ctx.h_cm + EPS}
    arr = any(v is not None and v < 0 for j in el["reactions"] for v in j["P"])
    lignes = [("Poinçonnement", "la méthode du bureau n'en a pas dans le logiciel", "inc")]
    if ctx.coule:
        lignes.append(("Brins verticaux sous le pied", "à dimensionner (poinçonnement non vérifié)", "inc"))
    if arr:
        lignes.append(("Arrachement", "une réaction négative", "nok"))
    nv = ["le poinçonnement"] + (["les brins verticaux sous le pied"] if ctx.coule else []) + (["l'arrachement"] if arr else [])
    ok = (fi["ok"] and fs["ok"] and all(t["etat"] in ("ok", "warn") for t in tr.values()) and all(h["ok"] for h in haut.values())
          and not arr)
    out = {"faces": faces, "tranchant": tr, "hauteur": haut, "U": u_fermeture(ctx, fi, fs), "cages": cages(ctx, fi),
           "arrachement": arr, "lignes": lignes, "non_verifies": nv, "ok": ok, "enrobage": ctx.c, "h": ctx.h, "hp": ctx.hp,
           "coule": ctx.coule}
    out["bordereau"] = bordereau(el, modele, out)
    return out

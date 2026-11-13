# -*- coding: utf-8 -*-
"""Les NIVEAUX réels des deux directions d'une dalle (module « Dalles béton armé », api_web/dalle.py) : la position
RÉELLE automatique de chaque couche, telle qu'elle est posée, la copie de la dalle que le moteur lit, les positions et
les axes de calcul que l'écran, le dessin et la note lisent (docs/site/DALLES_FUSION.md § 4 ; la règle, beton/position.py).

DEUX POSITIONS (beton/position.py — la règle du 13/11/2026, commune à la poutre, à la dalle et à la prédalle) : la
position RÉELLE, ce que le DESSIN montre (c + Ø/2 au contact de l'enrobage ; une couche posée sur une autre, le dessus
de celle-ci + Ø/2 ; les Ø réels, aucun arrondi) ; l'axe de CALCUL, ⌈réel + jeu de pose⌉ au pas d'arrondi, que le MOTEUR
fait de la position qu'il reçoit (``dist``). Ce fichier ne calcule que des positions réelles ; le moteur ne sait pas
empiler deux couches : le site écrit, dans la copie qu'il lui envoie, la position réelle de chaque couche empilée.

LES NAPPES SONT TOUJOURS EMPILÉES (décision de l'ingénieur du 09/11/2026 ; la clé d'état ``empilement`` d'avant ne pilote
plus rien) :
  · direction PRINCIPALE, base et renforts : au contact de l'enrobage, c + Ø/2 (la position du moteur) ;
  · direction SECONDAIRE, base ET renforts (S3) : posée SUR la nappe principale, sur la couche la plus haute de son LIT
    DE CONTACT (S5, ci-dessous) : max(e_k + Ø_k/2) + Ø/2 ;
  · un TREILLIS UNIQUE (DAL5b : ses deux directions de fils soudées l'une sur l'autre) : ses fils secondaires sur ses
    fils principaux, toujours ; un renfort de la direction PRINCIPALE posé SUR le treillis (``appui_treillis`` : ses
    fils secondaires, ou un renfort secondaire plus haut qu'eux) ;
  · une PRÉDALLE, faces inférieures : les deux BASES dans la peau — la principale au contact de l'enrobage, la
    secondaire posée sur elle (demande du propriétaire du 13/11/2026 ; l'écran Streamlit la posait sur la prédalle) ;
    les renforts inférieurs sur la prédalle, h_p + Ø/2 (posés au chantier : ``beton/predalle/moteur.pose``).
  Un axe SAISI (``dist``) est une position réelle : il n'est jamais déplacé ; le moteur lui ajoute le jeu de pose.

LE LIT DE CONTACT (S5, ``lit_contact``) — les couches principales que la nappe secondaire croise et sur lesquelles elle
repose : le DESSOUS d'une couche est e − Ø/2 (vers le parement), son DESSUS e + Ø/2. Le lit part de la BASE principale ;
toute couche principale AUTOMATIQUE en est (posée sur les cales, comme la base) ; une couche SAISIE en est si son dessous
ne dépasse pas le dessus du lit (à ``TOL_LIT`` près : les arrondis des flottants), le dessus du lit montant alors au
sien. Une couche saisie plus haut est un AUTRE lit (posée sur la nappe secondaire ou sur des cales) : la nappe
secondaire ne s'y appuie pas. La plus haute du lit porte, la base à égalité. Un treillis unique, ou la peau d'une
prédalle : la base principale seule porte.

EXEMPLE (dalle coulée, c = 3 cm, jeu de pose 1 cm, arrondi 0,5 cm, principale Y, base Y Ø12, base X treillis Ø10) :
  inf. Y : réel 3 + 0,6 = 3,6 ; calcul 3,6 + 1 = 4,6 → 5,0 cm ;
  inf. X, posée sur la base Y : réel 3 + 1,2 + 0,5 = 4,7 ; calcul 4,7 + 1 = 5,7 → 6,0 cm ;
  S5 — un renfort 3 Ø25 en inf. Y (réel 3 + 1,25 = 4,25, dessus 5,5) : la base X repose sur lui, réel 5,5 + 0,5 = 6,0,
  calcul 7,0 cm ; DAL5b — un treillis unique 10/10/100/100 : fils Y 3,5, fils X 3,5 + 0,5 + 0,5 = 4,5, un renfort Ø12
  en inf. Y posé sur les fils X : 5,0 + 0,6 = 5,6 (calcul 6,6 → 7,0 cm) ;
  PRÉDALLE (h_p = 6 cm), même dalle : inf. Y 3,6 et inf. X 4,7, dans la peau ; un renfort Ø12 inférieur, sur la
  prédalle : 6 + 0,6 = 6,6 (calcul 7,6 → 8,0 cm).

API
---
    position_auto(d, s, f, j) -> float      la position RÉELLE automatique (cm, depuis le parement) de la couche j de f
    termes(d, s, f, j) -> [(symbole, cm)]   la position réelle effective en termes (« c », « Ø_Y », « Ø/2 », « h_p »,
                                           « posé ») — le détail du calcul de l'axe, ``P.detail(termes, jeu, arrondi)``
    treillis(d, s, cote) -> bool           la nappe est un treillis unique (ses fils soudés : DAL5b)
    lit_contact(d, s, cote) -> [rangs]     les couches principales de la face (inf | sup) que la nappe secondaire croise
    appui(d, s, cote) -> dict              celle sur laquelle elle repose : {couche, e, d, demi, dessus, lit}
    appui_treillis(d, s, cote) -> dict     DAL5b : la couche SECONDAIRE sur laquelle reposent les renforts principaux d'un
                                           treillis unique (ses fils, ou un renfort secondaire qui les dépasse), + face
    support(d, s, f, j) -> dict            ce qui porte une couche empilée : appui, ou appui_treillis (« sec » : vrai)
    nom_appui(d, s, cote, court) / nom_support(d, s, f, j, court) -> str    les mêmes, en mots
    empilee(d, s, f, j) -> bool            la couche est-elle posée par le site sur une autre (position ≠ celle du moteur)
    positions(d, s) -> {face: [cm]}        la position RÉELLE effective de chaque couche (saisie, sinon automatique) —
                                           le dessin, les croisements
    axes(d, s) -> {face: [cm]}             l'axe de CALCUL de chaque couche (⌈position + jeu⌉) — ce que le moteur utilise
    pour_moteur(d) -> dict                 la dalle telle que le moteur la lit : sans les clés du site, ``dist`` = la
                                           position réelle d'une couche empilée ; ``d`` n'est pas modifié
    yG_calcule(d, s, f) / h_utile(d, s, f) le centre de gravité calculé et la hauteur utile, aux axes de calcul
    annoter(r, d) -> r                     le résultat du moteur relu par le site (positions, détail, empilement, formules
                                           de l'axe) — ce que l'écran et les dessins lisent
    candidats_base(d, i, f, familles)      le sélecteur de la nappe de base, chaque candidat posé à SA position empilée
    deplacer(d, i, f, j, sens) -> float    DAL5c : la position de la couche montée (sens 1) ou descendue (− 1) de PAS_AXE,
                                           sur la grille ; ValueError (le message) si elle traverserait une couche ou
                                           l'enrobage
    conflits(d, s) -> set                  ce qui se traverse dans la section (couches croisées, enrobage), aux Ø réels
    nom_couche(d, s, f, j, nappe) -> str   la couche en mots (« les fils Y du treillis », « le renfort R2 Y »)
    moteur(d), type_dalle(d), principale(d)

``d`` est une dalle du SITE (normalisée par le moteur de son type, plus ``verrou`` et ``type`` ∈ {coulee, predalle} —
absent : coulee) ; ``s`` une de ses sections ; ``f`` une face (« inf_x », « inf_y », « sup_x », « sup_y ») ; ``j`` le
rang de la couche (0 : la base). Rien de normatif n'est calculé ici : l'empilement est une addition de diamètres
(beton/position.py), l'axe de calcul et la hauteur utile sont ceux du moteur.
"""
import copy
from typing import Dict, List, Optional, Sequence

from beton import position as P
from beton.dalle import entree as EN_D
from beton.dalle import moteur as MO_D
from beton.predalle import moteur as MO_P

from . import dalle_nappes as NAP

FACES = EN_D.FACES
TYPES = ("coulee", "predalle")
TYPE_DEFAUT = "coulee"
CLES_SITE = frozenset({"verrou", "type"})     # ce que le site ajoute à la dalle du moteur (= dalle.CLES_SITE, lié par un test)
TOL_LIT = 1e-6                                # cm : le contact d'une couche avec le lit, aux arrondis des flottants près (S5)


# ------------------------------------------------------------------ lire la dalle du site
def type_dalle(d: Dict) -> str:
    """« coulee » ou « predalle » (absent ou inconnu : coulee)."""
    t = d.get("type")
    return t if t in TYPES else TYPE_DEFAUT


def moteur(d: Dict):
    """Le moteur du type : ``beton.dalle.moteur`` ou ``beton.predalle.moteur``."""
    return MO_P if type_dalle(d) == "predalle" else MO_D


def principale(d: Dict) -> str:
    """La direction principale, en minuscule (« x » ou « y »), quelle que soit la casse stockée (le moteur garde « X »/« Y »)."""
    return str(d.get("dir_principale") or EN_D.DEFAUTS_DALLE["dir_principale"]).strip().lower()


def _jeu(d: Dict) -> float:
    """Le jeu de pose de la dalle (cm) — la clé « jeu_premier_lit » (beton/dalle/entree.py)."""
    return float(d.get("jeu_premier_lit", EN_D.DEFAUTS_DALLE["jeu_premier_lit"]) or 0.0)


def _arrondi(d: Dict) -> float:
    """Le pas d'arrondi de l'axe de calcul (cm)."""
    return float(d.get("arrondi", EN_D.DEFAUTS_DALLE["arrondi"]) or 0.0)


def _enrobage(d: Dict) -> float:
    return float(d.get("enrobage", EN_D.DEFAUTS_DALLE["enrobage"]) or MO_D.ENROBAGE_SI_NUL)


def _reel_moteur(d: Dict, c: Dict, j: int, f: str) -> float:
    """La position RÉELLE automatique selon le MOTEUR du type (cm) : au contact de l'enrobage ; un renfort inférieur
    d'une prédalle, sur la prédalle."""
    if type_dalle(d) == "predalle":
        return MO_P.position_auto(c, j, f, d.get("dir_principale", EN_D.DEFAUTS_DALLE["dir_principale"]),
                                  d.get("enrobage", EN_D.DEFAUTS_DALLE["enrobage"]), d.get("h_pre"))
    return MO_D.position_auto(c, d.get("enrobage", EN_D.DEFAUTS_DALLE["enrobage"]))


def _termes_moteur(d: Dict, c: Dict, j: int, f: str) -> list:
    """La position du moteur en termes : (« c », Ø/2) ou, sur la prédalle, (« h_p », Ø/2)."""
    if type_dalle(d) == "predalle":
        return MO_P.termes_auto(c, j, f, d.get("dir_principale", EN_D.DEFAUTS_DALLE["dir_principale"]),
                                d.get("enrobage", EN_D.DEFAUTS_DALLE["enrobage"]), d.get("h_pre"))
    return [("c", _enrobage(d)), ("Ø/2", _demi(c))]


def _demi(c: Dict) -> float:
    """Le demi-diamètre RÉEL de la couche (cm) : Ø/20, Ø en mm (Øl pour un treillis)."""
    return MO_D.couche_diametre(c) / 20.0


def _peau(d: Dict, cote: str) -> bool:
    """La face inférieure d'une prédalle : ses bases dans la peau, ses renforts sur la prédalle."""
    return cote == "inf" and type_dalle(d) == "predalle"


# ------------------------------------------------------------------ la position automatique du site
def treillis(d: Dict, s: Dict, cote: str) -> bool:
    """La nappe ``cote`` (« inf » | « sup ») de la section ``s`` est-elle un treillis UNIQUE — ses deux bases, l'une la
    transposée de l'autre (la règle de la rubrique Armatures : api_web/dalle_nappes.mode, une source) ? Ses deux
    directions de fils sont alors soudées l'une sur l'autre (DAL5b, en tête du fichier)."""
    return NAP.mode(s, cote, principale(d))["mode"] in ("treillis", "treillis_renforts")


def position_auto(d: Dict, s: Dict, f: str, j: int) -> float:
    """La position RÉELLE automatique (cm, depuis le parement de la face) de la couche ``j`` de la face ``f`` de la
    section ``s`` — la règle et l'exemple chiffré sont en tête du fichier : la position du moteur, ou posée sur la
    couche qui la porte (``support``) : son dessus + Ø/2."""
    c = s["faces"][f]["couches"][j]
    cote, dk = f.split("_")
    if dk == principale(d):
        if j > 0 and treillis(d, s, cote) and not _peau(d, cote):
            return P.sur_couche(appui_treillis(d, s, cote)["dessus"], MO_D.couche_diametre(c), P.JEU_COUCHES)
        return _reel_moteur(d, c, j, f)
    if _peau(d, cote) and j > 0:
        return _reel_moteur(d, c, j, f)                 # un renfort secondaire inférieur : sur la prédalle
    return P.sur_couche(appui(d, s, cote)["dessus"], MO_D.couche_diametre(c), P.JEU_COUCHES)


def _couches_principales(d: Dict, s: Dict, cote: str) -> List[tuple]:
    """(rang, position EFFECTIVE, Ø/2) de chaque couche de la direction principale de la face ``cote`` (inf | sup), base
    d'abord : la position saisie si elle l'est, sinon celle du moteur — là où la nappe secondaire la croise."""
    f = f"{cote}_{principale(d)}"
    return [(j, float(c["dist"]) if c.get("dist") is not None else _reel_moteur(d, c, j, f), _demi(c))
            for j, c in enumerate(s["faces"][f]["couches"])]


def lit_contact(d: Dict, s: Dict, cote: str) -> List[int]:
    """Les rangs des couches principales de la face ``cote`` dans le LIT DE CONTACT (en tête du fichier). Triés."""
    pts = _couches_principales(d, s, cote)
    auto = [c.get("dist") is None for c in s["faces"][f"{cote}_{principale(d)}"]["couches"]]
    return _lit(pts, auto)


def _lit(pts: List[tuple], auto: List[bool]) -> List[int]:
    """Le LIT DE CONTACT d'une face (en tête du fichier), ``pts`` ses couches (rang, position effective, Ø/2), base
    d'abord, ``auto`` lesquelles sont automatiques : la base, toute couche automatique, puis toute couche dont le dessous
    ne dépasse pas le dessus du lit, le dessus montant à chaque couche admise. Triés."""
    dans, reste = [pts[0]], pts[1:]
    dessus = pts[0][1] + pts[0][2]
    entre = True
    while entre:
        entre = False
        for p in list(reste):
            if auto[p[0]] or p[1] - p[2] <= dessus + TOL_LIT:
                dans.append(p)
                reste.remove(p)
                dessus = max(dessus, p[1] + p[2])
                entre = True
    return sorted(p[0] for p in dans)


def _plus_haute(pts: Dict[int, tuple], lit: List[int], c_de) -> Dict:
    """La couche du lit au plus grand dessus e + Ø/2, la base (ou le premier rang) à égalité, à ``TOL_LIT`` près."""
    haut = max(pts[j][1] + pts[j][2] for j in lit)
    k = next(j for j in lit if pts[j][1] + pts[j][2] >= haut - TOL_LIT)
    return {"couche": k, "e": pts[k][1], "d": MO_D.couche_diametre(c_de(k)), "demi": pts[k][2],
            "dessus": pts[k][1] + pts[k][2], "lit": lit}


def appui(d: Dict, s: Dict, cote: str) -> Dict:
    """La couche principale sur laquelle repose la nappe secondaire de la face ``cote`` (S5) : la plus HAUTE du lit de
    contact (le plus grand dessus e + Ø/2), la base à égalité — ``{"couche": rang, "e": position effective, "d": Ø (mm,
    Øl d'un treillis), "demi": Ø/2, "dessus": e + Ø/2, "lit": rangs du lit}``. Un TREILLIS UNIQUE (DAL5b) ou la peau
    d'une prédalle : la base principale, toujours (les fils secondaires y sont soudés ; dans la peau, la base secondaire
    est posée sur la principale, les renforts sont sur la prédalle)."""
    pts = {p[0]: p for p in _couches_principales(d, s, cote)}
    lit = [0] if treillis(d, s, cote) or _peau(d, cote) else lit_contact(d, s, cote)
    return _plus_haute(pts, lit, lambda k: s["faces"][f"{cote}_{principale(d)}"]["couches"][k])


def _secondaire(d: Dict) -> str:
    return "x" if principale(d) == "y" else "y"


def appui_treillis(d: Dict, s: Dict, cote: str) -> Dict:
    """DAL5b — la couche SECONDAIRE d'un treillis unique sur laquelle reposent ses renforts principaux : la plus haute
    du lit de contact de la face secondaire (``_lit`` : ses fils, toute couche automatique, une couche saisie qui les
    touche) — ses fils, ou un renfort secondaire plus épais qui les dépasse. Même forme qu'``appui``, plus « face »
    (la face secondaire). Les positions sont EFFECTIVES : saisies, sinon automatiques (posées sur les fils principaux
    seuls : aucune boucle)."""
    f = f"{cote}_{_secondaire(d)}"
    couches = s["faces"][f]["couches"]
    liste = [(k, float(c["dist"]) if c.get("dist") is not None else position_auto(d, s, f, k), _demi(c))
             for k, c in enumerate(couches)]
    lit = _lit(liste, [c.get("dist") is None for c in couches])
    return dict(_plus_haute({p[0]: p for p in liste}, lit, lambda k: couches[k]), face=f)


def support(d: Dict, s: Dict, f: str, j: int) -> Dict:
    """Ce qui porte la couche ``j`` de la face ``f`` quand le site l'empile : une couche secondaire repose sur la nappe
    principale (``appui``) ; un renfort principal d'un treillis unique repose sur le treillis (``appui_treillis``,
    DAL5b) — « sec » le dit ; « face » est la face de la couche qui porte."""
    cote, dk = f.split("_")
    if dk == principale(d):
        return dict(appui_treillis(d, s, cote), sec=True)
    return dict(appui(d, s, cote), sec=False, face=f"{cote}_{principale(d)}")


def nom_support(d: Dict, s: Dict, f: str, j: int, court: bool = False) -> str:
    """``support`` en mots : celui d'une couche secondaire, ``nom_appui`` (« la base Y », « le renfort Y »…) ; celui d'un
    renfort principal posé sur un treillis (DAL5b), « le treillis » (ses fils secondaires), « le renfort X » ou « le
    renfort R2 X » (un renfort secondaire plus haut) — court (légendes) : « le treillis », « renfort X »."""
    cote, dk = f.split("_")
    if dk != principale(d):
        return nom_appui(d, s, cote, court)
    k = appui_treillis(d, s, cote)["couche"]
    q = _secondaire(d).upper()
    if k == 0:
        return "le treillis"
    if court:
        return f"renfort {q}"
    n = len(s["faces"][f"{cote}_{_secondaire(d)}"]["couches"]) - 1
    return f"le renfort {q}" if n == 1 else f"le renfort R{k} {q}"


def nom_appui(d: Dict, s: Dict, cote: str, court: bool = False) -> str:
    """Ce sur quoi repose la nappe secondaire de la face ``cote``, en mots (formules, pose de la couche) : « la base Y »,
    « le renfort Y » (la face principale n'en porte qu'un) ou « le renfort R2 Y » (le repère de la couche à l'écran,
    api_web/plaque_compact.couche). ``court`` : la forme des LÉGENDES des coupes — « renfort Y » pour tout renfort (sans
    article ni repère ; « la base Y » inchangé) : une ligne de légende doit tenir dans sa colonne (capture S5 du
    09/11/2026). Le repère reste dans la forme longue (formule, pose)."""
    k = appui(d, s, cote)["couche"]
    dk = principale(d).upper()
    if k == 0:
        return f"la base {dk}"
    if court:
        return f"renfort {dk}"
    n = len(s["faces"][f"{cote}_{principale(d)}"]["couches"]) - 1
    return f"le renfort {dk}" if n == 1 else f"le renfort R{k} {dk}"


def empilee(d: Dict, s: Dict, f: str, j: int) -> bool:
    """La couche est-elle posée PAR LE SITE sur une autre : automatique (``dist`` vide) et posée ailleurs que là où le
    moteur la poserait (la nappe secondaire sur la principale, un renfort principal sur un treillis unique)."""
    c = s["faces"][f]["couches"][j]
    return c.get("dist") is None and abs(position_auto(d, s, f, j) - _reel_moteur(d, c, j, f)) > 1e-9


def termes(d: Dict, s: Dict, f: str, j: int) -> list:
    """La position RÉELLE effective de la couche en termes, ``[(symbole, cm), …]`` — ce que le détail de l'axe écrit
    (« c + Ø_Y + Ø/2 + jeu = 3 + 1,2 + 0,5 + 1 = 5,7 → 6,0 cm ») : saisie, (« posé », e) ; celle du moteur, (« c », Ø/2)
    ou (« h_p », Ø/2) ; empilée, les termes de la couche qui la porte, son demi-Ø devenu son Ø entier (son dessus, le
    symbole dit sa direction : « Ø_Y »), puis Ø/2."""
    c = s["faces"][f]["couches"][j]
    if c.get("dist") is not None:
        return [("posé", float(c["dist"]))]
    if not empilee(d, s, f, j):
        return _termes_moteur(d, c, j, f)
    a = support(d, s, f, j)
    sous = termes(d, s, a["face"], a["couche"])
    q = a["face"].split("_")[1].upper()
    dessous = sous[:-1] + [(f"Ø_{q}", a["d"] / 10.0)] if sous[-1][0] == "Ø/2" else sous + [(f"Ø_{q}/2", a["demi"])]
    return dessous + ([("jeu couches", P.JEU_COUCHES)] if P.JEU_COUCHES else []) + [("Ø/2", _demi(c))]


def positions(d: Dict, s: Dict) -> Dict[str, List[float]]:
    """{face: [position RÉELLE effective de chaque couche]} : la position saisie, sinon l'automatique du site — le dessin."""
    return {f: [float(c["dist"]) if c.get("dist") is not None else position_auto(d, s, f, j)
                for j, c in enumerate(s["faces"][f]["couches"])] for f in FACES}


def axes(d: Dict, s: Dict) -> Dict[str, List[float]]:
    """{face: [axe de CALCUL de chaque couche]} : ⌈position + jeu de pose⌉ au pas d'arrondi — ce que le moteur utilise."""
    jeu, pas = _jeu(d), _arrondi(d)
    return {f: [P.axe_calcul(e, jeu, pas) for e in es] for f, es in positions(d, s).items()}


# ------------------------------------------------------------------ la dalle que le moteur lit
def pour_moteur(d: Dict) -> Dict:
    """La copie de la dalle telle que le MOTEUR la lit : sans les clés du site (verrou, type) ; pour chaque couche
    automatique que le site pose ailleurs que le moteur (une couche empilée), ``dist`` prend sa position RÉELLE (le
    moteur en fait l'axe de calcul, comme d'une position saisie) ; les autres couches sont inchangées. ``d`` n'est
    jamais modifié."""
    out = {k: copy.deepcopy(v) for k, v in d.items() if k not in CLES_SITE}
    for s_d, s_m in zip(d.get("sections") or [], out.get("sections") or []):
        for f in FACES:
            for j, (c_d, c_m) in enumerate(zip(s_d["faces"][f]["couches"], s_m["faces"][f]["couches"])):
                if empilee(d, s_d, f, j):
                    c_m["dist"] = position_auto(d, s_d, f, j)
    return out


def _face_figee(d: Dict, s: Dict, f: str) -> Dict:
    """La géométrie de la face par le moteur des dalles (``_face`` : Σ As, y_G), ses couches aux positions EFFECTIVES."""
    face = s["faces"][f]
    couches = [dict(c, dist=e) for c, e in zip(face["couches"], positions(d, s)[f])]
    return MO_D._face(couches, face.get("yG"), d["b"], d.get("enrobage", EN_D.DEFAUTS_DALLE["enrobage"]), _jeu(d),
                      _arrondi(d))


def yG_calcule(d: Dict, s: Dict, f: str) -> float:
    """Le centre de gravité CALCULÉ de la face (cm), Σ As·e / Σ As sur les axes de calcul — ce que le champ « y_G »
    montre quand il est vide."""
    return _face_figee(d, s, f)["yG_calcule"]


def h_utile(d: Dict, s: Dict, f: str) -> float:
    """La hauteur utile de la face (cm) : h − y_G, le y_G imposé s'il l'est, sinon calculé sur les axes de calcul."""
    return float(d["h"]) - _face_figee(d, s, f)["yG"]


# ------------------------------------------------------------------ le résultat relu par le site
def formule_axe(c_r: Dict, j: int) -> Dict:
    """La formule « e » d'une couche (le détail de son axe de calcul) : termes et résultat du moteur, plus ce que l'écran
    dit (« posée sur la base Y », la pose dans une prédalle)."""
    det = c_r["detail"]
    out = {"cle": "e", "termes": [{"symbole": t["symbole"], "valeur": t["valeur"], "unite": "cm"} for t in det["termes"]],
           "resultat": c_r["e"], "unite": "cm", "couche": j, "auto": c_r["auto"], "empilee": c_r["empilee"],
           "libelle": c_r["libelle"], "detail": det}
    for k in ("pose", "appui", "appui_sec", "sur"):
        if k in c_r:
            out[k] = c_r[k]
    return out


def annoter(r: Dict, d: Dict) -> Dict:
    """Le résultat du moteur (``moteur(d).verifier(pour_moteur(d))``) relu par le site, en place. Pour chaque couche :
    ``auto`` (l'axe est automatique dans l'ÉTAT : le moteur, lui, a reçu la position d'une couche empilée),
    ``reel_auto`` (la position automatique du site : ce que le champ vide affiche), ``e_auto`` (son axe de calcul),
    ``e_auto_moteur`` (l'axe automatique du moteur, gardé), ``empilee``, ``detail`` (le calcul de l'axe, terme à terme :
    ``P.detail(termes(...), jeu, arrondi)``) ; une couche empilée reçoit aussi ``appui`` (le rang de la couche qui la
    porte ; ``appui_sec`` vrai pour un renfort principal posé sur un treillis), ``sur`` (la même en mots :
    ``nom_support``) et ``sur_court`` (sa forme courte, celle des légendes). Chaque vérification de face commence par la
    formule « e » de chacune de ses couches (``formule_axe`` : la dalle comme la prédalle ; celles du moteur des
    prédalles sont remplacées). Rend ``r``."""
    jeu, pas = _jeu(d), _arrondi(d)
    for s_d, s_r in zip(d.get("sections") or [], r["sections"]):
        for f in FACES:
            for j, (c_d, c_r) in enumerate(zip(s_d["faces"][f]["couches"], s_r["faces"][f]["couches"])):
                reel = position_auto(d, s_d, f, j)
                c_r["reel_auto"] = reel
                c_r["e_auto"] = P.axe_calcul(reel, jeu, pas)
                c_r["e_auto_moteur"] = P.axe_calcul(_reel_moteur(d, c_d, j, f), jeu, pas)
                c_r["auto"] = c_d.get("dist") is None
                c_r["empilee"] = empilee(d, s_d, f, j)
                c_r["detail"] = P.detail(termes(d, s_d, f, j), jeu, pas)
                for k in ("appui", "sur", "sur_court", "appui_sec"):
                    c_r.pop(k, None)
                if c_r["empilee"]:
                    a = support(d, s_d, f, j)
                    c_r["appui"] = a["couche"]
                    c_r["sur"] = nom_support(d, s_d, f, j)
                    c_r["sur_court"] = nom_support(d, s_d, f, j, court=True)
                    if a["sec"]:
                        c_r["appui_sec"] = True          # DAL5b : un renfort principal posé sur le treillis
        for v in s_r["verifications"]:
            if v["cle"] in FACES:
                v["formules"] = ([formule_axe(c_r, j) for j, c_r in enumerate(s_r["faces"][v["cle"]]["couches"])]
                                 + [fo for fo in v["formules"] if fo.get("cle") != "e"])
    if type_dalle(d) == "predalle":
        _peau_secondaire(r, d)
    return r


def _peau_secondaire(r: Dict, d: Dict) -> None:
    """Prédalle : la base inférieure SECONDAIRE est dans la peau, posée sur la principale ; si le dessus de ses barres
    (ou de ses fils, un treillis) sort de la peau, l'avertissement « secondaire_hors_predalle » (comme
    « base_hors_predalle » du moteur pour la base principale)."""
    p, q = principale(d), _secondaire(d)
    h_pre = float(d.get("h_pre") or MO_P.H_PRE_SI_NUL)
    for i, (s_d, s_r) in enumerate(zip(d.get("sections") or [], r["sections"])):
        c_r = s_r["faces"][f"inf_{q}"]["couches"][0]
        if not c_r["auto"]:
            continue
        haut = P.dessus(c_r["e_reel"], c_r["d"])
        if haut > h_pre + 1e-9:
            if treillis(d, s_d, "inf"):
                quoi = (f"le treillis est dans la prédalle, ses fils {q.upper()} soudés sur les fils {p.upper()} (à "
                        f"{_fr(c_r['e_reel'], 2)} cm, Ø{c_r['d']:g}) ; le dessus des fils {q.upper()}", "un treillis plus mince")
            else:
                quoi = (f"la base {q.upper()} est dans la prédalle, posée sur la base {p.upper()} (à {_fr(c_r['e_reel'], 2)} cm, "
                        f"Ø{c_r['d']:g}) ; le dessus des barres {q.upper()}", "des barres plus minces")
            r["avertissements"].append({
                "code": "secondaire_hors_predalle", "section": i, "face": f"inf_{q}", "e": c_r["e_reel"], "d": c_r["d"],
                "haut": haut, "h_pre": h_pre,
                "message": f"{s_r['titre']} · armature inférieure : {quoi[0]} ({_fr(haut, 2)} cm) sort de la peau de "
                           f"{_fr(h_pre)} cm. Épaissir la prédalle ou choisir {quoi[1]}."})


def _fr(x: float, nd: int = 1) -> str:
    """Un nombre à la française pour un message (« 4,5 », « 5,02 ») — l'écriture des messages du moteur."""
    return f"{x:.{nd}f}".replace(".", ",")


# ------------------------------------------------------------------ monter, descendre une couche (DAL5c)
PAS_AXE = 0.5           # cm : le pas des flèches ▲▼ de l'axe d'une couche (charte d'ergonomie E9 et § 3 : accepté le 13/11/2026)
TOL_CONTACT = 1e-6      # cm : deux couches au contact exact ne se traversent pas (les arrondis des flottants, rien de plus)


def _niveaux(d: Dict, s: Dict) -> List[tuple]:
    """Chaque couche de la section ``s`` à sa position RÉELLE effective, mesurée depuis le DESSOUS de la dalle (cm) :
    (face, rang, y, ½ Ø réel en cm, direction)."""
    h = float(d["h"])
    ax = positions(d, s)
    out = []
    for f in FACES:
        cote, dk = f.split("_")
        for j, c in enumerate(s["faces"][f]["couches"]):
            e = ax[f][j]
            out.append((f, j, e if cote == "inf" else h - e, MO_D.couche_diametre(c) / 20.0, dk))
    return out


def conflits(d: Dict, s: Dict) -> set:
    """Ce qui se traverse dans la section ``s`` aux axes effectifs (DAL5c) : ``("enrobage", f, j)``, une couche dont le
    bord entre dans l'enrobage (c depuis chaque parement) ; ``("croise", f1, j1, f2, j2)``, deux couches de directions
    différentes plus proches, d'axe à axe, que la demi-somme de leurs Ø — elles se croisent, l'une traverse l'autre (le
    critère du moteur des prédalles, « croisement_sur_predalle »). Les positions et les Ø RÉELS : deux couches
    automatiques posées l'une sur l'autre sont au contact exact, qui n'est pas un croisement (``TOL_CONTACT``). Deux
    couches de la MÊME direction peuvent partager un plan (un renfort entre les barres de sa base) : elles ne se gênent
    pas."""
    c = float(d.get("enrobage") or MO_D.ENROBAGE_SI_NUL)
    h = float(d["h"])
    niv = _niveaux(d, s)
    out = set()
    for f, j, y, r, _ in niv:
        if y - r < c - TOL_CONTACT or y + r > h - c + TOL_CONTACT:
            out.add(("enrobage", f, j))
    for k, (f1, j1, y1, r1, d1) in enumerate(niv):
        for f2, j2, y2, r2, d2 in niv[k + 1:]:
            if d1 != d2 and abs(y1 - y2) < r1 + r2 - TOL_CONTACT:
                out.add(("croise", f1, j1, f2, j2))
    return out


def nom_couche(d: Dict, s: Dict, f: str, j: int, nappe: bool = False) -> str:
    """La couche en mots, pour un message : « les fils Y du treillis », « la base X », « le renfort Y », « le renfort R2
    Y » (les repères de la rubrique Armatures) ; ``nappe`` : l'armature dite en plus (« … de l'armature supérieure »)."""
    cote, dk = f.split("_")
    if j == 0:
        nom = f"les fils {dk.upper()} du treillis" if treillis(d, s, cote) else f"la base {dk.upper()}"
    else:
        n = len(s["faces"][f]["couches"]) - 1
        nom = f"le renfort {dk.upper()}" if n == 1 else f"le renfort R{j} {dk.upper()}"
    return nom + (f" de l'armature {'inférieure' if cote == 'inf' else 'supérieure'}" if nappe else "")


def deplacer(d: Dict, i: int, f: str, j: int, sens: int) -> float:
    """La position RÉELLE (cm, depuis le parement de sa face) de la couche ``j`` de la face ``f`` de la section ``i``, MONTÉE
    (``sens`` = 1) ou DESCENDUE (− 1) d'un pas (``PAS_AXE``) et posée sur la grille du pas — vers le haut de la dalle :
    l'axe d'une couche inférieure grandit, celui d'une supérieure diminue (DAL5c : les flèches disent ce que l'œil voit
    sur la coupe). Les couches posées sur elle (automatiques) la suivent. REFUSÉ (ValueError, un message en français) si
    la couche, ou une couche qui la suit, traverserait alors une autre couche ou l'enrobage (un conflit qui n'existait
    pas : ``conflits``), et pour les fils secondaires d'un treillis unique (soudés : le treillis se déplace par ses fils
    principaux). Un pas qui atteint ou franchit la position automatique s'y arrête (un cran : elle est rarement sur la
    grille). La dalle n'est pas modifiée : l'appelant écrit la position rendue (« dist » ; l'automatique : vide)."""
    s = d["sections"][i]
    c = s["faces"][f]["couches"][j]
    cote, dk = f.split("_")
    verbe = "monter" if sens > 0 else "descendre"
    p = principale(d)
    if j == 0 and dk != p and treillis(d, s, cote):
        raise ValueError(f"Les fils {dk.upper()} du treillis sont soudés sur les fils {p.upper()} : le treillis se déplace "
                         f"par ses fils {p.upper()} (porteurs de la direction principale).")
    auto = position_auto(d, s, f, j)
    e = float(c["dist"]) if c.get("dist") is not None else auto
    delta = PAS_AXE if (sens > 0) == (cote == "inf") else -PAS_AXE
    e2 = round((e + delta) / PAS_AXE) * PAS_AXE
    # la position AUTOMATIQUE est un cran : réelle, elle tombe rarement sur la grille (3,6 ; 4,7 ; 5,6 cm — la règle du
    # 13/11/2026) ; un pas qui l'atteint ou la franchit s'y arrête, et ▲ puis ▼ ramènent la couche là où elle était
    if c.get("dist") is not None and abs(auto - e) > TOL_CONTACT and min(e, e2) - TOL_CONTACT <= auto <= max(e, e2) + TOL_CONTACT:
        e2 = auto
    essai = copy.deepcopy(d)
    s2 = essai["sections"][i]
    s2["faces"][f]["couches"][j]["dist"] = e2
    nouveaux = sorted(conflits(essai, s2) - conflits(d, s))
    if nouveaux:
        x = nouveaux[0]
        if x[0] == "enrobage":
            qui = "la couche" if (x[1], x[2]) == (f, j) else nom_couche(essai, s2, x[1], x[2], x[1].split("_")[0] != cote)
            raison = f"{qui} entrerait dans l'enrobage ({_fr(float(d.get('enrobage') or MO_D.ENROBAGE_SI_NUL))} cm)"
        else:
            a, b = (x[1], x[2]), (x[3], x[4])
            autre = b if a == (f, j) else a
            un = "la couche" if (f, j) in (a, b) else nom_couche(essai, s2, *(a if autre == b else b))
            e_autre = positions(essai, s2)[autre[0]][autre[1]]
            raison = (f"{un} croiserait {nom_couche(essai, s2, autre[0], autre[1], autre[0].split('_')[0] != cote)} "
                      f"(axe à {_fr(e_autre)} cm de son parement)")
        raise ValueError(f"Impossible de {verbe} {nom_couche(d, s, f, j)} de {_fr(PAS_AXE)} cm : {raison}.")
    return e2


# ------------------------------------------------------------------ le sélecteur de la nappe de base
def candidats_base(d: Dict, i: int, f: str, familles: Optional[Sequence[str]] = None,
                   espacements: Optional[Sequence[int]] = None) -> List[Dict]:
    """Le sélecteur de la nappe de BASE d'une face (``candidats_base`` du moteur du type), les niveaux du site respectés :
    le Ø d'un candidat déplace l'axe empilé — le sien (base secondaire) ou celui des couches secondaires posées sur lui
    (base principale : la base ET les renforts de la direction secondaire, S3 — tant qu'elle est la plus haute de son
    lit : sous un renfort principal plus haut, ce renfort les porte et un candidat plus mince ne les fait pas
    descendre, S5) —, que le moteur ne sait pas recalculer sur un axe imposé ; la dalle est donc recomposée candidat
    par candidat, et chacune calculée par le moteur (``verifier(pour_moteur(essai))``). Mêmes clés que le moteur (plus
    « e », l'axe de la couche), même tri. ``espacements`` (mm, les espacements ADMIS de la note — api_web/dalle.py,
    13/11/2026) : les barres à un autre espacement ne sont pas proposées ; sans lui, toutes celles du moteur."""
    MO = moteur(d)
    if f not in FACES:
        raise ValueError(f"face : {' '.join(FACES)} attendu (reçu {f!r})")
    sections = d.get("sections") or []
    if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(sections):
        raise ValueError(f"section n° {i!r} introuvable ({len(sections)} section(s))")
    fams = tuple(x for x in (familles or MO_D.FAMILLES) if x in MO_D.FAMILLES) or MO_D.FAMILLES
    base = sections[i]["faces"][f]["couches"][0]
    out = []
    cote, dk = f.split("_")
    soude = treillis(d, sections[i], cote)
    for cand in MO._liste_candidats(fams):
        if espacements is not None and cand["type"] == "barres" and cand["esp"] not in espacements:
            continue
        essai = copy.deepcopy(d)
        c1 = essai["sections"][i]["faces"][f]["couches"][0]
        c1.update(cand)
        if soude and cand["type"] == "treillis":
            # (comme api_web/dalle.choisir_section : sur un treillis unique, un treillis choisi pose sa transposée sur
            # l'autre direction — la nappe reste un treillis, ses fils soudés, DAL5b)
            autre = essai["sections"][i]["faces"][f"{cote}_{'y' if dk == 'x' else 'x'}"]["couches"][0]
            autre.update({"type": "treillis", "treillis": NAP.transposee(cand["treillis"])})
        s = MO.verifier(pour_moteur(essai))["sections"][i]
        g, geo = s["grandeurs"], s["faces"][f]["couches"][0]
        retenue = base["type"] == cand["type"] and (base["treillis"] == cand["treillis"] if cand["type"] == "treillis"
                                                    else (base["d"], base["esp"]) == (cand["d"], cand["esp"]))
        out.append({"nom": MO_D.candidat_nom(c1), "famille": cand["type"], "type": cand["type"], "treillis": c1["treillis"],
                    "d": c1["d"], "esp": c1["esp"], "As_pm": geo["As_pm"], "As": g[f"As_{f}"], "e": geo["e"], "yG": g[f"yG_{f}"],
                    "taux": g[f"besoin_{f}"] / g[f"As_{f}"], "etat": g[f"etat_{f}"], "ok": g[f"etat_{f}"] == MO_D.OK,
                    "statut_section": s["statut"], "retenue": retenue, "optimal": False})
    out.sort(key=lambda x: (x["As_pm"], MO_D.FAMILLES.index(x["famille"]), x["d"]))
    optimal = next((x for x in out if x["ok"]), None)
    if optimal is not None:
        optimal["optimal"] = True
    return out

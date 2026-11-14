# -*- coding: utf-8 -*-
"""Prédalle en béton armé — le MOTEUR, sans Streamlit : bibliothèque standard seule, Python 3.12 et 3.13 (donc
Pyodide 0.29). Extrait de l'écran Streamlit ``modules/predalle.py`` v1.1, qui reste la RÉFÉRENCE :
``tests/test_predalle_moteur.py`` prouve, cas par cas, que ce moteur rend les mêmes grandeurs, les mêmes états et les
mêmes pourcentages que son écran. Méthode, provenance, preuve : ``docs/beton/PREDALLE_MOTEUR.md``.

CE QU'EST L'ÉCRAN PRÉDALLE. L'en-tête de ``modules/predalle.py`` le dit : une « copie adaptée » de l'écran Dalle
(``modules/dalle.py``), « AUCUNE formule nouvelle » ; ce qui change est « la GÉOMÉTRIE DES COUCHES, rien d'autre ». Le
code le confirme ligne à ligne (le calcul ``_dimensionnement_compute_states`` est le même, mot pour mot) : la seule
différence de calcul est ``_auto_dist_couche``, l'axe AUTOMATIQUE d'une couche :

  - couche 1 (base) INFÉRIEURE de la direction PRINCIPALE : DANS la peau préfabriquée, en fond de moule —
    enrobage + demi-Ø arrondi au 0,5 cm supérieur, SANS jeu premier lit ;
  - toute autre couche INFÉRIEURE (base de la direction secondaire, tous les renforts inférieurs, posés sur chantier) :
    SUR la prédalle — h_pre + demi-Ø arrondi au 0,5 cm supérieur ;
  - couches SUPÉRIEURES : la règle de la dalle (enrobage + demi-Ø arrondi + jeu premier lit, depuis le dessus).

Ce moteur fait donc EXACTEMENT ce que fait l'écran pour sa note PDF (``_valeurs_pour_export``) : il FIGE l'axe
effectif de chaque couche (l'automatique de la prédalle, ou l'axe saisi) puis confie le calcul au moteur des dalles
(``beton/dalle``, extrait de ``modules/dalle.py`` et prouvé contre lui). Une seule copie des formules ; la parité avec
``modules/predalle.py`` est prouvée DIRECTEMENT (le test rejoue l'écran prédalle, pas l'écran dalle).

API
---
    nouvelle_predalle() / nouvelle_section(nom) / nouvelle_couche(base) / nom_section_suivant(predalle)
    normaliser(predalle) -> dict          copie complétée et contrôlée (ValueError hors du domaine de l'écran)
    distance_auto(couche, rang, face, principale, enrobage, jeu, h_pre) -> float    l'axe automatique (cm)
    pose(face, rang, principale) -> str   « predalle » | « sur_predalle » | « superieure » (où la règle pose la couche)
    verifier(predalle) -> dict            le calcul de toutes les sections
    candidats_base(predalle, i_section, face, familles=None) -> list   le sélecteur de la nappe de base d'une face

ENTRÉE — celle du moteur des dalles (docstring de ``beton/dalle/moteur.py``) plus ``"h_pre"`` : l'épaisseur de la peau
préfabriquée (cm, 3 ≤ h_pre ≤ min(15 ; h − 1)) ; défauts de l'écran prédalle : « Prédalle 1 », h = 22, h_pre = 6 ; et,
depuis le lot 6 (14/11/2026), ``"pas_couture"`` : le pas d'arrondi de la longueur des barres de couture (cm : 1, 5 ou 10 ;
5 par défaut, beton/predalle/joint.py) ; chaque section peut porter ``"couture"`` (lot 5).

SORTIE de verifier — celle du moteur des dalles (même format, mêmes clés), avec :
    "reference": "modules/predalle.py v1.1", "predalle": l'entrée normalisée (à la place de "dalle") ;
    chaque section : grandeurs « h_pre » et « h_coule » (h − h_pre, le béton coulé en place) ;
    chaque couche de chaque face : "auto" (axe automatique ?), "e_auto" (l'automatique DE LA PRÉDALLE), "pose" ;
    chaque vérification de face : en tête de ses "formules", une formule « e » par couche (l'axe et sa règle) ;
    "avertissements" : ceux du moteur des dalles, plus deux propres à la prédalle (absents de l'écran Streamlit, voir
    la doc) : « base_hors_predalle » (la base posée dans la peau dépasse la peau) et « croisement_sur_predalle » (deux
    couches inférieures croisées, posées toutes deux sur la prédalle à des niveaux trop proches pour se croiser) ;
    LE JOINT (lot 5, beton/predalle/joint.py) : la vérification « joint » de chaque section (la direction secondaire
    vérifiée une seconde fois, au joint de deux prédalles, avec les barres de couture), ``section["joint"]``, la hauteur
    du joint quand elle gouverne, les grandeurs « …_joint », les statuts recalculés — ceux de la pleine prédalle (l'écran
    Streamlit, la note PDF) gardés dans ``pleine_predalle`` —, et ses avertissements (« joint_… »).
"""
import copy

from beton import position as P
from beton import treillis as TR
from beton.dalle import moteur as MD

from . import entree as E
from . import joint as J
from .entree import (CLES_PROPRES, classes_beton, h_pre_max, nom_section_suivant, normaliser, nouvelle_couche,  # noqa: F401
                     nouvelle_predalle, nouvelle_section)

REFERENCE = "modules/predalle.py v1.1"
OK, NOK = MD.OK, MD.NOK
H_PRE_SI_NUL = 6.0          # cm : « float(h_pre or 6.0) » dans _auto_dist_couche (inatteignable : h_pre ≥ 3)

POSES = {"predalle": "dans la prédalle", "sur_predalle": "sur la prédalle", "superieure": "depuis le dessus"}
FAMILLES = MD.FAMILLES
direction_secondaire = MD.direction_secondaire
ordre_faces = MD.ordre_faces
titre_face = MD.titre_face
couche_libelle = MD.couche_libelle
couche_valeur = MD.couche_valeur
couche_diametre = MD.couche_diametre
couche_as_par_m = MD.couche_as_par_m
candidat_nom = MD.candidat_nom
lire_candidat = MD.lire_candidat


# ============================================================ la règle prédalle
def pose(face: str, rang: int, principale: str) -> str:
    """Où la règle pose la couche ``rang`` (0 = la base) d'une face : « predalle » (une BASE inférieure, de l'une ou de
    l'autre direction : dans la peau, en usine — la base secondaire DEDANS aussi depuis le 13/11/2026, la demande du
    propriétaire ; l'écran Streamlit la posait sur la prédalle), « sur_predalle » (un renfort inférieur, posé au
    chantier), « superieure ». ``principale`` n'est plus lue (gardée pour les appelants)."""
    cote, _ = face.split("_")
    if cote == "inf":
        return "predalle" if rang == 0 else "sur_predalle"
    return "superieure"


def position_auto(c: dict, rang: int, face: str, principale: str, enrobage: float, h_pre: float) -> float:
    """La position RÉELLE automatique de la couche (cm, depuis son parement) : un renfort inférieur sur la prédalle,
    h_pre + Ø/2 ; toute autre couche au contact de l'enrobage, c + Ø/2 (beton/position.py). Le moteur ne pose pas une
    couche sur une autre : la base secondaire de la peau, posée sur la principale, l'est par le site
    (api_web/dalle_positions.py) ; seul, le moteur la dit au niveau de la principale (« nappes_confondues »)."""
    d = MD.couche_diametre(c)
    if pose(face, rang, principale) == "sur_predalle":
        return float(h_pre or H_PRE_SI_NUL) + d / 20.0
    return P.sur_parement(float(enrobage or MD.ENROBAGE_SI_NUL), d)


def termes_auto(c: dict, rang: int, face: str, principale: str, enrobage: float, h_pre: float) -> list:
    """La position réelle automatique en termes, pour le détail du calcul : [(« h_p », …), (« Ø/2 », …)] sur la
    prédalle, sinon [(« c », …), (« Ø/2 », …)]."""
    d = MD.couche_diametre(c)
    if pose(face, rang, principale) == "sur_predalle":
        return [("h_p", float(h_pre or H_PRE_SI_NUL)), ("Ø/2", d / 20.0)]
    return [("c", float(enrobage or MD.ENROBAGE_SI_NUL)), ("Ø/2", d / 20.0)]


def distance_auto(c: dict, rang: int, face: str, principale: str, enrobage: float, jeu: float, h_pre: float,
                  arrondi: float = P.ARRONDI) -> float:
    """L'axe de CALCUL automatique (cm) : ⌈position réelle + jeu de pose⌉ au pas d'arrondi — le jeu vaut pour toute
    couche, celles de la peau comprises (décision du propriétaire du 13/11/2026 : une prédalle de 6 cm garde c = 3 cm et
    le jeu de 1 cm)."""
    return P.axe_calcul(position_auto(c, rang, face, principale, enrobage, h_pre), jeu, arrondi)


def _positions_auto(p: dict) -> list:
    """Pour chaque section, {face: [position réelle automatique de chaque couche]}."""
    pr = p["dir_principale"]
    return [{f: [position_auto(c, j, f, pr, p["enrobage"], p["h_pre"]) for j, c in enumerate(s["faces"][f]["couches"])]
             for f in E.FACES} for s in p["sections"]]


def figer(p: dict, reels: list = None) -> dict:
    """La prédalle (normalisée) en DALLE du moteur des dalles : la position de chaque couche automatique est FIGÉE à la
    position réelle de la prédalle (« dist ») ; le moteur des dalles en fait l'axe de calcul (jeu, arrondi)."""
    reels = _positions_auto(p) if reels is None else reels
    d = {k: copy.deepcopy(v) for k, v in p.items() if k not in CLES_PROPRES}
    for s in d["sections"]:
        s.pop("couture", None)                   # (le joint : beton/predalle/joint.py — le moteur des dalles l'ignore)
    for s, a in zip(d["sections"], reels):
        for f in E.FACES:
            for j, c in enumerate(s["faces"][f]["couches"]):
                if c["dist"] is None:
                    c["dist"] = a[f][j]
    return d


# ============================================================ le calcul
def _fr(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _t(symbole: str, valeur, unite: str = "") -> dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def _f(cle: str, termes: list, resultat, unite: str) -> dict:
    return {"cle": cle, "termes": termes, "resultat": resultat, "unite": unite}


def _formule_axe(geo_c: dict, j: int) -> dict:
    """La formule de l'axe d'une couche : son calcul terme à terme (``detail`` : la position réelle, le jeu, l'arrondi —
    beton/position.py), l'axe de calcul pour résultat."""
    det = geo_c["detail"]
    out = _f("e", [_t(t["symbole"], t["valeur"], "cm") for t in det["termes"]], geo_c["e"], "cm")
    out.update(couche=j, pose=geo_c["pose"], auto=geo_c["auto"], libelle=geo_c["libelle"], detail=det)
    return out


def verifier(predalle: dict) -> dict:
    """Calcule chaque section de la prédalle (format en tête du fichier). Lève ValueError si la prédalle sort du
    domaine de l'écran."""
    p = normaliser(predalle)
    reels = _positions_auto(p)
    figee = figer(p, reels)
    r = MD.verifier(figee)
    r["reference"] = REFERENCE
    del r["dalle"]
    r["predalle"] = p
    pr = p["dir_principale"]
    for s, sp, a in zip(r["sections"], p["sections"], reels):
        g = s["grandeurs"]
        g["h_pre"] = p["h_pre"]
        g["h_coule"] = p["h"] - p["h_pre"]
        for f in E.FACES:
            for j, (c, cp) in enumerate(zip(s["faces"][f]["couches"], sp["faces"][f]["couches"])):
                c["auto"] = cp["dist"] is None
                c["reel_auto"] = a[f][j]
                c["e_auto"] = P.axe_calcul(a[f][j], p["jeu_premier_lit"], p["arrondi"])
                c["pose"] = pose(f, j, pr)
                if c["auto"]:                     # le moteur des dalles a reçu une position figée : ses termes
                    c["detail"] = P.detail(termes_auto(cp, j, f, pr, p["enrobage"], p["h_pre"]), p["jeu_premier_lit"],
                                           p["arrondi"])
        for v in s["verifications"]:
            if v["cle"] in E.FACES:
                v["formules"] = [_formule_axe(c, j) for j, c in enumerate(s["faces"][v["cle"]]["couches"])] + v["formules"]
    r["avertissements"] += _avertissements(p, r)
    J.joints(p, figee, r, _formule_axe)          # le JOINT des prédalles et ses barres de couture (lot 5)
    return r


# ============================================================ les avertissements propres à la prédalle
def _avertissements(p: dict, r: dict) -> list:
    """Deux contrôles de GÉOMÉTRIE que l'écran ne fait pas (il calcule sans rien dire) — aucun chiffre n'en change :
    - la base inférieure principale, posée dans la peau, en sort (axe + Ø/2 > h_pre) ;
    - deux couches inférieures CROISÉES (directions X et Y), posées sur la prédalle, calculées à des niveaux trop
      proches pour se croiser (axes distants de moins de la demi-somme des Ø) : l'une
      passe sur l'autre, son d est plus petit (le cas base contre base est celui du moteur des dalles,
      « nappes_confondues »)."""
    out = []
    pr = p["dir_principale"].lower()
    for i, s in enumerate(r["sections"]):
        f_p = f"inf_{pr}"
        c0 = s["faces"][f_p]["couches"][0]
        haut = P.dessus(c0["e_reel"], c0["d"])
        if c0["auto"] and haut > p["h_pre"] + 1e-9:
            out.append({"code": "base_hors_predalle", "section": i, "face": f_p, "e": c0["e_reel"], "d": c0["d"], "haut": haut,
                        "h_pre": p["h_pre"],
                        "message": f"{s['titre']} · armature inférieure {pr.upper()} (base, dans la prédalle) : posée à "
                                   f"{_fr(c0['e_reel'])} cm, Ø{c0['d']:g} ; le dessus des "
                                   f"{'fils' if c0['type'] == 'treillis' else 'barres'} ({_fr(haut, 2)} cm) sort de "
                                   f"la peau de {_fr(p['h_pre'])} cm. Épaissir la prédalle ou saisir la position de la couche."})
        paires = []
        fx, fy = s["faces"]["inf_x"]["couches"], s["faces"]["inf_y"]["couches"]
        for jx, cx in enumerate(fx):
            for jy, cy in enumerate(fy):
                if jx == 0 and jy == 0:
                    continue                     # base contre base : l'avertissement du moteur des dalles
                if cx["pose"] != "sur_predalle" or cy["pose"] != "sur_predalle" or not (cx["auto"] and cy["auto"]):
                    continue
                mini = (cx["d"] + cy["d"]) / 2.0
                if abs(cx["e_reel"] - cy["e_reel"]) * 10.0 < mini - 1e-9:
                    paires.append((jx, cx, jy, cy, mini))
        if paires:
            jx, cx, jy, cy, mini = paires[0]

            def nom(dk, j, c):
                return f"{dk.upper()} {'base' if j == 0 else f'renfort {j}'} ({c['valeur']}, posé à {_fr(c['e_reel'])} cm)"
            plus = f" (et {len(paires) - 1} autre(s) paire(s))" if len(paires) > 1 else ""
            out.append({"code": "croisement_sur_predalle", "section": i, "paires": [[a, b] for a, _, b, _, _ in paires],
                        "ecart_mini": max(x[4] for x in paires) / 10.0,
                        "message": f"{s['titre']} · armatures inférieures : {nom('x', jx, cx)} et {nom('y', jy, cy)} sont posées "
                                   f"toutes deux sur la prédalle, à des niveaux trop proches pour se croiser (il faut au moins "
                                   f"{_fr(mini / 10.0)} cm d'axe à axe){plus}. Croisées, l'une "
                                   "passe sur l'autre : son d est plus petit que celui calculé. Saisir sa position pour en "
                                   "tenir compte."})
    return out


# ============================================================ le sélecteur de la nappe de base
def _liste_candidats(familles) -> list:
    out = []
    if "treillis" in familles:
        out += [{"type": "treillis", "treillis": des} for des in TR.liste_choix()]
    if "barres" in familles:
        out += [{"type": "barres", "d": dd, "esp": e} for dd in MD.CANDIDATS_D for e in MD.CANDIDATS_ESP]
    return out


def candidats_base(predalle: dict, i_section: int, face: str, familles=None) -> list:
    """Le sélecteur de la nappe de BASE d'une face : chaque treillis de la base et chaque disposition de barres (Ø 8 à
    25, espacements 100 à 250 mm) remplace la couche 1 ; son axe automatique est RECALCULÉ par la règle de la prédalle
    (le Ø le déplace), un axe saisi est gardé, les renforts et le y_G saisi restent. Triées par section au mètre de la
    couche ; « optimal » marque la plus petite « ok », « retenue » la nappe actuelle."""
    if face not in E.FACES:
        raise ValueError(f"face : {' '.join(E.FACES)} attendu (reçu {face!r})")
    p = normaliser(predalle)
    if not isinstance(i_section, int) or isinstance(i_section, bool) or not 0 <= i_section < len(p["sections"]):
        raise ValueError(f"section n° {i_section!r} introuvable ({len(p['sections'])} section(s))")
    fams = tuple(f for f in (familles or FAMILLES) if f in FAMILLES) or FAMILLES
    seule = dict(p, sections=[p["sections"][i_section]])
    base = p["sections"][i_section]["faces"][face]["couches"][0]
    out = []
    for cand in _liste_candidats(fams):
        essai = copy.deepcopy(seule)
        c1 = essai["sections"][0]["faces"][face]["couches"][0]
        c1.update(cand)
        # le calcul du moteur des dalles sur les axes figés par la règle prédalle (le candidat est contrôlé par sa
        # normalisation) — le chemin de verifier(), sans ce qui ne sert pas ici (formules d'axes, avertissements prédalle)
        s = MD.verifier(figer(essai))["sections"][0]
        g = s["grandeurs"]
        retenue = base["type"] == cand["type"] and (base["treillis"] == cand["treillis"] if cand["type"] == "treillis"
                                                    else (base["d"], base["esp"]) == (cand["d"], cand["esp"]))
        out.append({"nom": MD.candidat_nom(c1), "famille": cand["type"], "type": cand["type"], "treillis": c1["treillis"],
                    "d": c1["d"], "esp": c1["esp"], "As_pm": MD.couche_as_par_m(c1, p["b"]), "As": g[f"As_{face}"],
                    "e": s["faces"][face]["couches"][0]["e"], "yG": g[f"yG_{face}"], "taux": g[f"besoin_{face}"] / g[f"As_{face}"],
                    "etat": g[f"etat_{face}"], "ok": g[f"etat_{face}"] == OK, "statut_section": s["statut"],
                    "retenue": retenue, "optimal": False})
    out.sort(key=lambda x: (x["As_pm"], FAMILLES.index(x["famille"]), x["d"]))
    optimal = next((x for x in out if x["ok"]), None)
    if optimal is not None:
        optimal["optimal"] = True
    return out

# -*- coding: utf-8 -*-
"""Cornière d'appui — l'ENTRÉE du moteur refondu (10/11/2026) : les grandeurs du § 2 de
``docs/acier/CORNIERE_DECISIONS.md`` (toutes en mm, kN, N/mm²), leurs défauts, leur domaine, le catalogue des
cornières (``acier/profiles``, source unique des profilés, familles L et LI), les nuances du site (source unique :
``acier/flambement/normes.py``, Tableau 3.1), la normalisation STRICTE et la relecture tolérante d'une note du module
d'avant (version 1, modèle lame).

Bibliothèque standard ; le catalogue (``acier.profiles.catalogue``) charge numpy, présent dans le navigateur.

Une valeur hors du domaine lève ValueError avec un message : jamais une valeur ramenée en silence. Les refus de
domaine (décisions § 2) : la charge doit tomber sur la partie droite de l'aile horizontale pour ses trois positions
(e + r < X − dx et X + dx ≤ larg) ; l'ancrage, trou compris, doit rester dans la partie droite de l'aile verticale
pour ses trois positions (Y − dy − d0/2 ≥ e + r et Y + dy + d0/2 ≤ haut — les mêmes deux conditions en A et en B,
puisque Y se compte depuis la face extérieure de l'aile horizontale dans les deux configurations) ; t < A, t < B et
e + r < A, e + r < B (une cornière, avec une partie droite) ; r ≥ 0 ; L > 0 ; n ≥ 1 ; n·d0 < L ; R ≥ 0 ; coef ≥ 1 ;
dx ≥ 0 ; dy ≥ 0 ; 0 ≤ jeu_ancrage < Y ; 0 ≤ yc < bras le plus court ; 0 ≤ uV ≤ e ; f_y parmi les nuances du site.
Révision § 9 (10/11 soir) : l'état limite (ELS : f_y/coef ; ELU : f_y/γ_M0, γ_M0 = 1,0, R saisie à l'ELU), les ancrages
(entraxe s avec les minima de l'EN 1993-1-8 Tableau 3.3 — s ≥ 2,2·d0, distance au bout e₁ ≥ 1,2·d0 —, type, profondeur,
rondelles ISO 7089 / 7093-1) et le support (voile, ou poutre dont les deux limites portent les trous et le contact).
"""
import math
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from acier.flambement import normes as NO
from acier.profiles import catalogue as CAT

REFERENCE = "classeur de l'ingénieur, feuille P8 (docs/acier/corniere_excel) — décisions du 10/11/2026"

# ------------------------------------------------------------------ les choix
AILES = ("A", "B")                               # laquelle des deux ailes porte la charge (A = la première du nom)
CONFIGURATIONS = ("A", "B")                      # A : aile horizontale en bas (le classeur) ; B : en haut (retournée)
HISTORIQUE, EQUILIBRE = "historique", "equilibre"
MODES = (HISTORIQUE, EQUILIBRE)
ELS, ELU = "ELS", "ELU"                          # l'état limite de la vérification (§ 9.5)
ETATS_LIMITES = (ELS, ELU)
GAMMA_M0 = 1.0                                   # NBN EN 1993-1-1 ANB : γ_M0 = 1,0 (ELU, critère élastique § 6.2.1(5))
VOILE, POUTRE = "voile", "poutre"
SUPPORTS = (VOILE, POUTRE)
TYPES_ANCRAGE = ("mecanique", "chimique", "traversant")
RONDELLES_ADMISES = (0, 1, 2)
FAMILLES = ("L", "LI")                           # cornières à ailes égales, à ailes inégales (EN 10056-1)
LIBELLES_FAMILLE = {"L": "Ailes égales", "LI": "Ailes inégales"}
LIBELLES_CONFIGURATION = {"A": "aile horizontale en bas", "B": "aile horizontale en haut"}
LIBELLES_MODE = {HISTORIQUE: "méthode historique (classeur P8)", EQUILIBRE: "équilibre"}
LIBELLES_ETAT_LIMITE = {ELS: "ELS — contraintes admissibles (f_y/coef)", ELU: "ELU — critère de von Mises (f_y/γ_M0, γ_M0 = 1,0)"}
LIBELLES_SUPPORT = {VOILE: "voile en béton", POUTRE: "poutre en béton"}
LIBELLES_TYPE_ANCRAGE = {"mecanique": "cheville mécanique", "chimique": "cheville chimique (tige scellée)",
                         "traversant": "boulon traversant"}
LIBELLES_RONDELLES = {0: "sans rondelle", 1: "une rondelle ISO 7089 sous l'écrou",
                      2: "rondelle large ISO 7093-1 contre l'aile et rondelle ISO 7089 sous l'écrou"}
# les rondelles plates (ISO 7089, série normale ; ISO 7093-1, série large) : d → (Ø extérieur, épaisseur) en mm
RONDELLES: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    8: ((16.0, 1.6), (24.0, 2.0)), 10: ((20.0, 2.0), (30.0, 2.5)), 12: ((24.0, 2.5), (37.0, 3.0)), 14: ((28.0, 2.5), (44.0, 3.0)),
    16: ((30.0, 3.0), (50.0, 3.0)), 18: ((34.0, 3.0), (56.0, 4.0)), 20: ((37.0, 3.0), (60.0, 4.0)), 22: ((39.0, 3.0), (66.0, 5.0)),
    24: ((44.0, 4.0), (72.0, 5.0)), 27: ((50.0, 4.0), (85.0, 6.0)), 30: ((56.0, 4.0), (92.0, 6.0)), 36: ((66.0, 5.0), (110.0, 8.0))}


def rondelle(d: float) -> Dict[str, Tuple[float, float]]:
    """Les dimensions (Ø extérieur, épaisseur) de la rondelle normale et de la rondelle large pour le diamètre ``d`` :
    la ligne de la table dont le diamètre nominal est le plus proche (au-dessus en cas d'égalité)."""
    k = min(RONDELLES, key=lambda x: (abs(x - d), -x))
    return {"normale": list(RONDELLES[k][0]), "large": list(RONDELLES[k][1]), "nominal": k}   # listes : JSON strict

# ------------------------------------------------------------------ les nuances : la source unique des f_y du site
# acier/flambement/normes.NUANCES (EN 1993-1-1 Tableau 3.1) : (nuance) → (f_y pour t ≤ 40 mm, f_y pour 40 < t ≤ 80)
NUANCES: Dict[str, float] = {nom: float(v[0]) for nom, v in NO.NUANCES.items()}
FY_ADMIS: Tuple[float, ...] = tuple(sorted(NUANCES.values()))
T_FY_REDUIT = 40.0                               # mm : au-delà, le Tableau 3.1 réduit f_y (information, pas de refus)


def nuance_de(fy: float) -> Optional[str]:
    """Le nom de la nuance du site dont la limite élastique (t ≤ 40 mm) vaut ``fy`` ; None si aucune."""
    for nom, v in NUANCES.items():
        if v == fy:
            return nom
    return None


# ------------------------------------------------------------------ les champs : (défaut, unité, libellé)
PROFIL_DEFAUT = "L 120x120x15"                   # la cornière du cas de contrôle du classeur
CHAMPS: Dict[str, Tuple[Optional[float], str, str]] = {
    "A": (120.0, "mm", "aile A"),
    "B": (120.0, "mm", "aile B"),
    "t": (15.0, "mm", "épaisseur t"),
    "r": (13.0, "mm", "congé r"),
    "fy": (355.0, "N/mm²", "limite élastique f_y"),
    "coef": (1.5, "", "coefficient sur f_y"),
    "R": (35.0, "kN", "réaction R"),
    "L": (300.0, "mm", "longueur reprise L"),
    "X": (75.0, "mm", "position de la charge X"),
    "dx": (10.0, "mm", "jeu de la charge dx"),
    "a": (None, "mm", "largeur d'appui a"),
    "R_ELU": (None, "kN", "réaction R_ELU"),
    "Y": (85.0, "mm", "position de l'ancrage Y"),
    "dy": (5.0, "mm", "tolérance de l'ancrage dy"),
    "n": (3, "", "nombre d'ancrages n"),
    "d": (16.0, "mm", "diamètre d'ancrage d"),
    "jeu_ancrage": (10.0, "mm", "jeu d'ancrage (C76)"),
    "yc": (0.0, "mm", "pivot y_c"),
    "uV": (None, "mm", "ligne d'action u_V"),
    "entraxe": (None, "mm", "entraxe des ancrages s"),
    "profondeur": (100.0, "mm", "profondeur d'ancrage"),
    "rondelles": (1, "", "rondelles"),
    "support_haut": (150.0, "mm", "poutre : étendue au-dessus de la face extérieure de l'aile horizontale"),
    "support_bas": (150.0, "mm", "poutre : étendue au-dessous de la face extérieure de l'aile horizontale"),
}
CHOIX: Dict[str, Tuple[str, Tuple[str, ...]]] = {"aile_horizontale": ("A", AILES), "configuration": ("A", CONFIGURATIONS),
                                                 "mode": (HISTORIQUE, MODES), "etat_limite": (ELS, ETATS_LIMITES),
                                                 "support": (VOILE, SUPPORTS), "type_ancrage": ("mecanique", TYPES_ANCRAGE)}
# les clés d'une cornière (version 2) ; ``v1`` : relue d'une note du module d'avant (le site en fait un avis)
CLES = ("nom", "standard", "profil", "A", "B", "t", "r", "aile_horizontale", "configuration", "etat_limite", "fy", "coef",
        "R", "L", "X", "dx", "a", "R_ELU", "Y", "dy", "n", "d", "entraxe", "type_ancrage", "profondeur", "rondelles",
        "support", "support_haut", "support_bas", "mode", "jeu_ancrage", "yc", "uV", "v1")
ENTIERS = ("n", "rondelles")
OPTIONNELS = ("a", "R_ELU", "uV", "entraxe")     # « vide = auto » : None reste None (a = L, R_ELU = 1,5·R, uV = e/2, s = L/n)
# EN 1993-1-8 Tableau 3.3 : entraxe minimal p₁ = 2,2·d0, distance au bout minimale e₁ = 1,2·d0 (trous dans une pièce d'acier)
ENTRAXE_MIN, BOUT_MIN = 2.2, 1.2
RAPPORT_ELU_DEFAUT = 1.5                         # R_ELU = 1,5·R quand R_ELU est vide [à confirmer]
JEU_TROU = 2.0                                   # d0 = d + 2 mm
NOM_MAX = 60
RHO_ACIER = 7850.0                               # kg/m³ (masse d'une saisie libre)

# ------------------------------------------------------------------ la version 1 (modèle lame, 07/11 – 10/11)
CLES_V1 = ("nom", "standard", "profil", "A", "B", "t", "charge", "V", "L", "e", "gamma_ELU", "s", "bord",
           "sig_lim_ELS", "tau_lim_ELS", "sig_lim_ELU")
_V1_SEULES = frozenset(CLES_V1) - frozenset(CLES)
_V2_SEULES = frozenset(CLES) - frozenset(CLES_V1)
AVIS_V1 = ("note d'avant le 10/11/2026, modèle lame : charges et ancrage à ressaisir, les anciens chiffres ne sont "
           "plus calculés")


# ------------------------------------------------------------------ le catalogue
@lru_cache(maxsize=1)
def _catalogue() -> Tuple[Dict, ...]:
    """Les cornières du catalogue (acier/donnees/profiles.json), triées par masse puis par nom."""
    recs = [p for p in CAT.profils() if p["famille"] in FAMILLES]
    return tuple(sorted(recs, key=lambda p: (p["masse"], p["nom"])))


def cornieres(familles=None) -> List[Dict]:
    """Les cornières du catalogue (les familles demandées ; toutes par défaut), triées par masse croissante."""
    fams = tuple(familles) if familles else FAMILLES
    return [p for p in _catalogue() if p["famille"] in fams]


def section_catalogue(nom: str) -> Dict:
    """Une cornière du catalogue par son nom (« L 120x120x15 ») ; ValueError si ce n'en est pas une."""
    for p in _catalogue():
        if p["nom"] == nom:
            return p
    raise ValueError(f"profil : « {nom} » n'est pas une cornière du catalogue")


def designation(nom: str) -> str:
    """« L 120x120x15 » → « L 120×120×15 », « L 45x45x4.5 » → « L 45×45×4,5 » (le signe de multiplication et la
    virgule du bureau)."""
    return nom.replace("x", "×").replace(".", ",")


def nom_catalogue(texte: str) -> Optional[str]:
    """Une désignation tapée (« 80x80x8 », « L 80×80×8 », « l80*80*8 », « 100 x 65 x 8 », ailes dans un ordre ou
    dans l'autre, « 4,5 ») → le nom de la cornière au catalogue (« L 80x80x8 ») ; None si elle n'y est pas."""
    if not isinstance(texte, str):
        return None
    s = texte.strip().lower().replace("×", "x").replace("*", "x").replace(",", ".").replace(" ", "").replace("\xa0", "")
    s = s[1:] if s.startswith("l") else s
    m = re.fullmatch(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    a, b, t = (float(x) for x in m.groups())
    noms = {p["nom"] for p in _catalogue()}
    for x, y in ((a, b), (b, a)):
        nom = f"L {x:g}x{y:g}x{t:g}"
        if nom in noms:
            return nom
    return None


# ------------------------------------------------------------------ la cornière neuve
def nouvelle_corniere(nom: str = "Cornière 1") -> Dict:
    """Une cornière aux valeurs par défaut des décisions (§ 2, § 9) : L 120×120×15 du catalogue, configuration A, mode
    historique, ELS, S355, coef 1,5, R = 35 kN sur 300 mm, X = 75 ± 10, Y = 85 ± 5, 3 chevilles mécaniques Ø16 à L/n,
    profondeur 100, une rondelle, voile."""
    c = {"nom": nom, "standard": True, "profil": PROFIL_DEFAUT, "v1": False}
    c.update({k: v[0] for k, v in CHOIX.items()})
    c.update({k: v[0] for k, v in CHAMPS.items()})
    return {k: c[k] for k in CLES}


# ------------------------------------------------------------------ la section (dimensions selon l'aile horizontale)
def _g(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _f(x: float, nd: int = 1) -> str:
    """Un nombre à ``nd`` décimales FIXES, à la virgule (« 236,7 ») : les textes du moteur (hypothèses, avertissements,
    informations) s'affichent tels quels à l'écran — jamais le point décimal (relecture de la mécanique)."""
    return f"{x:.{nd}f}".replace(".", ",")


def section(c: Dict) -> Dict:
    """La cornière calculée : celle du catalogue (``standard``) ou la saisie libre A × B × t, r. ``larg`` est l'aile
    qui porte la charge (``aile_horizontale``), ``haut`` l'autre ; ``e`` = t ; ``r`` = congé (r1 au catalogue, saisi
    sinon). Masse et aire : du catalogue, ou calculées (t·(A + B − t) + (1 − π/4)·r², sans les arrondis de bout
    d'aile, qui manquent à la saisie : +0,5 % sur la L 120×120×15) pour une saisie libre."""
    if c["standard"]:
        p = section_catalogue(c["profil"])
        A, B, t, r = float(p["h"]), float(p["b"]), float(p["t"]), float(p.get("r1") or 0.0)
        sec = {"source": "catalogue", "nom": p["nom"], "designation": designation(p["nom"]), "famille": p["famille"],
               "A": A, "B": B, "t": t, "r1": r, "r2": float(p.get("r2") or 0.0), "masse": p["masse"], "aire": p["A"],
               "masse_source": "catalogue", "norme": "EN 10056-1"}
    else:
        A, B, t, r = c["A"], c["B"], c["t"], c["r"]
        aire_mm2 = t * (A + B - t) + (1.0 - math.pi / 4.0) * r * r
        sec = {"source": "saisie", "nom": None, "designation": f"{_g(A)} × {_g(B)} × {_g(t)}", "famille": None,
               "A": A, "B": B, "t": t, "r1": r, "r2": None, "masse": aire_mm2 * 1e-6 * RHO_ACIER, "aire": aire_mm2 / 100.0,
               "masse_source": "calculée", "norme": None}
    sec["aile_horizontale"] = c["aile_horizontale"]
    sec["larg"], sec["haut"] = (A, B) if c["aile_horizontale"] == "A" else (B, A)
    sec["e"], sec["r"] = t, r
    return sec


def d0(c: Dict) -> float:
    """Le trou de passage : d + 2 mm."""
    return c["d"] + JEU_TROU


# ------------------------------------------------------------------ la version 1 : détection, conversion
def est_v1(c) -> bool:
    """Une cornière du module d'avant (modèle lame) : une de ses clés propres (V, e, s, gamma_ELU, bord, charge,
    sig_lim_*…) et aucune clé propre au modèle refondu."""
    return isinstance(c, dict) and bool(_V1_SEULES & set(c)) and not (_V2_SEULES & set(c))


def _demi(x: float) -> float:
    """Arrondi au demi-millimètre."""
    return round(2.0 * x) / 2.0


def depuis_v1(c: Dict) -> Dict:
    """Une cornière v1 relue : le nom et la section (catalogue, ou A, B, t de la saisie libre — le congé, inconnu du
    modèle lame, vaut 0 : le côté sûr) sont GARDÉS ; tout le reste prend les défauts du § 2, la charge et l'ancrage
    étant posés DANS LES AILES de cette cornière (X, dx proportionnels à l'aile horizontale, Y, dy à l'aile
    verticale, sur le rapport du cas de contrôle L120 ; le diamètre ramené au trou qui tient) — pour que la note
    s'ouvre et se calcule, l'avis ``AVIS_V1`` disant que ces valeurs sont à ressaisir. Rien n'est refusé ici ; la
    normalisation ordinaire suit."""
    out = nouvelle_corniere(str(c.get("nom") or "Cornière 1"))
    if isinstance(c.get("standard"), bool):
        out["standard"] = c["standard"]
    if isinstance(c.get("profil"), str):
        out["profil"] = c["profil"]
    for k in ("A", "B", "t"):
        v = c.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out[k] = float(v)
    out["r"] = 0.0
    out["v1"] = True
    try:
        sec = section(out)
    except ValueError:
        return out                                   # profil inconnu : la normalisation le dira
    ref = section_catalogue(PROFIL_DEFAUT)
    larg, haut = sec["larg"], sec["haut"]
    out["X"], out["dx"] = _demi(CHAMPS["X"][0] * larg / ref["h"]), _demi(CHAMPS["dx"][0] * larg / ref["h"])
    out["Y"], out["dy"] = _demi(CHAMPS["Y"][0] * haut / ref["b"]), _demi(CHAMPS["dy"][0] * haut / ref["b"])
    marge = min(out["Y"] - out["dy"] - (sec["e"] + sec["r"]), haut - out["Y"] - out["dy"])
    out["d"] = float(max(1, min(int(CHAMPS["d"][0]), math.floor(2.0 * marge - JEU_TROU))))
    return out


# ------------------------------------------------------------------ la normalisation stricte
def _nombre(cle: str, x) -> float:
    libelle = CHAMPS[cle][2]
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{libelle} : un nombre est attendu, pas {x!r}")
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"{libelle} : nombre infini ou indéterminé")
    return x + 0.0                                   # « -0 » relu 0


def _u(cle: str) -> str:
    u = CHAMPS[cle][1]
    return f" {u}" if u else ""


def _exiger(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def normaliser(c: Dict) -> Dict:
    """Une copie complétée (une clé absente ou None prend le défaut ; a, R_ELU, uV restent None = auto) et
    contrôlée : une clé inconnue, un type faux, un choix inconnu, un nombre hors du domaine, un profil absent du
    catalogue, une charge ou un ancrage hors de la partie droite d'une aile lèvent ValueError avec un message.
    Une cornière v1 (modèle lame) est d'abord relue par ``depuis_v1`` (clé ``v1`` à vrai)."""
    if not isinstance(c, dict):
        raise ValueError("une cornière est un objet JSON")
    if est_v1(c):
        c = depuis_v1(c)
    inconnues = sorted(set(c) - set(CLES))
    if inconnues:
        raise ValueError(f"clé(s) inconnue(s) : {', '.join(inconnues)}")
    d = nouvelle_corniere()
    out: Dict = {}
    for k in CLES:
        v = c.get(k)
        if v is None:
            v = d[k]
        if v is None and k in OPTIONNELS:
            out[k] = None                            # vide = auto (a = L, R_ELU = 1,5·R, uV = e/2)
            continue
        if k == "nom":
            _exiger(isinstance(v, str), "nom : un texte est attendu")
            _exiger(len(v) <= NOM_MAX, f"nom : {NOM_MAX} caractères au plus")
        elif k in ("standard", "v1"):
            _exiger(isinstance(v, bool), f"{k} : vrai ou faux attendu")
        elif k == "profil":
            _exiger(isinstance(v, str), "profil : un nom de cornière est attendu")
            if c.get("standard", d["standard"]) is True:
                section_catalogue(v)
        elif k in CHOIX:
            _exiger(v in CHOIX[k][1], f"{k} : « {v} » n'est pas un choix admis ({' ou '.join(CHOIX[k][1])})")
        elif k in ENTIERS:
            _exiger(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and float(v).is_integer(),
                    f"{CHAMPS[k][2]} : un entier est attendu, pas {v!r}")
            v = int(v)
        else:
            v = _nombre(k, v)
        out[k] = v
    # --- la section : une cornière, avec une partie droite sur chaque aile
    A, B, t, r = out["A"], out["B"], out["t"], out["r"]
    _exiger(A > 0 and B > 0 and t > 0, "les ailes A, B et l'épaisseur t sont strictement positives")
    _exiger(r >= 0, f"congé r = {_g(r)} mm : il ne peut pas être négatif")
    aile, nom_aile = min((A, "A"), (B, "B"))
    _exiger(t < aile, f"épaisseur t = {_g(t)} mm ≥ aile {nom_aile} = {_g(aile)} mm : ce n'est pas une cornière "
                      "(les ailes doivent être plus larges que l'épaisseur)")
    _exiger(t + r < aile, f"congé r = {_g(r)} mm : e + r = {_g(t + r)} mm atteint l'aile {nom_aile} = {_g(aile)} mm, "
                          "il ne reste pas de partie droite")
    sec = section(out)
    e, r, larg, haut = sec["e"], sec["r"], sec["larg"], sec["haut"]
    # --- matériau
    _exiger(out["fy"] in FY_ADMIS, f"f_y = {_g(out['fy'])} N/mm² : pas une nuance du site "
                                   f"({', '.join(_g(x) for x in FY_ADMIS)})")
    _exiger(out["coef"] >= 1.0, f"coefficient sur f_y = {_g(out['coef'])} : au moins 1")
    # --- charge
    _exiger(out["R"] >= 0, f"réaction R = {_g(out['R'])} kN : elle ne peut pas être négative")
    _exiger(out["L"] > 0, f"longueur reprise L = {_g(out['L'])} mm : strictement positive")
    _exiger(out["dx"] >= 0, f"jeu de la charge dx = {_g(out['dx'])} mm : dx ≥ 0 (il s'applique dans les deux sens)")
    X, dx = out["X"], out["dx"]
    _exiger(X - dx > e + r, f"la charge tombe dans l'angle : X − dx = {_g(X - dx)} mm ≤ e + r = {_g(e + r)} mm "
                            "(elle doit porter sur la partie droite de l'aile horizontale)")
    _exiger(X + dx <= larg, f"la charge tombe au-delà de l'aile : X + dx = {_g(X + dx)} mm > larg = {_g(larg)} mm")
    _exiger(out["a"] is None or out["a"] > 0, f"largeur d'appui a = {_g(out['a'] or 0)} mm : strictement positive (ou vide = L)")
    _exiger(out["R_ELU"] is None or out["R_ELU"] >= 0, f"R_ELU = {_g(out['R_ELU'] or 0)} kN : elle ne peut pas être négative")
    # --- ancrage
    _exiger(out["n"] >= 1, f"nombre d'ancrages n = {out['n']} : au moins 1")
    _exiger(out["d"] > 0, f"diamètre d'ancrage d = {_g(out['d'])} mm : strictement positif")
    _exiger(out["dy"] >= 0, f"tolérance de l'ancrage dy = {_g(out['dy'])} mm : dy ≥ 0 (elle s'applique dans les deux sens)")
    trou = d0(out)
    _exiger(out["n"] * trou < out["L"], f"{out['n']} trous Ø{_g(trou)} sur L = {_g(out['L'])} mm : il ne reste pas de section nette")
    _exiger(out["entraxe"] is None or out["entraxe"] > 0, f"entraxe s = {_g(out['entraxe'] or 0)} mm : strictement positif (ou vide = L/n)")
    s_ancr = entraxe_effectif(out)
    if out["n"] >= 2:
        _exiger(s_ancr >= ENTRAXE_MIN * trou, f"entraxe s = {_g(s_ancr)} mm < 2,2·d0 = {_g(ENTRAXE_MIN * trou)} mm "
                                             "(EN 1993-1-8 Tableau 3.3, p₁ minimal)")
    bout = distance_bout(out)
    _exiger(bout >= BOUT_MIN * trou, f"distance de l'ancrage au bout de la cornière e₁ = {_g(bout)} mm < 1,2·d0 = {_g(BOUT_MIN * trou)} mm "
                                     f"(EN 1993-1-8 Tableau 3.3 ; {out['n']} ancrages à s = {_g(s_ancr)} mm sur L = {_g(out['L'])} mm)")
    _exiger(out["profondeur"] > 0, f"profondeur d'ancrage = {_g(out['profondeur'])} mm : strictement positive")
    _exiger(out["rondelles"] in RONDELLES_ADMISES, f"rondelles = {out['rondelles']} : 0, 1 ou 2")
    Y, dy = out["Y"], out["dy"]
    _exiger(Y - dy - trou / 2 >= e + r, f"l'ancrage entre dans le congé de l'aile verticale : Y − dy − d0/2 = "
                                        f"{_g(Y - dy - trou / 2)} mm < e + r = {_g(e + r)} mm")
    _exiger(Y + dy + trou / 2 <= haut, f"l'ancrage sort de l'aile verticale : Y + dy + d0/2 = {_g(Y + dy + trou / 2)} mm "
                                       f"> haut = {_g(haut)} mm")
    # --- hypothèses
    _exiger(0 <= out["jeu_ancrage"] < Y, f"jeu d'ancrage = {_g(out['jeu_ancrage'])} mm : de 0 à moins de Y = {_g(Y)} mm "
                                         "(le bras Y − jeu de la traction doit rester positif)")
    bras_min = (Y - dy) if out["configuration"] == "A" else (haut - Y - dy)
    _exiger(0 <= out["yc"] < bras_min, f"pivot y_c = {_g(out['yc'])} mm : de 0 à moins de {_g(bras_min)} mm "
                                       "(la résultante de compression doit rester du côté opposé à l'ancrage, bras positif)")
    _exiger(out["uV"] is None or 0 <= out["uV"] <= e, f"ligne d'action u_V = {_g(out['uV'] or 0)} mm : de 0 (nu du mur) à "
                                                      f"e = {_g(e)} mm (dans l'épaisseur de l'aile verticale)")
    # --- le support : une poutre porte les trous d'ancrage et, en B, le contact du bout libre (§ 9.3)
    if out["support"] == POUTRE:
        hs, bs = out["support_haut"], out["support_bas"]
        _exiger(hs + bs > 0, f"poutre : hauteur = {_g(hs)} + {_g(bs)} = {_g(hs + bs)} mm, elle doit être positive")
        if out["configuration"] == "A":
            _exiger(Y + dy + trou / 2 <= hs, f"l'ancrage sort de la poutre par le haut : Y + dy + d0/2 = {_g(Y + dy + trou / 2)} mm "
                                             f"> {_g(hs)} mm au-dessus de la face extérieure de l'aile horizontale")
            _exiger(Y - dy - trou / 2 >= -bs, "l'ancrage sort de la poutre par le bas")
        else:
            _exiger(Y + dy + trou / 2 <= bs, f"l'ancrage sort de la poutre par le bas : Y + dy + d0/2 = {_g(Y + dy + trou / 2)} mm "
                                             f"> {_g(bs)} mm au-dessous de la face extérieure de l'aile horizontale")
            _exiger(Y - dy - trou / 2 >= -hs, "l'ancrage sort de la poutre par le haut")
            _exiger(haut - out["yc"] <= bs, f"le bout libre de l'aile verticale dépasse la poutre : haut − c = {_g(haut - out['yc'])} mm "
                                            f"> {_g(bs)} mm, il n'y a pas de contact (configuration B)")
    return out


# ------------------------------------------------------------------ les positions des ancrages (§ 9.2)
def entraxe_effectif(c: Dict) -> float:
    """L'entraxe des ancrages : celui saisi, sinon L/n."""
    return c["entraxe"] if c["entraxe"] is not None else c["L"] / c["n"]


def distance_bout(c: Dict) -> float:
    """La distance e₁ de l'axe du premier ancrage au bout de la cornière : (L − (n − 1)·s)/2 (les ancrages centrés sur L)."""
    return (c["L"] - (c["n"] - 1) * entraxe_effectif(c)) / 2.0


def positions_ancrages(c: Dict) -> List[float]:
    """Les abscisses z_i des n ancrages depuis le bout de la cornière, centrées sur L : e₁ + i·s."""
    s, e1 = entraxe_effectif(c), distance_bout(c)
    return [e1 + i * s for i in range(c["n"])]

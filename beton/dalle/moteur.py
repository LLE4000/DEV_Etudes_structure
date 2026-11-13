# -*- coding: utf-8 -*-
"""Dalle en béton armé — le MOTEUR, sans Streamlit : bibliothèque standard seule (math, json, os), Python 3.12 et 3.13
(donc Pyodide 0.29). Extrait de l'écran Streamlit ``modules/dalle.py`` v2.2 (dalle BIDIRECTIONNELLE), qui reste la
RÉFÉRENCE : ``tests/test_dalle_moteur.py`` prouve, cas par cas, que ce moteur rend les mêmes grandeurs, les mêmes états
et les mêmes pourcentages que son écran. Méthode, provenance de chaque grandeur, preuve de parité :
``docs/beton/DALLE_MOTEUR.md``.

API
---
    classes_beton() -> dict                       les classes de béton (beton_classes.json)
    nouvelle_dalle() / nouvelle_section(nom) / nouvelle_face() / nouvelle_couche(base)
    nom_section_suivant(dalle) -> str             la première lettre libre (A, B, …, Z, AA…)
    normaliser(dalle) -> dict                     copie complétée et contrôlée (ValueError sinon)
    verifier(dalle) -> dict                       le calcul de toutes les sections
    candidats_base(dalle, i_section, face, familles=None) -> list   le sélecteur de la nappe de base d'une face

ENTRÉE — un dict JSON. Unités : cm (b, h, enrobage, distances), kN·m (moments par bande), kN (V par bande), mm (Ø,
espacements). Une clé absente ou ``null`` prend la valeur par défaut de l'écran ; une clé inconnue ou une valeur hors du
domaine de l'écran lève ValueError.
------
    {"nom": "Dalle 1",
     "b": 100, "h": 20,                  largeur de la bande (entier, 20 à 500 cm) et épaisseur (entier, 5 à 100 cm)
     "beton": "C30/37",                  une classe de beton_classes.json
     "fyk": 500,                         nuance : 400 ou 500 (N/mm²)
     "enrobage": 3.0,                    enrobage béton (0 à 20 cm) ; 0 est lu 3,0 cm, comme l'écran
     "dir_principale": "Y",              « X » ou « Y » : l'ordre d'affichage (aucune formule ne la lit)
     "gamma_s": 1.5,                     coefficient acier ELS (1 à 2) : fyd = fyk / γs
     "jeu_premier_lit": 0.0,             jeu ajouté à la distance d'axe automatique (cm, ≥ 0)
     "sections": [                       une section au moins
       {"nom": "A",
        "Mx_inf": 0.0, "Mx_sup": 0.0,    moments de la direction X qui tendent la face inférieure / supérieure
        "My_inf": 0.0, "My_sup": 0.0,    (kN·m par bande, ≥ 0)
        "V": 0.0,                        effort tranchant maximal de la bande (kN, ≥ 0)
        "faces": {"inf_x" | "sup_x" | "inf_y" | "sup_y":
                    {"couches": [        1 à 4 couches, la base puis les renforts
                       {"type": "treillis" | "barres" | "n_barres",   (libellés de l'écran admis)
                        "treillis": "10/10/100/100",   désignation Øl/Øt/el/et (entiers)
                        "d": 12, "esp": 150, "n": 3,   Ø (6 8 10 12 16 20 25 32 40), espacement (25 à 500 mm),
                                                       nombre de barres dans la bande (1 à 50)
                        "dist": null}],                axe de la couche IMPOSÉ (cm, > 0) ; null = automatique
                     "yG": null}}}]}     centre de gravité IMPOSÉ (cm, > 0) ; null = calculé

    Comme l'écran, une couche garde ses quatre réglages (treillis, Ø, espacement, nombre) quel que soit son type :
    seul le type dit lequel compte.

SORTIE de verifier — JSON-sérialisable, nombres non arrondis (le site formate)
------
    {"reference": "modules/dalle.py v2.2", "dalle": {...} (l'entrée normalisée), "materiaux": {...},
     "sections": [<section>, ...], "taux_max", "statut": "ok" | "nok", "sollicitee" (toutes ses sections le sont),
     "gouvernant": {"section" (indice), "nom", "cle", "titre", "taux"}, "avertissements": [{"code", "message"}, ...]}

  <section> = {"nom", "titre" (« Section A »), "Mx_inf", "Mx_sup", "My_inf", "My_sup", "V",
               "principale" ("x" | "y"), "statut" (l'etat_global de l'écran), "taux_max", "gouvernant" (une cle),
               "sollicitee" (la section porte au moins un effort ; sans aucun, ses vérifications passent toutes et ne
               disent rien : le site l'écrit « Données incomplètes », jamais « Conforme » — comme le moteur des poutres),
               "grandeurs": {...} (ci-dessous), "faces": {<face>: <geometrie de la face>},
               "verifications": [<verification>, ...]}   dans l'ordre de l'écran : hauteur, inf. principale,
                                                         inf. secondaire, sup. principale, sup. secondaire, puis tau
                                                         si V > 0

  <verification> = {"cle": "hauteur" | "inf_x" | "inf_y" | "sup_x" | "sup_y" | "tau",
                    "titre" (le titre du bloc de l'écran), "valeur", "limite", "unite",
                    "taux": valeur / limite (le pourcentage du bandeau divisé par 100), "etat": "ok" | "nok",
                    "motifs": [...] (ce qui rend l'état non conforme), "formules": [{"cle", "termes":
                    [{"symbole", "valeur", "unite"}, ...], "resultat", "unite"}, ...]}
     hauteur : valeur h_min, limite h (cm) ; + "face" (la famille la plus exigeante, ou null sans moment)
               formules hu_min = √(M·10⁶ / (α_b·b·μ)) / 10 ; h_min = hu_min + y_G      motif « hauteur_insuffisante »
     <face>  : valeur besoin = max(As_req ; As_min), limite As fourni sur la bande (mm²) ; + "face", "direction",
               "principale" (bool), "As_max", "libelle"
               formules d = h − y_G ; [y_G = Σ As·e / Σ As] ; As = Σ As/m × b/100 ; As_req = M·10⁶ / (fyd·0,9·d·10) ;
               As_min_ec = 0,26·fctm/fyk·b·h ; As_min_plancher = 0,0013·b·h ; As_min_opposee = 0,25·As_req de la
               face opposée de la MÊME direction ; As_min ; besoin ; As_max = 0,04·b·h
               motifs « d_utile_negatif », « acier_insuffisant », « acier_excessif » (As > As_max)
     tau     : valeur τ, limite τ_adm,I (N/mm²) ; une dalle ne reçoit pas d'étriers : τ ≤ τ_adm,I seulement
               formules τ = V·10³ / (0,75·b·h) ; τ_adm,I = 0,016·fck,cube / 1,05      motif « cisaillement_excessif »

  <section>["grandeurs"] (cm, mm², mm²/m, N/mm², kN, kN·m) :
     matériaux  beton fck fck_cube fctm alpha_b mu fyk gamma_s fyd tau_1 tau_2 tau_4
     section    b h enrobage enrobage_calcul jeu_premier_lit arrondi Mx_inf Mx_sup My_inf My_sup V principale
     par face   As_pm_<f> As_<f> yG_<f> yG_calcule_<f> yG_impose_<f> d_<f> d_calc_<f> geometrie_<f>_ok M_<f>
                As_req_<f> As_min_opposee_<f> As_min_<f> As_min_terme_<f> besoin_<f> etat_<f>   (f = inf_x sup_x …)
     communes   As_min_ec As_min_plancher As_min_base As_max
     hauteur    face_hauteur M_hauteur yG_hauteur hu_min h_min etat_hauteur
     tranchant  tau tau_adm etat_tau ; etat_inf etat_sup (fusion des deux directions, les bandeaux de l'écran)

  <geometrie de la face> = {"couches": [{"type", "libelle" (« Treillis 10/10/100/100 », « Ø12/150 », « 3 Ø12 »),
       "valeur" (« 10/10/100/100 »…), "d" (Ø mm), "esp" (mm : el du treillis, espacement des barres), "n" (« n barres »,
       sinon 0), "treillis" (Øl, Øt, el, et) ou null, "As_pm" (mm²/m), "e" (axe, cm), "auto" (bool), "e_auto" (cm)}, …],
       "As_pm", "As", "yG", "yG_calcule", "yG_impose", "libelle" (« Treillis 10/10/100/100 + 3 Ø12 »)}

SORTIE de candidats_base — la nappe de BASE (couche 1) d'une face remplacée par chaque treillis standard et chaque
disposition de barres (Ø 8 à 25, espacements 100 à 250 mm) ; renforts, axes imposés et y_G imposé gardés ; la section est
recalculée (le Ø déplace l'axe automatique, donc d). Triées par section au mètre :
    [{"nom" (« Treillis 8/8/150/150 », « Ø12/150 »), "famille" ("treillis" | "barres"), "type", "treillis", "d", "esp",
      "As_pm" (de la couche), "As" (de la face), "taux", "etat", "ok", "statut_section", "retenue", "optimal"}, ...]
"""
import copy
import math

from beton import position as P
from beton import treillis as TR

from . import entree as E
from .entree import (classes_beton, libelle_section, nom_section_suivant, normaliser, nouvelle_couche,  # noqa: F401
                     nouvelle_dalle, nouvelle_face, nouvelle_section)

REFERENCE = "modules/dalle.py v2.2"
OK, NOK = "ok", "nok"

# ---- les coefficients de la méthode du bureau (_dimensionnement_compute_states) ----
K_FCTM = 0.30              # fctm = 0,30·fck^(2/3)
K_BRAS_DE_LEVIER = 0.9     # As,req = M / (fyd·0,9·d)
K_AS_MIN_EC = 0.26         # 0,26·fctm/fyk·b·h      (h et non d, comme la poutre)
K_AS_MIN_PLANCHER = 0.0013 # 0,0013·b·h
K_AS_MIN_OPPOSEE = 0.25    # 0,25·As,req de la face opposée de la MÊME direction
K_AS_MAX = 0.04            # 0,04·b·h
K_TAU_SECTION = 0.75       # τ = V / (0,75·b·h)
K_TAU = {"I": 0.016, "II": 0.032, "IV": 0.064}   # τ_adm = k·fck,cube / 1,05 ; la dalle n'emploie que la bande I
GAMMA_TAU = 1.05
D_CALCUL_MIN = 0.1         # cm : les formules prennent d = max(d, 0,1) ; d ≤ 0 rend la face « nok »
ENROBAGE_SI_NUL = 3.0      # cm : « float(enrobage or 3.0) » dans _auto_dist_couche
DIAMETRE_SI_ILLISIBLE = 10.0   # mm : Ø d'un treillis illisible (inatteignable : normaliser le refuse)

TITRES_FACES = {"inf": "Armatures inférieures", "sup": "Armatures supérieures"}
TITRE_HAUTEUR = "Vérification de la hauteur"
TITRE_TAU = "Vérification de l'effort tranchant"

# le sélecteur de la nappe de base : les treillis de la base, et des barres courantes
CANDIDATS_D = (8, 10, 12, 16, 20, 25)
CANDIDATS_ESP = (100, 125, 150, 200, 250)
FAMILLES = ("treillis", "barres")


def fusion_statuts(*etats: str) -> str:
    """« nok » si un état l'est, sinon « warn » si un l'est, sinon « ok » — _status_merge."""
    if any(e == NOK for e in etats):
        return NOK
    if any(e == "warn" for e in etats):
        return "warn"
    return OK


def arrondi_demi_cm(x_cm: float) -> float:
    """Arrondi au 0,5 cm SUPÉRIEUR — _round_up_to_half_cm."""
    return math.ceil(float(x_cm) * 2.0) / 2.0


def direction_secondaire(p: str) -> str:
    return "y" if p == "x" else "x"


def titre_face(face: str, principale: str) -> str:
    """Le titre du bloc de l'écran : « Armatures inférieures — dir. Y (principale) »."""
    cote, dk = face.split("_")
    return f"{TITRES_FACES[cote]} — dir. {dk.upper()}" + (" (principale)" if dk == principale else " (secondaire)")


def ordre_faces(principale: str) -> tuple:
    """L'ordre de l'écran et de la note (audit I-3) : par face, la principale puis la secondaire."""
    s = direction_secondaire(principale)
    return (f"inf_{principale}", f"inf_{s}", f"sup_{principale}", f"sup_{s}")


# ============================================================ matériaux
def materiaux(d: dict) -> dict:
    """Les matériaux d'une dalle NORMALISÉE — lectures de _dimensionnement_compute_states."""
    bd = E._classes_lues()[d["beton"]]
    fck_cube = bd["fck_cube"]
    alpha_b = bd["alpha_b"]
    fyk = float(int(d["fyk"]))
    gamma_s = float(d["gamma_s"])
    fyd = fyk / gamma_s
    cle_mu = f"mu_a{int(d['fyk'])}"
    if cle_mu not in bd:  # repli de l'écran (inatteignable avec beton_classes.json)
        cle_mu = "mu_a500" if "mu_a500" in bd else next(k for k in bd if k.startswith("mu_a"))
    fck = float(bd.get("fck", 0.8 * fck_cube) or (0.8 * fck_cube))
    return {"beton": d["beton"], "fck": fck, "fck_cube": float(fck_cube), "alpha_b": alpha_b, "mu": bd[cle_mu],
            "fctm": K_FCTM * (fck ** (2.0 / 3.0)) if fck > 0 else 0.0, "fyk": fyk, "gamma_s": gamma_s, "fyd": fyd,
            "tau_1": K_TAU["I"] * fck_cube / GAMMA_TAU, "tau_2": K_TAU["II"] * fck_cube / GAMMA_TAU,
            "tau_4": K_TAU["IV"] * fck_cube / GAMMA_TAU}


# ============================================================ les couches d'une face
def couche_as_par_m(c: dict, b: float) -> float:
    """Section d'acier de la couche (mm²/m) — _couche_as_per_m. « n barres » : n barres dans la bande de largeur b,
    n·(π·Ø·Ø/4) / (b/100)."""
    if c["type"] == "treillis":
        return TR.as_treillis_mm2_m(c["treillis"])
    if c["type"] == "n_barres":
        bb = float(b or 100)
        return c["n"] * (math.pi * c["d"] * c["d"] / 4.0) / max(0.01, bb / 100.0)
    return TR.as_barres_mm2_m(c["d"], float(c["esp"]))


def couche_diametre(c: dict) -> float:
    """Ø des fils (porteurs) ou des barres de la couche (mm) — _couche_diam_mm."""
    if c["type"] == "treillis":
        t = TR.parse_designation(c["treillis"])
        return float(t[0]) if t else DIAMETRE_SI_ILLISIBLE
    return float(c["d"])


def couche_libelle(c: dict) -> str:
    """« Treillis 10/10/100/100 », « Ø12/150 » ou « 3 Ø12 » — _couche_label."""
    if c["type"] == "treillis":
        return f"Treillis {c['treillis']}"
    return couche_valeur(c)


def couche_valeur(c: dict) -> str:
    """La valeur seule (légende de la coupe) : « 10/10/100/100 », « Ø12/150 » ou « 3 Ø12 » — _couche_valeur."""
    if c["type"] == "treillis":
        return c["treillis"]
    if c["type"] == "n_barres":
        return f"{c['n']} Ø{c['d']}"
    esp = float(c["esp"])
    esp_txt = f"{esp:.0f}" if abs(esp - round(esp)) < 1e-9 else f"{esp:.1f}".replace(".", ",")
    return f"Ø{c['d']}/{esp_txt}"


def position_auto(c: dict, enrobage: float) -> float:
    """La position RÉELLE automatique de la couche (cm) : au contact de l'enrobage, c + Ø/2 (beton/position.py) ; un
    enrobage nul est lu 3,0 cm (« or 3.0 », la lecture de l'écran Streamlit)."""
    return P.sur_parement(float(enrobage or ENROBAGE_SI_NUL), couche_diametre(c))


def distance_auto(c: dict, enrobage: float, jeu: float, arrondi: float = P.ARRONDI) -> float:
    """L'axe de CALCUL automatique de la couche (cm) : ⌈c + Ø/2 + jeu de pose⌉ au pas d'arrondi (beton/position.py,
    la règle du 13/11/2026 — c'était c + ⌈Ø/2⌉ + jeu, l'arrondi sur le seul demi-Ø)."""
    return P.axe_calcul(position_auto(c, enrobage), jeu, arrondi)


def detail_auto(c: dict, enrobage: float, jeu: float, arrondi: float = P.ARRONDI) -> dict:
    """Le calcul de l'axe automatique terme à terme (``P.detail``) : c, Ø/2, puis le jeu et l'arrondi."""
    return P.detail([("c", float(enrobage or ENROBAGE_SI_NUL)), ("Ø/2", couche_diametre(c) / 20.0)], jeu, arrondi)


def _face(couches: list, yG, b: float, enrobage: float, jeu: float, arrondi: float = P.ARRONDI) -> dict:
    """Une face d'une direction : As au mètre (Σ des couches), As sur la bande, y_G = Σ As·e / Σ As (ou la valeur
    imposée). Chaque couche a sa position RÉELLE (``e_reel`` : l'axe saisi « dist », sinon au contact de l'enrobage) et
    son axe de CALCUL ``e`` = ⌈réel + jeu⌉ au pas d'arrondi (beton/position.py) ; y_G et d lisent ``e``."""
    As_pm_tot = 0.0
    somme = 0.0
    lignes = []
    for c in couches:
        As_pm = couche_as_par_m(c, b)
        reel_auto = position_auto(c, enrobage)
        reel = reel_auto if c["dist"] is None else float(c["dist"])
        e = P.axe_calcul(reel, jeu, arrondi)
        As_pm_tot += As_pm
        somme += As_pm * e
        t = TR.parse_designation(c["treillis"]) if c["type"] == "treillis" else None
        det = (detail_auto(c, enrobage, jeu, arrondi) if c["dist"] is None
               else P.detail([("posé", reel)], jeu, arrondi))
        lignes.append({"type": c["type"], "libelle": couche_libelle(c), "valeur": couche_valeur(c), "d": couche_diametre(c),
                       "esp": float(t[2]) if t else float(c["esp"]), "n": c["n"] if c["type"] == "n_barres" else 0,
                       "treillis": list(t) if t else None, "As_pm": As_pm, "e": e, "e_reel": reel, "auto": c["dist"] is None,
                       "e_auto": P.axe_calcul(reel_auto, jeu, arrondi), "reel_auto": reel_auto, "detail": det})
    yG_calcule = (somme / As_pm_tot) if As_pm_tot > 0 else lignes[0]["e"]
    return {"couches": lignes, "As_pm": As_pm_tot, "As": As_pm_tot * (float(b) / 100.0), "yG_calcule": yG_calcule,
            "yG": yG_calcule if yG is None else yG, "yG_impose": yG is not None,
            "libelle": " + ".join(x["libelle"] for x in lignes)}


# ============================================================ le calcul d'une section
def _hu_min(m: float, alpha_b: float, b: float, mu: float) -> float:
    """Hauteur utile minimale (cm) : √(M·10⁶ / (α_b·b·10·μ)) / 10, M en kN·m, b en cm ; 0 sans moment."""
    return math.sqrt((m * 1e6) / (alpha_b * b * 10 * mu)) / 10 if m > 0 else 0.0


def _calcul(d: dict, s: dict, mat: dict) -> dict:
    """Toutes les grandeurs d'une section et ses états — _dimensionnement_compute_states, dans le même ordre
    d'opérations (les flottants sont ceux de l'écran au bit près)."""
    b, h = float(d["b"]), float(d["h"])
    fyk, fyd, alpha_b, mu, fctm = mat["fyk"], mat["fyd"], mat["alpha_b"], mat["mu"], mat["fctm"]
    As_min_ec = K_AS_MIN_EC * fctm / fyk * b * h * 1e2          # mm² (b, h en cm → ·1e2)
    As_min_plancher = K_AS_MIN_PLANCHER * b * h * 1e2
    As_min_base = max(As_min_ec, As_min_plancher)
    As_max = K_AS_MAX * b * h * 1e2
    principale = d["dir_principale"].lower()
    g = {"beton": mat["beton"], "fck": mat["fck"], "fck_cube": mat["fck_cube"], "fctm": fctm, "alpha_b": alpha_b,
         "mu": mu, "fyk": fyk, "gamma_s": mat["gamma_s"], "fyd": fyd, "tau_1": mat["tau_1"], "tau_2": mat["tau_2"],
         "tau_4": mat["tau_4"], "b": b, "h": h, "enrobage": d["enrobage"],
         "enrobage_calcul": float(d["enrobage"] or ENROBAGE_SI_NUL), "jeu_premier_lit": d["jeu_premier_lit"],
         "arrondi": d["arrondi"],
         "Mx_inf": s["Mx_inf"], "Mx_sup": s["Mx_sup"], "My_inf": s["My_inf"], "My_sup": s["My_sup"], "V": s["V"],
         "principale": principale, "As_min_ec": As_min_ec, "As_min_plancher": As_min_plancher,
         "As_min_base": As_min_base, "As_max": As_max}
    faces = {}
    for f in E.FACES:
        faces[f] = _face(s["faces"][f]["couches"], s["faces"][f]["yG"], d["b"], d["enrobage"], d["jeu_premier_lit"],
                         d["arrondi"])
    # une passe par direction : la face opposée du critère 0,25·As,req est celle de la MÊME direction
    for dk in E.DIRECTIONS:
        fi, fs = f"inf_{dk}", f"sup_{dk}"
        req = {}
        for f in (fi, fs):
            geo = faces[f]
            d_u = h - geo["yG"]
            g.update({f"As_pm_{f}": geo["As_pm"], f"As_{f}": geo["As"], f"yG_{f}": geo["yG"],
                      f"yG_calcule_{f}": geo["yG_calcule"], f"yG_impose_{f}": geo["yG_impose"], f"d_{f}": d_u,
                      f"geometrie_{f}_ok": d_u > 0.0, f"d_calc_{f}": max(d_u, D_CALCUL_MIN)})
            m = float(s["M" + dk + "_" + f.split("_")[0]] or 0.0)
            g[f"M_{f}"] = m
            req[f] = (m * 1e6) / (fyd * K_BRAS_DE_LEVIER * g[f"d_calc_{f}"] * 10) if m > 0 else 0.0
            g[f"As_req_{f}"] = req[f]
        for f, opp in ((fi, fs), (fs, fi)):
            opposee = K_AS_MIN_OPPOSEE * req[opp]
            As_min = max(As_min_base, opposee)
            besoin = max(req[f], As_min)
            As = g[f"As_{f}"]
            g[f"As_min_opposee_{f}"] = opposee
            g[f"As_min_{f}"] = As_min
            # le terme qui gouverne As,min (max garde le premier à égalité : ec, puis plancher, puis opposée)
            g[f"As_min_terme_{f}"] = ("opposee" if opposee > As_min_base else "ec" if As_min_ec >= As_min_plancher else "plancher")
            g[f"besoin_{f}"] = besoin
            g[f"etat_{f}"] = OK if (g[f"geometrie_{f}_ok"] and As >= max(req[f], As_min) and As <= As_max) else NOK

    # hauteur : CHAQUE famille demande hu,min(M) + son y_G ; la PLUS EXIGEANTE est retenue (à égalité, l'ordre
    # inf_x, sup_x, inf_y, sup_y tranche : max garde le premier) ; sans moment, h_min = y_G inférieur X
    familles = [(f, g[f"M_{f}"], g[f"yG_{f}"]) for f in E.FACES]
    actives = [x for x in familles if x[1] > 0]
    if actives:
        face_h, M_h, yG_h = max(actives, key=lambda x: _hu_min(x[1], alpha_b, b, mu) + x[2])
    else:
        face_h, M_h, yG_h = None, 0.0, familles[0][2]
    hu = _hu_min(M_h, alpha_b, b, mu)
    h_min = hu + yG_h
    g.update({"face_hauteur": face_h, "M_hauteur": M_h, "yG_hauteur": yG_h, "hu_min": hu, "h_min": h_min,
              "etat_hauteur": OK if h_min <= h else NOK})
    g["etat_inf"] = fusion_statuts(g["etat_inf_x"], g["etat_inf_y"])
    g["etat_sup"] = fusion_statuts(g["etat_sup_x"], g["etat_sup_y"])

    # effort tranchant (v2.1) : pas d'étriers dans une dalle — τ = V / (0,75·b·h) ≤ τ_adm,I seulement
    V = float(s["V"] or 0.0)
    if V > 0:
        tau = V * 1e3 / (K_TAU_SECTION * b * h * 100)
        etat_tau = OK if tau <= mat["tau_1"] else NOK
    else:
        tau, etat_tau = 0.0, OK
    g.update({"tau": tau, "tau_adm": mat["tau_1"], "etat_tau": etat_tau})
    statut = fusion_statuts(g["etat_hauteur"], g["etat_inf"], g["etat_sup"], etat_tau)
    return {"g": g, "faces": faces, "statut": statut}


# ============================================================ les vérifications (blocs de l'écran)
def _t(symbole: str, valeur, unite: str = "") -> dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def _f(cle: str, termes: list, resultat, unite: str) -> dict:
    return {"cle": cle, "termes": termes, "resultat": resultat, "unite": unite}


def _verif(cle: str, titre: str, valeur: float, limite: float, unite: str, etat: str, motifs: list, formules: list,
           **plus) -> dict:
    v = {"cle": cle, "titre": titre, "valeur": valeur, "limite": limite, "unite": unite, "taux": valeur / limite,
         "etat": etat, "motifs": motifs, "formules": formules}
    v.update(plus)
    return v


def _verifications(c: dict, mat: dict) -> list:
    g, faces = c["g"], c["faces"]
    b_mm, h_mm = g["b"] * 10.0, g["h"] * 10.0
    out = [_verif("hauteur", TITRE_HAUTEUR, g["h_min"], g["h"], "cm", g["etat_hauteur"],
                  [] if g["etat_hauteur"] == OK else ["hauteur_insuffisante"],
                  [_f("hu_min", [_t("M", g["M_hauteur"], "kN·m"), _t("alpha_b", g["alpha_b"], "N/mm²"), _t("b", b_mm, "mm"),
                                 _t("mu", g["mu"])], g["hu_min"], "cm"),
                   _f("h_min", [_t("hu_min", g["hu_min"], "cm"), _t("yG", g["yG_hauteur"], "cm")], g["h_min"], "cm")],
                  face=g["face_hauteur"])]
    for f in ordre_faces(g["principale"]):
        cote, dk = f.split("_")
        opp = ("sup_" if cote == "inf" else "inf_") + dk
        geo = faces[f]
        As, besoin = g[f"As_{f}"], g[f"besoin_{f}"]
        motifs = ([] if g[f"geometrie_{f}_ok"] else ["d_utile_negatif"]) \
            + (["acier_insuffisant"] if As < besoin else []) + (["acier_excessif"] if As > g["As_max"] else [])
        formules = [_f("d", [_t("h", g["h"], "cm"), _t("yG", g[f"yG_{f}"], "cm")], g[f"d_{f}"], "cm")]
        if len(geo["couches"]) > 1 and not geo["yG_impose"]:
            formules.append(_f("yG", [_t(f"couche {k + 1}", [x["As_pm"], x["e"]], "mm²/m · cm") for k, x in enumerate(geo["couches"])],
                               geo["yG_calcule"], "cm"))
        formules += [
            _f("As", [_t("As_pm", geo["As_pm"], "mm²/m"), _t("b", g["b"], "cm")], As, "mm²"),
            _f("As_req", [_t("M", g[f"M_{f}"], "kN·m"), _t("fyd", g["fyd"], "N/mm²"), _t("k", K_BRAS_DE_LEVIER),
                          _t("d", g[f"d_calc_{f}"], "cm")], g[f"As_req_{f}"], "mm²"),
            _f("As_min_ec", [_t("k", K_AS_MIN_EC), _t("fctm", g["fctm"], "N/mm²"), _t("fyk", g["fyk"], "N/mm²"),
                             _t("b", b_mm, "mm"), _t("h", h_mm, "mm")], g["As_min_ec"], "mm²"),
            _f("As_min_plancher", [_t("k", K_AS_MIN_PLANCHER), _t("b", b_mm, "mm"), _t("h", h_mm, "mm")],
               g["As_min_plancher"], "mm²"),
            _f("As_min_opposee", [_t("k", K_AS_MIN_OPPOSEE), _t("As_req_opp", g[f"As_req_{opp}"], "mm²")],
               g[f"As_min_opposee_{f}"], "mm²"),
            _f("As_min", [_t("As_min_ec", g["As_min_ec"], "mm²"), _t("As_min_plancher", g["As_min_plancher"], "mm²"),
                          _t("As_min_opposee", g[f"As_min_opposee_{f}"], "mm²")], g[f"As_min_{f}"], "mm²"),
            _f("besoin", [_t("As_req", g[f"As_req_{f}"], "mm²"), _t("As_min", g[f"As_min_{f}"], "mm²")], besoin, "mm²"),
            _f("As_max", [_t("k", K_AS_MAX), _t("b", b_mm, "mm"), _t("h", h_mm, "mm")], g["As_max"], "mm²")]
        out.append(_verif(f, titre_face(f, g["principale"]), besoin, As, "mm²", g[f"etat_{f}"], motifs, formules,
                          face=cote, direction=dk, principale=dk == g["principale"], As_max=g["As_max"],
                          As_pm=geo["As_pm"], libelle=geo["libelle"], opposee=opp))
    if g["V"] > 0:
        out.append(_verif("tau", TITRE_TAU, g["tau"], g["tau_adm"], "N/mm²", g["etat_tau"],
                          [] if g["etat_tau"] == OK else ["cisaillement_excessif"],
                          [_f("tau", [_t("V", g["V"], "kN"), _t("k", K_TAU_SECTION), _t("b", b_mm, "mm"), _t("h", h_mm, "mm")],
                              g["tau"], "N/mm²"),
                           _f("tau_lim", [_t("k", K_TAU["I"]), _t("fck_cube", g["fck_cube"], "N/mm²"), _t("gamma", GAMMA_TAU)],
                              g["tau_adm"], "N/mm²")]))
    return out


def _section(d: dict, s: dict, mat: dict) -> dict:
    c = _calcul(d, s, mat)
    verifs = _verifications(c, mat)
    gouv = max(verifs, key=lambda v: v["taux"])
    return {"nom": s["nom"], "titre": libelle_section(s["nom"]), "Mx_inf": s["Mx_inf"], "Mx_sup": s["Mx_sup"],
            "My_inf": s["My_inf"], "My_sup": s["My_sup"], "V": s["V"], "principale": c["g"]["principale"],
            "statut": c["statut"], "taux_max": gouv["taux"], "gouvernant": gouv["cle"], "grandeurs": c["g"],
            "faces": c["faces"], "verifications": verifs,
            "sollicitee": any(s[k] > 0 for k in E.MOMENTS) or s["V"] > 0}


# ============================================================ les avertissements
def _fr(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _avertissements(d: dict, sections: list) -> list:
    out = []
    if not d["enrobage"]:
        out.append({"code": "enrobage_nul",
                    "message": f"Enrobage nul : l'écran place les couches avec un enrobage de {ENROBAGE_SI_NUL:g} cm "
                               "(il lit « enrobage or 3,0 »)."})
    for i, s in enumerate(sections):
        for v in s["verifications"]:
            if "d_utile_negatif" in v["motifs"]:
                out.append({"code": "d_utile_negatif", "section": i, "cle": v["cle"],
                            "message": f"{s['titre']} · {v['titre']} : position des couches incompatible avec l'épaisseur, "
                                       "d utile ≤ 0."})
        # les nappes croisées d'une même face calculées au même niveau (l'écran ne les superpose pas)
        for cote in ("inf", "sup"):
            fx, fy = s["faces"][f"{cote}_x"]["couches"][0], s["faces"][f"{cote}_y"]["couches"][0]
            ecart = abs(fx["e_reel"] - fy["e_reel"]) * 10.0
            mini = (fx["d"] + fy["d"]) / 2.0
            if ecart < mini - 1e-9:
                out.append({"code": "nappes_confondues", "section": i, "face": cote, "e_x": fx["e_reel"], "e_y": fy["e_reel"],
                            "ecart_mini": mini / 10.0,
                            "message": f"{s['titre']} · armatures {'inférieures' if cote == 'inf' else 'supérieures'} X et Y : "
                                       f"posées à {_fr(fx['e_reel'])} et {_fr(fy['e_reel'])} cm, sans tenir compte du "
                                       f"croisement ; posées l'une sur l'autre, leurs axes sont distants d'au moins "
                                       f"{_fr(mini / 10.0)} cm et l'armature du dessus a un d plus petit. "
                                       "Saisir sa position pour en tenir compte."})
    return out


# ============================================================ l'API
def verifier(dalle: dict) -> dict:
    """Calcule chaque section de la dalle (voir le format en tête du fichier). Lève ValueError si la dalle sort du
    domaine de l'écran."""
    d = normaliser(dalle)
    mat = materiaux(d)
    sections = [_section(d, s, mat) for s in d["sections"]]
    i_g, v_g = max(((i, v) for i, sec in enumerate(sections) for v in sec["verifications"]), key=lambda x: x[1]["taux"])
    return {"reference": REFERENCE, "dalle": d, "materiaux": mat, "sections": sections, "taux_max": v_g["taux"],
            "statut": fusion_statuts(*(s["statut"] for s in sections)),
            "sollicitee": all(s["sollicitee"] for s in sections),
            "gouvernant": {"section": i_g, "nom": sections[i_g]["nom"], "cle": v_g["cle"], "titre": v_g["titre"],
                           "taux": v_g["taux"]},
            "avertissements": _avertissements(d, sections)}


def candidat_nom(c: dict) -> str:
    """Le nom d'une nappe de base dans le sélecteur : « Treillis 8/8/150/150 » ou « Ø12/150 »."""
    return f"Treillis {c['treillis']}" if c["type"] == "treillis" else f"Ø{c['d']}/{c['esp']}"


def lire_candidat(nom: str) -> dict:
    """« Treillis 8/8/150/150 » → {type, treillis} ; « Ø12/150 » → {type, d, esp} ; ValueError sinon."""
    if not isinstance(nom, str):
        raise ValueError(f"armature : texte attendu (reçu {nom!r})")
    if nom.startswith("Treillis "):
        des = nom[len("Treillis "):]
        if not TR.est_canonique(des):
            raise ValueError(f"treillis illisible : « {des} »")
        return {"type": "treillis", "treillis": des}
    if nom.startswith("Ø") and "/" in nom:
        a, _, b = nom[1:].partition("/")
        if a.isdigit() and b.isdigit():
            return {"type": "barres", "d": int(a), "esp": int(b)}
    raise ValueError(f"armature illisible : « {nom} » (« Treillis Øl/Øt/el/et » ou « Ø12/150 »)")


def _liste_candidats(familles) -> list:
    out = []
    if "treillis" in familles:
        out += [{"type": "treillis", "treillis": des} for des in TR.liste_choix()]
    if "barres" in familles:
        out += [{"type": "barres", "d": dd, "esp": e} for dd in CANDIDATS_D for e in CANDIDATS_ESP]
    return out


def candidats_base(dalle: dict, i_section: int, face: str, familles=None) -> list:
    """Le sélecteur de la nappe de BASE d'une face : chaque treillis de la base et chaque disposition de barres
    remplace la couche 1 (son axe imposé, s'il l'est, est gardé), les renforts et le y_G imposé restent ; la section est
    recalculée. Triées par section au mètre de la couche ; « optimal » marque la plus petite « ok », « retenue » la
    nappe actuelle."""
    if face not in E.FACES:
        raise ValueError(f"face : {' '.join(E.FACES)} attendu (reçu {face!r})")
    d = normaliser(dalle)
    if not isinstance(i_section, int) or isinstance(i_section, bool) or not 0 <= i_section < len(d["sections"]):
        raise ValueError(f"section n° {i_section!r} introuvable ({len(d['sections'])} section(s))")
    fams = tuple(f for f in (familles or FAMILLES) if f in FAMILLES) or FAMILLES
    mat = materiaux(d)
    s = d["sections"][i_section]
    base = s["faces"][face]["couches"][0]
    out = []
    for cand in _liste_candidats(fams):
        essai = copy.deepcopy(s)
        c1 = essai["faces"][face]["couches"][0]
        c1.update(cand)
        E._couche(c1, "candidat", True)          # le candidat reste dans le domaine de l'écran
        c = _calcul(d, essai, mat)
        g = c["g"]
        As_pm_c = couche_as_par_m(c1, d["b"])
        retenue = base["type"] == cand["type"] and (base["treillis"] == cand["treillis"] if cand["type"] == "treillis"
                                                    else (base["d"], base["esp"]) == (cand["d"], cand["esp"]))
        out.append({"nom": candidat_nom(c1), "famille": cand["type"], "type": cand["type"], "treillis": c1["treillis"],
                    "d": c1["d"], "esp": c1["esp"], "As_pm": As_pm_c, "As": g[f"As_{face}"], "yG": g[f"yG_{face}"],
                    "taux": g[f"besoin_{face}"] / g[f"As_{face}"], "etat": g[f"etat_{face}"], "ok": g[f"etat_{face}"] == OK,
                    "statut_section": c["statut"], "retenue": retenue, "optimal": False})
    out.sort(key=lambda x: (x["As_pm"], FAMILLES.index(x["famille"]), x["d"]))
    optimal = next((x for x in out if x["ok"]), None)
    if optimal is not None:
        optimal["optimal"] = True
    return out

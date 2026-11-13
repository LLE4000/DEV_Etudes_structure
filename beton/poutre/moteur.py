# -*- coding: utf-8 -*-
"""Poutre en béton armé — le MOTEUR, sans Streamlit : bibliothèque standard seule (math, json, os),
Python 3.12 et 3.13 (donc Pyodide 0.29). Extrait du module Streamlit ``modules/poutre.py`` v2.40
(APP_VERSION 1.2.41), qui reste la RÉFÉRENCE : ``tests/test_poutre_moteur.py`` prouve, cas par cas,
que ce moteur rend les mêmes grandeurs, les mêmes états et les mêmes pourcentages que son écran.
Formules, provenance de chaque grandeur et preuve de parité : ``docs/beton/POUTRE_MOTEUR.md``.

API
---
    classes_beton() -> dict                      les classes de béton (beton_classes.json)
    nouvelle_poutre() -> dict                    une poutre aux valeurs par défaut du module
    nouvelle_section(nom="A") -> dict            une section aux valeurs par défaut du module
    nom_section_suivant(poutre) -> str           la première lettre libre (A, B, …, Z, AA…)
    normaliser(poutre) -> dict                   copie complétée et contrôlée (ValueError sinon)
    verifier(poutre) -> dict                     le calcul de toutes les sections
    candidats_armatures(poutre, nom_section, face) -> list   le sélecteur d'armatures d'une face

ENTRÉE — un dict JSON. Unités : cm, kN, kN·m ; Ø en mm. Une clé absente ou ``null`` prend la valeur
par défaut du module ; une clé inconnue ou une valeur hors du domaine de l'écran lève ValueError.
------
    {"nom": "Poutre 1",
     "b": 20.0, "h": 40.0,               largeur et hauteur (cm, > 0)
     "beton": "C30/37",                  une classe de beton_classes.json
     "fyk": 500,                         nuance : 400 ou 500 (N/mm²)
     "gamma_s": 1.5,                     coefficient acier (> 0) : fyd = fyk / γs
     "enrobage": 3.0,                    enrobage béton (cm, ≥ 0) ; 0 est lu 3,0 cm, comme le module
     "jeu_premier_lit": 1.0,             JEU DE POSE (cm, ≥ 0) : ajouté à la position réelle de chaque lit avant
                                         l'arrondi (beton/position.py, règle du 13/11/2026)
     "jeu_entre_lits": 1.0,              jour libre entre deux lits (cm, ≥ 0 ; position réelle)
     "arrondi": 0.5,                     pas d'arrondi de l'axe de calcul (cm : 0, 0,5 ou 1)
     "sections": [                       une section au moins
       {"nom": "A",                      sans nom : la première lettre libre
        "M_inf": 0.0,                    moment qui tend la face INFÉRIEURE (kN·m, ≥ 0 ; travée)
        "M_sup": 0.0,                    moment qui tend la face SUPÉRIEURE (kN·m, ≥ 0 ; appui)
        "V": 0.0,                        effort tranchant (kN, ≥ 0)
        "inf": [{"n": 2, "d": 16}],      lits de la face inférieure, du parement vers le cœur :
        "sup": [{"n": 2, "d": 16}],      1 à 4 lits ; n ≥ 1 barres ; d = Ø parmi 6 8 10 12 16 20 25 32 40
        "etriers": [{"type": "etrier", "d": 10}],
                                         une ligne = un étrier (« etrier », 2 brins) ou une épingle
                                         (« epingle », 1 brin) ; Ø parmi 6 8 10 12 ; une ligne au moins
        "pas": 30.0,                     pas des armatures d'effort tranchant (cm, > 0)
        "yG_inf": null, "yG_sup": null}  centre de gravité des lits IMPOSÉ (cm, > 0) ; null = calculé
     ]}

SORTIE de verifier — JSON-sérialisable, nombres non arrondis (le site formate)
------
    {"reference": "modules/poutre.py v2.40 (APP_VERSION 1.2.41)",
     "poutre": {...},                    l'entrée normalisée
     "materiaux": {...},                 beton fck fck_cube fctm alpha_b mu fyk gamma_s fyd tau_1 tau_2 tau_4
     "sections": [<section>, ...],
     "taux_max": float,                  le plus grand taux de toutes les vérifications
     "statut": "ok" | "warn" | "nok",    fusion des statuts des sections (_status_merge)
     "gouvernant": {"section", "cle", "nom", "taux"},   la vérification au plus grand taux
     "avertissements": [{"code", "message"}, ...],
     "taux_armature": {...}}             SEULEMENT si l'option est allumée (poutre["taux"]["actif"]) : le taux
                                         d'armature simplifié, par section et pour la poutre (beton/poutre/taux.py)

  <section> = {"nom", "M_inf", "M_sup", "V",
               "statut": "ok" | "warn" | "nok",       l'etat_global du module
               "taux_max", "gouvernant": <cle>,       le plus grand taux de la section et sa vérification
               "grandeurs": {...},                    toutes les grandeurs intermédiaires (ci-dessous)
               "verifications": [<verification>, ...], dans l'ordre de l'écran : hauteur, inf, sup,
                                                      puis tau et pas si V > 0 (comme les blocs)
               "geometrie": {...}}                    les lits et les étriers, pour le dessin

  <verification> = {"cle": "hauteur" | "inf" | "sup" | "tau" | "pas",
                    "nom": le titre du bloc de l'écran,
                    "valeur", "limite", "unite",
                    "taux": valeur / limite (le pourcentage du bandeau, divisé par 100),
                    "etat": "ok" | "warn" | "nok" (l'état du module),
                    "motifs": [...] (ce qui rend l'état non conforme),
                    "formules": [{"cle", "termes": [{"symbole", "valeur", "unite"}, ...],
                                  "resultat", "unite"[, "resultat_tronque"]}, ...]}
     hauteur : valeur h_min, limite h (cm) ; + "face" (la famille la plus exigeante, ou null)
               formules hu_min = √(M·10⁶ / (α_b·b·μ)) / 10 ; h_min = hu_min + y_G
               motif « hauteur_insuffisante »
     inf/sup : valeur besoin = max(As_req ; As_min), limite As réel (mm²) ; + "As_max", "libelle"
               formules As_req = M·10⁶ / (fyd·0,9·d·10) ; As_min_ec = 0,26·fctm/fyk·b·h ;
               As_min_plancher = 0,0013·b·h ; As_min_opposee = 0,25·As_req(face opposée) ;
               As_min = max(des trois) ; besoin = max(As_req ; As_min) ; As_max = 0,04·b·h
               motifs « d_utile_negatif », « acier_insuffisant », « acier_excessif » (As > As_max)
     tau     : valeur τ, limite τ_adm de la bande atteinte (N/mm²) ; + "bande" (I, II, IV, hors),
               "besoin" (pas_d_etriers, etriers, barres_inclinees_et_etriers, inacceptable),
               "etat_bande" (l'état que le bandeau de l'écran affiche, sans le contrôle de d)
               formules τ = V·10³ / (0,75·b·h) ; τ_adm = k·fck,cube / 1,05 (k = 0,016 | 0,032 | 0,064)
               motifs « d_utile_negatif », « barres_inclinees » (état warn), « cisaillement_excessif »
     pas     : valeur pas choisi, limite s_lim (cm)
               formules s_th = Ast·fyd·d / (V·10³) (d en cm) ; s_max = min(0,75·d ; 30 cm) ;
               s_lim = min(s_th ; s_max) — avec "resultat_tronque" : la valeur tronquée au mm
               inférieur que l'écran affiche (v2.41), recopiable comme pas sans changer le verdict
               motifs « d_utile_negatif », « pas_trop_grand »

  <section>["grandeurs"] (unités : cm, mm², N/mm², kN, kN·m ; None quand V = 0 pour ce qui en dépend) :
     matériaux  beton fck fck_cube fctm alpha_b mu fyk gamma_s fyd tau_1 tau_2 tau_4
     sections   b h enrobage enrobage_calcul d_etrier M_inf M_sup V
     aciers     dist_lit1_inf dist_lit1_sup yG_calcule_inf yG_calcule_sup yG_inf yG_sup
                yG_impose_inf yG_impose_sup As_inf As_sup
     d utile    d_inf d_sup d_cisaillement d_calc_inf d_calc_sup d_calc_cisaillement
                geometrie_inf_ok geometrie_sup_ok geometrie_cisaillement_ok
     hauteur    face_hauteur M_hauteur yG_hauteur hu_min h_min
     flexion    As_req_inf As_req_sup As_min_ec As_min_plancher As_min_base As_min_opposee_inf
                As_min_opposee_sup As_min_inf As_min_sup As_min_terme_inf As_min_terme_sup
                besoin_inf besoin_sup As_max
     tranchant  tau bande_tau tau_lim besoin_tau
     étriers    Ast pas s_th s_max s_lim s_th_tronque s_max_tronque s_lim_tronque

  <section>["geometrie"] = {"b", "h", "enrobage", "enrobage_calcul",
       "inf" | "sup": {"lits": [{"n", "d", "distance" (axe de calcul), "reel" (position réelle : le dessin),
                                 "detail" (beton/position.detail), "As"}, ...], "As", "yG", "yG_calcule", "libelle"},
       "etriers": {"lignes": [{"type", "d", "brins"}, ...], "d_max", "pas", "Ast", "resume"}}
     distance = parement → axe du lit (cm), calculée comme le module (_auto_dist_lit / _get_dist_lit).

SORTIE de candidats_armatures — les dispositions d'UN lit (n de 2 à 6, Ø 10 12 16 20 25 32) pour une
face, chacune calculée par ``verifier`` (le Ø change la position du lit, donc d) ; triées par aire :
    [{"n", "d", "nom": "3Ø20", "aire" (mm²), "distance" (cm), "yG" (cm), "taux" (besoin / aire),
      "etat", "ok", "statut_section", "retenue" (la disposition actuelle), "optimal" (la plus petite
      aire « ok »)}, ...]
"""
import math

from beton import position as P

from . import dispositions as DI
from . import entree as E
from . import geometrie as G
from . import taux as TA
from .entree import classes_beton, nom_section_suivant, normaliser, nouvelle_poutre, nouvelle_section  # noqa: F401

REFERENCE = "modules/poutre.py v2.40 (APP_VERSION 1.2.41)"

OK, WARN, NOK = "ok", "warn", "nok"

# ---- les coefficients de la méthode du bureau (_dimensionnement_compute_states, render_dimensionnement_section) ----
K_FCTM = 0.30              # fctm = 0,30·fck^(2/3)
K_BRAS_DE_LEVIER = 0.9     # As,req = M / (fyd·0,9·d)
K_AS_MIN_EC = 0.26         # 0,26·fctm/fyk·b·h      (h partout, « en cohérence avec le rapport PDF »)
K_AS_MIN_PLANCHER = 0.0013 # 0,0013·b·h
K_AS_MIN_OPPOSEE = 0.25    # 0,25·As,req de la face opposée
K_AS_MAX = 0.04            # 0,04·b·h
K_TAU_SECTION = 0.75       # τ = V / (0,75·b·h)
GAMMA_TAU = 1.05           # τ_adm = k·fck,cube / 1,05
# les bandes de τ, dans l'ordre : (bande, k, état, besoin) ; au-delà de τ_adm,IV : « hors », nok
BANDES_TAU = (("I", 0.016, OK, "pas_d_etriers"),
              ("II", 0.032, OK, "etriers"),
              ("IV", 0.064, WARN, "barres_inclinees_et_etriers"))
HORS_BANDES = ("hors", NOK, "inacceptable")
CLES_TAU = {"I": "tau_1", "II": "tau_2", "IV": "tau_4"}   # τ_adm,I = tau_1, τ_adm,II = tau_2, τ_adm,IV = tau_4
K_S_MAX = 0.75             # s_max = min(0,75·d ; 30 cm)
S_MAX_PLAFOND = 30.0       # cm
D_CALCUL_MIN = 0.1         # cm : les formules prennent d = max(d, 0,1) ; d ≤ 0 rend la vérification nok

# les titres des blocs de l'écran
NOMS = {"hauteur": "Vérification de la hauteur", "inf": "Armatures inférieures",
        "sup": "Armatures supérieures", "tau": "Vérification de l'effort tranchant",
        "pas": "Détermination des étriers"}

# le sélecteur d'armatures : un lit de 2 à 6 barres, Ø de 10 à 32 parmi ceux du module
CANDIDATS_N = (2, 3, 4, 5, 6)
CANDIDATS_D = tuple(d for d in E.DIAMETRES if 10 <= d <= 32)


def fusion_statuts(*etats: str) -> str:
    """« nok » si un état l'est, sinon « warn » si un l'est, sinon « ok » — _status_merge."""
    if any(e == NOK for e in etats):
        return NOK
    if any(e == WARN for e in etats):
        return WARN
    return OK


def tronque_mm(x_cm: float) -> float:
    """Valeur tronquée au mm inférieur (cm, une décimale), comme l'écran affiche s_th et s_max (v2.41)."""
    return math.floor(x_cm * 10.0) / 10.0


def materiaux(p: dict) -> dict:
    """Les matériaux d'une poutre NORMALISÉE — lectures de _dimensionnement_compute_states."""
    bd = E._classes_lues()[p["beton"]]
    fck_cube = bd["fck_cube"]
    fyk = float(p["fyk"])
    gamma_s = float(p["gamma_s"])
    cle_mu = f"mu_a{int(p['fyk'])}"
    if cle_mu not in bd:  # repli du module (inatteignable avec beton_classes.json)
        cle_mu = "mu_a500" if "mu_a500" in bd else next(k for k in bd if k.startswith("mu_a"))
    fck = float(bd.get("fck", 0.8 * fck_cube) or (0.8 * fck_cube))
    out = {"beton": p["beton"], "fck": fck, "fck_cube": float(fck_cube),
           "fctm": K_FCTM * (fck ** (2.0 / 3.0)) if fck > 0 else 0.0,
           "alpha_b": bd["alpha_b"], "mu": bd[cle_mu],
           "fyk": fyk, "gamma_s": gamma_s, "fyd": fyk / gamma_s}
    for bande, k, _, _ in BANDES_TAU:
        out[CLES_TAU[bande]] = k * fck_cube / GAMMA_TAU
    return out


def _tau_adm(mat: dict, bande: str) -> float:
    return mat[CLES_TAU[bande]]


def _hu_min(m: float, alpha_b: float, b: float, mu: float) -> float:
    """Hauteur utile minimale (cm) : √(M·10⁶ / (α_b·b·10·μ)) / 10, M en kN·m, b en cm ; 0 sans moment."""
    return math.sqrt((m * 1e6) / (alpha_b * b * 10 * mu)) / 10 if m > 0 else 0.0


# ============================================================ le calcul d'une section
def _calcul(p: dict, s: dict, mat: dict) -> dict:
    """Toutes les grandeurs d'une section et ses états : _dimensionnement_compute_states, plus τ, s_th,
    s_max et s_lim, que le module calcule dans son code d'ÉCRAN (render_dimensionnement_section)."""
    b, h = float(p["b"]), float(p["h"])
    enrob = G.enrobage_de_calcul(p["enrobage"])
    d_et = G.diametre_etrier(s["etriers"])
    geo = {}
    for face in ("inf", "sup"):
        reels = G.positions_lits(s[face], enrob, d_et, p["jeu_entre_lits"])
        dist = [P.axe_calcul(r, p["jeu_premier_lit"], p["arrondi"]) for r in reels]
        As, yG_calcule, yG = G.face(s[face], dist, s[f"yG_{face}"])
        geo[face] = {"distances": dist, "reels": reels, "As": As, "yG_calcule": yG_calcule, "yG": yG,
                     "details": G.details_lits(s[face], enrob, d_et, p["jeu_premier_lit"], p["jeu_entre_lits"],
                                               p["arrondi"])}
    fyk, fyd, alpha_b, mu, fctm = mat["fyk"], mat["fyd"], mat["alpha_b"], mat["mu"], mat["fctm"]
    yG_inf, yG_sup = geo["inf"]["yG"], geo["sup"]["yG"]
    dist1_inf, dist1_sup = geo["inf"]["distances"][0], geo["sup"]["distances"][0]

    # hauteurs utiles : la flexion au c.d.g. des lits de SA face (inf. depuis le bas, sup. depuis le haut) ;
    # le cisaillement au lit 1, la plus petite distance des deux faces (« logique conservée » du module)
    d_inf = h - yG_inf
    d_sup = h - yG_sup
    d_cis = h - min(dist1_inf, dist1_sup)
    g_inf, g_sup, g_cis = d_inf > 0.0, d_sup > 0.0, d_cis > 0.0
    dc_inf, dc_sup, dc_cis = max(d_inf, D_CALCUL_MIN), max(d_sup, D_CALCUL_MIN), max(d_cis, D_CALCUL_MIN)
    M_inf, M_sup, V = s["M_inf"], s["M_sup"], s["V"]

    # --- hauteur : la famille la PLUS EXIGEANTE, hu,min(M) + son y_G (à égalité, l'inférieure : max garde
    #     le premier) ; sans moment, h_min = y_G inférieur
    actives = [f for f in (("inf", M_inf, yG_inf), ("sup", M_sup, yG_sup)) if f[1] > 0]
    if actives:
        face_h, M_h, yG_h = max(actives, key=lambda f: _hu_min(f[1], alpha_b, b, mu) + f[2])
    else:
        face_h, M_h, yG_h = None, 0.0, yG_inf
    hu = _hu_min(M_h, alpha_b, b, mu)
    h_min = hu + yG_h
    etat_h = OK if h_min <= h else NOK

    # --- flexion : As,req (z = 0,9·d), As,min = max(0,26·fctm/fyk·b·h ; 0,0013·b·h ; 0,25·As,req opposé), As,max
    As_min_ec = K_AS_MIN_EC * fctm / fyk * b * h * 1e2       # mm² (b, h en cm → ·1e2)
    As_min_plancher = K_AS_MIN_PLANCHER * b * h * 1e2
    As_min_base = max(As_min_ec, As_min_plancher)
    As_max = K_AS_MAX * b * h * 1e2
    As_req = {"inf": (M_inf * 1e6) / (fyd * K_BRAS_DE_LEVIER * dc_inf * 10) if M_inf > 0 else 0.0,
              "sup": (M_sup * 1e6) / (fyd * K_BRAS_DE_LEVIER * dc_sup * 10) if M_sup > 0 else 0.0}
    g = {"beton": mat["beton"], "fck": mat["fck"], "fck_cube": mat["fck_cube"], "fctm": fctm,
         "alpha_b": alpha_b, "mu": mu, "fyk": fyk, "gamma_s": mat["gamma_s"], "fyd": fyd,
         "tau_1": mat["tau_1"], "tau_2": mat["tau_2"], "tau_4": mat["tau_4"],
         "b": b, "h": h, "enrobage": p["enrobage"], "enrobage_calcul": enrob, "d_etrier": d_et,
         "M_inf": M_inf, "M_sup": M_sup, "V": V,
         "dist_lit1_inf": dist1_inf, "dist_lit1_sup": dist1_sup,
         "yG_calcule_inf": geo["inf"]["yG_calcule"], "yG_calcule_sup": geo["sup"]["yG_calcule"],
         "yG_inf": yG_inf, "yG_sup": yG_sup,
         "yG_impose_inf": s["yG_inf"] is not None, "yG_impose_sup": s["yG_sup"] is not None,
         "As_inf": geo["inf"]["As"], "As_sup": geo["sup"]["As"],
         "d_inf": d_inf, "d_sup": d_sup, "d_cisaillement": d_cis,
         "d_calc_inf": dc_inf, "d_calc_sup": dc_sup, "d_calc_cisaillement": dc_cis,
         "geometrie_inf_ok": g_inf, "geometrie_sup_ok": g_sup, "geometrie_cisaillement_ok": g_cis,
         "face_hauteur": face_h, "M_hauteur": M_h, "yG_hauteur": yG_h, "hu_min": hu, "h_min": h_min,
         "As_req_inf": As_req["inf"], "As_req_sup": As_req["sup"],
         "As_min_ec": As_min_ec, "As_min_plancher": As_min_plancher, "As_min_base": As_min_base,
         "As_max": As_max}
    etats = {"hauteur": etat_h}
    for face, opp, geom_ok in (("inf", "sup", g_inf), ("sup", "inf", g_sup)):
        opposee = K_AS_MIN_OPPOSEE * As_req[opp]
        As_min = max(As_min_base, opposee)
        besoin = max(As_req[face], As_min)
        As = geo[face]["As"]
        g[f"As_min_opposee_{face}"] = opposee
        g[f"As_min_{face}"] = As_min
        # le terme qui gouverne As,min (max garde le premier à égalité : ec, puis plancher, puis opposée)
        g[f"As_min_terme_{face}"] = ("opposee" if opposee > As_min_base
                                     else "ec" if As_min_ec >= As_min_plancher else "plancher")
        g[f"besoin_{face}"] = besoin
        etats[face] = OK if (geom_ok and As >= besoin and As <= As_max) else NOK

    # --- effort tranchant : τ = V / (0,75·b·h), comparé aux τ_adm (I, II, IV) du béton
    Ast = G.aire_etriers(s["etriers"])
    s_max = min(K_S_MAX * dc_cis, S_MAX_PLAFOND)
    pas = s["pas"]
    if V > 0:
        tau = V * 1e3 / (K_TAU_SECTION * b * h * 100)
        bande = next((x for x in BANDES_TAU if tau <= _tau_adm(mat, x[0])), None)
        if bande is not None:
            nom_bande, k_bande, etat_bande, besoin_tau = bande
            tau_lim = _tau_adm(mat, nom_bande)
        else:
            (nom_bande, etat_bande, besoin_tau), k_bande = HORS_BANDES, BANDES_TAU[-1][1]
            tau_lim = mat["tau_4"]
        etats["tau"] = etat_bande if g_cis else NOK
        # pas : s_th = Ast·fyd·d / V (d en mm, V en N → mm, puis /10 → cm)
        s_th = Ast * fyd * (dc_cis * 10.0) / (V * 1e3) / 10.0
        s_lim = min(s_th, s_max)
        etats["pas"] = OK if (pas <= s_lim and g_cis) else NOK
    else:  # sans effort tranchant, le module ne montre ni τ ni étriers et les tient pour « ok »
        tau = tau_lim = s_th = s_lim = None
        nom_bande = etat_bande = besoin_tau = k_bande = None
        etats["tau"] = etats["pas"] = OK
    g.update({"tau": tau, "bande_tau": nom_bande, "tau_lim": tau_lim, "besoin_tau": besoin_tau,
              "Ast": Ast, "pas": pas, "s_th": s_th, "s_max": s_max, "s_lim": s_lim,
              "s_th_tronque": None if s_th is None else tronque_mm(s_th), "s_max_tronque": tronque_mm(s_max),
              "s_lim_tronque": None if s_lim is None else tronque_mm(s_lim)})
    return {"g": g, "etats": etats, "geo": geo, "etat_bande": etat_bande, "k_bande": k_bande,
            "statut": fusion_statuts(*etats.values())}


# ============================================================ les vérifications (blocs de l'écran)
def _t(symbole: str, valeur, unite: str = "") -> dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def _f(cle: str, termes: list, resultat, unite: str, tronque=None) -> dict:
    f = {"cle": cle, "termes": termes, "resultat": resultat, "unite": unite}
    if tronque is not None:
        f["resultat_tronque"] = tronque
    return f


def _verif(cle: str, valeur: float, limite: float, unite: str, etat: str, motifs: list, formules: list,
           **plus) -> dict:
    v = {"cle": cle, "nom": NOMS[cle], "valeur": valeur, "limite": limite, "unite": unite,
         "taux": valeur / limite, "etat": etat, "motifs": motifs, "formules": formules}
    v.update(plus)
    return v


def _verifications(c: dict, s: dict) -> list:
    g, etats = c["g"], c["etats"]
    b_mm, h_mm = g["b"] * 10.0, g["h"] * 10.0
    out = [_verif("hauteur", g["h_min"], g["h"], "cm", etats["hauteur"],
                  [] if etats["hauteur"] == OK else ["hauteur_insuffisante"],
                  [_f("hu_min", [_t("M", g["M_hauteur"], "kN·m"), _t("alpha_b", g["alpha_b"], "N/mm²"),
                                 _t("b", b_mm, "mm"), _t("mu", g["mu"])], g["hu_min"], "cm"),
                   _f("h_min", [_t("hu_min", g["hu_min"], "cm"), _t("yG", g["yG_hauteur"], "cm")], g["h_min"], "cm")],
                  face=g["face_hauteur"])]
    for face, opp in (("inf", "sup"), ("sup", "inf")):
        As, besoin = g[f"As_{face}"], g[f"besoin_{face}"]
        motifs = ([] if g[f"geometrie_{face}_ok"] else ["d_utile_negatif"]) \
            + (["acier_insuffisant"] if As < besoin else []) + (["acier_excessif"] if As > g["As_max"] else [])
        # l'axe de calcul de chaque lit (beton/position.py : position réelle, jeu de pose, arrondi), puis d = h − y_G
        axes = [dict(_f("e", [_t(x["symbole"], x["valeur"], "cm") for x in det["termes"]], dist, "cm"), couche=k,
                     detail=det, auto=lit.get("pos") is None)
                for k, (lit, dist, det) in enumerate(zip(s[face], c["geo"][face]["distances"], c["geo"][face]["details"]))]
        formules = axes + [
            _f("d", [_t("h", g["h"], "cm"), _t("yG", g[f"yG_{face}"], "cm")], g[f"d_{face}"], "cm"),
            _f("As_req", [_t("M", g[f"M_{face}"], "kN·m"), _t("fyd", g["fyd"], "N/mm²"),
                          _t("k", K_BRAS_DE_LEVIER), _t("d", g[f"d_calc_{face}"], "cm")], g[f"As_req_{face}"], "mm²"),
            _f("As_min_ec", [_t("k", K_AS_MIN_EC), _t("fctm", g["fctm"], "N/mm²"), _t("fyk", g["fyk"], "N/mm²"),
                             _t("b", b_mm, "mm"), _t("h", h_mm, "mm")], g["As_min_ec"], "mm²"),
            _f("As_min_plancher", [_t("k", K_AS_MIN_PLANCHER), _t("b", b_mm, "mm"), _t("h", h_mm, "mm")],
               g["As_min_plancher"], "mm²"),
            _f("As_min_opposee", [_t("k", K_AS_MIN_OPPOSEE), _t("As_req_opp", g[f"As_req_{opp}"], "mm²")],
               g[f"As_min_opposee_{face}"], "mm²"),
            _f("As_min", [_t("As_min_ec", g["As_min_ec"], "mm²"), _t("As_min_plancher", g["As_min_plancher"], "mm²"),
                          _t("As_min_opposee", g[f"As_min_opposee_{face}"], "mm²")], g[f"As_min_{face}"], "mm²"),
            _f("besoin", [_t("As_req", g[f"As_req_{face}"], "mm²"), _t("As_min", g[f"As_min_{face}"], "mm²")],
               besoin, "mm²"),
            _f("As_max", [_t("k", K_AS_MAX), _t("b", b_mm, "mm"), _t("h", h_mm, "mm")], g["As_max"], "mm²")]
        out.append(_verif(face, besoin, As, "mm²", etats[face], motifs, formules,
                          As_max=g["As_max"], libelle=G.libelle_lits(s[face])))
    if g["V"] > 0:
        geom = [] if g["geometrie_cisaillement_ok"] else ["d_utile_negatif"]
        motifs_tau = geom + {"IV": ["barres_inclinees"], "hors": ["cisaillement_excessif"]}.get(g["bande_tau"], [])
        out.append(_verif("tau", g["tau"], g["tau_lim"], "N/mm²", etats["tau"], motifs_tau,
                          [_f("tau", [_t("V", g["V"], "kN"), _t("k", K_TAU_SECTION), _t("b", b_mm, "mm"),
                                      _t("h", h_mm, "mm")], g["tau"], "N/mm²"),
                           _f("tau_lim", [_t("k", c["k_bande"]), _t("fck_cube", g["fck_cube"], "N/mm²"),
                                          _t("gamma", GAMMA_TAU)], g["tau_lim"], "N/mm²")],
                          bande=g["bande_tau"], besoin=g["besoin_tau"], etat_bande=c["etat_bande"]))
        motifs_pas = geom + (["pas_trop_grand"] if g["pas"] > g["s_lim"] else [])
        d = g["d_calc_cisaillement"]
        out.append(_verif("pas", g["pas"], g["s_lim"], "cm", etats["pas"], motifs_pas,
                          [_f("s_th", [_t("Ast", g["Ast"], "mm²"), _t("fyd", g["fyd"], "N/mm²"), _t("d", d, "cm"),
                                       _t("V", g["V"], "kN")], g["s_th"], "cm", g["s_th_tronque"]),
                           _f("s_max", [_t("k", K_S_MAX), _t("d", d, "cm"), _t("plafond", S_MAX_PLAFOND, "cm")],
                              g["s_max"], "cm", g["s_max_tronque"]),
                           _f("s_lim", [_t("s_th", g["s_th"], "cm"), _t("s_max", g["s_max"], "cm")],
                              g["s_lim"], "cm", g["s_lim_tronque"])]))
    return out


def _geometrie(p: dict, s: dict, c: dict, dispo: dict) -> dict:
    """La géométrie pour le dessin et la note : les lits (positions, abscisses), les lignes d'effort tranchant (emprise),
    les armatures latérales. Les clés d'avant gardent leurs valeurs ; « x », « rep », « pos », « cadres » et
    « laterales » s'y ajoutent (site)."""
    g, geo = c["g"], c["geo"]
    out = {"b": g["b"], "h": g["h"], "enrobage": p["enrobage"], "enrobage_calcul": g["enrobage_calcul"],
           "jeu_pose": p["jeu_premier_lit"], "arrondi": p["arrondi"]}
    for face in ("inf", "sup"):
        lits = []
        for lit, e, r, det, xs in zip(s[face], geo[face]["distances"], geo[face]["reels"], geo[face]["details"],
                                      dispo["abscisses"][face]):
            lits.append({"n": lit["n"], "d": lit["d"], "distance": e, "reel": r, "detail": det,
                         "As": lit["n"] * G.aire_barre(lit["d"]),
                         "x": xs, "rep": lit.get("rep", "uniforme"), "pos": lit.get("pos"), "sh": lit.get("sh")})
        out[face] = {"lits": lits, "As": geo[face]["As"], "yG": geo[face]["yG"], "yG_calcule": geo[face]["yG_calcule"],
                     "libelle": G.libelle_lits(s[face])}
    out["etriers"] = {"lignes": [{"type": e["type"], "d": e["d"], "brins": E.BRINS[e["type"]], "de": e.get("de"),
                                  "a": e.get("a")} for e in s["etriers"]],
                      "d_max": g["d_etrier"], "pas": g["pas"], "Ast": g["Ast"], "resume": G.resume_etriers(s["etriers"]),
                      "cadres": dispo["cadres"], "brins": sum(E.BRINS[e["type"]] for e in s["etriers"])}
    out["laterales"] = dispo["laterales"]
    return out


def _section(p: dict, s: dict, mat: dict) -> dict:
    c = _calcul(p, s, mat)
    verifs = _verifications(c, s)
    gouv = max(verifs, key=lambda v: v["taux"])
    dispo = DI.controler(p, s, c["g"]["enrobage_calcul"], c["g"]["d_etrier"],
                         {f: c["geo"][f]["reels"] for f in ("inf", "sup")})
    # « statut » : la méthode du bureau seule (parité avec l'écran Streamlit) ; « statut_complet » y ajoute les
    # dispositions constructives (site). « sollicitee » : la section porte au moins un effort — sans aucun, ses
    # vérifications ne disent rien de la poutre (elles passent toutes), le site la dit « données incomplètes » et
    # jamais « conforme ». « non_verifiees » : les vérifications que le module ne fait pas, et pourquoi (sans effort
    # tranchant, ni τ ni étriers : l'écran Streamlit ne les montre pas et les tient pour « ok »).
    non_verifiees = [] if s["V"] > 0 else [{"cle": k, "nom": NOMS[k], "raison": "sans effort tranchant (V = 0)"}
                                           for k in ("tau", "pas")]
    return {"nom": s["nom"], "M_inf": s["M_inf"], "M_sup": s["M_sup"], "V": s["V"],
            "statut": c["statut"], "statut_complet": fusion_statuts(c["statut"], dispo["etat"]),
            "sollicitee": s["M_inf"] > 0 or s["M_sup"] > 0 or s["V"] > 0, "non_verifiees": non_verifiees,
            "taux_max": gouv["taux"], "gouvernant": gouv["cle"],
            "grandeurs": c["g"], "verifications": verifs, "dispositions": dispo, "geometrie": _geometrie(p, s, c, dispo)}


# ============================================================ l'API
def verifier(poutre: dict) -> dict:
    """Calcule chaque section de la poutre (voir le format en tête du fichier). Lève ValueError si la
    poutre sort du domaine de l'écran."""
    p = normaliser(poutre)
    mat = materiaux(p)
    sections = [_section(p, s, mat) for s in p["sections"]]
    gouv_s, gouv_v = max(((sec, v) for sec in sections for v in sec["verifications"]), key=lambda x: x[1]["taux"])
    avert = []
    if not p["enrobage"]:
        avert.append({"code": "enrobage_nul",
                      "message": "Enrobage nul : le module place les lits avec un enrobage de "
                                 f"{G.ENROBAGE_SI_NUL:g} cm (il lit « enrobage or 3,0 »)."})
    out = {"reference": REFERENCE, "poutre": p, "materiaux": mat, "sections": sections,
            "taux_max": gouv_v["taux"], "statut": fusion_statuts(*(sec["statut"] for sec in sections)),
            "statut_complet": fusion_statuts(*(sec["statut_complet"] for sec in sections)),
            "sollicitee": all(sec["sollicitee"] for sec in sections),
            "gouvernant": {"section": gouv_s["nom"], "cle": gouv_v["cle"], "nom": gouv_v["nom"], "taux": gouv_v["taux"]},
            "avertissements": avert}
    # l'option « Taux d'armature » (beton/poutre/taux.py) : la clé n'existe que si l'option est allumée — éteinte, la
    # sortie est celle d'avant, à l'octet
    ta = TA.calculer(p, sections)
    if ta is not None:
        out["taux_armature"] = ta
    return out


def candidats_armatures(poutre: dict, nom_section: str, face: str) -> list:
    """Le sélecteur d'armatures d'une face : chaque disposition d'UN lit (n de 2 à 6 barres, Ø de 10 à
    32 parmi ceux du module) remplace les lits de la face, et la section est recalculée (le Ø déplace
    le lit, donc d) ; le reste de la poutre est inchangé (y_G imposé compris). Triées par aire
    croissante ; « optimal » marque la plus petite aire « ok », « retenue » la disposition actuelle."""
    if face not in ("inf", "sup"):
        raise ValueError(f"face : « inf » ou « sup » attendu (reçu {face!r})")
    p = normaliser(poutre)
    s = next((x for x in p["sections"] if x["nom"] == nom_section), None)
    if s is None:
        raise ValueError(f"section {nom_section!r} introuvable ({', '.join(x['nom'] for x in p['sections'])})")
    mat = materiaux(p)
    out = []
    for d in CANDIDATS_D:
        for n in CANDIDATS_N:
            essai = dict(s)
            essai[face] = [{"n": n, "d": d}]
            c = _calcul(p, essai, mat)
            g = c["g"]
            # les dispositions constructives du lit (site) : « possible » faux quand les barres se chevauchent ou
            # sortent de l'étrier (la disposition ne se pose pas : le site ne la propose pas), « serree » sous
            # l'espacement minimal de l'EC2 (elle se pose, mais n'est pas conforme)
            dispo = DI.controler(p, essai, g["enrobage_calcul"], g["d_etrier"],
                                 {f: c["geo"][f]["reels"] for f in ("inf", "sup")})
            du_lit = [it for it in dispo["items"] if it["face"] == face and it["cle"] in ("lit", "cadre")]
            out.append({"n": n, "d": d, "nom": f"{n}Ø{d}", "aire": g[f"As_{face}"],
                        "distance": g[f"dist_lit1_{face}"], "yG": g[f"yG_{face}"],
                        "taux": g[f"besoin_{face}"] / g[f"As_{face}"], "etat": c["etats"][face],
                        "ok": c["etats"][face] == OK, "statut_section": c["statut"],
                        "possible": not any(it["impossible"] for it in du_lit),
                        "serree": any(it["etat"] == DI.NOK and not it["impossible"] for it in du_lit),
                        "retenue": [{"n": x["n"], "d": x["d"]} for x in s[face]] == [{"n": n, "d": d}], "optimal": False})
    out.sort(key=lambda x: (x["aire"], x["n"]))
    optimal = next((x for x in out if x["ok"]), None)
    if optimal is not None:
        optimal["optimal"] = True
    return out

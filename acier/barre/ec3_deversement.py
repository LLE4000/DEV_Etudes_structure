# -*- coding: utf-8 -*-
"""EUROCODE 3, le DÉVERSEMENT d'une barre fléchie autour de son axe fort (NBN EN 1993-1-1 § 6.3.2).
docs/acier/BARRES_ACIER.md § 5.6.

    M_cr : numérique, par TRONÇON entre deux sections tenues en déplacement latéral et en torsion (acier/barre/mcr.py)
    λ̄_LT = √(W_y·f_y / M_cr)        W_y : W_pl,y (classes 1 et 2), W_el,y (3), W_eff,y (4)
    I et H, laminés ou soudés (§ 6.3.2.3) :
        Φ_LT = 0,5·[1 + α_LT·(λ̄_LT − λ̄_LT,0) + β·λ̄_LT²]       χ_LT = 1/(Φ_LT + √(Φ_LT² − β·λ̄_LT²)) ≤ 1 et ≤ 1/λ̄_LT²
        tableau 6.5 : laminé h/b ≤ 2 : b ; > 2 : c — soudé h/b ≤ 2 : c ; > 2 : d
        f = 1 − 0,5·(1 − k_c)·[1 − 2·(λ̄_LT − 0,8)²] ≤ 1        χ_LT,mod = χ_LT/f ≤ 1 et ≤ 1/λ̄_LT²
        k_c (tableau 6.6, normatif : ANB:2018, 6.3.2.3(2)) : moment constant 1 ; linéaire 1/(1,33 − 0,33·ψ) ; deux appuis
        simples : charge répartie 0,94, charge concentrée à mi-portée 0,86 ; deux encastrements : 0,90 et 0,77 ; un
        encastrement et un appui simple : 0,91 et 0,82 — ces six-là seulement quand le tronçon EST la portée de ce
        schéma, sous cette charge seule (kc_portee) ; tout autre diagramme : k_c = 1 (f = 1, du côté sûr)
    § 6.3.2.2 (cas général), pour information à côté du § 6.3.2.3 (DÉCISION BUREAU n° 4, brief § 7.7) :
        Φ_LT = 0,5·[1 + α_LT·(λ̄_LT − 0,2) + λ̄_LT²]      χ_LT = 1/(Φ_LT + √(Φ_LT² − λ̄_LT²)) ≤ 1
        tableau 6.4 : laminé h/b ≤ 2 : a ; > 2 : b — soudé h/b ≤ 2 : c ; > 2 : d
    I SOUDÉ — PRS ou I saisi « soudé » (brief § 7.7 et 18.5, DÉCISION BUREAU n° 10) : § 6.3.2.2 par défaut (k_c = 1,
        f = 1), le § 6.3.2.3 pour information ; § 6.3.2.3 seulement pour une section bisymétrique déclarée « section
        soudée équivalente » (prs_soudee_equivalente), le § 6.3.2.2 alors pour information ; un PRS monosymétrique :
        § 6.3.2.2 seul. (Les tableaux 6.4 et 6.5 donnent les MÊMES courbes c et d à un I soudé, et le § 6.3.2.3 divise
        ensuite par f ≤ 1 : pour une section soudée, le § 6.3.2.2 n'est jamais le moins sévère.)
        L'autre méthode a UNE écriture : grandeurs["deversement"]["information"] (laminés et I soudés)
    tube rectangulaire (cas général, § 6.3.2.2) : courbe d, λ̄_LT,0 = 0,2 et β = 1 dans Φ_LT, f = 1 ; le seuil de
        6.3.2.2(4) renvoie au λ̄_LT,0 de 6.3.2.3 — celui de l'ANB, 0,2
    M_b,Rd = χ_LT,mod·W_y·f_y / γ_M1
    négligé (χ_LT = 1) si λ̄_LT ≤ λ̄_LT,0 ou M_Ed/M_cr ≤ λ̄_LT,0²                                        (§ 6.3.2.2(4))

Tube rond ou carré : pas de déversement. Semelle comprimée tenue en continu (« Lc » nul, ou « actif » faux) : pas de
déversement, et c'est dit.

PARAMÈTRES NATIONAUX (acier/barre/parametres.py) : (λ̄_LT,0 ; β) = (0,2 ; 1,0), NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) —
M_cr est calculé par le solveur du module avec les combinaisons, la distribution réelle des moments et les maintiens,
qu'il y en ait ou non dans le tronçon. La VARIANTE (0,4 ; 0,75) du même alinéa n'est appliquée que sur demande
(``deversement.couple`` = « variante »), pour une poutre de BÂTIMENT qui a des maintiens, TOUS ignorés dans M_cr
(``deversement.maintiens_negliges`` ≥ 1 ; M_cr du solveur, sur un tronçon qui va d'un appui à l'autre sans maintien
dedans — couple_anb) ; ailleurs elle est refusée, avec la raison, et le couple par défaut s'applique (brief § 7.7, test 9
du § 11.5). Le couple employé est nommé ; un couple par défaut changé par surcharge est signalé (« couple_hors_anb »).

Pour l'annexe D ANB (flexion composée), ce fichier rend aussi M_cr,0 — le moment critique du même tronçon sous moment
CONSTANT — λ̄_0 = √(W_y·f_y/M_cr,0) et C₁ selon l'annexe E ANB, qui admet les méthodes avancées : le rapport
M_cr/M_cr,0 du solveur, les charges ramenées au centre de cisaillement (brief § 7.8 ; la règle C₁ = k_c⁻² de l'AC:2009
n'est pas employée : l'ANB, note de l'annexe A, renvoie à l'annexe E ANB, de domaine plus large).
"""
import math
from typing import Callable, Dict, List, Optional, Tuple

from acier import formats as FMT
from acier.flambement import normes as NO

from . import mcr as MC
from . import parametres as PA
from . import prs as PR
from . import section as SE
from .resultat import Journal, Refus, fr as _fr, terme as _t

REF = "EN 1993-1-1 § 6.3.2"
CONSTANT, LINEAIRE, REPARTI, PONCTUEL, GENERAL = "constant", "lineaire", "reparti", "ponctuel", "general"
DIAGRAMMES = (CONSTANT, LINEAIRE, REPARTI, PONCTUEL, GENERAL)
KC_REPARTI, KC_PONCTUEL = 0.94, 0.86             # tableau 6.6
# tableau 6.6 : la PORTÉE entière d'un de ses schémas (appuis de flexion du modèle de la barre chargée, forme de la charge)
KC_PORTEES = {("appuis", REPARTI): KC_REPARTI, ("appuis", PONCTUEL): KC_PONCTUEL, ("bi_enc", REPARTI): 0.90, ("bi_enc", PONCTUEL): 0.77,
              ("enc_app", REPARTI): 0.91, ("enc_app", PONCTUEL): 0.82}
MODELES_KC = ("appuis", "bi_enc", "enc_app")
ELANCE = 2.0                                     # h/b : la limite du tableau 6.5
COURBES_LT = {"lamine": ("b", "c"), "soude": ("c", "d")}        # (h/b ≤ 2, h/b > 2)
COURBE_GENERALE, LAMBDA0_GENERAL, BETA_GENERAL = "d", 0.2, 1.0  # § 6.3.2.2, « autres sections »
COURBES_GENERALES = {"lamine": ("a", "b"), "soude": ("c", "d")}    # § 6.3.2.2, tableau 6.4 (h/b ≤ 2, h/b > 2)
OUVRAGES = ("batiment", "autre")                 # le type d'ouvrage (brief § 4.1) : la variante du couple n'est admise qu'en bâtiment
COUPLES = ("defaut", "variante")
SOUDES = ("soude_oxycoupe", "soude_larges_plats")
EMPECHE, SANS_OBJET, NEGLIGE, CALCULE = "empeche", "sans_objet", "neglige", "calcule"
METHODE_GENERALE, METHODE_LAMINES = "§ 6.3.2.2", "§ 6.3.2.3"


def coefficient_kc(diagramme: str, psi: float = 1.0) -> Tuple[float, bool]:
    """(k_c, « lu dans le tableau 6.6 ») : hors tableau, k_c = 1."""
    if diagramme == CONSTANT:
        return 1.0, True
    if diagramme == LINEAIRE:
        return 1.0 / (1.33 - 0.33 * psi), True
    if diagramme == REPARTI:
        return KC_REPARTI, True
    if diagramme == PONCTUEL:
        return KC_PONCTUEL, True
    return 1.0, False


def forme_portee(charges, L: float) -> Optional[str]:
    """La forme de la charge d'une PORTÉE de longueur ``L`` (m), charges de l'écran ({"type", "v", "a", "b"} en m et kN) :
    REPARTI si toutes sont uniformes (« q ») sur toute la portée ; PONCTUEL si toutes sont ponctuelles à mi-portée ;
    sinon None (un couple, une charge triangulaire ou partielle, deux formes mêlées). Une charge nulle ne compte pas."""
    vraies = [ch for ch in charges if ch.get("v")]
    tol = 1e-6 * L
    if not vraies:
        return None
    if all(ch["type"] == "q" and ch.get("a", 0.0) <= tol and ch.get("b", L) >= L - tol for ch in vraies):
        return REPARTI
    if all(ch["type"] == "P" and abs(ch["a"] - L / 2.0) <= tol for ch in vraies):
        return PONCTUEL
    return None


def kc_portee(modele: str, charges, L: float) -> Optional[float]:
    """k_c du tableau 6.6 (normatif : NBN EN 1993-1-1 ANB:2018, 6.3.2.3(2)) d'une portée entière qui EST un de ses
    schémas : le modèle de la barre (« appuis » : deux appuis simples ; « bi_enc » : deux encastrements ; « enc_app » :
    un encastrement et un appui simple) et la forme de sa charge (forme_portee). Tout le reste — un encastrement partiel,
    des charges mêlées, une console, une travée de poutre continue — : None (k_c = 1, du côté sûr ; brief § 7.7)."""
    return KC_PORTEES.get((modele, forme_portee(charges, L)))


def kc_lineaire(charges, x1: float, x2: float, M1: float, M2: float) -> Optional[float]:
    """k_c = 1/(1,33 − 0,33·ψ) (tableau 6.6, moment linéaire) d'un tronçon [x1 ; x2] (m) entre deux sections tenues qui ne
    porte AUCUNE charge à l'intérieur (charges de l'écran, en m) : son moment va en ligne droite de M1 à M2 ; ψ est le
    rapport du plus petit au plus grand (signé, −1 ≤ ψ ≤ 1). Une charge dedans, ou pas de moment : None (k_c = 1)."""
    tol = 1e-6 * max(x2 - x1, 1e-9)
    for ch in charges:
        if not ch.get("v"):
            continue
        if ch["type"] in ("P", "C") and x1 + tol < ch["a"] < x2 - tol:
            return None
        if ch["type"] in ("q", "tri") and ch.get("a", 0.0) < x2 - tol and ch.get("b", x2) > x1 + tol:
            return None
    grand, petit = (M1, M2) if abs(M1) >= abs(M2) else (M2, M1)
    if abs(grand) <= 1e-9:
        return None
    return coefficient_kc(LINEAIRE, petit / grand)[0]


def troncon_impose(L: float, M: float, diagramme: str, psi: float, zg: float, sens_charge: float = 1.0) -> Dict:
    """Le tronçon de déversement du mode « efforts imposés » : entre deux fourches, le moment maximal ``M`` (N·mm) et
    la forme de son diagramme ; les charges, appliquées à la cote ``zg``, se calculent sur |M| et DESCENDENT
    (``sens_charge`` = +1) ou MONTENT (−1). ``M`` < 0 (un PRS monosymétrique dont la semelle basse est comprimée) ne
    change que le signe du moment, donc z_j et la semelle comprimée. Quand M_y,Ed < 0, ajouter() calcule les deux sens."""
    sens, M = (-1.0 if M < 0 else 1.0), abs(M)              # les charges sur |M| ; le signe, au moment seul (plus bas)
    if diagramme == LINEAIRE:
        def moment(x: float) -> float:
            return M * (1.0 - (1.0 - psi) * x / L)
        charges: Dict = {}
    elif diagramme == REPARTI:
        def moment(x: float) -> float:
            return 4.0 * M * x * (L - x) / L ** 2
        charges = {"reparties": [(0.0, L, 8.0 * M / L ** 2, zg)]}
    elif diagramme == PONCTUEL:
        def moment(x: float) -> float:
            return 2.0 * M * (x if x <= L / 2.0 else L - x) / L
        charges = {"ponctuelles": [(L / 2.0, 4.0 * M / L, zg)]}
    else:                                        # constant ; « general » sans tronçon décrit : pris constant, du côté sûr
        def moment(x: float) -> float:
            return M
        charges = {}
    if sens < 0:                                             # M_y,Ed < 0 : le diagramme retourné, les charges gardées
        brut = moment

        def moment(x: float) -> float:
            return -brut(x)
    if sens_charge < 0:                                      # la charge MONTE : chaque intensité retournée, sa cote gardée
        charges = {k: [(*c[:-2], -c[-2], c[-1]) for c in v] for k, v in charges.items()}
    return {"L": L, "moment": moment, "M_max": abs(M), "appuis": (MC.FOURCHE, MC.FOURCHE), "maintiens": [], **charges}


def _critique(s: Dict, P: Dict, tr: Dict, moment: Optional[Callable[[float], float]] = None) -> Dict:
    """M_cr du tronçon ``tr`` ; avec ``moment``, sous ce diagramme et SANS charge transversale (le moment constant).
    Un PRS monosymétrique : avec le terme de Wagner (z_j de la semelle haute comprimée, acier/barre/prs.zj_haut)."""
    EIz, EIw, GIt = P["E"] * s["Iz"], P["E"] * (s.get("Iw") or 0.0), P["G"] * s["It"]
    zj = PR.zj_haut(s)
    if moment is not None:
        return MC.moment_critique(tr["L"], EIz, EIw, GIt, moment, appuis=tr["appuis"], maintiens=tr["maintiens"], M_max=1.0, zj=zj)
    return MC.moment_critique(tr["L"], EIz, EIw, GIt, tr["moment"], reparties=tr.get("reparties", ()), ponctuelles=tr.get("ponctuelles", ()),
                              appuis=tr["appuis"], maintiens=tr["maintiens"], M_max=tr.get("M_max"), zj=zj)


def chi_general(lbLT: float, alpha: float) -> float:
    """χ_LT du cas général (§ 6.3.2.2(1)) : Φ_LT = 0,5·[1 + α_LT·(λ̄_LT − 0,2) + λ̄_LT²], χ_LT ≤ 1 (6.56)."""
    Phi = 0.5 * (1.0 + alpha * (lbLT - 0.2) + lbLT ** 2)
    return min(1.0 / (Phi + math.sqrt(Phi ** 2 - lbLT ** 2)), 1.0)


def chi_lamine(lbLT: float, alpha: float, lbLT0: float, beta: float, kc: float) -> Dict:
    """§ 6.3.2.3 : Φ_LT, χ_LT ≤ 1 et ≤ 1/λ̄_LT² (6.57), f ≤ 1 et χ_LT,mod = χ_LT/f ≤ 1 et ≤ 1/λ̄_LT² (6.58, AC:2009).
    k_c = 1 donne f = 1 (le tube rectangulaire, le cas général passent par ici avec (0,2 ; 1,0) et k_c = 1)."""
    Phi = 0.5 * (1.0 + alpha * (lbLT - lbLT0) + beta * lbLT ** 2)
    brut = 1.0 / (Phi + math.sqrt(Phi ** 2 - beta * lbLT ** 2))
    plafond = min(1.0, 1.0 / lbLT ** 2)
    chi0 = min(brut, plafond)
    f = min(1.0 - 0.5 * (1.0 - kc) * (1.0 - 2.0 * (lbLT - 0.8) ** 2), 1.0)
    chi = min(chi0 / f, plafond)
    return {"Phi": Phi, "brut": brut, "plafond": plafond, "chi0": chi0, "f": f, "chi": chi}


def _au_centre(tr: Dict) -> Dict:
    """Le tronçon ``tr``, ses charges ramenées au centre de cisaillement (z_g = 0) : le C₁ de l'annexe E ANB."""
    return {**tr, "reparties": [(*c[:-1], 0.0) for c in tr.get("reparties", ())],
            "ponctuelles": [(*c[:-1], 0.0) for c in tr.get("ponctuelles", ())]}


MAINTIENS_COMPTES = "maintiens comptés dans M_cr"


def couple_anb(dev: Dict, tr: Dict, ouvrage: str, portee: float = 0.0) -> Tuple[bool, Optional[str]]:
    """(variante appliquée, motif du refus) — la variante de NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1), n'est permise que
    pour une poutre de BÂTIMENT avec des maintiens, à condition qu'ils soient TOTALEMENT ignorés dans M_cr. Le module ne
    l'admet que s'il le VOIT : au moins un maintien déclaré « négligé » ; M_cr calculé par le solveur (un M_cr donné peut
    les compter) ; aucun maintien dans le tronçon ; et un tronçon qui va d'un appui à l'autre — un tronçon de la barre
    chargée le déclare (``entre_appuis``, faux à défaut), un tronçon des efforts imposés l'est quand L_c couvre la
    ``portee`` (la plus grande longueur des axes) : plus court, il est borné par des maintiens que M_cr compte.
    Ailleurs : refusée, avec son motif court (le couple par défaut s'applique)."""
    if dev.get("couple", "defaut") != "variante":
        return False, None
    if ouvrage != "batiment":
        return False, "hors bâtiment"
    if not dev.get("maintiens_negliges"):
        return False, "aucun maintien"
    if dev.get("Mcr") is not None:
        return False, "M_cr donné"
    if tr["maintiens"]:
        return False, MAINTIENS_COMPTES
    if dev.get("troncon") is not None:
        if not tr.get("entre_appuis"):
            return False, MAINTIENS_COMPTES
    elif not portee > 0:
        return False, "portée non donnée"
    elif tr["L"] < portee * (1.0 - 1e-9):
        return False, MAINTIENS_COMPTES
    return True, None


def ajouter(J: Journal, d: Dict) -> None:
    """Le déversement (``G["deversement"]``) et la vérification E-Mb."""
    G, s = J.G, d["section"]
    if not G.get("couvert"):
        return
    My = G["My_calcul"]
    dev = d["deversement"]
    D: Dict = {"cas": SANS_OBJET, "chi_LT": 1.0, "symbole_chi": "χ_LT", "lambda_0": 0.0, "C1": 1.0, "kc": 1.0}
    G["deversement"] = D
    if not My > 0 or s["forme"] not in ("I", "tube_rect"):
        return
    if not dev["actif"] or not dev["Lc"] > 0:
        D["cas"] = EMPECHE
        J.hyp.append("Déversement empêché : semelle comprimée tenue en continu.")
        return
    P = d["parametres"]
    fy, gM1, Wy = G["fy"], P["gamma_M1"], G["W_y"]
    if not s.get("It") or (s["forme"] == "I" and s.get("Iw") is None):
        raise Refus("I_t et I_w manquent : le déversement en dépend")
    fl = d["flexion"]["y"]
    # un PRS monosymétrique : le SIGNE du moment dit la semelle comprimée (M_y > 0 : la semelle haute) ; une section
    # bisymétrique se calcule comme avant, moment pris positif
    signe = -1.0 if (PR.monosymetrique(s) and d.get("semelle_comprimee") == PR.BAS) else 1.0
    tr = dev["troncon"] or troncon_impose(dev["Lc"], signe * My, fl["diagramme"], fl["psi"], dev["zg"])
    # efforts imposés, M_y,Ed < 0 : le sens réel de la charge n'est pas connu — une charge de gravité, ou un soulèvement
    # (le seul qui donne ce diagramme entre deux fourches). À VÉRIFIER : le plus petit M_cr des deux sens, toute section
    inconnu = (dev["troncon"] is None and dev["Mcr"] is None and d.get("MyEd", 0.0) < 0 and bool(dev["zg"])
               and bool(tr.get("reparties") or tr.get("ponctuelles")))
    kc, tabule = (dev["kc"], True) if dev["kc"] is not None else coefficient_kc(fl["diagramme"] if dev["troncon"] is None else GENERAL, fl["psi"])
    nomW = "W_pl,eff,y" if G.get("classe_2_efficace") else {1: "W_pl,y", 2: "W_pl,y", 3: "W_el,y", 4: "W_eff,y"}[G["classe"]]
    cles = []
    # ---- le moment critique
    if dev["Mcr"] is not None:
        Mcr = dev["Mcr"]
        D["Mcr_donne"] = True
        J.hyp.append(f"Moment critique donné : M_cr = {_fr(Mcr / 1e6, 1)} kN·m.")
    else:
        r = _critique(s, P, tr)
        if inconnu:
            tr_m = troncon_impose(dev["Lc"], signe * My, fl["diagramme"], fl["psi"], dev["zg"], sens_charge=-1.0)
            r_m = _critique(s, P, tr_m)
            D["sens_charge"] = {"descendante": r["M_cr"], "montante": r_m["M_cr"]}
            if r_m["M_cr"] is not None and (r["M_cr"] is None or r_m["M_cr"] < r["M_cr"]):
                tr, r = tr_m, r_m                                # la charge qui MONTE est la plus défavorable
            D["sens_charge"]["retenu"] = "montante" if sens_charges(tr) < 0 else "descendante"
        Mcr = r["M_cr"]
        if Mcr is None:
            D["cas"] = EMPECHE
            J.hyp.append("Déversement empêché par les maintiens du tronçon (aucun moment critique).")
            return
        D.update({"lambda_cr": r["lambda"], "M_max": r["M_max"]})
        cles.append(J.f("M_cr", "M_cr", [_t("λ_cr", r["lambda"]), _t("|M_y|max", r["M_max"] / 1e6, "kN·m")], Mcr / 1e6, "kN·m", "e_M_cr"))
    # ---- DÉCISION BUREAU n° 12 (mise à jour 1, § 1.2) : la formule de l'annexe E ANB, tronçon par tronçon, et le solveur —
    # les deux toujours calculés ; celui de la table entre dans λ̄_LT, χ_LT et M_b,Rd (le tronçon qui gouverne : le plus
    # grand M_Ed/M_b,Rd). Hors domaine, ou une formule au-dessus du solveur de plus de 2 % : le solveur.
    Mcr_s = Mcr
    F = None if dev["Mcr"] is not None else formule_annexe_e(d, tr, s, P)
    alpha_f = min((t_["M_cr"] / t_["M_Ed"] for t_ in F["troncons"] if t_["M_Ed"] > 0), default=None) if F and F["domaine"] else None
    au_dessus = alpha_f is not None and alpha_f > GARDE_FORMULE * Mcr_s / My
    retenu = "annexe_E" if (P["mcr_retenu"] == "annexe_E" and alpha_f is not None and not au_dessus) else "solveur"
    D["mcr"] = {"retenu": retenu, "decision": P["mcr_retenu"], "solveur": Mcr_s, "formule": F, "au_dessus": au_dessus,
                "x0": (dev["troncon"] or {}).get("x0", 0.0), "entier": (dev["troncon"] or {}).get("entier", dev["troncon"] is None)}
    simple = tr["appuis"] == (MC.FOURCHE, MC.FOURCHE) and not tr["maintiens"]
    if simple:
        Mcr0 = MC.moment_constant(tr["L"], P["E"] * s["Iz"], P["E"] * (s.get("Iw") or 0.0), P["G"] * s["It"], signe * PR.zj_haut(s))
    else:
        Mcr0 = _critique(s, P, tr, moment=lambda x: signe)["M_cr"] or Mcr
    # ---- la courbe et le couple (λ̄_LT,0 ; β) de l'ANB:2018, 6.3.2.3(1) : (0,2 ; 1,0), ou la variante si elle est permise
    # (un coefficient de la table s'écrit comme son champ, zéros inutiles ôtés : acier/formats.nb, le formateur des
    # valeurs saisies — jamais le format « g » de Python, qui écrit un point décimal ; test_barre_ecriture le refuse)
    D["a_confirmer"] = [k for k in ("lambda_LT0_avec_maintiens", "beta_avec_maintiens") if PA.PARAMETRES[k]["statut"] == PA.A_CONFIRMER]
    defaut = (P["lambda_LT0_avec_maintiens"], P["beta_avec_maintiens"])
    if defaut != (PA.valeur("lambda_LT0_avec_maintiens"), PA.valeur("beta_avec_maintiens")):
        # le couple par défaut changé par une surcharge (une autre annexe, un cas d'essai) : il s'applique partout, sans le
        # contrôle bâtiment / maintiens de la variante — c'est dit
        J.alerte("couple_hors_anb", "attention", f"Couple ({FMT.nb(defaut[0])} ; {FMT.nb(defaut[1])}) imposé : hors ANB:2018, 6.3.2.3(1).")
    famille, autre, autre_variante = None, None, False
    prs = PR.est_prs(s)
    if s["forme"] == "I":
        famille = "soude" if (s["fabrication"] in SOUDES or SE.soude(s)) else "lamine"
        if not s.get("h") or not s.get("b"):
            raise Refus("h et b manquent : la courbe de déversement en dépend")
        elance = s["h"] / s["b"] > ELANCE
        courbe = COURBES_LT[famille][elance]
        lbLT0, beta = P["lambda_LT0_avec_maintiens"], P["beta_avec_maintiens"]
        variante, refus = couple_anb(dev, tr, d["ouvrage"], max(d["axes"]["y"]["L"], d["axes"]["z"]["L"]))
        if variante:
            lbLT0, beta = P["lambda_LT0_variante"], P["beta_variante"]
        elif refus:
            J.alerte("variante_refusee", "attention", f"Variante ({FMT.nb(P['lambda_LT0_variante'])} ; {FMT.nb(P['beta_variante'])}) refusée : {refus}.")
        D["variante"] = variante
        seuil = lbLT0
        methode = "§ 6.3.2.3"
        # l'AUTRE méthode, pour information (DÉCISION BUREAU n° 4, brief § 7.7) : (méthode, courbe, λ̄_LT,0, β, k_c)
        autre = ("§ 6.3.2.2", COURBES_GENERALES[famille][elance], LAMBDA0_GENERAL, BETA_GENERAL, 1.0)
        generale = False
        if famille == "lamine" and P["methode_deversement_lamines"] == "6.3.2.2":     # le défaut changé dans la table
            generale = True
        if famille == "soude" and not (P["prs_soudee_equivalente"] and not PR.monosymetrique(s)):
            # une section en I SOUDÉE, PRS ou I saisi « soudé » (brief § 7.7 et 18.5, DÉCISION BUREAU n° 10) : § 6.3.2.2 par
            # défaut — tableau 6.4, « sections en I soudées », les mêmes courbes c et d que le tableau 6.5, sans f ≤ 1 : jamais
            # moins sévère ; § 6.3.2.3 seulement pour une section bisymétrique déclarée « section soudée équivalente » ; un
            # PRS monosymétrique : § 6.3.2.2 seul. (Intégration du lot B : l'I soudé qui n'est pas un PRS restait au
            # § 6.3.2.3 — M_b,Rd trop haut de 2 à 7,5 % — défaut non conservatif, latent : aucun écran ne le saisit encore.)
            generale = True
        if generale:                             # § 6.3.2.2 : pas de f (k_c = 1), ni de variante du couple
            # le § 6.3.2.3 pour information garde le couple que couple_anb a admis — la variante (0,4 ; 0,75) comprise — et
            # le DIT (D["information"]["variante"], et le message d'un PRS) : intégration du lot B, il la prenait sans le dire
            autre_variante = bool(D["variante"])
            autre = (methode, courbe, lbLT0, beta, kc)
            courbe, lbLT0, beta, seuil, methode = COURBES_GENERALES[famille][elance], LAMBDA0_GENERAL, BETA_GENERAL, LAMBDA0_GENERAL, METHODE_GENERALE
            kc, tabule, D["variante"] = 1.0, False, False
        if PR.monosymetrique(s):                 # un PRS monosymétrique : § 6.3.2.2 seul, sans autre méthode (brief § 18.5)
            autre = None
    else:
        courbe, lbLT0, beta = COURBE_GENERALE, LAMBDA0_GENERAL, BETA_GENERAL
        seuil = P["lambda_LT0_avec_maintiens"]                   # 6.3.2.2(4) renvoie au λ̄_LT,0 de 6.3.2.3 (ANB : 0,2)
        kc, tabule = 1.0, False
        methode = "§ 6.3.2.2"
    alpha = NO.ALPHA[courbe]
    if retenu == "annexe_E":
        # le tronçon qui gouverne : le plus grand M_Ed/M_b,Rd (k_c = 1 et f = 1 avec des maintiens : moment constant)
        if len(F["troncons"]) > 1:
            kc, tabule = 1.0, False

        def _chi(Mcr_: float, M_: float) -> float:
            lb_ = math.sqrt(Wy * fy / Mcr_)
            if lb_ <= seuil or M_ / Mcr_ <= seuil ** 2:
                return 1.0
            if methode == METHODE_GENERALE:
                return chi_general(lb_, alpha)
            return chi_lamine(lb_, alpha, lbLT0, beta, kc)["chi"]

        gouv = max(F["troncons"], key=lambda t_: t_["M_Ed"] / _chi(t_["M_cr"], t_["M_Ed"]))
        F["gouvernant"] = F["troncons"].index(gouv)
        Mcr, My = gouv["M_cr"], gouv["M_Ed"]
        cles += blocs_e1(J, P, s, gouv["L"], gouv["C1"], gouv["C2"], gouv["C3"], gouv["z_g"], gouv["z_j"], Mcr, "M_cr")
    lbLT = math.sqrt(Wy * fy / Mcr)
    lb0 = math.sqrt(Wy * fy / Mcr0)
    # ---- C₁ de l'étape 1 de l'annexe D ANB, selon l'annexe E ANB (méthode avancée admise : le solveur) : M_cr du tronçon,
    # charges au centre de cisaillement, sur M_cr,0 — calculé seulement si la flexion composée le lira (N_Ed > 0 ou M_z)
    C1 = 1.0
    if d["NEd"] > 0 or G["Mz_calcul"] > 0:
        centre = dev["Mcr"] is None and all(c[-1] == 0.0 for c in (*tr.get("reparties", ()), *tr.get("ponctuelles", ())))
        Mcr_c = Mcr_s if centre else _critique(s, P, _au_centre(tr))["M_cr"]
        C1 = (Mcr_c or Mcr0) / Mcr0
        D["M_cr_centre"] = Mcr_c
    cles.append(J.f("lambda_LT", "λ̄_LT", [_t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("M_cr", Mcr / 1e6, "kN·m")], lbLT, "",
                    "e_lambda_LT"))
    D.update({"cas": CALCULE, "troncon": {"L": tr["L"], "appuis": list(tr["appuis"]), "maintiens": [list(m) for m in tr["maintiens"]]},
              "M_cr": Mcr, "M_cr0": Mcr0, "lambda_LT": lbLT, "lambda_0": lb0, "courbe": courbe, "alpha_LT": alpha, "lambda_LT0": lbLT0,
              "beta": beta, "kc": kc, "kc_tabule": tabule, "C1": C1, "methode": methode, "W_y": Wy})
    nom_chi = "χ_LT"                             # le nom du facteur que M_b,Rd emploie : celui de la ligne qui le donne
    if lbLT <= seuil or My / Mcr <= seuil ** 2:
        D.update({"cas": NEGLIGE, "Phi_LT": None, "chi_LT_brut": 1.0, "f": 1.0, "chi_LT": 1.0})
        chi = 1.0
        messages = [f"Déversement négligeable (λ̄_LT = {_fr(lbLT, 2)} ≤ {_fr(seuil, 2)})." if lbLT <= seuil else
                    "Déversement négligeable (M_Ed/M_cr ≤ λ̄_LT,0²)."]
    else:
        messages = []
        x = chi_lamine(lbLT, alpha, lbLT0, beta, kc)
        Phi, brut, plafond, chi0, f, chi = (x[k] for k in ("Phi", "brut", "plafond", "chi0", "f", "chi"))
        cles.append(J.f("Phi_LT", "Φ_LT", [_t("α_LT", alpha), _t("λ̄_LT", lbLT), _t("λ̄_LT,0", lbLT0), _t("β", beta)], Phi, "", "e_Phi_LT"))
        if brut <= plafond:
            cles.append(J.f("chi_LT", "χ_LT", [_t("Φ_LT", Phi), _t("β", beta), _t("λ̄_LT", lbLT)], chi0, "", "e_chi_LT"))
        else:                                    # χ_LT ≤ 1/λ̄_LT² : le plafond gouverne (grands élancements, β < 1)
            cles.append(J.f("chi_LT", "χ_LT", [_t("λ̄_LT", lbLT)], chi0, "", "e_chi_LT_plafond"))
        if s["forme"] == "I" and methode == "§ 6.3.2.3":          # (tubes et § 6.3.2.2 : k_c = 1, donc f = 1)
            if fl["diagramme"] == LINEAIRE and dev["troncon"] is None and dev["kc"] is None:
                cles.append(J.f("kc", "k_c", [_t("ψ", fl["psi"])], kc, "", "e_kc_lineaire"))
            if f < 1.0:
                cles.append(J.f("f", "f", [_t("k_c", kc), _t("λ̄_LT", lbLT)], f, "", "e_f"))
        if f < 1.0:                              # χ_LT,mod = χ_LT/f, plafonné à 1 et à 1/λ̄_LT² : la ligne dit lequel
            nom_chi = "χ_LT,mod"
            if chi0 / f <= plafond:
                cles.append(J.f("chi_LT_mod", "χ_LT,mod", [_t("χ_LT", chi0), _t("f", f)], chi, "", "e_chi_LT_mod"))
            elif plafond < 1.0:
                cles.append(J.f("chi_LT_mod", "χ_LT,mod", [_t("λ̄_LT", lbLT)], chi, "", "e_chi_LT_plafond"))
            else:
                messages.append("χ_LT/f ≥ 1 : χ_LT,mod = 1.")
        D.update({"Phi_LT": Phi, "chi_LT_brut": chi0, "f": f, "chi_LT": chi, "symbole_chi": nom_chi})
    MbRd = chi * Wy * fy / gM1
    D["M_b_Rd"] = MbRd
    if autre:                                    # la méthode qui ne décide pas : ses chiffres et sa source, sans verdict
        m_, c_, l0_, b_, k_ = autre
        a_ = NO.ALPHA[c_]
        if lbLT <= l0_ or My / Mcr <= l0_ ** 2:
            x_, f_ = 1.0, 1.0
        elif m_ == METHODE_GENERALE:
            x_, f_ = chi_general(lbLT, a_), 1.0
        else:
            y_ = chi_lamine(lbLT, a_, l0_, b_, k_)
            x_, f_ = y_["chi"], y_["f"]
        # (la seule écriture de l'autre méthode : un PRS bisymétrique la lit aussi — _prs ; DÉCISION BUREAU n° 10 pour lui)
        D["information"] = {"methode": m_, "courbe": c_, "alpha_LT": a_, "lambda_LT0": l0_, "beta": b_, "kc": k_, "chi_LT": x_, "f": f_,
                            "variante": autre_variante,
                            "M_b_Rd": x_ * Wy * fy / gM1, "taux": My / (x_ * Wy * fy / gM1),
                            "source": f"{REF} ({m_}) — pour information (DÉCISION BUREAU n° {10 if famille == 'soude' else 4} : {methode} décide)"}
    cles.append(J.f("M_b_Rd", "M_b,Rd", [_t(nom_chi, chi), _t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("γ_M1", gM1)], MbRd / 1e6,
                    "kN·m", "e_M_b_Rd"))
    D["formules"] = cles
    if prs:
        m_, c_ = _prs(J, d, D, tr, signe)
        messages += m_
        cles += c_
    messages = lignes_mcr(D["mcr"]) + messages
    J.verif("E-Mb", "Déversement", f"{REF} ({methode})", "M_y,Ed ≤ M_b,Rd", My / 1e6, MbRd / 1e6, "kN·m", cles, messages=messages)
    niveau = "au centre de cisaillement" if not dev["zg"] else ("sur la semelle haute" if dev["zg"] > 0 else "sous la semelle basse")
    # (le sens se dit quand il compte : un PRS monosymétrique, ou M_y,Ed < 0 — celui que le plus petit M_cr a retenu)
    sens = " montante" if sens_charges(tr) < 0 else (" descendante" if PR.monosymetrique(s) or inconnu else "")
    charge = f", charge{sens} {niveau}" if (tr.get("reparties") or tr.get("ponctuelles")) and dev["troncon"] is None else ""
    source = ""
    if methode == "§ 6.3.2.3":
        source = " (variante de l'ANB:2018, 6.3.2.3(1))." if D.get("variante") else " (ANB:2018, 6.3.2.3(1))."
    # (lot C, C5 : une ligne par fait, 120 caractères au plus — la hauteur de la charge et la variante du couple à part)
    J.hyp.append(f"Déversement : tronçon de {_fr(tr['L'] / 1e3, 2)} m ; {methode}, courbe {courbe}, "
                 f"λ̄_LT,0 = {_fr(lbLT0, 2)}, β = {_fr(beta, 2)}" + (source or "."))
    if charge:
        J.hyp.append(f"M_cr : charge{charge[len(', charge'):]}.")
    if methode == "§ 6.3.2.3" and D.get("variante"):
        J.hyp.append("Variante du couple : poutre de bâtiment, maintiens ignorés dans M_cr.")
    if "sens_charge" in D:
        sc = D["sens_charge"]
        J.hyp.append(f"M_y,Ed < 0, sens de la charge inconnu : le plus petit M_cr des deux sens (montante {_mcr_txt(sc['montante'])}, "
                     f"descendante {_mcr_txt(sc['descendante'])}). À VÉRIFIER.")


# ------------------------------------------------------------------ M_cr par la formule (DÉCISION BUREAU n° 12)
GARDE_FORMULE = 1.02             # la formule dépasse le solveur de plus de 2 % : le solveur est retenu (mise à jour 1, § 1.2)


def _moment_simple(L: float, x: float, rep, pon) -> List[float]:
    """Le moment en ``x`` d'une poutre sur deux appuis de portée ``L`` (mm), charge par charge (N·mm) : la part de chacune
    dans M_y — réparties (x1, x2, q1, q2, z) intégrées par tranches, ponctuelles (x, P, z)."""
    parts = []
    for r_ in rep:
        x1, x2, q1, q2 = (r_[0], r_[1], r_[2], r_[3]) if len(r_) == 5 else (r_[0], r_[1], r_[2], r_[2])   # (x1, x2, q, z) : uniforme
        n, m = 40, 0.0
        for k in range(n):
            u = x1 + (x2 - x1) * (k + 0.5) / n
            P = (q1 + (q2 - q1) * (k + 0.5) / n) * (x2 - x1) / n
            m += P * (L - u) * x / L if x <= u else P * u * (L - x) / L
        parts.append(m)
    for u, P, _z in pon:
        parts.append(P * (L - u) * x / L if x <= u else P * u * (L - x) / L)
    return parts


def formule_annexe_e(d: Dict, tr: Dict, s: Dict, P: Dict) -> Dict:
    """M_cr par (E.1) de l'annexe E ANB, k_z = k_ω = 1 (mise à jour 1, § 1.2) : {"domaine", "raison" (hors domaine),
    "troncons" [{"a", "b" (mm, depuis le début du tronçon de la barre), "L", "M_Ed", "M_cr", "C1", "C2", "C3", "z_g", "z_j",
    "diagramme"}]}. Domaine : deux appuis à fourche, sans débord, moment d'un seul signe. Les tronçons vont d'une section
    tenue à la suivante (appui à fourche, maintien sur la semelle comprimée ; un maintien de la semelle tendue ne coupe
    pas) : avec maintiens, chacun sous un moment CONSTANT égal à son |M_y|max (C₁ = 1, C₂ = 0, C₃ = 1) ; sans maintien, la
    portée avec les coefficients de son diagramme (tableaux E.1 et E.2 ANB) et z_g = la hauteur des charges, moyenne
    pondérée par la part de chacune dans |M_y|max."""
    dev, fl = d["deversement"], d["flexion"]["y"]

    def hors(raison: str) -> Dict:
        return {"domaine": False, "raison": raison, "troncons": []}

    if d.get("console"):
        return hors("console")
    if tuple(tr["appuis"]) != (MC.FOURCHE, MC.FOURCHE):
        return hors("appuis qui ne sont pas des fourches")
    if dev["troncon"] is not None and not dev["troncon"].get("deux_appuis"):
        return hors("barre qui n'est pas sur deux appuis sans débord")
    L = tr["L"]
    xs = [L * i / 400.0 for i in range(401)]
    Ms = [tr["moment"](x) for x in xs]
    Mmax = max(abs(m) for m in Ms)
    if not Mmax > 0:
        return hors("moment nul")
    if min(Ms) < -1e-6 * Mmax and max(Ms) > 1e-6 * Mmax:
        return hors("moment qui change de signe")
    sens = 1.0 if max(Ms) > 1e-6 * Mmax else -1.0              # > 0 : la semelle haute est comprimée
    zp = PR.zj_psi(s, PR.HAUT if sens > 0 else PR.BAS)
    EIz, EIw, GIt = P["E"] * s["Iz"], P["E"] * (s.get("Iw") or 0.0), P["G"] * s["It"]
    coupes = sorted(x for x, z in tr["maintiens"] if z * sens > 0 and 1e-6 * L < x < L * (1.0 - 1e-6))
    if coupes:
        bornes = [0.0] + coupes + [L]
        trs = []
        for a, b in zip(bornes[:-1], bornes[1:]):
            M_Ed = max(abs(tr["moment"](a + (b - a) * k / 100.0)) for k in range(101))
            Mcr = MC.annexe_e(b - a, EIz, EIw, GIt, 1.0, 0.0, 1.0, 0.0, zp["z_j"])
            trs.append({"a": a, "b": b, "L": b - a, "M_Ed": M_Ed, "M_cr": Mcr, "C1": 1.0, "C2": 0.0, "C3": 1.0, "z_g": 0.0, "z_j": zp["z_j"],
                        "diagramme": "constant"})
        return {"domaine": True, "raison": None, "troncons": trs}
    diag = dev.get("diagramme") or (fl["diagramme"] if dev["troncon"] is None else None)
    if not diag:
        return hors("diagramme hors des tableaux")
    charge = diag in MC.E2
    co = MC.coefficients_annexe_e(diag, fl["psi"], zp["psi_f"], None if charge else MC.kwt(L, EIw, GIt))
    if co["bloque"]:
        return hors(co["bloque"])
    zg = 0.0
    if charge:
        rep, pon = tr.get("reparties") or (), tr.get("ponctuelles") or ()
        if rep or pon:
            xm = xs[max(range(len(xs)), key=lambda k: abs(Ms[k]))]
            parts = _moment_simple(L, xm, rep, pon)
            cotes = [r_[-1] for r_ in rep] + [p_[-1] for p_ in pon]
            total = sum(parts)
            zg = sum(w * z for w, z in zip(parts, cotes)) / total if abs(total) > 0 else cotes[0]
        else:
            zg = dev["zg"]
        zg *= sens_charges(tr)
    Mcr = MC.annexe_e(L, EIz, EIw, GIt, co["C1"], co["C2"], co["C3"], zg, zp["z_j"])
    return {"domaine": True, "raison": None, "troncons": [{"a": 0.0, "b": L, "L": L, "M_Ed": Mmax, "M_cr": Mcr, "C1": co["C1"], "C2": co["C2"],
                                                            "C3": co["C3"], "z_g": zg, "z_j": zp["z_j"], "diagramme": diag}]}


QUOI_DIAGRAMME = {"constant": "moment constant", "lineaire": "moment linéaire", "reparti": "charge répartie", "ponctuel": "charge au milieu",
                  "quarts": "charges aux quarts"}


def blocs_e1(J: Journal, P: Dict, s: Dict, L: float, C1: float, C2: float, C3: float, zg: float, zj: float, Mcr: float,
             symbole: str) -> List:
    """(E.1) de l'annexe E ANB écrite en DEUX blocs (k_z = k_ω = 1) : N_cr,z = π²·E·I_z/(k_z·L)², puis
    M_cr = C₁·N_cr,z·[√((k_z/k_ω)²·I_ω/I_z + G·I_t/N_cr,z + (C₂·z_g − C₃·z_j)²) − (C₂·z_g − C₃·z_j)] — la même valeur que
    MC.annexe_e (``Mcr``, N·mm), écrite pour tenir dans une colonne de la note."""
    Ncr = math.pi ** 2 * P["E"] * s["Iz"] / L ** 2
    n = J.f("N_cr_E1", "N_cr,z", [_t("E", P["E"], "N/mm²"), _t("I_z", s["Iz"] / 1e4, "cm⁴"), _t("k_z", 1.0), _t("L", L / 1e3, "m")],
            Ncr / 1e3, "kN", "p_N_cr_E1")
    m = J.f("M_cr_E1", symbole, [_t("C₁", C1), _t("N_cr,z", Ncr / 1e3, "kN"), _t("k_z", 1.0), _t("k_ω", 1.0),
                                 _t("I_ω", (s.get("Iw") or 0.0) / 1e6, "cm⁶"), _t("I_z", s["Iz"] / 1e4, "cm⁴"), _t("G", P["G"], "N/mm²"),
                                 _t("I_t", s["It"] / 1e4, "cm⁴"), _t("C₂", C2), _t("z_g", zg, "mm"), _t("C₃", C3), _t("z_j", zj, "mm")],
            Mcr / 1e6, "kN·m", "p_M_cr_E1")
    return [n, m]


def lignes_mcr(m: Dict) -> List[str]:
    """Les deux lignes du M_cr (DÉCISION BUREAU n° 12), à l'écran et dans la note : celui qui est retenu, puis l'autre et
    son écart (ou pourquoi le solveur est retenu)."""
    F = m.get("formule")
    if F is None:                                            # (un M_cr donné : rien à comparer)
        return []
    Ms = m["solveur"]
    ou_s = "poutre entière" if m.get("entier") else "tronçon"
    if not F["domaine"]:
        return ["Formule hors domaine : solveur."]                 # (le texte du brief ; la raison est dans D["mcr"])
    k = F.get("gouvernant")
    if k is None:                                            # le solveur décide : le tronçon le plus critique de la formule
        k = min(range(len(F["troncons"])), key=lambda i: F["troncons"][i]["M_cr"] / F["troncons"][i]["M_Ed"] if F["troncons"][i]["M_Ed"] > 0 else 1e99)
    g = F["troncons"][k]
    x0 = m.get("x0", 0.0)
    ou_f = f"tronçon {_fr(x0 + g['a'] / 1e3, 2)} à {_fr(x0 + g['b'] / 1e3, 2)} m, {QUOI_DIAGRAMME.get(g['diagramme'], g['diagramme'])}"

    def ecart(a: float, b: float) -> str:
        e = round((a / b - 1.0) * 100.0)
        return f"{'+' if e >= 0 else '−'}{abs(e)} %"

    if m["retenu"] == "annexe_E":
        return [f"M_cr retenu : formule, {ou_f} = {_fr(g['M_cr'] / 1e6, 1)} kN·m.",
                f"Solveur ({ou_s}) : {_fr(Ms / 1e6, 1)} kN·m ({ecart(Ms, g['M_cr'])})."]
    tete = "Formule au-dessus du solveur : solveur retenu." if m.get("au_dessus") else f"M_cr retenu : solveur ({ou_s}) = {_fr(Ms / 1e6, 1)} kN·m."
    return [tete, f"Formule, {ou_f} : {_fr(g['M_cr'] / 1e6, 1)} kN·m ({ecart(g['M_cr'], Ms)})."]


# ------------------------------------------------------------------ les PRS (brief § 18.5)
def _mcr_txt(Mcr: Optional[float]) -> str:
    return f"{_fr(Mcr / 1e6, 1)} kN·m" if Mcr else "sans déversement"


def sens_charges(tr: Dict) -> float:
    """+1 quand la résultante des charges du tronçon DESCEND, −1 quand elle monte (un soulèvement)."""
    total = sum((r[2] + r[3]) / 2.0 * (r[1] - r[0]) if len(r) == 5 else r[2] * (r[1] - r[0]) for r in tr.get("reparties") or ())
    total += sum(p[1] for p in tr.get("ponctuelles") or ())
    return -1.0 if total < 0 else 1.0


def _prs(J: Journal, d: Dict, D: Dict, tr: Dict, signe: float) -> Tuple[list, list]:
    """Ce que le déversement d'un PRS ajoute (brief § 18.5 et 18.7) : le b retenu pour h/b, l'autre méthode (PRS
    bisymétrique : celle que ajouter() a écrite dans ``D["information"]``, une seule source) ou son refus
    (monosymétrique), le contrôle analytique (E.1) de l'annexe E ANB. Rend (les messages de E-Mb, les clés des blocs à
    lui ajouter). ``D["variante"]`` reste celui de ajouter() : la variante (0,4 ; 0,75) du couple, appliquée ou non."""
    s, P = d["section"], d["parametres"]
    msgs, cles = [], []
    D["b_h_b"] = s["b"]
    D["soudee_equivalente"] = bool(P["prs_soudee_equivalente"]) and not s["monosymetrique"]
    if s["monosymetrique"]:
        J.hyp.append(f"PRS monosymétrique : h/b = {PR.mm(s['h'])}/{PR.mm(s['b'])} = {_fr(s['h'] / s['b'], 2)} avec b = la plus petite largeur de "
                     f"semelle (choix prudent) ; § 6.3.2.2 seul.")
        msgs.append(f"{METHODE_LAMINES} refusé : section monosymétrique.")
    else:
        info = D["information"]
        couple = f" (variante {FMT.nb(info['lambda_LT0'])} ; {FMT.nb(info['beta'])})" if info["variante"] else ""
        msgs.append(f"{info['methode']}{couple} pour information : M_b,Rd = {_fr(info['M_b_Rd'] / 1e6, 1)} kN·m.")
        J.hyp.append("PRS bisymétrique déclaré « section soudée équivalente » : § 6.3.2.3 (décision du bureau)." if D["soudee_equivalente"] else
                     "PRS : § 6.3.2.2 (défaut) ; § 6.3.2.3 seulement si la section est déclarée « soudée équivalente ».")
    # ---- le contrôle analytique : (E.1) de l'annexe E ANB (le solveur reste la référence)
    dev, fl = d["deversement"], d["flexion"]["y"]
    diag = dev.get("diagramme") or (fl["diagramme"] if dev["troncon"] is None else None)
    simple = tr["appuis"] == (MC.FOURCHE, MC.FOURCHE) and not tr["maintiens"]
    if diag and simple:
        semelle = PR.HAUT if signe > 0 else PR.BAS
        zp = PR.zj_psi(s, semelle)
        EIz, EIw, GIt = P["E"] * s["Iz"], P["E"] * s["Iw"], P["G"] * s["It"]
        charge = diag in MC.E2
        co = MC.coefficients_annexe_e(diag, fl["psi"], zp["psi_f"], None if charge else MC.kwt(tr["L"], EIw, GIt))
        E: Dict = {"diagramme": diag, "psi_f": zp["psi_f"], "z_j": zp["z_j"], "z_j_approche": zp["z_j_approche"], **co}
        zgs = {c_[-1] for c_ in (*tr.get("reparties", ()), *tr.get("ponctuelles", ()))}
        F_ = (D.get("mcr") or {}).get("formule")
        dit = bool(F_ and F_["domaine"])                          # (la décision n° 12 dit déjà la formule et le solveur)
        if co["bloque"]:
            if not dit:
                msgs.append(f"Contrôle (E.1) bloqué : {co['bloque']}.")
        elif charge and len(zgs) > 1:
            # (lot C, C1) des charges à plusieurs hauteurs (le poids propre au centre de gravité, les autres sur une semelle) :
            # (E.1) n'a qu'un z_g — le solveur seul (brief § 7.7 et 18.5), dit en une ligne
            if not dit:
                msgs.append("Contrôle (E.1) : charges à plusieurs hauteurs — solveur seul.")
        else:
            # z_g de l'annexe E ANB, 2(8) : positif quand la charge va VERS le centre de cisaillement depuis son point
            # d'application — la cote de la charge, retournée si elle monte (un soulèvement, en barre chargée)
            zg = dev["zg"] * sens_charges(tr) if charge else 0.0
            Mcr_E = MC.annexe_e(tr["L"], EIz, EIw, GIt, co["C1"], co["C2"], co["C3"], zg, zp["z_j"])
            E.update({"z_g": zg, "M_cr": Mcr_E, "ecart": Mcr_E / (D.get("mcr") or {}).get("solveur", D["M_cr"]) - 1.0})
            if not (dit and D["mcr"]["retenu"] == "annexe_E"):
                # (mise à jour 1) la formule RETENUE a déjà son bloc « M_cr_E1 » (décision n° 12) : pas une seconde fois
                bl = blocs_e1(J, P, s, tr["L"], co["C1"], co["C2"], co["C3"], zg, zp["z_j"], Mcr_E, "M_cr,E.1")
                E["formule"] = bl[-1]
                cles += bl
            ecart = round(E["ecart"] * 100, 2) + 0.0                # (arrondi d'abord : un écart nul s'écrit « +0,00 % »)
            if not dit:
                msgs.append(f"Contrôle (E.1) : M_cr = {_fr(Mcr_E / 1e6, 1)} kN·m ({'+' if ecart >= 0 else '−'}{_fr(abs(ecart), 2)} %).")
        D["annexe_E"] = E
    if s["monosymetrique"]:
        J.hyp.append(f"PRS monosymétrique, semelle {'haute' if signe > 0 else 'basse'} comprimée : ψ_f = "
                     f"{_fr(PR.zj_psi(s, PR.HAUT if signe > 0 else PR.BAS)['psi_f'], 4)}, z_s = {_fr(s['z_s'], 1)} mm, "
                     f"z_j = {_fr(PR.zj_psi(s, PR.HAUT if signe > 0 else PR.BAS)['z_j'], 2)} mm (Wagner).")
    return msgs, cles

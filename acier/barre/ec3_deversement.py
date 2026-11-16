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
(``deversement.maintiens_negliges`` ≥ 1, aucun maintien dans le tronçon du solveur) ; ailleurs elle est refusée, avec
la raison, et le couple par défaut s'applique (brief § 7.7, test 9 du § 11.5). Le couple employé est nommé.

Pour l'annexe D ANB (flexion composée), ce fichier rend aussi M_cr,0 — le moment critique du même tronçon sous moment
CONSTANT — λ̄_0 = √(W_y·f_y/M_cr,0) et C₁ selon l'annexe E ANB, qui admet les méthodes avancées : le rapport
M_cr/M_cr,0 du solveur, les charges ramenées au centre de cisaillement (brief § 7.8 ; ANB, note de l'annexe A : la
règle C₁ = k_c⁻² de l'AC:2009 ne couvre qu'incomplètement les situations — elle n'est pas employée).
"""
import math
from typing import Callable, Dict, Optional, Tuple

from acier.flambement import normes as NO

from . import mcr as MC
from . import parametres as PA
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


def troncon_impose(L: float, M: float, diagramme: str, psi: float, zg: float) -> Dict:
    """Le tronçon de déversement du mode « efforts imposés » : entre deux fourches, le moment maximal ``M`` (N·mm) et
    la forme de son diagramme ; les charges qui le produisent sont appliquées à la cote ``zg``."""
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
    return {"L": L, "moment": moment, "M_max": abs(M), "appuis": (MC.FOURCHE, MC.FOURCHE), "maintiens": [], **charges}


def _critique(s: Dict, P: Dict, tr: Dict, moment: Optional[Callable[[float], float]] = None) -> Dict:
    """M_cr du tronçon ``tr`` ; avec ``moment``, sous ce diagramme et SANS charge transversale (le moment constant)."""
    EIz, EIw, GIt = P["E"] * s["Iz"], P["E"] * (s.get("Iw") or 0.0), P["G"] * s["It"]
    if moment is not None:
        return MC.moment_critique(tr["L"], EIz, EIw, GIt, moment, appuis=tr["appuis"], maintiens=tr["maintiens"], M_max=1.0)
    return MC.moment_critique(tr["L"], EIz, EIw, GIt, tr["moment"], reparties=tr.get("reparties", ()), ponctuelles=tr.get("ponctuelles", ()),
                              appuis=tr["appuis"], maintiens=tr["maintiens"], M_max=tr.get("M_max"))


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


def couple_anb(dev: Dict, tr: Dict, ouvrage: str) -> Tuple[bool, Optional[str]]:
    """(variante appliquée, motif du refus) — la variante (0,4 ; 0,75) de NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1), n'est
    permise que pour une poutre de BÂTIMENT avec des maintiens, à condition qu'ils soient TOTALEMENT ignorés dans M_cr :
    au moins un maintien « négligé », et aucun dans le tronçon que le solveur calcule. Demandée ailleurs, elle est
    refusée avec sa raison (le couple par défaut s'applique)."""
    if dev.get("couple", "defaut") != "variante":
        return False, None
    if ouvrage != "batiment":
        return False, "la variante (0,4 ; 0,75) n'est admise que pour une poutre de bâtiment"
    if not dev.get("maintiens_negliges"):
        return False, "la variante (0,4 ; 0,75) suppose des maintiens ; sans maintien, le couple (0,2 ; 1,0) s'applique"
    if tr["maintiens"]:
        return False, "la variante (0,4 ; 0,75) suppose TOUS les maintiens ignorés dans M_cr ; ce tronçon en compte"
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
    tr = dev["troncon"] or troncon_impose(dev["Lc"], My, fl["diagramme"], fl["psi"], dev["zg"])
    kc, tabule = (dev["kc"], True) if dev["kc"] is not None else coefficient_kc(fl["diagramme"] if dev["troncon"] is None else GENERAL, fl["psi"])
    nomW = {1: "W_pl,y", 2: "W_pl,y", 3: "W_el,y", 4: "W_eff,y"}[G["classe"]]
    cles = []
    # ---- le moment critique
    if dev["Mcr"] is not None:
        Mcr = dev["Mcr"]
        D["Mcr_donne"] = True
        J.hyp.append(f"Moment critique donné : M_cr = {_fr(Mcr / 1e6, 1)} kN·m.")
    else:
        r = _critique(s, P, tr)
        Mcr = r["M_cr"]
        if Mcr is None:
            D["cas"] = EMPECHE
            J.hyp.append("Déversement empêché par les maintiens du tronçon (aucun moment critique).")
            return
        D.update({"lambda_cr": r["lambda"], "M_max": r["M_max"]})
        cles.append(J.f("M_cr", "M_cr", [_t("λ_cr", r["lambda"]), _t("|M_y|max", r["M_max"] / 1e6, "kN·m")], Mcr / 1e6, "kN·m", "e_M_cr"))
    simple = tr["appuis"] == (MC.FOURCHE, MC.FOURCHE) and not tr["maintiens"]
    if simple:
        Mcr0 = MC.moment_constant(tr["L"], P["E"] * s["Iz"], P["E"] * (s.get("Iw") or 0.0), P["G"] * s["It"])
    else:
        Mcr0 = _critique(s, P, tr, moment=lambda x: 1.0)["M_cr"] or Mcr
    lbLT = math.sqrt(Wy * fy / Mcr)
    lb0 = math.sqrt(Wy * fy / Mcr0)
    # ---- C₁ de l'étape 1 de l'annexe D ANB, selon l'annexe E ANB (méthode avancée admise : le solveur) : M_cr du tronçon,
    # charges au centre de cisaillement, sur M_cr,0 — calculé seulement si la flexion composée le lira (N_Ed > 0 ou M_z)
    C1 = 1.0
    if d["NEd"] > 0 or G["Mz_calcul"] > 0:
        centre = dev["Mcr"] is None and all(c[-1] == 0.0 for c in (*tr.get("reparties", ()), *tr.get("ponctuelles", ())))
        Mcr_c = Mcr if centre else _critique(s, P, _au_centre(tr))["M_cr"]
        C1 = (Mcr_c or Mcr0) / Mcr0
        D["M_cr_centre"] = Mcr_c
    cles.append(J.f("lambda_LT", "λ̄_LT", [_t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("M_cr", Mcr / 1e6, "kN·m")], lbLT, "",
                    "e_lambda_LT"))
    # ---- la courbe et le couple (λ̄_LT,0 ; β) de l'ANB:2018, 6.3.2.3(1) : (0,2 ; 1,0), ou la variante si elle est permise
    D["a_confirmer"] = [k for k in ("lambda_LT0_avec_maintiens", "beta_avec_maintiens") if PA.PARAMETRES[k]["statut"] == PA.A_CONFIRMER]
    famille, autre = None, None
    if s["forme"] == "I":
        famille = "soude" if s["fabrication"] in SOUDES else "lamine"
        if not s.get("h") or not s.get("b"):
            raise Refus("h et b manquent : la courbe de déversement en dépend")
        elance = s["h"] / s["b"] > ELANCE
        courbe = COURBES_LT[famille][elance]
        lbLT0, beta = P["lambda_LT0_avec_maintiens"], P["beta_avec_maintiens"]
        variante, refus = couple_anb(dev, tr, d["ouvrage"])
        if variante:
            lbLT0, beta = P["lambda_LT0_variante"], P["beta_variante"]
        elif refus:
            J.alerte("variante_refusee", "attention", f"Variante du déversement refusée : {refus}.")
        D["variante"] = variante
        seuil = lbLT0
        methode = "§ 6.3.2.3"
        # l'AUTRE méthode, pour information (DÉCISION BUREAU n° 4, brief § 7.7) : (méthode, courbe, λ̄_LT,0, β, k_c)
        autre = ("§ 6.3.2.2", COURBES_GENERALES[famille][elance], LAMBDA0_GENERAL, BETA_GENERAL, 1.0)
        if famille == "lamine" and P["methode_deversement_lamines"] == "6.3.2.2":     # le défaut changé dans la table
            autre = (methode, courbe, lbLT0, beta, kc)
            courbe, lbLT0, beta, seuil, methode = COURBES_GENERALES[famille][elance], LAMBDA0_GENERAL, BETA_GENERAL, LAMBDA0_GENERAL, "§ 6.3.2.2"
            kc, tabule, D["variante"] = 1.0, False, False
    else:
        courbe, lbLT0, beta = COURBE_GENERALE, LAMBDA0_GENERAL, BETA_GENERAL
        seuil = P["lambda_LT0_avec_maintiens"]                   # 6.3.2.2(4) renvoie au λ̄_LT,0 de 6.3.2.3 (ANB : 0,2)
        kc, tabule = 1.0, False
        methode = "§ 6.3.2.2"
    alpha = NO.ALPHA[courbe]
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
        x_ = 1.0 if (lbLT <= l0_ or My / Mcr <= l0_ ** 2) else (chi_general(lbLT, a_) if m_ == "§ 6.3.2.2" else chi_lamine(lbLT, a_, l0_, b_, k_)["chi"])
        D["information"] = {"methode": m_, "courbe": c_, "alpha_LT": a_, "lambda_LT0": l0_, "beta": b_, "kc": k_, "chi_LT": x_,
                            "M_b_Rd": x_ * Wy * fy / gM1, "taux": My / (x_ * Wy * fy / gM1),
                            "source": f"{REF} ({m_}) — pour information (DÉCISION BUREAU n° 4 : {methode} décide)"}
    cles.append(J.f("M_b_Rd", "M_b,Rd", [_t(nom_chi, chi), _t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("γ_M1", gM1)], MbRd / 1e6,
                    "kN·m", "e_M_b_Rd"))
    D["formules"] = cles
    J.verif("E-Mb", "Déversement", f"{REF} ({methode})", "M_y,Ed ≤ M_b,Rd", My / 1e6, MbRd / 1e6, "kN·m", cles, messages=messages)
    niveau = "au centre de cisaillement" if not dev["zg"] else ("sur la semelle haute" if dev["zg"] > 0 else "sous la semelle basse")
    charge = f", charge {niveau}" if (tr.get("reparties") or tr.get("ponctuelles")) and dev["troncon"] is None else ""
    source = ""
    if methode == "§ 6.3.2.3":
        source = " (variante de l'ANB:2018, 6.3.2.3(1) : bâtiment, maintiens ignorés dans M_cr)." if D.get("variante") else " (ANB:2018, 6.3.2.3(1))."
    J.hyp.append(f"Déversement : tronçon de {_fr(tr['L'] / 1e3, 2)} m entre sections tenues{charge} ; {methode}, courbe {courbe}, "
                 f"λ̄_LT,0 = {_fr(lbLT0, 2)}, β = {_fr(beta, 2)}" + (source or "."))

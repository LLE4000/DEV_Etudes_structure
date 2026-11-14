# -*- coding: utf-8 -*-
"""EUROCODE 3, le DÉVERSEMENT d'une barre fléchie autour de son axe fort (NBN EN 1993-1-1 § 6.3.2).
docs/acier/BARRES_ACIER.md § 5.6.

    M_cr : numérique, par TRONÇON entre deux sections tenues en déplacement latéral et en torsion (acier/barre/mcr.py)
    λ̄_LT = √(W_y·f_y / M_cr)        W_y : W_pl,y (classes 1 et 2), W_el,y (3), W_eff,y (4)
    I et H, laminés ou soudés (§ 6.3.2.3) :
        Φ_LT = 0,5·[1 + α_LT·(λ̄_LT − λ̄_LT,0) + β·λ̄_LT²]       χ_LT = 1/(Φ_LT + √(Φ_LT² − β·λ̄_LT²)) ≤ 1 et ≤ 1/λ̄_LT²
        tableau 6.5 : laminé h/b ≤ 2 : b ; > 2 : c — soudé h/b ≤ 2 : c ; > 2 : d
        f = 1 − 0,5·(1 − k_c)·[1 − 2·(λ̄_LT − 0,8)²] ≤ 1        χ_LT,mod = χ_LT/f ≤ 1 et ≤ 1/λ̄_LT²
        k_c (tableau 6.6) : moment constant 1 ; linéaire 1/(1,33 − 0,33·ψ) ; charge répartie sur deux appuis 0,94 ;
        charge concentrée à mi-portée 0,86 ; tout autre diagramme : k_c = 1 (f = 1, du côté sûr)
    tube rectangulaire (cas général, § 6.3.2.2) : courbe d, λ̄_LT,0 = 0,2 et β = 1 dans Φ_LT, f = 1
    M_b,Rd = χ_LT,mod·W_y·f_y / γ_M1
    négligé (χ_LT = 1) si λ̄_LT ≤ λ̄_LT,0 ou M_Ed/M_cr ≤ λ̄_LT,0²                                        (§ 6.3.2.2(4))

Tube rond ou carré : pas de déversement. Semelle comprimée tenue en continu (« Lc » nul, ou « actif » faux) : pas de
déversement, et c'est dit.

PARAMÈTRES NATIONAUX (acier/barre/parametres.py), À CONFIRMER sur le texte de l'annexe belge : (λ̄_LT,0 ; β) vaut
(0,4 ; 0,75) quand M_cr ignore les maintiens intermédiaires — un tronçon nu entre deux sections tenues —, et (0,2 ; 1,0)
quand M_cr en tient compte (un maintien d'une seule semelle à l'intérieur du tronçon). Le couple employé est nommé.

Pour l'annexe A (flexion composée), ce fichier rend aussi M_cr,0 — le moment critique du même tronçon sous moment
CONSTANT — λ̄_0 = √(W_y·f_y/M_cr,0) et C₁ = 1/k_c².
"""
import math
from typing import Callable, Dict, Optional, Tuple

from acier.flambement import normes as NO

from . import mcr as MC
from .resultat import Journal, Refus, fr as _fr, terme as _t

REF = "EN 1993-1-1 § 6.3.2"
CONSTANT, LINEAIRE, REPARTI, PONCTUEL, GENERAL = "constant", "lineaire", "reparti", "ponctuel", "general"
DIAGRAMMES = (CONSTANT, LINEAIRE, REPARTI, PONCTUEL, GENERAL)
KC_REPARTI, KC_PONCTUEL = 0.94, 0.86             # tableau 6.6
ELANCE = 2.0                                     # h/b : la limite du tableau 6.5
COURBES_LT = {"lamine": ("b", "c"), "soude": ("c", "d")}        # (h/b ≤ 2, h/b > 2)
COURBE_GENERALE, LAMBDA0_GENERAL, BETA_GENERAL = "d", 0.2, 1.0  # § 6.3.2.2, « autres sections »
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


def ajouter(J: Journal, d: Dict) -> None:
    """Le déversement (``G["deversement"]``) et la vérification E-Mb."""
    G, s = J.G, d["section"]
    if not G.get("couvert"):
        return
    My = G["My_calcul"]
    dev = d["deversement"]
    D: Dict = {"cas": SANS_OBJET, "chi_LT": 1.0, "lambda_0": 0.0, "C1": 1.0, "kc": 1.0}
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
    cles.append(J.f("lambda_LT", "λ̄_LT", [_t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("M_cr", Mcr / 1e6, "kN·m")], lbLT, "",
                    "e_lambda_LT"))
    # ---- la courbe et le couple (λ̄_LT,0 ; β)
    avec = bool(tr["maintiens"])
    if s["forme"] == "I":
        famille = "soude" if s["fabrication"] in SOUDES else "lamine"
        if not s.get("h") or not s.get("b"):
            raise Refus("h et b manquent : la courbe de déversement en dépend")
        courbe = COURBES_LT[famille][s["h"] / s["b"] > ELANCE]
        lbLT0 = P["lambda_LT0_avec_maintiens" if avec else "lambda_LT0_sans_maintiens"]
        beta = P["beta_avec_maintiens" if avec else "beta_sans_maintiens"]
        seuil = lbLT0
        D["a_confirmer"] = ["lambda_LT0_avec_maintiens", "beta_avec_maintiens"] if avec else ["lambda_LT0_sans_maintiens", "beta_sans_maintiens"]
        methode = "§ 6.3.2.3"
    else:
        courbe, lbLT0, beta = COURBE_GENERALE, LAMBDA0_GENERAL, BETA_GENERAL
        seuil = P["lambda_LT0_sans_maintiens"]
        D["a_confirmer"] = ["lambda_LT0_sans_maintiens"]
        kc, tabule = 1.0, False
        methode = "§ 6.3.2.2"
    alpha = NO.ALPHA[courbe]
    D.update({"cas": CALCULE, "troncon": {"L": tr["L"], "appuis": list(tr["appuis"]), "maintiens": [list(m) for m in tr["maintiens"]]},
              "M_cr": Mcr, "M_cr0": Mcr0, "lambda_LT": lbLT, "lambda_0": lb0, "courbe": courbe, "alpha_LT": alpha, "lambda_LT0": lbLT0,
              "beta": beta, "kc": kc, "kc_tabule": tabule, "C1": 1.0 / kc ** 2, "methode": methode, "W_y": Wy})
    if lbLT <= seuil or My / Mcr <= seuil ** 2:
        D.update({"cas": NEGLIGE, "Phi_LT": None, "chi_LT_brut": 1.0, "f": 1.0, "chi_LT": 1.0})
        chi = 1.0
        messages = [f"Déversement négligeable (λ̄_LT = {_fr(lbLT, 2)} ≤ {_fr(seuil, 2)})." if lbLT <= seuil else
                    "Déversement négligeable (M_Ed/M_cr ≤ λ̄_LT,0²)."]
    else:
        Phi = 0.5 * (1.0 + alpha * (lbLT - lbLT0) + beta * lbLT ** 2)
        brut = 1.0 / (Phi + math.sqrt(Phi ** 2 - beta * lbLT ** 2))
        plafond = min(1.0, 1.0 / lbLT ** 2)
        chi0 = min(brut, plafond)
        cles.append(J.f("Phi_LT", "Φ_LT", [_t("α_LT", alpha), _t("λ̄_LT", lbLT), _t("λ̄_LT,0", lbLT0), _t("β", beta)], Phi, "", "e_Phi_LT"))
        if brut <= plafond:
            cles.append(J.f("chi_LT", "χ_LT", [_t("Φ_LT", Phi), _t("β", beta), _t("λ̄_LT", lbLT)], chi0, "", "e_chi_LT"))
        else:                                    # χ_LT ≤ 1/λ̄_LT² : le plafond gouverne (grands élancements, β < 1)
            cles.append(J.f("chi_LT", "χ_LT", [_t("λ̄_LT", lbLT)], chi0, "", "e_chi_LT_plafond"))
        f = 1.0
        if s["forme"] == "I":
            if fl["diagramme"] == LINEAIRE and dev["troncon"] is None and dev["kc"] is None:
                cles.append(J.f("kc", "k_c", [_t("ψ", fl["psi"])], kc, "", "e_kc_lineaire"))
            f = min(1.0 - 0.5 * (1.0 - kc) * (1.0 - 2.0 * (lbLT - 0.8) ** 2), 1.0)
            if f < 1.0:
                cles.append(J.f("f", "f", [_t("k_c", kc), _t("λ̄_LT", lbLT)], f, "", "e_f"))
        chi = min(chi0 / f, plafond)
        if f < 1.0 and chi0 / f <= plafond:
            cles.append(J.f("chi_LT_mod", "χ_LT,mod", [_t("χ_LT", chi0), _t("f", f)], chi, "", "e_chi_LT_mod"))
        D.update({"Phi_LT": Phi, "chi_LT_brut": chi0, "f": f, "chi_LT": chi})
        messages = []
    MbRd = chi * Wy * fy / gM1
    D["M_b_Rd"] = MbRd
    cles.append(J.f("M_b_Rd", "M_b,Rd", [_t("χ_LT", chi), _t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("γ_M1", gM1)], MbRd / 1e6,
                    "kN·m", "e_M_b_Rd"))
    D["formules"] = cles
    J.verif("E-Mb", "Déversement", f"{REF} ({methode})", "M_y,Ed ≤ M_b,Rd", My / 1e6, MbRd / 1e6, "kN·m", cles, messages=messages)
    niveau = "au centre de cisaillement" if not dev["zg"] else ("sur la semelle haute" if dev["zg"] > 0 else "sous la semelle basse")
    charge = f", charge {niveau}" if (tr.get("reparties") or tr.get("ponctuelles")) and dev["troncon"] is None else ""
    J.hyp.append(f"Déversement : tronçon de {_fr(tr['L'] / 1e3, 2)} m entre sections tenues{charge} ; courbe {courbe}, "
                 f"λ̄_LT,0 = {_fr(lbLT0, 2)}, β = {_fr(beta, 2)}" + (" (annexe belge, à confirmer)." if s["forme"] == "I" else "."))

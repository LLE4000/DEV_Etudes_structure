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
    tube rectangulaire (cas général, § 6.3.2.2) : courbe d, λ̄_LT,0 = 0,2 et β = 1 dans Φ_LT, f = 1 ; le seuil de
        6.3.2.2(4) renvoie au λ̄_LT,0 de 6.3.2.3 — celui de l'ANB, 0,2
    M_b,Rd = χ_LT,mod·W_y·f_y / γ_M1
    négligé (χ_LT = 1) si λ̄_LT ≤ λ̄_LT,0 ou M_Ed/M_cr ≤ λ̄_LT,0²                                        (§ 6.3.2.2(4))

Tube rond ou carré : pas de déversement. Semelle comprimée tenue en continu (« Lc » nul, ou « actif » faux) : pas de
déversement, et c'est dit.

PARAMÈTRES NATIONAUX (acier/barre/parametres.py) : (λ̄_LT,0 ; β) = (0,2 ; 1,0), NBN EN 1993-1-1 ANB:2018, 6.3.2.3(1) —
M_cr est calculé par le solveur du module avec les combinaisons, la distribution réelle des moments et les maintiens,
qu'il y en ait ou non dans le tronçon. La variante (0,4 ; 0,75) suppose les maintiens TOTALEMENT ignorés dans M_cr :
le module n'a pas cette option, il ne l'applique donc jamais (brief § 7.7). Le couple employé est nommé.

Pour l'annexe A (flexion composée), ce fichier rend aussi M_cr,0 — le moment critique du même tronçon sous moment
CONSTANT — λ̄_0 = √(W_y·f_y/M_cr,0) et C₁ = 1/k_c².
"""
import math
from typing import Callable, Dict, Optional, Tuple

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
ELANCE = 2.0                                     # h/b : la limite du tableau 6.5
COURBES_LT = {"lamine": ("b", "c"), "soude": ("c", "d")}        # (h/b ≤ 2, h/b > 2)
COURBE_GENERALE, LAMBDA0_GENERAL, BETA_GENERAL = "d", 0.2, 1.0  # § 6.3.2.2, « autres sections »
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
    """M_cr du tronçon ``tr`` ; avec ``moment``, sous ce diagramme et SANS charge transversale (le moment constant).
    Un PRS monosymétrique : avec le terme de Wagner (z_j de la semelle haute comprimée, acier/barre/prs.zj_haut)."""
    EIz, EIw, GIt = P["E"] * s["Iz"], P["E"] * (s.get("Iw") or 0.0), P["G"] * s["It"]
    zj = PR.zj_haut(s)
    if moment is not None:
        return MC.moment_critique(tr["L"], EIz, EIw, GIt, moment, appuis=tr["appuis"], maintiens=tr["maintiens"], M_max=1.0, zj=zj)
    return MC.moment_critique(tr["L"], EIz, EIw, GIt, tr["moment"], reparties=tr.get("reparties", ()), ponctuelles=tr.get("ponctuelles", ()),
                              appuis=tr["appuis"], maintiens=tr["maintiens"], M_max=tr.get("M_max"), zj=zj)


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
        Mcr0 = MC.moment_constant(tr["L"], P["E"] * s["Iz"], P["E"] * (s.get("Iw") or 0.0), P["G"] * s["It"], signe * PR.zj_haut(s))
    else:
        Mcr0 = _critique(s, P, tr, moment=lambda x: signe)["M_cr"] or Mcr
    lbLT = math.sqrt(Wy * fy / Mcr)
    lb0 = math.sqrt(Wy * fy / Mcr0)
    cles.append(J.f("lambda_LT", "λ̄_LT", [_t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("M_cr", Mcr / 1e6, "kN·m")], lbLT, "",
                    "e_lambda_LT"))
    # ---- la courbe et le couple (λ̄_LT,0 ; β) de l'ANB:2018, 6.3.2.3(1) — un seul : M_cr tient compte des maintiens
    D["a_confirmer"] = [k for k in ("lambda_LT0_avec_maintiens", "beta_avec_maintiens") if PA.PARAMETRES[k]["statut"] == PA.A_CONFIRMER]
    prs = PR.est_prs(s)
    if s["forme"] == "I":
        famille = "soude" if (s["fabrication"] in SOUDES or SE.soude(s)) else "lamine"
        if not s.get("h") or not s.get("b"):
            raise Refus("h et b manquent : la courbe de déversement en dépend")
        courbe = COURBES_LT[famille][s["h"] / s["b"] > ELANCE]
        lbLT0, beta = P["lambda_LT0_avec_maintiens"], P["beta_avec_maintiens"]
        seuil = lbLT0
        methode = "§ 6.3.2.3"
        if prs and not (P["prs_soudee_equivalente"] and not s["monosymetrique"]):
            # PRS (brief § 7.7 et 18.5, DÉCISION BUREAU n° 10) : § 6.3.2.2 par défaut — tableau 6.4, « sections en I
            # soudées », les mêmes courbes c et d ; § 6.3.2.3 seulement pour un PRS bisymétrique déclaré « section soudée
            # équivalente » ; un PRS monosymétrique : § 6.3.2.2 seul
            lbLT0, beta = LAMBDA0_GENERAL, BETA_GENERAL
            methode = METHODE_GENERALE
    else:
        courbe, lbLT0, beta = COURBE_GENERALE, LAMBDA0_GENERAL, BETA_GENERAL
        seuil = P["lambda_LT0_avec_maintiens"]                   # 6.3.2.2(4) renvoie au λ̄_LT,0 de 6.3.2.3 (ANB : 0,2)
        kc, tabule = 1.0, False
        methode = "§ 6.3.2.2"
    alpha = NO.ALPHA[courbe]
    D.update({"cas": CALCULE, "troncon": {"L": tr["L"], "appuis": list(tr["appuis"]), "maintiens": [list(m) for m in tr["maintiens"]]},
              "M_cr": Mcr, "M_cr0": Mcr0, "lambda_LT": lbLT, "lambda_0": lb0, "courbe": courbe, "alpha_LT": alpha, "lambda_LT0": lbLT0,
              "beta": beta, "kc": kc, "kc_tabule": tabule, "C1": 1.0 / kc ** 2, "methode": methode, "W_y": Wy})
    nom_chi = "χ_LT"                             # le nom du facteur que M_b,Rd emploie : celui de la ligne qui le donne
    if lbLT <= seuil or My / Mcr <= seuil ** 2:
        D.update({"cas": NEGLIGE, "Phi_LT": None, "chi_LT_brut": 1.0, "f": 1.0, "chi_LT": 1.0})
        chi = 1.0
        messages = [f"Déversement négligeable (λ̄_LT = {_fr(lbLT, 2)} ≤ {_fr(seuil, 2)})." if lbLT <= seuil else
                    "Déversement négligeable (M_Ed/M_cr ≤ λ̄_LT,0²)."]
    else:
        messages = []
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
        if s["forme"] == "I" and methode != METHODE_GENERALE:
            if fl["diagramme"] == LINEAIRE and dev["troncon"] is None and dev["kc"] is None:
                cles.append(J.f("kc", "k_c", [_t("ψ", fl["psi"])], kc, "", "e_kc_lineaire"))
            f = min(1.0 - 0.5 * (1.0 - kc) * (1.0 - 2.0 * (lbLT - 0.8) ** 2), 1.0)
            if f < 1.0:
                cles.append(J.f("f", "f", [_t("k_c", kc), _t("λ̄_LT", lbLT)], f, "", "e_f"))
        chi = min(chi0 / f, plafond)
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
    cles.append(J.f("M_b_Rd", "M_b,Rd", [_t(nom_chi, chi), _t(nomW, Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("γ_M1", gM1)], MbRd / 1e6,
                    "kN·m", "e_M_b_Rd"))
    D["formules"] = cles
    if prs:
        m_, c_ = _prs(J, d, D, tr, signe, seuil, My, Mcr, kc)
        messages += m_
        cles += c_
    J.verif("E-Mb", "Déversement", f"{REF} ({methode})", "M_y,Ed ≤ M_b,Rd", My / 1e6, MbRd / 1e6, "kN·m", cles, messages=messages)
    niveau = "au centre de cisaillement" if not dev["zg"] else ("sur la semelle haute" if dev["zg"] > 0 else "sous la semelle basse")
    charge = f", charge {niveau}" if (tr.get("reparties") or tr.get("ponctuelles")) and dev["troncon"] is None else ""
    J.hyp.append(f"Déversement : tronçon de {_fr(tr['L'] / 1e3, 2)} m entre sections tenues{charge} ; courbe {courbe}, "
                 f"λ̄_LT,0 = {_fr(lbLT0, 2)}, β = {_fr(beta, 2)}" + (" (ANB:2018, 6.3.2.3(1))." if s["forme"] == "I" and methode != METHODE_GENERALE else "."))


# ------------------------------------------------------------------ les PRS (brief § 18.5)
def chi_LT(lb: float, a_lt: float, lb0: float, b_: float, seuil: float, M: float, Mcr: float, kc: Optional[float] = None) -> Tuple[float, float]:
    """(χ_LT — χ_LT,mod si ``kc`` est donné —, f) : § 6.3.2.3 avec (λ̄_LT,0 ; β) = (``lb0`` ; ``b_``) et f, ou § 6.3.2.2
    (0,2 ; 1, ``kc`` None). L'autre méthode d'un PRS bisymétrique, rendue POUR INFORMATION (DÉCISION BUREAU n° 10 : la
    variante qui ne décide pas est affichée avec sa source) — les expressions de la vérification E-Mb (un test lie les
    deux écritures : la variante d'un calcul est le résultat de l'autre)."""
    if not (lb > seuil and M / Mcr > seuil * seuil):
        return 1.0, 1.0
    phi = 0.5 * (1.0 + a_lt * (lb - lb0) + b_ * lb * lb)
    borne = min(1.0, 1.0 / (lb * lb))
    x0 = min(1.0 / (phi + math.sqrt(phi * phi - b_ * lb * lb)), borne)
    f = 1.0 if kc is None else min(1.0 - 0.5 * (1.0 - kc) * (1.0 - 2.0 * (lb - 0.8) ** 2), 1.0)
    return min(x0 / f, borne), f


def _prs(J: Journal, d: Dict, D: Dict, tr: Dict, signe: float, seuil: float, My: float, Mcr: float, kc: float) -> Tuple[list, list]:
    """Ce que le déversement d'un PRS ajoute (brief § 18.5 et 18.7) : le b retenu pour h/b, l'autre méthode (PRS
    bisymétrique) ou son refus (monosymétrique), le contrôle analytique (E.1) de l'annexe E ANB. Rend (les messages de
    E-Mb, les clés des blocs à lui ajouter)."""
    s, P = d["section"], d["parametres"]
    msgs, cles = [], []
    D["b_h_b"] = s["b"]
    D["soudee_equivalente"] = bool(P["prs_soudee_equivalente"]) and not s["monosymetrique"]
    if s["monosymetrique"]:
        J.hyp.append(f"PRS monosymétrique : h/b = {_fr(s['h'])}/{_fr(s['b'])} = {_fr(s['h'] / s['b'], 2)} avec b = la plus petite largeur de "
                     f"semelle (choix prudent) ; § 6.3.2.2 seul.")
        msgs.append(f"{METHODE_LAMINES} refusé : section monosymétrique.")
    else:
        autre = METHODE_GENERALE if D["methode"] == METHODE_LAMINES else METHODE_LAMINES
        if autre == METHODE_LAMINES:
            chi, f = chi_LT(D["lambda_LT"], D["alpha_LT"], P["lambda_LT0_avec_maintiens"], P["beta_avec_maintiens"], seuil, My, Mcr, kc)
        else:
            chi, f = chi_LT(D["lambda_LT"], D["alpha_LT"], LAMBDA0_GENERAL, BETA_GENERAL, seuil, My, Mcr)
        Mb = chi * D["W_y"] * d["fy"] / P["gamma_M1"]
        D["variante"] = {"methode": autre, "chi_LT": chi, "f": f, "M_b_Rd": Mb}
        msgs.append(f"{autre} pour information : M_b,Rd = {_fr(Mb / 1e6, 1)} kN·m.")
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
        if co["bloque"]:
            msgs.append(f"Contrôle (E.1) bloqué : {co['bloque']}.")
        else:
            zg = (dev["zg"] if signe > 0 else -dev["zg"]) if charge else 0.0
            Mcr_E = MC.annexe_e(tr["L"], EIz, EIw, GIt, co["C1"], co["C2"], co["C3"], zg, zp["z_j"])
            E.update({"z_g": zg, "M_cr": Mcr_E, "ecart": Mcr_E / Mcr - 1.0})
            E["formule"] = J.f("M_cr_E1", "M_cr,E.1", [_t("C₁", co["C1"]), _t("E", P["E"], "N/mm²"), _t("I_z", s["Iz"] / 1e4, "cm⁴"), _t("k_z", 1.0),
                                                       _t("L", tr["L"] / 1e3, "m"), _t("k_ω", 1.0), _t("I_ω", s["Iw"] / 1e6, "cm⁶"),
                                                       _t("G", P["G"], "N/mm²"), _t("I_t", s["It"] / 1e4, "cm⁴"), _t("C₂", co["C2"]), _t("z_g", zg, "mm"),
                                                       _t("C₃", co["C3"]), _t("z_j", zp["z_j"], "mm")], Mcr_E / 1e6, "kN·m", "p_M_cr_E1")
            cles.append(E["formule"])
            msgs.append(f"Contrôle (E.1) : M_cr = {_fr(Mcr_E / 1e6, 1)} kN·m ({'+' if E['ecart'] >= 0 else '−'}{_fr(abs(E['ecart']) * 100, 2)} %).")
        D["annexe_E"] = E
    if s["monosymetrique"]:
        J.hyp.append(f"PRS monosymétrique, semelle {'haute' if signe > 0 else 'basse'} comprimée : ψ_f = "
                     f"{_fr(PR.zj_psi(s, PR.HAUT if signe > 0 else PR.BAS)['psi_f'], 4)}, z_s = {_fr(s['z_s'], 1)} mm, "
                     f"z_j = {_fr(PR.zj_psi(s, PR.HAUT if signe > 0 else PR.BAS)['z_j'], 2)} mm (terme de Wagner dans M_cr).")
    return msgs, cles

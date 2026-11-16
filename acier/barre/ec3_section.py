# -*- coding: utf-8 -*-
"""EUROCODE 3, la résistance de la SECTION (NBN EN 1993-1-1 § 6.2, annexe belge) : efforts de CALCUL (ELU, pondérés).
docs/acier/BARRES_ACIER.md § 5.1 à 5.4. Sections à deux axes de symétrie : I et H, tubes carrés, rectangulaires, ronds.

    f_y de calcul : la table unique acier/materiaux.py (option a de l'ANB:2018, 3.2.1(1) ; tableaux 3.3 ANB à 3.11 ANB,
        produit selon la fabrication), épaisseur max(t_f ; t_w ; t) — la plus petite f_y des parois (brief § 7.1)
    ε = √(235 / f_y) PAROI PAR PAROI, avec la f_y de chaque paroi selon SON épaisseur (classe, section efficace)
                                                                                    classe (acier/barre/ec3_classe.py)
    N_c,Rd = A·f_y/γ_M0 (A_eff en classe 4)             N_t,Rd = A·f_y/γ_M0 (sans trous)                (§ 6.2.3, 6.2.4)
    M_c,Rd = W·f_y/γ_M0 : W_pl (classes 1, 2), W_el (3), W_eff (4)                                      (§ 6.2.5)
    V_pl,Rd = A_v·f_y/(√3·γ_M0)   A_v,z d'un I ≥ η·h_w·t_w (η : ec3_ame.eta_de) ; A_v,y d'un I laminé 2·b·t_f   (§ 6.2.6)
    V_Ed > 0,5·V_pl,Rd : ρ = (2·V_Ed/V_pl,Rd − 1)²                                                      (§ 6.2.8)
        I de classe 1 ou 2, axe fort : M_y,V,Rd = (W_pl,y − ρ·A_w²/(4·t_w))·f_y/γ_M0, A_w = h_w·t_w     (6.30)
        sinon, et dès qu'un effort normal s'ajoute (§ 6.2.10) : (1 − ρ) sur TOUTE la section — du côté sûr
    flexion et effort normal (§ 6.2.9), n = N_Ed/N_pl,Rd :
        classes 1 et 2, I et H : a = (A − 2·b·t_f)/A ≤ 0,5 ; M_N,y = M_pl,y·(1 − n)/(1 − 0,5·a) ≤ M_pl,y            (6.36)
                                 M_N,z = M_pl,z si n ≤ a, sinon M_pl,z·[1 − ((n − a)/(1 − a))²]         (6.37, 6.38)
                        tubes carrés et rectangulaires : a_w = (A − 2·b·t)/A, a_f = (A − 2·h·t)/A ≤ 0,5 (6.39, 6.40)
                        flexion déviée : (M_y/M_N,y)^α + (M_z/M_N,z)^β ≤ 1 — I : α = 2, β = 5n ≥ 1 ;
                                         tubes : α = β = 1,66/(1 − 1,13·n²) ≤ 6                         (6.41)
                        tube rond : n + M_Ed/M_pl,Rd ≤ 1, M_Ed la RÉSULTANTE des deux moments           (6.2, prudent)
        classe 3 : σ_x = N/A + M_y/W_el,y + M_z/W_el,z ≤ f_y/γ_M0                                       (6.42)
        classe 4 : N/(A_eff·f_y/γ_M0) + M_y/(W_eff,y·f_y/γ_M0) + M_z/(W_eff,z·f_y/γ_M0) ≤ 1 (e_N = 0)   (6.44)

UNE classe par combinaison, celle de l'état de contrainte réel (N et M ensemble), employée par toutes les
vérifications de la combinaison. Ce que ce fichier NE vérifie PAS, et qu'il dit : le voilement par cisaillement d'une
âme élancée (h_w/t_w > 72·ε/η — EN 1993-1-5 § 5, vérification « E-Vb » rendue non couverte) ; les trous de boulons.

Le moteur CALCULE et NOMME : blocs de formules en kN, kN·m, mm, mm², cm³, N/mm² (gabarits « e_… » de
acier/barre/formules.py). L'instabilité de la barre (flambement, déversement) : lots suivants.
"""
import math
from typing import Dict, List, Optional

from acier import materiaux as MA
from acier.flambement import normes as NO

from . import ec3_ame as AM
from . import ec3_classe as CL
from . import parametres as PA
from . import section as SE
from .resultat import NOK, NON_COUVERT, Journal, NonCouvert, Refus, fr as _fr, terme as _t

METHODE = "ec3"
NORME = "NBN EN 1993-1-1"
REF = "EN 1993-1-1"
FORMES = ("I",) + SE.TUBES
SEUIL_V = 0.5                    # § 6.2.8(2) : pas de réduction tant que V_Ed ≤ 0,5·V_pl,Rd
A_MAX = 0.5                      # a, a_w, a_f ≤ 0,5 (§ 6.2.9.1(5))
ALPHA_I = 2.0                    # (6.41), I et H
ALPHA_TUBE_MAX = 6.0             # (6.41), tubes carrés et rectangulaires
EFFORTS = ("NEd", "MyEd", "MzEd", "VzEd", "VyEd")
NOMS_W = {1: "W_pl", 2: "W_pl", 3: "W_el", 4: "W_eff"}


# la fabrication du profilé → le produit de la table des aciers (acier/materiaux.py) : les profils creux finis à chaud
# (EN 10210-1) et formés à froid (EN 10219-1) ont leurs tableaux ; tout le reste — laminés, plats et tôles d'un PRS ou
# d'un caisson — est un produit laminé (EN 10025)
PRODUITS = {"creux_chaud": "creux_chaud", "creux_froid": "creux_froid"}


def epaisseur(s: Dict) -> float:
    """L'épaisseur nominale qui fixe la f_y de CALCUL de la section : la plus forte paroi, mm (la plus petite f_y)."""
    return max(float(s.get(k) or 0.0) for k in ("tf", "tw", "t"))


def produit(s: Dict) -> str:
    """Le produit de la table des aciers selon la fabrication : « lamine », « creux_chaud » ou « creux_froid »."""
    return PRODUITS.get(s.get("fabrication"), "lamine")


def limite_elastique(nuance: str, s: Dict, t: Optional[float] = None) -> float:
    """f_y de la nuance pour l'épaisseur ``t`` (par défaut la f_y de CALCUL : la plus forte paroi), le produit de la
    section — la seule lecture de la table des barres (Eurocode 3, méthode historique, âme, raidisseurs) ; Refus hors
    du domaine (nuance inconnue, épaisseur de plus de 80 mm, au-delà du tableau)."""
    try:
        return NO.limite_elastique(nuance, epaisseur(s) if t is None else t, produit=produit(s))
    except NO.HorsDomaine as e:
        raise Refus(str(e)) from None


def source_fy(nuance: str, s: Dict, t: Optional[float] = None) -> str:
    """D'où vient la f_y de calcul : « tableau 3.3 ANB (EN 10025-2:2004, S235JR), 16 < t ≤ 40 mm »."""
    return MA.source(nuance, epaisseur(s) if t is None else t, produit(s), "fy")


def tableau_fy(nuance: str, s: Dict) -> str:
    """Le tableau de l'ANB que la f_y lit, en court (« tableau 3.3 ANB ») : pour une ligne où la source entière ne tient pas."""
    return f"tableau {MA.ligne(nuance, produit(s))['tableau']}"


def limites_parois(nuance: str, s: Dict) -> Dict[str, float]:
    """La f_y de CHAQUE paroi selon SON épaisseur (brief § 7.1 ; EN 1993-1-1 tableau 5.2 : ε de la paroi) : I et H —
    {"semelle": f_y(t_f), "ame": f_y(t_w)} ; tube — une seule épaisseur, {"b", "h", "tube": f_y(t)}."""
    if s.get("forme") == "I" and s.get("tf") and s.get("tw"):
        return {"semelle": limite_elastique(nuance, s, s["tf"]), "ame": limite_elastique(nuance, s, s["tw"])}
    fy = limite_elastique(nuance, s)
    return {"b": fy, "h": fy, "tube": fy, "semelle": fy, "ame": fy}


def lire(barre: Dict) -> Dict:
    """Les données, complétées et contrôlées (une COPIE) : {"section", "nuance", "fy", "fy_parois", "fy_source", "t",
    "fy_tableau", "parametres", "NEd", "MyEd", "MzEd", "VzEd", "VyEd", "voilement"}. ``fy`` donné passe devant la table
    (toutes les parois le prennent) ; sinon f_y de calcul et f_y de chaque paroi selon l'épaisseur. ``voilement`` faux :
    sans le voilement par cisaillement d'une âme élancée — le moteur des barres le range dans SON bloc de l'âme
    (acier/barre/ame.py), avec les autres vérifications de l'EN 1993-1-5."""
    if not isinstance(barre, dict) or not isinstance(barre.get("section"), dict):
        raise Refus("la barre doit porter une section")
    s = barre["section"]
    try:
        P = PA.jeu(barre.get("parametres"))
    except PA.CleRetiree as e:
        raise Refus(f"paramètre retiré : {e.args[0]!r} ({e.motif})") from None
    except KeyError as e:
        raise Refus(f"paramètre inconnu : {e.args[0]!r}") from None
    d = {"section": s, "parametres": P, "nuance": barre.get("nuance"), "t": epaisseur(s), "voilement": bool(barre.get("voilement", True))}
    for k in EFFORTS:
        v = barre.get(k, 0.0)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise Refus(f"« {k} » doit être un nombre : {v!r}")
        d[k] = float(v)
    fy = barre.get("fy")
    if fy is None:
        if d["nuance"] not in NO.NUANCES:
            raise Refus(f"nuance inconnue : {d['nuance']!r} (S235, S275, S355 ou S460, ou bien f_y)")
        fy = limite_elastique(d["nuance"], s)
        d["fy_parois"] = limites_parois(d["nuance"], s)
        d["fy_source"] = source_fy(d["nuance"], s)
        d["fy_ref"] = tableau_fy(d["nuance"], s)
        d["fy_tableau"] = True
    else:
        if isinstance(fy, bool) or not isinstance(fy, (int, float)) or not math.isfinite(float(fy)) or fy <= 0:
            raise Refus("la limite élastique f_y doit être un nombre positif")
        d["fy_tableau"] = False
        d["fy_parois"] = {k: float(fy) for k in ("semelle", "ame", "b", "h", "tube")}
        d["fy_source"] = d["fy_ref"] = None
    d["fy"] = float(fy)
    for k in ("gamma_M0", "gamma_M1", "eta", "eta_soude"):
        if not isinstance(P[k], (int, float)) or isinstance(P[k], bool) or not P[k] > 0:
            raise Refus(f"paramètre « {k} » : un nombre positif est attendu")
    return d


def aires_cisaillement(s: Dict, eta: float) -> Dict:
    """A_v (mm²) pour V_z (parallèle à l'âme) et V_y (parallèle aux semelles), § 6.2.6(3) : {"z", "y", "plancher"
    (η·h_w·t_w d'un I, mm²), "plancher_gouverne"}. I et H : A − 2·b·t_f + (t_w + 2r)·t_f ≥ η·h_w·t_w (le catalogue porte
    la première expression) ; parallèle aux semelles, I ou H LAMINÉ : 2·b·t_f (DÉCISION BUREAU n° 3, brief § 7.5 ;
    EN 1993-1-1:2022 § 8.2.6(3)b)), I SOUDÉ : A − h_w·t_w (6.2.6(3) e)) ; U : parallèle à l'âme A − 2·b·t_f + (t_w + r)·t_f
    (6.2.6(3) b), le catalogue la porte), parallèle aux semelles les SEMELLES — U LAMINÉ : la plus petite de 2·b·t_f (la
    règle de la DÉCISION BUREAU n° 3 des I et H laminés) et de A − h_w·t_w (6.2.6(3) e), qui nomme les U soudés ; aucun
    cas de 6.2.6(3) ne vise un U laminé chargé parallèlement aux semelles), U SOUDÉ : A − h_w·t_w ; tubes carrés et
    rectangulaires : A·h/(b + h) et A·b/(b + h) ; tube rond : 2A/π. Une autre forme (cornière, « autre ») : Refus — le
    module ne la vérifie pas (``FORMES``)."""
    A, forme = s["A"], s["forme"]
    if forme == "I":
        hw = SE.hauteur_ame(s)
        if hw is None or not s.get("tw") or not s.get("b"):
            raise Refus("h, b, t_w, t_f manquent : l'aire de cisaillement d'un I en dépend")
        base = s.get("Avz")
        if base is None:
            base = A - 2.0 * s["b"] * s["tf"] + (s["tw"] + 2.0 * (s.get("r") or 0.0)) * s["tf"]
        plancher = eta * hw * s["tw"]
        avy = A - hw * s["tw"] if s.get("fabrication") in AM.SOUDES else 2.0 * s["b"] * s["tf"]
        return {"z": max(base, plancher), "y": avy, "plancher": plancher, "plancher_gouverne": plancher > base, "hw": hw}
    if forme in ("tube_carre", "tube_rect"):
        if not s.get("h") or not s.get("b"):
            raise Refus("h, b manquent : l'aire de cisaillement d'un tube en dépend")
        avz, avy = s.get("Avz"), s.get("Avy")
        return {"z": avz if avz is not None else A * s["h"] / (s["b"] + s["h"]),
                "y": avy if avy is not None else A * s["b"] / (s["b"] + s["h"]), "plancher": None, "plancher_gouverne": False}
    if forme == "U":
        if not all(s.get(k) for k in ("h", "b", "tw", "tf")):
            raise Refus("h, b, t_w, t_f manquent : l'aire de cisaillement d'un U en dépend")
        avz = s.get("Avz")
        if avz is None:                                      # 6.2.6(3) b) ; sans congé saisi, r = 0 (du côté sûr)
            avz = A - 2.0 * s["b"] * s["tf"] + (s["tw"] + (s.get("r") or 0.0)) * s["tf"]
        geo = A - (s["h"] - 2.0 * s["tf"]) * s["tw"]         # 6.2.6(3) e) : A − Σ(h_w·t_w)
        avy = geo if s.get("fabrication") in AM.SOUDES else min(2.0 * s["b"] * s["tf"], geo)
        return {"z": avz, "y": avy, "plancher": None, "plancher_gouverne": False}
    if forme != "tube_rond":
        raise Refus(f"forme « {forme} » : l'aire de cisaillement n'est pas définie hors des I, H, U et tubes")
    av = s.get("Avz")
    av = av if av is not None else 2.0 * A / math.pi
    return {"z": av, "y": av, "plancher": None, "plancher_gouverne": False}


# l'indice de l'ε d'une paroi quand il diffère de celui de la f_y de calcul (I et H : semelles, âme)
INDICE_EPS = {"semelle": "f", "ame": "w"}


def _eps_paroi(J: Journal, p: Dict, eps_calcul: float) -> str:
    """Le symbole de l'ε d'une paroi : « ε » quand il est celui de la f_y de calcul ; sinon « ε_f » (semelles) ou « ε_w »
    (âme), écrit une fois dans le journal avec la f_y de la paroi (brief § 7.1)."""
    if p["epsilon"] == eps_calcul or p["role"] not in INDICE_EPS:
        return "ε"
    i = INDICE_EPS[p["role"]]
    if not any(f["cle"] == f"eps_{i}" for f in J.calcul):
        J.f(f"eps_{i}", f"ε_{i}", [_t(f"f_y,{i}", p["fy"], "N/mm²")], p["epsilon"], "", "e_eps")
    return f"ε_{i}"


def _classification(J: Journal, d: Dict, N: float, My: float, Mz: float) -> Dict:
    """La classe, nommée : les blocs α, ψ et la limite qui décide, paroi par paroi — chaque paroi avec SON ε."""
    s, fy = d["section"], d["fy"]
    cl = CL.classer(s, fy, N, My, Mz, d.get("fy_parois"))
    cles = []
    for p in cl["parois"]:
        r = p["role"]
        if p["etat"] == CL.TENDUE:
            continue
        eps = p["epsilon"]
        se = _eps_paroi(J, p, cl["epsilon"])
        if se != "ε":
            cles.append(f"eps_{INDICE_EPS[r]}")
        if p["etat"] == CL.FLEXION_COMPOSEE:
            if 0.0 < p["alpha"] < 1.0:
                cles.append(J.f(f"alpha_{r}", "α", [_t("N_Ed", N / 1e3, "kN"), _t("c", p["c"], "mm"),
                                                    _t("t_w" if p["n"] == 1 else "2·t", p["n"] * p["t"], "mm"), _t("f_y", fy, "N/mm²")],
                                p["alpha"], "", "e_alpha"))
            if p["psi"] < 1.0:
                cles.append(J.f(f"psi_{r}", "ψ", [_t("N_Ed", N / 1e3, "kN"), _t("A", s["A"], "mm²"), _t("f_y", fy, "N/mm²")], p["psi"],
                                "", "e_psi"))
        i = min(p["classe"], 3) - 1                          # la limite de SA classe ; en classe 4, celle de la classe 3 (dépassée)
        gabarit, k = p["regles"][i]
        if not gabarit:                                      # âme tendue sur toute sa hauteur (α = 0) : aucune limite
            continue
        termes = {"e_lim": [_t("k", k), _t(se, eps)], "e_lim_tube": [_t("k", k), _t(se, eps)],
                  "e_lim12_haut": [_t("k", k), _t(se, eps), _t("α", p["alpha"])], "e_lim12_bas": [_t("k", k), _t(se, eps), _t("α", p["alpha"])],
                  "e_lim3_haut": [_t(se, eps), _t("ψ", p["psi"])], "e_lim3_bas": [_t(se, eps), _t("ψ", p["psi"])]}[gabarit]
        p["limite"] = {"classe": i + 1, "valeur": p["limites"][i], "formule": f"lim_{r}"}
        cles.append(J.f(f"lim_{r}", f"(c/t){'₁₂₃'[i]}", termes, p["limites"][i], "", gabarit))      # la limite de la classe i + 1
    cl["formules"] = cles
    return cl


def _blocs_paroi(J: Journal, cle: str, p: Dict, eps: float) -> List[str]:
    """λ̄_p et ρ d'une paroi réduite — avec l'ε de LA paroi (``p["epsilon"]``, à défaut ``eps``)."""
    e = p.get("epsilon", eps)
    se = "ε" if e == eps or p.get("role") not in INDICE_EPS else f"ε_{INDICE_EPS[p['role']]}"
    out = [J.f(f"lambda_p_{cle}", "λ̄_p", [_t("c/t", p["c_t"]), _t(se, e), _t("k_σ", p["k_sigma"])], p["lambda_p"], "", "e_lambda_p")]
    if p["type"] == "interne":
        out.append(J.f(f"rho_{cle}", "ρ", [_t("λ̄_p", p["lambda_p"]), _t("ψ", p["psi"])], p["rho"], "", "e_rho_interne"))
    else:
        out.append(J.f(f"rho_{cle}", "ρ", [_t("λ̄_p", p["lambda_p"])], p["rho"], "", "e_rho_console"))
    return out


INDICE_PAROI = {"semelle": "f", "ame": "w", "b": "b", "h": "h", "tube": "t"}      # l'indice des parois d'un rôle, dans A_eff


def _efficace(J: Journal, s: Dict, eps: float, comprime: bool, My: float, Mz: float, eps_parois: Optional[Dict] = None) -> Dict:
    """La section efficace d'une classe 4 : A_eff si la barre est comprimée, W_eff autour de chaque axe fléchi — chaque
    paroi avec son ε (``eps_parois`` {rôle: ε} ; à défaut ``eps``, celui de la f_y de calcul)."""
    ef: Dict = {"A_eff": s["A"], "Weff_y": s.get("Wel_y"), "Weff_z": s.get("Wel_z"), "formules": {"A": [], "y": [], "z": []}}
    ep = eps_parois or eps
    if comprime:
        a = CL.aire_efficace(s, ep)
        ef["A_eff"], ef["aire"] = a["A_eff"], a
        if a["reduite"]:
            cles = []
            for p in a["parois"]:
                if p["rho"] < 1.0:
                    cles += _blocs_paroi(J, f"N_{p['role']}", p, eps)
            termes = [_t("A", s["A"], "mm²")]
            for p in a["parois"]:                                    # deux familles de parois, chacune à son indice
                i = INDICE_PAROI[p["role"]]
                termes += [_t(f"n_{i}", p["n"]), _t(f"ρ_{i}", p["rho"]), _t(f"c_{i}", p["c"], "mm"), _t(f"t_{i}", p["t"], "mm")]
            cles.append(J.f("A_eff", "A_eff", termes, a["A_eff"], "mm²", "e_A_eff"))
            ef["formules"]["A"] = cles
    for axe, M in (("y", My), ("z", Mz)):
        if M <= 0:
            continue
        w = CL.module_efficace(s, ep, axe)
        ef[f"Weff_{axe}"], ef[f"module_{axe}"] = w["W_eff"], w
        if w["reduit"]:
            cles = []
            for nom, p in (("m", w["membrure"]), ("a", w["ame"])):
                if p is not None and p["rho"] < 1.0:
                    cles += _blocs_paroi(J, f"M{axe}_{nom}", p, eps)
            cles.append(J.f(f"W_eff_{axe}", f"W_eff,{axe}", [_t("I_eff", w["I_eff"] / 1e4, "cm⁴"), _t("v", w["v"], "mm"), _t("e", w["e"], "mm")],
                            w["W_eff"] / 1e3, "cm³", "e_W_eff"))
            ef["formules"][axe] = cles
    return ef


def ajouter(J: Journal, d: Dict) -> None:
    """Les vérifications de la SECTION, versées dans le journal : E-N, E-My, E-Mz, E-Vz, E-Vy, le voilement par
    cisaillement d'une âme élancée (A-Vb, A-VbM : ec3_ame.py), puis E-MV (flexion et effort tranchant) ou E-NM (flexion
    composée, déviée)."""
    s, fy, P = d["section"], d["fy"], d["parametres"]
    gM0, eta = P["gamma_M0"], AM.eta_de(s, P)
    N = d["NEd"]
    My, Mz, Vz, Vy = abs(d["MyEd"]), abs(d["MzEd"]), abs(d["VzEd"]), abs(d["VyEd"])
    forme, A = s["forme"], s["A"]
    G = J.G
    eps = NO.epsilon(fy)
    fyp = d.get("fy_parois") or {}
    G.update({"fy": fy, "epsilon": eps, "gamma_M0": gM0, "t": d["t"], "A": A, "NEd": N, "MyEd": My, "MzEd": Mz, "VzEd": Vz, "VyEd": Vy,
              "fy_parois": fyp, "fy_source": d.get("fy_source"), "fy_ref": d.get("fy_ref")})
    J.f("eps", "ε", [_t("f_y", fy, "N/mm²")], eps, "", "e_eps")
    J.hyp.append(f"Efforts de calcul (ELU) ; γ_M0 = {_fr(gM0, 2)}.")
    nuance = f"Acier {d['nuance']}" if d["nuance"] else "Acier"
    if d["fy_tableau"]:
        detail = f" (épaisseur {_fr(d['t'], 1)} mm ; {d['fy_source']})"
        if forme == "I" and len({fyp.get("semelle"), fyp.get("ame")}) == 2:
            detail += (f" ; ε paroi par paroi : semelles f_y = {_fr(fyp['semelle'])}, âme f_y = {_fr(fyp['ame'])} N/mm²")
        J.hyp.append(f"{nuance} : f_y = {_fr(fy)} N/mm²{detail}.")
    else:
        J.hyp.append(f"{nuance} : f_y = {_fr(fy)} N/mm² (donnée).")
    G["couvert"] = False                                             # (vrai plus bas : les vérifications de la barre en dépendent)
    if forme not in FORMES:
        J.verif("E-SECTION", "Résistance de la section", NORME, "", None, None, "", [], statut=NON_COUVERT,
                messages=[J.alerte("forme", "attention", "Cornière, U ou té : hors du module (compression simple : module Flambement).")])
        return

    # ---- un tube rond n'a pas d'axe : la résultante des deux moments, celle des deux efforts tranchants
    cles_M: List[str] = []
    cles_V: List[str] = []
    rond = forme == "tube_rond"
    if rond:
        if My > 0 and Mz > 0:
            M = math.hypot(My, Mz)
            cles_M.append(J.f("M_res", "M_Ed", [_t("M_y,Ed", My / 1e6, "kN·m"), _t("M_z,Ed", Mz / 1e6, "kN·m")], M / 1e6, "kN·m", "e_resultante"))
        My, Mz = math.hypot(My, Mz), 0.0
        if Vz > 0 and Vy > 0:
            V = math.hypot(Vz, Vy)
            cles_V.append(J.f("V_res", "V_Ed", [_t("V_z,Ed", Vz / 1e3, "kN"), _t("V_y,Ed", Vy / 1e3, "kN")], V / 1e3, "kN", "e_resultante"))
        Vz, Vy = math.hypot(Vz, Vy), 0.0
    comprime = N > 0
    flechi = My > 0 or Mz > 0
    G["My_calcul"], G["Mz_calcul"] = My, Mz

    # ---- la classe (une par combinaison) et, en classe 4, la section efficace
    cl = _classification(J, d, N, My, Mz)
    classe = cl["classe"]
    G["classe"], G["classification"] = classe, cl
    detail = ", ".join(f"{p['nom']} {p['classe']}" for p in cl["parois"])
    J.hyp.append(f"Section de classe {classe}" + (f" ({detail})." if len(cl["parois"]) > 1 else "."))
    if forme in ("tube_carre", "tube_rect") and My > 0 and Mz > 0:
        J.hyp.append("Flexion déviée d'un tube : parois classées en compression uniforme (du côté sûr).")
    ef = None
    if classe == 4:
        try:
            ef = _efficace(J, s, eps, comprime, My, Mz, {p["role"]: p["epsilon"] for p in cl["parois"]})
        except NonCouvert as e:
            J.verif("E-SECTION", "Résistance de la section", "EN 1993-1-6", "", None, None, "", [], statut=NON_COUVERT,
                    messages=[J.alerte("classe4", "attention", str(e))])
            return
        G["efficace"] = ef
        J.alerte("classe4", "info", "Classe 4 : aire et modules efficaces (EN 1993-1-5 § 4.4).")

    # ---- les résistances
    A_N = ef["A_eff"] if (ef is not None and comprime) else A
    nomW = NOMS_W[classe]
    if classe <= 2:
        Wy, Wz = s.get("Wpl_y"), s.get("Wpl_z")
    elif classe == 3:
        Wy, Wz = s.get("Wel_y"), s.get("Wel_z")
    else:
        Wy, Wz = ef["Weff_y"], ef["Weff_z"]
    manque = [n_ for n_, v, m in ((f"{nomW},y", Wy, My), (f"{nomW},z", Wz, Mz)) if m > 0 and not v]
    if manque:
        raise Refus(f"propriété(s) manquante(s) pour ces efforts : {', '.join(manque)}")
    NRd = A_N * fy / gM0
    MyRd = Wy * fy / gM0 if My > 0 else None
    MzRd = Wz * fy / gM0 if Mz > 0 else None
    nomN = "N_c,Rd" if comprime else "N_t,Rd"
    cle_N = cle_My = cle_Mz = None
    if N != 0:
        cle_N = J.f("N_Rd", nomN, [_t("A_eff" if A_N != A else "A", A_N, "mm²"), _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)], NRd / 1e3, "kN", "e_N_Rd")
    if My > 0:
        cle_My = J.f("M_y_Rd", "M_c,Rd" if rond else "M_c,y,Rd", [_t(nomW if rond else f"{nomW},y", Wy / 1e3, "cm³"), _t("f_y", fy, "N/mm²"),
                                                                 _t("γ_M0", gM0)], MyRd / 1e6, "kN·m", "e_M_Rd")
    if Mz > 0:
        cle_Mz = J.f("M_z_Rd", "M_c,z,Rd", [_t(f"{nomW},z", Wz / 1e3, "cm³"), _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)], MzRd / 1e6, "kN·m", "e_M_Rd")
    G.update({"A_N": A_N, "W_y": Wy, "W_z": Wz, "N_Rd": NRd, "M_y_Rd": MyRd, "M_z_Rd": MzRd, "couvert": True})
    ef_cles = ef["formules"] if ef is not None else {"A": [], "y": [], "z": []}

    if N != 0:
        J.verif("E-N", "Compression" if comprime else "Traction", f"{REF} § 6.2.4" if comprime else f"{REF} § 6.2.3",
                f"N_Ed ≤ {nomN}", abs(N) / 1e3, NRd / 1e3, "kN", ef_cles["A"] + [cle_N])
    if My > 0:
        J.verif("E-My", "Flexion" if rond else "Flexion, axe fort", f"{REF} § 6.2.5", "M_Ed ≤ M_c,Rd" if rond else "M_y,Ed ≤ M_c,y,Rd",
                My / 1e6, MyRd / 1e6, "kN·m", cles_M + ef_cles["y"] + [cle_My])
    if Mz > 0:
        J.verif("E-Mz", "Flexion, axe faible", f"{REF} § 6.2.5", "M_z,Ed ≤ M_c,z,Rd", Mz / 1e6, MzRd / 1e6, "kN·m", ef_cles["z"] + [cle_Mz])

    # ---- l'effort tranchant
    rho = {"z": 0.0, "y": 0.0}
    cles_rho: List[str] = []
    if Vz > 0 or Vy > 0:
        av = aires_cisaillement(s, eta)
        G["A_v"] = av
        for a, V in (("z", Vz), ("y", Vy)):
            if V <= 0:
                continue
            cles = list(cles_V)
            if a == "z" and av["plancher_gouverne"]:
                cles.append(J.f("Av_z", "A_v,z", [_t("η", eta), _t("h_w", av["hw"], "mm"), _t("t_w", s["tw"], "mm")], av["z"], "mm²", "e_Av_plancher"))
            Vpl = av[a] * fy / (math.sqrt(3.0) * gM0)
            nomV = "V_pl,Rd" if rond else f"V_pl,{a},Rd"
            cles.append(J.f(f"V_pl_{a}", nomV, [_t("A_v" if rond else f"A_v,{a}", av[a], "mm²"), _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)],
                            Vpl / 1e3, "kN", "e_V_pl"))
            G[f"V_pl_{a}"] = Vpl
            J.verif(f"E-V{a}", "Effort tranchant" + ("" if rond else (", parallèle à l'âme" if a == "z" else ", parallèle aux semelles")),
                    f"{REF} § 6.2.6", "V_Ed ≤ V_pl,Rd" if rond else f"V_{a},Ed ≤ V_pl,{a},Rd", V / 1e3, Vpl / 1e3, "kN", cles)
            if V > SEUIL_V * Vpl:
                rho[a] = min((2.0 * V / Vpl - 1.0) ** 2, 1.0)
                if V < Vpl:
                    cles_rho.append(J.f(f"rho_{a}", "ρ" if rond else f"ρ_{a}", [_t("V_Ed" if rond else f"V_{a},Ed", V / 1e3, "kN"),
                                                                                 _t(nomV, Vpl / 1e3, "kN")], rho[a], "", "e_rho_V"))
            if d["voilement"]:                               # une âme élancée : EN 1993-1-5 § 5 et § 7.1 (« A-Vb », « A-VbM »)
                AM.cisaillement(J, s, fy, P, V, My if a == "z" else Mz, N, direction=a, fy_ame=fyp.get("ame"))
        if forme == "I" and Vy > 0:
            if s.get("fabrication") in AM.SOUDES:
                J.alerte("Avy", "info", "A_v,y = A − h_w·t_w : section soudée (EN 1993-1-1 § 6.2.6(3) e)).")
            else:
                J.alerte("Avy", "info", "A_v,y = 2·b·t_f : I ou H laminé, décision du bureau (défaut ; EN 1993-1-1:2022 § 8.2.6(3)b)).")
    G["rho_z"], G["rho_y"] = rho["z"], rho["y"]
    r = max(rho["z"], rho["y"])
    G["rho"] = r

    # ---- les interactions
    if flechi and (N != 0 or (My > 0 and Mz > 0)):
        _flexion_composee(J, d, classe, ef, N, My, Mz, NRd, MyRd, MzRd, r, cles_rho, cles_M, [cle_N, cle_My, cle_Mz], rond)
    elif flechi and r > 0:
        _flexion_tranchant(J, d, classe, My, Mz, MyRd, MzRd, rho, cles_rho, cles_M, [cle_My, cle_Mz], rond)


def _flexion_tranchant(J: Journal, d: Dict, classe: int, My: float, Mz: float, MyRd, MzRd, rho: Dict, cles_rho: List[str],
                       cles_M: List[str], cles_R: List[Optional[str]], rond: bool) -> None:
    """§ 6.2.8 : un seul moment, sans effort normal, sous un effort tranchant de plus de 0,5·V_pl,Rd."""
    s, fy, gM0 = d["section"], d["fy"], d["parametres"]["gamma_M0"]
    a = "y" if My > 0 else "z"
    M, MRd = (My, MyRd) if a == "y" else (Mz, MzRd)
    r = max(rho["z"], rho["y"])
    nom = "M_V,Rd" if rond else f"M_{a},V,Rd"
    if a == "y" and s["forme"] == "I" and classe <= 2 and rho["y"] == 0.0:
        Aw = SE.hauteur_ame(s) * s["tw"]
        MV = min((s["Wpl_y"] - r * Aw ** 2 / (4.0 * s["tw"])) * fy / gM0, MRd)
        cle = J.f("M_V", nom, [_t("W_pl,y", s["Wpl_y"] / 1e3, "cm³"), _t("ρ_z", r), _t("A_w", Aw, "mm²"), _t("t_w", s["tw"], "mm"),
                               _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)], MV / 1e6, "kN·m", "e_M_yV")
        ref = f"{REF} § 6.2.8 (6.30)"
    else:
        MV = (1.0 - r) * MRd
        cle = J.f("M_V", nom, [_t("ρ", r), _t("M_c,Rd" if rond else f"M_c,{a},Rd", MRd / 1e6, "kN·m")], MV / 1e6, "kN·m", "e_reduit")
        ref = f"{REF} § 6.2.8"
        J.hyp.append("Effort tranchant élevé : limite d'élasticité réduite de (1 − ρ) sur toute la section (du côté sûr).")
    J.G["M_V_Rd"] = MV
    critere = "M_Ed ≤ M_V,Rd" if rond else f"M_{a},Ed ≤ M_{a},V,Rd"
    if MV <= 0:
        J.verif("E-MV", "Flexion et effort tranchant", ref, critere, M / 1e6, 0.0, "kN·m", cles_M + cles_rho, statut=NOK,
                messages=["V_Ed ≥ V_pl,Rd : aucune réserve pour la flexion."])
    else:
        J.verif("E-MV", "Flexion et effort tranchant", ref, critere, M / 1e6, MV / 1e6, "kN·m", cles_M + cles_R + cles_rho + [cle])


def _flexion_composee(J: Journal, d: Dict, classe: int, ef, N: float, My: float, Mz: float, NRd: float, MyRd, MzRd, r: float,
                      cles_rho: List[str], cles_M: List[str], cles_R: List[Optional[str]], rond: bool) -> None:
    """§ 6.2.9 (et 6.2.10 sous un effort tranchant élevé) : effort normal et flexion, flexion déviée."""
    s, fy, gM0 = d["section"], d["fy"], d["parametres"]["gamma_M0"]
    A, forme, G = s["A"], s["forme"], J.G
    devie = My > 0 and Mz > 0
    if N > 0:
        titre = "Flexion composée déviée" if devie else "Flexion composée"
    elif N < 0:
        titre = "Traction et flexion déviée" if devie else "Traction et flexion"
    else:
        titre = "Flexion déviée"
    cles = cles_M + [k for k in cles_R if k] + cles_rho
    ref = f"{REF} § 6.2.9.{min(classe, 4) - 1 if classe >= 3 else 1}"
    k = 1.0 - r
    if r > 0:
        titre += ", avec effort tranchant"
        ref += " et § 6.2.10"
        J.hyp.append("Effort tranchant élevé : limite d'élasticité réduite de (1 − ρ) sur toute la section (du côté sûr).")
        if k <= 0:
            J.verif("E-NM", titre, ref, "", None, None, "", cles, statut=NOK, messages=["V_Ed ≥ V_pl,Rd : aucune réserve pour la flexion."])
            return

    def reduit(cle: str, nom: str, R: Optional[float], unite: str, echelle: float) -> Optional[float]:
        """(1 − ρ)·R, nommé — R tel quel quand l'effort tranchant est faible."""
        if R is None or r == 0:
            return R
        cles.append(J.f(cle, nom, [_t("ρ", r), _t(nom.replace(",V", ""), R / echelle, unite)], k * R / echelle, unite, "e_reduit"))
        return k * R

    if classe == 3:
        # (6.42) : la contrainte longitudinale, au point le plus sollicité
        couples = [(n_, e_, p_) for n_, e_, p_ in (("N_Ed", abs(N) / 1e3, A), ("M_y,Ed", My / 1e6, (s.get("Wel_y") or 0.0) / 1e3),
                                                    ("M_z,Ed", Mz / 1e6, (s.get("Wel_z") or 0.0) / 1e3)) if e_ > 0]
        sigma = sum(e_ * 1e3 / p_ for _, e_, p_ in couples)
        lim = k * fy / gM0
        termes = []
        for n_, e_, p_ in couples:
            if n_ == "N_Ed":
                termes += [_t(n_, e_, "kN"), _t("A", p_, "mm²")]
            else:
                termes += [_t("M_Ed" if rond else n_, e_, "kN·m"), _t("W_el" if rond else f"W_el,{n_[2]}", p_, "cm³")]
        cles.append(J.f("sigma_x", "σ_x,Ed", termes, sigma, "N/mm²", f"e_sigma{len(couples)}"))
        cles.append(J.f("sigma_Rd", "σ_Rd", [_t("ρ", r), _t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)] if r > 0 else
                        [_t("f_y", fy, "N/mm²"), _t("γ_M0", gM0)], lim, "N/mm²", "e_sigma_Rd_V" if r > 0 else "e_sigma_Rd"))
        G["sigma_x"] = sigma
        J.verif("E-NM", titre, ref + " (6.42)", "σ_x,Ed ≤ f_y/γ_M0", sigma, lim, "N/mm²", cles)
        return

    NRd = reduit("N_V_Rd", "N_V,Rd", NRd, "kN", 1e3) if N != 0 else NRd
    MyRd = reduit("M_y_V_Rd", "M_V,Rd" if rond else "M_y,V,Rd", MyRd, "kN·m", 1e6)
    MzRd = reduit("M_z_V_Rd", "M_z,V,Rd", MzRd, "kN·m", 1e6)

    if classe == 4:
        # (6.44) : A_eff sous compression seule, W_eff sous flexion seule ; e_N = 0 (deux axes de symétrie)
        couples = [(n_, e_, R_, u_, ech) for n_, e_, R_, u_, ech in (("N_", abs(N), NRd, "kN", 1e3), ("M_y,", My, MyRd, "kN·m", 1e6),
                                                                      ("M_z,", Mz, MzRd, "kN·m", 1e6)) if e_ > 0]
        taux = sum(e_ / R_ for _, e_, R_, _, _ in couples)
        termes = []
        for n_, e_, R_, u_, ech in couples:                          # « N_Ed », « N_Rd » ; « M_y,Ed », « M_y,Rd »
            termes += [_t(f"{n_}Ed", e_ / ech, u_), _t(f"{n_}Rd", R_ / ech, u_)]
        cles.append(J.f("interaction", "η", termes, taux, "", f"e_lineaire{len(couples)}"))
        J.verif("E-NM", titre, ref + " (6.44)", "N_Ed/N_Rd + M_y,Ed/M_y,Rd + M_z,Ed/M_z,Rd ≤ 1", taux, 1.0, "", cles, taux=taux)
        return

    # ---- classes 1 et 2 : interaction plastique
    n = abs(N) / NRd if N != 0 else 0.0
    G["n"] = n
    if N != 0:
        cles.append(J.f("n", "n", [_t("N_Ed", abs(N) / 1e3, "kN"), _t("N_V,Rd" if r > 0 else "N_pl,Rd", NRd / 1e3, "kN")], n, "", "e_n"))
    if n >= 1.0:
        J.verif("E-NM", titre, ref, "n < 1", None, None, "", cles, statut=NOK, messages=["N_Ed ≥ N_pl,Rd : aucune réserve pour la flexion."])
        return
    if rond:
        taux = n + My / MyRd
        cles.append(J.f("interaction", "η", [_t("N_Ed", abs(N) / 1e3, "kN"), _t("N_Rd", NRd / 1e3, "kN"), _t("M_Ed", My / 1e6, "kN·m"),
                                             _t("M_Rd", MyRd / 1e6, "kN·m")], taux, "", "e_lineaire2"))
        J.verif("E-NM", titre, f"{REF} § 6.2.1(7)" + (" et § 6.2.10" if r > 0 else ""), "N_Ed/N_Rd + M_Ed/M_Rd ≤ 1", taux, 1.0, "", cles, taux=taux)
        return

    def coefficient(cle: str, nom: str, largeur: float, t: float, sym_l: str, sym_t: str) -> float:
        """a = (A − 2·largeur·t)/A, plafonné à 0,5 — le bloc n'est écrit que sous le plafond."""
        brut = (A - 2.0 * largeur * t) / A
        if brut <= A_MAX:
            cles.append(J.f(cle, nom, [_t("A", A, "mm²"), _t(sym_l, largeur, "mm"), _t(sym_t, t, "mm")], brut, "", "e_a"))
        return min(brut, A_MAX)

    MNy = MNz = None
    if forme == "I":
        a = coefficient("a", "a", s["b"], s["tf"], "b", "t_f") if N != 0 else min((A - 2.0 * s["b"] * s["tf"]) / A, A_MAX)
        G["a"] = a
        if My > 0:
            f_ = (1.0 - n) / (1.0 - 0.5 * a)
            MNy = MyRd * f_ if f_ < 1.0 else MyRd
            if f_ < 1.0:
                cles.append(J.f("M_N_y", "M_N,y,Rd", [_t("M_pl,y,Rd", MyRd / 1e6, "kN·m"), _t("n", n), _t("a", a)], MNy / 1e6, "kN·m", "e_M_N"))
        if Mz > 0:
            if n <= a:
                MNz = MzRd
            else:
                MNz = MzRd * (1.0 - ((n - a) / (1.0 - a)) ** 2)
                cles.append(J.f("M_N_z", "M_N,z,Rd", [_t("M_pl,z,Rd", MzRd / 1e6, "kN·m"), _t("n", n), _t("a", a)], MNz / 1e6, "kN·m", "e_M_Nz_I"))
        alpha, beta = ALPHA_I, max(5.0 * n, 1.0)
        if devie and 5.0 * n > 1.0:
            cles.append(J.f("beta", "β", [_t("n", n)], beta, "", "e_beta_I"))
    else:
        t = s["t"]
        if My > 0:
            aw = coefficient("a_w", "a_w", s["b"], t, "b", "t") if N != 0 else min((A - 2.0 * s["b"] * t) / A, A_MAX)
            G["a_w"] = aw
            f_ = (1.0 - n) / (1.0 - 0.5 * aw)
            MNy = MyRd * f_ if f_ < 1.0 else MyRd
            if f_ < 1.0:
                cles.append(J.f("M_N_y", "M_N,y,Rd", [_t("M_pl,y,Rd", MyRd / 1e6, "kN·m"), _t("n", n), _t("a_w", aw)], MNy / 1e6, "kN·m", "e_M_N"))
        if Mz > 0:
            af = coefficient("a_f", "a_f", s["h"], t, "h", "t") if N != 0 else min((A - 2.0 * s["h"] * t) / A, A_MAX)
            G["a_f"] = af
            f_ = (1.0 - n) / (1.0 - 0.5 * af)
            MNz = MzRd * f_ if f_ < 1.0 else MzRd
            if f_ < 1.0:
                cles.append(J.f("M_N_z", "M_N,z,Rd", [_t("M_pl,z,Rd", MzRd / 1e6, "kN·m"), _t("n", n), _t("a_f", af)], MNz / 1e6, "kN·m", "e_M_N"))
        den = 1.0 - 1.13 * n * n
        if den > 1.66 / ALPHA_TUBE_MAX:
            alpha = beta = 1.66 / den
            if devie:
                cles.append(J.f("alpha_beta", "α = β", [_t("n", n)], alpha, "", "e_alpha_tube"))
        else:
            alpha = beta = ALPHA_TUBE_MAX
    G.update({"M_N_y": MNy, "M_N_z": MNz})
    if devie:
        taux = (My / MNy) ** alpha + (Mz / MNz) ** beta
        G.update({"alpha": alpha, "beta": beta})
        cles.append(J.f("interaction", "η", [_t("M_y,Ed", My / 1e6, "kN·m"), _t("M_N,y,Rd", MNy / 1e6, "kN·m"), _t("α", alpha),
                                             _t("M_z,Ed", Mz / 1e6, "kN·m"), _t("M_N,z,Rd", MNz / 1e6, "kN·m"), _t("β", beta)], taux, "",
                        "e_deviee"))
        J.verif("E-NM", titre, ref + " (6.41)", "(M_y,Ed/M_N,y,Rd)^α + (M_z,Ed/M_N,z,Rd)^β ≤ 1", taux, 1.0, "", cles, taux=taux)
    elif My > 0:
        J.verif("E-NM", titre, ref, "M_y,Ed ≤ M_N,y,Rd", My / 1e6, MNy / 1e6, "kN·m", cles,
                messages=["Effort normal faible : moment plastique non réduit."] if MNy == MyRd else ())
    else:
        J.verif("E-NM", titre, ref, "M_z,Ed ≤ M_N,z,Rd", Mz / 1e6, MNz / 1e6, "kN·m", cles,
                messages=["Effort normal faible : moment plastique non réduit."] if MNz == MzRd else ())


def verifier(barre: Dict) -> Dict:
    """La SECTION vérifiée selon l'Eurocode 3 (§ 6.2).

    ``barre`` (N, mm) : {"section" (acier.barre.section), "nuance" (S235 à S460) ou "fy", "NEd" (compression > 0,
    traction < 0), "MyEd", "MzEd", "VzEd", "VyEd" (efforts de CALCUL ; moments et efforts tranchants en valeur absolue),
    "parametres" ({clé: valeur} : d'autres coefficients que ceux de acier/barre/parametres.py)}.

    Rend la forme de acier/barre/resultat.py ; vérifications E-N, E-My, E-Mz, E-Vz, E-Vy, E-Vb, E-MV, E-NM — seules
    celles dont l'effort existe."""
    d = lire(barre)
    J = Journal(METHODE, NORME)
    ajouter(J, d)
    return J.rendre()

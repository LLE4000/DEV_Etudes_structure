# -*- coding: utf-8 -*-
"""La MÉTHODE HISTORIQUE du bureau : ancienne norme belge NBN B 51-001, contraintes admissibles, charges de SERVICE
(non pondérées), sécurité s = 1,5. Écrite d'après la documentation théorique du bureau (« doc p. X ») et ses classeurs
« Flambement », qu'elle retrouve à 10⁻⁹ près (tests/test_barre_historique.py) ; docs/acier/BARRES_ACIER.md § 4.

Pour chaque axe (y fort, z faible) :

    λ_E = π·√(E / f_y)            λ = u·L / i            λ̄ = λ / λ_E                              (doc p. 1 à 5)
    A′ = 0,5 + α·λ̄²   B′ = β·λ̄²   φ = 1 / (A′ + √(A′² − B′))                                    (doc p. 3)
    P_adm = A·φ·f_y / s                                                                        (doc p. 2, 6)
    P_cr = π²·E·I / (u·L)²        PE = 1 − s·P / P_cr  (> 0)                                    (doc p. 11, 23)

puis, pour la barre :

    σ = P/(A·φ_min) + C_y·M_y/(φ_ed·W_y·PE_y) + C_z·M_z/(W_z·PE_z) ≤ f_y / s                    (doc p. 20, 26)
    τ = V_z/A_âme + V_y/A_ailes        σ_eq = √(σ² + 3·τ²) ≤ f_y / s                            (classeurs)

φ_ed, le déversement (doc p. 15 à 19) : la semelle comprimée et un sixième de l'âme, vus comme une barre comprimée —
I_d = t_f·b³/12, A_d = b·t_f + h_w·t_w/6, λ_ed = L_c/√(I_d/A_d), courbe de l'axe faible. Tube rectangulaire : la
formule des profils fermés (doc p. 18). Tube rond ou carré : pas de déversement.

CE QUE LE MODULE CORRIGE par rapport aux classeurs (chaque point a son contrôle) : P_cr avec (u·L)² ; u et L par axe ;
φ_min dans la flexion composée ; une seule courbe par axe ; des verdicts ; des garde-fous (λ̄ > 1,8, λ > 200, PE ≤ 0,
nuance hors AE235–AE355). CE QU'IL NE REPREND PAS de la doc : la dispense « C/φ_ed = 1 si P/(A·φ) ≤ 0,15·σ_adm »
(annotée « FAUX » p. 20) ; la tolérance de 5 % des exemples (option éteinte).

Le moteur CALCULE et NOMME : ``verifier`` rend les grandeurs (N, mm), les vérifications (valeur, limite, taux), les
blocs de formules (``calcul`` : termes et résultat dans les unités des classeurs — kN, cm, cm⁴, cm³, mm², N/mm² ;
gabarits dans acier/barre/formules.py) et les alertes. L'écran et la note composent et formatent.
"""
import math
from typing import Dict, List, Optional

from . import section as SE

METHODE = "historique"
NORME = "NBN B 51-001"
E = 210000.0                    # N/mm² (doc p. 2)
E_SUR_G = 2.6                   # E/G = 2·(1 + ν), ν = 0,3 (doc p. 18)
S_DEFAUT = 1.5                  # le coefficient de sécurité du « cumul I » (doc p. 6, 11)
S_MIN, S_MAX = 1.0, 3.0
# les trois courbes de flambement : classe → (α, β) (doc p. 3)
COURBES: Dict[str, tuple] = {"a": (0.514, 0.795), "b": (0.554, 0.738), "c": (0.532, 0.377)}
LAMBDA_BAR_TABLEAU = 1.80       # la fin du tableau de la norme (doc p. 5)
LAMBDA_MAX = 200.0              # élancement « formellement déconseillé » au-delà (doc p. 7)
FY_DOC = (235.0, 355.0)         # AE235 à AE355 : les nuances que la doc couvre (p. 3, 5)
RAPPORT_HB = 1.2                # h/b : la limite du tableau des courbes (doc p. 4)
PAROI_FROID = 6.0               # mm : « profil creux fini à froid, paroi ≤ 6 mm » (doc p. 4)
C_CHARGES = 0.85                # charges appliquées sur la barre (doc p. 21)
MODES_C = ("unite", "moments", "charges")
TOLERANCE = 0.05                # la « classe de précision de 5 % » des exemples (doc p. 13, 23) — option, éteinte
OK, NOK, NON_COUVERT = "ok", "nok", "non_couvert"
# le choix de la courbe (doc p. 4) : fabrication → (axe fort, axe faible) ; les I et H laminés dépendent de h/b
COURBES_FABRICATION = {
    "soude_oxycoupe": ("a", "b"), "soude_larges_plats": ("b", "c"), "lamine_semelles_additionnelles": ("b", "a"),
    "recuit": ("a", "b"), "creux_chaud": ("a", "a"), "creux_froid": ("c", "c"), "caisson": ("b", "b"),
    "caisson_recuit": ("a", "a"),
}
NOMS_AXE = {"y": "fort", "z": "faible"}


class Refus(ValueError):
    """Une donnée hors du domaine de la méthode : le calcul est refusé, avec ce message."""


# ------------------------------------------------------------------ la norme
def lambda_E(fy: float) -> float:
    """λ_E = π·√(E / f_y) : 93,9 pour l'AE235, 76,4 pour l'AE355 (doc p. 5)."""
    return math.pi * math.sqrt(E / fy)


def phi(lambda_bar: float, courbe: str) -> float:
    """Le coefficient de flambement φ(λ̄) de la classe ``courbe`` (doc p. 3) ; il redonne les 108 valeurs du tableau de
    la doc p. 5 au millième."""
    alpha, beta = COURBES[courbe]
    a = 0.5 + alpha * lambda_bar * lambda_bar
    b = beta * lambda_bar * lambda_bar
    return 1.0 / (a + math.sqrt(a * a - b))


def coefficient_C(mode: str = "unite", psi: float = 1.0) -> float:
    """Le coefficient C du diagramme de moments (doc p. 20, 21) : 1 (« unite », le choix des classeurs) ; moments
    d'extrémité seuls, C = √(0,3·(1 + ψ²) + 0,4·ψ) avec ψ = M₂/M₁, |M₂| ≤ |M₁| (« moments ») ; 0,85 quand des charges
    sont appliquées sur la barre (« charges »)."""
    if mode == "unite":
        return 1.0
    if mode == "charges":
        return C_CHARGES
    if mode == "moments":
        if not -1.0 <= psi <= 1.0:
            raise Refus(f"ψ = M₂/M₁ doit être entre −1 et 1 (|M₂| ≤ |M₁|) : {psi!r}")
        return math.sqrt(0.3 * (1.0 + psi * psi) + 0.4 * psi)
    raise Refus(f"mode du coefficient C inconnu : {mode!r}")


def courbes(s: Dict) -> Dict:
    """Les courbes de flambement que la norme donne au profil (doc p. 4) : {"y", "z", "regle", "message"}.
    ``message`` : ce que la doc ne dit pas et que le module a dû trancher (None sinon)."""
    forme, fab = s["forme"], s["fabrication"]
    message = None
    if forme in ("U", "L") or (forme == "autre" and fab == "lamine"):
        return {"y": "c", "z": "c", "regle": "té, U ou cornière", "message": None}
    if forme == "I" and fab == "lamine":
        if not s.get("h") or not s.get("b"):
            raise Refus("h et b manquent : la courbe d'un I laminé dépend de h/b")
        if s["h"] / s["b"] > RAPPORT_HB:
            return {"y": "a", "z": "b", "regle": "I ou H laminé, h/b > 1,2", "message": None}
        return {"y": "b", "z": "c", "regle": "I ou H laminé, h/b ≤ 1,2", "message": None}
    if fab not in COURBES_FABRICATION:
        raise Refus(f"la norme ne donne pas de courbe pour : {forme}, {fab}")
    cy, cz = COURBES_FABRICATION[fab]
    regle = {"soude_oxycoupe": "I ou H soudé, semelles oxycoupées", "soude_larges_plats": "I ou H soudé, semelles en larges plats",
             "lamine_semelles_additionnelles": "I ou H laminé avec semelles additionnelles", "recuit": "I ou H recuit",
             "creux_chaud": "tube fini à chaud", "creux_froid": "profil creux fini à froid", "caisson": "caisson soudé non recuit",
             "caisson_recuit": "caisson soudé recuit"}[fab]
    if fab == "creux_froid" and (s.get("t") or 0.0) > PAROI_FROID:
        message = "Creux fini à froid de plus de 6 mm : la norme ne donne pas de courbe ; la courbe c est gardée."
    if fab == "creux_chaud" and forme == "tube_rect":
        message = "Tube rectangulaire fini à chaud : la norme nomme le tube rond ou carré ; la même courbe a est prise."
    return {"y": cy, "z": cz, "regle": regle, "message": message}


def aires_cisaillement(s: Dict) -> Dict[str, Optional[float]]:
    """Les deux aires des classeurs (mm²) : « section âme » pour l'effort tranchant de l'axe fort (V_z), « section
    ailes » pour celui de l'axe faible (V_y). Saisies (``A_ame``, ``A_ailes``) ou, pour un profilé du catalogue :
    I et H — l'aire de cisaillement du catalogue A_v,z (c'est la valeur que les classeurs recopient : 2483 mm² pour le
    HEB 200) et les deux semelles 2·b·t_f ; tube rond — A/2 chacune (classeur du tube 168,3) ; tube carré ou
    rectangulaire — les aires de cisaillement du catalogue."""
    ame, ailes = s.get("A_ame"), s.get("A_ailes")
    if ame is None or ailes is None:
        if s["forme"] == "I":
            auto = (s.get("Avz"), 2.0 * s["b"] * s["tf"] if s.get("b") and s.get("tf") else None)
        elif s["forme"] == "tube_rond":
            auto = (s["A"] / 2.0, s["A"] / 2.0)
        elif s["forme"] in ("tube_carre", "tube_rect"):
            auto = (s.get("Avz"), s.get("Avy"))
        else:
            auto = (None, None)
        ame = auto[0] if ame is None else ame
        ailes = auto[1] if ailes is None else ailes
    return {"ame": ame, "ailes": ailes}


def deversement(s: Dict, Lc: Optional[float], fy: float, courbe_z: str) -> Dict:
    """φ_ed du déversement (doc p. 15 à 19) pour la longueur ``Lc`` (mm) entre maintiens latéraux de la semelle
    comprimée : {"cas", "phi", …}. ``cas`` : « maintenu » (Lc nul ou None : φ_ed = 1), « I » (semelle + âme/6),
    « ferme » (tube rectangulaire, doc p. 18), « sans_objet » (tube rond ou carré), « non_couvert » (autre forme)."""
    if not Lc:
        return {"cas": "maintenu", "phi": 1.0}
    if s["forme"] in ("tube_rond", "tube_carre"):
        return {"cas": "sans_objet", "phi": 1.0}
    lE = lambda_E(fy)
    if s["forme"] == "I":
        hw = SE.hauteur_ame(s)
        if hw is None or not s.get("b") or not s.get("tw"):
            raise Refus("h, b, t_w et t_f manquent : le déversement d'un I en dépend")
        Id = s["tf"] * s["b"] ** 3 / 12.0
        Ad = s["b"] * s["tf"] + hw * s["tw"] / 6.0
        rd = math.sqrt(Id / Ad)
        lam = Lc / rd
        lb = lam / lE
        return {"cas": "I", "phi": phi(lb, courbe_z), "Id": Id, "Ad": Ad, "rd": rd, "hw": hw, "lambda": lam, "lambda_bar": lb,
                "courbe": courbe_z, "Lc": Lc}
    if s["forme"] == "tube_rect":
        if not s.get("It") or not s.get("Wel_y"):
            raise Refus("I_t et W_el,y manquent : le déversement d'un tube rectangulaire en dépend")
        lam = math.sqrt(math.pi * math.sqrt(E_SUR_G) * Lc * s["Wel_y"] / math.sqrt(s["Iz"] * s["It"]))
        lb = lam / lE
        return {"cas": "ferme", "phi": phi(lb, courbe_z), "lambda": lam, "lambda_bar": lb, "courbe": courbe_z, "Lc": Lc}
    return {"cas": "non_couvert", "phi": None}


# ------------------------------------------------------------------ les blocs de formules (unités des classeurs)
def _t(symbole: str, valeur, unite: str = "") -> Dict:
    return {"symbole": symbole, "valeur": valeur, "unite": unite}


def _f(cle: str, symbole: str, termes: List[Dict], resultat, unite: str = "", gabarit: Optional[str] = None) -> Dict:
    """Une formule nommée : ``gabarit`` est la clé de acier/barre/formules.GABARITS (``cle`` à défaut)."""
    return {"cle": cle, "gabarit": gabarit or cle, "symbole": symbole, "termes": termes, "resultat": resultat, "unite": unite}


def _fr(x: float, nd: int = 0) -> str:
    """Un nombre dans une alerte : virgule, espace fine des milliers."""
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


# ------------------------------------------------------------------ la vérification
def _lire(d: Dict) -> Dict:
    """Les données, complétées et contrôlées (une COPIE)."""
    if not isinstance(d, dict) or not isinstance(d.get("section"), dict):
        raise Refus("la barre doit porter une section")
    out = {"section": d["section"], "tolerance": bool(d.get("tolerance", False))}
    for k, defaut in (("fy", None), ("s", S_DEFAUT), ("P", 0.0), ("My", 0.0), ("Mz", 0.0), ("Vy", 0.0), ("Vz", 0.0)):
        v = d.get(k, defaut)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise Refus(f"« {k} » doit être un nombre : {v!r}")
        out[k] = float(v)
    if out["fy"] <= 0:
        raise Refus("la limite élastique f_y doit être positive")
    if not S_MIN <= out["s"] <= S_MAX:
        raise Refus(f"le coefficient de sécurité s doit être entre {S_MIN:g} et {S_MAX:g} : {out['s']!r}")
    axes = d.get("axes") or {}
    out["axes"] = {}
    for a in ("y", "z"):
        ax = dict(axes.get(a) or {})
        L, u = float(ax.get("L", 0.0)), float(ax.get("u", 1.0))
        if L < 0 or u <= 0 or not (math.isfinite(L) and math.isfinite(u)):
            raise Refus(f"axe {a} : la longueur L doit être positive ou nulle et le coefficient u positif")
        courbe = ax.get("courbe")
        if courbe is not None and courbe not in COURBES:
            raise Refus(f"axe {a} : une seule courbe, a, b ou c — reçu {courbe!r}")
        out["axes"][a] = {"L": L, "u": u, "courbe": courbe}
    C = d.get("C") or {}
    out["C"] = {}
    for a in ("y", "z"):
        c = dict(C.get(a) or {})
        mode = c.get("mode", "unite")
        psi = float(c.get("psi", 1.0))
        out["C"][a] = {"mode": mode, "psi": psi, "valeur": coefficient_C(mode, psi)}
    dev = dict(d.get("deversement") or {})
    out["deversement"] = {"actif": bool(dev.get("actif", True)), "Lc": dev.get("Lc")}
    if out["deversement"]["Lc"] is not None:
        Lc = float(out["deversement"]["Lc"])
        if Lc < 0 or not math.isfinite(Lc):
            raise Refus("la distance L_c entre maintiens latéraux doit être positive ou nulle")
        out["deversement"]["Lc"] = Lc
    return out


def verifier(barre: Dict) -> Dict:
    """La barre vérifiée par la méthode historique.

    ``barre`` (N, mm) : {"section" (acier.barre.section), "fy", "s" (1,5), "P" (compression > 0, traction < 0), "My",
    "Mz", "Vz", "Vy" (efforts de SERVICE ; les moments et les efforts tranchants comptent en valeur absolue),
    "axes": {"y": {"L", "u" (1), "courbe" (None : celle de la norme)}, "z": {…}},
    "C": {"y": {"mode": "unite"|"moments"|"charges", "psi"}, "z": {…}},
    "deversement": {"actif" (True), "Lc" (None : la longueur L de l'axe faible)}, "tolerance" (False)}.

    Rend {"methode", "statut" (ok | nok | non_couvert), "taux_max", "gouvernant", "verifications", "grandeurs",
    "calcul", "alertes", "hypotheses"} ; une vérification : {"id", "methode", "titre", "reference", "critere",
    "valeur", "limite", "unite", "taux", "statut", "formules", "messages"}. En compression simple, la ligne des
    contraintes n'est pas rendue : elle répéterait le flambement de l'axe le plus défavorable (σ/σ_adm = P/P_adm) ;
    la contrainte reste dans ``grandeurs``."""
    d = _lire(barre)
    s = d["section"]
    fy, sec, P = d["fy"], d["s"], d["P"]
    My, Mz, Vz, Vy = abs(d["My"]), abs(d["Mz"]), abs(d["Vz"]), abs(d["Vy"])
    A = s["A"]
    alertes: List[Dict] = []
    hyp: List[str] = []

    def alerte(code, niveau, message):
        alertes.append({"code": code, "niveau": niveau, "message": message})

    sigma_adm = fy / sec
    limite = sigma_adm * (1.0 + TOLERANCE) if d["tolerance"] else sigma_adm
    lE = lambda_E(fy)
    try:
        cb = courbes(s)
    except Refus:
        # la norme ne dit rien de ce profil (ou h et b manquent) : le calcul reste possible si les DEUX courbes sont données
        if d["axes"]["y"]["courbe"] is None or d["axes"]["z"]["courbe"] is None:
            raise
        cb = {"y": None, "z": None, "regle": "courbes données", "message": None}
    if cb["message"]:
        alerte("courbe", "info", cb["message"])
    comprime = P > 0
    G: Dict = {"fy": fy, "s": sec, "sigma_adm": sigma_adm, "limite": limite, "lambda_E": lE, "P": P, "My": My, "Mz": Mz, "Vz": Vz,
               "Vy": Vy, "A": A, "regle_courbes": cb["regle"]}
    calcul: List[Dict] = [_f("sigma_adm", "σ_adm", [_t("f_y", fy, "N/mm²"), _t("s", sec)], sigma_adm, "N/mm²"),
                          _f("lambda_E", "λ_E", [_t("E", E, "N/mm²"), _t("f_y", fy, "N/mm²")], lE)]
    if not (FY_DOC[0] <= fy <= FY_DOC[1]):
        alerte("nuance", "attention", f"f_y = {_fr(fy)} N/mm² : hors des nuances AE235 à AE355 de la norme (extrapolation).")
    if d["tolerance"]:
        alerte("tolerance", "attention", "Tolérance de 5 % appliquée à la contrainte admissible (exemples de la documentation).")

    # ---- chaque axe : élancement, φ, charge admissible, charge critique, facteur du second ordre
    for a in ("y", "z"):
        ax = d["axes"][a]
        nom = s["axes"][0 if a == "y" else 1]
        i, I = s[f"i{a}"], s[f"I{a}"]
        courbe = ax["courbe"] or cb[a]
        Lf = ax["u"] * ax["L"]
        lam = Lf / i
        lb = lam / lE
        ph = phi(lb, courbe)
        Padm = A * ph * fy / sec
        Pcr = math.pi ** 2 * E * I / Lf ** 2 if Lf > 0 else None
        PE = 1.0 if (Pcr is None or not comprime) else 1.0 - sec * P / Pcr
        G[a] = {"nom": nom, "L": ax["L"], "u": ax["u"], "Lf": Lf, "i": i, "I": I, "courbe": courbe, "courbe_norme": cb[a],
                "courbe_forcee": ax["courbe"] is not None and cb[a] is not None and ax["courbe"] != cb[a], "lambda": lam,
                "lambda_bar": lb,
                "alpha": COURBES[courbe][0], "beta": COURBES[courbe][1], "A_prime": 0.5 + COURBES[courbe][0] * lb * lb,
                "B_prime": COURBES[courbe][1] * lb * lb, "phi": ph, "P_adm": Padm, "P_cr": Pcr, "PE": PE,
                "reserve": Padm / P if comprime else None}
        g = G[a]
        calcul += [
            _f(f"lambda_{a}", f"λ_{a}", [_t("u", ax["u"]), _t("L", ax["L"] / 10.0, "cm"), _t(f"i_{a}", i / 10.0, "cm")], lam, "", "lambda"),
            _f(f"lambda_bar_{a}", f"λ̄_{a}", [_t(f"λ_{a}", lam), _t("λ_E", lE)], lb, "", "lambda_bar"),
            _f(f"A_prime_{a}", f"A′_{a}", [_t("α", g["alpha"]), _t(f"λ̄_{a}", lb)], g["A_prime"], "", "A_prime"),
            _f(f"B_prime_{a}", f"B′_{a}", [_t("β", g["beta"]), _t(f"λ̄_{a}", lb)], g["B_prime"], "", "B_prime"),
            _f(f"phi_{a}", f"φ_{a}", [_t(f"A′_{a}", g["A_prime"]), _t(f"B′_{a}", g["B_prime"])], ph, "", "phi"),
            _f(f"P_adm_{a}", f"P_adm,{a}", [_t("A", A, "mm²"), _t(f"φ_{a}", ph), _t("f_y", fy, "N/mm²"), _t("s", sec)], Padm / 1e3, "kN", "P_adm"),
        ]
        if Pcr is not None:
            calcul.append(_f(f"P_cr_{a}", f"P_cr,{a}", [_t("E", E / 10.0, "kN/cm²"), _t(f"I_{a}", I / 1e4, "cm⁴"), _t("u", ax["u"]),
                                                       _t("L", ax["L"] / 10.0, "cm")], Pcr / 1e3, "kN", "P_cr"))
            if comprime:
                calcul.append(_f(f"PE_{a}", f"PE_{a}", [_t("s", sec), _t("P", P / 1e3, "kN"), _t(f"P_cr,{a}", Pcr / 1e3, "kN")], PE, "", "PE"))
        if g["courbe_forcee"]:
            alerte(f"courbe_{a}", "attention", f"Axe {nom} : courbe {courbe} imposée (la norme donne {cb[a]} : {cb['regle']}).")
        if comprime:
            if lb > LAMBDA_BAR_TABLEAU:
                alerte(f"tableau_{a}", "attention", f"Axe {nom} : λ̄ = {_fr(lb, 2)} > 1,8, au-delà du tableau de la norme.")
            if lam > LAMBDA_MAX:
                alerte(f"elancement_{a}", "attention", f"Axe {nom} : λ = {_fr(lam)} > 200, élancement déconseillé.")
            if PE <= 0:
                Imin = sec * P * Lf ** 2 / (math.pi ** 2 * E)
                alerte(f"second_ordre_{a}", "erreur", f"Axe {nom} : s·P ≥ P_cr, la barre flambe. Inertie minimale : "
                                                     f"{_fr(Imin / 1e4)} cm⁴ (le profil en a {_fr(I / 1e4)}).")
                g["I_min"] = Imin
    phi_min = min(G["y"]["phi"], G["z"]["phi"]) if comprime else 1.0
    G["phi_min"] = phi_min
    G["axe_phi_min"] = ("z" if G["z"]["phi"] <= G["y"]["phi"] else "y") if comprime else None

    # ---- le déversement (le terme de moment de l'axe fort)
    couvert = s["forme"] in ("I",) + SE.TUBES
    if My > 0 and d["deversement"]["actif"] and couvert:
        Lc = d["deversement"]["Lc"] if d["deversement"]["Lc"] is not None else d["axes"]["z"]["L"]
        dev = deversement(s, Lc, fy, G["z"]["courbe"])
    else:
        dev = {"cas": "maintenu" if (My > 0 and couvert) else "sans_objet", "phi": 1.0}
    G["deversement"] = dev
    phi_ed = dev["phi"]
    if dev["cas"] == "I":
        calcul += [
            _f("I_d", "I_d", [_t("t_f", s["tf"] / 10.0, "cm"), _t("b", s["b"] / 10.0, "cm")], dev["Id"] / 1e4, "cm⁴"),
            _f("A_d", "A_d", [_t("b", s["b"] / 10.0, "cm"), _t("t_f", s["tf"] / 10.0, "cm"), _t("h_w", dev["hw"] / 10.0, "cm"),
                              _t("t_w", s["tw"] / 10.0, "cm")], dev["Ad"] / 1e2, "cm²"),
            _f("r_d", "r_d", [_t("I_d", dev["Id"] / 1e4, "cm⁴"), _t("A_d", dev["Ad"] / 1e2, "cm²")], dev["rd"] / 10.0, "cm"),
            _f("lambda_ed", "λ_ed", [_t("L_c", dev["Lc"] / 10.0, "cm"), _t("r_d", dev["rd"] / 10.0, "cm")], dev["lambda"]),
        ]
    elif dev["cas"] == "ferme":
        calcul.append(_f("lambda_ed", "λ_ed", [_t("L_c", dev["Lc"] / 10.0, "cm"), _t("W_y", s["Wel_y"] / 1e3, "cm³"),
                                               _t("I_z", s["Iz"] / 1e4, "cm⁴"), _t("I_t", s["It"] / 1e4, "cm⁴")], dev["lambda"], "",
                         "lambda_ed_ferme"))
    if dev["cas"] in ("I", "ferme"):
        al, be = COURBES[dev["courbe"]]
        lbd = dev["lambda_bar"]
        calcul += [
            _f("lambda_bar_ed", "λ̄_ed", [_t("λ_ed", dev["lambda"]), _t("λ_E", lE)], lbd, "", "lambda_bar"),
            _f("A_prime_ed", "A′_ed", [_t("α", al), _t("λ̄_ed", lbd)], 0.5 + al * lbd * lbd, "", "A_prime"),
            _f("B_prime_ed", "B′_ed", [_t("β", be), _t("λ̄_ed", lbd)], be * lbd * lbd, "", "B_prime"),
            _f("phi_ed", "φ_ed", [_t("A′_ed", 0.5 + al * lbd * lbd), _t("B′_ed", be * lbd * lbd)], phi_ed, "", "phi"),
        ]
        if lbd > LAMBDA_BAR_TABLEAU:
            alerte("tableau_ed", "attention", f"Déversement : λ̄_ed = {_fr(lbd, 2)} > 1,8, au-delà du tableau de la norme.")
        hyp.append(f"Déversement : semelle comprimée maintenue tous les {_fr(dev['Lc'] / 1e3, 2)} m.")
    elif My > 0 and couvert and dev["cas"] == "maintenu":
        hyp.append("Déversement empêché : semelle comprimée maintenue en continu (φ_ed = 1).")

    # ---- les contraintes
    aires = aires_cisaillement(s)
    G["A_ame"], G["A_ailes"] = aires["ame"], aires["ailes"]
    Cy, Cz = d["C"]["y"]["valeur"], d["C"]["z"]["valeur"]
    G["C_y"], G["C_z"] = Cy, Cz
    flechi = My > 0 or Mz > 0
    cisaille = Vz > 0 or Vy > 0
    # PE ≤ 0 n'arrête le calcul des contraintes que s'il divise un moment : en compression simple, la barre qui flambe
    # se lit déjà dans P > P_adm (A·φ·f_y ≤ P_cr toujours : φ ≤ 1/λ̄²)
    instable = comprime and ((My > 0 and G["y"]["PE"] <= 0) or (Mz > 0 and G["z"]["PE"] <= 0))
    non_couvert = (flechi or cisaille) and not couvert
    sigma = sigma_eq = tau = None
    if not instable and not non_couvert:
        manque = [n for n, v, m in (("W_el,y", s.get("Wel_y"), My), ("W_el,z", s.get("Wel_z"), Mz), ("A_âme", aires["ame"], Vz),
                                    ("A_ailes", aires["ailes"], Vy)) if m > 0 and not v]
        if manque:
            raise Refus(f"propriété(s) manquante(s) pour ces efforts : {', '.join(manque)}")
        sP = abs(P) / (A * phi_min)
        sMy = Cy * My / (phi_ed * s["Wel_y"] * G["y"]["PE"]) if My > 0 else 0.0
        sMz = Cz * Mz / (s["Wel_z"] * G["z"]["PE"]) if Mz > 0 else 0.0
        sigma = sP + sMy + sMz
        tz = Vz / aires["ame"] if Vz > 0 else 0.0
        ty = Vy / aires["ailes"] if Vy > 0 else 0.0
        tau = tz + ty
        sigma_eq = math.sqrt(sigma * sigma + 3.0 * tau * tau)
        G.update({"sigma_P": sP, "sigma_My": sMy, "sigma_Mz": sMz, "sigma": sigma, "tau_z": tz, "tau_y": ty, "tau": tau,
                  "sigma_eq": sigma_eq,
                  # les sommes « par axe » des classeurs (φ de l'axe, sans déversement) : gardées pour la comparaison
                  "sigma_axe_y": abs(P) / (A * (G["y"]["phi"] if comprime else 1.0)) + (Cy * My / (s["Wel_y"] * G["y"]["PE"]) if My > 0 else 0.0),
                  "sigma_axe_z": abs(P) / (A * (G["z"]["phi"] if comprime else 1.0)) + sMz})
        if P != 0:
            calcul.append(_f("sigma_P", "σ_P", [_t("P", abs(P) / 1e3, "kN"), _t("A", A, "mm²"), _t("φ_min", phi_min)], sP, "N/mm²"))
        if My > 0:
            calcul.append(_f("sigma_My", "σ_My", [_t("C_y", Cy), _t("M_y", My / 1e6, "kN·m"), _t("φ_ed", phi_ed),
                                                  _t("W_y", s["Wel_y"] / 1e3, "cm³"), _t("PE_y", G["y"]["PE"])], sMy, "N/mm²"))
        if Mz > 0:
            calcul.append(_f("sigma_Mz", "σ_Mz", [_t("C_z", Cz), _t("M_z", Mz / 1e6, "kN·m"), _t("W_z", s["Wel_z"] / 1e3, "cm³"),
                                                  _t("PE_z", G["z"]["PE"])], sMz, "N/mm²"))
        calcul.append(_f("sigma", "σ", [_t("σ_P", sP, "N/mm²"), _t("σ_My", sMy, "N/mm²"), _t("σ_Mz", sMz, "N/mm²")], sigma, "N/mm²"))
        if cisaille:
            if Vz > 0:
                calcul.append(_f("tau_z", "τ_z", [_t("V_z", Vz / 1e3, "kN"), _t("A_âme", aires["ame"], "mm²")], tz, "N/mm²", "tau"))
            if Vy > 0:
                calcul.append(_f("tau_y", "τ_y", [_t("V_y", Vy / 1e3, "kN"), _t("A_ailes", aires["ailes"], "mm²")], ty, "N/mm²", "tau"))
            calcul.append(_f("sigma_eq", "σ_eq", [_t("σ", sigma, "N/mm²"), _t("τ", tau, "N/mm²")], sigma_eq, "N/mm²"))
    if P < 0:
        alerte("traction", "info", "Traction : résistance simple σ = N/A + M/W (la norme ne traite que la compression).")
    if non_couvert:
        alerte("forme", "attention", "Cornière, U ou té : seule la compression simple est vérifiée (flexion et effort tranchant non couverts).")

    # ---- les vérifications
    verifs: List[Dict] = []

    def verif(id_, titre, reference, critere, valeur, lim, unite, formules, taux=None, statut=None, messages=()):
        if statut is None:
            taux = valeur / lim
            statut = OK if taux <= 1.0 else NOK
        verifs.append({"id": id_, "methode": METHODE, "titre": titre, "reference": reference, "critere": critere, "valeur": valeur,
                       "limite": lim, "unite": unite, "taux": taux, "statut": statut, "formules": list(formules),
                       "messages": list(messages)})

    if comprime:
        for a in ("z", "y"):
            g = G[a]
            verif(f"H-N-{a}", f"Flambement, axe {NOMS_AXE[a]} {g['nom']}", f"{NORME} · doc p. 2 à 6", "P ≤ P_adm", P / 1e3, g["P_adm"] / 1e3, "kN",
                  ["lambda_E", f"lambda_{a}", f"lambda_bar_{a}", f"A_prime_{a}", f"B_prime_{a}", f"phi_{a}", f"P_adm_{a}"])
    cles_axes = [k for a in ("y", "z") for k in (f"P_cr_{a}", f"PE_{a}")]
    cles_dev = ["I_d", "A_d", "r_d", "lambda_ed", "lambda_bar_ed", "A_prime_ed", "B_prime_ed", "phi_ed"]
    presentes = {f["cle"] for f in calcul}
    if non_couvert:                     # (avant l'arrêt du second ordre : ce que la méthode ne couvre pas n'est pas jugé)
        verif("H-SIGMA", "Flexion et effort tranchant", NORME, "σ ≤ σ_adm", None, limite, "N/mm²", [], taux=None, statut=NON_COUVERT,
              messages=[x["message"] for x in alertes if x["code"] == "forme"])
    elif instable:
        verif("H-SIGMA", "Flexion composée", f"{NORME} · doc p. 11", "PE > 0", None, limite, "N/mm²",
              [k for k in cles_axes if k in presentes], taux=None, statut=NOK,
              messages=[x["message"] for x in alertes if x["code"].startswith("second_ordre")])
    else:
        if flechi or P < 0:
            devie = My > 0 and Mz > 0
            if comprime:
                titre = "Flexion composée déviée" if devie else "Flexion composée"
            elif P < 0:
                titre = "Traction et flexion" if flechi else "Traction"
            else:
                titre = "Flexion déviée" if devie else "Flexion"
            ref = f"{NORME} · doc p. 26" if (comprime and devie) else f"{NORME} · doc p. 20" if comprime else NORME
            cles = ["sigma_adm"] + [k for k in cles_axes + cles_dev + ["sigma_P", "sigma_My", "sigma_Mz", "sigma"] if k in presentes]
            verif("H-SIGMA", titre, ref, "σ ≤ σ_adm", sigma, limite, "N/mm²", cles)
        if cisaille:
            verif("H-EQ", "Contrainte équivalente", f"{NORME} · classeur", "σ_eq ≤ σ_adm", sigma_eq, limite, "N/mm²",
                  ["sigma_adm"] + [k for k in ("sigma", "tau_z", "tau_y", "sigma_eq") if k in presentes])
    for v in verifs:
        if v["taux"] is not None:
            calcul.append(_f(f"taux_{v['id']}", "η", [_t("valeur", v["valeur"], v["unite"]), _t("limite", v["limite"], v["unite"])],
                             v["taux"], "", "taux"))
            v["formules"].append(f"taux_{v['id']}")

    # ---- la synthèse
    hyp.insert(0, f"Charges de service, non pondérées ; contrainte admissible σ_adm = f_y / s = {_fr(fy)} / {_fr(sec, 2).rstrip('0').rstrip(',')}"
                  f" = {_fr(sigma_adm, 1)} N/mm².")
    hyp.append(f"Courbes de flambement : {G['y']['courbe']} (axe fort), {G['z']['courbe']} (axe faible) — {cb['regle']}.")
    if flechi and (Cy != 1.0 or Cz != 1.0):
        hyp.append(f"Coefficient C : {_fr(Cy, 2)} (axe fort), {_fr(Cz, 2)} (axe faible).")
    juges = [v for v in verifs if v["statut"] in (OK, NOK)]
    if any(v["statut"] == NOK for v in juges):
        statut = NOK
    elif any(v["statut"] == NON_COUVERT for v in verifs):
        statut = NON_COUVERT            # rien ne dépasse, mais une vérification manque : jamais « ok »
    else:
        statut = OK
    avec_taux = [v for v in juges if v["taux"] is not None]
    gouv = max(avec_taux, key=lambda v: v["taux"]) if avec_taux else None
    sans_taux = [v for v in juges if v["taux"] is None]
    if sans_taux:
        gouv = sans_taux[0]
    return {"methode": METHODE, "norme": NORME, "statut": statut, "taux_max": gouv["taux"] if gouv else None,
            "gouvernant": gouv["id"] if gouv else None, "verifications": verifs, "grandeurs": G, "calcul": calcul, "alertes": alertes,
            "hypotheses": hyp}

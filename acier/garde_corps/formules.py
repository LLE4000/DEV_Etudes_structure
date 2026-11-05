# -*- coding: utf-8 -*-
"""Les formules de la vérification d'un garde-corps, écrites comme à la main — note (page 1)
et écran (onglet Détail). Chaque ligne est une substitution « grandeur = formule = valeurs =
résultat » (``Sub`` du kit des assemblages) dont ``verifier(sub)`` ré-évalue le gabarit
imprimé pour retrouver le nombre du moteur : chaque formule de la note est TESTABLE
(tests/test_garde_corps_formules.py). Rien n'est recalculé : les nombres sont ceux de
``etat.verifier`` ; les formules ne font que les relier.

Chaque élément rend un BLOC : ``{"cas": libellé du cas de charge qui gouverne, "autres":
les autres cas et leur taux, "cas_taux": [(libellé, taux), …], "groupes": [(libellé, [Sub, …]),
…], "subs": [Sub, …]}``. Les GROUPES suivent l'ordre de lecture d'une vérification — charges,
sollicitations, contraintes (avec leur limite et leur taux), flèche, taux — et ``subs`` les met à
plat dans le même ordre. Unités imprimées : kN, kN·m, mm, N/mm², cm³ → les gabarits portent les
facteurs 10³ et 10⁶ (``*1000``, ``*1e6``) que le kit efface de la forme littérale et imprime dans
la forme numérique.
"""
from typing import Dict, List

from acier.assemblages.poutre_poutre.doubles_cornieres.formules import Sub, fmt, morceaux, nombre, rendre, verifier  # noqa: F401 (ré-exportés)
from acier.soudures import formules as FOS

from . import charges as CH
from . import fixation as FX
from . import membres as ME

CRITERES_MATH = {
    "els": r"σ = \frac{M}{W_{el}} + \frac{N}{A} \le σ_{adm} \t{ ; } τ = \frac{V}{A_v} \le τ_{adm} \t{ ; } f \le f_{adm}",
    "eurocode": r"σ_{Ed} = \frac{γ_Q \cdot M}{W_{el}} + \frac{γ_Q \cdot N}{A} \le \frac{f_y}{γ_{M0}} \t{ ; } τ_{Ed} = \frac{γ_Q \cdot V}{A_v} \le \frac{f_y}{\sqrt{3} \cdot γ_{M0}} \t{ ; } f \le f_{adm}",
}
TITRES = {"els": "Vérification à l'ELS (charges de service, contraintes admissibles)",
          "eurocode": "Vérification à l'ELU selon l'Eurocode (charges de calcul, EN 1993-1-1 §6.2, élastique)"}


CHARGES, SOLLICITATIONS, CONTRAINTES, FLECHE, TAUX = "Charges", "Sollicitations", "Contraintes", "Flèche", "Taux"


def _bloc(cas: str, cas_taux: Dict, groupes: List) -> Dict:
    """Le bloc d'un élément : cas gouvernant, autres cas et leur taux, les formules par GROUPE
    dans l'ordre de lecture (les groupes vides disparaissent) et à plat."""
    groupes = [(lib, [x for x in subs if x is not None]) for lib, subs in groupes]
    groupes = [(lib, subs) for lib, subs in groupes if subs]
    autres_l = [(CH.CAS_LIBELLES.get(k, k), t) for k, t in (cas_taux or {}).items() if k != cas]
    autres = [f"{lib} : η = {t * 100:.0f} %" for lib, t in autres_l]
    return {"cas": CH.CAS_LIBELLES.get(cas, cas), "autres": " ; ".join(autres), "cas_taux": autres_l, "groupes": groupes,
            "subs": [x for _, subs in groupes for x in subs]}


def _vide() -> Dict:
    return {"cas": "", "autres": "", "cas_taux": [], "groupes": [], "subs": []}


def _arr(v: float, nd: int) -> float:
    """La valeur telle qu'elle est IMPRIMÉE à ``nd`` décimales (pour qu'un taux reprenne le nombre
    de la ligne précédente), sauf si l'arrondi la dénature (valeur minuscule)."""
    r = nombre(fmt(v, nd))
    return r if r != 0 and abs(r - v) <= 0.01 * abs(v) else v


def _gq(v: Dict):
    g = v["gamma"]
    return ({"g": ("γ_Q", g)} if g != 1.0 else {}), ("g*" if g != 1.0 else "")


def _sym_lim(res: Dict) -> str:
    """Le nom de la limite de contrainte de la méthode : σ_adm à l'ELS, σ_lim en Eurocode."""
    return "σ_lim" if (res.get("methode") or {}).get("mode") == ME.EUROCODE else "σ_adm"


def _limites(v: Dict, meth: Dict, fy: float) -> Dict:
    """Les limites et les taux d'une vérification de section : ``s_lim``/``t_lim`` (Eurocode : f_y/γ_M0,
    f_y/(√3·γ_M0)), ``eta_s``, ``eta_t``, ``f_adm``, ``eta_f``, ``eta`` — chaque taux reprend le nombre
    imprimé de la grandeur qu'il compare."""
    ec = meth.get("mode") == ME.EUROCODE
    s_sym, t_sym = ("σ_lim", "τ_lim") if ec else ("σ_adm", "τ_adm")
    out = {"s_lim": None, "t_lim": None}
    if ec:
        gM0 = float(meth.get("gamma_M0") or 1.0)
        out["s_lim"] = Sub(s_sym, "fy/gM0", {"fy": ("f_y", fy), "gM0": ("γ_M0", gM0)}, v["lim_sigma"], "N/mm²", 1)
        out["t_lim"] = Sub(t_sym, "fy/(sqrt(3)*gM0)", {"fy": ("f_y", fy), "gM0": ("γ_M0", gM0)}, v["lim_tau"], "N/mm²", 1)
    # à l'ELS, σ_adm = f_y/γ_adm et τ_adm = σ_adm/√3 sont écrites une fois, dans les hypothèses de la note
    out["eta_s"] = Sub("η_σ", "s/lim", {"s": ("σ", _arr(v["sigma"], 1)), "lim": (s_sym, _arr(v["lim_sigma"], 1))}, v["taux"]["sigma"], "", -1)
    out["eta_t"] = Sub("η_τ", "t/lim", {"t": ("τ", _arr(v["tau"], 2)), "lim": (t_sym, _arr(v["lim_tau"], 1))}, v["taux"]["tau"], "", -1)
    out["s_eq"] = Sub("σ_eq", "sqrt(s**2 + 3*t**2)", {"s": ("σ", _arr(v["sigma"], 1)), "t": ("τ", _arr(v["tau"], 2))}, v["sigma_eq"], "N/mm²", 1)
    out["eta_eq"] = Sub("η_eq", "se/lim", {"se": ("σ_eq", _arr(v["sigma_eq"], 1)), "lim": (s_sym, _arr(v["lim_sigma"], 1))}, v["taux"]["sigma_eq"], "", -1)
    if meth.get("fleche_mode") == "mm":
        out["f_adm"] = Sub("f_adm", "", {}, v["fleche_lim"], "mm", 1)
    else:
        out["f_adm"] = Sub("f_adm", "L/x", {"L": ("L", v["L"]), "x": ("x", float(meth.get("fleche_val") or ME.FLECHE_X_DEFAUT))}, v["fleche_lim"], "mm", 1)
    out["eta_f"] = Sub("η_f", "f/fa", {"f": ("f", _arr(v["fleche"], 2)), "fa": ("f_adm", _arr(v["fleche_lim"], 1))}, v["taux"]["fleche"], "", -1)
    # σ_eq ≥ σ avec la même limite : η_σ n'est jamais le pire, le taux retient η_τ, η_eq et η_f
    out["eta"] = Sub("η", "max(b, c, d)", {"b": ("η_τ", v["taux"]["tau"]), "c": ("η_eq", v["taux"]["sigma_eq"]), "d": ("η_f", v["taux"]["fleche"])},
                     v["taux_max"], "", -1)
    return out


def _groupes(charges, sollic, sigma, tau, fleches, lim) -> List:
    """Les cinq groupes d'une vérification de section, dans l'ordre de lecture : σ (et sa limite en
    Eurocode), τ, sa limite et son taux, puis la contrainte équivalente σ_eq = √(σ² + 3τ²) et son taux
    (σ_eq ≥ σ avec la même limite : le taux de σ seule n'est pas imprimé)."""
    return [(CHARGES, charges), (SOLLICITATIONS, sollic),
            (CONTRAINTES, [sigma, lim["s_lim"], tau, lim["t_lim"], lim["eta_t"], lim["s_eq"], lim["eta_eq"]]),
            (FLECHE, list(fleches) + [lim["f_adm"], lim["eta_f"]]), (TAUX, [lim["eta"]])]


# ------------------------------------------------------------------ 1 · montant principal
def montant(res: Dict) -> Dict:
    m = res.get("montant")
    if not m:
        return _vide()
    v, sec, ch, meth = m["verif"], m["section"], res["charges"], res["methode"]
    E = float(meth.get("E") or ME.E_ACIER)
    cas = v["cas_charge"]
    trib = v["trib1"] + v["trib2"]
    inv = v.get("orientation_inversee")
    gq, pref = _gq(v)
    charges, sollic, fleches = [], [], []
    coin = v["P2"] > 0 or v["trib2"] > 0
    s1, s2 = (v["trib2"], v["trib1"]) if inv else (v["trib1"], v["trib2"])    # la longueur tributaire de chaque direction
    if v["P1"] > 0 or v["P2"] > 0:
        if ch["mode"] == "P":
            charges.append(Sub("P₁", "", {}, v["P1"] / 1e3, "kN", 2))
            if v["P2"] > 0:
                charges.append(Sub("P₂", "", {}, v["P2"] / 1e3, "kN", 2))
        else:
            charges.append(Sub("P₁", "qh*s/1000", {"qh": ("q_h", ch["qh"]), "s": ("s_trib,1" if coin else "s_trib", s1)}, v["P1"] / 1e3, "kN", 3))
            if v["P2"] > 0:
                charges.append(Sub("P₂", "qh*s/1000", {"qh": ("q_h", ch["qh"]), "s": ("s_trib,2", s2)}, v["P2"] / 1e3, "kN", 3))
        sollic.append(Sub("M₁", "P*H/1000", {"P": ("P₁", v["P1"] / 1e3), "H": ("H", v["H"])}, v["M1"] / 1e6, "kN·m", 3))
        if v["P2"] > 0:
            sollic.append(Sub("M₂", "P*H/1000", {"P": ("P₂", v["P2"] / 1e3), "H": ("H", v["H"])}, v["M2"] / 1e6, "kN·m", 3))
            sollic.append(Sub("V₁", "", {}, v["V1"] / 1e3, "kN", 3))
            sollic.append(Sub("V₂", "", {}, v["V2"] / 1e3, "kN", 3))
        else:
            sollic.append(Sub("V", "", {}, v["V1"] / 1e3, "kN", 3))
    if v["N"] > 0:
        if coin:
            charges.append(Sub("N", "qv*(s1 + s2)/1000", {"qv": ("q_v", ch["qv"]), "s1": ("s_trib,1", s1), "s2": ("s_trib,2", s2)}, v["N"] / 1e3, "kN", 3))
        else:
            charges.append(Sub("N", "qv*s/1000", {"qv": ("q_v", ch["qv"]), "s": ("s_trib", trib)}, v["N"] / 1e3, "kN", 3))
    termes, vals = [], dict(gq)
    if v["P1"] > 0:
        termes.append(f"{pref}M1*1e6/Wy")
        vals.update({"M1": ("M₁", v["M1"] / 1e6), "Wy": ("W_el,y", v["Wy"])})
    if v["P2"] > 0:
        termes.append(f"{pref}M2*1e6/Wz")
        vals.update({"M2": ("M₂", v["M2"] / 1e6), "Wz": ("W_el,z", v["Wz"])})
    if v["N"] > 0:
        termes.append(f"{pref}N*1000/A")
        vals.update({"N": ("N", v["N"] / 1e3), "A": ("A", v["A"])})
    sigma = Sub("σ", " + ".join(termes), vals, v["sigma"], "N/mm²", 1) if termes else None
    tau = None
    if v["V1"] > 0 or v["V2"] > 0:
        # le cisaillement qui gouverne : V₁ sur A_v,z ou, pour un montant d'angle, V₂ sur A_v,y — celui que le moteur retient
        t1, t2 = v["gamma"] * v["V1"] / (sec["Avz"] * 1e2), (v["gamma"] * v["V2"] / (sec["Avy"] * 1e2) if v["V2"] > 0 else 0.0)
        if v["P2"] > 0 and t2 > t1:
            tau = Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V₂", v["V2"] / 1e3), "Av": ("A_v,y", sec["Avy"] * 1e2)}, t2, "N/mm²", 2)
        else:
            tau = Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V₁" if v["P2"] > 0 else "V", v["V1"] / 1e3), "Av": ("A_v" if v["P2"] <= 0 else "A_v,z", sec["Avz"] * 1e2)},
                      t1, "N/mm²", 2)
        fleches.append(Sub("f₁" if v["P2"] > 0 else "f", "P*1000*H**3/(3*E*I)", {"P": ("P₁", v["P1"] / 1e3), "H": ("H", v["H"]), "E": ("E", E), "I": ("I_y", sec["Iy"] * 1e4)}, v["f1"], "mm", 2))
        if v["P2"] > 0:
            fleches.append(Sub("f₂", "P*1000*H**3/(3*E*I)", {"P": ("P₂", v["P2"] / 1e3), "H": ("H", v["H"]), "E": ("E", E), "I": ("I_z", sec["Iz"] * 1e4)}, v["f2"], "mm", 2))
            fleches.append(Sub("f_rés", "sqrt(f1**2 + f2**2)", {"f1": ("f₁", v["f1"]), "f2": ("f₂", v["f2"])}, v["fleche"], "mm", 2))
    lim = _limites(v, meth, res["acier"]["fy"])
    return _bloc(cas, v.get("cas_taux"), _groupes(charges, sollic, sigma, tau, fleches, lim))


# ------------------------------------------------------------------ 2 · main courante
def lisse(res: Dict) -> Dict:
    li = res.get("lisse")
    if not li:
        return _vide()
    v, sec, ch, meth = li["verif"], li["section"], res["charges"], res["methode"]
    E = float(meth.get("E") or ME.E_ACIER)
    L, n, cont = v["L"], v.get("n_travees", 1), v.get("appuis") == "continue" and v.get("n_travees", 1) >= 2
    cas = v["cas_charge"]
    gq, pref = _gq(v)
    charges, sollic, fleches = [], [], []
    if ch["mode"] == "P":
        charges.append(Sub("q_h", "P/s*1000", {"P": ("P", ch["P"]), "s": ("s", L)}, v["qh"], "kN/m", 3))
    kM, kV, kf = (v["kM"], v["kV"], v["kf"]) if cont else (1 / 8.0, 0.5, 5 / 384.0)
    if cas in ("1", "1+2"):
        if cont:
            sollic.append(Sub("M_h", "kM*qh*L**2/1e6" + (" + Q*L/4/1000" if v["Q"] > 0 else ""), {"kM": ("k_M", kM), "qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Mh"] / 1e6, "kN·m", 3))
        elif v["Q"] > 0 and v["comb"] == "max":
            sollic.append(Sub("M_h", "max(qh*L**2/8/1e6, Q*L/4/1000)", {"qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Mh"] / 1e6, "kN·m", 3))
        else:
            sollic.append(Sub("M_h", "qh*L**2/8/1e6" + (" + Q*L/4/1000" if v["Q"] > 0 else ""), {"qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Mh"] / 1e6, "kN·m", 3))
    if cas in ("2", "1+2"):
        if cont:
            sollic.append(Sub("M_v", "kM*qv*L**2/1e6", {"kM": ("k_M", kM), "qv": ("q_v", v["qv"]), "L": ("L", L)}, v["Mv"] / 1e6, "kN·m", 3))
        else:
            sollic.append(Sub("M_v", "qv*L**2/8/1e6", {"qv": ("q_v", v["qv"]), "L": ("L", L)}, v["Mv"] / 1e6, "kN·m", 3))
    tau = None
    if cas == "1":
        sigma = Sub("σ", f"{pref}Mh*1e6/Wz", {**gq, "Mh": ("M_h", v["Mh"] / 1e6), "Wz": ("W_el,z", v["Wz"])}, v["sigma"], "N/mm²", 1)
        sollic.append(Sub("V", ("kV*qh*L/1000" if cont else "qh*L/2/1000") + (" + Q/2" if v["Q"] > 0 else ""), {"kV": ("k_V", kV), "qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Vh"] / 1e3, "kN", 3))
        tau = Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V", v["Vh"] / 1e3), "Av": ("A_v,y", sec["Avy"] * 1e2)}, v["tau"], "N/mm²", 2)
        fleches.append(Sub("f", ("kf*qh*L**4/(E*I)" if cont else "5*qh*L**4/(384*E*I)") + (" + Q*1000*L**3/(48*E*I)" if v["Q"] > 0 else ""),
                           {"kf": ("k_f", kf), "qh": ("q_h", v["qh"]), "L": ("L", L), "E": ("E", E), "I": ("I_z", sec["Iz"] * 1e4), "Q": ("Q", v["Q"] / 1e3)}, v["fleche"], "mm", 2))
    elif cas == "2":
        sigma = Sub("σ", f"{pref}Mv*1e6/Wy", {**gq, "Mv": ("M_v", v["Mv"] / 1e6), "Wy": ("W_el,y", v["Wy"])}, v["sigma"], "N/mm²", 1)
        sollic.append(Sub("V", "kV*qv*L/1000" if cont else "qv*L/2/1000", {"kV": ("k_V", kV), "qv": ("q_v", v["qv"]), "L": ("L", L)}, v["Vv"] / 1e3, "kN", 3))
        tau = Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V", v["Vv"] / 1e3), "Av": ("A_v,z", sec["Avz"] * 1e2)}, v["tau"], "N/mm²", 2)
        fleches.append(Sub("f", "kf*qv*L**4/(E*I)" if cont else "5*qv*L**4/(384*E*I)", {"kf": ("k_f", kf), "qv": ("q_v", v["qv"]), "L": ("L", L), "E": ("E", E), "I": ("I_y", sec["Iy"] * 1e4)}, v["fleche"], "mm", 2))
    else:
        sigma = Sub("σ", f"{pref}Mv*1e6/Wy + {pref}Mh*1e6/Wz", {**gq, "Mv": ("M_v", v["Mv"] / 1e6), "Wy": ("W_el,y", v["Wy"]), "Mh": ("M_h", v["Mh"] / 1e6), "Wz": ("W_el,z", v["Wz"])}, v["sigma"], "N/mm²", 1)
        fleches.append(Sub("f_h", ("kf*qh*L**4/(E*I)" if cont else "5*qh*L**4/(384*E*I)") + (" + Q*1000*L**3/(48*E*I)" if v["Q"] > 0 else ""),
                           {"kf": ("k_f", kf), "qh": ("q_h", v["qh"]), "L": ("L", L), "E": ("E", E), "I": ("I_z", sec["Iz"] * 1e4), "Q": ("Q", v["Q"] / 1e3)}, v["fh"], "mm", 2))
        fleches.append(Sub("f_v", "kf*qv*L**4/(E*I)" if cont else "5*qv*L**4/(384*E*I)", {"kf": ("k_f", kf), "qv": ("q_v", v["qv"]), "L": ("L", L), "E": ("E", E), "I": ("I_y", sec["Iy"] * 1e4)}, v["fv"], "mm", 2))
        fleches.append(Sub("f", "sqrt(fh**2 + fv**2)", {"fh": ("f_h", v["fh"]), "fv": ("f_v", v["fv"])}, v["fleche"], "mm", 2))
    lim = _limites(v, meth, res["acier"]["fy"])
    return _bloc(cas, v.get("cas_taux"), _groupes(charges, sollic, sigma, tau, fleches, lim))


# ------------------------------------------------------------------ 3 · barreaux ou lisses intermédiaires
def barre(res: Dict) -> Dict:
    rp = res.get("remplissage") or {}
    v = rp.get("verif")
    if not v:
        return _vide()
    sec, ch, meth = rp["section"], res["charges"], res["methode"]
    E = float(meth.get("E") or ME.E_ACIER)
    L = v["L"]
    gq, pref = _gq(v)
    charges, sollic, fleches = [], [], []
    faible = v.get("axe") == "faible"                 # lisse intermédiaire h vertical : axe faible (W_el,z, I_z)
    M = v["M2"] if faible else v["M1"]
    W_sym, W = ("W_el,z", v["Wz"]) if faible else ("W_el", v["Wy"])
    I_sym, I = ("I_z", sec["Iz"] * 1e4) if faible else ("I", sec["Iy"] * 1e4)
    if v["cas"] == "q":
        charges.append(Sub("q", "qr*e/1000", {"qr": ("q_r", ch["qr"]), "e": ("e", rp["entraxe"])}, v["q"], "kN/m", 3))
        sollic.append(Sub("M", "q*L**2/8/1e6", {"q": ("q", v["q"]), "L": ("L", L)}, M / 1e6, "kN·m", 4))
        fleches.append(Sub("f", "5*q*L**4/(384*E*I)", {"q": ("q", v["q"]), "L": ("L", L), "E": ("E", E), "I": (I_sym, I)}, v["fleche"], "mm", 2))
    else:
        sollic.append(Sub("M", "Q*L/4/1000", {"Q": ("Q_r", v["Q"] / 1e3), "L": ("L", L)}, M / 1e6, "kN·m", 4))
        fleches.append(Sub("f", "Q*1000*L**3/(48*E*I)", {"Q": ("Q_r", v["Q"] / 1e3), "L": ("L", L), "E": ("E", E), "I": (I_sym, I)}, v["fleche"], "mm", 2))
    sigma = Sub("σ", f"{pref}M*1e6/W", {**gq, "M": ("M", M / 1e6), "W": (W_sym, W)}, v["sigma"], "N/mm²", 1)
    V, Av_sym, Av = (v["V2"], "A_v,y", sec["Avy"] * 1e2) if faible else (v["V1"], "A_v", sec["Avz"] * 1e2)
    sollic.append(Sub("V", "", {}, V / 1e3, "kN", 3))
    tau = Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V", V / 1e3), "Av": (Av_sym, Av)}, v["tau"], "N/mm²", 2)
    lim = _limites(v, meth, res["acier"]["fy"])
    return _bloc(v["cas_charge"], v.get("cas_taux"), _groupes(charges, sollic, sigma, tau, fleches, lim))


# ------------------------------------------------------------------ console (système mur)
def console(res: Dict) -> Dict:
    c = res.get("console")
    if not c:
        return _vide()
    v, sec, ch, meth = c["verif"], c["section"], res["charges"], res["methode"]
    cas = v["cas_charge"]
    trib = v["trib1"] + v["trib2"]
    gq, pref = _gq(v)
    charges, sollic, fleches = [], [], []
    tau = None
    termes, vals = [], dict(gq)
    E = float(meth.get("E") or ME.E_ACIER)
    if cas in ("2", "1+2"):
        charges.append(Sub("R_v", "qv*s/1000", {"qv": ("q_v", ch["qv"]), "s": ("s_trib", trib)}, v["Rv"] / 1e3, "kN", 3))
        sollic.append(Sub("M", "Rv*e/1000", {"Rv": ("R_v", v["Rv"] / 1e3), "e": ("e", v["e"])}, v["M1"] / 1e6, "kN·m", 3))
        sollic.append(Sub("V", "", {}, v["V1"] / 1e3, "kN", 3))
        termes.append(f"{pref}M*1e6/W")
        vals.update({"M": ("M", v["M1"] / 1e6), "W": ("W_el", v["Wy"])})
        tau = Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V", v["V1"] / 1e3), "Av": ("A_v", sec["Avz"] * 1e2)}, v["tau"], "N/mm²", 2)
        fleches.append(Sub("f", "Rv*1000*e**3/(3*E*I)", {"Rv": ("R_v", v["Rv"] / 1e3), "e": ("e", v["e"]), "E": ("E", E), "I": ("I", sec["Iy"] * 1e4)}, v["fleche"], "mm", 3))
    if cas in ("1", "1+2"):
        if ch["mode"] == "P":
            charges.append(Sub("R_h", "P", {"P": ("P", ch["P"])}, v["Rh"] / 1e3, "kN", 3))
        else:
            charges.append(Sub("R_h", "qh*s/1000", {"qh": ("q_h", ch["qh"]), "s": ("s_trib", v["trib1"])}, v["Rh"] / 1e3, "kN", 3))
        termes.append(f"{pref}N*1000/A")
        vals.update({"N": ("R_h", v["Rh"] / 1e3), "A": ("A", v["A"])})
        if v["P2"] > 0:
            sollic.append(Sub("M₂", "R*e/1000", {"R": ("R_h,2", v["P2"] / 1e3), "e": ("e", v["e"])}, v["M2"] / 1e6, "kN·m", 3))
            termes.append(f"{pref}M2*1e6/Wz")
            vals.update({"M2": ("M₂", v["M2"] / 1e6), "Wz": ("W_el,z", v["Wz"])})
    sigma = Sub("σ", " + ".join(termes), vals, v["sigma"], "N/mm²", 1)
    lim = _limites(v, meth, res["acier"]["fy"])
    return _bloc(cas, v.get("cas_taux"), _groupes(charges, sollic, sigma, tau, fleches, lim))


# ------------------------------------------------------------------ 4 · ancrage : platine, chevilles ou boulons
ANCRAGES_EFFORTS, ANCRAGES_RESISTANCES, PLATINE_FLEXION = "Ancrages — efforts", "Ancrages — résistances et taux", "Platine en flexion"


def platine(res: Dict) -> Dict:
    fx = res.get("fixation") or {}
    p = fx.get("platine")
    if not p or p.get("taux_max") is None:
        return _vide()
    meth = res["methode"]
    if p.get("mode") == FX.LATERALE:
        return _platine_laterale(res, p, meth)
    efforts = [Sub("z", "ex/2 + hp/2", {"ex": ("e_x", p["ex"]), "hp": ("h_p", p["hp"])}, p["z"], "mm", 1)]
    if p["N"] >= 0:
        efforts.append(Sub("Z", "max(0, (M*1e6 - N*1000*hp/2)/z)/1000", {"M": ("M", p["M"] / 1e6), "N": ("N", p["N"] / 1e3), "hp": ("h_p", p["hp"]), "z": ("z", p["z"])}, p["Z"] / 1e3, "kN", 3))
    else:
        efforts.append(Sub("Z", "max((M*1e6 + Nt*1000*hp/2)/z, M*1e6/ex + Nt*1000/2)/1000",
                           {"M": ("M", p["M"] / 1e6), "Nt": ("N_t", -p["N"] / 1e3), "hp": ("h_p", p["hp"]), "z": ("z", p["z"]), "ex": ("e_x", p["ex"])}, p["Z"] / 1e3, "kN", 3))
    efforts.append(Sub("F_t", "Z/n", {"Z": ("Z", p["Z"] / 1e3), "n": ("n", p["n_tendus"])}, p["Ft"] / 1e3, "kN", 3))
    efforts.append(Sub("F_v", "V/n", {"V": ("V", p["V"] / 1e3), "n": ("2n", p["n_total"])}, p["Fv"] / 1e3, "kN", 3))
    gF = p["gamma_F"]
    resist = []
    if "As" in p:
        gM2 = float(meth.get("gamma_M2") or 1.25)
        if gF != 1.0:
            resist.append(Sub("F_t,Rd/γ_F", "0.9*fub*As/gM2/gF/1000", {"fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2), "gF": ("γ_F", gF)}, p["FtRd"] / 1e3, "kN", 2))
            resist.append(Sub("F_v,Rd/γ_F", "av*fub*As/gM2/gF/1000", {"av": ("α_v", p["av"]), "fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2), "gF": ("γ_F", gF)}, p["FvRd"] / 1e3, "kN", 2))
        else:
            resist.append(Sub("F_t,Rd", "0.9*fub*As/gM2/1000", {"fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2)}, p["FtRd"] / 1e3, "kN", 2))
            resist.append(Sub("F_v,Rd", "av*fub*As/gM2/1000", {"av": ("α_v", p["av"]), "fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2)}, p["FvRd"] / 1e3, "kN", 2))
        resist.append(Sub("η_inter", "Fv/FvRd + Ft/(1.4*FtRd)", {"Fv": ("F_v", p["Fv"] / 1e3), "FvRd": ("F_v,Rd", p["FvRd"] / 1e3), "Ft": ("F_t", p["Ft"] / 1e3), "FtRd": ("F_t,Rd", p["FtRd"] / 1e3)},
                          p["inter"], "", -1))
    else:
        if gF != 1.0:
            resist.append(Sub("N_Rd/γ_F", "", {}, p["FtRd"] / 1e3, "kN", 2))
            resist.append(Sub("V_Rd/γ_F", "", {}, p["FvRd"] / 1e3, "kN", 2))
        resist.append(Sub("η_inter", "(Ft/NRd)**1.5 + (Fv/VRd)**1.5", {"Ft": ("F_t", p["Ft"] / 1e3), "NRd": ("N_Rd", p["FtRd"] / 1e3), "Fv": ("F_v", p["Fv"] / 1e3), "VRd": ("V_Rd", p["FvRd"] / 1e3)},
                          p["inter"], "", -1))
    resist.append(Sub("η_t", "Ft/FtRd", {"Ft": ("F_t", p["Ft"] / 1e3), "FtRd": ("F_t,Rd", p["FtRd"] / 1e3)}, p["taux_t"], "", -1))
    flexion = []
    if p.get("m", 0) > 0:
        deduc = 0.8 * float((res["fixation"].get("soudure") or {}).get("cas", {}).get("params", {}).get("a", 0) or 0) * 2 ** 0.5
        flexion.append(Sub("m", "ex/2 - hp/2 - d", {"ex": ("e_x", p["ex"]), "hp": ("h_p", p["hp"]), "d": ("0,8·a·√2", deduc)}, p["m"], "mm", 1))
        flexion.append(Sub("b_eff", "min(lp, n*min(2*pi*m, 4*m + 1.25*e))", {"lp": ("l_p", p["lp"]), "n": ("n", p["n_tendus"]), "m": ("m", p["m"]), "e": ("e", p["e_bord"])}, p["b_eff"], "mm", 1))
        flexion.append(Sub("M_p", "Z*m/1000", {"Z": ("Z", p["Z"] / 1e3), "m": ("m", p["m"])}, p["Mp"] / 1e6, "kN·m", 4))
        flexion.append(Sub("σ_p", "6*Mp*1e6/(b*tp**2)", {"Mp": ("M_p", p["Mp"] / 1e6), "b": ("b_eff", p["b_eff"]), "tp": ("t_p", p["tp"])}, p["sigma_p"], "N/mm²", 1))
        flexion.append(Sub("t_p", "", {}, p["tp"], "mm", 0))
        flexion.append(Sub("t_req", "sqrt(6*Mp*1e6/(b*lim))", {"Mp": ("M_p", p["Mp"] / 1e6), "b": ("b_eff", p["b_eff"]), "lim": (_sym_lim(res), p["lim_sigma"])}, p["tp_req"], "mm", 1))
        flexion.append(Sub("η_p", "sp/lim", {"sp": ("σ_p", _arr(p["sigma_p"], 1)), "lim": (_sym_lim(res), p["lim_sigma"])}, p["taux_platine"], "", -1))
    return _bloc(p.get("cas_charge", ""), None, [(ANCRAGES_EFFORTS, efforts), (ANCRAGES_RESISTANCES, resist), (PLATINE_FLEXION, flexion)])


def _resistances_ancrage(p: Dict, meth: Dict) -> List[Sub]:
    """F_t,Rd et F_v,Rd (boulon) ou N_Rd et V_Rd (cheville), divisées par γ_F en ELS, l'interaction et le taux de traction."""
    gF = p["gamma_F"]
    out = []
    if "As" in p:
        gM2 = float(meth.get("gamma_M2") or 1.25)
        if gF != 1.0:
            out.append(Sub("F_t,Rd/γ_F", "0.9*fub*As/gM2/gF/1000", {"fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2), "gF": ("γ_F", gF)}, p["FtRd"] / 1e3, "kN", 2))
            out.append(Sub("F_v,Rd/γ_F", "av*fub*As/gM2/gF/1000", {"av": ("α_v", p["av"]), "fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2), "gF": ("γ_F", gF)}, p["FvRd"] / 1e3, "kN", 2))
        else:
            out.append(Sub("F_t,Rd", "0.9*fub*As/gM2/1000", {"fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2)}, p["FtRd"] / 1e3, "kN", 2))
            out.append(Sub("F_v,Rd", "av*fub*As/gM2/1000", {"av": ("α_v", p["av"]), "fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", gM2)}, p["FvRd"] / 1e3, "kN", 2))
        out.append(Sub("η_inter", "Fv/FvRd + Ft/(1.4*FtRd)", {"Fv": ("F_v", p["Fv"] / 1e3), "FvRd": ("F_v,Rd", p["FvRd"] / 1e3), "Ft": ("F_t", p["Ft"] / 1e3), "FtRd": ("F_t,Rd", p["FtRd"] / 1e3)},
                       p["inter"], "", -1))
    else:
        if gF != 1.0:
            out.append(Sub("N_Rd/γ_F", "", {}, p["FtRd"] / 1e3, "kN", 2))
            out.append(Sub("V_Rd/γ_F", "", {}, p["FvRd"] / 1e3, "kN", 2))
        out.append(Sub("η_inter", "(Ft/NRd)**1.5 + (Fv/VRd)**1.5", {"Ft": ("F_t", p["Ft"] / 1e3), "NRd": ("N_Rd", p["FtRd"] / 1e3), "Fv": ("F_v", p["Fv"] / 1e3), "VRd": ("V_Rd", p["FvRd"] / 1e3)},
                       p["inter"], "", -1))
    out.append(Sub("η_t", "Ft/FtRd", {"Ft": ("F_t", p["Ft"] / 1e3), "FtRd": ("F_t,Rd", p["FtRd"] / 1e3)}, p["taux_t"], "", -1))
    return out


def _flexion_platine(res: Dict, p: Dict, bras: Sub, largeur_sym: str, largeur: float) -> List[Sub]:
    """m, b_eff, M_p, σ_p, t_p, t_req, η_p : la platine en console du nu du montant à la rangée tendue."""
    if p.get("m", 0) <= 0:
        return []
    out = [bras]
    out.append(Sub("b_eff", "min(lp, n*min(2*pi*m, 4*m + 1.25*e))", {"lp": (largeur_sym, largeur), "n": ("n", p["n_tendus"]), "m": ("m", p["m"]), "e": ("e", p["e_bord"])}, p["b_eff"], "mm", 1))
    out.append(Sub("M_p", "Z*m/1000", {"Z": ("Z", p["Z"] / 1e3), "m": ("m", p["m"])}, p["Mp"] / 1e6, "kN·m", 4))
    out.append(Sub("σ_p", "6*Mp*1e6/(b*tp**2)", {"Mp": ("M_p", p["Mp"] / 1e6), "b": ("b_eff", p["b_eff"]), "tp": ("t_p", p["tp"])}, p["sigma_p"], "N/mm²", 1))
    out.append(Sub("t_p", "", {}, p["tp"], "mm", 0))
    out.append(Sub("t_req", "sqrt(6*Mp*1e6/(b*lim))", {"Mp": ("M_p", p["Mp"] / 1e6), "b": ("b_eff", p["b_eff"]), "lim": (_sym_lim(res), p["lim_sigma"])}, p["tp_req"], "mm", 1))
    out.append(Sub("η_p", "sp/lim", {"sp": ("σ_p", _arr(p["sigma_p"], 1)), "lim": (_sym_lim(res), p["lim_sigma"])}, p["taux_platine"], "", -1))
    return out


def _platine_laterale(res: Dict, p: Dict, meth: Dict) -> Dict:
    """La platine latérale : couple d'ancrages (z = e_x, Z = M/z + P/2), F_t, F_v (avec la part de
    torsion à l'angle), résistances et taux, la platine en console horizontale (m = e_y/2 − b/2 − 0,8·a·√2)."""
    efforts = [Sub("z", "ex", {"ex": ("e_x", p["ex"])}, p["z"], "mm", 1),
               Sub("Z", "M*1e6/z/1000 + P/2", {"M": ("M", p["M"] / 1e6), "z": ("z", p["z"]), "P": ("P", p["P"] / 1e3)}, p["Z"] / 1e3, "kN", 3),
               Sub("F_t", "Z/n", {"Z": ("Z", p["Z"] / 1e3), "n": ("n", p["n_tendus"])}, p["Ft"] / 1e3, "kN", 3)]
    if p.get("Fv_tors", 0.0) > 0:
        efforts.append(Sub("F_v", "V/n + Ft", {"V": ("V", p["V"] / 1e3), "n": ("2n", p["n_total"]), "Ft": ("F_v,tors", p["Fv_tors"] / 1e3)}, p["Fv"] / 1e3, "kN", 3))
    else:
        efforts.append(Sub("F_v", "V/n", {"V": ("V", p["V"] / 1e3), "n": ("2n", p["n_total"])}, p["Fv"] / 1e3, "kN", 3))
    deduc = 0.8 * float((res["fixation"].get("soudure") or {}).get("cas", {}).get("params", {}).get("elements", [{}])[0].get("a", 0) or 0) * 2 ** 0.5
    bras = Sub("m", "ey/2 - b/2 - d", {"ey": ("e_y", p["ey"]), "b": ("b", p["b_post"]), "d": ("0,8·a·√2", deduc)}, p["m"], "mm", 1)
    return _bloc(p.get("cas_charge", ""), None, [(ANCRAGES_EFFORTS, efforts), (ANCRAGES_RESISTANCES, _resistances_ancrage(p, meth)),
                                                 (PLATINE_FLEXION, _flexion_platine(res, p, bras, "b_p", p["bp"]))])


def soudure(res: Dict) -> Dict[str, List[Sub]]:
    s = (res.get("fixation") or {}).get("soudure")
    if not s:
        return {"groupe": [], "point": [], "critere": []}
    return FOS.toutes(s["res"])


def toutes(res: Dict) -> Dict:
    """Tous les blocs : montant (ou console), lisse, barre, platine, soudure."""
    return {"montant": montant(res), "console": console(res), "lisse": lisse(res), "barre": barre(res), "platine": platine(res),
            "soudure": soudure(res)}


def cle_methode(res: Dict) -> str:
    return "eurocode" if res["methode"].get("mode") == ME.EUROCODE else "els"

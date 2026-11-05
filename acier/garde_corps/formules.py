# -*- coding: utf-8 -*-
"""Les formules de la vérification d'un garde-corps, écrites comme à la main — note (page 1)
et écran (onglet Détail). Chaque ligne est une substitution « grandeur = formule = valeurs =
résultat » (``Sub`` du kit des assemblages) dont ``verifier(sub)`` ré-évalue le gabarit
imprimé pour retrouver le nombre du moteur : chaque formule de la note est TESTABLE
(tests/test_garde_corps_formules.py). Rien n'est recalculé : les nombres sont ceux de
``etat.verifier`` ; les formules ne font que les relier.

Chaque élément rend un BLOC : ``{"cas": libellé du cas de charge qui gouverne, "autres":
les autres cas et leur taux, "subs": [Sub, …]}``. Unités imprimées : kN, kN·m, mm, N/mm²,
cm³ → les gabarits portent les facteurs 10³ et 10⁶ (``*1000``, ``*1e6``) que le kit efface de la
forme littérale et imprime dans la forme numérique.
"""
from typing import Dict, List

from acier.assemblages.poutre_poutre.doubles_cornieres.formules import Sub, fmt, rendre, verifier  # noqa: F401 (ré-exportés)
from acier.soudures import formules as FOS

from . import charges as CH
from . import membres as ME

CRITERES_MATH = {
    "els": r"σ = \frac{M}{W_{el}} + \frac{N}{A} \le σ_{adm} \t{ ; } τ = \frac{V}{A_v} \le τ_{adm} \t{ ; } f \le f_{adm}",
    "eurocode": r"σ_{Ed} = \frac{γ_Q \cdot M}{W_{el}} + \frac{γ_Q \cdot N}{A} \le \frac{f_y}{γ_{M0}} \t{ ; } τ_{Ed} = \frac{γ_Q \cdot V}{A_v} \le \frac{f_y}{\sqrt{3} \cdot γ_{M0}} \t{ ; } f \le f_{adm}",
}
TITRES = {"els": "Méthode ELS du bureau (charges de service, contraintes admissibles)",
          "eurocode": "Eurocode (charges de calcul, EN 1993-1-1 §6.2, élastique)"}


def _bloc(cas: str, cas_taux: Dict, subs: List[Sub]) -> Dict:
    autres = [f"{CH.CAS_LIBELLES.get(k, k)} : η = {t * 100:.0f} %" for k, t in (cas_taux or {}).items() if k != cas]
    return {"cas": CH.CAS_LIBELLES.get(cas, cas), "autres": " ; ".join(autres), "subs": subs}


def _gq(v: Dict):
    g = v["gamma"]
    return ({"g": ("γ_Q", g)} if g != 1.0 else {}), ("g*" if g != 1.0 else "")


def _limites(v: Dict, meth: Dict, fy: float) -> List[Sub]:
    """η_σ, η_τ, η_f et η : les taux d'une vérification de section."""
    out = []
    if meth.get("mode") == ME.EUROCODE:
        out.append(Sub("σ_lim = fy/γM0", "fy/gM0", {"fy": ("f_y", fy), "gM0": ("γ_M0", float(meth.get("gamma_M0") or 1.0))}, v["lim_sigma"], "N/mm²", 1))
        out.append(Sub("τ_lim = fy/(√3·γM0)", "fy/(sqrt(3)*gM0)", {"fy": ("f_y", fy), "gM0": ("γ_M0", float(meth.get("gamma_M0") or 1.0))}, v["lim_tau"], "N/mm²", 1))
    out.append(Sub("η_σ", "s/lim", {"s": ("σ", v["sigma"]), "lim": ("σ_lim", v["lim_sigma"])}, v["taux"]["sigma"], "", -1))
    out.append(Sub("η_τ", "t/lim", {"t": ("τ", v["tau"]), "lim": ("τ_lim", v["lim_tau"])}, v["taux"]["tau"], "", -1))
    if meth.get("fleche_mode") == "mm":
        out.append(Sub("f_adm", "", {}, v["fleche_lim"], "mm", 1))
    else:
        out.append(Sub("f_adm", "L/x", {"L": ("L", v["L"]), "x": ("x", float(meth.get("fleche_val") or ME.FLECHE_X_DEFAUT))}, v["fleche_lim"], "mm", 1))
    out.append(Sub("η_f", "f/fa", {"f": ("f", v["fleche"]), "fa": ("f_adm", v["fleche_lim"])}, v["taux"]["fleche"], "", -1))
    out.append(Sub("η", "max(a, b, c)", {"a": ("η_σ", v["taux"]["sigma"]), "b": ("η_τ", v["taux"]["tau"]), "c": ("η_f", v["taux"]["fleche"])}, v["taux_max"], "", -1))
    return out


# ------------------------------------------------------------------ 1 · montant principal
def montant(res: Dict) -> Dict:
    m = res.get("montant")
    if not m:
        return {"cas": "", "autres": "", "subs": []}
    v, sec, ch, meth = m["verif"], m["section"], res["charges"], res["methode"]
    E = float(meth.get("E") or ME.E_ACIER)
    cas = v["cas_charge"]
    trib = v["trib1"] + v["trib2"]
    inv = v.get("orientation_inversee")
    out: List[Sub] = []
    gq, pref = _gq(v)
    if v["P1"] > 0 or v["P2"] > 0:
        if ch["mode"] == "P":
            out.append(Sub("P₁", "", {}, v["P1"] / 1e3, "kN", 2))
        else:
            out.append(Sub("P₁", "qh*s/1000", {"qh": ("q_h", ch["qh"]), "s": ("s_trib", v["trib2"] if inv else v["trib1"])}, v["P1"] / 1e3, "kN", 3))
            if v["P2"] > 0:
                out.append(Sub("P₂ (angle)", "qh*s/1000", {"qh": ("q_h", ch["qh"]), "s": ("s_trib,2", v["trib1"] if inv else v["trib2"])}, v["P2"] / 1e3, "kN", 3))
        out.append(Sub("M₁", "P*H/1000", {"P": ("P₁", v["P1"] / 1e3), "H": ("H", v["H"])}, v["M1"] / 1e6, "kN·m", 3))
        if v["P2"] > 0:
            out.append(Sub("M₂", "P*H/1000", {"P": ("P₂", v["P2"] / 1e3), "H": ("H", v["H"])}, v["M2"] / 1e6, "kN·m", 3))
        out.append(Sub("V", "", {}, v["V1"] / 1e3, "kN", 3))
    if v["N"] > 0:
        out.append(Sub("N", "qv*s/1000", {"qv": ("q_v", ch["qv"]), "s": ("s_trib", trib)}, v["N"] / 1e3, "kN", 3))
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
    if termes:
        out.append(Sub("σ", " + ".join(termes), vals, v["sigma"], "N/mm²", 1))
    if v["V1"] > 0:
        out.append(Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V", v["V1"] / 1e3), "Av": ("A_v", sec["Avz"] * 1e2)}, v["gamma"] * v["V1"] / (sec["Avz"] * 1e2), "N/mm²", 2))
        out.append(Sub("f", "P*1000*H**3/(3*E*I)", {"P": ("P₁", v["P1"] / 1e3), "H": ("H", v["H"]), "E": ("E", E), "I": ("I_y", sec["Iy"] * 1e4)}, v["f1"], "mm", 2))
        if v["P2"] > 0:
            out.append(Sub("f₂", "P*1000*H**3/(3*E*I)", {"P": ("P₂", v["P2"] / 1e3), "H": ("H", v["H"]), "E": ("E", E), "I": ("I_z", sec["Iz"] * 1e4)}, v["f2"], "mm", 2))
            out.append(Sub("f_rés", "sqrt(f1**2 + f2**2)", {"f1": ("f", v["f1"]), "f2": ("f₂", v["f2"])}, v["fleche"], "mm", 2))
    out += _limites(v, meth, res["acier"]["fy"])
    return _bloc(cas, v.get("cas_taux"), out)


# ------------------------------------------------------------------ 2 · main courante
def lisse(res: Dict) -> Dict:
    li = res.get("lisse")
    if not li:
        return {"cas": "", "autres": "", "subs": []}
    v, sec, ch, meth = li["verif"], li["section"], res["charges"], res["methode"]
    E = float(meth.get("E") or ME.E_ACIER)
    L, n, cont = v["L"], v.get("n_travees", 1), v.get("appuis") == "continue" and v.get("n_travees", 1) >= 2
    cas = v["cas_charge"]
    out: List[Sub] = []
    gq, pref = _gq(v)
    if ch["mode"] == "P":
        out.append(Sub("q_h = P/s", "P/s*1000", {"P": ("P", ch["P"]), "s": ("s", L)}, v["qh"], "kN/m", 3))
    kM, kV, kf = (v["kM"], v["kV"], v["kf"]) if cont else (1 / 8.0, 0.5, 5 / 384.0)
    if cas in ("1", "1+2"):
        if cont:
            out.append(Sub("M_h", "kM*qh*L**2/1e6" + (" + Q*L/4/1000" if v["Q"] > 0 else ""), {"kM": ("k_M", kM), "qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Mh"] / 1e6, "kN·m", 3))
        elif v["Q"] > 0 and v["comb"] == "max":
            out.append(Sub("M_h", "max(qh*L**2/8/1e6, Q*L/4/1000)", {"qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Mh"] / 1e6, "kN·m", 3))
        else:
            out.append(Sub("M_h", "qh*L**2/8/1e6" + (" + Q*L/4/1000" if v["Q"] > 0 else ""), {"qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Mh"] / 1e6, "kN·m", 3))
    if cas in ("2", "1+2"):
        if cont:
            out.append(Sub("M_v", "kM*qv*L**2/1e6", {"kM": ("k_M", kM), "qv": ("q_v", v["qv"]), "L": ("L", L)}, v["Mv"] / 1e6, "kN·m", 3))
        else:
            out.append(Sub("M_v", "qv*L**2/8/1e6", {"qv": ("q_v", v["qv"]), "L": ("L", L)}, v["Mv"] / 1e6, "kN·m", 3))
    if cas == "1":
        out.append(Sub("σ", f"{pref}Mh*1e6/Wz", {**gq, "Mh": ("M_h", v["Mh"] / 1e6), "Wz": ("W_el,z", v["Wz"])}, v["sigma"], "N/mm²", 1))
        out.append(Sub("V", ("kV*qh*L/1000" if cont else "qh*L/2/1000") + (" + Q/2" if v["Q"] > 0 else ""), {"kV": ("k_V", kV), "qh": ("q_h", v["qh"]), "L": ("L", L), "Q": ("Q", v["Q"] / 1e3)}, v["Vh"] / 1e3, "kN", 3))
        out.append(Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V", v["Vh"] / 1e3), "Av": ("A_v,y", sec["Avy"] * 1e2)}, v["tau"], "N/mm²", 2))
        out.append(Sub("f", ("kf*qh*L**4/(E*I)" if cont else "5*qh*L**4/(384*E*I)") + (" + Q*1000*L**3/(48*E*I)" if v["Q"] > 0 else ""),
                       {"kf": ("k_f", kf), "qh": ("q_h", v["qh"]), "L": ("L", L), "E": ("E", E), "I": ("I_z", sec["Iz"] * 1e4), "Q": ("Q", v["Q"] / 1e3)}, v["fleche"], "mm", 2))
    elif cas == "2":
        out.append(Sub("σ", f"{pref}Mv*1e6/Wy", {**gq, "Mv": ("M_v", v["Mv"] / 1e6), "Wy": ("W_el,y", v["Wy"])}, v["sigma"], "N/mm²", 1))
        out.append(Sub("V", "kV*qv*L/1000" if cont else "qv*L/2/1000", {"kV": ("k_V", kV), "qv": ("q_v", v["qv"]), "L": ("L", L)}, v["Vv"] / 1e3, "kN", 3))
        out.append(Sub("τ", f"{pref}V*1000/Av", {**gq, "V": ("V", v["Vv"] / 1e3), "Av": ("A_v,z", sec["Avz"] * 1e2)}, v["tau"], "N/mm²", 2))
        out.append(Sub("f", "kf*qv*L**4/(E*I)" if cont else "5*qv*L**4/(384*E*I)", {"kf": ("k_f", kf), "qv": ("q_v", v["qv"]), "L": ("L", L), "E": ("E", E), "I": ("I_y", sec["Iy"] * 1e4)}, v["fleche"], "mm", 2))
    else:
        out.append(Sub("σ", f"{pref}Mv*1e6/Wy + {pref}Mh*1e6/Wz", {**gq, "Mv": ("M_v", v["Mv"] / 1e6), "Wy": ("W_el,y", v["Wy"]), "Mh": ("M_h", v["Mh"] / 1e6), "Wz": ("W_el,z", v["Wz"])}, v["sigma"], "N/mm²", 1))
        out.append(Sub("f_h", ("kf*qh*L**4/(E*I)" if cont else "5*qh*L**4/(384*E*I)") + (" + Q*1000*L**3/(48*E*I)" if v["Q"] > 0 else ""),
                       {"kf": ("k_f", kf), "qh": ("q_h", v["qh"]), "L": ("L", L), "E": ("E", E), "I": ("I_z", sec["Iz"] * 1e4), "Q": ("Q", v["Q"] / 1e3)}, v["fh"], "mm", 2))
        out.append(Sub("f_v", "kf*qv*L**4/(E*I)" if cont else "5*qv*L**4/(384*E*I)", {"kf": ("k_f", kf), "qv": ("q_v", v["qv"]), "L": ("L", L), "E": ("E", E), "I": ("I_y", sec["Iy"] * 1e4)}, v["fv"], "mm", 2))
        out.append(Sub("f", "sqrt(fh**2 + fv**2)", {"fh": ("f_h", v["fh"]), "fv": ("f_v", v["fv"])}, v["fleche"], "mm", 2))
    out += _limites(v, meth, res["acier"]["fy"])
    return _bloc(cas, v.get("cas_taux"), out)


# ------------------------------------------------------------------ 3 · barreaux ou lisses intermédiaires
def barre(res: Dict) -> Dict:
    rp = res.get("remplissage") or {}
    v = rp.get("verif")
    if not v:
        return {"cas": "", "autres": "", "subs": []}
    sec, ch, meth = rp["section"], res["charges"], res["methode"]
    E = float(meth.get("E") or ME.E_ACIER)
    L = v["L"]
    out: List[Sub] = []
    gq, pref = _gq(v)
    if v["cas"] == "q":
        out.append(Sub("q", "qr*e/1000", {"qr": ("q_r", ch["qr"]), "e": ("e", rp["entraxe"])}, v["q"], "kN/m", 3))
        out.append(Sub("M", "q*L**2/8/1e6", {"q": ("q", v["q"]), "L": ("L", L)}, v["M1"] / 1e6, "kN·m", 4))
        out.append(Sub("σ", f"{pref}M*1e6/W", {**gq, "M": ("M", v["M1"] / 1e6), "W": ("W_el", v["Wy"])}, v["sigma"], "N/mm²", 1))
        out.append(Sub("f", "5*q*L**4/(384*E*I)", {"q": ("q", v["q"]), "L": ("L", L), "E": ("E", E), "I": ("I", sec["Iy"] * 1e4)}, v["fleche"], "mm", 2))
    else:
        out.append(Sub("M", "Q*L/4/1000", {"Q": ("Q_r", v["Q"] / 1e3), "L": ("L", L)}, v["M1"] / 1e6, "kN·m", 4))
        out.append(Sub("σ", f"{pref}M*1e6/W", {**gq, "M": ("M", v["M1"] / 1e6), "W": ("W_el", v["Wy"])}, v["sigma"], "N/mm²", 1))
        out.append(Sub("f", "Q*1000*L**3/(48*E*I)", {"Q": ("Q_r", v["Q"] / 1e3), "L": ("L", L), "E": ("E", E), "I": ("I", sec["Iy"] * 1e4)}, v["fleche"], "mm", 2))
    out += _limites(v, meth, res["acier"]["fy"])
    return _bloc(v["cas_charge"], v.get("cas_taux"), out)


# ------------------------------------------------------------------ console (système mur)
def console(res: Dict) -> Dict:
    c = res.get("console")
    if not c:
        return {"cas": "", "autres": "", "subs": []}
    v, sec, ch, meth = c["verif"], c["section"], res["charges"], res["methode"]
    cas = v["cas_charge"]
    trib = v["trib1"] + v["trib2"]
    out: List[Sub] = []
    gq, pref = _gq(v)
    termes, vals = [], dict(gq)
    if cas in ("2", "1+2"):
        out.append(Sub("R_v", "qv*s/1000", {"qv": ("q_v", ch["qv"]), "s": ("s_trib", trib)}, v["Rv"] / 1e3, "kN", 3))
        out.append(Sub("M", "Rv*e/1000", {"Rv": ("R_v", v["Rv"] / 1e3), "e": ("e", v["e"])}, v["M1"] / 1e6, "kN·m", 3))
        termes.append(f"{pref}M*1e6/W")
        vals.update({"M": ("M", v["M1"] / 1e6), "W": ("W_el", v["Wy"])})
    if cas in ("1", "1+2"):
        if ch["mode"] == "P":
            out.append(Sub("R_h = P", "", {}, v["Rh"] / 1e3, "kN", 3))
        else:
            out.append(Sub("R_h", "qh*s/1000", {"qh": ("q_h", ch["qh"]), "s": ("s_trib", v["trib1"])}, v["Rh"] / 1e3, "kN", 3))
        termes.append(f"{pref}N*1000/A")
        vals.update({"N": ("R_h", v["Rh"] / 1e3), "A": ("A", v["A"])})
        if v["P2"] > 0:
            out.append(Sub("M₂ (angle)", "R*e/1000", {"R": ("R_h,2", v["P2"] / 1e3), "e": ("e", v["e"])}, v["M2"] / 1e6, "kN·m", 3))
            termes.append(f"{pref}M2*1e6/Wz")
            vals.update({"M2": ("M₂", v["M2"] / 1e6), "Wz": ("W_el,z", v["Wz"])})
    out.append(Sub("σ", " + ".join(termes), vals, v["sigma"], "N/mm²", 1))
    out += _limites(v, meth, res["acier"]["fy"])
    return _bloc(cas, v.get("cas_taux"), out)


# ------------------------------------------------------------------ 4 · ancrage : platine, chevilles ou boulons
def platine(res: Dict) -> Dict:
    fx = res.get("fixation") or {}
    p = fx.get("platine")
    if not p or p.get("taux_max") is None:
        return {"cas": "", "autres": "", "subs": []}
    meth = res["methode"]
    out = [Sub("z", "ex/2 + hp/2", {"ex": ("e_x", p["ex"]), "hp": ("h_p", p["hp"])}, p["z"], "mm", 1)]
    if p["N"] >= 0:
        out.append(Sub("Z", "max(0, (M*1e6 - N*1000*hp/2)/z)/1000", {"M": ("M", p["M"] / 1e6), "N": ("N", p["N"] / 1e3), "hp": ("h_p", p["hp"]), "z": ("z", p["z"])}, p["Z"] / 1e3, "kN", 3))
    else:
        out.append(Sub("Z", "max((M*1e6 + Nt*1000*hp/2)/z, M*1e6/ex + Nt*1000/2)/1000",
                       {"M": ("M", p["M"] / 1e6), "Nt": ("N_t", -p["N"] / 1e3), "hp": ("h_p", p["hp"]), "z": ("z", p["z"]), "ex": ("e_x", p["ex"])}, p["Z"] / 1e3, "kN", 3))
    out.append(Sub("F_t", "Z/n", {"Z": ("Z", p["Z"] / 1e3), "n": ("n", p["n_tendus"])}, p["Ft"] / 1e3, "kN", 3))
    out.append(Sub("F_v", "V/n", {"V": ("V", p["V"] / 1e3), "n": ("2n", p["n_total"])}, p["Fv"] / 1e3, "kN", 3))
    gF = p["gamma_F"]
    if "As" in p:
        if gF != 1.0:
            out.append(Sub("F_t,Rd/γ_F", "0.9*fub*As/gM2/gF/1000", {"fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", float(meth.get("gamma_M2") or 1.25)), "gF": ("γ_F", gF)},
                           p["FtRd"] / 1e3, "kN", 2))
            out.append(Sub("F_v,Rd/γ_F", "av*fub*As/gM2/gF/1000", {"av": ("α_v", p["av"]), "fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", float(meth.get("gamma_M2") or 1.25)), "gF": ("γ_F", gF)},
                           p["FvRd"] / 1e3, "kN", 2))
        else:
            out.append(Sub("F_t,Rd", "0.9*fub*As/gM2/1000", {"fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", float(meth.get("gamma_M2") or 1.25))}, p["FtRd"] / 1e3, "kN", 2))
            out.append(Sub("F_v,Rd", "av*fub*As/gM2/1000", {"av": ("α_v", p["av"]), "fub": ("f_ub", p["fub"]), "As": ("A_s", p["As"]), "gM2": ("γ_M2", float(meth.get("gamma_M2") or 1.25))}, p["FvRd"] / 1e3, "kN", 2))
        out.append(Sub("η_inter", "Fv/FvRd + Ft/(1.4*FtRd)", {"Fv": ("F_v", p["Fv"] / 1e3), "FvRd": ("F_v,Rd", p["FvRd"] / 1e3), "Ft": ("F_t", p["Ft"] / 1e3), "FtRd": ("F_t,Rd", p["FtRd"] / 1e3)},
                       p["inter"], "", -1))
    else:
        if gF != 1.0:
            out.append(Sub("N_Rd/γ_F", "", {}, p["FtRd"] / 1e3, "kN", 2))
            out.append(Sub("V_Rd/γ_F", "", {}, p["FvRd"] / 1e3, "kN", 2))
        out.append(Sub("η_inter", "(Ft/NRd)**1.5 + (Fv/VRd)**1.5", {"Ft": ("F_t", p["Ft"] / 1e3), "NRd": ("N_Rd", p["FtRd"] / 1e3), "Fv": ("F_v", p["Fv"] / 1e3), "VRd": ("V_Rd", p["FvRd"] / 1e3)},
                       p["inter"], "", -1))
    out.append(Sub("η_t", "Ft/FtRd", {"Ft": ("F_t", p["Ft"] / 1e3), "FtRd": ("F_t,Rd", p["FtRd"] / 1e3)}, p["taux_t"], "", -1))
    if p.get("m", 0) > 0:
        deduc = 0.8 * float((res["fixation"].get("soudure") or {}).get("cas", {}).get("params", {}).get("a", 0) or 0) * 2 ** 0.5
        out.append(Sub("m", "ex/2 - hp/2 - d", {"ex": ("e_x", p["ex"]), "hp": ("h_p", p["hp"]), "d": ("0,8·a·√2", deduc)}, p["m"], "mm", 1))
        out.append(Sub("b_eff", "min(lp, n*min(2*pi*m, 4*m + 1.25*e))", {"lp": ("l_p", p["lp"]), "n": ("n", p["n_tendus"]), "m": ("m", p["m"]), "e": ("e", p["e_bord"])}, p["b_eff"], "mm", 1))
        out.append(Sub("M_p", "Z*m/1000", {"Z": ("Z", p["Z"] / 1e3), "m": ("m", p["m"])}, p["Mp"] / 1e6, "kN·m", 4))
        out.append(Sub("σ_p", "6*Mp*1e6/(b*tp**2)", {"Mp": ("M_p", p["Mp"] / 1e6), "b": ("b_eff", p["b_eff"]), "tp": ("t_p", p["tp"])}, p["sigma_p"], "N/mm²", 1))
        out.append(Sub("t_req", "sqrt(6*Mp*1e6/(b*lim))", {"Mp": ("M_p", p["Mp"] / 1e6), "b": ("b_eff", p["b_eff"]), "lim": ("σ_lim", p["lim_sigma"])}, p["tp_req"], "mm", 1))
        out.append(Sub("η_p", "sp/lim", {"sp": ("σ_p", p["sigma_p"]), "lim": ("σ_lim", p["lim_sigma"])}, p["taux_platine"], "", -1))
    return _bloc(p.get("cas_charge", ""), None, out)


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

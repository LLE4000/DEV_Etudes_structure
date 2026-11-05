# -*- coding: utf-8 -*-
"""Les formules de la vérification, écrites comme à la main, pour la méthode
CHOISIE — note (page 1) et écran (onglet Détail).

Chaque ligne est une substitution « grandeur = formule = valeurs = résultat »
(``Sub`` du kit des assemblages, ``acier.assemblages….formules``) : le gabarit
Python de la formule est rempli une fois avec les symboles, une fois avec les
nombres tels qu'ils sont imprimés, et ``verifier(sub)`` ré-évalue le gabarit
imprimé pour retrouver le résultat du moteur — ce qui rend chaque formule de
la note TESTABLE (la leçon de l'audit béton du 21/09/2026, reprise ici).

Rien n'est recalculé : tous les nombres sont ceux de ``etat.verifier`` (le
point gouvernant, les propriétés du groupe, les limites) ; les formules ne
font que les relier.
"""
from collections import Counter
from typing import Dict, List

from acier.assemblages.poutre_poutre.doubles_cornieres.formules import Sub, fmt, rendre, verifier  # noqa: F401 (ré-exportés)

from . import criteres as K

# le critère de chaque méthode, en typographie mathématique (mini-LaTeX de ndc_pdf/mathx)
CRITERES_MATH = {
    "directionnelle": r"\sqrt{σ_⊥^2 + 3(τ_⊥^2 + τ_∥^2)} \le \frac{f_u}{β_w \cdot γ_{M2}} \t{  et  } σ_⊥ \le \frac{0,9 \cdot f_u}{γ_{M2}}",
    "simplifiee": r"|f| \le F_{w,Rd} = f_{vw,d} \cdot a = \frac{f_u}{\sqrt{3} \cdot β_w \cdot γ_{M2}} \cdot a",
    "els": r"|f| \le F_{adm} = τ_{adm} \cdot a = \frac{k \cdot g}{10} \cdot a",
    "pleine": r"\sqrt{σ^2 + 3 τ^2} \le \frac{f_y}{γ_{M0}}",
    "pleine_els": r"\sqrt{σ^2 + 3 τ^2} \le \frac{f_y}{γ_F}",
}
TITRES = {"directionnelle": "Eurocode, méthode directionnelle (EN 1993-1-8 §4.5.3.2)",
          "simplifiee": "Eurocode, méthode simplifiée (EN 1993-1-8 §4.5.3.3)",
          "els": "Méthode ELS du bureau (efforts de service)",
          "pleine": "Pleine pénétration : la pièce gouverne (EN 1993-1-8 §4.7.1)",
          "pleine_els": "Pleine pénétration : la pièce gouverne (§4.7.1), efforts de service"}


def methode_du_point(res: Dict, g: Dict) -> str:
    """La clé de la méthode qui juge le point gouvernant : directionnelle | simplifiee | els | pleine | pleine_els."""
    m = res["methode"]
    if g.get("pleine"):
        return "pleine_els" if m.mode == K.ELS else "pleine"
    if m.mode == K.ELS:
        return "els"
    return "simplifiee" if m.ec == K.SIMPLIFIEE else "directionnelle"


def groupe(res: Dict) -> List[Sub]:
    """A_w = Σ a·l_eff (regroupé par gorge et longueur), I_x, I_y, J : les propriétés du groupe."""
    pr = res.get("proprietes") or {}
    cordons = res.get("cordons") or []
    if not pr or not cordons:
        return []
    groupes = Counter((round(c["a"], 3), round(c["l_eff"], 3)) for c in cordons)
    termes, vals = [], {}
    for i, ((a, l), n) in enumerate(sorted(groupes.items(), key=lambda kv: (-kv[0][0], -kv[0][1])), 1):
        vals[f"a{i}"] = ("a", a)
        vals[f"l{i}"] = ("l_eff", l)
        termes.append((f"{n}*" if n > 1 else "") + f"a{i}*l{i}")
    out = [Sub("A_w", " + ".join(termes), vals, pr["A_w"], "mm²", 0)]
    if pr.get("Ix", 0) > 0 or pr.get("Iy", 0) > 0:
        out.append(Sub("I_x", "", {}, pr["Ix"] / 1e4, "cm⁴", 1))
        out.append(Sub("I_y", "", {}, pr["Iy"] / 1e4, "cm⁴", 1))
        out.append(Sub("J", "Ix + Iy", {"Ix": ("I_x", pr["Ix"] / 1e4), "Iy": ("I_y", pr["Iy"] / 1e4)}, pr["J"] / 1e4, "cm⁴", 1))
    return out


def point(res: Dict, g: Dict) -> List[Sub]:
    """Au point gouvernant : la force par unité de longueur |f| et ses composantes sur la gorge."""
    fx, fy, fz = g["f_vec"]
    a = g["a"]
    out = [Sub("|f|", "sqrt(fx**2 + fy**2 + fz**2)", {"fx": ("f_x", fx), "fy": ("f_y", fy), "fz": ("f_z", fz)}, g["f"], "N/mm", 0)]
    out.append(Sub("σ⊥", "fn/a", {"fn": ("f·n_t", g["sigma_perp"] * a), "a": ("a", a)}, g["sigma_perp"], "N/mm²", 1))
    out.append(Sub("τ⊥", "ft/a", {"ft": ("f·t⊥", g["tau_perp"] * a), "a": ("a", a)}, g["tau_perp"], "N/mm²", 1))
    out.append(Sub("τ∥", "fa/a", {"fa": ("f·e_a", g["tau_par"] * a), "a": ("a", a)}, g["tau_par"], "N/mm²", 1))
    return out


def critere(res: Dict, g: Dict) -> List[Sub]:
    """Le critère de la méthode choisie, substitué au point gouvernant."""
    m = res["methode"]
    ac = res["acier"]
    cle = methode_du_point(res, g)
    sp, tp, tpar, a, f = g["sigma_perp"], g["tau_perp"], g["tau_par"], g["a"], g["f"]
    out: List[Sub] = []
    if cle in ("pleine", "pleine_els"):
        d = K.metal_base(sp, tp, tpar, ac["fy_min"], K.GAMMA_M0 if cle == "pleine" else m.gamma_F, service=(cle == "pleine_els"))
        out.append(Sub("σ_c", "sqrt(sp**2 + 3*(tp**2 + tpar**2))", {"sp": ("σ", sp), "tp": ("τ⊥", tp), "tpar": ("τ∥", tpar)}, d["sigma_c"], "N/mm²", 1))
        if cle == "pleine":
            out.append(Sub("lim", "fy/gM0", {"fy": ("f_y", ac["fy_min"]), "gM0": ("γ_M0", K.GAMMA_M0)}, d["lim_c"], "N/mm²", 1))
        else:
            out.append(Sub("lim", "fy/gF", {"fy": ("f_y", ac["fy_min"]), "gF": ("γ_F", m.gamma_F)}, d["lim_c"], "N/mm²", 1))
        out.append(Sub("η", "sc/lim", {"sc": ("σ_c", d["sigma_c"]), "lim": ("lim", d["lim_c"])}, d["taux"], "", -1))
        return out
    if cle == "directionnelle":
        d = K.directionnelle(sp, tp, tpar, ac["fu"], ac["bw"], m.gamma_M2)
        out.append(Sub("σ_c", "sqrt(sp**2 + 3*(tp**2 + tpar**2))", {"sp": ("σ⊥", sp), "tp": ("τ⊥", tp), "tpar": ("τ∥", tpar)}, d["sigma_c"], "N/mm²", 1))
        out.append(Sub("lim_c", "fu/(bw*gM2)", {"fu": ("f_u", ac["fu"]), "bw": ("β_w", ac["bw"]), "gM2": ("γ_M2", m.gamma_M2)}, d["lim_c"], "N/mm²", 1))
        out.append(Sub("η_c", "sc/lim", {"sc": ("σ_c", d["sigma_c"]), "lim": ("lim_c", d["lim_c"])}, d["taux_c"], "", -1))
        out.append(Sub("lim_⊥", "0.9*fu/gM2", {"fu": ("f_u", ac["fu"]), "gM2": ("γ_M2", m.gamma_M2)}, d["lim_perp"], "N/mm²", 1))
        out.append(Sub("η⊥", "abs(sp)/lim", {"sp": ("σ⊥", sp), "lim": ("lim_⊥", d["lim_perp"])}, d["taux_perp"], "", -1))
        out.append(Sub("η", "max(ec, ep)", {"ec": ("η_c", d["taux_c"]), "ep": ("η⊥", d["taux_perp"])}, d["taux"], "", -1))
        return out
    if cle == "simplifiee":
        s = K.simplifiee(f, a, ac["fu"], ac["bw"], m.gamma_M2)
        out.append(Sub("f_vw,d", "fu/(sqrt(3)*bw*gM2)", {"fu": ("f_u", ac["fu"]), "bw": ("β_w", ac["bw"]), "gM2": ("γ_M2", m.gamma_M2)}, s["f_vw_d"], "N/mm²", 1))
        out.append(Sub("F_w,Rd", "fv*a", {"fv": ("f_vw,d", s["f_vw_d"]), "a": ("a", a)}, s["F_w_Rd"], "N/mm", 0))
        out.append(Sub("η", "f/Fw", {"f": ("|f|", f), "Fw": ("F_w,Rd", s["F_w_Rd"])}, s["taux"], "", -1))
        return out
    e = K.els_bureau(f, a, m.k, m.g)
    out.append(Sub("τ_adm", "k*g/10", {"k": ("k", m.k), "g": ("g", m.g)}, e["tau_adm"], "N/mm²", 1))
    out.append(Sub("F_adm", "tau*a", {"tau": ("τ_adm", e["tau_adm"]), "a": ("a", a)}, e["F_adm"], "N/mm", 0))
    out.append(Sub("η", "f/Fadm", {"f": ("|f|", f), "Fadm": ("F_adm", e["F_adm"])}, e["taux"], "", -1))
    kq = res.get("k_equivalent") or {}
    if kq:
        out.append(Sub("k équivalent Eurocode", "fv/gF*10/g", {"fv": ("f_vw,d", kq["f_vw_d"]), "gF": ("γ_F", m.gamma_F), "g": ("g", m.g)},
                       kq["k_kg_mm_cm"], "kg/(mm·cm)", 0))
    return out


def toutes(res: Dict) -> Dict[str, List[Sub]]:
    """Les trois blocs (groupe, point gouvernant, critère) d'un résultat ; vides sur un refus."""
    g = res.get("gouvernant")
    if not g or "f_vec" not in g:
        return {"groupe": [], "point": [], "critere": []}
    return {"groupe": groupe(res), "point": point(res, g), "critere": critere(res, g)}


def lignes(res: Dict) -> List[str]:
    """Toutes les substitutions, rendues en texte (« σ_c = √(σ⊥² + 3(τ⊥² + τ∥²)) = √(106² + …) = 218,5 N/mm² »)."""
    t = toutes(res)
    return [rendre(s) for s in t["groupe"] + t["point"] + t["critere"]]

# -*- coding: utf-8 -*-
"""La fixation d'un montant (ou d'une console) sur une platine : la SOUDURE du montant sur
la platine par le moteur des soudures (``acier.soudures`` — tube du catalogue : C6 tube sur
platine ; plat : C3 console fléchie ; section libre : C8 cordons libres sur le contour), les
BOULONS ou CHEVILLES qui tiennent la platine, et la PLATINE elle-même, comme à la main :

- platine rigide, deux rangées de n boulons à ±e_x/2 de l'axe du montant dans le sens de
  l'effort ; le moment M est repris par la traction Z de la rangée tendue et l'appui au nu
  comprimé du montant : bras de levier z = e_x/2 + h_p/2, Z = (M − N·h_p/2)/z (N > 0 en
  compression soulage, ≥ 0), traction par boulon F_t = Z/n, cisaillement par boulon
  F_v = V/(2n) ;
- boulon (EN 1993-1-8 Tab. 3.4) : F_t,Rd = 0,9·f_ub·A_s/γ_M2, F_v,Rd = α_v·f_ub·A_s/γ_M2
  (filetage dans le plan de cisaillement), interaction F_v/F_v,Rd + F_t/(1,4·F_t,Rd) ≤ 1 ;
  cheville : N_Rd, V_Rd de l'ETA (valeurs de calcul saisies), interaction
  (F_t/N_Rd)^1,5 + (F_v/V_Rd)^1,5 ≤ 1 (EN 1992-4 §7.2.3, du côté sûr pour tous les modes) ;
  en méthode ELS (efforts de service), les résistances de calcul sont divisées par γ_F ;
- platine : bande en console du nu du montant à l'axe des boulons tendus, bras
  m = e_x/2 − h_p/2 − 0,8·a·√2 (EN 1993-1-8 §6.2.6.5, cordon d'angle), largeur efficace
  b_eff = min(l_p ; Σ min(2πm ; 4m + 1,25e)) (Tab. 6.6, motifs circulaire et non circulaire),
  M_p = Z·m, σ_p = 6·M_p/(b_eff·t_p²) ≤ σ_lim, d'où l'épaisseur requise.

Le béton sous la platine n'est pas vérifié (jamais dimensionnant pour un garde-corps ; le
module Pied de poteau le fait pour le cas général). Unités : mm, N, N·mm en interne.
"""
import copy
import math
from typing import Dict, List, Optional

from acier.bibliotheques import DB
from acier.resistances import Ft_Rd, Fv_Rd
from acier.soudures import configurations as CF
from acier.soudures import etat as SET

from . import bibliotheque as BI
from .membres import ELS, EUROCODE, limites, statut

TYPES = {"platine": "platine boulonnée (chevilles ou boulons)", "soudee": "montant soudé sur un plat existant", "scellee": "montant scellé (sans vérification)"}
ANCRAGES = {"cheville": "chevilles (valeurs de l'ETA)", "boulon": "boulons (EN 1993-1-8)"}
S2 = math.sqrt(2.0)


def fixation_defaut() -> Dict:
    return {"type": "platine", "platine": {"bp": 150.0, "lp": 150.0, "tp": 10.0},
            "soudure": {"jonction": "angle", "a": 4.0, "p": 3.0, "renfort": 0.0, "troncons": {"n": 0, "l": 60.0}},
            "ancrage": {"type": "cheville", "n_par_rangee": 2, "ex": 100.0, "ey": 100.0, "boulon": "M12", "classe": "8.8",
                        "NRd": 9.0, "VRd": 12.0, "nom": "cheville M12"},
            "console": {"e": 100.0, "section": {"nom": "Plat 50x10"}}}


def normaliser(fx: Dict) -> Dict:
    d = fixation_defaut()
    for k, v in (fx or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    if d["type"] not in TYPES:
        d["type"] = "platine"
    if d["ancrage"].get("type") not in ANCRAGES:
        d["ancrage"]["type"] = "cheville"
    d["ancrage"]["n_par_rangee"] = max(1, min(4, int(d["ancrage"].get("n_par_rangee") or 2)))
    return d


def boulons() -> List[str]:
    return [b["n"] for b in DB["boulons"]]


def classes() -> List[str]:
    return [c["n"] for c in DB["classes"]]


# ------------------------------------------------------------------ soudure : le cas du module Soudures
def cas_soudure(sec: Dict, N_kN: float, P1_kN: float, P2_kN: float, L_mm: float, soud: Dict, tp: float, meth: Dict, nuance: str) -> Dict:
    """Le cas ``acier.soudures`` de la soudure d'une section sur sa platine, sous un effort
    axial N (kN, > 0 traction), un effort P₁ (kN) dans le sens de h et P₂ dans le sens de b
    appliqués à la distance L (mm) de la platine (moments P·L), avec la jonction ``soud``
    (angle a / partielle p + renfort / pleine) et la méthode du garde-corps (ELS du bureau →
    méthode ELS, k ; Eurocode → directionnelle)."""
    M1, M2 = P1_kN * L_mm / 1e3, P2_kN * L_mm / 1e3       # kN·m
    jon = soud.get("jonction", "angle")
    base = {"a": float(soud.get("a", 4.0)), "jonction": jon, "p": float(soud.get("p", 3.0)), "renfort": float(soud.get("renfort", 0.0)),
            "troncons": dict(soud.get("troncons") or CF.CONTINU), "tp": float(tp), "nuance_A": nuance, "nuance_B": nuance}
    if sec.get("standard") and sec["famille"] in BI.FAMILLES_CREUX:
        cas = SET.nouveau_cas("C6")
        cas["params"].update(base)
        cas["params"]["profil"] = sec["nom"]
        cas["efforts"] = {"N": N_kN, "Vz": P1_kN, "Vy": P2_kN, "My": -M1, "Mz": M2, "T": 0.0}
    elif sec["forme"] == "plein_rect":
        cas = SET.nouveau_cas("C3")
        cas["params"].update(base)
        cas["params"].update({"h": float(sec["h"]), "t": float(sec["b"]), "e": float(L_mm), "deux_cordons": True, "retours": True, "poteau": "plat"})
        # C3 : V vertical à l'excentricité e (sens de h du plat), N > 0 tire la console hors de l'appui ;
        # l'effort P₂ (sens de l'épaisseur) ne s'y modélise pas : il est porté par le couple des cordons (négligé)
        cas["efforts"] = {"V": abs(P1_kN), "N": N_kN}
    else:
        cas = SET.nouveau_cas("C8")
        cas["params"].update({k: v for k, v in base.items() if k not in ("a", "troncons", "jonction", "p", "renfort")})
        a = base["a"] if jon == "angle" else (base["p"] + base["renfort"] if jon == "partielle" else float(sec.get("t") or sec.get("b") or base["a"]))
        tr = base["troncons"]
        if sec["forme"] in ("creux_rond", "plein_rond"):
            els = [{"type": "arc", "nom": "pourtour", "xc": 0.0, "yc": 0.0, "r": float(sec["d"]) / 2, "ang1": 0.0, "ang2": 360.0, "a": a, "cote": 1, "troncons": tr}]
        else:
            h, b = float(sec["h"]), float(sec["b"])
            coins = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, h / 2), (-b / 2, h / 2)]
            els = []
            for i in range(4):
                (x1, y1), (x2, y2) = coins[i], coins[(i + 1) % 4]
                els.append({"type": "segment", "nom": ("bas", "droite", "haut", "gauche")[i], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "a": a, "cote": 1,
                            "retours": True, "troncons": tr})
        cas["params"]["elements"] = els
        cas["params"].update({"xR": 0.0, "yR": 0.0, "zR": 0.0})
        cas["efforts"] = {"N": N_kN, "Vx": P2_kN, "Vy": P1_kN, "Mx": -M1, "My": M2, "Mz": 0.0}
    if meth.get("mode") == EUROCODE:
        cas["methode"].update({"mode": "eurocode", "ec": meth.get("ec", "directionnelle"), "gamma_M2": float(meth.get("gamma_M2", 1.25))})
    else:
        cas["methode"].update({"mode": "els", "k": float(meth.get("k", 100.0)), "g": float(meth.get("g", 9.81)), "gamma_F": float(meth.get("gamma_F", 1.5))})
    cas["nom"] = "soudure du montant sur la platine"
    return cas


def verifier_soudure(cas: Dict) -> Dict:
    """Le résultat du moteur des soudures, et son résumé (taux, statut, cordon gouvernant)."""
    res = SET.verifier(cas)
    g = res.get("gouvernant") or {}
    return {"res": res, "taux": res.get("taux_max"), "statut": res["statut"], "critere": g.get("critere", ""),
            "position": g.get("position", ""), "messages": list(res.get("messages", [])), "cas": cas}


# ------------------------------------------------------------------ boulons, chevilles, platine
def _boulon(nom: str, classe: str) -> Dict:
    b = next((x for x in DB["boulons"] if x["n"] == nom), DB["boulons"][0])
    c = next((x for x in DB["classes"] if x["n"] == classe), DB["classes"][0])
    return {**b, "classe": c}


def platine_boulonnee(M: float, N: float, V: float, hp: float, pl: Dict, anc: Dict, a_cordon: float, meth: Dict, fy: float,
                      jonction: str = "angle") -> Dict:
    """M (N·mm) à la platine dans le sens de b_p, N (N, > 0 compression), V (N) ; h_p (mm)
    la dimension du montant dans le sens de l'effort ; ``pl`` {bp, lp, tp}, ``anc`` {type,
    n_par_rangee, ex, ey, boulon, classe, NRd, VRd}. Rend efforts, résistances, taux."""
    lim = limites(meth, fy)
    g = lim["gamma"]
    gF = float(meth.get("gamma_F") or 1.5) if meth.get("mode") != EUROCODE else 1.0
    n = int(anc.get("n_par_rangee") or 2)
    ex, ey = float(anc.get("ex") or 0.0), float(anc.get("ey") or 0.0)
    bp, lp, tp = float(pl.get("bp") or 0.0), float(pl.get("lp") or 0.0), float(pl.get("tp") or 0.0)
    out: Dict = {"n_total": 2 * n, "n_tendus": n, "ex": ex, "ey": ey, "bp": bp, "lp": lp, "tp": tp, "hp": hp, "messages": [], "gamma": g, "gamma_F": gF}
    if ex <= hp:
        out["messages"].append(f"Refus : l'entraxe e_x = {ex:g} mm des rangées doit dépasser la dimension h_p = {hp:g} mm du montant")
        out["taux_max"], out["statut"] = None, "refus"
        return out
    if n > 1 and (n - 1) * ey >= lp:
        out["messages"].append(f"Refus : {n} boulons à e_y = {ey:g} mm ne tiennent pas dans la largeur l_p = {lp:g} mm")
        out["taux_max"], out["statut"] = None, "refus"
        return out
    Md, Nd, Vd = g * abs(M), g * N, g * abs(V)
    z = ex / 2 + hp / 2
    Z = max(0.0, (Md - Nd * hp / 2) / z)
    Ft, Fv = Z / n, Vd / (2 * n)
    out.update({"z": z, "Z": Z, "Ft": Ft, "Fv": Fv, "M": Md, "N": Nd, "V": Vd})
    if anc.get("type") == "boulon":
        b = _boulon(anc.get("boulon", "M12"), anc.get("classe", "8.8"))
        FtRd = Ft_Rd(b["classe"]["fub"], b["As"], float(meth.get("gamma_M2") or 1.25)) * 1e3 / gF
        FvRd = Fv_Rd(b["classe"]["av"], b["classe"]["fub"], b["As"], float(meth.get("gamma_M2") or 1.25)) * 1e3 / gF
        inter = Fv / FvRd + Ft / (1.4 * FtRd) if FtRd > 0 and FvRd > 0 else float("inf")
        out.update({"nom_ancrage": f"{b['n']} classe {b['classe']['n']}", "FtRd": FtRd, "FvRd": FvRd, "As": b["As"], "fub": b["classe"]["fub"],
                    "av": b["classe"]["av"], "formule_inter": "F_v/F_v,Rd + F_t/(1,4·F_t,Rd) ≤ 1", "d": b["d"], "d0": b["d0"]})
    else:
        FtRd, FvRd = float(anc.get("NRd") or 0.0) * 1e3 / gF, float(anc.get("VRd") or 0.0) * 1e3 / gF
        bN = Ft / FtRd if FtRd > 0 else float("inf")
        bV = Fv / FvRd if FvRd > 0 else float("inf")
        inter = bN ** 1.5 + bV ** 1.5 if math.isfinite(bN) and math.isfinite(bV) else float("inf")
        out.update({"nom_ancrage": anc.get("nom") or "cheville", "FtRd": FtRd, "FvRd": FvRd, "formule_inter": "(F_t/N_Rd)^1,5 + (F_v/V_Rd)^1,5 ≤ 1",
                    "d": 12.0, "d0": 14.0})
    taux_t = Ft / FtRd if FtRd > 0 else float("inf")
    taux_v = Fv / FvRd if FvRd > 0 else float("inf")
    out.update({"taux_t": taux_t, "taux_v": taux_v, "inter": inter, "taux_boulons": max(taux_t, taux_v, inter)})
    # la platine : bande en console du nu du montant à la rangée tendue
    deduc = 0.8 * a_cordon * S2 if jonction == "angle" else 0.0
    m = ex / 2 - hp / 2 - deduc
    e_bord = (lp - (n - 1) * ey) / 2
    if m <= 0:
        out["messages"].append("Boulons au nu du montant : pas de flexion de la platine (bras m ≤ 0)")
        out.update({"m": 0.0, "b_eff": lp, "Mp": 0.0, "sigma_p": 0.0, "tp_req": 0.0, "taux_platine": 0.0, "e_bord": e_bord})
    else:
        leff = n * min(2 * math.pi * m, 4 * m + 1.25 * e_bord)
        b_eff = min(lp, leff)
        Mp = Z * m
        sigma_p = 6 * Mp / (b_eff * tp ** 2) if b_eff > 0 and tp > 0 else float("inf")
        tp_req = math.sqrt(6 * Mp / (b_eff * lim["sigma"])) if b_eff > 0 else float("inf")
        out.update({"m": m, "b_eff": b_eff, "leff": leff, "Mp": Mp, "sigma_p": sigma_p, "tp_req": tp_req, "e_bord": e_bord,
                    "taux_platine": sigma_p / lim["sigma"]})
    out["lim_sigma"] = lim["sigma"]
    out["taux_max"] = max(out["taux_boulons"], out["taux_platine"])
    out["statut"] = statut(out["taux_max"])
    out["critere"] = max((("boulons · traction", taux_t), ("boulons · cisaillement", taux_v), ("boulons · interaction", inter),
                          ("platine · flexion", out["taux_platine"])), key=lambda kv: kv[1])[0]
    return out


def texte_ancrage(anc: Dict) -> str:
    n = int(anc.get("n_par_rangee") or 2)
    if anc.get("type") == "boulon":
        return f"2 × {n} boulons {anc.get('boulon', 'M12')} {anc.get('classe', '8.8')}"
    return f"2 × {n} {anc.get('nom') or 'chevilles'} (N_Rd {float(anc.get('NRd') or 0):g} kN, V_Rd {float(anc.get('VRd') or 0):g} kN)"


def texte_soudure(soud: Dict) -> str:
    jon = soud.get("jonction", "angle")
    tr = soud.get("troncons") or {}
    disc = f", par {int(tr.get('n'))} tronçons de {float(tr.get('l')):g} mm" if int(tr.get("n") or 0) >= 1 else ""
    if jon == "pleine":
        return "pleine pénétration (la soudure vaut la pièce)"
    if jon == "partielle":
        r = float(soud.get("renfort") or 0)
        return f"pénétration partielle p = {float(soud.get('p', 0)):g} mm" + (f" + renfort a_r = {r:g}" if r > 0 else "") + disc
    return f"cordon d'angle a = {float(soud.get('a', 0)):g} mm" + disc

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

PLATINE LATÉRALE (en applique sur le nez de dalle, le cas des balcons) : la platine est
verticale, le montant est soudé LE LONG de sa face par deux cordons longitudinaux (C8 du
module Soudures), deux rangées horizontales de n chevilles à e_x l'une de l'autre (la haute
tendue, la basse en appui), les colonnes à e_y de part et d'autre du montant. Au centre de la
platine : M = P·(H + d_h + b_p/2) (d_h la descente du dessus de dalle au haut de la platine),
l'arrachement P (normal à la platine) et le cisaillement V (vertical, et horizontal à l'angle).
Couple d'ancrages : z = e_x, Z = M/z + P/2, F_t = Z/n ; F_v = V/(2n) ; la platine fléchit en
console HORIZONTALE du nu du montant à la colonne tendue : m = e_y/2 − b/2 − 0,8·a·√2,
b_eff = min(b_p ; n·min(2πm ; 4m + 1,25e)), e la distance de la rangée haute au bord.
Le montant lui-même est vérifié sur H + d_h (console jusqu'au haut de la platine).

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

TYPES = {"platine": "platine boulonnée (chevilles ou boulons)", "laterale": "platine latérale en nez de dalle (chevilles ou boulons)",
         "soudee": "montant soudé sur un plat existant", "scellee": "montant scellé (sans vérification)"}
LATERALE = "laterale"
ANCRAGES = {"cheville": "chevilles (valeurs de l'ETA)", "boulon": "boulons (EN 1993-1-8)"}
S2 = math.sqrt(2.0)


def fixation_defaut() -> Dict:
    return {"type": "platine", "platine": {"bp": 150.0, "lp": 150.0, "tp": 10.0, "d_h": 50.0},
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


def cas_soudure_laterale(sec: Dict, P_kN: float, P2_kN: float, Nv_kN: float, y_P: float, soud: Dict, lw: float, tp: float, meth: Dict, nuance: str) -> Dict:
    """Le cas C8 du montant soudé LE LONG d'une platine latérale : deux cordons longitudinaux de
    longueur l_w (la hauteur de la platine) aux deux bords du contact, à ±b/2 (b la largeur du
    montant le long de la platine) ; les efforts au point d'application, à y_P (mm) au-dessus du
    centre des cordons : l'arrachement P (normal au plan de joint, > 0), le cisaillement horizontal
    P₂ (dans le plan, à l'angle) et le cisaillement vertical N_v — le moteur transporte le torseur au
    centre des gorges (M_x = P·y_P, torsion M_z = −P₂·y_P)."""
    jon = soud.get("jonction", "angle")
    a = float(soud.get("a", 4.0)) if jon == "angle" else (float(soud.get("p", 3.0)) + float(soud.get("renfort", 0.0)) if jon == "partielle"
                                                          else float(sec.get("t") or sec.get("b") or soud.get("a", 4.0)))
    b = largeur_contact(sec)
    tr = dict(soud.get("troncons") or CF.CONTINU)
    els = [{"type": "segment", "nom": "cordon gauche", "x1": -b / 2, "y1": lw / 2, "x2": -b / 2, "y2": -lw / 2, "a": a, "cote": 1, "retours": False, "troncons": tr},
           {"type": "segment", "nom": "cordon droit", "x1": b / 2, "y1": -lw / 2, "x2": b / 2, "y2": lw / 2, "a": a, "cote": 1, "retours": False, "troncons": tr}]
    cas = SET.nouveau_cas("C8")
    cas["params"].update({"elements": els, "xR": 0.0, "yR": float(y_P), "zR": 0.0, "tp": float(tp), "nuance_A": nuance, "nuance_B": nuance})
    cas["efforts"] = {"N": P_kN, "Vx": P2_kN, "Vy": -Nv_kN, "Mx": 0.0, "My": 0.0, "Mz": 0.0}
    if meth.get("mode") == EUROCODE:
        cas["methode"].update({"mode": "eurocode", "ec": meth.get("ec", "directionnelle"), "gamma_M2": float(meth.get("gamma_M2", 1.25))})
    else:
        cas["methode"].update({"mode": "els", "k": float(meth.get("k", 100.0)), "g": float(meth.get("g", 9.81)), "gamma_F": float(meth.get("gamma_F", 1.5))})
    cas["nom"] = "soudure du montant le long de la platine latérale"
    return cas


def largeur_contact(sec: Dict) -> float:
    """La largeur du montant LE LONG d'une platine latérale (sa dimension b, perpendiculaire à
    la charge ; le diamètre d'un rond)."""
    return float(sec.get("b") or sec.get("d") or 0.0)


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


def _resistances(anc: Dict, meth: Dict, gF: float, Ft: float, Fv: float) -> Dict:
    """Les résistances d'un ancrage (boulon EN 1993-1-8 Tab. 3.4 ou cheville de l'ETA), divisées par
    γ_F en ELS, et l'interaction ; les taux de traction et de cisaillement."""
    out: Dict = {}
    if anc.get("type") == "boulon":
        b = _boulon(anc.get("boulon", "M12"), anc.get("classe", "8.8"))
        FtRd = Ft_Rd(b["classe"]["fub"], b["As"], float(meth.get("gamma_M2") or 1.25)) * 1e3 / gF
        FvRd = Fv_Rd(b["classe"]["av"], b["classe"]["fub"], b["As"], float(meth.get("gamma_M2") or 1.25)) * 1e3 / gF
        inter = Fv / FvRd + Ft / (1.4 * FtRd) if FtRd > 0 and FvRd > 0 else float("inf")
        out.update({"nom_ancrage": f"boulons {b['n']} classe {b['classe']['n']}", "FtRd": FtRd, "FvRd": FvRd, "As": b["As"], "fub": b["classe"]["fub"],
                    "av": b["classe"]["av"], "formule_inter": "F_v/F_v,Rd + F_t/(1,4·F_t,Rd) ≤ 1", "d": b["d"], "d0": b["d0"]})
    else:
        FtRd, FvRd = float(anc.get("NRd") or 0.0) * 1e3 / gF, float(anc.get("VRd") or 0.0) * 1e3 / gF
        bN = Ft / FtRd if FtRd > 0 else float("inf")
        bV = Fv / FvRd if FvRd > 0 else float("inf")
        inter = bN ** 1.5 + bV ** 1.5 if math.isfinite(bN) and math.isfinite(bV) else float("inf")
        out.update({"nom_ancrage": pluriel_ancrage(anc.get("nom")), "FtRd": FtRd, "FvRd": FvRd, "formule_inter": "(F_t/N_Rd)^1,5 + (F_v/V_Rd)^1,5 ≤ 1",
                    "d": 12.0, "d0": 14.0})
    taux_t = Ft / FtRd if FtRd > 0 else float("inf")
    taux_v = Fv / FvRd if FvRd > 0 else float("inf")
    out.update({"taux_t": taux_t, "taux_v": taux_v, "inter": inter, "taux_boulons": max(taux_t, taux_v, inter)})
    return out


def _flexion_platine(out: Dict, Z: float, m: float, n: int, largeur: float, e_bord: float, tp: float, lim: Dict) -> None:
    """La platine en console du nu du montant à la rangée tendue : b_eff = min(largeur ; n·min(2πm ;
    4m + 1,25e)), M_p = Z·m, σ_p = 6·M_p/(b_eff·t_p²), épaisseur requise, taux."""
    if m <= 0:
        out["messages"].append("Boulons au nu du montant : pas de flexion de la platine (bras m ≤ 0)")
        out.update({"m": 0.0, "b_eff": largeur, "Mp": 0.0, "sigma_p": 0.0, "tp_req": 0.0, "taux_platine": 0.0, "e_bord": e_bord})
        return
    leff = n * min(2 * math.pi * m, 4 * m + 1.25 * e_bord)
    b_eff = min(largeur, leff)
    Mp = Z * m
    sigma_p = 6 * Mp / (b_eff * tp ** 2) if b_eff > 0 and tp > 0 else float("inf")
    tp_req = math.sqrt(6 * Mp / (b_eff * lim["sigma"])) if b_eff > 0 else float("inf")
    out.update({"m": m, "b_eff": b_eff, "leff": leff, "Mp": Mp, "sigma_p": sigma_p, "tp_req": tp_req, "e_bord": e_bord, "taux_platine": sigma_p / lim["sigma"]})


def _conclure(out: Dict, lim: Dict) -> Dict:
    out["lim_sigma"] = lim["sigma"]
    out["taux_max"] = max(out["taux_boulons"], out["taux_platine"])
    out["statut"] = statut(out["taux_max"])
    out["critere"] = max((("boulons · traction", out["taux_t"]), ("boulons · cisaillement", out["taux_v"]), ("boulons · interaction", out["inter"]),
                          ("platine · flexion", out["taux_platine"])), key=lambda kv: kv[1])[0]
    return out


def _refus(out: Dict, msg: str) -> Dict:
    out["messages"].append("Refus : " + msg)
    out["taux_max"], out["statut"] = None, "refus"
    return out


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
    out: Dict = {"mode": "platine", "n_total": 2 * n, "n_tendus": n, "ex": ex, "ey": ey, "bp": bp, "lp": lp, "tp": tp, "hp": hp, "messages": [], "gamma": g, "gamma_F": gF}
    if ex <= hp:
        return _refus(out, f"l'entraxe e_x = {ex:g} mm des rangées doit dépasser la dimension h_p = {hp:g} mm du montant")
    if n > 1 and (n - 1) * ey >= lp:
        return _refus(out, f"{n} boulons à e_y = {ey:g} mm ne tiennent pas dans la largeur l_p = {lp:g} mm")
    Md, Nd, Vd = g * abs(M), g * N, g * abs(V)
    z = ex / 2 + hp / 2
    Z = max(0.0, (Md - Nd * hp / 2) / z)
    Ft, Fv = Z / n, Vd / (2 * n)
    out.update({"z": z, "Z": Z, "Ft": Ft, "Fv": Fv, "M": Md, "N": Nd, "V": Vd})
    out.update(_resistances(anc, meth, gF, Ft, Fv))
    # la platine : bande en console du nu du montant à la rangée tendue
    deduc = 0.8 * a_cordon * S2 if jonction == "angle" else 0.0
    _flexion_platine(out, Z, ex / 2 - hp / 2 - deduc, n, lp, (lp - (n - 1) * ey) / 2, tp, lim)
    return _conclure(out, lim)


def platine_laterale(M: float, P: float, V: float, Mz: float, b_post: float, pl: Dict, anc: Dict, a_cordon: float, meth: Dict, fy: float,
                     jonction: str = "angle") -> Dict:
    """Platine LATÉRALE : M (N·mm) au centre de la platine (axe horizontal le long du garde-corps),
    P (N, > 0) l'arrachement normal à la platine, V (N) le cisaillement dans son plan (vertical,
    et horizontal à l'angle), M_z (N·mm) la torsion autour de la normale (montant d'angle) ;
    b_post (mm) la largeur du montant le long de la platine ; ``pl`` {bp (hauteur), lp (largeur),
    tp}, ``anc`` {type, n_par_rangee (les colonnes, par paires de part et d'autre du montant), ex
    (entraxe vertical des deux rangées), ey (entraxe horizontal), …}. Couple d'ancrages (la
    rangée haute tendue, la basse en appui) : z = e_x, Z = M/z + P/2."""
    lim = limites(meth, fy)
    g = lim["gamma"]
    gF = float(meth.get("gamma_F") or 1.5) if meth.get("mode") != EUROCODE else 1.0
    n = int(anc.get("n_par_rangee") or 2)
    ex, ey = float(anc.get("ex") or 0.0), float(anc.get("ey") or 0.0)
    bp, lp, tp = float(pl.get("bp") or 0.0), float(pl.get("lp") or 0.0), float(pl.get("tp") or 0.0)
    d_h = float(pl.get("d_h") or 0.0)
    out: Dict = {"mode": LATERALE, "n_total": 2 * n, "n_tendus": n, "ex": ex, "ey": ey, "bp": bp, "lp": lp, "tp": tp, "hp": b_post, "b_post": b_post, "d_h": d_h,
                 "messages": [], "gamma": g, "gamma_F": gF}
    d0 = 14.0 if anc.get("type") != "boulon" else _boulon(anc.get("boulon", "M12"), anc.get("classe", "8.8"))["d0"]
    if n % 2:
        return _refus(out, f"platine latérale : les chevilles vont par paires de part et d'autre du montant ({n} par rangée saisi)")
    if ey <= b_post + d0:
        return _refus(out, f"platine latérale : l'entraxe e_y = {ey:g} mm des colonnes doit dépasser la largeur b = {b_post:g} mm du montant plus un trou")
    if ex >= bp:
        return _refus(out, f"platine latérale : l'entraxe e_x = {ex:g} mm des rangées ne tient pas dans la hauteur b_p = {bp:g} mm")
    if (n - 1) * ey >= lp:
        return _refus(out, f"platine latérale : {n} colonnes à e_y = {ey:g} mm ne tiennent pas dans la largeur l_p = {lp:g} mm")
    Md, Pd, Vd, Mzd = g * abs(M), g * max(P, 0.0), g * abs(V), g * abs(Mz)
    z = ex
    Z = Md / z + Pd / 2
    xs = [(k - (n - 1) / 2) * ey for k in range(n)]
    r2 = sum(x * x + (ex / 2) ** 2 for x in xs) * 2
    r_max = math.hypot(max(abs(x) for x in xs), ex / 2)
    Fv_tors = Mzd * r_max / r2 if r2 > 0 else 0.0
    Ft, Fv = Z / n, Vd / (2 * n) + Fv_tors
    out.update({"z": z, "Z": Z, "Ft": Ft, "Fv": Fv, "Fv_tors": Fv_tors, "M": Md, "N": -Pd, "P": Pd, "V": Vd, "Mz": Mzd})
    out.update(_resistances(anc, meth, gF, Ft, Fv))
    deduc = 0.8 * a_cordon * S2 if jonction == "angle" else 0.0
    _flexion_platine(out, Z, ey / 2 - b_post / 2 - deduc, n, bp, (bp - ex) / 2, tp, lim)
    return _conclure(out, lim)


def pluriel_ancrage(nom: str) -> str:
    """« cheville M12 » → « chevilles M12 » : le premier mot, s'il est un nom commun au singulier,
    prend son pluriel (le nom de la cheville est saisi librement, toujours comptée par deux au moins)."""
    mots = (nom or "chevilles").strip().split(" ", 1)
    m0 = mots[0]
    if m0 and m0[0].islower() and m0[-1].isalpha() and not m0.endswith(("s", "x")):
        mots[0] = m0 + "s"
    return " ".join(mots)


def texte_ancrage(anc: Dict) -> str:
    n = int(anc.get("n_par_rangee") or 2)
    if anc.get("type") == "boulon":
        return f"2 × {n} boulons {anc.get('boulon', 'M12')} {anc.get('classe', '8.8')}"
    return f"2 × {n} {pluriel_ancrage(anc.get('nom'))} (N_Rd {float(anc.get('NRd') or 0):g} kN, V_Rd {float(anc.get('VRd') or 0):g} kN)"


def texte_platine(fx: Dict) -> str:
    """« platine 150 × 150 × 10 mm » (sur la dalle : b_p × l_p × t_p ; latérale : l_p × b_p × t_p, largeur × hauteur)."""
    pl = fx.get("platine") or {}
    bp, lp, tp = float(pl.get("bp") or 0), float(pl.get("lp") or 0), float(pl.get("tp") or 0)
    if fx.get("type") == LATERALE:
        return f"platine latérale {lp:g} × {bp:g} × {tp:g} mm (largeur × hauteur × épaisseur), descente d_h = {float(pl.get('d_h') or 0):g} mm"
    return f"platine {bp:g} × {lp:g} × {tp:g} mm"


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

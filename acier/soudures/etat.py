# -*- coding: utf-8 -*-
"""L'état d'une étude de soudures : des cas (configuration, paramètres,
efforts, méthode), la vérification complète d'un cas, le fichier JSON.
Une seule source : ``st.session_state["sdr"]`` à l'écran est ce dictionnaire.
"""
import copy
import json
from typing import Dict, List, Optional

import numpy as np

from acier.bibliotheques import DB
from acier.resistances import fu_bw_cordon

from . import configurations as CF
from . import criteres as K
from .cordon import A_MIN
from .groupe import Groupe

VERSION = 1
SEUIL_ATTENTION, SEUIL_NOK = 0.85, 1.0
OK, ATTENTION, NOK, REFUS = "ok", "attention", "nok", "refus"


# ------------------------------------------------------------------ cas et état
def methode_defaut() -> Dict:
    return {"mode": K.EUROCODE, "ec": K.DIRECTIONNELLE, "gamma_M2": K.GAMMA_M2, "k": K.K_BUREAU, "g": K.G_KG, "gamma_F": K.GAMMA_F_EQUIV}


def nouveau_cas(config: str = "C3", nom: Optional[str] = None) -> Dict:
    return {"nom": nom or f"{config} · {CF.CONFIGS[config]['libelle']}", "config": config,
            "params": copy.deepcopy(CF.PARAMS_DEFAUT[config]), "efforts": copy.deepcopy(CF.EFFORTS_DEFAUT[config]),
            "methode": methode_defaut()}


def changer_config(cas: Dict, config: str) -> None:
    """Change la configuration d'un cas : paramètres et efforts par défaut
    de la nouvelle, nuances et méthode conservées."""
    nuances = {k: cas["params"].get(k) for k in ("nuance_A", "nuance_B") if cas["params"].get(k)}
    cas["config"] = config
    cas["params"] = {**copy.deepcopy(CF.PARAMS_DEFAUT[config]), **nuances}
    cas["efforts"] = copy.deepcopy(CF.EFFORTS_DEFAUT[config])
    if cas["nom"].startswith("C") and "·" in cas["nom"]:
        cas["nom"] = f"{config} · {CF.CONFIGS[config]['libelle']}"


def nouvel_etat() -> Dict:
    return {"version": VERSION, "cas": [nouveau_cas("C3")], "actif": 0,
            "identification": {"projet": "", "partie": "", "indice": "0"}}


def cas_actif(etat: Dict) -> Dict:
    return etat["cas"][etat["actif"]]


def vers_json(etat: Dict) -> str:
    return json.dumps(etat, ensure_ascii=False, indent=1)


def depuis_json(texte: str) -> Dict:
    d = json.loads(texte)
    if not isinstance(d, dict) or "cas" not in d or not d["cas"]:
        raise ValueError("fichier de soudures sans cas")
    for c in d["cas"]:
        if c.get("config") not in CF.CONFIGS:
            raise ValueError(f"configuration inconnue : {c.get('config')!r}")
        c.setdefault("methode", methode_defaut())
        for k, v in methode_defaut().items():
            c["methode"].setdefault(k, v)
        c.setdefault("params", {})
        if c["config"] == "C8" and "elements" not in c["params"] and "cordons" in c["params"]:
            c["params"]["elements"] = CF.elements_c8(c["params"])      # ancien fichier : segments seuls
            del c["params"]["cordons"]
        for k, v in CF.PARAMS_DEFAUT[c["config"]].items():
            c["params"].setdefault(k, copy.deepcopy(v))
        for k, v in CF.EFFORTS_DEFAUT[c["config"]].items():
            c.setdefault("efforts", {}).setdefault(k, v)
    d.setdefault("actif", 0)
    d["actif"] = min(max(0, int(d["actif"])), len(d["cas"]) - 1)
    d.setdefault("identification", {"projet": "", "partie": "", "indice": "0"})
    d["version"] = VERSION
    return d


# ------------------------------------------------------------------ aciers
def nuances() -> Dict[str, Dict]:
    return {a["n"]: a for a in DB["aciers"]}


def acier_cordon(params: Dict) -> Dict:
    """fu et βw de référence du cordon : la pièce la plus faible (EN 1993-1-8 §4.5.3, Tab. 4.1)."""
    nz = nuances()
    A = nz.get(params.get("nuance_A", "S235"), nz["S235"])
    B = nz.get(params.get("nuance_B", "S235"), nz["S235"])
    fu, bw = fu_bw_cordon(B["fu"], B["bw"], A["fu"], A["bw"])
    ref = B["n"] if B["fu"] <= A["fu"] else A["n"]
    return {"fu": float(fu), "bw": float(bw), "nuance_ref": ref, "fy_min": float(min(A["fy"], B["fy"])), "A": A, "B": B}


# ------------------------------------------------------------------ vérification
def _position(cordon, s: float) -> str:
    ea = cordon.e_a
    if abs(ea[1]) > 0.7:
        return ("bas" if ea[1] > 0 else "haut") if s == 0.0 else ("haut" if ea[1] > 0 else "bas") if s == 1.0 else "milieu"
    if abs(ea[0]) > 0.7:
        return ("gauche" if ea[0] > 0 else "droite") if s == 0.0 else ("droite" if ea[0] > 0 else "gauche") if s == 1.0 else "milieu"
    return "début" if s == 0.0 else "fin" if s == 1.0 else "milieu"


def _statut(taux: float) -> str:
    return NOK if taux > SEUIL_NOK else ATTENTION if taux > SEUIL_ATTENTION else OK


def _chaines(cordons: List[Dict]) -> List[Dict]:
    """Les cordons tels que l'utilisateur les nomme : un cordon droit, ou une
    chaîne (pourtour, arc, tronçon) regroupant ses segments de calcul."""
    out: List[Dict] = []
    index: Dict[str, Dict] = {}
    for c in cordons:
        nom = c["chaine"]
        if nom not in index:
            index[nom] = {"nom": nom, "a": c["a"], "l": 0.0, "l_eff": 0.0, "segments": 0, "taux": c["taux"],
                          "position": c["position"], "f_max": c["f_max"], "retours": c["retours"], "continu": c["continu"],
                          "ferme": c["ferme"], "bout_a_bout": c["bout_a_bout"]}
            out.append(index[nom])
        ch = index[nom]
        ch["l"] += c["l"]
        ch["l_eff"] += c["l_eff"]
        ch["segments"] += 1
        if c["taux"] > ch["taux"]:
            ch["taux"], ch["position"], ch["f_max"] = c["taux"], c["position"], c["f_max"]
        if c["continu"]:
            ch["l"] = c["l_chaine"]        # la chaîne porte sa longueur réelle (avant raccourcissement de a à chaque bout)
    return out


def verifier(cas: Dict) -> Dict:
    """Vérification complète d'un cas : propriétés du groupe, force et
    composantes de gorge en chaque point, taux selon la méthode choisie et
    selon les autres (comparaison), point et cordon gouvernants, messages."""
    meth = K.Methode(**{k: cas["methode"][k] for k in ("mode", "ec", "gamma_M2", "k", "g", "gamma_F")})
    acier = acier_cordon(cas["params"])
    out = {"config": cas["config"], "methode": meth, "acier": acier, "messages": [], "points": [], "cordons": [], "chaines": [],
           "cordons_exclus": [], "statut": REFUS, "taux_max": None, "gouvernant": None, "comparaison": {}, "pleine": False}
    try:
        built = CF.construire(cas["config"], cas["params"], cas["efforts"])
    except (ValueError, KeyError) as e:
        out["messages"].append(f"Refus : {e}")
        return out
    out["pieces"], out["efforts"], out["beta_Lw"] = built["pieces"], built["efforts"], built["beta_Lw"]
    if built.get("pleine"):
        return _verifier_pleine(cas, out, acier, meth)
    cordons = built["cordons"]
    out["cordons_exclus"] = []
    vus = set()
    for c in cordons:
        if not c.utile:
            out["cordons_exclus"].append({"nom": c.nom, "chaine": c.chaine or c.nom, "a": c.a, "p1": [float(v) for v in c.P1],
                                          "p2": [float(v) for v in c.P2]})
            nom = c.chaine or c.nom
            if nom in vus:
                continue
            vus.add(nom)
            if c.chaine:
                out["messages"].append(f"{nom} : ne compte pas (gorge ≥ {A_MIN:g} mm et l − 2a ≥ max(30 mm ; 6a) exigés ; "
                                       f"a = {c.a:g}, l = {c.l_chaine:.0f} mm)")
            else:
                out["messages"].append(f"{c.nom} : ne compte pas (gorge ≥ {A_MIN:g} mm et l_eff ≥ max(30 mm ; 6a) exigés ; "
                                       f"a = {c.a:g}, l_eff = {c.l_eff:g} mm)")
    for d in built.get("discontinu", []):
        t_min = (built.get("pieces") or {}).get("t_min")
        esp = d.get("espacement")
        if esp is None:
            out["messages"].append(f"Soudure discontinue ({d['nom']}) : {d['n']} tronçon de {d['l']:g} mm")
            continue
        lim = f" ≤ min(200 mm ; 12 t = {12 * t_min:.0f} en compression ; 16 t = {16 * t_min:.0f} en traction) pour t = {t_min:g} mm" if t_min else " ≤ min(200 mm ; 12 t en compression ; 16 t en traction)"
        trop = t_min and esp > min(200.0, 16.0 * t_min) + 1e-9 or (not t_min and esp > 200.0 + 1e-9)
        out["messages"].append(f"Soudure discontinue ({d['nom']}) : {d['n']} tronçons de {d['l']:g} mm, espacement {esp:.0f} mm ; "
                               f"EN 1993-1-8 §4.3.2.2 : espacement{lim}" + (" — DÉPASSÉ" if trop else ""))
    if built["beta_Lw"] < 1.0:
        out["messages"].append(f"Assemblage long par recouvrement (§4.11) : βLw = {built['beta_Lw']:.3f} appliqué")
    try:
        g = Groupe(cordons, built["beta_Lw"])
        pr = g.proprietes()
        pts = g.forces(built["efforts"])
    except ValueError as e:
        out["messages"].append(f"Refus : {e}")
        return out
    out["proprietes"] = pr
    lim = {"directionnelle": 0.0, "simplifiee": 0.0, "bureau": 0.0, "bureau_equiv": 0.0}
    kq = K.k_equivalent_eurocode(acier["fu"], acier["bw"], meth.gamma_M2, meth.gamma_F, meth.g)
    out["k_equivalent"] = kq
    gouv = None
    for p in pts:
        ev = K.evaluer(p, meth, acier["fu"], acier["bw"])
        equiv = K.els_bureau(p["f"], p["cordon"].a, kq["k_kg_mm_cm"], meth.g)
        rec = {"cordon": p["cordon"].nom, "a": p["cordon"].a, "s": p["s"], "position": _position(p["cordon"], p["s"]),
               "point": [float(v) for v in p["point"]], "f_vec": [float(v) for v in p["f_vec"]], "f": p["f"],
               "sigma_perp": p["sigma_perp"], "tau_perp": p["tau_perp"], "tau_par": p["tau_par"],
               "sigma_c": ev["directionnelle"]["sigma_c"], "taux": ev["taux"], "critere": ev["critere"],
               "taux_directionnelle": ev["directionnelle"]["taux"], "taux_simplifiee": ev["simplifiee"]["taux"],
               "taux_bureau": ev["bureau"]["taux"], "taux_bureau_equiv": equiv["taux"]}
        out["points"].append(rec)
        lim["directionnelle"] = max(lim["directionnelle"], rec["taux_directionnelle"])
        lim["simplifiee"] = max(lim["simplifiee"], rec["taux_simplifiee"])
        lim["bureau"] = max(lim["bureau"], rec["taux_bureau"])
        lim["bureau_equiv"] = max(lim["bureau_equiv"], rec["taux_bureau_equiv"])
        if gouv is None or rec["taux"] > gouv["taux"]:
            gouv = rec
    for c in g.utiles:
        mine = [p for p in out["points"] if p["cordon"] == c.nom]
        pire = max(mine, key=lambda p: p["taux"])
        out["cordons"].append({"nom": c.nom, "chaine": c.chaine or c.nom, "a": c.a, "l": c.longueur, "l_eff": c.l_eff,
                               "retours": c.retours, "continu": c.continu, "ferme": c.ferme, "l_chaine": c.l_chaine,
                               "bout_a_bout": c.bout_a_bout,
                               "p1": [float(v) for v in c.P1], "p2": [float(v) for v in c.P2],
                               "u_A": [float(v) for v in c.u_A], "taux": pire["taux"], "position": pire["position"], "f_max": pire["f"]})
    out["chaines"] = _chaines(out["cordons"])
    out["comparaison"] = lim
    out["gouvernant"] = gouv
    out["taux_max"] = gouv["taux"]
    out["statut"] = _statut(gouv["taux"])
    d = K.directionnelle(0, 0, 0, acier["fu"], acier["bw"], meth.gamma_M2)
    s = K.simplifiee(0, 1.0, acier["fu"], acier["bw"], meth.gamma_M2)
    out["resistances"] = {"lim_c": d["lim_c"], "lim_perp": d["lim_perp"], "f_vw_d": s["f_vw_d"],
                          "tau_adm_bureau": K.tau_adm_bureau(meth.k, meth.g), "tau_equiv": kq["tau_N_mm2"], "k_equiv": kq["k_kg_mm_cm"]}
    if meth.mode == K.ELS:
        out["messages"].append(f"Méthode ELS du bureau : efforts de service, résultante ≤ k·a avec k = {meth.k:g} kg/(mm·cm) "
                               f"= {out['resistances']['tau_adm_bureau']:.1f} N/mm² ; l'Eurocode donnerait f_vw,d/{meth.gamma_F:g} = "
                               f"{kq['tau_N_mm2']:.1f} N/mm² ({kq['k_kg_mm_cm']:.0f} kg/(mm·cm)) pour le {acier['nuance_ref']}")
    return out


def _verifier_pleine(cas: Dict, out: Dict, acier: Dict, meth) -> Dict:
    """Bout à bout à pleine pénétration : la pièce la plus faible gouverne (§4.7.1)."""
    p, e = cas["params"], cas["efforts"]
    gM0 = 1.0
    r = K.piece_pleine_penetration(float(e.get("N", 0)) * 1e3, float(e.get("V", 0)) * 1e3, float(e.get("M", 0)) * 1e6,
                                   float(p["b"]), float(p["t"]), acier["fy_min"], gM0)
    taux = r["taux"] / (meth.gamma_F if meth.mode == K.ELS else 1.0) if False else r["taux"]
    out.update({"pleine": True, "piece": r, "taux_max": taux, "statut": _statut(taux),
                "gouvernant": {"cordon": "pièce", "position": "section du plat", "taux": taux, "critere": r["critere"],
                               "sigma_perp": r["sigma"], "tau_perp": 0.0, "tau_par": r["tau"], "sigma_c": r["sigma_c"], "f": 0.0,
                               "a": float(p["t"]), "s": 0.5},
                "comparaison": {"piece": taux}, "proprietes": {"A_w": float(p["b"]) * float(p["t"]), "n_cordons": 0, "l_total": float(p["b"])},
                "resistances": {"lim_piece": r["lim"]}})
    out["messages"].append("Pleine pénétration : la soudure vaut la pièce la plus faible, aucune vérification de cordon "
                           f"(§4.7.1) ; pièce {p['b']:g} × {p['t']:g} mm vérifiée élastiquement avec fy = {acier['fy_min']:g} N/mm², γM0 = {gM0:g}")
    if meth.mode == K.ELS:
        out["messages"].append("Méthode ELS : les efforts saisis sont de service ; la pièce est comparée à fy/γM0 sans coefficient de charge")
    return out

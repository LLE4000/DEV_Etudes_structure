# -*- coding: utf-8 -*-
"""L'état d'une étude de garde-corps : des cas (type, géométrie, charges, sections,
remplissage, fixation, méthode), la VÉRIFICATION complète d'un cas, la bibliothèque des
candidats pour chaque élément, le fichier JSON. Une seule source : ``st.session_state["gc"]``
à l'écran est ce dictionnaire.

Un cas :
- ``type`` : le type de garde-corps du menu (``types.py``) — il fixe le système et le remplissage ;
- ``geometrie`` : système (montants verticaux | lisse ancrée au mur), forme en plan (droit, L,
  U), hauteur H de la main courante au-dessus de la fixation, tronçons (longueur + entraxe
  maximal, ou entraxes saisis), extrémités ;
- ``charges`` : les quatre charges nommées (``charges.py``), cas distincts ou cumul ;
- ``montant`` (section), ``lisse`` (section de la main courante, appuis « deux » ou « continue »),
  ``remplissage`` (barreaux à entraxe e_b entre la lisse basse à h_bas et la main courante, lisses
  intermédiaires, panneau, aucun ; la portée des barreaux se DÉDUIT de H, h_bas et des sections),
  ``fixation`` (platine, soudure, ancrage, console), ``methode`` (ELS du bureau | Eurocode).

``verifier`` rend tout ce que l'écran et la note impriment : les montants un à un avec leur
longueur tributaire, chaque élément sous le cas de charge qui le gouverne, la soudure et la
platine, les messages — rien n'est recalculé ailleurs.
"""
import copy
import json
import math
from typing import Dict, List, Optional

from acier.bibliotheques import DB

from . import bibliotheque as BI
from . import charges as CH
from . import fixation as FX
from . import geometrie as GE
from . import membres as ME
from . import types as TY

VERSION = 2
OK, ATTENTION, NOK, REFUS = ME.OK, ME.ATTENTION, ME.NOK, ME.REFUS
REMPLISSAGES = {"barreaux": "barreaux verticaux", "lisses": "lisses intermédiaires", "panneau": "panneau (verre, tôle)", "aucun": "aucun (main courante seule)"}
LIBELLES = {"montant": "Montant principal", "lisse": "Main courante", "barreau": "Barreaux", "lisse_inter": "Lisses intermédiaires", "console": "Console",
            "soudure": "Soudure sur la platine", "ancrage": "Chevilles / boulons", "platine": "Platine"}
ROLES = {"montant": "montant principal", "lisse": "main courante", "barreau": "barreaux", "lisse_inter": "lisses intermédiaires", "console": "console"}
SECTIONS_DEFAUT = {"montant": "RHS 50x30x3", "lisse": "RHS 50x30x3", "barreau": "Plat 30x8", "lisse_inter": "RHS 50x25x2.5 (froid)",
                   "lisse_basse": "RHS 50x25x2.5 (froid)", "console": "Plat 50x10"}


# ------------------------------------------------------------------ cas et état
def nouveau_cas(nom: Optional[str] = None, type_: str = TY.TYPE_DEFAUT) -> Dict:
    c = {"nom": nom or "Garde-corps 1", "type": type_, "geometrie": GE.geometrie_defaut(), "charges": CH.charges_defaut(),
         "montant": {"section": {"nom": SECTIONS_DEFAUT["montant"]}},
         "lisse": {"section": {"nom": SECTIONS_DEFAUT["lisse"]}, "appuis": "deux"},
         "remplissage": {"type": "barreaux", "section": {"nom": SECTIONS_DEFAUT["barreau"]}, "entraxe": 110.0, "h_bas": 100.0,
                         "section_basse": {"nom": SECTIONS_DEFAUT["lisse_basse"]},
                         "n_lisses": 2, "entraxe_lisses": 330.0, "section_lisses": {"nom": SECTIONS_DEFAUT["lisse_inter"]}},
         "fixation": FX.fixation_defaut(), "methode": ME.methode_defaut()}
    TY.appliquer_type(c, type_)
    return c


def nouvel_etat() -> Dict:
    return {"version": VERSION, "cas": [nouveau_cas()], "actif": 0, "identification": {"projet": "", "partie": "", "indice": "0"}}


def cas_actif(etat: Dict) -> Dict:
    return etat["cas"][etat["actif"]]


def normaliser_cas(c: Dict) -> Dict:
    """Complète un cas (fichier ancien, clés manquantes) par les défauts ; un cas de la version 1
    (sans type, barreaux à portée saisie) est relu : le type se déduit, la portée se déduit."""
    d = nouveau_cas(c.get("nom"))
    for k in ("geometrie", "montant", "lisse", "remplissage", "fixation", "methode", "charges"):
        v = c.get(k)
        if isinstance(v, dict):
            if k == "fixation":
                d[k] = FX.normaliser(v)
            elif k == "charges":
                d[k] = CH.normaliser(v)
            else:
                d[k] = {**d[k], **v}
    g = d["geometrie"]
    if g.get("systeme") not in GE.SYSTEMES:
        g["systeme"] = "montants"
    if g.get("forme") not in GE.FORMES:
        g["forme"] = "droit"
    for k in ("debut", "fin"):
        if g.get(k) not in GE.EXTREMITES:
            g[k] = "montant"
    trs = [t for t in (g.get("troncons") or []) if isinstance(t, dict)]
    while len(trs) < 3:
        trs.append(GE.troncon_defaut(2400.0))
    g["troncons"] = [{**GE.troncon_defaut(), **t} for t in trs[:3]]
    if d["remplissage"].get("type") not in REMPLISSAGES:
        d["remplissage"]["type"] = "barreaux"
    if d["lisse"].get("appuis") not in ME.APPUIS:
        d["lisse"]["appuis"] = "deux"
    d["type"] = TY.type_de({**d, "type": c.get("type")})
    return d


def vers_json(etat: Dict) -> str:
    return json.dumps(etat, ensure_ascii=False, indent=1)


def depuis_json(texte: str) -> Dict:
    d = json.loads(texte)
    if not isinstance(d, dict) or not d.get("cas"):
        raise ValueError("fichier de garde-corps sans cas")
    d["cas"] = [normaliser_cas(c) for c in d["cas"]]
    d["actif"] = min(max(0, int(d.get("actif", 0))), len(d["cas"]) - 1)
    d.setdefault("identification", {"projet": "", "partie": "", "indice": "0"})
    d["version"] = VERSION
    return d


# ------------------------------------------------------------------ aciers
def nuances() -> Dict[str, Dict]:
    return {a["n"]: a for a in DB["aciers"]}


def acier(meth: Dict) -> Dict:
    nz = nuances()
    a = nz.get(meth.get("nuance", "S235"), nz["S235"])
    return {"nuance": a["n"], "fy": float(a["fy"]), "fu": float(a["fu"])}


# ------------------------------------------------------------------ éléments
def _kn(v: float) -> float:
    return v / 1e3


def _p_montant(ch: Dict, m: Dict) -> Dict:
    """Les efforts de service (N) en tête d'un montant : P₁, P₂ (horizontaux, par direction) et N (vertical)."""
    if ch["mode"] == "P":
        P1 = ch["P"] * 1e3
        P2 = ch["P"] * 1e3 if m["coin"] else 0.0
    else:
        P1 = ch["qh"] * m["trib1"]        # kN/m × mm = N
        P2 = ch["qh"] * m["trib2"]
    N = ch["qv"] * (m["trib1"] + m["trib2"])
    return {"P1": P1, "P2": P2, "N": N}


def _cas_montant(ch: Dict, p: Dict):
    """Les cas de charge d'un montant : (label, P₁, P₂, N)."""
    if ch.get("cumul"):
        return [("1+2", p["P1"], p["P2"], p["N"])]
    return [("1", p["P1"], p["P2"], 0.0), ("2", 0.0, 0.0, p["N"])]


def verifier_montant(sec: Dict, ch: Dict, m: Dict, H: float, meth: Dict, fy: float) -> Dict:
    """La vérification d'un montant : console de hauteur H, sous chaque cas de charge ; le pire
    gouverne. Un montant d'angle d'une section rectangulaire est vérifié dans les deux
    orientations (P₁ selon h, puis P₂ selon h) et garde la pire."""
    p = _p_montant(ch, m)
    E = float(meth.get("E") or ME.E_ACIER)
    Iy, Iz = sec["Iy"] * 1e4, sec["Iz"] * 1e4

    def essai(P1, P2, N):
        c = ME.console(P1, P2, N, H, E, Iy, Iz)
        v = ME.verifier_section(sec, c["M1"], c["M2"], N, c["V1"], c["V2"], c["f"], H, meth, fy)
        v.update({"P1": P1, "P2": P2, "f1": c["f1"], "f2": c["f2"]})
        return v
    meilleur, cas_taux = None, {}
    for lab, P1, P2, N in _cas_montant(ch, p):
        v = essai(P1, P2, N)
        if m["coin"] and abs(sec["Wel_y"] - sec["Wel_z"]) > 1e-9 and P2 > 0:
            v2 = essai(P2, P1, N)
            if v2["taux_max"] > v["taux_max"]:
                v = v2
                v["orientation_inversee"] = True
        v["cas_charge"] = lab
        cas_taux[lab] = v["taux_max"]
        if meilleur is None or v["taux_max"] > meilleur["taux_max"]:
            meilleur = v
    meilleur.update({"nom": m["nom"], "coin": m["coin"], "trib1": m["trib1"], "trib2": m["trib2"], "x": m["x"], "H": H, "cas_taux": cas_taux,
                     "P_h": p["P1"], "P_h2": p["P2"], "N_v": p["N"]})
    return meilleur


def _flexion(q: float, Q: float, L: float, n: int, appuis: str, E: float, I: float, comb: str) -> Dict:
    if appuis == "continue" and n >= 2:
        return ME.poutre_continue(q, Q, L, n, E, I)
    return ME.poutre(q, Q, L, E, I, comb)


def verifier_lisse(sec: Dict, qh: float, qv: float, Q: float, L: float, n: int, appuis: str, meth: Dict, fy: float, comb: str, cumul: bool) -> Dict:
    """La main courante (ou une lisse principale) de portée L (mm), sur deux appuis ou continue
    sur n travées : la charge horizontale (q_h + Q) fléchit l'axe faible (h vertical), la charge
    verticale q_v l'axe fort — cas distincts (le pire gouverne) ou cumulés."""
    E = float(meth.get("E") or ME.E_ACIER)
    Iy, Iz = sec["Iy"] * 1e4, sec["Iz"] * 1e4
    ph = _flexion(qh, Q, L, n, appuis, E, Iz, comb)
    pv = _flexion(qv, 0.0, L, n, appuis, E, Iy, "somme")
    base = {"qh": qh, "qv": qv, "Q": Q, "Mh": ph["M"], "Mv": pv["M"], "Vh": ph["V"], "Vv": pv["V"], "fh": ph["f"], "fv": pv["f"],
            "Mq": ph["Mq"], "MQ": ph["MQ"], "comb": comb, "L": L, "appuis": appuis, "n_travees": n, "kM": ph.get("kM"), "kV": ph.get("kV"), "kf": ph.get("kf")}
    if cumul:
        v = ME.verifier_section(sec, pv["M"], ph["M"], 0.0, pv["V"], ph["V"], math.hypot(ph["f"], pv["f"]), L, meth, fy)
        v.update(base)
        v.update({"cas_charge": "1+2", "cas_taux": {"1+2": v["taux_max"]}})
        return v
    v1 = ME.verifier_section(sec, 0.0, ph["M"], 0.0, 0.0, ph["V"], ph["f"], L, meth, fy)
    v2 = ME.verifier_section(sec, pv["M"], 0.0, 0.0, pv["V"], 0.0, pv["f"], L, meth, fy)
    v = dict(v1 if v1["taux_max"] >= v2["taux_max"] else v2)
    v.update(base)
    v.update({"cas_charge": "1" if v1["taux_max"] >= v2["taux_max"] else "2", "cas_taux": {"1": v1["taux_max"], "2": v2["taux_max"]},
              "verif_1": v1, "verif_2": v2})
    return v


def verifier_barre(sec: Dict, q: float, Q: float, L: float, meth: Dict, fy: float) -> Dict:
    """Un élément du remplissage (barreau, lisse intermédiaire) de portée L sous la charge
    ponctuelle Q_r (cas 3) OU la pression q_r × entraxe (cas 4) — deux cas, le pire."""
    E = float(meth.get("E") or ME.E_ACIER)
    Iy = sec["Iy"] * 1e4
    pQ = ME.poutre(0.0, Q, L, E, Iy, "somme")
    pq = ME.poutre(q, 0.0, L, E, Iy, "somme")
    vQ = ME.verifier_section(sec, pQ["M"], 0.0, 0.0, pQ["V"], 0.0, pQ["f"], L, meth, fy)
    vq = ME.verifier_section(sec, pq["M"], 0.0, 0.0, pq["V"], 0.0, pq["f"], L, meth, fy)
    pire_Q = vQ["taux_max"] >= vq["taux_max"]
    v = dict(vQ if pire_Q else vq)
    v.update({"cas": "Q" if pire_Q else "q", "cas_charge": "3" if pire_Q else "4", "cas_taux": {"3": vQ["taux_max"], "4": vq["taux_max"]},
              "q": q, "Q": Q, "L": L, "verif_q": vq, "verif_Q": vQ})
    return v


def verifier_console(sec: Dict, ch: Dict, m: Dict, e: float, meth: Dict, fy: float) -> Dict:
    """Une console de fixation au mur : la charge verticale (cas 2) la fléchit (M = R_v·e), la
    charge horizontale (cas 1) la tire hors du mur (axial R_h) et, à l'angle, la fléchit
    latéralement ; cas distincts ou cumulés."""
    E = float(meth.get("E") or ME.E_ACIER)
    Iy, Iz = sec["Iy"] * 1e4, sec["Iz"] * 1e4
    Rv = ch["qv"] * (m["trib1"] + m["trib2"])
    Rh1 = ch["P"] * 1e3 if ch["mode"] == "P" else ch["qh"] * m["trib1"]
    Rh2 = (ch["P"] * 1e3 if ch["mode"] == "P" else ch["qh"] * m["trib2"]) if m["coin"] else 0.0
    cas = [("1+2", Rv, Rh2, Rh1)] if ch.get("cumul") else [("1", 0.0, Rh2, Rh1), ("2", Rv, 0.0, 0.0)]
    meilleur, cas_taux = None, {}
    for lab, P1, P2, N in cas:
        c = ME.console(P1, P2, N, e, E, Iy, Iz)
        v = ME.verifier_section(sec, c["M1"], c["M2"], N, c["V1"], c["V2"], c["f"], e, meth, fy)
        v.update({"P1": P1, "P2": P2, "f1": c["f1"], "f2": c["f2"], "cas_charge": lab})
        cas_taux[lab] = v["taux_max"]
        if meilleur is None or v["taux_max"] > meilleur["taux_max"]:
            meilleur = v
    meilleur.update({"nom": m["nom"].replace("M", "C"), "coin": m["coin"], "trib1": m["trib1"], "trib2": m["trib2"], "x": m["x"],
                     "Rv": Rv, "Rh": Rh1, "Rh2": Rh2, "e": e, "H": e, "cas_taux": cas_taux})
    return meilleur


# ------------------------------------------------------------------ géométrie du remplissage
def portee_barreaux(H: float, rp: Dict, sec_lisse: Dict, sec_basse: Optional[Dict]) -> float:
    """La portée d'un barreau : entre le dessus de la lisse basse (à h_bas) et le dessous de la main courante."""
    h_mc = float(sec_lisse.get("h") or sec_lisse.get("d") or 0.0)
    h_lb = float((sec_basse or {}).get("h") or (sec_basse or {}).get("d") or 0.0)
    return max(100.0, H - float(rp.get("h_bas") or 0.0) - h_mc / 2 - h_lb / 2)


# ------------------------------------------------------------------ vérification complète
def _section(spec: Dict, quoi: str, messages: List[str]) -> Optional[Dict]:
    try:
        return BI.section(spec)
    except ValueError as e:
        messages.append(f"Refus : section {quoi} — {e}")
        return None


def _n_travees_max(ana: Dict) -> int:
    return max((len(e) for e in ana["entraxes"]), default=1)


def verifier(cas: Dict) -> Dict:
    cas = normaliser_cas(cas)
    geom, ch, meth = cas["geometrie"], cas["charges"], cas["methode"]
    ac = acier(meth)
    fy = ac["fy"]
    lim = ME.limites(meth, fy)
    ana = GE.analyser(geom)
    H = float(geom.get("H") or GE.H_DEFAUT)
    systeme = geom.get("systeme", "montants")
    typ_gc = TY.type_de(cas)
    out: Dict = {"statut": REFUS, "taux_max": None, "gouvernant": None, "elements": [], "messages": list(ana["messages"]),
                 "type": typ_gc, "type_libelle": TY.TYPES[typ_gc]["libelle"],
                 "charges": ch, "charges_tableau": CH.tableau(ch), "cumul": bool(ch.get("cumul")),
                 "geometrie": {**ana, "H": H, "systeme": systeme, "phrase": GE.phrase(geom, ana)}, "acier": ac, "methode": meth,
                 "limites": lim, "montants": [], "montant": None, "lisse": None, "remplissage": None, "consoles": [], "console": None,
                 "fixation": None, "masse": 0.0}
    if ana["messages"]:
        return out
    if H > CH.H_MAX_APPLICATION:
        out["messages"].append(f"La charge linéaire s'applique au plus à {CH.H_MAX_APPLICATION / 1000:.2f} m (EN 1991-1-1 §6.4) : "
                               f"H = {H / 1000:.2f} m est pris comme bras de levier, du côté sûr")
    comb = meth.get("comb", "somme")
    s_max = ana["s_max"]
    n_tr = _n_travees_max(ana)
    appuis = cas["lisse"].get("appuis", "deux")
    elements = []
    masse = 0.0
    sec_lisse = _section(cas["lisse"]["section"], "de la main courante", out["messages"])
    if sec_lisse is None:
        return out
    qh_lisse = (ch["P"] * 1e3 / s_max if ch["mode"] == "P" else ch["qh"]) if s_max > 0 else 0.0     # N/mm
    if systeme == "montants":
        sec = _section(cas["montant"]["section"], "du montant", out["messages"])
        if sec is None:
            return out
        out["section_montant"] = sec
        for m in ana["montants"]:
            out["montants"].append(verifier_montant(sec, ch, m, H, meth, fy))
        gouv = max(out["montants"], key=lambda v: v["taux_max"])
        out["montant"] = {"section": sec, "verif": gouv, "nom": gouv["nom"], "n": len(out["montants"])}
        elements.append(("montant", f"Montant principal {gouv['nom']} · {sec['nom']}", gouv["taux_max"], gouv["critere"], gouv))
        masse += len(out["montants"]) * BI.masse_lineaire(sec, H)
    vl = verifier_lisse(sec_lisse, qh_lisse, ch["qv"], ch["Q"] * 1e3, s_max, n_tr, appuis, meth, fy, comb, bool(ch.get("cumul")))
    out["lisse"] = {"section": sec_lisse, "verif": vl, "L": s_max, "appuis": appuis, "n_travees": n_tr,
                    "role": "principal" if systeme == "mur" else "secondaire"}
    elements.append(("lisse", f"Main courante · {sec_lisse['nom']} · portée {s_max / 1000:.2f} m, {ME.APPUIS[appuis]}".replace(".", ","), vl["taux_max"], vl["critere"], vl))
    masse += BI.masse_lineaire(sec_lisse, ana["L_total"])
    if systeme == "mur":
        out["section_montant"] = sec_lisse
        sec_c = _section(cas["fixation"]["console"]["section"], "de la console", out["messages"])
        if sec_c is None:
            return out
        e = float(cas["fixation"]["console"].get("e") or 100.0)
        for m in ana["montants"]:
            out["consoles"].append(verifier_console(sec_c, ch, m, e, meth, fy))
        gc = max(out["consoles"], key=lambda v: v["taux_max"])
        out["console"] = {"section": sec_c, "verif": gc, "nom": gc["nom"], "e": e, "n": len(out["consoles"])}
        elements.append(("console", f"Console {gc['nom']} · {sec_c['nom']} · e = {e:g} mm", gc["taux_max"], gc["critere"], gc))
        masse += len(out["consoles"]) * BI.masse_lineaire(sec_c, e)
    for r in ana["reprises_mur"]:
        out["messages"].append(f"Ancrage de la main courante dans le mur ({r['ou']}) : il reprend H = {_kn(qh_lisse * r['trib']):.2f} kN "
                               f"et V = {_kn(ch['qv'] * r['trib']):.2f} kN (service) — à justifier avec le fabricant de la cheville")
    # le remplissage
    rp = cas["remplissage"]
    typ = rp.get("type", "barreaux")
    out["remplissage"] = {"type": typ, "libelle": REMPLISSAGES[typ], "h_bas": float(rp.get("h_bas") or 100.0)}
    if typ in ("barreaux", "panneau"):
        sec_lb = _section(rp.get("section_basse") or {"nom": SECTIONS_DEFAUT["lisse_basse"]}, "de la lisse basse", out["messages"])
        out["remplissage"]["section_basse"] = sec_lb
        if sec_lb is not None:
            masse += BI.masse_lineaire(sec_lb, ana["L_total"])
    if typ == "barreaux":
        sec_b = _section(rp["section"], "des barreaux", out["messages"])
        if sec_b is not None:
            eb = float(rp.get("entraxe") or 110.0)
            Lb = portee_barreaux(H, rp, sec_lisse, out["remplissage"].get("section_basse"))
            vb = verifier_barre(sec_b, ch["qr"] * eb / 1e3, ch["Qr"] * 1e3, Lb, meth, fy)
            out["remplissage"].update({"section": sec_b, "verif": vb, "entraxe": eb, "L": Lb})
            elements.append(("barreau", f"Barreaux · {sec_b['nom']} · e = {eb:g} mm, portée {Lb:.0f} mm", vb["taux_max"], vb["critere"], vb))
            masse += (ana["L_total"] / eb) * BI.masse_lineaire(sec_b, Lb)
    elif typ == "lisses":
        sec_l = _section(rp.get("section_lisses") or {"nom": SECTIONS_DEFAUT["lisse_inter"]}, "des lisses intermédiaires", out["messages"])
        if sec_l is not None:
            el, nl = float(rp.get("entraxe_lisses") or 300.0), int(rp.get("n_lisses") or 1)
            vb = verifier_barre(sec_l, ch["qr"] * el / 1e3, ch["Qr"] * 1e3, s_max, meth, fy)
            out["remplissage"].update({"section": sec_l, "verif": vb, "entraxe": el, "L": s_max, "n": nl})
            elements.append(("lisse_inter", f"Lisses intermédiaires ({nl}) · {sec_l['nom']} · e = {el:g} mm, portée {s_max / 1000:.2f} m".replace(".", ","),
                             vb["taux_max"], vb["critere"], vb))
            masse += nl * BI.masse_lineaire(sec_l, ana["L_total"])
    elif typ == "panneau":
        out["messages"].append(f"Remplissage en panneau : sa résistance sous Q_r = {ch['Qr']:g} kN et q_r = {ch['qr']:g} kN/m² est à justifier par le fournisseur")
    # la fixation
    out["fixation"] = _fixation(cas, out, meth, fy, ac)
    fx = out["fixation"]
    if fx.get("soudure"):
        s = fx["soudure"]
        elements.append(("soudure", f"Soudure · {FX.texte_soudure(cas['fixation']['soudure'])}", s["taux"], s["critere"], s))
    if fx.get("platine"):
        pl = fx["platine"]
        if pl.get("taux_max") is not None:
            elements.append(("ancrage", f"{LIBELLES['ancrage']} · {FX.texte_ancrage(cas['fixation']['ancrage'])}", pl["taux_boulons"],
                             "traction" if pl["taux_t"] >= max(pl["taux_v"], pl["inter"]) else ("cisaillement" if pl["taux_v"] >= pl["inter"] else "interaction"), pl))
            elements.append(("platine", f"Platine {pl['bp']:g} × {pl['lp']:g} × {pl['tp']:g} · flexion", pl["taux_platine"], "flexion", pl))
        else:
            out["messages"].extend(pl.get("messages", []))
    out["elements"] = [{"cle": k, "libelle": lib, "taux": t, "critere": c, "statut": ME.statut(t), "detail": d} for k, lib, t, c, d in elements
                       if t is not None]
    if not out["elements"]:
        return out
    g = max(out["elements"], key=lambda e: e["taux"])
    out["gouvernant"] = {"cle": g["cle"], "libelle": g["libelle"], "critere": g["critere"], "taux": g["taux"]}
    out["taux_max"] = g["taux"]
    out["statut"] = ME.statut(g["taux"])
    out["masse"] = masse
    return out


def _cas_fixation(out: Dict, cumul: bool):
    """Les cas de charge à la platine : (label, M N·mm, N (> 0 traction), V, P₁, P₂, L) pour la soudure et la platine."""
    systeme = out["geometrie"]["systeme"]
    if systeme == "montants":
        v = out["montant"]["verif"]
        P1, P2, N, H = v["P_h"], v["P_h2"], v["N_v"], v["H"]
        if cumul:
            return [("1+2", math.hypot(P1 * H, P2 * H), -N, math.hypot(P1, P2), P1, P2, H)]
        return [("1", math.hypot(P1 * H, P2 * H), 0.0, math.hypot(P1, P2), P1, P2, H), ("2", 0.0, -N, 0.0, 0.0, 0.0, H)]
    v = out["console"]["verif"]
    Rv, Rh, Rh2, e = v["Rv"], v["Rh"], v["Rh2"], v["e"]
    if cumul:
        return [("1+2", math.hypot(Rv * e, Rh2 * e), Rh, math.hypot(Rv, Rh2), Rv, Rh2, e)]
    return [("1", Rh2 * e, Rh, Rh2, 0.0, Rh2, e), ("2", Rv * e, 0.0, Rv, Rv, 0.0, e)]


def _fixation(cas: Dict, out: Dict, meth: Dict, fy: float, ac: Dict) -> Dict:
    """La soudure et la platine de l'élément fixé le plus chargé (montant gouvernant, ou console
    gouvernante dans le système mur), sous chaque cas de charge ; le pire gouverne."""
    fx = cas["fixation"]
    typ = fx.get("type", "platine")
    res: Dict = {"type": typ, "libelle": FX.TYPES[typ], "soudure": None, "platine": None, "taux_max": None, "statut": None}
    if typ == "scellee":
        out["messages"].append("Montant scellé : la fixation n'est pas vérifiée (scellement à justifier séparément)")
        return res
    systeme = out["geometrie"]["systeme"]
    if (systeme == "montants" and not out.get("montant")) or (systeme == "mur" and not out.get("console")):
        return res
    sec = out["montant"]["section"] if systeme == "montants" else out["console"]["section"]
    hp = float(sec["h"] or sec["d"])
    soud = fx["soudure"]
    tp = float(fx["platine"].get("tp") or 10.0)
    g = out["limites"]["gamma"]
    for lab, M, N_trac, V, P1, P2, L in _cas_fixation(out, bool(cas["charges"].get("cumul"))):
        cas_s = FX.cas_soudure(sec, _kn(N_trac) * g, _kn(P1) * g, _kn(P2) * g, L, soud, tp, meth, ac["nuance"])
        s = FX.verifier_soudure(cas_s)
        s["cas_charge"] = lab
        s["efforts_txt"] = (f"N = {_kn(N_trac) * g:.2f} kN, V = {_kn(V) * g:.2f} kN, M = {M * g / 1e6:.3f} kN·m"
                            + (" (efforts de calcul)" if g != 1.0 else " (efforts de service)"))
        if res["soudure"] is None or (s["taux"] or 0) > (res["soudure"]["taux"] or 0):
            res["soudure"] = s
        if typ == "platine":
            pl = FX.platine_boulonnee(M, -N_trac, V, hp, fx["platine"], fx["ancrage"], float(soud.get("a") or 0.0), meth, fy, soud.get("jonction", "angle"))
            pl["cas_charge"] = lab
            if res["platine"] is None or (pl.get("taux_max") or 0) > (res["platine"].get("taux_max") or 0):
                res["platine"] = pl
    taux = [t for t in (res["soudure"]["taux"], (res["platine"] or {}).get("taux_max")) if t is not None]
    res["taux_max"] = max(taux) if taux else None
    res["statut"] = ME.statut(res["taux_max"]) if taux else REFUS
    return res


# ------------------------------------------------------------------ bibliothèque : les candidats par élément
def element_principal(cas: Dict) -> str:
    return "lisse" if normaliser_cas(cas)["geometrie"].get("systeme") == "mur" else "montant"


def roles_du_cas(cas: Dict) -> List[str]:
    """Les rôles que la bibliothèque peut dimensionner pour ce cas, dans l'ordre des onglets."""
    c = normaliser_cas(cas)
    out = []
    if c["geometrie"].get("systeme") == "mur":
        out += ["lisse", "console"]
    else:
        out += ["montant", "lisse"]
    rp = c["remplissage"].get("type")
    if rp == "barreaux":
        out.append("barreau")
    elif rp == "lisses":
        out.append("lisse_inter")
    return out


def spec_du_role(cas: Dict, role: str) -> Dict:
    if role == "montant":
        return cas["montant"]["section"]
    if role == "lisse":
        return cas["lisse"]["section"]
    if role == "barreau":
        return cas["remplissage"]["section"]
    if role == "lisse_inter":
        return cas["remplissage"].get("section_lisses") or {"nom": SECTIONS_DEFAUT["lisse_inter"]}
    if role == "console":
        return cas["fixation"]["console"]["section"]
    raise ValueError(f"rôle inconnu : {role}")


def appliquer_section(cas: Dict, sec: Dict, role: str = "montant") -> None:
    """Enregistre la section dans l'élément du rôle."""
    spec = BI.spec_de(sec)
    if role == "montant":
        cas["montant"]["section"] = spec
    elif role == "lisse":
        cas["lisse"]["section"] = spec
    elif role == "barreau":
        cas["remplissage"]["section"] = spec
    elif role == "lisse_inter":
        cas["remplissage"]["section_lisses"] = spec
    elif role == "console":
        cas["fixation"]["console"]["section"] = spec
    else:
        raise ValueError(f"rôle inconnu : {role}")


def _verifier_role(sec: Dict, role: str, cas: Dict, res: Dict) -> Dict:
    meth, ch = cas["methode"], cas["charges"]
    fy = acier(meth)["fy"]
    ana = res["geometrie"]
    H, s_max = ana["H"], ana["s_max"]
    qh = (ch["P"] * 1e3 / s_max if ch["mode"] == "P" else ch["qh"]) if s_max > 0 else 0.0
    if role == "montant":
        return max((verifier_montant(sec, ch, m, H, meth, fy) for m in ana["montants"]), key=lambda v: v["taux_max"])
    if role == "lisse":
        return verifier_lisse(sec, qh, ch["qv"], ch["Q"] * 1e3, s_max, _n_travees_max(ana), cas["lisse"].get("appuis", "deux"), meth, fy,
                              meth.get("comb", "somme"), bool(ch.get("cumul")))
    if role == "barreau":
        rp = cas["remplissage"]
        Lb = (res.get("remplissage") or {}).get("L") or portee_barreaux(H, rp, res["lisse"]["section"], (res.get("remplissage") or {}).get("section_basse"))
        return verifier_barre(sec, ch["qr"] * float(rp.get("entraxe") or 110.0) / 1e3, ch["Qr"] * 1e3, Lb, meth, fy)
    if role == "lisse_inter":
        return verifier_barre(sec, ch["qr"] * float(cas["remplissage"].get("entraxe_lisses") or 300.0) / 1e3, ch["Qr"] * 1e3, s_max, meth, fy)
    if role == "console":
        e = float(cas["fixation"]["console"].get("e") or 100.0)
        return max((verifier_console(sec, ch, m, e, meth, fy) for m in ana["montants"]), key=lambda v: v["taux_max"])
    raise ValueError(f"rôle inconnu : {role}")


def candidats(cas: Dict, role: str = "montant", groupes: Optional[List[str]] = None, masse_max: Optional[float] = 25.0,
              dim_max: Optional[float] = 150.0) -> List[Dict]:
    """Chaque section standard des groupes demandés, vérifiée dans le rôle (montant gouvernant,
    main courante, barreaux, lisses intermédiaires, console) : [{section, taux, critere, statut,
    taux_sigma, taux_tau, taux_fleche, masse}], par masse croissante. ``plus_leger`` en tire le
    premier qui tient."""
    cas = normaliser_cas(cas)
    res = verifier(cas)
    if not res.get("geometrie", {}).get("montants") or not res.get("lisse"):
        return []
    out = []
    for sec in BI.catalogue(groupes, masse_max, dim_max):
        v = _verifier_role(sec, role, cas, res)
        out.append({"section": sec, "taux": v["taux_max"], "critere": v["critere"], "statut": v["statut"], "taux_sigma": v["taux"]["sigma"],
                    "taux_tau": v["taux"]["tau"], "taux_fleche": v["taux"]["fleche"], "masse": sec["masse"]})
    return out


def plus_leger(cands: List[Dict], marge: float = 1.0) -> Optional[Dict]:
    for c in cands:
        if c["taux"] <= marge:
            return c
    return None

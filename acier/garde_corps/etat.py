# -*- coding: utf-8 -*-
"""L'état d'une étude de garde-corps : des cas (géométrie, charges, sections, remplissage,
fixation, méthode), la VÉRIFICATION complète d'un cas, la bibliothèque des candidats pour
l'élément principal, le fichier JSON. Une seule source : ``st.session_state["gc"]`` à
l'écran est ce dictionnaire.

Un cas :
- ``geometrie`` : système (montants verticaux | lisse ancrée au mur), forme en plan (droit, L,
  U), hauteur H de la main courante au-dessus de la fixation, tronçons (longueur + entraxe
  maximal, ou entraxes saisis), extrémités, longueur e des consoles (système mur) ;
- ``charges`` : catégorie et q_h, ou P direct ; q_v, Q ; remplissage q_r, Q_r ;
- ``montant`` (section du montant), ``lisse`` (section de la lisse haute), ``remplissage``
  (barreaux verticaux, lisses intermédiaires, panneau, aucun), ``fixation`` (platine,
  soudure, ancrage, console), ``methode`` (ELS du bureau | Eurocode, limites, flèche, acier).

``verifier`` rend tout ce que l'écran et la note impriment : les montants un à un avec leur
longueur tributaire, l'élément gouvernant, les taux par critère, la soudure et la platine,
les messages — rien n'est recalculé ailleurs.
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

VERSION = 1
OK, ATTENTION, NOK, REFUS = ME.OK, ME.ATTENTION, ME.NOK, ME.REFUS
REMPLISSAGES = {"barreaux": "barreaux verticaux", "lisses": "lisses intermédiaires", "panneau": "panneau (verre, tôle)", "aucun": "aucun (main courante seule)"}
LIBELLES = {"montant": "Montant", "lisse": "Lisse haute", "barreau": "Barreau", "lisse_inter": "Lisse intermédiaire", "console": "Console",
            "soudure": "Soudure sur la platine", "ancrage": "Boulons / chevilles", "platine": "Platine"}


# ------------------------------------------------------------------ cas et état
def nouveau_cas(nom: Optional[str] = None) -> Dict:
    return {"nom": nom or "Garde-corps 1", "geometrie": GE.geometrie_defaut(), "charges": CH.charges_defaut(),
            "montant": {"section": {"nom": "CHS 48.3x3.2"}},
            "lisse": {"section": {"nom": "CHS 42.4x3.2"}},
            "remplissage": {"type": "barreaux", "section": {"nom": "Plat 30x8"}, "entraxe": 110.0, "L": 900.0,
                            "n_lisses": 2, "entraxe_lisses": 330.0, "section_lisses": {"nom": "CHS 26.9x3.2"}},
            "fixation": FX.fixation_defaut(), "methode": ME.methode_defaut()}


def nouvel_etat() -> Dict:
    return {"version": VERSION, "cas": [nouveau_cas()], "actif": 0, "identification": {"projet": "", "partie": "", "indice": "0"}}


def cas_actif(etat: Dict) -> Dict:
    return etat["cas"][etat["actif"]]


def normaliser_cas(c: Dict) -> Dict:
    """Complète un cas (fichier ancien, clés manquantes) par les défauts."""
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
    """Les efforts de service (N) en tête d'un montant : P₁, P₂ (horizontaux, par direction) et
    N (vertical, compression)."""
    if ch["mode"] == "P":
        P1 = ch["P"] * 1e3
        P2 = ch["P"] * 1e3 if m["coin"] else 0.0
    else:
        P1 = ch["qh"] * m["trib1"]        # kN/m × mm = N
        P2 = ch["qh"] * m["trib2"]
    N = ch["qv"] * (m["trib1"] + m["trib2"])
    return {"P1": P1, "P2": P2, "N": N}


def verifier_montant(sec: Dict, ch: Dict, m: Dict, H: float, meth: Dict, fy: float) -> Dict:
    """La vérification d'un montant : console de hauteur H sous P₁, P₂, N. Un montant d'angle
    d'une section rectangulaire est vérifié dans les deux orientations (P₁ selon h, puis P₂
    selon h) et garde la pire."""
    p = _p_montant(ch, m)
    E = float(meth.get("E") or ME.E_ACIER)
    Iy, Iz = sec["Iy"] * 1e4, sec["Iz"] * 1e4

    def essai(P1, P2):
        c = ME.console(P1, P2, p["N"], H, E, Iy, Iz)
        v = ME.verifier_section(sec, c["M1"], c["M2"], p["N"], c["V1"], c["V2"], c["f"], H, meth, fy)
        v.update({"P1": P1, "P2": P2, "f1": c["f1"], "f2": c["f2"]})
        return v
    v = essai(p["P1"], p["P2"])
    if m["coin"] and abs(sec["Wel_y"] - sec["Wel_z"]) > 1e-9 and p["P2"] > 0:
        v2 = essai(p["P2"], p["P1"])
        if v2["taux_max"] > v["taux_max"]:
            v = v2
            v["orientation_inversee"] = True
    v.update({"nom": m["nom"], "coin": m["coin"], "trib1": m["trib1"], "trib2": m["trib2"], "x": m["x"], "H": H})
    return v


def verifier_poutre(sec: Dict, qh: float, qv: float, Q: float, L: float, meth: Dict, fy: float, comb: str) -> Dict:
    """Une lisse de portée L (mm) : q_h (N/mm) et Q (N) horizontaux → flexion d'axe faible (h de
    la lisse vertical), q_v (N/mm) vertical → flexion d'axe fort ; flèche résultante."""
    E = float(meth.get("E") or ME.E_ACIER)
    Iy, Iz = sec["Iy"] * 1e4, sec["Iz"] * 1e4
    ph = ME.poutre(qh, Q, L, E, Iz, comb)
    pv = ME.poutre(qv, 0.0, L, E, Iy, "somme")
    v = ME.verifier_section(sec, pv["M"], ph["M"], 0.0, pv["V"], ph["V"], math.hypot(ph["f"], pv["f"]), L, meth, fy)
    v.update({"qh": qh, "qv": qv, "Q": Q, "Mh": ph["M"], "Mv": pv["M"], "Vh": ph["V"], "Vv": pv["V"], "fh": ph["f"], "fv": pv["f"],
              "Mq": ph["Mq"], "MQ": ph["MQ"], "comb": comb, "L": L})
    return v


def verifier_barre(sec: Dict, q: float, Q: float, L: float, meth: Dict, fy: float) -> Dict:
    """Un élément du remplissage (barreau, lisse intermédiaire) de portée L sous la pression du
    remplissage (q, N/mm) OU la charge ponctuelle Q_r (N) à mi-portée — deux cas, le pire."""
    E = float(meth.get("E") or ME.E_ACIER)
    Iy = sec["Iy"] * 1e4
    pq = ME.poutre(q, 0.0, L, E, Iy, "somme")
    pQ = ME.poutre(0.0, Q, L, E, Iy, "somme")
    vq = ME.verifier_section(sec, pq["M"], 0.0, 0.0, pq["V"], 0.0, pq["f"], L, meth, fy)
    vQ = ME.verifier_section(sec, pQ["M"], 0.0, 0.0, pQ["V"], 0.0, pQ["f"], L, meth, fy)
    pire_q = vq["taux_max"] >= vQ["taux_max"]
    v = dict(vq if pire_q else vQ)
    v.update({"cas": "q" if pire_q else "Q", "q": q, "Q": Q, "L": L, "verif_q": vq, "verif_Q": vQ})
    return v


# ------------------------------------------------------------------ vérification complète
def _section(spec: Dict, quoi: str, messages: List[str]) -> Optional[Dict]:
    try:
        return BI.section(spec)
    except ValueError as e:
        messages.append(f"Refus : section {quoi} — {e}")
        return None


def verifier(cas: Dict) -> Dict:
    cas = normaliser_cas(cas)
    geom, ch, meth = cas["geometrie"], cas["charges"], cas["methode"]
    ac = acier(meth)
    fy = ac["fy"]
    lim = ME.limites(meth, fy)
    ana = GE.analyser(geom)
    H = float(geom.get("H") or GE.H_DEFAUT)
    systeme = geom.get("systeme", "montants")
    out: Dict = {"statut": REFUS, "taux_max": None, "gouvernant": None, "elements": [], "messages": list(ana["messages"]),
                 "charges": ch, "geometrie": {**ana, "H": H, "systeme": systeme, "phrase": GE.phrase(geom, ana)}, "acier": ac, "methode": meth,
                 "limites": lim, "montants": [], "montant": None, "lisse": None, "remplissage": None, "consoles": [], "console": None,
                 "fixation": None, "masse": 0.0}
    if ana["messages"]:
        return out
    if H > CH.H_MAX_APPLICATION:
        out["messages"].append(f"La charge linéaire s'applique au plus à {CH.H_MAX_APPLICATION / 1000:.2f} m (EN 1991-1-1 §6.4) : "
                               f"H = {H / 1000:.2f} m est pris comme bras de levier, du côté sûr")
    comb = meth.get("comb", "somme")
    s_max = ana["s_max"]
    elements = []
    masse = 0.0
    sec_lisse = _section(cas["lisse"]["section"], "de la lisse haute", out["messages"])
    qh_lisse = (ch["P"] * 1e3 / s_max if ch["mode"] == "P" else ch["qh"]) if s_max > 0 else 0.0     # N/mm
    if systeme == "montants":
        sec = _section(cas["montant"]["section"], "du montant", out["messages"])
        if sec is None or sec_lisse is None:
            return out
        out["section_montant"] = sec
        for m in ana["montants"]:
            v = verifier_montant(sec, ch, m, H, meth, fy)
            out["montants"].append(v)
        gouv = max(out["montants"], key=lambda v: v["taux_max"])
        out["montant"] = {"section": sec, "verif": gouv, "nom": gouv["nom"], "n": len(out["montants"])}
        elements.append(("montant", f"Montant {gouv['nom']} · {sec['nom']}", gouv["taux_max"], gouv["critere"], gouv))
        masse += len(out["montants"]) * BI.masse_lineaire(sec, H)
        # la lisse haute entre montants
        vl = verifier_poutre(sec_lisse, qh_lisse, ch["qv"], ch["Q"] * 1e3, s_max, meth, fy, comb)
        out["lisse"] = {"section": sec_lisse, "verif": vl, "L": s_max, "role": "secondaire"}
        elements.append(("lisse", f"Lisse haute · {sec_lisse['nom']} · portée {s_max / 1000:.2f} m".replace(".", ","), vl["taux_max"], vl["critere"], vl))
        masse += BI.masse_lineaire(sec_lisse, ana["L_total"])
    else:
        if sec_lisse is None:
            return out
        # système mur : la lisse est l'élément principal, les consoles la portent
        vl = verifier_poutre(sec_lisse, qh_lisse, ch["qv"], ch["Q"] * 1e3, s_max, meth, fy, comb)
        out["lisse"] = {"section": sec_lisse, "verif": vl, "L": s_max, "role": "principal"}
        out["section_montant"] = sec_lisse
        elements.append(("lisse", f"Lisse · {sec_lisse['nom']} · portée {s_max / 1000:.2f} m".replace(".", ","), vl["taux_max"], vl["critere"], vl))
        masse += BI.masse_lineaire(sec_lisse, ana["L_total"])
        sec_c = _section(cas["fixation"]["console"]["section"], "de la console", out["messages"])
        if sec_c is None:
            return out
        e = float(cas["fixation"]["console"].get("e") or 100.0)
        E = float(meth.get("E") or ME.E_ACIER)
        for m in ana["montants"]:
            Rv = ch["qv"] * (m["trib1"] + m["trib2"])
            Rh1 = (ch["P"] * 1e3 if ch["mode"] == "P" else ch["qh"] * m["trib1"])
            Rh2 = (ch["P"] * 1e3 if ch["mode"] == "P" and m["coin"] else ch["qh"] * m["trib2"])
            c = ME.console(Rv, Rh2, Rh1, e, E, sec_c["Iy"] * 1e4, sec_c["Iz"] * 1e4)
            v = ME.verifier_section(sec_c, c["M1"], c["M2"], Rh1, c["V1"], c["V2"], c["f"], e, meth, fy)
            v.update({"nom": m["nom"].replace("M", "C"), "coin": m["coin"], "trib1": m["trib1"], "trib2": m["trib2"], "x": m["x"],
                      "Rv": Rv, "Rh": Rh1, "Rh2": Rh2, "e": e, "P1": Rv, "P2": Rh2, "H": e})
            out["consoles"].append(v)
        gc = max(out["consoles"], key=lambda v: v["taux_max"])
        out["console"] = {"section": sec_c, "verif": gc, "nom": gc["nom"], "e": e, "n": len(out["consoles"])}
        elements.append(("console", f"Console {gc['nom']} · {sec_c['nom']} · e = {e:g} mm", gc["taux_max"], gc["critere"], gc))
        masse += len(out["consoles"]) * BI.masse_lineaire(sec_c, e)
    for r in ana["reprises_mur"]:
        out["messages"].append(f"Ancrage de la lisse dans le mur ({r['ou']}) : il reprend H = {_kn(qh_lisse * r['trib']):.2f} kN "
                               f"et V = {_kn(ch['qv'] * r['trib']):.2f} kN (service) — à justifier avec le fabricant de la cheville")
    # le remplissage
    rp = cas["remplissage"]
    typ = rp.get("type", "barreaux")
    out["remplissage"] = {"type": typ, "libelle": REMPLISSAGES[typ]}
    if typ == "barreaux":
        sec_b = _section(rp["section"], "des barreaux", out["messages"])
        if sec_b is not None:
            eb, Lb = float(rp.get("entraxe") or 110.0), float(rp.get("L") or 900.0)
            vb = verifier_barre(sec_b, ch["qr"] * eb / 1e3, ch["Qr"] * 1e3, Lb, meth, fy)
            out["remplissage"].update({"section": sec_b, "verif": vb, "entraxe": eb, "L": Lb})
            elements.append(("barreau", f"Barreaux · {sec_b['nom']} · e = {eb:g} mm, L = {Lb:g} mm", vb["taux_max"], vb["critere"], vb))
            masse += (ana["L_total"] / eb) * BI.masse_lineaire(sec_b, Lb)
    elif typ == "lisses":
        sec_l = _section(rp.get("section_lisses") or {"nom": "CHS 26.9x3.2"}, "des lisses intermédiaires", out["messages"])
        if sec_l is not None:
            el, nl = float(rp.get("entraxe_lisses") or 300.0), int(rp.get("n_lisses") or 1)
            vb = verifier_barre(sec_l, ch["qr"] * el / 1e3, ch["Qr"] * 1e3, s_max, meth, fy)
            out["remplissage"].update({"section": sec_l, "verif": vb, "entraxe": el, "L": s_max, "n": nl})
            elements.append(("lisse_inter", f"Lisses intermédiaires ({nl}) · {sec_l['nom']} · e = {el:g} mm, portée {s_max / 1000:.2f} m".replace(".", ","),
                             vb["taux_max"], vb["critere"], vb))
            masse += nl * BI.masse_lineaire(sec_l, ana["L_total"])
    elif typ == "panneau":
        out["messages"].append(f"Remplissage en panneau : sa résistance sous q_r = {ch['qr']:g} kN/m² et Q_r = {ch['Qr']:g} kN est à justifier par le fournisseur")
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


def _fixation(cas: Dict, out: Dict, meth: Dict, fy: float, ac: Dict) -> Dict:
    """La soudure et la platine de l'élément fixé le plus chargé (montant gouvernant, ou console
    gouvernante dans le système mur)."""
    fx = cas["fixation"]
    typ = fx.get("type", "platine")
    res: Dict = {"type": typ, "libelle": FX.TYPES[typ], "soudure": None, "platine": None, "taux_max": None, "statut": None}
    if typ == "scellee":
        out["messages"].append("Montant scellé : la fixation n'est pas vérifiée (scellement à justifier séparément)")
        return res
    systeme = out["geometrie"]["systeme"]
    if systeme == "montants":
        if not out.get("montant"):
            return res
        sec, v = out["montant"]["section"], out["montant"]["verif"]
        P1, P2, N_trac, L = v["P1"], v["P2"], -v["N"], v["H"]      # N : compression du montant → N < 0 pour la soudure (traction > 0)
        M = math.hypot(v["M1"], v["M2"])
        hp = float(sec["h"] or sec["d"])
    else:
        if not out.get("console"):
            return res
        sec, v = out["console"]["section"], out["console"]["verif"]
        P1, P2, N_trac, L = v["Rv"], v["Rh2"], v["Rh"], v["e"]      # la console est tirée hors du mur par l'effort horizontal
        M = math.hypot(v["M1"], v["M2"])
        hp = float(sec["h"] or sec["d"])
    soud = fx["soudure"]
    tp = float(fx["platine"].get("tp") or 10.0)
    g = out["limites"]["gamma"]
    cas_s = FX.cas_soudure(sec, _kn(N_trac) * g, _kn(P1) * g, _kn(P2) * g, L, soud, tp, meth, ac["nuance"])
    res["soudure"] = FX.verifier_soudure(cas_s)
    res["soudure"]["efforts_txt"] = (f"N = {_kn(N_trac) * g:.2f} kN, V = {_kn(math.hypot(P1, P2)) * g:.2f} kN, M = {M * g / 1e6:.3f} kN·m"
                                     + (" (efforts de calcul)" if g != 1.0 else " (efforts de service)"))
    if typ == "platine":
        N_comp = -N_trac
        res["platine"] = FX.platine_boulonnee(M, N_comp, math.hypot(P1, P2), hp, fx["platine"], fx["ancrage"], float(soud.get("a") or 0.0), meth, fy,
                                             soud.get("jonction", "angle"))
    taux = [t for t in (res["soudure"]["taux"], (res["platine"] or {}).get("taux_max")) if t is not None]
    res["taux_max"] = max(taux) if taux else None
    res["statut"] = ME.statut(res["taux_max"]) if taux else REFUS
    return res


# ------------------------------------------------------------------ bibliothèque : les candidats
def element_principal(cas: Dict) -> str:
    """La clé de l'élément que la bibliothèque dimensionne : le montant, ou la lisse (système mur)."""
    return "lisse" if normaliser_cas(cas)["geometrie"].get("systeme") == "mur" else "montant"


def candidats(cas: Dict, groupes: Optional[List[str]] = None, masse_max: Optional[float] = 25.0, dim_max: Optional[float] = 150.0) -> List[Dict]:
    """Chaque section standard des groupes demandés, vérifiée comme élément principal du cas
    (montant gouvernant, ou lisse) : [{section, taux, critere, statut, sigma, fleche}], par
    masse croissante. ``plus_leger`` en tire le premier qui tient."""
    cas = normaliser_cas(cas)
    res = verifier(cas)
    out = []
    if res["statut"] == REFUS and not res.get("geometrie", {}).get("montants"):
        return out
    meth, ch = cas["methode"], cas["charges"]
    fy = acier(meth)["fy"]
    ana = res["geometrie"]
    H = ana["H"]
    cle = element_principal(cas)
    for sec in BI.catalogue(groupes, masse_max, dim_max):
        if cle == "montant":
            vs = [verifier_montant(sec, ch, m, H, meth, fy) for m in ana["montants"]]
            v = max(vs, key=lambda x: x["taux_max"])
        else:
            qh = (ch["P"] * 1e3 / ana["s_max"] if ch["mode"] == "P" else ch["qh"]) if ana["s_max"] > 0 else 0.0
            v = verifier_poutre(sec, qh, ch["qv"], ch["Q"] * 1e3, ana["s_max"], meth, fy, meth.get("comb", "somme"))
        out.append({"section": sec, "taux": v["taux_max"], "critere": v["critere"], "statut": v["statut"], "taux_sigma": v["taux"]["sigma"],
                    "taux_tau": v["taux"]["tau"], "taux_fleche": v["taux"]["fleche"], "masse": sec["masse"]})
    return out


def plus_leger(cands: List[Dict], marge: float = 1.0) -> Optional[Dict]:
    """Le candidat le plus léger dont le taux ne dépasse pas ``marge``."""
    for c in cands:
        if c["taux"] <= marge:
            return c
    return None


def appliquer_section(cas: Dict, sec: Dict) -> None:
    """Enregistre la section dans l'élément principal du cas."""
    cas[element_principal(cas)]["section"] = BI.spec_de(sec)

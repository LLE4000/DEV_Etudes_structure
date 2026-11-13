# -*- coding: utf-8 -*-
"""La CORRESPONDANCE entre une poutre du site (``beton/poutre``, normalisée) et les valeurs de l'écran Streamlit
(``modules/poutre.py`` : les clés ``b<i>_…`` et ``b<i>_sec<j>_…`` de ``st.session_state``, et son fichier
« poutre_ba.json » = {"version", "beams", "values"}), dans les deux sens :

- ``valeurs(poutre)`` : ce que l'écran Streamlit aurait en mémoire — la note PDF du site (api_web/note_poutre.py) le
  donne au générateur validé de Streamlit ;
- ``depuis_streamlit(payload)`` : un fichier « poutre_ba.json » de l'écran Streamlit devient une note du site (il
  s'ouvre comme un .stab : l'accueil ou « Ouvrir » le reconnaît, js/notes.js). Chaque poutre et chaque section, les
  efforts, les lits (et leurs positions imposées), les centres de gravité imposés, les lignes d'effort tranchant (et
  les barres qu'elles entourent), le pas, les réglages (γs, jeux, armatures de peau) passent.

Les défauts d'une clé absente sont ceux que la note PDF de Streamlit lit (``modules/export_pdf._g`` et ses appelants) :
ce que Streamlit calculait pour ce fichier, le site le calcule (tests/test_note_poutre.py, aller-retour).
"""
import math
from typing import Dict, List, Optional, Tuple

from beton import position as P
from beton.poutre import entree as EN
from beton.poutre import geometrie as G
from beton.poutre import moteur as MO

TYPES_LIGNE = {"etrier": "Étrier (2 brins)", "epingle": "Épingle (1 brin)"}   # les libellés du module (_brins_from_type)
MAX_POUTRES = 50
MAX_SECTIONS = 50


def _ks(base: str, bid: int, sid: int) -> str:
    return f"b{bid}_sec{sid}_{base}"


def valeurs(p: Dict, bid: int = 1) -> Tuple[List[Dict], Dict]:
    """(beams, values) de l'écran Streamlit pour UNE poutre NORMALISÉE du site (``bid`` : son numéro dans la note).

    La position de CHAQUE lit passe par les surcharges de l'écran (enrob_calc_*_override, dist_*_l*_override) avec l'axe
    de calcul du moteur du site (règle du 13/11/2026, beton/position.py) ; les centres de gravité imposés par ycdg_* ; les barres entourées
    d'un cadre partiel par shear_line*_from / _to (celles que le moteur a retenues)."""
    calc = MO.verifier(p)
    v = {"gamma_s": p["gamma_s"], "jeu_enrobage_cm": p["jeu_premier_lit"], "jeu_entre_lits_cm": p["jeu_entre_lits"],
         "techno_d_mm": p.get("peau_d") or EN.DEFAUTS_DETAIL["peau_d"],
         "techno_s_max_cm": p.get("peau_smax") or EN.DEFAUTS_DETAIL["peau_smax"],
         f"meta_beam_nom_{bid}": p.get("nom") or f"Poutre {bid}",
         f"b{bid}_beton": p["beton"], f"b{bid}_fyk": p["fyk"], f"b{bid}_b": p["b"], f"b{bid}_h": p["h"],
         f"b{bid}_enrobage_beton": p["enrobage"]}
    sections = []
    for j, (s, rs) in enumerate(zip(p["sections"], calc["sections"])):
        sid = j + 1
        sections.append({"id": sid, "nom": s["nom"]})
        v[f"meta_b{bid}_nom_{sid}"] = s["nom"]
        for k in ("M_inf", "M_sup", "V"):
            v[_ks(k, bid, sid)] = s[k]
        for face in ("inf", "sup"):
            v[_ks(f"nlits_{face}", bid, sid)] = len(s[face])
            for i, (lit, geo) in enumerate(zip(s[face], rs["geometrie"][face]["lits"]), start=1):
                suf = "" if i == 1 else f"_l{i}"
                v[_ks(f"n_as_{face}{suf}", bid, sid)] = lit["n"]
                v[_ks(f"ø_as_{face}{suf}", bid, sid)] = lit["d"]
                # chaque lit passe par la surcharge de l'écran avec l'axe de CALCUL du moteur du site (la règle de
                # position du 13/11/2026, beton/position.py : jeu de pose et arrondi) — le générateur ne refait plus
                # la position d'un lit par sa règle d'avant (_auto_dist_lit)
                cle = f"enrob_calc_{face}" if i == 1 else f"dist_{face}_l{i}"
                v[_ks(f"{cle}_override", bid, sid)] = True
                v[_ks(cle, bid, sid)] = geo["distance"]
            yG = s[f"yG_{face}"]
            v[_ks(f"ycdg_auto_{face}", bid, sid)] = yG is None
            v[_ks(f"ycdg_{face}", bid, sid)] = "" if yG is None else repr(float(yG))
        v[_ks("shear_n_lines", bid, sid)] = len(s["etriers"])
        for i, (e, cd) in enumerate(zip(s["etriers"], rs["geometrie"]["etriers"]["cadres"])):
            v[_ks(f"shear_line{i}_type", bid, sid)] = TYPES_LIGNE[e["type"]]
            v[_ks(f"shear_line{i}_d", bid, sid)] = e["d"]
            if e["type"] == "epingle":
                v[_ks(f"shear_line{i}_from", bid, sid)] = cd["de"]
                v[_ks(f"shear_line{i}_to", bid, sid)] = cd["de"]
            elif not cd["complet"]:
                v[_ks(f"shear_line{i}_from", bid, sid)] = cd["de"]
                v[_ks(f"shear_line{i}_to", bid, sid)] = cd["a"]
        v[_ks("shear_pas", bid, sid)] = s["pas"]
    return [{"id": bid, "nom": p.get("nom") or f"Poutre {bid}", "sections": sections}], v


# ------------------------------------------------------------------ un fichier de l'écran Streamlit → une note du site
def _nombre(v, defaut: float) -> float:
    try:
        x = float(str(v).replace(",", ".")) if v is not None and v != "" else defaut
    except (TypeError, ValueError):
        return defaut
    return x if math.isfinite(x) else defaut


def _entier(v, defaut: int) -> int:
    return int(round(_nombre(v, defaut)))


def _brins(t) -> int:
    """_brins_from_type du module : « 3 brins » → 3 ; « pingle » ou « 1 brin » → 1 ; sinon 2."""
    t = str(t)
    if "3 brins" in t:
        return 3
    if "pingle" in t or "1 brin" in t:
        return 1
    return 2


def est_streamlit(d) -> bool:
    """Un fichier « poutre_ba.json » de l'écran Streamlit : {"beams": [...], "values": {"b1_…": …}}."""
    return (isinstance(d, dict) and isinstance(d.get("beams"), list) and isinstance(d.get("values"), dict)
            and any(isinstance(k, str) and k.startswith("b") and "_" in k for k in d["values"]))


def depuis_streamlit(payload: Dict) -> Dict:
    """L'état d'une note du site (``{"version", "poutres", "actif"}``) lu dans un fichier de l'écran Streamlit. Lève
    ValueError (message en français) si le fichier n'en est pas un. Les poutres sont relues par le moteur
    (``EN.normaliser``) : une valeur hors domaine lève, comme une saisie."""
    if not est_streamlit(payload):
        raise ValueError("ce fichier n'est pas un enregistrement de l'écran Streamlit « Poutre en béton armé »")
    v = payload["values"]
    g = lambda k, d=None: v.get(k, d)  # noqa: E731
    # (un fichier de l'écran Streamlit est une note d'AVANT la règle de position du 13/11/2026 : son jeu de pose relu par
    #  elle, 1 cm au moins — comme une note d'avant du site, api_web/poutre._jeu_du_bureau)
    reglages = {"gamma_s": _nombre(g("gamma_s"), 1.5) or 1.5,
                "jeu_premier_lit": P.jeu_note_d_avant(_nombre(g("jeu_enrobage_cm"), 0.0)),
                "jeu_entre_lits": _nombre(g("jeu_entre_lits_cm"), 1.0), "peau_d": _entier(g("techno_d_mm"), 10),
                "peau_smax": _nombre(g("techno_s_max_cm"), 30.0)}
    poutres = []
    for b in payload["beams"][:MAX_POUTRES]:
        try:
            bid = int(b.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        p = {"nom": str(g(f"meta_beam_nom_{bid}", b.get("nom") or f"Poutre {bid}"))[:200],
             "beton": str(g(f"b{bid}_beton", "C30/37")), "fyk": _entier(g(f"b{bid}_fyk"), 500),
             "b": _nombre(g(f"b{bid}_b"), 20.0), "h": _nombre(g(f"b{bid}_h"), 40.0),
             "enrobage": _nombre(g(f"b{bid}_enrobage_beton"), 3.0), **reglages, "sections": []}
        if p["fyk"] not in EN.NUANCES:
            p["fyk"] = 500                                      # _get_fyk du module
        for s in (b.get("sections") or [{"id": 1, "nom": "A"}])[:MAX_SECTIONS]:
            try:
                sid = int(s.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            ks = lambda base: f"b{bid}_sec{sid}_{base}"  # noqa: E731
            sec = {"nom": str(g(f"meta_b{bid}_nom_{sid}", s.get("nom") or "")).strip()[:40],
                   "M_inf": _nombre(g(ks("M_inf")), 0.0), "M_sup": _nombre(g(ks("M_sup")), 0.0), "V": _nombre(g(ks("V")), 0.0),
                   "pas": _nombre(g(ks("shear_pas")), 30.0) or 30.0}
            if sec["nom"].lower().startswith("section "):
                sec["nom"] = sec["nom"][8:].strip()
            # les lignes d'effort tranchant : un cadre à 3 brins devient un étrier ET une épingle (même Ø, même A_st)
            etriers = []
            for i in range(max(1, _entier(g(ks("shear_n_lines")), 1))):
                t = g(ks(f"shear_line{i}_type"), "Étriers (2 brins)")
                d = _entier(g(ks(f"shear_line{i}_d")), 8)
                frm = _entier(g(ks(f"shear_line{i}_from")), 0) or None
                to = _entier(g(ks(f"shear_line{i}_to")), 0) or None
                for _ in range(max(1, _entier(g(ks(f"shear_line{i}_n")), 1))):
                    n_brins = _brins(t)
                    if n_brins in (2, 3):
                        e = {"type": "etrier", "d": d}
                        if frm:
                            e["de"] = frm
                        if to:
                            e["a"] = to
                        etriers.append(e)
                    if n_brins in (1, 3):
                        e = {"type": "epingle", "d": d}
                        if frm and n_brins == 1:
                            e["de"] = frm
                        etriers.append(e)
            sec["etriers"] = etriers
            # les lits, et leurs positions imposées. Une surcharge de l'écran est l'axe de CALCUL du lit (distance absolue
            # parement → axe, celle que le calcul utilise) ; le site saisit une position RÉELLE, que le calcul augmente du
            # jeu de pose et arrondit (beton/position.py). L'axe automatique du site retrouvé : le lit reste automatique
            # (l'aller-retour d'une note du site, api_web/note_poutre.py) ; un autre axe A : la position réelle A − jeu,
            # que le calcul ramène à A (A sur la grille de l'arrondi ; sinon au pas supérieur : d diminue)
            enrob = G.enrobage_de_calcul(p["enrobage"])
            d_et = G.diametre_etrier(etriers)
            arrondi = EN.DEFAUTS_POUTRE["arrondi"]
            for face in ("inf", "sup"):
                lits = []
                for i in range(1, max(1, min(EN.MAX_LITS, _entier(g(ks(f"nlits_{face}")), 1))) + 1):
                    suf = "" if i == 1 else f"_l{i}"
                    lits.append({"n": _entier(g(ks(f"n_as_{face}{suf}")), 2), "d": _entier(g(ks(f"ø_as_{face}{suf}")), 16)})
                for i, lit in enumerate(lits, start=1):
                    cle = f"enrob_calc_{face}" if i == 1 else f"dist_{face}_l{i}"
                    if bool(g(ks(f"{cle}_override"), False)):
                        absolu = _nombre(g(ks(cle)), float("nan"))
                        auto = G.distances_lits(lits[:i], enrob, d_et, p["jeu_premier_lit"], p["jeu_entre_lits"], arrondi)[-1]
                        if math.isfinite(absolu) and abs(absolu - auto) > 1e-9:
                            reel = absolu - p["jeu_premier_lit"]
                            avant = G.positions_lits(lits[:i - 1], enrob, d_et, p["jeu_entre_lits"])[-1] if i > 1 else 0.0
                            lit["pos"] = reel - avant
                sec[face] = lits
                yG = None
                if not bool(g(ks(f"ycdg_auto_{face}"), False)):
                    yG = _nombre(g(ks(f"ycdg_{face}")), 0.0) or None
                    if yG is not None and yG <= 0:
                        yG = None
                sec[f"yG_{face}"] = yG
            p["sections"].append(sec)
        if not p["sections"]:
            p["sections"] = [EN.nouvelle_section("A")]
        poutres.append(EN.normaliser(p))
    if not poutres:
        raise ValueError("ce fichier Streamlit ne contient aucune poutre lisible")
    return {"version": 1, "poutres": poutres, "actif": 0}

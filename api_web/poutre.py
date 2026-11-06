# -*- coding: utf-8 -*-
"""Le module « Poutres béton armé » du site : l'état d'une note (une ou plusieurs poutres, chacune avec ses sections),
les champs de saisie, le résultat section par section, la grande fenêtre de calcul et le sélecteur d'armatures —
rendus en fragments HTML.

Le calcul est celui du moteur ``beton/poutre`` (extrait de l'écran Streamlit modules/poutre.py, dont il reproduit les
résultats — tests/test_poutre_moteur.py). Ce fichier MET EN FORME : il ne calcule rien.
"""
import copy
from typing import Dict, List, Optional, Sequence

from beton.poutre import entree as EN
from beton.poutre import moteur as MO

from . import commun as C
from . import dessin_poutre as DP

MODULE = "poutre"


# ------------------------------------------------------------------ l'état de la note : des poutres
def nouvel_etat() -> Dict:
    p = EN.nouvelle_poutre()
    p["nom"] = "V1"
    return {"version": 1, "poutres": [p], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran relu par le moteur ; une poutre illisible (fichier abîmé) lève une erreur claire."""
    poutres = etat.get("poutres") if isinstance(etat, dict) else None
    if not poutres:
        return nouvel_etat()
    out = {"version": 1, "poutres": [EN.normaliser(p) for p in poutres]}
    out["actif"] = min(max(0, int(etat.get("actif", 0) or 0)), len(out["poutres"]) - 1)
    return out


def _poutre(etat: Dict) -> Dict:
    return etat["poutres"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les poutres de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    ps = etat["poutres"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(ps) - 1)
    elif action == "ajouter":
        p = EN.nouvelle_poutre()
        p["nom"] = f"V{len(ps) + 1}"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "dupliquer":
        p = copy.deepcopy(ps[etat["actif"]])
        p["nom"] = f"{p.get('nom') or 'Poutre'} (copie)"
        ps.append(p)
        etat["actif"] = len(ps) - 1
    elif action == "supprimer" and len(ps) > 1:
        i = min(max(0, int(index or 0)), len(ps) - 1)
        del ps[i]
        if etat["actif"] > i or etat["actif"] >= len(ps):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
ENTIERS = {"n", "d", "fyk"}


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la poutre ouverte. ``chemin`` : les clés (« sections », 0, « M_inf ») ; les boutons
    d'ajout et de retrait passent par le même chemin, terminé par « + » (ajouter) ou « - » (retirer). Le moteur relit
    la poutre : une valeur hors de son domaine est refusée avec un message, et la poutre reste intacte."""
    p = _poutre(etat)
    avant = copy.deepcopy(p)
    chemin = [int(c) if isinstance(c, str) and c.lstrip("-").isdigit() and c != "-" else c for c in chemin]
    try:
        if chemin[-1] == "+":
            liste = _aller(p, chemin[:-1])
            if chemin[:-1] == ["sections"]:
                liste.append(EN.nouvelle_section(EN.nom_section_suivant(p)))
            elif chemin[-2] in ("inf", "sup"):
                if len(liste) >= EN.MAX_LITS:
                    return f"Au plus {EN.MAX_LITS} lits par face"
                liste.append(dict(EN.LIT_DEFAUT))
            elif chemin[-2] == "etriers":
                liste.append(dict(EN.ETRIER_DEFAUT))
        elif chemin[-1] == "-":
            liste = _aller(p, chemin[:-2])
            if len(liste) <= 1:
                return "Il en faut au moins un"
            del liste[int(chemin[-2])]
        else:
            d = _aller(p, chemin[:-1])
            k = chemin[-1]
            if k in ("nom", "beton", "type"):
                d[k] = str(valeur)
            else:
                x = C.lire_nombre(valeur)
                if x is None:
                    return f"Nombre illisible : « {valeur} »"
                d[k] = int(round(x)) if k in ENTIERS else x
        etat["poutres"][etat["actif"]] = EN.normaliser(p)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        etat["poutres"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return None


def _aller(d, chemin: Sequence):
    for c in chemin:
        d = d[c]
    return d


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur d'armatures : « inf-0 » (face inférieure de la section 0) reçoit la disposition « 3Ø20 » en un lit."""
    face, i = role.split("-")
    n, d = nom.split("Ø")
    p = _poutre(etat)
    p["sections"][int(i)][face] = [{"n": int(n), "d": int(d)}]
    etat["poutres"][etat["actif"]] = EN.normaliser(p)


def _lits(chemin: List, lits: List[Dict], titre: str) -> str:
    out = f'<div class="grp">{titre}</div>'
    for j, lit in enumerate(lits):
        retirer = (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(chemin + [j, "-"])}" data-valeur="1" title="Retirer ce lit">'
                   f'{C.icon("x")}</button>') if len(lits) > 1 else '<span class="moins"></span>'
        out += (f'<div class="lit"><span>Lit {j + 1}</span>{C.num(chemin + [j, "n"], lit["n"], "barres", entier=True, mini=1)}'
                f'{C.choix(chemin + [j, "d"], lit["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES])}{retirer}</div>')
    if len(lits) < EN.MAX_LITS:
        out += f'<button type="button" class="addrow" data-chemin="{C._attr_chemin(chemin + ["+"])}" data-valeur="1">{C.icon("plus")}Ajouter un lit</button>'
    return out


def formulaire(etat: Dict, res: Optional[Dict] = None) -> str:
    p = _poutre(etat)
    L = C.ligne
    s1 = (L("Nom", C.texte(["nom"], p.get("nom"), "ex. V1"), full=True)
          + L("Béton", C.choix(["beton"], p.get("beton"), [(k, k) for k in EN.classes_beton()]), full=True)
          + L("Acier", C.segments(["fyk"], p.get("fyk"), [(n, f"B{n}") for n in EN.NUANCES]), full=True)
          + L("largeur", C.num(["b"], p.get("b"), "cm"), "b")
          + L("hauteur", C.num(["h"], p.get("h"), "cm"), "h")
          + L("enrobage", C.num(["enrobage"], p.get("enrobage"), "cm"), "c", "Distance du parement au nu de l'étrier."))
    cartes = ""
    for i, s in enumerate(p["sections"]):
        retirer = (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(["sections", i, "-"])}" data-valeur="1" title="Retirer la section">'
                   f'{C.icon("x")}</button>') if len(p["sections"]) > 1 else ""
        r_s = res["sections"][i] if res and i < len(res.get("sections", [])) else None
        n_s = ("r" if r_s["statut"] == MO.NOK else C.niveau(r_s["taux_max"])) if r_s else ""
        pct = f'<span class="pct {n_s}">{C.pc(C.pct(r_s["taux_max"]))}{C.NBSP}%</span>' if r_s else ""
        cartes += (f'<div class="secard"><div class="sh2"><span class="indice">{C.esc(s["nom"])}</span>Section {C.esc(s["nom"])}{pct}{retirer}</div>'
                   f'<div class="trio">'
                   f'<label><span>M<sub>inf</sub></span>{C.num(["sections", i, "M_inf"], s["M_inf"], "kN·m")}</label>'
                   f'<label><span>M<sub>sup</sub></span>{C.num(["sections", i, "M_sup"], s["M_sup"], "kN·m")}</label>'
                   f'<label><span>V</span>{C.num(["sections", i, "V"], s["V"], "kN")}</label></div></div>')
    s2 = (f'<div class="secards">{cartes}</div>'
          f'<button type="button" class="addrow" data-chemin="sections/+" data-valeur="1">{C.icon("plus")}Ajouter une section</button>'
          + C.aide("M<sub>inf</sub> tend le bas (travée), M<sub>sup</sub> tend le haut (appui) ; valeurs de l'état limite de service."))
    blocs = ""
    for i, s in enumerate(p["sections"]):
        etr = '<div class="grp">Effort tranchant</div>'
        for j, e in enumerate(s["etriers"]):
            retirer = (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(["sections", i, "etriers", j, "-"])}" data-valeur="1" title="Retirer">'
                       f'{C.icon("x")}</button>') if len(s["etriers"]) > 1 else '<span class="moins"></span>'
            etr += (f'<div class="lit"><span>Ligne {j + 1}</span>'
                    f'{C.choix(["sections", i, "etriers", j, "type"], e["type"], [("etrier", "Étrier"), ("epingle", "Épingle")])}'
                    f'{C.choix(["sections", i, "etriers", j, "d"], e["d"], [(x, f"Ø{x}") for x in EN.DIAMETRES_ETRIERS])}{retirer}</div>')
        etr += (f'<button type="button" class="addrow" data-chemin="{C._attr_chemin(["sections", i, "etriers", "+"])}" data-valeur="1">'
                f'{C.icon("plus")}Ajouter une ligne d\'étriers</button>'
                + L("pas des étriers", C.num(["sections", i, "pas"], s["pas"], "cm"), "e"))
        corps = _lits(["sections", i, "inf"], s["inf"], "Inférieures") + _lits(["sections", i, "sup"], s["sup"], "Supérieures") + etr
        resume = f'{_detail_lits(s["inf"])} / {_detail_lits(s["sup"])} · e {C.nb(s["pas"])}'
        blocs += C.section(3 + i, f"Armatures · section {C.esc(s['nom'])}", resume, corps, i == 0, f"arm{i}")
    s_meth = (L("coefficient acier", C.num(["gamma_s"], p.get("gamma_s"), ""), "γ<sub>s</sub>", "f<sub>yd</sub> = f<sub>yk</sub> / γ<sub>s</sub>")
              + L("jeu au premier lit", C.num(["jeu_premier_lit"], p.get("jeu_premier_lit"), "cm"), "")
              + L("jeu entre lits", C.num(["jeu_entre_lits"], p.get("jeu_entre_lits"), "cm"), ""))
    n = 3 + len(p["sections"])
    return (C.section(1, f"Poutre {C.esc(p.get('nom') or '')}", f"{C.nb(p['b'])} × {C.nb(p['h'])} · {p['beton']} · B{p['fyk']}", s1, True, "pt")
            + C.section(2, "Sections et sollicitations", f"{len(p['sections'])} section(s)", s2, True, "so")
            + blocs
            + C.section(n, "Méthode", f"γ<sub>s</sub> {C.nb(p['gamma_s'])} · jeux {C.nb(p['jeu_premier_lit'])} / {C.nb(p['jeu_entre_lits'])} cm",
                        s_meth, False, "me"))


def _detail_lits(lits: List[Dict]) -> str:
    return " + ".join(f"{lit['n']}Ø{lit['d']}" for lit in lits)


# ------------------------------------------------------------------ le résultat
MOTIFS = {"hauteur_insuffisante": "hauteur insuffisante", "acier_insuffisant": "acier insuffisant",
          "acier_excessif": "acier excessif (A<sub>s</sub> > A<sub>s,max</sub>)", "d_utile_negatif": "hauteur utile négative",
          "barres_inclinees": "barres inclinées nécessaires", "cisaillement_excessif": "cisaillement excessif",
          "pas_trop_grand": "pas trop grand"}
# les noms courts des lignes du résultat (les titres complets du module restent dans le détail du calcul)
COURTS = {"hauteur": "Hauteur", "inf": "Armatures inférieures", "sup": "Armatures supérieures", "tau": "Effort tranchant", "pas": "Étriers"}
BESOINS = {"pas_d_etriers": "pas besoin d'étriers", "etriers": "besoin d'étriers", "barres_inclinees_et_etriers": "barres inclinées et étriers",
           "inacceptable": "inacceptable"}


def _niv(v: Dict) -> str:
    """La couleur d'une vérification : le taux du moteur, et rouge dès que le module la déclare non conforme
    (acier excessif, hauteur utile négative…) même sous 100 %."""
    return "r" if v.get("etat") == MO.NOK else C.niveau(v["taux"])


def _n(x: Optional[float], unite: str = "") -> str:
    """Un RÉSULTAT écrit avec la précision d'usage : mm² entiers, cm au millimètre, contraintes au centième."""
    if x is None:
        return "—"
    nd = {"mm²": 0, "cm": 1, "N/mm²": 2, "kN": 1, "kN·m": 1}.get(unite)
    if nd is None:
        a = abs(x)
        nd = 0 if a >= 1000 else (1 if a >= 10 else 2)
    return C.fr(x, nd)


def _v(x: Optional[float]) -> str:
    """Une DONNÉE d'une formule (effort, dimension, coefficient) écrite sans la dénaturer : 12,96 reste 12,96,
    0,1709 reste 0,1709, 200 reste 200 ; seules les grandeurs calculées longues (f_yd = 333,33…) sont arrondies."""
    if x is None:
        return "—"
    a = abs(x)
    if a >= 100:
        return C.nb(round(x, 1), 1)
    if a >= 10:
        return C.nb(round(x, 2), 2)
    return C.nb(round(x, 4), 4)


def _formule(f: Dict) -> Dict:
    """Une formule du moteur → les morceaux « symbole = expression = valeurs = résultat » du détail."""
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    u = f["unite"]
    r = _n(f["resultat"], u) + (f" {u}" if u else "")
    k = f["cle"]
    if k == "hu_min":
        parts = ["h_u,min", "√(M·10⁶ / (α_b·b·μ)) / 10", f"√({_v(t['M'])} × 10⁶ / ({_v(t['alpha_b'])} × {_v(t['b'])} × {_v(t['mu'])})) / 10"]
    elif k == "h_min":
        parts = ["h_min", "h_u,min + y_G", f"{_v(t['hu_min'])} + {_v(t['yG'])}"]
    elif k == "As_req":
        parts = ["A_s,req", "M·10⁶ / (f_yd·0,9·d)", f"{_v(t['M'])} × 10⁶ / ({_v(t['fyd'])} × 0,9 × {_v(t['d'] * 10)})"]
    elif k == "As_min_ec":
        parts = ["A_s,min,EC", "0,26·f_ctm / f_yk·b·h", f"0,26 × {_v(t['fctm'])} / {_v(t['fyk'])} × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "As_min_plancher":
        parts = ["A_s,min,pl", "0,0013·b·h", f"0,0013 × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "As_min_opposee":
        parts = ["A_s,min,opp", "0,25·A_s,req,opp", f"0,25 × {_n(t['As_req_opp'], 'mm²')}"]
    elif k == "As_min":
        parts = ["A_s,min", "max(A_s,min,EC ; A_s,min,pl ; A_s,min,opp)",
                 f"max({_n(t['As_min_ec'], 'mm²')} ; {_n(t['As_min_plancher'], 'mm²')} ; {_n(t['As_min_opposee'], 'mm²')})"]
    elif k == "besoin":
        parts = ["A_s,nécessaire", "max(A_s,req ; A_s,min)", f"max({_n(t['As_req'], 'mm²')} ; {_n(t['As_min'], 'mm²')})"]
    elif k == "As_max":
        parts = ["A_s,max", "0,04·b·h", f"0,04 × {_v(t['b'])} × {_v(t['h'])}"]
    elif k == "tau":
        parts = ["τ", "V·10³ / (0,75·b·h)", f"{_v(t['V'])} × 10³ / (0,75 × {_v(t['b'])} × {_v(t['h'])})"]
    elif k == "tau_lim":
        parts = ["τ_adm", "k·f_ck,cube / 1,05", f"{C.nb(t['k'])} × {_v(t['fck_cube'])} / 1,05"]
    elif k == "s_th":
        parts = ["s_th", "A_st·f_yd·d / (V·10³)", f"{_v(t['Ast'])} × {_v(t['fyd'])} × {_v(t['d'])} / ({_v(t['V'])} × 10³)"]
    elif k == "s_max":
        parts = ["s_max", "min(0,75·d ; 30 cm)", f"min(0,75 × {_v(t['d'])} ; {_v(t['plafond'])})"]
    elif k == "s_lim":
        parts = ["s_lim", "min(s_th ; s_max)", f"min({_v(t['s_th'])} ; {_v(t['s_max'])})"]
    else:                                                        # pragma: no cover — une formule nouvelle du moteur
        parts = [k, " · ".join(f"{s} = {_n(v)}" for s, v in t.items())]
    if f.get("resultat_tronque") is not None and k in ("s_lim",):
        r += f" (retenu : {_n(f['resultat_tronque'])} cm)"
    return {"parts": parts, "res": r, "niveau": ""}


def _piece(v: Dict, s: Dict) -> str:
    geo = s["geometrie"]
    if v["cle"] == "hauteur":
        return f"{C.nb(geo['b'])} × {C.nb(geo['h'])} cm"
    if v["cle"] in ("inf", "sup"):
        return f"{geo[v['cle']]['libelle']} · {C.fr(geo[v['cle']]['As'], 0)} mm²"
    if v["cle"] == "tau":
        return f"V = {C.nb(s['V'])} kN"
    return f"{geo['etriers']['resume']} · e = {C.nb(s['pas'] if 'pas' in s else geo['etriers']['pas'])} cm"


def _ligne_tableau(v: Dict, s: Dict, gov: bool) -> Dict:
    g = s["grandeurs"]
    k = v["cle"]
    if k == "hauteur":
        f, val, lim = "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h", f"h<sub>min</sub> = {_n(v['valeur'])} cm", f"h = {_n(v['limite'])} cm"
    elif k in ("inf", "sup"):
        f, val, lim = "max(A<sub>s,req</sub> ; A<sub>s,min</sub>) ≤ A<sub>s</sub>", f"{_n(v['valeur'], 'mm²')} mm²", f"A<sub>s</sub> = {_n(v['limite'], 'mm²')} mm²"
    elif k == "tau":
        f = "τ = V / (0,75·b·h) ≤ τ<sub>adm</sub>"
        val, lim = f"τ = {_n(v['valeur'])} N/mm²", f"τ<sub>adm,{C.esc(v.get('bande') or '')}</sub> = {_n(v['limite'])} N/mm²"
    else:
        f, val, lim = "e ≤ min(s<sub>th</sub> ; s<sub>max</sub>)", f"e = {_n(v['valeur'])} cm", f"s<sub>lim</sub> = {_n(g.get('s_lim_tronque') or v['limite'])} cm"
    sous = C.esc(_piece(v, s))
    if k == "tau" and v.get("besoin"):
        sous += " · " + BESOINS.get(v["besoin"], v["besoin"])
    if v.get("motifs"):
        sous += " · " + ", ".join(MOTIFS.get(m, m) for m in v["motifs"])
    return {"nom": C.esc(COURTS.get(k, v["nom"])), "sous": sous, "formule": f, "valeur": val, "limite": lim, "taux": C.pct(v["taux"]), "niveau": _niv(v), "gov": gov}


def _efforts_svg(s: Dict) -> str:
    def m(indice, x):
        return f"M<tspan font-size='70%' dy='4'>{indice}</tspan><tspan dy='-4'> = {C.fr(x, 0)} kN·m</tspan>"
    morceaux = ([m("inf", s["M_inf"])] if s["M_inf"] else []) + ([m("sup", s["M_sup"])] if s["M_sup"] else []) + [f"V = {C.fr(s['V'], 0)} kN"]
    return " · ".join(morceaux)


def _coupe(s: Dict) -> Dict:
    geo = s["geometrie"]
    return {"nom": f"Section {s['nom']}", "b": geo["b"], "h": geo["h"], "c": geo["enrobage_calcul"], "d_et": geo["etriers"]["d_max"],
            "inf": [(lit["n"], lit["d"], lit["distance"]) for lit in geo["inf"]["lits"]],
            "sup": [(lit["n"], lit["d"], lit["distance"]) for lit in geo["sup"]["lits"]],
            "etrier": f"{geo['etriers']['resume']} · e = {C.nb(geo['etriers']['pas'])} cm", "efforts": _efforts_svg(s),
            "tendu": "sup" if s["M_sup"] > s["M_inf"] else "inf"}


def _calculer(p: Dict) -> Dict:
    try:
        return MO.verifier(p)
    except ValueError as e:                                     # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(e)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : sa poutre la plus chargée."""
    vals = [(r["taux_max"], i) for i, r in enumerate(resultats) if "taux_max" in r]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    r = resultats[i]
    niv = "r" if r["statut"] == MO.NOK else C.niveau(t)
    g = r["gouvernant"]
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['poutres'][i].get('nom') or 'Poutre'} · section {g['section']} · {g['nom']}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    items = ""
    for i, (p, r) in enumerate(zip(etat["poutres"], resultats)):
        t = r.get("taux_max")
        niv = ("r" if r.get("statut") == MO.NOK else C.niveau(t)) if t is not None else "x"
        txt = f"{C.pc(C.pct(t))}{C.NBSP}%" if t is not None else "—"
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(etat["poutres"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(p.get("nom") or f"Poutre {i + 1}")} '
                  f'<span>· {C.nb(p["b"])} × {C.nb(p["h"])} · {C.esc(p["beton"])}</span></span><span class="pct {niv}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(etat["poutres"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la poutre ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une poutre</div></div>')


def resultat(etat: Dict) -> Dict:
    p = _poutre(etat)
    r = MO.verifier(p)
    resultats = [r if i == etat["actif"] else _calculer(q) for i, q in enumerate(etat["poutres"])]
    g = r["gouvernant"]
    t = r["taux_max"]
    niv = "r" if r["statut"] == MO.NOK else C.niveau(t)
    sec_g = next(s for s in r["sections"] if s["nom"] == g["section"])
    v_g = next(v for v in sec_g["verifications"] if v["cle"] == g["cle"])
    li_g = _ligne_tableau(v_g, sec_g, True)
    verdict = (f'<div class="verdict">{C.chip(C.pct(t), niv, xl=True)}<div class="gov"><b>Section {C.esc(g["section"])} · {C.esc(COURTS.get(g["cle"], g["nom"]).lower())}</b>'
               f'{li_g["valeur"]} {"≤" if v_g["taux"] <= 1 else "&gt;"} {li_g["limite"]}</div></div>')
    rows = ""
    for i, s in enumerate(r["sections"]):
        n_s = "r" if s["statut"] == MO.NOK else C.niveau(s["taux_max"])
        rows += f'<div class="elgrp">Section {C.esc(s["nom"])}<span class="pct {n_s}">{C.pc(C.pct(s["taux_max"]))}{C.NBSP}%</span></div>'
        for v in s["verifications"]:
            pk = f"{v['cle']}-{i}" if v["cle"] in ("inf", "sup") else ""
            rows += C.ligne_res(COURTS.get(v["cle"], v["nom"]), C.esc(_piece(v, s)), C.pct(v["taux"]), _niv(v), calcul=f"s{i}", pk=pk,
                                gov=(s["nom"] == g["section"] and v["cle"] == g["cle"]))
    messages = "".join(C.message(C.esc(a["message"])) for a in r.get("avertissements", []))
    pied = (f'<div class="pfoot"><div class="mass"><span>Méthode du bureau · moteur poutre</span><b>{C.esc(MO.REFERENCE.split(" (")[0].split()[-1])}</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    m = r["materiaux"]
    hyps = [f"Béton <b>{C.esc(m['beton'])}</b> · f<sub>ck,cube</sub> = {C.nb(m['fck_cube'])} N/mm²",
            f"Acier <b>B{int(m['fyk'])}</b> · f<sub>yd</sub> = f<sub>yk</sub> / γ<sub>s</sub> = {int(m['fyk'])} / {C.nb(m['gamma_s'])} = {C.fr(m['fyd'], 1)} N/mm²",
            f"α<sub>b</sub> = {C.nb(m['alpha_b'])} · μ = {C.nb(m['mu'], 4)}",
            f"b × h = {C.nb(p['b'])} × {C.nb(p['h'])} cm · enrobage {C.nb(p['enrobage'])} cm"]
    volets = ""
    for i, s in enumerate(r["sections"]):
        t_s = max(v["taux"] for v in s["verifications"])
        lignes = [_ligne_tableau(v, s, v["taux"] == t_s) for v in s["verifications"]]
        detail = "".join(f'<div class="ck-g"><div class="ck-t">{C.esc(v["nom"])}</div>{C.lignes_fx([_formule(f) for f in v["formules"]])}</div>'
                         for v in s["verifications"])
        efforts = " · ".join(x for x in (f"M<sub>inf</sub> = {C.nb(s['M_inf'])} kN·m" if s["M_inf"] else "",
                                         f"M<sub>sup</sub> = {C.nb(s['M_sup'])} kN·m" if s["M_sup"] else "", f"V = {C.nb(s['V'])} kN") if x)
        volets += C.fenetre_calcul(f"s{i}", f"{C.esc(p.get('nom') or 'Poutre')} · section {C.esc(s['nom'])}", f"{efforts} · méthode du bureau",
                                   hyps, lignes, detail)
    coupes = [_coupe(s) for s in r["sections"]]
    vues = [{"cle": f"s{i}", "titre": f"Section {s['nom']}", "court": f"Section {s['nom']}", "svg": DP.svg(f"pba_{i}", [c]),
             "legende": f"{s['geometrie']['inf']['libelle']} · {s['geometrie']['sup']['libelle']}"} for i, (s, c) in enumerate(zip(r["sections"], coupes))]
    if len(coupes) > 1:
        vues.append({"cle": "toutes", "titre": "Toutes les sections", "court": "Toutes", "svg": DP.svg("pba_toutes", coupes), "legende": "même échelle"})
    return {"resultat": f'<div class="pbody">{verdict}<div class="els">{rows}</div>{messages}</div>{pied}', "volets": volets, "vues": vues,
            "phrase": f"{C.nb(p['b'])} × {C.nb(p['h'])} cm · {C.esc(p['beton'])} · B{p['fyk']}",
            "statut_html": C.statut(niv), "resume": resume_note(etat, resultats), "donnees": formulaire(etat, r),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur d'armatures
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les dispositions d'un lit pour une face d'une section, triées par section d'acier, calculées par le moteur."""
    face, i = role.split("-")
    p = _poutre(etat)
    s = p["sections"][int(i)]
    cands = MO.candidats_armatures(p, s["nom"], face)
    i_o = next((k for k, c in enumerate(cands) if c["optimal"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenue"]), i_o)
    rows = [{"nom": c["nom"], "col": C.fr(c["aire"], 0), "taux": C.pct(min(c["taux"], 9.99)), "niveau": "r" if c["etat"] == MO.NOK else C.niveau(c["taux"]),
             "cur": c["retenue"], "opt": c["optimal"],
             "prop": f"{c['nom']} · A<sub>s</sub> = {C.fr(c['aire'], 0)} mm² · un lit à {C.nb(c['distance'], 2)} cm du parement"} for c in cands]
    face_txt = "inférieures" if face == "inf" else "supérieures"
    return C.selecteur(role, f"Armatures {face_txt} · section {C.esc(s['nom'])}", "triées par section d'acier · un clic retient la disposition",
                       "", rows, max(0, min(i_o, i_c) - 4), "mm²")

# -*- coding: utf-8 -*-
"""Les VÉRIFICATIONS RAPIDES du « Générateur SAF — Fondation de grue » (api_web/saf.py) : la colonne de droite, l'en-tête
du plan (la carte, le cas et le k affichés) et le champ dessiné sur le plan.

Dans l'ordre du cahier des charges (§ 3, « Droite ») : la contrainte de contact maximale face à la limite, avec le cas et
le k qui la donnent ; le tableau par k ; le repère σ moyen ; la distance au bord et les contrôles ; les positions, une à
une ; « Proposer » ; l'export. Ce fichier ne calcule rien : il lit le calcul (``saf.calculer`` : le modèle et le résultat
de chaque k, ``saf.fondation_grue.verification``) et le met en mots.
"""
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from saf.fondation_grue import geometrie as G
from saf.fondation_grue import projet as PR
from saf.fondation_grue import proposition as PP
from saf.fondation_grue import verification as V

from . import commun as C
from . import etabli_compact as E
from . import saf_saisie as SS

N = C.NBSP
CARTES = (("sigma", "Contrainte"), ("w", "Tassement"), ("aucune", "Sans carte"))
CLASSES = {"p": "ok", "f": "warn", "r": "nok", "n": "inc", "x": "nok"}


def _kpa(x: float) -> str:
    return C.fr(x, 1)


def _k(k: float) -> str:
    return C.nb(k)


def _reactions(R) -> str:
    return " / ".join(C.nb(v) for v in R)


# ------------------------------------------------------------------ la synthèse du calcul
def synthese(el: Dict, calc: Dict) -> Dict:
    """Ce que le calcul dit d'une fondation :
    {"niveau" (p, f, r ; n : pas de réactions ; x : pas de calcul), "taux" (σ max / limite, ou None), "phrase" (texte),
     "limite" (kPa), "sigma", "k", "cas" (le maximum parmi les k cochés « SCIA » — tous, si aucun n'est coché),
     "r_vue" (le résultat du k affiché), "i_cas" (le rang du cas affiché), "auto" (la vue est celle du cas le plus
     défavorable), "equilibre" (le plus grand écart d'équilibre), "pp" (poids propre seul)}"""
    limite = el["limite"] * V.KG_CM2
    par_k = calc["par_k"]
    if par_k is None:
        return {"niveau": "x", "taux": None, "phrase": calc["refus"] or "Calcul impossible", "limite": limite, "r_vue": None}
    scia = set(PR.k_scia(el))
    retenus = [r for r in par_k if r["k"] in scia] or par_k
    gov = retenus[V.premier_max([r["sigma_max"] for r in retenus])]          # (à égalité, le premier : le même partout)
    pp = not calc["modele"]["cas"]
    taux = gov["sigma_max"] / limite
    vue = el["vue"]
    r_vue = next((r for r in par_k if vue["k"] is not None and abs(r["k"] - vue["k"]) < 1e-9), gov)
    noms = [c["nom"] for c in r_vue["cas"]]
    i_cas = noms.index(vue["cas"]) if vue["cas"] in noms else noms.index(r_vue["cas_max"])
    phrase = (f"poids propre seul · σ {_kpa(gov['sigma_max'])} kPa · réactions à saisir" if pp
              else f"σ {_kpa(gov['sigma_max'])} kPa · cas {gov['cas_max']} · k = {_k(gov['k'])}")
    return {"niveau": "n" if pp else C.niveau(taux), "taux": taux, "phrase": phrase, "limite": limite, "sigma": gov["sigma_max"], "k": gov["k"],
            "cas": gov["cas_max"], "r_vue": r_vue, "i_cas": i_cas, "auto": r_vue is gov and noms[i_cas] == r_vue["cas_max"],
            "equilibre": max(r["ecart_equilibre"] for r in par_k), "pp": pp}


def champ(el: Dict, calc: Dict, s: Dict) -> Optional[Dict]:
    """Le champ dessiné sur le plan (``dessin_saf.plan``) : la contrainte de contact ou le tassement du cas et du k
    affichés ; None sans carte ou sans calcul."""
    r = s.get("r_vue")
    mode = el["vue"]["carte"]
    if r is None or mode == "aucune":
        return None
    sigma, w = r["champs"][s["i_cas"]]
    d = r["dalle"]
    c = r["cas"][s["i_cas"]]
    decolle = (w < 0) & d.on0
    if mode == "sigma":
        return {"mode": "sigma", "dalle": d, "valeurs": sigma, "decolle": decolle, "maxi": c["sigma_max"], "x": c["x_max"], "y": c["y_max"],
                "limite": s["limite"]}
    n = V.premier_argmax(np.where(d.on0, w, -np.inf))
    return {"mode": "w", "dalle": d, "valeurs": np.maximum(w, 0.0), "decolle": decolle, "maxi": c["w_max"],
            "x": d.x0 + (n % d.nnx) * d.a, "y": d.y0 + (n // d.nnx) * d.b, "limite": s["limite"]}


# ------------------------------------------------------------------ l'en-tête du plan : la carte, le cas, le k
def entete(el: Dict, calc: Dict, s: Dict) -> str:
    """L'en-tête du plan : ce qu'on y lit, et les trois choix de la carte — contrainte ou tassement, la position, le k."""
    m = calc["modele"]
    ch = m["chassis"]
    l1 = f'<div class="l1"><b>PLAN</b><span>· {SS.dimensions(el)} · {C.esc(el["dalle"]["beton"])}</span></div>'
    l2 = (f'<div class="l2">{C.esc(ch["nom"] or "châssis saisi")} · {C.fr(ch["a"], 2)} × {C.fr(ch["b"], 2)}{N}m · '
          f'appuis {C.esc(ch["dims"])}</div>')
    segs = "".join(f'<button type="button" class="{"on" if k == el["vue"]["carte"] else ""}" data-agir="1" data-chemin="vue/carte" '
                   f'data-valeur="{k}">{t}</button>' for k, t in CARTES)
    choix = ""
    r = s.get("r_vue")
    if r is not None and el["vue"]["carte"] != "aucune":
        def menu(chemin, valeur, options, titre):
            opts = "".join(f'<option value="{C.esc(v)}"{" selected" if v == valeur else ""}>{C.esc(t)}</option>' for v, t in options)
            return (f'<div class="field select"><select data-agir="1" data-chemin="{chemin}" aria-label="{C.esc(titre)}" title="{C.esc(titre)}">{opts}</select>'
                    f'{C.icon("chevron-down")}</div>')
        if not s["pp"]:
            cas_txt = [("", f"Cas le plus défavorable ({r['cas_max']})")] + [(c["nom"], f"{c['nom']} · {_reactions(c['R'])}") for c in r["cas"]]
            choix += menu("vue/cas", el["vue"]["cas"] if not s["auto"] and el["vue"]["cas"] else "", cas_txt, "La position affichée")
        ks = [("", f"k le plus défavorable ({_k(s['k'])})")] + [(repr(x["k"]), f"k = {_k(x['k'])} MN/m³") for x in el["sol"]["k"]]
        choix += menu("vue/k", repr(el["vue"]["k"]) if el["vue"]["k"] is not None and abs(r["k"] - el["vue"]["k"]) < 1e-9 else "", ks,
                      "La raideur de sol affichée")
    return (f'<div class="et-ent sf-ent"><div class="et-ent-t">{l1}{l2}</div>'
            f'<div class="sf-vue"><div class="seg" role="group" aria-label="Carte affichée sur le plan">{segs}</div>{choix}</div></div>')


# ------------------------------------------------------------------ la colonne des vérifications
def _table_k(el: Dict, calc: Dict, s: Dict) -> str:
    """Le tableau par k : k | σ max | tassement max | surface décollée | case SCIA ; chaque ligne à la couleur de sa
    contrainte face à la limite ; un clic sur un k l'affiche sur le plan."""
    lignes = ""
    par_k = {r["k"]: r for r in calc["par_k"]}
    for i, x in enumerate(el["sol"]["k"]):
        r = par_k[x["k"]]
        niv = C.niveau(r["sigma_max"] / s["limite"])
        vu = s["r_vue"] is r
        lignes += (f'<tr class="{CLASSES[niv]}{" vu" if vu else ""}"><td><button type="button" class="sf-lien" data-agir="1" data-chemin="vue/k" '
                   f'data-valeur="{C.esc(repr(x["k"]))}" title="Afficher ce k sur le plan">{_k(x["k"])}</button></td>'
                   f'<td class="n"><span class="chip {niv}">{_kpa(r["sigma_max"])}</span></td><td class="n">{C.fr(r["w_max"], 1)}</td>'
                   f'<td class="n">{C.fr(r["decolle"], 1)}{N}%</td>'
                   f'<td class="c"><label class="sf-scia" title="Un fichier SAF est exporté pour ce k"><input type="checkbox" data-agir="1" '
                   f'data-chemin="sol/k/{i}/scia"{" checked" if x["scia"] else ""} aria-label="SCIA, k = {_k(x["k"])}"></label></td></tr>')
    return (f'<table class="sf-tk"><thead><tr><th>k<small>MN/m³</small></th><th class="n">σ max<small>kPa</small></th><th class="n">w max<small>mm</small></th>'
            f'<th class="n">décollé</th><th class="c">SCIA</th></tr></thead><tbody>{lignes}</tbody></table>')


def _controles(el: Dict, calc: Dict, s: Dict) -> Tuple[str, str]:
    """(les lignes des contrôles, la classe du bloc)."""
    m = calc["modele"]
    ctl, g = m["controles"], el["grue"]
    rows, pire = "", "ok"

    def ligne(nom, texte, bon, grave=True):
        # (une ligne par contrôle, celle de l'établi — C.ligne_verif : son nom et son état, puis ce qu'il compare ; elle
        # ouvre le détail des contrôles)
        nonlocal rows, pire
        niv = "p" if bon else ("r" if grave else "v")
        if not bon:
            pire = "nok" if grave or pire == "nok" else "warn"
        rows += C.ligne_verif(nom, texte, None, niv, calcul="s1", etat_txt="OK" if bon else ("Non" if grave else "Attention"))
    dedans = all(ctl["plaques_dedans"])
    ligne("Appuis dans la dalle", "" if dedans else "une surface d'appui sort", dedans)
    if dedans:
        ligne("Distance au bord", f"{C.fr(ctl['distance'], 3)}{N}m {'≥' if ctl['distance_ok'] else '&lt;'} {C.fr(g['dmin'], 2)}{N}m",
              ctl["distance_ok"], grave=False)
    if m["plots"]:
        ligne("Plots dans la dalle", "" if all(ctl["plots_dedans"]) else "un plot sort", all(ctl["plots_dedans"]))
        ligne("Plots séparés", "" if ctl["plots_separes"] else "deux plots se chevauchent", ctl["plots_separes"])
        ligne("Appuis sur leur plot", "" if all(ctl["plots_couvrent"]) else "une surface d'appui déborde", all(ctl["plots_couvrent"]))
    if s.get("r_vue") is not None:
        bon = s["equilibre"] <= V.EQUILIBRE
        ligne("Équilibre", f"écart {C.fr(s['equilibre'] * 100, 4)}{N}%", bon)
    return rows, pire


def _positions(el: Dict, calc: Dict, s: Dict) -> str:
    """Les positions, une à une, pour le k affiché : le nombre de cas générés, et la liste REPLIABLE (repliée d'abord :
    seize lignes repousseraient « Proposer » et l'export hors de l'écran) de σ max de chacune ; un clic l'affiche sur le
    plan. L'écran se souvient qu'on l'a dépliée (js/plan.js, data-plan-volet)."""
    r = s["r_vue"]
    out = ""
    for i, c in enumerate(r["cas"]):
        niv = C.niveau(c["sigma_max"] / s["limite"])
        out += (f'<button type="button" class="sf-cas{" vu" if i == s["i_cas"] else ""}" data-agir="1" data-chemin="vue/cas" data-valeur="{C.esc(c["nom"])}" '
                f'title="Afficher cette position sur le plan"><b>{C.esc(c["nom"])}</b><span>{_reactions(c["R"])}</span>'
                f'<span class="chip {niv}">{_kpa(c["sigma_max"])}</span></button>')
    n = len(r["cas"])
    return (f'<details class="sf-pos" data-plan-volet="positions"><summary><b>{n} cas généré{"s" if n > 1 else ""}</b>'
            f'<span>σ max de chaque position · k{N}={N}{_k(r["k"])}</span>{C.icon("chevron-down")}</summary><div class="sf-cas-l">{out}</div>'
            f'<button type="button" class="sf-detail" data-calcul="s2">{C.icon("file-text")}Le tableau des positions</button></details>')


def tableau_proposition(el: Dict, lignes: List[Dict], complet: bool = True) -> str:
    """Le tableau de « Proposer » : h | dimensions | σ max | volume de béton (dalle + plots) ; la ligne au plus petit
    volume est mise en évidence ; un clic sur une ligne l'applique (quand le tableau est gardé dans la note)."""
    if not lignes:
        return ""
    best = PP.meilleure(lignes) if complet else None
    d = el["dalle"]
    rows = ""
    for i, li in enumerate(lignes):
        if li["d"] is None:
            rows += (f'<tr class="non"><td>{C.fr(li["h"], 2)}</td><td colspan="3">ne tient pas jusqu\'à +{C.fr(PP.D_MAX, 0)}{N}m de chaque côté</td></tr>')
            continue
        dims = (f"{C.fr(li['Lx'], 2)} × {C.fr(li['Ly'], 2)}" if li["Lx"] is not None else f"contour +{C.fr(li['d'], 2)}")
        retenu = abs(li["h"] - d["h"]) < 1e-9 and ((li["Lx"] is not None and d["forme"] == "rectangle" and abs(li["Lx"] - d["Lx"]) < 1e-6
                                                    and abs(li["Ly"] - d["Ly"]) < 1e-6)
                                                   or (li["sommets"] is not None and d["forme"] == "contour" and li["sommets"] == d["sommets"]))
        agir = f' data-agir="1" data-chemin="proposition/appliquer" data-valeur="{i}" title="Appliquer cette dalle"' if complet else " disabled"
        rows += (f'<tr class="{"best" if i == best else ""}{" cur" if retenu else ""}"><td><button type="button" class="sf-lien"{agir}>{C.fr(li["h"], 2)}</button></td>'
                 f'<td>{dims}{N}m</td><td class="n">{_kpa(li["sigma_max"])}</td><td class="n">{C.fr(li["volume"], 1)}</td></tr>')
    return (f'<table class="sf-tk sf-tp"><thead><tr><th>h<small>m</small></th><th>dalle</th><th class="n">σ max<small>kPa</small></th>'
            f'<th class="n">béton<small>m³</small></th></tr></thead><tbody>{rows}</tbody></table>')


def _proposer(el: Dict, calc: Dict, s: Dict) -> str:
    """Le bouton « Proposer », sa barre de progression (js/plan.js : une suite de lectures, une épaisseur par requête)
    et son tableau."""
    possible = bool(calc["modele"]["cas"]) and bool(PR.k_scia(el)) and s.get("r_vue") is not None
    suite = {"role": "proposer", "etapes": [f"{h:.3f}" for h in PR.epaisseurs(el)], "fin": "proposition/garder", "sortie": "#sf-prop-t"}
    pourquoi = "" if possible else ' title="Il faut un jeu de réactions complet et un k coché « SCIA »."'
    p = el.get("proposition")
    table = tableau_proposition(el, p["lignes"]) if p else ""
    retirer = (f'<button type="button" class="sf-x" data-agir="1" data-chemin="proposition/-" data-valeur="1" title="Retirer ce tableau" '
               f'aria-label="Retirer ce tableau">{C.icon("x")}</button>') if p else ""
    return (f'<div class="sf-prop" data-suite-hote="1"><div class="sf-prop-h"><button type="button" class="btn sm" data-suite="{C.esc(json.dumps(suite))}"'
            f'{"" if possible else " disabled"}{pourquoi}>{C.icon("lightbulb")}Proposer</button>'
            f'<span class="sf-prop-s">la plus petite dalle sous la limite, par épaisseur</span>{retirer}</div><div id="sf-prop-t">{table}</div></div>')


def _export(el: Dict, calc: Dict) -> str:
    """Le SEUL bouton d'export : les SAF (un par k coché) et la note, en un fichier .zip ; ce qui l'empêche est dit
    dessous, ce qui l'accompagne d'un avertissement aussi."""
    from . import saf as M                                        # (les contrôles de l'export sont ceux du module)
    bloque, avertit = M.controles_export(el, calc)
    n = len(PR.k_scia(el))
    # (éteint, le bouton n'annonce aucun fichier : il ne porte pas « data-telecharger »)
    quoi = ' disabled' if bloque else ' data-telecharger="export" data-avec-note="1"'
    bouton = (f'<button type="button" class="btn primary sf-exp-b"{quoi}>'
              f'{C.icon("download")}Exporter{f" · {n} SAF + note" if not bloque else ""}</button>')
    mots = "".join(C.message(C.esc(SS.phrase(t)), True) for t in bloque) + "".join(C.message(C.esc(SS.phrase(t))) for t in avertit)
    return f'<div class="sf-export">{bouton}{mots}</div>'


def colonne(el: Dict, calc: Dict, s: Dict) -> Tuple[str, str]:
    """(la colonne des vérifications, les fenêtres de calcul)."""
    m = calc["modele"]
    nom = C.esc(el.get("nom") or "Fondation")
    tete = E.tete_resultats(nom, f"{SS.dimensions(el)} · {C.esc(el['dalle']['beton'])}")
    lim = s["limite"]
    ctl_rows, ctl_cls = _controles(el, calc, s)
    if s.get("r_vue") is None:
        verdict = f'<div class="verdict"><div class="gov"><b>Calcul impossible</b>{C.esc(s["phrase"])}</div></div>'
        blocs = E.bloc(1, "Contrôles", ctl_cls, "", "", ctl_rows, theme=True)
        return f'<div class="pbody">{tete}{verdict}<div class="et-blocs">{blocs}</div>{_export(el, calc)}</div>', _fenetre_controles(el, calc, s)
    niv = s["niveau"]
    signe = "≤" if s["taux"] <= 1 else "&gt;"
    suite = "réactions à saisir" if s["pp"] else f"cas {C.esc(s['cas'])} · k{N}={N}{_k(s['k'])}{N}MN/m³"
    verdict = C.verdict_valeur(s["sigma"] / V.KG_CM2, "kg/cm²", "Poids propre seul" if s["pp"] else "Contrainte de contact",
                               f"{_kpa(s['sigma'])}{N}kPa {signe} {_kpa(lim)}{N}kPa · {suite}", "c" if s["pp"] else niv, nd=2, symbole="σ", calcul="s0")
    sm = V.sigma_moyen(m)
    moyen = (f'<div class="sf-moyen" title="Le poids de la dalle, le poids des plots et la somme des réactions du jeu le plus lourd, divisés par l\'aire de la dalle">'
             f'<span>σ moyen</span><b>{_kpa(sm["sigma"])}{N}kPa</b><span class="d">({C.fr(sm["dalle"], 0)}'
             + (f' + {C.fr(sm["plots"], 0)}' if sm["plots"] else "") + (f' + {C.fr(sm["reactions"], 0)}' if sm["reactions"] else "")
             + f') / {C.fr(sm["aire"], 1)}{N}m²</span></div>')
    n_scia = len(PR.k_scia(el))
    cls0 = "inc" if s["pp"] else CLASSES[niv]
    blocs = E.bloc(0, "Raideur du sol", cls0, f"{n_scia} k coché{'s' if n_scia > 1 else ''} SCIA" if n_scia else "aucun k coché SCIA", moyen,
                   _table_k(el, calc, s), theme=True)
    blocs += E.bloc(1, "Contrôles", ctl_cls, "", "", ctl_rows, theme=True)
    positions = "" if s["pp"] else _positions(el, calc, s)
    html = (f'<div class="pbody">{tete}{verdict}<div class="et-blocs">{blocs}</div>{positions}{_proposer(el, calc, s)}{_export(el, calc)}</div>')
    volets = _fenetre_k(el, calc, s) + _fenetre_controles(el, calc, s) + ("" if s["pp"] else _fenetre_positions(el, calc, s))
    return html, volets


# ------------------------------------------------------------------ les fenêtres de calcul
def hypotheses(el: Dict, calc: Dict) -> List[str]:
    """Les hypothèses de la vérification rapide, en phrases courtes (HTML) : celles de l'écran et de la note."""
    m, o = calc["modele"], el["options"]
    D = m["E"] * 1000.0 * m["h"] ** 3 / (12 * (1 - m["nu"] ** 2))
    plots = m["plots"]
    return [f"Plaque mince sur sol élastique · maille <b>{C.fr(m['maille'], 2)}{N}m</b>",
            f"{C.esc(m['beton'])} · E = <b>{C.fr(m['E'], 0)}{N}MPa</b> · ν = {C.nb(m['nu'])} · D = {C.fr(D / 1000.0, 0)}{N}MN·m",
            "Sol en <b>compression seule</b> : les zones qui se soulèvent sont libérées",
            f"C<sub>2x</sub> = k × {C.nb(o['c2x'])} · C<sub>2y</sub> = k × {C.nb(o['c2y'])} (MN/m)",
            f"Charges ELS caractéristiques : poids propre {C.fr(m['gamma'] * m['h'], 2)}{N}kN/m²"
            + (f", plots {C.fr(m['gamma'] * plots['hauteur'], 2)}{N}kN/m²" if plots else "") + ", réactions sur leur surface d'appui"
            + (" (sur tout le plot : préfabriqué)" if plots and plots["type"] == "prefabrique" else ""),
            "Une <b>estimation</b> pour fixer la dalle avant SCIA, qui reste la référence"]


def _fenetre_k(el: Dict, calc: Dict, s: Dict) -> str:
    lignes = []
    scia = set(PR.k_scia(el))
    for r in calc["par_k"]:
        pire = r["cas"][V.premier_max([c["sigma_max"] for c in r["cas"]])]
        t = r["sigma_max"] / s["limite"]
        lignes.append({"nom": f"k = {_k(r['k'])}{N}MN/m³", "sous": "coché SCIA" if r["k"] in scia else "", "formule": f"cas {C.esc(pire['nom'])} · {_reactions(pire['R'])}{N}kN",
                       "valeur": f"{_kpa(r['sigma_max'])}{N}kPa", "limite": f"{_kpa(s['limite'])}{N}kPa", "taux": C.pct(t), "niveau": C.niveau(t),
                       "gov": r["k"] == s["k"]})
    detail = "".join(f'<div class="ck-g"><div class="ck-t">k = {_k(r["k"])}{N}MN/m³</div><div class="fx">tassement max <span class="eq">=</span>'
                     f'<span class="r">{C.fr(r["w_max"], 2)}{N}mm</span></div><div class="fx">surface décollée <span class="eq">=</span>'
                     f'<span class="r">{C.fr(r["decolle"], 1)}{N}%</span></div><div class="fx">itérations <span class="eq">=</span>'
                     f'<span class="r">{max(c["iterations"] for c in r["cas"])}</span></div><div class="fx">écart d\'équilibre <span class="eq">=</span>'
                     f'<span class="r">{C.fr(r["ecart_equilibre"] * 100, 4)}{N}%</span></div></div>' for r in calc["par_k"])
    return C.fenetre_calcul("s0", "Vérification rapide", "la contrainte de contact maximale de chaque raideur de sol", hypotheses(el, calc), lignes, detail,
                            colonnes=("Raideur", "Position la plus défavorable", "σ max", "Limite", "Taux"))


def _fenetre_controles(el: Dict, calc: Dict, s: Dict) -> str:
    m = calc["modele"]
    ctl, g = m["controles"], el["grue"]
    lignes = []
    for i in range(4):
        bon = ctl["plaques_dedans"][i] and ctl["distances"][i] >= g["dmin"] - G.TOLERANCE
        lignes.append({"nom": f"P{i + 1}", "formule": "dans la dalle" if ctl["plaques_dedans"][i] else "hors de la dalle",
                       "valeur": f"{C.fr(ctl['distances'][i], 3)}{N}m" if ctl["plaques_dedans"][i] else "—", "limite": f"{C.fr(g['dmin'], 2)}{N}m",
                       "niveau": "p" if bon else ("r" if not ctl["plaques_dedans"][i] else "f"), "etat_txt": "OK" if bon else "Non"})
    hyps = [f"Surface d'appui : <b>{C.esc(m['chassis']['dims'])}</b> · {C.fr(m['chassis']['aire'], 4)}{N}m²",
            f"Distance mini plaque–bord : <b>{C.fr(g['dmin'], 2)}{N}m</b> — en dessous, l'export reste possible, avec un avertissement",
            "Un bord sur un bord est admis, à 1 mm près"]
    return C.fenetre_calcul("s1", "Contrôles", "les surfaces d'appui et le bord de la dalle", hyps, lignes, "",
                            colonnes=("Pied", "Position", "Distance au bord", "Minimum", "État"))


def _fenetre_positions(el: Dict, calc: Dict, s: Dict) -> str:
    r = s["r_vue"]
    lignes = []
    for c, mc in zip(r["cas"], calc["modele"]["cas"]):
        t = c["sigma_max"] / s["limite"]
        lignes.append({"nom": C.esc(c["nom"]), "sous": f"{C.esc(mc['lc'])} · jeu {mc['jeu_saisi']}", "formule": f"{_reactions(c['R'])}{N}kN",
                       "valeur": f"{_kpa(c['sigma_max'])}{N}kPa", "limite": f"{C.fr(c['w_max'], 2)}{N}mm", "taux": C.pct(t), "niveau": C.niveau(t),
                       "gov": c["nom"] == r["cas_max"]})
    hyps = ["Les charges <b>tournent</b> : chaque jeu est placé dans toutes les positions de la grue (rotations de 90° et miroirs), les doublons retirés",
            f"Raideur affichée : <b>k = {_k(r['k'])}{N}MN/m³</b>"]
    return C.fenetre_calcul("s2", "Positions des charges", "P1 / P2 / P3 / P4 de chaque cas généré", hyps, lignes, "",
                            colonnes=("Cas", "P1 / P2 / P3 / P4", "σ max", "Tassement", "Taux"))

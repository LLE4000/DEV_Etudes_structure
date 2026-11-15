# -*- coding: utf-8 -*-
"""Les RÉSULTATS du « Générateur SAF — Fondation de grue » (api_web/saf.py) : la colonne de droite, la barre du centre
(l'onglet, la carte, le cas et le k affichés) et le champ dessiné sur le plan.

La colonne reprend EXACTEMENT les composants du module « Poutres béton armé » (cahier du lot 4, § 2.2) : le nom de la
fondation et sa ligne grise, puis une CARTE par vérification (E.bloc : un titre, son état à droite — « Conforme »,
« Non conforme », « non vérifié »), une ligne par contrôle (C.ligne_verif : le nom, le taux en %, la formule avec ses
valeurs, une petite barre), la ligne grise « Non vérifiés : … », les cas générés, « Proposer », un seul bouton d'export,
et le pied « Méthode du bureau · moteur poutre » avec la version du moteur. Plus de grand rectangle de contrainte.

La carte du plan est en ENVELOPPE par défaut (§ 2.3) : en chaque point, le maximum de tous les cas et de tous les k cochés
« SCIA » ; un cas ou un k précis se choisit dans les listes, dont « Enveloppe » est la première valeur. Les contraintes de
sol sont écrites dans l'unité choisie (saf_saisie.sol). Ce fichier ne calcule rien : il lit le calcul (``saf.calculer``)
et le met en mots.
"""
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from saf.fondation_grue import geometrie as G
from saf.fondation_grue import projet as PR
from saf.fondation_grue import proposition as PP
from saf.fondation_grue import verification as V

from beton.poutre import moteur as MOTEUR_POUTRE

from . import commun as C
from . import etabli_compact as E
from . import saf_saisie as SS

N = C.NBSP
CARTES = (("sigma", "Contrainte"), ("w", "Tassement"), ("aucune", "Sans carte"))
CLASSES = {"p": "ok", "f": "warn", "r": "nok", "n": "inc", "x": "nok"}
ETATS = {"ok": "Conforme", "warn": "Conforme", "nok": "Non conforme", "inc": "non vérifié"}
ENVELOPPE = "Enveloppe"
# les onglets du centre (§ 2.3) : le plan (la carte), le plan ferraillé et les coupes (lot 4d)
ONGLETS = (("plan", "Plan"),)
# ce que la méthode ne vérifie pas (§ 9) : la ligne grise sous les cartes
NON_VERIFIES = ("la fissuration", "les efforts horizontaux et les moments aux pieds de la grue")
METHODE = "Méthode du bureau · moteur poutre"
VERSION_MOTEUR = MOTEUR_POUTRE.REFERENCE.split(" (")[0].split()[-1]       # « v2.40 », comme le pied du module poutre


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
    phrase = (f"poids propre seul · σ {SS.sol(el, gov['sigma_max'])} · réactions à saisir" if pp
              else f"σ {SS.sol(el, gov['sigma_max'])} · cas {gov['cas_max']} · k = {_k(gov['k'])}")
    return {"niveau": "n" if pp else C.niveau(taux), "taux": taux, "phrase": phrase, "limite": limite, "sigma": gov["sigma_max"], "k": gov["k"],
            "cas": gov["cas_max"], "r_vue": r_vue, "i_cas": i_cas, "auto": r_vue is gov and noms[i_cas] == r_vue["cas_max"],
            "equilibre": max(r["ecart_equilibre"] for r in par_k), "pp": pp}


def selection(el: Dict, calc: Dict) -> Tuple[List[Dict], Optional[str]]:
    """(les résultats des k retenus, le nom du cas retenu ou None) : ce que la carte montre — par défaut l'ENVELOPPE de
    tous les cas et de tous les k cochés « SCIA » (tous les k si aucun n'est coché) ; un k et/ou un cas choisis dans les
    listes la restreignent."""
    par_k = calc["par_k"] or []
    vue = el["vue"]
    if vue["k"] is not None:
        ks = [r for r in par_k if abs(r["k"] - vue["k"]) < 1e-9]
    else:
        scia = set(PR.k_scia(el))
        ks = [r for r in par_k if r["k"] in scia] or list(par_k)
    noms = [c["nom"] for c in (ks[0]["cas"] if ks else [])]
    return ks, (vue["cas"] if vue["cas"] in noms else None)


def enveloppe_champs(ks: List[Dict], cas: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(σ, w) aux nœuds : le maximum, nœud par nœud, des cas et des k retenus (le même maillage pour tous les k)."""
    sig, w = None, None
    for r in ks:
        for c, (s_, w_) in zip(r["cas"], r["champs"]):
            if cas is not None and c["nom"] != cas:
                continue
            sig = s_.copy() if sig is None else np.maximum(sig, s_)
            w = w_.copy() if w is None else np.fmax(w, w_)
    return sig, w


def champ(el: Dict, calc: Dict, s: Dict) -> Optional[Dict]:
    """Le champ dessiné sur le plan (``dessin_saf.plan``) : la contrainte de contact ou le tassement — l'ENVELOPPE des
    cas et des k retenus (``selection``) ; None sans carte ou sans calcul. Un nœud est « décollé » quand il l'est dans
    TOUS les cas retenus."""
    mode = el["vue"]["carte"]
    if s.get("r_vue") is None or mode == "aucune":
        return None
    ks, cas = selection(el, calc)
    if not ks:
        return None
    sigma, w = enveloppe_champs(ks, cas)
    d = ks[0]["dalle"]
    decolle = (sigma <= 0) & d.on0
    if mode == "sigma":
        n = V.premier_argmax(np.where(d.on0, sigma, -np.inf))
        return {"mode": "sigma", "dalle": d, "valeurs": sigma, "decolle": decolle, "maxi": float(sigma[n]),
                "x": d.x0 + (n % d.nnx) * d.a, "y": d.y0 + (n // d.nnx) * d.b, "limite": s["limite"], "unite": SS.unite(el), "symbole": "σ"}
    n = V.premier_argmax(np.where(d.on0, w, -np.inf))
    return {"mode": "w", "dalle": d, "valeurs": np.maximum(w, 0.0), "decolle": decolle, "maxi": float(w[n]),
            "x": d.x0 + (n % d.nnx) * d.a, "y": d.y0 + (n // d.nnx) * d.b, "limite": None, "unite": ("mm", 0.001, 2), "symbole": "w"}


# ------------------------------------------------------------------ l'en-tête du plan : la carte, le cas, le k
def _menu(chemin: str, valeur: str, options, titre: str, classe: str = "") -> str:
    """Une liste déroulante de la barre du centre (data-agir : js/plan.js l'envoie au moteur)."""
    opts = "".join(f'<option value="{C.esc(v)}"{" selected" if v == valeur else ""}>{C.esc(t)}</option>' for v, t in options)
    return (f'<div class="field select{(" " + classe) if classe else ""}"><select data-agir="1" data-chemin="{chemin}" aria-label="{C.esc(titre)}" '
            f'title="{C.esc(titre)}">{opts}</select>{C.icon("chevron-down")}</div>')


def entete(el: Dict, calc: Dict, s: Dict, onglets=ONGLETS) -> str:
    """La barre du centre, sur UNE ligne (§ 2.3) : les onglets (Plan, Ferraillage, Coupes), le choix de la carte, la liste
    « Cas » et la liste « k » (« Enveloppe » en tête). Sur tablette, elle tient sur une ligne (saf.css) ; les libellés
    longs des cartes ont un court (« σ », « w ») qui les remplace quand la place manque."""
    vue = el["vue"]
    tabs = "".join(f'<button type="button" class="{"on" if k == vue["onglet"] else ""}" data-agir="1" data-chemin="vue/onglet" data-valeur="{k}">{t}</button>'
                   for k, t in onglets)
    out = f'<div class="seg sf-onglets" role="tablist" aria-label="Vue du centre">{tabs}</div>'
    if vue["onglet"] == "plan":
        out += _menu("vue/carte", vue["carte"], CARTES, "La carte peinte sur le plan", "sf-m-carte")
        r = s.get("r_vue")
        if r is not None and vue["carte"] != "aucune":
            ks, cas = selection(el, calc)
            if not s["pp"]:
                cas_txt = [("", ENVELOPPE)] + [(c["nom"], f"{c['nom']} · {_reactions(c['R'])}") for c in r["cas"]]
                out += _menu("vue/cas", cas or "", cas_txt, "Le cas affiché : l'enveloppe de tous les cas, ou une position", "sf-m-cas")
            k_txt = [("", ENVELOPPE)] + [(repr(x["k"]), f"k = {_k(x['k'])}") for x in el["sol"]["k"]]
            out += _menu("vue/k", repr(vue["k"]) if vue["k"] is not None else "", k_txt,
                         "La raideur de sol affichée : l'enveloppe des k cochés « SCIA », ou un k", "sf-m-k")
    return f'<div class="et-ent sf-ent sf-barre">{out}</div>'


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
                   f'<td class="n"><span class="chip {niv}">{SS.sol_nb(el, r["sigma_max"])}</span></td><td class="n">{C.fr(r["w_max"], 1)}</td>'
                   f'<td class="n">{C.fr(r["decolle"], 1)}{N}%</td>'
                   f'<td class="c"><label class="sf-scia" title="Coché : ce k entre dans l\'enveloppe ; le SAF porte le premier coché"><input type="checkbox" data-agir="1" '
                   f'data-chemin="sol/k/{i}/scia"{" checked" if x["scia"] else ""} aria-label="SCIA, k = {_k(x["k"])}"></label></td></tr>')
    return (f'<table class="sf-tk"><thead><tr><th>k<small>MN/m³</small></th><th class="n">σ max<small>{SS.unite(el)[0]}</small></th><th class="n">w max<small>mm</small></th>'
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
                f'<span class="chip {niv}">{SS.sol_nb(el, c["sigma_max"])}</span></button>')
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
                 f'<td>{dims}{N}m</td><td class="n">{SS.sol_nb(el, li["sigma_max"])}</td><td class="n">{C.fr(li["volume"], 1)}</td></tr>')
    return (f'<table class="sf-tk sf-tp"><thead><tr><th>h<small>m</small></th><th>dalle</th><th class="n">σ max<small>{SS.unite(el)[0]}</small></th>'
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
    """Le SEUL bouton d'export (lot 4, § 7) : « Exporter · SAF + note » — UN fichier SAF (le premier k coché « SCIA ») et la
    note, en un fichier .zip ; ce qui l'empêche est dit dessous, ce qui l'accompagne d'un avertissement aussi."""
    from . import saf as M                                        # (les contrôles de l'export sont ceux du module)
    bloque, avertit = M.controles_export(el, calc)
    # (éteint, le bouton n'annonce aucun fichier : il ne porte pas « data-telecharger »)
    quoi = ' disabled' if bloque else ' data-telecharger="export" data-avec-note="1"'
    k1 = PR.k_scia(el)[0] if PR.k_scia(el) else None
    titre = f' title="Le SAF est écrit avec k = {_k(k1)} MN/m³ (le premier coché « SCIA ») ; la note rappelle les k à passer dans SCIA"' if k1 is not None else ""
    bouton = (f'<button type="button" class="btn primary sf-exp-b"{quoi}{titre}>'
              f'{C.icon("download")}Exporter · SAF + note</button>')
    mots = "".join(C.message(C.esc(SS.phrase(t)), True) for t in bloque) + "".join(C.message(C.esc(SS.phrase(t))) for t in avertit)
    return f'<div class="sf-export">{bouton}{mots}</div>'


def non_verifies(el: Dict, calc: Dict, extra: Sequence[str] = ()) -> str:
    """La ligne grise sous les cartes (§ 9) : ce que la méthode ne vérifie pas, selon le cas."""
    t = list(extra) + list(NON_VERIFIES)
    return f"Non vérifiés : {', '.join(t)}."


def _carte_sol(el: Dict, calc: Dict, s: Dict) -> str:
    """La carte SOL : la contrainte de contact (taux = σ max / limite), le tassement max, la surface décollée, σ moyen ;
    le tableau par k, dépliable dans la carte."""
    m = calc["modele"]
    lim = s["limite"]
    retenus, _ = selection(dict(el, vue=dict(el["vue"], k=None, cas=None)), calc)
    w = max(retenus, key=lambda r: r["w_max"])
    dec = max(retenus, key=lambda r: r["decolle"])
    if s["pp"]:
        cmp_ = f"σ max = {SS.sol(el, s['sigma'])} · poids propre seul · réactions à saisir"
        ligne = C.ligne_verif("Contrainte de contact", cmp_, None, "c", calcul="s0", etat_txt="non vérifié")
        cls = "inc"
    else:
        signe = "≤" if s["taux"] <= 1 else "&gt;"
        cmp_ = f"σ max = {SS.sol(el, s['sigma'])} {signe} {SS.sol(el, lim)} · cas {C.esc(s['cas'])} · k = {_k(s['k'])}"
        ligne = C.ligne_verif("Contrainte de contact", cmp_, C.pct(s["taux"]), C.niveau(s["taux"]), calcul="s0")
        cls = CLASSES[s["niveau"]]
    sm = V.sigma_moyen(m)
    moyen = (f"σ moyen = ({C.fr(sm['dalle'], 0)}" + (f" + {C.fr(sm['plots'], 0)}" if sm["plots"] else "")
             + (f" + {C.fr(sm['reactions'], 0)}" if sm["reactions"] else "") + f") / {C.fr(sm['aire'], 1)}{N}m² = {SS.sol(el, sm['sigma'])}")
    rows = (ligne
            + C.ligne_verif("Tassement max", f"w max = {C.fr(w['w_max'], 1)}{N}mm · k = {_k(w['k'])}", None, "c", calcul="s0", etat_txt="info")
            + C.ligne_verif("Surface décollée", f"{C.fr(dec['decolle'], 1)}{N}% · k = {_k(dec['k'])}", None, "c", calcul="s0", etat_txt="info")
            + C.ligne_verif("σ moyen", moyen, None, "c", etat_txt="repère"))
    n_scia = len(PR.k_scia(el))
    table = (f'<details class="sf-tkd" data-plan-volet="tablek"><summary><b>Tableau par k</b><span>{n_scia} k coché{"s" if n_scia > 1 else ""} '
             f'« SCIA »</span>{C.icon("chevron-down")}</summary>{_table_k(el, calc, s)}</details>')
    return E.bloc(0, "Sol", cls, ETATS[cls] if not s["pp"] else "non vérifié", "", rows, table, theme=True)


def pied() -> str:
    """Le pied de la colonne, comme le module poutre : « Méthode du bureau · moteur poutre » et la version du moteur."""
    return (f'<div class="pfoot"><div class="mass"><span>{METHODE}</span><b>{C.esc(VERSION_MOTEUR)}</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')


def colonne(el: Dict, calc: Dict, s: Dict, cartes_ferraillage: str = "", non_verifies_extra: Sequence[str] = ()) -> Tuple[str, str]:
    """(la colonne des résultats, les fenêtres de calcul) : SOL, CONTRÔLES (et les cartes du ferraillage, lot 4c), la
    ligne grise « Non vérifiés », les cas générés, « Proposer », l'export, le pied."""
    nom = C.esc(el.get("nom") or "Fondation")
    tete = E.tete_resultats(nom, f"{SS.dimensions(el)} · {C.esc(el['dalle']['beton'])}")
    ctl_rows, ctl_cls = _controles(el, calc, s)
    if s.get("r_vue") is None:
        rows = C.ligne_verif("Calcul", C.esc(s["phrase"]), None, "r", etat_txt="impossible")
        blocs = E.bloc(0, "Sol", "nok", "Calcul impossible", "", rows, theme=True) + E.bloc(1, "Contrôles", ctl_cls, ETATS[ctl_cls], "", ctl_rows, theme=True)
        return (f'<div class="pbody">{tete}<div class="et-blocs">{blocs}</div>{C.aide(non_verifies(el, calc, non_verifies_extra))}'
                f'{_export(el, calc)}</div>{pied()}', _fenetre_controles(el, calc, s))
    blocs = _carte_sol(el, calc, s) + E.bloc(1, "Contrôles", ctl_cls, ETATS[ctl_cls], "", ctl_rows, theme=True) + cartes_ferraillage
    positions = "" if s["pp"] else _positions(el, calc, s)
    html = (f'<div class="pbody">{tete}<div class="et-blocs">{blocs}</div>{C.aide(non_verifies(el, calc, non_verifies_extra))}'
            f'{positions}{_proposer(el, calc, s)}{_export(el, calc)}</div>{pied()}')
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
            + (" (sous un plot préfabriqué : diffusée à 45° dans le plot, sur un carré centré sur le pied)" if plots and plots["type"] == "prefabrique" else ""),
            "Une <b>estimation</b> pour fixer la dalle avant SCIA, qui reste la référence"]


def _fenetre_k(el: Dict, calc: Dict, s: Dict) -> str:
    lignes = []
    scia = set(PR.k_scia(el))
    for r in calc["par_k"]:
        pire = r["cas"][V.premier_max([c["sigma_max"] for c in r["cas"]])]
        t = r["sigma_max"] / s["limite"]
        lignes.append({"nom": f"k = {_k(r['k'])}{N}MN/m³", "sous": "coché SCIA" if r["k"] in scia else "", "formule": f"cas {C.esc(pire['nom'])} · {_reactions(pire['R'])}{N}kN",
                       "valeur": SS.sol(el, r["sigma_max"]), "limite": SS.sol(el, s["limite"]), "taux": C.pct(t), "niveau": C.niveau(t),
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
                       "valeur": SS.sol(el, c["sigma_max"]), "limite": f"{C.fr(c['w_max'], 2)}{N}mm", "taux": C.pct(t), "niveau": C.niveau(t),
                       "gov": c["nom"] == r["cas_max"]})
    hyps = ["Les charges <b>tournent</b> : chaque jeu est placé dans toutes les positions de la grue (rotations de 90° et miroirs), les doublons retirés",
            f"Raideur affichée : <b>k = {_k(r['k'])}{N}MN/m³</b>"]
    return C.fenetre_calcul("s2", "Positions des charges", "P1 / P2 / P3 / P4 de chaque cas généré", hyps, lignes, "",
                            colonnes=("Cas", "P1 / P2 / P3 / P4", "σ max", "Tassement", "Taux"))

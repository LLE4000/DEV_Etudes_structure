# -*- coding: utf-8 -*-
"""Les RÉSULTATS du « Générateur SAF — Fondation de grue » (api_web/saf.py) : la colonne de droite, la barre du centre
(la carte, le cas et le k affichés) et le champ dessiné sur le plan.

La colonne reprend les composants du module « Poutres béton armé » (lot 5, § 5.2) : le nom de la fondation, la ligne
d'étapes (1 Données · 2 Vérification · 3 SAF · 4 Efforts SCIA · 5 Note), puis une CARTE par sujet — SOL, ÉQUILIBRE,
CONTRÔLES, ARMATURES (E.bloc : un titre, son état à droite) —, une ligne par contrôle (C.ligne_verif : le nom, le taux en
%, la formule avec ses valeurs, une petite barre), la ligne grise « Non vérifiés : … », « Proposer », les boutons
« Exporter le SAF » et « Imprimer la note », le pied. Pas de grand chiffre en tête.

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


from . import commun as C
from . import etabli_compact as E
from . import saf_saisie as SS

N = C.NBSP
CARTES = (("sigma", "Contrainte"), ("w", "Tassement"), ("armatures", "Armatures"), ("aucune", "Sans carte"))
MOMENT_UNITE = ("kN·m/m", 1.0, 1)
CLASSES = {"p": "ok", "f": "warn", "r": "nok", "n": "inc", "x": "nok"}
ETATS = {"ok": "Conforme", "warn": "Conforme", "nok": "Non conforme", "inc": "Données incomplètes"}
ENVELOPPE_CAS, ENVELOPPE_K = "Enveloppe des cas", "Enveloppe des k"
# ce que la méthode ne vérifie pas (§ 11) : la ligne grise sous les cartes (les armatures y ajoutent les leurs)
NON_VERIFIES = ("la fissuration", "les efforts horizontaux et les moments aux pieds de la grue", "la torsion",
                "les ancrages et les recouvrements (le principe n'est pas un plan d'exécution)")


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
    cas et des k retenus (``selection``) ; None sans carte, sur la carte Armatures (une couche à part) ou sans calcul.
    Un nœud est « décollé » quand il l'est dans TOUS les cas retenus."""
    mode = el["vue"]["carte"]
    if s.get("r_vue") is None or mode in ("aucune", "armatures"):
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
def _menu(chemin: str, valeur: str, options, titre: str, classe: str = "", courts: Optional[Dict[str, str]] = None) -> str:
    """Une liste déroulante de la barre du centre (data-agir : js/plan.js l'envoie au moteur). ``courts`` : le libellé
    court de chaque choix — js/plan.js l'affiche à la place du libellé du choix QUAND celui-ci ne tient pas ; la liste
    dépliée garde les longs."""
    opts = "".join(f'<option value="{C.esc(v)}"{" selected" if v == valeur else ""}>{C.esc(t)}</option>' for v, t in options)
    dc = f' data-courts="{C.esc(json.dumps(courts, ensure_ascii=False))}"' if courts else ""
    return (f'<div class="field select{(" " + classe) if classe else ""}"><select data-agir="1" data-chemin="{chemin}" aria-label="{C.esc(titre)}" '
            f'title="{C.esc(titre)}"{dc}>{opts}</select>{C.icon("chevron-down")}</div>')


# les libellés courts de la barre du centre (affichés seulement quand le long ne tient pas)
COURTS_CARTES = {"sigma": "σ", "w": "w", "armatures": "Arm.", "aucune": "Aucune"}


def entete(el: Dict, calc: Dict, s: Dict) -> str:
    """La barre du centre, sur UNE ligne (lot 5, § 5.1) : le choix de la carte (Contrainte, Tassement, Armatures, Sans
    carte), puis la liste « Cas » et la liste « k » (« Env. cas », « Env. k » en tête) ; la carte Armatures montre le
    principe en enveloppe, avec son choix « Bas · Haut » à la place des deux listes."""
    vue = el["vue"]
    out = _menu("vue/carte", vue["carte"], CARTES, "La carte peinte sur le plan", "sf-m-carte", COURTS_CARTES)
    r = s.get("r_vue")
    if vue["carte"] == "armatures":
        bt = "".join(f'<button type="button" class="{"on" if k == vue["face"] else ""}" data-agir="1" data-chemin="vue/face" data-valeur="{k}">{t}</button>'
                     for k, t in (("inf", "Bas"), ("sup", "Haut")))
        out += f'<div class="seg sf-faces" role="group" aria-label="Face des armatures">{bt}</div>'
    elif r is not None and vue["carte"] != "aucune":
        ks, cas = selection(el, calc)
        if not s["pp"]:
            cas_txt = [("", ENVELOPPE_CAS)] + [(c["nom"], f"{c['nom']} · {_reactions(c['R'])}") for c in r["cas"]]
            out += _menu("vue/cas", cas or "", cas_txt, "Le cas affiché : l'enveloppe de tous les cas, ou une position", "sf-m-cas",
                         {"": "Env. cas", **{c["nom"]: c["nom"] for c in r["cas"]}})
        k_txt = [("", ENVELOPPE_K)] + [(repr(x["k"]), f"k{N}={N}{_k(x['k'])}") for x in el["sol"]["k"]]
        out += _menu("vue/k", repr(vue["k"]) if vue["k"] is not None else "", k_txt,
                     "La raideur de sol affichée : l'enveloppe des k cochés « SCIA », ou un k", "sf-m-k",
                     {"": "Env. k", **{repr(x["k"]): _k(x["k"]) for x in el["sol"]["k"]}})
    return f'<div class="et-ent sf-ent sf-barre">{out}</div>'


# ------------------------------------------------------------------ la ligne d'étapes (§ 1)
def etapes(el: Dict, calc: Dict, s: Dict, fer: Optional[Dict], sc: Optional[Dict], faites_saf: bool, faite_note: bool) -> List[Tuple[str, bool]]:
    """Les cinq étapes et ce qui est fait (lot 5, § 1) : 1 Données (un jeu complet, un k coché, le calcul possible) ;
    2 Vérification (le sol et le principe d'armatures calculés) ; 3 SAF (exporté pour ces données) ; 4 Efforts SCIA (le
    tableau rempli) ; 5 Note (imprimée pour ces données)."""
    d1 = s.get("r_vue") is not None and not s.get("pp") and bool(PR.k_scia(el))
    d2 = d1 and fer is not None and not fer.get("hors_domaine")
    return [("Données", d1), ("Vérification", d2), ("SAF", d1 and faites_saf), ("Efforts SCIA", bool(sc and sc["complet"])),
            ("Note", faite_note)]


def ligne_etapes(items: List[Tuple[str, bool]]) -> str:
    """La ligne d'étapes en tête de la colonne de droite (composant partagé de l'établi : ``E.etapes``)."""
    return E.etapes(items)


# ------------------------------------------------------------------ la colonne des vérifications
def _table_k(el: Dict, calc: Dict, s: Dict) -> str:
    """Le tableau par k : k | σ max | tassement max | surface décollée | case SCIA ; chaque σ à la couleur de sa
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


def _etat_ligne(bon: bool, grave: bool = True) -> Tuple[str, str]:
    """(niveau, mot) d'une ligne de contrôle sans taux : « OK », « Non conforme », « À vérifier » (les mots de l'établi)."""
    if bon:
        return "p", "OK"
    return ("r", "Non conforme") if grave else ("v", "À vérifier")


def _controles(el: Dict, calc: Dict, s: Dict) -> Tuple[str, str]:
    """(les lignes de la carte CONTRÔLES, la classe de la carte) : appuis dans la dalle, distance au bord, plots."""
    m = calc["modele"]
    ctl, g = m["controles"], el["grue"]
    rows, pire = "", "ok"

    def ligne(nom, texte, bon, grave=True):
        nonlocal rows, pire
        niv, mot = _etat_ligne(bon, grave)
        if not bon:
            pire = "nok" if grave or pire == "nok" else "warn"
        rows += C.ligne_verif(nom, texte, None, niv, calcul="s1", etat_txt=mot)
    dedans = all(ctl["plaques_dedans"])
    ligne("Appuis dans la dalle", "" if dedans else "une surface d'appui sort", dedans)
    if dedans:
        ligne("Distance au bord", f"{C.fr(ctl['distance'], 2)}{N}m {'≥' if ctl['distance_ok'] else '&lt;'} {C.fr(g['dmin'], 2)}{N}m",
              ctl["distance_ok"], grave=False)
    if m["plots"]:
        ligne("Plots dans la dalle", "" if all(ctl["plots_dedans"]) else "un plot sort", all(ctl["plots_dedans"]))
        ligne("Plots séparés", "" if ctl["plots_separes"] else "deux plots se chevauchent", ctl["plots_separes"])
        ligne("Appuis sur leur plot", "" if all(ctl["plots_couvrent"]) else "une surface d'appui déborde", all(ctl["plots_couvrent"]))
    return rows, pire


def equilibre_cas(el: Dict, calc: Dict, s: Dict) -> Optional[Dict]:
    """L'équilibre du cas affiché (lot 5, § 5.2 ; avec l'enveloppe : le cas de σ max) : {"cas", "k", "resultante" (la
    contrainte de contact intégrée sur la dalle : Σ σ × aire des nœuds, les zones décollées à 0), "grue" (Σ des quatre
    réactions), "dalle" (25 × h × aire), "plots" (25 × hauteur × aire d'un plot × nombre de plots), "charges", "ecart",
    "ecart_max" (sur tous les cas et tous les k)}."""
    r = s.get("r_vue")
    if r is None:
        return None
    c = r["cas"][s["i_cas"]]
    m = calc["modele"]
    gamma = m.get("gamma", PR.GAMMA_BETON)
    dalle_ = gamma * m["h"] * G.aire(m["contour"])
    pl = m["plots"]
    plots_ = sum(gamma * pl["hauteur"] * G.aire(q) for q in pl["polys"]) if pl else 0.0
    grue = float(sum(c["R"]))
    charges = grue + dalle_ + plots_
    return {"cas": c["nom"], "k": r["k"], "resultante": c["reaction_sol"], "grue": grue, "dalle": dalle_, "plots": plots_,
            "charges": charges, "ecart": abs(c["reaction_sol"] - charges) / charges if charges else 0.0, "ecart_max": s["equilibre"],
            "n_cas": sum(len(x["cas"]) for x in calc["par_k"]), "n_k": len(calc["par_k"])}


def phrase_equilibre(q: Dict) -> str:
    """« Résultante du sol 2 380,0 kN · Charges : grue 1 580,0 + dalle 800,0 + plots 0,0 = 2 380,0 kN · Écart 0,00 % »."""
    return (f"Résultante du sol {C.fr(q['resultante'], 1)}{N}kN · Charges : grue {C.fr(q['grue'], 1)} + dalle {C.fr(q['dalle'], 1)} "
            f"+ plots {C.fr(q['plots'], 1)} = {C.fr(q['charges'], 1)}{N}kN · Écart {C.fr(q['ecart'] * 100, 2)}{N}%")


ECART_EQUILIBRE = 1e-3          # § 5.2 : pastille verte sous 0,1 %


def _carte_equilibre(el: Dict, calc: Dict, s: Dict) -> str:
    """La carte ÉQUILIBRE (§ 5.2) : la résultante du sol face aux charges, en kN, pour le cas affiché ; dessous, l'écart
    le plus grand sur tous les cas et tous les k (vert sous 0,1 %)."""
    q = equilibre_cas(el, calc, s)
    bon = q["ecart_max"] < ECART_EQUILIBRE
    rows = (C.ligne_verif(f"Cas {q['cas']} · k{N}={N}{_k(q['k'])}", phrase_equilibre(q), None, "p" if q["ecart"] < ECART_EQUILIBRE else "r",
                          calcul="s0", etat_txt=f"{C.fr(q['ecart'] * 100, 2)}{N}%")
            + C.ligne_verif("Écart le plus grand", f"sur {q['n_cas']} cas et {q['n_k']} k{N}: {C.fr(q['ecart_max'] * 100, 2)}{N}% "
                            f"{'&lt;' if bon else '≥'} 0,1{N}%", None, "p" if bon else "r", calcul="s0", etat_txt="OK" if bon else "Non conforme"))
    return E.bloc(1, "Équilibre", "ok" if bon else "nok", "Conforme" if bon else "Non conforme", "", rows, theme=True, calcul="s0")


def tableau_proposition(el: Dict, lignes: List[Dict], complet: bool = True) -> str:
    """Le tableau de « Proposer » : h | dimensions | σ max | volume de béton (dalle + plots) ; la ligne au plus petit
    volume est mise en évidence ; un clic sur une ligne l'APPLIQUE (Lx, Ly, h ; § 9) quand le tableau est complet."""
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
    """« Proposer » (§ 9), sa barre de progression (js/plan.js : une épaisseur par requête), son tableau ; un clic sur une
    ligne applique la dalle, « Annuler » rend les dimensions d'avant."""
    possible = bool(calc["modele"]["cas"]) and bool(PR.k_scia(el)) and s.get("r_vue") is not None
    suite = {"role": "proposer", "etapes": [f"{h:.3f}" for h in PR.epaisseurs(el)], "fin": "proposition/garder", "sortie": "#sf-prop-t"}
    pourquoi = ' title="La plus petite dalle sous la limite, par épaisseur"' if possible else ' title="Il faut un jeu de réactions complet et un k coché « SCIA »."'
    p = el.get("proposition")
    table = tableau_proposition(el, p["lignes"]) if p else ""
    annuler = (f'<button type="button" class="btn sm" data-agir="1" data-chemin="proposition/annuler" data-valeur="1" '
               f'title="Revenir aux dimensions d\'avant">{C.icon("undo-2")}Annuler</button>') if p and p.get("avant") else ""
    retirer = (f'<button type="button" class="sf-x" data-agir="1" data-chemin="proposition/-" data-valeur="1" title="Retirer ce tableau" '
               f'aria-label="Retirer ce tableau">{C.icon("x")}</button>') if p else ""
    return (f'<div class="sf-prop" data-suite-hote="1"><div class="sf-prop-h"><button type="button" class="btn sm" data-suite="{C.esc(json.dumps(suite))}"'
            f'{"" if possible else " disabled"}{pourquoi}>{C.icon("lightbulb")}Proposer</button>{annuler}{retirer}</div><div id="sf-prop-t">{table}</div></div>')


def _boutons(el: Dict, calc: Dict, joindre: bool = True) -> str:
    """Les deux boutons (§ 5.2, § 10) : « Exporter le SAF » (UN fichier, le premier k coché « SCIA ») et « Imprimer la
    note » (un PDF) — à côté, « Joindre les notes SCIA » (les PDF de SCIA ajoutés à la fin de la note). Ce qui empêche
    l'export est dit dessous, ce qui l'accompagne d'un avertissement aussi."""
    from . import saf as M                                        # (les contrôles de l'export sont ceux du module)
    bloque, avertit = M.controles_export(el, calc)
    k1 = PR.k_scia(el)[0] if PR.k_scia(el) else None
    titre = f' title="Le SAF est écrit avec k{N}={N}{_k(k1)}{N}MN/m³ (le premier coché « SCIA »)"' if k1 is not None else ""
    saf = (f'<button type="button" class="btn primary sf-exp-b"{" disabled" if bloque else " data-telecharger=" + chr(34) + "saf" + chr(34) + " data-apres=" + chr(34) + "etapes/saf" + chr(34)}{titre}>'
           f'{C.icon("download")}Exporter le SAF</button>')
    note = (f'<button type="button" class="btn sf-exp-b" data-telecharger="note_pdf" data-apres="etapes/note" title="La note de calcul, un PDF">'
            f'{C.icon("printer")}Imprimer la note</button>')
    jn = (f'<button type="button" class="btn sm sf-joindre" data-joindre="note_pdf" data-apres="etapes/note" '
          f'title="Choisir les PDF de SCIA (un par k) : ils sont ajoutés à la fin de la note">{C.icon("upload")}Joindre les notes SCIA</button>') if joindre else ""
    mots = "".join(C.message(C.esc(SS.phrase(t)), True) for t in bloque) + "".join(C.message(C.esc(SS.phrase(t))) for t in avertit)
    return f'<div class="sf-export"><div class="sf-exp-l">{saf}{note}</div>{jn}{mots}</div>'


def non_verifies(el: Dict, calc: Dict, extra: Sequence[str] = ()) -> str:
    """La ligne grise sous les cartes (§ 11) : ce que la méthode ne vérifie pas, selon le cas."""
    t = list(extra) + list(NON_VERIFIES)
    m = calc["modele"]
    if any(any(v < 0 for v in c["R"]) for c in m["cas"]):
        t.append("l'arrachement")
    return f"Non vérifiés : {', '.join(dict.fromkeys(t))}."


def _carte_sol(el: Dict, calc: Dict, s: Dict) -> str:
    """La carte SOL : la contrainte de contact (taux = σ max / limite, avec le cas et le k), le tassement max, la surface
    décollée, σ moyen, les cas générés ; le tableau par k, dépliable dans la carte."""
    m = calc["modele"]
    lim = s["limite"]
    retenus, _ = selection(dict(el, vue=dict(el["vue"], k=None, cas=None)), calc)
    w = max(retenus, key=lambda r: r["w_max"])
    dec = max(retenus, key=lambda r: r["decolle"])
    if s["pp"]:
        cmp_ = f"σ max{N}={N}{SS.sol(el, s['sigma'])} · poids propre seul · réactions à saisir"
        ligne = C.ligne_verif("Contrainte de contact", cmp_, None, "c", calcul="s0", etat_txt="Données incomplètes")
        cls = "inc"
    else:
        signe = "≤" if s["taux"] <= 1 else "&gt;"
        cmp_ = f"σ max{N}={N}{SS.sol(el, s['sigma'])} {signe} {SS.sol(el, lim)} · cas{N}{C.esc(s['cas'])} · k{N}={N}{_k(s['k'])}"
        ligne = C.ligne_verif("Contrainte de contact", cmp_, C.pct(s["taux"]), C.niveau(s["taux"]), calcul="s0")
        cls = CLASSES[s["niveau"]]
    sm = V.sigma_moyen(m)
    moyen = (f"({C.fr(sm['dalle'], 0)}" + (f" + {C.fr(sm['plots'], 0)}" if sm["plots"] else "")
             + (f" + {C.fr(sm['reactions'], 0)}" if sm["reactions"] else "") + f") ÷ {C.fr(sm['aire'], 1)}{N}m² = {SS.sol(el, sm['sigma'])}")
    n = len(s["r_vue"]["cas"]) if s.get("r_vue") else 0
    rows = (ligne
            + C.ligne_verif("Tassement max", f"w max{N}={N}{C.fr(w['w_max'], 1)}{N}mm · k{N}={N}{_k(w['k'])}", None, "c", calcul="s0", etat_txt="valeur")
            + C.ligne_verif("Surface décollée", f"{C.fr(dec['decolle'], 1)}{N}% · k{N}={N}{_k(dec['k'])}", None, "c", calcul="s0", etat_txt="valeur")
            + C.ligne_verif("σ moyen", moyen, None, "c", calcul="s0", etat_txt="repère")
            + ("" if s["pp"] else C.ligne_verif("Cas générés", f"{n} positions · σ max de chacune", None, "c", calcul="s2", etat_txt=f"{n}{N}cas")))
    n_scia = len(PR.k_scia(el))
    table = (f'<details class="sf-tkd" data-plan-volet="tablek"><summary><b>Tableau par k</b><span>{n_scia} coché{"s" if n_scia > 1 else ""} '
             f'« SCIA »</span>{C.icon("chevron-down")}</summary>{_table_k(el, calc, s)}</details>')
    return E.bloc(0, "Sol", cls, ETATS[cls] if not s["pp"] else "Données incomplètes", "", rows, table, theme=True)


def colonne(el: Dict, calc: Dict, s: Dict, carte_armatures: str = "", non_verifies_extra: Sequence[str] = (),
            etapes_: Optional[List[Tuple[str, bool]]] = None) -> Tuple[str, str]:
    """(la colonne des résultats, les fenêtres de calcul) : la ligne d'étapes, SOL, ÉQUILIBRE, CONTRÔLES, ARMATURES, la
    ligne grise « Non vérifiés », « Proposer », « Exporter le SAF » et « Imprimer la note », le pied (§ 5.2)."""
    nom = C.esc(el.get("nom") or "Fondation")
    tete = E.tete_resultats(nom, f"{SS.dimensions(el)} · {C.esc(el['dalle']['beton'])}")
    tete += ligne_etapes(etapes_ or [(t, False) for t in ("Données", "Vérification", "SAF", "Efforts SCIA", "Note")])
    ctl_rows, ctl_cls = _controles(el, calc, s)
    if s.get("r_vue") is None:
        rows = C.ligne_verif("Calcul", C.esc(s["phrase"]), None, "r", etat_txt="impossible")
        blocs = E.bloc(0, "Sol", "nok", "Calcul impossible", "", rows, theme=True) + E.bloc(2, "Contrôles", ctl_cls, ETATS[ctl_cls], "", ctl_rows, theme=True, calcul="s1")
        return (f'<div class="pbody">{tete}<div class="et-blocs">{blocs}</div>{C.aide(non_verifies(el, calc, non_verifies_extra))}'
                f'{_boutons(el, calc, joindre=False)}</div>', _fenetre_controles(el, calc, s))
    blocs = (_carte_sol(el, calc, s) + _carte_equilibre(el, calc, s)
             + E.bloc(2, "Contrôles", ctl_cls, ETATS[ctl_cls], "", ctl_rows, theme=True, calcul="s1") + carte_armatures)
    html = (f'<div class="pbody">{tete}<div class="et-blocs">{blocs}</div>{C.aide(non_verifies(el, calc, non_verifies_extra))}'
            f'{_proposer(el, calc, s)}{_boutons(el, calc)}</div>')
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
            + (" (sous un plot : diffusée à 45° depuis une plaque de 0,50 × 0,50, sur un carré centré sur le pied)" if plots else ""),
            "Une <b>vérification complémentaire</b> de SCIA, par un calcul indépendant"]


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

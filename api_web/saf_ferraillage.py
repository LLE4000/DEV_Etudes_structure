# -*- coding: utf-8 -*-
"""Les cartes du FERRAILLAGE du « Générateur SAF — Fondation de grue » (lot 4c, § 5) : la colonne de droite, sous SOL et
CONTRÔLES, avec les composants du module « Poutres béton armé » (E.bloc : un titre, son état ; C.ligne_verif : le nom,
le taux en %, ce qu'on compare, une jauge) — DALLE (les treillis, la hauteur, le poinçonnement), BX1 à BY2 (une bande =
une poutre : inférieure, supérieure, effort tranchant, hauteur), PLOTS — et leurs fenêtres de calcul.

Ce fichier ne calcule rien : il met en mots ``saf.fondation_grue.ferraillage.calculer`` (``saf.ferraillage``).
Les valeurs par mètre = valeur intégrée sur la bande ÷ largeur de la bande (§ 4.3).
"""
from typing import Dict, List, Tuple

import numpy as np

from saf.fondation_grue import ferraillage as FE

from . import commun as C
from . import etabli_compact as E

N = C.NBSP
CLASSES = {"p": "ok", "f": "warn", "r": "nok", "c": "inc"}
ETATS = {"ok": "Conforme", "warn": "Conforme", "nok": "Non conforme", "inc": "non vérifié"}
BLOC_DALLE, BLOC_BANDES, BLOC_PLOTS = 3, 4, 8              # (s0 : sol, s1 : contrôles, s2 : positions)
MOTS_TAU = {"pas_d_etriers": "sans armature", "etriers": "C verticaux", "barres_inclinees_et_etriers": "C verticaux (barres inclinées à examiner)",
            "inacceptable": "inacceptable"}


def _mm2(x: float) -> str:
    return C.fr(x, 0)


def _pire(*taux) -> str:
    """La classe d'une carte : celle de son plus grand taux."""
    t = [x for x in taux if x is not None]
    return CLASSES[C.niveau(max(t))] if t else "inc"


def _renfort(r: Dict, rep: str) -> str:
    return f"{rep} : {r['n']}{N}Ø{r['d']} (e{N}≤{N}{C.fr(r['s'], 1)}{N}cm), L{N}={N}{C.fr(r['L'], 2)}{N}m"


def _reperes(fer: Dict) -> Dict[Tuple[str, str], str]:
    """(bande, face) → le repère du bordereau (R1…)."""
    return {(ln["bande"], ln["cote"]): ln["rep"] for ln in fer["bordereau"]["lignes"] if ln["famille"] == "R"}


def _taux_treillis(f: Dict) -> float:
    """Le plus grand taux de travail de l'acier en place dans la zone courante (treillis + renforts locaux)."""
    t = 0.0
    for cv in f["couverture"].values():
        if cv["controle"].any():
            t = max(t, float((cv["besoin"][cv["controle"]] / cv["As_place"][cv["controle"]]).max()))
    return t


def carte_dalle(el: Dict, fer: Dict) -> str:
    """DALLE : le treillis de chaque face (et ses renforts locaux), la hauteur dans la zone courante, les U, le
    poinçonnement (non vérifié), l'arrachement s'il y a lieu, l'acier et le béton."""
    rows, taux = "", []
    for cote, nom in (("inf", "Treillis inférieur"), ("sup", "Treillis supérieur")):
        f = fer["faces"][cote]
        t = _taux_treillis(f)
        taux.append(t)
        n_loc = sum(1 for r in f["locaux"] if not r["impossible"])
        loc = (f" + {n_loc} renforts locaux" if n_loc > 1 else " + 1 renfort local") if n_loc else ""
        auto = "automatique" if f["auto"] else "imposé"
        rows += C.ligne_verif(nom, f"{FE.libelle_treillis(f['code'])} (x et y) · {auto}{loc} · A_s{N}={N}{_mm2(f['As_pm'])}{N}mm²/m",
                              C.pct(t), C.niveau(t), calcul=f"s{BLOC_DALLE}")
    hc = fer["hauteur"]["courante"]
    taux.append(hc["taux"])
    rows += C.ligne_verif("Hauteur (zone courante)", f"h_min{N}={N}{C.fr(hc['h_min'], 1)} ≤ h{N}={N}{C.fr(hc['h'], 0)}{N}cm",
                          C.pct(hc["taux"]), C.niveau(hc["taux"]), calcul=f"s{BLOC_DALLE}")
    u = fer["U"]
    par = sorted({b["n"] for b in u["bords"]})
    rows += C.ligne_verif("U de fermeture", f"U1 : Ø{u['d']} e{N}={N}{C.fr(u['e'], 0)}{N}cm – "
                          + (f"{par[0]} U par bord" if len(par) == 1 else f"{u['n']} U en tout"), None, "c", etat_txt="info")
    for titre, texte, niv in fer["lignes"]:
        if titre == "Brins verticaux sous le pied":
            continue
        rows += C.ligne_verif(titre, C.esc(texte), None, "r" if niv == "nok" else "c", etat_txt="non traité" if niv == "nok" else "non vérifié")
    bd = fer["bordereau"]
    rows += C.ligne_verif("Acier", f"{C.fr(bd['total'], 0)}{N}kg · béton {C.fr(bd['beton'], 1)}{N}m³ · {C.fr(bd['ratio'], 0)}{N}kg/m³",
                          None, "c", etat_txt="info")
    cls = "nok" if fer["arrachement"] else _pire(*taux)
    return E.bloc(BLOC_DALLE, "Dalle", cls, ETATS[cls], "", rows, bordereau(fer), theme=True)


def bordereau(fer: Dict) -> str:
    """Le bordereau des aciers (§ 6.3), dépliable : repère, Ø, forme (cotes en m), nombre, longueur unitaire, longueur
    totale, poids ; les totaux par Ø, le total, le béton et le ratio."""
    from . import note_saf_ferraillage as NF                     # (la forme en mots : la même que la note)
    bd = fer["bordereau"]
    rows = "".join(f'<tr><td><b>{C.esc(ln["rep"])}</b></td><td class="n">{ln["d"]}</td><td>{C.esc(NF.forme_texte(ln))}</td>'
                   f'<td class="n">{ln["n"]}</td><td class="n">{C.fr(ln["L"], 2)}</td><td class="n">{C.fr(ln["Lt"], 1)}</td>'
                   f'<td class="n">{C.fr(ln["poids"], 1)}</td></tr>' for ln in bd["lignes"])
    tot = " · ".join(f"Ø{d} : {C.fr(p, 0)}{N}kg" for d, p in bd["par_d"].items())
    return (f'<details class="sf-tkd sf-bord" data-plan-volet="bordereau"><summary><b>Bordereau</b><span>{C.fr(bd["total"], 0)}{N}kg</span>'
            f'{C.icon("chevron-down")}</summary><table class="sf-bd"><thead><tr><th>Rep.</th><th class="n">Ø</th><th>Forme (m)</th>'
            f'<th class="n">Nb</th><th class="n">L</th><th class="n">L tot.</th><th class="n">kg</th></tr></thead><tbody>{rows}</tbody></table>'
            f'<p class="sf-bord-t">{tot} · total {C.fr(bd["total"], 0)}{N}kg · béton {C.fr(bd["beton"], 2)}{N}m³ · {C.fr(bd["ratio"], 0)}{N}kg/m³</p></details>')


def carte_bande(el: Dict, fer: Dict, ef: Dict, i: int, nom: str) -> str:
    """Une BANDE = une poutre de largeur b (§ 5.1) : ses efforts (intégrés, et par mètre), puis inférieure, supérieure
    (treillis + renfort ≥ l'acier demandé par le moteur), l'effort tranchant (V à d au d du moteur), la hauteur."""
    rb = next(x for x in ef["bandes"] if x["nom"] == nom)
    b = rb["largeur"]
    reps = _reperes(fer)
    t = fer["tranchant"][nom]
    eff = (f'<span>M⁺{N}={N}{C.fr(rb["M_plus"], 1)}{N}kNm ({C.fr(rb["M_plus"] / b, 1)}{N}kNm/m)</span> · '
           f'<span>M⁻{N}={N}{C.fr(rb["M_moins"], 1)}{N}kNm ({C.fr(rb["M_moins"] / b, 1)}{N}kNm/m)</span> · '
           f'<span>V à d{N}={N}{C.fr(t["V"], 1)}{N}kN ({C.fr(t["V"] / b, 1)}{N}kN/m)</span>')
    rows, taux = "", []
    for cote, titre in (("inf", "Inférieure"), ("sup", "Supérieure")):
        bf = fer["faces"][cote]["bandes"][nom]
        r = bf["renfort"]
        pose = f"treillis {_mm2(bf['As_t_pm'])}" + (f" + {_mm2(r['As_pm'])} ({_renfort(r, reps.get((nom, cote), 'R'))})" if r else "")
        signe = "≤" if bf["taux"] <= 1 + 1e-9 else "&gt;"
        txt = (f"A_s,néc{N}={N}{_mm2(bf['retenu']['necessaire_pm'])} {signe} {_mm2(bf['As_place_pm'])}{N}mm²/m · {pose}"
               + (" · renfort impossible : augmenter h" if bf["impossible"] else ""))
        taux.append(bf["taux"] if not bf["impossible"] else 9.99)
        rows += C.ligne_verif(titre, txt, C.pct(taux[-1]), C.niveau(taux[-1]), calcul=f"s{BLOC_BANDES + i}")
    if t["tau"] is None:
        rows += C.ligne_verif("Effort tranchant", "V à d = 0", None, "c", etat_txt="sans objet")
    else:
        tt = t["taux"] if t["besoin_tau"] != "inacceptable" else max(t["taux"], 1.01)
        quoi = MOTS_TAU.get(t["besoin_tau"], "")
        c_ = (f" → V : {t['C']['n']} C Ø{t['C']['d']} e{N}={N}{C.fr(t['C']['s'], 1)}{N}cm" if t["C"] else "")
        rows += C.ligne_verif("Effort tranchant", f"τ{N}={N}{C.fr(t['tau'], 3)} ≤ τ_adm,{t['bande']}{N}={N}{C.fr(t['tau_lim'], 3)}{N}N/mm² "
                              f"(d{N}={N}{C.fr(t['d_v'] * 100, 1)}{N}cm) · {quoi}{c_}", C.pct(tt), C.niveau(tt), calcul=f"s{BLOC_BANDES + i}")
        taux.append(tt)
    hb = fer["hauteur"][nom]
    rows += C.ligne_verif("Hauteur", f"h_min{N}={N}{C.fr(hb['h_min'], 1)} ≤ h{N}={N}{C.fr(hb['h'], 0)}{N}cm", C.pct(hb["taux"]), C.niveau(hb["taux"]),
                          calcul=f"s{BLOC_BANDES + i}")
    taux.append(hb["taux"])
    cls = _pire(*taux)
    return E.bloc(BLOC_BANDES + i, nom, cls, ETATS[cls], f'<div class="sf-ef">{eff}</div>', rows, theme=True)


def carte_plots(el: Dict, fer: Dict, ef: Dict) -> str:
    """PLOTS : la cage d'un plot coulé (C sur les 4 faces, 2 U en tête), les moments dans le plot à titre d'information
    (section h + plot : non contrôlés avec le treillis de la dalle), les brins verticaux « à dimensionner » ; un plot
    préfabriqué est un produit fourni, sans cage."""
    pl = el["grue"]["plots"]
    if pl["type"] == "prefabrique":
        rows = C.ligne_verif("Plots préfabriqués", f"{C.fr(pl['a'], 2)} × {C.fr(pl['b'], 2)} × {C.fr(pl['h'], 2)}{N}m · produit fourni : pas de cage, "
                             "la dalle est ferraillée seule", None, "c", etat_txt="info")
        return E.bloc(BLOC_PLOTS, "Plots", "inc", "info", "", rows, theme=True)
    cg = fer["cages"][0]
    reps = [ln for ln in fer["bordereau"]["lignes"] if ln["famille"] == "C"]
    rows = C.ligne_verif("Cage", f"C verticaux Ø{cg['d']} e{N}={N}{C.fr(cg['e'], 0)}{N}cm sur les 4 faces ({cg['C'][0]['n']} + {cg['C'][1]['n']} par plot), "
                         f"2 U en tête · {', '.join(ln['rep'] for ln in reps)}", None, "c", etat_txt="info")
    ch = ef["champ"]
    m_inf = float(np.maximum(ch["inf_x"], ch["inf_y"])[ch["plot"]].max())     # (tous les éléments des plots, zones comprises)
    m_sup = float(np.minimum(ch["sup_x"], ch["sup_y"])[ch["plot"]].min())
    rows += C.ligne_verif("Moments dans le plot", f"m_inf{N}={N}{C.fr(m_inf, 1)} · m_sup{N}={N}{C.fr(m_sup, 1)}{N}kNm/m "
                          "(section h + plot, non contrôlés avec le treillis)", None, "c", etat_txt="info")
    rows += C.ligne_verif("Brins verticaux sous le pied", "à dimensionner (poinçonnement non vérifié)", None, "c", etat_txt="non vérifié")
    return E.bloc(BLOC_PLOTS, "Plots", "inc", "non vérifié", "", rows, theme=True)


def carte_erreur(texte: str) -> str:
    """Le ferraillage n'a pas pu se faire : la raison, en rouge."""
    return E.bloc(BLOC_DALLE, "Ferraillage", "nok", "Calcul impossible", "",
                  C.ligne_verif("Ferraillage", C.esc(texte), None, "r", etat_txt="impossible"), theme=True)


def cartes(el: Dict, fer: Dict, ef: Dict) -> str:
    """Les cartes du ferraillage, dans l'ordre de la colonne : DALLE, BX1, BX2, BY1, BY2, PLOTS."""
    out = carte_dalle(el, fer)
    for i, nom in enumerate(b["nom"] for b in ef["bandes"]):
        out += carte_bande(el, fer, ef, i, nom)
    if el["grue"]["plots"]["type"] != "aucun":
        out += carte_plots(el, fer, ef)
    return out


# ------------------------------------------------------------------ les fenêtres de calcul
def _hyps(el: Dict, fer: Dict) -> List[str]:
    o, a = el["options"], el["armatures"]
    return [f"Méthode du bureau, ELS — le moteur du module « Poutres béton armé » : σ_s = f_yk / γ_s (γ_s = {C.fr(o['gamma_s'], 2)})",
            f"Enrobage <b>{C.fr(a['enrobage'] * 100, 1)}{N}cm</b> · jeu de pose {C.fr(o['jeu'], 1)}{N}cm · axe arrondi à {C.fr(o['arrondi'], 1)}{N}cm",
            "Nappes : les barres en x à l'extérieur, en y à l'intérieur (posées sur la plus grosse barre en x) ; un renfort dans le plan de sa direction",
            f"Recouvrement {C.fr(o['recouvrement'], 0)}{N}Ø (le logiciel n'a pas de règle) · barres de {C.fr(o['l_max'], 0)}{N}m au plus",
            "A_s,min « 0,25·A_s,req opposé » calculé avec la face opposée la plus défavorable (Ø20)"]


def _fenetre_dalle(el: Dict, fer: Dict) -> str:
    lignes = []
    for cote in ("inf", "sup"):
        f = fer["faces"][cote]
        for c in (f["candidats"] or [{"code": f["code"], "poids": f["poids"], "poids_T": f["poids_T"], "poids_R": f["poids_R"],
                                       "poids_L": f["poids_L"], "ok": f["ok"]}]):
            lignes.append({"nom": f"{'Inférieur' if cote == 'inf' else 'Supérieur'} · {FE.libelle_treillis(c['code'])}",
                           "sous": "retenu" if c["code"] == f["code"] else "",
                           "formule": f"treillis {C.fr(c['poids_T'], 0)} + bandes {C.fr(c['poids_R'], 0)} + locaux {C.fr(c['poids_L'], 0)}{N}kg",
                           "valeur": f"{C.fr(c['poids'], 0)}{N}kg", "limite": "", "taux": None, "niveau": "p" if c["ok"] else "r",
                           "etat_txt": "retenu" if c["code"] == f["code"] else "", "gov": c["code"] == f["code"]})
    detail = ""
    for cote in ("inf", "sup"):
        for r in fer["faces"][cote]["locaux"]:
            if r["impossible"]:
                continue
            x0, y0, x1, y1 = r["rect"]
            detail += (f'<div class="ck-g"><div class="ck-t">Renfort local {cote}. {r["sens"]}</div><div class="fx">{r["n"]}{N}Ø{r["d"]} '
                       f'<span class="eq">·</span><span class="r">x {C.fr(x0, 2)} à {C.fr(x1, 2)} · y {C.fr(y0, 2)} à {C.fr(y1, 2)}{N}m</span></div></div>')
    return C.fenetre_calcul(f"s{BLOC_DALLE}", "Dalle", "le treillis de chaque face : les quatre choix et leur poids total (le plus léger est retenu)",
                            _hyps(el, fer), lignes, detail, colonnes=("Treillis", "Poids", "Total", "", "État"))


def _fenetre_bande(el: Dict, fer: Dict, ef: Dict, i: int, nom: str) -> str:
    rb = next(x for x in ef["bandes"] if x["nom"] == nom)
    lignes, detail = [], ""
    for cote in ("inf", "sup"):
        bf = fer["faces"][cote]["bandes"][nom]
        g = bf["retenu"]["moteur"]["grandeurs"]
        lignes.append({"nom": "Inférieure" if cote == "inf" else "Supérieure", "sous": f"d{N}={N}{C.fr(g[f'd_{cote}'], 1)}{N}cm",
                       "formule": f"bande {_mm2(bf['retenu']['besoin_pm'])} · prolongement {_mm2(bf['retenu']['prol_pm'])}{N}mm²/m",
                       "valeur": f"{_mm2(bf['retenu']['necessaire_pm'])}{N}mm²/m", "limite": f"{_mm2(bf['As_place_pm'])}{N}mm²/m",
                       "taux": C.pct(bf["taux"]), "niveau": C.niveau(bf["taux"])})
        detail += (f'<div class="ck-g"><div class="ck-t">{"Inférieure" if cote == "inf" else "Supérieure"} — moteur poutre, b{N}={N}{C.fr(g["b"], 0)}'
                   f'{N}cm, h{N}={N}{C.fr(g["h"], 0)}{N}cm</div>'
                   f'<div class="fx">M <span class="eq">=</span><span class="r">{C.fr(bf["retenu"]["M"], 1)}{N}kNm</span></div>'
                   f'<div class="fx">A_s,req <span class="eq">=</span><span class="r">{_mm2(g[f"As_req_{cote}"])}{N}mm²</span></div>'
                   f'<div class="fx">A_s,min <span class="eq">=</span><span class="r">{_mm2(g[f"As_min_{cote}"])}{N}mm²</span></div>'
                   f'<div class="fx">A_s demandé <span class="eq">=</span><span class="r">{_mm2(g[f"besoin_{cote}"])}{N}mm² → '
                   f'{_mm2(g[f"besoin_{cote}"] * 100 / g["b"])}{N}mm²/m</span></div></div>')
    t = fer["tranchant"][nom]
    if t["tau"] is not None:
        lignes.append({"nom": "Effort tranchant", "sous": f"d{N}={N}{C.fr(t['d_v'] * 100, 1)}{N}cm", "formule": f"V à d{N}={N}{C.fr(t['V'], 1)}{N}kN",
                       "valeur": f"{C.fr(t['tau'], 3)}{N}N/mm²", "limite": f"{C.fr(t['tau_lim'], 3)}{N}N/mm²", "taux": C.pct(t["taux"]),
                       "niveau": C.niveau(t["taux"])})
    hb = fer["hauteur"][nom]
    lignes.append({"nom": "Hauteur", "formule": "h_u,min + y_G", "valeur": f"{C.fr(hb['h_min'], 1)}{N}cm", "limite": f"{C.fr(hb['h'], 0)}{N}cm",
                   "taux": C.pct(hb["taux"]), "niveau": C.niveau(hb["taux"])})
    sous = f"largeur {C.fr(rb['largeur'], 2)}{N}m, zone de renfort {C.fr(rb['zone'][1] - rb['zone'][0], 2)}{N}m, longueur {C.fr(rb['L'], 2)}{N}m"
    return C.fenetre_calcul(f"s{BLOC_BANDES + i}", nom, sous, _hyps(el, fer), lignes, detail)


def _fenetre_plots(el: Dict, fer: Dict) -> str:
    lignes = [{"nom": ln["rep"], "formule": C.esc(ln["libelle"]), "valeur": f"{ln['n']} × Ø{ln['d']}", "limite": f"{C.fr(ln['L'], 2)}{N}m",
               "taux": None, "niveau": "c", "etat_txt": ""} for ln in fer["bordereau"]["lignes"] if ln["famille"] == "C"]
    return C.fenetre_calcul(f"s{BLOC_PLOTS}", "Plots", "la cage des plots coulés en place", _hyps(el, fer), lignes, "",
                            colonnes=("Repère", "Barres", "Nombre", "Longueur", ""))


def fenetres(el: Dict, fer: Dict, ef: Dict) -> str:
    out = _fenetre_dalle(el, fer)
    for i, nom in enumerate(b["nom"] for b in ef["bandes"]):
        out += _fenetre_bande(el, fer, ef, i, nom)
    if el["grue"]["plots"]["type"] != "aucun":
        out += _fenetre_plots(el, fer)
    return out


def taux_max(fer: Dict) -> float:
    """Le plus grand taux du ferraillage (le résumé de la note)."""
    t = [_taux_treillis(fer["faces"][c]) for c in ("inf", "sup")]
    t += [fer["faces"][c]["bandes"][n]["taux"] for c in ("inf", "sup") for n in fer["faces"][c]["bandes"]]
    t += [x["taux"] for x in fer["tranchant"].values()] + [h["taux"] for h in fer["hauteur"].values()]
    return float(np.max(t)) if t else 0.0

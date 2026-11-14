# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du « Générateur SAF — Fondation de grue » (api_web/saf.py), sur le site.

Elle a LA PLANCHE des autres notes du site (le kit ``ndc_pdf.styles`` : la garde A4 portrait, puis des planches A4
paysage — le panneau de gauche et son dessin, deux colonnes de blocs numérotés, des bandes de verdict à la couleur de
l'état ; une planche qui déborde se poursuit sur une page suivante) et elle dit LES CHIFFRES DE L'ÉCRAN : elle ne
calcule rien — le calcul est celui de ``saf.calculer``, la synthèse celle de ``saf_resultat.synthese``, les dessins ceux
de ``dessin_saf.plan``.

Par fondation (cahier des charges § 9) :
- la planche « Plan et données » — le PLAN COTÉ (la dalle, le châssis, les surfaces d'appui numérotées, les plots, les
  bandes d'intégration, les cotes) ; les données (le projet, le châssis : son nom et son id dans la bibliothèque, ou
  « saisie » ; les appuis : réels ou rectangle, la forme, les dimensions avec leur origine, l'aire ; les plots ; les jeux
  de réactions ; le béton, l'épaisseur, les valeurs de k, la limite) ; les CAS DE CHARGE générés ; les BANDES
  d'intégration et leurs coordonnées ;
- la planche « Vérification rapide » — la CARTE de la contrainte de contact du cas le plus défavorable ; le tableau par
  k ; les contrôles ; les positions ;
- l'ANNEXE, quand une note de raideur du sol a été reprise (bloc Sol, « Reprendre ») : sa page de résultats, telle que le
  module « Rigidité du sol » l'a rendue (``sol.page_resultats``, gardée dans le projet).

Un champ vide reste vide : une ligne sans valeur ne s'imprime pas.

    sections(etat) -> [planche]        les planches (les dictionnaires que ``Style.planche`` dessine)
    pdf(etat, infos) -> bytes          infos = {nom, chantier, indice, date} de la note du site
    fichier(etat, infos) -> {"nom", "type": "application/pdf", "base64"}
"""
import base64
import os
import tempfile
import xml.etree.ElementTree as ET_
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ndc_pdf import data as ndc_data
from ndc_pdf.styles import Encre
from ndc_pdf.svg import peindre_svg
from saf.fondation_grue import geometrie as G
from saf.fondation_grue import projet as PR
from saf.fondation_grue import verification as V

from . import commun as C
from . import dessin_saf as DS
from . import saf_resultat as SR
from . import saf_saisie as SS

TITRE = "Fondation de grue"
MARGE_DESSIN = 2.0
COULEURS = {"p": "ok", "f": "att", "r": "ko", "n": "att", "x": "ko"}
MOTS = {"p": "Conforme", "f": "Conforme, à la limite", "r": "Non conforme", "n": "Données incomplètes", "x": "Calcul impossible"}
LEGENDE = "Vert : sous la limite   ·   Ocre : à la limite (95 à 100 %), ou à compléter   ·   Rouge : au-dessus de la limite"
ORIGINES = {"mesure": "≈ ", "calcule": "calc. "}
PLOTS = {"aucun": "aucun", "prefabrique": "préfabriqués", "coule": "coulés en place"}


def _n(x: float, nd: int = 2) -> str:
    return C.fr(x, nd).replace(C.NBSP, " ").replace(C.FINE, " ")


def _texte(h: str) -> str:
    """Un texte de l'écran (HTML court) en texte de la note."""
    import html as H
    import re
    return H.unescape(re.sub(r"<[^>]+>", "", h)).replace(C.NBSP, " ").replace(C.FINE, " ")


class Planche(Encre):
    """La planche des notes du site : elle se poursuit sur une page suivante quand elle déborde ; la couleur de sa
    pastille est celle de l'état de la fondation à l'écran."""
    planche_suite = True
    LEGENDE_GARDE = LEGENDE

    def coul_etat(self, sec):
        return {"ok": self.ok, "att": self.att, "ko": self.ko}[sec["couleur"]]

    def largeur_colonne(self) -> float:
        zx = self.margin + self.coupe_w + self.gap
        return (self.page[0] - self.margin - zx - self.col_gap * (self.n_cols - 1)) / self.n_cols


def hauteur_naturelle(svg: str, largeur: float) -> float:
    """La hauteur que le dessin prend quand il occupe toute la largeur du panneau."""
    vb = [float(t) for t in ET_.fromstring(svg).get("viewBox").split()]
    return vb[3] * (largeur - 2 * MARGE_DESSIN) / vb[2] + 2 * MARGE_DESSIN


def _dessin(svg: str):
    def dessin(d, x, y, w, h, S):
        peindre_svg(d.c, svg, x, y, w, h, marge=MARGE_DESSIN, riche=True)
    return dessin


def _v(label: str, valeur: str, unite: str = "", symbole: str = "") -> tuple:
    return ("v", label, symbole, valeur, unite)


# ------------------------------------------------------------------ la planche « Plan et données »
def _origine(platine: Optional[Dict], champ: str, texte: str) -> str:
    return ORIGINES.get(((platine or {}).get("origines") or {}).get(champ), "") + texte


def appuis_texte(ch: Dict) -> List[Tuple[str, str]]:
    """Les appuis, en lignes (libellé, valeur) : réels ou rectangle, la forme, les dimensions avec leur origine (« ≈ » :
    mesurée ; « calc. » : calculée), l'aire ; « saisie » quand les dimensions le sont."""
    p = ch["platine"]
    out = [("Appuis", "rectangle (saisie)" if ch["appuis"] == "rectangle" else "réels (la bibliothèque)")]
    if ch["appuis"] == "rectangle":
        out.append(("Surface d'appui", ch["dims"]))
    elif p["forme"] == "rectangle":
        out.append(("Surface d'appui", f"rectangle {_origine(p, 'a', _n(p['a']))} × {_origine(p, 'b', _n(p['b']))} m" + (" à 45°" if ch["angle"] == 45 else "")))
    elif p["forme"] == "cercle":
        out.append(("Surface d'appui", f"ronde Ø {_origine(p, 'diametre', _n(p['diametre']))} m — "
                    + ("polygone à 48 côtés" if ch["forme"] == "cercle" else f"carré de même aire, côté {_origine(p, 'cote_carre_meme_aire', _n(p['cote_carre'], 3))} m")))
    else:
        out.append(("Surface d'appui", f"octogone, {_origine(p, 'cote_sur_plats', _n(p['cote_sur_plats']))} m sur plats"))
    if p and p["excentrement"] and ch["appuis"] != "rectangle" or (ch["appuis"] == "rectangle" and p and p["excentrement"]):
        out.append(("Excentrement radial", f"{_origine(p, 'excentrement_radial', _n(p['excentrement'] * 1000, 0))} mm"))
    out.append(("Aire d'une surface", f"{_n(ch['aire'], 4)} m²"))
    return out


def _donnees(el: Dict, calc: Dict) -> List[Dict]:
    """Les blocs de données : le projet, la dalle et le sol, le châssis et ses appuis, les réactions."""
    m = calc["modele"]
    ch, g, d, p = m["chassis"], el["grue"], el["dalle"], el["projet"]
    blocs = []
    items = [_v(lab, p[k]) for k, lab in (("projet", "Projet"), ("grue", "Grue"), ("creation", "Date de création"), ("mise_a_jour", "Date de mise à jour"),
                                          ("indice", "Indice")) if p[k]]
    if items:
        blocs.append({"titre": "Projet", "items": items, "verdicts": []})
    items = [_v("Dalle", "rectangle" if d["forme"] == "rectangle" else f"contour de {len(m['contour'])} sommets"),
             _v("Dimensions", f"{_n(d['Lx'])} × {_n(d['Ly'])}" if d["forme"] == "rectangle" else f"aire {_n(G.aire(m['contour']), 2)}",
                "m" if d["forme"] == "rectangle" else "m²"),
             _v("Épaisseur", _n(d["h"]), "m", "h"), _v(f"Béton {d['beton']}", _n(m["E"], 0), "MPa", "E"),
             _v("Raideurs de sol", " ; ".join(C.nb(x["k"]) for x in el["sol"]["k"]), "MN/m³", "k"),
             _v("Exportées vers SCIA", " ; ".join(C.nb(k) for k in PR.k_scia(el)) or "aucune", "MN/m³" if PR.k_scia(el) else ""),
             _v("Contrainte de contact admise", f"{C.nb(el['limite'])} kg/cm² = {_n(el['limite'] * V.KG_CM2, 1)}", "kPa")]
    if d["forme"] == "contour":
        items.append(("pr", "Sommets (x ; y) en m : " + " · ".join(f"({C.nb(round(x, 3))} ; {C.nb(round(y, 3))})" for x, y in m["contour"]) + "."))
    blocs.append({"titre": "Dalle et sol", "items": items, "verdicts": []})
    # (des valeurs courtes : une valeur trop longue écrase son libellé — le constructeur a sa ligne)
    items = [_v("Châssis", ch["nom"] if ch["id"] else "saisie")]
    if ch["id"]:
        items += [_v("Constructeur", ch["constructeur"]), _v("Id dans la bibliothèque", ch["id"])]
    items.append(_v("Entraxes des appuis", f"{_n(ch['a'])} × {_n(ch['b'])}", "m", "a × b"))
    items += [_v(lab, val) for lab, val in appuis_texte(ch)]
    if ch["impose"]:
        items.append(("pr", "Platine non indiquée dans la bibliothèque : les dimensions de la surface d'appui sont saisies."))
    if ch["note_fondation"]:
        items.append(("pr", f"Note de la bibliothèque : {ch['note_fondation']}"))
    pl = g["plots"]
    items.append(_v("Plots sous les pieds", PLOTS[pl["type"]]))
    if pl["type"] != "aucun":
        items.append(_v("Plots : a × b × hauteur", f"{_n(pl['a'])} × {_n(pl['b'])} × {_n(pl['h'])}", "m"))
    items += [_v("Position du centre", f"{_n(g['ex'])} ; {_n(g['ey'])}", "m", "e_x ; e_y"), _v("Angle", C.nb(round(g["beta"], 2), 2), "°", "β"),
              _v("Distance mini plaque–bord", _n(g["dmin"]), "m")]
    blocs.append({"titre": "Grue : châssis, appuis, plots", "items": items, "verdicts": []})
    return blocs


def _reactions(el: Dict, calc: Dict, largeur: float) -> Dict:
    """Les jeux de réactions saisis (P1 à P4, kN), puis les cas de charge générés : cas | jeu | P1 | P2 | P3 | P4."""
    m = calc["modele"]
    wp = 38.0
    cols = [("JEU", largeur - 4 * wp, "left")] + [(f"P{i + 1} [kN]", wp, "right") for i in range(4)]
    lignes = [[("lab", f"{i + 1}" + (f" — {j['nom']}" if j["nom"] else ""), False)] + [("num", "" if v is None else C.nb(v)) for v in j["P"]]
              for i, j in enumerate(el["reactions"])]
    items = [("pr", "P1 en haut à gauche, P2 en haut à droite, P3 en bas à droite, P4 en bas à gauche : sens horaire vu du dessus, châssis non tourné."),
             ("tab", cols, lignes)]
    if m["cas"]:
        wc, wl = 30.0, 34.0
        cols2 = [("CAS", wc, "left"), ("SAF", wl, "left"), ("JEU", largeur - wc - wl - 4 * wp, "left")] + [(f"P{i + 1}", wp, "right") for i in range(4)]
        lignes2 = [[("lab", c["nom"], True), ("ref", c["lc"]), ("ref", str(c["jeu_saisi"]))] + [("num", C.nb(v)) for v in c["R"]] for c in m["cas"]]
        items += [("s", f"Cas de charge générés : {len(m['cas'])}"),
                  ("pr", "Chaque jeu est placé dans toutes les positions de la grue (rotations de 90° et miroirs) ; les doublons sont retirés. "
                         "LC1 est le poids propre."),
                  ("tab", cols2, lignes2)]
    else:
        items.append(("pr", "Aucun jeu de réactions complet : aucun cas de charge n'est généré."))
    return {"titre": "Réactions et cas de charge", "items": items, "verdicts": []}


def _bandes(el: Dict, calc: Dict, largeur: float) -> Dict:
    m = calc["modele"]
    wn, wl = 34.0, 46.0
    wc = (largeur - wn - wl) / 2
    cols = [("BANDE", wn, "left"), ("DÉBUT (x ; y) [m]", wc, "left"), ("FIN (x ; y) [m]", wc, "left"), ("LARGEUR [m]", wl, "right")]
    lignes = [[("lab", b["nom"], True), ("lab", f"{_n(b['debut'][0], 3)} ; {_n(b['debut'][1], 3)}"), ("lab", f"{_n(b['fin'][0], 3)} ; {_n(b['fin'][1], 3)}"),
               ("num", _n(b["largeur"]))] for b in m["bandes"]]
    return {"titre": "Bandes d'intégration", "items": [
        ("pr", "Quatre bandes par les centres des surfaces d'appui, le long des axes du châssis, limitées au contour de la dalle : BX1 par P1 et P2, "
               "BX2 par P4 et P3, BY1 par P1 et P4, BY2 par P2 et P3. Dans SCIA : dalle D1, « Constant symmetric »."),
        ("tab", cols, lignes)], "verdicts": []}


def planche_donnees(el: Dict, calc: Dict, s: Dict, style: Planche) -> Dict:
    largeur = style.largeur_colonne()
    m = calc["modele"]
    verifs = _donnees(el, calc) + [_reactions(el, calc, largeur), _bandes(el, calc, largeur)]
    svg = DS.plan(el, m, None, interactif=False, ident="note_plan")
    ch, g = m["chassis"], el["grue"]
    blocs = [("DALLE", [("Dalle", "", SS_dims(el), ""), ("Béton", "", el["dalle"]["beton"], "")]),
             ("CHÂSSIS", [("Entraxes", "", f"{_n(ch['a'])} × {_n(ch['b'])}", "m"), ("Appuis", "", _texte(ch["dims"]), ""),
                          ("Position", "", f"{_n(g['ex'])} ; {_n(g['ey'])}", "m"), ("Angle", "β", C.nb(round(g["beta"], 2), 2), "°")])]
    return _planche(el, s, "Plan et données", "PLAN COTÉ — COTES EN m", svg, blocs, verifs, style)


def SS_dims(el: Dict) -> str:
    return _texte(SS.dimensions(el))


# ------------------------------------------------------------------ la planche « Vérification rapide »
def _verdict(el: Dict, s: Dict) -> Dict:
    if s.get("r_vue") is None:
        return {"texte": s["phrase"], "etat": "ko"}
    signe = "≤" if s["taux"] <= 1 else ">"
    quoi = "Poids propre seul" if s["pp"] else MOTS[s["niveau"]]
    ou = "les réactions sont à saisir" if s["pp"] else f"cas {s['cas']}, k = {C.nb(s['k'])} MN/m³"
    return {"texte": f"{quoi} — σ max = {_n(s['sigma'], 1)} kPa ({_n(s['sigma'] / V.KG_CM2, 2)} kg/cm²) {signe} {_n(s['limite'], 1)} kPa "
                     f"({C.nb(el['limite'])} kg/cm²) · {ou}", "etat": COULEURS[s["niveau"]], "cale": f"{C.pct(s['taux'])} %".replace(C.NBSP, " ")}


def _par_k(el: Dict, calc: Dict, s: Dict, largeur: float) -> Dict:
    scia = set(PR.k_scia(el))
    w = (38.0, 44.0, 36.0, 40.0, 40.0)
    cols = [("k [MN/m³]", w[0], "left"), ("σ MAX [kPa]", w[1], "right"), ("CAS", largeur - sum(w), "left"), ("w MAX [mm]", w[2], "right"),
            ("DÉCOLLÉ", w[3], "right"), ("SCIA", w[4], "right")]
    lignes = []
    for r in calc["par_k"]:
        gov = r["k"] == s["k"]
        lignes.append([("lab", C.nb(r["k"]), gov), ("eta", _n(r["sigma_max"], 1), r["sigma_max"] <= s["limite"], gov), ("ref", r["cas_max"]),
                       ("num", _n(r["w_max"], 2)), ("num", f"{_n(r['decolle'], 1)} %"), ("num", "oui" if r["k"] in scia else "—")])
    sm = V.sigma_moyen(calc["modele"])
    note = (f"σ moyen = (dalle {_n(sm['dalle'], 0)}" + (f" + plots {_n(sm['plots'], 0)}" if sm["plots"] else "")
            + (f" + réactions {_n(sm['reactions'], 0)}" if sm["reactions"] else "") + f") kN / {_n(sm['aire'], 1)} m² = {_n(sm['sigma'], 1)} kPa — un repère.")
    return {"titre": "Par raideur de sol", "items": [("tab", cols, lignes, note), ("k", 0)], "verdicts": [_verdict(el, s)]}


def _controles(el: Dict, calc: Dict, s: Dict) -> Dict:
    m = calc["modele"]
    ctl, g = m["controles"], el["grue"]
    items, verdicts = [], []

    def verdict(texte, bon, grave=True):
        items.append(("k", len(verdicts)))
        verdicts.append({"texte": texte, "etat": "ok" if bon else ("ko" if grave else "att")})
    dedans = all(ctl["plaques_dedans"])
    verdict("Les quatre surfaces d'appui sont dans la dalle." if dedans else "Une surface d'appui sort de la dalle.", dedans)
    if dedans:
        verdict(f"Distance mini plaque–bord : {_n(ctl['distance'], 3)} m {'≥' if ctl['distance_ok'] else '<'} {_n(g['dmin'])} m"
                + ("." if ctl["distance_ok"] else " — sous le minimum."), ctl["distance_ok"], grave=False)
    if m["plots"]:
        bon = ctl["plots_ok"]
        verdict("Les plots sont dans la dalle, ne se chevauchent pas, et chaque surface d'appui est sur son plot." if bon
                else "Un plot sort de la dalle, en chevauche un autre, ou ne porte pas toute sa surface d'appui.", bon)
    if s.get("r_vue") is not None:
        bon = s["equilibre"] <= V.EQUILIBRE
        verdict(f"Équilibre : la réaction du sol vaut les charges, à {_n(s['equilibre'] * 100, 4)} % près (cas le plus écarté).", bon)
    return {"titre": "Contrôles", "items": items, "verdicts": verdicts}


def _positions(el: Dict, calc: Dict, s: Dict, largeur: float) -> Dict:
    r = next(x for x in calc["par_k"] if x["k"] == s["k"])
    w = (30.0, 44.0, 40.0, 40.0)
    cols = [("CAS", w[0], "left"), ("P1 / P2 / P3 / P4 [kN]", largeur - sum(w), "left"), ("σ MAX [kPa]", w[1], "right"), ("w MAX [mm]", w[2], "right"),
            ("DÉCOLLÉ", w[3], "right")]
    lignes = [[("lab", c["nom"], c["nom"] == r["cas_max"]), ("ref", " / ".join(C.nb(v) for v in c["R"])),
               ("eta", _n(c["sigma_max"], 1), c["sigma_max"] <= s["limite"], c["nom"] == r["cas_max"]), ("num", _n(c["w_max"], 2)),
               ("num", f"{_n(c['decolle'], 1)} %")] for c in r["cas"]]
    return {"titre": f"Positions, k = {C.nb(s['k'])} MN/m³", "items": [("tab", cols, lignes)], "verdicts": []}


def planche_verification(el: Dict, calc: Dict, s: Dict, style: Planche) -> Dict:
    largeur = style.largeur_colonne()
    m = calc["modele"]
    hyp = {"titre": "Hypothèses", "items": [("pr", _texte(h) + ".") for h in SR.hypotheses(el, calc)], "verdicts": []}
    verifs = [hyp]
    champ, titre = None, "PLAN"
    if s.get("r_vue") is not None:
        verifs.append(_par_k(el, calc, s, largeur))
        verifs.append(_controles(el, calc, s))
        if not s["pp"]:
            verifs.append(_positions(el, calc, s, largeur))
        # la carte : la contrainte de contact du cas le plus défavorable, au k qui gouverne (jamais la vue de l'écran)
        r = next(x for x in calc["par_k"] if x["k"] == s["k"])
        i = [c["nom"] for c in r["cas"]].index(r["cas_max"])
        sigma, w = r["champs"][i]
        c = r["cas"][i]
        champ = {"mode": "sigma", "dalle": r["dalle"], "valeurs": sigma, "decolle": (w < 0) & r["dalle"].on0, "maxi": c["sigma_max"], "x": c["x_max"],
                 "y": c["y_max"], "limite": s["limite"]}
        titre = ("CONTRAINTE DE CONTACT — POIDS PROPRE" if s["pp"] else f"CONTRAINTE DE CONTACT — CAS {c['nom']}") + f" · k = {C.nb(s['k'])} MN/m³"
    else:
        verifs.append(_controles(el, calc, s))
    svg = DS.plan(el, m, champ, interactif=False, carte="vecteur", ident="note_carte")
    o = el["options"]
    blocs = [("MODÈLE", [("Maille", "", _n(m["maille"]), "m"), ("Module", "E", _n(m["E"], 0), "MPa"), ("Poisson", "ν", C.nb(m["nu"]), ""),
                         ("Pasternak", "C_2/k", f"{C.nb(o['c2x'])} ; {C.nb(o['c2y'])}", "m")]),
             ("CHARGES", [("Poids propre", "", _n(m["gamma"] * m["h"], 2), "kN/m²")]
              + ([("Plots", "", _n(m["gamma"] * m["plots"]["hauteur"], 2), "kN/m²")] if m["plots"] else [])
              + [("Cas générés", "", str(len(m["cas"])), ""), ("Limite", "", _n(s["limite"], 1), "kPa")])]
    return _planche(el, s, "Vérification rapide", titre, svg, blocs, verifs, style)


# ------------------------------------------------------------------ l'annexe : la page de résultats du module de raideur du sol
def planche_annexe(el: Dict, s: Dict, style: Planche) -> Optional[Dict]:
    r = el["sol"].get("reprise")
    page = (r or {}).get("page")
    if not page:
        return None
    largeur = style.largeur_colonne()
    verifs = []
    for b in page.get("blocs") or []:
        wv = 96.0
        cols = [("", (largeur - wv) * 0.42, "left"), ("", (largeur - wv) * 0.58, "left"), ("k", wv, "right")]
        lignes = [[("lab", str(li[0]), False), ("ref", str(li[1])), ("num", str(li[2]), True)] for li in b.get("lignes") or []]
        verifs.append({"titre": str(b.get("titre") or ""), "items": ([("pr", str(b["sous"]) + ".")] if b.get("sous") else []) + [("tab", cols, lignes)],
                       "verdicts": []})
    verifs.append({"titre": "Remarques du module « Rigidité du sol »", "items": [("pr", str(t)) for t in page.get("messages") or []], "verdicts": []})
    blocs = [("NOTE REPRISE", [("Note", "", str(r.get("note") or "sans nom"), "")] + ([("Chantier", "", str(r["chantier"]), "")] if r.get("chantier") else [])
              + [("Fondation", "", str(page.get("fondation") or ""), ""), ("Méthode", "", str(page.get("methode") or ""), "")]),
             ("RAIDEUR", [("Résultat", "", str(page.get("resume") or "").replace("k = ", ""), ""), ("Moteur", "", str(page.get("moteur") or ""), "")])]
    if page.get("phrase"):
        verifs.insert(0, {"titre": "Données de la fondation reprise", "items": [("pr", str(page["phrase"]) + ".")], "verdicts": []})
    sec = _planche(el, s, "Annexe — raideur du sol", "MODULE « RIGIDITÉ DU SOL »", None, blocs, verifs, style)
    # (l'annexe ne dit pas l'état de la fondation de grue : c'est la page d'un autre module)
    sec["etat"], sec["couleur"] = "", "ok"
    return sec


def _planche(el: Dict, s: Dict, section: str, titre_dessin: str, svg: Optional[str], blocs: List, verifs: List[Dict], style: Planche) -> Dict:
    for i, v in enumerate(verifs, 1):
        v["num"] = i
    reserve = 20.0 + 7.6 * 1.42 * max((len(rows) for _t, rows in blocs), default=0) + 6.0
    sec = {"poutre": el.get("nom") or "Fondation", "section": section, "beton": el["dalle"]["beton"], "acier": SS_dims(el), "etat": MOTS[s["niveau"]],
           "couleur": COULEURS[s["niveau"]], "titre_coupe": titre_dessin, "blocs": blocs, "reserve": reserve, "verifs": verifs}
    if svg is not None:
        sec["dessin"] = _dessin(svg)
        sec["coupe_h"] = hauteur_naturelle(svg, style.coupe_w)
    else:
        sec["dessin"] = lambda d, x, y, w, h, S: None
        sec["coupe_h"] = 4.0
    return sec


# ------------------------------------------------------------------ le document
def sections(etat: Dict, style: Optional[Planche] = None) -> List[Dict]:
    """Les planches de la note, dans l'ordre des fondations : le plan et les données, la vérification rapide, puis
    l'annexe de la raideur du sol quand une note a été reprise."""
    from . import saf as M                                        # (le calcul est celui de l'écran)
    style = style or Planche()
    out = []
    for el in etat["fondations"]:
        calc = M.calculer(el)
        s = SR.synthese(el, calc)
        out += [planche_donnees(el, calc, s, style), planche_verification(el, calc, s, style)]
        annexe = planche_annexe(el, s, style)
        if annexe:
            out.append(annexe)
    return out


def nom_note(etat: Dict, infos: Optional[Dict]) -> str:
    """Le nom de la note : celui que la note du site porte ; à défaut, le projet de la fondation quand elle est seule,
    son nom, le titre du module sinon."""
    nom = str((infos or {}).get("nom") or "").strip()
    if nom:
        return nom
    fs = etat.get("fondations") or []
    if len(fs) == 1:
        return str(fs[0]["projet"].get("projet") or fs[0].get("nom") or "").strip() or TITRE
    return TITRE


def infos_doc(etat: Dict, infos: Optional[Dict]) -> Dict:
    """Les « infos » de la garde depuis celles de la note du site : la correspondance des autres notes (projet = le
    chantier, à défaut le nom de la note ; partie = le nom de la note quand il y a un chantier, sinon le titre)."""
    infos = infos or {}
    nom, chantier = nom_note(etat, infos), str(infos.get("chantier") or "").strip()
    out = {"nom_projet": chantier or nom, "partie": nom if chantier else TITRE, "indice": str(infos.get("indice") or "").strip() or "A"}
    if str(infos.get("date") or "").strip():
        out["date"] = str(infos.get("date")).strip()
    return out


def document(etat: Dict, infos: Optional[Dict], chemin: str):
    """Le document construit (pas encore écrit) : la garde et les planches. ``d.warnings`` : ce qui déborde encore."""
    from . import saf as M
    etat = M.normaliser(etat)
    style = Planche()
    doc_meta = ndc_data.construire_doc(infos_doc(etat, infos), date_defaut=datetime.today().strftime("%d/%m/%Y"))
    return style.build(chemin, sections=sections(etat, style), doc=doc_meta)


def pdf(etat: Dict, infos: Optional[Dict] = None) -> bytes:
    """La note PDF (octets). Ce qui déborderait encore est REFUSÉ avec un message : jamais une note tronquée."""
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_saf_")
    os.close(fd)
    try:
        d = document(etat, infos, chemin)
        if d.warnings:
            raise ValueError("La note ne tient pas sur sa planche (" + " ; ".join(d.warnings) + ") : elle ne sort pas tronquée. À signaler "
                             "au bureau ; l'écran garde tout le calcul.")
        d.save()
        with open(chemin, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle)."""
    infos = dict(infos or {})
    octets = pdf(etat, infos)
    return {"nom": C.nom_fichier(f"Note de calcul - {nom_note(etat, infos)}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}

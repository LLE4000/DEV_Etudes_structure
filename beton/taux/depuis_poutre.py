# -*- coding: utf-8 -*-
"""Taux d'armature — RECEVOIR une poutre dimensionnée dans « Poutres béton armé » (docs/site/LOT_Q_QUATRE_MODULES.md
§ 3.4 et § 6.2) : ``convertir(poutre_ba, longueurs)`` rend une poutre de CE module, toute préparée — on y règle ensuite
où s'arrêtent les barres, les zones d'étriers, les retours. Fonction PURE : elle ne lit ni l'heure ni un fichier, et
ne modifie pas ce qu'elle reçoit.

La règle :
- b, h et l'enrobage sont ceux de la poutre (un enrobage nul y est lu 3 cm : ``beton.poutre.geometrie.enrobage_de_calcul``) ;
- chaque SECTION de longueur l_i devient un TRONÇON [x_i ; x_i + l_i], dans l'ordre ; L = Σ l_i ;
- pour chaque lit (face, rang) : si les sections qui l'ont portent le même Ø, les barres COMMUNES à toutes les sections
  (le plus petit nombre ; 0 si une section n'a pas ce lit) deviennent des barres FILANTES, et le surplus de chaque
  section un RENFORT sur son tronçon — les tronçons voisins de même surplus sont réunis ; si le Ø change d'une section
  à l'autre, le lit donne des barres par tronçon (les tronçons voisins identiques réunis) ;
- les barres LATÉRALES de même (leur nombre par face et leur Ø sont ceux que la poutre calcule : imposés, ou la règle
  automatique des armatures de peau, ``beton.poutre.geometrie.laterales``) ;
- chaque COMPOSITION d'étriers distincte devient un type (« E1 », « E2 »… dans l'ordre où elles apparaissent) : un
  cadre de toute la largeur → « cadre » ; un cadre entre deux barres → « cadre intérieur », de la largeur hors tout que
  la poutre calcule (``beton.poutre.geometrie.emprise_etrier``) ; une épingle → « épingle » ;
- chaque section devient une ZONE, à son pas.

Les fonctions du moteur des poutres sont APPELÉES, pas recopiées : la largeur d'un cadre intérieur et la position des
barres sont celles de sa coupe et de sa note.
"""
from typing import List, Optional

from beton.poutre import entree as EP
from beton.poutre import geometrie as GP

from . import entree as EN

OU_SAISIR = ("saisir la « Longueur (m) » de chaque section dans « Poutres béton armé » (rubrique « Sections et sollicitations », "
             "option « Taux d'armature »)")


def _longueurs(longueurs, sections: list) -> List[float]:
    """La longueur de chaque section (m), dans l'ordre : toutes données, toutes strictement positives — sinon le refus
    dit où les saisir."""
    if not isinstance(longueurs, (list, tuple)):
        raise ValueError(f"Longueurs des sections manquantes : {OU_SAISIR}.")
    if len(longueurs) != len(sections):
        raise ValueError(f"Longueurs des sections : {len(longueurs)} donnée(s) pour {len(sections)} section(s) — {OU_SAISIR}.")
    out = []
    for s, l in zip(sections, longueurs):
        if l is None or isinstance(l, bool) or not isinstance(l, (int, float)) or l != l or l in (float("inf"), float("-inf")):
            raise ValueError(f"Section {s['nom']} : sa longueur manque — {OU_SAISIR}.")
        if l <= 0:
            raise ValueError(f"Section {s['nom']} : longueur nulle ({EN.nombre_fr(float(l))} m) — {OU_SAISIR}.")
        out.append(float(l))
    return out


def _barre(famille: str, lit: Optional[int], n: int, d: int, de=None, a=None) -> dict:
    return {"famille": famille, "lit": lit, "n": int(n), "d": int(d), "de": de, "a": a, "retours": 0, "l_retour": None}


def _suites(valeurs: list) -> list:
    """Les suites de valeurs voisines ÉGALES : [(i0, i1, valeur)] (i1 compris)."""
    out = []
    for i, v in enumerate(valeurs):
        if out and out[-1][2] == v:
            out[-1][1] = i
        else:
            out.append([i, i, v])
    return [tuple(x) for x in out]


def _lignes(famille: str, lit: Optional[int], par_section: list, xs: list) -> list:
    """Les lignes de barres d'UN lit : ``par_section`` = [(n, Ø) ou None (la section n'a pas ce lit)], ``xs`` = les
    abscisses des limites des tronçons (n + 1 valeurs)."""
    presents = [v for v in par_section if v]
    if not presents:
        return []
    out = []
    diametres = {d for _, d in presents}
    if len(diametres) == 1:
        d = diametres.pop()
        ns = [v[0] if v else 0 for v in par_section]
        commun = min(ns)
        if commun > 0:
            out.append(_barre(famille, lit, commun, d))
        for i0, i1, surplus in _suites([n - commun for n in ns]):
            if surplus > 0:
                out.append(_barre(famille, lit, surplus, d, xs[i0], xs[i1 + 1]))
    else:
        for i0, i1, v in _suites(par_section):
            if v:
                out.append(_barre(famille, lit, v[0], v[1], xs[i0], xs[i1 + 1]))
    return out


def convertir(poutre_ba: dict, longueurs, date: str = "") -> dict:
    """Une poutre du module « Poutres béton armé » (le format de ``beton.poutre.entree`` : elle est normalisée ici) et la
    longueur de chacune de ses sections (m, dans l'ordre) → une poutre de ce module, NORMALISÉE, son ``origine``
    renseignée ({"module": "poutre", "nom", "date"} ; ``date`` : le jour de l'envoi, « JJ/MM/AAAA », donné par
    l'appelant). ``ValueError`` en français si une longueur manque ou est nulle, ou si la poutre sort du domaine de ce
    module (un pas de plus de 50 cm, une section de moins de 10 cm…)."""
    pb = EP.normaliser(poutre_ba)
    sections = pb["sections"]
    ls = _longueurs(longueurs, sections)
    xs = [0.0]
    for l in ls:
        xs.append(round(xs[-1] + l, 6))
    c = GP.enrobage_de_calcul(pb["enrobage"])
    barres = []
    for face in ("inf", "sup"):
        for j in range(max(len(s[face]) for s in sections)):
            barres += _lignes(face, j + 1, [(s[face][j]["n"], s[face][j]["d"]) if j < len(s[face]) else None for s in sections], xs)
    # les latérales : par face, le nombre et le Ø que la poutre calcule pour chaque section
    laterales = []
    for s in sections:
        d_et = GP.diametre_etrier(s["etriers"])
        di = GP.distances_lits(s["inf"], c, d_et, pb["jeu_premier_lit"], pb["jeu_entre_lits"])
        ds = GP.distances_lits(s["sup"], c, d_et, pb["jeu_premier_lit"], pb["jeu_entre_lits"])
        lat = GP.laterales(s, pb, pb["h"], di[0], ds[0])
        laterales.append((lat["n"], lat["d"]) if lat["n"] else None)
    barres += _lignes("peau", None, laterales, xs)
    # les étriers : un type par composition distincte, une zone par section
    types, noms, zones = [], {}, []
    for i, s in enumerate(sections):
        d_et = GP.diametre_etrier(s["etriers"])
        xs_inf1 = GP.abscisses_lit(s["inf"][0], pb["b"], c, d_et)
        lignes = []
        for e in s["etriers"]:
            emprise = GP.emprise_etrier(e, xs_inf1, s["inf"][0]["d"], pb["b"], c)
            if e["type"] == "epingle":
                lignes.append({"forme": "epingle", "d": int(e["d"]), "largeur": None})
            elif emprise["complet"]:
                lignes.append({"forme": "cadre", "d": int(e["d"]), "largeur": None})
            else:
                lignes.append({"forme": "cadre_interieur", "d": int(e["d"]), "largeur": round(emprise["x1"] - emprise["x0"], 3)})
        cle = tuple((li["forme"], li["d"], li["largeur"]) for li in lignes)
        if cle not in noms:
            noms[cle] = f"E{len(noms) + 1}"
            types.append({"nom": noms[cle], "lignes": lignes, "cadre_talon": False})
        zones.append({"de": xs[i], "a": xs[i + 1], "s": float(s["pas"]), "type": noms[cle]})
    p = EN.nouvelle_poutre(pb["nom"][:EN.NOM_MAX])
    p.update({"forme": "rectangulaire", "b": float(pb["b"]), "h": float(pb["h"]), "L": xs[-1], "enrobage": float(c), "barres": barres,
              "types_etriers": types, "zones": zones,
              "origine": {"module": "poutre", "nom": pb["nom"][:EN.NOM_MAX], "date": str(date or "")}})
    try:
        return EN.normaliser(p)
    except ValueError as e:
        raise ValueError(f"La poutre « {pb['nom']} » ne peut pas être détaillée telle quelle dans « Taux d'armature » : {e}.") from None

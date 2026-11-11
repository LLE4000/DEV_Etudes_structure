# -*- coding: utf-8 -*-
"""Fondations de grues — le MOTEUR : ``verifier(fondation)`` rend tout ce que l'écran, la note et l'export montrent.

Il ne contient AUCUNE formule de dimensionnement : les valeurs viennent de ``classeur.py`` (les fiches du bureau, mot
pour mot), les diagrammes de ``statique.py`` (la poutre sur deux appuis), les écarts de ``ecarts.py``, l'écriture des
formules de ``formules.py``. Ce fichier ASSEMBLE et NOMME. Bibliothèque standard seulement.

    nouvelle_fondation(type) / normaliser(f) / exemple(cle)      (entree.py)
    verifier(fondation) -> dict

SORTIE (JSON-sérialisable, nombres non arrondis) :
    {"reference", "classeur" (BASE, VAR4 ou VAR8 : la fiche appliquée), "fondation" (normalisée), "materiaux",
     "valeurs": {clé: valeur}, "cellules": {clé: cellule},          # tout le relevé de la fiche
     "blocs": [{"cle", "nom", "piece", "sollicitations", "verifications", "informations"}],
     "verifications": [...],                                         # les mêmes, à plat, avec "bloc" et "etat"
     "elements": [{"cle", "nom", "positions": [...], "enveloppe", "extremes"}],   # les diagrammes
     "tableau_cas": [...],                                           # « semelles8 » : les 48 colonnes par sous-cas
     "controles": [...],                                             # contrôles de géométrie de la fiche BASE
     "ecarts": [...],                                                # écarts signalés (ecarts.py)
     "armatures": [...], "pieces": [...],                            # ce que l'export transmet
     "statut": "ok" | "warn" | "nok", "taux_max", "gouvernant": {...}}

États : « ok » (OK du classeur), « warn » (LIMITE, A VERIFIER, zone 3 de cisaillement), « nok » (PAS OK, au-delà de
la zone 3), « info » (ligne sans verdict dans le classeur).
"""
from . import classeur as K
from . import donnees as D
from . import ecarts as X
from . import entree as E
from . import formules as F
from . import statique as S
from .entree import exemple, normaliser, nouvelle_fondation  # noqa: F401 — l'API

REFERENCE = "classeurs du bureau « Fondation grue tour sur pieux » (BASE, GWW, variantes 4 et 8 pieux) relus le 11/11/2026"
OK, WARN, NOK, INFO = "ok", "warn", "nok", "info"
ETATS = {K.OK: OK, K.LIMITE: WARN, K.A_VERIFIER: WARN, K.PAS_OK: NOK}
ETATS_ZONE = {"ZONE 1": OK, "ZONE 2": OK, "ZONE 3": WARN, "> ZONE 3": NOK}
N_ENVELOPPE = 90                 # points de l'enveloppe d'un grand tableau (dessin)


def fusion(*etats: str) -> str:
    """« nok » si un état l'est, sinon « warn » si un l'est, sinon « ok » (les « info » ne comptent pas)."""
    if NOK in etats:
        return NOK
    return WARN if WARN in etats else OK


# ------------------------------------------------------------------ diagrammes
def _position(cle: str, titre: str, texte: str, poutre: S.Poutre, charges: list) -> dict:
    d = poutre.decrire()
    d.update({"cle": cle, "titre": titre, "texte": texte, "charges": charges})
    return d


def _element(cle: str, nom: str, positions: list, poutres: list, repere: dict) -> dict:
    env = S.enveloppe(poutres)
    return {"cle": cle, "nom": nom, "positions": positions, "enveloppe": env, "repere": repere,
            "extremes": _extremes_enveloppe(env)}


def _extremes_enveloppe(env: dict) -> dict:
    def pic(serie, plus_grand):
        i = (max if plus_grand else min)(range(len(env["x"])), key=lambda k: env[serie][k])
        return {"valeur": env[serie][i], "x": env["x"][i]}

    return {"M_max": pic("M_max", True), "M_min": pic("M_min", False), "V_max": pic("V_max", True),
            "V_min": pic("V_min", False)}


def _elements_un_pied(cle: str, nom: str, L: float, e: float, P: float, pp: float) -> dict:
    p = S.poutre_un_pied(L, e, P, pp)
    cm = f"{e * 100:g} cm"
    charges = [{"nom": "pied de la grue", "x": 0.0, "P": P}]
    return _element(cle, nom, [
        _position("A", "CAS A", f"pieu A rentré de e = {cm} : le pied de la grue est en porte-à-faux", p["A"], charges),
        _position("B", "CAS B", f"pieu A sorti de e = {cm} : portée L + e", p["B"], charges)],
        [p["A"], p["B"]], {"appuis_nominaux": [0.0, L], "longueur": L, "e": e})


def _elements_deux_pieux(cle: str, nom: str, Lp: float, s: float, e: float, P: float, pp: float, c: float,
                         nom_charge: str) -> dict:
    p = S.poutre_deux_pieux(Lp, s, e, P, pp, c)
    cm = f"{e * 100 / 2:g} cm"

    def charges(x):
        return [{"nom": nom_charge, "x": x, "P": P, "c": c}]

    return _element(cle, nom, [
        _position("A", "CAS A", f"chaque pieu rentré de e/2 = {cm} (portée s − e), charge décalée de e/2", p["A"],
                  charges(-e / 2)),
        _position("B", "CAS B", f"chaque pieu sorti de e/2 = {cm} (portée s + e), charge au milieu", p["B"], charges(0.0))],
        [p["A"], p["B"]], {"appuis_nominaux": [-s / 2, s / 2], "longueur": Lp, "e": e})


TEXTES_SOUS_CAS = {
    "A.a": "pieux de la semelle A décalés de e : l'appui A recule (portée + e)",
    "A.b": "pieux de l'appui B décalés de e : l'appui B recule (portée + e)",
    "B.a": "châssis posé décalé de + e (vers l'appui B)",
    "B.b": "châssis posé décalé de − e (vers l'appui A)",
}


def _element_support(f: dict, r: K.Releve) -> dict:
    """La poutre support : quatre positions de tolérance ; pour chacune, l'enveloppe de ses colonnes (un cas de la
    grue × une orientation) et la colonne qui donne son plus grand moment. Les colonnes d'un sous-cas ont la même
    géométrie : leur enveloppe se calcule directement, avec les RA du relevé."""
    Lc, a, b, e = f["chassis"] / 100, f["dist_a"] / 100, f["dist_b"] / 100, f["e_support"] / 100
    pp = r["pp.support"]
    positions, gouvernantes = [], []
    for k in K.SOUS_CAS:
        t = r.tableaux[k]
        modele = S.poutre_support(Lc, a, b, e, k, 0.0, 0.0, pp)
        x1, x2 = modele.ponctuelles[0][0], modele.ponctuelles[1][0]
        x0, L = modele.x0, modele.x1 - modele.x0
        xs = sorted({x0 + L * i / N_ENVELOPPE for i in range(N_ENVELOPPE + 1)} | {x1, x2})
        env = {"x": [], "M_max": [], "M_min": [], "V_max": [], "V_min": []}
        for x in xs:
            u = x - x0
            ms, vs = [], []
            for col in t["colonnes"]:
                m = col["RA"] * u - pp * u * u / 2
                v = col["RA"] - pp * u
                vg = v
                if x > x1:
                    m -= col["P1"] * (x - x1)
                    v -= col["P1"]
                    vg -= col["P1"]
                elif x == x1:
                    v -= col["P1"]
                if x > x2:
                    m -= col["P2"] * (x - x2)
                    v -= col["P2"]
                    vg -= col["P2"]
                elif x == x2:
                    v -= col["P2"]
                ms.append(m), vs.extend((v, vg))
            env["x"].append(x)
            env["M_max"].append(max(ms)), env["M_min"].append(min(ms))
            env["V_max"].append(max(vs)), env["V_min"].append(min(vs))
        j = t["max"]["M_max"]["colonne"]
        col = t["colonnes"][j]
        gouv = S.poutre_support(Lc, a, b, e, k, col["P1"], col["P2"], pp)
        gouvernantes.append(gouv)
        d = _position(k, f"CAS {k}", TEXTES_SOUS_CAS[k], gouv,
                      [{"nom": "pied 1", "x": x1, "P": col["P1"]}, {"nom": "pied 2", "x": x2, "P": col["P2"]}])
        d.update({"enveloppe": env, "colonne": col["libelle"], "extremes_enveloppe": _extremes_enveloppe(env),
                  "max": {g: {"valeur": v["valeur"], "libelle": v["libelle"]} for g, v in t["max"].items()}})
        positions.append(d)
    # l'enveloppe générale : celle des quatre enveloppes, sur leurs abscisses réunies
    xs = sorted({x for p in positions for x in p["enveloppe"]["x"]})
    env = {"x": [], "M_max": [], "M_min": [], "V_max": [], "V_min": []}
    for x in xs:
        dedans = [p["enveloppe"] for p in positions if p["x0"] - S.EPS <= x <= p["x1"] + S.EPS]
        vals = {s: [_interp(en["x"], en[s], x) for en in dedans] for s in ("M_max", "M_min", "V_max", "V_min")}
        env["x"].append(x)
        env["M_max"].append(max(vals["M_max"])), env["M_min"].append(min(vals["M_min"]))
        env["V_max"].append(max(vals["V_max"])), env["V_min"].append(min(vals["V_min"]))
    return {"cle": "support", "nom": "Poutre support châssis", "positions": positions, "enveloppe": env,
            "repere": {"appuis_nominaux": [0.0, Lc + a + b], "pieds_nominaux": [a, a + Lc], "longueur": Lc + a + b, "e": e},
            "extremes": _extremes_enveloppe(env)}


def _interp(xs: list, ys: list, x: float) -> float:
    """Valeur d'une série (abscisses croissantes) en x, par interpolation linéaire entre ses deux voisins."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mi = (lo + hi) // 2
        if xs[mi] <= x:
            lo = mi
        else:
            hi = mi
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


def _elements(f: dict, r: K.Releve) -> list:
    """Les poutres et semelles de la fondation, avec leurs diagrammes ; ``dimensionnant`` : les valeurs que le classeur
    retient pour l'élément (les diagrammes les repèrent d'un trait fin, à côté des extremums exacts)."""
    t = f["type"]
    if t == E.CADRE4:
        els = [_elements_un_pied("poutre", "Poutre", f["chassis"] / 100, f["e"] / 100, f["charges"]["P"], r["pp.poutre"])]
    elif t == E.DIAGONALES8:
        pr = f["primaire"]
        els = [_elements_un_pied("secondaire", "Poutre secondaire", f["chassis"] / 100, f["e_secondaire"] / 100,
                                 f["charges"]["P"], r["pp.secondaire"]),
               _elements_deux_pieux("primaire", "Poutre primaire", pr["L"] / 100, f["pieux"]["entraxe"] / 100,
                                    f["e_primaire"] / 100, r["primaire.charge"], r["pp.primaire"], 0.0,
                                    "poutre secondaire")]
    else:
        sm = f["semelle"]
        els = [_element_support(f, r),
               _elements_deux_pieux("semelle", "Semelle sur pieux", sm["L"] / 100, f["pieux"]["entraxe"] / 100,
                                    f["e_semelle"] / 100, r["semelle.charge"], r["pp.semelle"], f["c_appui"] / 100,
                                    "poutre support")]
    for el in els:
        c = el["cle"]
        el["dimensionnant"] = {"M_max": r.get(f"{c}.dim.M_max"), "M_min": r.get(f"{c}.dim.M_min"), "V_max": r.get(f"{c}.dim.V_max")}
    return els


# ------------------------------------------------------------------ contrôles de géométrie (fiche BASE)
def _controles(f: dict, r: K.Releve) -> list:
    """Les « Check » de la fiche BASE (mises en forme conditionnelles : la cellule passe en rouge)."""
    if f["type"] != E.SEMELLES8:
        return []
    su, sm, pi_ = f["support"], f["semelle"], f["pieux"]

    def c(cle, cellule, texte, ok, valeur, limite, unite="cm"):
        return {"cle": cle, "cellule": cellule, "texte": texte, "ok": bool(ok), "valeur": valeur, "limite": limite,
                "unite": unite}

    return [
        c("h_semelle", "C25", "hauteur de la semelle ≥ hauteur de la poutre support + 5 cm (conflit d'armatures)",
          not sm["h"] < su["h"] + 5, sm["h"], su["h"] + 5),
        c("b_semelle", "C26", "largeur de la semelle ≥ Ø du pieu + 2 × 20 cm de dépassant",
          not sm["b"] < pi_["D"] + 2 * 20, sm["b"], pi_["D"] + 2 * 20),
        c("entraxe", "C29", "entredistance des pieux ≥ 3 × Ø du pieu", not pi_["entraxe"] < pi_["D"] * 3,
          pi_["entraxe"], pi_["D"] * 3),
        c("dist_a", "C35", "distance a ≥ 25 cm (l'appui du châssis ne doit pas sortir de la semelle)",
          not f["dist_a"] < 25, f["dist_a"], 25.0),
        c("dist_b", "C37", "distance b ≥ 25 cm", not f["dist_b"] < 25, f["dist_b"], 25.0),
        c("pas_centre", "G152", "pas des étriers centraux ≤ hauteur de l'étrier",
          not su["etriers_centre"]["pas"] > r["section.support.hauteur_etrier_centre"], su["etriers_centre"]["pas"],
          r["section.support.hauteur_etrier_centre"], "mm"),
    ]


# ------------------------------------------------------------------ armatures et pièces (ce que l'export transmet)
def _acier(element, nom_element, poste, designation, lit, requis, etat, saisie):
    lits = lit if isinstance(lit, list) else [lit]
    return {"element": element, "nom_element": nom_element, "poste": poste, "designation": designation,
            "genre": "barres", "groupes": [{"n": g["n"], "d": g["d"]} for g in lits if g["n"] > 0],
            "texte": F._barres(lit), "section": sum(D.aire(g["n"], g["d"]) for g in lits), "requis": requis,
            "etat": ETATS.get(etat, INFO), "saisie": saisie}


def _etr(element, nom_element, poste, designation, et, pas_calcule, etat, saisie, zone):
    return {"element": element, "nom_element": nom_element, "poste": poste, "designation": designation,
            "genre": "etriers", "brins": et["brins"], "d": et["d"], "pas": et["pas"], "zone": zone,
            "texte": F._cadres(et), "pas_calcule": pas_calcule, "etat": ETATS.get(etat, INFO), "saisie": saisie}


def _armatures(f: dict, r: K.Releve) -> list:
    t = f["type"]
    out = []
    if t == E.CADRE4:
        po = f["poutre"]
        out += [
            _acier("poutre", "Poutre", "inf", "Armatures inférieures", po["inf"], r["section.poutre.omega_inf"],
                   r["etat.poutre.inf"], "poutre/inf"),
            _acier("poutre", "Poutre", "sup", "Armatures supérieures", po["sup"], r["section.poutre.omega_sup"],
                   r["etat.poutre.sup"], "poutre/sup"),
            _etr("poutre", "Poutre", "etriers_appui", "Étriers à l'appui", po["etriers_appui"],
                 r["section.poutre.pas_appui"], r["etat.poutre.etriers_appui"], "poutre/etriers_appui", "appuis"),
            _etr("poutre", "Poutre", "etriers_travee", "Étriers en travée", po["etriers_travee"],
                 r["section.poutre.pas_travee"], r["etat.poutre.etriers_travee"], "poutre/etriers_travee", "travée"),
        ]
        hote = ("poutre", "Poutre")
    elif t == E.DIAGONALES8:
        for cle, nom in (("secondaire", "Poutre secondaire"), ("primaire", "Poutre primaire")):
            p = f[cle]
            out += [
                _acier(cle, nom, "inf", "Armatures inférieures", p["inf"], r[f"section.{cle}.omega_inf"],
                       r[f"etat.{cle}.inf"], f"{cle}/inf"),
                _acier(cle, nom, "sup", "Armatures supérieures", p["sup"], r[f"section.{cle}.omega_sup"],
                       r[f"etat.{cle}.sup"], f"{cle}/sup"),
                _etr(cle, nom, "etriers", "Étriers", p["etriers"], r[f"section.{cle}.pas"], r[f"etat.{cle}.etriers"],
                     f"{cle}/etriers", "toute la longueur"),
            ]
        hote = ("primaire", "Poutre primaire")
    else:
        su, sm = f["support"], f["semelle"]
        out += [
            _acier("support", "Poutre support châssis", "inf", "Armatures inférieures", su["inf"],
                   r["section.support.omega_inf"], r["etat.support.inf"], "support/inf"),
            _acier("support", "Poutre support châssis", "sup", "Armatures supérieures", su["sup"],
                   r["section.support.omega_sup"], r["etat.support.sup"], "support/sup"),
            _etr("support", "Poutre support châssis", "etriers_extremites", "Étriers aux extrémités",
                 su["etriers_extremites"], r["section.support.pas_extremites"], r["etat.support.etriers_extremites"],
                 "support/etriers_extremites", "extrémités"),
            _etr("support", "Poutre support châssis", "etriers_centre", "Étriers en partie centrale",
                 su["etriers_centre"], r["section.support.pas_centre"], r["etat.support.etriers_centre"],
                 "support/etriers_centre", "partie centrale"),
            _acier("semelle", "Semelle sur pieux", "inf", "Armatures inférieures", sm["inf"],
                   r["section.semelle.omega_inf"], r["etat.semelle.inf"], "semelle/inf"),
            _acier("semelle", "Semelle sur pieux", "sup", "Armatures supérieures", sm["sup"],
                   r["section.semelle.omega_sup"], r["etat.semelle.sup"], "semelle/sup"),
            _etr("semelle", "Semelle sur pieux", "etriers", "Étriers", sm["etriers"], r["section.semelle.pas"],
                 r["etat.semelle.etriers"], "semelle/etriers", "toute la longueur"),
        ]
        for cle, nom in (("support", "Poutre support châssis"), ("semelle", "Semelle sur pieux")):
            out.append({"element": cle, "nom_element": nom, "poste": "laterales", "designation": "Armatures latérales",
                        "genre": "prescription", "d": 12, "texte": "2Ø12 tous les 30 à 40 cm",
                        "etat": INFO, "saisie": None})
        hote = ("semelle", "Semelle sur pieux")
    re = f["renfort"]
    texte = f"{re['brins']} brins Ø{re['d']}"
    if "brins_base" in re:
        texte += f" + {re['brins_base']} brins des étriers de base"
    out.append({"element": hote[0], "nom_element": hote[1], "poste": "renfort_pieu",
                "designation": "Étriers de renfort au droit de chaque pieu", "genre": "renfort", "brins": re["brins"],
                "d": re["d"], "brins_base": re.get("brins_base"), "texte": texte,
                "section": r["renfort.omega_etriers_choisie"], "requis": r["renfort.omega_etriers"],
                "etat": ETATS.get(r["etat.renfort.etriers"], INFO), "saisie": "renfort"})
    pf = f["perfos"]
    out.append({"element": "perfos", "nom_element": "Plaques de répartition", "poste": "perfos",
                "designation": "Perfos par plaque", "genre": "perfos", "n": r["horizontal.perfos_par_plaque"], "d": pf["d"],
                "texte": f"{r['horizontal.perfos_par_plaque']}Ø{pf['d']} par plaque (calcul : {pf['n']}Ø{pf['d']} par pied)",
                "section": r["horizontal.omega_perfos_choisie"], "requis": r["horizontal.omega_perfos"],
                "etat": ETATS.get(r["etat.perfos"], INFO), "saisie": "perfos"})
    return out


def _pieces(f: dict, r: K.Releve) -> list:
    """Les pièces de béton et les pieux, avec leurs dimensions du calcul (cm) et leur nombre."""
    t = f["type"]
    if t == E.CADRE4:
        po = f["poutre"]
        return [{"cle": "poutre", "nom": "Poutre", "nombre": 4, "b": po["b"], "h": po["h"], "portee": f["chassis"]},
                {"cle": "pieux", "nom": "Pieux", "nombre": 4, "D": f["pieux"]["D"], "capacite": f["pieux"]["capacite"],
                 "unite_capacite": "kN"}]
    if t == E.DIAGONALES8:
        se, pr = f["secondaire"], f["primaire"]
        return [{"cle": "secondaire", "nom": "Poutre secondaire", "nombre": 4, "b": se["b"], "h": se["h"],
                 "portee": f["chassis"]},
                {"cle": "primaire", "nom": "Poutre primaire", "nombre": 4, "b": pr["b"], "h": pr["h"],
                 "longueur": pr["L"], "portee": f["pieux"]["entraxe"]},
                {"cle": "pieux", "nom": "Pieux", "nombre": 8, "D": f["pieux"]["D"], "entraxe": f["pieux"]["entraxe"]}]
    su, sm = f["support"], f["semelle"]
    paroi = f["cote_b"] == E.COTE_PAROI
    return [{"cle": "support", "nom": "Poutre support châssis", "nombre": 2, "b": su["b"], "h": su["h"],
             "portee": r["geometrie.longueur_support"]},
            {"cle": "semelle", "nom": "Semelle sur pieux", "nombre": 2 if paroi else 4, "b": sm["b"], "h": sm["h"],
             "longueur": sm["L"], "portee": f["pieux"]["entraxe"]},
            {"cle": "pieux", "nom": "Pieux", "nombre": 4 if paroi else 8, "D": f["pieux"]["D"],
             "entraxe": f["pieux"]["entraxe"], "capacite": f["pieux"]["capacite"], "unite_capacite": "T"}]


# ------------------------------------------------------------------ le calcul
def _tableau_cas(r: K.Releve) -> list:
    """Les colonnes des quatre grands tableaux, sans leurs 211 valeurs de M(x) (pour la fenêtre de calcul)."""
    cles = ("libelle", "cas", "orientation", "P1", "P2", "H4", "RA", "RB", "RB_resp", "symetrique", "V_max",
            "V_central", "M_hor_eq", "M_max", "M_max_eq")
    return [{"sous_cas": k, "feuille": t["feuille"], "geometrie": t["geometrie"],
             "colonnes": [{c: col[c] for c in cles} for col in t["colonnes"]],
             "max": t["max"]} for k, t in r.tableaux.items()]


def verifier(fondation: dict) -> dict:
    """Le calcul complet d'une fondation de grue (voir le format en tête de fichier). ValueError si la fondation est
    hors du domaine (``entree.normaliser``)."""
    f = E.normaliser(fondation)
    r = K.FICHES[f["type"]](f)
    m = D.materiaux(f["beton"], f["acier"])
    blocs = F.blocs(f, r, m)
    verifications = []
    for b in blocs:
        for v in b["verifications"]:
            v["bloc"] = b["cle"]
            v["etat"] = ETATS[v["etat_classeur"]] if v["valeur"] is not None else NOK
            verifications.append(v)
        for info in b["informations"]:
            info["etat"] = ETATS_ZONE.get(info.get("zone"), INFO)
        b["etat"] = fusion(*[v["etat"] for v in b["verifications"]],
                           *[i["etat"] for i in b["informations"] if i["etat"] != INFO])
    ecarts = X.signaler(f, r)
    for b in blocs:
        b["ecarts"] = [e for e in ecarts if e["bloc"] == b["cle"]]
    avec_taux = [v for v in verifications if v["taux"] is not None]
    gouv = max(avec_taux, key=lambda v: v["taux"]) if avec_taux else None
    etats = [v["etat"] for v in verifications] + [b["etat"] for b in blocs]
    out = {"reference": REFERENCE, "classeur": E.CLASSEUR_TYPE[f["type"]], "fondation": f, "materiaux": m,
           "valeurs": dict(r), "cellules": dict(r.cellules), "blocs": blocs, "verifications": verifications,
           "elements": _elements(f, r), "controles": _controles(f, r), "ecarts": ecarts,
           "armatures": _armatures(f, r), "pieces": _pieces(f, r),
           "statut": fusion(*etats), "taux_max": gouv["taux"] if gouv else None,
           "gouvernant": None if gouv is None else {"cle": gouv["cle"], "bloc": gouv["bloc"], "nom": gouv["nom"],
                                                    "taux": gouv["taux"]}}
    if f["type"] == E.SEMELLES8:
        out["tableau_cas"] = _tableau_cas(r)
    return out

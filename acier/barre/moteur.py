# -*- coding: utf-8 -*-
"""LA BARRE VÉRIFIÉE par ses deux méthodes — l'historique du bureau (NBN B 51-001, charges de service) et
l'Eurocode 3 (ELU) —, depuis ses données d'écran (acier/barre/entree.py). docs/acier/BARRES_ACIER.md § 9.

DEUX MODES :

« efforts » (le mode des classeurs) — les efforts sont donnés : ceux de service vont à la méthode historique, ceux de
calcul à l'Eurocode. Une vérification par méthode.

« chargee » — la barre a ses appuis, ses cas de charge, ses maintiens ; le module fait la statique (statique.py) et les
combinaisons (combinaisons.py), puis cherche, vérification par vérification, l'occurrence la plus DÉFAVORABLE :

    Eurocode 3 : pour chaque combinaison (6.10) —
        la SECTION dans chaque section à vérifier (appuis, ruptures de chargement, extremum du moment de chaque
        travée), avec les efforts qui s'y trouvent ENSEMBLE, des deux côtés d'un saut ;
        la BARRE par tronçon de déversement : flambement sous N, déversement sous le diagramme réel du tronçon
        (M_cr numérique, charges à leur niveau), flexion composée avec le plus grand moment du tronçon.
    Méthode historique : pour chaque combinaison de service (G + ΣQ) — dans chaque section à vérifier, avec la
        longueur de déversement L_c de son tronçon.

L'enveloppe garde, pour chaque vérification, où elle a été trouvée (la combinaison, l'abscisse ou le tronçon) et les
formules de CETTE occurrence : la note imprime ce qui a décidé.

Ce que la barre chargée ne fait pas (v1) : la flexion d'axe faible et les charges horizontales (le mode « efforts »
les prend) ; la flèche n'est donnée qu'à titre d'information.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple


from . import ame as AME
from . import combinaisons as CO
from . import ec3 as EC
from . import ec3_deversement as DV
from . import ec3_section as ES
from . import entree as EN
from . import historique as HI
from . import mcr as MC
from . import parametres as PA
from . import resultat as RE
from . import section as SE
from . import statique as ST
from .enveloppe import IDS_BARRE, ORDRE, Enveloppe          # noqa: F401 — ORDRE, relu par les tests

# la comparaison des deux méthodes : (sujet, vérifications historiques, vérifications de l'Eurocode)
# (la méthode historique n'a pas de vérification de l'effort tranchant seul : il entre dans la contrainte équivalente)
SUJETS = (("Flambement, axe faible", ("H-N-z",), ("E-Nb-z", "E-Nb-T")), ("Flambement, axe fort", ("H-N-y",), ("E-Nb-y",)),
          ("Flexion et contraintes, déversement compris", ("H-SIGMA", "H-EQ"), ("E-My", "E-Mz", "E-MV", "E-NM", "E-Mb", "E-661", "E-662")),
          ("Effort tranchant seul", (), ("E-Vz", "E-Vy")))
SUJET_AME = "Âme et raidisseurs (EN 1993-1-5)"
POINTS_DIAGRAMME = 40            # points par travée des enveloppes rendues pour le dessin
BALAYAGE_TRONCON = 24            # points par tronçon pour le plus grand moment
EPS = 1e-9
SAUT = 1e-9                      # l'écart relatif entre les lectures de gauche et de droite qui fait un saut de diagramme


def _refusee(methode: str, norme: str, message: str) -> Dict:
    """Le résultat d'une méthode qui REFUSE la barre (une épaisseur au-delà de 80 mm (domaine du module), une section sans courbe…) :
    « non couvert », avec le motif — l'autre méthode reste rendue."""
    J = RE.Journal(methode, norme)
    J.verif("E-SECTION" if methode == "ec3" else "H-SIGMA", "Vérification impossible", norme, "", None, None, "", [], statut=RE.NON_COUVERT,
            messages=[J.alerte("refus", "attention", message)])
    r = J.rendre()
    r.update({"combinaison": None, "combinaisons": [], "a_confirmer": []})
    return r


def _fy_historique(nuance: str, s: Dict) -> Dict:
    """La limite élastique de la méthode historique : celle de l'Eurocode 3, selon l'épaisseur (brief § 13.2 point 5,
    DÉCISION BUREAU acceptée : « f_y des plats de plus de 16 mm : la valeur EC3 selon l'épaisseur ») — la même lecture
    de la table (ec3_section.limite_elastique) ; Refus au-delà de son domaine, comme l'Eurocode 3. Le DOMAINE de la doc
    (AE235 à AE355) se lit sur la nuance : elle est passée avec f_y. {"fy", "nuance", "fy_source", "fy_ref"}."""
    return {"fy": ES.limite_elastique(nuance, s), "nuance": nuance, "fy_source": ES.source_fy(nuance, s), "fy_ref": ES.tableau_fy(nuance, s)}


def _C_historique(mode: str, forme: str, psi: float) -> Dict:
    """Le coefficient C d'un axe : 1 (« unite », les classeurs) ou, selon la doc p. 20 et 21 (« norme ») : 0,85 sous des
    charges appliquées sur la barre, √(0,3·(1 + ψ²) + 0,4·ψ) sous des moments d'extrémité seuls."""
    if mode == "unite":
        return {"mode": "unite"}
    if forme in (DV.REPARTI, DV.PONCTUEL, DV.GENERAL):
        return {"mode": "charges"}
    return {"mode": "moments", "psi": 1.0 if forme == DV.CONSTANT else psi}


CONSOLE_LZ = "Console à bout libre : L_cr,z = 2·L, maintiens intermédiaires non comptés — À VÉRIFIER."


def _bout_libre(b: Dict, trs: Optional[List[Dict]]) -> bool:
    """Une console dont le bout n'est tenu par aucun maintien complet : son dernier tronçon finit LIBRE (statique.troncons)."""
    return b["modele"] == "console" and bool(trs) and trs[-1]["appuis"][1] == MC.LIBRE


def _longueurs(b: Dict, trs: Optional[List[Dict]]) -> Dict:
    """Les longueurs de flambement (mm) : celles données, sinon la plus grande travée (axe fort) et le plus long tronçon
    entre sections tenues (axe faible). Une CONSOLE flambe sur deux fois sa longueur autour de l'axe fort, et autour de
    l'axe faible tant que son bout est libre : il se déplace latéralement comme dans le plan (brief § 13.3 : u = 2).
    Un maintien complet au bout en fait une section tenue : la plus grande distance entre sections tenues (côté sûr).
    Des maintiens intermédiaires seuls ne raccourcissent rien : la partie libre ne se prend pas isolément sans
    justification — 2·L, hypothèse prudente, À VÉRIFIER (CONSOLE_LZ, dite par les deux méthodes)."""
    out = {}
    for a in ("y", "z"):
        L = b["flambement"][a]["L"]
        if L is None:
            L = max(b["travees"]) if a == "y" else max(t["x2"] - t["x1"] for t in trs)
            if a == "y" and b["modele"] == "console":
                L = 2.0 * L                                  # une console flambe sur deux fois sa longueur
            if a == "z" and _bout_libre(b, trs):
                L = 2.0 * sum(b["travees"])                  # bout libre : 2·L, même avec des maintiens intermédiaires
        out[a] = {"L": L * 1e3, "u": b["flambement"][a]["u"]}
    return out


def _avis_longueurs(b: Dict, trs: Optional[List[Dict]]) -> List[Dict]:
    """L'alerte À VÉRIFIER d'une console à bout libre dont L_cr,z automatique ignore des maintiens complets intermédiaires."""
    if b["flambement"]["z"]["L"] is None and _bout_libre(b, trs) and len(trs) > 1:
        return [{"code": "console_Lz", "niveau": "attention", "message": CONSOLE_LZ}]
    return []


def _longueurs_rendues(b: Dict, axes: Dict) -> Dict:
    """Les longueurs de flambement PRISES, en mètres : {"y": {"L", "u", "auto"}, "z": …} ; « auto » quand la barre ne
    la donne pas (l'écran l'écrit dans le champ vide)."""
    return {a: {"L": axes[a]["L"] / 1e3, "u": axes[a]["u"], "auto": b["flambement"][a]["L"] is None} for a in ("y", "z")}


# ------------------------------------------------------------------ efforts imposés
def _efforts_imposes(b: Dict, s: Dict) -> Dict:
    axes = _longueurs(b, None)
    Lc = b["Lc"]
    methodes = {}

    def historique() -> Dict:
        e, h = b["efforts"]["service"], b["historique"]
        return HI.verifier({"section": s, **_fy_historique(b["nuance"], s), "s": h["s"], "P": e["N"] * 1e3, "My": e["My"] * 1e6, "Mz": e["Mz"] * 1e6,
                         "Vz": e["Vz"] * 1e3, "Vy": e["Vy"] * 1e3, "axes": axes, "tolerance": h["tolerance"],
                         "C": {a: _C_historique(h["C"], b["diagramme"][a]["forme"], b["diagramme"][a]["psi"]) for a in ("y", "z")},
                         "deversement": {"actif": h["deversement"], "Lc": None if Lc is None else Lc * 1e3}})

    def ec3() -> Dict:
        e = b["efforts"]["elu"]
        r = EC.verifier({"section": s, "nuance": b["nuance"], "NEd": e["N"] * 1e3, "MyEd": e["My"] * 1e6, "MzEd": e["Mz"] * 1e6,
                         "VzEd": e["Vz"] * 1e3, "VyEd": e["Vy"] * 1e3, "axes": axes, "parametres": b["parametres"], "voilement": False,
                         "flexion": {a: {"diagramme": b["diagramme"][a]["forme"], "psi": b["diagramme"][a]["psi"]} for a in ("y", "z")},
                         "deversement": {"Lc": None if Lc is None else Lc * 1e3, "niveau": b["niveau"]}})
        r["a_confirmer"] = r["grandeurs"].get("a_confirmer", [])
        return r

    for nom, calcul, norme in (("historique", historique, HI.NORME), ("ec3", ec3, EC.NORME)):
        if nom in b["methodes"]:
            try:
                methodes[nom] = calcul()
            except RE.Refus as e:
                methodes[nom] = _refusee(nom, norme, str(e))
    return {"statique": None, "methodes": methodes, "ame": AME.efforts(b, s), "longueurs": _longueurs_rendues(b, axes)}


# ------------------------------------------------------------------ barre chargée
def _moment_max(comb: ST.Combinaison, tr: Dict, xs: Sequence[float]) -> float:
    """Le plus grand |M| de la combinaison sur le tronçon (kN·m) : balayage et sections à vérifier."""
    a, b = tr["x1"], tr["x2"]
    pts = [a + (b - a) * i / BALAYAGE_TRONCON for i in range(BALAYAGE_TRONCON + 1)] + [x for x in xs if a < x < b]
    m = 0.0
    for x in pts:
        e = comb.lire(x)
        m = max(m, abs(e["M_g"]) if x > a + EPS else 0.0, abs(e["M_d"]) if x < b - EPS else 0.0)
    return m


def _kc_tableau(b: Dict, cas: Sequence[Dict], facteurs: Sequence[float], tr: Dict, C: Optional[ST.Combinaison] = None) -> Optional[float]:
    """k_c du tableau 6.6 quand le tronçon EST un de ses schémas : un tronçon entre deux fourches, sans maintien intérieur
    ni charge dedans — moment linéaire (ec3_deversement.kc_lineaire, ψ lu sur la combinaison ``C``) ; ou une portée seule
    (deux appuis simples, deux encastrements, un encastrement et un appui), tronçon = la portée, sous des charges uniformes
    sur toute la portée ou des charges à mi-portée seulement (ec3_deversement.kc_portee). Sinon None (k_c = 1)."""
    L = sum(b["travees"])
    charges = [ch for f, c in zip(facteurs, cas) if f for ch in c.get("charges") or []]
    if C is not None and not tr["maintiens"] and tuple(tr["appuis"]) == (MC.FOURCHE, MC.FOURCHE):
        k = DV.kc_lineaire(charges, tr["x1"], tr["x2"], C.lire(tr["x1"])["M_d"], C.lire(tr["x2"])["M_g"])
        if k is not None:
            return k
    if b["modele"] not in DV.MODELES_KC or tr["maintiens"] or tr["x2"] - tr["x1"] < L - EPS:
        return None
    return DV.kc_portee(b["modele"], charges, L)


def _Lc_historique(comb: ST.Combinaison, tr: Dict, xs: Sequence[float]) -> float:
    """La longueur de déversement de la méthode historique sur le tronçon (m) : la plus grande distance entre deux
    points où la semelle COMPRIMÉE est tenue. Un maintien d'une seule semelle ne compte que si elle est comprimée sur
    tout le tronçon (moment d'un seul signe)."""
    a, b = tr["x1"], tr["x2"]
    pts = [a + (b - a) * i / BALAYAGE_TRONCON for i in range(1, BALAYAGE_TRONCON)]
    Ms = [comb.M(x) for x in pts]
    haut, bas = all(m >= -EPS for m in Ms), all(m <= EPS for m in Ms)
    tenus = [a, b]
    for x_mm, z in tr["maintiens"]:
        if (z > 0 and haut) or (z < 0 and bas):
            tenus.append(a + x_mm / 1e3)
    tenus.sort()
    return max(v - u for u, v in zip(tenus, tenus[1:]))


def _chargee(b: Dict, s: Dict) -> Dict:
    P = PA.jeu(b["parametres"])
    cas = ST.cas_complets(b, s)
    EI = P["E"] * s["Iy"] / 1e9                                # kN·m²
    sols = ST.resoudre(b, cas, EI)
    N_cas = [c.get("N", 0.0) for c in cas]
    pts = ST.ruptures(sols)
    trs = ST.troncons(b, s)
    axes = _longueurs(b, trs)
    zg = {"haut": 1.0, "centre": 0.0, "bas": -1.0}[b["niveau"]] * (s.get("h") or 0.0) / 2.0
    Lt = sum(b["travees"])
    bouts = {round(sum(b["travees"][:k]), 9) for k in range(len(b["travees"]) + 1)}     # les appuis et les bouts de la barre
    grille = sorted({round(x, 9) for x in pts} | {round(x0 + L * i / POINTS_DIAGRAMME, 9) for x0, L in
                                                  zip([sum(b["travees"][:k]) for k in range(len(b["travees"]))], b["travees"])
                                                  for i in range(POINTS_DIAGRAMME + 1)})
    statique: Dict = {"L": Lt, "x": grille, "troncons": [{"x1": t["x1"], "x2": t["x2"], "appuis": list(t["appuis"]),
                                                           "maintiens": [[t["x1"] + x / 1e3, z] for x, z in t["maintiens"]]} for t in trs],
                      "cas": [{"nom": c["nom"], "nature": c["nature"], "N": c.get("N", 0.0),
                               "reactions": [{"nom": a["nom"], "x": a["x"], "R": a["R"], "M": a.get("M", 0.0)} for a in sol.appuis]}
                              for c, sol in zip(cas, sols)], "enveloppes": {}}
    methodes = {}

    def enveloppe_diagrammes(nom: str, combs: List[ST.Combinaison]) -> None:
        """Les enveloppes de M et de V, avec LEURS abscisses : là où un diagramme saute (un appui, une charge ponctuelle,
        un couple), l'abscisse y est deux fois — la lecture de gauche, puis celle de droite — pour que le saut se trace
        vertical ; aux deux bouts de la barre, seule la lecture du côté de la barre compte (dehors, il n'y a rien)."""
        xs, M_max, M_min, V_max, V_min = [], [], [], [], []
        for x in grille:
            es = [c.lire(x) for c in combs]
            cotes = ("d",) if x <= EPS else (("g",) if x >= Lt - EPS else ("g", "d"))
            if len(cotes) == 2 and all(abs(e[k + "_g"] - e[k + "_d"]) <= SAUT * max(1.0, abs(e[k + "_g"]), abs(e[k + "_d"]))
                                       for e in es for k in ("M", "V")):
                cotes = ("d",)                                       # diagrammes continus ici : un seul point
            for cote in cotes:
                xs.append(x)
                M_max.append(max(e["M_" + cote] for e in es))
                M_min.append(min(e["M_" + cote] for e in es))
                V_max.append(max(e["V_" + cote] for e in es))
                V_min.append(min(e["V_" + cote] for e in es))
        statique["enveloppes"][nom] = {"x": xs, "M_max": M_max, "M_min": M_min, "V_max": V_max, "V_min": V_min}

    def ec3() -> Dict:
        env = Enveloppe(EC.METHODE, EC.NORME)
        combs = []
        for co in CO.elu(cas, P):
            C = ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas)
            combs.append(C)
            xs = ST.sections(b, C, pts)
            for x in xs:                                         # la SECTION, là où les efforts sont ensemble
                for M, V in ST.cotes(C.lire(x), EPS):
                    if abs(M) < EPS and abs(V) < EPS and abs(C.N) < EPS:
                        continue
                    r = ES.verifier({"section": s, "nuance": b["nuance"], "NEd": C.N * 1e3, "MyEd": M * 1e6, "VzEd": V * 1e3,
                                     "parametres": b["parametres"], "voilement": False})
                    env.verser(r, {"combinaison": co["nom"], "x": x})
            for i, tr in enumerate(trs):                         # la BARRE, tronçon par tronçon
                Mmax = _moment_max(C, tr, xs)
                if Mmax < EPS and C.N <= EPS:
                    continue
                rep, pon = ST.charges_du_troncon(cas, co["facteurs"], tr, zg)
                a = tr["x1"]
                troncon = {"L": (tr["x2"] - a) * 1e3, "moment": (lambda x, C=C, a=a: C.M(a + x / 1e3) * 1e6), "reparties": rep,
                           "ponctuelles": pon, "appuis": tr["appuis"], "maintiens": tr["maintiens"], "M_max": Mmax * 1e6,
                           "entre_appuis": round(tr["x1"], 9) in bouts and round(tr["x2"], 9) in bouts}   # (la variante du couple le lit)
                flexion = {"diagramme": DV.CONSTANT}
                if b["modele"] == "appuis" and Mmax > EPS:      # tableau A.2, formule générale : la flèche et le moment de la portée
                    fl = max(abs(C.lire(x)["f"] or 0.0) for x in grille)
                    flexion = {"diagramme": DV.GENERAL, "delta": fl * 1e3, "L": Lt * 1e3}
                r = EC.verifier({"section": s, "nuance": b["nuance"], "NEd": C.N * 1e3, "MyEd": Mmax * 1e6, "axes": axes,
                                 "flexion": {"y": flexion}, "parametres": b["parametres"], "voilement": False, "console": b["modele"] == "console",
                                 "deversement": {"troncon": troncon, "kc": _kc_tableau(b, cas, co["facteurs"], tr, C), "zg": zg}})
                env.verser(r, {"combinaison": co["nom"], "troncon": [tr["x1"], tr["x2"]]}, prefixes=IDS_BARRE)
        enveloppe_diagrammes("ec3", combs)
        r = env.rendre()
        ou = {"haut": "sur la semelle haute", "centre": "au centre de cisaillement", "bas": "sous la semelle basse"}[b["niveau"]]
        r["hypotheses"].append(f"Charges appliquées {ou} ; appuis en fourche (déplacement latéral et torsion empêchés).")
        return r

    def historique() -> Dict:
        env = Enveloppe(HI.METHODE, HI.NORME)
        h = b["historique"]
        fy = _fy_historique(b["nuance"], s)
        combs = []
        fleche = {"valeur": 0.0, "x": None, "combinaison": None}
        for co in CO.service(cas):
            C = ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas)
            combs.append(C)
            xs = ST.sections(b, C, pts)
            Lcs = [(tr, _Lc_historique(C, tr, xs)) for tr in trs]
            for x in xs:
                tr_x = next(((tr, Lc) for tr, Lc in Lcs if tr["x1"] - EPS <= x <= tr["x2"] + EPS), Lcs[-1])
                for M, V in ST.cotes(C.lire(x), EPS):
                    if abs(M) < EPS and abs(V) < EPS and abs(C.N) < EPS:
                        continue
                    r = HI.verifier({"section": s, **fy, "s": h["s"], "P": C.N * 1e3, "My": M * 1e6, "Vz": V * 1e3, "axes": axes,
                                     "C": {"y": _C_historique(h["C"], DV.GENERAL, 1.0)}, "tolerance": h["tolerance"],
                                     "deversement": {"actif": h["deversement"], "Lc": tr_x[1] * 1e3}})
                    env.verser(r, {"combinaison": co["nom"], "x": x, "troncon": [tr_x[0]["x1"], tr_x[0]["x2"]]})
            for x in grille:                                     # la flèche de service, à titre d'information
                f = C.lire(x)["f"]
                if f is not None and abs(f) > abs(fleche["valeur"]):
                    fleche = {"valeur": f, "x": x, "combinaison": co["nom"]}
        enveloppe_diagrammes("historique", combs)
        portee = max(b["travees"]) * (2.0 if b["modele"] == "console" else 1.0)
        fleche["valeur"] *= 1e3                                  # mm
        fleche["rapport"] = portee * 1e3 / abs(fleche["valeur"]) if fleche["valeur"] else None
        statique["fleche"] = fleche
        return env.rendre()

    for nom, calcul, norme in (("historique", historique, HI.NORME), ("ec3", ec3, EC.NORME)):
        if nom in b["methodes"]:
            try:
                methodes[nom] = calcul()
                methodes[nom]["alertes"] += _avis_longueurs(b, trs)       # (les deux méthodes lisent L_cr,z)
            except RE.Refus as e:
                methodes[nom] = _refusee(nom, norme, str(e))
    return {"statique": statique, "methodes": methodes, "ame": AME.chargee(b, s, cas, sols, N_cas, pts),
            "longueurs": _longueurs_rendues(b, axes)}


def _comparaison(methodes: Dict, ame: Dict) -> List[Dict]:
    """Les deux méthodes côte à côte, sujet par sujet : le plus fort taux de chacune ; puis, sous « Le plus
    défavorable », le taux de l'âme et de ses raidisseurs (« commun » : il ne se compare pas, il vaut pour les deux)."""
    if len(methodes) < 2:
        return []

    def pire(r: Dict, ids: Sequence[str]) -> Optional[float]:
        t = [v["taux"] for v in r["verifications"] if v["id"] in ids and v["taux"] is not None]
        return max(t) if t else None

    lignes = [{"sujet": sujet, "historique": pire(methodes["historique"], h), "ec3": pire(methodes["ec3"], e)} for sujet, h, e in SUJETS]
    lignes = [li for li in lignes if li["historique"] is not None or li["ec3"] is not None]
    lignes.append({"sujet": "Le plus défavorable", "historique": methodes["historique"]["taux_max"], "ec3": methodes["ec3"]["taux_max"]})
    if ame.get("taux_max") is not None:              # l'âme (EN 1993-1-5) vaut pour les deux méthodes : une ligne à part, un seul taux
        lignes.append({"sujet": SUJET_AME, "historique": None, "ec3": None, "commun": ame["taux_max"]})
    return lignes


def verifier(barre: Dict) -> Dict:
    """La barre vérifiée : {"barre" (relue), "profil", "masse" (kg/m), "poids_propre" (kN/m : celui que la barre chargée
    compte — acier/barre/statique.poids_propre ; l'écran le lit, il ne le recalcule pas), "statique" (None en efforts imposés),
    "longueurs" (de flambement, celles qui sont prises), "methodes": {"historique": …, "ec3": …} (la forme de
    acier/barre/resultat.py ; en barre chargée, chaque vérification dit sa combinaison et son lieu), "ame" (la même
    forme : l'âme selon l'EN 1993-1-5, toujours calculée, avec la décision de chaque raidisseur — acier/barre/ame.py),
    "statut" (le plus défavorable des méthodes cochées et de l'âme : nok, non couvert, hors domaine, ok), "comparaison"}."""
    b = EN.normaliser(barre)
    s = SE.du_catalogue(b["profil"])
    out = _efforts_imposes(b, s) if b["mode"] == "efforts" else _chargee(b, s)
    statuts = [r["statut"] for r in out["methodes"].values()] + [out["ame"]["statut"]]
    statut = RE.pire_statut(statuts)
    out.update({"barre": b, "profil": s["nom"], "masse": s.get("masse"), "poids_propre": ST.poids_propre(s), "statut": statut,
                "comparaison": _comparaison(out["methodes"], out["ame"])})
    return out

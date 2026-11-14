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

from acier.flambement import normes as NO

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

ORDRE = {"ec3": ("E-SECTION", "E-N", "E-My", "E-Mz", "E-Vz", "E-Vy", "E-Vb", "E-Vb-y", "E-MV", "E-NM", "E-Nb-y", "E-Nb-z", "E-Nb-T", "E-Mb",
                 "E-661", "E-662"),
         "historique": ("H-N-z", "H-N-y", "H-SIGMA", "H-EQ")}
IDS_BARRE = ("E-Nb", "E-Mb", "E-66")             # les vérifications de la barre (les autres sont celles de la section)
# la comparaison des deux méthodes : (sujet, vérifications historiques, vérifications de l'Eurocode)
# (la méthode historique n'a pas de vérification de l'effort tranchant seul : il entre dans la contrainte équivalente)
SUJETS = (("Flambement, axe faible", ("H-N-z",), ("E-Nb-z", "E-Nb-T")), ("Flambement, axe fort", ("H-N-y",), ("E-Nb-y",)),
          ("Flexion et contraintes, déversement compris", ("H-SIGMA", "H-EQ"), ("E-My", "E-Mz", "E-MV", "E-NM", "E-Mb", "E-661", "E-662")),
          ("Effort tranchant seul", (), ("E-Vz", "E-Vy")))
POINTS_DIAGRAMME = 40            # points par travée des enveloppes rendues pour le dessin
BALAYAGE_TRONCON = 24            # points par tronçon pour le plus grand moment
EPS = 1e-9


def _rang(v: Dict) -> Tuple[int, float]:
    """L'ordre de gravité d'une occurrence : un refus sans taux passe devant tout taux ; un taux devant le reste."""
    if v["statut"] == RE.NOK and v["taux"] is None:
        return (2, 0.0)
    if v["taux"] is not None:
        return (1, v["taux"])
    return (0, 0.0)


class Enveloppe:
    """Par vérification, l'occurrence la plus défavorable rencontrée, et le bilan de chaque combinaison."""

    def __init__(self, methode: str, norme: str):
        self.methode, self.norme = methode, norme
        self.pire: Dict[str, Dict] = {}
        self.bilan: Dict[str, Dict] = {}
        self.alertes: Dict[str, Dict] = {}
        self.a_confirmer: Dict[str, Dict] = {}

    def verser(self, r: Dict, ou: Dict, prefixes: Optional[Sequence[str]] = None, sauf: Optional[Sequence[str]] = None) -> None:
        """Verse le résultat ``r`` d'une méthode, trouvé ``ou`` ({"combinaison", "x" | "troncon"}) ; ``prefixes`` : ne
        garder que ces vérifications ; ``sauf`` : les écarter."""
        blocs = {f["cle"]: f for f in r["calcul"]}
        gardees = []
        for v in r["verifications"]:
            if prefixes and not v["id"].startswith(tuple(prefixes)):
                continue
            if sauf and v["id"].startswith(tuple(sauf)):
                continue
            gardees.append(v)
            rang = _rang(v)
            courant = self.pire.get(v["id"])
            if courant is None or rang > courant["rang"]:
                self.pire[v["id"]] = {"rang": rang, "v": v, "blocs": [blocs[k] for k in v["formules"]], "ou": ou, "r": r}
        if gardees:
            b = self.bilan.setdefault(ou["combinaison"], {"nom": ou["combinaison"], "rang": (0, 0.0), "gouvernant": None, "non_couvert": False})
            for v in gardees:
                if _rang(v) > b["rang"]:
                    b["rang"], b["gouvernant"] = _rang(v), v["id"]
                b["non_couvert"] = b["non_couvert"] or v["statut"] == RE.NON_COUVERT
        for a in r["alertes"]:
            self.alertes.setdefault(a["code"], a)
        for p in r["grandeurs"].get("a_confirmer") or []:
            self.a_confirmer.setdefault(p["cle"], p)

    def _grandeurs(self, gouv: Optional[Dict]) -> Dict:
        """Les grandeurs rendues avec la méthode. Historique : celles de l'occurrence gouvernante. Eurocode 3 : celles de
        la SECTION la plus sollicitée (classe, résistances…), puis — chacune prise là où SA vérification est la plus
        défavorable — le flambement, le déversement et l'annexe A."""
        if gouv is None:
            return {}
        if self.methode != "ec3":
            return gouv["r"]["grandeurs"]

        def de(ids: Sequence[str]) -> Optional[Dict]:
            trouves = [self.pire[i] for i in ids if i in self.pire]
            return max(trouves, key=lambda o: o["rang"])["r"]["grandeurs"] if trouves else None

        barre = de([i for i in self.pire if i.startswith(IDS_BARRE)])
        section = de([i for i in self.pire if not i.startswith(IDS_BARRE)])
        G = dict(section or barre or {})
        for cle, ids in (("flambement", ("E-Nb-z", "E-Nb-y", "E-Nb-T")), ("deversement", ("E-Mb",)), ("annexe_A", ("E-661", "E-662"))):
            g = de(ids) or barre
            if g is not None and cle in g:
                G[cle] = g[cle]
        return G

    def rendre(self) -> Dict:
        """Le résultat de la méthode sur toute la barre : la forme de acier/barre/resultat.py, où chaque vérification
        dit sa combinaison et son lieu ; ``calcul`` porte les formules de chaque occurrence retenue (clés « id/clé »)."""
        verifs, calcul = [], []
        for id_ in list(ORDRE[self.methode]) + sorted(set(self.pire) - set(ORDRE[self.methode])):
            if id_ not in self.pire:
                continue
            o = self.pire[id_]
            v = dict(o["v"])
            v["formules"] = [f"{id_}/{b['cle']}" for b in o["blocs"]]
            v["messages"] = list(v["messages"])
            v.update({"combinaison": o["ou"]["combinaison"], "x": o["ou"].get("x"), "troncon": o["ou"].get("troncon")})
            calcul += [dict(b, cle=f"{id_}/{b['cle']}") for b in o["blocs"]]
            verifs.append(v)
        s = RE.synthese(verifs)
        gouv = self.pire[s["gouvernant"]] if s["gouvernant"] else None
        combis = [{"nom": b["nom"], "gouvernant": b["gouvernant"], "taux": b["rang"][1] if b["rang"][0] == 1 else None,
                   "statut": RE.NOK if (b["rang"][0] == 2 or b["rang"][1] > 1.0) else RE.NON_COUVERT if b["non_couvert"] else RE.OK}
                  for b in self.bilan.values()]
        return {"methode": self.methode, "norme": self.norme, "statut": s["statut"], "taux_max": s["taux_max"], "gouvernant": s["gouvernant"],
                "combinaison": gouv["ou"]["combinaison"] if gouv else None, "verifications": verifs, "calcul": calcul,
                "grandeurs": self._grandeurs(gouv), "alertes": list(self.alertes.values()),
                "hypotheses": list(gouv["r"]["hypotheses"]) if gouv else [], "a_confirmer": list(self.a_confirmer.values()),
                "combinaisons": combis}


def _refusee(methode: str, norme: str, message: str) -> Dict:
    """Le résultat d'une méthode qui REFUSE la barre (une épaisseur hors du tableau 3.1, une section sans courbe…) :
    « non couvert », avec le motif — l'autre méthode reste rendue."""
    J = RE.Journal(methode, norme)
    J.verif("E-SECTION" if methode == "ec3" else "H-SIGMA", "Vérification impossible", norme, "", None, None, "", [], statut=RE.NON_COUVERT,
            messages=[J.alerte("refus", "attention", message)])
    r = J.rendre()
    r.update({"combinaison": None, "combinaisons": [], "a_confirmer": []})
    return r


def _fy_historique(nuance: str) -> float:
    """La limite élastique de la méthode historique : la valeur nominale de la nuance (les classeurs : 235, 355)."""
    return NO.NUANCES[nuance][0]


def _C_historique(mode: str, forme: str, psi: float) -> Dict:
    """Le coefficient C d'un axe : 1 (« unite », les classeurs) ou, selon la doc p. 20 et 21 (« norme ») : 0,85 sous des
    charges appliquées sur la barre, √(0,3·(1 + ψ²) + 0,4·ψ) sous des moments d'extrémité seuls."""
    if mode == "unite":
        return {"mode": "unite"}
    if forme in (DV.REPARTI, DV.PONCTUEL, DV.GENERAL):
        return {"mode": "charges"}
    return {"mode": "moments", "psi": 1.0 if forme == DV.CONSTANT else psi}


def _longueurs(b: Dict, trs: Optional[List[Dict]]) -> Dict:
    """Les longueurs de flambement (mm) : celles données, sinon la plus grande travée (axe fort) et le plus long tronçon
    entre sections tenues (axe faible)."""
    out = {}
    for a in ("y", "z"):
        L = b["flambement"][a]["L"]
        if L is None:
            L = max(b["travees"]) if a == "y" else max(t["x2"] - t["x1"] for t in trs)
            if a == "y" and b["modele"] == "console":
                L = 2.0 * L                                  # une console flambe sur deux fois sa longueur
        out[a] = {"L": L * 1e3, "u": b["flambement"][a]["u"]}
    return out


# ------------------------------------------------------------------ efforts imposés
def _efforts_imposes(b: Dict, s: Dict) -> Dict:
    axes = _longueurs(b, None)
    Lc = b["Lc"]
    methodes = {}

    def historique() -> Dict:
        e, h = b["efforts"]["service"], b["historique"]
        return HI.verifier({"section": s, "fy": _fy_historique(b["nuance"]), "s": h["s"], "P": e["N"] * 1e3, "My": e["My"] * 1e6, "Mz": e["Mz"] * 1e6,
                         "Vz": e["Vz"] * 1e3, "Vy": e["Vy"] * 1e3, "axes": axes, "tolerance": h["tolerance"],
                         "C": {a: _C_historique(h["C"], b["diagramme"][a]["forme"], b["diagramme"][a]["psi"]) for a in ("y", "z")},
                         "deversement": {"actif": h["deversement"], "Lc": None if Lc is None else Lc * 1e3}})

    def ec3() -> Dict:
        e = b["efforts"]["elu"]
        r = EC.verifier({"section": s, "nuance": b["nuance"], "NEd": e["N"] * 1e3, "MyEd": e["My"] * 1e6, "MzEd": e["Mz"] * 1e6,
                         "VzEd": e["Vz"] * 1e3, "VyEd": e["Vy"] * 1e3, "axes": axes, "parametres": b["parametres"],
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
    return {"statique": None, "methodes": methodes}


# ------------------------------------------------------------------ barre chargée
def _cotes(e: Dict) -> List[Tuple[float, float]]:
    """Les couples (M, V) de part et d'autre d'une section — un seul quand rien ne saute."""
    g, d = (e["M_g"], e["V_g"]), (e["M_d"], e["V_d"])
    return [d] if (abs(g[0] - d[0]) < EPS and abs(g[1] - d[1]) < EPS) else [g, d]


def _moment_max(comb: ST.Combinaison, tr: Dict, xs: Sequence[float]) -> float:
    """Le plus grand |M| de la combinaison sur le tronçon (kN·m) : balayage et sections à vérifier."""
    a, b = tr["x1"], tr["x2"]
    pts = [a + (b - a) * i / BALAYAGE_TRONCON for i in range(BALAYAGE_TRONCON + 1)] + [x for x in xs if a < x < b]
    m = 0.0
    for x in pts:
        e = comb.lire(x)
        m = max(m, abs(e["M_g"]) if x > a + EPS else 0.0, abs(e["M_d"]) if x < b - EPS else 0.0)
    return m


def _kc_tableau(b: Dict, cas: Sequence[Dict], facteurs: Sequence[float], tr: Dict) -> Optional[float]:
    """k_c du tableau 6.6 quand le tronçon EST un de ses cas : poutre sur deux appuis, tronçon = la portée, sans maintien
    intérieur, et soit des charges uniformes sur toute la portée (0,94), soit une seule charge à mi-portée (0,86)."""
    L = sum(b["travees"])
    if b["modele"] != "appuis" or tr["maintiens"] or tr["x2"] - tr["x1"] < L - EPS:
        return None
    charges = [ch for f, c in zip(facteurs, cas) if f for ch in c.get("charges") or []]
    if charges and all(ch["type"] == "q" and ch["a"] <= EPS and ch["b"] >= L - EPS for ch in charges):
        return DV.KC_REPARTI
    if len(charges) == 1 and charges[0]["type"] == "P" and abs(charges[0]["a"] - L / 2.0) < 1e-6 * L:
        return DV.KC_PONCTUEL
    return None


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
        M_max, M_min, V_max, V_min = [], [], [], []
        for x in grille:
            es = [c.lire(x) for c in combs]
            M_max.append(max(max(e["M_g"], e["M_d"]) for e in es))
            M_min.append(min(min(e["M_g"], e["M_d"]) for e in es))
            V_max.append(max(max(e["V_g"], e["V_d"]) for e in es))
            V_min.append(min(min(e["V_g"], e["V_d"]) for e in es))
        statique["enveloppes"][nom] = {"M_max": M_max, "M_min": M_min, "V_max": V_max, "V_min": V_min}

    def ec3() -> Dict:
        env = Enveloppe(EC.METHODE, EC.NORME)
        combs = []
        for co in CO.elu(cas, P):
            C = ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas)
            combs.append(C)
            xs = ST.sections(b, C, pts)
            for x in xs:                                         # la SECTION, là où les efforts sont ensemble
                for M, V in _cotes(C.lire(x)):
                    if abs(M) < EPS and abs(V) < EPS and abs(C.N) < EPS:
                        continue
                    r = ES.verifier({"section": s, "nuance": b["nuance"], "NEd": C.N * 1e3, "MyEd": M * 1e6, "VzEd": V * 1e3,
                                     "parametres": b["parametres"]})
                    env.verser(r, {"combinaison": co["nom"], "x": x})
            for i, tr in enumerate(trs):                         # la BARRE, tronçon par tronçon
                Mmax = _moment_max(C, tr, xs)
                if Mmax < EPS and C.N <= EPS:
                    continue
                rep, pon = ST.charges_du_troncon(cas, co["facteurs"], tr, zg)
                a = tr["x1"]
                troncon = {"L": (tr["x2"] - a) * 1e3, "moment": (lambda x, C=C, a=a: C.M(a + x / 1e3) * 1e6), "reparties": rep,
                           "ponctuelles": pon, "appuis": tr["appuis"], "maintiens": tr["maintiens"], "M_max": Mmax * 1e6}
                flexion = {"diagramme": DV.CONSTANT}
                if b["modele"] == "appuis" and Mmax > EPS:      # tableau A.2, formule générale : la flèche et le moment de la portée
                    fl = max(abs(C.lire(x)["f"] or 0.0) for x in grille)
                    flexion = {"diagramme": DV.GENERAL, "delta": fl * 1e3, "L": Lt * 1e3}
                r = EC.verifier({"section": s, "nuance": b["nuance"], "NEd": C.N * 1e3, "MyEd": Mmax * 1e6, "axes": axes,
                                 "flexion": {"y": flexion}, "parametres": b["parametres"],
                                 "deversement": {"troncon": troncon, "kc": _kc_tableau(b, cas, co["facteurs"], tr), "zg": zg}})
                env.verser(r, {"combinaison": co["nom"], "troncon": [tr["x1"], tr["x2"]]}, prefixes=IDS_BARRE)
        enveloppe_diagrammes("ec3", combs)
        r = env.rendre()
        ou = {"haut": "sur la semelle haute", "centre": "au centre de cisaillement", "bas": "sous la semelle basse"}[b["niveau"]]
        r["hypotheses"].append(f"Charges appliquées {ou} ; appuis en fourche (déplacement latéral et torsion empêchés).")
        return r

    def historique() -> Dict:
        env = Enveloppe(HI.METHODE, HI.NORME)
        h = b["historique"]
        fy = _fy_historique(b["nuance"])
        combs = []
        fleche = {"valeur": 0.0, "x": None, "combinaison": None}
        for co in CO.service(cas):
            C = ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas)
            combs.append(C)
            xs = ST.sections(b, C, pts)
            Lcs = [(tr, _Lc_historique(C, tr, xs)) for tr in trs]
            for x in xs:
                tr_x = next(((tr, Lc) for tr, Lc in Lcs if tr["x1"] - EPS <= x <= tr["x2"] + EPS), Lcs[-1])
                for M, V in _cotes(C.lire(x)):
                    if abs(M) < EPS and abs(V) < EPS and abs(C.N) < EPS:
                        continue
                    r = HI.verifier({"section": s, "fy": fy, "s": h["s"], "P": C.N * 1e3, "My": M * 1e6, "Vz": V * 1e3, "axes": axes,
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
            except RE.Refus as e:
                methodes[nom] = _refusee(nom, norme, str(e))
    return {"statique": statique, "methodes": methodes}


def _comparaison(methodes: Dict) -> List[Dict]:
    """Les deux méthodes côte à côte, sujet par sujet : le plus fort taux de chacune."""
    if len(methodes) < 2:
        return []

    def pire(r: Dict, ids: Sequence[str]) -> Optional[float]:
        t = [v["taux"] for v in r["verifications"] if v["id"] in ids and v["taux"] is not None]
        return max(t) if t else None

    lignes = [{"sujet": sujet, "historique": pire(methodes["historique"], h), "ec3": pire(methodes["ec3"], e)} for sujet, h, e in SUJETS]
    lignes = [li for li in lignes if li["historique"] is not None or li["ec3"] is not None]
    lignes.append({"sujet": "Le plus défavorable", "historique": methodes["historique"]["taux_max"], "ec3": methodes["ec3"]["taux_max"]})
    return lignes


def verifier(barre: Dict) -> Dict:
    """La barre vérifiée : {"barre" (relue), "profil", "masse" (kg/m), "statique" (None en efforts imposés),
    "methodes": {"historique": …, "ec3": …} (la forme de acier/barre/resultat.py ; en barre chargée, chaque
    vérification dit sa combinaison et son lieu), "statut", "comparaison"}."""
    b = EN.normaliser(barre)
    s = SE.du_catalogue(b["profil"])
    out = _efforts_imposes(b, s) if b["mode"] == "efforts" else _chargee(b, s)
    statuts = [r["statut"] for r in out["methodes"].values()]
    statut = RE.NOK if RE.NOK in statuts else RE.NON_COUVERT if RE.NON_COUVERT in statuts else RE.OK
    out.update({"barre": b, "profil": s["nom"], "masse": s.get("masse"), "statut": statut, "comparaison": _comparaison(out["methodes"])})
    return out

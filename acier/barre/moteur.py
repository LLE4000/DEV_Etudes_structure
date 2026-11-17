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
    Méthode historique : pour chaque combinaison de service (G + ΣQ) — dans chaque section à vérifier et aux deux
        bouts de chaque tronçon, avec la longueur de déversement L_c de son tronçon (à une frontière : le côté gauche
        avec le tronçon de gauche, le côté droit avec celui de droite).

L'enveloppe garde, pour chaque vérification, où elle a été trouvée (la combinaison, l'abscisse ou le tronçon) et les
formules de CETTE occurrence : la note imprime ce qui a décidé.

Ce que la barre chargée ne fait pas (v1) : la flexion d'axe faible et les charges horizontales (le mode « efforts »
les prend) ; la flèche n'est donnée qu'à titre d'information.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple


from acier import materiaux as MA

from . import ame as AME
from . import combinaisons as CO
from . import ec3 as EC
from . import ec3_deversement as DV
from . import ec3_section as ES
from . import entree as EN
from . import historique as HI
from . import mcr as MC
from . import parametres as PA
from . import prs as PR
from . import resultat as RE
from . import section as SE
from . import statique as ST
from .enveloppe import IDS_BARRE, ORDRE, Enveloppe          # noqa: F401 — ORDRE, relu par les tests

# la comparaison des deux méthodes : (sujet, vérifications historiques, vérifications de l'Eurocode)
# (la méthode historique n'a pas de vérification de l'effort tranchant seul : il entre dans la contrainte équivalente)
SUJETS = (("Flambement, axe faible", ("H-N-z",), ("E-Nb-z", "E-Nb-T")), ("Flambement, axe fort", ("H-N-y",), ("E-Nb-y",)),
          ("Flexion et contraintes, déversement compris", ("H-SIGMA", "H-DEV", "H-EQ"), ("E-My", "E-Mz", "E-MV", "E-NM", "E-Mb", "E-661", "E-662")),
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
    """La limite élastique de la méthode ELS : la NOMINALE de la nuance (235, 275, 355 N/mm²), quelle que soit
    l'épaisseur, comme les classeurs du bureau (brief v12 § 6.1, § 13.2 point 5, § 18.3) — l'Eurocode 3 garde la table
    selon l'épaisseur. Au-delà de 80 mm, la méthode se calcule (« hors domaine », brief § 19.1 point 5). Le DOMAINE de
    la doc (AE235 à AE355) se lit sur la nuance : elle est passée avec f_y. {"fy", "nuance", "fy_source", "fy_ref"}."""
    return {"fy": MA.nominale(nuance), "nuance": nuance, "fy_source": HI.SOURCE_NOMINALE, "fy_ref": HI.FY_NOMINALE}


def _deversement_historique(b: Dict, Ld: Optional[float], bout_libre: bool = False) -> Dict:
    """Le déversement de la méthode ELS : actif ou non, L_d (mm ; None : la longueur de l'axe faible), C_d, la variante
    (classeurs par défaut), le facteur 5,1 de la doc (profil fermé), une classe imposée (None : le tableau 12), le niveau
    des charges (le message de C_d > 1 sous une charge sur la semelle haute), et une console à bout libre (hors du
    domaine de la formule des classeurs : historique.verifier le dit)."""
    h = b["historique"]
    return {"actif": h["deversement"], "Lc": Ld, "Cd": h["Cd"], "variante": h["variante"], "facteur_doc": h["facteur_doc"],
            "classe": h["classe_d"], "niveau": b["niveau"], "bout_libre": bout_libre}


def _C_historique(mode: str, forme: str, psi: float) -> Dict:
    """Le coefficient C d'un axe : 1 (« unite », les classeurs) ou, selon la doc p. 20 et 21 (« norme ») : 0,85 sous des
    charges appliquées sur la barre, √(0,3·(1 + ψ²) + 0,4·ψ) sous des moments d'extrémité seuls."""
    if mode == "unite":
        return {"mode": "unite"}
    if forme in (DV.REPARTI, DV.PONCTUEL, DV.GENERAL):
        return {"mode": "charges"}
    return {"mode": "moments", "psi": 1.0 if forme == DV.CONSTANT else psi}


# une CONSOLE (encastrée – libre) : u = 2,1, la valeur RECOMMANDÉE du tableau d'aide (doc p. 16, brief § 13.3 ; 2 en
# théorie) — sur tout axe où son bout est libre, la longueur de flambement PROPOSÉE (champ L vide) est 2,1 × sa longueur ;
# u reste le multiplicateur saisi (1 par défaut : les notes enregistrées portent u = 1)
U_CONSOLE = 2.1
CONSOLE_LZ = "Console à bout libre : L_cr,z = 2,1·L, maintiens intermédiaires non comptés — À VÉRIFIER."


def _bout_libre(b: Dict, trs: Optional[List[Dict]]) -> bool:
    """Une console dont le bout n'est tenu par aucun maintien complet : son dernier tronçon finit LIBRE (statique.troncons)."""
    return b["modele"] == "console" and bool(trs) and trs[-1]["appuis"][1] == MC.LIBRE


def _longueurs(b: Dict, trs: Optional[List[Dict]]) -> Dict:
    """Les longueurs de flambement (mm) : celles données, sinon la plus grande travée (axe fort) et le plus long tronçon
    entre sections tenues (axe faible). Une CONSOLE flambe sur U_CONSOLE = 2,1 fois sa longueur (brief § 13.3,
    « encastrée – libre », valeur recommandée) autour de l'axe fort, et autour de l'axe faible tant que son bout est
    libre : il se déplace latéralement comme dans le plan. Un maintien complet au bout en fait une section tenue : la
    plus grande distance entre sections tenues (côté sûr). Des maintiens intermédiaires seuls ne raccourcissent rien :
    la longueur exacte est plus courte (3,77 m pour une console de 3 m tenue à mi-longueur, au lieu de 6,30) mais la
    partie libre ne se prend pas isolément sans justification — 2,1·L, hypothèse prudente, À VÉRIFIER (CONSOLE_LZ :
    dite une fois, et seulement sous compression ; l'ingénieur peut saisir la longueur)."""
    out = {}
    for a in ("y", "z"):
        L = b["flambement"][a]["L"]
        if L is None:
            L = max(b["travees"]) if a == "y" else max(t["x2"] - t["x1"] for t in trs)
            if a == "y" and b["modele"] == "console":
                L = round(U_CONSOLE * L, 9)                  # une console flambe sur 2,1 fois sa longueur (§ 13.3)
            if a == "z" and _bout_libre(b, trs):
                L = round(U_CONSOLE * sum(b["travees"]), 9)  # bout libre : 2,1·L, même avec des maintiens intermédiaires
        out[a] = {"L": L * 1e3, "u": b["flambement"][a]["u"]}
    return out


def _avis_longueurs(b: Dict, trs: Optional[List[Dict]]) -> List[Dict]:
    """L'alerte À VÉRIFIER d'une console à bout libre dont L_cr,z automatique ignore des maintiens complets intermédiaires
    (_chargee la donne à la première méthode COMPRIMÉE : sans N > 0, L_cr,z ne change aucun résultat)."""
    if b["flambement"]["z"]["L"] is None and _bout_libre(b, trs) and len(trs) > 1:
        return [{"code": "console_Lz", "niveau": "attention", "message": CONSOLE_LZ}]
    return []


def _longueurs_rendues(b: Dict, axes: Dict) -> Dict:
    """Les longueurs de flambement PRISES, en mètres : {"y": {"L", "u", "auto"}, "z": …} ; « auto » quand la barre ne
    la donne pas (l'écran l'écrit dans le champ vide)."""
    return {a: {"L": axes[a]["L"] / 1e3, "u": axes[a]["u"], "auto": b["flambement"][a]["L"] is None} for a in ("y", "z")}


def _bords_saisis(s: Dict) -> None:
    """Un PRS dont les bords des semelles ne sont pas saisis : la méthode ELS est BLOQUÉE (elle en tire ses courbes, brief
    § 18.2 ; test 3 des « autres tests PRS ») — Refus, que la boucle des méthodes rend « non couvert » avec ce motif ;
    l'Eurocode 3 se calcule."""
    if PR.est_prs(s) and s.get("fabrication") is None:
        raise RE.Refus(PR.MSG_BORDS)


def section_de(b: Dict) -> Dict:
    """La section de la barre relue : un PRS (ses plats, sa nuance — acier/barre/prs.py) ou un profilé du catalogue."""
    if b.get("prs"):
        return PR.section_prs(**b["prs"], nuance=b["nuance"])
    return SE.du_catalogue(b["profil"])


# ------------------------------------------------------------------ efforts imposés
def _efforts_imposes(b: Dict, s: Dict) -> Dict:
    axes = _longueurs(b, None)
    Lc = b["Lc"]
    methodes = {}

    def historique() -> Dict:
        _bords_saisis(s)
        e, h = b["efforts"]["service"], b["historique"]
        return HI.verifier({"section": s, **_fy_historique(b["nuance"], s), "s": h["s"], "P": e["N"] * 1e3, "My": e["My"] * 1e6, "Mz": e["Mz"] * 1e6,
                         "Vz": e["Vz"] * 1e3, "Vy": e["Vy"] * 1e3, "axes": axes, "tolerance": h["tolerance"],
                         "C": {a: _C_historique(h["C"], b["diagramme"][a]["forme"], b["diagramme"][a]["psi"]) for a in ("y", "z")},
                         "deversement": _deversement_historique(b, None if Lc is None else Lc * 1e3)})

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
    return abs(_moment_extreme(comb, tr, xs))


def _moment_extreme(comb: ST.Combinaison, tr: Dict, xs: Sequence[float]) -> float:
    """Le moment de plus grande valeur absolue du tronçon, AVEC son signe (kN·m ; M > 0 comprime la semelle haute) —
    un PRS monosymétrique en tire sa semelle comprimée."""
    a, b = tr["x1"], tr["x2"]
    pts = [a + (b - a) * i / BALAYAGE_TRONCON for i in range(BALAYAGE_TRONCON + 1)] + [x for x in xs if a < x < b]
    m = 0.0
    for x in pts:
        e = comb.lire(x)
        for v in ((e["M_g"],) if x > a + EPS else ()) + ((e["M_d"],) if x < b - EPS else ()):
            if abs(v) > abs(m):
                m = v
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


def _Lc_historique(comb: ST.Combinaison, tr: Dict, xs: Sequence[float], console_libre: Optional[float] = None) -> float:
    """La longueur de déversement de la méthode historique sur le tronçon (m) : la plus grande distance entre deux
    points où la semelle COMPRIMÉE est tenue. Un maintien d'une seule semelle ne compte que si elle est comprimée sur
    tout le tronçon (moment d'un seul signe). Une console à bout LIBRE (``console_libre`` : sa longueur, m) : le bout
    libre n'est pas une section tenue — U_CONSOLE × la longueur de la console, la règle de son flambement (§ 13.3),
    maintiens intermédiaires non comptés : hypothèse prudente, À VÉRIFIER (intégration du lot B : le bout libre comptait
    comme un maintien, L_d = la longueur de la console)."""
    if console_libre is not None:
        return round(U_CONSOLE * console_libre, 9)
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


def _cotes_historique(e: Dict, x: float, Lcs: Sequence[Tuple[Dict, float]]) -> List[Tuple[Tuple[Dict, float], Tuple[float, float]]]:
    """Les couples (M, V) de la section lue ``e`` (abscisse ``x``), chacun avec le tronçon (et son L_d) qui le vérifie dans
    la méthode ELS. À la FRONTIÈRE de deux tronçons, le côté gauche se vérifie avec le tronçon de gauche et le côté droit
    avec celui de droite, même quand le moment est le même des deux côtés : leurs L_d diffèrent (sinon le taux dépend du
    sens de dessin de la poutre). Ailleurs, le tronçon de la section, pour ses deux côtés (``ST.cotes``)."""
    gauche = next((t for t in Lcs if t[0]["x1"] + EPS < x <= t[0]["x2"] + EPS), None)
    droite = next((t for t in Lcs if t[0]["x1"] - EPS <= x < t[0]["x2"] - EPS), None)
    if gauche is not None and droite is not None and gauche is not droite:
        return [(gauche, (e["M_g"], e["V_g"])), (droite, (e["M_d"], e["V_d"]))]
    t = gauche or droite or Lcs[-1]
    return [(t, mv) for mv in ST.cotes(e, EPS)]


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
    if PR.est_prs(s):                                          # un PRS : au-dessus de SON centre de cisaillement
        zg = SE.cote(s, b["niveau"])
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
    comprimee: Dict[str, bool] = {}          # une combinaison de la méthode a-t-elle N > 0 ? (alors elle lit L_cr,z)

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
                kc = _kc_tableau(b, cas, co["facteurs"], tr, C)
                dev = {"troncon": troncon, "kc": kc, "zg": zg}
                if PR.est_prs(s):                                # le diagramme d'un cas des tableaux : le contrôle (E.1) d'un PRS
                    dev["diagramme"] = {DV.KC_REPARTI: DV.REPARTI, DV.KC_PONCTUEL: DV.PONCTUEL}.get(kc)
                barre_ec3 = {"section": s, "nuance": b["nuance"], "NEd": C.N * 1e3, "MyEd": Mmax * 1e6, "axes": axes,
                             "flexion": {"y": flexion}, "parametres": b["parametres"], "voilement": False, "console": b["modele"] == "console",
                             "deversement": dev}
                if PR.monosymetrique(s) and _moment_extreme(C, tr, xs) < 0:
                    barre_ec3["MyEd"] = -barre_ec3["MyEd"]       # la semelle BASSE est comprimée (PRS monosymétrique)
                r = EC.verifier(barre_ec3)
                env.verser(r, {"combinaison": co["nom"], "troncon": [tr["x1"], tr["x2"]]}, prefixes=IDS_BARRE)
        enveloppe_diagrammes("ec3", combs)
        comprimee["ec3"] = any(C.N > EPS for C in combs)
        r = env.rendre()
        ou = {"haut": "sur la semelle haute", "centre": "au centre de cisaillement", "bas": "sous la semelle basse"}[b["niveau"]]
        r["hypotheses"].append(f"Charges appliquées {ou} ; appuis en fourche (déplacement latéral et torsion empêchés).")
        return r

    def historique() -> Dict:
        _bords_saisis(s)
        env = Enveloppe(HI.METHODE, HI.NORME)
        h = b["historique"]
        fy = _fy_historique(b["nuance"], s)
        libre = sum(b["travees"]) if _bout_libre(b, trs) else None      # une console à bout libre : sa longueur (m)
        combs = []
        fleche = {"valeur": 0.0, "x": None, "combinaison": None}
        for co in CO.service(cas):
            C = ST.Combinaison(co["nom"], sols, co["facteurs"], N_cas)
            combs.append(C)
            xs = ST.sections(b, C, pts)
            Lcs = [(tr, _Lc_historique(C, tr, xs, libre)) for tr in trs]
            # les BOUTS de chaque tronçon sont des sections vérifiées : un maintien qui n'est ni un appui ni un point de
            # charge borne deux tronçons, et le plus grand moment d'un tronçon peut tomber à son bout
            for x in sorted(set(xs) | {tr[k] for tr in trs for k in ("x1", "x2")}):
                for (tr, Lc), (M, V) in _cotes_historique(C.lire(x), x, Lcs):
                    if abs(M) < EPS and abs(V) < EPS and abs(C.N) < EPS:
                        continue
                    r = HI.verifier({"section": s, **fy, "s": h["s"], "P": C.N * 1e3, "My": M * 1e6, "Vz": V * 1e3, "axes": axes,
                                     "C": {"y": _C_historique(h["C"], DV.GENERAL, 1.0)}, "tolerance": h["tolerance"],
                                     "deversement": _deversement_historique(b, Lc * 1e3, libre is not None)})
                    env.verser(r, {"combinaison": co["nom"], "x": x, "troncon": [tr["x1"], tr["x2"]]})
            for x in grille:                                     # la flèche de service, à titre d'information
                f = C.lire(x)["f"]
                if f is not None and abs(f) > abs(fleche["valeur"]):
                    fleche = {"valeur": f, "x": x, "combinaison": co["nom"]}
        enveloppe_diagrammes("historique", combs)
        comprimee["historique"] = any(C.N > EPS for C in combs)
        portee = max(b["travees"]) * (2.0 if b["modele"] == "console" else 1.0)
        fleche["valeur"] *= 1e3                                  # mm
        fleche["rapport"] = portee * 1e3 / abs(fleche["valeur"]) if fleche["valeur"] else None
        statique["fleche"] = fleche
        return env.rendre()

    avis = _avis_longueurs(b, trs)               # une donnée commune : dite UNE fois, sous la première méthode qui la lit
    for nom, calcul, norme in (("historique", historique, HI.NORME), ("ec3", ec3, EC.NORME)):
        if nom in b["methodes"]:
            try:
                methodes[nom] = calcul()
                if comprimee[nom]:
                    methodes[nom]["alertes"] += avis
                    avis = []                    # (l'autre méthode lit la même longueur : pas de redite)
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
    s = section_de(b)
    out = _efforts_imposes(b, s) if b["mode"] == "efforts" else _chargee(b, s)
    ec = out["methodes"].get("ec3")
    if ec and s.get("forme") == "I" and not AME.AM.est_soude(s):       # brief § 8.3, point 8 : un laminé dont le CISAILLEMENT
        # gouverne — V_pl,Rd ne passe pas (le remède dit), ou E-Vz a le plus fort taux de l'Eurocode 3 et de l'âme (À VÉRIFIER :
        # « gouverne » lu ainsi) — un raidisseur n'y peut rien ; sans force concentrée sur l'âme (aucun lieu), pas de question de
        # raidisseur : rien n'est dit tant qu'il passe
        autres = [x["taux"] for r_ in (ec, out["ame"]) for x in r_["verifications"] if x["id"] != "E-Vz" and x["taux"] is not None]
        for v in ec["verifications"]:
            if v["id"] == "E-Vz" and v["statut"] == RE.NOK:
                v["messages"].append(AME.AM.MOT_CISAILLEMENT)
            elif v["id"] == "E-Vz" and out["ame"]["lieux"] and v["taux"] is not None and v["taux"] >= max(autres, default=0.0):
                v["messages"].append(AME.AM.MOT_CISAILLEMENT_GOUVERNE)
    statuts = [r["statut"] for r in out["methodes"].values()] + [out["ame"]["statut"]]
    statut = RE.pire_statut(statuts)
    out.update({"barre": b, "profil": s["nom"], "masse": s.get("masse"), "poids_propre": ST.poids_propre(s), "statut": statut,
                "comparaison": _comparaison(out["methodes"], out["ame"])})
    return out

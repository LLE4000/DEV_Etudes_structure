# -*- coding: utf-8 -*-
"""La BARRE en acier telle que l'écran la garde : ses données, leurs défauts, leur relecture (docs/acier/BARRES_ACIER.md
§ 9). Unités de la saisie : m, kN, kN/m, kN·m — celles du bureau.

    {"nom": "B1", "profil": "IPE 300", "nuance": "S235",
     # --- un PRS (acier/barre/prs.py, brief § 18) au lieu d'un profilé du catalogue : par ses plats —
     "prs": {"b1", "t1", "hw", "tw", "b2", "t2" (mm), "a" (gorge, mm ; None : s = 0), "bords" (« oxycoupes »,
             « larges_plats » ou None), "nom"},
     #   — ou par son nom dans la bibliothèque des profilés personnalisés (le format du Catalogue) :
     "profil": "PRS 600x300x15x30", "bibliotheque": [{"famille": "PRS", "dims": {…}, "nuance": …}, …],
     "mode": "chargee" | "efforts",
     # --- barre CHARGÉE : la statique est faite par le module
     "modele": "appuis" | "console" | "enc_app" | "bi_enc" | "continue",      "travees": [6.0],
     "cas": [{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]}, …],
             une charge ponctuelle {"type": "P", "v", "a"} porte aussi "ss" : sa longueur d'appui rigide, en mm (None :
             non saisie — jamais supposée, brief § 8.2 ; 0 est une valeur saisie)
     "poids_propre": True,
     "niveau": "haut" | "centre" | "bas"      où les charges sont appliquées (déversement ; âme sous charge concentrée)
     "maintiens": [{"x": 3.0, "genre": "complet" | "haut" | "bas"}]
     "appuis": {"mode": "poses" | "ame", "ss": None, "c": None}    la barre POSÉE sur ses appuis (longueur d'appui rigide
                                              s_s et débord c en rive, mm ; None : non saisis — l'âme dit « décision
                                              impossible », brief § 8.3, point 5), ou tenue par l'âme (platine, cornières) ; « contact »
                                              (« oui » | « non » | « inconnu », à défaut) : le contact d'extrémité du type (c),
                                              EN 1993-1-5 § 6 — une charge ponctuelle peut porter le sien
     # --- EFFORTS IMPOSÉS : le mode des classeurs du bureau
     "efforts": {"service": {"N", "My", "Mz", "Vz", "Vy"}, "elu": {…}}      N > 0 en compression
     "diagramme": {"y": {"forme": "constant" | "lineaire" | "reparti" | "ponctuel", "psi": 1.0}, "z": {…}}
     # --- commun
     "flambement": {"y": {"L": None, "u": 1.0, "courbe": None}, "z": {…}}   L None : la plus grande travée (y) ;
                                                                              la plus grande distance entre maintiens (z) ;
                                                       console : 2,1·L autour des deux axes tant que le bout est libre ;
                                                       courbe : None (celle de la norme) | « a » | « b » | « c »
     "Lc": None                               distance entre sections tenues (efforts imposés ; None : L de l'axe faible)
     "methodes": ["historique", "ec3"],       (les anciennes notes : lue, elle ne décide plus du calcul — les DEUX méthodes sont
                                              toujours calculées, brief § 0 ; ce qu'une note imprime : api_web/profil.IMPRESSION)
     "historique": {"s": 1.5, "C": "unite" | "norme", "deversement": True, "tolerance": False,
                    "Cd": 1.0, "variante": "classeurs" | "fascicule",   C_d et la variante du déversement (brief § 6.4)
                    "facteur_doc": False, "classe_d": None}             profil fermé : le facteur 5,1 de la doc ; une classe
                                                                       de déversement imposée (a à d ; None ou "" : le tableau 12)
                    la tolérance de 5 % est RETIRÉE (brief § 13.2, point 5 : « non ») : « tolerance » est toujours False ; une
                    ancienne barre où elle était cochée garde « tolerance_retiree » True (la note le dit)
     "parametres": {}}                        d'autres coefficients que ceux de la table (acier/barre/parametres.py)

LE CONTRAT DES ENTRÉES DU LOT C (lot C, C0e : chaque clé a son défaut et son refus ; aucune ne change encore un calcul —
les blocs qui les emploient le diront) :
     "norme_produit": None (EN 10025-2) | « M » (M/ML, EN 10025-4)          "ouvrage": « batiment » | « autre »
     "combinaisons": {"expression": « 6.10 » | « 6.10ab », "geotechnique": False, "regle_courte_duree": True}
     "position": « horizontale » | « verticale » | « inclinee », "angle": None (degrés, 0 à 90)
     "points": [] (abscisses vérifiées en plus, m)       "debords": [0.0, 0.0] (m ; « appuis » et « continue »)
     "poids_propre_efforts": {"N": False, "M": False}    (efforts imposés : le poids propre dans P, en flexion)
     cas : "courte_duree" False, "psi0" None (0 à 1), "N_troncons" None | [{"x2", "N"}, …] (jusqu'au bout de la barre)
     charge : "niveau" None (celui de la barre) | « haut » | « centre » | « bas » ; ponctuelle : "introduction" « auto » |
              « a » | « b » | « c »
     maintien : "neglige" False (négligé dans M_cr)
     appuis : "detail" None | [None | {"ss", "c", "contact", "torsion" (« fourche » | « encastre »)}, …] (un par appui)
"""
import copy
import math
from typing import Dict, List, Optional

from acier.flambement import normes as NO
from rdm import poutres as PO

from . import combinaisons as CO
from . import ec3_deversement as DV
from . import historique_deversement as HD
from . import prs as PR
from . import section as SE
from . import statique as ST
from .resultat import Refus, fr


def _m(x: float) -> str:
    """Une longueur dans un refus, à la virgule, sans zéro inutile (« 6,2 »)."""
    t = fr(x, 3)
    return t.rstrip("0").rstrip(",") if "," in t else t

MODES = ("chargee", "efforts")
METHODES = ("historique", "ec3")
NIVEAUX = ("haut", "centre", "bas")
C_MODES = ("unite", "norme")        # C = 1 (les classeurs du bureau) ; ou C selon le diagramme (doc p. 20, 21)
EFFORTS = ("N", "My", "Mz", "Vz", "Vy")
CAS_MAX, MAINTIENS_MAX = 6, 24           # (lot C : 24 maintiens — le PRS-C du brief § 11.9 en demande 18)
MODES_APPUI = ("poses", "ame")            # la barre posée sur ses appuis ; ou tenue par l'âme (aucune force sur une semelle)
CONTACTS = ("oui", "non", "inconnu")      # contact d'extrémité du type (c), EN 1993-1-5 § 6 (brief § 4.1, § 8.2) : « inconnu » à défaut
SS_MAX = 2000.0                           # mm : la plus grande longueur d'appui rigide, le plus grand débord
L_MIN, L_MAX = 0.05, 60.0                 # m
FORMES_DIAGRAMME = (DV.CONSTANT, DV.LINEAIRE, DV.REPARTI, DV.PONCTUEL)
# ---- le contrat des entrées du lot C (C0e) : les valeurs admises, la première est le défaut
NORMES_PRODUIT = (None, "M")              # None : EN 10025-2 (S420 et S460 : N/NL, EN 10025-3) ; « M » : M/ML, EN 10025-4
EXPRESSIONS = ("6.10", "6.10ab")          # (6.10) ; ou (6.10a) et (6.10b), l'enveloppe (brief § 5)
POSITIONS = ("horizontale", "verticale", "inclinee")       # la barre, pour son poids propre (brief § 4.1)
INTRODUCTIONS = ("auto", "a", "b", "c")   # comment une ponctuelle entre dans l'âme (EN 1993-1-5, figure 6.1) : « auto », la règle du module
TORSIONS = ("fourche", "encastre")        # la condition de torsion d'un appui (fourche par défaut, brief § 4.1)
COURBES_IMPOSEES = (None, "a", "b", "c")  # la courbe de flambement d'un axe (None : celle de la norme)
POINTS_MAX = 12


def nouvelle_barre(nom: str = "B1") -> Dict:
    return {"nom": nom, "profil": "IPE 300", "nuance": "S235", "mode": "chargee",
            "modele": "appuis", "travees": [6.0],
            "cas": [{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]},
                    {"nom": "Q", "nature": "exploitation_A", "N": 0.0, "charges": [{"type": "q", "v": 5.0, "a": 0.0, "b": 6.0}]}],
            "poids_propre": True, "niveau": "haut", "maintiens": [], "appuis": {"mode": "poses", "ss": None, "c": None, "detail": None},
            "efforts": {"service": {k: 0.0 for k in EFFORTS}, "elu": {k: 0.0 for k in EFFORTS}},
            "diagramme": {"y": {"forme": DV.CONSTANT, "psi": 1.0}, "z": {"forme": DV.CONSTANT, "psi": 1.0}},
            "flambement": {"y": {"L": None, "u": 1.0, "courbe": None}, "z": {"L": None, "u": 1.0, "courbe": None}}, "Lc": None,
            "methodes": list(METHODES), "historique": {"s": 1.5, "C": "unite", "deversement": True, "tolerance": False, "Cd": 1.0,
                                                       "variante": "classeurs", "facteur_doc": False, "classe_d": None},
            "parametres": {},
            # (lot C, C0e) le contrat des entrées : leurs défauts
            "prs": None, "norme_produit": None, "ouvrage": DV.OUVRAGES[0],
            "combinaisons": {"expression": EXPRESSIONS[0], "geotechnique": False, "regle_courte_duree": True},
            "position": POSITIONS[0], "angle": None, "points": [], "debords": [0.0, 0.0], "poids_propre_efforts": {"N": False, "M": False}}


def _nombre(v, nom: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise Refus(f"{nom} : nombre attendu, reçu {v!r}")
    return float(v)


def _longueur_appui(v, nom: str) -> Optional[float]:
    """Une longueur d'appui rigide ou un débord, en mm : de 0 à SS_MAX ; None (non saisi) reste None — jamais supposé
    (brief § 8.2 : s_s est une valeur saisie)."""
    if v is None:
        return None
    x = _nombre(v, nom)
    if not 0.0 <= x <= SS_MAX:
        raise Refus(f"{nom} : de 0 à {SS_MAX:g} mm")
    return x


def _oui_non(v, nom: str) -> bool:
    """Une case : True ou False — rien d'autre (un texte « non » serait vrai)."""
    if not isinstance(v, bool):
        raise Refus(f"{nom} : oui ou non (true / false) attendu, reçu {v!r}")
    return v


def _parmi(v, valeurs, nom: str):
    if v not in valeurs:
        raise Refus(f"{nom} : {' ; '.join('vide' if x is None else f'« {x} »' for x in valeurs)} — reçu {v!r}")
    return v


def _contrat_commun(src: Dict, b: Dict) -> None:
    """(lot C, C0e) Les clés communes aux deux modes : la norme produit, le type d'ouvrage."""
    b["norme_produit"] = _parmi(src.get("norme_produit", b["norme_produit"]), NORMES_PRODUIT, "norme produit")
    b["ouvrage"] = _parmi(src.get("ouvrage", b["ouvrage"]), DV.OUVRAGES, "type d'ouvrage")


def _contrat_chargee(src: Dict, b: Dict, Lt: float) -> None:
    """(lot C, C0e) Les clés de la barre chargée : combinaisons, position et angle, points vérifiés, débords."""
    co = src.get("combinaisons", b["combinaisons"])
    if not isinstance(co, dict) or set(co) - set(b["combinaisons"]):
        raise Refus(f"combinaisons : {{{', '.join(b['combinaisons'])}}} attendu")
    co = dict(b["combinaisons"], **co)
    b["combinaisons"] = {"expression": _parmi(co["expression"], EXPRESSIONS, "expression des combinaisons"),
                         "geotechnique": _oui_non(co["geotechnique"], "application géotechnique"),
                         "regle_courte_duree": _oui_non(co["regle_courte_duree"], "règle de courte durée")}
    b["position"] = _parmi(src.get("position", b["position"]), POSITIONS, "position de la barre")
    angle = src.get("angle", b["angle"])
    if angle is not None and not 0.0 <= _nombre(angle, "angle de la barre") <= 90.0:
        raise Refus("angle de la barre : de 0 à 90°")
    b["angle"] = None if angle is None else float(angle)
    pts = src.get("points", b["points"])
    if not isinstance(pts, (list, tuple)) or len(pts) > POINTS_MAX:
        raise Refus(f"points vérifiés : une liste d'au plus {POINTS_MAX} abscisses")
    b["points"] = sorted(PO.ramener(_nombre(x, "abscisse d'un point vérifié"), Lt) for x in pts)
    if any(not 0.0 <= x <= Lt for x in b["points"]):
        raise Refus(f"un point vérifié est hors de la barre (0 à {_m(Lt)} m)")
    db = src.get("debords", b["debords"])
    if not isinstance(db, (list, tuple)) or len(db) != 2:
        raise Refus("débords : deux longueurs, à gauche et à droite (m)")
    b["debords"] = [_nombre(x, "débord") for x in db]
    if any(not 0.0 <= x <= L_MAX for x in b["debords"]):
        raise Refus(f"débords : de 0 à {_m(L_MAX)} m")
    if any(b["debords"]) and b["modele"] not in ("appuis", "continue"):
        raise Refus("débords : seulement pour une barre sur deux appuis ou continue")


def _contrat_cas(c: Dict, nom: str, Lt: float) -> Dict:
    """(lot C, C0e) Les clés d'un cas de charge : courte durée, ψ₀ saisi, N par tronçon."""
    out = {"courte_duree": _oui_non(c.get("courte_duree", False), f"cas « {nom} », courte durée")}
    psi0 = c.get("psi0")
    if psi0 is not None and not 0.0 <= _nombre(psi0, f"cas « {nom} », ψ₀") <= 1.0:
        raise Refus(f"cas « {nom} » : ψ₀ de 0 à 1")
    out["psi0"] = None if psi0 is None else float(psi0)
    nt = c.get("N_troncons")
    if nt is not None:
        if not isinstance(nt, (list, tuple)) or not nt or any(not isinstance(t, dict) or set(t) != {"x2", "N"} for t in nt):
            raise Refus(f"cas « {nom} », N par tronçon : une liste de {{x2, N}} jusqu'au bout de la barre")
        nt = [{"x2": PO.ramener(_nombre(t["x2"], f"cas « {nom} », fin d'un tronçon de N"), Lt),
               "N": _nombre(t["N"], f"cas « {nom} », N d'un tronçon")} for t in nt]
        xs = [t["x2"] for t in nt]
        if any(not 0.0 < x <= Lt for x in xs) or any(v <= u for u, v in zip(xs, xs[1:])) or abs(xs[-1] - Lt) > 1e-9:
            raise Refus(f"cas « {nom} », N par tronçon : des fins croissantes, la dernière au bout de la barre ({_m(Lt)} m)")
    out["N_troncons"] = nt
    return out


def _contrat_appuis(detail, b: Dict) -> Optional[List]:
    """(lot C, C0e) Les appuis un par un : None (la valeur commune), ou une liste — un par appui, None ou {"ss", "c", "contact",
    "torsion"} (chaque clé absente : la valeur commune ; torsion : fourche à défaut)."""
    if detail is None:
        return None
    if not isinstance(detail, (list, tuple)) or len(detail) > len(b["travees"]) + 1:
        raise Refus(f"appuis un par un : une liste d'au plus {len(b['travees']) + 1} appuis")
    out = []
    for i, d in enumerate(detail):
        if d is None:
            out.append(None)
            continue
        if not isinstance(d, dict) or set(d) - {"ss", "c", "contact", "torsion"}:
            raise Refus(f"appui {i + 1} : {{ss, c, contact, torsion}} attendu")
        e = {}
        for k in ("ss", "c"):
            if k in d:
                e[k] = _longueur_appui(d[k], f"appui {i + 1}, {'longueur d\'appui s_s' if k == 'ss' else 'débord c'}")
        if "contact" in d:
            e["contact"] = _parmi(d["contact"], CONTACTS, f"appui {i + 1}, contact d'extrémité")
        e["torsion"] = _parmi(d.get("torsion", TORSIONS[0]), TORSIONS, f"appui {i + 1}, torsion")
        out.append(e)
    return out


PLATS = ("b1", "t1", "hw", "tw", "b2", "t2", "a", "bords", "nom")


def _plats(prs: Dict) -> Dict:
    return {k: prs.get(k) for k in PLATS}


def _prs(src: Dict, b: Dict):
    """Le PRS de la barre — {"b1", …, "a", "bords", "nom"} — saisi par ses plats (« prs ») ou trouvé par son nom
    (« profil ») dans la bibliothèque des profilés personnalisés (« bibliotheque ») ; None pour un profilé du catalogue.
    Refus pour une saisie impossible (acier/barre/prs.py : contrôles bloquants du brief § 18.2). Une nuance portée par
    l'enregistrement de la bibliothèque passe devant celle de la barre (une seule nuance par PRS)."""
    p = src.get("prs")
    if p is None and isinstance(b["profil"], str) and SE.CAT.par_nom().get(b["profil"]) is None and src.get("bibliotheque"):
        if not isinstance(src["bibliotheque"], (list, tuple)):
            raise Refus("« bibliotheque » : une liste de profilés personnalisés")
        s = PR.chercher(src["bibliotheque"], b["profil"], b["nuance"])
        if s is None:
            raise Refus(f"profilé inconnu du catalogue et de la bibliothèque : {b['profil']!r}")
        b["nuance"] = s["nuance"]
        q = s["prs"]
        p = {"b1": q["b1"], "t1": q["t1"], "hw": q["hw"], "tw": q["tw"], "b2": q["b2"], "t2": q["t2"], "a": q["a"], "bords": s["bords"],
             "nom": s["nom"]}
    if p is None:
        return None
    if not isinstance(p, dict):
        raise Refus("« prs » : les plats du PRS (b1, t1, hw, tw, b2, t2, a, bords)")
    inconnus = sorted(set(p) - set(PLATS))
    if inconnus:
        raise Refus(f"PRS : donnée(s) inconnue(s) : {', '.join(inconnus)}")
    out = _plats(p)
    PR.section_prs(**out, nuance=b["nuance"])                    # les contrôles bloquants
    return out


def normaliser(barre: Dict) -> Dict:
    """La barre complétée de ses défauts et contrôlée (une COPIE) ; Refus dit ce qui ne va pas."""
    if not isinstance(barre, dict):
        raise Refus("la barre doit être un dictionnaire")
    b = nouvelle_barre(str(barre.get("nom") or "B1"))
    src = copy.deepcopy(barre)
    for k in ("profil", "nuance", "mode", "modele", "poids_propre", "niveau", "Lc"):
        if k in src:
            b[k] = src[k]
    if b["mode"] not in MODES:
        raise Refus(f"mode inconnu : {b['mode']!r}")
    if b["nuance"] not in NO.NUANCES:
        raise Refus(f"nuance inconnue : {b['nuance']!r}")
    prs = _prs(src, b)
    b["prs"] = prs                                               # (None : un profilé du catalogue — l'écran la garde, lot C)
    if prs is not None:
        b["profil"] = PR.section_prs(**_plats(prs), nuance=b["nuance"])["nom"]
    elif not isinstance(b["profil"], str) or SE.CAT.par_nom().get(b["profil"]) is None:
        raise Refus(f"profilé inconnu du catalogue : {b['profil']!r}")
    if b["niveau"] not in NIVEAUX:
        raise Refus(f"niveau des charges : {', '.join(NIVEAUX)} — reçu {b['niveau']!r}")
    b["poids_propre"] = bool(b["poids_propre"])
    _contrat_commun(src, b)
    if b["Lc"] is not None and not _nombre(b["Lc"], "L_c") >= 0:
        raise Refus("la distance L_c entre sections tenues doit être positive ou nulle")
    meth = src.get("methodes", list(METHODES))
    if not isinstance(meth, (list, tuple)) or not meth or any(m not in METHODES for m in meth):
        raise Refus(f"méthodes : une ou deux parmi {', '.join(METHODES)}")
    b["methodes"] = [m for m in METHODES if m in meth]
    h = dict(b["historique"], **(src.get("historique") or {}))
    if not 1.0 <= _nombre(h["s"], "s") <= 3.0:
        raise Refus("le coefficient de sécurité s doit être entre 1 et 3")
    if h["C"] not in C_MODES:
        raise Refus(f"coefficient C : {', '.join(C_MODES)}")
    if not HD.CD_MIN <= _nombre(h["Cd"], "C_d") <= HD.CD_MAX:
        raise Refus(f"le coefficient de moment C_d du déversement doit être entre {HD.CD_MIN:g} et {HD.CD_MAX:g}")
    if h["variante"] not in HD.VARIANTES:
        raise Refus(f"variante du déversement : {', '.join(HD.VARIANTES)}")
    if h["classe_d"] == "":                 # une liste sans choix (l'écran) : celle du tableau 12
        h["classe_d"] = None
    if h["classe_d"] is not None and h["classe_d"] not in HD.CLASSES:
        raise Refus(f"classe de déversement : {', '.join(HD.CLASSES)} ou vide (celle du tableau 12)")
    # la tolérance de 5 % est RETIRÉE (brief § 13.2, point 5 : « non », une seule valeur) : toujours False ; une ancienne barre où
    # elle était cochée le garde dans « tolerance_retiree » (la note le dit, lot C)
    b["historique"] = {"s": float(h["s"]), "C": h["C"], "deversement": bool(h["deversement"]), "tolerance": False,
                       "Cd": float(h["Cd"]), "variante": h["variante"], "facteur_doc": bool(h["facteur_doc"]), "classe_d": h["classe_d"],
                       "tolerance_retiree": bool(h["tolerance"]) or bool(h.get("tolerance_retiree"))}
    if not isinstance(src.get("parametres", {}), dict):
        raise Refus("« parametres » doit être un dictionnaire")
    b["parametres"] = dict(src.get("parametres") or {})
    fl = src.get("flambement") or {}
    for a in ("y", "z"):
        ax = dict(b["flambement"][a], **(fl.get(a) or {}))
        if ax["L"] is not None and not _nombre(ax["L"], f"longueur de flambement {a}") >= 0:
            raise Refus(f"axe {a} : la longueur de flambement doit être positive ou nulle")
        if not _nombre(ax["u"], f"coefficient u de l'axe {a}") > 0:
            raise Refus(f"axe {a} : le coefficient u doit être positif")
        b["flambement"][a] = {"L": None if ax["L"] is None else float(ax["L"]), "u": float(ax["u"]),
                              "courbe": _parmi(ax.get("courbe"), COURBES_IMPOSEES, f"courbe de flambement de l'axe {a}")}
    if b["mode"] == "efforts":
        ef = src.get("efforts") or {}
        for jeu in ("service", "elu"):
            for k in EFFORTS:
                b["efforts"][jeu][k] = _nombre((ef.get(jeu) or {}).get(k, 0.0), f"{k} ({jeu})")
        dg = src.get("diagramme") or {}
        for a in ("y", "z"):
            d = dict(b["diagramme"][a], **(dg.get(a) or {}))
            if d["forme"] not in FORMES_DIAGRAMME:
                raise Refus(f"diagramme du moment {a} : {', '.join(FORMES_DIAGRAMME)}")
            if not -1.0 <= _nombre(d["psi"], "ψ") <= 1.0:
                raise Refus("ψ = M₂/M₁ doit être entre −1 et 1")
            b["diagramme"][a] = {"forme": d["forme"], "psi": float(d["psi"])}
        for a in ("y", "z"):
            if b["flambement"][a]["L"] is None:
                raise Refus(f"efforts imposés : la longueur de flambement de l'axe {a} est à donner")
        pp = src.get("poids_propre_efforts", b["poids_propre_efforts"])
        if not isinstance(pp, dict) or set(pp) - {"N", "M"}:
            raise Refus("poids propre des efforts imposés : {N, M} attendu")
        pp = dict(b["poids_propre_efforts"], **pp)
        b["poids_propre_efforts"] = {k: _oui_non(pp[k], f"poids propre {'dans P' if k == 'N' else 'en flexion'}") for k in ("N", "M")}
        return b
    # ---- barre chargée
    if b["modele"] not in PO.MODELES:
        raise Refus(f"modèle inconnu : {b['modele']!r}")
    tr = src.get("travees", b["travees"])
    if not isinstance(tr, (list, tuple)) or not tr:
        raise Refus("la barre a au moins une travée")
    b["travees"] = [_nombre(x, "portée") for x in tr]
    if any(not L_MIN <= x <= L_MAX for x in b["travees"]):
        raise Refus(f"chaque portée doit être entre {L_MIN:g} et {L_MAX:g} m")
    if b["modele"] == "continue":
        if not 2 <= len(b["travees"]) <= PO.TRAVEES_MAX:
            raise Refus(f"une poutre continue a de 2 à {PO.TRAVEES_MAX} travées")
    elif len(b["travees"]) != 1:
        raise Refus("ce modèle n'a qu'une travée")
    Lt = sum(b["travees"])
    _contrat_chargee(src, b, Lt)
    cas = src.get("cas", b["cas"])
    if not isinstance(cas, (list, tuple)) or not cas or len(cas) > CAS_MAX:
        raise Refus(f"de 1 à {CAS_MAX} cas de charge")
    b["cas"] = []
    noms = set()
    for i, c in enumerate(cas):
        if not isinstance(c, dict):
            raise Refus(f"cas {i + 1} : dictionnaire attendu")
        nom = str(c.get("nom") or f"C{i + 1}")
        if nom in noms or nom == ST.POIDS_PROPRE:
            raise Refus(f"deux cas de charge portent le nom « {nom} »")
        noms.add(nom)
        if c.get("nature") not in CO.NATURES:
            raise Refus(f"cas « {nom} » : nature inconnue ({c.get('nature')!r})")
        try:
            _, charges = PO.normaliser(b["modele"], b["travees"], c.get("charges") or [])
        except PO.ErreurDonnees as e:
            raise Refus(f"cas « {nom} » : {e}") from None
        charges = [{k: ch[k] for k in ("type", "v", "a", "b", "sens") if k in ch} for ch in charges]
        for ch, saisie in zip(charges, c.get("charges") or []):          # (une charge rendue par charge saisie, dans l'ordre)
            # (lot C, C0e) le niveau d'application de CETTE charge (None : celui de la barre)
            ch["niveau"] = _parmi(saisie.get("niveau"), (None,) + NIVEAUX, f"cas « {nom} », niveau d'une charge")
            if ch["type"] == "P":
                ch["introduction"] = _parmi(saisie.get("introduction", INTRODUCTIONS[0]), INTRODUCTIONS, f"cas « {nom} », introduction d'une ponctuelle")
                ch["ss"] = _longueur_appui(saisie.get("ss"), f"cas « {nom} », longueur d'appui s_s")
                if "contact" in saisie:                                     # (lot B) gardé seulement s'il est saisi
                    if saisie["contact"] not in CONTACTS:
                        raise Refus(f"cas « {nom} », contact d'extrémité : {', '.join(CONTACTS)}")
                    ch["contact"] = saisie["contact"]
        b["cas"].append({"nom": nom, "nature": c["nature"], "N": _nombre(c.get("N", 0.0), f"effort normal du cas « {nom} »"), "charges": charges,
                         **_contrat_cas(c, nom, Lt)})
    ap = dict(b["appuis"], **(src.get("appuis") or {})) if isinstance(src.get("appuis") or {}, dict) else None
    if ap is None or ap["mode"] not in MODES_APPUI:
        raise Refus(f"appuis : {', '.join(MODES_APPUI)}")
    b["appuis"] = {"mode": ap["mode"], "ss": _longueur_appui(ap["ss"], "longueur d'appui s_s"), "c": _longueur_appui(ap["c"], "débord c"),
                   "detail": _contrat_appuis(ap.get("detail"), b)}
    if "contact" in ap:                                                     # (lot B) gardé seulement s'il est saisi : « inconnu » à défaut
        if ap["contact"] not in CONTACTS:
            raise Refus(f"appuis, contact d'extrémité : {', '.join(CONTACTS)}")
        b["appuis"]["contact"] = ap["contact"]
    mts = src.get("maintiens", [])
    if not isinstance(mts, (list, tuple)) or len(mts) > MAINTIENS_MAX:
        raise Refus(f"au plus {MAINTIENS_MAX} maintiens")
    b["maintiens"] = []
    for m in mts:
        x = PO.ramener(_nombre((m or {}).get("x"), "abscisse d'un maintien"), Lt)      # « 6,2 » sur 2,8 + 3,4 m : le bout
        genre = (m or {}).get("genre", "complet")
        if genre not in ST.GENRES_MAINTIEN:
            raise Refus(f"maintien : genre parmi {', '.join(ST.GENRES_MAINTIEN)}")
        if not 0.0 <= x <= Lt:
            raise Refus(f"maintien à {x:g} m : hors de la barre (0 à {Lt:g} m)")
        b["maintiens"].append({"x": x, "genre": genre, "neglige": _oui_non((m or {}).get("neglige", False), "maintien négligé dans M_cr")})
    b["maintiens"].sort(key=lambda m: m["x"])
    return b

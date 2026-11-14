# -*- coding: utf-8 -*-
"""Le module « Choix de profilé » du site : la vérification d'une BARRE en acier — poutre, poteau, poteau-poutre — par
la méthode historique du bureau (NBN B 51-001, charges de service), par l'Eurocode 3 (NBN EN 1993-1-1, ELU), ou par les
deux, côte à côte (docs/acier/BARRES_ACIER.md). Le moteur : ``acier/barre``.

Une note garde une ou plusieurs BARRES. Chacune se décrit de deux façons :
- « barre chargée » : un modèle (deux appuis, console, encastrée, continue), des portées, des cas de charge, des
  maintiens — le module fait la statique, les combinaisons, et cherche la situation la plus défavorable ;
- « efforts imposés » (les classeurs du bureau) : les efforts de service et de calcul sont donnés.

L'écran est l'établi COMPACT (api_web/etabli_compact.py) : à gauche la ligne de la barre et ses quatre rubriques
(api_web/profil_saisie.py) ; au centre la barre et ses enveloppes, la section, les taux (api_web/dessin_barre.py,
dessin_profil.py) ; à droite un verdict par méthode, le profilé, un bloc par méthode, leur comparaison, et les fenêtres
de calcul (api_web/profil_resultat.py).

Ce fichier tient l'ÉTAT de la note (``{"version": 2, "elements": [barre…], "actif": i}``), relit les anciennes notes
(version 1 : des efforts de calcul et l'Eurocode seul — elles deviennent des barres en « efforts imposés »), écrit les
saisies et sert le sélecteur de profilés. Il ne calcule rien : toute saisie est relue par ``acier.barre.entree`` — hors
domaine, elle est REFUSÉE avec le message du moteur et la barre reste intacte.
"""
import copy
import json
from typing import Dict, List, Optional, Sequence, Tuple

from acier.barre import combinaisons as CO
from acier.barre import entree as EN
from acier.barre import moteur as MO
from acier.barre import parametres as PA
from acier.barre import resultat as RE
from acier.barre import section as SE
from acier.flambement import normes as NO
from acier.profiles import catalogue as CAT
from acier.profiles import familles as FA
from rdm import poutres as PO

from . import commun as C
from . import dessin_barre as DB
from . import dessin_profil as DP
from . import etabli_compact as E
from . import profil_mots as M
from . import profil_resultat as R
from . import profil_saisie as S

MODULE = "profil"
PAGE = "choix-profile"
VERSION = 2

ROLE = "profil"                 # le seul sélecteur du module : le profilé de la barre ouverte
GENRE = E.Genre("Barres", "la barre", "cette barre", "une barre")
NOM_DEFAUT = "B1"
NOM_MAX = 60                    # caractères du nom d'une barre : la borne des éléments du site (dalle, cornière, fondation de grue)
NOM_CAS_MAX = 12                # … et du nom d'un cas de charge : il s'écrit à gauche de sa bande, sur le dessin de la barre
GAMMA_MIN, GAMMA_MAX = 0.9, 1.5
# les familles du catalogue que le module vérifie par ses deux méthodes : I et H laminés, tubes
FAMILLES = tuple(f for f, (forme, _) in SE.FAMILLES.items() if forme in ("I",) + SE.TUBES)
# ce qui appartient à chaque mode (l'autre est gardé tel quel dans la note : changer de mode ne perd rien)
COMMUNS = ("profil", "nuance", "mode", "poids_propre", "niveau", "Lc", "methodes", "historique", "parametres", "flambement")
CHARGEE = ("modele", "travees", "cas", "maintiens")
IMPOSES = ("efforts", "diagramme")
CLES = ("nom",) + COMMUNS + CHARGEE + IMPOSES
MEMOIRE, MEMOIRE_CANDIDATS = 48, 6000
_RESULTATS: Dict[str, Dict] = {}                    # la barre (sans son nom) → son résultat
_CANDIDATS: Dict[Tuple[str, str], Tuple] = {}       # (la barre sans nom ni profilé, un profilé) → (taux, statut)


def barre_neuve(nom: str = NOM_DEFAUT) -> Dict:
    """La barre d'une note neuve : celle du moteur (IPE 300 sur deux appuis, 6 m, G et Q de 5 kN/m, poids propre), tenue
    à mi-portée — sans maintien, la méthode historique la refuse d'emblée."""
    b = EN.nouvelle_barre(nom)
    b["maintiens"] = [{"x": 3.0, "genre": "complet"}]
    return b


# ------------------------------------------------------------------ le parcours propre : ce qu'un clic ne suffit pas à atteindre
_POTEAU = dict(barre_neuve("P1"), profil="HEB 200", mode="efforts", flambement={"y": {"L": 5.0, "u": 1.0}, "z": {"L": 5.0, "u": 1.0}}, Lc=5.0,
               efforts={"service": {"N": 300.0, "My": 20.0, "Mz": 5.0, "Vz": 30.0, "Vy": 0.0}, "elu": {"N": 420.0, "My": 28.0, "Mz": 7.0, "Vz": 42.0, "Vy": 0.0}})
_CONTINUE = dict(barre_neuve("C1"), profil="HEA 200", modele="continue", travees=[5.0, 4.0], maintiens=[{"x": 2.5, "genre": "haut"}],
                 cas=[{"nom": "G", "nature": "permanent", "N": 0.0, "charges": [{"type": "q", "v": 6.0, "a": 0.0, "b": 9.0}]},
                      {"nom": "Q", "nature": "exploitation_B", "N": 0.0, "charges": [{"type": "q", "v": 8.0, "a": 0.0, "b": 5.0},
                                                                                     {"type": "P", "v": 12.0, "a": 7.0}]},
                      {"nom": "S", "nature": "neige", "N": 0.0, "charges": [{"type": "tri", "v": 3.0, "a": 5.0, "b": 9.0, "sens": "croissante"}]}])
_ANCIENNE = {"version": 1, "actif": 0, "elements": [{"nom": "P1", "acier": "S355", "familles": ["HEB"], "My": 60.0, "Vz": 40.0, "avec_N": True,
                                                     "N": 600.0, "flambement": True, "Lcr_y": 6.0, "Lcr_z": 3.0, "courbe": "c", "profil": "HEB 200"}]}
PARCOURS = [
    {"action": "calculer", "etat": {"version": VERSION, "actif": 0, "elements": [_POTEAU]}},
    {"action": "calculer", "etat": {"version": VERSION, "actif": 0, "elements": [_CONTINUE]}},
    {"action": "calculer", "etat": _ANCIENNE},
    {"action": "selecteur", "role": ROLE, "familles": ["HEA", "CHS"]},
    {"action": "choisir", "role": ROLE, "nom": "HEA 220"},
    {"action": "modifier", "chemin": ["cas", "+"], "valeur": "1"},
    {"action": "modifier", "chemin": ["cas", "1", "-"], "valeur": "1"},
    {"action": "modifier", "chemin": ["cas", "0", "charges", "+"], "valeur": "P"},
    {"action": "modifier", "chemin": ["cas", "0", "charges", "0", "-"], "valeur": "1"},
    {"action": "modifier", "chemin": ["cas", "0", "charges", "0", "type"], "valeur": "tri"},
    {"action": "modifier", "chemin": ["maintiens", "+"], "valeur": "1"},
    {"action": "modifier", "chemin": ["maintiens", "0", "-"], "valeur": "1"},
    {"action": "modifier", "chemin": ["travees", "0"], "valeur": "4,5"},
    {"action": "modifier", "chemin": ["travees", "+"], "valeur": "1", "etat": {"version": VERSION, "actif": 0, "elements": [_CONTINUE]}},
    {"action": "modifier", "chemin": ["travees", "0", "-"], "valeur": "1", "etat": {"version": VERSION, "actif": 0, "elements": [dict(_CONTINUE, travees=[5.0, 4.0, 3.0])]}},
    {"action": "modifier", "chemin": ["flambement", "z", "L"], "valeur": ""},
    {"action": "modifier", "chemin": ["flambement", "y", "L"], "valeur": "abc"},
    {"action": "modifier", "chemin": ["travees", "0"], "valeur": "-1"},
    {"action": "modifier", "chemin": ["parametres", "gamma_M1"], "valeur": "1,1"},
    {"action": "modifier", "chemin": ["efforts", "elu", "N"], "valeur": "900", "etat": {"version": VERSION, "actif": 0, "elements": [_POTEAU]}},
    {"action": "modifier", "chemin": ["nom"], "valeur": "Panne"},
]


# ------------------------------------------------------------------ l'état de la note : des barres
def nouvel_etat() -> Dict:
    return {"version": VERSION, "elements": [barre_neuve()], "actif": 0}


def _nombre(x, defaut: float = 0.0) -> float:
    v = C.lire_nombre(x)
    return defaut if v is None else v


def _migrer(e: Dict) -> Dict:
    """Une barre d'une note de la version 1 (des efforts de CALCUL, l'Eurocode seul, des longueurs de flambement) : elle
    devient une barre en « efforts imposés », Eurocode 3 seul. Une vérification que l'ancienne note n'activait pas garde
    sa longueur à 0 (axe tenu, pas de déversement) : la note relue vérifie ce qu'elle vérifiait."""
    if not isinstance(e, dict):
        raise ValueError("barre illisible")
    b = barre_neuve(str(e.get("nom") or NOM_DEFAUT))
    b.update({"mode": "efforts", "methodes": ["ec3"], "nuance": e.get("acier") if e.get("acier") in NO.NUANCES else "S235"})
    elu = b["efforts"]["elu"]
    elu["My"], elu["Vz"] = abs(_nombre(e.get("My"))), abs(_nombre(e.get("Vz")))
    for case, cle, absolu in (("avec_N", "N", False), ("avec_Mz", "Mz", True), ("avec_Vy", "Vy", True)):
        if e.get(case):
            elu[cle] = abs(_nombre(e.get(cle))) if absolu else _nombre(e.get(cle))
    fl = bool(e.get("flambement"))
    b["flambement"] = {a: {"L": max(0.0, _nombre(e.get(f"Lcr_{a}"))) if fl else 0.0, "u": 1.0} for a in ("y", "z")}
    b["Lc"] = max(0.0, _nombre(e.get("L_LT"))) if e.get("deversement") else 0.0
    for cle in ("gamma_M0", "gamma_M1"):
        g = C.lire_nombre(e.get(cle))
        if g is not None and GAMMA_MIN <= g <= GAMMA_MAX and abs(g - PA.valeur(cle)) > 1e-9:
            b["parametres"][cle] = g
    nom = e.get("profil")
    if isinstance(nom, str) and CAT.par_nom().get(nom, {}).get("famille") in FAMILLES:
        b["profil"] = nom
    else:                                   # l'ancienne note suivait « l'optimal » : le plus léger qui tient aujourd'hui
        fams = [f for f in (e.get("familles") or []) if f in FAMILLES] or ["IPE"]
        b["profil"] = _plus_leger(b, fams) or _noms(fams)[-1]
    return b


def _completer(el: Dict) -> Dict:
    """La barre avec toutes ses clés : celles qui manquent prennent la valeur d'une barre neuve."""
    if not isinstance(el, dict):
        raise ValueError("barre illisible")
    b = barre_neuve(str(el.get("nom") or NOM_DEFAUT)[:NOM_MAX])
    for k in CLES[1:]:
        if k in el:
            b[k] = copy.deepcopy(el[k])
    for k in ("efforts", "diagramme", "flambement"):
        neuf = barre_neuve()[k]
        if not isinstance(b[k], dict):
            b[k] = neuf
        for sous, defaut in neuf.items():
            b[k][sous] = dict(defaut, **(b[k].get(sous) if isinstance(b[k].get(sous), dict) else {}))
    return b


def _relire(el: Dict) -> Dict:
    """La barre complétée, puis RELUE par le moteur dans son mode ; ValueError (le message du moteur) si elle ne tient
    pas debout. Les données de l'autre mode restent dans la note, telles quelles."""
    b = _completer(el)
    n = EN.normaliser(b)
    for k in COMMUNS + (CHARGEE if n["mode"] == "chargee" else IMPOSES):
        b[k] = n[k]
    if n["mode"] == "chargee":
        CO.service(b["cas"])                 # (au plus quatre cas variables : le moteur le dit ici plutôt qu'au calcul)
    return b


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran ou d'un fichier, relu : une note de la version 1 est convertie ; une barre illisible lève
    une erreur claire."""
    if not isinstance(etat, dict) or "elements" not in etat:
        return nouvel_etat()
    els = etat["elements"]
    if not isinstance(els, list) or not els:
        raise ValueError("les barres de la note sont illisibles (une liste de barres est attendue)")
    ancienne = etat.get("version", 1) != VERSION
    out = {"version": VERSION, "elements": []}
    for i, e in enumerate(els):
        try:
            out["elements"].append(_relire(_migrer(e) if ancienne else e))
        except (ValueError, TypeError, KeyError) as ex:
            raise ValueError(f"barre {i + 1} : {ex}") from None
    a = etat.get("actif", 0)
    a = a if isinstance(a, int) and not isinstance(a, bool) else 0
    out["actif"] = min(max(0, a), len(out["elements"]) - 1)
    return out


def _el(etat: Dict) -> Dict:
    return etat["elements"][etat["actif"]]


def _nom_libre(etat: Dict) -> str:
    pris = {e.get("nom") for e in etat["elements"]}
    k = len(etat["elements"]) + 1
    while f"B{k}" in pris:
        k += 1
    return f"B{k}"


def _indice(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les barres de la note : en ouvrir une, en ajouter, en copier une (``index`` ; sans : l'ouverte — la copie
    s'insère juste après), en supprimer une (jamais la dernière)."""
    els = etat["elements"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(els) - 1)
    elif action == "ajouter":
        els.append(barre_neuve(_nom_libre(etat)))
        etat["actif"] = len(els) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, _indice(index)), len(els) - 1)
        e = copy.deepcopy(els[i])
        e["nom"] = f"{e.get('nom') or 'Barre'} (copie)"[:NOM_MAX]
        els.insert(i + 1, e)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(els) > 1:
        i = min(max(0, _indice(index)), len(els) - 1)
        del els[i]
        if etat["actif"] > i or etat["actif"] >= len(els):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ le calcul, gardé
def _signature(el: Dict, sans: Sequence[str] = ("nom",)) -> str:
    return json.dumps({k: el[k] for k in CLES if k not in sans}, sort_keys=True, ensure_ascii=False)


def calculer(el: Dict) -> Dict:
    """Le résultat du moteur pour cette barre — gardé : une note de plusieurs barres ne recalcule que celle qu'on
    modifie ; le résultat, la note PDF et le sélecteur lisent le même. Il ne s'écrit pas."""
    cle = _signature(el)
    r = _RESULTATS.get(cle)
    if r is None:
        if len(_RESULTATS) >= MEMOIRE:
            _RESULTATS.clear()
        r = _RESULTATS[cle] = MO.verifier(el)
    return r


def _taux(r: Dict) -> Optional[float]:
    """Le taux le plus fort d'une barre, toutes méthodes confondues ; None quand rien n'a de taux."""
    t = [x["taux_max"] for x in r["methodes"].values() if x.get("taux_max") is not None]
    return max(t) if t else None


def _niveau(r: Dict) -> str:
    """Le niveau d'une barre pour la note : non conforme (r) ; une vérification non couverte : à vérifier (v) ; aucun
    effort : données incomplètes (n) ; sinon la couleur de son taux."""
    if r["statut"] == RE.NOK:
        return "r"
    if r["statut"] == RE.NON_COUVERT:
        return "v"
    t = _taux(r)
    return "n" if t is None or t <= 0 else C.niveau(t)


# ------------------------------------------------------------------ les profilés candidats
def _noms(familles: Sequence[str]) -> List[str]:
    """Les profilés de ces familles, du plus léger au plus lourd."""
    ps = [p for p in CAT.profils() if p["famille"] in familles]
    return [p["nom"] for p in sorted(ps, key=lambda p: (p.get("masse") or 0.0, p["nom"]))]


def _candidat(el: Dict, sig: str, nom: str) -> Tuple[Optional[float], str]:
    """(taux le plus fort, statut) de la barre avec ce profilé — gardé."""
    cle = (sig, nom)
    c = _CANDIDATS.get(cle)
    if c is None:
        if len(_CANDIDATS) >= MEMOIRE_CANDIDATS:
            _CANDIDATS.clear()
        try:
            r = MO.verifier(dict(el, profil=nom))
            c = (_taux(r), r["statut"])
        except ValueError:                   # un profilé que le moteur refuse d'entrée (une donnée du catalogue manque)
            c = (None, RE.NON_COUVERT)
        _CANDIDATS[cle] = c
    return c


def _plus_leger(el: Dict, familles: Sequence[str]) -> Optional[str]:
    """Le plus léger des profilés de ces familles que les méthodes de la barre acceptent ; None si aucun."""
    sig = _signature(el, ("nom", "profil"))
    return next((n for n in _noms(familles) if _candidat(el, sig, n)[1] == RE.OK), None)


# ------------------------------------------------------------------ la saisie
def _oui(valeur) -> bool:
    b = C.lire_bool(valeur)
    if b is None:
        raise ValueError(f"case attendue (oui ou non), reçu « {valeur} »")
    return b


def _lire(valeur, quoi: str) -> float:
    x = C.lire_nombre(valeur)
    if x is None:
        raise ValueError(f"{quoi} : « {valeur} » n'est pas un nombre")
    return x


def _rang(texte: str, liste: Sequence, quoi: str) -> int:
    i = int(texte) if texte.isdigit() else -1
    if not 0 <= i < len(liste):
        raise ValueError(f"{quoi} introuvable")
    return i


def _ajuster(el: Dict, L_avant: float) -> bool:
    """Après un changement de portée ou de modèle : une charge qui allait jusqu'au bout de la barre la suit ; ce qui
    dépasse est ramené dans la barre (une charge ou un maintien entièrement dehors est retiré). Vrai si quelque chose a
    été ramené ou retiré."""
    L = float(sum(el["travees"]))
    touche = False
    for c in el["cas"]:
        gardees = []
        for ch in c["charges"]:
            if ch["type"] in ("q", "tri"):
                a, b = ch.get("a", 0.0), ch.get("b", L_avant)
                if abs(b - L_avant) < 1e-9:
                    b = L
                if a >= L - 1e-9:
                    touche = True
                    continue
                if b > L:
                    b, touche = L, True
                ch["a"], ch["b"] = a, b
            elif ch["a"] > L:
                ch["a"], touche = L, True
            gardees.append(ch)
        c["charges"] = gardees
    dedans = [mt for mt in el["maintiens"] if mt["x"] <= L]
    touche = touche or len(dedans) != len(el["maintiens"])
    el["maintiens"] = dedans
    return touche


def _charge_neuve(type_: str, L: float, ancienne: Optional[Dict] = None) -> Dict:
    """Une charge du type demandé ; d'une ancienne (un changement de type), elle garde la valeur et, au mieux, la place."""
    v = ancienne["v"] if ancienne else {"q": 5.0, "tri": 5.0, "P": 10.0, "C": 10.0}[type_]
    if type_ in ("q", "tri"):
        a, b = (ancienne["a"], ancienne["b"]) if ancienne and "b" in ancienne else (0.0, L)
        ch = {"type": type_, "v": v, "a": a, "b": b}
        if type_ == "tri":
            ch["sens"] = (ancienne or {}).get("sens", "croissante")
        return ch
    a = ancienne["a"] if ancienne and "b" not in ancienne else (round(0.5 * (ancienne["a"] + ancienne["b"]), 3) if ancienne else round(L / 2.0, 3))
    return {"type": type_, "v": v, "a": a}


def _cas_neuf(el: Dict) -> Dict:
    """Un cas de charge de plus : variable (exploitation A) tant que le moteur en admet, permanent sinon."""
    L = float(sum(el["travees"]))
    variables = sum(1 for c in el["cas"] if c["nature"] != CO.PERMANENT)
    lettre, nature = ("Q", "exploitation_A") if variables < CO.VARIABLES_MAX else ("G", CO.PERMANENT)
    pris = {c["nom"] for c in el["cas"]}
    nom, k = lettre, 2
    while nom in pris:
        nom, k = f"{lettre}{k}", k + 1
    return {"nom": nom, "nature": nature, "N": 0.0, "charges": [_charge_neuve("q", L)]}


def _maintien_neuf(el: Dict) -> Dict:
    """Un maintien de plus, au milieu du plus long intervalle que rien ne tient encore."""
    L = float(sum(el["travees"]))
    tenus, x = {0.0, L}, 0.0
    for t in el["travees"]:
        x += t
        tenus.add(round(x, 6))
    tenus |= {round(mt["x"], 6) for mt in el["maintiens"]}
    pts = sorted(tenus)
    a, b = max(zip(pts, pts[1:]), key=lambda p: p[1] - p[0])
    return {"x": round(0.5 * (a + b), 3), "genre": "complet"}


def _appliquer(el: Dict, cles: List[str], valeur, r: Dict) -> Optional[str]:
    """Écrit la saisie dans ``el`` (une copie) ; rend un mot à dire (« charges ramenées dans la barre ») ou None.
    KeyError : le champ n'existe pas ; ValueError : la valeur est refusée."""
    cle = "/".join(cles)
    k = cles[0]
    L = float(sum(el["travees"]))
    if cle == "nom":
        nom = " ".join(str(valeur if valeur is not None else "").split())
        if len(nom) > NOM_MAX:
            raise ValueError(f"nom trop long : {len(nom)} caractères (au plus {NOM_MAX})")
        el["nom"] = nom or NOM_DEFAUT
    elif cle == "mode":
        if valeur not in EN.MODES:
            raise ValueError(f"mode inconnu : « {valeur} »")
        if valeur == "efforts":                 # les longueurs que la barre chargée prenait d'elle-même deviennent des données
            for a in ("y", "z"):
                if el["flambement"][a]["L"] is None:
                    el["flambement"][a]["L"] = round(r["longueurs"][a]["L"], 3) if r.get("longueurs") else L
        el["mode"] = valeur
    elif cle in ("nuance", "niveau"):
        el[cle] = str(valeur)
    elif cle == "poids_propre":
        el[cle] = _oui(valeur)
    elif cle == "methodes":
        if valeur not in ("deux", "historique", "ec3"):
            raise ValueError(f"méthode inconnue : « {valeur} »")
        el["methodes"] = list(EN.METHODES) if valeur == "deux" else [valeur]
    elif cle == "modele":
        if valeur not in PO.MODELES:
            raise ValueError(f"modèle inconnu : « {valeur} »")
        if valeur == "continue" and len(el["travees"]) < 2:
            el["travees"] = [el["travees"][0], el["travees"][0]]
        elif valeur != "continue":
            el["travees"] = el["travees"][:1]
        el["modele"] = valeur
        return "Charges et maintiens ramenés dans la barre." if _ajuster(el, L) else None
    elif k == "travees":
        if cle == "travees/+":
            if el["modele"] != "continue" or len(el["travees"]) >= PO.TRAVEES_MAX:
                raise ValueError(f"une poutre continue a de 2 à {PO.TRAVEES_MAX} travées")
            el["travees"].append(el["travees"][-1])
        elif len(cles) == 3 and cles[2] == "-":
            i = _rang(cles[1], el["travees"], "travée")
            if el["modele"] != "continue" or len(el["travees"]) <= 2:
                raise ValueError("une poutre continue garde au moins deux travées")
            del el["travees"][i]
        elif len(cles) == 2:
            i = _rang(cles[1], el["travees"], "travée")
            el["travees"][i] = _lire(valeur, "portée")
            if not EN.L_MIN <= el["travees"][i] <= EN.L_MAX:
                raise ValueError(f"chaque portée doit être entre {M.n(EN.L_MIN)} et {M.n(EN.L_MAX)} m")
        else:
            raise KeyError(cle)
        return "Charges et maintiens ramenés dans la barre." if _ajuster(el, L) else None
    elif k == "cas":
        cas = el["cas"]
        if cle == "cas/+":
            if len(cas) >= EN.CAS_MAX:
                raise ValueError(f"au plus {EN.CAS_MAX} cas de charge")
            cas.append(_cas_neuf(el))
        elif len(cles) == 3 and cles[2] == "-":
            i = _rang(cles[1], cas, "cas de charge")
            if len(cas) == 1:
                raise ValueError("il faut au moins un cas de charge")
            del cas[i]
        elif len(cles) == 3 and cles[2] in ("nom", "nature", "N"):
            c = cas[_rang(cles[1], cas, "cas de charge")]
            if cles[2] == "nom":
                nom = " ".join(str(valeur or "").split())
                if len(nom) > NOM_CAS_MAX:
                    raise ValueError(f"nom de cas trop long : {len(nom)} caractères (au plus {NOM_CAS_MAX})")
                c["nom"] = nom or f"C{int(cles[1]) + 1}"
            elif cles[2] == "nature":
                c["nature"] = str(valeur)
            else:
                c["N"] = _lire(valeur, "effort normal")
        elif len(cles) >= 4 and cles[2] == "charges":
            c = cas[_rang(cles[1], cas, "cas de charge")]
            chs = c["charges"]
            if cles[3] == "+" and len(cles) == 4:
                if valeur not in M.UNITE_CHARGE:
                    raise ValueError(f"type de charge inconnu : « {valeur} »")
                if len(chs) >= PO.CHARGES_MAX:
                    raise ValueError(f"au plus {PO.CHARGES_MAX} charges par cas")
                chs.append(_charge_neuve(valeur, L))
            elif len(cles) == 5:
                j = _rang(cles[3], chs, "charge")
                champ = cles[4]
                if champ == "-":
                    del chs[j]
                elif champ == "type":
                    if valeur not in M.UNITE_CHARGE:
                        raise ValueError(f"type de charge inconnu : « {valeur} »")
                    chs[j] = _charge_neuve(valeur, L, chs[j])
                elif champ == "sens" and chs[j]["type"] == "tri":
                    chs[j]["sens"] = str(valeur)
                elif champ in ("v", "a") or (champ == "b" and "b" in chs[j]):
                    chs[j][champ] = _lire(valeur, {"v": "valeur de la charge", "a": "abscisse", "b": "abscisse"}[champ])
                else:
                    raise KeyError(cle)
            else:
                raise KeyError(cle)
        else:
            raise KeyError(cle)
    elif k == "maintiens":
        mts = el["maintiens"]
        if cle == "maintiens/+":
            if len(mts) >= EN.MAINTIENS_MAX:
                raise ValueError(f"au plus {EN.MAINTIENS_MAX} maintiens")
            mts.append(_maintien_neuf(el))
        elif len(cles) == 3:
            j = _rang(cles[1], mts, "maintien")
            if cles[2] == "-":
                del mts[j]
            elif cles[2] == "x":
                mts[j]["x"] = _lire(valeur, "abscisse du maintien")
            elif cles[2] == "genre":
                mts[j]["genre"] = str(valeur)
            else:
                raise KeyError(cle)
        else:
            raise KeyError(cle)
    elif k == "flambement" and len(cles) == 3 and cles[1] in ("y", "z") and cles[2] in ("L", "u"):
        if cles[2] == "L" and el["mode"] == "chargee":
            el["flambement"][cles[1]]["L"] = C.lire_auto(valeur)
        else:
            el["flambement"][cles[1]][cles[2]] = _lire(valeur, "longueur de flambement" if cles[2] == "L" else "coefficient u")
    elif cle == "Lc":
        el["Lc"] = C.lire_auto(valeur)
    elif k == "efforts" and len(cles) == 3 and cles[1] in ("service", "elu") and cles[2] in EN.EFFORTS:
        el["efforts"][cles[1]][cles[2]] = _lire(valeur, cles[2])
    elif k == "diagramme" and len(cles) == 3 and cles[1] in ("y", "z") and cles[2] in ("forme", "psi"):
        el["diagramme"][cles[1]][cles[2]] = str(valeur) if cles[2] == "forme" else _lire(valeur, "ψ")
    elif k == "historique" and len(cles) == 2 and cles[1] in ("s", "C", "deversement", "tolerance"):
        el["historique"][cles[1]] = _lire(valeur, "s") if cles[1] == "s" else (str(valeur) if cles[1] == "C" else _oui(valeur))
    elif k == "parametres" and len(cles) == 2 and cles[1] in ("gamma_M0", "gamma_M1"):
        g = C.lire_auto(valeur)
        if g is None:
            el["parametres"].pop(cles[1], None)
        elif not GAMMA_MIN <= g <= GAMMA_MAX:
            raise ValueError(f"{PA.PARAMETRES[cles[1]]['nom']} doit être entre {M.n(GAMMA_MIN)} et {M.n(GAMMA_MAX)}")
        else:
            el["parametres"][cles[1]] = g
    else:
        raise KeyError(cle)
    return None


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la barre ouverte. ``chemin`` : « travees/0 », « cas/1/charges/0/v », « cas/+ »,
    « maintiens/2/- », « flambement/z/L », « efforts/elu/My », « methodes »… Le moteur relit la barre : une valeur
    hors de son domaine est REFUSÉE avec son message, et la barre reste intacte."""
    el = _el(etat)
    essai = copy.deepcopy(el)
    cles = [str(c) for c in chemin]
    if not cles:
        return "Champ inconnu : «  »"
    try:
        mot = _appliquer(essai, cles, valeur, calculer(el))
        etat["elements"][etat["actif"]] = _relire(essai)
    except KeyError:
        return f"Champ inconnu : « {'/'.join(cles)} »"
    except (ValueError, TypeError, IndexError) as e:
        texte = str(e)
        return f"Saisie refusée : {texte[:1].lower() + texte[1:]}"
    return mot


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le clic dans le sélecteur : le profilé devient celui de la barre."""
    if role != ROLE:
        raise KeyError(f"sélecteur inconnu : {role!r}")
    if CAT.par_nom().get(nom, {}).get("famille") not in FAMILLES:
        raise ValueError(f"profilé inconnu : {nom!r}")
    etat["elements"][etat["actif"]] = _relire(dict(_el(etat), profil=nom))


# ------------------------------------------------------------------ la colonne de gauche
def arbre(etat: Dict, r: Dict, resultats: List[Dict]) -> str:
    """La colonne de gauche COMPACTE : la ligne de la barre ouverte (son nom, la liste des barres de la note, copier,
    supprimer — le nom de chacune à la couleur de son état), puis ses quatre rubriques."""
    noms = [(e.get("nom") or f"B{i + 1}", "· " + R.dims(e), False) for i, e in enumerate(etat["elements"])]
    niveaux = [_niveau(x) if "methodes" in x else "n" for x in resultats]
    return E.arbre(GENRE, noms, etat["actif"], S.formulaire(_el(etat), r), "choixprofil", compact=True, niveaux=niveaux)


# ------------------------------------------------------------------ les vues
def _bande(r: Dict) -> Optional[List[tuple]]:
    """Les plus grands efforts de la barre chargée, pour la bande de l'en-tête des vues : (symbole, indice, valeur, unité)."""
    env = DB.enveloppes(r)
    if not env:
        return None
    k, e = env
    ind = "Ed" if k == "ec3" else "ser"
    return [("M", ind, max(max(e["M_max"]), -min(e["M_min"])), "kN·m"), ("V", ind, max(max(e["V_max"]), -min(e["V_min"])), "kN")]


def _entete(el: Dict, r: Dict, titre: str, legende: str) -> str:
    """L'en-tête technique d'une vue : le profilé et l'acier ; la barre en une ligne ; ses maintiens ; les plus grands efforts."""
    if el["mode"] == "chargee":
        ch = sum(len(c["charges"]) for c in el["cas"])
        l2 = (f"{M.NOM_MODELE[el['modele']]} · {' + '.join(M.m(x) for x in el['travees'])} m · {len(el['cas'])} cas de charge, "
              f"{ch} charge{'s' if ch > 1 else ''}")
        mts = " ; ".join(f"{M.NOM_GENRE[mt['genre']].lower()} à {M.m(mt['x'])} m" for mt in el["maintiens"])
        l3 = f"Maintiens : {mts}" if mts else "Aucun maintien entre les appuis"
    else:
        l2 = "Efforts imposés"
        Ls = r["longueurs"]
        l3 = f"Flambement : {M.m(Ls['y']['L'] * Ls['y']['u'])} m (axe fort) · {M.m(Ls['z']['L'] * Ls['z']['u'])} m (axe faible)"
    return E.entete(titre.upper(), f"{C.esc(el['profil'])} · {C.esc(el['nuance'])}", [l2, l3], _bande(r), legende)


def fiche_profil(nom: str) -> Dict:
    """Ce qu'on écrit d'un profilé du catalogue : {"famille", "dims" (mm), "dimensions" et "proprietes" : [(symbole,
    valeur écrite, unité)], "sous" (sa famille et sa norme)} — la vue « Section » de l'écran et le panneau de la note PDF."""
    rec = CAT.par_nom()[nom]
    fam = rec["famille"]
    dims = {k: rec[k] for k in FA.FAMILLES[fam]["dims"]}
    sym = {"tw": "t_w", "tf": "t_f", "r1": "r_1", "r2": "r_2"}
    dimensions = [(sym.get(k, k), M.n(v), "mm") for k, v in dims.items()]
    props = [("masse", rec.get("masse"), "kg/m", 1), ("A", rec.get("A"), "cm²", 2), ("I_y", rec.get("Iy"), "cm⁴", 1), ("I_z", rec.get("Iz"), "cm⁴", 1),
             ("W_el,y", rec.get("Wel_y"), "cm³", 1), ("W_pl,y", rec.get("Wpl_y"), "cm³", 1), ("W_pl,z", rec.get("Wpl_z"), "cm³", 1),
             ("A_v,z", rec.get("Avz"), "cm²", 2), ("I_t", rec.get("It"), "cm⁴", 2), ("I_w", rec.get("Iw"), "cm⁶", 0)]
    # une propriété absente ou nulle ne s'écrit pas (le gauchissement d'un tube)
    return {"famille": fam, "dims": dims, "dimensions": dimensions, "proprietes": [(s, M.n(v, nd), u) for s, v, u, nd in props if v],
            "sous": f"{FA.FAMILLES[fam]['libelle']} · {FA.FAMILLES[fam]['norme']}"}


def vue_section(el: Dict, ident: str = "pf_section", fiche: bool = True) -> str:
    """La section du profilé de la barre, à l'échelle ; ``fiche`` : ses dimensions et ses propriétés dessous."""
    f = fiche_profil(el["profil"])
    return DP.section(ident, el["profil"], f["sous"], f["famille"], f["dims"], f["dimensions"], f["proprietes"], "", fiche)


def _vues(el: Dict, r: Dict, ident: str) -> List[Dict]:
    """La barre et ses enveloppes (barre chargée), la section à l'échelle, les taux de toutes les vérifications."""
    vues = []
    if el["mode"] == "chargee":
        leg = "les cas de charge, la barre et ses maintiens, les enveloppes du moment et de l'effort tranchant"
        vues.append({"cle": "barre", "titre": "Barre et efforts", "court": "Barre", "svg": DB.barre(f"{ident}_barre", el, r), "legende": leg,
                     "entete": _entete(el, r, "Barre et efforts", leg)})
    leg = "le profilé à l'échelle, ses dimensions et ses propriétés"
    vues.append({"cle": "section", "titre": f"Section {el['profil']}", "court": "Section", "svg": vue_section(el), "legende": leg,
                 "entete": _entete(el, r, "Section", leg)})
    if any(x["verifications"] for x in r["methodes"].values()):
        leg = "le taux de chaque vérification ; la limite de 100 % en tireté"
        ms = [(f"{M.METHODES[k]}", x) for k, x in r["methodes"].items()]
        vues.append({"cle": "taux", "titre": "Taux des vérifications", "court": "Taux", "svg": DB.taux(f"{ident}_taux", ms), "legende": leg,
                     "entete": _entete(el, r, "Taux des vérifications", leg)})
    return vues


# ------------------------------------------------------------------ la note
def _sans_echec(el: Dict) -> Dict:
    try:
        return calculer(el)
    except (ValueError, KeyError, TypeError) as ex:              # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(ex)}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE (l'accueil, la liste des notes) : sa barre la plus chargée, son état le moins bon."""
    bons = [(i, x) for i, x in enumerate(resultats) if "methodes" in x]
    if not bons:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    nivs = [_niveau(x) for _, x in bons]
    niv = next((k for k in ("r", "v", "n", "f") if k in nivs), "p")
    avec = [(t, i, x) for i, x in bons for t in [_taux(x)] if t is not None]
    if not avec:
        return {"taux": None, "niveau": niv, "statut": C.STATUTS[niv][1], "gouvernant": ""}
    t, i, x = max(avec, key=lambda a: a[0])
    k, m = max(((k, m) for k, m in x["methodes"].items() if m.get("taux_max") is not None), key=lambda a: a[1]["taux_max"])
    g = R.gouvernante(m)
    quoi = f" · {g['titre'][:1].lower() + g['titre'][1:]} ({M.METHODES_COURT[k].lower()})" if g else ""
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['elements'][i].get('nom') or 'Barre'} · {x['profil']}{quoi}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les barres de la note sont la LIGNE de la barre, en tête de la colonne de gauche (``arbre``)."""
    return ""


def resultat(etat: Dict) -> Dict:
    """Le résultat de la barre ouverte : la tête, un verdict par méthode, les blocs ; les fenêtres de calcul, les vues,
    la colonne de gauche, le résumé de la note."""
    el = _el(etat)
    r = calculer(el)
    resultats = [r if i == etat["actif"] else _sans_echec(e) for i, e in enumerate(etat["elements"])]
    nom = C.esc(el.get("nom") or "Barre")
    corps = (f'<div class="pbody">{E.tete_resultats(nom, R.dims(el))}{R.verdicts(r)}'
             f'<div class="et-blocs">{R.blocs(el, r)}</div></div>{R.pied(r)}')
    resume = resume_note(etat, resultats)
    statut = C.statut(resume["niveau"]) if len(resultats) > 1 else ""
    return {"resultat": corps, "volets": R.volets(el, r, nom), "vues": _vues(el, r, f"pf{etat['actif']}"), "phrase": "", "statut_html": statut,
            "resume": resume, "donnees": arbre(etat, r, resultats), "elements": elements(etat, resultats)}


def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« note_pdf » : la note de calcul PDF de TOUTE la note — la garde, puis la planche de chaque barre, avec les
    méthodes que la barre a de cochées (api_web/note_profil.py, chargé à la demande : il lui faut le paquet de la note) ;
    ``infos`` : le nom, le chantier, l'indice et la date de la note."""
    if quoi != "note_pdf":
        return None
    from . import note_profil as NP
    return NP.fichier(etat, infos)


# ------------------------------------------------------------------ le sélecteur de profilés
def _proprietes(nom: str, taux: Optional[float]) -> str:
    p = CAT.par_nom()[nom]
    dims = f"h × b = {M.n(p['h'])} × {M.n(p['b'])} mm" if p.get("h") and p.get("b") else (f"d = {M.n(p['d'])} mm" if p.get("d") else "")
    eta = f" · η = {C.fr(taux, 3)}" if taux is not None else ""
    return (f"<b>{C.esc(nom)}</b> · {dims} · A = {M.n(p.get('A'), 2)} cm² · I<sub>y</sub> = {M.n(p.get('Iy'), 1)} cm⁴ · "
            f"W<sub>pl,y</sub> = {M.n(p.get('Wpl_y'), 1)} cm³{eta}")


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les profilés des familles demandées (par défaut celle du profilé de la barre), TRIÉS PAR MASSE, chacun vérifié
    par le moteur avec les méthodes de la barre : son taux le plus fort ; le retenu (✓) et le plus léger qui tient (★)
    marqués. Un profilé qu'une méthode ne couvre pas le dit, sans couleur de conformité."""
    if role != ROLE:
        raise KeyError(f"sélecteur inconnu : {role!r}")
    el = _el(etat)
    demandees = familles if isinstance(familles, (list, tuple)) else []      # une requête abîmée : la famille du profilé
    fams = [f for f in FAMILLES if f in demandees] or [CAT.par_nom()[el["profil"]]["famille"]]
    sig = _signature(el, ("nom", "profil"))
    rows, vu_leger = [], False
    for nom in _noms(fams):
        t, statut = _candidat(el, sig, nom)
        tient = statut == RE.OK
        row = {"nom": nom, "col": C.fr(CAT.par_nom()[nom]["masse"], 1), "taux": None, "niveau": "c", "cur": nom == el["profil"],
               "opt": tient and not vu_leger, "prop": _proprietes(nom, t)}
        vu_leger = vu_leger or tient
        if statut == RE.NON_COUVERT:
            row["etat"] = "non couvert" if t is None else f"{C.pc(C.pct(min(t, 9.99)))} % ?"
        elif t is not None:
            row.update({"taux": C.pct(min(t, 9.99)), "niveau": "r" if statut == RE.NOK else C.niveau(t)})
        else:
            row["etat"] = "refusé" if statut == RE.NOK else "—"
        rows.append(row)
    i_o = next((k for k, x in enumerate(rows) if x["opt"]), 0)
    i_c = next((k for k, x in enumerate(rows) if x["cur"]), i_o)
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{C.esc(f)}">{C.esc(f)}</button>' for f in FAMILLES)
    methodes = " et ".join(M.METHODES_COURT[k].lower() for k in el["methodes"])
    sous = f"triés par masse · vérifiés par : {methodes} · un clic retient le profilé"
    return C.selecteur(ROLE, f"{C.esc(el.get('nom') or 'Barre')} · profilé", sous, filtres, rows, max(0, min(i_o, i_c) - 4), "kg/m")

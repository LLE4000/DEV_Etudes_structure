# -*- coding: utf-8 -*-
"""Le module « Choix de profilé » du site : la vérification d'une BARRE en acier — poutre, poteau, poteau-poutre — par
la méthode ELS du bureau (NBN B 51-001, charges de service), par l'Eurocode 3 (NBN EN 1993-1-1, ELU), ou par les
deux, côte à côte (docs/acier/BARRES_ACIER.md). Le moteur : ``acier/barre``.

Une note garde une ou plusieurs BARRES. Chacune se décrit de deux façons :
- « barre chargée » : un modèle (deux appuis, console, encastrée, continue), des portées, des cas de charge, des
  maintiens — le module fait la statique, les combinaisons, et cherche la situation la plus défavorable ;
- « efforts imposés » (les classeurs du bureau) : les efforts de service et de calcul sont donnés.

L'écran est l'établi COMPACT (api_web/etabli_compact.py) : à gauche la ligne de la barre et ses quatre rubriques
(api_web/profil_saisie.py) ; au centre la barre et ses enveloppes, la section, les taux (api_web/dessin_barre.py,
dessin_profil.py) ; à droite un verdict par méthode, le profilé, un bloc par méthode, leur comparaison, et les fenêtres
de calcul (api_web/profil_resultat.py).

Ce fichier tient l'ÉTAT de la note (``{"version": 2, "zero_saisi": True, "elements": [barre…], "actif": i}``), relit les
anciennes notes (version 1 : des efforts de calcul et l'Eurocode seul — elles deviennent des barres en « efforts imposés » ;
version 2 sans « zero_saisi » : le site d'avant le lot B, ses s_s et c à 0 y sont « non saisis », ``_de_v2``), écrit les
saisies et sert le sélecteur de profilés. Il ne calcule rien : toute saisie est relue par ``acier.barre.entree`` — hors
domaine, elle est REFUSÉE avec le message du moteur et la barre reste intacte.
"""
import copy
import functools
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from acier.barre import combinaisons as CO
from acier.barre import entree as EN
from acier.barre import moteur as MO
from acier.barre import nuances as NU
from acier.barre import parametres as PA
from acier.barre import resultat as RE
from acier.barre import section as SE
from acier.barre import statique as ST
from acier.profiles import catalogue as CAT
from acier.profiles import familles as FA
from rdm import poutres as PO

from . import commun as C
from . import dessin_barre as DB
from . import dessin_profil as DP
from . import etabli_compact as E
from . import profil_mots as M
from . import profil_prs as PP
from . import profil_resultat as R
from . import profil_saisie as S

MODULE = "profil"
PAGE = "choix-profile"
VERSION = 2
# (lot B) une clé de la NOTE, sans changer de version : ses 0 de s_s et de c sont des SAISIES (vide : null, non saisi). Une note
# sans elle vient du site d'avant le lot B, qui écrivait 0 pour « non saisi » : ses 0 y sont relus « non saisis » (_de_v2, le
# côté sûr). Le site d'avant ignore la clé : il relit une note neuve telle quelle (0 = 0) ou la REFUSE (une donnée vide, null,
# « nombre attendu ») — jamais une autre lecture (une version 3 y était prise pour une version 1 : « efforts imposés »). Mais il
# suffit qu'il l'OUVRE : il la réécrit aussitôt sans la clé (son etabli.js enregistre au premier calcul) — ses 0 saisis redeviennent
# alors « non saisis » pour le site neuf (le côté sûr, à ressaisir).
ZERO_SAISI = "zero_saisi"
# (lot C, C0d) ce que la NOTE imprime — une clé de la note, pour toutes ses barres : les deux méthodes (le défaut, brief § 9),
# la méthode ELS seule, ou l'Eurocode 3 seul. Le moteur calcule TOUJOURS les deux ; l'écran les montre toujours. Une note sans
# elle (d'avant le lot C) : une seule méthode cochée dans toutes ses barres → celle-là, sinon les deux ; une note de la version 1
# (l'Eurocode seul) → « ec3 ». La clé « methodes » d'une barre reste lue (pour cette déduction), elle ne décide plus du calcul.
IMPRESSION = "impression"
IMPRESSIONS = ("deux", "historique", "ec3")

ROLE = "profil"                 # le seul sélecteur du module : le profilé de la barre ouverte
GENRE = E.Genre("Barres", "la barre", "cette barre", "une barre")
NOM_DEFAUT = "B1"
NOM_MAX = 60                    # caractères du nom d'une barre : la borne des éléments du site (dalle, cornière, fondation de grue)
NOM_CAS_MAX = 12                # … et du nom d'un cas de charge : il s'écrit à gauche de sa bande, sur le dessin de la barre
GAMMA_MIN, GAMMA_MAX = 0.9, 1.5
# les familles du catalogue que le module vérifie par ses deux méthodes : I et H laminés, tubes
FAMILLES_DEUX = tuple(f for f, (forme, _) in SE.FAMILLES.items() if forme in ("I",) + SE.TUBES)
# (lot C, C1b) les U et les cornières : la compression simple de la méthode ELS, en efforts imposés seulement (brief § 3 ;
# entree.MSG_EFFORTS_SEULS les refuse en barre chargée) — le sélecteur ne les propose qu'en efforts imposés
FAMILLES_EFFORTS = ("UPN", "UPE", "L")
FAMILLES = FAMILLES_DEUX + FAMILLES_EFFORTS
# ce qui appartient à chaque mode (l'autre est gardé tel quel dans la note : changer de mode ne perd rien)
# (lot C, C0e : le contrat des entrées — « prs » y est, sans quoi _completer la jetait et _signature ne la voyait pas : deux PRS
# de plats différents partageaient un résultat gardé ; chaque clé a son défaut, acier/barre/entree.nouvelle_barre)
COMMUNS = ("profil", "nuance", "mode", "poids_propre", "niveau", "Lc", "methodes", "historique", "parametres", "flambement", "prs",
           "norme_produit", "ouvrage")
CHARGEE = ("modele", "travees", "cas", "maintiens", "appuis", "combinaisons", "position", "angle", "points", "debords")
IMPOSES = ("efforts", "diagramme", "poids_propre_efforts")
CLES = ("nom",) + COMMUNS + CHARGEE + IMPOSES
# (lot C, reprise de C0) les longueurs de flambement que le passage en efforts imposés a COPIÉES de l'automatique, {axe: L (m)} :
# au retour en barre chargée, celles que l'ingénieur n'a pas retouchées redeviennent automatiques. Sans cela, une poutre de
# 6 m passée en efforts imposés puis rendue console de 3 m gardait L_z = 3 m comme une saisie (2,1·L = 6,3 m), sans un mot.
# Un état de l'ÉCRAN, hors du calcul (hors de CLES et de la signature) ; une barre qui ne l'a pas : rien n'a été copié.
COPIEES = "longueurs_copiees"
MEMOIRE, MEMOIRE_CANDIDATS = 48, 6000
# (lot C, C6) la BIBLIOTHÈQUE DES PRS est le carnet du module (api_web/routeur.py : il voyage avec chaque requête, l'écran le
# garde dans ce navigateur) — jamais dans la note : la barre garde une copie de ses plats (clé « prs »). api_web/profil_prs.py
CARNET = True
_RESULTATS: Dict[str, Dict] = {}                    # la barre (sans son nom) → son résultat
_CANDIDATS: Dict[Tuple[str, str], Tuple] = {}       # (la barre sans nom ni profilé, un profilé) → (taux, statut)


@functools.lru_cache(maxsize=1)
def _barre_type() -> Dict:
    """La barre neuve, RELUE par le moteur : chaque clé du contrat des entrées porte son défaut (lot C, C0e) — une note
    neuve est déjà ce que la relecture rend (une saisie refusée rend la note telle qu'elle était, à l'octet)."""
    b = EN.nouvelle_barre(NOM_DEFAUT)
    b["maintiens"] = [{"x": 3.0, "genre": "complet"}]
    return EN.normaliser(b)


def barre_neuve(nom: str = NOM_DEFAUT) -> Dict:
    """La barre d'une note neuve : celle du moteur (IPE 300 sur deux appuis, 6 m, G et Q de 5 kN/m, poids propre), tenue
    à mi-portée — sans maintien, la méthode historique la refuse d'emblée."""
    b = copy.deepcopy(_barre_type())
    b["nom"] = nom
    return b


def etat_de(elements: List[Dict], actif: int = 0, impression: str = IMPRESSIONS[0]) -> Dict:
    """Une note au format du site (la version, sa clé ``ZERO_SAISI``, ce qu'elle imprime) : la source unique de ce format."""
    return {"version": VERSION, ZERO_SAISI: True, "elements": elements, "actif": actif, IMPRESSION: impression}


def methodes_imprimees(impression: str) -> Tuple[str, ...]:
    """Les méthodes qu'une note imprime (``IMPRESSION``), dans l'ordre du moteur."""
    return tuple(EN.METHODES) if impression == "deux" else (impression,)


def _impression_deduite(etat: Dict) -> str:
    """Ce qu'imprime une note qui n'a pas la clé ``IMPRESSION`` (d'avant le lot C) : une note de la version 1 → « ec3 » ;
    une seule et même méthode cochée dans toutes ses barres → celle-là ; sinon les deux. Rien n'est réécrit dans les barres."""
    if etat.get("version", 1) == 1:
        return "ec3"
    choix = set()
    for e in etat.get("elements") or []:
        m = e.get("methodes") if isinstance(e, dict) else None
        m = tuple(x for x in EN.METHODES if isinstance(m, (list, tuple)) and x in m) if m is not None else tuple(EN.METHODES)
        choix.add(m)
    if len(choix) == 1:
        (m,) = choix
        if len(m) == 1:
            return m[0]
    return "deux"


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
    {"action": "calculer", "etat": etat_de([_POTEAU])},
    {"action": "calculer", "etat": etat_de([_CONTINUE])},
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
    {"action": "modifier", "chemin": ["travees", "+"], "valeur": "1", "etat": etat_de([_CONTINUE])},
    {"action": "modifier", "chemin": ["travees", "0", "-"], "valeur": "1", "etat": etat_de([dict(_CONTINUE, travees=[5.0, 4.0, 3.0])])},
    {"action": "modifier", "chemin": ["flambement", "z", "L"], "valeur": ""},
    {"action": "modifier", "chemin": ["flambement", "y", "L"], "valeur": "abc"},
    {"action": "modifier", "chemin": ["travees", "0"], "valeur": "-1"},
    {"action": "modifier", "chemin": ["parametres", "gamma_M1"], "valeur": "1,1"},
    {"action": "modifier", "chemin": ["efforts", "elu", "N"], "valeur": "900", "etat": etat_de([_POTEAU])},
    {"action": "modifier", "chemin": ["nom"], "valeur": "Panne"},
]


# ------------------------------------------------------------------ l'état de la note : des barres
def nouvel_etat() -> Dict:
    return etat_de([barre_neuve()])


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
    b.update({"mode": "efforts", "methodes": ["ec3"], "nuance": e.get("acier") if e.get("acier") in NU.NUANCES else "S235"})
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
    # (une note de la version 1 ne connaissait que les I, les H et les tubes : FAMILLES_DEUX — elle se relit comme avant)
    if isinstance(nom, str) and CAT.par_nom().get(nom, {}).get("famille") in FAMILLES_DEUX:
        b["profil"] = nom
    else:                                   # l'ancienne note suivait « l'optimal » : le plus léger qui tient aujourd'hui
        fams = [f for f in (e.get("familles") or []) if f in FAMILLES_DEUX] or ["IPE"]
        b["profil"] = _plus_leger(b, fams) or _noms(fams)[-1]
    return b


def _zero(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0


def _de_v2(e: Dict) -> Dict:
    """Une barre d'une note du site d'avant le lot B (version 2 sans ``ZERO_SAISI``) : il y écrivait 0,0 pour s_s et c non
    saisis (appuis et charges ponctuelles) — un 0 n'y distingue pas une saisie d'un défaut. Ramené à « non saisi » (None) :
    l'âme le prend à sa borne, le dit, ou demande la donnée (brief § 8.2, § 8.3 point 5) — le côté prudent. Une note neuve
    (``ZERO_SAISI``) garde ses 0 saisis."""
    if not isinstance(e, dict):
        return e
    e = copy.deepcopy(e)
    ap = e.get("appuis")
    if isinstance(ap, dict):
        for k in ("ss", "c"):
            if _zero(ap.get(k)):
                ap[k] = None
    for c in e.get("cas") if isinstance(e.get("cas"), list) else []:
        for ch in (c.get("charges") if isinstance(c, dict) and isinstance(c.get("charges"), list) else []):
            if isinstance(ch, dict) and ch.get("type") == "P" and _zero(ch.get("ss")):
                ch["ss"] = None
    return e


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
    cop = el.get(COPIEES)
    cop = {a: float(v) for a, v in cop.items() if a in ("y", "z") and isinstance(v, (int, float)) and not isinstance(v, bool)} \
        if isinstance(cop, dict) else {}
    if cop:
        b[COPIEES] = cop
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
        b.pop(COPIEES, None)                 # (le souvenir des longueurs copiées ne vaut qu'en efforts imposés)
    return b


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran ou d'un fichier, relu : une note de la version 1 est convertie ; une de la version 2 sans
    ``ZERO_SAISI`` (le site d'avant le lot B) relue, ses s_s et c à 0 « non saisis » ; une barre illisible lève une erreur claire.
    Ce que la note imprime (``IMPRESSION``) est gardé, ou déduit d'une note qui ne l'a pas (``_impression_deduite``)."""
    if not isinstance(etat, dict) or "elements" not in etat:
        return nouvel_etat()
    els = etat["elements"]
    if not isinstance(els, list) or not els:
        raise ValueError("les barres de la note sont illisibles (une liste de barres est attendue)")
    version = etat.get("version", 1)
    # une note d'un autre format : refusée, jamais lue comme une autre — un ENTIER (pour Python, true vaut 1 et 2.0 vaut 2)
    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, VERSION):
        raise ValueError(f"note d'un format inconnu (version {str(version)[:12]}) : elle n'est pas relue")
    neuve = version == VERSION and etat.get(ZERO_SAISI) is True
    lire = (lambda e: e) if neuve else _de_v2 if version == VERSION else _migrer
    impression = etat.get(IMPRESSION, _impression_deduite(etat))
    if impression not in IMPRESSIONS:
        raise ValueError(f"impression de la note : {', '.join(IMPRESSIONS)} — reçu {str(impression)[:20]!r}")
    out = etat_de([], impression=impression)
    for i, e in enumerate(els):
        try:
            out["elements"].append(_relire(lire(e)))
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


def lire_carnet(carnet) -> List[Dict]:
    """(lot C, C6) La bibliothèque des PRS reçue de l'écran, relue (``profil_prs.lire_carnet``) : ce que la réponse rend."""
    return PP.lire_carnet(carnet)


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
    """Le taux le plus fort d'une barre, toutes méthodes confondues — et celui de l'âme quand elle ne passe pas
    (``profil_resultat.taux_barre``) ; None quand rien n'a de taux."""
    return R.taux_barre(r)


def _niveau(r: Dict) -> str:
    """Le niveau d'une barre pour la note : non conforme (r) ; une donnée à saisir (bloqué), une vérification non couverte,
    ou une méthode hors de son domaine : à vérifier (v) ; aucun effort : données incomplètes (n) ; sinon la couleur de son
    taux."""
    if r["statut"] == RE.NOK:
        return "r"
    if r["statut"] in M.STATUTS_A_VERIFIER:
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


def _lire_vide(valeur, quoi: str) -> Optional[float]:
    """Un champ qui peut se VIDER (s_s, c : « non saisi ») : vidé, None ; sinon un nombre, comme ``_lire``."""
    if valeur is None or (isinstance(valeur, str) and not valeur.strip()):
        return None
    return _lire(valeur, quoi)


def _rang(texte: str, liste: Sequence, quoi: str) -> int:
    i = int(texte) if texte.isdigit() else -1
    if not 0 <= i < len(liste):
        raise ValueError(f"{quoi} introuvable")
    return i


def _ajuster(el: Dict, L_avant: float) -> bool:
    """Après un changement de portée ou de modèle : une charge qui allait jusqu'au bout de la barre la suit ; ce qui
    dépasse est ramené dans la barre (une charge ou un maintien entièrement dehors est retiré). Vrai si quelque chose a
    été ramené ou retiré."""
    L = float(ST.longueur(el))            # la BARRE : débords compris (intégration du lot C)
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
        nt = c.get("N_troncons")
        if nt:                               # (lot C) N par tronçon : le dernier va jusqu'au bout ; ceux qui sortent sont retirés
            gardes = [dict(t) for t in nt[:-1] if t["x2"] < L - 1e-9]
            gardes.append(dict(nt[-1], x2=L))
            touche = touche or len(gardes) != len(nt)
            c["N_troncons"] = gardes
    dedans = [mt for mt in el["maintiens"] if mt["x"] <= L]
    touche = touche or len(dedans) != len(el["maintiens"])
    el["maintiens"] = dedans
    pts = [x for x in el.get("points") or [] if x <= L]
    touche = touche or len(pts) != len(el.get("points") or [])
    el["points"] = pts
    return touche


def _decaler(el: Dict, dx: float) -> None:
    """(lot C) Un débord GAUCHE allongé de ``dx`` (m) : les abscisses partent du bout gauche de la barre (rdm/poutres.py) —
    charges, maintiens, points vérifiés et fins des tronçons de N glissent d'autant et restent où ils étaient sur la barre
    (jamais une charge qui changerait de travée en silence)."""
    for c in el["cas"]:
        for ch in c["charges"]:
            for k in ("a", "b"):
                if k in ch:
                    ch[k] = round(max(0.0, ch[k] + dx), 6)
        for t in (c.get("N_troncons") or [])[:-1]:
            t["x2"] = round(max(0.0, t["x2"] + dx), 6)
    for mt in el["maintiens"]:
        mt["x"] = round(max(0.0, mt["x"] + dx), 6)
    el["points"] = [round(max(0.0, x + dx), 6) for x in el.get("points") or []]


def _milieu_libre(L: float, pris) -> float:
    """Le milieu du plus long intervalle de [0, L] que rien n'occupe (``pris`` : des abscisses, m)."""
    pts = sorted({0.0, L} | {round(x, 6) for x in pris if 0.0 <= x <= L})
    a, b = max(zip(pts, pts[1:]), key=lambda p: p[1] - p[0])
    return round(0.5 * (a + b), 3)


def maintiens_reguliers(texte, L: float) -> Tuple[int, float, float, List[float]]:
    """(lot C) Le geste « n maintiens réguliers entre a et b » : « 8 » (de 0 à L) ou « 8 2 20 » (de 2 à 20 m) → (n, a, b, les
    abscisses a + k·(b − a)/(n + 1), au mm). PRS-C (28 m) : 8 → un pas de 3,111 m ; 18 → 1,474 m (brief § 11.9)."""
    nombres = [C.lire_nombre(t) for t in re.findall(r"\d+(?:[.,]\d+)?", str(texte if texte is not None else ""))]
    if len(nombres) not in (1, 3):
        raise ValueError("maintiens réguliers : « n » (entre les bouts de la barre) ou « n a b » (de a à b, en m)")
    n, a, b = (nombres[0], 0.0, L) if len(nombres) == 1 else nombres
    if n != int(n) or not 1 <= n <= EN.MAINTIENS_MAX:
        raise ValueError(f"maintiens réguliers : n entier, de 1 à {EN.MAINTIENS_MAX}")
    if not 0.0 <= a < b <= L + 1e-9:
        raise ValueError(f"maintiens réguliers : 0 ≤ a < b ≤ {M.m(L)} m")
    n = int(n)
    return n, a, min(b, L), [round(a + k * (min(b, L) - a) / (n + 1), 3) for k in range(1, n + 1)]


def _contact(d: Dict, valeur, defaut: str) -> None:
    """(lot C) Le contact d'extrémité saisi (« oui », « non », « inconnu ») ; sa valeur à défaut n'est pas écrite (la clé absente la
    suit : « inconnu » pour une charge et pour les appuis, acier/barre/entree.py ; la valeur commune pour un appui un par un)."""
    if valeur == defaut:
        d.pop("contact", None)
    else:
        d["contact"] = str(valeur)


def _detail_neuf(el: Dict) -> List[Dict]:
    """Les appuis un par un, au passage à « un par un » : chacun reprend s_s et c communs (c aux appuis de rive) et la fourche."""
    ap, n = el["appuis"], len(ST.appuis_x(el))
    return [dict({"ss": ap["ss"]}, **({"c": ap["c"]} if i in (0, n - 1) else {}), torsion="fourche") for i in range(n)]


def _detail_ajuste(el: Dict) -> None:
    """Les appuis un par un suivent le nombre d'appuis (une travée ajoutée ou retirée, un autre modèle)."""
    d = el["appuis"].get("detail")
    if d is None:
        return
    n = len(ST.appuis_x(el)) if S.a_des_appuis(el) else 0
    if n == 0:
        el["appuis"]["detail"] = None
        return
    el["appuis"]["detail"] = [x if x is not None else {"torsion": "fourche"} for x in d[:n]] + [{"torsion": "fourche"}] * (n - len(d))


def _charge_neuve(type_: str, L: float, ancienne: Optional[Dict] = None) -> Dict:
    """Une charge du type demandé ; d'une ancienne (un changement de type), elle garde la valeur et, au mieux, la place."""
    v = ancienne["v"] if ancienne else {"q": 5.0, "tri": 5.0, "P": 10.0, "C": 10.0}[type_]
    niveau = {"niveau": ancienne["niveau"]} if ancienne and ancienne.get("niveau") and type_ != "C" else {}     # (lot C : le sien)
    if type_ in ("q", "tri"):
        a, b = (ancienne["a"], ancienne["b"]) if ancienne and "b" in ancienne else (0.0, L)
        ch = {"type": type_, "v": v, "a": a, "b": b, **niveau}
        if type_ == "tri":
            ch["sens"] = (ancienne or {}).get("sens", "croissante")
        return ch
    a = ancienne["a"] if ancienne and "b" not in ancienne else (round(0.5 * (ancienne["a"] + ancienne["b"]), 3) if ancienne else round(L / 2.0, 3))
    if type_ == "P":                         # (sa longueur d'appui rigide, en mm : None tant qu'elle n'est pas saisie — jamais supposée)
        return {"type": type_, "v": v, "a": a, "ss": (ancienne or {}).get("ss"), **niveau}
    return {"type": type_, "v": v, "a": a}


def _cas_neuf(el: Dict) -> Dict:
    """Un cas de charge de plus : variable (exploitation A) tant que le moteur en admet, permanent sinon."""
    L = float(ST.longueur(el))
    variables = sum(1 for c in el["cas"] if c["nature"] != CO.PERMANENT)
    lettre, nature = ("Q", "exploitation_A") if variables < CO.VARIABLES_MAX else ("G", CO.PERMANENT)
    pris = {c["nom"] for c in el["cas"]}
    nom, k = lettre, 2
    while nom in pris:
        nom, k = f"{lettre}{k}", k + 1
    return {"nom": nom, "nature": nature, "N": 0.0, "charges": [_charge_neuve("q", L)]}


def _maintien_neuf(el: Dict) -> Dict:
    """Un maintien de plus, au milieu du plus long intervalle que rien ne tient encore."""
    L = float(ST.longueur(el))
    tenus = {0.0, L} | {round(x, 6) for x in ST.appuis_x(el)}           # (les appuis : g, g + L₁, … — débord gauche compris)
    tenus |= {round(mt["x"], 6) for mt in el["maintiens"]}
    pts = sorted(tenus)
    a, b = max(zip(pts, pts[1:]), key=lambda p: p[1] - p[0])
    return {"x": round(0.5 * (a + b), 3), "genre": "complet"}


def _appliquer(el: Dict, cles: List[str], valeur, r: Dict) -> Optional[str]:
    """Écrit la saisie dans ``el`` (une copie) ; rend un mot à dire (« charges ramenées dans la barre ») ou None.
    KeyError : le champ n'existe pas ; ValueError : la valeur est refusée."""
    cle = "/".join(cles)
    k = cles[0]
    L = float(ST.longueur(el))
    if cle == "nom":
        nom = " ".join(str(valeur if valeur is not None else "").split())
        if len(nom) > NOM_MAX:
            raise ValueError(f"nom trop long : {len(nom)} caractères (au plus {NOM_MAX})")
        el["nom"] = nom or NOM_DEFAUT
    elif cle == "mode":
        if valeur not in EN.MODES:
            raise ValueError(f"mode inconnu : « {valeur} »")
        if valeur == "efforts":                 # les longueurs que la barre chargée prenait d'elle-même deviennent des données
            copiees = {}
            for a in ("y", "z"):
                if el["flambement"][a]["L"] is None:
                    el["flambement"][a]["L"] = copiees[a] = round(r["longueurs"][a]["L"], 3) if r.get("longueurs") else L
            if copiees:                         # (COPIEES : on s'en souvient, pour les rendre automatiques au retour)
                el[COPIEES] = copiees
        else:                                   # retour en barre chargée : une longueur copiée, pas retouchée, redevient automatique
            for a, Lc in (el.pop(COPIEES, None) or {}).items():
                if el["flambement"][a]["L"] == Lc:
                    el["flambement"][a]["L"] = None
        el["mode"] = valeur
    elif cle == "nuance":
        el[cle] = str(valeur)
        if valeur not in NU.AVEC_NORME:          # (lot C) la norme produit ne change que S420 et S460 : ailleurs, le défaut
            el["norme_produit"] = None
    elif cle in ("niveau", "ouvrage"):
        el[cle] = str(valeur)
    elif cle == "norme_produit":                 # (lot C) « » : N/NL (EN 10025-3) ; « M » : M/ML (EN 10025-4)
        el[cle] = str(valeur) if valeur else None
    elif cle == "poids_propre":
        el[cle] = _oui(valeur)
    elif cle == "position":                      # (lot C) horizontale, verticale, inclinée — l'angle écrit, jamais supposé en silence
        el["position"] = str(valeur)
        if valeur == "inclinee" and el.get("angle") is None:
            el["angle"] = M.ANGLE_DEFAUT
    elif cle == "angle":
        el["angle"] = _lire(valeur, "angle de la barre")
    elif k == "combinaisons" and len(cles) == 2 and cles[1] in ("expression", "geotechnique", "regle_courte_duree"):
        el["combinaisons"][cles[1]] = str(valeur) if cles[1] == "expression" else _oui(valeur)
    elif k == "points":                          # (lot C) les sections vérifiées en plus
        pts = el["points"]
        if cle == "points/+":
            if len(pts) >= EN.POINTS_MAX:
                raise ValueError(f"au plus {EN.POINTS_MAX} points vérifiés")
            pts.append(_milieu_libre(L, pts))
        elif len(cles) == 3 and cles[2] == "-":
            del pts[_rang(cles[1], pts, "point vérifié")]
        elif len(cles) == 2:
            pts[_rang(cles[1], pts, "point vérifié")] = _lire(valeur, "abscisse du point vérifié")
        else:
            raise KeyError(cle)
    elif k == "debords" and len(cles) == 2 and cles[1] in ("0", "1"):
        if el["modele"] not in PO.MODELES_DEBORDS:
            raise ValueError("débords : seulement pour une barre sur deux appuis ou continue")
        x = C.lire_auto(valeur)
        x = 0.0 if x is None else x
        if cles[1] == "0":                       # un débord gauche : les abscisses glissent avec la barre
            _decaler(el, x - el["debords"][0])
        el["debords"][int(cles[1])] = x
    elif k == "poids_propre_efforts" and len(cles) == 2 and cles[1] in ("N", "M"):
        el[k][cles[1]] = _oui(valeur)
    elif cle == "modele":
        if valeur not in PO.MODELES:
            raise ValueError(f"modèle inconnu : « {valeur} »")
        if valeur == "continue" and len(el["travees"]) < 2:
            el["travees"] = [el["travees"][0], el["travees"][0]]
        elif valeur != "continue":
            el["travees"] = el["travees"][:1]
        el["modele"] = valeur
        if valeur not in PO.MODELES_DEBORDS and any(el["debords"]):     # (lot C) un débord n'existe que pour ces deux modèles
            _decaler(el, -el["debords"][0])
            el["debords"] = [0.0, 0.0]
        _detail_ajuste(el)
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
        _detail_ajuste(el)
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
                if c["nature"] not in CO.PSI0_SAISI:          # (lot C) ψ₀ n'est saisi que pour l'exécution et les machines
                    c["psi0"] = None
                if c["nature"] in CO.COURTES or c["nature"] == CO.PERMANENT:
                    c["courte_duree"] = False
            else:
                c["N"] = _lire(valeur, "effort normal")
        elif len(cles) == 3 and cles[2] in ("courte_duree", "psi0", "N_troncons"):          # (lot C)
            c = cas[_rang(cles[1], cas, "cas de charge")]
            if cles[2] == "courte_duree":
                c["courte_duree"] = _oui(valeur)
            elif cles[2] == "psi0":
                c["psi0"] = _lire_vide(valeur, "ψ₀")
            elif valeur == "constant":
                c["N_troncons"] = None
            elif valeur == "troncons":
                if not c.get("N_troncons"):          # deux moitiés au N du cas : rien ne change tant qu'on n'y touche pas
                    c["N_troncons"] = [{"x2": round(L / 2.0, 3), "N": c["N"]}, {"x2": L, "N": c["N"]}]
            else:
                raise ValueError(f"effort normal : « constant » ou « troncons », reçu « {valeur} »")
        elif len(cles) >= 4 and cles[2] == "N_troncons":
            c = cas[_rang(cles[1], cas, "cas de charge")]
            nt = c.get("N_troncons")
            if not nt:
                raise KeyError(cle)
            if len(cles) == 4 and cles[3] == "+":    # le plus long tronçon coupé en son milieu, au même N
                debuts = [0.0] + [t["x2"] for t in nt[:-1]]
                k_ = max(range(len(nt)), key=lambda q: nt[q]["x2"] - debuts[q])
                nt.insert(k_, {"x2": round(0.5 * (debuts[k_] + nt[k_]["x2"]), 3), "N": nt[k_]["N"]})
            elif len(cles) == 5:
                j = _rang(cles[3], nt, "tronçon")
                if cles[4] == "-":
                    if len(nt) == 1:
                        raise ValueError("il reste un tronçon : choisir « constant »")
                    del nt[j]
                    nt[-1]["x2"] = L
                elif cles[4] == "N":
                    nt[j]["N"] = _lire(valeur, "N du tronçon")
                elif cles[4] == "x2" and j < len(nt) - 1:
                    nt[j]["x2"] = _lire(valeur, "fin du tronçon")
                else:
                    raise KeyError(cle)
            else:
                raise KeyError(cle)
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
                elif champ == "ss" and chs[j]["type"] == "P":
                    chs[j]["ss"] = _lire_vide(valeur, "longueur d'appui")
                elif champ == "niveau" and chs[j]["type"] != "C":       # (lot C) le niveau de CETTE charge ; celui de la barre : None
                    chs[j]["niveau"] = None if valeur == el["niveau"] else str(valeur)
                elif champ == "introduction" and chs[j]["type"] == "P":
                    chs[j]["introduction"] = str(valeur)
                elif champ == "contact" and chs[j]["type"] == "P":
                    _contact(chs[j], valeur, "inconnu")
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
        elif cle == "maintiens/reguliers":       # (lot C) n maintiens réguliers de a à b : ceux qui y étaient sont remplacés
            if valeur is None or not str(valeur).strip():
                return None                      # (le champ vide, tel qu'il s'affiche : rien à poser)
            n, a, b, xs = maintiens_reguliers(valeur, L)
            gardes = [mt for mt in mts if not a + 1e-9 < mt["x"] < b - 1e-9]
            if len(gardes) + n > EN.MAINTIENS_MAX:
                raise ValueError(f"au plus {EN.MAINTIENS_MAX} maintiens (la barre en aurait {len(gardes) + n})")
            el["maintiens"] = sorted(gardes + [{"x": x, "genre": "complet", "neglige": False} for x in xs], key=lambda m: m["x"])
        elif len(cles) == 3:
            j = _rang(cles[1], mts, "maintien")
            if cles[2] == "-":
                del mts[j]
            elif cles[2] == "x":
                mts[j]["x"] = _lire(valeur, "abscisse du maintien")
            elif cles[2] == "genre":
                mts[j]["genre"] = str(valeur)
            elif cles[2] == "neglige":              # (lot C) négligé dans M_cr (7.7)
                mts[j]["neglige"] = _oui(valeur)
            else:
                raise KeyError(cle)
        else:
            raise KeyError(cle)
    elif cle == "appuis/mode":
        if valeur not in EN.MODES_APPUI:
            raise ValueError(f"appuis : « {valeur} » inconnu")
        el["appuis"]["mode"] = valeur
    elif cle in ("appuis/ss", "appuis/c"):
        el["appuis"][cles[1]] = _lire_vide(valeur, "longueur d'appui" if cles[1] == "ss" else "débord")
    elif cle == "appuis/contact":                # (lot C) le contact d'extrémité commun (un lieu du type (c))
        _contact(el["appuis"], valeur, "inconnu")
    elif cle == "appuis/detail":                 # (lot C) les appuis identiques, ou un par un
        if valeur == "identiques":
            el["appuis"]["detail"] = None
        elif valeur == "un_par_un":
            if el["appuis"].get("detail") is None:
                el["appuis"]["detail"] = _detail_neuf(el)
        else:
            raise ValueError(f"appuis : « identiques » ou « un_par_un », reçu « {valeur} »")
    elif k == "appuis" and len(cles) == 4 and cles[1] == "detail" and cles[3] in ("ss", "c", "contact", "torsion"):
        d = el["appuis"].get("detail")
        if d is None:
            raise KeyError(cle)
        i = _rang(cles[2], d, "appui")
        e = d[i] = dict(d[i] or {})
        if cles[3] == "contact":                 # (la valeur commune n'est pas recopiée : la clé absente la suit)
            _contact(e, valeur, el["appuis"].get("contact", "inconnu"))
        else:
            e[cles[3]] = (_lire_vide(valeur, "longueur d'appui" if cles[3] == "ss" else "débord") if cles[3] in ("ss", "c") else str(valeur))
    elif k == "flambement" and len(cles) == 3 and cles[1] in ("y", "z") and cles[2] in ("L", "u"):
        if cles[2] == "L" and isinstance(el.get(COPIEES), dict):     # une longueur retouchée est une saisie : elle reste au retour
            el[COPIEES].pop(cles[1], None)
            if not el[COPIEES]:
                del el[COPIEES]
        if cles[2] == "L" and el["mode"] == "chargee":
            el["flambement"][cles[1]]["L"] = C.lire_auto(valeur)
        else:
            el["flambement"][cles[1]][cles[2]] = _lire(valeur, "longueur de flambement" if cles[2] == "L" else "coefficient u")
    elif k == "flambement" and len(cles) == 3 and cles[1] in ("y", "z") and cles[2] == "courbe":      # (lot C) « » : celle de la norme
        el["flambement"][cles[1]]["courbe"] = str(valeur) if valeur else None
    elif cle == "Lc":
        el["Lc"] = C.lire_auto(valeur)
    elif k == "efforts" and len(cles) == 3 and cles[1] in ("service", "elu") and cles[2] in EN.EFFORTS:
        el["efforts"][cles[1]][cles[2]] = _lire(valeur, cles[2])
    elif k == "diagramme" and len(cles) == 3 and cles[1] in ("y", "z") and cles[2] in ("forme", "psi"):
        el["diagramme"][cles[1]][cles[2]] = str(valeur) if cles[2] == "forme" else _lire(valeur, "ψ")
    elif k == "historique" and len(cles) == 2 and cles[1] in ("s", "C", "deversement", "Cd", "variante", "facteur_doc", "classe_d"):
        # (la tolérance de 5 % : retirée, § 13.2 point 5) ; (lot C) C_d, la variante, le facteur 5,1, la classe imposée (B1)
        h, q = el["historique"], cles[1]
        if q in ("s", "Cd"):
            h[q] = _lire(valeur, "s" if q == "s" else "C_d")
        elif q in ("deversement", "facteur_doc"):
            h[q] = _oui(valeur)
        else:
            h[q] = (str(valeur) if valeur else None) if q == "classe_d" else str(valeur)
    elif k == "prs":                          # (lot C, C6) le PRS : « prs » (catalogue ∣ prs), ses plats, la gorge, les bords, la case
        return PP.appliquer(el, cles, valeur)
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
    if cles in (["impression"], ["methodes"]):
        # (lot C, C0d) ce que la NOTE imprime, pour toutes ses barres (le segment de la rubrique « Méthodes » ; « methodes »,
        # son ancien chemin, y mène aussi) : les deux méthodes restent calculées et montrées
        if valeur not in IMPRESSIONS:
            return f"Saisie refusée : méthode inconnue : « {valeur} » ({', '.join(IMPRESSIONS)})"
        etat[IMPRESSION] = valeur
        return None
    if cles[0] == "carnet":
        # (lot C, C6) la bibliothèque des PRS : y enregistrer le PRS de la barre, l'en retirer (la barre le garde)
        try:
            if cles not in (["carnet", "garder"], ["carnet", "retirer"]):
                raise KeyError("/".join(cles))
            etat["carnet"], mot = (PP.garder if cles[1] == "garder" else PP.retirer)(el, etat.get("carnet"))
        except KeyError:
            return f"Champ inconnu : « {'/'.join(cles)} »"
        except ValueError as e:
            return f"Saisie refusée : {M.minuscule(str(e))}"
        return C.Info(mot)
    try:
        mot = _appliquer(essai, cles, valeur, calculer(el))
        etat["elements"][etat["actif"]] = _relire(essai)
    except KeyError:
        return f"Champ inconnu : « {'/'.join(cles)} »"
    except (ValueError, TypeError, IndexError) as e:
        # (lot C, C6) un sigle garde ses capitales : « PRS : … », jamais « pRS : … » (profil_mots.minuscule)
        return f"Saisie refusée : {M.minuscule(str(e))}"
    return mot


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le clic dans le sélecteur : le profilé devient celui de la barre — (lot C, C6) un PRS de la bibliothèque (le carnet)
    y apporte une copie de ses plats et sa nuance ; un profilé du catalogue efface le PRS."""
    if role != ROLE:
        raise KeyError(f"sélecteur inconnu : {role!r}")
    el = copy.deepcopy(_el(etat))
    entree = next((e for e in PP.lire_carnet(etat.get("carnet")) if e["nom"] == nom), None)
    if entree is not None:
        PP.prendre(el, entree)
    elif CAT.par_nom().get(nom, {}).get("famille") not in FAMILLES:
        raise ValueError(f"profilé inconnu : {nom!r}")
    else:
        el.update(profil=nom, prs=None)
        el["parametres"].pop(PP.SOUDEE, None)
    etat["elements"][etat["actif"]] = _relire(el)


# ------------------------------------------------------------------ la colonne de gauche
def arbre(etat: Dict, r: Dict, resultats: List[Dict]) -> str:
    """La colonne de gauche COMPACTE : la ligne de la barre ouverte (son nom, la liste des barres de la note, copier,
    supprimer — le nom de chacune à la couleur de son état), puis ses quatre rubriques."""
    noms = [(e.get("nom") or f"B{i + 1}", "· " + R.dims(e), False) for i, e in enumerate(etat["elements"])]
    niveaux = [_niveau(x) if "methodes" in x else "n" for x in resultats]
    return E.arbre(GENRE, noms, etat["actif"], S.formulaire(_el(etat), r, etat.get(IMPRESSION, IMPRESSIONS[0]), lire_carnet(etat.get("carnet"))),
                   "choixprofil", compact=True, niveaux=niveaux)


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


def refus_prs(nom: str) -> str:
    """(lot C, reprise de C0) Le refus d'un nom de PRS demandé SANS sa barre (``fiche_profil``) : une ligne qui le nomme —
    jamais un KeyError, jamais un profilé du catalogue à sa place. Une barre PRS, elle, se relit, se calcule, se montre à
    l'écran (C6 : sa saisie, sa fiche, son croquis) et sort dans la note (C5 : son croquis et ses propriétés au panneau)."""
    return f"« {nom} » : la note d'un PRS n'est pas encore prête."


def fiche_profil(nom: str, barre: Optional[Dict] = None) -> Dict:
    """Ce qu'on écrit d'un profilé du catalogue : {"famille", "dims" (mm), "dimensions" et "proprietes" : [(symbole,
    valeur écrite, unité)], "sous" (sa famille et sa norme)} — la vue « Section » de l'écran et le panneau de la note PDF.
    (lot C, C6) ``barre`` : une barre PRS — la fiche de SON PRS (``profil_prs.fiche`` : la même forme, et « tableau », les
    propriétés avec leur origine). Un nom hors du catalogue sans sa barre : ValueError, ``refus_prs``."""
    if barre is not None and barre.get("prs"):
        return PP.fiche(barre)
    rec = CAT.par_nom().get(nom)
    if rec is None:
        raise ValueError(refus_prs(nom))
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
    """La section du profilé de la barre, à l'échelle ; ``fiche`` : ses dimensions et ses propriétés dessous. (lot C, C6)
    Un PRS : son croquis coté (dessin_profil.section_prs), le tableau de ses propriétés dessous (profil_prs.proprietes)."""
    if el.get("prs"):
        s = PP.section(el)
        return DP.section_prs(s, None, ident, PP.proprietes(s) if fiche else None)
    f = fiche_profil(el["profil"])
    return DP.section(ident, el["profil"], f["sous"], f["famille"], f["dims"], f["dimensions"], f["proprietes"], "", fiche)


def _ame_en_lieux(a: Dict) -> Dict:
    """Le bloc de l'âme pour la vue des taux : ses vérifications du cisaillement, puis UNE barre par lieu (son taux le
    plus fort, et la paire de plats quand il en reçoit une) — les titres de ses vérifications sont trop longs pour la
    marge du dessin, et c'est la décision du lieu qu'on y lit."""
    des_lieux = {k for L in a["lieux"] for k in L["verifications"]}
    vs = [v for v in a["verifications"] if v["id"] not in des_lieux]
    for L in a["lieux"]:
        plats = f" · {L['raidisseur']['designation'].replace(' mm', '')}" if L["decision"] == "raidisseur" else ""
        vs.append({"id": L["gouvernant"], "titre": L["nom"] + plats, "taux": L["taux"], "statut": L["statut"]})
    return {"norme": a["norme"], "gouvernant": a.get("gouvernant"), "verifications": vs}


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
    if any(x["verifications"] for x in r["methodes"].values()) or r["ame"]["verifications"]:
        leg = "le taux de chaque vérification ; la limite de 100 % en tireté"
        ms = [(f"{M.METHODES[k]}", x) for k, x in r["methodes"].items() if x["verifications"]]
        if r["ame"]["verifications"]:
            ms.append((M.TITRE_AME, _ame_en_lieux(r["ame"])))
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
    k, m = R.pire(x)
    g = R.gouvernante(m)
    ou = "âme" if k == M.AME else M.minuscule(M.METHODES_COURT[k])
    quoi = f" · {g['titre'][:1].lower() + g['titre'][1:]} ({ou})" if g else ""
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
    méthodes que la note imprime (``IMPRESSION`` ; api_web/note_profil.py, chargé à la demande : il lui faut le paquet de la note) ;
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


def _candidat_prs(el: Dict, sig: str, plats: Dict, nuance: str) -> Tuple[Optional[float], str]:
    """(taux le plus fort, statut) de la barre avec ce PRS de la bibliothèque — gardé."""
    cle = (sig, json.dumps([plats, nuance], sort_keys=True))
    c = _CANDIDATS.get(cle)
    if c is None:
        if len(_CANDIDATS) >= MEMOIRE_CANDIDATS:
            _CANDIDATS.clear()
        try:
            r = MO.verifier(dict(el, prs=plats, nuance=nuance))
            c = (_taux(r), r["statut"])
        except ValueError:
            c = (None, RE.NON_COUVERT)
        _CANDIDATS[cle] = c
    return c


def _selecteur_prs(etat: Dict, el: Dict) -> str:
    """(lot C, C6) La BIBLIOTHÈQUE DES PRS (le carnet du module, dans ce navigateur), triée par masse, chacun vérifié par le
    moteur avec la barre : son taux ; le retenu (✓) marqué — pas de ★ (une bibliothèque n'a pas de « plus léger qui
    tient » : elle ne contient que ce qu'on y a mis)."""
    sig = _signature(el, ("nom", "profil", "prs", "nuance"))
    rows = []
    sien = PP.enregistrement(el)                # (le retenu : le MÊME enregistrement — plats, gorge, bords, nuance —, pas le seul nom)
    for e in PP.lire_carnet(etat.get("carnet")):
        s, plats = PP.plats_entree(e, el["nuance"])
        t, statut = _candidat_prs(el, sig, plats, s["nuance"])
        row = {"nom": s["nom"], "col": C.fr(s["masse"], 1), "taux": None, "niveau": "c", "cur": e == sien, "opt": False,
               "prop": PP.ligne_selecteur(s, t), "masse": s["masse"]}
        if statut in M.STATUTS_A_VERIFIER:
            row["etat"] = M.MOT_STATUT[statut].lower() if t is None else f"{C.pc(C.pct(min(t, 9.99)))} % ?"
        elif t is not None:
            row.update({"taux": C.pct(min(t, 9.99)), "niveau": "r" if statut == RE.NOK else C.niveau(t)})
        else:
            row["etat"] = "refusé" if statut == RE.NOK else "—"
        rows.append(row)
    rows.sort(key=lambda x: (x.pop("masse"), x["nom"]))
    i_c = next((k for k, x in enumerate(rows) if x["cur"]), 0)
    sous = "bibliothèque de ce navigateur · triés par masse · un clic retient le PRS · « Enregistrer » y ajoute celui de la barre"
    return C.selecteur(ROLE, f"{C.esc(el.get('nom') or 'Barre')} · PRS", sous, "", rows, max(0, i_c - 4), "kg/m", etoile="")


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les profilés des familles demandées (par défaut celle du profilé de la barre), TRIÉS PAR MASSE, chacun vérifié
    par le moteur avec les méthodes de la barre : son taux le plus fort ; le retenu (✓) et le plus léger qui tient (★)
    marqués. Un profilé qu'une méthode ne couvre pas le dit, sans couleur de conformité."""
    if role != ROLE:
        raise KeyError(f"sélecteur inconnu : {role!r}")
    el = _el(etat)
    if el.get("prs") is not None:          # (lot C, C6) un PRS : la bibliothèque des PRS, jamais le catalogue calculé avec ses plats
        return _selecteur_prs(etat, el)
    demandees = familles if isinstance(familles, (list, tuple)) else []      # une requête abîmée : la famille du profilé
    # (lot C) les U et les cornières ne se proposent qu'en efforts imposés (la barre chargée les refuse : entree.MSG_EFFORTS_SEULS)
    offertes = FAMILLES if el["mode"] == "efforts" else FAMILLES_DEUX
    fams = [f for f in offertes if f in demandees] or [CAT.par_nom()[el["profil"]]["famille"]]
    sig = _signature(el, ("nom", "profil"))
    rows, vu_leger = [], False
    for nom in _noms(fams):
        t, statut = _candidat(el, sig, nom)
        tient = statut == RE.OK
        row = {"nom": nom, "col": C.fr(CAT.par_nom()[nom]["masse"], 1), "taux": None, "niveau": "c", "cur": nom == el["profil"],
               "opt": tient and not vu_leger, "prop": _proprietes(nom, t)}
        vu_leger = vu_leger or tient
        if statut in M.STATUTS_A_VERIFIER:      # (bloqué, non couvert, hors domaine : sans couleur ni ★ ; le taux suivi de « ? »)
            row["etat"] = M.MOT_STATUT[statut].lower() if t is None else f"{C.pc(C.pct(min(t, 9.99)))} % ?"
        elif t is not None:
            row.update({"taux": C.pct(min(t, 9.99)), "niveau": "r" if statut == RE.NOK else C.niveau(t)})
        else:
            row["etat"] = "refusé" if statut == RE.NOK else "—"
        rows.append(row)
    i_o = next((k for k, x in enumerate(rows) if x["opt"]), 0)
    i_c = next((k for k, x in enumerate(rows) if x["cur"]), i_o)
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{C.esc(f)}">{C.esc(f)}</button>' for f in offertes)
    methodes = " et ".join(M.minuscule(M.METHODES_COURT[k]) for k in EN.METHODES)          # (toujours les deux : lot C, C0d)
    sous = f"triés par masse · vérifiés par : {methodes} · un clic retient le profilé"
    return C.selecteur(ROLE, f"{C.esc(el.get('nom') or 'Barre')} · profilé", sous, filtres, rows, max(0, min(i_o, i_c) - 4), "kg/m")

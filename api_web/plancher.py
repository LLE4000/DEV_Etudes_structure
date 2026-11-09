# -*- coding: utf-8 -*-
"""Le module « Charges sur plancher » du site : charges temporaires de chantier (engins sur stabilisateurs ou sur
roues, étançons, files d'étais, stocks) sur un plancher de hourdis, une dalle coulée en place ou une prédalle ; les
plateaux et les filières qui répartissent ; plusieurs cas par note — rendus en fragments HTML.

Le calcul est celui du moteur ``plancher/charges`` (le même que l'écran Streamlit modules/charges_plancher.py, qui
reste le secours) : l'état de la note EST l'état du moteur (etat.py : cas multiples, fiches ajoutées), construit en
objets par ``etat.construire`` et vérifié par ``verification.verifier`` ; les opérations que l'écran d'origine faisait
lui-même (proposer une filière, les fiches, les parts de l'annexe C…) sont extraites dans ``operations.py`` et leur
parité est prouvée (tests/test_charges_plancher_parite.py). Ce fichier MET EN FORME : il ne calcule rien. Les
formules des fenêtres de calcul substituent les valeurs que le moteur a nommées ; les sélecteurs (plateau sous un
appui, profilé d'une filière) relancent le moteur pour chaque ligne, comme la proposition de filière.

Ce que le plan cliquable de l'écran d'origine réglait à la souris (glisser un engin, un plateau, une filière, la cote
L, le sens porteur, le nombre de hourdis, les fenêtres de chaque objet) se SAISIT ici dans le cadre Données : position
x, y et rotation de chaque charge, répartiteur de chaque appui, élément, longueur, orientation et position de chaque
filière ; le plan à l'échelle se redessine. La note PDF se fait sur le site (``telecharger``, api_web/note_cas.py) par le
générateur validé de l'écran d'origine.
"""
import copy
import json
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from plancher.charges import bibliotheque as BIB
from plancher.charges import diffusion as DF
from plancher.charges import etat as ET
from plancher.charges import operations as OP
from plancher.charges import repartiteurs as RP
from plancher.charges import resistances as RS
from plancher.charges import verification as V
from plancher.charges.parametres import DEFAUT as PAS

from . import commun as C
from . import dessin_plancher as DP
from . import note_cas as NC

MODULE = "plancher"
PAGE = "charges-plancher"

# Les fichiers que le moteur ne lit que sur des chemins rares (une filière d'un profilé du catalogue lit la nuance
# d'acier) sont lus ICI, au chargement : le relevé du paquet du site (site/web/construire.py) les voit toujours.
BIB.aciers()
BIB.profils()

TYPES_COURTS = {"hourdis": "Hourdis", "dalle": "Dalle BA", "predalle": "Prédalle"}
NATURES = [("Rd", "Rd (calcul)"), ("service", "Service")]
SENS = [(1, "Gauche / droite"), (2, "Haut / bas")]
ANGLES = [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]
ORIENT_CHARGE = [("y", "En travers"), ("x", "Sens porteur")]
ORIENT_FIL = [("y", "En travers"), ("x", "Sens porteur")]
TYPES_CHARGE = {"engin": "Engin", "ponctuel": "Étançon", "lineaire": "File d'étais", "surfacique": "Stock"}
TYPES_APPUIS = [("pieds", "Stabilisateurs"), ("roues", "Roues"), ("chenilles", "Chenilles")]
SANS = "Sans répartiteur"
FAM_FIL = ("IPE", "HEA", "HEB", "FICHE")
LIB_FAM_FIL = {"IPE": "IPE", "HEA": "HEA", "HEB": "HEB", "FICHE": "Fiches"}
UNITES = {"kN·m": 1, "kN": 1, "kN/m": 2, "kN·m/m": 2, "N/mm²": 2, "mm": 0}


# ------------------------------------------------------------------ l'état de la note
def _site_defaut() -> Dict:
    """Les brouillons des formulaires de la note (nouvelle fiche d'engin, nouvel élément de répartition, nom d'une
    fiche de hourdis) : l'écran d'origine les tenait dans ses widgets, le site les garde dans la note."""
    return {"fiche_engin": {"nom": "", "type_appuis": "pieds", **OP.FICHE_ENGIN_DEFAUT, "source": "",
                            "appuis": copy.deepcopy(OP.APPUIS_FICHE_DEFAUT)},
            "fiche_rep": dict(OP.FICHE_REP_DEFAUT), "nom_fiche_hourdis": ""}


def _site_cas() -> Dict:
    """Ce que le site garde par cas en plus de l'état du moteur : l'engin balayé et la dernière proposition."""
    return {"balaye": None, "proposition": None}


def nouvel_etat() -> Dict:
    """La note neuve : celle de l'écran d'origine (un cas C1, hourdis générique à renseigner, aucune charge)."""
    e = ET.nouvel_etat()
    e["cas"][0]["site"] = _site_cas()
    e["site"] = _site_defaut()
    return e


def _nombre(x, ou: str, nul: bool = False):
    if x is None and nul:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{ou} : nombre attendu, lu {x!r}")
    return x


def _texte(x, ou: str) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        raise ValueError(f"{ou} : texte attendu, lu {x!r}")
    return x


def _parmi(x, options, ou: str):
    if x not in options:
        raise ValueError(f"{ou} : {x!r} n'est pas une valeur admise ({', '.join(map(str, options))})")
    return x


def _completer(d: Dict, defaut: Dict) -> Dict:
    """Complète ``d`` avec les clés manquantes de ``defaut``, en profondeur (les fichiers anciens)."""
    for k, v in defaut.items():
        if k not in d:
            d[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and v and isinstance(d[k], dict):
            _completer(d[k], v)
    return d


NULS_HOURDIS = ("I", "M_Rd_chape", "V_Rd_chape", "M_cr", "p_loc_adm")
NOMBRES_HOURDIS = ("b_element", "h", "g", "E", "M_Rd", "V_Rd")
NOMBRES_DALLE = ("h", "d", "As_inf", "fyk", "gamma_s", "h1", "largeur")
NOMBRES_PREDALLE = NOMBRES_DALLE + ("b_predalle",)
NULS_PARAMS = ("k_flexion_heft240", "joint_adm_kN_m", "joint_longueur_m", "coef_vrd_max")
CLES_CHARGE = {"engin": ("x", "y"), "ponctuel": ("x", "y", "a", "b", "P", "phi"),
               "lineaire": ("x", "y", "longueur", "largeur", "p", "phi"), "surfacique": ("x0", "y0", "lx", "ly", "q")}


def _relire_fiches(f) -> Dict:
    if f is None:
        f = {}
    if not isinstance(f, dict):
        raise ValueError("fiches : dictionnaire attendu")
    out = {"engins": [], "repartiteurs": [], "hourdis": []}
    for k in out:
        lst = f.get(k) or []
        if not isinstance(lst, list) or not all(isinstance(x, dict) for x in lst):
            raise ValueError(f"fiches {k} : liste de fiches attendue")
        out[k] = lst
    for i, g in enumerate(out["engins"]):
        ou = f"fiche d'engin {i + 1}"
        _texte(g.get("nom"), ou)
        if not g.get("nom") or not isinstance(g.get("appuis"), list):
            raise ValueError(f"{ou} : nom et appuis attendus")
        for a in g["appuis"]:
            if not isinstance(a, dict):
                raise ValueError(f"{ou} : appui illisible")
            _texte(a.get("nom"), ou)
            for c in ("dx", "dy", "a", "b"):
                _nombre(a.get(c), f"{ou}, appui {c}")
            _nombre(a.get("P", 0.0), f"{ou}, appui P")
            _nombre(a.get("phi", 1.0), f"{ou}, appui φ")
    for i, d in enumerate(out["repartiteurs"]):
        ou = f"fiche de répartiteur {i + 1}"
        if not isinstance(d.get("nom"), str) or not d["nom"]:
            raise ValueError(f"{ou} : nom attendu")
        _parmi(d.get("type"), ("plateau", "filiere"), ou)
        _nombre(d.get("f"), f"{ou}, f", nul=True)
        if d["type"] == "plateau":
            for c in ("t", "A", "Bp"):
                _nombre(d.get(c), f"{ou}, {c}")
        else:
            sec = d.get("section")
            if not isinstance(sec, dict) or not (sec.get("profil") in BIB.profils() or ("b" in sec and "h" in sec)):
                raise ValueError(f"{ou} : section (profilé du catalogue ou b × h) attendue")
            if not sec.get("profil"):
                _nombre(sec["b"], f"{ou}, b")
                _nombre(sec["h"], f"{ou}, h")
    for i, d in enumerate(out["hourdis"]):
        if not isinstance(d.get("nom"), str) or not d["nom"]:
            raise ValueError(f"fiche de hourdis {i + 1} : nom attendu")
    return out


def _relire_cas(cas, i: int, etat: Dict) -> Dict:
    """Un cas relu : clés manquantes complétées (défauts de l'écran d'origine), types contrôlés."""
    ou = f"cas {i + 1}"
    if not isinstance(cas, dict):
        raise ValueError(f"{ou} : dictionnaire attendu")
    _completer(cas, ET.nouveau_cas(cas.get("nom") if isinstance(cas.get("nom"), str) else f"C{i + 1}"))
    _completer(cas, {"site": _site_cas()})
    cas["nom"] = _texte(cas["nom"], f"{ou}, nom")
    pl = cas["plancher"]
    _parmi(pl["type"], tuple(TYPES_COURTS), f"{ou}, type de plancher")
    _parmi(pl["sens"], (1, 2), f"{ou}, sens")
    _nombre(pl["L"], f"{ou}, L")
    _nombre(pl["q_presentes"], f"{ou}, q")
    h = pl["hourdis"]
    for k in NOMBRES_HOURDIS:
        _nombre(h[k], f"{ou}, hourdis {k}")
    h["n_elements"] = int(_nombre(h["n_elements"], f"{ou}, nombre d'éléments"))
    for k in NULS_HOURDIS:
        _nombre(h[k], f"{ou}, hourdis {k}", nul=True)
    _parmi(h["etat"], tuple(OP.ETATS_H), f"{ou}, état des hourdis")
    _parmi(h["nature"], ("Rd", "service"), f"{ou}, nature")
    _texte(h["type_cat"], f"{ou}, type de hourdis")
    h["source"] = _texte(h["source"], f"{ou}, source")
    ch = h["chape"]
    for k in ("epaisseur", "fck_t"):
        _nombre(ch[k], f"{ou}, chape {k}")
    _nombre(ch["fck_exigee"], f"{ou}, classe exigée", nul=True)
    ch["oui"], ch["collaborante"] = bool(ch["oui"]), bool(ch["collaborante"])
    if h["parts_annexe_C"] is not None:
        if not isinstance(h["parts_annexe_C"], dict):
            raise ValueError(f"{ou} : parts de l'annexe C illisibles")
        h["parts_annexe_C"] = {str(int(k)): float(_nombre(v, f"{ou}, part {k}")) for k, v in h["parts_annexe_C"].items()} or None
    for t, cles in (("dalle", NOMBRES_DALLE), ("predalle", NOMBRES_PREDALLE)):
        d = pl[t]
        for k in cles:
            _nombre(d[k], f"{ou}, {t} {k}")
        _nombre(d["fck_t"], f"{ou}, {t} fck(t)", nul=True)
        _parmi(d["beton"], tuple(RS.BETON), f"{ou}, béton")
        d["armature_transversale_ok"] = bool(d["armature_transversale_ok"])
    pd = pl["predalle"]
    pd["joints_armes"], pd["chape_durcie"] = bool(pd["joints_armes"]), bool(pd["chape_durcie"])
    _nombre(pd["portee_etayee"], f"{ou}, portée étayée", nul=True)
    pr = cas["params"]
    for k, v in pr.items():
        _nombre(v, f"{ou}, paramètre {k}", nul=k in NULS_PARAMS)
    # les bornes hautes du site (LIMITES) : un fichier ne doit pas pouvoir bloquer le navigateur
    for x, lim, quoi in ((pl["L"], LIMITES["L"], "portée L (m)"), (h["n_elements"], LIMITES["n_elements"], "nombre d'éléments"),
                         (h["b_element"], LIMITES["b_element"], "largeur d'élément (m)"), (pl["dalle"]["largeur"], LIMITES["largeur"], "largeur de dalle (m)"),
                         (pl["predalle"]["largeur"], LIMITES["largeur"], "largeur de prédalle (m)")):
        if x > lim:
            raise ValueError(f"{ou} : {quoi} = {C.nb(x)}, au plus {C.nb(lim)}")
    gabs = ET.gabarits(etat)
    if not isinstance(cas["charges"], list):
        raise ValueError(f"{ou} : liste de charges attendue")
    for c in cas["charges"]:
        if not isinstance(c, dict):
            raise ValueError(f"{ou} : charge illisible")
        _parmi(c.get("type"), tuple(TYPES_CHARGE), f"{ou}, type de charge")
        if not isinstance(c.get("ref"), str) or not c["ref"]:
            raise ValueError(f"{ou} : charge sans référence")
        for k in CLES_CHARGE[c["type"]]:
            _nombre(c.get(k), f"{ou}, charge {c['ref']} {k}")
        if c["type"] == "engin":
            if c.get("gabarit") not in gabs:
                raise ValueError(f"{ou} : engin {c['ref']} — fiche « {c.get('gabarit')} » absente de la bibliothèque du fichier")
            c["angle"] = int(_parmi(c.get("angle", 0), (0, 90, 180, 270), f"{ou}, angle de {c['ref']}"))
            if not isinstance(c.get("P") or {}, dict):
                raise ValueError(f"{ou} : charges par appui de {c['ref']} illisibles")
            c["P"] = {str(k): _nombre(v, f"{ou}, {c['ref']}.{k}") for k, v in (c.get("P") or {}).items()}
        if c["type"] == "lineaire":
            _parmi(c.get("orientation", "y"), ("x", "y"), f"{ou}, orientation de {c['ref']}")
            if not 0 < c["longueur"] <= LIMITES["L"]:
                raise ValueError(f"{ou} : longueur de {c['ref']} hors de ]0 ; {C.nb(LIMITES['L'])}] m")
        if c["type"] == "surfacique":
            if not (c["lx"] > 0 and c["ly"] > 0) or math.ceil(c["lx"] / PAS.pas_surfacique) * math.ceil(c["ly"] / PAS.pas_surfacique) > LIMITES["morceaux_stock"]:
                raise ValueError(f"{ou} : stock {c['ref']} de {C.nb(c['lx'])} × {C.nb(c['ly'])} m hors des limites du site")
    if not isinstance(cas["repartiteurs"], dict):
        raise ValueError(f"{ou} : répartiteurs illisibles")
    for ref, rp in cas["repartiteurs"].items():
        if not isinstance(rp, dict) or rp.get("type") not in ("plateau", "filiere"):
            raise ValueError(f"{ou} : répartiteur de {ref} illisible")
        if rp["type"] == "plateau":
            _texte(rp.get("nom"), f"{ou}, plateau de {ref}")
            rp.setdefault("dx", 0.0)
            rp.setdefault("dy", 0.0)
            _nombre(rp["dx"], f"{ou}, décalage de {ref}")
            _nombre(rp["dy"], f"{ou}, décalage de {ref}")
        else:
            _texte(rp.get("id"), f"{ou}, filière de {ref}")
    if not isinstance(cas["filieres"], list):
        raise ValueError(f"{ou} : liste de filières attendue")
    for f in cas["filieres"]:
        if not isinstance(f, dict) or not isinstance(f.get("id"), str) or not isinstance(f.get("nom"), str):
            raise ValueError(f"{ou} : filière illisible")
        for k in ("longueur", "x", "y"):
            _nombre(f.get(k), f"{ou}, filière {f['id']} {k}")
        f.setdefault("orientation", "y")
        _parmi(f["orientation"], ("x", "y"), f"{ou}, orientation de {f['id']}")
    for k in ("cotes", "aires", "taux"):
        cas["options"][k] = bool(cas["options"].get(k, True))
    for k in ("projet", "partie", "indice"):
        cas["identification"][k] = str(cas["identification"].get(k) or ("0" if k == "indice" else ""))
    if cas["selection"] is not None and not isinstance(cas["selection"], str):
        cas["selection"] = None
    s = cas["site"] if isinstance(cas["site"], dict) else _site_cas()
    p = s.get("proposition")
    if p is not None and not (isinstance(p, dict) and isinstance(p.get("empreinte"), str) and isinstance(p.get("trouve"), bool)):
        p = None
    cas["site"] = {"balaye": s.get("balaye") if isinstance(s.get("balaye"), str) else None, "proposition": p}
    return cas


def _relire_site(s) -> Dict:
    """Les brouillons des formulaires : relus s'ils sont lisibles, remis à neuf sinon (ils ne sont pas des données)."""
    d = _site_defaut()
    if not isinstance(s, dict):
        return d
    try:
        fe = _completer(dict(s.get("fiche_engin") or {}), d["fiche_engin"])
        _texte(fe["nom"], "")
        _texte(fe["source"], "")
        _parmi(fe["type_appuis"], [k for k, _ in TYPES_APPUIS], "")
        for k in ("longueur", "largeur", "fleche"):
            _nombre(fe[k], "")
        if not isinstance(fe["appuis"], list) or not fe["appuis"]:
            raise ValueError
        for a in fe["appuis"]:
            _texte(a["nom"], "")
            for k in ("dx", "dy", "a", "b", "P", "phi"):
                _nombre(a[k], "")
        d["fiche_engin"] = fe
    except (ValueError, KeyError, TypeError):
        pass
    try:
        fr_ = _completer(dict(s.get("fiche_rep") or {}), d["fiche_rep"])
        _parmi(fr_["type"], ("plateau", "filiere"), "")
        _parmi(fr_["materiau"], OP.MATERIAUX, "")
        for k in ("nom", "source", "profil"):
            _texte(fr_[k], "")
        for k in ("f", "E", "masse", "A", "B", "t"):
            _nombre(fr_[k], "")
        d["fiche_rep"] = fr_
    except (ValueError, KeyError, TypeError):
        pass
    if isinstance(s.get("nom_fiche_hourdis"), str):
        d["nom_fiche_hourdis"] = s["nom_fiche_hourdis"]
    return d


def normaliser(etat) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab, ou d'un fichier JSON de l'écran d'origine) relu : clés
    manquantes complétées, types contrôlés ; une note illisible lève une erreur claire."""
    if isinstance(etat, dict) and etat.get("module") == "charges_plancher" and "etat" in etat:
        etat = etat["etat"]
    if etat is None or etat == {}:
        return nouvel_etat()
    if not isinstance(etat, dict) or not isinstance(etat.get("cas"), list) or not etat["cas"]:
        raise ValueError("note « Charges sur plancher » illisible : il lui faut une liste de cas")
    e = copy.deepcopy(etat)
    base = {"version": ET.VERSION, "actif": 0, "cas": e["cas"], "fiches": _relire_fiches(e.get("fiches"))}
    actif = e.get("actif", 0)
    base["actif"] = min(max(0, int(actif)), len(base["cas"]) - 1) if isinstance(actif, (int, float)) and not isinstance(actif, bool) else 0
    base["cas"] = [_relire_cas(c, i, base) for i, c in enumerate(base["cas"])]
    base["site"] = _relire_site(e.get("site"))
    return base


def _cas(etat: Dict) -> Dict:
    return ET.cas_actif(etat)


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les cas de la note (l'écran d'origine : sélecteur de cas, « ＋ Cas », « ⧉ », « 🗑️ ») : en ouvrir un, en ajouter
    (C2, C3…), copier l'ouvert, en supprimer un (jamais le dernier)."""
    cs = etat["cas"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(cs) - 1)
    elif action == "ajouter":
        c = ET.nouveau_cas(f"C{len(cs) + 1}")
        c["site"] = _site_cas()
        cs.append(c)
        etat["actif"] = len(cs) - 1
    elif action == "dupliquer":
        c = copy.deepcopy(cs[etat["actif"]])
        c["nom"] = f"{c['nom']} (copie)"
        cs.append(c)
        etat["actif"] = len(cs) - 1
    elif action == "supprimer" and len(cs) > 1:
        i = min(max(0, int(index or 0)), len(cs) - 1)
        del cs[i]
        if etat["actif"] > i or etat["actif"] >= len(cs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
def _S(sorte: str, nom: str = "", unite: str = "", mini=None, strict: bool = False, maxi=None, ech: float = 1.0, arrondi=None,
       options=None) -> Dict:
    """La définition d'un champ : sa sorte (nombre, nul = vide ou 0 → non renseigné, entier, texte, case, choix), son
    domaine (``mini`` inclus, ou exclu si ``strict``), l'échelle d'affichage (mm affichés pour des m stockés)."""
    return {"sorte": sorte, "nom": nom, "unite": unite, "mini": mini, "strict": strict, "maxi": maxi, "ech": ech, "arrondi": arrondi,
            "options": options}


# Bornes HAUTES du site, que l'écran d'origine n'a pas : une saisie absurde (portée de 1 000 m, 10 000 éléments…) ne
# donnerait aucun résultat utile et bloquerait le navigateur (le balayage refait le calcul tous les 10 cm de la portée,
# un stock est découpé en charges de 0,30 m). Larges : aucun plancher réel n'en approche.
LIMITES = {"L": 40.0, "n_elements": 100, "b_element": 5.0, "epaisseur": 2.0, "largeur": 100.0, "gamma": 5.0, "semelle_mm": 5000.0,
           "morceaux_stock": 2500}


def _champs_fixes() -> Dict[Tuple, Dict]:
    c = {("nom",): _S("texte", "Nom du cas"),
         ("plancher", "type"): _S("choix", options=list(TYPES_COURTS.items())),
         ("plancher", "sens"): _S("choix", options=SENS),
         ("plancher", "L"): _S("nombre", "Portée L", "m", mini=0.5, maxi=LIMITES["L"]),
         ("plancher", "q_presentes"): _S("nombre", "Charges présentes", "kN/m²", mini=0.0),
         ("plancher", "hourdis", "n_elements"): _S("entier", "Nombre d'éléments", "", mini=1, maxi=LIMITES["n_elements"]),
         ("plancher", "hourdis", "b_element"): _S("nombre", "Largeur d'élément", "m", mini=0.0, strict=True, maxi=LIMITES["b_element"]),
         ("plancher", "hourdis", "h"): _S("nombre", "Hauteur", "m", mini=0.0, strict=True, maxi=LIMITES["epaisseur"]),
         ("plancher", "hourdis", "g"): _S("nombre", "Poids propre", "kN/m²", mini=0.0),
         ("plancher", "hourdis", "M_Rd"): _S("nombre", "M_Rd", "kN·m", mini=0.0),
         ("plancher", "hourdis", "V_Rd"): _S("nombre", "V_Rd", "kN", mini=0.0),
         ("plancher", "hourdis", "I"): _S("nul", "Inertie I", "m⁴", mini=0.0, strict=True),
         ("plancher", "hourdis", "p_loc_adm"): _S("nul", "Pression admissible", "N/mm²", mini=0.0, strict=True),
         ("plancher", "hourdis", "M_cr"): _S("nul", "M_cr", "kN·m", mini=0.0, strict=True),
         ("plancher", "hourdis", "M_Rd_chape"): _S("nul", "M_Rd avec chape", "kN·m", mini=0.0, strict=True),
         ("plancher", "hourdis", "V_Rd_chape"): _S("nul", "V_Rd avec chape", "kN", mini=0.0, strict=True),
         ("plancher", "hourdis", "nature"): _S("choix", options=NATURES),
         ("plancher", "hourdis", "etat"): _S("choix", options=list(OP.ETATS_H.items())),
         ("plancher", "hourdis", "chape", "oui"): _S("case"),
         ("plancher", "hourdis", "chape", "collaborante"): _S("case"),
         ("plancher", "hourdis", "chape", "epaisseur"): _S("nombre", "Épaisseur de chape", "m", mini=0.0, strict=True, maxi=LIMITES["epaisseur"]),
         ("plancher", "hourdis", "chape", "fck_t"): _S("nombre", "fck(t) de la chape", "MPa", mini=0.0),
         ("plancher", "hourdis", "chape", "fck_exigee"): _S("nul", "Classe exigée", "MPa", mini=0.0, strict=True),
         ("plancher", "hourdis", "source"): _S("texte"),
         ("params", "gamma_G"): _S("nombre", "γG", "", mini=0.0, strict=True, maxi=LIMITES["gamma"]),
         ("params", "gamma_Q"): _S("nombre", "γQ", "", mini=0.0, strict=True, maxi=LIMITES["gamma"]),
         ("params", "part_tranchant_element"): _S("nombre", "Part du tranchant", "", mini=0.0, maxi=1.0),
         ("params", "k_flexion_heft240"): _S("nul", "k (Heft 240)", "", mini=0.0, strict=True),
         ("params", "joint_adm_kN_m"): _S("nul", "Résistance du joint", "kN/m", mini=0.0, strict=True),
         ("params", "joint_longueur_m"): _S("nul", "Longueur de joint", "m", mini=0.0, strict=True),
         ("params", "alpha_cc"): _S("nombre", "αcc", "", mini=0.0, strict=True, maxi=1.0),
         ("options", "cotes"): _S("case"), ("options", "aires"): _S("case"), ("options", "taux"): _S("case")}
    for k in ("projet", "partie", "indice"):
        c[("identification", k)] = _S("texte")
    for t in ("dalle", "predalle"):
        c.update({("plancher", t, "h"): _S("nombre", "Épaisseur h", "m", mini=0.0, strict=True, maxi=LIMITES["epaisseur"]),
                  ("plancher", t, "d"): _S("nombre", "Hauteur utile d", "m", mini=0.0, strict=True, maxi=LIMITES["epaisseur"]),
                  ("plancher", t, "As_inf"): _S("nombre", "As inférieure", "mm²/m", mini=0.0),
                  ("plancher", t, "h1"): _S("nombre", "Revêtement h1", "m", mini=0.0),
                  ("plancher", t, "fyk"): _S("nombre", "fyk", "MPa", mini=0.0, strict=True),
                  ("plancher", t, "largeur"): _S("nombre", "Largeur", "m", mini=0.5, maxi=LIMITES["largeur"]),
                  ("plancher", t, "gamma_s"): _S("nombre", "γs", "", mini=0.0, strict=True),
                  ("plancher", t, "fck_t"): _S("nul", "fck(t)", "MPa", mini=0.0, strict=True),
                  ("plancher", t, "beton"): _S("choix", options=[(k, k) for k in RS.BETON]),
                  ("plancher", t, "armature_transversale_ok"): _S("case")})
    c.update({("plancher", "predalle", "b_predalle"): _S("nombre", "Largeur de prédalle", "m", mini=0.0, strict=True),
              ("plancher", "predalle", "joints_armes"): _S("case"), ("plancher", "predalle", "chape_durcie"): _S("case"),
              ("plancher", "predalle", "portee_etayee"): _S("nul", "Entraxe des étais", "m", mini=0.0, strict=True)})
    return c


CHAMPS = _champs_fixes()
POS = _S("nombre", "Position", "m", arrondi=3)
CHAMPS_CHARGE = {
    "engin": {"x": POS, "y": POS, "angle": _S("choix", options=ANGLES)},
    "ponctuel": {"x": POS, "y": POS, "P": _S("nombre", "Charge P", "kN", mini=0.0),
                 "a": _S("nombre", "Semelle a", "mm", mini=0.0, strict=True, maxi=LIMITES["semelle_mm"], ech=1e3),
                 "b": _S("nombre", "Semelle b", "mm", mini=0.0, strict=True, maxi=LIMITES["semelle_mm"], ech=1e3)},
    "lineaire": {"x": POS, "y": POS, "p": _S("nombre", "Charge p", "kN/m", mini=0.0),
                 "longueur": _S("nombre", "Longueur", "m", mini=0.0, strict=True, maxi=LIMITES["L"]),
                 "orientation": _S("choix", options=ORIENT_CHARGE)},
    "surfacique": {"x0": POS, "y0": POS, "q": _S("nombre", "Charge q", "kN/m²", mini=0.0),
                   "lx": _S("nombre", "Longueur", "m", mini=0.0, strict=True, maxi=LIMITES["L"]),
                   "ly": _S("nombre", "Largeur", "m", mini=0.0, strict=True, maxi=LIMITES["largeur"])},
}
CHAMPS_FIL = {"longueur": _S("nombre", "Longueur de filière", "m", mini=0.1, maxi=LIMITES["largeur"]), "x": POS, "y": POS,
              "orientation": _S("choix", options=ORIENT_FIL)}
CHAMPS_FICHE_ENGIN = {"nom": _S("texte"), "source": _S("texte"), "type_appuis": _S("choix", options=TYPES_APPUIS),
                      "longueur": _S("nombre", "Longueur du châssis", "m", mini=0.0, strict=True),
                      "largeur": _S("nombre", "Largeur du châssis", "m", mini=0.0, strict=True),
                      "fleche": _S("nombre", "Flèche", "m", mini=0.0)}
CHAMPS_APPUI_FICHE = {"nom": _S("texte"), "dx": _S("nombre", "dx", "m"), "dy": _S("nombre", "dy", "m"),
                      "a": _S("nombre", "Empreinte a", "m", mini=0.0, strict=True), "b": _S("nombre", "Empreinte b", "m", mini=0.0, strict=True),
                      "P": _S("nombre", "Charge P", "kN", mini=0.0), "phi": _S("nombre", "φ", "", mini=0.0, strict=True)}
CHAMPS_FICHE_REP = {"nom": _S("texte"), "source": _S("texte"), "type": _S("choix", options=[("plateau", "Plateau"), ("filiere", "Filière")]),
                    "materiau": _S("choix", options=[(m, m) for m in OP.MATERIAUX]),
                    "f": _S("nombre", "f", "MPa", mini=0.0), "E": _S("nombre", "E", "MPa", mini=0.0, strict=True),
                    "masse": _S("nombre", "Masse", "", mini=0.0), "A": _S("nombre", "A ou b", "mm", mini=0.0, strict=True),
                    "B": _S("nombre", "B ou h", "mm", mini=0.0, strict=True), "t": _S("nombre", "Épaisseur t", "mm", mini=0.0, strict=True)}


def _lire(spec: Dict, valeur, actuel):
    """La valeur saisie lue selon son champ ; (valeur, message) — un message dit pourquoi la saisie est refusée."""
    s = spec["sorte"]
    if s == "texte":
        return ("" if valeur is None else str(valeur)), None
    if s == "case":
        return bool(valeur), None
    if s == "choix":
        for k, _ in spec["options"]:
            if str(k) == str(valeur):
                return k, None
        return actuel, f"Choix inconnu : « {valeur} »"
    x = C.lire_nombre(valeur) if not (s == "nul" and (valeur is None or str(valeur).strip() == "")) else 0.0
    if x is None:
        return actuel, f"Nombre illisible : « {valeur} »"
    if s == "nul" and x == 0.0:
        return None, None
    nom, u = spec["nom"], spec["unite"]
    if spec["mini"] is not None and (x < spec["mini"] or (spec["strict"] and x <= spec["mini"])):
        borne = "strictement positif" if spec["strict"] and spec["mini"] == 0 else f"au moins {C.nb(spec['mini'])} {u}".strip()
        return actuel, f"{nom} : {borne} (saisi : {C.nb(x, 6)})"
    if spec["maxi"] is not None and x > spec["maxi"]:
        return actuel, f"{nom} : au plus {C.nb(spec['maxi'])} {u}".strip() + f" (saisi : {C.nb(x, 6)})"
    if s == "entier":
        if x != int(x):
            return actuel, f"{nom} : nombre entier attendu (saisi : {C.nb(x, 6)})"
        return int(x), None
    x = x / spec["ech"]
    if spec["arrondi"] is not None:
        x = round(x, spec["arrondi"])
    return x, None


def _meme_choix(spec: Dict, valeur, actuel) -> bool:
    """Le même choix que l'actuel (l'écran d'origine ne rappelle rien) ; un choix inconnu n'est PAS le même : il sera
    refusé par _ecrire avec son message."""
    v, msg = _lire(spec, valeur, actuel)
    return msg is None and v == actuel


def _ecrire(d: Dict, k, spec: Dict, valeur) -> Optional[str]:
    v, msg = _lire(spec, valeur, d.get(k))
    if msg:
        return msg
    d[k] = v
    return None


def _entier(c):
    return int(c) if isinstance(c, str) and c.lstrip("-").isdigit() and c != "-" else c


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans le cas ouvert (ou dans la note : « note/… »), avec les mêmes effets que l'écran
    d'origine. Rend un message si la saisie est refusée — la note reste alors intacte —, None sinon."""
    avant = copy.deepcopy(etat)
    ch = [_entier(c) for c in chemin]
    try:
        msg = _note(etat, ch[1:], valeur) if ch and ch[0] == "note" else _saisie(etat, _cas(etat), ch, valeur)
    except (ValueError, KeyError, IndexError, TypeError) as e:
        msg = f"Saisie refusée : {e}"
    if msg:
        etat.clear()
        etat.update(avant)
    return msg


def _saisie(etat: Dict, cas: Dict, ch: List, v) -> Optional[str]:
    k = tuple(ch)
    if k == ("plancher", "hourdis", "type_cat"):
        types = ET.types_hourdis(etat)
        if v not in types:
            return f"Type de hourdis inconnu : « {v} »"
        if v != cas["plancher"]["hourdis"].get("type_cat"):
            ET.appliquer_type_hourdis(cas, types[v])
        return None
    if k == ("plancher", "hourdis", "parts_annexe_C"):
        return _parts(cas, "" if v is None else str(v))
    if k in (("plancher", "dalle", "fck_t"), ("plancher", "predalle", "fck_t")):
        x, msg = _lire(CHAMPS[k], v, None)
        if msg:
            return msg
        if x is not None and RS.classe_pour_fck(x) is None:
            mini = min(RS.BETON.items(), key=lambda kv: kv[1]["fck"])
            return f"fck(t) = {C.nb(x)} MPa : sous la plus petite classe ({mini[0]}, fck = {C.nb(mini[1]['fck'])} MPa)"
        cas["plancher"][k[1]]["fck_t"] = x
        return None
    if k in CHAMPS:
        d = cas
        for c in k[:-1]:
            d = d[c]
        actuel = d.get(k[-1])
        if CHAMPS[k]["sorte"] == "choix" and _meme_choix(CHAMPS[k], v, actuel):
            return None                                   # le même choix : rien ne change (l'écran d'origine ne rappelle rien)
        return _ecrire(d, k[-1], CHAMPS[k], v) or _coherence(cas, k)
    if k == ("selection",):
        refs = [f"rep:{r}" for r in _refs(etat, cas)]
        if v not in refs:
            return f"Appui inconnu : « {v} »"
        if v != _appui_coupe(cas, [r[4:] for r in refs]):          # l'appui déjà montré : rien ne change
            cas["selection"] = v
        return None
    if k == ("site", "balaye"):
        if v not in [c["ref"] for c in cas["charges"] if c["type"] == "engin"]:
            return f"Engin inconnu : « {v} »"
        if v != _balaye(cas):                                       # l'engin déjà balayé : rien ne change
            cas["site"]["balaye"] = v
        return None
    if ch[0] == "charges":
        return _saisie_charge(etat, cas, ch[1:], v)
    if ch[0] == "appuis":
        return _saisie_appui(etat, cas, ch[1], ch[2], v)
    if ch[0] == "filieres":
        return _saisie_filiere(etat, cas, ch[1], ch[2:], v)
    if k == ("proposition", "+"):
        r = _calculer(etat, cas)
        if "res" not in r:
            return "Proposition impossible : " + r.get("erreur", "calcul refusé")
        prop = OP.proposer_filiere(etat, cas, r["res"], r["plancher"], r["index"])
        prop["empreinte"] = _empreinte(cas)
        cas["site"]["proposition"] = prop
        return None
    if k == ("proposition", "appliquer"):
        prop = cas["site"].get("proposition")
        if not prop or prop.get("empreinte") != _empreinte(cas) or not prop.get("trouve"):
            return "Aucune proposition à appliquer (le cas a changé depuis : relancer la proposition)"
        _ids_libres(cas, prop)
        OP.appliquer_proposition(cas, prop)
        cas["site"]["proposition"] = None
        return None
    return f"Champ inconnu : {'/'.join(map(str, ch))}"


def _parts(cas: Dict, s: str) -> Optional[str]:
    """« Parts annexe C » : lues comme l'écran d'origine les lit ; un morceau qu'il ignorait en silence (la virgule
    décimale « 0:0,5 » coupait le couple), un nombre illisible ou négatif sont REFUSÉS."""
    ignores = OP.morceaux_ignores(s)
    if ignores:
        return (f"Parts annexe C : « {', '.join(ignores)} » sans « : » — écrire « 0:0.5 ; 1:0.25 » (couples séparés par "
                f"« ; », décimales au point)")
    try:
        parts = OP.lire_parts_annexe_c(s)
    except ValueError:
        return f"Parts annexe C illisibles : « {s} » — écrire « 0:0.5 ; 1:0.25 »"
    if parts and any(k < 0 or p < 0 for k, p in parts.items()):
        return "Parts annexe C : rangs et parts positifs ou nuls"
    cas["plancher"]["hourdis"]["parts_annexe_C"] = {str(k): p for k, p in parts.items()} if parts else None
    return None


def _canon(x):
    """Les nombres en flottants : le navigateur écrit 45.0 « 45 » (JSON.stringify), l'empreinte ne doit pas le voir."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    return [_canon(v) for v in x]


def _empreinte(cas: Dict) -> str:
    """L'empreinte de ce qui fonde une proposition : une proposition faite sur un autre état n'est plus montrée."""
    return json.dumps(_canon({k: cas[k] for k in ("plancher", "charges", "repartiteurs", "filieres", "params")}), sort_keys=True,
                      ensure_ascii=False)


def _ids_libres(cas: Dict, prop: Dict) -> None:
    """Défaut de l'écran d'origine : une deuxième proposition reprend l'identifiant « F1p0 » d'une filière déjà
    appliquée (ref_libre ne connaît que « F1 ») et crée une filière fantôme. Le site donne un identifiant libre."""
    pris = {f["id"] for f in cas.get("filieres", [])} | {c["ref"] for c in cas["charges"]}
    for f in prop["filieres"]:
        if f["id"] in pris:
            n, neuf = 1, f["id"]
            while neuf in pris:
                n += 1
                neuf = f"F{n}p{f['id'].split('p')[-1]}"
            for rp in prop["repartiteurs"].values():
                if rp.get("id") == f["id"]:
                    rp["id"] = neuf
            f["id"] = neuf
        pris.add(f["id"])


def _saisie_charge(etat: Dict, cas: Dict, ch: List, v) -> Optional[str]:
    if ch == ["+"]:
        if v not in ("ponctuel", "lineaire", "surfacique"):
            return f"Charge inconnue : « {v} »"
        ET.ajouter_charge(etat, cas, v)
        return None
    if ch == ["+engin"]:
        if v not in ET.gabarits(etat):
            return f"Engin inconnu : « {v} »"
        ET.ajouter_engin(etat, cas, v)
        return None
    c = cas["charges"][ch[0]]
    if ch[1] == "-":
        ET._action_charge(cas, c, "Supprimer")
        return None
    if ch[1] == "dupliquer":
        ET._action_charge(cas, c, "Dupliquer")
        return None
    if ch[1] == "P" and c["type"] == "engin":
        gab = ET.gabarits(etat)[c["gabarit"]]
        nom = gab["appuis"][ch[2]]["nom"]
        return _ecrire(c.setdefault("P", {}), nom, _S("nombre", f"Charge de {c['ref']}.{nom}", "kN", mini=0.0), v)
    spec = CHAMPS_CHARGE[c["type"]].get(ch[1])
    if spec is None:
        return f"Champ inconnu : {ch[1]}"
    if spec["sorte"] == "choix" and _meme_choix(spec, v, c.get(ch[1])):
        return None
    msg = _ecrire(c, ch[1], spec, v)
    if msg is None and c["type"] == "surfacique":
        n = math.ceil(c["lx"] / PAS.pas_surfacique - 1e-9) * math.ceil(c["ly"] / PAS.pas_surfacique - 1e-9)
        if n > LIMITES["morceaux_stock"]:
            msg = (f"Stock de {C.nb(c['lx'])} × {C.nb(c['ly'])} m : {n} charges de {C.nb(PAS.pas_surfacique)} m pour le moteur, au plus "
                   f"{LIMITES['morceaux_stock']} — une charge répartie sur tout le plancher se saisit en « charges présentes » (section 1)")
    return msg


def _coherence(cas: Dict, k: Tuple) -> Optional[str]:
    """Ce qu'une saisie ne peut pas rendre incohérent : la hauteur utile d'une dalle reste sous son épaisseur (l'écran
    d'origine acceptait d ≥ h, et le moment admissible grandissait avec d : non conservatif)."""
    if len(k) == 3 and k[:2] in (("plancher", "dalle"), ("plancher", "predalle")) and k[2] in ("h", "d"):
        dl = cas["plancher"][k[1]]
        if not dl["d"] < dl["h"]:
            return f"Hauteur utile d = {C.nb(dl['d'])} m : elle doit rester sous l'épaisseur h = {C.nb(dl['h'])} m"
    return None


def _refs(etat: Dict, cas: Dict) -> List[str]:
    """Les appuis du cas (engins et étançons), dans l'ordre du moteur ; [] si le cas ne se construit pas."""
    r = _calculer(etat, cas, verifier=False)
    return list(r["index"]["empreintes"]) if "index" in r else []


def _choix_repartiteur(cas: Dict, ref: str) -> str:
    rp = cas["repartiteurs"].get(ref)
    if not rp:
        return "aucun"
    return f"plateau:{rp['nom']}" if rp["type"] == "plateau" else f"filiere:{rp['id']}"


def _saisie_appui(etat: Dict, cas: Dict, i: int, champ: str, v) -> Optional[str]:
    refs = _refs(etat, cas)
    ref = refs[i]
    if champ == "repartiteur":
        v = str(v)
        if v == _choix_repartiteur(cas, ref):
            return None
        dispo = ET.repartiteurs_disponibles(etat)
        if v == "aucun":
            ET.regler_repartiteur(etat, cas, ref, "type", "aucun")
        elif v.startswith("plateau:"):
            nom = v[len("plateau:"):]
            if dispo.get(nom, {}).get("type") != "plateau":
                return f"Plateau inconnu : « {nom} »"
            if (cas["repartiteurs"].get(ref) or {}).get("type") != "plateau":
                ET.regler_repartiteur(etat, cas, ref, "type", "plateau")
            ET.regler_repartiteur(etat, cas, ref, "element", nom)
        elif v == "filiere:nouvelle":
            ET.regler_repartiteur(etat, cas, ref, "type", "filiere")
        elif v.startswith("filiere:"):
            fid = v[len("filiere:"):]
            if fid not in [f["id"] for f in cas.get("filieres", [])]:
                return f"Filière inconnue : « {fid} »"
            cas["repartiteurs"][ref] = {"type": "filiere", "id": fid}
        else:
            return f"Répartiteur inconnu : « {v} »"
        return None
    if champ == "idem":
        ET.regler_repartiteur(etat, cas, ref, "idem", "oui")
        return None
    if champ in ("dx", "dy"):
        rp = cas["repartiteurs"].get(ref)
        if not rp or rp["type"] != "plateau":
            return f"Pas de plateau sous {ref}"
        return _ecrire(rp, champ, _S("nombre", "Décalage", "m"), v)
    return f"Champ inconnu : {champ}"


def _saisie_filiere(etat: Dict, cas: Dict, i: int, ch: List, v) -> Optional[str]:
    f = cas["filieres"][i]
    if ch == ["-"]:
        ET.supprimer_filiere(cas, f["id"])
        return None
    if ch[0] == "nom":
        dispo = ET.repartiteurs_disponibles(etat)
        if dispo.get(v, {}).get("type") != "filiere":
            return f"Élément de filière inconnu : « {v} »"
        f["nom"] = v
        return None
    if ch[0] == "porte":
        ref = _refs(etat, cas)[ch[1]]
        lie = (cas["repartiteurs"].get(ref) or {}).get("id") == f["id"]
        if bool(v) and not lie:
            cas["repartiteurs"][ref] = {"type": "filiere", "id": f["id"]}
        elif not bool(v) and lie:
            cas["repartiteurs"].pop(ref, None)
        return None
    spec = CHAMPS_FIL.get(ch[0])
    if spec is None:
        return f"Champ inconnu : {ch[0]}"
    if spec["sorte"] == "choix" and _meme_choix(spec, v, f.get(ch[0])):
        return None
    return _ecrire(f, ch[0], spec, v)


def _note(etat: Dict, ch: List, v) -> Optional[str]:
    """Les saisies de la NOTE : brouillons des fiches, leur enregistrement, un fichier de l'écran d'origine."""
    s = etat["site"]
    cas = _cas(etat)
    if ch == ["nom_fiche_hourdis"]:
        s["nom_fiche_hourdis"] = "" if v is None else str(v)
        return None
    if ch == ["fiche_hourdis", "enregistrer"]:
        nom = s["nom_fiche_hourdis"].strip()
        if not nom:
            return "Donner un nom à la fiche"
        ET.ajouter_fiche_hourdis(etat, cas, nom)
        s["nom_fiche_hourdis"] = ""
        return None
    if ch == ["ouvrir"]:
        if not isinstance(v, dict) or "texte" not in v:
            return "Fichier illisible"
        try:
            neuf = normaliser(ET.charger(json.loads(str(v["texte"]).lstrip("﻿"))))
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            return f"Fichier refusé : {e}"
        etat.clear()
        etat.update(neuf)
        return None
    if ch[0] == "fiche_engin":
        fe = s["fiche_engin"]
        if ch[1:] == ["enregistrer"]:
            return _enregistrer_engin(etat, fe)
        if ch[1] == "appuis":
            if ch[2:] == ["+"]:
                fe["appuis"].append({"nom": f"P{len(fe['appuis']) + 1}", "dx": 0.0, "dy": 0.0, "a": 0.3, "b": 0.3, "P": 0.0, "phi": 1.0})
                return None
            if ch[3:] == ["-"]:
                if len(fe["appuis"]) <= 1:
                    return "Il faut au moins un appui"
                del fe["appuis"][ch[2]]
                return None
            return _ecrire(fe["appuis"][ch[2]], ch[3], CHAMPS_APPUI_FICHE[ch[3]], v)
        spec = CHAMPS_FICHE_ENGIN[ch[1]]
        if spec["sorte"] == "choix" and _meme_choix(spec, v, fe[ch[1]]):
            return None
        return _ecrire(fe, ch[1], spec, v)
    if ch[0] == "fiche_rep":
        fr_ = s["fiche_rep"]
        if ch[1:] == ["enregistrer"]:
            return _enregistrer_rep(etat, fr_)
        if ch[1] == "profil":
            if v != OP.RECTANGLE and v not in BIB.profils():
                return f"Profilé inconnu : « {v} »"
            fr_["profil"] = v
            return None
        spec = CHAMPS_FICHE_REP[ch[1]]
        if spec["sorte"] == "choix" and _meme_choix(spec, v, fr_[ch[1]]):
            return None
        return _ecrire(fr_, ch[1], spec, v)
    return f"Champ inconnu : note/{'/'.join(map(str, ch))}"


def _enregistrer_engin(etat: Dict, fe: Dict) -> Optional[str]:
    noms = [a["nom"].strip() for a in fe["appuis"]]
    if not fe["nom"].strip():
        return "Un nom est nécessaire."
    if any(not n for n in noms) or len(set(noms)) != len(noms):
        return "Chaque appui a un nom, différent des autres"
    fiche = OP.fiche_engin(fe["nom"], fe["type_appuis"], fe["longueur"], fe["largeur"], fe["fleche"], fe["appuis"], fe["source"])
    etat["fiches"]["engins"].append(fiche)
    etat["site"]["fiche_engin"] = _site_defaut()["fiche_engin"]
    return None


def _enregistrer_rep(etat: Dict, fr_: Dict) -> Optional[str]:
    if not fr_["nom"].strip():
        return "Un nom est nécessaire."
    nom = fr_["nom"].strip()
    livres = {k for k, d in ET.repartiteurs_disponibles(etat).items() if d.get("origine") != "fiche"}
    if nom in livres:
        return f"« {nom} » est déjà un élément de la bibliothèque livrée : choisir un autre nom"
    autre = next((d for d in etat["fiches"]["repartiteurs"] if d["nom"] == nom and d.get("type") != fr_["type"]), None)
    if autre:
        return f"« {nom} » est déjà une fiche d'un autre type : choisir un autre nom"
    fiche = OP.fiche_repartiteur(fr_["type"], fr_["nom"], fr_["materiau"], fr_["f"], fr_["E"], fr_["masse"], fr_["source"],
                                 fr_["A"], fr_["B"], fr_["t"], fr_["profil"])
    etat["fiches"]["repartiteurs"].append(fiche)
    etat["site"]["fiche_rep"] = dict(OP.FICHE_REP_DEFAUT, type=fr_["type"])
    return None


# ------------------------------------------------------------------ le calcul
def _calculer(etat: Dict, cas: Dict, verifier: bool = True) -> Dict:
    """Le moteur sur un cas : (plancher, charges, répartiteurs, paramètres, index) d'``etat.construire`` puis
    ``verification.verifier`` — exactement l'enchaînement de l'écran d'origine. Une donnée qui empêche la construction
    (l'écran d'origine s'arrêtait sur « Donnée manquante ») rend {"erreur": …}."""
    try:
        plancher, charges, reps, params, index = ET.construire(etat, cas)
    except Exception as e:  # noqa: BLE001 — une note qui ne se construit pas se DIT, l'écran reste utilisable
        return {"erreur": f"Donnée manquante : {e}"}
    out = {"plancher": plancher, "charges": charges, "reps": reps, "params": params, "index": index}
    if verifier:
        try:
            out["res"] = V.verifier(plancher, charges, reps, params)
        except Exception as e:  # noqa: BLE001 — l'écran d'origine s'arrêtait sur l'exception
            out["erreur"] = f"Calcul impossible : {e}"
    return out


def _taux(res: Optional[Dict]) -> Optional[float]:
    if not res or res.get("statut") in (None, "refus"):
        return None
    return res.get("taux_max")


def _niveau(res: Dict) -> str:
    """La couleur d'un cas : celle du taux (C.niveau), et rouge dès que le moteur le déclare non conforme."""
    if not res or "statut" not in res:
        return "x"
    if res["statut"] == "refus":
        return "c"
    return "r" if res["statut"] == "nok" else C.niveau(res["taux_max"])


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : son cas le plus défavorable (un cas non conforme d'abord, puis le plus chargé)."""
    vals = [((r["statut"] == "nok"), r["taux_max"] or 0.0, i) for i, r in enumerate(resultats) if _taux(r) is not None]
    if not vals:
        refus = any((r or {}).get("statut") == "refus" for r in resultats)
        return {"taux": None, "niveau": "c" if refus else "x", "statut": "À compléter" if refus else "Calcul impossible", "gouvernant": ""}
    _, t, i = max(vals)
    r = resultats[i]
    niv = _niveau(r)
    g = r.get("gouverne") or {}
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['cas'][i]['nom']} · {g.get('famille', '').capitalize()} · {_txt(g.get('objet', ''))}".rstrip(" ·")}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """« Dans cette note » : chaque cas avec son plancher et son taux ; un clic l'ouvre."""
    items = ""
    for i, (c, r) in enumerate(zip(etat["cas"], resultats)):
        t = _taux(r)
        niv = _niveau(r) if t is not None else ("c" if (r or {}).get("statut") == "refus" else "x")
        txt = f"{C.pc(C.pct(t))}{C.NBSP}%" if t is not None else "—"
        pl = c["plancher"]
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer ce cas">{C.icon("x")}</button>' if len(etat["cas"]) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(c["nom"] or f"Cas {i + 1}")} '
                  f'<span>· {TYPES_COURTS[pl["type"]]} · {C.fr(pl["L"], 2)} m</span></span><span class="pct {niv}">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Cas de cette note · {len(etat["cas"])}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier le cas ouvert">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter un cas</div></div>')


# ------------------------------------------------------------------ le cadre Données
def _court(nom: str, n: int = 24) -> str:
    """Un nom de fiche raccourci pour une ligne étroite : « Tôle acier S235 · 600 × 600 × 20 » → « Tôle 600 × 600 × 20 »."""
    morceaux = [m.strip() for m in str(nom).split(" · ")]
    if len(morceaux) > 1 and any(ch.isdigit() for ch in morceaux[1]):
        s = f"{morceaux[0].split()[0]} {morceaux[1]}"
    else:
        s = morceaux[0]
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def _num(chemin, valeur, unite: str = "", dec: int = 6, **k) -> str:
    return C.num(chemin, valeur, unite, dec=dec, **k)


def _bouton(chemin: Sequence, texte: str, icone: str = "plus", valeur: str = "1") -> str:
    """Un bouton d'action du cadre Données (ajouter, poser, enregistrer…) : il envoie ``valeur`` au chemin."""
    return (f'<button type="button" class="addrow" data-chemin="{C._attr_chemin(chemin)}" data-valeur="{C.esc(valeur)}">'
            f'{C.icon(icone)}{texte}</button>')


def _mini_bouton(chemin: Sequence, icone: str, titre: str) -> str:
    return (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1" title="{C.esc(titre)}">'
            f'{C.icon(icone)}</button>')


def _trio(*cellules: Tuple[str, str]) -> str:
    """Deux ou trois champs côte à côte, chacun avec son petit libellé au-dessus."""
    return '<div class="trio">' + "".join(f"<label><span>{lab}</span>{champ}</label>" for lab, champ in cellules) + "</div>"


def _sec_plancher(cas: Dict) -> str:
    pl = cas["plancher"]
    L = C.ligne
    return (L("Nom", C.texte(["nom"], cas["nom"], "ex. C1 grue en rive"), full=True)
            + L("Plancher", C.segments(["plancher", "type"], pl["type"], [("hourdis", "Hourdis"), ("dalle", "Dalle"), ("predalle", "Prédalle")]), full=True)
            + L("Appuis A–B", C.segments(["plancher", "sens"], pl["sens"], SENS), full=True)
            + L("portée", _num(["plancher", "L"], pl["L"], "m", mini=0.5), "L", "Entre les appuis A et B ; deux appuis simples, sans continuité.")
            + L("présentes", _num(["plancher", "q_presentes"], pl["q_presentes"], "kN/m²", mini=0), "q",
                "Charges présentes : stocks, personnel… (EN 1991-1-6 §4.11), en plus du poids propre."))


def _sec_hourdis(etat: Dict, cas: Dict) -> str:
    h = cas["plancher"]["hourdis"]
    L = C.ligne
    types = ET.types_hourdis(etat)
    cur = h.get("type_cat")
    opts = [(n, n) for n in types] + ([] if cur in types else [(cur, cur)])
    s = (L("Type", C.choix(["plancher", "hourdis", "type_cat"], cur, opts), full=True)
         + C.aide("Le catalogue ou une fiche de ce fichier : ses valeurs se recopient ci-dessous et restent modifiables (P-1).")
         + '<div class="grp">Éléments</div>'
         + L("nombre d'éléments", _num(["plancher", "hourdis", "n_elements"], h["n_elements"], "", entier=True, mini=1), "n")
         + L("largeur d'élément", _num(["plancher", "hourdis", "b_element"], h["b_element"], "m"), "b")
         + L("hauteur", _num(["plancher", "hourdis", "h"], h["h"], "m"), "h")
         + L("poids propre", _num(["plancher", "hourdis", "g"], h["g"], "kN/m²"), "g")
         + L("État", C.segments(["plancher", "hourdis", "etat"], h["etat"], list(OP.ETATS_H.items())), full=True)
         + '<div class="grp">Résistances par élément</div>'
         + L("Nature", C.segments(["plancher", "hourdis", "nature"], h["nature"], NATURES), full=True)
         + C.aide("Rd : valeurs de CALCUL du catalogue, efforts à l'ELU γ<sub>G</sub>·G + γ<sub>Q</sub>·Q ; service : valeurs "
                  "ADMISSIBLES, coefficients 1 (P-1, P-2).")
         + L("moment résistant", _num(["plancher", "hourdis", "M_Rd"], h["M_Rd"], "kN·m", mini=0), "M<sub>Rd</sub>")
         + L("tranchant résistant", _num(["plancher", "hourdis", "V_Rd"], h["V_Rd"], "kN", mini=0), "V<sub>Rd</sub>"))
    if not (h["M_Rd"] or 0) > 0 or not (h["V_Rd"] or 0) > 0:
        s += C.aide("<b>À renseigner</b> depuis la fiche du fabricant : sans M<sub>Rd</sub> et V<sub>Rd</sub>, aucune vérification.")
    s += (L("inertie", _num(["plancher", "hourdis", "I"], h["I"], "m⁴", dec=10), "I",
            "Inertie du fabricant ; vide : estimation nommée 0,6·b·h³/12, raideur relative sous une filière seulement.")
          + L("pression locale", _num(["plancher", "hourdis", "p_loc_adm"], h["p_loc_adm"], "N/mm²"), "p<sub>adm</sub>",
              "Pression locale admissible sur la membrure, du fabricant (P-5) ; vide : non renseignée.")
          + L("fissuration", _num(["plancher", "hourdis", "M_cr"], h["M_cr"], "kN·m"), "M<sub>cr</sub>",
              "Moment de fissuration du fabricant (P-11) : ligne d'information ; vide : non renseigné.")
          + '<div class="grp">Couche de compression</div>'
          + C.case(["plancher", "hourdis", "chape", "oui"], h["chape"]["oui"], "Couche de compression"))
    if h["chape"]["oui"]:
        ch = h["chape"]
        s += (L("épaisseur", _num(["plancher", "hourdis", "chape", "epaisseur"], ch["epaisseur"], "m"), "e")
              + L("résistance à l'âge", _num(["plancher", "hourdis", "chape", "fck_t"], ch["fck_t"], "MPa"), "f<sub>ck</sub>(t)",
                  "Résistance du béton de la chape à l'âge du calcul (module Âge du béton).")
              + C.case(["plancher", "hourdis", "chape", "collaborante"], ch["collaborante"], "Chape collaborante (surface rugueuse, joints remplis)")
              + L("classe exigée", _num(["plancher", "hourdis", "chape", "fck_exigee"], ch["fck_exigee"], "MPa"), "f<sub>ck,ex</sub>",
                  "Classe que le fabricant exige pour ses valeurs avec chape ; vide : inconnue → valeurs nues (P-17).")
              + L("M<sub>Rd</sub> avec chape", _num(["plancher", "hourdis", "M_Rd_chape"], h["M_Rd_chape"], "kN·m"), "")
              + L("V<sub>Rd</sub> avec chape", _num(["plancher", "hourdis", "V_Rd_chape"], h["V_Rd_chape"], "kN"), ""))
    actif = h["etat"] in ("clavetes", "chapes") and bool(h["parts_annexe_C"])
    s += ('<div class="grp">Répartition entre éléments</div>'
          + L("Parts C.1", C.texte(["plancher", "hourdis", "parts_annexe_C"], OP.texte_parts_annexe_c(h["parts_annexe_C"]), "0:0.5 ; 1:0.25"),
              full=True)
          + C.aide("Figure C.1 de l'EN 1168 transcrite par le bureau : « 0 » l'élément chargé, « 1 » chaque premier voisin… ; vide : "
                   "l'élément chargé porte tout. " + ("Répartition <b>active</b>." if actif else
                                                       ("Hourdis posés : la répartition ne s'applique pas." if h["parts_annexe_C"] else "")))
          + L("Source", C.texte(["plancher", "hourdis", "source"], h["source"], "fiche, date, page"), full=True)
          + '<div class="grp">Garder comme fiche</div>'
          + L("Nom", C.texte(["note", "nom_fiche_hourdis"], etat["site"]["nom_fiche_hourdis"], "ex. SP 200 · fiche 2026"), full=True)
          + _bouton(["note", "fiche_hourdis", "enregistrer"], "Garder ces valeurs comme fiche du catalogue", "file-plus-2"))
    return s


def _sec_dalle(cas: Dict) -> str:
    t = cas["plancher"]["type"]
    d = cas["plancher"][t]
    L = C.ligne
    b = ["plancher", t]
    s = (L("Béton", C.choix(b + ["beton"], d["beton"], [(k, k) for k in RS.BETON]), full=True)
         + L("épaisseur", _num(b + ["h"], d["h"], "m"), "h")
         + L("hauteur utile", _num(b + ["d"], d["d"], "m"), "d")
         + L("armatures", _num(b + ["As_inf"], d["As_inf"], "mm²/m", mini=0), "A<sub>s</sub>", "Dans le sens porteur.")
         + L("limite élastique", _num(b + ["fyk"], d["fyk"], "MPa"), "f<sub>yk</sub>", "μ de B500 sauf f_yk = 400 (μ de B400).")
         + L("coefficient acier", _num(b + ["gamma_s"], d["gamma_s"], ""), "γ<sub>s</sub>")
         + L("résistance à l'âge", _num(b + ["fck_t"], d["fck_t"], "MPa"), "f<sub>ck</sub>(t)",
             "Résistance à l'âge du calcul : la classe retenue est arrondie vers le bas ; vide : la classe saisie.")
         + L("revêtement", _num(b + ["h1"], d["h1"], "m", mini=0), "h<sub>1</sub>", "Revêtement diffusant au-dessus de la dalle.")
         + L("largeur du plancher", _num(b + ["largeur"], d["largeur"], "m", mini=0.5), "B")
         + C.case(b + ["armature_transversale_ok"], d["armature_transversale_ok"], "Armature transversale ≥ 20 % (largeur efficace)"))
    if t == "predalle":
        s += ('<div class="grp">Prédalle</div>'
              + L("largeur de prédalle", _num(b + ["b_predalle"], d["b_predalle"], "m"), "b<sub>p</sub>")
              + C.case(b + ["joints_armes"], d["joints_armes"], "Joints armés (treillis de chape continu)")
              + C.case(b + ["chape_durcie"], d["chape_durcie"], "Chape durcie (sinon : ni engin ni étai, P-7)")
              + L("entraxe des étais", _num(b + ["portee_etayee"], d["portee_etayee"], "m"), "L<sub>ét</sub>",
                  "Étais en place : portée réduite, report sur l'étage inférieur non modélisé (P-8) ; vide : sans étais."))
    return s


def _options_repartiteur(etat: Dict, cas: Dict, ref: str) -> List[Tuple[str, str]]:
    dispo = ET.repartiteurs_disponibles(etat)
    opts = [("aucun", SANS)] + [(f"plateau:{n}", _court(n, 30).replace(" × ", "×")) for n, d in dispo.items() if d.get("type") == "plateau"]
    opts += [(f"filiere:{f['id']}", f"{f['id']} · {f['nom']}") for f in cas.get("filieres", [])]
    opts.append(("filiere:nouvelle", "+ filière HEB 140"))
    cur = _choix_repartiteur(cas, ref)
    if cur not in [k for k, _ in opts]:
        opts.insert(1, (cur, cur.split(":", 1)[1]))
    return opts


def _ligne_appui(etat: Dict, cas: Dict, k: int, ref: str, nom: str, champ_P: str, dyn: str) -> str:
    """Un appui : son nom, sa charge, son répartiteur (et le décalage du plateau), « idem » pour un engin."""
    rp = cas["repartiteurs"].get(ref)
    idem = _mini_bouton(["appuis", k, "idem"], "copy", "Même répartiteur sous tous les appuis de l'engin") if "." in ref and not ref.startswith(("T", "F", "S")) else '<span class="moins"></span>'
    out = (f'<div class="lit"><span title="{C.esc(ref)}">{C.esc(nom)}</span>{champ_P}'
           f'{C.choix(["appuis", k, "repartiteur"], _choix_repartiteur(cas, ref), _options_repartiteur(etat, cas, ref))}{idem}</div>')
    if dyn:
        out += f'<div class="hint chp-dyn">{dyn}</div>'
    if rp and rp["type"] == "plateau":
        out += (f'<div class="lit chp-dec" title="Décalage du plateau par rapport à l\'appui"><span>Δx</span>{_num(["appuis", k, "dx"], rp.get("dx", 0.0), "m")}'
                f'<span>Δy</span>{_num(["appuis", k, "dy"], rp.get("dy", 0.0), "m")}</div>')
    return out


def _sec_charges(etat: Dict, cas: Dict, index: Optional[Dict]) -> str:
    gabs = ET.gabarits(etat)
    refs = list(index["empreintes"]) if index else []
    cartes = ""
    for i, c in enumerate(cas["charges"]):
        t = c["type"]
        actions = (_mini_bouton(["charges", i, "dupliquer"], "copy", "Copier (décalée de 1 m)")
                   + _mini_bouton(["charges", i, "-"], "x", "Retirer cette charge"))
        if t == "engin":
            g = gabs[c["gabarit"]]
            tot = sum(float((c.get("P") or {}).get(a["nom"], a.get("P", 0.0)) or 0.0) for a in g["appuis"])
            titre = C.esc(_court(g["nom"].split(" · ")[0].split(" (")[0], 24)) + f' <small class="chp-tot">· {C.fr(tot, 0)}{C.NBSP}kN</small>'
            corps = _trio(("x", _num(["charges", i, "x"], c["x"], "m", dec=3)), ("y", _num(["charges", i, "y"], c["y"], "m", dec=3)),
                          ("rotation", C.choix(["charges", i, "angle"], int(c.get("angle", 0)), ANGLES)))
            corps += '<div class="chp-tete"><span></span><span>charge</span><span>répartiteur</span></div>'
            for j, ap in enumerate(g["appuis"]):
                ref = f"{c['ref']}.{ap['nom']}"
                P = float((c.get("P") or {}).get(ap["nom"], ap.get("P", 0.0)) or 0.0)
                phi = float(ap.get("phi", 1.0))
                dyn = f"φ = {C.nb(phi)} : P·φ = {C.fr(P * phi, 1)} kN" if phi != 1.0 else ""
                if ref in refs:
                    corps += _ligne_appui(etat, cas, refs.index(ref), ref, ap["nom"], _num(["charges", i, "P", j], P, "kN"), dyn)
                else:
                    corps += f'<div class="lit"><span>{C.esc(ap["nom"])}</span>{_num(["charges", i, "P", j], P, "kN")}<span></span><span></span></div>'
            src = (g.get("source") or "").strip()
            if any(not ((c.get("P") or {}).get(a["nom"], a.get("P", 0.0)) or 0) > 0 for a in g["appuis"]):
                corps += C.aide("<b>Charge par appui à saisir</b> : tableau de stabilité du constructeur, pour l'opération prévue (P-13).")
            if src:
                corps += f'<div class="hint chp-src-eng" title="{C.esc(src)}">{C.esc(src)}</div>'
        elif t == "ponctuel":
            titre = "Étançon"
            corps = _trio(("x", _num(["charges", i, "x"], c["x"], "m", dec=3)), ("y", _num(["charges", i, "y"], c["y"], "m", dec=3)),
                          ("charge P", _num(["charges", i, "P"], c["P"], "kN")))
            corps += _trio(("semelle a", _num(["charges", i, "a"], c["a"] * 1e3, "mm")), ("semelle b", _num(["charges", i, "b"], c["b"] * 1e3, "mm")))
            if c["ref"] in refs:
                corps += ('<div class="chp-tete"><span></span><span></span><span>répartiteur</span></div>'
                          + _ligne_appui(etat, cas, refs.index(c["ref"]), c["ref"], c["ref"], "<span></span>", ""))
        elif t == "lineaire":
            titre = "File d'étais"
            corps = _trio(("x (centre)", _num(["charges", i, "x"], c["x"], "m", dec=3)), ("y (centre)", _num(["charges", i, "y"], c["y"], "m", dec=3)),
                          ("charge p", _num(["charges", i, "p"], c["p"], "kN/m")))
            corps += _trio(("longueur", _num(["charges", i, "longueur"], c["longueur"], "m")),
                           ("orientation", C.choix(["charges", i, "orientation"], c.get("orientation", "y"), ORIENT_CHARGE)))
            corps += C.aide(f"Largeur de contact {C.fr(c['largeur'] * 1e3, 0)} mm ; découpée en charges de 0,25 m au plus.")
        else:
            titre = "Stock"
            corps = _trio(("x (coin)", _num(["charges", i, "x0"], c["x0"], "m", dec=3)), ("y (coin)", _num(["charges", i, "y0"], c["y0"], "m", dec=3)),
                          ("charge q", _num(["charges", i, "q"], c["q"], "kN/m²")))
            corps += _trio(("longueur (x)", _num(["charges", i, "lx"], c["lx"], "m")), ("largeur (y)", _num(["charges", i, "ly"], c["ly"], "m")))
        cartes += (f'<div class="secard"><div class="sh2"><span class="indice">{C.esc(c["ref"])}</span><span class="chp-tt">{titre}</span>'
                   f'<span class="chp-act">{actions}</span></div>{corps}</div>')
    vide = C.aide("Aucune charge : poser un engin depuis la bibliothèque (section 5), ou un étançon, une file d'étais, un stock.") if not cas["charges"] else ""
    return (vide + f'<div class="secards">{cartes}</div>'
            + '<div class="chp-ajouts">' + _bouton(["charges", "+"], "Étançon", valeur="ponctuel")
            + _bouton(["charges", "+"], "File d'étais", valeur="lineaire") + _bouton(["charges", "+"], "Stock", valeur="surfacique") + "</div>"
            + C.aide("Positions en m dans le repère du plan : x le long de la portée depuis l'appui A, y en travers depuis le bord haut. "
                     "Charges en valeurs de service."))


def _sec_repartiteurs(etat: Dict, cas: Dict, calc: Dict) -> str:
    L = C.ligne
    index = calc.get("index")
    refs = list(index["empreintes"]) if index else []
    res = calc.get("res") or {}
    s = ""
    prop = cas["site"].get("proposition")
    frais = prop if prop and prop.get("empreinte") == _empreinte(cas) else None
    if res.get("statut") == "nok" or frais:
        s += _bouton(["proposition", "+"], "Proposer une filière sous les appuis en défaut", "sparkles")
        if frais:
            s += C.aide(C.esc(OP.phrase_proposition(frais)))
            if frais["trouve"]:
                s += _bouton(["proposition", "appliquer"], "Appliquer la proposition", "check")
    if not refs:
        s += C.aide("Posez d'abord un engin ou un étançon (section 3) : les plateaux et les filières se placent sous leurs appuis.")
    else:
        s += C.aide("Le plateau ou la filière de chaque appui se choisit dans sa carte (section 3), ou dans le résultat, ligne par "
                    "ligne, avec le taux de chaque choix.")
    dispo = ET.repartiteurs_disponibles(etat)
    noms_fil = [n for n, d in dispo.items() if d.get("type") == "filiere"]
    cartes = ""
    for i, f in enumerate(cas.get("filieres", [])):
        opts = [(n, n) for n in noms_fil] + ([] if f["nom"] in noms_fil else [(f["nom"], f["nom"])])
        portes = "".join(C.case(["filieres", i, "porte", k], (cas["repartiteurs"].get(r) or {}).get("id") == f["id"], C.esc(r))
                         for k, r in enumerate(refs))
        d = dispo.get(f["nom"], {})
        cartes += (f'<div class="secard"><div class="sh2"><span class="indice">{C.esc(f["id"])}</span><span class="chp-tt">Filière · {C.esc(f["nom"])}</span>'
                   f'<span class="chp-act">{_mini_bouton(["filieres", i, "-"], "x", "Retirer cette filière")}</span></div>'
                   + L("Élément", C.choix(["filieres", i, "nom"], f["nom"], opts), full=True)
                   + _trio(("x (centre)", _num(["filieres", i, "x"], f["x"], "m", dec=3)), ("y (centre)", _num(["filieres", i, "y"], f["y"], "m", dec=3)),
                           ("longueur", _num(["filieres", i, "longueur"], f["longueur"], "m", mini=0.1)))
                   + L("Orientation", C.segments(["filieres", i, "orientation"], f.get("orientation", "y"), ORIENT_FIL), full=True)
                   + (f'<div class="chp-portes"><span>appuis portés</span>{portes}</div>' if refs else "")
                   + C.aide(C.esc(f"{d.get('origine', '')} · {(d.get('source') or '')[:140]}")) + "</div>")
    if cartes:
        s += f'<div class="secards">{cartes}</div>'
    return s


def _sec_bibliotheque(etat: Dict) -> str:
    L = C.ligne
    lignes = ""
    for nom, g in ET.gabarits(etat).items():
        src = (g.get("source") or "").strip()
        lignes += (f'<div class="chp-eng"><span class="chp-pic">{DP.pictogramme(g)}</span><span class="chp-eng-t"><b>{C.esc(nom.split(" · ")[0])}</b>'
                   f'<span>{C.esc(" · ".join(nom.split(" · ")[1:]) or g.get("type_appuis", ""))}</span>'
                   f'<small>{C.esc(src[:140] + ("…" if len(src) > 140 else ""))}</small></span>'
                   f'{_bouton(["charges", "+engin"], "Poser", valeur=nom)}</div>')
    fe = etat["site"]["fiche_engin"]
    tete = '<div class="chp-ap chp-ap-h"><span>appui</span><span>dx (m)</span><span>dy (m)</span><span></span><span>a (m)</span><span>b (m)</span><span>P (kN)</span><span>φ</span></div>'
    rangs = ""
    for j, a in enumerate(fe["appuis"]):
        b = ["note", "fiche_engin", "appuis", j]
        rangs += (f'<div class="chp-ap">{C.texte(b + ["nom"], a["nom"], "nom")}{_num(b + ["dx"], a["dx"], "")}{_num(b + ["dy"], a["dy"], "")}'
                  f'{_mini_bouton(b + ["-"], "x", "Retirer cet appui")}{_num(b + ["a"], a["a"], "")}{_num(b + ["b"], a["b"], "")}'
                  f'{_num(b + ["P"], a["P"], "")}{_num(b + ["phi"], a["phi"], "")}</div>')
    fr_ = etat["site"]["fiche_rep"]
    plateau = fr_["type"] == "plateau"
    rep = (L("Type", C.segments(["note", "fiche_rep", "type"], fr_["type"], [("plateau", "Plateau sous un appui"), ("filiere", "Filière")]), full=True)
           + L("Nom", C.texte(["note", "fiche_rep", "nom"], fr_["nom"], "ex. Plateau azobé 800 × 800 × 80"), full=True)
           + L("Matériau", C.choix(["note", "fiche_rep", "materiau"], fr_["materiau"], [(m, m) for m in OP.MATERIAUX]), full=True)
           + L("résistance", _num(["note", "fiche_rep", "f"], fr_["f"], "MPa", mini=0), "f",
               "f_y/γ_M0 (acier) ou f_m,d (bois, OSB) ; 0 : f_y de la nuance (S235, S355), sinon à renseigner — sans elle, l'élément ne se calcule pas.")
           + L("module", _num(["note", "fiche_rep", "E"], fr_["E"], "MPa"), "E", "Filière : raideur EI ; plateau : sans effet.")
           + L("masse", _num(["note", "fiche_rep", "masse"], fr_["masse"], "kg" if plateau else "kg/m", mini=0), "")
           + (L("dimension A", _num(["note", "fiche_rep", "A"], fr_["A"], "mm"), "A") + L("dimension B", _num(["note", "fiche_rep", "B"], fr_["B"], "mm"), "B")
              + L("épaisseur", _num(["note", "fiche_rep", "t"], fr_["t"], "mm"), "t") if plateau else
              L("Section", C.choix(["note", "fiche_rep", "profil"], fr_["profil"], [(OP.RECTANGLE, "Rectangle b × h sur champ")] + [(n, n) for n in BIB.profils()]), full=True)
              + (L("largeur", _num(["note", "fiche_rep", "A"], fr_["A"], "mm"), "b") + L("hauteur sur champ", _num(["note", "fiche_rep", "B"], fr_["B"], "mm"), "h")
                 if fr_["profil"] == OP.RECTANGLE else ""))
           + L("Source", C.texte(["note", "fiche_rep", "source"], fr_["source"], "fiche, norme, date"), full=True)
           + _bouton(["note", "fiche_rep", "enregistrer"], "Enregistrer l'élément", "file-plus-2"))
    return (f'<div class="chp-lib">{lignes}</div>'
            + '<div class="grp">Nouvelle fiche d\'engin</div>'
            + L("Nom", C.texte(["note", "fiche_engin", "nom"], fe["nom"], "modèle, constructeur"), full=True)
            + L("Appuis", C.segments(["note", "fiche_engin", "type_appuis"], fe["type_appuis"], TYPES_APPUIS), full=True)
            + _trio(("longueur châssis", _num(["note", "fiche_engin", "longueur"], fe["longueur"], "m")),
                    ("largeur châssis", _num(["note", "fiche_engin", "largeur"], fe["largeur"], "m")),
                    ("flèche (projection)", _num(["note", "fiche_engin", "fleche"], fe["fleche"], "m")))
            + C.aide("Appuis : position depuis le centre (x vers l'avant, y vers la gauche), empreinte, charge maximale, φ.")
            + tete + rangs + _bouton(["note", "fiche_engin", "appuis", "+"], "Ajouter un appui")
            + L("Source", C.texte(["note", "fiche_engin", "source"], fe["source"], "document, date, page"), full=True)
            + _bouton(["note", "fiche_engin", "enregistrer"], "Enregistrer la fiche", "file-plus-2")
            + '<div class="grp">Nouvel élément de répartition</div>' + rep
            + C.aide("Les fiches ajoutées voyagent avec la note (fichier .stab)."))


def _sec_hypotheses(cas: Dict, calc: Dict) -> str:
    pr = cas["params"]
    L = C.ligne
    s = (L("coefficient sur G", _num(["params", "gamma_G"], pr["gamma_G"], ""), "γ<sub>G</sub>", "Poids propres (hourdis, chape, dalle) : 1,35 à l'ELU (P-2).")
         + L("coefficient sur Q", _num(["params", "gamma_Q"], pr["gamma_Q"], ""), "γ<sub>Q</sub>",
             "Charges présentes, engins × φ, étais, stocks : 1,5 à l'ELU ; aussi sur les effets locaux, la filière et le plateau (P-2).")
         + L("part du tranchant", _num(["params", "part_tranchant_element"], pr["part_tranchant_element"], ""), "",
             "Part de la réaction d'une charge concentrée gardée par l'élément chargé (P-3) : 1 = sans partage par les joints, prudent ; "
             "l'annexe C de l'EN 1168 répartit la flexion, pas le tranchant près de l'appui.")
         + L("k du Heft 240", _num(["params", "k_flexion_heft240"], pr["k_flexion_heft240"], ""), "k",
             "Largeur efficace en flexion b_m = t_y + k·x·(1 − x/l) ; vide : non transcrit, b_m = t_y (P-4).")
         + L("résistance du joint", _num(["params", "joint_adm_kN_m"], pr["joint_adm_kN_m"], "kN/m"), "",
             "Joint longitudinal des hourdis (P-15) ; vide : non vérifié.")
         + L("longueur de joint", _num(["params", "joint_longueur_m"], pr["joint_longueur_m"], "m"), "", "Longueur de joint mobilisée ; vide : non vérifié.")
         + L("coefficient α<sub>cc</sub>", _num(["params", "alpha_cc"], pr["alpha_cc"], ""), "", "Effets locaux EN 1992-1-1 (ANB à confirmer)."))
    p = calc.get("params")
    if p is not None:
        s += C.aide(f"ELU quand la résistance est une valeur de calcul : γ<sub>G</sub> {C.nb(p.gamma_G)} sur les poids propres, γ<sub>Q</sub> "
                    f"{C.nb(p.gamma_Q)} sur les charges de chantier (coefficients 1 avec une valeur admissible de service) · tranchant : "
                    f"{_pc100(p.part_tranchant_element)} de la réaction d'une charge concentrée sur l'élément chargé · "
                    + ("k Heft 240 transcrit" if p.k_flexion_heft240 is not None else "b<sub>m</sub> = t<sub>y</sub> (k à transcrire)"))
    for h in (calc.get("res") or {}).get("hypotheses", []):
        s += C.aide("• " + C.symb_math(_txt(h)))
    return s


def _sec_affichage(etat: Dict, cas: Dict, calc: Dict) -> str:
    L = C.ligne
    o = cas["options"]
    index = calc.get("index")
    refs = list(index["empreintes"]) if index else []
    s = ('<div class="grp">Plan</div>' + C.case(["options", "cotes"], o["cotes"], "Cotes")
         + C.case(["options", "aires"], o["aires"], "Aires efficaces sous les plateaux") + C.case(["options", "taux"], o["taux"], "Taux par élément"))
    if refs:
        s += L("Coupe sous", C.choix(["selection"], _appui_coupe(cas, refs), [(f"rep:{r}", r) for r in refs]), full=True)
    engins = [c["ref"] for c in cas["charges"] if c["type"] == "engin"]
    if engins:
        s += L("Engin balayé", C.choix(["site", "balaye"], _balaye(cas), [(r, r) for r in engins]), full=True)
    s += ('<div class="grp">Identification de la note</div>'
          + C.aide(NC.AIDE_IDENTIFICATION)
          + '<div class="grp">Fichier de l\'écran Streamlit</div>'
          + C.fichier(["note", "ouvrir"], "Ouvrir un fichier JSON du module", ".json")
          + C.aide("Le fichier « Enregistrer » de l'écran Streamlit (tous les cas et les fiches ajoutées) remplace cette note."))
    return s


def _appui_coupe(cas: Dict, refs: List[str]) -> str:
    sel = cas.get("selection") or ""
    r = sel.split(":", 1)[1] if sel.startswith("rep:") else None
    return f"rep:{r if r in refs else refs[0]}"


def _balaye(cas: Dict) -> Optional[str]:
    engins = [c["ref"] for c in cas["charges"] if c["type"] == "engin"]
    b = cas["site"].get("balaye")
    return b if b in engins else (engins[0] if engins else None)


def formulaire(etat: Dict, calc: Dict) -> str:
    cas = _cas(etat)
    pl = cas["plancher"]
    res = calc.get("res") or {}
    n = len(cas["charges"])
    t = pl["type"]
    if t == "hourdis":
        h = pl["hourdis"]
        s2 = C.section(2, "Hourdis", f"{C.esc(_court(h['type_cat'], 26))} · M<sub>Rd</sub> {C.nb(h['M_Rd'])} · V<sub>Rd</sub> {C.nb(h['V_Rd'])}",
                       _sec_hourdis(etat, cas), True, "ho")
    else:
        d = pl[t]
        s2 = C.section(2, TYPES_COURTS[t], f"{C.esc(d['beton'])} · h {C.fr(d['h'] * 1e3, 0)} mm · A<sub>s</sub> {C.fr(d['As_inf'], 0)} mm²/m",
                       _sec_dalle(cas), True, "da")
    nf = len(cas.get("filieres", []))
    np_ = sum(1 for r in cas["repartiteurs"].values() if r["type"] == "plateau")
    gabs = ET.gabarits(etat)
    p = calc.get("params")
    return (C.section(1, "Plancher", C.esc(OP.resume_plancher(cas)), _sec_plancher(cas), True, "pl")
            + s2
            + C.section(3, "Charges", f"{n} objet(s) · {OP.fmt(OP.total_charges(cas), 0)} kN d'engins et d'étançons",
                        _sec_charges(etat, cas, calc.get("index")), True, "ch")
            + C.section(4, "Répartiteurs", f"{np_} plateau(x) · {nf} filière(s)", _sec_repartiteurs(etat, cas, calc), res.get("statut") == "nok", "re")
            + C.section(5, "Bibliothèque et fiches", f"{len(gabs)} engins · {len(etat['fiches']['repartiteurs'])} fiche(s) de répartiteur",
                        _sec_bibliotheque(etat), not cas["charges"], "bi")
            + C.section(6, "Hypothèses", (f"γ<sub>G</sub> {C.nb(p.gamma_G)} · γ<sub>Q</sub> {C.nb(p.gamma_Q)} · tranchant {_pc100(p.part_tranchant_element)}"
                                          if p is not None else ""), _sec_hypotheses(cas, calc), False, "hy")
            + C.section(7, "Affichage et note", "plan, coupe, balayage, identification", _sec_affichage(etat, cas, calc), False, "af"))


# ------------------------------------------------------------------ le résultat : nombres et couleurs
LIB_FAM = dict(OP.FAMILLES, fissuration="Fissuration")
ORDRE_FAM = [f for f, _ in OP.FAMILLES] + ["fissuration"]
CHIFFRES = ("ok", "attention", "nok")
ND = {"kN·m": 2, "kN": 2, "kN/m": 2, "kN·m/m": 2, "N/mm²": 2, "mm": 0, "mm²": 0, "mm²/m": 0, "m": 2, "kN/m²": 2, "MPa": 1, "kg": 1}
TEXTES_SANS_TAUX = {"nok": "répartiteur exigé", "non_verifie": "non vérifié", "info": "information"}


def _nd(x, u: str) -> int:
    """La précision d'usage d'une unité ; une contrainte sous 1 N/mm² garde trois décimales (0,564 et non 0,56)."""
    return 3 if u == "N/mm²" and abs(x) < 1 else ND.get(u, 2)


def _v(x, u: str = "") -> str:
    """Un nombre écrit à la précision de son unité, SANS l'unité : les valeurs substituées d'une formule."""
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "∞"
    return C.fr(x, _nd(x, u))


def _r(x, u: str = "") -> str:
    """Un RÉSULTAT écrit à la précision d'usage de son unité, l'unité à la suite."""
    s = _v(x, u)
    return f"{s}{C.NBSP}{u}" if u and s not in ("—", "∞") else s


def _pc100(x: float) -> str:
    """Une part écrite en pour cent à la française : 1 → « 100 % », 0,7 → « 70 % »."""
    return f"{C.fr(x * 100, 0)}{C.NBSP}%"


def _d(x, nd: int = 4) -> str:
    """Une DONNÉE d'une formule, écrite sans la dénaturer : 1,35 reste 1,35, 0,05 reste 0,05 (C.nb)."""
    if x is None:
        return "—"
    return C.nb(round(float(x), nd), nd)


def _niv(v: Dict) -> str:
    """La couleur d'une vérification : rouge dès que le moteur la déclare non conforme, sinon celle de son taux ;
    gris pour une ligne d'information ou non vérifiée."""
    if v["statut"] == "nok":
        return "r"
    if v["taux"] is None or v["statut"] not in CHIFFRES:
        return "c"
    return C.niveau(v["taux"])


def _texte_taux(v: Dict) -> str:
    """Ce qui s'écrit à la place du taux quand le moteur n'en donne pas (ou que la ligne n'est qu'une information)."""
    if v["taux"] is not None and v["statut"] in CHIFFRES:
        return ""
    if v["statut"] == "info" and v["taux"] is not None:
        return f"info · {C.pc(C.pct(v['taux']))}{C.NBSP}%"
    return TEXTES_SANS_TAUX.get(v["statut"], "—")


def _ligne_res(nom: str, piece: str, v: Optional[Dict], taux, niv: str, calcul: str = "", pk: str = "", gov: bool = False,
               texte: str = "") -> str:
    """La ligne de résultat du site (C.ligne_res) ; quand le moteur ne chiffre pas la ligne (répartiteur exigé, non
    vérifié, information), son texte remplace « à compléter »."""
    h = C.ligne_res(nom, piece, None if texte else taux, niv, calcul=calcul, pk=pk, gov=gov)
    return h.replace(">à compléter<", f">{C.esc(texte)}<") if texte else h


def _remplacer_puces(h: str, textes: List[str]) -> str:
    """Les lignes d'un tableau de calcul (ou d'un sélecteur) sans taux portent la puce grise « 0 % » de C.chip : on y
    écrit leur texte (« non vérifié », « refusé »…), dans l'ordre."""
    for t in textes:
        h = h.replace(f'<span class="chip c">0{C.NBSP}%</span>', f'<span class="chip c">{C.esc(t)}</span>', 1)
    return h


# ------------------------------------------------------------------ le résultat : chaque vérification en clair
def _comparaison(v: Dict, nature: str, gQ: float) -> Tuple[str, str, str]:
    """(formule, valeur, limite) d'une vérification du moteur, en HTML : ce que le tableau de la fenêtre et le verdict
    écrivent. Les valeurs sont celles que le moteur a nommées (E, R)."""
    fam, u, E, R, obj = v["famille"], v.get("unite", ""), v["E"], v["R"], v["objet"]
    rd = "Rd" if nature == "Rd" else "adm"
    if fam == "flexion" and u == "kN·m":
        f = (f"M<sub>Ed</sub> = γ<sub>G</sub>·M<sub>G</sub> + γ<sub>Q</sub>·M<sub>Q</sub> ≤ M<sub>{rd}</sub>" if nature == "Rd"
             else "M<sub>G</sub> + M<sub>Q</sub> ≤ M<sub>adm</sub>")
        return f, f"M<sub>Ed</sub> = {_r(E, u)}", f"M<sub>{rd}</sub> = {_r(R, u)}"
    if fam == "tranchant" and u == "kN":
        return (f"V<sub>Ed</sub> = max(V<sub>A</sub> ; V<sub>B</sub>) ≤ V<sub>{rd}</sub>", f"V<sub>Ed</sub> = {_r(E, u)}",
                f"V<sub>{rd}</sub> = {_r(R, u)}")
    if fam == "flexion":
        return ("M = M<sub>w</sub> + max Σ M<sub>P</sub>/b<sub>m</sub> ≤ M<sub>adm</sub>", f"M = {_r(E, u)}", f"M<sub>adm</sub> = {_r(R, u)}")
    if fam == "tranchant":
        return ("V = V<sub>w</sub> + max Σ R/b<sub>v</sub> ≤ V<sub>adm</sub>", f"V = {_r(E, u)}", f"V<sub>adm</sub> = {_r(R, u)}")
    if fam == "fissuration":
        return "M<sub>G</sub> + M<sub>Q</sub> ≤ M<sub>cr</sub>", f"M<sub>ser</sub> = {_r(E, u)}", f"M<sub>cr</sub> = {_r(R, u)}"
    if fam == "local":
        if obj.startswith("membrure"):
            return "p = P / (a·b) ≤ p<sub>adm</sub>", f"p = {_r(E, u)}", (f"p<sub>adm</sub> = {_r(R, u)}" if R is not None else "non renseignée")
        if obj.startswith("pression localisée"):
            return "γ<sub>Q</sub>·P ≤ F<sub>Rdu</sub> (6.63)", f"F<sub>Ed</sub> = {_r(E, u)}", f"F<sub>Rdu</sub> = {_r(R, u)}"
        if obj.startswith("poinçonnement au nu"):
            return "v<sub>Ed,0</sub> ≤ v<sub>Rd,max</sub>", f"v<sub>Ed,0</sub> = {_r(E, u)}", f"v<sub>Rd,max</sub> = {_r(R, u)}"
        return "v<sub>Ed</sub> = β·V<sub>Ed</sub>/(u<sub>1</sub>·d) ≤ v<sub>Rd,c</sub>", f"v<sub>Ed</sub> = {_r(E, u)}", f"v<sub>Rd,c</sub> = {_r(R, u)}"
    if fam == "plateau":
        if u == "mm":
            return "c = t·√(f / (3·p<sub>Ed</sub>)) ≤ c<sub>max</sub>", f"c = {_r(E, u)}", f"c<sub>max</sub> = {_r(R, u)}"
        return "p ≤ p<sub>adm</sub> (déclarée)", f"p = {_r(E, u)}", f"p<sub>adm</sub> = {_r(R, u)}"
    if fam == "filiere":
        if E is None:
            return "—", "—", "—"
        return ("γ<sub>Q</sub>·M<sub>max</sub> ≤ W<sub>el</sub>·f<sub>y</sub>/γ<sub>M0</sub>", f"γ<sub>Q</sub>·M = {_r(gQ * E, 'kN·m')}",
                f"M<sub>Rd</sub> = {_r(R, 'kN·m')}")
    if fam == "joint":
        return ("γ<sub>Q</sub>·F ≤ R<sub>j</sub>·ℓ<sub>j</sub>", f"γ<sub>Q</sub>·F = {_r(gQ * E, 'kN')}",
                f"R<sub>j</sub>·ℓ<sub>j</sub> = {_r(R, 'kN')}" if R is not None else "à transcrire")
    return "—", _r(E, u), _r(R, u)


def _ligne_tab(v: Dict, nature: str, gQ: float, gov: bool = False, nom: str = "") -> Dict:
    f, val, lim = _comparaison(v, nature, gQ)
    t = _texte_taux(v)
    return {"nom": C.esc(nom or LIB_FAM.get(v["famille"], v["famille"])), "sous": C.esc(_txt(v["objet"])), "formule": f, "valeur": val,
            "limite": lim, "taux": 0 if t else C.pct(v["taux"]), "niveau": "c" if t else _niv(v), "gov": gov, "_texte": t}


def _fenetre(ident: str, titre: str, sous_titre: str, hyps: Sequence[str], lignes: List[Dict], detail: str, pied: str = "") -> str:
    """C.fenetre_calcul, les lignes sans taux portant leur texte (non vérifié, information, répartiteur exigé)."""
    h = C.fenetre_calcul(ident, titre, sous_titre, hyps, lignes, detail, pied)
    return _remplacer_puces(h, [li["_texte"] for li in lignes if li.get("_texte")])


def _groupe(titre: str, lignes: List[Dict], apres: str = "", large: bool = False) -> str:
    """Un groupe du détail du calcul ; ``large`` : sur les deux colonnes (un tableau de plusieurs colonnes)."""
    return f'<div class="ck-g{" chp-pleine" if large else ""}"><div class="ck-t">{titre}</div>{C.lignes_fx(lignes)}{apres}</div>'


def _fx(sym: str, formule: str, valeurs: str, res: str, niveau: str = "") -> Dict:
    parts = [sym] + [p for p in (formule, valeurs) if p]
    return {"parts": parts, "res": res, "niveau": niveau}


def _eta(sym: str, num: float, den: Optional[float], u: str, niv: Optional[str] = None) -> Dict:
    """η = E / R = valeurs = taux (puce de couleur)."""
    if not den:
        return _fx(sym, "", f"{_r(num, u)} / —", "non vérifié", "c")
    t = num / den
    return _fx(sym, "", f"{_r(num, u)} / {_r(den, u)}", f"{C.pc(C.pct(t))} %", niv or C.niveau(t))


def _mini_table(tete: Sequence[str], lignes: List[Sequence[str]]) -> str:
    """Un petit tableau dans le détail du calcul (les charges d'un élément, les largeurs efficaces)."""
    n = ' class="n"'
    th = "".join(f"<th{n if i else ''}>{t}</th>" for i, t in enumerate(tete))
    tr = "".join("<tr>" + "".join(f"<td{n if i else ''}>{c}</td>" for i, c in enumerate(li)) + "</tr>" for li in lignes)
    return f'<div class="chp-defile"><table class="vtab chp-mini"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def _texte_p3(ptr: float, repartition: bool) -> str:
    """Ce que P-3 fait du tranchant d'une charge concentrée, tel que le moteur le calcule (verification._efforts_hourdis)."""
    base = f"P-3 : {_pc100(ptr)} de la réaction d'appui de chaque charge concentrée reste sur l'élément chargé"
    pose = ("Σ R porte les charges POSÉES sur l'élément (sa part du levier entre deux éléments), sans les parts de l'annexe C "
            "de la flexion : ce ne sont pas toujours celles du tableau ci-dessus")
    if ptr >= 1.0:
        return f"{base}, sans partage par les joints. {pose}."
    if repartition:
        return f"{base} ; les {_pc100(1 - ptr)} restants passent aux voisins selon les parts de l'annexe C. {pose}, plus ces parts des voisins."
    return (f"{base} ; répartition inactive : les {_pc100(1 - ptr)} restants ne sont portés par AUCUN élément — défaut du moteur, "
            f"NON CONSERVATIF (voir le message du résultat). {pose}.")


def _alerte_p3(calc: Dict) -> str:
    """Défaut NON CONSERVATIF du moteur (le même dans l'écran Streamlit, reproduit et DIT) : une part du tranchant < 1 sans
    répartition active (hourdis posés, ou parts de l'annexe C absentes) perd la part (1 − p) de chaque charge concentrée."""
    pl, p, res = calc.get("plancher"), calc.get("params"), calc.get("res") or {}
    if pl is None or pl.type != "hourdis" or p is None or p.part_tranchant_element >= 1.0 or pl.repartition_active() or not res.get("patches"):
        return ""
    return C.message(f"<b>Tranchant sous-estimé (P-3)</b> : part {_pc100(p.part_tranchant_element)} sans répartition active — "
                     f"{_pc100(1 - p.part_tranchant_element)} de chaque charge concentrée ne sont portés par aucun élément (défaut du "
                     "moteur, non conservatif, aussi dans Streamlit). Remettre la part à 1 (section 6) ou activer la répartition "
                     "(section 2).", grave=True)


def _table_charges(cas: Dict, res: Dict, mp: List[Tuple[str, float, float, float]], titre_m: str) -> str:
    """Les charges concentrées d'un élément (ou d'une dalle), (réf, x, P, M) : une ligne par charge — les morceaux d'un
    appui (levier sur deux éléments, parts de l'annexe C) additionnés, une file d'étais ou un stock (découpés en
    charges de 0,25 ou 0,30 m par le moteur) sur UNE ligne, avec l'étendue de ses x ; puis la somme."""
    if not mp:
        return '<div class="hint">Aucune charge concentrée sur cet élément.</div>'
    origine = {pt["ref"]: pt.get("origine") for pt in res.get("patches", [])}
    sortes = {c["ref"]: c["type"] for c in cas["charges"]}
    groupes: Dict[str, List] = {}
    for ref, x, P, m in mp:
        o = origine.get(ref)
        cle = o if sortes.get(o) in ("lineaire", "surfacique") else ref
        groupes.setdefault(cle, []).append((ref, x, P, m))
    lignes = []
    for cle, morceaux in groupes.items():
        xs = [x for _, x, _, _ in morceaux]
        n = len({r for r, *_ in morceaux})
        nom = C.esc(cle) + (f" <small>({n} morceaux)</small>" if n > 1 else "")
        xt = _r(xs[0], "m") if max(xs) - min(xs) < 1e-9 else f"{_v(min(xs), 'm')} à {_r(max(xs), 'm')}"
        lignes.append((nom, xt, _r(sum(P for _, _, P, _ in morceaux), "kN"), _r(sum(m for *_, m in morceaux), "kN·m")))
    lignes.append(("<b>Σ</b>", "", f"<b>{_r(sum(P for _, _, P, _ in mp), 'kN')}</b>", f"<b>{_r(sum(m for *_, m in mp), 'kN·m')}</b>"))
    return _mini_table(("charge", "x", "P", titre_m), lignes)


# ------------------------------------------------------------------ les fenêtres de calcul
def _hyps_communes(cas: Dict, calc: Dict) -> List[str]:
    pl, p = calc["plancher"], calc["params"]
    if pl.type == "hourdis":
        rh = RS.resistance_hourdis(pl, p.gamma_G, p.gamma_Q)
        rd = "Rd" if rh["nature"] == "Rd" else "adm"
        nat = (f"résistances de <b>calcul</b> · ELU γ<sub>G</sub> = {_d(rh['gamma_G'])}, γ<sub>Q</sub> = {_d(rh['gamma_Q'])}"
               if rh["nature"] == "Rd" else "valeurs <b>admissibles</b> de service · coefficients 1")
        return [f"<b>{C.esc(_court(pl.nom, 34))}</b> · {nat}",
                f"L = {_d(pl.L)} m · {pl.n_elements} × b = {_d(pl.b_element)} m",
                f"M<sub>{rd}</sub> = {_r(rh['M_Rd'], 'kN·m')} · V<sub>{rd}</sub> = {_r(rh['V_Rd'], 'kN')}"
                + (" · <b>avec chape</b>" if rh["avec_chape"] else "")]
    rdl = RS.resistance_dalle(pl)
    L = pl.portee_etayee if getattr(pl, "portee_etayee", None) else pl.L
    return [f"<b>{TYPES_COURTS[pl.type]}</b> {C.esc(rdl['classe'])} · f<sub>yk</sub> = {_d(pl.fyk)} N/mm² · γ<sub>s</sub> = {_d(pl.gamma_s)}",
            f"h = {_r(pl.h * 1e3, 'mm')} · d = {_r(pl.d * 1e3, 'mm')} · A<sub>s</sub> = {_r(pl.As_inf, 'mm²/m')}",
            f"L = {_d(L)} m" + (" (étais en place)" if L != pl.L else "") + f" · contraintes admissibles, charges de service"]


def _fen_element(cas: Dict, calc: Dict, e: Dict) -> str:
    """La fenêtre d'un élément de hourdis : ses charges, sa flexion et son tranchant, comme une note faite à la main —
    chaque grandeur est celle du moteur (verification._efforts_hourdis), chaque formule écrite avec ses valeurs."""
    pl, p, res = calc["plancher"], calc["params"], calc["res"]
    rh = RS.resistance_hourdis(pl, p.gamma_G, p.gamma_Q)
    nature = rh["nature"]
    rd = "Rd" if nature == "Rd" else "adm"
    gG, gQ, L, b, x = e["gamma_G"], e["gamma_Q"], pl.L, e["y1"] - e["y0"], e["x_M"]
    vs = [v for v in res["verifications"] if v["objet"] == e["nom"] and v["famille"] in ("flexion", "tranchant", "fissuration")]
    t_max = max((v["taux"] for v in vs if v["statut"] in CHIFFRES), default=None)
    lignes = [_ligne_tab(v, nature, gQ, gov=v["taux"] == t_max and v["statut"] in CHIFFRES) for v in vs]
    g_txt = (f"(g + γ_c·e)·b", f"({_d(pl.g)} + 25 × {_d(pl.chape.epaisseur)}) × {_d(b)}") if pl.chape else ("g·b", f"{_d(pl.g)} × {_d(b)}")
    charges = [_fx("w_G", g_txt[0], g_txt[1], _r(e["w_g"], "kN/m")),
               _fx("w_Q", "q·b", f"{_d(pl.q_presentes)} × {_d(b)}", _r(e["w_q"], "kN/m"))]
    mp = [(o, xp, P, DF.moment_ponctuel(P, xp, x, L)) for xp, P, o in e["charges"]]
    tab = _table_charges(cas, res, mp, f"M<sub>P</sub> en x = {_d(x, 3)} m")
    som = sum(m for *_, m in mp)
    m_wq = DF.moment_uniforme(e["w_q"], x, L)
    flex = [_fx("M_G", "w_G·x·(L − x)/2", f"{_r(e['w_g'])} × {_d(x, 3)} × ({_d(L)} − {_d(x, 3)}) / 2", _r(e["M_G"], "kN·m")),
            _fx("M_Q", "w_Q·x·(L − x)/2 + Σ M_P", f"{_r(m_wq)} + {_r(som)}", _r(e["M_Q"], "kN·m"))]
    if nature == "Rd":
        flex.append(_fx("M_Ed", "γ_G·M_G + γ_Q·M_Q", f"{_d(gG)} × {_r(e['M_G'])} + {_d(gQ)} × {_r(e['M_Q'])}", _r(e["M_max"], "kN·m")))
    else:
        flex.append(_fx("M_Ed", "M_G + M_Q", f"{_r(e['M_G'])} + {_r(e['M_Q'])}", _r(e["M_max"], "kN·m")))
    flex.append(_eta("η_M", e["M_max"], e["M_adm"], "kN·m"))
    VG, VQ = DF.tranchant_uniforme(e["w_g"], L), DF.tranchant_uniforme(e["w_q"], L)
    # Σ R : la part des charges concentrées dans la réaction (P-3), lue dans V_A et V_B du moteur (projection d'affichage)
    RA = (e["V_A"] - gG * VG) / gQ - VQ if gQ else 0.0
    RB = (e["V_B"] - gG * VG) / gQ - VQ if gQ else 0.0
    ptr = p.part_tranchant_element
    tranch = [_fx("V_G", "w_G·L/2", f"{_r(e['w_g'])} × {_d(L)} / 2", _r(VG, "kN")),
              _fx("V_Q", "w_Q·L/2", f"{_r(e['w_q'])} × {_d(L)} / 2", _r(VQ, "kN")),
              _fx("Σ R_A", f"{_d(ptr)}·Σ P·(L − x)/L" if ptr != 1 else "Σ P·(L − x)/L", "", _r(RA, "kN")),
              _fx("Σ R_B", f"{_d(ptr)}·Σ P·x/L" if ptr != 1 else "Σ P·x/L", "", _r(RB, "kN")),
              _fx("V_A", "γ_G·V_G + γ_Q·(V_Q + Σ R_A)", f"{_d(gG)} × {_r(VG)} + {_d(gQ)} × ({_r(VQ)} + {_r(RA)})", _r(e["V_A"], "kN")),
              _fx("V_B", "γ_G·V_G + γ_Q·(V_Q + Σ R_B)", f"{_d(gG)} × {_r(VG)} + {_d(gQ)} × ({_r(VQ)} + {_r(RB)})", _r(e["V_B"], "kN")),
              _fx("V_Ed", "max(V_A ; V_B)", f"max({_r(e['V_A'])} ; {_r(e['V_B'])})", _r(e["V_max"], "kN")),
              _eta("η_V", e["V_max"], e["V_adm"], "kN")]
    detail = (_groupe("Charges réparties", charges, '<div class="ck-t chp-sous">Charges concentrées portées par l\'élément</div>' + tab)
              + _groupe(f"Flexion · section x = {_d(x, 3)} m", flex)
              + _groupe("Tranchant aux appuis A et B", tranch, f'<div class="hint">{_texte_p3(ptr, pl.repartition_active())}</div>'))
    fis = next((v for v in vs if v["famille"] == "fissuration"), None)
    if fis:
        detail += _groupe("Fissuration (information, P-11)", [
            _fx("M_ser", "M_G + M_Q", f"{_r(e['M_G'])} + {_r(e['M_Q'])}", _r(fis["E"], "kN·m")),
            _fx("η_cr", "M_ser / M_cr", f"{_r(fis['E'])} / {_r(fis['R'])}", f"{C.pc(C.pct(fis['taux']))} %", "c")])
    hyps = _hyps_communes(cas, calc) + [f"élément {C.esc(e['nom'])} : y de {_d(e['y0'], 3)} à {_d(e['y1'], 3)} m"]
    return _fenetre(f"el-{e['index']}", f"Élément {C.esc(e['nom'])}", f"cas {C.esc(cas['nom'])} · flexion et tranchant · M<sub>{rd}</sub>, V<sub>{rd}</sub> "
                    f"du catalogue", hyps, lignes, detail)


def _fen_dalle(cas: Dict, calc: Dict) -> str:
    """La fenêtre de la dalle (ou de la prédalle) : résistances par mètre (formules de l'application inversées),
    charges réparties, largeurs efficaces de chaque charge, moment et tranchant de la bande gouvernante."""
    pl, p, res = calc["plancher"], calc["params"], calc["res"]
    rdl = RS.resistance_dalle(pl)
    L = pl.portee_etayee if getattr(pl, "portee_etayee", None) else pl.L
    vs = [v for v in res["verifications"] if v["famille"] in ("flexion", "tranchant")]
    t_max = max((v["taux"] for v in vs), default=None)
    lignes = [_ligne_tab(v, "service", p.gamma_Q, gov=v["taux"] == t_max) for v in vs]
    d_mm, h_mm = pl.d * 1e3, pl.h * 1e3
    resist = [_fx("f_yd", "f_yk / γ_s", f"{_d(pl.fyk)} / {_d(pl.gamma_s)}", _r(rdl["fyd"], "N/mm²")),
              _fx("M_adm,b", "α_b·μ·b·d² / 10⁶", f"{_d(rdl['alpha_b'])} × {_d(rdl['mu'])} × 1000 × {_d(d_mm, 1)}² / 10⁶", _r(rdl["M_adm_beton"], "kN·m/m")),
              _fx("M_adm,s", "A_s·f_yd·0,9·d / 10⁶", f"{_d(pl.As_inf, 1)} × {_v(rdl['fyd'], 'N/mm²')} × 0,9 × {_d(d_mm, 1)} / 10⁶", _r(rdl["M_adm_acier"], "kN·m/m")),
              _fx("M_adm", "min(M_adm,b ; M_adm,s)", f"min({_r(rdl['M_adm_beton'])} ; {_r(rdl['M_adm_acier'])})", _r(rdl["M_adm"], "kN·m/m")),
              _fx("τ_adm,I", "0,016·f_ck,cube / 1,05", f"0,016 × {_d(rdl['fck_cube'])} / 1,05", _r(rdl["tau_adm_I"], "N/mm²")),
              _fx("V_adm", "τ_adm,I·0,75·b·h / 10³", f"{_v(rdl['tau_adm_I'], 'N/mm²')} × 0,75 × 1000 × {_d(h_mm, 1)} / 10³", _r(rdl["V_adm"], "kN/m"))]
    w = pl.g_total()
    rep = [_fx("w", "25·h + q", f"25 × {_d(pl.h)} + {_d(pl.q_presentes)}", _r(w, "kN/m²"))]
    lign = []
    sortes = {c["ref"]: c["type"] for c in cas["charges"]}
    groupes: Dict[str, List] = {}
    for pt in res.get("patches", []):
        bm, bvA, bvB, _ = V._largeurs(pl, pt, p)
        o = pt.get("origine")
        cle = o if sortes.get(o) in ("lineaire", "surfacique") else pt["ref"]
        groupes.setdefault(cle, []).append((pt["x"], pt["y"], pt["P"], DF.t_y(pt["b"], pl.h1, pl.h, pl.L), bm, bvA, bvB))

    def etendue(vals, u):
        return _r(vals[0], u) if max(vals) - min(vals) < 5e-4 else f"{_v(min(vals), u)}–{_r(max(vals), u)}"
    for cle, morceaux in groupes.items():
        col = list(zip(*morceaux))
        nom = C.esc(cle) + (f" <small>({len(morceaux)} morceaux)</small>" if len(morceaux) > 1 else "")
        lign.append((nom, etendue(col[0], "m"), etendue(col[1], "m"), _r(sum(col[2]), "kN"),
                     *(etendue(c, "m") for c in col[3:])))
    tab = _mini_table(("charge", "x", "y", "P", "t<sub>y</sub>", "b<sub>m</sub>", "b<sub>v,A</sub>", "b<sub>v,B</sub>"), lign) if lign else \
        '<div class="hint">Aucune charge concentrée.</div>'
    par = "par le béton" if rdl["gouverne_M"] == "béton" else "par l'acier"
    detail = _groupe(f"Résistance par mètre · {C.esc(rdl['classe'])} (M gouverné {par})", resist)
    detail += _groupe("Charges réparties", rep, '<div class="hint">Poids propre de la dalle (25 kN/m³) et charges présentes, par mètre carré.</div>')
    detail += _groupe("Largeurs efficaces des charges concentrées (Heft 240)", [],
                      '<div class="hint">t<sub>y</sub> = b<sub>0</sub> + 2·h<sub>1</sub> + h ≤ 0,8·L ; '
                      'b<sub>m</sub> = t<sub>y</sub>' + (" + k·x·(1 − x/L)" if p.k_flexion_heft240 is not None else " (k non transcrit, P-4)") +
                      ' ; b<sub>v</sub> = t<sub>y</sub> + 0,5·a (a : distance à l\'appui)</div>' + tab, large=True)
    fl = next((v for v in vs if v["famille"] == "flexion"), None)
    if fl:
        x = fl.get("x", L / 2)
        Mw = DF.moment_uniforme(w, x, L)
        detail += _groupe(f"Flexion · section x = {_d(x, 3)} m, {C.esc(_txt(fl['objet']))}", [
            _fx("M_w", "w·x·(L − x)/2", f"{_r(w)} × {_d(x, 3)} × ({_d(L)} − {_d(x, 3)}) / 2", _r(Mw, "kN·m/m")),
            _fx("M_P", "max_y Σ M_P,i / b_m,i", "", _r(fl["E"] - Mw, "kN·m/m")),
            _fx("M", "M_w + M_P", f"{_r(Mw)} + {_r(fl['E'] - Mw)}", _r(fl["E"], "kN·m/m")),
            _eta("η_M", fl["E"], fl["R"], "kN·m/m")])
    tr = next((v for v in vs if v["famille"] == "tranchant"), None)
    if tr:
        Vw = DF.tranchant_uniforme(w, L)
        detail += _groupe(f"Tranchant · {C.esc(_txt(tr['objet']))}", [
            _fx("V_w", "w·L/2", f"{_r(w)} × {_d(L)} / 2", _r(Vw, "kN/m")),
            _fx("v_P", "max_y Σ R_i / b_v,i", "", _r(tr["E"] - Vw, "kN/m")),
            _fx("V", "V_w + v_P", f"{_r(Vw)} + {_r(tr['E'] - Vw)}", _r(tr["E"], "kN/m")),
            _eta("η_V", tr["E"], tr["R"], "kN/m")])
    return _fenetre("dalle", f"{TYPES_COURTS[pl.type]} · flexion et tranchant", f"cas {C.esc(cas['nom'])} · méthode de l'application, par mètre de largeur",
                    _hyps_communes(cas, calc), lignes, detail)


def _fen_local(cas: Dict, calc: Dict) -> str:
    """Les effets locaux : pression sur la membrure des hourdis (P-5), ou pression localisée et poinçonnement de la
    dalle (EN 1992-1-1 §6.7 et §6.4, P-9) — le tableau de toutes les charges, le détail des plus défavorables."""
    pl, p, res = calc["plancher"], calc["params"], calc["res"]
    vs = [v for v in res["verifications"] if v["famille"] == "local"]
    nature = getattr(pl, "nature", "service")

    def cle(v):
        return (v["statut"] == "nok", v["taux"] if v["taux"] is not None else -1.0, v["E"] or 0.0)
    pire = max(vs, key=cle) if vs else None
    tries = sorted(vs, key=cle, reverse=True)
    MAX_LIGNES = 8
    lignes = [_ligne_tab(v, nature, p.gamma_Q, gov=v is pire, nom="Local") for v in tries[:MAX_LIGNES]]
    patches = {pt["ref"]: pt for pt in res.get("patches", [])}
    detail = ""
    vus = []
    for v in sorted(vs, key=cle, reverse=True):
        ref = v["objet"].split(" sous ", 1)[-1]
        if ref in vus or ref not in patches or len(vus) >= 3:
            continue
        vus.append(ref)
        pt = patches[ref]
        a, b = pt["a"] * 1e3, pt["b"] * 1e3
        if pl.type == "hourdis":
            e = pl.chape.epaisseur * 1e3 if pl.chape else 0.0
            fx = [_fx("p", "P / ((a + 2e)·(b + 2e))" if e else "P / (a·b)",
                      f"{_r(pt['P'])} × 10³ / ({_d(a + 2 * e, 1)} × {_d(b + 2 * e, 1)})", _r(v["E"], "N/mm²"))]
            fx.append(_eta("η", v["E"], v["R"], "N/mm²") if v["R"] else
                      _fx("p_adm", "", "", "répartiteur exigé" if v["statut"] == "nok" else "non renseignée", "r" if v["statut"] == "nok" else "c"))
            detail += _groupe(f"Membrure sous {C.esc(ref)}" + (f" · via {C.esc(_court(pt['repartiteur'] or '', 26))}" if pt.get("repartiteur") else ""), fx)
        else:
            fck = RS.resistance_dalle(pl)["fck"]
            plr = RS.pression_localisee_Rd(a, b, pl.h * 1e3, fck, p.gamma_c, p.alpha_cc)
            po = RS.poinconnement(p.gamma_Q * pt["P"], a, b, pl.d * 1e3, fck, pl.rho_l, p.gamma_c, p.alpha_cc, p.beta_poinconnement, p.coef_vrd_max)
            fx = [_fx("f_cd", "α_cc·f_ck / γ_c", f"{_d(p.alpha_cc)} × {_d(fck)} / {_d(p.gamma_c)}", _r(plr["f_cd"], "N/mm²")),
                  _fx("A_c0", "a·b", f"{_d(a, 1)} × {_d(b, 1)}", _r(plr["A_c0"], "mm²")),
                  _fx("A_c1", "min(3a ; a + h)·min(3b ; b + h)", f"{_d(plr['a1'], 1)} × {_d(plr['b1'], 1)}", _r(plr["A_c1"], "mm²")),
                  _fx("F_Rdu", "min(A_c0·f_cd·√(A_c1/A_c0) ; 3·f_cd·A_c0)",
                      f"min({_v(plr['A_c0'], 'mm²')} × {_v(plr['f_cd'], 'N/mm²')} × √({_v(plr['A_c1'], 'mm²')} / {_v(plr['A_c0'], 'mm²')}) ; "
                      f"3 × {_v(plr['f_cd'], 'N/mm²')} × {_v(plr['A_c0'], 'mm²')}) / 10³", _r(plr["F_Rdu_kN"], "kN")),
                  _eta("η_loc", p.gamma_Q * pt["P"], plr["F_Rdu_kN"], "kN"),
                  _fx("u_1", "2·(a + b) + 2π·2d", f"2 × ({_d(a, 1)} + {_d(b, 1)}) + 4π × {_d(pl.d * 1e3, 1)}", _r(po["u1_mm"], "mm")),
                  _fx("k", "1 + √(200/d) ≤ 2", f"1 + √(200 / {_d(pl.d * 1e3, 1)})", _d(po["k"], 3)),
                  _fx("v_Rd,c", "max(0,18/γ_c·k·(100·ρ_l·f_ck)^(1/3) ; 0,035·k^1,5·√f_ck)",
                      f"max(0,18 / {_d(p.gamma_c)} × {_d(po['k'], 3)} × (100 × {_d(po['rho_l'], 5)} × {_d(fck)})^(1/3) ; "
                      f"0,035 × {_d(po['k'], 3)}^1,5 × √{_d(fck)})", _r(po["v_Rd_c"], "N/mm²")),
                  _fx("v_Ed", "β·γ_Q·P / (u_1·d)", f"{_d(p.beta_poinconnement)} × {_d(p.gamma_Q)} × {_v(pt['P'], 'kN')} × 10³ / ({_v(po['u1_mm'], 'mm')} × {_d(pl.d * 1e3, 1)})",
                      _r(po["v_Ed"], "N/mm²")),
                  _eta("η_p", po["v_Ed"], po["v_Rd_c"], "N/mm²")]
            if "v_Rd_max" in po:
                nu = 0.6 * (1 - fck / 250)                    # ν de l'EN 1992-1-1 (6.6N), écrit avec la valeur que le moteur a prise
                fx += [_fx("u_0", "2·(a + b)", f"2 × ({_d(a, 1)} + {_d(b, 1)})", _r(po["u0_mm"], "mm")),
                       _fx("v_Ed,0", "β·γ_Q·P / (u_0·d)", f"{_d(p.beta_poinconnement)} × {_d(p.gamma_Q)} × {_v(pt['P'], 'kN')} × 10³ / "
                           f"({_v(po['u0_mm'], 'mm')} × {_d(pl.d * 1e3, 1)})", _r(po["v_Ed0"], "N/mm²")),
                       _fx("v_Rd,max", "c·ν·f_cd, ν = 0,6·(1 − f_ck/250)", f"{_d(p.coef_vrd_max)} × {_d(nu, 4)} × {_v(plr['f_cd'], 'N/mm²')}",
                           _r(po["v_Rd_max"], "N/mm²")),
                       _eta("η_0", po["v_Ed0"], po["v_Rd_max"], "N/mm²")]
            detail += _groupe(f"Sous {C.esc(ref)} · a × b = {_d(a, 1)} × {_d(b, 1)} mm", fx)
    if pl.type == "hourdis":
        hyps = [f"p<sub>adm</sub> du fabricant : {_r(pl.p_loc_adm, 'N/mm²') if pl.p_loc_adm else 'non renseignée'} (P-5)",
                ("diffusion à 45° dans la chape de " + _r(pl.chape.epaisseur * 1e3, "mm")) if pl.chape else "sans chape",
                "P = charge de service × φ, sous l'aire efficace du répartiteur s'il y en a un"]
    else:
        hyps = [f"EN 1992-1-1 §6.7 et §6.4 · ELU, γ<sub>Q</sub> = {_d(p.gamma_Q)} sur la charge (P-9)",
                f"γ<sub>c</sub> = {_d(p.gamma_c)} · α<sub>cc</sub> = {_d(p.alpha_cc)} · β = {_d(p.beta_poinconnement)}",
                f"d = {_r(pl.d * 1e3, 'mm')} · ρ<sub>l</sub> = {_d(pl.rho_l, 5)}"]
    reste = tries[MAX_LIGNES:]
    t_reste = max((v["taux"] for v in reste if v["taux"] is not None), default=None)
    pied = (f'<div class="ck-autres">{len(vs)} vérification(s) locale(s), les plus défavorables d\'abord'
            + (f" ; les {len(reste)} autres restent sous {C.pc(C.pct(t_reste))} % (vue « Détail »)" if reste and t_reste is not None else "")
            + (f" ; le détail montre les {len(vus)} premières." if len(vs) > len(vus) else ".") + "</div>")
    return _fenetre("local", "Effets locaux", f"cas {C.esc(cas['nom'])} · sous chaque appui, après répartition", hyps, lignes, detail, pied)


def _fen_plateau(cas: Dict, calc: Dict) -> str:
    """Les plateaux : la bande efficace c = t·√(f/(3·p_Ed)) de chaque côté de l'empreinte (EN 1993-1-8 §6.2.5,
    distribution plastique), plafonnée au bord du plateau ; l'aire efficace qui en résulte."""
    p, res, index = calc["params"], calc["res"], calc["index"]
    vs = [v for v in res["verifications"] if v["famille"] == "plateau"]
    lignes = [_ligne_tab(v, "Rd", p.gamma_Q, nom="Plateau") for v in vs]
    patches = {pt["ref"]: pt for pt in res.get("patches", [])}
    detail = ""
    for v in vs[:4]:
        ref = v["objet"].split(" sous ", 1)[-1]
        pt, e, pla = patches.get(ref), index["empreintes"].get(ref), index["plateaux"].get(ref)
        if not (pt and e and pla):
            continue
        a, b = e.a * 1e3, e.b * 1e3
        if pla.rigide_declare:
            fx = [_fx("a_eff × b_eff", "A × B (plaque déclarée rigide)", "", f"{_d(pla.A, 1)} × {_d(pla.Bp, 1)} mm"),
                  _fx("p", "P / (A·B)", f"{_r(pt['P'])} × 10³ / ({_d(pla.A, 1)} × {_d(pla.Bp, 1)})", _r(pt["pression"], "N/mm²"))]
        else:
            pEd = p.gamma_Q * pt["P"] * 1e3 / (v["a_eff"] * v["b_eff"])
            c_brut = RP.bande_c(pla.t, pla.f, pEd)
            fx = [_fx("p_Ed", "γ_Q·P / (a_eff·b_eff)", f"{_d(p.gamma_Q)} × {_r(pt['P'])} × 10³ / ({_v(v['a_eff'], 'mm')} × {_v(v['b_eff'], 'mm')})", _r(pEd, "N/mm²")),
                  _fx("c", "t·√(f / (3·p_Ed))", f"{_d(pla.t, 1)} × √({_d(pla.f, 1)} / (3 × {_v(pEd, 'N/mm²')}))", _r(c_brut, "mm")),
                  _fx("c_max", "min((A − a)/2 ; (B − b)/2)", f"min(({_d(pla.A, 1)} − {_d(a, 1)})/2 ; ({_d(pla.Bp, 1)} − {_d(b, 1)})/2)", _r(v["R"], "mm")),
                  _fx("a_eff", "a + 2·min(c ; c_max)", f"{_d(a, 1)} + 2 × {_r(v['E'])}", _r(v["a_eff"], "mm")),
                  _fx("b_eff", "b + 2·min(c ; c_max)", f"{_d(b, 1)} + 2 × {_r(v['E'])}", _r(v["b_eff"], "mm")),
                  _fx("p", "P / (a_eff·b_eff)", f"{_r(pt['P'])} × 10³ / ({_v(v['a_eff'], 'mm')} × {_v(v['b_eff'], 'mm')})", _r(pt["pression"], "N/mm²"))]
        etat = "le plateau travaille en entier" if pt["complet"] else f"part efficace {C.pc(C.pct(v.get('part_efficace') or 0))} %"
        detail += _groupe(f"{C.esc(_court(pla.nom, 28))} sous {C.esc(ref)}", fx, f'<div class="hint">{etat}.</div>')
    hyps = [f"EN 1993-1-8 §6.2.5 · distribution plastique · γ<sub>Q</sub> = {_d(p.gamma_Q)} sur la pression",
            "f = f<sub>y</sub>/γ<sub>M0</sub> (acier) ou f<sub>m,d</sub> (bois, OSB), saisi avec sa source"]
    return _fenetre("plateau", "Plateaux sous les appuis", f"cas {C.esc(cas['nom'])} · aire efficace de chaque plateau", hyps, lignes, detail)


def _fen_filiere(cas: Dict, calc: Dict) -> str:
    """Les filières : poutre sur appuis élastiques (un ressort par élément), réactions, flexion à l'ELU."""
    p, res, index = calc["params"], calc["res"], calc["index"]
    vs = [v for v in res["verifications"] if v["famille"] == "filiere"]
    t_max = max((v["taux"] for v in vs if v["taux"] is not None), default=None)
    lignes = [_ligne_tab(v, "Rd", p.gamma_Q, gov=v["taux"] is not None and v["taux"] == t_max, nom="Filière") for v in vs]
    fils = {f.nom: f for f in index["filieres"].values()}
    detail = ""
    for v in vs:
        f = fils.get(v["objet"])
        if f is None or v["E"] is None:
            continue
        reac = [(C.esc(pt["ref"].split("→")[-1]), _r(pt["P"], "kN")) for pt in res.get("patches", []) if pt.get("repartiteur") == f.nom and pt["via"] == "filiere"]
        tab = _mini_table(("élément", "réaction"), reac) if reac else ""
        souleves = v.get("souleve") or []
        apres = tab + (f'<div class="hint">Élément(s) soulevé(s) : {C.esc(", ".join(souleves))} — le modèle linéaire n\'est plus exact.</div>' if souleves else "")
        detail += _groupe(f"{C.esc(f.nom)} · L = {_d(f.longueur, 3)} m", [
            _fx("M_max", "poutre sur ressorts k = 3·E·I·L/(a²·(L − a)²)", "", _r(v["E"], "kN·m")),
            _fx("M_Rd", "W_el·f_y / γ_M0", f"{_d(f.W_el, 1)} × 10³ × {_d(f.fy)} / {_d(f.gamma_M0)} / 10⁶", _r(v["R"], "kN·m")),
            _fx("M_Ed", "γ_Q·M_max", f"{_d(p.gamma_Q)} × {_r(v['E'])}", _r(p.gamma_Q * v["E"], "kN·m")),
            _eta("η", p.gamma_Q * v["E"], v["R"], "kN·m")], apres)
    hyps = ["poutre d'Euler sur appuis élastiques · un ressort par élément", f"ELU : γ<sub>Q</sub> = {_d(p.gamma_Q)} sur les charges de chantier",
            "poids propre de la filière négligé"]
    return _fenetre("filiere", "Filières", f"cas {C.esc(cas['nom'])} · réactions sur les éléments et flexion", hyps, lignes, detail)


def _fen_joint(cas: Dict, calc: Dict) -> str:
    p, res = calc["params"], calc["res"]
    vs = [v for v in res["verifications"] if v["famille"] == "joint"]
    lignes = [_ligne_tab(v, "Rd", p.gamma_Q, nom="Joint") for v in vs]
    fx = []
    for v in vs[:6]:
        fx.append(_fx("F", C.esc(v["objet"]), "", _r(v["E"], "kN")))
    if p.joint_adm_kN_m is not None and p.joint_longueur_m is not None:
        cap = p.joint_adm_kN_m * p.joint_longueur_m
        fx.append(_fx("R_j·ℓ_j", "", f"{_d(p.joint_adm_kN_m)} × {_d(p.joint_longueur_m)}", _r(cap, "kN")))
    hyps = ["parts de la figure C.1 de l'EN 1168 (P-15)",
            f"résistance du joint : {_r(p.joint_adm_kN_m, 'kN/m') if p.joint_adm_kN_m is not None else 'à transcrire'}"
            f" · longueur mobilisée : {_r(p.joint_longueur_m, 'm') if p.joint_longueur_m is not None else 'à transcrire'}"]
    return _fenetre("joint", "Joints longitudinaux", f"cas {C.esc(cas['nom'])} · effort transmis par chaque joint", hyps, lignes,
                    _groupe("Efforts transmis", fx))


# ------------------------------------------------------------------ le cadre Résultat
_DECIMALE = re.compile(r"(= ?-?\d+)\.(\d+)")
_POURCENT = re.compile(r"(\d)%")


def _txt(s: str) -> str:
    """Un texte du moteur écrit à la française pour l'écran : « y = 3.74 m » → « y = 3,74 m », « 70% » → « 70 % » (les
    références « E1.P2 » ne sont pas touchées)."""
    return _POURCENT.sub("\\1\u00a0%", _DECIMALE.sub(r"\1,\2", str(s)))


def _piece_famille(vs: List[Dict]) -> str:
    """La « pièce » d'une ligne de famille : l'objet gouvernant, et le nombre d'autres lignes."""
    g = vs[0]
    autres = len(vs) - 1
    o = _txt(g["objet"]).replace("bande gouvernante ", "")
    if " sous " in o:
        o = "sous " + o.split(" sous ", 1)[1]
    elif g["famille"] == "filiere":
        o = o.replace(" · ", " ")
    return C.esc(_court(o, 20)) + (f" +{autres}" if autres else "")


def _cle_famille(v: Dict):
    """L'ordre des lignes d'une famille, la plus défavorable d'abord : non conforme, puis taux, puis non vérifiée."""
    return (v["statut"] == "nok", v["taux"] if v["taux"] is not None and v["statut"] in CHIFFRES + ("info",) else -1.0,
            v["statut"] == "non_verifie")


def _calcul_de(fam: str, v: Dict, calc: Dict) -> str:
    """La fenêtre qu'ouvre une ligne : l'élément (hourdis), la dalle, ou la famille."""
    if fam in ("flexion", "tranchant", "fissuration"):
        if calc["plancher"].type == "hourdis":
            el = next((e for e in calc["res"]["elements"] if e["nom"] == v["objet"]), None)
            return f"el-{el['index']}" if el else ""
        return "dalle"
    return {"local": "local", "plateau": "plateau", "filiere": "filiere", "joint": "joint"}.get(fam, "")


def _lignes_familles(calc: Dict) -> Tuple[str, List[str]]:
    """Une ligne par famille de vérifications (les barres de l'écran d'origine), la plus défavorable de chacune."""
    res = calc["res"]
    g = res.get("gouverne") or {}
    out, fenetres = "", []
    for fam in ORDRE_FAM:
        vs = sorted((v for v in res["verifications"] if v["famille"] == fam), key=_cle_famille, reverse=True)
        if not vs:
            continue
        v = vs[0]
        cal = _calcul_de(fam, v, calc)
        fenetres.append(cal)
        texte = _texte_taux(v)
        gov = not texte and fam == g.get("famille") and v["objet"] == g.get("objet")
        out += _ligne_res(LIB_FAM[fam], _piece_famille(vs), v, C.pct(v["taux"]) if not texte else None, _niv(v), calcul=cal, gov=gov, texte=texte)
    return out, fenetres


def _lignes_appuis(etat: Dict, cas: Dict, calc: Dict) -> str:
    """Par engin, puis pour les étançons : chaque appui, sa charge, son répartiteur (un clic ouvre le sélecteur) et
    l'élément qui le porte avec son taux — le tableau « Appuis de E1 » de l'écran d'origine."""
    res, index = calc["res"], calc["index"]
    refs = list(index["empreintes"])
    hourdis = calc["plancher"].type == "hourdis"
    els = {e["nom"]: e for e in res.get("elements", [])}
    out = ""

    def ligne(ref, nom, charge, el_nom, taux):
        rp = cas["repartiteurs"].get(ref)
        piece = SANS if not rp else (_court(rp["nom"], 22) if rp["type"] == "plateau" else f"Filière {rp['id']}")
        el = els.get(el_nom)
        cal = (f"el-{el['index']}" if hourdis else "dalle") if el else ""
        niv = C.niveau(taux) if taux is not None else "c"
        nm = f"{nom} · {C.fr(charge, 0)}{C.NBSP}kN" + (f" → {el_nom}" if el_nom else "")
        return _ligne_res(nm, C.esc(piece), None, C.pct(taux) if taux is not None else None, niv, calcul=cal, pk=f"rep-{refs.index(ref)}",
                          texte="" if taux is not None else "—")
    for c in cas["charges"]:
        if c["type"] != "engin" or c["ref"] not in index["engins"]:
            continue
        lignes = OP.appuis_engin(cas, res, index, c["ref"])
        ts = [li["taux"] for li in lignes if li["taux"] is not None]
        t = max(ts) if ts else None
        pct = f'<span class="pct {C.niveau(t)}">{C.pc(C.pct(t))}{C.NBSP}%</span>' if t is not None else ""
        out += f'<div class="elgrp">Appuis de {C.esc(c["ref"])}{pct}</div>'
        out += "".join(ligne(li["ref"], li["appui"], li["charge"], li["element"], li["taux"]) for li in lignes)
    tours = [c for c in cas["charges"] if c["type"] == "ponctuel" and c["ref"] in index["empreintes"]]
    if tours:
        out += '<div class="elgrp">Étançons</div>'
        for c in tours:
            e = index["empreintes"][c["ref"]]
            el = next((x for x in res.get("elements", []) if x["y0"] <= e.y < x["y1"]), None)
            out += ligne(c["ref"], c["ref"], e.P * e.phi, el["nom"] if el else None, el["taux"] if el else None)
    return out


def _verdict(calc: Dict) -> str:
    res = calc.get("res")
    if not res or res["statut"] == "refus":
        titre, txt = (("Vérification impossible", "le moteur s'arrête avant tout calcul : voir ci-dessous") if res else
                      ("Calcul impossible", "le moteur n'a pas pu calculer ce cas : voir ci-dessous"))
        return (f'<div class="verdict"><span class="chip xl c"><small>η</small>—</span><div class="gov"><b>{titre}</b>'
                f'{txt}</div></div>')
    t = res["taux_max"]
    niv = "r" if res["statut"] == "nok" else C.niveau(t)
    p, pl = calc["params"], calc["plancher"]
    nature = getattr(pl, "nature", "service")
    sans = sorted((v for v in res["verifications"] if v["statut"] == "nok" and v["taux"] is None), key=lambda v: v["objet"])
    if sans and t <= 1:
        v = sans[0]
        return (f'<div class="verdict">{C.chip(C.pct(t), niv, xl=True)}<div class="gov"><b>{LIB_FAM.get(v["famille"], v["famille"])} · '
                f'{C.esc(v["objet"])}</b>répartiteur exigé : p<sub>adm</sub> non renseignée (P-5)'
                + (f" · {len(sans)} charges concernées" if len(sans) > 1 else "") + "</div></div>")
    g = res.get("gouverne")
    if not g:
        return (f'<div class="verdict">{C.chip(0, "p", xl=True)}<div class="gov"><b>Aucune charge posée</b>'
                f'poids propre et charges présentes seulement</div></div>')
    v = next(x for x in res["verifications"] if x["famille"] == g["famille"] and x["objet"] == g["objet"] and x["taux"] == g["taux"])
    _, val, lim = _comparaison(v, nature, p.gamma_Q)
    return (f'<div class="verdict">{C.chip(C.pct(t), niv, xl=True)}<div class="gov"><b>{LIB_FAM.get(g["famille"], g["famille"])} · '
            f'{C.esc(_txt(g["objet"]))}</b>{val} {"≤" if t <= 1 else "&gt;"} {lim}</div></div>')


OU_COMPLETER = [  # (morceau du message du moteur, où le site le règle) — les messages parlent des étapes de l'écran d'origine
    ("M_Rd et V_Rd", "Section 2 « Hourdis » : choisir un type du catalogue, ou saisir M<sub>Rd</sub> et V<sub>Rd</sub> de la fiche du fabricant."),
    ("charge par appui non renseignée", "Section 3 « Charges » : la charge de chaque appui de l'engin (tableau de stabilité du constructeur)."),
    ("Prédalle sans chape durcie", "Section 2 « Prédalle » : cocher « Chape durcie », ou ne garder que des stocks (section 3)."),
    ("hors du plancher", "Section 3 « Charges » : la position x, y de la charge, ou section 1 la portée ; un plateau décalé (Δx, Δy) "
                         "peut aussi sortir du plancher."),
    ("hors de la filière", "Section 4 « Répartiteurs » : la position ou la longueur de la filière, ou les appuis qu'elle porte."),
    ("dépasse le plancher", "Section 4 « Répartiteurs » : la position, la longueur ou l'orientation de la filière."),
    ("résistance f absente (fy/γM0", "Section 4 : un autre élément pour cette filière, ou section 5 une fiche de filière avec sa résistance f."),
    ("résistance de flexion f absente","Section 3 : un autre répartiteur sous cet appui, ou section 5 une fiche de plateau avec sa résistance f."),
    ("Empreinte plus grande que le plateau", "Section 3 : un plateau plus grand que la semelle de l'appui (défaut de l'écran Streamlit, qui "
                                             "s'arrêtait sur ce cas)."),
    ("même référence", "Section 3 : deux charges portent la même référence (fichier modifié à la main) : en retirer une."),
]


def _ou_completer(msg: str) -> str:
    """Où, sur le site, régler ce que le message du moteur demande."""
    return "".join(f'<span class="chp-ou"><b>Où compléter</b> · {t}</span>' for k, t in OU_COMPLETER if k in (msg or ""))


def _alertes(calc: Dict) -> str:
    """Ce qui doit se lire AVANT les lignes du résultat, sous le verdict : le défaut non conservatif P-3 et les hypothèses
    que le moteur déclare « attention » (P-8 : étais en place, report sur l'étage inférieur non modélisé)."""
    res = calc.get("res")
    if not res or res["statut"] == "refus":
        return ""
    return _alerte_p3(calc) + "".join(C.message(C.symb_math(_txt(v["methode"])), grave=True) for v in res["verifications"]
                                      if v["famille"] == "hypothese" and v["statut"] == "attention")


def _messages(cas: Dict, calc: Dict) -> str:
    res = calc.get("res")
    out = ""
    if not res:
        return C.message(C.symb_math(calc.get("erreur", "Calcul impossible")) + _ou_completer(calc.get("erreur", "")), grave=True)
    if res["statut"] == "refus":
        return C.message(C.symb_math(res["message"]) + _ou_completer(res["message"]), grave=True)
    nv = OP.non_verifies(res)
    if nv:
        out += C.message(f"{len(nv)} ligne(s) non vérifiée(s) : paramètre à transcrire ou donnée absente — voir le détail.")
    for v in res["verifications"]:
        if v["famille"] == "hypothese" and v["statut"] != "attention":
            out += C.message(C.symb_math(_txt(v["methode"])))
    if res["statut"] == "nok":
        out += C.message("Section 4 « Répartiteurs » : proposer une filière sous les appuis en défaut"
                         + (" ; vue « Balayage » : les positions admissibles de l'engin." if any(c["type"] == "engin" for c in cas["charges"]) else "."))
    prop = cas["site"].get("proposition")
    if prop and prop.get("empreinte") == _empreinte(cas):
        out += C.message(C.esc(OP.phrase_proposition(prop)), grave=not prop["trouve"])
    return out


def _fenetres(cas: Dict, calc: Dict) -> str:
    res = calc.get("res")
    if not res or res["statut"] == "refus":
        return ""
    out = ""
    if calc["plancher"].type == "hourdis":
        out += "".join(_fen_element(cas, calc, e) for e in res["elements"])
    else:
        out += _fen_dalle(cas, calc)
    fams = {v["famille"] for v in res["verifications"]}
    for fam, f in (("local", _fen_local), ("plateau", _fen_plateau), ("filiere", _fen_filiere), ("joint", _fen_joint)):
        if fam in fams:
            out += f(cas, calc)
    return out


def _vues(etat: Dict, cas: Dict, calc: Dict) -> List[Dict]:
    """Les vues de la toile : le plan à l'échelle, la coupe sous un appui, le balayage de l'engin, le détail des
    vérifications, la méthode et la bibliothèque (tableaux)."""
    vues = []
    res = calc.get("res")
    i = etat["actif"]
    if "index" in calc:
        res_d = res or {"elements": [], "patches": []}
        ctx = ET.contexte_dessin(etat, cas, res_d, calc["plancher"], calc["index"])
        L, B = calc["plancher"].L, calc["plancher"].B
        vues.append({"cle": "plan", "titre": "Plan du plancher", "court": "Plan", "svg": DP.plan(ctx, f"chp{i}p"),
                     "legende": f"{C.fr(L, 2)} × {C.fr(B, 2)} m"})
        refs = list(calc["index"]["empreintes"])
        if refs:
            cas_c = dict(cas, selection=_appui_coupe(cas, refs))
            cc = ET.contexte_coupe(etat, cas_c, res_d, calc["plancher"], calc["index"])
            ref = cas_c["selection"].split(":", 1)[1]
            vues.append({"cle": "coupe", "titre": f"Coupe sous {ref}", "court": "Coupe", "svg": DP.coupe(cc, f"chp{i}c"),
                         "legende": f"élément {cc['nom_element']} · {C.fr(cc['P'], 0)} kN · éléments voisins, couche de compression, diffusion"})
        ref_b = _balaye(cas)
        if ref_b and res and res["statut"] != "refus":
            b = OP.balayage(etat, cas, res, ref_b)
            if b:
                bb = b["balayage"]
                vues.append({"cle": "balayage", "titre": f"Balayage de {ref_b}", "court": "Balayage",
                             "svg": DP.vue_balayage(bb, b["x_actuel"], b["L"], OP.legende_balayage(bb), DP.plan(ctx, f"chp{i}b"), f"chp{i}s"),
                             "legende": "taux du plancher entier selon la position x du centre de l'engin"})
    if res and res["statut"] != "refus" and res["verifications"]:
        vues.append({"cle": "detail", "titre": "Détail des vérifications", "court": "Détail", "svg": DP.vue_detail(res, calc["plancher"].type),
                     "legende": "par élément, puis chaque vérification avec sa méthode"})
    vues.append({"cle": "methode", "titre": "Méthode et bibliothèque", "court": "Méthode",
                 "svg": DP.vue_methode(ET.gabarits(etat), ET.repartiteurs_disponibles(etat), ET.types_hourdis(etat)),
                 "legende": "les mêmes textes que la note · engins, plateaux, hourdis, profilés et leurs sources"})
    return vues


def resultat(etat: Dict) -> Dict:
    cas = _cas(etat)
    calc = _calculer(etat, cas)
    resultats = [calc.get("res") or {"statut": None} if k == etat["actif"] else (_calculer(etat, c).get("res") or {"statut": None})
                 for k, c in enumerate(etat["cas"])]
    res = calc.get("res")
    if res and res["statut"] != "refus":
        t = res["taux_max"]
        niv = "r" if res["statut"] == "nok" else C.niveau(t)
        fam, _ = _lignes_familles(calc)
        n_s = f'<span class="pct {niv}">{C.pc(C.pct(t))}{C.NBSP}%</span>'
        corps = (f'<div class="elgrp">Vérifications{n_s}</div>{fam}' if fam else "") + _lignes_appuis(etat, cas, calc)
        # l'écran d'origine passe tout le cas en « attention » quand une HYPOTHÈSE l'est (P-8 : étais en place, charges
        # reportées sur l'étage inférieur non modélisé) : le site le dit aussi, quel que soit le taux
        statut = C.statut("f" if niv == "p" and any(v["famille"] == "hypothese" and v["statut"] == "attention"
                                                      for v in res["verifications"]) else niv)
    else:
        corps, statut = "", C.statut("c" if res else "x")
    tot = OP.total_charges(cas)
    pied = (f'<div class="pfoot"><div class="mass"><span>Engins et étançons (charges de service)</span><b>{C.fr(tot, 0)}{C.NBSP}kN</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    return {"resultat": f'<div class="pbody">{_verdict(calc)}{_alertes(calc)}<div class="els">{corps}</div>{_messages(cas, calc)}</div>{pied}',
            "volets": _fenetres(cas, calc), "vues": _vues(etat, cas, calc), "phrase": C.esc(OP.resume_plancher(cas)),
            "statut_html": statut, "resume": resume_note(etat, resultats), "donnees": formulaire(etat, calc),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ la note PDF
# le générateur VALIDÉ de l'écran Streamlit (plancher/charges/note.py), deux pages par cas (api_web/note_cas.py) ; sa
# page 2 reçoit l'échelle du plan de la page 1 ; l'identification imprimée est celle de la note du site (celle de chaque
# cas dans l'état, « identification », venue de Streamlit, ne s'imprime pas et ne se montre plus)
NOTE = NC.Generateur("plancher.charges.note", "Charges sur plancher", lambda n1: (n1.echelle_plan,),
                     ("plancher/charges/note.py", "plancher/charges/peintre.py"))


def telecharger(etat: Dict, quoi: str, infos: Optional[Dict] = None) -> Optional[Dict]:
    """« note_pdf » : la note de calcul PDF de TOUTE la note (chaque cas, deux pages), faite par le générateur validé de
    l'écran Streamlit ; ``infos`` : le nom, le chantier, l'indice et la date de la note du site. Un cas que le moteur ne
    sait pas construire ou pas vérifier (l'écran le dit : « Donnée manquante : … », « Calcul impossible : … ») arrête la
    note, avec le même message."""
    if quoi != "note_pdf":
        return None
    for i, cas in enumerate(etat["cas"]):
        err = _calculer(etat, cas).get("erreur")
        if err:
            raise ValueError(f"Cas « {cas.get('nom') or f'Cas {i + 1}'} » : {err} — la note PDF n'est pas préparée.")
    return NC.fichier(NOTE, etat, infos)


# ------------------------------------------------------------------ les sélecteurs : répartiteur d'un appui, profilé d'une filière
def _options_selecteur(etat: Dict, cas: Dict, role: str) -> List[Tuple[str, str, float, str]]:
    """Les lignes d'un sélecteur : (nom affiché, valeur à saisir, masse en kg, propriétés) — triées par masse."""
    dispo = ET.repartiteurs_disponibles(etat)
    if role.startswith("rep-"):
        out = [(SANS, "aucun", 0.0, "l'appui charge le plancher directement")]
        noms = [(n, d) for n, d in dispo.items() if d.get("type") == "plateau"]
        courts = [_court(n, 26) for n, _ in noms]
        for (n, d), c in zip(noms, courts):
            f = f"f = {_d(d['f'])} MPa" if d.get("f") else ("déclaré rigide" if d.get("rigide_declare") else "f à renseigner")
            out.append((c if courts.count(c) == 1 else n, f"plateau:{n}", float(d.get("masse") or 0.0),
                        f"{C.fr(d.get('A', 0), 0)} × {C.fr(d.get('Bp', 0), 0)} × {C.fr(d.get('t', 0), 0)} mm · {f} · {C.esc(d.get('origine', ''))}"))
        for f in cas.get("filieres", []):
            d = dispo.get(f["nom"], {})
            out.append((f"Filière {f['id']} · {_court(f['nom'], 16)}", f"filiere:{f['id']}", float(d.get("masse") or 0.0) * f["longueur"],
                        f"{C.esc(f['nom'])} · L = {_d(f['longueur'], 3)} m · centrée en x = {_d(f['x'], 3)}, y = {_d(f['y'], 3)} m"))
        return sorted(out, key=lambda o: (o[2], o[0]))
    out = []
    for n, d in dispo.items():
        if d.get("type") == "filiere":
            sec = d.get("section", {})
            prop = (f"I<sub>y</sub> = {C.fr(BIB.profils()[n]['Iy'], 0)} cm⁴ · W<sub>el</sub> = {C.fr(BIB.profils()[n]['Wel'], 1)} cm³"
                    if d.get("origine") == "catalogue" else f"{C.esc(sec.get('profil') or 'rectangle sur champ')} · {C.esc(d.get('materiau', ''))}")
            out.append((n, n, float(d.get("masse") or 0.0), prop))
    return sorted(out, key=lambda o: (o[2], o[0]))


def _famille_profil(nom: str, d: Dict) -> str:
    if d.get("origine") != "catalogue":
        return "FICHE"
    return next((f for f in FAM_FIL if nom.startswith(f + " ")), "FICHE")


def _essayer(etat: Dict, chemin: List, valeur) -> Tuple[Dict, Dict]:
    """Le moteur sur une COPIE de la note où l'on a fait ce choix : (son résultat ou le refus, le cas copié)."""
    e = copy.deepcopy(etat)
    msg = modifier(e, chemin, valeur)
    if msg:
        return {"statut": "refus", "message": msg}, _cas(e)
    r = _calculer(e, _cas(e))
    return (r.get("res") or {"statut": "refus", "message": r.get("erreur", "")}), _cas(e)


def _taux_cible(res: Dict, cas: Dict, ref: Optional[str] = None, fid: Optional[str] = None) -> Tuple[Optional[float], bool, List[str]]:
    """Ce que le choix d'un répartiteur GOUVERNE, lu dans le résultat du moteur : les éléments que chargent les
    charges de l'appui ``ref`` (ou de la filière ``fid``) après répartition, l'effet local sous ces charges, la flexion
    du plateau ou de la filière. Rend (le plus grand taux, une ligne déclarée non conforme sans taux, les éléments)."""
    if ref is not None:
        rp = cas["repartiteurs"].get(ref) or {}
        fid = rp.get("id") if rp.get("type") == "filiere" else None
    if fid is not None:
        f = next((x for x in cas.get("filieres", []) if x["id"] == fid), None)
        nom = f"{fid} · {f['nom']}" if f else ""
        pts = [pt for pt in res.get("patches", []) if pt.get("origine") == nom]
        propres = [v for v in res["verifications"] if v["famille"] == "filiere" and v["objet"] == nom]
    else:
        pts = [pt for pt in res.get("patches", []) if pt["ref"] == ref]
        propres = [v for v in res["verifications"] if v["famille"] == "plateau" and v["objet"].endswith(" sous " + str(ref))]
    refs = {pt["ref"] for pt in pts}
    ts, nok, els = [], False, []
    for pt in pts:
        for e in res.get("elements", []):
            if min(e["y1"], pt["y"] + pt["b"] / 2) - max(e["y0"], pt["y"] - pt["b"] / 2) > 1e-9 or e["y0"] <= pt["y"] < e["y1"]:
                ts.append(e["taux"])
                if e["nom"] not in els:
                    els.append(e["nom"])
    locaux = [v for v in res["verifications"] if v["famille"] == "local" and v["objet"].split(" sous ", 1)[-1] in refs]
    for v in locaux + propres:
        nok = nok or v["statut"] == "nok"
        if v["taux"] is not None and v["statut"] in CHIFFRES:
            ts.append(v["taux"])
    return (max(ts) if ts else None), nok, els


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Le sélecteur du répartiteur d'un appui (« rep-<k> ») ou du profilé d'une filière (« fil-<i> ») : chaque ligne
    est CALCULÉE (le moteur sur une copie de la note), triée par masse ; le retenu (✓) et le plus léger qui ramène le
    système entier sous 100 % (★) sont marqués."""
    cas = _cas(etat)
    sorte, _, k = role.partition("-")
    k = int(k)
    dispo = ET.repartiteurs_disponibles(etat)
    if sorte == "rep":
        ref = _refs(etat, cas)[k]
        cur = _choix_repartiteur(cas, ref)
        chemin = ["appuis", k, "repartiteur"]
        opts = _options_selecteur(etat, cas, role)
        titre, sous, filtres, unite = (f"Répartiteur sous {C.esc(ref)}", "triés par masse · taux des éléments chargés et de l'effet local sous l'appui",
                                       "", "kg")
    else:
        f = cas["filieres"][k]
        cur = f["nom"]
        chemin = ["filieres", k, "nom"]
        fams = tuple(familles) if familles else FAM_FIL
        opts = [o for o in _options_selecteur(etat, cas, role) if _famille_profil(o[0], dispo.get(o[0], {})) in fams]
        titre, unite = f"Filière {C.esc(f['id'])} · profilé", "kg/m"
        sous = f"L = {_d(f['longueur'], 3)} m · triés par masse · taux de la filière et des éléments qu'elle charge"
        filtres = "".join(f'<button type="button" class="fchip{" on" if x in fams else ""}" data-famille="{x}">{LIB_FAM_FIL[x]}</button>' for x in FAM_FIL)
    lignes, textes = [], []
    for nom, val, masse, prop in opts:
        r, c2 = _essayer(etat, chemin, val)
        ok = r.get("statut") in CHIFFRES and r.get("taux_max") is not None
        t, nok, els = _taux_cible(r, c2, ref=ref) if sorte == "rep" and ok else (_taux_cible(r, c2, fid=cas["filieres"][k]["id"]) if ok else (None, False, []))
        ok = ok and t is not None
        niv = ("r" if nok else C.niveau(t)) if ok else "c"
        if not ok:
            textes.append("refusé" if r.get("statut") == "refus" else "—")
        g = r.get("gouverne") or {}
        prop2 = prop + (f" · charge {C.esc(', '.join(els))}" if els else "") + (
            f" · plancher entier {C.pc(C.pct(r['taux_max']))} % ({C.esc(g.get('famille', ''))} · {C.esc(_txt(g.get('objet', '')))})" if ok and g else
            (f" · {C.esc(r.get('message', ''))}" if r.get("statut") == "refus" else ""))
        lignes.append({"nom": nom, "col": C.fr(masse, 1), "taux": C.pct(min(t, 9.99)) if ok else 0, "niveau": niv, "cur": val == cur,
                       "opt": False, "prop": prop2, "_tient": ok and not nok and t <= 1.0})
    opt = next((li for li in lignes if li["_tient"]), None)
    if opt:
        opt["opt"] = True
    i_o = lignes.index(opt) if opt else 0
    i_c = next((i for i, li in enumerate(lignes) if li["cur"]), i_o)
    h = C.selecteur(role, titre, sous, filtres, lignes, max(0, min(i_o, i_c) - 4), unite, etoile="le plus léger qui tient")
    return _remplacer_puces(h, textes)


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """La ligne choisie dans un sélecteur : le répartiteur de l'appui, ou le profilé de la filière."""
    cas = _cas(etat)
    sorte, _, k = role.partition("-")
    k = int(k)
    opts = {o[0]: o[1] for o in _options_selecteur(etat, cas, role)}
    if nom not in opts:
        raise ValueError(f"choix inconnu : « {nom} »")
    chemin = ["appuis", k, "repartiteur"] if sorte == "rep" else ["filieres", k, "nom"]
    msg = modifier(etat, chemin, opts[nom])
    if msg:
        raise ValueError(msg)


# ------------------------------------------------------------------ des notes d'exemple (parcours, tests, captures)
def _faire(etat: Dict, *saisies) -> Dict:
    for chemin, valeur in saisies:
        msg = modifier(etat, chemin, valeur)
        if msg:
            raise ValueError(f"exemple : {'/'.join(map(str, chemin))} = {valeur!r} refusé : {msg}")
    return etat


def exemple(sorte: str = "hourdis") -> Dict:
    """Une note complète, construite par les MÊMES saisies que l'écran : « hourdis » (télescopique rotatif sur
    plateaux, étançon sur filière, file d'étais, stock, chape, annexe C), « dalle » (télescopique à roues sur dalle
    coulée en place), « predalle » (prédalle sans chape durcie : refus nommé P-7)."""
    e = nouvel_etat()
    gabs = list(ET.gabarits(e))
    if sorte == "hourdis":
        _faire(e, (["nom"], "C1 levage en rive"), (["plancher", "hourdis", "M_Rd"], "120"), (["plancher", "hourdis", "V_Rd"], "90"),
               (["plancher", "hourdis", "M_cr"], "60"), (["plancher", "hourdis", "p_loc_adm"], "1,5"),
               (["plancher", "hourdis", "parts_annexe_C"], "0:0.5 ; 1:0.25"),
               (["charges", "+engin"], gabs[0]))
        for j in range(4):
            _faire(e, (["charges", 0, "P", j], 45 + 5 * j))
        _faire(e, (["appuis", 0, "repartiteur"], "plateau:" + next(n for n, d in ET.repartiteurs_disponibles(e).items() if d.get("type") == "plateau")),
               (["appuis", 0, "idem"], "1"), (["charges", "+"], "ponctuel"), (["charges", 1, "x"], "6,7"), (["charges", 1, "y"], "4,8"),
               (["charges", 1, "P"], "40"), (["appuis", 4, "repartiteur"], "filiere:nouvelle"),
               (["charges", "+"], "surfacique"), (["charges", 2, "x0"], "0,3"), (["charges", 2, "y0"], "0,3"), (["charges", 2, "lx"], "1,2"),
               (["charges", 2, "ly"], "1,2"), (["charges", 2, "q"], "8"))
    elif sorte == "dalle":
        _faire(e, (["nom"], "C2 dalle"), (["plancher", "type"], "dalle"), (["plancher", "L"], "5"), (["plancher", "dalle", "As_inf"], "785"),
               (["charges", "+engin"], gabs[1]))
        for j in range(len(ET.gabarits(e)[gabs[1]]["appuis"])):
            _faire(e, (["charges", 0, "P", j], 20))
        _faire(e, (["appuis", 0, "repartiteur"], "plateau:Tôle acier S235 · 800 × 800 × 20"), (["appuis", 0, "idem"], "1"),
               (["charges", "+"], "lineaire"), (["charges", 1, "x"], "1,2"), (["charges", 1, "p"], "4"))
    elif sorte == "predalle":
        _faire(e, (["nom"], "C3 prédalle"), (["plancher", "type"], "predalle"), (["plancher", "predalle", "chape_durcie"], False),
               (["charges", "+"], "ponctuel"))
    else:
        raise ValueError(f"exemple inconnu : {sorte}")
    return e


def _parcours() -> List[Dict]:
    """Les requêtes propres au module (api_web/parcours.py) : les chemins que la note neuve n'ouvre pas — engins,
    plateaux, filières, sélecteurs, proposition, balayage, fiches, fichier de l'écran Streamlit."""
    h, d, p = exemple("hourdis"), exemple("dalle"), exemple("predalle")
    nok = copy.deepcopy(h)
    _faire(nok, (["appuis", 4, "repartiteur"], "aucun"), (["filieres", 0, "-"], "1"))
    prop = copy.deepcopy(nok)
    _faire(prop, (["proposition", "+"], "1"))
    fichier = {"nom": "c.json", "texte": ET.texte_json({k: v for k, v in h.items() if k != "site"})}
    out = [{"action": "calculer", "etat": x} for x in (h, d, p, nok, prop)]
    out += [{"action": "selecteur", "etat": h, "role": r} for r in ("rep-0", "rep-4", "fil-0")]
    out += [{"action": "selecteur", "etat": d, "role": "rep-0"}, {"action": "selecteur", "etat": h, "role": "fil-0", "familles": ["IPE"]}]
    out += [{"action": "choisir", "etat": h, "role": "rep-0", "nom": SANS}, {"action": "choisir", "etat": h, "role": "fil-0", "nom": "HEB 160"},
            {"action": "choisir", "etat": d, "role": "rep-1", "nom": "Tôle 800 × 800 × 20"}]
    out += [{"action": "modifier", "etat": prop, "chemin": ["proposition", "appliquer"], "valeur": "1"},
            {"action": "modifier", "etat": h, "chemin": ["note", "ouvrir"], "valeur": fichier},
            {"action": "modifier", "etat": h, "chemin": ["charges", 0, "dupliquer"], "valeur": "1"},
            {"action": "modifier", "etat": h, "chemin": ["charges", 0, "angle"], "valeur": "90"},
            {"action": "modifier", "etat": h, "chemin": ["selection"], "valeur": "rep:T1"},
            {"action": "modifier", "etat": d, "chemin": ["plancher", "type"], "valeur": "predalle"},
            {"action": "modifier", "etat": h, "chemin": ["plancher", "hourdis", "nature"], "valeur": "service"},
            {"action": "modifier", "etat": h, "chemin": ["plancher", "sens"], "valeur": "2"}]
    return out


def __getattr__(nom: str):
    """PARCOURS se construit à la demande (il calcule des notes d'exemple) : l'établi du navigateur n'en paie pas le prix
    à l'ouverture."""
    if nom == "PARCOURS":
        return _parcours()
    raise AttributeError(nom)

# -*- coding: utf-8 -*-
"""Le module « Catalogue des profilés » du site : une note de CONSULTATION (plusieurs recherches, chacune avec ses
familles, sa recherche, ses filtres numériques, son tri, ses colonnes, ses profilés personnalisés et sa comparaison),
rendue en fragments HTML : le TABLEAU des profilés dans la toile, la section dessinée à l'échelle, la comparaison, la
fiche du profilé choisi dans le cadre Résultat, les formules de ses propriétés dans la fenêtre de calcul.

Tout ce que l'écran décide vient de ``acier/profiles/consultation.py``, extrait mot pour mot de l'écran Streamlit
``modules/tableau_profiles.py`` (preuve : tests/test_catalogue_moteur.py) ; les nombres viennent du catalogue livré
(``acier/donnees/profiles.json``) ou du moteur de sections (``acier/profiles/familles.calculer``). Ce fichier MET EN
FORME : il ne calcule rien. Les formules de la fenêtre de calcul sont écrites avec les valeurs du profilé et leur
RÉSULTAT est celui du catalogue (tests/test_api_web_catalogue.py vérifie que chaque formule écrite redonne ce résultat).
"""
import copy
import functools
import math
import re
from typing import Dict, List, Optional, Sequence

from acier.profiles import catalogue as CAT
from acier.profiles import consultation as CO
from acier.profiles import familles as F

from . import commun as C
from . import dessin_catalogue as D

MODULE = "catalogue"
PAGE = "catalogue-profiles"

LIGNES = (100, 200, 500, 0)               # lignes du tableau affichées (0 : toutes)
LIGNES_DEFAUT = 200
GROUPES_FAMILLES = {"I/H": "Poutrelles I et H", "U": "Profilés U", "L": "Cornières", "creux": "Profils creux", "plein": "Pleins"}
NOMS_DIM = {"h": "hauteur", "b": "largeur", "tw": "épaisseur d'âme", "tf": "épaisseur de semelle", "t": "épaisseur",
            "d": "diamètre", "r": "congé", "r1": "congé", "r2": "arrondi"}
ACTIONS_BOUTON = {"effacer", "toutes", "aucune", "retablir", "+", "-"}


# ------------------------------------------------------------------ nombres et symboles
def symbole(c: str) -> str:
    """Le symbole d'une propriété, écrit comme dans une note (HTML) : « Wel,y » → W<sub>el,y</sub>, « cf/tf » →
    c<sub>f</sub>/t<sub>f</sub>. Lu dans le libellé du catalogue (CAT.LIBELLE), source unique."""
    lib = CAT.LIBELLE[c]
    if c in ("masse", "perimetre"):
        return lib
    if "/" in lib:
        return "/".join(symbole_txt(x) for x in lib.split("/"))
    if " " in lib:
        a, b = lib.split(" ", 1)
        return f"{C.esc(a)}<sub>{C.esc(b)}</sub>"
    return symbole_txt(lib)


def symbole_txt(lib: str) -> str:
    return C.esc(lib) if len(lib) == 1 else f"{C.esc(lib[0])}<sub>{C.esc(lib[1:])}</sub>"


def _valeur(v: Optional[float], c: str) -> str:
    """Une cellule du tableau : la valeur ARRONDIE de l'écran, écrite à décimales fixes (les colonnes s'alignent)."""
    if v is None:
        return "—"
    n = CAT.DECIMALES.get(c, 2)
    return C.fr(0.0 if abs(v) < 0.5 * 10 ** (-n) else v, n)


def _petite_unite(c: str) -> str:
    """L'unité d'une propriété en petit, après son symbole (les cases des colonnes : « I<sub>y</sub> cm⁴ »)."""
    u = CAT.UNITE.get(c, "")
    return f"<small>{C.esc(u)}</small>" if u else ""


def _nom_titre(nom: str) -> str:
    """Un nom de profilé pour un TITRE de la toile, que l'établi écrit en capitales : « L 100x100x10 » deviendrait
    « L 100X100X10 » ; le « x » entre deux chiffres s'y écrit « × » (la police l'affiche ainsi partout ailleurs)."""
    return re.sub(r"(?<=\d)x(?=\d)", "×", nom)


def _masse(m: float) -> str:
    return C.fr(m, 1 if abs(m) < 1000 else 0)


def _u(c: str) -> str:
    u = CAT.UNITE.get(c, "")
    return f" {u}" if u else ""


def _methode_txt(t: str) -> str:
    """La méthode de I_t écrite à la française (« pas 0.55 mm » → « pas 0,55 mm »)."""
    return re.sub(r"(\d)\.(\d)", r"\1,\2", t)


# ------------------------------------------------------------------ l'état : des recherches
def _recherche_neuve(nom: str) -> Dict:
    r = CO.reglages_initiaux()
    r.update({"nom": nom, "recherche": "", "lignes": LIGNES_DEFAUT, "fiche": None, "compare": [], "persos": [],
              "perso": {"forme": CO.formes_perso()[0], "dims": {f: CO.dims_defaut(f) for f in CO.formes_perso()}},
              "export": {"tableau": False, "fiche": False}})
    _resoudre(r)
    return r


def nouvel_etat() -> Dict:
    return {"version": 1, "recherches": [_recherche_neuve("Recherche 1")], "actif": 0}


def _nombre(x, quoi: str, vide: bool = False) -> Optional[float]:
    if x is None and vide:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{quoi} : nombre attendu ({x!r})"[:120])
    if not math.isfinite(x):                    # NaN, Infinity : JSON les laisse passer, un calcul n'en fait rien
        raise ValueError(f"{quoi} : nombre fini attendu ({x!r})")
    return float(x)


def _liste(r: Dict, cle: str) -> List:
    """Une liste lue dans un fichier ; autre chose (un texte, un nombre) est une note abîmée."""
    x = r[cle]
    if not isinstance(x, list):
        raise ValueError(f"« {cle} » illisible : une liste est attendue")
    return x


def _def_perso(d) -> Dict:
    """Un profilé personnalisé lu dans un fichier : sa forme et ses dimensions, contrôlées comme à la saisie."""
    if not isinstance(d, dict) or d.get("famille") not in CO.formes_perso() or not isinstance(d.get("dims"), dict):
        raise ValueError(f"profilé personnalisé illisible : {d!r}"[:160])
    fam = d["famille"]
    dims = {k: _nombre(d["dims"].get(k), f"dimension {k}") for k in F.FAMILLES[fam]["dims"]}
    CO.refus_du_site(fam, dims)
    CO.controle_ecran(fam, dims)
    return {"famille": fam, "dims": dims}


def _normaliser_recherche(r, i: int) -> Dict:
    if not isinstance(r, dict):
        raise ValueError(f"recherche {i + 1} illisible")
    out = _recherche_neuve(str(r.get("nom") or f"Recherche {i + 1}"))
    out["nom"] = str(r.get("nom") if r.get("nom") is not None else out["nom"])
    if "familles" in r:
        out["familles"] = CO.familles_ordonnees([f for f in _liste(r, "familles") if isinstance(f, str)])
    if "colonnes" in r:
        out["colonnes"] = CO.colonnes_ordonnees([c for c in _liste(r, "colonnes") if isinstance(c, str)])
    if "tri" in r:
        out["tri"] = r["tri"] if r["tri"] in CO.cles_colonnes() else ""
    for k in ("decroissant", "courants"):
        if k in r:
            out[k] = bool(r[k])
    if "recherche" in r:
        out["recherche"] = str(r["recherche"] or "")
    if "lignes" in r:
        out["lignes"] = r["lignes"] if r["lignes"] in LIGNES and not isinstance(r["lignes"], bool) else LIGNES_DEFAUT
    if r.get("filtres") is not None:
        fs = r["filtres"]
        if not isinstance(fs, list) or len(fs) > CO.N_FILTRES:
            raise ValueError("filtres illisibles")
        filtres = CO.filtres_vides()
        for j, f in enumerate(fs):
            if not isinstance(f, dict):
                raise ValueError(f"filtre {j + 1} illisible")
            cle = f.get("cle") or ""
            if cle and cle not in CO.cles_colonnes():
                raise ValueError(f"filtre {j + 1} : propriété inconnue {cle!r}")
            filtres[j] = {"cle": cle, "min": _nombre(f.get("min"), "borne", True), "max": _nombre(f.get("max"), "borne", True)}
        cles = [f["cle"] for f in filtres if f["cle"]]
        if len(set(cles)) != len(cles):
            raise ValueError("deux filtres sur la même propriété")
        out["filtres"] = filtres
    if r.get("persos") is not None:
        if not isinstance(r["persos"], list):
            raise ValueError("profilés personnalisés illisibles")
        out["persos"] = [_def_perso(d) for d in r["persos"]]
    pe = r.get("perso")
    if isinstance(pe, dict):
        if pe.get("forme") in CO.formes_perso():
            out["perso"]["forme"] = pe["forme"]
        for fam, dims in (pe.get("dims") or {}).items():
            if fam in CO.formes_perso() and isinstance(dims, dict):
                for k in F.FAMILLES[fam]["dims"]:      # le formulaire (rien n'est calculé) : une valeur hors
                    x = dims.get(k)                        # bornes garde la valeur proposée
                    if CO.refus_dimension(k, x) is None:
                        out["perso"]["dims"][fam][k] = float(x)
    ex = r.get("export")
    if isinstance(ex, dict):
        out["export"] = {k: bool(ex.get(k, False)) for k in ("tableau", "fiche")}
    out["fiche"] = r["fiche"] if isinstance(r.get("fiche"), str) else None
    if r.get("compare") is not None:
        if not isinstance(r["compare"], list):
            raise ValueError("comparaison illisible")
        out["compare"] = list(dict.fromkeys(n for n in r["compare"] if isinstance(n, str)))[-CO.MAX_COMPARES:]
    _resoudre(out)
    return out


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu : clés manquantes complétées ; ce qui ne se lit pas (un
    filtre sur une propriété inconnue, un profilé personnalisé impossible) lève une erreur claire."""
    rs = etat.get("recherches") if isinstance(etat, dict) else None
    if not rs:
        return nouvel_etat()
    if not isinstance(rs, list):
        raise ValueError("note illisible : « recherches » doit être une liste")
    out = {"version": 1, "recherches": [_normaliser_recherche(r, i) for i, r in enumerate(rs)]}
    a = etat.get("actif", 0)
    out["actif"] = min(max(0, int(a) if isinstance(a, (int, float)) and not isinstance(a, bool) else 0), len(out["recherches"]) - 1)
    return out


def _rech(etat: Dict) -> Dict:
    return etat["recherches"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les recherches de la note : en ouvrir une, en ajouter (réglages d'ouverture), copier l'ouverte, en supprimer une
    (jamais la dernière)."""
    rs = etat["recherches"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(rs) - 1)
    elif action == "ajouter":
        rs.append(_recherche_neuve(f"Recherche {len(rs) + 1}"))
        etat["actif"] = len(rs) - 1
    elif action == "dupliquer":
        r = copy.deepcopy(rs[etat["actif"]])
        r["nom"] = f"{r.get('nom') or 'Recherche'} (copie)"
        rs.append(r)
        etat["actif"] = len(rs) - 1
    elif action == "supprimer" and len(rs) > 1:
        i = min(max(0, int(index or 0)), len(rs) - 1)
        del rs[i]
        if etat["actif"] > i or etat["actif"] >= len(rs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la liste d'une recherche
@functools.lru_cache(maxsize=96)
def _perso_calcule(fam: str, dims: tuple) -> Dict:
    """Un profilé personnalisé calculé une fois par session du navigateur (un tube se calcule en 0,2 s)."""
    return CO.calculer_perso(fam, dict(dims))


def _cle_dims(d: Dict) -> tuple:
    return tuple((k, float(d["dims"][k])) for k in F.FAMILLES[d["famille"]]["dims"])


def persos_calcules(r: Dict) -> List[Dict]:
    return [dict(_perso_calcule(d["famille"], _cle_dims(d))) for d in r["persos"]]


def _vue(r: Dict) -> Dict:
    """Ce que la recherche affiche : la liste, les colonnes visibles, la fiche, les profilés comparés."""
    persos = persos_calcules(r)
    lst = CO.liste(r["familles"], r["recherche"], r["filtres"], r["courants"], persos, r["tri"], r["decroissant"])
    noms = [p["nom"] for p in lst]
    par = {p["nom"]: p for p in lst}
    fiche = CO.fiche_retenue(r["fiche"], noms)
    return {"persos": persos, "liste": lst, "noms": noms, "par": par, "cols": CO.colonnes_visibles(lst, r["colonnes"]),
            "p": par.get(fiche), "comp": [par[n] for n in CO.compares_retenus(r["compare"], noms)],
            "n_persos": CO.n_persos_affiches(r["recherche"], r["filtres"], persos)}


def _resoudre(r: Dict) -> None:
    """Ce que l'écran refait à chaque exécution : la fiche hors de la liste passe au premier profilé, les profilés
    comparés sortis du tableau quittent la comparaison."""
    v = _vue(r)
    r["fiche"] = CO.fiche_retenue(r["fiche"], v["noms"])
    r["compare"] = CO.compares_retenus(r["compare"], v["noms"])


# ------------------------------------------------------------------ la saisie
class _Refus(Exception):
    """Une saisie refusée, avec son message (en français) : la recherche reste intacte."""


def _indice(c, n: int) -> int:
    """Le rang d'une ligne (filtre, comparé, personnalisé) dans un chemin, contrôlé : 0 ≤ i < n."""
    if isinstance(c, int) and not isinstance(c, bool) and 0 <= c < n:
        return c
    raise _Refus("Saisie inconnue : cette ligne n'existe pas (ou plus)")


def _vrai(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "vrai", "oui", "on")
    return bool(v)


def _appliquer(r: Dict, chemin: List, valeur) -> Optional[str]:
    k0 = chemin[0]
    cles = CO.cles_colonnes()
    if chemin == ["nom"]:
        r["nom"] = str(valeur or "")
    elif chemin == ["recherche"]:
        r["recherche"] = str(valeur or "")
    elif chemin == ["courants"]:
        r["courants"] = _vrai(valeur)
    elif chemin == ["ordre"]:
        if valeur not in ("croissant", "decroissant"):
            return f"Ordre inconnu : « {valeur} »"
        r["decroissant"] = valeur == "decroissant"
    elif chemin == ["tri"]:
        if valeur not in [""] + cles:
            return f"Propriété inconnue : « {valeur} »"
        r["tri"] = valeur
    elif chemin == ["lignes"]:
        x = C.lire_nombre(valeur)
        if x is None or int(x) not in LIGNES:
            return f"Nombre de lignes inconnu : « {valeur} »"
        r["lignes"] = int(x)
    elif k0 == "familles":
        if chemin[1] == "toutes":
            r["familles"] = CO.toutes_familles()
        elif chemin[1] == "aucune":
            r["familles"] = []
        elif chemin[1] in CO.toutes_familles():
            sel = set(r["familles"]) - {chemin[1]} | ({chemin[1]} if _vrai(valeur) else set())
            r["familles"] = CO.familles_ordonnees(sel)
        else:
            return f"Famille inconnue : « {chemin[1]} »"
    elif k0 == "colonnes":
        if chemin[1] not in cles:
            return f"Colonne inconnue : « {chemin[1]} »"
        sel = set(r["colonnes"]) - {chemin[1]} | ({chemin[1]} if _vrai(valeur) else set())
        r["colonnes"] = CO.colonnes_ordonnees(sel)
    elif chemin == ["reglages", "retablir"]:
        r.update(CO.reglages_livres())
    elif k0 == "filtres":
        if chemin[1] == "effacer":
            r["filtres"] = CO.filtres_vides()
            return None
        i = _indice(chemin[1], CO.N_FILTRES)
        f = r["filtres"][i]
        quoi = chemin[2]
        if quoi == "-":
            r["filtres"][i] = {"cle": "", "min": None, "max": None}
        elif quoi == "cle":
            if valeur not in [""] + cles:
                return f"Propriété inconnue : « {valeur} »"
            autre = next((j for j, g in enumerate(r["filtres"]) if j != i and valeur and g["cle"] == valeur), None)
            if autre is not None:
                return (f"{CO.entete(valeur)} est déjà filtrée (filtre {autre + 1}) : régler ses deux bornes sur la même ligne "
                        f"(l'écran Streamlit ne garderait que la seconde)")
            f["cle"] = valeur
        elif quoi in ("min", "max"):
            if valeur is None or (isinstance(valeur, str) and not valeur.strip()):
                f[quoi] = None
            else:
                x = C.lire_nombre(valeur)
                if x is None:
                    return f"Nombre illisible : « {valeur} »"
                f[quoi] = x
        else:
            return "Saisie inconnue"
    elif chemin == ["fiche"]:
        if valeur not in _vue(r)["noms"]:
            return f"« {valeur} » n'est pas dans le tableau"
        r["fiche"] = valeur
    elif k0 == "perso":
        pe = r["perso"]
        if chemin == ["perso", "forme"]:
            if valeur not in CO.formes_perso():
                return f"Forme inconnue : « {valeur} »"
            pe["forme"] = valeur
        elif chemin == ["perso", "+"]:
            fam = pe["forme"]
            dims = dict(pe["dims"][fam])
            try:
                CO.refus_du_site(fam, dims)
                p = _perso_calcule(fam, tuple((k, float(dims[k])) for k in F.FAMILLES[fam]["dims"]))
            except ValueError as e:
                return f"Dimensions impossibles : {e}"
            r["persos"] = [d for d in r["persos"] if CO.nom_perso(d["famille"], d["dims"]) != p["nom"]] + [{"famille": fam, "dims": dims}]
            r["fiche"] = p["nom"]
        elif len(chemin) == 4 and chemin[1] == "dims" and chemin[2] in CO.formes_perso() and chemin[3] in F.FAMILLES[chemin[2]]["dims"]:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            refus = CO.refus_dimension(chemin[3], x)
            if refus:
                return refus
            pe["dims"][chemin[2]][chemin[3]] = x
        else:
            return "Saisie inconnue"
    elif k0 == "export":
        if chemin[1:] in (["tableau"], ["fiche"]):
            r["export"][chemin[1]] = _vrai(valeur)
        elif chemin[1:] in (["texte_tableau"], ["texte_fiche"]):
            return None                                     # le texte se copie ; le retoucher ne change rien
        else:
            return "Saisie inconnue"
    elif k0 == "persos" and len(chemin) == 3 and chemin[2] == "-":
        del r["persos"][_indice(chemin[1], len(r["persos"]))]
    elif k0 == "compare":
        if chemin[1] == "effacer":
            r["compare"] = []
        elif chemin[1] == "+":
            n = r["fiche"]
            if n is None:
                return "Aucun profilé ouvert"
            if n in r["compare"]:
                return f"{n} est déjà comparé"
            if len(r["compare"]) >= CO.MAX_COMPARES:
                return f"Au plus {CO.MAX_COMPARES} profilés à comparer"
            r["compare"] = r["compare"] + [n]
        elif len(chemin) == 3 and chemin[2] == "-":
            del r["compare"][_indice(chemin[1], len(r["compare"]))]
        else:
            return "Saisie inconnue"
    else:
        return "Saisie inconnue"
    return None


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la recherche ouverte ; les boutons (toutes, aucune, effacer, rétablir, ajouter, retirer)
    passent par le même chemin. Une saisie refusée rend un message et la recherche reste intacte. Après chaque saisie,
    la règle de l'écran : fiche et comparaison suivent le tableau."""
    r = _rech(etat)
    avant = copy.deepcopy(r)
    if not isinstance(chemin, (list, tuple)) or not chemin:
        return "Saisie inconnue"
    # « 0 », « 12 » → rang ; « ² » est un chiffre pour str.isdigit, pas pour int : chiffres ASCII seulement
    chemin = [int(c) if isinstance(c, str) and c.isascii() and c.isdigit() else c for c in chemin]
    try:
        msg = _appliquer(r, chemin, valeur)
        if msg is None:
            _resoudre(r)
    except _Refus as e:
        msg = str(e)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        msg = "Saisie inconnue"                         # un chemin que l'écran n'envoie pas : pas de jargon Python
    if msg is not None:
        etat["recherches"][etat["actif"]] = avant
    return msg


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Un clic dans un sélecteur : « fiche » ouvre le profilé ; « compare » l'ajoute à la comparaison (au-delà de cinq,
    le plus ancien laisse sa place), ou l'en RETIRE s'il y était déjà (la ligne cochée ✓ se décoche, comme une case
    de la liste à choix multiples de l'écran)."""
    r = _rech(etat)
    if nom not in _vue(r)["noms"]:
        return
    if role == "compare":
        if nom in r["compare"]:
            r["compare"] = [n for n in r["compare"] if n != nom]
        else:
            r["compare"] = (r["compare"] + [nom])[-CO.MAX_COMPARES:]
    else:
        r["fiche"] = nom
    _resoudre(r)


# ------------------------------------------------------------------ le cadre Données
def _bouton(chemin: Sequence, texte: str, icone: str = "plus", valeur: str = "1") -> str:
    """Un bouton d'action du cadre Données (la classe « addrow » de la poutre) : il envoie ``valeur`` au chemin."""
    return (f'<button type="button" class="addrow" data-chemin="{C._attr_chemin(chemin)}" data-valeur="{C.esc(valeur)}">'
            f'{C.icon(icone)}{texte}</button>')


def _borne(i: int, quoi: str, valeur: Optional[float], unite: str) -> str:
    """Une borne de filtre : le champ numérique commun (unité, alignement, clavier décimal), mais FACULTATIF. L'établi
    refuse un champ numérique vide, alors qu'une borne vidée veut dire « pas de borne » (le champ ≥ (min) de l'écran
    se vide) : le champ part donc comme TEXTE, et le module le lit (C.lire_nombre), refuse ce qui n'est pas un nombre
    et prend le vide pour « aucune ». Besoin commun signalé : un C.num facultatif."""
    h = C.num(["filtres", i, quoi], valeur, unite, dec=6)
    return h.replace('data-sorte="nombre"', 'data-sorte="texte" placeholder="—"', 1)


def _bornes_actives(r: Dict) -> bool:
    """Au moins une borne porte sur une propriété choisie : la liste est alors « ce qui satisfait les filtres »."""
    return any(f["cle"] and (f["min"] is not None or f["max"] is not None) for f in r["filtres"])


def _moins(chemin: Sequence, titre: str) -> str:
    return (f'<button type="button" class="moins" data-chemin="{C._attr_chemin(chemin)}" data-valeur="1" title="{C.esc(titre)}">'
            f'{C.icon("x")}</button>')


def _resume_filtres(filtres: Sequence[Dict]) -> str:
    out = []
    for f in filtres:
        if not f["cle"] or (f["min"] is None and f["max"] is None):
            continue
        s = symbole(f["cle"])
        if f["min"] is not None and f["max"] is not None:
            out.append(f"{C.nb(f['min'], 6)} ≤ {s} ≤ {C.nb(f['max'], 6)}")
        elif f["min"] is not None:
            out.append(f"{s} ≥ {C.nb(f['min'], 6)}")
        else:
            out.append(f"{s} ≤ {C.nb(f['max'], 6)}")
    return " · ".join(out)


def formulaire(etat: Dict, v: Dict) -> str:
    r = _rech(etat)
    L = C.ligne
    cles = CO.cles_colonnes()
    # 1 · recherche et tri
    s1 = (L("Nom", C.texte(["nom"], r["nom"], "ex. Poutres R+1"), full=True)
          + L("Rechercher", C.texte(["recherche"], r["recherche"], "IPE 200 · 100x100 · HEB…"), full=True)
          + L("Trier par", C.choix(["tri"], r["tri"], [("", "ordre du catalogue")] + [(c, CO.entete(c)) for c in cles]), full=True)
          + L("Ordre", C.segments(["ordre"], "decroissant" if r["decroissant"] else "croissant",
                                  [("croissant", "Croissant"), ("decroissant", "Décroissant")]), full=True)
          + C.aide("La recherche ignore la casse et les espaces : « ipe200 », « 100x100 », « HEB 3 »."))
    resume1 = " · ".join(x for x in (f"« {C.esc(r['recherche'])} »" if r["recherche"].strip() else "",
                                     f"trié par {symbole(r['tri'])}{' ↓' if r['decroissant'] else ' ↑'}" if r["tri"] else "ordre du catalogue") if x)
    # 2 · familles
    comptes = dict(CAT.familles_presentes())
    s2 = ""
    for groupe, fams in CAT.groupes_presents():
        s2 += f'<div class="grp">{C.esc(GROUPES_FAMILLES.get(groupe, groupe))}</div><div class="cat-puces">'
        s2 += "".join(C.case(["familles", f], f in r["familles"], f'{C.esc(f)}<small>{comptes.get(f, 0)}</small>') for f in fams)
        s2 += "</div>"
    s2 += (C.case(["courants"], r["courants"], "Séries courantes seulement") + C.aide(C.esc(CO.AIDE_COURANTS))
           + f'<div class="cat-actions">{_bouton(["familles", "toutes"], "Toutes les familles", "check")}'
           f'{_bouton(["familles", "aucune"], "Aucune", "x")}</div>')
    nf = len(r["familles"])
    resume2 = f"{nf} famille{'s' if nf > 1 else ''}" + (" · séries courantes" if r["courants"] else "")
    # 3 · filtres numériques
    s3 = ""
    opts = [("", "— propriété —")] + [(c, CO.entete(c)) for c in cles]
    for i, f in enumerate(r["filtres"]):
        u = CAT.UNITE.get(f["cle"], "") if f["cle"] else ""
        vide = not f["cle"] and f["min"] is None and f["max"] is None
        s3 += (f'<div class="cat-filtre"><div class="cat-f1"><span class="cat-fn">{i + 1}</span>{C.choix(["filtres", i, "cle"], f["cle"], opts)}'
               f'{"<span class=" + chr(34) + "moins" + chr(34) + "></span>" if vide else _moins(["filtres", i, "-"], "Effacer ce filtre")}</div>'
               f'<div class="cat-bornes"><span class="cat-b">min ≥</span>{_borne(i, "min", f["min"], u)}'
               f'<span class="cat-b">max ≤</span>{_borne(i, "max", f["max"], u)}</div></div>')
    s3 += (C.aide("Par exemple I<sub>y</sub> ≥ 1 500 cm⁴, masse ≤ 30 kg/m. Une borne vide ne filtre pas (effacer le "
                  "nombre retire la borne) ; un profilé sans "
                  "la valeur est écarté dès qu'une borne porte sur elle ; la borne s'applique à la valeur exacte, pas à "
                  "l'arrondi du tableau. Une propriété ne se filtre qu'une fois.")
           + f'<div class="cat-actions">{_bouton(["filtres", "effacer"], "Effacer les filtres", "x")}</div>')
    leger = _plus_leger(v["liste"]) if _bornes_actives(r) else None
    if leger is not None:
        ouvert = v["p"] is not None and v["p"]["nom"] == leger["nom"]
        s3 += (f'<div class="cat-leger">{C.icon("star")}<span>Le plus léger qui satisfait les filtres : <b>{C.esc(leger["nom"])}</b> · '
               f'{_masse(leger["masse"])}{C.NBSP}kg/m</span></div>'
               + ("" if ouvert else _bouton(["fiche"], f"Ouvrir la fiche de {C.esc(leger['nom'])}", "star", valeur=leger["nom"])))
    resume3 = _resume_filtres(r["filtres"]) or "aucun"
    # 4 · colonnes et lignes du tableau
    s4 = ""
    for g in CAT.GROUPES_COLONNES:
        s4 += f'<div class="grp">{C.esc(g)}</div><div class="cat-puces cat-cols">'
        s4 += "".join(C.case(["colonnes", c], c in r["colonnes"], symbole(c) + _petite_unite(c)) for c in CAT.colonnes_du_groupe(g))
        s4 += "</div>"
    s4 += (C.aide("Les colonnes vides pour les familles affichées sont masquées d'elles-mêmes ; la définition d'une "
                  "propriété s'affiche au survol de son en-tête et dans la fiche.")
           + L("Lignes", C.segments(["lignes"], str(r["lignes"]), [(str(n), str(n) if n else "Toutes") for n in LIGNES]), full=True)
           + f'<div class="cat-actions">{_bouton(["reglages", "retablir"], "Rétablir les réglages livrés", "history")}</div>'
           + C.aide("Familles, colonnes, tri et séries courantes reviennent aux défauts livrés ; les filtres sont effacés."))
    nc = len(r["colonnes"])
    resume4 = f"{nc} sur {len(cles)} · {r['lignes'] or 'toutes les'} lignes"
    # 5 · comparaison
    s5 = ""
    for i, n in enumerate(r["compare"]):
        s5 += (f'<div class="cat-cmp"><span class="cat-fn">{i + 1}</span><span class="nm">{C.esc(n)}</span>'
               f'{_moins(["compare", i, "-"], "Retirer de la comparaison")}</div>')
    fiche = r["fiche"]
    if fiche and fiche not in r["compare"] and len(r["compare"]) < CO.MAX_COMPARES:
        s5 += _bouton(["compare", "+"], f"Comparer {C.esc(fiche)}")
    if r["compare"]:
        s5 += f'<div class="cat-actions">{_bouton(["compare", "effacer"], "Vider la comparaison", "x")}</div>'
    s5 += C.aide((C.esc(CO.AIDE_COMPARER) + " " if len(r["compare"]) < 2 else "")
                 + f"Jusqu'à {CO.MAX_COMPARES} profilés du tableau ; la ligne « Comparer » du résultat en ajoute ou en "
                 "retire aussi. Dès deux profilés, la toile montre leurs sections à la même échelle et leurs propriétés "
                 "côte à côte.")
    resume5 = f"{len(r['compare'])} profilé{'s' if len(r['compare']) > 1 else ''}" if r["compare"] else "aucun"
    # 6 · profilé personnalisé
    pe = r["perso"]
    fam = pe["forme"]
    s6 = (L("Forme", C.choix(["perso", "forme"], fam, [(f, F.FAMILLES[f]["libelle"]) for f in CO.formes_perso()]), full=True)
          + C.aide(C.esc(F.FAMILLES[fam]["description"])))
    for k in F.FAMILLES[fam]["dims"]:
        s6 += L(NOMS_DIM.get(k, k), C.num(["perso", "dims", fam, k], pe["dims"][fam][k], "mm", mini=CO.DIM_MIN), symbole_txt(k))
    for av in CO.avertissements_perso(fam, pe["dims"][fam]):
        s6 += f'<div class="cat-alerte">{C.icon("info")}<span>{C.symb_math(av[0].upper() + av[1:])}.</span></div>'
    s6 += _bouton(["perso", "+"], "Calculer et ajouter au tableau")
    if r["persos"]:
        s6 += '<div class="grp">Au tableau</div>'
        for i, d in enumerate(r["persos"]):
            nom = CO.nom_perso(d["famille"], d["dims"])
            on = nom == fiche
            s6 += (f'<div class="cat-cmp"><button type="button" class="cat-ouvrir{" on" if on else ""}" data-chemin="fiche" data-sorte="choix" '
                   f'data-valeur="{C.esc(nom)}" title="Ouvrir la fiche">{C.esc(nom)}</button>{_moins(["persos", i, "-"], "Retirer du tableau")}</div>')
    resume6 = f"{len(r['persos'])} au tableau" if r["persos"] else "PRS, caisson, tube, plat, rond"
    # 7 · export : le tableau et la fiche en TEXTE à copier (le téléchargement d'un fichier attend une brique commune)
    ex = r["export"]
    s7 = C.case(["export", "tableau"], ex["tableau"], "Le tableau en texte, pour Excel")
    if ex["tableau"]:
        n = len(v["liste"])
        s7 += (C.zone_texte(["export", "texte_tableau"], CO.texte_tableau(CO.lignes_tableau(v["liste"], v["cols"]), "\t"), 8)
               + C.aide(f"Les {n} ligne{'s' if n > 1 else ''} et les {len(v['cols']) + 2} colonnes du tableau, séparées par des tabulations, "
                        "virgule décimale : cliquer dans la zone, Ctrl+A puis Ctrl+C, coller dans Excel. Les mêmes cellules que "
                        "l'export CSV de Streamlit (« profiles.csv »)."))
    s7 += C.case(["export", "fiche"], ex["fiche"], "La fiche du profilé ouvert (JSON)")
    if ex["fiche"] and v["p"] is not None:
        s7 += (C.zone_texte(["export", "texte_fiche"], CO.fiche_json(v["p"]), 8)
               + C.aide(f"L'enregistrement complet de {C.esc(v['p']['nom'])}, celui du fichier « {C.esc(CO.nom_fichier_fiche(v['p']))} » "
                        "de Streamlit."))
    s7 += C.aide("Le téléchargement direct d'un fichier .csv ou .json n'est pas encore offert par le site : copier le texte. "
                 "Retoucher le texte ici ne change rien au catalogue.")
    resume7 = " · ".join(x for x, on in (("tableau", ex["tableau"]), ("fiche", ex["fiche"])) if on) or "tableau, fiche"
    return (C.section(1, "Recherche et tri", resume1, s1, True, "re")
            + C.section(2, "Familles", resume2, s2, True, "fa")
            + C.section(3, "Filtres numériques", resume3, s3, True, "fi")
            + C.section(4, "Colonnes du tableau", resume4, s4, False, "co")
            + C.section(5, "Comparaison", resume5, s5, bool(r["compare"]), "cp")
            + C.section(6, "Profilé personnalisé", resume6, s6, True, "pe")
            + C.section(7, "Export", resume7, f'<div class="cat-export">{s7}</div>', False, "ex"))


# ------------------------------------------------------------------ la toile : le tableau, la section, la comparaison
def _entetes(cols: Sequence[str], tri: str, desc: bool) -> str:
    th = '<th class="nm">Profilé</th><th class="fa">Famille</th>'
    for c in cols:
        fl = f'<i>{"↓" if desc else "↑"}</i>' if c == tri else ""
        aide = CAT.AIDE_COLONNES.get(c, "")
        titre = f' title="{C.esc(CO.entete(c) + (" — " + aide if aide else ""))}"'
        th += (f'<th class="n{" tri" if c == tri else ""}"{titre}><span class="s">{symbole(c)}{fl}</span>'
               f'<span class="u">{C.esc(CAT.UNITE.get(c, "")) or "&nbsp;"}</span></th>')
    return f"<thead><tr>{th}</tr></thead>"


def _ligne_tab(p: Dict, ligne: Dict, cols: Sequence[str], tri: str, fiche: Optional[str], compares: Sequence[str]) -> str:
    cls = " ".join(x for x in ("on" if p["nom"] == fiche else "", "cmp" if p["nom"] in compares else "", "perso" if p.get("perso") else "") if x)
    tag = '<span class="cat-tag">perso</span>' if p.get("perso") else ""
    mk = f'<span class="cat-k">{compares.index(p["nom"]) + 1}</span>' if p["nom"] in compares else ""
    tds = "".join(f'<td{" class=" + chr(34) + "tri" + chr(34) if c == tri else ""}>{_valeur(ligne[CO.entete(c)], c)}</td>' for c in cols)
    return (f'<tr{" class=" + chr(34) + cls + chr(34) if cls else ""}><td class="nm">{C.esc(p["nom"])}{tag}{mk}</td>'
            f'<td class="fa">{C.esc(p["famille"])}</td>{tds}</tr>')


def tableau(r: Dict, v: Dict) -> str:
    """La vue principale : le tableau des profilés (en-têtes avec unités, lignes alternées, colonne triée marquée,
    profilé ouvert surligné) ; au plus ``lignes`` lignes, et le profilé ouvert toujours visible. Les colonnes et leurs
    valeurs sont celles de l'écran (CO.colonnes_visibles, CO.lignes_tableau) ; seul leur ORDRE d'affichage change :
    la colonne de tri vient juste après le nom (même si elle n'est pas cochée), pour qu'on la voie sans défiler."""
    lst, cols = v["liste"], colonnes_affichees(r, v)
    if not lst:
        return (f'<div class="cat-tab vide"><div class="cat-vide">{C.icon("search")}<b>{C.esc(CO.MESSAGE_VIDE)}</b>'
                f'<span>{C.esc(_raisons_vide(r))}</span></div></div>')
    n = len(lst)
    cap = r["lignes"] or n
    montres = list(range(min(cap, n)))
    fiche = v["p"]["nom"] if v["p"] else None
    i_f = v["noms"].index(fiche) if fiche in v["noms"] else -1
    lignes = CO.lignes_tableau(lst, cols)
    corps = "".join(_ligne_tab(lst[i], lignes[i], cols, r["tri"], fiche, r["compare"]) for i in montres)
    if i_f >= len(montres):
        corps += (f'<tr class="saut"><td colspan="{len(cols) + 2}">… {i_f - len(montres)} ligne{"s" if i_f - len(montres) > 1 else ""} …</td></tr>'
                  + _ligne_tab(lst[i_f], lignes[i_f], cols, r["tri"], fiche, r["compare"]))
    if len(montres) < n:
        pied = (f"Lignes 1 à {len(montres)} sur {n}" + (f" (et {C.esc(fiche)}, ouvert)" if i_f >= len(montres) else "")
                + " — les autres : affiner la recherche, ou « Lignes : toutes » (Données, section 4)")
    else:
        pied = f"{n} ligne{'s' if n > 1 else ''}"
    if len(cols) > 7:
        pied += f" · {len(cols) + 2} colonnes : faire défiler en largeur, ou replier Données et Résultat"
    if r["tri"] and r["tri"] not in v["cols"] and r["tri"] in cols:
        pied += f" · la colonne de tri {symbole(r['tri'])} est montrée, sans être cochée (Données, section 4)"
    pied += " · le profilé ouvert est surligné, il se choisit au Résultat (ligne « Profilé »)"
    return (f'<div class="cat-tab"><div class="cat-defile"><table class="cat-t">{_entetes(cols, r["tri"], r["decroissant"])}'
            f'<tbody>{corps}</tbody></table></div><div class="cat-tpied">{pied}</div></div>')


def colonnes_affichees(r: Dict, v: Dict) -> List[str]:
    """Les colonnes de la vue « Tableau » : celles de l'écran (cochées et non vides), la colonne de tri en tête."""
    cols, tri = list(v["cols"]), r["tri"]
    if tri and any(p.get(tri) is not None for p in v["liste"]):
        cols = [tri] + [c for c in cols if c != tri]
    return cols


def _raisons_vide(r: Dict) -> str:
    out = []
    if not r["familles"]:
        out.append("aucune famille cochée")
    if r["recherche"].strip():
        out.append(f"recherche « {r['recherche'].strip()} »")
    for f in r["filtres"]:
        if f["cle"] and f["min"] is not None and f["max"] is not None and f["min"] > f["max"]:
            out.append(f"{CO.entete(f['cle'])} : min > max")
    if any(f["cle"] and (f["min"] is not None or f["max"] is not None) for f in r["filtres"]):
        out.append("filtres numériques")
    if r["courants"]:
        out.append("séries courantes seulement")
    return "En cause : " + ", ".join(out) + "." if out else ""


def comparaison(v: Dict) -> str:
    """La vue « Comparaison » : les sections à la même échelle, puis le tableau de comparaison de l'écran."""
    comp = v["comp"]
    lignes = CO.comparaison(comp, CO.colonnes_comparaison(v["cols"]))
    th = "".join(f'<th class="n">{C.esc(p["nom"])}</th>' for p in comp)
    corps = "".join(f'<tr><td class="nm" title="{C.esc(CAT.AIDE_COLONNES.get(c, ""))}">{symbole(c)}<span class="u">{C.esc(CAT.UNITE.get(c, ""))}</span></td>'
                    + "".join(f"<td>{C.esc(x)}</td>" for x in vals) + "</tr>" for c, _, vals in lignes)
    return (f'<div class="cat-comp"><div class="cat-cdes">{_dessin(lambda: D.comparaison("cat_comp", comp), "")}</div>'
            f'<div class="cat-defile"><table class="cat-t cat-c"><thead><tr><th class="nm">Propriété</th>{th}</tr></thead>'
            f'<tbody>{corps}</tbody></table></div></div>')


# ------------------------------------------------------------------ la fenêtre de calcul : les formules d'un profilé
def _d(x: float) -> str:
    """Une dimension (mm) ou une valeur saisie, exacte."""
    return C.nb(x, 3)


def _cm(x_mm: float) -> str:
    """Une dimension en mm écrite en cm (les formules en cm donnent des cm², cm³, cm⁴)."""
    return C.nb(x_mm / 10.0, 4)


def _p(p: Dict, c: str) -> str:
    """Une propriété du profilé écrite comme dans la fiche."""
    return CAT.formater(p[c], c)


def _sv(x: float) -> str:
    """Une valeur SUBSTITUÉE dans une formule : quatre chiffres significatifs au moins (0,3942 et non 0,4 pour l'I_y
    d'une L 20 : la formule écrite doit redonner le résultat), entre parenthèses si elle est négative (un carré
    « −42,2² » se lirait −(42,2²))."""
    n = 2
    if x:
        n = min(6, max(1, 3 - math.floor(math.log10(abs(x)))))
    t = C.fr(x, n).replace("-", "−")
    return f"({t})" if x < 0 else t


def _v(p: Dict, c: str) -> str:
    """La propriété ``c`` du profilé, substituée dans une formule (au moins les décimales de sa colonne)."""
    x = float(p[c])
    n = CAT.DECIMALES.get(c, 2)
    if x:
        n = min(6, max(n, 3 - math.floor(math.log10(abs(x)))))
    t = C.fr(x, n).replace("-", "−")
    return f"({t})" if x < 0 else t


def _fx(cle: str, parts: List[str], p: Dict, res: Optional[str] = None) -> Dict:
    return {"cle": cle, "parts": parts, "res": res if res is not None else _p(p, cle) + _u(cle), "niveau": ""}


def formules(p: Dict) -> List[tuple]:
    """Les formules des propriétés d'un profilé, groupées : [(titre, [ligne])] ; chaque ligne porte la clé de la
    propriété (``cle``) dont elle donne la valeur du catalogue. Rien n'est calculé : les valeurs substituées et le
    résultat sont ceux du profilé."""
    fam = p["famille"]
    forme = F.FAMILLES[fam]["forme"]
    d = {k: float(x) for k, x in (p.get("dims") or {}).items()}
    h, b, tw, tf, t = d.get("h"), d.get("b"), d.get("tw"), d.get("tf"), d.get("t")
    r = d.get("r", d.get("r1", 0.0))
    out = []
    g = [_fx("A", ["A", "aire du contour exact (formule de Green)"], p),
         _fx("masse", ["m", "A·ρ", f"{_v(p, 'A')} × 10⁻⁴ m² × 7 850 kg/m³"], p),
         _fx("perimetre", ["S", "périmètre extérieur × 1 m"], p)]
    out.append(("Section et masse", g))
    g = []
    if forme in ("I", "I_conique"):
        g.append(_fx("d_ame", ["d", "h − 2·t_f − 2·r", f"{_d(h)} − 2 × {_d(tf)} − 2 × {_d(r)}"], p))
        g.append(_fx("cf_tf", ["c_f/t_f", "((b − t_w)/2 − r)/t_f", f"(({_d(b)} − {_d(tw)})/2 − {_d(r)})/{_d(tf)}"], p))
        g.append(_fx("cw_tw", ["c_w/t_w", "(h − 2·t_f − 2·r)/t_w", f"({_d(h)} − 2 × {_d(tf)} − 2 × {_d(r)})/{_d(tw)}"], p))
    elif forme in ("U", "U_conique"):
        g.append(_fx("cf_tf", ["c_f/t_f", "(b − t_w − r)/t_f", f"({_d(b)} − {_d(tw)} − {_d(r)})/{_d(tf)}"], p))
        g.append(_fx("cw_tw", ["c_w/t_w", "(h − 2·t_f − 2·r)/t_w", f"({_d(h)} − 2 × {_d(tf)} − 2 × {_d(r)})/{_d(tw)}"], p))
    elif forme == "L":
        g.append(_fx("cf_tf", ["c_f/t_f", "h/t (grande aile)", f"{_d(h)}/{_d(t)}"], p))
        g.append(_fx("cw_tw", ["c_w/t_w", "b/t (petite aile)", f"{_d(b)}/{_d(t)}"], p))
    elif forme == "creux_rect":
        re_, ri = F.rayons(fam, t)
        g.append(_fx("cf_tf", ["c_f/t_f", "(b − 2·r_ext)/t", f"({_d(b)} − 2 × {_d(re_)})/{_d(t)}"], p))
        g.append(_fx("cw_tw", ["c_w/t_w", "(h − 2·r_ext)/t", f"({_d(h)} − 2 × {_d(re_)})/{_d(t)}"], p))
    elif forme == "creux_rond":
        g.append(_fx("cf_tf", ["d/t", "d/t", f"{_d(d['d'])}/{_d(t)}"], p))
    if g:
        out.append(("Parois · élancements c/t (EN 1993-1-1 tab. 5.2)", g))
    g = []
    if forme in ("I", "I_conique"):
        g.append(_fx("Avz", ["A_v,z", "A − 2·b·t_f + (t_w + 2·r)·t_f",
                             f"{_v(p, 'A')} − 2 × {_cm(b)} × {_cm(tf)} + ({_cm(tw)} + 2 × {_cm(r)}) × {_cm(tf)}"], p))
        g.append(_fx("Avy", ["A_v,y", "A − (h − 2·t_f)·t_w", f"{_v(p, 'A')} − ({_cm(h)} − 2 × {_cm(tf)}) × {_cm(tw)}"], p))
    elif forme in ("U", "U_conique"):
        g.append(_fx("Avz", ["A_v,z", "A − 2·b·t_f + (t_w + r)·t_f",
                             f"{_v(p, 'A')} − 2 × {_cm(b)} × {_cm(tf)} + ({_cm(tw)} + {_cm(r)}) × {_cm(tf)}"], p))
        g.append(_fx("Avy", ["A_v,y", "A − (h − 2·t_f)·t_w", f"{_v(p, 'A')} − ({_cm(h)} − 2 × {_cm(tf)}) × {_cm(tw)}"], p))
    elif forme == "creux_rect":
        g.append(_fx("Avz", ["A_v,z", "A·h/(b + h)", f"{_v(p, 'A')} × {_d(h)}/({_d(b)} + {_d(h)})"], p))
        g.append(_fx("Avy", ["A_v,y", "A·b/(b + h)", f"{_v(p, 'A')} × {_d(b)}/({_d(b)} + {_d(h)})"], p))
    elif forme == "creux_rond":
        g.append(_fx("Avz", ["A_v", "2·A/π", f"2 × {_v(p, 'A')}/π"], p))
    if g:
        out.append(("Aires de cisaillement (EN 1993-1-1 §6.2.6(3))", g))
    zm, ym = max(p["z_sup"], p["z_inf"]), max(p["y_droite"], p["y_gauche"])
    out.append(("Flexion autour de y-y", [
        _fx("Wel_y", ["W_el,y", "I_y/max(z_sup ; z_inf)", f"{_v(p, 'Iy')}/{_cm(zm)}"], p),
        _fx("Wpl_y", ["W_pl,y", "axe neutre plastique à aires égales"], p),
        _fx("iy", ["i_y", "√(I_y/A)", f"√({_v(p, 'Iy')}/{_v(p, 'A')})"], p)]))
    out.append(("Flexion autour de z-z", [
        _fx("Wel_z", ["W_el,z", "I_z/max(y_droite ; y_gauche)", f"{_v(p, 'Iz')}/{_cm(ym)}"], p),
        _fx("Wpl_z", ["W_pl,z", "axe neutre plastique à aires égales"], p),
        _fx("iz", ["i_z", "√(I_z/A)", f"√({_v(p, 'Iz')}/{_v(p, 'A')})"], p)]))
    if forme == "L":                       # un creux carré a I_y = I_z : ses axes principaux sont quelconques
        iyz = _sv(p.get("Iyz") or 0.0)
        out.append(("Axes principaux", [
            _fx("Iu", ["I_u", "(I_y + I_z)/2 + √(((I_y − I_z)/2)² + I_yz²)", f"({_v(p, 'Iy')} + {_v(p, 'Iz')})/2 + √((({_v(p, 'Iy')} − {_v(p, 'Iz')})/2)² + {iyz}²)"], p),
            _fx("Iv", ["I_v", "(I_y + I_z)/2 − √(((I_y − I_z)/2)² + I_yz²)", f"({_v(p, 'Iy')} + {_v(p, 'Iz')})/2 − √((({_v(p, 'Iy')} − {_v(p, 'Iz')})/2)² + {iyz}²)"], p),
            _fx("alpha_deg", ["α", "angle de u (axe fort) avec y-y", f"I_yz = {_v(p, 'Iyz')} cm⁴"], p)]))
    g = []
    meth = _methode_txt(p.get("methode_It") or "")
    if forme == "creux_rond":
        g.append(_fx("It", ["I_t", "π·(d⁴ − (d − 2·t)⁴)/32", f"π × ({_cm(d['d'])}⁴ − ({_cm(d['d'])} − 2 × {_cm(t)})⁴)/32"], p))
        g.append(_fx("Wt", ["W_t", "2·I_t/d", f"2 × {_v(p, 'It')}/{_cm(d['d'])}"], p))
    elif forme == "creux_rect":
        re_, ri = F.rayons(fam, t)
        g.append(_fx("It", ["I_t", "t³·h_m/3 + 2·k·A_h (EN 10210-2 annexe B)",
                            f"t = {_d(t)} · b × h = {_d(b)} × {_d(h)} · r_ext = {_d(re_)} · r_int = {_d(ri)} mm"], p))
        g.append(_fx("Wt", ["W_t", "I_t/(t + k/t)"], p))
    elif forme == "plein_rond":
        g.append(_fx("It", ["I_t", "π·d⁴/32", f"π × {_cm(d['d'])}⁴/32"], p))
    else:
        g.append(_fx("It", ["I_t", meth or "—"], p))
    if forme in ("I", "I_conique"):
        g.append(_fx("Iw", ["I_w", "I_z·(h − t_f)²/4", f"{_v(p, 'Iz')} × ({_cm(h)} − {_cm(tf)})²/4"], p))
    elif forme in ("U", "U_conique"):
        g.append(_fx("Iw", ["I_w", "Vlasov sur la ligne moyenne"], p))
        if p.get("y_s") is not None:
            g.append(_fx("y_s", ["y_s", "centre de cisaillement depuis le centre de gravité"], p))
    elif p.get("Iw") is not None:
        g.append(_fx("Iw", ["I_w", "section sans gauchissement"], p))
    out.append(("Torsion et gauchissement", g))
    return out


def _fenetre_sans_tableau(html: str, sous: str = "") -> str:
    """La fenêtre de calcul commune, SANS son tableau de vérifications (un catalogue ne vérifie rien) : le tableau
    vide est retiré ; le sous-titre du détail peut être changé. Besoin commun signalé : C.fenetre_calcul sans tableau."""
    html = re.sub(r'<table class="vtab">.*?</table>', "", html, count=1, flags=re.S)
    if sous:
        html = html.replace('<div class="ck-sub">Le détail du calcul</div>', f'<div class="ck-sub">{sous}</div>', 1)
    return html


def fenetre_fiche(p: Dict) -> str:
    f = F.FAMILLES[p["famille"]]
    hyps = [f"<b>{C.esc(p['nom'])}</b> · {C.esc(f['libelle'])}", C.esc(f["norme"]), "ρ = 7 850 kg/m³ (EN 1991-1-1 tab. A.4)",
            "contour exact · congés à 36 segments", "y-y : axe fort · z-z : axe faible (EN 1993-1-1 §1.7)"]
    detail = "".join(f'<div class="ck-g"><div class="ck-t">{C.esc(t)}</div>{C.lignes_fx(lignes)}</div>' for t, lignes in formules(p))
    pied = '<div class="ck-autres">Les valeurs viennent du catalogue (acier/donnees/profiles.json) ou, pour un profilé personnalisé, du même moteur de sections.</div>'
    return _fenetre_sans_tableau(C.fenetre_calcul("fiche", f"{C.esc(p['nom'])} · d'où viennent les propriétés",
                                                  "les formules du moteur de sections, avec les valeurs du profilé", hyps, [], detail, pied),
                                 "Les propriétés, formule par formule")


def fenetre_methode() -> str:
    m = CO.methode_et_sources()
    hyps = [f"Source <b>{C.esc(m['fichier'])}</b>", f"version {C.esc(m['version'])} · {C.esc(_date(m['date']))}",
            f"<b>{C.fr(m['nombre'], 0)}</b> profilés", f"écran de référence : Streamlit, version {CO.VERSION}"]

    def bloc(titre, corps):
        return f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div><div class="cat-txt">{corps}</div></div>'
    detail = (bloc("Le catalogue", "Profilés laminés, creux et reconstitués — dimensions normalisées, propriétés calculées depuis "
                                   f"le contour exact · {C.fr(len(CAT.profils()), 0)} profilés")
              + bloc("Conventions", C.symb_math(m["conventions"]))
              + bloc("Dimensions", "<ul>" + "".join(f"<li>{C.esc(s)}</li>" for s in m["sources"]) + "</ul>")
              + bloc("Calcul", C.esc(m["calcul"]))
              + bloc("Contrôle croisé", C.esc(m["controle"]))
              + bloc("Définitions", "<ul>" + "".join(f"<li><b>{symbole(c)}</b> — {C.esc(t)}</li>" for c, _, t in m["definitions"]) + "</ul>")
              + bloc("Familles", "<ul>" + "".join(f"<li><b>{C.esc(f)}</b> — {C.esc(lib)} · {C.esc(n)} · {C.esc(dsc)}</li>"
                                                  for f, lib, n, dsc in m["familles"]) + "</ul>"))
    return _fenetre_sans_tableau(C.fenetre_calcul("methode", "Méthode et sources", "d'où viennent les profilés et leurs propriétés",
                                                  hyps, [], detail), "Les sources et les définitions")


def _date(iso: str) -> str:
    a, m, j = (iso.split("-") + ["", "", ""])[:3]
    return f"{j}/{m}/{a}" if j else iso


# ------------------------------------------------------------------ le cadre Résultat
def _ligne(nom: str, piece: str, calcul: str = "", pk: str = "", gov: bool = False, leger: str = "") -> str:
    """Une ligne du résultat SANS jauge (un catalogue n'a pas de taux) : la brique C.ligne_res, sa seconde ligne
    retirée. Besoin commun signalé : C.ligne_res sans taux."""
    h = C.ligne_res(nom, piece, None, "c", calcul=calcul, pk=pk, gov=gov, leger=leger)
    return re.sub(r'<div class="l2">.*?</span></div>', "", h, count=1, flags=re.S)


def _plus_leger(lst: Sequence[Dict]) -> Optional[Dict]:
    """Le plus léger de la liste (le premier en cas d'égalité) : on CHOISIT parmi les masses du catalogue."""
    return min(lst, key=lambda q: q["masse"]) if lst else None


def _table_fiche(p: Dict, filtres: Sequence[Dict]) -> str:
    """Toutes les propriétés du profilé, groupe par groupe (la fiche de l'écran) ; une propriété filtrée est marquée."""
    fil = {f["cle"]: f for f in filtres if f["cle"]}
    out = ""
    for g, props in CO.fiche(p):
        out += f'<tr class="g"><td colspan="3">{C.esc(g)}</td></tr>'
        for c, _lib, val, unite, aide in props:
            f = fil.get(c)
            borne = ""
            if f and (f["min"] is not None or f["max"] is not None):
                borne = " · filtre : " + _resume_filtres([f])
            i = f'<span class="cat-i" title="{C.esc(aide)}">{C.icon("info")}</span>' if aide else ""
            out += (f'<tr{" class=" + chr(34) + "f" + chr(34) if borne else ""}><td class="s"{" title=" + chr(34) + C.esc(borne[3:]) + chr(34) if borne else ""}>'
                    f'{symbole(c)}{i}</td><td class="n">{C.esc(val)}</td><td class="u">{C.esc(unite)}</td></tr>')
    return f'<div class="cat-fiche"><div class="cat-fh">Toutes les propriétés</div><table>{out}</table></div>'


def _verdict(p: Optional[Dict]) -> str:
    if p is None:
        return (f'<div class="verdict cat-v"><span class="cat-vig vide">{C.icon("search")}</span><div class="gov"><b>Aucun profilé</b>'
                f'{C.esc(CO.MESSAGE_VIDE)}</div></div>')
    f = F.FAMILLES[p["famille"]]
    tags = ('<span class="tag perso">personnalisé</span>' if p.get("perso") else
            '<span class="tag courant">série courante</span>' if p.get("courant") else '<span class="tag">série peu courante</span>')
    return (f'<div class="verdict cat-v"><span class="cat-vig">{_dessin(lambda: D.vignette("cat_vig", p), "")}</span><div class="gov"><b>{C.esc(p["nom"])}</b>'
            f'{C.esc(f["libelle"])}<span class="cat-n">{_masse(p["masse"])} kg/m · A = {C.fr(p["A"], 2)} cm²</span>'
            f'<span class="cat-tags">{tags}<span class="tag">{C.esc(f["norme"])}</span></span></div></div>')


def _dessin(faire, repli: Optional[str] = None) -> str:
    """Un dessin qui ne peut pas faire tomber la réponse : s'il échoue, la fiche reste lisible (l'écran Streamlit fait
    de même : « dessin indisponible »)."""
    try:
        return faire()
    except Exception as e:  # noqa: BLE001 — un contour impossible ne doit pas cacher les propriétés
        if repli is not None:
            return repli
        return (f'<div class="cat-tab vide"><div class="cat-vide">{C.icon("info")}<b>Dessin indisponible</b>'
                f'<span>{C.esc(type(e).__name__)} : {C.esc(str(e)[:160])}</span></div></div>')


def resume_note(etat: Dict) -> Dict:
    r = _rech(etat)
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1],
            "gouvernant": f"{r.get('nom') or 'Recherche'} · {r['fiche'] or 'aucun profilé'}"}


def _masse_de(r: Dict, nom: Optional[str]) -> Optional[float]:
    if nom is None:
        return None
    p = CAT.par_nom().get(nom)
    if p is None:
        p = next((q for q in persos_calcules(r) if q["nom"] == nom), None)
    return p["masse"] if p else None


def elements(etat: Dict) -> str:
    items = ""
    rs = etat["recherches"]
    for i, r in enumerate(rs):
        m = _masse_de(r, r["fiche"])
        txt = f"{_masse(m)}{C.NBSP}kg/m" if m is not None else "—"
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(rs) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(r.get("nom") or f"Recherche {i + 1}")} '
                  f'<span>· {C.esc(r["fiche"] or "aucun profilé")}</span></span><span class="pct c">{txt}</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(rs)}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la recherche ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une recherche</div></div>')


def resultat(etat: Dict) -> Dict:
    r = _rech(etat)
    v = _vue(r)
    p = v["p"]
    lst = v["liste"]
    # l'optimal ★ : le plus léger de la liste, qui n'a de sens que si une borne a été posée (« Iy ≥ 1 500 cm⁴ : le plus
    # léger qui convient ») ; sans borne, le plus léger de toutes les familles cochées serait une petite cornière
    leger = _plus_leger(lst) if _bornes_actives(r) else None
    autre = leger is not None and p is not None and leger["nom"] != p["nom"]
    rows = ""
    if p is not None:
        msg_leger = f"Le plus léger qui satisfait les filtres : {leger['nom']} · {_masse(leger['masse'])} kg/m" if autre else ""
        rows += _ligne("Profilé", C.esc(p["nom"]), calcul="fiche", pk="fiche", gov=True, leger=msg_leger)
        nc = len(v["comp"])
        rows += _ligne("Comparer", f"{nc} profilé{'s' if nc > 1 else ''}" if nc else "choisir", pk="compare")
    rows += _ligne("Méthode et sources", f"{C.fr(len(CAT.profils()), 0)} profilés", calcul="methode")
    messages = ""
    if p is not None and CO.methode_it(p):
        messages += C.message(f"I<sub>t</sub> : {C.esc(_methode_txt(CO.methode_it(p)))}.")
    if p is not None and p.get("perso"):
        for av in CO.avertissements_perso(p["famille"], p["dims"]):
            messages += C.message(f"{C.esc(p['nom'])} : {C.symb_math(av)}.", grave=True)
    for reserve in (CO.reserve_torsion, CO.reserve_cisaillement):
        if p is not None and reserve(p):
            messages += C.message(C.symb_math(reserve(p)) + ".")
    if autre:
        messages += C.message(f'<span class="cat-etoile">{C.icon("star")}</span>Le plus léger qui satisfait les filtres : '
                              f'<b>{C.esc(leger["nom"])}</b>{C.NBSP}·{C.NBSP}{_masse(leger["masse"])}{C.NBSP}kg/m — le choisir dans la ligne « Profilé » '
                              f'ou par le bouton de la section 3 des Données.')
    elif leger is not None:
        messages += C.message(f'<span class="cat-etoile">{C.icon("star")}</span>{C.esc(p["nom"])} est le plus léger qui satisfait les filtres.')
    if not lst:
        messages += C.message(C.esc(CO.MESSAGE_VIDE + " " + _raisons_vide(r)))
    corps = _verdict(p) + f'<div class="els">{rows}</div>' + messages + (_table_fiche(p, r["filtres"]) if p is not None else "")
    m = CAT.meta()
    pied = (f'<div class="pfoot"><div class="mass"><span>Catalogue du bureau · v{C.esc(m["version"])} · {C.esc(_date(m["date"]))}</span>'
            f'<b class="cat-nb">{C.fr(len(CAT.profils()), 0)}{C.NBSP}profilés</b></div></div>')
    volets = (fenetre_fiche(p) if p is not None else "") + fenetre_methode()
    # la toile
    nf = len(r["familles"])
    phrase = (f"<b>{len(lst)} profilé{'s' if len(lst) > 1 else ''}</b>" + (f" · {nf} famille{'s' if nf > 1 else ''}" if r["familles"] else "")
              + (f" · {v['n_persos']} personnalisé{'s' if v['n_persos'] > 1 else ''}" if v["n_persos"] else "")
              + (f" · trié par {symbole(r['tri'])} {'décroissant' if r['decroissant'] else 'croissant'}" if r["tri"] else " · ordre du catalogue"))
    vues = [{"cle": "tableau", "titre": "Tableau des profilés", "court": "Tableau", "svg": tableau(r, v),
             "legende": CO.compteur(len(lst), r["familles"], v["n_persos"]).replace("**", "")}]
    if p is not None:
        vues.append({"cle": "section", "titre": f"Section {_nom_titre(p['nom'])}", "court": "Section",
                     "svg": _dessin(lambda: D.section("cat_section", p)), "legende": D.legende(p)})
    if len(v["comp"]) >= 2:
        vues.append({"cle": "comparaison", "titre": "Comparaison", "court": f"Comparaison · {len(v['comp'])}", "svg": comparaison(v),
                     "legende": "sections à la même échelle · " + " · ".join(q["nom"] for q in v["comp"])})
    return {"resultat": f'<div class="pbody">{corps}</div>{pied}', "volets": volets, "vues": vues, "phrase": phrase,
            "statut_html": C.statut("i"), "resume": resume_note(etat), "donnees": formulaire(etat, v), "elements": elements(etat)}


# ------------------------------------------------------------------ les sélecteurs
def _prop(p: Dict) -> str:
    f = F.FAMILLES[p["famille"]]
    vals = [f"{C.esc(f['libelle'])}", f"m = {_masse(p['masse'])} kg/m", f"A = {C.fr(p['A'], 2)} cm²",
            f"I<sub>y</sub> = {CAT.formater(p['Iy'], 'Iy')} cm⁴"]
    if p.get("Wpl_y") is not None:
        vals.append(f"W<sub>pl,y</sub> = {CAT.formater(p['Wpl_y'], 'Wpl_y')} cm³")
    vals.append(f"I<sub>z</sub> = {CAT.formater(p['Iz'], 'Iz')} cm⁴")
    return " · ".join(vals)


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Le sélecteur commun sur la liste du tableau (même filtre, même ordre) : « fiche » ouvre un profilé, « compare »
    l'ajoute à la comparaison ; les familles du tableau en pastilles pour resserrer la liste ; le retenu ✓ (la fiche,
    ou les comparés) et le plus léger de la liste ★ (« qui satisfait les filtres » quand une borne est posée). Sans
    taux : un catalogue ne vérifie rien (la pastille de taux de la brique est retirée — besoin commun signalé)."""
    r = _rech(etat)
    v = _vue(r)
    presentes = list(dict.fromkeys(p["famille"] for p in v["liste"]))
    gardees = [f for f in presentes if familles is None or f in familles] or presentes
    lst = [p for p in v["liste"] if p["famille"] in gardees]
    leger = _plus_leger(lst)
    if role == "compare":
        cur = set(r["compare"])
        titre, sous = ("Comparer · profilés", f"dans l'ordre du tableau · un clic ajoute ou retire (au plus "
                                               f"{CO.MAX_COMPARES} ; au-delà, le plus ancien sort)")
    else:
        cur = {v["p"]["nom"]} if v["p"] else set()
        titre, sous = "Profilé · fiche", "dans l'ordre du tableau · un clic ouvre la fiche"
    rows = [{"nom": p["nom"], "col": _masse(p["masse"]), "taux": 0, "niveau": "c", "cur": p["nom"] in cur,
             "opt": leger is not None and p["nom"] == leger["nom"], "prop": _prop(p)} for p in lst]
    puces = "".join(f'<button class="fchip{" on" if f in gardees else ""}" data-famille="{C.esc(f)}">{C.esc(f)}</button>'
                    for f in presentes) if len(presentes) > 1 else ""
    i_c = next((i for i, x in enumerate(rows) if x["cur"]), next((i for i, x in enumerate(rows) if x["opt"]), 0))
    etoile = "le plus léger qui satisfait les filtres" if _bornes_actives(r) else "le plus léger de la liste"
    html = C.selecteur(role, titre, sous, puces, rows, max(0, i_c - 4), "kg/m", etoile=etoile)
    return html.replace(C.chip(0, "c") + "</button>", "</button>").replace("<span>kg/m · taux</span>", "<span>kg/m</span>")

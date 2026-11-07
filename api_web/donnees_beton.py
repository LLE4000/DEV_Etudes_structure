# -*- coding: utf-8 -*-
"""Le module « Données béton » du site : l'AIDE-MÉMOIRE du bureau pour le béton armé, en CONSULTATION (pas de
verdict). Cinq rubriques, chacune lue à sa source unique — rien n'est recopié ici :

- Classes de béton : ``beton_classes.json`` (lu par ``beton/poutre/entree``), les grandeurs du tableau 3.1 de
  l'EN 1992-1-1 (fonctions nommées de ``beton/donnees/ec2.py``, confrontées au tableau publié) et celles de la méthode
  du bureau (α_b, μ du fichier, τ_adm calculés par ``beton/poutre/moteur.materiaux``) ;
- Aciers d'armature : les nuances des modules béton, f_yd par le même moteur (γ_s = 1,5 du bureau, 1,15 de
  l'EN 1992-1-1), E_s et ε_yd (``beton/donnees/ec2.py``) ;
- Tableau des armatures : ``beton/armatures/tableau.py`` (le tableau du bureau, relu au chiffre près) ;
- Enrobages : ``beton/enrobage/donnees.py`` (c_min,dur, classes proposées, Δc_dev, tableaux feu) et son moteur
  (``beton/enrobage/moteur.calculer``) pour la valeur retenue et les avertissements ;
- Ancrages et recouvrements : ``beton/donnees/ancrages_recouvrements.json``, VIDE tant que l'ingénieur n'a pas fourni
  ses tableaux (« tableaux du bureau à venir ») — aucune valeur n'est inventée.

Une note porte plusieurs CONSULTATIONS (une par question qu'on se pose) ; chacune garde sa rubrique et ses choix.
La toile montre le TABLEAU de la rubrique (bloc HTML, en-têtes figés, unités, lignes alternées), puis un dessin quand
il aide (les diamètres à l'échelle, la coupe d'enrobage) ; le Résultat, la fiche de la ligne choisie et sa source ;
la fenêtre « d'où viennent les valeurs », les formules avec leurs valeurs. Ce fichier MET EN FORME : il ne calcule rien.
"""
import copy
import re
from typing import Dict, List, Optional, Sequence

from beton.armatures import dessins as DS
from beton.armatures import moteur as AM
from beton.armatures import tableau as T
from beton.dalle import entree as DE_
from beton.donnees import ancrages as AN
from beton.donnees import consultation as K
from beton.donnees import ec2 as EC
from beton.enrobage import donnees as ED
from beton.enrobage import moteur as EM
from beton.poutre import entree as PE
from beton.poutre import moteur as PM

from . import commun as C
from . import dessin_enrobage as DEN

MODULE = "donnees_beton"
PAGE = "donnees-beton"

RUBRIQUES = (("classes", "Classes de béton"), ("aciers", "Aciers d'armature"), ("armatures", "Tableau des armatures"),
             ("enrobages", "Enrobages"), ("ancrages", "Ancrages et recouvrements"))
NOMS = dict(RUBRIQUES)
# les noms courts de la liste « Dans cette note » (un nom de rubrique entier y repoussait le nom de la consultation :
# « Consul… · Ancrages et recouvrements · à venir », capture du 07/11/2026)
COURTS = {"classes": "Béton", "aciers": "Acier", "armatures": "Armatures", "enrobages": "Enrobage", "ancrages": "Ancrages"}
LECTURES = (("barres", "Barres · mm²"), ("nappe", "Par mètre · mm²/m"))
TABLEAUX_ENROBAGE = (("durabilite", "Durabilité"), ("feu", "Feu"))
# les choix d'ouverture d'une consultation (des choix d'écran, pas des données) : la classe et la nuance des modules
# béton, le Ø du tableau des armatures, l'élément intérieur courant (XC1) en classe structurale de référence S4
# (« S4 = référence, durée de vie 50 ans » : beton/enrobage/donnees.AIDE_CLASSE_STRUCT), le Ø et la tolérance de
# l'écran Enrobage
EXPO_DEFAUT, STRUCT_DEFAUT = "XC1", "S4"
assert EXPO_DEFAUT in ED.EXPO_CLASSES and STRUCT_DEFAUT in ED.CLASSES_STRUCTURALES
DIAMETRES_ENROBAGE = tuple(d for d in T.DIAMETRES if ED.DIAM_BARRE_MIN <= d <= ED.DIAM_BARRE_MAX)


# ------------------------------------------------------------------ l'état : des consultations
def _neuve(nom: str) -> Dict:
    return {"nom": nom, "rubrique": RUBRIQUES[0][0],
            "classes": {"groupe": K.GROUPES_CLASSES[0][0], "classe": PE.BETON_DEFAUT},
            "aciers": {"nuance": PE.DEFAUTS_POUTRE["fyk"]},
            "armatures": {"lecture": LECTURES[0][0], "d": AM.DEFAUT["d"]},
            "enrobages": {"tableau": TABLEAUX_ENROBAGE[0][0], "classe_expo": EXPO_DEFAUT, "classe_struct": STRUCT_DEFAUT,
                          "diam_barre": ED.DIAM_BARRE_DEFAUT, "delta_dev": ED.DELTAS_DEV[0], "r": ED.R_DEFAUT},
            "ancrages": {"tableau": None, "ligne": 0}}


def nouvel_etat() -> Dict:
    return {"version": 1, "consultations": [_neuve("Consultation 1")], "actif": 0}


def _etat(rubrique: str, **choix) -> Dict:
    """Une note d'une consultation ouverte sur ``rubrique`` (et des choix « rubrique__clé » : enrobages__tableau=…)."""
    e = nouvel_etat()
    c = e["consultations"][0]
    c["rubrique"] = rubrique
    for k, v in choix.items():
        a, b = k.split("__")
        c[a][b] = v
    return e


# le parcours propre (api_web/parcours.py) : ce que deux niveaux de choix depuis une note neuve n'atteignent pas — le
# tableau feu (sa durée, son sélecteur) et le sélecteur de chaque rubrique
PARCOURS = ([{"action": "modifier", "etat": _etat("enrobages", enrobages__tableau="feu"), "chemin": ["enrobages", "r"], "valeur": r}
             for r in ("R30", "R240")]
            + [{"action": "selecteur", "etat": _etat(r, **x), "role": "ligne"} for r, x in (
                ("aciers", {}), ("armatures", {}), ("enrobages", {}), ("enrobages", {"enrobages__tableau": "feu"}), ("ancrages", {}))]
            + [{"action": "choisir", "etat": _etat("enrobages", enrobages__tableau="feu"), "role": "ligne", "nom": "R120"},
               {"action": "choisir", "etat": _etat("enrobages"), "role": "ligne", "nom": "XD3/XS3"},
               {"action": "choisir", "etat": _etat("armatures"), "role": "ligne", "nom": "Ø40"},
               {"action": "choisir", "etat": _etat("aciers"), "role": "ligne", "nom": "B400"}])


# les choix permis, rubrique par rubrique : (sous-clé, valeurs permises, conversion du texte reçu de l'écran)
def _permis() -> Dict:
    return {"classes": {"groupe": [g for g, _ in K.GROUPES_CLASSES], "classe": K.noms_classes()},
            "aciers": {"nuance": list(PE.NUANCES)},
            "armatures": {"lecture": [k for k, _ in LECTURES], "d": list(T.DIAMETRES)},
            "enrobages": {"tableau": [k for k, _ in TABLEAUX_ENROBAGE], "classe_expo": list(ED.EXPO_CLASSES),
                          "classe_struct": list(ED.CLASSES_STRUCTURALES), "diam_barre": list(DIAMETRES_ENROBAGE),
                          "delta_dev": list(ED.DELTAS_DEV), "r": list(ED.R_LEVELS)}}


def _lire_choix(valeur, permis: Sequence):
    """Une valeur reçue de l'écran (un texte : « 500 », « C30/37 ») ramenée à celle de la liste permise, ou None."""
    for p in permis:
        if valeur == p or (not isinstance(valeur, bool) and str(valeur) == str(p)):
            return p
    return None


def _normaliser_consultation(c, i: int) -> Dict:
    if not isinstance(c, dict):
        raise ValueError(f"consultation {i + 1} illisible : un objet est attendu")
    out = _neuve(f"Consultation {i + 1}")
    if c.get("nom") is not None:
        out["nom"] = str(c["nom"])
    if c.get("rubrique") is not None:
        if c["rubrique"] not in NOMS:
            raise ValueError(f"consultation {i + 1} : rubrique inconnue {c['rubrique']!r}")
        out["rubrique"] = c["rubrique"]
    for rub, champs in _permis().items():
        lu = c.get(rub)
        if lu is None:
            continue
        if not isinstance(lu, dict):
            raise ValueError(f"consultation {i + 1}, {rub} : un objet est attendu")
        for k, permis in champs.items():
            if lu.get(k) is None:
                continue
            v = _lire_choix(lu[k], permis)
            if v is None:
                raise ValueError(f"consultation {i + 1}, {rub}/{k} : {lu[k]!r} n'est pas proposé ({' · '.join(map(str, permis))})"[:240])
            out[rub][k] = v
    an = c.get("ancrages")
    if isinstance(an, dict):
        # le fichier des tableaux évolue indépendamment des notes : une clé qui n'y est plus revient au premier
        # tableau (None), une ligne hors du tableau à la première — c'est dit dans la fiche (le tableau ouvert)
        t = an.get("tableau")
        out["ancrages"]["tableau"] = t if isinstance(t, str) and t else None
        li = an.get("ligne")
        out["ancrages"]["ligne"] = li if isinstance(li, int) and not isinstance(li, bool) and li >= 0 else 0
    return out


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu : clés manquantes complétées ; un choix qui n'est pas
    proposé (une classe inconnue, une nuance B450…) lève une erreur claire — jamais remplacé en silence."""
    cs = etat.get("consultations") if isinstance(etat, dict) else None
    if not cs:
        return nouvel_etat()
    if not isinstance(cs, list):
        raise ValueError("note illisible : « consultations » doit être une liste")
    out = {"version": 1, "consultations": [_normaliser_consultation(c, i) for i, c in enumerate(cs)]}
    a = etat.get("actif", 0)
    out["actif"] = min(max(0, a if isinstance(a, int) and not isinstance(a, bool) else 0), len(out["consultations"]) - 1)
    return out


def _cons(etat: Dict) -> Dict:
    return etat["consultations"][etat["actif"]]


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les consultations de la note : en ouvrir une, en ajouter, copier l'ouverte, en supprimer une (jamais la dernière)."""
    cs = etat["consultations"]
    if action == "activer":
        etat["actif"] = min(max(0, int(index or 0)), len(cs) - 1)
    elif action == "ajouter":
        pris = {c["nom"] for c in cs}
        k = len(cs) + 1
        while f"Consultation {k}" in pris:
            k += 1
        cs.append(_neuve(f"Consultation {k}"))
        etat["actif"] = len(cs) - 1
    elif action == "dupliquer":
        c = copy.deepcopy(cs[etat["actif"]])
        c["nom"] = f"{c.get('nom') or 'Consultation'} (copie)"
        cs.append(c)
        etat["actif"] = len(cs) - 1
    elif action == "supprimer" and len(cs) > 1:
        i = min(max(0, int(index or 0)), len(cs) - 1)
        del cs[i]
        if etat["actif"] > i or etat["actif"] >= len(cs):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ les ancrages : le fichier du bureau
def _ancrages() -> Dict:
    """Le fichier des tableaux d'ancrage et de recouvrement, lu et contrôlé ; ``erreur`` s'il ne se relit pas (le
    site le dit, en rouge, plutôt que d'afficher un tableau tronqué)."""
    try:
        d = AN.lire()
        d["erreur"] = None
    except (OSError, ValueError) as e:
        d = {"titre": "Ancrages et recouvrements", "etat": "Tableaux du bureau à venir", "aide": "", "tableaux": [], "erreur": str(e)}
    return d


def _tableau_ancrage(c: Dict, d: Dict) -> Optional[Dict]:
    """Le tableau ouvert : celui de la clé retenue, sinon le premier du fichier ; None si le fichier n'en a aucun."""
    if not d["tableaux"]:
        return None
    return next((t for t in d["tableaux"] if t["cle"] == c["ancrages"]["tableau"]), d["tableaux"][0])


def _ligne_ancrage(c: Dict, t: Dict) -> int:
    return c["ancrages"]["ligne"] if c["ancrages"]["ligne"] < len(t["lignes"]) else 0


# ------------------------------------------------------------------ la saisie
def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans la consultation ouverte : le nom, la rubrique, un choix d'une rubrique (« classes/classe »,
    « enrobages/delta_dev »…). Un choix qui n'est pas proposé est refusé avec un message ; la consultation reste
    intacte."""
    c = _cons(etat)
    if not isinstance(chemin, (list, tuple)) or not chemin:
        return "Saisie inconnue"
    chemin = [str(x) for x in chemin]
    if chemin == ["nom"]:
        c["nom"] = str(valeur if valeur is not None else "")
        return None
    if chemin == ["rubrique"]:
        if valeur not in NOMS:
            return f"Rubrique inconnue : « {str(valeur)[:40]} »"
        c["rubrique"] = valeur
        return None
    if len(chemin) == 2 and chemin[0] == "ancrages" and chemin[1] in ("tableau", "ligne"):
        d = _ancrages()
        if chemin[1] == "tableau":
            if valeur not in [t["cle"] for t in d["tableaux"]]:
                return f"Tableau inconnu : « {str(valeur)[:40]} »"
            c["ancrages"].update(tableau=valeur, ligne=0)
        else:
            t = _tableau_ancrage(c, d)
            n = len(t["lignes"]) if t else 0
            v = _lire_choix(valeur, list(range(n)))
            if v is None:
                return f"Ligne inconnue : « {str(valeur)[:40]} »"
            c["ancrages"]["ligne"] = v
        return None
    champs = _permis()
    if len(chemin) != 2 or chemin[0] not in champs or chemin[1] not in champs[chemin[0]]:
        return f"Champ inconnu : « {'/'.join(chemin)[:60]} »"
    permis = champs[chemin[0]][chemin[1]]
    v = _lire_choix(valeur, permis)
    if v is None:
        return f"« {str(valeur)[:40]} » n'est pas proposé ({' · '.join(map(str, permis))})"
    c[chemin[0]][chemin[1]] = v
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le sélecteur de la ligne : une classe, une nuance (« B500 »), un Ø (« Ø12 »), une classe d'exposition, une
    durée de résistance au feu, une ligne d'un tableau d'ancrage."""
    c = _cons(etat)
    rows = _rangs(c)[3]
    n = next((r for r in rows if r["nom"] == nom), None)
    if n is None:
        raise ValueError(f"ligne inconnue : {nom!r}")
    cle, val = n["cible"]
    a, b = cle.split("/")
    c[a][b] = val


# ------------------------------------------------------------------ les nombres et les briques propres
def _u(u: str) -> str:
    return C.esc(u)


def _th(sym: str, unite: str, titre: str = "", cls: str = "n") -> str:
    """Une case d'en-tête : le symbole, l'unité dessous ; le nom complet au survol."""
    t = f' title="{C.esc(titre)}"' if titre else ""
    return f'<th class="{cls}"{t}><span class="s">{C.symb_math(sym)}</span><span class="u">{_u(unite) or "&nbsp;"}</span></th>'


_NORME = re.compile(r"(<option[^>]*>.*?</option>|<[^>]*>)|EN 1992-1-[12](?::2004)?", re.S)


def _normes(h: str) -> str:
    """Les références de norme (« EN 1992-1-1 ») d'un fragment gardées sur une ligne : coupées à leurs traits d'union,
    elles se lisaient « EN 1992-1- » puis « 1. » (relevé sur la capture du cadre Données, 1 440 px). Seul le TEXTE est
    touché : ni une balise, ni un attribut (un « title »), ni une option de menu (qui n'admet pas de balise)."""
    return _NORME.sub(lambda m: m.group(1) or f'<span class="db-nw">{m.group(0)}</span>', h)


def _vue(tableau: str, legende: str, pied: str, classe: str = "") -> str:
    """Un tableau de la toile : la légende (ce qu'on lit), le tableau qui défile dans sa feuille, le pied (la source)."""
    return (f'<div class="db-vue{(" " + classe) if classe else ""}"><div class="db-leg">{legende}</div>'
            f'<div class="db-defile">{tableau}</div><div class="db-pied">{pied}</div></div>')


def _ligne(nom: str, piece: str, calcul: str = "", pk: str = "", gov: bool = False) -> str:
    """Une ligne du résultat SANS jauge (une consultation n'a pas de taux) : la brique C.ligne_res, sa seconde ligne
    retirée — comme le catalogue des profilés (besoin commun déjà signalé : C.ligne_res sans taux)."""
    h = C.ligne_res(nom, piece, None, "c", calcul=calcul, pk=pk, gov=gov)
    return re.sub(r'<div class="l2">.*?</span></div>', "", h, count=1, flags=re.S)


def _fiche(groupes: List) -> str:
    """La fiche de la ligne choisie : des groupes de (symbole HTML, valeur, unité, nom au survol)."""
    out = ""
    for titre, lignes in groupes:
        out += f'<tr class="g"><td colspan="3">{C.esc(titre)}</td></tr>'
        for sym, val, unite, nom in lignes:
            t = f' title="{C.esc(nom)}"' if nom else ""
            out += f'<tr><td class="s"{t}>{sym}</td><td class="n">{val}</td><td class="u">{_u(unite)}</td></tr>'
    return f'<div class="db-fiche"><table>{out}</table></div>'


def _fenetre(titre: str, sous: str, hyps: List[str], groupes: List, pied: str = "") -> str:
    """La fenêtre « d'où viennent les valeurs » : la fenêtre de calcul commune SANS son tableau de vérifications
    (une consultation ne vérifie rien), les formules avec leurs valeurs, groupe par groupe."""
    detail = "".join(f'<div class="ck-g"><div class="ck-t">{C.esc(t)}</div>{corps}</div>' for t, corps in groupes)
    h = C.fenetre_calcul("source", titre, sous, hyps, [], detail, pied)
    h = re.sub(r'<table class="vtab">.*?</table>', "", h, count=1, flags=re.S)
    return h.replace('<div class="ck-sub">Le détail du calcul</div>', '<div class="ck-sub">Les valeurs, une à une</div>', 1)


def _fx(parts: List[str], res: str) -> Dict:
    return {"parts": parts, "res": res, "niveau": ""}


def _texte_fx(groupes: List) -> List:
    """Des groupes (titre, [lignes de formule]) → (titre, HTML)."""
    return [(t, C.lignes_fx(li)) for t, li in groupes]


def _src(texte: str) -> str:
    return f'<div class="db-src">{C.icon("file-text")}<span>{texte}</span></div>'


# ================================================================== rubrique 1 · classes de béton
def _classe(c: Dict) -> Dict:
    return next(x for x in K.classes() if x["nom"] == c["classes"]["classe"])


def _k_tau() -> List:
    """Les bandes de τ du moteur des poutres : (bande, k, clé)."""
    return [(b, k, PM.CLES_TAU[b]) for b, k, _, _ in PM.BANDES_TAU]


def _r_classes(c: Dict) -> Dict:
    lignes = K.classes()
    x = _classe(c)
    groupe = c["classes"]["groupe"]
    cols = [col for col in K.COLONNES_CLASSES if col[5] in ("classe", groupe)]
    th = '<th class="nm">Classe</th>' + "".join(_th(sym, u, nom) for _, sym, nom, u, _, _ in cols)
    corps = ""
    for li in lignes:
        on = ' class="on"' if li["nom"] == x["nom"] else ""
        corps += (f'<tr{on}><th class="nm">{C.esc(li["nom"])}</th>'
                  + "".join(f'<td>{C.fr(li[k], nd)}</td>' for k, _, _, _, nd, _ in cols) + "</tr>")
    taus = " · ".join(f"{C.nb(k)} ({b})" for b, k, _ in _k_tau())
    if groupe == "ec2":
        leg = (f"<b>{C.esc(EC.SOURCE_BETON)}</b> : les formules du tableau, arrondies comme lui (f<sub>ctm</sub> au 0,1 MPa, "
               "E<sub>cm</sub> au GPa) ; la valeur exacte est dans la fiche.")
    else:
        leg = (f"<b>Méthode du bureau</b> : α<sub>b</sub> et μ lus dans {C.esc(K.FICHIER_CLASSES)} ; "
               f"τ<sub>adm</sub> = k·f<sub>ck,cube</sub> / {C.nb(PM.GAMMA_TAU)}, k = {taus}, calculés par le moteur des poutres.")
    tab = f'<table class="db-tab"><thead><tr>{th}</tr></thead><tbody>{corps}</tbody></table>'
    pied = (f"{len(lignes)} classes · {C.esc(K.FICHIER_CLASSES)} (les classes des modules Poutre, Dalle et Prédalle) · "
            "la classe choisie est surlignée ; « Colonnes » (Données) passe de l'EN 1992-1-1 à la méthode du bureau.")
    # la fiche
    def v(k, nd):
        return C.fr(x[k], nd)
    defs = {k: (sym, nom, u, nd) for k, sym, nom, u, nd, _ in K.COLONNES_CLASSES}

    def ligne(k):
        sym, nom, u, nd = defs[k]
        return (C.symb_math(sym), v(k, nd), u, nom)
    fiche = _fiche([("Classe", [ligne("fck"), ligne("fck_cube")]),
                    ("EN 1992-1-1 · tableau 3.1", [ligne(k) for k in ("fcm", "fctm", "fctk_005", "fctk_095", "Ecm")]),
                    ("Méthode du bureau", [ligne(k) for k in ("alpha_b", "mu_a500", "mu_a400", "tau_1", "tau_2", "tau_4")])])
    verdict = C.verdict_valeur(x["fck"], "MPa", f"Béton {x['nom']}",
                               f"f<sub>ck,cube</sub> = {C.qte(x['fck_cube'], 'MPa', 0)}<br>E<sub>cm</sub> = {C.qte(x['Ecm'], 'GPa', 0)}"
                               f" · f<sub>ctm</sub> = {C.qte(x['fctm'], 'MPa', 1)}", nd=0, symbole="f<sub>ck</sub>", calcul="source",
                               classe="db-verdict")
    messages = ""
    if x.get("tau_lim_fichier") is not None:
        messages += C.message(f"« tau_lim » de {C.esc(K.FICHIER_CLASSES)} ({C.qte(x['tau_lim_fichier'], 'N/mm²', 2)} pour "
                              f"{C.esc(x['nom'])}) n'est lu par aucun moteur : les τ<sub>adm</sub> des modules Poutre, Dalle et "
                              "Prédalle sont calculés depuis f<sub>ck,cube</sub> (fiche, méthode du bureau).")
    # la fenêtre : les formules avec les valeurs de la classe
    fck = x["fck"]
    if fck <= EC.FCK_LIMITE_FCTM:
        f_ctm = _fx(["f_ctm", "0,30·f_ck^(2/3)", f"0,30 × {C.nb(fck)}^(2/3)"], f"{C.fr(x['fctm'], 3)} MPa")
    else:
        f_ctm = _fx(["f_ctm", "2,12·ln(1 + f_cm/10)", f"2,12 × ln(1 + {C.nb(x['fcm'])}/10)"], f"{C.fr(x['fctm'], 3)} MPa")
    ec2 = [_fx(["f_cm", "f_ck + 8", f"{C.nb(fck)} + 8"], f"{C.fr(x['fcm'], 0)} MPa"), f_ctm,
           _fx(["f_ctk,0,05", "0,7·f_ctm", f"0,7 × {C.fr(x['fctm'], 3)}"], f"{C.fr(x['fctk_005'], 3)} MPa"),
           _fx(["f_ctk,0,95", "1,3·f_ctm", f"1,3 × {C.fr(x['fctm'], 3)}"], f"{C.fr(x['fctk_095'], 3)} MPa"),
           _fx(["E_cm", "22·(f_cm/10)^0,3", f"22 × ({C.nb(x['fcm'])}/10)^0,3"], f"{C.fr(x['Ecm'], 2)} GPa")]
    # (le nom du fichier n'entre pas dans une formule : « beton_classes » y deviendrait un indice ; il est en tête)
    bureau = [_fx([sym, "valeur du fichier des classes"], C.nb(x[k], 4)) for k, sym in (("alpha_b", "α_b"), ("mu_a500", "μ_B500"), ("mu_a400", "μ_B400"))]
    bureau += [_fx([f"τ_adm,{b}", f"k_{b}·f_ck,cube / {C.nb(PM.GAMMA_TAU)}", f"{C.nb(k)} × {C.nb(x['fck_cube'])} / {C.nb(PM.GAMMA_TAU)}"],
                   f"{C.fr(x[cle], 3)} N/mm²") for b, k, cle in _k_tau()]
    fen = _fenetre(f"Béton {C.esc(x['nom'])} · d'où viennent les valeurs", "chaque valeur avec sa formule ou son fichier",
                   [f"Classe lue dans <b>{C.esc(K.FICHIER_CLASSES)}</b>", C.esc(EC.SOURCE_BETON),
                    "τ<sub>adm</sub> : beton/poutre/moteur.materiaux"],
                   _texte_fx([("EN 1992-1-1 · tableau 3.1", ec2), ("Méthode du bureau", bureau)]),
                   '<div class="ck-autres">Les fonctions du tableau 3.1 (beton/donnees/ec2.py) sont confrontées au tableau '
                   'publié, de C12/15 à C90/105 (tests/test_donnees_beton_moteur.py).</div>')
    return {"ligne": x["nom"], "titre_ligne": "Classe", "table": _vue(tab, leg, pied), "legende": f"{len(lignes)} classes · {dict(K.GROUPES_CLASSES)[groupe]}",
            "verdict": verdict, "fiche": fiche, "messages": messages, "fenetre": fen, "vues_plus": [],
            "phrase": f"<b>Béton {C.esc(x['nom'])}</b> · f<sub>ck</sub> = {C.qte(x['fck'], 'MPa', 0)} · f<sub>ck,cube</sub> = {C.qte(x['fck_cube'], 'MPa', 0)}",
            "sources": [f"Classes : <b>{C.esc(K.FICHIER_CLASSES)}</b> (beton/poutre/entree.classes_beton)",
                        f"EN 1992-1-1 : beton/donnees/ec2.py ({C.esc(EC.SOURCE_BETON)})",
                        "τ<sub>adm</sub> : beton/poutre/moteur.materiaux"]}


# ================================================================== rubrique 2 · aciers d'armature
def _r_aciers(c: Dict) -> Dict:
    lignes = K.aciers()
    x = next(a for a in lignes if a["nuance"] == c["aciers"]["nuance"])
    gb, ge = C.nb(K.GAMMA_S_BUREAU), C.nb(EC.GAMMA_S)
    cols = K.COLONNES_ACIERS
    h1 = ('<th class="nm" rowspan="2">Nuance</th><th class="grp" colspan="1">Caractéristique</th>'
          f'<th class="grp" colspan="1">Bureau · γ<sub>s</sub> = {gb}</th>'
          f'<th class="grp" colspan="3">EN 1992-1-1 · γ<sub>s</sub> = {ge}</th>')
    h2 = "".join(_th(sym, u, nom.format(g_bureau=gb, g_ec2=ge)) for _, sym, nom, u, _, _ in cols)
    corps = ""
    for li in lignes:
        on = ' class="on"' if li["nuance"] == x["nuance"] else ""
        corps += f'<tr{on}><th class="nm">{C.esc(li["nom"])}</th>' + "".join(f'<td>{C.fr(li[k], nd)}</td>' for k, _, _, _, nd, _ in cols) + "</tr>"
    tab = f'<table class="db-tab db-deux"><thead><tr class="r1">{h1}</tr><tr class="r2">{h2}</tr></thead><tbody>{corps}</tbody></table>'
    leg = (f"<b>f<sub>yd</sub> = f<sub>yk</sub> / γ<sub>s</sub></b>, par le moteur des poutres : γ<sub>s</sub> = {gb}, le "
           f"« coefficient acier ELS » des modules béton (méthode du bureau) ; γ<sub>s</sub> = {ge}, {C.esc(EC.SOURCE_GAMMA_S)}.")
    pied = (f"Nuances des modules Poutre, Dalle et Prédalle (beton/poutre/entree) · E<sub>s</sub> : {C.esc(EC.SOURCE_E_S)} · "
            "ε<sub>yd</sub> = f<sub>yd</sub> / E<sub>s</sub>.")
    fiche = _fiche([("Nuance", [("f<sub>yk</sub>", C.fr(x["fyk"], 0), "MPa", "limite d'élasticité caractéristique")]),
                    ("Méthode du bureau", [("γ<sub>s</sub>", C.nb(x["gamma_bureau"]), "", "coefficient acier ELS des modules béton"),
                                           ("f<sub>yd</sub>", C.fr(x["fyd_bureau"], 1), "MPa", "f_yk / γ_s")]),
                    ("EN 1992-1-1", [("γ<sub>s</sub>", C.nb(x["gamma_ec2"]), "", EC.SOURCE_GAMMA_S),
                                     ("f<sub>yd</sub>", C.fr(x["fyd_ec2"], 1), "MPa", "f_yk / γ_s"),
                                     ("E<sub>s</sub>", C.fr(x["Es"], 0), "GPa", EC.SOURCE_E_S),
                                     ("ε<sub>yd</sub>", C.fr(x["eps_yd"], 2), "‰", "f_yd / E_s")]),
                    ("Tableau des armatures", [("ρ", C.fr(x["masse_volumique"], 0), "kg/m³", "masse volumique du tableau du bureau")])])
    verdict = C.verdict_valeur(x["fyk"], "MPa", f"Acier {x['nom']}",
                               f"f<sub>yd</sub> = {C.qte(x['fyd_bureau'], 'MPa', 0)} · bureau<br>"
                               f"f<sub>yd</sub> = {C.qte(x['fyd_ec2'], 'MPa', 0)} · EN 1992-1-1",
                               nd=0, symbole="f<sub>yk</sub>", calcul="source", classe="db-verdict")
    messages = C.message(f"γ<sub>s</sub> = {gb} est le « coefficient acier ELS » des modules Poutre, Dalle et Prédalle (méthode "
                         f"ancienne du bureau, réglable de {C.nb(DE_.GAMMA_S_MIN)} à {C.nb(DE_.GAMMA_S_MAX)}) ; γ<sub>s</sub> = {ge} "
                         "est la valeur recommandée de l'EN 1992-1-1 à l'ELU — l'annexe nationale belge est à confirmer par le bureau.")
    messages += C.message(f"ρ = {C.qte(x['masse_volumique'], 'kg/m³', 0)} : la masse volumique du tableau des armatures du bureau "
                          "(l'usage courant est 7 850 kg/m³ ; les masses du tableau sont plus fortes de 0,13 %).")
    n = C.fr(x["fyk"], 0)
    fx = [_fx(["f_yd", "f_yk / γ_s", f"{n} / {gb}"], f"{C.fr(x['fyd_bureau'], 1)} MPa")]
    fx2 = [_fx(["f_yd", "f_yk / γ_s", f"{n} / {ge}"], f"{C.fr(x['fyd_ec2'], 1)} MPa"),
           _fx(["ε_yd", "f_yd / E_s", f"{C.fr(x['fyd_ec2'], 1)} / {C.fr(x['Es'] * 1000, 0)}"], f"{C.fr(x['eps_yd'], 3)} ‰")]
    fen = _fenetre(f"Acier {C.esc(x['nom'])} · d'où viennent les valeurs", "f_yd par le moteur des poutres, E_s de l'EN 1992-1-1",
                   [f"f<sub>yk</sub> = <b>{n} MPa</b>", f"bureau : γ<sub>s</sub> = <b>{gb}</b>", f"EN 1992-1-1 : γ<sub>s</sub> = <b>{ge}</b>",
                    f"E<sub>s</sub> = <b>{C.fr(x['Es'], 0)} GPa</b>"],
                   _texte_fx([("Méthode du bureau", fx), ("EN 1992-1-1", fx2)]))
    return {"ligne": x["nom"], "titre_ligne": "Nuance", "table": _vue(tab, leg, pied), "legende": "B400 et B500 · bureau et EN 1992-1-1",
            "verdict": verdict, "fiche": fiche, "messages": messages, "fenetre": fen, "vues_plus": [],
            "phrase": f"<b>Acier {C.esc(x['nom'])}</b> · f<sub>yk</sub> = {C.qte(x['fyk'], 'MPa', 0)}",
            "sources": ["Nuances : beton/poutre/entree.NUANCES (« Qualité acier (B) » des modules béton)",
                        f"γ<sub>s</sub> du bureau : beton/poutre/entree (défaut {gb}) ; f<sub>yd</sub> : beton/poutre/moteur.materiaux",
                        f"E<sub>s</sub>, γ<sub>s</sub> = {ge} : beton/donnees/ec2.py"]}


# ================================================================== rubrique 3 · tableau des armatures
def _r_armatures(c: Dict) -> Dict:
    tab = T.tableau()
    d = c["armatures"]["d"]
    lecture = c["armatures"]["lecture"]
    li = next(x for x in tab["lignes"] if x["d"] == d)
    nb_n = len(T.NOMBRES)
    h1 = (f'<th class="nm" rowspan="2">Ø<small>mm</small></th>'
          f'<th colspan="{nb_n}" class="grp{" on" if lecture == "barres" else ""}">'
          + ("Nombre de barres <small>mm²</small>" if lecture == "barres" else "Barres par mètre · e en cm <small>mm²/m</small>")
          + '</th><th rowspan="2" class="m">Masse<small>kg/m</small></th>'
          f'<th colspan="{len(T.ESPACEMENTS_FIXES)}" class="grp{" on" if lecture == "nappe" else ""}">Espacement fixe <small>mm²/m</small></th>')
    h2 = "".join(f'<th class="k"><b>{n}</b>' + (f"<small>{e}</small>" if lecture == "nappe" else "") + "</th>"
                 for n, e in zip(T.NOMBRES, tab["espacements_n_txt"]))
    h2 += "".join(f'<th class="k"><b>{e}</b><small>cm</small></th>' for e in T.ESPACEMENTS_FIXES)
    corps = ""
    for x in tab["lignes"]:
        on = ' class="on"' if x["d"] == d else ""
        cells = "".join(f"<td>{C.fr(v, 0)}</td>" for v in x["barres_arrondi"])
        cells += f'<td class="m">{x["masse_txt"]}</td>'
        cells += "".join(f"<td>{C.fr(v, 0)}</td>" for v in x["fixes_arrondi"])
        corps += f'<tr{on}><th class="nm">{x["d"]}</th>{cells}</tr>'
    html = f'<table class="db-tab db-deux db-arm"><thead><tr class="r1">{h1}</tr><tr class="r2">{h2}</tr></thead><tbody>{corps}</tbody></table>'
    leg = ("<b>Barres</b> : la section de n barres, en mm²." if lecture == "barres"
           else "<b>Par mètre</b> : n barres par mètre (e = 100/n cm) et les espacements fixes, en mm²/m.")
    leg += " La masse est celle d'une barre, en kg/m."
    pied = (f"<b>{C.esc(T.LEGENDE_IMAGE)}.</b> Le tableau du bureau (« {C.esc(T.IMAGE)} »), recalculé par beton/armatures/tableau.py : "
            f"a = π·Ø²/4, arrondi au mm² ; masse ρ·a avec ρ = {C.fr(T.MASSE_VOLUMIQUE, 0)} kg/m³.")
    unite = "mm²" if lecture == "barres" else "mm²/m"
    grp_n = [(f"{n} barre{'s' if n > 1 else ''}" if lecture == "barres" else f"{n} /m · e {e} cm", C.fr(v, 0), unite, "")
             for n, e, v in zip(T.NOMBRES, tab["espacements_n_txt"], li["barres_arrondi"])]
    fiche = _fiche([(f"Une barre Ø{d}", [("a", C.fr(li["aire"], 1), "mm²", "π·Ø²/4"), ("m", li["masse_txt"], "kg/m", "ρ·a")]),
                    ("Nombre de barres" if lecture == "barres" else "Barres par mètre", [(C.esc(a), b, u, n) for a, b, u, n in grp_n]),
                    ("Espacement fixe", [(f"e = {e} cm", C.fr(v, 0), "mm²/m", "") for e, v in zip(T.ESPACEMENTS_FIXES, li["fixes_arrondi"])])])
    verdict = C.verdict_valeur(li["aire"], "mm²", f"Barre Ø{d}",
                               f"m = {li['masse_txt']}{C.NBSP}kg/m<br>ρ = {C.qte(T.MASSE_VOLUMIQUE, 'kg/m³', 0)} (tableau du bureau)",
                               nd=1, symbole="a", calcul="source", classe="db-verdict")
    a1 = C.fr(li["aire"], 1)
    fx = [_fx(["a", "π·Ø²/4", f"π × {d}² / 4"], f"{a1} mm²"),
          _fx(["m", "ρ·a·10⁻⁶", f"{C.fr(T.MASSE_VOLUMIQUE, 0)} × {a1} × 10⁻⁶"], f"{li['masse_txt']} kg/m")]
    fx2 = [_fx(["A", "n·a", f"4 × {a1}"], f"{C.fr(T.section(4, d), 1)} mm²"),
           _fx(["A/m", "a·100 / e", f"{a1} × 100 / 15"], f"{C.fr(T.section_par_metre(d, 15), 1)} mm²/m")]
    fen = _fenetre(f"Barre Ø{d} · d'où viennent les valeurs", "le tableau du bureau, recalculé", [
        f"Tableau « {C.esc(T.IMAGE)} »", f"ρ = <b>{C.fr(T.MASSE_VOLUMIQUE, 0)} kg/m³</b>", "sections arrondies au mm²"],
        _texte_fx([(f"Une barre Ø{d}", fx), ("Exemples : 4 barres, e = 15 cm", fx2)]),
        '<div class="ck-autres">Chaque case du tableau est relue dans l\'image du bureau par tests/test_armatures_moteur.py.</div>')
    vues = [{"cle": "diametres", "titre": "Diamètres à l'échelle", "court": "Diamètres",
             "svg": DS.diametres("db_diam", d), "legende": f"Ø6 à Ø40 · section et masse linéique · Ø{d} encadré"}]
    return {"ligne": f"Ø{d}", "titre_ligne": "Diamètre", "table": _vue(html, leg, pied), "legende": T.LEGENDE_IMAGE,
            "verdict": verdict, "fiche": fiche, "messages": "", "fenetre": fen, "vues_plus": vues,
            "phrase": f"<b>Ø{d}</b> · a = {C.qte(li['aire'], 'mm²', 1)} · m = {li['masse_txt']}{C.NBSP}kg/m",
            "sources": [f"beton/armatures/tableau.py : le tableau « {C.esc(T.IMAGE)} » recalculé, relu au chiffre près"]}


# ================================================================== rubrique 4 · enrobages
def _element_enrobage(c: Dict) -> Dict:
    """L'élément que le moteur de l'enrobage calcule pour la case choisie : une dalle (sans étrier), la classe
    d'exposition, la classe structurale, le Ø et la tolérance choisis, sans vérification au feu."""
    e = c["enrobages"]
    return {**EM.nouvel_element(c.get("nom") or "Consultation"), "type": "Dalle", "b_min": None, "diam_etrier": 0,
            "classe_expo": e["classe_expo"], "classe_struct": e["classe_struct"], "diam_barre": e["diam_barre"],
            "delta_dev": e["delta_dev"]}


def _expo_court(x: str) -> str:
    return f"{x} · {ED.EXPOSITION_INFO[x][0]}"


def _r_durabilite(c: Dict) -> Dict:
    e = c["enrobages"]
    res = EM.calculer(_element_enrobage(c))
    dur = res["durabilite"]
    expo, struct = e["classe_expo"], e["classe_struct"]
    th = '<th class="nm">Classe</th>' + "".join(
        f'<th class="n{" on" if x == expo else ""}" title="{C.esc(_expo_court(x))}"><span class="s">{C.esc(x)}</span>'
        f'<span class="u">mm</span></th>' for x in ED.EXPO_CLASSES)
    corps = ""
    for s in ED.CLASSES_STRUCTURALES:
        cells = ""
        for i, x in enumerate(ED.EXPO_CLASSES):
            cls = []
            if ED.CLASSE_STRUCTURALE_DEFAULT[x] == s:
                cls.append("pro")
            if s == struct and x == expo:
                cls.append("cur")
            elif x == expo:
                cls.append("col")
            marque = '<i class="mk">●</i>' if "pro" in cls else ""
            titre = f"{s} · {x} : enrobage minimal de durabilité {ED.C_MIN_DUR[s][i]} mm" + (" · classe proposée" if "pro" in cls else "")
            cells += f'<td class="{" ".join(cls)}" title="{C.esc(titre)}">{ED.C_MIN_DUR[s][i]}{marque}</td>'
        corps += f'<tr{" class=" + chr(34) + "on" + chr(34) if s == struct else ""}><th class="nm">{s}</th>{cells}</tr>'
    tab = f'<table class="db-tab db-enr"><thead><tr>{th}</tr></thead><tbody>{corps}</tbody></table>'
    leg = ("<b>c<sub>min,dur</sub> (mm)</b> selon la classe structurale (lignes) et la classe d'exposition (colonnes) — "
           "EN 1992-1-1 tab. 4.4N, valeurs de l'écran Enrobage ; <i class=\"mk\">●</i> la classe proposée par l'écran.")
    pied = ("beton/enrobage/donnees.py (C_MIN_DUR, CLASSE_STRUCTURALE_DEFAULT) · colonnes X0 et XC1 : 5 mm de plus que le "
            "tableau 4.4N de S4 à S6 (et S3 en XC1), côté sûr — relevé D4 de docs/beton/ENROBAGE_MOTEUR.md.")
    # la fiche : les termes du moteur (durabilité), dans l'ordre de l'écran
    termes = [t for t in res["termes"] if t["cle"] in ("c_min_b", "c_min_dur", "plancher", "delta_dur", "c_min", "delta_dev", "c_nom_dur")]
    info = ED.EXPOSITION_INFO[expo]
    fiche = (f'<div class="db-expo"><p><b>{C.esc(expo)} · {C.esc(info[0])}</b></p><p>{C.esc(info[1])}</p>'
             f'<p><span>Exemples</span>{C.esc(info[2])}</p><p><span>La choisir</span>{C.esc(info[3])}</p></div>')
    fiche += _fiche([("Durabilité · dalle, sans étrier", [(C.symb_math(t["symbole"]), C.fr(t["valeur"], 0), t["unite"], t["regle"]) for t in termes]),
                    ("Classe structurale", [("proposée", C.esc(dur["classe_structurale_proposee"]), "", "classe proposée par l'écran Enrobage"),
                                            ("retenue", C.esc(struct), "", ED.AIDE_CLASSE_STRUCT)])])
    verdict = C.verdict_valeur(res["c_nom_dur"], "mm", f"{expo} · {struct} · Ø{e['diam_barre']}",
                               f"c<sub>min</sub> = {C.qte(dur['c_min'], 'mm', 0)} · Δc<sub>dev</sub> = {C.qte(dur['delta_dev'], 'mm', 0)}<br>"
                               f"c<sub>min,dur</sub> = {C.qte(dur['c_min_dur'], 'mm', 0)} (tab. 4.4N)",
                               nd=0, symbole="c<sub>nom</sub>", calcul="source", classe="db-verdict")
    messages = "".join(C.message(C.symb_math(a["message"]), a["grave"]) for a in res["avertissements"]
                       if a["code"] in ("classe_reduite", "contre_sol"))
    messages += C.message(C.esc(ED.AIDE_DELTA_DEV) + " Le calcul complet (feu compris) : module Enrobage.")
    fx = [_fx(["c_min", "max(c_min,b ; c_min,dur ; 10 mm)", f"max({dur['c_min_b']} ; {dur['c_min_dur']} ; {dur['plancher']})"],
              f"{C.fr(dur['c_min'], 0)} mm"),
          _fx(["c_nom", "c_min + Δc_dev", f"{dur['c_min']} + {dur['delta_dev']}"], f"{C.fr(res['c_nom_dur'], 0)} mm")]
    fen = _fenetre(f"Enrobage {C.esc(expo)} · {C.esc(struct)} · d'où viennent les valeurs", "le moteur de l'écran Enrobage, pour une dalle sans étrier",
                   [f"Exposition <b>{C.esc(expo)}</b>", f"Classe structurale <b>{C.esc(struct)}</b>", f"Ø = <b>{e['diam_barre']} mm</b>",
                    f"Δc<sub>dev</sub> = <b>{e['delta_dev']} mm</b>"],
                   _texte_fx([("EN 1992-1-1 §4.4.1", fx)])
                   + [("Les règles", "".join(f'<div class="db-regle"><b>{C.symb_math(t["symbole"])}</b> {C.symb_math(t["regle"])}</div>' for t in termes))])
    vues = [{"cle": "coupe", "titre": "Coupe d'enrobage", "court": "Coupe", "svg": DEN.coupe("db_enrob", res),
             "legende": f"dalle · {expo} · {struct} · Ø{e['diam_barre']} · enrobage nominal {C.fr(res['c_nom_dur'], 0)} mm"}]
    return {"ligne": expo, "titre_ligne": "Exposition", "table": _vue(tab, leg, pied), "legende": "enrobage minimal de durabilité (mm) · EN 1992-1-1 tab. 4.4N",
            "verdict": verdict, "fiche": fiche, "messages": messages, "fenetre": fen, "vues_plus": vues,
            "phrase": f"<b>{C.esc(expo)} · {C.esc(struct)}</b> · c<sub>min,dur</sub> = {C.qte(dur['c_min_dur'], 'mm', 0)} · c<sub>nom</sub> = {C.qte(res['c_nom_dur'], 'mm', 0)} (Ø{e['diam_barre']}, Δc<sub>dev</sub> {e['delta_dev']} mm)",
            "sources": ["beton/enrobage/donnees.py : C_MIN_DUR, CLASSE_STRUCTURALE_DEFAULT, DELTAS_DEV (l'écran Enrobage)",
                        "c<sub>nom</sub> : beton/enrobage/moteur.calculer (une dalle sans étrier)"]}


FEU_ELEMENTS = (("Dalle", "dalle"), ("Voile", "voile"), ("Poutre", "poutre"), ("Poteau", "poteau"))


def _alertes_feu(r: str) -> Dict:
    """Les avertissements du moteur de l'enrobage pour chaque élément vérifié au feu à la durée ``r`` :
    {type: [avertissement du tableau feu]} (codes « tableau_poteau », « tableau_voile »)."""
    out = {}
    for typ, _ in FEU_ELEMENTS:
        var = EM.variantes(typ, r)
        el = {**EM.nouvel_element(), "type": typ, "feu": True, "r": r, "b_min": var[0][0] if var else None,
              "diam_etrier": 0 if typ in ("Dalle", "Voile") else ED.DIAM_ETRIER_DEFAUT}
        out[typ] = [a for a in EM.calculer(el)["avertissements"] if a["code"].startswith("tableau_")]
    return out


def _r_feu(c: Dict) -> Dict:
    r = c["enrobages"]["r"]
    n_pou = max(len(ED.FEU_POUTRE[x]) for x in ED.R_LEVELS)
    n_pot = max(len(ED.FEU_POTEAU[x]) for x in ED.R_LEVELS)
    h1 = (f'<th class="nm" rowspan="2">Durée</th><th class="grp" title="EN 1992-1-2 tab. 5.8">Dalle</th>'
          f'<th class="grp" title="EN 1992-1-2 tab. 5.4">Voile</th>'
          f'<th class="grp" colspan="{n_pou}" title="EN 1992-1-2 tab. 5.5">Poutre · b<sub>min</sub> / a</th>'
          f'<th class="grp" colspan="{n_pot}" title="EN 1992-1-2 tab. 5.2a">Poteau · b<sub>min</sub> / a</th>')
    h2 = ('<th class="k"><b>a</b><small>tab. 5.8</small></th><th class="k"><b>a</b><small>tab. 5.4</small></th>'
          + "".join(f'<th class="k"><b>{i + 1}</b><small>tab. 5.5</small></th>' for i in range(n_pou))
          + "".join(f'<th class="k"><b>{i + 1}</b><small>tab. 5.2a</small></th>' for i in range(n_pot)))
    corps = ""
    for x in ED.R_LEVELS:
        al = _alertes_feu(x)

        def case(a, b=None, alerte=False, typ=""):
            t = f"{x} · {typ} : a = {a} mm" + (f" pour b ≥ {b} mm" if b is not None else "") + (" · à vérifier dans la norme" if alerte else "")
            bb = f"<small>b ≥ {b}</small>" if b is not None else ""
            mk = '<i class="mk">!</i>' if alerte else ""
            return f'<td class="{"dout" if alerte else ""}" title="{C.esc(t)}">{mk}<b>{a}</b>{bb}</td>'
        cells = case(ED.FEU_DALLE[x], typ="dalle", alerte=bool(al["Dalle"])) + case(ED.FEU_VOILE[x], typ="voile", alerte=bool(al["Voile"]))
        pou, pot = ED.FEU_POUTRE[x], ED.FEU_POTEAU[x]
        cells += "".join(case(a, b, bool(al["Poutre"]), "poutre") for b, a in pou) + "<td></td>" * (n_pou - len(pou))
        cells += "".join(case(a, b, bool(al["Poteau"]), "poteau") for b, a in pot) + "<td></td>" * (n_pot - len(pot))
        corps += f'<tr{" class=" + chr(34) + "on" + chr(34) if x == r else ""}><th class="nm">{x}</th>{cells}</tr>'
    tab = f'<table class="db-tab db-deux db-feu"><thead><tr class="r1">{h1}</tr><tr class="r2">{h2}</tr></thead><tbody>{corps}</tbody></table>'
    leg = ("<b>Distance à l'axe a (mm)</b> des armatures principales, méthode tabulée de l'EN 1992-1-2 ; poutre et poteau : "
           "la largeur minimale b<sub>min</sub> (mm) en petit. <i class=\"mk r\">!</i> valeur à vérifier dans la norme (rouge).")
    pied = f"beton/enrobage/donnees.py (tableaux FEU_…) · {C.symb_math(ED.CAPTION_FEU)}"
    al = _alertes_feu(r)

    def var(lst):
        return [(f"b ≥ {b}", f"{a}", "mm", f"largeur minimale {b} mm : distance à l'axe {a} mm") for b, a in lst]
    fiche = _fiche([("Dalle · tab. 5.8", [("a", str(ED.FEU_DALLE[r]), "mm", "dalle pleine portant sur une direction")]),
                    ("Voile · tab. 5.4", [("a", str(ED.FEU_VOILE[r]), "mm", "voile porteur exposé sur une face, μ_fi ≈ 0,7")]),
                    ("Poutre · tab. 5.5", var(ED.FEU_POUTRE[r])),
                    ("Poteau · tab. 5.2a", var(ED.FEU_POTEAU[r]))])
    minutes = int(r[1:])
    verdict = C.verdict_valeur(minutes, "min", f"Résistance au feu {r}",
                               f"dalle : a = {ED.FEU_DALLE[r]}{C.NBSP}mm · voile : a = {ED.FEU_VOILE[r]}{C.NBSP}mm<br>"
                               f"poutre, poteau : selon b<sub>min</sub> (fiche)", nd=0, symbole="R", calcul="source", classe="db-verdict")
    vus = set()
    messages = ""
    for typ, _ in FEU_ELEMENTS:
        for a in al[typ]:
            if a["code"] not in vus:
                vus.add(a["code"])
                messages += C.message(C.symb_math(a["message"]), a["grave"])
    messages += C.message("a est mesurée à l'axe de la barre principale : l'écran Enrobage en tire c<sub>feu,min</sub> = a − Ø/2 − "
                          "Ø<sub>étrier</sub>, puis c<sub>nom</sub> = max(durabilité ; feu).")
    fen = _fenetre(f"Feu {C.esc(r)} · d'où viennent les valeurs", "les tableaux de la méthode tabulée de l'EN 1992-1-2",
                   ["Dalle : tab. 5.8", "Voile : tab. 5.4", "Poutre : tab. 5.5", "Poteau : tab. 5.2a"],
                   [("Les tableaux de l'écran Enrobage", "".join(f'<div class="db-regle"><b>{C.esc(t)}</b> {C.esc(EM.TABLEAUX_FEU[t])} '
                                                                  f'({C.esc(EM.SECTIONS_FEU[t])})</div>' for t, _ in FEU_ELEMENTS)),
                    ("Hypothèse", f'<div class="db-regle">{C.symb_math(ED.CAPTION_FEU)}</div>')],
                   '<div class="ck-autres">Écarts relevés avec la norme : docs/beton/ENROBAGE_MOTEUR.md, défauts D1 et D2.</div>')
    return {"ligne": r, "titre_ligne": "Durée", "table": _vue(tab, leg, pied), "legende": "distance à l'axe a (mm) · EN 1992-1-2",
            "verdict": verdict, "fiche": fiche, "messages": messages, "fenetre": fen, "vues_plus": [],
            "phrase": f"<b>Feu {C.esc(r)}</b> · dalle a = {ED.FEU_DALLE[r]}{C.NBSP}mm · voile a = {ED.FEU_VOILE[r]}{C.NBSP}mm",
            "sources": ["beton/enrobage/donnees.py : FEU_DALLE, FEU_VOILE, FEU_POUTRE, FEU_POTEAU (l'écran Enrobage)",
                        "avertissements : beton/enrobage/moteur.avertissements"]}


def _r_enrobages(c: Dict) -> Dict:
    return _r_durabilite(c) if c["enrobages"]["tableau"] == "durabilite" else _r_feu(c)


# ================================================================== rubrique 5 · ancrages et recouvrements
def _r_ancrages(c: Dict) -> Dict:
    d = _ancrages()
    t = _tableau_ancrage(c, d)
    fichier = C.esc(AN.FICHIER)
    if t is None:
        etat = C.esc(d["etat"])
        corps = (f'<div class="db-vide">{C.icon("file-text")}<b>{etat}</b>'
                 f"<span>Les longueurs d'ancrage et de recouvrement viendront des tableaux du bureau, fournis par l'ingénieur.</span>"
                 f"<span>Ils seront versés dans {fichier} et s'afficheront ici tels quels, sans autre changement.</span>"
                 f"<span>Aucune valeur n'est proposée en attendant.</span></div>")
        verdict = (f'<div class="verdict db-v"><span class="db-vig">{C.icon("file-text")}</span><div class="gov"><b>{etat}</b>'
                   f"Les tableaux de l'ingénieur s'afficheront ici.</div></div>")
        fiche = (f'<div class="db-expo"><p>Le fichier <b>{fichier}</b> attend les tableaux : un tableau par grandeur (ancrage droit, '
                 "recouvrement…), une ligne par cas (classe de béton, adhérence…), une colonne par diamètre.</p>"
                 "<p><span>Format</span>docs/beton/DONNEES_BETON.md § 6 et beton/donnees/ancrages.py.</p></div>")
        messages = C.message(f"{C.esc(d['erreur'])}", True) if d.get("erreur") else ""
        fen = _fenetre("Ancrages et recouvrements · le fichier à remplir", "aucune valeur tant que le bureau n'a pas fourni ses tableaux",
                       [f"<b>{fichier}</b>", f"« {etat} »"],
                       [("Ce que contiendra un tableau", "".join(f'<div class="db-regle">{x}</div>' for x in (
                           "<b>titre</b> · <b>grandeur</b> (l<sub>bd</sub>, l<sub>0</sub>…) · <b>unité</b> (mm, cm, m ou Ø)",
                           "<b>source</b> : d'où vient le tableau (obligatoire)",
                           "<b>conditions</b> : béton, acier, adhérence, forme des barres…",
                           "<b>colonnes</b> : les diamètres ; <b>lignes</b> : un libellé et une valeur par colonne (null : case vide)")))])
        return {"ligne": None, "titre_ligne": "Tableau", "table": f'<div class="db-vue db-attente">{corps}</div>',
                "legende": AN.FICHIER, "verdict": verdict, "fiche": fiche, "messages": messages, "fenetre": fen,
                "vues_plus": [], "phrase": f"<b>{etat}</b>", "sources": [f"{fichier} (vide : à remplir par le bureau)"]}
    i = _ligne_ancrage(c, t)
    li = t["lignes"][i]
    u = t["unite"]
    col = t["colonnes"]

    def ent(v):
        return C.esc(v) if isinstance(v, str) else C.fr(v, 0) if float(v).is_integer() else C.nb(v)

    def val(v):
        return "—" if v is None else C.nb(v, 3)
    th = f'<th class="nm">{C.esc(col["titre"]) or "&nbsp;"}<small>{C.esc(col["unite"])}</small></th>' + "".join(
        f'<th class="k"><b>{ent(v)}</b></th>' for v in col["valeurs"])
    corps = "".join(f'<tr{" class=" + chr(34) + "on" + chr(34) if k == i else ""}><th class="nm">{C.esc(x["libelle"])}</th>'
                    + "".join(f"<td>{val(v)}</td>" for v in x["valeurs"]) + "</tr>" for k, x in enumerate(t["lignes"]))
    tab = f'<table class="db-tab db-anc"><thead><tr>{th}</tr></thead><tbody>{corps}</tbody></table>'
    cond = " · ".join(C.esc(x) for x in t["conditions"])
    leg = f"<b>{C.symb_math(t['grandeur'])}</b> en {C.esc(u)}" + (f" · {cond}" if cond else "")
    pied = f"Source : {C.esc(t['source'])} · {fichier}" + "".join(f" · {C.esc(n)}" for n in t["notes"])
    fiche = _fiche([(t["titre"], [(f"{ent(v)} {C.esc(col['unite'])}".strip(), val(x), u, "") for v, x in zip(col["valeurs"], li["valeurs"])])])
    if t["conditions"]:
        fiche += '<div class="db-expo">' + "".join(f"<p>{C.esc(x)}</p>" for x in t["conditions"]) + "</div>"
    verdict = (f'<div class="verdict db-v" data-calcul="source" title="Voir la source"><span class="db-vig">{C.symb_math(t["grandeur"])}</span>'
               f'<div class="gov"><b>{C.esc(t["titre"])}</b>{C.esc(li["libelle"])} · en {C.esc(u)}</div></div>')
    messages = "".join(C.message(C.esc(n)) for n in t["notes"])
    fen = _fenetre(f"{C.esc(t['titre'])} · la source", "le tableau du bureau, tel que fourni",
                   [f"<b>{C.esc(t['source'])}</b>", f"{fichier}"] + [C.esc(x) for x in t["conditions"]],
                   [("Le tableau", f'<div class="db-regle">{len(t["lignes"])} ligne(s) × {len(col["valeurs"])} colonne(s), '
                                   f"en {C.esc(u)} ; aucune valeur n'est calculée par le site.</div>")])
    return {"ligne": li["libelle"], "titre_ligne": "Ligne", "table": _vue(tab, leg, pied), "legende": f"{t['titre']} · {u}",
            "verdict": verdict, "fiche": fiche, "messages": messages, "fenetre": fen, "vues_plus": [],
            "phrase": f"<b>{C.esc(t['titre'])}</b> · {len(d['tableaux'])} tableau{'x' if len(d['tableaux']) > 1 else ''} du bureau",
            "sources": [f"{fichier} : {C.esc(t['source'])}"]}


RENDUS = {"classes": _r_classes, "aciers": _r_aciers, "armatures": _r_armatures, "enrobages": _r_enrobages, "ancrages": _r_ancrages}


# ------------------------------------------------------------------ le sélecteur de la ligne
def _rangs(c: Dict):
    """Les lignes que le sélecteur propose pour la rubrique : (clé, titre, sous-titre, [rangée], unité de la colonne).
    Chaque rangée porte ``cible`` : le choix qu'elle écrit (« classes/classe », valeur)."""
    rub = c["rubrique"]
    if rub == "classes":
        rows = [{"nom": x["nom"], "col": C.fr(x["fck"], 0), "cur": x["nom"] == c["classes"]["classe"], "cible": ("classes/classe", x["nom"]),
                 "prop": f"f<sub>ck</sub> = {C.qte(x['fck'], 'MPa', 0)} · f<sub>ck,cube</sub> = {C.qte(x['fck_cube'], 'MPa', 0)} · "
                         f"E<sub>cm</sub> = {C.qte(x['Ecm'], 'GPa', 0)} · α<sub>b</sub> = {C.nb(x['alpha_b'])}"} for x in K.classes()]
        return "classes", "Classe de béton", "les classes du bureau · un clic l'ouvre", rows, "f<sub>ck</sub> MPa"
    if rub == "aciers":
        rows = [{"nom": x["nom"], "col": C.fr(x["fyk"], 0), "cur": x["nuance"] == c["aciers"]["nuance"], "cible": ("aciers/nuance", x["nuance"]),
                 "prop": f"f<sub>yd</sub> = {C.qte(x['fyd_bureau'], 'MPa', 0)} (bureau) · {C.qte(x['fyd_ec2'], 'MPa', 0)} (EN 1992-1-1)"}
                for x in K.aciers()]
        return "aciers", "Nuance d'acier", "les nuances des modules béton", rows, "f<sub>yk</sub> MPa"
    if rub == "armatures":
        rows = [{"nom": f"Ø{x['d']}", "col": C.fr(x["aire"], 1), "cur": x["d"] == c["armatures"]["d"], "cible": ("armatures/d", x["d"]),
                 "prop": f"a = {C.qte(x['aire'], 'mm²', 1)} · m = {x['masse_txt']}{C.NBSP}kg/m"} for x in T.tableau()["lignes"]]
        return "armatures", "Diamètre", "les diamètres du tableau du bureau", rows, "a mm²"
    if rub == "enrobages" and c["enrobages"]["tableau"] == "durabilite":
        s = c["enrobages"]["classe_struct"]
        rows = [{"nom": x, "col": str(ED.C_MIN_DUR[s][i]), "cur": x == c["enrobages"]["classe_expo"], "cible": ("enrobages/classe_expo", x),
                 "prop": C.esc(_expo_court(x))} for i, x in enumerate(ED.EXPO_CLASSES)]
        return "enrobages", "Classe d'exposition", f"c<sub>min,dur</sub> en classe {s}", rows, "c<sub>min,dur</sub> mm"
    if rub == "enrobages":
        rows = [{"nom": x, "col": str(ED.FEU_DALLE[x]), "cur": x == c["enrobages"]["r"], "cible": ("enrobages/r", x),
                 "prop": f"dalle a = {ED.FEU_DALLE[x]} mm · voile a = {ED.FEU_VOILE[x]} mm"} for x in ED.R_LEVELS]
        return "enrobages", "Durée de résistance au feu", "la distance à l'axe d'une dalle en colonne", rows, "a dalle mm"
    d = _ancrages()
    t = _tableau_ancrage(c, d)
    lignes = t["lignes"] if t else []
    i = _ligne_ancrage(c, t) if t else 0
    rows = [{"nom": x["libelle"], "col": C.nb(x["valeurs"][0], 3) if x["valeurs"][0] is not None else "—", "cur": k == i,
             "cible": ("ancrages/ligne", k), "prop": C.esc(" · ".join("—" if v is None else C.nb(v, 3) for v in x["valeurs"]))}
            for k, x in enumerate(lignes)]
    return "ancrages", "Ligne du tableau", C.esc(t["titre"]) if t else "aucun tableau", rows, C.esc(t["unite"]) if t else ""


def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Le sélecteur commun sur les lignes du tableau de la rubrique : la ligne ouverte ✓ ; ni optimal ni taux (une
    consultation)."""
    c = _cons(etat)
    _k, titre, sous, rangs, unite = _rangs(c)
    rows = [{"nom": r["nom"], "col": r["col"], "taux": None, "niveau": "c", "cur": r["cur"], "opt": False, "prop": r["prop"]} for r in rangs]
    i_c = next((i for i, r in enumerate(rows) if r["cur"]), 0)
    return C.selecteur(role, titre, sous, "", rows, max(0, i_c - 4), unite, etoile="", pastille="")


# ------------------------------------------------------------------ le cadre Données
def _options_expo() -> List:
    return [(x, _expo_court(x)) for x in ED.EXPO_CLASSES]


def formulaire(etat: Dict, rendu: Dict) -> str:
    c = _cons(etat)
    L = C.ligne
    rub = c["rubrique"]
    s1 = (L("Nom", C.texte(["nom"], c["nom"], "ex. Dalle R+1 · enrobage"), full=True)
          + L("Rubrique", C.choix(["rubrique"], rub, RUBRIQUES), full=True))
    s1 += C.aide({"classes": "Les classes de béton du bureau : la méthode du bureau (α<sub>b</sub>, μ, τ<sub>adm</sub>) et les "
                             "valeurs du tableau 3.1 de l'EN 1992-1-1.",
                  "aciers": "B400 et B500 : f<sub>yd</sub> de la méthode du bureau et de l'EN 1992-1-1, E<sub>s</sub>.",
                  "armatures": "Le tableau des sections du bureau : n barres, par mètre, espacements fixes, masses.",
                  "enrobages": "c<sub>min,dur</sub> par classe structurale et exposition ; distances à l'axe au feu.",
                  "ancrages": "Les tableaux d'ancrage et de recouvrement du bureau (à venir)."}[rub])
    # 2 · les choix de la rubrique
    if rub == "classes":
        s2 = (L("Colonnes", C.segments(["classes", "groupe"], c["classes"]["groupe"], [("ec2", "EN 1992-1-1"), ("bureau", "Bureau")]), full=True)
              + L("Classe", C.choix(["classes", "classe"], c["classes"]["classe"], [(n, n) for n in K.noms_classes()]), full=True))
        resume2 = f"{C.esc(c['classes']['classe'])} · {'EN 1992-1-1' if c['classes']['groupe'] == 'ec2' else 'bureau'}"
    elif rub == "aciers":
        s2 = L("Nuance", C.segments(["aciers", "nuance"], c["aciers"]["nuance"], [(n, f"B{n}") for n in PE.NUANCES]), full=True)
        resume2 = f"B{c['aciers']['nuance']}"
    elif rub == "armatures":
        s2 = (L("Lecture", C.segments(["armatures", "lecture"], c["armatures"]["lecture"], LECTURES), full=True)
              + L("Diamètre", C.choix(["armatures", "d"], c["armatures"]["d"],
                                      [(d, f"Ø{d} · {C.fr(T.arrondi(T.aire(d)), 0)} mm²") for d in T.DIAMETRES]), full=True))
        resume2 = f"Ø{c['armatures']['d']} · {dict(LECTURES)[c['armatures']['lecture']]}"
    elif rub == "enrobages":
        e = c["enrobages"]
        s2 = L("Tableau", C.segments(["enrobages", "tableau"], e["tableau"], TABLEAUX_ENROBAGE), full=True)
        if e["tableau"] == "durabilite":
            pro = ED.CLASSE_STRUCTURALE_DEFAULT[e["classe_expo"]]
            s2 += (L("Exposition", C.choix(["enrobages", "classe_expo"], e["classe_expo"], _options_expo()), full=True)
                   + L("Classe structurale", C.choix(["enrobages", "classe_struct"], e["classe_struct"],
                                                     [(s, f"{s} · proposée" if s == pro else s) for s in ED.CLASSES_STRUCTURALES]), full=True)
                   + L("Ø de la barre", C.choix(["enrobages", "diam_barre"], e["diam_barre"], [(d, f"Ø{d}") for d in DIAMETRES_ENROBAGE]), full=True)
                   + L("Tolérance", C.segments(["enrobages", "delta_dev"], e["delta_dev"], [(x, f"{x} mm") for x in ED.DELTAS_DEV]),
                       "", f"Δc_dev : {ED.AIDE_DELTA_DEV}", full=True)
                   + C.aide(C.esc(ED.AIDE_CLASSE_STRUCT)))
            resume2 = f"{C.esc(e['classe_expo'])} · {C.esc(e['classe_struct'])} · Ø{e['diam_barre']} · Δc<sub>dev</sub> {e['delta_dev']} mm"
        else:
            s2 += L("Durée", C.choix(["enrobages", "r"], e["r"], [(x, x) for x in ED.R_LEVELS]), full=True)
            resume2 = f"Feu · {C.esc(e['r'])}"
    else:
        d = _ancrages()
        t = _tableau_ancrage(c, d)
        if t is None:
            s2 = C.aide(f"<b>{C.esc(d['etat'])}</b> : l'ingénieur fournira les tableaux du bureau ; ils seront versés dans "
                        f"{C.esc(AN.FICHIER)} et s'afficheront ici. Aucune valeur n'est proposée en attendant.")
            resume2 = C.esc(d["etat"])
        else:
            s2 = (L("Tableau", C.choix(["ancrages", "tableau"], t["cle"], [(x["cle"], x["titre"]) for x in d["tableaux"]]), full=True)
                  + L("Ligne", C.choix(["ancrages", "ligne"], _ligne_ancrage(c, t), [(k, x["libelle"]) for k, x in enumerate(t["lignes"])]), full=True))
            resume2 = f"{C.esc(t['titre'])} · {C.esc(t['lignes'][_ligne_ancrage(c, t)]['libelle'])}"
    s3 = "".join(_src(x) for x in rendu["sources"])
    return (C.section(1, "Rubrique", C.esc(NOMS[rub]), s1, True, "ru")
            + C.section(2, "Choix", resume2, s2, True, "ch")
            + C.section(3, "Sources", f"{len(rendu['sources'])} source{'s' if len(rendu['sources']) > 1 else ''}", s3, False, "so"))


# ------------------------------------------------------------------ le résultat
def _resume_ligne(c: Dict, rendu: Dict) -> str:
    return f"{NOMS[c['rubrique']]}" + (f" · {rendu['ligne']}" if rendu["ligne"] else "")


def resume_note(etat: Dict, rendu: Dict) -> Dict:
    c = _cons(etat)
    return {"taux": None, "niveau": "i", "statut": C.STATUTS["i"][1],
            "gouvernant": f"{c.get('nom') or 'Consultation'} · {_resume_ligne(c, rendu)}"}


def _resume_court(c: Dict) -> str:
    """Ce que dit une consultation dans la liste de la note, sans la rendre entière : la rubrique et son choix."""
    rub = c["rubrique"]
    choix = {"classes": c["classes"]["classe"], "aciers": f"B{c['aciers']['nuance']}", "armatures": f"Ø{c['armatures']['d']}",
             "enrobages": (f"{c['enrobages']['classe_expo']} · {c['enrobages']['classe_struct']}" if c["enrobages"]["tableau"] == "durabilite"
                           else f"feu {c['enrobages']['r']}"),
             "ancrages": "à venir"}[rub]
    return f"{COURTS[rub]} · {choix}"


def elements(etat: Dict) -> str:
    items = ""
    cs = etat["consultations"]
    for i, c in enumerate(cs):
        sup = f'<button class="x" data-supprimer="{i}" title="Supprimer">{C.icon("x")}</button>' if len(cs) > 1 else ""
        items += (f'<div class="ni{" on" if i == etat["actif"] else ""}" data-activer="{i}"><span class="t">{C.esc(c.get("nom") or f"Consultation {i + 1}")} '
                  f'<span>· {C.esc(_resume_court(c))}</span></span><span class="pct c">—</span>{sup}</div>')
    return (f'<div class="elements"><div class="nl-h"><span class="kicker">Dans cette note · {len(cs)}</span>'
            f'<button class="nl-add" data-dupliquer="1" title="Copier la consultation ouverte">{C.icon("copy")}Copier</button></div>{items}'
            f'<div class="nl-add" data-ajouter="1">{C.icon("plus")}Ajouter une consultation</div></div>')


def resultat(etat: Dict) -> Dict:
    c = _cons(etat)
    rendu = RENDUS[c["rubrique"]](c)
    rows = ""
    if rendu["ligne"] is not None:
        rows += _ligne(rendu["titre_ligne"], C.esc(rendu["ligne"]), calcul="source", pk="ligne", gov=True)
    rows += _ligne("D'où viennent les valeurs", C.esc(NOMS[c["rubrique"]]), calcul="source")
    corps = rendu["verdict"] + f'<div class="els">{rows}</div>' + rendu["messages"] + rendu["fiche"]
    pied = (f'<div class="pfoot"><div class="mass"><span>Aide-mémoire du bureau</span><b class="db-nb">{C.esc(NOMS[c["rubrique"]])}</b></div></div>')
    vues = [{"cle": "tableau", "titre": NOMS[c["rubrique"]], "court": "Tableau", "svg": _normes(rendu["table"]), "legende": rendu["legende"]}]
    vues += rendu["vues_plus"]
    return {"resultat": _normes(f'<div class="pbody">{corps}</div>{pied}'), "volets": _normes(rendu["fenetre"]), "vues": vues,
            "phrase": _normes(rendu["phrase"]), "statut_html": C.statut("i"), "resume": resume_note(etat, rendu),
            "donnees": _normes(formulaire(etat, rendu)), "elements": elements(etat)}

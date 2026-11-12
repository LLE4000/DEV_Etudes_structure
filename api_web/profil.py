# -*- coding: utf-8 -*-
"""Le module « Choix de profilé » du site : l'état d'une note (un ou plusieurs éléments, chacun avec ses efforts, ses
familles de profilés et son profilé retenu), les champs de saisie, le résultat du profilé retenu, la grande fenêtre de
calcul, le sélecteur de profilés et les vues (section à l'échelle, tableau de l'écran, taux / masse, flambement) —
rendus en fragments HTML.

Le calcul est celui du moteur ``acier/choix_profil`` (extrait de l'écran Streamlit modules/choix_profile.py, dont il
reproduit les résultats au bit près — tests/test_choix_profil_moteur.py). Ce fichier MET EN FORME : il ne calcule rien.
Ce que le site fait de plus que l'écran (le plus léger qui tient, le gouvernant, les formules développées, les
avertissements) vient lui aussi du moteur, qui le nomme comme un ajout.

L'écran est l'établi COMPACT (api_web/etabli_compact.py, la mise en forme des poutres — 11/11/2026) : à gauche la ligne
de l'élément ouvert (son nom, la liste, copier, supprimer) et quatre rubriques en accordéon — Efforts, Profilés comparés,
Instabilités, Hypothèses (secondaire) ; au centre les vues sous leur en-tête technique (le profilé retenu, l'acier, la
sélection ; les efforts dans la bande) ; à droite la tête, le verdict et le bloc « Vérifications » : le profilé retenu
(il s'y choisit, avec le plus léger qui tient) puis chaque vérification, qui dit ce qu'elle compare.
"""
import copy
import re
from typing import Dict, List, Optional, Sequence

from acier.choix_profil import calcul as CA
from acier.choix_profil import entree as EN
from acier.choix_profil import moteur as MO
from acier.profiles import catalogue as CAT
from acier.profiles import familles as FA

from . import commun as C
from . import dessin_profil as DP
from . import etabli_compact as E

MODULE = "profil"
PAGE = "choix-profile"

ROLE = "profil"                 # le seul sélecteur du module : le profilé de l'élément ouvert
GENRE = E.Genre("Éléments", "l'élément", "cet élément", "un élément")
CLASSES_BLOC = {"p": "ok", "f": "warn", "r": "nok"}
NB = C.NBSP
FINE = "\u202f"                # l'espace fine des milliers (celle de commun.fr)

# ------------------------------------------------------------------ le parcours propre : des états que l'écran n'atteint
# pas en un seul clic depuis une note neuve (flambement calculé, aucun profilé, choix hors des familles…)
_FLAMB = {"version": 1, "actif": 0, "elements": [dict(EN.nouvel_element("P1"), familles=["HEB"], My=60.0, Vz=40.0, avec_N=True, N=600.0,
                                                      flambement=True, Lcr_y=6.0, Lcr_z=3.0, courbe="c")]}
_TOUT = {"version": 1, "actif": 0, "elements": [dict(EN.nouvel_element("P1"), familles=["HEA", "HEB"], My=90.0, Vz=70.0, avec_N=True, N=400.0,
                                                     avec_Mz=True, Mz=15.0, avec_Vy=True, Vy=30.0, avec_T=True, T=3.0, flambement=True,
                                                     Lcr_y=5.0, Lcr_z=2.5, deversement=True, L_LT=4.0, gamma_M0=1.05, gamma_M1=1.1)]}
PARCOURS = [
    {"action": "calculer", "etat": _FLAMB},
    {"action": "calculer", "etat": _TOUT},
    {"action": "selecteur", "role": ROLE, "etat": _TOUT, "familles": ["HD", "HEM"]},
    {"action": "choisir", "role": ROLE, "nom": "IPE 300", "etat": _FLAMB},
    {"action": "modifier", "chemin": ["My"], "valeur": "5000"},
    {"action": "modifier", "chemin": ["Iy_min"], "valeur": "1e7"},
    {"action": "modifier", "chemin": ["My"], "valeur": "1e308"},
    {"action": "modifier", "chemin": ["auto"], "valeur": False},
    {"action": "modifier", "chemin": ["N"], "valeur": "-300"},
    {"action": "modifier", "chemin": ["gamma_M0"], "valeur": "2"},
    {"action": "modifier", "chemin": ["Lcr_y"], "valeur": "-1"},
]


# ------------------------------------------------------------------ l'état de la note : des éléments
def nouvel_etat() -> Dict:
    """La note neuve : un élément aux valeurs de l'écran Streamlit à son ouverture (HEA, S235, aucun effort)."""
    return {"version": 1, "elements": [EN.nouvel_element("P1")], "actif": 0}


def normaliser(etat: Dict) -> Dict:
    """L'état reçu de l'écran (ou d'un fichier .stab) relu par le moteur ; un élément illisible lève une erreur claire."""
    if not isinstance(etat, dict) or "elements" not in etat:
        return nouvel_etat()
    els = etat["elements"]
    if not isinstance(els, list) or not els:
        raise ValueError("les éléments de la note sont illisibles (une liste d'éléments est attendue)")
    out = {"version": 1, "elements": []}
    for i, e in enumerate(els):
        try:
            out["elements"].append(EN.normaliser(e))
        except ValueError as ex:
            raise ValueError(f"élément {i + 1} : {ex}") from None
    a = etat.get("actif", 0)
    a = a if isinstance(a, int) and not isinstance(a, bool) else 0
    out["actif"] = min(max(0, a), len(out["elements"]) - 1)
    return out


def _el(etat: Dict) -> Dict:
    return etat["elements"][etat["actif"]]


def _nom_libre(etat: Dict) -> str:
    pris = {e.get("nom") for e in etat["elements"]}
    k = len(etat["elements"]) + 1
    while f"P{k}" in pris:
        k += 1
    return f"P{k}"


def _indice(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def gerer_elements(etat: Dict, action: str, index: Optional[int] = None) -> None:
    """Les éléments de la note : en ouvrir un, en ajouter, en copier un (``index`` ; sans : l'ouvert — la copie s'insère
    juste après), en supprimer un (jamais le dernier)."""
    els = etat["elements"]
    if action == "activer":
        etat["actif"] = min(max(0, _indice(index)), len(els) - 1)
    elif action == "ajouter":
        els.append(EN.nouvel_element(_nom_libre(etat)))
        etat["actif"] = len(els) - 1
    elif action == "dupliquer":
        i = etat["actif"] if index is None else min(max(0, _indice(index)), len(els) - 1)
        e = copy.deepcopy(els[i])
        e["nom"] = f"{e.get('nom') or 'Élément'} (copie)"
        els.insert(i + 1, e)
        etat["actif"] = i + 1
    elif action == "supprimer" and len(els) > 1:
        i = min(max(0, _indice(index)), len(els) - 1)
        del els[i]
        if etat["actif"] > i or etat["actif"] >= len(els):
            etat["actif"] = max(0, etat["actif"] - 1)


# ------------------------------------------------------------------ la saisie
# le nom de chaque grandeur dans un message (le français du bureau, pas la clé du moteur)
LIBELLES = {"My": "M_y,Ed", "Vz": "V_z,Ed", "N": "N_Ed", "Mz": "M_z,Ed", "Vy": "V_y,Ed", "T": "T_Ed", "gamma_M0": "γ_M0", "gamma_M1": "γ_M1",
            "Iy_min": "I_y min.", "Lcr_y": "L_cr,y", "Lcr_z": "L_cr,z", "L_LT": "L_LT"}


def modifier(etat: Dict, chemin: Sequence, valeur) -> Optional[str]:
    """Écrit une saisie dans l'élément ouvert. ``chemin`` : [« My »], [« familles », « HEA »] (une case), [« auto »] (le
    profilé retenu suit l'optimal de l'écran)… Une valeur hors du domaine de l'écran Streamlit est REFUSÉE avec un
    message (l'écran la bornerait) ; l'élément reste alors intact."""
    el = _el(etat)
    avant = copy.deepcopy(el)
    chemin = list(chemin)
    k = chemin[0] if chemin else None
    try:
        if k == "familles" and len(chemin) == 2:
            f = str(chemin[1])
            if f not in EN.familles_disponibles():
                return f"Famille inconnue : « {f} »"
            fams = [x for x in el["familles"] if x != f] + ([f] if valeur in (True, "true", "1", 1) else [])
            el["familles"] = fams
        elif k == "auto" and len(chemin) == 1:
            # coché : le profilé retenu suit l'optimal de l'écran ; décoché : le retenu du moment est FIGÉ
            if valeur in (True, "true", "1", 1):
                el["profil"] = None
            elif el["profil"] is None:
                el["profil"] = MO.verifier(el)["retenu"]
        elif len(chemin) != 1 or k not in EN.CLES or k == "familles":
            return f"Champ inconnu : « {'/'.join(str(c) for c in chemin)} »"
        elif k in EN.CASES:
            el[k] = bool(valeur) if isinstance(valeur, bool) else str(valeur).lower() in ("true", "1", "on")
        elif k in ("nom", "acier", "courbe"):
            el[k] = str(valeur)
        elif k == "profil":
            el[k] = str(valeur) if valeur not in (None, "") else None
        else:
            x = C.lire_nombre(valeur)
            if x is None:
                return f"Nombre illisible : « {valeur} »"
            if k in EN.POSITIFS and x < 0:
                return f"{LIBELLES[k]} ne peut pas être négatif (l'écran Streamlit le borne à 0) : saisie refusée"
            if k in ("gamma_M0", "gamma_M1") and not EN.GAMMA_MIN <= x <= EN.GAMMA_MAX:
                return (f"{LIBELLES[k]} doit être compris entre {C.nb(EN.GAMMA_MIN)} et {C.nb(EN.GAMMA_MAX)} "
                        f"(bornes de l'écran Streamlit) : saisie refusée")
            if EN.hors_domaine(k, x):       # l'écran l'accepterait, mais son calcul y déborde (entree.EFFORT_MAX…)
                if k in EN.EFFORTS:
                    return f"{LIBELLES[k]} : au plus {_n(EN.EFFORT_MAX)} en valeur absolue (au-delà, le calcul déborde) : saisie refusée"
                if k in EN.LONGUEURS:
                    return (f"{LIBELLES[k]} : 0 (sans vérification), ou de {_n(EN.LONGUEUR_MIN * 1e3)} mm à {_n(EN.LONGUEUR_MAX)} m "
                            f"(au-delà, le calcul déborde) : saisie refusée")
                return f"{LIBELLES[k]} : au plus {_n(EN.IY_MIN_MAX)} cm⁴ : saisie refusée"
            el[k] = x
        etat["elements"][etat["actif"]] = EN.normaliser(el)
    except (ValueError, KeyError, TypeError) as e:
        etat["elements"][etat["actif"]] = avant
        return f"Saisie refusée : {e}"
    return None


def choisir_section(etat: Dict, role: str, nom: str) -> None:
    """Le clic dans le sélecteur : le profilé devient celui de l'élément. Choisi dans une famille que l'élément ne
    compare pas, sa famille est ajoutée aux familles cochées (l'écran Streamlit l'exige aussi : on ne peut y
    sélectionner qu'un profilé du tableau)."""
    if role != ROLE:
        raise KeyError(f"sélecteur inconnu : {role!r}")
    el = _el(etat)
    if nom not in CA.noms():
        raise ValueError(f"profilé inconnu : {nom!r}")
    fam = CA.profil(nom)["type"]
    if el["familles"] and fam not in el["familles"]:
        el["familles"] = el["familles"] + [fam]
    el["profil"] = nom
    etat["elements"][etat["actif"]] = EN.normaliser(el)


# ------------------------------------------------------------------ nombres
def _n(x: Optional[float], dec: int = 3) -> str:
    """Une donnée du catalogue ou une saisie, sans zéro inutile (le format des tableaux de l'écran,
    ``fmt_no_trailing_zeros``), à la française, les milliers séparés."""
    if x is None:
        return "—"
    s = f"{float(x):.{dec}f}"
    if "." in s:                         # (commun.nb ôterait aussi les zéros de la partie entière quand dec = 0)
        s = s.rstrip("0").rstrip(".")
    if float(s) == 0:
        s = "0"
    ent, _, d = s.partition(".")
    signe = "−" if ent.startswith("-") else ""
    ent = f"{int(ent.lstrip('-')):,}".replace(",", FINE)
    return signe + ent + ("," + d if d else "")


def _r(x: Optional[float]) -> str:
    """Une résistance ou un effort calculé : au dixième (le format de l'écran, ``:.1f``)."""
    return C.fr(x, 1)


def _t(x: Optional[float]) -> str:
    """Un taux sans dimension : trois décimales (le format de l'écran, ``:.3f``)."""
    return C.fr(x, 3)


_GREC = re.compile(r"([χλφ]̄?)_([A-Za-z0-9]+(?:,[A-Za-z0-9]+)*)")


def _grec(h: str) -> str:
    """χ_y, λ̄_z, φ_y : les indices des lettres que ``commun.symb_math`` ne connaît pas (texte déjà échappé)."""
    return _GREC.sub(r"\1<sub>\2</sub>", h).replace("λ\u0304", '<span class="pf-barre">λ</span>')


def _sub(s: str) -> str:
    return _grec(C.symb_math(s))


# ------------------------------------------------------------------ le cadre Données
def _efforts_txt(r: Dict, html: bool = True) -> str:
    """Les efforts saisis, dans l'ordre de l'écran (M_y et V_z toujours ; les autres quand leur case est cochée)."""
    el = r["element"]
    morceaux = [("M_y,Ed", el["My"], "kN·m"), ("V_z,Ed", el["Vz"], "kN")]
    for case, s, k, u in (("avec_N", "N_Ed", "N", "kN"), ("avec_Mz", "M_z,Ed", "Mz", "kN·m"), ("avec_Vy", "V_y,Ed", "Vy", "kN"),
                          ("avec_T", "T_Ed", "T", "kN·m")):
        if el[case]:
            morceaux.append((s, el[k], u))
    if html:
        return " · ".join(f"{C.symb_math(s)} = {_n(v)}{NB}{u}" for s, v, u in morceaux)
    return " · ".join(f"{s} = {_n(v)} {u}" for s, v, u in morceaux)


def _selection_txt(r: Dict) -> str:
    """La sélection en une ligne (du TEXTE) : ce que l'écran compare et ce qui tient."""
    fams = ", ".join(r["familles"]) if r["element"]["familles"] else "toutes les familles"
    n, n_t = r["n_profils"], r["n_tiennent"]
    if not n:
        return f"aucun profilé à comparer ({fams})"
    tient = "aucun ne tient" if not n_t else ("1 tient" if n_t == 1 else f"{n_t} tiennent")
    return f"{'1 profilé comparé' if n == 1 else f'{n} profilés comparés'} ({fams}) · {tient}"


def _candidat_txt(r: Dict, nom: Optional[str]) -> str:
    """Un profilé de la sélection en une ligne (HTML) : « HEB 180 · 51,2 kg/m · 99 % » ; « aucun » s'il n'y en a pas."""
    li = next((x for x in r["lignes"] if x["nom"] == nom), None) if nom else None
    if not li:
        return "aucun"
    return f"<b>{C.esc(nom)}</b> · {_n(li['masse'], 1)}{NB}kg/m · {C.pc(C.pct(li['taux']))}{NB}%"


def formulaire(etat: Dict, r: Dict) -> str:
    """Les quatre rubriques de l'élément ouvert : 1 Efforts (ELU), 2 Profilés comparés (l'acier, les familles, le filtre,
    le profilé retenu), 3 Instabilités, 4 Hypothèses (secondaire). Le nom de l'élément se modifie sur sa ligne, en tête
    de la colonne."""
    el = _el(etat)
    L = C.ligne
    num = C.num
    # 1 · les efforts
    s1 = (L("moment", num(["My"], el["My"], "kN·m", dec=4), "M<sub>y,Ed</sub>", "Moment autour de l'axe fort y-y (valeur absolue).")
          + L("tranchant", num(["Vz"], el["Vz"], "kN", dec=4), "V<sub>z,Ed</sub>", "Parallèle à l'âme (valeur absolue).")
          + '<div class="grp">Autres efforts</div>'
          + C.case(["avec_N"], el["avec_N"], "Effort normal N<sub>Ed</sub> (compression +)"))
    if el["avec_N"]:
        s1 += L("normal", num(["N"], el["N"], "kN", dec=4), "N<sub>Ed</sub>", "Positif en compression ; négatif : traction.")
    s1 += C.case(["avec_Mz"], el["avec_Mz"], "Moment M<sub>z,Ed</sub> (axe faible)")
    if el["avec_Mz"]:
        s1 += L("moment", num(["Mz"], el["Mz"], "kN·m", dec=4), "M<sub>z,Ed</sub>", "Moment autour de l'axe faible z-z.")
    s1 += C.case(["avec_Vy"], el["avec_Vy"], "Effort tranchant V<sub>y,Ed</sub>")
    if el["avec_Vy"]:
        s1 += L("tranchant", num(["Vy"], el["Vy"], "kN", dec=4), "V<sub>y,Ed</sub>", "Parallèle aux semelles.")
    s1 += C.case(["avec_T"], el["avec_T"], "Torsion T<sub>Ed</sub>")
    if el["avec_T"]:
        s1 += (L("torsion", num(["T"], el["T"], "kN·m", dec=4), "T<sub>Ed</sub>")
               + C.aide("La torsion n'entre dans aucun calcul de l'écran Streamlit : elle n'est pas vérifiée."))
    # 2 · les profilés comparés : l'acier, les familles, le filtre, le profilé retenu
    cases = "".join(C.case(["familles", f], f in el["familles"], C.esc(f)) for f in r["familles_disponibles"])
    tous = not el["familles"]
    s2 = (L("Acier", C.segments(["acier"], el["acier"], [(a, a) for a in EN.ACIERS]), full=True)
          + C.aide(f"f<sub>y</sub> = {r['acier']['fy']} N/mm² (la nuance de l'écran, sans réduction pour l'épaisseur).")
          + '<div class="grp">Familles comparées</div>'
          + f'<div class="pf-fams">{cases}</div>'
          + C.aide("Aucune famille cochée : <b>toutes</b> sont comparées (comme l'écran Streamlit)." if tous else
                   f"{r['n_profils']} profilés comparés, {r['n_tiennent']} tiennent.")
          + L("filtre", num(["Iy_min"], el["Iy_min"], "cm⁴", dec=4, mini=0), "I<sub>y,min</sub>",
              "Inertie minimale : les profilés d'inertie I_y inférieure sont écartés (un critère de flèche, par exemple) ; 0 : sans filtre.")
          + '<div class="grp">Profilé retenu</div>'
          + C.case(["auto"], el["profil"] is None, "Suivre l'optimal (le plus chargé qui tient)"))
    optimal = f"Optimal de l'écran : {_candidat_txt(r, r.get('optimal'))}."
    if el["profil"] is not None:
        s2 += C.aide(f"<b>{C.esc(el['profil'])}</b> est choisi à la main (ligne « Profilé retenu » du résultat) ; "
                     f"cocher la case le rend au choix de l'écran. {optimal}")
    else:
        s2 += C.aide(f"{optimal} Un autre profilé se choisit dans le résultat, ligne « Profilé retenu ».")
    # 3 · les instabilités
    s3 = C.case(["flambement"], el["flambement"], "Vérifier le flambement (si N<sub>Ed</sub> &gt; 0)")
    if el["flambement"]:
        s3 += (L("selon y-y", num(["Lcr_y"], el["Lcr_y"], "m", dec=4, mini=0), "L<sub>cr,y</sub>", "Longueur de flambement autour de y-y ; 0 : pas de flambement selon y.")
               + L("selon z-z", num(["Lcr_z"], el["Lcr_z"], "m", dec=4, mini=0), "L<sub>cr,z</sub>", "Longueur de flambement autour de z-z ; 0 : pas de flambement selon z.")
               + L("Courbe", C.choix(["courbe"], el["courbe"], [(k, f"{k} · α = {C.nb(a)}") for k, a, _ in EN.COURBES]), full=True))
        if not el["avec_N"] or el["N"] <= 0:
            s3 += C.aide("Le flambement n'est calculé qu'en compression : cocher N<sub>Ed</sub> et le saisir positif.")
    s3 += C.case(["deversement"], el["deversement"], "Vérifier le déversement")
    if el["deversement"]:
        s3 += (L("non tenue", num(["L_LT"], el["L_LT"], "m", dec=4, mini=0), "L<sub>LT</sub>", "Longueur non maintenue latéralement.")
               + C.aide("L'écran Streamlit ne calcule pas le déversement (ni M<sub>cr</sub> ni χ<sub>LT</sub>) : cette longueur n'est pas utilisée."))
    # 4 · les hypothèses : les coefficients partiels
    s4 = (C.ligne2(("γ<sub>M0</sub>", num(["gamma_M0"], el["gamma_M0"], "", dec=4, mini=EN.GAMMA_MIN, maxi=EN.GAMMA_MAX),
                    "Résistances de section : N_Rd, M_Rd, V_Rd (entre 0,8 et 1,5)."),
                   ("γ<sub>M1</sub>", num(["gamma_M1"], el["gamma_M1"], "", dec=4, mini=EN.GAMMA_MIN, maxi=EN.GAMMA_MAX),
                    "Flambement : N_b,Rd (entre 0,8 et 1,5)."))
          + C.aide("Section élastique (W<sub>el</sub>) · interaction linéaire N + M · f<sub>y</sub> de la nuance, sans réduction pour l'épaisseur."))
    fams = ", ".join(el["familles"]) if el["familles"] else "toutes les familles"
    inst = " · ".join(x for x in (f"flambement, courbe {el['courbe']}" if el["flambement"] else "",
                                  "déversement" if el["deversement"] else "") if x) or "aucune"
    efforts = " · ".join(f"{s} {_n(v)}" for s, v in (("M<sub>y</sub>", el["My"]), ("V<sub>z</sub>", el["Vz"]))) \
        + (f" · N {_n(el['N'])}" if el["avec_N"] else "")
    return (C.section(1, "Efforts (ELU)", efforts, s1, True, "ef")
            + C.section(2, "Profilés comparés", f"{el['acier']} · {C.esc(fams)}", s2, False, "pf")
            + C.section(3, "Instabilités", inst, s3, False, "in")
            + C.section(4, "Hypothèses", f"γ<sub>M0</sub> {C.nb(el['gamma_M0'])} · γ<sub>M1</sub> {C.nb(el['gamma_M1'])}", s4, False, "co",
                        secondaire=True))


def _dims(e: Dict, r: Optional[Dict]) -> str:
    """« · HEB 180 · S235 » : ce qu'on lit d'un élément dans la liste de la note (HTML)."""
    return f"· {C.esc((r or {}).get('retenu') or 'aucun profilé')} · {C.esc(e['acier'])}"


def arbre(etat: Dict, r: Dict, resultats: List[Dict]) -> str:
    """La colonne de gauche COMPACTE (api_web/etabli_compact.arbre_compact, la même que la poutre) : la ligne de
    l'élément ouvert (son nom, qu'un clic modifie ; la liste des éléments de la note ; copier, supprimer), que l'écran
    pose dans l'en-tête du panneau, puis ses quatre rubriques."""
    noms = [(e.get("nom") or f"P{i + 1}", _dims(e, x), False) for i, (e, x) in enumerate(zip(etat["elements"], resultats))]
    return E.arbre(GENRE, noms, etat["actif"], formulaire(etat, r), "choixprofil", compact=True)


# ------------------------------------------------------------------ le résultat
NOMS = {"N": "Effort normal", "My": "Flexion autour de y", "Mz": "Flexion autour de z", "NM": "Interaction N + M", "Vz": "Cisaillement selon z",
        "Vy": "Cisaillement selon y", "flambement": "Flambement"}
GRAVES = {"Vy_Rd_absent", "traction_retranchee", "aucun_ne_tient", "aucun_profil", "fy_epaisseur", "classe_4"}


def _bas(t: str) -> str:
    """La minuscule initiale seulement : « Interaction N + M » → « interaction N + M »."""
    return t[:1].lower() + t[1:]


def _lignes_affichees(r: Dict) -> List[Dict]:
    """Les vérifications du profilé retenu que le résultat montre : M_y et V_z toujours, les autres quand leur effort est
    saisi ; l'interaction N + M seulement quand elle diffère de la flexion seule (N ou M_z saisi)."""
    el = r["element"]
    v = {x["cle"]: x for x in r["verifications"]}
    out = []
    for cle in ("N", "My", "Mz", "NM", "Vz", "Vy", "flambement"):
        if cle not in v:
            continue
        if cle in ("N", "Mz", "Vy") and not v[cle]["actif"]:
            continue
        if cle == "NM" and not (el["avec_N"] or el["avec_Mz"]):
            continue
        out.append(v[cle])
    return out


def _gov_cle(r: Dict) -> str:
    """La ligne du résultat qui gouverne : la clé du moteur, la flexion seule quand l'interaction n'est pas montrée."""
    g = r["gouvernant"]["cle"]
    if g == "NM" and not (r["element"]["avec_N"] or r["element"]["avec_Mz"]):
        return "My"
    return g


def _axe_gouvernant(r: Dict) -> Optional[Dict]:
    axes = r["profil"]["axes"]
    return min(axes, key=lambda a: a["Nb_Rd_N"]) if axes else None


def _piece(v: Dict, r: Dict) -> str:
    k = v["cle"]
    if k == "N":
        return f"N<sub>Ed</sub> = {_n(v['valeur'])}{NB}kN"
    if k == "My":
        return f"M<sub>y,Ed</sub> = {_n(v['valeur'])}{NB}kN·m"
    if k == "Mz":
        return f"M<sub>z,Ed</sub> = {_n(v['valeur'])}{NB}kN·m"
    if k == "Vz":
        return f"V<sub>z,Ed</sub> = {_n(v['valeur'])}{NB}kN"
    if k == "Vy":
        return "A<sub>v,y</sub> absent · non vérifié" if v["resistance_absente"] else f"V<sub>y,Ed</sub> = {_n(v['valeur'])}{NB}kN"
    if k == "NM":
        return "η<sub>N</sub> + η<sub>My</sub> + η<sub>Mz</sub>"
    a = _axe_gouvernant(r)
    return f"χ<sub>{a['axe']}</sub> = {_t(a['chi'])} · axe {a['axe']}-{a['axe']}"


def _comparaison(v: Dict, r: Dict) -> Dict[str, str]:
    """Pour une vérification : la formule, la valeur et la limite écrites (tableau de la fenêtre de calcul, verdict)."""
    k = v["cle"]
    signe = "≤" if v["taux"] <= 1 else "&gt;"
    if k in ("N", "My", "Mz", "Vz", "Vy"):
        s_ed = {"N": "N<sub>Ed</sub>", "My": "|M<sub>y,Ed</sub>|", "Mz": "|M<sub>z,Ed</sub>|", "Vz": "|V<sub>z,Ed</sub>|", "Vy": "|V<sub>y,Ed</sub>|"}[k]
        s_rd = {"N": "N<sub>Rd</sub>", "My": "M<sub>y,Rd</sub>", "Mz": "M<sub>z,Rd</sub>", "Vz": "V<sub>z,Rd</sub>", "Vy": "V<sub>y,Rd</sub>"}[k]
        u = v["unite"]
        val = f"{s_ed} = {_n(v['valeur'])}{NB}{u}"
        lim = f"{s_rd} = {_r(v['limite'])}{NB}{u}" if v["limite"] is not None else f"{s_rd} : donnée absente"
        return {"formule": f"η = {s_ed} / {s_rd}", "valeur": val, "limite": lim, "verdict": f"{val} {signe} {lim}" if v["limite"] is not None else lim}
    if k == "NM":
        rt = r["profil"]["ratios"]
        val = f"{_t(rt['rN'])} + {_t(rt['rMy'])} + {_t(rt['rMz'])} = {_t(v['valeur'])}"
        return {"formule": "η<sub>N</sub> + η<sub>My</sub> + η<sub>Mz</sub> ≤ 1", "valeur": _t(v["valeur"]), "limite": "1",
                "verdict": f"η<sub>N</sub> + η<sub>My</sub> + η<sub>Mz</sub> = {val} {signe} 1"}
    val = f"N<sub>Ed</sub> = {_n(v['valeur'])}{NB}kN"
    lim = f"N<sub>b,Rd</sub> = {_r(v['limite'])}{NB}kN"
    return {"formule": "η<sub>b</sub> = N<sub>Ed</sub> / min(N<sub>b,Rd,y</sub> ; N<sub>b,Rd,z</sub>)", "valeur": val, "limite": lim,
            "verdict": f"{val} {signe} {lim}"}


def _cmp_ligne(v: Dict, r: Dict) -> str:
    """Ce qu'une ligne du résultat compare, en court (HTML) : « |M_y,Ed| 60 ≤ M_y,Rd 100,0 kN·m »."""
    k = v["cle"]
    signe = "≤" if v["taux"] <= 1 else "&gt;"
    if k in ("N", "My", "Mz", "Vz", "Vy"):
        if k == "Vy" and v["resistance_absente"]:
            return "A<sub>v,y</sub> absent : V<sub>y,Ed</sub> n'est pas vérifié"
        s_ed = {"N": "N<sub>Ed</sub>", "My": "|M<sub>y,Ed</sub>|", "Mz": "|M<sub>z,Ed</sub>|", "Vz": "|V<sub>z,Ed</sub>|", "Vy": "|V<sub>y,Ed</sub>|"}[k]
        s_rd = {"N": "N<sub>Rd</sub>", "My": "M<sub>y,Rd</sub>", "Mz": "M<sub>z,Rd</sub>", "Vz": "V<sub>z,Rd</sub>", "Vy": "V<sub>y,Rd</sub>"}[k]
        if v["limite"] is None:
            return f"{s_rd} : donnée absente"
        return f"{s_ed} {_n(v['valeur'])} {signe} {s_rd} {_r(v['limite'])}{NB}{v['unite']}"
    if k == "NM":
        return f"η<sub>N</sub> + η<sub>My</sub> + η<sub>Mz</sub> = {_t(v['valeur'])} {signe} 1"
    a = _axe_gouvernant(r)
    return f"N<sub>Ed</sub> {_n(v['valeur'])} {signe} N<sub>b,Rd</sub> {_r(v['limite'])}{NB}kN · axe {a['axe']}-{a['axe']}"


def _niv_ligne(v: Dict) -> str:
    return "c" if v["resistance_absente"] and v["cle"] == "Vy" else C.niveau(v["taux"])


def _taux_ligne(v: Dict):
    return None if v["resistance_absente"] and v["cle"] == "Vy" else C.pct(v["taux"])


# --- les formules développées (« symbole = expression = valeurs = résultat »), à partir des termes du moteur
def _formule(f: Dict, r: Dict) -> Optional[Dict]:
    t = {x["symbole"]: x["valeur"] for x in f["termes"]}
    k = f["cle"]
    res = f["resultat"]
    eta = {"rN": ("η_N", "N_Ed / N_Rd", "N"), "rMy": ("η_My", "|M_y,Ed| / M_y,Rd", "My"), "rMz": ("η_Mz", "|M_z,Ed| / M_z,Rd", "Mz"),
           "rVz": ("η_Vz", "|V_z,Ed| / V_z,Rd", "Vz"), "rVy": ("η_Vy", "|V_y,Ed| / V_y,Rd", "Vy")}
    if k in ("N_Rd", "My_Rd", "Mz_Rd", "Vz_Rd", "Vy_Rd"):
        prop, puiss, sqrt3, u = {"N_Rd": ("A", "10²", False, "kN"), "My_Rd": ("Wel_y", "10³", False, "kN·m"), "Mz_Rd": ("Wel_z", "10³", False, "kN·m"),
                                 "Vz_Rd": ("Avz", "10²", True, "kN"), "Vy_Rd": ("Avy", "10²", True, "kN")}[k]
        nom_prop = {"A": "A", "Wel_y": "W_el,y", "Wel_z": "W_el,z", "Avz": "A_v,z", "Avy": "A_v,y"}[prop]
        sym = {"N_Rd": "N_Rd", "My_Rd": "M_y,Rd", "Mz_Rd": "M_z,Rd", "Vz_Rd": "V_z,Rd", "Vy_Rd": "V_y,Rd"}[k]
        den = f"(√3 × {C.nb(t['gamma_M0'])})" if sqrt3 else C.nb(t["gamma_M0"])
        expr = f"{nom_prop}·f_y / (√3·γ_M0)" if sqrt3 else f"{nom_prop}·f_y / γ_M0"
        # les puissances de 10 écrites : 10² ou 10³ ramènent en mm, 10⁻³ ou 10⁻⁶ rendent des kN ou des kN·m (le N·mm du calcul)
        vers = " × 10⁻⁶" if u == "kN·m" else " × 10⁻³"
        return {"parts": [sym, expr, f"{C.fr(t[prop], 2)} × {puiss} × {t['fy']} / {den}{vers}"], "res": f"{_r(res)} {u}", "niveau": ""}
    if k in eta:
        s, expr, e = eta[k]
        if not f["termes"]:
            absent = {"rN": "N_Rd", "rMy": "M_y,Rd", "rMz": "M_z,Rd", "rVz": "V_z,Rd", "rVy": "V_y,Rd"}[k]
            return {"parts": [s, expr, f"{absent} absent : le module prend 0"], "res": _t(res), "niveau": "c"}
        (_, ed), (_, rd) = [(x["symbole"], x["valeur"]) for x in f["termes"]]
        return {"parts": [s, expr, f"{_n(ed)} / {_r(rd)}"], "res": _t(res), "niveau": C.niveau(res)}
    if k == "util_NM":
        return {"parts": ["η_NM", "η_N + η_My + η_Mz", f"{_t(t['rN'])} + {_t(t['rMy'])} + {_t(t['rMz'])}"], "res": _t(res), "niveau": C.niveau(res)}
    if k == "util":
        return {"parts": ["η_section", "max(η_NM ; η_Vy ; η_Vz)", f"max({_t(t['util_NM'])} ; {_t(t['rVy'])} ; {_t(t['rVz'])})"],
                "res": _t(res), "niveau": C.niveau(res)}
    m = re.fullmatch(r"(Ncr|lambda|phi|chi|Nb_Rd)_([yz])", k)
    if m:
        quoi, a = m.groups()
        if quoi == "Ncr":
            return {"parts": [f"N_cr,{a}", f"π²·E·I_{a} / L_cr,{a}²",
                              f"π² × {_n(t['E'], 0)} × {_n(t[f'I_{a}'])} × 10⁴ / ({_n(t[f'Lcr_{a}'])} × 10³)² × 10⁻³"], "res": f"{_r(res)} kN", "niveau": ""}
        if quoi == "lambda":
            return {"parts": [f"λ̄_{a}", f"√(A·f_y / N_cr,{a})", f"√({C.fr(t['A'], 2)} × 10² × {t['fy']} / ({_r(t[f'Ncr_{a}'])} × 10³))"],
                    "res": _t(res), "niveau": ""}
        if quoi == "phi":
            return {"parts": [f"φ_{a}", f"0,5·(1 + α·(λ̄_{a} − 0,2) + λ̄_{a}²)",
                              f"0,5 × (1 + {C.nb(t['alpha'])} × ({_t(t[f'lambda_{a}'])} − 0,2) + {_t(t[f'lambda_{a}'])}²)"], "res": _t(res), "niveau": ""}
        if quoi == "chi":
            return {"parts": [f"χ_{a}", f"1 / (φ_{a} + √(φ_{a}² − λ̄_{a}²)) ≤ 1",
                              f"1 / ({_t(t[f'phi_{a}'])} + √({_t(t[f'phi_{a}'])}² − {_t(t[f'lambda_{a}'])}²))"], "res": _t(res), "niveau": ""}
        return {"parts": [f"N_b,Rd,{a}", f"χ_{a}·A·f_y / γ_M1",
                          f"{_t(t[f'chi_{a}'])} × {C.fr(t['A'], 2)} × 10² × {t['fy']} / {C.nb(t['gamma_M1'])} × 10⁻³"], "res": f"{_r(res)} kN", "niveau": ""}
    if k == "util_buck":
        return {"parts": ["η_b", "N_Ed / min(N_b,Rd,y ; N_b,Rd,z)", f"{_n(t['N_Ed'])} / {_r(t['Nb_Rd'])}"], "res": _t(res), "niveau": C.niveau(res)}
    if k == "util_global":
        if t.get("util_buck") is None:
            return {"parts": ["η", "η_section (sans flambement calculé)", _t(t["util"])], "res": _t(res), "niveau": C.niveau(res)}
        return {"parts": ["η", "max(η_section ; η_b)", f"max({_t(t['util'])} ; {_t(t['util_buck'])})"], "res": _t(res), "niveau": C.niveau(res)}
    return None                                                      # pragma: no cover — une formule nouvelle du moteur


def _groupes_formules(r: Dict) -> List[tuple]:
    """Les formules du profilé retenu, rangées comme une note : résistances, taux de section, flambement par axe, taux."""
    par = {f["cle"]: f for f in r["profil"]["formules"]}
    el = r["element"]
    res = [k for k in ("N_Rd", "My_Rd", "Mz_Rd", "Vz_Rd", "Vy_Rd") if k in par and (k not in ("Mz_Rd", "Vy_Rd") or el[{"Mz_Rd": "avec_Mz", "Vy_Rd": "avec_Vy"}[k]])]
    taux = ["rN"] * el["avec_N"] + ["rMy"] + ["rMz"] * el["avec_Mz"] + ["rVz"] + ["rVy"] * el["avec_Vy"] + ["util_NM", "util"]
    g = [("Résistances de section (élastiques)", res), ("Taux de la section", taux)]
    for a in r["profil"]["axes"]:
        x = a["axe"]
        g.append((f"Flambement selon {x}-{x}", [f"Ncr_{x}", f"lambda_{x}", f"phi_{x}", f"chi_{x}", f"Nb_Rd_{x}"]))
    g.append(("Taux du profilé", (["util_buck"] if "util_buck" in par else []) + ["util_global"]))
    return [(titre, [par[k] for k in cles if k in par]) for titre, cles in g]


def _fenetre(r: Dict) -> str:
    el = r["element"]
    p = r["profil"]
    gov = _gov_cle(r)
    lignes = []
    for v in _lignes_affichees(r):
        c = _comparaison(v, r)
        lignes.append({"nom": C.esc(NOMS[v["cle"]]), "sous": _piece(v, r), "formule": c["formule"], "valeur": c["valeur"], "limite": c["limite"],
                       "taux": _taux_ligne(v) or 0, "niveau": _niv_ligne(v), "gov": v["cle"] == gov})
    detail = ""
    for titre, fs in _groupes_formules(r):
        fx = [x for x in (_formule(f, r) for f in fs) if x]
        if fx:
            detail += f'<div class="ck-g"><div class="ck-t">{C.esc(titre)}</div>{_grec(C.lignes_fx(fx))}</div>'
    hyps = [f"Acier <b>{C.esc(el['acier'])}</b> · f<sub>y</sub> = {r['acier']['fy']} N/mm²",
            f"γ<sub>M0</sub> = {C.nb(el['gamma_M0'])} · γ<sub>M1</sub> = {C.nb(el['gamma_M1'])}",
            "Section élastique (W<sub>el</sub>) · interaction linéaire N + M"]
    if r["flambement"]["calcule"]:
        hyps.append(f"E = {_n(CA.E_ACIER, 0)} N/mm² · courbe <b>{C.esc(el['courbe'])}</b> (α = {C.nb(r['flambement']['alpha'])})")
    hyps.append(f"<b>{C.esc(p['nom'])}</b> · {_n(p['masse'], 2)} kg/m")
    pied = ('<div class="ck-autres">Unités de l\'écran : aires en cm², modules en cm³, inerties en cm⁴, longueurs en m, f<sub>y</sub> '
            'et E en N/mm² ; 10², 10³ et 10⁴ ramènent en mm, 10⁻³ et 10⁻⁶ rendent les N et N·mm du calcul en kN et kN·m.</div>')
    return C.fenetre_calcul("s0", f"{C.esc(el.get('nom') or 'Élément')} · {C.esc(p['nom'])}", _efforts_txt(r) + " · EN 1993-1-1, section élastique",
                            hyps, lignes, detail, pied)


def _verdict(r: Dict) -> str:
    if not r["profil"]:
        return (f'<div class="verdict"><div class="gov"><b>Aucun profilé</b>aucun profilé des familles cochées ne passe le filtre '
                f'I<sub>y</sub> ≥ {_n(r["Iy_min"])}{NB}cm⁴</div></div>')
    v = {x["cle"]: x for x in r["verifications"]}
    g = _gov_cle(r)
    c = _comparaison(v[g], r)
    t = r["taux_max"]
    # un taux démesuré (un profilé choisi à la main bien trop faible) s'écrit en entier : la puce se resserre pour le tenir
    chiffres = len(C.pc(C.pct(t)))
    long_ = " pf-long" if chiffres >= 7 else (" pf-moyen" if chiffres >= 5 else "")
    return (f'<div class="verdict{long_}" data-calcul="s0" title="Voir le calcul">{C.chip(C.pct(t), C.niveau(t), xl=True)}<div class="gov">'
            f'<b>{C.esc(r["retenu"])} · {C.esc(_bas(NOMS[g]))}</b>'
            f'{c["verdict"]}</div></div>')


def _ligne_retenu(r: Dict) -> str:
    """La ligne « Profilé retenu » : le bouton qui ouvre le sélecteur (une étoile quand un plus léger tient), le taux du
    profilé, puis sa masse et le plus léger qui tient."""
    t = r["taux_max"]
    pl = r["plus_leger"]
    par = {li["nom"]: li for li in r["lignes"]}
    leger = ""
    if not pl:
        quoi = "aucun profilé comparé ne tient"
    elif pl == r["retenu"]:
        quoi = "le plus léger qui tient"
    else:
        quoi = f"plus léger : {C.esc(pl)}, {_n(par[pl]['masse'], 1)}{NB}kg/m"
        leger = f"Le plus léger qui tient : {pl} · {C.pc(C.pct(par[pl]['taux']))} %"
    return C.ligne_verif("Profilé retenu", f"{_n(r['profil']['masse'], 1)}{NB}kg/m · {quoi}", C.pct(t), C.niveau(t), calcul="s0", pk=ROLE,
                         piece=C.esc(r["retenu"]), leger=leger)


def _etat_bloc(t: float) -> str:
    """L'état du bloc (du texte) : « Conforme · 63 % » ; un taux démesuré s'écrit « > 999 % », comme dans les lignes."""
    p = C.pct(t)
    return f"{C.STATUTS[C.niveau(t)][1]} · " + (f">{NB}{C.PLAFOND_TAUX}{NB}%" if p > C.PLAFOND_TAUX else f"{C.pc(p)}{NB}%")


def _messages(r: Dict) -> str:
    out = ""
    for a in r["avertissements"]:
        out += C.message(_sub(a["message"]), a["code"] in GRAVES)
    if r["profil"]:
        li = next(x for x in r["lignes"] if x["nom"] == r["retenu"])
        if li["tient"] and li["taux"] > 1:
            out += C.message(f"L'écran Streamlit arrondit l'utilisation de {C.esc(li['nom'])} à {_n(li['utilisation'])} % et le compte comme "
                             f"tenant ; son taux exact dépasse 1 (η = {C.fr(li['taux'], 6)}).", True)
    return out


def _resultat_html(r: Dict, nom: str) -> str:
    """Le cadre des résultats : la tête (le nom de l'élément ; le profilé retenu, l'acier, les efforts), le verdict, le
    bloc « Vérifications » — le profilé retenu d'abord, puis chaque vérification (deux lignes : ce qu'elle compare, sa
    jauge) —, les avertissements du moteur, la masse."""
    quoi = f"{C.esc(r['retenu'])} · " if r["profil"] else ""
    tete = E.tete_resultats(C.esc(nom), f"{quoi}{C.esc(r['element']['acier'])} · {_efforts_txt(r)}")
    if not r["profil"]:
        return (f'<div class="pbody">{tete}{_verdict(r)}{_messages(r)}</div>'
                f'<div class="pfoot"><div class="legend">{C.legende()}</div></div>')
    gov = _gov_cle(r)
    t = r["taux_max"]
    rows = _ligne_retenu(r)
    for v in _lignes_affichees(r):
        # η_N d'une traction est NÉGATIF (défaut D2 de l'écran, écrit tel quel) : sa jauge reste vide (commun._largeur)
        absent = v["cle"] == "Vy" and v["resistance_absente"]
        rows += C.ligne_verif(NOMS[v["cle"]], _cmp_ligne(v, r), _taux_ligne(v), _niv_ligne(v), calcul="s0", gov=v["cle"] == gov,
                              etat_txt="non vérifié" if absent else "")
    bloc = E.bloc(0, "Vérifications", CLASSES_BLOC.get(C.niveau(t), "inc"), _etat_bloc(t), C.esc(_selection_txt(r)), rows, theme=True)
    pied = (f'<div class="pfoot"><div class="mass"><span>Masse du profilé retenu</span><b>{_n(r["profil"]["masse"], 1)}{NB}kg/m</b></div>'
            f'<div class="legend">{C.legende()}</div></div>')
    return f'<div class="pbody">{tete}{_verdict(r)}<div class="et-blocs">{bloc}</div>{_messages(r)}</div>{pied}'


# ------------------------------------------------------------------ les vues
def _vue_section(r: Dict) -> str:
    p = r["profil"]
    rec = CAT.par_nom()[p["nom"]]
    fam = rec["famille"]
    dims = {k: rec[k] for k in FA.FAMILLES[fam]["dims"]}
    syms = {"h [mm]": ("h", "mm"), "b [mm]": ("b", "mm"), "tw [mm]": ("t_w", "mm"), "tf [mm]": ("t_f", "mm"), "r [mm]": ("r", "mm"),
            "Poids [kg/m]": ("masse", "kg/m"), "A [cm²]": ("A", "cm²"), "Wel_y [cm³]": ("W_el,y", "cm³"), "Wel_z [cm³]": ("W_el,z", "cm³"),
            "Avy [cm²]": ("A_v,y", "cm²"), "Avz [cm²]": ("A_v,z", "cm²"), "Iy [cm⁴]": ("I_y", "cm⁴"), "Iz [cm⁴]": ("I_z", "cm⁴")}
    dimensions = [(syms[a][0], _n(v), syms[a][1]) for a, v in p["dimensions"]]
    if fam == "IPN":
        dimensions[-1] = ("r_1", _n(rec["r1"]), "mm")
        dimensions.append(("r_2", _n(rec["r2"]), "mm"))
    proprietes = [(syms[a][0], _n(v), syms[a][1]) for a, v in p["proprietes"]]
    sous = f"{FA.FAMILLES[fam]['libelle']} · {FA.FAMILLES[fam]['norme']}"
    return DP.section("pf_section", p["nom"], sous, fam, dims, dimensions, proprietes, _efforts_txt(r, html=False))


def _vue_tableau(r: Dict) -> str:
    """Le tableau de l'écran Streamlit (« Afficher tous les profilés ✓/✗ »), trié comme lui, ses lignes colorées comme
    les siennes : l'optimal de l'écran, ceux qui tiennent, ceux qui ne tiennent pas ; le retenu ✓ et le plus léger ★."""
    flamb = r["flambement"]["calcule"]
    # les colonnes de la DÉCISION d'abord (utilisation, masse, taux), puis les propriétés du catalogue : sur un écran étroit,
    # c'est la fin du tableau qui défile. A_v,y : l'écran le montre, toujours vide (il ne le lit pas, défaut D1) ; le
    # site l'omet et le dit sous le tableau
    cols = [("Utilisation", "%"), ("Masse", "kg/m"), ("η<sub>section</sub>", "%")] + ([("η<sub>flamb</sub>", "%")] if flamb else []) \
        + [("h", "mm"), ("W<sub>el,y</sub>", "cm³"), ("W<sub>el,z</sub>", "cm³"), ("A<sub>v,z</sub>", "cm²"), ("A", "cm²"), ("I<sub>y</sub>", "cm⁴"),
           ("I<sub>z</sub>", "cm⁴")]
    tete = '<th class="mk"></th><th class="nm">Profilé</th>' + "".join(f"<th>{a}<small>{u}</small></th>" for a, u in cols)
    corps = ""
    # triés par MASSE (l'écran les trie par utilisation : les plus gros d'abord, loin de la décision) ; à masse égale,
    # l'ordre de l'écran
    for li in sorted(r["lignes"], key=lambda x: (x["masse"] is None, x["masse"] or 0.0)):
        cls = "opt" if li["nom"] == r["optimal"] else ("ok" if li["tient"] else "nok")
        cls += " cur" if li["nom"] == r["retenu"] else ""
        mk = "✓" if li["nom"] == r["retenu"] else ('<span class="pf-or">★</span>' if li["nom"] == r["plus_leger"] else "")
        vals = [li["utilisation"], li["masse"], li["u_section"]] + ([li["u_flamb"]] if flamb else []) \
            + [li["h"], li["Wel_y"], li["Wel_z"], li["Avz"], li["A"], li["Iy"], li["Iz"]]
        corps += (f'<tr class="{cls}"><td class="mk">{mk}</td><th class="nm">{C.esc(li["nom"])}</th>'
                  + "".join(f"<td>{_n(v)}</td>" for v in vals) + "</tr>")
    leg = ('<span class="lg opt"><i></i>optimal de l\'écran (le plus chargé qui tient)</span><span class="lg ok"><i></i>tient (≤ 100 %)</span>'
           '<span class="lg nok"><i></i>ne tient pas</span><span class="lg">✓ retenu</span><span class="lg"><b class="or">★</b> le plus léger qui tient</span>')
    pied = ("Le tableau de l'écran Streamlit (mêmes colonnes, même utilisation arrondie à 10⁻³ %), trié ici par masse croissante ; "
            "les nombres sont ceux du catalogue (acier/donnees/profiles.json). La colonne A<sub>v,y</sub> de l'écran, toujours vide (il ne lit "
            "pas cette aire), est omise.")
    return (f'<div class="pf-vue"><div class="pf-leg">{leg}</div><div class="pf-defile"><table class="pf-tab"><thead><tr>{tete}</tr></thead>'
            f'<tbody>{corps}</tbody></table></div><div class="pf-pied">{pied}</div></div>')


def _vue_courbe(r: Dict) -> str:
    pts = []
    for li in r["lignes"]:
        aide = f"{li['nom']} · {_n(li['masse'], 1)} kg/m · η = {C.pc(C.pct(li['taux'], 1))} %"
        pts.append({"nom": li["nom"], "famille": li["famille"], "masse": li["masse"], "pct": li["taux"] * 100, "niveau": C.niveau(li["taux"]), "aide": aide})
    return DP.courbe("pf_courbe", pts, r["retenu"], r["plus_leger"])


def _vue_flambement(r: Dict) -> str:
    axes = [{"axe": a["axe"], "lambdabar": a["lambdabar"], "chi": a["chi"],
             "texte": f"{a['axe']}-{a['axe']} · λ̄ = {_t(a['lambdabar'])} · χ = {_t(a['chi'])}"} for a in r["profil"]["axes"]]
    return DP.flambement("pf_flambement", r["element"]["courbe"], [(k, a) for k, a, _ in EN.COURBES], axes)


def _vue_vide(r: Dict) -> str:
    """La toile quand aucun profilé ne passe les filtres : ce qui les ferme, et comment les rouvrir."""
    fams = ", ".join(r["familles"]) if r["element"]["familles"] else "toutes les familles"
    filtre = (f" avec I<sub>y</sub> ≥ {_n(r['Iy_min'])}{NB}cm⁴" if r["Iy_min"] > 0 else "")
    return (f'<div class="pf-vue"><div class="pf-vide"><b>Aucun profilé à comparer</b>'
            f'<span>Familles : {C.esc(fams)}{filtre}. Abaisser I<sub>y,min</sub> ou cocher d\'autres familles '
            f'(cadre Données, section 3).</span></div></div>')


def _bande(r: Dict) -> List[tuple]:
    """Les efforts saisis, pour la bande de l'en-tête des vues : (symbole, indice, valeur, unité) — M_y et V_z toujours,
    les autres quand leur case est cochée (l'ordre de l'écran)."""
    el = r["element"]
    out = [("M", "y,Ed", el["My"], "kN·m"), ("V", "z,Ed", el["Vz"], "kN")]
    for case, s, x, k, u in (("avec_N", "N", "Ed", "N", "kN"), ("avec_Mz", "M", "z,Ed", "Mz", "kN·m"), ("avec_Vy", "V", "y,Ed", "Vy", "kN"),
                             ("avec_T", "T", "Ed", "T", "kN·m")):
        if el[case]:
            out.append((s, x, el[k], u))
    return out


def _entete(r: Dict, titre: str, legende: str) -> str:
    """L'en-tête technique d'une vue : le profilé retenu et l'acier ; la sélection (ce qui est comparé, ce qui tient) ; à
    droite, les efforts saisis."""
    sous = (f"{C.esc(r['retenu'])} · " if r["profil"] else "") + C.esc(r["element"]["acier"])
    return E.entete(titre.upper(), sous, [_selection_txt(r)], _bande(r), legende)


def _vues(r: Dict) -> List[Dict]:
    if not r["profil"]:
        return [{"cle": "tableau", "titre": "Tableau des profilés", "court": "Tableau", "svg": _vue_vide(r), "legende": "aucun profilé",
                 "entete": _entete(r, "Tableau des profilés", "")}]
    p = r["profil"]
    vues = [{"cle": "section", "titre": f"Section {p['nom']}", "court": "Section", "svg": _vue_section(r),
             "legende": f"{p['nom']} · à l'échelle", "entete": _entete(r, "Section", "le profilé retenu, à l'échelle ; ses dimensions et ses propriétés")},
            {"cle": "tableau", "titre": "Tableau des profilés", "court": "Tableau", "svg": _vue_tableau(r),
             "legende": f"{r['n_profils']} profilés · {r['n_tiennent']} tiennent",
             "entete": _entete(r, "Tableau des profilés", "triés par masse ; ✓ le retenu, ★ le plus léger qui tient")},
            {"cle": "courbe", "titre": "Taux et masse", "court": "Taux / masse", "svg": _vue_courbe(r),
             "legende": "le taux de chaque profilé de la sélection en fonction de sa masse",
             "entete": _entete(r, "Taux et masse", "le taux de chaque profilé de la sélection en fonction de sa masse")}]
    if r["flambement"]["calcule"] and p["axes"]:
        vues.append({"cle": "flambement", "titre": "Flambement", "court": "Flambement", "svg": _vue_flambement(r),
                     "legende": f"courbes χ(λ̄) · courbe {r['element']['courbe']} retenue",
                     "entete": _entete(r, "Flambement", f"les courbes de flambement ; la courbe {r['element']['courbe']} est retenue, un point par axe")})
    return vues


# ------------------------------------------------------------------ la note
def _calculer(e: Dict) -> Dict:
    try:
        return MO.verifier(e)
    except ValueError as ex:                                       # pragma: no cover — normaliser l'a déjà refusé
        return {"erreur": str(ex), "taux_max": None}


def resume_note(etat: Dict, resultats: List[Dict]) -> Dict:
    """Le résumé de la NOTE : son élément le plus chargé."""
    vals = [(r["taux_max"], i) for i, r in enumerate(resultats) if r.get("taux_max") is not None]
    if not vals:
        return {"taux": None, "niveau": "x", "statut": "Calcul impossible", "gouvernant": ""}
    t, i = max(vals)
    r = resultats[i]
    niv = C.niveau(t)
    return {"taux": C.pct(t), "niveau": niv, "statut": C.STATUTS[niv][1],
            "gouvernant": f"{etat['elements'][i].get('nom') or 'Élément'} · {r['retenu']} · {_bas(NOMS[_gov_cle(r)])}"}


def elements(etat: Dict, resultats: List[Dict]) -> str:
    """Rien : les éléments de la note sont la LIGNE de l'élément, en tête de la colonne de gauche (``arbre``)."""
    return ""


def resultat(etat: Dict) -> Dict:
    el = _el(etat)
    r = MO.verifier(el)
    resultats = [r if i == etat["actif"] else _calculer(e) for i, e in enumerate(etat["elements"])]
    # le statut de la NOTE (dès deux éléments) : le moins bon de ses éléments ; seul, le verdict le dit déjà
    nivs = [C.niveau(x["taux_max"]) if x.get("taux_max") is not None else "x" for x in resultats]
    statut = C.statut(next((n for n in ("x", "r", "f") if n in nivs), "p")) if len(resultats) > 1 else ""
    return {"resultat": _resultat_html(r, el.get("nom") or "Élément"), "volets": _fenetre(r) if r["profil"] else "", "vues": _vues(r),
            "phrase": "", "statut_html": statut, "resume": resume_note(etat, resultats), "donnees": arbre(etat, r, resultats),
            "elements": elements(etat, resultats)}


# ------------------------------------------------------------------ le sélecteur de profilés
def selecteur(etat: Dict, role: str, familles=None, froid: bool = True) -> str:
    """Les profilés des familles cochées (celles de l'élément par défaut ; toutes s'il n'en a aucune), filtre I_y min
    compris, TRIÉS PAR MASSE, chacun avec son taux calculé par le moteur ; le retenu (✓) et le plus léger qui tient (★)
    marqués ; sous la liste, les propriétés de la ligne survolée."""
    if role != ROLE:
        raise KeyError(f"sélecteur inconnu : {role!r}")
    el = _el(etat)
    dispo = EN.familles_disponibles()
    demandees = familles if isinstance(familles, (list, tuple)) else []      # une requête abîmée : celles de l'élément
    fams = [f for f in dispo if f in demandees] or el["familles"] or dispo
    cands = MO.candidats(el, fams)
    i_o = next((k for k, c in enumerate(cands) if c["plus_leger"]), 0)
    i_c = next((k for k, c in enumerate(cands) if c["retenu"]), i_o)

    def prop(c):
        p = CA.profil(c["nom"])
        return (f"<b>{C.esc(c['nom'])}</b> · h × b = {_n(p['h'])} × {_n(p['b'])} mm · A = {_n(p['A'], 2)} cm² · "
                f"I<sub>y</sub> = {_n(p['Iy'], 1)} cm⁴ · W<sub>el,y</sub> = {_n(p['Wel_y'], 1)} cm³ · A<sub>v,z</sub> = {_n(p['Avz'], 2)} cm² · "
                f"η = {C.fr(c['taux'], 3)}")
    rows = [{"nom": c["nom"], "col": C.fr(c["masse"], 1), "taux": C.pct(min(c["taux"], 9.99)), "niveau": C.niveau(c["taux"]),
             "cur": c["retenu"], "opt": c["plus_leger"], "prop": prop(c)} for c in cands]
    filtres = "".join(f'<button type="button" class="fchip{" on" if f in fams else ""}" data-famille="{C.esc(f)}">{C.esc(f)}</button>' for f in dispo)
    sous = "triés par masse · un clic retient le profilé"
    if el["Iy_min"] > 0:
        sous += f" · I<sub>y</sub> ≥ {_n(el['Iy_min'])} cm⁴"
    return C.selecteur(ROLE, f"{C.esc(el.get('nom') or 'Élément')} · profilé", sous, filtres, rows, max(0, min(i_o, i_c) - 4), "kg/m")
